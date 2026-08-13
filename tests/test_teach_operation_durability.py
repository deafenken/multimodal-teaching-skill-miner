from __future__ import annotations

import base64
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from teaching_skill_miner.harness import (
    HarnessEventJournal,
    TeachOperationCheckpoint,
)
from teaching_skill_miner.harness.events import HarnessEventEmitter
from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardSnapshot,
    _teach_inner_journal_path,
    build_teacher_agent_dashboard_snapshot,
)
from teaching_skill_miner.teacher_agent_live import LiveAgentOptions


class _LiveClient:
    def __init__(self, *, crash: bool = False) -> None:
        self.chat_json_call_count = 0
        self.crash = crash

    def public_status(self) -> dict[str, object]:
        return {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_origin": "https://api.deepseek.com",
            "configured": True,
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "remote_student_data_opt_in": True,
            "api_key_exposed": False,
        }

    def chat_json(
        self,
        messages: object,
        *,
        request_kind: str,
        require_remote_consent: bool = True,
    ) -> tuple[dict[str, object], dict[str, object]]:
        del messages, require_remote_consent
        self.chat_json_call_count += 1
        if self.crash:
            raise KeyboardInterrupt("injected process death during provider effect")
        return (
            {
                "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
                "diagnosis": {
                    "signal": "not_observed",
                    "confidence": 0.0,
                    "diagnosis_reason": "等待学生作答",
                    "evidence_excerpt": "",
                    "misconception_tag": None,
                    "misconception_description": "",
                    "resolved_misconception_tags": [],
                    "response_quality": "empty",
                    "engagement_level": "unknown",
                    "needs_human_review": False,
                },
                "decision": {
                    "primary_skill_id": "skill_diagnostic_questioning",
                    "supporting_skill_ids": ["skill_wait_and_elicit"],
                    "selection_reason": "先确认前置知识。",
                    "next_focus": "prerequisite",
                },
                "teacher_action": {
                    "type": "ask_one_question",
                    "message": "请先说出一个完成当前目标所需的前置概念。",
                    "expected_signal": "学生给出一个相关前置概念。",
                },
                "stop_recommendation": {"should_stop": False, "reason": ""},
            },
            {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "request_kind": request_kind,
                "request_sha256": "a" * 64,
                "latency_ms": 1.0,
                "attempt_count": 1,
                "http_status": 200,
                "response_id": "fake",
                "usage": {},
                "credential_logged": False,
            },
        )


class TeachOperationDurabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = project_root()
        cls.library = root / "data/teacher_agent_skill_library.json"
        cls.demo = root / "data/teacher_agent_demo_input.json"
        cls.cases = root / "data/teacher_agent_evaluation_cases.json"

    def _snapshot(
        self,
        store: Path,
        *,
        client: _LiveClient | None = None,
        resource_index_store: Path | None = None,
    ) -> TeacherAgentDashboardSnapshot:
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.library,
            self.demo,
            self.cases,
            store_path=store,
            client=client,
            live_options=LiveAgentOptions(agent_loop_enabled=client is not None),
            resource_index_store_path=resource_index_store,
            consent_store_path=(
                store.with_name(store.name + ".consent")
                if client is not None
                else None
            ),
            consent_signing_secret=(
                b"durability-test-consent-signing-secret-material-32-bytes"
                if client is not None
                else None
            ),
        )
        if client is not None and not any(
            receipt["purpose"] == "remote_teaching"
            and receipt["status"] == "active"
            for receipt in snapshot.list_remote_consents({})["receipts"]
        ):
            snapshot.grant_remote_consent(
                {
                    "purpose": "remote_teaching",
                    "validity_days": 1,
                    "likely_minor": False,
                    "guardian_or_school_policy": "not_required",
                }
            )
        return snapshot

    @staticmethod
    def _start_body(
        snapshot: TeacherAgentDashboardSnapshot, *, suffix: str
    ) -> dict[str, object]:
        payload = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": f"durable-start-key-{suffix}",
        }
        if snapshot.client is not None:
            payload["remote_consent_id"] = next(
                receipt["consent_id"]
                for receipt in snapshot.list_remote_consents({})["receipts"]
                if receipt["purpose"] == "remote_teaching"
                and receipt["status"] == "active"
            )
        return {
            "operation": "start",
            "request_id": f"durable-start-request-{suffix}",
            "payload": payload,
        }

    def test_production_teach_operation_links_outer_and_nested_journals(self) -> None:
        with TemporaryDirectory() as directory:
            store = Path(directory) / "sessions.jsonl"
            snapshot = self._snapshot(store, client=_LiveClient())
            body = self._start_body(snapshot, suffix="linked")

            record, _ = snapshot.open_harness_stream(body)
            self.assertEqual(record.handle.wait(timeout=5)["status"], "completed")

            checkpoint = TeachOperationCheckpoint.from_value(
                record.journal.load_checkpoint() or {}
            )
            self.assertEqual(checkpoint.status, "completed")
            self.assertEqual(checkpoint.program_counter, "completed")
            self.assertEqual(checkpoint.run_id, record.handle.run_id)
            self.assertEqual(checkpoint.turn_id, record.handle.turn_id)
            self.assertEqual(checkpoint.domain_effect_state, "committed")
            inner = HarnessEventJournal(
                _teach_inner_journal_path(
                    snapshot.stream_journal_directory, record.handle.run_id
                ),
                run_id=checkpoint.inner_run_id,
                turn_id=checkpoint.inner_turn_id,
            )
            self.assertGreater(inner.last_sequence, 0)
            self.assertIsNotNone(inner.terminal_type)
            self.assertIsNotNone(inner.load_checkpoint())

    def test_restart_after_domain_commit_finishes_without_repeating_step(self) -> None:
        with TemporaryDirectory() as directory:
            store = Path(directory) / "sessions.jsonl"
            first = self._snapshot(store)
            start_body = self._start_body(first, suffix="step-recovery")
            start_record, _ = first.open_harness_stream(start_body)
            self.assertEqual(start_record.handle.wait(timeout=5)["status"], "completed")
            start_result = first.start_idempotency_cache[
                "durable-start-key-step-recovery"
            ]["response"]
            session_id = str(start_result["session_id"])
            original_round = int(start_result["rounds_completed"])
            step_body = {
                "operation": "step",
                "request_id": "durable-step-request-recovery",
                "payload": {
                    "session_id": session_id,
                    "learner_response": "我不会，请先讲一个例子。",
                    "idempotency_key": "durable-step-key-recovery",
                    "expected_round": original_round,
                    "expected_context_version": start_result["context_version"],
                    "profile_revision": start_result["profile_summary"][
                        "profile_revision"
                    ],
                    "expected_question_id": start_result["expected_question_id"],
                },
            }
            original_emit = HarnessEventEmitter.emit

            def crash_before_outer_message(
                emitter: HarnessEventEmitter,
                event_type: str,
                payload: object,
                **kwargs: object,
            ):
                emitted = original_emit(emitter, event_type, payload, **kwargs)
                if (
                    event_type == "message.delta"
                    and emitter.run_id.startswith("stream_")
                ):
                    raise KeyboardInterrupt(
                        "injected process death during outer message commit"
                    )
                return emitted

            with (
                patch.object(HarnessEventEmitter, "emit", new=crash_before_outer_message),
                patch.object(
                    TeacherAgentDashboardSnapshot,
                    "_finalize_stream_retention",
                    return_value=None,
                ),
            ):
                interrupted, _ = first.open_harness_stream(step_body)
                with self.assertRaises(KeyboardInterrupt):
                    interrupted.handle.wait(timeout=5)
            self.assertIsNone(interrupted.journal.terminal_type)

            restarted = self._snapshot(store)
            recovered, _ = restarted.open_harness_stream(step_body)
            self.assertEqual(recovered.handle.wait(timeout=5)["status"], "completed")
            streamed = "".join(
                str(event.get("payload", {}).get("delta", ""))
                for event in recovered.journal.replay()
                if event.get("type") == "message.delta"
            )
            resumed = restarted.resume({"session_id": session_id})
            self.assertEqual(resumed["rounds_completed"], original_round + 1)
            self.assertEqual(streamed, restarted._stream_teacher_message(resumed))
            committed = [
                event
                for event in restarted.store.events
                if event.get("event_type") == "turn_committed"
                and event.get("idempotency_key") == "durable-step-key-recovery"
            ]
            self.assertEqual(len(committed), 1)

    def test_restart_with_unknown_provider_effect_handoffs_without_reexecution(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = Path(directory) / "sessions.jsonl"
            crashing_client = _LiveClient(crash=True)
            first = self._snapshot(store, client=crashing_client)
            body = self._start_body(first, suffix="unknown-effect")
            with patch.object(
                TeacherAgentDashboardSnapshot,
                "_finalize_stream_retention",
                return_value=None,
            ):
                interrupted, _ = first.open_harness_stream(body)
                with self.assertRaises(KeyboardInterrupt):
                    interrupted.handle.wait(timeout=5)
            self.assertEqual(crashing_client.chat_json_call_count, 1)
            self.assertIsNone(interrupted.journal.terminal_type)

            replacement_client = _LiveClient()
            restarted = self._snapshot(store, client=replacement_client)
            recovered, _ = restarted.open_harness_stream(body)
            outcome = recovered.handle.wait(timeout=5)
            self.assertEqual(outcome["status"], "handoff")
            self.assertEqual(outcome["reason"], "unknown_domain_effect_after_restart")
            self.assertEqual(replacement_client.chat_json_call_count, 0)
            self.assertEqual(recovered.journal.terminal_type, "run.handoff")

    def test_restart_before_provider_effect_safely_reexecutes_once(self) -> None:
        with TemporaryDirectory() as directory:
            store = Path(directory) / "sessions.jsonl"
            first_client = _LiveClient()
            first = self._snapshot(store, client=first_client)
            body = self._start_body(first, suffix="safe-replay")
            original_checkpoint = (
                TeacherAgentDashboardSnapshot._write_teach_operation_checkpoint
            )

            def crash_after_context_checkpoint(*args: object, **kwargs: object):
                checkpoint = original_checkpoint(*args, **kwargs)
                if kwargs.get("program_counter") == "context_prepared":
                    raise KeyboardInterrupt(
                        "injected process death before provider effect"
                    )
                return checkpoint

            with (
                patch.object(
                    TeacherAgentDashboardSnapshot,
                    "_write_teach_operation_checkpoint",
                    new=staticmethod(crash_after_context_checkpoint),
                ),
                patch.object(
                    TeacherAgentDashboardSnapshot,
                    "_finalize_stream_retention",
                    return_value=None,
                ),
            ):
                interrupted, _ = first.open_harness_stream(body)
                with self.assertRaises(KeyboardInterrupt):
                    interrupted.handle.wait(timeout=5)
            self.assertEqual(first_client.chat_json_call_count, 0)
            checkpoint = TeachOperationCheckpoint.from_value(
                interrupted.journal.load_checkpoint() or {}
            )
            self.assertEqual(checkpoint.domain_effect_state, "not_started")

            replacement_client = _LiveClient()
            restarted = self._snapshot(store, client=replacement_client)
            recovered, _ = restarted.open_harness_stream(body)
            self.assertEqual(recovered.handle.wait(timeout=5)["status"], "completed")
            self.assertGreater(replacement_client.chat_json_call_count, 0)
            self.assertEqual(
                len(restarted.start_idempotency_cache),
                1,
            )

    def test_restart_before_provider_recovers_staged_resource_from_private_index(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "sessions.jsonl"
            resource_index_store = root / "resource-index"
            first_client = _LiveClient()
            first = self._snapshot(
                store,
                client=first_client,
                resource_index_store=resource_index_store,
            )
            private_tail = "只应留在私有检索索引的末尾锚点"
            resource_bytes = (
                "动态规划教学讲义。\n" + "状态与转移材料。" * 2_000 + private_tail
            ).encode("utf-8")
            uploaded = first.upload_resource(
                {
                    "resource_idempotency_key": "durable-resource-upload-001",
                    "mime_type": "text/plain",
                    "display_name": "恢复讲义.txt",
                    "data_base64": base64.b64encode(resource_bytes).decode("ascii"),
                }
            )
            staged_id = str(uploaded["resource"]["staged_resource_id"])
            content_hash = str(uploaded["resource"]["content_sha256"])
            body = self._start_body(first, suffix="resource-safe-replay")
            payload = body["payload"]
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            payload["staged_resource_ids"] = [staged_id]
            original_checkpoint = (
                TeacherAgentDashboardSnapshot._write_teach_operation_checkpoint
            )

            def crash_after_context_checkpoint(*args: object, **kwargs: object):
                checkpoint = original_checkpoint(*args, **kwargs)
                if kwargs.get("program_counter") == "context_prepared":
                    raise KeyboardInterrupt(
                        "injected process death before provider effect"
                    )
                return checkpoint

            with (
                patch.object(
                    TeacherAgentDashboardSnapshot,
                    "_write_teach_operation_checkpoint",
                    new=staticmethod(crash_after_context_checkpoint),
                ),
                patch.object(
                    TeacherAgentDashboardSnapshot,
                    "_finalize_stream_retention",
                    return_value=None,
                ),
            ):
                interrupted, _ = first.open_harness_stream(body)
                with self.assertRaises(KeyboardInterrupt):
                    interrupted.handle.wait(timeout=5)
            self.assertEqual(first_client.chat_json_call_count, 0)
            checkpoint = TeachOperationCheckpoint.from_value(
                interrupted.journal.load_checkpoint() or {}
            )
            self.assertEqual(checkpoint.domain_effect_state, "not_started")

            replacement_client = _LiveClient()
            restarted = self._snapshot(
                store,
                client=replacement_client,
                resource_index_store=resource_index_store,
            )
            self.assertEqual(restarted.staged_resources, {})
            recovered, _ = restarted.open_harness_stream(body)
            self.assertEqual(recovered.handle.wait(timeout=5)["status"], "completed")
            self.assertGreater(replacement_client.chat_json_call_count, 0)

            response = restarted.start_idempotency_cache[
                "durable-start-key-resource-safe-replay"
            ]["response"]
            session_id = str(response["session_id"])
            session_resources = restarted.sessions[session_id].session[
                "teaching_resources"
            ]
            self.assertEqual(len(session_resources), 1)
            self.assertNotIn("staged_resource_id", session_resources[0])
            self.assertLessEqual(len(session_resources[0]["extracted_text"]), 12_000)
            self.assertNotIn(private_tail, session_resources[0]["extracted_text"])

            assert restarted.resource_index_store is not None
            retrieval_resource = restarted.resource_index_store.get_retrieval_resource(
                content_hash
            )
            self.assertIsNotNone(retrieval_resource)
            assert retrieval_resource is not None
            self.assertIn(private_tail, retrieval_resource["extracted_text"])


if __name__ == "__main__":
    unittest.main()
