from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from teaching_skill_miner.deepseek_client import DeepSeekClientError
from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.teacher_agent import _refresh_integrity, canonical_sha256
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardError,
    _request_fingerprint,
    build_teacher_agent_dashboard_snapshot,
)
from teaching_skill_miner.teacher_agent_live import (
    LIVE_PROMPT_VERSION,
    LiveAgentOptions,
)
from teaching_skill_miner.teacher_agent_store import TeacherAgentStoreError


class _OfflineLiveClient:
    """Credential-free test double that always exercises the safe fallback."""

    def __init__(
        self,
        *,
        provider: str = "deepseek",
        model: str = "deepseek-v4-flash",
    ) -> None:
        self.provider = provider
        self.model = model

    def public_status(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_origin": "https://api.deepseek.com",
            "configured": True,
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "remote_student_data_opt_in": True,
            "api_key_exposed": False,
        }

    def chat_json(
        self,
        _messages: object,
        *,
        request_kind: str,
        require_remote_consent: bool = True,
    ) -> tuple[dict[str, object], dict[str, object]]:
        del request_kind, require_remote_consent
        raise DeepSeekClientError("offline live client")


class TeacherAgentStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = project_root()
        cls.library_path = root / "data/teacher_agent_skill_library.json"
        cls.library_v2_path = root / "data/teacher_agent_skill_library_v2.json"
        cls.input_path = root / "data/teacher_agent_demo_input.json"
        cls.cases_path = root / "data/teacher_agent_evaluation_cases.json"

    def _snapshot(
        self,
        store_path: Path | None = None,
        *,
        library_path: Path | None = None,
        client: object | None = None,
        live_options: LiveAgentOptions | None = None,
    ):
        kwargs = {}
        if client is not None:
            assert store_path is not None
            kwargs = {
                "consent_store_path": store_path.with_name(
                    store_path.name + ".consent"
                ),
                "consent_signing_secret": (
                    b"store-test-consent-signing-secret-material-32-bytes"
                ),
            }
        snapshot = build_teacher_agent_dashboard_snapshot(
            library_path or self.library_path,
            self.input_path,
            self.cases_path,
            client=client,
            live_options=live_options,
            store_path=store_path,
            **kwargs,
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
    def _start_body(snapshot, key: str, *, profile_ref: str | None = None):
        profile = deepcopy(snapshot.demo_input["student_profile"])
        if profile_ref is not None:
            profile["profile_ref"] = profile_ref
        body = {
            "goal": deepcopy(snapshot.demo_input["goal"]),
            "student_profile": profile,
            "start_idempotency_key": key,
        }
        if snapshot.client is not None:
            body["remote_consent_id"] = next(
                receipt["consent_id"]
                for receipt in snapshot.list_remote_consents({})["receipts"]
                if receipt["purpose"] == "remote_teaching"
                and receipt["status"] == "active"
            )
        return body

    @staticmethod
    def _step_body(session: dict, key: str, *, answer: str = "我还需要一个例子"):
        return {
            "session_id": session["session_id"],
            "expected_round": session["rounds_completed"],
            "expected_question_id": session["expected_question_id"],
            "expected_context_version": session["context_version"],
            "profile_revision": session["profile_summary"]["profile_revision"],
            "idempotency_key": key,
            "learner_response": answer,
            "signal": "partial",
        }

    def test_store_is_opt_in_and_restart_recovers_idempotent_responses(self) -> None:
        self.assertIsNone(self._snapshot().store)
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            first = self._snapshot(store_path)
            start_body = self._start_body(first, "cold-start-001")
            started = first.start(start_body)
            step_body = self._step_body(started, "cold-turn-001")
            stepped = first.step(step_body)
            command_body = {
                "session_id": stepped["session_id"],
                "command": "auto",
                "command_idempotency_key": "cold-command-001",
                "expected_round": stepped["rounds_completed"],
                "expected_question_id": stepped["expected_question_id"],
                "expected_context_version": stepped["context_version"],
                "profile_revision": stepped["profile_summary"]["profile_revision"],
            }
            commanded = first.command(command_body)

            restarted = self._snapshot(store_path)
            self.assertEqual(
                restarted.resume({"session_id": started["session_id"]}), commanded
            )
            self.assertEqual(restarted.start(start_body), started)
            self.assertEqual(restarted.step(step_body), stepped)
            self.assertEqual(restarted.command(command_body), commanded)
            event_types = [event["event_type"] for event in restarted.store.events]
            self.assertIn("session_started", event_types)
            self.assertIn("turn_started", event_types)
            self.assertIn("turn_committed", event_types)
            self.assertIn("context_checkpoint", event_types)
            for seq, event in enumerate(restarted.store.events, 1):
                self.assertEqual(event["seq"], seq)
                self.assertEqual(len(event["hash"]), 64)
                self.assertIn("previous_hash", event)
                self.assertEqual(event["session_id"], started["session_id"])
                self.assertIn("round", event)
                self.assertIn("question_id", event)
                self.assertIn("context_version", event)
                self.assertIn("profile_revision", event)
                self.assertIn("idempotency_key", event)
                self.assertIn("request_fingerprint", event)

    def test_truncated_tail_is_ignored_repaired_and_session_remains_resumable(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            snapshot = self._snapshot(store_path)
            started = snapshot.start(self._start_body(snapshot, "tail-start-001"))
            with store_path.open("ab") as stream:
                stream.write(b'{"truncated":true')

            restarted = self._snapshot(store_path)
            self.assertEqual(
                restarted.resume({"session_id": started["session_id"]}), started
            )
            repaired = store_path.read_bytes()
            self.assertTrue(repaired.endswith(b"\n"))
            self.assertNotIn(b"truncated", repaired)

    def test_complete_hash_tamper_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            snapshot = self._snapshot(store_path)
            snapshot.start(self._start_body(snapshot, "tamper-start-001"))
            lines = store_path.read_text(encoding="utf-8").splitlines()
            first_event = json.loads(lines[0])
            first_event["data"]["backend"] = "tampered"
            lines[0] = json.dumps(
                first_event,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            store_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(TeacherAgentDashboardError, "cannot be opened"):
                self._snapshot(store_path)

    def test_flush_failure_never_publishes_unflushed_candidate(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            snapshot = self._snapshot(store_path)
            started = snapshot.start(self._start_body(snapshot, "flush-start-001"))
            step_body = self._step_body(started, "flush-turn-001")
            original_flush = snapshot.store._flush
            flush_count = 0

            def fail_commit_barrier(stream) -> None:
                nonlocal flush_count
                flush_count += 1
                if flush_count in {2, 3}:
                    raise OSError("simulated fsync failure")
                original_flush(stream)

            with patch.object(
                snapshot.store, "_flush", side_effect=fail_commit_barrier
            ):
                with self.assertRaisesRegex(
                    TeacherAgentDashboardError, "store write failed"
                ):
                    snapshot.step(step_body)
            self.assertEqual(
                snapshot.resume({"session_id": started["session_id"]}), started
            )
            self.assertNotIn(
                "flush-turn-001",
                snapshot.sessions[started["session_id"]].step_idempotency_cache,
            )
            self.assertEqual(
                [event["event_type"] for event in snapshot.store.events][-2:],
                ["turn_started", "turn_aborted"],
            )

    def test_dangling_turn_is_recovered_as_aborted_without_same_key_replay(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            snapshot = self._snapshot(store_path)
            started = snapshot.start(self._start_body(snapshot, "dangling-start-001"))
            record = snapshot.sessions[started["session_id"]]
            step_body = self._step_body(started, "dangling-turn-001")
            fingerprint = _request_fingerprint(step_body)
            turn_id = "dangling-turn-id-001"
            snapshot.store.append_batch(
                [
                    snapshot._event_specification(
                        "turn_started",
                        started["session_id"],
                        record,
                        idempotency_key="dangling-turn-001",
                        request_fingerprint=fingerprint,
                        turn_id=turn_id,
                        data={"remote_model_call_may_follow": True},
                    )
                ]
            )

            restarted = self._snapshot(store_path)
            aborts = [
                event
                for event in restarted.store.events
                if event["event_type"] == "turn_aborted"
                and event.get("turn_id") == turn_id
            ]
            self.assertEqual(len(aborts), 1)
            self.assertTrue(aborts[0]["data"]["recovered"])
            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "aborted during cold recovery"
            ):
                restarted.step(step_body)
            retried = restarted.step(
                {**step_body, "idempotency_key": "dangling-turn-retry-002"}
            )
            self.assertEqual(retried["rounds_completed"], 1)

            restarted_again = self._snapshot(store_path)
            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "aborted during cold recovery"
            ):
                restarted_again.step(step_body)

    def test_latest_checkpoint_replays_a_complete_committed_tail(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            snapshot = self._snapshot(store_path)
            started = snapshot.start(self._start_body(snapshot, "replay-start-001"))
            stepped = snapshot.step(self._step_body(started, "replay-turn-001"))
            lines = store_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[-1])["event_type"], "context_checkpoint")
            store_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

            restarted = self._snapshot(store_path)
            self.assertEqual(
                restarted.resume({"session_id": started["session_id"]}), stepped
            )

    def test_multiple_sessions_recover_independently(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            snapshot = self._snapshot(store_path)
            first = snapshot.start(
                self._start_body(snapshot, "multi-start-a", profile_ref="student-a")
            )
            second = snapshot.start(
                self._start_body(snapshot, "multi-start-b", profile_ref="student-b")
            )
            first_stepped = snapshot.step(
                self._step_body(first, "multi-turn-a", answer="学生 A 的回答")
            )

            restarted = self._snapshot(store_path)
            self.assertEqual(
                restarted.resume({"session_id": first["session_id"]}), first_stepped
            )
            self.assertEqual(
                restarted.resume({"session_id": second["session_id"]}), second
            )
            second_stepped = restarted.step(
                self._step_body(second, "multi-turn-b", answer="学生 B 的回答")
            )
            self.assertEqual(second_stepped["rounds_completed"], 1)
            self.assertEqual(
                restarted.resume({"session_id": first["session_id"]}), first_stepped
            )

    def test_capacity_archives_101_sessions_and_project_reference_lazy_loads(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store_path = root / "teacher-agent.jsonl"
            project_path = root / "projects"
            snapshot = build_teacher_agent_dashboard_snapshot(
                self.library_path,
                self.input_path,
                self.cases_path,
                store_path=store_path,
                project_store_path=project_path,
            )
            contract = snapshot.bootstrap()["interaction_contract"]
            self.assertEqual(
                contract["session_capacity_policy"], "durable_archive_lazy_load"
            )
            self.assertTrue(
                contract["capacity_archive_never_removes_authoritative_session"]
            )
            project = snapshot.create_project({"title": "容量归档课程"})["project"]
            first = snapshot.start(
                self._start_body(snapshot, "archive-start-000", profile_ref="student-000")
            )
            snapshot.add_project_reference(
                project["project_id"],
                {"kind": "teaching_session", "reference_id": first["session_id"]},
            )
            for index in range(1, 101):
                snapshot.start(
                    self._start_body(
                        snapshot,
                        f"archive-start-{index:03d}",
                        profile_ref=f"student-{index:03d}",
                    )
                )

            self.assertEqual(len(snapshot.sessions), 16)
            self.assertEqual(len(snapshot.archived_session_ids), 85)
            self.assertNotIn(first["session_id"], snapshot.sessions)
            self.assertTrue(
                any(
                    event["event_type"] == "session_archived"
                    for event in snapshot.store.events
                )
            )

            restarted = build_teacher_agent_dashboard_snapshot(
                self.library_path,
                self.input_path,
                self.cases_path,
                store_path=store_path,
                project_store_path=project_path,
            )
            self.assertEqual(len(restarted.sessions), 16)
            self.assertIn(first["session_id"], restarted.archived_session_ids)
            self.assertIn(
                first["session_id"],
                restarted.read_project(project["project_id"])["project"][
                    "teaching_session_ids"
                ],
            )

            with patch.object(
                restarted.store,
                "recover_session",
                wraps=restarted.store.recover_session,
            ) as recover_session:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    responses = list(
                        executor.map(
                            lambda _index: restarted.resume(
                                {"session_id": first["session_id"]}
                            ),
                            range(8),
                        )
                    )
                self.assertEqual(recover_session.call_count, 1)
            self.assertTrue(all(response == first for response in responses))
            self.assertEqual(len(restarted.sessions), 16)
            self.assertIn(first["session_id"], restarted.sessions)
            self.assertNotIn(first["session_id"], restarted.archived_session_ids)

            restarted_again = build_teacher_agent_dashboard_snapshot(
                self.library_path,
                self.input_path,
                self.cases_path,
                store_path=store_path,
                project_store_path=project_path,
            )
            self.assertEqual(
                restarted_again.resume({"session_id": first["session_id"]}), first
            )

    def test_legacy_capacity_remove_event_migrates_but_explicit_remove_does_not(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            snapshot = self._snapshot(store_path)
            started = snapshot.start(
                self._start_body(snapshot, "legacy-capacity-start-001")
            )
            record = snapshot.sessions[started["session_id"]]
            snapshot.store.append_batch(
                [
                    snapshot._event_specification(
                        "session_stopped",
                        started["session_id"],
                        record,
                        idempotency_key=None,
                        request_fingerprint=None,
                        data={
                            "reason": "active_session_capacity_eviction",
                            "remove_session": True,
                        },
                    )
                ]
            )

            restarted = self._snapshot(store_path)
            resumed = restarted.resume({"session_id": started["session_id"]})
            self.assertEqual(resumed, started)
            stepped = restarted.step(
                self._step_body(resumed, "legacy-capacity-turn-001")
            )
            self.assertEqual(stepped["rounds_completed"], 1)

    def test_replacement_persists_session_stopped_and_removal(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            snapshot = self._snapshot(store_path)
            original = snapshot.start(
                self._start_body(snapshot, "replace-original-001")
            )
            replacement_body = self._start_body(
                snapshot, "replace-new-002", profile_ref="replacement-student"
            )
            replacement_body.update(
                {
                    "replace_session_id": original["session_id"],
                    "replace_expected_round": original["rounds_completed"],
                    "replace_expected_question_id": original["expected_question_id"],
                    "replace_expected_context_version": original["context_version"],
                    "replace_expected_profile_revision": original["profile_summary"][
                        "profile_revision"
                    ],
                }
            )
            replacement = snapshot.start(replacement_body)
            stopped = [
                event
                for event in snapshot.store.events
                if event["event_type"] == "session_stopped"
                and event["session_id"] == original["session_id"]
            ]
            self.assertEqual(len(stopped), 1)
            self.assertTrue(stopped[0]["data"]["remove_session"])

            restarted = self._snapshot(store_path)
            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "no longer available"
            ):
                restarted.resume({"session_id": original["session_id"]})
            self.assertEqual(
                restarted.resume({"session_id": replacement["session_id"]}),
                replacement,
            )

    def test_live_primary_skill_subset_recovers_and_keeps_all_support_skills(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            client = _OfflineLiveClient()
            options = LiveAgentOptions(
                maximum_supporting_skills=1,
                minimum_assessment_confidence=0.58,
                maximum_context_chars=8_000,
                maximum_context_turns=4,
            )
            snapshot = self._snapshot(
                store_path, client=client, live_options=options
            )
            primary_ids = [
                skill["skill_id"]
                for skill in snapshot.library["skills"]
                if skill["role"] != "support"
            ][:5]
            support_ids = [
                skill["skill_id"]
                for skill in snapshot.library["skills"]
                if skill["role"] == "support"
            ]
            start_body = {
                **self._start_body(snapshot, "live-subset-start-001"),
                "allowed_skill_ids": primary_ids,
            }
            started = snapshot.start(start_body)

            restarted = self._snapshot(
                store_path,
                client=_OfflineLiveClient(),
                live_options=options,
            )
            resumed = restarted.resume({"session_id": started["session_id"]})
            self.assertEqual(resumed, started)
            stored_skills = restarted.sessions[started["session_id"]].session[
                "skill_library"
            ]["skills"]
            self.assertEqual(
                [skill["skill_id"] for skill in stored_skills if skill["role"] != "support"],
                primary_ids,
            )
            self.assertEqual(
                [skill["skill_id"] for skill in stored_skills if skill["role"] == "support"],
                support_ids,
            )

            advanced = restarted.step(
                {
                    **self._step_body(
                    resumed,
                    "live-subset-turn-001",
                    answer="我仍然需要一个更具体的例子",
                    ),
                    "remote_consent_id": next(
                        receipt["consent_id"]
                        for receipt in restarted.list_remote_consents({})["receipts"]
                        if receipt["purpose"] == "remote_teaching"
                        and receipt["status"] == "active"
                    ),
                }
            )
            self.assertEqual(advanced["rounds_completed"], 1)

    def test_live_teach_first_session_cold_resumes_with_derived_skill_contract(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            options = LiveAgentOptions(maximum_context_chars=8_000)
            snapshot = self._snapshot(
                store_path,
                library_path=self.library_v2_path,
                client=_OfflineLiveClient(),
                live_options=options,
            )
            start_body = self._start_body(snapshot, "teach-first-resume-start-001")
            start_body["goal"]["learning_intent"] = "teach_first"
            started = snapshot.start(start_body)

            stored_skills = {
                skill["skill_id"]: skill
                for skill in snapshot.sessions[started["session_id"]]
                .session["skill_library"]["skills"]
            }
            self.assertIn(
                "not_observed",
                stored_skills["skill_contextual_problem_setup"][
                    "applicable_signals"
                ],
            )
            self.assertIn(
                "correct",
                stored_skills["skill_concrete_example_bridge"][
                    "applicable_signals"
                ],
            )

            restarted = self._snapshot(
                store_path,
                library_path=self.library_v2_path,
                client=_OfflineLiveClient(),
                live_options=options,
            )
            self.assertEqual(
                restarted.resume({"session_id": started["session_id"]}),
                started,
            )

            record = restarted.sessions[started["session_id"]]
            record.session["skill_library"]["skills"][0]["name"] += "（已篡改）"
            record.session["skill_library_fingerprint"] = canonical_sha256(
                record.session["skill_library"]
            )
            _refresh_integrity(record.session)
            restarted.store.append_batch(
                [
                    restarted._checkpoint_specification(
                        started["session_id"],
                        record,
                        idempotency_key=None,
                        request_fingerprint=None,
                        reason="test_teach_first_skill_mutation",
                    )
                ]
            )

            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "Skill definitions differ"
            ):
                self._snapshot(
                    store_path,
                    library_path=self.library_v2_path,
                    client=_OfflineLiveClient(),
                    live_options=options,
                )

    def test_live_recovery_rejects_runtime_policy_drift(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            original_options = LiveAgentOptions(
                maximum_supporting_skills=1,
                minimum_assessment_confidence=0.58,
                maximum_context_chars=8_000,
                maximum_context_turns=4,
            )
            snapshot = self._snapshot(
                store_path,
                client=_OfflineLiveClient(),
                live_options=original_options,
            )
            start_body = {
                **self._start_body(snapshot, "live-policy-start-001"),
            }
            snapshot.start(start_body)

            drift_cases = (
                (
                    "context policy",
                    _OfflineLiveClient(),
                    LiveAgentOptions(
                        maximum_supporting_skills=1,
                        minimum_assessment_confidence=0.58,
                        maximum_context_chars=8_000,
                        maximum_context_turns=5,
                    ),
                ),
                (
                    "model",
                    _OfflineLiveClient(model="deepseek-v4-flash-reconfigured"),
                    original_options,
                ),
                (
                    "provider",
                    _OfflineLiveClient(provider="different-provider"),
                    original_options,
                ),
            )
            for label, client, options in drift_cases:
                with self.subTest(label=label), self.assertRaisesRegex(
                    TeacherAgentDashboardError, "runtime policy does not match"
                ):
                    self._snapshot(
                        store_path,
                        client=client,
                        live_options=options,
                    )

    def test_live_recovery_migrates_only_the_compatible_prompt_version(
        self,
    ) -> None:
        legacy_prompt_versions = (
            "teaching_agent_assess_route_act_v15_"
            "correction_chain_taxonomy_contract",
            "teaching_agent_assess_route_act_v16_"
            "grounded_clarification_contract",
            "teaching_agent_assess_route_act_v17_"
            "guide_learning_confusion_recovery",
        )
        for index, legacy_prompt_version in enumerate(legacy_prompt_versions, 1):
            with self.subTest(legacy_prompt_version=legacy_prompt_version):
                with TemporaryDirectory() as directory:
                    store_path = Path(directory) / "teacher-agent.jsonl"
                    options = LiveAgentOptions(maximum_context_chars=8_000)
                    with patch(
                        "teaching_skill_miner.teacher_agent_live.LIVE_PROMPT_VERSION",
                        legacy_prompt_version,
                    ):
                        legacy = self._snapshot(
                            store_path,
                            client=_OfflineLiveClient(),
                            live_options=options,
                        )
                        start_body = {
                            **self._start_body(
                                legacy,
                                f"live-prompt-migration-start-{index:03d}",
                            ),
                        }
                        started = legacy.start(start_body)

                    restarted = self._snapshot(
                        store_path,
                        client=_OfflineLiveClient(),
                        live_options=options,
                    )
                    session = restarted.sessions[started["session_id"]].session
                    runtime = session["agent_runtime"]
                    self.assertEqual(runtime["prompt_version"], LIVE_PROMPT_VERSION)
                    trace = runtime["last_model_trace"]
                    self.assertEqual(
                        trace["runtime_policy_contract"]["prompt_version"],
                        LIVE_PROMPT_VERSION,
                    )
                    self.assertEqual(
                        trace["runtime_policy_migration"],
                        {
                            "from_prompt_version": legacy_prompt_version,
                            "to_prompt_version": LIVE_PROMPT_VERSION,
                            "only_prompt_version_changed": True,
                            "historical_action_traces_preserved": True,
                        },
                    )

                    restarted_again = self._snapshot(
                        store_path,
                        client=_OfflineLiveClient(),
                        live_options=options,
                    )
                    self.assertEqual(
                        restarted_again.sessions[started["session_id"]]
                        .session["agent_runtime"]["prompt_version"],
                        LIVE_PROMPT_VERSION,
                    )

    def test_live_recovery_rejects_an_unapproved_prompt_version_jump(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            options = LiveAgentOptions(maximum_context_chars=8_000)
            with patch(
                "teaching_skill_miner.teacher_agent_live.LIVE_PROMPT_VERSION",
                "teaching_agent_assess_route_act_v14_unapproved_test_prompt",
            ):
                legacy = self._snapshot(
                    store_path,
                    client=_OfflineLiveClient(),
                    live_options=options,
                )
                legacy.start(
                    {
                        **self._start_body(
                            legacy, "live-prompt-reject-start-001"
                        ),
                    }
                )

            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "runtime policy does not match"
            ):
                self._snapshot(
                    store_path,
                    client=_OfflineLiveClient(),
                    live_options=options,
                )

    def test_live_recovery_rejects_unknown_or_modified_subset_skills(self) -> None:
        for mutation in ("unknown", "modified"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as directory:
                store_path = Path(directory) / "teacher-agent.jsonl"
                options = LiveAgentOptions(maximum_context_chars=8_000)
                snapshot = self._snapshot(
                    store_path,
                    client=_OfflineLiveClient(),
                    live_options=options,
                )
                primary_ids = [
                    skill["skill_id"]
                    for skill in snapshot.library["skills"]
                    if skill["role"] != "support"
                ][:5]
                started = snapshot.start(
                    {
                        **self._start_body(
                            snapshot, f"live-skill-{mutation}-start-001"
                        ),
                        "allowed_skill_ids": primary_ids,
                    }
                )
                record = snapshot.sessions[started["session_id"]]
                if mutation == "unknown":
                    unknown = deepcopy(record.session["skill_library"]["skills"][-1])
                    unknown["skill_id"] = "skill_unknown_injected_support"
                    unknown["name"] = "未知注入支持 Skill"
                    record.session["skill_library"]["skills"].append(unknown)
                else:
                    record.session["skill_library"]["skills"][0]["name"] += (
                        "（已篡改）"
                    )
                record.session["skill_library_fingerprint"] = canonical_sha256(
                    record.session["skill_library"]
                )
                _refresh_integrity(record.session)
                snapshot.store.append_batch(
                    [
                        snapshot._checkpoint_specification(
                            started["session_id"],
                            record,
                            idempotency_key=None,
                            request_fingerprint=None,
                            reason=f"test_{mutation}_skill_mutation",
                        )
                    ]
                )

                expected = (
                    "contains unknown Skills"
                    if mutation == "unknown"
                    else "Skill definitions differ"
                )
                with self.assertRaisesRegex(TeacherAgentDashboardError, expected):
                    self._snapshot(
                        store_path,
                        client=_OfflineLiveClient(),
                        live_options=options,
                    )

    def test_live_recovery_rejects_subset_that_drops_an_unused_support_skill(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            options = LiveAgentOptions(maximum_context_chars=8_000)
            snapshot = self._snapshot(
                store_path,
                client=_OfflineLiveClient(),
                live_options=options,
            )
            primary_ids = [
                "skill_concrete_example_bridge",
                "skill_concept_mapping",
                "skill_misconception_contrast",
                "skill_transfer_check",
                "skill_learner_summary",
            ]
            started = snapshot.start(
                {
                    **self._start_body(snapshot, "live-support-start-001"),
                    "allowed_skill_ids": primary_ids,
                }
            )
            record = snapshot.sessions[started["session_id"]]
            record.session["skill_library"]["skills"] = [
                skill
                for skill in record.session["skill_library"]["skills"]
                if skill["skill_id"] != "skill_minimal_hint"
            ]
            record.session["skill_library_fingerprint"] = canonical_sha256(
                record.session["skill_library"]
            )
            _refresh_integrity(record.session)
            snapshot.store.append_batch(
                [
                    snapshot._checkpoint_specification(
                        started["session_id"],
                        record,
                        idempotency_key=None,
                        request_fingerprint=None,
                        reason="test_missing_support_skill",
                    )
                ]
            )

            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "preserve all support Skills"
            ):
                self._snapshot(
                    store_path,
                    client=_OfflineLiveClient(),
                    live_options=options,
                )

    def test_store_rejects_terminal_turn_without_a_started_turn(self) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            snapshot = self._snapshot(store_path)
            started = snapshot.start(self._start_body(snapshot, "lifecycle-start-001"))
            record = snapshot.sessions[started["session_id"]]
            with self.assertRaisesRegex(
                TeacherAgentStoreError, "terminal turn event has no active turn"
            ):
                snapshot.store.append_batch(
                    [
                        snapshot._event_specification(
                            "turn_committed",
                            started["session_id"],
                            record,
                            idempotency_key="lifecycle-turn-001",
                            request_fingerprint="fingerprint",
                            turn_id="never-started",
                            data={"record": {}},
                        )
                    ]
                )

    def test_store_cannot_resurrect_a_replaced_session_with_a_late_checkpoint(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            snapshot = self._snapshot(store_path)
            started = snapshot.start(self._start_body(snapshot, "lifecycle-start-002"))
            record = snapshot.sessions[started["session_id"]]
            snapshot.store.append_batch(
                [
                    snapshot._event_specification(
                        "session_stopped",
                        started["session_id"],
                        record,
                        idempotency_key="lifecycle-stop-001",
                        request_fingerprint="fingerprint",
                        data={"remove_session": True},
                    )
                ]
            )
            with self.assertRaisesRegex(
                TeacherAgentStoreError, "event follows a removed session"
            ):
                snapshot.store.append_batch(
                    [
                        snapshot._checkpoint_specification(
                            started["session_id"],
                            record,
                            idempotency_key=None,
                            request_fingerprint=None,
                            reason="late_checkpoint",
                        )
                    ]
                )

    def test_store_allows_only_inflight_abort_after_profile_replacement(self) -> None:
        """A late remote response must be receipted, never committed or revived."""

        with TemporaryDirectory() as directory:
            store_path = Path(directory) / "teacher-agent.jsonl"
            snapshot = self._snapshot(store_path)
            started = snapshot.start(self._start_body(snapshot, "lifecycle-start-003"))
            record = snapshot.sessions[started["session_id"]]
            turn_id = "inflight-before-replacement"
            snapshot.store.append_batch(
                [
                    snapshot._event_specification(
                        "turn_started",
                        started["session_id"],
                        record,
                        idempotency_key="lifecycle-inflight-001",
                        request_fingerprint="fingerprint",
                        turn_id=turn_id,
                        data={"remote_model_call_may_follow": True},
                    ),
                    snapshot._event_specification(
                        "session_stopped",
                        started["session_id"],
                        record,
                        idempotency_key="lifecycle-replace-001",
                        request_fingerprint="fingerprint",
                        data={"remove_session": True},
                    ),
                    snapshot._event_specification(
                        "turn_aborted",
                        started["session_id"],
                        record,
                        idempotency_key="lifecycle-inflight-001",
                        request_fingerprint="fingerprint",
                        turn_id=turn_id,
                        data={"reason": "explicit_session_replacement"},
                    ),
                ]
            )
            restarted = self._snapshot(store_path)
            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "no longer available"
            ):
                restarted.resume({"session_id": started["session_id"]})


if __name__ == "__main__":
    unittest.main()
