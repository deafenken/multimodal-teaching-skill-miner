from __future__ import annotations

from contextlib import redirect_stdout
from html.parser import HTMLParser
import http.client
import io
import json
import threading
import unittest
from urllib.parse import urlsplit

from teaching_skill_miner.cli import main
from teaching_skill_miner.deepseek_client import DeepSeekClientError
from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.teacher_agent_dashboard import (
    HTML_RESOURCE,
    SCRIPT_RESOURCE,
    STYLE_RESOURCE,
    TeacherAgentDashboardError,
    _resource_bytes,
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
    teacher_agent_dashboard_self_check,
)
from teaching_skill_miner.teacher_agent_live import LiveAgentOptions


class _AgentDashboardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.screens: list[str] = []
        self.external_assets: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del tag
        values = dict(attrs)
        if values.get("data-screen-label"):
            self.screens.append(str(values["data-screen-label"]))
        for key in ("src", "href"):
            value = values.get(key)
            if value and value.startswith(("http://", "https://", "//")):
                self.external_assets.append(value)


class _FakeLiveClient:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.chat_json_call_count = 0
        self.fail_first = fail_first

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
        if self.fail_first and self.chat_json_call_count == 1:
            raise DeepSeekClientError("simulated transient start failure")
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
                    "selection_reason": "先确认学生是否具备必要的前置知识。",
                    "next_focus": "prerequisite",
                },
                "teacher_action": {
                    "type": "ask_one_question",
                    "message": "请先说出一个完成当前目标所需的前置概念。",
                    "expected_signal": "学生给出一个相关的前置概念。",
                },
                "stop_recommendation": {"should_stop": False, "reason": ""},
            },
            {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "thinking_mode": "disabled",
                "temperature": 0.0,
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


class TeacherAgentDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = project_root()
        cls.library_path = cls.root / "data/teacher_agent_skill_library.json"
        cls.input_path = cls.root / "data/teacher_agent_demo_input.json"
        cls.cases_path = cls.root / "data/teacher_agent_evaluation_cases.json"
        cls.v2_library_path = cls.root / "data/teacher_agent_skill_library_v2.json"
        cls.neural_manifest_path = cls.root / "data/neural_v1_runtime_manifest.json"
        cls.outcome_path = cls.root / "data/teacher_agent_learning_outcome_demo.json"
        cls.benchmark_receipt_path = (
            cls.root / "data/teacher_agent_free_text_benchmark_receipt.json"
        )

    def _offline_snapshot(self):
        return build_teacher_agent_dashboard_snapshot(
            self.library_path, self.input_path, self.cases_path
        )

    def _start_offline(self, snapshot, *, start_key: str | None = None):
        body = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": start_key
            or f"test-start-{len(snapshot.start_idempotency_cache) + 1:03d}",
        }
        if snapshot.session_id is not None:
            body["replace_session_id"] = snapshot.session_id
        return snapshot.start(body)

    def test_generic_frontend_has_four_clear_screens_and_no_external_assets(self) -> None:
        html = _resource_bytes(HTML_RESOURCE).decode("utf-8")
        parser = _AgentDashboardParser()
        parser.feed(html)
        self.assertEqual(
            parser.screens,
            [
                "01 Session setup",
                "02 Live teaching loop",
                "03 Student state",
                "04 Reproducible evaluation",
            ],
        )
        self.assertEqual(parser.external_assets, [])
        self.assertIn("teacher_agent_demo.css", html)
        self.assertIn("teacher_agent_demo.js", html)
        script = _resource_bytes(SCRIPT_RESOURCE).decode("utf-8")
        self.assertIn("fetch(", script)
        self.assertIn("replaceChildren", script)
        self.assertNotIn(".innerHTML", script)
        self.assertIn('id="adaptiveProfileStatus"', html)
        self.assertIn('id="adaptiveConfidenceBar"', html)
        self.assertIn("Agent 候选画像", html)
        self.assertIn("renderAdaptiveStudentProfile(session)", script)
        self.assertIn("不会覆盖教师输入", script)
        self.assertIn('needsReview ? "需人工复核" : "未确认"', script)
        self.assertIn('id="learningViewButton"', html)
        self.assertIn('id="evaluationViewButton"', html)
        self.assertIn('id="contextMemoryList"', html)
        self.assertIn('id="contextOperationText"', html)
        self.assertIn('id="contextSnapshotRound"', html)
        self.assertIn("暂无请求快照", html)
        self.assertIn("模型实际读取的请求快照", html)
        self.assertIn('id="learnerResponse"', html)
        self.assertIn("Skill 选择依据", html)
        self.assertIn("renderContextMemory(session)", script)
        self.assertIn('snapshot.operation', script)
        self.assertIn('snapshot.session_round_before_request', script)
        self.assertIn("本区不会混入这些处理后结果", script)
        self.assertNotIn("const fallbackRecent = history.slice(-6)", script)
        self.assertNotIn("session.student_state?.misconceptions", script)
        self.assertNotIn("session.goal_plan?.intermediate_objectives", script)
        self.assertIn("确定性统计与原文抽取检查点", script)
        self.assertIn("startPayload.start_idempotency_key = startKey", script)
        self.assertIn("startPayload.replace_session_id", script)
        self.assertIn("payload.idempotency_key = idempotencyKey", script)
        self.assertIn("expected_round: round", script)
        self.assertIn("function turnRequestFingerprint(payload)", script)
        self.assertIn("manual_skill_id: payload.manual_skill_id ?? null", script)
        self.assertIn("signal: payload.signal ?? null", script)
        self.assertIn('app.pendingTurn?.fingerprint === requestFingerprint', script)
        self.assertIn('select("#fallbackSignalInput").addEventListener("change", clearPendingTurn)', script)
        self.assertIn('setAppView("learning")', script)
        self.assertIn("showSetupForm(false)", script)
        self.assertNotIn("scrollIntoView", script)
        context_renderer = script.split(
            "function renderContextMemory(session) {", 1
        )[1].split("function masteryDeltaNodes", 1)[0]
        self.assertIn("if (!hasSnapshot)", context_renderer)
        self.assertIn('textContent = "暂无请求快照"', context_renderer)
        for forbidden_fallback in (
            "session.history",
            "session.student_state",
            "session.goal_plan",
            "app.lastSetupPayload",
        ):
            with self.subTest(forbidden_fallback=forbidden_fallback):
                self.assertNotIn(forbidden_fallback, context_renderer)
        fingerprint_builder = script.split(
            "function turnRequestFingerprint(payload) {", 1
        )[1].split("function clearPendingTurn", 1)[0]
        for logical_field in (
            "learner_response",
            "session_id",
            "expected_round",
            "signal",
            "signal_confidence",
            "misconception_tag",
            "manual_skill_id",
        ):
            with self.subTest(logical_field=logical_field):
                self.assertIn(f"{logical_field}: payload.{logical_field}", fingerprint_builder)
        self.assertGreater(len(_resource_bytes(STYLE_RESOURCE)), 10_000)

    def test_self_check_and_cli_check_pass(self) -> None:
        report = teacher_agent_dashboard_self_check(
            self.library_path, self.input_path, self.cases_path
        )
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["real_time_skill_switching"])
        self.assertFalse(report["real_learning_effectiveness_established"])
        with redirect_stdout(io.StringIO()) as output:
            code = main(["teacher-agent-dashboard", "--check"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())["passed"])

    def test_v2_bootstrap_exposes_live_model_evidence_without_overclaiming(self) -> None:
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            neural_v1_manifest_path=self.neural_manifest_path,
            learning_outcome_path=self.outcome_path,
            free_text_benchmark_receipt_path=self.benchmark_receipt_path,
        )
        bootstrap = snapshot.bootstrap()
        receipt = bootstrap["evaluation"]["free_text_benchmark"]
        outcome = bootstrap["evaluation"]["learning_outcome"]
        self.assertEqual(
            receipt["online_deepseek"]["model"], "deepseek-v4-flash"
        )
        self.assertEqual(
            receipt["online_deepseek"]["metrics"]["signal"][
                "end_to_end_all_attempts"
            ]["macro_f1"],
            0.875325,
        )
        self.assertFalse(receipt["claim_boundary"]["expert_validated"])
        self.assertFalse(
            receipt["claim_boundary"]["deployment_accuracy_established"]
        )
        self.assertEqual(outcome["metrics"]["absolute_gain"], 0.4)
        self.assertFalse(
            outcome["claim_boundary"]["real_learner_effectiveness_established"]
        )
        self.assertFalse(
            bootstrap["neural_v1"]["materialization_gate"]["passed"]
        )
        self.assertFalse(
            bootstrap["interaction_contract"][
                "adaptive_profile_candidates_are_teacher_confirmed"
            ]
        )

    def test_start_is_idempotent_and_requires_explicit_session_replacement(
        self,
    ) -> None:
        snapshot = self._offline_snapshot()
        contract = snapshot.bootstrap()["interaction_contract"]
        self.assertTrue(contract["start_requires_idempotency_key"])
        self.assertTrue(
            contract["active_session_replacement_requires_session_id"]
        )
        base = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
        }
        with self.assertRaisesRegex(
            TeacherAgentDashboardError, "start_idempotency_key"
        ):
            snapshot.start(base)

        first_body = {**base, "start_idempotency_key": "start-001"}
        first = snapshot.start(first_body)
        replay = snapshot.start(first_body)
        self.assertEqual(replay, first)
        self.assertEqual(snapshot.session_id, first["session_id"])

        with self.assertRaisesRegex(
            TeacherAgentDashboardError, "different request"
        ):
            snapshot.start(
                {
                    **first_body,
                    "goal": {
                        **snapshot.demo_input["goal"],
                        "objective": "不能复用同一启动键的新目标",
                    },
                }
            )
        with self.assertRaisesRegex(
            TeacherAgentDashboardError, "replace_session_id"
        ):
            snapshot.start({**base, "start_idempotency_key": "start-002"})
        with self.assertRaisesRegex(
            TeacherAgentDashboardError, "replace_session_id"
        ):
            snapshot.start(
                {
                    **base,
                    "start_idempotency_key": "start-002",
                    "replace_session_id": "another-tab-without-current-session",
                }
            )
        self.assertEqual(snapshot.session_id, first["session_id"])

        second_body = {
            **base,
            "start_idempotency_key": "start-002",
            "replace_session_id": first["session_id"],
        }
        second = snapshot.start(second_body)
        self.assertNotEqual(second["session_id"], first["session_id"])
        self.assertEqual(snapshot.session_id, second["session_id"])
        with self.assertRaisesRegex(
            TeacherAgentDashboardError, "inactive session"
        ):
            snapshot.start(first_body)

    def test_concurrent_identical_online_start_calls_model_once(self) -> None:
        client = _FakeLiveClient()
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        body = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": "concurrent-live-start-001",
            "remote_processing_acknowledged": True,
        }
        barrier = threading.Barrier(3)
        results: list[dict[str, object]] = []
        errors: list[BaseException] = []

        def worker() -> None:
            barrier.wait()
            try:
                results.append(snapshot.start(body))
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        workers = [threading.Thread(target=worker) for _ in range(2)]
        for worker_thread in workers:
            worker_thread.start()
        barrier.wait()
        for worker_thread in workers:
            worker_thread.join(timeout=5)

        self.assertFalse(errors)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(client.chat_json_call_count, 1)

    def test_failed_online_start_does_not_reserve_key_or_replace_session(
        self,
    ) -> None:
        client = _FakeLiveClient(fail_first=True)
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
            live_options=LiveAgentOptions(fallback_to_rules=False),
        )
        body = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": "retry-after-start-failure-001",
            "remote_processing_acknowledged": True,
        }

        with self.assertRaisesRegex(Exception, "initial action"):
            snapshot.start(body)
        self.assertIsNone(snapshot.session)
        self.assertIsNone(snapshot.session_id)
        self.assertEqual(snapshot.start_idempotency_cache, {})

        recovered = snapshot.start(body)
        self.assertEqual(recovered["rounds_completed"], 0)
        self.assertEqual(client.chat_json_call_count, 2)
        self.assertEqual(snapshot.session_id, recovered["session_id"])

        client.chat_json_call_count = 0
        replacement = {
            **body,
            "start_idempotency_key": "retry-replacement-after-failure-002",
            "replace_session_id": recovered["session_id"],
        }
        with self.assertRaisesRegex(Exception, "initial action"):
            snapshot.start(replacement)
        self.assertEqual(snapshot.session_id, recovered["session_id"])
        self.assertEqual(snapshot.start(body), recovered)
        self.assertNotIn(
            "retry-replacement-after-failure-002",
            snapshot.start_idempotency_cache,
        )

        replaced = snapshot.start(replacement)
        self.assertNotEqual(replaced["session_id"], recovered["session_id"])
        self.assertEqual(client.chat_json_call_count, 2)

    def test_concurrent_distinct_starts_do_not_replace_first_session(self) -> None:
        snapshot = self._offline_snapshot()
        barrier = threading.Barrier(3)
        results: list[dict[str, object]] = []
        errors: list[BaseException] = []

        def worker(start_key: str) -> None:
            barrier.wait()
            try:
                results.append(
                    snapshot.start(
                        {
                            "goal": snapshot.demo_input["goal"],
                            "student_profile": snapshot.demo_input["student_profile"],
                            "start_idempotency_key": start_key,
                        }
                    )
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        workers = [
            threading.Thread(target=worker, args=(f"tab-{index}",))
            for index in range(2)
        ]
        for worker_thread in workers:
            worker_thread.start()
        barrier.wait()
        for worker_thread in workers:
            worker_thread.join(timeout=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("replace_session_id", str(errors[0]))
        self.assertEqual(snapshot.session_id, results[0]["session_id"])

    def test_start_cache_is_bounded_and_old_keys_cannot_restore_old_sessions(
        self,
    ) -> None:
        snapshot = self._offline_snapshot()
        first_body = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": "bounded-start-000",
        }
        first = snapshot.start(first_body)
        for index in range(1, 19):
            current_id = snapshot.session_id
            snapshot.start(
                {
                    "goal": snapshot.demo_input["goal"],
                    "student_profile": snapshot.demo_input["student_profile"],
                    "start_idempotency_key": f"bounded-start-{index:03d}",
                    "replace_session_id": current_id,
                }
            )
        self.assertLessEqual(len(snapshot.start_idempotency_cache), 16)
        self.assertNotEqual(snapshot.session_id, first["session_id"])
        with self.assertRaisesRegex(
            TeacherAgentDashboardError, "replace_session_id"
        ):
            snapshot.start(first_body)

    def test_step_requires_session_round_and_idempotency_key(self) -> None:
        snapshot = self._offline_snapshot()
        started = self._start_offline(snapshot)
        body = {
            "session_id": started["session_id"],
            "expected_round": 0,
            "idempotency_key": "turn-001",
            "learner_response": "状态只需要看前一个位置。",
            "signal": "misconception",
            "misconception_tag": "missing_transition",
        }

        for required_field in ("session_id", "expected_round", "idempotency_key"):
            invalid = dict(body)
            invalid.pop(required_field)
            with self.subTest(required_field=required_field):
                with self.assertRaisesRegex(
                    TeacherAgentDashboardError, required_field
                ):
                    snapshot.step(invalid)

        for invalid_round in (True, -1, 0.5, "0"):
            invalid = {**body, "expected_round": invalid_round}
            with self.subTest(invalid_round=invalid_round):
                with self.assertRaisesRegex(
                    TeacherAgentDashboardError, "expected_round"
                ):
                    snapshot.step(invalid)

        for invalid_key in ("", " padded", "padded "):
            invalid = {**body, "idempotency_key": invalid_key}
            with self.subTest(invalid_key=invalid_key):
                with self.assertRaisesRegex(
                    TeacherAgentDashboardError, "idempotency_key"
                ):
                    snapshot.step(invalid)

    def test_step_replay_is_cached_and_does_not_advance_twice(self) -> None:
        snapshot = self._offline_snapshot()
        started = self._start_offline(snapshot)
        body = {
            "session_id": started["session_id"],
            "expected_round": 0,
            "idempotency_key": "turn-001",
            "learner_response": "状态只需要看前一个位置。",
            "signal": "misconception",
            "misconception_tag": "missing_transition",
        }

        first = snapshot.step(body)
        replay = snapshot.step(body)
        self.assertEqual(replay, first)
        self.assertEqual(snapshot.session["round"], 1)
        self.assertEqual(len(snapshot.session["history"]), 1)

        with self.assertRaisesRegex(
            TeacherAgentDashboardError, "different request"
        ):
            snapshot.step({**body, "learner_response": "这是另一条回答。"})
        self.assertEqual(snapshot.session["round"], 1)

        with self.assertRaisesRegex(TeacherAgentDashboardError, "expected_round"):
            snapshot.step({**body, "idempotency_key": "turn-002"})
        self.assertEqual(snapshot.session["round"], 1)

        second = snapshot.step(
            {
                **body,
                "expected_round": 1,
                "idempotency_key": "turn-002",
                "learner_response": "我还需要同时看输入字符。",
                "signal": "partial",
                "misconception_tag": None,
            }
        )
        self.assertEqual(second["rounds_completed"], 2)
        self.assertEqual(len(second["history"]), 2)

    def test_concurrent_identical_step_is_applied_once(self) -> None:
        snapshot = self._offline_snapshot()
        started = self._start_offline(snapshot)
        body = {
            "session_id": started["session_id"],
            "expected_round": 0,
            "idempotency_key": "concurrent-turn-001",
            "learner_response": "我不确定状态转移需要什么信息。",
            "signal": "confused",
        }
        barrier = threading.Barrier(3)
        results: list[dict[str, object]] = []
        errors: list[BaseException] = []

        def worker() -> None:
            barrier.wait()
            try:
                results.append(snapshot.step(body))
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        workers = [threading.Thread(target=worker) for _ in range(2)]
        for worker_thread in workers:
            worker_thread.start()
        barrier.wait()
        for worker_thread in workers:
            worker_thread.join(timeout=5)

        self.assertFalse(errors)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(snapshot.session["round"], 1)
        self.assertEqual(len(snapshot.session["history"]), 1)

    def test_starting_new_session_clears_idempotency_cache(self) -> None:
        snapshot = self._offline_snapshot()
        first_started = self._start_offline(snapshot)
        reused_key = "safe-to-reuse-after-new-session"
        first = snapshot.step(
            {
                "session_id": first_started["session_id"],
                "expected_round": 0,
                "idempotency_key": reused_key,
                "learner_response": "第一次会话。",
                "signal": "confused",
            }
        )
        self.assertEqual(first["rounds_completed"], 1)

        second_started = self._start_offline(snapshot)
        self.assertNotEqual(second_started["session_id"], first_started["session_id"])
        second = snapshot.step(
            {
                "session_id": second_started["session_id"],
                "expected_round": 0,
                "idempotency_key": reused_key,
                "learner_response": "第二次会话。",
                "signal": "partial",
            }
        )
        self.assertEqual(second["rounds_completed"], 1)

    def test_command_requires_session_and_rejects_stale_round(self) -> None:
        snapshot = self._offline_snapshot()
        started = self._start_offline(snapshot)
        session_id = started["session_id"]

        with self.assertRaisesRegex(TeacherAgentDashboardError, "session_id"):
            snapshot.command({"command": "auto"})
        with self.assertRaisesRegex(TeacherAgentDashboardError, "session_id"):
            snapshot.command({"command": "auto", "session_id": "wrong"})
        with self.assertRaisesRegex(TeacherAgentDashboardError, "expected_round"):
            snapshot.command(
                {
                    "command": "auto",
                    "session_id": session_id,
                    "expected_round": 1,
                }
            )

        current = snapshot.command(
            {
                "command": "auto",
                "session_id": session_id,
                "expected_round": 0,
            }
        )
        self.assertEqual(current["rounds_completed"], 0)
        without_optional_round = snapshot.command(
            {"command": "auto", "session_id": session_id}
        )
        self.assertEqual(without_optional_round["rounds_completed"], 0)

    def test_online_start_requires_explicit_remote_processing_acknowledgement(
        self,
    ) -> None:
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=_FakeLiveClient(),
        )
        self.assertTrue(
            snapshot.bootstrap()["interaction_contract"][
                "remote_processing_acknowledgement_required"
            ]
        )
        start_body = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": "online-consent-start-001",
        }
        for acknowledgement in (None, False, 1, "true"):
            invalid = dict(start_body)
            if acknowledgement is not None:
                invalid["remote_processing_acknowledged"] = acknowledgement
            with self.subTest(acknowledgement=acknowledgement):
                with self.assertRaisesRegex(
                    TeacherAgentDashboardError,
                    "remote_processing_acknowledged=true",
                ):
                    snapshot.start(invalid)
                self.assertIsNone(snapshot.session)

        started = snapshot.start(
            {**start_body, "remote_processing_acknowledged": True}
        )
        self.assertEqual(started["rounds_completed"], 0)
        self.assertTrue(started["session_id"])

        offline = self._offline_snapshot()
        self.assertFalse(
            offline.bootstrap()["interaction_contract"][
                "remote_processing_acknowledgement_required"
            ]
        )
        self.assertEqual(self._start_offline(offline)["rounds_completed"], 0)

    def test_loopback_server_runs_real_start_and_step_policy(self) -> None:
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.library_path, self.input_path, self.cases_path
        )
        try:
            server, url = create_teacher_agent_dashboard_server(
                snapshot, capability_token="a" * 24
            )
        except PermissionError:
            self.skipTest("sandbox does not permit loopback socket binding")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        parsed = urlsplit(url)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        try:
            connection.request("GET", f"{parsed.path}api/bootstrap")
            response = connection.getresponse()
            bootstrap = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Cache-Control"), "no-store, max-age=0")
            self.assertIn("default-src 'none'", response.getheader("Content-Security-Policy"))
            self.assertTrue(bootstrap["interaction_contract"]["one_action_per_turn"])

            start_body = json.dumps(
                {
                    "goal": snapshot.demo_input["goal"],
                    "student_profile": snapshot.demo_input["student_profile"],
                    "start_idempotency_key": "loopback-start-001",
                },
                ensure_ascii=False,
            ).encode("utf-8")
            connection.request(
                "POST",
                f"{parsed.path}api/start",
                body=start_body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            started = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(started["rounds_completed"], 0)
            self.assertEqual(started["history"], [])
            self.assertEqual(
                started["next_action"]["primary_skill"]["skill_id"],
                "skill_diagnostic_questioning",
            )

            step_body = json.dumps(
                {
                    "session_id": started["session_id"],
                    "expected_round": 0,
                    "idempotency_key": "loopback-turn-001",
                    "learner_response": "状态只需要看前一个位置。",
                    "signal": "misconception",
                    "misconception_tag": "missing_transition",
                },
                ensure_ascii=False,
            ).encode("utf-8")
            connection.request(
                "POST",
                f"{parsed.path}api/step",
                body=step_body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            stepped = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(stepped["rounds_completed"], 1)
            self.assertEqual(len(stepped["history"]), 1)
            self.assertEqual(
                stepped["next_action"]["primary_skill"]["skill_id"],
                "skill_misconception_contrast",
            )
            self.assertEqual(
                stepped["student_state"]["misconceptions"][0]["status"],
                "active",
            )
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

    def test_server_rejects_invalid_capability_and_non_json_post(self) -> None:
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.library_path, self.input_path, self.cases_path
        )
        try:
            server, url = create_teacher_agent_dashboard_server(
                snapshot, capability_token="b" * 24
            )
        except PermissionError:
            self.skipTest("sandbox does not permit loopback socket binding")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        parsed = urlsplit(url)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        try:
            connection.request("GET", "/wrong/api/bootstrap")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 403)

            connection.request(
                "POST",
                f"{parsed.path}api/start",
                body=b"not-json",
                headers={"Content-Type": "text/plain"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 400)
            self.assertIn("Content-Type", payload["error"])
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
