from __future__ import annotations

import base64
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from html.parser import HTMLParser
import http.client
import io
import json
from pathlib import Path
import threading
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from teaching_skill_miner.cli import main
from teaching_skill_miner.deepseek_client import DeepSeekClientError
from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.teacher_agent_dashboard import (
    HTML_RESOURCE,
    SCRIPT_RESOURCE,
    STYLE_RESOURCE,
    TeacherAgentDashboardError,
    TeacherAgentDashboardSnapshot,
    _resource_bytes,
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
    teacher_agent_dashboard_self_check,
)
from teaching_skill_miner.teacher_agent_live import LiveAgentOptions


_SYNTHETIC_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _AgentDashboardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.screens: list[str] = []
        self.external_assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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


class _BlockingCallLiveClient(_FakeLiveClient):
    """Pause one deterministic model call so lock ordering can be asserted."""

    def __init__(self, *, block_call: int) -> None:
        super().__init__()
        self.block_call = block_call
        self.blocked_call_entered = threading.Event()
        self.release_blocked_call = threading.Event()

    def chat_json(
        self,
        messages: object,
        *,
        request_kind: str,
        require_remote_consent: bool = True,
    ) -> tuple[dict[str, object], dict[str, object]]:
        result = super().chat_json(
            messages,
            request_kind=request_kind,
            require_remote_consent=require_remote_consent,
        )
        if self.chat_json_call_count == self.block_call:
            self.blocked_call_entered.set()
            if not self.release_blocked_call.wait(timeout=5):
                raise AssertionError("timed out waiting to release blocked model call")
        return result


class _FailOnCallLiveClient(_FakeLiveClient):
    """Fail exactly one model call while allowing a same-key retry."""

    def __init__(self, *, fail_call: int) -> None:
        super().__init__()
        self.fail_call = fail_call

    def chat_json(
        self,
        messages: object,
        *,
        request_kind: str,
        require_remote_consent: bool = True,
    ) -> tuple[dict[str, object], dict[str, object]]:
        next_call = self.chat_json_call_count + 1
        if next_call == self.fail_call:
            self.chat_json_call_count = next_call
            raise DeepSeekClientError("simulated replacement fallback")
        return super().chat_json(
            messages,
            request_kind=request_kind,
            require_remote_consent=require_remote_consent,
        )


class _ValidManualTurnLiveClient(_FakeLiveClient):
    """Return a valid partial plan after the initial diagnostic action."""

    def chat_json(
        self,
        messages: object,
        *,
        request_kind: str,
        require_remote_consent: bool = True,
    ) -> tuple[dict[str, object], dict[str, object]]:
        plan, trace = super().chat_json(
            messages,
            request_kind=request_kind,
            require_remote_consent=require_remote_consent,
        )
        if self.chat_json_call_count == 1:
            return plan, trace
        plan["diagnosis"] = {
            "signal": "partial",
            "confidence": 0.8,
            "answer_alignment": "partially_aligned",
            "matched_concepts": ["状态"],
            "missing_concepts": ["转移依据"],
            "diagnosis_reason": "回答方向相关，但还缺少转移依据。",
            "evidence_excerpt": "",
            "misconception_tag": None,
            "misconception_description": "",
            "resolved_misconception_tags": [],
            "response_quality": "partial",
            "engagement_level": "medium",
            "needs_human_review": False,
        }
        plan["decision"] = {
            "primary_skill_id": "skill_concrete_example_bridge",
            "supporting_skill_ids": ["skill_minimal_hint"],
            "selection_reason": "用最小例子连接抽象概念。",
            "next_focus": "conceptual",
        }
        plan["teacher_action"] = {
            "type": "ask_with_example",
            "message": "请用一个两步的小例子说明状态如何变化。",
            "expected_signal": "学生能指出至少一个状态和一条转移依据。",
            "question_contract": {
                "answer_type": "example",
                "target_concepts": ["状态", "转移依据"],
                "accepted_aliases": [],
                "success_criteria": ["给出状态并说明转移依据"],
            },
        }
        return plan, trace


class _RecordingVisionExtractor:
    """Return valid synthetic evidence without reading or writing real media."""

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str, str]] = []

    def __call__(
        self,
        image_bytes: bytes,
        mime_type: str,
        *,
        display_name: str,
    ) -> dict[str, object]:
        self.calls.append((image_bytes, mime_type, display_name))
        return {
            "schema": "teaching_skill_miner.local_visual_evidence.v1",
            "source_modality": "image",
            "source_kind": "learner_answer_attachment",
            "display_name": display_name,
            "mime_type": mime_type,
            "byte_size": len(image_bytes),
            "content_sha256": sha256(image_bytes).hexdigest(),
            "engine": "synthetic_test_extractor",
            "status": "recognized",
            "recognized_text": "状态转移依赖当前状态与输入符号。",
            "confidence": 0.99,
            "confidence_semantics": (
                "engine_native_ocr_heuristic_not_formula_correctness"
            ),
            "formula_like_text_detected": False,
            "formula_accuracy_established": False,
            "extractor_fallback_used": False,
            "needs_student_confirmation": False,
            "raw_media_retained": False,
            "remote_media_sent": False,
            "remote_representation": "bounded_redacted_ocr_text_only",
        }


class _ConfirmationRequiredVisionExtractor(_RecordingVisionExtractor):
    """Synthetic OCR whose text is useful only after the student checks it."""

    def __call__(
        self,
        image_bytes: bytes,
        mime_type: str,
        *,
        display_name: str,
    ) -> dict[str, object]:
        evidence = super().__call__(
            image_bytes,
            mime_type,
            display_name=display_name,
        )
        evidence["recognized_text"] = "疑似识别文字，需要学生核对。"
        evidence["confidence"] = 0.43
        evidence["formula_like_text_detected"] = True
        evidence["needs_student_confirmation"] = True
        return evidence


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

    def _start_offline(
        self,
        snapshot,
        *,
        start_key: str | None = None,
        student_profile: dict | None = None,
        replace_session_id: str | None = None,
        profile_revision: str | None = None,
        profile_display_name: str | None = None,
    ):
        body = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": student_profile
            or snapshot.demo_input["student_profile"],
            "start_idempotency_key": start_key
            or f"test-start-{len(snapshot.start_idempotency_cache) + 1:03d}",
        }
        if replace_session_id is not None:
            body["replace_session_id"] = replace_session_id
            record = snapshot.sessions[replace_session_id]
            action = record.session.get("current_action", {})
            teacher_action = (
                action.get("teacher_action", {}) if isinstance(action, dict) else {}
            )
            body.update(
                {
                    "replace_expected_round": record.session["round"],
                    "replace_expected_question_id": (
                        teacher_action.get("question_id") or action.get("action_id")
                    ),
                    "replace_expected_context_version": record.context_version,
                    "replace_expected_profile_revision": record.profile_revision,
                }
            )
        if profile_revision is not None:
            body["profile_revision"] = profile_revision
        if profile_display_name is not None:
            body["profile_display_name"] = profile_display_name
        return snapshot.start(body)

    def _replacement_guards(self, session: dict) -> dict[str, object]:
        return {
            "replace_expected_round": session["rounds_completed"],
            "replace_expected_question_id": session["expected_question_id"],
            "replace_expected_context_version": session["context_version"],
            "replace_expected_profile_revision": session["profile_summary"][
                "profile_revision"
            ],
        }

    def _command_body(
        self,
        session: dict,
        command: str,
        *,
        key: str,
        skill_id: str | None = None,
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "session_id": session["session_id"],
            "command": command,
            "command_idempotency_key": key,
            "expected_round": session["rounds_completed"],
            "expected_question_id": session["expected_question_id"],
            "expected_context_version": session["context_version"],
            "profile_revision": session["profile_summary"]["profile_revision"],
        }
        if skill_id is not None:
            body["skill_id"] = skill_id
        return body

    def _attachment_body(
        self,
        session: dict,
        *,
        key: str,
        display_name: str = "synthetic-answer.png",
    ) -> dict[str, object]:
        return {
            "session_id": session["session_id"],
            "expected_round": session["rounds_completed"],
            "expected_question_id": session["expected_question_id"],
            "expected_context_version": session["context_version"],
            "profile_revision": session["profile_summary"]["profile_revision"],
            "attachment_idempotency_key": key,
            "mime_type": "image/png",
            "display_name": display_name,
            "data_base64": base64.b64encode(_SYNTHETIC_ONE_PIXEL_PNG).decode("ascii"),
        }

    def test_generic_frontend_has_four_clear_screens_and_no_external_assets(
        self,
    ) -> None:
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
        self.assertIn('id="liveSelectionReason"', html)
        self.assertIn('id="studentStateTimeline"', html)
        self.assertIn("renderContextMemory(session)", script)
        self.assertIn("const replacingCurrentSession = Boolean(app.session?.session_id)", script)
        self.assertNotIn(
            'const replacingActiveSession = app.session?.status === "active"',
            script,
        )
        self.assertIn("renderStudentStateTimeline(session)", script)
        self.assertIn('select("#liveSelectionReason").hidden = false', script)
        self.assertIn("snapshot.operation", script)
        self.assertIn("snapshot.session_round_before_request", script)
        self.assertIn("本区不会混入这些处理后结果", script)
        self.assertNotIn("const fallbackRecent = history.slice(-6)", script)
        self.assertNotIn("session.student_state?.misconceptions", script)
        self.assertNotIn("session.goal_plan?.intermediate_objectives", script)
        self.assertIn("确定性统计与原文抽取检查点", script)
        self.assertIn("startPayload.start_idempotency_key = startKey", script)
        self.assertIn("startPayload.replace_session_id", script)
        self.assertIn('postJson("api/session"', script)
        self.assertIn("window.sessionStorage.setItem", script)
        self.assertIn("persistSessionHandle()", script)
        self.assertIn("payload.idempotency_key = idempotencyKey", script)
        self.assertIn("expected_round: round", script)
        self.assertIn("expected_question_id: app.session.expected_question_id", script)
        self.assertIn(
            "expected_context_version: finite(app.session.context_version", script
        )
        self.assertIn("profile_revision:", script)
        self.assertIn("function turnRequestFingerprint(payload)", script)
        self.assertIn("manual_skill_id: payload.manual_skill_id ?? null", script)
        self.assertIn("signal: payload.signal ?? null", script)
        self.assertIn("app.pendingTurn?.fingerprint === requestFingerprint", script)
        self.assertIn(
            'select("#fallbackSignalInput").addEventListener("change", clearPendingTurn)',
            script,
        )
        self.assertIn('setAppView("learning")', script)
        self.assertIn("showSetupForm(false)", script)
        self.assertNotIn("scrollIntoView", script)
        context_renderer = script.split("function renderContextMemory(session) {", 1)[
            1
        ].split("function masteryDeltaNodes", 1)[0]
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
            "expected_question_id",
            "expected_context_version",
            "profile_revision",
        ):
            with self.subTest(logical_field=logical_field):
                self.assertIn(
                    f"{logical_field}: payload.{logical_field}", fingerprint_builder
                )
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

    def test_v2_bootstrap_exposes_live_model_evidence_without_overclaiming(
        self,
    ) -> None:
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
        self.assertEqual(receipt["online_deepseek"]["model"], "deepseek-v4-flash")
        self.assertEqual(
            receipt["online_deepseek"]["metrics"]["signal"]["end_to_end_all_attempts"][
                "macro_f1"
            ],
            0.875325,
        )
        self.assertFalse(receipt["claim_boundary"]["expert_validated"])
        self.assertFalse(receipt["claim_boundary"]["deployment_accuracy_established"])
        self.assertEqual(outcome["metrics"]["absolute_gain"], 0.4)
        self.assertFalse(
            outcome["claim_boundary"]["real_learner_effectiveness_established"]
        )
        self.assertFalse(bootstrap["neural_v1"]["materialization_gate"]["passed"])
        self.assertFalse(
            bootstrap["interaction_contract"][
                "adaptive_profile_candidates_are_teacher_confirmed"
            ]
        )

    def test_start_is_idempotent_and_registers_independent_sessions(self) -> None:
        snapshot = self._offline_snapshot()
        contract = snapshot.bootstrap()["interaction_contract"]
        self.assertTrue(contract["start_requires_idempotency_key"])
        self.assertTrue(contract["independent_session_registry"])
        self.assertFalse(contract["active_session_replacement_requires_session_id"])
        self.assertTrue(contract["explicit_replacement_is_transactional"])
        self.assertTrue(contract["validated_safety_fallback_can_commit_replacement"])
        self.assertTrue(contract["session_resume_supported"])
        self.assertTrue(contract["replacement_requires_expected_round"])
        self.assertTrue(contract["replacement_requires_question_id"])
        self.assertTrue(contract["replacement_requires_context_version"])
        self.assertTrue(contract["replacement_requires_profile_revision"])
        self.assertTrue(
            contract["manual_skill_lock_effective_from_next_learner_response"]
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

        with self.assertRaisesRegex(TeacherAgentDashboardError, "different request"):
            snapshot.start(
                {
                    **first_body,
                    "goal": {
                        **snapshot.demo_input["goal"],
                        "objective": "不能复用同一启动键的新目标",
                    },
                }
            )

        second = snapshot.start({**base, "start_idempotency_key": "start-002"})
        self.assertNotEqual(second["session_id"], first["session_id"])
        self.assertEqual(snapshot.session_id, second["session_id"])
        self.assertEqual(
            set(snapshot.sessions), {first["session_id"], second["session_id"]}
        )
        self.assertEqual(snapshot.resume({"session_id": first["session_id"]}), first)
        self.assertEqual(snapshot.resume({"session_id": second["session_id"]}), second)

        with self.assertRaisesRegex(TeacherAgentDashboardError, "replace_session_id"):
            snapshot.start(
                {
                    **base,
                    "start_idempotency_key": "start-003",
                    "replace_session_id": "another-tab-without-current-session",
                    "replace_expected_round": 0,
                    "replace_expected_question_id": "unknown-question",
                    "replace_expected_context_version": 1,
                    "replace_expected_profile_revision": "unknown-profile",
                }
            )

    def test_setup_snapshot_round_trips_only_teacher_setup_fields(self) -> None:
        snapshot = self._offline_snapshot()
        goal = deepcopy(snapshot.demo_input["goal"])
        goal["concept"] = "自定义状态机"
        goal["objective"] = "能够解释状态、事件与转移条件。"
        goal["max_rounds"] = 9
        goal["materials"] = {
            "example": "门锁状态机",
            "practice": "写出开门事件的转移",
            "transfer_task": "迁移到交通灯控制",
        }
        goal["success_thresholds"] = {
            "prerequisite": 0.51,
            "conceptual": 0.62,
            "procedural": 0.73,
            "transfer": 0.44,
        }
        profile = deepcopy(snapshot.demo_input["student_profile"])
        profile.update(
            {
                "profile_ref": "setup_snapshot_student",
                "learner_level": "intermediate",
                "preferences": ["先例子后定义", "一次一步"],
                "initial_mastery": {
                    "prerequisite": 0.31,
                    "conceptual": 0.27,
                    "procedural": 0.22,
                    "transfer": 0.18,
                },
                "known_misconceptions": [
                    {
                        "tag": "event_equals_state",
                        "description": "把事件误当成状态",
                        "confidence": 0.8,
                    }
                ],
                "conversation_history": [
                    {
                        "response": "我能区分状态和事件，但还不会写转移。",
                        "signal": "partial",
                        "focus_dimension": "conceptual",
                    }
                ],
                "background_history": ["学过有限集合", "没有画过状态图"],
                "accessibility_needs": ["短句", "一次一个问题"],
                "contains_direct_identity": False,
            }
        )
        started = snapshot.start(
            {
                "goal": goal,
                "student_profile": profile,
                "profile_revision": "setup-snapshot-v7",
                "profile_display_name": "快照学生",
                "start_idempotency_key": "setup-snapshot-start-001",
            }
        )

        self.assertEqual(started["setup_snapshot"]["goal"], started["goal"])
        self.assertEqual(
            started["setup_snapshot"]["goal"]["knowledge_components"],
            goal["knowledge_components"],
        )
        self.assertEqual(
            started["setup_snapshot"]["goal"]["knowledge_spec"]["status"],
            "teacher_provided",
        )
        self.assertEqual(
            started["setup_snapshot"]["student_profile"],
            {
                "profile_ref": profile["profile_ref"],
                "learner_level": profile["learner_level"],
                "preferences": profile["preferences"],
                "initial_mastery": profile["initial_mastery"],
                "known_misconceptions": profile["known_misconceptions"],
                "background_history": profile["background_history"],
                "conversation_history": profile["conversation_history"],
                "accessibility_needs": profile["accessibility_needs"],
                "contains_direct_identity": False,
            },
        )
        snapshot_keys = set(started["setup_snapshot"])
        snapshot_keys.update(started["setup_snapshot"]["goal"])
        snapshot_keys.update(started["setup_snapshot"]["student_profile"])
        for forbidden in (
            "adaptive_observations",
            "agent_runtime",
            "current_action",
            "adaptive_summary",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, snapshot_keys)

    def test_stale_replacement_guards_fail_before_model_or_state_mutation(self) -> None:
        client = _FakeLiveClient()
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        original = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "stale-replacement-original",
                "remote_processing_acknowledged": True,
            }
        )
        base = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "replace_session_id": original["session_id"],
            **self._replacement_guards(original),
            "remote_processing_acknowledged": True,
        }
        stale_values = (
            ("replace_expected_round", original["rounds_completed"] + 1),
            ("replace_expected_question_id", "stale-question"),
            (
                "replace_expected_context_version",
                original["context_version"] + 1,
            ),
            ("replace_expected_profile_revision", "stale-profile"),
        )
        for index, (field, value) in enumerate(stale_values):
            with self.subTest(field=field):
                before = snapshot.resume({"session_id": original["session_id"]})
                calls_before = client.chat_json_call_count
                with self.assertRaisesRegex(TeacherAgentDashboardError, field):
                    snapshot.start(
                        {
                            **base,
                            field: value,
                            "start_idempotency_key": f"stale-replacement-{index}",
                        }
                    )
                self.assertEqual(client.chat_json_call_count, calls_before)
                self.assertEqual(
                    snapshot.resume({"session_id": original["session_id"]}),
                    before,
                )

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
        replacement_profile = deepcopy(snapshot.demo_input["student_profile"])
        replacement_profile["profile_ref"] = "advanced_replacement"
        replacement_profile["learner_level"] = "advanced"
        replacement_profile["initial_mastery"] = {
            "prerequisite": 0.85,
            "conceptual": 0.75,
            "procedural": 0.7,
            "transfer": 0.6,
        }
        replacement = {
            **body,
            "student_profile": replacement_profile,
            "start_idempotency_key": "retry-replacement-after-failure-002",
            "replace_session_id": recovered["session_id"],
            **self._replacement_guards(recovered),
            "profile_revision": "advanced-v2",
            "profile_display_name": "进阶学生",
        }
        with self.assertRaisesRegex(Exception, "initial action"):
            snapshot.start(replacement)
        self.assertEqual(snapshot.session_id, recovered["session_id"])
        self.assertEqual(
            snapshot.resume({"session_id": recovered["session_id"]}), recovered
        )
        self.assertEqual(snapshot.start(body), recovered)
        self.assertNotIn(
            "retry-replacement-after-failure-002",
            snapshot.start_idempotency_cache,
        )

        replaced = snapshot.start(replacement)
        self.assertNotEqual(replaced["session_id"], recovered["session_id"])
        self.assertEqual(
            replaced["profile_summary"]["profile_ref"], "advanced_replacement"
        )
        self.assertEqual(replaced["profile_summary"]["profile_revision"], "advanced-v2")
        self.assertEqual(replaced["profile_summary"]["display_name"], "进阶学生")
        self.assertEqual(
            replaced["student_state"]["knowledge_mastery"],
            replacement_profile["initial_mastery"],
        )
        with self.assertRaisesRegex(TeacherAgentDashboardError, "no longer available"):
            snapshot.resume({"session_id": recovered["session_id"]})
        self.assertEqual(
            snapshot.resume({"session_id": replaced["session_id"]}), replaced
        )
        self.assertEqual(client.chat_json_call_count, 2)

    def test_deepseek_fallback_can_commit_a_valid_replacement_session(self) -> None:
        client = _FailOnCallLiveClient(fail_call=2)
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        original_body = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": "fallback-preservation-original",
            "remote_processing_acknowledged": True,
        }
        original = snapshot.start(original_body)
        replacement_profile = deepcopy(snapshot.demo_input["student_profile"])
        replacement_profile["profile_ref"] = "fallback_replacement_candidate"
        replacement_profile["initial_mastery"] = {
            "prerequisite": 0.9,
            "conceptual": 0.8,
            "procedural": 0.7,
            "transfer": 0.6,
        }
        replacement_body = {
            **original_body,
            "student_profile": replacement_profile,
            "start_idempotency_key": "fallback-preservation-replacement",
            "replace_session_id": original["session_id"],
            **self._replacement_guards(original),
            "profile_revision": "fallback-candidate-v2",
        }

        replacement = snapshot.start(replacement_body)
        self.assertEqual(client.chat_json_call_count, 2)
        self.assertNotEqual(replacement["session_id"], original["session_id"])
        self.assertEqual(set(snapshot.sessions), {replacement["session_id"]})
        self.assertEqual(snapshot.session_id, replacement["session_id"])
        self.assertEqual(replacement["agent_runtime"]["fallback_count"], 1)
        self.assertEqual(replacement["rounds_completed"], 0)
        self.assertEqual(replacement["history"], [])
        self.assertIn(
            replacement_body["start_idempotency_key"],
            snapshot.start_idempotency_cache,
        )
        self.assertEqual(snapshot.start(replacement_body), replacement)
        self.assertEqual(client.chat_json_call_count, 2)
        with self.assertRaisesRegex(TeacherAgentDashboardError, "no longer available"):
            snapshot.resume({"session_id": original["session_id"]})

    def test_replacement_preempts_inflight_step_and_late_response_cannot_commit(
        self,
    ) -> None:
        client = _BlockingCallLiveClient(block_call=2)
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        original = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "step-first-original",
                "remote_processing_acknowledged": True,
            }
        )
        step_body = {
            "session_id": original["session_id"],
            "expected_round": 0,
            "expected_question_id": original["expected_question_id"],
            "expected_context_version": original["context_version"],
            "profile_revision": original["profile_summary"]["profile_revision"],
            "idempotency_key": "step-first-turn",
            "learner_response": "先完成这一轮，再切换画像。",
        }
        replacement_profile = deepcopy(snapshot.demo_input["student_profile"])
        replacement_profile["profile_ref"] = "step_first_replacement"
        replacement_body = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": replacement_profile,
            "start_idempotency_key": "step-first-replacement",
            "replace_session_id": original["session_id"],
            **self._replacement_guards(original),
            "remote_processing_acknowledged": True,
        }
        step_results: list[dict[str, object]] = []
        replacement_results: list[dict[str, object]] = []
        errors: list[BaseException] = []
        replacement_attempted = threading.Event()
        replacement_done = threading.Event()

        def run_step() -> None:
            try:
                step_results.append(snapshot.step(step_body))
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        def run_replacement() -> None:
            replacement_attempted.set()
            try:
                replacement_results.append(snapshot.start(replacement_body))
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)
            finally:
                replacement_done.set()

        step_thread = threading.Thread(target=run_step)
        step_thread.start()
        self.assertTrue(client.blocked_call_entered.wait(timeout=2))
        replacement_thread = threading.Thread(target=run_replacement)
        replacement_thread.start()
        self.assertTrue(replacement_attempted.wait(timeout=2))
        self.assertTrue(replacement_done.wait(timeout=2))
        self.assertEqual(len(replacement_results), 1)
        replacement = replacement_results[0]
        self.assertNotEqual(replacement["session_id"], original["session_id"])
        self.assertEqual(snapshot.session_id, replacement["session_id"])
        self.assertNotIn(original["session_id"], snapshot.sessions)
        self.assertTrue(step_thread.is_alive())

        client.release_blocked_call.set()
        step_thread.join(timeout=5)
        replacement_thread.join(timeout=5)

        self.assertFalse(step_thread.is_alive())
        self.assertFalse(replacement_thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], TeacherAgentDashboardError)
        self.assertIn("cancelled before commit", str(errors[0]))
        self.assertEqual(step_results, [])
        self.assertEqual(set(snapshot.sessions), {replacement["session_id"]})
        self.assertEqual(
            snapshot.resume({"session_id": replacement["session_id"]}),
            replacement,
        )
        self.assertEqual(client.chat_json_call_count, 3)

    def test_active_turn_rejects_duplicate_and_competing_step_without_extra_call(
        self,
    ) -> None:
        client = _BlockingCallLiveClient(block_call=2)
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        started = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "active-turn-start",
                "remote_processing_acknowledged": True,
            }
        )
        body = {
            "session_id": started["session_id"],
            "expected_round": 0,
            "expected_question_id": started["expected_question_id"],
            "expected_context_version": started["context_version"],
            "profile_revision": started["profile_summary"]["profile_revision"],
            "idempotency_key": "active-turn-one",
            "learner_response": "我先提交这一条回答。",
        }
        first_results: list[dict[str, object]] = []
        first_errors: list[BaseException] = []

        def run_first() -> None:
            try:
                first_results.append(snapshot.step(body))
            except BaseException as exc:  # pragma: no cover - asserted below
                first_errors.append(exc)

        first_thread = threading.Thread(target=run_first)
        first_thread.start()
        self.assertTrue(client.blocked_call_entered.wait(timeout=2))

        with self.assertRaisesRegex(
            TeacherAgentDashboardError, "idempotent turn is still running"
        ):
            snapshot.step(body)
        with self.assertRaisesRegex(
            TeacherAgentDashboardError, "another turn is already running"
        ):
            snapshot.step(
                {
                    **body,
                    "idempotency_key": "active-turn-two",
                    "learner_response": "竞争提交不应调用模型。",
                }
            )
        self.assertEqual(client.chat_json_call_count, 2)

        client.release_blocked_call.set()
        first_thread.join(timeout=5)
        self.assertFalse(first_thread.is_alive())
        self.assertFalse(first_errors)
        self.assertEqual(len(first_results), 1)
        self.assertEqual(snapshot.step(body), first_results[0])
        self.assertEqual(client.chat_json_call_count, 2)

    def test_stop_preempts_inflight_step_and_fences_late_model_response(self) -> None:
        client = _BlockingCallLiveClient(block_call=2)
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        started = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "stop-preempt-start",
                "remote_processing_acknowledged": True,
            }
        )
        step_body = {
            "session_id": started["session_id"],
            "expected_round": 0,
            "expected_question_id": started["expected_question_id"],
            "expected_context_version": started["context_version"],
            "profile_revision": started["profile_summary"]["profile_revision"],
            "idempotency_key": "stop-preempt-turn",
            "learner_response": "这条模型响应会晚到。",
        }
        step_results: list[dict[str, object]] = []
        step_errors: list[BaseException] = []

        def run_step() -> None:
            try:
                step_results.append(snapshot.step(step_body))
            except BaseException as exc:  # pragma: no cover - asserted below
                step_errors.append(exc)

        step_thread = threading.Thread(target=run_step)
        step_thread.start()
        self.assertTrue(client.blocked_call_entered.wait(timeout=2))
        running = snapshot.resume({"session_id": started["session_id"]})
        self.assertTrue(running["turn_runtime"]["active"])

        stopped = snapshot.command(
            {
                "session_id": started["session_id"],
                "command": "stop",
                "command_idempotency_key": "stop-preempt-command",
                "expected_round": 0,
                "expected_question_id": started["expected_question_id"],
                "expected_context_version": started["context_version"],
                "profile_revision": started["profile_summary"]["profile_revision"],
            }
        )
        self.assertNotEqual(stopped["status"], "active")
        self.assertEqual(stopped["rounds_completed"], 0)
        self.assertTrue(stopped["turn_runtime"]["cancellation_requested"])
        self.assertEqual(
            stopped["turn_runtime"]["cancel_reason"], "teacher_requested_stop"
        )
        self.assertTrue(step_thread.is_alive())

        client.release_blocked_call.set()
        step_thread.join(timeout=5)
        self.assertFalse(step_thread.is_alive())
        self.assertEqual(step_results, [])
        self.assertEqual(len(step_errors), 1)
        self.assertIsInstance(step_errors[0], TeacherAgentDashboardError)
        self.assertIn("cancelled before commit", str(step_errors[0]))

        resumed = snapshot.resume({"session_id": started["session_id"]})
        self.assertEqual(resumed["rounds_completed"], 0)
        self.assertEqual(resumed["history"], [])
        self.assertFalse(resumed["turn_runtime"]["active"])
        self.assertEqual(client.chat_json_call_count, 2)

    def test_cancel_turn_preempts_inflight_step_but_keeps_session_active(self) -> None:
        client = _BlockingCallLiveClient(block_call=2)
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        started = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "cancel-turn-start",
                "remote_processing_acknowledged": True,
            }
        )
        step_body = {
            "session_id": started["session_id"],
            "expected_round": 0,
            "expected_question_id": started["expected_question_id"],
            "expected_context_version": started["context_version"],
            "profile_revision": started["profile_summary"]["profile_revision"],
            "idempotency_key": "cancel-turn-step",
            "learner_response": "这条响应必须被取消而不是结束会话。",
        }
        errors: list[BaseException] = []

        def run_step() -> None:
            try:
                snapshot.step(step_body)
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        step_thread = threading.Thread(target=run_step)
        step_thread.start()
        self.assertTrue(client.blocked_call_entered.wait(timeout=2))
        cancelled = snapshot.command(
            {
                "session_id": started["session_id"],
                "command": "cancel_turn",
                "command_idempotency_key": "cancel-turn-command",
                "expected_round": 0,
                "expected_question_id": started["expected_question_id"],
                "expected_context_version": started["context_version"],
                "profile_revision": started["profile_summary"]["profile_revision"],
            }
        )
        self.assertEqual(cancelled["status"], "active")
        self.assertEqual(cancelled["rounds_completed"], 0)
        self.assertTrue(cancelled["turn_runtime"]["cancellation_requested"])
        self.assertEqual(
            cancelled["turn_runtime"]["cancel_reason"],
            "teacher_requested_turn_cancel",
        )

        client.release_blocked_call.set()
        step_thread.join(timeout=5)
        self.assertFalse(step_thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("cancelled before commit", str(errors[0]))

        resumed = snapshot.resume({"session_id": started["session_id"]})
        self.assertEqual(resumed["status"], "active")
        self.assertEqual(resumed["rounds_completed"], 0)
        self.assertEqual(resumed["history"], [])
        self.assertFalse(resumed["turn_runtime"]["active"])

    def test_preempted_turn_persists_one_aborted_terminal_event(self) -> None:
        with TemporaryDirectory() as directory:
            client = _BlockingCallLiveClient(block_call=2)
            store_path = Path(directory) / "active-turn-preemption.jsonl"
            snapshot = build_teacher_agent_dashboard_snapshot(
                self.v2_library_path,
                self.input_path,
                self.cases_path,
                client=client,
                store_path=store_path,
            )
            started = snapshot.start(
                {
                    "goal": snapshot.demo_input["goal"],
                    "student_profile": snapshot.demo_input["student_profile"],
                    "start_idempotency_key": "stored-preempt-start",
                    "remote_processing_acknowledged": True,
                }
            )
            step_body = {
                "session_id": started["session_id"],
                "expected_round": 0,
                "expected_question_id": started["expected_question_id"],
                "expected_context_version": started["context_version"],
                "profile_revision": started["profile_summary"]["profile_revision"],
                "idempotency_key": "stored-preempt-turn",
                "learner_response": "这条响应必须被终止事件封口。",
            }
            errors: list[BaseException] = []

            def run_step() -> None:
                try:
                    snapshot.step(step_body)
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            step_thread = threading.Thread(target=run_step)
            step_thread.start()
            self.assertTrue(client.blocked_call_entered.wait(timeout=2))
            active = snapshot.resume({"session_id": started["session_id"]})
            turn_id = active["turn_runtime"]["turn_id"]
            snapshot.command(
                {
                    "session_id": started["session_id"],
                    "command": "stop",
                    "command_idempotency_key": "stored-preempt-stop",
                    "expected_round": 0,
                    "expected_question_id": started["expected_question_id"],
                    "expected_context_version": started["context_version"],
                    "profile_revision": started["profile_summary"][
                        "profile_revision"
                    ],
                }
            )
            client.release_blocked_call.set()
            step_thread.join(timeout=5)
            self.assertFalse(step_thread.is_alive())
            self.assertEqual(len(errors), 1)

            turn_events = [
                event
                for event in snapshot.store.events
                if event.get("turn_id") == turn_id
            ]
            self.assertEqual(
                [event["event_type"] for event in turn_events],
                ["turn_started", "turn_aborted"],
            )
            self.assertEqual(
                turn_events[-1]["data"]["reason"], "teacher_requested_stop"
            )
            self.assertFalse(
                any(event["event_type"] == "turn_committed" for event in turn_events)
            )

            restarted = build_teacher_agent_dashboard_snapshot(
                self.v2_library_path,
                self.input_path,
                self.cases_path,
                client=client,
                store_path=store_path,
            )
            recovered = restarted.resume({"session_id": started["session_id"]})
            self.assertEqual(recovered["rounds_completed"], 0)
            self.assertFalse(recovered["turn_runtime"]["active"])

    def test_step_waiting_behind_replacement_is_rejected_without_model_call(
        self,
    ) -> None:
        client = _BlockingCallLiveClient(block_call=2)
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        original = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "replacement-first-original",
                "remote_processing_acknowledged": True,
            }
        )
        replacement_profile = deepcopy(snapshot.demo_input["student_profile"])
        replacement_profile["profile_ref"] = "replacement_first_candidate"
        replacement_body = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": replacement_profile,
            "start_idempotency_key": "replacement-first-new",
            "replace_session_id": original["session_id"],
            **self._replacement_guards(original),
            "remote_processing_acknowledged": True,
        }
        step_body = {
            "session_id": original["session_id"],
            "expected_round": 0,
            "expected_question_id": original["expected_question_id"],
            "expected_context_version": original["context_version"],
            "profile_revision": original["profile_summary"]["profile_revision"],
            "idempotency_key": "replacement-first-stale-turn",
            "learner_response": "这条回答不应写入已经替换的会话。",
        }
        replacement_results: list[dict[str, object]] = []
        replacement_errors: list[BaseException] = []
        step_errors: list[BaseException] = []
        step_attempted = threading.Event()
        step_done = threading.Event()

        def run_replacement() -> None:
            try:
                replacement_results.append(snapshot.start(replacement_body))
            except BaseException as exc:  # pragma: no cover - asserted below
                replacement_errors.append(exc)

        def run_step() -> None:
            step_attempted.set()
            try:
                snapshot.step(step_body)
            except BaseException as exc:  # pragma: no cover - asserted below
                step_errors.append(exc)
            finally:
                step_done.set()

        replacement_thread = threading.Thread(target=run_replacement)
        replacement_thread.start()
        self.assertTrue(client.blocked_call_entered.wait(timeout=2))
        step_thread = threading.Thread(target=run_step)
        step_thread.start()
        self.assertTrue(step_attempted.wait(timeout=2))
        self.assertTrue(step_done.wait(timeout=2))
        self.assertEqual(len(step_errors), 1)
        self.assertIsInstance(step_errors[0], TeacherAgentDashboardError)
        self.assertIn("replacement is in progress", str(step_errors[0]))
        self.assertEqual(client.chat_json_call_count, 2)

        client.release_blocked_call.set()
        replacement_thread.join(timeout=5)
        step_thread.join(timeout=5)

        self.assertFalse(replacement_thread.is_alive())
        self.assertFalse(step_thread.is_alive())
        self.assertFalse(replacement_errors)
        self.assertEqual(len(replacement_results), 1)
        self.assertEqual(len(step_errors), 1)
        replacement = replacement_results[0]
        self.assertEqual(set(snapshot.sessions), {replacement["session_id"]})
        self.assertEqual(
            snapshot.resume({"session_id": replacement["session_id"]}), replacement
        )
        self.assertEqual(client.chat_json_call_count, 2)

    def test_start_rejects_non_list_allowed_skill_ids_before_model_call(self) -> None:
        client = _FakeLiveClient()
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        base = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": snapshot.demo_input["student_profile"],
            "start_idempotency_key": "allowed-skills-type-check",
            "remote_processing_acknowledged": True,
        }
        invalid_values = (
            None,
            "skill_diagnostic_questioning",
            ("skill_diagnostic_questioning",),
            {"skill_diagnostic_questioning"},
            {"primary": "skill_diagnostic_questioning"},
            True,
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    TeacherAgentDashboardError, "allowed_skill_ids must be a list"
                ):
                    snapshot.start({**base, "allowed_skill_ids": value})
                self.assertEqual(client.chat_json_call_count, 0)
                self.assertEqual(snapshot.sessions, {})
                self.assertEqual(snapshot.start_idempotency_cache, {})

        valid = snapshot.start(
            {
                **base,
                "allowed_skill_ids": [
                    item["skill_id"] for item in snapshot.library["skills"]
                ],
            }
        )
        self.assertEqual(valid["rounds_completed"], 0)
        self.assertEqual(client.chat_json_call_count, 1)

    def test_concurrent_distinct_starts_create_isolated_sessions(self) -> None:
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

        self.assertFalse(errors)
        self.assertEqual(len(results), 2)
        session_ids = {str(item["session_id"]) for item in results}
        self.assertEqual(len(session_ids), 2)
        self.assertEqual(session_ids, set(snapshot.sessions))
        for item in results:
            self.assertEqual(snapshot.resume({"session_id": item["session_id"]}), item)

    def test_two_sessions_keep_round_state_and_idempotency_isolated(self) -> None:
        snapshot = self._offline_snapshot()
        first = self._start_offline(snapshot, start_key="isolated-start-a")
        second_profile = deepcopy(snapshot.demo_input["student_profile"])
        second_profile["profile_ref"] = "isolated_student_b"
        second_profile["learner_level"] = "intermediate"
        second = self._start_offline(
            snapshot,
            start_key="isolated-start-b",
            student_profile=second_profile,
            profile_revision="student-b-v1",
            profile_display_name="学生 B",
        )
        shared_turn_key = "same-key-is-safe-across-sessions"

        first_advanced = snapshot.step(
            {
                "session_id": first["session_id"],
                "expected_round": 0,
                "expected_question_id": first["expected_question_id"],
                "expected_context_version": first["context_version"],
                "profile_revision": first["profile_summary"]["profile_revision"],
                "idempotency_key": shared_turn_key,
                "learner_response": "第一位学生仍在理解状态定义。",
                "signal": "partial",
            }
        )
        self.assertEqual(first_advanced["rounds_completed"], 1)
        self.assertEqual(snapshot.resume({"session_id": second["session_id"]}), second)

        second_advanced = snapshot.step(
            {
                "session_id": second["session_id"],
                "expected_round": 0,
                "expected_question_id": second["expected_question_id"],
                "expected_context_version": second["context_version"],
                "profile_revision": "student-b-v1",
                "idempotency_key": shared_turn_key,
                "learner_response": "第二位学生可以说出状态转移。",
                "signal": "correct",
            }
        )
        self.assertEqual(second_advanced["rounds_completed"], 1)
        self.assertEqual(
            snapshot.resume({"session_id": first["session_id"]}), first_advanced
        )
        self.assertNotEqual(
            first_advanced["profile_summary"]["profile_ref"],
            second_advanced["profile_summary"]["profile_ref"],
        )

    def test_resume_returns_latest_session_view_after_page_refresh(self) -> None:
        snapshot = self._offline_snapshot()
        started = self._start_offline(snapshot, start_key="resume-start-001")
        stepped = snapshot.step(
            {
                "session_id": started["session_id"],
                "expected_round": 0,
                "expected_question_id": started["expected_question_id"],
                "expected_context_version": started["context_version"],
                "profile_revision": started["profile_summary"]["profile_revision"],
                "idempotency_key": "resume-turn-001",
                "learner_response": "页面刷新后这一轮仍应存在。",
                "signal": "partial",
            }
        )

        resumed = snapshot.resume({"session_id": started["session_id"]})
        self.assertEqual(resumed, stepped)
        self.assertEqual(resumed["rounds_completed"], 1)
        self.assertEqual(len(resumed["history"]), 1)
        self.assertEqual(resumed["context_version"], 2)
        with self.assertRaisesRegex(TeacherAgentDashboardError, "no longer available"):
            snapshot.resume({"session_id": "unknown-session-handle"})

    def test_start_cache_and_independent_session_registry_are_bounded(
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
            snapshot.start(
                {
                    "goal": snapshot.demo_input["goal"],
                    "student_profile": snapshot.demo_input["student_profile"],
                    "start_idempotency_key": f"bounded-start-{index:03d}",
                }
            )
        self.assertLessEqual(len(snapshot.start_idempotency_cache), 16)
        self.assertLessEqual(len(snapshot.sessions), 16)
        self.assertNotEqual(snapshot.session_id, first["session_id"])
        with self.assertRaisesRegex(TeacherAgentDashboardError, "no longer available"):
            snapshot.resume({"session_id": first["session_id"]})
        recreated = snapshot.start(first_body)
        self.assertNotEqual(recreated["session_id"], first["session_id"])

    def test_attachment_is_bound_idempotent_consumed_and_never_retains_raw_media(
        self,
    ) -> None:
        extractor = _RecordingVisionExtractor()
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.library_path,
            self.input_path,
            self.cases_path,
            vision_extractor=extractor,
        )
        started = self._start_offline(
            snapshot,
            start_key="attachment-binding-start-001",
            profile_revision="attachment-profile-v1",
        )
        first_body = self._attachment_body(started, key="attachment-upload-001")

        with self.assertRaisesRegex(TeacherAgentDashboardError, "profile_revision"):
            snapshot.upload_attachment(
                {
                    **first_body,
                    "attachment_idempotency_key": "attachment-wrong-profile",
                    "profile_revision": "another-profile-v1",
                }
            )
        self.assertEqual(extractor.calls, [])
        self.assertEqual(
            snapshot.resume({"session_id": started["session_id"]})["context_version"],
            started["context_version"],
        )

        first_upload = snapshot.upload_attachment(first_body)
        first_attachment_id = first_upload["attachment"]["attachment_id"]
        self.assertEqual(
            first_upload["context_version"], started["context_version"] + 1
        )
        self.assertEqual(
            first_upload["expected_question_id"], started["expected_question_id"]
        )
        self.assertEqual(first_upload["profile_revision"], "attachment-profile-v1")
        self.assertEqual(
            extractor.calls,
            [(_SYNTHETIC_ONE_PIXEL_PNG, "image/png", "synthetic-answer.png")],
        )

        record = snapshot.sessions[started["session_id"]]
        stored_first = record.attachments[first_attachment_id]
        self.assertEqual(stored_first["question_id"], started["expected_question_id"])
        self.assertEqual(stored_first["round"], started["rounds_completed"])
        self.assertEqual(stored_first["profile_revision"], "attachment-profile-v1")
        self.assertFalse(stored_first["consumed"])
        self.assertFalse(stored_first["expired"])
        self.assertNotIn("data_base64", stored_first)
        self.assertNotIn("image_bytes", stored_first)
        serialized_record = json.dumps(stored_first, ensure_ascii=False)
        self.assertNotIn(str(first_body["data_base64"]), serialized_record)
        self.assertFalse(stored_first["evidence"]["raw_media_retained"])
        self.assertFalse(stored_first["evidence"]["remote_media_sent"])

        replayed_upload = snapshot.upload_attachment(first_body)
        self.assertEqual(replayed_upload, first_upload)
        self.assertEqual(len(extractor.calls), 1)
        self.assertEqual(record.context_version, first_upload["context_version"])
        with self.assertRaisesRegex(TeacherAgentDashboardError, "different request"):
            snapshot.upload_attachment(
                {**first_body, "display_name": "different-synthetic-name.png"}
            )
        self.assertEqual(len(extractor.calls), 1)
        self.assertEqual(record.context_version, first_upload["context_version"])

        second_body = {
            **self._attachment_body(started, key="attachment-upload-002"),
            "expected_context_version": first_upload["context_version"],
        }
        second_upload = snapshot.upload_attachment(second_body)
        second_attachment_id = second_upload["attachment"]["attachment_id"]
        self.assertEqual(
            second_upload["context_version"], first_upload["context_version"] + 1
        )

        step_body = {
            "session_id": started["session_id"],
            "expected_round": started["rounds_completed"],
            "expected_question_id": started["expected_question_id"],
            "expected_context_version": second_upload["context_version"],
            "profile_revision": "attachment-profile-v1",
            "idempotency_key": "attachment-step-001",
            "learner_response": "",
            "attachment_ids": [first_attachment_id],
            "signal": "partial",
        }
        with self.assertRaisesRegex(TeacherAgentDashboardError, "profile_revision"):
            snapshot.step(
                {
                    **step_body,
                    "idempotency_key": "attachment-step-wrong-profile",
                    "profile_revision": "another-profile-v1",
                }
            )
        self.assertFalse(record.attachments[first_attachment_id]["consumed"])

        stepped = snapshot.step(step_body)
        self.assertEqual(stepped["rounds_completed"], 1)
        self.assertEqual(
            stepped["context_version"], second_upload["context_version"] + 1
        )
        self.assertEqual(
            stepped["history"][-1]["learner_response"],
            "状态转移依赖当前状态与输入符号。",
        )
        self.assertEqual(stepped["history"][-1]["learner_text"], "")
        self.assertEqual(len(stepped["history"][-1]["multimodal_evidence"]), 1)
        public_evidence = stepped["history"][-1]["multimodal_evidence"][0]
        self.assertEqual(
            public_evidence["recognized_text"],
            "状态转移依赖当前状态与输入符号。",
        )
        self.assertFalse(public_evidence["raw_media_retained"])
        self.assertFalse(public_evidence["remote_media_sent"])
        self.assertFalse(public_evidence["formula_accuracy_established"])
        self.assertTrue(record.attachments[first_attachment_id]["consumed"])
        self.assertFalse(record.attachments[first_attachment_id]["expired"])
        self.assertFalse(record.attachments[second_attachment_id]["consumed"])
        self.assertTrue(record.attachments[second_attachment_id]["expired"])

        replayed_step = snapshot.step(step_body)
        self.assertEqual(replayed_step, stepped)
        self.assertEqual(record.session["round"], 1)
        self.assertEqual(len(record.session["history"]), 1)
        with self.assertRaisesRegex(TeacherAgentDashboardError, "different request"):
            snapshot.step({**step_body, "learner_response": "不同的重试载荷"})

        current_turn = {
            "session_id": started["session_id"],
            "expected_round": stepped["rounds_completed"],
            "expected_question_id": stepped["expected_question_id"],
            "expected_context_version": stepped["context_version"],
            "profile_revision": stepped["profile_summary"]["profile_revision"],
            "learner_response": "",
            "signal": "partial",
        }
        for label, unavailable_id in (
            ("consumed", first_attachment_id),
            ("old-round", second_attachment_id),
        ):
            with self.subTest(unavailable_attachment=label):
                with self.assertRaisesRegex(
                    TeacherAgentDashboardError,
                    "unavailable or already consumed",
                ):
                    snapshot.step(
                        {
                            **current_turn,
                            "idempotency_key": f"attachment-reuse-{label}",
                            "attachment_ids": [unavailable_id],
                        }
                    )
        self.assertEqual(record.session["round"], 1)
        self.assertEqual(record.context_version, stepped["context_version"])

        with self.assertRaisesRegex(TeacherAgentDashboardError, "expected_round"):
            snapshot.step(
                {
                    **current_turn,
                    "expected_round": 0,
                    "idempotency_key": "attachment-stale-round-guard",
                    "attachment_ids": [second_attachment_id],
                }
            )
        self.assertEqual(record.session["round"], 1)

    def test_confirmation_required_ocr_cannot_consume_a_turn_without_student_input(
        self,
    ) -> None:
        extractor = _ConfirmationRequiredVisionExtractor()
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.library_path,
            self.input_path,
            self.cases_path,
            vision_extractor=extractor,
        )
        started = self._start_offline(
            snapshot,
            start_key="confirmation-gate-start-001",
            profile_revision="confirmation-gate-profile-v1",
        )
        uploaded = snapshot.upload_attachment(
            self._attachment_body(started, key="confirmation-gate-upload-001")
        )
        attachment_id = uploaded["attachment"]["attachment_id"]
        self.assertTrue(uploaded["attachment"]["needs_student_confirmation"])
        body = {
            "session_id": started["session_id"],
            "expected_round": started["rounds_completed"],
            "expected_question_id": started["expected_question_id"],
            "expected_context_version": uploaded["context_version"],
            "profile_revision": "confirmation-gate-profile-v1",
            "idempotency_key": "confirmation-gate-step-001",
            "learner_response": "",
            "attachment_ids": [attachment_id],
            "signal": "partial",
        }
        with self.assertRaisesRegex(TeacherAgentDashboardError, "needs student confirmation"):
            snapshot.step(body)
        record = snapshot.sessions[started["session_id"]]
        self.assertEqual(record.session["round"], 0)
        self.assertFalse(record.attachments[attachment_id]["consumed"])
        self.assertIsNone(record.active_turn_id)

        confirmed = snapshot.step(
            {
                **body,
                "confirmed_attachment_ids": [attachment_id],
            }
        )
        self.assertEqual(confirmed["rounds_completed"], 1)
        self.assertTrue(record.attachments[attachment_id]["consumed"])
        stored_evidence = record.session["history"][-1]["multimodal_evidence"][0]
        self.assertTrue(stored_evidence["student_confirmed_recognized_text"])
        self.assertFalse(stored_evidence["needs_student_confirmation"])
        self.assertFalse(
            stored_evidence["student_confirmation_establishes_answer_correctness"]
        )

        correction_snapshot = build_teacher_agent_dashboard_snapshot(
            self.library_path,
            self.input_path,
            self.cases_path,
            vision_extractor=_ConfirmationRequiredVisionExtractor(),
        )
        correction_started = self._start_offline(
            correction_snapshot,
            start_key="confirmation-gate-correction-start-002",
        )
        correction_upload = correction_snapshot.upload_attachment(
            self._attachment_body(
                correction_started,
                key="confirmation-gate-correction-upload-002",
            )
        )
        correction = correction_snapshot.step(
            {
                "session_id": correction_started["session_id"],
                "expected_round": correction_started["rounds_completed"],
                "expected_question_id": correction_started["expected_question_id"],
                "expected_context_version": correction_upload["context_version"],
                "profile_revision": correction_started["profile_summary"][
                    "profile_revision"
                ],
                "idempotency_key": "confirmation-gate-correction-step-002",
                "learner_response": "图片里的正确答案是：状态与输入共同决定下一状态。",
                "attachment_ids": [correction_upload["attachment"]["attachment_id"]],
                "signal": "partial",
            }
        )
        self.assertEqual(correction["rounds_completed"], 1)
        self.assertEqual(
            correction["history"][-1]["learner_text"],
            "图片里的正确答案是：状态与输入共同决定下一状态。",
        )

    def test_failed_replacement_preserves_an_inflight_turn_until_it_commits(self) -> None:
        client = _BlockingCallLiveClient(block_call=2)
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        started = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "preserve-active-turn-start-001",
                "remote_processing_acknowledged": True,
            }
        )
        step_body = {
            "session_id": started["session_id"],
            "expected_round": started["rounds_completed"],
            "expected_question_id": started["expected_question_id"],
            "expected_context_version": started["context_version"],
            "profile_revision": started["profile_summary"]["profile_revision"],
            "idempotency_key": "preserve-active-turn-step-001",
            "learner_response": "这条回答应该在失败的画像切换后继续完成。",
        }
        step_results: list[dict[str, object]] = []
        step_errors: list[BaseException] = []

        def run_step() -> None:
            try:
                step_results.append(snapshot.step(step_body))
            except BaseException as exc:  # pragma: no cover - asserted below
                step_errors.append(exc)

        worker = threading.Thread(target=run_step)
        worker.start()
        self.assertTrue(client.blocked_call_entered.wait(timeout=2))
        replacement_profile = deepcopy(snapshot.demo_input["student_profile"])
        replacement_profile["profile_ref"] = "failed_candidate_profile"
        replacement_profile["contains_direct_identity"] = True
        replacement_body = {
            "goal": snapshot.demo_input["goal"],
            "student_profile": replacement_profile,
            "start_idempotency_key": "preserve-active-turn-replacement-002",
            "replace_session_id": started["session_id"],
            **self._replacement_guards(started),
            "remote_processing_acknowledged": True,
        }
        with self.assertRaisesRegex(Exception, "direct identity"):
            snapshot.start(replacement_body)
        record = snapshot.sessions[started["session_id"]]
        self.assertFalse(record.retiring)
        self.assertIsNotNone(record.active_turn_id)
        self.assertTrue(snapshot.resume({"session_id": started["session_id"]})["turn_runtime"]["active"])
        self.assertEqual(set(snapshot.sessions), {started["session_id"]})

        client.release_blocked_call.set()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(step_errors)
        self.assertEqual(len(step_results), 1)
        self.assertEqual(step_results[0]["rounds_completed"], 1)

    def test_step_requires_session_round_and_idempotency_key(self) -> None:
        snapshot = self._offline_snapshot()
        started = self._start_offline(snapshot)
        body = {
            "session_id": started["session_id"],
            "expected_round": 0,
            "expected_question_id": started["expected_question_id"],
            "expected_context_version": started["context_version"],
            "profile_revision": started["profile_summary"]["profile_revision"],
            "idempotency_key": "turn-001",
            "learner_response": "状态只需要看前一个位置。",
            "signal": "misconception",
            "misconception_tag": "missing_transition",
        }

        for required_field in (
            "session_id",
            "expected_round",
            "expected_question_id",
            "expected_context_version",
            "profile_revision",
            "idempotency_key",
        ):
            invalid = dict(body)
            invalid.pop(required_field)
            with self.subTest(required_field=required_field):
                with self.assertRaisesRegex(TeacherAgentDashboardError, required_field):
                    snapshot.step(invalid)
                self.assertEqual(
                    snapshot.resume({"session_id": started["session_id"]}), started
                )
                self.assertEqual(
                    snapshot.sessions[started["session_id"]].step_idempotency_cache,
                    {},
                )

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

        invalid_bindings = (
            ("expected_question_id", None),
            ("expected_question_id", ""),
            ("expected_question_id", " padded-question"),
            ("expected_context_version", None),
            ("expected_context_version", True),
            ("expected_context_version", "1"),
            ("expected_context_version", 1.0),
            ("profile_revision", None),
            ("profile_revision", ""),
            ("profile_revision", "padded-profile "),
        )
        for field, value in invalid_bindings:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(TeacherAgentDashboardError, field):
                    snapshot.step({**body, field: value})
                self.assertEqual(
                    snapshot.resume({"session_id": started["session_id"]}), started
                )
                self.assertEqual(
                    snapshot.sessions[started["session_id"]].step_idempotency_cache,
                    {},
                )

        accepted = snapshot.step(body)
        self.assertEqual(accepted["rounds_completed"], 1)
        self.assertEqual(accepted["context_version"], 2)
        self.assertEqual(len(accepted["history"]), 1)

    def test_step_rejects_question_context_and_profile_version_mismatch(self) -> None:
        snapshot = self._offline_snapshot()
        started = self._start_offline(
            snapshot,
            start_key="binding-start-001",
            profile_revision="profile-binding-v1",
        )
        base = {
            "session_id": started["session_id"],
            "expected_round": 0,
            "expected_question_id": started["expected_question_id"],
            "expected_context_version": started["context_version"],
            "profile_revision": "profile-binding-v1",
            "idempotency_key": "binding-turn-001",
            "learner_response": "这是对当前问题的回答。",
            "signal": "partial",
        }

        mismatches = (
            ("expected_question_id", "stale-question-id", "expected_question_id"),
            ("expected_context_version", 999, "expected_context_version"),
            ("profile_revision", "stale-profile-v0", "profile_revision"),
        )
        for field, value, message in mismatches:
            with self.subTest(field=field):
                with self.assertRaisesRegex(TeacherAgentDashboardError, message):
                    snapshot.step(
                        {
                            **base,
                            field: value,
                            "idempotency_key": f"mismatch-{field}",
                        }
                    )
                unchanged = snapshot.resume({"session_id": started["session_id"]})
                self.assertEqual(unchanged["rounds_completed"], 0)
                self.assertEqual(unchanged["context_version"], 1)

        advanced = snapshot.step(base)
        self.assertEqual(advanced["rounds_completed"], 1)
        self.assertEqual(advanced["context_version"], 2)
        with self.assertRaisesRegex(
            TeacherAgentDashboardError, "expected_context_version"
        ):
            snapshot.step(
                {
                    **base,
                    "expected_round": 1,
                    "expected_question_id": advanced["expected_question_id"],
                    "idempotency_key": "binding-turn-002",
                }
            )

    def test_step_replay_is_cached_and_does_not_advance_twice(self) -> None:
        snapshot = self._offline_snapshot()
        started = self._start_offline(snapshot)
        body = {
            "session_id": started["session_id"],
            "expected_round": 0,
            "expected_question_id": started["expected_question_id"],
            "expected_context_version": started["context_version"],
            "profile_revision": started["profile_summary"]["profile_revision"],
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

        with self.assertRaisesRegex(TeacherAgentDashboardError, "different request"):
            snapshot.step({**body, "learner_response": "这是另一条回答。"})
        self.assertEqual(snapshot.session["round"], 1)

        with self.assertRaisesRegex(TeacherAgentDashboardError, "expected_round"):
            snapshot.step({**body, "idempotency_key": "turn-002"})
        self.assertEqual(snapshot.session["round"], 1)

        second = snapshot.step(
            {
                **body,
                "expected_round": 1,
                "expected_question_id": first["expected_question_id"],
                "expected_context_version": first["context_version"],
                "profile_revision": first["profile_summary"]["profile_revision"],
                "idempotency_key": "turn-002",
                "learner_response": "我还需要同时看输入字符。",
                "signal": "partial",
                "misconception_tag": None,
            }
        )
        self.assertEqual(second["rounds_completed"], 2)
        self.assertEqual(len(second["history"]), 2)

    def test_concurrent_identical_step_is_applied_once_or_explicitly_running(self) -> None:
        snapshot = self._offline_snapshot()
        started = self._start_offline(snapshot)
        body = {
            "session_id": started["session_id"],
            "expected_round": 0,
            "expected_question_id": started["expected_question_id"],
            "expected_context_version": started["context_version"],
            "profile_revision": started["profile_summary"]["profile_revision"],
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

        self.assertEqual(len(results) + len(errors), 2)
        self.assertLessEqual(len(errors), 1)
        if errors:
            self.assertIsInstance(errors[0], TeacherAgentDashboardError)
            self.assertIn("idempotent turn is still running", str(errors[0]))
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(item == results[0] for item in results))
        self.assertEqual(snapshot.session["round"], 1)
        self.assertEqual(len(snapshot.session["history"]), 1)

    def test_step_idempotency_keys_are_scoped_to_each_session(self) -> None:
        snapshot = self._offline_snapshot()
        first_started = self._start_offline(snapshot)
        reused_key = "safe-to-reuse-after-new-session"
        first = snapshot.step(
            {
                "session_id": first_started["session_id"],
                "expected_round": 0,
                "expected_question_id": first_started["expected_question_id"],
                "expected_context_version": first_started["context_version"],
                "profile_revision": first_started["profile_summary"][
                    "profile_revision"
                ],
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
                "expected_question_id": second_started["expected_question_id"],
                "expected_context_version": second_started["context_version"],
                "profile_revision": second_started["profile_summary"][
                    "profile_revision"
                ],
                "idempotency_key": reused_key,
                "learner_response": "第二次会话。",
                "signal": "partial",
            }
        )
        self.assertEqual(second["rounds_completed"], 1)
        self.assertEqual(
            snapshot.resume({"session_id": first_started["session_id"]}), first
        )
        self.assertEqual(
            snapshot.resume({"session_id": second_started["session_id"]}), second
        )

    def test_command_requires_session_and_rejects_stale_round(self) -> None:
        snapshot = self._offline_snapshot()
        started = self._start_offline(snapshot)
        session_id = started["session_id"]

        with self.assertRaisesRegex(TeacherAgentDashboardError, "session_id"):
            snapshot.command({"command": "auto"})
        with self.assertRaisesRegex(TeacherAgentDashboardError, "session_id"):
            snapshot.command({"command": "auto", "session_id": "wrong"})
        valid = self._command_body(started, "auto", key="command-contract-valid-001")
        with self.assertRaisesRegex(TeacherAgentDashboardError, "expected_round"):
            snapshot.command({**valid, "expected_round": 1})

        current = snapshot.command(valid)
        self.assertEqual(current["rounds_completed"], 0)
        replay = snapshot.command(valid)
        self.assertEqual(replay, current)
        with self.assertRaisesRegex(TeacherAgentDashboardError, "different request"):
            snapshot.command({**valid, "command": "select_skill"})

        next_command = self._command_body(
            current, "auto", key="command-contract-valid-002"
        )
        second = snapshot.command(next_command)
        self.assertEqual(second["rounds_completed"], 0)
        self.assertGreater(second["context_version"], current["context_version"])

        for missing in (
            "command_idempotency_key",
            "expected_round",
            "expected_question_id",
            "expected_context_version",
            "profile_revision",
        ):
            with self.subTest(missing=missing):
                body = self._command_body(
                    second, "auto", key=f"command-missing-{missing}"
                )
                del body[missing]
                before = snapshot.resume({"session_id": session_id})
                with self.assertRaises(TeacherAgentDashboardError):
                    snapshot.command(body)
                self.assertEqual(snapshot.resume({"session_id": session_id}), before)

    def test_stale_commands_are_rejected_without_mutation(self) -> None:
        snapshot = self._offline_snapshot()
        started = self._start_offline(snapshot)
        stale_variants = (
            ("expected_round", started["rounds_completed"] + 1),
            ("expected_question_id", "stale-question"),
            ("expected_context_version", started["context_version"] + 1),
            ("profile_revision", "stale-profile"),
        )
        for command in ("auto", "select_skill", "stop"):
            for field, value in stale_variants:
                with self.subTest(command=command, field=field):
                    body = self._command_body(
                        started,
                        command,
                        key=f"stale-{command}-{field}",
                        skill_id="skill_concrete_example_bridge",
                    )
                    body[field] = value
                    before = snapshot.resume({"session_id": started["session_id"]})
                    with self.assertRaises(TeacherAgentDashboardError):
                        snapshot.command(body)
                    self.assertEqual(
                        snapshot.resume({"session_id": started["session_id"]}),
                        before,
                    )

    def test_response_failure_does_not_commit_start_step_or_command(self) -> None:
        snapshot = self._offline_snapshot()
        original = self._start_offline(snapshot, start_key="atomic-original")
        original_record = snapshot.sessions[original["session_id"]]
        registry_before = set(snapshot.sessions)
        replacement_profile = deepcopy(snapshot.demo_input["student_profile"])
        replacement_profile["profile_ref"] = "atomic_replacement"

        with patch.object(
            TeacherAgentDashboardSnapshot,
            "_response",
            side_effect=TeacherAgentDashboardError("simulated response failure"),
        ):
            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "simulated response failure"
            ):
                self._start_offline(
                    snapshot,
                    start_key="atomic-replacement",
                    student_profile=replacement_profile,
                    replace_session_id=original["session_id"],
                )
        self.assertEqual(set(snapshot.sessions), registry_before)
        self.assertIs(snapshot.sessions[original["session_id"]], original_record)
        self.assertEqual(
            snapshot.resume({"session_id": original["session_id"]}), original
        )
        self.assertNotIn("atomic-replacement", snapshot.start_idempotency_cache)

        step_body = {
            "session_id": original["session_id"],
            "expected_round": original["rounds_completed"],
            "expected_question_id": original["expected_question_id"],
            "expected_context_version": original["context_version"],
            "profile_revision": original["profile_summary"]["profile_revision"],
            "idempotency_key": "atomic-step",
            "learner_response": "这是一次不应提交的候选回答。",
            "signal": "partial",
        }
        with patch.object(
            TeacherAgentDashboardSnapshot,
            "_response",
            side_effect=TeacherAgentDashboardError("simulated response failure"),
        ):
            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "simulated response failure"
            ):
                snapshot.step(step_body)
        self.assertEqual(
            snapshot.resume({"session_id": original["session_id"]}), original
        )
        self.assertNotIn("atomic-step", original_record.step_idempotency_cache)

        command_body = self._command_body(original, "auto", key="atomic-command")
        with patch.object(
            TeacherAgentDashboardSnapshot,
            "_response",
            side_effect=TeacherAgentDashboardError("simulated response failure"),
        ):
            with self.assertRaisesRegex(
                TeacherAgentDashboardError, "simulated response failure"
            ):
                snapshot.command(command_body)
        self.assertEqual(
            snapshot.resume({"session_id": original["session_id"]}), original
        )
        self.assertNotIn("atomic-command", original_record.command_idempotency_cache)

    def test_command_idempotency_cache_is_bounded_per_session(self) -> None:
        snapshot = self._offline_snapshot()
        current = self._start_offline(snapshot, start_key="bounded-command-start")
        for index in range(70):
            current = snapshot.command(
                self._command_body(
                    current,
                    "auto",
                    key=f"bounded-command-{index:03d}",
                )
            )
        cache = snapshot.sessions[current["session_id"]].command_idempotency_cache
        self.assertLessEqual(len(cache), 64)
        self.assertNotIn("bounded-command-000", cache)
        self.assertIn("bounded-command-069", cache)

    def test_manual_skill_lock_persists_then_releases_at_execution_guard(self) -> None:
        client = _ValidManualTurnLiveClient()
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        skill_id = "skill_concrete_example_bridge"
        started = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "manual-lock-start-001",
                "remote_processing_acknowledged": True,
                "manual_skill_id": skill_id,
            }
        )
        self.assertEqual(started["pending_skill_id"], skill_id)
        self.assertEqual(
            started["pending_skill_effective_from"],
            "next_learner_response",
        )
        self.assertNotEqual(
            started["next_action"]["primary_skill"]["skill_id"], skill_id
        )
        self.assertFalse(started["next_action"]["manual_override_applied"])

        current = started
        for round_number in range(2):
            current = snapshot.step(
                {
                    "session_id": current["session_id"],
                    "expected_round": round_number,
                    "expected_question_id": current["expected_question_id"],
                    "expected_context_version": current["context_version"],
                    "profile_revision": current["profile_summary"]["profile_revision"],
                    "idempotency_key": f"manual-lock-turn-{round_number + 1:03d}",
                    "learner_response": "我能说出状态，但转移依据还不完整。",
                }
            )
            self.assertEqual(current["pending_skill_id"], skill_id)
            self.assertTrue(current["next_action"]["manual_override_applied"])
            self.assertIsNone(current["control_notice"])

        released = snapshot.step(
            {
                "session_id": current["session_id"],
                "expected_round": 2,
                "expected_question_id": current["expected_question_id"],
                "expected_context_version": current["context_version"],
                "profile_revision": current["profile_summary"]["profile_revision"],
                "idempotency_key": "manual-lock-turn-003",
                "learner_response": "我再尝试一次。",
            }
        )
        self.assertIsNone(released["pending_skill_id"])
        self.assertEqual(
            released["control_notice"],
            "manual_skill_released_by_skill_contract_guard",
        )
        self.assertEqual(
            snapshot.resume({"session_id": released["session_id"]}), released
        )

    def test_inapplicable_manual_skill_is_safely_released_without_fallback(
        self,
    ) -> None:
        client = _ValidManualTurnLiveClient()
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        started = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "manual-guard-start-001",
                "remote_processing_acknowledged": True,
                "manual_skill_id": "skill_transfer_check",
            }
        )

        updated = snapshot.step(
            {
                "session_id": started["session_id"],
                "expected_round": 0,
                "expected_question_id": started["expected_question_id"],
                "expected_context_version": started["context_version"],
                "profile_revision": started["profile_summary"]["profile_revision"],
                "idempotency_key": "manual-guard-turn-001",
                "learner_response": "我只能说出一部分。",
            }
        )

        self.assertIsNone(updated["pending_skill_id"])
        self.assertEqual(
            updated["control_notice"],
            "manual_skill_released_by_skill_contract_guard",
        )
        self.assertFalse(updated["next_action"]["manual_override_applied"])
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 0)

    def test_start_rejects_invalid_manual_skill_before_model_call(self) -> None:
        client = _FakeLiveClient()
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )
        with self.assertRaisesRegex(TeacherAgentDashboardError, "manual_skill_id"):
            snapshot.start(
                {
                    "goal": snapshot.demo_input["goal"],
                    "student_profile": snapshot.demo_input["student_profile"],
                    "start_idempotency_key": "invalid-manual-start-001",
                    "remote_processing_acknowledged": True,
                    "manual_skill_id": "skill_wait_and_elicit",
                }
            )
        self.assertEqual(client.chat_json_call_count, 0)

        with self.assertRaisesRegex(TeacherAgentDashboardError, "allowed_skill_ids"):
            snapshot.start(
                {
                    "goal": snapshot.demo_input["goal"],
                    "student_profile": snapshot.demo_input["student_profile"],
                    "start_idempotency_key": "excluded-manual-start-002",
                    "remote_processing_acknowledged": True,
                    "manual_skill_id": "skill_concrete_example_bridge",
                    "allowed_skill_ids": ["skill_diagnostic_questioning"],
                }
            )
        self.assertEqual(client.chat_json_call_count, 0)

        offline = self._offline_snapshot()
        with self.assertRaisesRegex(TeacherAgentDashboardError, "live DeepSeek"):
            offline.start(
                {
                    "goal": offline.demo_input["goal"],
                    "student_profile": offline.demo_input["student_profile"],
                    "start_idempotency_key": "offline-manual-start-003",
                    "manual_skill_id": "skill_concrete_example_bridge",
                }
            )

    def test_terminal_session_rejects_all_commands_without_mutation(self) -> None:
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=_FakeLiveClient(),
        )
        started = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "terminal-command-start-001",
                "remote_processing_acknowledged": True,
            }
        )
        stop_body = self._command_body(started, "stop", key="terminal-stop-001")
        stopped = snapshot.command(stop_body)
        self.assertEqual(stopped["status"], "terminated_unable")
        self.assertEqual(snapshot.command(stop_body), stopped)

        for command in ("auto", "select_skill", "stop"):
            with self.subTest(command=command):
                with self.assertRaisesRegex(
                    TeacherAgentDashboardError, "terminal session"
                ):
                    snapshot.command(
                        self._command_body(
                            stopped,
                            command,
                            key=f"terminal-{command}-002",
                            skill_id="skill_concrete_example_bridge",
                        )
                    )
                self.assertEqual(
                    snapshot.resume({"session_id": stopped["session_id"]}),
                    stopped,
                )

    def test_terminal_session_replacement_is_atomic_and_does_not_retain_old_state(
        self,
    ) -> None:
        """Starting a new profile after handoff retires the terminal session."""

        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=_FakeLiveClient(),
        )
        started = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "terminal-replacement-start-001",
                "remote_processing_acknowledged": True,
            }
        )
        stopped = snapshot.command(
            self._command_body(started, "stop", key="terminal-replacement-stop-002")
        )
        self.assertEqual(stopped["status"], "terminated_unable")

        replacement_profile = deepcopy(snapshot.demo_input["student_profile"])
        replacement_profile.update(
            {
                "profile_ref": "terminal-replacement-profile",
                "learner_level": "advanced",
                "initial_mastery": {
                    "prerequisite": 0.8,
                    "conceptual": 0.7,
                    "procedural": 0.65,
                    "transfer": 0.6,
                },
            }
        )
        replacement = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": replacement_profile,
                "profile_revision": "terminal-replacement-profile-v2",
                "profile_display_name": "终止后新学生",
                "start_idempotency_key": "terminal-replacement-start-003",
                "replace_session_id": stopped["session_id"],
                **self._replacement_guards(stopped),
                "remote_processing_acknowledged": True,
            }
        )
        self.assertNotEqual(replacement["session_id"], stopped["session_id"])
        self.assertEqual(set(snapshot.sessions), {replacement["session_id"]})
        self.assertEqual(replacement["rounds_completed"], 0)
        self.assertEqual(replacement["history"], [])
        self.assertEqual(
            replacement["profile_summary"]["profile_revision"],
            "terminal-replacement-profile-v2",
        )
        self.assertEqual(snapshot.resume({"session_id": replacement["session_id"]}), replacement)
        with self.assertRaisesRegex(TeacherAgentDashboardError, "no longer available"):
            snapshot.resume({"session_id": stopped["session_id"]})

    def test_session_capacity_rejects_when_every_eviction_candidate_is_busy(
        self,
    ) -> None:
        snapshot = self._offline_snapshot()
        sessions = [
            self._start_offline(snapshot, start_key=f"capacity-{index:03d}")
            for index in range(16)
        ]
        locked_records = [snapshot.sessions[item["session_id"]] for item in sessions]
        for record in locked_records:
            self.assertTrue(record.lock.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(TeacherAgentDashboardError, "capacity is busy"):
                self._start_offline(snapshot, start_key="capacity-overflow")
            self.assertEqual(len(snapshot.sessions), 16)
            self.assertNotIn("capacity-overflow", snapshot.start_idempotency_cache)
        finally:
            for record in locked_records:
                record.lock.release()

        admitted = self._start_offline(snapshot, start_key="capacity-after-release")
        self.assertEqual(len(snapshot.sessions), 16)
        self.assertIn(admitted["session_id"], snapshot.sessions)

    def test_online_capacity_is_reserved_before_call_and_replacement_reuses_slot(
        self,
    ) -> None:
        client = _FakeLiveClient()
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            client=client,
        )

        def start_body(index: int, *, replace_session_id: str | None = None):
            body = {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": f"online-capacity-{index:03d}",
                "remote_processing_acknowledged": True,
            }
            if replace_session_id is not None:
                body["replace_session_id"] = replace_session_id
                current = snapshot.resume({"session_id": replace_session_id})
                body.update(self._replacement_guards(current))
            return body

        sessions = [snapshot.start(start_body(index)) for index in range(16)]
        records = [snapshot.sessions[item["session_id"]] for item in sessions]
        for record in records:
            self.assertTrue(record.lock.acquire(blocking=False))
        calls_before_rejection = client.chat_json_call_count
        try:
            with self.assertRaisesRegex(TeacherAgentDashboardError, "capacity is busy"):
                snapshot.start(start_body(16))
            self.assertEqual(
                client.chat_json_call_count,
                calls_before_rejection,
                "a rejected full-capacity start must not call DeepSeek",
            )
            self.assertEqual(len(snapshot.sessions), 16)

            replacement_id = sessions[-1]["session_id"]
            records[-1].lock.release()
            replacement = snapshot.start(
                start_body(17, replace_session_id=replacement_id)
            )
            self.assertEqual(client.chat_json_call_count, calls_before_rejection + 1)
            self.assertEqual(len(snapshot.sessions), 16)
            self.assertNotIn(replacement_id, snapshot.sessions)
            self.assertIn(replacement["session_id"], snapshot.sessions)
        finally:
            for record in records:
                if record.lock.locked():
                    record.lock.release()

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

        started = snapshot.start({**start_body, "remote_processing_acknowledged": True})
        self.assertEqual(started["rounds_completed"], 0)
        self.assertTrue(started["session_id"])

        offline = self._offline_snapshot()
        self.assertFalse(
            offline.bootstrap()["interaction_contract"][
                "remote_processing_acknowledgement_required"
            ]
        )
        self.assertEqual(self._start_offline(offline)["rounds_completed"], 0)

    def test_loopback_attachment_api_routes_extracted_evidence_into_step(self) -> None:
        extractor = _RecordingVisionExtractor()
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.library_path,
            self.input_path,
            self.cases_path,
            vision_extractor=extractor,
        )
        try:
            server, url = create_teacher_agent_dashboard_server(
                snapshot, capability_token="c" * 24
            )
        except PermissionError:
            self.skipTest("sandbox does not permit loopback socket binding")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        parsed = urlsplit(url)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)

        def post_json(route: str, payload: dict[str, object]) -> tuple[int, dict]:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            connection.request(
                "POST",
                f"{parsed.path}{route}",
                body=encoded,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            return response.status, json.loads(response.read())

        try:
            status, started = post_json(
                "api/start",
                {
                    "goal": snapshot.demo_input["goal"],
                    "student_profile": snapshot.demo_input["student_profile"],
                    "profile_revision": "http-attachment-profile-v1",
                    "start_idempotency_key": "http-attachment-start-001",
                },
            )
            self.assertEqual(status, 200)

            attachment_body = self._attachment_body(
                started, key="http-attachment-upload-001"
            )
            status, uploaded = post_json("api/attachment", attachment_body)
            self.assertEqual(status, 200)
            self.assertEqual(
                uploaded["context_version"], started["context_version"] + 1
            )
            self.assertEqual(len(extractor.calls), 1)
            self.assertNotIn(
                str(attachment_body["data_base64"]),
                json.dumps(uploaded, ensure_ascii=False),
            )

            attachment_id = uploaded["attachment"]["attachment_id"]
            status, stepped = post_json(
                "api/step",
                {
                    "session_id": started["session_id"],
                    "expected_round": started["rounds_completed"],
                    "expected_question_id": started["expected_question_id"],
                    "expected_context_version": uploaded["context_version"],
                    "profile_revision": "http-attachment-profile-v1",
                    "idempotency_key": "http-attachment-step-001",
                    "learner_response": "",
                    "attachment_ids": [attachment_id],
                    "signal": "partial",
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual(stepped["rounds_completed"], 1)
            self.assertEqual(
                stepped["history"][-1]["learner_response"],
                "状态转移依赖当前状态与输入符号。",
            )
            stored = snapshot.sessions[started["session_id"]].attachments[attachment_id]
            self.assertTrue(stored["consumed"])
            self.assertFalse(stored["expired"])
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

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
            self.assertIn(
                "default-src 'none'", response.getheader("Content-Security-Policy")
            )
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
                    "expected_question_id": started["expected_question_id"],
                    "expected_context_version": started["context_version"],
                    "profile_revision": started["profile_summary"]["profile_revision"],
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
