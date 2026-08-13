from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import unittest

from teaching_skill_miner.deepseek_client import (
    DeepSeekClient,
    DeepSeekClientError,
    DeepSeekConfig,
)
from teaching_skill_miner.teacher_agent import (
    advance_teacher_agent_session,
    start_teacher_agent_session,
)
from teaching_skill_miner.teacher_agent_loop import (
    LOOP_SCHEMA,
    PLAN_SCHEMA,
    TeachingAgentLoopOptions,
    public_agent_loop_trace,
    run_teaching_agent_loop,
)
from teaching_skill_miner.teacher_agent_context import build_layered_context
from teaching_skill_miner.teacher_agent_live import (
    LiveAgentOptions,
    advance_live_teacher_agent_session,
    start_live_teacher_agent_session,
)


ROOT = Path(__file__).resolve().parents[1]


def _envelope(content: dict[str, object], *, response_id: str) -> bytes:
    return json.dumps(
        {
            "id": response_id,
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "prompt_cache_hit_tokens": 9,
                "prompt_cache_miss_tokens": 3,
            },
        }
    ).encode()


def _tool(*calls: dict[str, object]) -> dict[str, object]:
    return {"schema": PLAN_SCHEMA, "kind": "tool_calls", "tool_calls": list(calls)}


def _call(
    name: str,
    arguments: dict[str, object] | None = None,
    *,
    call_id: str | None = None,
) -> dict[str, object]:
    return {
        "call_id": call_id or f"call_{name}",
        "name": name,
        "arguments": arguments or {},
    }


def _action(
    *,
    message: str = "请先用自己的话说明动态规划状态的含义。",
) -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "kind": "teaching_action",
        "selected_skill_id": "skill_diagnostic_questioning",
        "supporting_skill_ids": ["skill_wait_and_elicit"],
        "next_focus": "prerequisite",
        "action_type": "probe_prior_knowledge",
        "message": message,
        "expected_signal": "学生给出一个可核验的前置概念。",
        "reason": "先诊断已有知识，再决定下一步教学。",
    }


def _selection_plan(
    *,
    primary_skill_id: str = "skill_diagnostic_questioning",
    supporting_skill_ids: list[str] | None = None,
    next_focus: str = "prerequisite",
) -> dict[str, object]:
    supports = (
        ["skill_wait_and_elicit"]
        if supporting_skill_ids is None
        else supporting_skill_ids
    )
    return _tool(
        _call("inspect_student_state", call_id="inspect-state-1"),
        _call(
            "select_skills",
            {
                "primary_skill_id": primary_skill_id,
                "supporting_skill_ids": supports,
                "reason": "首轮先诊断前置知识",
            },
            call_id="select-skill-1",
        ),
        _call(
            "set_next_focus",
            {"next_focus": next_focus},
            call_id="set-focus-1",
        ),
    )


def _route_ready(reason: str = "已完成本轮状态读取与 Skill 路由。") -> dict[str, object]:
    return {"schema": PLAN_SCHEMA, "kind": "route_ready", "reason": reason}


def _live_plan(
    *,
    signal: str,
    skill_id: str,
    action_type: str,
    message: str,
    support: list[str] | None = None,
) -> dict[str, object]:
    alignment = {
        "not_observed": "not_applicable",
        "partial": "partially_aligned",
        "correct": "aligned",
    }[signal]
    return {
        "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
        "diagnosis": {
            "signal": signal,
            "confidence": 0.0 if signal == "not_observed" else 0.82,
            "answer_alignment": alignment,
            "matched_concepts": [],
            "missing_concepts": [],
            "diagnosis_reason": "根据当前回合的结构化学习证据完成判断。",
            "evidence_excerpt": "",
            "misconception_tag": None,
            "misconception_description": "",
            "resolved_misconception_tags": [],
            "response_quality": "empty" if signal == "not_observed" else "partial",
            "engagement_level": "unknown" if signal == "not_observed" else "medium",
            "needs_human_review": False,
        },
        "decision": {
            "primary_skill_id": skill_id,
            "supporting_skill_ids": support or [],
            "selection_reason": "根据学生当前状态选择适用 Skill。",
            "next_focus": "prerequisite" if signal == "not_observed" else "conceptual",
        },
        "teacher_action": {
            "type": action_type,
            "message": message,
            "expected_signal": "学生给出可核验的解释或例子。",
            "question_contract": {
                "answer_type": "explanation",
                "target_concepts": ["动态规划的状态与转移"],
                "accepted_aliases": [],
                "success_criteria": ["说明状态含义并给出依据"],
            },
        },
        "stop_recommendation": {"should_stop": False, "reason": ""},
    }


def _deepseek_script_client(responses: list[dict[str, object]]) -> DeepSeekClient:
    queue = deque(responses)

    def transport(
        _url: str,
        _headers: dict[str, str],
        _payload: bytes,
        _timeout: float,
    ) -> tuple[int, bytes]:
        content = queue.popleft()
        return 200, _envelope(content, response_id=f"script-{len(responses) - len(queue)}")

    return DeepSeekClient(
        DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
        api_key="integration-secret",
        transport=transport,
    )


class _CapturingModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = deque(responses)
        self.messages: list[list[dict[str, str]]] = []

    def __call__(self, messages: object) -> object:
        self.messages.append(deepcopy(list(messages)))
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class TeacherAgentLoopIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = json.loads(
            (ROOT / "data/teacher_agent_demo_input.json").read_text(encoding="utf-8")
        )
        cls.library = json.loads(
            (ROOT / "data/teacher_agent_skill_library_v2.json").read_text(
                encoding="utf-8"
            )
        )

    def _session(self) -> dict[str, object]:
        return start_teacher_agent_session(
            deepcopy(self.demo["goal"]),
            deepcopy(self.demo["student_profile"]),
            deepcopy(self.library),
        )

    def test_real_deepseek_chat_json_adapter_runs_tool_observation_loop(self) -> None:
        responses = deque(
            [
                _envelope(_selection_plan(), response_id="loop-tool-plan"),
                _envelope(_action(), response_id="loop-final-action"),
            ]
        )
        captured_bodies: list[dict[str, object]] = []

        def transport(
            _url: str,
            headers: dict[str, str],
            payload: bytes,
            _timeout: float,
        ) -> tuple[int, bytes]:
            self.assertEqual(headers["Authorization"], "Bearer integration-secret")
            captured_bodies.append(json.loads(payload))
            return 200, responses.popleft()

        client = DeepSeekClient(
            DeepSeekConfig(
                allow_remote_student_data=True,
                max_retries=0,
                temperature=0.0,
            ),
            api_key="integration-secret",
            transport=transport,
        )

        result = run_teaching_agent_loop(
            self._session(),
            client,
            options=TeachingAgentLoopOptions(model_retries=0),
        )

        self.assertEqual(result["schema"], LOOP_SCHEMA)
        self.assertEqual(result["status"], "action_ready")
        self.assertFalse(result["deterministic_fallback"])
        self.assertEqual(len(captured_bodies), 2)
        self.assertTrue(all(body["model"] == "deepseek-v4-flash" for body in captured_bodies))
        self.assertTrue(
            all(
                body["response_format"] == {"type": "json_object"}
                for body in captured_bodies
            )
        )
        second_user_payload = json.loads(captured_bodies[1]["messages"][1]["content"])
        observed_calls = {
            item["call_id"] for item in second_user_payload["previous_tool_results"]
        }
        self.assertEqual(
            observed_calls,
            {"inspect-state-1", "select-skill-1", "set-focus-1"},
        )
        traces = [
            event["trace"]
            for event in result["events"]
            if event["type"] == "model_plan"
        ]
        self.assertEqual([trace["request_kind"] for trace in traces], ["teacher_agent_loop"] * 2)
        public_traces = [
            event["trace"]
            for event in public_agent_loop_trace(result)["events"]
            if event["type"] == "model_plan"
        ]
        self.assertEqual(
            [
                {
                    "prompt_cache_hit_tokens": trace["usage"][
                        "prompt_cache_hit_tokens"
                    ],
                    "prompt_cache_miss_tokens": trace["usage"][
                        "prompt_cache_miss_tokens"
                    ],
                }
                for trace in public_traces
            ],
            [
                {
                    "prompt_cache_hit_tokens": 9,
                    "prompt_cache_miss_tokens": 3,
                }
            ]
            * 2,
        )
        self.assertNotIn("integration-secret", json.dumps(result, ensure_ascii=False))

    def test_initial_and_followup_sessions_expose_only_bounded_round_context(self) -> None:
        initial = self._session()
        first_model = _CapturingModel([_selection_plan(), _action()])
        first_result = run_teaching_agent_loop(initial, first_model)
        first_context = json.loads(first_model.messages[0][1]["content"])[
            "teaching_context"
        ]

        self.assertEqual(first_context["round"], 0)
        self.assertEqual(first_context["history"], [])
        self.assertEqual(first_result["action"]["skill"]["skill_id"], "skill_diagnostic_questioning")

        followup = advance_teacher_agent_session(
            initial,
            learner_response="dp[i] 只表示走到第 i 级。",
            signal="partial",
            signal_confidence=0.72,
            answer_alignment="partially_aligned",
        )
        second_model = _CapturingModel([_selection_plan(), _action(message="再说明 dp[i] 保存的数值是什么。")])
        second_result = run_teaching_agent_loop(followup, second_model)
        second_context = json.loads(second_model.messages[0][1]["content"])[
            "teaching_context"
        ]

        self.assertEqual(second_context["round"], 1)
        self.assertEqual(len(second_context["history"]), 1)
        self.assertEqual(second_context["history"][0]["signal"], "partial")
        self.assertEqual(
            second_context["history"][0]["learner_response"],
            "dp[i] 只表示走到第 i 级。",
        )
        self.assertEqual(second_result["status"], "action_ready")
        self.assertNotEqual(first_result["session_fingerprint"], second_result["session_fingerprint"])

    def test_tool_events_have_a_persistable_secret_free_receipt_contract(self) -> None:
        model = _CapturingModel([_selection_plan(), _action()])
        result = run_teaching_agent_loop(self._session(), model)
        tool_events = [event for event in result["events"] if event["type"] == "tool_result"]

        self.assertEqual(len(tool_events), 3)
        for event in tool_events:
            self.assertEqual(set(event), {"type", "step", "call_id", "tool", "ok", "result"})
            self.assertTrue(event["ok"])
            self.assertIsInstance(event["step"], int)
            self.assertIsInstance(event["call_id"], str)
            self.assertIsInstance(event["tool"], str)
            self.assertIsInstance(event["result"], dict)

        # This is the exact bounded shape that live/dashboard persistence should
        # project into ``session.agent_runtime.last_agent_loop``.  It contains
        # no model messages, prompt text, learner response, credential or raw media.
        persisted_receipt = {
            "schema": result["schema"],
            "status": result["status"],
            "steps": result["steps"],
            "tool_call_count": len(tool_events),
            "termination_reason": result["termination_reason"],
            "events": deepcopy(result["events"]),
        }
        encoded = json.dumps(persisted_receipt, ensure_ascii=False, allow_nan=False)
        self.assertEqual(
            set(persisted_receipt),
            {
                "schema",
                "status",
                "steps",
                "tool_call_count",
                "termination_reason",
                "events",
            },
        )
        self.assertNotIn("messages", encoded)
        self.assertNotIn("previous_tool_results", encoded)
        self.assertNotIn("learner_response", encoded)
        self.assertNotIn("Authorization", encoded)
        self.assertNotIn("raw_media", encoded)

    def test_model_transport_failure_uses_deterministic_action_fallback(self) -> None:
        calls = 0

        def unavailable_transport(*_args: object) -> tuple[int, bytes]:
            nonlocal calls
            calls += 1
            raise TimeoutError("simulated outage containing no learner data")

        client = DeepSeekClient(
            DeepSeekConfig(
                allow_remote_student_data=True,
                max_retries=0,
            ),
            api_key="fallback-secret",
            transport=unavailable_transport,
        )

        result = run_teaching_agent_loop(
            self._session(),
            client,
            options=TeachingAgentLoopOptions(model_retries=1),
        )

        self.assertEqual(calls, 2)
        self.assertEqual(result["status"], "action_ready")
        self.assertTrue(result["deterministic_fallback"])
        self.assertTrue(result["action"]["reason"].startswith("deterministic_safety_fallback:"))
        self.assertEqual(
            [event["type"] for event in result["events"]],
            ["model_error", "model_error", "fallback"],
        )
        self.assertNotIn("fallback-secret", json.dumps(result, ensure_ascii=False))

    def test_live_initial_turn_persists_public_agent_loop_receipt(self) -> None:
        client = _deepseek_script_client(
            [
                _selection_plan(),
                _route_ready(),
                _live_plan(
                    signal="not_observed",
                    skill_id="skill_diagnostic_questioning",
                    action_type="probe_prior_knowledge",
                    message="请先说出一个完成当前目标所需的前置概念。",
                ),
            ]
        )
        options = LiveAgentOptions(
            agent_loop_enabled=True,
            maximum_agent_steps=4,
            maximum_agent_tool_calls_per_step=3,
            agent_loop_model_retries=0,
        )
        session = start_live_teacher_agent_session(
            deepcopy(self.demo["goal"]),
            deepcopy(self.demo["student_profile"]),
            deepcopy(self.library),
            client,
            options=options,
        )

        runtime = session["agent_runtime"]
        receipt = runtime["last_agent_loop"]
        self.assertEqual(receipt["schema"], "teaching_skill_miner.teacher_agent_loop.v1")
        self.assertEqual(receipt["status"], "route_ready")
        self.assertEqual(receipt["steps"], 2)
        self.assertEqual(receipt["tool_call_count"], 3)
        self.assertEqual(receipt["selected_skill_id"], "skill_diagnostic_questioning")
        self.assertEqual(runtime["agent_loop_run_count"], 1)
        self.assertEqual(runtime["agent_loop_tool_call_count"], 3)
        harness_trace = runtime["last_harness_trace"]
        self.assertEqual(
            harness_trace["schema"], "teaching_skill_miner.agent_harness.v1"
        )
        self.assertEqual(harness_trace["status"], "completed")
        self.assertEqual(harness_trace["tool_call_count"], 3)
        self.assertTrue(harness_trace["trace_sha256"])
        self.assertEqual(
            session["current_action"]["primary_skill"]["skill_id"],
            "skill_diagnostic_questioning",
        )
        encoded = json.dumps(receipt, ensure_ascii=False)
        for forbidden in ("messages", "learner_response", "raw_media"):
            self.assertNotIn(forbidden, encoded)

    def test_live_turn_persists_event_derived_lifecycle_receipt(self) -> None:
        client = _deepseek_script_client(
            [
                _selection_plan(),
                _route_ready(),
                _live_plan(
                    signal="not_observed",
                    skill_id="skill_diagnostic_questioning",
                    action_type="probe_prior_knowledge",
                    message="请先说出一个完成当前目标所需的前置概念。",
                ),
                _selection_plan(primary_skill_id="skill_diagnostic_questioning"),
                _route_ready("继续核对前置概念。"),
                _live_plan(
                    signal="partial",
                    skill_id="skill_diagnostic_questioning",
                    action_type="probe_prior_knowledge",
                    message="请再补充一个前置概念，并说明它的作用。",
                ),
            ]
        )
        options = LiveAgentOptions(
            agent_loop_enabled=True,
            maximum_agent_steps=4,
            agent_loop_model_retries=0,
        )
        session = start_live_teacher_agent_session(
            deepcopy(self.demo["goal"]),
            deepcopy(self.demo["student_profile"]),
            deepcopy(self.library),
            client,
            options=options,
        )
        updated = advance_live_teacher_agent_session(
            session,
            learner_response="我知道状态保存每一级的走法数，但还说不清它如何由前一步转移得到。",
            client=client,
            options=options,
        )
        receipt = updated["history"][-1]["turn_lifecycle"]
        self.assertEqual(receipt["lifecycle_mode"], "event_derived")
        self.assertEqual(receipt["status"], "completed")
        event_names = [event["event"] for event in receipt["events"]]
        self.assertEqual(event_names[:2], ["observe", "assess"])
        self.assertEqual(event_names[-2:], ["act", "commit"])
        self.assertIn(event_names.count("route"), {1, 2})
        self.assertEqual(receipt["commit_round"], updated["round"])
        self.assertTrue(receipt["verification"]["checks"])
        encoded = json.dumps(receipt, ensure_ascii=False)
        for forbidden in ("learner_response", "raw_media", "消息", "我知道状态"):
            self.assertNotIn(forbidden, encoded)

    def test_live_outbound_context_exposes_current_answer_to_route_loop(self) -> None:
        session = self._session()
        learner_response = "我把每一级的编号当成了状态值，邮箱 test@example.com 不应被发送。"
        context = build_layered_context(session, learner_response)
        model = _CapturingModel([_selection_plan(), _action()])

        result = run_teaching_agent_loop(
            session,
            model,
            outbound_context=context,
        )

        self.assertEqual(result["status"], "action_ready")
        payload = json.loads(model.messages[0][1]["content"])
        observation = payload["teaching_context"]["current_observation"]
        self.assertIn("我把每一级的编号当成了状态值", observation["learner_response"])
        self.assertNotIn("test@example.com", observation["learner_response"])
        # The route loop must use the projected action from the layered context,
        # never a raw session field that could bypass the privacy projection.
        projected_action = context["current_plan"].get("current_action", {})
        expected_message = (
            projected_action.get("teacher_message", "")
            if isinstance(projected_action, dict)
            else ""
        )
        self.assertEqual(
            payload["teaching_context"]["current_action"]["message"],
            expected_message[:500],
        )

    def test_live_followup_binds_loop_summary_without_copying_learner_text(self) -> None:
        initial_client = _deepseek_script_client(
            [
                _selection_plan(),
                _route_ready(),
                _live_plan(
                    signal="not_observed",
                    skill_id="skill_diagnostic_questioning",
                    action_type="probe_prior_knowledge",
                    message="请先说出一个前置概念。",
                ),
            ]
        )
        options = LiveAgentOptions(
            agent_loop_enabled=True,
            state_first_route_adjudication_enabled=True,
            maximum_agent_steps=4,
            agent_loop_model_retries=0,
        )
        initial = start_live_teacher_agent_session(
            deepcopy(self.demo["goal"]),
            deepcopy(self.demo["student_profile"]),
            deepcopy(self.library),
            initial_client,
            options=options,
        )
        followup_client = _deepseek_script_client(
            [
                _selection_plan(
                    primary_skill_id="skill_concrete_example_bridge",
                    supporting_skill_ids=["skill_wait_and_elicit"],
                    next_focus="conceptual",
                ),
                _route_ready("部分掌握，转入最小例子建立概念连接。"),
                _live_plan(
                    signal="partial",
                    skill_id="skill_concrete_example_bridge",
                    action_type="present_minimal_example",
                    message="请用一个两步小例子说明状态如何变化。",
                ),
            ]
        )
        learner_text = "dp[i] 只表示第 i 级，没有说明保存的走法数。"
        updated = advance_live_teacher_agent_session(
            initial,
            learner_response=learner_text,
            client=followup_client,
            options=options,
        )

        self.assertEqual(updated["round"], 1)
        summary = updated["history"][-1]["agent_loop_summary"]
        self.assertEqual(summary, updated["agent_runtime"]["last_agent_loop"])
        self.assertEqual(summary["selected_skill_id"], "skill_concrete_example_bridge")
        self.assertGreaterEqual(summary["tool_call_count"], 3)
        encoded = json.dumps(summary, ensure_ascii=False)
        for forbidden in ("messages", "learner_response", "raw_media"):
            self.assertNotIn(forbidden, encoded)
        # The learner response remains in the normal teaching history, but is
        # not duplicated into the loop's durable public receipt.
        self.assertEqual(updated["history"][-1]["learner_text"], learner_text)

    def test_post_assessment_loop_sees_current_signal_and_owns_final_route(self) -> None:
        captured: list[dict[str, object]] = []
        responses = deque(
            [
                _live_plan(
                    signal="not_observed",
                    skill_id="skill_concrete_example_bridge",
                    action_type="present_minimal_example",
                    message="请先看一个最小例子。",
                ),
                _selection_plan(),
                _route_ready("已依据首轮未观察状态选择诊断 Skill。"),
                _live_plan(
                    signal="partial",
                    skill_id="skill_concrete_example_bridge",
                    action_type="present_minimal_example",
                    message="请继续看一个最小例子。",
                ),
                _selection_plan(
                    primary_skill_id="skill_concrete_example_bridge",
                    supporting_skill_ids=[],
                    next_focus="conceptual",
                ),
                _route_ready("本轮部分理解已验证，先补足最小例子表征。"),
            ]
        )

        def transport(
            _url: str,
            _headers: dict[str, str],
            payload: bytes,
            _timeout: float,
        ) -> tuple[int, bytes]:
            captured.append(json.loads(payload))
            content = responses.popleft()
            return 200, _envelope(content, response_id=f"post-{len(captured)}")

        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
            api_key="route-key",
            transport=transport,
        )
        options = LiveAgentOptions(
            agent_loop_enabled=True,
            agent_loop_post_assessment_enabled=True,
            state_first_route_adjudication_enabled=True,
            maximum_agent_steps=4,
            agent_loop_model_retries=0,
        )
        session = start_live_teacher_agent_session(
            deepcopy(self.demo["goal"]),
            deepcopy(self.demo["student_profile"]),
            deepcopy(self.library),
            client,
            options=options,
        )
        updated = advance_live_teacher_agent_session(
            session,
            learner_response="我知道状态保存走法数，但还没有说明它和转移的关系。",
            client=client,
            options=options,
        )

        self.assertFalse(responses)
        # Call order is now assess -> route loop -> route_ready for each turn.
        followup_route_request = captured[4]
        loop_payload = json.loads(followup_route_request["messages"][1]["content"])
        current_signal = loop_payload["teaching_context"]["student"][
            "understanding_signal"
        ]
        self.assertEqual(current_signal["label"], "partial")
        self.assertEqual(
            current_signal["source"], "provisional_validated_diagnosis"
        )
        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_concrete_example_bridge",
        )
        authority = updated["history"][-1]["model_trace"]["route_authority"]
        self.assertEqual(authority["mode"], "post_assessment_agent_loop")
        self.assertTrue(authority["current_turn_provisional_state_used"])
        self.assertTrue(authority["route_consistent"])
        self.assertEqual(authority["route_replan_count"], 0)
        self.assertEqual(
            authority["loop_selected_skill_id"], authority["final_skill_id"]
        )
        self.assertNotIn(
            "route-key", json.dumps(updated, ensure_ascii=False)
        )

    def test_safe_loop_route_stays_first_when_normalization_fallback_contains_it(
        self,
    ) -> None:
        """A valid loop proposal must outrank generic related-answer fallbacks."""

        initial_plan = _live_plan(
            signal="not_observed",
            skill_id="skill_diagnostic_questioning",
            action_type="probe_prior_knowledge",
            message="请先说出一个前置概念。",
        )
        followup_plan = _live_plan(
            signal="partial",
            skill_id="skill_socratic_understanding_check",
            action_type="socratic_comprehension_probe",
            message="请说明你的判断依据。",
        )
        # The short, term-like learner response below is not explicit
        # confusion, so the server conservatively normalizes a model
        # ``confused`` label to ``partial / related_but_not_answer``.  The
        # generic fallback order for that normalization starts with Socratic,
        # then context, then concrete-example.  The loop's concrete-example
        # proposal is nevertheless contract-safe and must remain authoritative.
        followup_plan["diagnosis"]["signal"] = "confused"
        followup_plan["diagnosis"]["answer_alignment"] = "ambiguous"
        responses = [
            initial_plan,
            _selection_plan(),
            _route_ready("首轮路由已完成。"),
            followup_plan,
            _selection_plan(
                primary_skill_id="skill_concrete_example_bridge",
                supporting_skill_ids=[],
                next_focus="conceptual",
            ),
            _route_ready("当前回答只给出术语，先保留最小例子桥接。"),
        ]
        client = _deepseek_script_client(responses)
        options = LiveAgentOptions(
            agent_loop_enabled=True,
            agent_loop_post_assessment_enabled=True,
            state_first_route_adjudication_enabled=True,
            maximum_agent_steps=4,
            agent_loop_model_retries=0,
        )
        session = start_live_teacher_agent_session(
            deepcopy(self.demo["goal"]),
            deepcopy(self.demo["student_profile"]),
            deepcopy(self.library),
            client,
            options=options,
        )
        updated = advance_live_teacher_agent_session(
            session,
            learner_response="状态",
            client=client,
            options=options,
        )

        action = updated["current_action"]
        self.assertEqual(
            action["primary_skill"]["skill_id"],
            "skill_concrete_example_bridge",
        )
        authority = updated["history"][-1]["model_trace"]["route_authority"]
        self.assertTrue(authority["route_consistent"])
        self.assertEqual(
            authority["loop_selected_skill_id"],
            authority["final_skill_id"],
        )
        reasons = updated["history"][-1]["deepseek_assessment"][
            "normalization_reasons"
        ]
        self.assertIn("agent_loop_route_preserved_after_safe_normalization", reasons)
        self.assertNotIn("agent_loop_route_rejected_by_server_contract", reasons)

    def test_inapplicable_loop_route_is_repaired_by_state_first_stage(self) -> None:
        """A loop route for an initial-only Skill must not become the final action."""

        options = LiveAgentOptions(
            agent_loop_enabled=True,
            state_first_route_adjudication_enabled=True,
            maximum_agent_steps=4,
            agent_loop_model_retries=0,
        )
        initial_client = _deepseek_script_client(
            [
                _selection_plan(),
                _route_ready(),
                _live_plan(
                    signal="not_observed",
                    skill_id="skill_diagnostic_questioning",
                    action_type="probe_prior_knowledge",
                    message="请先说出一个前置概念。",
                ),
            ]
        )
        session = start_live_teacher_agent_session(
            deepcopy(self.demo["goal"]),
            deepcopy(self.demo["student_profile"]),
            deepcopy(self.library),
            initial_client,
            options=options,
        )
        followup_client = _deepseek_script_client(
            [
                _selection_plan(primary_skill_id="skill_diagnostic_questioning"),
                _route_ready("已读取状态；当前回答已进入部分理解阶段。"),
                _live_plan(
                    signal="partial",
                    skill_id="skill_diagnostic_questioning",
                    action_type="probe_prior_knowledge",
                    message="请再说一个前置概念。",
                ),
            ]
        )

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="我知道状态会保存结果，但还说不清转移怎么来。",
            client=followup_client,
            options=options,
        )
        action = updated["current_action"]
        audit = action["action_provenance"]["route_adjudication"]
        self.assertTrue(audit["enabled"])
        self.assertNotEqual(
            action["primary_skill"]["skill_id"], "skill_diagnostic_questioning"
        )
        self.assertIn(
            "agent_loop_route_repaired_by_state_first",
            updated["history"][-1]["deepseek_assessment"]["normalization_reasons"],
        )
        self.assertFalse(action["agent_loop_route_applied"])

    def test_live_final_planner_failure_keeps_loop_receipt_on_rule_fallback(self) -> None:
        client = _deepseek_script_client(
            [
                _selection_plan(),
                _route_ready(),
            ]
        )

        # Exhausting the scripted transport at the final planner boundary is
        # intentionally converted into a DeepSeek failure; live mode must keep
        # the already-completed route/tool receipt while materializing its safe
        # deterministic action.
        original_transport = client._transport

        def fail_after_loop(
            url: str,
            headers: dict[str, str],
            payload: bytes,
            timeout: float,
        ) -> tuple[int, bytes]:
            try:
                return original_transport(url, headers, payload, timeout)
            except IndexError as exc:
                raise DeepSeekClientError("final planner unavailable") from exc

        client._transport = fail_after_loop
        options = LiveAgentOptions(
            agent_loop_enabled=True,
            maximum_agent_steps=4,
            agent_loop_model_retries=0,
            fallback_to_rules=True,
        )
        session = start_live_teacher_agent_session(
            deepcopy(self.demo["goal"]),
            deepcopy(self.demo["student_profile"]),
            deepcopy(self.library),
            client,
            options=options,
        )

        receipt = session["agent_runtime"]["last_agent_loop"]
        self.assertIsInstance(receipt, dict)
        self.assertEqual(receipt["status"], "route_ready")
        self.assertEqual(receipt["tool_call_count"], 3)
        self.assertEqual(
            session["current_action"]["decision_origin"],
            "deterministic_safety_fallback",
        )
        encoded = json.dumps(receipt, ensure_ascii=False)
        for forbidden in ("messages", "learner_response", "raw_media"):
            self.assertNotIn(forbidden, encoded)


if __name__ == "__main__":
    unittest.main()
