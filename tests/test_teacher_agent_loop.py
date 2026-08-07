from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from teaching_skill_miner.teacher_agent import start_teacher_agent_session
from teaching_skill_miner.teacher_agent_loop import (
    PLAN_SCHEMA,
    TeachingAgentLoopOptions,
    run_teaching_agent_loop,
)


ROOT = Path(__file__).resolve().parents[1]


class _ScriptedModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, _messages: object) -> object:
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _tool(*calls: dict[str, object]) -> dict[str, object]:
    return {"schema": PLAN_SCHEMA, "kind": "tool_calls", "tool_calls": list(calls)}


def _call(name: str, arguments: dict[str, object] | None = None) -> dict[str, object]:
    return {"call_id": f"call_{name}", "name": name, "arguments": arguments or {}}


class TeachingAgentLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = json.loads((ROOT / "data/teacher_agent_demo_input.json").read_text())
        cls.library = json.loads(
            (ROOT / "data/teacher_agent_skill_library_v2.json").read_text()
        )

    def _session(self) -> dict[str, object]:
        return start_teacher_agent_session(
            deepcopy(self.demo["goal"]),
            deepcopy(self.demo["student_profile"]),
            deepcopy(self.library),
        )

    def test_model_can_plan_tools_then_emit_validated_teaching_action(self) -> None:
        model = _ScriptedModel(
            [
                _tool(
                    _call("inspect_student_state"),
                    _call("search_skills", {"signals": ["not_observed"]}),
                ),
                _tool(
                    _call(
                        "select_skills",
                        {
                            "primary_skill_id": "skill_diagnostic_questioning",
                            "supporting_skill_ids": ["skill_wait_and_elicit"],
                            "reason": "首轮需诊断前置知识",
                        },
                    ),
                    _call("set_next_focus", {"next_focus": "prerequisite"}),
                ),
                {
                    "schema": PLAN_SCHEMA,
                    "kind": "teaching_action",
                    "selected_skill_id": "skill_diagnostic_questioning",
                    "supporting_skill_ids": ["skill_wait_and_elicit"],
                    "next_focus": "prerequisite",
                    "action_type": "probe_prior_knowledge",
                    "message": "请用自己的话说出一个必要的前置概念，并举一个最小例子。",
                    "expected_signal": "学生能说明前置概念及其作用。",
                    "reason": "先确认已有知识，再决定是否进入状态转移。",
                },
            ]
        )

        result = run_teaching_agent_loop(self._session(), model)

        self.assertEqual(result["status"], "action_ready")
        self.assertFalse(result["deterministic_fallback"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["action"]["skill"]["skill_id"], "skill_diagnostic_questioning")
        self.assertEqual(result["action"]["action_type"], "probe_prior_knowledge")
        self.assertEqual(result["loop_state"]["next_focus"], "prerequisite")
        self.assertGreaterEqual(
            len([item for item in result["events"] if item["type"] == "tool_result"]),
            4,
        )

    def test_repeat_guard_produces_deterministic_fallback(self) -> None:
        model = _ScriptedModel(
            [
                _tool(_call("inspect_student_state")),
                _tool(_call("inspect_student_state")),
            ]
        )

        result = run_teaching_agent_loop(
            self._session(),
            model,
            options=TeachingAgentLoopOptions(max_repeated_tool_calls=1),
        )

        self.assertTrue(result["deterministic_fallback"])
        self.assertEqual(result["status"], "action_ready")
        self.assertTrue(
            any(
                item["type"] == "guard" and item["reason"] == "repeated_tool_call"
                for item in result["events"]
            )
        )

    def test_transient_model_error_retries_without_replaying_tools(self) -> None:
        model = _ScriptedModel(
            [
                TimeoutError("temporary"),
                _tool(
                    _call(
                        "select_skills",
                        {"primary_skill_id": "skill_diagnostic_questioning"},
                    )
                ),
                {
                    "schema": PLAN_SCHEMA,
                    "kind": "teaching_action",
                    "selected_skill_id": "skill_diagnostic_questioning",
                    "supporting_skill_ids": [],
                    "next_focus": "prerequisite",
                    "action_type": "probe_prior_knowledge",
                    "message": "请说出一个必要的前置概念。",
                    "expected_signal": "学生给出一个可核验概念。",
                    "reason": "从前置知识开始。",
                },
            ]
        )

        result = run_teaching_agent_loop(self._session(), model)

        self.assertFalse(result["deterministic_fallback"])
        self.assertEqual(model.calls, 3)
        self.assertTrue(any(item["type"] == "model_error" for item in result["events"]))
        self.assertEqual(result["action"]["skill"]["skill_id"], "skill_diagnostic_questioning")

    def test_unselected_model_action_cannot_bypass_skill_runtime(self) -> None:
        model = _ScriptedModel(
            [
                {
                    "schema": PLAN_SCHEMA,
                    "kind": "teaching_action",
                    "selected_skill_id": "skill_diagnostic_questioning",
                    "supporting_skill_ids": [],
                    "next_focus": "prerequisite",
                    "action_type": "probe_prior_knowledge",
                    "message": "请回答一个前置概念。",
                    "expected_signal": "学生回答。",
                    "reason": "test",
                }
            ]
        )

        result = run_teaching_agent_loop(self._session(), model)

        self.assertTrue(result["deterministic_fallback"])
        self.assertTrue(
            any(
                item["type"] == "guard"
                and item["reason"] == "action_without_runtime_skill_selection"
                for item in result["events"]
            )
        )

    def test_malformed_validated_action_fails_closed_without_key_error(self) -> None:
        model = _ScriptedModel([{"kind": "teaching_action"}])
        malformed = {
            "schema": PLAN_SCHEMA,
            "kind": "teaching_action",
            "message": "不完整动作",
        }
        with patch(
            "teaching_skill_miner.teacher_agent_loop._validate_plan",
            return_value=malformed,
        ):
            result = run_teaching_agent_loop(self._session(), model)

        self.assertEqual(result["status"], "action_ready")
        self.assertTrue(result["deterministic_fallback"])
        self.assertTrue(
            any(
                item["type"] == "fallback"
                and "action emitted before select_skills" in item["reason"]
                for item in result["events"]
            )
        )

    def test_termination_success_requires_local_readiness_tool(self) -> None:
        model = _ScriptedModel(
            [{"schema": PLAN_SCHEMA, "kind": "terminate", "outcome": "success", "reason": "完成"}]
        )

        result = run_teaching_agent_loop(self._session(), model)

        self.assertEqual(result["status"], "action_ready")
        self.assertTrue(result["deterministic_fallback"])
        self.assertTrue(
            any(
                item["reason"] == "success_requires_evaluate_termination"
                for item in result["events"]
                if item["type"] == "guard"
            )
        )

    def test_invalid_support_is_rejected_without_losing_primary_route(self) -> None:
        model = _ScriptedModel(
            [
                _tool(
                    _call(
                        "select_skills",
                        {
                            "primary_skill_id": "skill_diagnostic_questioning",
                            "supporting_skill_ids": [
                                "skill_diagnostic_questioning",
                                "skill_wait_and_elicit",
                            ],
                            "reason": "先诊断前置知识",
                        },
                    ),
                    _call("set_next_focus", {"next_focus": "prerequisite"}),
                ),
                {"schema": PLAN_SCHEMA, "kind": "route_ready", "reason": "路由完成"},
            ]
        )

        result = run_teaching_agent_loop(self._session(), model)

        self.assertEqual(result["status"], "route_ready")
        self.assertFalse(result["deterministic_fallback"])
        self.assertEqual(
            result["loop_state"]["supporting_skill_ids"],
            ["skill_wait_and_elicit"],
        )
        self.assertFalse(any(item["type"] == "tool_error" for item in result["events"]))

    def test_step_budget_keeps_last_validated_route(self) -> None:
        model = _ScriptedModel(
            [
                _tool(
                    _call(
                        "select_skills",
                        {"primary_skill_id": "skill_diagnostic_questioning"},
                    ),
                    _call("set_next_focus", {"next_focus": "prerequisite"}),
                ),
                _tool(_call("inspect_student_state")),
            ]
        )

        result = run_teaching_agent_loop(
            self._session(),
            model,
            options=TeachingAgentLoopOptions(max_steps=2),
        )

        self.assertEqual(result["status"], "route_ready")
        self.assertFalse(result["deterministic_fallback"])
        self.assertEqual(
            result["loop_state"]["selected_skill_id"],
            "skill_diagnostic_questioning",
        )

    def test_termination_tool_checks_every_mastery_threshold(self) -> None:
        model = _ScriptedModel(
            [
                _tool(
                    _call("evaluate_termination"),
                    _call(
                        "select_skills",
                        {"primary_skill_id": "skill_diagnostic_questioning"},
                    ),
                    _call("set_next_focus", {"next_focus": "prerequisite"}),
                ),
                {"schema": PLAN_SCHEMA, "kind": "route_ready", "reason": "尚未达标"},
            ]
        )

        result = run_teaching_agent_loop(self._session(), model)
        termination_result = next(
            item["result"]
            for item in result["events"]
            if item.get("tool") == "evaluate_termination" and item.get("ok") is True
        )

        self.assertFalse(termination_result["eligible"])
        self.assertEqual(
            set(termination_result["unmet_dimensions"]),
            {"prerequisite", "conceptual", "procedural", "transfer"},
        )
