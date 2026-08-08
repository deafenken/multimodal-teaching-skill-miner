from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from teaching_skill_miner.teacher_agent import (
    _refresh_integrity,
    start_teacher_agent_session,
)
from teaching_skill_miner.teacher_agent_loop import PLAN_SCHEMA, TeachingAgentLoopOptions
from teaching_skill_miner.teacher_agent_orchestration import (
    TeacherAgentOrchestrationError,
    TeacherAgentOrchestrationOptions,
    advance_teacher_agent_orchestration,
    build_turn_lifecycle_receipt,
    initialize_teacher_agent_orchestration,
    run_teacher_agent_orchestration,
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


def _selection() -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "kind": "tool_calls",
        "tool_calls": [
            {
                "call_id": "state",
                "name": "inspect_student_state",
                "arguments": {},
            },
            {
                "call_id": "select",
                "name": "select_skills",
                "arguments": {
                    "primary_skill_id": "skill_diagnostic_questioning",
                    "supporting_skill_ids": ["skill_wait_and_elicit"],
                    "reason": "先诊断前置知识",
                },
            },
            {
                "call_id": "focus",
                "name": "set_next_focus",
                "arguments": {"next_focus": "prerequisite"},
            },
        ],
    }


def _action() -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "kind": "teaching_action",
        "selected_skill_id": "skill_diagnostic_questioning",
        "supporting_skill_ids": ["skill_wait_and_elicit"],
        "next_focus": "prerequisite",
        "action_type": "probe_prior_knowledge",
        "message": "请说出一个必要的前置概念，并举一个最小例子。",
        "expected_signal": "学生能说明概念及其作用。",
        "reason": "获得可核验的前置知识证据。",
    }


class TeacherAgentOrchestrationTests(unittest.TestCase):
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

    def test_runs_explicit_five_phase_workflow(self) -> None:
        result = run_teacher_agent_orchestration(
            self._session(),
            _ScriptedModel([_selection(), _action()]),
        )

        self.assertEqual(result["status"], "waiting_for_learner")
        self.assertEqual(result["phase"], "reflect")
        self.assertEqual(
            [event["phase"] for event in result["events"]],
            ["goal", "plan", "execute", "verify", "reflect"],
        )
        self.assertEqual(result["verification"]["status"], "passed")
        self.assertEqual(
            result["output_action"]["selected_skill_id"],
            "skill_diagnostic_questioning",
        )
        self.assertEqual(result["uncertainty"]["level"], "low")
        self.assertFalse(result["uncertainty"]["needs_human_review"])
        encoded = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("previous_tool_results", encoded)
        self.assertNotIn("messages", encoded)

    def test_checkpoint_can_resume_after_goal_phase(self) -> None:
        session = self._session()
        initial = initialize_teacher_agent_orchestration(session)
        paused = advance_teacher_agent_orchestration(
            initial,
            session,
            _ScriptedModel([]),
        )
        self.assertEqual(paused["phase"], "plan")

        result = run_teacher_agent_orchestration(
            session,
            _ScriptedModel([_selection(), _action()]),
            checkpoint=paused,
        )

        self.assertEqual(result["status"], "waiting_for_learner")
        self.assertEqual(result["events"][0]["phase"], "goal")
        self.assertRegex(result["checkpoint_sha256"], r"^[0-9a-f]{64}$")

    def test_checkpoint_is_bound_to_exact_session_snapshot(self) -> None:
        session = self._session()
        checkpoint = initialize_teacher_agent_orchestration(session)
        changed = deepcopy(session)
        changed["round"] = int(changed["round"]) + 1

        with self.assertRaisesRegex(
            TeacherAgentOrchestrationError,
            "different teaching-session snapshot",
        ):
            run_teacher_agent_orchestration(
                changed,
                _ScriptedModel([]),
                checkpoint=checkpoint,
            )

    def test_fallback_reflects_then_replans_and_recovers(self) -> None:
        model = _ScriptedModel(
            [
                TimeoutError("first planning attempt failed"),
                TimeoutError("retry also failed"),
                _selection(),
                _action(),
            ]
        )

        result = run_teacher_agent_orchestration(
            self._session(),
            model,
            options=TeacherAgentOrchestrationOptions(max_replans=1),
        )

        self.assertEqual(result["status"], "waiting_for_learner")
        self.assertEqual(result["replan_count"], 1)
        self.assertEqual(result["cycle"], 2)
        self.assertTrue(
            any(
                event["phase"] == "reflect" and event["outcome"] == "replan"
                for event in result["events"]
            )
        )
        self.assertEqual(result["verification"]["status"], "passed")

    def test_exhausted_fallback_is_actionable_but_requires_review(self) -> None:
        result = run_teacher_agent_orchestration(
            self._session(),
            _ScriptedModel([TimeoutError("offline")]),
            options=TeacherAgentOrchestrationOptions(max_replans=0),
            loop_options=TeachingAgentLoopOptions(model_retries=0),
        )

        self.assertEqual(result["status"], "waiting_for_learner")
        self.assertTrue(result["reflection"]["replan_exhausted"])
        self.assertTrue(result["uncertainty"]["needs_human_review"])
        self.assertEqual(result["uncertainty"]["level"], "high")
        self.assertIsNotNone(result["output_action"])

    def test_success_requires_local_readiness_evidence(self) -> None:
        session = self._session()
        session["round"] = 3
        for dimension, threshold in session["goal"]["success_thresholds"].items():
            session["student_state"]["knowledge_mastery"][dimension] = threshold
        session["student_state"]["understanding_signal"] = {
            "label": "correct",
            "confidence": 0.9,
            "response_excerpt": "",
            "source": "test_teacher",
        }
        session = _refresh_integrity(session)
        model = _ScriptedModel(
            [
                {
                    "schema": PLAN_SCHEMA,
                    "kind": "tool_calls",
                    "tool_calls": [
                        {
                            "call_id": "ready",
                            "name": "evaluate_termination",
                            "arguments": {},
                        }
                    ],
                },
                {
                    "schema": PLAN_SCHEMA,
                    "kind": "terminate",
                    "outcome": "success",
                    "reason": "本地完成条件已满足",
                },
            ]
        )

        result = run_teacher_agent_orchestration(session, model)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["verification"]["status"], "completed")
        self.assertIsNone(result["output_action"])

    def test_live_receipt_uses_terminal_session_status(self) -> None:
        action = {
            "type": "terminate_success",
            "primary_skill": {"skill_id": "skill_learner_summary"},
            "supporting_skills": [],
            "next_focus": "transfer",
        }
        succeeded = self._session()
        succeeded["status"] = "succeeded"
        success_receipt = build_turn_lifecycle_receipt(
            succeeded,
            loop_trace={},
            plan=None,
            output_action=action,
        )
        self.assertEqual(success_receipt["status"], "completed")
        self.assertEqual(success_receipt["phases"][-1]["status"], "completed")

        unable = self._session()
        unable["status"] = "terminated_unable"
        unable_receipt = build_turn_lifecycle_receipt(
            unable,
            loop_trace={},
            plan=None,
            output_action={**action, "type": "terminate_unable"},
            verification_status="fallback",
            fallback_reason="maximum teaching rounds reached",
        )
        self.assertEqual(unable_receipt["status"], "handoff_required")
        self.assertEqual(
            unable_receipt["phases"][-1]["status"], "handoff_required"
        )

    def test_event_lifecycle_receipt_derives_a_real_commit_without_raw_text(self) -> None:
        session = self._session()
        action = {
            "type": "probe_prior_knowledge",
            "primary_skill": {"skill_id": "skill_diagnostic_questioning"},
            "supporting_skills": [{"skill_id": "skill_wait_and_elicit"}],
            "next_focus": "prerequisite",
        }
        receipt = build_turn_lifecycle_receipt(
            session,
            loop_trace={},
            plan=None,
            output_action=action,
            route_authority="state_first_policy",
            turn_outcome="commit",
            commit_round=0,
            lifecycle_events=[
                {
                    "event": "observe",
                    "observation_present": True,
                    "evidence_count": 1,
                    "source": "learner_turn",
                    "learner_text": "PRIVATE_LEARNER_TEXT",
                },
                {
                    "event": "assess",
                    "signal": "partial",
                    "confidence": 0.82,
                    "assessment_source": "validated_diagnosis",
                    "evidence_excerpt": "PRIVATE_ASSESSMENT_EXCERPT",
                },
                {
                    "event": "route",
                    "selected_skill_id": "skill_diagnostic_questioning",
                    "supporting_skill_ids": ["skill_wait_and_elicit"],
                },
                {
                    "event": "act",
                    "action": {
                        **action,
                        "message": "PRIVATE_TEACHER_MESSAGE",
                    },
                },
                {"event": "commit", "committed": True, "round": 0},
            ],
        )

        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["turn_outcome"], "commit")
        self.assertEqual(receipt["verification"]["status"], "verified")
        self.assertEqual(receipt["route_authority"], "state_first_policy")
        self.assertEqual(receipt["commit_round"], 0)
        self.assertFalse(receipt["replan"]["occurred"])
        self.assertEqual(
            [event["event"] for event in receipt["events"]],
            ["observe", "assess", "route", "act", "commit"],
        )
        self.assertEqual(receipt["events"][-1]["status"], "committed")
        self.assertTrue(receipt["claim_boundary"]["commit_established"])
        encoded = json.dumps(receipt, ensure_ascii=False)
        for forbidden in (
            "PRIVATE_LEARNER_TEXT",
            "PRIVATE_ASSESSMENT_EXCERPT",
            "PRIVATE_TEACHER_MESSAGE",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_event_lifecycle_commit_rejects_asserted_pass_without_facts(self) -> None:
        session = self._session()
        action = {
            "type": "probe_prior_knowledge",
            "primary_skill": {"skill_id": "skill_diagnostic_questioning"},
            "supporting_skills": [],
            "next_focus": "prerequisite",
        }
        receipt = build_turn_lifecycle_receipt(
            session,
            loop_trace={},
            plan=None,
            output_action=action,
            route_authority="state_first_policy",
            turn_outcome="commit",
            commit_round=0,
            lifecycle_events=[
                {"event": "observe", "status": "passed", "observed": False},
                {"event": "assess", "signal": "partial", "confidence": 0.8},
                {
                    "event": "route",
                    "selected_skill_id": "skill_diagnostic_questioning",
                },
                {
                    "event": "act",
                    "selected_skill_id": "skill_diagnostic_questioning",
                    "action_type": "probe_prior_knowledge",
                    "action_materialized": True,
                },
                {"event": "commit", "committed": True, "round": 0},
            ],
        )

        self.assertEqual(receipt["status"], "blocked")
        self.assertEqual(receipt["turn_outcome"], "invalid")
        self.assertEqual(receipt["verification"]["status"], "rejected")
        self.assertIn(
            "observation_fact_missing", receipt["verification"]["failures"]
        )
        self.assertFalse(receipt["claim_boundary"]["commit_established"])
        self.assertNotIn("passed", json.dumps(receipt, ensure_ascii=False))

    def test_event_lifecycle_records_abort_without_free_text_reason(self) -> None:
        receipt = build_turn_lifecycle_receipt(
            self._session(),
            loop_trace={},
            plan=None,
            lifecycle_events=[
                {
                    "event": "observe",
                    "observation_present": True,
                    "learner_text": "PRIVATE_ABORTED_TURN",
                },
                {
                    "event": "abort",
                    "aborted": True,
                    "reason": "PRIVATE_EXCEPTION_DETAIL",
                    "reason_codes": ["client_cancelled"],
                },
            ],
            turn_outcome="abort",
        )

        self.assertEqual(receipt["status"], "aborted")
        self.assertEqual(receipt["turn_outcome"], "abort")
        self.assertEqual(receipt["events"][-1]["status"], "aborted")
        self.assertEqual(
            receipt["events"][-1]["reason_codes"], ["client_cancelled"]
        )
        encoded = json.dumps(receipt, ensure_ascii=False)
        self.assertNotIn("PRIVATE_ABORTED_TURN", encoded)
        self.assertNotIn("PRIVATE_EXCEPTION_DETAIL", encoded)

    def test_event_lifecycle_exposes_route_replan_authority(self) -> None:
        action = {
            "type": "present_minimal_example",
            "primary_skill": {"skill_id": "skill_concrete_example_bridge"},
            "supporting_skills": [],
            "next_focus": "conceptual",
        }
        receipt = build_turn_lifecycle_receipt(
            self._session(),
            loop_trace={},
            plan=None,
            output_action=action,
            route_authority="state_first_policy",
            turn_outcome="commit",
            commit_round=0,
            lifecycle_events=[
                {"event": "observe", "observation_present": True},
                {"event": "assess", "signal": "partial", "confidence": 0.8},
                {
                    "event": "route",
                    "selected_skill_id": "skill_diagnostic_questioning",
                },
                {
                    "event": "act",
                    "selected_skill_id": "skill_diagnostic_questioning",
                    "action_type": "probe_prior_knowledge",
                    "action_materialized": True,
                },
                {
                    "event": "route",
                    "selected_skill_id": "skill_concrete_example_bridge",
                    "replan_count": 1,
                    "reason_codes": ["state_changed_after_verification"],
                },
                {
                    "event": "act",
                    "selected_skill_id": "skill_concrete_example_bridge",
                    "action_type": "present_minimal_example",
                    "action_materialized": True,
                },
                {"event": "commit", "round": 0},
            ],
        )

        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["replan_count"], 1)
        self.assertTrue(receipt["replan"]["occurred"])
        self.assertTrue(receipt["replan"]["route_changed"])
        self.assertEqual(
            receipt["replan"]["initial_skill_id"],
            "skill_diagnostic_questioning",
        )
        self.assertEqual(
            receipt["replan"]["final_skill_id"],
            "skill_concrete_example_bridge",
        )
        self.assertIn(
            "state_changed_after_verification",
            receipt["replan"]["reason_codes"],
        )


if __name__ == "__main__":
    unittest.main()
