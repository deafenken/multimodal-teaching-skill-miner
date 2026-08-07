from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import unittest

from jsonschema import validate as jsonschema_validate

from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig
from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.teacher_agent_live import (
    advance_live_teacher_agent_session,
    start_live_teacher_agent_session,
)


ROOT = Path(__file__).resolve().parents[1]


def _plan(signal: str, skill_id: str, confidence: float = 0.8) -> dict:
    action_types = {
        "skill_diagnostic_questioning": "probe_prior_knowledge",
        "skill_contextual_problem_setup": "establish_problem_context",
    }
    return {
        "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
        "diagnosis": {
            "signal": signal,
            "confidence": confidence,
            "answer_alignment": "not_applicable" if signal == "not_observed" else "aligned",
            "matched_concepts": [],
            "missing_concepts": [],
            "diagnosis_reason": "当前回答提供了一个可核验的学习信号。",
            "evidence_excerpt": (
                "递归是一个前置概念，它会把大问题分解为更小的同类问题。"
                if signal != "not_observed"
                else ""
            ),
            "misconception_tag": None,
            "misconception_description": "",
            "resolved_misconception_tags": [],
            "response_quality": "empty" if signal == "not_observed" else "complete",
            "engagement_level": "unknown" if signal == "not_observed" else "medium",
            "needs_human_review": False,
        },
        "decision": {
            "primary_skill_id": skill_id,
            "supporting_skill_ids": ["skill_wait_and_elicit"],
            "selection_reason": "根据当前学习信号选择下一步。",
            "next_focus": "prerequisite" if signal == "not_observed" else "conceptual",
        },
        "teacher_action": {
            "type": action_types[skill_id],
            "message": (
                "请说出一个前置概念，并说明它的作用。"
                if signal == "not_observed"
                else "请把刚才的直观理解与正式概念逐项对应，并说明理由。"
            ),
            "expected_signal": "学生能说出概念及其作用。",
            "question_contract": {
                "answer_type": "explanation",
                "target_concepts": ["递归", "重叠子问题"],
                "accepted_aliases": ["递归分解"],
                "success_criteria": ["说出概念并说明作用"],
            },
        },
        "stop_recommendation": {"should_stop": False, "reason": ""},
    }


class StudentModelLiveIntegrationTests(unittest.TestCase):
    def test_live_session_stores_estimate_and_schema_accepts_it(self) -> None:
        library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
        demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
        responses = deque([
            _plan("not_observed", "skill_diagnostic_questioning", 0.0),
            _plan("correct", "skill_contextual_problem_setup", 0.9),
        ])

        def transport(_url: str, _headers: dict, _payload: bytes, _timeout: float):
            envelope = {
                "id": "student_model_integration",
                "choices": [{"message": {"content": json.dumps(responses.popleft(), ensure_ascii=False)}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 20},
            }
            return 200, json.dumps(envelope, ensure_ascii=False).encode()

        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True),
            api_key="test-key",
            transport=transport,
        )
        session = start_live_teacher_agent_session(
            demo["goal"], demo["student_profile"], library, client
        )
        self.assertEqual(
            session["student_state"]["student_model"]["overall"]["evidence_count"],
            0,
        )
        updated = advance_live_teacher_agent_session(
            session,
            learner_response="递归是一个前置概念，它会把大问题分解为更小的同类问题。",
            client=client,
        )
        model = updated["student_state"]["student_model"]
        self.assertGreater(model["overall"]["evidence_count"], 0)
        self.assertEqual(model["last_update"]["evidence_id"], "session_history:r1:structured_signal")
        schema = read_json(ROOT / "schema/teacher_agent_live_session.schema.json")
        jsonschema_validate(updated, schema)


if __name__ == "__main__":
    unittest.main()
