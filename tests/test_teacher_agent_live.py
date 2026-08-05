from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import unittest

import jsonschema

from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig
from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.teacher_agent import validate_session
from teaching_skill_miner.teacher_agent_live import (
    ADAPTIVE_OBSERVATION_LIMIT,
    LiveAgentOptions,
    _update_adaptive_student_profile_candidates,
    advance_live_teacher_agent_session,
    live_session_view,
    parse_skill_command,
    start_live_teacher_agent_session,
    stop_live_teacher_agent_session,
)


ROOT = Path(__file__).resolve().parents[1]


def _plan(
    *,
    signal: str,
    confidence: float,
    skill_id: str,
    support: list[str] | None = None,
    misconception_tag: str | None = None,
    message: str = "请先说明你判断这一步的依据是什么？",
) -> dict:
    return {
        "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
        "diagnosis": {
            "signal": signal,
            "confidence": confidence,
            "diagnosis_reason": "学生回答提供了可定位的当前理解证据。",
            "evidence_excerpt": "状态只看上一步",
            "misconception_tag": misconception_tag,
            "misconception_description": "忽略了另一条状态转移路径" if misconception_tag else "",
            "resolved_misconception_tags": [],
            "response_quality": "partial" if signal != "not_observed" else "empty",
            "engagement_level": "medium" if signal != "not_observed" else "unknown",
            "needs_human_review": False,
        },
        "decision": {
            "primary_skill_id": skill_id,
            "supporting_skill_ids": support or [],
            "selection_reason": "根据当前误解和目标依赖选择一个主 Skill。",
            "next_focus": "conceptual",
        },
        "teacher_action": {
            "type": "ask_one_question",
            "message": message,
            "expected_signal": "学生能定位第一处错误并说明原因。",
        },
        "stop_recommendation": {"should_stop": False, "reason": ""},
    }


def _client(plans: list[dict]) -> DeepSeekClient:
    queue = deque(plans)

    def transport(_url: str, _headers: dict, _payload: bytes, _timeout: float):
        content = queue.popleft()
        envelope = {
            "id": "live_test",
            "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 80},
        }
        return 200, json.dumps(envelope).encode()

    return DeepSeekClient(
        DeepSeekConfig(allow_remote_student_data=True),
        api_key="secret-test-key",
        transport=transport,
    )


class LiveTeacherAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
        cls.demo = read_json(ROOT / "data/teacher_agent_demo_input.json")

    def test_live_agent_assesses_routes_composes_and_generates(self) -> None:
        client = _client(
            [
                _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                    support=["skill_wait_and_elicit"],
                    message="开始前，请说出动态规划依赖的一个前置概念。",
                ),
                _plan(
                    signal="misconception",
                    confidence=0.92,
                    skill_id="skill_misconception_contrast",
                    support=["skill_wait_and_elicit", "skill_confidence_support"],
                    misconception_tag="missing_transition",
                ),
            ]
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        validate_session(session)
        jsonschema.Draft202012Validator(
            read_json(ROOT / "schema/teacher_agent_live_session.schema.json")
        ).validate(session)
        self.assertEqual(session["agent_runtime"]["model"], "deepseek-v4-flash")
        self.assertEqual(session["agent_runtime"]["model_call_count"], 1)
        self.assertEqual(
            session["current_action"]["decision_origin"],
            "deepseek_v4_flash_constrained",
        )
        self.assertEqual(len(session["current_action"]["supporting_skills"]), 1)

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="状态只需要看上一步。",
            client=client,
        )
        validate_session(updated)
        jsonschema.Draft202012Validator(
            read_json(ROOT / "schema/teacher_agent_live_session.schema.json")
        ).validate(updated)
        self.assertEqual(
            updated["student_state"]["understanding_signal"]["label"],
            "misconception",
        )
        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_misconception_contrast",
        )
        self.assertEqual(len(updated["current_action"]["supporting_skills"]), 2)
        self.assertEqual(updated["history"][-1]["structured_signal"]["source"], "deepseek_v4_flash")
        self.assertEqual(updated["history"][-1]["deepseek_assessment"]["confidence"], 0.92)
        self.assertEqual(updated["student_state"]["interaction_statistics"]["attempt_count"], 1)
        self.assertFalse(updated["claim_boundary"]["free_text_answer_grading_established"])
        observation = updated["student_profile"]["adaptive_observations"][0]
        self.assertEqual(observation["round"], 1)
        self.assertEqual(observation["status"], "candidate_unconfirmed")
        self.assertEqual(
            observation["source"], "deepseek_v4_flash_validated_diagnosis"
        )
        self.assertEqual(
            observation["candidate"]["misconception_tag"], "missing_transition"
        )
        self.assertEqual(
            observation["candidate"]["next_focus"], "conceptual"
        )
        self.assertEqual(
            updated["student_profile"]["adaptive_summary"][
                "total_observation_count"
            ],
            1,
        )
        self.assertEqual(
            live_session_view(updated)["adaptive_student_profile"]["summary"],
            updated["student_profile"]["adaptive_summary"],
        )

    def test_manual_skill_override_is_auditable(self) -> None:
        client = _client(
            [
                _plan(signal="not_observed", confidence=0.0, skill_id="skill_diagnostic_questioning"),
                _plan(signal="partial", confidence=0.8, skill_id="skill_concrete_example_bridge"),
            ]
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        updated = advance_live_teacher_agent_session(
            session,
            learner_response="我知道一点，但还不确定。",
            client=client,
            manual_skill_id="skill_socratic_understanding_check",
        )
        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_socratic_understanding_check",
        )
        self.assertTrue(updated["current_action"]["manual_override_applied"])

    def test_invalid_manual_skill_is_rejected_without_consuming_a_turn(self) -> None:
        client = _client(
            [
                _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                )
            ]
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        with self.assertRaisesRegex(Exception, "manual Skill"):
            advance_live_teacher_agent_session(
                session,
                learner_response="这轮不应被消费。",
                client=client,
                manual_skill_id="unknown_skill",
            )
        self.assertEqual(session["round"], 0)
        self.assertEqual(session["history"], [])

    def test_allowed_skill_subset_recomputes_library_integrity(self) -> None:
        primary_ids = [
            skill["skill_id"]
            for skill in self.library["skills"]
            if skill["role"] != "support"
        ][:5]
        client = _client(
            [
                _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id=primary_ids[0],
                )
            ]
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            client,
            allowed_skill_ids=primary_ids,
        )
        validate_session(session)
        selected_primary_ids = {
            skill["skill_id"]
            for skill in session["skill_library"]["skills"]
            if skill["role"] != "support"
        }
        self.assertEqual(selected_primary_ids, set(primary_ids))
        self.assertNotIn("content_sha256", session["skill_library"])

    def test_initial_history_is_redacted_and_included_in_remote_context(self) -> None:
        captured: dict[str, str] = {}
        plan = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )

        def transport(_url: str, _headers: dict, payload: bytes, _timeout: float):
            body = json.loads(payload)
            captured["prompt"] = body["messages"][1]["content"]
            envelope = {
                "id": "history_redaction_test",
                "choices": [{"message": {"content": json.dumps(plan)}}],
            }
            return 200, json.dumps(envelope).encode()

        profile = deepcopy(self.demo["student_profile"])
        profile["conversation_history"] = [
            {
                "response": "请联系 learner@example.com，我还不理解。",
                "signal": "confused",
                "focus_dimension": "conceptual",
            }
        ]
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True),
            api_key="secret-test-key",
            transport=transport,
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"], profile, self.library, client
        )
        self.assertNotIn("learner@example.com", captured["prompt"])
        self.assertIn("[REDACTED_EMAIL]", captured["prompt"])
        self.assertTrue(session["current_action"]["privacy_trace"]["redaction_applied"])

    def test_fabricated_evidence_excerpt_falls_back_to_redacted_current_answer(self) -> None:
        calls: list[dict] = []
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        turn = _plan(
            signal="partial",
            confidence=0.8,
            skill_id="skill_concrete_example_bridge",
        )
        turn["diagnosis"]["evidence_excerpt"] = "学生从未说过的伪造证据"
        plans = deque([initial, turn])

        def transport(_url: str, _headers: dict, payload: bytes, _timeout: float):
            calls.append(json.loads(payload))
            plan = plans.popleft()
            envelope = {
                "id": "evidence_grounding_test",
                "choices": [{"message": {"content": json.dumps(plan)}}],
            }
            return 200, json.dumps(envelope).encode()

        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True),
            api_key="secret-test-key",
            transport=transport,
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        response = "我的邮箱是 learner@example.com，我理解了一部分。"
        updated = advance_live_teacher_agent_session(
            session, learner_response=response, client=client
        )
        assessment = updated["history"][-1]["deepseek_assessment"]
        evidence = assessment["evidence_excerpt"]
        self.assertNotEqual(evidence, "学生从未说过的伪造证据")
        self.assertIn("[REDACTED_EMAIL]", evidence)
        self.assertNotIn("learner@example.com", evidence)
        self.assertNotIn(
            "learner@example.com", json.dumps(updated["history"][-1]["model_trace"])
        )
        self.assertNotIn(
            "learner@example.com", json.dumps(updated["history"][-1]["privacy_trace"])
        )
        self.assertNotIn("learner@example.com", json.dumps(calls[-1]))
        adaptive = updated["student_profile"]["adaptive_observations"][-1]
        self.assertNotIn("learner@example.com", json.dumps(adaptive))
        self.assertIn("[REDACTED_EMAIL]", adaptive["evidence"]["excerpt"])

    def test_low_confidence_candidate_requires_review_without_overwriting_profile(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        turn = _plan(
            signal="partial",
            confidence=0.2,
            skill_id="skill_concrete_example_bridge",
        )
        profile_before = deepcopy(self.demo["student_profile"])
        client = _client([initial, turn])
        session = start_live_teacher_agent_session(
            self.demo["goal"], profile_before, self.library, client
        )
        updated = advance_live_teacher_agent_session(
            session, learner_response="我只理解了其中一部分。", client=client
        )
        observation = updated["student_profile"]["adaptive_observations"][-1]
        self.assertTrue(observation["evidence"]["needs_human_review"])
        self.assertEqual(observation["evidence"]["review_reasons"], ["low_confidence"])
        self.assertEqual(observation["evidence"]["confidence"], 0.2)
        for field in ("learner_level", "preferences", "accessibility_needs"):
            self.assertEqual(
                updated["student_profile"][field],
                session["student_profile"][field],
            )
        self.assertFalse(
            updated["student_profile"]["adaptive_summary"][
                "teacher_provided_fields_overwritten"
            ]
        )
        validate_session(updated)

    def test_adaptive_observations_are_bounded(self) -> None:
        client = _client(
            [
                _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                )
            ]
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        diagnosis = _plan(
            signal="partial",
            confidence=0.8,
            skill_id="skill_concrete_example_bridge",
        )["diagnosis"]
        for round_number in range(1, ADAPTIVE_OBSERVATION_LIMIT + 2):
            session["round"] = round_number
            _update_adaptive_student_profile_candidates(
                session,
                diagnosis=diagnosis,
                next_focus="conceptual",
                minimum_review_confidence=0.35,
            )
        observations = session["student_profile"]["adaptive_observations"]
        summary = session["student_profile"]["adaptive_summary"]
        self.assertEqual(len(observations), ADAPTIVE_OBSERVATION_LIMIT)
        self.assertEqual(observations[0]["round"], 2)
        self.assertEqual(
            summary["total_observation_count"], ADAPTIVE_OBSERVATION_LIMIT + 1
        )
        self.assertEqual(
            summary["retained_observation_count"], ADAPTIVE_OBSERVATION_LIMIT
        )

    def test_invalid_model_output_uses_visible_rule_fallback(self) -> None:
        bad = _plan(signal="not_observed", confidence=0.0, skill_id="unknown_skill")
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            _client([bad]),
            options=LiveAgentOptions(fallback_to_rules=True),
        )
        self.assertEqual(session["agent_runtime"]["fallback_count"], 1)
        self.assertEqual(
            session["current_action"]["decision_origin"],
            "deterministic_safety_fallback",
        )
        self.assertEqual(session["student_profile"]["adaptive_observations"], [])

    def test_turn_fallback_does_not_create_adaptive_profile_candidate(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        invalid = _plan(
            signal="partial",
            confidence=0.8,
            skill_id="unknown_skill",
        )
        client = _client([initial, invalid])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        updated = advance_live_teacher_agent_session(
            session, learner_response="模型输出无效时应走规则回退。", client=client
        )
        self.assertEqual(updated["student_profile"]["adaptive_observations"], [])
        self.assertEqual(
            updated["history"][-1]["structured_signal"]["source"],
            "deterministic_safety_fallback",
        )
        validate_session(updated)

    def test_commands_and_manual_stop(self) -> None:
        command = parse_skill_command("/+skill 误解对比纠错", self.library)
        self.assertEqual(command["skill_id"], "skill_misconception_contrast")
        self.assertEqual(parse_skill_command("/auto", self.library)["command"], "auto")
        client = _client(
            [_plan(signal="not_observed", confidence=0.0, skill_id="skill_diagnostic_questioning")]
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        stopped = stop_live_teacher_agent_session(session, reason="teacher demo stop")
        self.assertEqual(stopped["status"], "terminated_unable")
        self.assertEqual(stopped["current_action"]["decision_origin"], "teacher_command")

    def test_model_stop_is_honored_only_after_guarded_no_progress(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        first = _plan(
            signal="confused",
            confidence=0.7,
            skill_id="skill_concrete_example_bridge",
        )
        first["diagnosis"]["needs_human_review"] = True
        first["stop_recommendation"] = {
            "should_stop": True,
            "reason": "one uncertain turn is insufficient",
        }
        second = _plan(
            signal="confused",
            confidence=0.7,
            skill_id="skill_engagement_recovery",
        )
        second["diagnosis"]["needs_human_review"] = True
        second["stop_recommendation"] = {
            "should_stop": True,
            "reason": "the learner needs human diagnosis",
        }
        client = _client([initial, first, second])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        session = advance_live_teacher_agent_session(
            session, learner_response="我还是不懂。", client=client
        )
        self.assertEqual(session["status"], "active")
        self.assertFalse(
            session["history"][-1]["model_stop_recommendation"]["honored"]
        )
        session = advance_live_teacher_agent_session(
            session, learner_response="还是完全不知道怎么开始。", client=client
        )
        self.assertEqual(session["status"], "terminated_unable")
        self.assertEqual(
            session["current_action"]["decision_origin"],
            "guarded_model_escalation",
        )
        self.assertTrue(
            session["history"][-1]["model_stop_recommendation"]["honored"]
        )


if __name__ == "__main__":
    unittest.main()
