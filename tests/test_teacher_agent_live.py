from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import jsonschema

from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig
from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.teacher_agent import (
    TeacherAgentError,
    _refresh_integrity,
    validate_session,
)
from teaching_skill_miner.teacher_agent_live import (
    ADAPTIVE_OBSERVATION_LIMIT,
    LIVE_PROMPT_VERSION,
    LiveAgentOptions,
    LiveTeacherAgentError,
    _contract_safe_retarget_action,
    _learner_evidence_validation_context,
    _materialize_contract_safe_fallback_action,
    _skill_prompt_view,
    _update_adaptive_student_profile_candidates,
    _validated_learner_evidence,
    advance_live_teacher_agent_session,
    live_session_view,
    parse_skill_command,
    start_live_teacher_agent_session,
    stop_live_teacher_agent_session,
)


ROOT = Path(__file__).resolve().parents[1]
ACTION_TYPES = {
    str(skill["skill_id"]): str(skill["action_type"])
    for skill in read_json(ROOT / "data/teacher_agent_skill_library_v2.json")["skills"]
}


class LearnerVisualEvidenceContractTests(unittest.TestCase):
    def test_formula_audit_metadata_survives_bounded_validation(self) -> None:
        evidence = _validated_learner_evidence(
            [
                {
                    "schema": "teaching_skill_miner.local_visual_evidence.v1",
                    "source_modality": "image",
                    "display_name": "formula.png",
                    "mime_type": "image/png",
                    "byte_size": 128,
                    "content_sha256": "a" * 64,
                    "engine": "apple_vision_local_objc_cli",
                    "status": "recognized",
                    "recognized_text": "x = (-b + sqrt(b^2 - 4ac)) / 2a",
                    "confidence": 0.99,
                    "formula_like_text_detected": True,
                    "formula_accuracy_established": False,
                    "extractor_fallback_used": False,
                    "needs_student_confirmation": False,
                    "raw_media_retained": False,
                    "remote_media_sent": False,
                }
            ]
        )

        self.assertEqual(len(evidence), 1)
        item = evidence[0]
        self.assertTrue(item["formula_like_text_detected"])
        self.assertFalse(item["formula_accuracy_established"])
        self.assertTrue(item["needs_student_confirmation"])
        self.assertIn("not_formula_correctness", item["confidence_semantics"])

    def test_multiline_ocr_quote_is_grounded_after_whitespace_normalization(
        self,
    ) -> None:
        library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
        demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
        contract = {
            "answer_type": "example",
            "target_concepts": ["任一必要前置概念"],
            "accepted_aliases": [],
            "success_criteria": ["说出概念并给出说明作用的最小例子"],
        }
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="请说出一个必要前置概念，并用最小例子说明作用。",
            question_contract=contract,
        )
        correct = _plan(
            signal="correct",
            confidence=0.96,
            skill_id="skill_socratic_understanding_check",
            answer_alignment="aligned",
        )
        correct["diagnosis"]["evidence_excerpt"] = (
            "Recursion is a prerequisite concept. Fibonacci contains repeated "
            "smaller subproblems. Dynamic programming stores and reuses each state result."
        )
        client = _client([initial, correct])
        session = start_live_teacher_agent_session(
            demo["goal"], demo["student_profile"], library, client
        )
        _install_server_question_contract(
            session,
            contract,
            message="请说出一个必要前置概念，并用最小例子说明作用。",
            expected_signal="学生说出概念、例子与作用。",
        )

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="",
            learner_evidence=[
                _visual_evidence(
                    "Recursion is a prerequisite concept.\n"
                    "Fibonacci contains repeated smaller subproblems.\n"
                    "Dynamic programming stores and reuses each state result."
                )
            ],
            client=client,
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "correct")
        self.assertTrue(assessment["model_evidence_excerpt_grounded"])
        self.assertEqual(
            assessment["evidence_binding_source"],
            "model_excerpt_current_response_substring",
        )
        self.assertIn("\n", assessment["evidence_excerpt"])
        self.assertNotIn(
            "high_impact_diagnosis_without_bound_evidence_downgraded",
            assessment["normalization_reasons"],
        )

    def test_generic_image_references_are_not_independent_typed_answers(self) -> None:
        library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
        demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        session = start_live_teacher_agent_session(
            demo["goal"], demo["student_profile"], library, _client([initial])
        )
        evidence = _validated_learner_evidence(
            [
                _visual_evidence(
                    "x+1",
                    confidence=0.2,
                    needs_student_confirmation=True,
                    formula_like_text_detected=True,
                )
            ]
        )

        for typed in (
            "照片这样显然是对的",
            "我这个答案就是正确的，见图片",
            "如图，我做完了",
            "看附件，我认为没问题",
        ):
            with self.subTest(typed=typed):
                _trusted, _exact, confirmation_required = (
                    _learner_evidence_validation_context(session, typed, evidence)
                )
                self.assertTrue(confirmation_required)

    def test_attachment_meta_statements_do_not_bypass_visual_confirmation(
        self,
    ) -> None:
        library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
        demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        session = start_live_teacher_agent_session(
            demo["goal"], demo["student_profile"], library, _client([initial])
        )
        _install_server_question_contract(
            session,
            {
                "answer_type": "open",
                "target_concepts": ["动态规划状态转移"],
                "accepted_aliases": ["dp"],
                "success_criteria": ["写出一个可核验的状态或转移依据"],
            },
            message="请写出你的状态或转移依据。",
            expected_signal="学生写出一个可核验的状态或转移依据。",
        )
        evidence = _validated_learner_evidence(
            [
                _visual_evidence(
                    "x+1",
                    confidence=0.2,
                    needs_student_confirmation=True,
                    formula_like_text_detected=True,
                )
            ]
        )

        for typed in (
            "我把计算过程拍下来了",
            "我上传的是我的答案",
            "我发的就是完整过程",
            "这是我刚拍的",
        ):
            with self.subTest(typed=typed):
                trusted, exact, confirmation_required = (
                    _learner_evidence_validation_context(session, typed, evidence)
                )
                self.assertEqual(trusted, [])
                self.assertEqual(exact, [])
                self.assertTrue(confirmation_required)

    def test_typed_formula_is_trusted_even_when_attached_ocr_needs_confirmation(
        self,
    ) -> None:
        library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
        demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        session = start_live_teacher_agent_session(
            demo["goal"], demo["student_profile"], library, _client([initial])
        )
        evidence = _validated_learner_evidence(
            [
                _visual_evidence(
                    "uncertain OCR",
                    confidence=0.2,
                    needs_student_confirmation=True,
                )
            ]
        )

        for answer_type in ("open", "worked_step"):
            _install_server_question_contract(
                session,
                {
                    "answer_type": answer_type,
                    "target_concepts": ["递推式"],
                    "accepted_aliases": [],
                    "success_criteria": ["键入一个可核验的计算步骤"],
                },
                message="请键入一个可核验的计算步骤。",
                expected_signal="学生键入一个可核验的计算步骤。",
            )
            for typed in ("x+1", "先算 x+1", "a[n]=a[n-1]+1"):
                with self.subTest(answer_type=answer_type, typed=typed):
                    trusted, exact, confirmation_required = (
                        _learner_evidence_validation_context(session, typed, evidence)
                    )
                    self.assertEqual(trusted, [typed])
                    self.assertEqual(exact, [typed])
                    self.assertFalse(confirmation_required)

    def test_independent_subject_question_is_not_treated_as_image_deictic(self) -> None:
        library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
        demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        session = start_live_teacher_agent_session(
            demo["goal"], demo["student_profile"], library, _client([initial])
        )
        _install_server_question_contract(
            session,
            {
                "answer_type": "open",
                "target_concepts": ["递归终止条件"],
                "accepted_aliases": ["终止条件"],
                "success_criteria": ["说明终止条件的作用"],
            },
            message="递归终止条件有什么作用？",
            expected_signal="学生说明终止条件的作用。",
        )
        evidence = _validated_learner_evidence(
            [
                _visual_evidence(
                    "uncertain OCR",
                    confidence=0.2,
                    needs_student_confirmation=True,
                )
            ]
        )

        for typed in (
            "为什么递归需要终止条件？",
            "终止条件为什么不能省略？",
        ):
            with self.subTest(typed=typed):
                trusted, exact, confirmation_required = (
                    _learner_evidence_validation_context(session, typed, evidence)
                )
                self.assertEqual(trusted, [typed])
                self.assertEqual(exact, [typed])
                self.assertFalse(confirmation_required)

        trusted, exact, confirmation_required = _learner_evidence_validation_context(
            session,
            "这个图为什么对？",
            evidence,
        )
        self.assertEqual(trusted, [])
        self.assertEqual(exact, [])
        self.assertTrue(confirmation_required)


def _plan(
    *,
    signal: str,
    confidence: float,
    skill_id: str,
    support: list[str] | None = None,
    misconception_tag: str | None = None,
    message: str = "请先说明你判断这一步的依据是什么？",
    answer_alignment: str | None = None,
    matched_concepts: list[str] | None = None,
    missing_concepts: list[str] | None = None,
    question_contract: dict | None = None,
    action_type: str | None = None,
) -> dict:
    if answer_alignment is None:
        answer_alignment = {
            "not_observed": "not_applicable",
            "correct": "aligned",
            "partial": "partially_aligned",
            "misconception": "contradicted",
            "confused": "ambiguous",
            "no_response": "no_response",
        }[signal]
    expected_signal = "学生能定位第一处错误并说明原因。"
    return {
        "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
        "diagnosis": {
            "signal": signal,
            "confidence": confidence,
            "answer_alignment": answer_alignment,
            "matched_concepts": matched_concepts or [],
            "missing_concepts": missing_concepts or [],
            "diagnosis_reason": "学生回答提供了可定位的当前理解证据。",
            "evidence_excerpt": "状态只看上一步",
            "misconception_tag": misconception_tag,
            "misconception_description": "忽略了另一条状态转移路径"
            if misconception_tag
            else "",
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
            "type": action_type or ACTION_TYPES.get(skill_id, "ask_one_question"),
            "message": message,
            "expected_signal": expected_signal,
            "question_contract": question_contract
            or {
                "answer_type": "explanation",
                "target_concepts": ["动态规划的状态与转移"],
                "accepted_aliases": [],
                "success_criteria": [expected_signal],
            },
        },
        "stop_recommendation": {"should_stop": False, "reason": ""},
    }


def _client(plans: list[dict]) -> DeepSeekClient:
    queue = deque(plans)

    def transport(_url: str, _headers: dict, _payload: bytes, _timeout: float):
        content = queue.popleft()
        envelope = {
            "id": "live_test",
            "choices": [
                {"message": {"content": json.dumps(content, ensure_ascii=False)}}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 80},
        }
        return 200, json.dumps(envelope).encode()

    return DeepSeekClient(
        DeepSeekConfig(allow_remote_student_data=True),
        api_key="secret-test-key",
        transport=transport,
    )


def _visual_evidence(
    recognized_text: str,
    *,
    status: str = "recognized",
    confidence: float = 0.99,
    needs_student_confirmation: bool = False,
    formula_like_text_detected: bool = False,
) -> dict[str, object]:
    return {
        "schema": "teaching_skill_miner.local_visual_evidence.v1",
        "source_modality": "image",
        "source_kind": "learner_answer_attachment",
        "display_name": "synthetic-answer.png",
        "mime_type": "image/png",
        "byte_size": 128,
        "content_sha256": "a" * 64,
        "engine": "synthetic_test_extractor",
        "status": status,
        "recognized_text": recognized_text,
        "confidence": confidence,
        "confidence_semantics": ("engine_native_ocr_heuristic_not_formula_correctness"),
        "formula_like_text_detected": formula_like_text_detected,
        "formula_accuracy_established": False,
        "extractor_fallback_used": False,
        "needs_student_confirmation": needs_student_confirmation,
        "raw_media_retained": False,
        "remote_media_sent": False,
        "remote_representation": "bounded_redacted_ocr_text_only",
    }


def _install_server_question_contract(
    session: dict,
    contract: dict,
    *,
    message: str,
    expected_signal: str,
) -> None:
    """Install a trusted server-owned contract for grading-boundary tests."""

    teacher_action = session["current_action"]["teacher_action"]
    teacher_action["message"] = message
    teacher_action["expected_signal"] = expected_signal
    teacher_action["question_contract"] = deepcopy(contract)
    _refresh_integrity(session)


class LiveTeacherAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
        cls.demo = read_json(ROOT / "data/teacher_agent_demo_input.json")

    def test_skill_prompt_and_materializer_share_library_action_contract(self) -> None:
        prompt_skills = {
            item["skill_id"]: item for item in _skill_prompt_view(self.library)
        }
        primary_skills = [
            skill for skill in self.library["skills"] if skill["role"] != "support"
        ]
        expected_targets = {
            "skill_diagnostic_questioning": ["任一与当前教学目标相关的必要前置概念"],
            "skill_contextual_problem_setup": [
                "情境中的关键对象或信息",
                "需要完成的目标",
            ],
            "skill_concrete_example_bridge": [
                "例子中的关键部分",
                "关键部分与当前概念的联系",
            ],
            "skill_retrieval_review": ["一个曾接触过的相关概念"],
            "skill_stepwise_scaffolding": [
                "第一步要处理的对象或信息",
                "先做这一步的理由",
            ],
            "skill_concept_mapping": ["例子对象与形式概念的对应", "关系与目标的对应"],
            "skill_socratic_understanding_check": ["判断依据", "反例或适用边界"],
            "skill_self_explanation": ["回答依据", "一个可核验条件"],
            "skill_misconception_contrast": ["最小反例", "失效条件", "局部修正"],
            "skill_practice_feedback": [
                self.demo["goal"]["concept"],
                "练习中的关键步骤与依据",
            ],
            "skill_transfer_check": [
                self.demo["goal"]["concept"],
                "新情境适用条件",
                "迁移第一步",
            ],
            "skill_learner_summary": [
                self.demo["goal"]["concept"],
                "适用条件",
                "检查方法",
                "失效边界",
            ],
            "skill_engagement_recovery": ["一个可确定关键词或具体卡点"],
        }
        self.assertEqual(len(primary_skills), 13)
        for skill in primary_skills:
            with self.subTest(skill_id=skill["skill_id"]):
                prompt_skill = prompt_skills[skill["skill_id"]]
                self.assertEqual(prompt_skill["action_type"], skill["action_type"])
                self.assertEqual(
                    prompt_skill["message_template"], skill["message_template"]
                )
                self.assertTrue(prompt_skill["direct_answer_prohibited"])
                action_type, message, expected, _reason, contract = (
                    _contract_safe_retarget_action(
                        skill["skill_id"],
                        {"goal": self.demo["goal"]},
                        prior_targets=["STALE_PRIOR_TARGET_MUST_NOT_LEAK"],
                        prior_aliases=["STALE_PRIOR_ALIAS_MUST_NOT_LEAK"],
                    )
                )
                self.assertEqual(action_type, skill["action_type"])
                self.assertTrue(message.strip())
                self.assertTrue(expected.strip())
                self.assertTrue(contract["success_criteria"])
                self.assertEqual(
                    contract["target_concepts"], expected_targets[skill["skill_id"]]
                )
                self.assertNotIn(
                    "STALE_PRIOR_TARGET_MUST_NOT_LEAK",
                    contract["target_concepts"],
                )
                self.assertNotIn(
                    "STALE_PRIOR_ALIAS_MUST_NOT_LEAK",
                    contract["accepted_aliases"],
                )

    def test_all_materializers_remain_domain_neutral_for_a_humanities_goal(
        self,
    ) -> None:
        humanities_goal = deepcopy(self.demo["goal"])
        humanities_goal.update(
            {
                "concept": "诗歌意象与情感表达",
                "objective": "学生能结合诗句解释意象如何表达情感。",
                "materials": {
                    "example": "月落乌啼霜满天。",
                    "practice": "指出诗句中的一个意象并说明它营造的氛围。",
                    "transfer_task": "比较另一首诗中相似意象的不同表达效果。",
                },
            }
        )
        banned_algorithm_assumptions = (
            "子情况",
            "前驱",
            "状态转移",
            "记录的量",
            "变化量",
        )
        for skill in self.library["skills"]:
            if skill["role"] == "support":
                continue
            with self.subTest(skill_id=skill["skill_id"]):
                _action_type, message, expected, _reason, contract = (
                    _contract_safe_retarget_action(
                        skill["skill_id"],
                        {"goal": humanities_goal},
                        prior_targets=["动态规划状态"],
                        prior_aliases=["dp"],
                    )
                )
                rendered = json.dumps(
                    {"message": message, "expected": expected, "contract": contract},
                    ensure_ascii=False,
                )
                for phrase in banned_algorithm_assumptions:
                    self.assertNotIn(phrase, rendered)

    def test_humanities_goal_completes_live_start_and_turn_without_stem_leakage(
        self,
    ) -> None:
        goal = deepcopy(self.demo["goal"])
        goal.update(
            {
                "concept": "诗歌意象与情感表达",
                "objective": "学生能结合诗句解释意象如何表达情感。",
                "knowledge_components": ["诗歌意象", "情感表达"],
                "materials": {
                    "example": "月落乌啼霜满天。",
                    "practice": "指出诗句中的一个意象并说明它营造的氛围。",
                    "transfer_task": "比较另一首诗中相似意象的不同表达效果。",
                },
            }
        )
        response = "阅读诗歌前要理解意象，例如月亮常能营造思乡氛围。"
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            support=["skill_wait_and_elicit"],
        )
        next_turn = _plan(
            signal="partial",
            confidence=0.84,
            skill_id="skill_concrete_example_bridge",
            support=["skill_wait_and_elicit"],
            answer_alignment="partially_aligned",
        )
        next_turn["diagnosis"]["evidence_excerpt"] = response.rstrip("。")
        client = _client([initial, next_turn])

        session = start_live_teacher_agent_session(
            goal, self.demo["student_profile"], self.library, client
        )
        validate_session(session)
        initial_action = session["current_action"]
        self.assertEqual(
            initial_action["primary_skill"]["skill_id"],
            "skill_diagnostic_questioning",
        )
        self.assertEqual(
            initial_action["teacher_action"]["question_contract"]["target_concepts"],
            ["任一与当前教学目标相关的必要前置概念"],
        )
        self.assertEqual(
            initial_action["composition_plan"]["support_execution"],
            {"skill_wait_and_elicit": "wait_contract_appended"},
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response=response, client=client
        )
        validate_session(updated)
        action = updated["current_action"]
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 0)
        self.assertEqual(
            action["primary_skill"]["skill_id"],
            "skill_concrete_example_bridge",
        )
        self.assertEqual(
            action["model_proposed_primary_skill_id"],
            "skill_concrete_example_bridge",
        )
        self.assertFalse(action["primary_skill_was_retargeted"])
        self.assertIn("月落乌啼霜满天", action["teacher_action"]["message"])
        self.assertIn("我会等你回答后再继续", action["teacher_action"]["message"])
        self.assertEqual(
            action["teacher_action"]["question_contract"]["target_concepts"],
            ["例子中的关键部分", "关键部分与当前概念的联系"],
        )
        self.assertEqual(
            action["composition_plan"]["support_execution"],
            {"skill_wait_and_elicit": "wait_contract_appended"},
        )
        rendered_action = json.dumps(action["teacher_action"], ensure_ascii=False)
        for phrase in ("动态规划", "dp[", "状态转移", "前驱", "子情况"):
            self.assertNotIn(phrase, rendered_action)

    def test_action_type_mismatch_is_materialized_as_selected_skill(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            action_type="generic_unbound_probe",
            message="请随便说点依据。",
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            _client([initial]),
        )

        self.assertEqual(session["agent_runtime"]["fallback_count"], 0)
        self.assertEqual(
            session["current_action"]["teacher_action"]["type"],
            "probe_prior_knowledge",
        )
        self.assertNotIn(
            "随便说点依据", session["current_action"]["teacher_action"]["message"]
        )
        self.assertIn(
            "teacher_action_type_mismatch_retargeted_to_primary_skill",
            session["initial_model_plan"]["diagnosis"]["normalization_reasons"],
        )
        action = session["current_action"]
        self.assertEqual(
            action["model_proposed_primary_skill_id"],
            "skill_diagnostic_questioning",
        )
        self.assertEqual(action["model_proposed_action_type"], "generic_unbound_probe")
        self.assertTrue(action["action_type_was_retargeted"])
        self.assertEqual(
            action["model_selection_reason"],
            "根据当前误解和目标依赖选择一个主 Skill。",
        )

    def test_matching_action_type_cannot_bypass_server_skill_materialization(
        self,
    ) -> None:
        leaked = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            action_type="probe_prior_knowledge",
            message=(
                "动态规划递推式为 dp[i]=min(dp[i-1],dp[i-2])+cost[i]。"
                "照此完整计算后，只需回复知道了。"
            ),
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            _client([leaked]),
        )

        serialized = json.dumps(session, ensure_ascii=False)
        self.assertEqual(session["agent_runtime"]["fallback_count"], 0)
        self.assertEqual(
            session["current_action"]["teacher_action"]["type"],
            "probe_prior_knowledge",
        )
        self.assertNotIn("min(dp[i-1],dp[i-2])+cost[i]", serialized)
        self.assertNotIn("只需回复知道了", serialized)
        self.assertIn(
            "前置概念",
            session["current_action"]["teacher_action"]["message"],
        )

    def test_direct_identity_declaration_fails_closed_before_remote_call(self) -> None:
        calls = 0

        def transport(_url: str, _headers: dict, _payload: bytes, _timeout: float):
            nonlocal calls
            calls += 1
            raise AssertionError("remote transport must not be called")

        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True),
            api_key="secret-test-key",
            transport=transport,
        )
        profile = deepcopy(self.demo["student_profile"])
        profile["contains_direct_identity"] = True
        profile["background_history"] = ["我是张三，学号12345678，来自海淀实验学校。"]

        with self.assertRaisesRegex(LiveTeacherAgentError, "direct identity"):
            start_live_teacher_agent_session(
                self.demo["goal"], profile, self.library, client
            )
        self.assertEqual(calls, 0)

    def test_identity_declaration_requires_a_json_boolean(self) -> None:
        for value in ("false", 0, 1, None, []):
            with self.subTest(value=value):
                profile = deepcopy(self.demo["student_profile"])
                profile["contains_direct_identity"] = value
                with self.assertRaisesRegex(
                    LiveTeacherAgentError, "must be a JSON boolean"
                ):
                    start_live_teacher_agent_session(
                        self.demo["goal"],
                        profile,
                        self.library,
                        _client([]),
                    )

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
        self.assertTrue(session["claim_boundary"]["learner_image_input_enabled"])
        self.assertEqual(
            session["claim_boundary"]["learner_image_processing"],
            "local_ephemeral_ocr_then_text_reasoning",
        )
        self.assertFalse(
            session["claim_boundary"]["raw_image_understanding_by_deepseek"]
        )
        self.assertEqual(session["agent_runtime"]["model"], "deepseek-v4-flash")
        self.assertEqual(session["agent_runtime"]["model_call_count"], 1)
        self.assertEqual(
            session["current_action"]["decision_origin"],
            "deepseek_v4_flash_constrained",
        )
        self.assertEqual(len(session["current_action"]["supporting_skills"]), 1)

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="状态只看上一步。",
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
        self.assertEqual(len(updated["current_action"]["supporting_skills"]), 1)
        self.assertEqual(
            updated["current_action"]["supporting_skills"][0]["skill_id"],
            "skill_wait_and_elicit",
        )
        self.assertEqual(
            updated["history"][-1]["structured_signal"]["source"], "deepseek_v4_flash"
        )
        self.assertEqual(
            updated["history"][-1]["deepseek_assessment"]["confidence"], 0.92
        )
        self.assertEqual(
            updated["student_state"]["interaction_statistics"]["attempt_count"], 1
        )
        self.assertFalse(
            updated["claim_boundary"]["free_text_answer_grading_established"]
        )
        observation = updated["student_profile"]["adaptive_observations"][0]
        self.assertEqual(observation["round"], 1)
        self.assertEqual(observation["status"], "candidate_unconfirmed")
        self.assertEqual(observation["source"], "deepseek_v4_flash_validated_diagnosis")
        self.assertEqual(
            observation["candidate"]["misconception_tag"], "missing_transition"
        )
        self.assertEqual(observation["candidate"]["next_focus"], "conceptual")
        self.assertEqual(
            updated["student_profile"]["adaptive_summary"]["total_observation_count"],
            1,
        )
        self.assertEqual(
            live_session_view(updated)["adaptive_student_profile"]["summary"],
            updated["student_profile"]["adaptive_summary"],
        )

    def test_initial_action_filters_support_skills_that_require_student_evidence(
        self,
    ) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            support=[
                "skill_confidence_support",
                "skill_minimal_hint",
                "skill_wait_and_elicit",
            ],
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            _client([initial]),
        )

        self.assertEqual(
            [
                item["skill_id"]
                for item in session["current_action"]["supporting_skills"]
            ],
            ["skill_wait_and_elicit"],
        )
        self.assertIn(
            "我会等你回答后再继续",
            session["current_action"]["teacher_action"]["message"],
        )
        self.assertEqual(
            session["current_action"]["supporting_skills"][0]["executed_as"],
            "wait_contract_appended",
        )
        action = session["current_action"]
        self.assertEqual(
            action["model_proposed_supporting_skill_ids"],
            [
                "skill_confidence_support",
                "skill_minimal_hint",
                "skill_wait_and_elicit",
            ],
        )
        self.assertEqual(
            action["composition_plan"]["supporting_skill_ids"],
            ["skill_wait_and_elicit"],
        )
        self.assertEqual(
            action["composition_plan"]["support_execution"],
            {"skill_wait_and_elicit": "wait_contract_appended"},
        )

    def test_support_skills_require_grounded_evidence_and_respect_repeat_limit(
        self,
    ) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            support=["skill_wait_and_elicit"],
        )
        supported_one = _plan(
            signal="partial",
            confidence=0.8,
            skill_id="skill_stepwise_scaffolding",
            support=["skill_minimal_hint", "skill_confidence_support"],
        )
        supported_two = deepcopy(supported_one)
        over_limit = deepcopy(supported_one)
        client = _client([initial, supported_one, supported_two, over_limit])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        first = advance_live_teacher_agent_session(
            session,
            learner_response="我不太确定，这一步是不是要比较所有前驱。",
            client=client,
        )
        self.assertEqual(
            {item["skill_id"] for item in first["current_action"]["supporting_skills"]},
            {"skill_minimal_hint"},
        )
        first_message = first["current_action"]["teacher_action"]["message"]
        self.assertIn("只给一个最小提示", first_message)
        self.assertNotIn("先保留你已经做出的具体尝试", first_message)
        self.assertEqual(
            {
                item["executed_as"]
                for item in first["current_action"]["supporting_skills"]
            },
            {"minimal_hint_scope_applied"},
        )

        second = advance_live_teacher_agent_session(
            first,
            learner_response="我还是不确定，这一步的前驱比较卡住了。",
            client=client,
        )
        self.assertEqual(
            {
                item["skill_id"]
                for item in second["current_action"]["supporting_skills"]
            },
            {"skill_minimal_hint"},
        )

        third = advance_live_teacher_agent_session(
            second,
            learner_response="我仍然没信心，这一步不知道怎么比较前驱。",
            client=client,
        )
        self.assertEqual(third["current_action"]["supporting_skills"], [])

    def test_primary_skill_declared_support_list_is_a_hard_allowlist(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        proposed = _plan(
            signal="partial",
            confidence=0.8,
            skill_id="skill_socratic_understanding_check",
            support=["skill_minimal_hint", "skill_wait_and_elicit"],
        )
        proposed["diagnosis"]["evidence_excerpt"] = "我认为要先保存子问题结果。"
        client = _client([initial, proposed])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="我认为要先保存子问题结果。",
            client=client,
        )

        action = updated["current_action"]
        self.assertEqual(
            [item["skill_id"] for item in action["supporting_skills"]],
            ["skill_wait_and_elicit"],
        )
        self.assertEqual(
            action["model_proposed_supporting_skill_ids"],
            ["skill_minimal_hint", "skill_wait_and_elicit"],
        )
        self.assertNotIn("只给一个最小提示", action["teacher_action"]["message"])

    def test_confidence_support_rejects_confusion_without_a_concrete_attempt(
        self,
    ) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        confused = _plan(
            signal="confused",
            confidence=0.9,
            skill_id="skill_stepwise_scaffolding",
            support=["skill_confidence_support"],
        )
        response = "我不知道状态。"
        confused["diagnosis"]["evidence_excerpt"] = response
        client = _client([initial, confused])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response=response, client=client
        )

        self.assertEqual(updated["current_action"]["supporting_skills"], [])

    def test_manual_skill_override_is_auditable(self) -> None:
        client = _client(
            [
                _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                ),
                _plan(
                    signal="partial",
                    confidence=0.8,
                    skill_id="skill_socratic_understanding_check",
                ),
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

    def test_manual_skill_mismatch_falls_back_instead_of_relabeling_action(
        self,
    ) -> None:
        client = _client(
            [
                _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                ),
                _plan(
                    signal="partial",
                    confidence=0.9,
                    skill_id="skill_concrete_example_bridge",
                    message="先看例子：把已经算出的结果放进数组。",
                ),
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

        action = updated["current_action"]
        self.assertEqual(action["decision_origin"], "deterministic_safety_fallback")
        self.assertNotEqual(
            action["primary_skill"]["skill_id"],
            "skill_socratic_understanding_check",
        )
        self.assertNotIn(
            "先看例子：把已经算出的结果放进数组。", action["teacher_action"]["message"]
        )
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 1)
        self.assertIn("did not honor", updated["agent_runtime"]["last_error"])

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
        privacy = session["current_action"]["privacy_trace"]
        self.assertTrue(privacy["redaction_applied"])
        self.assertTrue(privacy["known_pattern_identifiers_redacted"])
        self.assertEqual(privacy["raw_identity_fields_sent"], "not_established")
        self.assertTrue(privacy["residual_identity_risk"])

    def test_unlabeled_background_history_does_not_fabricate_a_student_signal(
        self,
    ) -> None:
        profile = deepcopy(self.demo["student_profile"])
        profile["conversation_history"] = []
        profile["background_history"] = [
            "学生先说状态定义，再提到边界；这一行仍是一个未标注背景。"
        ]
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )

        session = start_live_teacher_agent_session(
            self.demo["goal"], profile, self.library, _client([initial])
        )

        self.assertEqual(
            session["student_state"]["understanding_signal"]["label"],
            "not_observed",
        )
        self.assertEqual(
            session["student_state"]["knowledge_mastery"],
            profile["initial_mastery"],
        )
        prior = session["context_memory"]["fixed_context"][
            "teacher_provided_student_profile"
        ]["provided_prior_context"]
        self.assertEqual(prior["turns"], [])
        self.assertEqual(len(prior["unlabeled_notes"]), 1)
        self.assertEqual(
            prior["unlabeled_notes"][0]["label_status"],
            "unlabeled_background_not_state_evidence",
        )
        self.assertFalse(prior["unlabeled_notes_may_update_state"])
        self.assertTrue(
            any(
                item["source"] == "teacher_provided_unlabeled_background"
                for item in session["context_memory"]["evidence_ledger"]
            )
        )

    def test_initial_payload_excludes_primary_skills_that_require_observed_answers(
        self,
    ) -> None:
        captured: dict[str, object] = {}
        plan = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )

        def transport(_url: str, _headers: dict, payload: bytes, _timeout: float):
            body = json.loads(payload)
            captured.update(
                json.loads(body["messages"][1]["content"].split("\n", 1)[1])
            )
            envelope = {
                "id": "initial_skill_filter_test",
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

        available = captured["available_skills"]
        primary = [item for item in available if not item["is_support"]]
        self.assertTrue(primary)
        self.assertTrue(
            all("not_observed" in item["applicable_signals"] for item in primary)
        )
        self.assertNotIn(
            "skill_transfer_check", {item["skill_id"] for item in available}
        )
        self.assertTrue(
            captured["constraints"]["initial_action_must_use_not_observed_skill"]
        )
        self.assertEqual(session["agent_runtime"]["fallback_count"], 0)

    def test_prompt_uses_one_layered_context_and_full_v2_skill_contract(self) -> None:
        captured: list[dict] = []
        plans = deque(
            [
                _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                ),
                _plan(
                    signal="partial",
                    confidence=0.82,
                    skill_id="skill_concrete_example_bridge",
                ),
                _plan(
                    signal="partial",
                    confidence=0.78,
                    skill_id="skill_socratic_understanding_check",
                ),
            ]
        )

        def transport(_url: str, _headers: dict, payload: bytes, _timeout: float):
            body = json.loads(payload)
            prompt = body["messages"][1]["content"]
            captured.append(json.loads(prompt.split("\n", 1)[1]))
            envelope = {
                "id": "layered_context_test",
                "choices": [{"message": {"content": json.dumps(plans.popleft())}}],
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
        session = advance_live_teacher_agent_session(
            session,
            learner_response="我知道递归，但状态的定义还不完整。",
            client=client,
        )
        unique_response = "UNIQUE_LIVE_RESPONSE_91f3 我认为只需要记录上一步。"
        session = advance_live_teacher_agent_session(
            session,
            learner_response=unique_response,
            client=client,
        )

        payload = captured[-1]
        self.assertIn("teaching_context", payload)
        for obsolete in (
            "goal",
            "student_profile",
            "student_state",
            "goal_plan",
            "current_action",
            "relevant_history",
            "learner_response",
        ):
            self.assertNotIn(obsolete, payload)
        self.assertEqual(
            json.dumps(payload, ensure_ascii=False).count("UNIQUE_LIVE_RESPONSE_91f3"),
            1,
        )
        context = payload["teaching_context"]
        self.assertEqual(
            context["working_memory"]["current_knowledge_components"],
            ["动态规划的状态与转移"],
        )
        self.assertEqual(
            context["candidate_long_term_memory"]["status"],
            "candidate_unconfirmed",
        )
        self.assertFalse(
            context["candidate_long_term_memory"]["may_override_teacher_profile"]
        )
        self.assertEqual(
            context["candidate_long_term_memory"]["retained_for_context"], 1
        )
        diagnostic = next(
            item
            for item in payload["available_skills"]
            if item["skill_id"] == "skill_diagnostic_questioning"
        )
        self.assertEqual(diagnostic["max_repeat"], 2)
        self.assertEqual(diagnostic["preconditions"], ["目标已定义"])
        self.assertEqual(diagnostic["failure_transition"], "skill_retrieval_review")
        self.assertIn("前置维度已稳定达标", diagnostic["contraindications"])
        self.assertIn("首轮", diagnostic["applicable_when"])
        self.assertEqual(
            session["agent_runtime"]["prompt_version"],
            LIVE_PROMPT_VERSION,
        )
        current_question = context["current_plan"]["current_action"]
        self.assertTrue(current_question["question_id"])
        self.assertEqual(
            current_question["question_contract"]["grading_scope"],
            "current_question_only",
        )
        self.assertEqual(
            current_question["question_contract"],
            session["history"][-1]["action"]["teacher_action"]["question_contract"],
        )
        self.assertEqual(
            live_session_view(session)["context_memory"],
            session["context_memory"],
        )

    def test_initial_prerequisite_question_contract_cannot_copy_lesson_goal(
        self,
    ) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message=(
                "学习 动态规划的状态与转移 前，请用自己的话说出一个必要的"
                "前置概念，并举一个最小例子。"
            ),
            question_contract={
                "answer_type": "open",
                "target_concepts": ["动态规划的状态与转移"],
                "accepted_aliases": [],
                "success_criteria": ["学生能说明前置概念、作用及其与当前目标的联系。"],
            },
        )

        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            _client([initial]),
        )

        contract = session["current_action"]["teacher_action"]["question_contract"]
        self.assertEqual(contract["answer_type"], "example")
        self.assertEqual(
            contract["target_concepts"],
            ["任一与当前教学目标相关的必要前置概念"],
        )
        self.assertEqual(
            contract["success_criteria"],
            [
                "明确说出一个必要前置概念",
                "给出一个最小例子，说明该概念如何发挥作用",
            ],
        )
        self.assertNotIn(self.demo["goal"]["concept"], contract["target_concepts"])

    def test_primary_skill_execution_contract_blocks_stage_skipping(self) -> None:
        scenarios = (
            (
                "transfer",
                "skill_transfer_check",
                "skill_self_explanation",
                "transfer_readiness_not_met",
                "elicit_self_explanation",
                "为什么这样回答",
            ),
            (
                "summary",
                "skill_learner_summary",
                "skill_self_explanation",
                "understanding_or_transfer_check_not_completed",
                "elicit_self_explanation",
                "为什么这样回答",
            ),
            (
                "practice",
                "skill_practice_feedback",
                "skill_stepwise_scaffolding",
                "scaffolded_attempt_not_yet_completed",
                "guide_one_micro_step",
                "只做练习的第一步",
            ),
            (
                "concept mapping",
                "skill_concept_mapping",
                "skill_contextual_problem_setup",
                "concrete_example_not_yet_discussed",
                "establish_problem_context",
                "具体情境",
            ),
        )
        for (
            label,
            requested,
            expected,
            reason,
            action_type,
            message_fragment,
        ) in scenarios:
            with self.subTest(label=label):
                initial = _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                )
                stage_skip = _plan(
                    signal="correct",
                    confidence=0.9,
                    skill_id=requested,
                    answer_alignment="aligned",
                )
                learner_response = (
                    "递归会把大问题拆成相同结构的子问题，例如爬楼梯可以"
                    "由两个更小台阶状态组合。"
                )
                stage_skip["diagnosis"]["evidence_excerpt"] = learner_response
                client = _client([initial, stage_skip])
                session = start_live_teacher_agent_session(
                    self.demo["goal"],
                    self.demo["student_profile"],
                    self.library,
                    client,
                )

                updated = advance_live_teacher_agent_session(
                    session,
                    learner_response=learner_response,
                    client=client,
                )

                assessment = updated["history"][-1]["deepseek_assessment"]
                action = updated["current_action"]
                self.assertEqual(
                    action["primary_skill"]["skill_id"],
                    expected,
                )
                self.assertEqual(
                    action["teacher_action"]["type"],
                    action_type,
                )
                self.assertIn(
                    message_fragment,
                    action["teacher_action"]["message"],
                )
                self.assertEqual(action["model_proposed_primary_skill_id"], requested)
                self.assertEqual(
                    action["model_proposed_action_type"], ACTION_TYPES[requested]
                )
                self.assertTrue(action["primary_skill_was_retargeted"])
                self.assertTrue(action["action_type_was_retargeted"])
                self.assertEqual(
                    action["model_selection_reason"],
                    "根据当前误解和目标依赖选择一个主 Skill。",
                )
                self.assertNotEqual(
                    action["selection_reason"], action["model_selection_reason"]
                )
                self.assertIn(
                    f"primary_skill_contract_violation:{reason}",
                    assessment["normalization_reasons"],
                )
                self.assertEqual(updated["agent_runtime"]["fallback_count"], 0)

    def test_manual_transfer_skill_is_released_when_readiness_contract_fails(
        self,
    ) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        premature_transfer = _plan(
            signal="correct",
            confidence=0.9,
            skill_id="skill_transfer_check",
            answer_alignment="aligned",
        )
        premature_transfer["diagnosis"]["evidence_excerpt"] = (
            "递归能够把原问题拆成更小的同类子问题"
        )
        client = _client([initial, premature_transfer])
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            client,
        )

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="递归能够把原问题拆成更小的同类子问题。",
            client=client,
            manual_skill_id="skill_transfer_check",
        )

        action = updated["current_action"]
        self.assertTrue(action["manual_override_requested"])
        self.assertFalse(action["manual_override_applied"])
        self.assertNotEqual(action["primary_skill"]["skill_id"], "skill_transfer_check")
        self.assertEqual(action["teacher_action"]["type"], "elicit_self_explanation")
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 0)

    def test_auto_skill_outside_signal_contract_is_safely_retargeted(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        response = (
            "递归是前置概念；斐波那契会产生重复的更小子问题，"
            "动态规划把子问题结果保存为状态并复用。"
        )
        non_applicable = _plan(
            signal="correct",
            confidence=0.95,
            skill_id="skill_diagnostic_questioning",
            answer_alignment="aligned",
            matched_concepts=["递归", "重复子问题", "状态复用"],
        )
        non_applicable["diagnosis"]["evidence_excerpt"] = response
        client = _client([initial, non_applicable])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response=response, client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        action = updated["current_action"]
        self.assertEqual(assessment["signal"], "correct")
        self.assertIn(
            "model_skill_not_applicable_to_signal",
            assessment["normalization_reasons"],
        )
        self.assertNotEqual(
            action["primary_skill"]["skill_id"], "skill_diagnostic_questioning"
        )
        self.assertTrue(action["primary_skill_was_retargeted"])
        self.assertTrue(action["action_type_was_retargeted"])
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 0)

    def test_related_term_is_partial_not_confused_or_misconception(self) -> None:
        caching_contract = {
            "answer_type": "short_concept",
            "target_concepts": ["记忆化"],
            "accepted_aliases": ["缓存", "memoization"],
            "success_criteria": ["指出保存并复用已经计算结果的机制"],
        }
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message=(
                "动态规划里有一个与保存有关的词，例如把已经算过的结果存下来。"
                "你能想到那个词吗？"
            ),
            question_contract=caching_contract,
        )
        model_misread = _plan(
            signal="confused",
            confidence=0.84,
            skill_id="skill_concrete_example_bridge",
            answer_alignment="ambiguous",
            message="你提到了相关概念；请再区分转移规则和保存结果的机制。",
        )
        model_misread["diagnosis"]["evidence_excerpt"] = "状态转移方程"
        client = _client([initial, model_misread])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        _install_server_question_contract(
            session,
            caching_contract,
            message="保存并复用已经计算结果的机制叫什么？",
            expected_signal="学生给出保存并复用计算结果的机制名称。",
        )
        self.assertEqual(
            session["current_action"]["teacher_action"]["question_contract"][
                "target_concepts"
            ],
            ["记忆化"],
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response="状态转移方程", client=client
        )
        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "partial")
        self.assertEqual(assessment["answer_alignment"], "related_but_not_answer")
        self.assertIsNone(assessment["misconception_tag"])
        self.assertTrue(assessment["needs_human_review"])
        self.assertEqual(
            updated["history"][-1]["structured_signal"]["label"], "partial"
        )
        self.assertFalse(updated["student_state"]["misconceptions"])

    def test_exact_short_concept_contract_repairs_model_false_negative(self) -> None:
        contract = {
            "answer_type": "short_concept",
            "target_concepts": ["记忆化"],
            "accepted_aliases": ["缓存", "memoization"],
            "success_criteria": ["给出保存并复用计算结果的机制名称"],
        }
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="保存并复用已经计算结果的机制叫什么？",
            question_contract=contract,
        )
        false_negative = _plan(
            signal="misconception",
            confidence=0.2,
            skill_id="skill_misconception_contrast",
            misconception_tag="wrong_concept",
            answer_alignment="contradicted",
        )
        false_negative["diagnosis"]["evidence_excerpt"] = "缓存"
        client = _client([initial, false_negative])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        _install_server_question_contract(
            session,
            contract,
            message="保存并复用已经计算结果的机制叫什么？",
            expected_signal="学生给出保存并复用计算结果的机制名称。",
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response="答案是：缓存。", client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "correct")
        self.assertEqual(assessment["answer_alignment"], "aligned")
        self.assertEqual(assessment["confidence"], 1.0)
        self.assertEqual(assessment["model_raw_confidence"], 0.2)
        self.assertEqual(
            assessment["assessment_source"],
            "active_question_contract_exact_match",
        )
        self.assertIn("缓存", assessment["matched_concepts"])
        self.assertIsNone(assessment["misconception_tag"])
        self.assertFalse(updated["student_state"]["misconceptions"])
        self.assertEqual(
            updated["current_action"]["teacher_action"]["type"],
            "elicit_self_explanation",
        )
        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_self_explanation",
        )
        self.assertEqual(
            updated["current_action"]["teacher_action"]["question_contract"][
                "answer_type"
            ],
            "explanation",
        )

    def test_trusted_image_only_short_concept_repairs_model_false_negative(
        self,
    ) -> None:
        contract = {
            "answer_type": "short_concept",
            "target_concepts": ["记忆化"],
            "accepted_aliases": ["缓存", "memoization"],
            "success_criteria": ["给出保存并复用计算结果的机制名称"],
        }
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="保存并复用已经计算结果的机制叫什么？",
            question_contract=contract,
        )
        false_negative = _plan(
            signal="partial",
            confidence=0.55,
            skill_id="skill_socratic_understanding_check",
            answer_alignment="partially_aligned",
        )
        false_negative["diagnosis"]["evidence_excerpt"] = "缓存"
        client = _client([initial, false_negative])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        _install_server_question_contract(
            session,
            contract,
            message="保存并复用已经计算结果的机制叫什么？",
            expected_signal="学生给出保存并复用计算结果的机制名称。",
        )

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="",
            learner_evidence=[_visual_evidence("缓存")],
            client=client,
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "correct")
        self.assertEqual(assessment["confidence"], 1.0)
        self.assertEqual(
            assessment["assessment_source"],
            "active_question_contract_exact_match",
        )
        self.assertEqual(assessment["evidence_excerpt"], "缓存")
        self.assertIn(
            "exact_short_concept_match_overrode_model_label",
            assessment["normalization_reasons"],
        )
        self.assertEqual(updated["history"][-1]["learner_text"], "")
        self.assertEqual(
            updated["history"][-1]["multimodal_evidence"][0]["recognized_text"],
            "缓存",
        )
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 0)

    def test_low_numeric_confidence_cannot_establish_high_impact_diagnosis(
        self,
    ) -> None:
        contract = {
            "answer_type": "short_concept",
            "target_concepts": ["记忆化"],
            "accepted_aliases": ["缓存"],
            "success_criteria": ["给出机制名称"],
        }
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="保存并复用已经计算结果的机制叫什么？",
            question_contract=contract,
        )
        overconfident = _plan(
            signal="correct",
            confidence=0.95,
            skill_id="skill_self_explanation",
            answer_alignment="aligned",
            matched_concepts=["缓存"],
        )
        overconfident["diagnosis"]["evidence_excerpt"] = "缓存"
        client = _client([initial, overconfident])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        _install_server_question_contract(
            session,
            contract,
            message="保存并复用已经计算结果的机制叫什么？",
            expected_signal="学生给出保存并复用计算结果的机制名称。",
        )

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="",
            learner_evidence=[
                _visual_evidence(
                    "缓存",
                    status="recognized",
                    confidence=0.2,
                    needs_student_confirmation=False,
                )
            ],
            client=client,
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["model_raw_signal"], "correct")
        self.assertEqual(assessment["signal"], "partial")
        self.assertEqual(assessment["answer_alignment"], "ambiguous")
        self.assertEqual(assessment["confidence"], 0.0)
        self.assertTrue(assessment["needs_human_review"])
        self.assertIn(
            "visual_evidence_requires_student_confirmation",
            assessment["normalization_reasons"],
        )
        self.assertFalse(updated["student_state"]["misconceptions"])
        self.assertIn("OCR", updated["current_action"]["teacher_action"]["message"])
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 0)

    def test_typed_see_image_cannot_bypass_low_confidence_confirmation(self) -> None:
        contract = {
            "answer_type": "short_concept",
            "target_concepts": ["记忆化"],
            "accepted_aliases": ["缓存"],
            "success_criteria": ["给出机制名称"],
        }
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="保存并复用已经计算结果的机制叫什么？",
            question_contract=contract,
        )
        overconfident = _plan(
            signal="correct",
            confidence=0.99,
            skill_id="skill_socratic_understanding_check",
            support=["skill_wait_and_elicit"],
            answer_alignment="aligned",
            matched_concepts=["缓存"],
        )
        overconfident["diagnosis"]["evidence_excerpt"] = "答案见图"
        client = _client([initial, overconfident])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        _install_server_question_contract(
            session,
            contract,
            message="保存并复用已经计算结果的机制叫什么？",
            expected_signal="学生给出机制名称。",
        )
        mastery_before = deepcopy(session["student_state"]["knowledge_mastery"])

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="答案见图",
            learner_evidence=[
                _visual_evidence(
                    "缓存",
                    confidence=0.2,
                    needs_student_confirmation=True,
                )
            ],
            client=client,
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "partial")
        self.assertEqual(assessment["answer_alignment"], "ambiguous")
        self.assertEqual(assessment["confidence"], 0.0)
        self.assertIn(
            "visual_evidence_requires_student_confirmation",
            assessment["normalization_reasons"],
        )
        self.assertIn("OCR", assessment["diagnosis_reason"])
        self.assertEqual(
            updated["student_state"]["knowledge_mastery"], mastery_before
        )
        action = updated["current_action"]
        self.assertEqual(
            action["primary_skill"]["skill_id"], "skill_self_explanation"
        )
        self.assertEqual(
            action["teacher_action"]["type"], "elicit_self_explanation"
        )
        self.assertEqual(action["supporting_skills"], [])
        self.assertEqual(action["composition_plan"]["support_execution"], {})
        self.assertIn("用自己的话", action["teacher_action"]["message"])

    def test_unconfirmed_formula_cannot_trigger_exact_contract_match(self) -> None:
        contract = {
            "answer_type": "short_concept",
            "target_concepts": ["x+1"],
            "accepted_aliases": [],
            "success_criteria": ["写出下一项"],
        }
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="若当前是 x，下一项写什么？",
            question_contract=contract,
        )
        false_negative = _plan(
            signal="partial",
            confidence=0.7,
            skill_id="skill_socratic_understanding_check",
        )
        false_negative["diagnosis"]["evidence_excerpt"] = "x+1"
        client = _client([initial, false_negative])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        _install_server_question_contract(
            session,
            contract,
            message="若当前是 x，下一项写什么？",
            expected_signal="学生写出下一项。",
        )

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="",
            learner_evidence=[
                _visual_evidence(
                    "x+1",
                    confidence=0.99,
                    needs_student_confirmation=True,
                    formula_like_text_detected=True,
                )
            ],
            client=client,
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "partial")
        self.assertEqual(assessment["confidence"], 0.0)
        self.assertNotIn(
            "exact_short_concept_match_overrode_model_label",
            assessment["normalization_reasons"],
        )
        self.assertNotEqual(
            assessment["assessment_source"], "active_question_contract_exact_match"
        )

    def test_exact_contract_repairs_internally_inconsistent_correct_label(self) -> None:
        contract = {
            "answer_type": "short_concept",
            "target_concepts": ["记忆化"],
            "accepted_aliases": ["缓存"],
            "success_criteria": ["给出机制名称"],
        }
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            question_contract=contract,
        )
        inconsistent = _plan(
            signal="correct",
            confidence=0.2,
            skill_id="skill_concept_mapping",
            answer_alignment="related_but_not_answer",
        )
        inconsistent["diagnosis"]["evidence_excerpt"] = "记忆化"
        client = _client([initial, inconsistent])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        _install_server_question_contract(
            session,
            contract,
            message="保存并复用已经计算结果的机制叫什么？",
            expected_signal="学生给出机制名称。",
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response="记忆化", client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "correct")
        self.assertEqual(assessment["answer_alignment"], "aligned")
        self.assertEqual(assessment["confidence"], 1.0)
        self.assertIn("按正确处理", assessment["diagnosis_reason"])
        self.assertEqual(
            updated["current_action"]["teacher_action"]["type"],
            "elicit_self_explanation",
        )
        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_self_explanation",
        )

    def test_short_concept_nonmatch_repairs_model_false_positive(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="保存并复用已经计算结果的机制叫什么？",
            question_contract={
                "answer_type": "short_concept",
                "target_concepts": ["记忆化"],
                "accepted_aliases": ["缓存"],
                "success_criteria": ["给出机制名称"],
            },
        )
        false_positive = _plan(
            signal="correct",
            confidence=0.95,
            skill_id="skill_concept_mapping",
            answer_alignment="aligned",
        )
        false_positive["diagnosis"]["evidence_excerpt"] = "状态转移方程"
        client = _client([initial, false_positive])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        _install_server_question_contract(
            session,
            {
                "answer_type": "short_concept",
                "target_concepts": ["记忆化"],
                "accepted_aliases": ["缓存"],
                "success_criteria": ["给出机制名称"],
                "grading_scope": "current_question_only",
            },
            message="保存并复用已经计算结果的机制叫什么？",
            expected_signal="学生给出机制名称。",
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response="状态转移方程", client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["model_raw_signal"], "correct")
        self.assertEqual(assessment["signal"], "partial")
        self.assertEqual(assessment["answer_alignment"], "related_but_not_answer")
        self.assertIn(
            "short_concept_nonmatch_overrode_model_positive",
            assessment["normalization_reasons"],
        )

    def test_content_subject_not_understanding_is_not_self_reported_confusion(
        self,
    ) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="为什么训练集表现好、测试集表现差？",
        )
        correct = _plan(
            signal="correct",
            confidence=0.95,
            skill_id="skill_concept_mapping",
            answer_alignment="aligned",
        )
        response = "因为模型不理解数据里的噪声，只会记住训练样本，所以泛化能力不足。"
        correct["diagnosis"]["evidence_excerpt"] = response
        client = _client([initial, correct])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response=response, client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "correct")
        self.assertNotIn(
            "explicit_confusion_overrode_model_label",
            assessment["normalization_reasons"],
        )

    def test_student_questions_remain_confusion_and_never_create_misconceptions(
        self,
    ) -> None:
        cases = (
            ("为什么需要保存结果？", "confused"),
            ("要用什么方法？", "confused"),
            ("状态转移方程是什么？", "misconception"),
        )
        for index, (response, raw_signal) in enumerate(cases):
            with self.subTest(response=response, raw_signal=raw_signal):
                initial = _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                )
                raw = _plan(
                    signal=raw_signal,
                    confidence=0.9,
                    skill_id=(
                        "skill_misconception_contrast"
                        if raw_signal == "misconception"
                        else "skill_stepwise_scaffolding"
                    ),
                    misconception_tag=(
                        f"question_{index}" if raw_signal == "misconception" else None
                    ),
                    answer_alignment=(
                        "contradicted" if raw_signal == "misconception" else "ambiguous"
                    ),
                )
                raw["diagnosis"]["evidence_excerpt"] = response
                client = _client([initial, raw])
                session = start_live_teacher_agent_session(
                    self.demo["goal"],
                    self.demo["student_profile"],
                    self.library,
                    client,
                )
                updated = advance_live_teacher_agent_session(
                    session, learner_response=response, client=client
                )
                assessment = updated["history"][-1]["deepseek_assessment"]
                self.assertEqual(assessment["signal"], "confused")
                self.assertIsNone(assessment["misconception_tag"])
                self.assertFalse(updated["student_state"]["misconceptions"])

    def test_unpunctuated_chinese_questions_are_not_treated_as_claims(self) -> None:
        responses = (
            "这个变量表示什么",
            "状态转移方程指的是什么",
            "这一步为什么需要比较所有前驱",
            "什么是动态规划",
        )
        for index, response in enumerate(responses):
            with self.subTest(response=response):
                initial = _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                )
                false_misconception = _plan(
                    signal="misconception",
                    confidence=0.95,
                    skill_id="skill_misconception_contrast",
                    misconception_tag=f"unpunctuated_question_{index}",
                    answer_alignment="contradicted",
                )
                false_misconception["diagnosis"]["evidence_excerpt"] = response
                client = _client([initial, false_misconception])
                session = start_live_teacher_agent_session(
                    self.demo["goal"],
                    self.demo["student_profile"],
                    self.library,
                    client,
                )

                updated = advance_live_teacher_agent_session(
                    session, learner_response=response, client=client
                )

                assessment = updated["history"][-1]["deepseek_assessment"]
                self.assertEqual(assessment["signal"], "confused")
                self.assertIsNone(assessment["misconception_tag"])
                self.assertIn(
                    "learner_question_cannot_establish_misconception",
                    assessment["normalization_reasons"],
                )
                self.assertFalse(updated["student_state"]["misconceptions"])

    def test_declarative_claim_containing_whether_remains_a_misconception(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        misconception = _plan(
            signal="misconception",
            confidence=0.95,
            skill_id="skill_misconception_contrast",
            misconception_tag="state_omits_cost",
            answer_alignment="contradicted",
        )
        response = "状态只表示当前节点是否可达，不需要保存代价。"
        misconception["diagnosis"]["evidence_excerpt"] = response
        client = _client([initial, misconception])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response=response, client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "misconception")
        self.assertEqual(assessment["misconception_tag"], "state_omits_cost")
        self.assertNotIn(
            "learner_question_cannot_establish_misconception",
            assessment["normalization_reasons"],
        )
        self.assertEqual(
            updated["current_action"]["primary_skill"]["role"], "correction"
        )

    def test_chinese_copula_statements_are_reviewable_claims(self) -> None:
        responses = (
            "动态规划是暴力递归。",
            "状态是当前步骤编号。",
            "记忆化是状态转移方程。",
        )
        for index, response in enumerate(responses):
            with self.subTest(response=response):
                initial = _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                )
                misconception = _plan(
                    signal="misconception",
                    confidence=0.95,
                    skill_id="skill_misconception_contrast",
                    misconception_tag=f"copula_claim_{index}",
                    answer_alignment="contradicted",
                )
                misconception["diagnosis"]["evidence_excerpt"] = response
                client = _client([initial, misconception])
                session = start_live_teacher_agent_session(
                    self.demo["goal"],
                    self.demo["student_profile"],
                    self.library,
                    client,
                )

                updated = advance_live_teacher_agent_session(
                    session, learner_response=response, client=client
                )

                assessment = updated["history"][-1]["deepseek_assessment"]
                self.assertEqual(assessment["signal"], "misconception")
                self.assertEqual(
                    assessment["misconception_tag"], f"copula_claim_{index}"
                )
                self.assertEqual(
                    updated["current_action"]["primary_skill"]["role"],
                    "correction",
                )

    def test_chinese_ba_construction_is_a_reviewable_misconception_claim(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="记忆化会在什么时候计算子问题？",
        )
        misconception = _plan(
            signal="misconception",
            confidence=0.95,
            skill_id="skill_misconception_contrast",
            misconception_tag="memoization_precomputes_everything",
            answer_alignment="contradicted",
        )
        response = "记忆化把所有子问题提前算一遍，再查询结果。"
        misconception["diagnosis"]["evidence_excerpt"] = response
        client = _client([initial, misconception])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response=response, client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "misconception")
        self.assertEqual(
            assessment["misconception_tag"],
            "memoization_precomputes_everything",
        )
        self.assertEqual(
            updated["current_action"]["primary_skill"]["role"], "correction"
        )

    def test_direction_and_action_sentences_are_not_reduced_to_bare_terms(self) -> None:
        responses = (
            "记忆化从终点往起点。",
            "状态描述下一步答案。",
            "每步选当前最大的值。",
        )
        for index, response in enumerate(responses):
            with self.subTest(response=response):
                initial = _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                )
                misconception = _plan(
                    signal="misconception",
                    confidence=0.9,
                    skill_id="skill_misconception_contrast",
                    misconception_tag=f"claim_{index}",
                    answer_alignment="contradicted",
                )
                misconception["diagnosis"]["evidence_excerpt"] = response.rstrip("。")
                client = _client([initial, misconception])
                session = start_live_teacher_agent_session(
                    self.demo["goal"],
                    self.demo["student_profile"],
                    self.library,
                    client,
                )
                updated = advance_live_teacher_agent_session(
                    session, learner_response=response, client=client
                )
                assessment = updated["history"][-1]["deepseek_assessment"]
                self.assertEqual(assessment["signal"], "misconception")
                self.assertEqual(assessment["misconception_tag"], f"claim_{index}")

    def test_selected_skill_focus_is_authoritative_over_model_focus(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        initial["decision"]["next_focus"] = "transfer"
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            _client([initial]),
        )

        action = session["current_action"]
        self.assertEqual(action["primary_skill"]["focus_dimension"], "prerequisite")
        self.assertEqual(
            session["student_state"]["next_focus"]["dimension"], "prerequisite"
        )
        self.assertEqual(action["model_requested_next_focus"], "transfer")
        self.assertTrue(action["focus_was_constrained"])

    def test_exact_correction_contract_resolves_before_last_round_termination(
        self,
    ) -> None:
        profile = deepcopy(self.demo["student_profile"])
        profile["initial_mastery"] = {
            dimension: 0.9
            for dimension in ("prerequisite", "conceptual", "procedural", "transfer")
        }
        profile["known_misconceptions"] = [
            {"tag": "m1", "description": "只看一个前驱", "confidence": 0.8}
        ]
        goal = deepcopy(self.demo["goal"])
        goal["max_rounds"] = 3
        goal["success_thresholds"] = {
            dimension: 0.5
            for dimension in ("prerequisite", "conceptual", "procedural", "transfer")
        }
        contract = {
            "answer_type": "short_concept",
            "target_concepts": ["全部合法前驱"],
            "accepted_aliases": ["所有合法前驱"],
            "success_criteria": ["指出转移应检查全部合法前驱"],
        }
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        correction_one = _plan(
            signal="misconception",
            confidence=0.95,
            skill_id="skill_misconception_contrast",
            misconception_tag="m1",
            question_contract=contract,
        )
        correction_one["diagnosis"]["evidence_excerpt"] = "状态一定只看前一步"
        correction_two = deepcopy(correction_one)
        correction_two["diagnosis"]["evidence_excerpt"] = "仍然只看前一步"
        exact = _plan(
            signal="confused",
            confidence=0.2,
            skill_id="skill_concept_mapping",
            answer_alignment="ambiguous",
        )
        exact["diagnosis"]["evidence_excerpt"] = "全部合法前驱"
        exact["diagnosis"]["resolved_misconception_tags"] = ["m1"]
        client = _client([initial, correction_one, correction_two, exact])
        session = start_live_teacher_agent_session(goal, profile, self.library, client)
        session = advance_live_teacher_agent_session(
            session, learner_response="状态一定只看前一步。", client=client
        )
        session = advance_live_teacher_agent_session(
            session, learner_response="仍然只看前一步。", client=client
        )
        self.assertEqual(session["round"], 2)
        self.assertEqual(session["current_action"]["target_misconception_tags"], ["m1"])
        _install_server_question_contract(
            session,
            {
                **contract,
                "grading_scope": "current_question_only",
            },
            message="转移时应检查哪一类前驱？",
            expected_signal="学生指出应检查全部合法前驱。",
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response="全部合法前驱", client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        observation = updated["student_profile"]["adaptive_observations"][-1]
        self.assertEqual(
            assessment["assessment_source"], "active_question_contract_exact_match"
        )
        self.assertEqual(assessment["resolved_misconception_tags"], ["m1"])
        self.assertEqual(updated["status"], "succeeded")
        self.assertEqual(
            updated["control"]["termination_reason"].split(",", 1)[0],
            "all mastery thresholds reached",
        )
        self.assertEqual(
            next(
                item
                for item in updated["student_state"]["misconceptions"]
                if item["tag"] == "m1"
            )["status"],
            "resolved",
        )
        self.assertEqual(
            observation["evidence"]["assessment_source"],
            "active_question_contract_exact_match",
        )
        self.assertEqual(observation["evidence"]["model_raw_signal"], "confused")
        self.assertEqual(observation["evidence"]["final_signal"], "correct")
        self.assertEqual(observation["candidate"]["response_quality"], "complete")
        self.assertEqual(observation["evidence"]["review_reasons"], [])

    def test_bare_related_term_cannot_create_a_durable_misconception(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="保存并复用已经计算结果的机制叫什么？",
            question_contract={
                "answer_type": "short_concept",
                "target_concepts": ["记忆化"],
                "accepted_aliases": ["缓存", "memoization"],
                "success_criteria": ["给出保存并复用计算结果的机制名称"],
            },
        )
        false_correction = _plan(
            signal="misconception",
            confidence=0.9,
            skill_id="skill_misconception_contrast",
            misconception_tag="confused_transition_with_memoization",
            answer_alignment="contradicted",
            message="这个说法不对，我来纠正你。",
        )
        false_correction["diagnosis"]["evidence_excerpt"] = "状态转移方程"
        client = _client([initial, false_correction])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        _install_server_question_contract(
            session,
            {
                "answer_type": "short_concept",
                "target_concepts": ["记忆化"],
                "accepted_aliases": ["缓存", "memoization"],
                "success_criteria": ["给出保存并复用计算结果的机制名称"],
                "grading_scope": "current_question_only",
            },
            message="保存并复用已经计算结果的机制叫什么？",
            expected_signal="学生给出保存并复用计算结果的机制名称。",
        )
        updated = advance_live_teacher_agent_session(
            session, learner_response="状态转移方程", client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "partial")
        self.assertEqual(assessment["answer_alignment"], "related_but_not_answer")
        self.assertIsNone(assessment["misconception_tag"])
        self.assertFalse(updated["student_state"]["misconceptions"])
        self.assertNotEqual(
            updated["current_action"]["primary_skill"]["role"], "correction"
        )
        self.assertEqual(
            updated["current_action"]["teacher_action"]["type"],
            "socratic_comprehension_probe",
        )
        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_socratic_understanding_check",
        )
        retry_contract = updated["current_action"]["teacher_action"][
            "question_contract"
        ]
        self.assertEqual(retry_contract["answer_type"], "comparison")
        self.assertEqual(
            retry_contract["target_concepts"], ["判断依据", "反例或适用边界"]
        )
        self.assertNotIn("记忆化", retry_contract["target_concepts"])
        self.assertNotIn("缓存", retry_contract["accepted_aliases"])
        self.assertEqual(
            retry_contract["success_criteria"],
            ["学生说明依据，并识别一个反例、边界或条件变化。"],
        )
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 0)

    def test_explicit_confusion_overrides_false_misconception_label(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        false_misconception = _plan(
            signal="misconception",
            confidence=0.86,
            skill_id="skill_misconception_contrast",
            misconception_tag="cannot_do_is_not_a_knowledge_claim",
            answer_alignment="contradicted",
        )
        client = _client([initial, false_misconception])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response="我不会，我真的不理解。", client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["model_raw_signal"], "misconception")
        self.assertEqual(assessment["signal"], "confused")
        self.assertEqual(assessment["answer_alignment"], "ambiguous")
        self.assertIsNone(assessment["misconception_tag"])
        self.assertEqual(
            updated["student_state"]["understanding_signal"]["label"],
            "confused",
        )
        self.assertEqual(
            updated["student_state"]["interaction_statistics"]["confused_count"],
            1,
        )
        self.assertFalse(updated["student_state"]["misconceptions"])
        self.assertNotEqual(
            updated["current_action"]["primary_skill"]["role"], "correction"
        )
        self.assertEqual(
            updated["current_action"]["teacher_action"]["type"],
            "present_minimal_example",
        )
        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_concrete_example_bridge",
        )

    def test_already_correct_confusion_and_empty_labels_get_matching_actions(
        self,
    ) -> None:
        cases = (
            (
                "我还是不懂。",
                "confused",
                "ambiguous",
                "skill_stepwise_scaffolding",
                "skill_concrete_example_bridge",
                "present_minimal_example",
            ),
            (
                "",
                "no_response",
                "no_response",
                "skill_diagnostic_questioning",
                "skill_retrieval_review",
                "retrieval_practice",
            ),
        )
        for response, signal, alignment, skill_id, expected_skill, action_type in cases:
            with self.subTest(signal=signal):
                initial = _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                )
                assessed = _plan(
                    signal=signal,
                    confidence=0.9,
                    skill_id=skill_id,
                    answer_alignment=alignment,
                )
                assessed["diagnosis"]["evidence_excerpt"] = response
                client = _client([initial, assessed])
                session = start_live_teacher_agent_session(
                    self.demo["goal"],
                    self.demo["student_profile"],
                    self.library,
                    client,
                )
                updated = advance_live_teacher_agent_session(
                    session, learner_response=response, client=client
                )
                assessment = updated["history"][-1]["deepseek_assessment"]
                self.assertEqual(assessment["signal"], signal)
                self.assertEqual(
                    updated["current_action"]["teacher_action"]["type"],
                    action_type,
                )
                self.assertEqual(
                    updated["current_action"]["primary_skill"]["skill_id"],
                    expected_skill,
                )

    def test_propositional_buhui_is_not_a_confusion_self_report(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="记忆化有什么作用？",
        )
        correct = _plan(
            signal="correct",
            confidence=0.91,
            skill_id="skill_concept_mapping",
            answer_alignment="aligned",
        )
        correct["diagnosis"]["evidence_excerpt"] = "它不会重复计算相同状态"
        client = _client([initial, correct])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="它不会重复计算相同状态。",
            client=client,
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "correct")
        self.assertEqual(assessment["answer_alignment"], "aligned")
        self.assertNotIn(
            "explicit_confusion_overrode_model_label",
            assessment["normalization_reasons"],
        )

    def test_maximum_question_contract_advances_at_minimum_context_budget(self) -> None:
        contract = {
            "answer_type": "comparison",
            "target_concepts": [
                f"目标-{index}-" + chr(0x4E00 + index) * 110 for index in range(4)
            ],
            "accepted_aliases": [
                f"别名-{index}-" + chr(0x4E20 + index) * 110 for index in range(8)
            ],
            "success_criteria": [
                f"判据-{index}-" + chr(0x4E40 + index) * 148 for index in range(4)
            ],
        }
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            question_contract=contract,
        )
        turn = _plan(
            signal="partial",
            confidence=0.8,
            skill_id="skill_concept_mapping",
            answer_alignment="partially_aligned",
        )
        client = _client([initial, turn])
        options = LiveAgentOptions(maximum_context_chars=6_000)
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            client,
            options=options,
        )

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="我先比较两个状态的定义。",
            client=client,
            options=options,
        )

        self.assertEqual(updated["round"], 1)
        self.assertLessEqual(
            updated["context_memory"]["budget"]["serialized_chars"], 6_000
        )
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 0)

    def test_low_confidence_label_is_consistent_across_state_and_audit(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        uncertain_correct = _plan(
            signal="correct",
            confidence=0.2,
            skill_id="skill_transfer_check",
            answer_alignment="aligned",
        )
        uncertain_correct["diagnosis"]["needs_human_review"] = True
        client = _client([initial, uncertain_correct])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response="状态表示当前子问题。", client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        statistics = updated["student_state"]["interaction_statistics"]
        self.assertEqual(assessment["model_raw_signal"], "correct")
        self.assertEqual(assessment["signal"], "partial")
        self.assertEqual(
            updated["history"][-1]["structured_signal"]["label"], "partial"
        )
        self.assertEqual(
            updated["student_state"]["understanding_signal"]["label"],
            "partial",
        )
        self.assertEqual(statistics["partial_count"], 1)
        self.assertEqual(statistics["correct_count"], 0)
        self.assertNotEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_transfer_check",
        )

    def test_low_confidence_cannot_resolve_an_active_misconception(self) -> None:
        profile = deepcopy(self.demo["student_profile"])
        profile["known_misconceptions"] = [
            {
                "tag": "m1",
                "description": "状态只依赖前一步",
                "confidence": 0.8,
            }
        ]
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        correction = _plan(
            signal="misconception",
            confidence=0.9,
            skill_id="skill_misconception_contrast",
            misconception_tag="m1",
        )
        correction["diagnosis"]["evidence_excerpt"] = "状态一定只看前一步"
        uncertain = _plan(
            signal="correct",
            confidence=0.2,
            skill_id="skill_concept_mapping",
            answer_alignment="aligned",
        )
        uncertain["diagnosis"]["evidence_excerpt"] = "我现在会了"
        uncertain["diagnosis"]["resolved_misconception_tags"] = ["m1"]
        client = _client([initial, correction, uncertain])
        session = start_live_teacher_agent_session(
            self.demo["goal"], profile, self.library, client
        )
        session = advance_live_teacher_agent_session(
            session,
            learner_response="状态一定只看前一步。",
            client=client,
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response="我现在会了。", client=client
        )

        m1 = next(
            item
            for item in updated["student_state"]["misconceptions"]
            if item["tag"] == "m1"
        )
        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(m1["status"], "active")
        self.assertEqual(assessment["signal"], "partial")
        self.assertEqual(assessment["resolved_misconception_tags"], [])
        self.assertEqual(assessment["rejected_resolved_misconception_tags"], ["m1"])

    def test_resolution_is_limited_to_current_correction_target(self) -> None:
        profile = deepcopy(self.demo["student_profile"])
        profile["known_misconceptions"] = [
            {"tag": "m1", "description": "误解一", "confidence": 0.8},
            {"tag": "m2", "description": "误解二", "confidence": 0.8},
        ]
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        correction = _plan(
            signal="misconception",
            confidence=0.9,
            skill_id="skill_misconception_contrast",
            misconception_tag="m1",
        )
        correction["diagnosis"]["evidence_excerpt"] = "状态一定只看前一步"
        verified = _plan(
            signal="correct",
            confidence=0.9,
            skill_id="skill_concept_mapping",
            answer_alignment="aligned",
        )
        verified["diagnosis"]["evidence_excerpt"] = "需要比较全部合法前驱"
        verified["diagnosis"]["resolved_misconception_tags"] = ["m1", "m2"]
        client = _client([initial, correction, verified])
        session = start_live_teacher_agent_session(
            self.demo["goal"], profile, self.library, client
        )
        session = advance_live_teacher_agent_session(
            session,
            learner_response="状态一定只看前一步。",
            client=client,
        )
        self.assertEqual(session["current_action"]["target_misconception_tags"], ["m1"])

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="需要比较全部合法前驱。",
            client=client,
        )

        statuses = {
            item["tag"]: item["status"]
            for item in updated["student_state"]["misconceptions"]
        }
        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(statuses, {"m1": "resolved", "m2": "active"})
        self.assertEqual(assessment["resolved_misconception_tags"], ["m1"])
        self.assertEqual(assessment["rejected_resolved_misconception_tags"], ["m2"])

    def test_response_presence_and_alignment_are_normalized_consistently(self) -> None:
        cases = (
            {
                "response": "学生确实提交了一条回答。",
                "plan": _plan(
                    signal="no_response",
                    confidence=0.8,
                    skill_id="skill_diagnostic_questioning",
                    answer_alignment="no_response",
                ),
                "expected_signal": "partial",
                "expected_alignment": "ambiguous",
            },
            {
                "response": "",
                "plan": _plan(
                    signal="correct",
                    confidence=0.8,
                    skill_id="skill_transfer_check",
                    answer_alignment="aligned",
                ),
                "expected_signal": "no_response",
                "expected_alignment": "no_response",
            },
            {
                "response": "我提到的是相关内容。",
                "plan": _plan(
                    signal="correct",
                    confidence=0.8,
                    skill_id="skill_transfer_check",
                    answer_alignment="contradicted",
                ),
                "expected_signal": "partial",
                "expected_alignment": "ambiguous",
            },
        )
        for index, case in enumerate(cases):
            with self.subTest(index=index):
                case["plan"]["diagnosis"]["evidence_excerpt"] = case["response"]
                initial = _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                )
                client = _client([initial, case["plan"]])
                session = start_live_teacher_agent_session(
                    self.demo["goal"],
                    self.demo["student_profile"],
                    self.library,
                    client,
                )
                updated = advance_live_teacher_agent_session(
                    session,
                    learner_response=case["response"],
                    client=client,
                )
                assessment = updated["history"][-1]["deepseek_assessment"]
                self.assertEqual(assessment["signal"], case["expected_signal"])
                self.assertEqual(
                    assessment["answer_alignment"], case["expected_alignment"]
                )
                self.assertEqual(
                    updated["history"][-1]["structured_signal"]["label"],
                    case["expected_signal"],
                )

    def test_explicit_unknown_answer_remains_confused(self) -> None:
        client = _client(
            [
                _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                ),
                _plan(
                    signal="confused",
                    confidence=0.78,
                    skill_id="skill_concrete_example_bridge",
                    answer_alignment="ambiguous",
                ),
            ]
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        updated = advance_live_teacher_agent_session(
            session, learner_response="我不知道，也没理解这个词。", client=client
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "confused")
        self.assertEqual(assessment["answer_alignment"], "ambiguous")
        self.assertEqual(
            updated["history"][-1]["structured_signal"]["label"], "confused"
        )

    def test_mechanism_negation_is_not_misread_as_self_reported_confusion(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            message="记忆化有什么作用？",
            question_contract={
                "answer_type": "explanation",
                "target_concepts": ["避免重复计算", "复用已保存结果"],
                "accepted_aliases": ["缓存结果"],
                "success_criteria": ["说明不会重复计算并会复用已保存结果"],
            },
        )
        correct = _plan(
            signal="correct",
            confidence=0.95,
            skill_id="skill_self_explanation",
            answer_alignment="aligned",
            matched_concepts=["避免重复计算", "复用已保存结果"],
        )
        correct["diagnosis"]["evidence_excerpt"] = "不会算两遍，会复用已经保存的结果"
        client = _client([initial, correct])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        updated = advance_live_teacher_agent_session(
            session,
            learner_response="不会算两遍，会复用已经保存的结果。",
            client=client,
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "correct")
        self.assertEqual(assessment["answer_alignment"], "aligned")
        self.assertNotIn(
            "explicit_confusion_overrode_model_label",
            assessment["normalization_reasons"],
        )

    def test_first_person_mechanism_negation_is_not_confusion(self) -> None:
        response = "我不会算两遍，会复用已经保存的结果。"
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        correct = _plan(
            signal="correct",
            confidence=0.95,
            skill_id="skill_self_explanation",
            answer_alignment="aligned",
        )
        correct["diagnosis"]["evidence_excerpt"] = response.rstrip("。")
        client = _client([initial, correct])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response=response, client=client
        )
        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "correct")
        self.assertNotIn(
            "explicit_confusion_overrode_model_label",
            assessment["normalization_reasons"],
        )

    def test_task_inability_remains_an_explicit_confusion_signal(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        wrong = _plan(
            signal="correct",
            confidence=0.9,
            skill_id="skill_self_explanation",
            answer_alignment="aligned",
        )
        wrong["diagnosis"]["evidence_excerpt"] = "我不会算这道题"
        client = _client([initial, wrong])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session, learner_response="我不会算这道题。", client=client
        )
        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "confused")
        self.assertIn(
            "explicit_confusion_overrode_model_label",
            assessment["normalization_reasons"],
        )

    def test_fallback_history_is_labeled_and_never_promoted_to_long_term_memory(
        self,
    ) -> None:
        captured: list[dict] = []
        plans = deque(
            [
                _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                ),
                _plan(
                    signal="partial",
                    confidence=0.8,
                    skill_id="unknown_skill",
                ),
                _plan(
                    signal="partial",
                    confidence=0.8,
                    skill_id="skill_concrete_example_bridge",
                ),
            ]
        )

        def transport(_url: str, _headers: dict, payload: bytes, _timeout: float):
            body = json.loads(payload)
            captured.append(
                json.loads(body["messages"][1]["content"].split("\n", 1)[1])
            )
            envelope = {
                "id": "fallback_context_test",
                "choices": [{"message": {"content": json.dumps(plans.popleft())}}],
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
        session = advance_live_teacher_agent_session(
            session,
            learner_response="第一次请求会触发规则回退。",
            client=client,
        )
        self.assertEqual(session["student_profile"]["adaptive_observations"], [])
        session = advance_live_teacher_agent_session(
            session,
            learner_response="请继续，但不要把规则标签冒充模型判断。",
            client=client,
        )

        context = captured[-1]["teaching_context"]
        sources = {
            item["observation_source"]
            for item in context["knowledge_state"]["unresolved_issues"]
        }
        self.assertIn("deterministic_safety_fallback", sources)
        self.assertEqual(context["candidate_long_term_memory"]["observations"], [])
        self.assertTrue(
            context["claim_boundary"][
                "fallback_observations_are_labeled_by_actual_source"
            ]
        )

    def test_fabricated_evidence_excerpt_is_not_silently_replaced(self) -> None:
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
        self.assertEqual(evidence, "")
        self.assertTrue(assessment["model_evidence_excerpt_provided"])
        self.assertFalse(assessment["model_evidence_excerpt_grounded"])
        self.assertEqual(assessment["evidence_binding_source"], "none")
        self.assertNotIn(
            "learner@example.com", json.dumps(updated["history"][-1]["model_trace"])
        )
        self.assertNotIn(
            "learner@example.com", json.dumps(updated["history"][-1]["privacy_trace"])
        )
        self.assertNotIn("learner@example.com", json.dumps(calls[-1]))
        adaptive = updated["student_profile"]["adaptive_observations"][-1]
        self.assertNotIn("learner@example.com", json.dumps(adaptive))
        self.assertEqual(adaptive["evidence"]["excerpt"], "")
        self.assertEqual(adaptive["evidence"]["grounding"], "no_grounded_excerpt")
        self.assertTrue(adaptive["evidence"]["needs_human_review"])
        self.assertEqual(
            adaptive["evidence"]["review_reasons"],
            ["no_grounded_excerpt"],
        )
        validate_session(updated)
        malformed = deepcopy(updated)
        malformed_evidence = malformed["student_profile"]["adaptive_observations"][-1][
            "evidence"
        ]
        malformed_evidence["needs_human_review"] = False
        malformed_evidence["review_reasons"] = []
        malformed["student_profile"]["adaptive_summary"]["needs_human_review"] = False
        _refresh_integrity(malformed)
        with self.assertRaisesRegex(TeacherAgentError, "review state"):
            validate_session(malformed)

    def test_fabricated_high_impact_evidence_cannot_create_misconception(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        fabricated = _plan(
            signal="misconception",
            confidence=0.94,
            skill_id="skill_misconception_contrast",
            misconception_tag="fabricated_unrelated_error",
            answer_alignment="contradicted",
        )
        fabricated["diagnosis"]["evidence_excerpt"] = "学生没有说过的错误命题"
        fabricated["diagnosis"]["resolved_misconception_tags"] = ["legacy_tag"]
        client = _client([initial, fabricated])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session,
            learner_response=(
                "记忆化会复用已经保存的结果，因此不需要重复计算相同状态。"
            ),
            client=client,
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["model_raw_signal"], "misconception")
        self.assertEqual(assessment["signal"], "partial")
        self.assertIsNone(assessment["misconception_tag"])
        self.assertEqual(assessment["evidence_excerpt"], "")
        self.assertFalse(assessment["model_evidence_excerpt_grounded"])
        self.assertTrue(assessment["needs_human_review"])
        self.assertIn(
            "high_impact_diagnosis_without_bound_evidence_downgraded",
            assessment["normalization_reasons"],
        )
        self.assertEqual(assessment["resolved_misconception_tags"], [])
        self.assertFalse(updated["student_state"]["misconceptions"])
        self.assertNotEqual(
            updated["current_action"]["primary_skill"]["role"], "correction"
        )

    def test_low_confidence_candidate_requires_review_without_overwriting_profile(
        self,
    ) -> None:
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
        self.assertEqual(
            observation["evidence"]["review_reasons"],
            ["low_confidence", "no_grounded_excerpt"],
        )
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

    def test_live_context_budget_is_runtime_configurable_and_bounded(self) -> None:
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
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            client,
            options=LiveAgentOptions(
                maximum_context_chars=8_000,
                maximum_context_turns=2,
            ),
        )
        budget = session["context_memory"]["budget"]
        self.assertEqual(budget["max_chars"], 8_000)
        self.assertLessEqual(budget["serialized_chars"], 8_000)
        self.assertLessEqual(budget["max_recent_turns"], 2)
        with self.assertRaisesRegex(Exception, "maximum_context_chars"):
            LiveAgentOptions(maximum_context_chars=5_999).validated()
        with self.assertRaisesRegex(Exception, "maximum_context_turns"):
            LiveAgentOptions(maximum_context_turns=13).validated()

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

    def test_automatic_skill_must_match_applicable_signal_contract(self) -> None:
        invalid = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_transfer_check",
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            _client([invalid]),
            options=LiveAgentOptions(fallback_to_rules=True),
        )

        self.assertEqual(session["agent_runtime"]["fallback_count"], 0)
        self.assertIsNone(session["agent_runtime"]["last_error"])
        self.assertIn(
            "model_skill_not_applicable_to_signal",
            session["initial_model_plan"]["diagnosis"]["normalization_reasons"],
        )
        self.assertEqual(
            session["current_action"]["decision_origin"],
            "deepseek_v4_flash_constrained",
        )
        self.assertNotEqual(
            session["current_action"]["primary_skill"]["skill_id"],
            "skill_transfer_check",
        )
        self.assertTrue(session["current_action"]["primary_skill_was_retargeted"])

    def test_automatic_correction_requires_evidence_bound_misconception(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        invalid_correction = _plan(
            signal="misconception",
            confidence=0.9,
            skill_id="skill_misconception_contrast",
            misconception_tag=None,
        )
        client = _client([initial, invalid_correction])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        updated = advance_live_teacher_agent_session(
            session,
            learner_response="我觉得状态只需要看前一步。",
            client=client,
        )

        self.assertEqual(updated["agent_runtime"]["fallback_count"], 1)
        self.assertIn("misconception tag", updated["agent_runtime"]["last_error"])
        self.assertEqual(
            updated["history"][-1]["structured_signal"]["source"],
            "deterministic_safety_fallback",
        )

    def test_missing_misconception_tag_rewrites_noncorrection_action(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        inconsistent = _plan(
            signal="misconception",
            confidence=0.9,
            skill_id="skill_concept_mapping",
            misconception_tag=None,
            answer_alignment="contradicted",
            message="你这个结论错了。",
        )
        inconsistent["diagnosis"]["evidence_excerpt"] = "状态一定只看前一步"
        client = _client([initial, inconsistent])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="状态一定只看前一步。",
            client=client,
        )

        assessment = updated["history"][-1]["deepseek_assessment"]
        self.assertEqual(assessment["signal"], "partial")
        self.assertIsNone(assessment["misconception_tag"])
        self.assertEqual(
            updated["current_action"]["teacher_action"]["type"],
            "socratic_comprehension_probe",
        )
        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_socratic_understanding_check",
        )
        self.assertNotIn(
            "你这个结论错了", updated["current_action"]["teacher_action"]["message"]
        )
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 0)

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
            updated["history"][-1]["structured_signal"]["label"], "partial"
        )
        self.assertEqual(
            updated["history"][-1]["structured_signal"]["source"],
            "deterministic_safety_fallback",
        )
        self.assertEqual(updated["history"][-1]["structured_signal"]["confidence"], 0.0)
        self.assertEqual(
            updated["student_state"]["understanding_signal"]["source"],
            "deterministic_safety_fallback",
        )
        self.assertEqual(
            updated["student_state"]["understanding_signal"]["confidence"],
            0.0,
        )
        self.assertTrue(updated["student_state"]["understanding_signal"]["provisional"])
        self.assertEqual(
            updated["student_state"]["assessment_evidence"]["source"],
            "deterministic_safety_fallback",
        )
        self.assertTrue(
            updated["student_state"]["assessment_evidence"]["needs_human_review"]
        )
        event_state = updated["history"][-1]["student_state_after_observation"]
        self.assertEqual(
            event_state["understanding_signal"],
            updated["student_state"]["understanding_signal"],
        )
        self.assertEqual(event_state["assessment_confidence"], 0.0)
        self.assertEqual(
            event_state["assessment_evidence"],
            updated["student_state"]["assessment_evidence"],
        )
        validate_session(updated)

    def test_visual_confirmation_survives_invalid_plan_fallback(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        invalid = _plan(
            signal="correct",
            confidence=0.99,
            skill_id="unknown_skill",
            support=["skill_wait_and_elicit"],
            answer_alignment="aligned",
        )
        client = _client([initial, invalid])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        mastery_before = deepcopy(session["student_state"]["knowledge_mastery"])

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="答案见图",
            learner_evidence=[
                _visual_evidence(
                    "可能是缓存",
                    confidence=0.2,
                    needs_student_confirmation=True,
                )
            ],
            client=client,
        )

        self.assertEqual(updated["agent_runtime"]["fallback_count"], 1)
        self.assertEqual(
            updated["history"][-1]["structured_signal"],
            {
                "label": "partial",
                "confidence": 0.0,
                "source": "deterministic_safety_fallback",
                "provisional": True,
            },
        )
        self.assertEqual(
            updated["student_state"]["knowledge_mastery"], mastery_before
        )
        action = updated["current_action"]
        self.assertEqual(
            action["decision_origin"], "deterministic_safety_fallback"
        )
        self.assertEqual(
            action["primary_skill"]["skill_id"], "skill_self_explanation"
        )
        self.assertEqual(
            action["teacher_action"]["type"], "elicit_self_explanation"
        )
        self.assertEqual(action["supporting_skills"], [])
        self.assertEqual(action["composition_plan"]["support_execution"], {})
        self.assertIn("OCR", action["teacher_action"]["message"])
        validate_session(updated)

    def test_visual_confirmation_rotates_to_socratic_after_self_explanation_limit(
        self,
    ) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        first_visual_turn = _plan(
            signal="correct",
            confidence=0.99,
            skill_id="skill_self_explanation",
            answer_alignment="aligned",
        )
        second_visual_turn = deepcopy(first_visual_turn)
        client = _client([initial, first_visual_turn, second_visual_turn])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        mastery_before = deepcopy(session["student_state"]["knowledge_mastery"])
        evidence = [
            _visual_evidence(
                "x+1",
                confidence=0.2,
                needs_student_confirmation=True,
                formula_like_text_detected=True,
            )
        ]

        first = advance_live_teacher_agent_session(
            session,
            learner_response="答案见图",
            learner_evidence=evidence,
            client=client,
        )
        self.assertEqual(
            first["current_action"]["primary_skill"]["skill_id"],
            "skill_self_explanation",
        )
        self.assertEqual(first["student_state"]["knowledge_mastery"], mastery_before)

        second = advance_live_teacher_agent_session(
            first,
            learner_response="我补拍了同一公式，仍请看图",
            learner_evidence=evidence,
            client=client,
        )

        self.assertEqual(second["status"], "active")
        self.assertEqual(
            second["current_action"]["primary_skill"]["skill_id"],
            "skill_socratic_understanding_check",
        )
        self.assertEqual(
            second["current_action"]["teacher_action"]["type"],
            "socratic_comprehension_probe",
        )
        self.assertEqual(second["current_action"]["supporting_skills"], [])
        self.assertEqual(
            second["history"][-1]["structured_signal"]["confidence"], 0.0
        )
        self.assertEqual(second["student_state"]["knowledge_mastery"], mastery_before)
        self.assertIn("OCR", second["current_action"]["teacher_action"]["message"])
        validate_session(second)

    def test_fallback_visual_confirmation_rotates_without_terminating(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        invalid_first = _plan(
            signal="correct",
            confidence=0.99,
            skill_id="unknown_skill",
            answer_alignment="aligned",
        )
        invalid_second = deepcopy(invalid_first)
        client = _client([initial, invalid_first, invalid_second])
        session = start_live_teacher_agent_session(
            self.demo["goal"], self.demo["student_profile"], self.library, client
        )
        evidence = [
            _visual_evidence(
                "f(n)=f(n-1)+1",
                confidence=0.2,
                needs_student_confirmation=True,
                formula_like_text_detected=True,
            )
        ]
        second_evidence = deepcopy(evidence)
        second_evidence[0]["content_sha256"] = "b" * 64

        first = advance_live_teacher_agent_session(
            session,
            learner_response="公式写在图片里",
            learner_evidence=evidence,
            client=client,
        )
        second = advance_live_teacher_agent_session(
            first,
            learner_response="答案见图",
            learner_evidence=second_evidence,
            client=client,
        )

        self.assertEqual(second["status"], "active")
        self.assertEqual(second["agent_runtime"]["fallback_count"], 2)
        self.assertEqual(
            second["current_action"]["decision_origin"],
            "deterministic_safety_fallback",
        )
        self.assertEqual(
            second["current_action"]["primary_skill"]["skill_id"],
            "skill_socratic_understanding_check",
        )
        self.assertEqual(
            second["history"][-1]["structured_signal"],
            {
                "label": "partial",
                "confidence": 0.0,
                "source": "deterministic_safety_fallback",
                "provisional": True,
            },
        )
        validate_session(second)

    def test_rule_fallback_skips_primary_skill_contract_violations(self) -> None:
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
        ranked = [
            {
                "skill_id": "skill_transfer_check",
                "score": 1_000.0,
                "reasons": ["synthetic signal-mismatched first choice"],
            },
            {
                "skill_id": "skill_concept_mapping",
                "score": 999.0,
                "reasons": ["synthetic unsafe first choice"],
            },
            {
                "skill_id": "skill_retrieval_review",
                "score": 10.0,
                "reasons": ["synthetic safe second choice"],
            },
        ]

        with patch(
            "teaching_skill_miner.teacher_agent_live._selection_scores",
            return_value=ranked,
        ):
            updated = advance_live_teacher_agent_session(
                session,
                learner_response="我先回忆一个以前学过的相关概念。",
                client=client,
            )

        self.assertEqual(updated["agent_runtime"]["fallback_count"], 1)
        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_retrieval_review",
        )
        self.assertNotIn(
            "把刚才的例子映射",
            updated["current_action"]["teacher_action"]["message"],
        )
        self.assertEqual(
            updated["current_action"]["candidate_ranking"], [ranked[2]]
        )

    def test_rule_fallback_skips_example_skill_when_example_material_is_missing(
        self,
    ) -> None:
        goal = deepcopy(self.demo["goal"])
        goal["materials"].pop("example", None)
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
            goal, self.demo["student_profile"], self.library, client
        )
        ranked = [
            {
                "skill_id": "skill_concrete_example_bridge",
                "score": 999.0,
                "reasons": ["synthetic missing-material first choice"],
            },
            {
                "skill_id": "skill_retrieval_review",
                "score": 10.0,
                "reasons": ["synthetic safe retrieval choice"],
            },
        ]

        with patch(
            "teaching_skill_miner.teacher_agent_live._selection_scores",
            return_value=ranked,
        ):
            updated = advance_live_teacher_agent_session(
                session,
                learner_response="我先回忆一个以前学过的相关概念。",
                client=client,
            )

        self.assertEqual(updated["status"], "active")
        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_retrieval_review",
        )
        self.assertNotIn(
            "先只看这个最小例子",
            updated["current_action"]["teacher_action"]["message"],
        )
        self.assertEqual(updated["current_action"]["candidate_ranking"], [ranked[1]])

    def test_rule_fallback_skips_practice_skills_when_practice_is_missing(
        self,
    ) -> None:
        goal = deepcopy(self.demo["goal"])
        goal["materials"].pop("practice", None)
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
            goal, self.demo["student_profile"], self.library, client
        )
        ranked = [
            {
                "skill_id": "skill_stepwise_scaffolding",
                "score": 999.0,
                "reasons": ["synthetic missing-practice scaffolding"],
            },
            {
                "skill_id": "skill_practice_feedback",
                "score": 998.0,
                "reasons": ["synthetic missing-practice feedback"],
            },
            {
                "skill_id": "skill_self_explanation",
                "score": 10.0,
                "reasons": ["synthetic safe explanation choice"],
            },
        ]

        with patch(
            "teaching_skill_miner.teacher_agent_live._selection_scores",
            return_value=ranked,
        ):
            updated = advance_live_teacher_agent_session(
                session,
                learner_response="我先解释自己刚才为什么这样判断。",
                client=client,
            )

        self.assertEqual(updated["status"], "active")
        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_self_explanation",
        )
        self.assertNotIn(
            "我们只做练习的第一步",
            updated["current_action"]["teacher_action"]["message"],
        )
        self.assertEqual(updated["current_action"]["candidate_ranking"], [ranked[2]])

    def test_fallback_stops_when_no_primary_skill_is_contract_executable(self) -> None:
        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            _client([initial]),
        )
        for skill in session["skill_library"]["skills"]:
            if skill["role"] != "support":
                skill["applicable_signals"] = ["correct"]
        _refresh_integrity(session)

        materialized = _materialize_contract_safe_fallback_action(
            session,
            response="我只理解了一部分。",
            signal="partial",
            initial=False,
            previous_primary_skill_id="skill_diagnostic_questioning",
            options=LiveAgentOptions(),
        )

        self.assertFalse(materialized)
        self.assertEqual(session["status"], "terminated_unable")
        self.assertEqual(
            session["current_action"]["teacher_action"]["type"],
            "stop_and_escalate",
        )
        self.assertIn(
            "no executable primary Skill",
            session["control"]["termination_reason"],
        )

    def test_rule_fallback_enforces_primary_max_repeat(self) -> None:
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
        self_explanation = next(
            skill
            for skill in self.library["skills"]
            if skill["skill_id"] == "skill_self_explanation"
        )
        session["current_action"]["primary_skill"].update(
            {
                "skill_id": self_explanation["skill_id"],
                "name": self_explanation["name"],
                "role": self_explanation["role"],
                "focus_dimension": self_explanation["focus_dimension"],
                "source": deepcopy(self_explanation["source"]),
            }
        )
        session["current_action"]["teacher_action"]["type"] = self_explanation[
            "action_type"
        ]
        _refresh_integrity(session)
        ranked = [
            {
                "skill_id": "skill_self_explanation",
                "score": 999.0,
                "reasons": ["synthetic repeated first choice"],
            },
            {
                "skill_id": "skill_retrieval_review",
                "score": 10.0,
                "reasons": ["synthetic safe second choice"],
            },
        ]

        with patch(
            "teaching_skill_miner.teacher_agent_live._selection_scores",
            return_value=ranked,
        ):
            updated = advance_live_teacher_agent_session(
                session,
                learner_response="我继续说明刚才的依据。",
                client=client,
            )

        self.assertEqual(
            updated["current_action"]["primary_skill"]["skill_id"],
            "skill_retrieval_review",
        )
        self.assertEqual(
            updated["current_action"]["candidate_ranking"], [ranked[1]]
        )

    def test_rule_fallback_preserves_explicit_self_reported_confusion(self) -> None:
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
            session, learner_response="我不会，我还是没理解。", client=client
        )

        self.assertEqual(
            updated["history"][-1]["structured_signal"]["label"], "confused"
        )
        self.assertEqual(
            updated["history"][-1]["structured_signal"]["source"],
            "deterministic_safety_fallback",
        )

    def test_model_boolean_and_probability_fields_use_strict_json_types(self) -> None:
        cases = (
            (
                "confidence_string",
                lambda plan: plan["diagnosis"].__setitem__("confidence", "0.8"),
                "finite JSON number",
            ),
            (
                "confidence_boolean",
                lambda plan: plan["diagnosis"].__setitem__("confidence", True),
                "finite JSON number",
            ),
            (
                "confidence_nan",
                lambda plan: plan["diagnosis"].__setitem__("confidence", float("nan")),
                "must be in [0, 1]",
            ),
            (
                "confidence_infinity",
                lambda plan: plan["diagnosis"].__setitem__("confidence", float("inf")),
                "must be in [0, 1]",
            ),
            (
                "review_string_false",
                lambda plan: plan["diagnosis"].__setitem__(
                    "needs_human_review", "false"
                ),
                "JSON boolean",
            ),
            (
                "stop_string_false",
                lambda plan: plan["stop_recommendation"].__setitem__(
                    "should_stop", "false"
                ),
                "JSON boolean",
            ),
        )
        for label, mutate, expected_error in cases:
            with self.subTest(label=label):
                initial = _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                )
                invalid = _plan(
                    signal="partial",
                    confidence=0.8,
                    skill_id="skill_concrete_example_bridge",
                )
                mutate(invalid)
                client = _client([initial, invalid])
                session = start_live_teacher_agent_session(
                    self.demo["goal"],
                    self.demo["student_profile"],
                    self.library,
                    client,
                )

                updated = advance_live_teacher_agent_session(
                    session,
                    learner_response="我先说明一个部分理解。",
                    client=client,
                )

                self.assertEqual(updated["agent_runtime"]["fallback_count"], 1)
                self.assertIn(expected_error, updated["agent_runtime"]["last_error"])
                self.assertEqual(updated["status"], "active")
                self.assertEqual(
                    updated["history"][-1]["structured_signal"]["source"],
                    "deterministic_safety_fallback",
                )

    def test_long_fallback_question_contract_can_be_used_on_the_next_turn(self) -> None:
        goal = deepcopy(self.demo["goal"])
        goal["concept"] = "长" * 120
        invalid_initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="unknown_skill",
        )
        valid_turn = _plan(
            signal="partial",
            confidence=0.8,
            skill_id="skill_concept_mapping",
            answer_alignment="partially_aligned",
        )
        client = _client([invalid_initial, valid_turn])
        session = start_live_teacher_agent_session(
            goal,
            self.demo["student_profile"],
            self.library,
            client,
        )
        self.assertEqual(session["agent_runtime"]["fallback_count"], 1)
        fallback_contract = session["current_action"]["teacher_action"][
            "question_contract"
        ]
        self.assertLessEqual(len(fallback_contract["target_concepts"][0]), 120)
        self.assertLessEqual(len(fallback_contract["success_criteria"][0]), 160)

        updated = advance_live_teacher_agent_session(
            session,
            learner_response="我能先说出部分定义。",
            client=client,
        )

        self.assertEqual(updated["round"], 1)
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 1)
        self.assertIn("deepseek_assessment", updated["history"][-1])

    def test_model_question_contract_is_ignored_and_server_contract_is_bounded(
        self,
    ) -> None:
        from teaching_skill_miner.teacher_agent_context import build_layered_context

        initial = _plan(
            signal="not_observed",
            confidence=0.0,
            skill_id="skill_diagnostic_questioning",
            question_contract={
                "answer_type": "explanation",
                "target_concepts": ["状态定义"],
                "accepted_aliases": [],
                "success_criteria": [],
            },
        )
        initial["teacher_action"]["expected_signal"] = "判" * 400
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            _client([initial]),
        )
        stored = session["current_action"]["teacher_action"]["question_contract"]
        self.assertTrue(stored["success_criteria"])
        self.assertTrue(all(len(item) <= 160 for item in stored["success_criteria"]))
        self.assertNotEqual(stored["target_concepts"], ["状态定义"])

        context = build_layered_context(session, "我先尝试解释状态定义。")
        outbound = context["current_plan"]["current_action"]["question_contract"]
        self.assertEqual(outbound, stored)

    def test_context_build_failure_uses_explicit_rule_fallback(self) -> None:
        initial_client = _client(
            [
                _plan(
                    signal="not_observed",
                    confidence=0.0,
                    skill_id="skill_diagnostic_questioning",
                )
            ]
        )
        session = start_live_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            initial_client,
        )
        unused_client = _client([])
        with patch(
            "teaching_skill_miner.teacher_agent_live.build_layered_context",
            side_effect=ValueError("injected context failure"),
        ):
            updated = advance_live_teacher_agent_session(
                session,
                learner_response="上下文失败时仍应保留本轮输入。",
                client=unused_client,
            )

        self.assertEqual(updated["round"], 1)
        self.assertEqual(updated["agent_runtime"]["fallback_count"], 1)
        self.assertEqual(
            updated["agent_runtime"]["last_context_trace"]["request_outcome"],
            "deterministic_safety_fallback",
        )
        self.assertEqual(
            updated["history"][-1]["structured_signal"]["source"],
            "deterministic_safety_fallback",
        )
        self.assertIn(
            "上下文失败时仍应保留本轮输入",
            updated["context_memory"]["working_memory"]["current_learner_response"],
        )
        self.assertFalse(
            updated["context_memory"]["claim_boundary"][
                "model_generated_history_summary"
            ]
        )
        validate_session(updated)

    def test_initial_context_build_failure_returns_auditable_fallback(self) -> None:
        unused_client = _client([])
        with patch(
            "teaching_skill_miner.teacher_agent_live.build_layered_context",
            side_effect=ValueError("injected initial context failure"),
        ):
            session = start_live_teacher_agent_session(
                self.demo["goal"],
                self.demo["student_profile"],
                self.library,
                unused_client,
            )

        self.assertEqual(session["round"], 0)
        self.assertEqual(session["agent_runtime"]["fallback_count"], 1)
        self.assertEqual(
            session["current_action"]["decision_origin"],
            "deterministic_safety_fallback",
        )
        self.assertEqual(
            session["agent_runtime"]["last_context_trace"]["request_outcome"],
            "deterministic_safety_fallback",
        )
        validate_session(session)

    def test_commands_and_manual_stop(self) -> None:
        command = parse_skill_command("/+skill 误解对比纠错", self.library)
        self.assertEqual(command["skill_id"], "skill_misconception_contrast")
        self.assertEqual(parse_skill_command("/auto", self.library)["command"], "auto")
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
        stopped = stop_live_teacher_agent_session(session, reason="teacher demo stop")
        self.assertEqual(stopped["status"], "terminated_unable")
        self.assertEqual(
            stopped["current_action"]["decision_origin"], "teacher_command"
        )

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
        self.assertFalse(session["history"][-1]["model_stop_recommendation"]["honored"])
        session = advance_live_teacher_agent_session(
            session, learner_response="还是完全不知道怎么开始。", client=client
        )
        self.assertEqual(session["status"], "terminated_unable")
        self.assertEqual(
            session["current_action"]["decision_origin"],
            "guarded_model_escalation",
        )
        self.assertTrue(session["history"][-1]["model_stop_recommendation"]["honored"])


if __name__ == "__main__":
    unittest.main()
