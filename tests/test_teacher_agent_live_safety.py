from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path

from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig
from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.teacher_agent import (
    _refresh_integrity,
    canonical_sha256,
    validate_session,
)
from teaching_skill_miner.teacher_agent_live import (
    _attach_accessible_action_projection,
    advance_live_teacher_agent_session,
    start_live_teacher_agent_session,
    stop_live_teacher_agent_session,
)


ROOT = Path(__file__).resolve().parents[1]


def _client() -> tuple[DeepSeekClient, list[bytes]]:
    plans = deque(
        [
            {
                "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
                "diagnosis": {
                    "signal": "not_observed",
                    "confidence": 0.0,
                    "answer_alignment": "not_applicable",
                    "matched_concepts": [],
                    "missing_concepts": [],
                    "diagnosis_reason": "尚未观察学生回答。",
                    "evidence_excerpt": "",
                    "misconception_tag": None,
                    "misconception_description": "",
                    "resolved_misconception_tags": [],
                    "response_quality": "empty",
                    "engagement_level": "unknown",
                    "needs_human_review": False,
                },
                "decision": {
                    "primary_skill_id": "skill_contextual_problem_setup",
                    "supporting_skill_ids": [],
                    "selection_reason": "先说明课程目标。",
                    "next_focus": "conceptual",
                },
                "teacher_action": {
                    "type": "establish_problem_context",
                    "message": "先说明本节目标和路径。请回复继续。",
                    "expected_signal": "学生确认继续。",
                    "question_contract": {
                        "answer_type": "reflection",
                        "target_concepts": ["继续"],
                        "accepted_aliases": ["继续"],
                        "success_criteria": ["学生确认继续"],
                    },
                },
                "stop_recommendation": {"should_stop": False, "reason": ""},
            }
        ]
    )
    provider_payloads: list[bytes] = []

    def transport(_url, _headers, payload, _timeout):
        provider_payloads.append(payload)
        plan = plans.popleft()
        return 200, json.dumps(
            {
                "id": "safety-live-test",
                "choices": [{"message": {"content": json.dumps(plan)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            }
        ).encode()

    return (
        DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True),
            api_key="test-key",
            transport=transport,
        ),
        provider_payloads,
    )


def test_live_safety_preempts_provider_and_preserves_mastery_phase_and_history() -> (
    None
):
    demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
    library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
    goal = deepcopy(demo["goal"])
    goal["learning_intent"] = "teach_first"
    client, provider_payloads = _client()
    session = start_live_teacher_agent_session(
        goal, deepcopy(demo["student_profile"]), library, client
    )
    calls_before = len(provider_payloads)
    mastery_before = deepcopy(session["student_state"]["knowledge_mastery"])
    model_before = deepcopy(session["student_state"]["student_model"])
    phase_before = deepcopy(session["lesson_state"])
    history_before = deepcopy(session["history"])

    updated = advance_live_teacher_agent_session(
        session,
        learner_response="我今晚想用药结束生命",
        client=client,
    )

    validate_session(updated)
    assert len(provider_payloads) == calls_before
    assert updated["student_state"]["knowledge_mastery"] == mastery_before
    assert updated["student_state"]["student_model"] == model_before
    assert updated["lesson_state"] == phase_before
    assert updated["history"] == history_before
    action = updated["current_action"]
    assert action["decision_origin"] == "deterministic_learner_safety_boundary"
    assert action["learning_evidence_policy"]["mastery_gain_allowed"] is False
    assert action["action_obligations"][0]["category"] == "self_harm"
    assert "我今晚想用药结束生命" not in json.dumps(updated, ensure_ascii=False)
    assert "急救" in action["teacher_action"]["message"]


def test_live_academic_integrity_redirects_to_scaffold_without_solving() -> None:
    demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
    library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
    client, provider_payloads = _client()
    session = start_live_teacher_agent_session(
        demo["goal"], demo["student_profile"], library, client
    )
    calls_before = len(provider_payloads)

    updated = advance_live_teacher_agent_session(
        session,
        learner_response="帮我绕过监考并代考",
        client=client,
    )

    assert len(provider_payloads) == calls_before
    message = updated["current_action"]["teacher_action"]["message"]
    assert "不能代考" in message
    assert "方法" in message
    assert "相似例子" in message
    assert updated["current_action"]["action_obligations"][0]["category"] == (
        "academic_integrity"
    )


def test_live_actions_expose_lossless_accessibility_contract_without_lowering_mastery() -> (
    None
):
    demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
    library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
    profile = deepcopy(demo["student_profile"])
    profile["accessibility_needs"] = ["一次一问", "使用屏幕阅读器 VoiceOver"]
    client, _provider_payloads = _client()

    session = start_live_teacher_agent_session(demo["goal"], profile, library, client)

    action = session["current_action"]
    contract = action["accessibility_contract"]
    rendering = action["accessible_rendering"]
    assert contract["enabled"]["reduced_cognitive_load"] is True
    assert contract["enabled"]["screen_reader_linear"] is True
    assert contract["mastery_standard_lowered"] is False
    assert rendering["authoritative_message_preserved"] is True
    assert rendering["mastery_standard_lowered"] is False
    assert (
        action["action_provenance"]["accessibility_projection"]["message_sha256"]
        == rendering["message_sha256"]
    )
    validate_session(session)


def test_live_final_accessibility_gate_repairs_idempotently_without_changing_grounding_or_grading() -> (
    None
):
    demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
    library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
    profile = deepcopy(demo["student_profile"])
    profile["accessibility_needs"] = [
        "一次一问并使用短句",
        "使用屏幕阅读器 VoiceOver",
        "色弱：不依赖颜色",
        "听障：需要文字稿",
    ]
    client, _provider_payloads = _client()
    session = start_live_teacher_agent_session(demo["goal"], profile, library, client)
    action = deepcopy(session["current_action"])
    excerpt = "教师原句？递推只依赖已经定义的前项。"
    authoritative_message = (
        "看上图，红色区域表示递推结果。这个关系是什么？为什么成立？"
        "请听录音。这里保留完整的解释内容和适用边界，不允许无障碍修复删去。"
        f"依据：“{excerpt}”"
    )
    action["teacher_action"]["message"] = authoritative_message
    contract_before = deepcopy(action["teacher_action"]["question_contract"])
    expected_before = action["teacher_action"]["expected_signal"]
    action["action_provenance"]["message_preserved_verbatim"] = True
    action["action_provenance"].pop("accessibility_projection", None)
    action["action_provenance"]["source_grounded_fallback"] = {
        "source_bindings": [
            {
                "ref": "knowledge_spec.canonical_claims:claim_01",
                "excerpt_sha256": canonical_sha256(excerpt),
            }
        ]
    }
    source_receipt_before = deepcopy(
        action["action_provenance"]["source_grounded_fallback"]
    )

    repaired = _attach_accessible_action_projection(session, action)
    projection = repaired["action_provenance"]["accessibility_projection"]
    observable = repaired["teacher_action"]["message"]

    assert projection["status"] == "repaired"
    assert projection["authoritative_message"] == authoritative_message
    assert projection["authoritative_message_preserved"] is True
    assert projection["answer_content_deleted"] is False
    assert projection["scoring_contract_changed"] is False
    assert projection["question_contract_preserved"] is True
    assert projection["question_contract_sha256"] == canonical_sha256(contract_before)
    assert projection["source_grounding_receipt_sha256"] == canonical_sha256(
        source_receipt_before
    )
    assert projection["source_grounding_receipt_preserved"] is True
    assert (
        repaired["action_provenance"]["source_grounded_fallback"]
        == source_receipt_before
    )
    assert repaired["teacher_action"]["question_contract"] == contract_before
    assert repaired["teacher_action"]["expected_signal"] == expected_before
    assert repaired["accessible_rendering"]["question_count"] == 1
    assert repaired["accessible_rendering"]["violations"] == []
    assert excerpt in observable
    assert "完整的解释内容和适用边界" in observable
    assert projection["verbatim_excerpts"] == [
        {
            "ref": "knowledge_spec.canonical_claims:claim_01",
            "excerpt_sha256": canonical_sha256(excerpt),
            "occurrences_before": 1,
            "occurrences_after": 1,
            "preserved_verbatim": True,
        }
    ]

    repeated = _attach_accessible_action_projection(session, deepcopy(repaired))
    assert repeated["teacher_action"]["message"] == observable
    assert (
        repeated["action_provenance"]["accessibility_projection"][
            "authoritative_message"
        ]
        == authoritative_message
    )
    session["current_action"] = repeated
    _refresh_integrity(session)
    validate_session(session)

    terminal = stop_live_teacher_agent_session(session)
    assert terminal["current_action"]["accessible_rendering"]["compliant"] is True
    assert (
        terminal["current_action"]["action_provenance"]["accessibility_projection"][
            "scoring_contract_changed"
        ]
        is False
    )


def test_live_safety_pause_requires_confirmation_then_restores_suspended_lesson() -> (
    None
):
    demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
    library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
    goal = deepcopy(demo["goal"])
    goal["learning_intent"] = "teach_first"
    client, provider_payloads = _client()
    session = start_live_teacher_agent_session(
        goal, deepcopy(demo["student_profile"]), library, client
    )
    calls_before = len(provider_payloads)
    suspended = deepcopy(session["current_action"])
    mastery_before = deepcopy(session["student_state"]["knowledge_mastery"])
    model_before = deepcopy(session["student_state"]["student_model"])
    lesson_before = deepcopy(session["lesson_state"])
    history_before = deepcopy(session["history"])

    paused = advance_live_teacher_agent_session(
        session,
        learner_response="我今晚想结束生命",
        client=client,
    )
    premature_resume = advance_live_teacher_agent_session(
        paused,
        learner_response="恢复课程",
        client=client,
    )
    assert "先确认" in premature_resume["current_action"]["teacher_action"]["message"]
    assert (
        premature_resume["current_action"]["action_obligations"][0][
            "safe_status_confirmed"
        ]
        is False
    )

    confirmed = advance_live_teacher_agent_session(
        premature_resume,
        learner_response="我现在安全，有家长陪着",
        client=client,
    )
    assert (
        confirmed["current_action"]["action_obligations"][0]["safe_status_confirmed"]
        is True
    )
    assert "恢复课程" in confirmed["current_action"]["teacher_action"]["message"]

    resumed = advance_live_teacher_agent_session(
        confirmed,
        learner_response="恢复课程",
        client=client,
    )
    validate_session(resumed)
    assert len(provider_payloads) == calls_before
    assert resumed["current_action"]["teacher_action"] == suspended["teacher_action"]
    assert resumed["current_action"]["primary_skill"] == suspended["primary_skill"]
    assert (
        resumed["current_action"]["action_provenance"]["latest_safety_resume_receipt"][
            "mastery_evidence"
        ]
        is False
    )
    assert resumed["student_state"]["knowledge_mastery"] == mastery_before
    assert resumed["student_state"]["student_model"] == model_before
    assert resumed["lesson_state"] == lesson_before
    assert resumed["history"] == history_before
    serialized = json.dumps(resumed, ensure_ascii=False)
    assert "我今晚想结束生命" not in serialized
    assert "我现在安全，有家长陪着" not in serialized
