from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import jsonschema

from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig
from teaching_skill_miner.teacher_agent import (
    _refresh_integrity,
    advance_teacher_agent_session,
    classify_lesson_clarification_text,
    is_lesson_clarification_response,
    lesson_required_primary_roles,
    lesson_progress_view,
    projected_lesson_closure,
    start_teacher_agent_session,
    validate_session,
)
from teaching_skill_miner.teacher_agent_live import (
    advance_live_teacher_agent_session,
    start_live_teacher_agent_session,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def _start(intent: str | None = None) -> dict:
    demo = _read("teacher_agent_demo_input.json")
    goal = deepcopy(demo["goal"])
    if intent is not None:
        goal["learning_intent"] = intent
    return start_teacher_agent_session(
        goal,
        demo["student_profile"],
        _read("teacher_agent_skill_library_v2.json"),
    )


def _start_high_mastery_teach_first() -> dict:
    demo = _read("teacher_agent_demo_input.json")
    goal = deepcopy(demo["goal"])
    goal.update({"learning_intent": "teach_first", "max_rounds": 20})
    profile = deepcopy(demo["student_profile"])
    profile["initial_mastery"] = {
        dimension: 1.0 for dimension in profile["initial_mastery"]
    }
    return start_teacher_agent_session(
        goal,
        profile,
        _read("teacher_agent_skill_library_v2.json"),
    )


def _advance_to_transfer(session: dict) -> dict:
    for response in (
        "继续",
        "看示范",
        "开始带练",
        "我完成了第一步，并说明了输入、操作和输出。",
        "我能独立解释依据、条件和一个边界。",
    ):
        session = _advance(session, response)
    assert session["lesson_state"]["lesson_phase"] == "transfer"
    return session


def _advance(session: dict, response: str, *, signal: str = "correct") -> dict:
    return advance_teacher_agent_session(
        session,
        learner_response=response,
        signal=signal,
        signal_confidence=0.95,
        answer_alignment="aligned" if signal == "correct" else "no_response",
    )


def _model_plan(skill_id: str, *, signal: str, confidence: float) -> dict:
    library = _read("teacher_agent_skill_library_v2.json")
    action_types = {item["skill_id"]: item["action_type"] for item in library["skills"]}
    alignment = "not_applicable" if signal == "not_observed" else "aligned"
    return {
        "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
        "diagnosis": {
            "signal": signal,
            "confidence": confidence,
            "answer_alignment": alignment,
            "matched_concepts": [],
            "missing_concepts": [],
            "diagnosis_reason": "当前回合的保守模型判断。",
            "evidence_excerpt": "",
            "misconception_tag": None,
            "misconception_description": "",
            "resolved_misconception_tags": [],
            "response_quality": "empty" if signal == "not_observed" else "minimal",
            "engagement_level": "unknown" if signal == "not_observed" else "medium",
            "needs_human_review": False,
        },
        "decision": {
            "primary_skill_id": skill_id,
            "supporting_skill_ids": [],
            "selection_reason": "模型提出旧式诊断路由。",
            "next_focus": "conceptual",
        },
        "teacher_action": {
            "type": action_types[skill_id],
            "message": "请先说明你判断这一步的依据。",
            "expected_signal": "学生说明当前依据。",
            "question_contract": {
                "answer_type": "explanation",
                "target_concepts": ["当前依据"],
                "accepted_aliases": [],
                "success_criteria": ["说明当前依据"],
            },
        },
        "stop_recommendation": {"should_stop": False, "reason": ""},
    }


def _client(plans: list[dict]) -> DeepSeekClient:
    queue = list(plans)

    def transport(_url: str, _headers: dict, _payload: bytes, _timeout: float):
        plan = queue.pop(0)
        envelope = {"choices": [{"message": {"content": json.dumps(plan)}}]}
        return 200, json.dumps(envelope).encode()

    return DeepSeekClient(
        DeepSeekConfig(allow_remote_student_data=True),
        api_key="test-key",
        transport=transport,
    )


def test_teach_first_starts_with_direct_explanation_instead_of_a_prequiz() -> None:
    session = _start("teach_first")

    assert session["round"] == 0
    assert session["history"] == []
    assert session["lesson_state"]["lesson_phase"] == "explanation"
    assert session["current_action"]["primary_skill"]["role"] == "example"
    assert session["current_action"]["primary_skill"]["role"] != "diagnostic"
    assert session["current_action"]["learning_evidence_policy"] == {
        "scope": "formative_observation_no_mastery",
        "mastery_gain_allowed": False,
        "teacher_action_is_learner_evidence": False,
    }
    assert lesson_progress_view(session)["phase_label"] == "讲解"
    assert any(
        item["status"] == "explained"
        for item in session["lesson_state"]["knowledge_exposure"].values()
    )
    validate_session(session)


def test_legacy_and_explicit_diagnostic_first_keep_the_benchmark_contract() -> None:
    legacy = _start()
    explicit = _start("diagnostic_first")
    for session in (legacy, explicit):
        assert (
            session["current_action"]["primary_skill"]["skill_id"]
            == "skill_diagnostic_questioning"
        )
    assert "lesson_state" not in legacy
    assert explicit["lesson_state"]["lesson_phase"] == "verification"


def test_task_first_starts_with_one_guided_step() -> None:
    session = _start("task_first")

    assert session["lesson_state"]["lesson_phase"] == "guided_practice"
    assert session["current_action"]["primary_skill"]["role"] in {
        "scaffolding",
        "practice",
    }
    assert (
        session["current_action"]["teacher_action"]["direct_answer_prohibited"] is True
    )


def test_navigation_cannot_inflate_mastery_or_no_progress_budget() -> None:
    session = _start("teach_first")
    mastery_before = deepcopy(session["student_state"]["knowledge_mastery"])

    updated = _advance(session, "继续")

    assert updated["student_state"]["knowledge_mastery"] == mastery_before
    assert updated["student_state"]["understanding_signal"]["label"] == "not_observed"
    assert updated["control"]["consecutive_no_progress"] == 0
    assert updated["lesson_state"]["lesson_phase"] == "worked_example"
    assert updated["history"][-1]["structured_signal"]["assessment_eligible"] is False
    assert updated["history"][-1]["structured_signal"]["applied_to_mastery"] is False
    validate_session(updated)


def test_lesson_clarification_classifier_covers_unpunctuated_questions_only() -> None:
    session = _start("teach_first")

    cases = {
        "过拟合是什么意思": "definition",
        "动态规划的定义是什么": "definition",
        "递推关系分哪几部分": "composition",
        "递推关系分哪几部分？": "composition",
        "状态转移由什么组成": "composition",
        "为什么要标准化": "rationale",
        "我知道状态是什么，但为什么还需要边界条件？": "rationale",
        "动态规划有哪些步骤": "procedure",
        "这个算法怎么工作": "procedure",
        "递推式怎么推导": "procedure",
        "这个公式里的 λ 表示什么": "symbol_meaning",
        "这个符号怎么读": "symbol_meaning",
        "正则化和早停有什么区别": "comparison",
        "能举一个例子吗": "example_request",
    }
    for response, expected_kind in cases.items():
        assert classify_lesson_clarification_text(response) == expected_kind
        assert is_lesson_clarification_response(session, response)

    for response in (
        "我知道为什么要用递推",
        "我把这个问题分成几个部分了",
        "递推关系包括初始条件和递推规则",
        "直接告诉我这道题的完整解法",
        "告诉我答案",
        "给我解法",
        "这道题怎么做",
        "下一步怎么写",
        "把代码写出来",
        "我之前不是说不会了吗，再说一遍是什么意思",
    ):
        assert classify_lesson_clarification_text(response) is None
        assert not is_lesson_clarification_response(session, response)

    task_session = _start("task_first")
    assert task_session["lesson_state"]["lesson_phase"] == "guided_practice"
    assert is_lesson_clarification_response(task_session, "标准化是什么意思")
    for current_task_request in (
        "这道题如何计算",
        "为什么这里要这样转移",
        "下一步怎么写",
    ):
        assert not is_lesson_clarification_response(task_session, current_task_request)


def test_exposure_progresses_without_becoming_mastery() -> None:
    session = _start("teach_first")
    mastery_before = deepcopy(session["student_state"]["knowledge_mastery"])

    assert session["lesson_state"]["lesson_phase"] == "explanation"
    assert any(
        item["status"] == "explained"
        for item in session["lesson_state"]["knowledge_exposure"].values()
    )
    assert session["student_state"]["knowledge_mastery"] == mastery_before

    session = _advance(session, "继续")
    assert session["lesson_state"]["lesson_phase"] == "worked_example"
    assert any(
        item["status"] == "example_seen"
        for item in session["lesson_state"]["knowledge_exposure"].values()
    )
    assert session["student_state"]["knowledge_mastery"] == mastery_before

    session = _advance(session, "看示范")
    assert session["lesson_state"]["lesson_phase"] == "guided_practice"
    assert session["student_state"]["knowledge_mastery"] == mastery_before

    session = _advance(session, "我完成了第一步，并说明了输入、操作和输出。")
    assert session["lesson_state"]["lesson_phase"] == "verification"
    assert any(
        item["status"] == "guided_practice_completed"
        for item in session["lesson_state"]["knowledge_exposure"].values()
    )

    session = _advance(session, "我能独立解释依据、条件和一个边界。")
    assert session["lesson_state"]["lesson_phase"] == "transfer"
    assert any(
        item["status"] == "learner_verified"
        for item in session["lesson_state"]["knowledge_exposure"].values()
    )
    assert session["lesson_state"]["claim_boundary"]["exposure_is_mastery"] is False
    validate_session(session)


def test_teach_first_closes_only_after_real_transfer_and_learner_summary() -> None:
    session = _advance_to_transfer(_start_high_mastery_teach_first())

    # High prior mastery and a successful verification still cannot terminate
    # before the learner has attempted transfer and summarized the result.
    assert session["status"] == "active"
    assert session["lesson_state"]["summary_required"] is False
    assert session["lesson_state"]["summary_completed"] is False
    assert session["current_action"]["primary_skill"]["role"] == "transfer"

    projection = projected_lesson_closure(
        session,
        learner_response="新情境中仍满足相同条件，所以我按同一关系迁移。",
        signal="correct",
        confidence=0.95,
        answer_alignment="aligned",
        needs_human_review=False,
    )
    assert projection == {
        "summary_required": True,
        "summary_completed": False,
    }
    assert session["lesson_state"]["summary_required"] is False

    session = _advance(session, "新情境中仍满足相同条件，所以我按同一关系迁移。")
    assert session["status"] == "active"
    assert session["lesson_state"]["summary_required"] is True
    assert session["lesson_state"]["summary_completed"] is False
    assert lesson_required_primary_roles(session) == ("summary",)
    assert lesson_progress_view(session)["phase_label"] == "总结"
    assert lesson_progress_view(session)["phase_index"] == 6
    assert lesson_progress_view(session)["phase_count"] == 6
    assert session["current_action"]["primary_skill"]["role"] == "summary"
    assert (
        session["history"][-1]["lesson_transition"]["reason"]
        == "transfer_evidence_requires_learner_summary"
    )

    session = _advance(
        session,
        "总结：先确认适用条件，再写出关系并检查边界；条件不成立时不能套用。",
    )
    assert session["status"] == "succeeded"
    assert session["lesson_state"]["summary_required"] is True
    assert session["lesson_state"]["summary_completed"] is True
    assert (
        session["history"][-1]["lesson_transition"]["reason"]
        == "learner_summary_completed"
    )
    terminal = session["current_action"]["teacher_action"]
    assert terminal["wait_for_student_before_next_action"] is False
    assert "迁移与学习者总结均已完成" in terminal["message"]
    assert "请学习者" not in terminal["message"]
    validate_session(session)


def test_last_regular_round_reserves_a_real_summary_closure_round() -> None:
    session = _advance_to_transfer(_start_high_mastery_teach_first())
    session["goal"]["max_rounds"] = session["round"] + 1
    _refresh_integrity(session)

    session = _advance(session, "我把方法迁移到新的边界条件并验证了结果。")

    assert session["round"] == session["goal"]["max_rounds"]
    assert session["status"] == "active"
    assert session["lesson_state"]["summary_required"] is True
    assert session["lesson_state"]["summary_closure_round_limit"] == 3
    assert session["lesson_state"]["summary_closure_rounds_used"] == 0
    assert session["current_action"]["primary_skill"]["role"] == "summary"

    session = _advance(
        session,
        "总结：先确认适用条件，再写出关系并检查边界；条件失效时不能套用。",
    )

    assert session["round"] == session["goal"]["max_rounds"] + 1
    assert session["status"] == "succeeded"
    assert session["lesson_state"]["summary_completed"] is True
    assert session["lesson_state"]["summary_closure_rounds_used"] == 1
    validate_session(session)


def test_summary_closure_failures_have_a_bounded_retry_budget() -> None:
    session = _advance_to_transfer(_start_high_mastery_teach_first())
    session["goal"]["max_rounds"] = session["round"] + 1
    _refresh_integrity(session)
    session = _advance(session, "我把方法迁移到新的边界条件并验证了结果。")

    regular_round_limit = session["goal"]["max_rounds"]
    for used in (1, 2):
        session = _advance(session, "不会", signal="confused")
        assert session["status"] == "active"
        assert session["lesson_state"]["summary_closure_rounds_used"] == used

    session = _advance(session, "还是不会", signal="confused")

    assert session["round"] == regular_round_limit + 3
    assert session["status"] == "terminated_unable"
    assert session["lesson_state"]["summary_completed"] is False
    assert session["lesson_state"]["summary_closure_rounds_used"] == 3
    assert (
        session["control"]["termination_reason"]
        == "summary closure round limit reached"
    )
    validate_session(session)


def test_summary_completion_requires_executed_summary_with_strong_evidence() -> None:
    session = _advance_to_transfer(_start_high_mastery_teach_first())
    session = _advance(session, "我把方法迁移到新的边界条件并验证了结果。")
    assert session["current_action"]["primary_skill"]["role"] == "summary"

    deferred = deepcopy(session)
    deferred["current_action"]["action_provenance"] = {
        "primary_skill_execution_deferred": True
    }
    _refresh_integrity(deferred)
    deferred = _advance(deferred, "我完整总结了条件、方法和失效边界。")
    assert deferred["status"] == "active"
    assert deferred["lesson_state"]["summary_completed"] is False
    assert (
        deferred["history"][-1]["lesson_transition"]["reason"]
        == "learner_summary_still_required"
    )

    wrong_role = deepcopy(session)
    wrong_role["current_action"]["primary_skill"]["role"] = "transfer"
    _refresh_integrity(wrong_role)
    wrong_role = _advance(wrong_role, "我完整总结了条件、方法和失效边界。")
    assert wrong_role["status"] == "active"
    assert wrong_role["lesson_state"]["summary_completed"] is False


def test_old_lesson_state_without_summary_flags_is_accepted_and_backfilled() -> None:
    session = _start("teach_first")
    state_without_flags = deepcopy(session["lesson_state"])
    state_without_flags.pop("summary_required")
    state_without_flags.pop("summary_completed")
    state_without_flags.pop("summary_closure_round_limit")
    state_without_flags.pop("summary_closure_rounds_used")
    session["lesson_state"] = deepcopy(state_without_flags)
    _refresh_integrity(session)

    validate_session(session)
    for schema_name in (
        "teacher_agent_session.schema.json",
        "teacher_agent_live_session.schema.json",
    ):
        schema = json.loads((ROOT / "schema" / schema_name).read_text(encoding="utf-8"))
        lesson_schema = {
            "$ref": "#/$defs/lessonState",
            "$defs": schema["$defs"],
        }
        validator = jsonschema.Draft202012Validator(lesson_schema)
        validator.validate(state_without_flags)
        validator.validate(_start("teach_first")["lesson_state"])

    updated = _advance(session, "继续")
    assert updated["lesson_state"]["summary_required"] is False
    assert updated["lesson_state"]["summary_completed"] is False
    assert updated["lesson_state"]["summary_closure_round_limit"] == 3
    assert updated["lesson_state"]["summary_closure_rounds_used"] == 0
    validate_session(updated)


def test_live_server_overrides_old_quiz_route_and_gates_navigation_evidence() -> None:
    demo = _read("teacher_agent_demo_input.json")
    goal = deepcopy(demo["goal"])
    goal.update({"learning_intent": "teach_first", "max_rounds": 18})
    client = _client(
        [
            _model_plan(
                "skill_diagnostic_questioning",
                signal="not_observed",
                confidence=0.0,
            ),
            _model_plan(
                "skill_socratic_understanding_check",
                signal="correct",
                confidence=0.99,
            ),
            _model_plan(
                "skill_socratic_understanding_check",
                signal="correct",
                confidence=0.99,
            ),
            _model_plan(
                "skill_socratic_understanding_check",
                signal="correct",
                confidence=0.99,
            ),
        ]
    )
    session = start_live_teacher_agent_session(
        goal,
        demo["student_profile"],
        _read("teacher_agent_skill_library_v2.json"),
        client,
    )
    assert session["lesson_state"]["lesson_phase"] == "explanation"
    assert session["current_action"]["primary_skill"]["role"] == "example"
    first_message = session["current_action"]["teacher_action"]["message"]
    assert first_message.startswith("具体来说，")
    for internal_phrase in (
        "本阶段不考前置知识",
        "入口不做定义测验",
        "教学路径",
        "来源证据卡",
        "本轮核心抓手",
        "只需回复“继续”",
    ):
        assert internal_phrase not in first_message
    mastery_before = deepcopy(session["student_state"]["knowledge_mastery"])
    student_model_before = deepcopy(session["student_state"]["student_model"])

    session = advance_live_teacher_agent_session(
        session, learner_response="继续", client=client
    )

    assert session["lesson_state"]["lesson_phase"] == "worked_example"
    assert session["current_action"]["primary_skill"]["role"] in {
        "example",
        "concept_mapping",
    }
    worked_example_action = session["current_action"]["teacher_action"]
    assert worked_example_action["message"].startswith(
        ("来看一个完整例子。", "再用同一个例子换一种拆法。")
    )
    assert "关键关系" in worked_example_action["message"]
    assert "检查" not in worked_example_action["message"] or "最后" in worked_example_action["message"]
    assert session["student_state"]["knowledge_mastery"] == mastery_before
    assert session["student_state"]["student_model"] == student_model_before
    assert session["student_profile"]["adaptive_observations"] == []
    assert session["control"]["consecutive_no_progress"] == 0
    assert session["history"][-1]["structured_signal"]["assessment_eligible"] is False

    session = advance_live_teacher_agent_session(
        session, learner_response="看示范", client=client
    )
    assert session["lesson_state"]["lesson_phase"] == "guided_practice"
    assert session["current_action"]["primary_skill"]["role"] in {
        "scaffolding",
        "practice",
    }
    assert goal["materials"]["practice"] not in worked_example_action["message"]
    grounding = session["history"][-1]["action"]["action_provenance"][
        "source_grounded_fallback"
    ]
    assert grounding["status"] == "source_grounded"
    assert len(grounding["grounding_refs"]) == 2
    assert worked_example_action["question_contract"]["answer_type"] == "reflection"
    assert len(worked_example_action["question_contract"]["target_concepts"]) == 1

    session = advance_live_teacher_agent_session(
        session, learner_response="开始带练", client=client
    )
    assert session["lesson_state"]["lesson_phase"] == "guided_practice"
    assert session["current_action"]["primary_skill"]["role"] in {
        "scaffolding",
        "practice",
    }
    assert session["student_state"]["knowledge_mastery"] == mastery_before
    assert session["student_state"]["student_model"] == student_model_before
    assert session["student_profile"]["adaptive_observations"] == []
    assert session["control"]["consecutive_no_progress"] == 0
