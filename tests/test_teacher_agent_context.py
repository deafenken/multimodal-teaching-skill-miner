from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from teaching_skill_miner.teacher_agent_context import (
    CONTEXT_SCHEMA,
    GOAL_PLAN_SCHEMA,
    LAYERED_CONTEXT_SCHEMA,
    _teaching_checkpoints,
    build_layered_context,
    build_goal_plan,
    build_minimal_layered_context,
    build_relevant_history,
    redact_remote_text,
    validate_layered_context,
)
from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.teacher_agent import (
    advance_teacher_agent_session,
    lesson_required_primary_roles,
    start_teacher_agent_session,
)
from teaching_skill_miner.teacher_agent_memory import (
    commit_teaching_memory_turn,
    initialize_teaching_memory,
)


ROOT = Path(__file__).resolve().parents[1]


def _event(
    round_number: int,
    *,
    focus: str,
    kc: str,
    signal: str,
    learner_response: str | None = None,
) -> dict:
    return {
        "round": round_number,
        "action": {
            "primary_skill": {
                "skill_id": f"skill_{focus}",
                "focus_dimension": focus,
                "knowledge_components": [kc],
            },
            "teacher_action": {
                "message": f"第 {round_number} 轮检查 {kc}。",
            },
        },
        "learner_response": learner_response or f"第 {round_number} 轮回答",
        "structured_signal": {"label": signal, "confidence": 0.8},
    }


def _session() -> dict:
    events = [
        _event(1, focus="prerequisite", kc="递归", signal="partial"),
        _event(2, focus="conceptual", kc="状态定义", signal="correct"),
        _event(3, focus="procedural", kc="边界条件", signal="confused"),
        _event(4, focus="conceptual", kc="状态转移", signal="partial"),
        _event(5, focus="procedural", kc="状态转移", signal="misconception"),
        _event(6, focus="transfer", kc="最优子结构", signal="correct"),
        _event(7, focus="conceptual", kc="状态转移", signal="correct"),
        _event(8, focus="transfer", kc="新情境", signal="partial"),
    ]
    return {
        "goal": {"concept": "动态规划", "knowledge_components": ["状态转移"]},
        "student_state": {
            "next_focus": {
                "dimension": "conceptual",
                "knowledge_component": "状态转移",
            }
        },
        "current_action": {
            "primary_skill": {
                "focus_dimension": "conceptual",
                "knowledge_components": ["状态转移"],
            }
        },
        "history": events,
    }


def _json_length(value: dict) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


@pytest.mark.parametrize(
    ("phase", "summary_required", "summary_completed", "expected_roles"),
    [
        ("orientation", False, False, ["context"]),
        ("explanation", False, False, ["example"]),
        ("worked_example", False, False, ["example", "concept_mapping"]),
        ("guided_practice", False, False, ["scaffolding", "practice"]),
        ("verification", False, False, ["assessment", "metacognition", "review"]),
        ("transfer", False, False, ["transfer"]),
        ("transfer", True, False, ["summary"]),
        ("transfer", True, True, ["transfer"]),
    ],
)
def test_rich_and_minimal_context_share_server_owned_lesson_role_policy(
    phase: str,
    summary_required: bool,
    summary_completed: bool,
    expected_roles: list[str],
) -> None:
    demo = read_json(ROOT / "data" / "teacher_agent_demo_input.json")
    goal = deepcopy(demo["goal"])
    goal["learning_intent"] = "teach_first"
    session = start_teacher_agent_session(
        goal,
        demo["student_profile"],
        read_json(ROOT / "data" / "teacher_agent_skill_library_v2.json"),
    )
    session["lesson_state"].update(
        {
            "lesson_phase": phase,
            "summary_required": summary_required,
            "summary_completed": summary_completed,
        }
    )

    assert list(lesson_required_primary_roles(session)) == expected_roles
    contexts = (
        build_layered_context(session, "继续"),
        build_minimal_layered_context(session, "继续"),
    )
    for context in contexts:
        contract = context["fixed_context"]["teaching_goal"]["lesson_contract"]
        assert contract["required_primary_roles"] == expected_roles
        assert contract["summary_required"] is summary_required
        assert contract["summary_completed"] is summary_completed
        assert contract["phase_sequence"] == [
            "orientation",
            "explanation",
            "worked_example",
            "guided_practice",
            "verification",
            "transfer",
        ]


def test_layered_context_reinjects_explicit_memory_and_teacher_knowledge_spec() -> None:
    session = _session()
    session["goal"].update(
        {
            "objective": "解释状态转移并完成迁移",
            "materials": {},
            "success_thresholds": {},
            "max_rounds": 12,
            "knowledge_spec": {
                "schema": "teaching_skill_miner.teacher_goal_knowledge_spec.v1",
                "status": "teacher_provided",
                "canonical_claims": [
                    {
                        "claim_id": "claim_state",
                        "statement": "状态转移必须说明当前状态如何依赖已解决子问题。",
                        "knowledge_components": ["状态转移"],
                        "required": True,
                        "source_ids": [],
                    }
                ],
                "rubric_criteria": [],
                "accepted_alternatives": [],
                "reference_steps": [],
                "misconception_catalog": [],
                "sources": [],
                "claim_boundary": {
                    "authoritative_for_runtime_grading": True,
                    "teacher_authored_or_imported": True,
                    "independently_verified_by_system": False,
                    "model_memory_is_authoritative_when_absent": False,
                },
            },
        }
    )
    session["student_profile"] = {
        "profile_ref": "synthetic",
        "learner_level": "beginner",
        "preferences": ["先看例子"],
        "accessibility_needs": [],
        "initial_mastery": {},
        "known_misconceptions": [],
        "conversation_history": [],
        "background_history": [],
    }
    memory = initialize_teaching_memory(session["goal"], session["student_profile"])
    session["teaching_memory"] = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text="请先举例，不要直接上公式。",
        teacher_action={"action_id": "turn_001", "message": "先看一个台阶例子。"},
    )
    context = build_layered_context(session, "第二种呢？", max_recent_turns=2)
    validate_layered_context(context)

    knowledge_spec = context["fixed_context"]["teaching_goal"]["knowledge_spec"]
    assert knowledge_spec["status"] == "teacher_provided"
    assert knowledge_spec["canonical_claims"][0]["claim_id"] == "claim_state"
    memory_view = context["semantic_summary"]["teaching_memory"]
    assert memory_view["history_version"] == 1
    assert any(
        "不要直接上公式" in item["statement"]
        for item in memory_view["active_preferences"]
    )
    ledger_ids = {item["evidence_id"] for item in context["evidence_ledger"]}
    assert {
        ref
        for item in memory_view["active_preferences"]
        for ref in item["evidence_refs"]
    } <= ledger_ids


def _continuity_session() -> dict:
    session = _session()
    session["goal"].update(
        {
            "objective": "比较两种条件概率求法并能解释选择依据",
            "materials": {},
            "success_thresholds": {},
            "max_rounds": 12,
        }
    )
    profile = {
        "profile_ref": "synthetic-continuity",
        "learner_level": "beginner",
        "preferences": [],
        "accessibility_needs": [],
        "initial_mastery": {},
        "known_misconceptions": [],
        "conversation_history": [],
        "background_history": [],
    }
    session["student_profile"] = profile
    turns = [
        (
            "我有两种方法：第一种画树状图，第二种直接用条件概率公式。",
            "请先比较两种方法各自需要的信息。",
        ),
        (
            "请先用骰子例子，之后再回到条件概率公式。",
            "我们先看一个只有两个结果的骰子情境。",
        ),
        (
            "为什么第二种可以直接除以条件事件的概率？",
            "先说说分母代表哪个样本空间。",
        ),
    ]
    memory = initialize_teaching_memory(session["goal"], profile)
    history = []
    for round_number, (learner_text, teacher_message) in enumerate(turns, 1):
        memory = commit_teaching_memory_turn(
            memory,
            round_number=round_number,
            learner_text=learner_text,
            teacher_action={
                "action_id": f"turn_{round_number:03d}",
                "message": teacher_message,
            },
        )
        history.append(
            {
                "round": round_number,
                "learner_text": learner_text,
                "learner_response": learner_text,
                "action": {
                    "action_id": f"turn_{round_number:03d}",
                    "primary_skill": {
                        "skill_id": "skill_socratic_understanding_check",
                        "focus_dimension": "conceptual",
                        "knowledge_components": ["条件概率"],
                    },
                    "teacher_action": {"message": teacher_message},
                },
                "structured_signal": {
                    "label": "partial",
                    "confidence": 0.8,
                    "source": "test_fixture",
                },
            }
        )
    session["history"] = history
    session["teaching_memory"] = memory
    return session


@pytest.mark.parametrize(
    ("cue", "expected_kind", "expected_text"),
    [
        ("第二种呢？", "learner_named_alternatives", "第二种直接用条件概率公式"),
        (
            "可以继续，但请按我最开始说的方式讲。",
            "learner_instruction",
            "先用骰子例子",
        ),
        (
            "回到一开始的问题。",
            "unresolved_learner_question",
            "为什么第二种可以直接除以条件事件的概率",
        ),
        (
            "按刚才约定继续。",
            "learner_future_agenda",
            "之后再回到条件概率公式",
        ),
    ],
)
def test_layered_context_resolves_explicit_continuity_cues_from_evidence(
    cue: str,
    expected_kind: str,
    expected_text: str,
) -> None:
    context = build_layered_context(
        _continuity_session(), cue, max_recent_turns=1, max_chars=14_000
    )
    validate_layered_context(context)

    recall = context["semantic_summary"]["continuity_recall"]
    assert recall["status"] == "resolved_evidence_linked"
    assert recall["target"]["kind"] == expected_kind
    assert expected_text in recall["target"]["excerpt"]
    ledger_ids = {item["evidence_id"] for item in context["evidence_ledger"]}
    assert set(recall["cue_evidence_refs"]) <= ledger_ids
    assert set(recall["target"]["evidence_refs"]) <= ledger_ids


def test_layered_context_fails_closed_when_recall_cue_has_no_matching_record() -> None:
    session = _session()
    session["student_profile"] = {
        "profile_ref": "synthetic-no-memory",
        "learner_level": "beginner",
        "preferences": [],
        "accessibility_needs": [],
        "initial_mastery": {},
        "known_misconceptions": [],
        "conversation_history": [],
        "background_history": [],
    }
    session["history"] = []
    session["teaching_memory"] = initialize_teaching_memory(
        session["goal"], session["student_profile"]
    )

    context = build_layered_context(session, "第二种呢？", max_chars=14_000)
    recall = context["semantic_summary"]["continuity_recall"]

    assert recall["status"] == "unresolved_no_matching_evidence"
    assert recall["target"] is None
    assert recall["must_not_invent"] is True


def test_prior_agreement_completion_cue_prioritizes_unresolved_work() -> None:
    context = build_layered_context(
        _continuity_session(),
        "按我们之前约定的方式继续，并提醒我哪里还没完成。",
        max_recent_turns=1,
        max_chars=14_000,
    )
    validate_layered_context(context)

    recall = context["semantic_summary"]["continuity_recall"]
    assert recall["cue_kind"] == "prior_agreement_or_agenda"
    assert recall["status"] == "resolved_evidence_linked"
    assert recall["target"]["kind"] == "unresolved_learner_question"
    assert recall["target"]["source_round"] == 3
    assert set(recall["target"]["evidence_refs"]) <= {
        item["evidence_id"] for item in context["evidence_ledger"]
    }


def test_continuity_recall_can_resolve_the_current_unanswered_teacher_action() -> None:
    session = _session()
    session["student_profile"] = {
        "profile_ref": "synthetic-current-action",
        "learner_level": "beginner",
        "preferences": [],
        "accessibility_needs": [],
        "initial_mastery": {},
        "known_misconceptions": [],
        "conversation_history": [],
        "background_history": [],
    }
    session["history"] = []
    session["round"] = 0
    session["current_action"] = {
        "round": 1,
        "primary_skill": {
            "skill_id": "skill_conceptual",
            "focus_dimension": "conceptual",
            "knowledge_components": ["状态转移"],
        },
        "teacher_action": {
            "question_id": "q_001",
            "message": (
                "第一种方法保留完整状态表，第二种方法只保留相邻状态。你想先比较哪一种？"
            ),
        },
    }
    session["teaching_memory"] = initialize_teaching_memory(
        session["goal"], session["student_profile"]
    )

    context = build_layered_context(session, "第二种呢？", max_chars=14_000)
    recall = context["semantic_summary"]["continuity_recall"]

    assert recall["status"] == "resolved_evidence_linked"
    assert recall["target"]["kind"] == "teacher_named_alternatives"
    assert recall["target"]["source_round"] == 1
    assert recall["target"]["evidence_refs"] == ["current_action:r1:teacher_action"]


@pytest.mark.parametrize(
    "cue",
    ("刚才那个我不会", "刚才那个是什么意思"),
)
def test_deictic_continuity_prefers_the_current_visible_action(cue: str) -> None:
    session = _continuity_session()
    session["round"] = 3
    session["current_action"] = {
        "round": 4,
        "primary_skill": {
            "skill_id": "skill_conceptual",
            "focus_dimension": "conceptual",
            "knowledge_components": ["条件概率"],
        },
        "teacher_action": {
            "question_id": "q_current",
            "message": "刚才我问的是：条件事件怎样限定当前样本空间？",
        },
    }

    context = build_layered_context(session, cue, max_chars=14_000)
    validate_layered_context(context)

    recall = context["semantic_summary"]["continuity_recall"]
    assert recall["cue_kind"] == "semantic_topic_reference"
    assert recall["status"] == "resolved_evidence_linked"
    assert recall["target"]["kind"] == "current_teacher_action"
    assert recall["target"]["source_round"] == 4
    assert recall["target"]["evidence_refs"] == ["current_action:r4:teacher_action"]
    assert "条件事件怎样限定" in recall["target"]["excerpt"]


def test_redact_remote_text_covers_all_required_identifier_types() -> None:
    posix_path = "/" + "Users/alice/private/note.txt"
    source = (
        "联系 learner@example.edu、138-0013-8000 或 +1 415-555-0199，身份证 11010519491231002X；"
        f"材料 https://school.example/a/b；本机 {posix_path}；"
        r"备份 C:\Users\Alice\secret.json。"
    )
    redacted, findings = redact_remote_text(source)

    assert "learner@example.edu" not in redacted
    assert "138-0013-8000" not in redacted
    assert "+1 415-555-0199" not in redacted
    assert "11010519491231002X" not in redacted
    assert "https://school.example/a/b" not in redacted
    assert posix_path not in redacted
    assert r"C:\Users\Alice\secret.json" not in redacted
    assert redacted.count("[REDACTED_LOCAL_PATH]") == 2
    assert [item["kind"] for item in findings] == [
        "email",
        "phone",
        "phone",
        "cn_id",
        "url",
        "local_path",
        "local_path",
    ]
    assert all("value" not in item and "text" not in item for item in findings)


def test_redaction_is_deterministic_preserves_safe_text_and_resolves_overlap() -> None:
    safe = "学生回答：dp[i] 需要同时查看前两个状态。"
    assert redact_remote_text(safe) == (safe, [])

    source = (
        "打开 file://" + "/" + "Users/alice/private.txt，然后访问 www.example.org/a。"
    )
    first = redact_remote_text(source)
    second = redact_remote_text(source)
    assert first == second
    redacted, findings = first
    assert redacted.count("[REDACTED_URL]") == 2
    assert [item["kind"] for item in findings] == ["url", "url"]
    assert "[REDACTED_LOCAL_PATH]" not in redacted


def test_redaction_preserves_formula_division_but_still_removes_real_paths() -> None:
    private_path = "/" + "Users" + "/alice/private/answer.png"
    source = (
        "比例 dp[i]/dp[i-1]、f(n)/g(n)、(a+b)/(c+d) 与 P(A)/P(B) 都是公式；"
        f"本机证据位于 {private_path}。"
    )

    redacted, findings = redact_remote_text(source)

    assert "dp[i]/dp[i-1]" in redacted
    assert "f(n)/g(n)" in redacted
    assert "(a+b)/(c+d)" in redacted
    assert "P(A)/P(B)" in redacted
    assert private_path not in redacted
    assert redacted.count("[REDACTED_LOCAL_PATH]") == 1
    assert [item["kind"] for item in findings] == ["local_path"]


def test_redaction_respects_digit_boundaries_and_rejects_non_text() -> None:
    source = "短号 1234567890，17位 12345678901234567，长串 991380013800077。"
    assert redact_remote_text(source) == (source, [])
    with pytest.raises(TypeError, match="string"):
        redact_remote_text(123)  # type: ignore[arg-type]


def test_relevant_history_prioritizes_kc_then_focus_and_keeps_latest_turns() -> None:
    session = _session()
    original = deepcopy(session)
    context = build_relevant_history(
        session,
        "我还是不理解状态转移",
        max_recent_turns=4,
        max_chars=8000,
    )

    assert context["schema"] == CONTEXT_SCHEMA
    selected_rounds = [item["round"] for item in context["recent_turns"]]
    assert selected_rounds == [4, 5, 7, 8]
    assert context["selection"]["selected_turn_count"] == 4
    assert context["earlier_summary"]["turn_count"] == 4
    assert context["earlier_summary"]["signal_counts"] == {
        "confused": 1,
        "correct": 2,
        "partial": 1,
    }
    assert session == original


def test_relevant_history_does_not_treat_every_goal_kc_as_current() -> None:
    session = _session()
    session["goal"]["knowledge_components"] = [
        "递归",
        "状态定义",
        "状态转移",
        "最优子结构",
    ]
    context = build_relevant_history(
        session,
        "继续检查状态转移",
        max_recent_turns=4,
        max_chars=8000,
    )

    assert context["current_knowledge_components"] == ["状态转移"]
    assert [item["round"] for item in context["recent_turns"]] == [4, 5, 7, 8]


@pytest.mark.parametrize(
    ("query", "max_recent_turns", "expected_rounds", "expected_cue_kind"),
    [
        (
            "我想回到第1轮讲的递归",
            3,
            [1, 7, 8],
            "explicit_round_reference",
        ),
        (
            "重新解释一下递归与状态定义的关系",
            4,
            [1, 2, 7, 8],
            "semantic_topic_reference",
        ),
        (
            "刚才第三个边界条件问题",
            3,
            [3, 7, 8],
            "semantic_topic_reference",
        ),
    ],
)
def test_query_aware_history_recalls_explicit_old_turns(
    query: str,
    max_recent_turns: int,
    expected_rounds: list[int],
    expected_cue_kind: str,
) -> None:
    session = _session()
    session["student_profile"] = {
        "profile_ref": "synthetic-query-recall",
        "learner_level": "beginner",
        "preferences": [],
        "accessibility_needs": [],
        "initial_mastery": {},
        "known_misconceptions": [],
        "conversation_history": [],
        "background_history": [],
    }
    session["teaching_memory"] = initialize_teaching_memory(
        session["goal"], session["student_profile"]
    )

    relevant = build_relevant_history(
        session,
        query,
        max_recent_turns=max_recent_turns,
        max_chars=8_000,
    )
    assert [item["round"] for item in relevant["recent_turns"]] == expected_rounds

    layered = build_layered_context(
        session,
        query,
        max_recent_turns=max_recent_turns,
        max_chars=14_000,
    )
    validate_layered_context(layered)
    recall = layered["semantic_summary"]["continuity_recall"]
    assert recall["cue_kind"] == expected_cue_kind
    assert recall["status"] == "resolved_evidence_linked"
    assert recall["target"]["source_round"] in expected_rounds[:-2]
    ledger_ids = {item["evidence_id"] for item in layered["evidence_ledger"]}
    assert set(recall["target"]["evidence_refs"]) <= ledger_ids


def test_correct_same_kc_clears_an_older_cross_focus_unresolved_event() -> None:
    session = _session()
    session["history"] = [
        _event(
            1,
            focus="prerequisite",
            kc="状态转移",
            signal="misconception",
            learner_response="状态只看前一步。",
        ),
        _event(
            2,
            focus="conceptual",
            kc="状态转移",
            signal="correct",
            learner_response="状态还可以依赖前两步，我已修正。",
        ),
    ]
    context = build_layered_context(session, "继续", max_chars=14_000)

    assert context["knowledge_state"]["unresolved_issues"] == []


def test_correct_different_kc_does_not_clear_same_focus_unresolved_event() -> None:
    session = _session()
    session["history"] = [
        _event(
            1,
            focus="conceptual",
            kc="状态定义",
            signal="misconception",
            learner_response="状态就是当前输入值。",
        ),
        _event(
            2,
            focus="conceptual",
            kc="状态转移",
            signal="correct",
            learner_response="转移同时考虑两个前驱状态。",
        ),
    ]

    context = build_layered_context(session, "继续", max_chars=14_000)
    unresolved = context["knowledge_state"]["unresolved_issues"]

    assert len(unresolved) == 1
    assert unresolved[0]["knowledge_components"] == ["状态定义"]
    assert unresolved[0]["observed_signal"] == "misconception"


def test_history_context_redacts_outbound_text_without_leaking_findings() -> None:
    session = _session()
    posix_path = "/" + "Volumes/Drive/private.txt"
    session["history"][-1]["learner_response"] = (
        f"我的邮箱 learner@example.edu，文件 {posix_path}"
    )
    context = build_relevant_history(
        session,
        "手机号 13800138000，详情 https://example.test/student/1",
    )
    payload = json.dumps(context, ensure_ascii=False)

    for secret in (
        "learner@example.edu",
        posix_path,
        "13800138000",
        "https://example.test/student/1",
    ):
        assert secret not in payload
    assert context["privacy"]["remote_text_redacted"] is True
    assert context["privacy"]["finding_counts"] == {
        "email": 1,
        "local_path": 1,
        "phone": 1,
        "url": 1,
    }
    assert context["privacy"]["original_values_retained"] is False
    assert context["privacy"]["known_pattern_identifiers_redacted"] is True
    assert context["privacy"]["raw_identity_fields_sent"] == "not_established"
    assert context["privacy"]["residual_identity_risk"] is True


def test_history_summary_redacts_sensitive_metadata_from_omitted_turns() -> None:
    session = _session()
    session["history"][0]["action"]["primary_skill"]["skill_id"] = "learner@example.edu"
    context = build_relevant_history(
        session,
        "next",
        max_recent_turns=0,
    )
    payload = json.dumps(context, ensure_ascii=False)

    assert "learner@example.edu" not in payload
    assert context["earlier_summary"]["skill_counts"] == {
        "[REDACTED_EMAIL]": 1,
        "skill_conceptual": 3,
        "skill_procedural": 2,
        "skill_transfer": 2,
    }


def test_layered_context_redacts_pii_in_mapping_keys_without_collision_loss() -> None:
    session = _session()
    session["goal"]["materials"] = {
        "learner@example.com": "first",
        "teacher@example.com": "second",
        "practice": "safe material",
    }

    context = build_layered_context(session, "继续", max_chars=14_000)
    payload = json.dumps(context, ensure_ascii=False)
    materials = context["fixed_context"]["teaching_goal"]["materials"]

    assert "learner@example.com" not in payload
    assert "teacher@example.com" not in payload
    assert materials["[REDACTED_EMAIL]"] == "first"
    assert materials["[REDACTED_EMAIL]__2"] == "second"
    assert materials["practice"] == "safe material"
    assert context["privacy"]["remote_text_redacted"] is True
    assert context["privacy"]["finding_counts"]["email"] == 2


def test_history_context_never_exceeds_character_budget() -> None:
    session = _session()
    for item in session["history"]:
        item["learner_response"] = "回答" * 2500
        item["action"]["teacher_action"]["message"] = "提示" * 2500
    response = "当前回答" * 2500

    for budget in (8000, 2500, 1000, 640):
        context = build_relevant_history(
            session,
            response,
            max_recent_turns=6,
            max_chars=budget,
        )
        assert _json_length(context) <= budget
        assert context["selection"]["truncated"] is True


def test_history_context_handles_empty_history_and_zero_recent_turns() -> None:
    session = _session()
    session["history"] = []
    empty = build_relevant_history(session, "first response", max_recent_turns=0)
    assert empty["recent_turns"] == []
    assert empty["earlier_summary"]["turn_count"] == 0

    summarized = build_relevant_history(_session(), "next", max_recent_turns=0)
    assert summarized["recent_turns"] == []
    assert summarized["earlier_summary"]["turn_count"] == 8


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_recent_turns": True}, TypeError),
        ({"max_recent_turns": -1}, ValueError),
        ({"max_recent_turns": 101}, ValueError),
        ({"max_chars": True}, TypeError),
        ({"max_chars": 639}, ValueError),
    ],
)
def test_history_context_validates_bounds(kwargs: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        build_relevant_history(_session(), "answer", **kwargs)


def test_build_goal_plan_produces_verifiable_bounded_plan() -> None:
    goal = {
        "concept": "动态规划的状态与转移",
        "objective": "能够定义状态、写出转移并判断新问题是否适用。",
        "success_thresholds": {
            "prerequisite": 0.6,
            "conceptual": 0.7,
            "procedural": 0.65,
            "transfer": 0.55,
        },
        "materials": {
            "practice": "写出爬楼梯问题的状态转移",
            "transfer_task": "判断硬币问题是否适合动态规划",
        },
    }
    plan = build_goal_plan(goal)

    assert plan["schema"] == GOAL_PLAN_SCHEMA
    assert 2 <= len(plan["intermediate_objectives"]) <= 7
    assert plan["active_step"] == "goal_step_01"
    assert plan["progress"] == {
        "completed_steps": 0,
        "total_steps": 4,
        "fraction": 0.0,
    }
    assert [item["status"] for item in plan["intermediate_objectives"]] == [
        "active",
        "pending",
        "pending",
        "pending",
    ]
    assert [item["dimension"] for item in plan["intermediate_objectives"]] == [
        "prerequisite",
        "conceptual",
        "procedural",
        "transfer",
    ]
    assert all(
        item["verification"]["success_criterion"].strip()
        and item["verification"]["method"]
        == "observable_learner_response_or_task_performance"
        for item in plan["intermediate_objectives"]
    )
    assert plan["claim_boundary"]["plan_is_learning_effect_evidence"] is False


def test_goal_plan_is_deterministic_and_uses_defaults_without_materials() -> None:
    goal = {"concept": "二分查找", "objective": "理解循环不变量。"}
    assert build_goal_plan(goal) == build_goal_plan(deepcopy(goal))
    plan = build_goal_plan(goal)
    assert len(plan["intermediate_objectives"]) == 4
    assert (
        "代表性练习"
        in plan["intermediate_objectives"][2]["verification"]["success_criterion"]
    )


@pytest.mark.parametrize(
    "goal",
    [
        {},
        {"concept": "", "objective": "x"},
        {"concept": "x", "objective": ""},
        {"concept": "x", "objective": "y", "success_thresholds": []},
        {
            "concept": "x",
            "objective": "y",
            "success_thresholds": {"conceptual": 1.1},
        },
        {"concept": "x", "objective": "y", "materials": []},
    ],
)
def test_goal_plan_rejects_invalid_contracts(goal: dict) -> None:
    with pytest.raises(ValueError):
        build_goal_plan(goal)


def test_goal_plan_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        build_goal_plan([])  # type: ignore[arg-type]


def _production_session_without_injected_kcs() -> dict:
    library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
    session = start_teacher_agent_session(
        {
            "concept": "动态规划的状态与转移",
            "objective": "能够定义状态、写出转移并判断新问题是否适用。",
            "max_rounds": 12,
            "materials": {"practice": "写出爬楼梯问题的状态转移"},
        },
        {
            "learner_level": "beginner",
            "preferences": ["直观例子"],
            "initial_mastery": {
                "prerequisite": 0.2,
                "conceptual": 0.1,
                "procedural": 0.0,
                "transfer": 0.0,
            },
            "known_misconceptions": [],
            "conversation_history": [],
        },
        library,
    )
    return advance_teacher_agent_session(
        session,
        learner_response="我知道递归，但还不会定义状态。",
        signal="partial",
        signal_confidence=0.8,
    )


def test_layered_context_uses_real_goal_anchor_and_separates_memory_layers() -> None:
    session = _production_session_without_injected_kcs()
    response = "UNIQUE_CURRENT_ANSWER_7d35 我认为状态只需要看上一步。"
    context = build_layered_context(session, response)

    validate_layered_context(context)
    assert context["schema"] == LAYERED_CONTEXT_SCHEMA
    assert session["goal"]["knowledge_components"] == ["动态规划的状态与转移"]
    assert context["working_memory"]["current_knowledge_components"] == [
        "动态规划的状态与转移"
    ]
    assert context["fixed_context"]["teaching_goal"]["mutable_by_model"] is False
    assert context["semantic_summary"]["narrative_inference_added"] is False
    assert (
        context["semantic_summary"]["compression_method"]
        == "deterministic_aggregate_and_extractive_checkpoints_no_model_generation"
    )
    assert context["candidate_long_term_memory"]["status"] == "candidate_unconfirmed"
    assert (
        context["candidate_long_term_memory"]["may_override_teacher_profile"] is False
    )
    assert context["budget"]["serialized_chars"] == _json_length(context)
    assert context["budget"]["serialized_chars"] <= context["budget"]["max_chars"]
    # The current answer has one authority location.  Its ledger row is only a
    # pointer, which avoids silently weighting the same evidence twice.
    assert (
        json.dumps(context, ensure_ascii=False).count("UNIQUE_CURRENT_ANSWER_7d35") == 1
    )


def test_layered_context_reports_residual_identity_risk_honestly() -> None:
    session = _production_session_without_injected_kcs()
    session["student_profile"]["background_history"] = [
        "联系 learner@example.edu 获取旧作业。"
    ]
    context = build_layered_context(session, "我继续回答当前问题。")
    payload = json.dumps(context, ensure_ascii=False)

    validate_layered_context(context)
    assert "learner@example.edu" not in payload
    assert context["privacy"]["known_pattern_identifiers_redacted"] is True
    assert context["privacy"]["raw_identity_fields_sent"] == "not_established"
    assert context["privacy"]["residual_identity_risk"] is True

    dishonest = deepcopy(context)
    dishonest["privacy"]["raw_identity_fields_sent"] = False
    dishonest["budget"]["serialized_chars"] = _json_length(dishonest)
    with pytest.raises(ValueError, match="privacy metadata"):
        validate_layered_context(dishonest)


def test_layered_context_retrieval_and_budget_are_deterministic() -> None:
    session = _production_session_without_injected_kcs()
    for index in range(2, 11):
        session = advance_teacher_agent_session(
            session,
            learner_response=f"第 {index} 轮仍需检查状态定义。" + "解释" * 120,
            signal="partial" if index % 2 else "confused",
            signal_confidence=0.7,
        )
        if session["status"] != "active":
            break
    first = build_layered_context(session, "继续检查我的理解。", max_chars=8_000)
    second = build_layered_context(session, "继续检查我的理解。", max_chars=8_000)

    assert first == second
    assert first["budget"]["serialized_chars"] <= 8_000
    assert first["budget"]["truncated"] is True
    assert first["retrieval"]["fixed_context_always_included"] is True
    assert first["retrieval"]["omitted_turns_accounted_for_statistically"] is True
    assert first["retrieval"]["teaching_checkpoints_are_selective_extracts"] is True
    assert "semantic_summary_covers_omitted_turns" not in first["retrieval"]
    assert first["claim_boundary"]["omitted_turn_semantics_are_exhaustive"] is False
    assert first["semantic_summary"]["turn_count"] >= 1
    assert first["semantic_summary"]["focus_checkpoints"]
    assert all(
        item["evidence_refs"] for item in first["semantic_summary"]["focus_checkpoints"]
    )


@pytest.mark.parametrize(
    "learner_question",
    [
        "递推关系分哪几部分",
        "递推关系分为几部分",
        "递推关系由什么组成",
        "递推关系包括哪些部分",
    ],
)
def test_unpunctuated_composition_question_becomes_explicit_checkpoint(
    learner_question: str,
) -> None:
    evidence: list[dict] = []
    checkpoints = _teaching_checkpoints(
        [
            {
                "round": 1,
                "focus_dimension": "conceptual",
                "knowledge_components": ["递推关系"],
                "signal": "not_recorded",
                "learner_response": learner_question,
                "teacher_message": "",
            }
        ],
        selected_rounds=set(),
        content_limit=240,
        evidence_excerpt_limit=240,
        evidence=evidence,
    )

    checkpoint = next(
        item for item in checkpoints if item["kind"] == "explicit_learner_question"
    )
    assert checkpoint["excerpt"] == learner_question
    assert checkpoint["status"] == "resolution_not_established"
    assert checkpoint["evidence_refs"] == ["session_history:r1:learner_response"]


@pytest.mark.parametrize(
    "learner_statement",
    [
        "递推关系分为三个部分",
        "递推关系由初始条件、递推规则和适用范围组成",
        "递推关系包括初始条件和递推规则",
        "我把这个问题分成几个部分了",
    ],
)
def test_composition_declaration_is_not_an_explicit_question_checkpoint(
    learner_statement: str,
) -> None:
    checkpoints = _teaching_checkpoints(
        [
            {
                "round": 1,
                "focus_dimension": "conceptual",
                "knowledge_components": ["递推关系"],
                "signal": "not_recorded",
                "learner_response": learner_statement,
                "teacher_message": "",
            }
        ],
        selected_rounds=set(),
        content_limit=240,
        evidence_excerpt_limit=240,
        evidence=[],
    )

    assert all(item["kind"] != "explicit_learner_question" for item in checkpoints)


def test_layered_context_keeps_selective_evidence_linked_teaching_checkpoints() -> None:
    session = _session()
    prerequisite = _event(
        1,
        focus="prerequisite",
        kc="递归",
        signal="correct",
        learner_response="递归会把原问题化成更小的同类问题。",
    )
    preference = _event(
        2,
        focus="conceptual",
        kc="状态定义",
        signal="partial",
        learner_response="请先用一个图解释，不要直接给最终答案。",
    )
    question = _event(
        3,
        focus="conceptual",
        kc="状态转移",
        signal="confused",
        learner_response="为什么这个状态必须同时查看两个前驱？",
    )
    unresolved = _event(
        4,
        focus="procedural",
        kc="边界条件",
        signal="misconception",
        learner_response="边界条件可以最后再补。",
    )
    commitment = _event(
        5,
        focus="conceptual",
        kc="最优子结构",
        signal="correct",
        learner_response="我能说明局部最优与整体最优的关系。",
    )
    commitment["action"]["teacher_action"]["message"] = (
        "接下来我会让你用一个反例检查这个条件。"
    )
    latest = _event(
        6,
        focus="transfer",
        kc="新情境",
        signal="correct",
        learner_response="我可以在新题里先找状态。",
    )
    session["history"] = [
        prerequisite,
        preference,
        question,
        unresolved,
        commitment,
        latest,
    ]
    session["student_state"]["next_focus"] = {
        "dimension": "transfer",
        "knowledge_component": "新情境",
    }
    session["current_action"]["primary_skill"] = {
        "focus_dimension": "transfer",
        "knowledge_components": ["新情境"],
    }

    context = build_layered_context(
        session,
        "继续做当前迁移题。",
        max_chars=14_000,
        max_recent_turns=1,
    )
    validate_layered_context(context)

    checkpoints = context["semantic_summary"]["teaching_checkpoints"]
    kinds = [item["kind"] for item in checkpoints]
    assert kinds.count("unresolved_learning_signal") == 2
    assert "explicit_learner_question" in kinds
    assert "explicit_learner_preference_or_constraint" in kinds
    assert "verified_prerequisite" in kinds
    assert "teacher_next_step_statement" in kinds
    ledger_ids = {item["evidence_id"] for item in context["evidence_ledger"]}
    assert all(set(item["evidence_refs"]) <= ledger_ids for item in checkpoints)
    assert (
        next(
            item for item in checkpoints if item["kind"] == "explicit_learner_question"
        )["status"]
        == "resolution_not_established"
    )
    assert (
        next(
            item
            for item in checkpoints
            if item["kind"] == "teacher_next_step_statement"
        )["status"]
        == "completion_not_established"
    )


def test_later_correct_signal_clears_omitted_unresolved_checkpoint() -> None:
    session = _session()
    session["history"] = [
        _event(
            1,
            focus="conceptual",
            kc="状态定义",
            signal="misconception",
            learner_response="状态就是当前输入。",
        ),
        _event(
            2,
            focus="conceptual",
            kc="状态定义",
            signal="correct",
            learner_response="状态是子问题的最小充分描述。",
        ),
        _event(
            3,
            focus="transfer",
            kc="新情境",
            signal="correct",
        ),
    ]
    session["student_state"]["next_focus"] = {
        "dimension": "transfer",
        "knowledge_component": "新情境",
    }
    session["current_action"]["primary_skill"] = {
        "focus_dimension": "transfer",
        "knowledge_components": ["新情境"],
    }

    context = build_layered_context(
        session,
        "继续",
        max_chars=14_000,
        max_recent_turns=1,
    )

    assert all(
        item["kind"] != "unresolved_learning_signal"
        for item in context["semantic_summary"]["teaching_checkpoints"]
    )


def test_layered_context_rejects_dangling_evidence_and_false_claims() -> None:
    context = build_layered_context(
        _production_session_without_injected_kcs(),
        "我还没有完全理解。",
    )
    dangling = deepcopy(context)
    dangling["knowledge_state"]["concept_mastery"][0]["evidence_refs"] = [
        "missing:evidence"
    ]
    dangling["budget"]["serialized_chars"] = _json_length(dangling)
    with pytest.raises(ValueError, match="dangling"):
        validate_layered_context(dangling)

    fabricated = deepcopy(context)
    fabricated["claim_boundary"]["model_generated_history_summary"] = True
    fabricated["budget"]["serialized_chars"] = _json_length(fabricated)
    with pytest.raises(ValueError, match="claim boundary"):
        validate_layered_context(fabricated)

    exhaustive = deepcopy(context)
    exhaustive["claim_boundary"]["omitted_turn_semantics_are_exhaustive"] = True
    exhaustive["budget"]["serialized_chars"] = _json_length(exhaustive)
    with pytest.raises(ValueError, match="claim boundary"):
        validate_layered_context(exhaustive)

    legacy_overclaim = deepcopy(context)
    legacy_overclaim["retrieval"]["semantic_summary_covers_omitted_turns"] = True
    legacy_overclaim["budget"]["serialized_chars"] = _json_length(legacy_overclaim)
    with pytest.raises(ValueError, match="retrieval metadata"):
        validate_layered_context(legacy_overclaim)


def test_minimum_budget_retains_current_answer_for_large_legal_session_shape() -> None:
    session = _session()
    session["goal"].update(
        {
            "objective": "检查复杂会话的最小上下文。",
            "knowledge_components": [f"知识点{i}" * 40 for i in range(12)],
            "success_thresholds": {
                "prerequisite": 0.6,
                "conceptual": 0.65,
                "procedural": 0.6,
                "transfer": 0.55,
            },
            "max_rounds": 50,
            "materials": {f"材料{i}": "内容" * 500 for i in range(12)},
        }
    )
    session["student_profile"] = {
        "learner_level": "beginner",
        "preferences": ["偏好" * 300 for _ in range(20)],
        "accessibility_needs": ["需求" * 300 for _ in range(20)],
        "initial_mastery": {
            "prerequisite": 0.1,
            "conceptual": 0.1,
            "procedural": 0.0,
            "transfer": 0.0,
        },
        "known_misconceptions": [
            {"tag": f"m{i}", "description": "描述" * 300, "confidence": 0.8}
            for i in range(8)
        ],
        "conversation_history": [
            {
                "response": "旧回答" * 300,
                "signal": "partial",
                "focus_dimension": "conceptual",
            }
            for _ in range(20)
        ],
        "adaptive_observations": [],
        "adaptive_summary": {
            "total_observation_count": 0,
            "needs_human_review": False,
        },
    }
    session["student_state"].update(
        {
            "knowledge_mastery": {
                "prerequisite": 0.1,
                "conceptual": 0.2,
                "procedural": 0.1,
                "transfer": 0.0,
            },
            "misconceptions": [
                {
                    "tag": f"m{i}",
                    "description": "错误描述" * 200,
                    "confidence": 0.8,
                    "status": "active",
                    "last_observed_round": 8,
                }
                for i in range(12)
            ],
            "understanding_signal": {
                "label": "partial",
                "confidence": 0.8,
                "source": "deepseek_v4_flash",
            },
            "assessment_evidence": {
                "source": "deepseek_v4_flash",
                "needs_human_review": False,
            },
        }
    )
    template = deepcopy(session["history"])
    session["history"] = []
    for index in range(50):
        event = deepcopy(template[index % len(template)])
        event["round"] = index + 1
        event["learner_response"] = "历史回答" * 500
        event["structured_signal"]["source"] = "deepseek_v4_flash"
        session["history"].append(event)

    response = "UNIQUE_MINIMUM_CONTEXT_43c1" + "当前回答" * 1000
    context = build_layered_context(
        session,
        response,
        max_chars=6_000,
        max_recent_turns=12,
    )

    validate_layered_context(context)
    assert context["budget"]["serialized_chars"] <= 6_000
    assert context["budget"]["truncated"] is True
    assert context["retrieval"]["priority"] == (
        "minimum_safety_envelope_after_budget_degradation"
    )
    assert context["semantic_summary"]["teaching_checkpoints"] == []
    assert context["retrieval"]["omitted_turns_accounted_for_statistically"] is True
    assert context["claim_boundary"]["omitted_turn_semantics_are_exhaustive"] is False
    assert context["working_memory"]["current_knowledge_components"] == ["状态转移"]
    assert (
        "UNIQUE_MINIMUM_CONTEXT_43c1"
        in (context["working_memory"]["current_learner_response"])
    )
    assert (
        json.dumps(context, ensure_ascii=False).count("UNIQUE_MINIMUM_CONTEXT_43c1")
        == 1
    )


def test_current_response_truncation_preserves_decisive_tail() -> None:
    session = _session()
    response = (
        "我先解释思路：" + "中间推理" * 1200 + "；最终答案是 dp[i]=dp[i-1]+dp[i-2]。"
    )

    context = build_layered_context(
        session,
        response,
        max_chars=6_000,
        max_recent_turns=2,
    )

    retained = context["working_memory"]["current_learner_response"]
    assert retained.startswith("我先解释思路")
    assert "middle truncated" in retained
    assert retained.endswith("最终答案是 dp[i]=dp[i-1]+dp[i-2]。")
    assert context["budget"]["serialized_chars"] <= 6_000


def test_maximum_legal_question_contract_is_bounded_at_minimum_budget() -> None:
    session = _session()
    session["current_action"]["teacher_action"] = {
        "message": "请回答当前问题。",
        "expected_signal": "学生直接回答本问。",
        "question_id": "question-large-contract",
        "question_contract": {
            "answer_type": "comparison",
            "target_concepts": ["目标" * 80 for _ in range(8)],
            "accepted_aliases": ["别名" * 80 for _ in range(12)],
            "success_criteria": ["判据" * 120 for _ in range(8)],
            "grading_scope": "current_question_only",
        },
    }

    context = build_layered_context(
        session,
        "这是当前回答。",
        max_chars=6_000,
        max_recent_turns=12,
    )

    validate_layered_context(context)
    contract = context["current_plan"]["current_action"]["question_contract"]
    assert context["budget"]["serialized_chars"] <= 6_000
    assert len(contract["target_concepts"]) <= 4
    assert len(contract["accepted_aliases"]) <= 8
    assert len(contract["success_criteria"]) <= 4
