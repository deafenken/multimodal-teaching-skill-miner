from __future__ import annotations

from copy import deepcopy
import json

import pytest

from teaching_skill_miner.teacher_agent_context import (
    CONTEXT_SCHEMA,
    GOAL_PLAN_SCHEMA,
    build_goal_plan,
    build_relevant_history,
    redact_remote_text,
)


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


def test_redact_remote_text_covers_all_required_identifier_types() -> None:
    posix_path = "/" + "Users/alice/private/note.txt"
    source = (
        "联系 learner@example.edu 或 138-0013-8000，身份证 11010519491231002X；"
        f"材料 https://school.example/a/b；本机 {posix_path}；"
        r"备份 C:\Users\Alice\secret.json。"
    )
    redacted, findings = redact_remote_text(source)

    assert "learner@example.edu" not in redacted
    assert "138-0013-8000" not in redacted
    assert "11010519491231002X" not in redacted
    assert "https://school.example/a/b" not in redacted
    assert posix_path not in redacted
    assert r"C:\Users\Alice\secret.json" not in redacted
    assert redacted.count("[REDACTED_LOCAL_PATH]") == 2
    assert [item["kind"] for item in findings] == [
        "email",
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
