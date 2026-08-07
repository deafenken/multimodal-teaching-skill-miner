from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig
from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.teacher_agent import (
    TeacherAgentError,
    _refresh_integrity,
    canonical_sha256,
)
from teaching_skill_miner.teacher_agent_context import (
    build_layered_context,
    validate_layered_context,
)
from teaching_skill_miner.teacher_agent_live import (
    LiveAgentOptions,
    advance_live_teacher_agent_session,
    live_session_view,
    start_live_teacher_agent_session,
)
from teaching_skill_miner.teacher_agent_memory import (
    TEACHING_MEMORY_PROJECTION_SCHEMA,
    TEACHING_MEMORY_SCHEMA,
    initialize_teaching_memory,
    project_teaching_memory,
)


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
DEMO = read_json(ROOT / "data/teacher_agent_demo_input.json")
ACTION_TYPES = {
    str(skill["skill_id"]): str(skill["action_type"])
    for skill in LIBRARY["skills"]
}
FOCUS_BY_SKILL = {
    str(skill["skill_id"]): str(skill["focus_dimension"])
    for skill in LIBRARY["skills"]
}


def _plan(
    *,
    signal: str,
    confidence: float,
    skill_id: str,
    message: str = "请先说明你判断这一步的依据是什么？",
    evidence_excerpt: str = "",
    question_contract: dict | None = None,
    expected_signal: str = "学生能说明当前判断依据，并指出一个可核验条件。",
    should_stop: bool = False,
    needs_human_review: bool = False,
) -> dict:
    answer_alignment = {
        "not_observed": "not_applicable",
        "correct": "aligned",
        "partial": "partially_aligned",
        "misconception": "contradicted",
        "confused": "ambiguous",
        "no_response": "no_response",
    }[signal]
    return {
        "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
        "diagnosis": {
            "signal": signal,
            "confidence": confidence,
            "answer_alignment": answer_alignment,
            "matched_concepts": [],
            "missing_concepts": [],
            "diagnosis_reason": "学生回答提供了当前轮次的可定位证据。",
            "evidence_excerpt": evidence_excerpt,
            "misconception_tag": None,
            "misconception_description": "",
            "resolved_misconception_tags": [],
            "response_quality": (
                "empty" if signal == "not_observed" else "partial"
            ),
            "engagement_level": (
                "unknown" if signal == "not_observed" else "medium"
            ),
            "needs_human_review": needs_human_review,
        },
        "decision": {
            "primary_skill_id": skill_id,
            "supporting_skill_ids": [],
            "selection_reason": "根据当前状态选择与证据相符的主 Skill。",
            "next_focus": FOCUS_BY_SKILL.get(skill_id, "conceptual"),
        },
        "teacher_action": {
            "type": ACTION_TYPES.get(skill_id, "ask_one_question"),
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
        "stop_recommendation": {
            "should_stop": should_stop,
            "reason": "连续两轮无进展，需要人工诊断" if should_stop else "",
        },
    }


def _initial_plan() -> dict:
    return _plan(
        signal="not_observed",
        confidence=0.0,
        skill_id="skill_diagnostic_questioning",
        message=(
            "开始前，请说出一个与当前目标有关的必要前置概念，"
            "并用一个最小例子说明它的作用。"
        ),
        expected_signal="学生说出一个必要前置概念，并用最小例子说明其作用。",
        question_contract={
            "answer_type": "example",
            "target_concepts": ["任一与当前教学目标相关的必要前置概念"],
            "accepted_aliases": [],
            "success_criteria": [
                "明确说出一个必要前置概念",
                "给出一个说明该概念作用的最小例子",
            ],
        },
    )


def _client(
    plans: list[dict], *, counter: dict[str, int] | None = None
) -> DeepSeekClient:
    queue = deque(deepcopy(plans))
    calls = counter if counter is not None else {"calls": 0}

    def transport(_url: str, _headers: dict, _payload: bytes, _timeout: float):
        calls["calls"] = calls.get("calls", 0) + 1
        content = queue.popleft()
        envelope = {
            "id": "live_memory_test",
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


def _assert_memory_checkpoint(session: dict, expected_round: int) -> None:
    memory = session["teaching_memory"]
    trace = session["history"][-1]["teaching_memory_trace"]
    assert session["round"] == expected_round
    assert memory["history_version"] == expected_round
    assert memory["last_observed_round"] == expected_round
    assert memory["compaction_generation"] == expected_round // 8
    assert trace == {
        "history_version": expected_round,
        "compaction_generation": expected_round // 8,
        "fixed_context_fingerprint": memory["fixed_context_fingerprint"],
        "content_sha256": canonical_sha256(memory),
        "source": "deterministic_evidence_linked_rollout_projection",
        "model_generated_summary": False,
    }


def test_live_start_initializes_goal_profile_bound_teaching_memory() -> None:
    session = start_live_teacher_agent_session(
        DEMO["goal"], DEMO["student_profile"], LIBRARY, _client([_initial_plan()])
    )

    memory = session["teaching_memory"]
    expected = initialize_teaching_memory(
        session["goal"], session["student_profile"]
    )
    assert memory["schema"] == TEACHING_MEMORY_SCHEMA
    assert memory["history_version"] == 0
    assert memory["last_observed_round"] == 0
    assert memory["compaction_generation"] == 0
    assert memory["fixed_context_fingerprint"] == expected[
        "fixed_context_fingerprint"
    ]
    assert session["history"] == []
    assert (
        session["context_memory"]["semantic_summary"]["teaching_memory"]
        == project_teaching_memory(memory)
    )


def test_valid_model_turns_commit_monotonic_memory_and_history_trace() -> None:
    plans = [
        _initial_plan(),
        _plan(
            signal="partial",
            confidence=0.8,
            skill_id="skill_concrete_example_bridge",
        ),
        _plan(
            signal="partial",
            confidence=0.8,
            skill_id="skill_stepwise_scaffolding",
        ),
        _plan(
            signal="partial",
            confidence=0.8,
            skill_id="skill_concept_mapping",
        ),
    ]
    client = _client(plans)
    session = start_live_teacher_agent_session(
        DEMO["goal"], DEMO["student_profile"], LIBRARY, client
    )

    fingerprints: list[str] = []
    for round_number in range(1, 4):
        session = advance_live_teacher_agent_session(
            session,
            learner_response=(
                f"第 {round_number} 轮我能说出部分状态关系，但还需要继续核对。"
            ),
            client=client,
        )
        _assert_memory_checkpoint(session, round_number)
        fingerprints.append(session["teaching_memory"]["fixed_context_fingerprint"])
        assert session["agent_runtime"]["fallback_count"] == 0

    assert len(set(fingerprints)) == 1
    assert [
        event["teaching_memory_trace"]["history_version"]
        for event in session["history"]
    ] == [1, 2, 3]


def test_rule_fallback_still_commits_exactly_one_memory_checkpoint() -> None:
    session = start_live_teacher_agent_session(
        DEMO["goal"], DEMO["student_profile"], LIBRARY, _client([_initial_plan()])
    )

    with patch(
        "teaching_skill_miner.teacher_agent_live.build_layered_context",
        side_effect=ValueError("injected context failure"),
    ):
        updated = advance_live_teacher_agent_session(
            session,
            learner_response="上下文构建失败时，请先举例，不要直接上公式。",
            client=_client([]),
        )

    _assert_memory_checkpoint(updated, 1)
    assert updated["agent_runtime"]["fallback_count"] == 1
    assert (
        updated["history"][-1]["structured_signal"]["source"]
        == "deterministic_safety_fallback"
    )
    assert any(
        "不要直接上公式" in item["statement"]
        for item in updated["teaching_memory"]["preferences"]
    )


def test_guarded_terminal_turn_preserves_monotonic_memory_trace() -> None:
    first = _plan(
        signal="confused",
        confidence=0.7,
        skill_id="skill_concrete_example_bridge",
        needs_human_review=True,
        should_stop=True,
    )
    second = _plan(
        signal="confused",
        confidence=0.7,
        skill_id="skill_engagement_recovery",
        needs_human_review=True,
        should_stop=True,
    )
    client = _client([_initial_plan(), first, second])
    session = start_live_teacher_agent_session(
        DEMO["goal"], DEMO["student_profile"], LIBRARY, client
    )

    session = advance_live_teacher_agent_session(
        session, learner_response="我还是不懂。", client=client
    )
    assert session["status"] == "active"
    _assert_memory_checkpoint(session, 1)

    session = advance_live_teacher_agent_session(
        session, learner_response="还是完全不知道怎么开始。", client=client
    )
    assert session["status"] == "terminated_unable"
    assert session["current_action"]["decision_origin"] == "guarded_model_escalation"
    _assert_memory_checkpoint(session, 2)
    assert [
        event["teaching_memory_trace"]["history_version"]
        for event in session["history"]
    ] == [1, 2]


def test_evidence_linked_memory_survives_nine_turn_bounded_context() -> None:
    first_turn = _plan(
        signal="partial",
        confidence=0.8,
        skill_id="skill_self_explanation",
        message=(
            "第一种方法保留完整表，第二种方法只保留相邻状态。"
            "接下来我会先用例子帮助你比较二者。"
            "请用自己的话解释：为什么第二种保存的状态更少？"
        ),
        evidence_excerpt="第二种为什么更省空间",
        expected_signal="学生用自己的话解释第二种为何保存更少状态。",
        question_contract={
            "answer_type": "explanation",
            "target_concepts": ["两种状态保存方法的差异"],
            "accepted_aliases": ["第一种", "第二种"],
            "success_criteria": [
                "学生用自己的话解释第二种为何保存更少状态"
            ],
        },
    )
    later_skills = [
        "skill_concrete_example_bridge",
        "skill_stepwise_scaffolding",
        "skill_concept_mapping",
        "skill_socratic_understanding_check",
        "skill_practice_feedback",
        "skill_retrieval_review",
        "skill_self_explanation",
        "skill_contextual_problem_setup",
    ]
    client = _client(
        [_initial_plan(), first_turn]
        + [
            _plan(signal="partial", confidence=0.8, skill_id=skill_id)
            for skill_id in later_skills
        ]
    )
    options = LiveAgentOptions(maximum_context_turns=2)
    session = start_live_teacher_agent_session(
        DEMO["goal"],
        DEMO["student_profile"],
        LIBRARY,
        client,
        options=options,
    )
    responses = [
        (
            "第一种保留完整表，第二种只保留相邻状态。"
            "请先举例，不要直接上公式。第二种为什么更省空间？"
        ),
        *[
            f"第 {round_number} 轮我继续比较两个状态，但暂时只说出其中一点。"
            for round_number in range(2, 10)
        ],
    ]
    for response in responses:
        session = advance_live_teacher_agent_session(
            session, learner_response=response, client=client, options=options
        )

    _assert_memory_checkpoint(session, 9)
    context = build_layered_context(
        session,
        "第二种呢？",
        max_chars=14_000,
        max_recent_turns=2,
    )
    validate_layered_context(context)
    memory = context["semantic_summary"]["teaching_memory"]
    assert memory["history_version"] == 9
    assert memory["compaction_generation"] == 1
    assert all(
        turn.get("round") != 1
        for turn in context["working_memory"]["recent_turns"]
    )
    assert any(
        "不要直接上公式" in item["statement"]
        for item in memory["active_preferences"]
    )
    assert any(
        "第二种为什么更省空间" in item["question"]
        for item in memory["unresolved_questions"]
    )
    assert any(
        "接下来我会" in item["statement"]
        for item in memory["pending_teacher_commitments"]
    )
    assert any(
        "第一种" in item["description"] and "第二种" in item["description"]
        for item in memory["active_referents"]
    )
    ledger_ids = {item["evidence_id"] for item in context["evidence_ledger"]}
    retained_items = (
        memory["active_preferences"]
        + memory["unresolved_questions"]
        + memory["pending_teacher_commitments"]
        + memory["active_referents"]
    )
    assert {
        evidence_ref
        for item in retained_items
        for evidence_ref in item["evidence_refs"]
    } <= ledger_ids


@pytest.mark.parametrize(
    "tamper",
    [
        lambda memory: memory.__setitem__("fixed_context_fingerprint", "0" * 64),
        lambda memory: memory.__setitem__(
            "history_version", int(memory["history_version"]) + 1
        ),
        lambda memory: memory["preferences"][0]["evidence_refs"].append(
            "session_history:r999:learner_response"
        ),
    ],
    ids=[
        "wrong_goal_profile_fingerprint",
        "forged_history_version",
        "forged_future_evidence_reference",
    ],
)
def test_memory_tampering_fails_closed_before_remote_turn_processing(tamper) -> None:
    counter = {"calls": 0}
    client = _client(
        [
            _initial_plan(),
            _plan(
                signal="partial",
                confidence=0.8,
                skill_id="skill_concrete_example_bridge",
            ),
        ],
        counter=counter,
    )
    session = start_live_teacher_agent_session(
        DEMO["goal"], DEMO["student_profile"], LIBRARY, client
    )
    assert counter["calls"] == 1
    tamper(session["teaching_memory"])
    _refresh_integrity(session)

    with pytest.raises(TeacherAgentError, match="teaching_memory"):
        advance_live_teacher_agent_session(
            session,
            learner_response="这轮不应发送给远程模型。",
            client=client,
        )

    assert counter["calls"] == 1


def test_live_session_view_exposes_projection_not_raw_memory_store() -> None:
    session = start_live_teacher_agent_session(
        DEMO["goal"], DEMO["student_profile"], LIBRARY, _client([_initial_plan()])
    )
    view = live_session_view(session)
    exposed = view["teaching_memory"]

    assert exposed == project_teaching_memory(session["teaching_memory"])
    assert exposed["schema"] == TEACHING_MEMORY_PROJECTION_SCHEMA
    assert exposed["model_may_mutate"] is False
    assert exposed["narrative_inference_added"] is False
    for raw_store_field in (
        "last_observed_round",
        "preferences",
        "open_questions",
        "commitments",
        "referents",
        "claim_boundary",
    ):
        assert raw_store_field not in exposed
