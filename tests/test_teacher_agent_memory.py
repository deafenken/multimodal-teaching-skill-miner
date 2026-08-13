from copy import deepcopy

import pytest

from teaching_skill_miner.teacher_agent import canonical_sha256
from teaching_skill_miner.teacher_agent_memory import (
    TeachingMemoryError,
    commit_teaching_memory_turn,
    initialize_teaching_memory,
    project_teaching_memory,
    rebuild_teaching_memory_from_rollout,
    validate_teaching_memory_checkpoint,
    validate_teaching_memory,
)


GOAL = {
    "concept": "动态规划",
    "objective": "理解状态与转移",
}
PROFILE = {
    "profile_ref": "synthetic-a",
    "learner_level": "beginner",
    "preferences": ["先看直观例子，再看公式"],
    "accessibility_needs": [],
    "initial_mastery": {
        "prerequisite": 0.2,
        "conceptual": 0.1,
        "procedural": 0.0,
        "transfer": 0.0,
    },
    "known_misconceptions": [
        {
            "tag": "greedy_confusion",
            "description": "把局部最优直接当作全局最优",
            "confidence": 0.8,
            "status": "active",
            "first_observed_round": 0,
            "last_observed_round": 0,
        }
    ],
    "conversation_history": [
        {
            "response": "我知道递归，但不会写状态转移。",
            "signal": "partial",
            "focus_dimension": "prerequisite",
        }
    ],
    "background_history": ["学过基础递归"],
    "contains_direct_identity": False,
    "identity_assessment": "not_asserted_present_not_independently_verified",
}


def _teacher(message: str, *, answered_question: str | None = None) -> dict:
    action = {"action_id": "turn", "message": message}
    if answered_question is not None:
        action["learner_question_answer"] = {
            "schema": "teaching_skill_miner.learner_question_answer_receipt.v2",
            "status": "answered_before_low_load_check",
            "question_sha256": canonical_sha256(answered_question),
            "clarification_contract_sha256": "a" * 64,
            "clarification_kind": "composition",
            "grounding_mode": "teacher_authoritative_context",
            "grounding_refs": [
                "goal.knowledge_spec.canonical_claims:claim_recurrence_parts"
            ],
            "answer_surface_contract_validated": True,
            "server_verified_semantic_truth": False,
            "general_knowledge_is_grading_authority": False,
            "conceptual_clarification_answered": True,
            "practice_final_solution_provided": False,
            "mastery_evidence": False,
        }
    return action


def test_profile_and_explicit_preferences_survive_compaction_projection() -> None:
    memory = initialize_teaching_memory(GOAL, PROFILE)
    memory = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text="请先举例，不要直接上公式。",
        teacher_action=_teacher("先看一个台阶例子。"),
    )
    for round_number in range(2, 10):
        memory = commit_teaching_memory_turn(
            memory,
            round_number=round_number,
            learner_text=f"第 {round_number} 轮回答",
            teacher_action=_teacher("继续检查当前步骤。"),
        )
    projection = project_teaching_memory(memory)
    assert projection["history_version"] == 9
    assert projection["compaction_generation"] >= 1
    statements = [item["statement"] for item in projection["active_preferences"]]
    assert any("不要直接上公式" in item for item in statements)
    assert projection["narrative_inference_added"] is False


def test_procedural_sequence_is_not_misclassified_as_a_teaching_preference() -> None:
    memory = initialize_teaching_memory(GOAL, PROFILE)
    initial_preference_count = len(memory["preferences"])
    memory = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text="先计算 dp[i-1] 和 dp[i-2]，再更新 dp[i]。",
        teacher_action=_teacher("请说明这两个来源分别对应哪一步选择。"),
    )

    assert len(memory["preferences"]) == initial_preference_count
    assert all(
        "dp[i-1]" not in item["statement"] for item in memory["preferences"]
    )


def test_open_question_requires_learner_confirmation_before_resolution() -> None:
    learner_question = "递归为什么必须有终止条件？"
    memory = initialize_teaching_memory(GOAL, PROFILE)
    memory = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text=learner_question,
        teacher_action=_teacher(
            "因为调用链必须在某个状态停止。你能用最小例子说明吗？",
            answered_question=learner_question,
        ),
    )
    question = memory["open_questions"][-1]
    assert question["status"] == "addressed_pending_confirmation"
    memory = commit_teaching_memory_turn(
        memory,
        round_number=2,
        learner_text="我明白了，终止条件让递归不再继续展开。",
        teacher_action=_teacher("很好，再比较有无终止条件的差异。"),
    )
    assert memory["open_questions"][-1]["status"] == "resolved"
    assert project_teaching_memory(memory)["unresolved_questions"] == []


def test_reopen_signal_prevents_false_resolution() -> None:
    learner_question = "第二种方法为什么更省空间？"
    memory = initialize_teaching_memory(GOAL, PROFILE)
    memory = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text=learner_question,
        teacher_action=_teacher(
            "第二种只保存后续计算仍会用到的相邻状态。",
            answered_question=learner_question,
        ),
    )
    memory = commit_teaching_memory_turn(
        memory,
        round_number=2,
        learner_text="我还是不明白，刚才的问题还没解决。",
        teacher_action=_teacher("我们换成两个数组格子的具体例子。"),
    )
    assert memory["open_questions"][-1]["status"] == "open"


@pytest.mark.parametrize(
    "learner_question",
    [
        "递推关系分哪几部分",
        "递推关系分为几部分",
        "递推关系由什么组成",
        "递推关系包括哪些部分",
    ],
)
def test_unpunctuated_composition_question_remains_open_without_receipt(
    learner_question: str,
) -> None:
    memory = commit_teaching_memory_turn(
        initialize_teaching_memory(GOAL, PROFILE),
        round_number=1,
        learner_text=learner_question,
        teacher_action=_teacher("我们先看斐波那契数列。你能找出已知量吗？"),
    )

    assert memory["open_questions"][-1]["question"] == learner_question
    assert memory["open_questions"][-1]["status"] == "open"
    assert (
        project_teaching_memory(memory)["unresolved_questions"][-1]["question"]
        == learner_question
    )


@pytest.mark.parametrize(
    "learner_statement",
    [
        "递推关系分为三个部分",
        "递推关系由初始条件、递推规则和适用范围组成",
        "递推关系包括初始条件和递推规则",
    ],
)
def test_composition_declaration_is_not_recorded_as_question(
    learner_statement: str,
) -> None:
    memory = commit_teaching_memory_turn(
        initialize_teaching_memory(GOAL, PROFILE),
        round_number=1,
        learner_text=learner_statement,
        teacher_action=_teacher("很好，我们继续。"),
    )

    assert memory["open_questions"] == []


def test_teacher_follow_up_does_not_claim_an_open_question_was_answered() -> None:
    learner_question = "递推关系分哪几部分"
    memory = commit_teaching_memory_turn(
        initialize_teaching_memory(GOAL, PROFILE),
        round_number=1,
        learner_text=learner_question,
        teacher_action=_teacher(
            "先看斐波那契数列。你能指出已知量和重复结构吗？"
        ),
    )

    question = memory["open_questions"][-1]
    assert question["status"] == "open"
    assert question["addressed_round"] is None
    assert question["answer_evidence_refs"] == []


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("schema", "teaching_skill_miner.learner_question_answer_receipt.v1"),
        ("status", "answered"),
        ("question_sha256", "0" * 64),
        ("clarification_contract_sha256", "not-a-sha"),
        ("clarification_kind", "other"),
        ("grounding_mode", "model_general_knowledge_unverified"),
        ("grounding_refs", []),
        ("answer_surface_contract_validated", False),
        ("server_verified_semantic_truth", True),
        ("general_knowledge_is_grading_authority", True),
        ("conceptual_clarification_answered", False),
        ("practice_final_solution_provided", True),
        ("mastery_evidence", True),
    ],
)
def test_question_closure_rejects_invalid_answer_receipt(
    field: str, invalid_value: object
) -> None:
    learner_question = "递推关系由什么组成"
    teacher_action = _teacher(
        "递推关系包括初始条件、递推规则和适用范围。",
        answered_question=learner_question,
    )
    teacher_action["learner_question_answer"][field] = invalid_value

    memory = commit_teaching_memory_turn(
        initialize_teaching_memory(GOAL, PROFILE),
        round_number=1,
        learner_text=learner_question,
        teacher_action=teacher_action,
    )

    assert memory["open_questions"][-1]["status"] == "open"


def test_referent_options_and_teacher_commitments_are_retained() -> None:
    memory = initialize_teaching_memory(GOAL, PROFILE)
    memory = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text="两种方法有什么区别？",
        teacher_action=_teacher(
            "第一种保留完整表，第二种只保留相邻状态。接下来我会让你比较二者。"
        ),
    )
    projection = project_teaching_memory(memory)
    assert projection["active_referents"]
    assert projection["pending_teacher_commitments"]


@pytest.mark.parametrize(
    "message",
    [
        "请先只回答这一问，我会等你回答后再继续。",
        (
            "你能指出这个例子中的已知量、目标和重复结构吗？"
            "请先只回答这一问，我会等你回答后再继续。"
        ),
        "请先只回答当前一问，我会等待你作答后再继续。",
    ],
)
def test_wait_for_response_control_is_not_a_teacher_commitment(message: str) -> None:
    memory = initialize_teaching_memory(GOAL, PROFILE)
    memory = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text="继续",
        teacher_action=_teacher(message),
    )

    assert memory["commitments"] == []
    assert project_teaching_memory(memory)["pending_teacher_commitments"] == []


@pytest.mark.parametrize(
    "message",
    [
        (
            "接下来我会用两个数组格子比较。"
            "请先只回答这一问，我会等你回答后再继续。"
        ),
        "等你回答之后，我会用图示讲清状态转移。",
    ],
)
def test_real_teacher_commitment_survives_wait_control_filter(message: str) -> None:
    memory = initialize_teaching_memory(GOAL, PROFILE)
    memory = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text="继续",
        teacher_action=_teacher(message),
    )

    assert len(memory["commitments"]) == 1
    assert memory["commitments"][0]["statement"] == message
    assert (
        project_teaching_memory(memory)["pending_teacher_commitments"][0][
            "statement"
        ]
        == message
    )


def test_legacy_wait_commitment_checkpoint_replays_and_migrates_safely() -> None:
    message = "请先只回答这一问，我会等你回答后再继续。"
    legacy = initialize_teaching_memory(GOAL, PROFILE)
    legacy["commitments"].append(
        {
            "memory_id": f"commitment_r001_{canonical_sha256(message)[:12]}",
            "statement": message,
            "status": "pending",
            "created_round": 1,
            "action_id": "turn",
            "evidence_refs": ["session_history:r1:teacher_action"],
        }
    )
    legacy["history_version"] = 1
    legacy["last_observed_round"] = 1
    validate_teaching_memory(legacy)
    event = {
        "round": 1,
        "learner_text": "继续",
        "learner_response": "继续",
        "action": {
            "action_id": "turn",
            "teacher_action": {"message": message},
        },
        "teaching_memory_trace": {
            "history_version": 1,
            "compaction_generation": 0,
            "fixed_context_fingerprint": legacy["fixed_context_fingerprint"],
            "content_sha256": canonical_sha256(legacy),
            "source": "deterministic_evidence_linked_rollout_projection",
            "model_generated_summary": False,
        },
    }

    rebuilt = rebuild_teaching_memory_from_rollout(
        GOAL,
        PROFILE,
        [event],
        validate_checkpoint_traces=True,
    )

    assert rebuilt == legacy
    assert project_teaching_memory(rebuilt)["pending_teacher_commitments"] == []
    migrated = commit_teaching_memory_turn(
        rebuilt,
        round_number=2,
        learner_text="继续",
        teacher_action=_teacher("我们继续检查当前步骤。"),
    )
    assert migrated["commitments"] == []


def test_legacy_auto_addressed_question_checkpoint_still_replays() -> None:
    learner_question = "递推关系分哪几部分"
    teacher_message = (
        "先看斐波那契数列。你能指出已知量吗？"
        "请先只回答这一问，我会等你回答后再继续。"
    )
    legacy = initialize_teaching_memory(GOAL, PROFILE)
    legacy["open_questions"].append(
        {
            "memory_id": (
                f"question_r001_{canonical_sha256(learner_question)[:12]}"
            ),
            "question": learner_question,
            "status": "addressed_pending_confirmation",
            "first_raised_round": 1,
            "last_raised_round": 1,
            "addressed_round": 1,
            "resolved_round": None,
            "evidence_refs": ["session_history:r1:learner_response"],
            "answer_evidence_refs": ["session_history:r1:teacher_action"],
        }
    )
    legacy["history_version"] = 1
    legacy["last_observed_round"] = 1
    validate_teaching_memory(legacy)
    event = {
        "round": 1,
        "learner_text": learner_question,
        "learner_response": learner_question,
        "action": {
            "action_id": "turn",
            "teacher_action": {"message": teacher_message},
        },
        "teaching_memory_trace": {
            "history_version": 1,
            "compaction_generation": 0,
            "fixed_context_fingerprint": legacy["fixed_context_fingerprint"],
            "content_sha256": canonical_sha256(legacy),
            "source": "deterministic_evidence_linked_rollout_projection",
            "model_generated_summary": False,
        },
    }

    rebuilt = rebuild_teaching_memory_from_rollout(
        GOAL,
        PROFILE,
        [event],
        validate_checkpoint_traces=True,
    )

    assert rebuilt == legacy


def test_legacy_unpunctuated_question_checkpoint_without_open_item_replays() -> None:
    learner_question = "递推关系分哪几部分"
    teacher_message = "我们先看一个最小例子。"
    legacy = initialize_teaching_memory(GOAL, PROFILE)
    legacy["history_version"] = 1
    legacy["last_observed_round"] = 1
    validate_teaching_memory(legacy)
    event = {
        "round": 1,
        "learner_text": learner_question,
        "learner_response": learner_question,
        "action": {
            "action_id": "turn",
            "teacher_action": {"message": teacher_message},
        },
        "teaching_memory_trace": {
            "history_version": 1,
            "compaction_generation": 0,
            "fixed_context_fingerprint": legacy["fixed_context_fingerprint"],
            "content_sha256": canonical_sha256(legacy),
            "source": "deterministic_evidence_linked_rollout_projection",
            "model_generated_summary": False,
        },
    }

    rebuilt = rebuild_teaching_memory_from_rollout(
        GOAL,
        PROFILE,
        [event],
        validate_checkpoint_traces=True,
    )

    assert rebuilt == legacy


def test_memory_rejects_non_monotonic_commit_and_provenance_tampering() -> None:
    memory = initialize_teaching_memory(GOAL, PROFILE)
    memory = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text="回答",
        teacher_action=_teacher("继续。"),
    )
    with pytest.raises(TeachingMemoryError):
        commit_teaching_memory_turn(
            memory,
            round_number=1,
            learner_text="重复",
            teacher_action=_teacher("继续。"),
        )
    broken = deepcopy(memory)
    broken["preferences"][0]["evidence_refs"] = []
    with pytest.raises(TeachingMemoryError):
        validate_teaching_memory(broken)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("learner_level", "advanced"),
        (
            "initial_mastery",
            {
                "prerequisite": 0.9,
                "conceptual": 0.8,
                "procedural": 0.7,
                "transfer": 0.6,
            },
        ),
        ("known_misconceptions", []),
        ("conversation_history", []),
        ("background_history", ["已经完成过动态规划项目"]),
    ],
)
def test_fixed_context_fingerprint_binds_every_immutable_profile_field(
    field: str, replacement
) -> None:
    original = initialize_teaching_memory(GOAL, PROFILE)
    changed_profile = deepcopy(PROFILE)
    changed_profile[field] = replacement
    changed = initialize_teaching_memory(GOAL, changed_profile)

    assert (
        changed["fixed_context_fingerprint"]
        != original["fixed_context_fingerprint"]
    )


def test_fixed_context_fingerprint_excludes_only_adaptive_rollout_fields() -> None:
    original = initialize_teaching_memory(GOAL, PROFILE)
    adapted_profile = deepcopy(PROFILE)
    adapted_profile["adaptive_observations"] = [
        {"round": 1, "candidate": {"response_quality": "partial"}}
    ]
    adapted_profile["adaptive_summary"] = {"latest_round": 1}

    assert (
        initialize_teaching_memory(GOAL, adapted_profile)[
            "fixed_context_fingerprint"
        ]
        == original["fixed_context_fingerprint"]
    )


def test_canonical_rollout_rebuild_recovers_a_legitimate_history_checkpoint() -> None:
    memory = initialize_teaching_memory(GOAL, PROFILE)
    history = []
    turns = [
        (
            "请先举例，不要直接上公式。",
            {
                "action_id": "action_000",
                "teacher_action": {
                    "message": "第一种保留完整表，第二种保留相邻状态。"
                },
            },
        ),
        (
            "第二种为什么更省空间？",
            {
                "action_id": "action_001",
                "teacher_action": {
                    "message": "接下来我会用两个数组格子比较。"
                },
            },
        ),
    ]
    for round_number, (learner_text, action) in enumerate(turns, 1):
        event = {
            "round": round_number,
            "learner_text": learner_text,
            "learner_response": learner_text,
            "action": action,
        }
        memory = commit_teaching_memory_turn(
            memory,
            round_number=round_number,
            learner_text=learner_text,
            teacher_action={
                **deepcopy(action["teacher_action"]),
                "action_id": action["action_id"],
            },
        )
        event["teaching_memory_trace"] = {
            "history_version": memory["history_version"],
            "compaction_generation": memory["compaction_generation"],
            "fixed_context_fingerprint": memory["fixed_context_fingerprint"],
            "content_sha256": canonical_sha256(memory),
            "source": "deterministic_evidence_linked_rollout_projection",
            "model_generated_summary": False,
        }
        history.append(event)

    rebuilt = rebuild_teaching_memory_from_rollout(
        GOAL,
        PROFILE,
        history,
        validate_checkpoint_traces=True,
    )
    assert rebuilt == memory
    assert (
        validate_teaching_memory_checkpoint(
            memory,
            goal=GOAL,
            profile=PROFILE,
            history=history,
            expected_round=2,
        )
        == memory
    )


def test_canonical_checkpoint_rejects_forged_r999_evidence() -> None:
    memory = initialize_teaching_memory(GOAL, PROFILE)
    action = {
        "action_id": "action_000",
        "teacher_action": {"message": "请先说出一个状态。"},
    }
    memory = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text="请先举例。",
        teacher_action={**action["teacher_action"], "action_id": "action_000"},
    )
    event = {
        "round": 1,
        "learner_text": "请先举例。",
        "learner_response": "请先举例。",
        "action": action,
        "teaching_memory_trace": {
            "history_version": memory["history_version"],
            "compaction_generation": memory["compaction_generation"],
            "fixed_context_fingerprint": memory["fixed_context_fingerprint"],
            "content_sha256": canonical_sha256(memory),
            "source": "deterministic_evidence_linked_rollout_projection",
            "model_generated_summary": False,
        },
    }
    forged = deepcopy(memory)
    forged["preferences"][-1]["evidence_refs"] = [
        "session_history:r999:learner_response"
    ]

    with pytest.raises(TeachingMemoryError, match="canonical history replay"):
        validate_teaching_memory_checkpoint(
            forged,
            goal=GOAL,
            profile=PROFILE,
            history=[event],
            expected_round=1,
        )
