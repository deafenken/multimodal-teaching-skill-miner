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


def _teacher(message: str) -> dict:
    return {"action_id": "turn", "message": message}


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
    memory = initialize_teaching_memory(GOAL, PROFILE)
    memory = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text="递归为什么必须有终止条件？",
        teacher_action=_teacher("因为调用链必须在某个状态停止。你能用最小例子说明吗？"),
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
    memory = initialize_teaching_memory(GOAL, PROFILE)
    memory = commit_teaching_memory_turn(
        memory,
        round_number=1,
        learner_text="第二种方法为什么更省空间？",
        teacher_action=_teacher("先比较两种方法保存的状态数量。"),
    )
    memory = commit_teaching_memory_turn(
        memory,
        round_number=2,
        learner_text="我还是不明白，刚才的问题还没解决。",
        teacher_action=_teacher("我们换成两个数组格子的具体例子。"),
    )
    assert memory["open_questions"][-1]["status"] == "addressed_pending_confirmation"


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
