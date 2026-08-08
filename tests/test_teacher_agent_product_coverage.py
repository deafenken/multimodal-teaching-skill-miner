"""Executable coverage map for the task-two Teaching Agent product goals.

The two bundled benchmark fixtures serve different purposes: product benchmark
v2 keeps input/gold separation and outcome plumbing small and auditable, while
the multi-turn fixture exercises profile replacement and terminal recovery.
These tests make that division explicit so a future fixture edit cannot quietly
remove one of the requirements from the reproducible evaluation surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from teaching_skill_miner.teacher_agent_benchmark_v2 import (
    validate_benchmark_gold,
    validate_benchmark_inputs,
)
from teaching_skill_miner.teacher_agent_multiturn_benchmark import (
    build_blind_episode_payload,
    validate_multiturn_benchmark,
)
from teaching_skill_miner.teacher_agent_outcomes import evaluate_learning_observation


ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _keys(value: Any) -> list[str]:
    """Collect object keys recursively for a blind-payload gold leak check."""

    result: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.append(str(key))
            result.extend(_keys(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_keys(item))
    return result


def test_task_two_product_goals_have_executable_fixture_coverage() -> None:
    """Long memory, routing, termination and learning interfaces stay covered."""

    inputs = _load("data/teacher_agent_benchmark_v2_development.json")
    gold = _load("data/teacher_agent_benchmark_v2_development_gold.json")
    library = _load("data/teacher_agent_skill_library_v2.json")
    validate_benchmark_inputs(inputs, library)
    validate_benchmark_gold(gold, inputs, library)

    categories = {str(case["category"]) for case in inputs["cases"]}
    assert {
        "long_horizon_memory",
        "misconception_resolution",
        "skill_switching",
        "prompt_injection",
        "cross_session_isolation",
    } <= categories

    gold_by_case = {str(case["case_id"]): case for case in gold["cases"]}
    switch_turns = [
        turn
        for case in gold_by_case.values()
        for turn in case["turns"]
        if turn.get("expected_switch") is True
    ]
    assert switch_turns, "v2 must retain at least one dynamic Skill-switch expectation"
    assert all(turn["allowed_primary_skill_ids"] for turn in switch_turns)

    # Outcome observations are a measurement interface, not a causal claim.
    observations = gold["outcome_observations"]
    assert observations
    for observation in observations:
        scored = evaluate_learning_observation(observation)
        assert scored["claim_boundary"][
            "real_learner_effectiveness_established"
        ] is False
        assert scored["metrics"]["posttest_improved"] is True
        assert scored["scores"]["transfer_test"] is not None
        assert scored["scores"]["delayed_test"] is not None


def test_multiturn_fixture_covers_profile_switch_and_terminal_recovery() -> None:
    """Profile replacement and stop/no-op behavior remain in the blind runner."""

    dataset = _load("data/teacher_agent_multiturn_benchmark_v1.json")
    library = _load("data/teacher_agent_skill_library_v2.json")
    validate_multiturn_benchmark(dataset, library)

    replacement_episodes = [
        episode
        for episode in dataset["episodes"]
        if any(turn.get("operation") == "replace_profile" for turn in episode["turns"])
    ]
    assert replacement_episodes, "profile-isolation coverage must include replace_profile"
    profile_episode = replacement_episodes[0]
    blind = build_blind_episode_payload(dataset, profile_episode)
    operations = blind["operations"]
    replacements = [
        operation
        for operation in operations
        if operation["operation"] == "replace_profile"
    ]
    assert len(replacements) == 1
    assert replacements[0]["student_profile"]["profile_ref"] != blind[
        "student_profile"
    ]["profile_ref"]
    assert "gold" not in _keys(blind)
    assert not {
        "acceptable_signals",
        "allowed_primary_skill_ids",
        "expected_switch",
        "should_stop",
    } & set(_keys(blind))

    terminal_turns = [
        turn
        for episode in dataset["episodes"]
        for turn in episode["turns"]
        if turn.get("operation") == "learner_turn"
        and turn.get("gold", {}).get("should_stop") is True
    ]
    assert terminal_turns, "termination/recovery coverage must include a stop expectation"
    assert any(
        episode["category"] == "termination_and_recovery"
        for episode in dataset["episodes"]
    )
