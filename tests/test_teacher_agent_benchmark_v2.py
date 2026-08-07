from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from teaching_skill_miner.teacher_agent_benchmark_v2 import (
    BENCHMARK_VERSION,
    PREDICTIONS_SCHEMA,
    TeachingAgentBenchmarkV2Error,
    _fingerprint,
    score_benchmark_v2,
    validate_benchmark_gold,
    validate_benchmark_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


@pytest.fixture
def artifacts() -> tuple[dict, dict, dict]:
    inputs = _load("data/teacher_agent_benchmark_v2_development.json")
    gold = _load("data/teacher_agent_benchmark_v2_development_gold.json")
    library = _load("data/teacher_agent_skill_library_v2.json")
    return inputs, gold, library


def _perfect_predictions(inputs: dict, gold: dict) -> dict:
    gold_by_id = {row["case_id"]: row for row in gold["cases"]}
    cases = []
    for case_index, case in enumerate(inputs["cases"]):
        turns = []
        for turn_gold in gold_by_id[case["case_id"]]["turns"]:
            message_terms = [
                group[0]
                for group in turn_gold["recall_term_groups"]
                if group
            ]
            message = "请先说明你的理由。"
            if message_terms:
                message = "我们按" + "、".join(message_terms) + "继续，请先说下一步。"
            turns.append(
                {
                    "turn_id": turn_gold["turn_id"],
                    "teacher_message": message,
                    "primary_skill_id": turn_gold["allowed_primary_skill_ids"][0],
                    "skill_switched": bool(turn_gold["expected_switch"]),
                    "memory_status": turn_gold["expected_memory_status"],
                    "memory_evidence_turn_ids": (
                        [case["turns"][0]["turn_id"]]
                        if turn_gold["expected_memory_status"]
                        == "resolved_evidence_linked"
                        else []
                    ),
                    "active_misconception_tags": list(
                        turn_gold["expected_active_misconception_tags"]
                    ),
                    "resolved_misconception_tags": list(
                        turn_gold["expected_resolved_misconception_tags"]
                    ),
                    "resolution_evidence_turn_ids": (
                        [turn_gold["turn_id"]]
                        if turn_gold["required_resolution_evidence"]
                        else []
                    ),
                    "prompt_injection_blocked": bool(
                        turn_gold["prompt_injection_blocked"]
                    ),
                    "terminal": bool(turn_gold["should_stop"]),
                    "deterministic_fallback": False,
                }
            )
        cases.append(
            {
                "case_id": case["case_id"],
                "session_instance_id": f"session-{case_index}-{case['case_id']}",
                "turns": turns,
                "final_status": "active",
            }
        )
    return {
        "schema": PREDICTIONS_SCHEMA,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_id": inputs["benchmark_id"],
        "input_fingerprint": _fingerprint(inputs),
        "runtime": {
            "provider": "scripted_test_double",
            "model": "none",
            "agent_loop_enabled": True,
        },
        "cases": cases,
    }


def test_development_fixture_is_input_gold_separated_and_valid(artifacts) -> None:
    inputs, gold, library = artifacts

    validate_benchmark_inputs(inputs, library)
    validate_benchmark_gold(gold, inputs, library)

    assert inputs["split"] == "development"
    assert inputs["claim_boundary"]["held_out_after_prompt_development"] is False
    assert inputs["claim_boundary"]["real_learning_effect_established"] is False
    assert len(inputs["cases"]) == 6
    assert sum(len(case["turns"]) for case in inputs["cases"]) == 20
    assert "allowed_primary_skill_ids" not in json.dumps(
        inputs, ensure_ascii=False
    )


def test_score_exposes_product_dimensions_without_claiming_accuracy(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)

    report = score_benchmark_v2(inputs, gold, predictions, library)

    assert report["metrics"]["long_horizon_memory"]["recall_group_coverage"] == 1.0
    assert report["metrics"]["misconception_resolution"][
        "resolution_exact_rate"
    ] == 1.0
    assert report["metrics"]["skill_switching"]["allowed_skill_hit_rate"] == 1.0
    assert report["metrics"]["skill_switching"]["switch_f1"] == 1.0
    assert report["metrics"]["prompt_injection"]["direct_answer_leak_rate"] == 0.0
    assert report["metrics"]["cross_session_isolation"]["leakage_rate"] == 0.0
    assert report["metrics"]["learning_outcome"]["record_count"] == 3
    assert report["metrics"]["learning_outcome"]["mean_absolute_gain"] > 0
    assert report["metrics"]["learning_outcome"][
        "mean_delayed_retention_ratio"
    ] is not None
    assert report["claim_boundary"]["metrics_are_not_accuracy"] is True
    assert report["claim_boundary"]["real_learning_effect_established"] is False
    assert "请先说明" not in json.dumps(report, ensure_ascii=False)


def test_cross_session_leak_and_reused_instance_are_detected(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    by_id = {row["case_id"]: row for row in predictions["cases"]}
    by_id["cross_session_beta"]["session_instance_id"] = by_id[
        "cross_session_alpha"
    ]["session_instance_id"]
    by_id["cross_session_beta"]["turns"][0][
        "teacher_message"
    ] = "我记得 ALPHA_MEMORY_SENTINEL。"

    report = score_benchmark_v2(inputs, gold, predictions, library)

    isolation = report["metrics"]["cross_session_isolation"]
    assert isolation["unique_session_instance_rate"] == 0.0
    assert isolation["leakage_rate"] > 0.0


def test_input_artifact_rejects_embedded_gold(artifacts) -> None:
    inputs, _gold, library = artifacts
    tainted = deepcopy(inputs)
    tainted["cases"][0]["gold"] = {"expected_switch": True}

    with pytest.raises(TeachingAgentBenchmarkV2Error, match="gold-only"):
        validate_benchmark_inputs(tainted, library)


def test_failed_injection_block_is_counted_in_denominator(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    injection_case = next(
        row
        for row in predictions["cases"]
        if row["case_id"] == "prompt_injection_resistance"
    )
    injection_case["turns"][0]["prompt_injection_blocked"] = False

    report = score_benchmark_v2(inputs, gold, predictions, library)

    metric = report["metrics"]["prompt_injection"]
    assert metric["blocked_case_rate"] < 1.0
    assert metric["blocked_case_rate"] == 0.5


def test_missed_misconception_resolution_is_counted_in_denominator(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    misconception_case = next(
        row for row in predictions["cases"] if row["case_id"] == "misconception_resolution"
    )
    misconception_case["turns"][0]["active_misconception_tags"] = []

    report = score_benchmark_v2(inputs, gold, predictions, library)

    metric = report["metrics"]["misconception_resolution"]
    assert metric["resolution_exact_rate"] < 1.0
    assert metric["resolution_exact_rate"] == 0.666667
