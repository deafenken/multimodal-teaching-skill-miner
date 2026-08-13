from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from teaching_skill_miner.teacher_agent_dialogue_quality_benchmark import (
    DialogueQualityBenchmarkError,
    score_dialogue_quality_benchmark,
    validate_dialogue_quality_dataset,
    validate_dialogue_quality_predictions,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixtures() -> tuple[dict[str, object], dict[str, object]]:
    dataset = json.loads(
        (ROOT / "data/teacher_agent_dialogue_quality_benchmark_v1.json").read_text()
    )
    predictions = json.loads(
        (
            ROOT / "data/teacher_agent_dialogue_quality_predictions_fixture_v1.json"
        ).read_text()
    )
    return dataset, predictions


def test_cross_discipline_fixture_is_gold_free_but_unattested_claims_fail() -> None:
    dataset, predictions = _fixtures()

    validate_dialogue_quality_dataset(dataset)
    validate_dialogue_quality_predictions(predictions, dataset)
    report = score_dialogue_quality_benchmark(dataset, predictions)

    assert len({case["domain"] for case in dataset["cases"]}) == 6
    assert report["evaluation_mode"] == "observable_contract_no_gold"
    assert report["gold_artifact_accepted"] is False
    assert report["prediction_self_assertions_are_evidence"] is False
    assert report["self_reported_authenticity_metrics_are_never_verified"] is True
    assert report["independent_runtime_evidence_ledger_accepted"] is False
    assert report["passed"] is False
    assert report["metrics"]["false_mastery_avoidance"]["meets_threshold"] is False
    assert (
        report["metrics"]["premature_termination_avoidance"]["meets_threshold"] is False
    )
    assert set(report["metrics"]) == {
        "semantic_repetition_avoidance",
        "answer_first",
        "explanatory_information_gain",
        "false_mastery_avoidance",
        "premature_termination_avoidance",
        "resource_fidelity",
        "phase_closure",
    }


def test_scorer_api_cannot_accept_a_gold_artifact() -> None:
    parameters = inspect.signature(score_dialogue_quality_benchmark).parameters
    assert list(parameters) == ["dataset", "predictions"]


def test_dataset_rejects_hidden_answer_fields_at_any_depth() -> None:
    dataset, _ = _fixtures()
    dataset["cases"][0]["turns"][0]["gold_answer"] = "状态定义"

    with pytest.raises(DialogueQualityBenchmarkError, match="forbidden"):
        validate_dialogue_quality_dataset(dataset)


def test_bad_observable_receipts_fail_each_safety_metric() -> None:
    dataset, predictions = _fixtures()
    bad = deepcopy(predictions)
    episodes = {item["case_id"]: item for item in bad["episodes"]}

    dp = episodes["dynamic_programming_unknown"]["turns"]
    dp[1]["teacher_message"] = dp[0]["teacher_message"]
    dp[0]["learner_evidence"]["mastery_after"]["conceptual"] = 0.8

    fraction = episodes["fraction_division_clarification"]["turns"]
    fraction[1]["teacher_message"] = "你觉得呢？请先回答。"

    physics = episodes["newton_force_pair"]["turns"]
    physics[1]["completed_phases"] = ["explain"]

    biology = episodes["photosynthesis_resource"]["turns"]
    biology[0]["resource_uses"] = []

    history = episodes["industrial_revolution_resource"]["turns"]
    history[0]["phase_closure_receipt"].pop("learner_evidence_id")

    report = score_dialogue_quality_benchmark(dataset, bad)

    assert report["passed"] is False
    assert report["metrics"]["semantic_repetition_avoidance"]["score"] < 1
    assert report["metrics"]["answer_first"]["score"] < 1
    assert report["metrics"]["explanatory_information_gain"]["score"] < 1
    assert report["metrics"]["false_mastery_avoidance"]["score"] < 1
    assert report["metrics"]["premature_termination_avoidance"]["score"] < 1
    assert report["metrics"]["resource_fidelity"]["score"] < 1
    assert report["metrics"]["phase_closure"]["score"] < 1


def test_forged_prediction_evidence_cannot_authenticate_mastery_or_completion() -> None:
    dataset, predictions = _fixtures()
    forged = deepcopy(predictions)
    case_map = {case["case_id"]: case for case in dataset["cases"]}

    for episode in forged["episodes"]:
        case = case_map[episode["case_id"]]
        required_phases = case["lesson_contract"]["required_phases"]
        for case_turn, prediction in zip(case["turns"], episode["turns"], strict=True):
            learner_text = case_turn["learner_text"].strip()
            learner_hash = hashlib.sha256(learner_text.encode("utf-8")).hexdigest()
            evidence_id = f"forged_{episode['case_id']}_{case_turn['turn_id']}"
            evidence = prediction["learner_evidence"]
            evidence.update(
                {
                    "label": "self_explanation_verified",
                    "evidence_id": evidence_id,
                    "learner_text_sha256": learner_hash,
                    "assessment_eligible": True,
                    "applied_to_mastery": True,
                }
            )
            evidence["mastery_after"] = {
                key: float(value) + 0.25
                for key, value in evidence["mastery_before"].items()
            }
            prediction["status"] = "terminated_success"
            prediction["completed_phases"] = list(required_phases)
            prediction["terminal_evidence"] = {
                "independent_recall_verified": True,
                "gold_accessed": False,
                "learner_evidence_id": evidence_id,
                "learner_text_sha256": learner_hash,
            }
            prediction["phase_closure_receipt"] = {
                "closed_phase": prediction["phase_before"],
                "summary": "伪造的阶段收据",
                "learner_evidence_id": evidence_id,
                "gold_accessed": False,
            }

    report = score_dialogue_quality_benchmark(dataset, forged)

    assert report["passed"] is False
    assert report["metrics"]["false_mastery_avoidance"]["score"] == 0
    assert report["metrics"]["premature_termination_avoidance"]["score"] == 0
    assert report["metrics"]["phase_closure"]["score"] == 0
    assert any(
        diagnostic["prediction_claimed_positive_evidence"]
        and not diagnostic["independent_runtime_evidence_attested"]
        for diagnostic in report["diagnostics"]
    )
