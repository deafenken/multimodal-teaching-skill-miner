"""Score paired pre/post, transfer, and optional delayed teaching assessments.

This module implements the measurement interface required by task two. It
does not turn a demo record into causal evidence: every report preserves the
input provenance and explicitly separates an observed score change from an
established learning effect.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from .teacher_agent import canonical_sha256


INPUT_SCHEMA = "teaching_skill_miner.teacher_agent_learning_observation.v1"
REPORT_SCHEMA = "teaching_skill_miner.teacher_agent_learning_report.v1"


class LearningOutcomeError(ValueError):
    """Raised when an outcome observation cannot be scored safely."""


def _score_block(value: Any, *, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise LearningOutcomeError(f"{name} must be an object")
    earned = value.get("earned")
    possible = value.get("possible")
    for field, number in (("earned", earned), ("possible", possible)):
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise LearningOutcomeError(f"{name}.{field} must be a finite number")
        if not math.isfinite(float(number)):
            raise LearningOutcomeError(f"{name}.{field} must be a finite number")
    earned_value = float(earned)
    possible_value = float(possible)
    if possible_value <= 0 or not 0 <= earned_value <= possible_value:
        raise LearningOutcomeError(f"{name} score is outside [0, possible]")
    return {
        "earned": earned_value,
        "possible": possible_value,
        "proportion": round(earned_value / possible_value, 6),
    }


def evaluate_learning_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return transparent paired-score diagnostics for one teaching session."""

    if not isinstance(observation, Mapping) or observation.get("schema") != INPUT_SCHEMA:
        raise LearningOutcomeError(f"observation schema must be {INPUT_SCHEMA}")
    case_id = str(observation.get("case_id", "")).strip()
    if not case_id:
        raise LearningOutcomeError("case_id is required")
    provenance = str(observation.get("provenance", "")).strip()
    if provenance not in {
        "author_constructed_demo_not_real",
        "teacher_provided_test_record",
        "authorized_real_learner_observation",
    }:
        raise LearningOutcomeError("provenance is unsupported")
    pretest = _score_block(observation.get("pretest"), name="pretest")
    posttest = _score_block(observation.get("posttest"), name="posttest")
    transfer = (
        _score_block(observation["transfer_test"], name="transfer_test")
        if observation.get("transfer_test") is not None
        else None
    )
    delayed = (
        _score_block(observation["delayed_test"], name="delayed_test")
        if observation.get("delayed_test") is not None
        else None
    )
    absolute_gain = round(posttest["proportion"] - pretest["proportion"], 6)
    available_gain = 1.0 - pretest["proportion"]
    normalized_gain = (
        round(absolute_gain / available_gain, 6)
        if available_gain > 0 and absolute_gain >= 0
        else None
    )
    delayed_retention = (
        round(delayed["proportion"] / posttest["proportion"], 6)
        if delayed is not None and posttest["proportion"] > 0
        else None
    )
    report = {
        "schema": REPORT_SCHEMA,
        "artifact_kind": "paired_teaching_assessment_score_report",
        "case_id": case_id,
        "provenance": provenance,
        "scores": {
            "pretest": pretest,
            "posttest": posttest,
            "transfer_test": transfer,
            "delayed_test": delayed,
        },
        "metrics": {
            "absolute_gain": absolute_gain,
            "normalized_gain": normalized_gain,
            "posttest_improved": absolute_gain > 0,
            "transfer_proportion": transfer["proportion"] if transfer else None,
            "delayed_retention_ratio": delayed_retention,
        },
        "claim_boundary": {
            "scores_are_user_or_fixture_supplied": True,
            "grading_correctness_independently_validated": False,
            "single_record_establishes_causal_learning_effect": False,
            "real_learner_effectiveness_established": False,
            "record_declares_real_learner_data": provenance
            == "authorized_real_learner_observation",
        },
        "input_fingerprint": canonical_sha256(deepcopy(dict(observation))),
    }
    report["content_sha256"] = canonical_sha256(report)
    return report
