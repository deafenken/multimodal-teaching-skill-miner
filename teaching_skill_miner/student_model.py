"""Evidence-weighted learner state estimation for the Teaching Agent.

This module is intentionally small and deterministic.  It is not a claim that
the agent can infer a learner's true ability from one free-text response.  It
maintains a calibrated *working estimate* with explicit uncertainty and
evidence pointers so the runtime can decide when to teach, ask, or hand off.

The legacy ``student_state.knowledge_mastery`` values remain untouched.  The
new estimate is stored beside them and can be compared in receipts without
silently changing historical behaviour.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping


STUDENT_MODEL_SCHEMA = "teaching_skill_miner.student_state_estimate.v1"
MASTERY_DIMENSIONS = ("prerequisite", "conceptual", "procedural", "transfer")
_SIGNALS = {"not_observed", "correct", "partial", "misconception", "confused", "no_response"}
_ALIGNMENTS = {
    "not_applicable",
    "aligned",
    "partially_aligned",
    "related_but_not_answer",
    "contradicted",
    "ambiguous",
    "no_response",
}

# Soft labels: they are observations, not truth.  A misconception is evidence
# against mastery only when the server has grounded it in the current answer.
_TARGETS = {
    "correct": 0.95,
    "partial": 0.62,
    "misconception": 0.12,
    "confused": 0.28,
    "no_response": 0.30,
}
_ALIGNMENT_RELIABILITY = {
    "aligned": 1.0,
    "partially_aligned": 0.72,
    "contradicted": 1.0,
    "related_but_not_answer": 0.0,
    "ambiguous": 0.25,
    "no_response": 0.35,
    "not_applicable": 1.0,
}


class StudentModelError(ValueError):
    """Raised when an estimate violates its bounded public contract."""


def _probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudentModelError(f"{field} must be a number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise StudentModelError(f"{field} must be a finite number in [0, 1]")
    return result


def _bounded_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StudentModelError(f"{field} must be an integer >= {minimum}")
    return value


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, float(value)))


def _round(value: float) -> float:
    return round(_clamp(value), 4)


def _dimension_row(initial: float, *, prior_strength: float) -> dict[str, Any]:
    prior_strength = max(2.0, float(prior_strength))
    alpha = max(1e-6, initial * prior_strength)
    beta = max(1e-6, (1.0 - initial) * prior_strength)
    mean = alpha / (alpha + beta)
    variance = mean * (1.0 - mean) / (alpha + beta + 1.0)
    return {
        "p_mastery": _round(mean),
        "uncertainty": _round(min(1.0, math.sqrt(max(0.0, variance)) * 2.0)),
        "alpha": round(alpha, 6),
        "beta": round(beta, 6),
        "evidence_count": 0,
        "positive_evidence": 0,
        "negative_evidence": 0,
        "last_signal": "not_observed",
        "last_round": 0,
        "last_evidence_id": None,
        "source": "teacher_profile_prior",
    }


def initialize_student_model(
    initial_mastery: Mapping[str, Any] | None = None,
    *,
    prior_strength: float = 4.0,
) -> dict[str, Any]:
    """Create a bounded estimate from teacher-provided priors."""

    initial_mastery = initial_mastery or {}
    if not isinstance(initial_mastery, Mapping):
        raise StudentModelError("initial_mastery must be an object")
    dimensions = {
        dimension: _dimension_row(
            _probability(initial_mastery.get(dimension, 0.0), field=dimension),
            prior_strength=prior_strength,
        )
        for dimension in MASTERY_DIMENSIONS
    }
    model = {
        "schema": STUDENT_MODEL_SCHEMA,
        "version": 1,
        "prior_strength": round(float(prior_strength), 4),
        "dimensions": dimensions,
        "overall": {
            "p_mastery": _round(sum(row["p_mastery"] for row in dimensions.values()) / len(dimensions)),
            "uncertainty": _round(sum(row["uncertainty"] for row in dimensions.values()) / len(dimensions)),
            "evidence_count": 0,
        },
        "last_update": {
            "round": 0,
            "signal": "not_observed",
            "focus_dimension": None,
            "evidence_id": None,
            "source": "teacher_profile_prior",
            "update_applied": False,
            "needs_human_review": False,
        },
        "recommended_focus": {
            "dimension": "prerequisite",
            "reason": "等待首个有界观察",
            "confidence": 0.0,
            "evidence_refs": [],
        },
        "claim_boundary": {
            "is_ground_truth": False,
            "teacher_prior_overwritten": False,
            "free_text_accuracy_established": False,
            "uncertainty_is_calibrated_on_external_lockbox": False,
        },
    }
    validate_student_model(model)
    return model


def validate_student_model(model: Mapping[str, Any]) -> None:
    if not isinstance(model, Mapping) or model.get("schema") != STUDENT_MODEL_SCHEMA:
        raise StudentModelError(f"student model schema must be {STUDENT_MODEL_SCHEMA}")
    if model.get("version") != 1:
        raise StudentModelError("unsupported student model version")
    dimensions = model.get("dimensions")
    if not isinstance(dimensions, Mapping) or set(dimensions) != set(MASTERY_DIMENSIONS):
        raise StudentModelError("student model dimensions are incomplete")
    for dimension in MASTERY_DIMENSIONS:
        row = dimensions[dimension]
        if not isinstance(row, Mapping):
            raise StudentModelError(f"student model dimension {dimension} is invalid")
        _probability(row.get("p_mastery"), field=f"dimensions.{dimension}.p_mastery")
        _probability(row.get("uncertainty"), field=f"dimensions.{dimension}.uncertainty")
        _bounded_int(row.get("evidence_count"), field=f"dimensions.{dimension}.evidence_count")
        _bounded_int(row.get("positive_evidence"), field=f"dimensions.{dimension}.positive_evidence")
        _bounded_int(row.get("negative_evidence"), field=f"dimensions.{dimension}.negative_evidence")
        _bounded_int(row.get("last_round"), field=f"dimensions.{dimension}.last_round")
        if row.get("last_signal") not in _SIGNALS:
            raise StudentModelError(f"dimensions.{dimension}.last_signal is invalid")
    overall = model.get("overall")
    if not isinstance(overall, Mapping):
        raise StudentModelError("student model overall is missing")
    _probability(overall.get("p_mastery"), field="overall.p_mastery")
    _probability(overall.get("uncertainty"), field="overall.uncertainty")
    _bounded_int(overall.get("evidence_count"), field="overall.evidence_count")
    update = model.get("last_update")
    if not isinstance(update, Mapping) or update.get("signal") not in _SIGNALS:
        raise StudentModelError("student model last_update is invalid")
    _bounded_int(update.get("round"), field="last_update.round")
    if update.get("focus_dimension") is not None and update.get("focus_dimension") not in MASTERY_DIMENSIONS:
        raise StudentModelError("student model last_update.focus_dimension is invalid")
    if not isinstance(update.get("update_applied"), bool) or not isinstance(update.get("needs_human_review"), bool):
        raise StudentModelError("student model update flags are invalid")
    recommendation = model.get("recommended_focus")
    if not isinstance(recommendation, Mapping) or recommendation.get("dimension") not in MASTERY_DIMENSIONS:
        raise StudentModelError("student model recommended_focus is invalid")
    _probability(recommendation.get("confidence"), field="recommended_focus.confidence")
    refs = recommendation.get("evidence_refs")
    if not isinstance(refs, list) or len(refs) > 8 or any(not isinstance(item, str) or not item for item in refs):
        raise StudentModelError("student model recommendation evidence_refs are invalid")
    boundary = model.get("claim_boundary")
    if not isinstance(boundary, Mapping) or any(boundary.get(key) is not False for key in (
        "is_ground_truth", "teacher_prior_overwritten", "free_text_accuracy_established", "uncertainty_is_calibrated_on_external_lockbox"
    )):
        raise StudentModelError("student model claim boundary is invalid")


def _recompute(model: dict[str, Any]) -> None:
    rows = model["dimensions"]
    for row in rows.values():
        alpha = max(1e-6, float(row["alpha"]))
        beta = max(1e-6, float(row["beta"]))
        total = alpha + beta
        mean = alpha / total
        variance = mean * (1.0 - mean) / (total + 1.0)
        row["p_mastery"] = _round(mean)
        row["uncertainty"] = _round(min(1.0, math.sqrt(max(0.0, variance)) * 2.0))
    model["overall"] = {
        "p_mastery": _round(sum(row["p_mastery"] for row in rows.values()) / len(rows)),
        "uncertainty": _round(sum(row["uncertainty"] for row in rows.values()) / len(rows)),
        "evidence_count": sum(int(row["evidence_count"]) for row in rows.values()),
    }


def update_student_model(
    model: Mapping[str, Any],
    *,
    signal: str,
    confidence: float,
    focus_dimension: str,
    answer_alignment: str = "not_applicable",
    needs_human_review: bool = False,
    round_number: int = 0,
    evidence_id: str | None = None,
    source: str = "validated_turn_diagnosis",
) -> dict[str, Any]:
    """Apply one soft, evidence-weighted observation without storing raw text."""

    candidate = deepcopy(dict(model))
    validate_student_model(candidate)
    if signal not in _SIGNALS or signal == "not_observed":
        if signal != "not_observed":
            raise StudentModelError("signal is invalid")
        candidate["last_update"] = {
            "round": max(0, int(round_number)),
            "signal": "not_observed",
            "focus_dimension": focus_dimension if focus_dimension in MASTERY_DIMENSIONS else None,
            "evidence_id": str(evidence_id)[:120] if evidence_id else None,
            "source": str(source)[:120],
            "update_applied": False,
            "needs_human_review": bool(needs_human_review),
        }
        validate_student_model(candidate)
        return candidate
    if focus_dimension not in MASTERY_DIMENSIONS:
        raise StudentModelError("focus_dimension is invalid")
    confidence = _probability(confidence, field="confidence")
    if answer_alignment not in _ALIGNMENTS:
        raise StudentModelError("answer_alignment is invalid")
    if not isinstance(needs_human_review, bool):
        raise StudentModelError("needs_human_review must be a boolean")
    row = candidate["dimensions"][focus_dimension]
    reliability = confidence * _ALIGNMENT_RELIABILITY[answer_alignment]
    # Human review and ungrounded/ambiguous evidence affect audit metadata but
    # do not move the estimate.  This is the key fail-closed property.
    applied = reliability > 0.0 and not needs_human_review
    if applied:
        target = _TARGETS[signal]
        weight = max(0.0, min(1.0, reliability))
        row["alpha"] = round(float(row["alpha"]) + weight * target, 6)
        row["beta"] = round(float(row["beta"]) + weight * (1.0 - target), 6)
        row["evidence_count"] += 1
        if target >= 0.5:
            row["positive_evidence"] += 1
        else:
            row["negative_evidence"] += 1
        row["source"] = str(source)[:120]
    row["last_signal"] = signal
    row["last_round"] = max(0, int(round_number))
    row["last_evidence_id"] = str(evidence_id)[:120] if evidence_id else None
    candidate["last_update"] = {
        "round": max(0, int(round_number)),
        "signal": signal,
        "focus_dimension": focus_dimension,
        "evidence_id": str(evidence_id)[:120] if evidence_id else None,
        "source": str(source)[:120],
        "update_applied": applied,
        "needs_human_review": needs_human_review,
    }
    _recompute(candidate)
    validate_student_model(candidate)
    return candidate


def recommend_focus(
    model: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
    *,
    active_misconception: bool = False,
) -> dict[str, Any]:
    """Choose the next dimension using deficit plus uncertainty."""

    validate_student_model(model)
    thresholds = thresholds or {}
    if active_misconception:
        dimension = "conceptual"
        reason = "存在活跃误解，先处理概念边界"
    else:
        rows = model["dimensions"]
        def score(dimension: str) -> tuple[float, float, int]:
            threshold = _clamp(float(thresholds.get(dimension, 0.75)))
            row = rows[dimension]
            deficit = max(0.0, threshold - float(row["p_mastery"]))
            # Uncertainty breaks ties in favour of asking for evidence.
            return (deficit + 0.35 * float(row["uncertainty"]), deficit, -MASTERY_DIMENSIONS.index(dimension))
        dimension = max(MASTERY_DIMENSIONS, key=score)
        row = rows[dimension]
        reason = "按达标缺口与不确定性排序，优先收集最有价值的证据"
    row = model["dimensions"][dimension]
    confidence = _clamp(1.0 - float(row["uncertainty"]))
    refs = []
    if row.get("last_evidence_id"):
        refs.append(str(row["last_evidence_id"]))
    return {
        "dimension": dimension,
        "reason": reason,
        "confidence": _round(confidence),
        "evidence_refs": refs[:8],
    }


def project_student_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Return UI/context-safe fields while retaining provenance and limits."""

    validate_student_model(model)
    result = deepcopy(dict(model))
    for row in result["dimensions"].values():
        row.pop("alpha", None)
        row.pop("beta", None)
    result["source"] = "deterministic_evidence_weighted_estimator"
    return result
