"""Per-knowledge-component learner-state estimation.

The estimator is deliberately conservative.  It updates exactly one stable
knowledge component only when a caller supplies assessment-eligible,
authoritative, rubric-bound evidence.  Free text, lesson navigation, model
confidence, and exposure are not evidence by themselves.

The model's four-dimensional compatibility view is recomputed from per-KC
facet states.  Deterministic legacy sessions may retain their historical
``student_state.knowledge_mastery`` decision state.  The live runtime must
explicitly opt into the KC model as its success authority and synchronise the
legacy projection; the compatibility receipt makes that boundary auditable.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence
import unicodedata

from .teacher_agent_authority import (
    AUTHENTICATED_TEACHER_ACTOR,
    AUTHORITY_ASSURANCE,
    TeacherAuthorityError,
    validate_teacher_authority_revalidation_receipt,
)


LEGACY_STUDENT_MODEL_SCHEMA = "teaching_skill_miner.student_state_estimate.v1"
STUDENT_MODEL_SCHEMA = "teaching_skill_miner.student_state_estimate.v2"
STUDENT_MODEL_ADJUDICATION_RESULT_SCHEMA = (
    "teaching_skill_miner.student_model_adjudication_result.v1"
)
STUDENT_MODEL_ADJUDICATION_REVISION_SCHEMA = (
    "teaching_skill_miner.student_model_adjudication_revision.v1"
)
STUDENT_MODEL_READINESS_SCHEMA = (
    "teaching_skill_miner.student_model_mastery_readiness.v1"
)
MASTERY_DIMENSIONS = ("prerequisite", "conceptual", "procedural", "transfer")
_SIGNALS = {
    "not_observed",
    "correct",
    "partial",
    "misconception",
    "confused",
    "no_response",
}
_UPDATABLE_SIGNALS = {"correct", "partial", "misconception"}
_ALIGNMENTS = {
    "not_applicable",
    "aligned",
    "partially_aligned",
    "related_but_not_answer",
    "contradicted",
    "ambiguous",
    "no_response",
}
_SAFE_KC_ID = re.compile(r"^kc_[a-z0-9][a-z0-9_-]{2,80}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_MAX_EVIDENCE_PER_COMPONENT = 64
_MAX_ADJUDICATION_REVISIONS = 64
_TIME_BASES = {"wall_clock_utc", "session_logical"}
_ADJUDICATION_INSTRUCTION_SCHEMA = (
    "teaching_skill_miner.student_model_adjudication_instruction.v1"
)
_ADJUDICATION_OPERATIONS = {
    "replay_original_assessment",
    "supersede_and_replay",
    "do_not_replay",
    "cancel_adjudication",
}


class StudentModelError(ValueError):
    """Raised when a learner model violates its bounded public contract."""


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


def _text(value: Any, *, field: str, maximum: int = 160) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise StudentModelError(
            f"{field} must be a non-empty string <= {maximum} chars"
        )
    return result


def _canonical_label(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def stable_knowledge_component_id(label: str) -> str:
    """Return a stable content-derived ID; list position is never an input."""

    safe_label = _text(label, field="knowledge component label")
    canonical = _canonical_label(safe_label)
    if not canonical:
        raise StudentModelError("knowledge component label must not normalize to empty")
    digest = sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"kc_{digest}"


def knowledge_component_definitions(goal: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build stable KC definitions from the persisted teacher goal.

    Explicit IDs are accepted when a future importer supplies them.  Current
    string-only goals receive a normalized-label hash, so reordering a syllabus
    or inserting a neighbouring component does not change any ID.
    """

    if not isinstance(goal, Mapping):
        raise StudentModelError("goal must be an object")
    raw_components = goal.get("knowledge_components")
    if raw_components is None:
        raw_components = [goal.get("concept")]
    if isinstance(raw_components, (str, bytes)) or not isinstance(
        raw_components, Sequence
    ):
        raise StudentModelError("goal.knowledge_components must be an array")
    if not 1 <= len(raw_components) <= 24:
        raise StudentModelError("goal.knowledge_components must contain 1 to 24 items")

    syllabus_ref = goal.get("syllabus_ref")
    spec = goal.get("knowledge_spec")
    spec_authoritative = bool(
        isinstance(spec, Mapping)
        and isinstance(spec.get("claim_boundary"), Mapping)
        and spec["claim_boundary"].get("authoritative_for_runtime_grading") is True
    )
    if isinstance(syllabus_ref, Mapping):
        source = "syllabus_lesson_component"
        scope = ":".join(
            str(syllabus_ref.get(key, ""))
            for key in ("syllabus_id", "module_id", "lesson_id")
        )
    elif isinstance(spec, Mapping) and spec.get("status") != "not_provided":
        source = "teacher_knowledge_spec_component"
        scope = str(spec.get("schema", "knowledge_spec"))
    else:
        source = "teacher_goal_component"
        scope = str(goal.get("concept", "goal"))

    result: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    for index, raw in enumerate(raw_components):
        if isinstance(raw, Mapping):
            label = _text(
                raw.get("label", raw.get("name")),
                field=f"goal.knowledge_components[{index}].label",
            )
            explicit = raw.get("kc_id", raw.get("component_id"))
            kc_id = str(explicit or stable_knowledge_component_id(label)).strip()
            if not _SAFE_KC_ID.fullmatch(kc_id):
                raise StudentModelError(
                    f"goal.knowledge_components[{index}].kc_id is invalid"
                )
        else:
            label = _text(raw, field=f"goal.knowledge_components[{index}]")
            kc_id = stable_knowledge_component_id(label)
        owner = seen_ids.get(kc_id)
        if owner is not None:
            raise StudentModelError(
                "knowledge component IDs must be unique after normalization: "
                f"{owner!r} and {label!r}"
            )
        seen_ids[kc_id] = label
        result.append(
            {
                "kc_id": kc_id,
                "label": label,
                "source": source,
                "source_ref": f"{scope}#{kc_id}"[:300],
                "teacher_grading_authority_available": spec_authoritative,
            }
        )
    return result


def _facet_row(initial: float, *, prior_strength: float) -> dict[str, Any]:
    return {
        "p_mastery": _round(initial),
        "prior_p_mastery": _round(initial),
        "uncertainty": _round(1.0 / math.sqrt(max(1.0, prior_strength / 2.0))),
        "evidence_count": 0,
        "positive_evidence": 0,
        "negative_evidence": 0,
        "last_signal": "not_observed",
        "last_round": 0,
        "last_evidence_id": None,
        "last_observed_at": None,
        "last_time_basis": None,
    }


def _component_row(
    definition: Mapping[str, Any],
    initial_mastery: Mapping[str, Any],
    *,
    prior_strength: float,
) -> dict[str, Any]:
    facets = {
        dimension: _facet_row(
            _probability(initial_mastery.get(dimension, 0.0), field=dimension),
            prior_strength=prior_strength,
        )
        for dimension in MASTERY_DIMENSIONS
    }
    return {
        "kc_id": str(definition["kc_id"]),
        "label": str(definition["label"]),
        "source": str(definition["source"]),
        "source_ref": str(definition["source_ref"]),
        "teacher_grading_authority_available": bool(
            definition.get("teacher_grading_authority_available", False)
        ),
        "p_mastery": _round(
            sum(row["p_mastery"] for row in facets.values()) / len(facets)
        ),
        "uncertainty": _round(
            sum(row["uncertainty"] for row in facets.values()) / len(facets)
        ),
        "evidence_count": 0,
        "positive_evidence": 0,
        "negative_evidence": 0,
        "last_signal": "not_observed",
        "last_round": 0,
        "last_observed_at": None,
        "last_time_basis": None,
        "last_evidence_id": None,
        "bkt": {
            "p_learn": 0.08,
            "p_slip": 0.10,
            "p_guess": 0.20,
            "forgetting_rate_per_day": 0.015,
        },
        "calibration": {
            "status": "descriptive_only_external_calibration_required",
            "sample_count": 0,
            "brier_score": None,
            "brier_sum": 0.0,
        },
        "dimensions": facets,
        "evidence_ledger": [],
        "evidence_ledger_state": {
            "capacity": _MAX_EVIDENCE_PER_COMPONENT,
            "total_recorded_evidence": 0,
            "retained_evidence": 0,
            "active_evidence": 0,
            "superseded_evidence": 0,
            "complete_from_initial_prior": True,
        },
    }


def _legacy_dimension_projection(
    components: Mapping[str, Mapping[str, Any]],
    *,
    prior_strength: float,
) -> dict[str, dict[str, Any]]:
    projection: dict[str, dict[str, Any]] = {}
    for dimension in MASTERY_DIMENSIONS:
        rows = [row["dimensions"][dimension] for row in components.values()]
        mean = sum(float(row["p_mastery"]) for row in rows) / len(rows)
        uncertainty = sum(float(row["uncertainty"]) for row in rows) / len(rows)
        evidence_count = sum(int(row["evidence_count"]) for row in rows)
        effective_strength = max(2.0, float(prior_strength) + evidence_count)
        projection[dimension] = {
            "p_mastery": _round(mean),
            "uncertainty": _round(uncertainty),
            # Retained only for persisted v1-compatible audit/read clients.
            "alpha": round(max(1e-6, mean * effective_strength), 6),
            "beta": round(max(1e-6, (1.0 - mean) * effective_strength), 6),
            "evidence_count": evidence_count,
            "positive_evidence": sum(int(row["positive_evidence"]) for row in rows),
            "negative_evidence": sum(int(row["negative_evidence"]) for row in rows),
            "last_signal": max(rows, key=lambda row: int(row["last_round"]))[
                "last_signal"
            ],
            "last_round": max(int(row["last_round"]) for row in rows),
            "last_evidence_id": next(
                (
                    row["last_evidence_id"]
                    for row in sorted(
                        rows, key=lambda item: int(item["last_round"]), reverse=True
                    )
                    if row["last_evidence_id"] is not None
                ),
                None,
            ),
            "source": "derived_from_per_kc_facets",
        }
    return projection


def _recompute(model: dict[str, Any]) -> None:
    components = model["knowledge_components"]
    for component in components.values():
        facets = component["dimensions"]
        component["p_mastery"] = _round(
            sum(float(row["p_mastery"]) for row in facets.values()) / len(facets)
        )
        component["uncertainty"] = _round(
            sum(float(row["uncertainty"]) for row in facets.values()) / len(facets)
        )
        component["evidence_count"] = sum(
            int(row["evidence_count"]) for row in facets.values()
        )
        component["positive_evidence"] = sum(
            int(row["positive_evidence"]) for row in facets.values()
        )
        component["negative_evidence"] = sum(
            int(row["negative_evidence"]) for row in facets.values()
        )
    model["dimensions"] = _legacy_dimension_projection(
        components, prior_strength=float(model["prior_strength"])
    )
    model["overall"] = {
        "p_mastery": _round(
            sum(float(row["p_mastery"]) for row in components.values())
            / len(components)
        ),
        "uncertainty": _round(
            sum(float(row["uncertainty"]) for row in components.values())
            / len(components)
        ),
        "evidence_count": sum(
            int(row["evidence_count"]) for row in components.values()
        ),
    }
    compatibility = model.get("compatibility")
    if isinstance(compatibility, dict):
        kc_projection = {
            dimension: float(model["dimensions"][dimension]["p_mastery"])
            for dimension in MASTERY_DIMENSIONS
        }
        if compatibility.get("legacy_runtime_state_synchronized_from_kc_model") is True:
            compatibility["legacy_projection_snapshot"] = dict(kc_projection)
            compatibility["kc_projection_snapshot"] = dict(kc_projection)
            compatibility["absolute_divergence"] = {
                dimension: 0.0 for dimension in MASTERY_DIMENSIONS
            }
        else:
            legacy_projection = compatibility.get("legacy_projection_snapshot", {})
            compatibility["kc_projection_snapshot"] = kc_projection
            compatibility["absolute_divergence"] = {
                dimension: _round(
                    abs(
                        float(
                            legacy_projection.get(dimension, kc_projection[dimension])
                        )
                        - kc_projection[dimension]
                    )
                )
                for dimension in MASTERY_DIMENSIONS
            }


def initialize_student_model(
    initial_mastery: Mapping[str, Any] | None = None,
    *,
    knowledge_components: Sequence[Mapping[str, Any] | str] | None = None,
    goal: Mapping[str, Any] | None = None,
    prior_strength: float = 4.0,
) -> dict[str, Any]:
    """Create an evidence-free per-KC learner model from teacher priors."""

    initial_mastery = initial_mastery or {}
    if not isinstance(initial_mastery, Mapping):
        raise StudentModelError("initial_mastery must be an object")
    if isinstance(prior_strength, bool) or not isinstance(prior_strength, (int, float)):
        raise StudentModelError("prior_strength must be a number >= 2")
    if not math.isfinite(float(prior_strength)) or float(prior_strength) < 2:
        raise StudentModelError("prior_strength must be a finite number >= 2")

    if goal is not None:
        definitions = knowledge_component_definitions(goal)
    elif knowledge_components is None:
        definitions = [
            {
                "kc_id": "kc_legacy_global",
                "label": "legacy_global",
                "source": "legacy_initial_mastery",
                "source_ref": "legacy_initial_mastery#kc_legacy_global",
                "teacher_grading_authority_available": False,
            }
        ]
    else:
        synthetic_goal = {
            "concept": "teacher_goal",
            "knowledge_components": list(knowledge_components),
        }
        definitions = knowledge_component_definitions(synthetic_goal)
    components = {
        str(definition["kc_id"]): _component_row(
            definition,
            initial_mastery,
            prior_strength=float(prior_strength),
        )
        for definition in definitions
    }
    initial_projection = {
        dimension: _probability(initial_mastery.get(dimension, 0.0), field=dimension)
        for dimension in MASTERY_DIMENSIONS
    }
    model: dict[str, Any] = {
        "schema": STUDENT_MODEL_SCHEMA,
        "version": 2,
        "prior_strength": round(float(prior_strength), 4),
        "knowledge_components": components,
        "dimensions": {},
        "overall": {},
        "adjudication_revisions": [],
        "last_update": {
            "round": 0,
            "signal": "not_observed",
            "focus_dimension": None,
            "knowledge_component_ids": [],
            "evidence_id": None,
            "item_id": None,
            "rubric_id": None,
            "source": "teacher_profile_prior",
            "update_applied": False,
            "needs_human_review": False,
            "reason": "waiting_for_authoritative_assessment_evidence",
        },
        "recommended_focus": {
            "dimension": "prerequisite",
            "knowledge_component_id": definitions[0]["kc_id"],
            "knowledge_component_label": definitions[0]["label"],
            "reason": "等待首个有界观察",
            "confidence": 0.0,
            "evidence_refs": [],
        },
        "migration": {
            "source_schema": None,
            "unattributed_legacy_evidence_count": 0,
            "legacy_evidence_attributed_to_components": False,
        },
        "compatibility": {
            "legacy_decision_state_path": "student_state.knowledge_mastery",
            "legacy_success_gate_uses_kc_model": False,
            "legacy_runtime_state_synchronized_from_kc_model": False,
            "legacy_projection_snapshot": initial_projection,
            "kc_projection_snapshot": dict(initial_projection),
            "absolute_divergence": {dimension: 0.0 for dimension in MASTERY_DIMENSIONS},
        },
        "claim_boundary": {
            "is_ground_truth": False,
            "teacher_prior_overwritten": False,
            "free_text_accuracy_established": False,
            "uncertainty_is_calibrated_on_external_lockbox": False,
            "single_evidence_updates_multiple_components": False,
            "unrubriced_observation_updates_mastery": False,
        },
    }
    _recompute(model)
    validate_student_model(model)
    return model


def migrate_student_model(
    model: Mapping[str, Any] | None,
    *,
    initial_mastery: Mapping[str, Any] | None = None,
    goal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Migrate a v1/missing model without inventing per-KC evidence.

    Old global evidence cannot be attributed to a component after the fact.
    Its count is retained in the migration receipt, while v1 mastery means are
    used only as v2 priors for each explicitly defined component.
    """

    if isinstance(model, Mapping) and model.get("schema") == STUDENT_MODEL_SCHEMA:
        candidate = deepcopy(dict(model))
        validate_student_model(candidate)
        return candidate
    priors = dict(initial_mastery or {})
    legacy_count = 0
    if (
        isinstance(model, Mapping)
        and model.get("schema") == LEGACY_STUDENT_MODEL_SCHEMA
    ):
        dimensions = model.get("dimensions", {})
        if isinstance(dimensions, Mapping):
            for dimension in MASTERY_DIMENSIONS:
                row = dimensions.get(dimension)
                if isinstance(row, Mapping) and isinstance(
                    row.get("p_mastery"), (int, float)
                ):
                    priors[dimension] = float(row["p_mastery"])
            legacy_count = sum(
                int(row.get("evidence_count", 0))
                for row in dimensions.values()
                if isinstance(row, Mapping)
                and isinstance(row.get("evidence_count", 0), int)
            )
    migrated = initialize_student_model(priors, goal=goal)
    migrated["migration"] = {
        "source_schema": (
            str(model.get("schema")) if isinstance(model, Mapping) else "missing"
        ),
        "unattributed_legacy_evidence_count": legacy_count,
        "legacy_evidence_attributed_to_components": False,
    }
    validate_student_model(migrated)
    return migrated


def _validate_facet(row: Any, *, field: str) -> None:
    if not isinstance(row, Mapping):
        raise StudentModelError(f"{field} must be an object")
    _probability(row.get("p_mastery"), field=f"{field}.p_mastery")
    _probability(row.get("prior_p_mastery"), field=f"{field}.prior_p_mastery")
    _probability(row.get("uncertainty"), field=f"{field}.uncertainty")
    for key in (
        "evidence_count",
        "positive_evidence",
        "negative_evidence",
        "last_round",
    ):
        _bounded_int(row.get(key), field=f"{field}.{key}")
    if row.get("last_signal") not in _SIGNALS:
        raise StudentModelError(f"{field}.last_signal is invalid")
    if row.get("last_observed_at") is not None and not _UTC_TIMESTAMP.fullmatch(
        str(row.get("last_observed_at"))
    ):
        raise StudentModelError(f"{field}.last_observed_at is invalid")
    if row.get("last_time_basis") not in {None, *_TIME_BASES}:
        raise StudentModelError(f"{field}.last_time_basis is invalid")


def validate_student_model(model: Mapping[str, Any]) -> None:
    if not isinstance(model, Mapping) or model.get("schema") != STUDENT_MODEL_SCHEMA:
        raise StudentModelError(f"student model schema must be {STUDENT_MODEL_SCHEMA}")
    if model.get("version") != 2:
        raise StudentModelError("unsupported student model version")
    prior_strength = model.get("prior_strength")
    if (
        isinstance(prior_strength, bool)
        or not isinstance(prior_strength, (int, float))
        or not math.isfinite(float(prior_strength))
        or float(prior_strength) < 2
    ):
        raise StudentModelError("student model prior_strength is invalid")
    components = model.get("knowledge_components")
    if not isinstance(components, Mapping) or not 1 <= len(components) <= 24:
        raise StudentModelError("student model knowledge_components are invalid")
    labels: set[str] = set()
    all_evidence_ids: dict[str, str] = {}
    for kc_id, component in components.items():
        if not isinstance(kc_id, str) or not _SAFE_KC_ID.fullmatch(kc_id):
            raise StudentModelError("student model knowledge component ID is invalid")
        if not isinstance(component, Mapping) or component.get("kc_id") != kc_id:
            raise StudentModelError(f"student model component {kc_id} is invalid")
        label = _text(
            component.get("label"), field=f"knowledge_components.{kc_id}.label"
        )
        canonical = _canonical_label(label)
        if canonical in labels:
            raise StudentModelError("student model component labels must be unique")
        labels.add(canonical)
        for field in ("source", "source_ref"):
            _text(
                component.get(field),
                field=f"knowledge_components.{kc_id}.{field}",
                maximum=300,
            )
        if not isinstance(component.get("teacher_grading_authority_available"), bool):
            raise StudentModelError(
                f"knowledge_components.{kc_id}.authority is invalid"
            )
        _probability(
            component.get("p_mastery"), field=f"knowledge_components.{kc_id}.p_mastery"
        )
        _probability(
            component.get("uncertainty"),
            field=f"knowledge_components.{kc_id}.uncertainty",
        )
        for field in (
            "evidence_count",
            "positive_evidence",
            "negative_evidence",
            "last_round",
        ):
            _bounded_int(
                component.get(field), field=f"knowledge_components.{kc_id}.{field}"
            )
        if component.get("last_signal") not in _SIGNALS:
            raise StudentModelError(
                f"knowledge_components.{kc_id}.last_signal is invalid"
            )
        if component.get(
            "last_observed_at"
        ) is not None and not _UTC_TIMESTAMP.fullmatch(
            str(component.get("last_observed_at"))
        ):
            raise StudentModelError(
                f"knowledge_components.{kc_id}.last_observed_at is invalid"
            )
        if component.get("last_time_basis") not in {None, *_TIME_BASES}:
            raise StudentModelError(
                f"knowledge_components.{kc_id}.last_time_basis is invalid"
            )
        bkt = component.get("bkt")
        if not isinstance(bkt, Mapping):
            raise StudentModelError(f"knowledge_components.{kc_id}.bkt is invalid")
        for field in ("p_learn", "p_slip", "p_guess", "forgetting_rate_per_day"):
            _probability(
                bkt.get(field), field=f"knowledge_components.{kc_id}.bkt.{field}"
            )
        calibration = component.get("calibration")
        if not isinstance(calibration, Mapping) or calibration.get("status") != (
            "descriptive_only_external_calibration_required"
        ):
            raise StudentModelError(
                f"knowledge_components.{kc_id}.calibration is invalid"
            )
        _bounded_int(
            calibration.get("sample_count"),
            field=f"knowledge_components.{kc_id}.calibration.sample_count",
        )
        brier_score = calibration.get("brier_score")
        if brier_score is not None:
            _probability(
                brier_score,
                field=f"knowledge_components.{kc_id}.calibration.brier_score",
            )
        brier_sum = calibration.get("brier_sum")
        if (
            isinstance(brier_sum, bool)
            or not isinstance(brier_sum, (int, float))
            or not math.isfinite(float(brier_sum))
            or float(brier_sum) < 0
        ):
            raise StudentModelError(
                f"knowledge_components.{kc_id}.calibration.brier_sum is invalid"
            )
        facets = component.get("dimensions")
        if not isinstance(facets, Mapping) or set(facets) != set(MASTERY_DIMENSIONS):
            raise StudentModelError(
                f"knowledge_components.{kc_id}.dimensions are invalid"
            )
        for dimension in MASTERY_DIMENSIONS:
            _validate_facet(
                facets[dimension],
                field=f"knowledge_components.{kc_id}.dimensions.{dimension}",
            )
        ledger = component.get("evidence_ledger")
        if not isinstance(ledger, list) or len(ledger) > _MAX_EVIDENCE_PER_COMPONENT:
            raise StudentModelError(
                f"knowledge_components.{kc_id}.evidence_ledger is invalid"
            )
        active_evidence = 0
        superseded_evidence = 0
        for entry in ledger:
            if (
                not isinstance(entry, Mapping)
                or entry.get("knowledge_component_id") != kc_id
            ):
                raise StudentModelError(
                    f"knowledge_components.{kc_id}.evidence entry is invalid"
                )
            evidence_id = _text(
                entry.get("evidence_id"), field="evidence_id", maximum=160
            )
            owner = all_evidence_ids.get(evidence_id)
            if owner is not None:
                raise StudentModelError(f"evidence_id {evidence_id} is duplicated")
            all_evidence_ids[evidence_id] = kc_id
            lifecycle_status = entry.get("lifecycle_status", "active")
            if lifecycle_status not in {"active", "superseded"}:
                raise StudentModelError("evidence lifecycle_status is invalid")
            superseded_by = entry.get("superseded_by_revision_id")
            if lifecycle_status == "active":
                active_evidence += 1
                if superseded_by is not None:
                    raise StudentModelError(
                        "active evidence cannot name a superseding revision"
                    )
            else:
                superseded_evidence += 1
                if not isinstance(superseded_by, str) or not re.fullmatch(
                    r"smrev_[0-9a-f]{24}", superseded_by
                ):
                    raise StudentModelError(
                        "superseded evidence must name its revision receipt"
                    )
            if entry.get("signal") not in _UPDATABLE_SIGNALS:
                raise StudentModelError("evidence signal is invalid")
            if (
                entry.get("assessment_eligible") is not True
                or entry.get("authoritative") is not True
            ):
                raise StudentModelError(
                    "persisted mastery evidence must be eligible and authoritative"
                )
            for field in (
                "item_id",
                "question_id",
                "rubric_id",
                "source",
                "observed_at",
                "time_basis",
                "focus_dimension",
                "evidence_fingerprint",
            ):
                _text(entry.get(field), field=f"evidence.{field}", maximum=300)
            if entry.get("focus_dimension") not in MASTERY_DIMENSIONS:
                raise StudentModelError("evidence focus_dimension is invalid")
            if not _UTC_TIMESTAMP.fullmatch(str(entry.get("observed_at"))):
                raise StudentModelError("evidence observed_at is invalid")
            if entry.get("time_basis") not in _TIME_BASES:
                raise StudentModelError("evidence time_basis is invalid")
            if not isinstance(entry.get("forgetting_applied"), bool):
                raise StudentModelError("evidence forgetting_applied is invalid")
            for field in ("confidence", "p_mastery_before", "p_mastery_after"):
                _probability(entry.get(field), field=f"evidence.{field}")
            round_value = entry.get("round_number")
            if round_value is not None:
                _bounded_int(round_value, field="evidence.round_number")
            authority_evidence_sha256 = entry.get("authority_evidence_sha256")
            if authority_evidence_sha256 is not None and (
                not isinstance(authority_evidence_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", authority_evidence_sha256) is None
            ):
                raise StudentModelError("evidence.authority_evidence_sha256 is invalid")
            if entry.get("evidence_fingerprint") != _evidence_fingerprint(
                _evidence_material_from_entry(entry)
            ):
                raise StudentModelError("evidence fingerprint integrity check failed")
            for field in ("difficulty", "discrimination"):
                value = entry.get(field)
                if value is not None:
                    _probability(value, field=f"evidence.{field}")
        ledger_state = component.get("evidence_ledger_state")
        if ledger_state is not None:
            if not isinstance(ledger_state, Mapping) or set(ledger_state) != {
                "capacity",
                "total_recorded_evidence",
                "retained_evidence",
                "active_evidence",
                "superseded_evidence",
                "complete_from_initial_prior",
            }:
                raise StudentModelError(
                    f"knowledge_components.{kc_id}.evidence_ledger_state is invalid"
                )
            if ledger_state.get("capacity") != _MAX_EVIDENCE_PER_COMPONENT:
                raise StudentModelError(
                    f"knowledge_components.{kc_id}.evidence ledger capacity is invalid"
                )
            for field in (
                "total_recorded_evidence",
                "retained_evidence",
                "active_evidence",
                "superseded_evidence",
            ):
                _bounded_int(
                    ledger_state.get(field),
                    field=f"knowledge_components.{kc_id}.evidence_ledger_state.{field}",
                )
            if ledger_state.get("retained_evidence") != len(ledger):
                raise StudentModelError(
                    f"knowledge_components.{kc_id}.retained evidence count mismatches"
                )
            if ledger_state.get("active_evidence") != active_evidence:
                raise StudentModelError(
                    f"knowledge_components.{kc_id}.active evidence count mismatches"
                )
            if ledger_state.get("superseded_evidence") != superseded_evidence:
                raise StudentModelError(
                    f"knowledge_components.{kc_id}.superseded evidence count mismatches"
                )
            complete = ledger_state.get("complete_from_initial_prior")
            if not isinstance(complete, bool):
                raise StudentModelError(
                    f"knowledge_components.{kc_id}.ledger completeness is invalid"
                )
            if complete != (ledger_state.get("total_recorded_evidence") == len(ledger)):
                raise StudentModelError(
                    f"knowledge_components.{kc_id}.ledger completeness is inconsistent"
                )
            if complete and component.get("evidence_count") != active_evidence:
                raise StudentModelError(
                    f"knowledge_components.{kc_id}.active evidence is inconsistent"
                )

    _validate_adjudication_revisions(model, components=components)

    dimensions = model.get("dimensions")
    if not isinstance(dimensions, Mapping) or set(dimensions) != set(
        MASTERY_DIMENSIONS
    ):
        raise StudentModelError("student model dimensions are incomplete")
    for dimension in MASTERY_DIMENSIONS:
        row = dimensions[dimension]
        if not isinstance(row, Mapping):
            raise StudentModelError(f"student model dimension {dimension} is invalid")
        _probability(row.get("p_mastery"), field=f"dimensions.{dimension}.p_mastery")
        _probability(
            row.get("uncertainty"), field=f"dimensions.{dimension}.uncertainty"
        )
        for field in (
            "evidence_count",
            "positive_evidence",
            "negative_evidence",
            "last_round",
        ):
            _bounded_int(row.get(field), field=f"dimensions.{dimension}.{field}")
        if row.get("last_signal") not in _SIGNALS:
            raise StudentModelError(f"dimensions.{dimension}.last_signal is invalid")
        for field in ("alpha", "beta"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) <= 0
            ):
                raise StudentModelError(f"dimensions.{dimension}.{field} is invalid")
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
    if (
        update.get("focus_dimension") is not None
        and update.get("focus_dimension") not in MASTERY_DIMENSIONS
    ):
        raise StudentModelError("student model last_update.focus_dimension is invalid")
    ids = update.get("knowledge_component_ids")
    if (
        not isinstance(ids, list)
        or len(ids) > 1
        or any(item not in components for item in ids)
    ):
        raise StudentModelError(
            "student model last_update knowledge_component_ids are invalid"
        )
    if not isinstance(update.get("update_applied"), bool) or not isinstance(
        update.get("needs_human_review"), bool
    ):
        raise StudentModelError("student model update flags are invalid")
    _text(update.get("source"), field="last_update.source", maximum=160)
    _text(update.get("reason"), field="last_update.reason", maximum=200)
    recommendation = model.get("recommended_focus")
    if (
        not isinstance(recommendation, Mapping)
        or recommendation.get("dimension") not in MASTERY_DIMENSIONS
    ):
        raise StudentModelError("student model recommended_focus is invalid")
    if recommendation.get("knowledge_component_id") not in components:
        raise StudentModelError("student model recommended KC is invalid")
    _probability(recommendation.get("confidence"), field="recommended_focus.confidence")
    refs = recommendation.get("evidence_refs")
    if (
        not isinstance(refs, list)
        or len(refs) > 8
        or any(not isinstance(item, str) or not item for item in refs)
    ):
        raise StudentModelError(
            "student model recommendation evidence_refs are invalid"
        )
    migration = model.get("migration")
    if (
        not isinstance(migration, Mapping)
        or migration.get("legacy_evidence_attributed_to_components") is not False
    ):
        raise StudentModelError("student model migration receipt is invalid")
    _bounded_int(
        migration.get("unattributed_legacy_evidence_count"),
        field="migration.unattributed_legacy_evidence_count",
    )
    compatibility = model.get("compatibility")
    if (
        not isinstance(compatibility, Mapping)
        or compatibility.get("legacy_decision_state_path")
        != "student_state.knowledge_mastery"
    ):
        raise StudentModelError("student model compatibility receipt is invalid")
    success_gate_uses_kc_model = compatibility.get("legacy_success_gate_uses_kc_model")
    runtime_state_synchronized = compatibility.get(
        "legacy_runtime_state_synchronized_from_kc_model"
    )
    if (
        not isinstance(success_gate_uses_kc_model, bool)
        or not isinstance(runtime_state_synchronized, bool)
        or success_gate_uses_kc_model != runtime_state_synchronized
    ):
        raise StudentModelError("student model compatibility mode is invalid")
    for field in (
        "legacy_projection_snapshot",
        "kc_projection_snapshot",
        "absolute_divergence",
    ):
        projection = compatibility.get(field)
        if not isinstance(projection, Mapping) or set(projection) != set(
            MASTERY_DIMENSIONS
        ):
            raise StudentModelError(f"student model compatibility.{field} is invalid")
        for dimension, value in projection.items():
            _probability(value, field=f"compatibility.{field}.{dimension}")
    if success_gate_uses_kc_model:
        if compatibility["legacy_projection_snapshot"] != compatibility[
            "kc_projection_snapshot"
        ] or any(
            float(value) != 0.0
            for value in compatibility["absolute_divergence"].values()
        ):
            raise StudentModelError(
                "KC-authoritative runtime projection must be synchronized"
            )
    boundary = model.get("claim_boundary")
    required_false = (
        "is_ground_truth",
        "teacher_prior_overwritten",
        "free_text_accuracy_established",
        "uncertainty_is_calibrated_on_external_lockbox",
        "single_evidence_updates_multiple_components",
        "unrubriced_observation_updates_mastery",
    )
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) is not False for key in required_false
    ):
        raise StudentModelError("student model claim boundary is invalid")


def _parse_timestamp(value: str, *, field: str) -> datetime:
    if not _UTC_TIMESTAMP.fullmatch(value):
        raise StudentModelError(f"{field} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise StudentModelError(f"{field} must be an RFC3339 UTC timestamp") from exc
    return parsed.astimezone(timezone.utc)


def _evidence_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _evidence_material_from_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "evidence_id",
        "item_id",
        "question_id",
        "rubric_id",
        "knowledge_component_id",
        "focus_dimension",
        "signal",
        "confidence",
        "answer_alignment",
        "difficulty",
        "discrimination",
        "observed_at",
        "time_basis",
        "source",
        "assessment_eligible",
        "authoritative",
    )
    material = {field: deepcopy(entry.get(field)) for field in fields}
    # These replay-critical fields were added after the first v2 rollout.  Their
    # absence is retained in the original fingerprint and later causes the
    # adjudication API to request an external ledger rather than invent values.
    for optional in ("round_number", "authority_evidence_sha256"):
        if optional in entry:
            material[optional] = deepcopy(entry[optional])
    return material


def _validate_adjudication_revisions(
    model: Mapping[str, Any], *, components: Mapping[str, Any]
) -> None:
    revisions = model.get("adjudication_revisions", [])
    if not isinstance(revisions, list) or len(revisions) > _MAX_ADJUDICATION_REVISIONS:
        raise StudentModelError("student model adjudication_revisions is invalid")
    previous_hash: str | None = None
    seen_instruction_hashes: set[str] = set()
    for index, raw in enumerate(revisions, start=1):
        base_keys = {
            "schema",
            "revision_number",
            "revision_id",
            "decision_kind",
            "operation",
            "target_kc_id",
            "source_evidence_id",
            "source_evidence_fingerprint",
            "instruction_sha256",
            "before_component_sha256",
            "after_component_sha256",
            "active_evidence_replayed",
            "superseded_evidence_count",
            "ledger_complete_from_initial_prior",
            "negative_delta_applied",
            "previous_revision_sha256",
            "revision_sha256",
        }
        if not isinstance(raw, Mapping) or frozenset(raw) not in {
            frozenset(base_keys),
            frozenset(base_keys | {"authority_revalidation_receipt_sha256"}),
        }:
            raise StudentModelError(
                f"adjudication_revisions[{index - 1}] shape is invalid"
            )
        receipt = dict(raw)
        if receipt.get("schema") != STUDENT_MODEL_ADJUDICATION_REVISION_SCHEMA:
            raise StudentModelError("adjudication revision schema is unsupported")
        if receipt.get("revision_number") != index:
            raise StudentModelError("adjudication revision numbers are non-contiguous")
        revision_id = receipt.get("revision_id")
        if (
            not isinstance(revision_id, str)
            or re.fullmatch(r"smrev_[0-9a-f]{24}", revision_id) is None
        ):
            raise StudentModelError("adjudication revision ID is invalid")
        if receipt.get("decision_kind") not in {"abstain", "correct"}:
            raise StudentModelError("adjudication revision decision kind is invalid")
        authority_receipt_hash = receipt.get("authority_revalidation_receipt_sha256")
        if receipt["decision_kind"] == "correct":
            if (
                not isinstance(authority_receipt_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", authority_receipt_hash) is None
            ):
                raise StudentModelError(
                    "authenticated correction revision authority is invalid"
                )
        elif authority_receipt_hash is not None:
            raise StudentModelError(
                "abstention revision cannot claim authenticated correction authority"
            )
        if receipt.get("operation") != "supersede_and_replay":
            raise StudentModelError("adjudication revision operation is invalid")
        kc_id = receipt.get("target_kc_id")
        if kc_id not in components:
            raise StudentModelError("adjudication revision target KC is invalid")
        _text(
            receipt.get("source_evidence_id"),
            field="adjudication revision source_evidence_id",
            maximum=160,
        )
        for field in (
            "source_evidence_fingerprint",
            "instruction_sha256",
            "before_component_sha256",
            "after_component_sha256",
            "revision_sha256",
        ):
            value = receipt.get(field)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise StudentModelError(f"adjudication revision {field} is invalid")
        instruction_hash = str(receipt["instruction_sha256"])
        if instruction_hash in seen_instruction_hashes:
            raise StudentModelError("adjudication instruction receipt is duplicated")
        seen_instruction_hashes.add(instruction_hash)
        for field in ("active_evidence_replayed", "superseded_evidence_count"):
            _bounded_int(receipt.get(field), field=f"adjudication revision {field}")
        if receipt.get("ledger_complete_from_initial_prior") is not True:
            raise StudentModelError(
                "applied adjudication revision must bind a full ledger"
            )
        if receipt.get("negative_delta_applied") is not False:
            raise StudentModelError(
                "adjudication revisions cannot apply negative deltas"
            )
        if receipt.get("previous_revision_sha256") != previous_hash:
            raise StudentModelError("adjudication revision hash chain is broken")
        material = deepcopy(receipt)
        declared = material.pop("revision_sha256")
        if declared != _evidence_fingerprint(material):
            raise StudentModelError("adjudication revision integrity check failed")
        component = components[str(kc_id)]
        linked_entry = next(
            (
                entry
                for entry in component["evidence_ledger"]
                if entry.get("evidence_id") == receipt["source_evidence_id"]
            ),
            None,
        )
        ledger_state = component.get("evidence_ledger_state")
        ledger_complete = (
            not isinstance(ledger_state, Mapping)
            or ledger_state.get("complete_from_initial_prior") is True
        )
        if linked_entry is None and ledger_complete:
            raise StudentModelError(
                "adjudication revision source is missing from a complete ledger"
            )
        if linked_entry is not None and (
            linked_entry.get("evidence_fingerprint")
            != receipt["source_evidence_fingerprint"]
            or linked_entry.get("lifecycle_status") != "superseded"
            or linked_entry.get("superseded_by_revision_id") != revision_id
        ):
            raise StudentModelError(
                "adjudication revision does not match the superseded evidence"
            )
        previous_hash = str(declared)


def _last_update_without_evidence(
    candidate: dict[str, Any],
    *,
    signal: str,
    focus_dimension: str | None,
    component_ids: list[str],
    round_number: int,
    evidence_id: str | None,
    item_id: str | None,
    rubric_id: str | None,
    source: str,
    needs_human_review: bool,
    reason: str,
) -> dict[str, Any]:
    candidate["last_update"] = {
        "round": max(0, int(round_number)),
        "signal": signal,
        "focus_dimension": focus_dimension,
        "knowledge_component_ids": component_ids[:1],
        "evidence_id": str(evidence_id)[:160] if evidence_id else None,
        "item_id": str(item_id)[:160] if item_id else None,
        "rubric_id": str(rubric_id)[:160] if rubric_id else None,
        "source": str(source or "unattributed_observation")[:160],
        "update_applied": False,
        "needs_human_review": bool(needs_human_review),
        "reason": reason,
    }
    validate_student_model(candidate)
    return candidate


def _adjudication_result(
    *,
    status: str,
    model: Mapping[str, Any],
    instruction_sha256: str,
    target_kc_id: str,
    source_evidence_id: str,
    revision_receipt: Mapping[str, Any] | None = None,
    pending_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": STUDENT_MODEL_ADJUDICATION_RESULT_SCHEMA,
        "status": status,
        "model": deepcopy(dict(model)),
        "instruction_sha256": instruction_sha256,
        "target_kc_id": target_kc_id,
        "source_evidence_id": source_evidence_id,
        "revision_receipt": (
            deepcopy(dict(revision_receipt))
            if isinstance(revision_receipt, Mapping)
            else None
        ),
        "pending": (
            None
            if pending_reason is None
            else {
                "reason": pending_reason,
                "model_mutated": False,
                "mastery_update_authorized": False,
            }
        ),
    }


def _validate_adjudication_instruction(
    instruction: Mapping[str, Any],
    *,
    target_kc_id: str,
    source_evidence_id: str,
) -> tuple[dict[str, Any], str]:
    if not isinstance(instruction, Mapping) or set(instruction) != {
        "schema",
        "instruction_id",
        "review_item_id",
        "review_item_version",
        "review_item_version_sha256",
        "operation",
        "supersedes_evidence_id",
        "evidence_sha256",
        "target_kc_ids",
        "correction",
        "authority",
        "mutates_student_model",
        "mastery_update_authorized",
        "requires_explicit_apply",
        "instruction_sha256",
    }:
        raise StudentModelError("adjudication instruction shape is invalid")
    value = deepcopy(dict(instruction))
    if value.get("schema") != _ADJUDICATION_INSTRUCTION_SCHEMA:
        raise StudentModelError("adjudication instruction schema is unsupported")
    operation = value.get("operation")
    if operation not in _ADJUDICATION_OPERATIONS or operation == "cancel_adjudication":
        raise StudentModelError("a decided adjudication instruction is required")
    if (
        not isinstance(value.get("instruction_id"), str)
        or re.fullmatch(r"adjinst_[0-9a-f]{24}", value["instruction_id"]) is None
    ):
        raise StudentModelError("adjudication instruction ID is invalid")
    if (
        not isinstance(value.get("review_item_id"), str)
        or re.fullmatch(r"adj_[0-9a-f]{24}", value["review_item_id"]) is None
    ):
        raise StudentModelError("adjudication review item ID is invalid")
    _bounded_int(
        value.get("review_item_version"),
        field="adjudication instruction review_item_version",
        minimum=1,
    )
    for field in (
        "review_item_version_sha256",
        "evidence_sha256",
        "instruction_sha256",
    ):
        raw = value.get(field)
        if not isinstance(raw, str) or re.fullmatch(r"[0-9a-f]{64}", raw) is None:
            raise StudentModelError(f"adjudication instruction {field} is invalid")
    expected_instruction_id = (
        "adjinst_"
        + _evidence_fingerprint(
            {
                "item_id": value["review_item_id"],
                "version_sha256": value["review_item_version_sha256"],
            }
        )[:24]
    )
    if value["instruction_id"] != expected_instruction_id:
        raise StudentModelError("adjudication instruction ID binding is invalid")
    declared_hash = str(value["instruction_sha256"])
    material = deepcopy(value)
    material.pop("instruction_sha256")
    if declared_hash != _evidence_fingerprint(material):
        raise StudentModelError("adjudication instruction integrity check failed")
    if value.get("supersedes_evidence_id") != source_evidence_id:
        raise StudentModelError("adjudication instruction evidence ID does not match")
    if value.get("target_kc_ids") != [target_kc_id]:
        raise StudentModelError(
            "student-model adjudication requires exactly the requested target KC"
        )
    authority = value.get("authority")
    local_authority = {
        "identity": "local_operator_not_authenticated",
        "authenticated": False,
        "teacher_identity_claimed": False,
        "authoritative_for_mastery_update": False,
        "requires_downstream_authority_revalidation": True,
    }
    authenticated_authority = False
    if isinstance(authority, Mapping) and dict(authority) == local_authority:
        pass
    elif isinstance(authority, Mapping) and set(authority) == {
        "identity",
        "authenticated",
        "teacher_identity_claimed",
        "principal_sha256",
        "roles_sha256",
        "authority_id",
        "assurance",
        "gateway_verification_receipt_sha256",
        "personal_non_repudiation",
        "authoritative_for_mastery_update",
        "requires_downstream_authority_revalidation",
        "server_revalidation_receipt",
    }:
        authenticated_authority = True
        if (
            authority.get("identity") != AUTHENTICATED_TEACHER_ACTOR
            or authority.get("authenticated") is not True
            or authority.get("teacher_identity_claimed") is not True
            or authority.get("assurance") != AUTHORITY_ASSURANCE
            or authority.get("personal_non_repudiation") is not False
            or not isinstance(authority.get("authority_id"), str)
            or re.fullmatch(r"tauth_[0-9a-f]{24}", authority["authority_id"]) is None
        ):
            raise StudentModelError(
                "adjudication instruction authority boundary is invalid"
            )
        for field in (
            "principal_sha256",
            "roles_sha256",
            "gateway_verification_receipt_sha256",
        ):
            if (
                not isinstance(authority.get(field), str)
                or re.fullmatch(r"[0-9a-f]{64}", authority[field]) is None
            ):
                raise StudentModelError(
                    "adjudication instruction authority boundary is invalid"
                )
        if operation == "supersede_and_replay":
            if (
                authority.get("authoritative_for_mastery_update") is not True
                or authority.get("requires_downstream_authority_revalidation")
                is not False
            ):
                raise StudentModelError(
                    "adjudication instruction authority boundary is invalid"
                )
            try:
                validate_teacher_authority_revalidation_receipt(
                    authority.get("server_revalidation_receipt")
                )
            except TeacherAuthorityError as exc:
                raise StudentModelError(
                    "adjudication instruction authority receipt is invalid"
                ) from exc
        elif (
            authority.get("authoritative_for_mastery_update") is not False
            or authority.get("requires_downstream_authority_revalidation") is not True
            or authority.get("server_revalidation_receipt") is not None
        ):
            raise StudentModelError(
                "adjudication instruction authority boundary is invalid"
            )
    else:
        raise StudentModelError(
            "adjudication instruction authority boundary is invalid"
        )
    if value.get("requires_explicit_apply") is not True:
        raise StudentModelError("adjudication instruction mutation boundary is invalid")
    expected_mutation = authenticated_authority and operation == "supersede_and_replay"
    if (
        value.get("mutates_student_model") is not expected_mutation
        or value.get("mastery_update_authorized") is not expected_mutation
    ):
        raise StudentModelError("adjudication instruction mutation boundary is invalid")
    correction = value.get("correction")
    if operation == "supersede_and_replay":
        if not isinstance(correction, Mapping) or not correction:
            raise StudentModelError(
                "correct adjudication instruction lacks a correction"
            )
        if not set(correction).issubset(
            {"signal", "answer_alignment", "focus_dimension", "target_kc_ids"}
        ):
            raise StudentModelError("adjudication correction fields are unbounded")
        if "signal" in correction and correction["signal"] not in _UPDATABLE_SIGNALS:
            raise StudentModelError("adjudication correction signal is invalid")
        if "answer_alignment" in correction and correction["answer_alignment"] not in {
            "aligned",
            "partially_aligned",
            "contradicted",
            "ambiguous",
        }:
            raise StudentModelError("adjudication correction alignment is invalid")
        if (
            "focus_dimension" in correction
            and correction["focus_dimension"] not in MASTERY_DIMENSIONS
        ):
            raise StudentModelError("adjudication correction dimension is invalid")
        if "target_kc_ids" in correction and correction["target_kc_ids"] != [
            target_kc_id
        ]:
            raise StudentModelError("adjudication correction target KC is invalid")
    elif correction is not None:
        raise StudentModelError(
            "approve and abstain adjudication instructions cannot contain corrections"
        )
    return value, declared_hash


def _component_ledger_is_complete(component: Mapping[str, Any]) -> bool:
    ledger = component["evidence_ledger"]
    state = component.get("evidence_ledger_state")
    if isinstance(state, Mapping):
        return bool(state.get("complete_from_initial_prior")) and int(
            state.get("total_recorded_evidence", -1)
        ) == len(ledger)
    # Pre-counter v2 records can prove truncation whenever the aggregate active
    # count exceeds the retained entries.  They have no superseded evidence.
    return int(component.get("evidence_count", 0)) == len(ledger)


def _revalidate_authenticated_adjudication(
    instruction: Mapping[str, Any],
    *,
    source_entry: Mapping[str, Any],
    authority_revalidator: (Callable[[Mapping[str, Any]], Mapping[str, Any]] | None),
) -> dict[str, Any] | None:
    authority = instruction.get("authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("identity") != AUTHENTICATED_TEACHER_ACTOR
    ):
        return None
    if instruction.get("operation") != "supersede_and_replay":
        return None
    if authority_revalidator is None:
        raise StudentModelError(
            "authenticated correction requires a server authority revalidator"
        )
    raw_receipt = authority.get("server_revalidation_receipt")
    try:
        receipt = validate_teacher_authority_revalidation_receipt(
            authority_revalidator(raw_receipt)
        )
    except Exception as exc:
        raise StudentModelError(
            "authenticated correction authority revalidation failed"
        ) from exc
    basis = deepcopy(dict(instruction))
    basis.pop("instruction_sha256", None)
    basis_authority = deepcopy(dict(authority))
    basis_authority["authoritative_for_mastery_update"] = False
    basis_authority["requires_downstream_authority_revalidation"] = True
    basis_authority["server_revalidation_receipt"] = None
    basis["authority"] = basis_authority
    basis["mutates_student_model"] = False
    basis["mastery_update_authorized"] = False
    evidence_binding = source_entry.get(
        "authority_evidence_sha256", source_entry.get("evidence_fingerprint")
    )
    correction = instruction.get("correction")
    if (
        receipt.get("authority_id") != authority.get("authority_id")
        or receipt.get("gateway_verification_receipt_sha256")
        != authority.get("gateway_verification_receipt_sha256")
        or receipt.get("actor_principal_sha256") != authority.get("principal_sha256")
        or receipt.get("roles_sha256") != authority.get("roles_sha256")
        or receipt.get("review_item_id") != instruction.get("review_item_id")
        or receipt.get("review_item_version_sha256")
        != instruction.get("review_item_version_sha256")
        or receipt.get("evidence_id") != instruction.get("supersedes_evidence_id")
        or receipt.get("model_evidence_sha256") != evidence_binding
        or receipt.get("instruction_authority_basis_sha256")
        != _evidence_fingerprint(basis)
        or receipt.get("correction_sha256")
        != _evidence_fingerprint(dict(correction or {}))
        or receipt.get("target_kc_ids") != instruction.get("target_kc_ids")
        or receipt.get("rubric_id") != source_entry.get("rubric_id")
    ):
        raise StudentModelError(
            "authenticated correction authority bindings do not match evidence"
        )
    return receipt


def _reset_component_to_initial_prior(
    component: Mapping[str, Any], *, prior_strength: float
) -> dict[str, Any]:
    reset = deepcopy(dict(component))
    reset["dimensions"] = {
        dimension: _facet_row(
            float(component["dimensions"][dimension]["prior_p_mastery"]),
            prior_strength=prior_strength,
        )
        for dimension in MASTERY_DIMENSIONS
    }
    reset.update(
        {
            "p_mastery": _round(
                sum(row["p_mastery"] for row in reset["dimensions"].values())
                / len(MASTERY_DIMENSIONS)
            ),
            "uncertainty": _round(
                sum(row["uncertainty"] for row in reset["dimensions"].values())
                / len(MASTERY_DIMENSIONS)
            ),
            "evidence_count": 0,
            "positive_evidence": 0,
            "negative_evidence": 0,
            "last_signal": "not_observed",
            "last_round": 0,
            "last_observed_at": None,
            "last_time_basis": None,
            "last_evidence_id": None,
            "calibration": {
                "status": "descriptive_only_external_calibration_required",
                "sample_count": 0,
                "brier_score": None,
                "brier_sum": 0.0,
            },
            "evidence_ledger": [],
            "evidence_ledger_state": {
                "capacity": _MAX_EVIDENCE_PER_COMPONENT,
                "total_recorded_evidence": 0,
                "retained_evidence": 0,
                "active_evidence": 0,
                "superseded_evidence": 0,
                "complete_from_initial_prior": True,
            },
        }
    )
    return reset


def require_exact_student_model_adjudication_replay(
    model: Mapping[str, Any],
    *,
    target_kc_id: str,
    source_evidence_id: str,
) -> None:
    """Fail closed unless an abstention can be replayed exactly in-process.

    The durable review decision is terminal, while its student-model revision is
    checkpointed immediately afterwards.  Callers therefore use this pure
    preflight before committing an ``abstain`` decision.  It deliberately
    checks only replay availability; evidence and instruction authority remain
    the responsibility of the adjudication boundary.
    """

    candidate = deepcopy(dict(model))
    validate_student_model(candidate)
    kc_id = _text(target_kc_id, field="target_kc_id", maximum=84)
    if kc_id not in candidate["knowledge_components"]:
        raise StudentModelError("target knowledge component does not exist")
    evidence_id = _text(source_evidence_id, field="source_evidence_id", maximum=160)
    component = candidate["knowledge_components"][kc_id]
    source_entry = next(
        (
            entry
            for entry in component["evidence_ledger"]
            if entry.get("evidence_id") == evidence_id
        ),
        None,
    )
    if source_entry is None:
        raise StudentModelError(
            "exact adjudication replay requires retained source evidence"
        )
    if source_entry.get("lifecycle_status", "active") != "active":
        raise StudentModelError(
            "exact adjudication replay requires active source evidence"
        )
    if not _component_ledger_is_complete(component):
        raise StudentModelError(
            "exact adjudication replay requires a complete embedded evidence ledger"
        )
    if any(
        entry.get("lifecycle_status", "active") == "active"
        and "round_number" not in entry
        for entry in component["evidence_ledger"]
    ):
        raise StudentModelError(
            "exact adjudication replay requires round numbers for all active evidence"
        )
    if len(candidate.get("adjudication_revisions", [])) >= (
        _MAX_ADJUDICATION_REVISIONS
    ):
        raise StudentModelError(
            "exact adjudication replay requires revision-ledger capacity"
        )


def apply_student_model_adjudication(
    model: Mapping[str, Any],
    *,
    target_kc_id: str,
    source_evidence_id: str,
    instruction: Mapping[str, Any],
    authority_revalidator: (
        Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    """Purely supersede and replay one adjudicated KC when proof is sufficient.

    ``approve`` confirms an already-authoritative evidence record and returns an
    equal model.  ``abstain`` removes the evidence's effect by reconstructing
    the target KC from its immutable priors and replaying every still-active
    ledger entry.  A local-operator ``correct`` instruction is deliberately not
    authority to change mastery.  A gateway correction must carry a deployment
    service receipt which the caller revalidates against rubric authority.

    No branch subtracts a probability or applies a negative delta.  Missing or
    truncated replay input returns a fail-closed pending result with an unchanged
    model, because the 64-entry embedded ledger cannot support exact inference.
    """

    candidate = deepcopy(dict(model))
    validate_student_model(candidate)
    kc_id = _text(target_kc_id, field="target_kc_id", maximum=84)
    if kc_id not in candidate["knowledge_components"]:
        raise StudentModelError("target knowledge component does not exist")
    evidence_id = _text(source_evidence_id, field="source_evidence_id", maximum=160)
    decided, instruction_hash = _validate_adjudication_instruction(
        instruction,
        target_kc_id=kc_id,
        source_evidence_id=evidence_id,
    )
    component = candidate["knowledge_components"][kc_id]
    revisions = candidate.get("adjudication_revisions", [])
    prior_receipt = next(
        (
            receipt
            for receipt in revisions
            if receipt.get("instruction_sha256") == instruction_hash
        ),
        None,
    )
    if prior_receipt is not None:
        if (
            prior_receipt.get("target_kc_id") != kc_id
            or prior_receipt.get("source_evidence_id") != evidence_id
        ):
            raise StudentModelError(
                "adjudication revision receipt does not match the requested source"
            )
        return _adjudication_result(
            status="already_applied_no_change",
            model=candidate,
            instruction_sha256=instruction_hash,
            target_kc_id=kc_id,
            source_evidence_id=evidence_id,
            revision_receipt=prior_receipt,
        )
    source_entry = next(
        (
            entry
            for entry in component["evidence_ledger"]
            if entry.get("evidence_id") == evidence_id
        ),
        None,
    )
    if source_entry is None:
        return _adjudication_result(
            status="pending_external_ledger",
            model=candidate,
            instruction_sha256=instruction_hash,
            target_kc_id=kc_id,
            source_evidence_id=evidence_id,
            pending_reason="source_evidence_not_retained_in_embedded_ledger",
        )
    evidence_binding = source_entry.get(
        "authority_evidence_sha256", source_entry.get("evidence_fingerprint")
    )
    if decided["evidence_sha256"] != evidence_binding:
        raise StudentModelError(
            "adjudication instruction evidence hash does not match the source ledger"
        )

    operation = decided["operation"]
    if source_entry.get("lifecycle_status", "active") != "active":
        raise StudentModelError("source evidence was superseded by another instruction")
    if operation == "replay_original_assessment":
        return _adjudication_result(
            status="approved_no_change",
            model=candidate,
            instruction_sha256=instruction_hash,
            target_kc_id=kc_id,
            source_evidence_id=evidence_id,
        )

    authority_receipt: dict[str, Any] | None = None
    if operation == "supersede_and_replay":
        authority_receipt = _revalidate_authenticated_adjudication(
            decided,
            source_entry=source_entry,
            authority_revalidator=authority_revalidator,
        )
        if authority_receipt is None:
            # The local placeholder cannot replace rubric-bound evidence.  This
            # check precedes ledger availability so local behavior is stable.
            return _adjudication_result(
                status="pending_authority_revalidation",
                model=candidate,
                instruction_sha256=instruction_hash,
                target_kc_id=kc_id,
                source_evidence_id=evidence_id,
                pending_reason=(
                    "authenticated_teacher_and_rubric_authority_receipt_required"
                ),
            )

    if not _component_ledger_is_complete(component):
        return _adjudication_result(
            status="pending_external_ledger",
            model=candidate,
            instruction_sha256=instruction_hash,
            target_kc_id=kc_id,
            source_evidence_id=evidence_id,
            pending_reason="embedded_evidence_ledger_truncated_before_initial_prior",
        )
    if any(
        entry.get("lifecycle_status", "active") == "active"
        and "round_number" not in entry
        for entry in component["evidence_ledger"]
    ):
        return _adjudication_result(
            status="pending_external_ledger",
            model=candidate,
            instruction_sha256=instruction_hash,
            target_kc_id=kc_id,
            source_evidence_id=evidence_id,
            pending_reason="legacy_embedded_evidence_lacks_replay_round_number",
        )

    if operation not in {"do_not_replay", "supersede_and_replay"}:
        raise StudentModelError("adjudication operation cannot revise student mastery")
    if len(revisions) >= _MAX_ADJUDICATION_REVISIONS:
        return _adjudication_result(
            status="pending_external_ledger",
            model=candidate,
            instruction_sha256=instruction_hash,
            target_kc_id=kc_id,
            source_evidence_id=evidence_id,
            pending_reason="embedded_adjudication_revision_ledger_capacity_reached",
        )
    if (
        operation == "supersede_and_replay"
        and len(component["evidence_ledger"]) >= _MAX_EVIDENCE_PER_COMPONENT
    ):
        return _adjudication_result(
            status="pending_external_ledger",
            model=candidate,
            instruction_sha256=instruction_hash,
            target_kc_id=kc_id,
            source_evidence_id=evidence_id,
            pending_reason="embedded_evidence_ledger_lacks_corrected_entry_capacity",
        )

    before_component = deepcopy(component)
    other_components = {
        other_id: deepcopy(other)
        for other_id, other in candidate["knowledge_components"].items()
        if other_id != kc_id
    }
    revision_number = len(revisions) + 1
    revision_id = (
        "smrev_"
        + _evidence_fingerprint(
            {
                "instruction_sha256": instruction_hash,
                "target_kc_id": kc_id,
                "source_evidence_id": evidence_id,
                "revision_number": revision_number,
            }
        )[:24]
    )
    original_ledger = deepcopy(component["evidence_ledger"])
    replay_model = deepcopy(candidate)
    replay_model["knowledge_components"][kc_id] = _reset_component_to_initial_prior(
        component, prior_strength=float(candidate["prior_strength"])
    )
    _recompute(replay_model)

    corrected_evidence_id: str | None = None
    if authority_receipt is not None:
        corrected_evidence_id = (
            "adjevidence_"
            + _evidence_fingerprint(
                {
                    "instruction_sha256": instruction_hash,
                    "source_evidence_id": evidence_id,
                    "authority_receipt_sha256": authority_receipt["receipt_sha256"],
                }
            )[:24]
        )
    replay_entries = [
        entry
        for entry in original_ledger
        if entry.get("lifecycle_status", "active") == "active"
        and (
            entry.get("evidence_id") != evidence_id or corrected_evidence_id is not None
        )
    ]
    correction = decided.get("correction")
    for entry in replay_entries:
        corrected = entry.get("evidence_id") == evidence_id
        bounded_correction = (
            correction if corrected and isinstance(correction, Mapping) else {}
        )
        replay_model = update_student_model(
            replay_model,
            signal=str(bounded_correction.get("signal", entry["signal"])),
            confidence=float(entry["confidence"]),
            focus_dimension=str(
                bounded_correction.get("focus_dimension", entry["focus_dimension"])
            ),
            knowledge_component_ids=[kc_id],
            answer_alignment=str(
                bounded_correction.get("answer_alignment", entry["answer_alignment"])
            ),
            needs_human_review=False,
            assessment_eligible=True,
            authoritative=True,
            round_number=int(entry["round_number"]),
            evidence_id=(
                str(corrected_evidence_id) if corrected else str(entry["evidence_id"])
            ),
            item_id=str(entry["item_id"]),
            question_id=str(entry["question_id"]),
            rubric_id=str(entry["rubric_id"]),
            difficulty=entry.get("difficulty"),
            discrimination=entry.get("discrimination"),
            observed_at=str(entry["observed_at"]),
            time_basis=str(entry["time_basis"]),
            source=(
                "authenticated_teacher_adjudication_correction"
                if corrected
                else str(entry["source"])
            ),
            authority_evidence_sha256=(
                str(authority_receipt["receipt_sha256"])
                if corrected and authority_receipt is not None
                else entry.get("authority_evidence_sha256")
            ),
        )

    replayed_by_id = {
        row["evidence_id"]: deepcopy(row)
        for row in replay_model["knowledge_components"][kc_id]["evidence_ledger"]
    }
    merged_ledger: list[dict[str, Any]] = []
    for original in original_ledger:
        original_id = str(original["evidence_id"])
        if original_id == evidence_id:
            superseded = deepcopy(original)
            superseded["lifecycle_status"] = "superseded"
            superseded["superseded_by_revision_id"] = revision_id
            merged_ledger.append(superseded)
        elif original.get("lifecycle_status", "active") == "superseded":
            merged_ledger.append(deepcopy(original))
        else:
            replayed = replayed_by_id.get(original_id)
            if replayed is None:
                raise StudentModelError(
                    "exact adjudication replay omitted an active evidence record"
                )
            merged_ledger.append(replayed)
    if corrected_evidence_id is not None:
        corrected_entry = replayed_by_id.get(corrected_evidence_id)
        if corrected_entry is None:  # pragma: no cover - update contract guard.
            raise StudentModelError("corrected evidence replay was not retained")
        merged_ledger.append(corrected_entry)

    revised_component = replay_model["knowledge_components"][kc_id]
    revised_component["evidence_ledger"] = merged_ledger
    superseded_count = sum(
        entry.get("lifecycle_status", "active") == "superseded"
        for entry in merged_ledger
    )
    revised_component["evidence_ledger_state"] = {
        "capacity": _MAX_EVIDENCE_PER_COMPONENT,
        "total_recorded_evidence": len(merged_ledger),
        "retained_evidence": len(merged_ledger),
        "active_evidence": len(merged_ledger) - superseded_count,
        "superseded_evidence": superseded_count,
        "complete_from_initial_prior": True,
    }
    _recompute(replay_model)
    source_round = int(source_entry.get("round_number", 0))
    is_authenticated_correction = authority_receipt is not None
    replay_model["last_update"] = {
        "round": source_round,
        "signal": (
            str(decided["correction"].get("signal", source_entry["signal"]))
            if is_authenticated_correction
            else "not_observed"
        ),
        "focus_dimension": (
            str(
                decided["correction"].get(
                    "focus_dimension", source_entry["focus_dimension"]
                )
            )
            if is_authenticated_correction
            else str(source_entry["focus_dimension"])
        ),
        "knowledge_component_ids": [kc_id],
        "evidence_id": corrected_evidence_id or evidence_id,
        "item_id": str(source_entry["item_id"]),
        "rubric_id": str(source_entry["rubric_id"]),
        "source": (
            "authenticated_teacher_adjudication_correction"
            if is_authenticated_correction
            else "teacher_adjudication_supersede_replay"
        ),
        "update_applied": True,
        "needs_human_review": False,
        "reason": (
            "authenticated_teacher_correction_revalidated_and_replayed"
            if is_authenticated_correction
            else "adjudicated_evidence_superseded_and_active_ledger_replayed"
        ),
    }
    receipt: dict[str, Any] = {
        "schema": STUDENT_MODEL_ADJUDICATION_REVISION_SCHEMA,
        "revision_number": revision_number,
        "revision_id": revision_id,
        "decision_kind": "correct" if is_authenticated_correction else "abstain",
        "operation": "supersede_and_replay",
        "target_kc_id": kc_id,
        "source_evidence_id": evidence_id,
        "source_evidence_fingerprint": str(source_entry["evidence_fingerprint"]),
        "instruction_sha256": instruction_hash,
        "before_component_sha256": _evidence_fingerprint(before_component),
        "after_component_sha256": _evidence_fingerprint(revised_component),
        "active_evidence_replayed": len(replay_entries),
        "superseded_evidence_count": superseded_count,
        "ledger_complete_from_initial_prior": True,
        "negative_delta_applied": False,
        "previous_revision_sha256": (
            revisions[-1]["revision_sha256"] if revisions else None
        ),
    }
    if authority_receipt is not None:
        receipt["authority_revalidation_receipt_sha256"] = authority_receipt[
            "receipt_sha256"
        ]
    receipt["revision_sha256"] = _evidence_fingerprint(receipt)
    replay_model.setdefault("adjudication_revisions", []).append(receipt)
    replay_model["recommended_focus"] = recommend_focus(replay_model)
    validate_student_model(replay_model)
    for other_id, before in other_components.items():
        if replay_model["knowledge_components"][other_id] != before:
            raise StudentModelError(
                "adjudication replay changed a non-target knowledge component"
            )
    return _adjudication_result(
        status="applied_supersede_replay",
        model=replay_model,
        instruction_sha256=instruction_hash,
        target_kc_id=kc_id,
        source_evidence_id=evidence_id,
        revision_receipt=receipt,
    )


# Explicit alias retained for integrations that name the state transition.
supersede_and_replay_student_model_evidence = apply_student_model_adjudication
adjudicate_student_model_evidence = apply_student_model_adjudication
apply_adjudication_to_student_model = apply_student_model_adjudication


def update_student_model(
    model: Mapping[str, Any],
    *,
    signal: str,
    confidence: float,
    focus_dimension: str,
    knowledge_component_ids: Sequence[str] | None = None,
    answer_alignment: str = "not_applicable",
    needs_human_review: bool = False,
    assessment_eligible: bool = False,
    authoritative: bool = False,
    round_number: int = 0,
    evidence_id: str | None = None,
    item_id: str | None = None,
    question_id: str | None = None,
    rubric_id: str | None = None,
    difficulty: float | None = None,
    discrimination: float | None = None,
    observed_at: str | None = None,
    time_basis: str = "wall_clock_utc",
    source: str = "validated_turn_diagnosis",
    authority_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    """Apply one rubric-bound, slip/guess-aware observation to exactly one KC."""

    candidate = deepcopy(dict(model))
    validate_student_model(candidate)
    if signal not in _SIGNALS:
        raise StudentModelError("signal is invalid")
    if focus_dimension not in MASTERY_DIMENSIONS:
        raise StudentModelError("focus_dimension is invalid")
    confidence = _probability(confidence, field="confidence")
    if answer_alignment not in _ALIGNMENTS:
        raise StudentModelError("answer_alignment is invalid")
    if not isinstance(needs_human_review, bool):
        raise StudentModelError("needs_human_review must be a boolean")
    if not isinstance(assessment_eligible, bool) or not isinstance(authoritative, bool):
        raise StudentModelError("evidence authority flags must be booleans")
    component_ids = [str(item) for item in (knowledge_component_ids or [])]
    unknown = set(component_ids) - set(candidate["knowledge_components"])
    if unknown:
        raise StudentModelError(f"unknown knowledge component IDs: {sorted(unknown)}")
    if len(component_ids) != 1:
        return _last_update_without_evidence(
            candidate,
            signal=signal,
            focus_dimension=focus_dimension,
            component_ids=[],
            round_number=round_number,
            evidence_id=evidence_id,
            item_id=item_id,
            rubric_id=rubric_id,
            source=source,
            needs_human_review=True,
            reason="exactly_one_knowledge_component_required",
        )
    if signal not in _UPDATABLE_SIGNALS:
        return _last_update_without_evidence(
            candidate,
            signal=signal,
            focus_dimension=focus_dimension,
            component_ids=component_ids,
            round_number=round_number,
            evidence_id=evidence_id,
            item_id=item_id,
            rubric_id=rubric_id,
            source=source,
            needs_human_review=needs_human_review,
            reason="signal_contains_no_assessment_outcome",
        )
    alignment_valid = (
        signal in {"correct", "partial"}
        and answer_alignment in {"aligned", "partially_aligned"}
    ) or (signal == "misconception" and answer_alignment == "contradicted")
    if (
        not assessment_eligible
        or not authoritative
        or confidence <= 0.0
        or needs_human_review
        or not alignment_valid
    ):
        return _last_update_without_evidence(
            candidate,
            signal=signal,
            focus_dimension=focus_dimension,
            component_ids=component_ids,
            round_number=round_number,
            evidence_id=evidence_id,
            item_id=item_id,
            rubric_id=rubric_id,
            source=source,
            needs_human_review=needs_human_review or not authoritative,
            reason="assessment_evidence_not_authoritative_or_aligned",
        )
    if (
        not evidence_id
        or not item_id
        or not question_id
        or not rubric_id
        or not observed_at
    ):
        return _last_update_without_evidence(
            candidate,
            signal=signal,
            focus_dimension=focus_dimension,
            component_ids=component_ids,
            round_number=round_number,
            evidence_id=evidence_id,
            item_id=item_id,
            rubric_id=rubric_id,
            source=source,
            needs_human_review=True,
            reason="complete_item_question_rubric_provenance_required",
        )
    evidence_id = _text(evidence_id, field="evidence_id", maximum=160)
    item_id = _text(item_id, field="item_id", maximum=160)
    question_id = _text(question_id, field="question_id", maximum=160)
    rubric_id = _text(rubric_id, field="rubric_id", maximum=160)
    source = _text(source, field="source", maximum=160)
    if authority_evidence_sha256 is not None and (
        not isinstance(authority_evidence_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", authority_evidence_sha256) is None
    ):
        raise StudentModelError("authority_evidence_sha256 must be a lowercase SHA-256")
    if time_basis not in _TIME_BASES:
        raise StudentModelError(f"time_basis must be one of {sorted(_TIME_BASES)}")
    observed = _parse_timestamp(str(observed_at), field="observed_at")
    difficulty_value = (
        0.5 if difficulty is None else _probability(difficulty, field="difficulty")
    )
    discrimination_value = (
        0.75
        if discrimination is None
        else _probability(discrimination, field="discrimination")
    )
    kc_id = component_ids[0]
    component = candidate["knowledge_components"][kc_id]
    evidence_material = {
        "evidence_id": evidence_id,
        "item_id": item_id,
        "question_id": question_id,
        "rubric_id": rubric_id,
        "knowledge_component_id": kc_id,
        "focus_dimension": focus_dimension,
        "signal": signal,
        "confidence": round(confidence, 6),
        "answer_alignment": answer_alignment,
        "difficulty": difficulty,
        "discrimination": discrimination,
        "observed_at": str(observed_at),
        "time_basis": time_basis,
        "source": source,
        "assessment_eligible": True,
        "authoritative": True,
        "round_number": max(0, int(round_number)),
    }
    if authority_evidence_sha256 is not None:
        evidence_material["authority_evidence_sha256"] = authority_evidence_sha256
    fingerprint = _evidence_fingerprint(evidence_material)
    for owner in candidate["knowledge_components"].values():
        for entry in owner["evidence_ledger"]:
            if entry["evidence_id"] != evidence_id:
                continue
            if entry["evidence_fingerprint"] != fingerprint:
                raise StudentModelError(
                    "evidence_id was replayed with different assessment content"
                )
            # Strict replay idempotence: not even last_update changes.
            return candidate

    facet = component["dimensions"][focus_dimension]
    prior = float(facet["p_mastery"])
    last_observed_at = facet.get("last_observed_at")
    last_time_basis = facet.get("last_time_basis")
    elapsed_days = 0.0
    forgetting_applied = False
    if (
        last_observed_at is not None
        and last_time_basis == "wall_clock_utc"
        and time_basis == "wall_clock_utc"
    ):
        previous = _parse_timestamp(str(last_observed_at), field="last_observed_at")
        if observed < previous:
            raise StudentModelError(
                "observed_at must not precede the KC's last observation"
            )
        elapsed_days = (observed - previous).total_seconds() / 86_400.0
        forgetting_applied = elapsed_days > 0
    forgetting = 1.0 - math.exp(
        -float(component["bkt"]["forgetting_rate_per_day"]) * elapsed_days
    )
    baseline = float(facet["prior_p_mastery"])
    effective_prior = baseline + (prior - baseline) * (1.0 - forgetting)
    base_slip = float(component["bkt"]["p_slip"])
    base_guess = float(component["bkt"]["p_guess"])
    slip = _clamp(base_slip + 0.30 * (difficulty_value - 0.5), 0.02, 0.45)
    guess = _clamp(base_guess + 0.25 * (0.5 - difficulty_value), 0.02, 0.45)
    correct_denominator = (
        effective_prior * (1.0 - slip) + (1.0 - effective_prior) * guess
    )
    wrong_denominator = effective_prior * slip + (1.0 - effective_prior) * (1.0 - guess)
    posterior_correct = effective_prior * (1.0 - slip) / max(correct_denominator, 1e-12)
    posterior_wrong = effective_prior * slip / max(wrong_denominator, 1e-12)
    outcome = {"correct": 1.0, "partial": 0.55, "misconception": 0.0}[signal]
    posterior = posterior_wrong + outcome * (posterior_correct - posterior_wrong)
    evidence_weight = confidence * discrimination_value
    blended = effective_prior + evidence_weight * (posterior - effective_prior)
    learning_rate = float(component["bkt"]["p_learn"])
    if signal == "partial":
        learning_rate *= 0.5
    elif signal == "misconception":
        learning_rate = 0.0
    after = blended + (1.0 - blended) * learning_rate
    after = _clamp(after, 0.001, 0.999)

    facet["p_mastery"] = _round(after)
    facet["evidence_count"] += 1
    if outcome >= 0.5:
        facet["positive_evidence"] += 1
    else:
        facet["negative_evidence"] += 1
    facet["last_signal"] = signal
    facet["last_round"] = max(0, int(round_number))
    facet["last_evidence_id"] = evidence_id
    facet["last_observed_at"] = str(observed_at)
    facet["last_time_basis"] = time_basis
    information = float(facet["evidence_count"]) * max(0.1, discrimination_value)
    entropy = 4.0 * after * (1.0 - after)
    facet["uncertainty"] = _round(max(0.04, entropy / math.sqrt(1.0 + information)))

    calibration = component["calibration"]
    predicted_correct = correct_denominator
    calibration["sample_count"] += 1
    calibration["brier_sum"] = round(
        float(calibration["brier_sum"]) + (predicted_correct - outcome) ** 2,
        8,
    )
    calibration["brier_score"] = _round(
        float(calibration["brier_sum"]) / calibration["sample_count"]
    )
    entry = {
        **evidence_material,
        "difficulty": difficulty,
        "discrimination": discrimination,
        "p_mastery_before": _round(prior),
        "p_mastery_after": _round(after),
        "elapsed_days": round(elapsed_days, 6),
        "forgetting_applied": forgetting_applied,
        "effective_slip": _round(slip),
        "effective_guess": _round(guess),
        "evidence_fingerprint": fingerprint,
        "lifecycle_status": "active",
        "superseded_by_revision_id": None,
    }
    ledger_state = component.get("evidence_ledger_state")
    if isinstance(ledger_state, Mapping):
        total_recorded = int(ledger_state["total_recorded_evidence"]) + 1
    else:
        # Legacy v2 models did not persist an explicit truncation counter.  A
        # component count larger than its retained ledger proves that exact
        # replay is already impossible; preserve that fact when upgrading.
        total_recorded = (
            max(
                int(component.get("evidence_count", 0)),
                len(component["evidence_ledger"]),
            )
            + 1
        )
    component["evidence_ledger"].append(entry)
    del component["evidence_ledger"][:-_MAX_EVIDENCE_PER_COMPONENT]
    retained_ledger = component["evidence_ledger"]
    component["evidence_ledger_state"] = {
        "capacity": _MAX_EVIDENCE_PER_COMPONENT,
        "total_recorded_evidence": total_recorded,
        "retained_evidence": len(retained_ledger),
        "active_evidence": sum(
            row.get("lifecycle_status", "active") == "active" for row in retained_ledger
        ),
        "superseded_evidence": sum(
            row.get("lifecycle_status", "active") == "superseded"
            for row in retained_ledger
        ),
        "complete_from_initial_prior": total_recorded == len(retained_ledger),
    }
    component["last_signal"] = signal
    component["last_round"] = max(0, int(round_number))
    component["last_observed_at"] = str(observed_at)
    component["last_time_basis"] = time_basis
    component["last_evidence_id"] = evidence_id
    _recompute(candidate)
    candidate["last_update"] = {
        "round": max(0, int(round_number)),
        "signal": signal,
        "focus_dimension": focus_dimension,
        "knowledge_component_ids": [kc_id],
        "evidence_id": evidence_id,
        "item_id": item_id,
        "rubric_id": rubric_id,
        "source": source,
        "update_applied": True,
        "needs_human_review": False,
        "reason": "authoritative_rubric_bound_evidence_applied",
    }
    validate_student_model(candidate)
    return candidate


def recommend_focus(
    model: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
    *,
    active_misconception: bool = False,
) -> dict[str, Any]:
    """Choose a KC and facet using mastery deficit plus uncertainty."""

    validate_student_model(model)
    thresholds = thresholds or {}
    candidates: list[tuple[float, str, str]] = []
    for kc_id, component in model["knowledge_components"].items():
        for dimension in MASTERY_DIMENSIONS:
            threshold = _clamp(float(thresholds.get(dimension, 0.75)))
            row = component["dimensions"][dimension]
            deficit = max(0.0, threshold - float(row["p_mastery"]))
            score = deficit + 0.35 * float(row["uncertainty"])
            if active_misconception and dimension == "conceptual":
                score += 1.0
            candidates.append((score, kc_id, dimension))
    _, kc_id, dimension = max(
        candidates,
        key=lambda item: (
            item[0],
            -MASTERY_DIMENSIONS.index(item[2]),
            # Deterministic ID tie-break, independent of list position.
            item[1],
        ),
    )
    component = model["knowledge_components"][kc_id]
    row = component["dimensions"][dimension]
    refs = [str(row["last_evidence_id"])] if row.get("last_evidence_id") else []
    return {
        "dimension": dimension,
        "knowledge_component_id": kc_id,
        "knowledge_component_label": component["label"],
        "reason": (
            "存在活跃误解，优先核验对应知识组件的概念边界"
            if active_misconception
            else "按每个知识组件的达标缺口与不确定性排序"
        ),
        "confidence": _round(1.0 - float(row["uncertainty"])),
        "evidence_refs": refs[:8],
    }


def record_legacy_state_compatibility(
    model: Mapping[str, Any], legacy_mastery: Mapping[str, Any]
) -> dict[str, Any]:
    """Record (without reconciling) legacy decision-state divergence."""

    candidate = deepcopy(dict(model))
    validate_student_model(candidate)
    if not isinstance(legacy_mastery, Mapping):
        raise StudentModelError("legacy_mastery must be an object")
    candidate["compatibility"]["legacy_projection_snapshot"] = {
        dimension: _probability(
            legacy_mastery.get(dimension), field=f"legacy_mastery.{dimension}"
        )
        for dimension in MASTERY_DIMENSIONS
    }
    _recompute(candidate)
    validate_student_model(candidate)
    return candidate


def synchronize_runtime_state_from_kc_model(
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Opt a model into KC-authoritative runtime decisions.

    This is deliberately explicit instead of a migration side effect.  A
    deterministic legacy caller can keep the old decision state, while a live
    caller uses this receipt before copying :func:`project_legacy_mastery` into
    ``student_state.knowledge_mastery``.  Both booleans are switched together
    and the persisted divergence must be exactly zero.
    """

    candidate = deepcopy(dict(model))
    validate_student_model(candidate)
    projection = {
        dimension: float(candidate["dimensions"][dimension]["p_mastery"])
        for dimension in MASTERY_DIMENSIONS
    }
    compatibility = candidate["compatibility"]
    compatibility["legacy_success_gate_uses_kc_model"] = True
    compatibility["legacy_runtime_state_synchronized_from_kc_model"] = True
    compatibility["legacy_projection_snapshot"] = dict(projection)
    compatibility["kc_projection_snapshot"] = dict(projection)
    compatibility["absolute_divergence"] = {
        dimension: 0.0 for dimension in MASTERY_DIMENSIONS
    }
    validate_student_model(candidate)
    return candidate


def student_model_mastery_readiness(
    model: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a conservative, evidence-bound automatic-success receipt.

    Unobserved priors never satisfy this gate.  Every target KC needs at least
    one active authoritative receipt, every required mastery dimension needs
    at least one observed KC facet, and only those observed facets are compared
    with the goal threshold.  This avoids both legacy false positives and the
    opposite error of averaging unobserved KC/facet priors into a mastery
    decision.
    """

    validate_student_model(model)
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(
        MASTERY_DIMENSIONS
    ):
        raise StudentModelError("success thresholds are incomplete")
    normalized_thresholds = {
        dimension: _probability(
            thresholds.get(dimension), field=f"success_thresholds.{dimension}"
        )
        for dimension in MASTERY_DIMENSIONS
    }
    compatibility = model["compatibility"]
    runtime_authoritative = bool(
        compatibility["legacy_success_gate_uses_kc_model"]
        and compatibility["legacy_runtime_state_synchronized_from_kc_model"]
    )

    component_rows: list[dict[str, Any]] = []
    for kc_id in sorted(model["knowledge_components"]):
        component = model["knowledge_components"][kc_id]
        active_entries = [
            entry
            for entry in component["evidence_ledger"]
            if entry.get("lifecycle_status", "active") == "active"
        ]
        observed_dimensions = sorted(
            {str(entry["focus_dimension"]) for entry in active_entries},
            key=MASTERY_DIMENSIONS.index,
        )
        below_threshold = [
            dimension
            for dimension in observed_dimensions
            if float(component["dimensions"][dimension]["p_mastery"])
            < normalized_thresholds[dimension]
        ]
        component_rows.append(
            {
                "knowledge_component_id": kc_id,
                "active_authoritative_evidence": len(active_entries),
                "observed_dimensions": observed_dimensions,
                "below_threshold_dimensions": below_threshold,
                "ready": bool(active_entries) and not below_threshold,
            }
        )

    dimension_rows: dict[str, dict[str, Any]] = {}
    for dimension in MASTERY_DIMENSIONS:
        observed = [
            float(component["dimensions"][dimension]["p_mastery"])
            for component in model["knowledge_components"].values()
            if int(component["dimensions"][dimension]["evidence_count"]) > 0
        ]
        observed_mastery = _round(sum(observed) / len(observed)) if observed else None
        dimension_rows[dimension] = {
            "threshold": normalized_thresholds[dimension],
            "observed_component_count": len(observed),
            "observed_mastery": observed_mastery,
            "ready": bool(observed)
            and float(observed_mastery) >= normalized_thresholds[dimension],
        }

    reasons: list[str] = []
    if not runtime_authoritative:
        reasons.append("kc_model_not_runtime_authority")
    missing_components = [
        row["knowledge_component_id"]
        for row in component_rows
        if not row["active_authoritative_evidence"]
    ]
    if missing_components:
        reasons.append("target_knowledge_components_without_authoritative_evidence")
    if any(row["below_threshold_dimensions"] for row in component_rows):
        reasons.append("observed_knowledge_component_facet_below_threshold")
    if any(not row["ready"] for row in dimension_rows.values()):
        reasons.append("required_mastery_dimension_not_observed_or_below_threshold")
    # ``last_update`` is an audit of the estimator's most recent attempt, not
    # necessarily the learner's current turn: an unbound/non-authoritative
    # observation intentionally leaves the last accepted KC evidence in place.
    # Success is already gated by the current session diagnosis and only active
    # authoritative ledger entries are counted above, so a stale abstention must
    # not veto a later server-validated summary forever.
    if any(
        component.get("evidence_ledger_state", {}).get("complete_from_initial_prior")
        is not True
        for component in model["knowledge_components"].values()
    ):
        reasons.append("embedded_evidence_ledger_is_incomplete")

    return {
        "schema": STUDENT_MODEL_READINESS_SCHEMA,
        "eligible": not reasons,
        "source": "active_authoritative_per_kc_evidence_only",
        "runtime_authoritative": runtime_authoritative,
        "component_coverage": component_rows,
        "dimensions": dimension_rows,
        "reasons": reasons,
        "claim_boundary": {
            "unobserved_prior_counts_as_mastery_evidence": False,
            "model_confidence_alone_is_mastery_evidence": False,
            "external_calibration_established": False,
        },
    }


def project_legacy_mastery(model: Mapping[str, Any]) -> dict[str, float]:
    """Return the historical four-number view derived from KC facet states."""

    validate_student_model(model)
    return {
        dimension: float(model["dimensions"][dimension]["p_mastery"])
        for dimension in MASTERY_DIMENSIONS
    }


def project_student_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Return UI-safe fields while retaining provenance and evidence receipts."""

    validate_student_model(model)
    result = deepcopy(dict(model))
    for row in result["dimensions"].values():
        row.pop("alpha", None)
        row.pop("beta", None)
    for component in result["knowledge_components"].values():
        component.pop("bkt", None)
        calibration = component.get("calibration")
        if isinstance(calibration, dict):
            calibration.pop("brier_sum", None)
    result["source"] = "deterministic_per_kc_bkt_estimator"
    return result
