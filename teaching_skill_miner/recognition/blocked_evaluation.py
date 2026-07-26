"""Conservative blocked descriptive evaluation for the single-site DIPSER study.

The routines in this module intentionally do not make inferential or deployment
claims.  They produce out-of-fold predictions for three fixed blocking designs
and disclose every fold that cannot be evaluated.  Preprocessing is fitted only
on each training fold, and the logistic-regression regularization is fixed at
``C=1.0`` rather than selected on these data.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


EXPECTED_COHORT_COUNT = 3
EXPECTED_ACTIVITY_COUNT = 9
FIXED_LOGISTIC_C = 1.0
MODALITY_ORDER = ("visual", "sensor", "fusion")


class BlockedEvaluationError(ValueError):
    """Raised when records or feature rows cannot support the fixed protocol."""


def _dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "install optional dependencies with `pip install -e '.[recognition]'`"
        ) from exc
    return np, LogisticRegression, StandardScaler, accuracy_score, f1_score, ConvergenceWarning


def _nonempty_identifier(record: Mapping[str, Any], field: str, sample_id: str) -> str:
    value = str(record.get(field, "")).strip()
    if not value:
        raise BlockedEvaluationError(f"record {sample_id} lacks {field}")
    return value


def _activity_id(record: Mapping[str, Any], sample_id: str) -> str:
    direct = str(record.get("activity_id", "")).strip()
    source = record.get("source")
    nested = ""
    if isinstance(source, Mapping):
        nested = str(source.get("experiment_id", "")).strip()
    if direct and nested and direct != nested:
        raise BlockedEvaluationError(
            f"record {sample_id} has conflicting activity_id and source.experiment_id"
        )
    activity = direct or nested
    if not activity:
        raise BlockedEvaluationError(
            f"record {sample_id} lacks activity_id or source.experiment_id"
        )
    return activity


def _validate_inputs(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    *,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    np, _, _, _, _, _ = _dependencies()
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise BlockedEvaluationError("records must be a non-empty sequence")

    normalized_records: list[Mapping[str, Any]] = []
    manifest_sample_ids: list[str] = []
    labels: list[int] = []
    cohorts: list[str] = []
    activities: list[str] = []
    sites: list[str] = []
    sessions: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise BlockedEvaluationError(f"record {index} is not an object")
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise BlockedEvaluationError(f"record {index} lacks sample_id")
        manifest_sample_ids.append(sample_id)
        try:
            label = int(record["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BlockedEvaluationError(f"record {sample_id} lacks an integer label") from exc
        if label < 0:
            raise BlockedEvaluationError(f"record {sample_id} has a negative label")
        labels.append(label)
        cohorts.append(_nonempty_identifier(record, "cohort_id", sample_id))
        activities.append(_activity_id(record, sample_id))
        sites.append(_nonempty_identifier(record, "site_id", sample_id))
        sessions.append(_nonempty_identifier(record, "session_id", sample_id))
        normalized_records.append(record)

    if len(set(manifest_sample_ids)) != len(manifest_sample_ids):
        raise BlockedEvaluationError("manifest sample_id values must be unique")
    feature_sample_ids = [str(value) for value in sample_ids]
    if feature_sample_ids != manifest_sample_ids:
        mismatch = next(
            (
                index
                for index, (observed, expected) in enumerate(
                    zip(feature_sample_ids, manifest_sample_ids)
                )
                if observed != expected
            ),
            min(len(feature_sample_ids), len(manifest_sample_ids)),
        )
        raise BlockedEvaluationError(
            "sample_ids must exactly match manifest record order; "
            f"first mismatch is at row {mismatch}"
        )

    unique_cohorts = sorted(set(cohorts))
    unique_activities = sorted(set(activities))
    unique_sites = sorted(set(sites))
    if len(unique_cohorts) != EXPECTED_COHORT_COUNT:
        raise BlockedEvaluationError(
            f"DIPSER blocked evaluation requires exactly {EXPECTED_COHORT_COUNT} cohorts; "
            f"found {len(unique_cohorts)}"
        )
    if len(unique_activities) != EXPECTED_ACTIVITY_COUNT:
        raise BlockedEvaluationError(
            f"DIPSER blocked evaluation requires exactly {EXPECTED_ACTIVITY_COUNT} activities; "
            f"found {len(unique_activities)}"
        )
    if len(unique_sites) != 1:
        raise BlockedEvaluationError(
            "this evaluator is restricted to one documented site; "
            f"found {len(unique_sites)} site IDs"
        )

    session_design: dict[str, tuple[str, str, str]] = {}
    cell_sessions: dict[tuple[str, str], set[str]] = {}
    for session, cohort, activity, site in zip(
        sessions, cohorts, activities, sites
    ):
        design = (cohort, activity, site)
        previous = session_design.setdefault(session, design)
        if previous != design:
            raise BlockedEvaluationError(
                f"session_id {session} maps to multiple cohort/activity/site cells"
            )
        cell_sessions.setdefault((cohort, activity), set()).add(session)
    ambiguous_cells = {
        cell: sorted(values)
        for cell, values in cell_sessions.items()
        if len(values) != 1
    }
    if ambiguous_cells:
        raise BlockedEvaluationError(
            "each observed cohort/activity cell must map to exactly one session_id; "
            f"ambiguous cells: {ambiguous_cells}"
        )

    global_classes = sorted(set(labels))
    if len(global_classes) < 2:
        raise BlockedEvaluationError("at least two global classes are required")

    prepared: dict[str, Any] = {}
    for name, values in (("visual", visual_features), ("sensor", sensor_features)):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(records) or matrix.shape[1] < 1:
            raise BlockedEvaluationError(
                f"{name} features must have shape ({len(records)}, positive_dimension)"
            )
        if not np.isfinite(matrix).all():
            raise BlockedEvaluationError(f"{name} features contain NaN or infinity")
        prepared[name] = matrix
    prepared["fusion"] = np.concatenate([prepared["visual"], prepared["sensor"]], axis=1)

    return {
        "records": normalized_records,
        "sample_ids": manifest_sample_ids,
        "labels": np.asarray(labels, dtype=int),
        "cohorts": np.asarray(cohorts, dtype=object),
        "activities": np.asarray(activities, dtype=object),
        "sites": np.asarray(sites, dtype=object),
        "sessions": np.asarray(sessions, dtype=object),
        "classes": global_classes,
        "unique_cohorts": unique_cohorts,
        "unique_activities": unique_activities,
        "unique_sites": unique_sites,
        "unique_sessions": sorted(session_design),
        "complete_session_grid": len(cell_sessions)
        == EXPECTED_COHORT_COUNT * EXPECTED_ACTIVITY_COUNT,
        "features": prepared,
        "feature_dimensions": {
            "visual": int(prepared["visual"].shape[1]),
            "sensor": int(prepared["sensor"].shape[1]),
            "fusion": int(prepared["fusion"].shape[1]),
        },
    }


def _class_counts(labels: Any, indices: Any) -> dict[str, int]:
    return {
        str(int(label)): int(count)
        for label, count in sorted(Counter(int(labels[index]) for index in indices).items())
    }


def _metrics(y_true: Any, y_pred: Any, classes: Sequence[int]) -> dict[str, Any]:
    _, _, _, accuracy_score, f1_score, _ = _dependencies()
    return {
        "sample_count": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=list(classes),
                    average="macro",
                    zero_division=0,
                )
            ),
            6,
        ),
    }


def _fit_predict_fold(
    train_x: Any,
    train_y: Any,
    test_x: Any,
    *,
    logistic_c: float,
    max_iter: int,
    random_state: int,
) -> tuple[Any, bool]:
    _, LogisticRegression, StandardScaler, _, _, ConvergenceWarning = _dependencies()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    model = LogisticRegression(
        C=float(logistic_c),
        class_weight=None,
        max_iter=int(max_iter),
        random_state=int(random_state),
        solver="lbfgs",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_scaled, train_y)
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    return model.predict(scaler.transform(test_x)), convergence_warning


def _fold_specifications(data: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    np, _, _, _, _, _ = _dependencies()
    cohorts = data["cohorts"]
    activities = data["activities"]
    all_indices = np.arange(len(cohorts), dtype=int)
    specifications: dict[str, list[dict[str, Any]]] = {
        "leave_one_cohort_out": [],
        "leave_one_activity_out": [],
        "double_blocked_cohort_activity": [],
    }
    for cohort in data["unique_cohorts"]:
        test_mask = cohorts == cohort
        specifications["leave_one_cohort_out"].append(
            {
                "fold_id": f"cohort={cohort}",
                "test_block": {"cohort_id": cohort},
                "train_indices": all_indices[~test_mask],
                "test_indices": all_indices[test_mask],
                "embargo_indices": np.asarray([], dtype=int),
            }
        )
    for activity in data["unique_activities"]:
        test_mask = activities == activity
        specifications["leave_one_activity_out"].append(
            {
                "fold_id": f"activity={activity}",
                "test_block": {"activity_id": activity},
                "train_indices": all_indices[~test_mask],
                "test_indices": all_indices[test_mask],
                "embargo_indices": np.asarray([], dtype=int),
            }
        )
    for cohort in data["unique_cohorts"]:
        for activity in data["unique_activities"]:
            test_mask = (cohorts == cohort) & (activities == activity)
            train_mask = (cohorts != cohort) & (activities != activity)
            embargo_mask = ~(test_mask | train_mask)
            specifications["double_blocked_cohort_activity"].append(
                {
                    "fold_id": f"cohort={cohort}|activity={activity}",
                    "test_block": {"cohort_id": cohort, "activity_id": activity},
                    "train_indices": all_indices[train_mask],
                    "test_indices": all_indices[test_mask],
                    "embargo_indices": all_indices[embargo_mask],
                }
            )
    return specifications


def _evaluate_protocol(
    name: str,
    fold_specs: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any],
    *,
    logistic_c: float,
    max_iter: int,
    random_state: int,
) -> dict[str, Any]:
    np, _, _, _, _, _ = _dependencies()
    labels = data["labels"]
    classes = data["classes"]
    class_set = set(classes)
    sample_ids = data["sample_ids"]
    cohorts = data["cohorts"]
    activities = data["activities"]
    sessions = data["sessions"]
    modality_predictions: dict[str, list[tuple[int, int, str]]] = {
        modality: [] for modality in MODALITY_ORDER
    }
    folds: list[dict[str, Any]] = []
    unevaluable: list[dict[str, Any]] = []

    for fold_number, spec in enumerate(fold_specs, start=1):
        train_indices = np.asarray(spec["train_indices"], dtype=int)
        test_indices = np.asarray(spec["test_indices"], dtype=int)
        embargo_indices = np.asarray(spec["embargo_indices"], dtype=int)
        train_classes = set(int(value) for value in labels[train_indices])
        missing_training_classes = sorted(class_set - train_classes)
        fold: dict[str, Any] = {
            "fold_number": fold_number,
            "fold_id": spec["fold_id"],
            "test_block": dict(spec["test_block"]),
            "train_sample_count": int(len(train_indices)),
            "test_sample_count": int(len(test_indices)),
            "embargo_sample_count": int(len(embargo_indices)),
            "train_class_counts": _class_counts(labels, train_indices),
            "test_class_counts": _class_counts(labels, test_indices),
            "train_cohort_ids": sorted(set(str(value) for value in cohorts[train_indices])),
            "test_cohort_ids": sorted(set(str(value) for value in cohorts[test_indices])),
            "train_activity_ids": sorted(
                set(str(value) for value in activities[train_indices])
            ),
            "test_activity_ids": sorted(
                set(str(value) for value in activities[test_indices])
            ),
            "train_session_ids": sorted(set(str(value) for value in sessions[train_indices])),
            "test_session_ids": sorted(set(str(value) for value in sessions[test_indices])),
            "status": "pending",
        }
        session_overlap = sorted(
            set(fold["train_session_ids"]) & set(fold["test_session_ids"])
        )
        fold["session_overlap_ids"] = session_overlap
        fold["session_overlap_detected"] = bool(session_overlap)
        if session_overlap:
            raise BlockedEvaluationError(
                f"blocked fold {spec['fold_id']} has train/test session overlap: "
                f"{session_overlap}"
            )
        reason: str | None = None
        if not len(test_indices):
            reason = "empty_test_block"
        elif not len(train_indices):
            reason = "empty_training_set"
        elif missing_training_classes:
            reason = "training_missing_global_classes"

        if reason is not None:
            fold["status"] = "unevaluable"
            fold["reason"] = reason
            fold["missing_training_classes"] = missing_training_classes
            folds.append(fold)
            unevaluable.append(
                {
                    "fold_number": fold_number,
                    "fold_id": spec["fold_id"],
                    "reason": reason,
                    "missing_training_classes": missing_training_classes,
                    "train_sample_count": int(len(train_indices)),
                    "test_sample_count": int(len(test_indices)),
                }
            )
            continue

        pending_predictions: dict[str, Any] = {}
        pending_reports: dict[str, dict[str, Any]] = {}
        fit_failure: str | None = None
        for modality in MODALITY_ORDER:
            matrix = data["features"][modality]
            try:
                predictions, convergence_warning = _fit_predict_fold(
                    matrix[train_indices],
                    labels[train_indices],
                    matrix[test_indices],
                    logistic_c=logistic_c,
                    max_iter=max_iter,
                    random_state=random_state,
                )
            except Exception as exc:  # fail the paired fold, but keep the audit report
                fit_failure = f"{modality}: {type(exc).__name__}: {exc}"
                break
            pending_predictions[modality] = predictions
            pending_reports[modality] = {
                **_metrics(labels[test_indices], predictions, classes),
                "convergence_warning": bool(convergence_warning),
            }
        if fit_failure is not None:
            fold["status"] = "unevaluable"
            fold["reason"] = "model_fit_or_prediction_failed"
            fold["error"] = fit_failure
            fold["missing_training_classes"] = []
            folds.append(fold)
            unevaluable.append(
                {
                    "fold_number": fold_number,
                    "fold_id": spec["fold_id"],
                    "reason": "model_fit_or_prediction_failed",
                    "error": fit_failure,
                    "missing_training_classes": [],
                    "train_sample_count": int(len(train_indices)),
                    "test_sample_count": int(len(test_indices)),
                }
            )
            continue

        fold["status"] = "evaluated"
        fold["modalities"] = pending_reports
        folds.append(fold)
        for modality in MODALITY_ORDER:
            modality_predictions[modality].extend(
                (int(index), int(prediction), spec["fold_id"])
                for index, prediction in zip(test_indices, pending_predictions[modality])
            )

    modality_reports: dict[str, Any] = {}
    for modality in MODALITY_ORDER:
        ordered = sorted(modality_predictions[modality], key=lambda item: item[0])
        evaluated_indices = np.asarray([item[0] for item in ordered], dtype=int)
        predictions = np.asarray([item[1] for item in ordered], dtype=int)
        fold_ids = [item[2] for item in ordered]
        pooled = (
            _metrics(labels[evaluated_indices], predictions, classes)
            if len(evaluated_indices)
            else {"sample_count": 0, "accuracy": None, "macro_f1": None}
        )
        cell_reports: list[dict[str, Any]] = []
        if len(evaluated_indices):
            cell_positions: dict[tuple[str, str], list[int]] = {}
            for position, index in enumerate(evaluated_indices):
                key = (str(cohorts[index]), str(activities[index]))
                cell_positions.setdefault(key, []).append(position)
            for (cohort, activity), positions in sorted(cell_positions.items()):
                cell_indices = np.asarray(positions, dtype=int)
                cell_reports.append(
                    {
                        "cohort_id": cohort,
                        "activity_id": activity,
                        **_metrics(
                            labels[evaluated_indices[cell_indices]],
                            predictions[cell_indices],
                            classes,
                        ),
                    }
                )
        equal_weighted = {
            "evaluated_cell_count": len(cell_reports),
            "accuracy": (
                round(float(np.mean([item["accuracy"] for item in cell_reports])), 6)
                if cell_reports
                else None
            ),
            "macro_f1": (
                round(float(np.mean([item["macro_f1"] for item in cell_reports])), 6)
                if cell_reports
                else None
            ),
            "cells": cell_reports,
        }
        modality_reports[modality] = {
            "pooled_oof_sample_weighted": pooled,
            "test_cell_session_equal_weighted": equal_weighted,
            "oof_coverage": round(float(len(evaluated_indices) / len(labels)), 6),
            "oof_sample_ids": [sample_ids[index] for index in evaluated_indices],
            "oof_true_labels": [int(labels[index]) for index in evaluated_indices],
            "oof_predicted_labels": predictions.tolist(),
            "oof_fold_ids": fold_ids,
        }

    def delta(weighting: str, metric: str, baseline: str) -> float | None:
        fusion_value = modality_reports["fusion"][weighting][metric]
        baseline_value = modality_reports[baseline][weighting][metric]
        if fusion_value is None or baseline_value is None:
            return None
        return round(float(fusion_value - baseline_value), 6)

    return {
        "protocol": name,
        "fold_count": len(fold_specs),
        "evaluated_fold_count": len(fold_specs) - len(unevaluable),
        "unevaluable_fold_count": len(unevaluable),
        "unevaluable_folds": unevaluable,
        "modalities": modality_reports,
        "descriptive_fusion_deltas": {
            weighting: {
                f"{metric}_minus_{baseline}": delta(weighting, metric, baseline)
                for metric in ("accuracy", "macro_f1")
                for baseline in ("visual", "sensor")
            }
            for weighting in (
                "pooled_oof_sample_weighted",
                "test_cell_session_equal_weighted",
            )
        },
        "folds": folds,
    }


def blocked_descriptive_evaluation(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    *,
    sample_ids: Sequence[str],
    logistic_c: float = FIXED_LOGISTIC_C,
    max_iter: int = 3000,
    random_state: int = 2026,
) -> dict[str, Any]:
    """Run fixed DIPSER cohort, activity, and dual-blocked OOF evaluations.

    ``sample_ids`` is mandatory and must exactly equal manifest order.  This
    prevents otherwise silent feature-row misalignment.  The function requires
    the DIPSER design of three cohorts and nine activities and restricts input to
    one documented site.
    """

    if float(logistic_c) != FIXED_LOGISTIC_C:
        raise BlockedEvaluationError(
            f"logistic_c is protocol-fixed at {FIXED_LOGISTIC_C}; tuning is not allowed"
        )
    if int(max_iter) < 1:
        raise BlockedEvaluationError("max_iter must be positive")
    data = _validate_inputs(
        records,
        visual_features,
        sensor_features,
        sample_ids=sample_ids,
    )
    specifications = _fold_specifications(data)
    evaluations = {
        name: _evaluate_protocol(
            name,
            folds,
            data,
            logistic_c=FIXED_LOGISTIC_C,
            max_iter=max_iter,
            random_state=random_state,
        )
        for name, folds in specifications.items()
    }
    return {
        "evaluation_kind": "single_site_descriptive_blocked_oof",
        "scope_limitations": [
            "single-site retrospective description only",
            "not an estimate of prospective deployment accuracy",
            "no confidence interval or hypothesis test is produced",
            "descriptive fusion differences do not establish multimodal gain",
        ],
        "claim_status": {
            "multimodal_gain_established": False,
            "cross_site_accuracy_established": False,
            "deployment_accuracy_established": False,
        },
        "contains_inferential_statistics": False,
        "dataset_summary": {
            "sample_count": len(data["sample_ids"]),
            "site_ids": data["unique_sites"],
            "cohort_ids": data["unique_cohorts"],
            "activity_ids": data["unique_activities"],
            "session_ids": data["unique_sessions"],
            "session_count": len(data["unique_sessions"]),
            "complete_three_by_nine_session_grid": data["complete_session_grid"],
            "class_labels": data["classes"],
            "feature_dimensions": data["feature_dimensions"],
            "feature_sample_order_verified": True,
        },
        "fixed_model_protocol": {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "C": FIXED_LOGISTIC_C,
            "solver": "lbfgs",
            "class_weight": None,
            "max_iter": int(max_iter),
            "random_state": int(random_state),
            "preprocessing": "StandardScaler fitted independently inside each training fold",
            "hyperparameter_selection": "none",
            "macro_f1_label_policy": "global class set with zero_division=0",
        },
        "expected_fold_counts": {
            "leave_one_cohort_out": EXPECTED_COHORT_COUNT,
            "leave_one_activity_out": EXPECTED_ACTIVITY_COUNT,
            "double_blocked_cohort_activity": EXPECTED_COHORT_COUNT
            * EXPECTED_ACTIVITY_COUNT,
        },
        "evaluations": evaluations,
    }
