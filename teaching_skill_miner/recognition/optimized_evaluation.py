"""Leakage-resistant exploratory model selection for small multimodal datasets.

This module is deliberately independent from the frozen DIPSER extraction and
credible-report pipeline.  It may be used to explore a *predeclared* compact
model registry, but it never upgrades an exploratory score into an accuracy,
multimodal-gain, or deployment claim.

Every estimator and preprocessing transform is fitted inside an inner grouped
cross-validation split.  The selected estimator is then refitted on the full
outer-training partition and evaluated once on its outer test partition.  The
outer labels are used only for the final descriptive metrics.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


SESSION_DESIGN = "session_grouped_5fold"
LOCO_DESIGN = "leave_one_cohort_out"
LOAO_DESIGN = "leave_one_activity_out"
DOUBLE_BLOCK_DESIGN = "double_blocked_cohort_activity"
SUPPORTED_DESIGNS = (
    SESSION_DESIGN,
    LOCO_DESIGN,
    LOAO_DESIGN,
    DOUBLE_BLOCK_DESIGN,
)
MODALITIES = ("visual", "sensor", "fusion")
IDENTITY_FIELDS = (
    "session_id",
    "participant_id",
    "teacher_id",
    "cohort_id",
    "activity_id",
    "site_id",
)


class OptimizedEvaluationError(ValueError):
    """Raised when inputs cannot support the strict exploratory protocol."""


def _dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "install optional dependencies with `pip install -e '.[recognition]'`"
        ) from exc
    return {
        "np": np,
        "DummyClassifier": DummyClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "ConvergenceWarning": ConvergenceWarning,
        "SelectKBest": SelectKBest,
        "f_classif": f_classif,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
    }


# The ordering is part of the protocol: it is the deterministic final tie-break.
# This compact registry is intentionally fixed rather than generated from outer
# scores.  It covers linear, sparse-selected linear, kernel, and tree models.
_CANDIDATE_CONFIGS: dict[str, dict[str, Any]] = {
    "majority": {"family": "dummy", "strategy": "most_frequent"},
    "logreg_unweighted_c0.03": {
        "family": "logistic_regression",
        "C": 0.03,
        "class_weight": None,
    },
    "logreg_unweighted_c0.3": {
        "family": "logistic_regression",
        "C": 0.3,
        "class_weight": None,
    },
    "logreg_unweighted_c3": {
        "family": "logistic_regression",
        "C": 3.0,
        "class_weight": None,
    },
    "logreg_balanced_c0.03": {
        "family": "logistic_regression",
        "C": 0.03,
        "class_weight": "balanced",
    },
    "logreg_balanced_c0.3": {
        "family": "logistic_regression",
        "C": 0.3,
        "class_weight": "balanced",
    },
    "logreg_balanced_c3": {
        "family": "logistic_regression",
        "C": 3.0,
        "class_weight": "balanced",
    },
    "select20_logreg_unweighted_c0.3": {
        "family": "selected_logistic_regression",
        "C": 0.3,
        "class_weight": None,
        "maximum_feature_count": 20,
    },
    "select20_logreg_balanced_c0.3": {
        "family": "selected_logistic_regression",
        "C": 0.3,
        "class_weight": "balanced",
        "maximum_feature_count": 20,
    },
    "rbf_svc_unweighted_c0.5": {
        "family": "rbf_svc",
        "C": 0.5,
        "gamma": "scale",
        "class_weight": None,
    },
    "rbf_svc_unweighted_c2": {
        "family": "rbf_svc",
        "C": 2.0,
        "gamma": "scale",
        "class_weight": None,
    },
    "rbf_svc_unweighted_c8": {
        "family": "rbf_svc",
        "C": 8.0,
        "gamma": "scale",
        "class_weight": None,
    },
    "rbf_svc_balanced_c0.5": {
        "family": "rbf_svc",
        "C": 0.5,
        "gamma": "scale",
        "class_weight": "balanced",
    },
    "rbf_svc_balanced_c2": {
        "family": "rbf_svc",
        "C": 2.0,
        "gamma": "scale",
        "class_weight": "balanced",
    },
    "linear_svc_unweighted_c0.1": {
        "family": "linear_svc",
        "C": 0.1,
        "class_weight": None,
    },
    "extra_trees_unweighted_leaf2": {
        "family": "extra_trees",
        "n_estimators": 160,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": None,
    },
    "extra_trees_balanced_leaf2": {
        "family": "extra_trees",
        "n_estimators": 160,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
}

DEFAULT_CANDIDATE_IDS = tuple(_CANDIDATE_CONFIGS)


def candidate_registry() -> dict[str, dict[str, Any]]:
    """Return a JSON-serializable copy of the frozen candidate registry."""

    return {name: dict(config) for name, config in _CANDIDATE_CONFIGS.items()}


def _activity_id(record: Mapping[str, Any], sample_id: str) -> str:
    raw_direct = record.get("activity_id")
    direct = "" if raw_direct is None else str(raw_direct).strip()
    source = record.get("source")
    nested = ""
    if isinstance(source, Mapping):
        raw_nested = source.get("experiment_id")
        nested = "" if raw_nested is None else str(raw_nested).strip()
    if direct and nested and direct != nested:
        raise OptimizedEvaluationError(
            f"record {sample_id} has conflicting activity identifiers"
        )
    value = direct or nested
    if not value:
        raise OptimizedEvaluationError(f"record {sample_id} lacks activity_id")
    return value


def _required_id(record: Mapping[str, Any], field: str, sample_id: str) -> str:
    raw_value = record.get(field)
    value = "" if raw_value is None else str(raw_value).strip()
    if not value:
        raise OptimizedEvaluationError(f"record {sample_id} lacks {field}")
    return value


def _prepare_inputs(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    dep = _dependencies()
    np = dep["np"]
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
    ):
        raise OptimizedEvaluationError("records must be a non-empty sequence")

    ids: list[str] = []
    labels: list[int] = []
    identities: dict[str, list[str | None]] = {
        field: [] for field in IDENTITY_FIELDS
    }
    normalized_records: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise OptimizedEvaluationError(f"record {index} is not an object")
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise OptimizedEvaluationError(f"record {index} lacks sample_id")
        try:
            label = int(record["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OptimizedEvaluationError(
                f"record {sample_id} lacks an integer label"
            ) from exc
        ids.append(sample_id)
        labels.append(label)
        identities["session_id"].append(
            _required_id(record, "session_id", sample_id)
        )
        identities["cohort_id"].append(
            _required_id(record, "cohort_id", sample_id)
        )
        identities["activity_id"].append(_activity_id(record, sample_id))
        identities["site_id"].append(_required_id(record, "site_id", sample_id))
        for field in ("participant_id", "teacher_id"):
            raw_value = record.get(field)
            value = "" if raw_value is None else str(raw_value).strip()
            identities[field].append(value or None)
        normalized_records.append(record)

    if len(set(ids)) != len(ids):
        raise OptimizedEvaluationError("record sample_id values must be unique")
    observed_ids = [str(value) for value in sample_ids]
    if observed_ids != ids:
        raise OptimizedEvaluationError(
            "sample_ids must exactly match record order"
        )
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise OptimizedEvaluationError("at least two label classes are required")

    matrices: dict[str, Any] = {}
    for name, values in (
        ("visual", visual_features),
        ("sensor", sensor_features),
    ):
        try:
            matrix = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise OptimizedEvaluationError(
                f"{name} features must be a numeric matrix"
            ) from exc
        if matrix.ndim != 2 or matrix.shape[0] != len(records) or matrix.shape[1] < 1:
            raise OptimizedEvaluationError(
                f"{name} features must have shape ({len(records)}, positive_dimension)"
            )
        if not np.isfinite(matrix).all():
            raise OptimizedEvaluationError(
                f"{name} features contain NaN or infinity"
            )
        matrices[name] = matrix
    matrices["fusion"] = np.concatenate(
        [matrices["visual"], matrices["sensor"]], axis=1
    )

    # A session is a recording/design cell.  Conflicting metadata would make
    # the session, cohort, and activity blocking audits uninterpretable.
    session_design: dict[str, tuple[str, str, str]] = {}
    for session, cohort, activity, site in zip(
        identities["session_id"],
        identities["cohort_id"],
        identities["activity_id"],
        identities["site_id"],
    ):
        assert session is not None and cohort is not None
        assert activity is not None and site is not None
        design = (cohort, activity, site)
        previous = session_design.setdefault(session, design)
        if previous != design:
            raise OptimizedEvaluationError(
                f"session_id {session} maps to multiple cohort/activity/site cells"
            )

    arrays = {
        field: np.asarray(values, dtype=object)
        for field, values in identities.items()
    }
    return {
        "records": normalized_records,
        "sample_ids": ids,
        "labels": np.asarray(labels, dtype=int),
        "classes": classes,
        "features": matrices,
        "identities": arrays,
        "feature_dimensions": {
            name: int(matrix.shape[1]) for name, matrix in matrices.items()
        },
        "unique_sessions": sorted(session_design),
        "unique_cohorts": sorted(set(str(value) for value in identities["cohort_id"])),
        "unique_activities": sorted(
            set(str(value) for value in identities["activity_id"])
        ),
        "unique_sites": sorted(set(str(value) for value in identities["site_id"])),
        "optional_identity_completeness": {
            field: sum(value is not None for value in identities[field]) == len(records)
            for field in ("participant_id", "teacher_id")
        },
    }


def _candidate_configs(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    ids = tuple(str(value) for value in candidate_ids)
    if not ids:
        raise OptimizedEvaluationError("candidate_ids must not be empty")
    if len(set(ids)) != len(ids):
        raise OptimizedEvaluationError("candidate_ids must be unique")
    unknown = [name for name in ids if name not in _CANDIDATE_CONFIGS]
    if unknown:
        raise OptimizedEvaluationError(
            "unknown candidate IDs: " + ", ".join(unknown)
        )
    return ids


def _build_estimator(candidate_id: str, *, feature_count: int, seed: int) -> Any:
    dep = _dependencies()
    config = _CANDIDATE_CONFIGS[candidate_id]
    family = config["family"]
    if family == "dummy":
        return dep["DummyClassifier"](strategy="most_frequent")
    if family in {"logistic_regression", "selected_logistic_regression"}:
        steps: list[tuple[str, Any]] = []
        if family == "selected_logistic_regression":
            steps.append(
                (
                    "training_only_univariate_selection",
                    dep["SelectKBest"](
                        score_func=dep["f_classif"],
                        k=min(int(config["maximum_feature_count"]), feature_count),
                    ),
                )
            )
        steps.extend(
            [
                ("training_only_scaler", dep["StandardScaler"]()),
                (
                    "classifier",
                    dep["LogisticRegression"](
                        C=float(config["C"]),
                        class_weight=config["class_weight"],
                        max_iter=4000,
                        random_state=int(seed),
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        return dep["Pipeline"](steps)
    if family == "rbf_svc":
        return dep["Pipeline"](
            [
                ("training_only_scaler", dep["StandardScaler"]()),
                (
                    "classifier",
                    dep["SVC"](
                        C=float(config["C"]),
                        gamma=config["gamma"],
                        class_weight=config["class_weight"],
                        kernel="rbf",
                    ),
                ),
            ]
        )
    if family == "linear_svc":
        return dep["Pipeline"](
            [
                ("training_only_scaler", dep["StandardScaler"]()),
                (
                    "classifier",
                    dep["SVC"](
                        C=float(config["C"]),
                        class_weight=config["class_weight"],
                        kernel="linear",
                    ),
                ),
            ]
        )
    if family == "extra_trees":
        return dep["ExtraTreesClassifier"](
            n_estimators=int(config["n_estimators"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=config["max_features"],
            class_weight=config["class_weight"],
            random_state=int(seed),
            n_jobs=1,
        )
    raise RuntimeError(f"unimplemented candidate family: {family}")


def _metrics(y_true: Any, y_pred: Any, classes: Sequence[int]) -> dict[str, Any]:
    dep = _dependencies()
    if not len(y_true):
        return {"sample_count": 0, "accuracy": None, "macro_f1": None}
    return {
        "sample_count": int(len(y_true)),
        "accuracy": round(float(dep["accuracy_score"](y_true, y_pred)), 6),
        "macro_f1": round(
            float(
                dep["f1_score"](
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


def _fit_predict(
    candidate_id: str,
    train_x: Any,
    train_y: Any,
    test_x: Any,
    *,
    seed: int,
) -> tuple[Any, bool]:
    dep = _dependencies()
    estimator = _build_estimator(
        candidate_id, feature_count=int(train_x.shape[1]), seed=seed
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", dep["ConvergenceWarning"])
        estimator.fit(train_x, train_y)
    convergence_warning = any(
        issubclass(item.category, dep["ConvergenceWarning"]) for item in caught
    )
    return estimator.predict(test_x), bool(convergence_warning)


def _class_group_support(labels: Any, groups: Any) -> dict[str, int]:
    dep = _dependencies()
    np = dep["np"]
    return {
        str(int(label)): int(len(np.unique(groups[labels == label])))
        for label in np.unique(labels)
    }


def _inner_splits(
    labels: Any,
    groups: Any,
    *,
    inner_splits: int,
    seed: int,
) -> list[tuple[Any, Any]]:
    dep = _dependencies()
    np = dep["np"]
    unique_groups = np.unique(groups)
    if len(unique_groups) < inner_splits:
        raise OptimizedEvaluationError(
            f"inner CV requires {inner_splits} sessions; found {len(unique_groups)}"
        )
    support = _class_group_support(labels, groups)
    weak = {label: count for label, count in support.items() if count < inner_splits}
    if weak:
        raise OptimizedEvaluationError(
            f"inner CV class/session support is below {inner_splits}: {weak}"
        )
    splitter = dep["StratifiedGroupKFold"](
        n_splits=inner_splits, shuffle=True, random_state=int(seed)
    )
    splits = list(splitter.split(np.zeros((len(labels), 1)), labels, groups))
    global_classes = set(int(value) for value in labels)
    for train_indices, validation_indices in splits:
        if set(groups[train_indices]) & set(groups[validation_indices]):
            raise RuntimeError("session leakage in inner CV")
        if set(int(value) for value in labels[train_indices]) != global_classes:
            raise OptimizedEvaluationError(
                "an inner training split is missing an outer-training class"
            )
    return splits


def _selection_key(item: Mapping[str, Any], candidate_order: Mapping[str, int]) -> tuple[Any, ...]:
    return (
        -float(item["inner_oof_accuracy"]),
        -float(item["inner_oof_macro_f1"]),
        int(candidate_order[str(item["candidate_id"])]),
    )


def _select_candidate(
    matrix: Any,
    labels: Any,
    session_groups: Any,
    *,
    classes: Sequence[int],
    candidate_ids: Sequence[str],
    inner_splits: int,
    seed: int,
) -> dict[str, Any]:
    dep = _dependencies()
    np = dep["np"]
    splits = _inner_splits(
        labels, session_groups, inner_splits=inner_splits, seed=seed
    )
    candidate_order = {name: index for index, name in enumerate(candidate_ids)}
    reports: list[dict[str, Any]] = []
    for candidate_number, candidate_id in enumerate(candidate_ids):
        predictions = np.empty(len(labels), dtype=int)
        assigned = np.zeros(len(labels), dtype=bool)
        fold_reports: list[dict[str, Any]] = []
        failure: str | None = None
        warning_count = 0
        for fold_number, (train_indices, validation_indices) in enumerate(splits, 1):
            try:
                fold_prediction, convergence_warning = _fit_predict(
                    candidate_id,
                    matrix[train_indices],
                    labels[train_indices],
                    matrix[validation_indices],
                    seed=seed + candidate_number * 101 + fold_number,
                )
            except Exception as exc:  # retain a complete audit of rejected candidates
                failure = f"{type(exc).__name__}: {exc}"
                break
            predictions[validation_indices] = fold_prediction
            assigned[validation_indices] = True
            warning_count += int(convergence_warning)
            fold_reports.append(
                {
                    "fold": fold_number,
                    "train_session_count": int(
                        len(np.unique(session_groups[train_indices]))
                    ),
                    "validation_session_count": int(
                        len(np.unique(session_groups[validation_indices]))
                    ),
                    "session_overlap_ids": sorted(
                        set(str(value) for value in session_groups[train_indices])
                        & set(str(value) for value in session_groups[validation_indices])
                    ),
                    **_metrics(
                        labels[validation_indices], fold_prediction, classes
                    ),
                }
            )
        if failure is not None or not assigned.all():
            reports.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "failure": failure or "inner CV did not assign every row",
                    "folds": fold_reports,
                    "convergence_warning_count": warning_count,
                }
            )
            continue
        aggregate = _metrics(labels, predictions, classes)
        reports.append(
            {
                "candidate_id": candidate_id,
                "status": "evaluated",
                "inner_oof_accuracy": aggregate["accuracy"],
                "inner_oof_macro_f1": aggregate["macro_f1"],
                "folds": fold_reports,
                "convergence_warning_count": warning_count,
            }
        )
    eligible = [item for item in reports if item["status"] == "evaluated"]
    if not eligible:
        raise OptimizedEvaluationError("all predeclared candidates failed inner CV")
    selected = sorted(
        eligible, key=lambda item: _selection_key(item, candidate_order)
    )[0]
    for item in reports:
        item["selected"] = item["candidate_id"] == selected["candidate_id"]
    return {
        "selected_candidate_id": selected["candidate_id"],
        "selected_inner_oof_accuracy": selected["inner_oof_accuracy"],
        "selected_inner_oof_macro_f1": selected["inner_oof_macro_f1"],
        "selection_rule": (
            "maximize inner session-grouped OOF Accuracy; then Macro-F1; "
            "then frozen candidate-registry order"
        ),
        "candidate_reports": reports,
    }


def _fold_specs(data: Mapping[str, Any], design: str, *, outer_splits: int, seed: int) -> list[dict[str, Any]]:
    dep = _dependencies()
    np = dep["np"]
    labels = data["labels"]
    identities = data["identities"]
    all_indices = np.arange(len(labels), dtype=int)
    specs: list[dict[str, Any]] = []
    if design == SESSION_DESIGN:
        sessions = identities["session_id"]
        if len(np.unique(sessions)) < outer_splits:
            raise OptimizedEvaluationError(
                f"session design requires {outer_splits} sessions"
            )
        weak = {
            label: count
            for label, count in _class_group_support(labels, sessions).items()
            if count < outer_splits
        }
        if weak:
            raise OptimizedEvaluationError(
                f"outer CV class/session support is below {outer_splits}: {weak}"
            )
        splitter = dep["StratifiedGroupKFold"](
            n_splits=outer_splits, shuffle=True, random_state=int(seed)
        )
        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(data["features"]["fusion"], labels, sessions), 1
        ):
            specs.append(
                {
                    "fold_id": f"session-fold-{fold}",
                    "test_block": {"outer_fold": fold},
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "embargo_indices": np.asarray([], dtype=int),
                    "expected_disjoint_fields": ("session_id",),
                }
            )
        return specs
    if design == LOCO_DESIGN:
        for cohort in data["unique_cohorts"]:
            test_mask = identities["cohort_id"] == cohort
            specs.append(
                {
                    "fold_id": f"cohort={cohort}",
                    "test_block": {"cohort_id": cohort},
                    "train_indices": all_indices[~test_mask],
                    "test_indices": all_indices[test_mask],
                    "embargo_indices": np.asarray([], dtype=int),
                    "expected_disjoint_fields": ("session_id", "cohort_id"),
                }
            )
        return specs
    if design == LOAO_DESIGN:
        for activity in data["unique_activities"]:
            test_mask = identities["activity_id"] == activity
            specs.append(
                {
                    "fold_id": f"activity={activity}",
                    "test_block": {"activity_id": activity},
                    "train_indices": all_indices[~test_mask],
                    "test_indices": all_indices[test_mask],
                    "embargo_indices": np.asarray([], dtype=int),
                    "expected_disjoint_fields": ("session_id", "activity_id"),
                }
            )
        return specs
    if design == DOUBLE_BLOCK_DESIGN:
        for cohort in data["unique_cohorts"]:
            for activity in data["unique_activities"]:
                test_mask = (identities["cohort_id"] == cohort) & (
                    identities["activity_id"] == activity
                )
                train_mask = (identities["cohort_id"] != cohort) & (
                    identities["activity_id"] != activity
                )
                embargo_mask = ~(test_mask | train_mask)
                specs.append(
                    {
                        "fold_id": f"cohort={cohort}|activity={activity}",
                        "test_block": {
                            "cohort_id": cohort,
                            "activity_id": activity,
                        },
                        "train_indices": all_indices[train_mask],
                        "test_indices": all_indices[test_mask],
                        "embargo_indices": all_indices[embargo_mask],
                        "expected_disjoint_fields": (
                            "session_id",
                            "cohort_id",
                            "activity_id",
                        ),
                    }
                )
        return specs
    raise OptimizedEvaluationError(f"unsupported design: {design}")


def _class_counts(labels: Any, indices: Any) -> dict[str, int]:
    return {
        str(int(label)): int(count)
        for label, count in sorted(
            Counter(int(labels[index]) for index in indices).items()
        )
    }


def _identity_audit(
    data: Mapping[str, Any], train_indices: Any, test_indices: Any
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    identities = data["identities"]
    for field in IDENTITY_FIELDS:
        values = identities[field]
        train_values = {
            str(value) for value in values[train_indices] if value is not None
        }
        test_values = {
            str(value) for value in values[test_indices] if value is not None
        }
        audit[field] = {
            "field_complete": bool(all(value is not None for value in values)),
            "train_values": sorted(train_values),
            "test_values": sorted(test_values),
            "overlap_ids": sorted(train_values & test_values),
            "overlap_detected": bool(train_values & test_values),
        }
    return audit


def _unimodal_selection_key(name: str, report: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(report["selected_inner_oof_accuracy"]),
        -float(report["selected_inner_oof_macro_f1"]),
        name,
    )


def _evaluate_design(
    data: Mapping[str, Any],
    design: str,
    *,
    outer_splits: int,
    inner_splits: int,
    candidate_ids: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    dep = _dependencies()
    np = dep["np"]
    labels = data["labels"]
    classes = data["classes"]
    sessions = data["identities"]["session_id"]
    specs = _fold_specs(data, design, outer_splits=outer_splits, seed=seed)
    predictions: dict[str, Any] = {
        name: np.full(len(labels), None, dtype=object)
        for name in ("majority_baseline", *MODALITIES, "best_unimodal_nested")
    }
    assigned_fold = np.full(len(labels), None, dtype=object)
    folds: list[dict[str, Any]] = []
    unevaluable: list[dict[str, Any]] = []

    for fold_number, spec in enumerate(specs, 1):
        train_indices = np.asarray(spec["train_indices"], dtype=int)
        test_indices = np.asarray(spec["test_indices"], dtype=int)
        embargo_indices = np.asarray(spec["embargo_indices"], dtype=int)
        identity_audit = _identity_audit(data, train_indices, test_indices)
        sample_overlap = sorted(
            set(data["sample_ids"][index] for index in train_indices)
            & set(data["sample_ids"][index] for index in test_indices)
        )
        expected_overlap = {
            field: identity_audit[field]["overlap_ids"]
            for field in spec["expected_disjoint_fields"]
            if identity_audit[field]["overlap_ids"]
        }
        if sample_overlap or expected_overlap:
            raise OptimizedEvaluationError(
                f"outer leakage in {design} {spec['fold_id']}: "
                f"sample={sample_overlap}, identity={expected_overlap}"
            )
        fold: dict[str, Any] = {
            "fold_number": fold_number,
            "fold_id": spec["fold_id"],
            "test_block": dict(spec["test_block"]),
            "train_sample_count": int(len(train_indices)),
            "test_sample_count": int(len(test_indices)),
            "embargo_sample_count": int(len(embargo_indices)),
            "train_class_counts": _class_counts(labels, train_indices),
            "test_class_counts": _class_counts(labels, test_indices),
            "sample_id_overlap": sample_overlap,
            "identity_overlap_audit": identity_audit,
            "expected_disjoint_fields": list(spec["expected_disjoint_fields"]),
            "outer_test_labels_used_for_model_or_hyperparameter_selection": False,
            "status": "pending",
        }
        reason: str | None = None
        if not len(test_indices):
            reason = "empty_test_block"
        elif not len(train_indices):
            reason = "empty_training_set"
        elif set(int(value) for value in labels[train_indices]) != set(classes):
            reason = "outer_training_missing_global_class"
        if reason is not None:
            fold["status"] = "unevaluable"
            fold["reason"] = reason
            folds.append(fold)
            unevaluable.append(
                {
                    "fold_number": fold_number,
                    "fold_id": spec["fold_id"],
                    "reason": reason,
                }
            )
            continue

        selections: dict[str, Any] = {}
        outer_predictions: dict[str, Any] = {}
        failure: str | None = None
        for modality_number, modality in enumerate(MODALITIES):
            matrix = data["features"][modality]
            try:
                selection = _select_candidate(
                    matrix[train_indices],
                    labels[train_indices],
                    sessions[train_indices],
                    classes=classes,
                    candidate_ids=candidate_ids,
                    inner_splits=inner_splits,
                    seed=seed + fold_number * 1009 + modality_number * 100_003,
                )
                prediction, convergence_warning = _fit_predict(
                    selection["selected_candidate_id"],
                    matrix[train_indices],
                    labels[train_indices],
                    matrix[test_indices],
                    seed=seed + fold_number * 10_007 + modality_number,
                )
            except Exception as exc:
                failure = f"{modality}: {type(exc).__name__}: {exc}"
                break
            selection["outer_refit_convergence_warning"] = bool(
                convergence_warning
            )
            selections[modality] = selection
            outer_predictions[modality] = prediction
        if failure is not None:
            fold["status"] = "unevaluable"
            fold["reason"] = "nested_selection_or_outer_refit_failed"
            fold["error"] = failure
            fold["completed_modality_selections"] = selections
            folds.append(fold)
            unevaluable.append(
                {
                    "fold_number": fold_number,
                    "fold_id": spec["fold_id"],
                    "reason": fold["reason"],
                    "error": failure,
                }
            )
            continue

        selected_unimodal = sorted(
            ("visual", "sensor"),
            key=lambda name: _unimodal_selection_key(name, selections[name]),
        )[0]
        majority_label = sorted(
            Counter(int(value) for value in labels[train_indices]).items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        outer_predictions["majority_baseline"] = np.full(
            len(test_indices), majority_label, dtype=int
        )
        outer_predictions["best_unimodal_nested"] = outer_predictions[
            selected_unimodal
        ].copy()
        for name, values in outer_predictions.items():
            predictions[name][test_indices] = [int(value) for value in values]
        assigned_fold[test_indices] = spec["fold_id"]

        fold["status"] = "evaluated"
        fold["majority_label_selected_from_outer_training_only"] = int(
            majority_label
        )
        fold["selected_unimodal_from_inner_training_scores"] = selected_unimodal
        fold["modality_selection"] = selections
        fold["outer_test_metrics"] = {
            name: _metrics(labels[test_indices], values, classes)
            for name, values in outer_predictions.items()
        }
        folds.append(fold)

    evaluated_indices = np.asarray(
        [index for index, value in enumerate(predictions["fusion"]) if value is not None],
        dtype=int,
    )
    metrics: dict[str, Any] = {}
    for name, values in predictions.items():
        if len(evaluated_indices):
            predicted = np.asarray(
                [int(values[index]) for index in evaluated_indices], dtype=int
            )
            metrics[name] = _metrics(
                labels[evaluated_indices], predicted, classes
            )
        else:
            metrics[name] = _metrics([], [], classes)

    def delta(candidate: str, reference: str) -> dict[str, float | None]:
        if metrics[candidate]["accuracy"] is None:
            return {"accuracy": None, "macro_f1": None}
        return {
            metric: round(
                float(metrics[candidate][metric] - metrics[reference][metric]), 6
            )
            for metric in ("accuracy", "macro_f1")
        }

    oof_rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(data["sample_ids"]):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "label": int(labels[index]),
            "fold_id": assigned_fold[index],
            "evaluated": predictions["fusion"][index] is not None,
        }
        for name, values in predictions.items():
            value = values[index]
            row[f"{name}_prediction"] = None if value is None else int(value)
        oof_rows.append(row)

    fusion_accuracy = metrics["fusion"]["accuracy"]
    return {
        "design": design,
        "fold_count": len(specs),
        "evaluated_fold_count": sum(fold["status"] == "evaluated" for fold in folds),
        "unevaluable_folds": unevaluable,
        "evaluated_sample_count": int(len(evaluated_indices)),
        "oof_coverage": round(float(len(evaluated_indices) / len(labels)), 6),
        "metrics": metrics,
        "descriptive_deltas": {
            "fusion_minus_fold_training_majority": delta(
                "fusion", "majority_baseline"
            ),
            "fusion_minus_nested_best_unimodal": delta(
                "fusion", "best_unimodal_nested"
            ),
        },
        "descriptive_threshold_checks": {
            "fusion_accuracy_above_fold_training_majority": bool(
                fusion_accuracy is not None
                and fusion_accuracy > metrics["majority_baseline"]["accuracy"]
            ),
            "fusion_accuracy_at_least_0_8": bool(
                fusion_accuracy is not None and fusion_accuracy >= 0.8
            ),
            "these_are_claims": False,
        },
        "folds": folds,
        "oof_predictions": oof_rows,
        "contains_confidence_intervals_or_hypothesis_tests": False,
        "claim_status": {
            "accuracy_established": False,
            "multimodal_gain_established": False,
            "session_disjoint_accuracy_established": False,
            "cohort_disjoint_accuracy_established": False,
            "activity_disjoint_accuracy_established": False,
            "deployment_accuracy_established": False,
        },
    }


def optimized_multimodal_evaluation(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    sample_ids: Sequence[str],
    *,
    designs: Sequence[str] = (SESSION_DESIGN,),
    outer_splits: int = 5,
    inner_splits: int = 3,
    candidate_ids: Sequence[str] = DEFAULT_CANDIDATE_IDS,
    seed: int = 2026,
) -> dict[str, Any]:
    """Run nested grouped exploratory evaluation for one or more fixed designs.

    ``records`` supplies labels and grouping identities only.  It is never
    converted into model features.  The caller must supply already frozen
    numeric visual and sensor matrices in exact ``sample_ids`` order.  Identity,
    path/name, sample ID, timestamps, labels, annotator fields, and any feature
    derived from those fields are forbidden as model inputs by protocol.
    """

    if outer_splits != 5:
        raise OptimizedEvaluationError(
            "the session protocol is fixed at five outer folds"
        )
    if inner_splits < 2:
        raise OptimizedEvaluationError("inner_splits must be at least two")
    selected_candidates = _candidate_configs(candidate_ids)
    selected_designs = tuple(str(value) for value in designs)
    if not selected_designs:
        raise OptimizedEvaluationError("designs must not be empty")
    if len(set(selected_designs)) != len(selected_designs):
        raise OptimizedEvaluationError("designs must be unique")
    unsupported = [name for name in selected_designs if name not in SUPPORTED_DESIGNS]
    if unsupported:
        raise OptimizedEvaluationError(
            "unsupported designs: " + ", ".join(unsupported)
        )

    data = _prepare_inputs(
        records, visual_features, sensor_features, sample_ids
    )
    evaluations = {
        design: _evaluate_design(
            data,
            design,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            candidate_ids=selected_candidates,
            seed=seed + design_index * 1_000_003,
        )
        for design_index, design in enumerate(selected_designs)
    }
    return {
        "protocol": "optimized_nested_grouped_multimodal_oof_v1",
        "evaluation_kind": "exploratory_model_optimization_not_confirmatory",
        "sample_count": len(data["records"]),
        "class_labels": data["classes"],
        "feature_dimensions": data["feature_dimensions"],
        "feature_sample_order_verified": True,
        "designs": list(selected_designs),
        "outer_splits_for_session_design": outer_splits,
        "inner_session_grouped_splits": inner_splits,
        "seed": seed,
        "candidate_ids": list(selected_candidates),
        "candidate_registry": {
            name: dict(_CANDIDATE_CONFIGS[name]) for name in selected_candidates
        },
        "candidate_set_frozen_before_outer_scoring": True,
        "all_preprocessing_fitted_inside_training_partitions": True,
        "all_model_selection_uses_inner_training_partitions_only": True,
        "outer_test_labels_used_for_model_or_hyperparameter_selection": False,
        "feature_policy": {
            "records_converted_to_features": False,
            "allowed_inputs": ["frozen_visual_numeric_matrix", "frozen_sensor_numeric_matrix"],
            "forbidden_inputs": [
                "identity_or_group_fields",
                "file_or_archive_paths_and_names",
                "sample_ids_or_timestamps",
                "ground_truth_or_annotator_fields",
                "features_derived_from_any_forbidden_field",
            ],
            "caller_attestation_required": True,
        },
        "dataset_design_summary": {
            "session_count": len(data["unique_sessions"]),
            "cohort_ids": data["unique_cohorts"],
            "activity_ids": data["unique_activities"],
            "site_ids": data["unique_sites"],
            "optional_identity_completeness": data[
                "optional_identity_completeness"
            ],
        },
        "selection_objective": (
            "inner session-grouped OOF Accuracy, with Macro-F1 and frozen registry "
            "order as deterministic tie-breakers"
        ),
        "evaluations": evaluations,
        "contains_inferential_statistics": False,
        "claim_status": {
            "accuracy_established": False,
            "multimodal_gain_established": False,
            "cross_session_accuracy_established": False,
            "cross_cohort_accuracy_established": False,
            "cross_activity_accuracy_established": False,
            "deployment_accuracy_established": False,
        },
        "claim_limitation": (
            "Iterative optimization on one retrospective single-site dataset is "
            "exploratory. A separately frozen prospective external dataset is "
            "required for any established accuracy or deployment claim."
        ),
    }


def optimized_session_grouped_evaluation(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    sample_ids: Sequence[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper for the fixed five-fold session design."""

    return optimized_multimodal_evaluation(
        records,
        visual_features,
        sensor_features,
        sample_ids,
        designs=(SESSION_DESIGN,),
        **kwargs,
    )


def optimized_blocked_evaluation(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    sample_ids: Sequence[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper for nested LOCO, LOAO, and double blocking."""

    return optimized_multimodal_evaluation(
        records,
        visual_features,
        sensor_features,
        sample_ids,
        designs=(LOCO_DESIGN, LOAO_DESIGN, DOUBLE_BLOCK_DESIGN),
        **kwargs,
    )
