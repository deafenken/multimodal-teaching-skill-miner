"""Exploratory evaluation of the post-selected offline sequence candidate.

This module deliberately isolates a model that was proposed after inspecting
the current DIPSER study results.  Its estimates are therefore exploratory and
must not be promoted to confirmatory, real-time, or deployment claims.

The candidate fits a fixed scaled logistic regression on each outer training
fold.  Window probabilities in a held-out fold may be pooled only within one
``(session_id, participant_id)`` sequence.  Pooling uses the geometric mean,
and the default medium class is replaced only when the strongest minority-class
odds exceed the fixed threshold.  No labels are used during pooling.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any


FIXED_LOGISTIC_C = 0.1
FIXED_CLASS_WEIGHT = "balanced"
FIXED_SOLVER = "lbfgs"
FIXED_MAX_ITER = 4000
FIXED_RANDOM_STATE = 2026
FIXED_OUTER_SPLITS = 5
FIXED_DEFAULT_LABEL = 1
FIXED_MINORITY_LABELS = (0, 2)
FIXED_MINORITY_TO_MEDIUM_RATIO = 3.0
FIXED_CLASS_NAMES = ("low", "medium", "high")
SUPPORTED_OUTER_DESIGNS = ("sgkf5", "leave_one_session_out")


class OfflineSequenceEvaluationError(ValueError):
    """Raised when data cannot support the fixed leakage-safe protocol."""


def _dependencies() -> tuple[Any, ...]:
    try:
        import numpy as np
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
        from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "install optional dependencies with `pip install -e '.[recognition]'`"
        ) from exc
    return (
        np,
        LogisticRegression,
        StandardScaler,
        StratifiedGroupKFold,
        LeaveOneGroupOut,
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        ConvergenceWarning,
    )


def _required_identifier(
    record: Mapping[str, Any], field: str, *, sample_id: str
) -> str:
    value = str(record.get(field, "")).strip()
    if not value:
        raise OfflineSequenceEvaluationError(f"record {sample_id} lacks {field}")
    return value


def _validate_inputs(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    *,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    np, *_ = _dependencies()
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise OfflineSequenceEvaluationError("records must be a non-empty sequence")

    normalized_records: list[Mapping[str, Any]] = []
    manifest_sample_ids: list[str] = []
    labels: list[int] = []
    sessions: list[str] = []
    participants: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise OfflineSequenceEvaluationError(f"record {index} is not an object")
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise OfflineSequenceEvaluationError(f"record {index} lacks sample_id")
        try:
            label = int(record["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OfflineSequenceEvaluationError(
                f"record {sample_id} lacks an integer label"
            ) from exc
        manifest_sample_ids.append(sample_id)
        labels.append(label)
        sessions.append(_required_identifier(record, "session_id", sample_id=sample_id))
        participants.append(
            _required_identifier(record, "participant_id", sample_id=sample_id)
        )
        normalized_records.append(record)

    if len(set(manifest_sample_ids)) != len(manifest_sample_ids):
        raise OfflineSequenceEvaluationError("manifest sample_id values must be unique")
    ordered_feature_ids = [str(value) for value in sample_ids]
    if ordered_feature_ids != manifest_sample_ids:
        mismatch = next(
            (
                index
                for index, (observed, expected) in enumerate(
                    zip(ordered_feature_ids, manifest_sample_ids)
                )
                if observed != expected
            ),
            min(len(ordered_feature_ids), len(manifest_sample_ids)),
        )
        raise OfflineSequenceEvaluationError(
            "sample_ids must exactly match manifest record order; "
            f"first mismatch is at row {mismatch}"
        )

    if sorted(set(labels)) != [0, 1, 2]:
        raise OfflineSequenceEvaluationError(
            "the fixed candidate requires labels 0=low, 1=medium, and 2=high"
        )

    participant_sessions: dict[str, set[str]] = {}
    for session, participant in zip(sessions, participants):
        participant_sessions.setdefault(participant, set()).add(session)
    repeated = {
        participant: sorted(values)
        for participant, values in participant_sessions.items()
        if len(values) > 1
    }
    if repeated:
        raise OfflineSequenceEvaluationError(
            "session-grouped evaluation would leak participant identity because "
            f"participants occur in multiple sessions: {repeated}"
        )

    prepared: dict[str, Any] = {}
    for name, values in (("visual", visual_features), ("sensor", sensor_features)):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(records) or matrix.shape[1] < 1:
            raise OfflineSequenceEvaluationError(
                f"{name} features must have shape ({len(records)}, positive_dimension)"
            )
        if not np.isfinite(matrix).all():
            raise OfflineSequenceEvaluationError(f"{name} features contain NaN or infinity")
        prepared[name] = matrix
    fusion = np.concatenate([prepared["visual"], prepared["sensor"]], axis=1)

    return {
        "records": normalized_records,
        "sample_ids": manifest_sample_ids,
        "labels": np.asarray(labels, dtype=int),
        "sessions": np.asarray(sessions, dtype=object),
        "participants": np.asarray(participants, dtype=object),
        "fusion": fusion,
        "feature_dimensions": {
            "visual": int(prepared["visual"].shape[1]),
            "sensor": int(prepared["sensor"].shape[1]),
            "fusion": int(fusion.shape[1]),
        },
    }


def _aligned_probabilities(model: Any, matrix: Any) -> Any:
    np, *_ = _dependencies()
    probabilities = np.zeros((len(matrix), 3), dtype=float)
    raw = model.predict_proba(matrix)
    for column, label in enumerate(model.classes_):
        probabilities[:, int(label)] = raw[:, column]
    return probabilities


def _fit_predict_probabilities(train_x: Any, train_y: Any, test_x: Any) -> tuple[Any, bool]:
    (
        _,
        LogisticRegression,
        StandardScaler,
        *_,
        ConvergenceWarning,
    ) = _dependencies()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    model = LogisticRegression(
        C=FIXED_LOGISTIC_C,
        class_weight=FIXED_CLASS_WEIGHT,
        solver=FIXED_SOLVER,
        max_iter=FIXED_MAX_ITER,
        random_state=FIXED_RANDOM_STATE,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_scaled, train_y)
    if set(int(value) for value in model.classes_) != {0, 1, 2}:
        raise OfflineSequenceEvaluationError(
            "an outer training fold is missing one or more fixed classes"
        )
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    return _aligned_probabilities(model, scaler.transform(test_x)), convergence_warning


def _hard_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    (
        _,
        _,
        _,
        _,
        _,
        accuracy_score,
        _balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        _,
    ) = _dependencies()
    labels = [0, 1, 2]
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "sample_count": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        # Average all three fixed recalls, including zero for a class absent from
        # a small test fold.  This avoids silently changing the denominator.
        "balanced_accuracy": round(float(sum(recall) / len(labels)), 6),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            6,
        ),
        "weighted_f1": round(
            float(
                f1_score(
                    y_true, y_pred, labels=labels, average="weighted", zero_division=0
                )
            ),
            6,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "per_class": {
            name: {
                "label": label,
                "precision": round(float(precision[label]), 6),
                "recall": round(float(recall[label]), 6),
                "f1": round(float(per_class_f1[label]), 6),
                "support": int(support[label]),
            }
            for label, name in enumerate(FIXED_CLASS_NAMES)
        },
    }


def _sequence_id(session: str, participant: str) -> str:
    return f"session={session}|participant={participant}"


def _pool_fold_probabilities(
    probabilities: Any,
    sessions: Any,
    participants: Any,
) -> tuple[Any, Any, Any, list[dict[str, Any]]]:
    np, *_ = _dependencies()
    pooled_probabilities = np.zeros_like(probabilities, dtype=float)
    pooled_predictions = np.full(len(probabilities), FIXED_DEFAULT_LABEL, dtype=int)
    decision_ratios = np.zeros(len(probabilities), dtype=float)
    sequence_reports: list[dict[str, Any]] = []
    sequence_keys = list(dict.fromkeys(zip(sessions.tolist(), participants.tolist())))
    for session, participant in sequence_keys:
        indices = np.flatnonzero((sessions == session) & (participants == participant))
        if not len(indices):  # pragma: no cover - protected by construction
            continue
        mean_log_probability = np.log(np.clip(probabilities[indices], 1e-12, 1.0)).mean(
            axis=0
        )
        geometric = np.exp(mean_log_probability - mean_log_probability.max())
        geometric /= geometric.sum()
        minority_label = (
            FIXED_MINORITY_LABELS[0]
            if geometric[FIXED_MINORITY_LABELS[0]]
            >= geometric[FIXED_MINORITY_LABELS[1]]
            else FIXED_MINORITY_LABELS[1]
        )
        ratio = float(geometric[minority_label] / max(geometric[FIXED_DEFAULT_LABEL], 1e-12))
        prediction = (
            int(minority_label)
            if ratio >= FIXED_MINORITY_TO_MEDIUM_RATIO
            else FIXED_DEFAULT_LABEL
        )
        pooled_probabilities[indices] = geometric
        pooled_predictions[indices] = prediction
        decision_ratios[indices] = ratio
        sequence_reports.append(
            {
                "sequence_id": _sequence_id(str(session), str(participant)),
                "session_id": str(session),
                "participant_id": str(participant),
                "sample_count": int(len(indices)),
                "pooled_probabilities": [round(float(value), 8) for value in geometric],
                "minority_to_medium_ratio": round(ratio, 8),
                "pooled_prediction": prediction,
            }
        )
    return pooled_probabilities, pooled_predictions, decision_ratios, sequence_reports


def _identity_values(records: Sequence[Mapping[str, Any]], indices: Any, field: str) -> set[str]:
    values: set[str] = set()
    for index in indices:
        value = str(records[int(index)].get(field, "")).strip()
        if value:
            values.add(value)
    return values


def offline_sequence_evaluation(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    *,
    sample_ids: Sequence[str],
    outer_design: str = "sgkf5",
) -> dict[str, Any]:
    """Evaluate the fixed post-selected fusion candidate without sequence leakage.

    ``sgkf5`` is the frozen exploratory design.  ``leave_one_session_out`` is a
    deterministic sensitivity analysis.  Neither design changes the fixed model,
    aggregation rule, or decision threshold.
    """

    if outer_design not in SUPPORTED_OUTER_DESIGNS:
        raise OfflineSequenceEvaluationError(
            f"outer_design must be one of {SUPPORTED_OUTER_DESIGNS}"
        )
    prepared = _validate_inputs(
        records, visual_features, sensor_features, sample_ids=sample_ids
    )
    (
        np,
        _,
        _,
        StratifiedGroupKFold,
        LeaveOneGroupOut,
        *_,
    ) = _dependencies()
    labels = prepared["labels"]
    sessions = prepared["sessions"]
    participants = prepared["participants"]
    fusion = prepared["fusion"]
    unique_sessions = sorted(set(str(value) for value in sessions))

    if outer_design == "sgkf5":
        if len(unique_sessions) < FIXED_OUTER_SPLITS:
            raise OfflineSequenceEvaluationError(
                f"sgkf5 requires at least {FIXED_OUTER_SPLITS} sessions"
            )
        splitter = StratifiedGroupKFold(
            n_splits=FIXED_OUTER_SPLITS,
            shuffle=True,
            random_state=FIXED_RANDOM_STATE,
        )
        splits = list(splitter.split(fusion, labels, sessions))
    else:
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(fusion, labels, sessions))

    raw_probabilities = np.zeros((len(labels), 3), dtype=float)
    pooled_probabilities = np.zeros((len(labels), 3), dtype=float)
    pooled_predictions = np.full(len(labels), -1, dtype=int)
    decision_ratios = np.zeros(len(labels), dtype=float)
    fold_ids = np.zeros(len(labels), dtype=int)
    assigned = np.zeros(len(labels), dtype=bool)
    fold_reports: list[dict[str, Any]] = []
    sequence_reports: list[dict[str, Any]] = []
    identity_fields = (
        "session_id",
        "participant_id",
        "cohort_id",
        "activity_id",
        "site_id",
        "teacher_id",
    )

    for fold_index, (train_indices, test_indices) in enumerate(splits, 1):
        train_sessions = set(str(value) for value in sessions[train_indices])
        test_sessions = set(str(value) for value in sessions[test_indices])
        session_overlap = sorted(train_sessions & test_sessions)
        train_participants = set(str(value) for value in participants[train_indices])
        test_participants = set(str(value) for value in participants[test_indices])
        participant_overlap = sorted(train_participants & test_participants)
        if session_overlap or participant_overlap:
            raise OfflineSequenceEvaluationError(
                f"identity leakage in outer fold {fold_index}: "
                f"sessions={session_overlap}, participants={participant_overlap}"
            )
        if set(int(value) for value in labels[train_indices]) != {0, 1, 2}:
            raise OfflineSequenceEvaluationError(
                f"outer fold {fold_index} training data do not contain all fixed classes"
            )

        fold_raw_probabilities, convergence_warning = _fit_predict_probabilities(
            fusion[train_indices], labels[train_indices], fusion[test_indices]
        )
        (
            fold_pooled_probabilities,
            fold_pooled_predictions,
            fold_ratios,
            fold_sequences,
        ) = _pool_fold_probabilities(
            fold_raw_probabilities,
            sessions[test_indices],
            participants[test_indices],
        )
        raw_probabilities[test_indices] = fold_raw_probabilities
        pooled_probabilities[test_indices] = fold_pooled_probabilities
        pooled_predictions[test_indices] = fold_pooled_predictions
        decision_ratios[test_indices] = fold_ratios
        fold_ids[test_indices] = fold_index
        assigned[test_indices] = True
        sequence_reports.extend(
            [{**item, "outer_fold": fold_index} for item in fold_sequences]
        )

        identity_overlap = {
            field: sorted(
                _identity_values(prepared["records"], train_indices, field)
                & _identity_values(prepared["records"], test_indices, field)
            )
            for field in identity_fields
        }
        fold_reports.append(
            {
                "fold": fold_index,
                "train_sample_count": int(len(train_indices)),
                "test_sample_count": int(len(test_indices)),
                "train_session_ids": sorted(train_sessions),
                "test_session_ids": sorted(test_sessions),
                "train_participant_ids": sorted(train_participants),
                "test_participant_ids": sorted(test_participants),
                "session_overlap": session_overlap,
                "participant_overlap": participant_overlap,
                "identity_overlap": identity_overlap,
                "test_sequence_count": int(
                    len(
                        set(
                            zip(
                                sessions[test_indices].tolist(),
                                participants[test_indices].tolist(),
                            )
                        )
                    )
                ),
                "convergence_warning": convergence_warning,
                "raw_window_metrics": _hard_metrics(
                    labels[test_indices], fold_raw_probabilities.argmax(axis=1)
                ),
                "pooled_sequence_metrics": _hard_metrics(
                    labels[test_indices], fold_pooled_predictions
                ),
            }
        )

    if not assigned.all() or (fold_ids == 0).any():
        raise RuntimeError("outer evaluation did not assign every sample exactly once")

    raw_predictions = raw_probabilities.argmax(axis=1)
    oof_predictions = [
        {
            "sample_id": prepared["sample_ids"][index],
            "label": int(labels[index]),
            "session_id": str(sessions[index]),
            "participant_id": str(participants[index]),
            "sequence_id": _sequence_id(
                str(sessions[index]), str(participants[index])
            ),
            "outer_fold": int(fold_ids[index]),
            "raw_window_probabilities": [
                round(float(value), 8) for value in raw_probabilities[index]
            ],
            "raw_window_prediction": int(raw_predictions[index]),
            "pooled_sequence_probabilities": [
                round(float(value), 8) for value in pooled_probabilities[index]
            ],
            "pooled_sequence_prediction": int(pooled_predictions[index]),
            "minority_to_medium_ratio": round(float(decision_ratios[index]), 8),
        }
        for index in range(len(labels))
    ]

    claim_status = {
        "offline_sequence_accuracy_established": False,
        "offline_sequence_multimodal_gain_established": False,
        "session_disjoint_accuracy_established": False,
        "real_time_accuracy_established": False,
        "cross_cohort_accuracy_established": False,
        "cross_activity_accuracy_established": False,
        "deployment_accuracy_established": False,
    }
    return {
        "evaluation_kind": "post_selected_offline_participant_recording_oof",
        "available": True,
        "estimate_available": True,
        "exploratory_only": True,
        "post_selected_on_current_dataset": True,
        "post_selection_reason": (
            "the sequence aggregation and threshold were proposed after inspection of "
            "the current DIPSER study outputs"
        ),
        "contains_inferential_statistics": False,
        "offline_full_sequence_required": True,
        "real_time_compatible": False,
        "aggregation_uses_labels": False,
        "aggregation_boundary": ["session_id", "participant_id"],
        "aggregation_crosses_sequence_boundary": False,
        "feature_sample_order_verified": True,
        "outer_design": outer_design,
        "fold_count": int(len(splits)),
        "sample_count": int(len(labels)),
        "session_count": int(len(unique_sessions)),
        "participant_count": int(len(set(str(value) for value in participants))),
        "sequence_count": int(
            len(set(zip(sessions.tolist(), participants.tolist())))
        ),
        "feature_dimensions": prepared["feature_dimensions"],
        "fixed_model_protocol": {
            "feature_order": ["visual", "sensor"],
            "preprocessing": "StandardScaler fitted only on each outer training fold",
            "estimator": "sklearn.linear_model.LogisticRegression",
            "C": FIXED_LOGISTIC_C,
            "class_weight": FIXED_CLASS_WEIGHT,
            "solver": FIXED_SOLVER,
            "max_iter": FIXED_MAX_ITER,
            "random_state": FIXED_RANDOM_STATE,
            "outer_group_field": "session_id",
            "outer_splits": FIXED_OUTER_SPLITS if outer_design == "sgkf5" else None,
            "outer_splitter": (
                "StratifiedGroupKFold(shuffle=True, random_state=2026)"
                if outer_design == "sgkf5"
                else "LeaveOneGroupOut(session_id)"
            ),
            "sequence_key": ["session_id", "participant_id"],
            "probability_pooling": "per-class geometric mean within held-out sequence",
            "default_label": FIXED_DEFAULT_LABEL,
            "default_class_name": FIXED_CLASS_NAMES[FIXED_DEFAULT_LABEL],
            "minority_labels": list(FIXED_MINORITY_LABELS),
            "minority_to_medium_ratio_threshold": FIXED_MINORITY_TO_MEDIUM_RATIO,
            "hyperparameter_selection": "none; post-selected constants are fixed",
        },
        "raw_window_metrics": _hard_metrics(labels, raw_predictions),
        "pooled_sequence_metrics": _hard_metrics(labels, pooled_predictions),
        "folds": fold_reports,
        "sequence_predictions": sequence_reports,
        "oof_predictions": oof_predictions,
        "claim_status": claim_status,
        **claim_status,
    }


__all__ = [
    "OfflineSequenceEvaluationError",
    "offline_sequence_evaluation",
]
