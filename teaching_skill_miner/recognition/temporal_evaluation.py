"""Strict exploratory temporal evaluation for frozen DIPSER feature rows.

The estimator is intentionally fixed: concatenated visual and sensor features,
``StandardScaler``, and ``SVC(kernel="linear", C=0.1)``.  The only selected
quantity is a causal score-averaging window from ``(1, 2, 3, 5)``.  Selection
uses session-grouped inner out-of-fold predictions from the outer-training data
only.  No score from another participant/recording or from the future enters a
prediction.

This protocol was designed after inspection of earlier results.  Consequently
all outputs are explicitly post-selection exploratory and cannot establish
accuracy, multimodal gain, or deployment performance.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


STRATIFIED_SESSION_5FOLD = "stratified_session_5fold"
LEAVE_ONE_SESSION_OUT = "leave_one_session_out"
SUPPORTED_DESIGNS = (STRATIFIED_SESSION_5FOLD, LEAVE_ONE_SESSION_OUT)
CAUSAL_WINDOWS = (1, 2, 3, 5)
OUTER_SPLITS = 5
INNER_SPLITS = 5
FIXED_SEED = 2026
FIXED_LINEAR_SVC_C = 0.1
MAXIMUM_GAP_SECONDS = 60.0
AUDITED_IDENTITIES = (
    "session_id",
    "participant_id",
    "teacher_id",
    "cohort_id",
    "activity_id",
    "site_id",
)


class TemporalEvaluationError(ValueError):
    """Raised when data cannot support the fixed temporal protocol."""


def _dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
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
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
    }


def _activity_id(record: Mapping[str, Any], sample_id: str) -> str | None:
    raw_direct = record.get("activity_id")
    direct = "" if raw_direct is None else str(raw_direct).strip()
    source = record.get("source")
    nested = ""
    if isinstance(source, Mapping):
        raw_nested = source.get("experiment_id")
        nested = "" if raw_nested is None else str(raw_nested).strip()
    if direct and nested and direct != nested:
        raise TemporalEvaluationError(
            f"record {sample_id} has conflicting activity identifiers"
        )
    return direct or nested or None


def _identifier(
    record: Mapping[str, Any], field: str, sample_id: str, *, required: bool
) -> str | None:
    raw_value = record.get(field)
    value = "" if raw_value is None else str(raw_value).strip()
    if required and not value:
        raise TemporalEvaluationError(f"record {sample_id} lacks {field}")
    return value or None


def _timestamp(record: Mapping[str, Any], sample_id: str) -> float:
    source = record.get("source")
    if not isinstance(source, Mapping):
        raise TemporalEvaluationError(
            f"record {sample_id} lacks source.metadata_median_seconds_of_day"
        )
    try:
        value = float(source["metadata_median_seconds_of_day"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TemporalEvaluationError(
            f"record {sample_id} lacks a numeric metadata_median_seconds_of_day"
        ) from exc
    if not math.isfinite(value) or value < 0 or value >= 86_400:
        raise TemporalEvaluationError(
            f"record {sample_id} has invalid metadata_median_seconds_of_day"
        )
    return value


def _prepare_temporal_inputs(
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
        raise TemporalEvaluationError("records must be a non-empty sequence")

    normalized: list[Mapping[str, Any]] = []
    ids: list[str] = []
    labels: list[int] = []
    times: list[float] = []
    identities: dict[str, list[str | None]] = {
        field: [] for field in AUDITED_IDENTITIES
    }
    label_names: dict[int, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TemporalEvaluationError(f"record {index} is not an object")
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise TemporalEvaluationError(f"record {index} lacks sample_id")
        raw_label = record.get("label")
        if isinstance(raw_label, bool):
            raise TemporalEvaluationError(f"record {sample_id} has invalid label")
        try:
            label = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise TemporalEvaluationError(
                f"record {sample_id} lacks an integer label"
            ) from exc
        if isinstance(raw_label, float) and not raw_label.is_integer():
            raise TemporalEvaluationError(f"record {sample_id} has non-integer label")
        label_name = str(record.get("label_name", label)).strip()
        previous_name = label_names.setdefault(label, label_name)
        if previous_name != label_name:
            raise TemporalEvaluationError(
                f"label {label} maps to inconsistent label_name values"
            )

        ids.append(sample_id)
        labels.append(label)
        times.append(_timestamp(record, sample_id))
        identities["session_id"].append(
            _identifier(record, "session_id", sample_id, required=True)
        )
        identities["participant_id"].append(
            _identifier(record, "participant_id", sample_id, required=True)
        )
        for field in ("teacher_id", "cohort_id", "site_id"):
            identities[field].append(
                _identifier(record, field, sample_id, required=False)
            )
        identities["activity_id"].append(_activity_id(record, sample_id))
        normalized.append(record)

    if len(set(ids)) != len(ids):
        raise TemporalEvaluationError("record sample_id values must be unique")
    if [str(value) for value in sample_ids] != ids:
        raise TemporalEvaluationError(
            "sample_ids must exactly match record order"
        )
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise TemporalEvaluationError("at least two classes are required")

    matrices: dict[str, Any] = {}
    for name, values in (
        ("visual", visual_features),
        ("sensor", sensor_features),
    ):
        try:
            matrix = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TemporalEvaluationError(
                f"{name} features must be a numeric matrix"
            ) from exc
        if matrix.ndim != 2 or matrix.shape[0] != len(records) or matrix.shape[1] < 1:
            raise TemporalEvaluationError(
                f"{name} features must have shape ({len(records)}, positive_dimension)"
            )
        if not np.isfinite(matrix).all():
            raise TemporalEvaluationError(
                f"{name} features contain NaN or infinity"
            )
        matrices[name] = matrix
    fusion = np.concatenate([matrices["visual"], matrices["sensor"]], axis=1)

    identity_arrays = {
        field: np.asarray(values, dtype=object)
        for field, values in identities.items()
    }
    sessions = identity_arrays["session_id"]
    participants = identity_arrays["participant_id"]
    # Within a participant/recording stream, timestamps must identify a stable
    # deterministic order; sample_id is used only to break timestamp ties.
    stream_order_keys: set[tuple[str, str, float, str]] = set()
    for session, participant, timestamp, sample_id in zip(
        sessions, participants, times, ids
    ):
        key = (str(session), str(participant), float(timestamp), sample_id)
        if key in stream_order_keys:
            raise TemporalEvaluationError("duplicate temporal stream order key")
        stream_order_keys.add(key)

    return {
        "records": normalized,
        "sample_ids": ids,
        "labels": np.asarray(labels, dtype=int),
        "classes": classes,
        "label_names": {str(label): label_names[label] for label in classes},
        "times": np.asarray(times, dtype=float),
        "identities": identity_arrays,
        "features": fusion,
        "feature_dimensions": {
            "visual": int(matrices["visual"].shape[1]),
            "sensor": int(matrices["sensor"].shape[1]),
            "fusion": int(fusion.shape[1]),
        },
        "unique_sessions": sorted(set(str(value) for value in sessions)),
        "unique_participants": sorted(
            set(str(value) for value in participants)
        ),
    }


def _class_session_support(labels: Any, sessions: Any) -> dict[str, int]:
    dep = _dependencies()
    np = dep["np"]
    return {
        str(int(label)): int(len(np.unique(sessions[labels == label])))
        for label in np.unique(labels)
    }


def _assert_five_fold_support(labels: Any, sessions: Any, *, scope: str) -> None:
    dep = _dependencies()
    np = dep["np"]
    if len(np.unique(sessions)) < INNER_SPLITS:
        raise TemporalEvaluationError(
            f"{scope} requires at least {INNER_SPLITS} sessions"
        )
    support = _class_session_support(labels, sessions)
    weak = {
        label: count for label, count in support.items() if count < INNER_SPLITS
    }
    if weak:
        raise TemporalEvaluationError(
            f"{scope} class/session support is below {INNER_SPLITS}: {weak}"
        )


def _fit_linear_svc_scores(
    train_x: Any,
    train_y: Any,
    test_x: Any,
    classes: Sequence[int],
) -> Any:
    dep = _dependencies()
    np = dep["np"]
    if set(int(value) for value in train_y) != set(int(value) for value in classes):
        raise TemporalEvaluationError("linear-SVC training split is missing a class")
    pipeline = dep["Pipeline"](
        [
            ("training_only_standard_scaler", dep["StandardScaler"]()),
            (
                "fixed_linear_svc",
                dep["SVC"](
                    kernel="linear",
                    C=FIXED_LINEAR_SVC_C,
                    class_weight=None,
                    decision_function_shape="ovr",
                    break_ties=True,
                ),
            ),
        ]
    )
    pipeline.fit(train_x, train_y)
    scores = np.asarray(pipeline.decision_function(test_x), dtype=float)
    fitted_classes = [
        int(value) for value in pipeline.named_steps["fixed_linear_svc"].classes_
    ]
    if scores.ndim == 1:
        if len(fitted_classes) != 2:
            raise RuntimeError("one-dimensional decision scores require two classes")
        scores = np.column_stack([-scores, scores])
    if scores.shape != (len(test_x), len(fitted_classes)):
        raise RuntimeError("unexpected linear-SVC decision score shape")
    aligned = np.full((len(test_x), len(classes)), -np.inf, dtype=float)
    class_position = {int(label): index for index, label in enumerate(classes)}
    for fitted_position, label in enumerate(fitted_classes):
        aligned[:, class_position[label]] = scores[:, fitted_position]
    if not np.isfinite(aligned).all():
        raise RuntimeError("linear-SVC score alignment is incomplete")
    return aligned


def _score_predictions(scores: Any, classes: Sequence[int]) -> Any:
    dep = _dependencies()
    np = dep["np"]
    class_array = np.asarray(classes, dtype=int)
    return class_array[np.asarray(scores).argmax(axis=1)]


def _causal_average(
    scores: Any,
    indices: Any,
    data: Mapping[str, Any],
    *,
    window: int,
) -> tuple[Any, list[list[str]]]:
    """Average current/past scores inside one session-participant stream only."""

    dep = _dependencies()
    np = dep["np"]
    if window not in CAUSAL_WINDOWS:
        raise TemporalEvaluationError(
            f"window must be one of {CAUSAL_WINDOWS}"
        )
    scores = np.asarray(scores, dtype=float)
    indices = np.asarray(indices, dtype=int)
    if scores.ndim != 2 or scores.shape[0] != len(indices):
        raise TemporalEvaluationError("scores and indices do not align")
    smoothed = np.empty_like(scores)
    histories: list[list[str]] = [[] for _ in range(len(indices))]
    sessions = data["identities"]["session_id"]
    participants = data["identities"]["participant_id"]
    times = data["times"]
    sample_ids = data["sample_ids"]
    streams: dict[tuple[str, str], list[int]] = {}
    for local_position, global_index in enumerate(indices):
        key = (str(sessions[global_index]), str(participants[global_index]))
        streams.setdefault(key, []).append(local_position)
    for positions in streams.values():
        ordered = sorted(
            positions,
            key=lambda local_position: (
                float(times[indices[local_position]]),
                sample_ids[indices[local_position]],
            ),
        )
        current_segment: list[int] = []
        previous_time: float | None = None
        for local_position in ordered:
            current_time = float(times[indices[local_position]])
            if (
                previous_time is not None
                and not 0.0 <= current_time - previous_time <= MAXIMUM_GAP_SECONDS
            ):
                current_segment = []
            current_segment.append(local_position)
            history_positions = current_segment[-window:]
            smoothed[local_position] = np.mean(scores[history_positions], axis=0)
            histories[local_position] = [
                sample_ids[indices[history_position]]
                for history_position in history_positions
            ]
            previous_time = current_time
    return smoothed, histories


def _metrics(
    y_true: Any,
    y_pred: Any,
    classes: Sequence[int],
    label_names: Mapping[str, str],
) -> dict[str, Any]:
    dep = _dependencies()
    np = dep["np"]
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if not len(y_true):
        return {
            "sample_count": 0,
            "accuracy": None,
            "macro_f1": None,
            "confusion_matrix": [],
            "per_class": [],
        }
    precision, recall, f1, support = dep[
        "precision_recall_fscore_support"
    ](
        y_true,
        y_pred,
        labels=list(classes),
        zero_division=0,
    )
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
        "class_order": [int(value) for value in classes],
        "confusion_matrix": dep["confusion_matrix"](
            y_true, y_pred, labels=list(classes)
        ).astype(int).tolist(),
        "per_class": [
            {
                "label": int(label),
                "label_name": label_names[str(label)],
                "support": int(support[position]),
                "precision": round(float(precision[position]), 6),
                "recall": round(float(recall[position]), 6),
                "f1": round(float(f1[position]), 6),
            }
            for position, label in enumerate(classes)
        ],
        "predicted_class_counts": {
            str(int(label)): int(count)
            for label, count in sorted(Counter(int(value) for value in y_pred).items())
        },
    }


def _identity_overlap(
    data: Mapping[str, Any], train_indices: Any, test_indices: Any
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for field in AUDITED_IDENTITIES:
        values = data["identities"][field]
        train_values = {
            str(value) for value in values[train_indices] if value is not None
        }
        test_values = {
            str(value) for value in values[test_indices] if value is not None
        }
        overlap = sorted(train_values & test_values)
        audit[field] = {
            "field_complete": bool(all(value is not None for value in values)),
            "train_ids": sorted(train_values),
            "test_ids": sorted(test_values),
            "overlap_ids": overlap,
            "overlap_detected": bool(overlap),
        }
    return audit


def _inner_window_selection(
    data: Mapping[str, Any], train_indices: Any
) -> dict[str, Any]:
    dep = _dependencies()
    np = dep["np"]
    train_indices = np.asarray(train_indices, dtype=int)
    labels = data["labels"][train_indices]
    sessions = data["identities"]["session_id"][train_indices]
    classes = data["classes"]
    _assert_five_fold_support(labels, sessions, scope="inner window selection")
    splitter = dep["StratifiedGroupKFold"](
        n_splits=INNER_SPLITS, shuffle=True, random_state=FIXED_SEED
    )
    oof_scores = np.zeros((len(train_indices), len(classes)), dtype=float)
    assigned = np.zeros(len(train_indices), dtype=bool)
    fold_audits: list[dict[str, Any]] = []
    for inner_fold, (fit_local, validation_local) in enumerate(
        splitter.split(data["features"][train_indices], labels, sessions), 1
    ):
        fit_global = train_indices[fit_local]
        validation_global = train_indices[validation_local]
        overlap = _identity_overlap(data, fit_global, validation_global)
        if overlap["session_id"]["overlap_ids"]:
            raise TemporalEvaluationError("session leakage in inner CV")
        scores = _fit_linear_svc_scores(
            data["features"][fit_global],
            data["labels"][fit_global],
            data["features"][validation_global],
            classes,
        )
        oof_scores[validation_local] = scores
        assigned[validation_local] = True
        fold_audits.append(
            {
                "inner_fold": inner_fold,
                "train_sample_count": int(len(fit_global)),
                "validation_sample_count": int(len(validation_global)),
                "train_session_ids": overlap["session_id"]["train_ids"],
                "validation_session_ids": overlap["session_id"]["test_ids"],
                "session_overlap_ids": overlap["session_id"]["overlap_ids"],
                "participant_overlap_ids": overlap["participant_id"]["overlap_ids"],
            }
        )
    if not assigned.all():
        raise RuntimeError("inner CV did not assign every outer-training row")

    candidates: list[dict[str, Any]] = []
    smoothed_by_window: dict[int, Any] = {}
    for window in CAUSAL_WINDOWS:
        smoothed, _ = _causal_average(
            oof_scores, train_indices, data, window=window
        )
        predictions = _score_predictions(smoothed, classes)
        metrics = _metrics(
            labels, predictions, classes, data["label_names"]
        )
        smoothed_by_window[window] = smoothed
        candidates.append(
            {
                "window": window,
                "inner_oof_accuracy": metrics["accuracy"],
                "inner_oof_macro_f1": metrics["macro_f1"],
                "metrics": metrics,
            }
        )
    selected = sorted(
        candidates,
        key=lambda item: (
            -float(item["inner_oof_accuracy"]),
            -float(item["inner_oof_macro_f1"]),
            int(item["window"]),
        ),
    )[0]
    for item in candidates:
        item["selected"] = item["window"] == selected["window"]
    return {
        "selected_window": int(selected["window"]),
        "selection_rule": (
            "maximize inner session-grouped OOF Accuracy; then Macro-F1; "
            "then choose the shorter causal window"
        ),
        "candidate_windows": list(CAUSAL_WINDOWS),
        "window_candidates": candidates,
        "inner_folds": fold_audits,
        "inner_splits": INNER_SPLITS,
        "inner_seed": FIXED_SEED,
        "outer_test_labels_used": False,
    }


def _outer_specs(data: Mapping[str, Any], design: str) -> list[dict[str, Any]]:
    dep = _dependencies()
    np = dep["np"]
    labels = data["labels"]
    sessions = data["identities"]["session_id"]
    all_indices = np.arange(len(labels), dtype=int)
    if design == STRATIFIED_SESSION_5FOLD:
        _assert_five_fold_support(labels, sessions, scope="outer session CV")
        splitter = dep["StratifiedGroupKFold"](
            n_splits=OUTER_SPLITS, shuffle=True, random_state=FIXED_SEED
        )
        return [
            {
                "fold_id": f"session-fold-{fold}",
                "train_indices": train_indices,
                "test_indices": test_indices,
            }
            for fold, (train_indices, test_indices) in enumerate(
                splitter.split(data["features"], labels, sessions), 1
            )
        ]
    if design == LEAVE_ONE_SESSION_OUT:
        return [
            {
                "fold_id": f"session={session}",
                "train_indices": all_indices[sessions != session],
                "test_indices": all_indices[sessions == session],
            }
            for session in sorted(set(str(value) for value in sessions))
        ]
    raise TemporalEvaluationError(f"unsupported temporal design: {design}")


def _evaluate_outer_fold(
    data: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    fold_number: int,
) -> dict[str, Any]:
    """Evaluate one fold; test labels are read only after selection/prediction."""

    dep = _dependencies()
    np = dep["np"]
    train_indices = np.asarray(spec["train_indices"], dtype=int)
    test_indices = np.asarray(spec["test_indices"], dtype=int)
    if not len(train_indices) or not len(test_indices):
        raise TemporalEvaluationError("outer fold has an empty train or test set")
    overlap = _identity_overlap(data, train_indices, test_indices)
    if overlap["session_id"]["overlap_ids"]:
        raise TemporalEvaluationError(
            f"outer fold {spec['fold_id']} has session leakage"
        )
    train_sample_ids = {data["sample_ids"][index] for index in train_indices}
    test_sample_ids = {data["sample_ids"][index] for index in test_indices}
    sample_overlap = sorted(train_sample_ids & test_sample_ids)
    if sample_overlap:
        raise TemporalEvaluationError(
            f"outer fold {spec['fold_id']} has sample leakage"
        )

    # Selection and estimator fitting end before y_test is accessed below.
    selection = _inner_window_selection(data, train_indices)
    raw_scores = _fit_linear_svc_scores(
        data["features"][train_indices],
        data["labels"][train_indices],
        data["features"][test_indices],
        data["classes"],
    )
    raw_predictions = _score_predictions(raw_scores, data["classes"])
    selected_scores, histories = _causal_average(
        raw_scores,
        test_indices,
        data,
        window=selection["selected_window"],
    )
    selected_predictions = _score_predictions(
        selected_scores, data["classes"]
    )
    majority_label = sorted(
        Counter(int(value) for value in data["labels"][train_indices]).items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0]
    majority_predictions = np.full(len(test_indices), majority_label, dtype=int)

    # Outer labels are used from this point onward only for descriptive scoring.
    y_test = data["labels"][test_indices]
    return {
        "fold_number": fold_number,
        "fold_id": str(spec["fold_id"]),
        "train_indices": train_indices.astype(int).tolist(),
        "test_indices": test_indices.astype(int).tolist(),
        "train_sample_count": int(len(train_indices)),
        "test_sample_count": int(len(test_indices)),
        "train_class_counts": {
            str(int(label)): int(count)
            for label, count in sorted(
                Counter(int(value) for value in data["labels"][train_indices]).items()
            )
        },
        "test_class_counts": {
            str(int(label)): int(count)
            for label, count in sorted(Counter(int(value) for value in y_test).items())
        },
        "train_session_ids": overlap["session_id"]["train_ids"],
        "test_session_ids": overlap["session_id"]["test_ids"],
        "session_overlap_ids": overlap["session_id"]["overlap_ids"],
        "train_participant_ids": overlap["participant_id"]["train_ids"],
        "test_participant_ids": overlap["participant_id"]["test_ids"],
        "participant_overlap_ids": overlap["participant_id"]["overlap_ids"],
        "sample_id_overlap": sample_overlap,
        "identity_overlap_audit": overlap,
        "selected_causal_window": selection["selected_window"],
        "inner_window_selection": selection,
        "majority_label_selected_from_outer_training_only": int(majority_label),
        "raw_metrics": _metrics(
            y_test, raw_predictions, data["classes"], data["label_names"]
        ),
        "selected_causal_metrics": _metrics(
            y_test,
            selected_predictions,
            data["classes"],
            data["label_names"],
        ),
        "fold_training_majority_metrics": _metrics(
            y_test,
            majority_predictions,
            data["classes"],
            data["label_names"],
        ),
        "raw_predictions": raw_predictions.astype(int).tolist(),
        "selected_causal_predictions": selected_predictions.astype(int).tolist(),
        "majority_predictions": majority_predictions.astype(int).tolist(),
        "raw_scores": raw_scores.astype(float).tolist(),
        "selected_causal_scores": selected_scores.astype(float).tolist(),
        "causal_history_sample_ids": histories,
        "outer_test_labels_used_for_window_or_model_selection": False,
        "status": "evaluated",
    }


def _evaluate_design(data: Mapping[str, Any], design: str) -> dict[str, Any]:
    dep = _dependencies()
    np = dep["np"]
    specs = _outer_specs(data, design)
    fold_reports: list[dict[str, Any]] = []
    assigned = np.zeros(len(data["records"]), dtype=bool)
    fold_ids: list[str | None] = [None] * len(data["records"])
    selected_windows: list[int | None] = [None] * len(data["records"])
    histories: list[list[str] | None] = [None] * len(data["records"])
    outputs = {
        "raw": np.zeros(len(data["records"]), dtype=int),
        "selected_causal": np.zeros(len(data["records"]), dtype=int),
        "fold_training_majority": np.zeros(len(data["records"]), dtype=int),
    }
    score_outputs = {
        "raw": np.zeros((len(data["records"]), len(data["classes"])), dtype=float),
        "selected_causal": np.zeros(
            (len(data["records"]), len(data["classes"])), dtype=float
        ),
    }
    for fold_number, spec in enumerate(specs, 1):
        report = _evaluate_outer_fold(data, spec, fold_number=fold_number)
        test_indices = np.asarray(report["test_indices"], dtype=int)
        if assigned[test_indices].any():
            raise RuntimeError("an outer sample was scored more than once")
        assigned[test_indices] = True
        outputs["raw"][test_indices] = report["raw_predictions"]
        outputs["selected_causal"][test_indices] = report[
            "selected_causal_predictions"
        ]
        outputs["fold_training_majority"][test_indices] = report[
            "majority_predictions"
        ]
        score_outputs["raw"][test_indices] = report["raw_scores"]
        score_outputs["selected_causal"][test_indices] = report[
            "selected_causal_scores"
        ]
        for local_position, global_index in enumerate(test_indices):
            fold_ids[global_index] = report["fold_id"]
            selected_windows[global_index] = report["selected_causal_window"]
            histories[global_index] = report["causal_history_sample_ids"][
                local_position
            ]
        fold_reports.append(report)
    if not assigned.all():
        raise RuntimeError("outer temporal evaluation did not score every sample")

    metrics = {
        name: _metrics(
            data["labels"], prediction, data["classes"], data["label_names"]
        )
        for name, prediction in outputs.items()
    }
    oof_rows: list[dict[str, Any]] = []
    sessions = data["identities"]["session_id"]
    participants = data["identities"]["participant_id"]
    for index, sample_id in enumerate(data["sample_ids"]):
        oof_rows.append(
            {
                "sample_id": sample_id,
                "label": int(data["labels"][index]),
                "fold_id": fold_ids[index],
                "session_id": str(sessions[index]),
                "participant_id": str(participants[index]),
                "metadata_median_seconds_of_day": round(
                    float(data["times"][index]), 6
                ),
                "selected_causal_window": selected_windows[index],
                "causal_history_sample_ids": histories[index],
                "raw_prediction": int(outputs["raw"][index]),
                "selected_causal_prediction": int(
                    outputs["selected_causal"][index]
                ),
                "fold_training_majority_prediction": int(
                    outputs["fold_training_majority"][index]
                ),
                "raw_scores": [
                    round(float(value), 8) for value in score_outputs["raw"][index]
                ],
                "selected_causal_scores": [
                    round(float(value), 8)
                    for value in score_outputs["selected_causal"][index]
                ],
            }
        )

    def delta(candidate: str, reference: str) -> dict[str, float]:
        return {
            metric: round(
                float(metrics[candidate][metric] - metrics[reference][metric]), 6
            )
            for metric in ("accuracy", "macro_f1")
        }

    return {
        "design": design,
        "fold_count": len(fold_reports),
        "evaluated_fold_count": len(fold_reports),
        "oof_coverage": 1.0,
        "metrics": metrics,
        "descriptive_deltas": {
            "selected_causal_minus_raw": delta("selected_causal", "raw"),
            "raw_minus_fold_training_majority": delta(
                "raw", "fold_training_majority"
            ),
            "selected_causal_minus_fold_training_majority": delta(
                "selected_causal", "fold_training_majority"
            ),
        },
        "fold_selected_windows": [
            {
                "fold_id": fold["fold_id"],
                "window": fold["selected_causal_window"],
            }
            for fold in fold_reports
        ],
        "folds": fold_reports,
        "oof_predictions": oof_rows,
        "post_selection_exploratory": True,
        "contains_inferential_statistics": False,
        "claim_status": {
            "accuracy_established": False,
            "temporal_gain_established": False,
            "multimodal_gain_established": False,
            "session_disjoint_accuracy_established": False,
            "participant_disjoint_accuracy_established": False,
            "deployment_accuracy_established": False,
        },
    }


def temporal_linear_svc_evaluation(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    sample_ids: Sequence[str],
    *,
    designs: Sequence[str] = (
        STRATIFIED_SESSION_5FOLD,
        LEAVE_ONE_SESSION_OUT,
    ),
) -> dict[str, Any]:
    """Run the fixed nested temporal protocol for one or both outer designs."""

    requested = tuple(str(value) for value in designs)
    if not requested:
        raise TemporalEvaluationError("designs must not be empty")
    if len(set(requested)) != len(requested):
        raise TemporalEvaluationError("designs must be unique")
    unsupported = [name for name in requested if name not in SUPPORTED_DESIGNS]
    if unsupported:
        raise TemporalEvaluationError(
            "unsupported temporal designs: " + ", ".join(unsupported)
        )
    data = _prepare_temporal_inputs(
        records, visual_features, sensor_features, sample_ids
    )
    evaluations = {
        design: _evaluate_design(data, design) for design in requested
    }
    return {
        "protocol": "post_selection_temporal_linear_svc_nested_oof_v1",
        "evaluation_kind": "post_selection_exploratory_not_confirmatory",
        "post_selection_exploratory": True,
        "sample_count": len(data["records"]),
        "class_labels": data["classes"],
        "label_names": data["label_names"],
        "feature_dimensions": data["feature_dimensions"],
        "feature_sample_order_verified": True,
        "fusion_definition": "ordered_concatenation_of_frozen_visual_and_sensor_features",
        "feature_policy": {
            "records_converted_to_estimator_features": False,
            "allowed_estimator_inputs": [
                "frozen_visual_numeric_matrix",
                "frozen_sensor_numeric_matrix",
            ],
            "forbidden_estimator_inputs": [
                "session_or_participant_identity",
                "cohort_activity_teacher_or_site_identity",
                "file_archive_or_member_paths_and_names",
                "sample_ids_or_timestamps",
                "ground_truth_or_annotator_fields",
                "features_derived_from_any_forbidden_field",
            ],
            "timestamp_and_sample_id_use": "causal ordering only; never estimator columns",
            "caller_attestation_required": True,
        },
        "fixed_model": {
            "pipeline": ["StandardScaler", "SVC"],
            "kernel": "linear",
            "C": FIXED_LINEAR_SVC_C,
            "class_weight": None,
            "decision_function_shape": "ovr",
            "break_ties": True,
            "preprocessing_fitted_inside_each_training_fold": True,
        },
        "temporal_protocol": {
            "candidate_windows": list(CAUSAL_WINDOWS),
            "window_unit": "current plus preceding complete-case samples",
            "stream_key": ["session_id", "participant_id"],
            "ordering": [
                "source.metadata_median_seconds_of_day",
                "sample_id_tie_break_only",
            ],
            "score_aggregation": "unweighted_causal_arithmetic_mean",
            "maximum_gap_seconds": MAXIMUM_GAP_SECONDS,
            "gap_reset_rule": (
                "clear stream history when adjacent valid timestamps differ by "
                "less than 0 or more than 60 seconds"
            ),
            "cross_session_smoothing": False,
            "cross_participant_smoothing": False,
            "future_score_use": False,
            "ordering_fields_used_as_estimator_features": False,
        },
        "nested_selection": {
            "inner_splits": INNER_SPLITS,
            "inner_group_field": "session_id",
            "seed": FIXED_SEED,
            "objective": "Accuracy then Macro-F1 then shorter window",
            "outer_test_labels_used": False,
        },
        "outer_protocols": {
            STRATIFIED_SESSION_5FOLD: {
                "splitter": "StratifiedGroupKFold",
                "splits": OUTER_SPLITS,
                "group_field": "session_id",
                "shuffle": True,
                "seed": FIXED_SEED,
            },
            LEAVE_ONE_SESSION_OUT: {
                "splitter": "deterministic sorted LeaveOneGroupOut equivalent",
                "group_field": "session_id",
                "shuffle": False,
            },
        },
        "designs": list(requested),
        "evaluations": evaluations,
        "contains_inferential_statistics": False,
        "claim_status": {
            "accuracy_established": False,
            "temporal_gain_established": False,
            "multimodal_gain_established": False,
            "cross_session_accuracy_established": False,
            "deployment_accuracy_established": False,
        },
        "claim_limitation": (
            "The fixed model and temporal protocol were added after earlier DIPSER "
            "results had been inspected. These nested OOF values are descriptive "
            "development results; confirmation requires a newly frozen external "
            "prospective dataset."
        ),
    }
