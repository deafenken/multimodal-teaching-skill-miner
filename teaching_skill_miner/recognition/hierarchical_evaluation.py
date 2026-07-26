"""Nested evaluation of an offline hierarchical classroom-context candidate.

This module evaluates a deliberately narrow, post-selected candidate on frozen
DIPSER feature rows.  Three prediction levels are combined:

* a participant-recording probability pool from a window logistic regression;
* a conservative low-class confirmation vote from a window linear SVC; and
* a whole-session context classifier trained from session median/IQR summaries.

Every estimator and every scaler is refitted inside each inner or outer training
partition.  Held-out labels are never used to aggregate a sequence or session.
The session component nevertheless consumes *all unlabeled feature rows* in the
held-out classroom recording, including other participants and future windows.
It is therefore a transductive, offline batch estimate -- never a real-time,
single-participant, inductive deployment estimate.

The method and its compact gate registry were proposed after inspecting this
development dataset.  Nested cross-validation prevents direct outer-label
selection within a run, but it cannot undo dataset-level post-selection.
Consequently every establishment claim returned by this module remains false.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


SGKF5 = "sgkf5"
LEAVE_ONE_SESSION_OUT = "leave_one_session_out"
SUPPORTED_OUTER_DESIGNS = (SGKF5, LEAVE_ONE_SESSION_OUT)
LEAVE_ONE_COHORT_OUT = "leave_one_cohort_out"
LEAVE_ONE_ACTIVITY_OUT = "leave_one_activity_out"
DOUBLE_BLOCKED_COHORT_ACTIVITY = "double_blocked_cohort_activity"
SUPPORTED_STRESS_DESIGNS = (
    LEAVE_ONE_COHORT_OUT,
    LEAVE_ONE_ACTIVITY_OUT,
    DOUBLE_BLOCKED_COHORT_ACTIVITY,
)
SUPPORTED_FEATURE_MODES = ("visual", "sensor", "fusion")
FIXED_CLASSES = (0, 1, 2)
CLASS_NAMES = ("low", "medium", "high")
DEFAULT_CLASS = 1
LOW_CLASS = 0
HIGH_CLASS = 2
OUTER_SPLITS = 5
INNER_SPLITS = 5
FIXED_SEED = 2026
LOGISTIC_C = 0.1
MINORITY_TO_MEDIUM_RATIO = 3.0
LINEAR_SVC_C = 0.1
SESSION_RBF_C = 3.0
SESSION_RBF_GAMMA = 0.003

# Ordering is the final deterministic selection tie-break.  It is intentionally
# explicit: no candidate is generated from an outer score.
GATE_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "gate_id": "sequence_base",
        "low_confirmation": False,
        "session_high_override": False,
        "ambiguous_high_ratio_threshold": None,
    },
    {
        "gate_id": "session_high_override",
        "low_confirmation": False,
        "session_high_override": True,
        "ambiguous_high_ratio_threshold": None,
    },
    {
        "gate_id": "low_confirmation",
        "low_confirmation": True,
        "session_high_override": False,
        "ambiguous_high_ratio_threshold": None,
    },
    {
        "gate_id": "high_override_plus_low_confirmation",
        "low_confirmation": True,
        "session_high_override": True,
        "ambiguous_high_ratio_threshold": None,
    },
    *tuple(
        {
            "gate_id": f"high_low_plus_ambiguous_guard_tau_{threshold:.1f}",
            "low_confirmation": True,
            "session_high_override": True,
            "ambiguous_high_ratio_threshold": threshold,
        }
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    ),
)


class HierarchicalEvaluationError(ValueError):
    """Raised when the fixed protocol cannot be evaluated safely."""


def _dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
        from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError(
            "install optional dependencies with `pip install -e '.[recognition]'`"
        ) from exc
    return {
        "np": np,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "LeaveOneGroupOut": LeaveOneGroupOut,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
    }


def gate_registry() -> list[dict[str, Any]]:
    """Return a JSON-safe copy of the frozen within-run gate registry."""

    return [dict(candidate) for candidate in GATE_REGISTRY]


def _required_identifier(
    record: Mapping[str, Any], field: str, *, sample_id: str
) -> str:
    value = str(record.get(field, "")).strip()
    if not value:
        raise HierarchicalEvaluationError(f"record {sample_id} lacks {field}")
    return value


def _optional_identifier(record: Mapping[str, Any], field: str) -> str | None:
    raw = record.get(field)
    value = "" if raw is None else str(raw).strip()
    return value or None


def _activity_identifier(record: Mapping[str, Any], sample_id: str) -> str | None:
    direct = _optional_identifier(record, "activity_id")
    source = record.get("source")
    nested: str | None = None
    if isinstance(source, Mapping):
        raw_nested = source.get("experiment_id")
        text = "" if raw_nested is None else str(raw_nested).strip()
        nested = text or None
    if direct and nested and direct != nested:
        raise HierarchicalEvaluationError(
            f"record {sample_id} has conflicting activity identifiers"
        )
    return direct or nested


def _prepare_inputs(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    sample_ids: Sequence[str],
    *,
    feature_mode: str = "fusion",
) -> dict[str, Any]:
    dep = _dependencies()
    np = dep["np"]
    if feature_mode not in SUPPORTED_FEATURE_MODES:
        raise HierarchicalEvaluationError(
            f"feature_mode must be one of {SUPPORTED_FEATURE_MODES}"
        )
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
    ):
        raise HierarchicalEvaluationError("records must be a non-empty sequence")

    ids: list[str] = []
    labels: list[int] = []
    identities: dict[str, list[str | None]] = {
        field: []
        for field in (
            "session_id",
            "participant_id",
            "teacher_id",
            "cohort_id",
            "activity_id",
            "site_id",
        )
    }
    normalized: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise HierarchicalEvaluationError(f"record {index} is not an object")
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise HierarchicalEvaluationError(f"record {index} lacks sample_id")
        raw_label = record.get("label")
        if isinstance(raw_label, bool):
            raise HierarchicalEvaluationError(f"record {sample_id} has invalid label")
        try:
            label = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise HierarchicalEvaluationError(
                f"record {sample_id} lacks an integer label"
            ) from exc
        if label not in FIXED_CLASSES:
            raise HierarchicalEvaluationError(
                f"record {sample_id} label must be one of {FIXED_CLASSES}"
            )
        ids.append(sample_id)
        labels.append(label)
        identities["session_id"].append(
            _required_identifier(record, "session_id", sample_id=sample_id)
        )
        identities["participant_id"].append(
            _required_identifier(record, "participant_id", sample_id=sample_id)
        )
        for field in ("teacher_id", "cohort_id", "site_id"):
            identities[field].append(_optional_identifier(record, field))
        identities["activity_id"].append(_activity_identifier(record, sample_id))
        normalized.append(record)

    if len(set(ids)) != len(ids):
        raise HierarchicalEvaluationError("record sample_id values must be unique")
    observed_ids = [str(value) for value in sample_ids]
    if observed_ids != ids:
        mismatch = next(
            (
                position
                for position, (observed, expected) in enumerate(zip(observed_ids, ids))
                if observed != expected
            ),
            min(len(observed_ids), len(ids)),
        )
        raise HierarchicalEvaluationError(
            "sample_ids must exactly match record order; "
            f"first mismatch is at row {mismatch}"
        )
    if sorted(set(labels)) != list(FIXED_CLASSES):
        raise HierarchicalEvaluationError(
            "the fixed candidate requires labels 0=low, 1=medium, and 2=high"
        )

    arrays = {
        field: np.asarray(values, dtype=object)
        for field, values in identities.items()
    }
    participant_sessions: dict[str, set[str]] = {}
    for session, participant in zip(
        arrays["session_id"], arrays["participant_id"]
    ):
        participant_sessions.setdefault(str(participant), set()).add(str(session))
    repeated = {
        participant: sorted(session_values)
        for participant, session_values in participant_sessions.items()
        if len(session_values) > 1
    }
    if repeated:
        raise HierarchicalEvaluationError(
            "session grouping would leak participant identity because participants "
            f"occur in multiple sessions: {repeated}"
        )

    matrices: dict[str, Any] = {}
    for name, values in (("visual", visual_features), ("sensor", sensor_features)):
        try:
            matrix = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise HierarchicalEvaluationError(
                f"{name} features must be a numeric matrix"
            ) from exc
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] < 1:
            raise HierarchicalEvaluationError(
                f"{name} features must have shape ({len(ids)}, positive_dimension)"
            )
        if not np.isfinite(matrix).all():
            raise HierarchicalEvaluationError(
                f"{name} features contain NaN or infinity"
            )
        matrices[name] = matrix
    available_fusion = np.concatenate(
        [matrices["visual"], matrices["sensor"]], axis=1
    )
    model_features = (
        available_fusion if feature_mode == "fusion" else matrices[feature_mode]
    )

    # A recording/session must not straddle design cells; otherwise group-blocked
    # audits and session-level supervision are ill-defined.
    session_cells: dict[str, tuple[str | None, str | None, str | None]] = {}
    for session, cohort, activity, site in zip(
        arrays["session_id"],
        arrays["cohort_id"],
        arrays["activity_id"],
        arrays["site_id"],
    ):
        cell = (
            None if cohort is None else str(cohort),
            None if activity is None else str(activity),
            None if site is None else str(site),
        )
        previous = session_cells.setdefault(str(session), cell)
        if previous != cell:
            raise HierarchicalEvaluationError(
                f"session_id {session} maps to multiple cohort/activity/site cells"
            )

    return {
        "records": normalized,
        "sample_ids": ids,
        "labels": np.asarray(labels, dtype=int),
        "identities": arrays,
        # ``fusion`` is retained as the internal estimator-matrix key so the
        # split/model helpers stay compact.  ``feature_mode`` and dimensions
        # below make its public meaning unambiguous for unimodal ablations.
        "fusion": model_features,
        "feature_mode": feature_mode,
        "feature_dimensions": {
            "visual": int(matrices["visual"].shape[1]),
            "sensor": int(matrices["sensor"].shape[1]),
            "available_fusion": int(available_fusion.shape[1]),
            "model_input": int(model_features.shape[1]),
            "session_summary": int(2 * model_features.shape[1]),
        },
    }


def _majority_label(values: Any) -> int:
    """Deterministic majority with an explicit medium-first tie-break."""

    counts = Counter(int(value) for value in values)
    if not counts:
        raise HierarchicalEvaluationError("cannot take a majority of zero labels")
    maximum = max(counts.values())
    tied = {label for label, count in counts.items() if count == maximum}
    for label in (DEFAULT_CLASS, LOW_CLASS, HIGH_CLASS):
        if label in tied:
            return label
    raise RuntimeError("unreachable majority tie-break")


def _hard_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    dep = _dependencies()
    precision, recall, per_class_f1, support = dep[
        "precision_recall_fscore_support"
    ](y_true, y_pred, labels=list(FIXED_CLASSES), zero_division=0)
    return {
        "sample_count": int(len(y_true)),
        "accuracy": round(float(dep["accuracy_score"](y_true, y_pred)), 6),
        "balanced_accuracy": round(float(sum(recall) / len(FIXED_CLASSES)), 6),
        "macro_f1": round(
            float(
                dep["f1_score"](
                    y_true,
                    y_pred,
                    labels=list(FIXED_CLASSES),
                    average="macro",
                    zero_division=0,
                )
            ),
            6,
        ),
        "weighted_f1": round(
            float(
                dep["f1_score"](
                    y_true,
                    y_pred,
                    labels=list(FIXED_CLASSES),
                    average="weighted",
                    zero_division=0,
                )
            ),
            6,
        ),
        "confusion_matrix": dep["confusion_matrix"](
            y_true, y_pred, labels=list(FIXED_CLASSES)
        ).astype(int).tolist(),
        "per_class": {
            name: {
                "label": label,
                "precision": round(float(precision[label]), 6),
                "recall": round(float(recall[label]), 6),
                "f1": round(float(per_class_f1[label]), 6),
                "support": int(support[label]),
            }
            for label, name in enumerate(CLASS_NAMES)
        },
    }


def _aligned_probabilities(model: Any, matrix: Any) -> Any:
    dep = _dependencies()
    np = dep["np"]
    aligned = np.zeros((len(matrix), len(FIXED_CLASSES)), dtype=float)
    raw = model.predict_proba(matrix)
    for source_column, label in enumerate(model.classes_):
        aligned[:, int(label)] = raw[:, source_column]
    return aligned


def _fit_window_components(
    train_x: Any, train_y: Any, test_x: Any
) -> tuple[Any, Any]:
    dep = _dependencies()
    if set(int(value) for value in train_y) != set(FIXED_CLASSES):
        raise HierarchicalEvaluationError("a training split is missing a window class")

    logistic_scaler = dep["StandardScaler"]()
    logistic_model = dep["LogisticRegression"](
        C=LOGISTIC_C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=4000,
        random_state=FIXED_SEED,
    )
    logistic_model.fit(logistic_scaler.fit_transform(train_x), train_y)
    logistic_probabilities = _aligned_probabilities(
        logistic_model, logistic_scaler.transform(test_x)
    )

    svc_scaler = dep["StandardScaler"]()
    svc_model = dep["SVC"](kernel="linear", C=LINEAR_SVC_C)
    svc_model.fit(svc_scaler.fit_transform(train_x), train_y)
    svc_predictions = svc_model.predict(svc_scaler.transform(test_x)).astype(int)
    return logistic_probabilities, svc_predictions


def _sequence_components(
    window_probabilities: Any,
    window_svc_predictions: Any,
    sessions: Any,
    participants: Any,
) -> dict[str, Any]:
    dep = _dependencies()
    np = dep["np"]
    sequence_probabilities = np.zeros_like(window_probabilities, dtype=float)
    base_predictions = np.full(len(window_probabilities), -1, dtype=int)
    svc_votes = np.full(len(window_probabilities), -1, dtype=int)
    sequence_ids = np.empty(len(window_probabilities), dtype=object)
    reports: list[dict[str, Any]] = []
    keys = list(dict.fromkeys(zip(sessions.tolist(), participants.tolist())))
    for session, participant in keys:
        mask = (sessions == session) & (participants == participant)
        indices = np.flatnonzero(mask)
        log_mean = np.log(np.clip(window_probabilities[indices], 1e-12, 1.0)).mean(
            axis=0
        )
        pooled = np.exp(log_mean - log_mean.max())
        pooled /= pooled.sum()
        minority = LOW_CLASS if pooled[LOW_CLASS] >= pooled[HIGH_CLASS] else HIGH_CLASS
        ratio = float(pooled[minority] / max(pooled[DEFAULT_CLASS], 1e-12))
        base = minority if ratio >= MINORITY_TO_MEDIUM_RATIO else DEFAULT_CLASS
        vote = _majority_label(window_svc_predictions[indices])
        sequence_id = f"session={session}|participant={participant}"
        sequence_probabilities[indices] = pooled
        base_predictions[indices] = int(base)
        svc_votes[indices] = int(vote)
        sequence_ids[indices] = sequence_id
        reports.append(
            {
                "sequence_id": sequence_id,
                "session_id": str(session),
                "participant_id": str(participant),
                "sample_count": int(len(indices)),
                "pooled_probabilities": [round(float(value), 8) for value in pooled],
                "minority_to_medium_ratio": round(ratio, 8),
                "base_prediction": int(base),
                "linear_svc_sequence_vote": int(vote),
            }
        )
    if (base_predictions < 0).any() or (svc_votes < 0).any():
        raise RuntimeError("sequence aggregation did not assign every row")
    return {
        "probabilities": sequence_probabilities,
        "base_predictions": base_predictions,
        "svc_votes": svc_votes,
        "sequence_ids": sequence_ids,
        "reports": reports,
    }


def _session_summary(matrix: Any) -> Any:
    dep = _dependencies()
    np = dep["np"]
    median = np.median(matrix, axis=0)
    iqr = np.percentile(matrix, 75.0, axis=0) - np.percentile(
        matrix, 25.0, axis=0
    )
    return np.concatenate([median, iqr])


def _session_components(
    data: Mapping[str, Any], train_indices: Any, test_indices: Any
) -> dict[str, Any]:
    dep = _dependencies()
    np = dep["np"]
    labels = data["labels"]
    sessions = data["identities"]["session_id"]
    fusion = data["fusion"]
    train_sessions = list(dict.fromkeys(sessions[train_indices].tolist()))
    test_sessions = list(dict.fromkeys(sessions[test_indices].tolist()))
    train_rows = []
    train_targets = []
    for session in train_sessions:
        indices = train_indices[sessions[train_indices] == session]
        train_rows.append(_session_summary(fusion[indices]))
        train_targets.append(_majority_label(labels[indices]))
    test_rows = []
    for session in test_sessions:
        indices = test_indices[sessions[test_indices] == session]
        test_rows.append(_session_summary(fusion[indices]))
    train_matrix = np.asarray(train_rows, dtype=float)
    test_matrix = np.asarray(test_rows, dtype=float)
    train_y = np.asarray(train_targets, dtype=int)
    if len(set(train_targets)) < 2:
        raise HierarchicalEvaluationError(
            "session-context training requires at least two majority classes"
        )
    scaler = dep["StandardScaler"]()
    model = dep["SVC"](
        kernel="rbf",
        C=SESSION_RBF_C,
        gamma=SESSION_RBF_GAMMA,
        probability=True,
        random_state=FIXED_SEED,
    )
    model.fit(scaler.fit_transform(train_matrix), train_y)
    probabilities_by_session = _aligned_probabilities(
        model, scaler.transform(test_matrix)
    )
    # SVC probability calibration is a separate Platt-scaling layer and its
    # argmax is not guaranteed to equal the classifier decision.  The hard
    # session state must therefore come from ``predict``; calibrated
    # probabilities are used only by the optional ambiguous-high guard.
    predictions_by_session = model.predict(scaler.transform(test_matrix)).astype(int)
    row_probabilities = np.zeros((len(test_indices), len(FIXED_CLASSES)), dtype=float)
    row_predictions = np.full(len(test_indices), -1, dtype=int)
    reports: list[dict[str, Any]] = []
    test_session_array = sessions[test_indices]
    for position, session in enumerate(test_sessions):
        local_indices = np.flatnonzero(test_session_array == session)
        row_probabilities[local_indices] = probabilities_by_session[position]
        row_predictions[local_indices] = predictions_by_session[position]
        reports.append(
            {
                "session_id": str(session),
                "sample_count": int(len(local_indices)),
                "probabilities": [
                    round(float(value), 8)
                    for value in probabilities_by_session[position]
                ],
                "prediction": int(predictions_by_session[position]),
            }
        )
    if (row_predictions < 0).any():
        raise RuntimeError("session aggregation did not assign every row")
    return {
        "probabilities": row_probabilities,
        "predictions": row_predictions,
        "reports": reports,
        "train_session_majority_counts": {
            str(label): int(count)
            for label, count in sorted(Counter(train_targets).items())
        },
    }


def _fit_component_predictions(
    data: Mapping[str, Any], train_indices: Any, test_indices: Any
) -> dict[str, Any]:
    dep = _dependencies()
    np = dep["np"]
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    sessions = data["identities"]["session_id"]
    participants = data["identities"]["participant_id"]
    overlap = set(str(value) for value in sessions[train_indices]) & set(
        str(value) for value in sessions[test_indices]
    )
    if overlap:
        raise HierarchicalEvaluationError(
            f"component fit has train/test session overlap: {sorted(overlap)}"
        )
    window_probabilities, window_svc_predictions = _fit_window_components(
        data["fusion"][train_indices],
        data["labels"][train_indices],
        data["fusion"][test_indices],
    )
    sequence = _sequence_components(
        window_probabilities,
        window_svc_predictions,
        sessions[test_indices],
        participants[test_indices],
    )
    session = _session_components(data, train_indices, test_indices)
    return {
        "window_logistic_probabilities": window_probabilities,
        "window_linear_svc_predictions": window_svc_predictions,
        "sequence_probabilities": sequence["probabilities"],
        "sequence_base_predictions": sequence["base_predictions"],
        "sequence_svc_votes": sequence["svc_votes"],
        "sequence_ids": sequence["sequence_ids"],
        "sequence_reports": sequence["reports"],
        "session_probabilities": session["probabilities"],
        "session_predictions": session["predictions"],
        "session_reports": session["reports"],
        "train_session_majority_counts": session["train_session_majority_counts"],
    }


def _apply_gate(components: Mapping[str, Any], candidate: Mapping[str, Any]) -> Any:
    """Apply low confirmation, session-high override, then ambiguous-high guard."""

    dep = _dependencies()
    np = dep["np"]
    predictions = np.asarray(
        components["sequence_base_predictions"], dtype=int
    ).copy()
    svc_votes = np.asarray(components["sequence_svc_votes"], dtype=int)
    session_predictions = np.asarray(components["session_predictions"], dtype=int)
    session_probabilities = np.asarray(components["session_probabilities"], dtype=float)
    if bool(candidate["low_confirmation"]):
        predictions[(predictions == LOW_CLASS) & (svc_votes != LOW_CLASS)] = DEFAULT_CLASS
    if bool(candidate["session_high_override"]):
        predictions[
            (predictions == DEFAULT_CLASS) & (session_predictions == HIGH_CLASS)
        ] = HIGH_CLASS
    threshold = candidate["ambiguous_high_ratio_threshold"]
    if threshold is not None:
        high_to_medium = session_probabilities[:, HIGH_CLASS] / np.maximum(
            session_probabilities[:, DEFAULT_CLASS], 1e-12
        )
        predictions[
            (session_predictions == DEFAULT_CLASS)
            & (predictions == HIGH_CLASS)
            & (svc_votes != HIGH_CLASS)
            & (high_to_medium >= float(threshold))
        ] = DEFAULT_CLASS
    return predictions


def _inner_gate_selection(
    data: Mapping[str, Any], outer_train_indices: Any
) -> dict[str, Any]:
    dep = _dependencies()
    np = dep["np"]
    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    labels = data["labels"]
    sessions = data["identities"]["session_id"]
    local_labels = labels[outer_train_indices]
    local_sessions = sessions[outer_train_indices]
    if len(np.unique(local_sessions)) < INNER_SPLITS:
        raise HierarchicalEvaluationError(
            f"inner selection requires at least {INNER_SPLITS} sessions"
        )
    splitter = dep["StratifiedGroupKFold"](
        n_splits=INNER_SPLITS, shuffle=True, random_state=FIXED_SEED
    )
    component_arrays = {
        "sequence_base_predictions": np.full(len(outer_train_indices), -1, dtype=int),
        "sequence_svc_votes": np.full(len(outer_train_indices), -1, dtype=int),
        "session_predictions": np.full(len(outer_train_indices), -1, dtype=int),
        "session_probabilities": np.zeros(
            (len(outer_train_indices), len(FIXED_CLASSES)), dtype=float
        ),
    }
    assigned = np.zeros(len(outer_train_indices), dtype=bool)
    fold_reports: list[dict[str, Any]] = []
    for fold, (inner_train_local, inner_validation_local) in enumerate(
        splitter.split(
            data["fusion"][outer_train_indices], local_labels, local_sessions
        ),
        1,
    ):
        inner_train = outer_train_indices[inner_train_local]
        inner_validation = outer_train_indices[inner_validation_local]
        train_sessions = set(str(value) for value in sessions[inner_train])
        validation_sessions = set(str(value) for value in sessions[inner_validation])
        overlap = sorted(train_sessions & validation_sessions)
        if overlap:
            raise RuntimeError(f"inner session leakage: {overlap}")
        components = _fit_component_predictions(data, inner_train, inner_validation)
        for name in component_arrays:
            component_arrays[name][inner_validation_local] = components[name]
        assigned[inner_validation_local] = True
        fold_reports.append(
            {
                "fold": fold,
                "train_session_count": len(train_sessions),
                "validation_session_count": len(validation_sessions),
                "session_overlap_ids": overlap,
                "validation_sample_count": int(len(inner_validation)),
            }
        )
    if not assigned.all():
        raise RuntimeError("inner OOF components did not assign every row")
    if any((values < 0).any() for name, values in component_arrays.items() if name != "session_probabilities"):
        raise RuntimeError("inner OOF component predictions are incomplete")

    candidate_reports: list[dict[str, Any]] = []
    for position, candidate in enumerate(GATE_REGISTRY):
        predictions = _apply_gate(component_arrays, candidate)
        metrics = _hard_metrics(local_labels, predictions)
        candidate_reports.append(
            {
                **dict(candidate),
                "registry_position": position,
                "inner_oof_accuracy": metrics["accuracy"],
                "inner_oof_macro_f1": metrics["macro_f1"],
                "selected": False,
            }
        )
    selected = sorted(
        candidate_reports,
        key=lambda item: (
            -float(item["inner_oof_accuracy"]),
            -float(item["inner_oof_macro_f1"]),
            int(item["registry_position"]),
        ),
    )[0]
    selected["selected"] = True
    return {
        "selected_gate_id": selected["gate_id"],
        "selected_gate": {
            key: selected[key]
            for key in (
                "gate_id",
                "low_confirmation",
                "session_high_override",
                "ambiguous_high_ratio_threshold",
            )
        },
        "selected_inner_oof_accuracy": selected["inner_oof_accuracy"],
        "selected_inner_oof_macro_f1": selected["inner_oof_macro_f1"],
        "selection_rule": (
            "maximize inner session-grouped OOF Accuracy, then Macro-F1, then "
            "the frozen registry order"
        ),
        "candidate_reports": candidate_reports,
        "inner_folds": fold_reports,
        "inner_components_refit_per_fold": True,
        "outer_test_labels_observed_during_selection": False,
    }


def _identity_overlap(
    data: Mapping[str, Any], train_indices: Any, test_indices: Any
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for field, values in data["identities"].items():
        train = {
            str(value) for value in values[train_indices] if value is not None
        }
        test = {str(value) for value in values[test_indices] if value is not None}
        result[field] = sorted(train & test)
    return result


def _outer_splits(data: Mapping[str, Any], design: str, seed: int) -> list[Any]:
    dep = _dependencies()
    sessions = data["identities"]["session_id"]
    if design == SGKF5:
        if len(set(sessions.tolist())) < OUTER_SPLITS:
            raise HierarchicalEvaluationError(
                f"sgkf5 requires at least {OUTER_SPLITS} sessions"
            )
        splitter = dep["StratifiedGroupKFold"](
            n_splits=OUTER_SPLITS, shuffle=True, random_state=int(seed)
        )
        return list(splitter.split(data["fusion"], data["labels"], sessions))
    if design == LEAVE_ONE_SESSION_OUT:
        splitter = dep["LeaveOneGroupOut"]()
        return list(splitter.split(data["fusion"], data["labels"], sessions))
    raise HierarchicalEvaluationError(
        f"outer_design must be one of {SUPPORTED_OUTER_DESIGNS}"
    )


def _oracle_constant_sequence_predictions(data: Mapping[str, Any]) -> Any:
    dep = _dependencies()
    np = dep["np"]
    sessions = data["identities"]["session_id"]
    participants = data["identities"]["participant_id"]
    labels = data["labels"]
    predictions = np.full(len(labels), -1, dtype=int)
    for session, participant in dict.fromkeys(
        zip(sessions.tolist(), participants.tolist())
    ):
        indices = np.flatnonzero(
            (sessions == session) & (participants == participant)
        )
        predictions[indices] = _majority_label(labels[indices])
    return predictions


def _gate_by_id(gate_id: str) -> dict[str, Any]:
    for candidate in GATE_REGISTRY:
        if candidate["gate_id"] == gate_id:
            return dict(candidate)
    raise HierarchicalEvaluationError(
        f"unknown gate_id {gate_id!r}; expected one of "
        f"{tuple(candidate['gate_id'] for candidate in GATE_REGISTRY)}"
    )


def _fixed_stress_specs(data: Mapping[str, Any], design: str) -> list[dict[str, Any]]:
    dep = _dependencies()
    np = dep["np"]
    if design not in SUPPORTED_STRESS_DESIGNS:
        raise HierarchicalEvaluationError(
            f"stress design must be one of {SUPPORTED_STRESS_DESIGNS}"
        )
    identities = data["identities"]
    all_indices = np.arange(len(data["labels"]), dtype=int)
    required = (
        ("cohort_id",)
        if design == LEAVE_ONE_COHORT_OUT
        else ("activity_id",)
        if design == LEAVE_ONE_ACTIVITY_OUT
        else ("cohort_id", "activity_id")
    )
    for field in required:
        if any(value is None for value in identities[field]):
            raise HierarchicalEvaluationError(
                f"{design} requires complete {field} values"
            )
    specs: list[dict[str, Any]] = []
    if design == LEAVE_ONE_COHORT_OUT:
        for cohort in sorted(set(str(value) for value in identities["cohort_id"])):
            test_mask = identities["cohort_id"] == cohort
            specs.append(
                {
                    "fold_id": f"cohort={cohort}",
                    "test_block": {"cohort_id": cohort},
                    "train_indices": all_indices[~test_mask],
                    "test_indices": all_indices[test_mask],
                    "embargo_indices": np.asarray([], dtype=int),
                }
            )
        return specs
    if design == LEAVE_ONE_ACTIVITY_OUT:
        for activity in sorted(set(str(value) for value in identities["activity_id"])):
            test_mask = identities["activity_id"] == activity
            specs.append(
                {
                    "fold_id": f"activity={activity}",
                    "test_block": {"activity_id": activity},
                    "train_indices": all_indices[~test_mask],
                    "test_indices": all_indices[test_mask],
                    "embargo_indices": np.asarray([], dtype=int),
                }
            )
        return specs

    cohorts = sorted(set(str(value) for value in identities["cohort_id"]))
    activities = sorted(set(str(value) for value in identities["activity_id"]))
    for cohort in cohorts:
        for activity in activities:
            test_mask = (identities["cohort_id"] == cohort) & (
                identities["activity_id"] == activity
            )
            if not test_mask.any():
                continue
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
                }
            )
    return specs


def fixed_hierarchical_stress_evaluation(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    *,
    sample_ids: Sequence[str],
    design: str,
    gate_id: str = "high_override_plus_low_confirmation",
    feature_mode: str = "fusion",
) -> dict[str, Any]:
    """Apply one fixed post-selected gate under cohort/activity blocking.

    This stress test intentionally performs no gate selection in a blocked
    outer fold.  It asks whether a simple gate frequently selected by the
    SGKF2026+LOSO primary development evaluations transports across a held-out
    cohort or activity.
    The gate itself is still post-selected on this dataset, so the result is a
    conservative diagnostic rather than a confirmatory generalization claim.
    """

    data = _prepare_inputs(
        records,
        visual_features,
        sensor_features,
        sample_ids,
        feature_mode=feature_mode,
    )
    dep = _dependencies()
    np = dep["np"]
    gate = _gate_by_id(gate_id)
    specs = _fixed_stress_specs(data, design)
    labels = data["labels"]
    predictions = np.full(len(labels), -1, dtype=int)
    assigned_count = np.zeros(len(labels), dtype=int)
    folds: list[dict[str, Any]] = []
    for spec in specs:
        train_indices = spec["train_indices"]
        test_indices = spec["test_indices"]
        components = _fit_component_predictions(data, train_indices, test_indices)
        fold_predictions = _apply_gate(components, gate)
        predictions[test_indices] = fold_predictions
        assigned_count[test_indices] += 1
        overlaps = _identity_overlap(data, train_indices, test_indices)
        if overlaps["session_id"] or overlaps["participant_id"]:
            raise HierarchicalEvaluationError(
                f"identity leakage in stress fold {spec['fold_id']}"
            )
        folds.append(
            {
                "fold_id": spec["fold_id"],
                "test_block": spec["test_block"],
                "train_sample_count": int(len(train_indices)),
                "test_sample_count": int(len(test_indices)),
                "embargo_sample_count": int(len(spec["embargo_indices"])),
                "session_overlap_ids": overlaps["session_id"],
                "participant_overlap_ids": overlaps["participant_id"],
                "cohort_overlap_ids": overlaps["cohort_id"],
                "activity_overlap_ids": overlaps["activity_id"],
                "metrics": _hard_metrics(labels[test_indices], fold_predictions),
            }
        )
    if not (assigned_count == 1).all() or (predictions < 0).any():
        raise RuntimeError(
            f"{design} did not assign every sample exactly once; "
            f"counts={Counter(int(value) for value in assigned_count)}"
        )
    return {
        "evaluation_kind": "fixed_post_selected_offline_hierarchical_block_stress",
        "design": design,
        "sample_count": len(labels),
        "fold_count": len(folds),
        "coverage_fraction": 1.0,
        "fixed_gate": gate,
        "feature_mode": feature_mode,
        "gate_selected_inside_stress_folds": False,
        "post_selected_on_current_dataset": True,
        "exploratory_only": True,
        "offline_full_sequence_required": True,
        "offline_full_session_required": True,
        "uses_unlabeled_test_session_context": True,
        "uses_future_feature_rows": True,
        "cross_participant_context": True,
        "real_time_compatible": False,
        "metrics": _hard_metrics(labels, predictions),
        "folds": folds,
        "claim_status": {
            "blocked_generalization_established": False,
            "cross_cohort_accuracy_established": False,
            "cross_activity_accuracy_established": False,
            "deployment_accuracy_established": False,
        },
    }


def hierarchical_offline_evaluation(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    *,
    sample_ids: Sequence[str],
    outer_design: str = SGKF5,
    outer_seed: int = FIXED_SEED,
    feature_mode: str = "fusion",
) -> dict[str, Any]:
    """Run honest nested OOF evaluation of the post-selected offline candidate."""

    if outer_design not in SUPPORTED_OUTER_DESIGNS:
        raise HierarchicalEvaluationError(
            f"outer_design must be one of {SUPPORTED_OUTER_DESIGNS}"
        )
    data = _prepare_inputs(
        records,
        visual_features,
        sensor_features,
        sample_ids,
        feature_mode=feature_mode,
    )
    dep = _dependencies()
    np = dep["np"]
    labels = data["labels"]
    sessions = data["identities"]["session_id"]
    participants = data["identities"]["participant_id"]
    splits = _outer_splits(data, outer_design, outer_seed)

    output_arrays = {
        "sequence_base_predictions": np.full(len(labels), -1, dtype=int),
        "sequence_svc_votes": np.full(len(labels), -1, dtype=int),
        "session_predictions": np.full(len(labels), -1, dtype=int),
        "session_probabilities": np.zeros((len(labels), len(FIXED_CLASSES)), dtype=float),
        "nested_predictions": np.full(len(labels), -1, dtype=int),
        "fold_ids": np.full(len(labels), -1, dtype=int),
    }
    selected_gate_ids = np.empty(len(labels), dtype=object)
    sequence_ids = np.empty(len(labels), dtype=object)
    assigned = np.zeros(len(labels), dtype=bool)
    fold_reports: list[dict[str, Any]] = []
    for fold, (train_indices, test_indices) in enumerate(splits, 1):
        train_sessions = set(str(value) for value in sessions[train_indices])
        test_sessions = set(str(value) for value in sessions[test_indices])
        train_participants = set(str(value) for value in participants[train_indices])
        test_participants = set(str(value) for value in participants[test_indices])
        if train_sessions & test_sessions or train_participants & test_participants:
            raise HierarchicalEvaluationError(
                f"identity leakage in outer fold {fold}"
            )
        selection = _inner_gate_selection(data, train_indices)
        components = _fit_component_predictions(data, train_indices, test_indices)
        predictions = _apply_gate(components, selection["selected_gate"])
        for name in (
            "sequence_base_predictions",
            "sequence_svc_votes",
            "session_predictions",
            "session_probabilities",
        ):
            output_arrays[name][test_indices] = components[name]
        output_arrays["nested_predictions"][test_indices] = predictions
        output_arrays["fold_ids"][test_indices] = fold
        selected_gate_ids[test_indices] = selection["selected_gate_id"]
        sequence_ids[test_indices] = components["sequence_ids"]
        assigned[test_indices] = True
        overlaps = _identity_overlap(data, train_indices, test_indices)
        fold_reports.append(
            {
                "fold": fold,
                "train_sample_count": int(len(train_indices)),
                "test_sample_count": int(len(test_indices)),
                "train_session_ids": sorted(train_sessions),
                "test_session_ids": sorted(test_sessions),
                "session_overlap_ids": overlaps["session_id"],
                "participant_overlap_ids": overlaps["participant_id"],
                "identity_overlap": overlaps,
                "inner_gate_selection": selection,
                "train_session_majority_counts": components[
                    "train_session_majority_counts"
                ],
                "sequence_base_metrics": _hard_metrics(
                    labels[test_indices], components["sequence_base_predictions"]
                ),
                "session_context_metrics": _hard_metrics(
                    labels[test_indices], components["session_predictions"]
                ),
                "selected_hierarchical_metrics": _hard_metrics(
                    labels[test_indices], predictions
                ),
                "session_context_predictions": components["session_reports"],
                "sequence_predictions": components["sequence_reports"],
            }
        )

    if not assigned.all() or any(
        (values < 0).any()
        for name, values in output_arrays.items()
        if name not in ("session_probabilities",)
    ):
        raise RuntimeError("outer OOF evaluation did not assign every row")

    oracle_predictions = _oracle_constant_sequence_predictions(data)
    global_session_majorities = [
        _majority_label(labels[sessions == session])
        for session in dict.fromkeys(sessions.tolist())
    ]
    oof_rows = []
    for index in range(len(labels)):
        oof_rows.append(
            {
                "sample_id": data["sample_ids"][index],
                "label": int(labels[index]),
                "session_id": str(sessions[index]),
                "participant_id": str(participants[index]),
                "sequence_id": str(sequence_ids[index]),
                "outer_fold": int(output_arrays["fold_ids"][index]),
                "selected_gate_id": str(selected_gate_ids[index]),
                "sequence_base_prediction": int(
                    output_arrays["sequence_base_predictions"][index]
                ),
                "linear_svc_sequence_vote": int(
                    output_arrays["sequence_svc_votes"][index]
                ),
                "session_context_prediction": int(
                    output_arrays["session_predictions"][index]
                ),
                "session_context_probabilities": [
                    round(float(value), 8)
                    for value in output_arrays["session_probabilities"][index]
                ],
                "nested_hierarchical_prediction": int(
                    output_arrays["nested_predictions"][index]
                ),
            }
        )

    claim_status = {
        "accuracy_0_9_established": False,
        "offline_full_session_accuracy_established": False,
        "multimodal_gain_established": False,
        "real_time_accuracy_established": False,
        "single_participant_accuracy_established": False,
        "cross_cohort_accuracy_established": False,
        "cross_activity_accuracy_established": False,
        "deployment_accuracy_established": False,
    }
    return {
        "evaluation_kind": "nested_post_selected_offline_full_session_transductive_oof",
        "available": True,
        "estimate_available": True,
        "exploratory_only": True,
        "post_selected_on_current_dataset": True,
        "post_selection_reason": (
            "the hierarchy, estimators, constants, and gate registry were proposed "
            "after inspecting results from this development dataset"
        ),
        "contains_inferential_statistics": False,
        "candidate_registry_frozen_within_evaluation": True,
        "candidate_registry_predeclared_before_current_dataset_inspection": False,
        "outer_test_labels_used_for_training_or_gate_selection": False,
        "outer_labels_used_for_stratified_split_construction": (
            outer_design == SGKF5
        ),
        "outer_split_label_use_note": (
            "SGKF5 uses labels only to construct stratified session folds; LOSO "
            "is label-independent. In both designs, held-out labels are excluded "
            "from fitting, aggregation, and gate selection."
        ),
        "inner_components_refit_per_fold": True,
        "offline_full_sequence_required": True,
        "offline_full_session_required": True,
        "uses_unlabeled_test_session_context": True,
        "uses_future_feature_rows": True,
        "uses_future_labels": False,
        "cross_participant_context": True,
        "real_time_compatible": False,
        "single_participant_inference_compatible": False,
        "aggregation_uses_labels_at_inference": False,
        "identity_or_path_used_as_model_feature": False,
        "identity_values_enter_numeric_estimator": False,
        "identities_used_as_aggregation_boundaries": True,
        "aggregation_boundaries": [
            ["session_id", "participant_id"],
            ["session_id"],
        ],
        "sample_count_used_as_model_feature_or_gate": False,
        "outer_design": outer_design,
        "outer_seed": int(outer_seed),
        "feature_mode": feature_mode,
        "fold_count": len(splits),
        "sample_count": len(labels),
        "session_count": len(set(str(value) for value in sessions)),
        "participant_count": len(set(str(value) for value in participants)),
        "sequence_count": len(
            set(zip(sessions.tolist(), participants.tolist()))
        ),
        "feature_dimensions": data["feature_dimensions"],
        "protocol": {
            "feature_order": (
                ["visual", "sensor"] if feature_mode == "fusion" else [feature_mode]
            ),
            "window_sequence_model": {
                "preprocessing": "outer/inner-training-only StandardScaler",
                "estimator": "LogisticRegression",
                "C": LOGISTIC_C,
                "class_weight": "balanced",
                "pooling": "geometric probability mean per session+participant",
                "minority_to_medium_ratio": MINORITY_TO_MEDIUM_RATIO,
            },
            "low_confirmation_model": {
                "preprocessing": "independent outer/inner-training-only StandardScaler",
                "estimator": "SVC(kernel=linear)",
                "C": LINEAR_SVC_C,
                "aggregation": "per-sequence hard majority with medium-first tie-break",
            },
            "session_context_model": {
                "summary": "per-feature median concatenated with IQR",
                "summary_boundary": "one complete session across participants",
                "summary_dimension": data["feature_dimensions"]["session_summary"],
                "summary_uses_labels_at_inference": False,
                "training_target": "training-session window-label majority",
                "training_target_tie_break": "medium, then low, then high",
                "preprocessing": "outer/inner-training-session-only StandardScaler",
                "estimator": "SVC(kernel=rbf, probability=True)",
                "C": SESSION_RBF_C,
                "gamma": SESSION_RBF_GAMMA,
                "random_state": FIXED_SEED,
            },
            "gate_order": [
                "linear-SVC low confirmation",
                "session high override",
                "ambiguous-high guard",
            ],
            "gate_registry": gate_registry(),
            "inner_splitter": (
                "StratifiedGroupKFold(n_splits=5, group=session_id, "
                "shuffle=True, random_state=2026)"
            ),
            "selection_rule": (
                "inner OOF Accuracy, then Macro-F1, then frozen registry order"
            ),
        },
        "sequence_base_metrics": _hard_metrics(
            labels, output_arrays["sequence_base_predictions"]
        ),
        "session_context_metrics": _hard_metrics(
            labels, output_arrays["session_predictions"]
        ),
        "nested_hierarchical_metrics": _hard_metrics(
            labels, output_arrays["nested_predictions"]
        ),
        "constant_sequence_oracle": {
            "diagnostic_only": True,
            "uses_ground_truth_labels": True,
            "used_as_model_input_or_for_selection": False,
            "interpretation": (
                "upper diagnostic for a predictor constrained to one label per "
                "participant-session sequence"
            ),
            "metrics": _hard_metrics(labels, oracle_predictions),
        },
        "session_majority_target_audit": {
            "diagnostic_only": True,
            "uses_ground_truth_labels": True,
            "used_as_model_input_for_held_out_sessions": False,
            "session_count": len(global_session_majorities),
            "class_counts": {
                str(label): int(count)
                for label, count in sorted(
                    Counter(global_session_majorities).items()
                )
            },
            "interpretation": (
                "The session-context expert is trained from majority targets of "
                "training sessions only; this full-data count is reported after OOF "
                "evaluation to expose the effective supervision problem."
            ),
        },
        "folds": fold_reports,
        "oof_predictions": oof_rows,
        "claim_status": claim_status,
        **claim_status,
    }


__all__ = [
    "DOUBLE_BLOCKED_COHORT_ACTIVITY",
    "HierarchicalEvaluationError",
    "LEAVE_ONE_SESSION_OUT",
    "LEAVE_ONE_ACTIVITY_OUT",
    "LEAVE_ONE_COHORT_OUT",
    "SGKF5",
    "fixed_hierarchical_stress_evaluation",
    "gate_registry",
    "hierarchical_offline_evaluation",
]
