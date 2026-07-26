"""Counterfactual context audit for the offline hierarchical candidate.

The primary hierarchical evaluator intentionally consumes every unlabeled row
in a held-out participant sequence and classroom session.  That is a valid
offline, transductive estimate, but it is not a causal deployment estimate.
This module refits the same models and reselects the same frozen gate registry
inside every outer-training partition under three inference contexts:

``full``
    The original complete-sequence and complete-session context.  The current
    row, other participants, and future feature rows may contribute.

``leave_current_out``
    Complete held-out context except for the row being predicted.  This is
    still offline and future-dependent.  Empty singleton contexts use an
    explicit training-only/default fallback rather than silently putting the
    current row back.

``causal_prefix``
    Only the current and timestamp-earlier rows are available.  Participant
    sequence prefixes and session-wide prefixes reset after a gap greater than
    60 seconds.  Session context may include past/current rows from other
    participants, so this remains a multi-participant online-context audit,
    not a single-participant deployment claim.

No mode uses held-out labels for aggregation, fitting, or gate selection.
Nevertheless, all results remain post-selection exploratory and every
establishment claim returned by this module is false.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from . import hierarchical_evaluation as _hier


FULL = "full"
LEAVE_CURRENT_OUT = "leave_current_out"
CAUSAL_PREFIX = "causal_prefix"
CONTEXT_MODES = (FULL, LEAVE_CURRENT_OUT, CAUSAL_PREFIX)
DEFAULT_MAXIMUM_GAP_SECONDS = 60.0


class HierarchicalCounterfactualError(ValueError):
    """Raised when the counterfactual audit cannot be evaluated safely."""


def _timestamps(records: Sequence[Mapping[str, Any]]) -> Any:
    dep = _hier._dependencies()
    np = dep["np"]
    values: list[float] = []
    for index, record in enumerate(records):
        sample_id = str(record.get("sample_id", index))
        source = record.get("source")
        if not isinstance(source, Mapping):
            raise HierarchicalCounterfactualError(
                f"record {sample_id} lacks source.metadata_median_seconds_of_day"
            )
        try:
            value = float(source["metadata_median_seconds_of_day"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HierarchicalCounterfactualError(
                f"record {sample_id} lacks a numeric "
                "source.metadata_median_seconds_of_day"
            ) from exc
        if not math.isfinite(value) or value < 0.0 or value >= 86_400.0:
            raise HierarchicalCounterfactualError(
                f"record {sample_id} has an invalid timestamp"
            )
        values.append(value)
    return np.asarray(values, dtype=float)


def _source_indices(
    mode: str,
    group_indices: Any,
    timestamps: Any,
    current_index: int,
    *,
    maximum_gap_seconds: float = DEFAULT_MAXIMUM_GAP_SECONDS,
) -> Any:
    """Return deterministic local source rows for one counterfactual context.

    ``group_indices`` and ``current_index`` address the same local held-out
    matrix.  Exact timestamp ties are all available in ``causal_prefix``:
    they are simultaneous rather than future observations.
    """

    dep = _hier._dependencies()
    np = dep["np"]
    if mode not in CONTEXT_MODES:
        raise HierarchicalCounterfactualError(
            f"mode must be one of {CONTEXT_MODES}"
        )
    try:
        gap = float(maximum_gap_seconds)
    except (TypeError, ValueError) as exc:
        raise HierarchicalCounterfactualError(
            "maximum_gap_seconds must be numeric"
        ) from exc
    if not math.isfinite(gap) or gap <= 0:
        raise HierarchicalCounterfactualError(
            "maximum_gap_seconds must be positive and finite"
        )
    indices = np.asarray(group_indices, dtype=int)
    if indices.ndim != 1 or not len(indices):
        raise HierarchicalCounterfactualError(
            "group_indices must be a non-empty one-dimensional array"
        )
    if int(current_index) not in set(int(value) for value in indices):
        raise HierarchicalCounterfactualError(
            "current_index must occur in group_indices"
        )
    # Stable index tie-break gives deterministic audit rows.  Prefix inclusion
    # itself is based on <= current timestamp, so every simultaneous row is
    # retained regardless of its record-order position.
    ordered = indices[
        np.lexsort((indices, np.asarray(timestamps, dtype=float)[indices]))
    ]
    if mode == FULL:
        return ordered
    if mode == LEAVE_CURRENT_OUT:
        return ordered[ordered != int(current_index)]

    current_time = float(timestamps[int(current_index)])
    eligible = ordered[
        np.asarray(timestamps, dtype=float)[ordered] <= current_time
    ]
    if not len(eligible):  # pragma: no cover - current row is always eligible
        raise RuntimeError("causal prefix unexpectedly excluded the current row")
    eligible_times = np.asarray(timestamps, dtype=float)[eligible]
    breaks = np.flatnonzero(np.diff(eligible_times) > gap)
    start = int(breaks[-1] + 1) if len(breaks) else 0
    return eligible[start:]


def _context_semantics() -> dict[str, dict[str, Any]]:
    return {
        FULL: {
            "description": (
                "complete held-out participant sequence plus complete held-out "
                "session across participants"
            ),
            "current_feature_row_included": True,
            "future_feature_rows_may_be_used": True,
            "other_participant_rows_may_be_used": True,
            "offline_full_sequence_required": True,
            "offline_full_session_required": True,
            "causal_feature_use": False,
            "transductive_test_context": True,
        },
        LEAVE_CURRENT_OUT: {
            "description": (
                "complete held-out contexts with the prediction target removed; "
                "future and other-participant rows remain available"
            ),
            "current_feature_row_included": False,
            "future_feature_rows_may_be_used": True,
            "other_participant_rows_may_be_used": True,
            "offline_full_sequence_required": True,
            "offline_full_session_required": True,
            "causal_feature_use": False,
            "transductive_test_context": True,
            "singleton_sequence_fallback": "fixed medium/default; no feature row",
            "singleton_session_fallback": (
                "componentwise median of outer-training session summaries"
            ),
        },
        CAUSAL_PREFIX: {
            "description": (
                "timestamp prefix through the current row; participant and "
                "session streams reset after the configured maximum gap"
            ),
            "current_feature_row_included": True,
            "future_feature_rows_may_be_used": False,
            "other_participant_rows_may_be_used": True,
            "offline_full_sequence_required": False,
            "offline_full_session_required": False,
            "causal_feature_use": True,
            "transductive_test_context": False,
            "online_cross_participant_context_required": True,
        },
    }


def _sequence_mode_components(
    window_probabilities: Any,
    window_svc_predictions: Any,
    sessions: Any,
    participants: Any,
    timestamps: Any,
    sample_ids: Sequence[str],
    mode: str,
    *,
    maximum_gap_seconds: float,
    include_audit: bool,
) -> dict[str, Any]:
    dep = _hier._dependencies()
    np = dep["np"]
    probabilities = np.zeros_like(window_probabilities, dtype=float)
    base_predictions = np.full(len(window_probabilities), -1, dtype=int)
    svc_votes = np.full(len(window_probabilities), -1, dtype=int)
    source_ids: list[list[str]] = [[] for _ in range(len(window_probabilities))]
    fallback = np.zeros(len(window_probabilities), dtype=bool)
    future_used = np.zeros(len(window_probabilities), dtype=bool)
    for current in range(len(window_probabilities)):
        group = np.flatnonzero(
            (sessions == sessions[current])
            & (participants == participants[current])
        )
        sources = _source_indices(
            mode,
            group,
            timestamps,
            current,
            maximum_gap_seconds=maximum_gap_seconds,
        )
        if not len(sources):
            # A singleton leave-current-out sequence has no admissible feature
            # context.  Medium is an explicit feature-free fallback.
            pooled = np.asarray([0.0, 1.0, 0.0], dtype=float)
            base = _hier.DEFAULT_CLASS
            vote = _hier.DEFAULT_CLASS
            fallback[current] = True
        else:
            log_mean = np.log(
                np.clip(window_probabilities[sources], 1e-12, 1.0)
            ).mean(axis=0)
            pooled = np.exp(log_mean - log_mean.max())
            pooled /= pooled.sum()
            minority = (
                _hier.LOW_CLASS
                if pooled[_hier.LOW_CLASS] >= pooled[_hier.HIGH_CLASS]
                else _hier.HIGH_CLASS
            )
            ratio = float(
                pooled[minority]
                / max(pooled[_hier.DEFAULT_CLASS], 1e-12)
            )
            base = (
                minority
                if ratio >= _hier.MINORITY_TO_MEDIUM_RATIO
                else _hier.DEFAULT_CLASS
            )
            vote = _hier._majority_label(window_svc_predictions[sources])
        probabilities[current] = pooled
        base_predictions[current] = int(base)
        svc_votes[current] = int(vote)
        if include_audit:
            source_ids[current] = [str(sample_ids[index]) for index in sources]
            future_used[current] = bool(
                len(sources)
                and (
                    np.asarray(timestamps, dtype=float)[sources]
                    > float(timestamps[current])
                ).any()
            )
    if (base_predictions < 0).any() or (svc_votes < 0).any():
        raise RuntimeError("sequence counterfactual did not assign every row")
    return {
        "probabilities": probabilities,
        "base_predictions": base_predictions,
        "svc_votes": svc_votes,
        "source_sample_ids": source_ids,
        "fallback_used": fallback,
        "future_source_used": future_used,
    }


def _fit_session_context_model(data: Mapping[str, Any], train_indices: Any) -> dict[str, Any]:
    dep = _hier._dependencies()
    np = dep["np"]
    sessions = data["identities"]["session_id"]
    labels = data["labels"]
    fusion = data["fusion"]
    train_sessions = list(dict.fromkeys(sessions[train_indices].tolist()))
    rows = []
    targets = []
    for session in train_sessions:
        indices = train_indices[sessions[train_indices] == session]
        rows.append(_hier._session_summary(fusion[indices]))
        targets.append(_hier._majority_label(labels[indices]))
    matrix = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=int)
    if len(set(targets)) < 2:
        raise HierarchicalCounterfactualError(
            "session-context training requires at least two majority classes"
        )
    scaler = dep["StandardScaler"]()
    model = dep["SVC"](
        kernel="rbf",
        C=_hier.SESSION_RBF_C,
        gamma=_hier.SESSION_RBF_GAMMA,
        probability=True,
        random_state=_hier.FIXED_SEED,
    )
    model.fit(scaler.fit_transform(matrix), target)
    return {
        "scaler": scaler,
        "model": model,
        # Training-only fallback for singleton leave-current-out sessions.
        "fallback_summary": np.median(matrix, axis=0),
        "majority_counts": {
            str(label): int(count)
            for label, count in sorted(Counter(targets).items())
        },
    }


def _session_mode_components(
    data: Mapping[str, Any],
    test_indices: Any,
    timestamps: Any,
    model_state: Mapping[str, Any],
    mode: str,
    *,
    maximum_gap_seconds: float,
    include_audit: bool,
) -> dict[str, Any]:
    dep = _hier._dependencies()
    np = dep["np"]
    sessions = data["identities"]["session_id"][test_indices]
    participants = data["identities"]["participant_id"][test_indices]
    fusion = data["fusion"]
    local_times = timestamps[test_indices]
    local_sample_ids = [data["sample_ids"][int(index)] for index in test_indices]
    rows = []
    source_ids: list[list[str]] = [[] for _ in range(len(test_indices))]
    fallback = np.zeros(len(test_indices), dtype=bool)
    future_used = np.zeros(len(test_indices), dtype=bool)
    other_participant_used = np.zeros(len(test_indices), dtype=bool)
    for current in range(len(test_indices)):
        group = np.flatnonzero(sessions == sessions[current])
        sources = _source_indices(
            mode,
            group,
            local_times,
            current,
            maximum_gap_seconds=maximum_gap_seconds,
        )
        if not len(sources):
            summary = np.asarray(model_state["fallback_summary"], dtype=float)
            fallback[current] = True
        else:
            summary = _hier._session_summary(fusion[test_indices[sources]])
        rows.append(summary)
        if include_audit:
            source_ids[current] = [local_sample_ids[index] for index in sources]
            future_used[current] = bool(
                len(sources)
                and (local_times[sources] > local_times[current]).any()
            )
            other_participant_used[current] = bool(
                len(sources)
                and (participants[sources] != participants[current]).any()
            )
    matrix = np.asarray(rows, dtype=float)
    scaled = model_state["scaler"].transform(matrix)
    model = model_state["model"]
    return {
        "probabilities": _hier._aligned_probabilities(model, scaled),
        # Calibrated probability argmax need not equal the SVC decision.
        "predictions": model.predict(scaled).astype(int),
        "source_sample_ids": source_ids,
        "fallback_used": fallback,
        "future_source_used": future_used,
        "other_participant_source_used": other_participant_used,
    }


def _fit_mode_components(
    data: Mapping[str, Any],
    timestamps: Any,
    train_indices: Any,
    test_indices: Any,
    *,
    maximum_gap_seconds: float,
    include_audit: bool,
) -> dict[str, dict[str, Any]]:
    dep = _hier._dependencies()
    np = dep["np"]
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    sessions = data["identities"]["session_id"]
    overlap = set(str(value) for value in sessions[train_indices]) & set(
        str(value) for value in sessions[test_indices]
    )
    if overlap:
        raise HierarchicalCounterfactualError(
            f"component fit has train/test session overlap: {sorted(overlap)}"
        )
    window_probabilities, window_svc_predictions = _hier._fit_window_components(
        data["fusion"][train_indices],
        data["labels"][train_indices],
        data["fusion"][test_indices],
    )
    model_state = _fit_session_context_model(data, train_indices)
    local_sessions = sessions[test_indices]
    local_participants = data["identities"]["participant_id"][test_indices]
    local_times = timestamps[test_indices]
    local_sample_ids = [data["sample_ids"][int(index)] for index in test_indices]
    output: dict[str, dict[str, Any]] = {}
    for mode in CONTEXT_MODES:
        sequence = _sequence_mode_components(
            window_probabilities,
            window_svc_predictions,
            local_sessions,
            local_participants,
            local_times,
            local_sample_ids,
            mode,
            maximum_gap_seconds=maximum_gap_seconds,
            include_audit=include_audit,
        )
        session = _session_mode_components(
            data,
            test_indices,
            timestamps,
            model_state,
            mode,
            maximum_gap_seconds=maximum_gap_seconds,
            include_audit=include_audit,
        )
        output[mode] = {
            "sequence_base_predictions": sequence["base_predictions"],
            "sequence_svc_votes": sequence["svc_votes"],
            "session_predictions": session["predictions"],
            "session_probabilities": session["probabilities"],
            "sequence_source_sample_ids": sequence["source_sample_ids"],
            "session_source_sample_ids": session["source_sample_ids"],
            "sequence_fallback_used": sequence["fallback_used"],
            "session_fallback_used": session["fallback_used"],
            "sequence_future_source_used": sequence["future_source_used"],
            "session_future_source_used": session["future_source_used"],
            "session_other_participant_source_used": session[
                "other_participant_source_used"
            ],
            "train_session_majority_counts": model_state["majority_counts"],
        }
    return output


def _select_gates_by_mode(
    data: Mapping[str, Any],
    timestamps: Any,
    outer_train_indices: Any,
    *,
    maximum_gap_seconds: float,
) -> dict[str, dict[str, Any]]:
    dep = _hier._dependencies()
    np = dep["np"]
    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    labels = data["labels"]
    sessions = data["identities"]["session_id"]
    local_labels = labels[outer_train_indices]
    local_sessions = sessions[outer_train_indices]
    splitter = dep["StratifiedGroupKFold"](
        n_splits=_hier.INNER_SPLITS,
        shuffle=True,
        random_state=_hier.FIXED_SEED,
    )
    fields = (
        "sequence_base_predictions",
        "sequence_svc_votes",
        "session_predictions",
        "session_probabilities",
    )
    arrays: dict[str, dict[str, Any]] = {}
    for mode in CONTEXT_MODES:
        arrays[mode] = {
            "sequence_base_predictions": np.full(
                len(outer_train_indices), -1, dtype=int
            ),
            "sequence_svc_votes": np.full(
                len(outer_train_indices), -1, dtype=int
            ),
            "session_predictions": np.full(
                len(outer_train_indices), -1, dtype=int
            ),
            "session_probabilities": np.zeros(
                (len(outer_train_indices), len(_hier.FIXED_CLASSES)),
                dtype=float,
            ),
        }
    assigned = np.zeros(len(outer_train_indices), dtype=bool)
    fold_reports = []
    for fold, (train_local, validation_local) in enumerate(
        splitter.split(
            data["fusion"][outer_train_indices],
            local_labels,
            local_sessions,
        ),
        1,
    ):
        train_indices = outer_train_indices[train_local]
        validation_indices = outer_train_indices[validation_local]
        train_sessions = set(str(value) for value in sessions[train_indices])
        validation_sessions = set(
            str(value) for value in sessions[validation_indices]
        )
        overlap = sorted(train_sessions & validation_sessions)
        if overlap:
            raise RuntimeError(f"inner session leakage: {overlap}")
        components = _fit_mode_components(
            data,
            timestamps,
            train_indices,
            validation_indices,
            maximum_gap_seconds=maximum_gap_seconds,
            include_audit=False,
        )
        for mode in CONTEXT_MODES:
            for field in fields:
                arrays[mode][field][validation_local] = components[mode][field]
        assigned[validation_local] = True
        fold_reports.append(
            {
                "fold": fold,
                "train_session_count": len(train_sessions),
                "validation_session_count": len(validation_sessions),
                "session_overlap_ids": overlap,
            }
        )
    if not assigned.all():
        raise RuntimeError("inner counterfactual OOF did not assign every row")

    selections: dict[str, dict[str, Any]] = {}
    for mode in CONTEXT_MODES:
        candidate_reports = []
        for position, candidate in enumerate(_hier.GATE_REGISTRY):
            predictions = _hier._apply_gate(arrays[mode], candidate)
            metrics = _hier._hard_metrics(local_labels, predictions)
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
        selections[mode] = {
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
            "candidate_reports": candidate_reports,
            "inner_folds": fold_reports,
            "outer_test_labels_observed_during_selection": False,
            "components_refit_per_inner_fold": True,
        }
    return selections


def hierarchical_counterfactual_evaluation(
    records: Sequence[Mapping[str, Any]],
    visual_features: Any,
    sensor_features: Any,
    *,
    sample_ids: Sequence[str],
    outer_design: str = _hier.SGKF5,
    outer_seed: int = _hier.FIXED_SEED,
    maximum_gap_seconds: float = DEFAULT_MAXIMUM_GAP_SECONDS,
) -> dict[str, Any]:
    """Evaluate full, leave-current-out, and causal-prefix contexts.

    Gate selection is repeated independently for each context mode using only
    session-grouped inner OOF predictions from the corresponding outer-training
    partition.  This isolates the inference-context assumption without giving
    any mode access to outer-test labels during selection.
    """

    if outer_design not in _hier.SUPPORTED_OUTER_DESIGNS:
        raise HierarchicalCounterfactualError(
            f"outer_design must be one of {_hier.SUPPORTED_OUTER_DESIGNS}"
        )
    try:
        maximum_gap_seconds = float(maximum_gap_seconds)
    except (TypeError, ValueError) as exc:
        raise HierarchicalCounterfactualError(
            "maximum_gap_seconds must be numeric"
        ) from exc
    if not math.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0:
        raise HierarchicalCounterfactualError(
            "maximum_gap_seconds must be positive and finite"
        )
    data = _hier._prepare_inputs(
        records, visual_features, sensor_features, sample_ids
    )
    timestamp_values = _timestamps(data["records"])
    dep = _hier._dependencies()
    np = dep["np"]
    labels = data["labels"]
    sessions = data["identities"]["session_id"]
    participants = data["identities"]["participant_id"]
    splits = _hier._outer_splits(data, outer_design, int(outer_seed))

    predictions = {
        mode: np.full(len(labels), -1, dtype=int) for mode in CONTEXT_MODES
    }
    selected_gate_ids = {
        mode: np.empty(len(labels), dtype=object) for mode in CONTEXT_MODES
    }
    fold_ids = np.full(len(labels), -1, dtype=int)
    source_audits = {
        mode: {
            "sequence": [[] for _ in range(len(labels))],
            "session": [[] for _ in range(len(labels))],
            "sequence_fallback": np.zeros(len(labels), dtype=bool),
            "session_fallback": np.zeros(len(labels), dtype=bool),
            "sequence_future": np.zeros(len(labels), dtype=bool),
            "session_future": np.zeros(len(labels), dtype=bool),
            "session_other_participant": np.zeros(len(labels), dtype=bool),
        }
        for mode in CONTEXT_MODES
    }
    assigned = np.zeros(len(labels), dtype=bool)
    folds = []
    for fold, (train_indices, test_indices) in enumerate(splits, 1):
        overlaps = _hier._identity_overlap(data, train_indices, test_indices)
        if overlaps["session_id"] or overlaps["participant_id"]:
            raise HierarchicalCounterfactualError(
                f"identity leakage in outer fold {fold}"
            )
        selections = _select_gates_by_mode(
            data,
            timestamp_values,
            train_indices,
            maximum_gap_seconds=maximum_gap_seconds,
        )
        components = _fit_mode_components(
            data,
            timestamp_values,
            train_indices,
            test_indices,
            maximum_gap_seconds=maximum_gap_seconds,
            include_audit=True,
        )
        fold_mode_metrics = {}
        for mode in CONTEXT_MODES:
            mode_predictions = _hier._apply_gate(
                components[mode], selections[mode]["selected_gate"]
            )
            predictions[mode][test_indices] = mode_predictions
            selected_gate_ids[mode][test_indices] = selections[mode][
                "selected_gate_id"
            ]
            audit = source_audits[mode]
            for local, global_index in enumerate(test_indices):
                audit["sequence"][int(global_index)] = components[mode][
                    "sequence_source_sample_ids"
                ][local]
                audit["session"][int(global_index)] = components[mode][
                    "session_source_sample_ids"
                ][local]
            for target, field in (
                ("sequence_fallback", "sequence_fallback_used"),
                ("session_fallback", "session_fallback_used"),
                ("sequence_future", "sequence_future_source_used"),
                ("session_future", "session_future_source_used"),
                (
                    "session_other_participant",
                    "session_other_participant_source_used",
                ),
            ):
                audit[target][test_indices] = components[mode][field]
            fold_mode_metrics[mode] = _hier._hard_metrics(
                labels[test_indices], mode_predictions
            )
        fold_ids[test_indices] = fold
        assigned[test_indices] = True
        folds.append(
            {
                "fold": fold,
                "train_sample_count": int(len(train_indices)),
                "test_sample_count": int(len(test_indices)),
                "train_session_ids": sorted(
                    set(str(value) for value in sessions[train_indices])
                ),
                "test_session_ids": sorted(
                    set(str(value) for value in sessions[test_indices])
                ),
                "session_overlap_ids": overlaps["session_id"],
                "participant_overlap_ids": overlaps["participant_id"],
                "gate_selection_by_mode": selections,
                "outer_metrics_by_mode": fold_mode_metrics,
            }
        )
    if not assigned.all() or any((value < 0).any() for value in predictions.values()):
        raise RuntimeError("counterfactual outer OOF did not assign every row")

    full_predictions = predictions[FULL]
    semantics = _context_semantics()
    mode_results: dict[str, dict[str, Any]] = {}
    for mode in CONTEXT_MODES:
        audit = source_audits[mode]
        changed = predictions[mode] != full_predictions
        mode_results[mode] = {
            "metrics": _hier._hard_metrics(labels, predictions[mode]),
            "changed_from_full_count": int(changed.sum()),
            "changed_from_full_fraction": round(float(changed.mean()), 6),
            "selected_gate_counts": {
                str(gate): int(count)
                for gate, count in sorted(
                    Counter(str(value) for value in selected_gate_ids[mode]).items()
                )
            },
            "sequence_empty_context_fallback_count": int(
                audit["sequence_fallback"].sum()
            ),
            "session_empty_context_fallback_count": int(
                audit["session_fallback"].sum()
            ),
            "rows_using_future_sequence_context": int(
                audit["sequence_future"].sum()
            ),
            "rows_using_future_session_context": int(
                audit["session_future"].sum()
            ),
            "rows_using_other_participant_session_context": int(
                audit["session_other_participant"].sum()
            ),
            "semantics": semantics[mode],
        }

    oof_rows = []
    for index in range(len(labels)):
        modes = {}
        for mode in CONTEXT_MODES:
            audit = source_audits[mode]
            modes[mode] = {
                "prediction": int(predictions[mode][index]),
                "selected_gate_id": str(selected_gate_ids[mode][index]),
                "sequence_source_sample_ids": audit["sequence"][index],
                "session_source_sample_ids": audit["session"][index],
                "sequence_fallback_used": bool(
                    audit["sequence_fallback"][index]
                ),
                "session_fallback_used": bool(audit["session_fallback"][index]),
                "future_sequence_source_used": bool(
                    audit["sequence_future"][index]
                ),
                "future_session_source_used": bool(
                    audit["session_future"][index]
                ),
                "other_participant_session_source_used": bool(
                    audit["session_other_participant"][index]
                ),
            }
        oof_rows.append(
            {
                "sample_id": data["sample_ids"][index],
                "label": int(labels[index]),
                "session_id": str(sessions[index]),
                "participant_id": str(participants[index]),
                "timestamp_seconds_of_day": round(
                    float(timestamp_values[index]), 6
                ),
                "outer_fold": int(fold_ids[index]),
                "modes": modes,
            }
        )

    claim_status = {
        "accuracy_0_9_established": False,
        "offline_transductive_accuracy_established": False,
        "leave_current_out_accuracy_established": False,
        "causal_prefix_accuracy_established": False,
        "real_time_accuracy_established": False,
        "single_participant_accuracy_established": False,
        "cross_cohort_accuracy_established": False,
        "cross_activity_accuracy_established": False,
        "deployment_accuracy_established": False,
    }
    return {
        "evaluation_kind": (
            "nested_post_selected_hierarchical_context_counterfactual_oof"
        ),
        "available": True,
        "estimate_available": True,
        "exploratory_only": True,
        "post_selected_on_current_dataset": True,
        "outer_test_labels_used_for_training_or_gate_selection": False,
        "outer_labels_used_for_stratified_split_construction": (
            outer_design == _hier.SGKF5
        ),
        "outer_split_label_use_note": (
            "SGKF5 labels construct stratified session folds only; held-out labels "
            "do not enter fitting, context aggregation, or gate selection."
        ),
        "aggregation_uses_labels_at_inference": False,
        "identity_or_path_used_as_model_feature": False,
        "identity_values_enter_numeric_estimator": False,
        "identities_used_as_aggregation_boundaries": True,
        "aggregation_boundaries": [
            ["session_id", "participant_id"],
            ["session_id"],
        ],
        "gate_reselected_inside_each_outer_training_partition_for_each_mode": True,
        "outer_design": outer_design,
        "outer_seed": int(outer_seed),
        "maximum_gap_seconds": maximum_gap_seconds,
        "sample_count": len(labels),
        "session_count": len(set(str(value) for value in sessions)),
        "participant_count": len(set(str(value) for value in participants)),
        "mode_order": list(CONTEXT_MODES),
        "mode_results": mode_results,
        "folds": folds,
        "oof_predictions": oof_rows,
        "claim_status": claim_status,
        **claim_status,
    }


__all__ = [
    "CAUSAL_PREFIX",
    "CONTEXT_MODES",
    "DEFAULT_MAXIMUM_GAP_SECONDS",
    "FULL",
    "HierarchicalCounterfactualError",
    "LEAVE_CURRENT_OUT",
    "hierarchical_counterfactual_evaluation",
]
