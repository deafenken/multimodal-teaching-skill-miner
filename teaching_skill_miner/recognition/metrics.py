from __future__ import annotations

from typing import Any


def classification_metrics(
    y_true: Any,
    probabilities: Any,
    *,
    class_names: list[str],
) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            brier_score_loss,
            classification_report,
            confusion_matrix,
            log_loss,
            precision_recall_fscore_support,
            roc_auc_score,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("real recognition requires the optional `recognition` dependencies") from exc

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="weighted", zero_division=0
    )
    labels = list(range(len(class_names)))
    one_hot = np.eye(len(class_names))[y_true]
    try:
        macro_auc = float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        macro_auc = None
    brier = float(
        sum(brier_score_loss(one_hot[:, index], probabilities[:, index]) for index in labels)
        / len(class_names)
    )

    confidences = probabilities.max(axis=1)
    correct = predictions == y_true
    ece = 0.0
    bins: list[dict[str, Any]] = []
    for lower in [index / 10 for index in range(10)]:
        upper = lower + 0.1
        mask = (confidences >= lower) & (confidences < upper if upper < 1 else confidences <= upper)
        count = int(mask.sum())
        if not count:
            continue
        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        ece += count / len(y_true) * abs(bin_accuracy - bin_confidence)
        bins.append(
            {
                "lower": round(lower, 1),
                "upper": round(upper, 1),
                "count": count,
                "accuracy": round(bin_accuracy, 4),
                "mean_confidence": round(bin_confidence, 4),
            }
        )

    return {
        "sample_count": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_ovr_auroc": round(macro_auc, 4) if macro_auc is not None else None,
        "log_loss": round(float(log_loss(y_true, probabilities, labels=labels)), 4),
        "multiclass_brier": round(brier, 4),
        "expected_calibration_error_10_bins": round(float(ece), 4),
        "calibration_bins": bins,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "per_class": {
            class_names[index]: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index in labels
        },
        "sklearn_classification_report": classification_report(
            y_true,
            predictions,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def grouped_bootstrap_interval(
    y_true: Any,
    probabilities: Any,
    groups: Any,
    *,
    seed: int = 2026,
    replicates: int = 2000,
) -> dict[str, list[float]]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities.argmax(axis=1)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        accuracies.append(float(accuracy_score(y_true[indices], predictions[indices])))
        macro_f1s.append(float(f1_score(y_true[indices], predictions[indices], average="macro", zero_division=0)))
    return {
        "accuracy_95_ci": [round(float(value), 4) for value in np.quantile(accuracies, [0.025, 0.975])],
        "macro_f1_95_ci": [round(float(value), 4) for value in np.quantile(macro_f1s, [0.025, 0.975])],
        "bootstrap_unit": "filename_index_surrogate_group",
        "replicates": replicates,
    }
