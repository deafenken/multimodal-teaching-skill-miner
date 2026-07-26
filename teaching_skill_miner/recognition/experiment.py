from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ..io_utils import ensure_private_directory
from .datasets import (
    OUC_CGE_CLASS_NAMES,
    discover_ouc_cge,
    file_sha256,
    probe_video,
    resolve_record_path,
)
from .features import extract_multimodal_features
from .metrics import classification_metrics, grouped_bootstrap_interval


def _provenance_scope(audit: dict[str, Any]) -> dict[str, str]:
    if audit.get("provenance_verified"):
        return {
            "training_scope": "OUC-CGE official public-sample pilot only",
            "metric_scope": "OUC-CGE verified public sample: low/medium/high group engagement",
            "dataset_limitation": "This is an executable pilot on the verified 36-file official public sample, with exact duplicates removed; it is not the 7,705-clip benchmark.",
            "inference_warning": "Pilot model trained on the verified OUC-CGE public sample; not production-calibrated.",
        }
    return {
        "training_scope": "unverified OUC-CGE-compatible directory",
        "metric_scope": "unverified directory labels; official OUC-CGE origin not established",
        "dataset_limitation": "The input directory did not match the pinned official sample hashes; real-classroom origin and independent human labels are unverified.",
        "inference_warning": "Model trained on an unverified OUC-CGE-compatible directory; provenance and production accuracy are not established.",
    }


def _dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("install optional dependencies with `pip install -e '.[recognition]'`") from exc
    return np, LogisticRegression, StratifiedGroupKFold, StandardScaler


def _softmax(values: Any) -> Any:
    import numpy as np

    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _fit_predict(
    train_x: Any,
    train_y: Any,
    test_x: Any,
) -> tuple[Any, dict[str, Any]]:
    np, LogisticRegression, _, StandardScaler = _dependencies()
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=2026,
    )
    model.fit(scaled_train, train_y)
    raw_probabilities = model.predict_proba(scaler.transform(test_x))
    probabilities = np.zeros((len(test_x), len(OUC_CGE_CLASS_NAMES)), dtype=float)
    for column, label in enumerate(model.classes_):
        probabilities[:, int(label)] = raw_probabilities[:, column]
    state = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "model_classes": [int(item) for item in model.classes_],
    }
    return probabilities, state


def _feature_matrix(
    manifest: dict[str, Any],
    dataset_root: str | Path,
    *,
    frame_count: int,
) -> tuple[Any, list[str], int, list[dict[str, Any]]]:
    np, _, _, _ = _dependencies()
    vectors: list[Any] = []
    feature_names: list[str] | None = None
    visual_dimension = 0
    diagnostics: list[dict[str, Any]] = []
    for index, record in enumerate(manifest["records"], 1):
        path = resolve_record_path(dataset_root, record)
        extracted = extract_multimodal_features(
            path,
            duration=float(record["duration_seconds"]),
            start_time=float(record.get("video_start_time_seconds", 0.0)),
            audio_stream_index=record.get("selected_audio_stream_index"),
            frame_count=frame_count,
        )
        if feature_names is None:
            feature_names = extracted["feature_names"]
            visual_dimension = int(extracted["visual_dimension"])
        elif feature_names != extracted["feature_names"]:
            raise RuntimeError("feature schema changed between samples")
        vectors.append(extracted["features"])
        diagnostics.append(
            {
                "sample_id": record["sample_id"],
                "label_name": record["label_name"],
                "audio_present": extracted["audio_present"],
                "processed_index": index,
            }
        )
    return np.vstack(vectors), feature_names or [], visual_dimension, diagnostics


def _audio_availability_audit(
    records: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostic_by_sample = {item["sample_id"]: item for item in diagnostics}
    by_label: dict[str, dict[str, Any]] = {}
    for label_name in OUC_CGE_CLASS_NAMES:
        label_records = [record for record in records if record["label_name"] == label_name]
        available = 0
        for record in label_records:
            diagnostic = diagnostic_by_sample.get(record["sample_id"])
            if diagnostic is None:
                present = bool(record.get("has_audio", True))
            else:
                present = bool(diagnostic["audio_present"])
            available += int(present)
        total = len(label_records)
        by_label[label_name] = {
            "available": available,
            "total": total,
            "coverage": round(available / total, 4) if total else 0.0,
        }
    rates = [item["coverage"] for item in by_label.values()]
    complete = all(item["available"] == item["total"] for item in by_label.values())
    range_value = round(max(rates) - min(rates), 4) if rates else 0.0
    confounded = range_value >= 0.25
    return {
        "by_label": by_label,
        "complete_audio_coverage": complete,
        "coverage_range_across_labels": range_value,
        "audio_availability_label_confounding_detected": confounded,
        "eligible_for_multimodal_accuracy_claim": complete and not confounded,
        "eligibility_rule": "every evaluated video must have decoded overlapping audio and label coverage must not differ",
    }


def _select_modality(matrix: Any, modality: str, visual_dimension: int) -> Any:
    if modality == "visual":
        return matrix[:, :visual_dimension]
    if modality == "audio":
        return matrix[:, visual_dimension:]
    if modality == "fusion":
        return matrix
    raise ValueError(f"unsupported modality: {modality}")


def _encoding_metadata_matrix(records: list[dict[str, Any]]) -> tuple[Any, list[str]]:
    """Build a label-free nuisance baseline from container/encoding metadata only."""

    np, _, _, _ = _dependencies()
    names = [
        "duration_seconds",
        "log_bytes_per_second",
        "video_width",
        "video_height",
        "video_stream_count",
        "audio_stream_count",
        "codec_h264",
        "codec_mpeg4",
        "codec_other",
        "has_audio",
    ]
    rows: list[list[float]] = []
    for record in records:
        streams = record.get("streams", [])
        video_streams = [item for item in streams if item.get("codec_type") == "video"]
        audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
        first_video = video_streams[0] if video_streams else {}
        duration = max(float(record.get("duration_seconds", 0) or 0), 1e-6)
        codec = str(first_video.get("codec_name", "")).lower()
        rows.append(
            [
                duration,
                float(np.log1p(float(record.get("size_bytes", 0) or 0) / duration)),
                float(first_video.get("width", 0) or 0),
                float(first_video.get("height", 0) or 0),
                float(len(video_streams)),
                float(len(audio_streams)),
                float(codec == "h264"),
                float(codec == "mpeg4"),
                float(codec not in {"h264", "mpeg4"}),
                float(bool(record.get("has_audio", audio_streams))),
            ]
        )
    return np.asarray(rows, dtype=float), names


def _grouped_oof(
    matrix: Any,
    labels: Any,
    groups: Any,
    *,
    folds: int,
    seed: int,
) -> tuple[Any, list[dict[str, Any]]]:
    np, _, StratifiedGroupKFold, _ = _dependencies()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.zeros((len(labels), len(OUC_CGE_CLASS_NAMES)), dtype=float)
    fold_reports: list[dict[str, Any]] = []
    assigned = np.zeros(len(labels), dtype=bool)
    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(matrix, labels, groups), 1):
        train_groups = set(groups[train_indices])
        test_groups = set(groups[test_indices])
        overlap = sorted(train_groups & test_groups)
        if overlap:
            raise RuntimeError(f"group leakage in fold {fold_index}: {overlap}")
        fold_probabilities, _ = _fit_predict(matrix[train_indices], labels[train_indices], matrix[test_indices])
        probabilities[test_indices] = fold_probabilities
        assigned[test_indices] = True
        fold_metrics = classification_metrics(
            labels[test_indices], fold_probabilities, class_names=OUC_CGE_CLASS_NAMES
        )
        fold_reports.append(
            {
                "fold": fold_index,
                "train_count": len(train_indices),
                "test_count": len(test_indices),
                "train_group_count": len(train_groups),
                "test_group_count": len(test_groups),
                "group_overlap": overlap,
                "accuracy": fold_metrics["accuracy"],
                "macro_f1": fold_metrics["macro_f1"],
            }
        )
    if not assigned.all():
        raise RuntimeError("not every sample received an out-of-fold prediction")
    return probabilities, fold_reports


def _post_feature_label_permutation_sanity(
    matrix: Any,
    labels: Any,
    groups: Any,
    *,
    folds: int,
    seed: int,
    observed_macro_f1: float,
    replicates: int = 20,
) -> dict[str, Any]:
    """Measure the null distribution after shuffling labels post feature extraction."""

    np, _, _, _ = _dependencies()
    rng = np.random.default_rng(seed + 17)
    scores: list[float] = []
    for _ in range(replicates):
        shuffled = rng.permutation(labels)
        probabilities, _ = _grouped_oof(
            matrix,
            shuffled,
            groups,
            folds=folds,
            seed=seed,
        )
        score = classification_metrics(
            shuffled,
            probabilities,
            class_names=OUC_CGE_CLASS_NAMES,
        )["macro_f1"]
        scores.append(float(score))
    equal_or_better = sum(score >= observed_macro_f1 for score in scores)
    return {
        "replicates": replicates,
        "mean_macro_f1": round(float(np.mean(scores)), 4),
        "min_macro_f1": round(float(np.min(scores)), 4),
        "max_macro_f1": round(float(np.max(scores)), 4),
        "empirical_p_value": round((equal_or_better + 1) / (replicates + 1), 4),
        "interpretation": "Post-feature association sanity check only. It does not test target leakage and does not rule out source, session, camera, audio-availability, or encoding shortcuts.",
    }


def _train_final_model(matrix: Any, labels: Any) -> dict[str, Any]:
    probabilities, state = _fit_predict(matrix, labels, matrix[:1])
    del probabilities
    return state


def _write_predictions(
    path: Path,
    records: list[dict[str, Any]],
    probabilities: Any,
    *,
    modality: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "group_id",
                "modality",
                "true_label",
                "predicted_label",
                "probability_low",
                "probability_medium",
                "probability_high",
            ],
        )
        writer.writeheader()
        for record, row in zip(records, probabilities):
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "group_id": record["group_id"],
                    "modality": modality,
                    "true_label": record["label_name"],
                    "predicted_label": OUC_CGE_CLASS_NAMES[int(row.argmax())],
                    "probability_low": f"{float(row[0]):.8f}",
                    "probability_medium": f"{float(row[1]):.8f}",
                    "probability_high": f"{float(row[2]):.8f}",
                }
            )


def run_real_classroom_benchmark(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    folds: int = 4,
    seed: int = 2026,
    frame_count: int = 12,
) -> dict[str, Any]:
    np, _, _, _ = _dependencies()
    manifest = discover_ouc_cge(dataset_root)
    audit = manifest["audit"]
    records = manifest["records"]
    scope = _provenance_scope(audit)
    labels = np.asarray([record["label"] for record in records], dtype=int)
    groups = np.asarray([record["group_id"] for record in records])
    unique_groups = np.unique(groups)
    if folds < 2 or folds > len(unique_groups):
        raise ValueError(f"folds must be between 2 and {len(unique_groups)}")
    if len(set(labels.tolist())) != len(OUC_CGE_CLASS_NAMES):
        raise ValueError("all three OUC-CGE classes are required")

    matrix, feature_names, visual_dimension, feature_diagnostics = _feature_matrix(
        manifest,
        dataset_root,
        frame_count=frame_count,
    )
    audio_availability = _audio_availability_audit(records, feature_diagnostics)
    modality_reports: dict[str, Any] = {}
    modality_probabilities: dict[str, Any] = {}
    for modality in ("visual", "audio", "fusion"):
        selected = _select_modality(matrix, modality, visual_dimension)
        probabilities, fold_reports = _grouped_oof(
            selected,
            labels,
            groups,
            folds=folds,
            seed=seed,
        )
        metrics = classification_metrics(labels, probabilities, class_names=OUC_CGE_CLASS_NAMES)
        metrics["grouped_bootstrap_95_ci"] = grouped_bootstrap_interval(
            labels, probabilities, groups, seed=seed
        )
        metrics["folds"] = fold_reports
        audio_eligible = (
            modality == "visual"
            or audio_availability["eligible_for_multimodal_accuracy_claim"]
        )
        metrics["eligible_with_respect_to_audio_coverage"] = audio_eligible
        metrics["valid_for_deployment_accuracy_claim"] = False
        if not audio_eligible:
            metrics["invalidity_reason"] = (
                "Decoded audio availability is label-confounded; this score can exploit modality missingness."
            )
        modality_reports[modality] = metrics
        modality_probabilities[modality] = probabilities

    metadata_matrix, metadata_feature_names = _encoding_metadata_matrix(records)
    metadata_probabilities, metadata_folds = _grouped_oof(
        metadata_matrix,
        labels,
        groups,
        folds=folds,
        seed=seed,
    )
    metadata_metrics = classification_metrics(
        labels,
        metadata_probabilities,
        class_names=OUC_CGE_CLASS_NAMES,
    )
    metadata_metrics["folds"] = metadata_folds
    metadata_metrics["feature_names"] = metadata_feature_names

    majority_probabilities = np.full((len(labels), len(OUC_CGE_CLASS_NAMES)), 1e-6, dtype=float)
    majority_label = int(np.bincount(labels).argmax())
    majority_probabilities[:, majority_label] = 1 - (len(OUC_CGE_CLASS_NAMES) - 1) * 1e-6
    majority_metrics = classification_metrics(
        labels, majority_probabilities, class_names=OUC_CGE_CLASS_NAMES
    )

    chosen_modality = (
        "fusion"
        if audio_availability["eligible_for_multimodal_accuracy_claim"]
        else "visual"
    )
    final_matrix = _select_modality(matrix, chosen_modality, visual_dimension)
    final_state = _train_final_model(final_matrix, labels)
    selected_names = (
        feature_names
        if chosen_modality == "fusion"
        else feature_names[:visual_dimension]
        if chosen_modality == "visual"
        else feature_names[visual_dimension:]
    )
    model_classes = final_state.pop(
        "model_classes",
        final_state.pop("classes", list(range(len(OUC_CGE_CLASS_NAMES)))),
    )
    checkpoint = {
        "schema_version": "1.0",
        "model_type": "standardized_multinomial_logistic_regression",
        "task": (
            "OUC-CGE three-class group engagement"
            if audit["provenance_verified"]
            else "OUC-CGE-compatible three-class directory labels"
        ),
        "classes": OUC_CGE_CLASS_NAMES,
        "model_classes": model_classes,
        "modality": chosen_modality,
        "feature_names": selected_names,
        "feature_configuration": {
            "frame_count": frame_count,
            "width": 64,
            "height": 36,
            "clip_seconds": 10.0,
            "sample_rate": 8000,
        },
        "dataset_fingerprint": manifest["audit"]["dataset_fingerprint"],
        "training_sample_count": len(records),
        "training_video_sha256s": sorted(record["video_sha256"] for record in records),
        "training_scope": scope["training_scope"],
        "model_scope_warning": scope["inference_warning"],
        "session_disjoint_validated": False,
        "deployment_validated": False,
        **final_state,
    }
    checkpoint_fingerprint = hashlib.sha256(
        json.dumps(checkpoint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    checkpoint["checkpoint_sha256"] = checkpoint_fingerprint

    fusion_gain = round(
        modality_reports["fusion"]["macro_f1"] - modality_reports["visual"]["macro_f1"], 4
    )
    metadata_margin = round(
        metadata_metrics["macro_f1"] - majority_metrics["macro_f1"],
        4,
    )
    nuisance_shortcut_detected = metadata_margin >= 0.2
    permutation_control = _post_feature_label_permutation_sanity(
        final_matrix,
        labels,
        groups,
        folds=folds,
        seed=seed,
        observed_macro_f1=modality_reports[chosen_modality]["macro_f1"],
    )
    report = {
        "schema_version": "1.0",
        "benchmark_kind": (
            "real_classroom_human_labeled_surrogate_group_oof_pilot"
            if audit["provenance_verified"]
            else "unverified_directory_surrogate_group_oof_pilot"
        ),
        "dataset_id": audit["dataset_id"],
        "dataset_variant": audit["dataset_variant"],
        "metric_scope": scope["metric_scope"],
        "provenance_verified": audit["provenance_verified"],
        "real_classroom_video_used": audit["real_classroom_video"],
        "independent_human_ground_truth": audit["independent_human_ground_truth"],
        "group_disjoint_evaluation": False,
        "surrogate_filename_group_disjoint_evaluation": True,
        "verified_source_group_disjoint_evaluation": False,
        "session_disjoint_evaluation": False,
        "full_official_test_set_used": False,
        "real_world_recognition_accuracy_established": False,
        "teaching_effectiveness_established": False,
        "automatic_model_inference_implemented": True,
        "primary_metric": "macro_f1",
        "fold_count": folds,
        "seed": seed,
        "usable_unique_sample_count": len(records),
        "surrogate_group_count": len(unique_groups),
        "surrogate_group_kind": "label-independent filename index; source/session relationship unknown",
        "feature_dimension": int(matrix.shape[1]),
        "visual_feature_dimension": visual_dimension,
        "audio_feature_dimension": int(matrix.shape[1] - visual_dimension),
        "majority_baseline": majority_metrics,
        "encoding_metadata_nuisance_baseline": metadata_metrics,
        "audio_availability_audit": audio_availability,
        "modality_results": modality_reports,
        "fusion_macro_f1_gain_over_visual": fusion_gain,
        "multimodal_gain_established_on_pilot": (
            audio_availability["eligible_for_multimodal_accuracy_claim"] and fusion_gain > 0
        ),
        "released_checkpoint_modality": chosen_modality,
        "sample_shortcut_risk_detected": nuisance_shortcut_detected,
        "encoding_metadata_macro_f1_gain_over_majority": metadata_margin,
        "post_feature_label_permutation_sanity": permutation_control,
        "claim_validity_checks": {
            "nuisance_metadata_shortcut_detected": nuisance_shortcut_detected,
            "nuisance_detection_rule": "encoding metadata Macro-F1 exceeds majority Macro-F1 by at least 0.20",
            "session_disjoint_identifiers_available": False,
            "verified_source_group_identifiers_available": False,
            "multimodal_audio_coverage_valid": audio_availability[
                "eligible_for_multimodal_accuracy_claim"
            ],
            "label_or_path_text_used_as_numeric_feature": False,
            "pilot_supports_deployment_accuracy_claim": False,
        },
        "quality_gates": {
            "all_videos_decodable": audit["invalid_video_count"] == 0,
            "official_sample_provenance_verified": audit["provenance_verified"],
            "exact_duplicates_removed": audit["exact_duplicate_count"] == audit["raw_file_count"] - len(records),
            "no_duplicate_label_conflicts": audit.get("duplicate_label_conflict_count", 0) == 0,
            "all_samples_have_oof_predictions": True,
            "zero_surrogate_group_overlap_each_fold": all(
                not fold["group_overlap"] for fold in modality_reports["fusion"]["folds"]
            ),
            "all_three_classes_present": len(set(labels.tolist())) == 3,
        },
        "limitations": [
            scope["dataset_limitation"],
            "The sample release has variable durations and class-correlated encoding differences; a fixed center window reduces but cannot eliminate shortcut risk.",
            "A label-free encoding-metadata baseline is reported explicitly; high performance there indicates source/encoding confounding rather than engagement recognition.",
            "No session/participant identifiers are published with these sample filenames, so cross-session generalization is not measured.",
            "The low sample videos have no decoded audio overlapping their video streams while medium/high samples do; audio and fusion scores are invalid modality-availability shortcuts.",
            "The released checkpoint therefore falls back to visual-only rather than deploying the confounded fusion model.",
            "OUC-CGE measures group engagement, not question detection, student confusion, or causal teaching effectiveness.",
            "The final checkpoint is trained on all pilot samples only and must not be treated as production-calibrated.",
        ],
    }
    output = ensure_private_directory(output_dir)
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "feature_diagnostics.json").write_text(
        json.dumps(feature_diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_predictions(
        output / "oof_predictions.csv",
        records,
        modality_probabilities[chosen_modality],
        modality=chosen_modality,
    )
    np.savez_compressed(
        output / "feature_cache.npz",
        features=matrix,
        labels=labels,
        groups=groups,
        sample_ids=np.asarray([record["sample_id"] for record in records]),
    )
    return report


def infer_classroom_engagement(
    video_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    import numpy as np

    checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    claimed_checkpoint_hash = checkpoint.get("checkpoint_sha256")
    checkpoint_without_hash = dict(checkpoint)
    checkpoint_without_hash.pop("checkpoint_sha256", None)
    computed_checkpoint_hash = hashlib.sha256(
        json.dumps(
            checkpoint_without_hash,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if claimed_checkpoint_hash != computed_checkpoint_hash:
        raise ValueError("checkpoint SHA-256 validation failed")
    path = Path(video_path)
    video_hash = file_sha256(path)
    probe = probe_video(path)
    if not probe.get("valid"):
        raise ValueError("input is not a decodable video")
    configuration = checkpoint["feature_configuration"]
    extracted = extract_multimodal_features(
        path,
        duration=float(probe["duration_seconds"]),
        start_time=float(probe.get("video_start_time_seconds", 0.0)),
        frame_count=int(configuration["frame_count"]),
        width=int(configuration["width"]),
        height=int(configuration["height"]),
        clip_seconds=float(configuration["clip_seconds"]),
        sample_rate=int(configuration["sample_rate"]),
        audio_stream_index=probe.get("selected_audio_stream_index"),
    )
    features = extracted["features"]
    extracted_names = extracted["feature_names"]
    if checkpoint["modality"] == "visual":
        features = features[: extracted["visual_dimension"]]
        extracted_names = extracted_names[: extracted["visual_dimension"]]
    elif checkpoint["modality"] == "audio":
        if not extracted["audio_present"]:
            raise ValueError("audio checkpoint requires decoded audio overlapping the video")
        features = features[extracted["visual_dimension"] :]
        extracted_names = extracted_names[extracted["visual_dimension"] :]
    elif checkpoint["modality"] == "fusion" and not extracted["audio_present"]:
        raise ValueError("fusion checkpoint requires decoded audio overlapping the video")
    if extracted_names != checkpoint["feature_names"]:
        raise ValueError("checkpoint feature schema is incompatible")
    scaled = (features - np.asarray(checkpoint["scaler_mean"])) / np.asarray(checkpoint["scaler_scale"])
    coefficient = np.asarray(checkpoint["coefficient"])
    intercept = np.asarray(checkpoint["intercept"])
    raw = scaled.reshape(1, -1) @ coefficient.T + intercept
    model_probabilities = _softmax(raw)[0]
    probabilities = np.zeros(len(checkpoint["classes"]), dtype=float)
    model_classes = checkpoint.get("model_classes", list(range(len(checkpoint["classes"]))))
    for column, class_index in enumerate(model_classes):
        probabilities[int(class_index)] = model_probabilities[column]
    predicted = int(probabilities.argmax())
    return {
        "schema_version": "1.0",
        "task": checkpoint["task"],
        "evidence_origin": "automatic_model_prediction",
        "video_sha256": video_hash,
        "checkpoint_sha256": claimed_checkpoint_hash,
        "modalities": ["visual", "audio"] if checkpoint["modality"] == "fusion" else [checkpoint["modality"]],
        "predicted_label": checkpoint["classes"][predicted],
        "confidence": round(float(probabilities[predicted]), 6),
        "probabilities": {
            label: round(float(probabilities[index]), 6)
            for index, label in enumerate(checkpoint["classes"])
        },
        "input_was_in_checkpoint_training_set": video_hash
        in set(checkpoint.get("training_video_sha256s", [])),
        "model_scope_warning": checkpoint.get(
            "model_scope_warning",
            "Checkpoint provenance scope was not recorded; production accuracy is not established.",
        ),
    }
