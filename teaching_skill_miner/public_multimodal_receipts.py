"""Build content-free public receipts from private multimodal artifacts.

The builders in this module deliberately use a field allowlist.  They never copy
lecture identifiers, source text, OCR output, frame paths, embeddings, local model
paths, or per-lecture records into the public artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .io_utils import read_json, write_json


ARM_ORDER = (
    "transcript_only",
    "transcript_audio",
    "transcript_visual",
    "full",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PUBLIC_DATASET_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")

_MANIFEST_AGGREGATE_FIELDS = (
    "duration_seconds",
    "keyframe_count",
    "nonempty_ocr_frame_count",
    "ocr_frame_with_three_or_more_accepted_words_count",
    "ocr_accepted_word_count",
    "visual_event_count",
    "fused_event_count",
    "full_timeline_sampling_passed_count",
    "caption_timeline_media_binding_verified_count",
    "semantic_feature_complete_count",
)

_ABLATION_AGGREGATE_FIELDS = (
    "lecture_count",
    "mean_internal_overall_score",
    "mean_paired_internal_score_delta",
    "median_paired_internal_score_delta",
    "mean_paired_evidence_grounding_delta",
    "lectures_with_positive_internal_score_delta",
    "lectures_with_zero_internal_score_delta",
    "lectures_with_negative_internal_score_delta",
    "lectures_with_retained_events",
    "total_retained_event_count",
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _number_or_none(value: Any, name: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _sha256(value: Any, name: str) -> str:
    digest = str(value)
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _source_hashes(
    supplied: Mapping[str, Any] | None,
    values: Mapping[str, Any],
) -> dict[str, str]:
    hashes = supplied or {
        key: _canonical_sha256(value) for key, value in values.items()
    }
    if set(hashes) != set(values):
        raise ValueError(
            "source artifact hash keys must be exactly: "
            + ", ".join(sorted(values))
        )
    return {
        key: _sha256(hashes[key], f"source_artifact_sha256.{key}")
        for key in sorted(values)
    }


def _validate_manifest_totals(
    manifest: Mapping[str, Any],
) -> tuple[int, list[Mapping[str, Any]], Mapping[str, Any]]:
    video_count = _integer(manifest.get("video_count"), "manifest.video_count", positive=True)
    videos = [
        _mapping(value, f"manifest.videos[{index}]")
        for index, value in enumerate(_list(manifest.get("videos"), "manifest.videos"))
    ]
    if len(videos) != video_count:
        raise ValueError("manifest video_count does not match videos")
    aggregate = _mapping(manifest.get("aggregate"), "manifest.aggregate")

    integer_fields = set(_MANIFEST_AGGREGATE_FIELDS) - {"duration_seconds"}
    for field in _MANIFEST_AGGREGATE_FIELDS:
        if field == "duration_seconds":
            value = _number_or_none(aggregate.get(field), f"manifest.aggregate.{field}")
            if value is None or value < 0:
                raise ValueError(f"manifest.aggregate.{field} must be non-negative")
        else:
            _integer(aggregate.get(field), f"manifest.aggregate.{field}")

    summary_sum_fields = integer_fields - {
        "full_timeline_sampling_passed_count",
        "caption_timeline_media_binding_verified_count",
        "semantic_feature_complete_count",
    }
    for field in summary_sum_fields:
        total = sum(
            _integer(
                _mapping(item.get("summary"), "manifest video summary").get(field),
                f"manifest video summary.{field}",
            )
            for item in videos
        )
        if total != aggregate[field]:
            raise ValueError(f"manifest aggregate {field} does not match lecture summaries")

    duration_total = sum(
        float(
            _number_or_none(
                _mapping(item.get("summary"), "manifest video summary").get(
                    "duration_seconds"
                ),
                "manifest video summary.duration_seconds",
            )
            or 0
        )
        for item in videos
    )
    if abs(duration_total - float(aggregate["duration_seconds"])) > 0.02:
        raise ValueError("manifest aggregate duration does not match lecture summaries")

    count_checks = {
        "full_timeline_sampling_passed_count": sum(
            _boolean(
                _mapping(item.get("summary"), "manifest video summary").get(
                    "full_timeline_sampling_passed"
                ),
                "manifest video summary.full_timeline_sampling_passed",
            )
            for item in videos
        ),
        "caption_timeline_media_binding_verified_count": sum(
            _boolean(
                _mapping(item.get("summary"), "manifest video summary").get(
                    "caption_timeline_media_binding_verified"
                ),
                "manifest video summary.caption_timeline_media_binding_verified",
            )
            for item in videos
        ),
        "semantic_feature_complete_count": sum(
            _mapping(item.get("summary"), "manifest video summary").get(
                "semantic_features_status"
            )
            == "complete_hash_bound_inference"
            for item in videos
        ),
    }
    for field, count in count_checks.items():
        if aggregate[field] != count:
            raise ValueError(f"manifest aggregate {field} does not match lecture summaries")
    return video_count, videos, aggregate


def build_public_multimodal_validation_receipt(
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
    semantic_batch: Mapping[str, Any],
    *,
    source_artifact_sha256: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a path/text/frame/embedding-free validation receipt.

    Private inputs are checked for cross-artifact consistency before any public
    summary is returned.  All output fields are constructed explicitly.
    """

    manifest = _mapping(manifest, "manifest")
    audit = _mapping(audit, "audit")
    semantic_batch = _mapping(semantic_batch, "semantic_batch")
    video_count, videos, aggregate = _validate_manifest_totals(manifest)

    dataset_id = str(manifest.get("dataset_id", ""))
    if not PUBLIC_DATASET_ID_RE.fullmatch(dataset_id):
        raise ValueError("manifest dataset_id is not a public-safe identifier")

    if audit.get("dataset_id") != manifest.get("dataset_id"):
        raise ValueError("audit and manifest dataset_id values differ")
    if _integer(audit.get("video_count"), "audit.video_count") != video_count:
        raise ValueError("audit and manifest video counts differ")
    for field in (
        "valid_transcript_count",
        "research_grade_transcript_count",
        "multimodal_ready_transcript_count",
    ):
        if _integer(audit.get(field), f"audit.{field}") != video_count:
            raise ValueError(f"audit {field} is incomplete")
    for field in (
        "dataset_structure_passed",
        "formal_empirical_ready",
        "multimodal_empirical_ready",
    ):
        if not _boolean(audit.get(field), f"audit.{field}"):
            raise ValueError(f"audit {field} must be true")
    errors = _list(audit.get("errors"), "audit.errors")
    warnings = _list(audit.get("warnings"), "audit.warnings")
    if errors:
        raise ValueError("data audit contains errors")

    course_counts = _mapping(
        audit.get("course_lesson_counts"), "audit.course_lesson_counts"
    )
    lesson_counts = sorted(
        _integer(value, "audit course lesson count", positive=True)
        for value in course_counts.values()
    )
    if sum(lesson_counts) != video_count:
        raise ValueError("audit course lesson counts do not match video_count")

    if _integer(semantic_batch.get("video_count"), "semantic_batch.video_count") != video_count:
        raise ValueError("semantic batch and manifest video counts differ")
    if not _boolean(semantic_batch.get("complete"), "semantic_batch.complete"):
        raise ValueError("semantic batch is incomplete")
    if _integer(semantic_batch.get("frame_count"), "semantic_batch.frame_count") != aggregate[
        "keyframe_count"
    ]:
        raise ValueError("semantic batch frame count differs from manifest keyframes")

    semantic_results = [
        _mapping(value, f"semantic_batch.results[{index}]")
        for index, value in enumerate(
            _list(semantic_batch.get("results"), "semantic_batch.results")
        )
    ]
    if len(semantic_results) != video_count:
        raise ValueError("semantic batch result count differs from video_count")
    manifest_result_hashes = {
        str(item.get("video_id")): _sha256(
            item.get("semantic_result_sha256"), "manifest semantic result SHA-256"
        )
        for item in videos
    }
    batch_result_hashes = {
        str(item.get("video_id")): _sha256(
            item.get("result_sha256"), "semantic batch result SHA-256"
        )
        for item in semantic_results
    }
    if len(manifest_result_hashes) != video_count or len(batch_result_hashes) != video_count:
        raise ValueError("duplicate or missing video_id in semantic result bindings")
    if manifest_result_hashes != batch_result_hashes:
        raise ValueError("semantic result hashes differ between manifest and batch receipt")
    if sum(
        _integer(item.get("frame_count"), "semantic result frame_count")
        for item in semantic_results
    ) != aggregate["keyframe_count"]:
        raise ValueError("semantic per-result frame counts differ from manifest keyframes")

    weight_hashes = {
        _sha256(item.get("weight_manifest_sha256"), "weight manifest SHA-256")
        for item in semantic_results
    }
    if len(weight_hashes) != 1:
        raise ValueError("all semantic results must use one hash-identical model manifest")

    for field in (
        "full_timeline_sampling_passed_count",
        "caption_timeline_media_binding_verified_count",
        "semantic_feature_complete_count",
    ):
        if aggregate[field] != video_count:
            raise ValueError(f"full-video validation is incomplete: {field}")
    for item in videos:
        summary = _mapping(item.get("summary"), "manifest video summary")
        if _boolean(
            summary.get("ocr_accuracy_established"),
            "manifest video summary.ocr_accuracy_established",
        ):
            raise ValueError("OCR accuracy cannot be claimed without independent labels")

    media_hashes = [
        _sha256(
            _mapping(item.get("summary"), "manifest video summary").get("media_sha256"),
            "manifest media SHA-256",
        )
        for item in videos
    ]
    if len(set(media_hashes)) != video_count:
        raise ValueError("manifest media SHA-256 values must be unique")

    manifest_claims = _mapping(manifest.get("claim_boundary"), "manifest.claim_boundary")
    semantic_claims = _mapping(
        semantic_batch.get("claim_boundary"), "semantic_batch.claim_boundary"
    )
    required_true_claims = {
        "full_video_bytes_downloaded_and_hashed": manifest_claims,
        "full_timeline_sampling_required": manifest_claims,
        "official_caption_timeline_alignment_required": manifest_claims,
        "visual_semantic_features_complete": manifest_claims,
        "visual_semantic_features_computed": semantic_claims,
    }
    for field, source in required_true_claims.items():
        if not _boolean(source.get(field), f"claim_boundary.{field}"):
            raise ValueError(f"required evidence claim is false: {field}")
    required_false_claims = {
        "audio_content_word_level_verified": manifest_claims,
        "recognition_accuracy_established": manifest_claims,
        "causal_multimodal_gain_established": manifest_claims,
        "human_ground_truth_used": semantic_claims,
    }
    for field, source in required_false_claims.items():
        if _boolean(source.get(field), f"claim_boundary.{field}"):
            raise ValueError(f"unsupported public claim is true: {field}")
    if _boolean(
        semantic_claims.get("recognition_accuracy_established"),
        "semantic_batch.claim_boundary.recognition_accuracy_established",
    ):
        raise ValueError("semantic recognition accuracy is not independently established")

    hashes = _source_hashes(
        source_artifact_sha256,
        {
            "semantic_dataset_manifest": manifest,
            "semantic_data_audit": audit,
            "visual_semantic_batch_receipt": semantic_batch,
        },
    )
    result_hash_list = sorted(batch_result_hashes.values())
    return {
        "artifact_kind": "public_full_multimodal_validation_receipt",
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "source_artifact_sha256": hashes,
        "integrity_commitments": {
            "media_set_sha256": _canonical_sha256(sorted(media_hashes)),
            "semantic_result_set_sha256": _canonical_sha256(result_hash_list),
            "visual_model_weight_manifest_sha256": next(iter(weight_hashes)),
            "semantic_result_count": len(result_hash_list),
        },
        "aggregate_validation": {
            "video_count": video_count,
            "course_count": len(lesson_counts),
            "lessons_per_course_sorted": lesson_counts,
            "total_duration_seconds": aggregate["duration_seconds"],
            "total_duration_hours": round(float(aggregate["duration_seconds"]) / 3600, 6),
            "keyframe_count": aggregate["keyframe_count"],
            "nonempty_thresholded_ocr_frame_count": aggregate[
                "nonempty_ocr_frame_count"
            ],
            "ocr_frame_with_three_or_more_accepted_words_count": aggregate[
                "ocr_frame_with_three_or_more_accepted_words_count"
            ],
            "ocr_accepted_word_count": aggregate["ocr_accepted_word_count"],
            "visual_event_count": aggregate["visual_event_count"],
            "fused_event_count": aggregate["fused_event_count"],
            "full_timeline_sampling_passed_count": aggregate[
                "full_timeline_sampling_passed_count"
            ],
            "caption_timeline_media_binding_verified_count": aggregate[
                "caption_timeline_media_binding_verified_count"
            ],
            "semantic_feature_complete_count": aggregate[
                "semantic_feature_complete_count"
            ],
            "valid_transcript_count": audit["valid_transcript_count"],
            "research_grade_transcript_count": audit[
                "research_grade_transcript_count"
            ],
            "multimodal_ready_transcript_count": audit[
                "multimodal_ready_transcript_count"
            ],
            "average_caption_segment_count": _number_or_none(
                audit.get("average_segments"), "audit.average_segments"
            ),
            "average_caption_word_count": _number_or_none(
                audit.get("average_words"), "audit.average_words"
            ),
            "audit_error_count": len(errors),
            "audit_warning_count": len(warnings),
        },
        "evidence_status": {
            "dataset_structure_passed": True,
            "formal_caption_provenance_ready": True,
            "multimodal_pipeline_ready": True,
            "full_timeline_sampling_passed_for_all_videos": (
                aggregate["full_timeline_sampling_passed_count"] == video_count
            ),
            "official_caption_timeline_media_binding_verified_for_all_videos": (
                aggregate["caption_timeline_media_binding_verified_count"]
                == video_count
            ),
            "hash_bound_visual_semantic_features_complete_for_all_videos": (
                aggregate["semantic_feature_complete_count"] == video_count
            ),
            "audio_content_word_level_verified": False,
            "independent_ocr_ground_truth_used": False,
            "independent_visual_semantic_ground_truth_used": False,
            "ocr_accuracy_established": False,
            "visual_semantic_accuracy_established": False,
            "recognition_accuracy_established": False,
            "recognition_precision_recall_f1_established": False,
            "causal_multimodal_gain_established": False,
            "deployment_accuracy_established": False,
            "learner_effect_established": False,
        },
        "content_exclusion": {
            "media_content_included": False,
            "caption_text_included": False,
            "ocr_text_included": False,
            "frame_content_included": False,
            "embedding_values_included": False,
            "per_lecture_records_included": False,
            "lecture_identifiers_included": False,
            "personal_identity_fields_included": False,
            "local_or_private_paths_included": False,
        },
    }


def build_public_multimodal_ablation_receipt(
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
    semantic_batch: Mapping[str, Any],
    ablation_report: Mapping[str, Any],
    *,
    source_artifact_sha256: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an aggregate-only four-arm ablation receipt.

    Internal rubric scores are retained as aggregate engineering measurements, but
    the receipt fails closed if the private report asserts recognition or causal
    effectiveness claims that the experiment did not establish.
    """

    # Reuse the stronger three-artifact validation and count reconciliation.
    validation = build_public_multimodal_validation_receipt(
        manifest,
        audit,
        semantic_batch,
    )
    report = _mapping(ablation_report, "ablation_report")
    video_count = validation["aggregate_validation"]["video_count"]
    paired_count = _integer(
        report.get("paired_lecture_count"), "ablation_report.paired_lecture_count"
    )
    if paired_count != video_count:
        raise ValueError("ablation paired lecture count differs from dataset video count")
    if tuple(_list(report.get("arm_order"), "ablation_report.arm_order")) != ARM_ORDER:
        raise ValueError("ablation report does not contain the required four arms")
    if not _boolean(
        report.get("all_arms_use_identical_transcript_segments"),
        "ablation_report.all_arms_use_identical_transcript_segments",
    ):
        raise ValueError("ablation arms do not use identical transcript segments")

    design = _mapping(report.get("design"), "ablation_report.design")
    if design.get("unit_of_pairing") != "lecture":
        raise ValueError("ablation unit of pairing must be lecture")
    if design.get("baseline_arm") != "transcript_only":
        raise ValueError("ablation baseline must be transcript_only")
    if _boolean(
        design.get("classroom_observations_included"),
        "ablation_report.design.classroom_observations_included",
    ):
        raise ValueError("classroom observations must be excluded from this ablation")
    if _boolean(
        design.get("model_fitting_performed"),
        "ablation_report.design.model_fitting_performed",
    ):
        raise ValueError("four-arm detector ablation must not fit a model on these lectures")

    per_lecture = _list(report.get("per_lecture"), "ablation_report.per_lecture")
    if len(per_lecture) != paired_count:
        raise ValueError("ablation per-lecture count differs from paired_lecture_count")
    if not all(
        _boolean(
            _mapping(row, "ablation per-lecture row").get(
                "paired_on_identical_transcript_segments"
            ),
            "ablation per-lecture paired segment flag",
        )
        for row in per_lecture
    ):
        raise ValueError("one or more lecture ablations changed transcript segments")
    video_ids = [str(_mapping(row, "ablation row").get("video_id", "")) for row in per_lecture]
    if not all(video_ids) or len(set(video_ids)) != paired_count:
        raise ValueError("ablation lecture identifiers are missing or duplicated")

    raw_metrics = _mapping(
        report.get("aggregate_internal_metrics"),
        "ablation_report.aggregate_internal_metrics",
    )
    if set(raw_metrics) != set(ARM_ORDER):
        raise ValueError("ablation aggregate metrics do not match the required four arms")
    public_metrics: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        source = _mapping(raw_metrics[arm], f"ablation aggregate {arm}")
        metrics: dict[str, Any] = {}
        for field in _ABLATION_AGGREGATE_FIELDS:
            value = source.get(field)
            if field.startswith("mean_") or field.startswith("median_"):
                metrics[field] = _number_or_none(value, f"ablation {arm}.{field}")
            else:
                metrics[field] = _integer(value, f"ablation {arm}.{field}")
        if metrics["lecture_count"] != paired_count:
            raise ValueError(f"ablation aggregate lecture count differs for {arm}")
        direction_total = sum(
            metrics[field]
            for field in (
                "lectures_with_positive_internal_score_delta",
                "lectures_with_zero_internal_score_delta",
                "lectures_with_negative_internal_score_delta",
            )
        )
        if direction_total != paired_count:
            raise ValueError(f"ablation paired delta direction counts differ for {arm}")
        public_metrics[arm] = metrics

    if public_metrics["transcript_only"]["total_retained_event_count"] != 0:
        raise ValueError("transcript-only arm unexpectedly retained multimodal events")
    if (
        public_metrics["transcript_audio"]["total_retained_event_count"]
        + public_metrics["transcript_visual"]["total_retained_event_count"]
        != public_metrics["full"]["total_retained_event_count"]
    ):
        raise ValueError("full-arm event count is not the sum of disjoint audio and visual arms")

    claims = _mapping(report.get("claim_boundary"), "ablation_report.claim_boundary")
    false_claim_fields = (
        "independent_event_ground_truth_used",
        "human_skill_quality_labels_used",
        "learner_outcomes_used",
        "recognition_accuracy_established",
        "recognition_precision_recall_f1_established",
        "multimodal_gain_established",
        "causal_multimodal_gain_established",
        "independent_skill_quality_gain_established",
        "teaching_effectiveness_established",
        "deployment_accuracy_established",
    )
    for field in false_claim_fields:
        if _boolean(claims.get(field), f"ablation claim_boundary.{field}"):
            raise ValueError(f"unsupported ablation claim is true: {field}")
    recognition_metrics = _mapping(
        claims.get("recognition_metrics"),
        "ablation_report.claim_boundary.recognition_metrics",
    )
    if set(recognition_metrics) != {"accuracy", "precision", "recall", "f1"}:
        raise ValueError("recognition metric keys must be accuracy/precision/recall/f1")
    if any(value is not None for value in recognition_metrics.values()):
        raise ValueError("recognition metrics must remain null without independent labels")

    hashes = _source_hashes(
        source_artifact_sha256,
        {
            "semantic_dataset_manifest": manifest,
            "semantic_data_audit": audit,
            "visual_semantic_batch_receipt": semantic_batch,
            "four_arm_ablation_report": report,
        },
    )
    transcript_fingerprints = sorted(
        _sha256(
            _mapping(row, "ablation per-lecture row").get(
                "source_transcript_fingerprint_sha256"
            ),
            "ablation source transcript fingerprint",
        )
        for row in per_lecture
    )
    return {
        "artifact_kind": "public_multimodal_four_arm_ablation_receipt",
        "schema_version": "1.0",
        "dataset_id": validation["dataset_id"],
        "source_artifact_sha256": hashes,
        "integrity_commitments": {
            "source_transcript_set_sha256": _canonical_sha256(
                transcript_fingerprints
            ),
            "aggregate_internal_metrics_sha256": _canonical_sha256(
                public_metrics
            ),
            "paired_lecture_record_set_sha256": _canonical_sha256(per_lecture),
            "paired_lecture_count": paired_count,
        },
        "design": {
            "unit_of_pairing": "lecture",
            "baseline_arm": "transcript_only",
            "arm_order": list(ARM_ORDER),
            "identical_transcript_segments_across_arms": True,
            "classroom_observations_included": False,
            "model_fitting_performed": False,
            "aggregation": "unweighted_macro_mean_of_lecture_level_paired_differences",
        },
        "aggregate_internal_metrics": public_metrics,
        "metric_boundary": {
            "internal_overall_score_is_accuracy": False,
            "internal_score_delta_is_recognition_gain": False,
            "detector_output_count_is_correct_detection_count": False,
            "recognition_accuracy_established": False,
            "recognition_precision_established": False,
            "recognition_recall_established": False,
            "recognition_f1_established": False,
            "independent_multimodal_gain_established": False,
            "causal_multimodal_gain_established": False,
            "deployment_accuracy_established": False,
            "learner_effect_established": False,
            "recognition_metrics": {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
            },
            "allowed_interpretation": (
                "Aggregate paired changes in deterministic output counts, exact "
                "evidence-reference consistency, and the project's internal rubric."
            ),
            "prohibited_interpretation": (
                "Internal rubric scores and deltas are not Accuracy, Precision, Recall, "
                "F1, independently validated multimodal gain, deployment accuracy, or "
                "learner benefit."
            ),
        },
        "content_exclusion": {
            "media_content_included": False,
            "caption_text_included": False,
            "ocr_text_included": False,
            "frame_content_included": False,
            "embedding_values_included": False,
            "per_lecture_records_included": False,
            "lecture_identifiers_included": False,
            "personal_identity_fields_included": False,
            "local_or_private_paths_included": False,
        },
    }


def write_public_multimodal_receipts(
    *,
    manifest_path: str | Path,
    audit_path: str | Path,
    semantic_batch_path: str | Path,
    ablation_report_path: str | Path,
    validation_output_path: str | Path,
    ablation_output_path: str | Path,
) -> tuple[Path, Path]:
    """Load four private artifacts, validate them, and write two public receipts."""

    paths = {
        "semantic_dataset_manifest": Path(manifest_path),
        "semantic_data_audit": Path(audit_path),
        "visual_semantic_batch_receipt": Path(semantic_batch_path),
        "four_arm_ablation_report": Path(ablation_report_path),
    }
    values = {key: read_json(path) for key, path in paths.items()}
    hashes = {key: _file_sha256(path) for key, path in paths.items()}
    validation = build_public_multimodal_validation_receipt(
        values["semantic_dataset_manifest"],
        values["semantic_data_audit"],
        values["visual_semantic_batch_receipt"],
        source_artifact_sha256={
            key: hashes[key]
            for key in (
                "semantic_dataset_manifest",
                "semantic_data_audit",
                "visual_semantic_batch_receipt",
            )
        },
    )
    ablation = build_public_multimodal_ablation_receipt(
        values["semantic_dataset_manifest"],
        values["semantic_data_audit"],
        values["visual_semantic_batch_receipt"],
        values["four_arm_ablation_report"],
        source_artifact_sha256=hashes,
    )
    validation_target = write_json(validation_output_path, validation)
    ablation_target = write_json(ablation_output_path, ablation)
    return validation_target, ablation_target
