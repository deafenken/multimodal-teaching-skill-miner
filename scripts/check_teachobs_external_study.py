#!/usr/bin/env python3
"""Fail-closed postcondition checks for the TeachObs external-study runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DATASET_ID = "teachobs_v0_1_human_validated"
SOURCE_COMMIT = "96c251ae09e79edd06a3a9bbaaa8b8f7fe99a15c"
MEDIA_PLAN_SCHEMA = "teaching_skill_miner.teachobs_private_media_plan.v1"
MEDIA_MANIFEST_SCHEMA = "teaching_skill_miner.teachobs_private_media_manifest.v1"
FEATURE_MANIFEST_SCHEMA = "teaching_skill_miner.teachobs_private_feature_manifest.v1"
FEATURE_FAILURE_SCHEMA = "teaching_skill_miner.teachobs_private_feature_failures.v1"
ARM_ORDER = (
    "transcript_only",
    "transcript_audio",
    "transcript_visual",
    "full",
)
FULL_23_TRAIN_7_TEST_PROFILE = "full_23_train_7_test"
PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE = "paper_track1_23_train_6_test"
BENCHMARK_INPUT_SCHEMA = "teaching_skill_miner.teachobs_benchmark_input.v1"
EVALUATION_PROFILES = {
    FULL_23_TRAIN_7_TEST_PROFILE: {
        "lesson_count": 30,
        "feature_lesson_count": 30,
        "scene_count": 5158,
        "test_lesson_count": 7,
        "test_scene_count": 1312,
        "bootstrap_cluster_count": 7,
        "missing_lesson_ids": [],
    },
    PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE: {
        "lesson_count": 29,
        "feature_lesson_count": 29,
        "scene_count": 4945,
        "test_lesson_count": 6,
        "test_scene_count": 1099,
        "bootstrap_cluster_count": 6,
        "missing_lesson_ids": ["S4"],
    },
}


def _evaluation_profile(value: str) -> dict[str, Any]:
    try:
        return EVALUATION_PROFILES[value]
    except KeyError as exc:
        raise StageCheckError(f"unknown TeachObs evaluation profile: {value}") from exc


class StageCheckError(ValueError):
    """Raised when a stage artifact cannot support the next stage."""


def _load_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageCheckError(f"JSON artifact must be an object: {source}")
    return value


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _media_canonical_sha256(value: Any) -> str:
    """Match teachobs_media's canonical JSON contract, including Unicode escaping."""

    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise StageCheckError(message)


def check_annotations(audit_path: str, receipt_path: str) -> dict[str, Any]:
    audit = _load_object(audit_path)
    receipt = _load_object(receipt_path)
    _expect(
        audit.get("artifact_kind") == "teachobs_annotation_dataset_audit",
        "annotation audit kind mismatch",
    )
    _expect(audit.get("dataset_id") == DATASET_ID, "annotation dataset mismatch")
    _expect(audit.get("source_commit") == SOURCE_COMMIT, "annotation pin mismatch")
    _expect(audit.get("import_complete") is True, "annotation import is incomplete")
    _expect(audit.get("official_split_verified") is True, "split is not verified")
    _expect(
        audit.get("consensus_gold_schema_verified") is True,
        "consensus-gold schema is not verified",
    )
    expected = {
        "lesson_count": 30,
        "train_lesson_count": 23,
        "test_lesson_count": 7,
        "scene_count": 5158,
        "code_count": 39,
        "visual_code_count": 20,
        "nonvisual_code_count": 19,
    }
    for key, value in expected.items():
        _expect(audit.get(key) == value, f"annotation {key} mismatch")
    _expect(
        receipt.get("artifact_kind") == "public_teachobs_annotation_receipt",
        "annotation receipt kind mismatch",
    )
    _expect(receipt.get("source_commit") == SOURCE_COMMIT, "receipt pin mismatch")
    _expect(
        receipt.get("private_audit_sha256") == _file_sha256(audit_path),
        "annotation receipt is not bound to the current private audit",
    )
    boundary = receipt.get("claim_boundary")
    _expect(isinstance(boundary, dict), "annotation claim boundary is missing")
    for key in (
        "confirmatory_multimodal_gain_established",
        "deployment_accuracy_established",
        "learner_effect_established",
    ):
        _expect(boundary.get(key) is False, f"unsafe annotation claim: {key}")
    return {"stage": "annotations", "complete": True, **expected}


def check_captions(audit_path: str, receipt_path: str) -> dict[str, Any]:
    audit = _load_object(audit_path)
    receipt = _load_object(receipt_path)
    _expect(
        audit.get("schema")
        == "teaching_skill_miner.teachobs_private_caption_audit.v1",
        "caption audit schema mismatch",
    )
    _expect(audit.get("dataset_id") == DATASET_ID, "caption dataset mismatch")
    _expect(
        audit.get("repository_commit") == SOURCE_COMMIT,
        "caption repository pin mismatch",
    )
    _expect(
        receipt.get("private_audit_sha256") == _file_sha256(audit_path),
        "caption receipt is not bound to the current private audit",
    )
    aggregate = audit.get("aggregate")
    claims = audit.get("claims")
    _expect(isinstance(aggregate, dict), "caption aggregate is missing")
    _expect(isinstance(claims, dict), "caption claims are missing")
    _expect(aggregate.get("lesson_count") == 30, "caption lesson count mismatch")
    audited = aggregate.get("caption_timeline_audited_lesson_count")
    failed = aggregate.get("caption_unavailable_or_failed_lesson_count")
    _expect(
        isinstance(audited, int)
        and isinstance(failed, int)
        and audited + failed == 30,
        "caption coverage denominator does not reconcile",
    )
    for key in (
        "caption_content_accuracy_established",
        "word_error_rate_established",
        "independent_human_transcript_audit_completed",
        "double_annotation_reliability_established",
    ):
        _expect(claims.get(key) is False, f"unsafe caption claim: {key}")
    formal = aggregate.get("formal_caption_timeline_audit_completed") is True
    _expect(
        claims.get("formal_caption_timeline_audit_completed") is formal,
        "caption completion flag mismatch",
    )
    return {
        "stage": "captions",
        "artifact_valid": True,
        "formal_caption_timeline_audit_completed": formal,
        "audited_lesson_count": audited,
        "pending_lesson_count": failed,
    }


def _asr_canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_safe_asr_object(path: str | Path, description: str) -> dict[str, Any]:
    source = Path(path)
    _expect(
        not source.is_symlink() and source.is_file(),
        f"{description} is missing or unsafe",
    )
    return _load_object(source)


def check_asr(
    receipt_path: str,
    media_manifest_path: str,
    caption_audit_path: str,
    *,
    job_manifest_path: str | None = None,
    import_audit_path: str | None = None,
    coverage_matrix_path: str | None = None,
    evaluation_profile: str | None = None,
) -> dict[str, Any]:
    """Verify an aggregate ASR receipt against its current private inputs."""

    from teaching_skill_miner.teachobs_asr_handoff import (
        COVERAGE_MATRIX_SCHEMA,
        IMPORT_AUDIT_SCHEMA,
        JOB_MANIFEST_SCHEMA,
        MEDIA_MANIFEST_SCHEMA,
        PUBLIC_RECEIPT_SCHEMA,
        validate_teachobs_asr_job_manifest,
    )
    from teaching_skill_miner.teachobs_captions import PRIVATE_AUDIT_SCHEMA

    profile = (
        _evaluation_profile(evaluation_profile)
        if evaluation_profile is not None
        else None
    )
    receipt = _load_safe_asr_object(receipt_path, "ASR public receipt")
    media = _load_safe_asr_object(media_manifest_path, "ASR media manifest")
    captions = _load_safe_asr_object(caption_audit_path, "ASR caption audit")
    _expect(media.get("schema") == MEDIA_MANIFEST_SCHEMA, "ASR media schema mismatch")
    _expect(media.get("dataset_id") == DATASET_ID, "ASR media dataset mismatch")
    _expect(
        media.get("selected_lesson_count") == 30
        and isinstance(media.get("lessons"), list)
        and 0 < len(media["lessons"]) <= 30,
        "ASR media denominator mismatch",
    )
    _expect(
        captions.get("schema") == PRIVATE_AUDIT_SCHEMA,
        "ASR caption schema mismatch",
    )
    _expect(captions.get("dataset_id") == DATASET_ID, "ASR caption dataset mismatch")
    _expect(
        isinstance(captions.get("records"), list)
        and len(captions["records"]) == 30,
        "ASR caption denominator mismatch",
    )
    _expect(
        receipt.get("artifact_kind")
        == "teachobs_aggregate_audited_asr_handoff_receipt",
        "ASR receipt kind mismatch",
    )
    _expect(receipt.get("schema") == PUBLIC_RECEIPT_SCHEMA, "ASR receipt schema mismatch")

    source_hashes = receipt.get("source_hashes")
    _expect(isinstance(source_hashes, dict), "ASR receipt source hashes are missing")
    _expect(
        source_hashes.get("media_manifest_file_sha256")
        == _file_sha256(media_manifest_path),
        "ASR receipt is not bound to the current media manifest",
    )
    _expect(
        source_hashes.get("caption_audit_file_sha256")
        == _file_sha256(caption_audit_path),
        "ASR receipt is not bound to the current caption audit",
    )

    aggregate = receipt.get("aggregate")
    _expect(isinstance(aggregate, dict), "ASR receipt aggregate is missing")
    integer_counts: dict[str, int] = {}
    for key in (
        "expected_lesson_count",
        "covered_lesson_count",
        "pending_lesson_count",
        "platform_creator_provided_caption_lesson_count",
        "platform_automatic_caption_lesson_count",
        "audited_asr_fallback_lesson_count",
        "pending_no_audited_source_lesson_count",
        "asr_job_count",
        "valid_asr_result_count",
        "pending_asr_result_count",
    ):
        value = aggregate.get(key)
        _expect(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"ASR receipt count is invalid: {key}",
        )
        integer_counts[key] = value
    _expect(
        integer_counts["expected_lesson_count"] == 30,
        "ASR receipt lesson denominator mismatch",
    )
    _expect(
        integer_counts["covered_lesson_count"]
        + integer_counts["pending_lesson_count"]
        == 30,
        "ASR receipt coverage denominator does not reconcile",
    )
    _expect(
        integer_counts["covered_lesson_count"]
        == integer_counts["platform_creator_provided_caption_lesson_count"]
        + integer_counts["platform_automatic_caption_lesson_count"]
        + integer_counts["audited_asr_fallback_lesson_count"],
        "ASR receipt covered-source counts do not reconcile",
    )
    _expect(
        integer_counts["pending_lesson_count"]
        == integer_counts["pending_no_audited_source_lesson_count"],
        "ASR receipt pending-source count does not reconcile",
    )
    _expect(
        integer_counts["valid_asr_result_count"]
        + integer_counts["pending_asr_result_count"]
        == integer_counts["asr_job_count"],
        "ASR receipt job-result counts do not reconcile",
    )
    coverage_complete = integer_counts["covered_lesson_count"] == 30
    _expect(
        aggregate.get("transcript_source_coverage_complete") is coverage_complete,
        "ASR receipt coverage-completion flag mismatch",
    )

    evidence = receipt.get("evidence_status")
    _expect(isinstance(evidence, dict), "ASR receipt evidence status is missing")
    _expect(
        evidence.get("transcript_source_coverage_complete") is coverage_complete,
        "ASR evidence coverage-completion flag mismatch",
    )
    provenance_complete = bool(
        integer_counts["valid_asr_result_count"]
        and not integer_counts["pending_asr_result_count"]
    )
    _expect(
        evidence.get("asr_provenance_and_timeline_audit_completed")
        is provenance_complete,
        "ASR evidence provenance-completion flag mismatch",
    )
    for key in (
        "asr_is_official_caption",
        "independent_human_content_audit_completed",
        "content_accuracy_established",
        "word_error_rate_established",
        "recognition_accuracy_established",
    ):
        _expect(evidence.get(key) is False, f"unsafe ASR receipt claim: {key}")
    exclusions = receipt.get("content_exclusion")
    _expect(isinstance(exclusions, dict), "ASR receipt content exclusion is missing")
    for key in (
        "media_bytes_included",
        "transcript_text_included",
        "source_urls_included",
        "lesson_or_video_ids_included",
        "filesystem_paths_included",
        "per_lesson_records_included",
        "runtime_host_identity_included",
    ):
        _expect(exclusions.get(key) is False, f"unsafe ASR receipt content: {key}")

    optional_paths = {
        "job_manifest_file_sha256": job_manifest_path,
        "asr_import_audit_file_sha256": import_audit_path,
        "coverage_matrix_file_sha256": coverage_matrix_path,
    }
    for field, path in optional_paths.items():
        claimed = source_hashes.get(field)
        if path is None:
            _expect(claimed is None, f"ASR receipt {field} requires its private input")
        else:
            _expect(
                claimed == _file_sha256(path),
                f"ASR receipt optional source hash mismatch: {field}",
            )
    _expect(
        (import_audit_path is None) == (coverage_matrix_path is None),
        "ASR import audit and coverage matrix must be checked together",
    )
    _expect(
        import_audit_path is None or job_manifest_path is not None,
        "ASR imported evidence requires its job manifest",
    )

    job_manifest: dict[str, Any] | None = None
    if job_manifest_path is not None:
        job_manifest = _load_safe_asr_object(job_manifest_path, "ASR job manifest")
        _expect(
            job_manifest.get("schema") == JOB_MANIFEST_SCHEMA,
            "ASR job manifest schema mismatch",
        )
        validate_teachobs_asr_job_manifest(job_manifest)
        _expect(
            job_manifest.get("source_media_manifest_file_sha256")
            == _file_sha256(media_manifest_path),
            "ASR job manifest is not bound to the current media manifest",
        )
        _expect(
            job_manifest.get("source_caption_audit_file_sha256")
            == _file_sha256(caption_audit_path),
            "ASR job manifest is not bound to the current caption audit",
        )
        _expect(
            integer_counts["asr_job_count"] == len(job_manifest["jobs"]),
            "ASR receipt job count differs from the job manifest",
        )
        for field, section in (
            ("model_contract_sha256", "model"),
            ("decoding_contract_sha256", "decoding_config"),
            ("runtime_contract_sha256", "runtime_contract"),
        ):
            _expect(
                source_hashes.get(field) == _asr_canonical_sha256(job_manifest[section]),
                f"ASR receipt contract hash mismatch: {field}",
            )
    else:
        _expect(integer_counts["asr_job_count"] == 0, "ASR job manifest is missing")
        for field in (
            "model_contract_sha256",
            "decoding_contract_sha256",
            "runtime_contract_sha256",
        ):
            _expect(
                source_hashes.get(field) is None,
                f"ASR receipt {field} requires a job manifest",
            )

    completed = False
    if import_audit_path is not None and coverage_matrix_path is not None:
        audit = _load_safe_asr_object(import_audit_path, "ASR import audit")
        matrix = _load_safe_asr_object(coverage_matrix_path, "ASR coverage matrix")
        _expect(audit.get("schema") == IMPORT_AUDIT_SCHEMA, "ASR import schema mismatch")
        _expect(audit.get("dataset_id") == DATASET_ID, "ASR import dataset mismatch")
        audit_unsigned = {key: value for key, value in audit.items() if key != "audit_sha256"}
        _expect(
            audit.get("audit_sha256") == _asr_canonical_sha256(audit_unsigned),
            "ASR import canonical hash mismatch",
        )
        _expect(
            audit.get("source_media_manifest_file_sha256")
            == _file_sha256(media_manifest_path),
            "ASR import is not bound to the current media manifest",
        )
        _expect(
            audit.get("source_caption_audit_file_sha256")
            == _file_sha256(caption_audit_path),
            "ASR import is not bound to the current caption audit",
        )
        _expect(
            audit.get("job_manifest_file_sha256") == _file_sha256(job_manifest_path),
            "ASR import is not bound to the current job manifest",
        )
        for field in (
            "model_contract_sha256",
            "decoding_contract_sha256",
            "runtime_contract_sha256",
        ):
            _expect(
                audit.get(field) == source_hashes.get(field),
                f"ASR import contract hash mismatch: {field}",
            )
        audit_aggregate = audit.get("aggregate")
        _expect(isinstance(audit_aggregate, dict), "ASR import aggregate is missing")
        audit_counts: dict[str, int] = {}
        for key in ("job_count", "valid_result_count", "pending_result_count"):
            value = audit_aggregate.get(key)
            _expect(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0,
                f"ASR import count is invalid: {key}",
            )
            audit_counts[key] = value
        for receipt_key, audit_key in (
            ("asr_job_count", "job_count"),
            ("valid_asr_result_count", "valid_result_count"),
            ("pending_asr_result_count", "pending_result_count"),
        ):
            _expect(
                integer_counts[receipt_key] == audit_aggregate.get(audit_key),
                f"ASR receipt/import count mismatch: {receipt_key}",
            )
        _expect(
            audit_counts["job_count"]
            == audit_counts["valid_result_count"]
            + audit_counts["pending_result_count"],
            "ASR import job-result counts do not reconcile",
        )
        audit_complete = bool(
            audit_counts["job_count"]
            and audit_counts["valid_result_count"] == audit_counts["job_count"]
        )
        _expect(
            audit_aggregate.get("asr_job_set_complete") is audit_complete,
            "ASR import completion flag mismatch",
        )
        _expect(
            isinstance(audit.get("records"), list)
            and len(audit["records"]) == audit_counts["job_count"],
            "ASR import record count mismatch",
        )
        audit_claims = audit.get("claims")
        _expect(isinstance(audit_claims, dict), "ASR import claims are missing")
        _expect(
            audit_claims.get("asr_provenance_and_timeline_audit_completed")
            is audit_complete,
            "ASR import provenance-completion flag mismatch",
        )
        for key in (
            "asr_is_official_caption",
            "independent_human_content_audit_completed",
            "content_accuracy_established",
            "word_error_rate_established",
        ):
            _expect(audit_claims.get(key) is False, f"unsafe ASR import claim: {key}")

        _expect(
            matrix.get("schema") == COVERAGE_MATRIX_SCHEMA,
            "ASR coverage matrix schema mismatch",
        )
        _expect(matrix.get("dataset_id") == DATASET_ID, "ASR matrix dataset mismatch")
        matrix_unsigned = {
            key: value for key, value in matrix.items() if key != "matrix_sha256"
        }
        _expect(
            matrix.get("matrix_sha256") == _asr_canonical_sha256(matrix_unsigned),
            "ASR coverage matrix canonical hash mismatch",
        )
        matrix_aggregate = matrix.get("aggregate")
        _expect(isinstance(matrix_aggregate, dict), "ASR matrix aggregate is missing")
        for key in (
            "expected_lesson_count",
            "covered_lesson_count",
            "pending_lesson_count",
            "platform_creator_provided_caption_lesson_count",
            "platform_automatic_caption_lesson_count",
            "audited_asr_fallback_lesson_count",
            "pending_no_audited_source_lesson_count",
            "transcript_source_coverage_complete",
        ):
            _expect(
                aggregate.get(key) == matrix_aggregate.get(key),
                f"ASR receipt/matrix aggregate mismatch: {key}",
            )
        _expect(
            isinstance(matrix.get("rows"), list) and len(matrix["rows"]) == 30,
            "ASR coverage matrix denominator mismatch",
        )
        if profile is not None:
            expected_pending_ids = list(profile["missing_lesson_ids"])
            rows = matrix["rows"]
            _expect(
                [row.get("lesson_id") for row in rows if isinstance(row, dict)]
                == [f"S{index}" for index in range(1, 31)],
                "ASR coverage matrix lesson order mismatch",
            )
            pending_ids = [
                row["lesson_id"]
                for row in rows
                if row.get("selected_source_tier")
                == "pending_no_audited_source"
            ]
            expected_asr_count = int(profile["lesson_count"]) - 23
            _expect(
                pending_ids == expected_pending_ids,
                "ASR pending lessons differ from the evaluation profile",
            )
            _expect(
                matrix_aggregate.get("covered_lesson_count")
                == profile["lesson_count"],
                "ASR covered lessons differ from the evaluation profile",
            )
            _expect(
                matrix_aggregate.get(
                    "platform_creator_provided_caption_lesson_count"
                )
                + matrix_aggregate.get("platform_automatic_caption_lesson_count")
                == 23,
                "ASR profile requires the fixed 23-lesson platform-caption set",
            )
            _expect(
                matrix_aggregate.get("audited_asr_fallback_lesson_count")
                == expected_asr_count
                and integer_counts["asr_job_count"] == expected_asr_count
                and integer_counts["valid_asr_result_count"] == expected_asr_count
                and integer_counts["pending_asr_result_count"] == 0,
                "ASR imported job set differs from the evaluation profile",
            )
        matrix_claims = matrix.get("claims")
        _expect(isinstance(matrix_claims, dict), "ASR matrix claims are missing")
        _expect(
            matrix_claims.get("asr_provenance_and_timeline_audit_completed")
            is bool(matrix_aggregate.get("audited_asr_fallback_lesson_count")),
            "ASR matrix provenance-completion flag mismatch",
        )
        for key in (
            "asr_is_official_caption",
            "independent_human_content_audit_completed",
            "content_accuracy_established",
            "word_error_rate_established",
        ):
            _expect(matrix_claims.get(key) is False, f"unsafe ASR matrix claim: {key}")
        completed = bool(
            matrix_aggregate.get("transcript_source_coverage_complete")
            and audit_complete
        )
    else:
        _expect(
            integer_counts["valid_asr_result_count"] == 0,
            "ASR results require an import audit",
        )
        _expect(
            integer_counts["pending_asr_result_count"]
            == integer_counts["asr_job_count"],
            "pending ASR receipt result count mismatch",
        )
        media_ready = sum(
            bool(item.get("media_sha256") and item.get("media_path"))
            for item in media["lessons"]
            if isinstance(item, dict)
        )
        _expect(
            aggregate.get("hash_bound_private_media_lesson_count") == media_ready,
            "pending ASR receipt media-ready count mismatch",
        )
    _expect(
        evidence.get("handoff_status") == ("completed" if completed else "pending"),
        "ASR receipt handoff status mismatch",
    )
    return {
        "stage": "asr",
        "artifact_valid": True,
        "evaluation_profile": evaluation_profile,
        "handoff_status": evidence["handoff_status"],
        "covered_lesson_count": integer_counts["covered_lesson_count"],
        "pending_lesson_count": integer_counts["pending_lesson_count"],
        "asr_job_count": integer_counts["asr_job_count"],
        "valid_asr_result_count": integer_counts["valid_asr_result_count"],
        "content_accuracy_established": False,
        "word_error_rate_established": False,
        "recognition_accuracy_established": False,
    }


def check_media(
    plan_path: str,
    manifest_path: str,
    *,
    source_override_manifest: str | None,
    evaluation_profile: str = FULL_23_TRAIN_7_TEST_PROFILE,
) -> dict[str, Any]:
    profile = _evaluation_profile(evaluation_profile)
    plan = _load_object(plan_path)
    manifest = _load_object(manifest_path)
    _expect(plan.get("schema") == MEDIA_PLAN_SCHEMA, "media plan schema mismatch")
    plan_unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    _expect(
        plan.get("plan_sha256") == _media_canonical_sha256(plan_unsigned),
        "media plan canonical hash mismatch",
    )
    _expect(
        manifest.get("schema") == MEDIA_MANIFEST_SCHEMA,
        "media manifest schema mismatch",
    )
    provenance = plan.get("repository_provenance")
    _expect(isinstance(provenance, dict), "media plan provenance is missing")
    _expect(
        provenance.get("repository_commit") == SOURCE_COMMIT,
        "media plan repository pin mismatch",
    )
    _expect(plan.get("lesson_count") == 30, "media plan lesson count mismatch")
    _expect(plan.get("scene_count") == 5158, "media plan scene count mismatch")
    _expect(manifest.get("dataset_id") == DATASET_ID, "media dataset mismatch")
    _expect(
        manifest.get("plan_sha256") == plan.get("plan_sha256"),
        "media manifest is not bound to the current plan",
    )
    _expect(manifest.get("selected_lesson_count") == 30, "media selection mismatch")
    expected_media_count = int(profile["lesson_count"])
    paper_profile = evaluation_profile == PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE
    expected_complete = not paper_profile
    _expect(
        manifest.get("selected_complete") is expected_complete,
        "media selection completeness differs from the evaluation profile",
    )
    _expect(
        manifest.get("downloaded_lesson_count_total") == expected_media_count,
        "media files do not exactly cover the evaluation profile",
    )
    if paper_profile:
        lessons = manifest.get("lessons")
        expected_ids = [f"S{index}" for index in range(1, 31) if index != 4]
        _expect(isinstance(lessons, list), "paper profile media rows are missing")
        _expect(
            [
                row.get("lesson_id") if isinstance(row, dict) else None
                for row in lessons
            ]
            == expected_ids,
            "paper profile must have the unique S4 media hole",
        )
    active_count = manifest.get("active_source_override_count")
    active_digest = manifest.get("active_source_override_manifest_sha256")
    boundary = manifest.get("claim_boundary")
    _expect(isinstance(boundary, dict), "media claim boundary is missing")
    if source_override_manifest is None:
        _expect(active_count == 0, "an unacknowledged source override is active")
        _expect(active_digest is None, "unexpected source-override binding")
        _expect(
            manifest.get("override_source_terms_acknowledged") is False,
            "override acknowledgement is set without an override",
        )
        _expect(
            boundary.get("candidate_same_content_mirror_count") == 0,
            "a candidate mirror remains in the no-override media set",
        )
        _expect(
            boundary.get("candidate_same_content_mirror_used") is False,
            "a candidate mirror remains active without explicit authorization",
        )
    else:
        _expect(
            isinstance(active_count, int) and active_count > 0,
            "explicit source override was not activated",
        )
        _expect(
            active_digest == _file_sha256(source_override_manifest),
            "media manifest is not bound to the explicit override file",
        )
        _expect(
            manifest.get("override_source_terms_acknowledged") is True,
            "override source terms were not recorded",
        )
    return {
        "stage": "media",
        "complete": True,
        "evaluation_profile": evaluation_profile,
        "lesson_count": expected_media_count,
        "scene_count": int(profile["scene_count"]),
        "unique_missing_lesson_ids": list(profile["missing_lesson_ids"]),
        "source_override_count": active_count,
    }


def check_features(
    manifest_path: str,
    media_manifest_path: str,
    *,
    evaluation_profile: str = FULL_23_TRAIN_7_TEST_PROFILE,
) -> dict[str, Any]:
    profile = _evaluation_profile(evaluation_profile)
    manifest = _load_object(manifest_path)
    media_manifest = _load_object(media_manifest_path)
    _expect(
        manifest.get("schema") == FEATURE_MANIFEST_SCHEMA,
        "feature manifest schema mismatch",
    )
    _expect(
        media_manifest.get("schema") == MEDIA_MANIFEST_SCHEMA,
        "feature media-manifest schema mismatch",
    )
    _expect(manifest.get("dataset_id") == DATASET_ID, "feature dataset mismatch")
    _expect(
        manifest.get("plan_sha256") == media_manifest.get("plan_sha256"),
        "features are not bound to the current media plan",
    )
    expected_lesson_count = int(profile["feature_lesson_count"])
    expected_scene_count = int(profile["scene_count"])
    missing_ids = list(profile["missing_lesson_ids"])
    paper_profile = evaluation_profile == PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE
    expected_complete = not paper_profile
    _expect(
        manifest.get("lesson_count") == expected_lesson_count,
        "feature lesson count mismatch",
    )
    _expect(
        manifest.get("expected_lesson_count") == 30,
        "feature denominator mismatch",
    )
    _expect(
        manifest.get("scene_count") == expected_scene_count,
        "feature scene count mismatch",
    )
    _expect(
        manifest.get("complete") is expected_complete,
        "feature extraction is incomplete or differs from the evaluation profile",
    )
    _expect(
        manifest.get("failed_lesson_count") == len(missing_ids),
        "feature failure count differs from the evaluation profile",
    )
    _expect(
        manifest.get("failed_lesson_ids", []) == missing_ids,
        "feature failures are not the exact profile exception",
    )
    _expect(
        manifest.get("media_manifest_sha256")
        == _media_canonical_sha256(media_manifest),
        "features are not bound to the current media manifest",
    )
    for key in (
        "audio_statistics_included",
        "visual_evidence_included",
        "ocr_requested",
        "clip_embeddings_requested",
        "clip_embeddings_included",
    ):
        _expect(manifest.get(key) is True, f"required feature block missing: {key}")
    if paper_profile:
        expected_ids = [f"S{index}" for index in range(1, 31) if index != 4]
        lessons = manifest.get("lessons")
        _expect(isinstance(lessons, list), "paper profile feature rows are missing")
        _expect(
            [
                row.get("lesson_id") if isinstance(row, dict) else None
                for row in lessons
            ]
            == expected_ids,
            "paper profile must have the unique S4 feature hole",
        )
        for key, expected in (
            ("clip_embeddings_complete_for_all_lessons", False),
            ("combined_clip_complete", False),
            ("combined_clip_lesson_count", 29),
            ("combined_clip_expected_lesson_count", 30),
            ("combined_clip_lesson_ids", expected_ids),
            ("combined_clip_failed_lesson_ids", ["S4"]),
            ("combined_clip_frame_count", 4945),
        ):
            _expect(
                manifest.get(key) == expected,
                f"paper profile combined CLIP binding mismatch: {key}",
            )
        relative_failure = manifest.get("failure_records_path")
        _expect(
            isinstance(relative_failure, str) and relative_failure,
            "paper profile feature failure receipt is missing",
        )
        relative_path = Path(relative_failure)
        _expect(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            "paper profile feature failure receipt path is unsafe",
        )
        failure_path = Path(manifest_path).resolve().parent / relative_path
        _expect(
            failure_path.is_file() and not failure_path.is_symlink(),
            "paper profile feature failure receipt is unsafe",
        )
        _expect(
            _file_sha256(failure_path) == manifest.get("failure_records_sha256"),
            "paper profile feature failure receipt hash mismatch",
        )
        failure = _load_object(failure_path)
        failure_rows = failure.get("failures")
        _expect(
            failure.get("schema") == FEATURE_FAILURE_SCHEMA
            and failure.get("dataset_id") == DATASET_ID
            and failure.get("plan_sha256") == manifest.get("plan_sha256")
            and failure.get("media_manifest_sha256")
            == manifest.get("media_manifest_sha256")
            and failure.get("failure_count") == 1
            and isinstance(failure_rows, list)
            and len(failure_rows) == 1
            and isinstance(failure_rows[0], dict)
            and failure_rows[0].get("lesson_id") == "S4",
            "paper profile feature failure receipt is not the unique S4 hole",
        )
    boundary = manifest.get("claim_boundary")
    _expect(isinstance(boundary, dict), "feature claim boundary is missing")
    _expect(
        boundary.get("recognition_accuracy_established") is False,
        "feature extraction cannot establish recognition accuracy",
    )
    _expect(
        boundary.get("multimodal_gain_established") is False,
        "feature extraction cannot establish multimodal gain",
    )
    if paper_profile:
        for key, expected in (
            ("full_scene_midpoint_frames_extracted", False),
            ("audio_statistics_computed", False),
            ("image_metrics_computed", False),
            ("adjacent_visual_events_inferred", False),
            ("clip_visual_embeddings_computed", True),
            ("clip_visual_embeddings_complete_for_all_lessons", False),
            ("clip_visual_embeddings_partial_success_only", True),
        ):
            _expect(
                boundary.get(key) is expected,
                f"paper profile feature claim mismatch: {key}",
            )
    return {
        "stage": "features",
        "complete": True,
        "evaluation_profile": evaluation_profile,
        "lesson_count": expected_lesson_count,
        "scene_count": expected_scene_count,
        "unique_missing_lesson_ids": missing_ids,
        "ocr_requested": True,
        "clip_embeddings_included": True,
    }


def check_transcripts(
    manifest_path: str,
    receipt_path: str,
    *,
    evaluation_profile: str = PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
) -> dict[str, Any]:
    _expect(
        evaluation_profile == PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
        "audited transcript materialization requires the paper Track 1 profile",
    )
    from teaching_skill_miner.teachobs_transcript_materialization import (
        ASR_TIMELINE_INTERSECTION_POLICY,
        PLATFORM_CUE_TIMELINE_INTERSECTION_POLICY,
        build_public_teachobs_transcript_materialization_receipt,
        validate_teachobs_transcript_materialization,
    )

    validation = validate_teachobs_transcript_materialization(
        manifest_path,
        expected_profile=evaluation_profile,
    )
    manifest = validation["manifest"]
    receipt = _load_object(receipt_path)
    expected_receipt = (
        build_public_teachobs_transcript_materialization_receipt(
            manifest,
            private_manifest_path=manifest_path,
        )
    )
    _expect(
        receipt == expected_receipt,
        "transcript materialization receipt is stale or not bound to the "
        "fully revalidated private file tree",
    )
    aggregate = manifest.get("aggregate")
    policy = manifest.get("policy")
    claims = manifest.get("claims")
    _expect(isinstance(aggregate, dict), "transcript aggregate is missing")
    _expect(isinstance(policy, dict), "transcript policy is missing")
    _expect(isinstance(claims, dict), "transcript claim boundary is missing")
    _expect(
        aggregate.get("lesson_count") == 29
        and aggregate.get("scene_count") == 4945,
        "transcript materialization differs from the paper profile",
    )
    _expect(
        aggregate.get("source_tier_lesson_counts")
        == {
            "platform_creator_provided_caption": 19,
            "platform_automatic_caption": 4,
            "audited_asr_fallback": 6,
        },
        "transcript source-tier lesson coverage is not 23 captions plus 6 ASR",
    )
    platform_source_count = aggregate.get("platform_cue_source_item_count")
    platform_retained_count = aggregate.get("platform_cue_retained_item_count")
    _expect(
        aggregate.get("platform_cue_timeline_intersection_policy")
        == PLATFORM_CUE_TIMELINE_INTERSECTION_POLICY
        and isinstance(platform_source_count, int)
        and platform_source_count > 0
        and platform_retained_count == platform_source_count
        and aggregate.get("platform_cue_source_text_items_silently_dropped") == 0
        and aggregate.get(
            "platform_cue_text_or_labels_used_for_timing_adjustment"
        )
        is False,
        "platform-caption endpoint projection provenance is incomplete",
    )
    asr_source_count = aggregate.get("asr_source_item_count")
    asr_retained_count = aggregate.get("asr_retained_item_count")
    asr_outside_count = aggregate.get(
        "asr_outside_selected_timeline_item_count"
    )
    _expect(
        aggregate.get("asr_target_window_projection_policy")
        == ASR_TIMELINE_INTERSECTION_POLICY
        and isinstance(asr_source_count, int)
        and isinstance(asr_retained_count, int)
        and isinstance(asr_outside_count, int)
        and asr_source_count > 0
        and asr_retained_count > 0
        and asr_retained_count + asr_outside_count == asr_source_count
        and aggregate.get("asr_all_source_items_within_hash_bound_media") is True
        and aggregate.get("asr_source_text_items_silently_dropped") == 0
        and aggregate.get("asr_text_or_labels_used_for_timing_adjustment")
        is False,
        "ASR target-window projection provenance is incomplete",
    )
    _expect(
        policy.get("released_transcript_fallback_used") is False
        and policy.get("labels_read_or_used") is False
        and policy.get("platform_cue_without_positive_timeline_intersection")
        == "reject"
        and policy.get("asr_segment_wholly_outside_target_window")
        == "explicitly_exclude_and_count",
        "transcript materialization used a forbidden fallback or label",
    )
    for key in (
        "content_accuracy_established",
        "word_error_rate_established",
        "independent_human_content_audit_completed",
    ):
        _expect(claims.get(key) is False, f"unsafe transcript claim: {key}")
    return {
        "stage": "transcripts",
        "complete": True,
        "evaluation_profile": evaluation_profile,
        "lesson_count": 29,
        "scene_count": 4945,
        "platform_caption_lesson_count": 23,
        "audited_asr_fallback_lesson_count": 6,
        "platform_cue_endpoint_clamped_item_count": aggregate[
            "platform_cue_endpoint_clamped_item_count"
        ],
        "asr_outside_selected_timeline_item_count": asr_outside_count,
        "released_transcript_fallback_used": False,
        "content_accuracy_established": False,
        "word_error_rate_established": False,
    }


def check_benchmark(
    result_path: str,
    receipt_path: str,
    frozen_model_output: str,
    feature_manifest_path: str,
    transcript_materialization_manifest_path: str,
    *,
    evaluation_profile: str = FULL_23_TRAIN_7_TEST_PROFILE,
) -> dict[str, Any]:
    profile = _evaluation_profile(evaluation_profile)
    result = _load_object(result_path)
    receipt = _load_object(receipt_path)
    from teaching_skill_miner.teachobs_transcript_materialization import (
        TeachObsTranscriptMaterializationError,
        validate_teachobs_transcript_materialization,
    )

    try:
        transcript_validation = validate_teachobs_transcript_materialization(
            transcript_materialization_manifest_path,
            expected_profile=evaluation_profile,
        )
    except TeachObsTranscriptMaterializationError as exc:
        raise StageCheckError(
            f"benchmark transcript materialization failed live validation: {exc}"
        ) from exc
    transcript_manifest = transcript_validation["manifest"]
    transcript_rows = transcript_validation["rows"]
    transcript_selection = transcript_manifest.get("selection")
    transcript_aggregate = transcript_manifest.get("aggregate")
    _expect(
        isinstance(transcript_selection, dict)
        and isinstance(transcript_aggregate, dict),
        "live transcript materialization lacks selection or aggregate evidence",
    )
    train_transcript_count = transcript_selection.get(
        "selected_train_scene_count"
    )
    _expect(
        isinstance(train_transcript_count, int)
        and not isinstance(train_transcript_count, bool)
        and train_transcript_count == 3846
        and len(transcript_rows) == profile["scene_count"],
        "live transcript materialization scene order/count differs from the profile",
    )
    transcript_texts = [row["text"] for row in transcript_rows]
    feature_manifest_sha256 = _file_sha256(feature_manifest_path)
    transcript_manifest_file_sha256 = _file_sha256(
        transcript_materialization_manifest_path
    )
    expected_transcript_audit = {
        "schema": transcript_manifest["schema"],
        "profile_id": transcript_manifest["profile_id"],
        "manifest_file_sha256": transcript_manifest_file_sha256,
        "manifest_sha256": transcript_manifest["manifest_sha256"],
        "materialization_fingerprint_sha256": transcript_manifest[
            "materialization_fingerprint_sha256"
        ],
        "ordered_sample_id_sha256": transcript_manifest[
            "ordered_sample_id_sha256"
        ],
        "ordered_scene_text_sha256": transcript_manifest[
            "ordered_scene_text_sha256"
        ],
        "repository_binding_sha256": _canonical_sha256(
            transcript_manifest["repository_binding"]
        ),
        "input_hashes_sha256": _canonical_sha256(
            transcript_manifest["input_hashes"]
        ),
        "selected_train_transcript_order_sha256": _canonical_sha256(
            transcript_texts[:train_transcript_count]
        ),
        "selected_test_transcript_order_sha256": _canonical_sha256(
            transcript_texts[train_transcript_count:]
        ),
        "lesson_count": transcript_aggregate["lesson_count"],
        "scene_count": transcript_aggregate["scene_count"],
        "nonempty_scene_count": transcript_aggregate[
            "nonempty_scene_count"
        ],
        "empty_scene_count": transcript_aggregate["empty_scene_count"],
        "source_tier_lesson_counts": transcript_aggregate[
            "source_tier_lesson_counts"
        ],
        "source_tier_scene_counts": transcript_aggregate[
            "source_tier_scene_counts"
        ],
        "all_selected_scene_transcripts_materialized": True,
        "sample_order_matches_benchmark_profile": True,
        "released_transcript_fallback_used": False,
        "labels_read_or_used": False,
        "empty_scenes_filled_from_released_transcript": False,
        "released_repository_transcripts_used_for_model_input": False,
    }
    _expect(
        result.get("benchmark_kind")
        == "teachobs_official_held_out_four_arm_multimodal_benchmark",
        "benchmark kind mismatch",
    )
    arms = result.get("arms")
    _expect(isinstance(arms, dict), "benchmark arms are missing")
    _expect(tuple(arms) == ARM_ORDER, "benchmark arm order or coverage mismatch")
    audit = result.get("dataset_audit")
    _expect(isinstance(audit, dict), "benchmark dataset audit is missing")
    for key, value in (
        ("benchmark_profile", evaluation_profile),
        ("lesson_count", profile["lesson_count"]),
        ("train_lesson_count", 23),
        ("test_lesson_count", profile["test_lesson_count"]),
        ("scene_count", profile["scene_count"]),
        ("test_scene_count", profile["test_scene_count"]),
        ("code_count", 39),
    ):
        _expect(audit.get(key) == value, f"benchmark {key} mismatch")
    dataset_profile_fingerprint = audit.get("dataset_profile_fingerprint")
    _expect(
        isinstance(dataset_profile_fingerprint, str)
        and len(dataset_profile_fingerprint) == 64,
        "benchmark dataset/profile fingerprint is missing",
    )
    feature_audit = result.get("private_feature_audit")
    _expect(isinstance(feature_audit, dict), "benchmark feature audit is missing")
    _expect(
        feature_audit.get("feature_manifest_sha256")
        == feature_manifest_sha256,
        "benchmark is not bound to the current feature manifest",
    )
    benchmark_input_fingerprint = _canonical_sha256(
        {
            "schema": BENCHMARK_INPUT_SCHEMA,
            "dataset_profile_fingerprint": dataset_profile_fingerprint,
            "feature_manifest_sha256": feature_manifest_sha256,
            "transcript_materialization_manifest_file_sha256": (
                transcript_manifest_file_sha256
            ),
            "transcript_materialization_manifest_sha256": (
                transcript_manifest["manifest_sha256"]
            ),
            "transcript_materialization_fingerprint_sha256": (
                transcript_manifest["materialization_fingerprint_sha256"]
            ),
            "transcript_ordered_sample_id_sha256": transcript_manifest[
                "ordered_sample_id_sha256"
            ],
            "transcript_ordered_scene_text_sha256": transcript_manifest[
                "ordered_scene_text_sha256"
            ],
        }
    )
    expected_transcript_audit["benchmark_input_fingerprint"] = (
        benchmark_input_fingerprint
    )
    _expect(
        audit.get("benchmark_input_fingerprint")
        == benchmark_input_fingerprint,
        "benchmark dataset audit input fingerprint mismatch",
    )
    private_transcript_audit = result.get("private_transcript_audit")
    _expect(
        private_transcript_audit == expected_transcript_audit,
        "benchmark private transcript audit differs from the live materialization",
    )
    protocol = result.get("protocol")
    _expect(isinstance(protocol, dict), "benchmark protocol is missing")
    _expect(
        protocol.get("benchmark_profile") == evaluation_profile,
        "benchmark protocol profile mismatch",
    )
    _expect(
        protocol.get("same_test_scenes_used_for_all_arms") is True,
        "four arms do not share the same test scenes",
    )
    _expect(
        protocol.get("test_labels_used_for_feature_fitting_training_or_tuning")
        is False,
        "test-label leakage flag is unsafe",
    )
    _expect(
        protocol.get("s4_labels_used_for_fitting_tuning_or_metrics") is False,
        "S4 label-use boundary is unsafe",
    )
    _expect(
        protocol.get("same_materialized_transcript_scenes_used_for_all_arms")
        is True
        and protocol.get(
            "released_repository_transcripts_used_for_model_input"
        )
        is False
        and protocol.get("transcript_materialization_manifest_required")
        is True
        and protocol.get("benchmark_input_fingerprint")
        == benchmark_input_fingerprint,
        "benchmark protocol lost its materialized-transcript input binding",
    )
    bootstrap = result.get("paired_cluster_bootstrap")
    _expect(isinstance(bootstrap, dict), "paired cluster bootstrap is missing")
    _expect(
        bootstrap.get("cluster_count") == profile["bootstrap_cluster_count"],
        "benchmark cluster count differs from the evaluation profile",
    )
    evidence = result.get("evidence_scope")
    _expect(isinstance(evidence, dict), "benchmark evidence scope is missing")
    _expect(evidence.get("provisional_result") is True, "benchmark is not provisional")
    for key in (
        "confirmatory_multimodal_gain_established",
        "external_lockbox_established",
        "deployment_accuracy_established",
        "learning_effectiveness_established",
    ):
        _expect(evidence.get(key) is False, f"unsafe benchmark claim: {key}")
    _expect(
        evidence.get("audited_transcript_materialization_used") is True
        and evidence.get("released_repository_transcripts_used_for_model_input")
        is False
        and evidence.get("transcript_content_accuracy_established") is False
        and evidence.get("transcript_word_error_rate_established") is False,
        "benchmark evidence scope weakens the transcript boundary",
    )
    _expect(
        receipt.get("receipt_kind")
        == "teachobs_aggregate_four_arm_multimodal_benchmark",
        "benchmark receipt kind mismatch",
    )
    _expect(
        receipt.get("private_result_canonical_sha256")
        == _canonical_sha256(result),
        "benchmark receipt is not bound to the current private result",
    )
    receipt_dataset = receipt.get("dataset_aggregate")
    _expect(isinstance(receipt_dataset, dict), "receipt dataset aggregate is missing")
    _expect(
        receipt_dataset.get("benchmark_profile") == evaluation_profile
        and receipt_dataset.get("lesson_count") == profile["lesson_count"]
        and receipt_dataset.get("test_lesson_count")
        == profile["test_lesson_count"]
        and receipt_dataset.get("test_scene_count") == profile["test_scene_count"]
        and receipt_dataset.get("dataset_profile_fingerprint")
        == dataset_profile_fingerprint,
        "receipt dataset/profile aggregate mismatch",
    )
    _expect(
        receipt_dataset.get("benchmark_input_fingerprint")
        == benchmark_input_fingerprint,
        "receipt dataset input fingerprint mismatch",
    )
    _expect(
        receipt.get("transcript_aggregate") == expected_transcript_audit,
        "public benchmark transcript aggregate differs from live private evidence",
    )
    receipt_boundary = receipt.get("claim_boundaries")
    _expect(isinstance(receipt_boundary, dict), "receipt claim boundary is missing")
    for key in (
        "confirmatory_multimodal_gain_established",
        "external_lockbox_established",
        "deployment_accuracy_established",
        "learning_effectiveness_established",
    ):
        _expect(receipt_boundary.get(key) is False, f"unsafe receipt claim: {key}")
    export = result.get("frozen_model_export")
    _expect(isinstance(export, dict), "private frozen-model export receipt is missing")
    frozen_root_value = Path(frozen_model_output)
    try:
        frozen_root = frozen_root_value.resolve(strict=True)
    except FileNotFoundError as exc:
        raise StageCheckError("private frozen-model output is missing") from exc
    _expect(
        not frozen_root_value.is_symlink() and frozen_root.is_dir(),
        "private frozen-model output is unsafe",
    )
    recorded_output_directory = export.get("output_directory")
    _expect(
        isinstance(recorded_output_directory, str)
        and bool(recorded_output_directory)
        and "\x00" not in recorded_output_directory,
        "frozen export receipt has no historical output directory",
    )
    # output_directory records where the immutable bundle was originally
    # produced.  It is provenance, not authority: release verification may run
    # from a copied source snapshot.  The caller-selected directory below is
    # authoritative and every manifest/array byte in it is hash-verified.
    _expect(export.get("arm_count") == 4, "frozen export arm count mismatch")
    _expect(
        export.get("benchmark_profile") == evaluation_profile,
        "frozen export profile mismatch",
    )
    _expect(
        export.get("dataset_profile_fingerprint")
        == dataset_profile_fingerprint,
        "frozen export dataset/profile fingerprint mismatch",
    )
    _expect(
        export.get("benchmark_input_fingerprint")
        == benchmark_input_fingerprint
        and export.get("transcript_materialization_manifest_file_sha256")
        == transcript_manifest_file_sha256
        and export.get("transcript_materialization_fingerprint_sha256")
        == transcript_manifest["materialization_fingerprint_sha256"],
        "frozen export is not bound to the live transcript materialization",
    )
    _expect(tuple(export.get("arms", ())) == ARM_ORDER, "frozen arm order mismatch")
    _expect(export.get("pickle_used") is False, "pickle is forbidden in frozen models")
    _expect(export.get("deterministic_npz") is True, "frozen arrays are not deterministic")
    _expect(
        export.get("test_prediction_bitwise_parity_verified") is True,
        "frozen/in-memory prediction parity was not verified",
    )
    maximum_delta = export.get("test_probability_maximum_absolute_delta")
    _expect(
        isinstance(maximum_delta, (int, float))
        and not isinstance(maximum_delta, bool)
        and 0 <= float(maximum_delta) <= 1e-12,
        "frozen/in-memory probability parity exceeds tolerance",
    )
    for key in (
        "confirmatory_lockbox_result_established",
        "deployment_accuracy_established",
    ):
        _expect(export.get(key) is False, f"unsafe frozen export claim: {key}")
    _expect(
        export.get("public_test_labels_were_accessible_before_freeze") is True,
        "frozen export lost its public-test provenance boundary",
    )

    bundle_manifest = frozen_root / "bundle_manifest.json"
    _expect(
        _file_sha256(bundle_manifest) == export.get("bundle_manifest_file_sha256"),
        "frozen bundle manifest hash mismatch",
    )
    artifacts = export.get("arm_model_artifacts")
    _expect(isinstance(artifacts, dict), "frozen arm artifact receipt is missing")
    _expect(tuple(artifacts) == ARM_ORDER, "frozen arm artifact coverage mismatch")
    from teaching_skill_miner.teachobs_frozen_model import (
        TeachObsFrozenModelError,
        load_teachobs_frozen_arm,
        load_teachobs_frozen_bundle,
    )

    try:
        bundle = load_teachobs_frozen_bundle(
            frozen_root,
            expected_bundle_manifest_file_sha256=str(
                export.get("bundle_manifest_file_sha256")
            ),
            expected_benchmark_profile=evaluation_profile,
            expected_dataset_profile_fingerprint=str(
                dataset_profile_fingerprint
            ),
            expected_transcript_materialization_fingerprint=str(
                transcript_manifest["materialization_fingerprint_sha256"]
            ),
            expected_benchmark_input_fingerprint=benchmark_input_fingerprint,
        )
        _expect(
            bundle.bundle_fingerprint == export.get("bundle_fingerprint"),
            "frozen bundle fingerprint mismatch",
        )
        _expect(tuple(bundle.arms) == ARM_ORDER, "loaded frozen bundle arm mismatch")
        _expect(
            bundle.transcript_materialization_fingerprint_sha256
            == transcript_manifest["materialization_fingerprint_sha256"]
            and bundle.benchmark_input_fingerprint
            == benchmark_input_fingerprint,
            "loaded frozen bundle transcript/input fingerprint mismatch",
        )
        for arm in ARM_ORDER:
            artifact = artifacts[arm]
            _expect(isinstance(artifact, dict), f"frozen {arm} receipt is malformed")
            manifest_path = frozen_root / arm / "manifest.json"
            arrays_path = frozen_root / arm / "arrays.npz"
            recorded_manifest_path = artifact.get("model_manifest_path")
            _expect(
                isinstance(recorded_manifest_path, str)
                and tuple(Path(recorded_manifest_path).parts[-2:])
                == (arm, "manifest.json"),
                f"frozen {arm} historical manifest path mismatch",
            )
            _expect(
                _file_sha256(manifest_path)
                == artifact.get("model_manifest_file_sha256"),
                f"frozen {arm} manifest hash mismatch",
            )
            _expect(
                _file_sha256(arrays_path)
                == artifact.get("numeric_state_file_sha256"),
                f"frozen {arm} companion arrays hash mismatch",
            )
            standalone = load_teachobs_frozen_arm(
                manifest_path,
                expected_manifest_file_sha256=str(
                    artifact.get("model_manifest_file_sha256")
                ),
                expected_benchmark_profile=evaluation_profile,
                expected_dataset_profile_fingerprint=str(
                    dataset_profile_fingerprint
                ),
                expected_transcript_materialization_fingerprint=str(
                    transcript_manifest[
                        "materialization_fingerprint_sha256"
                    ]
                ),
                expected_benchmark_input_fingerprint=(
                    benchmark_input_fingerprint
                ),
            )
            _expect(standalone.arm == arm, f"standalone frozen arm mismatch: {arm}")
            standalone_transcript = standalone.training_provenance.get(
                "transcript_materialization"
            )
            _expect(
                isinstance(standalone_transcript, dict)
                and standalone_transcript.get(
                    "materialization_fingerprint_sha256"
                )
                == transcript_manifest["materialization_fingerprint_sha256"]
                and standalone.training_provenance.get(
                    "benchmark_input_fingerprint"
                )
                == benchmark_input_fingerprint,
                f"loaded frozen arm transcript/input fingerprint mismatch: {arm}",
            )
            _expect(
                standalone.arrays_file_sha256
                == artifact.get("numeric_state_file_sha256"),
                f"loaded frozen arrays binding mismatch: {arm}",
            )
    except TeachObsFrozenModelError as exc:
        raise StageCheckError(f"frozen-model integrity check failed: {exc}") from exc
    return {
        "stage": "benchmark",
        "complete": True,
        "evaluation_profile": evaluation_profile,
        "arm_count": 4,
        "test_scene_count": profile["test_scene_count"],
        "bootstrap_cluster_count": profile["bootstrap_cluster_count"],
        "frozen_bundle_integrity_verified": True,
        "frozen_prediction_parity_verified": True,
        "transcript_materialization_live_validation_verified": True,
        "benchmark_input_fingerprint": benchmark_input_fingerprint,
        "claim_scope": (
            "provisional_exploratory_public_source_aligned_six_lesson_intersection"
            if evaluation_profile == PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE
            else "provisional_exploratory_public_official_23_7_split"
        ),
    }


def check_human_template_reusable(manifest_path: str) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = _load_object(manifest_file)
    _expect(manifest.get("human_completion") is False, "human work has started")
    codebook = manifest.get("operational_codebook")
    _expect(isinstance(codebook, dict), "human codebook binding is missing")
    _expect(
        codebook.get("operational_definitions_complete") is False,
        "an external operational codebook is already bound",
    )
    _expect(
        codebook.get("annotation_execution_ready") is False,
        "annotation package is already execution-ready",
    )
    assignments = manifest.get("assignments")
    _expect(isinstance(assignments, dict), "assignment bindings are missing")
    for slot in ("A", "B"):
        record = assignments.get(slot)
        _expect(isinstance(record, dict), f"assignment {slot} binding is missing")
        relative = record.get("template_file")
        _expect(isinstance(relative, str), f"assignment {slot} path is invalid")
        current = manifest_file.parent / relative
        _expect(
            _file_sha256(current) == record.get("template_file_sha256"),
            f"assignment {slot} has changed; refusing to overwrite possible human work",
        )
    return {"stage": "human_template", "safe_to_regenerate_pending_skeleton": True}


def check_human(
    manifest_path: str,
    receipt_path: str,
    *,
    evaluation_profile: str = FULL_23_TRAIN_7_TEST_PROFILE,
) -> dict[str, Any]:
    profile = _evaluation_profile(evaluation_profile)
    reusable = check_human_template_reusable(manifest_path)
    manifest = _load_object(manifest_path)
    receipt = _load_object(receipt_path)
    _expect(
        receipt.get("artifact_kind") == "public_teachobs_double_annotation_receipt",
        "human receipt kind mismatch",
    )
    _expect(
        receipt.get("assignment_manifest_sha256") == manifest.get("manifest_sha256"),
        "human receipt is not bound to the current assignment manifest",
    )
    _expect(receipt.get("human_completion") is False, "human completion was fabricated")
    aggregate = receipt.get("aggregate")
    _expect(isinstance(aggregate, dict), "human receipt aggregate is missing")
    for key, value in (
        ("official_scene_count", 5158),
        ("selected_lesson_count", profile["lesson_count"]),
        ("assigned_scene_count", profile["scene_count"]),
        ("code_count", 39),
        ("available_private_media_reference_count", profile["lesson_count"]),
    ):
        _expect(aggregate.get(key) == value, f"human receipt {key} mismatch")
    status = receipt.get("status")
    _expect(isinstance(status, dict), "human receipt status is missing")
    for key in (
        "assignments_generated",
        "different_random_order_verified",
        "all_selected_private_media_references_available",
    ):
        _expect(status.get(key) is True, f"human skeleton is incomplete: {key}")
    for key in (
        "operational_definitions_complete",
        "annotation_execution_ready",
        "complete_binary_labels_received",
        "two_distinct_annotators_received",
        "inter_rater_reliability_computed",
        "third_party_adjudication_completed",
    ):
        _expect(status.get(key) is False, f"unexpected human status: {key}")
    return {
        **reusable,
        "evaluation_profile": evaluation_profile,
        "pending_skeleton_complete": True,
        "human_completion": False,
        "inter_rater_reliability_computed": False,
    }


def check_lockbox(
    path: str,
    study_id: str,
    *,
    evaluation_profile: str | None = None,
    frozen_model_output: str,
    analysis_code_path: str,
) -> dict[str, Any]:
    value = _load_object(path)
    _expect(
        value.get("protocol")
        == "teachobs_new_site_confirmatory_multimodal_preregistration_v2",
        "lockbox protocol mismatch",
    )
    _expect(value.get("study_id") == study_id, "lockbox study id mismatch")
    artifacts = value.get("frozen_artifacts")
    _expect(isinstance(artifacts, dict), "lockbox artifact bindings are missing")
    _expect(
        artifacts.get("all_four_arm_models_bound") is True,
        "lockbox draft does not bind all four frozen arms",
    )
    _expect(
        artifacts.get("frozen_artifact_set_complete") is True,
        "lockbox frozen artifact set is incomplete",
    )
    for key in ("system_artifact", "analysis_code_artifact"):
        binding = artifacts.get(key)
        _expect(isinstance(binding, dict), f"lockbox {key} binding is missing")
        _expect(binding.get("bound") is True, f"lockbox {key} is not bound")
    arms = artifacts.get("arms")
    _expect(isinstance(arms, dict), "lockbox arm bindings are missing")
    _expect(tuple(arms) == ARM_ORDER, "lockbox arm binding coverage mismatch")
    for arm in ARM_ORDER:
        model = arms[arm].get("model_artifact") if isinstance(arms[arm], dict) else None
        _expect(isinstance(model, dict), f"lockbox {arm} model binding is missing")
        _expect(model.get("bound") is True, f"lockbox {arm} model is not bound")
    transitive = artifacts.get("transitive_model_binding")
    _expect(isinstance(transitive, dict), "transitive frozen-model binding is missing")
    _expect(
        transitive.get("schema")
        == "teaching_skill_miner.teachobs_frozen_lockbox_artifact_set.v2",
        "lockbox threshold-aware frozen binding is missing",
    )
    _expect(transitive.get("bound") is True, "transitive frozen-model gate failed")
    _expect(
        transitive.get("companion_arrays_loaded_with_allow_pickle_false") is True,
        "lockbox did not verify all companion arrays with allow_pickle=False",
    )
    _expect(transitive.get("paths_included") is False, "lockbox draft leaks paths")
    transitive_arms = transitive.get("arms")
    _expect(
        isinstance(transitive_arms, dict) and tuple(transitive_arms) == ARM_ORDER,
        "lockbox threshold arm coverage mismatch",
    )
    for arm in ARM_ORDER:
        threshold_sha256 = transitive_arms[arm].get("model_thresholds_sha256")
        _expect(
            isinstance(threshold_sha256, str) and len(threshold_sha256) == 64,
            f"lockbox {arm} threshold array is not bound",
        )
        _expect(
            transitive_arms[arm].get("model_threshold_count") == 39,
            f"lockbox {arm} threshold count mismatch",
        )
    analysis_plan = value.get("analysis_plan")
    _expect(isinstance(analysis_plan, dict), "lockbox analysis plan is missing")
    threshold_policy = analysis_plan.get("decision_threshold_policy")
    _expect(
        isinstance(threshold_policy, dict)
        and threshold_policy.get("binding_complete") is True,
        "lockbox decision-threshold policy is incomplete",
    )
    _expect(
        threshold_policy.get("arm_model_thresholds_sha256")
        == {
            arm: transitive_arms[arm]["model_thresholds_sha256"]
            for arm in ARM_ORDER
        },
        "lockbox decision-threshold policy differs from frozen arms",
    )
    _expect(
        threshold_policy.get("frozen_artifact_set_fingerprint")
        == transitive.get("artifact_set_fingerprint"),
        "lockbox decision-threshold policy binds another artifact set",
    )
    execution = value.get("execution_evidence")
    _expect(isinstance(execution, dict), "lockbox execution evidence is missing")
    _expect(execution.get("results_computed") is False, "lockbox results were fabricated")
    governance = value.get("external_governance_gate")
    _expect(isinstance(governance, dict), "external governance gate is missing")
    _expect(
        governance.get("status") == "pending_external_registration",
        "lockbox draft must remain pending external registration",
    )
    for key in (
        "registration_signature_verified",
        "registered_before_target_outcome_access_verified",
        "independent_external_governance_verified",
    ):
        _expect(governance.get(key) is False, f"unsafe governance claim: {key}")
    status = value.get("claim_status")
    _expect(isinstance(status, dict), "lockbox claim status is missing")
    for key in (
        "preregistration_execution_ready",
        "external_lockbox_established",
        "confirmatory_multimodal_gain_established",
        "deployment_accuracy_established",
        "learner_effectiveness_established",
    ):
        _expect(status.get(key) is False, f"unsafe lockbox claim: {key}")
    frozen_root_value = Path(frozen_model_output)
    analysis_code_value = Path(analysis_code_path)
    _expect(
        isinstance(frozen_model_output, str) and frozen_model_output,
        "lockbox check requires the frozen model directory",
    )
    _expect(
        isinstance(analysis_code_path, str) and analysis_code_path,
        "lockbox check requires the frozen analysis code",
    )
    try:
        frozen_root = frozen_root_value.resolve(strict=True)
    except FileNotFoundError as exc:
        raise StageCheckError("lockbox frozen model directory is missing") from exc
    _expect(
        not frozen_root_value.is_symlink() and frozen_root.is_dir(),
        "lockbox frozen model directory is unsafe",
    )
    _expect(
        not analysis_code_value.is_symlink() and analysis_code_value.is_file(),
        "lockbox analysis code is missing or unsafe",
    )

    from teaching_skill_miner.teachobs_frozen_model import (
        TeachObsFrozenModelError,
        load_teachobs_frozen_bundle,
    )
    from teaching_skill_miner.teachobs_lockbox import (
        ARM_ORDER as LOCKBOX_ARM_ORDER,
        TeachObsLockboxError,
        verify_teachobs_lockbox_artifact_files,
    )

    arm_model_paths = {
        arm: frozen_root / arm / "manifest.json" for arm in LOCKBOX_ARM_ORDER
    }
    try:
        artifact_verification = verify_teachobs_lockbox_artifact_files(
            value,
            system_artifact_path=frozen_root / "bundle_manifest.json",
            analysis_code_path=analysis_code_value,
            arm_model_artifact_paths=arm_model_paths,
        )
        if evaluation_profile is not None:
            _evaluation_profile(evaluation_profile)
            load_teachobs_frozen_bundle(
                frozen_root,
                expected_benchmark_profile=evaluation_profile,
            )
    except (TeachObsFrozenModelError, TeachObsLockboxError) as exc:
        raise StageCheckError(f"lockbox frozen artifact binding failed: {exc}") from exc
    _expect(
        artifact_verification.get("verified") is True
        and artifact_verification.get(
            "transitive_model_and_companion_arrays_verified"
        )
        is True,
        "lockbox frozen artifact verification is incomplete",
    )
    return {
        "stage": "lockbox_draft",
        "evaluation_profile": evaluation_profile,
        "draft_valid": True,
        "frozen_artifact_set_complete": True,
        "artifact_file_bindings_verified": True,
        "transitive_companion_arrays_verified": True,
        "preregistration_execution_ready": False,
        "confirmatory_multimodal_gain_established": False,
        "deployment_accuracy_established": False,
        "learner_effectiveness_established": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    annotations = subparsers.add_parser("annotations")
    annotations.add_argument("--audit", required=True)
    annotations.add_argument("--receipt", required=True)

    captions = subparsers.add_parser("captions")
    captions.add_argument("--audit", required=True)
    captions.add_argument("--receipt", required=True)

    asr = subparsers.add_parser("asr")
    asr.add_argument("--receipt", required=True)
    asr.add_argument("--media-manifest", required=True)
    asr.add_argument("--caption-audit", required=True)
    asr.add_argument("--job-manifest")
    asr.add_argument("--import-audit")
    asr.add_argument("--coverage-matrix")
    asr.add_argument(
        "--evaluation-profile", choices=tuple(EVALUATION_PROFILES)
    )

    media = subparsers.add_parser("media")
    media.add_argument("--plan", required=True)
    media.add_argument("--manifest", required=True)
    media.add_argument("--source-override-manifest")
    media.add_argument(
        "--evaluation-profile",
        choices=tuple(EVALUATION_PROFILES),
        default=FULL_23_TRAIN_7_TEST_PROFILE,
    )

    features = subparsers.add_parser("features")
    features.add_argument("--manifest", required=True)
    features.add_argument("--media-manifest", required=True)
    features.add_argument(
        "--evaluation-profile",
        choices=tuple(EVALUATION_PROFILES),
        default=FULL_23_TRAIN_7_TEST_PROFILE,
    )

    transcripts = subparsers.add_parser("transcripts")
    transcripts.add_argument("--manifest", required=True)
    transcripts.add_argument("--receipt", required=True)
    transcripts.add_argument(
        "--evaluation-profile",
        choices=(PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,),
        default=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
    )

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--result", required=True)
    benchmark.add_argument("--receipt", required=True)
    benchmark.add_argument("--frozen-model-output", required=True)
    benchmark.add_argument("--feature-manifest", required=True)
    benchmark.add_argument(
        "--transcript-materialization-manifest",
        required=True,
    )
    benchmark.add_argument(
        "--evaluation-profile",
        choices=tuple(EVALUATION_PROFILES),
        default=FULL_23_TRAIN_7_TEST_PROFILE,
    )

    human_reusable = subparsers.add_parser("human-template-reusable")
    human_reusable.add_argument("--manifest", required=True)

    human = subparsers.add_parser("human")
    human.add_argument("--manifest", required=True)
    human.add_argument("--receipt", required=True)
    human.add_argument(
        "--evaluation-profile",
        choices=tuple(EVALUATION_PROFILES),
        default=FULL_23_TRAIN_7_TEST_PROFILE,
    )

    lockbox = subparsers.add_parser("lockbox")
    lockbox.add_argument("--draft", required=True)
    lockbox.add_argument("--study-id", required=True)
    lockbox.add_argument(
        "--evaluation-profile", choices=tuple(EVALUATION_PROFILES)
    )
    lockbox.add_argument("--frozen-model-output", required=True)
    lockbox.add_argument("--analysis-code", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.stage == "annotations":
            result = check_annotations(args.audit, args.receipt)
        elif args.stage == "captions":
            result = check_captions(args.audit, args.receipt)
        elif args.stage == "asr":
            result = check_asr(
                args.receipt,
                args.media_manifest,
                args.caption_audit,
                job_manifest_path=args.job_manifest,
                import_audit_path=args.import_audit,
                coverage_matrix_path=args.coverage_matrix,
                evaluation_profile=args.evaluation_profile,
            )
        elif args.stage == "media":
            result = check_media(
                args.plan,
                args.manifest,
                source_override_manifest=args.source_override_manifest,
                evaluation_profile=args.evaluation_profile,
            )
        elif args.stage == "features":
            result = check_features(
                args.manifest,
                args.media_manifest,
                evaluation_profile=args.evaluation_profile,
            )
        elif args.stage == "transcripts":
            result = check_transcripts(
                args.manifest,
                args.receipt,
                evaluation_profile=args.evaluation_profile,
            )
        elif args.stage == "benchmark":
            result = check_benchmark(
                args.result,
                args.receipt,
                args.frozen_model_output,
                args.feature_manifest,
                args.transcript_materialization_manifest,
                evaluation_profile=args.evaluation_profile,
            )
        elif args.stage == "human-template-reusable":
            result = check_human_template_reusable(args.manifest)
        elif args.stage == "human":
            result = check_human(
                args.manifest,
                args.receipt,
                evaluation_profile=args.evaluation_profile,
            )
        else:
            result = check_lockbox(
                args.draft,
                args.study_id,
                evaluation_profile=args.evaluation_profile,
                frozen_model_output=args.frozen_model_output,
                analysis_code_path=args.analysis_code,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"stage check failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
