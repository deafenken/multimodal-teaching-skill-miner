#!/usr/bin/env python3
"""Create machine-bound release verification receipts and final acceptance JSON.

This script deliberately has no mode that copies fields from an older acceptance
file.  Every positive field is either recomputed from the exact artifact or read
from a receipt emitted at the successful end of one of the verification runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_RE = re.compile(r"[0-9a-f]{64}")
PROJECT_RECEIPT_SCHEMA = "teaching_skill_miner.project_verification_receipt.v1"
WHEEL_RECEIPT_SCHEMA = "teaching_skill_miner.exact_wheel_verification_receipt.v1"
ACCEPTANCE_SCHEMA = "teaching_skill_miner.release_acceptance.v2"
PENDING_SCHEMA = "teaching_skill_miner.release_acceptance.pending.v1"
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_JUNIT_BYTES = 32 * 1024 * 1024
MAX_WHEEL_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_MEMBER_BYTES = 256 * 1024 * 1024
MAX_MEMBER_COUNT = 10_000
MAX_COMPRESSION_RATIO = 1_000.0
TASK_TWO_RELEASE_FILES = {
    "neural_v1_runtime_manifest": Path("data/neural_v1_runtime_manifest.json"),
    "teacher_agent_demo_input": Path("data/teacher_agent_demo_input.json"),
    "teacher_agent_evaluation_cases": Path("data/teacher_agent_evaluation_cases.json"),
    "teacher_agent_free_text_benchmark": Path(
        "data/teacher_agent_free_text_benchmark.json"
    ),
    "teacher_agent_free_text_benchmark_receipt": Path(
        "data/teacher_agent_free_text_benchmark_receipt.json"
    ),
    "teacher_agent_learning_outcome_demo": Path(
        "data/teacher_agent_learning_outcome_demo.json"
    ),
    "teacher_agent_skill_library_v1": Path("data/teacher_agent_skill_library.json"),
    "teacher_agent_skill_library_v2": Path("data/teacher_agent_skill_library_v2.json"),
    "teacher_agent_dashboard_css": Path(
        "teaching_skill_miner/web/teacher_agent_demo.css"
    ),
    "teacher_agent_dashboard_html": Path(
        "teaching_skill_miner/web/teacher_agent_demo.html"
    ),
    "teacher_agent_dashboard_js": Path(
        "teaching_skill_miner/web/teacher_agent_demo.js"
    ),
    "general_teaching_skill_schema": Path("schema/general_teaching_skill.schema.json"),
    "neural_v1_runtime_manifest_schema": Path(
        "schema/neural_v1_runtime_manifest.schema.json"
    ),
    "teacher_agent_evaluation_cases_schema": Path(
        "schema/teacher_agent_evaluation_cases.schema.json"
    ),
    "teacher_agent_evaluation_report_schema": Path(
        "schema/teacher_agent_evaluation_report.schema.json"
    ),
    "teacher_agent_free_text_benchmark_schema": Path(
        "schema/teacher_agent_free_text_benchmark.schema.json"
    ),
    "teacher_agent_free_text_benchmark_report_schema": Path(
        "schema/teacher_agent_free_text_benchmark_report.schema.json"
    ),
    "teacher_agent_free_text_benchmark_receipt_schema": Path(
        "schema/teacher_agent_free_text_benchmark_receipt.schema.json"
    ),
    "teacher_agent_learning_observation_schema": Path(
        "schema/teacher_agent_learning_observation.schema.json"
    ),
    "teacher_agent_learning_report_schema": Path(
        "schema/teacher_agent_learning_report.schema.json"
    ),
    "teacher_agent_live_session_schema": Path(
        "schema/teacher_agent_live_session.schema.json"
    ),
    "teacher_agent_session_schema": Path("schema/teacher_agent_session.schema.json"),
    "teacher_agent_skill_library_v1_schema": Path(
        "schema/teacher_agent_skill_library.schema.json"
    ),
    "teacher_agent_skill_library_v2_schema": Path(
        "schema/teacher_agent_skill_library_v2.schema.json"
    ),
}


class AcceptanceError(ValueError):
    """Raised when evidence is missing, stale, malformed, or inconsistent."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_regular_file(path: Path, *, max_bytes: int | None = None) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise AcceptanceError(f"required file is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise AcceptanceError(f"required path is not a regular file: {path}")
    if max_bytes is not None and metadata.st_size > max_bytes:
        raise AcceptanceError(f"file is too large for this receipt: {path}")
    return path.read_bytes()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise AcceptanceError(f"non-finite JSON number is forbidden: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = _require_regular_file(path, max_bytes=MAX_JSON_BYTES)
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"invalid JSON receipt: {path}") from exc
    if not isinstance(value, dict):
        raise AcceptanceError(f"JSON receipt must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise AcceptanceError(f"output parent must be an existing directory: {parent}")
    resolved_parent = parent.resolve(strict=True)
    try:
        output_metadata = path.lstat()
    except FileNotFoundError:
        output_metadata = None
    if output_metadata is not None and not stat.S_ISREG(output_metadata.st_mode):
        raise AcceptanceError(f"refusing to replace non-regular output: {path}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=resolved_parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_member_name(name: str) -> None:
    pure = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise AcceptanceError(f"unsafe wheel member path: {name!r}")


def _wheel_summary(path: Path) -> tuple[dict[str, Any], dict[str, tuple[int, str]]]:
    payload = _require_regular_file(path, max_bytes=MAX_WHEEL_BYTES)
    members: dict[str, tuple[int, str]] = {}
    metadata_payloads: list[bytes] = []
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBER_COUNT:
                raise AcceptanceError("release wheel contains too many members")
            if sum(info.file_size for info in infos) > MAX_TOTAL_MEMBER_BYTES:
                raise AcceptanceError(
                    "release wheel expands beyond the safe receipt limit"
                )
            for info in infos:
                if info.is_dir():
                    raise AcceptanceError(
                        "release wheel must not contain directory entries"
                    )
                if info.file_size > MAX_MEMBER_BYTES:
                    raise AcceptanceError(
                        f"release wheel member is too large: {info.filename}"
                    )
                if info.file_size and not info.compress_size:
                    raise AcceptanceError(
                        f"release wheel member has an invalid compressed size: {info.filename}"
                    )
                if (
                    info.compress_size
                    and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
                ):
                    raise AcceptanceError(
                        f"release wheel member compression ratio is unsafe: {info.filename}"
                    )
                _safe_member_name(info.filename)
                if info.filename in members:
                    raise AcceptanceError(f"duplicate wheel member: {info.filename}")
                member_payload = archive.read(info)
                if len(member_payload) != info.file_size:
                    raise AcceptanceError(
                        f"wheel member size mismatch: {info.filename}"
                    )
                members[info.filename] = (info.file_size, _sha256_bytes(member_payload))
                if info.filename.endswith(".dist-info/METADATA"):
                    metadata_payloads.append(member_payload)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise AcceptanceError(f"invalid release wheel: {path}") from exc
    if len(metadata_payloads) != 1:
        raise AcceptanceError(
            "release wheel must contain exactly one dist-info/METADATA"
        )
    message = BytesParser(policy=email_policy).parsebytes(metadata_payloads[0])
    metadata_headers = {
        "distribution": "Name",
        "version": "Version",
        "license_expression": "License-Expression",
        "requires_python": "Requires-Python",
    }
    required_metadata: dict[str, str | None] = {}
    for field, header in metadata_headers.items():
        values = message.get_all(header, [])
        if len(values) != 1:
            raise AcceptanceError(f"release wheel metadata must contain one {header}")
        required_metadata[field] = values[0]
    if any(
        not isinstance(value, str) or not value for value in required_metadata.values()
    ):
        raise AcceptanceError("release wheel metadata is incomplete")
    if str(required_metadata["distribution"]).lower().replace("_", "-") != (
        "teaching-skill-miner"
    ):
        raise AcceptanceError("release wheel has the wrong distribution name")
    summary = {
        "filename": path.name,
        "size_bytes": len(payload),
        "uncompressed_bytes": sum(size for size, _ in members.values()),
        "member_count": len(members),
        "sha256": _sha256_bytes(payload),
        **required_metadata,
    }
    return summary, members


def _directory_members(root: Path) -> dict[str, tuple[int, str]]:
    try:
        root_metadata = root.lstat()
    except FileNotFoundError as exc:
        raise AcceptanceError(f"public artifact directory is missing: {root}") from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise AcceptanceError(f"public artifact path is not a real directory: {root}")
    members: dict[str, tuple[int, str]] = {}
    total_size = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            if child.is_symlink():
                raise AcceptanceError(
                    f"public artifact directory contains a symlink: {child}"
                )
        for name in file_names:
            child = directory_path / name
            payload = _require_regular_file(child, max_bytes=MAX_MEMBER_BYTES)
            relative = child.relative_to(root).as_posix()
            _safe_member_name(relative)
            total_size += len(payload)
            if total_size > MAX_TOTAL_MEMBER_BYTES:
                raise AcceptanceError(
                    "public artifacts exceed the safe receipt byte limit"
                )
            members[relative] = (len(payload), _sha256_bytes(payload))
            if len(members) > MAX_MEMBER_COUNT:
                raise AcceptanceError("public artifacts contain too many files")
    return members


def _teachobs_private_inputs_exist(root: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            root
            / "artifacts/private/external_datasets/teachobs/media/media_manifest.json",
            root
            / "artifacts/private/external_datasets/teachobs/captions/caption_audit.json",
        )
    )


def _teachobs_private_receipt_binding(root: Path) -> dict[str, Any]:
    """Recompute the exact private/public files used by the conditional audit."""

    relative_paths = {
        "annotation_audit": Path(
            "artifacts/private/external_datasets/teachobs/"
            "imported_annotations/dataset_audit.json"
        ),
        "annotation_receipt": Path("artifacts/public/teachobs_annotation_receipt.json"),
        "media_manifest": Path(
            "artifacts/private/external_datasets/teachobs/media/media_manifest.json"
        ),
        "caption_audit": Path(
            "artifacts/private/external_datasets/teachobs/captions/caption_audit.json"
        ),
        "caption_receipt": Path("artifacts/public/teachobs_caption_receipt.json"),
        "asr_receipt": Path("artifacts/public/teachobs_asr_receipt.json"),
    }
    paths = {name: root / path for name, path in relative_paths.items()}
    payloads = {
        name: _require_regular_file(path, max_bytes=MAX_JSON_BYTES)
        for name, path in paths.items()
    }
    hashes = {name: _sha256_bytes(payload) for name, payload in payloads.items()}
    annotation_receipt = _load_json(paths["annotation_receipt"])
    caption_receipt = _load_json(paths["caption_receipt"])
    asr_receipt = _load_json(paths["asr_receipt"])
    if annotation_receipt.get("private_audit_sha256") != hashes["annotation_audit"]:
        raise AcceptanceError(
            "TeachObs annotation receipt is stale for the private audit"
        )
    if caption_receipt.get("private_audit_sha256") != hashes["caption_audit"]:
        raise AcceptanceError("TeachObs caption receipt is stale for the private audit")
    source_hashes = asr_receipt.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise AcceptanceError("TeachObs ASR receipt has no source-hash binding")
    for field, name in (
        ("media_manifest_file_sha256", "media_manifest"),
        ("caption_audit_file_sha256", "caption_audit"),
    ):
        if source_hashes.get(field) != hashes[name]:
            raise AcceptanceError(f"TeachObs ASR receipt is stale for {name}")

    optional = (
        (
            "job_manifest_file_sha256",
            "asr_job_manifest",
            Path("artifacts/private/external_datasets/teachobs/asr/job_manifest.json"),
        ),
        (
            "asr_import_audit_file_sha256",
            "asr_import_audit",
            Path("artifacts/private/external_datasets/teachobs/asr/import_audit.json"),
        ),
        (
            "coverage_matrix_file_sha256",
            "asr_coverage_matrix",
            Path(
                "artifacts/private/external_datasets/teachobs/asr/"
                "transcript_coverage_matrix.json"
            ),
        ),
    )
    for field, name, relative in optional:
        claimed = source_hashes.get(field)
        if claimed is None:
            continue
        if not isinstance(claimed, str) or SHA256_RE.fullmatch(claimed) is None:
            raise AcceptanceError(f"TeachObs ASR receipt has invalid {field}")
        path = root / relative
        payload = _require_regular_file(path, max_bytes=MAX_JSON_BYTES)
        actual = _sha256_bytes(payload)
        if actual != claimed:
            raise AcceptanceError(f"TeachObs ASR receipt is stale for {name}")
        relative_paths[name] = relative
        hashes[name] = actual

    benchmark_paths = {
        "teachobs_feature_manifest": Path(
            "artifacts/private/external_datasets/teachobs/media/feature_manifest.json"
        ),
        "teachobs_transcript_materialization_manifest": Path(
            "artifacts/private/external_datasets/teachobs/"
            "materialized_transcripts/manifest.json"
        ),
        "teachobs_transcript_materialization_receipt": Path(
            "artifacts/public/teachobs_transcript_materialization_receipt.json"
        ),
        "teachobs_benchmark_result": Path(
            "artifacts/private/external_datasets/teachobs/"
            "multimodal_benchmark_result.json"
        ),
        "teachobs_benchmark_receipt": Path(
            "artifacts/public/teachobs_multimodal_benchmark_receipt.json"
        ),
        "teachobs_frozen_bundle_manifest": Path(
            "artifacts/private/external_datasets/teachobs/"
            "frozen_models/bundle_manifest.json"
        ),
    }
    for arm in ("transcript_only", "transcript_audio", "transcript_visual", "full"):
        benchmark_paths[f"teachobs_frozen_{arm}_manifest"] = (
            Path("artifacts/private/external_datasets/teachobs/frozen_models")
            / arm
            / "manifest.json"
        )
        benchmark_paths[f"teachobs_frozen_{arm}_arrays"] = (
            Path("artifacts/private/external_datasets/teachobs/frozen_models")
            / arm
            / "arrays.npz"
        )
    benchmark_trigger_roles = (
        "teachobs_benchmark_result",
        "teachobs_benchmark_receipt",
        "teachobs_frozen_bundle_manifest",
    )
    if any((root / benchmark_paths[role]).exists() for role in benchmark_trigger_roles):
        for name, relative in benchmark_paths.items():
            payload = _require_regular_file(
                root / relative,
                max_bytes=(
                    MAX_MEMBER_BYTES if relative.suffix == ".npz" else MAX_JSON_BYTES
                ),
            )
            relative_paths[name] = relative
            hashes[name] = _sha256_bytes(payload)

    rows = [
        {
            "role": name,
            "sha256": hashes[name],
        }
        for name in sorted(hashes)
    ]
    return {
        "file_count": len(rows),
        "files": rows,
        "binding_sha256": _sha256_bytes(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def _teachobs_governance_binding(
    root: Path,
    *,
    human_manifest: Path | None,
    human_receipt: Path | None,
    lockbox_draft: Path | None,
) -> dict[str, Any] | None:
    """Bind the exact current-profile human skeleton and lockbox draft."""

    root = root.resolve(strict=True)
    supplied = {
        "human_manifest": human_manifest,
        "human_receipt": human_receipt,
        "lockbox_draft": lockbox_draft,
    }
    present = {role: path for role, path in supplied.items() if path is not None}
    if not present:
        return None
    if set(present) != set(supplied):
        raise AcceptanceError(
            "TeachObs governance binding requires human manifest, human receipt, "
            "and lockbox draft together"
        )
    rows: list[dict[str, Any]] = []
    expected_prefixes = {
        "human_manifest": PurePosixPath("artifacts/private"),
        "human_receipt": PurePosixPath("artifacts/public"),
        "lockbox_draft": PurePosixPath("artifacts/public"),
    }
    for role, requested in supplied.items():
        assert requested is not None
        candidate = requested if requested.is_absolute() else root / requested
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(root).as_posix()
        except (FileNotFoundError, ValueError) as exc:
            raise AcceptanceError(
                f"TeachObs governance path is missing or outside the repository: {role}"
            ) from exc
        pure = PurePosixPath(relative)
        _safe_member_name(relative)
        prefix = expected_prefixes[role]
        if pure.parts[: len(prefix.parts)] != prefix.parts:
            raise AcceptanceError(
                f"TeachObs governance path is outside its permitted artifact root: {role}"
            )
        payload = _require_regular_file(resolved, max_bytes=MAX_JSON_BYTES)
        rows.append(
            {
                "role": role,
                "path": relative,
                "sha256": _sha256_bytes(payload),
            }
        )
    rows.sort(key=lambda row: str(row["role"]))
    return {
        "file_count": len(rows),
        "files": rows,
        "binding_sha256": _sha256_bytes(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def _canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AcceptanceError("Task 2 evidence is not canonical JSON") from exc
    return _sha256_bytes(payload)


def _task_two_release_binding(root: Path) -> dict[str, Any] | None:
    """Bind and sanity-check the complete public Task 2 evidence surface.

    A fixture repository may contain no Task 2 files.  Once any Task 2 release
    resource is present, however, the complete reviewed set is required.  This
    prevents a partial dashboard, stale neural-v1 status, or detached aggregate
    benchmark receipt from receiving a positive engineering acceptance.
    """

    root = root.resolve(strict=True)
    resolved = {
        role: root / relative for role, relative in TASK_TWO_RELEASE_FILES.items()
    }
    present = {role for role, path in resolved.items() if path.exists()}
    if not present:
        return None
    missing = sorted(set(resolved) - present)
    if missing:
        raise AcceptanceError(
            "Task 2 public evidence is partial; missing roles: " + ", ".join(missing)
        )

    payloads: dict[str, bytes] = {}
    documents: dict[str, dict[str, Any]] = {}
    for role, path in resolved.items():
        payloads[role] = _require_regular_file(path, max_bytes=MAX_JSON_BYTES)
        if path.suffix == ".json":
            documents[role] = _load_json(path)

    neural = documents["neural_v1_runtime_manifest"]
    neural_gate = neural.get("materialization_gate")
    neural_boundary = neural.get("claim_boundary")
    if (
        neural.get("schema") != "teaching_skill_miner.neural_v1_runtime_manifest.v1"
        or neural.get("artifact_kind")
        != "public_runtime_link_to_model_assisted_neural_v1"
        or not isinstance(neural_gate, dict)
        or not isinstance(neural_gate.get("passed"), bool)
        or not isinstance(neural_boundary, dict)
        or neural_boundary.get("neural_v1_passed_materialization_gate")
        is not neural_gate["passed"]
        or neural_boundary.get("recognition_accuracy_established") is not False
        or neural_boundary.get("cross_course_generality_established") is not False
        or neural_boundary.get("teaching_effectiveness_established") is not False
    ):
        raise AcceptanceError(
            "neural-v1 runtime manifest overstates or omits its evidence gate"
        )

    library_v1 = documents["teacher_agent_skill_library_v1"]
    library_v2 = documents["teacher_agent_skill_library_v2"]
    library_v1_boundary = library_v1.get("claim_boundary")
    library_v2_boundary = library_v2.get("claim_boundary")
    derivation = library_v2.get("derivation")
    skills = library_v2.get("skills")
    if (
        library_v1.get("schema")
        != "teaching_skill_miner.teacher_agent_skill_library.v1"
        or not isinstance(library_v1_boundary, dict)
        or library_v1_boundary.get("real_learner_effectiveness_established")
        is not False
        or library_v2.get("schema")
        != "teaching_skill_miner.teacher_agent_skill_library.v2"
        or not isinstance(skills, list)
        or not skills
        or not isinstance(derivation, dict)
        or derivation.get("source_artifact") != "data/neural_v1_runtime_manifest.json"
        or not isinstance(library_v2_boundary, dict)
        or library_v2_boundary.get("neural_v1_materialization_gate_passed")
        is not neural_gate["passed"]
        or library_v2_boundary.get("free_text_grading_accuracy_established")
        is not False
        or library_v2_boundary.get("real_learner_effectiveness_established")
        is not False
    ):
        raise AcceptanceError(
            "Task 2 v2 Skill Library is stale or overstates neural-v1 evidence"
        )

    demo_input = documents["teacher_agent_demo_input"]
    evaluation_cases = documents["teacher_agent_evaluation_cases"]
    evaluation_boundary = evaluation_cases.get("claim_boundary")
    if (
        demo_input.get("schema") != "teaching_skill_miner.teacher_agent_demo_input.v1"
        or evaluation_cases.get("schema")
        != "teaching_skill_miner.teacher_agent_evaluation_cases.v1"
        or not isinstance(evaluation_boundary, dict)
        or evaluation_boundary.get("fixtures_are_synthetic") is not True
        or evaluation_boundary.get("real_students_involved") is not False
        or evaluation_boundary.get("free_text_scoring_accuracy_established")
        is not False
        or evaluation_boundary.get("real_learning_effect_established") is not False
    ):
        raise AcceptanceError(
            "Task 2 demo/evaluation fixtures cross their claim boundary"
        )

    benchmark = documents["teacher_agent_free_text_benchmark"]
    benchmark_boundary = benchmark.get("claim_boundary")
    benchmark_cases = benchmark.get("cases")
    if (
        benchmark.get("schema")
        != "teaching_skill_miner.teacher_agent_free_text_benchmark.v1"
        or not isinstance(benchmark_cases, list)
        or len(benchmark_cases) < 24
        or not isinstance(benchmark_boundary, dict)
        or benchmark_boundary.get("source_type")
        != "author_constructed_not_expert_validated"
        or benchmark_boundary.get("expert_validated") is not False
        or benchmark_boundary.get("real_students_involved") is not False
        or benchmark_boundary.get("free_text_diagnostic_accuracy_established")
        is not False
        or benchmark_boundary.get("skill_routing_quality_established") is not False
        or benchmark_boundary.get("real_learning_effect_established") is not False
    ):
        raise AcceptanceError(
            "Task 2 free-text benchmark overstates development evidence"
        )

    receipt = documents["teacher_agent_free_text_benchmark_receipt"]
    receipt_source = receipt.get("source_report")
    receipt_inputs = receipt.get("input_fingerprints")
    receipt_config = receipt.get("configuration")
    receipt_privacy = receipt.get("privacy")
    receipt_boundary = receipt.get("claim_boundary")
    if (
        receipt.get("schema")
        != "teaching_skill_miner.teacher_agent_free_text_benchmark_receipt.v1"
        or receipt.get("run_status") != "completed"
        or not isinstance(receipt_source, dict)
        or not isinstance(receipt_inputs, dict)
        or receipt_inputs.get("benchmark_sha256") != _canonical_json_sha256(benchmark)
        or receipt_inputs.get("skill_library_sha256")
        != _canonical_json_sha256(library_v2)
        or not isinstance(receipt_config, dict)
        or receipt_config.get("provider") != "deepseek"
        or receipt_config.get("model") != "deepseek-v4-flash"
        or receipt_config.get("case_count") != len(benchmark_cases)
        or not isinstance(receipt_privacy, dict)
        or receipt_privacy.get("raw_case_text_included") is not False
        or receipt_privacy.get("provider_response_body_included") is not False
        or receipt_privacy.get("api_key_included") is not False
        or receipt_privacy.get("only_aggregate_metrics_and_hashes") is not True
        or not isinstance(receipt_boundary, dict)
        or receipt_boundary.get("source_type")
        != "author_constructed_not_expert_validated"
        or receipt_boundary.get("expert_validated") is not False
        or receipt_boundary.get("real_students_involved") is not False
        or receipt_boundary.get("held_out_after_prompt_development") is not False
        or receipt_boundary.get("free_text_diagnostic_accuracy_established")
        is not False
        or receipt_boundary.get("skill_routing_quality_established") is not False
        or receipt_boundary.get("deployment_accuracy_established") is not False
        or receipt_boundary.get("real_learning_effect_established") is not False
    ):
        raise AcceptanceError(
            "Task 2 free-text benchmark receipt is stale or overclaimed"
        )

    report_filename = receipt_source.get("filename")
    if (
        not isinstance(report_filename, str)
        or PurePosixPath(report_filename).name != report_filename
        or not report_filename.endswith(".json")
        or not isinstance(receipt_source.get("content_sha256"), str)
        or SHA256_RE.fullmatch(receipt_source["content_sha256"]) is None
        or not isinstance(receipt_source.get("run_fingerprint"), str)
        or SHA256_RE.fullmatch(receipt_source["run_fingerprint"]) is None
    ):
        raise AcceptanceError(
            "Task 2 benchmark receipt has an unsafe source-report pointer"
        )

    learning = documents["teacher_agent_learning_outcome_demo"]
    if (
        learning.get("schema")
        != "teaching_skill_miner.teacher_agent_learning_observation.v1"
        or learning.get("provenance") != "author_constructed_demo_not_real"
    ):
        raise AcceptanceError(
            "Task 2 learning outcome demo is not explicitly synthetic"
        )

    for role, document in documents.items():
        if role.endswith("_schema") and document.get("$schema") != (
            "https://json-schema.org/draft/2020-12/schema"
        ):
            raise AcceptanceError(f"Task 2 public schema is malformed: {role}")

    rows = [
        {
            "role": role,
            "path": TASK_TWO_RELEASE_FILES[role].as_posix(),
            "sha256": _sha256_bytes(payloads[role]),
        }
        for role in sorted(TASK_TWO_RELEASE_FILES)
    ]
    return {
        "file_count": len(rows),
        "files": rows,
        "binding_sha256": _sha256_bytes(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "evidence_summary": {
            "neural_v1_materialization_gate_passed": neural_gate["passed"],
            "runtime_skill_count": len(skills),
            "free_text_benchmark_case_count": len(benchmark_cases),
            "free_text_benchmark_provider": receipt_config["provider"],
            "free_text_benchmark_model": receipt_config["model"],
            "free_text_online_development_run_completed": True,
            "learning_outcome_provenance": learning["provenance"],
            "frontend_resource_count": 3,
        },
        "claim_boundaries": {
            "benchmark_expert_validated": False,
            "real_students_in_benchmark": False,
            "free_text_diagnostic_accuracy_established": False,
            "skill_routing_quality_established": False,
            "deployment_accuracy_established": False,
            "real_learner_effectiveness_established": False,
        },
    }


def _recompute_teachobs_governance_binding(
    root: Path,
    recorded: Any,
) -> dict[str, Any] | None:
    if recorded is None:
        return None
    if not isinstance(recorded, dict):
        raise AcceptanceError("TeachObs governance binding must be an object")
    files = recorded.get("files")
    if (
        recorded.get("file_count") != 3
        or not isinstance(files, list)
        or len(files) != 3
    ):
        raise AcceptanceError("TeachObs governance binding has the wrong file set")
    paths: dict[str, Path] = {}
    for row in files:
        if (
            not isinstance(row, dict)
            or set(row) != {"role", "path", "sha256"}
            or row.get("role") in paths
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
            or SHA256_RE.fullmatch(row["sha256"]) is None
        ):
            raise AcceptanceError("TeachObs governance binding row is malformed")
        paths[str(row["role"])] = Path(row["path"])
    if set(paths) != {"human_manifest", "human_receipt", "lockbox_draft"}:
        raise AcceptanceError("TeachObs governance binding roles are incomplete")
    return _teachobs_governance_binding(
        root,
        human_manifest=paths["human_manifest"],
        human_receipt=paths["human_receipt"],
        lockbox_draft=paths["lockbox_draft"],
    )


def _members_digest(members: dict[str, tuple[int, str]]) -> str:
    rows = [
        {"path": path, "size_bytes": size, "sha256": digest}
        for path, (size, digest) in sorted(members.items())
    ]
    return _sha256_bytes(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _audit_members(receipt: dict[str, Any]) -> dict[str, tuple[int, str]]:
    rows = receipt.get("members")
    if not isinstance(rows, list):
        raise AcceptanceError("release-audit receipt has no member list")
    result: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise AcceptanceError("release-audit member must be an object")
        path = row.get("path")
        size = row.get("size_bytes")
        digest = row.get("sha256")
        if not isinstance(path, str):
            raise AcceptanceError("release-audit member path must be a string")
        _safe_member_name(path)
        if path in result:
            raise AcceptanceError(f"duplicate release-audit member: {path}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise AcceptanceError(f"invalid release-audit member size: {path}")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise AcceptanceError(f"invalid release-audit member digest: {path}")
        result[path] = (size, digest)
    return result


def _validate_release_audit(
    receipt: dict[str, Any],
    *,
    expected_kind: str,
    actual_members: dict[str, tuple[int, str]],
) -> dict[str, Any]:
    if receipt.get("audit_kind") != "public_release_privacy_audit":
        raise AcceptanceError("unexpected release-audit kind")
    if receipt.get("target_kind") != expected_kind:
        raise AcceptanceError("release-audit target kind does not match")
    if receipt.get("passed") is not True:
        raise AcceptanceError("release-audit receipt did not pass")
    if receipt.get("finding_count") != 0 or receipt.get("findings") != []:
        raise AcceptanceError("release-audit receipt contains findings")
    audited_members = _audit_members(receipt)
    if audited_members != actual_members:
        raise AcceptanceError(
            "release-audit receipt is stale or bound to another target"
        )
    total_size = sum(size for size, _ in actual_members.values())
    if receipt.get("member_count") != len(actual_members):
        raise AcceptanceError("release-audit member count does not match")
    if receipt.get("total_uncompressed_bytes") != total_size:
        raise AcceptanceError("release-audit byte count does not match")
    return {
        "passed": True,
        "finding_count": 0,
        "member_count": len(actual_members),
        "total_size_bytes": total_size,
        "members_sha256": _members_digest(actual_members),
    }


def _validate_allowlist(
    receipt: dict[str, Any], wheel_summary: dict[str, Any]
) -> dict[str, Any]:
    if receipt.get("audit_kind") != "exact_release_wheel_allowlist":
        raise AcceptanceError("unexpected wheel allowlist receipt kind")
    if receipt.get("passed") is not True:
        raise AcceptanceError("wheel allowlist receipt did not pass")
    for field in (
        "errors",
        "missing_members",
        "unexpected_members",
        "content_mismatches",
        "duplicate_members",
    ):
        if receipt.get(field) != []:
            raise AcceptanceError(f"wheel allowlist receipt contains {field}")
    if receipt.get("member_count") != wheel_summary["member_count"]:
        raise AcceptanceError("wheel allowlist member count does not match")
    if receipt.get("wheel_size_bytes") != wheel_summary["size_bytes"]:
        raise AcceptanceError("wheel allowlist byte count does not match")
    if receipt.get("wheel_sha256") != wheel_summary["sha256"]:
        raise AcceptanceError("wheel allowlist is bound to another wheel")
    return {
        "passed": True,
        "member_count": wheel_summary["member_count"],
        "wheel_sha256": wheel_summary["sha256"],
    }


def _verification_scope(root: Path) -> dict[str, Any]:
    roots = (
        root / "teaching_skill_miner",
        root / "scripts",
        root / "tests",
        root / "release",
        root / ".github" / "workflows",
        root / "data" / "demo",
        root / "data" / "transcripts",
        root / "configs",
        root / "constraints",
        root / "docker",
        root / "docs",
        root / "schema",
    )
    suffixes = {
        ".Dockerfile",
        ".html",
        ".json",
        ".md",
        ".mp4",
        ".png",
        ".py",
        ".sh",
        ".svg",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    paths: list[Path] = []
    fixed_paths = (
        root / "pyproject.toml",
        root / "requirements.txt",
        root / "requirements-dev.txt",
        root / "README.md",
        root / "CHANGELOG.md",
        root / "LICENSE",
        root / "PRIVACY.md",
        root / "SECURITY.md",
        root / "THIRD_PARTY_DATA.md",
        root / "data" / "dataset_manifest.json",
        root / "data" / "evaluation_cases.json",
        root / "data" / "formal_caption_sources.json",
        root / "data" / "teacher_agent_demo_input.json",
        root / "data" / "teacher_agent_evaluation_cases.json",
        root / "data" / "teacher_agent_skill_library.json",
        root / "data" / "neural_v1_runtime_manifest.json",
        root / "data" / "teacher_agent_free_text_benchmark.json",
        root / "data" / "teacher_agent_free_text_benchmark_receipt.json",
        root / "data" / "teacher_agent_learning_outcome_demo.json",
        root / "data" / "teacher_agent_skill_library_v2.json",
    )
    for fixed in fixed_paths:
        if fixed.is_file() and not fixed.is_symlink():
            paths.append(fixed)
    for tree in roots:
        if not tree.exists():
            continue
        if tree.is_symlink() or not tree.is_dir():
            raise AcceptanceError(f"verification scope is not a real directory: {tree}")
        for path in tree.rglob("*"):
            if "__pycache__" in path.parts or path.suffix not in suffixes:
                continue
            if path.is_symlink():
                raise AcceptanceError(f"verification scope contains a symlink: {path}")
            if path.is_file():
                paths.append(path)
    rows = []
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        payload = _require_regular_file(path)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    if not rows:
        raise AcceptanceError("verification source scope is empty")
    digest = _sha256_bytes(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return {"file_count": len(rows), "sha256": digest}


def _pytest_summary(path: Path) -> dict[str, Any]:
    payload = _require_regular_file(path, max_bytes=MAX_JUNIT_BYTES)
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise AcceptanceError("pytest JUnit receipt is invalid XML") from exc
    testcases = root.findall(".//testcase")
    suites = root.findall(".//testsuite")
    if root.tag == "testsuite":
        suites.insert(0, root)
    leaf_suites = [suite for suite in suites if not suite.findall("./testsuite")]
    if not leaf_suites or not testcases:
        raise AcceptanceError("pytest JUnit receipt contains no tests")

    def total_attribute(name: str) -> int:
        total = 0
        for suite in leaf_suites:
            raw = suite.get(name, "0")
            try:
                value = int(raw)
            except ValueError as exc:
                raise AcceptanceError(f"invalid pytest JUnit {name} count") from exc
            if value < 0:
                raise AcceptanceError(f"negative pytest JUnit {name} count")
            total += value
        return total

    declared_total = total_attribute("tests")
    failures = total_attribute("failures")
    errors = total_attribute("errors")
    skipped = total_attribute("skipped")
    if failures or errors:
        raise AcceptanceError("pytest JUnit receipt contains failures or errors")
    if declared_total < len(testcases) or declared_total < failures + errors + skipped:
        raise AcceptanceError("pytest JUnit counts are inconsistent")
    subtests = declared_total - len(testcases)
    return {
        "passed": True,
        "testcase_count": len(testcases),
        "subtest_count": subtests,
        "total_count": declared_total,
        "skipped_count": skipped,
        "passed_count": declared_total - failures - errors - skipped,
    }


def _tool_version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    output = (completed.stdout or completed.stderr).splitlines()
    return output[0].strip() if output else None


def _validate_exact_receipt(
    receipt: dict[str, Any],
    wheel_summary: dict[str, Any],
    scope: dict[str, Any],
) -> None:
    if receipt.get("schema_version") != WHEEL_RECEIPT_SCHEMA:
        raise AcceptanceError("unexpected exact-wheel receipt schema")
    if receipt.get("status") != "passed":
        raise AcceptanceError("exact-wheel verification did not pass")
    if receipt.get("wheel") != wheel_summary:
        raise AcceptanceError("exact-wheel receipt is bound to another wheel")
    if receipt.get("verification_scope") != scope:
        raise AcceptanceError(
            "exact-wheel receipt is stale for the current verification code"
        )
    checks = receipt.get("checks")
    required_checks = {
        "source_release_audit_passed",
        "exact_allowlist_passed",
        "isolated_core_install_passed",
        "version_and_metadata_passed",
        "bundled_governance_passed",
        "entrypoint_smoke_passed",
        "asr_hash_entrypoint_passed",
        "captioned_video_pipeline_passed",
        "recognition_install_and_entrypoints_passed",
        "task_two_entrypoints_and_claim_boundaries_passed",
    }
    if not isinstance(checks, dict) or not required_checks.issubset(checks):
        raise AcceptanceError("exact-wheel receipt is missing required checks")
    if any(value is not True for value in checks.values()):
        raise AcceptanceError("exact-wheel receipt contains a non-passing check")
    release_audit = receipt.get("release_audit")
    if not isinstance(release_audit, dict) or release_audit.get("passed") is not True:
        raise AcceptanceError("exact-wheel receipt has no passing release audit")
    if release_audit.get("finding_count") != 0:
        raise AcceptanceError("exact-wheel receipt release audit contains findings")
    if release_audit.get("member_count") != wheel_summary["member_count"]:
        raise AcceptanceError("exact-wheel receipt release-audit member count differs")
    if release_audit.get("total_size_bytes") != wheel_summary["uncompressed_bytes"]:
        raise AcceptanceError("exact-wheel receipt release-audit byte count differs")
    member_digest = release_audit.get("members_sha256")
    if not isinstance(member_digest, str) or SHA256_RE.fullmatch(member_digest) is None:
        raise AcceptanceError("exact-wheel receipt has no valid member-set digest")
    allowlist = receipt.get("allowlist")
    if not isinstance(allowlist, dict) or allowlist.get("passed") is not True:
        raise AcceptanceError("exact-wheel receipt has no passing allowlist")
    if allowlist.get("member_count") != wheel_summary["member_count"]:
        raise AcceptanceError("exact-wheel receipt allowlist member count differs")
    if allowlist.get("wheel_sha256") != wheel_summary["sha256"]:
        raise AcceptanceError("exact-wheel receipt allowlist is bound to another wheel")


def record_wheel(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve()
    wheel_summary, wheel_members = _wheel_summary(args.wheel.resolve())
    release_audit = _validate_release_audit(
        _load_json(args.release_audit),
        expected_kind="archive",
        actual_members=wheel_members,
    )
    allowlist = _validate_allowlist(_load_json(args.allowlist_audit), wheel_summary)
    result = {
        "schema_version": WHEEL_RECEIPT_SCHEMA,
        "status": "passed",
        "runner": "scripts/verify_release_wheel.sh",
        "wheel": wheel_summary,
        "verification_scope": _verification_scope(root),
        "checks": {
            "source_release_audit_passed": release_audit["passed"],
            "exact_allowlist_passed": allowlist["passed"],
            "isolated_core_install_passed": True,
            "version_and_metadata_passed": True,
            "bundled_governance_passed": True,
            "entrypoint_smoke_passed": True,
            "asr_hash_entrypoint_passed": True,
            "captioned_video_pipeline_passed": True,
            "recognition_install_and_entrypoints_passed": True,
            "task_two_entrypoints_and_claim_boundaries_passed": True,
        },
        "release_audit": release_audit,
        "allowlist": allowlist,
    }
    _write_json_atomic(args.output, result)
    return result


def record_project(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve()
    first_summary, _ = _wheel_summary(args.first_wheel.resolve())
    second_summary, _ = _wheel_summary(args.second_wheel.resolve())
    if _require_regular_file(args.first_wheel.resolve()) != _require_regular_file(
        args.second_wheel.resolve()
    ):
        raise AcceptanceError("the two clean release builds are not byte-identical")
    if first_summary != second_summary:
        raise AcceptanceError("the two clean release build summaries do not match")
    scope = _verification_scope(root)
    exact_receipt = _load_json(args.exact_wheel_receipt)
    _validate_exact_receipt(exact_receipt, first_summary, scope)
    public_members = _directory_members(args.public_directory.resolve())
    public_audit = _validate_release_audit(
        _load_json(args.public_release_audit),
        expected_kind="directory",
        actual_members=public_members,
    )
    git_available = (root / ".git").exists()
    if git_available is not args.tracked_file_privacy_scan_run:
        raise AcceptanceError(
            "tracked-file privacy scan state does not match Git metadata availability"
        )
    formal_manifest_exists = (
        root / "artifacts/private/formal_captions/dataset_manifest.json"
    ).is_file()
    if formal_manifest_exists != args.formal_caption_audit_run:
        raise AcceptanceError(
            "formal-caption audit state does not match the current checkout"
        )
    teachobs_private_inputs_exist = _teachobs_private_inputs_exist(root)
    if teachobs_private_inputs_exist != args.teachobs_private_receipt_audit_run:
        raise AcceptanceError(
            "TeachObs private receipt-audit state does not match the current checkout"
        )
    teachobs_private_receipt_binding = (
        _teachobs_private_receipt_binding(root)
        if args.teachobs_private_receipt_audit_run
        else None
    )
    teachobs_governance_binding = _teachobs_governance_binding(
        root,
        human_manifest=getattr(args, "teachobs_human_manifest", None),
        human_receipt=getattr(args, "teachobs_human_receipt", None),
        lockbox_draft=getattr(args, "teachobs_lockbox_draft", None),
    )
    task_two_evidence_binding = _task_two_release_binding(root)
    if (
        teachobs_private_inputs_exist
        and teachobs_governance_binding is None
        and (
            root / "artifacts/private/external_datasets/teachobs/"
            "materialized_transcripts/manifest.json"
        ).is_file()
    ):
        raise AcceptanceError(
            "completed TeachObs materialization requires a bound human skeleton "
            "and lockbox draft in the project receipt"
        )
    result = {
        "schema_version": PROJECT_RECEIPT_SCHEMA,
        "status": "passed",
        "runner": "scripts/verify_project.sh",
        "wheel": first_summary,
        "verification_scope": scope,
        "pytest": _pytest_summary(args.junit_xml),
        "checks": {
            "ruff_passed": True,
            "compileall_passed": True,
            "shell_syntax_passed": True,
            "source_entrypoint_smoke_passed": True,
            "pip_check_passed": True,
            "doctor_passed": True,
            "demo_passed": True,
            "delivery_verification_passed": True,
            "public_release_audit_passed": public_audit["passed"],
            "byte_identical_double_build_passed": True,
            "exact_release_wheel_verification_passed": True,
            "generated_skill_schema_validation_passed": True,
            "task_two_entrypoints_and_claim_boundaries_passed": True,
        },
        "conditional_checks": {
            "formal_caption_private_audit_run": args.formal_caption_audit_run,
            "formal_caption_private_audit_passed": (
                True if args.formal_caption_audit_run else None
            ),
            "teachobs_private_receipt_audit_run": (
                args.teachobs_private_receipt_audit_run
            ),
            "teachobs_private_receipt_audit_passed": (
                True if args.teachobs_private_receipt_audit_run else None
            ),
            "teachobs_private_receipt_binding": teachobs_private_receipt_binding,
            "teachobs_governance_artifacts_checked": (
                teachobs_governance_binding is not None
            ),
            "teachobs_governance_binding": teachobs_governance_binding,
            "tracked_file_privacy_scan_run": args.tracked_file_privacy_scan_run,
            "tracked_file_privacy_scan_passed": (
                True if args.tracked_file_privacy_scan_run else None
            ),
            "task_two_public_evidence_checked": (task_two_evidence_binding is not None),
            "task_two_public_evidence_binding": task_two_evidence_binding,
        },
        "public_release_audit": public_audit,
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "ruff": _tool_version([sys.executable, "-m", "ruff", "--version"]),
            "ffmpeg": _tool_version(["ffmpeg", "-version"]),
            "ffprobe": _tool_version(["ffprobe", "-version"]),
            "tesseract": _tool_version(["tesseract", "--version"]),
        },
        "repository_state": {
            "git_checkout_available": git_available,
            "tracked_file_privacy_scan_run": args.tracked_file_privacy_scan_run,
            "remote_github_actions_run_verified": False,
        },
    }
    _write_json_atomic(args.output, result)
    return result


def _relative_path(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise AcceptanceError(f"{label} must be inside the repository root") from exc


def write_pending(args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(args.release_version, str) or not args.release_version.strip():
        raise AcceptanceError("release version is required for a pending receipt")
    result = {
        "schema_version": PENDING_SCHEMA,
        "release_version": args.release_version.strip(),
        "overall_status": "stale_not_accepted",
        "engineering_release_accepted": False,
        "prior_acceptance_values_valid_for_current_source": False,
        "reason": (
            "Release acceptance regeneration is in progress or the last attempt "
            "did not complete successfully."
        ),
        "regenerate_with": "sh scripts/build_release_acceptance.sh",
        "research_claim_boundaries": {
            "independent_human_review_complete": False,
            "confirmatory_multimodal_gain_established": False,
            "prospective_deployment_accuracy_established": False,
            "real_learner_effectiveness_established": False,
            "stable_cross_session_accuracy_0_9_established": False,
            "teacher_agent_free_text_diagnostic_accuracy_established": False,
            "teacher_agent_skill_routing_quality_established": False,
            "teacher_agent_deployment_accuracy_established": False,
        },
    }
    _write_json_atomic(args.output, result)
    return result


def build_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve()
    scope = _verification_scope(root)
    wheel_path = args.wheel.resolve()
    public_path = args.public_directory.resolve()
    wheel_summary, wheel_members = _wheel_summary(wheel_path)
    public_members = _directory_members(public_path)
    project_receipt = _load_json(args.project_verification_receipt)
    exact_receipt = _load_json(args.exact_wheel_receipt)
    _validate_exact_receipt(exact_receipt, wheel_summary, scope)
    if project_receipt.get("schema_version") != PROJECT_RECEIPT_SCHEMA:
        raise AcceptanceError("unexpected project-verification receipt schema")
    if project_receipt.get("status") != "passed":
        raise AcceptanceError("project verification did not pass")
    if project_receipt.get("wheel") != wheel_summary:
        raise AcceptanceError("project-verification receipt is bound to another wheel")
    if project_receipt.get("verification_scope") != scope:
        raise AcceptanceError(
            "project-verification receipt is stale for current verification code"
        )
    pytest_receipt = project_receipt.get("pytest")
    if not isinstance(pytest_receipt, dict) or pytest_receipt.get("passed") is not True:
        raise AcceptanceError(
            "project receipt does not contain a passing pytest result"
        )
    pytest_count_fields = (
        "testcase_count",
        "subtest_count",
        "total_count",
        "skipped_count",
        "passed_count",
    )
    if any(
        isinstance(pytest_receipt.get(field), bool)
        or not isinstance(pytest_receipt.get(field), int)
        or pytest_receipt[field] < 0
        for field in pytest_count_fields
    ):
        raise AcceptanceError("project receipt contains invalid pytest counts")
    if (
        pytest_receipt["total_count"]
        != pytest_receipt["testcase_count"] + pytest_receipt["subtest_count"]
        or pytest_receipt["passed_count"] + pytest_receipt["skipped_count"]
        != pytest_receipt["total_count"]
    ):
        raise AcceptanceError("project receipt pytest counts are inconsistent")
    checks = project_receipt.get("checks")
    required_project_checks = {
        "ruff_passed",
        "compileall_passed",
        "shell_syntax_passed",
        "source_entrypoint_smoke_passed",
        "pip_check_passed",
        "doctor_passed",
        "demo_passed",
        "delivery_verification_passed",
        "public_release_audit_passed",
        "byte_identical_double_build_passed",
        "exact_release_wheel_verification_passed",
        "generated_skill_schema_validation_passed",
        "task_two_entrypoints_and_claim_boundaries_passed",
    }
    if (
        not isinstance(checks, dict)
        or not required_project_checks.issubset(checks)
        or any(value is not True for value in checks.values())
    ):
        raise AcceptanceError("project receipt contains a non-passing check")
    wheel_audit = _validate_release_audit(
        _load_json(args.wheel_release_audit),
        expected_kind="archive",
        actual_members=wheel_members,
    )
    public_audit = _validate_release_audit(
        _load_json(args.public_release_audit),
        expected_kind="directory",
        actual_members=public_members,
    )
    project_public = project_receipt.get("public_release_audit")
    if not isinstance(project_public, dict) or project_public.get("passed") is not True:
        raise AcceptanceError("project receipt has no passing public release audit")
    if (
        exact_receipt["release_audit"]["members_sha256"]
        != wheel_audit["members_sha256"]
    ):
        raise AcceptanceError("exact-wheel receipt member set differs from final wheel")
    environment = project_receipt.get("environment")
    repository_state = project_receipt.get("repository_state")
    conditional_checks = project_receipt.get("conditional_checks")
    if not isinstance(environment, dict):
        raise AcceptanceError("project receipt has no environment record")
    if not isinstance(repository_state, dict):
        raise AcceptanceError("project receipt has no repository-state record")
    if not isinstance(conditional_checks, dict):
        raise AcceptanceError("project receipt has no conditional-check record")
    for field in (
        "git_checkout_available",
        "tracked_file_privacy_scan_run",
        "remote_github_actions_run_verified",
    ):
        if not isinstance(repository_state.get(field), bool):
            raise AcceptanceError(f"project repository state is missing {field}")
    for ran_field, passed_field in (
        ("formal_caption_private_audit_run", "formal_caption_private_audit_passed"),
        (
            "teachobs_private_receipt_audit_run",
            "teachobs_private_receipt_audit_passed",
        ),
        ("tracked_file_privacy_scan_run", "tracked_file_privacy_scan_passed"),
    ):
        ran = conditional_checks.get(ran_field)
        passed = conditional_checks.get(passed_field)
        if not isinstance(ran, bool) or passed is not (True if ran else None):
            raise AcceptanceError(
                f"project conditional check is inconsistent: {ran_field}"
            )
    teachobs_audit_ran = conditional_checks["teachobs_private_receipt_audit_run"]
    if _teachobs_private_inputs_exist(root) is not teachobs_audit_ran:
        raise AcceptanceError(
            "TeachObs private receipt-audit availability changed after project verification"
        )
    current_teachobs_binding = (
        _teachobs_private_receipt_binding(root) if teachobs_audit_ran else None
    )
    if (
        conditional_checks.get("teachobs_private_receipt_binding")
        != current_teachobs_binding
    ):
        raise AcceptanceError(
            "TeachObs private receipt binding changed after project verification"
        )
    governance_checked = conditional_checks.get("teachobs_governance_artifacts_checked")
    recorded_governance_binding = conditional_checks.get("teachobs_governance_binding")
    if not isinstance(governance_checked, bool):
        raise AcceptanceError("project receipt has no TeachObs governance-check state")
    if governance_checked is not (recorded_governance_binding is not None):
        raise AcceptanceError(
            "TeachObs governance-check state and binding are inconsistent"
        )
    current_governance_binding = _recompute_teachobs_governance_binding(
        root, recorded_governance_binding
    )
    if current_governance_binding != recorded_governance_binding:
        raise AcceptanceError(
            "TeachObs governance binding changed after project verification"
        )
    recorded_task_two_binding = conditional_checks.get(
        "task_two_public_evidence_binding"
    )
    task_two_checked = conditional_checks.get("task_two_public_evidence_checked")
    if not isinstance(task_two_checked, bool):
        raise AcceptanceError("project receipt has no Task 2 evidence-check state")
    if task_two_checked is not (recorded_task_two_binding is not None):
        raise AcceptanceError(
            "Task 2 evidence-check state and binding are inconsistent"
        )
    current_task_two_binding = _task_two_release_binding(root)
    if current_task_two_binding != recorded_task_two_binding:
        raise AcceptanceError(
            "Task 2 public evidence binding changed after project verification"
        )
    if (
        teachobs_audit_ran
        and (
            root / "artifacts/private/external_datasets/teachobs/"
            "materialized_transcripts/manifest.json"
        ).is_file()
        and not governance_checked
    ):
        raise AcceptanceError(
            "completed TeachObs materialization lacks bound governance artifacts"
        )
    if (
        repository_state["tracked_file_privacy_scan_run"]
        is not conditional_checks["tracked_file_privacy_scan_run"]
    ):
        raise AcceptanceError("tracked-file scan state is inconsistent across receipts")
    if repository_state["remote_github_actions_run_verified"] is not False:
        raise AcceptanceError(
            "this local runner cannot establish completed remote CI state"
        )
    result = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "release_version": wheel_summary["version"],
        "overall_status": "engineering_ready_external_validation_pending",
        "generation_policy": {
            "method": "fresh_receipts_and_exact_artifact_recomputation",
            "copied_from_previous_acceptance": False,
            "timestamp_omitted_for_reproducibility": True,
        },
        "validation": {
            "pytest": pytest_receipt,
            **checks,
            "exact_release_wheel_verification_passed": True,
            "wheel_release_audit_passed": wheel_audit["passed"],
            "public_release_audit_passed": public_audit["passed"],
        },
        "conditional_validation": conditional_checks,
        "release_artifact": {
            "artifact": _relative_path(wheel_path, root, "release wheel"),
            **{key: value for key, value in wheel_summary.items() if key != "filename"},
            "release_audit_finding_count": 0,
            "members_sha256": wheel_audit["members_sha256"],
        },
        "public_artifacts": {
            "directory": _relative_path(public_path, root, "public artifact directory"),
            "member_count": public_audit["member_count"],
            "total_size_bytes": public_audit["total_size_bytes"],
            "release_audit_finding_count": 0,
            "members_sha256": public_audit["members_sha256"],
        },
        "environment": environment,
        "verification_binding": {
            "verification_scope": scope,
            "project_verification_receipt_sha256": _sha256_bytes(
                _require_regular_file(args.project_verification_receipt)
            ),
            "exact_wheel_verification_receipt_sha256": _sha256_bytes(
                _require_regular_file(args.exact_wheel_receipt)
            ),
        },
        "task_two_evidence": current_task_two_binding,
        "research_claim_boundaries": {
            "independent_human_review_complete": False,
            "confirmatory_multimodal_gain_established": False,
            "prospective_deployment_accuracy_established": False,
            "real_learner_effectiveness_established": False,
            "stable_cross_session_accuracy_0_9_established": False,
            "teacher_agent_free_text_diagnostic_accuracy_established": False,
            "teacher_agent_skill_routing_quality_established": False,
            "teacher_agent_deployment_accuracy_established": False,
            "note": (
                "Release engineering checks do not establish annotation validity, "
                "confirmatory multimodal gain, deployment accuracy, or learner effects."
            ),
        },
        "unverified_external_state": repository_state,
    }
    _write_json_atomic(args.output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pending = subparsers.add_parser(
        "pending", help="invalidate an old acceptance before starting a new run"
    )
    pending.add_argument("--release-version", required=True)
    pending.add_argument("--output", type=Path, required=True)
    pending.set_defaults(handler=write_pending)

    wheel = subparsers.add_parser(
        "record-wheel", help="record a successfully completed exact-wheel verification"
    )
    wheel.add_argument("--wheel", type=Path, required=True)
    wheel.add_argument("--release-audit", type=Path, required=True)
    wheel.add_argument("--allowlist-audit", type=Path, required=True)
    wheel.add_argument("--repository-root", type=Path, default=Path.cwd())
    wheel.add_argument("--output", type=Path, required=True)
    wheel.set_defaults(handler=record_wheel)

    project = subparsers.add_parser(
        "record-project", help="record a successfully completed project verification"
    )
    project.add_argument("--junit-xml", type=Path, required=True)
    project.add_argument("--first-wheel", type=Path, required=True)
    project.add_argument("--second-wheel", type=Path, required=True)
    project.add_argument("--exact-wheel-receipt", type=Path, required=True)
    project.add_argument("--public-directory", type=Path, required=True)
    project.add_argument("--public-release-audit", type=Path, required=True)
    project.add_argument("--repository-root", type=Path, default=Path.cwd())
    project.add_argument("--formal-caption-audit-run", action="store_true")
    project.add_argument("--teachobs-private-receipt-audit-run", action="store_true")
    project.add_argument("--teachobs-human-manifest", type=Path)
    project.add_argument("--teachobs-human-receipt", type=Path)
    project.add_argument("--teachobs-lockbox-draft", type=Path)
    project.add_argument("--tracked-file-privacy-scan-run", action="store_true")
    project.add_argument("--output", type=Path, required=True)
    project.set_defaults(handler=record_project)

    acceptance = subparsers.add_parser(
        "acceptance", help="build final acceptance from fresh bound receipts"
    )
    acceptance.add_argument("--wheel", type=Path, required=True)
    acceptance.add_argument("--project-verification-receipt", type=Path, required=True)
    acceptance.add_argument("--exact-wheel-receipt", type=Path, required=True)
    acceptance.add_argument("--wheel-release-audit", type=Path, required=True)
    acceptance.add_argument("--public-directory", type=Path, required=True)
    acceptance.add_argument("--public-release-audit", type=Path, required=True)
    acceptance.add_argument("--repository-root", type=Path, default=Path.cwd())
    acceptance.add_argument("--output", type=Path, required=True)
    acceptance.set_defaults(handler=build_acceptance)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (AcceptanceError, OSError) as exc:
        print(f"release acceptance failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
