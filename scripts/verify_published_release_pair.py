#!/usr/bin/env python3
"""Verify that a published acceptance receipt names this exact wheel and scope."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import sys
from typing import Any
import zipfile


REPOSITORY_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_CODE_ROOT))

from scripts.generate_release_acceptance import (  # noqa: E402
    AcceptanceError,
    PROJECT_RECEIPT_SCHEMA,
    WHEEL_RECEIPT_SCHEMA,
    _recompute_teachobs_governance_binding,
    _task_two_release_binding,
    _teachobs_private_inputs_exist,
    _teachobs_private_receipt_binding,
    _validate_exact_receipt,
    _verification_scope,
)
from scripts.verify_wheel_allowlist import verify_wheel_allowlist  # noqa: E402
from teaching_skill_miner.release_audit import audit_release_path  # noqa: E402


SHA256_RE = re.compile(r"[0-9a-f]{64}")
SCOPE_SCHEMA = "teaching_skill_miner.verification_scope.v1"
ACCEPTANCE_SCHEMA = "teaching_skill_miner.release_acceptance.v2"
ACCEPTED_STATUS = "engineering_ready_external_validation_pending"
MAX_WHEEL_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_MEMBER_COUNT = 10_000
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_MEMBER_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1_000.0

TOP_LEVEL_KEYS = {
    "schema_version",
    "release_version",
    "overall_status",
    "generation_policy",
    "validation",
    "conditional_validation",
    "release_artifact",
    "public_artifacts",
    "environment",
    "verification_binding",
    "task_two_evidence",
    "research_claim_boundaries",
    "unverified_external_state",
}
VALIDATION_CHECKS = {
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
    "api_locked_install_audit_tests_typecheck_build_passed",
    "console_locked_install_audit_tests_typecheck_sealed_build_passed",
    "deployment_sources_bound_to_receipt",
    "wheel_release_audit_passed",
}
CONDITIONAL_KEYS = {
    "formal_caption_private_audit_run",
    "formal_caption_private_audit_passed",
    "teachobs_private_receipt_audit_run",
    "teachobs_private_receipt_audit_passed",
    "teachobs_private_receipt_binding",
    "teachobs_governance_artifacts_checked",
    "teachobs_governance_binding",
    "tracked_file_privacy_scan_run",
    "tracked_file_privacy_scan_passed",
    "task_two_public_evidence_checked",
    "task_two_public_evidence_binding",
}
RESEARCH_BOUNDARIES = {
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
}
TASK_TWO_BOUNDARIES = {
    "benchmark_expert_validated": False,
    "real_students_in_benchmark": False,
    "free_text_diagnostic_accuracy_established": False,
    "skill_routing_quality_established": False,
    "full_live_session_quality_established": False,
    "deployment_accuracy_established": False,
    "real_learner_effectiveness_established": False,
}
PROJECT_CHECKS = VALIDATION_CHECKS - {"wheel_release_audit_passed"}
PROJECT_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "runner",
    "wheel",
    "verification_scope",
    "release_source_snapshot",
    "pytest",
    "checks",
    "conditional_checks",
    "public_release_audit",
    "environment",
    "repository_state",
}
EXACT_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "runner",
    "wheel",
    "verification_scope",
    "checks",
    "release_audit",
    "allowlist",
}
EXACT_CHECKS = {
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


class PublishedPairError(ValueError):
    """Raised when the wheel, acceptance, and sealed scope are not one pair."""


def _regular_bytes(path: Path, description: str) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise PublishedPairError(f"{description} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise PublishedPairError(f"{description} is not a regular file: {path}")
    return path.read_bytes()


def _object(payload: bytes, description: str) -> dict[str, Any]:
    if len(payload) > MAX_JSON_BYTES:
        raise PublishedPairError(f"{description} exceeds the JSON verification limit")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PublishedPairError(
                    f"{description} contains a duplicate JSON key: {key}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PublishedPairError(
            f"{description} contains a non-finite JSON number: {value}"
        )

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except PublishedPairError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishedPairError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PublishedPairError(f"{description} must be a JSON object")
    return value


def _exact_keys(value: Any, expected: set[str], description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise PublishedPairError(f"{description} has an unexpected field set")
    return value


def _non_negative_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublishedPairError(f"{description} must be a non-negative integer")
    return value


def _safe_relative_path(value: Any, description: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise PublishedPairError(f"{description} must be a relative path")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PublishedPairError(f"{description} is unsafe")
    return path


def _members_digest(members: dict[str, tuple[int, str]]) -> str:
    rows = [
        {"path": path, "size_bytes": size, "sha256": digest}
        for path, (size, digest) in sorted(members.items())
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _wheel_members(path: Path) -> tuple[dict[str, Any], dict[str, tuple[int, str]]]:
    payload = _regular_bytes(path, "published wheel")
    if len(payload) > MAX_WHEEL_BYTES:
        raise PublishedPairError("published wheel exceeds the verification limit")
    members: dict[str, tuple[int, str]] = {}
    metadata_payloads: list[bytes] = []
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBER_COUNT:
                raise PublishedPairError("published wheel contains too many members")
            if sum(info.file_size for info in infos) > MAX_TOTAL_MEMBER_BYTES:
                raise PublishedPairError(
                    "published wheel expands beyond the safe limit"
                )
            for info in infos:
                if info.is_dir():
                    raise PublishedPairError(
                        "published wheel contains a directory entry"
                    )
                _safe_relative_path(info.filename, "published wheel member")
                if info.filename in members:
                    raise PublishedPairError("published wheel has a duplicate member")
                if info.file_size > MAX_MEMBER_BYTES:
                    raise PublishedPairError("published wheel has an oversized member")
                if info.file_size and not info.compress_size:
                    raise PublishedPairError(
                        "published wheel has an invalid compressed member"
                    )
                if (
                    info.compress_size
                    and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
                ):
                    raise PublishedPairError(
                        "published wheel has an unsafe compression ratio"
                    )
                member_payload = archive.read(info)
                if len(member_payload) != info.file_size:
                    raise PublishedPairError("published wheel member size is stale")
                members[info.filename] = (
                    len(member_payload),
                    hashlib.sha256(member_payload).hexdigest(),
                )
                if info.filename.endswith(".dist-info/METADATA"):
                    metadata_payloads.append(member_payload)
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PublishedPairError("published wheel is not a safe ZIP archive") from exc
    if len(metadata_payloads) != 1:
        raise PublishedPairError("published wheel must contain exactly one METADATA")
    metadata = BytesParser(policy=email_policy).parsebytes(metadata_payloads[0])
    headers = {
        "distribution": "Name",
        "version": "Version",
        "license_expression": "License-Expression",
        "requires_python": "Requires-Python",
    }
    parsed: dict[str, str] = {}
    for field, header in headers.items():
        values = metadata.get_all(header, [])
        if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
            raise PublishedPairError(f"published wheel has invalid {header} metadata")
        parsed[field] = values[0]
    return (
        {
            "filename": path.name,
            "size_bytes": len(payload),
            "uncompressed_bytes": sum(size for size, _ in members.values()),
            "member_count": len(members),
            "sha256": hashlib.sha256(payload).hexdigest(),
            **parsed,
            "members_sha256": _members_digest(members),
        },
        members,
    )


def _directory_members(root: Path) -> dict[str, tuple[int, str]]:
    try:
        metadata = root.lstat()
    except FileNotFoundError as exc:
        raise PublishedPairError(
            "published public-artifact directory is missing"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise PublishedPairError(
            "published public-artifact path is not a real directory"
        )
    members: dict[str, tuple[int, str]] = {}
    total_size = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise PublishedPairError("published public artifacts contain a symlink")
        for name in file_names:
            child = directory_path / name
            if child.is_symlink():
                raise PublishedPairError("published public artifacts contain a symlink")
            payload = _regular_bytes(child, "published public artifact")
            relative = child.relative_to(root).as_posix()
            _safe_relative_path(relative, "published public-artifact member")
            if relative in members:
                raise PublishedPairError(
                    "published public artifacts contain duplicates"
                )
            total_size += len(payload)
            if len(payload) > MAX_MEMBER_BYTES or total_size > MAX_TOTAL_MEMBER_BYTES:
                raise PublishedPairError(
                    "published public artifacts exceed safe limits"
                )
            members[relative] = (len(payload), hashlib.sha256(payload).hexdigest())
            if len(members) > MAX_MEMBER_COUNT:
                raise PublishedPairError(
                    "published public artifacts contain too many files"
                )
    return members


def _validate_pytest(value: Any) -> None:
    pytest = _exact_keys(
        value,
        {
            "passed",
            "testcase_count",
            "subtest_count",
            "total_count",
            "skipped_count",
            "passed_count",
        },
        "published pytest validation",
    )
    if pytest["passed"] is not True:
        raise PublishedPairError("published pytest validation is not passing")
    for field in (
        "testcase_count",
        "subtest_count",
        "total_count",
        "skipped_count",
        "passed_count",
    ):
        _non_negative_int(pytest[field], f"published pytest {field}")
    if (
        pytest["total_count"] != pytest["testcase_count"] + pytest["subtest_count"]
        or pytest["passed_count"] + pytest["skipped_count"] != pytest["total_count"]
    ):
        raise PublishedPairError("published pytest counts are inconsistent")


def _validate_task_two(root: Path, value: Any) -> None:
    if value is None:
        return
    evidence = _exact_keys(
        value,
        {
            "file_count",
            "files",
            "binding_sha256",
            "evidence_summary",
            "claim_boundaries",
        },
        "published Task 2 evidence",
    )
    rows = evidence["files"]
    if not isinstance(rows, list) or not rows:
        raise PublishedPairError("published Task 2 evidence has no files")
    if _non_negative_int(evidence["file_count"], "Task 2 file_count") != len(rows):
        raise PublishedPairError("published Task 2 file count is inconsistent")
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    normalized_rows: list[dict[str, str]] = []
    documents: dict[str, Any] = {}
    for row in rows:
        row = _exact_keys(row, {"role", "path", "sha256"}, "Task 2 file row")
        role = row["role"]
        digest = row["sha256"]
        path = _safe_relative_path(row["path"], "Task 2 evidence path")
        if (
            not isinstance(role, str)
            or not role
            or role in seen_roles
            or row["path"] in seen_paths
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise PublishedPairError("published Task 2 evidence row is malformed")
        seen_roles.add(role)
        seen_paths.add(row["path"])
        payload = _regular_bytes(root.joinpath(*path.parts), "Task 2 evidence file")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise PublishedPairError("published Task 2 evidence file binding is stale")
        normalized_rows.append({"role": role, "path": row["path"], "sha256": digest})
        if path.suffix == ".json":
            try:
                documents[role] = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PublishedPairError("Task 2 evidence JSON is invalid") from exc
    if normalized_rows != sorted(normalized_rows, key=lambda row: row["role"]):
        raise PublishedPairError("published Task 2 evidence rows are not canonical")
    row_digest = hashlib.sha256(
        json.dumps(normalized_rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if evidence["binding_sha256"] != row_digest:
        raise PublishedPairError("published Task 2 binding digest is stale")
    if evidence["claim_boundaries"] != TASK_TWO_BOUNDARIES:
        raise PublishedPairError("published Task 2 claim boundaries are overclaimed")

    summary = _exact_keys(
        evidence["evidence_summary"],
        {
            "neural_v1_materialization_gate_passed",
            "runtime_skill_count",
            "free_text_benchmark_case_count",
            "free_text_benchmark_provider",
            "free_text_benchmark_model",
            "free_text_online_development_run_completed",
            "multiturn_benchmark_episode_count",
            "multiturn_benchmark_learner_turn_count",
            "multiturn_benchmark_profile_replacement_count",
            "multiturn_benchmark_online_run_completed",
            "learning_outcome_provenance",
            "frontend_resource_count",
        },
        "published Task 2 evidence summary",
    )
    required_documents = {
        "neural_v1_runtime_manifest",
        "teacher_agent_skill_library_v2",
        "teacher_agent_free_text_benchmark",
        "teacher_agent_free_text_benchmark_receipt",
        "teacher_agent_multiturn_benchmark",
        "teacher_agent_learning_outcome_demo",
    }
    if not required_documents.issubset(documents):
        raise PublishedPairError(
            "published Task 2 evidence is missing source documents"
        )
    neural_gate = documents["neural_v1_runtime_manifest"].get("materialization_gate")
    skills = documents["teacher_agent_skill_library_v2"].get("skills")
    benchmark_cases = documents["teacher_agent_free_text_benchmark"].get("cases")
    benchmark_receipt = documents["teacher_agent_free_text_benchmark_receipt"]
    benchmark_config = benchmark_receipt.get("configuration")
    multiturn_episodes = documents["teacher_agent_multiturn_benchmark"].get("episodes")
    learning = documents["teacher_agent_learning_outcome_demo"]
    if (
        not isinstance(neural_gate, dict)
        or not isinstance(neural_gate.get("passed"), bool)
        or not isinstance(skills, list)
        or not isinstance(benchmark_cases, list)
        or benchmark_receipt.get("run_status") != "completed"
        or not isinstance(benchmark_config, dict)
        or not isinstance(multiturn_episodes, list)
    ):
        raise PublishedPairError("published Task 2 source evidence is malformed")
    turns = [
        turn
        for episode in multiturn_episodes
        if isinstance(episode, dict)
        for turn in episode.get("turns", [])
        if isinstance(turn, dict)
    ]
    expected_summary = {
        "neural_v1_materialization_gate_passed": neural_gate["passed"],
        "runtime_skill_count": len(skills),
        "free_text_benchmark_case_count": len(benchmark_cases),
        "free_text_benchmark_provider": benchmark_config.get("provider"),
        "free_text_benchmark_model": benchmark_config.get("model"),
        "free_text_online_development_run_completed": True,
        "multiturn_benchmark_episode_count": len(multiturn_episodes),
        "multiturn_benchmark_learner_turn_count": sum(
            turn.get("operation") == "learner_turn" for turn in turns
        ),
        "multiturn_benchmark_profile_replacement_count": sum(
            turn.get("operation") == "replace_profile" for turn in turns
        ),
        "multiturn_benchmark_online_run_completed": False,
        "learning_outcome_provenance": learning.get("provenance"),
        "frontend_resource_count": 3,
    }
    if summary != expected_summary:
        raise PublishedPairError("published Task 2 evidence summary is stale")


def _snapshot_bindings(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "snapshot_sha256",
            "source_snapshot_binding",
            "conditional_private_evidence_binding",
            "files",
        },
        "source snapshot manifest",
    )
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise PublishedPairError("source snapshot manifest has no file rows")
    source_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        row = _exact_keys(
            row,
            {"path", "size_bytes", "sha256", "source_kind", "executable"},
            "source snapshot manifest row",
        )
        path = row.get("path")
        size = row.get("size_bytes")
        digest = row.get("sha256")
        kind = row.get("source_kind")
        try:
            pure = _safe_relative_path(path, "source snapshot manifest path")
        except PublishedPairError:
            pure = None
        if (
            pure is None
            or path in seen
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            or not isinstance(row.get("executable"), bool)
            or kind
            not in {
                "conditional_private_evidence",
                "release_source_or_public_evidence",
            }
        ):
            raise PublishedPairError("source snapshot manifest has an invalid row")
        seen.add(path)
        assert pure is not None
        payload = _regular_bytes(
            root.joinpath(*pure.parts), "source snapshot manifest member"
        )
        member_mode = root.joinpath(*pure.parts).lstat().st_mode
        if (
            len(payload) != size
            or hashlib.sha256(payload).hexdigest() != digest
            or bool(member_mode & 0o111) is not row["executable"]
        ):
            raise PublishedPairError(
                "source snapshot manifest member binding is stale"
            )
        compact = {"path": path, "size_bytes": size, "sha256": digest}
        source_rows.append(compact)
        is_private = pure.parts[:2] == ("artifacts", "private")
        if (kind == "conditional_private_evidence") is not is_private:
            raise PublishedPairError(
                "source snapshot manifest misclassifies private evidence"
            )
        if kind == "conditional_private_evidence":
            private_rows.append(compact)

    if source_rows != sorted(source_rows, key=lambda row: str(row["path"])):
        raise PublishedPairError("source snapshot manifest rows are not canonical")

    def binding(selected: list[dict[str, Any]]) -> dict[str, Any]:
        payload = json.dumps(
            selected,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "file_count": len(selected),
            "total_size_bytes": sum(int(row["size_bytes"]) for row in selected),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    source_binding = binding(source_rows)
    private_binding = binding(private_rows)
    if (
        manifest.get("snapshot_sha256") != source_binding["sha256"]
        or manifest.get("source_snapshot_binding") != source_binding
        or manifest.get("conditional_private_evidence_binding") != private_binding
    ):
        raise PublishedPairError("source snapshot manifest aggregate is stale")
    return {
        "source_snapshot": source_binding,
        "conditional_private_evidence": private_binding,
    }


def _canonical_file(path: Path, expected: Path, description: str) -> Path:
    if path.is_symlink():
        raise PublishedPairError(f"{description} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        canonical = expected.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PublishedPairError(f"{description} is missing") from exc
    if resolved != canonical:
        raise PublishedPairError(f"{description} is not at its canonical path")
    return resolved


def _generator_call(description: str, function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except (AcceptanceError, OSError, ValueError) as exc:
        raise PublishedPairError(f"{description} failed: {exc}") from exc


def _validate_recomputed_release_audit(
    path: Path,
    *,
    expected_kind: str,
    expected_members: dict[str, tuple[int, str]],
    description: str,
) -> dict[str, Any]:
    report = audit_release_path(path)
    if (
        report.get("audit_kind") != "public_release_privacy_audit"
        or report.get("target_kind") != expected_kind
        or report.get("passed") is not True
        or report.get("finding_count") != 0
        or report.get("findings") != []
    ):
        raise PublishedPairError(f"{description} privacy audit did not pass")
    try:
        audited_members = _audit_members_for_verifier(report)
    except PublishedPairError as exc:
        raise PublishedPairError(f"{description} privacy audit is malformed") from exc
    if audited_members != expected_members:
        raise PublishedPairError(f"{description} privacy audit member set is stale")
    if (
        report.get("member_count") != len(expected_members)
        or report.get("total_uncompressed_bytes")
        != sum(size for size, _ in expected_members.values())
    ):
        raise PublishedPairError(f"{description} privacy audit totals are stale")
    return report


def _audit_members_for_verifier(
    report: dict[str, Any],
) -> dict[str, tuple[int, str]]:
    rows = report.get("members")
    if not isinstance(rows, list):
        raise PublishedPairError("release audit has no member rows")
    members: dict[str, tuple[int, str]] = {}
    for row in rows:
        row = _exact_keys(
            row,
            {"path", "size_bytes", "sha256"},
            "release audit member",
        )
        path = _safe_relative_path(row["path"], "release audit member path").as_posix()
        size = _non_negative_int(row["size_bytes"], "release audit member size")
        digest = row["sha256"]
        if (
            path in members
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise PublishedPairError("release audit member is malformed")
        members[path] = (size, digest)
    return members


def _validate_original_proofs(
    *,
    receipt: dict[str, Any],
    binding: dict[str, Any],
    project_path: Path,
    exact_path: Path,
    wheel_summary: dict[str, Any],
    scope: dict[str, Any],
    snapshot_binding: dict[str, Any],
    validation: dict[str, Any],
    conditional: dict[str, Any],
    public_members: dict[str, tuple[int, str]],
) -> None:
    proof_wheel_summary = {
        key: value for key, value in wheel_summary.items() if key != "members_sha256"
    }
    project_payload = _regular_bytes(project_path, "project verification proof")
    exact_payload = _regular_bytes(exact_path, "exact-wheel verification proof")
    if (
        hashlib.sha256(project_payload).hexdigest()
        != binding["project_verification_receipt_sha256"]
    ):
        raise PublishedPairError("project verification proof hash is stale")
    if (
        hashlib.sha256(exact_payload).hexdigest()
        != binding["exact_wheel_verification_receipt_sha256"]
    ):
        raise PublishedPairError("exact-wheel verification proof hash is stale")

    project = _object(project_payload, "project verification proof")
    exact = _object(exact_payload, "exact-wheel verification proof")
    _exact_keys(project, PROJECT_TOP_LEVEL_KEYS, "project verification proof")
    _exact_keys(exact, EXACT_TOP_LEVEL_KEYS, "exact-wheel verification proof")
    if (
        project.get("schema_version") != PROJECT_RECEIPT_SCHEMA
        or project.get("status") != "passed"
        or project.get("runner") != "scripts/verify_project.sh"
    ):
        raise PublishedPairError("project verification proof is not a passing proof")
    if (
        exact.get("schema_version") != WHEEL_RECEIPT_SCHEMA
        or exact.get("status") != "passed"
        or exact.get("runner") != "scripts/verify_release_wheel.sh"
    ):
        raise PublishedPairError("exact-wheel verification proof is not passing")
    try:
        _validate_exact_receipt(exact, proof_wheel_summary, scope)
    except (AcceptanceError, OSError, ValueError) as exc:
        raise PublishedPairError(f"exact-wheel verification proof failed: {exc}") from exc
    if set(exact.get("checks", {})) != EXACT_CHECKS:
        raise PublishedPairError("exact-wheel proof has an unexpected check set")
    if exact.get("release_audit") != {
        "passed": True,
        "finding_count": 0,
        "member_count": wheel_summary["member_count"],
        "total_size_bytes": wheel_summary["uncompressed_bytes"],
        "members_sha256": wheel_summary["members_sha256"],
    }:
        raise PublishedPairError("exact-wheel proof privacy-audit binding is stale")
    if exact.get("allowlist") != {
        "passed": True,
        "member_count": wheel_summary["member_count"],
        "wheel_sha256": wheel_summary["sha256"],
    }:
        raise PublishedPairError("exact-wheel proof allowlist binding is stale")
    if project.get("wheel") != proof_wheel_summary:
        raise PublishedPairError("project verification proof names another wheel")
    if project.get("verification_scope") != scope:
        raise PublishedPairError("project verification proof names another scope")
    if project.get("release_source_snapshot") != snapshot_binding:
        raise PublishedPairError("project verification proof names another snapshot")
    if project.get("pytest") != validation["pytest"]:
        raise PublishedPairError("published pytest result differs from original proof")
    if (
        not isinstance(project.get("checks"), dict)
        or set(project["checks"]) != PROJECT_CHECKS
        or project["checks"]
        != {key: validation[key] for key in PROJECT_CHECKS}
    ):
        raise PublishedPairError("published validation differs from original proof")
    if project.get("conditional_checks") != conditional:
        raise PublishedPairError(
            "published conditional validation differs from original proof"
        )
    if project.get("environment") != receipt.get("environment"):
        raise PublishedPairError("published environment differs from original proof")
    if project.get("repository_state") != receipt.get("unverified_external_state"):
        raise PublishedPairError(
            "published external-state boundary differs from original proof"
        )
    project_public = project.get("public_release_audit")
    if not isinstance(project_public, dict) or project_public != {
        "passed": True,
        "finding_count": 0,
        "member_count": len(public_members),
        "total_size_bytes": sum(size for size, _ in public_members.values()),
        "members_sha256": _members_digest(public_members),
    }:
        raise PublishedPairError("project proof public audit binding is stale")


def verify_published_pair(
    *,
    repository_root: Path,
    wheel: Path,
    acceptance: Path,
    expected_source_scope: Path,
    source_snapshot_manifest: Path,
    project_verification_receipt: Path,
    exact_wheel_verification_receipt: Path,
) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    wheel = _canonical_file(
        wheel, root / "dist" / wheel.name, "published wheel"
    )
    acceptance = _canonical_file(
        acceptance,
        root / "artifacts" / acceptance.name,
        "published acceptance",
    )
    expected_source_scope = _canonical_file(
        expected_source_scope,
        root / "artifacts/final-verification-scope.json",
        "sealed source scope",
    )
    source_snapshot_manifest = _canonical_file(
        source_snapshot_manifest,
        root / ".release-source-snapshot.json",
        "source snapshot manifest",
    )
    receipt = _object(
        _regular_bytes(acceptance, "published acceptance"),
        "published acceptance",
    )
    # A pending/stale receipt deliberately retains enough artifact and scope
    # metadata to explain what failed.  Those shared fields must never be
    # sufficient to promote it back into a positive release pair.  The
    # positive v2 contract omits ``engineering_release_accepted`` entirely;
    # that flag belongs only to the fail-closed pending schema.
    if receipt.get("schema_version") != ACCEPTANCE_SCHEMA:
        raise PublishedPairError(
            "published acceptance has an unexpected positive schema"
        )
    if receipt.get("overall_status") != ACCEPTED_STATUS:
        raise PublishedPairError(
            "published acceptance is not in the accepted engineering state"
        )
    if "engineering_release_accepted" in receipt:
        raise PublishedPairError(
            "published acceptance mixes pending and positive receipt fields"
        )
    _exact_keys(receipt, TOP_LEVEL_KEYS, "published acceptance")
    wheel_payload = _regular_bytes(wheel, "published wheel")
    wheel_summary, wheel_members = _wheel_members(wheel)

    if receipt.get("generation_policy") != {
        "method": "fresh_receipts_and_exact_artifact_recomputation",
        "copied_from_previous_acceptance": False,
        "timestamp_omitted_for_reproducibility": True,
    }:
        raise PublishedPairError("published acceptance generation policy is invalid")

    validation = _exact_keys(
        receipt.get("validation"),
        {"pytest", *VALIDATION_CHECKS},
        "published acceptance validation",
    )
    _validate_pytest(validation["pytest"])
    for check in VALIDATION_CHECKS:
        if validation[check] is not True:
            raise PublishedPairError(
                f"published acceptance validation is not passing: {check}"
            )

    conditional = _exact_keys(
        receipt.get("conditional_validation"),
        CONDITIONAL_KEYS,
        "published acceptance conditional validation",
    )
    for ran_field, passed_field in (
        ("formal_caption_private_audit_run", "formal_caption_private_audit_passed"),
        (
            "teachobs_private_receipt_audit_run",
            "teachobs_private_receipt_audit_passed",
        ),
        ("tracked_file_privacy_scan_run", "tracked_file_privacy_scan_passed"),
    ):
        ran = conditional[ran_field]
        if not isinstance(ran, bool) or conditional[passed_field] is not (
            True if ran else None
        ):
            raise PublishedPairError(
                f"published conditional validation is inconsistent: {ran_field}"
            )
    formal_manifest = root / "artifacts/private/formal_captions/dataset_manifest.json"
    if formal_manifest.is_symlink():
        raise PublishedPairError("formal-caption evidence path is unsafe")
    if conditional["formal_caption_private_audit_run"] is not formal_manifest.is_file():
        raise PublishedPairError(
            "published formal-caption audit state differs from snapshot evidence"
        )
    teachobs_inputs_exist = _generator_call(
        "TeachObs private evidence discovery", _teachobs_private_inputs_exist, root
    )
    if conditional["teachobs_private_receipt_audit_run"] is not teachobs_inputs_exist:
        raise PublishedPairError(
            "published TeachObs audit state differs from snapshot evidence"
        )
    current_private_binding = (
        _generator_call(
            "TeachObs private receipt binding", _teachobs_private_receipt_binding, root
        )
        if teachobs_inputs_exist
        else None
    )
    if conditional["teachobs_private_receipt_binding"] != current_private_binding:
        raise PublishedPairError("published TeachObs private binding is stale")
    governance_checked = conditional["teachobs_governance_artifacts_checked"]
    if not isinstance(governance_checked, bool) or governance_checked is not (
        conditional["teachobs_governance_binding"] is not None
    ):
        raise PublishedPairError(
            "published TeachObs governance validation is inconsistent"
        )
    current_governance_binding = _generator_call(
        "TeachObs governance binding",
        _recompute_teachobs_governance_binding,
        root,
        conditional["teachobs_governance_binding"],
    )
    if current_governance_binding != conditional["teachobs_governance_binding"]:
        raise PublishedPairError("published TeachObs governance binding is stale")
    task_two_checked = conditional["task_two_public_evidence_checked"]
    if (
        not isinstance(task_two_checked, bool)
        or task_two_checked
        is not (conditional["task_two_public_evidence_binding"] is not None)
        or conditional["task_two_public_evidence_binding"]
        != receipt.get("task_two_evidence")
    ):
        raise PublishedPairError("published Task 2 validation is inconsistent")
    current_task_two = _generator_call(
        "Task 2 evidence recomputation", _task_two_release_binding, root
    )
    if current_task_two != receipt.get("task_two_evidence"):
        raise PublishedPairError("published Task 2 evidence binding is stale")

    if receipt.get("research_claim_boundaries") != RESEARCH_BOUNDARIES:
        raise PublishedPairError("published research claim boundaries are overclaimed")

    external_state = _exact_keys(
        receipt.get("unverified_external_state"),
        {
            "git_checkout_available",
            "tracked_file_privacy_scan_run",
            "remote_github_actions_run_verified",
        },
        "published unverified external state",
    )
    if (
        not isinstance(external_state["git_checkout_available"], bool)
        or not isinstance(external_state["tracked_file_privacy_scan_run"], bool)
        or external_state["remote_github_actions_run_verified"] is not False
        or external_state["tracked_file_privacy_scan_run"]
        is not conditional["tracked_file_privacy_scan_run"]
    ):
        raise PublishedPairError("published external-state boundary is invalid")
    git_available = (root / ".git").exists()
    if (
        external_state["git_checkout_available"] is not git_available
        or external_state["tracked_file_privacy_scan_run"] is not git_available
    ):
        raise PublishedPairError(
            "published tracked-file privacy state differs from snapshot Git state"
        )

    environment = _exact_keys(
        receipt.get("environment"),
        {
            "python",
            "python_implementation",
            "platform",
            "ruff",
            "ffmpeg",
            "ffprobe",
            "tesseract",
        },
        "published environment",
    )
    for field in ("python", "python_implementation", "platform"):
        if not isinstance(environment[field], str) or not environment[field]:
            raise PublishedPairError(f"published environment is missing {field}")
    for field in ("ruff", "ffmpeg", "ffprobe", "tesseract"):
        if environment[field] is not None and (
            not isinstance(environment[field], str) or not environment[field]
        ):
            raise PublishedPairError(f"published environment has invalid {field}")

    _validate_task_two(root, receipt.get("task_two_evidence"))

    scope_receipt = _object(
        _regular_bytes(expected_source_scope, "sealed source scope"),
        "sealed source scope",
    )
    _exact_keys(
        scope_receipt,
        {"schema_version", "verification_scope"},
        "sealed source scope",
    )
    if scope_receipt.get("schema_version") != SCOPE_SCHEMA:
        raise PublishedPairError("sealed source scope has an unexpected schema")
    expected_scope = scope_receipt.get("verification_scope")
    if (
        not isinstance(expected_scope, dict)
        or not isinstance(expected_scope.get("file_count"), int)
        or expected_scope["file_count"] < 1
        or not isinstance(expected_scope.get("sha256"), str)
        or SHA256_RE.fullmatch(expected_scope["sha256"]) is None
    ):
        raise PublishedPairError("sealed source scope is malformed")
    recomputed_scope = _generator_call(
        "verification scope recomputation", _verification_scope, root
    )
    if recomputed_scope != expected_scope:
        raise PublishedPairError("sealed source scope is stale for snapshot root")

    artifact = _exact_keys(
        receipt.get("release_artifact"),
        {
            "artifact",
            "distribution",
            "version",
            "size_bytes",
            "uncompressed_bytes",
            "member_count",
            "sha256",
            "license_expression",
            "requires_python",
            "release_audit_finding_count",
            "members_sha256",
        },
        "published release artifact",
    )
    try:
        relative_wheel = wheel.relative_to(root).as_posix()
    except ValueError as exc:
        raise PublishedPairError("published wheel is outside the repository") from exc
    wheel_sha256 = hashlib.sha256(wheel_payload).hexdigest()
    if artifact.get("artifact") != relative_wheel:
        raise PublishedPairError(
            "published acceptance artifact path does not match final wheel"
        )
    artifact_path = PurePosixPath(relative_wheel)
    release_version = receipt.get("release_version")
    distribution = artifact.get("distribution")
    if (
        artifact_path.parts != ("dist", wheel.name)
        or not isinstance(release_version, str)
        or not release_version
        or not isinstance(distribution, str)
        or not distribution
        or not wheel.name.startswith(
            f"{distribution.replace('-', '_')}-{release_version}-"
        )
        or wheel.suffix != ".whl"
    ):
        raise PublishedPairError("published wheel filename/version policy mismatch")
    expected_acceptance_name = f"release_acceptance_{release_version}.json"
    if acceptance.name != expected_acceptance_name:
        raise PublishedPairError(
            "published acceptance filename/version policy mismatch"
        )
    if artifact.get("size_bytes") != len(wheel_payload):
        raise PublishedPairError(
            "published acceptance artifact size does not match final wheel"
        )
    if artifact.get("sha256") != wheel_sha256:
        raise PublishedPairError(
            "published acceptance artifact hash does not match final wheel"
        )
    for field in (
        "distribution",
        "version",
        "size_bytes",
        "uncompressed_bytes",
        "member_count",
        "sha256",
        "license_expression",
        "requires_python",
        "members_sha256",
    ):
        if artifact[field] != wheel_summary[field]:
            raise PublishedPairError(
                f"published acceptance wheel summary is stale: {field}"
            )
    if artifact["release_audit_finding_count"] != 0:
        raise PublishedPairError("published wheel release audit has findings")
    _validate_recomputed_release_audit(
        wheel,
        expected_kind="archive",
        expected_members=wheel_members,
        description="published wheel",
    )
    allowlist = verify_wheel_allowlist(wheel, root)
    if (
        allowlist.get("passed") is not True
        or allowlist.get("errors") != []
        or allowlist.get("missing_members") != []
        or allowlist.get("unexpected_members") != []
        or allowlist.get("content_mismatches") != []
        or allowlist.get("duplicate_members") != []
        or allowlist.get("member_count") != wheel_summary["member_count"]
        or allowlist.get("wheel_size_bytes") != wheel_summary["size_bytes"]
        or allowlist.get("wheel_sha256") != wheel_summary["sha256"]
    ):
        raise PublishedPairError("published wheel fails the current exact allowlist")

    public_artifacts = _exact_keys(
        receipt.get("public_artifacts"),
        {
            "directory",
            "member_count",
            "total_size_bytes",
            "release_audit_finding_count",
            "members_sha256",
        },
        "published public artifacts",
    )
    if public_artifacts["directory"] != "artifacts/public":
        raise PublishedPairError("published public-artifact directory is noncanonical")
    public_relative = _safe_relative_path(
        public_artifacts["directory"], "published public-artifact directory"
    )
    public_members = _directory_members(root.joinpath(*public_relative.parts))
    public_size = sum(size for size, _ in public_members.values())
    if (
        public_artifacts["member_count"] != len(public_members)
        or public_artifacts["total_size_bytes"] != public_size
        or public_artifacts["members_sha256"] != _members_digest(public_members)
        or public_artifacts["release_audit_finding_count"] != 0
    ):
        raise PublishedPairError("published public-artifact binding is stale")
    _validate_recomputed_release_audit(
        root.joinpath(*public_relative.parts),
        expected_kind="directory",
        expected_members=public_members,
        description="published public artifacts",
    )

    binding = _exact_keys(
        receipt.get("verification_binding"),
        {
            "verification_scope",
            "release_source_snapshot",
            "project_verification_receipt_sha256",
            "exact_wheel_verification_receipt_sha256",
        },
        "published verification binding",
    )
    if binding.get("verification_scope") != expected_scope:
        raise PublishedPairError(
            "published acceptance is bound to another source snapshot"
        )
    snapshot_manifest = _object(
        _regular_bytes(source_snapshot_manifest, "source snapshot manifest"),
        "source snapshot manifest",
    )
    if snapshot_manifest.get("schema_version") != (
        "teaching_skill_miner.release_source_snapshot.v1"
    ):
        raise PublishedPairError(
            "source snapshot manifest has an unexpected schema"
        )
    expected_snapshot_binding = _snapshot_bindings(root, snapshot_manifest)
    source_binding = binding.get("release_source_snapshot")
    if (
        source_binding != expected_snapshot_binding
        or expected_snapshot_binding["source_snapshot"]["file_count"] < 1
    ):
        raise PublishedPairError(
            "published acceptance is not bound to this nonempty release snapshot"
        )
    for name in (
        "project_verification_receipt_sha256",
        "exact_wheel_verification_receipt_sha256",
    ):
        value = binding.get(name)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise PublishedPairError(f"published acceptance has invalid {name}")
    _validate_original_proofs(
        receipt=receipt,
        binding=binding,
        project_path=project_verification_receipt,
        exact_path=exact_wheel_verification_receipt,
        wheel_summary=wheel_summary,
        scope=expected_scope,
        snapshot_binding=expected_snapshot_binding,
        validation=validation,
        conditional=conditional,
        public_members=public_members,
    )
    if release_version != artifact.get("version"):
        raise PublishedPairError("published acceptance release version is inconsistent")
    return {
        "artifact": relative_wheel,
        "size_bytes": len(wheel_payload),
        "sha256": wheel_sha256,
        "verification_scope": expected_scope,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--expected-source-scope", type=Path, required=True)
    parser.add_argument("--source-snapshot-manifest", type=Path, required=True)
    parser.add_argument(
        "--project-verification-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--exact-wheel-verification-receipt", type=Path, required=True
    )
    args = parser.parse_args()
    try:
        result = verify_published_pair(
            repository_root=args.repository_root,
            wheel=args.wheel,
            acceptance=args.acceptance,
            expected_source_scope=args.expected_source_scope,
            source_snapshot_manifest=args.source_snapshot_manifest,
            project_verification_receipt=args.project_verification_receipt,
            exact_wheel_verification_receipt=(
                args.exact_wheel_verification_receipt
            ),
        )
    except (OSError, PublishedPairError) as exc:
        parser.error(str(exc))
    print(f"published_release_pair_verified={result['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
