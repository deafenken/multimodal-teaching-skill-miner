"""Allowlist-oriented public release and privacy auditing."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable
import zipfile

from .io_utils import write_json


FORBIDDEN_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".npz",
    ".npy",
    ".parquet",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".safetensors",
    ".pkl",
    ".pickle",
    ".joblib",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".pdf",
    ".dylib",
    ".so",
    ".dll",
    ".exe",
}
FORBIDDEN_PARTS = {
    "data/real",
    "archive_cache",
    "artifacts/private",
    "artifacts/dipser_credible",
    ".env",
    "credentials",
}
SENSITIVE_DATA_FIELDS = {
    "participant_id",
    "student_id",
    "teacher_id",
    "reviewer_id",
    "annotator_id",
    "face_id",
    "person_id",
    "watch_path",
    "pose_path",
    "sample_id",
    "session_id",
    "cohort_id",
    "activity_id",
    "file_path",
    "relative_path",
}
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_./+-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
ABSOLUTE_LOCAL_PATH = re.compile(rb"/(?:Users|Volumes|home)/[^\s\"']+")
SECRET_TEXT_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(
        r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_./+-]{20,}"
    ),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
ABSOLUTE_LOCAL_PATH_TEXT = re.compile(r"/(?:Users|Volumes|home)/[^\s\"']+")
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".txt",
    ".md",
    ".log",
    ".toml",
    ".yaml",
    ".yml",
    ".py",
    ".sh",
    ".cfg",
    ".ini",
}
BINARY_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"PK\x03\x04", "zip"),
    (b"SQLite format 3\x00", "sqlite"),
    (b"\x1a\x45\xdf\xa3", "matroska-or-webm"),
    (b"fLaC", "flac"),
    (b"OggS", "ogg"),
    (b"ID3", "mp3"),
    (b"\x93NUMPY", "numpy"),
    (b"PAR1", "parquet"),
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bzip2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"7z\xbc\xaf'\x1c", "7z"),
    (b"Rar!\x1a\x07", "rar"),
    (b"%PDF-", "pdf"),
    (b"\x7fELF", "elf"),
    (b"\xfe\xed\xfa\xce", "mach-o"),
    (b"\xfe\xed\xfa\xcf", "mach-o"),
    (b"\xce\xfa\xed\xfe", "mach-o"),
    (b"\xcf\xfa\xed\xfe", "mach-o"),
    (b"\xca\xfe\xba\xbe", "mach-o-fat"),
)
WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:")
MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO = 1_000.0


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalized(name: str) -> str:
    return str(PurePosixPath(name.replace("\\", "/")))


def _unsafe_member_path_reasons(name: str) -> list[str]:
    raw = name.replace("\\", "/")
    reasons: list[str] = []
    if not raw or "\x00" in raw:
        reasons.append("empty-or-nul")
    if raw.startswith("/") or WINDOWS_DRIVE_PATH.match(raw):
        reasons.append("absolute")
    parts = raw.split("/")
    if any(part == ".." for part in parts):
        reasons.append("parent-traversal")
    if any(part in {"", "."} for part in parts):
        reasons.append("ambiguous-segment")
    return sorted(set(reasons))


def _iter_directory(path: Path) -> Iterable[tuple[str, bytes]]:
    for file_path in sorted(value for value in path.rglob("*") if value.is_file()):
        yield file_path.relative_to(path).as_posix(), file_path.read_bytes()


def _iter_archive(path: Path) -> Iterable[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda value: value.filename):
            if info.is_dir():
                continue
            yield info.filename, archive.read(info)


def _archive_preflight(path: Path) -> tuple[list[zipfile.ZipInfo], list[dict[str, Any]]]:
    """Inspect ZIP metadata before decompression so the auditor cannot be zip-bombed."""

    findings: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as archive:
            infos = sorted(archive.infolist(), key=lambda value: value.filename)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        return [], [
            {
                "path": path.name,
                "rule": "invalid_archive",
                "detail": type(exc).__name__,
            }
        ]

    if len(infos) > MAX_ARCHIVE_MEMBERS:
        findings.append(
            {
                "path": path.name,
                "rule": "archive_resource_limit",
                "detail": f"member_count={len(infos)} exceeds {MAX_ARCHIVE_MEMBERS}",
            }
        )
    total_uncompressed = sum(info.file_size for info in infos if not info.is_dir())
    if total_uncompressed > MAX_ARCHIVE_TOTAL_BYTES:
        findings.append(
            {
                "path": path.name,
                "rule": "archive_resource_limit",
                "detail": (
                    f"total_uncompressed_bytes={total_uncompressed} exceeds "
                    f"{MAX_ARCHIVE_TOTAL_BYTES}"
                ),
            }
        )
    for info in infos:
        if info.is_dir():
            continue
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(unix_mode):
            findings.append(
                {
                    "path": info.filename,
                    "rule": "symlink_member",
                    "detail": "public release archives must contain regular files only",
                }
            )
        if info.flag_bits & 0x1:
            findings.append(
                {
                    "path": info.filename,
                    "rule": "encrypted_archive_member",
                    "detail": "encrypted members cannot be privacy-audited",
                }
            )
        if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            findings.append(
                {
                    "path": info.filename,
                    "rule": "archive_resource_limit",
                    "detail": (
                        f"member_uncompressed_bytes={info.file_size} exceeds "
                        f"{MAX_ARCHIVE_MEMBER_BYTES}"
                    ),
                }
            )
        if info.file_size:
            ratio = (
                float("inf")
                if info.compress_size == 0
                else info.file_size / info.compress_size
            )
            if ratio > MAX_ARCHIVE_COMPRESSION_RATIO:
                findings.append(
                    {
                        "path": info.filename,
                        "rule": "archive_resource_limit",
                        "detail": (
                            f"compression_ratio={ratio:.1f} exceeds "
                            f"{MAX_ARCHIVE_COMPRESSION_RATIO:.1f}"
                        ),
                    }
                )
    return infos, findings


def forbidden_binary_payload_kind(payload: bytes) -> str | None:
    """Identify common private binary/media formats even after extension changes."""

    for signature, kind in BINARY_SIGNATURES:
        if payload.startswith(signature):
            return kind
    if len(payload) >= 12 and payload[4:8] == b"ftyp":
        return "iso-base-media"
    if len(payload) >= 12 and payload.startswith(b"RIFF"):
        riff_kind = payload[8:12]
        if riff_kind == b"WAVE":
            return "wav"
        if riff_kind == b"AVI ":
            return "avi"
        if riff_kind == b"WEBP":
            return "webp"
    if len(payload) >= 2 and payload[0] == 0x80 and payload[1] in range(2, 6):
        return "pickle"
    if (
        len(payload) >= 2
        and not payload.startswith((b"\xff\xfe", b"\xfe\xff"))
        and payload[0] == 0xFF
        and payload[1] & 0xE0 == 0xE0
    ):
        return "mp3-or-aac-frame"
    if payload.startswith(b"MZ"):
        return "portable-executable"
    if len(payload) >= 262 and payload[257:262] == b"ustar":
        return "tar"
    return None


def _text_is_printable(text: str) -> bool:
    if not text:
        return True
    control_count = sum(
        ord(char) < 32 and char not in "\n\r\t\f\b" for char in text
    )
    return control_count <= max(1, len(text) // 100)


def decode_text_payload(payload: bytes) -> str | None:
    """Decode ordinary UTF-8/UTF-16 publication text, including BOM-less ASCII UTF-16."""

    if not payload:
        return ""
    if payload.startswith(b"\xef\xbb\xbf"):
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None
        return text if _text_is_printable(text) else None
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = payload.decode("utf-16")
        except UnicodeDecodeError:
            return None
        return text if _text_is_printable(text) else None
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is not None and "\x00" not in text and _text_is_printable(text):
        return text

    sample = payload[:65536]
    if len(sample) < 4:
        return None
    even = sample[0::2]
    odd = sample[1::2]
    even_null_fraction = even.count(0) / max(1, len(even))
    odd_null_fraction = odd.count(0) / max(1, len(odd))
    encoding = None
    if odd_null_fraction >= 0.30 and even_null_fraction <= 0.10:
        encoding = "utf-16-le"
    elif even_null_fraction >= 0.30 and odd_null_fraction <= 0.10:
        encoding = "utf-16-be"
    if encoding is None:
        return None
    try:
        text = payload.decode(encoding)
    except UnicodeDecodeError:
        return None
    return text if _text_is_printable(text) else None


def _probably_text(payload: bytes) -> bool:
    return decode_text_payload(payload) is not None


def secret_pattern_details(payload: bytes) -> list[str]:
    """Return matched secret-pattern descriptions across supported text encodings."""

    details = {
        pattern.pattern.decode("ascii", errors="replace")
        for pattern in SECRET_PATTERNS
        if pattern.search(payload)
    }
    text = decode_text_payload(payload)
    if text is not None:
        details.update(
            pattern.pattern
            for pattern in SECRET_TEXT_PATTERNS
            if pattern.search(text)
        )
    return sorted(details)


def absolute_local_path_value(payload: bytes) -> str | None:
    """Return one absolute workstation path from UTF-8/UTF-16 text, if present."""

    absolute = ABSOLUTE_LOCAL_PATH.search(payload)
    if absolute:
        return absolute.group(0).decode("utf-8", errors="replace")[:180]
    text = decode_text_payload(payload)
    if text is None:
        return None
    match = ABSOLUTE_LOCAL_PATH_TEXT.search(text)
    return match.group(0)[:180] if match else None


def _json_identity_keys(
    value: Any,
    *,
    schema_document: bool,
    parent_key: str | None = None,
    suppress_string_tokens: bool = False,
) -> set[str]:
    fields: set[str] = set()
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            schema_property_definition = bool(
                schema_document
                and (
                    (
                        parent_key
                        in {
                            "properties",
                            "patternProperties",
                            "dependentSchemas",
                            "$defs",
                            "definitions",
                        }
                        and isinstance(child, (dict, bool))
                    )
                    or parent_key == "dependentRequired"
                )
            )
            if key in SENSITIVE_DATA_FIELDS and not schema_property_definition:
                fields.add(key)
            fields.update(
                _json_identity_keys(
                    child,
                    schema_document=schema_document,
                    parent_key=key,
                    suppress_string_tokens=bool(
                        schema_document
                        and (
                            key in {"required", "enum", "const", "examples"}
                            or parent_key == "dependentRequired"
                        )
                    ),
                )
            )
    elif isinstance(value, list):
        for child in value:
            if (
                isinstance(child, str)
                and child in SENSITIVE_DATA_FIELDS
                and not suppress_string_tokens
            ):
                fields.add(child)
            fields.update(
                _json_identity_keys(
                    child,
                    schema_document=schema_document,
                    parent_key=parent_key,
                    suppress_string_tokens=suppress_string_tokens,
                )
            )
    elif (
        isinstance(value, str)
        and value in SENSITIVE_DATA_FIELDS
        and not suppress_string_tokens
    ):
        fields.add(value)
    return fields


def sensitive_identity_fields(name: str, payload: bytes) -> list[str]:
    """Return row-level identity fields using format-aware parsing."""

    normalized = _normalized(name)
    lowered = normalized.lower()
    suffix = PurePosixPath(lowered).suffix
    text = decode_text_payload(payload)
    if text is None:
        return []
    try:
        stripped = text.lstrip("\ufeff \t\r\n")
        if suffix == ".json" or stripped.startswith(("{", "[")):
            value = json.loads(text)
            schema_document = bool(
                isinstance(value, dict)
                and isinstance(value.get("$schema"), str)
                and value["$schema"].startswith("https://json-schema.org/")
            )
            return sorted(
                _json_identity_keys(value, schema_document=schema_document)
            )
        if suffix == ".jsonl":
            fields: set[str] = set()
            for line in text.splitlines():
                if line.strip():
                    fields.update(
                        _json_identity_keys(
                            json.loads(line),
                            schema_document=False,
                        )
                    )
            return sorted(fields)
        delimiter = "\t" if suffix == ".tsv" or "\t" in text.partition("\n")[0] else ","
        rows = csv.reader(io.StringIO(text), delimiter=delimiter)
        header = [value.strip().lstrip("\ufeff") for value in next(rows, [])]
        return sorted(set(header) & SENSITIVE_DATA_FIELDS)
    except (csv.Error, json.JSONDecodeError, TypeError, ValueError):
        return [
            field
            for field in sorted(SENSITIVE_DATA_FIELDS)
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])",
                text,
            )
        ]


def _inspect_member(name: str, payload: bytes) -> list[dict[str, Any]]:
    normalized = _normalized(name)
    lowered = normalized.lower()
    suffix = PurePosixPath(lowered).suffix
    findings: list[dict[str, Any]] = []
    for reason in _unsafe_member_path_reasons(name):
        findings.append(
            {
                "path": name,
                "rule": "unsafe_member_path",
                "detail": reason,
            }
        )
    if suffix in FORBIDDEN_SUFFIXES:
        findings.append({"path": normalized, "rule": "forbidden_binary_or_archive", "detail": suffix})
    binary_kind = forbidden_binary_payload_kind(payload)
    if binary_kind:
        findings.append(
            {
                "path": normalized,
                "rule": "forbidden_binary_signature",
                "detail": binary_kind,
            }
        )
    elif decode_text_payload(payload) is None:
        findings.append(
            {
                "path": normalized,
                "rule": "unknown_binary_payload",
                "detail": "public releases allow recognized text members only",
            }
        )
    for part in FORBIDDEN_PARTS:
        if lowered == part or lowered.startswith(part + "/") or f"/{part}/" in f"/{lowered}/":
            findings.append({"path": normalized, "rule": "forbidden_private_path", "detail": part})
    if any(
        part == ".env" or part.startswith(".env.")
        for part in PurePosixPath(lowered).parts
    ):
        findings.append(
            {
                "path": normalized,
                "rule": "forbidden_private_path",
                "detail": ".env*",
            }
        )
    if suffix in TEXT_SUFFIXES or _probably_text(payload):
        for detail in secret_pattern_details(payload):
            findings.append(
                {"path": normalized, "rule": "possible_secret", "detail": detail}
            )
        absolute = absolute_local_path_value(payload)
        if absolute:
            findings.append(
                {
                    "path": normalized,
                    "rule": "absolute_local_path",
                    "detail": absolute,
                }
            )
        for field in sensitive_identity_fields(normalized, payload):
            findings.append(
                {
                    "path": normalized,
                    "rule": "row_level_identity_field",
                    "detail": field,
                }
            )
    return findings


def audit_release_path(path: str | Path) -> dict[str, Any]:
    """Audit a wheel/zip or directory intended for public distribution."""

    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    findings: list[dict[str, Any]] = []
    if target.is_dir():
        for file_path in sorted(target.rglob("*")):
            if file_path.is_symlink():
                findings.append(
                    {
                        "path": file_path.relative_to(target).as_posix(),
                        "rule": "symlink_member",
                        "detail": "public release directories must contain regular files only",
                    }
                )
        members = list(_iter_directory(target))
        kind = "directory"
    elif target.suffix.lower() in {".whl", ".zip"}:
        kind = "archive"
        infos, archive_findings = _archive_preflight(target)
        findings.extend(archive_findings)
        unsafe_to_decompress = any(
            finding["rule"]
            in {
                "invalid_archive",
                "archive_resource_limit",
                "encrypted_archive_member",
            }
            for finding in archive_findings
        )
        if unsafe_to_decompress:
            members = []
            for info in infos:
                if info.is_dir():
                    continue
                for reason in _unsafe_member_path_reasons(info.filename):
                    findings.append(
                        {
                            "path": info.filename,
                            "rule": "unsafe_member_path",
                            "detail": reason,
                        }
                    )
        else:
            try:
                members = list(_iter_archive(target))
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                members = []
                findings.append(
                    {
                        "path": target.name,
                        "rule": "archive_read_error",
                        "detail": type(exc).__name__,
                    }
                )
    else:
        members = [(target.name, target.read_bytes())]
        kind = "file"
    member_rows: list[dict[str, Any]] = []
    seen_members: set[str] = set()
    for name, payload in members:
        normalized = _normalized(name)
        if normalized in seen_members:
            findings.append(
                {
                    "path": normalized,
                    "rule": "duplicate_member_path",
                    "detail": "multiple archive entries normalize to the same path",
                }
            )
        seen_members.add(normalized)
        member_rows.append(
            {
                "path": normalized,
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
        findings.extend(_inspect_member(normalized, payload))
    return {
        "schema_version": "1.0",
        "audit_kind": "public_release_privacy_audit",
        "target": str(target),
        "target_kind": kind,
        "member_count": len(member_rows),
        "total_uncompressed_bytes": sum(item["size_bytes"] for item in member_rows),
        "passed": not findings,
        "finding_count": len(findings),
        "findings": findings,
        "members": member_rows,
        "policy": {
            "forbidden_suffixes": sorted(FORBIDDEN_SUFFIXES),
            "forbidden_private_paths": sorted(FORBIDDEN_PARTS),
            "secret_scan": True,
            "row_level_identity_scan": True,
            "unknown_binary_payloads_fail_closed": True,
            "archive_limits": {
                "max_members": MAX_ARCHIVE_MEMBERS,
                "max_member_uncompressed_bytes": MAX_ARCHIVE_MEMBER_BYTES,
                "max_total_uncompressed_bytes": MAX_ARCHIVE_TOTAL_BYTES,
                "max_compression_ratio": MAX_ARCHIVE_COMPRESSION_RATIO,
            },
        },
    }


def _metric_subset(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    allowed = (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "sample_count",
        "confusion_matrix",
        "per_class",
    )
    output = {key: value[key] for key in allowed if key in value}
    return output or None


def public_dipser_summary(report: dict[str, Any], *, source_sha256: str) -> dict[str, Any]:
    """Export aggregate-only DIPSER facts without row-level identities or features."""

    nested = report.get("nested_offline_full_session", {})
    nested_metrics: dict[str, Any] = {}
    if isinstance(nested, dict):
        for design, value in nested.items():
            if isinstance(value, dict):
                metric = _metric_subset(value.get("nested_hierarchical_metrics"))
                if metric:
                    nested_metrics[str(design)] = metric
    causal = report.get("strict_causal_reference", {})
    modality = report.get("nested_modality_ablation", {})
    modalities: dict[str, Any] = {}
    if isinstance(modality, dict):
        for design, design_value in modality.items():
            if not isinstance(design_value, dict):
                continue
            design_metrics: dict[str, Any] = {}
            for name, value in design_value.items():
                if isinstance(value, dict):
                    metric = _metric_subset(value.get("metrics", value))
                    if metric:
                        design_metrics[str(name)] = metric
            if design_metrics:
                modalities[str(design)] = design_metrics
    return {
        "schema_version": "1.0",
        "report_kind": "aggregate_public_dipser_research_summary",
        "source_report_sha256": source_sha256,
        "evaluation_kind": report.get("evaluation_kind"),
        "requested_target": report.get("requested_target"),
        "claim_status": report.get("claim_status", {}),
        "claim_limitation": report.get("claim_limitation"),
        "strict_causal_reference": {
            "session_sgkf5": _metric_subset(causal.get("session_sgkf5")) if isinstance(causal, dict) else None,
            "leave_one_session_out": _metric_subset(causal.get("leave_one_session_out")) if isinstance(causal, dict) else None,
            "real_time_compatible": causal.get("real_time_compatible") if isinstance(causal, dict) else None,
            "post_selection_exploratory": causal.get("post_selection_exploratory") if isinstance(causal, dict) else None,
        },
        "offline_full_session": nested_metrics,
        "modality_ablation": modalities,
        "outer_seed_sensitivity": report.get("outer_seed_sensitivity", {}).get("metrics")
        if isinstance(report.get("outer_seed_sensitivity"), dict)
        else None,
        "privacy": {
            "row_level_predictions_included": False,
            "sample_ids_included": False,
            "participant_ids_included": False,
            "feature_matrices_included": False,
            "source_paths_included": False,
        },
    }


def export_public_dipser_report(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    source = Path(input_path)
    payload = source.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DIPSER report must be one JSON object")
    summary = public_dipser_summary(value, source_sha256=_sha256_bytes(payload))
    target = write_json(output_path, summary)
    target.chmod(0o644)
    return summary
