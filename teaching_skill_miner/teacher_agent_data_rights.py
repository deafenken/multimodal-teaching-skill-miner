"""Private export, deletion receipts, and encrypted local backup primitives.

The public dashboard projection intentionally omits most learner content.  Data
rights operations must use the validated private store objects instead: an
export made from a browser projection would be incomplete, while a deletion
made from unvalidated client-provided IDs could remove another project's data.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import struct
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from .io_utils import ensure_private_directory, ensure_private_file, write_json


PROJECT_EXPORT_SCHEMA = "teaching_skill_miner.project_private_export.v1"
DELETION_RECEIPT_SCHEMA = "teaching_skill_miner.project_deletion_receipt.v1"
LOCAL_BACKUP_SCHEMA = "teaching_skill_miner.local_encrypted_backup.v1"
LOCAL_BACKUP_MANIFEST_SCHEMA = "teaching_skill_miner.local_encrypted_backup_manifest.v1"
RESTORE_DRILL_SCHEMA = "teaching_skill_miner.local_backup_restore_drill.v1"
_BACKUP_MAGIC = b"TSM-BACKUP-1\n"
_MAX_EXPORT_BYTES = 256 * 1024 * 1024
_MAX_BACKUP_BYTES = 1024 * 1024 * 1024
_MAX_BACKUP_FILES = 20_000
_SAFE_ARCHIVE_PART = re.compile(r"^[A-Za-z0-9_.-]{1,200}$")


class TeacherAgentDataRightsError(RuntimeError):
    """Raised when a private data-rights operation cannot complete exactly."""


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeacherAgentDataRightsError("data-rights JSON is not canonical") from exc


def _json_document(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _safe_archive_name(value: str) -> str:
    if _SAFE_ARCHIVE_PART.fullmatch(value) is None or value in {".", ".."}:
        raise TeacherAgentDataRightsError("private export identifier is unsafe")
    return value


@dataclass(frozen=True, slots=True)
class PrivateExportEntry:
    path: str
    content: bytes
    data_class: str
    reference_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectExportArchive:
    filename: str
    payload: bytes
    manifest_sha256: str
    entry_count: int


def build_project_export_archive(
    *,
    project: Mapping[str, Any],
    sessions: Mapping[str, Mapping[str, Any]],
    syllabi: Mapping[str, Mapping[str, Any]],
    syllabus_versions: Mapping[str, Mapping[str, Any]] | None = None,
    curriculum_authorities: Mapping[str, Mapping[str, Any]] | None = None,
    resources: Mapping[str, Mapping[str, Any]],
    resource_reviews: Mapping[str, Mapping[str, Any]] | None = None,
    learning_records: Mapping[str, Mapping[str, Any]] | None = None,
    metacognition_records: Mapping[str, Mapping[str, Any]] | None = None,
    adjudications: Mapping[str, Mapping[str, Any]] | None = None,
    consent_receipts: Mapping[str, Mapping[str, Any]] | None = None,
    background_tasks: Mapping[str, Mapping[str, Any]] | None = None,
    stream_artifacts: Mapping[str, bytes] | None = None,
) -> ProjectExportArchive:
    """Build a complete private project ZIP with a canonical hash manifest."""

    project_id = _safe_archive_name(str(project.get("project_id", "")))
    entries: list[PrivateExportEntry] = [
        PrivateExportEntry(
            "project/project.json",
            _json_document(project),
            "learning_project_private",
            project_id,
        )
    ]
    for session_id, value in sorted(sessions.items()):
        safe_id = _safe_archive_name(session_id)
        entries.append(
            PrivateExportEntry(
                f"sessions/{safe_id}.json",
                _json_document(value),
                "teaching_session_private",
                session_id,
            )
        )
    for syllabus_id, value in sorted(syllabi.items()):
        safe_id = _safe_archive_name(syllabus_id)
        entries.append(
            PrivateExportEntry(
                f"syllabi/{safe_id}.json",
                _json_document(value),
                "teaching_syllabus_private",
                syllabus_id,
            )
        )
    for family_id, value in sorted((syllabus_versions or {}).items()):
        safe_id = _safe_archive_name(family_id)
        entries.append(
            PrivateExportEntry(
                f"syllabus_versions/{safe_id}.json",
                _json_document(value),
                "teaching_syllabus_version_projection_private",
                family_id,
            )
        )
    for family_id, value in sorted((curriculum_authorities or {}).items()):
        safe_id = _safe_archive_name(family_id)
        entries.append(
            PrivateExportEntry(
                f"curriculum_authorities/{safe_id}.json",
                _json_document(value),
                "teacher_curriculum_authority_private_audit",
                family_id,
            )
        )
    for resource_id, value in sorted(resources.items()):
        safe_id = _safe_archive_name(resource_id)
        entries.append(
            PrivateExportEntry(
                f"resources/{safe_id}.json",
                _json_document(value),
                "teaching_resource_private_index",
                resource_id,
            )
        )
    for resource_id, value in sorted((resource_reviews or {}).items()):
        safe_id = _safe_archive_name(resource_id)
        entries.append(
            PrivateExportEntry(
                f"resource_reviews/{safe_id}.json",
                _json_document(value),
                "teaching_resource_review_private_audit",
                resource_id,
            )
        )
    for learner_key, value in sorted((learning_records or {}).items()):
        safe_id = _safe_archive_name(learner_key)
        entries.append(
            PrivateExportEntry(
                f"learning_records/{safe_id}.json",
                _json_document(value),
                "learner_spaced_review_private",
                learner_key,
            )
        )
    for learner_key, value in sorted((metacognition_records or {}).items()):
        safe_id = _safe_archive_name(learner_key)
        entries.append(
            PrivateExportEntry(
                f"metacognition_records/{safe_id}.json",
                _json_document(value),
                "learner_metacognition_calibration_private",
                learner_key,
            )
        )
    for item_id, value in sorted((adjudications or {}).items()):
        safe_id = _safe_archive_name(item_id)
        entries.append(
            PrivateExportEntry(
                f"adjudications/{safe_id}.json",
                _json_document(value),
                "assessment_adjudication_private_audit",
                item_id,
            )
        )
    for consent_id, value in sorted((consent_receipts or {}).items()):
        safe_id = _safe_archive_name(consent_id)
        entries.append(
            PrivateExportEntry(
                f"consent_receipts/{safe_id}.json",
                _json_document(value),
                "remote_processing_consent_private_audit",
                consent_id,
            )
        )
    for task_id, value in sorted((background_tasks or {}).items()):
        safe_id = _safe_archive_name(task_id)
        entries.append(
            PrivateExportEntry(
                f"background_tasks/{safe_id}.json",
                _json_document(value),
                "background_task_private_recovery",
                task_id,
            )
        )
    for raw_name, content in sorted((stream_artifacts or {}).items()):
        name = PurePosixPath(raw_name)
        if name.is_absolute() or ".." in name.parts or not name.parts:
            raise TeacherAgentDataRightsError("stream export path is unsafe")
        if not all(_SAFE_ARCHIVE_PART.fullmatch(part) for part in name.parts):
            raise TeacherAgentDataRightsError("stream export path is unsafe")
        entries.append(
            PrivateExportEntry(
                f"stream_journals/{name.as_posix()}",
                bytes(content),
                "harness_stream_private",
            )
        )

    entries.sort(key=lambda item: item.path)
    if len({item.path for item in entries}) != len(entries):
        raise TeacherAgentDataRightsError("private export contains duplicate paths")
    if sum(len(item.content) for item in entries) > _MAX_EXPORT_BYTES:
        raise TeacherAgentDataRightsError(
            "private project export expanded contents exceed their limit"
        )
    entry_manifest = [
        {
            "path": item.path,
            "bytes": len(item.content),
            "sha256": sha256(item.content).hexdigest(),
            "media_type": (
                "application/json"
                if item.path.endswith(".json")
                else "application/octet-stream"
            ),
            "data_class": item.data_class,
            "reference_id": item.reference_id,
        }
        for item in entries
    ]
    manifest = {
        "schema": PROJECT_EXPORT_SCHEMA,
        "project_id": project_id,
        "exported_at": _now(),
        "entry_count": len(entry_manifest),
        "entries": entry_manifest,
        "claim_boundary": {
            "contains_private_content": True,
            "built_from_public_projection": False,
            "suitable_for_public_release": False,
        },
    }
    manifest_bytes = _json_document(manifest)
    manifest_sha = sha256(manifest_bytes).hexdigest()

    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, content in [
            *[(item.path, item.content) for item in entries],
            ("manifest.json", manifest_bytes),
            ("manifest.sha256", f"{manifest_sha}  manifest.json\n".encode("ascii")),
        ]:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, content)
    payload = output.getvalue()
    if not payload or len(payload) > _MAX_EXPORT_BYTES:
        raise TeacherAgentDataRightsError("private project export exceeds its limit")
    return ProjectExportArchive(
        filename=f"{project_id}-private-export.zip",
        payload=payload,
        manifest_sha256=manifest_sha,
        entry_count=len(entry_manifest),
    )


def validate_project_export_archive(payload: bytes) -> dict[str, Any]:
    """Validate every declared export entry and the canonical manifest hash."""

    if not payload or len(payload) > _MAX_EXPORT_BYTES:
        raise TeacherAgentDataRightsError("private project export size is invalid")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if (
                len(names) != len(set(names))
                or len(names) > 20_002
                or "manifest.json" not in names
                or archive.getinfo("manifest.json").file_size > 16_000_000
                or "manifest.sha256" not in names
                or archive.getinfo("manifest.sha256").file_size > 256
                or any(
                    PurePosixPath(name).is_absolute()
                    or ".." in PurePosixPath(name).parts
                    for name in names
                )
            ):
                raise TeacherAgentDataRightsError("private export paths are unsafe")
            manifest_bytes = archive.read("manifest.json")
            digest_line = archive.read("manifest.sha256").decode("ascii")
            manifest = json.loads(manifest_bytes)
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema") != PROJECT_EXPORT_SCHEMA
                or manifest_bytes != _json_document(manifest)
                or digest_line
                != f"{sha256(manifest_bytes).hexdigest()}  manifest.json\n"
            ):
                raise TeacherAgentDataRightsError("private export manifest is invalid")
            entries = manifest.get("entries")
            if not isinstance(entries, list) or manifest.get("entry_count") != len(
                entries
            ):
                raise TeacherAgentDataRightsError(
                    "private export entry count is invalid"
                )
            declared = {"manifest.json", "manifest.sha256"}
            cumulative_bytes = 0
            for entry in entries:
                if (
                    not isinstance(entry, Mapping)
                    or set(entry)
                    != {
                        "path",
                        "bytes",
                        "sha256",
                        "media_type",
                        "data_class",
                        "reference_id",
                    }
                    or not isinstance(entry.get("path"), str)
                    or isinstance(entry.get("bytes"), bool)
                    or not isinstance(entry.get("bytes"), int)
                    or not 0 <= int(entry["bytes"]) <= _MAX_EXPORT_BYTES
                    or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
                    is None
                    or entry.get("media_type")
                    not in {"application/json", "application/octet-stream"}
                    or not isinstance(entry.get("data_class"), str)
                    or entry.get("reference_id") is not None
                    and not isinstance(entry.get("reference_id"), str)
                ):
                    raise TeacherAgentDataRightsError("private export entry is invalid")
                name = str(entry["path"])
                if name in declared:
                    raise TeacherAgentDataRightsError(
                        "private export path is duplicated"
                    )
                cumulative_bytes += int(entry["bytes"])
                if cumulative_bytes > _MAX_EXPORT_BYTES:
                    raise TeacherAgentDataRightsError(
                        "private export expanded size exceeds its limit"
                    )
                info = archive.getinfo(name)
                if info.is_dir() or info.file_size != entry["bytes"]:
                    raise TeacherAgentDataRightsError(
                        "private export ZIP entry size is invalid"
                    )
                content = archive.read(name)
                if (
                    entry.get("bytes") != len(content)
                    or entry.get("sha256") != sha256(content).hexdigest()
                ):
                    raise TeacherAgentDataRightsError(
                        "private export entry hash failed"
                    )
                declared.add(name)
            if set(names) != declared:
                raise TeacherAgentDataRightsError(
                    "private export has undeclared entries"
                )
            return dict(manifest)
    except (
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        if isinstance(exc, TeacherAgentDataRightsError):
            raise
        raise TeacherAgentDataRightsError("private project export is invalid") from exc


def build_content_free_deletion_receipt(
    *,
    project_id: str,
    recovery_token: str,
    deleted_counts: Mapping[str, int],
    retained_shared_counts: Mapping[str, int],
    target_identifiers: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Describe a completed purge without retaining learner-authored content."""

    normalized_targets = {
        key: sorted(str(item) for item in values)
        for key, values in sorted(target_identifiers.items())
    }
    return {
        "schema": DELETION_RECEIPT_SCHEMA,
        "receipt_id": "delete_" + secrets.token_hex(16),
        "project_id": project_id,
        "deleted_at": _now(),
        "request_token_sha256": sha256(recovery_token.encode("utf-8")).hexdigest(),
        "target_set_sha256": sha256(
            canonical_json_bytes(normalized_targets)
        ).hexdigest(),
        "deleted_counts": {
            key: int(value) for key, value in sorted(deleted_counts.items())
        },
        "retained_shared_counts": {
            key: int(value) for key, value in sorted(retained_shared_counts.items())
        },
        "storage_cleanup_state": "completed",
        "content_retained_in_receipt": False,
        "user_managed_export_copies_deleted": False,
        "user_managed_export_copies_status": "not_verifiable_by_running_app",
        "backup_scope": "configured_live_stores_only",
    }


def deletion_confirmation(project_id: str) -> str:
    return f"PERMANENTLY DELETE {project_id}"


def _iter_store_files(alias: str, source: Path) -> Iterable[tuple[str, Path]]:
    if not _SAFE_ARCHIVE_PART.fullmatch(alias):
        raise TeacherAgentDataRightsError("backup store alias is invalid")
    if not source.exists():
        return
    if source.is_symlink():
        raise TeacherAgentDataRightsError("backup source cannot be a symlink")
    if source.is_file():
        yield f"stores/{alias}/{source.name}", source
        return
    if not source.is_dir():
        raise TeacherAgentDataRightsError("backup source is not a file or directory")
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise TeacherAgentDataRightsError("backup source contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(
            not _SAFE_ARCHIVE_PART.fullmatch(part) or part in {".", ".."}
            for part in relative.parts
        ):
            raise TeacherAgentDataRightsError("backup source path is unsafe")
        yield f"stores/{alias}/{relative.as_posix()}", path


def _backup_plaintext(stores: Mapping[str, str | Path]) -> bytes:
    entries: list[tuple[str, bytes]] = []
    total = 0
    for alias, raw_source in sorted(stores.items()):
        unresolved = Path(raw_source).expanduser()
        # Check the caller-provided leaf before resolution.  Resolving first
        # would erase the evidence that a recoverability-critical secret/store
        # path was itself a symlink.
        if unresolved.is_symlink():
            raise TeacherAgentDataRightsError("backup source cannot be a symlink")
        source = unresolved.resolve(strict=False)
        for name, path in _iter_store_files(alias, source):
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise TeacherAgentDataRightsError(
                    "backup source cannot be read"
                ) from exc
            total += len(content)
            if len(entries) >= _MAX_BACKUP_FILES or total > _MAX_BACKUP_BYTES:
                raise TeacherAgentDataRightsError(
                    "local backup exceeds its safety limit"
                )
            entries.append((name, content))
    manifest_entries = [
        {"path": name, "bytes": len(content), "sha256": sha256(content).hexdigest()}
        for name, content in entries
    ]
    manifest = {
        "schema": LOCAL_BACKUP_MANIFEST_SCHEMA,
        "created_at": _now(),
        "file_count": len(entries),
        "entries": manifest_entries,
    }
    manifest_bytes = _json_document(manifest)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in [*entries, ("backup-manifest.json", manifest_bytes)]:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, content)
    return output.getvalue()


def _derive_backup_key(passphrase: str, salt: bytes) -> bytes:
    if not isinstance(passphrase, str) or len(passphrase) < 12:
        raise TeacherAgentDataRightsError(
            "backup passphrase must contain at least 12 characters"
        )
    try:
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    except ImportError as exc:  # pragma: no cover - packaging test covers dependency.
        raise TeacherAgentDataRightsError(
            "encrypted backups require the cryptography package"
        ) from exc
    return Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(
        passphrase.encode("utf-8")
    )


def create_encrypted_local_backup(
    stores: Mapping[str, str | Path], *, passphrase: str
) -> tuple[bytes, dict[str, Any]]:
    """Create one AES-256-GCM local backup; no cloud transfer is performed."""

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - packaging test covers dependency.
        raise TeacherAgentDataRightsError(
            "encrypted backups require the cryptography package"
        ) from exc
    plaintext = _backup_plaintext(stores)
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(12)
    header = {
        "schema": LOCAL_BACKUP_SCHEMA,
        "created_at": _now(),
        "cipher": "AES-256-GCM",
        "kdf": "scrypt-n32768-r8-p1",
        "salt_b64": b64encode(salt).decode("ascii"),
        "nonce_b64": b64encode(nonce).decode("ascii"),
        "plaintext_bytes": len(plaintext),
        "plaintext_sha256": sha256(plaintext).hexdigest(),
        "local_only": True,
    }
    header_bytes = canonical_json_bytes(header)
    ciphertext = AESGCM(_derive_backup_key(passphrase, salt)).encrypt(
        nonce, plaintext, header_bytes
    )
    payload = (
        _BACKUP_MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + ciphertext
    )
    receipt = {
        "schema": LOCAL_BACKUP_SCHEMA,
        "created_at": header["created_at"],
        "backup_sha256": sha256(payload).hexdigest(),
        "backup_bytes": len(payload),
        "cipher": header["cipher"],
        "kdf": header["kdf"],
        "local_only": True,
        "cloud_backup_claimed": False,
    }
    return payload, receipt


def _decrypt_local_backup(
    payload: bytes, *, passphrase: str
) -> tuple[bytes, dict[str, Any]]:
    try:
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover
        raise TeacherAgentDataRightsError(
            "encrypted backups require the cryptography package"
        ) from exc
    if (
        not payload.startswith(_BACKUP_MAGIC)
        or len(payload) < len(_BACKUP_MAGIC) + 4
        or len(payload) > _MAX_BACKUP_BYTES + 1_000_000
    ):
        raise TeacherAgentDataRightsError("local backup header is invalid")
    offset = len(_BACKUP_MAGIC)
    header_length = struct.unpack(">I", payload[offset : offset + 4])[0]
    offset += 4
    if not 1 <= header_length <= 16_384 or len(payload) <= offset + header_length:
        raise TeacherAgentDataRightsError("local backup header size is invalid")
    header_bytes = payload[offset : offset + header_length]
    ciphertext = payload[offset + header_length :]
    try:
        header = json.loads(header_bytes)
        if (
            not isinstance(header, dict)
            or set(header)
            != {
                "schema",
                "created_at",
                "cipher",
                "kdf",
                "salt_b64",
                "nonce_b64",
                "plaintext_bytes",
                "plaintext_sha256",
                "local_only",
            }
            or header.get("schema") != LOCAL_BACKUP_SCHEMA
            or canonical_json_bytes(header) != header_bytes
            or header.get("cipher") != "AES-256-GCM"
            or header.get("kdf") != "scrypt-n32768-r8-p1"
            or header.get("local_only") is not True
            or not isinstance(header.get("created_at"), str)
            or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                str(header.get("created_at", "")),
            )
            is None
            or isinstance(header.get("plaintext_bytes"), bool)
            or not isinstance(header.get("plaintext_bytes"), int)
            or not 0 <= int(header["plaintext_bytes"]) <= _MAX_BACKUP_BYTES
            or re.fullmatch(r"[0-9a-f]{64}", str(header.get("plaintext_sha256", "")))
            is None
        ):
            raise TeacherAgentDataRightsError("local backup contract is invalid")
        salt = b64decode(str(header["salt_b64"]), validate=True)
        nonce = b64decode(str(header["nonce_b64"]), validate=True)
        if len(salt) != 16 or len(nonce) != 12:
            raise TeacherAgentDataRightsError(
                "local backup crypto parameters are invalid"
            )
        plaintext = AESGCM(_derive_backup_key(passphrase, salt)).decrypt(
            nonce, ciphertext, header_bytes
        )
    except (InvalidTag, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, TeacherAgentDataRightsError):
            raise
        raise TeacherAgentDataRightsError("local backup authentication failed") from exc
    if (
        header.get("plaintext_bytes") != len(plaintext)
        or header.get("plaintext_sha256") != sha256(plaintext).hexdigest()
    ):
        raise TeacherAgentDataRightsError("local backup plaintext integrity failed")
    return plaintext, header


def restore_encrypted_local_backup(
    payload: bytes,
    *,
    passphrase: str,
    destination: str | Path,
) -> dict[str, Any]:
    """Restore into a new/empty directory and verify every file after writing."""

    plaintext, header = _decrypt_local_backup(payload, passphrase=passphrase)
    target = Path(destination).expanduser().resolve(strict=False)
    if target.exists() and (
        target.is_symlink() or not target.is_dir() or any(target.iterdir())
    ):
        raise TeacherAgentDataRightsError(
            "restore drill destination must be a new or empty directory"
        )
    ensure_private_directory(target)
    staging = Path(tempfile.mkdtemp(prefix=".restore-", dir=target))
    restored: list[tuple[Path, str]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(plaintext)) as archive:
            names = archive.namelist()
            if (
                len(names) != len(set(names))
                or len(names) > _MAX_BACKUP_FILES + 1
                or "backup-manifest.json" not in names
                or archive.getinfo("backup-manifest.json").file_size > 16_000_000
            ):
                raise TeacherAgentDataRightsError("backup ZIP inventory is invalid")
            manifest_raw = archive.read("backup-manifest.json")
            manifest = json.loads(manifest_raw)
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema") != LOCAL_BACKUP_MANIFEST_SCHEMA
                or manifest_raw != _json_document(manifest)
                or not isinstance(manifest.get("entries"), list)
                or manifest.get("file_count") != len(manifest["entries"])
            ):
                raise TeacherAgentDataRightsError("backup manifest is invalid")
            if len(manifest["entries"]) > _MAX_BACKUP_FILES:
                raise TeacherAgentDataRightsError("backup file count exceeds its limit")
            declared = {"backup-manifest.json"}
            cumulative_bytes = 0
            for entry in manifest["entries"]:
                if not isinstance(entry, Mapping) or not isinstance(
                    entry.get("path"), str
                ):
                    raise TeacherAgentDataRightsError("backup entry is invalid")
                name = str(entry["path"])
                parts = PurePosixPath(name).parts
                if (
                    name in declared
                    or not parts
                    or PurePosixPath(name).is_absolute()
                    or ".." in parts
                    or any(not _SAFE_ARCHIVE_PART.fullmatch(part) for part in parts)
                ):
                    raise TeacherAgentDataRightsError("backup entry path is unsafe")
                declared_bytes = entry.get("bytes")
                if (
                    isinstance(declared_bytes, bool)
                    or not isinstance(declared_bytes, int)
                    or not 0 <= declared_bytes <= _MAX_BACKUP_BYTES
                ):
                    raise TeacherAgentDataRightsError("backup entry size is invalid")
                cumulative_bytes += declared_bytes
                if cumulative_bytes > _MAX_BACKUP_BYTES:
                    raise TeacherAgentDataRightsError(
                        "backup expanded size exceeds its limit"
                    )
                info = archive.getinfo(name)
                if info.is_dir() or info.file_size != declared_bytes:
                    raise TeacherAgentDataRightsError(
                        "backup ZIP entry size is invalid"
                    )
                content = archive.read(name)
                if (
                    entry.get("bytes") != len(content)
                    or entry.get("sha256") != sha256(content).hexdigest()
                ):
                    raise TeacherAgentDataRightsError("backup entry hash failed")
                output = staging.joinpath(*parts)
                output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                descriptor = os.open(
                    output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                restored.append((output, str(entry["sha256"])))
                declared.add(name)
            if set(names) != declared:
                raise TeacherAgentDataRightsError("backup contains undeclared files")
        for path, expected_hash in restored:
            if sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise TeacherAgentDataRightsError("restored file verification failed")
        final_root = target / "restored"
        staging.replace(final_root)
        write_json(
            target / "restore-drill-receipt.json",
            {
                "schema": RESTORE_DRILL_SCHEMA,
                "verified_at": _now(),
                "backup_created_at": header["created_at"],
                "backup_sha256": sha256(payload).hexdigest(),
                "restored_file_count": len(restored),
                "all_hashes_verified": True,
                "live_store_overwritten": False,
            },
        )
        return {
            "schema": RESTORE_DRILL_SCHEMA,
            "destination": str(final_root),
            "restored_file_count": len(restored),
            "all_hashes_verified": True,
            "live_store_overwritten": False,
            "backup_sha256": sha256(payload).hexdigest(),
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_encrypted_local_backup(
    output: str | Path,
    stores: Mapping[str, str | Path],
    *,
    passphrase: str,
) -> dict[str, Any]:
    payload, receipt = create_encrypted_local_backup(stores, passphrase=passphrase)
    target = Path(output).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        ensure_private_file(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {**receipt, "output": str(target)}


__all__ = [
    "DELETION_RECEIPT_SCHEMA",
    "LOCAL_BACKUP_SCHEMA",
    "PROJECT_EXPORT_SCHEMA",
    "ProjectExportArchive",
    "TeacherAgentDataRightsError",
    "build_content_free_deletion_receipt",
    "build_project_export_archive",
    "canonical_json_bytes",
    "create_encrypted_local_backup",
    "deletion_confirmation",
    "restore_encrypted_local_backup",
    "validate_project_export_archive",
    "write_encrypted_local_backup",
]
