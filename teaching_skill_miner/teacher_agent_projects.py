"""Private, durable learning projects for the local Teaching Agent Console.

Projects are organizational context, not learner evidence.  They bind syllabi,
teacher resources, Chat transcripts, Teach sessions, and teacher notes without
allowing any of those artifacts to silently raise mastery or become scoring
gold.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import threading
from typing import Any, Mapping

try:  # pragma: no cover - exercised on the POSIX production path.
    import fcntl
except ImportError:  # pragma: no cover - Windows remains process-local.
    fcntl = None  # type: ignore[assignment]

from .io_utils import ensure_private_directory, read_json, write_json


LEARNING_PROJECT_SCHEMA = "teaching_skill_miner.learning_project.v1"
PROJECT_ID_PATTERN = re.compile(r"^project_[0-9a-f]{24}$")
CHAT_THREAD_ID_PATTERN = re.compile(r"^chat_[0-9a-f]{24}$")
NOTE_ID_PATTERN = re.compile(r"^note_[0-9a-f]{24}$")
_RESTORE_TOKEN = re.compile(r"^restore_(project_[0-9a-f]{24})_([0-9a-f]{12})$")
_SAFE_REF = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_PROJECT_STATUSES = frozenset({"active", "archived"})
_MESSAGE_ROLES = frozenset({"user", "assistant", "tool"})
_MESSAGE_STATUSES = frozenset({"queued", "running", "stopped", "failed", "completed"})
_MAX_PROJECT_BYTES = 64 * 1024 * 1024
_MAX_THREADS = 1_000
_MAX_MESSAGES_PER_THREAD = 10_000
_MAX_NOTES = 1_000
_MAX_REFS = 5_000
_MAX_OPERATION_RECEIPTS = 8_192
_OPERATION_SCHEMA = "teaching_skill_miner.learning_project_operations.v1"
_OPERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}$")


class LearningProjectError(ValueError):
    """Raised when a learning-project document or operation is invalid."""


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _next_timestamp(previous: str) -> str:
    """Mint a strict CAS revision while retaining the public v1 schema."""

    prior = datetime.strptime(previous, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    current = datetime.now(timezone.utc).replace(microsecond=0)
    return max(current, prior + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _text(
    value: Any,
    field: str,
    *,
    maximum: int,
    minimum: int = 1,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise LearningProjectError(f"{field} must be a trimmed string")
    if allow_empty and value == "":
        return value
    if not minimum <= len(value) <= maximum:
        raise LearningProjectError(
            f"{field} must have length in [{minimum}, {maximum}]"
        )
    return value


def _timestamp(value: Any, field: str) -> str:
    text = _text(value, field, maximum=30)
    if _UTC_TIMESTAMP.fullmatch(text) is None:
        raise LearningProjectError(f"{field} must be a UTC timestamp")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise LearningProjectError(f"{field} must be a real UTC timestamp") from exc
    if parsed.year < 2000 or parsed.year > 2200:
        raise LearningProjectError(f"{field} is outside the supported range")
    return text


def _identifier(value: Any, field: str, pattern: re.Pattern[str]) -> str:
    text = _text(value, field, maximum=180)
    if pattern.fullmatch(text) is None:
        raise LearningProjectError(f"{field} is invalid")
    return text


def _reference_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or len(value) > _MAX_REFS:
        raise LearningProjectError(f"{field} must be a bounded array")
    result: list[str] = []
    for index, item in enumerate(value):
        ref = _text(item, f"{field}[{index}]", maximum=160)
        if _SAFE_REF.fullmatch(ref) is None:
            raise LearningProjectError(f"{field}[{index}] is invalid")
        result.append(ref)
    if len(result) != len(set(result)):
        raise LearningProjectError(f"{field} must not contain duplicates")
    return result


def _sources(value: Any, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) > 24:
        raise LearningProjectError(f"{field} must be a bounded array")
    result: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"title", "url"}:
            raise LearningProjectError(f"{field}[{index}] is invalid")
        title = _text(item["title"], f"{field}[{index}].title", maximum=300)
        url = _text(item["url"], f"{field}[{index}].url", maximum=2_000)
        if not url.startswith(("https://", "http://")):
            raise LearningProjectError(f"{field}[{index}].url is invalid")
        result.append({"title": title, "url": url})
    return result


def _message(value: Any, field: str) -> dict[str, Any]:
    expected = {
        "message_id",
        "role",
        "content",
        "status",
        "created_at",
        "web_search_used",
        "sources",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LearningProjectError(f"{field} has invalid fields")
    message_id = _text(value["message_id"], f"{field}.message_id", maximum=180)
    if _SAFE_REF.fullmatch(message_id) is None:
        raise LearningProjectError(f"{field}.message_id is invalid")
    role = _text(value["role"], f"{field}.role", maximum=20)
    status = _text(value["status"], f"{field}.status", maximum=20)
    if role not in _MESSAGE_ROLES or status not in _MESSAGE_STATUSES:
        raise LearningProjectError(f"{field} role or status is invalid")
    content = _text(
        value["content"], f"{field}.content", maximum=64_000, allow_empty=True
    )
    web_search_used = value["web_search_used"]
    if not isinstance(web_search_used, bool):
        raise LearningProjectError(f"{field}.web_search_used must be boolean")
    return {
        "message_id": message_id,
        "role": role,
        "content": content,
        "status": status,
        "created_at": _timestamp(value["created_at"], f"{field}.created_at"),
        "web_search_used": web_search_used,
        "sources": _sources(value["sources"], f"{field}.sources"),
    }


def _chat_thread(value: Any, field: str) -> dict[str, Any]:
    expected = {"thread_id", "title", "created_at", "updated_at", "messages"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LearningProjectError(f"{field} has invalid fields")
    messages = value["messages"]
    if not isinstance(messages, list) or len(messages) > _MAX_MESSAGES_PER_THREAD:
        raise LearningProjectError(f"{field}.messages must be a bounded array")
    normalized = [
        _message(message, f"{field}.messages[{index}]")
        for index, message in enumerate(messages)
    ]
    message_ids = [message["message_id"] for message in normalized]
    if len(message_ids) != len(set(message_ids)):
        raise LearningProjectError(f"{field}.messages contains duplicate ids")
    created_at = _timestamp(value["created_at"], f"{field}.created_at")
    updated_at = _timestamp(value["updated_at"], f"{field}.updated_at")
    if updated_at < created_at:
        raise LearningProjectError(f"{field}.updated_at predates created_at")
    return {
        "thread_id": _identifier(
            value["thread_id"], f"{field}.thread_id", CHAT_THREAD_ID_PATTERN
        ),
        "title": _text(value["title"], f"{field}.title", maximum=160),
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": normalized,
    }


def _note(value: Any, field: str) -> dict[str, str]:
    expected = {"note_id", "title", "body", "created_at", "updated_at"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LearningProjectError(f"{field} has invalid fields")
    created_at = _timestamp(value["created_at"], f"{field}.created_at")
    updated_at = _timestamp(value["updated_at"], f"{field}.updated_at")
    if updated_at < created_at:
        raise LearningProjectError(f"{field}.updated_at predates created_at")
    return {
        "note_id": _identifier(value["note_id"], f"{field}.note_id", NOTE_ID_PATTERN),
        "title": _text(value["title"], f"{field}.title", maximum=160),
        "body": _text(value["body"], f"{field}.body", maximum=64_000, allow_empty=True),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def validate_learning_project(value: Any) -> dict[str, Any]:
    expected = {
        "schema",
        "project_id",
        "title",
        "description",
        "status",
        "pinned",
        "created_at",
        "updated_at",
        "syllabus_ids",
        "teaching_session_ids",
        "resource_ids",
        "chat_threads",
        "notes",
        "claim_boundary",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LearningProjectError("learning project has invalid fields")
    if value["schema"] != LEARNING_PROJECT_SCHEMA:
        raise LearningProjectError("learning project schema is unsupported")
    status = _text(value["status"], "status", maximum=20)
    if status not in _PROJECT_STATUSES or not isinstance(value["pinned"], bool):
        raise LearningProjectError("learning project status or pinned flag is invalid")
    created_at = _timestamp(value["created_at"], "created_at")
    updated_at = _timestamp(value["updated_at"], "updated_at")
    if updated_at < created_at:
        raise LearningProjectError("updated_at predates created_at")
    raw_threads = value["chat_threads"]
    raw_notes = value["notes"]
    if not isinstance(raw_threads, list) or len(raw_threads) > _MAX_THREADS:
        raise LearningProjectError("chat_threads must be a bounded array")
    if not isinstance(raw_notes, list) or len(raw_notes) > _MAX_NOTES:
        raise LearningProjectError("notes must be a bounded array")
    threads = [
        _chat_thread(thread, f"chat_threads[{index}]")
        for index, thread in enumerate(raw_threads)
    ]
    notes = [_note(note, f"notes[{index}]") for index, note in enumerate(raw_notes)]
    if len({item["thread_id"] for item in threads}) != len(threads):
        raise LearningProjectError("chat_threads contains duplicate ids")
    if len({item["note_id"] for item in notes}) != len(notes):
        raise LearningProjectError("notes contains duplicate ids")
    boundary = value["claim_boundary"]
    if boundary != {
        "project_context_is_learner_evidence": False,
        "project_resources_are_scoring_gold": False,
        "learner_evidence_required_for_mastery": True,
    }:
        raise LearningProjectError("learning project claim_boundary is invalid")
    normalized = {
        "schema": LEARNING_PROJECT_SCHEMA,
        "project_id": _identifier(
            value["project_id"], "project_id", PROJECT_ID_PATTERN
        ),
        "title": _text(value["title"], "title", maximum=160),
        "description": _text(
            value["description"], "description", maximum=2_000, allow_empty=True
        ),
        "status": status,
        "pinned": value["pinned"],
        "created_at": created_at,
        "updated_at": updated_at,
        "syllabus_ids": _reference_list(value["syllabus_ids"], "syllabus_ids"),
        "teaching_session_ids": _reference_list(
            value["teaching_session_ids"], "teaching_session_ids"
        ),
        "resource_ids": _reference_list(value["resource_ids"], "resource_ids"),
        "chat_threads": threads,
        "notes": notes,
        "claim_boundary": deepcopy(dict(boundary)),
    }
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_PROJECT_BYTES:
        raise LearningProjectError("learning project exceeds the storage safety limit")
    return normalized


def new_learning_project(
    *, title: str, description: str = "", project_id: str | None = None
) -> dict[str, Any]:
    now = _now()
    project = {
        "schema": LEARNING_PROJECT_SCHEMA,
        "project_id": project_id or "project_" + secrets.token_hex(12),
        "title": str(title),
        "description": str(description),
        "status": "active",
        "pinned": False,
        "created_at": now,
        "updated_at": now,
        "syllabus_ids": [],
        "teaching_session_ids": [],
        "resource_ids": [],
        "chat_threads": [],
        "notes": [],
        "claim_boundary": {
            "project_context_is_learner_evidence": False,
            "project_resources_are_scoring_gold": False,
            "learner_evidence_required_for_mastery": True,
        },
    }
    return validate_learning_project(project)


class LearningProjectStore:
    """Atomic per-project JSON persistence with recoverable archival."""

    def __init__(self, root: str | Path) -> None:
        self.root = ensure_private_directory(Path(root).expanduser().resolve())
        self._trash = ensure_private_directory(self.root / ".trash")
        self._process_lock_path = self.root / ".projects.lock"
        self._operations_path = self.root / ".operations.json"
        self._lock = threading.RLock()
        self._lock_state = threading.local()

    @contextmanager
    def _locked(self):
        """Serialize the file set across threads and POSIX processes."""

        with self._lock:
            depth = int(getattr(self._lock_state, "depth", 0))
            if depth:
                self._lock_state.depth = depth + 1
                try:
                    yield
                finally:
                    self._lock_state.depth -= 1
                return
            descriptor = os.open(
                self._process_lock_path,
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
            self._lock_state.depth = 1
            try:
                os.fchmod(descriptor, 0o600)
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                self._lock_state.depth = 0
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _read_operations(self) -> dict[str, Any]:
        if not self._operations_path.exists():
            return {"schema": _OPERATION_SCHEMA, "receipts": {}}
        value = read_json(self._operations_path)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"schema", "receipts"}
            or value.get("schema") != _OPERATION_SCHEMA
            or not isinstance(value.get("receipts"), Mapping)
            or len(value["receipts"]) > _MAX_OPERATION_RECEIPTS
        ):
            raise LearningProjectError("project operation receipts are invalid")
        receipts: dict[str, dict[str, str]] = {}
        for key, receipt in value["receipts"].items():
            if (
                not isinstance(key, str)
                or _OPERATION_ID_PATTERN.fullmatch(key) is None
                or not isinstance(receipt, Mapping)
                or set(receipt)
                != {"request_sha256", "project_id", "operation", "committed_at"}
            ):
                raise LearningProjectError("project operation receipt is invalid")
            request_sha256 = _text(
                receipt["request_sha256"], "request_sha256", maximum=64
            )
            if re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None:
                raise LearningProjectError("project operation receipt hash is invalid")
            receipts[key] = {
                "request_sha256": request_sha256,
                "project_id": _identifier(
                    receipt["project_id"], "project_id", PROJECT_ID_PATTERN
                ),
                "operation": _text(receipt["operation"], "operation", maximum=80),
                "committed_at": _timestamp(receipt["committed_at"], "committed_at"),
            }
        return {"schema": _OPERATION_SCHEMA, "receipts": receipts}

    def _operation_replay(
        self,
        operation_id: str,
        *,
        operation: str,
        request: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
        key = _text(operation_id, "operation_id", maximum=160)
        if _OPERATION_ID_PATTERN.fullmatch(key) is None:
            raise LearningProjectError("operation_id is invalid")
        request_sha256 = _canonical_sha256(
            {"operation": operation, "request": dict(request)}
        )
        operations = self._read_operations()
        receipt = operations["receipts"].get(key)
        if receipt is not None:
            if (
                receipt["operation"] != operation
                or receipt["request_sha256"] != request_sha256
            ):
                raise LearningProjectError(
                    "operation_id was already used for a different request"
                )
            return operations, request_sha256, dict(receipt)
        return operations, request_sha256, None

    def _commit_operation(
        self,
        operations: dict[str, Any],
        *,
        operation_id: str,
        operation: str,
        request_sha256: str,
        project_id: str,
    ) -> None:
        receipts = operations["receipts"]
        if len(receipts) >= _MAX_OPERATION_RECEIPTS:
            raise LearningProjectError("project operation receipt capacity is full")
        receipts[operation_id] = {
            "request_sha256": request_sha256,
            "project_id": project_id,
            "operation": operation,
            "committed_at": _now(),
        }
        write_json(self._operations_path, operations)

    @staticmethod
    def _require_revision(
        project: Mapping[str, Any], expected_updated_at: str | None
    ) -> None:
        if expected_updated_at is None:
            return
        expected = _timestamp(expected_updated_at, "expected_updated_at")
        if project.get("updated_at") != expected:
            raise LearningProjectError("project revision conflict")

    def _path(self, project_id: str) -> Path:
        identifier = _identifier(project_id, "project_id", PROJECT_ID_PATTERN)
        return self.root / f"{identifier}.json"

    def create(self, *, title: str, description: str = "") -> dict[str, Any]:
        project = new_learning_project(title=title, description=description)
        path = self._path(project["project_id"])
        with self._locked():
            if path.exists():  # cryptographically improbable, still fail closed
                raise LearningProjectError("generated project_id already exists")
            write_json(path, project)
        return deepcopy(project)

    def create_idempotent(
        self, *, operation_id: str, title: str, description: str = ""
    ) -> dict[str, Any]:
        request = {"title": title, "description": description}
        with self._locked():
            operations, request_sha256, replay = self._operation_replay(
                operation_id,
                operation="create",
                request=request,
            )
            if replay is not None:
                return self.read(replay["project_id"])
            project_id = (
                "project_"
                + hashlib.sha256(f"create:{operation_id}".encode("utf-8")).hexdigest()[
                    :24
                ]
            )
            path = self._path(project_id)
            if path.exists():
                project = self.read(project_id)
                if project["title"] != title or project["description"] != description:
                    raise LearningProjectError("project operation identity conflicts")
            else:
                project = new_learning_project(
                    title=title, description=description, project_id=project_id
                )
                write_json(path, project)
            self._commit_operation(
                operations,
                operation_id=operation_id,
                operation="create",
                request_sha256=request_sha256,
                project_id=project_id,
            )
            return deepcopy(project)

    def create_default(
        self,
        *,
        idempotency_key: str,
        title: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Create or replay one deterministic local default workspace."""

        request = {"title": title, "description": description}
        operation_id = f"default:{idempotency_key}"
        with self._locked():
            operations, request_sha256, replay = self._operation_replay(
                operation_id,
                operation="create_default",
                request=request,
            )
            project_id = (
                "project_"
                + hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:24]
            )
            if replay is not None:
                return self.read(replay["project_id"])
            path = self._path(project_id)
            if path.exists():
                project = self.read(project_id)
                if project["title"] != title or project["description"] != description:
                    raise LearningProjectError(
                        "default project identity conflicts with stored content"
                    )
            else:
                project = new_learning_project(
                    title=title,
                    description=description,
                    project_id=project_id,
                )
                write_json(path, project)
            self._commit_operation(
                operations,
                operation_id=operation_id,
                operation="create_default",
                request_sha256=request_sha256,
                project_id=project_id,
            )
            return deepcopy(project)

    def read(self, project_id: str) -> dict[str, Any]:
        path = self._path(project_id)
        with self._locked():
            try:
                value = read_json(path)
            except FileNotFoundError as exc:
                raise LearningProjectError("project_id was not found") from exc
        return deepcopy(validate_learning_project(value))

    def save(self, project: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_learning_project(project)
        path = self._path(normalized["project_id"])
        with self._locked():
            if not path.exists():
                raise LearningProjectError("project_id was not found")
            write_json(path, normalized)
        return deepcopy(normalized)

    def list(self, *, include_archived: bool = True) -> list[dict[str, Any]]:
        with self._locked():
            paths = sorted(self.root.glob("project_*.json"))
            values = []
            for path in paths:
                try:
                    values.append(read_json(path))
                except (OSError, json.JSONDecodeError) as exc:
                    raise LearningProjectError(
                        f"stored project {path.name} cannot be read"
                    ) from exc
        projects = [validate_learning_project(value) for value in values]
        if not include_archived:
            projects = [item for item in projects if item["status"] == "active"]
        projects.sort(
            key=lambda item: (item["pinned"], item["updated_at"], item["project_id"]),
            reverse=True,
        )
        return [
            {
                "project_id": item["project_id"],
                "title": item["title"],
                "description": item["description"],
                "status": item["status"],
                "pinned": item["pinned"],
                "updated_at": item["updated_at"],
                "syllabus_count": len(item["syllabus_ids"]),
                "teaching_session_count": len(item["teaching_session_ids"]),
                "resource_count": len(item["resource_ids"]),
                "chat_thread_count": len(item["chat_threads"]),
                "note_count": len(item["notes"]),
            }
            for item in projects
        ]

    def update_metadata(
        self,
        project_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        pinned: bool | None = None,
        expected_updated_at: str | None = None,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        with self._locked():
            request = {
                "project_id": project_id,
                "title": title,
                "description": description,
                "status": status,
                "pinned": pinned,
            }
            operations = None
            request_sha256 = None
            if operation_id is not None:
                operations, request_sha256, replay = self._operation_replay(
                    operation_id,
                    operation="update_metadata",
                    request=request,
                )
                if replay is not None:
                    return self.read(replay["project_id"])
            project = self.read(project_id)
            desired_already_applied = all(
                value is None or project[field] == value
                for field, value in {
                    "title": title,
                    "description": description,
                    "status": status,
                    "pinned": pinned,
                }.items()
            )
            if operation_id is not None and desired_already_applied:
                assert operations is not None and request_sha256 is not None
                self._commit_operation(
                    operations,
                    operation_id=operation_id,
                    operation="update_metadata",
                    request_sha256=request_sha256,
                    project_id=project_id,
                )
                return project
            self._require_revision(project, expected_updated_at)
            if title is not None:
                project["title"] = title
            if description is not None:
                project["description"] = description
            if status is not None:
                project["status"] = status
            if pinned is not None:
                project["pinned"] = pinned
            project["updated_at"] = _next_timestamp(project["updated_at"])
            project = self.save(project)
            if operation_id is not None:
                assert operations is not None and request_sha256 is not None
                self._commit_operation(
                    operations,
                    operation_id=operation_id,
                    operation="update_metadata",
                    request_sha256=request_sha256,
                    project_id=project_id,
                )
            return project

    def add_reference(
        self,
        project_id: str,
        *,
        kind: str,
        reference_id: str,
        expected_updated_at: str | None = None,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        fields = {
            "syllabus": "syllabus_ids",
            "teaching_session": "teaching_session_ids",
            "resource": "resource_ids",
        }
        if kind not in fields:
            raise LearningProjectError("project reference kind is unsupported")
        with self._locked():
            request = {
                "project_id": project_id,
                "kind": kind,
                "reference_id": reference_id,
            }
            operations = None
            request_sha256 = None
            if operation_id is not None:
                operations, request_sha256, replay = self._operation_replay(
                    operation_id,
                    operation="add_reference",
                    request=request,
                )
                if replay is not None:
                    return self.read(replay["project_id"])
            project = self.read(project_id)
            field = fields[kind]
            reference = _text(reference_id, "reference_id", maximum=160)
            if _SAFE_REF.fullmatch(reference) is None:
                raise LearningProjectError("reference_id is invalid")
            if reference in project[field]:
                if operation_id is not None:
                    assert operations is not None and request_sha256 is not None
                    self._commit_operation(
                        operations,
                        operation_id=operation_id,
                        operation="add_reference",
                        request_sha256=request_sha256,
                        project_id=project_id,
                    )
                return project
            self._require_revision(project, expected_updated_at)
            project[field].append(reference)
            project["updated_at"] = _next_timestamp(project["updated_at"])
            project = self.save(project)
            if operation_id is not None:
                assert operations is not None and request_sha256 is not None
                self._commit_operation(
                    operations,
                    operation_id=operation_id,
                    operation="add_reference",
                    request_sha256=request_sha256,
                    project_id=project_id,
                )
            return project

    def remove_reference(
        self,
        project_id: str,
        *,
        kind: str,
        reference_id: str,
        expected_updated_at: str,
        operation_id: str,
    ) -> dict[str, Any]:
        fields = {
            "syllabus": "syllabus_ids",
            "teaching_session": "teaching_session_ids",
            "resource": "resource_ids",
        }
        if kind not in fields:
            raise LearningProjectError("project reference kind is unsupported")
        request = {
            "project_id": project_id,
            "kind": kind,
            "reference_id": reference_id,
        }
        with self._locked():
            operations, request_sha256, replay = self._operation_replay(
                operation_id,
                operation="remove_reference",
                request=request,
            )
            if replay is not None:
                return self.read(replay["project_id"])
            project = self.read(project_id)
            reference = _text(reference_id, "reference_id", maximum=160)
            field = fields[kind]
            if reference not in project[field]:
                self._commit_operation(
                    operations,
                    operation_id=operation_id,
                    operation="remove_reference",
                    request_sha256=request_sha256,
                    project_id=project_id,
                )
                return project
            self._require_revision(project, expected_updated_at)
            project[field].remove(reference)
            project["updated_at"] = _next_timestamp(project["updated_at"])
            project = self.save(project)
            self._commit_operation(
                operations,
                operation_id=operation_id,
                operation="remove_reference",
                request_sha256=request_sha256,
                project_id=project_id,
            )
            return project

    def _commit_chat_thread(
        self, project_id: str, thread: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Commit a server-authoritative Chat thread.

        This is deliberately a private method.  Dashboard model execution is
        the only path allowed to append assistant/tool records and sources;
        the public HTTP projection is handled by ``upsert_client_chat_thread``
        below and cannot select this capability with request data.
        """

        with self._locked():
            project = self.read(project_id)
            normalized = _chat_thread(thread, "chat_thread")
            matches = [
                index
                for index, item in enumerate(project["chat_threads"])
                if item["thread_id"] == normalized["thread_id"]
            ]
            if matches:
                previous = project["chat_threads"][matches[0]]
                if len(normalized["messages"]) < len(previous["messages"]):
                    raise LearningProjectError(
                        "chat thread update cannot truncate durable messages"
                    )
                for index, durable_message in enumerate(previous["messages"]):
                    submitted_message = normalized["messages"][index]
                    if (
                        submitted_message["role"] != durable_message["role"]
                        or submitted_message["content"] != durable_message["content"]
                    ):
                        raise LearningProjectError(
                            "chat thread update conflicts with durable history"
                        )
                    # Once a message is durable, its receipt, sources, status,
                    # identifier, and timestamp remain server-authoritative.
                    # The browser may append newer messages but cannot rewrite
                    # an older turn with a stale local projection.
                    normalized["messages"][index] = deepcopy(durable_message)
                # The server may have created an authoritative request record
                # just before a browser saved its richer display projection.
                # Preserve the first durable timestamp instead of rejecting
                # that idempotent reconciliation.
                normalized["created_at"] = previous["created_at"]
                project["chat_threads"][matches[0]] = normalized
            else:
                if len(project["chat_threads"]) >= _MAX_THREADS:
                    raise LearningProjectError("chat thread limit reached")
                project["chat_threads"].append(normalized)
            project["chat_threads"].sort(
                key=lambda item: item["updated_at"], reverse=True
            )
            project["updated_at"] = max(
                _next_timestamp(project["updated_at"]), normalized["updated_at"]
            )
            return self.save(project)

    def upsert_client_chat_thread(
        self, project_id: str, thread: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Reconcile one browser-authored learner message, fail closed.

        Existing records remain fully server-authoritative.  A client may
        create/reconcile a thread and append at most one learner message after
        an assistant turn; it may never append assistant/tool content, search
        claims, or sources.  Server IDs, timestamps, and status are minted
        here rather than trusted from the submitted projection.
        """

        with self._locked():
            project = self.read(project_id)
            normalized = _chat_thread(thread, "chat_thread")
            previous = next(
                (
                    item
                    for item in project["chat_threads"]
                    if item["thread_id"] == normalized["thread_id"]
                ),
                None,
            )
            durable_messages = previous["messages"] if previous is not None else []
            if len(normalized["messages"]) < len(durable_messages):
                raise LearningProjectError(
                    "client chat thread update cannot truncate durable messages"
                )
            for index, durable_message in enumerate(durable_messages):
                submitted_message = normalized["messages"][index]
                if (
                    submitted_message["role"] != durable_message["role"]
                    or submitted_message["content"] != durable_message["content"]
                ):
                    raise LearningProjectError(
                        "client chat thread update conflicts with durable history"
                    )
                normalized["messages"][index] = deepcopy(durable_message)

            appended = normalized["messages"][len(durable_messages) :]
            if len(appended) > 1:
                raise LearningProjectError(
                    "client chat thread may append only one learner message"
                )
            if appended:
                message = appended[0]
                if message["role"] != "user":
                    raise LearningProjectError(
                        "client chat thread cannot append assistant or tool messages"
                    )
                if durable_messages and durable_messages[-1]["role"] != "assistant":
                    raise LearningProjectError(
                        "client chat thread cannot append after an unpaired learner message"
                    )
                if message["web_search_used"] or message["sources"]:
                    raise LearningProjectError(
                        "client learner messages cannot claim search results or sources"
                    )
                now = _now()
                digest = secrets.token_hex(12)
                normalized["messages"][-1] = {
                    "message_id": f"server_{digest}",
                    "role": "user",
                    "content": message["content"],
                    "status": "completed",
                    "created_at": now,
                    "web_search_used": False,
                    "sources": [],
                }
                normalized["updated_at"] = max(normalized["updated_at"], now)
            if previous is not None:
                normalized["created_at"] = previous["created_at"]
            return self._commit_chat_thread(project_id, normalized)

    def upsert_note(self, project_id: str, note: Mapping[str, Any]) -> dict[str, Any]:
        with self._locked():
            project = self.read(project_id)
            normalized = _note(note, "note")
            matches = [
                index
                for index, item in enumerate(project["notes"])
                if item["note_id"] == normalized["note_id"]
            ]
            if matches:
                if (
                    normalized["created_at"]
                    != project["notes"][matches[0]]["created_at"]
                ):
                    raise LearningProjectError("note created_at is immutable")
                project["notes"][matches[0]] = normalized
            else:
                if len(project["notes"]) >= _MAX_NOTES:
                    raise LearningProjectError("note limit reached")
                project["notes"].append(normalized)
            project["notes"].sort(key=lambda item: item["updated_at"], reverse=True)
            project["updated_at"] = max(
                _next_timestamp(project["updated_at"]), normalized["updated_at"]
            )
            return self.save(project)

    def upsert_note_content(
        self,
        project_id: str,
        *,
        note_id: str | None,
        title: str,
        body: str,
        expected_updated_at: str,
        operation_id: str,
    ) -> dict[str, Any]:
        """Server-mint note identity/timestamps with CAS and durable replay."""

        resolved_note_id = note_id or (
            "note_"
            + hashlib.sha256(
                f"{project_id}:{operation_id}".encode("utf-8")
            ).hexdigest()[:24]
        )
        request = {
            "project_id": project_id,
            "note_id": resolved_note_id,
            "title": title,
            "body": body,
        }
        with self._locked():
            operations, request_sha256, replay = self._operation_replay(
                operation_id,
                operation="upsert_note",
                request=request,
            )
            if replay is not None:
                return self.read(replay["project_id"])
            project = self.read(project_id)
            existing = next(
                (
                    item
                    for item in project["notes"]
                    if item["note_id"] == resolved_note_id
                ),
                None,
            )
            if (
                existing is not None
                and existing["title"] == title
                and existing["body"] == body
            ):
                self._commit_operation(
                    operations,
                    operation_id=operation_id,
                    operation="upsert_note",
                    request_sha256=request_sha256,
                    project_id=project_id,
                )
                return project
            self._require_revision(project, expected_updated_at)
            now = _next_timestamp(project["updated_at"])
            normalized = _note(
                {
                    "note_id": resolved_note_id,
                    "title": title,
                    "body": body,
                    "created_at": existing["created_at"] if existing else now,
                    "updated_at": now,
                },
                "note",
            )
            if existing is None:
                if len(project["notes"]) >= _MAX_NOTES:
                    raise LearningProjectError("note limit reached")
                project["notes"].append(normalized)
            else:
                project["notes"] = [
                    normalized if item["note_id"] == resolved_note_id else item
                    for item in project["notes"]
                ]
            project["notes"].sort(key=lambda item: item["updated_at"], reverse=True)
            project["updated_at"] = now
            project = self.save(project)
            self._commit_operation(
                operations,
                operation_id=operation_id,
                operation="upsert_note",
                request_sha256=request_sha256,
                project_id=project_id,
            )
            return project

    def migrate_legacy_handles(
        self,
        project_id: str,
        *,
        operation_id: str,
        chat_threads: list[Mapping[str, Any]],
        teaching_session_ids: list[str],
    ) -> dict[str, Any]:
        """Import a one-time browser cache without granting evidence authority.

        Assistant text is visibly labelled as unverified legacy cache, all
        source/search claims are stripped, and identifiers/timestamps are
        minted by this store.  The project claim boundary continues to exclude
        this organizational history from mastery evidence.
        """

        if len(chat_threads) > _MAX_THREADS:
            raise LearningProjectError("legacy chat thread list is too large")
        session_ids = _reference_list(teaching_session_ids, "teaching_session_ids")
        request = {
            "project_id": project_id,
            "chat_threads": [dict(item) for item in chat_threads],
            "teaching_session_ids": session_ids,
        }
        with self._locked():
            operations, request_sha256, replay = self._operation_replay(
                operation_id,
                operation="migrate_legacy_handles",
                request=request,
            )
            if replay is not None:
                return self.read(replay["project_id"])
            project = self.read(project_id)
            changed = False
            now = _next_timestamp(project["updated_at"])
            existing_by_id = {
                item["thread_id"]: item for item in project["chat_threads"]
            }
            for thread_index, submitted in enumerate(chat_threads):
                normalized = _chat_thread(submitted, f"chat_threads[{thread_index}]")
                messages: list[dict[str, Any]] = []
                for message_index, message in enumerate(normalized["messages"]):
                    role = message["role"]
                    content = message["content"]
                    if role == "assistant":
                        content = "[旧版本地缓存，未验证为模型原始输出]\n" + content
                    elif role == "tool":
                        content = "[旧版本地工具记录，未验证]\n" + content
                    digest = hashlib.sha256(
                        (
                            f"{operation_id}:{normalized['thread_id']}:"
                            f"{message_index}:{role}:{content}"
                        ).encode("utf-8")
                    ).hexdigest()[:24]
                    messages.append(
                        {
                            "message_id": f"legacy_{digest}",
                            "role": role,
                            "content": content[:64_000],
                            "status": "completed",
                            "created_at": now,
                            "web_search_used": False,
                            "sources": [],
                        }
                    )
                migrated = {
                    "thread_id": normalized["thread_id"],
                    "title": f"旧版迁移 · {normalized['title']}"[:160],
                    "created_at": now,
                    "updated_at": now,
                    "messages": messages,
                }
                existing = existing_by_id.get(migrated["thread_id"])
                if existing is not None:
                    if [
                        (item["role"], item["content"]) for item in existing["messages"]
                    ] != [
                        (item["role"], item["content"]) for item in migrated["messages"]
                    ]:
                        raise LearningProjectError(
                            "legacy chat migration conflicts with durable history"
                        )
                    continue
                if len(project["chat_threads"]) >= _MAX_THREADS:
                    raise LearningProjectError("chat thread limit reached")
                project["chat_threads"].append(migrated)
                changed = True
            for session_id in session_ids:
                if session_id not in project["teaching_session_ids"]:
                    project["teaching_session_ids"].append(session_id)
                    changed = True
            if changed:
                project["chat_threads"].sort(
                    key=lambda item: item["updated_at"], reverse=True
                )
                project["updated_at"] = now
                project = self.save(project)
            self._commit_operation(
                operations,
                operation_id=operation_id,
                operation="migrate_legacy_handles",
                request_sha256=request_sha256,
                project_id=project_id,
            )
            return project

    def browse(
        self,
        project_id: str,
        *,
        section: str,
        query: str = "",
        cursor: str | None = None,
        limit: int = 50,
        thread_id: str | None = None,
        reference_search: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return a stable searchable page over every durable project section."""

        sections = {
            "chat_threads",
            "chat_messages",
            "notes",
            "syllabi",
            "teaching_sessions",
            "resources",
        }
        if section not in sections:
            raise LearningProjectError("project browse section is invalid")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise LearningProjectError("project browse limit must be in [1, 100]")
        search = _text(
            query, "query", maximum=200, minimum=0, allow_empty=True
        ).casefold()
        if reference_search is None:
            normalized_reference_search: dict[str, str] = {}
        else:
            if not isinstance(reference_search, Mapping) or len(reference_search) > _MAX_REFS:
                raise LearningProjectError("project reference search metadata is invalid")
            normalized_reference_search = {}
            for raw_reference_id, raw_search_text in reference_search.items():
                reference_id = _text(
                    raw_reference_id, "reference_search.reference_id", maximum=160
                )
                if _SAFE_REF.fullmatch(reference_id) is None:
                    raise LearningProjectError(
                        "project reference search metadata is invalid"
                    )
                normalized_reference_search[reference_id] = _text(
                    raw_search_text,
                    f"reference_search[{reference_id}]",
                    maximum=20_000,
                    minimum=0,
                    allow_empty=True,
                ).casefold()
        project = self.read(project_id)
        if section == "chat_threads":
            values: list[Any] = [
                {
                    "thread_id": item["thread_id"],
                    "title": item["title"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "message_count": len(item["messages"]),
                    "preview": next(
                        (
                            message["content"][:240]
                            for message in reversed(item["messages"])
                            if message["content"]
                        ),
                        "",
                    ),
                }
                for item in project["chat_threads"]
            ]
        elif section == "chat_messages":
            identifier = _identifier(thread_id, "thread_id", CHAT_THREAD_ID_PATTERN)
            thread = next(
                (
                    item
                    for item in project["chat_threads"]
                    if item["thread_id"] == identifier
                ),
                None,
            )
            if thread is None:
                raise LearningProjectError("chat thread was not found")
            values = list(thread["messages"])
        elif section == "notes":
            values = list(project["notes"])
        else:
            field = {
                "syllabi": "syllabus_ids",
                "teaching_sessions": "teaching_session_ids",
                "resources": "resource_ids",
            }[section]
            values = [{"reference_id": value} for value in project[field]]
        if search:
            values = [
                value
                for value in values
                if search
                in (
                    json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
                    + "\n"
                    + normalized_reference_search.get(
                        str(value.get("reference_id", "")), ""
                    )
                )
            ]
        binding = _canonical_sha256(
            {
                "project_id": project_id,
                "updated_at": project["updated_at"],
                "section": section,
                "query": search,
                "thread_id": thread_id,
                "reference_search_sha256": (
                    _canonical_sha256(normalized_reference_search)
                    if normalized_reference_search
                    else None
                ),
            }
        )[:16]
        offset = 0
        if cursor is not None:
            match = re.fullmatch(r"cursor_(\d{1,8})_([0-9a-f]{16})", cursor)
            if match is None or match.group(2) != binding:
                raise LearningProjectError("project browse cursor is stale or invalid")
            offset = int(match.group(1))
            if offset > len(values):
                raise LearningProjectError("project browse cursor is out of range")
        page = deepcopy(values[offset : offset + limit])
        next_offset = offset + len(page)
        return {
            "schema": "teaching_skill_miner.learning_project_page.v1",
            "project_id": project_id,
            "project_updated_at": project["updated_at"],
            "section": section,
            "query": query,
            "items": page,
            "total": len(values),
            "next_cursor": (
                f"cursor_{next_offset}_{binding}" if next_offset < len(values) else None
            ),
        }

    def trash(self, project_id: str) -> dict[str, str]:
        """Move a project to private trash and return an opaque recovery token."""

        path = self._path(project_id)
        with self._locked():
            if not path.exists():
                raise LearningProjectError("project_id was not found")
            # Validate before moving so corrupt active state cannot be laundered
            # into a recoverable artifact.
            validate_learning_project(read_json(path))
            nonce = secrets.token_hex(6)
            target = self._trash / f"{project_id}.{nonce}.json"
            path.replace(target)
        return {
            "project_id": project_id,
            "recovery_token": f"restore_{project_id}_{nonce}",
            "trashed_at": _now(),
        }

    def restore(self, recovery_token: str) -> dict[str, Any]:
        project_id, source = self._trash_identity(recovery_token)
        target = self._path(project_id)
        with self._locked():
            if target.exists():
                raise LearningProjectError("project_id is already active")
            try:
                value = read_json(source)
            except FileNotFoundError as exc:
                raise LearningProjectError("recovery_token was not found") from exc
            project = validate_learning_project(value)
            if project["project_id"] != project_id:
                raise LearningProjectError(
                    "trashed project identity does not match token"
                )
            project["updated_at"] = _now()
            write_json(source, project)
            source.replace(target)
        return deepcopy(project)

    def _trash_identity(self, recovery_token: str) -> tuple[str, Path]:
        token = _text(recovery_token, "recovery_token", maximum=80)
        match = _RESTORE_TOKEN.fullmatch(token)
        if match is None:
            raise LearningProjectError("recovery_token is invalid")
        project_id, nonce = match.groups()
        return project_id, self._trash / f"{project_id}.{nonce}.json"

    def read_trash(self, recovery_token: str) -> dict[str, Any]:
        """Resolve one opaque recovery capability to its full private project."""

        project_id, source = self._trash_identity(recovery_token)
        with self._locked():
            try:
                value = read_json(source)
            except FileNotFoundError as exc:
                raise LearningProjectError("recovery_token was not found") from exc
            project = validate_learning_project(value)
            if project["project_id"] != project_id:
                raise LearningProjectError(
                    "trashed project identity does not match token"
                )
        return deepcopy(project)

    def list_private_documents(
        self, *, include_trash: bool = True
    ) -> list[dict[str, Any]]:
        """Return validated private documents for ownership/reference analysis.

        This method is deliberately not used by HTTP list projections.  It is
        reserved for export/deletion controllers that must identify shared
        references before mutating any store.
        """

        documents = [self.read(item["project_id"]) for item in self.list()]
        if include_trash:
            documents.extend(
                self.read_trash(item["recovery_token"]) for item in self.list_trash()
            )
        return documents

    def purge_trash(self, recovery_token: str) -> str:
        """Permanently remove exactly one already-validated trash artifact.

        Cross-store ownership checks and the explicit confirmation gate belong
        to the data-rights controller.  Keeping this primitive token-bound
        prevents a caller from supplying a raw filesystem path.
        """

        project_id, source = self._trash_identity(recovery_token)
        with self._locked():
            try:
                value = read_json(source)
            except FileNotFoundError as exc:
                raise LearningProjectError("recovery_token was not found") from exc
            project = validate_learning_project(value)
            if project["project_id"] != project_id:
                raise LearningProjectError(
                    "trashed project identity does not match token"
                )
            operations = self._read_operations()
            receipts = operations["receipts"]
            retained = {
                operation_id: receipt
                for operation_id, receipt in receipts.items()
                if receipt["project_id"] != project_id
            }
            try:
                source.unlink()
            except OSError as exc:
                raise LearningProjectError(
                    "trashed project could not be purged"
                ) from exc
            if len(retained) != len(receipts):
                operations["receipts"] = retained
                write_json(self._operations_path, operations)
        return project_id

    def list_trash(self) -> list[dict[str, str]]:
        """Return bounded metadata only; trashed content remains private."""

        rows: list[dict[str, str]] = []
        with self._locked():
            paths = sorted(self._trash.glob("project_*.????????????.json"))
            for path in paths:
                match = re.fullmatch(
                    r"(project_[0-9a-f]{24})\.([0-9a-f]{12})\.json", path.name
                )
                if match is None:
                    continue
                try:
                    project = validate_learning_project(read_json(path))
                except (OSError, json.JSONDecodeError, LearningProjectError) as exc:
                    raise LearningProjectError(
                        f"trashed project {path.name} cannot be read"
                    ) from exc
                project_id, nonce = match.groups()
                if project["project_id"] != project_id:
                    raise LearningProjectError("trashed project identity mismatch")
                rows.append(
                    {
                        "project_id": project_id,
                        "title": project["title"],
                        "updated_at": project["updated_at"],
                        "recovery_token": f"restore_{project_id}_{nonce}",
                    }
                )
        rows.sort(
            key=lambda item: (item["updated_at"], item["project_id"]), reverse=True
        )
        return rows


__all__ = [
    "CHAT_THREAD_ID_PATTERN",
    "LEARNING_PROJECT_SCHEMA",
    "LearningProjectError",
    "LearningProjectStore",
    "new_learning_project",
    "validate_learning_project",
]
