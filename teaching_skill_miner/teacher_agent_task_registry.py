"""Durable, content-private registry for long-running dashboard work.

The Harness journal is authoritative for an individual run's ordered events.
This store is the small control plane around those journals: a task is
registered, together with a dispatch command, before any provider or domain
effect is allowed to start.  Control commands remain in the durable outbox
until the dashboard has applied them to a live handle (or safely reconciled
them after a restart).

The private records deliberately contain the request needed for safe recovery.
Callers must expose only :func:`public_task_projection`; learner/model/tool
content never belongs in the public task APIs.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Any, Iterator, Mapping

try:  # pragma: no cover - production uses POSIX locks.
    import fcntl
except ImportError:  # pragma: no cover - Windows keeps the process lock.
    fcntl = None  # type: ignore[assignment]

from .io_utils import ensure_private_directory, read_json, write_json


TASK_REGISTRY_SCHEMA = "teaching_skill_miner.background_task_registry.v1"
TASK_RECORD_SCHEMA = "teaching_skill_miner.background_task_private.v1"
TASK_PUBLIC_SCHEMA = "teaching_skill_miner.background_task.v1"
TASK_COMMAND_SCHEMA = "teaching_skill_miner.background_task_command.v1"
TASK_EXPORT_SCHEMA = "teaching_skill_miner.project_background_tasks_export.v1"

_TASK_ID_PATTERN = re.compile(r"task_[0-9a-f]{40}")
_RUN_ID_PATTERN = re.compile(r"stream_[0-9a-f]{40}")
_TURN_ID_PATTERN = re.compile(r"turn_[0-9a-f]{40}")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_SAFE_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,79}")
_PRIVATE_TASK_FIELDS = frozenset(
    {
        "schema",
        "task_id",
        "run_id",
        "turn_id",
        "operation",
        "request_fingerprint",
        "private_request",
        "scope",
        "session_id",
        "status",
        "version",
        "created_at_utc",
        "updated_at_utc",
        "last_sequence",
        "terminal_type",
        "error_code",
        "restart_policy",
        "command_outbox",
        "idempotency_receipts",
    }
)
_COMMAND_FIELDS = frozenset(
    {
        "schema",
        "command_id",
        "command_type",
        "idempotency_key_sha256",
        "reason_code",
        "created_at_utc",
    }
)
_TERMINAL_STATUSES = frozenset(
    {"completed", "cancelled", "failed", "handoff", "purged"}
)
_PUBLIC_STATUSES = frozenset(
    {
        "queued",
        "running",
        "cancel_requested",
        "suspended",
        "completed",
        "cancelled",
        "failed",
        "handoff",
    }
)
_MAX_TASKS = 4096
_MAX_COMMANDS_PER_TASK = 16
_MAX_IDEMPOTENCY_RECEIPTS = 64
_MAX_PENDING_CANCELLATIONS = 256
_MAX_PRIVATE_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_REGISTRY_BYTES = 64 * 1024 * 1024


class BackgroundTaskRegistryError(RuntimeError):
    """Raised when the durable task control plane cannot be trusted."""


class BackgroundTaskConflictError(BackgroundTaskRegistryError):
    """Raised for a stale version or idempotency-key collision."""


class BackgroundTaskNotFoundError(BackgroundTaskRegistryError):
    """Raised when an opaque task handle is unavailable."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BackgroundTaskRegistryError(
            "background task value is not canonical JSON"
        ) from exc


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _safe_code(value: Any, *, fallback: str | None = None) -> str | None:
    candidate = str(value or "")
    return candidate if _SAFE_CODE_PATTERN.fullmatch(candidate) else fallback


def _is_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc


def task_id_for_run(run_id: str) -> str:
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise BackgroundTaskRegistryError("background task run identity is invalid")
    return "task_" + sha256(("task\x00" + run_id).encode("utf-8")).hexdigest()[:40]


def public_task_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded control metadata without request, output, or error text."""

    status = str(record.get("status", ""))
    if status not in _PUBLIC_STATUSES:
        raise BackgroundTaskRegistryError("background task status is invalid")
    command_types = {
        str(item.get("command_type", ""))
        for item in record.get("command_outbox", [])
        if isinstance(item, Mapping)
    }
    return {
        "schema": TASK_PUBLIC_SCHEMA,
        "task_id": record["task_id"],
        "run_id": record["run_id"],
        "turn_id": record["turn_id"],
        "operation": record["operation"],
        "status": status,
        "version": record["version"],
        "created_at_utc": record["created_at_utc"],
        "updated_at_utc": record["updated_at_utc"],
        "last_sequence": record["last_sequence"],
        "terminal_type": record.get("terminal_type"),
        "error_code": _safe_code(record.get("error_code")),
        "restart_policy": record["restart_policy"],
        "cancelable": status
        in {"queued", "running", "cancel_requested", "suspended"},
        "resumable": status == "suspended"
        and record.get("restart_policy") == "safe_checkpoint_only",
        "cancel_command_pending": "cancel" in command_types,
        "resume_command_pending": "resume" in command_types,
        "content_included": False,
    }


class DurableBackgroundTaskRegistry:
    """Atomic task state plus a persistent command outbox.

    The registry is a compact private snapshot guarded by both a process lock
    and a POSIX file lock.  Each mutation writes a checksum-sealed replacement,
    fsyncs it, atomically replaces the previous file, then fsyncs the directory.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = ensure_private_directory(Path(root))
        self.path = self.root / "task_registry.json"
        self.lock_path = self.root / ".task_registry.lock"
        self._process_lock = threading.RLock()
        with self._locked():
            if self.path.exists() or self.path.is_symlink():
                self._read_locked()
            else:
                self._write_locked(self._empty_state())

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "schema": TASK_REGISTRY_SCHEMA,
            "tasks": {},
            "pending_cancellations": {},
            "purged_task_tombstones": {},
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._process_lock:
            if self.lock_path.is_symlink():
                raise BackgroundTaskRegistryError(
                    "background task registry lock is unsafe"
                )
            try:
                descriptor = os.open(
                    self.lock_path,
                    os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
            except OSError as exc:
                raise BackgroundTaskRegistryError(
                    "background task registry lock cannot be opened"
                ) from exc
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise BackgroundTaskRegistryError(
                        "background task registry lock is unsafe"
                    )
                os.fchmod(descriptor, 0o600)
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _read_locked(self) -> dict[str, Any]:
        if self.path.is_symlink() or not self.path.is_file():
            raise BackgroundTaskRegistryError("background task registry is unsafe")
        try:
            if self.path.stat().st_size > _MAX_REGISTRY_BYTES:
                raise BackgroundTaskRegistryError(
                    "background task registry exceeds its byte budget"
                )
            value = read_json(self.path)
        except BackgroundTaskRegistryError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackgroundTaskRegistryError(
                "background task registry cannot be read"
            ) from exc
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "schema",
                "tasks",
                "pending_cancellations",
                "purged_task_tombstones",
                "registry_sha256",
            }
            or value.get("schema") != TASK_REGISTRY_SCHEMA
        ):
            raise BackgroundTaskRegistryError("background task registry is invalid")
        sealed = deepcopy(value)
        supplied = sealed.pop("registry_sha256", None)
        if (
            not isinstance(supplied, str)
            or _DIGEST_PATTERN.fullmatch(supplied) is None
            or not secrets.compare_digest(supplied, _digest(sealed))
        ):
            raise BackgroundTaskRegistryError(
                "background task registry integrity failed"
            )
        tasks = sealed.get("tasks")
        pending_cancellations = sealed.get("pending_cancellations", {})
        tombstones = sealed.get("purged_task_tombstones")
        if (
            not isinstance(tasks, dict)
            or len(tasks) > _MAX_TASKS
            or not isinstance(pending_cancellations, dict)
            or len(pending_cancellations) > _MAX_PENDING_CANCELLATIONS
            or not isinstance(tombstones, dict)
            or len(tombstones) > _MAX_TASKS
        ):
            raise BackgroundTaskRegistryError("background task registry is invalid")
        sealed["pending_cancellations"] = pending_cancellations
        for task_id, record in tasks.items():
            self._validate_record(task_id, record)
        for request_digest, command in pending_cancellations.items():
            if (
                _DIGEST_PATTERN.fullmatch(str(request_digest)) is None
                or not isinstance(command, Mapping)
                or set(command) != _COMMAND_FIELDS | {"request_id_sha256"}
                or command.get("schema") != TASK_COMMAND_SCHEMA
                or command.get("command_type") != "cancel"
                or _DIGEST_PATTERN.fullmatch(str(command.get("command_id", "")))
                is None
                or _DIGEST_PATTERN.fullmatch(
                    str(command.get("idempotency_key_sha256", ""))
                )
                is None
                or _SAFE_CODE_PATTERN.fullmatch(
                    str(command.get("reason_code", ""))
                )
                is None
                or not _is_utc_timestamp(command.get("created_at_utc"))
                or command.get("request_id_sha256") != request_digest
            ):
                raise BackgroundTaskRegistryError(
                    "pending background task cancellation is invalid"
                )
        for task_id, tombstone in tombstones.items():
            if (
                _TASK_ID_PATTERN.fullmatch(str(task_id)) is None
                or not isinstance(tombstone, Mapping)
                or set(tombstone) != {"task_id", "purged_at_utc", "record_sha256"}
                or tombstone.get("task_id") != task_id
                or not _is_utc_timestamp(tombstone.get("purged_at_utc"))
                or _DIGEST_PATTERN.fullmatch(str(tombstone.get("record_sha256", "")))
                is None
            ):
                raise BackgroundTaskRegistryError(
                    "background task purge tombstone is invalid"
                )
        return sealed

    @staticmethod
    def _validate_record(task_id: str, value: Any) -> None:
        if not isinstance(value, Mapping):
            raise BackgroundTaskRegistryError("background task record is invalid")
        private_request = value.get("private_request")
        request_fields = (
            {"operation", "payload", "run_id", "turn_id", "after_sequence"}
            | (
                {"request_id"}
                if isinstance(private_request, Mapping)
                and "request_id" in private_request
                else set()
            )
        )
        terminal_by_status = {
            "completed": "run.completed",
            "cancelled": "run.cancelled",
            "failed": "run.failed",
            "handoff": "run.handoff",
        }
        operation = (
            value.get("operation") if isinstance(value.get("operation"), str) else ""
        )
        status = value.get("status") if isinstance(value.get("status"), str) else ""
        restart_policy = (
            value.get("restart_policy")
            if isinstance(value.get("restart_policy"), str)
            else ""
        )
        terminal_type = value.get("terminal_type")
        session_id = value.get("session_id")
        terminal_invalid = (
            terminal_type != terminal_by_status[str(status)]
            if status in terminal_by_status
            else terminal_type is not None
        )
        if (
            set(value) != _PRIVATE_TASK_FIELDS
            or _TASK_ID_PATTERN.fullmatch(task_id) is None
            or value.get("schema") != TASK_RECORD_SCHEMA
            or value.get("task_id") != task_id
            or task_id_for_run(str(value.get("run_id", ""))) != task_id
            or _TURN_ID_PATTERN.fullmatch(str(value.get("turn_id", ""))) is None
            or operation not in {"chat", "start", "step"}
            or _DIGEST_PATTERN.fullmatch(
                str(value.get("request_fingerprint", ""))
            )
            is None
            or status not in _PUBLIC_STATUSES
            or isinstance(value.get("version"), bool)
            or not isinstance(value.get("version"), int)
            or int(value["version"]) < 0
            or isinstance(value.get("last_sequence"), bool)
            or not isinstance(value.get("last_sequence"), int)
            or int(value["last_sequence"]) < 0
            or restart_policy
            not in {"safe_checkpoint_only", "unsafe_external_effect_handoff"}
            or restart_policy
            != (
                "unsafe_external_effect_handoff"
                if operation == "chat"
                else "safe_checkpoint_only"
            )
            or terminal_invalid
            or (
                value.get("error_code") is not None
                and _SAFE_CODE_PATTERN.fullmatch(str(value.get("error_code"))) is None
            )
            or (status == "completed" and value.get("error_code") is not None)
            or (status == "suspended" and value.get("error_code") is None)
            or not _is_utc_timestamp(value.get("created_at_utc"))
            or not _is_utc_timestamp(value.get("updated_at_utc"))
            or (
                session_id is not None
                and (
                    not isinstance(session_id, str)
                    or not session_id
                    or session_id != session_id.strip()
                    or len(session_id) > 160
                )
            )
            or not isinstance(private_request, Mapping)
            or set(private_request) != request_fields
            or private_request.get("operation") != operation
            or private_request.get("run_id") != value.get("run_id")
            or private_request.get("turn_id") != value.get("turn_id")
            or private_request.get("after_sequence") != 0
            or not isinstance(private_request.get("payload"), Mapping)
            or (
                "request_id" in private_request
                and (
                    not isinstance(private_request.get("request_id"), str)
                    or not private_request.get("request_id")
                    or private_request.get("request_id")
                    != str(private_request.get("request_id")).strip()
                    or len(str(private_request.get("request_id"))) > 160
                )
            )
            or not isinstance(value.get("scope"), Mapping)
            or not isinstance(value.get("command_outbox"), list)
            or len(value["command_outbox"]) > _MAX_COMMANDS_PER_TASK
            or not isinstance(value.get("idempotency_receipts"), Mapping)
            or len(value["idempotency_receipts"]) > _MAX_IDEMPOTENCY_RECEIPTS
        ):
            raise BackgroundTaskRegistryError("background task record is invalid")
        if len(_canonical_bytes(value["private_request"])) > _MAX_PRIVATE_REQUEST_BYTES:
            raise BackgroundTaskRegistryError(
                "background task private request exceeds its byte budget"
            )
        for command in value["command_outbox"]:
            if (
                not isinstance(command, Mapping)
                or set(command) != _COMMAND_FIELDS
                or command.get("schema") != TASK_COMMAND_SCHEMA
                or str(command.get("command_type", ""))
                not in {"dispatch", "cancel", "resume"}
                or _DIGEST_PATTERN.fullmatch(str(command.get("command_id", "")))
                is None
                or _DIGEST_PATTERN.fullmatch(
                    str(command.get("idempotency_key_sha256", ""))
                )
                is None
                or (
                    command.get("reason_code") is not None
                    and _SAFE_CODE_PATTERN.fullmatch(
                        str(command.get("reason_code"))
                    )
                    is None
                )
                or not _is_utc_timestamp(command.get("created_at_utc"))
            ):
                raise BackgroundTaskRegistryError(
                    "background task command outbox is invalid"
                )
        for receipt_digest, receipt in value["idempotency_receipts"].items():
            if (
                _DIGEST_PATTERN.fullmatch(str(receipt_digest)) is None
                or not isinstance(receipt, Mapping)
                or set(receipt) != {"fingerprint", "applied", "command_type"}
                or _DIGEST_PATTERN.fullmatch(str(receipt.get("fingerprint", "")))
                is None
                or not isinstance(receipt.get("applied"), bool)
                or str(receipt.get("command_type", ""))
                not in {"cancel", "resume"}
            ):
                raise BackgroundTaskRegistryError(
                    "background task idempotency receipt is invalid"
                )

    def _write_locked(self, state: Mapping[str, Any]) -> None:
        material = deepcopy(dict(state))
        material.pop("registry_sha256", None)
        material["registry_sha256"] = _digest(material)
        if len(_canonical_bytes(material)) > _MAX_REGISTRY_BYTES:
            raise BackgroundTaskRegistryError(
                "background task registry exceeds its byte budget"
            )
        try:
            write_json(self.path, material)
            descriptor = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise BackgroundTaskRegistryError(
                "background task registry mutation is not durable"
            ) from exc

    @staticmethod
    def _command(
        task_id: str,
        command_type: str,
        idempotency_key: str,
        *,
        reason_code: str | None = None,
    ) -> dict[str, Any]:
        command_id = _digest(
            {
                "task_id": task_id,
                "command_type": command_type,
                "idempotency_key": idempotency_key,
            }
        )
        return {
            "schema": TASK_COMMAND_SCHEMA,
            "command_id": command_id,
            "command_type": command_type,
            "idempotency_key_sha256": sha256(
                idempotency_key.encode("utf-8")
            ).hexdigest(),
            "reason_code": reason_code,
            "created_at_utc": _utc_now(),
        }

    def register(
        self,
        *,
        run_id: str,
        turn_id: str,
        operation: str,
        request_fingerprint: str,
        private_request: Mapping[str, Any],
        scope: Mapping[str, Any],
        session_id: str | None,
    ) -> tuple[dict[str, Any], bool]:
        """Durably enqueue a task before any external or domain effect."""

        task_id = task_id_for_run(run_id)
        if (
            _TURN_ID_PATTERN.fullmatch(turn_id) is None
            or operation not in {"chat", "start", "step"}
            or _DIGEST_PATTERN.fullmatch(request_fingerprint) is None
        ):
            raise BackgroundTaskRegistryError("background task identity is invalid")
        request = deepcopy(dict(private_request))
        task_scope = deepcopy(dict(scope))
        if len(_canonical_bytes(request)) > _MAX_PRIVATE_REQUEST_BYTES:
            raise BackgroundTaskRegistryError(
                "background task private request exceeds its byte budget"
            )
        with self._locked():
            state = self._read_locked()
            if task_id in state["purged_task_tombstones"]:
                raise BackgroundTaskConflictError(
                    "background task identity was permanently purged"
                )
            existing = state["tasks"].get(task_id)
            if existing is not None:
                if (
                    existing.get("run_id") != run_id
                    or existing.get("turn_id") != turn_id
                    or existing.get("operation") != operation
                    or existing.get("request_fingerprint") != request_fingerprint
                    or existing.get("private_request") != request
                    or existing.get("scope") != task_scope
                ):
                    raise BackgroundTaskConflictError(
                        "background task identity collides with another request"
                    )
                return deepcopy(existing), False
            if len(state["tasks"]) >= _MAX_TASKS:
                raise BackgroundTaskRegistryError(
                    "background task retention capacity is exhausted"
                )
            now = _utc_now()
            dispatch = self._command(task_id, "dispatch", "initial-dispatch")
            request_id = request.get("request_id")
            request_digest = (
                sha256(request_id.encode("utf-8")).hexdigest()
                if isinstance(request_id, str) and request_id
                else None
            )
            pending_cancel = (
                state["pending_cancellations"].pop(request_digest, None)
                if request_digest is not None
                else None
            )
            commands = [dispatch]
            if pending_cancel is not None:
                command = deepcopy(dict(pending_cancel))
                command.pop("request_id_sha256", None)
                commands.append(command)
            record = {
                "schema": TASK_RECORD_SCHEMA,
                "task_id": task_id,
                "run_id": run_id,
                "turn_id": turn_id,
                "operation": operation,
                "request_fingerprint": request_fingerprint,
                "private_request": request,
                "scope": task_scope,
                "session_id": session_id,
                "status": "cancel_requested" if pending_cancel else "queued",
                "version": 0,
                "created_at_utc": now,
                "updated_at_utc": now,
                "last_sequence": 0,
                "terminal_type": None,
                "error_code": None,
                "restart_policy": (
                    "safe_checkpoint_only"
                    if operation in {"start", "step"}
                    else "unsafe_external_effect_handoff"
                ),
                "command_outbox": commands,
                "idempotency_receipts": {},
            }
            self._validate_record(task_id, record)
            state["tasks"][task_id] = record
            self._write_locked(state)
            return deepcopy(record), True

    def request_cancel_by_request_id(
        self, request_id: str, *, reason_code: str
    ) -> tuple[dict[str, Any] | None, bool]:
        """Persist Stop even when registration and response headers race."""

        if (
            not isinstance(request_id, str)
            or not request_id
            or request_id != request_id.strip()
            or len(request_id) > 160
            or _SAFE_CODE_PATTERN.fullmatch(reason_code) is None
        ):
            raise BackgroundTaskRegistryError(
                "pending background task cancellation is invalid"
            )
        request_digest = sha256(request_id.encode("utf-8")).hexdigest()
        while True:
            with self._locked():
                state = self._read_locked()
                matches = [
                    deepcopy(record)
                    for record in state["tasks"].values()
                    if isinstance(record.get("private_request"), Mapping)
                    and record["private_request"].get("request_id") == request_id
                ]
                if len(matches) > 1:
                    raise BackgroundTaskConflictError(
                        "request_id matches more than one background task"
                    )
                if not matches:
                    existing = state["pending_cancellations"].get(request_digest)
                    if existing is not None:
                        if existing.get("reason_code") != reason_code:
                            raise BackgroundTaskConflictError(
                                "pending cancellation reason conflicts"
                            )
                        return None, True
                    if (
                        len(state["pending_cancellations"])
                        >= _MAX_PENDING_CANCELLATIONS
                    ):
                        raise BackgroundTaskRegistryError(
                            "pending background task cancellation capacity is full"
                        )
                    command = self._command(
                        "task_" + "0" * 40,
                        "cancel",
                        "pending:" + request_digest,
                        reason_code=reason_code,
                    )
                    command["request_id_sha256"] = request_digest
                    state["pending_cancellations"][request_digest] = command
                    self._write_locked(state)
                    return None, True
                match = matches[0]
            try:
                return self.request_control(
                    str(match["task_id"]),
                    command_type="cancel",
                    expected_version=int(match["version"]),
                    idempotency_key="request-cancel:" + request_digest,
                    reason_code=reason_code,
                )
            except BackgroundTaskConflictError:
                # A simultaneous worker transition advanced the version. Read
                # the new current state and retry the same deterministic command.
                current = self.get_private(str(match["task_id"]))
                receipt_key = sha256(
                    ("request-cancel:" + request_digest).encode("utf-8")
                ).hexdigest()
                if receipt_key in current["idempotency_receipts"]:
                    return current, bool(
                        current["idempotency_receipts"][receipt_key].get(
                            "applied", False
                        )
                    )
                if current["status"] in _TERMINAL_STATUSES:
                    return current, False

    def get_private(self, task_id: str) -> dict[str, Any]:
        with self._locked():
            record = self._read_locked()["tasks"].get(task_id)
            if record is None:
                raise BackgroundTaskNotFoundError(
                    "background task is no longer available"
                )
            return deepcopy(record)

    def find_by_run_id(self, run_id: str) -> dict[str, Any] | None:
        task_id = task_id_for_run(run_id)
        with self._locked():
            record = self._read_locked()["tasks"].get(task_id)
            return deepcopy(record) if record is not None else None

    def list_private(self) -> list[dict[str, Any]]:
        with self._locked():
            records = list(self._read_locked()["tasks"].values())
        return sorted(
            (deepcopy(item) for item in records),
            key=lambda item: (item["created_at_utc"], item["task_id"]),
            reverse=True,
        )

    def list_public(self) -> list[dict[str, Any]]:
        return [public_task_projection(item) for item in self.list_private()]

    def _mutate(
        self,
        task_id: str,
        callback: Any,
    ) -> dict[str, Any]:
        with self._locked():
            state = self._read_locked()
            record = state["tasks"].get(task_id)
            if record is None:
                raise BackgroundTaskNotFoundError(
                    "background task is no longer available"
                )
            candidate = deepcopy(record)
            changed = bool(callback(candidate))
            if changed:
                candidate["updated_at_utc"] = _utc_now()
                candidate["version"] = int(candidate["version"]) + 1
                self._validate_record(task_id, candidate)
                state["tasks"][task_id] = candidate
                self._write_locked(state)
            return deepcopy(candidate)

    def mark_running(self, task_id: str) -> dict[str, Any]:
        def update(record: dict[str, Any]) -> bool:
            commands = [
                item
                for item in record["command_outbox"]
                if item.get("command_type") not in {"dispatch", "resume"}
            ]
            changed = commands != record["command_outbox"]
            record["command_outbox"] = commands
            if record["status"] == "queued":
                record["status"] = "running"
                changed = True
            return changed

        return self._mutate(task_id, update)

    def update_runtime(
        self,
        task_id: str,
        *,
        last_sequence: int,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(last_sequence, bool) or last_sequence < 0:
            raise BackgroundTaskRegistryError("task sequence is invalid")

        def update(record: dict[str, Any]) -> bool:
            changed = False
            if last_sequence > int(record["last_sequence"]):
                record["last_sequence"] = last_sequence
                changed = True
            if session_id and record.get("session_id") != session_id:
                record["session_id"] = session_id
                changed = True
            return changed

        return self._mutate(task_id, update)

    def mark_terminal(
        self,
        task_id: str,
        *,
        status: str,
        terminal_type: str,
        last_sequence: int,
        error_code: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if status not in _TERMINAL_STATUSES - {"purged"}:
            raise BackgroundTaskRegistryError("task terminal status is invalid")

        def update(record: dict[str, Any]) -> bool:
            if record["status"] in _TERMINAL_STATUSES:
                if (
                    record["status"] != status
                    or record.get("terminal_type") != terminal_type
                ):
                    raise BackgroundTaskConflictError(
                        "background task already has another terminal state"
                    )
                return False
            record["status"] = status
            record["terminal_type"] = terminal_type
            record["last_sequence"] = max(
                int(record["last_sequence"]), int(last_sequence)
            )
            record["error_code"] = (
                None
                if status == "completed"
                else _safe_code(error_code, fallback=f"task_{status}")
            )
            if session_id:
                record["session_id"] = session_id
            record["command_outbox"] = []
            return True

        return self._mutate(task_id, update)

    def mark_suspended(
        self,
        task_id: str,
        *,
        error_code: str,
        last_sequence: int,
        unsafe_handoff: bool,
    ) -> dict[str, Any]:
        def update(record: dict[str, Any]) -> bool:
            if record["status"] in _TERMINAL_STATUSES:
                return False
            record["status"] = "handoff" if unsafe_handoff else "suspended"
            record["terminal_type"] = "run.handoff" if unsafe_handoff else None
            record["error_code"] = _safe_code(
                error_code, fallback="task_recovery_required"
            )
            record["last_sequence"] = max(
                int(record["last_sequence"]), int(last_sequence)
            )
            record["command_outbox"] = []
            return True

        return self._mutate(task_id, update)

    @staticmethod
    def _validate_control(
        expected_version: int, idempotency_key: str, reason_code: str | None
    ) -> None:
        if (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version < 0
            or not isinstance(idempotency_key, str)
            or not idempotency_key
            or idempotency_key != idempotency_key.strip()
            or len(idempotency_key) > 160
            or (
                reason_code is not None
                and (
                    not isinstance(reason_code, str)
                    or not reason_code
                    or reason_code != reason_code.strip()
                    or len(reason_code) > 160
                    or _SAFE_CODE_PATTERN.fullmatch(reason_code) is None
                )
            )
        ):
            raise BackgroundTaskRegistryError(
                "background task control request is invalid"
            )

    def request_control(
        self,
        task_id: str,
        *,
        command_type: str,
        expected_version: int,
        idempotency_key: str,
        reason_code: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        if command_type not in {"cancel", "resume"}:
            raise BackgroundTaskRegistryError("task control command is invalid")
        self._validate_control(expected_version, idempotency_key, reason_code)
        fingerprint = _digest(
            {
                "task_id": task_id,
                "command_type": command_type,
                "expected_version": expected_version,
                "reason_code": reason_code,
            }
        )
        receipt_key = sha256(idempotency_key.encode("utf-8")).hexdigest()
        applied = False

        def update(record: dict[str, Any]) -> bool:
            nonlocal applied
            prior = record["idempotency_receipts"].get(receipt_key)
            if prior is not None:
                if prior.get("fingerprint") != fingerprint:
                    raise BackgroundTaskConflictError(
                        "background task idempotency key was reused"
                    )
                applied = bool(prior.get("applied", False))
                return False
            if int(record["version"]) != expected_version:
                raise BackgroundTaskConflictError(
                    "background task expected_version is stale"
                )
            if command_type == "cancel":
                allowed = record["status"] not in _TERMINAL_STATUSES
            else:
                allowed = (
                    record["status"] == "suspended"
                    and record["restart_policy"] == "safe_checkpoint_only"
                )
            if command_type == "resume" and not allowed:
                raise BackgroundTaskConflictError(
                    "background task cannot be safely resumed"
                )
            if allowed:
                if len(record["command_outbox"]) >= _MAX_COMMANDS_PER_TASK:
                    raise BackgroundTaskRegistryError(
                        "background task command outbox is full"
                    )
                record["command_outbox"].append(
                    self._command(
                        task_id,
                        command_type,
                        idempotency_key,
                        reason_code=reason_code,
                    )
                )
                record["status"] = (
                    "cancel_requested" if command_type == "cancel" else "queued"
                )
                record["terminal_type"] = None
                record["error_code"] = None
                applied = True
            receipts = record["idempotency_receipts"]
            if len(receipts) >= _MAX_IDEMPOTENCY_RECEIPTS:
                oldest = next(iter(receipts))
                receipts.pop(oldest, None)
            receipts[receipt_key] = {
                "fingerprint": fingerprint,
                "applied": applied,
                "command_type": command_type,
            }
            return True

        return self._mutate(task_id, update), applied

    def acknowledge_command(self, task_id: str, command_id: str) -> dict[str, Any]:
        def update(record: dict[str, Any]) -> bool:
            retained = [
                item
                for item in record["command_outbox"]
                if item.get("command_id") != command_id
            ]
            if len(retained) == len(record["command_outbox"]):
                return False
            record["command_outbox"] = retained
            return True

        return self._mutate(task_id, update)

    def project_records(
        self,
        project_id: str,
        session_ids: set[str],
        *,
        require_terminal: bool,
    ) -> dict[str, dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for record in self.list_private():
            scope = record.get("scope", {})
            belongs = (
                isinstance(scope, Mapping)
                and scope.get("project_id") == project_id
            ) or record.get("session_id") in session_ids
            if not belongs:
                continue
            if require_terminal and record["status"] not in _TERMINAL_STATUSES:
                raise BackgroundTaskRegistryError(
                    "a referenced background task is still active"
                )
            selected[record["task_id"]] = record
        return selected

    @staticmethod
    def project_export(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "schema": TASK_EXPORT_SCHEMA,
            "tasks": [deepcopy(records[key]) for key in sorted(records)],
            "contains_private_request_content": True,
        }

    def purge_tasks(self, task_ids: list[str]) -> dict[str, int]:
        unique = sorted(set(task_ids))
        with self._locked():
            state = self._read_locked()
            removed = 0
            for task_id in unique:
                record = state["tasks"].get(task_id)
                if record is None:
                    if task_id not in state["purged_task_tombstones"]:
                        raise BackgroundTaskNotFoundError(
                            "background task purge target is unavailable"
                        )
                    continue
                if record["status"] not in _TERMINAL_STATUSES:
                    raise BackgroundTaskConflictError(
                        "an active background task cannot be purged"
                    )
                state["purged_task_tombstones"][task_id] = {
                    "task_id": task_id,
                    "purged_at_utc": _utc_now(),
                    "record_sha256": _digest(record),
                }
                state["tasks"].pop(task_id)
                removed += 1
            if removed:
                self._write_locked(state)
            cleaned_locks = 0
            for task_id in unique:
                worker_lock = self.root / f".{task_id}.worker.lock"
                try:
                    if worker_lock.is_symlink():
                        raise BackgroundTaskRegistryError(
                            "background task worker lock is unsafe"
                        )
                    if worker_lock.exists():
                        worker_lock.unlink()
                        cleaned_locks += 1
                except BackgroundTaskRegistryError:
                    raise
                except OSError as exc:
                    raise BackgroundTaskRegistryError(
                        "background task private content was purged but worker-lock "
                        "cleanup is pending"
                    ) from exc
            if cleaned_locks:
                try:
                    descriptor = os.open(self.root, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                except OSError as exc:
                    raise BackgroundTaskRegistryError(
                        "background task purge directory durability is uncertain"
                    ) from exc
            return {"tasks": removed, "tombstones": removed}


__all__ = [
    "BackgroundTaskConflictError",
    "BackgroundTaskNotFoundError",
    "BackgroundTaskRegistryError",
    "DurableBackgroundTaskRegistry",
    "TASK_EXPORT_SCHEMA",
    "TASK_PUBLIC_SCHEMA",
    "public_task_projection",
    "task_id_for_run",
]
