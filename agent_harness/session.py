"""Private, workspace-scoped transcript and run index persistence."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from threading import RLock
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

import fcntl


SESSION_SCHEMA = "agent_harness.session.v1"
_ID = re.compile(r"^[a-z][a-z0-9_-]{7,159}$")
_MAX_MESSAGES = 2_000
_MAX_MESSAGE_CHARS = 200_000
_MAX_SESSION_BYTES = 64 * 1024 * 1024


class SessionStoreError(RuntimeError):
    """Raised when local session state cannot be trusted or persisted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _real_workspace(value: os.PathLike[str] | str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise SessionStoreError("workspace must be an existing directory")
    return path


def _assert_private_directory(path: Path) -> None:
    created = False
    try:
        current = path.lstat()
    except FileNotFoundError:
        path.mkdir(parents=True, mode=0o700)
        os.chmod(path, 0o700)
        current = path.lstat()
        created = True
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
        raise SessionStoreError("session directory must be a real directory")
    if current.st_mode & 0o077:
        if created:
            os.chmod(path, 0o700)
        else:
            raise SessionStoreError("session directory permissions are too broad")


def _assert_regular_file(path: Path) -> bool:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise SessionStoreError("session file must be a regular non-symlink file")
    if current.st_mode & 0o077:
        raise SessionStoreError("session file permissions are too broad")
    return True


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_private_json(path: Path) -> Mapping[str, Any]:
    expected = path.lstat()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SessionStoreError("session file is unreadable") from exc
    try:
        actual = os.fstat(descriptor)
        if (
            not stat.S_ISREG(actual.st_mode)
            or actual.st_mode & 0o077
            or actual.st_dev != expected.st_dev
            or actual.st_ino != expected.st_ino
            or actual.st_size > _MAX_SESSION_BYTES
        ):
            raise SessionStoreError("session file cannot be trusted")
        chunks: list[bytes] = []
        remaining = actual.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        final = os.fstat(descriptor)
        if (
            remaining
            or final.st_size != actual.st_size
            or final.st_dev != actual.st_dev
            or final.st_ino != actual.st_ino
            or final.st_mtime_ns != actual.st_mtime_ns
            or final.st_ctime_ns != actual.st_ctime_ns
        ):
            raise SessionStoreError("session file changed while reading")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionStoreError("session file is unreadable") from exc
    if not isinstance(value, Mapping):
        raise SessionStoreError("session file must contain an object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_SESSION_BYTES:
        raise SessionStoreError("session exceeds the storage limit")
    _assert_private_directory(path.parent)
    _assert_regular_file(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("session write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _validate_message(value: Mapping[str, Any]) -> dict[str, Any]:
    role = value.get("role")
    content = value.get("content")
    if role not in {"user", "assistant"} or not isinstance(content, str):
        raise SessionStoreError("session message is invalid")
    content = content.strip()
    if not content or len(content) > _MAX_MESSAGE_CHARS:
        raise SessionStoreError("session message size is invalid")
    timestamp = value.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        timestamp = _now()
    result = {"role": role, "content": content, "timestamp": timestamp}
    run_id = value.get("run_id")
    if isinstance(run_id, str) and _ID.fullmatch(run_id):
        result["run_id"] = run_id
    return result


def _validate_session(value: Mapping[str, Any], *, workspace: Path) -> dict[str, Any]:
    if value.get("schema") != SESSION_SCHEMA:
        raise SessionStoreError("unsupported session schema")
    session_id = value.get("session_id")
    if not isinstance(session_id, str) or _ID.fullmatch(session_id) is None:
        raise SessionStoreError("session id is invalid")
    stored_workspace = value.get("workspace")
    if stored_workspace != str(workspace):
        raise SessionStoreError("session belongs to another workspace")
    raw_messages = value.get("messages", [])
    if not isinstance(raw_messages, list) or len(raw_messages) > _MAX_MESSAGES:
        raise SessionStoreError("session messages are invalid")
    raw_runs = value.get("runs", [])
    if not isinstance(raw_runs, list) or len(raw_runs) > _MAX_MESSAGES:
        raise SessionStoreError("session runs are invalid")
    runs: list[dict[str, Any]] = []
    for item in raw_runs:
        if not isinstance(item, Mapping):
            raise SessionStoreError("session run reference is invalid")
        run_id = item.get("run_id")
        turn_id = item.get("turn_id")
        status = item.get("status")
        if (
            not isinstance(run_id, str)
            or _ID.fullmatch(run_id) is None
            or not isinstance(turn_id, str)
            or _ID.fullmatch(turn_id) is None
            or status
            not in {"running", "completed", "failed", "cancelled", "handoff"}
        ):
            raise SessionStoreError("session run reference is invalid")
        run: dict[str, Any] = {
            "run_id": run_id,
            "turn_id": turn_id,
            "status": status,
        }
        for timestamp_field in ("started_at", "finished_at"):
            timestamp_value = item.get(timestamp_field)
            if isinstance(timestamp_value, str) and timestamp_value:
                run[timestamp_field] = timestamp_value[:80]
        requires_reconciliation = item.get("requires_reconciliation", False)
        if not isinstance(requires_reconciliation, bool):
            raise SessionStoreError("session run reconciliation flag is invalid")
        run["requires_reconciliation"] = requires_reconciliation
        raw_effects = item.get("effects", [])
        if not isinstance(raw_effects, list) or len(raw_effects) > 64:
            raise SessionStoreError("session run effect ledger is invalid")
        effects: list[dict[str, str]] = []
        for effect in raw_effects:
            if not isinstance(effect, Mapping):
                raise SessionStoreError("session run effect ledger is invalid")
            clean_effect: dict[str, str] = {}
            for field in (
                "call_id",
                "tool_name",
                "arguments_sha256",
                "result_sha256",
            ):
                field_value = effect.get(field)
                if isinstance(field_value, str) and field_value:
                    clean_effect[field] = field_value[:160]
            if not clean_effect.get("call_id") or not clean_effect.get("tool_name"):
                raise SessionStoreError("session run effect ledger is invalid")
            effects.append(clean_effect)
        run["effects"] = effects
        runs.append(run)
    result = dict(value)
    result["messages"] = [_validate_message(item) for item in raw_messages if isinstance(item, Mapping)]
    if len(result["messages"]) != len(raw_messages):
        raise SessionStoreError("session message is invalid")
    result["runs"] = runs
    result["archived"] = bool(result.get("archived", False))
    raw_notices = result.get("risk_notices", [])
    if not isinstance(raw_notices, list) or len(raw_notices) > 32:
        raise SessionStoreError("session risk notices are invalid")
    result["risk_notices"] = [
        str(item).strip()[:500]
        for item in raw_notices
        if isinstance(item, str) and item.strip()
    ]
    if len(result["risk_notices"]) != len(raw_notices):
        raise SessionStoreError("session risk notices are invalid")
    return result


class SessionStore:
    """Workspace-isolated local session store.

    The store never follows a symlink for its managed directories or files and
    never persists provider credentials.
    """

    def __init__(
        self,
        workspace: os.PathLike[str] | str,
        *,
        state_home: os.PathLike[str] | str | None = None,
    ) -> None:
        self.workspace = _real_workspace(workspace)
        base = Path(
            state_home
            or os.getenv("AGENT_HARNESS_HOME", "")
            or (Path.home() / ".agent-harness")
        ).expanduser().absolute()
        _assert_private_directory(base)
        workspace_key = sha256(str(self.workspace).encode("utf-8")).hexdigest()[:24]
        self.root = base / "workspaces" / workspace_key
        self.sessions_directory = self.root / "sessions"
        self.runs_directory = self.root / "runs"
        for directory in (base / "workspaces", self.root, self.sessions_directory, self.runs_directory):
            _assert_private_directory(directory)
        self._lock = RLock()
        self._lock_depth = 0
        lock_flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            lock_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            lock_flags |= os.O_NOFOLLOW
        try:
            self._lock_descriptor = os.open(
                self.root / ".store.lock",
                lock_flags,
                0o600,
            )
        except OSError as exc:
            raise SessionStoreError("session lock file cannot be opened") from exc
        lock_stat = os.fstat(self._lock_descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_mode & 0o077:
            os.close(self._lock_descriptor)
            raise SessionStoreError("session lock file cannot be trusted")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._lock:
            outermost = self._lock_depth == 0
            if outermost:
                fcntl.flock(self._lock_descriptor, fcntl.LOCK_EX)
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
                if outermost:
                    fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)

    @contextmanager
    def turn_lock(self, session_id: str) -> Iterator[None]:
        path = self._path(session_id).with_suffix(".turn.lock")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise SessionStoreError("session turn lock cannot be opened") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
                raise SessionStoreError("session turn lock cannot be trusted")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SessionStoreError("session already has an active run") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @contextmanager
    def workspace_run_lock(self) -> Iterator[None]:
        path = self.root / ".workspace-run.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise SessionStoreError("workspace run lock cannot be opened") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
                raise SessionStoreError("workspace run lock cannot be trusted")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SessionStoreError("workspace already has an active run") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _path(self, session_id: str) -> Path:
        if _ID.fullmatch(session_id) is None:
            raise SessionStoreError("session id is invalid")
        return self.sessions_directory / f"{session_id}.json"

    def create(
        self,
        *,
        provider: str,
        model: str,
        permission_mode: str,
        title: str = "New session",
    ) -> dict[str, Any]:
        session_id = f"session_{uuid4().hex}"
        timestamp = _now()
        session = {
            "schema": SESSION_SCHEMA,
            "session_id": session_id,
            "title": str(title).strip()[:120] or "New session",
            "workspace": str(self.workspace),
            "provider": str(provider).strip()[:80],
            "model": str(model).strip()[:160],
            "permission_mode": str(permission_mode).strip()[:80],
            "created_at": timestamp,
            "updated_at": timestamp,
            "archived": False,
            "messages": [],
            "runs": [],
            "usage": {},
            "risk_notices": [],
        }
        with self._locked():
            _atomic_json(self._path(session_id), session)
        return deepcopy(session)

    def load(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id)
        with self._locked():
            if not _assert_regular_file(path):
                raise SessionStoreError("session does not exist")
            value = _read_private_json(path)
            return deepcopy(_validate_session(value, workspace=self.workspace))

    def save(self, session: Mapping[str, Any]) -> dict[str, Any]:
        candidate = _validate_session(session, workspace=self.workspace)
        candidate["updated_at"] = _now()
        with self._locked():
            _atomic_json(self._path(candidate["session_id"]), candidate)
        return deepcopy(candidate)

    def list(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        with self._locked():
            for path in self.sessions_directory.glob("session_*.json"):
                try:
                    session = self.load(path.stem)
                except SessionStoreError:
                    continue
                if include_archived or not session["archived"]:
                    sessions.append(session)
        return sorted(sessions, key=lambda item: str(item.get("updated_at", "")), reverse=True)

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        run_id: str | None = None,
        reserve_messages: int = 0,
    ) -> dict[str, Any]:
        if (
            isinstance(reserve_messages, bool)
            or not isinstance(reserve_messages, int)
            or not 0 <= reserve_messages <= 2
        ):
            raise SessionStoreError("message reservation is invalid")
        with self._locked():
            session = self.load(session_id)
            if session["archived"]:
                raise SessionStoreError("session is archived; fork it before continuing")
            message: dict[str, Any] = {"role": role, "content": content, "timestamp": _now()}
            if run_id is not None:
                message["run_id"] = run_id
            if len(session["messages"]) + 1 + reserve_messages > _MAX_MESSAGES:
                raise SessionStoreError("session message limit exceeded")
            session["messages"].append(_validate_message(message))
            if len(session["messages"]) == 1 and role == "user":
                session["title"] = " ".join(content.strip().split())[:80] or session["title"]
            return self.save(session)

    def record_run(
        self,
        session_id: str,
        *,
        run_id: str,
        turn_id: str,
        status: str,
        usage: Mapping[str, Any] | None = None,
        effects: Sequence[Mapping[str, Any]] | None = None,
        requires_reconciliation: bool = False,
    ) -> dict[str, Any]:
        if status not in {"running", "completed", "failed", "cancelled", "handoff"}:
            raise SessionStoreError("session run status is invalid")
        if not isinstance(requires_reconciliation, bool):
            raise SessionStoreError("session run reconciliation flag is invalid")
        with self._locked():
            session = self.load(session_id)
            existing = next(
                (item for item in session["runs"] if item.get("run_id") == run_id),
                None,
            )
            timestamp = _now()
            if existing is None:
                if len(session["runs"]) >= _MAX_MESSAGES:
                    raise SessionStoreError("session run limit exceeded")
                existing = {
                    "run_id": run_id,
                    "turn_id": turn_id,
                    "status": status,
                    "started_at": timestamp,
                    "requires_reconciliation": requires_reconciliation,
                    "effects": [dict(item) for item in effects or ()],
                }
                session["runs"].append(existing)
            else:
                if existing.get("turn_id") != turn_id:
                    raise SessionStoreError("session run identity conflict")
                if existing.get("status") != "running":
                    raise SessionStoreError("session run is already settled")
                existing["status"] = status
                existing["requires_reconciliation"] = requires_reconciliation
                existing["effects"] = [dict(item) for item in effects or ()]
            if status != "running":
                existing["finished_at"] = timestamp
            if usage:
                current = session.setdefault("usage", {})
                for key, value in usage.items():
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        current[key] = int(current.get(key, 0)) + value
            return self.save(session)

    def unfinished_runs(self, session_id: str) -> list[dict[str, Any]]:
        session = self.load(session_id)
        return [
            deepcopy(item)
            for item in session["runs"]
            if item.get("status") == "running"
        ]

    def unresolved_workspace_runs(self) -> list[dict[str, str]]:
        unresolved: list[dict[str, str]] = []
        for session in self.list(include_archived=True):
            for run in session["runs"]:
                if (
                    run.get("status") == "running"
                    or run.get("requires_reconciliation") is True
                ):
                    unresolved.append(
                        {
                            "session_id": str(session["session_id"]),
                            "run_id": str(run["run_id"]),
                            "turn_id": str(run["turn_id"]),
                            "status": str(run["status"]),
                        }
                    )
        return unresolved

    def acknowledge_run(self, run_id: str) -> dict[str, str]:
        if _ID.fullmatch(run_id) is None:
            raise SessionStoreError("run id is invalid")
        with self.workspace_run_lock():
            with self._locked():
                matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
                for session in self.list(include_archived=True):
                    for run in session["runs"]:
                        if run.get("run_id") == run_id:
                            matches.append((session, run))
                if len(matches) != 1:
                    raise SessionStoreError(
                        "unresolved run does not exist or is ambiguous"
                    )
                session, run = matches[0]
                if (
                    run.get("status") != "running"
                    and run.get("requires_reconciliation") is not True
                ):
                    raise SessionStoreError("run does not require reconciliation")
                run["status"] = "handoff"
                run["requires_reconciliation"] = False
                run["finished_at"] = _now()
                session.setdefault("risk_notices", []).append(
                    f"Operator acknowledged unresolved run {run_id}; inspect workspace state "
                    "before repeating its external effects."
                )
                self.save(session)
                return {
                    "session_id": str(session["session_id"]),
                    "run_id": run_id,
                    "status": "acknowledged",
                }

    def archive(self, session_id: str, *, archived: bool = True) -> dict[str, Any]:
        with self._locked():
            session = self.load(session_id)
            session["archived"] = bool(archived)
            return self.save(session)

    def fork(self, session_id: str) -> dict[str, Any]:
        with self._locked():
            source = self.load(session_id)
            if any(
                item.get("status") == "running"
                or item.get("requires_reconciliation") is True
                for item in source["runs"]
            ):
                raise SessionStoreError(
                    "cannot fork a session with an unfinished or uncertain run"
                )
            forked = self.create(
                provider=source["provider"],
                model=source["model"],
                permission_mode=source["permission_mode"],
                title=f"{source['title']} (fork)",
            )
            forked["messages"] = deepcopy(source["messages"])
            forked["forked_from"] = source["session_id"]
            return self.save(forked)

    def update_runtime(
        self,
        session_id: str,
        *,
        provider: str,
        model: str,
        permission_mode: str,
    ) -> dict[str, Any]:
        with self._locked():
            session = self.load(session_id)
            if session["archived"]:
                raise SessionStoreError("session is archived; fork it before continuing")
            session["provider"] = str(provider).strip()[:80]
            session["model"] = str(model).strip()[:160]
            session["permission_mode"] = str(permission_mode).strip()[:80]
            return self.save(session)

    def run_paths(self, run_id: str) -> tuple[Path, Path]:
        if _ID.fullmatch(run_id) is None:
            raise SessionStoreError("run id is invalid")
        return (
            self.runs_directory / f"{run_id}.events.jsonl",
            self.runs_directory / f"{run_id}.checkpoint.json",
        )


__all__ = ["SESSION_SCHEMA", "SessionStore", "SessionStoreError"]
