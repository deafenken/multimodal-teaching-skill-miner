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
from threading import RLock
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

import fcntl

from .core.approvals import ApprovalRule


SESSION_SCHEMA = "agent_harness.session.v1"
APPROVAL_RULES_SCHEMA = "agent_harness.approval_rules.v1"
HOOK_TRUST_SCHEMA = "agent_harness.hook_trust.v1"
MCP_TRUST_SCHEMA = "agent_harness.mcp_trust.v1"
MCP_CATALOG_SCHEMA = "agent_harness.mcp_catalog_state.v1"
CONTEXT_COMPACTION_SCHEMA = "agent_harness.context_compaction.v1"
_ID = re.compile(r"^[a-z][a-z0-9_-]{7,159}$")
_HOOK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_MESSAGE_ID = re.compile(r"^message_[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_MESSAGES = 2_000
_MAX_MESSAGE_CHARS = 200_000
_MAX_SESSION_BYTES = 64 * 1024 * 1024
_MAX_APPROVAL_RULES = 2_048
_MAX_HOOK_TRUST_RECORDS = 512
_MAX_MCP_TRUST_RECORDS = 128
_MAX_MCP_CATALOG_SERVERS = 64
_MAX_COMPACTIONS = 512
_MAX_COMPACTION_SUMMARY_CHARS = 40_000


class SessionStoreError(RuntimeError):
    """Raised when local session state cannot be trusted or persisted."""


class _SessionDirectoryMissing(SessionStoreError):
    """Internal sentinel for a missing managed directory."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _real_workspace(value: os.PathLike[str] | str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise SessionStoreError("workspace must be an existing directory")
    return path


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _canonical_state_path(value: os.PathLike[str] | str) -> Path:
    """Normalize trusted OS aliases while rejecting user-controlled symlinks."""

    raw = Path(value).expanduser()
    path = Path(os.path.abspath(os.fspath(raw)))
    if not path.is_absolute():
        raise SessionStoreError("session directory path must be absolute")
    parts = path.parts[1:]
    current = Path(path.anchor)
    for index, component in enumerate(parts):
        candidate = current / component
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            return current.joinpath(*parts[index:])
        except OSError as exc:
            raise SessionStoreError("session directory path is unreadable") from exc
        if not stat.S_ISLNK(metadata.st_mode):
            current = candidate
            continue

        try:
            parent = current.stat()
        except OSError as exc:
            raise SessionStoreError("session directory path is unreadable") from exc
        if (
            metadata.st_uid != 0
            or parent.st_uid != 0
            or stat.S_IMODE(parent.st_mode) & 0o022
        ):
            raise SessionStoreError(
                "session directory must be a real directory; "
                "path contains an untrusted symlink"
            )
        try:
            current = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SessionStoreError("session directory path is unreadable") from exc
    return current


def _validate_directory_ancestor(metadata: os.stat_result) -> None:
    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid not in {0, os.getuid()}:
        raise SessionStoreError("session directory ancestor owner is invalid")
    if mode & 0o022:
        root_sticky_directory = (
            metadata.st_uid == 0
            and bool(mode & stat.S_ISVTX)
            and bool(mode & 0o002)
        )
        if not root_sticky_directory:
            raise SessionStoreError(
                "session directory ancestor permissions are too broad"
            )


@contextmanager
def _open_private_directory(
    path: Path,
    *,
    create: bool,
) -> Iterator[int]:
    """Open a private directory without following any path-component symlink."""

    if not path.is_absolute():
        raise SessionStoreError("session directory path must be absolute")
    flags = _directory_open_flags()
    try:
        descriptor = os.open(path.anchor, flags)
    except OSError as exc:
        raise SessionStoreError("session directory path is unreadable") from exc
    private_tail = False
    try:
        components = path.parts[1:]
        if not components:
            metadata = os.fstat(descriptor)
            if metadata.st_uid != os.getuid():
                raise SessionStoreError("session directory owner is invalid")
            if metadata.st_mode & 0o077:
                raise SessionStoreError(
                    "session directory permissions are too broad"
                )
        for index, component in enumerate(components):
            final = index == len(components) - 1
            created = False
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError as exc:
                if not create:
                    raise _SessionDirectoryMissing(
                        "session directory does not exist"
                    ) from exc
                private_tail = True
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                except OSError as mkdir_exc:
                    raise SessionStoreError(
                        "session directory cannot be created"
                    ) from mkdir_exc
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except OSError as open_exc:
                    raise SessionStoreError(
                        "session directory must be a real directory"
                    ) from open_exc
            except OSError as exc:
                raise SessionStoreError(
                    "session directory must be a real directory; "
                    "path contains an untrusted symlink"
                ) from exc

            try:
                metadata = os.fstat(child)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise SessionStoreError(
                        "session directory must be a real directory"
                    )
                if created:
                    os.fchmod(child, 0o700)
                    metadata = os.fstat(child)
                if final or private_tail:
                    if metadata.st_uid != os.getuid():
                        raise SessionStoreError(
                            "session directory owner is invalid"
                        )
                    if metadata.st_mode & 0o077:
                        raise SessionStoreError(
                            "session directory permissions are too broad"
                        )
                else:
                    _validate_directory_ancestor(metadata)
            except Exception:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _assert_private_directory(path: Path) -> None:
    with _open_private_directory(path, create=True):
        pass


def _assert_regular_file_at(parent_descriptor: int, name: str) -> bool:
    try:
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or current.st_uid != os.getuid()
        or current.st_nlink != 1
    ):
        raise SessionStoreError("session file must be a regular non-symlink file")
    if current.st_mode & 0o077:
        raise SessionStoreError("session file permissions are too broad")
    return True


def _assert_regular_file(path: Path) -> bool:
    try:
        with _open_private_directory(
            path.parent,
            create=False,
        ) as parent_descriptor:
            return _assert_regular_file_at(parent_descriptor, path.name)
    except _SessionDirectoryMissing:
        return False


def _read_private_json(path: Path) -> Mapping[str, Any]:
    with _open_private_directory(path.parent, create=False) as parent_descriptor:
        try:
            expected = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise SessionStoreError("session file is unreadable") from exc
        if (
            stat.S_ISLNK(expected.st_mode)
            or not stat.S_ISREG(expected.st_mode)
            or expected.st_mode & 0o077
            or expected.st_uid != os.getuid()
            or expected.st_nlink != 1
        ):
            raise SessionStoreError("session file cannot be trusted")
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise SessionStoreError("session file is unreadable") from exc
        try:
            actual = os.fstat(descriptor)
            if (
                not stat.S_ISREG(actual.st_mode)
                or actual.st_mode & 0o077
                or actual.st_uid != os.getuid()
                or actual.st_nlink != 1
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
    with _open_private_directory(path.parent, create=True) as parent_descriptor:
        _assert_regular_file_at(parent_descriptor, path.name)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        temporary_name = f".{path.name}.{uuid4().hex}.tmp"
        try:
            descriptor = os.open(
                temporary_name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise SessionStoreError("session temporary file cannot be created") from exc
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
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            raise


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SessionStoreError("session value is not canonical JSON") from exc


def _message_content_sha256(*, role: str, content: str) -> str:
    material = {"role": role, "content": content}
    return sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _legacy_message_id(
    *,
    session_id: str,
    index: int,
    role: str,
    content_sha256: str,
    timestamp: str,
    run_id: str | None,
) -> str:
    material = {
        "session_id": session_id,
        "index": index,
        "role": role,
        "content_sha256": content_sha256,
        "timestamp": timestamp,
        "run_id": run_id,
    }
    digest = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    return f"message_{digest[:32]}"


def _message_prefix_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    material = [
        {
            "message_id": item["message_id"],
            "role": item["role"],
            "content_sha256": item["content_sha256"],
        }
        for item in messages
    ]
    return sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _validate_message(
    value: Mapping[str, Any],
    *,
    session_id: str,
    index: int,
    fallback_timestamp: str,
) -> dict[str, Any]:
    role = value.get("role")
    content = value.get("content")
    if role not in {"user", "assistant"} or not isinstance(content, str):
        raise SessionStoreError("session message is invalid")
    content = content.strip()
    if not content or len(content) > _MAX_MESSAGE_CHARS:
        raise SessionStoreError("session message size is invalid")
    timestamp = value.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        timestamp = fallback_timestamp
    timestamp = timestamp[:80]
    identities: dict[str, str] = {}
    for field in ("run_id", "turn_id"):
        identity = value.get(field)
        if isinstance(identity, str) and _ID.fullmatch(identity) is not None:
            identities[field] = identity
    run_id = identities.get("run_id")
    content_sha256 = _message_content_sha256(role=role, content=content)
    stored_content_sha256 = value.get("content_sha256")
    if stored_content_sha256 is not None and stored_content_sha256 != content_sha256:
        raise SessionStoreError("session message content digest is invalid")
    message_id = value.get("message_id")
    if message_id is None:
        message_id = _legacy_message_id(
            session_id=session_id,
            index=index,
            role=role,
            content_sha256=content_sha256,
            timestamp=timestamp,
            run_id=run_id,
        )
    if not isinstance(message_id, str) or _MESSAGE_ID.fullmatch(message_id) is None:
        raise SessionStoreError("session message id is invalid")
    result = {
        "message_id": message_id,
        "role": role,
        "content": content,
        "content_sha256": content_sha256,
        "timestamp": timestamp,
    }
    result.update(identities)
    return result


def _validate_compactions(
    raw_compactions: Any,
    *,
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(raw_compactions, list) or len(raw_compactions) > _MAX_COMPACTIONS:
        raise SessionStoreError("session compactions are invalid")
    result: list[dict[str, Any]] = []
    previous_id: str | None = None
    previous_count = 0
    for raw in raw_compactions:
        if not isinstance(raw, Mapping):
            raise SessionStoreError("session compaction is invalid")
        if raw.get("schema") != CONTEXT_COMPACTION_SCHEMA:
            raise SessionStoreError("unsupported context compaction schema")
        compaction_id = raw.get("compaction_id")
        source_message_count = raw.get("source_message_count")
        summary = raw.get("summary")
        if (
            not isinstance(compaction_id, str)
            or _ID.fullmatch(compaction_id) is None
            or isinstance(source_message_count, bool)
            or not isinstance(source_message_count, int)
            or not previous_count < source_message_count <= len(messages)
            or not isinstance(summary, str)
        ):
            raise SessionStoreError("session compaction is invalid")
        summary = summary.strip()
        if not summary or len(summary) > _MAX_COMPACTION_SUMMARY_CHARS:
            raise SessionStoreError("session compaction summary is invalid")
        covered = messages[:source_message_count]
        if covered[-1]["role"] != "assistant":
            raise SessionStoreError("session compaction must end at an assistant boundary")
        expected_prefix_sha256 = _message_prefix_sha256(covered)
        expected_summary_sha256 = sha256(summary.encode("utf-8")).hexdigest()
        if (
            raw.get("source_first_message_id") != covered[0]["message_id"]
            or raw.get("source_last_message_id") != covered[-1]["message_id"]
            or raw.get("source_messages_sha256") != expected_prefix_sha256
            or raw.get("summary_sha256") != expected_summary_sha256
        ):
            raise SessionStoreError("session compaction lineage is invalid")
        parent = raw.get("parent_compaction_id")
        if parent != previous_id:
            raise SessionStoreError("session compaction parent is invalid")
        expected_parent_summary = result[-1]["summary_sha256"] if result else None
        if raw.get("parent_summary_sha256") != expected_parent_summary:
            raise SessionStoreError("session compaction parent summary is invalid")
        created_at = raw.get("created_at")
        provider = raw.get("provider")
        model = raw.get("model")
        trigger = raw.get("trigger")
        if (
            not isinstance(created_at, str)
            or not created_at
            or not isinstance(provider, str)
            or not provider.strip()
            or not isinstance(model, str)
            or not model.strip()
            or trigger not in {"manual", "automatic"}
        ):
            raise SessionStoreError("session compaction metadata is invalid")
        instruction_digest = raw.get("instructions_sha256")
        if instruction_digest is not None and (
            not isinstance(instruction_digest, str)
            or _SHA256.fullmatch(instruction_digest) is None
        ):
            raise SessionStoreError("session compaction instruction digest is invalid")
        raw_usage = raw.get("usage", {})
        if not isinstance(raw_usage, Mapping) or len(raw_usage) > 16:
            raise SessionStoreError("session compaction usage is invalid")
        usage: dict[str, int] = {}
        for key, value in raw_usage.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key) > 80
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise SessionStoreError("session compaction usage is invalid")
            usage[key] = value
        record: dict[str, Any] = {
            "schema": CONTEXT_COMPACTION_SCHEMA,
            "compaction_id": compaction_id,
            "created_at": created_at[:80],
            "source_first_message_id": covered[0]["message_id"],
            "source_last_message_id": covered[-1]["message_id"],
            "source_message_count": source_message_count,
            "source_messages_sha256": expected_prefix_sha256,
            "parent_compaction_id": previous_id,
            "parent_summary_sha256": expected_parent_summary,
            "summary": summary,
            "summary_sha256": expected_summary_sha256,
            "provider": provider.strip()[:80],
            "model": model.strip()[:160],
            "prompt_version": "context_summary.v1",
            "trigger": trigger,
            "usage": usage,
        }
        if raw.get("prompt_version") != "context_summary.v1":
            raise SessionStoreError("session compaction prompt version is invalid")
        if instruction_digest is not None:
            record["instructions_sha256"] = instruction_digest
        provider_request_id = raw.get("provider_request_id")
        if provider_request_id is not None:
            if not isinstance(provider_request_id, str) or not provider_request_id.strip():
                raise SessionStoreError("session compaction provider request id is invalid")
            record["provider_request_id"] = provider_request_id.strip()[:160]
        result.append(record)
        previous_id = compaction_id
        previous_count = source_message_count
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
    created_at = value.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        created_at = "1970-01-01T00:00:00Z"
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
    messages = [
        _validate_message(
            item,
            session_id=session_id,
            index=index,
            fallback_timestamp=created_at,
        )
        for index, item in enumerate(raw_messages)
        if isinstance(item, Mapping)
    ]
    if len(messages) != len(raw_messages):
        raise SessionStoreError("session message is invalid")
    if len({item["message_id"] for item in messages}) != len(messages):
        raise SessionStoreError("session message ids are not unique")
    result["messages"] = messages
    result["compactions"] = _validate_compactions(
        value.get("compactions", []),
        messages=messages,
    )
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
        base = _canonical_state_path(
            state_home
            or os.getenv("AGENT_HARNESS_HOME", "")
            or (Path.home() / ".agent-harness")
        )
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
            with _open_private_directory(
                self.root,
                create=False,
            ) as root_descriptor:
                self._lock_descriptor = os.open(
                    ".store.lock",
                    lock_flags,
                    0o600,
                    dir_fd=root_descriptor,
                )
        except OSError as exc:
            raise SessionStoreError("session lock file cannot be opened") from exc
        lock_stat = os.fstat(self._lock_descriptor)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_mode & 0o077
            or lock_stat.st_uid != os.getuid()
            or lock_stat.st_nlink != 1
        ):
            os.close(self._lock_descriptor)
            raise SessionStoreError("session lock file cannot be trusted")

    @property
    def approval_rules_path(self) -> Path:
        return self.root / "approvals.json"

    @property
    def hook_trust_path(self) -> Path:
        return self.root / "hook-trust.json"

    @property
    def mcp_trust_path(self) -> Path:
        return self.root / "mcp-trust.json"

    @property
    def mcp_catalog_path(self) -> Path:
        return self.root / "mcp-catalog.json"

    def _approval_state_is_private(self) -> bool:
        with _open_private_directory(self.root, create=False):
            pass
        try:
            self.root.relative_to(self.workspace)
        except ValueError:
            return True
        return False

    def _load_hook_trust_state(self) -> dict[str, Any]:
        path = self.hook_trust_path
        if not _assert_regular_file(path):
            return {"schema": HOOK_TRUST_SCHEMA, "hooks": {}}
        value = _read_private_json(path)
        if value.get("schema") != HOOK_TRUST_SCHEMA:
            raise SessionStoreError("unsupported hook trust schema")
        raw_hooks = value.get("hooks")
        if not isinstance(raw_hooks, Mapping):
            raise SessionStoreError("hook trust state is invalid")
        if len(raw_hooks) > _MAX_HOOK_TRUST_RECORDS:
            raise SessionStoreError("hook trust record limit exceeded")
        clean: dict[str, dict[str, str]] = {}
        for raw_hook_id, raw_record in raw_hooks.items():
            hook_id = str(raw_hook_id)
            if _HOOK_ID.fullmatch(hook_id) is None or not isinstance(
                raw_record, Mapping
            ):
                raise SessionStoreError("hook trust record is invalid")
            if set(raw_record) != {"action", "definition_sha256"}:
                raise SessionStoreError("hook trust record contains unknown fields")
            action = raw_record.get("action")
            digest = raw_record.get("definition_sha256")
            if (
                action not in {"trusted", "disabled"}
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
            ):
                raise SessionStoreError("hook trust record is invalid")
            clean[hook_id] = {
                "action": str(action),
                "definition_sha256": digest,
            }
        return {"schema": HOOK_TRUST_SCHEMA, "hooks": clean}

    def hook_trust_state(self) -> dict[str, dict[str, str]]:
        """Return content-free trust decisions for project hook definitions."""

        if not self._approval_state_is_private():
            raise SessionStoreError(
                "hook trust requires a state directory outside the workspace"
            )
        with self._locked():
            state = self._load_hook_trust_state()
            return deepcopy(state["hooks"])

    def set_hook_trust(
        self,
        hook_id: str,
        definition_sha256: str,
        *,
        action: str,
    ) -> None:
        """Persist one exact trusted/disabled decision outside the workspace."""

        if (
            not isinstance(hook_id, str)
            or _HOOK_ID.fullmatch(hook_id) is None
            or not isinstance(definition_sha256, str)
            or _SHA256.fullmatch(definition_sha256) is None
            or action not in {"trusted", "disabled"}
        ):
            raise SessionStoreError("hook trust decision is invalid")
        if not self._approval_state_is_private():
            raise SessionStoreError(
                "hook trust requires a state directory outside the workspace"
            )
        with self._locked():
            state = self._load_hook_trust_state()
            hooks = state["hooks"]
            if hook_id not in hooks and len(hooks) >= _MAX_HOOK_TRUST_RECORDS:
                raise SessionStoreError("hook trust record limit exceeded")
            hooks[hook_id] = {
                "action": action,
                "definition_sha256": definition_sha256,
            }
            _atomic_json(self.hook_trust_path, state)

    def revoke_hook_trust(self, hook_id: str) -> bool:
        if not isinstance(hook_id, str) or _HOOK_ID.fullmatch(hook_id) is None:
            raise SessionStoreError("hook id is invalid")
        if not self._approval_state_is_private():
            raise SessionStoreError(
                "hook trust requires a state directory outside the workspace"
            )
        with self._locked():
            state = self._load_hook_trust_state()
            removed = state["hooks"].pop(hook_id, None) is not None
            if removed:
                _atomic_json(self.hook_trust_path, state)
            return removed

    def _load_mcp_trust_state(self) -> dict[str, Any]:
        path = self.mcp_trust_path
        if not _assert_regular_file(path):
            return {"schema": MCP_TRUST_SCHEMA, "servers": {}}
        value = _read_private_json(path)
        if value.get("schema") != MCP_TRUST_SCHEMA:
            raise SessionStoreError("unsupported MCP trust schema")
        raw_servers = value.get("servers")
        if not isinstance(raw_servers, Mapping):
            raise SessionStoreError("MCP trust state is invalid")
        if len(raw_servers) > _MAX_MCP_TRUST_RECORDS:
            raise SessionStoreError("MCP trust record limit exceeded")
        clean: dict[str, dict[str, str]] = {}
        for raw_server_id, raw_record in raw_servers.items():
            server_id = str(raw_server_id)
            if _HOOK_ID.fullmatch(server_id) is None or not isinstance(
                raw_record, Mapping
            ):
                raise SessionStoreError("MCP trust record is invalid")
            if set(raw_record) != {"action", "definition_sha256"}:
                raise SessionStoreError("MCP trust record contains unknown fields")
            action = raw_record.get("action")
            digest = raw_record.get("definition_sha256")
            if (
                action not in {"trusted", "disabled"}
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
            ):
                raise SessionStoreError("MCP trust record is invalid")
            clean[server_id] = {
                "action": str(action),
                "definition_sha256": digest,
            }
        return {"schema": MCP_TRUST_SCHEMA, "servers": clean}

    def mcp_trust_state(self) -> dict[str, dict[str, str]]:
        """Return content-free trust decisions for project MCP servers."""

        if not self._approval_state_is_private():
            raise SessionStoreError(
                "MCP trust requires a state directory outside the workspace"
            )
        with self._locked():
            return deepcopy(self._load_mcp_trust_state()["servers"])

    def set_mcp_trust(
        self,
        server_id: str,
        definition_sha256: str,
        *,
        action: str,
    ) -> None:
        if (
            not isinstance(server_id, str)
            or _HOOK_ID.fullmatch(server_id) is None
            or not isinstance(definition_sha256, str)
            or _SHA256.fullmatch(definition_sha256) is None
            or action not in {"trusted", "disabled"}
        ):
            raise SessionStoreError("MCP trust decision is invalid")
        if not self._approval_state_is_private():
            raise SessionStoreError(
                "MCP trust requires a state directory outside the workspace"
            )
        with self._locked():
            state = self._load_mcp_trust_state()
            servers = state["servers"]
            if server_id not in servers and len(servers) >= _MAX_MCP_TRUST_RECORDS:
                raise SessionStoreError("MCP trust record limit exceeded")
            servers[server_id] = {
                "action": action,
                "definition_sha256": definition_sha256,
            }
            _atomic_json(self.mcp_trust_path, state)

    def revoke_mcp_trust(self, server_id: str) -> bool:
        if not isinstance(server_id, str) or _HOOK_ID.fullmatch(server_id) is None:
            raise SessionStoreError("MCP server id is invalid")
        if not self._approval_state_is_private():
            raise SessionStoreError(
                "MCP trust requires a state directory outside the workspace"
            )
        with self._locked():
            state = self._load_mcp_trust_state()
            removed = state["servers"].pop(server_id, None) is not None
            if removed:
                _atomic_json(self.mcp_trust_path, state)
            return removed

    @staticmethod
    def _clean_mcp_catalog_record(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise SessionStoreError("MCP catalog record is invalid")
        required = {
            "schema",
            "definition_sha256",
            "protocol_version",
            "catalog_sha256",
            "server_info_sha256",
            "instructions_sha256",
            "tools",
            "rejected_tools",
        }
        allowed = required | {"refreshed_at"}
        if set(raw) - allowed or not required.issubset(raw):
            raise SessionStoreError("MCP catalog record contains unknown fields")
        if raw.get("schema") != "agent_harness.mcp_catalog_record.v1":
            raise SessionStoreError("MCP catalog record schema is invalid")
        for field_name in (
            "definition_sha256",
            "catalog_sha256",
            "server_info_sha256",
            "instructions_sha256",
        ):
            value = raw.get(field_name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise SessionStoreError("MCP catalog digest is invalid")
        protocol_version = raw.get("protocol_version")
        if not isinstance(protocol_version, str) or len(protocol_version) > 40:
            raise SessionStoreError("MCP catalog protocol version is invalid")
        raw_tools = raw.get("tools")
        rejected = raw.get("rejected_tools")
        if (
            not isinstance(raw_tools, list)
            or len(raw_tools) > 128
            or not isinstance(rejected, list)
            or len(rejected) > 128
        ):
            raise SessionStoreError("MCP catalog tool lists are invalid")
        tools: list[dict[str, Any]] = []
        for item in raw_tools:
            if not isinstance(item, Mapping) or set(item) != {
                "raw_name",
                "local_name",
                "input_schema",
                "output_schema",
                "definition_sha256",
            }:
                raise SessionStoreError("MCP cached tool is invalid")
            if (
                not isinstance(item.get("raw_name"), str)
                or not isinstance(item.get("local_name"), str)
                or not isinstance(item.get("input_schema"), Mapping)
                or item.get("output_schema") is not None
                and not isinstance(item.get("output_schema"), Mapping)
                or not isinstance(item.get("definition_sha256"), str)
                or _SHA256.fullmatch(str(item.get("definition_sha256"))) is None
            ):
                raise SessionStoreError("MCP cached tool is invalid")
            tools.append(json.loads(_canonical_json(dict(item))))
        clean_rejected: list[dict[str, str]] = []
        for item in rejected:
            if (
                not isinstance(item, Mapping)
                or set(item) != {"tool_sha256", "reason_code"}
                or not isinstance(item.get("tool_sha256"), str)
                or _SHA256.fullmatch(str(item.get("tool_sha256"))) is None
                or not isinstance(item.get("reason_code"), str)
                or not 1 <= len(str(item.get("reason_code"))) <= 128
            ):
                raise SessionStoreError("MCP rejected-tool record is invalid")
            clean_rejected.append(
                {
                    "tool_sha256": str(item["tool_sha256"]),
                    "reason_code": str(item["reason_code"]),
                }
            )
        clean: dict[str, Any] = {
            "schema": "agent_harness.mcp_catalog_record.v1",
            "definition_sha256": str(raw["definition_sha256"]),
            "protocol_version": protocol_version,
            "catalog_sha256": str(raw["catalog_sha256"]),
            "server_info_sha256": str(raw["server_info_sha256"]),
            "instructions_sha256": str(raw["instructions_sha256"]),
            "tools": tools,
            "rejected_tools": clean_rejected,
        }
        refreshed_at = raw.get("refreshed_at")
        if refreshed_at is not None:
            if not isinstance(refreshed_at, str) or len(refreshed_at) > 80:
                raise SessionStoreError("MCP catalog timestamp is invalid")
            clean["refreshed_at"] = refreshed_at
        return clean

    def _load_mcp_catalog_state(self) -> dict[str, Any]:
        path = self.mcp_catalog_path
        if not _assert_regular_file(path):
            return {"schema": MCP_CATALOG_SCHEMA, "servers": {}}
        value = _read_private_json(path)
        if value.get("schema") != MCP_CATALOG_SCHEMA:
            raise SessionStoreError("unsupported MCP catalog schema")
        raw_servers = value.get("servers")
        if not isinstance(raw_servers, Mapping):
            raise SessionStoreError("MCP catalog state is invalid")
        if len(raw_servers) > _MAX_MCP_CATALOG_SERVERS:
            raise SessionStoreError("MCP catalog server limit exceeded")
        clean: dict[str, dict[str, Any]] = {}
        for raw_server_id, raw_record in raw_servers.items():
            server_id = str(raw_server_id)
            if _HOOK_ID.fullmatch(server_id) is None:
                raise SessionStoreError("MCP catalog server id is invalid")
            clean[server_id] = self._clean_mcp_catalog_record(raw_record)
        return {"schema": MCP_CATALOG_SCHEMA, "servers": clean}

    def mcp_catalog_state(self) -> dict[str, dict[str, Any]]:
        if not self._approval_state_is_private():
            raise SessionStoreError(
                "MCP catalog requires a state directory outside the workspace"
            )
        with self._locked():
            return deepcopy(self._load_mcp_catalog_state()["servers"])

    def set_mcp_catalog(self, server_id: str, record: Mapping[str, Any]) -> None:
        if not isinstance(server_id, str) or _HOOK_ID.fullmatch(server_id) is None:
            raise SessionStoreError("MCP catalog server id is invalid")
        if not self._approval_state_is_private():
            raise SessionStoreError(
                "MCP catalog requires a state directory outside the workspace"
            )
        clean = self._clean_mcp_catalog_record(record)
        clean["refreshed_at"] = _now()
        with self._locked():
            state = self._load_mcp_catalog_state()
            servers = state["servers"]
            if server_id not in servers and len(servers) >= _MAX_MCP_CATALOG_SERVERS:
                raise SessionStoreError("MCP catalog server limit exceeded")
            servers[server_id] = clean
            _atomic_json(self.mcp_catalog_path, state)

    def _load_approval_state(self) -> dict[str, Any]:
        path = self.approval_rules_path
        if not _assert_regular_file(path):
            return {
                "schema": APPROVAL_RULES_SCHEMA,
                "workspace_rules": [],
                "session_rules": {},
            }
        value = _read_private_json(path)
        if value.get("schema") != APPROVAL_RULES_SCHEMA:
            raise SessionStoreError("unsupported approval rules schema")
        raw_workspace = value.get("workspace_rules", [])
        raw_sessions = value.get("session_rules", {})
        if not isinstance(raw_workspace, list) or not isinstance(raw_sessions, Mapping):
            raise SessionStoreError("approval rules are invalid")
        if len(raw_workspace) > _MAX_APPROVAL_RULES:
            raise SessionStoreError("approval rule limit exceeded")
        clean_sessions: dict[str, list[dict[str, str]]] = {}
        total = len(raw_workspace)

        def clean_rule(raw: Any) -> dict[str, str]:
            if not isinstance(raw, Mapping):
                raise SessionStoreError("approval rule is invalid")
            allowed = {
                "action",
                "tool_name",
                "tool_version",
                "arguments_sha256",
            }
            if set(raw) - allowed:
                raise SessionStoreError("approval rule contains unknown fields")
            try:
                return ApprovalRule(
                    action=raw.get("action"),  # type: ignore[arg-type]
                    tool_name=raw.get("tool_name"),  # type: ignore[arg-type]
                    tool_version=raw.get("tool_version"),  # type: ignore[arg-type]
                    arguments_sha256=raw.get("arguments_sha256"),  # type: ignore[arg-type]
                ).material()
            except (TypeError, ValueError) as exc:
                raise SessionStoreError("approval rule is invalid") from exc

        clean_workspace = [clean_rule(item) for item in raw_workspace]
        for raw_session_id, raw_rules in raw_sessions.items():
            session_id = str(raw_session_id)
            if _ID.fullmatch(session_id) is None or not isinstance(raw_rules, list):
                raise SessionStoreError("session approval rules are invalid")
            total += len(raw_rules)
            if total > _MAX_APPROVAL_RULES:
                raise SessionStoreError("approval rule limit exceeded")
            clean_sessions[session_id] = [clean_rule(item) for item in raw_rules]
        return {
            "schema": APPROVAL_RULES_SCHEMA,
            "workspace_rules": clean_workspace,
            "session_rules": clean_sessions,
        }

    def approval_rules(self, session_id: str) -> tuple[ApprovalRule, ...]:
        if _ID.fullmatch(session_id) is None:
            raise SessionStoreError("session id is invalid")
        if not self._approval_state_is_private():
            raise SessionStoreError(
                "approval rules require a state directory outside the workspace"
            )
        with self._locked():
            state = self._load_approval_state()
            raw = [
                *state["workspace_rules"],
                *state["session_rules"].get(session_id, []),
            ]
            return tuple(ApprovalRule(**item) for item in raw)

    def add_approval_rule(
        self,
        rule: ApprovalRule,
        *,
        scope: str,
        session_id: str,
    ) -> None:
        if not isinstance(rule, ApprovalRule):
            raise SessionStoreError("approval rule is invalid")
        if scope not in {"session", "workspace"}:
            raise SessionStoreError("approval rule scope is invalid")
        if _ID.fullmatch(session_id) is None:
            raise SessionStoreError("session id is invalid")
        if not self._approval_state_is_private():
            raise SessionStoreError(
                "approval rules require a state directory outside the workspace"
            )
        material = rule.material()
        with self._locked():
            state = self._load_approval_state()
            target = (
                state["workspace_rules"]
                if scope == "workspace"
                else state["session_rules"].setdefault(session_id, [])
            )
            if material not in target:
                if sum(
                    len(items)
                    for items in (
                        state["workspace_rules"],
                        *state["session_rules"].values(),
                    )
                ) >= _MAX_APPROVAL_RULES:
                    raise SessionStoreError("approval rule limit exceeded")
                target.append(material)
                _atomic_json(self.approval_rules_path, state)

    def clear_approval_rules(self, *, scope: str, session_id: str) -> int:
        if scope not in {"session", "workspace"}:
            raise SessionStoreError("approval rule scope is invalid")
        if _ID.fullmatch(session_id) is None:
            raise SessionStoreError("session id is invalid")
        if not self._approval_state_is_private():
            raise SessionStoreError(
                "approval rules require a state directory outside the workspace"
            )
        with self._locked():
            state = self._load_approval_state()
            if scope == "workspace":
                removed = len(state["workspace_rules"])
                state["workspace_rules"] = []
            else:
                removed = len(state["session_rules"].get(session_id, []))
                state["session_rules"].pop(session_id, None)
            if removed:
                _atomic_json(self.approval_rules_path, state)
            return removed

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
            with _open_private_directory(
                path.parent,
                create=False,
            ) as parent_descriptor:
                descriptor = os.open(
                    path.name,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
        except OSError as exc:
            raise SessionStoreError("session turn lock cannot be opened") from exc
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_mode & 0o077
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
            ):
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
            with _open_private_directory(
                path.parent,
                create=False,
            ) as parent_descriptor:
                descriptor = os.open(
                    path.name,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
        except OSError as exc:
            raise SessionStoreError("workspace run lock cannot be opened") from exc
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_mode & 0o077
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
            ):
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
            "compactions": [],
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
            path = self._path(candidate["session_id"])
            if _assert_regular_file(path):
                existing = _validate_session(
                    _read_private_json(path),
                    workspace=self.workspace,
                )
                existing_messages = existing["messages"]
                if (
                    len(candidate["messages"]) < len(existing_messages)
                    or candidate["messages"][: len(existing_messages)]
                    != existing_messages
                ):
                    raise SessionStoreError("persisted transcript is append-only")
                existing_compactions = existing["compactions"]
                if (
                    len(candidate["compactions"]) < len(existing_compactions)
                    or candidate["compactions"][: len(existing_compactions)]
                    != existing_compactions
                ):
                    raise SessionStoreError("context compaction lineage is append-only")
            _atomic_json(path, candidate)
        return deepcopy(candidate)

    def list(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        with self._locked():
            with _open_private_directory(
                self.sessions_directory,
                create=False,
            ) as directory_descriptor:
                names = tuple(os.listdir(directory_descriptor))
            for name in names:
                if not name.startswith("session_") or not name.endswith(".json"):
                    continue
                try:
                    session = self.load(Path(name).stem)
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
        turn_id: str | None = None,
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
            message: dict[str, Any] = {
                "message_id": f"message_{uuid4().hex}",
                "role": role,
                "content": content,
                "timestamp": _now(),
            }
            if run_id is not None:
                message["run_id"] = run_id
            if turn_id is not None:
                message["turn_id"] = turn_id
            if len(session["messages"]) + 1 + reserve_messages > _MAX_MESSAGES:
                raise SessionStoreError("session message limit exceeded")
            session["messages"].append(
                _validate_message(
                    message,
                    session_id=session_id,
                    index=len(session["messages"]),
                    fallback_timestamp=str(session.get("created_at", "")) or _now(),
                )
            )
            if len(session["messages"]) == 1 and role == "user":
                session["title"] = " ".join(content.strip().split())[:80] or session["title"]
            return self.save(session)

    def context_view(self, session_id: str) -> dict[str, Any]:
        """Return the model-facing summary + suffix without altering the transcript."""

        session = self.load(session_id)
        latest = session["compactions"][-1] if session["compactions"] else None
        compacted_count = int(latest["source_message_count"]) if latest else 0
        suffix = deepcopy(session["messages"][compacted_count:])
        summary = str(latest["summary"]) if latest else ""
        lineage: dict[str, Any] = {
            "schema": "agent_harness.context_lineage.v1",
            "compaction_id": latest["compaction_id"] if latest else None,
            "source_message_count": compacted_count,
            "source_messages_sha256": (
                latest["source_messages_sha256"] if latest else None
            ),
            "summary_sha256": latest["summary_sha256"] if latest else None,
            "active_message_ids": [item["message_id"] for item in suffix],
        }
        active_material = {
            "summary_sha256": lineage["summary_sha256"],
            "source_messages_sha256": lineage["source_messages_sha256"],
            "messages": [
                {
                    "message_id": item["message_id"],
                    "role": item["role"],
                    "content_sha256": item["content_sha256"],
                }
                for item in suffix
            ],
        }
        lineage["active_context_sha256"] = sha256(
            _canonical_json(active_material).encode("utf-8")
        ).hexdigest()
        return {
            "schema": "agent_harness.context_view.v1",
            "session_id": session_id,
            "transcript_message_count": len(session["messages"]),
            "compacted_message_count": compacted_count,
            "active_message_count": len(suffix),
            "summary": summary,
            "messages": suffix,
            "lineage": lineage,
        }

    def record_compaction(
        self,
        session_id: str,
        *,
        summary: str,
        source_message_count: int,
        provider: str,
        model: str,
        trigger: str,
        usage: Mapping[str, Any] | None = None,
        instructions_sha256: str | None = None,
        provider_request_id: str | None = None,
    ) -> dict[str, Any]:
        if trigger not in {"manual", "automatic"}:
            raise SessionStoreError("context compaction trigger is invalid")
        clean_summary = str(summary).strip()
        if not clean_summary or len(clean_summary) > _MAX_COMPACTION_SUMMARY_CHARS:
            raise SessionStoreError("context compaction summary is invalid")
        with self._locked():
            session = self.load(session_id)
            if session["archived"]:
                raise SessionStoreError("session is archived; fork it before compacting")
            if any(
                item.get("status") == "running"
                or item.get("requires_reconciliation") is True
                for item in session["runs"]
            ):
                raise SessionStoreError(
                    "cannot compact a session with an unfinished or uncertain run"
                )
            if len(session["compactions"]) >= _MAX_COMPACTIONS:
                raise SessionStoreError("context compaction limit exceeded")
            if (
                isinstance(source_message_count, bool)
                or not isinstance(source_message_count, int)
                or not 1 <= source_message_count <= len(session["messages"])
            ):
                raise SessionStoreError("context compaction source range is invalid")
            covered = session["messages"][:source_message_count]
            if covered[-1]["role"] != "assistant":
                raise SessionStoreError(
                    "context compaction source must end at an assistant boundary"
                )
            parent = session["compactions"][-1] if session["compactions"] else None
            if parent and source_message_count <= int(parent["source_message_count"]):
                raise SessionStoreError("context compaction must advance its lineage")
            record: dict[str, Any] = {
                "schema": CONTEXT_COMPACTION_SCHEMA,
                "compaction_id": f"compaction_{uuid4().hex}",
                "created_at": _now(),
                "source_first_message_id": covered[0]["message_id"],
                "source_last_message_id": covered[-1]["message_id"],
                "source_message_count": source_message_count,
                "source_messages_sha256": _message_prefix_sha256(covered),
                "parent_compaction_id": parent["compaction_id"] if parent else None,
                "parent_summary_sha256": parent["summary_sha256"] if parent else None,
                "summary": clean_summary,
                "summary_sha256": sha256(clean_summary.encode("utf-8")).hexdigest(),
                "provider": str(provider).strip()[:80],
                "model": str(model).strip()[:160],
                "prompt_version": "context_summary.v1",
                "trigger": trigger,
                "usage": dict(usage or {}),
            }
            if instructions_sha256 is not None:
                record["instructions_sha256"] = instructions_sha256
            if provider_request_id is not None:
                record["provider_request_id"] = provider_request_id
            session["compactions"].append(record)
            if usage:
                current = session.setdefault("usage", {})
                for key, value in usage.items():
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        current[str(key)] = int(current.get(str(key), 0)) + value
            saved = self.save(session)
            return deepcopy(saved["compactions"][-1])

    def begin_run(
        self,
        session_id: str,
        *,
        run_id: str,
        turn_id: str,
        user_content: str,
        provider: str,
        model: str,
        permission_mode: str,
    ) -> dict[str, Any]:
        """Atomically append the user turn and its unresolved run fence."""

        if _ID.fullmatch(run_id) is None or _ID.fullmatch(turn_id) is None:
            raise SessionStoreError("run identity is invalid")
        with self._locked():
            session = self.load(session_id)
            if session["archived"]:
                raise SessionStoreError("session is archived; fork it before continuing")
            if any(
                item.get("status") == "running"
                or item.get("requires_reconciliation") is True
                for item in session["runs"]
            ):
                raise SessionStoreError("session already has an unresolved run")
            if len(session["messages"]) + 2 > _MAX_MESSAGES:
                raise SessionStoreError("session message limit exceeded")
            if len(session["runs"]) >= _MAX_MESSAGES:
                raise SessionStoreError("session run limit exceeded")
            if any(item.get("run_id") == run_id for item in session["runs"]):
                raise SessionStoreError("session run identity conflict")
            timestamp = _now()
            message = _validate_message(
                {
                    "message_id": f"message_{uuid4().hex}",
                    "role": "user",
                    "content": user_content,
                    "timestamp": timestamp,
                    "run_id": run_id,
                    "turn_id": turn_id,
                },
                session_id=session_id,
                index=len(session["messages"]),
                fallback_timestamp=timestamp,
            )
            session["messages"].append(message)
            if len(session["messages"]) == 1:
                session["title"] = (
                    " ".join(str(user_content).strip().split())[:80]
                    or session["title"]
                )
            session["runs"].append(
                {
                    "run_id": run_id,
                    "turn_id": turn_id,
                    "status": "running",
                    "started_at": timestamp,
                    "requires_reconciliation": False,
                    "effects": [],
                }
            )
            session["provider"] = str(provider).strip()[:80]
            session["model"] = str(model).strip()[:160]
            session["permission_mode"] = str(permission_mode).strip()[:80]
            return self.save(session)

    def finish_run(
        self,
        session_id: str,
        *,
        run_id: str,
        turn_id: str,
        status: str,
        assistant_content: str | None = None,
        usage: Mapping[str, Any] | None = None,
        effects: Sequence[Mapping[str, Any]] | None = None,
        requires_reconciliation: bool = False,
    ) -> dict[str, Any]:
        """Atomically settle a run and append its assistant answer, if any."""

        if status not in {"completed", "failed", "cancelled", "handoff"}:
            raise SessionStoreError("session run status is invalid")
        if not isinstance(requires_reconciliation, bool):
            raise SessionStoreError("session run reconciliation flag is invalid")
        with self._locked():
            session = self.load(session_id)
            matches = [item for item in session["runs"] if item.get("run_id") == run_id]
            if len(matches) != 1 or matches[0].get("turn_id") != turn_id:
                raise SessionStoreError("session run identity conflict")
            run = matches[0]
            if run.get("status") != "running":
                raise SessionStoreError("session run is already settled")
            run["status"] = status
            run["requires_reconciliation"] = requires_reconciliation
            run["effects"] = [dict(item) for item in effects or ()]
            run["finished_at"] = _now()
            clean_assistant = (
                str(assistant_content).strip()
                if assistant_content is not None
                else ""
            )
            if status == "completed" and clean_assistant:
                if len(session["messages"]) >= _MAX_MESSAGES:
                    raise SessionStoreError("session message limit exceeded")
                timestamp = _now()
                session["messages"].append(
                    _validate_message(
                        {
                            "message_id": f"message_{uuid4().hex}",
                            "role": "assistant",
                            "content": clean_assistant,
                            "timestamp": timestamp,
                            "run_id": run_id,
                            "turn_id": turn_id,
                        },
                        session_id=session_id,
                        index=len(session["messages"]),
                        fallback_timestamp=timestamp,
                    )
                )
            if usage:
                current = session.setdefault("usage", {})
                for key, value in usage.items():
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        current[str(key)] = int(current.get(str(key), 0)) + value
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
            forked["compactions"] = deepcopy(source["compactions"])
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


__all__ = [
    "APPROVAL_RULES_SCHEMA",
    "CONTEXT_COMPACTION_SCHEMA",
    "SESSION_SCHEMA",
    "SessionStore",
    "SessionStoreError",
]
