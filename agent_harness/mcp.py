"""Exact-trust MCP stdio catalog management and tool bridging.

Project MCP configuration is only a proposal.  A server can run after its
exact definition digest is trusted outside the workspace.  Tool catalogs are
refreshed explicitly and frozen into private state; normal provider requests
never start an untrusted server or discover a live tool surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import select
import stat
import subprocess
import tempfile
from threading import Lock, Thread
import time
from typing import Any, Mapping

from . import __version__
from .core import (
    CancellationToken,
    HarnessCancelled,
    RetryPolicy,
    ToolExecutionContext,
    ToolExecutionError,
    ToolRegistry,
    ToolSpec,
)
from .core.events import canonical_json, canonical_sha256
from .core.mcp_protocol import (
    MAX_MCP_CATALOG_BYTES,
    MAX_MCP_FRAME_BYTES,
    MAX_MCP_LIST_PAGES,
    MAX_MCP_TOOLS_PER_SERVER,
    MCP_CLIENT_NAME,
    MCP_PROTOCOL_VERSION,
    McpProtocolError,
    McpToolDefinition,
    decode_mcp_frame,
    encode_mcp_frame,
    normalize_tool_result,
    parse_server_message,
    parse_tool_page,
    response_result,
)
from .session import SessionStore
from .toolsets import (
    mcp_stdio_sandbox_launch,
    terminate_managed_process_group,
    workspace_sandbox_status,
)


MCP_CONFIG_SCHEMA = "agent_harness.mcp.v1"
MCP_SNAPSHOT_SCHEMA = "agent_harness.mcp_snapshot.v1"
MCP_CATALOG_RECORD_SCHEMA = "agent_harness.mcp_catalog_record.v1"
MCP_POLICY_SCHEMA = "agent_harness.mcp_policy.v1"
MCP_CONFIG_RELATIVE_PATH = ".agent-harness/mcp.json"
_SANDBOX_POLICY_VERSION = "macos-seatbelt-mcp-readonly-v1"

_SERVER_ID = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONFIG_BYTES = 128 * 1024
_MAX_SERVERS = 16
_MAX_ARGS = 64
_MAX_ARG_CHARS = 4_000
_MAX_COMMAND_BYTES = 64 * 1024 * 1024
_MAX_PASS_ENV = 32
_MAX_STDERR_BYTES = 64 * 1024
_MAX_ACTIVE_MCP_TOOLS = 256
_MAX_ACTIVE_MCP_CATALOG_BYTES = 4 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_TERM_GRACE_SECONDS = 0.25
_CONTROL_WRITE_TIMEOUT_SECONDS = 0.1
_FORBIDDEN_ENV = frozenset(
    {
        "AGENT_HARNESS",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_OPTIONAL_LOCKS",
        "GIT_TERMINAL_PROMPT",
        "HARNESS_DEEPSEEK_API_KEY",
        "HARNESS_DEEPSEEK_API_KEY_FILE",
        "HOME",
        "OPENAI_API_KEY",
        "PATH",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "ZDOTDIR",
    }
)
_FORBIDDEN_CODE_LOADING_ENV = frozenset(
    {
        "BASHOPTS",
        "BASH_ENV",
        "CDPATH",
        "CLASSPATH",
        "DOTNET_STARTUP_HOOKS",
        "ENV",
        "GCONV_PATH",
        "GEM_HOME",
        "GEM_PATH",
        "JAVA_TOOL_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "LUA_CPATH",
        "LUA_PATH",
        "NODE_OPTIONS",
        "NODE_PATH",
        "PERL5LIB",
        "PERL5OPT",
        "PHPRC",
        "PHP_INI_SCAN_DIR",
        "PYTHONBREAKPOINT",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "RUBYLIB",
        "RUBYOPT",
        "SHELLOPTS",
        "_JAVA_OPTIONS",
    }
)
_FORBIDDEN_CODE_LOADING_PREFIXES = ("COR_", "CORECLR_", "DYLD_", "LD_")


class McpLoadError(RuntimeError):
    """Raised when MCP configuration, trust or catalog state is unsafe."""


class McpTrustRequired(McpLoadError):
    def __init__(self, server_ids: tuple[str, ...]) -> None:
        self.server_ids = server_ids
        rendered = ", ".join(server_ids[:8]) + ("…" if len(server_ids) > 8 else "")
        super().__init__(
            "project MCP servers require an exact trust or disable decision: "
            f"{rendered}; run `harness mcp` for full digests"
        )


class McpRefreshRequired(McpLoadError):
    def __init__(self, server_ids: tuple[str, ...]) -> None:
        self.server_ids = server_ids
        rendered = ", ".join(server_ids[:8]) + ("…" if len(server_ids) > 8 else "")
        super().__init__(
            "trusted MCP servers require a frozen tool catalog: "
            f"{rendered}; run `harness mcp refresh SERVER`"
        )


def _strict_json(text: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate object key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> Any:
        raise ValueError("non-finite JSON number")

    try:
        return json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise McpLoadError("MCP config must be strict JSON") from exc


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise McpLoadError("secure no-follow MCP config reads are unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _file_flags() -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _executable_is_safe(metadata: os.stat_result) -> bool:
    return bool(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid in {0, os.getuid()}
        and (metadata.st_uid == 0 or metadata.st_nlink == 1)
        and not metadata.st_mode & 0o022
        and metadata.st_mode & stat.S_IXUSR
        and 1 <= metadata.st_size <= _MAX_COMMAND_BYTES
    )


def _executable_identity(metadata: os.stat_result) -> bytes:
    return canonical_json(
        {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        }
    ).encode("utf-8")


def _is_root_anchored_system_path(path: Path, metadata: os.stat_result) -> bool:
    """Whether the current user lacks authority to replace any path component."""

    if metadata.st_uid != 0 or os.access(path, os.W_OK):
        return False
    current = path.parent
    while True:
        try:
            parent = current.lstat()
        except OSError:
            return False
        if (
            stat.S_ISLNK(parent.st_mode)
            or not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != 0
            or parent.st_mode & 0o022
            or os.access(current, os.W_OK)
        ):
            return False
        if current.parent == current:
            return True
        current = current.parent


def _safe_relative(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise McpLoadError(f"MCP {label} path is invalid")
    pure = PurePosixPath(value.strip().replace("\\", "/"))
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise McpLoadError(f"MCP {label} path must stay inside the workspace")
    return tuple(pure.parts)


def _read_project_config(workspace: Path) -> bytes | None:
    parts = _safe_relative(MCP_CONFIG_RELATIVE_PATH, label="config")
    try:
        current = os.open(workspace, _directory_flags())
    except OSError as exc:
        raise McpLoadError("workspace cannot be opened securely for MCP") from exc
    try:
        for component in parts[:-1]:
            try:
                next_descriptor = os.open(component, _directory_flags(), dir_fd=current)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise McpLoadError("MCP config directory cannot be opened securely") from exc
            metadata = os.fstat(next_descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o022
            ):
                os.close(next_descriptor)
                raise McpLoadError("MCP config directory permissions are unsafe")
            os.close(current)
            current = next_descriptor
        try:
            expected = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise McpLoadError("MCP config cannot be inspected securely") from exc
        if (
            stat.S_ISLNK(expected.st_mode)
            or not stat.S_ISREG(expected.st_mode)
            or expected.st_uid != os.getuid()
            or expected.st_nlink != 1
            or expected.st_mode & 0o022
            or expected.st_size > _MAX_CONFIG_BYTES
        ):
            raise McpLoadError("MCP config type, owner, mode or size is unsafe")
        try:
            descriptor = os.open(parts[-1], _file_flags(), dir_fd=current)
        except OSError as exc:
            raise McpLoadError("MCP config cannot be opened securely") from exc
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != expected.st_dev
                or opened.st_ino != expected.st_ino
                or opened.st_size != expected.st_size
                or opened.st_mtime_ns != expected.st_mtime_ns
                or opened.st_ctime_ns != expected.st_ctime_ns
            ):
                raise McpLoadError("MCP config changed before reading")
            chunks: list[bytes] = []
            unread = opened.st_size
            while unread:
                chunk = os.read(descriptor, min(unread, _READ_CHUNK_BYTES))
                if not chunk:
                    raise McpLoadError("MCP config changed while reading")
                chunks.append(chunk)
                unread -= len(chunk)
            final = os.fstat(descriptor)
            if (
                final.st_dev != opened.st_dev
                or final.st_ino != opened.st_ino
                or final.st_size != opened.st_size
                or final.st_mtime_ns != opened.st_mtime_ns
                or final.st_ctime_ns != opened.st_ctime_ns
            ):
                raise McpLoadError("MCP config changed while reading")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(current)


def _stable_executable(path: Path) -> tuple[str, bytes]:
    try:
        expected = path.lstat()
    except OSError as exc:
        raise McpLoadError("MCP command does not exist") from exc
    if (
        stat.S_ISLNK(expected.st_mode)
        or not _executable_is_safe(expected)
    ):
        raise McpLoadError("MCP command type, owner, mode or size is unsafe")
    flags = _file_flags()
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise McpLoadError("MCP command cannot be opened securely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != expected.st_dev
            or opened.st_ino != expected.st_ino
            or opened.st_size != expected.st_size
            or opened.st_mtime_ns != expected.st_mtime_ns
            or opened.st_ctime_ns != expected.st_ctime_ns
        ):
            raise McpLoadError("MCP command changed before reading")
        digest = sha256()
        unread = opened.st_size
        while unread:
            chunk = os.read(descriptor, min(unread, _READ_CHUNK_BYTES))
            if not chunk:
                raise McpLoadError("MCP command changed while reading")
            digest.update(chunk)
            unread -= len(chunk)
        final = os.fstat(descriptor)
        if (
            final.st_dev != opened.st_dev
            or final.st_ino != opened.st_ino
            or final.st_size != opened.st_size
            or final.st_mtime_ns != opened.st_mtime_ns
            or final.st_ctime_ns != opened.st_ctime_ns
        ):
            raise McpLoadError("MCP command changed while reading")
        return digest.hexdigest(), _executable_identity(opened)
    finally:
        os.close(descriptor)


def _resolve_command(workspace: Path, raw: Any) -> tuple[Path, str, str]:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise McpLoadError("MCP command must be an explicit path")
    candidate = Path(raw.strip()).expanduser()
    if not candidate.is_absolute():
        parts = _safe_relative(raw, label="command")
        current = workspace
        for part in parts:
            current = current / part
            try:
                if current.is_symlink():
                    raise McpLoadError("workspace MCP command cannot use symbolic links")
            except OSError as exc:
                raise McpLoadError("workspace MCP command cannot be inspected") from exc
        candidate = workspace.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise McpLoadError("MCP command path cannot be resolved") from exc
    digest, identity = _stable_executable(resolved)
    return resolved, digest, sha256(identity).hexdigest()


def _resolve_cwd(workspace: Path, raw: Any) -> Path:
    value = "." if raw is None else raw
    if value == ".":
        return workspace
    parts = _safe_relative(value, label="cwd")
    current = workspace
    for part in parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise McpLoadError("MCP cwd does not exist") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise McpLoadError("MCP cwd cannot use symbolic links")
    resolved = current.resolve(strict=True)
    if not resolved.is_dir():
        raise McpLoadError("MCP cwd must be a directory")
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise McpLoadError("MCP cwd escaped the workspace") from exc
    return resolved


def _bounded_arg(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_ARG_CHARS
        or "\x00" in value
        or any(ord(character) < 32 and character not in {"\t"} for character in value)
    ):
        raise McpLoadError("MCP command argument is invalid")
    return value


@dataclass(frozen=True, slots=True)
class McpServerDefinition:
    server_id: str
    transport: str
    command: str
    args: tuple[str, ...]
    cwd: str
    pass_env: tuple[str, ...]
    network_access: bool
    allow_process_fork: bool
    startup_timeout_seconds: float
    tool_timeout_seconds: float
    config_sha256: str
    command_sha256: str
    command_identity_sha256: str
    definition_sha256: str
    workspace_root: str = field(repr=False, compare=False)

    @property
    def argv(self) -> list[str]:
        return [self.command, *self.args]

    def revalidate_command(self) -> None:
        digest, identity = _stable_executable(Path(self.command))
        if (
            digest != self.command_sha256
            or sha256(identity).hexdigest() != self.command_identity_sha256
        ):
            raise McpLoadError("MCP command changed after its trust digest was computed")

    def environment_values(self) -> dict[str, str]:
        missing = [name for name in self.pass_env if name not in os.environ]
        if missing:
            raise McpLoadError(
                "MCP server is missing explicitly allowed environment values: "
                + ", ".join(missing)
            )
        return {name: os.environ[name] for name in self.pass_env}

    def metadata(self, *, trust_status: str, catalog_status: str, catalog: Mapping[str, Any] | None) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "transport": self.transport,
            "command": self.command,
            "args": list(self.args),
            "cwd": self.cwd,
            "pass_env": list(self.pass_env),
            "network_access": self.network_access,
            "allow_process_fork": self.allow_process_fork,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "tool_timeout_seconds": self.tool_timeout_seconds,
            "command_sha256": self.command_sha256,
            "definition_sha256": self.definition_sha256,
            "trust_status": trust_status,
            "catalog_status": catalog_status,
            "catalog_sha256": str(catalog.get("catalog_sha256", "")) if catalog else "",
            "tool_count": len(catalog.get("tools", [])) if catalog else 0,
            "rejected_tool_count": len(catalog.get("rejected_tools", [])) if catalog else 0,
            "sandbox": {
                "backend": "macos-seatbelt",
                "workspace_writable": False,
                "protected_project_data_readable": False,
                "network_allowed": self.network_access,
                "process_fork_allowed": self.allow_process_fork,
                "private_runtime_home": True,
            },
            "transitive_dependency_integrity": False,
        }


def _materialize_trusted_executable(
    definition: McpServerDefinition,
    runtime: Path,
) -> Path:
    """Return a race-resistant path for exactly verified command bytes.

    Mutable user-owned commands are copied into the private per-connection
    runtime. Root-owned commands whose complete canonical ancestry is also
    root-owned and non-writable remain at their platform path because macOS
    platform binaries may not be executable after copying.
    """

    source_descriptor = -1
    runtime_descriptor = -1
    destination_descriptor = -1
    destination_name = "mcp-command"
    completed = False
    try:
        source_path = Path(definition.command)
        expected = source_path.lstat()
        if stat.S_ISLNK(expected.st_mode) or not _executable_is_safe(expected):
            raise McpLoadError("MCP command is no longer a safe executable")
        source_descriptor = os.open(source_path, _file_flags())
        opened = os.fstat(source_descriptor)
        if (
            opened.st_dev != expected.st_dev
            or opened.st_ino != expected.st_ino
            or opened.st_size != expected.st_size
            or opened.st_mtime_ns != expected.st_mtime_ns
            or opened.st_ctime_ns != expected.st_ctime_ns
            or not _executable_is_safe(opened)
            or sha256(_executable_identity(opened)).hexdigest()
            != definition.command_identity_sha256
        ):
            raise McpLoadError("MCP command identity changed before materialization")
        root_anchored = _is_root_anchored_system_path(source_path, opened)
        if not root_anchored:
            runtime_descriptor = os.open(runtime, _directory_flags())
            destination_flags = (
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            )
            if hasattr(os, "O_CLOEXEC"):
                destination_flags |= os.O_CLOEXEC
            destination_descriptor = os.open(
                destination_name,
                destination_flags,
                0o500,
                dir_fd=runtime_descriptor,
            )
        digest = sha256()
        unread = opened.st_size
        while unread:
            chunk = os.read(source_descriptor, min(unread, _READ_CHUNK_BYTES))
            if not chunk:
                raise McpLoadError("MCP command changed while materializing")
            digest.update(chunk)
            if destination_descriptor >= 0:
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_descriptor, view)
                    if written <= 0:
                        raise McpLoadError("MCP command copy could not be completed")
                    view = view[written:]
            unread -= len(chunk)
        final_source = os.fstat(source_descriptor)
        if (
            final_source.st_dev != opened.st_dev
            or final_source.st_ino != opened.st_ino
            or final_source.st_size != opened.st_size
            or final_source.st_mtime_ns != opened.st_mtime_ns
            or final_source.st_ctime_ns != opened.st_ctime_ns
            or digest.hexdigest() != definition.command_sha256
        ):
            raise McpLoadError("MCP command changed while materializing")
        if root_anchored:
            completed = True
            return source_path
        os.fchmod(destination_descriptor, 0o500)
        os.fsync(destination_descriptor)
        copied = os.fstat(destination_descriptor)
        if (
            not stat.S_ISREG(copied.st_mode)
            or copied.st_uid != os.getuid()
            or copied.st_nlink != 1
            or copied.st_mode & 0o7777 != 0o500
            or copied.st_size != opened.st_size
        ):
            raise McpLoadError("materialized MCP command is unsafe")
        os.fsync(runtime_descriptor)
        completed = True
        return runtime / destination_name
    except McpLoadError:
        raise
    except OSError as exc:
        raise McpLoadError("MCP command could not be materialized securely") from exc
    finally:
        for descriptor in (destination_descriptor, source_descriptor):
            if descriptor >= 0:
                os.close(descriptor)
        if not completed and runtime_descriptor >= 0:
            try:
                os.unlink(destination_name, dir_fd=runtime_descriptor)
            except FileNotFoundError:
                pass
        if runtime_descriptor >= 0:
            os.close(runtime_descriptor)


@dataclass(frozen=True, slots=True)
class McpSnapshot:
    workspace_root: str
    config_present: bool
    config_sha256: str
    servers: tuple[McpServerDefinition, ...]
    snapshot_sha256: str
    schema: str = MCP_SNAPSHOT_SCHEMA

    def server(self, server_id: str) -> McpServerDefinition:
        for definition in self.servers:
            if definition.server_id == server_id:
                return definition
        raise McpLoadError(f"project MCP server is not configured: {server_id}")

    def statuses(self, trust_state: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for definition in self.servers:
            record = trust_state.get(definition.server_id)
            if not isinstance(record, Mapping):
                statuses[definition.server_id] = "untrusted"
            elif record.get("definition_sha256") != definition.definition_sha256:
                statuses[definition.server_id] = "modified"
            elif record.get("action") in {"trusted", "disabled"}:
                statuses[definition.server_id] = str(record["action"])
            else:
                statuses[definition.server_id] = "untrusted"
        return statuses

    def unresolved(self, trust_state: Mapping[str, Mapping[str, str]]) -> tuple[str, ...]:
        statuses = self.statuses(trust_state)
        return tuple(
            item.server_id
            for item in self.servers
            if statuses[item.server_id] in {"untrusted", "modified"}
        )

    def metadata(self, trust_state: Mapping[str, Mapping[str, str]], catalogs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        statuses = self.statuses(trust_state)
        items: list[dict[str, Any]] = []
        active_tool_count = 0
        active_catalog_bytes = 0
        catalog_budget_exceeded = False
        for definition in self.servers:
            trust_status = statuses[definition.server_id]
            raw_catalog = catalogs.get(definition.server_id)
            catalog = raw_catalog if isinstance(raw_catalog, Mapping) else None
            if trust_status != "trusted":
                catalog_status = "not_active"
            elif catalog is None:
                catalog_status = "refresh_required"
            elif catalog.get("definition_sha256") != definition.definition_sha256:
                catalog_status = "stale"
            else:
                try:
                    cached_tools = _validated_cached_tools(definition, catalog)
                    catalog_bytes = _cached_catalog_bytes(catalog)
                except McpLoadError:
                    catalog_status = "invalid"
                else:
                    if (
                        catalog_budget_exceeded
                        or active_tool_count + len(cached_tools)
                        > _MAX_ACTIVE_MCP_TOOLS
                        or active_catalog_bytes + catalog_bytes
                        > _MAX_ACTIVE_MCP_CATALOG_BYTES
                    ):
                        catalog_status = "capacity_exceeded"
                        catalog_budget_exceeded = True
                    else:
                        catalog_status = "ready"
                        active_tool_count += len(cached_tools)
                        active_catalog_bytes += catalog_bytes
            items.append(
                definition.metadata(
                    trust_status=trust_status,
                    catalog_status=catalog_status,
                    catalog=catalog,
                )
            )
        return {
            "schema": self.schema,
            "workspace_root": self.workspace_root,
            "source": MCP_CONFIG_RELATIVE_PATH,
            "config_present": self.config_present,
            "config_sha256": self.config_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "server_count": len(self.servers),
            "trusted_server_count": sum(item["trust_status"] == "trusted" for item in items),
            "disabled_server_count": sum(item["trust_status"] == "disabled" for item in items),
            "pending_server_count": sum(item["trust_status"] in {"untrusted", "modified"} for item in items),
            "ready_server_count": sum(item["catalog_status"] == "ready" for item in items),
            "refresh_required_count": sum(
                item["catalog_status"]
                in {"refresh_required", "stale", "invalid", "capacity_exceeded"}
                for item in items
            ),
            "servers": items,
            "protocol_version": MCP_PROTOCOL_VERSION,
            "transport_support": ["stdio"],
            "sandbox_available": bool(workspace_sandbox_status().get("available")),
        }


def _server_definition(workspace: Path, server_id: str, raw: Any, *, config_sha256: str) -> McpServerDefinition:
    if _SERVER_ID.fullmatch(server_id) is None or not isinstance(raw, Mapping):
        raise McpLoadError("MCP server id or definition is invalid")
    allowed = {
        "transport",
        "command",
        "args",
        "cwd",
        "pass_env",
        "network_access",
        "allow_process_fork",
        "startup_timeout_seconds",
        "tool_timeout_seconds",
    }
    if set(raw) - allowed:
        raise McpLoadError("MCP server definition contains unknown fields")
    if raw.get("transport") != "stdio":
        raise McpLoadError("only MCP stdio transport is supported")
    command, command_sha256, command_identity_sha256 = _resolve_command(workspace, raw.get("command"))
    raw_args = raw.get("args", [])
    if not isinstance(raw_args, list) or len(raw_args) > _MAX_ARGS:
        raise McpLoadError("MCP args must be a bounded array")
    args = tuple(_bounded_arg(value) for value in raw_args)
    cwd_path = _resolve_cwd(workspace, raw.get("cwd", "."))
    cwd = "." if cwd_path == workspace else cwd_path.relative_to(workspace).as_posix()
    raw_env = raw.get("pass_env", [])
    if not isinstance(raw_env, list) or len(raw_env) > _MAX_PASS_ENV:
        raise McpLoadError("MCP pass_env must be a bounded array")
    pass_env: list[str] = []
    for value in raw_env:
        if (
            not isinstance(value, str)
            or _ENV_NAME.fullmatch(value) is None
            or value in _FORBIDDEN_ENV
            or value in _FORBIDDEN_CODE_LOADING_ENV
            or value.startswith(_FORBIDDEN_CODE_LOADING_PREFIXES)
            or value.startswith("HARNESS_")
        ):
            raise McpLoadError(
                "MCP pass_env contains an invalid, reserved, or code-loading name"
            )
        if value not in pass_env:
            pass_env.append(value)
    network_access = raw.get("network_access", False)
    allow_process_fork = raw.get("allow_process_fork", False)
    if not isinstance(network_access, bool) or not isinstance(allow_process_fork, bool):
        raise McpLoadError("MCP sandbox authority flags must be boolean")
    startup_timeout = raw.get("startup_timeout_seconds", 10.0)
    tool_timeout = raw.get("tool_timeout_seconds", 30.0)
    if (
        isinstance(startup_timeout, bool)
        or not isinstance(startup_timeout, (int, float))
        or not 0.5 <= float(startup_timeout) <= 60.0
        or isinstance(tool_timeout, bool)
        or not isinstance(tool_timeout, (int, float))
        or not 1.0 <= float(tool_timeout) <= 300.0
    ):
        raise McpLoadError("MCP timeout is invalid")
    material = {
        "schema": "agent_harness.mcp_server_definition.v1",
        "server_id": server_id,
        "transport": "stdio",
        "command": str(command),
        "command_sha256": command_sha256,
        "command_identity_sha256": command_identity_sha256,
        "args": list(args),
        "cwd": cwd,
        "pass_env": sorted(pass_env),
        "network_access": network_access,
        "allow_process_fork": allow_process_fork,
        "startup_timeout_seconds": float(startup_timeout),
        "tool_timeout_seconds": float(tool_timeout),
        "config_sha256": config_sha256,
        "sandbox_policy_version": _SANDBOX_POLICY_VERSION,
    }
    return McpServerDefinition(
        server_id=server_id,
        transport="stdio",
        command=str(command),
        args=args,
        cwd=cwd,
        pass_env=tuple(sorted(pass_env)),
        network_access=network_access,
        allow_process_fork=allow_process_fork,
        startup_timeout_seconds=float(startup_timeout),
        tool_timeout_seconds=float(tool_timeout),
        config_sha256=config_sha256,
        command_sha256=command_sha256,
        command_identity_sha256=command_identity_sha256,
        definition_sha256=canonical_sha256(material),
        workspace_root=str(workspace),
    )


def load_project_mcp(workspace_root: os.PathLike[str] | str) -> McpSnapshot:
    candidate = Path(workspace_root).expanduser()
    try:
        if candidate.is_symlink():
            raise McpLoadError("workspace symlinks are not accepted for project MCP")
        workspace = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise McpLoadError("workspace must be an existing directory") from exc
    if not workspace.is_dir() or workspace.is_symlink():
        raise McpLoadError("workspace must be a real directory")
    config_bytes = _read_project_config(workspace)
    if config_bytes is None:
        material = {"schema": MCP_SNAPSHOT_SCHEMA, "config_present": False, "servers": []}
        return McpSnapshot(
            workspace_root=str(workspace),
            config_present=False,
            config_sha256=canonical_sha256(None),
            servers=(),
            snapshot_sha256=canonical_sha256(material),
        )
    try:
        value = _strict_json(config_bytes.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as exc:
        raise McpLoadError("MCP config must be valid UTF-8 JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "servers"}
        or value.get("schema") != MCP_CONFIG_SCHEMA
        or not isinstance(value.get("servers"), Mapping)
    ):
        raise McpLoadError("MCP config schema is invalid")
    raw_servers = value["servers"]
    if len(raw_servers) > _MAX_SERVERS:
        raise McpLoadError("MCP server limit exceeded")
    config_sha256 = sha256(config_bytes).hexdigest()
    servers = tuple(
        _server_definition(workspace, str(server_id), raw, config_sha256=config_sha256)
        for server_id, raw in sorted(raw_servers.items())
    )
    material = {
        "schema": MCP_SNAPSHOT_SCHEMA,
        "config_present": True,
        "config_sha256": config_sha256,
        "servers": [item.definition_sha256 for item in servers],
    }
    return McpSnapshot(
        workspace_root=str(workspace),
        config_present=True,
        config_sha256=config_sha256,
        servers=servers,
        snapshot_sha256=canonical_sha256(material),
    )


def _validated_cached_tools(
    definition: McpServerDefinition,
    catalog: Mapping[str, Any],
) -> tuple[McpToolDefinition, ...]:
    """Validate a private catalog's complete identity and server binding."""

    if (
        catalog.get("schema") != MCP_CATALOG_RECORD_SCHEMA
        or catalog.get("definition_sha256") != definition.definition_sha256
        or catalog.get("protocol_version") != MCP_PROTOCOL_VERSION
    ):
        raise McpLoadError("cached MCP catalog identity is invalid")
    for field_name in ("catalog_sha256", "server_info_sha256", "instructions_sha256"):
        value = catalog.get(field_name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise McpLoadError("cached MCP catalog digest is invalid")
    raw_tools = catalog.get("tools")
    raw_rejected = catalog.get("rejected_tools")
    if (
        not isinstance(raw_tools, list)
        or len(raw_tools) > MAX_MCP_TOOLS_PER_SERVER
        or not isinstance(raw_rejected, list)
        or len(raw_rejected) > MAX_MCP_TOOLS_PER_SERVER
        or len(raw_tools) + len(raw_rejected) > MAX_MCP_TOOLS_PER_SERVER
    ):
        raise McpLoadError("cached MCP catalog tool lists are invalid")
    tools: list[McpToolDefinition] = []
    seen_raw: set[str] = set()
    seen_local: set[str] = set()
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            raise McpLoadError("cached MCP tool is invalid")
        try:
            tool = McpToolDefinition.from_cache(definition.server_id, raw_tool)
        except McpProtocolError as exc:
            raise McpLoadError("cached MCP tool identity is invalid") from exc
        if tool.raw_name in seen_raw or tool.local_name in seen_local:
            raise McpLoadError("cached MCP tool identity is duplicated")
        seen_raw.add(tool.raw_name)
        seen_local.add(tool.local_name)
        tools.append(tool)
    frozen_tools = [item.to_cache() for item in sorted(tools, key=lambda item: item.raw_name)]
    if raw_tools != frozen_tools:
        raise McpLoadError("cached MCP tool order or representation is invalid")
    rejected: list[dict[str, str]] = []
    for raw_item in raw_rejected:
        if (
            not isinstance(raw_item, Mapping)
            or set(raw_item) != {"tool_sha256", "reason_code"}
            or not isinstance(raw_item.get("tool_sha256"), str)
            or _SHA256.fullmatch(str(raw_item.get("tool_sha256"))) is None
            or not isinstance(raw_item.get("reason_code"), str)
            or not 1 <= len(str(raw_item.get("reason_code"))) <= 128
        ):
            raise McpLoadError("cached MCP rejected-tool record is invalid")
        rejected.append(
            {
                "tool_sha256": str(raw_item["tool_sha256"]),
                "reason_code": str(raw_item["reason_code"]),
            }
        )
    frozen_rejected = sorted(
        rejected,
        key=lambda item: (item["tool_sha256"], item["reason_code"]),
    )
    if raw_rejected != frozen_rejected:
        raise McpLoadError("cached MCP rejected-tool order is invalid")
    material = {
        "schema": "agent_harness.mcp_catalog.v1",
        "protocol_version": MCP_PROTOCOL_VERSION,
        "server_definition_sha256": definition.definition_sha256,
        "tools": frozen_tools,
        "rejected_tools": frozen_rejected,
    }
    if len(canonical_json(material).encode("utf-8")) > MAX_MCP_CATALOG_BYTES:
        raise McpLoadError("cached MCP catalog exceeded its byte limit")
    if catalog.get("catalog_sha256") != canonical_sha256(material):
        raise McpLoadError("cached MCP catalog digest does not match its contents")
    return tuple(tools)


def _cached_catalog_bytes(catalog: Mapping[str, Any]) -> int:
    try:
        return len(canonical_json(dict(catalog)).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise McpLoadError("cached MCP catalog is not canonical JSON") from exc


class StdioMcpConnection:
    """One bounded legacy MCP stdio connection with strict frame demux."""

    def __init__(self, definition: McpServerDefinition) -> None:
        self.definition = definition
        self.process: subprocess.Popen[bytes] | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._stdout_buffer = bytearray()
        self._stderr_hasher = sha256()
        self._stderr_bytes = 0
        self._stderr_truncated = False
        self._stderr_thread: Thread | None = None
        self._stderr_errors: list[BaseException] = []
        self._stderr_lock = Lock()
        self._write_lock = Lock()
        self._request_lock = Lock()
        self._process_lock = Lock()
        self._termination_errors: list[BaseException] = []
        self._next_request_id = 1
        self._active_request_id: int | None = None
        self._initialized = False
        self._state = "new"
        self.catalog_stale = False
        self.server_info_sha256 = canonical_sha256(None)
        self.instructions_sha256 = canonical_sha256(None)

    @property
    def stderr_metadata(self) -> dict[str, Any]:
        with self._stderr_lock:
            return {
                "byte_length": self._stderr_bytes,
                "sha256": self._stderr_hasher.hexdigest(),
                "truncated": self._stderr_truncated,
            }

    def start(self) -> None:
        if self._state != "new" or self.process is not None:
            raise McpProtocolError("MCP connection is already started")
        self._state = "starting"
        try:
            self.definition.revalidate_command()
            environment = self.definition.environment_values()
        except McpLoadError as exc:
            self._state = "failed"
            raise McpProtocolError(
                "MCP launch preflight no longer matches the trusted definition",
                code="mcp_preflight_failed",
            ) from exc
        self._temporary = tempfile.TemporaryDirectory(prefix="agent-harness-mcp-")
        runtime = Path(self._temporary.name)
        os.chmod(runtime, 0o700)
        try:
            materialized_command = _materialize_trusted_executable(
                self.definition,
                runtime,
            )
            command, process_environment = mcp_stdio_sandbox_launch(
                self.definition.workspace_root,
                runtime,
                [str(materialized_command), *self.definition.args],
                network_allowed=self.definition.network_access,
                allow_process_fork=self.definition.allow_process_fork,
                environment_overrides=environment,
            )
        except McpLoadError as exc:
            self._temporary.cleanup()
            self._temporary = None
            self._state = "failed"
            raise McpProtocolError(
                "MCP command no longer matches the trusted executable",
                code="mcp_preflight_failed",
            ) from exc
        except ToolExecutionError as exc:
            self._temporary.cleanup()
            self._temporary = None
            self._state = "failed"
            raise McpProtocolError(
                "MCP sandbox launch could not be prepared",
                code=exc.code,
            ) from exc
        cwd = Path(self.definition.workspace_root)
        if self.definition.cwd != ".":
            cwd = cwd / self.definition.cwd
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=process_environment,
                bufsize=0,
            )
        except OSError as exc:
            self._temporary.cleanup()
            self._temporary = None
            self._state = "failed"
            raise McpProtocolError("MCP server could not be started", code="mcp_spawn_failed") from exc
        self.process = process
        assert process.stdout is not None and process.stderr is not None

        def drain_stderr() -> None:
            try:
                while True:
                    chunk = process.stderr.read(_READ_CHUNK_BYTES)
                    if not chunk:
                        return
                    with self._stderr_lock:
                        self._stderr_hasher.update(chunk)
                        self._stderr_bytes += len(chunk)
                        if self._stderr_bytes > _MAX_STDERR_BYTES:
                            self._stderr_truncated = True
            except BaseException as exc:
                with self._stderr_lock:
                    self._stderr_errors.append(exc)

        try:
            os.set_blocking(process.stdout.fileno(), False)
            os.set_blocking(process.stdin.fileno(), False)
            self._stderr_thread = Thread(
                target=drain_stderr,
                name=f"agent-harness-mcp-stderr-{self.definition.server_id}",
                daemon=True,
            )
            self._stderr_thread.start()
        except BaseException as exc:
            self._state = "failed"
            self._terminate()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
            self.process = None
            raise McpProtocolError(
                "MCP server stream setup failed",
                code="mcp_spawn_failed",
            ) from exc
        self._state = "started"

    def _write(
        self,
        value: Mapping[str, Any],
        *,
        deadline_monotonic: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            raise McpProtocolError("MCP server is not connected", code="mcp_disconnected")
        encoded = encode_mcp_frame(value)
        deadline = (
            time.monotonic() + _CONTROL_WRITE_TIMEOUT_SECONDS
            if deadline_monotonic is None
            else deadline_monotonic
        )
        if not math.isfinite(deadline):
            raise McpProtocolError("MCP write deadline is invalid", code="mcp_timeout")
        lock_acquired = False
        try:
            while not lock_acquired:
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise McpProtocolError("MCP request write timed out", code="mcp_timeout")
                lock_acquired = self._write_lock.acquire(
                    timeout=min(0.05, remaining),
                )
            descriptor = process.stdin.fileno()
            offset = 0
            while offset < len(encoded):
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                if process.poll() is not None:
                    raise McpProtocolError(
                        "MCP server disconnected while writing",
                        code="mcp_disconnected",
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise McpProtocolError("MCP request write timed out", code="mcp_timeout")
                try:
                    _, writable, _ = select.select(
                        [],
                        [descriptor],
                        [],
                        min(0.05, remaining),
                    )
                except (OSError, ValueError) as exc:
                    if cancellation_token is not None:
                        cancellation_token.raise_if_cancelled()
                    raise McpProtocolError(
                        "MCP stdin wait failed",
                        code="mcp_output_error",
                    ) from exc
                if not writable:
                    continue
                try:
                    written = os.write(descriptor, encoded[offset:])
                except BlockingIOError:
                    continue
                if written <= 0:
                    raise McpProtocolError(
                        "MCP request could not be written",
                        code="mcp_disconnected",
                    )
                offset += written
        except (BrokenPipeError, OSError) as exc:
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            raise McpProtocolError("MCP request could not be written", code="mcp_disconnected") from exc
        finally:
            if lock_acquired:
                self._write_lock.release()

    def _next_frame(self, *, deadline_monotonic: float, cancellation_token: CancellationToken) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise McpProtocolError("MCP server is not connected", code="mcp_disconnected")
        descriptor = process.stdout.fileno()
        while True:
            cancellation_token.raise_if_cancelled()
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self._stdout_buffer[:newline])
                del self._stdout_buffer[: newline + 1]
                return decode_mcp_frame(raw)
            if len(self._stdout_buffer) > MAX_MCP_FRAME_BYTES:
                raise McpProtocolError("MCP stdout frame exceeded its limit", code="mcp_frame_too_large")
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise McpProtocolError("MCP request timed out", code="mcp_timeout")
            try:
                readable, _, _ = select.select(
                    [descriptor],
                    [],
                    [],
                    min(0.05, remaining),
                )
            except (OSError, ValueError) as exc:
                cancellation_token.raise_if_cancelled()
                raise McpProtocolError(
                    "MCP stdout wait failed",
                    code="mcp_output_error",
                ) from exc
            if not readable:
                cancellation_token.raise_if_cancelled()
                if process.poll() is not None:
                    raise McpProtocolError("MCP server exited before responding", code="mcp_disconnected")
                continue
            try:
                chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            except BlockingIOError:
                continue
            except OSError as exc:
                cancellation_token.raise_if_cancelled()
                raise McpProtocolError("MCP stdout could not be read", code="mcp_output_error") from exc
            if not chunk:
                cancellation_token.raise_if_cancelled()
                raise McpProtocolError("MCP server closed stdout before responding", code="mcp_disconnected")
            self._stdout_buffer.extend(chunk)

    def _terminate_from_cancel(self) -> None:
        request_id = self._active_request_id
        if self._initialized and request_id is not None:
            try:
                self._write(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/cancelled",
                        "params": {"requestId": request_id, "reason": "user cancelled"},
                    },
                    deadline_monotonic=(
                        time.monotonic() + _CONTROL_WRITE_TIMEOUT_SECONDS
                    ),
                )
            except Exception:
                pass
        self._terminate()

    def _terminate(self) -> None:
        with self._process_lock:
            process = self.process
            if process is None:
                return
            try:
                terminate_managed_process_group(process)
            except Exception as exc:
                self._termination_errors.append(exc)
                try:
                    process.kill()
                    process.wait(timeout=1)
                except Exception as fallback_exc:
                    self._termination_errors.append(fallback_exc)

    def request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        deadline_monotonic: float,
        cancellation_token: CancellationToken,
        cancellable: bool = True,
    ) -> Mapping[str, Any]:
        if not math.isfinite(deadline_monotonic):
            raise McpProtocolError("MCP request deadline is invalid", code="mcp_timeout")
        expected_state = "initializing" if method == "initialize" else "ready"
        if self._state != expected_state:
            raise McpProtocolError(
                "MCP request violates the connection lifecycle",
                code="mcp_lifecycle_invalid",
            )
        cancellation_token.raise_if_cancelled()
        if not self._request_lock.acquire(blocking=False):
            raise McpProtocolError(
                "concurrent MCP requests are unsupported",
                code="mcp_concurrent_request",
            )
        request_id = self._next_request_id
        self._next_request_id += 1
        self._active_request_id = request_id
        cancel_callback = self._terminate_from_cancel if cancellable else self._terminate
        unsubscribe = cancellation_token.add_callback(cancel_callback)
        try:
            cancellation_token.raise_if_cancelled()
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": dict(params),
                },
                deadline_monotonic=deadline_monotonic,
                cancellation_token=cancellation_token,
            )
            incidental_frames = 0
            while True:
                frame = self._next_frame(
                    deadline_monotonic=deadline_monotonic,
                    cancellation_token=cancellation_token,
                )
                if "method" in frame:
                    incidental_frames += 1
                    if incidental_frames > 256:
                        raise McpProtocolError(
                            "MCP incidental-message flood",
                            code="mcp_notification_flood",
                        )
                    (
                        method_name,
                        server_request_id,
                        _params,
                        is_server_request,
                    ) = parse_server_message(frame)
                    if is_server_request:
                        if method_name == "ping":
                            self._write(
                                {
                                    "jsonrpc": "2.0",
                                    "id": server_request_id,
                                    "result": {},
                                },
                                deadline_monotonic=deadline_monotonic,
                                cancellation_token=cancellation_token,
                            )
                        else:
                            self._write(
                                {
                                    "jsonrpc": "2.0",
                                    "id": server_request_id,
                                    "error": {"code": -32601, "message": "Method not supported"},
                                },
                                deadline_monotonic=deadline_monotonic,
                                cancellation_token=cancellation_token,
                            )
                        continue
                    if method_name == "notifications/tools/list_changed":
                        self.catalog_stale = True
                    continue
                return response_result(frame, request_id)
        except HarnessCancelled:
            self._terminate()
            raise
        except McpProtocolError:
            self._state = "failed"
            self._terminate()
            raise
        finally:
            unsubscribe()
            self._active_request_id = None
            self._request_lock.release()

    def initialize(self, *, deadline_monotonic: float, cancellation_token: CancellationToken) -> None:
        if self._state != "started" or self._initialized:
            raise McpProtocolError(
                "MCP initialize violates the connection lifecycle",
                code="mcp_lifecycle_invalid",
            )
        self._state = "initializing"
        try:
            result = self.request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": MCP_CLIENT_NAME, "version": __version__},
                },
                deadline_monotonic=deadline_monotonic,
                cancellation_token=cancellation_token,
                cancellable=False,
            )
            if result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
                raise McpProtocolError(
                    "MCP server selected an unsupported protocol version",
                    code="mcp_version_unsupported",
                )
            capabilities = result.get("capabilities")
            server_info = result.get("serverInfo")
            if not isinstance(capabilities, Mapping) or not isinstance(server_info, Mapping):
                raise McpProtocolError("MCP initialize result is invalid")
            tools_capability = capabilities.get("tools")
            if not isinstance(tools_capability, Mapping):
                raise McpProtocolError(
                    "MCP server does not declare tools capability",
                    code="mcp_tools_unavailable",
                )
            list_changed = tools_capability.get("listChanged")
            if list_changed is not None and not isinstance(list_changed, bool):
                raise McpProtocolError("MCP tools capability is invalid")
            if list_changed is True:
                raise McpProtocolError(
                    "dynamic MCP tool catalogs are unsupported",
                    code="mcp_dynamic_catalog_unsupported",
                )
            server_name = server_info.get("name")
            server_version = server_info.get("version")
            if (
                not isinstance(server_name, str)
                or not 1 <= len(server_name) <= 200
                or not isinstance(server_version, str)
                or not 1 <= len(server_version) <= 200
            ):
                raise McpProtocolError("MCP serverInfo is invalid")
            self.server_info_sha256 = canonical_sha256(dict(server_info))
            instructions = result.get("instructions")
            if instructions is not None and not isinstance(instructions, str):
                raise McpProtocolError("MCP server instructions are invalid")
            self.instructions_sha256 = canonical_sha256(instructions)
            self._write(
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                deadline_monotonic=deadline_monotonic,
                cancellation_token=cancellation_token,
            )
        except BaseException:
            self._state = "failed"
            self._terminate()
            raise
        self._initialized = True
        self._state = "ready"

    def list_tools(self, *, deadline_monotonic: float, cancellation_token: CancellationToken) -> dict[str, Any]:
        tools: list[McpToolDefinition] = []
        rejected: list[dict[str, str]] = []
        seen_names: set[str] = set()
        seen_local: set[str] = set()
        seen_cursors: set[str] = set()
        advertised_tool_count = 0
        cursor: str | None = None
        for _page in range(MAX_MCP_LIST_PAGES):
            params = {"cursor": cursor} if cursor is not None else {}
            result = self.request(
                "tools/list",
                params,
                deadline_monotonic=deadline_monotonic,
                cancellation_token=cancellation_token,
            )
            page_tools, next_cursor, page_rejected = parse_tool_page(
                self.definition.server_id,
                result,
            )
            rejected.extend(page_rejected)
            advertised_tool_count += len(page_tools) + len(page_rejected)
            if advertised_tool_count > MAX_MCP_TOOLS_PER_SERVER:
                raise McpProtocolError(
                    "MCP tool count exceeded its limit",
                    code="mcp_catalog_too_large",
                )
            for tool in page_tools:
                if tool.raw_name in seen_names or tool.local_name in seen_local:
                    raise McpProtocolError("MCP tool names are duplicated", code="mcp_catalog_collision")
                seen_names.add(tool.raw_name)
                seen_local.add(tool.local_name)
                tools.append(tool)
                if len(tools) > MAX_MCP_TOOLS_PER_SERVER:
                    raise McpProtocolError("MCP tool count exceeded its limit", code="mcp_catalog_too_large")
            if next_cursor is None:
                break
            if next_cursor in seen_cursors:
                raise McpProtocolError("MCP tools/list cursor cycle detected", code="mcp_cursor_cycle")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        else:
            raise McpProtocolError("MCP tools/list page limit exceeded", code="mcp_catalog_too_large")
        frozen_tools = [item.to_cache() for item in sorted(tools, key=lambda item: item.raw_name)]
        rejected = sorted(rejected, key=lambda item: (item["tool_sha256"], item["reason_code"]))
        catalog_material = {
            "schema": "agent_harness.mcp_catalog.v1",
            "protocol_version": MCP_PROTOCOL_VERSION,
            "server_definition_sha256": self.definition.definition_sha256,
            "tools": frozen_tools,
            "rejected_tools": rejected,
        }
        if len(canonical_json(catalog_material).encode("utf-8")) > MAX_MCP_CATALOG_BYTES:
            raise McpProtocolError("MCP catalog exceeded its byte limit", code="mcp_catalog_too_large")
        return {
            "schema": MCP_CATALOG_RECORD_SCHEMA,
            "definition_sha256": self.definition.definition_sha256,
            "protocol_version": MCP_PROTOCOL_VERSION,
            "catalog_sha256": canonical_sha256(catalog_material),
            "server_info_sha256": self.server_info_sha256,
            "instructions_sha256": self.instructions_sha256,
            "tools": frozen_tools,
            "rejected_tools": rejected,
        }

    def call_tool(
        self,
        tool: McpToolDefinition,
        arguments: Mapping[str, Any],
        *,
        deadline_monotonic: float,
        cancellation_token: CancellationToken,
    ) -> dict[str, Any]:
        result = self.request(
            "tools/call",
            {"name": tool.raw_name, "arguments": dict(arguments)},
            deadline_monotonic=min(
                deadline_monotonic,
                time.monotonic() + self.definition.tool_timeout_seconds,
            ),
            cancellation_token=cancellation_token,
        )
        return normalize_tool_result(self.definition.server_id, tool, result)

    def close(self) -> None:
        if self._state == "closed":
            return
        self._state = "closing"
        cleanup_error: BaseException | None = None
        process = self.process
        if process is not None:
            try:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    try:
                        process.wait(timeout=_TERM_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        terminate_managed_process_group(process)
                elif process.returncode is None:
                    process.wait(timeout=_TERM_GRACE_SECONDS)
                # A server launched with explicit fork authority may let its
                # parent exit while children keep the process group alive.
                terminate_managed_process_group(process)
            except BaseException as exc:
                cleanup_error = exc
                self._terminate()
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except Exception:
                    pass
        if self._stderr_thread is not None:
            try:
                self._stderr_thread.join(0.5)
                if self._stderr_thread.is_alive() and process is not None:
                    # The process group is already gone, so EOF should settle
                    # the reader. Close only as a fallback to unblock a broken
                    # stream implementation, then wait once more.
                    if process.stderr is not None:
                        process.stderr.close()
                    self._stderr_thread.join(0.5)
                with self._stderr_lock:
                    stderr_failed = bool(self._stderr_errors)
                if self._stderr_thread.is_alive() or stderr_failed:
                    cleanup_error = cleanup_error or RuntimeError(
                        "MCP stderr drain did not settle"
                    )
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        if process is not None and process.stderr is not None:
            try:
                process.stderr.close()
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        if self._termination_errors:
            cleanup_error = cleanup_error or self._termination_errors[0]
        if self._temporary is not None:
            try:
                self._temporary.cleanup()
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            self._temporary = None
        self.process = None
        self._state = "closed"
        if cleanup_error is not None:
            raise McpProtocolError("MCP process cleanup failed", code="mcp_cleanup_failed") from cleanup_error

    def __enter__(self) -> "StdioMcpConnection":
        self.start()
        return self

    def __exit__(self, _type: Any, value: Any, _traceback: Any) -> None:
        _close_preserving_primary(self, value)


def _close_preserving_primary(
    connection: StdioMcpConnection,
    primary_error: BaseException | None,
) -> None:
    """Close a connection without replacing a failure from the main operation."""

    try:
        connection.close()
    except McpProtocolError as cleanup_error:
        if primary_error is None:
            raise
        if hasattr(primary_error, "add_note"):
            primary_error.add_note(f"suppressed MCP cleanup failure: {cleanup_error}")


def _catalog_for_server(
    store: SessionStore,
    snapshot: McpSnapshot,
    server_id: str,
) -> tuple[McpServerDefinition, Mapping[str, Any]]:
    definition = snapshot.server(server_id)
    trust = snapshot.statuses(store.mcp_trust_state()).get(server_id)
    if trust != "trusted":
        raise McpLoadError("MCP server must have an exact trusted definition before refresh")
    catalogs = store.mcp_catalog_state()
    cached = catalogs.get(server_id)
    if not isinstance(cached, Mapping) or cached.get("definition_sha256") != definition.definition_sha256:
        raise McpRefreshRequired((server_id,))
    return definition, cached


def refresh_mcp_catalog(
    store: SessionStore,
    snapshot: McpSnapshot,
    server_id: str,
    *,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    definition = snapshot.server(server_id)
    trust = snapshot.statuses(store.mcp_trust_state()).get(server_id)
    if trust != "trusted":
        raise McpLoadError("MCP server must have an exact trusted definition before refresh")
    definition.environment_values()
    definition.revalidate_command()
    token = cancellation_token or CancellationToken()
    deadline = time.monotonic() + definition.startup_timeout_seconds
    connection = StdioMcpConnection(definition)
    primary_error: BaseException | None = None
    try:
        connection.start()
        connection.initialize(deadline_monotonic=deadline, cancellation_token=token)
        record = connection.list_tools(deadline_monotonic=deadline, cancellation_token=token)
        if connection.catalog_stale:
            raise McpProtocolError(
                "MCP catalog changed during refresh",
                code="mcp_catalog_stale",
            )
    except BaseException as exc:
        primary_error = exc
        connection._terminate()
        raise
    finally:
        _close_preserving_primary(connection, primary_error)
    store.set_mcp_catalog(server_id, record)
    return {
        **record,
        "server_id": server_id,
        "stderr": connection.stderr_metadata,
    }


class TrustedMcpCatalog:
    """Frozen trusted MCP catalog used to build one run's tool registry."""

    def __init__(
        self,
        snapshot: McpSnapshot,
        trust_state: Mapping[str, Mapping[str, str]],
        catalogs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        unresolved = snapshot.unresolved(trust_state)
        if unresolved:
            raise McpTrustRequired(unresolved)
        statuses = snapshot.statuses(trust_state)
        missing: list[str] = []
        active: list[tuple[McpServerDefinition, Mapping[str, Any]]] = []
        validated_tools: dict[str, tuple[McpToolDefinition, ...]] = {}
        active_tool_count = 0
        active_catalog_bytes = 0
        for definition in snapshot.servers:
            if statuses[definition.server_id] != "trusted":
                continue
            cached = catalogs.get(definition.server_id)
            if (
                not isinstance(cached, Mapping)
                or cached.get("definition_sha256") != definition.definition_sha256
                or cached.get("protocol_version") != MCP_PROTOCOL_VERSION
            ):
                missing.append(definition.server_id)
                continue
            server_tools = _validated_cached_tools(definition, cached)
            catalog_bytes = _cached_catalog_bytes(cached)
            active_tool_count += len(server_tools)
            active_catalog_bytes += catalog_bytes
            if (
                active_tool_count > _MAX_ACTIVE_MCP_TOOLS
                or active_catalog_bytes > _MAX_ACTIVE_MCP_CATALOG_BYTES
            ):
                raise McpLoadError("active MCP catalog budget exceeded")
            validated_tools[definition.server_id] = server_tools
            active.append((definition, cached))
        if missing:
            raise McpRefreshRequired(tuple(missing))
        self.snapshot = snapshot
        self._statuses = statuses
        self._active = tuple(active)
        self._tools: list[tuple[McpServerDefinition, Mapping[str, Any], McpToolDefinition]] = []
        for definition, catalog in self._active:
            for tool in validated_tools[definition.server_id]:
                self._tools.append((definition, catalog, tool))
        policy = {
            "schema": MCP_POLICY_SCHEMA,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "protocol_version": MCP_PROTOCOL_VERSION,
            "servers": [
                {
                    "server_id": definition.server_id,
                    "definition_sha256": definition.definition_sha256,
                    "catalog_sha256": catalog.get("catalog_sha256"),
                    "tool_count": len(catalog.get("tools", [])),
                    "network_access": definition.network_access,
                    "allow_process_fork": definition.allow_process_fork,
                    "trust_status": "trusted",
                }
                for definition, catalog in self._active
            ],
        }
        self._policy_material = json.loads(canonical_json(policy))

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def policy_material(self) -> Mapping[str, Any]:
        return json.loads(canonical_json(self._policy_material))

    def register_tools(self, registry: ToolRegistry) -> None:
        for definition, catalog, tool in self._tools:
            tool_version = canonical_sha256(
                {
                    "schema": "agent_harness.mcp_tool_binding.v1",
                    "server_definition_sha256": definition.definition_sha256,
                    "catalog_sha256": catalog.get("catalog_sha256"),
                    "remote_tool_sha256": tool.definition_sha256,
                    "local_name": tool.local_name,
                    "risk": "high",
                    "replay_policy": "never",
                    "persistent_approval_allowed": False,
                }
            )

            def handler(
                arguments: Mapping[str, Any],
                context: ToolExecutionContext,
                *,
                server_definition: McpServerDefinition = definition,
                cached_catalog: Mapping[str, Any] = catalog,
                remote_tool: McpToolDefinition = tool,
            ) -> dict[str, Any]:
                try:
                    server_definition.environment_values()
                    server_definition.revalidate_command()
                except McpLoadError as exc:
                    raise ToolExecutionError(str(exc), code="mcp_preflight_failed") from exc
                context.begin_effect()
                connection = StdioMcpConnection(server_definition)
                primary_error: BaseException | None = None
                try:
                    connection.start()
                    startup_deadline = min(
                        context.deadline_monotonic,
                        time.monotonic() + server_definition.startup_timeout_seconds,
                    )
                    connection.initialize(
                        deadline_monotonic=startup_deadline,
                        cancellation_token=context.cancellation_token,
                    )
                    live_catalog = connection.list_tools(
                        deadline_monotonic=startup_deadline,
                        cancellation_token=context.cancellation_token,
                    )
                    if (
                        live_catalog.get("catalog_sha256") != cached_catalog.get("catalog_sha256")
                        or connection.catalog_stale
                    ):
                        raise McpProtocolError("MCP live catalog changed; explicit refresh is required", code="mcp_catalog_stale")
                    return connection.call_tool(
                        remote_tool,
                        arguments,
                        deadline_monotonic=context.deadline_monotonic,
                        cancellation_token=context.cancellation_token,
                    )
                except McpProtocolError as exc:
                    connection._terminate()
                    primary_error = ToolExecutionError(str(exc), code=exc.code)
                    raise primary_error from exc
                except BaseException as exc:
                    primary_error = exc
                    connection._terminate()
                    raise
                finally:
                    try:
                        _close_preserving_primary(connection, primary_error)
                    except McpProtocolError as exc:
                        raise ToolExecutionError(str(exc), code=exc.code) from exc

            registry.register(
                ToolSpec(
                    name=tool.local_name,
                    version=tool_version,
                    description=(
                        f"External MCP tool {tool.raw_name} from explicitly trusted "
                        f"server {definition.server_id}. Server annotations are not authority."
                    ),
                    input_schema=dict(tool.input_schema),
                    output_schema=None,
                    permission="mcp.external",
                    risk="high",
                    timeout_seconds=min(
                        360.0,
                        definition.startup_timeout_seconds + definition.tool_timeout_seconds,
                    ),
                    replay_policy="never",
                    parallel_safe=False,
                    execution_isolation="trusted_inline",
                    trusted_inline_reason=(
                        "Exact-trust MCP broker owns a bounded Seatbelt stdio process"
                    ),
                    data_scope="external_service",
                    requires_user_consent=True,
                    persistent_approval_allowed=False,
                    retry_policy=RetryPolicy(max_attempts=1),
                ),
                handler,
            )


def mcp_status(store: SessionStore, snapshot: McpSnapshot | None = None) -> dict[str, Any]:
    active = snapshot or load_project_mcp(store.workspace)
    return active.metadata(store.mcp_trust_state(), store.mcp_catalog_state())


__all__ = [
    "MCP_CATALOG_RECORD_SCHEMA",
    "MCP_CONFIG_RELATIVE_PATH",
    "MCP_CONFIG_SCHEMA",
    "MCP_POLICY_SCHEMA",
    "MCP_SNAPSHOT_SCHEMA",
    "McpLoadError",
    "McpRefreshRequired",
    "McpServerDefinition",
    "McpSnapshot",
    "McpTrustRequired",
    "StdioMcpConnection",
    "TrustedMcpCatalog",
    "load_project_mcp",
    "mcp_status",
    "refresh_mcp_catalog",
]
