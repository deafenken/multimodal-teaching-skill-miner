"""Secure discovery, trust binding and execution for project policy hooks.

Project hook files are proposals until an exact definition digest is trusted
or disabled in the private workspace state. Enabled hooks execute from the
snapshotted entrypoint bytes under a read-only, no-network Seatbelt profile and
can only preserve or tighten the central tool decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import time
from typing import Any, Mapping
from uuid import uuid4

from .core import (
    CancellationToken,
    HarnessCancelled,
    HarnessDeadlineExceeded,
    HookAuditSink,
    ToolExecutionContext,
    ToolExecutionError,
    ToolHookDecision,
    ToolHookRequest,
)
from .core.events import canonical_json, canonical_sha256
from .toolsets.workspace import WorkspaceToolset, workspace_sandbox_status


HOOK_CONFIG_SCHEMA = "agent_harness.hooks.v1"
HOOK_SNAPSHOT_SCHEMA = "agent_harness.hook_snapshot.v1"
HOOK_POLICY_SCHEMA = "agent_harness.hook_policy.v1"
HOOK_OUTPUT_SCHEMA = "agent_harness.hook_output.v1"
HOOK_CONFIG_RELATIVE_PATH = ".agent-harness/hooks.json"

_SUPPORTED_EVENTS = frozenset(
    {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
)
_HOOK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_MAX_CONFIG_BYTES = 64 * 1024
_MAX_ENTRYPOINT_BYTES = 256 * 1024
_MAX_HOOKS = 64
_MAX_MATCHERS = 32
_MAX_ARGS = 16
_MAX_ARG_CHARS = 1_000
_MAX_HOOK_INPUT_BYTES = 512 * 1024
_MAX_HOOK_INVOCATIONS = 64
_MAX_TOTAL_WALL_SECONDS = 30.0
_READ_CHUNK_BYTES = 64 * 1024
_DECISION_PRECEDENCE = {"pass": 0, "ask": 1, "deny": 2}


class HookLoadError(RuntimeError):
    """Raised when a project hook definition cannot be trusted."""


class HookTrustRequired(HookLoadError):
    """Raised before a run when hook proposals are unresolved."""

    def __init__(self, hook_ids: tuple[str, ...]) -> None:
        self.hook_ids = hook_ids
        joined = ", ".join(hook_ids[:8])
        suffix = "…" if len(hook_ids) > 8 else ""
        super().__init__(
            "project hooks require an exact trust or disable decision: "
            f"{joined}{suffix}; run `harness hooks` for digests"
        )


def _strict_json(text: str, *, label: str) -> Any:
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
        raise HookLoadError(f"{label} must be strict JSON") from exc


@dataclass(frozen=True, slots=True)
class HookDefinition:
    hook_id: str
    event_name: str
    matchers: tuple[str, ...]
    entrypoint: str
    args: tuple[str, ...]
    timeout_seconds: float
    config_sha256: str
    entrypoint_sha256: str
    definition_sha256: str
    entrypoint_bytes: bytes = field(repr=False, compare=False)

    def matches(self, event_name: str, tool_name: str) -> bool:
        return self.event_name == event_name and (
            "*" in self.matchers or tool_name in self.matchers
        )

    def policy_material(self, *, trust_status: str) -> dict[str, Any]:
        return {
            "hook_id": self.hook_id,
            "event_name": self.event_name,
            "matchers": list(self.matchers),
            "entrypoint": self.entrypoint,
            "args_sha256": canonical_sha256(list(self.args)),
            "timeout_seconds": self.timeout_seconds,
            "config_sha256": self.config_sha256,
            "entrypoint_sha256": self.entrypoint_sha256,
            "definition_sha256": self.definition_sha256,
            "trust_status": trust_status,
        }

    def metadata(self, *, trust_status: str) -> dict[str, Any]:
        return {
            "hook_id": self.hook_id,
            "event_name": self.event_name,
            "matchers": list(self.matchers),
            "entrypoint": self.entrypoint,
            "arg_count": len(self.args),
            "timeout_seconds": self.timeout_seconds,
            "entrypoint_sha256": self.entrypoint_sha256,
            "definition_sha256": self.definition_sha256,
            "trust_status": trust_status,
        }


@dataclass(frozen=True, slots=True)
class HookSnapshot:
    workspace_root: str
    config_present: bool
    config_sha256: str
    definitions: tuple[HookDefinition, ...]
    snapshot_sha256: str
    schema: str = HOOK_SNAPSHOT_SCHEMA

    def statuses(
        self,
        trust_state: Mapping[str, Mapping[str, str]],
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for definition in self.definitions:
            record = trust_state.get(definition.hook_id)
            if not isinstance(record, Mapping):
                result[definition.hook_id] = "untrusted"
            elif record.get("definition_sha256") != definition.definition_sha256:
                result[definition.hook_id] = "modified"
            elif record.get("action") in {"trusted", "disabled"}:
                result[definition.hook_id] = str(record["action"])
            else:
                result[definition.hook_id] = "untrusted"
        return result

    def unresolved(
        self,
        trust_state: Mapping[str, Mapping[str, str]],
    ) -> tuple[str, ...]:
        statuses = self.statuses(trust_state)
        return tuple(
            definition.hook_id
            for definition in self.definitions
            if statuses[definition.hook_id] in {"untrusted", "modified"}
        )

    def metadata(
        self,
        trust_state: Mapping[str, Mapping[str, str]],
    ) -> dict[str, Any]:
        statuses = self.statuses(trust_state)
        sandbox = workspace_sandbox_status()
        return {
            "schema": self.schema,
            "workspace_root": self.workspace_root,
            "source": HOOK_CONFIG_RELATIVE_PATH,
            "config_present": self.config_present,
            "config_sha256": self.config_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "hook_count": len(self.definitions),
            "trusted_hook_count": sum(
                statuses[item.hook_id] == "trusted" for item in self.definitions
            ),
            "disabled_hook_count": sum(
                statuses[item.hook_id] == "disabled" for item in self.definitions
            ),
            "pending_hook_count": sum(
                statuses[item.hook_id] in {"untrusted", "modified"}
                for item in self.definitions
            ),
            "hooks": [
                item.metadata(trust_status=statuses[item.hook_id])
                for item in self.definitions
            ],
            "sandbox": {
                "available": sandbox["available"],
                "backend": sandbox["backend"],
                "workspace_writable": False,
                "network_denied": sandbox["network_denied"],
                "process_fork_denied": sandbox["available"],
            },
        }


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise HookLoadError("secure no-follow hook reads are unavailable")
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


def _safe_relative(value: Any, *, entrypoint: bool = False) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise HookLoadError("hook path is invalid")
    normalized = value.strip().replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise HookLoadError("hook path must stay inside the workspace")
    if entrypoint and tuple(pure.parts[:2]) != (".agent-harness", "hooks"):
        raise HookLoadError("hook entrypoint must be below .agent-harness/hooks")
    return tuple(pure.parts)


def _open_root(workspace: Path) -> int:
    try:
        descriptor = os.open(workspace, _directory_flags())
    except OSError as exc:
        raise HookLoadError("workspace cannot be opened securely") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise HookLoadError("workspace is not a real directory")
    return descriptor


def _read_relative_file(
    root_descriptor: int,
    parts: tuple[str, ...],
    *,
    maximum: int,
    required: bool,
    require_executable: bool = False,
) -> bytes | None:
    current = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            try:
                next_descriptor = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=current,
                )
            except FileNotFoundError:
                if required:
                    raise HookLoadError("hook path does not exist")
                return None
            except OSError as exc:
                raise HookLoadError("hook directory cannot be opened securely") from exc
            metadata = os.fstat(next_descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o022
            ):
                os.close(next_descriptor)
                raise HookLoadError("hook directory permissions are unsafe")
            os.close(current)
            current = next_descriptor
        try:
            expected = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        except FileNotFoundError:
            if required:
                raise HookLoadError("hook file does not exist")
            return None
        except OSError as exc:
            raise HookLoadError("hook file cannot be inspected securely") from exc
        if (
            stat.S_ISLNK(expected.st_mode)
            or not stat.S_ISREG(expected.st_mode)
            or expected.st_mode & 0o022
            or expected.st_uid != os.getuid()
            or expected.st_nlink != 1
            or expected.st_size > maximum
            or (require_executable and not expected.st_mode & stat.S_IXUSR)
        ):
            raise HookLoadError("hook file type, size or permissions are unsafe")
        try:
            descriptor = os.open(parts[-1], _file_flags(), dir_fd=current)
        except OSError as exc:
            raise HookLoadError("hook file cannot be opened securely") from exc
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != expected.st_dev
                or opened.st_ino != expected.st_ino
                or opened.st_size != expected.st_size
                or opened.st_mtime_ns != expected.st_mtime_ns
                or opened.st_ctime_ns != expected.st_ctime_ns
            ):
                raise HookLoadError("hook file changed before reading")
            chunks: list[bytes] = []
            unread = opened.st_size
            while unread:
                chunk = os.read(descriptor, min(unread, _READ_CHUNK_BYTES))
                if not chunk:
                    raise HookLoadError("hook file changed while reading")
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
                raise HookLoadError("hook file changed while reading")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(current)


def _bounded_argument(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_ARG_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise HookLoadError("hook argument is invalid")
    return value


def _definition(
    raw: Any,
    *,
    event_name: str,
    config_sha256: str,
    root_descriptor: int,
) -> HookDefinition:
    if not isinstance(raw, Mapping):
        raise HookLoadError("hook definition must be an object")
    allowed = {"id", "matcher", "entrypoint", "args", "timeout_seconds"}
    if set(raw) - allowed:
        raise HookLoadError("hook definition contains unknown fields")
    hook_id = raw.get("id")
    if not isinstance(hook_id, str) or _HOOK_ID.fullmatch(hook_id) is None:
        raise HookLoadError("hook id is invalid")
    raw_matchers = raw.get("matcher")
    if (
        not isinstance(raw_matchers, list)
        or not 1 <= len(raw_matchers) <= _MAX_MATCHERS
    ):
        raise HookLoadError("hook matcher must be a non-empty array")
    matchers: list[str] = []
    for raw_matcher in raw_matchers:
        if raw_matcher != "*" and (
            not isinstance(raw_matcher, str)
            or _TOOL_NAME.fullmatch(raw_matcher) is None
        ):
            raise HookLoadError("hook matcher is invalid")
        if raw_matcher not in matchers:
            matchers.append(str(raw_matcher))
    entrypoint_parts = _safe_relative(raw.get("entrypoint"), entrypoint=True)
    entrypoint = "/".join(entrypoint_parts)
    entrypoint_bytes = _read_relative_file(
        root_descriptor,
        entrypoint_parts,
        maximum=_MAX_ENTRYPOINT_BYTES,
        required=True,
        require_executable=True,
    )
    assert entrypoint_bytes is not None
    raw_args = raw.get("args", [])
    if not isinstance(raw_args, list) or len(raw_args) > _MAX_ARGS:
        raise HookLoadError("hook args must be a bounded array")
    args = tuple(_bounded_argument(value) for value in raw_args)
    timeout = raw.get("timeout_seconds", 5.0)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0.1 <= float(timeout) <= 30.0
    ):
        raise HookLoadError("hook timeout_seconds is invalid")
    # Keep the raw file hash as the public identity; the definition digest
    # binds it to every execution-relevant field and the exact config bytes.
    entrypoint_sha256 = sha256(entrypoint_bytes).hexdigest()
    material = {
        "schema": "agent_harness.hook_definition.v1",
        "hook_id": hook_id,
        "event_name": event_name,
        "matchers": sorted(matchers),
        "entrypoint": entrypoint,
        "args": list(args),
        "timeout_seconds": float(timeout),
        "config_sha256": config_sha256,
        "entrypoint_sha256": entrypoint_sha256,
    }
    return HookDefinition(
        hook_id=hook_id,
        event_name=event_name,
        matchers=tuple(sorted(matchers)),
        entrypoint=entrypoint,
        args=args,
        timeout_seconds=float(timeout),
        config_sha256=config_sha256,
        entrypoint_sha256=entrypoint_sha256,
        definition_sha256=canonical_sha256(material),
        entrypoint_bytes=entrypoint_bytes,
    )


def load_project_hooks(workspace_root: os.PathLike[str] | str) -> HookSnapshot:
    candidate = Path(workspace_root).expanduser()
    try:
        if candidate.is_symlink():
            raise HookLoadError("workspace symlinks are not accepted for project hooks")
    except OSError as exc:
        raise HookLoadError("workspace cannot be inspected securely") from exc
    try:
        workspace = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HookLoadError("workspace must be an existing directory") from exc
    if not workspace.is_dir() or workspace.is_symlink():
        raise HookLoadError("workspace must be a real directory")
    root_descriptor = _open_root(workspace)
    try:
        config_bytes = _read_relative_file(
            root_descriptor,
            _safe_relative(HOOK_CONFIG_RELATIVE_PATH),
            maximum=_MAX_CONFIG_BYTES,
            required=False,
        )
        if config_bytes is None:
            empty_material = {
                "schema": HOOK_SNAPSHOT_SCHEMA,
                "config_present": False,
                "definitions": [],
            }
            return HookSnapshot(
                workspace_root=str(workspace),
                config_present=False,
                config_sha256=canonical_sha256(None),
                definitions=(),
                snapshot_sha256=canonical_sha256(empty_material),
            )
        try:
            config_text = config_bytes.decode("utf-8", errors="strict")
            value = _strict_json(config_text, label="hook config")
        except (UnicodeDecodeError, HookLoadError) as exc:
            raise HookLoadError("hook config must be valid UTF-8 JSON") from exc
        if not isinstance(value, Mapping):
            raise HookLoadError("hook config must contain an object")
        if set(value) != {"schema", "hooks"} or value.get("schema") != HOOK_CONFIG_SCHEMA:
            raise HookLoadError("hook config schema is invalid")
        raw_hooks = value.get("hooks")
        if not isinstance(raw_hooks, Mapping) or set(raw_hooks) - _SUPPORTED_EVENTS:
            raise HookLoadError("hook event configuration is invalid")
        config_sha256 = sha256(config_bytes).hexdigest()
        definitions: list[HookDefinition] = []
        seen_ids: set[str] = set()
        for event_name in ("PreToolUse", "PostToolUse", "PostToolUseFailure"):
            raw_definitions = raw_hooks.get(event_name, [])
            if not isinstance(raw_definitions, list):
                raise HookLoadError("hook event value must be an array")
            for raw_definition in raw_definitions:
                if len(definitions) >= _MAX_HOOKS:
                    raise HookLoadError("hook definition limit exceeded")
                definition = _definition(
                    raw_definition,
                    event_name=event_name,
                    config_sha256=config_sha256,
                    root_descriptor=root_descriptor,
                )
                if definition.hook_id in seen_ids:
                    raise HookLoadError("hook ids must be unique")
                seen_ids.add(definition.hook_id)
                definitions.append(definition)
        frozen = tuple(definitions)
        snapshot_material = {
            "schema": HOOK_SNAPSHOT_SCHEMA,
            "config_present": True,
            "config_sha256": config_sha256,
            "definitions": [item.definition_sha256 for item in frozen],
        }
        return HookSnapshot(
            workspace_root=str(workspace),
            config_present=True,
            config_sha256=config_sha256,
            definitions=frozen,
            snapshot_sha256=canonical_sha256(snapshot_material),
        )
    finally:
        os.close(root_descriptor)


class TrustedHookRunner:
    """Frozen, digest-bound broker for one run."""

    def __init__(
        self,
        snapshot: HookSnapshot,
        trust_state: Mapping[str, Mapping[str, str]],
    ) -> None:
        unresolved = snapshot.unresolved(trust_state)
        if unresolved:
            raise HookTrustRequired(unresolved)
        self.snapshot = snapshot
        self._statuses = snapshot.statuses(trust_state)
        self._workspace = Path(snapshot.workspace_root)
        self._toolset = WorkspaceToolset(self._workspace)
        self._sandbox_status = workspace_sandbox_status()
        if any(status == "trusted" for status in self._statuses.values()) and not bool(
            self._sandbox_status.get("available")
        ):
            raise HookLoadError(
                "trusted project hooks require the macOS Seatbelt sandbox"
            )
        self._invocations = 0
        self._wall_seconds = 0.0
        material = {
            "schema": HOOK_POLICY_SCHEMA,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "sandbox_protocol": (
                "macos-seatbelt-read-only-v1"
                if self._sandbox_status.get("available")
                else "unavailable-fail-closed-v1"
            ),
            "definitions": [
                item.policy_material(trust_status=self._statuses[item.hook_id])
                for item in snapshot.definitions
            ],
        }
        self._policy_material = json.loads(canonical_json(material))

    @property
    def policy_material(self) -> Mapping[str, Any]:
        return json.loads(canonical_json(self._policy_material))

    def matches(self, event_name: str, tool_name: str) -> bool:
        return any(
            self._statuses[item.hook_id] == "trusted"
            and item.matches(event_name, tool_name)
            for item in self.snapshot.definitions
        )

    def _matching(self, request: ToolHookRequest) -> tuple[HookDefinition, ...]:
        return tuple(
            item
            for item in self.snapshot.definitions
            if self._statuses[item.hook_id] == "trusted"
            and item.matches(request.event_name, request.tool_name)
        )

    def _run_one(
        self,
        definition: HookDefinition,
        request: ToolHookRequest,
        *,
        ordinal: int,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
        audit_sink: HookAuditSink,
    ) -> str:
        invocation_id = f"hook_{uuid4().hex}"
        hook_input = {**request.to_input(), "hook_id": definition.hook_id}
        encoded = (canonical_json(hook_input) + "\n").encode("utf-8")
        common = {
            "invocation_id": invocation_id,
            "hook_id": definition.hook_id,
            "hook_event_name": request.event_name,
            "hook_sha256": definition.definition_sha256,
            "input_sha256": canonical_sha256(hook_input),
            "call_id": request.call_id,
            "tool_name": request.tool_name,
            "ordinal": ordinal,
        }
        audit_sink("hook.started", common)
        if len(encoded) > _MAX_HOOK_INPUT_BYTES:
            audit_sink(
                "hook.failed",
                {
                    **common,
                    "duration_ms": 0,
                    "error_code": "hook_input_too_large",
                    "output_sha256": canonical_sha256(None),
                },
            )
            return "deny" if request.event_name == "PreToolUse" else "pass"
        started = time.monotonic()
        output_text = ""
        try:
            audit_sink("hook.effect_started", common)
            with tempfile.TemporaryDirectory(
                prefix=".agent-harness-hook-",
                dir=self._workspace,
            ) as temporary_text:
                temporary = Path(temporary_text)
                os.chmod(temporary, 0o700)
                executable = temporary / "entrypoint"
                descriptor = os.open(
                    executable,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o700,
                )
                try:
                    view = memoryview(definition.entrypoint_bytes)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError("hook materialization made no progress")
                        view = view[written:]
                    os.fsync(descriptor)
                    os.fchmod(descriptor, 0o700)
                finally:
                    os.close(descriptor)
                context = ToolExecutionContext(
                    run_id=request.run_id,
                    turn_id=request.turn_id,
                    call_id=request.call_id,
                    tool_name=request.tool_name,
                    cancellation_token=cancellation_token,
                    deadline_monotonic=min(
                        deadline_monotonic,
                        started + definition.timeout_seconds,
                    ),
                    idempotency_key=None,
                    emit_progress=lambda _kind, _payload=None: None,
                    session_id=request.session_id,
                )
                result = self._toolset.run_policy_hook(
                    [str(executable), *definition.args],
                    context,
                    stdin=encoded,
                    timeout=definition.timeout_seconds,
                )
            output_text = str(result.get("output", ""))
            output_sha256 = canonical_sha256(output_text)
            if result.get("truncated") is True:
                raise ToolExecutionError(
                    "hook output exceeded its limit",
                    code="hook_output_too_large",
                )
            exit_code = result.get("exit_code")
            if exit_code == 2:
                decision = "deny"
            elif exit_code != 0:
                raise ToolExecutionError(
                    "hook exited unsuccessfully",
                    code="hook_exit_nonzero",
                )
            else:
                try:
                    raw_output = _strict_json(output_text, label="hook output")
                except HookLoadError as exc:
                    raise ToolExecutionError(
                        "hook output is not valid JSON",
                        code="hook_output_invalid",
                    ) from exc
                if (
                    not isinstance(raw_output, Mapping)
                    or set(raw_output) - {"schema", "decision", "reason_code"}
                    or raw_output.get("schema") != HOOK_OUTPUT_SCHEMA
                    or raw_output.get("decision") not in {"pass", "ask", "deny"}
                ):
                    raise ToolExecutionError(
                        "hook output contract is invalid",
                        code="hook_output_invalid",
                    )
                reason_code = raw_output.get("reason_code")
                if reason_code is not None and (
                    not isinstance(reason_code, str)
                    or _REASON_CODE.fullmatch(reason_code) is None
                ):
                    raise ToolExecutionError(
                        "hook reason_code is invalid",
                        code="hook_output_invalid",
                    )
                decision = str(raw_output["decision"])
            if request.event_name != "PreToolUse" and decision != "pass":
                raise ToolExecutionError(
                    "post-tool hooks are observe-only",
                    code="hook_post_decision_invalid",
                )
            audit_sink(
                "hook.completed",
                {
                    **common,
                    "duration_ms": max(0, round((time.monotonic() - started) * 1_000)),
                    "action": decision,
                    "output_sha256": output_sha256,
                },
            )
            return decision
        except (HarnessCancelled, HarnessDeadlineExceeded):
            audit_sink(
                "hook.failed",
                {
                    **common,
                    "duration_ms": max(0, round((time.monotonic() - started) * 1_000)),
                    "error_code": "hook_cancelled",
                    "output_sha256": canonical_sha256(output_text),
                },
            )
            raise
        except (OSError, ToolExecutionError) as exc:
            error_code = (
                exc.code if isinstance(exc, ToolExecutionError) else "hook_spawn_failed"
            )
            audit_sink(
                "hook.failed",
                {
                    **common,
                    "duration_ms": max(0, round((time.monotonic() - started) * 1_000)),
                    "error_code": str(error_code)[:128],
                    "output_sha256": canonical_sha256(output_text),
                },
            )
            if error_code in {
                "command_cleanup_failed",
                "command_output_error",
                "command_output_timeout",
            }:
                raise
            return "deny" if request.event_name == "PreToolUse" else "pass"

    def evaluate(
        self,
        request: ToolHookRequest,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
        audit_sink: HookAuditSink,
    ) -> ToolHookDecision:
        matching = self._matching(request)
        if not matching:
            return ToolHookDecision()
        decisions: list[str] = []
        executed: list[str] = []
        skipped: list[str] = []
        for ordinal, definition in enumerate(matching, start=1):
            cancellation_token.raise_if_cancelled()
            if (
                self._invocations >= _MAX_HOOK_INVOCATIONS
                or self._wall_seconds >= _MAX_TOTAL_WALL_SECONDS
            ):
                invocation_id = f"hook_{uuid4().hex}"
                hook_input = {**request.to_input(), "hook_id": definition.hook_id}
                common = {
                    "invocation_id": invocation_id,
                    "hook_id": definition.hook_id,
                    "hook_event_name": request.event_name,
                    "hook_sha256": definition.definition_sha256,
                    "input_sha256": canonical_sha256(hook_input),
                    "call_id": request.call_id,
                    "tool_name": request.tool_name,
                    "ordinal": ordinal,
                }
                audit_sink("hook.started", common)
                audit_sink(
                    "hook.failed",
                    {
                        **common,
                        "duration_ms": 0,
                        "error_code": "hook_budget_exhausted",
                        "output_sha256": canonical_sha256(None),
                    },
                )
                decisions.append("deny" if request.event_name == "PreToolUse" else "pass")
                skipped.append(definition.hook_id)
                continue
            self._invocations += 1
            executed.append(definition.hook_id)
            invocation_started = time.monotonic()
            try:
                decisions.append(
                    self._run_one(
                        definition,
                        request,
                        ordinal=ordinal,
                        cancellation_token=cancellation_token,
                        deadline_monotonic=deadline_monotonic,
                        audit_sink=audit_sink,
                    )
                )
            finally:
                self._wall_seconds += max(
                    0.0,
                    time.monotonic() - invocation_started,
                )
        action = max(decisions or ["pass"], key=_DECISION_PRECEDENCE.__getitem__)
        if request.event_name != "PreToolUse":
            action = "pass"
        return ToolHookDecision(
            action=action,  # type: ignore[arg-type]
            matched_hook_ids=tuple(item.hook_id for item in matching),
            executed_hook_ids=tuple(executed),
            skipped_hook_ids=tuple(skipped),
        )


__all__ = [
    "HOOK_CONFIG_RELATIVE_PATH",
    "HOOK_CONFIG_SCHEMA",
    "HOOK_OUTPUT_SCHEMA",
    "HOOK_POLICY_SCHEMA",
    "HOOK_SNAPSHOT_SCHEMA",
    "HookDefinition",
    "HookLoadError",
    "HookSnapshot",
    "HookTrustRequired",
    "TrustedHookRunner",
    "load_project_hooks",
]
