"""Central typed tool registry and effect executor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from dataclasses import replace
import ctypes
import errno
import json
import os
import re
import select
import signal
import sys
import time
from typing import Any, Callable, Mapping

from .cancellation import (
    CancellationToken,
    HarnessClock,
    check_deadline,
    remaining_seconds,
)
from .contracts import (
    HarnessCancelled,
    HarnessContractError,
    HarnessDeadlineExceeded,
    REPLAY_POLICIES,
    RISK_LEVELS,
    RetryPolicy,
    ToolCall,
    ToolExecutionError,
    ToolPermissionError,
)
from .events import HarnessEventEmitter, canonical_json, canonical_sha256
from .schema import validate_schema, validate_schema_definition


_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_PERMISSION = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_DATA_SCOPES = frozenset(
    {
        "internal",
        "learner_profile",
        "learner_answer",
        "teacher_resource",
        "public_web",
        "workspace_read",
        "workspace_write",
    }
)
_TRUST_GRANTS = _DATA_SCOPES | {"remote_consent"}
DEFAULT_TRUSTED_DATA_SCOPES = frozenset(
    {"internal", "learner_profile", "learner_answer", "teacher_resource"}
)
TOOL_EXECUTION_MANIFEST_SCHEMA = (
    "teaching_skill_miner.agent_harness_tool_execution_manifest.v1"
)
_EXECUTION_ISOLATIONS = frozenset({"isolated_process", "trusted_inline"})
_INLINE_REASON = re.compile(r"^[^\x00-\x1f\x7f]{8,300}$")
_ISOLATED_WORKER_MAX_FRAME_BYTES = 1_000_000
_ISOLATED_WORKER_TERM_GRACE_SECONDS = 0.25
_ISOLATED_WORKER_KILL_GRACE_SECONDS = 0.25
_DARWIN_TOOL_SANDBOX_PROFILE = (
    "(version 1)"
    "(allow default)"
    "(deny network*)"
    "(deny process-exec)"
    "(deny process-fork)"
    "(deny file-write*)"
)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    version: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None = None
    permission: str = "tool.read"
    risk: str = "low"
    timeout_seconds: float = 10.0
    replay_policy: str = "safe"
    parallel_safe: bool = False
    execution_mode: str = "central"
    execution_isolation: str = "isolated_process"
    trusted_inline_reason: str | None = None
    data_scope: str = "internal"
    requires_user_consent: bool = False
    retry_policy: RetryPolicy = field(
        default_factory=lambda: RetryPolicy(max_attempts=1)
    )

    def __post_init__(self) -> None:
        self.validated()

    def validated(self) -> "ToolSpec":
        if _TOOL_NAME.fullmatch(self.name) is None:
            raise HarnessContractError("tool name is invalid")
        if not self.version.strip() or len(self.version) > 64:
            raise HarnessContractError("tool version is invalid")
        if not self.description.strip() or len(self.description) > 1_000:
            raise HarnessContractError("tool description is invalid")
        if _PERMISSION.fullmatch(self.permission) is None:
            raise HarnessContractError("tool permission is invalid")
        if self.risk not in RISK_LEVELS:
            raise HarnessContractError("tool risk level is invalid")
        if self.replay_policy not in REPLAY_POLICIES:
            raise HarnessContractError("tool replay policy is invalid")
        if self.execution_mode not in {"central", "provider_managed"}:
            raise HarnessContractError("tool execution_mode is invalid")
        if self.execution_isolation not in _EXECUTION_ISOLATIONS:
            raise HarnessContractError("tool execution_isolation is invalid")
        if self.execution_isolation == "trusted_inline":
            if (
                not isinstance(self.trusted_inline_reason, str)
                or _INLINE_REASON.fullmatch(self.trusted_inline_reason.strip()) is None
            ):
                raise HarnessContractError(
                    "trusted inline tools require a bounded audit reason"
                )
        elif self.trusted_inline_reason is not None:
            raise HarnessContractError(
                "isolated tools cannot declare a trusted inline reason"
            )
        if self.data_scope not in _DATA_SCOPES:
            raise HarnessContractError("tool data_scope is invalid")
        if not isinstance(self.requires_user_consent, bool):
            raise HarnessContractError("tool requires_user_consent is invalid")
        if not 0.01 <= float(self.timeout_seconds) <= 86_400:
            raise HarnessContractError("tool timeout_seconds is invalid")
        # Validate schema syntax with representative empty objects when the
        # root says object. Required-field failures are expected and ignored;
        # unknown keywords are still rejected by the validator at execution.
        if not isinstance(self.input_schema, Mapping):
            raise HarnessContractError("tool input_schema must be an object")
        if self.output_schema is not None and not isinstance(
            self.output_schema, Mapping
        ):
            raise HarnessContractError("tool output_schema must be an object")
        validate_schema_definition(self.input_schema, path=f"tool.{self.name}.input")
        validate_schema_definition(self.output_schema, path=f"tool.{self.name}.output")
        self.retry_policy.validated()
        if self.replay_policy == "never" and self.retry_policy.max_attempts > 1:
            raise HarnessContractError("never-replay tools cannot be retried")
        return self

    def model_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "permission": self.permission,
            "risk": self.risk,
            "execution_mode": self.execution_mode,
            "data_scope": self.data_scope,
            "requires_user_consent": self.requires_user_consent,
        }

    def execution_manifest(self) -> dict[str, Any]:
        """Return the complete, credential-free execution contract.

        Provider-visible definitions intentionally omit harness-only controls.
        Recovery policy cannot: changing any ``ToolSpec`` field can alter model
        selection, validation, authorization, execution, retry, or replay
        semantics.  ``asdict`` deliberately covers every present and future
        dataclass field so a newly added execution knob cannot be forgotten in
        the checkpoint policy digest.
        """

        return {
            "schema": TOOL_EXECUTION_MANIFEST_SCHEMA,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    run_id: str
    turn_id: str
    call_id: str
    tool_name: str
    cancellation_token: CancellationToken
    deadline_monotonic: float
    idempotency_key: str | None
    emit_progress: Callable[[str, Mapping[str, Any] | None], None]
    principal_id: str = "local-user"
    session_id: str | None = None
    trusted_data_scopes: frozenset[str] = DEFAULT_TRUSTED_DATA_SCOPES


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    call_id: str
    tool_name: str
    ok: bool
    result: Any = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    result_sha256: str | None = None
    source_call_id: str | None = None

    def observation(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "kind": "tool_result",
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "ok": self.ok,
        }
        if self.ok:
            value["result"] = self.result
            value["result_sha256"] = self.result_sha256
        else:
            value["error"] = {
                "code": self.error_code,
                "message": self.error_message,
                "retryable": self.retryable,
            }
        if self.source_call_id:
            value["source_call_id"] = self.source_call_id
        return value


ToolHandler = Callable[[Mapping[str, Any], ToolExecutionContext], Any]


class ToolRegistry:
    """One authoritative namespace for all executable tools."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        spec.validated()
        if not callable(handler):
            raise HarnessContractError("tool handler must be callable")
        if spec.name in self._entries:
            raise HarnessContractError(f"tool is already registered: {spec.name}")
        self._entries[spec.name] = (spec, handler)

    def get(self, name: str) -> tuple[ToolSpec, ToolHandler] | None:
        return self._entries.get(name)

    def definitions(
        self,
        allowed_permissions: set[str],
        *,
        trusted_data_scopes: frozenset[str] | set[str] = DEFAULT_TRUSTED_DATA_SCOPES,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            spec.model_definition()
            for spec in self._authorized_specs(
                allowed_permissions,
                trusted_data_scopes=trusted_data_scopes,
            )
        )

    def execution_manifests(
        self,
        allowed_permissions: set[str],
        *,
        trusted_data_scopes: frozenset[str] | set[str] = DEFAULT_TRUSTED_DATA_SCOPES,
    ) -> tuple[dict[str, Any], ...]:
        """Return canonical-order manifests for every executable tool."""

        return tuple(
            spec.execution_manifest()
            for spec in self._authorized_specs(
                allowed_permissions,
                trusted_data_scopes=trusted_data_scopes,
            )
        )

    def _authorized_specs(
        self,
        allowed_permissions: set[str],
        *,
        trusted_data_scopes: frozenset[str] | set[str],
    ) -> tuple[ToolSpec, ...]:
        permissions = validate_allowed_permissions(allowed_permissions)
        scopes = validate_trusted_data_scopes(trusted_data_scopes)
        return tuple(
            spec
            for _name, (spec, _handler) in sorted(self._entries.items())
            if spec.permission in permissions
            and spec.data_scope in scopes
            and (not spec.requires_user_consent or "remote_consent" in scopes)
        )

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(spec for spec, _handler in self._entries.values())


def validate_allowed_permissions(value: set[str]) -> set[str]:
    if not isinstance(value, set):
        raise ToolPermissionError("allowed_permissions must be an explicit set")
    if not value or "*" in value:
        raise ToolPermissionError("wildcard or empty tool permissions are forbidden")
    for permission in value:
        if not isinstance(permission, str) or _PERMISSION.fullmatch(permission) is None:
            raise ToolPermissionError("allowed permission is invalid")
    return set(value)


def validate_trusted_data_scopes(value: frozenset[str] | set[str]) -> frozenset[str]:
    """Validate explicit data-domain grants independently from tool names."""

    if not isinstance(value, (set, frozenset)):
        raise ToolPermissionError("trusted_data_scopes must be an explicit set")
    normalized = frozenset(value)
    if not normalized or any(
        not isinstance(scope, str) or scope not in _TRUST_GRANTS for scope in normalized
    ):
        raise ToolPermissionError("trusted data scope is invalid")
    return normalized


def _run_trusted_inline_handler(
    handler: ToolHandler,
    arguments: Mapping[str, Any],
    context: ToolExecutionContext,
    *,
    clock: HarnessClock,
) -> Any:
    """Run one explicitly trusted, bounded, in-memory handler synchronously.

    Python threads cannot be forcefully stopped.  Executing this small class
    synchronously means a timeout can never be reported while the handler is
    still running and able to mutate state.  Only repository-owned,
    deterministic handlers may opt into this path with an audit reason.
    """

    context.cancellation_token.raise_if_cancelled()
    check_deadline(clock=clock, deadline_monotonic=context.deadline_monotonic)
    result = handler(arguments, context)
    context.cancellation_token.raise_if_cancelled()
    if (
        remaining_seconds(clock=clock, deadline_monotonic=context.deadline_monotonic)
        <= 0
    ):
        raise HarnessDeadlineExceeded("tool execution deadline exceeded")
    return result


def _write_worker_frame(fd: int, frame: Mapping[str, Any]) -> None:
    encoded = canonical_json(dict(frame)).encode("utf-8") + b"\n"
    if len(encoded) > _ISOLATED_WORKER_MAX_FRAME_BYTES:
        raise ValueError("isolated worker frame is too large")
    view = memoryview(encoded)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("isolated worker pipe write failed")
        view = view[written:]


def _activate_darwin_tool_sandbox() -> None:
    """Apply the kernel Seatbelt profile used by untrusted central tools."""

    if sys.platform != "darwin" or not hasattr(os, "fork"):
        raise RuntimeError("kernel tool sandbox is unavailable on this platform")
    library = ctypes.CDLL("/usr/lib/libsandbox.dylib")
    library.sandbox_init.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    library.sandbox_init.restype = ctypes.c_int
    error_buffer = ctypes.c_char_p()
    status = library.sandbox_init(
        _DARWIN_TOOL_SANDBOX_PROFILE.encode("utf-8"),
        0,
        ctypes.byref(error_buffer),
    )
    if status != 0:
        if error_buffer.value:
            try:
                library.sandbox_free_error(error_buffer)
            except Exception:
                pass
        raise RuntimeError("kernel tool sandbox activation failed")


def _wait_for_worker(pid: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        try:
            waited, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return True
        if waited == pid:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.005)


def _stop_isolated_worker(pid: int, *, process_group_ready: bool) -> None:
    """Terminate and reap exactly one worker before returning to the caller."""

    if _wait_for_worker(pid, 0.0):
        return
    target = -pid if process_group_ready else pid
    try:
        os.kill(target, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise HarnessContractError(
                "isolated tool worker could not be terminated"
            ) from exc
    if _wait_for_worker(pid, _ISOLATED_WORKER_TERM_GRACE_SECONDS):
        return
    try:
        os.kill(target, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise HarnessContractError(
                "isolated tool worker could not be killed"
            ) from exc
    if not _wait_for_worker(pid, _ISOLATED_WORKER_KILL_GRACE_SECONDS):
        raise HarnessContractError("isolated tool worker remained alive")


def _run_isolated_handler(
    handler: ToolHandler,
    arguments: Mapping[str, Any],
    context: ToolExecutionContext,
    *,
    clock: HarnessClock,
) -> Any:
    """Run an untrusted handler in a killable, deny-network Seatbelt worker.

    Unsupported platforms fail closed before the handler runs.  On macOS the
    child becomes its own process group, activates a kernel profile which
    denies network, filesystem writes, exec, and further forks, and relays
    only bounded JSON progress/result frames.  Timeout and cancellation reap
    the complete group before control returns, so no late side effect can
    occur after a terminal tool event.
    """

    if sys.platform != "darwin" or not hasattr(os, "fork"):
        raise ToolExecutionError(
            "isolated tool sandbox is unavailable",
            code="sandbox_unavailable",
            retryable=False,
        )
    read_fd, write_fd = os.pipe()
    os.set_inheritable(read_fd, False)
    os.set_inheritable(write_fd, False)
    pid = os.fork()
    if pid == 0:  # pragma: no branch - exercised through the parent contract
        try:
            os.close(read_fd)
            os.setsid()

            def child_progress(
                kind: str, payload: Mapping[str, Any] | None = None
            ) -> None:
                _write_worker_frame(
                    write_fd,
                    {
                        "kind": "progress",
                        "progress_kind": str(kind)[:80],
                        "progress": dict(payload or {}),
                    },
                )

            child_context = replace(context, emit_progress=child_progress)
            try:
                _activate_darwin_tool_sandbox()
            except BaseException:
                _write_worker_frame(
                    write_fd,
                    {"kind": "sandbox_error", "error_code": "sandbox_unavailable"},
                )
                os._exit(78)
            _write_worker_frame(write_fd, {"kind": "ready"})
            try:
                result = handler(dict(arguments), child_context)
                _write_worker_frame(write_fd, {"kind": "result", "result": result})
            except HarnessCancelled:
                _write_worker_frame(write_fd, {"kind": "cancelled"})
            except HarnessDeadlineExceeded:
                _write_worker_frame(write_fd, {"kind": "deadline"})
            except ToolExecutionError as exc:
                _write_worker_frame(
                    write_fd,
                    {
                        "kind": "tool_error",
                        "error_code": exc.code,
                        "error_message": str(exc)[:500],
                        "retryable": exc.retryable,
                    },
                )
            except BaseException as exc:
                _write_worker_frame(
                    write_fd,
                    {"kind": "unexpected_error", "error_type": type(exc).__name__},
                )
            os._exit(0)
        except BaseException:
            os._exit(70)

    os.close(write_fd)
    os.set_blocking(read_fd, False)
    process_group_ready = False
    buffer = bytearray()
    terminal_frame: dict[str, Any] | None = None
    initial_remaining = remaining_seconds(
        clock=clock, deadline_monotonic=context.deadline_monotonic
    )
    wall_deadline = time.monotonic() + initial_remaining
    try:
        while terminal_frame is None:
            context.cancellation_token.raise_if_cancelled()
            if (
                remaining_seconds(
                    clock=clock, deadline_monotonic=context.deadline_monotonic
                )
                <= 0
                or time.monotonic() >= wall_deadline
            ):
                raise HarnessDeadlineExceeded("tool execution deadline exceeded")
            ready, _writable, _exceptional = select.select(
                [read_fd], [], [], min(0.01, max(0.0, wall_deadline - time.monotonic()))
            )
            if ready:
                chunk = os.read(read_fd, 65_536)
                if chunk:
                    buffer.extend(chunk)
                    if len(buffer) > _ISOLATED_WORKER_MAX_FRAME_BYTES:
                        raise ToolExecutionError(
                            "isolated tool worker output is too large",
                            code="worker_protocol_error",
                        )
                    while b"\n" in buffer:
                        line, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        try:
                            frame = json.loads(line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise ToolExecutionError(
                                "isolated tool worker returned invalid data",
                                code="worker_protocol_error",
                            ) from exc
                        if not isinstance(frame, dict):
                            raise ToolExecutionError(
                                "isolated tool worker returned invalid data",
                                code="worker_protocol_error",
                            )
                        kind = frame.get("kind")
                        if kind == "ready":
                            process_group_ready = True
                        elif kind == "progress":
                            context.emit_progress(
                                str(frame.get("progress_kind", "update"))[:80],
                                frame.get("progress")
                                if isinstance(frame.get("progress"), Mapping)
                                else {},
                            )
                        elif kind in {
                            "result",
                            "cancelled",
                            "deadline",
                            "tool_error",
                            "unexpected_error",
                            "sandbox_error",
                        }:
                            terminal_frame = frame
                            break
                        else:
                            raise ToolExecutionError(
                                "isolated tool worker returned an unknown frame",
                                code="worker_protocol_error",
                            )
                elif not _wait_for_worker(pid, 0.0):
                    continue
            if terminal_frame is None and _wait_for_worker(pid, 0.0):
                raise ToolExecutionError(
                    "isolated tool worker exited without settlement",
                    code="worker_crashed",
                )
        if not _wait_for_worker(pid, _ISOLATED_WORKER_TERM_GRACE_SECONDS):
            _stop_isolated_worker(pid, process_group_ready=process_group_ready)
        kind = terminal_frame.get("kind")
        if kind == "result":
            return terminal_frame.get("result")
        if kind == "cancelled":
            raise HarnessCancelled("isolated tool worker cancelled")
        if kind == "deadline":
            raise HarnessDeadlineExceeded("tool execution deadline exceeded")
        if kind == "tool_error":
            raise ToolExecutionError(
                str(terminal_frame.get("error_message", "tool execution failed"))[:500],
                code=str(terminal_frame.get("error_code", "tool_execution_failed"))[
                    :80
                ],
                retryable=terminal_frame.get("retryable") is True,
            )
        if kind == "sandbox_error":
            raise ToolExecutionError(
                "isolated tool sandbox is unavailable",
                code="sandbox_unavailable",
                retryable=False,
            )
        raise ToolExecutionError(
            "unexpected tool failure "
            f"({str(terminal_frame.get('error_type', 'Error'))[:80]})",
            code="tool_execution_failed",
            retryable=False,
        )
    except BaseException:
        _stop_isolated_worker(pid, process_group_ready=process_group_ready)
        raise
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass


def _run_handler(
    handler: ToolHandler,
    arguments: Mapping[str, Any],
    context: ToolExecutionContext,
    *,
    spec: ToolSpec,
    clock: HarnessClock,
) -> Any:
    if spec.execution_isolation == "trusted_inline":
        return _run_trusted_inline_handler(handler, arguments, context, clock=clock)
    return _run_isolated_handler(handler, arguments, context, clock=clock)


def execute_tool_call(
    call: ToolCall,
    *,
    registry: ToolRegistry,
    allowed_permissions: set[str],
    emitter: HarnessEventEmitter,
    clock: HarnessClock,
    cancellation_token: CancellationToken,
    run_deadline_monotonic: float,
    max_output_chars: int,
    idempotency_receipts: Mapping[str, Mapping[str, Any]],
    principal_id: str = "local-user",
    session_id: str | None = None,
    trusted_data_scopes: frozenset[str] = DEFAULT_TRUSTED_DATA_SCOPES,
    effect_started_sink: Callable[[], None] | None = None,
) -> tuple[ToolExecutionResult, dict[str, Any] | None]:
    """Validate, authorize, execute, and normalize one tool call.

    The optional second return value is the durable idempotency receipt to add
    after successful settlement.
    """

    entry = registry.get(call.name)
    emitter.emit(
        "tool.requested",
        {
            "call_id": call.call_id,
            "tool_name": call.name,
            "arguments_sha256": canonical_sha256(dict(call.arguments)),
        },
    )
    if entry is None:
        emitter.emit(
            "tool.rejected",
            {
                "call_id": call.call_id,
                "tool_name": call.name,
                "error_code": "unknown_tool",
            },
        )
        return (
            ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error_code="unknown_tool",
                error_message="requested tool is not registered",
            ),
            None,
        )
    spec, handler = entry
    if spec.execution_mode != "central":
        emitter.emit(
            "tool.rejected",
            {
                "call_id": call.call_id,
                "tool_name": call.name,
                "tool_version": spec.version,
                "error_code": "provider_managed_tool_cannot_execute_centrally",
            },
        )
        return (
            ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error_code="provider_managed_tool_cannot_execute_centrally",
                error_message="provider-managed tool must execute inside its adapter",
            ),
            None,
        )
    if spec.permission not in allowed_permissions:
        emitter.emit(
            "tool.rejected",
            {
                "call_id": call.call_id,
                "tool_name": call.name,
                "tool_version": spec.version,
                "error_code": "permission_denied",
                "required_permission": spec.permission,
            },
        )
        return (
            ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error_code="permission_denied",
                error_message="tool permission was not granted",
            ),
            None,
        )
    if spec.data_scope not in trusted_data_scopes:
        emitter.emit(
            "tool.rejected",
            {
                "call_id": call.call_id,
                "tool_name": call.name,
                "tool_version": spec.version,
                "error_code": "data_scope_not_authorized",
                "data_scope": spec.data_scope,
            },
        )
        return (
            ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error_code="data_scope_not_authorized",
                error_message="tool data scope was not authorized",
            ),
            None,
        )
    if spec.requires_user_consent and "remote_consent" not in trusted_data_scopes:
        emitter.emit(
            "tool.rejected",
            {
                "call_id": call.call_id,
                "tool_name": call.name,
                "tool_version": spec.version,
                "error_code": "explicit_user_consent_required",
                "data_scope": spec.data_scope,
            },
        )
        return (
            ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error_code="explicit_user_consent_required",
                error_message="explicit user consent is required for this tool",
            ),
            None,
        )
    try:
        validate_schema(dict(call.arguments), spec.input_schema)
    except HarnessContractError as exc:
        emitter.emit(
            "tool.rejected",
            {
                "call_id": call.call_id,
                "tool_name": call.name,
                "tool_version": spec.version,
                "error_code": "invalid_arguments",
                "error_type": type(exc).__name__,
            },
        )
        return (
            ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error_code="invalid_arguments",
                error_message=str(exc)[:500],
            ),
            None,
        )
    if spec.replay_policy == "idempotent" and call.idempotency_key is None:
        emitter.emit(
            "tool.rejected",
            {
                "call_id": call.call_id,
                "tool_name": call.name,
                "tool_version": spec.version,
                "error_code": "idempotency_key_required",
            },
        )
        return (
            ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error_code="idempotency_key_required",
                error_message="idempotent tool requires an idempotency key",
            ),
            None,
        )
    receipt_key = (
        f"{spec.name}@{spec.version}:{call.idempotency_key}"
        if call.idempotency_key
        else None
    )
    existing = idempotency_receipts.get(receipt_key or "")
    if isinstance(existing, Mapping):
        expected_arguments = canonical_sha256(dict(call.arguments))
        if existing.get("arguments_sha256") != expected_arguments:
            emitter.emit(
                "tool.rejected",
                {
                    "call_id": call.call_id,
                    "tool_name": call.name,
                    "error_code": "idempotency_conflict",
                },
            )
            return (
                ToolExecutionResult(
                    call_id=call.call_id,
                    tool_name=call.name,
                    ok=False,
                    error_code="idempotency_conflict",
                    error_message="idempotency key was reused with different arguments",
                ),
                None,
            )
        emitter.emit(
            "tool.replayed",
            {
                "call_id": call.call_id,
                "tool_name": call.name,
                "tool_version": spec.version,
                "idempotency_key": call.idempotency_key,
                "source_call_id": existing.get("call_id"),
                "result_sha256": existing.get("result_sha256"),
            },
        )
        return (
            ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=True,
                result=existing.get("result"),
                result_sha256=str(existing.get("result_sha256", "")),
                source_call_id=str(existing.get("call_id", "")),
            ),
            None,
        )
    started = clock.monotonic()
    tool_deadline = min(run_deadline_monotonic, started + spec.timeout_seconds)

    def emit_progress(kind: str, payload: Mapping[str, Any] | None = None) -> None:
        emitter.emit(
            "tool.progress",
            {
                "call_id": call.call_id,
                "tool_name": call.name,
                "progress_kind": str(kind)[:80],
                "progress": dict(payload or {}),
            },
        )

    context = ToolExecutionContext(
        run_id=emitter.run_id,
        turn_id=emitter.turn_id,
        call_id=call.call_id,
        tool_name=call.name,
        cancellation_token=cancellation_token,
        deadline_monotonic=tool_deadline,
        idempotency_key=call.idempotency_key,
        emit_progress=emit_progress,
        principal_id=principal_id,
        session_id=session_id,
        trusted_data_scopes=frozenset(trusted_data_scopes),
    )
    last_error: BaseException | None = None
    for attempt in range(1, spec.retry_policy.max_attempts + 1):
        cancellation_token.raise_if_cancelled()
        check_deadline(clock=clock, deadline_monotonic=run_deadline_monotonic)
        emitter.emit(
            "tool.started",
            {
                "call_id": call.call_id,
                "tool_name": call.name,
                "tool_version": spec.version,
                "attempt": attempt,
                "risk": spec.risk,
            },
        )
        if effect_started_sink is not None:
            effect_started_sink()
        try:
            result = _run_handler(
                handler,
                dict(call.arguments),
                context,
                spec=spec,
                clock=clock,
            )
            validate_schema(result, spec.output_schema)
            rendered = canonical_json(result)
            if len(rendered) > max_output_chars:
                raise ToolExecutionError(
                    "tool result exceeds the configured output budget",
                    code="output_too_large",
                )
            result_hash = canonical_sha256(result)
            duration_ms = max(0, round((clock.monotonic() - started) * 1_000))
            emitter.emit(
                "tool.completed",
                {
                    "call_id": call.call_id,
                    "tool_name": call.name,
                    "tool_version": spec.version,
                    "attempt": attempt,
                    "duration_ms": duration_ms,
                    "result": result,
                    "result_sha256": result_hash,
                },
            )
            receipt = None
            if receipt_key is not None:
                receipt = {
                    "call_id": call.call_id,
                    "tool_name": call.name,
                    "tool_version": spec.version,
                    "arguments_sha256": canonical_sha256(dict(call.arguments)),
                    "result": result,
                    "result_sha256": result_hash,
                }
            return (
                ToolExecutionResult(
                    call_id=call.call_id,
                    tool_name=call.name,
                    ok=True,
                    result=result,
                    result_sha256=result_hash,
                ),
                receipt,
            )
        except HarnessCancelled:
            raise
        except HarnessDeadlineExceeded as exc:
            last_error = exc
            error = ToolExecutionError(str(exc), code="timeout", retryable=False)
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            last_error = exc
            if isinstance(exc, ToolExecutionError):
                error = exc
            else:
                error = ToolExecutionError(
                    f"unexpected tool failure ({type(exc).__name__})",
                    code="tool_execution_failed",
                    retryable=isinstance(exc, (TimeoutError, ConnectionError)),
                )
        can_retry = (
            error.retryable
            and attempt < spec.retry_policy.max_attempts
            and spec.replay_policy in {"safe", "idempotent"}
        )
        if can_retry:
            emitter.emit(
                "tool.retrying",
                {
                    "call_id": call.call_id,
                    "tool_name": call.name,
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "error_code": error.code,
                },
            )
            continue
        emitter.emit(
            "tool.failed",
            {
                "call_id": call.call_id,
                "tool_name": call.name,
                "tool_version": spec.version,
                "attempt": attempt,
                "error_code": error.code,
                "error_type": type(last_error).__name__ if last_error else "Error",
            },
        )
        return (
            ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.name,
                ok=False,
                error_code=error.code,
                error_message=str(error)[:500],
                retryable=error.retryable,
            ),
            None,
        )
    raise HarnessContractError("unreachable tool retry state")
