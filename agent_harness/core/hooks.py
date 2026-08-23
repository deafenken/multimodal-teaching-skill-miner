"""Provider-neutral contracts for deterministic local policy hooks.

The core never discovers or executes project files itself. A higher layer may
implement :class:`ToolHookBroker`, but every decision returned through this
seam is intentionally monotonic: hooks can pass, require an approval, or deny;
they cannot grant authority or replace tool arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

from .cancellation import CancellationToken
from .contracts import HarnessContractError
from .events import canonical_json, canonical_sha256


HOOK_EVENT_NAMES = frozenset(
    {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
)
HOOK_ACTIONS = frozenset({"pass", "ask", "deny"})

HookAction = Literal["pass", "ask", "deny"]
HookAuditSink = Callable[[str, Mapping[str, Any]], None]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _snapshot_json(value: Any, *, label: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (HarnessContractError, json.JSONDecodeError) as exc:
        raise HarnessContractError(f"{label} must be canonical JSON") from exc


@dataclass(frozen=True, slots=True)
class ToolHookRequest:
    """One immutable tool-lifecycle input for a trusted hook broker."""

    event_name: str
    run_id: str
    turn_id: str
    session_id: str | None
    call_id: str
    tool_name: str
    tool_version: str
    tool_input: Mapping[str, Any]
    arguments_sha256: str
    result: Any = None
    result_sha256: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.event_name not in HOOK_EVENT_NAMES:
            raise HarnessContractError("hook event name is invalid")
        for label, value in (
            ("run_id", self.run_id),
            ("turn_id", self.turn_id),
            ("call_id", self.call_id),
        ):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise HarnessContractError(f"hook {label} is invalid")
        if self.session_id is not None and (
            not isinstance(self.session_id, str)
            or _IDENTIFIER.fullmatch(self.session_id) is None
        ):
            raise HarnessContractError("hook session_id is invalid")
        if (
            not isinstance(self.tool_name, str)
            or _TOOL_NAME.fullmatch(self.tool_name) is None
        ):
            raise HarnessContractError("hook tool name is invalid")
        if (
            not isinstance(self.tool_version, str)
            or not self.tool_version.strip()
            or len(self.tool_version) > 64
        ):
            raise HarnessContractError("hook tool version is invalid")
        if not isinstance(self.tool_input, Mapping):
            raise HarnessContractError("hook tool input must be an object")
        input_snapshot = _snapshot_json(dict(self.tool_input), label="hook tool input")
        if not isinstance(input_snapshot, dict):
            raise HarnessContractError("hook tool input must be an object")
        object.__setattr__(self, "tool_input", input_snapshot)
        if (
            not isinstance(self.arguments_sha256, str)
            or _SHA256.fullmatch(self.arguments_sha256) is None
            or canonical_sha256(input_snapshot) != self.arguments_sha256
        ):
            raise HarnessContractError("hook tool input digest is invalid")
        if self.event_name == "PreToolUse":
            if (
                self.result is not None
                or self.result_sha256 is not None
                or self.error_code is not None
            ):
                raise HarnessContractError("pre-tool hook cannot contain a tool result")
        elif self.event_name == "PostToolUse":
            result_snapshot = _snapshot_json(self.result, label="hook tool result")
            object.__setattr__(self, "result", result_snapshot)
            if (
                not isinstance(self.result_sha256, str)
                or _SHA256.fullmatch(self.result_sha256) is None
                or canonical_sha256(result_snapshot) != self.result_sha256
                or self.error_code is not None
            ):
                raise HarnessContractError("post-tool hook result binding is invalid")
        else:
            if (
                not isinstance(self.error_code, str)
                or not self.error_code.strip()
                or len(self.error_code) > 128
                or self.result is not None
                or self.result_sha256 is not None
            ):
                raise HarnessContractError("failed-tool hook binding is invalid")

    def to_input(self) -> dict[str, Any]:
        """Return the bounded local JSON object delivered on hook stdin."""

        value: dict[str, Any] = {
            "schema": "agent_harness.hook_input.v1",
            "hook_event_name": self.event_name,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "tool_use_id": self.call_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "tool_input": dict(self.tool_input),
            "arguments_sha256": self.arguments_sha256,
        }
        if self.event_name == "PostToolUse":
            value["tool_result"] = self.result
            value["result_sha256"] = self.result_sha256
        elif self.event_name == "PostToolUseFailure":
            value["error_code"] = self.error_code
        return value


@dataclass(frozen=True, slots=True)
class ToolHookDecision:
    """Aggregated monotonic result from every matching hook."""

    action: HookAction = "pass"
    matched_hook_ids: tuple[str, ...] = ()
    executed_hook_ids: tuple[str, ...] = ()
    skipped_hook_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in HOOK_ACTIONS:
            raise HarnessContractError("hook action is invalid")
        for field_name in (
            "matched_hook_ids",
            "executed_hook_ids",
            "skipped_hook_ids",
        ):
            raw = tuple(getattr(self, field_name))
            if len(raw) > 256 or len(set(raw)) != len(raw):
                raise HarnessContractError(f"{field_name} is invalid")
            if any(_IDENTIFIER.fullmatch(value) is None for value in raw):
                raise HarnessContractError(f"{field_name} is invalid")
            object.__setattr__(self, field_name, raw)
        matched = set(self.matched_hook_ids)
        executed = set(self.executed_hook_ids)
        skipped = set(self.skipped_hook_ids)
        if executed & skipped or not executed <= matched or not skipped <= matched:
            raise HarnessContractError("hook decision identities are inconsistent")

    def validated(self) -> "ToolHookDecision":
        return self


@runtime_checkable
class ToolHookBroker(Protocol):
    """Trusted local hook seam used by the central tool executor."""

    @property
    def policy_material(self) -> Mapping[str, Any]:
        """Return content-free material bound into the run policy digest."""

    def matches(self, event_name: str, tool_name: str) -> bool:
        """Return whether an enabled hook may execute for this lifecycle point."""

    def evaluate(
        self,
        request: ToolHookRequest,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
        audit_sink: HookAuditSink,
    ) -> ToolHookDecision:
        """Run matching hooks and return their most restrictive decision."""


__all__ = [
    "HOOK_ACTIONS",
    "HOOK_EVENT_NAMES",
    "HookAction",
    "HookAuditSink",
    "ToolHookBroker",
    "ToolHookDecision",
    "ToolHookRequest",
]
