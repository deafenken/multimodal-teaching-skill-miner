"""Fail-closed approval contracts for effectful tool calls.

This module deliberately knows nothing about CLI or TUI interaction.  It
binds an approval to one canonical tool call and lets adapters implement the
human interaction through :class:`ApprovalBroker`.  Raw tool arguments are
never retained in approval requests, decisions, or persistent rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import secrets
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from .contracts import HarnessContractError, RISK_LEVELS
from .events import canonical_json, canonical_sha256


ApprovalAction = Literal["allow", "ask", "deny"]
ApprovalVerdict = Literal["allow", "deny"]

_ACTIONS = frozenset({"allow", "ask", "deny"})
_VERDICTS = frozenset({"allow", "deny"})
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class ApprovalContractError(HarnessContractError):
    """Raised when an approval object is malformed or incorrectly bound."""


def arguments_sha256(arguments: Mapping[str, Any]) -> str:
    """Return the stable canonical digest used by every approval contract."""

    if not isinstance(arguments, Mapping):
        raise ApprovalContractError("approval arguments must be an object")
    return canonical_sha256(dict(arguments))


def _validate_text(value: str, field: str, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ApprovalContractError(f"{field} is invalid")


def _validate_sha256(value: str, field: str) -> None:
    _validate_text(value, field, _SHA256)


@dataclass(frozen=True, slots=True)
class ApprovalRule:
    """One persistent, argument-free approval policy rule.

    A rule with only ``tool_name`` is tool-wide.  An exact rule must provide
    both ``tool_version`` and ``arguments_sha256``.  ``tool_name='*'`` is also
    accepted for an all-tools default rule, but cannot be made exact.
    """

    action: ApprovalAction
    tool_name: str
    tool_version: str | None = None
    arguments_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.action not in _ACTIONS:
            raise ApprovalContractError("approval rule action is invalid")
        if self.tool_name != "*":
            _validate_text(self.tool_name, "approval rule tool_name", _TOOL_NAME)
        exact_parts = (self.tool_version is not None, self.arguments_sha256 is not None)
        if exact_parts[0] != exact_parts[1]:
            raise ApprovalContractError(
                "exact approval rules require tool_version and arguments_sha256"
            )
        if self.tool_name == "*" and exact_parts[0]:
            raise ApprovalContractError("all-tools approval rules cannot be exact")
        if self.tool_version is not None:
            if (
                not isinstance(self.tool_version, str)
                or not self.tool_version.strip()
                or len(self.tool_version) > 64
            ):
                raise ApprovalContractError("approval rule tool_version is invalid")
            _validate_sha256(
                self.arguments_sha256 or "", "approval rule arguments_sha256"
            )

    @property
    def exact(self) -> bool:
        return self.tool_version is not None

    def matches(
        self,
        *,
        tool_name: str,
        tool_version: str,
        arguments_sha256: str,
    ) -> bool:
        if self.tool_name not in {"*", tool_name}:
            return False
        if not self.exact:
            return True
        return (
            self.tool_version == tool_version
            and self.arguments_sha256 == arguments_sha256
        )

    def material(self) -> dict[str, str]:
        value = {"action": self.action, "tool_name": self.tool_name}
        if self.exact:
            value["tool_version"] = self.tool_version or ""
            value["arguments_sha256"] = self.arguments_sha256 or ""
        return value


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    """Immutable approval rules with conservative risk defaults.

    Matching rules are combined by safety precedence: ``deny`` wins over
    ``ask``, which wins over ``allow``.  Rule order therefore cannot weaken a
    policy.  With no matching rule, low-risk calls are allowed while medium-
    and high-risk calls require an explicit approval.
    """

    rules: tuple[ApprovalRule, ...] = ()
    low_risk: ApprovalAction = "allow"
    medium_risk: ApprovalAction = "ask"
    high_risk: ApprovalAction = "ask"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        if not all(isinstance(rule, ApprovalRule) for rule in self.rules):
            raise ApprovalContractError("approval policy rules are invalid")
        for field, value in (
            ("low_risk", self.low_risk),
            ("medium_risk", self.medium_risk),
            ("high_risk", self.high_risk),
        ):
            if value not in _ACTIONS:
                raise ApprovalContractError(f"approval policy {field} is invalid")

    def action_for(
        self,
        *,
        tool_name: str,
        tool_version: str,
        arguments_sha256: str,
        risk: str,
    ) -> ApprovalAction:
        _validate_text(tool_name, "approval tool_name", _TOOL_NAME)
        if not isinstance(tool_version, str) or not tool_version.strip():
            raise ApprovalContractError("approval tool_version is invalid")
        _validate_sha256(arguments_sha256, "approval arguments_sha256")
        if risk not in RISK_LEVELS:
            raise ApprovalContractError("approval risk is invalid")

        matched = {
            rule.action
            for rule in self.rules
            if rule.matches(
                tool_name=tool_name,
                tool_version=tool_version,
                arguments_sha256=arguments_sha256,
            )
        }
        for action in ("deny", "ask", "allow"):
            if action in matched:
                return action  # type: ignore[return-value]
        return {
            "low": self.low_risk,
            "medium": self.medium_risk,
            "high": self.high_risk,
        }[risk]

    def material(self) -> dict[str, Any]:
        # Rules are set-like policy statements.  Canonical sorting makes the
        # policy digest stable even when a configuration loader changes order.
        rules = sorted(
            (rule.material() for rule in self.rules),
            key=canonical_json,
        )
        return {
            "schema": "agent_harness.approval_policy.v1",
            "defaults": {
                "high": self.high_risk,
                "low": self.low_risk,
                "medium": self.medium_risk,
            },
            "rules": rules,
        }

    @property
    def policy_sha256(self) -> str:
        return canonical_sha256(self.material())


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """An approval challenge cryptographically bound to one tool call."""

    approval_id: str
    run_id: str
    call_id: str
    tool_name: str
    tool_version: str
    arguments_sha256: str
    policy_sha256: str
    risk: str
    persistent_scope_allowed: bool = True

    def __post_init__(self) -> None:
        _validate_text(self.approval_id, "approval_id", _IDENTIFIER)
        _validate_text(self.run_id, "approval run_id", _IDENTIFIER)
        _validate_text(self.call_id, "approval call_id", _IDENTIFIER)
        _validate_text(self.tool_name, "approval tool_name", _TOOL_NAME)
        if (
            not isinstance(self.tool_version, str)
            or not self.tool_version.strip()
            or len(self.tool_version) > 64
        ):
            raise ApprovalContractError("approval tool_version is invalid")
        _validate_sha256(self.arguments_sha256, "approval arguments_sha256")
        _validate_sha256(self.policy_sha256, "approval policy_sha256")
        if self.risk not in RISK_LEVELS:
            raise ApprovalContractError("approval risk is invalid")
        if type(self.persistent_scope_allowed) is not bool:
            raise ApprovalContractError(
                "approval persistent-scope flag is invalid"
            )

    @classmethod
    def for_call(
        cls,
        *,
        run_id: str,
        call_id: str,
        tool_name: str,
        tool_version: str,
        arguments: Mapping[str, Any],
        policy: ApprovalPolicy,
        risk: str,
        approval_id: str | None = None,
        persistent_scope_allowed: bool = True,
    ) -> "ApprovalRequest":
        return cls(
            approval_id=approval_id or f"approval_{secrets.token_hex(16)}",
            run_id=run_id,
            call_id=call_id,
            tool_name=tool_name,
            tool_version=tool_version,
            arguments_sha256=arguments_sha256(arguments),
            policy_sha256=policy.policy_sha256,
            risk=risk,
            persistent_scope_allowed=persistent_scope_allowed,
        )

    def binding(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "run_id": self.run_id,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "arguments_sha256": self.arguments_sha256,
            "policy_sha256": self.policy_sha256,
            "persistent_scope_allowed": self.persistent_scope_allowed,
        }


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """A broker response bound to every security-relevant request field."""

    verdict: ApprovalVerdict
    approval_id: str
    run_id: str
    call_id: str
    tool_name: str
    tool_version: str
    arguments_sha256: str
    policy_sha256: str
    reason_code: str
    persistent_scope_allowed: bool = True

    def __post_init__(self) -> None:
        if self.verdict not in _VERDICTS:
            raise ApprovalContractError("approval decision verdict is invalid")
        _validate_text(self.approval_id, "approval_id", _IDENTIFIER)
        _validate_text(self.run_id, "approval run_id", _IDENTIFIER)
        _validate_text(self.call_id, "approval call_id", _IDENTIFIER)
        _validate_text(self.tool_name, "approval tool_name", _TOOL_NAME)
        if (
            not isinstance(self.tool_version, str)
            or not self.tool_version.strip()
            or len(self.tool_version) > 64
        ):
            raise ApprovalContractError("approval tool_version is invalid")
        _validate_sha256(self.arguments_sha256, "approval arguments_sha256")
        _validate_sha256(self.policy_sha256, "approval policy_sha256")
        _validate_text(self.reason_code, "approval reason_code", _REASON_CODE)
        if type(self.persistent_scope_allowed) is not bool:
            raise ApprovalContractError(
                "approval persistent-scope flag is invalid"
            )

    @classmethod
    def for_request(
        cls,
        request: ApprovalRequest,
        *,
        verdict: ApprovalVerdict,
        reason_code: str,
    ) -> "ApprovalDecision":
        return cls(
            verdict=verdict,
            reason_code=reason_code,
            **request.binding(),
        )

    def binding(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "run_id": self.run_id,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "arguments_sha256": self.arguments_sha256,
            "policy_sha256": self.policy_sha256,
            "persistent_scope_allowed": self.persistent_scope_allowed,
        }


@runtime_checkable
class ApprovalBroker(Protocol):
    """Synchronous interaction boundary implemented by a CLI or TUI."""

    def decide(
        self,
        request: ApprovalRequest,
        *,
        preview: str,
    ) -> ApprovalDecision:
        """Return one decision; ``preview`` is transient UI-only context."""


class HeadlessApprovalBroker:
    """Non-interactive broker which always fails closed."""

    def decide(
        self,
        request: ApprovalRequest,
        *,
        preview: str = "",
    ) -> ApprovalDecision:
        del preview
        return ApprovalDecision.for_request(
            request,
            verdict="deny",
            reason_code="approval_unavailable",
        )


def validate_decision(
    request: ApprovalRequest,
    decision: ApprovalDecision,
) -> ApprovalDecision:
    """Validate that a broker response belongs to the current request."""

    if not isinstance(decision, ApprovalDecision):
        raise ApprovalContractError("approval broker returned an invalid decision")
    if decision.binding() != request.binding():
        raise ApprovalContractError("approval decision binding does not match request")
    return decision


def _deny(request: ApprovalRequest, reason_code: str) -> ApprovalDecision:
    return ApprovalDecision.for_request(
        request,
        verdict="deny",
        reason_code=reason_code,
    )


def resolve_approval(
    request: ApprovalRequest,
    policy: ApprovalPolicy,
    broker: ApprovalBroker | None = None,
    *,
    preview: str = "",
) -> ApprovalDecision:
    """Resolve one request without allowing broker failures to escape.

    Policy changes, malformed/stale responses, missing interactive brokers,
    and broker exceptions are all normalized to a bound denial.
    """

    if request.policy_sha256 != policy.policy_sha256:
        return _deny(request, "approval_policy_changed")
    try:
        action = policy.action_for(
            tool_name=request.tool_name,
            tool_version=request.tool_version,
            arguments_sha256=request.arguments_sha256,
            risk=request.risk,
        )
    except Exception:
        return _deny(request, "approval_policy_invalid")
    if action == "allow":
        return ApprovalDecision.for_request(
            request,
            verdict="allow",
            reason_code="approval_policy_allowed",
        )
    if action == "deny":
        return _deny(request, "approval_policy_denied")

    active_broker: ApprovalBroker = broker or HeadlessApprovalBroker()
    try:
        decision = active_broker.decide(request, preview=str(preview)[:20_000])
    except Exception:
        return _deny(request, "approval_broker_failed")
    try:
        return validate_decision(request, decision)
    except Exception:
        return _deny(request, "approval_response_invalid")


__all__ = [
    "ApprovalAction",
    "ApprovalBroker",
    "ApprovalContractError",
    "ApprovalDecision",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalRule",
    "ApprovalVerdict",
    "HeadlessApprovalBroker",
    "arguments_sha256",
    "resolve_approval",
    "validate_decision",
]
