"""Small, explicit provider registry and conformance boundary.

The registry is intentionally narrower than coding-agent vendor catalogs.  It
selects only adapters that have declared and tested the capabilities required
by one operation; model-name conditionals do not leak into planners.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, Callable, Protocol

from .contracts import HarnessContractError, HarnessModelRequest, HarnessModelResponse
from .providers import ProviderCapabilities, ProviderStreamEvent


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")

class ProviderErrorKind(str, Enum):
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_OVERFLOW = "context_overflow"
    SAFETY_REFUSAL = "safety_refusal"
    CANCELLED = "cancelled"
    MALFORMED_RESPONSE = "malformed_response"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    kind: ProviderErrorKind
    retryable: bool
    safe_code: str
    retry_after_seconds: float | None = None

    def validated(self) -> "ProviderFailure":
        if not isinstance(self.kind, ProviderErrorKind):
            raise HarnessContractError("provider failure kind is invalid")
        if type(self.retryable) is not bool:
            raise HarnessContractError("provider failure retryable must be boolean")
        if (
            not isinstance(self.safe_code, str)
            or _IDENTIFIER.fullmatch(self.safe_code) is None
        ):
            raise HarnessContractError("provider failure safe_code is invalid")
        if self.retry_after_seconds is not None and (
            not math.isfinite(self.retry_after_seconds)
            or not 0 <= self.retry_after_seconds <= 3_600
        ):
            raise HarnessContractError("provider retry_after_seconds is invalid")
        retryable_kinds = {
            ProviderErrorKind.RATE_LIMIT,
            ProviderErrorKind.TIMEOUT,
            ProviderErrorKind.UNAVAILABLE,
        }
        if self.retryable and self.kind not in retryable_kinds:
            raise HarnessContractError("provider failure retryability is unsafe")
        if self.retry_after_seconds is not None and not self.retryable:
            raise HarnessContractError(
                "provider retry_after_seconds requires a retryable failure"
            )
        return self


@dataclass(frozen=True, slots=True)
class ProviderModelSpec:
    provider: str
    model: str
    capabilities: ProviderCapabilities
    context_window_tokens: int
    maximum_output_tokens: int
    tokenizer: str = "approx_unicode_v1"

    def validated(self) -> "ProviderModelSpec":
        if _IDENTIFIER.fullmatch(self.provider) is None:
            raise HarnessContractError("provider identifier is invalid")
        if not self.model.strip() or len(self.model) > 160:
            raise HarnessContractError("model identifier is invalid")
        self.capabilities.validated()
        if (
            self.capabilities.provider != self.provider
            or self.capabilities.model != self.model
        ):
            raise HarnessContractError("provider model capabilities do not match spec")
        if not 1_024 <= self.context_window_tokens <= 20_000_000:
            raise HarnessContractError("context window is invalid")
        if not 64 <= self.maximum_output_tokens < self.context_window_tokens:
            raise HarnessContractError("maximum output tokens is invalid")
        if self.tokenizer not in {"approx_unicode_v1"}:
            raise HarnessContractError("tokenizer is unsupported")
        return self

    def supports(self, required: set[str]) -> bool:
        flags = {
            "structured_output": self.capabilities.structured_output,
            "native_stream": self.capabilities.native_stream,
            "native_tools": self.capabilities.native_tools,
            "vision": self.capabilities.vision,
            "web_search": self.capabilities.web_search,
            "cancellation": self.capabilities.cancellation,
            "text_attachment": "text" in self.capabilities.attachment_kinds,
            "image_attachment": "image" in self.capabilities.attachment_kinds,
            "pdf_attachment": "pdf" in self.capabilities.attachment_kinds,
        }
        unknown = required - set(flags)
        if unknown:
            raise HarnessContractError(
                "unknown provider capabilities: " + ", ".join(sorted(unknown))
            )
        return all(flags[name] for name in required)


class ProviderAdapter(Protocol):
    @property
    def model_spec(self) -> ProviderModelSpec: ...

    def plan(
        self, request: HarnessModelRequest, **kwargs: Any
    ) -> HarnessModelResponse: ...

    def classify_error(self, error: BaseException) -> ProviderFailure: ...


ProviderFactory = Callable[[], ProviderAdapter]


def approximate_tokens(value: str) -> int:
    """Conservative, deterministic estimate used for preflight only."""

    if not isinstance(value, str):
        raise HarnessContractError("token estimate input must be text")
    # CJK and other non-ASCII symbols tend to occupy roughly one token each;
    # ASCII prose/code is conservatively budgeted at one token per three bytes.
    ascii_count = sum(1 for character in value if ord(character) < 128)
    non_ascii_count = len(value) - ascii_count
    return non_ascii_count + math.ceil(ascii_count / 3)


class ProviderRegistry:
    """One authoritative configured set of provider/model adapters."""

    def __init__(self) -> None:
        self._entries: dict[
            tuple[str, str], tuple[ProviderModelSpec, ProviderFactory]
        ] = {}

    def register(self, spec: ProviderModelSpec, factory: ProviderFactory) -> None:
        spec.validated()
        if not callable(factory):
            raise HarnessContractError("provider factory must be callable")
        key = (spec.provider, spec.model)
        if key in self._entries:
            raise HarnessContractError("provider model is already registered")
        self._entries[key] = (spec, factory)

    def list(self) -> tuple[ProviderModelSpec, ...]:
        return tuple(
            spec
            for spec, _factory in sorted(
                self._entries.values(), key=lambda row: (row[0].provider, row[0].model)
            )
        )

    def resolve(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        required_capabilities: set[str] | None = None,
        minimum_context_tokens: int = 0,
    ) -> ProviderAdapter:
        required = set(required_capabilities or set())
        candidates = [
            (spec, factory)
            for (candidate_provider, candidate_model), (
                spec,
                factory,
            ) in self._entries.items()
            if (provider is None or candidate_provider == provider)
            and (model is None or candidate_model == model)
            and spec.supports(required)
            and spec.context_window_tokens >= minimum_context_tokens
        ]
        if not candidates:
            raise HarnessContractError(
                "no configured provider satisfies this operation"
            )
        # Deterministic preference: exact caller selection first, then the
        # smallest sufficient context window to avoid silently escalating cost.
        candidates.sort(
            key=lambda row: (
                row[0].context_window_tokens,
                row[0].provider,
                row[0].model,
            )
        )
        spec, factory = candidates[0]
        adapter = factory()
        if adapter.model_spec.validated() != spec:
            raise HarnessContractError(
                "provider factory returned a different model spec"
            )
        return adapter

    def public_status(self) -> list[dict[str, Any]]:
        return [
            {
                "provider": spec.provider,
                "model": spec.model,
                "context_window_tokens": spec.context_window_tokens,
                "maximum_output_tokens": spec.maximum_output_tokens,
                "capabilities": spec.capabilities.to_dict(),
            }
            for spec in self.list()
        ]


def provider_stream_contract(events: list[ProviderStreamEvent]) -> None:
    """Validate the canonical start/delta/end ordering for conformance tests."""

    if not events or events[0].type != "message.start":
        raise HarnessContractError("provider stream must start with message.start")
    if events[-1].type != "message.end":
        raise HarnessContractError("provider stream must end with message.end")
    if sum(event.type == "message.start" for event in events) != 1:
        raise HarnessContractError("provider stream has duplicate message.start")
    if sum(event.type == "message.end" for event in events) != 1:
        raise HarnessContractError("provider stream has duplicate message.end")
    for event in events:
        event.validated()


__all__ = [
    "ProviderAdapter",
    "ProviderErrorKind",
    "ProviderFailure",
    "ProviderModelSpec",
    "ProviderRegistry",
    "approximate_tokens",
    "provider_stream_contract",
]
