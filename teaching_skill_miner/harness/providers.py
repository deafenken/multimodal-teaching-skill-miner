"""Provider-neutral capabilities and canonical stream events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import HarnessContractError


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    provider: str
    model: str
    structured_output: bool
    native_stream: bool
    native_tools: bool = False
    vision: bool = False
    web_search: bool = False
    cancellation: bool = False

    def validated(self) -> "ProviderCapabilities":
        if not self.provider.strip() or not self.model.strip():
            raise HarnessContractError("provider and model identifiers are required")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validated()
        return {
            "provider": self.provider,
            "model": self.model,
            "structured_output": self.structured_output,
            "native_stream": self.native_stream,
            "native_tools": self.native_tools,
            "vision": self.vision,
            "web_search": self.web_search,
            "cancellation": self.cancellation,
        }


_STREAM_EVENT_TYPES = frozenset(
    {
        "message.start",
        "message.delta",
        "message.end",
        "reasoning.delta",
        "tool_call.delta",
        "usage.update",
    }
)


@dataclass(frozen=True, slots=True)
class ProviderStreamEvent:
    """Canonical provider event; raw vendor envelopes never cross this seam."""

    type: str
    delta: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    channel: str = "assistant"

    def validated(self) -> "ProviderStreamEvent":
        if self.type not in _STREAM_EVENT_TYPES:
            raise HarnessContractError("provider stream event type is unsupported")
        if self.channel not in {"assistant", "internal"}:
            raise HarnessContractError("provider stream channel is invalid")
        if not isinstance(self.delta, str) or len(self.delta) > 32_000:
            raise HarnessContractError("provider stream delta is invalid")
        if not isinstance(self.payload, Mapping):
            raise HarnessContractError("provider stream payload must be an object")
        return self
