"""Provider-neutral capabilities and canonical stream events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import HarnessContractError


_ATTACHMENT_KINDS = frozenset({"text", "image", "pdf"})


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
    attachment_kinds: tuple[str, ...] = ()
    attachment_mime_types: tuple[str, ...] = ()
    max_attachment_count: int = 0
    max_attachment_bytes: int = 0

    def validated(self) -> "ProviderCapabilities":
        if (
            not isinstance(self.provider, str)
            or not self.provider.strip()
            or len(self.provider) > 128
            or not isinstance(self.model, str)
            or not self.model.strip()
            or len(self.model) > 160
        ):
            raise HarnessContractError("provider and model identifiers are required")
        for value in (
            self.structured_output,
            self.native_stream,
            self.native_tools,
            self.vision,
            self.web_search,
            self.cancellation,
        ):
            if type(value) is not bool:
                raise HarnessContractError(
                    "provider capability flags must be exact booleans"
                )
        if type(self.attachment_kinds) is not tuple or any(
            not isinstance(item, str) for item in self.attachment_kinds
        ):
            raise HarnessContractError("provider attachment kinds are invalid")
        if (
            len(set(self.attachment_kinds)) != len(self.attachment_kinds)
            or set(self.attachment_kinds) - _ATTACHMENT_KINDS
        ):
            raise HarnessContractError("provider attachment kinds are invalid")
        if type(self.attachment_mime_types) is not tuple or any(
            not isinstance(item, str) for item in self.attachment_mime_types
        ):
            raise HarnessContractError("provider attachment MIME types are invalid")
        if len(set(self.attachment_mime_types)) != len(self.attachment_mime_types):
            raise HarnessContractError("provider attachment MIME types are invalid")
        for media_type in self.attachment_mime_types:
            base_type, separator, parameter = media_type.partition(";")
            if (
                not media_type
                or media_type != media_type.strip().casefold()
                or "/" not in base_type
                or len(media_type) > 127
                or any(character.isspace() for character in base_type)
                or (
                    separator
                    and (
                        not base_type.startswith("text/")
                        or parameter != " charset=utf-8"
                    )
                )
            ):
                raise HarnessContractError(
                    "provider attachment MIME types are invalid"
                )
            kind = (
                "text"
                if base_type.startswith("text/")
                else "image"
                if base_type.startswith("image/")
                else "pdf"
                if base_type == "application/pdf" and not separator
                else None
            )
            if kind is None or kind not in self.attachment_kinds:
                raise HarnessContractError(
                    "provider attachment MIME types do not match kinds"
                )
        for kind in self.attachment_kinds:
            if not any(
                (kind == "text" and media_type.startswith("text/"))
                or (kind == "image" and media_type.startswith("image/"))
                or (kind == "pdf" and media_type == "application/pdf")
                for media_type in self.attachment_mime_types
            ):
                raise HarnessContractError(
                    "provider attachment kinds require an exact MIME type"
                )
        if (
            type(self.max_attachment_count) is not int
            or type(self.max_attachment_bytes) is not int
            or not 0 <= self.max_attachment_count <= 256
            or not 0 <= self.max_attachment_bytes <= 1024 * 1024 * 1024
        ):
            raise HarnessContractError("provider attachment limits are invalid")
        has_attachments = bool(self.attachment_kinds)
        if (
            has_attachments
            and (
                not self.attachment_mime_types
                or self.max_attachment_count == 0
                or self.max_attachment_bytes == 0
            )
        ) or (
            not has_attachments
            and (
                self.attachment_mime_types
                or self.max_attachment_count != 0
                or self.max_attachment_bytes != 0
            )
        ):
            raise HarnessContractError("provider attachment limits are inconsistent")
        if "image" in self.attachment_kinds and not self.vision:
            raise HarnessContractError(
                "provider image attachments require vision capability"
            )
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
            "attachment_kinds": list(self.attachment_kinds),
            "attachment_mime_types": list(self.attachment_mime_types),
            "max_attachment_count": self.max_attachment_count,
            "max_attachment_bytes": self.max_attachment_bytes,
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
