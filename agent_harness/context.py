"""Immutable active-context views and bounded compaction planning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from .core.contracts import HarnessContractError


_MESSAGE_OVERHEAD_BYTES = 64
DEFAULT_COMPACTION_RETAIN_MESSAGES = 6
DEFAULT_COMPACTION_SOURCE_BYTES = 32_000


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HarnessContractError("context value must be canonical JSON") from exc


@dataclass(frozen=True, slots=True)
class ContextCompactionResult:
    """Provider-produced summary which has not yet been persisted."""

    summary: str
    usage: Mapping[str, int]
    provider_request_id: str | None = None

    def validated(self) -> "ContextCompactionResult":
        summary = self.summary.strip() if isinstance(self.summary, str) else ""
        if not summary or len(summary) > 40_000:
            raise HarnessContractError("context compaction summary is invalid")
        clean_usage: dict[str, int] = {}
        if not isinstance(self.usage, Mapping) or len(self.usage) > 16:
            raise HarnessContractError("context compaction usage is invalid")
        for key, value in self.usage.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key) > 80
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise HarnessContractError("context compaction usage is invalid")
            clean_usage[key] = value
        request_id = self.provider_request_id
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id.strip():
                raise HarnessContractError(
                    "context compaction provider request id is invalid"
                )
            request_id = request_id.strip()[:160]
        return ContextCompactionResult(
            summary=summary,
            usage=clean_usage,
            provider_request_id=request_id,
        )


@dataclass(frozen=True, slots=True)
class ContextCompactionPlan:
    """One incremental, assistant-boundary summary operation."""

    parent_compaction_id: str | None
    parent_summary: str
    parent_source_message_count: int
    source_message_count: int
    source_messages: tuple[Mapping[str, Any], ...]

    @property
    def advances(self) -> bool:
        return self.source_message_count > self.parent_source_message_count


def select_compaction_plan(
    session: Mapping[str, Any],
    *,
    retain_messages: int = DEFAULT_COMPACTION_RETAIN_MESSAGES,
    max_source_bytes: int = DEFAULT_COMPACTION_SOURCE_BYTES,
) -> ContextCompactionPlan | None:
    """Select the largest bounded new prefix ending on an assistant message."""

    if (
        isinstance(retain_messages, bool)
        or not isinstance(retain_messages, int)
        or retain_messages < 2
    ):
        raise HarnessContractError("compaction retain_messages is invalid")
    if (
        isinstance(max_source_bytes, bool)
        or not isinstance(max_source_bytes, int)
        or max_source_bytes < 1_024
    ):
        raise HarnessContractError("compaction source byte limit is invalid")
    raw_messages = session.get("messages", [])
    raw_compactions = session.get("compactions", [])
    if not isinstance(raw_messages, list) or not isinstance(raw_compactions, list):
        raise HarnessContractError("session context is invalid")
    latest = raw_compactions[-1] if raw_compactions else None
    if latest is not None and not isinstance(latest, Mapping):
        raise HarnessContractError("session compaction lineage is invalid")
    parent_count = int(latest.get("source_message_count", 0)) if latest else 0
    maximum_count = len(raw_messages) - retain_messages
    if maximum_count <= parent_count:
        return None

    selected_count = parent_count
    selected_messages: tuple[Mapping[str, Any], ...] = ()
    for candidate_count in range(parent_count + 1, maximum_count + 1):
        candidate = raw_messages[candidate_count - 1]
        if not isinstance(candidate, Mapping) or candidate.get("role") != "assistant":
            continue
        delta = raw_messages[parent_count:candidate_count]
        if not all(isinstance(item, Mapping) for item in delta):
            raise HarnessContractError("session message is invalid")
        if len(_canonical_bytes(delta)) > max_source_bytes:
            break
        selected_count = candidate_count
        selected_messages = tuple(dict(item) for item in delta)
    if selected_count == parent_count:
        return None
    return ContextCompactionPlan(
        parent_compaction_id=(
            str(latest.get("compaction_id")) if latest is not None else None
        ),
        parent_summary=(str(latest.get("summary", "")) if latest is not None else ""),
        parent_source_message_count=parent_count,
        source_message_count=selected_count,
        source_messages=selected_messages,
    )


def estimate_context_tokens(
    *,
    summary: str,
    messages: Sequence[Mapping[str, Any]],
    project_instruction_bytes: int = 0,
    tool_definitions: Sequence[Mapping[str, Any]] = (),
    prospective_prompt: str = "",
) -> int:
    """Conservative UTF-8 byte upper bound used for automatic compaction."""

    if project_instruction_bytes < 0:
        raise HarnessContractError("project instruction byte count is invalid")
    material = {
        "history_summary": summary,
        "messages": [
            {
                "role": str(item.get("role", "")),
                "content": str(item.get("content", "")),
            }
            for item in messages
        ],
        "prospective_prompt": prospective_prompt,
        "tools": [dict(item) for item in tool_definitions],
    }
    framing = (len(messages) + len(tool_definitions) + 4) * _MESSAGE_OVERHEAD_BYTES
    return len(_canonical_bytes(material)) + project_instruction_bytes + framing


__all__ = [
    "ContextCompactionPlan",
    "ContextCompactionResult",
    "DEFAULT_COMPACTION_RETAIN_MESSAGES",
    "DEFAULT_COMPACTION_SOURCE_BYTES",
    "estimate_context_tokens",
    "select_compaction_plan",
]
