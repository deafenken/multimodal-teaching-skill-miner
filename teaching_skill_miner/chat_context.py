"""Deterministic Chat context compaction backed by a durable full transcript.

The compact view is a transport projection only.  It never replaces the full
project transcript and it never becomes learner evidence or a scoring source.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping, Sequence


class ChatContextError(ValueError):
    """Raised when a Chat transcript cannot be projected safely."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validated_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ChatContextError("Chat transcript must be an array")
    if not 1 <= len(value) <= 10_000:
        raise ChatContextError("Chat transcript length is outside the safety limit")
    result: list[dict[str, str]] = []
    expected_role = "user"
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
            raise ChatContextError(f"messages[{index}] has invalid fields")
        role = item["role"]
        content = item["content"]
        if role != expected_role or role not in {"user", "assistant"}:
            raise ChatContextError("Chat transcript roles must alternate from user")
        if (
            not isinstance(content, str)
            or content != content.strip()
            or not 1 <= len(content) <= 64_000
        ):
            raise ChatContextError(f"messages[{index}].content is invalid")
        result.append({"role": role, "content": content})
        expected_role = "assistant" if role == "user" else "user"
    if result[-1]["role"] != "user":
        raise ChatContextError("the final Chat message must be from user")
    return result


def _snippet(content: str, maximum: int) -> str:
    single_line = " ".join(content.split())
    if len(single_line) <= maximum:
        return single_line
    return single_line[: maximum - 1].rstrip() + "…"


def compact_chat_context(
    messages: Sequence[Mapping[str, str]],
    *,
    max_messages: int = 24,
    max_context_chars: int = 48_000,
    keep_recent_messages: int = 18,
    max_summary_chars: int = 8_000,
    durable_message_count: int = 0,
) -> dict[str, Any]:
    """Return a bounded provider projection plus an auditable receipt.

    The caller must retain ``messages`` in its durable project store.  The
    system appendix is an extractive index, not a model-authored summary, so it
    cannot silently invent decisions or erase the original transcript.
    """

    if not 3 <= max_messages <= 64:
        raise ChatContextError("max_messages is invalid")
    if not 2_000 <= max_context_chars <= 2_000_000:
        raise ChatContextError("max_context_chars is invalid")
    if not 3 <= keep_recent_messages <= max_messages:
        raise ChatContextError("keep_recent_messages is invalid")
    if not 512 <= max_summary_chars <= max_context_chars // 2:
        raise ChatContextError("max_summary_chars is invalid")

    normalized = _validated_messages(messages)
    if (
        isinstance(durable_message_count, bool)
        or not isinstance(durable_message_count, int)
        or not 0 <= durable_message_count <= len(normalized)
    ):
        raise ChatContextError("durable_message_count is invalid")
    original_chars = sum(len(item["content"]) for item in normalized)
    transcript_sha256 = _canonical_sha256(normalized)
    if len(normalized) <= max_messages and original_chars <= max_context_chars:
        return {
            "messages": normalized,
            "system_appendix": "",
            "receipt": {
                "schema": "teaching_skill_miner.chat_context_receipt.v1",
                "compacted": False,
                "original_message_count": len(normalized),
                "projected_message_count": len(normalized),
                "indexed_message_count": 0,
                "unindexed_archived_message_count": 0,
                "original_chars": original_chars,
                "projected_chars": original_chars,
                "transcript_sha256": transcript_sha256,
                "compacted_history_remains_durable": True,
                "full_transcript_remains_durable": (
                    durable_message_count == len(normalized)
                ),
            },
        }

    recent_count = min(keep_recent_messages, len(normalized))
    # An odd-length transcript ends in user.  Starting the retained tail at a
    # user message preserves the provider's strict alternating-role contract.
    if recent_count % 2 == 0:
        recent_count -= 1
    recent = normalized[-recent_count:]
    while (
        sum(len(item["content"]) for item in recent) > max_context_chars - 1_000
        and len(recent) > 1
    ):
        recent = recent[2:]
    recent_chars = sum(len(item["content"]) for item in recent)
    if recent_chars > max_context_chars - 512:
        raise ChatContextError(
            "the current Chat request alone exceeds the safety limit"
        )

    omitted = normalized[: len(normalized) - len(recent)]
    if durable_message_count < len(omitted):
        raise ChatContextError(
            "compacted Chat history must be durably stored before it is omitted"
        )
    available = min(max_summary_chars, max_context_chars - recent_chars - 512)
    selected_lines: list[str] = []
    used = 0
    # Prefer the most recent archived context while retaining original indices
    # and content hashes for audit/retrieval from the durable transcript.
    for index in range(len(omitted) - 1, -1, -1):
        item = omitted[index]
        label = "用户" if item["role"] == "user" else "助手"
        digest = sha256(item["content"].encode("utf-8")).hexdigest()[:12]
        line = f"- {label}消息 {index + 1} [{digest}]：{_snippet(item['content'], 320)}"
        cost = len(line) + 1
        if used + cost > available:
            break
        selected_lines.append(line)
        used += cost
    selected_lines.reverse()
    unindexed = len(omitted) - len(selected_lines)
    appendix = (
        "以下是已持久化早期对话的抽取式索引，不是新的事实、学生证据或评分依据。"
        "回答时优先遵循后面的原始近期消息；需要被省略的原文时应请求读取项目历史，"
        "不得猜测。\n"
        f"完整历史摘要：共 {len(normalized)} 条，早期 {len(omitted)} 条，"
        f"索引 {len(selected_lines)} 条，未展开 {unindexed} 条，"
        f"transcript_sha256={transcript_sha256}。\n" + "\n".join(selected_lines)
    )
    projected_chars = recent_chars + len(appendix)
    if projected_chars > max_context_chars:
        raise ChatContextError("compacted Chat context still exceeds the safety limit")
    return {
        "messages": recent,
        "system_appendix": appendix,
        "receipt": {
            "schema": "teaching_skill_miner.chat_context_receipt.v1",
            "compacted": True,
            "original_message_count": len(normalized),
            "projected_message_count": len(recent),
            "indexed_message_count": len(selected_lines),
            "unindexed_archived_message_count": unindexed,
            "original_chars": original_chars,
            "projected_chars": projected_chars,
            "transcript_sha256": transcript_sha256,
            "compacted_history_remains_durable": True,
            "full_transcript_remains_durable": (
                durable_message_count == len(normalized)
            ),
        },
    }


__all__ = ["ChatContextError", "compact_chat_context"]
