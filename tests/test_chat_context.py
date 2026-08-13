from __future__ import annotations

import pytest

from teaching_skill_miner.chat_context import ChatContextError, compact_chat_context


def _messages(turns: int, *, width: int = 40) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index in range(turns):
        result.append({"role": "user", "content": f"用户问题 {index} " + "问" * width})
        if index + 1 < turns:
            result.append(
                {"role": "assistant", "content": f"助手回答 {index} " + "答" * width}
            )
    return result


def test_short_chat_is_not_compacted() -> None:
    messages = _messages(3)
    result = compact_chat_context(messages)
    assert result["messages"] == messages
    assert result["system_appendix"] == ""
    assert result["receipt"]["compacted"] is False


def test_long_chat_uses_extract_index_and_keeps_recent_raw_turns() -> None:
    messages = _messages(30, width=160)
    result = compact_chat_context(
        messages,
        max_messages=24,
        max_context_chars=7_000,
        keep_recent_messages=9,
        max_summary_chars=2_000,
        durable_message_count=len(messages) - 1,
    )
    projected = result["messages"]
    receipt = result["receipt"]
    assert projected == messages[-9:]
    assert projected[0]["role"] == "user"
    assert projected[-1]["role"] == "user"
    assert receipt["compacted"] is True
    assert receipt["compacted_history_remains_durable"] is True
    assert receipt["full_transcript_remains_durable"] is False
    assert receipt["original_message_count"] == len(messages)
    assert "不是新的事实、学生证据或评分依据" in result["system_appendix"]
    assert "transcript_sha256=" in result["system_appendix"]


def test_compaction_reports_archived_messages_that_do_not_fit_index() -> None:
    result = compact_chat_context(
        _messages(60, width=300),
        max_messages=24,
        max_context_chars=5_000,
        keep_recent_messages=5,
        max_summary_chars=512,
        durable_message_count=len(_messages(60, width=300)) - 1,
    )
    assert result["receipt"]["unindexed_archived_message_count"] > 0
    assert "不得猜测" in result["system_appendix"]


def test_invalid_roles_and_oversized_current_request_fail_closed() -> None:
    with pytest.raises(ChatContextError, match="alternate"):
        compact_chat_context(
            [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ]
        )
    with pytest.raises(ChatContextError, match="current Chat request"):
        compact_chat_context(
            [{"role": "user", "content": "x" * 4_500}],
            max_context_chars=4_000,
            max_summary_chars=1_000,
        )


def test_long_chat_without_durable_compacted_prefix_fails_closed() -> None:
    with pytest.raises(ChatContextError, match="durably stored"):
        compact_chat_context(_messages(30))
