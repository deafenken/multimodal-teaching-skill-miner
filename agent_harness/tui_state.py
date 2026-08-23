"""Pure state reducer and Unicode layout helpers for the terminal UI."""

from __future__ import annotations

from dataclasses import dataclass, field
import unicodedata
from typing import Any, Mapping

from .core import HARNESS_EVENT_SCHEMA, HARNESS_EVENT_TYPES


_BIDI_CONTROLS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)


def sanitize_terminal_text(value: Any) -> str:
    text = str(value or "")
    return "".join(
        character
        if (
            character in {"\n", "\t"}
            or 32 <= ord(character) < 127
            or ord(character) >= 160
        )
        and character not in _BIDI_CONTROLS
        else "�"
        for character in text
    )


def display_width(value: str) -> int:
    width = 0
    for character in value:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return width


def truncate_display(value: str, maximum: int) -> str:
    if maximum <= 0:
        return ""
    result: list[str] = []
    width = 0
    for character in value:
        character_width = 0 if unicodedata.combining(character) else (
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        )
        if width + character_width > maximum:
            break
        result.append(character)
        width += character_width
    return "".join(result)


def wrap_display(value: str, maximum: int) -> list[str]:
    if maximum <= 1:
        return [truncate_display(value, max(0, maximum))]
    lines: list[str] = []
    for paragraph in sanitize_terminal_text(value).splitlines() or [""]:
        current = ""
        for character in paragraph:
            if current and display_width(current + character) > maximum:
                lines.append(current)
                current = character
            else:
                current += character
        lines.append(current)
    return lines or [""]


@dataclass(slots=True)
class ToolActivity:
    call_id: str
    name: str
    phase: str
    detail: str = ""


@dataclass(slots=True)
class TuiState:
    session_id: str
    status: str = "idle"
    active_run_id: str | None = None
    active_turn_id: str | None = None
    last_sequence: int = 0
    assistant_draft: str = ""
    reasoning_chars: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    tools: dict[str, ToolActivity] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)
    terminal_seen: bool = False

    def begin_turn(self) -> None:
        self.status = "accepted"
        self.active_run_id = None
        self.active_turn_id = None
        self.last_sequence = 0
        self.assistant_draft = ""
        self.reasoning_chars = 0
        self.usage = {}
        self.tools = {}
        self.terminal_seen = False

    def notice(self, value: str) -> None:
        clean = " ".join(sanitize_terminal_text(value).split())[:500]
        if clean:
            self.notices.append(clean)
            self.notices[:] = self.notices[-20:]

    def apply_event(self, event: Mapping[str, Any]) -> None:
        if event.get("schema") != HARNESS_EVENT_SCHEMA:
            raise ValueError("unsupported harness event schema")
        event_type = event.get("type")
        run_id = event.get("run_id")
        turn_id = event.get("turn_id")
        sequence = event.get("sequence")
        if not isinstance(event_type, str) or not isinstance(run_id, str) or not isinstance(turn_id, str):
            raise ValueError("invalid harness event identity")
        if event_type not in HARNESS_EVENT_TYPES:
            raise ValueError("unknown harness event type")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise ValueError("invalid harness event sequence")
        if self.active_run_id is None:
            if event_type != "run.started" or sequence != 1:
                raise ValueError("event stream must begin with run.started sequence 1")
            self.active_run_id = run_id
            self.active_turn_id = turn_id
        elif run_id != self.active_run_id or turn_id != self.active_turn_id:
            raise ValueError("cross-run event rejected")
        if sequence != self.last_sequence + 1:
            raise ValueError("non-monotonic event sequence rejected")
        if self.terminal_seen:
            raise ValueError("event after terminal rejected")
        self.last_sequence = sequence
        payload = event.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}

        if event_type == "run.started":
            self.status = "running"
        elif event_type == "model.started":
            self.status = "thinking"
        elif event_type == "message.start":
            self.status = "responding"
        elif event_type == "message.delta":
            if payload.get("channel", "assistant") != "internal":
                self.assistant_draft += sanitize_terminal_text(payload.get("delta", ""))
            self.status = "responding"
        elif event_type == "reasoning.delta":
            chars = payload.get("chars", 0)
            if isinstance(chars, int) and not isinstance(chars, bool) and chars >= 0:
                self.reasoning_chars += chars
            self.status = "thinking"
        elif event_type == "usage.update":
            for key, value in payload.items():
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    clean_key = str(key)
                    self.usage[clean_key] = self.usage.get(clean_key, 0) + value
        elif event_type.startswith("tool."):
            call_id = str(payload.get("call_id", "tool"))[:160]
            name = str(payload.get("tool_name", "tool"))[:160]
            phase = event_type.removeprefix("tool.")
            detail = str(payload.get("error_code", payload.get("progress_kind", "")))[:160]
            self.tools[call_id] = ToolActivity(call_id=call_id, name=name, phase=phase, detail=detail)
            self.status = "tool"
        elif event_type in {"run.completed", "run.cancelled", "run.failed", "run.handoff"}:
            self.status = event_type.removeprefix("run.")
            self.terminal_seen = True


__all__ = [
    "ToolActivity",
    "TuiState",
    "display_width",
    "sanitize_terminal_text",
    "truncate_display",
    "wrap_display",
]
