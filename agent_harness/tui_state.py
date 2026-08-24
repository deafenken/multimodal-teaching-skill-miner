"""Pure state reducer and Unicode layout helpers for the terminal UI."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
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
_OPAQUE_AGENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OPAQUE_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUBAGENT_STATUSES = frozenset({"running", "completed", "failed", "cancelled"})


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
class SubagentActivity:
    """Content-free foreground child lifecycle metadata.

    Prompts, task labels, summaries, paths, refs, transcripts, and reasoning are
    intentionally absent.  ``artifact_id`` is an opaque reconciliation handle,
    not a filesystem location.
    """

    call_id: str
    agent_id: str
    depth: int
    ordinal: int
    status: str
    artifact_id: str | None = None
    changed: bool | None = None


@dataclass(slots=True)
class SubagentBatchActivity:
    """Bounded foreground batch counts for status-bar observability."""

    call_id: str
    count: int
    depth: int
    status: str = "running"


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
    agents: dict[str, SubagentActivity] = field(default_factory=dict)
    agent_batches: dict[str, SubagentBatchActivity] = field(default_factory=dict)
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
        self.agents = {}
        self.agent_batches = {}
        self.terminal_seen = False

    @property
    def running_agent_count(self) -> int:
        return sum(activity.status == "running" for activity in self.agents.values())

    @property
    def settled_agent_count(self) -> int:
        return len(self.agents) - self.running_agent_count

    def ordered_agents(self) -> tuple[SubagentActivity, ...]:
        """Return deterministic content-free activity for rendering/listing."""

        return tuple(
            sorted(
                self.agents.values(),
                key=lambda item: (item.call_id, item.depth, item.ordinal, item.agent_id),
            )
        )

    @staticmethod
    def _opaque_identifier(value: Any, *, label: str) -> str:
        if not isinstance(value, str) or _OPAQUE_AGENT_ID.fullmatch(value) is None:
            raise ValueError(f"invalid subagent {label}")
        return value

    @staticmethod
    def _bounded_integer(
        value: Any,
        *,
        label: str,
        minimum: int,
        maximum: int,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"invalid subagent {label}")
        return value

    def _settle_running_agents(self, call_id: str, status: str) -> None:
        if status not in {"failed", "cancelled"}:
            raise ValueError("invalid subagent settlement status")
        for activity in self.agents.values():
            if activity.call_id == call_id and activity.status == "running":
                activity.status = status
        batch = self.agent_batches.get(call_id)
        if batch is not None and batch.status == "running":
            batch.status = status

    def _apply_subagent_progress(
        self,
        *,
        call_id: str,
        progress_kind: str,
        progress: Any,
    ) -> None:
        if progress_kind not in {
            "subagent.batch_started",
            "subagent.child_started",
            "subagent.child_finished",
            "subagent.batch_finished",
        }:
            raise ValueError("unknown subagent progress kind")
        if not isinstance(progress, Mapping):
            raise ValueError("subagent progress must be an object")
        if progress_kind in {"subagent.batch_started", "subagent.batch_finished"}:
            if set(progress) != {"count", "depth"}:
                raise ValueError("invalid subagent batch progress fields")
            count = self._bounded_integer(
                progress.get("count"),
                label="batch count",
                minimum=1,
                maximum=4,
            )
            depth = self._bounded_integer(
                progress.get("depth"),
                label="depth",
                minimum=1,
                maximum=8,
            )
            if progress_kind == "subagent.batch_started":
                if call_id in self.agent_batches:
                    raise ValueError("duplicate subagent batch start")
                self.agent_batches[call_id] = SubagentBatchActivity(
                    call_id=call_id,
                    count=count,
                    depth=depth,
                )
                return
            batch = self.agent_batches.get(call_id)
            if batch is None or batch.count != count or batch.depth != depth:
                raise ValueError("subagent batch finish does not match its start")
            activities = [
                item for item in self.agents.values() if item.call_id == call_id
            ]
            if len(activities) != count or any(
                item.status == "running" for item in activities
            ):
                raise ValueError("subagent batch finished before every child settled")
            batch.status = "completed"
            return

        expected_fields = (
            {"agent_id", "depth", "ordinal"}
            if progress_kind == "subagent.child_started"
            else {"agent_id", "depth", "ordinal", "status"}
        )
        if set(progress) != expected_fields:
            raise ValueError("invalid subagent child progress fields")
        agent_id = self._opaque_identifier(progress.get("agent_id"), label="agent id")
        depth = self._bounded_integer(
            progress.get("depth"),
            label="depth",
            minimum=1,
            maximum=8,
        )
        ordinal = self._bounded_integer(
            progress.get("ordinal"),
            label="ordinal",
            minimum=0,
            maximum=255,
        )
        batch = self.agent_batches.get(call_id)
        if (
            batch is None
            or batch.status != "running"
            or batch.depth != depth
            or ordinal >= batch.count
        ):
            raise ValueError("subagent child does not match an active batch")
        existing = self.agents.get(agent_id)
        if progress_kind == "subagent.child_started":
            if existing is not None or any(
                item.call_id == call_id and item.ordinal == ordinal
                for item in self.agents.values()
            ):
                raise ValueError("duplicate subagent child start")
            self.agents[agent_id] = SubagentActivity(
                call_id=call_id,
                agent_id=agent_id,
                depth=depth,
                ordinal=ordinal,
                status="running",
            )
            return
        status = progress.get("status")
        if status not in _SUBAGENT_STATUSES - {"running"}:
            raise ValueError("invalid subagent child status")
        if (
            existing is None
            or existing.call_id != call_id
            or existing.depth != depth
            or existing.ordinal != ordinal
            or existing.status != "running"
        ):
            raise ValueError("subagent child finish does not match its start")
        existing.status = str(status)

    def _apply_subagent_result(self, *, call_id: str, result: Any) -> None:
        """Enrich lifecycle entries using only content-free result fields."""

        if not isinstance(result, Mapping):
            return
        if result.get("schema") != "agent_harness.subagent_batch.v1":
            return
        raw_results = result.get("results")
        if not isinstance(raw_results, list) or len(raw_results) > 4:
            raise ValueError("invalid subagent result list")
        seen: set[str] = set()
        for raw in raw_results:
            if not isinstance(raw, Mapping):
                raise ValueError("invalid subagent result entry")
            # Deliberately do not read summary, task_id, session IDs, output,
            # path/ref fields, raw errors, or any unknown extension field.
            agent_id = self._opaque_identifier(raw.get("agent_id"), label="agent id")
            if agent_id in seen:
                raise ValueError("duplicate subagent result entry")
            seen.add(agent_id)
            activity = self.agents.get(agent_id)
            if activity is None or activity.call_id != call_id:
                raise ValueError("subagent result has no lifecycle entry")
            depth = self._bounded_integer(
                raw.get("depth"),
                label="depth",
                minimum=1,
                maximum=8,
            )
            ordinal = self._bounded_integer(
                raw.get("ordinal"),
                label="ordinal",
                minimum=0,
                maximum=255,
            )
            status = raw.get("status")
            changed = raw.get("changed")
            if (
                depth != activity.depth
                or ordinal != activity.ordinal
                or status != activity.status
                or not isinstance(changed, bool)
            ):
                raise ValueError("subagent result conflicts with lifecycle metadata")
            artifact_id = raw.get("artifact_id")
            if artifact_id is not None and (
                not isinstance(artifact_id, str)
                or _OPAQUE_ARTIFACT_ID.fullmatch(artifact_id) is None
            ):
                raise ValueError("invalid subagent artifact id")
            activity.changed = changed
            activity.artifact_id = artifact_id
        expected = {
            activity.agent_id
            for activity in self.agents.values()
            if activity.call_id == call_id
        }
        if seen != expected:
            raise ValueError("subagent result does not settle the complete batch")

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
        elif event_type == "approval.requested":
            self.status = "approval"
        elif event_type == "approval.resolved":
            self.status = "running"
        elif event_type.startswith("hook."):
            self.status = "hook"
        elif event_type.startswith("tool."):
            call_id = str(payload.get("call_id", "tool"))[:160]
            name = str(payload.get("tool_name", "tool"))[:160]
            phase = event_type.removeprefix("tool.")
            detail = str(payload.get("error_code", payload.get("progress_kind", "")))[:160]
            if name == "agent.delegate":
                if event_type == "tool.progress":
                    progress_kind = payload.get("progress_kind")
                    if not isinstance(progress_kind, str):
                        raise ValueError("subagent progress kind is invalid")
                    self._apply_subagent_progress(
                        call_id=call_id,
                        progress_kind=progress_kind,
                        progress=payload.get("progress"),
                    )
                elif event_type == "tool.completed":
                    self._apply_subagent_result(
                        call_id=call_id,
                        result=payload.get("result"),
                    )
                elif event_type in {"tool.failed", "tool.rejected"}:
                    self._settle_running_agents(call_id, "failed")
            self.tools[call_id] = ToolActivity(
                call_id=call_id,
                name=name,
                phase=phase,
                detail=detail,
            )
            self.status = "agents" if self.running_agent_count else "tool"
        elif event_type in {"run.completed", "run.cancelled", "run.failed", "run.handoff"}:
            terminal_agent_status = (
                "cancelled" if event_type == "run.cancelled" else "failed"
            )
            for call_id in tuple(self.agent_batches):
                self._settle_running_agents(call_id, terminal_agent_status)
            self.status = event_type.removeprefix("run.")
            self.terminal_seen = True


__all__ = [
    "ToolActivity",
    "SubagentActivity",
    "SubagentBatchActivity",
    "TuiState",
    "display_width",
    "sanitize_terminal_text",
    "truncate_display",
    "wrap_display",
]
