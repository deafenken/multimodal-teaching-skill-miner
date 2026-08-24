from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from agent_harness.core import HARNESS_EVENT_SCHEMA
from agent_harness.tui import HarnessTui
from agent_harness.tui_state import TuiState


def _event(
    sequence: int,
    event_type: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": HARNESS_EVENT_SCHEMA,
        "event_id": f"event_{sequence:032x}",
        "run_id": "run_agents_12345678",
        "turn_id": "turn_agents_12345678",
        "sequence": sequence,
        "timestamp": "2026-08-24T00:00:00Z",
        "type": event_type,
        "payload": dict(payload or {}),
    }


def _tool_payload(**values: Any) -> dict[str, Any]:
    return {
        "call_id": "call_agents_12345678",
        "tool_name": "agent.delegate",
        **values,
    }


def _apply_started_batch(state: TuiState, *, count: int = 2) -> None:
    state.begin_turn()
    state.apply_event(_event(1, "run.started"))
    state.apply_event(_event(2, "tool.started", _tool_payload()))
    state.apply_event(
        _event(
            3,
            "tool.progress",
            _tool_payload(
                progress_kind="subagent.batch_started",
                progress={"count": count, "depth": 1},
            ),
        )
    )
    for ordinal in range(count):
        state.apply_event(
            _event(
                4 + ordinal,
                "tool.progress",
                _tool_payload(
                    progress_kind="subagent.child_started",
                    progress={
                        "agent_id": f"agent-{ordinal:024x}",
                        "depth": 1,
                        "ordinal": ordinal,
                    },
                ),
            )
        )


def test_tui_state_reduces_content_free_subagent_lifecycle_and_artifacts() -> None:
    state = TuiState(session_id="session_12345678")
    _apply_started_batch(state)
    assert state.status == "agents"
    assert state.running_agent_count == 2
    assert state.settled_agent_count == 0

    state.apply_event(
        _event(
            6,
            "tool.progress",
            _tool_payload(
                progress_kind="subagent.child_finished",
                progress={
                    "agent_id": "agent-000000000000000000000000",
                    "depth": 1,
                    "ordinal": 0,
                    "status": "completed",
                },
            ),
        )
    )
    state.apply_event(
        _event(
            7,
            "tool.progress",
            _tool_payload(
                progress_kind="subagent.child_finished",
                progress={
                    "agent_id": "agent-000000000000000000000001",
                    "depth": 1,
                    "ordinal": 1,
                    "status": "failed",
                },
            ),
        )
    )
    state.apply_event(
        _event(
            8,
            "tool.progress",
            _tool_payload(
                progress_kind="subagent.batch_finished",
                progress={"count": 2, "depth": 1},
            ),
        )
    )
    secret_prompt = "SECRET child prompt"
    secret_summary = "SECRET child output"
    secret_path = "/private/worktree/SECRET"
    secret_ref = "refs/heads/SECRET"
    state.apply_event(
        _event(
            9,
            "tool.completed",
            _tool_payload(
                result={
                    "schema": "agent_harness.subagent_batch.v1",
                    "foreground": True,
                    "results": [
                        {
                            "agent_id": "agent-000000000000000000000000",
                            "depth": 1,
                            "ordinal": 0,
                            "status": "completed",
                            "changed": False,
                            "summary": secret_summary,
                            "prompt": secret_prompt,
                            "worktree_path": secret_path,
                            "branch_ref": secret_ref,
                        },
                        {
                            "agent_id": "agent-000000000000000000000001",
                            "depth": 1,
                            "ordinal": 1,
                            "status": "failed",
                            "changed": True,
                            "artifact_id": "wt_0123456789abcdef0123456789abcdef",
                            "summary": secret_summary,
                            "worktree_path": secret_path,
                            "branch_ref": secret_ref,
                        },
                    ],
                }
            ),
        )
    )

    assert state.running_agent_count == 0
    assert state.settled_agent_count == 2
    first, second = state.ordered_agents()
    assert (first.status, first.changed, first.artifact_id) == (
        "completed",
        False,
        None,
    )
    assert (second.status, second.changed, second.artifact_id) == (
        "failed",
        True,
        "wt_0123456789abcdef0123456789abcdef",
    )
    serialized = json.dumps(
        [asdict(item) for item in state.ordered_agents()],
        sort_keys=True,
    )
    assert secret_prompt not in serialized
    assert secret_summary not in serialized
    assert secret_path not in serialized
    assert secret_ref not in serialized


def test_tui_state_rejects_content_bearing_progress_and_path_artifact() -> None:
    state = TuiState(session_id="session_12345678")
    _apply_started_batch(state, count=1)

    with pytest.raises(ValueError, match="fields"):
        state.apply_event(
            _event(
                5,
                "tool.progress",
                _tool_payload(
                    progress_kind="subagent.child_finished",
                    progress={
                        "agent_id": "agent-000000000000000000000000",
                        "depth": 1,
                        "ordinal": 0,
                        "status": "completed",
                        "prompt": "must not be retained",
                    },
                ),
            )
        )
    assert "must not be retained" not in json.dumps(
        {key: asdict(value) for key, value in state.tools.items()}
    )

    unknown = TuiState(session_id="session_12345678")
    _apply_started_batch(unknown, count=1)
    with pytest.raises(ValueError, match="unknown subagent progress"):
        unknown.apply_event(
            _event(
                5,
                "tool.progress",
                _tool_payload(
                    progress_kind="subagent.SECRET_PROMPT",
                    progress={"content": "SECRET_PROMPT"},
                ),
            )
        )
    assert "SECRET_PROMPT" not in json.dumps(
        {key: asdict(value) for key, value in unknown.tools.items()}
    )

    clean = TuiState(session_id="session_12345678")
    _apply_started_batch(clean, count=1)
    clean.apply_event(
        _event(
            5,
            "tool.progress",
            _tool_payload(
                progress_kind="subagent.child_finished",
                progress={
                    "agent_id": "agent-000000000000000000000000",
                    "depth": 1,
                    "ordinal": 0,
                    "status": "failed",
                },
            ),
        )
    )
    clean.apply_event(
        _event(
            6,
            "tool.progress",
            _tool_payload(
                progress_kind="subagent.batch_finished",
                progress={"count": 1, "depth": 1},
            ),
        )
    )
    with pytest.raises(ValueError, match="artifact id"):
        clean.apply_event(
            _event(
                7,
                "tool.completed",
                _tool_payload(
                    result={
                        "schema": "agent_harness.subagent_batch.v1",
                        "results": [
                            {
                                "agent_id": "agent-000000000000000000000000",
                                "depth": 1,
                                "ordinal": 0,
                                "status": "failed",
                                "changed": True,
                                "artifact_id": "/private/worktree/path",
                            }
                        ],
                    }
                ),
            )
        )


def test_terminal_event_settles_any_visible_running_agents() -> None:
    state = TuiState(session_id="session_12345678")
    _apply_started_batch(state, count=1)

    state.apply_event(_event(5, "run.cancelled"))

    assert state.running_agent_count == 0
    assert state.settled_agent_count == 1
    assert state.ordered_agents()[0].status == "cancelled"
    assert state.agent_batches["call_agents_12345678"].status == "cancelled"


class _Screen:
    def __init__(self, rows: int = 18, columns: int = 100) -> None:
        self.rows = rows
        self.columns = columns
        self.output: list[tuple[int, str]] = []

    def getmaxyx(self) -> tuple[int, int]:
        return self.rows, self.columns

    def erase(self) -> None:
        self.output.clear()

    def addstr(self, row: int, _column: int, value: str, *_args: Any) -> None:
        self.output.append((row, value))

    def refresh(self) -> None:
        return None

    def move(self, _row: int, _column: int) -> None:
        return None


def _application(artifact_loader: Any) -> HarnessTui:
    runner = SimpleNamespace(
        client=SimpleNamespace(config=SimpleNamespace(model="test-model")),
        permission_mode="workspace-write",
        workspace=Path("/tmp/workspace"),
        subagent_artifacts=artifact_loader,
    )
    return HarnessTui(
        runner,  # type: ignore[arg-type]
        {"session_id": "session_12345678", "messages": []},
    )


def test_tui_header_status_and_body_show_foreground_agent_activity() -> None:
    application = _application(lambda **_kwargs: {"artifacts": []})
    _apply_started_batch(application.state, count=1)
    screen = _Screen()

    application.render(screen)

    rendered = "\n".join(value for _row, value in screen.output)
    header = next(value for row, value in screen.output if row == 0)
    status = next(value for row, value in screen.output if row == screen.rows - 4)
    assert "fg-agents 1/1" in header
    assert "fg-agents 1 running/0 settled" in status
    assert "[running] agent-000000000000000000000000" in rendered


def test_agents_command_lists_only_content_free_activity_and_inventory() -> None:
    secret_path = "/private/worktree/never-display"
    secret_ref = "refs/heads/never-display"
    calls: list[bool] = []

    def artifacts(*, reveal_paths: bool) -> dict[str, Any]:
        calls.append(reveal_paths)
        return {
            "schema": "agent_harness.subagent_artifacts.v1",
            "artifacts": [
                {
                    "artifact_id": "wt_0123456789abcdef0123456789abcdef",
                    "phase": "active",
                    "worktree_path": secret_path,
                    "branch_ref": secret_ref,
                }
            ],
        }

    application = _application(artifacts)
    _apply_started_batch(application.state, count=1)
    application.command("/agents")
    application.command("/help")

    notices = "\n".join(application.state.notices)
    assert calls == [False]
    assert "agent-000000000000000000000000" in notices
    assert "artifact=wt_0123456789abcdef0123456789abcdef" in notices
    assert "phase=active" in notices
    assert "/agents" in notices
    assert secret_path not in notices
    assert secret_ref not in notices

    application.command("/agents path wt_0123456789abcdef0123456789abcdef")
    assert calls == [False]
    assert "用法：/agents" in application.state.notices[-1]


def test_agents_inventory_error_never_exposes_exception_path() -> None:
    def broken_inventory(*, reveal_paths: bool) -> Any:
        assert reveal_paths is False
        raise RuntimeError("failed at /private/secret/worktree")

    application = _application(broken_inventory)

    application.command("/agents")

    notices = "\n".join(application.state.notices)
    assert "Agent artifact 清单不可用" in notices
    assert "/private/secret/worktree" not in notices
