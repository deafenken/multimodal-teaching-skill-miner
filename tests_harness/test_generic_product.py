from __future__ import annotations

import curses
import json
import os
from pathlib import Path
import stat
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import pytest

from agent_harness.cli import EXIT_USAGE, main as cli_main
from agent_harness.core import (
    CancellationToken,
    HarnessContractError,
    HarnessModelRequest,
    HarnessModelResponse,
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
    ProviderCapabilities,
    ToolCall,
    ToolExecutionContext,
    ToolExecutionError,
    ToolRegistry,
    ToolSpec,
)
from agent_harness.providers.deepseek import DeepSeekCodingModel
from agent_harness.runner import AgentRunner
from agent_harness.session import SESSION_SCHEMA, SessionStore, SessionStoreError
from agent_harness.toolsets import build_workspace_registry, permission_profile
from agent_harness.toolsets.workspace import WorkspaceToolset
from agent_harness.tui import HarnessTui
from agent_harness.tui_state import TuiState, display_width, sanitize_terminal_text, wrap_display


class _FakeDeepSeek:
    def __init__(
        self,
        planner: Mapping[str, Any],
        chunks: list[Mapping[str, Any]] | None = None,
        *,
        planner_usage: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = SimpleNamespace(model="deepseek-test", max_tokens=2048)
        self.planner = dict(planner)
        self.chunks = list(chunks or [])
        self.planner_usage = dict(
            planner_usage
            if planner_usage is not None
            else {"prompt_tokens": 10, "prompt_cache_hit_tokens": 8}
        )
        self.json_messages: list[Mapping[str, str]] = []
        self.answer_messages: list[Mapping[str, str]] = []

    def chat_json_stream(self, messages: list[Mapping[str, str]], **_kwargs: Any):
        self.json_messages = list(messages)
        return self.planner, {
            "response_id": "planner-1",
            "usage": dict(self.planner_usage),
        }

    def chat_text_stream(self, messages: list[Mapping[str, str]], **_kwargs: Any) -> Iterator[Mapping[str, Any]]:
        self.answer_messages = list(messages)
        yield from self.chunks


def _request(*, observations: tuple[Mapping[str, Any], ...] = ()) -> HarnessModelRequest:
    return HarnessModelRequest(
        run_id="run_12345678",
        turn_id="turn_12345678",
        step=1,
        context={
            "workspace": "/tmp/work",
            "messages": [{"role": "user", "content": "inspect the repository"}],
        },
        observations=observations,
        tools=(
            {
                "name": "workspace.read",
                "version": "1",
                "description": "read",
                "input_schema": {"type": "object"},
            },
        ),
        state={"remaining_steps": 3},
    )


def test_tool_aware_provider_returns_central_calls_without_leaking_planner() -> None:
    client = _FakeDeepSeek(
        {
            "action": "tool_calls",
            "tool_calls": [
                {
                    "call_id": "read-1",
                    "name": "workspace.read",
                    "arguments": {"path": "README.md"},
                }
            ],
        }
    )
    model = DeepSeekCodingModel(client)  # type: ignore[arg-type]
    events = list(
        model.plan_stream(
            _request(),
            cancellation_token=CancellationToken(),
            deadline_monotonic=time.monotonic() + 10,
        )
    )

    response = events[-1]
    assert isinstance(response, HarnessModelResponse)
    assert response.kind == "tool_calls"
    assert response.tool_calls[0].name == "workspace.read"
    assert all(getattr(item, "type", None) != "message.delta" for item in events[:-1])
    assert "student" not in json.dumps(client.json_messages)


def test_provider_streams_only_final_answer_and_redacts_reasoning() -> None:
    client = _FakeDeepSeek(
        {"action": "answer"},
        [
            {"type": "reasoning_delta", "text": "private chain of thought"},
            {"type": "text_delta", "text": "Direct "},
            {"type": "text_delta", "text": "answer"},
            {"type": "usage", "usage": {"completion_tokens": 2, "total_tokens": 5}},
            {"type": "completed", "trace": {"response_id": "answer-1"}},
        ],
    )
    model = DeepSeekCodingModel(client)  # type: ignore[arg-type]
    events = list(
        model.plan_stream(
            _request(),
            cancellation_token=CancellationToken(),
            deadline_monotonic=time.monotonic() + 10,
        )
    )
    deltas = [item for item in events if getattr(item, "type", None) == "message.delta"]
    reasoning = [item for item in events if getattr(item, "type", None) == "reasoning.delta"]
    response = events[-1]

    assert "".join(item.delta for item in deltas) == "Direct answer"
    assert reasoning and all(item.delta == "" for item in reasoning)
    assert isinstance(response, HarnessModelResponse)
    assert response.output == {"message": "Direct answer"}
    assert response.usage["prompt_cache_hit_tokens"] == 8


def test_provider_rejects_unauthorized_planner_tool() -> None:
    client = _FakeDeepSeek(
        {
            "action": "tool_calls",
            "tool_calls": [{"call_id": "bad-1", "name": "process.exec", "arguments": {}}],
        }
    )
    model = DeepSeekCodingModel(client)  # type: ignore[arg-type]
    with pytest.raises(HarnessContractError, match="unauthorized"):
        list(
            model.plan_stream(
                _request(),
                cancellation_token=CancellationToken(),
                deadline_monotonic=time.monotonic() + 10,
            )
        )


def test_provider_normalizes_usage_before_streaming_and_aggregation() -> None:
    client = _FakeDeepSeek(
        {"action": "answer"},
        [
            {"type": "text_delta", "text": "done"},
            {
                "type": "usage",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                    "total_tokens": 24,
                    "prompt_cache_hit_tokens": 5,
                    "prompt_cache_miss_tokens": 15,
                    "provider_private_counter": 888,
                    "input_tokens": -1,
                },
            },
            {"type": "completed", "trace": {"response_id": "answer-1"}},
        ],
        planner_usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_cache_hit_tokens": 7,
            "prompt_cache_miss_tokens": 3,
            "provider_private_counter": 999,
            "output_tokens": True,
        },
    )
    model = DeepSeekCodingModel(client)  # type: ignore[arg-type]

    events = list(
        model.plan_stream(
            _request(),
            cancellation_token=CancellationToken(),
            deadline_monotonic=time.monotonic() + 10,
        )
    )
    updates = [
        dict(item.payload)
        for item in events
        if getattr(item, "type", None) == "usage.update"
    ]
    response = events[-1]

    assert updates == [
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_cache_hit_tokens": 7,
            "prompt_cache_miss_tokens": 3,
        },
        {
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "total_tokens": 24,
            "prompt_cache_hit_tokens": 5,
            "prompt_cache_miss_tokens": 15,
        },
    ]
    assert isinstance(response, HarnessModelResponse)
    assert response.usage == {
        "prompt_tokens": 30,
        "completion_tokens": 6,
        "total_tokens": 36,
        "prompt_cache_hit_tokens": 12,
        "prompt_cache_miss_tokens": 18,
    }


def test_provider_rejects_total_context_budget_before_transport() -> None:
    client = _FakeDeepSeek({"action": "answer"})
    model = DeepSeekCodingModel(
        client,  # type: ignore[arg-type]
        planner_max_tokens=256,
        answer_max_tokens=256,
        context_window_tokens=1_024,
    )
    request = HarnessModelRequest(
        run_id="run_12345678",
        turn_id="turn_12345678",
        step=1,
        context={
            "workspace": "/tmp/work",
            "messages": [{"role": "user", "content": "x" * 4_000}],
        },
        observations=(),
        tools=(),
        state={"remaining_steps": 3},
    )

    with pytest.raises(HarnessContractError, match="context|budget|window"):
        list(
            model.plan_stream(
                request,
                cancellation_token=CancellationToken(),
                deadline_monotonic=time.monotonic() + 10,
            )
        )

    assert client.json_messages == []
    assert client.answer_messages == []


def test_session_store_is_private_workspace_scoped_and_forkable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_home = tmp_path / "state"
    store = SessionStore(workspace, state_home=state_home)
    session = store.create(
        provider="deepseek",
        model="deepseek-test",
        permission_mode="read-only",
    )
    session = store.append_message(session["session_id"], role="user", content="hello")
    session = store.append_message(session["session_id"], role="assistant", content="world")
    forked = store.fork(session["session_id"])

    assert session["schema"] == SESSION_SCHEMA
    assert [item["content"] for item in forked["messages"]] == ["hello", "world"]
    assert forked["forked_from"] == session["session_id"]
    assert stat.S_IMODE(state_home.stat().st_mode) == 0o700
    session_path = store.sessions_directory / f"{session['session_id']}.json"
    assert stat.S_IMODE(session_path.stat().st_mode) == 0o600
    assert store.archive(session["session_id"])["archived"] is True
    assert all(item["session_id"] != session["session_id"] for item in store.list())


def test_session_store_rejects_symlink_home(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(SessionStoreError, match="real directory"):
        SessionStore(workspace, state_home=link)


def test_session_store_does_not_chmod_an_existing_shared_directory(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)

    with pytest.raises(SessionStoreError, match="permissions are too broad"):
        SessionStore(workspace, state_home=shared)

    assert stat.S_IMODE(shared.stat().st_mode) == 0o755


def test_session_store_reserves_capacity_and_serializes_turns(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    first = SessionStore(workspace, state_home=state)
    second = SessionStore(workspace, state_home=state)
    session = first.create(
        provider="deepseek",
        model="deepseek-test",
        permission_mode="read-only",
    )
    material = first.load(session["session_id"])
    material["messages"] = [
        {"role": "user", "content": f"message-{index}", "timestamp": "now"}
        for index in range(1_999)
    ]
    first.save(material)

    with pytest.raises(SessionStoreError, match="message limit"):
        first.append_message(
            session["session_id"],
            role="user",
            content="one more",
            reserve_messages=1,
        )

    with first.turn_lock(session["session_id"]):
        with pytest.raises(SessionStoreError, match="active run"):
            with second.turn_lock(session["session_id"]):
                pass


def test_cli_usage_exit_and_session_json_exclude_transcripts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["--definitely-not-an-option"]) == EXIT_USAGE
    assert "unrecognized arguments" in capsys.readouterr().err

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    store = SessionStore(workspace, state_home=state)
    session = store.create(
        provider="deepseek",
        model="deepseek-test",
        permission_mode="read-only",
    )
    store.append_message(
        session["session_id"],
        role="user",
        content="PRIVATE_TRANSCRIPT_MARKER",
    )

    assert (
        cli_main(
            [
                "--cwd",
                str(workspace),
                "--state-home",
                str(state),
                "sessions",
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "PRIVATE_TRANSCRIPT_MARKER" not in output
    assert '"messages"' not in output


def _context(name: str = "tool") -> ToolExecutionContext:
    return ToolExecutionContext(
        run_id="run_12345678",
        turn_id="turn_12345678",
        call_id="call_12345678",
        tool_name=name,
        cancellation_token=CancellationToken(),
        deadline_monotonic=time.monotonic() + 30,
        idempotency_key=None,
        emit_progress=lambda _kind, _payload=None: None,
    )


def test_workspace_read_search_and_escape_boundary(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("alpha\nbeta alpha\n", encoding="utf-8")
    outside = tmp_path.parent / "outside-secret"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "linked").symlink_to(outside)
    tools = WorkspaceToolset(tmp_path)

    read = tools.read_file({"path": "src/main.py", "start_line": 2, "end_line": 2}, _context())
    searched = tools.search({"query": "alpha", "path": "src"}, _context())
    listed = tools.list_files({"path": ".", "pattern": "*.py"}, _context())

    assert read["content"] == "2: beta alpha"
    assert len(searched["matches"]) == 2
    assert listed["files"] == ["src/main.py"]
    with pytest.raises(Exception, match="workspace|escape"):
        tools.read_file({"path": "../outside-secret"}, _context())
    with pytest.raises(Exception, match="symbolic"):
        tools.read_file({"path": "linked"}, _context())


def test_permission_profiles_change_actual_model_tool_surface(tmp_path: Path) -> None:
    registry = build_workspace_registry(tmp_path)
    read = registry.definitions(
        permission_profile("read-only"),
        trusted_data_scopes={"internal", "workspace_read"},
    )
    write = registry.definitions(
        permission_profile("workspace-write"),
        trusted_data_scopes={"internal", "workspace_read", "workspace_write"},
    )
    full = registry.definitions(
        permission_profile("full-access"),
        trusted_data_scopes={
            "internal",
            "workspace_read",
            "workspace_write",
            "host_access",
        },
    )
    assert "workspace.patch" not in {item["name"] for item in read}
    assert "workspace.patch" in {item["name"] for item in write}
    assert "process.exec" not in {item["name"] for item in write}
    assert "process.exec" in {item["name"] for item in full}
    assert "git.status" not in {item["name"] for item in full}
    assert "git.diff" not in {item["name"] for item in full}


def test_workspace_patch_and_command_secret_scrubbing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "demo.txt").write_text("old\n", encoding="utf-8")
    tools = WorkspaceToolset(tmp_path)
    patch = """diff --git a/demo.txt b/demo.txt
--- a/demo.txt
+++ b/demo.txt
@@ -1 +1 @@
-old
+new
"""
    result = tools.apply_patch({"patch": patch}, _context("workspace.patch"))
    assert result == {"applied": True, "paths": ["demo.txt"]}
    assert (tmp_path / "demo.txt").read_text(encoding="utf-8") == "new\n"

    monkeypatch.setenv("SHOULD_NOT_LEAK_API_KEY", "secret-value")
    command = tools.run_command(
        {"command": "printenv SHOULD_NOT_LEAK_API_KEY || true", "timeout_seconds": 5},
        _context("process.exec"),
    )
    assert "secret-value" not in command["output"]

    large = tools.run_command(
        {
            "command": "python3 -c 'import sys; sys.stdout.write(\"x\" * 120000)'",
            "timeout_seconds": 10,
        },
        _context("process.exec"),
    )
    assert large["exit_code"] == 0
    assert len(large["output"]) == 8_000
    assert large["truncated"] is True

    binary = tools.run_command(
        {
            "command": "python3 -c 'import sys; sys.stdout.buffer.write(bytes([0,255]))'",
            "timeout_seconds": 5,
        },
        _context("process.exec"),
    )
    assert binary["exit_code"] == 0
    assert "�" in binary["output"]


def test_workspace_command_reaps_background_stdout_holder(tmp_path: Path) -> None:
    tools = WorkspaceToolset(tmp_path)
    started = time.monotonic()

    result = tools.run_command(
        {"command": "(sleep 10) &", "timeout_seconds": 5},
        _context("process.exec"),
    )

    assert result["exit_code"] == 125
    assert result["background_processes_reaped"] is True
    assert time.monotonic() - started < 2

    stubborn = tools.run_command(
        {
            "command": (
                "(trap '' TERM; exec sleep 300) </dev/null >/dev/null 2>&1 & "
                "echo $!"
            ),
            "timeout_seconds": 5,
        },
        _context("process.exec"),
    )
    child_pid = int(stubborn["output"].strip())
    assert stubborn["exit_code"] == 125
    assert stubborn["background_processes_reaped"] is True
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_workspace_patch_rejects_symlink_and_protected_rename(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "demo.txt").write_text("old\n", encoding="utf-8")
    tools = WorkspaceToolset(tmp_path)
    symlink_patch = """diff --git a/link b/link
new file mode 120000
index 0000000..7f0f59f
--- /dev/null
+++ b/link
@@ -0,0 +1 @@
+../outside
"""
    with pytest.raises(Exception, match="symbolic"):
        tools.apply_patch(
            {"patch": symlink_patch},
            _context("workspace.patch"),
        )

    protected_rename = """diff --git a/demo.txt b/.private/secret
similarity index 100%
rename from demo.txt
rename to .private/secret
"""
    with pytest.raises(Exception, match="protected"):
        tools.apply_patch(
            {"patch": protected_rename},
            _context("workspace.patch"),
        )


def test_workspace_patch_does_not_execute_repository_git_filters(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "demo.txt").write_text("old\n", encoding="utf-8")
    (tmp_path / ".gitattributes").write_text("demo.txt filter=hostile\n", encoding="utf-8")
    marker = tmp_path / "FILTER_EXECUTED"
    hostile_filter = tmp_path / "hostile-filter.sh"
    hostile_filter.write_text(
        "#!/bin/sh\ntouch FILTER_EXECUTED\ncat\n",
        encoding="utf-8",
    )
    hostile_filter.chmod(0o755)
    subprocess.run(
        ["git", "config", "filter.hostile.clean", str(hostile_filter)],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "filter.hostile.smudge", str(hostile_filter)],
        cwd=tmp_path,
        check=True,
    )
    tools = WorkspaceToolset(tmp_path)
    patch = """diff --git a/demo.txt b/demo.txt
--- a/demo.txt
+++ b/demo.txt
@@ -1 +1 @@
-old
+new
"""

    result = tools.apply_patch({"patch": patch}, _context("workspace.patch"))

    assert result == {"applied": True, "paths": ["demo.txt"]}
    assert (tmp_path / "demo.txt").read_text(encoding="utf-8") == "new\n"
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="process-group assertions require POSIX")
def test_workspace_command_timeout_reaps_stubborn_background_group(
    tmp_path: Path,
) -> None:
    tools = WorkspaceToolset(tmp_path)
    started = time.monotonic()

    with pytest.raises(ToolExecutionError) as captured:
        tools.run_command(
            {
                "command": (
                    "(trap '' TERM; exec sleep 300) & child=$!; "
                    "printf '%s' \"$child\" > child.pid; wait"
                ),
                "timeout_seconds": 0.15,
            },
            _context("process.exec"),
        )

    assert captured.value.code == "command_timeout"
    assert time.monotonic() - started < 2
    child_pid = int((tmp_path / "child.pid").read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_workspace_protected_paths_are_case_insensitive(
    tmp_path: Path,
) -> None:
    private = tmp_path / ".private"
    private.mkdir()
    (private / "secret.txt").write_text("secret", encoding="utf-8")
    tools = WorkspaceToolset(tmp_path)

    listed = tools.list_files(
        {"path": ".", "pattern": "*", "max_results": 100},
        _context(),
    )

    assert not any(path.casefold().startswith(".private/") for path in listed["files"])
    with pytest.raises(Exception, match="protected"):
        tools.read_file(
            {"path": ".PRIVATE/secret.txt"},
            _context(),
        )


def _event(sequence: int, event_type: str, payload: Mapping[str, Any] | None = None):
    return {
        "schema": "agent_harness.event.v1",
        "run_id": "run_12345678",
        "turn_id": "turn_12345678",
        "sequence": sequence,
        "type": event_type,
        "payload": dict(payload or {}),
    }


def test_tui_reducer_hides_reasoning_and_rejects_bad_streams() -> None:
    state = TuiState(session_id="session_12345678")
    state.begin_turn()
    state.apply_event(_event(1, "run.started"))
    state.apply_event(_event(2, "reasoning.delta", {"delta": "secret", "chars": 6}))
    state.apply_event(_event(3, "message.delta", {"delta": "visible", "channel": "assistant"}))
    state.apply_event(_event(4, "run.completed"))
    assert state.assistant_draft == "visible"
    assert state.reasoning_chars == 6
    with pytest.raises(ValueError, match="after terminal"):
        state.apply_event(_event(5, "message.delta", {"delta": "late"}))

    fresh = TuiState(session_id="session_12345678")
    fresh.begin_turn()
    invalid_schema = _event(1, "run.started")
    invalid_schema["schema"] = "agent_harness.event.v0"
    with pytest.raises(ValueError, match="schema"):
        fresh.apply_event(invalid_schema)
    assert fresh.last_sequence == 0
    with pytest.raises(ValueError, match="unknown"):
        fresh.apply_event(_event(1, "future.event"))
    assert fresh.last_sequence == 0


def test_tui_usage_updates_are_accumulated_across_provider_calls() -> None:
    state = TuiState(session_id="session_12345678")
    state.begin_turn()
    state.apply_event(_event(1, "run.started"))
    state.apply_event(
        _event(
            2,
            "usage.update",
            {
                "prompt_tokens": 10,
                "total_tokens": 12,
                "prompt_cache_hit_tokens": 7,
                "ignored_boolean": True,
            },
        )
    )
    state.apply_event(
        _event(
            3,
            "usage.update",
            {
                "prompt_tokens": 20,
                "total_tokens": 24,
                "prompt_cache_hit_tokens": 5,
                "prompt_cache_miss_tokens": 15,
                "ignored_negative": -1,
            },
        )
    )

    assert state.usage == {
        "prompt_tokens": 30,
        "total_tokens": 36,
        "prompt_cache_hit_tokens": 12,
        "prompt_cache_miss_tokens": 15,
    }


def test_tui_unicode_helpers_are_control_safe() -> None:
    assert display_width("中A") == 3
    assert sanitize_terminal_text("ok\x1b[31m") == "ok�[31m"
    assert sanitize_terminal_text("a\x85b\u202ec") == "a�b�c"
    assert all(display_width(line) <= 4 for line in wrap_display("中文AB", 4))


def test_tui_tolerates_terminal_without_cursor_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = HarnessTui(
        SimpleNamespace(),  # type: ignore[arg-type]
        {"session_id": "session_12345678", "messages": []},
    )
    application.running = False
    screen = SimpleNamespace(
        keypad=lambda _enabled: None,
        nodelay=lambda _enabled: None,
    )

    def unsupported_cursor(_visibility: int) -> None:
        raise curses.error("cursor mode unsupported")

    monkeypatch.setattr(curses, "curs_set", unsupported_cursor)
    assert application.run(screen) == 0


def test_tui_settles_outcome_identity_before_accepting_another_turn() -> None:
    application = HarnessTui(
        SimpleNamespace(),  # type: ignore[arg-type]
        {"session_id": "session_12345678", "messages": []},
    )
    application.state.begin_turn()
    application.state.active_run_id = "run_12345678"
    application.state.active_turn_id = "turn_12345678"
    application.worker = threading.Thread(target=lambda: None)
    outcome = SimpleNamespace(
        session_id="session_12345678",
        run_id="run_12345678",
        turn_id="turn_12345678",
        status="completed",
        message="done",
        reason="final_output",
    )
    application.queue.put(("outcome", outcome))

    assert application.busy is True
    application.drain()
    assert application.busy is False
    assert application.transcript[-1]["content"] == "done"

    application.state.begin_turn()
    application.state.active_run_id = "run_12345678"
    application.state.active_turn_id = "turn_12345678"
    wrong = SimpleNamespace(**{**outcome.__dict__, "run_id": "run_87654321"})
    with pytest.raises(ValueError, match="another run"):
        application._complete_turn(wrong)  # noqa: SLF001


class _FinalModel:
    @property
    def model_spec(self) -> ProviderModelSpec:
        return ProviderModelSpec(
            provider="test",
            model="test-v1",
            capabilities=ProviderCapabilities(provider="test", model="test-v1"),
            context_window_tokens=8_192,
            maximum_output_tokens=1_024,
        )

    def plan(self, _request: Any, **_kwargs: Any) -> HarnessModelResponse:
        return HarnessModelResponse(kind="final", output={"message": "done"})

    def classify_error(self, _error: BaseException) -> ProviderFailure:
        return ProviderFailure(
            kind=ProviderErrorKind.UNKNOWN,
            retryable=False,
            safe_code="test_error",
        )


def test_runner_persists_only_generic_session_fields(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key = tmp_path / "key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    tmp_path.chmod(0o700)
    runner = AgentRunner(
        workspace,
        state_home=tmp_path / "state",
        api_key_file=key,
    )
    runner.model = _FinalModel()  # type: ignore[assignment]
    session = runner.new_session()
    outcome = runner.run_turn(session["session_id"], "do work")
    stored = runner.store.load(session["session_id"])

    assert outcome.status == "completed"
    assert [item["content"] for item in stored["messages"]] == ["do work", "done"]
    assert outcome.result["schema"] == "agent_harness.run.v1"
    assert set(stored).issuperset(
        {"schema", "session_id", "messages", "runs", "usage", "permission_mode"}
    )


def test_runner_remote_content_can_be_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key = tmp_path / "key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    tmp_path.chmod(0o700)
    monkeypatch.setenv("HARNESS_ALLOW_REMOTE_CONTENT", "0")

    runner = AgentRunner(
        workspace,
        state_home=tmp_path / "state",
        api_key_file=key,
    )

    assert runner.provider_status["remote_content_opt_in"] is False


def test_runner_resume_downgrades_permissions_and_rejects_unsafe_history(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key = tmp_path / "key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    tmp_path.chmod(0o700)
    runner = AgentRunner(
        workspace,
        state_home=tmp_path / "state",
        api_key_file=key,
        permission_mode="full-access",
    )
    session = runner.new_session()

    resumed = runner.resume_session(session["session_id"])
    assert resumed["permission_mode"] == "read-only"
    assert runner.permission_mode == "read-only"

    runner.store.archive(session["session_id"])
    with pytest.raises(SessionStoreError, match="archived"):
        runner.resume_session(session["session_id"])

    interrupted = runner.new_session()
    runner.store.record_run(
        interrupted["session_id"],
        run_id="run_12345678",
        turn_id="turn_12345678",
        status="running",
    )
    with pytest.raises(SessionStoreError, match="unfinished or uncertain run"):
        runner.resume_session(interrupted["session_id"])


def test_runner_never_uses_stored_permissions_as_authority(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key = tmp_path / "key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    tmp_path.chmod(0o700)
    state = tmp_path / "state"
    privileged = AgentRunner(
        workspace,
        state_home=state,
        api_key_file=key,
        permission_mode="full-access",
    )
    session = privileged.new_session()

    class CaptureModel(_FinalModel):
        def __init__(self) -> None:
            self.requests: list[Any] = []

        def plan(self, request: Any, **_kwargs: Any) -> HarnessModelResponse:
            self.requests.append(request)
            return super().plan(request, **_kwargs)

    restricted = AgentRunner(
        workspace,
        state_home=state,
        api_key_file=key,
        permission_mode="read-only",
    )
    model = CaptureModel()
    restricted.model = model  # type: ignore[assignment]

    outcome = restricted.run_turn(session["session_id"], "inspect only")

    assert outcome.status == "completed"
    assert model.requests
    assert "process.exec" not in {item["name"] for item in model.requests[0].tools}
    assert restricted.store.load(session["session_id"])["permission_mode"] == "read-only"


def test_workspace_unresolved_run_fences_other_sessions_until_cli_reconcile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key = tmp_path / "key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    tmp_path.chmod(0o700)
    state = tmp_path / "state"
    runner = AgentRunner(
        workspace,
        state_home=state,
        api_key_file=key,
    )
    runner.model = _FinalModel()  # type: ignore[assignment]
    uncertain = runner.new_session(title="uncertain")
    other = runner.new_session(title="other")
    run_id = "run_unresolved_12345678"
    runner.store.record_run(
        uncertain["session_id"],
        run_id=run_id,
        turn_id="turn_unresolved_12345678",
        status="running",
        requires_reconciliation=True,
    )

    with pytest.raises(SessionStoreError, match="workspace has an unresolved run"):
        runner.run_turn(other["session_id"], "must remain fenced")

    common = ["--cwd", str(workspace), "--state-home", str(state)]
    assert cli_main([*common, "effects"]) == 0
    effects = json.loads(capsys.readouterr().out)
    assert effects == {
        "schema": "agent_harness.unresolved_runs.v1",
        "runs": [
            {
                "session_id": uncertain["session_id"],
                "run_id": run_id,
                "turn_id": "turn_unresolved_12345678",
                "status": "running",
            }
        ],
    }

    assert cli_main([*common, "reconcile", run_id]) == 0
    reconciliation = json.loads(capsys.readouterr().out)
    assert reconciliation == {
        "schema": "agent_harness.reconciliation.v1",
        "session_id": uncertain["session_id"],
        "run_id": run_id,
        "status": "acknowledged",
    }
    reconciled = runner.store.load(uncertain["session_id"])
    recorded = next(item for item in reconciled["runs"] if item["run_id"] == run_id)
    assert recorded["status"] == "handoff"
    assert recorded["requires_reconciliation"] is False
    assert any(run_id in notice for notice in reconciled["risk_notices"])

    outcome = runner.run_turn(other["session_id"], "now allowed")
    assert outcome.status == "completed"


def test_runner_persists_core_effect_handoff_as_workspace_fence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key = tmp_path / "key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    tmp_path.chmod(0o700)
    runner = AgentRunner(
        workspace,
        state_home=tmp_path / "state",
        api_key_file=key,
        permission_mode="workspace-write",
    )

    class UncertainModel(_FinalModel):
        def __init__(self) -> None:
            self.calls = 0

        def plan(self, _request: Any, **_kwargs: Any) -> HarnessModelResponse:
            self.calls += 1
            if self.calls == 1:
                return HarnessModelResponse(
                    kind="tool_calls",
                    tool_calls=(
                        ToolCall(
                            call_id="call_uncertain_12345678",
                            name="workspace.uncertain",
                            arguments={},
                        ),
                    ),
                )
            return HarnessModelResponse(
                kind="final",
                output={"message": "must not be reached"},
            )

    def uncertain_effect(
        _arguments: Mapping[str, Any], context: ToolExecutionContext
    ) -> Mapping[str, Any]:
        context.begin_effect()
        raise ToolExecutionError("effect receipt lost", code="receipt_lost")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="workspace.uncertain",
            version="1",
            description="Exercise the durable uncertain-effect handoff boundary.",
            input_schema={"type": "object", "additionalProperties": False},
            permission="workspace.write",
            replay_policy="never",
            data_scope="workspace_write",
            execution_isolation="trusted_inline",
            trusted_inline_reason="bounded deterministic uncertain-effect test fixture",
        ),
        uncertain_effect,
    )
    model = UncertainModel()
    runner.registry = registry
    runner.model = model  # type: ignore[assignment]
    session = runner.new_session()

    outcome = runner.run_turn(session["session_id"], "perform the effect")

    assert outcome.status == "handoff"
    assert model.calls == 1
    stored = runner.store.load(session["session_id"])
    run = stored["runs"][-1]
    assert run["status"] == "handoff"
    assert run["requires_reconciliation"] is True
    assert [message["role"] for message in stored["messages"]] == ["user"]
    assert runner.store.unresolved_workspace_runs()[0]["run_id"] == outcome.run_id
    with pytest.raises(SessionStoreError, match="workspace has an unresolved run"):
        runner.run_turn(runner.new_session()["session_id"], "must remain fenced")
