from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
from urllib.parse import quote

import pytest

import agent_harness.runner as runner_module
from agent_harness.core import (
    CancellationToken,
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
)
from agent_harness.runner import AgentRunner, TurnOutcome
from agent_harness.subagents import SubagentLineage, SubagentTask
from agent_harness.worktrees import (
    WorktreeCleanupResult,
    WorktreeError,
    WorktreeManager,
    WorktreeRecord,
)


GIT = Path(shutil.which("git") or "/usr/bin/git").resolve()


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        [os.fspath(GIT), "-C", os.fspath(repository), *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": os.fspath(repository.parent),
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        },
    )


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Harness Test")
    _git(repository, "config", "user.email", "harness@example.invalid")
    (repository / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-q", "-m", "baseline")
    return repository


def _key(tmp_path: Path) -> Path:
    tmp_path.chmod(0o700)
    key = tmp_path / "deepseek.key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    return key


def _lineage() -> SubagentLineage:
    return SubagentLineage(
        root_run_id="run-root-12345678",
        parent_session_id="session-parent-12345678",
        parent_run_id="run-parent-12345678",
        parent_turn_id="turn-parent-12345678",
        parent_call_id="call-parent-12345678",
        agent_id="agent-child-12345678",
        depth=1,
        ordinal=0,
        task_id="child-task",
    )


def test_subagent_artifacts_are_path_private_by_default(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    state_home = tmp_path / "state"
    worktree_home = tmp_path / "worktree-home"
    runner = AgentRunner(
        repository,
        state_home=state_home,
        worktree_home=worktree_home,
        api_key_file=_key(tmp_path),
    )
    manager = WorktreeManager(
        repository,
        state_directory=runner.store.root / "worktrees",
        worktree_home=worktree_home,
        git_executable=GIT,
    )
    record = manager.create()

    private = runner.subagent_artifacts()

    assert private == {
        "schema": "agent_harness.subagent_artifacts.v1",
        "artifacts": [
            {
                "artifact_id": record.worktree_id,
                "phase": "active",
                "created_at": record.created_at,
            }
        ],
    }
    rendered = repr(private)
    for secret in (
        record.repository,
        record.common_git_dir,
        record.worktree_path,
        record.branch_ref,
        record.git_dir or "",
    ):
        assert secret not in rendered

    revealed = runner.subagent_artifacts(reveal_paths=True)
    assert revealed["artifacts"][0] == {
        "artifact_id": record.worktree_id,
        "phase": "active",
        "created_at": record.created_at,
        "worktree_path": record.worktree_path,
    }
    assert "repository" not in repr(revealed)
    assert "git_dir" not in repr(revealed)
    with pytest.raises(ValueError, match="boolean"):
        runner.subagent_artifacts(reveal_paths=1)  # type: ignore[arg-type]

    assert manager.remove_if_pristine(record.worktree_id).removed is True


def test_delegated_runner_forces_no_extensions_nesting_or_saved_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = AgentRunner(
        workspace,
        state_home=tmp_path / "state",
        api_key_file=_key(tmp_path),
        permission_mode="workspace-write",
        subagents_enabled=True,
        project_extensions_enabled=True,
        agent_context={
            "kind": "subagent",
            "agent_id": "agent-child-12345678",
            "task_id": "child-task",
            "root_run_id": "run-root-12345678",
            "parent_run_id": "run-parent-12345678",
            "parent_turn_id": "turn-parent-12345678",
            "parent_call_id": "call-parent-12345678",
            "depth": 1,
            "result_contract": "concise result for parent",
        },
    )

    assert runner._subagent_scheduler is None
    assert runner._project_extensions_enabled is False
    assert runner.registry.get("agent.delegate") is None

    def saved_rules_must_not_load(_session_id: str) -> Any:
        raise AssertionError("delegated runs must not inherit saved approval rules")

    monkeypatch.setattr(runner.store, "approval_rules", saved_rules_must_not_load)

    def extensions_must_not_load() -> Any:
        raise AssertionError("delegated runs must not load project extensions")

    monkeypatch.setattr(runner, "hook_snapshot", extensions_must_not_load)
    monkeypatch.setattr(runner, "mcp_snapshot", extensions_must_not_load)

    class FinalModel:
        @property
        def model_spec(self) -> ProviderModelSpec:
            return ProviderModelSpec(
                provider="test",
                model="test-v1",
                capabilities=ProviderCapabilities(
                    provider="test",
                    model="test-v1",
                    structured_output=False,
                    native_stream=False,
                ),
                context_window_tokens=8_192,
                maximum_output_tokens=1_024,
            )

        def plan(self, _request: Any, **_kwargs: Any) -> HarnessModelResponse:
            return HarnessModelResponse(
                kind="final",
                output={"message": "child done"},
            )

        def classify_error(self, _error: BaseException) -> ProviderFailure:
            return ProviderFailure(
                kind=ProviderErrorKind.UNKNOWN,
                retryable=False,
                safe_code="test_error",
            )

    runner.model = FinalModel()  # type: ignore[assignment]
    session = runner.new_session()
    outcome = runner.run_turn(session["session_id"], "inspect")

    assert outcome.status == "completed"


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_permission_downgrade_after_allocation_stops_before_child_provider(
    tmp_path: Path,
    cleanup_fails: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "isolated"
    root.mkdir()
    runner = AgentRunner(
        workspace,
        state_home=tmp_path / "state",
        api_key_file=_key(tmp_path),
        permission_mode="workspace-write",
        worktree_home=tmp_path / "worktree-home",
    )
    record = WorktreeRecord(
        schema="agent_harness.worktree.v1",
        worktree_id="wt_" + "1" * 32,
        phase="active",
        repository=os.fspath(workspace),
        common_git_dir=os.fspath(tmp_path / "git-common"),
        worktree_path=os.fspath(root),
        branch_ref="refs/heads/agent-harness/worktrees/" + "2" * 32,
        baseline_commit="3" * 40,
        baseline_manifest_sha256="4" * 64,
        git_dir=os.fspath(tmp_path / "git-common" / "worktrees" / "child"),
        lock_reason="test",
        created_at="2026-08-24T00:00:00Z",
    )

    class Manager:
        def __init__(self) -> None:
            self.controls: tuple[Any, Any] | None = None

        def create(self, **kwargs: Any) -> WorktreeRecord:
            self.controls = (
                kwargs.get("cancellation_token"),
                kwargs.get("deadline_monotonic"),
            )
            runner.permission_mode = "read-only"
            return record

        def remove_if_pristine(self, _worktree_id: str, **_kwargs: Any) -> Any:
            if cleanup_fails:
                raise WorktreeError("synthetic cleanup failure")
            return WorktreeCleanupResult(True, "removed", record)

    manager = Manager()
    runner._worktree_manager_instance = manager  # type: ignore[assignment]
    token = CancellationToken()
    result = runner._execute_subagent(
        SubagentTask("child-task", "Change a file.", "workspace-write"),
        _lineage(),
        token,
        time.monotonic() + 5.0,
    )

    assert result.error_code == "subagent_permission_denied"
    assert result.changed is cleanup_fails
    assert result.requires_reconciliation is cleanup_fails
    assert result.artifact_id == (record.worktree_id if cleanup_fails else None)
    assert manager.controls is not None
    assert manager.controls[0] is token


def test_child_summary_redacts_local_path_and_uri_spellings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    child_root = tmp_path / "isolated child"
    child_root.mkdir()
    common = tmp_path / "private git"
    state_root = tmp_path / "private state"
    runner = AgentRunner(
        workspace,
        state_home=tmp_path / "state",
        api_key_file=_key(tmp_path),
        worktree_home=tmp_path / "worktree-home",
    )
    record = WorktreeRecord(
        schema="agent_harness.worktree.v1",
        worktree_id="wt_" + "a" * 32,
        phase="active",
        repository=os.fspath(workspace),
        common_git_dir=os.fspath(common),
        worktree_path=os.fspath(child_root),
        branch_ref="refs/heads/agent-harness/worktrees/" + "b" * 32,
        baseline_commit="c" * 40,
        baseline_manifest_sha256="d" * 64,
        git_dir=os.fspath(common / "worktrees" / "child"),
        lock_reason="test",
        created_at="2026-08-24T00:00:00Z",
    )

    class Manager:
        def create(self, **_kwargs: Any) -> WorktreeRecord:
            return record

        def remove_if_pristine(self, _worktree_id: str, **_kwargs: Any) -> Any:
            return WorktreeCleanupResult(False, "git_status_changed", record)

    class Store:
        root = state_root

        @staticmethod
        def load(_session_id: str) -> dict[str, Any]:
            return {
                "runs": [
                    {
                        "run_id": "run-child-12345678",
                        "requires_reconciliation": False,
                    }
                ]
            }

        @staticmethod
        def unresolved_workspace_runs() -> list[Any]:
            return []

    captured: dict[str, Any] = {}

    class Child:
        def __init__(self, child_workspace: Path, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.workspace = Path(child_workspace)
            self.store = Store()

        @staticmethod
        def new_session(**_kwargs: Any) -> dict[str, str]:
            return {"session_id": "session-child-12345678"}

        @staticmethod
        def run_turn(*_args: Any, **_kwargs: Any) -> TurnOutcome:
            message = " ".join(
                (
                    os.fspath(child_root),
                    child_root.as_uri(),
                    quote(os.fspath(state_root), safe="/:"),
                    os.fspath(workspace),
                )
            )
            return TurnOutcome(
                session_id="session-child-12345678",
                run_id="run-child-12345678",
                turn_id="turn-child-12345678",
                status="completed",
                message=message,
                reason="",
                usage={},
                result={},
            )

    runner._worktree_manager_instance = Manager()  # type: ignore[assignment]
    monkeypatch.setattr(runner_module, "AgentRunner", Child)

    result = runner._execute_subagent(
        SubagentTask("child-task", "Inspect paths.", "read-only"),
        _lineage(),
        CancellationToken(),
        time.monotonic() + 5.0,
    )

    assert result.status == "completed"
    assert result.changed is True
    assert captured["subagents_enabled"] is False
    assert captured["project_extensions_enabled"] is False
    for secret in (
        os.fspath(child_root),
        child_root.as_uri(),
        quote(os.fspath(state_root), safe="/:"),
        os.fspath(workspace),
    ):
        assert secret.casefold() not in result.summary.casefold()


def test_root_budget_is_finalized_when_runtime_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = AgentRunner(
        workspace,
        state_home=tmp_path / "state",
        api_key_file=_key(tmp_path),
    )
    assert runner._subagent_scheduler is not None
    captured: dict[str, str] = {}

    def fail_runtime(*_args: Any, **kwargs: Any) -> Any:
        run_id = str(kwargs["run_id"])
        captured["run_id"] = run_id
        reservation = runner._subagent_scheduler.budget_ledger.reserve(
            root_run_id=run_id,
            parent_agent_id=run_id,
            depth=1,
            count=1,
        )
        reservation.release()
        raise RuntimeError("synthetic runtime failure")

    monkeypatch.setattr(runner_module, "run_agent_harness", fail_runtime)
    session = runner.new_session()

    with pytest.raises(RuntimeError, match="synthetic runtime failure"):
        runner.run_turn(session["session_id"], "fail")

    run_id = captured["run_id"]
    snapshot = runner._subagent_scheduler.budget_ledger.snapshot(
        root_run_id=run_id,
        parent_agent_id=run_id,
    )
    assert snapshot.active_global == 0
    assert snapshot.total_for_root == 0
