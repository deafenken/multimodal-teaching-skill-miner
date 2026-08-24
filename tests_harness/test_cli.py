from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

import agent_harness.cli as cli_module
from agent_harness.cli import EXIT_CONFIG, EXIT_USAGE, main as cli_main
from agent_harness.session import SessionStore
from agent_harness.worktrees import WorktreeError, WorktreeManager


GIT = Path(shutil.which("git") or "/usr/bin/git").resolve()


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [os.fspath(GIT), "-C", os.fspath(repository), *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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
    (repository / "tracked.txt").write_text(
        "PRIVATE_REPOSITORY_CONTENT\n", encoding="utf-8"
    )
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-q", "-m", "baseline")
    return repository


def _artifact(tmp_path: Path) -> tuple[Path, Path, Path, Any]:
    repository = _repository(tmp_path)
    state_home = tmp_path / "state"
    worktree_home = tmp_path / "isolated-worktrees"
    store = SessionStore(repository, state_home=state_home)
    manager = WorktreeManager(
        repository,
        state_directory=store.root / "worktrees",
        worktree_home=worktree_home,
        git_executable=GIT,
    )
    return repository, state_home, worktree_home, manager.create()


def _common(
    repository: Path,
    state_home: Path,
    worktree_home: Path,
) -> list[str]:
    return [
        "--cwd",
        os.fspath(repository),
        "--state-home",
        os.fspath(state_home),
        "--worktree-home",
        os.fspath(worktree_home),
    ]


def test_agents_list_and_detail_are_content_free_without_provider(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, state_home, worktree_home, record = _artifact(tmp_path)
    common = _common(repository, state_home, worktree_home)

    def provider_must_not_start(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("agents must not construct AgentRunner")

    monkeypatch.setattr(cli_module, "AgentRunner", provider_must_not_start)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY_FILE", raising=False)

    assert cli_main([*common, "agents", "--json"]) == 0
    listed_text = capsys.readouterr().out
    listed = json.loads(listed_text)
    assert listed["schema"] == "agent_harness.worktree_list.v1"
    assert listed["worktrees"][0]["worktree_id"] == record.worktree_id
    assert set(listed["worktrees"][0]) == {
        "schema",
        "worktree_id",
        "phase",
        "baseline_commit",
        "baseline_manifest_sha256",
        "created_at",
    }

    for private_value in (
        record.repository,
        record.common_git_dir,
        record.worktree_path,
        record.git_dir,
        os.fspath(worktree_home),
    ):
        assert private_value is not None
        assert str(private_value) not in listed_text
    assert "PRIVATE_REPOSITORY_CONTENT" not in listed_text

    assert cli_main([*common, "agents", record.worktree_id, "--json"]) == 0
    detail_text = capsys.readouterr().out
    detail = json.loads(detail_text)
    assert detail["schema"] == "agent_harness.worktree_status.v1"
    assert detail["worktree"]["worktree_id"] == record.worktree_id
    assert "worktree_path" not in detail["worktree"]
    assert record.worktree_path not in detail_text
    assert record.repository not in detail_text


def test_agents_path_requires_exact_id_and_reveals_only_worktree_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository, state_home, worktree_home, record = _artifact(tmp_path)
    common = _common(repository, state_home, worktree_home)

    assert cli_main([*common, "agents", "--path"]) == EXIT_USAGE
    captured = capsys.readouterr()
    assert "--path requires WORKTREE_ID" in captured.err
    assert captured.out == ""

    assert cli_main([*common, "agents", record.worktree_id, "--path"]) == 0
    assert capsys.readouterr().out.strip() == record.worktree_path

    assert (
        cli_main(
            [*common, "agents", record.worktree_id, "--path", "--json"]
        )
        == 0
    )
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["worktree"]["worktree_path"] == record.worktree_path
    assert record.repository not in output
    assert record.common_git_dir not in output
    assert str(record.git_dir) not in output
    assert "repository" not in payload["worktree"]
    assert "common_git_dir" not in payload["worktree"]
    assert "git_dir" not in payload["worktree"]


def test_agents_uses_default_worktree_home_and_empty_list_is_content_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    state_home = tmp_path / "state"
    expected_home = tmp_path / "default-worktrees"
    captured_home: list[Path] = []

    class FakeManager:
        def __init__(
            self,
            _repository: Path,
            *,
            state_directory: Path,
            worktree_home: Path,
        ) -> None:
            assert state_directory.name == "worktrees"
            captured_home.append(worktree_home)

        def list_records(self, *, limit: int | None = None) -> tuple[()]:
            assert limit == 256
            return ()

    monkeypatch.setattr(cli_module, "default_worktree_home", lambda: expected_home)
    monkeypatch.setattr(cli_module, "WorktreeManager", FakeManager)

    assert (
        cli_main(
            [
                "--cwd",
                os.fspath(repository),
                "--state-home",
                os.fspath(state_home),
                "agents",
                "--json",
            ]
        )
        == 0
    )
    assert captured_home == [expected_home]
    assert json.loads(capsys.readouterr().out) == {
        "schema": "agent_harness.worktree_list.v1",
        "worktrees": [],
    }


def test_worktree_errors_are_redacted_and_return_config_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    state_home = tmp_path / "state"
    private_marker = "/private/secret/worktree-path"

    def fail_manager(*_args: Any, **_kwargs: Any) -> Any:
        raise WorktreeError(f"tampered record at {private_marker}")

    monkeypatch.setattr(cli_module, "WorktreeManager", fail_manager)
    code = cli_main(
        [
            "--cwd",
            os.fspath(repository),
            "--state-home",
            os.fspath(state_home),
            "--worktree-home",
            os.fspath(tmp_path / "worktrees"),
            "agents",
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_CONFIG
    assert captured.out == ""
    assert captured.err == "harness: isolated worktree state is unavailable\n"
    assert private_marker not in captured.err


def test_agents_over_limit_inventory_fails_without_partial_or_private_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, state_home, worktree_home, record = _artifact(tmp_path)
    store = SessionStore(repository, state_home=state_home)
    records_directory = store.root / "worktrees" / "records"
    index = 0
    while len(tuple(records_directory.iterdir())) <= 256:
        candidate = records_directory / f"wt_{index:032x}.json"
        index += 1
        if candidate.exists():
            continue
        candidate.write_text("PRIVATE_RECORD_MARKER", encoding="utf-8")
        candidate.chmod(0o600)

    observed_limits: list[int | None] = []
    original_list_records = WorktreeManager.list_records

    def tracked_list_records(
        self: WorktreeManager,
        *,
        limit: int | None = None,
    ) -> tuple[Any, ...]:
        observed_limits.append(limit)
        return original_list_records(self, limit=limit)

    monkeypatch.setattr(WorktreeManager, "list_records", tracked_list_records)
    code = cli_main([*_common(repository, state_home, worktree_home), "agents", "--json"])

    captured = capsys.readouterr()
    assert code == EXIT_CONFIG
    assert observed_limits == [256]
    assert captured.out == ""
    assert captured.err == "harness: isolated worktree state is unavailable\n"
    assert "PRIVATE_RECORD_MARKER" not in captured.err
    assert record.repository not in captured.err
    assert record.worktree_path not in captured.err


def test_global_worktree_home_is_forwarded_to_agent_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "workspace"
    repository.mkdir()
    selected_home = tmp_path / "selected-worktree-home"
    captured_kwargs: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, _workspace: Path, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        @property
        def provider_status(self) -> dict[str, Any]:
            return {"provider": "fixture"}

    monkeypatch.setattr(cli_module, "AgentRunner", FakeRunner)
    assert (
        cli_main(
            [
                "--cwd",
                os.fspath(repository),
                "--worktree-home",
                os.fspath(selected_home),
                "status",
            ]
        )
        == 0
    )
    assert captured_kwargs["worktree_home"] == os.fspath(selected_home)
    assert json.loads(capsys.readouterr().out)["provider"] == "fixture"
