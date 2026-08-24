from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from threading import Thread
import time
from typing import Any

import pytest

import agent_harness.worktrees as worktrees_module
from agent_harness.core import CancellationToken, HarnessCancelled
from agent_harness.worktrees import WorktreeError, WorktreeManager


GIT = Path(shutil.which("git") or "/usr/bin/git").resolve()


def _git(repository: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [os.fspath(GIT), "-C", os.fspath(repository), *arguments],
        check=check,
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


def _repository(tmp_path: Path, *, committed: bool = True) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Harness Test")
    _git(repository, "config", "user.email", "harness@example.invalid")
    if committed:
        (repository / "tracked.txt").write_text("baseline\n", encoding="utf-8")
        _git(repository, "add", "tracked.txt")
        _git(repository, "commit", "-q", "-m", "baseline")
    return repository


def _manager(tmp_path: Path, repository: Path | None = None) -> WorktreeManager:
    return WorktreeManager(
        repository or _repository(tmp_path),
        state_directory=tmp_path / "state",
        worktree_home=tmp_path / "worktree-home",
        git_executable=GIT,
    )


def _ref_value(repository: Path, ref: str, *, check: bool = True) -> str:
    result = _git(repository, "rev-parse", "--verify", ref, check=check)
    return result.stdout.strip()


def test_create_uses_exact_committed_head_and_opaque_private_state(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    expected_head = _ref_value(repository, "HEAD^{commit}")
    manager = _manager(tmp_path, repository)

    record = manager.create()

    assert record.baseline_commit == expected_head
    assert record.worktree_id.startswith("wt_")
    assert len(record.worktree_id) == 35
    assert record.branch_ref.startswith("refs/heads/agent-harness/worktrees/")
    assert record.worktree_id.removeprefix("wt_") not in record.branch_ref
    assert _ref_value(repository, f"{record.branch_ref}^{{commit}}") == expected_head
    assert Path(record.worktree_path).is_dir()
    assert Path(record.worktree_path).parent == manager.worktree_home
    assert stat.S_IMODE(Path(record.worktree_path).stat().st_mode) == 0o700
    assert stat.S_IMODE((Path(record.worktree_path) / ".git").stat().st_mode) == 0o600
    assert Path(record.git_dir or "").is_relative_to(manager.common_git_dir / "worktrees")
    git_dir = Path(record.git_dir or "")
    assert (git_dir / "gitdir").read_bytes() == f"{Path(record.worktree_path) / '.git'}\n".encode()
    assert (git_dir / "commondir").read_bytes() == b"../..\n"
    assert (git_dir / "locked").read_bytes() == f"{record.lock_reason}\n".encode()
    for mapping in (git_dir / "gitdir", git_dir / "commondir", git_dir / "locked"):
        assert stat.S_IMODE(mapping.stat().st_mode) == 0o600

    loaded = manager.get(record.worktree_id)
    assert loaded == record
    assert manager.list_records() == (record,)
    record_path = manager.records_directory / f"{record.worktree_id}.json"
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    for directory in (
        manager.state_directory,
        manager.records_directory,
        manager.hooks_directory,
        manager.runtime_directory,
        manager.worktree_home,
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_managers_for_linked_worktrees_share_common_git_lock(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    linked = tmp_path / "linked-parent"
    _git(repository, "worktree", "add", "-q", "-b", "linked-parent", os.fspath(linked), "HEAD")
    shared_home = tmp_path / "shared-home"

    first = WorktreeManager(
        repository,
        state_directory=tmp_path / "state-one",
        worktree_home=shared_home,
        git_executable=GIT,
    )
    second = WorktreeManager(
        linked,
        state_directory=tmp_path / "state-two",
        worktree_home=shared_home,
        git_executable=GIT,
    )

    assert first.common_git_dir == second.common_git_dir
    assert first.lock_path == second.lock_path
    assert first.lock_path.parent == shared_home / ".locks"
    with first._locked():
        assert first.lock_path.exists()
        assert stat.S_IMODE(first.lock_path.stat().st_mode) == 0o600


def test_pristine_cleanup_removes_worktree_record_and_ref(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    record = manager.create()

    result = manager.remove_if_pristine(record.worktree_id)

    assert result.removed is True
    assert result.reason == "removed"
    assert not Path(record.worktree_path).exists()
    assert not (manager.records_directory / f"{record.worktree_id}.json").exists()
    assert manager.list_records() == ()
    assert _git(repository, "show-ref", "--verify", record.branch_ref, check=False).returncode != 0


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_create_rejects_dirty_source_including_non_ignored_untracked(
    tmp_path: Path,
    dirty_kind: str,
) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    if dirty_kind == "tracked":
        (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
    else:
        (repository / "untracked.txt").write_text("untracked\n", encoding="utf-8")

    with pytest.raises(WorktreeError, match="including non-ignored untracked"):
        manager.create()

    assert manager.list_records() == ()
    assert not tuple(manager.worktree_home.glob("wt_*"))


def test_ignored_source_content_is_not_misreported_as_untracked(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    (repository / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    _git(repository, "add", ".gitignore")
    _git(repository, "commit", "-q", "-m", "ignore local content")
    ignored = repository / "ignored"
    ignored.mkdir()
    (ignored / "local.txt").write_text("source-only\n", encoding="utf-8")
    manager = _manager(tmp_path, repository)

    record = manager.create()

    assert not (Path(record.worktree_path) / "ignored" / "local.txt").exists()
    assert manager.remove_if_pristine(record.worktree_id).removed is True


def test_create_rejects_unborn_and_non_git_directories(tmp_path: Path) -> None:
    unborn = _repository(tmp_path, committed=False)
    manager = _manager(tmp_path, unborn)
    with pytest.raises(WorktreeError, match="no committed HEAD"):
        manager.create()

    plain = tmp_path / "plain"
    plain.mkdir(mode=0o700)
    with pytest.raises(WorktreeError, match="not a Git worktree"):
        WorktreeManager(
            plain,
            state_directory=tmp_path / "plain-state",
            worktree_home=tmp_path / "plain-home",
            git_executable=GIT,
        )


def test_create_supports_detached_committed_head(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    expected = _ref_value(repository, "HEAD^{commit}")
    _git(repository, "checkout", "-q", "--detach", expected)
    manager = _manager(tmp_path, repository)

    record = manager.create()

    assert record.baseline_commit == expected
    assert _ref_value(repository, record.branch_ref) == expected
    assert manager.remove_if_pristine(record.worktree_id).removed is True


def test_repository_root_rejects_group_writable_mode(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.chmod(0o770)

    with pytest.raises(WorktreeError, match="repository root owner, mode or type"):
        _manager(tmp_path, repository)


@pytest.mark.parametrize("placement", ["state_in_repo", "home_in_repo", "overlap"])
def test_private_locations_must_be_outside_and_disjoint(
    tmp_path: Path,
    placement: str,
) -> None:
    repository = _repository(tmp_path)
    state = tmp_path / "state"
    home = tmp_path / "home"
    if placement == "state_in_repo":
        state = repository / "state"
    elif placement == "home_in_repo":
        home = repository / "home"
    else:
        home = state / "home"

    with pytest.raises(WorktreeError, match="outside|disjoint"):
        WorktreeManager(
            repository,
            state_directory=state,
            worktree_home=home,
            git_executable=GIT,
        )


@pytest.mark.parametrize("kind", ["state", "home"])
def test_private_locations_reject_user_controlled_symlink(
    tmp_path: Path,
    kind: str,
) -> None:
    repository = _repository(tmp_path)
    real = tmp_path / f"real-{kind}"
    real.mkdir(mode=0o700)
    linked = tmp_path / f"linked-{kind}"
    linked.symlink_to(real, target_is_directory=True)
    state = linked if kind == "state" else tmp_path / "state"
    home = linked if kind == "home" else tmp_path / "home"

    with pytest.raises(WorktreeError, match="untrusted symlink"):
        WorktreeManager(
            repository,
            state_directory=state,
            worktree_home=home,
            git_executable=GIT,
        )


def test_private_locations_reject_explicit_relative_paths(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(WorktreeError, match="must be absolute"):
        WorktreeManager(
            repository,
            state_directory=Path("relative-state"),
            worktree_home=tmp_path / "home",
            git_executable=GIT,
        )


def test_changed_tracked_or_untracked_worktree_is_preserved(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    tracked = manager.create()
    tracked_root = Path(tracked.worktree_path)
    (tracked_root / "tracked.txt").write_text("changed\n", encoding="utf-8")

    tracked_result = manager.remove_if_pristine(tracked.worktree_id)

    assert tracked_result == type(tracked_result)(False, "git_status_changed", tracked)
    assert tracked_root.exists()
    assert _ref_value(repository, tracked.branch_ref) == tracked.baseline_commit

    # A second managed worktree proves untracked files are also surfaced by
    # porcelain-v2 rather than being omitted from automatic-cleanup checks.
    (tracked_root / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    assert manager.remove_if_pristine(tracked.worktree_id).removed is True
    untracked = manager.create()
    untracked_root = Path(untracked.worktree_path)
    (untracked_root / "new.txt").write_text("new\n", encoding="utf-8")

    untracked_result = manager.remove_if_pristine(untracked.worktree_id)

    assert untracked_result.reason == "git_status_changed"
    assert untracked_root.exists()


def test_manifest_preserves_git_ignored_content_changes(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    _git(repository, "add", ".gitignore")
    _git(repository, "commit", "-q", "-m", "ignore generated files")
    manager = _manager(tmp_path, repository)
    record = manager.create()
    root = Path(record.worktree_path)
    ignored = root / "ignored"
    ignored.mkdir()
    (ignored / "result.txt").write_text("must survive\n", encoding="utf-8")
    assert _git(root, "status", "--porcelain=v2", "--untracked-files=all").stdout == ""

    result = manager.remove_if_pristine(record.worktree_id)

    assert result.removed is False
    assert result.reason == "content_manifest_changed"
    assert (ignored / "result.txt").read_text(encoding="utf-8") == "must survive\n"
    assert _ref_value(repository, record.branch_ref) == record.baseline_commit


def test_empty_commit_changes_ref_and_preserves_worktree(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    record = manager.create()
    root = Path(record.worktree_path)
    _git(root, "commit", "-q", "--allow-empty", "-m", "child checkpoint")
    assert _git(root, "status", "--porcelain=v2", "--untracked-files=all").stdout == ""

    result = manager.remove_if_pristine(record.worktree_id)

    assert result.removed is False
    assert result.reason == "branch_ref_changed"
    assert root.exists()
    assert _ref_value(repository, record.branch_ref) != record.baseline_commit


def test_git_mapping_mode_mismatch_fails_closed_and_preserves_data(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    record = manager.create()
    root = Path(record.worktree_path)
    mapping = root / ".git"
    mapping.chmod(0o644)

    with pytest.raises(WorktreeError, match=r"\.git mapping owner, mode or type"):
        manager.remove_if_pristine(record.worktree_id)

    assert root.exists()
    assert _ref_value(repository, record.branch_ref) == record.baseline_commit
    assert manager.get(record.worktree_id) == record


def test_git_mapping_symlink_is_rejected_without_following_it(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    record = manager.create()
    root = Path(record.worktree_path)
    mapping = root / ".git"
    relocated = root / ".git.saved"
    mapping.rename(relocated)
    mapping.symlink_to(relocated.name)

    with pytest.raises(WorktreeError, match=r"\.git mapping owner, mode or type"):
        manager.remove_if_pristine(record.worktree_id)

    assert mapping.is_symlink()
    assert relocated.exists()
    assert _ref_value(repository, record.branch_ref) == record.baseline_commit


@pytest.mark.parametrize("mapping_name", ["gitdir", "commondir"])
def test_admin_mapping_mismatch_preserves_worktree(
    tmp_path: Path,
    mapping_name: str,
) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    record = manager.create()
    mapping = Path(record.git_dir or "") / mapping_name
    mapping.write_text("../untrusted\n", encoding="utf-8")
    mapping.chmod(0o600)

    with pytest.raises(WorktreeError, match=f"{mapping_name}.*does not match"):
        manager.remove_if_pristine(record.worktree_id)

    assert Path(record.worktree_path).exists()
    assert _ref_value(repository, record.branch_ref) == record.baseline_commit
    assert manager.get(record.worktree_id) == record


def test_exact_lock_reason_mismatch_preserves_worktree(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    record = manager.create()
    locked = Path(record.git_dir or "") / "locked"
    locked.write_text("some-other-owner\n", encoding="utf-8")
    locked.chmod(0o600)

    with pytest.raises(WorktreeError, match="lock reason does not match"):
        manager.remove_if_pristine(record.worktree_id)

    assert Path(record.worktree_path).exists()
    assert _ref_value(repository, record.branch_ref) == record.baseline_commit


def test_post_record_creation_failure_exposes_opaque_artifact_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)

    def fail_after_git_created(_record: Any) -> str:
        raise WorktreeError("synthetic mapping failure")

    monkeypatch.setattr(manager, "_prepare_git_mapping", fail_after_git_created)

    with pytest.raises(WorktreeError, match="synthetic mapping failure") as raised:
        manager.create()

    assert raised.value.code == "create_incomplete"
    assert raised.value.worktree_id is not None
    record = manager.get(raised.value.worktree_id)
    assert record.phase == "creating"
    assert record.worktree_id == raised.value.worktree_id
    assert Path(record.worktree_path).exists()
    assert _ref_value(repository, record.branch_ref) == record.baseline_commit


def test_tampered_or_symlinked_record_is_rejected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    record = manager.create()
    path = manager.records_directory / f"{record.worktree_id}.json"
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    raw["worktree_path"] = os.fspath(tmp_path / "outside")
    path.write_text(json.dumps(raw), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(WorktreeError, match="path escapes"):
        manager.get(record.worktree_id)

    path.unlink()
    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    external.chmod(0o600)
    path.symlink_to(external)
    with pytest.raises(WorktreeError, match="owner-only regular file"):
        manager.get(record.worktree_id)


def test_replaced_state_component_is_rejected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    record = manager.create()
    relocated = tmp_path / "state-relocated"
    manager.state_directory.rename(relocated)
    manager.state_directory.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(WorktreeError, match="path contains a symlink"):
        manager.get(record.worktree_id)

    assert Path(record.worktree_path).exists()
    assert _ref_value(repository, record.branch_ref) == record.baseline_commit


@pytest.mark.parametrize("config_kind", ["filter", "include"])
def test_executable_filter_or_config_include_is_rejected_before_creation(
    tmp_path: Path,
    config_kind: str,
) -> None:
    repository = _repository(tmp_path)
    if config_kind == "filter":
        _git(repository, "config", "filter.unsafe.smudge", "/bin/sh -c true")
    else:
        included = tmp_path / "included.gitconfig"
        included.write_text('[filter "unsafe"]\n\tsmudge = /bin/sh -c true\n', encoding="utf-8")
        _git(repository, "config", "include.path", os.fspath(included))
    manager = _manager(tmp_path, repository)

    with pytest.raises(WorktreeError, match="filters|includes"):
        manager.create()

    assert manager.list_records() == ()
    assert not tuple(manager.worktree_home.glob("wt_*"))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("core.worktree", "/tmp/redirected"),
        ("core.sparseCheckout", "true"),
        ("core.attributesFile", "/tmp/attributes"),
        ("submodule.recurse", "true"),
        ("checkout.workers", "8"),
        ("extensions.worktreeConfig", "true"),
    ],
)
def test_unsafe_checkout_controls_are_rejected_before_creation(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    repository = _repository(tmp_path)
    _git(repository, "config", name, value)
    manager = _manager(tmp_path, repository)

    with pytest.raises(WorktreeError, match="unsupported checkout control"):
        manager.create()

    assert manager.list_records() == ()


def test_control_file_mutation_is_detected_before_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    original_prepare = manager._prepare_git_mapping

    def mutate_after_allocation(record: Any) -> str:
        git_dir = original_prepare(record)
        _git(repository, "config", "filter.raced.smudge", "/bin/sh -c true")
        return git_dir

    monkeypatch.setattr(manager, "_prepare_git_mapping", mutate_after_allocation)

    with pytest.raises(WorktreeError, match="control files changed") as raised:
        manager.create()

    assert raised.value.code == "create_incomplete"
    assert raised.value.worktree_id is not None
    assert manager.get(raised.value.worktree_id).phase == "creating"


def test_trusted_git_requires_non_writable_root_owned_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_lstat = Path.lstat
    unsafe_ancestor = GIT.parent

    def lstat_with_writable_ancestor(path: Path) -> os.stat_result:
        metadata = real_lstat(path)
        if path == unsafe_ancestor:
            values = list(metadata)
            values[0] |= stat.S_IWGRP
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr(Path, "lstat", lstat_with_writable_ancestor)

    with pytest.raises(WorktreeError, match="ancestry is not trusted"):
        worktrees_module._trusted_git_executable(GIT)


def test_git_processes_use_direct_argv_and_scrubbed_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    real_popen = subprocess.Popen
    observed: list[tuple[Any, dict[str, Any]]] = []

    def recording_popen(argv: Any, **kwargs: Any) -> Any:
        observed.append((argv, kwargs))
        return real_popen(argv, **kwargs)

    monkeypatch.setattr("agent_harness.worktrees.subprocess.Popen", recording_popen)
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/untrusted/library")
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", "/untrusted/diff")
    monkeypatch.setenv("GIT_DIR", "/untrusted/repository")
    manager = _manager(tmp_path, repository)
    record = manager.create()
    assert manager.remove_if_pristine(record.worktree_id).removed is True

    rendered = [tuple(str(item) for item in argv) for argv, _kwargs in observed]
    add_index = next(
        index for index, argv in enumerate(rendered) if "worktree" in argv and "add" in argv
    )
    checkout_index = next(index for index, argv in enumerate(rendered) if "checkout" in argv)
    assert "--no-checkout" in rendered[add_index]
    assert add_index < checkout_index

    assert observed
    for argv, kwargs in observed:
        assert isinstance(argv, tuple)
        assert Path(argv[0]).is_absolute()
        assert kwargs["shell"] is False
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["start_new_session"] is True
        environment = kwargs["env"]
        assert "DYLD_INSERT_LIBRARIES" not in environment
        assert "GIT_EXTERNAL_DIFF" not in environment
        assert "GIT_DIR" not in environment
        assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
        assert "core.fsmonitor=false" in argv
        assert "diff.external=" in argv
        assert any(str(item).startswith("core.hooksPath=") for item in argv)
        assert "--force" not in argv
        assert not {"reset", "clean", "prune"}.intersection(argv)


def test_cas_ref_race_preserves_changed_ref_and_durable_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    # Create another object that can replace the managed ref during the narrow
    # remove/ref-delete interval.
    original = _ref_value(repository, "HEAD^{commit}")
    _git(repository, "commit", "-q", "--allow-empty", "-m", "baseline two")
    manager = _manager(tmp_path, repository)
    record = manager.create()
    assert record.baseline_commit != original
    original_git = manager._git
    raced = False

    def racing_git(*arguments: str, **kwargs: Any) -> tuple[int, bytes, bytes]:
        nonlocal raced
        if arguments[:2] == ("update-ref", "-d") and not raced:
            raced = True
            _git(repository, "update-ref", record.branch_ref, original)
        return original_git(*arguments, **kwargs)

    monkeypatch.setattr(manager, "_git", racing_git)

    result = manager.remove_if_pristine(record.worktree_id)

    assert raced is True
    assert result.removed is False
    assert result.reason == "branch_ref_preserved"
    assert result.record.phase == "ref_preserved"
    assert _ref_value(repository, record.branch_ref) == original
    assert manager.get(record.worktree_id).phase == "ref_preserved"


def test_removing_phase_recovers_after_crash_following_unlock(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    record = manager.create()
    removing = replace(record, phase="removing")
    manager._store_record(removing)
    _git(repository, "worktree", "unlock", record.worktree_path)
    assert not (Path(record.git_dir or "") / "locked").exists()

    result = manager.remove_if_pristine(record.worktree_id)

    assert result.removed is True
    assert not Path(record.worktree_path).exists()
    assert manager.list_records() == ()


def test_removing_phase_finalizes_after_worktree_path_was_removed(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    record = manager.create()
    manager._store_record(replace(record, phase="removing"))
    _git(repository, "worktree", "unlock", record.worktree_path)
    _git(repository, "worktree", "remove", record.worktree_path)
    assert not Path(record.worktree_path).exists()
    assert not Path(record.git_dir or "").exists()
    assert _ref_value(repository, record.branch_ref) == record.baseline_commit

    result = manager.remove_if_pristine(record.worktree_id)

    assert result.removed is True
    assert manager.list_records() == ()
    assert (
        _git(
            repository,
            "show-ref",
            "--verify",
            record.branch_ref,
            check=False,
        ).returncode
        != 0
    )


def test_lock_wait_honors_operation_deadline_without_allocating(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    failures: list[BaseException] = []

    def attempt() -> None:
        try:
            manager.create(deadline_monotonic=time.monotonic() + 0.1)
        except BaseException as exc:
            failures.append(exc)

    with manager._locked():
        thread = Thread(target=attempt)
        thread.start()
        thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], WorktreeError)
    assert failures[0].code == "worktree_deadline_exceeded"
    assert manager.list_records() == ()


def test_lock_wait_honors_cancellation_without_allocating(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manager = _manager(tmp_path, repository)
    token = CancellationToken()
    failures: list[BaseException] = []

    def attempt() -> None:
        try:
            manager.create(cancellation_token=token)
        except BaseException as exc:
            failures.append(exc)

    with manager._locked():
        thread = Thread(target=attempt)
        thread.start()
        time.sleep(0.05)
        assert token.cancel("test cancellation") is True
        thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], HarnessCancelled)
    assert manager.list_records() == ()
