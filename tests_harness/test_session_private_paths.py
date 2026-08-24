from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

from agent_harness.session import SessionStore, SessionStoreError


_DIGEST = "a" * 64


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    return workspace


def _catalog_record() -> dict[str, object]:
    return {
        "schema": "agent_harness.mcp_catalog_record.v1",
        "definition_sha256": _DIGEST,
        "protocol_version": "2025-06-18",
        "catalog_sha256": _DIGEST,
        "server_info_sha256": _DIGEST,
        "instructions_sha256": _DIGEST,
        "tools": [],
        "rejected_tools": [],
    }


def test_state_home_rejects_user_owned_intermediate_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    state_home = linked_parent / "state"

    with pytest.raises(SessionStoreError, match="untrusted symlink"):
        SessionStore(workspace, state_home=state_home)

    monkeypatch.setenv("AGENT_HARNESS_HOME", os.fspath(state_home))
    with pytest.raises(SessionStoreError, match="untrusted symlink"):
        SessionStore(workspace)
    assert not (real_parent / "state").exists()


def test_state_home_rejects_writable_user_controlled_ancestor(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    unsafe_parent = tmp_path / "unsafe-parent"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)

    with pytest.raises(SessionStoreError, match="ancestor permissions"):
        SessionStore(workspace, state_home=unsafe_parent / "state")

    assert stat.S_IMODE(unsafe_parent.stat().st_mode) == 0o777
    assert not (unsafe_parent / "state").exists()


def test_nested_missing_state_directories_are_created_private(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    state_home = tmp_path / "private-parent" / "nested" / "state"
    store = SessionStore(workspace, state_home=state_home)

    for directory in (
        tmp_path / "private-parent",
        tmp_path / "private-parent" / "nested",
        state_home,
        state_home / "workspaces",
        store.root,
        store.sessions_directory,
        store.runs_directory,
    ):
        metadata = directory.lstat()
        assert stat.S_ISDIR(metadata.st_mode)
        assert not stat.S_ISLNK(metadata.st_mode)
        assert metadata.st_uid == os.getuid()
        assert stat.S_IMODE(metadata.st_mode) == 0o700


@pytest.mark.skipif(
    not Path("/var").is_symlink(),
    reason="macOS root filesystem alias is unavailable",
)
def test_root_owned_macos_var_alias_remains_supported(tmp_path: Path) -> None:
    resolved = os.fspath(tmp_path.resolve())
    if not resolved.startswith("/private/var/"):
        pytest.skip("temporary directory is not under the macOS /var alias")
    alias = Path(resolved.removeprefix("/private")) / "state-via-var-alias"
    workspace = _workspace(tmp_path)

    store = SessionStore(workspace, state_home=alias)

    assert store.root.is_relative_to(Path("/private/var"))
    assert stat.S_IMODE(alias.stat().st_mode) == 0o700


def test_mcp_trust_and_catalog_are_private_regular_files(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = SessionStore(workspace, state_home=tmp_path / "state")

    store.set_mcp_trust("fixture", _DIGEST, action="trusted")
    store.set_mcp_catalog("fixture", _catalog_record())

    assert store.mcp_trust_state()["fixture"] == {
        "action": "trusted",
        "definition_sha256": _DIGEST,
    }
    assert store.mcp_catalog_state()["fixture"]["catalog_sha256"] == _DIGEST
    for path in (store.mcp_trust_path, store.mcp_catalog_path):
        metadata = path.lstat()
        assert stat.S_ISREG(metadata.st_mode)
        assert not stat.S_ISLNK(metadata.st_mode)
        assert metadata.st_uid == os.getuid()
        assert metadata.st_nlink == 1
        assert stat.S_IMODE(metadata.st_mode) == 0o600


def test_mcp_state_rejects_replaced_intermediate_directory(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    state_home = tmp_path / "state"
    store = SessionStore(workspace, state_home=state_home)
    store.set_mcp_trust("fixture", _DIGEST, action="trusted")
    store.set_mcp_catalog("fixture", _catalog_record())
    workspaces = state_home / "workspaces"
    relocated = state_home / "workspaces-relocated"
    workspaces.rename(relocated)
    workspaces.symlink_to(relocated, target_is_directory=True)

    try:
        with pytest.raises(SessionStoreError, match="untrusted symlink"):
            store.mcp_trust_state()
        with pytest.raises(SessionStoreError, match="untrusted symlink"):
            store.mcp_catalog_state()
    finally:
        workspaces.unlink()
        relocated.rename(workspaces)

    assert store.mcp_trust_state()["fixture"]["action"] == "trusted"
    assert store.mcp_catalog_state()["fixture"]["catalog_sha256"] == _DIGEST


@pytest.mark.parametrize("kind", ["trust", "catalog"])
def test_mcp_state_rejects_symlink_leaf(tmp_path: Path, kind: str) -> None:
    workspace = _workspace(tmp_path)
    store = SessionStore(workspace, state_home=tmp_path / "state")
    if kind == "trust":
        store.set_mcp_trust("fixture", _DIGEST, action="trusted")
        path = store.mcp_trust_path
        load = store.mcp_trust_state
    else:
        store.set_mcp_catalog("fixture", _catalog_record())
        path = store.mcp_catalog_path
        load = store.mcp_catalog_state
    relocated = path.with_name(f"{path.name}.relocated")
    path.rename(relocated)
    path.symlink_to(relocated)

    try:
        with pytest.raises(SessionStoreError, match="regular non-symlink"):
            load()
    finally:
        path.unlink()
        relocated.rename(path)
