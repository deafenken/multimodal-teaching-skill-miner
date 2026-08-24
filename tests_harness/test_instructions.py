from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import agent_harness.instructions as instructions_module
from agent_harness.cli import main as cli_main
from agent_harness.instructions import (
    INSTRUCTION_SNAPSHOT_SCHEMA,
    MAX_INSTRUCTION_BYTES,
    InstructionLoadError,
    load_project_instructions,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_discovers_root_to_leaf_and_override_wins_per_level(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    leaf = workspace / "packages" / "api"
    leaf.mkdir(parents=True)
    _write(workspace / "AGENTS.md", "root")
    _write(workspace / "packages" / "AGENTS.md", "shadowed")
    _write(workspace / "packages" / "AGENTS.override.md", "package override")
    _write(leaf / "AGENTS.md", "leaf")

    snapshot = load_project_instructions(workspace, leaf)

    assert snapshot.schema == INSTRUCTION_SNAPSHOT_SCHEMA
    assert snapshot.active_relative_path == "packages/api"
    assert [item.relative_path for item in snapshot.documents] == [
        "AGENTS.md",
        "packages/AGENTS.override.md",
        "packages/api/AGENTS.md",
    ]
    assert [item.content for item in snapshot.documents] == ["root", "package override", "leaf"]
    assert snapshot.total_bytes == len("rootpackage overrideleaf".encode())


def test_discovery_is_scoped_to_explicit_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "outer" / "workspace"
    leaf = workspace / "src"
    leaf.mkdir(parents=True)
    _write(tmp_path / "outer" / "AGENTS.md", "must not load")
    _write(workspace / "AGENTS.md", "workspace only")

    snapshot = load_project_instructions(workspace, leaf)

    assert [item.content for item in snapshot.documents] == ["workspace only"]
    with pytest.raises(InstructionLoadError, match="inside workspace root"):
        load_project_instructions(workspace, tmp_path / "outer")


def test_root_default_and_empty_snapshot_are_deterministic(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    first = load_project_instructions(workspace)
    second = load_project_instructions(workspace, workspace)

    assert first.documents == ()
    assert first.total_bytes == 0
    assert first.active_relative_path == "."
    assert first.snapshot_sha256 == second.snapshot_sha256


def test_selected_symlink_fails_closed_instead_of_falling_back(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "outside.md"
    target.write_text("outside", encoding="utf-8")
    (workspace / "AGENTS.override.md").symlink_to(target)
    _write(workspace / "AGENTS.md", "fallback must not load")

    with pytest.raises(InstructionLoadError, match="regular non-symlink"):
        load_project_instructions(workspace)


def test_non_regular_candidate_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "AGENTS.md").mkdir(parents=True)

    with pytest.raises(InstructionLoadError, match="regular non-symlink"):
        load_project_instructions(workspace)


def test_replacement_between_stat_and_open_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = workspace / "AGENTS.md"
    replacement = workspace / "replacement.tmp"
    _write(candidate, "first")
    _write(replacement, "other")
    original_open = instructions_module.os.open
    replaced = False

    def racing_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if path == "AGENTS.md" and not replaced:
            replaced = True
            replacement.replace(candidate)
        return original_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(instructions_module.os, "open", racing_open)
    with pytest.raises(InstructionLoadError, match="changed before reading"):
        load_project_instructions(workspace)


def test_invalid_utf8_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_bytes(b"\xff\xfe")

    with pytest.raises(InstructionLoadError, match="valid UTF-8"):
        load_project_instructions(workspace)


def test_combined_byte_limit_is_enforced_and_exact_limit_is_allowed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    child = workspace / "child"
    child.mkdir(parents=True)
    _write(workspace / "AGENTS.md", "a" * (MAX_INSTRUCTION_BYTES - 1))
    _write(child / "AGENTS.md", "b")

    snapshot = load_project_instructions(workspace, child)
    assert snapshot.total_bytes == MAX_INSTRUCTION_BYTES

    _write(child / "AGENTS.md", "bb")
    with pytest.raises(InstructionLoadError, match="exceed 32 KiB"):
        load_project_instructions(workspace, child)


def test_custom_zero_limit_allows_only_empty_documents(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write(workspace / "AGENTS.md", "")
    assert load_project_instructions(workspace, max_bytes=0).total_bytes == 0

    _write(workspace / "AGENTS.md", "x")
    with pytest.raises(InstructionLoadError, match="exceed 32 KiB"):
        load_project_instructions(workspace, max_bytes=0)


def test_content_and_snapshot_digests_are_canonical_and_relocation_stable(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write(first_root / "sub" / "AGENTS.md", "使用 UTF-8。\n")
    _write(second_root / "sub" / "AGENTS.md", "使用 UTF-8。\n")

    first = load_project_instructions(first_root, first_root / "sub")
    relocated = load_project_instructions(second_root, second_root / "sub")

    assert len(first.documents[0].content_sha256) == 64
    assert first.documents[0].content_sha256 == relocated.documents[0].content_sha256
    assert first.snapshot_sha256 == relocated.snapshot_sha256

    _write(second_root / "sub" / "AGENTS.md", "不同内容。\n")
    changed = load_project_instructions(second_root, second_root / "sub")
    assert changed.snapshot_sha256 != first.snapshot_sha256


def test_model_content_is_structured_and_metadata_does_not_expose_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write(workspace / "AGENTS.md", "keep this exact")

    snapshot = load_project_instructions(workspace)
    rendered = snapshot.to_model_content()
    metadata = snapshot.metadata()

    assert "untrusted model context" in rendered
    assert "keep this exact" in rendered
    assert snapshot.snapshot_sha256 in rendered
    assert "keep this exact" not in repr(metadata)
    assert metadata["documents"][0]["relative_path"] == "AGENTS.md"


def test_snapshot_and_documents_are_immutable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write(workspace / "AGENTS.md", "rules")
    snapshot = load_project_instructions(workspace)

    with pytest.raises(FrozenInstanceError):
        snapshot.total_bytes = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.documents[0].content = "changed"  # type: ignore[misc]


def test_invalid_max_bytes_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="max_bytes"):
            load_project_instructions(workspace, max_bytes=value)  # type: ignore[arg-type]


def test_cli_reports_metadata_without_provider_credentials_or_content(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write(workspace / "AGENTS.md", "PRIVATE-INSTRUCTION-CONTENT")

    assert cli_main(["--cwd", str(workspace), "instructions", "--json"]) == 0
    output = capsys.readouterr().out

    assert '"schema":"agent_harness.instructions.v1"' in output
    assert '"relative_path":"AGENTS.md"' in output
    assert "PRIVATE-INSTRUCTION-CONTENT" not in output
