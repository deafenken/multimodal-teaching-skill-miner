#!/usr/bin/env python3
"""Create one bounded source-and-evidence snapshot for release verification.

The normal source set is every tracked file plus every untracked, non-ignored
file.  Ignored private research trees are not copied wholesale: only the small
files that the conditional release gates actually read are admitted.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
from typing import Any, Iterable


SNAPSHOT_SCHEMA = "teaching_skill_miner.release_source_snapshot.v1"
MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
EXCLUDED_PARTS = frozenset(
    {
        ".next",
        ".release-acceptance.lock",
        ".test-dist",
        ".turbo",
        "__pycache__",
        "coverage",
        "dist",
        "node_modules",
    }
)
EXCLUDED_PREFIXES = (
    PurePosixPath("artifacts/private"),
    PurePosixPath(".private"),
)
TEACHOBS_ROOT = PurePosixPath("artifacts/private/external_datasets/teachobs")
TEACHOBS_GATE_FILES = (
    TEACHOBS_ROOT / "imported_annotations/dataset_audit.json",
    TEACHOBS_ROOT / "media/media_manifest.json",
    TEACHOBS_ROOT / "media/feature_manifest.json",
    TEACHOBS_ROOT / "captions/caption_audit.json",
    TEACHOBS_ROOT / "asr/job_manifest.json",
    TEACHOBS_ROOT / "asr/import_audit.json",
    TEACHOBS_ROOT / "asr/transcript_coverage_matrix.json",
    TEACHOBS_ROOT / "multimodal_benchmark_result.json",
)


def _included(relative: PurePosixPath) -> bool:
    if not relative.parts or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if (
        len(relative.parts) == 2
        and relative.parts[0] == "artifacts"
        and relative.name.startswith("release_acceptance_")
        and relative.suffix == ".json"
    ):
        return False
    return not any(
        relative.parts[: len(prefix.parts)] == prefix.parts
        for prefix in EXCLUDED_PREFIXES
    )


def _git_paths(root: Path) -> set[PurePosixPath]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("release source snapshot requires a Git checkout")
    paths: set[PurePosixPath] = set()
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            relative = PurePosixPath(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise RuntimeError("release source path is not valid UTF-8") from exc
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Git returned an unsafe release source path")
        if _included(relative):
            paths.add(relative)
    return paths


def _relative_optional_path(root: Path, requested: str | Path) -> PurePosixPath:
    """Resolve a possibly absent configured path without permitting escape."""

    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "conditional release evidence path is outside the repository"
        ) from exc
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise RuntimeError("conditional release evidence path is unsafe")
    return pure


def _tree_files(root: Path, relative_root: PurePosixPath) -> set[PurePosixPath]:
    tree = root.joinpath(*relative_root.parts)
    if not os.path.lexists(tree):
        return set()
    metadata = tree.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(
            f"conditional evidence tree is not a directory: {relative_root}"
        )
    paths: set[PurePosixPath] = set()
    for directory, directory_names, file_names in os.walk(tree, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            if child.is_symlink():
                raise RuntimeError(f"conditional evidence contains a symlink: {child}")
        for name in file_names:
            child = directory_path / name
            relative = PurePosixPath(child.relative_to(root).as_posix())
            if child.is_symlink() or not child.is_file():
                raise RuntimeError(f"conditional evidence is not regular: {relative}")
            paths.add(relative)
    return paths


def _tree_contains_material(path: Path) -> bool:
    """Return whether a private evidence tree has any non-directory entry.

    Conditional evidence is intentionally ignored by Git.  Detecting only a
    completed manifest pair would therefore make a partial run disappear from
    the immutable snapshot and let the in-snapshot gate mistake it for an
    absent study.  Symlinks and special files count as material here so they
    fail closed rather than being hidden by the conditional copy policy.
    """

    if not os.path.lexists(path):
        return False
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        return True
    for directory, directory_names, file_names in os.walk(path, followlinks=False):
        if file_names:
            return True
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directory_names):
            return True
    return False


def _real_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _conditional_evidence_paths(root: Path) -> set[PurePosixPath]:
    paths: set[PurePosixPath] = set()
    paths.update(_tree_files(root, PurePosixPath("artifacts/public")))

    formal_root = root / "artifacts/private/formal_captions"
    formal_manifest = formal_root / "dataset_manifest.json"
    formal_present = _tree_contains_material(formal_root)
    if formal_present and not _real_file(formal_manifest):
        raise RuntimeError(
            "partial formal-caption evidence is present without its dataset manifest"
        )
    if _real_file(formal_manifest):
        paths.update(
            _tree_files(root, PurePosixPath("artifacts/private/formal_captions"))
        )

    frozen_requested = os.environ.get(
        "TSM_TEACHOBS_FROZEN_MODEL_ROOT",
        (TEACHOBS_ROOT / "frozen_models").as_posix(),
    )
    frozen_relative = _relative_optional_path(root, frozen_requested)
    if frozen_relative.parts[:2] != ("artifacts", "private"):
        raise RuntimeError("TeachObs frozen evidence must be below artifacts/private")
    human_requested = os.environ.get(
        "TSM_TEACHOBS_HUMAN_ROOT",
        (TEACHOBS_ROOT / "human_annotation").as_posix(),
    )
    human_relative = _relative_optional_path(root, human_requested)
    if human_relative.parts[:2] != ("artifacts", "private"):
        raise RuntimeError("TeachObs human evidence must be below artifacts/private")
    public_optional_paths: list[PurePosixPath] = []
    for variable, default_path in (
        (
            "TSM_TEACHOBS_HUMAN_RECEIPT",
            "artifacts/public/teachobs_human_annotation_receipt.json",
        ),
        (
            "TSM_TEACHOBS_LOCKBOX_DRAFT",
            "artifacts/public/teachobs_new_site_lockbox_preregistration_draft.json",
        ),
    ):
        requested = os.environ.get(variable, default_path)
        relative = _relative_optional_path(root, requested)
        if relative.parts[:2] != ("artifacts", "public"):
            raise RuntimeError(f"{variable} must be below artifacts/public")
        public_optional_paths.append(relative)

    media_manifest = root.joinpath(*(TEACHOBS_ROOT / "media/media_manifest.json").parts)
    caption_audit = root.joinpath(
        *(TEACHOBS_ROOT / "captions/caption_audit.json").parts
    )
    teachobs_root = root.joinpath(*TEACHOBS_ROOT.parts)
    configured_material_present = any(
        _tree_contains_material(root.joinpath(*relative.parts))
        for relative in (frozen_relative, human_relative, *public_optional_paths)
    )
    teachobs_present = (
        _tree_contains_material(teachobs_root) or configured_material_present
    )
    media_ready = _real_file(media_manifest)
    caption_ready = _real_file(caption_audit)
    if teachobs_present and not (media_ready and caption_ready):
        raise RuntimeError(
            "partial TeachObs private evidence is present without both the media "
            "manifest and caption audit"
        )
    if media_ready and caption_ready:
        for relative in TEACHOBS_GATE_FILES:
            candidate = root.joinpath(*relative.parts)
            if candidate.exists() or candidate.is_symlink():
                paths.add(relative)
        paths.update(_tree_files(root, TEACHOBS_ROOT / "materialized_transcripts"))

        paths.update(_tree_files(root, frozen_relative))
        paths.update(_tree_files(root, human_relative))
        for relative in public_optional_paths:
            candidate = root.joinpath(*relative.parts)
            if os.path.lexists(candidate):
                if candidate.is_symlink() or not candidate.is_file():
                    raise RuntimeError(
                        "configured TeachObs public evidence is not a regular file"
                    )
                paths.add(relative)
    return paths


def _copy_file(source: Path, target: Path) -> tuple[int, str, int]:
    before = source.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"release snapshot source is not regular: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()
    size = 0
    with source.open("rb") as source_handle, target.open("xb") as target_handle:
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            target_handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    after = source.lstat()
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"release source changed while snapshotting: {source}")
    mode = 0o555 if before.st_mode & 0o111 else 0o444
    os.chmod(target, mode)
    return size, digest.hexdigest(), stat.S_IMODE(before.st_mode)


def _binding(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    payload = json.dumps(
        selected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "file_count": len(selected),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in selected),
        "sha256": sha256(payload).hexdigest(),
    }


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _copy_minimal_git_index(root: Path, destination: Path) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "--git-path", "index"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("release source snapshot cannot locate the Git index")
    index = Path(result.stdout.strip())
    if not index.is_absolute():
        index = root / index
    if index.is_symlink() or not index.is_file():
        raise RuntimeError("release source snapshot Git index is unsafe")
    git_directory = destination / ".git"
    git_directory.mkdir(mode=0o700)
    shutil.copyfile(index, git_directory / "index")
    (git_directory / "HEAD").write_text(
        "ref: refs/heads/release-snapshot\n", encoding="utf-8"
    )
    (git_directory / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = false\n",
        encoding="utf-8",
    )
    (git_directory / "objects/info").mkdir(parents=True)
    (git_directory / "objects/pack").mkdir()
    (git_directory / "refs/heads").mkdir(parents=True)


def create_snapshot(
    root: Path,
    destination: Path,
    *,
    manifest_output: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    destination = destination.parent.resolve(strict=True) / destination.name
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("release source snapshot destination must not exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o700)

    source_paths = _git_paths(root)
    evidence_paths = _conditional_evidence_paths(root)
    all_paths = sorted(source_paths | evidence_paths, key=str)
    if not all_paths:
        raise RuntimeError("release source snapshot is empty")

    rows: list[dict[str, Any]] = []
    total_size = 0
    for relative in all_paths:
        source = root.joinpath(*relative.parts)
        target = destination.joinpath(*relative.parts)
        size, digest, original_mode = _copy_file(source, target)
        total_size += size
        if total_size > MAX_SNAPSHOT_BYTES:
            raise RuntimeError(
                "minimal release source/evidence snapshot exceeds the 128 MiB limit"
            )
        rows.append(
            {
                "path": relative.as_posix(),
                "size_bytes": size,
                "sha256": digest,
                "source_kind": (
                    "conditional_private_evidence"
                    if relative.parts[:2] == ("artifacts", "private")
                    else "release_source_or_public_evidence"
                ),
                "executable": bool(original_mode & 0o111),
            }
        )
    _copy_minimal_git_index(root, destination)

    private_rows = [
        {key: row[key] for key in ("path", "size_bytes", "sha256")}
        for row in rows
        if row["source_kind"] == "conditional_private_evidence"
    ]
    binding_rows = [
        {key: row[key] for key in ("path", "size_bytes", "sha256")} for row in rows
    ]
    snapshot_binding = _binding(binding_rows)
    result = {
        "schema_version": SNAPSHOT_SCHEMA,
        "snapshot_sha256": snapshot_binding["sha256"],
        "source_snapshot_binding": snapshot_binding,
        # Paths stay only in the temporary manifest.  Published receipts expose
        # this aggregate, which binds every private byte without naming it.
        "conditional_private_evidence_binding": _binding(private_rows),
        "files": rows,
    }
    internal_manifest = destination / ".release-source-snapshot.json"
    _write_manifest(internal_manifest, result)
    if manifest_output is not None:
        _write_manifest(manifest_output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    result = create_snapshot(
        args.repository_root,
        args.destination,
        manifest_output=args.manifest_output,
    )
    print(
        "release_source_snapshot_files="
        f"{result['source_snapshot_binding']['file_count']}"
    )
    print(f"release_source_snapshot_sha256={result['snapshot_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
