"""Fail-closed Git worktree lifecycle control for isolated child agents.

The manager deliberately implements only a narrow control plane:

* a worktree is created from the repository's exact, committed ``HEAD``;
* the source repository must be clean, including non-ignored untracked files;
* Git is invoked directly through a trusted absolute executable with a scrubbed
  environment and project hooks, fsmonitor, and external diff execution disabled;
* lifecycle records live in an explicitly supplied owner-only state directory;
* automatic removal is allowed only while the branch still names the baseline
  commit and both Git status and a no-follow content manifest prove that the
  worktree is unchanged.

The repository and its Git control files are trusted to the current OS user.
Another process running as that same user is outside this module's hostile
concurrency boundary: it could mutate user-owned config in the final instant
before Git reads it (or alter user files directly).  Control-file fingerprint
rechecks detect ordinary concurrent changes, while the repository-wide lock
serializes all cooperating Harness managers.

There is intentionally no merge, reset, clean, force-remove, or prune API.  A
changed or structurally suspicious worktree is preserved for a human to inspect.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import errno
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from typing import Any, Iterator, Mapping

import fcntl


WORKTREE_RECORD_SCHEMA = "agent_harness.worktree.v1"

_WORKTREE_ID = re.compile(r"^wt_[0-9a-f]{32}$")
_BRANCH_REF = re.compile(r"^refs/heads/agent-harness/worktrees/[0-9a-f]{32}$")
_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MAX_RECORD_BYTES = 64 * 1024
_MAX_CONTROL_FILE_BYTES = 4 * 1024 * 1024
_MAX_GIT_OUTPUT_BYTES = 16 * 1024 * 1024
_MAX_MANIFEST_ENTRIES = 250_000
_MAX_MANIFEST_FILE_BYTES = 256 * 1024 * 1024
_MAX_MANIFEST_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_SYMLINK_BYTES = 64 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_GIT_TIMEOUT_SECONDS = 180.0
_PHASES = frozenset({"creating", "active", "removing", "ref_preserved"})
_OPERATION_CONTROL: ContextVar[tuple[object | None, float | None]] = ContextVar(
    "agent_harness_worktree_operation_control",
    default=(None, None),
)


class WorktreeError(RuntimeError):
    """Raised when worktree state or repository state cannot be trusted.

    ``worktree_id`` is populated when creation failed after its provisional
    durable record was committed.  Callers can then surface or reconcile the
    preserved artifact without parsing human-readable text.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "worktree_error",
        worktree_id: str | None = None,
    ) -> None:
        self.code = code
        self.worktree_id = worktree_id
        super().__init__(message)


@dataclass(frozen=True)
class WorktreeRecord:
    """Durable, content-free metadata for one managed Git worktree."""

    schema: str
    worktree_id: str
    phase: str
    repository: str
    common_git_dir: str
    worktree_path: str
    branch_ref: str
    baseline_commit: str
    baseline_manifest_sha256: str | None
    git_dir: str | None
    lock_reason: str
    created_at: str


@dataclass(frozen=True)
class WorktreeCleanupResult:
    """Outcome of a conservative cleanup request.

    ``removed`` is false for every mismatch.  ``reason`` is a stable,
    non-content-bearing code suitable for CLI/TUI presentation.
    """

    removed: bool
    reason: str
    record: WorktreeRecord


@contextmanager
def _operation_control(
    cancellation_token: object | None,
    deadline_monotonic: float | None,
) -> Iterator[None]:
    """Bind cooperative authority to one thread-local worktree operation."""

    if cancellation_token is not None and not callable(
        getattr(cancellation_token, "raise_if_cancelled", None)
    ):
        raise WorktreeError("worktree cancellation token is invalid")
    if deadline_monotonic is not None and (
        isinstance(deadline_monotonic, bool)
        or not isinstance(deadline_monotonic, (int, float))
        or not math.isfinite(float(deadline_monotonic))
    ):
        raise WorktreeError("worktree deadline is invalid")
    normalized_deadline = float(deadline_monotonic) if deadline_monotonic is not None else None
    marker = _OPERATION_CONTROL.set((cancellation_token, normalized_deadline))
    try:
        _check_operation_control()
        yield
    finally:
        _OPERATION_CONTROL.reset(marker)


def _check_operation_control() -> None:
    cancellation_token, deadline_monotonic = _OPERATION_CONTROL.get()
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()  # type: ignore[attr-defined]
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise WorktreeError(
            "worktree operation exceeded its deadline",
            code="worktree_deadline_exceeded",
        )


def _operation_pause(seconds: float) -> None:
    _check_operation_control()
    time.sleep(max(0.0, seconds))
    _check_operation_control()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise WorktreeError("secure no-follow directory access is unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise WorktreeError("secure no-follow file access is unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _canonical_private_path(value: os.PathLike[str] | str, *, label: str) -> Path:
    """Canonicalize trusted OS aliases while rejecting user-controlled links."""

    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise WorktreeError(f"{label} must be absolute")
    path = Path(os.path.abspath(os.fspath(raw)))
    if any(character in os.fspath(path) for character in ("\x00", "\n", "\r")):
        raise WorktreeError(f"{label} contains unsupported control characters")

    parts = path.parts[1:]
    current = Path(path.anchor)
    for index, component in enumerate(parts):
        candidate = current / component
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            return current.joinpath(*parts[index:])
        except OSError as exc:
            raise WorktreeError(f"{label} is unreadable") from exc
        if not stat.S_ISLNK(metadata.st_mode):
            current = candidate
            continue

        # macOS exposes immutable root-owned aliases such as /var -> /private/var.
        try:
            parent = current.stat()
        except OSError as exc:
            raise WorktreeError(f"{label} is unreadable") from exc
        if metadata.st_uid != 0 or parent.st_uid != 0 or stat.S_IMODE(parent.st_mode) & 0o022:
            raise WorktreeError(f"{label} contains an untrusted symlink")
        try:
            current = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorktreeError(f"{label} is unreadable") from exc
    return current


def _validate_ancestor(metadata: os.stat_result, *, label: str) -> None:
    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid not in {0, os.getuid()}:
        raise WorktreeError(f"{label} ancestor owner is invalid")
    if mode & 0o022:
        root_sticky = metadata.st_uid == 0 and bool(mode & stat.S_ISVTX) and bool(mode & 0o002)
        if not root_sticky:
            raise WorktreeError(f"{label} ancestor permissions are too broad")


@contextmanager
def _open_private_directory(
    path: Path,
    *,
    create: bool,
    label: str,
) -> Iterator[int]:
    """Open a 0700 owner directory without following path-component links."""

    if not path.is_absolute():
        raise WorktreeError(f"{label} must be absolute")
    try:
        descriptor = os.open(path.anchor, _directory_flags())
    except OSError as exc:
        raise WorktreeError(f"{label} is unreadable") from exc
    private_tail = False
    try:
        components = path.parts[1:]
        for index, component in enumerate(components):
            final = index == len(components) - 1
            created = False
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError as exc:
                if not create:
                    raise WorktreeError(f"{label} does not exist") from exc
                private_tail = True
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                except OSError as mkdir_exc:
                    raise WorktreeError(f"{label} cannot be created") from mkdir_exc
                try:
                    child = os.open(component, _directory_flags(), dir_fd=descriptor)
                except OSError as open_exc:
                    raise WorktreeError(f"{label} must be a real directory") from open_exc
            except OSError as exc:
                raise WorktreeError(
                    f"{label} must be a real directory; path contains a symlink"
                ) from exc

            try:
                metadata = os.fstat(child)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise WorktreeError(f"{label} must be a real directory")
                if created:
                    os.fchmod(child, 0o700)
                    metadata = os.fstat(child)
                if final or private_tail:
                    if metadata.st_uid != os.getuid():
                        raise WorktreeError(f"{label} owner is invalid")
                    if stat.S_IMODE(metadata.st_mode) != 0o700:
                        raise WorktreeError(f"{label} permissions must be 0700")
                else:
                    _validate_ancestor(metadata, label=label)
            except Exception:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _assert_private_file(parent_descriptor: int, name: str, *, label: str) -> bool:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise WorktreeError(f"{label} cannot be inspected") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise WorktreeError(f"{label} must be an owner-only regular file")
    return True


def _read_private_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    with _open_private_directory(path.parent, create=False, label=label) as parent:
        if not _assert_private_file(parent, path.name, label=label):
            raise WorktreeError(f"{label} does not exist")
        expected = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        try:
            descriptor = os.open(path.name, _file_flags(), dir_fd=parent)
        except OSError as exc:
            raise WorktreeError(f"{label} cannot be opened securely") from exc
        try:
            actual = os.fstat(descriptor)
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise WorktreeError(f"{label} changed while being opened")
            if actual.st_size > maximum:
                raise WorktreeError(f"{label} exceeds its size limit")
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > maximum:
                raise WorktreeError(f"{label} exceeds its size limit")
            final = os.fstat(descriptor)
            if (
                (final.st_dev, final.st_ino) != (actual.st_dev, actual.st_ino)
                or final.st_size != actual.st_size
                or final.st_mtime_ns != actual.st_mtime_ns
                or final.st_ctime_ns != actual.st_ctime_ns
            ):
                raise WorktreeError(f"{label} changed while being read")
            return payload
        finally:
            os.close(descriptor)


def _write_private_json(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_RECORD_BYTES:
        raise WorktreeError(f"{label} exceeds its size limit")
    with _open_private_directory(path.parent, create=True, label=label) as parent:
        if _assert_private_file(parent, path.name, label=label):
            pass
        temporary = f".{path.name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(temporary, flags, 0o600, dir_fd=parent)
        except OSError as exc:
            raise WorktreeError(f"{label} cannot be created securely") from exc
        try:
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise WorktreeError(f"{label} could not be written completely")
                offset += written
            os.fsync(descriptor)
        except Exception:
            try:
                os.unlink(temporary, dir_fd=parent)
            except OSError:
                pass
            raise
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
        except OSError as exc:
            try:
                os.unlink(temporary, dir_fd=parent)
            except OSError:
                pass
            raise WorktreeError(f"{label} could not be committed") from exc


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate object key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> Any:
        raise ValueError("non-finite number")

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorktreeError(f"{label} is not strict JSON") from exc
    if not isinstance(decoded, Mapping):
        raise WorktreeError(f"{label} must contain an object")
    return decoded


def _owned_file_fingerprint(paths: tuple[Path, ...]) -> str:
    """Hash optional user-owned control files without following links."""

    digest = sha256()
    for path in paths:
        encoded_path = os.fsencode(path)
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        try:
            expected = path.lstat()
        except FileNotFoundError:
            digest.update(b"M")
            continue
        except OSError as exc:
            raise WorktreeError("Git control file cannot be inspected") from exc
        if (
            stat.S_ISLNK(expected.st_mode)
            or not stat.S_ISREG(expected.st_mode)
            or expected.st_uid != os.getuid()
            or expected.st_nlink != 1
            or expected.st_mode & 0o022
            or expected.st_size > _MAX_CONTROL_FILE_BYTES
        ):
            raise WorktreeError("Git control file owner or type is unsafe")
        try:
            descriptor = os.open(path, _file_flags())
        except OSError as exc:
            raise WorktreeError("Git control file cannot be opened securely") from exc
        try:
            actual = os.fstat(descriptor)
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise WorktreeError("Git control file changed while opening")
            remaining = _MAX_CONTROL_FILE_BYTES + 1
            content = sha256()
            read_bytes = 0
            while remaining > 0:
                _check_operation_control()
                chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                content.update(chunk)
                read_bytes += len(chunk)
                remaining -= len(chunk)
            final = os.fstat(descriptor)
            if (
                read_bytes > _MAX_CONTROL_FILE_BYTES
                or (final.st_dev, final.st_ino) != (actual.st_dev, actual.st_ino)
                or final.st_size != actual.st_size
                or final.st_mtime_ns != actual.st_mtime_ns
                or final.st_ctime_ns != actual.st_ctime_ns
                or read_bytes != actual.st_size
            ):
                raise WorktreeError("Git control file changed while reading")
            digest.update(b"F")
            digest.update(content.digest())
            digest.update(actual.st_mode.to_bytes(8, "big"))
        finally:
            os.close(descriptor)
    return digest.hexdigest()


def _trusted_git_executable(value: os.PathLike[str] | str | None) -> Path:
    candidate = os.fspath(value) if value is not None else shutil.which("git")
    if not candidate:
        raise WorktreeError("Git is not installed")
    raw = Path(candidate).expanduser()
    if not raw.is_absolute():
        raise WorktreeError("Git executable must be an absolute path")
    try:
        path = raw.resolve(strict=True)
        metadata = path.lstat()
    except (OSError, RuntimeError) as exc:
        raise WorktreeError("Git executable cannot be inspected") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & 0o022
        or not metadata.st_mode & stat.S_IXUSR
    ):
        raise WorktreeError("Git executable is not a trusted root-owned binary")
    for ancestor in path.parents:
        try:
            ancestor_metadata = ancestor.lstat()
        except OSError as exc:
            raise WorktreeError("Git executable ancestry cannot be inspected") from exc
        if (
            stat.S_ISLNK(ancestor_metadata.st_mode)
            or not stat.S_ISDIR(ancestor_metadata.st_mode)
            or ancestor_metadata.st_uid != 0
            or ancestor_metadata.st_mode & 0o022
        ):
            raise WorktreeError("Git executable ancestry is not trusted")
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


class WorktreeManager:
    """Create and conservatively clean isolated Git worktrees.

    ``state_directory`` and ``worktree_home`` must be absolute, disjoint,
    owner-only locations outside the repository.  Callers cannot choose paths
    or refs; both are generated as opaque identifiers by :meth:`create`.
    """

    def __init__(
        self,
        repository: os.PathLike[str] | str,
        *,
        state_directory: os.PathLike[str] | str,
        worktree_home: os.PathLike[str] | str,
        git_executable: os.PathLike[str] | str | None = None,
        git_timeout_seconds: float = _GIT_TIMEOUT_SECONDS,
    ) -> None:
        try:
            repository_path = Path(repository).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorktreeError("repository does not exist") from exc
        if not repository_path.is_dir():
            raise WorktreeError("repository must be a directory")
        if any(character in os.fspath(repository_path) for character in ("\x00", "\n", "\r")):
            raise WorktreeError("repository path contains unsupported characters")
        repository_metadata = repository_path.lstat()
        if (
            stat.S_ISLNK(repository_metadata.st_mode)
            or not stat.S_ISDIR(repository_metadata.st_mode)
            or repository_metadata.st_uid != os.getuid()
            or repository_metadata.st_mode & 0o022
        ):
            raise WorktreeError("repository root owner, mode or type is unsafe")

        self.repository = repository_path
        self.state_directory = _canonical_private_path(
            state_directory,
            label="worktree state directory",
        )
        self.worktree_home = _canonical_private_path(
            worktree_home,
            label="worktree home",
        )
        if _paths_overlap(self.repository, self.state_directory):
            raise WorktreeError("worktree state directory must be outside the repository")
        if _paths_overlap(self.repository, self.worktree_home):
            raise WorktreeError("worktree home must be outside the repository")
        if _paths_overlap(self.state_directory, self.worktree_home):
            raise WorktreeError("worktree home and state directory must be disjoint")
        if not isinstance(git_timeout_seconds, (int, float)) or not (
            1.0 <= float(git_timeout_seconds) <= 3600.0
        ):
            raise WorktreeError("Git timeout must be between 1 and 3600 seconds")
        self.git_timeout_seconds = float(git_timeout_seconds)
        self.git_executable = _trusted_git_executable(git_executable)

        self.records_directory = self.state_directory / "records"
        self.hooks_directory = self.state_directory / "empty-hooks"
        self.runtime_directory = self.state_directory / "runtime"
        for path, label in (
            (self.state_directory, "worktree state directory"),
            (self.records_directory, "worktree record directory"),
            (self.hooks_directory, "empty hooks directory"),
            (self.runtime_directory, "Git runtime directory"),
            (self.worktree_home, "worktree home"),
        ):
            with _open_private_directory(path, create=True, label=label):
                pass

        root_metadata = self.repository.stat()
        self._repository_identity = (root_metadata.st_dev, root_metadata.st_ino)
        self._common_git_identity: tuple[int, int] | None = None
        self._git_env = {
            "HOME": os.fspath(self.runtime_directory),
            "XDG_CONFIG_HOME": os.fspath(self.runtime_directory),
            "TMPDIR": os.fspath(self.runtime_directory),
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
        }
        self._git_prefix = (
            os.fspath(self.git_executable),
            "--no-pager",
            "--literal-pathspecs",
            "-c",
            f"core.hooksPath={self.hooks_directory}",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.fsmonitorHookVersion=2",
            "-c",
            "core.sparseCheckout=false",
            "-c",
            "core.sparseCheckoutCone=false",
            "-c",
            "core.attributesFile=/dev/null",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.symlinks=true",
            "-c",
            "submodule.recurse=false",
            "-c",
            "checkout.workers=1",
            "-c",
            "diff.external=",
            "-c",
            "interactive.diffFilter=",
            "-c",
            "color.ui=false",
            "-c",
            "advice.detachedHead=false",
        )

        # Resolve and validate repository identity through the scrubbed Git path.
        top_level = self._git_text(
            "rev-parse",
            "--show-toplevel",
            cwd=self.repository,
            failure="directory is not a Git worktree",
        )
        try:
            reported_top = Path(top_level).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorktreeError("Git reported an invalid repository root") from exc
        if reported_top != self.repository:
            raise WorktreeError("repository must be the Git worktree root")
        common = self._git_text(
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
            cwd=self.repository,
            failure="Git common directory cannot be resolved",
        )
        try:
            self.common_git_dir = Path(common).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorktreeError("Git common directory cannot be trusted") from exc
        common_metadata = self.common_git_dir.lstat()
        if (
            stat.S_ISLNK(common_metadata.st_mode)
            or not stat.S_ISDIR(common_metadata.st_mode)
            or common_metadata.st_uid != os.getuid()
            or common_metadata.st_mode & 0o022
        ):
            raise WorktreeError("Git common directory owner or permissions are unsafe")
        self._common_git_identity = (common_metadata.st_dev, common_metadata.st_ino)
        repository_git = self._git_text(
            "rev-parse",
            "--path-format=absolute",
            "--git-dir",
            cwd=self.repository,
            failure="repository Git directory cannot be resolved",
        )
        try:
            self.repository_git_dir = Path(repository_git).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorktreeError("repository Git directory cannot be trusted") from exc
        if self.repository_git_dir != self.common_git_dir:
            self._validate_git_dir_descendant(self.repository_git_dir)
        # Managers built from different linked worktrees can have different
        # SessionStore roots.  Key the mutation lock by the canonical common
        # Git directory in the explicitly shared worktree home instead.
        self.locks_directory = self.worktree_home / ".locks"
        with _open_private_directory(
            self.locks_directory,
            create=True,
            label="worktree lock directory",
        ):
            pass
        lock_key = sha256(os.fsencode(self.common_git_dir)).hexdigest()
        self.lock_path = self.locks_directory / f"repository-{lock_key}.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with _open_private_directory(
            self.locks_directory,
            create=False,
            label="worktree lock directory",
        ) as parent:
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(
                    self.lock_path.name,
                    flags | os.O_EXCL,
                    0o600,
                    dir_fd=parent,
                )
                created = True
            except FileExistsError:
                created = False
                try:
                    descriptor = os.open(
                        self.lock_path.name,
                        os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=parent,
                    )
                except OSError as exc:
                    raise WorktreeError("worktree state lock cannot be opened") from exc
            except OSError as exc:
                raise WorktreeError("worktree state lock cannot be opened") from exc
            try:
                if created:
                    os.fchmod(descriptor, 0o600)
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                ):
                    raise WorktreeError("worktree state lock cannot be trusted")
                while True:
                    _check_operation_control()
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        _operation_pause(0.05)
                    except OSError as exc:
                        if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                            raise WorktreeError("worktree state lock cannot be acquired") from exc
                        _operation_pause(0.05)
                yield
            finally:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)

    def _assert_repository_identity(self) -> None:
        try:
            metadata = self.repository.lstat()
        except OSError as exc:
            raise WorktreeError("repository is no longer available") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != self._repository_identity
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o022
        ):
            raise WorktreeError("repository identity changed")
        if self._common_git_identity is not None:
            try:
                common = self.common_git_dir.lstat()
            except OSError as exc:
                raise WorktreeError("Git common directory is no longer available") from exc
            if (
                stat.S_ISLNK(common.st_mode)
                or not stat.S_ISDIR(common.st_mode)
                or (common.st_dev, common.st_ino) != self._common_git_identity
                or common.st_uid != os.getuid()
                or common.st_mode & 0o022
            ):
                raise WorktreeError("Git common directory identity changed")

    def _git(
        self,
        *arguments: str,
        cwd: Path,
        allowed_returncodes: frozenset[int] = frozenset({0}),
        failure: str = "Git command failed",
    ) -> tuple[int, bytes, bytes]:
        _check_operation_control()
        self._assert_repository_identity()
        argv = (
            *self._git_prefix,
            "-c",
            f"core.worktree={cwd}",
            "-c",
            "core.bare=false",
            *arguments,
        )
        command_environment = dict(self._git_env)
        command_environment["GIT_WORK_TREE"] = os.fspath(cwd)
        with tempfile.TemporaryFile(mode="w+b", dir=self.runtime_directory) as stdout_file:
            with tempfile.TemporaryFile(mode="w+b", dir=self.runtime_directory) as stderr_file:
                try:
                    process = subprocess.Popen(
                        argv,
                        cwd=os.fspath(cwd),
                        env=command_environment,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        shell=False,
                        close_fds=True,
                        start_new_session=True,
                    )
                except OSError as exc:
                    raise WorktreeError(failure) from exc
                try:
                    command_deadline = time.monotonic() + self.git_timeout_seconds
                    while True:
                        _check_operation_control()
                        remaining = command_deadline - time.monotonic()
                        if remaining <= 0:
                            raise WorktreeError(
                                "Git command exceeded its deadline",
                                code="git_deadline_exceeded",
                            )
                        try:
                            returncode = process.wait(timeout=min(0.05, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            continue
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except (OSError, ProcessLookupError):
                        pass
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except (OSError, ProcessLookupError):
                            pass
                        process.wait()
                    raise
                _check_operation_control()
                stdout_file.seek(0, os.SEEK_END)
                stderr_file.seek(0, os.SEEK_END)
                if (
                    stdout_file.tell() > _MAX_GIT_OUTPUT_BYTES
                    or stderr_file.tell() > _MAX_GIT_OUTPUT_BYTES
                ):
                    raise WorktreeError("Git command output exceeded its size limit")
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout = stdout_file.read(_MAX_GIT_OUTPUT_BYTES + 1)
                stderr = stderr_file.read(_MAX_GIT_OUTPUT_BYTES + 1)
                if returncode not in allowed_returncodes:
                    raise WorktreeError(failure)
                return returncode, stdout, stderr

    def _git_text(
        self,
        *arguments: str,
        cwd: Path,
        allowed_returncodes: frozenset[int] = frozenset({0}),
        failure: str,
    ) -> str:
        _returncode, stdout, _stderr = self._git(
            *arguments,
            cwd=cwd,
            allowed_returncodes=allowed_returncodes,
            failure=failure,
        )
        try:
            value = stdout.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise WorktreeError(failure) from exc
        if not value or "\x00" in value or "\n" in value or "\r" in value:
            raise WorktreeError(failure)
        return value

    def _head_commit(self) -> str:
        value = self._git_text(
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            cwd=self.repository,
            failure="repository has no committed HEAD",
        )
        if not _COMMIT.fullmatch(value):
            raise WorktreeError("repository HEAD is not a valid commit object id")
        return value

    def _control_file_fingerprint(self) -> str:
        return _owned_file_fingerprint(
            (
                self.common_git_dir / "config",
                self.common_git_dir / "info" / "attributes",
                self.repository_git_dir / "config.worktree",
            )
        )

    def _assert_control_files_unchanged(self, expected: str) -> None:
        if self._control_file_fingerprint() != expected:
            raise WorktreeError("repository Git control files changed during preparation")

    def _reject_executable_filters(self) -> None:
        returncode, stdout, _stderr = self._git(
            "config",
            "--local",
            "--name-only",
            "--list",
            cwd=self.repository,
            allowed_returncodes=frozenset({0}),
            failure="repository Git configuration cannot be inspected",
        )
        if returncode != 0:
            raise WorktreeError("repository Git configuration cannot be inspected")
        try:
            names = stdout.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise WorktreeError("repository Git configuration is not valid UTF-8") from exc
        for raw_name in names:
            name = raw_name.casefold()
            if name.startswith("include.") or name.startswith("includeif."):
                raise WorktreeError("repository Git configuration contains includes")
            if name.startswith("filter.") and name.rsplit(".", 1)[-1] in {
                "clean",
                "smudge",
                "process",
            }:
                raise WorktreeError("repository Git configuration contains executable filters")
            if name in {
                "core.worktree",
                "core.sparsecheckout",
                "core.sparsecheckoutcone",
                "core.attributesfile",
                "submodule.recurse",
                "checkout.workers",
                "extensions.worktreeconfig",
            }:
                raise WorktreeError(
                    "repository Git configuration contains an unsupported checkout control"
                )

    def _status_is_clean(self, cwd: Path) -> bool:
        _returncode, stdout, _stderr = self._git(
            "status",
            "--porcelain=v2",
            "--untracked-files=all",
            "--ignore-submodules=none",
            cwd=cwd,
            failure="Git status could not be established",
        )
        return stdout == b""

    def _record_path(self, worktree_id: str) -> Path:
        if not _WORKTREE_ID.fullmatch(worktree_id):
            raise WorktreeError("worktree id is invalid")
        return self.records_directory / f"{worktree_id}.json"

    def _validate_record(self, raw: Mapping[str, Any], *, path: Path) -> WorktreeRecord:
        expected_keys = {
            "schema",
            "worktree_id",
            "phase",
            "repository",
            "common_git_dir",
            "worktree_path",
            "branch_ref",
            "baseline_commit",
            "baseline_manifest_sha256",
            "git_dir",
            "lock_reason",
            "created_at",
        }
        if set(raw) != expected_keys:
            raise WorktreeError("worktree record fields are invalid")
        if not all(
            isinstance(raw[key], str)
            for key in expected_keys - {"baseline_manifest_sha256", "git_dir"}
        ):
            raise WorktreeError("worktree record value types are invalid")
        manifest = raw["baseline_manifest_sha256"]
        git_dir = raw["git_dir"]
        if manifest is not None and not isinstance(manifest, str):
            raise WorktreeError("worktree record manifest is invalid")
        if git_dir is not None and not isinstance(git_dir, str):
            raise WorktreeError("worktree record Git directory is invalid")

        record = WorktreeRecord(**dict(raw))
        if record.schema != WORKTREE_RECORD_SCHEMA:
            raise WorktreeError("worktree record schema is unsupported")
        if not _WORKTREE_ID.fullmatch(record.worktree_id):
            raise WorktreeError("worktree record id is invalid")
        if path.name != f"{record.worktree_id}.json":
            raise WorktreeError("worktree record filename does not match its id")
        if record.phase not in _PHASES:
            raise WorktreeError("worktree record phase is invalid")
        if record.repository != os.fspath(self.repository):
            raise WorktreeError("worktree record repository does not match")
        if record.common_git_dir != os.fspath(self.common_git_dir):
            raise WorktreeError("worktree record Git directory does not match")
        if record.worktree_path != os.fspath(self.worktree_home / record.worktree_id):
            raise WorktreeError("worktree record path escapes the managed home")
        if not _BRANCH_REF.fullmatch(record.branch_ref):
            raise WorktreeError("worktree record branch ref is invalid")
        if not _COMMIT.fullmatch(record.baseline_commit):
            raise WorktreeError("worktree record baseline commit is invalid")
        if len(record.created_at) > 64 or not record.created_at.endswith("Z"):
            raise WorktreeError("worktree record timestamp is invalid")
        expected_lock_reason = f"agent-harness-v1:{record.worktree_id}:{record.baseline_commit}"
        if record.lock_reason != expected_lock_reason:
            raise WorktreeError("worktree record lock reason is invalid")
        if record.phase == "creating":
            if record.baseline_manifest_sha256 is not None or record.git_dir is not None:
                raise WorktreeError("creating worktree record has impossible fields")
        else:
            if not isinstance(record.baseline_manifest_sha256, str) or not _DIGEST.fullmatch(
                record.baseline_manifest_sha256
            ):
                raise WorktreeError("worktree record manifest digest is invalid")
            if not isinstance(record.git_dir, str):
                raise WorktreeError("worktree record Git directory is missing")
            try:
                Path(record.git_dir).relative_to(self.common_git_dir / "worktrees")
            except ValueError as exc:
                raise WorktreeError("worktree record Git directory escapes common state") from exc
        return record

    def _load_record(self, worktree_id: str) -> WorktreeRecord:
        path = self._record_path(worktree_id)
        payload = _read_private_bytes(
            path,
            maximum=_MAX_RECORD_BYTES,
            label="worktree record",
        )
        return self._validate_record(_strict_json(payload, label="worktree record"), path=path)

    def _store_record(self, record: WorktreeRecord) -> None:
        path = self._record_path(record.worktree_id)
        self._validate_record(asdict(record), path=path)
        _write_private_json(path, asdict(record), label="worktree record")

    def _delete_record(self, record: WorktreeRecord) -> None:
        path = self._record_path(record.worktree_id)
        with _open_private_directory(
            self.records_directory,
            create=False,
            label="worktree record directory",
        ) as parent:
            if not _assert_private_file(parent, path.name, label="worktree record"):
                raise WorktreeError("worktree record disappeared")
            try:
                os.unlink(path.name, dir_fd=parent)
                os.fsync(parent)
            except OSError as exc:
                raise WorktreeError("worktree record could not be removed") from exc

    def list_records(self, *, limit: int | None = None) -> tuple[WorktreeRecord, ...]:
        """Return all durable records, failing on any unexpected state entry."""

        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 4096
        ):
            raise WorktreeError("worktree record list limit is invalid")

        with self._locked():
            with _open_private_directory(
                self.records_directory,
                create=False,
                label="worktree record directory",
            ) as descriptor:
                names = sorted(os.listdir(descriptor))
            if any(re.fullmatch(r"wt_[0-9a-f]{32}\.json", name) is None for name in names):
                raise WorktreeError("worktree record directory contains an unknown entry")
            if limit is not None and len(names) > limit:
                raise WorktreeError("worktree record list exceeds its bounded limit")
            records: list[WorktreeRecord] = []
            for name in names:
                records.append(self._load_record(name.removesuffix(".json")))
            return tuple(records)

    def get(self, worktree_id: str) -> WorktreeRecord:
        """Load and validate one durable worktree record."""

        with self._locked():
            return self._load_record(worktree_id)

    def _validate_git_mapping(self, record: WorktreeRecord) -> None:
        if record.git_dir is None:
            raise WorktreeError("worktree Git mapping is incomplete")
        root = Path(record.worktree_path)
        try:
            root_metadata = root.lstat()
        except OSError as exc:
            raise WorktreeError("worktree root is unavailable") from exc
        if (
            stat.S_ISLNK(root_metadata.st_mode)
            or not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.getuid()
            or stat.S_IMODE(root_metadata.st_mode) != 0o700
        ):
            raise WorktreeError("worktree root owner, mode or type is unsafe")

        try:
            root_descriptor = os.open(root, _directory_flags())
        except OSError as exc:
            raise WorktreeError("worktree root cannot be opened securely") from exc
        try:
            try:
                expected = os.stat(".git", dir_fd=root_descriptor, follow_symlinks=False)
            except OSError as exc:
                raise WorktreeError("worktree .git mapping is unavailable") from exc
            if (
                stat.S_ISLNK(expected.st_mode)
                or not stat.S_ISREG(expected.st_mode)
                or expected.st_uid != os.getuid()
                or expected.st_nlink != 1
                or stat.S_IMODE(expected.st_mode) != 0o600
                or expected.st_size > 16 * 1024
            ):
                raise WorktreeError("worktree .git mapping owner, mode or type is unsafe")
            try:
                descriptor = os.open(".git", _file_flags(), dir_fd=root_descriptor)
            except OSError as exc:
                raise WorktreeError("worktree .git mapping cannot be opened securely") from exc
            try:
                actual = os.fstat(descriptor)
                if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                    raise WorktreeError("worktree .git mapping changed while opening")
                payload = os.read(descriptor, 16 * 1024 + 1)
                final = os.fstat(descriptor)
                if (
                    len(payload) > 16 * 1024
                    or (final.st_dev, final.st_ino) != (actual.st_dev, actual.st_ino)
                    or final.st_size != actual.st_size
                    or final.st_mtime_ns != actual.st_mtime_ns
                    or final.st_ctime_ns != actual.st_ctime_ns
                ):
                    raise WorktreeError("worktree .git mapping changed while reading")
            finally:
                os.close(descriptor)
        finally:
            os.close(root_descriptor)

        git_dir = Path(record.git_dir)
        expected_payload = f"gitdir: {git_dir}\n".encode("utf-8")
        if payload != expected_payload:
            raise WorktreeError("worktree .git mapping target does not match its record")
        try:
            relative = git_dir.relative_to(self.common_git_dir / "worktrees")
        except ValueError as exc:
            raise WorktreeError("worktree Git directory escapes common state") from exc
        if len(relative.parts) != 1:
            raise WorktreeError("worktree Git directory nesting is invalid")
        self._validate_git_dir_descendant(git_dir)
        self._validate_admin_mappings(record)

    def _read_admin_file(
        self,
        git_dir: Path,
        name: str,
        *,
        prepare: bool,
    ) -> bytes:
        try:
            descriptor = os.open(git_dir, _directory_flags())
        except OSError as exc:
            raise WorktreeError("worktree Git directory cannot be opened securely") from exc
        try:
            try:
                expected = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise WorktreeError(f"worktree Git {name} mapping is unavailable") from exc
            if (
                stat.S_ISLNK(expected.st_mode)
                or not stat.S_ISREG(expected.st_mode)
                or expected.st_uid != os.getuid()
                or expected.st_nlink != 1
                or expected.st_size > 16 * 1024
            ):
                raise WorktreeError(f"worktree Git {name} mapping cannot be trusted")
            flags = os.O_RDWR | os.O_NOFOLLOW if prepare else _file_flags()
            try:
                child = os.open(name, flags, dir_fd=descriptor)
            except OSError as exc:
                raise WorktreeError(
                    f"worktree Git {name} mapping cannot be opened securely"
                ) from exc
            try:
                actual = os.fstat(child)
                if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                    raise WorktreeError(f"worktree Git {name} mapping changed while opening")
                if prepare:
                    os.fchmod(child, 0o600)
                    os.fsync(child)
                    actual = os.fstat(child)
                if stat.S_IMODE(actual.st_mode) != 0o600:
                    raise WorktreeError(f"worktree Git {name} mapping permissions must be 0600")
                payload = os.read(child, 16 * 1024 + 1)
                final = os.fstat(child)
                if (
                    len(payload) > 16 * 1024
                    or (final.st_dev, final.st_ino) != (actual.st_dev, actual.st_ino)
                    or final.st_size != actual.st_size
                    or final.st_mtime_ns != actual.st_mtime_ns
                    or final.st_ctime_ns != actual.st_ctime_ns
                ):
                    raise WorktreeError(f"worktree Git {name} mapping changed while reading")
                return payload
            finally:
                os.close(child)
        finally:
            os.close(descriptor)

    def _validate_admin_mappings(
        self,
        record: WorktreeRecord,
        *,
        prepare: bool = False,
    ) -> None:
        if record.git_dir is None:
            raise WorktreeError("worktree Git directory is incomplete")
        git_dir = Path(record.git_dir)
        worktree_mapping = self._read_admin_file(
            git_dir,
            "gitdir",
            prepare=prepare,
        )
        expected_worktree_mapping = f"{Path(record.worktree_path) / '.git'}\n".encode("utf-8")
        if worktree_mapping != expected_worktree_mapping:
            raise WorktreeError("worktree Git gitdir backlink does not match")
        common_mapping = self._read_admin_file(
            git_dir,
            "commondir",
            prepare=prepare,
        )
        expected_relative = os.path.relpath(self.common_git_dir, git_dir)
        if common_mapping != f"{expected_relative}\n".encode("utf-8"):
            raise WorktreeError("worktree Git commondir mapping does not match")
        try:
            resolved_common = (git_dir / expected_relative).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorktreeError("worktree Git commondir cannot be resolved") from exc
        if resolved_common != self.common_git_dir:
            raise WorktreeError("worktree Git commondir escapes common state")

    def _validate_lock_reason(
        self,
        record: WorktreeRecord,
        *,
        prepare: bool = False,
    ) -> None:
        if record.git_dir is None:
            raise WorktreeError("worktree Git directory is incomplete")
        payload = self._read_admin_file(
            Path(record.git_dir),
            "locked",
            prepare=prepare,
        )
        if payload != f"{record.lock_reason}\n".encode("utf-8"):
            raise WorktreeError("worktree Git lock reason does not match its record")

    def _ensure_managed_lock(self, record: WorktreeRecord) -> None:
        """Re-establish only a missing lock; never replace a foreign lock."""

        if record.git_dir is None:
            raise WorktreeError("worktree Git directory is incomplete")
        git_dir = Path(record.git_dir)
        try:
            descriptor = os.open(git_dir, _directory_flags())
        except OSError as exc:
            raise WorktreeError("worktree Git lock state cannot be inspected") from exc
        try:
            metadata = os.stat(
                "locked",
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            os.close(descriptor)
            self._git(
                "worktree",
                "lock",
                "--reason",
                record.lock_reason,
                record.worktree_path,
                cwd=self.repository,
                failure="Git could not restore the managed worktree lock",
            )
            self._validate_lock_reason(record, prepare=True)
            return
        except OSError as exc:
            os.close(descriptor)
            raise WorktreeError("worktree Git lock state cannot be inspected") from exc
        os.close(descriptor)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise WorktreeError("worktree Git lock state cannot be trusted")
        self._validate_lock_reason(record)

    def _validate_git_dir_descendant(self, git_dir: Path) -> None:
        base = self.common_git_dir / "worktrees"
        try:
            relative = git_dir.relative_to(base)
        except ValueError as exc:
            raise WorktreeError("worktree Git directory escapes common state") from exc
        try:
            descriptor = os.open(base, _directory_flags())
        except OSError as exc:
            raise WorktreeError("Git worktree metadata directory is unsafe") from exc
        try:
            base_metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(base_metadata.st_mode)
                or base_metadata.st_uid != os.getuid()
                or base_metadata.st_mode & 0o022
            ):
                raise WorktreeError("Git worktree metadata root owner or mode is unsafe")
            for component in relative.parts:
                try:
                    child = os.open(component, _directory_flags(), dir_fd=descriptor)
                except OSError as exc:
                    raise WorktreeError("worktree Git directory contains a symlink") from exc
                metadata = os.fstat(child)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o022
                ):
                    os.close(child)
                    raise WorktreeError("worktree Git directory owner or mode is unsafe")
                os.close(descriptor)
                descriptor = child
        finally:
            os.close(descriptor)

    def _prepare_git_mapping(self, record: WorktreeRecord) -> str:
        # Make the worktree root and mapping private before recording them.
        worktree_path = Path(record.worktree_path)
        try:
            metadata = worktree_path.lstat()
        except OSError as exc:
            raise WorktreeError("Git did not create the worktree root") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
        ):
            raise WorktreeError("created worktree root cannot be trusted")
        os.chmod(worktree_path, 0o700, follow_symlinks=False)
        try:
            root = os.open(worktree_path, _directory_flags())
        except OSError as exc:
            raise WorktreeError("created worktree root cannot be opened securely") from exc
        try:
            try:
                mapping = os.stat(".git", dir_fd=root, follow_symlinks=False)
            except OSError as exc:
                raise WorktreeError("Git did not create a .git mapping") from exc
            if (
                stat.S_ISLNK(mapping.st_mode)
                or not stat.S_ISREG(mapping.st_mode)
                or mapping.st_uid != os.getuid()
                or mapping.st_nlink != 1
            ):
                raise WorktreeError("created worktree .git mapping cannot be trusted")
            try:
                descriptor = os.open(".git", os.O_RDWR | os.O_NOFOLLOW, dir_fd=root)
            except OSError as exc:
                raise WorktreeError("created worktree .git mapping cannot be opened") from exc
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino) != (mapping.st_dev, mapping.st_ino):
                    raise WorktreeError("created worktree .git mapping changed")
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            os.close(root)

        git_dir_value = self._git_text(
            "rev-parse",
            "--path-format=absolute",
            "--git-dir",
            cwd=worktree_path,
            failure="created worktree Git directory cannot be resolved",
        )
        try:
            git_dir = Path(git_dir_value).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorktreeError("created worktree Git directory cannot be trusted") from exc
        self._validate_git_dir_descendant(git_dir)
        prepared = replace(record, git_dir=os.fspath(git_dir))
        self._validate_admin_mappings(prepared, prepare=True)
        return os.fspath(git_dir)

    def _manifest(self, root: Path) -> str:
        """Hash a bounded tree without following symlinks, excluding root .git."""

        _check_operation_control()
        digest = sha256()
        counters = {"entries": 0, "bytes": 0}
        try:
            descriptor = os.open(root, _directory_flags())
        except OSError as exc:
            raise WorktreeError("worktree cannot be opened for manifesting") from exc
        try:
            self._manifest_directory(
                descriptor,
                relative=(),
                digest=digest,
                counters=counters,
            )
        finally:
            os.close(descriptor)
        return digest.hexdigest()

    def _manifest_directory(
        self,
        descriptor: int,
        *,
        relative: tuple[bytes, ...],
        digest: Any,
        counters: dict[str, int],
    ) -> None:
        try:
            names = sorted(os.listdir(descriptor), key=os.fsencode)
        except OSError as exc:
            raise WorktreeError("worktree directory cannot be enumerated securely") from exc
        for name in names:
            _check_operation_control()
            if not relative and name == ".git":
                continue
            encoded_name = os.fsencode(name)
            if not encoded_name or b"/" in encoded_name or b"\x00" in encoded_name:
                raise WorktreeError("worktree contains an invalid entry name")
            counters["entries"] += 1
            if counters["entries"] > _MAX_MANIFEST_ENTRIES:
                raise WorktreeError("worktree manifest entry limit exceeded")
            path_parts = (*relative, encoded_name)
            path_bytes = b"/".join(path_parts)
            try:
                expected = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise WorktreeError("worktree entry cannot be inspected securely") from exc
            if expected.st_uid != os.getuid():
                raise WorktreeError("worktree entry owner is invalid")

            digest.update(len(path_bytes).to_bytes(8, "big"))
            digest.update(path_bytes)
            digest.update(stat.S_IMODE(expected.st_mode).to_bytes(4, "big"))
            if stat.S_ISDIR(expected.st_mode):
                digest.update(b"D")
                try:
                    child = os.open(name, _directory_flags(), dir_fd=descriptor)
                except OSError as exc:
                    raise WorktreeError("worktree directory contains a symlink race") from exc
                try:
                    actual = os.fstat(child)
                    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                        raise WorktreeError("worktree directory changed while opening")
                    self._manifest_directory(
                        child,
                        relative=path_parts,
                        digest=digest,
                        counters=counters,
                    )
                finally:
                    os.close(child)
            elif stat.S_ISREG(expected.st_mode):
                if expected.st_nlink != 1:
                    raise WorktreeError("worktree manifest refuses hard-linked files")
                if expected.st_size > _MAX_MANIFEST_FILE_BYTES:
                    raise WorktreeError("worktree manifest file limit exceeded")
                counters["bytes"] += expected.st_size
                if counters["bytes"] > _MAX_MANIFEST_TOTAL_BYTES:
                    raise WorktreeError("worktree manifest byte limit exceeded")
                digest.update(b"F")
                digest.update(expected.st_size.to_bytes(8, "big"))
                try:
                    child = os.open(name, _file_flags(), dir_fd=descriptor)
                except OSError as exc:
                    raise WorktreeError("worktree file contains a symlink race") from exc
                try:
                    actual = os.fstat(child)
                    if (actual.st_dev, actual.st_ino) != (
                        expected.st_dev,
                        expected.st_ino,
                    ) or actual.st_nlink != 1:
                        raise WorktreeError("worktree file changed while opening")
                    content_digest = sha256()
                    read_bytes = 0
                    while True:
                        _check_operation_control()
                        chunk = os.read(child, _READ_CHUNK_BYTES)
                        if not chunk:
                            break
                        read_bytes += len(chunk)
                        if read_bytes > _MAX_MANIFEST_FILE_BYTES:
                            raise WorktreeError("worktree file changed beyond its size limit")
                        content_digest.update(chunk)
                    final = os.fstat(child)
                    if (
                        (final.st_dev, final.st_ino) != (actual.st_dev, actual.st_ino)
                        or final.st_size != actual.st_size
                        or final.st_mtime_ns != actual.st_mtime_ns
                        or final.st_ctime_ns != actual.st_ctime_ns
                        or read_bytes != actual.st_size
                    ):
                        raise WorktreeError("worktree file changed while being read")
                    digest.update(content_digest.digest())
                finally:
                    os.close(child)
            elif stat.S_ISLNK(expected.st_mode):
                digest.update(b"L")
                try:
                    target = os.readlink(name, dir_fd=descriptor)
                except OSError as exc:
                    raise WorktreeError("worktree symlink cannot be read safely") from exc
                target_bytes = os.fsencode(target)
                if len(target_bytes) > _MAX_SYMLINK_BYTES:
                    raise WorktreeError("worktree symlink target is too large")
                final = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (
                    (final.st_dev, final.st_ino) != (expected.st_dev, expected.st_ino)
                    or final.st_mtime_ns != expected.st_mtime_ns
                    or final.st_ctime_ns != expected.st_ctime_ns
                ):
                    raise WorktreeError("worktree symlink changed while being read")
                digest.update(len(target_bytes).to_bytes(8, "big"))
                digest.update(target_bytes)
            else:
                raise WorktreeError("worktree manifest refuses special files")

    def create(
        self,
        *,
        cancellation_token: object | None = None,
        deadline_monotonic: float | None = None,
    ) -> WorktreeRecord:
        """Create under an optional cooperative cancellation/deadline fence."""

        with _operation_control(cancellation_token, deadline_monotonic):
            return self._create()

    def _create(self) -> WorktreeRecord:
        """Create a private worktree from the exact clean committed ``HEAD``.

        A provisional durable record is written before Git mutates refs.  If a
        later step fails, that record and any partial Git state are intentionally
        preserved rather than guessed-cleaned.
        """

        with self._locked():
            control_fingerprint = self._control_file_fingerprint()
            self._reject_executable_filters()
            self._assert_control_files_unchanged(control_fingerprint)
            baseline = self._head_commit()
            if not self._status_is_clean(self.repository):
                raise WorktreeError(
                    "repository must be clean, including non-ignored untracked files"
                )
            if self._head_commit() != baseline:
                raise WorktreeError("repository HEAD changed during worktree preparation")
            self._assert_control_files_unchanged(control_fingerprint)

            worktree_id = f"wt_{secrets.token_hex(16)}"
            branch_token = secrets.token_hex(16)
            branch_ref = f"refs/heads/agent-harness/worktrees/{branch_token}"
            branch_short = branch_ref.removeprefix("refs/heads/")
            worktree_path = self.worktree_home / worktree_id
            if worktree_path.exists() or worktree_path.is_symlink():
                raise WorktreeError("opaque worktree path unexpectedly exists")
            provisional = WorktreeRecord(
                schema=WORKTREE_RECORD_SCHEMA,
                worktree_id=worktree_id,
                phase="creating",
                repository=os.fspath(self.repository),
                common_git_dir=os.fspath(self.common_git_dir),
                worktree_path=os.fspath(worktree_path),
                branch_ref=branch_ref,
                baseline_commit=baseline,
                baseline_manifest_sha256=None,
                git_dir=None,
                lock_reason=f"agent-harness-v1:{worktree_id}:{baseline}",
                created_at=_utc_now(),
            )
            self._store_record(provisional)
            try:
                self._git(
                    "worktree",
                    "add",
                    "--no-checkout",
                    "-b",
                    branch_short,
                    os.fspath(worktree_path),
                    baseline,
                    cwd=self.repository,
                    failure="Git could not create the isolated worktree",
                )
                git_dir = self._prepare_git_mapping(provisional)
                # This checkout is deliberately separate from worktree creation.
                # Local config includes and executable clean/smudge/process filters
                # were rejected above; global/system config is disabled, project
                # hooks and fsmonitor are overridden, and external diff is disabled.
                self._assert_control_files_unchanged(control_fingerprint)
                self._reject_executable_filters()
                self._assert_control_files_unchanged(control_fingerprint)
                self._git(
                    "checkout",
                    "--no-recurse-submodules",
                    "--ignore-skip-worktree-bits",
                    cwd=worktree_path,
                    failure="Git could not populate the isolated worktree safely",
                )
                self._assert_control_files_unchanged(control_fingerprint)
                if not self._status_is_clean(worktree_path):
                    raise WorktreeError("new worktree is not baseline-clean")
                manifest = self._manifest(worktree_path)
                resolved_branch = self._git_text(
                    "rev-parse",
                    "--verify",
                    f"{branch_ref}^{{commit}}",
                    cwd=self.repository,
                    failure="new worktree branch cannot be verified",
                )
                if resolved_branch != baseline:
                    raise WorktreeError("new worktree branch does not match its baseline")
                active = replace(
                    provisional,
                    phase="active",
                    baseline_manifest_sha256=manifest,
                    git_dir=git_dir,
                )
                self._git(
                    "worktree",
                    "lock",
                    "--reason",
                    active.lock_reason,
                    os.fspath(worktree_path),
                    cwd=self.repository,
                    failure="Git could not lock the managed worktree",
                )
                self._validate_git_mapping(active)
                self._validate_lock_reason(active, prepare=True)
                self._store_record(active)
                return active
            except WorktreeError as exc:
                if exc.worktree_id == worktree_id:
                    raise
                raise WorktreeError(
                    str(exc),
                    code="create_incomplete",
                    worktree_id=worktree_id,
                ) from exc
            except Exception as exc:
                raise WorktreeError(
                    "worktree creation failed after durable allocation",
                    code="create_incomplete",
                    worktree_id=worktree_id,
                ) from exc

    def remove_if_pristine(
        self,
        worktree_id: str,
        *,
        cancellation_token: object | None = None,
        deadline_monotonic: float | None = None,
    ) -> WorktreeCleanupResult:
        """Remove only under proof, with optional cooperative operation bounds."""

        with _operation_control(cancellation_token, deadline_monotonic):
            return self._remove_if_pristine(worktree_id)

    def _remove_if_pristine(self, worktree_id: str) -> WorktreeCleanupResult:
        """Remove only a proven unchanged worktree and CAS-delete its branch.

        Every precondition mismatch returns a preservation result.  Structural
        trust failures raise :class:`WorktreeError` and likewise leave data in
        place.  Git's normal worktree removal is used without ``--force``.
        """

        with self._locked():
            record = self._load_record(worktree_id)
            if record.phase not in {"active", "removing"}:
                return WorktreeCleanupResult(False, "record_not_active", record)
            root = Path(record.worktree_path)
            if not os.path.lexists(root):
                if record.phase != "removing":
                    return WorktreeCleanupResult(
                        False,
                        "worktree_missing_or_replaced",
                        record,
                    )
                # A crash can land after Git removed the worktree but before
                # the branch CAS and record deletion.  Finalize only when Git's
                # per-worktree administrative directory is also gone.
                if record.git_dir is None or os.path.lexists(record.git_dir):
                    return WorktreeCleanupResult(
                        False,
                        "worktree_metadata_preserved",
                        record,
                    )
                return self._finalize_removed_worktree(record)
            if root.is_symlink():
                return WorktreeCleanupResult(False, "worktree_missing_or_replaced", record)
            self._validate_git_mapping(record)
            self._ensure_managed_lock(record)
            if not self._status_is_clean(root):
                return WorktreeCleanupResult(False, "git_status_changed", record)
            if self._manifest(root) != record.baseline_manifest_sha256:
                return WorktreeCleanupResult(False, "content_manifest_changed", record)
            ref_returncode, _stdout, _stderr = self._git(
                "show-ref",
                "--verify",
                "--quiet",
                record.branch_ref,
                cwd=self.repository,
                allowed_returncodes=frozenset({0, 1}),
                failure="managed worktree ref cannot be inspected",
            )
            if ref_returncode == 1:
                return WorktreeCleanupResult(False, "branch_ref_missing", record)
            current_ref = self._git_text(
                "rev-parse",
                "--verify",
                f"{record.branch_ref}^{{commit}}",
                cwd=self.repository,
                failure="managed worktree ref cannot be verified",
            )
            if current_ref != record.baseline_commit:
                return WorktreeCleanupResult(False, "branch_ref_changed", record)

            removing = replace(record, phase="removing")
            self._store_record(removing)
            self._git(
                "worktree",
                "unlock",
                os.fspath(root),
                cwd=self.repository,
                failure="Git refused to unlock the exact managed worktree",
            )
            self._git(
                "worktree",
                "remove",
                os.fspath(root),
                cwd=self.repository,
                failure="Git refused safe worktree removal",
            )
            if os.path.lexists(root):
                return WorktreeCleanupResult(
                    False,
                    "worktree_path_preserved",
                    removing,
                )
            if removing.git_dir is None or os.path.lexists(removing.git_dir):
                return WorktreeCleanupResult(
                    False,
                    "worktree_metadata_preserved",
                    removing,
                )
            return self._finalize_removed_worktree(removing)

    def _finalize_removed_worktree(
        self,
        record: WorktreeRecord,
    ) -> WorktreeCleanupResult:
        """Finish a removal only after the path and Git metadata are absent."""

        returncode, _stdout, _stderr = self._git(
            "show-ref",
            "--verify",
            "--quiet",
            record.branch_ref,
            cwd=self.repository,
            allowed_returncodes=frozenset({0, 1}),
            failure="managed worktree ref cannot be inspected",
        )
        if returncode == 1:
            self._delete_record(record)
            return WorktreeCleanupResult(True, "removed", record)
        current_ref = self._git_text(
            "rev-parse",
            "--verify",
            f"{record.branch_ref}^{{commit}}",
            cwd=self.repository,
            failure="managed worktree ref cannot be verified",
        )
        if current_ref != record.baseline_commit:
            preserved = replace(record, phase="ref_preserved")
            self._store_record(preserved)
            return WorktreeCleanupResult(False, "branch_ref_preserved", preserved)
        try:
            self._git(
                "update-ref",
                "-d",
                record.branch_ref,
                record.baseline_commit,
                cwd=self.repository,
                failure="worktree branch changed during compare-and-swap deletion",
            )
        except WorktreeError as exc:
            if exc.code in {"git_deadline_exceeded", "worktree_deadline_exceeded"}:
                raise
            preserved = replace(record, phase="ref_preserved")
            self._store_record(preserved)
            return WorktreeCleanupResult(False, "branch_ref_preserved", preserved)
        self._delete_record(record)
        return WorktreeCleanupResult(True, "removed", record)


__all__ = [
    "WORKTREE_RECORD_SCHEMA",
    "WorktreeCleanupResult",
    "WorktreeError",
    "WorktreeManager",
    "WorktreeRecord",
]
