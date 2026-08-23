"""Workspace-confined coding tools with explicit permission profiles."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import tempfile
from threading import Thread
import time
from typing import Any, Mapping

from ..core import (
    RetryPolicy,
    ToolExecutionContext,
    ToolExecutionError,
    ToolRegistry,
    ToolSpec,
)


PERMISSION_PROFILES: dict[str, frozenset[str]] = {
    "read-only": frozenset({"workspace.read"}),
    "workspace-write": frozenset({"workspace.read", "workspace.write"}),
    "full-access": frozenset(
        {"workspace.read", "workspace.write", "process.exec"}
    ),
}

_SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".private",
        ".agent-harness",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)
_PROTECTED_PATHS = frozenset({".git", ".private", ".agent-harness"})
_MAX_FILE_BYTES = 1_000_000
_MAX_PATCH_BYTES = 250_000
_MAX_TOOL_TEXT_CHARS = 8_000
_MAX_COMMAND_OUTPUT_BYTES = 8_000
_MAX_TRAVERSED_ENTRIES = 50_000
_MAX_SEARCH_BYTES = 32 * 1024 * 1024
_PROCESS_TERM_GRACE_SECONDS = 0.25
_PROCESS_KILL_GRACE_SECONDS = 0.25
_PATCH_BINARY = "/usr/bin/patch"
_SAFE_EXEC_PATH = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PATCH_MODE = re.compile(r"^(?:old|new|deleted file|new file) mode\s+120000$")


def permission_profile(name: str) -> set[str]:
    try:
        return set(PERMISSION_PROFILES[name])
    except KeyError as exc:
        raise ValueError(f"unknown permission profile: {name}") from exc


def _schema(properties: Mapping[str, Any], *, required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _string(*, max_length: int, min_length: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "string", "maxLength": max_length}
    if min_length:
        result["minLength"] = min_length
    return result


def _integer(*, minimum: int, maximum: int) -> dict[str, Any]:
    return {"type": "integer", "minimum": minimum, "maximum": maximum}


def _safe_relative(value: Any, *, allow_dot: bool = True) -> Path:
    if not isinstance(value, str) or "\x00" in value:
        raise ToolExecutionError("path must be a string", code="invalid_path")
    normalized = value.strip().replace("\\", "/")
    if allow_dot and normalized in {"", "."}:
        return Path(".")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ToolExecutionError("path must stay inside the workspace", code="path_escape")
    if any(part.casefold() in _PROTECTED_PATHS for part in pure.parts):
        raise ToolExecutionError("path is protected", code="protected_path")
    return Path(*pure.parts)


def _bounded_text(data: bytes, *, maximum: int) -> tuple[str, bool]:
    truncated = len(data) > maximum
    data = data[:maximum]
    if b"\x00" in data:
        raise ToolExecutionError("binary content is not supported", code="binary_content")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolExecutionError("file is not valid UTF-8", code="invalid_utf8") from exc
    return _CONTROL.sub("�", text), truncated


def _bounded_process_text(data: bytes, *, maximum: int) -> tuple[str, bool]:
    truncated = len(data) > maximum
    text = data[:maximum].decode("utf-8", errors="replace")
    return _CONTROL.sub("�", text.replace("\x00", "�")), truncated


def _check_context(context: ToolExecutionContext) -> None:
    context.cancellation_token.raise_if_cancelled()
    if time.monotonic() >= context.deadline_monotonic:
        raise ToolExecutionError("tool deadline exceeded", code="timeout")


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process: subprocess.Popen[bytes]) -> bool:
    process_group_id = process.pid
    existed = _process_group_exists(process_group_id)
    if not existed:
        return False
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.poll()
        if not _process_group_exists(process_group_id):
            return existed
    deadline = time.monotonic() + _PROCESS_TERM_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process_group_id):
            return existed
        time.sleep(0.01)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.poll()
        if not _process_group_exists(process_group_id):
            return existed
    deadline = time.monotonic() + _PROCESS_KILL_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process_group_id):
            return existed
        time.sleep(0.01)
    process.poll()
    if not _process_group_exists(process_group_id):
        return existed
    raise ToolExecutionError(
        "command process group could not be reaped",
        code="command_cleanup_failed",
    )


def _safe_process_environment() -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in ("LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TZ")
        if key in os.environ
    }
    environment.update(
        {
            "AGENT_HARNESS": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "PATH": _SAFE_EXEC_PATH,
            "ZDOTDIR": "/nonexistent",
        }
    )
    return environment


def _patch_path(raw: str, *, strip_git_prefix: bool) -> Path | None:
    value = raw.split("\t", 1)[0].strip()
    if value == "/dev/null":
        return None
    if (
        not value
        or any(character in value for character in ('"', "'", "\\"))
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ToolExecutionError(
            "quoted or escaped patch paths are not supported",
            code="invalid_patch_path",
        )
    if strip_git_prefix:
        if not value.startswith(("a/", "b/")):
            raise ToolExecutionError(
                "patch path is missing its Git prefix",
                code="invalid_patch_path",
            )
        value = value[2:]
    return _safe_relative(value, allow_dot=False)


class WorkspaceToolset:
    """Handlers whose filesystem authority is confined to one real directory."""

    def __init__(self, root: os.PathLike[str] | str) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("workspace root must be a directory")

    def _resolve_existing(self, relative: Any) -> Path:
        path = _safe_relative(relative)
        candidate = self.root / path
        current = self.root
        for part in path.parts:
            current = current / part
            try:
                entry = current.lstat()
            except FileNotFoundError as exc:
                raise ToolExecutionError("path does not exist", code="path_not_found") from exc
            if stat.S_ISLNK(entry.st_mode):
                raise ToolExecutionError("symbolic links are not followed", code="symlink_rejected")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ToolExecutionError("path escaped the workspace", code="path_escape") from exc
        return resolved

    def _read_regular_file(self, relative: Path) -> bytes:
        root_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            root_flags |= os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            root_flags |= os.O_CLOEXEC
        descriptor = os.open(self.root, root_flags)
        try:
            parts = relative.parts
            for index, part in enumerate(parts):
                try:
                    entry = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
                except OSError as exc:
                    raise ToolExecutionError(
                        "file path does not exist",
                        code="path_not_found",
                    ) from exc
                if stat.S_ISLNK(entry.st_mode):
                    raise ToolExecutionError(
                        "symbolic links are not followed",
                        code="symlink_rejected",
                    )
                flags = os.O_RDONLY
                if hasattr(os, "O_CLOEXEC"):
                    flags |= os.O_CLOEXEC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                if hasattr(os, "O_NONBLOCK"):
                    flags |= os.O_NONBLOCK
                if index < len(parts) - 1 and hasattr(os, "O_DIRECTORY"):
                    flags |= os.O_DIRECTORY
                try:
                    next_descriptor = os.open(part, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise ToolExecutionError(
                        "file path cannot be opened safely",
                        code="path_not_found",
                    ) from exc
                os.close(descriptor)
                descriptor = next_descriptor
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ToolExecutionError("read path must be a regular file", code="not_file")
            if before.st_size > _MAX_FILE_BYTES:
                raise ToolExecutionError("file exceeds the read limit", code="file_too_large")
            chunks: list[bytes] = []
            remaining = _MAX_FILE_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            after = os.fstat(descriptor)
            if (
                len(data) > _MAX_FILE_BYTES
                or after.st_dev != before.st_dev
                or after.st_ino != before.st_ino
                or after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns
            ):
                raise ToolExecutionError(
                    "file changed while it was being read",
                    code="file_changed",
                )
            return data
        finally:
            os.close(descriptor)

    def list_files(self, arguments: Mapping[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        _check_context(context)
        base = self._resolve_existing(arguments.get("path", "."))
        if not base.is_dir():
            raise ToolExecutionError("list path must be a directory", code="not_directory")
        pattern = str(arguments.get("pattern", "*")).strip() or "*"
        maximum = int(arguments.get("max_results", 200))
        files: list[str] = []
        result_chars = 0
        traversed = 0
        for directory, names, filenames in os.walk(base, followlinks=False):
            _check_context(context)
            names[:] = sorted(
                name
                for name in names
                if name.casefold() not in _SKIP_DIRECTORIES
                and not (Path(directory) / name).is_symlink()
            )
            for filename in sorted(filenames):
                traversed += 1
                if traversed > _MAX_TRAVERSED_ENTRIES:
                    return {"files": files, "truncated": True}
                _check_context(context)
                candidate = Path(directory) / filename
                try:
                    candidate_stat = candidate.lstat()
                except OSError:
                    continue
                if not stat.S_ISREG(candidate_stat.st_mode):
                    continue
                relative = candidate.relative_to(self.root).as_posix()
                try:
                    _safe_relative(relative, allow_dot=False)
                except ToolExecutionError:
                    continue
                if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(filename, pattern):
                    if result_chars + len(relative) > _MAX_TOOL_TEXT_CHARS:
                        return {"files": files, "truncated": True}
                    files.append(relative)
                    result_chars += len(relative)
                    if len(files) >= maximum:
                        return {"files": files, "truncated": True}
        return {"files": files, "truncated": False}

    def read_file(self, arguments: Mapping[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        _check_context(context)
        relative_path = _safe_relative(arguments.get("path"), allow_dot=False)
        text, truncated = _bounded_text(
            self._read_regular_file(relative_path),
            maximum=_MAX_FILE_BYTES,
        )
        lines = text.splitlines()
        start = int(arguments.get("start_line", 1))
        end = int(arguments.get("end_line", min(len(lines), start + 399)))
        if end < start:
            raise ToolExecutionError("end_line must not precede start_line", code="invalid_range")
        requested = lines[start - 1 : end]
        selected: list[str] = []
        content_chars = 0
        result_truncated = truncated or end < len(lines)
        for index, line in enumerate(requested, start=start):
            _check_context(context)
            rendered = f"{index}: {line}"
            remaining = _MAX_TOOL_TEXT_CHARS - content_chars
            if remaining <= 0:
                result_truncated = True
                break
            if len(rendered) > remaining:
                selected.append(rendered[:remaining])
                result_truncated = True
                break
            selected.append(rendered)
            content_chars += len(rendered) + 1
        return {
            "path": relative_path.as_posix(),
            "start_line": start,
            "end_line": start + max(0, len(selected) - 1),
            "total_lines": len(lines),
            "content": "\n".join(selected),
            "truncated": result_truncated or len(selected) < len(requested),
        }

    def search(self, arguments: Mapping[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        _check_context(context)
        query = str(arguments.get("query", ""))
        if not query or len(query) > 1_000:
            raise ToolExecutionError("search query is invalid", code="invalid_query")
        base = self._resolve_existing(arguments.get("path", "."))
        if not base.is_dir():
            raise ToolExecutionError("search path must be a directory", code="not_directory")
        glob = str(arguments.get("glob", "*")).strip() or "*"
        maximum = int(arguments.get("max_results", 100))
        matches: list[dict[str, Any]] = []
        result_chars = 0
        traversed = 0
        searched_bytes = 0
        for directory, names, filenames in os.walk(base, followlinks=False):
            _check_context(context)
            names[:] = [
                name
                for name in names
                if name.casefold() not in _SKIP_DIRECTORIES
                and not (Path(directory) / name).is_symlink()
            ]
            for filename in filenames:
                traversed += 1
                if traversed > _MAX_TRAVERSED_ENTRIES:
                    return {"matches": matches, "truncated": True}
                _check_context(context)
                candidate = Path(directory) / filename
                relative_path = candidate.relative_to(self.root)
                relative = relative_path.as_posix()
                try:
                    _safe_relative(relative, allow_dot=False)
                except ToolExecutionError:
                    continue
                if not (fnmatch.fnmatch(relative, glob) or fnmatch.fnmatch(filename, glob)):
                    continue
                try:
                    data = self._read_regular_file(relative_path)
                    searched_bytes += len(data)
                    if searched_bytes > _MAX_SEARCH_BYTES:
                        return {"matches": matches, "truncated": True}
                    text, _ = _bounded_text(data, maximum=_MAX_FILE_BYTES)
                except (OSError, ToolExecutionError):
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    _check_context(context)
                    if query.casefold() in line.casefold():
                        rendered = line[:500]
                        added_chars = len(relative) + len(rendered) + 32
                        if result_chars + added_chars > _MAX_TOOL_TEXT_CHARS:
                            return {"matches": matches, "truncated": True}
                        matches.append(
                            {
                                "path": relative,
                                "line": line_number,
                                "text": rendered,
                            }
                        )
                        result_chars += added_chars
                        if len(matches) >= maximum:
                            return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def _run_process(
        self,
        command: list[str],
        context: ToolExecutionContext,
        *,
        stdin: bytes | None = None,
        timeout: float,
        environment: Mapping[str, str] | None = None,
        begin_effect: bool = False,
    ) -> dict[str, Any]:
        _check_context(context)
        started = time.monotonic()
        effective_deadline = min(started + timeout, context.deadline_monotonic)
        stdin_file = None
        if stdin is not None:
            stdin_file = tempfile.TemporaryFile()
            stdin_file.write(stdin)
            stdin_file.seek(0)
        try:
            if begin_effect:
                # All validation and temporary-input preparation precede this
                # durable boundary. Returning from it is the authority to
                # create the process or mutate the workspace.
                context.begin_effect()
            process = subprocess.Popen(
                command,
                cwd=self.root,
                stdin=stdin_file if stdin_file is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=(
                    dict(environment)
                    if environment is not None
                    else _safe_process_environment()
                ),
            )
        finally:
            if stdin_file is not None:
                stdin_file.close()
        assert process.stdout is not None
        output = bytearray()
        reader_errors: list[BaseException] = []

        def drain_output() -> None:
            try:
                while True:
                    chunk = process.stdout.read(64 * 1024)
                    if not chunk:
                        return
                    remaining = _MAX_COMMAND_OUTPUT_BYTES + 1 - len(output)
                    if remaining > 0:
                        output.extend(chunk[:remaining])
            except BaseException as exc:
                reader_errors.append(exc)

        reader = Thread(
            target=drain_output,
            name="agent-harness-process-output",
            daemon=True,
        )
        reader.start()
        try:
            while process.poll() is None:
                context.cancellation_token.raise_if_cancelled()
                if time.monotonic() >= effective_deadline:
                    raise ToolExecutionError("command timed out", code="command_timeout")
                time.sleep(0.02)
        except BaseException:
            cleanup_error: ToolExecutionError | None = None
            try:
                _terminate_process_group(process)
            except ToolExecutionError as exc:
                cleanup_error = exc
            try:
                process.wait(timeout=_PROCESS_KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            reader.join(_PROCESS_TERM_GRACE_SECONDS)
            if reader.is_alive():
                process.stdout.close()
                reader.join(_PROCESS_KILL_GRACE_SECONDS)
            if cleanup_error is not None:
                raise cleanup_error
            raise
        # Reap descendants that remain in the command's process group. A
        # full-access command can deliberately create a new session and escape
        # this best-effort boundary; the permission documentation makes that
        # host-level authority explicit.
        background_processes_reaped = _terminate_process_group(process)
        reader.join(_PROCESS_TERM_GRACE_SECONDS)
        if reader.is_alive():
            process.stdout.close()
            reader.join(_PROCESS_KILL_GRACE_SECONDS)
        if reader.is_alive():
            raise ToolExecutionError(
                "command output pipe did not settle",
                code="command_output_timeout",
            )
        process.stdout.close()
        if reader_errors:
            raise ToolExecutionError(
                "command output could not be collected",
                code="command_output_error",
            ) from reader_errors[0]
        text, truncated = _bounded_process_text(
            bytes(output),
            maximum=_MAX_COMMAND_OUTPUT_BYTES,
        )
        return {
            "exit_code": (
                125
                if background_processes_reaped and process.returncode == 0
                else int(process.returncode)
            ),
            "output": text,
            "truncated": truncated,
            "background_processes_reaped": background_processes_reaped,
            "duration_ms": round((time.monotonic() - started) * 1_000),
        }

    def apply_patch(self, arguments: Mapping[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        patch = arguments.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            raise ToolExecutionError("patch is required", code="invalid_patch")
        encoded = patch.encode("utf-8")
        if len(encoded) > _MAX_PATCH_BYTES or b"\x00" in encoded:
            raise ToolExecutionError("patch exceeds the safety limit", code="invalid_patch")
        touched: set[str] = set()
        for line in patch.splitlines():
            candidate_paths: list[Path | None] = []
            if _PATCH_MODE.fullmatch(line):
                raise ToolExecutionError(
                    "patches may not create or modify symbolic links",
                    code="symlink_rejected",
                )
            if line.startswith("diff --git "):
                parts = line.split(" ")
                if len(parts) != 4:
                    raise ToolExecutionError(
                        "quoted or spaced patch paths are not supported",
                        code="invalid_patch_path",
                    )
                candidate_paths.extend(
                    (
                        _patch_path(parts[2], strip_git_prefix=True),
                        _patch_path(parts[3], strip_git_prefix=True),
                    )
                )
            elif line.startswith(("--- ", "+++ ")):
                candidate_paths.append(
                    _patch_path(line[4:], strip_git_prefix=True)
                )
            elif line.startswith(("rename from ", "rename to ")):
                candidate_paths.append(
                    _patch_path(line.split(" ", 2)[2], strip_git_prefix=False)
                )
            elif line.startswith(("copy from ", "copy to ")):
                candidate_paths.append(
                    _patch_path(line.split(" ", 2)[2], strip_git_prefix=False)
                )
            else:
                continue
            for relative in candidate_paths:
                if relative is None:
                    continue
                touched.add(relative.as_posix())
                if (
                    len(touched) > 1_000
                    or sum(len(item) for item in touched) > _MAX_TOOL_TEXT_CHARS
                ):
                    raise ToolExecutionError(
                        "patch touches too many paths",
                        code="patch_result_too_large",
                    )
                current = self.root
                for component in relative.parts:
                    current = current / component
                    try:
                        metadata = current.lstat()
                    except FileNotFoundError:
                        break
                    if stat.S_ISLNK(metadata.st_mode):
                        raise ToolExecutionError(
                            "patch target cannot traverse a symlink",
                            code="symlink_rejected",
                        )
        if not touched:
            raise ToolExecutionError("patch contains no workspace paths", code="invalid_patch")
        checked = self._run_process(
            [
                _PATCH_BINARY,
                "-C",
                "-f",
                "-s",
                "-p1",
                "-r",
                os.devnull,
                "-V",
                "none",
                "-d",
                str(self.root),
            ],
            context,
            stdin=encoded,
            timeout=15.0,
        )
        if checked["exit_code"] != 0:
            raise ToolExecutionError("patch preflight failed", code="patch_rejected")
        applied = self._run_process(
            [
                _PATCH_BINARY,
                "-f",
                "-s",
                "-p1",
                "-r",
                os.devnull,
                "-V",
                "none",
                "-d",
                str(self.root),
            ],
            context,
            stdin=encoded,
            timeout=20.0,
            begin_effect=True,
        )
        if applied["exit_code"] != 0:
            raise ToolExecutionError("patch application failed", code="patch_failed")
        for relative_text in touched:
            candidate = self.root / Path(relative_text)
            if candidate.is_symlink():
                raise ToolExecutionError(
                    "patch unexpectedly created a symlink",
                    code="symlink_rejected",
                )
        return {"applied": True, "paths": sorted(touched)}

    def run_command(self, arguments: Mapping[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip() or len(command) > 20_000:
            raise ToolExecutionError("command is invalid", code="invalid_command")
        timeout = float(arguments.get("timeout_seconds", 60))
        if not 0.1 <= timeout <= 300 or not timeout == timeout:
            raise ToolExecutionError("command timeout is invalid", code="invalid_timeout")
        shell = "/bin/zsh" if Path("/bin/zsh").is_file() else "/bin/bash"
        shell_arguments = [shell, "-f", "-c", command]
        if shell.endswith("bash"):
            shell_arguments = [shell, "--noprofile", "--norc", "-c", command]
        return self._run_process(
            shell_arguments,
            context,
            timeout=timeout,
            environment=_safe_process_environment(),
            begin_effect=True,
        )


def build_workspace_registry(root: os.PathLike[str] | str) -> ToolRegistry:
    tools = WorkspaceToolset(root)
    registry = ToolRegistry()
    trusted_reason = "Repository-owned bounded workspace handler with explicit path checks"
    specs = (
        (
            ToolSpec(
                name="workspace.list",
                version="1",
                description="List files below a workspace directory without following symlinks.",
                input_schema=_schema(
                    {
                        "path": _string(max_length=1_024),
                        "pattern": _string(max_length=500),
                        "max_results": _integer(minimum=1, maximum=1_000),
                    }
                ),
                permission="workspace.read",
                data_scope="workspace_read",
                execution_isolation="trusted_inline",
                trusted_inline_reason=trusted_reason,
            ),
            tools.list_files,
        ),
        (
            ToolSpec(
                name="workspace.read",
                version="1",
                description="Read a bounded UTF-8 line range from one workspace file.",
                input_schema=_schema(
                    {
                        "path": _string(max_length=1_024, min_length=1),
                        "start_line": _integer(minimum=1, maximum=10_000_000),
                        "end_line": _integer(minimum=1, maximum=10_000_000),
                    },
                    required=("path",),
                ),
                permission="workspace.read",
                data_scope="workspace_read",
                execution_isolation="trusted_inline",
                trusted_inline_reason=trusted_reason,
            ),
            tools.read_file,
        ),
        (
            ToolSpec(
                name="workspace.search",
                version="1",
                description="Search UTF-8 workspace files for a literal case-insensitive string.",
                input_schema=_schema(
                    {
                        "query": _string(max_length=1_000, min_length=1),
                        "path": _string(max_length=1_024),
                        "glob": _string(max_length=500),
                        "max_results": _integer(minimum=1, maximum=1_000),
                    },
                    required=("query",),
                ),
                permission="workspace.read",
                data_scope="workspace_read",
                execution_isolation="trusted_inline",
                trusted_inline_reason=trusted_reason,
            ),
            tools.search,
        ),
        (
            ToolSpec(
                name="workspace.patch",
                version="1",
                description="Apply one validated unified diff inside the workspace.",
                input_schema=_schema(
                    {"patch": _string(max_length=_MAX_PATCH_BYTES, min_length=1)},
                    required=("patch",),
                ),
                permission="workspace.write",
                risk="medium",
                replay_policy="never",
                data_scope="workspace_write",
                timeout_seconds=30.0,
                execution_isolation="trusted_inline",
                trusted_inline_reason=trusted_reason,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            tools.apply_patch,
        ),
        (
            ToolSpec(
                name="process.exec",
                version="1",
                description="Run one zsh command in the workspace with secrets removed from the environment.",
                input_schema=_schema(
                    {
                        "command": _string(max_length=20_000, min_length=1),
                        "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 300},
                    },
                    required=("command",),
                ),
                permission="process.exec",
                risk="high",
                replay_policy="never",
                data_scope="host_access",
                timeout_seconds=310.0,
                execution_isolation="trusted_inline",
                trusted_inline_reason=trusted_reason,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            tools.run_command,
        ),
    )
    for spec, handler in specs:
        registry.register(spec, handler)
    return registry


__all__ = [
    "PERMISSION_PROFILES",
    "WorkspaceToolset",
    "build_workspace_registry",
    "permission_profile",
]
