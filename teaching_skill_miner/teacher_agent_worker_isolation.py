"""Linux confinement and per-process budgets for one Harness scope worker.

The API container deliberately runs all workers under one Unix uid. Directory
mode bits alone therefore do not isolate sibling tenant roots. Production
workers install an irreversible Landlock allowlist before constructing the
dashboard or starting any document parser. They also install irreversible
RLIMIT_CORE, RLIMIT_NOFILE, RLIMIT_FSIZE, and RLIMIT_AS ceilings inherited by
parser children. Unsupported kernels fail startup rather than silently
weakening the advertised boundary.
"""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping


class WorkerIsolationError(RuntimeError):
    """Raised when required process confinement cannot be installed."""


WORKER_PROCESS_RESOURCE_LIMITS_SCHEMA = (
    "teaching_skill_miner.worker_process_resource_limits.v1"
)
WORKER_PROCESS_RESOURCE_LIMITS_STATUS_SCHEMA = (
    "teaching_skill_miner.worker_process_resource_limits_status.v1"
)

# A project export can contain 256 MiB of private data and the worker's local
# parsers accept resources through a 17 MiB request envelope.  These defaults
# leave room for a ZIP input/output pair, Python's resident runtime, and one
# inherited parser process without making the container's 4 GiB memory limit
# the only boundary around a single worker.  RLIMIT_FSIZE is per file, not a
# storage quota; 512 MiB therefore preserves the 256 MiB export contract.
DEFAULT_WORKER_ADDRESS_SPACE_BYTES = 1536 * 1024 * 1024
DEFAULT_WORKER_FILE_SIZE_BYTES = 512 * 1024 * 1024
DEFAULT_WORKER_OPEN_FILES = 256

_MIN_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_MAX_WORKER_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024
_MIN_WORKER_FILE_SIZE_BYTES = 320 * 1024 * 1024
_MAX_WORKER_FILE_SIZE_BYTES = 512 * 1024 * 1024
_MIN_WORKER_OPEN_FILES = 128
_MAX_WORKER_OPEN_FILES = 512
_PROCESS_LIMIT_POLICY_FIELDS = frozenset(
    {
        "schema",
        "address_space_bytes",
        "file_size_bytes",
        "open_files",
        "core_dump_bytes",
    }
)


def default_worker_process_resource_limit_policy() -> dict[str, int | str]:
    """Return the versioned production policy sent in the private bootstrap."""

    return {
        "schema": WORKER_PROCESS_RESOURCE_LIMITS_SCHEMA,
        "address_space_bytes": DEFAULT_WORKER_ADDRESS_SPACE_BYTES,
        "file_size_bytes": DEFAULT_WORKER_FILE_SIZE_BYTES,
        "open_files": DEFAULT_WORKER_OPEN_FILES,
        "core_dump_bytes": 0,
    }


_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_EXECUTE = 1 << 0
_WRITE_FILE = 1 << 1
_READ_FILE = 1 << 2
_READ_DIR = 1 << 3
_REMOVE_DIR = 1 << 4
_REMOVE_FILE = 1 << 5
_MAKE_CHAR = 1 << 6
_MAKE_DIR = 1 << 7
_MAKE_REG = 1 << 8
_MAKE_SOCK = 1 << 9
_MAKE_FIFO = 1 << 10
_MAKE_BLOCK = 1 << 11
_MAKE_SYM = 1 << 12
_REFER = 1 << 13
_TRUNCATE = 1 << 14
_IOCTL_DEV = 1 << 15

_READ_ACCESS = _EXECUTE | _READ_FILE | _READ_DIR
_WRITE_ACCESS = (
    _WRITE_FILE
    | _REMOVE_DIR
    | _REMOVE_FILE
    | _MAKE_CHAR
    | _MAKE_DIR
    | _MAKE_REG
    | _MAKE_SOCK
    | _MAKE_FIFO
    | _MAKE_BLOCK
    | _MAKE_SYM
    | _REFER
    | _TRUNCATE
    | _IOCTL_DEV
)

_TRUSTED_READ_ONLY_SYSTEM_PATHS = (
    Path("/usr"),
    Path("/bin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc/ssl"),
    Path("/etc/ca-certificates"),
    Path("/etc/hosts"),
    Path("/etc/resolv.conf"),
    Path("/etc/nsswitch.conf"),
    Path("/etc/localtime"),
    Path("/dev/null"),
    Path("/dev/urandom"),
)


def _bounded_policy_integer(
    value: Any, *, minimum: int, maximum: int, field: str
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise WorkerIsolationError(f"worker resource limit {field} is invalid")
    return value


def _validated_process_resource_limit_policy(
    value: Mapping[str, Any] | None,
) -> dict[str, int | str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _PROCESS_LIMIT_POLICY_FIELDS
        or value.get("schema") != WORKER_PROCESS_RESOURCE_LIMITS_SCHEMA
        or value.get("core_dump_bytes") != 0
        or isinstance(value.get("core_dump_bytes"), bool)
        or not isinstance(value.get("core_dump_bytes"), int)
    ):
        raise WorkerIsolationError("worker resource limit policy is invalid")
    return {
        "schema": WORKER_PROCESS_RESOURCE_LIMITS_SCHEMA,
        "address_space_bytes": _bounded_policy_integer(
            value.get("address_space_bytes"),
            minimum=_MIN_WORKER_ADDRESS_SPACE_BYTES,
            maximum=_MAX_WORKER_ADDRESS_SPACE_BYTES,
            field="address_space_bytes",
        ),
        "file_size_bytes": _bounded_policy_integer(
            value.get("file_size_bytes"),
            minimum=_MIN_WORKER_FILE_SIZE_BYTES,
            maximum=_MAX_WORKER_FILE_SIZE_BYTES,
            field="file_size_bytes",
        ),
        "open_files": _bounded_policy_integer(
            value.get("open_files"),
            minimum=_MIN_WORKER_OPEN_FILES,
            maximum=_MAX_WORKER_OPEN_FILES,
            field="open_files",
        ),
        "core_dump_bytes": 0,
    }


def _resource_limit_target(
    *,
    desired: int,
    minimum_operational: int,
    current_hard: int,
    infinity: int,
) -> int:
    if isinstance(current_hard, bool) or not isinstance(current_hard, int):
        raise WorkerIsolationError("worker resource hard limit is invalid")
    target = desired if current_hard == infinity else min(desired, current_hard)
    if target < minimum_operational:
        raise WorkerIsolationError("worker resource hard limit is below safe minimum")
    return target


def _resource_limit_status(
    *,
    enforcement: str,
    address_space_bytes: int | None,
    file_size_bytes: int | None,
    open_files: int | None,
) -> dict[str, Any]:
    """Return only non-sensitive, aggregate policy facts for worker status."""

    return {
        "schema": WORKER_PROCESS_RESOURCE_LIMITS_STATUS_SCHEMA,
        "enforcement": enforcement,
        "address_space_bytes": address_space_bytes,
        "file_size_bytes": file_size_bytes,
        "open_files": open_files,
        "core_dump_bytes": 0 if enforcement == "linux_rlimit_v1" else None,
        # RLIMIT_CPU is cumulative over the lifetime of this long-lived worker.
        # Request wall-clock governors are the correct boundary instead.
        "cpu_time_limit": (
            "not_set_long_lived_worker_uses_request_wall_clock"
            if enforcement == "linux_rlimit_v1"
            else "not_required_local_or_test"
        ),
        # All API and worker processes share uid 10001 in the production image,
        # so RLIMIT_NPROC would make one worker consume another worker's budget.
        "process_count_limit": (
            "not_set_shared_uid_unsafe"
            if enforcement == "linux_rlimit_v1"
            else "not_required_local_or_test"
        ),
        "scope": (
            "per_process_individual_inherited_not_process_tree_aggregate"
            if enforcement == "linux_rlimit_v1"
            else "none"
        ),
    }


def install_worker_process_resource_limits(
    *, required: bool, policy: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Install irreversible Linux limits before the worker serves a request.

    The limits apply to this Python process and are inherited independently by
    parser children.  They are not an aggregate process-tree/cgroup memory or
    process-count budget; the production container remains the aggregate
    boundary.  Unsupported or unexpectedly weak Linux limits fail startup.
    """

    if not isinstance(required, bool):
        raise WorkerIsolationError("worker resource limit requirement is invalid")
    if not required:
        if policy is not None:
            raise WorkerIsolationError(
                "worker resource limit policy is forbidden when not required"
            )
        return _resource_limit_status(
            enforcement="not_required_local_or_test",
            address_space_bytes=None,
            file_size_bytes=None,
            open_files=None,
        )
    validated = _validated_process_resource_limit_policy(policy)
    if sys.platform != "linux":
        raise WorkerIsolationError("required process resource limits need Linux")

    required_resources = {
        "core_dump_bytes": "RLIMIT_CORE",
        "open_files": "RLIMIT_NOFILE",
        "file_size_bytes": "RLIMIT_FSIZE",
        "address_space_bytes": "RLIMIT_AS",
    }
    identifiers: dict[str, int] = {}
    existing: dict[str, tuple[int, int]] = {}
    try:
        for field, name in required_resources.items():
            identifier = getattr(resource, name)
            identifiers[field] = identifier
            existing[field] = resource.getrlimit(identifier)
        infinity = int(resource.RLIM_INFINITY)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise WorkerIsolationError(
            "required process resource limits are unavailable"
        ) from exc

    targets = {
        "core_dump_bytes": 0,
        "open_files": _resource_limit_target(
            desired=int(validated["open_files"]),
            minimum_operational=_MIN_WORKER_OPEN_FILES,
            current_hard=existing["open_files"][1],
            infinity=infinity,
        ),
        "file_size_bytes": _resource_limit_target(
            desired=int(validated["file_size_bytes"]),
            minimum_operational=_MIN_WORKER_FILE_SIZE_BYTES,
            current_hard=existing["file_size_bytes"][1],
            infinity=infinity,
        ),
        "address_space_bytes": _resource_limit_target(
            desired=int(validated["address_space_bytes"]),
            minimum_operational=_MIN_WORKER_ADDRESS_SPACE_BYTES,
            current_hard=existing["address_space_bytes"][1],
            infinity=infinity,
        ),
    }

    # Validate every inherited hard limit before irreversibly lowering any of
    # them.  Applying AS last leaves enough runtime available to verify all
    # four kernel results and construct the bounded status projection.
    try:
        for field in (
            "core_dump_bytes",
            "file_size_bytes",
            "open_files",
            "address_space_bytes",
        ):
            target = targets[field]
            identifier = identifiers[field]
            resource.setrlimit(identifier, (target, target))
            if resource.getrlimit(identifier) != (target, target):
                raise WorkerIsolationError("process resource limit verification failed")
    except (OSError, TypeError, ValueError) as exc:
        raise WorkerIsolationError(
            "process resource limit installation failed"
        ) from exc

    return _resource_limit_status(
        enforcement="linux_rlimit_v1",
        address_space_bytes=targets["address_space_bytes"],
        file_size_bytes=targets["file_size_bytes"],
        open_files=targets["open_files"],
    )


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _NetworkRulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


def _syscall_numbers() -> tuple[int, int, int]:
    if os.uname().machine not in {"x86_64", "aarch64"}:
        raise WorkerIsolationError("unsupported Linux architecture for Landlock")
    return 444, 445, 446


def _supported_access(abi: int) -> int:
    access = (_READ_ACCESS | _WRITE_ACCESS) & ~(_REFER | _TRUNCATE | _IOCTL_DEV)
    if abi >= 2:
        access |= _REFER
    if abi >= 3:
        access |= _TRUNCATE
    if abi >= 5:
        access |= _IOCTL_DEV
    return access


def landlock_abi_version() -> int:
    if sys.platform != "linux":
        return 0
    libc = ctypes.CDLL(None, use_errno=True)
    create_nr, _, _ = _syscall_numbers()
    result = int(libc.syscall(create_nr, 0, 0, _CREATE_RULESET_VERSION))
    return result if result > 0 else 0


def parser_network_isolation_required() -> bool:
    return os.environ.get("TSM_REQUIRE_PARSER_NETWORK_ISOLATION") == "1"


def parser_network_isolation_available() -> bool:
    return not parser_network_isolation_required() or landlock_abi_version() >= 4


def sandboxed_parser_command(command: Iterable[str]) -> list[str]:
    values = [str(value) for value in command]
    if not values or not values[0]:
        raise WorkerIsolationError("parser command is invalid")
    if not parser_network_isolation_required():
        return values
    if not parser_network_isolation_available():
        raise WorkerIsolationError("required parser network isolation is unavailable")
    return [
        sys.executable,
        "-m",
        "teaching_skill_miner.teacher_agent_parser_exec",
        "--",
        *values,
    ]


def install_parser_network_isolation() -> str:
    """Deny TCP bind/connect in a dedicated parser wrapper process."""

    if sys.platform != "linux":
        raise WorkerIsolationError("parser network isolation needs Linux")
    abi = landlock_abi_version()
    if abi < 4:
        raise WorkerIsolationError("Landlock network isolation is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    create_nr, _, restrict_nr = _syscall_numbers()
    ruleset = _NetworkRulesetAttr(
        handled_access_fs=0,
        handled_access_net=(1 << 0) | (1 << 1),
    )
    ruleset_fd = int(
        libc.syscall(create_nr, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    )
    if ruleset_fd < 0:
        raise WorkerIsolationError("parser network ruleset creation failed")
    try:
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise WorkerIsolationError("parser no_new_privs installation failed")
        if int(libc.syscall(restrict_nr, ruleset_fd, 0)) < 0:
            raise WorkerIsolationError("parser network restriction failed")
    finally:
        os.close(ruleset_fd)
    return f"linux_landlock_no_tcp_abi_{abi}"


def _existing_paths(
    values: Iterable[Path], *, resolve_trusted_system_symlinks: bool = False
) -> list[Path]:
    result: list[Path] = []
    for path in values:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) and not resolve_trusted_system_symlinks:
            raise WorkerIsolationError("Landlock allowlist path must not be a symlink")
        result.append(path.resolve(strict=True))
    return result


def _worker_allowlist_paths(
    *,
    private_root: Path,
    runtime_data_root: Path,
    provider_key_file: Path | None,
    application_root: Path | None,
) -> tuple[Path, list[Path]]:
    private_paths = _existing_paths([private_root])
    if len(private_paths) != 1:
        raise WorkerIsolationError("Landlock private root is unavailable")
    read_paths = _existing_paths(
        _TRUSTED_READ_ONLY_SYSTEM_PATHS,
        resolve_trusted_system_symlinks=True,
    )
    read_paths.extend(
        _existing_paths(
            [
                runtime_data_root,
                *(tuple([application_root]) if application_root is not None else ()),
                *(
                    tuple([provider_key_file])
                    if provider_key_file is not None
                    else ()
                ),
            ]
        )
    )
    return private_paths[0], read_paths


def install_worker_filesystem_isolation(
    *,
    required: bool,
    private_root: Path,
    runtime_data_root: Path,
    provider_key_file: Path | None,
    application_root: Path | None = None,
) -> str:
    """Install an irreversible filesystem allowlist for this process tree."""

    if not required:
        return "not_required_local_or_test"
    if sys.platform != "linux":
        raise WorkerIsolationError("required Landlock isolation needs Linux")
    resolved_private_root, read_paths = _worker_allowlist_paths(
        private_root=private_root,
        runtime_data_root=runtime_data_root,
        provider_key_file=provider_key_file,
        application_root=application_root,
    )
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    create_nr, add_nr, restrict_nr = _syscall_numbers()
    abi = int(syscall(create_nr, 0, 0, _CREATE_RULESET_VERSION))
    if abi < 1:
        error = ctypes.get_errno()
        if error in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
            raise WorkerIsolationError("Landlock is unavailable on this kernel")
        raise WorkerIsolationError("Landlock ABI query failed")
    handled = _supported_access(abi)
    ruleset = _RulesetAttr(handled_access_fs=handled)
    ruleset_fd = int(
        syscall(create_nr, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    )
    if ruleset_fd < 0:
        raise WorkerIsolationError("Landlock ruleset creation failed")

    scope_tmp = resolved_private_root / ".tmp"
    scope_tmp.mkdir(mode=0o700, exist_ok=True)
    scope_tmp.chmod(0o700)
    os.environ["TMPDIR"] = str(scope_tmp)
    tempfile.tempdir = str(scope_tmp)
    try:
        for path, allowed in [
            *((item, _READ_ACCESS & handled) for item in read_paths),
            (
                resolved_private_root,
                (_READ_ACCESS | _WRITE_ACCESS) & handled,
            ),
        ]:
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                if path.is_dir():
                    compatible = allowed
                else:
                    compatible = allowed & ~(
                        _READ_DIR
                        | _REMOVE_DIR
                        | _MAKE_CHAR
                        | _MAKE_DIR
                        | _MAKE_REG
                        | _MAKE_SOCK
                        | _MAKE_FIFO
                        | _MAKE_BLOCK
                        | _MAKE_SYM
                        | _REFER
                    )
                rule = _PathBeneathAttr(
                    allowed_access=compatible,
                    parent_fd=descriptor,
                    reserved=0,
                )
                if (
                    int(
                        syscall(
                            add_nr,
                            ruleset_fd,
                            _RULE_PATH_BENEATH,
                            ctypes.byref(rule),
                            0,
                        )
                    )
                    < 0
                ):
                    raise WorkerIsolationError("Landlock path rule installation failed")
            finally:
                os.close(descriptor)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise WorkerIsolationError("no_new_privs installation failed")
        if int(syscall(restrict_nr, ruleset_fd, 0)) < 0:
            raise WorkerIsolationError("Landlock restriction failed")
    finally:
        os.close(ruleset_fd)
    return f"linux_landlock_scope_allowlist_abi_{abi}"


__all__ = [
    "DEFAULT_WORKER_ADDRESS_SPACE_BYTES",
    "DEFAULT_WORKER_FILE_SIZE_BYTES",
    "DEFAULT_WORKER_OPEN_FILES",
    "WORKER_PROCESS_RESOURCE_LIMITS_SCHEMA",
    "WORKER_PROCESS_RESOURCE_LIMITS_STATUS_SCHEMA",
    "WorkerIsolationError",
    "default_worker_process_resource_limit_policy",
    "install_worker_filesystem_isolation",
    "install_parser_network_isolation",
    "install_worker_process_resource_limits",
    "landlock_abi_version",
    "parser_network_isolation_available",
    "parser_network_isolation_required",
    "sandboxed_parser_command",
]
