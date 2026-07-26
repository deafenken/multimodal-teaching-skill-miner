from __future__ import annotations

import json
import os
from pathlib import Path
import sysconfig
import tempfile
from typing import Any


def ensure_private_directory(path: str | Path) -> Path:
    """Create an artifact directory restricted to the current user on POSIX."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        target.chmod(0o700)
    return target


def ensure_private_file(path: str | Path) -> Path:
    """Restrict an existing sensitive artifact to the current user on POSIX."""

    target = Path(path)
    if os.name == "posix" and target.exists():
        target.chmod(0o600)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        target,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )
    return target


def write_text(path: str | Path, value: str) -> Path:
    target = Path(path)
    _atomic_write_text(target, value)
    return target


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _atomic_write_text(target: Path, value: str) -> None:
    """Write one UTF-8 artifact atomically in the target directory."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        ensure_private_file(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def resource_root() -> Path:
    """Locate bundled demo data in a source tree, wheel, or explicit override."""

    override = os.getenv("TSM_RESOURCE_ROOT", "").strip()
    package_parent = Path(__file__).resolve().parent.parent
    candidates = [
        Path(override).expanduser() if override else None,
        package_parent,
        package_parent / "share" / "teaching-skill-miner",
        Path(sysconfig.get_path("data")) / "share" / "teaching-skill-miner",
    ]
    for candidate in candidates:
        if candidate is not None and (
            candidate / "data" / "dataset_manifest.json"
        ).is_file():
            return candidate.resolve()
    locations = ", ".join(str(value) for value in candidates if value is not None)
    raise FileNotFoundError(
        "bundled demo resources were not found; set TSM_RESOURCE_ROOT or reinstall "
        f"the package with data files (searched: {locations})"
    )


def resolve_resource_path(path: str | Path) -> Path:
    """Resolve an existing user path, then fall back to bundled resources."""

    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    bundled = resource_root() / candidate
    if bundled.exists():
        return bundled.resolve()
    raise FileNotFoundError(candidate)


def resolve_manifest_root(
    manifest_path: str | Path, manifest: dict[str, Any]
) -> Path:
    """Find the root against which every manifest transcript path resolves."""

    path = Path(manifest_path).resolve()
    candidates = list(
        dict.fromkeys(
            [path.parent, path.parent.parent, resource_root(), Path.cwd().resolve()]
        )
    )
    relative_paths = [
        Path(str(item.get("transcript_path", "")))
        for item in manifest.get("videos", [])
        if str(item.get("transcript_path", "")).strip()
    ]
    for candidate in candidates:
        if relative_paths and all((candidate / value).is_file() for value in relative_paths):
            return candidate
    raise FileNotFoundError(
        "manifest transcript paths do not resolve beneath any known resource root"
    )
