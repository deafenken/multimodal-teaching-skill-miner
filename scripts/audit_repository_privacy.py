#!/usr/bin/env python3
"""Fail CI when tracked or pending release files cross the privacy boundary."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.release_audit import (  # noqa: E402
    REVIEWED_PUBLIC_BINARY_RESOURCES,
    absolute_local_path_value,
    decode_text_payload,
    forbidden_binary_payload_kind,
    sensitive_identity_fields,
)
from scripts.verify_wheel_allowlist import (  # noqa: E402
    BUNDLED_TRANSCRIPTS,
    GOVERNANCE_FILES,
    PACKAGE_RESOURCE_FILES,
    PUBLIC_DATA_FILES,
    PUBLIC_JSON_ALLOWLIST,
)


FORBIDDEN_PREFIXES = (
    "data/real/",
    "artifacts/dipser_credible/",
    "artifacts/real_classroom/",
    "artifacts/private/",
    "defense/",
    "output/",
    "runs/",
    "tmp/",
)
ALLOWED_DOCUMENTS = {
    "data/real/README.md",
    "artifacts/dipser_credible/README.md",
    "artifacts/real_classroom/README.md",
}
ALLOWED_SYNTHETIC_MEDIA = {
    "data/demo/slide_example.png",
    "data/demo/slide_code.png",
    "data/demo/synthetic_lesson.mp4",
    "teaching_skill_miner/web/assets/student-xiaoyu.png",
    "teaching_skill_miner/web/assets/student-zimo.png",
    "teaching_skill_miner/web/assets/student-zhixing.png",
}
FORBIDDEN_SUFFIXES = {
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".npz",
    ".npy",
    ".parquet",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".safetensors",
    ".pkl",
    ".pickle",
    ".joblib",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".vtt",
    ".srt",
    ".ass",
    ".ssa",
    ".ttml",
}
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
OBVIOUS_SECRET_FIXTURE_MARKERS = (
    b"dummy",
    b"example",
    b"fake",
    b"fixture",
    b"placeholder",
    b"test",
)
LOCAL_PATH = re.compile(rb"/(?:Users|Volumes|home)/[^\s\"']+")
FORBIDDEN_PUBLIC_REPORT_PREFIXES = (
    "artifacts/public/teacher_agent_free_text_benchmark_deepseek",
)
RELEASE_SOURCE_TREES = (
    ".github/workflows",
    "apps",
    "configs",
    "data",
    "deploy",
    "docs",
    "docker",
    "release",
    "schema",
    "scripts",
    "teaching_skill_miner",
    "tests",
)
RELEASE_TEXT_SUFFIXES = {
    ".cjs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".lock",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
GENERATED_SOURCE_PARTS = {
    ".next",
    ".test-dist",
    ".turbo",
    "__pycache__",
    "coverage",
    "dist",
    "node_modules",
}
TOP_LEVEL_RELEASE_FILES = (
    ".env.example",
    "CHANGELOG.md",
    "LICENSE",
    "PRIVACY.md",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_DATA.md",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements.txt",
)


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("repository privacy audit requires a Git checkout")
    return sorted(
        value.decode("utf-8") for value in result.stdout.split(b"\0") if value
    )


def release_candidate_files() -> list[str]:
    """Return public release inputs even before they are staged in Git.

    This closes the gap where a newly created Task 2 resource could enter a
    clean build before ``git ls-files`` was able to inspect it.  Private/output
    roots are intentionally absent; tracking one of them is still rejected by
    ``FORBIDDEN_PREFIXES``.
    """

    candidates: set[str] = set()

    def include(path: Path) -> None:
        if path.exists() or path.is_symlink():
            candidates.add(path.relative_to(ROOT).as_posix())

    for relative in TOP_LEVEL_RELEASE_FILES:
        include(ROOT / relative)
    for relative in PACKAGE_RESOURCE_FILES:
        include(ROOT / relative)
    for filename in PUBLIC_DATA_FILES:
        include(ROOT / "data" / filename)
    for filename in BUNDLED_TRANSCRIPTS:
        include(ROOT / "data" / "transcripts" / filename)
    for filename in GOVERNANCE_FILES:
        include(ROOT / filename)

    allowlist = ROOT / PUBLIC_JSON_ALLOWLIST
    include(allowlist)
    if allowlist.is_file() and not allowlist.is_symlink():
        for raw in allowlist.read_text(encoding="utf-8").splitlines():
            relative = raw.strip()
            if relative:
                include(ROOT / relative)

    for relative_root in RELEASE_SOURCE_TREES:
        tree = ROOT / relative_root
        if not tree.is_dir() or tree.is_symlink():
            include(tree)
            continue
        for path in tree.rglob("*"):
            relative_parts = path.relative_to(ROOT).parts
            if (
                any(part in GENERATED_SOURCE_PARTS for part in relative_parts)
                or (
                    len(relative_parts) >= 2
                    and relative_parts[0] == "data"
                    and relative_parts[1] in {"generated", "private", "real"}
                )
            ):
                continue
            if path.is_symlink() or (
                path.is_file()
                and (
                    path.suffix.lower() in RELEASE_TEXT_SUFFIXES
                    or path.name
                    in {
                        ".dockerignore",
                        ".env.example",
                        ".gitignore",
                        "Caddyfile",
                        "Dockerfile",
                    }
                )
            ):
                include(path)

    # A release snapshot can contain dirty/untracked API, Console, deployment,
    # or other source files that its minimal historical Git index does not
    # name.  The snapshot manifest is the authoritative captured surface.
    snapshot_manifest = ROOT / ".release-source-snapshot.json"
    if snapshot_manifest.is_file() and not snapshot_manifest.is_symlink():
        try:
            snapshot_value = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("release source snapshot manifest is invalid") from exc
        rows = snapshot_value.get("files")
        if (
            snapshot_value.get("schema_version")
            != "teaching_skill_miner.release_source_snapshot.v1"
            or not isinstance(rows, list)
        ):
            raise RuntimeError("release source snapshot manifest is malformed")
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("release source snapshot file row is malformed")
            relative = row.get("path")
            if row.get("source_kind") != "release_source_or_public_evidence":
                continue
            if not isinstance(relative, str) or not relative:
                raise RuntimeError("release source snapshot path is malformed")
            include(ROOT / relative)

    public_artifacts = ROOT / "artifacts" / "public"
    if public_artifacts.is_dir() and not public_artifacts.is_symlink():
        for path in public_artifacts.rglob("*"):
            if path.is_file() or path.is_symlink():
                include(path)
    else:
        include(public_artifacts)
    return sorted(candidates)


def main() -> int:
    findings: list[str] = []
    try:
        tracked = set(tracked_files())
        candidates = set(release_candidate_files())
        files = sorted(tracked | candidates)
    except RuntimeError as exc:
        print(f"Repository privacy audit unavailable: {exc}", file=sys.stderr)
        return 2
    for relative in files:
        lowered = relative.lower()
        path = ROOT / relative
        source_kind = "tracked file" if relative in tracked else "release candidate"
        if path.is_symlink():
            findings.append(f"unsafe symlink is a {source_kind}: {relative}")
            continue
        if relative not in ALLOWED_DOCUMENTS and any(
            lowered.startswith(prefix) for prefix in FORBIDDEN_PREFIXES
        ):
            findings.append(f"private path is a {source_kind}: {relative}")
        if any(
            lowered.startswith(prefix) for prefix in FORBIDDEN_PUBLIC_REPORT_PREFIXES
        ):
            findings.append(
                "row-level Teacher Agent benchmark report must not be public: "
                f"{relative}"
            )
        if (
            path.suffix.lower() in FORBIDDEN_SUFFIXES
            and relative not in ALLOWED_SYNTHETIC_MEDIA
        ):
            findings.append(
                f"private binary/archive type is a {source_kind}: {relative}"
            )
        if not path.is_file():
            continue
        size_bytes = path.stat().st_size
        if size_bytes > 2_000_000 and relative not in ALLOWED_SYNTHETIC_MEDIA:
            findings.append(
                "oversized release file requires explicit privacy review: "
                f"{relative} ({size_bytes} bytes)"
            )
        with path.open("rb") as handle:
            payload = handle.read() if size_bytes <= 2_000_000 else handle.read(65536)
        reviewed_digest = REVIEWED_PUBLIC_BINARY_RESOURCES.get(relative)
        if reviewed_digest is not None and sha256(payload).hexdigest() != reviewed_digest:
            findings.append(
                f"reviewed synthetic resource digest changed: {relative}"
            )
        if relative not in ALLOWED_SYNTHETIC_MEDIA:
            binary_kind = forbidden_binary_payload_kind(payload)
            if binary_kind:
                findings.append(
                    f"private binary signature is a {source_kind} {relative}: {binary_kind}"
                )
            elif decode_text_payload(payload) is None:
                findings.append(
                    f"unknown binary payload is a {source_kind}: {relative}"
                )
        for field in sensitive_identity_fields(relative, payload):
            findings.append(
                f"row-level identity field is in {source_kind} {relative}: {field}"
            )
        # Source code legitimately contains configuration identifiers and
        # explicit dummy test values.  Scan it for provider/key formats with a
        # high-confidence signature; the broader release-audit heuristic still
        # applies to distributable/public artifacts.
        secret_matches = [
            match.group(0)
            for pattern in SECRET_PATTERNS
            for match in pattern.finditer(payload)
        ]
        if any(
            not any(marker in match.lower() for marker in OBVIOUS_SECRET_FIXTURE_MARKERS)
            for match in secret_matches
        ):
            findings.append(f"possible secret in {source_kind}: {relative}")
        if absolute_local_path_value(payload):
            findings.append(f"absolute local path in {source_kind}: {relative}")
    if findings:
        print("Repository privacy audit failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 2
    print(
        "Repository privacy audit passed for "
        f"{len(tracked)} tracked files and {len(candidates - tracked)} pending release candidates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
