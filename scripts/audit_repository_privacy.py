#!/usr/bin/env python3
"""Fail CI when tracked public files cross the documented privacy boundary."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.release_audit import (  # noqa: E402
    absolute_local_path_value,
    decode_text_payload,
    forbidden_binary_payload_kind,
    secret_pattern_details,
    sensitive_identity_fields,
)


FORBIDDEN_PREFIXES = (
    "data/real/",
    "artifacts/dipser_credible/",
    "artifacts/real_classroom/",
    "artifacts/private/",
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
}
FORBIDDEN_SUFFIXES = {
    ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz",
    ".npz", ".npy", ".parquet", ".mp4", ".mov", ".mkv", ".webm",
    ".avi", ".m4v", ".wav", ".mp3", ".m4a", ".aac", ".flac",
    ".ogg", ".opus", ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".gif", ".tif", ".tiff", ".pt", ".pth", ".ckpt", ".onnx",
    ".safetensors", ".pkl", ".pickle", ".joblib", ".sqlite", ".sqlite3", ".db",
}
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
LOCAL_PATH = re.compile(rb"/(?:Users|Volumes|home)/[^\s\"']+")


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("repository privacy audit requires a Git checkout")
    return sorted(value.decode("utf-8") for value in result.stdout.split(b"\0") if value)


def main() -> int:
    findings: list[str] = []
    try:
        files = tracked_files()
    except RuntimeError as exc:
        print(f"Repository privacy audit unavailable: {exc}", file=sys.stderr)
        return 2
    for relative in files:
        lowered = relative.lower()
        path = ROOT / relative
        if relative not in ALLOWED_DOCUMENTS and any(
            lowered.startswith(prefix) for prefix in FORBIDDEN_PREFIXES
        ):
            findings.append(f"private path is tracked: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and relative not in ALLOWED_SYNTHETIC_MEDIA:
            findings.append(f"private binary/archive type is tracked: {relative}")
        if not path.is_file():
            continue
        size_bytes = path.stat().st_size
        if size_bytes > 2_000_000 and relative not in ALLOWED_SYNTHETIC_MEDIA:
            findings.append(
                f"oversized tracked file requires explicit privacy review: {relative} ({size_bytes} bytes)"
            )
        with path.open("rb") as handle:
            payload = handle.read() if size_bytes <= 2_000_000 else handle.read(65536)
        if relative not in ALLOWED_SYNTHETIC_MEDIA:
            binary_kind = forbidden_binary_payload_kind(payload)
            if binary_kind:
                findings.append(
                    f"private binary signature is tracked as {relative}: {binary_kind}"
                )
            elif decode_text_payload(payload) is None:
                findings.append(f"unknown binary payload is tracked: {relative}")
        for field in sensitive_identity_fields(relative, payload):
            findings.append(
                f"row-level identity field is tracked in {relative}: {field}"
            )
        if secret_pattern_details(payload):
            findings.append(f"possible secret in tracked file: {relative}")
        if absolute_local_path_value(payload):
            findings.append(f"absolute local path in tracked file: {relative}")
    if findings:
        print("Repository privacy audit failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 2
    print(f"Repository privacy audit passed for {len(files)} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
