#!/usr/bin/env python3
"""Run the externally governed Teaching Agent semantic lockbox once."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Any

from teaching_skill_miner.io_utils import write_json
from teaching_skill_miner.teacher_agent_semantic_lockbox import (
    SemanticLockboxError,
    score_semantic_lockbox,
)


_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_KEY_BYTES = 64 * 1024


def _read_regular_file(path_value: str, *, maximum: int, label: str) -> bytes:
    path = Path(path_value).expanduser()
    try:
        details = path.lstat()
    except OSError as exc:
        raise SemanticLockboxError(f"{label} cannot be read") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise SemanticLockboxError(f"{label} must be a regular non-symlink file")
    if not 0 < details.st_size <= maximum:
        raise SemanticLockboxError(f"{label} exceeds its size bound")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        actual = os.fstat(descriptor)
        if (
            not stat.S_ISREG(actual.st_mode)
            or actual.st_dev != details.st_dev
            or actual.st_ino != details.st_ino
            or actual.st_size != details.st_size
        ):
            raise SemanticLockboxError(f"{label} changed while opening")
        payload = os.read(descriptor, maximum + 1)
    finally:
        os.close(descriptor)
    if len(payload) != details.st_size or len(payload) > maximum:
        raise SemanticLockboxError(f"{label} changed or exceeds its size bound")
    return payload


def _read_json(path: str, label: str) -> Any:
    payload = _read_regular_file(path, maximum=_MAX_JSON_BYTES, label=label)
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticLockboxError(f"{label} is not valid JSON") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify independent gold/runtime signatures, consume one external "
            "registration, and write an aggregate-only semantic safety report."
        )
    )
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--runtime-ledger", required=True)
    parser.add_argument("--gold-attestation", required=True)
    parser.add_argument("--runtime-attestation", required=True)
    parser.add_argument("--trusted-gold-public-key", required=True)
    parser.add_argument("--trusted-runtime-public-key", required=True)
    parser.add_argument("--consumption-ledger", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = score_semantic_lockbox(
            _read_json(args.inputs, "inputs"),
            _read_json(args.gold, "gold"),
            _read_json(args.runtime_ledger, "runtime ledger"),
            _read_json(args.gold_attestation, "gold attestation"),
            _read_json(args.runtime_attestation, "runtime attestation"),
            trusted_gold_public_key_pem=_read_regular_file(
                args.trusted_gold_public_key,
                maximum=_MAX_KEY_BYTES,
                label="trusted gold public key",
            ),
            trusted_runtime_public_key_pem=_read_regular_file(
                args.trusted_runtime_public_key,
                maximum=_MAX_KEY_BYTES,
                label="trusted runtime public key",
            ),
            consumption_ledger_dir=args.consumption_ledger,
        )
        write_json(args.output, report)
    except (OSError, SemanticLockboxError) as exc:
        print(json.dumps({"passed": False, "error": type(exc).__name__}))
        return 2
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "content_sha256": report["content_sha256"],
                "claim_boundary": report["claim_boundary"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
