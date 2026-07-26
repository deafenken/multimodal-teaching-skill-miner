#!/usr/bin/env python3
"""Fit, validate, and export the private TeachObs four-arm frozen bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.teachobs_multimodal_benchmark import (  # noqa: E402
    run_teachobs_multimodal_benchmark,
)


def _write_private_result(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("private result output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    path.write_text(payload + "\n", encoding="utf-8")
    path.chmod(0o600)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the fixed TeachObs four-arm exploratory benchmark and export "
            "deterministic JSON+NPZ frozen models without pickle. Export is not "
            "deployment or confirmatory lockbox evidence."
        )
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument(
        "--transcript-materialization-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-result", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_teachobs_multimodal_benchmark(
        args.repository,
        args.feature_manifest,
        transcript_materialization_manifest_path=(
            args.transcript_materialization_manifest
        ),
        frozen_model_output=args.output,
    )
    if args.private_result is not None:
        _write_private_result(args.private_result, result)
    print(
        json.dumps(
            {
                **result["frozen_model_export"],
                "exploratory_public_test_context": True,
                "confirmatory_lockbox_result_established": False,
                "deployment_accuracy_established": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
