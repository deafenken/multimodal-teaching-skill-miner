#!/usr/bin/env python3
"""Offline GPU worker for a hash-bound TeachObs ASR handoff manifest."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner import teachobs_asr_handoff  # noqa: E402
from teaching_skill_miner.teachobs_asr_handoff import (  # noqa: E402
    model_directory_sha256,
    run_teachobs_asr_gpu_jobs,
)


def _runner_source_sha256() -> str:
    return sha256(Path(teachobs_asr_handoff.__file__).resolve().read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hash a prepositioned Whisper snapshot or execute private TeachObs "
            "ASR jobs on CUDA without downloading models or media"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    hash_parser = subparsers.add_parser(
        "hash-model", help="print the deterministic local model-tree SHA-256"
    )
    hash_parser.add_argument("--model-directory", required=True)

    run_parser = subparsers.add_parser(
        "run", help="execute selected hash-bound jobs with local faster-whisper"
    )
    run_parser.add_argument("--job-manifest", required=True)
    run_parser.add_argument("--media-root", required=True)
    run_parser.add_argument("--model-directory", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument(
        "--container-image-digest",
        required=True,
        help=(
            "sha256:<64-hex> digest that must exactly match the frozen job "
            "manifest runtime contract"
        ),
    )
    run_parser.add_argument(
        "--lesson-id", action="append", help="optional repeatable private job subset"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hash-model":
        print(model_directory_sha256(args.model_directory))
        return 0
    summary = run_teachobs_asr_gpu_jobs(
        args.job_manifest,
        args.media_root,
        args.model_directory,
        args.output,
        container_image_digest=args.container_image_digest,
        runner_source_sha256=_runner_source_sha256(),
        lesson_ids=args.lesson_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
