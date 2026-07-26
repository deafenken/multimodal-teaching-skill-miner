#!/usr/bin/env python3
"""Download and validate the 10 complete MIT OCW research videos privately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data/formal_caption_sources.json",
    )
    parser.add_argument(
        "--formal-caption-manifest",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "artifacts/private/formal_captions/dataset_manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/private/full_videos",
    )
    parser.add_argument("--curl", default="curl")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--connect-timeout", type=int, default=30)
    parser.add_argument("--download-timeout", type=int, default=14_400)
    parser.add_argument("--ffprobe-timeout", type=int, default=180)
    parser.add_argument(
        "--public-receipt",
        type=Path,
        help="optional content-free public validation receipt",
    )
    parser.add_argument(
        "--acknowledge-source-terms",
        action="store_true",
        help=(
            "confirm that MIT OCW media retains its upstream license and that "
            "the complete videos will remain private"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from teaching_skill_miner.full_video_dataset import (
        build_public_full_video_receipt,
        download_full_video_dataset,
    )
    from teaching_skill_miner.io_utils import write_json

    parser = _parser()
    args = parser.parse_args(argv)
    if not args.acknowledge_source_terms:
        parser.error(
            "--acknowledge-source-terms is required; complete MIT OCW media "
            "retains its upstream license"
        )
    result = download_full_video_dataset(
        args.source_manifest,
        args.formal_caption_manifest,
        args.output,
        acknowledge_source_terms=True,
        curl_command=args.curl,
        ffprobe_command=args.ffprobe,
        connect_timeout_seconds=args.connect_timeout,
        download_timeout_seconds=args.download_timeout,
        ffprobe_timeout_seconds=args.ffprobe_timeout,
        progress=lambda message: print(f"verified: {message}", flush=True),
    )
    if args.public_receipt:
        write_json(
            args.public_receipt,
            build_public_full_video_receipt(result["manifest"]),
        )
    summary = {
        "output_directory": result["output_directory"],
        "media_manifest": result["manifest_path"],
        "private_receipt": result["receipt_path"],
        "complete": result["manifest"]["complete"],
        "video_count": result["manifest"]["video_count"],
        "total_media_bytes": result["receipt"]["total_media_bytes"],
        "raw_media_publicly_exported": False,
        "publisher_media_hashes_pinned": False,
        "public_receipt": (
            str(args.public_receipt.resolve()) if args.public_receipt else None
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
