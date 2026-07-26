#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.longform_multimodal import (  # noqa: E402
    process_longform_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable whole-timeline audio/frame/OCR/event analysis for a "
            "formal-caption full-video dataset."
        )
    )
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--transcript-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-seconds", type=float, default=300.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--frame-interval", type=float, default=15.0)
    parser.add_argument("--scene-threshold", type=float, default=0.32)
    parser.add_argument("--max-scenes-per-chunk", type=int, default=12)
    parser.add_argument("--ocr-workers", type=int, default=4)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = process_longform_dataset(
        args.media_manifest,
        args.transcript_manifest,
        args.output_dir,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        frame_interval_seconds=args.frame_interval,
        scene_threshold=args.scene_threshold,
        max_scene_frames_per_chunk=args.max_scenes_per_chunk,
        use_ocr=not args.no_ocr,
        ocr_workers=args.ocr_workers,
        resume=not args.no_resume,
    )
    summary = {
        "manifest": str((args.output_dir / "dataset_manifest.json").resolve()),
        "video_count": report["video_count"],
        **report["aggregate"],
        "recognition_accuracy_established": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if (
        report["aggregate"]["full_timeline_sampling_passed_count"]
        == report["video_count"]
        and report["aggregate"][
            "caption_timeline_media_binding_verified_count"
        ]
        == report["video_count"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())

