#!/usr/bin/env python3
"""Prepare private, hash-bound TeachObs source media and scene features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.io_utils import (  # noqa: E402
    ensure_private_directory,
    read_json,
    write_json,
)
from teaching_skill_miner.teachobs_media import (  # noqa: E402
    build_teachobs_media_plan,
    download_teachobs_media,
    extract_teachobs_multimodal_features,
    validate_teachobs_cookies_from_browser,
    validate_teachobs_ytdlp_transport,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        default="artifacts/private/external_datasets/teachobs/repository",
    )
    parser.add_argument(
        "--output",
        default="artifacts/private/external_datasets/teachobs/media",
    )
    parser.add_argument(
        "--stage",
        choices=("plan", "download", "features", "all"),
        default="all",
    )
    parser.add_argument("--lesson-id", action="append", dest="lesson_ids")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--feature-jobs", type=int, default=2)
    parser.add_argument("--visual-jobs", type=int, default=1)
    parser.add_argument("--ocr-jobs", type=int, default=1)
    parser.add_argument("--yt-dlp-python", default="python3")
    parser.add_argument("--js-runtime", default="node:/opt/homebrew/bin/node")
    parser.add_argument(
        "--cookies-from-browser",
        help=(
            "explicit browser-cookie opt-in, e.g. chrome or "
            "'chrome:Profile 1'; profile paths are rejected"
        ),
    )
    parser.add_argument(
        "--yt-dlp-direct",
        action="store_true",
        help="explicitly bypass inherited proxy variables with yt-dlp --proxy ''",
    )
    parser.add_argument(
        "--yt-dlp-impersonate",
        choices=("chrome",),
        help="request the local yt-dlp/curl_cffi Chrome HTTP impersonation backend",
    )
    parser.add_argument("--source-override-manifest")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--no-audio-statistics", action="store_true")
    parser.add_argument("--clip-model")
    parser.add_argument("--clip-source-revision")
    parser.add_argument("--clip-device", default="cpu")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--acknowledge-source-terms", action="store_true")
    parser.add_argument(
        "--acknowledge-override-source-terms", action="store_true"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_teachobs_cookies_from_browser(args.cookies_from_browser)
    validate_teachobs_ytdlp_transport(
        direct=args.yt_dlp_direct,
        impersonate=args.yt_dlp_impersonate,
    )
    output = ensure_private_directory(args.output)
    plan = build_teachobs_media_plan(
        args.repository,
        lesson_ids=args.lesson_ids,
    )
    write_json(output / "media_plan.json", plan)
    if args.stage == "plan":
        print(
            json.dumps(
                {
                    "stage": "plan",
                    "lesson_count": plan["lesson_count"],
                    "scene_count": plan["scene_count"],
                    "duration_hours": round(
                        plan["reference_duration_seconds"] / 3600, 6
                    ),
                    "source_terms_acknowledged": args.acknowledge_source_terms,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not args.acknowledge_source_terms:
        raise ValueError(
            "media execution requires --acknowledge-source-terms; source-video "
            "rights and platform terms are separate from CC BY 4.0 annotations"
        )

    if args.stage in {"download", "all"}:
        media_result = download_teachobs_media(
            plan,
            output,
            acknowledge_source_terms=True,
            source_override_manifest_path=args.source_override_manifest,
            acknowledge_override_source_terms=(
                args.acknowledge_override_source_terms
            ),
            yt_dlp_command=[args.yt_dlp_python, "-m", "yt_dlp"],
            js_runtime=args.js_runtime,
            cookies_from_browser=args.cookies_from_browser,
            yt_dlp_direct=args.yt_dlp_direct,
            yt_dlp_impersonate=args.yt_dlp_impersonate,
            ffprobe_command=args.ffprobe,
            jobs=args.jobs,
        )
        media_manifest = media_result["manifest"]
    else:
        media_manifest = read_json(output / "media_manifest.json")

    if args.stage in {"features", "all"}:
        feature_result = extract_teachobs_multimodal_features(
            plan,
            media_manifest,
            output,
            acknowledge_source_terms=True,
            include_audio_statistics=not args.no_audio_statistics,
            clip_model=args.clip_model,
            clip_source_revision=args.clip_source_revision,
            clip_device=args.clip_device,
            clip_batch_size=args.clip_batch_size,
            ffmpeg_command=args.ffmpeg,
            feature_jobs=args.feature_jobs,
            visual_jobs=args.visual_jobs,
            ocr_jobs=args.ocr_jobs,
        )
        feature_manifest = feature_result["manifest"]
    else:
        feature_manifest = None

    print(
        json.dumps(
            {
                "stage": args.stage,
                "private_output": str(output.resolve()),
                "downloaded_lesson_count": media_manifest[
                    "downloaded_lesson_count_total"
                ],
                "selected_media_complete": media_manifest["selected_complete"],
                "feature_lesson_count": (
                    feature_manifest["lesson_count"] if feature_manifest else 0
                ),
                "features_complete": (
                    feature_manifest["complete"] if feature_manifest else False
                ),
                "public_release_authorized": False,
                "recognition_accuracy_established": False,
                "multimodal_gain_established": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
