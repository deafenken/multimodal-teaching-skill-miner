#!/usr/bin/env python3
"""Prepare or analyze private, blind TeachObs double-annotation assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teaching_skill_miner.io_utils import read_json, write_json  # noqa: E402
from teaching_skill_miner.teachobs_human_annotation import (  # noqa: E402
    analyze_teachobs_double_annotations,
    build_completed_public_teachobs_annotation_receipt,
    build_pending_public_teachobs_annotation_receipt,
    prepare_teachobs_double_annotation,
)


def _prepare(args: argparse.Namespace) -> int:
    manifest = prepare_teachobs_double_annotation(
        args.repository,
        args.output,
        lesson_ids=args.lesson_id,
        media_root=args.media_root,
        media_reference_prefix=args.media_reference_prefix,
        operational_codebook_path=args.operational_codebook,
        seed_a=args.seed_a,
        seed_b=args.seed_b,
        require_media=args.require_media,
    )
    if args.public_receipt:
        write_json(
            args.public_receipt,
            build_pending_public_teachobs_annotation_receipt(manifest),
        )
    print(
        json.dumps(
            {
                "private_output": str(Path(args.output).resolve()),
                "selected_lesson_count": manifest["selection"]["lesson_count"],
                "assigned_scene_count": manifest["selection"]["scene_item_count"],
                "code_count": len(manifest["codes"]),
                "assignment_orders_differ": manifest["blindness"][
                    "assignment_orders_differ"
                ],
                "operational_definitions_complete": manifest[
                    "operational_codebook"
                ]["operational_definitions_complete"],
                "annotation_execution_ready": manifest["operational_codebook"][
                    "annotation_execution_ready"
                ],
                "gold_predictions_or_transcript_text_included": False,
                "media_bytes_copied": False,
                "human_completion": False,
                "inter_rater_reliability_computed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _analyze(args: argparse.Namespace) -> int:
    result = analyze_teachobs_double_annotations(
        args.manifest,
        args.assignment_a,
        args.assignment_b,
        args.output,
    )
    manifest = read_json(args.manifest)
    if args.public_receipt:
        write_json(
            args.public_receipt,
            build_completed_public_teachobs_annotation_receipt(
                manifest, result["report"]
            ),
        )
    overall = result["report"]["overall"]
    print(
        json.dumps(
            {
                "private_output": str(Path(args.output).resolve()),
                "validated_annotator_count": 2,
                "human_completion": True,
                "completion_basis": (
                    "complete binary labels and two distinct signed self-declarations"
                ),
                "human_identity_independently_verified": False,
                "macro_label_cohen_kappa": overall[
                    "macro_label_cohen_kappa"
                ],
                "pooled_binary_cohen_kappa": overall["pooled_binary"][
                    "cohen_kappa"
                ],
                "disagreement_count": overall["disagreement_count"],
                "adjudication_completed": False,
                "recognition_accuracy_established": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and validate two independent TeachObs scene-level human "
            "annotation assignments without reading gold labels or predictions."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="generate private blind A/B assignment CSV files"
    )
    prepare.add_argument(
        "--repository",
        default="artifacts/private/external_datasets/teachobs/repository",
    )
    prepare.add_argument(
        "--output",
        default="artifacts/private/external_datasets/teachobs/human_annotation",
    )
    prepare.add_argument(
        "--public-receipt",
        default="artifacts/public/teachobs_human_annotation_receipt.json",
    )
    prepare.add_argument(
        "--lesson-id",
        action="append",
        help="optional repeatable S1-S30 subset; default uses all 5158 scenes",
    )
    prepare.add_argument(
        "--media-root",
        default="artifacts/private/external_datasets/teachobs/media",
        help="private base directory; only relative references are emitted",
    )
    prepare.add_argument("--media-reference-prefix", default="videos")
    prepare.add_argument(
        "--operational-codebook",
        help=(
            "external 39/39 operational JSON codebook; without it assignments are "
            "a non-executable template and human completion cannot be imported"
        ),
    )
    prepare.add_argument("--seed-a", type=int, default=1729)
    prepare.add_argument("--seed-b", type=int, default=2718)
    prepare.add_argument(
        "--require-media",
        action="store_true",
        help="fail unless every selected private videos/S*.mp4 exists",
    )
    prepare.set_defaults(func=_prepare)

    analyze = subparsers.add_parser(
        "analyze",
        help="validate two completed CSV files and compute agreement/adjudication",
    )
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--assignment-a", required=True)
    analyze.add_argument("--assignment-b", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--public-receipt")
    analyze.set_defaults(func=_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
