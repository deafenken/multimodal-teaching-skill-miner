"""Attach hash-matched visual-semantic results to a longform dataset."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .io_utils import read_json, write_json
from .longform_multimodal import (
    attach_visual_semantic_results,
    file_sha256,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach hash-matched GPU visual features to a full-video dataset."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--semantic-results-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def execute_visual_semantic_apply(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = read_json(manifest_path)
    updated = copy.deepcopy(manifest)
    completed = 0
    for item in updated.get("videos", []):
        video_id = str(item["video_id"])
        transcript_path = root / str(item["transcript_path"])
        semantic_path = (
            args.semantic_results_dir.resolve()
            / f"{video_id}.visual_semantics.json"
        )
        enriched = attach_visual_semantic_results(
            read_json(transcript_path), read_json(semantic_path)
        )
        target = transcript_path.with_name("enriched_transcript.semantic.json")
        write_json(target, enriched)
        analysis_target = transcript_path.with_name("analysis.semantic.json")
        write_json(analysis_target, enriched["multimodal"])
        item["transcript_path"] = str(target.relative_to(root))
        item["analysis_path"] = str(analysis_target.relative_to(root))
        try:
            item["semantic_result_path"] = str(semantic_path.relative_to(root))
        except ValueError as exc:
            raise ValueError(
                "semantic results must be stored inside the dataset root"
            ) from exc
        item["semantic_result_sha256"] = file_sha256(semantic_path)
        item["summary"]["semantic_features_status"] = (
            "complete_hash_bound_inference"
        )
        completed += 1
    updated["aggregate"].pop("gpu_semantic_feature_complete_count", None)
    updated["aggregate"]["semantic_feature_complete_count"] = completed
    updated["claim_boundary"]["visual_semantic_features_complete"] = bool(
        completed == updated.get("video_count") and completed > 0
    )
    updated["claim_boundary"]["recognition_accuracy_established"] = False
    updated["claim_boundary"]["causal_multimodal_gain_established"] = False
    write_json(args.output_manifest, updated)
    print(args.output_manifest.resolve())
    print(f"hash-bound visual semantic features attached: {completed}")
    print("Recognition accuracy remains unestablished without independent labels.")
    return 0


def main(argv: list[str] | None = None) -> int:
    return execute_visual_semantic_apply(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
