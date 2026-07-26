"""Batch orchestration for hash-bound visual-semantic extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_private_json(path: Path, value: Any) -> None:
    from .io_utils import write_json

    write_json(path, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hash-bound CLIP extraction for every lecture in a dataset."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def execute_visual_semantic_dataset(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    results = []
    extractor = Path(__file__).with_name("visual_semantics.py")
    for index, item in enumerate(manifest.get("videos", []), 1):
        video_id = str(item["video_id"])
        task_path = manifest_root / str(item["semantic_task_path"])
        lecture_root = task_path.parent
        output = args.output_dir / f"{video_id}.visual_semantics.json"
        print(
            f"[{index}/{manifest['video_count']}] CLIP visual semantics: {video_id}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(extractor),
                "--tasks",
                str(task_path),
                "--frame-root",
                str(lecture_root),
                "--output",
                str(output),
                "--model",
                str(args.model.resolve()),
                "--source-model-id",
                args.source_model_id,
                "--source-revision",
                args.source_revision,
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
            ],
            check=True,
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        results.append(
            {
                "video_id": video_id,
                "frame_count": result["frame_count"],
                "result_path": output.name,
                "result_sha256": _sha256(output),
                "weight_manifest_sha256": result["model_provenance"][
                    "weight_manifest"
                ]["manifest_sha256"],
            }
        )
    receipt = {
        "artifact_kind": "private_visual_semantic_batch_receipt",
        "schema_version": "1.0",
        "dataset_manifest_sha256": _sha256(manifest_path),
        "model_path": str(args.model.resolve()),
        "source_model_id": args.source_model_id,
        "source_revision": args.source_revision,
        "device": args.device,
        "video_count": len(results),
        "frame_count": sum(item["frame_count"] for item in results),
        "complete": len(results) == manifest.get("video_count"),
        "results": results,
        "claim_boundary": {
            "visual_semantic_features_computed": True,
            "recognition_accuracy_established": False,
            "human_ground_truth_used": False,
        },
    }
    _write_private_json(args.output_dir / "semantic_batch_receipt.json", receipt)
    return 0 if receipt["complete"] else 2


def main(argv: list[str] | None = None) -> int:
    return execute_visual_semantic_dataset(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
