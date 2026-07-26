"""Hash-bound CLIP visual semantics with no import-time ML dependency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any


SCHEMA_TASK = "teaching_skill_miner.visual_semantic_tasks.v1"
SCHEMA_RESULT = "teaching_skill_miner.visual_semantic_results.v1"

ONTOLOGY: dict[str, tuple[str, ...]] = {
    "handwritten_blackboard": (
        "a teacher's handwritten chalkboard or whiteboard full of lesson content",
        "handwritten equations or notes being developed on a classroom board",
        "a lecture frame dominated by handwriting on a blackboard or whiteboard",
    ),
    "presentation_slide": (
        "a projected lecture slide with typed educational text",
        "a presentation slide shown during a university lecture",
        "a classroom screen displaying a structured presentation slide",
    ),
    "programming_code": (
        "computer programming source code displayed during a lesson",
        "a code editor or terminal with programming code",
        "a lecture frame showing Python or another programming language",
    ),
    "mathematical_formula": (
        "mathematical formulas and equations shown in a lecture",
        "a university lesson frame centered on symbolic mathematics",
        "an equation or mathematical derivation being explained",
    ),
    "diagram_or_graph": (
        "an educational diagram, chart, plot, or geometric drawing",
        "a teacher explaining a graph or visual diagram",
        "a lecture frame containing a schematic, graph, or plotted figure",
    ),
    "instructor_talking": (
        "a university instructor speaking to the class without a prominent board",
        "a lecturer talking in front of students",
        "a teacher-centered classroom camera view",
    ),
    "classroom_wide_view": (
        "a wide view of a university classroom with students and instructor",
        "an audience view inside a lecture hall",
        "a classroom-wide camera shot during a lesson",
    ),
    "other_lecture_visual": (
        "another kind of educational lecture visual",
        "a university lecture frame not described by the other categories",
        "miscellaneous visual content from a recorded class",
    ),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_task_manifest(task: dict[str, Any], frame_root: Path) -> list[dict[str, Any]]:
    if task.get("schema") != SCHEMA_TASK:
        raise ValueError("unsupported semantic task schema")
    frames = task.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("semantic task requires at least one frame")
    if int(task.get("frame_count", -1)) != len(frames):
        raise ValueError("semantic task frame_count does not match frames")
    root = frame_root.resolve()
    checked: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, row in enumerate(frames):
        frame_id = str(row.get("frame_id", ""))
        digest = str(row.get("sha256", ""))
        if not frame_id or frame_id in ids:
            raise ValueError(f"duplicate or empty frame_id at index {index}")
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError(f"invalid frame SHA-256 at index {index}")
        candidate = (root / str(row.get("path", ""))).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("frame path escapes the declared frame root") from exc
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        actual = _file_sha256(candidate)
        if actual != digest:
            raise ValueError(f"frame SHA-256 mismatch for {frame_id}")
        ids.add(frame_id)
        checked.append({**row, "absolute_path": str(candidate)})
    return checked


def _resolve_snapshot(model_id_or_path: str, revision: str | None) -> Path:
    direct = Path(model_id_or_path)
    if direct.is_dir():
        return direct.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_id_or_path,
            revision=revision,
            local_files_only=True,
        )
    ).resolve()


def _weight_manifest(snapshot: Path) -> dict[str, Any]:
    suffixes = {".bin", ".safetensors"}
    weights = [
        path
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    ]
    if not weights:
        raise ValueError(f"no model weight file found in {snapshot}")
    rows = [
        {
            "path": str(path.relative_to(snapshot)),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in weights
    ]
    return {
        "files": rows,
        "manifest_sha256": _canonical_sha256(rows),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _projected_feature_tensor(output: Any) -> Any:
    """Return projected CLIP features across Transformers 4.x and 5.x APIs."""

    if hasattr(output, "norm"):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        return output[1]
    raise TypeError(
        "CLIP feature output has neither tensor operations nor pooler_output"
    )


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import PIL
        from PIL import Image
        import torch
        import transformers
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "visual-semantic inference dependencies are unavailable; "
            "install teaching-skill-miner[visual]"
        ) from exc

    task = json.loads(args.tasks.read_text(encoding="utf-8"))
    checked = validate_task_manifest(task, args.frame_root)
    snapshot = _resolve_snapshot(args.model, args.revision)
    weights = _weight_manifest(snapshot)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = CLIPProcessor.from_pretrained(
        snapshot, local_files_only=True, use_fast=False
    )
    model = CLIPModel.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=dtype,
    ).eval().to(device)

    labels = list(ONTOLOGY)
    prompts = [prompt for label in labels for prompt in ONTOLOGY[label]]
    counts = [len(ONTOLOGY[label]) for label in labels]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    with torch.inference_mode():
        text_features = _projected_feature_tensor(
            model.get_text_features(**text_inputs)
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        class_features = []
        cursor = 0
        for count in counts:
            value = text_features[cursor : cursor + count].mean(dim=0)
            class_features.append(value / value.norm())
            cursor += count
        class_matrix = torch.stack(class_features)

    rows: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    for offset in range(0, len(checked), batch_size):
        batch = checked[offset : offset + batch_size]
        images = []
        for row in batch:
            with Image.open(row["absolute_path"]) as opened:
                images.append(opened.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        with torch.inference_mode():
            embeddings = _projected_feature_tensor(
                model.get_image_features(pixel_values=pixel_values)
            )
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp().clamp(max=100)
            logits = logit_scale * embeddings @ class_matrix.T
            scores = logits.float().softmax(dim=-1)
        embeddings_cpu = embeddings.float().cpu()
        scores_cpu = scores.cpu()
        for row, embedding, score in zip(batch, embeddings_cpu, scores_cpu):
            embedding_values = [round(float(value), 7) for value in embedding.tolist()]
            score_values = [round(float(value), 7) for value in score.tolist()]
            order = sorted(range(len(labels)), key=lambda index: score_values[index], reverse=True)
            top = order[0]
            runner_up = order[1]
            rows.append(
                {
                    "frame_id": row["frame_id"],
                    "timestamp": row["timestamp"],
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "top_label": labels[top],
                    "top_relative_score": score_values[top],
                    "score_margin": round(score_values[top] - score_values[runner_up], 7),
                    "relative_prompt_scores": {
                        label: score_values[index] for index, label in enumerate(labels)
                    },
                    "embedding": embedding_values,
                    "embedding_sha256": _canonical_sha256(embedding_values),
                }
            )
        del inputs, pixel_values, embeddings, scores

    backend = "transformers.CLIPModel.zero_shot_closed_ontology"
    return {
        "schema": SCHEMA_RESULT,
        "video_id": task["video_id"],
        "media_sha256": task["media_sha256"],
        "task_manifest_sha256": _canonical_sha256(task),
        "frame_count": len(rows),
        "backend": backend,
        "model_provenance": {
            "model_id": args.source_model_id or args.model,
            "requested_revision": args.source_revision or args.revision,
            "loaded_model_path": str(snapshot),
            "resolved_snapshot_name": snapshot.name,
            "weight_manifest": weights,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "pillow_version": PIL.__version__,
            "python_version": platform.python_version(),
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "inference_dtype": str(dtype),
            "batch_size": batch_size,
        },
        "ontology": {
            "labels": labels,
            "prompts": {key: list(value) for key, value in ONTOLOGY.items()},
            "closed_set_relative_softmax": True,
            "scores_are_calibrated_probabilities": False,
            "human_ground_truth_used": False,
        },
        "privacy": {
            "face_recognition_performed": False,
            "identity_recognition_performed": False,
        },
        "frames": rows,
        "claim_boundary": {
            "visual_semantic_features_computed": True,
            "recognition_accuracy_established": False,
            "calibrated_class_probability_established": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract hash-bound CLIP visual embeddings and closed-ontology relative "
            "prompt scores. This does not estimate recognition accuracy."
        )
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--revision")
    parser.add_argument("--source-model-id")
    parser.add_argument("--source-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def execute_visual_semantic_extraction(args: argparse.Namespace) -> int:
    result = run_inference(args)
    _write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "video_id": result["video_id"],
                "frame_count": result["frame_count"],
                "model_revision": result["model_provenance"][
                    "resolved_snapshot_name"
                ],
                "gpu_name": result["model_provenance"]["gpu_name"],
                "accuracy_established": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return execute_visual_semantic_extraction(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
