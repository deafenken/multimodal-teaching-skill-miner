from __future__ import annotations

import concurrent.futures
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Iterable
from urllib.parse import urlparse

from .io_utils import (
    ensure_private_directory,
    ensure_private_file,
    read_json,
    write_json,
)
from .models import validate_transcript
from .multimodal import (
    _available_tesseract_language,
    _tool_version,
    fuse_multimodal_events,
    infer_visual_events,
    parse_silencedetect,
    probe_media,
)


PIPELINE_VERSION = "teaching_skill_miner.longform_multimodal.v9"
EXTRACTION_VERSION = "teaching_skill_miner.longform_extraction.v2"
SEMANTIC_TASK_SCHEMA = "teaching_skill_miner.visual_semantic_tasks.v1"
SEMANTIC_RESULT_SCHEMA = "teaching_skill_miner.visual_semantic_results.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_archive_media_url(value: str) -> str:
    parsed = urlparse(str(value).strip())
    if parsed.hostname not in {"archive.org", "www.archive.org"}:
        raise ValueError("media URL must use an approved Internet Archive host")
    if parsed.username or parsed.password or parsed.scheme not in {"http", "https"}:
        raise ValueError("unsafe media URL")
    return parsed._replace(scheme="https", netloc="archive.org").geturl()


def plan_chunks(
    duration_seconds: float,
    *,
    chunk_seconds: float = 300.0,
    overlap_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    """Plan contiguous nominal chunks with bounded decode overlap.

    Nominal ranges partition the whole timeline without gaps. Decode ranges add a
    small overlap so scene/audio events at boundaries are not lost; records are
    clipped back to the nominal range before the chunks are merged.
    """

    duration = float(duration_seconds)
    chunk = float(chunk_seconds)
    overlap = float(overlap_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    if not math.isfinite(chunk) or chunk <= 0:
        raise ValueError("chunk_seconds must be positive and finite")
    if not math.isfinite(overlap) or overlap < 0 or overlap >= chunk / 2:
        raise ValueError("overlap_seconds must satisfy 0 <= overlap < chunk_seconds/2")
    count = int(math.ceil(duration / chunk))
    result: list[dict[str, Any]] = []
    for index in range(count):
        nominal_start = index * chunk
        nominal_end = min(duration, (index + 1) * chunk)
        decode_start = max(0.0, nominal_start - overlap)
        decode_end = min(duration, nominal_end + overlap)
        result.append(
            {
                "chunk_id": f"chunk_{index:04d}",
                "index": index,
                "nominal_start": round(nominal_start, 6),
                "nominal_end": round(nominal_end, 6),
                "decode_start": round(decode_start, 6),
                "decode_end": round(decode_end, 6),
            }
        )
    return result


def _frame_filter_extract(
    video_path: Path,
    frame_dir: Path,
    *,
    prefix: str,
    video_filter: str,
    maximum_frames: int | None,
    decode_start: float,
    decode_end: float,
) -> list[dict[str, Any]]:
    ffmpeg = os.getenv("TSM_FFMPEG", "ffmpeg")
    if not shutil.which(ffmpeg):
        raise RuntimeError(f"long-form analysis requires `{ffmpeg}` on PATH")
    pattern = frame_dir / f"{prefix}_%04d.jpg"
    for stale in frame_dir.glob(f"{prefix}_*.jpg"):
        stale.unlink()
    duration = max(0.001, decode_end - decode_start)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "info",
        "-y",
        "-ss",
        f"{decode_start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(video_path),
        "-vf",
        f"{video_filter},scale=960:-2,showinfo",
    ]
    if maximum_frames is not None:
        command.extend(["-frames:v", str(maximum_frames)])
    command.extend(
        [
            "-fps_mode",
            "vfr",
            "-q:v",
            "3",
            str(pattern),
        ]
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr[-1200:]
        raise RuntimeError(f"ffmpeg frame extraction failed for {prefix}: {detail}")
    local_timestamps = [
        float(value) for value in re.findall(r"pts_time:([-0-9.]+)", result.stderr)
    ]
    paths = sorted(frame_dir.glob(f"{prefix}_*.jpg"))
    if len(paths) != len(local_timestamps):
        raise RuntimeError(
            f"ffmpeg frame/timestamp mismatch for {prefix}: "
            f"{len(paths)} files vs {len(local_timestamps)} timestamps"
        )
    frames: list[dict[str, Any]] = []
    for frame_path, local_timestamp in zip(paths, local_timestamps):
        # Input seeking normally resets PTS to approximately zero. Some FFmpeg
        # builds preserve input PTS; distinguish the cases instead of offsetting twice.
        global_timestamp = (
            decode_start + local_timestamp
            if local_timestamp <= duration + 2.0
            else local_timestamp
        )
        ensure_private_file(frame_path)
        frames.append(
            {
                "timestamp": round(max(0.0, global_timestamp), 3),
                "absolute_path": str(frame_path),
                "sampling_source": prefix,
            }
        )
    return frames


def _dhash_and_image_metrics(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return {
            "backend": "unavailable",
            "reason": "Pillow is not installed",
            "dhash64": None,
        }
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        small_rgb = rgb.resize((32, 32))
        stat = ImageStat.Stat(small_rgb)
        grayscale = rgb.convert("L").resize((32, 32))
        gray_values = list(grayscale.getdata())
        horizontal = [
            abs(gray_values[row * 32 + column] - gray_values[row * 32 + column - 1])
            for row in range(32)
            for column in range(1, 32)
        ]
        vertical = [
            abs(gray_values[row * 32 + column] - gray_values[(row - 1) * 32 + column])
            for row in range(1, 32)
            for column in range(32)
        ]
        dhash_image = rgb.convert("L").resize((9, 8))
        dhash_values = list(dhash_image.getdata())
        bits = [
            dhash_values[row * 9 + column] > dhash_values[row * 9 + column + 1]
            for row in range(8)
            for column in range(8)
        ]
        dhash_value = sum(int(bit) << index for index, bit in enumerate(bits))
        return {
            "backend": "Pillow",
            "width": width,
            "height": height,
            "rgb_mean": [round(value, 3) for value in stat.mean],
            "rgb_stddev": [round(value, 3) for value in stat.stddev],
            "luminance_mean": round(sum(gray_values) / len(gray_values), 3),
            "edge_difference_mean": round(
                (sum(horizontal) + sum(vertical))
                / max(1, len(horizontal) + len(vertical)),
                3,
            ),
            "dark_pixel_fraction": round(
                sum(value < 48 for value in gray_values) / len(gray_values), 6
            ),
            "bright_pixel_fraction": round(
                sum(value > 208 for value in gray_values) / len(gray_values), 6
            ),
            "dhash64": f"{dhash_value:016x}",
        }


def _ocr_frame_audited(
    path: Path,
    language: str | None,
    *,
    minimum_word_confidence: float = 35.0,
) -> tuple[str, dict[str, Any]]:
    tesseract = os.getenv("TSM_TESSERACT", "tesseract")
    if language is None or not shutil.which(tesseract):
        return "", {
            "status": "unavailable",
            "language": language,
            "minimum_word_confidence": minimum_word_confidence,
            "raw_word_count": 0,
            "accepted_word_count": 0,
            "accepted_mean_confidence": None,
            "confidence_is_calibrated_probability": False,
        }
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for page_segmentation_mode in (6, 11):
        result = subprocess.run(
            [
                tesseract,
                str(path),
                "stdout",
                "-l",
                language,
                "--psm",
                str(page_segmentation_mode),
                "tsv",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        # Tesseract TSV is tab-delimited but does not CSV-escape quote characters
        # recognized in code. Disabling CSV quote handling prevents one `"` token
        # from absorbing subsequent TSV rows into its text field.
        reader = csv.DictReader(
            io.StringIO(result.stdout),
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )
        raw_words: list[tuple[str, float]] = []
        for row in reader:
            text = re.sub(r"\s+", " ", str(row.get("text", ""))).strip()
            if not text:
                continue
            try:
                confidence = float(row.get("conf", -1))
            except (TypeError, ValueError):
                continue
            if confidence >= 0:
                raw_words.append((text, confidence))
        accepted = [
            (text, confidence)
            for text, confidence in raw_words
            if confidence >= minimum_word_confidence
        ]
        accepted_text = " ".join(text for text, _ in accepted)[:1500]
        accepted_mean = (
            sum(confidence for _, confidence in accepted) / len(accepted)
            if accepted
            else None
        )
        # Prefer candidates with more accepted evidence, then higher confidence.
        selection_score = len(accepted) * 100 + (accepted_mean or 0.0)
        audit = {
            "status": "completed",
            "backend": "tesseract_tsv",
            "language": language,
            "page_segmentation_mode": page_segmentation_mode,
            "minimum_word_confidence": minimum_word_confidence,
            "raw_word_count": len(raw_words),
            "accepted_word_count": len(accepted),
            "accepted_mean_confidence": (
                round(accepted_mean, 3) if accepted_mean is not None else None
            ),
            "confidence_is_calibrated_probability": False,
        }
        candidates.append((selection_score, accepted_text, audit))
    if not candidates:
        return "", {
            "status": "failed",
            "language": language,
            "minimum_word_confidence": minimum_word_confidence,
            "raw_word_count": 0,
            "accepted_word_count": 0,
            "accepted_mean_confidence": None,
            "confidence_is_calibrated_probability": False,
        }
    _, text, audit = max(candidates, key=lambda value: value[0])
    return text, audit


def _annotate_frame(
    frame: dict[str, Any],
    *,
    output_root: Path,
    ocr_language: str | None,
) -> dict[str, Any]:
    result = dict(frame)
    absolute_path = Path(str(result.pop("absolute_path")))
    result["path"] = str(absolute_path.relative_to(output_root))
    result["sha256"] = file_sha256(absolute_path)
    result["size_bytes"] = absolute_path.stat().st_size
    ocr_text, ocr_audit = _ocr_frame_audited(absolute_path, ocr_language)
    result["ocr_text"] = ocr_text
    result["ocr_language"] = ocr_language
    result["ocr_audit"] = ocr_audit
    result["interpretable_visual_features"] = _dhash_and_image_metrics(absolute_path)
    return result


def _detect_chunk_silences(
    video_path: Path,
    *,
    chunk: dict[str, Any],
    noise_db: float,
    minimum_duration: float,
) -> list[dict[str, float]]:
    ffmpeg = os.getenv("TSM_FFMPEG", "ffmpeg")
    decode_start = float(chunk["decode_start"])
    decode_end = float(chunk["decode_end"])
    duration = decode_end - decode_start
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-ss",
            f"{decode_start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(video_path),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={minimum_duration}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr[-1200:]
        raise RuntimeError(f"ffmpeg chunk silence detection failed: {detail}")
    nominal_start = float(chunk["nominal_start"])
    nominal_end = float(chunk["nominal_end"])
    records: list[dict[str, float]] = []
    for record in parse_silencedetect(result.stderr, duration):
        start = max(nominal_start, decode_start + float(record["start"]))
        end = min(nominal_end, decode_start + float(record["end"]))
        if end > start:
            records.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                }
            )
    return records


def _checkpoint_valid(
    checkpoint_path: Path,
    *,
    expected_fingerprint: str,
    output_root: Path,
) -> dict[str, Any] | None:
    if not checkpoint_path.is_file():
        return None
    try:
        checkpoint = read_json(checkpoint_path)
    except (OSError, ValueError, TypeError):
        return None
    if (
        checkpoint.get("status") != "completed"
        or checkpoint.get("chunk_fingerprint_sha256") != expected_fingerprint
    ):
        return None
    for frame in checkpoint.get("frames", []):
        frame_path = output_root / str(frame.get("path", ""))
        digest = str(frame.get("sha256", ""))
        if (
            not frame_path.is_file()
            or not SHA256_RE.fullmatch(digest)
            or file_sha256(frame_path) != digest
        ):
            return None
    return checkpoint


def _extract_chunk(
    video_path: Path,
    output_root: Path,
    job_dir: Path,
    *,
    chunk: dict[str, Any],
    job_fingerprint: str,
    language: str,
    frame_interval_seconds: float,
    scene_threshold: float,
    max_scene_frames_per_chunk: int,
    ocr_workers: int,
    use_ocr: bool,
    has_audio: bool,
    noise_db: float,
    minimum_silence_seconds: float,
    resume: bool,
) -> dict[str, Any]:
    chunk_dir = ensure_private_directory(job_dir / "chunks" / str(chunk["chunk_id"]))
    frame_dir = ensure_private_directory(chunk_dir / "frames")
    checkpoint_path = chunk_dir / "checkpoint.json"
    chunk_fingerprint = canonical_sha256(
        {
            "job_fingerprint_sha256": job_fingerprint,
            "chunk": chunk,
        }
    )
    if resume:
        cached = _checkpoint_valid(
            checkpoint_path,
            expected_fingerprint=chunk_fingerprint,
            output_root=output_root,
        )
        if cached is not None:
            cached["resumed_from_verified_checkpoint"] = True
            return cached

    started = time.monotonic()
    nominal_duration = float(chunk["nominal_end"]) - float(chunk["nominal_start"])
    expected_uniform = int(math.ceil(nominal_duration / frame_interval_seconds)) + 3
    uniform = _frame_filter_extract(
        video_path,
        frame_dir,
        prefix="uniform",
        video_filter=(
            f"fps=1/{frame_interval_seconds}:start_time=0:eof_action=pass"
        ),
        maximum_frames=max(2, expected_uniform),
        decode_start=float(chunk["nominal_start"]),
        decode_end=float(chunk["nominal_end"]),
    )
    scene = _frame_filter_extract(
        video_path,
        frame_dir,
        prefix="scene",
        video_filter=f"select='eq(n\\,0)+gt(scene\\,{scene_threshold})'",
        maximum_frames=None,
        decode_start=float(chunk["decode_start"]),
        decode_end=float(chunk["decode_end"]),
    )
    scene_candidate_count = len(scene)
    if len(scene) > max_scene_frames_per_chunk:
        if max_scene_frames_per_chunk == 1:
            retained_indexes = {len(scene) // 2}
        else:
            retained_indexes = {
                round(index * (len(scene) - 1) / (max_scene_frames_per_chunk - 1))
                for index in range(max_scene_frames_per_chunk)
            }
        retained_scene: list[dict[str, Any]] = []
        for index, frame in enumerate(scene):
            if index in retained_indexes:
                retained_scene.append(frame)
            else:
                Path(str(frame["absolute_path"])).unlink(missing_ok=True)
        scene = retained_scene
    nominal_start = float(chunk["nominal_start"])
    nominal_end = float(chunk["nominal_end"])
    is_last = math.isclose(nominal_end, float(chunk["decode_end"]), abs_tol=1e-6)
    candidates = [
        item
        for item in scene + uniform
        if item["timestamp"] >= nominal_start - 0.001
        and (
            item["timestamp"] < nominal_end - 0.001
            or (is_last and item["timestamp"] <= nominal_end + 0.001)
        )
    ]
    merged: list[dict[str, Any]] = []
    for frame in sorted(
        candidates,
        key=lambda item: (float(item["timestamp"]), item["sampling_source"] != "scene"),
    ):
        close = next(
            (
                existing
                for existing in reversed(merged[-3:])
                if abs(float(existing["timestamp"]) - float(frame["timestamp"])) < 0.4
            ),
            None,
        )
        if close is not None:
            combined_sources = {
                *str(close.get("sampling_source", "")).split("+"),
                *str(frame.get("sampling_source", "")).split("+"),
            }
            if {"scene", "uniform"} <= combined_sources:
                close["sampling_source"] = "scene+uniform"
            continue
        merged.append(frame)

    ocr_language = _available_tesseract_language(language) if use_ocr else None
    workers = max(1, int(ocr_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        frames = list(
            executor.map(
                lambda value: _annotate_frame(
                    value,
                    output_root=output_root,
                    ocr_language=ocr_language,
                ),
                merged,
            )
        )
    silences = (
        _detect_chunk_silences(
            video_path,
            chunk=chunk,
            noise_db=noise_db,
            minimum_duration=minimum_silence_seconds,
        )
        if has_audio
        else []
    )
    checkpoint = {
        "schema_version": "1.0",
        "status": "completed",
        "chunk": chunk,
        "chunk_fingerprint_sha256": chunk_fingerprint,
        "frames": frames,
        "scene_candidate_count": scene_candidate_count,
        "scene_retained_count": len(scene),
        "silences": silences,
        "ocr_language": ocr_language,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "resumed_from_verified_checkpoint": False,
    }
    write_json(checkpoint_path, checkpoint)
    return checkpoint


def _merge_silences(records: Iterable[dict[str, float]]) -> list[dict[str, float]]:
    sorted_records = sorted(records, key=lambda item: (item["start"], item["end"]))
    merged: list[dict[str, float]] = []
    for record in sorted_records:
        if not merged or float(record["start"]) > float(merged[-1]["end"]) + 0.12:
            merged.append(dict(record))
            continue
        merged[-1]["end"] = round(
            max(float(merged[-1]["end"]), float(record["end"])), 3
        )
        merged[-1]["duration"] = round(
            float(merged[-1]["end"]) - float(merged[-1]["start"]), 3
        )
    return merged


def _merge_frames(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_hashes_nearby: dict[str, float] = {}
    for frame in sorted(records, key=lambda item: float(item["timestamp"])):
        digest = str(frame.get("sha256", ""))
        previous_time = seen_hashes_nearby.get(digest)
        if previous_time is not None and float(frame["timestamp"]) - previous_time < 2.0:
            continue
        seen_hashes_nearby[digest] = float(frame["timestamp"])
        if merged and abs(float(frame["timestamp"]) - float(merged[-1]["timestamp"])) < 0.4:
            combined_sources = {
                *str(merged[-1].get("sampling_source", "")).split("+"),
                *str(frame.get("sampling_source", "")).split("+"),
            }
            if {"scene", "uniform"} <= combined_sources:
                merged[-1]["sampling_source"] = "scene+uniform"
            continue
        merged.append(frame)
    return merged


def _hamming_hex(first: str | None, second: str | None) -> int | None:
    if not first or not second:
        return None
    try:
        return (int(first, 16) ^ int(second, 16)).bit_count()
    except ValueError:
        return None


def _add_pixel_change_evidence(
    events: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> None:
    frame_by_path = {str(frame.get("path")): frame for frame in frames}
    for event in events:
        evidence = event.get("evidence", {})
        before = frame_by_path.get(str(evidence.get("before_frame", "")))
        after = frame_by_path.get(
            str(evidence.get("after_frame") or evidence.get("frame_path") or "")
        )
        if after is None:
            continue
        after_hash = after.get("interpretable_visual_features", {}).get("dhash64")
        if before is not None:
            before_hash = before.get("interpretable_visual_features", {}).get("dhash64")
            distance = _hamming_hex(before_hash, after_hash)
            if distance is not None:
                evidence["perceptual_hash_hamming_distance"] = distance
                evidence["perceptual_hash_bits"] = 64
        evidence["after_frame_sha256"] = after.get("sha256")
        if before is not None:
            evidence["before_frame_sha256"] = before.get("sha256")


def _restore_exact_frame_ocr_evidence(
    events: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> None:
    """Replace detector display truncation with the exact hash-bound frame OCR."""

    frame_by_path = {str(frame.get("path")): frame for frame in frames}
    for event in events:
        evidence = event.get("evidence", {})
        frame = frame_by_path.get(str(evidence.get("frame_path", "")))
        before = frame_by_path.get(str(evidence.get("before_frame", "")))
        after = frame_by_path.get(str(evidence.get("after_frame", "")))
        if frame is not None and "ocr_text" in evidence:
            evidence["ocr_text"] = str(frame.get("ocr_text", ""))
        if before is not None and "before_text" in evidence:
            evidence["before_text"] = str(before.get("ocr_text", ""))
        if after is not None and "after_text" in evidence:
            evidence["after_text"] = str(after.get("ocr_text", ""))


def _event_anchor(event: dict[str, Any]) -> float:
    return (float(event["start"]) + float(event["end"])) / 2


def deduplicate_visual_events(
    events: list[dict[str, Any]],
    *,
    continuous_gap_seconds: float,
) -> list[dict[str, Any]]:
    """Merge repeated continuous detections and suppress chunk-boundary duplicates."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in sorted(events, key=lambda value: (value["type"], _event_anchor(value))):
        grouped.setdefault(str(event["type"]), []).append(copy.deepcopy(event))
    result: list[dict[str, Any]] = []
    for event_type, rows in grouped.items():
        current: dict[str, Any] | None = None
        for event in rows:
            if current is None:
                current = event
                continue
            gap = float(event["start"]) - float(current["end"])
            duplicate_window = 2.0 if event_type != "code_or_formula_visible" else continuous_gap_seconds
            if gap > duplicate_window:
                result.append(current)
                current = event
                continue
            if event_type == "code_or_formula_visible":
                current["end"] = max(float(current["end"]), float(event["end"]))
                current["evidence"]["track_last_timestamp"] = round(
                    float(event["end"]), 3
                )
                current["evidence"]["track_detection_count"] = int(
                    current["evidence"].get("track_detection_count", 1)
                ) + 1
                if float(event.get("confidence", 0)) > float(current.get("confidence", 0)):
                    current["confidence"] = event["confidence"]
                continue
            # Scene/slide/board detectors often emit the same boundary from the
            # overlapping decode ranges. Keep the strongest record only near that point.
            if float(event.get("confidence", 0)) > float(current.get("confidence", 0)):
                current = event
        if current is not None:
            result.append(current)
    return sorted(result, key=lambda value: (float(value["start"]), value["type"]))


def _interval_union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    rows = sorted((max(0.0, start), max(start, end)) for start, end in intervals)
    merged: list[list[float]] = []
    for start, end in rows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _intersection_duration(
    first: Iterable[tuple[float, float]],
    second: Iterable[tuple[float, float]],
) -> float:
    left = sorted(first)
    right = sorted(second)
    i = 0
    j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def evaluate_caption_media_alignment(
    transcript: dict[str, Any],
    *,
    media_duration_seconds: float,
    silences: list[dict[str, float]],
    expected_media_url: str | None = None,
) -> dict[str, Any]:
    segments = transcript.get("segments", [])
    caption_intervals = [
        (float(segment["start"]), float(segment["end"])) for segment in segments
    ]
    silence_intervals = [
        (float(record["start"]), float(record["end"])) for record in silences
    ]
    caption_duration = _interval_union_duration(caption_intervals)
    caption_silence_overlap = _intersection_duration(
        caption_intervals, silence_intervals
    )
    activity_fraction = (
        max(0.0, 1.0 - caption_silence_overlap / caption_duration)
        if caption_duration
        else 0.0
    )
    last_caption_end = max((value[1] for value in caption_intervals), default=0.0)
    first_caption_start = min((value[0] for value in caption_intervals), default=0.0)
    provenance = transcript.get("provenance", {})
    verification = provenance.get("caption_source_verification", {})
    declared_media_url = str(verification.get("page_declared_media_url", ""))
    expected_normalized = (
        normalize_archive_media_url(expected_media_url)
        if expected_media_url
        else None
    )
    try:
        declared_normalized = (
            normalize_archive_media_url(declared_media_url)
            if declared_media_url
            else None
        )
    except ValueError:
        declared_normalized = None
    media_binding_matches = bool(
        expected_normalized
        and declared_normalized
        and expected_normalized == declared_normalized
    )
    coverage = provenance.get("transcript_coverage", {})
    reference_duration = coverage.get("source_duration_seconds")
    try:
        duration_delta = abs(float(reference_duration) - media_duration_seconds)
    except (TypeError, ValueError):
        duration_delta = None
    duration_matches = duration_delta is not None and duration_delta <= max(
        2.0, media_duration_seconds * 0.002
    )
    endpoints_valid = (
        bool(caption_intervals)
        and first_caption_start >= 0
        and last_caption_end <= media_duration_seconds + 0.5
    )
    source_verified = bool(
        transcript.get("transcript_kind") == "caption_import"
        and verification.get("caption_source_verified", True)
        and provenance.get("caption_source_verified", False)
    )
    # Older formal-caption artifacts place caption_source_verified directly in
    # provenance and detailed page/media fields in caption_source_verification.
    source_verified = bool(
        transcript.get("transcript_kind") == "caption_import"
        and provenance.get("caption_source_verified", False)
        and verification.get("trusted_source_host") == "ocw.mit.edu"
    )
    timeline_passed = bool(
        source_verified
        and media_binding_matches
        and duration_matches
        and endpoints_valid
        and activity_fraction >= 0.5
    )
    return {
        "status": "passed" if timeline_passed else "failed",
        "official_caption_timeline_media_binding_verified": timeline_passed,
        "audio_content_verified": False,
        "source_caption_verified": source_verified,
        "declared_media_url_normalized": declared_normalized,
        "expected_media_url_normalized": expected_normalized,
        "declared_media_matches_download_source": media_binding_matches,
        "media_duration_seconds": round(media_duration_seconds, 3),
        "caption_reference_media_duration_seconds": (
            round(float(reference_duration), 3)
            if isinstance(reference_duration, (int, float))
            else None
        ),
        "duration_delta_seconds": (
            round(duration_delta, 6) if duration_delta is not None else None
        ),
        "duration_matches_reference": duration_matches,
        "first_caption_start_seconds": round(first_caption_start, 3),
        "last_caption_end_seconds": round(last_caption_end, 3),
        "caption_endpoints_within_media": endpoints_valid,
        "caption_union_duration_seconds": round(caption_duration, 3),
        "caption_silence_overlap_seconds": round(caption_silence_overlap, 3),
        "caption_audio_activity_overlap_fraction": round(activity_fraction, 6),
        "audio_activity_method": "ffmpeg_silencedetect_complement_within_caption_intervals",
        "interpretation": (
            "This verifies official caption timestamps against the declared media and "
            "observed audio activity. It does not verify every caption word against speech."
        ),
    }


def _sampling_coverage(
    frames: list[dict[str, Any]],
    *,
    duration_seconds: float,
    interval_seconds: float,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    uniform_times = sorted(
        float(frame["timestamp"])
        for frame in frames
        if "uniform" in str(frame.get("sampling_source", ""))
    )
    gaps = [
        later - earlier for earlier, later in zip(uniform_times, uniform_times[1:])
    ]
    covered = _interval_union_duration(
        (
            max(0.0, timestamp - interval_seconds),
            min(duration_seconds, timestamp + interval_seconds),
        )
        for timestamp in uniform_times
    )
    chunks_with_uniform = sum(
        any(
            float(chunk["nominal_start"]) - 0.001
            <= timestamp
            <= float(chunk["nominal_end"]) + 0.001
            for timestamp in uniform_times
        )
        for chunk in chunks
    )
    first = uniform_times[0] if uniform_times else None
    last = uniform_times[-1] if uniform_times else None
    endpoints_covered = bool(
        first is not None
        and last is not None
        and first <= interval_seconds + 1.0
        and last >= duration_seconds - interval_seconds - 1.0
    )
    fraction = covered / duration_seconds if duration_seconds else 0.0
    return {
        "uniform_frame_count": len(uniform_times),
        "first_uniform_timestamp": round(first, 3) if first is not None else None,
        "last_uniform_timestamp": round(last, 3) if last is not None else None,
        "maximum_uniform_gap_seconds": round(max(gaps), 3) if gaps else None,
        "timeline_neighborhood_coverage_fraction": round(min(1.0, fraction), 6),
        "chunk_count": len(chunks),
        "chunks_with_uniform_frame": chunks_with_uniform,
        "every_chunk_has_uniform_frame": chunks_with_uniform == len(chunks),
        "timeline_endpoints_covered": endpoints_covered,
        "full_timeline_sampling_passed": bool(
            endpoints_covered
            and chunks_with_uniform == len(chunks)
            and fraction >= 0.99
            and (not gaps or max(gaps) <= interval_seconds * 1.6 + 1.0)
        ),
    }


def _ocr_quality_summary(frames: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [
        frame.get("ocr_audit", {})
        for frame in frames
        if isinstance(frame.get("ocr_audit"), dict)
    ]
    word_counts = [int(value.get("accepted_word_count", 0)) for value in audits]
    confidences = [
        float(value["accepted_mean_confidence"])
        for value in audits
        if isinstance(value.get("accepted_mean_confidence"), (int, float))
    ]
    return {
        "attempted_frame_count": len(audits),
        "frame_with_accepted_word_count": sum(value > 0 for value in word_counts),
        "frame_with_three_or_more_accepted_words_count": sum(
            value >= 3 for value in word_counts
        ),
        "accepted_word_count": sum(word_counts),
        "mean_selected_frame_word_confidence": (
            round(sum(confidences) / len(confidences), 6)
            if confidences
            else None
        ),
        "tesseract_word_confidence_is_calibrated_probability": False,
        "ocr_accuracy_established": False,
        "interpretation": (
            "Accepted words passed Tesseract's heuristic confidence threshold. "
            "They are not human-verified, and handwritten mathematics may remain noisy."
        ),
    }


def _semantic_task_manifest(
    *,
    video_id: str,
    media_sha256: str,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": SEMANTIC_TASK_SCHEMA,
        "video_id": video_id,
        "media_sha256": media_sha256,
        "frame_count": len(frames),
        "frames": [
            {
                "frame_id": f"{video_id}:{index:05d}",
                "timestamp": frame["timestamp"],
                "path": frame["path"],
                "sha256": frame["sha256"],
            }
            for index, frame in enumerate(frames, 1)
        ],
    }


def analyze_longform_video(
    video_path: str | Path,
    transcript: dict[str, Any],
    output_dir: str | Path,
    *,
    expected_media_url: str | None,
    chunk_seconds: float = 300.0,
    overlap_seconds: float = 2.0,
    frame_interval_seconds: float = 15.0,
    scene_threshold: float = 0.32,
    max_scene_frames_per_chunk: int = 12,
    use_ocr: bool = True,
    ocr_workers: int = 4,
    noise_db: float = -35.0,
    minimum_silence_seconds: float = 0.8,
    resume: bool = True,
) -> dict[str, Any]:
    validation = validate_transcript(transcript)
    if not validation.valid:
        raise ValueError("invalid transcript: " + "; ".join(validation.errors))
    if frame_interval_seconds <= 0:
        raise ValueError("frame_interval_seconds must be positive")
    if max_scene_frames_per_chunk < 1:
        raise ValueError("max_scene_frames_per_chunk must be positive")
    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    output = ensure_private_directory(output_dir).resolve()
    media = probe_media(path)
    media_sha256 = file_sha256(path)
    media["sha256"] = media_sha256
    streams = media.get("streams", [])
    has_video = any(stream.get("codec_type") == "video" for stream in streams)
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    if not has_video:
        raise ValueError("long-form multimodal analysis requires a video stream")
    chunks = plan_chunks(
        float(media["duration_seconds"]),
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    config = {
        "pipeline": PIPELINE_VERSION,
        "extraction_pipeline": EXTRACTION_VERSION,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "frame_interval_seconds": frame_interval_seconds,
        "scene_threshold": scene_threshold,
        "max_scene_frames_per_chunk": max_scene_frames_per_chunk,
        "use_ocr": use_ocr,
        "ocr_workers": ocr_workers,
        "noise_db": noise_db,
        "minimum_silence_seconds": minimum_silence_seconds,
    }
    transcript_sha256 = canonical_sha256(transcript)
    extraction_config = {
        key: value for key, value in config.items() if key != "pipeline"
    }
    job_fingerprint = canonical_sha256(
        {
            "media_sha256": media_sha256,
            "transcript_sha256": transcript_sha256,
            "extraction_config": extraction_config,
        }
    )
    job_dir = ensure_private_directory(output / "jobs" / job_fingerprint[:20])
    checkpoints: list[dict[str, Any]] = []
    for chunk in chunks:
        checkpoints.append(
            _extract_chunk(
                path,
                output,
                job_dir,
                chunk=chunk,
                job_fingerprint=job_fingerprint,
                language=str(transcript.get("language", "en")),
                frame_interval_seconds=frame_interval_seconds,
                scene_threshold=scene_threshold,
                max_scene_frames_per_chunk=max_scene_frames_per_chunk,
                ocr_workers=ocr_workers,
                use_ocr=use_ocr,
                has_audio=has_audio,
                noise_db=noise_db,
                minimum_silence_seconds=minimum_silence_seconds,
                resume=resume,
            )
        )
    frames = _merge_frames(
        frame for checkpoint in checkpoints for frame in checkpoint.get("frames", [])
    )
    silences = _merge_silences(
        silence
        for checkpoint in checkpoints
        for silence in checkpoint.get("silences", [])
    )
    visual_events = infer_visual_events(frames)
    _restore_exact_frame_ocr_evidence(visual_events, frames)
    _add_pixel_change_evidence(visual_events, frames)
    visual_events = deduplicate_visual_events(
        visual_events,
        continuous_gap_seconds=frame_interval_seconds * 1.6 + 1.0,
    )
    events = fuse_multimodal_events(
        transcript,
        silences,
        visual_events,
        observations=None,
        language_modality="transcript",
    )
    for event in events:
        event["event_fingerprint_sha256"] = canonical_sha256(
            {key: value for key, value in event.items() if key != "event_id"}
        )
    sampling_coverage = _sampling_coverage(
        frames,
        duration_seconds=float(media["duration_seconds"]),
        interval_seconds=frame_interval_seconds,
        chunks=chunks,
    )
    ocr_quality = _ocr_quality_summary(frames)
    alignment = evaluate_caption_media_alignment(
        transcript,
        media_duration_seconds=float(media["duration_seconds"]),
        silences=silences,
        expected_media_url=expected_media_url,
    )
    modalities = ["transcript"]
    if has_audio:
        modalities.append("audio")
    modalities.append("visual")
    if any(str(frame.get("ocr_text", "")).strip() for frame in frames):
        modalities.append("ocr")
    total_silence = sum(float(record["duration"]) for record in silences)
    semantic_tasks = _semantic_task_manifest(
        video_id=str(transcript["video_id"]),
        media_sha256=media_sha256,
        frames=frames,
    )
    write_json(output / "visual_semantic_tasks.json", semantic_tasks)
    analysis = {
        "schema_version": "1.1-longform",
        "media": media,
        "language": {
            "modality": "transcript",
            "status": (
                "official_caption_timeline_media_binding_verified"
                if alignment["official_caption_timeline_media_binding_verified"]
                else "provided_transcript"
            ),
            "transcript_kind": transcript.get("transcript_kind"),
            "audio_content_verified": False,
            "note": (
                "The official caption timeline is checked against this media and audio "
                "activity; caption words are not relabeled as ASR-verified speech."
            ),
        },
        "modalities_available": modalities,
        "caption_media_alignment": alignment,
        "audio": {
            "status": "available" if has_audio else "unavailable",
            "silences": silences,
            "silence_ratio": (
                round(total_silence / float(media["duration_seconds"]), 6)
                if media["duration_seconds"]
                else 0.0
            ),
            "analysis_scope": "all_chunks_full_timeline",
        },
        "visual": {
            "status": "available",
            "keyframes": frames,
            "events": visual_events,
            "ocr_enabled": use_ocr,
            "ocr_quality": ocr_quality,
            "sampling_coverage": sampling_coverage,
            "semantic_features": {
                "status": "pending_gpu_inference",
                "task_manifest": "visual_semantic_tasks.json",
                "frame_count": len(frames),
                "zero_shot_scores_are_calibrated_probabilities": False,
            },
        },
        "events": events,
        "privacy": {
            "identity_recognition_performed": False,
            "face_recognition_performed": False,
            "raw_frames_private": True,
        },
        "provenance": {
            "pipeline": PIPELINE_VERSION,
            "job_fingerprint_sha256": job_fingerprint,
            "media_sha256": media_sha256,
            "transcript_canonical_sha256": transcript_sha256,
            "config_sha256": canonical_sha256(config),
            "extraction_config_sha256": canonical_sha256(extraction_config),
            "config": config,
            "ffmpeg": _tool_version(os.getenv("TSM_FFMPEG", "ffmpeg")),
            "ffprobe": _tool_version(os.getenv("TSM_FFPROBE", "ffprobe")),
            "tesseract": (
                _tool_version(os.getenv("TSM_TESSERACT", "tesseract"))
                if use_ocr
                else None
            ),
            "chunk_count": len(chunks),
            "completed_chunk_count": len(checkpoints),
            "resumed_chunk_count": sum(
                bool(value.get("resumed_from_verified_checkpoint"))
                for value in checkpoints
            ),
            "chunks": [
                {
                    **value["chunk"],
                    "chunk_fingerprint_sha256": value[
                        "chunk_fingerprint_sha256"
                    ],
                    "frame_count": len(value.get("frames", [])),
                    "scene_candidate_count": int(
                        value.get("scene_candidate_count", 0)
                    ),
                    "scene_retained_count": int(
                        value.get("scene_retained_count", 0)
                    ),
                    "silence_count": len(value.get("silences", [])),
                    "checkpoint_sha256": file_sha256(
                        job_dir
                        / "chunks"
                        / str(value["chunk"]["chunk_id"])
                        / "checkpoint.json"
                    ),
                }
                for value in checkpoints
            ],
        },
    }
    enriched = copy.deepcopy(transcript)
    enriched["multimodal"] = analysis
    enriched_validation = validate_transcript(enriched)
    if not enriched_validation.valid:
        raise ValueError(
            "long-form enrichment produced invalid transcript: "
            + "; ".join(enriched_validation.errors)
        )
    write_json(output / "analysis.json", analysis)
    write_json(output / "enriched_transcript.json", enriched)
    write_json(
        output / "summary.json",
        {
            "video_id": transcript["video_id"],
            "media_sha256": media_sha256,
            "duration_seconds": media["duration_seconds"],
            "chunk_count": len(chunks),
            "resumed_chunk_count": analysis["provenance"]["resumed_chunk_count"],
            "keyframe_count": len(frames),
            "nonempty_ocr_frame_count": sum(
                bool(str(frame.get("ocr_text", "")).strip()) for frame in frames
            ),
            "ocr_frame_with_three_or_more_accepted_words_count": ocr_quality[
                "frame_with_three_or_more_accepted_words_count"
            ],
            "ocr_accepted_word_count": ocr_quality["accepted_word_count"],
            "ocr_accuracy_established": False,
            "silence_count": len(silences),
            "visual_event_count": len(visual_events),
            "fused_event_count": len(events),
            "full_timeline_sampling_passed": sampling_coverage[
                "full_timeline_sampling_passed"
            ],
            "caption_timeline_media_binding_verified": alignment[
                "official_caption_timeline_media_binding_verified"
            ],
            "audio_content_verified": False,
            "semantic_features_status": "pending_gpu_inference",
        },
    )
    return enriched


def _resolve_manifest_file(manifest_path: Path, value: str) -> Path:
    raw = Path(value)
    candidates = [raw, manifest_path.parent / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(value)


def process_longform_dataset(
    media_manifest_path: str | Path,
    transcript_manifest_path: str | Path,
    output_dir: str | Path,
    **analysis_options: Any,
) -> dict[str, Any]:
    media_path = Path(media_manifest_path).resolve()
    transcript_path = Path(transcript_manifest_path).resolve()
    media_manifest = read_json(media_path)
    transcript_manifest = read_json(transcript_path)
    media_by_id = {
        str(item["video_id"]): item for item in media_manifest.get("videos", [])
    }
    source_by_id = {
        str(item["video_id"]): item
        for item in media_manifest.get("source_videos", media_manifest.get("videos", []))
    }
    output = ensure_private_directory(output_dir).resolve()
    lecture_rows: list[dict[str, Any]] = []
    for item in transcript_manifest.get("videos", []):
        video_id = str(item["video_id"])
        if video_id not in media_by_id:
            raise ValueError(f"media manifest has no video_id {video_id}")
        media_item = media_by_id[video_id]
        transcript_file = _resolve_manifest_file(
            transcript_path, str(item["transcript_path"])
        )
        video_value = str(
            media_item.get("video_path")
            or media_item.get("media_path")
            or media_item.get("path")
            or ""
        )
        video_file = _resolve_manifest_file(media_path, video_value)
        expected_media_url = str(
            media_item.get("media_url")
            or media_item.get("requested_media_url")
            or media_item.get("effective_download_url")
            or media_item.get("source_media_url")
            or source_by_id.get(video_id, {}).get("media_url")
            or ""
        )
        lecture_dir = ensure_private_directory(output / "lectures" / video_id)
        analyze_longform_video(
            video_file,
            read_json(transcript_file),
            lecture_dir,
            expected_media_url=expected_media_url,
            **analysis_options,
        )
        summary = read_json(lecture_dir / "summary.json")
        lecture_rows.append(
            {
                "video_id": video_id,
                "course_id": item["course_id"],
                "title": item["title"],
                "source_url": item["source_url"],
                "transcript_path": str(
                    (lecture_dir / "enriched_transcript.json").relative_to(output)
                ),
                "analysis_path": str((lecture_dir / "analysis.json").relative_to(output)),
                "semantic_task_path": str(
                    (lecture_dir / "visual_semantic_tasks.json").relative_to(output)
                ),
                "summary": summary,
            }
        )
    dataset_manifest = {
        "dataset_id": "mit_ocw_10_full_video_multimodal",
        "schema_version": "1.0",
        "pipeline": PIPELINE_VERSION,
        "media_manifest_sha256": file_sha256(media_path),
        "transcript_manifest_sha256": file_sha256(transcript_path),
        "video_count": len(lecture_rows),
        "videos": lecture_rows,
        "aggregate": {
            "duration_seconds": round(
                sum(float(item["summary"]["duration_seconds"]) for item in lecture_rows),
                3,
            ),
            "keyframe_count": sum(
                int(item["summary"]["keyframe_count"]) for item in lecture_rows
            ),
            "nonempty_ocr_frame_count": sum(
                int(item["summary"]["nonempty_ocr_frame_count"])
                for item in lecture_rows
            ),
            "ocr_frame_with_three_or_more_accepted_words_count": sum(
                int(
                    item["summary"][
                        "ocr_frame_with_three_or_more_accepted_words_count"
                    ]
                )
                for item in lecture_rows
            ),
            "ocr_accepted_word_count": sum(
                int(item["summary"]["ocr_accepted_word_count"])
                for item in lecture_rows
            ),
            "visual_event_count": sum(
                int(item["summary"]["visual_event_count"]) for item in lecture_rows
            ),
            "fused_event_count": sum(
                int(item["summary"]["fused_event_count"]) for item in lecture_rows
            ),
            "full_timeline_sampling_passed_count": sum(
                bool(item["summary"]["full_timeline_sampling_passed"])
                for item in lecture_rows
            ),
            "caption_timeline_media_binding_verified_count": sum(
                bool(item["summary"]["caption_timeline_media_binding_verified"])
                for item in lecture_rows
            ),
            "semantic_feature_complete_count": 0,
        },
        "claim_boundary": {
            "full_video_bytes_downloaded_and_hashed": True,
            "full_timeline_sampling_required": True,
            "official_caption_timeline_alignment_required": True,
            "audio_content_word_level_verified": False,
            "visual_semantic_features_complete": False,
            "recognition_accuracy_established": False,
            "causal_multimodal_gain_established": False,
        },
    }
    write_json(output / "dataset_manifest.json", dataset_manifest)
    return dataset_manifest


def _semantic_index(
    results: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    if results.get("schema") != SEMANTIC_RESULT_SCHEMA:
        raise ValueError("unsupported visual semantic result schema")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in results.get("frames", []):
        digest = str(row.get("sha256", ""))
        if not SHA256_RE.fullmatch(digest):
            raise ValueError("semantic result contains an invalid frame SHA-256")
        key = (str(row.get("path", "")), digest)
        if not key[0] or key in index:
            raise ValueError("semantic result contains a duplicate or empty frame path")
        index[key] = row
    return index


def attach_visual_semantic_results(
    enriched_transcript: dict[str, Any],
    semantic_results: dict[str, Any],
) -> dict[str, Any]:
    """Bind hash-matched visual features to frames and event evidence."""

    result = copy.deepcopy(enriched_transcript)
    analysis = result.get("multimodal")
    if not isinstance(analysis, dict):
        raise ValueError("transcript has no multimodal analysis")
    media_sha = str(analysis.get("media", {}).get("sha256", ""))
    if semantic_results.get("media_sha256") != media_sha:
        raise ValueError("semantic result media SHA-256 does not match analysis")
    if semantic_results.get("video_id") != result.get("video_id"):
        raise ValueError("semantic result video_id does not match transcript")
    index = _semantic_index(semantic_results)
    frames = analysis.get("visual", {}).get("keyframes", [])
    missing: list[str] = []
    semantic_by_path: dict[str, dict[str, Any]] = {}
    for frame in frames:
        digest = str(frame.get("sha256", ""))
        key = (str(frame.get("path", "")), digest)
        semantic = index.get(key)
        if semantic is None:
            missing.append(f"{key[0]}:{digest}")
            continue
        payload = {
            "backend": semantic_results.get("backend"),
            "top_label": semantic.get("top_label"),
            "top_relative_score": semantic.get("top_relative_score"),
            "score_margin": semantic.get("score_margin"),
            "relative_prompt_scores": semantic.get("relative_prompt_scores"),
            "embedding": semantic.get("embedding"),
            "embedding_sha256": semantic.get("embedding_sha256"),
            "scores_are_calibrated_probabilities": False,
        }
        frame["visual_semantics"] = payload
        semantic_by_path[str(frame.get("path"))] = payload
    if missing:
        raise ValueError(
            f"semantic results do not cover {len(missing)} hash-bound frames"
        )
    for collection in (
        analysis.get("visual", {}).get("events", []),
        analysis.get("events", []),
    ):
        for event in collection:
            evidence = event.get("evidence", {})
            paths = [
                str(evidence.get(key, ""))
                for key in ("frame_path", "after_frame", "before_frame")
                if evidence.get(key)
            ]
            bound = [semantic_by_path[path] for path in paths if path in semantic_by_path]
            if bound:
                evidence["visual_semantic_labels"] = [
                    {
                        "frame_path": path,
                        "top_label": semantic_by_path[path]["top_label"],
                        "top_relative_score": semantic_by_path[path][
                            "top_relative_score"
                        ],
                    }
                    for path in paths
                    if path in semantic_by_path
                ]
    analysis["visual"]["semantic_features"] = {
        "status": "complete_hash_bound_inference",
        "frame_count": len(frames),
        "covered_frame_count": len(frames),
        "backend": semantic_results.get("backend"),
        "model_provenance": semantic_results.get("model_provenance"),
        "ontology": semantic_results.get("ontology"),
        "result_sha256": canonical_sha256(semantic_results),
        "zero_shot_scores_are_calibrated_probabilities": False,
        "interpretation": (
            "CLIP embeddings and closed-ontology relative prompt scores are semantic "
            "features, not independently labeled recognition correctness."
        ),
    }
    validation = validate_transcript(result)
    if not validation.valid:
        raise ValueError(
            "semantic attachment produced invalid transcript: "
            + "; ".join(validation.errors)
        )
    return result
