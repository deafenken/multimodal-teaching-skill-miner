"""Hash-bound, offline GPU ASR handoff for private TeachObs media.

This module validates *technical* ASR provenance and timeline coverage.  It does
not call an ASR result an official caption, and it does not establish content
accuracy or WER without an independent human reference audit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
from typing import Any, Callable, Sequence

from .full_video_dataset import file_sha256
from .io_utils import (
    ensure_private_directory,
    ensure_private_file,
    read_json,
    write_json,
)
from .teachobs_captions import PRIVATE_AUDIT_SCHEMA as CAPTION_AUDIT_SCHEMA


DATASET_ID = "teachobs_v0_1_human_validated"
MEDIA_MANIFEST_SCHEMA = "teaching_skill_miner.teachobs_private_media_manifest.v1"
JOB_MANIFEST_SCHEMA = "teaching_skill_miner.teachobs_asr_job_manifest.v2"
RESULT_SCHEMA = "teaching_skill_miner.teachobs_asr_lesson_result.v4"
IMPORT_AUDIT_SCHEMA = "teaching_skill_miner.teachobs_asr_import_audit.v1"
COVERAGE_MATRIX_SCHEMA = "teaching_skill_miner.teachobs_transcript_coverage.v1"
PUBLIC_RECEIPT_SCHEMA = "teaching_skill_miner.teachobs_asr_receipt.v1"
RUNNER_NAME = "teaching_skill_miner.teachobs_asr_gpu_runner.v4"
COVERAGE_POLICY_NAME = "full_media_single_pass_vad_relative_endpoints_v1"
LONG_SEGMENT_POLICY = (
    "deterministic_greedy_positive_word_anchor_with_point_anchored_"
    "zero_duration_nearest_attachment_v3"
)
ZERO_DURATION_WORD_ATTACHMENT_POLICY = (
    "point_anchored_zero_duration_word_nearest_positive_attachment"
)
EXPECTED_LESSON_COUNT = 30

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_LESSON_RE = re.compile(r"^S(?:[1-9]|[12][0-9]|30)$")
_LANGUAGE_RE = re.compile(r"^(?:auto|[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}$")
_CONTAINER_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _safe_lesson_id(value: Any) -> str:
    lesson_id = str(value or "").strip()
    if not _LESSON_RE.fullmatch(lesson_id):
        raise ValueError(f"unsafe TeachObs lesson id: {value!r}")
    return lesson_id


def _safe_relative_path(value: Any, *, suffix: str | None = None) -> str:
    text = str(value or "").strip()
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or "\x00" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe private relative path: {value!r}")
    if suffix is not None and path.suffix.casefold() != suffix.casefold():
        raise ValueError(f"private relative path must end in {suffix}")
    return path.as_posix()


def _load_object(path: str | Path, *, description: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{description} is missing or unsafe")
    source = candidate.resolve()
    value = read_json(source)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return source, value


def _verify_canonical_digest(
    value: dict[str, Any], *, digest_field: str, description: str
) -> str:
    claimed = _require_sha256(value.get(digest_field), field=digest_field)
    unsigned = {key: item for key, item in value.items() if key != digest_field}
    actual = _canonical_sha256(unsigned)
    if claimed != actual:
        raise ValueError(f"{description} canonical hash mismatch")
    return claimed


def _ffprobe_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            os.fspath(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"ffprobe returned no valid duration for {path.name}") from exc
    if result.returncode != 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"ffprobe failed for private media {path.name}")
    return duration


def model_directory_sha256(path: str | Path) -> str:
    """Hash a local model snapshot without reading outside that directory."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("local model snapshot directory is missing or unsafe")
    root = candidate.resolve()
    files: list[tuple[Path, str, int]] = []
    pending_directories = [root]
    while pending_directories:
        current = pending_directories.pop()
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            raise ValueError("local model snapshot tree cannot be read safely") from exc
        for entry in entries:
            item = Path(entry.path)
            try:
                metadata = item.lstat()
            except OSError as exc:
                raise ValueError("local model snapshot entry cannot be inspected") from exc
            relative = item.relative_to(root).as_posix()
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("local model snapshot may not contain symlinks")
            if stat.S_ISDIR(mode):
                pending_directories.append(item)
            elif stat.S_ISREG(mode):
                files.append((item, relative, metadata.st_size))
            else:
                raise ValueError(
                    "local model snapshot may contain only directories and regular files"
                )
    files.sort(key=lambda item: item[1])
    if not files:
        raise ValueError("local model snapshot directory contains no files")
    digest = sha256()
    for item, relative, expected_size in files:
        try:
            current = item.lstat()
        except OSError as exc:
            raise ValueError("local model snapshot changed during hashing") from exc
        if not stat.S_ISREG(current.st_mode) or current.st_size != expected_size:
            raise ValueError("local model snapshot changed during hashing")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(expected_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_caption_audit(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source, audit = _load_object(path, description="TeachObs caption audit")
    if audit.get("schema") != CAPTION_AUDIT_SCHEMA or audit.get("dataset_id") != DATASET_ID:
        raise ValueError("unsupported TeachObs caption audit")
    _verify_canonical_digest(
        audit,
        digest_field="audit_canonical_sha256",
        description="TeachObs caption audit",
    )
    records = audit.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_LESSON_COUNT:
        raise ValueError("TeachObs caption audit must contain all 30 lessons")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("TeachObs caption audit contains a non-object record")
        lesson_id = _safe_lesson_id(record.get("lesson_id"))
        if lesson_id in seen:
            raise ValueError("TeachObs caption audit contains duplicate lessons")
        seen.add(lesson_id)
    return source, audit


def _media_records(
    media_manifest_path: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    source, manifest = _load_object(
        media_manifest_path, description="TeachObs media manifest"
    )
    if manifest.get("schema") != MEDIA_MANIFEST_SCHEMA or manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("unsupported TeachObs media manifest")
    lessons = manifest.get("lessons")
    if (
        not isinstance(lessons, list)
        or not lessons
        or len(lessons) > EXPECTED_LESSON_COUNT
        or manifest.get("selected_lesson_count") != EXPECTED_LESSON_COUNT
    ):
        raise ValueError(
            "TeachObs media manifest must declare the 30-lesson selection and "
            "contain one or more completed media records"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for item in lessons:
        if not isinstance(item, dict):
            raise ValueError("TeachObs media manifest contains a non-object lesson")
        lesson_id = _safe_lesson_id(item.get("lesson_id"))
        if lesson_id in by_id:
            raise ValueError("TeachObs media manifest contains duplicate lessons")
        by_id[lesson_id] = item
    return source, manifest, by_id


def _validated_media_binding(
    item: dict[str, Any],
    *,
    media_root: Path,
    duration_probe: Callable[[Path], float],
) -> dict[str, Any]:
    lesson_id = _safe_lesson_id(item.get("lesson_id"))
    relative = _safe_relative_path(item.get("media_path"), suffix=".mp4")
    media_path = media_root.joinpath(*PurePosixPath(relative).parts)
    try:
        media_path.relative_to(media_root)
    except ValueError as exc:
        raise ValueError(f"private media escapes media root for {lesson_id}") from exc
    if media_path.is_symlink() or not media_path.is_file():
        raise ValueError(f"private media is missing or unsafe for {lesson_id}")
    expected_digest = _require_sha256(item.get("media_sha256"), field="media_sha256")
    actual_digest = file_sha256(media_path)
    if actual_digest != expected_digest:
        raise ValueError(f"private media SHA-256 mismatch for {lesson_id}")
    expected_size = item.get("media_size_bytes")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError(f"invalid private media size for {lesson_id}")
    if media_path.stat().st_size != expected_size:
        raise ValueError(f"private media size mismatch for {lesson_id}")
    probe = item.get("media_probe")
    if not isinstance(probe, dict):
        raise ValueError(f"private media probe is missing for {lesson_id}")
    try:
        declared_duration = float(probe["duration_seconds"])
        actual_duration = float(duration_probe(media_path))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid private media duration for {lesson_id}") from exc
    tolerance = max(0.5, declared_duration * 0.001)
    if (
        not math.isfinite(declared_duration)
        or not math.isfinite(actual_duration)
        or declared_duration <= 0
        or actual_duration <= 0
        or abs(actual_duration - declared_duration) > tolerance
    ):
        raise ValueError(f"private media duration binding mismatch for {lesson_id}")
    return {
        "lesson_id": lesson_id,
        "split": str(item.get("split") or ""),
        "media_relative_path": relative,
        "media_sha256": expected_digest,
        "media_size_bytes": expected_size,
        "media_duration_seconds": round(actual_duration, 6),
        "media_manifest_duration_seconds": round(declared_duration, 6),
        "duration_tolerance_seconds": round(tolerance, 6),
    }


def _language_map(path: str | Path | None, default_language: str) -> dict[str, str]:
    if not _LANGUAGE_RE.fullmatch(default_language):
        raise ValueError("default ASR language must be `auto` or a safe language tag")
    if path is None:
        return {}
    _, value = _load_object(path, description="TeachObs ASR language map")
    result: dict[str, str] = {}
    for raw_id, raw_language in value.items():
        lesson_id = _safe_lesson_id(raw_id)
        language = str(raw_language).strip()
        if not _LANGUAGE_RE.fullmatch(language):
            raise ValueError(f"invalid ASR language for {lesson_id}")
        result[lesson_id] = language
    return result


def _fixed_contract(
    *,
    model_id: str,
    model_revision: str,
    model_files_sha256: str,
    faster_whisper_version: str,
    ctranslate2_version: str,
    container_image_digest: str,
    min_timeline_span_fraction: float,
    max_endpoint_gap_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_id = model_id.strip()
    if not model_id or len(model_id) > 160 or any(char.isspace() for char in model_id):
        raise ValueError("ASR model id must be a non-empty repository-style identifier")
    if not _REVISION_RE.fullmatch(model_revision):
        raise ValueError("ASR model revision must be an exact lowercase 40-hex commit")
    if not _VERSION_RE.fullmatch(faster_whisper_version) or not _VERSION_RE.fullmatch(
        ctranslate2_version
    ):
        raise ValueError("ASR runtime package versions must be exact safe strings")
    if not _CONTAINER_RE.fullmatch(container_image_digest):
        raise ValueError("container image digest must be sha256:<64 lowercase hex>")
    if not 0.5 <= min_timeline_span_fraction <= 1.0:
        raise ValueError("minimum timeline-span coverage must be between 0.5 and 1")
    if (
        not math.isfinite(max_endpoint_gap_fraction)
        or max_endpoint_gap_fraction < 0
        or max_endpoint_gap_fraction > 1 - min_timeline_span_fraction + 1e-12
    ):
        raise ValueError(
            "maximum endpoint gap fraction must be non-negative and no greater "
            "than one minus the minimum timeline-span fraction"
        )
    model = {
        "model_id": model_id,
        "model_revision": model_revision,
        "model_snapshot_sha256_v1": _require_sha256(
            model_files_sha256, field="model_files_sha256"
        ),
        "model_hash_algorithm": "relative_path_nul_size_nul_file_sha256_newline_v1",
        "model_source_location": "prepositioned_private_local_snapshot",
        "network_download_allowed_by_runner": False,
    }
    decoding = {
        "engine": "faster_whisper",
        "task": "transcribe",
        "translation_enabled": False,
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
        "device": "cuda",
        "compute_type": "float16",
        "audio_sample_rate_hz": 16000,
        "input_mode": "full_media_single_pass",
        "timestamp_unit": "seconds",
        "maximum_segment_duration_seconds": 30.5,
        "long_segment_postprocessing_policy": LONG_SEGMENT_POLICY,
        "long_segment_duration_basis": "restored_original_media_wall_clock",
        "zero_duration_word_attachment_policy": (
            ZERO_DURATION_WORD_ATTACHMENT_POLICY
        ),
        "zero_duration_word_maximum_attachment_distance_seconds": 30.5,
    }
    runtime = {
        "runner_name": RUNNER_NAME,
        "runner_module_sha256": file_sha256(Path(__file__).resolve()),
        "faster_whisper_version": faster_whisper_version,
        "ctranslate2_version": ctranslate2_version,
        "device_type": "cuda",
        "container_image_digest": container_image_digest,
    }
    coverage = {
        "policy_name": COVERAGE_POLICY_NAME,
        "input_scope": "hash_bound_full_media_single_pass",
        "vad_filter_enabled": True,
        "endpoint_timestamps_mean": "first_and_last_vad_retained_speech",
        "minimum_timeline_span_fraction": round(min_timeline_span_fraction, 6),
        "maximum_endpoint_gap_fraction": round(max_endpoint_gap_fraction, 6),
        "endpoint_gap_fraction_applied_symmetrically": True,
        "endpoint_gap_fraction_denominator": "hash_bound_media_duration_seconds",
        "absolute_endpoint_gap_seconds_gate_enabled": False,
        "interpretation": (
            "technical full-media ASR/VAD timeline coverage only; endpoint gaps "
            "are measured relative to media duration because VAD emits speech "
            "anchors rather than silence; not content accuracy or WER"
        ),
    }
    return model, {"decoding": decoding, "coverage": coverage}, runtime


def build_teachobs_asr_job_manifest(
    media_manifest_path: str | Path,
    media_root: str | Path,
    caption_audit_path: str | Path,
    *,
    model_id: str,
    model_revision: str,
    model_files_sha256: str,
    faster_whisper_version: str,
    ctranslate2_version: str,
    container_image_digest: str,
    fallback_only: bool = True,
    lesson_ids: Sequence[str] | None = None,
    default_language: str = "auto",
    language_map_path: str | Path | None = None,
    min_timeline_span_fraction: float = 0.90,
    max_endpoint_gap_fraction: float = 0.10,
    require_all_selected_media: bool = False,
    duration_probe: Callable[[Path], float] = _ffprobe_duration,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a private content-free manifest for an offline GPU worker."""

    media_source, media_manifest, media_by_id = _media_records(media_manifest_path)
    caption_source, caption_audit = _validate_caption_audit(caption_audit_path)
    if media_manifest.get("repository_commit") != caption_audit.get("repository_commit"):
        raise ValueError("TeachObs media and caption evidence use different commits")
    caption_by_id = {item["lesson_id"]: item for item in caption_audit["records"]}
    if lesson_ids is None:
        requested = sorted(caption_by_id, key=lambda value: int(value[1:]))
    else:
        requested = [_safe_lesson_id(value) for value in lesson_ids]
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("ASR lesson selection must be non-empty and unique")
        if not set(requested).issubset(caption_by_id):
            raise ValueError("ASR lesson selection is absent from the caption audit")
    if fallback_only:
        requested = [
            lesson_id
            for lesson_id in requested
            if caption_by_id[lesson_id].get("status") != "caption_timeline_audited"
        ]
    model, policies, runtime_contract = _fixed_contract(
        model_id=model_id,
        model_revision=model_revision,
        model_files_sha256=model_files_sha256,
        faster_whisper_version=faster_whisper_version,
        ctranslate2_version=ctranslate2_version,
        container_image_digest=container_image_digest,
        min_timeline_span_fraction=min_timeline_span_fraction,
        max_endpoint_gap_fraction=max_endpoint_gap_fraction,
    )
    languages = _language_map(language_map_path, default_language)
    media_root_candidate = Path(media_root)
    if media_root_candidate.is_symlink() or not media_root_candidate.is_dir():
        raise ValueError("private TeachObs media root is missing or unsafe")
    media_root_path = media_root_candidate.resolve()
    jobs: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    for lesson_id in requested:
        item = media_by_id.get(lesson_id)
        if (
            not isinstance(item, dict)
            or not item.get("media_sha256")
            or not item.get("media_path")
        ):
            unavailable.append(
                {"lesson_id": lesson_id, "reason": "hash_bound_private_media_pending"}
            )
            continue
        binding = _validated_media_binding(
            item, media_root=media_root_path, duration_probe=duration_probe
        )
        language = languages.get(lesson_id, default_language)
        unsigned_job: dict[str, Any] = {
            "job_id": f"teachobs-asr-{lesson_id}",
            "lesson_id": lesson_id,
            "media_binding": binding,
            "language": {
                "requested": language,
                "automatic_detection_required": language == "auto",
            },
            "model": model,
            "decoding_config": policies["decoding"],
            "coverage_policy": policies["coverage"],
            "runtime_contract": runtime_contract,
            "expected_result_schema": RESULT_SCHEMA,
            "expected_result_relative_path": f"{lesson_id}.json",
            "caption_source_role": "audited_asr_fallback_candidate",
            "official_caption_claim_allowed": False,
        }
        unsigned_job["job_sha256"] = _canonical_sha256(unsigned_job)
        jobs.append(unsigned_job)
    if require_all_selected_media and unavailable:
        raise ValueError("not every selected TeachObs ASR lesson has hash-bound media")
    manifest: dict[str, Any] = {
        "schema": JOB_MANIFEST_SCHEMA,
        "dataset_id": DATASET_ID,
        "generated_at_utc": generated_at_utc or _utc_now(),
        "private_artifact": True,
        "contains_media_bytes": False,
        "contains_transcript_text": False,
        "public_release_authorized": False,
        "source_media_manifest_file_sha256": file_sha256(media_source),
        "source_caption_audit_file_sha256": file_sha256(caption_source),
        "repository_commit": media_manifest.get("repository_commit"),
        "selection": {
            "fallback_only": fallback_only,
            "requested_lesson_count": len(requested),
            "job_count": len(jobs),
            "media_pending_count": len(unavailable),
        },
        "model": model,
        "decoding_config": policies["decoding"],
        "coverage_policy": policies["coverage"],
        "runtime_contract": runtime_contract,
        "output_contract": {
            "schema": RESULT_SCHEMA,
            "one_json_object_per_lesson": True,
            "utf8_required": True,
            "segments_sorted_and_nonoverlapping": True,
            "segment_text_nonempty": True,
            "official_caption_claim_allowed": False,
        },
        "jobs": jobs,
        "media_pending": unavailable,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def validate_teachobs_asr_job_manifest(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != JOB_MANIFEST_SCHEMA or value.get("dataset_id") != DATASET_ID:
        raise ValueError("unsupported TeachObs ASR job manifest")
    digest = _verify_canonical_digest(
        value,
        digest_field="manifest_sha256",
        description="TeachObs ASR job manifest",
    )
    if value.get("contains_media_bytes") is not False or value.get("contains_transcript_text") is not False:
        raise ValueError("TeachObs ASR job manifest content boundary is invalid")
    if not _UTC_RE.fullmatch(str(value.get("generated_at_utc") or "")):
        raise ValueError("TeachObs ASR job manifest timestamp is invalid")
    model = value.get("model")
    decoding = value.get("decoding_config")
    coverage = value.get("coverage_policy")
    runtime = value.get("runtime_contract")
    if not all(isinstance(item, dict) for item in (model, decoding, coverage, runtime)):
        raise ValueError("TeachObs ASR job manifest contracts are malformed")
    assert isinstance(model, dict)
    assert isinstance(decoding, dict)
    assert isinstance(coverage, dict)
    assert isinstance(runtime, dict)
    try:
        expected_model, expected_policies, expected_runtime = _fixed_contract(
            model_id=str(model["model_id"]),
            model_revision=str(model["model_revision"]),
            model_files_sha256=str(model["model_snapshot_sha256_v1"]),
            faster_whisper_version=str(runtime["faster_whisper_version"]),
            ctranslate2_version=str(runtime["ctranslate2_version"]),
            container_image_digest=str(runtime["container_image_digest"]),
            min_timeline_span_fraction=float(
                coverage["minimum_timeline_span_fraction"]
            ),
            max_endpoint_gap_fraction=float(
                coverage["maximum_endpoint_gap_fraction"]
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("TeachObs ASR job manifest contracts are invalid") from exc
    if (
        model != expected_model
        or decoding != expected_policies["decoding"]
        or coverage != expected_policies["coverage"]
        or runtime != expected_runtime
    ):
        raise ValueError("TeachObs ASR job manifest fixed contract mismatch")
    jobs = value.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("TeachObs ASR job manifest jobs must be a list")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("TeachObs ASR job manifest has a non-object job")
        lesson_id = _safe_lesson_id(job.get("lesson_id"))
        if lesson_id in seen_ids:
            raise ValueError("TeachObs ASR job manifest has duplicate lessons")
        seen_ids.add(lesson_id)
        expected_path = _safe_relative_path(
            job.get("expected_result_relative_path"), suffix=".json"
        )
        if (
            len(PurePosixPath(expected_path).parts) != 1
            or PurePosixPath(expected_path).name != f"{lesson_id}.json"
        ):
            raise ValueError("TeachObs ASR result filename is not lesson-bound")
        if expected_path in seen_paths:
            raise ValueError("TeachObs ASR job manifest has duplicate result paths")
        seen_paths.add(expected_path)
        claimed_job_hash = _require_sha256(job.get("job_sha256"), field="job_sha256")
        unsigned = {key: item for key, item in job.items() if key != "job_sha256"}
        if claimed_job_hash != _canonical_sha256(unsigned):
            raise ValueError(f"TeachObs ASR job hash mismatch for {lesson_id}")
        if (
            job.get("model") != value.get("model")
            or job.get("decoding_config") != value.get("decoding_config")
            or job.get("coverage_policy") != value.get("coverage_policy")
            or job.get("runtime_contract") != value.get("runtime_contract")
            or job.get("official_caption_claim_allowed") is not False
        ):
            raise ValueError(f"TeachObs ASR job contract mismatch for {lesson_id}")
        binding = job.get("media_binding")
        if not isinstance(binding, dict) or binding.get("lesson_id") != lesson_id:
            raise ValueError(f"TeachObs ASR job media binding is malformed for {lesson_id}")
        _safe_relative_path(binding.get("media_relative_path"), suffix=".mp4")
        _require_sha256(binding.get("media_sha256"), field="media_sha256")
        language = job.get("language")
        if not isinstance(language, dict) or not _LANGUAGE_RE.fullmatch(
            str(language.get("requested") or "")
        ):
            raise ValueError(f"TeachObs ASR job language is invalid for {lesson_id}")
    selection = value.get("selection")
    if not isinstance(selection, dict) or selection.get("job_count") != len(jobs):
        raise ValueError("TeachObs ASR job manifest count mismatch")
    return {"manifest_sha256": digest, "job_count": len(jobs)}


def _timeline_coverage(
    segments: Sequence[dict[str, Any]],
    *,
    duration: float,
    maximum_segment_duration: float,
    policy: dict[str, Any],
) -> dict[str, Any]:
    if not segments:
        raise ValueError("ASR result contains no segments")
    first: float | None = None
    last = 0.0
    previous_end = 0.0
    union = 0.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError("ASR result contains a non-object segment")
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ASR result has invalid segment timestamps") from exc
        text = str(segment.get("text") or "").strip()
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or end > duration + 0.5
            or end - start > maximum_segment_duration + 1e-6
            or not text
            or "\x00" in text
            or len(text) > 20_000
        ):
            raise ValueError(f"ASR result has an invalid segment at index {index}")
        if index and start + 0.05 < previous_end:
            raise ValueError("ASR result segments are unsorted or overlapping")
        first = start if first is None else first
        last = end
        union += end - start
        previous_end = end
    assert first is not None
    initial_gap = first
    trailing_gap = max(0.0, duration - last)
    span = max(0.0, last - first)
    span_fraction = min(1.0, span / duration)
    initial_gap_fraction = min(1.0, initial_gap / duration)
    trailing_gap_fraction = min(1.0, trailing_gap / duration)
    passes = bool(
        span_fraction >= float(policy["minimum_timeline_span_fraction"])
        and initial_gap_fraction
        <= float(policy["maximum_endpoint_gap_fraction"])
        and trailing_gap_fraction
        <= float(policy["maximum_endpoint_gap_fraction"])
    )
    return {
        "segment_count": len(segments),
        "first_segment_start_seconds": round(first, 3),
        "last_segment_end_seconds": round(last, 3),
        "timeline_span_seconds": round(span, 3),
        "timeline_span_fraction": round(span_fraction, 6),
        "speech_segment_union_seconds": round(union, 3),
        "speech_segment_union_fraction": round(min(1.0, union / duration), 6),
        "initial_gap_seconds": round(initial_gap, 3),
        "trailing_gap_seconds": round(trailing_gap, 3),
        "initial_gap_fraction": round(initial_gap_fraction, 6),
        "trailing_gap_fraction": round(trailing_gap_fraction, 6),
        "endpoint_gap_policy_name": str(policy["policy_name"]),
        "absolute_endpoint_gap_seconds_gate_applied": False,
        "timeline_policy_passed": passes,
        "content_accuracy_established": False,
        "word_error_rate_established": False,
    }


def _postprocess_whisper_segments(
    source_segments: Sequence[Any],
    *,
    maximum_segment_duration: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split overlong segments on positive-duration word anchors.

    faster-whisper may restore a non-empty word to a zero-width point timestamp.
    Such text is assigned to the temporally nearest preceding-word end or
    following-word start (ties go to the preceding word).  Point timestamps
    never become output boundaries, and attachment never changes a positive
    word's timestamps.
    """

    if (
        not math.isfinite(maximum_segment_duration)
        or maximum_segment_duration <= 0
    ):
        raise ValueError("ASR maximum segment duration must be positive and finite")

    output: list[dict[str, Any]] = []
    split_count = 0
    overlong_source_word_count = 0
    positive_duration_word_count = 0
    attached_zero_duration_word_count = 0
    maximum_observed_attachment_distance = 0.0

    for source_index, segment in enumerate(source_segments):
        try:
            start = float(segment.start)
            end = float(segment.end)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"ASR engine segment timestamps are invalid at index {source_index}"
            ) from exc
        text = str(getattr(segment, "text", "") or "").strip()
        rounded_start = round(start, 3)
        rounded_end = round(end, 3)
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or rounded_end <= rounded_start
            or not text
        ):
            raise ValueError(
                f"ASR engine segment is invalid at index {source_index}"
            )
        if (
            rounded_end - rounded_start
            <= maximum_segment_duration + 1e-6
        ):
            output.append(
                {"start": rounded_start, "end": rounded_end, "text": text}
            )
            continue

        raw_words = getattr(segment, "words", None)
        if not raw_words:
            raise ValueError(
                "ASR overlong engine segment lacks required word timestamps"
            )
        positive_anchors: list[dict[str, Any]] = []
        zero_point_words: list[dict[str, Any]] = []
        previous_word_start: float | None = None
        previous_positive_end: float | None = None
        for word_index, word in enumerate(raw_words):
            try:
                word_start = float(word.start)
                word_end = float(word.end)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "ASR overlong engine segment has invalid word timestamps"
                ) from exc
            word_text = str(getattr(word, "word", "") or "")
            if (
                not math.isfinite(word_start)
                or not math.isfinite(word_end)
                or word_end < word_start
                or not word_text.strip()
                or (
                    previous_word_start is not None
                    and word_start < previous_word_start
                )
            ):
                raise ValueError(
                    "ASR overlong engine segment has an invalid word at "
                    f"index {word_index}"
                )
            overlong_source_word_count += 1
            if word_end == word_start:
                if word_start < start or word_start > end:
                    raise ValueError(
                        "ASR zero-duration word point anchor is outside its "
                        "source segment"
                    )
                record = {
                    "raw_index": word_index,
                    "point": word_start,
                    "text": word_text,
                }
                zero_point_words.append(record)
                attached_zero_duration_word_count += 1
                previous_word_start = word_start
                continue

            if (
                word_start < start - 0.5
                or word_end > end + 0.5
                or (
                    previous_positive_end is not None
                    and word_start + 0.05 < previous_positive_end
                )
            ):
                raise ValueError(
                    "ASR overlong engine segment has an invalid positive-duration "
                    f"word at index {word_index}"
                )
            rounded_word_start = round(word_start, 3)
            rounded_word_end = round(word_end, 3)
            if (
                rounded_word_end <= rounded_word_start
                or rounded_word_end - rounded_word_start
                > maximum_segment_duration + 1e-6
            ):
                raise ValueError(
                    "ASR positive-duration word has invalid bounded timestamps"
                )
            record = {
                "raw_index": word_index,
                "start": word_start,
                "end": word_end,
                "text": word_text,
            }
            positive_anchors.append(record)
            positive_duration_word_count += 1
            previous_word_start = word_start
            previous_positive_end = word_end
        if not positive_anchors:
            raise ValueError(
                "ASR overlong engine segment has no positive-duration word anchors"
            )

        before_text: list[list[str]] = [[] for _ in positive_anchors]
        after_text: list[list[str]] = [[] for _ in positive_anchors]
        previous_assignment: int | None = None
        for point_word in zero_point_words:
            raw_index = int(point_word["raw_index"])
            point = float(point_word["point"])
            previous_anchor_index: int | None = None
            next_anchor_index: int | None = None
            for anchor_index, anchor in enumerate(positive_anchors):
                anchor_raw_index = int(anchor["raw_index"])
                if anchor_raw_index < raw_index:
                    previous_anchor_index = anchor_index
                    continue
                if anchor_raw_index > raw_index:
                    next_anchor_index = anchor_index
                    break

            previous_distance = (
                abs(point - float(positive_anchors[previous_anchor_index]["end"]))
                if previous_anchor_index is not None
                else None
            )
            next_distance = (
                abs(float(positive_anchors[next_anchor_index]["start"]) - point)
                if next_anchor_index is not None
                else None
            )
            if previous_distance is not None and (
                next_distance is None or previous_distance <= next_distance
            ):
                assigned_anchor_index = previous_anchor_index
                attachment_distance = previous_distance
            elif next_anchor_index is not None and next_distance is not None:
                assigned_anchor_index = next_anchor_index
                attachment_distance = next_distance
            else:
                raise ValueError(
                    "ASR zero-duration word has no positive-duration attachment "
                    "candidate"
                )
            if (
                not math.isfinite(attachment_distance)
                or attachment_distance > maximum_segment_duration
            ):
                raise ValueError(
                    "ASR zero-duration word exceeds the maximum attachment distance"
                )
            if (
                previous_assignment is not None
                and assigned_anchor_index < previous_assignment
            ):
                raise ValueError(
                    "ASR zero-duration word assignments are not monotonic"
                )
            previous_assignment = assigned_anchor_index
            maximum_observed_attachment_distance = max(
                maximum_observed_attachment_distance,
                attachment_distance,
            )
            if raw_index < int(
                positive_anchors[assigned_anchor_index]["raw_index"]
            ):
                before_text[assigned_anchor_index].append(str(point_word["text"]))
            else:
                after_text[assigned_anchor_index].append(str(point_word["text"]))

        words = [
            {
                "start": anchor["start"],
                "end": anchor["end"],
                "text": (
                    "".join(before_text[anchor_index])
                    + str(anchor["text"])
                    + "".join(after_text[anchor_index])
                ),
            }
            for anchor_index, anchor in enumerate(positive_anchors)
        ]
        if "".join(str(word["text"]) for word in words).strip() != text:
            raise ValueError(
                "ASR word timestamps do not preserve the source segment text"
            )

        chunk: list[dict[str, Any]] = []
        grouped_words: list[list[dict[str, Any]]] = []
        for word in words:
            candidate_start = round(
                float((chunk[0] if chunk else word)["start"]), 3
            )
            candidate_end = round(float(word["end"]), 3)
            if (
                chunk
                and candidate_end - candidate_start
                > maximum_segment_duration + 1e-6
            ):
                grouped_words.append(chunk)
                chunk = []
            chunk.append(word)
        grouped_words.append(chunk)
        if len(grouped_words) < 2:
            raise ValueError(
                "ASR overlong engine segment cannot be split into multiple "
                "bounded word groups"
            )
        split_segments: list[dict[str, Any]] = []
        for chunk_index, chunk_words in enumerate(grouped_words):
            if not chunk_words:
                raise ValueError(
                    "ASR long-segment split produced an empty word group"
                )
            chunk_start = round(float(chunk_words[0]["start"]), 3)
            chunk_end = round(float(chunk_words[-1]["end"]), 3)
            chunk_text = "".join(str(word["text"]) for word in chunk_words)
            if chunk_index == 0:
                chunk_text = chunk_text.lstrip()
            if chunk_index == len(grouped_words) - 1:
                chunk_text = chunk_text.rstrip()
            if (
                chunk_end <= chunk_start
                or chunk_end - chunk_start
                > maximum_segment_duration + 1e-6
                or not chunk_text.strip()
            ):
                raise ValueError(
                    "ASR long-segment split produced an invalid chunk"
                )
            split_segments.append(
                {
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": chunk_text,
                }
            )
        if "".join(segment["text"] for segment in split_segments) != text:
            raise ValueError("ASR long-segment split changed the segment text")
        output.extend(split_segments)
        split_count += 1

    return output, {
        "policy": LONG_SEGMENT_POLICY,
        "maximum_output_segment_duration_seconds": maximum_segment_duration,
        "source_segment_count": len(source_segments),
        "source_segments_split": split_count,
        "output_segment_count": len(output),
        "overlong_source_word_timestamp_count": overlong_source_word_count,
        "positive_duration_word_anchor_count": positive_duration_word_count,
        "attached_zero_duration_word_count": attached_zero_duration_word_count,
        "zero_duration_word_attachment_policy": (
            ZERO_DURATION_WORD_ATTACHMENT_POLICY
        ),
        "zero_duration_word_maximum_attachment_distance_seconds": (
            maximum_segment_duration
        ),
        "maximum_observed_zero_duration_word_attachment_distance_seconds": (
            round(maximum_observed_attachment_distance, 6)
        ),
        "zero_duration_point_anchors_within_source_segment_verified": True,
        "zero_duration_assignment_anchor_indices_monotonic_verified": True,
        "positive_word_boundaries_preserved": True,
        "split_text_trimmed_exact_equivalence_verified": True,
        "reference_or_label_used": False,
        "boundary_selection_uses_text_content": False,
    }


def _validate_runtime(runtime: Any, contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        raise ValueError("ASR result runtime provenance is missing")
    if runtime.get("runner_name") != RUNNER_NAME:
        raise ValueError("ASR result runner identity mismatch")
    runner_source_sha256 = _require_sha256(
        runtime.get("runner_source_sha256"), field="runner_source_sha256"
    )
    if runner_source_sha256 != contract.get("runner_module_sha256"):
        raise ValueError("ASR result runner source differs from job contract")
    versions = runtime.get("package_versions")
    if not isinstance(versions, dict) or (
        versions.get("faster-whisper") != contract.get("faster_whisper_version")
        or versions.get("ctranslate2") != contract.get("ctranslate2_version")
    ):
        raise ValueError("ASR result runtime package versions differ from contract")
    python_version = str(runtime.get("python_version") or "")
    if not python_version or len(python_version) > 128:
        raise ValueError("ASR result Python runtime provenance is invalid")
    accelerator = runtime.get("accelerator")
    if (
        not isinstance(accelerator, dict)
        or accelerator.get("device_type") != "cuda"
        or not isinstance(accelerator.get("device_count"), int)
        or accelerator["device_count"] < 1
        or not str(accelerator.get("device_name") or "").strip()
        or not str(accelerator.get("driver_version") or "").strip()
    ):
        raise ValueError("ASR result lacks verified CUDA accelerator provenance")
    container_image_digest = str(runtime.get("container_image_digest") or "")
    if not _CONTAINER_RE.fullmatch(container_image_digest):
        raise ValueError("ASR result requires a content-addressed container image")
    if container_image_digest != contract.get("container_image_digest"):
        raise ValueError("ASR result container image differs from job contract")
    if not _UTC_RE.fullmatch(str(runtime.get("executed_at_utc") or "")):
        raise ValueError("ASR result execution timestamp is invalid")
    return runtime


def validate_teachobs_asr_result(
    result: dict[str, Any],
    *,
    manifest_sha256: str,
    job: dict[str, Any],
) -> dict[str, Any]:
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported TeachObs ASR result schema")
    _verify_canonical_digest(
        result, digest_field="result_sha256", description="TeachObs ASR result"
    )
    lesson_id = job["lesson_id"]
    if (
        result.get("job_manifest_sha256") != manifest_sha256
        or result.get("job_sha256") != job.get("job_sha256")
        or result.get("job_id") != job.get("job_id")
        or result.get("lesson_id") != lesson_id
    ):
        raise ValueError(f"ASR result job binding mismatch for {lesson_id}")
    if (
        result.get("transcript_kind") != "automatic_speech_recognition"
        or result.get("source_tier") != "audited_asr_fallback_candidate"
        or result.get("official_caption") is not False
        or result.get("human_content_review_completed") is not False
        or result.get("content_accuracy_established") is not False
        or result.get("word_error_rate_established") is not False
    ):
        raise ValueError(f"ASR result contains an invalid evidence claim for {lesson_id}")
    if (
        result.get("media_binding") != job.get("media_binding")
        or result.get("model") != job.get("model")
        or result.get("decoding_config") != job.get("decoding_config")
        or result.get("language", {}).get("requested")
        != job.get("language", {}).get("requested")
    ):
        raise ValueError(f"ASR result provenance differs from job for {lesson_id}")
    language = result.get("language")
    if not isinstance(language, dict):
        raise ValueError(f"ASR result language is missing for {lesson_id}")
    detected = str(language.get("detected") or "").strip()
    if not _LANGUAGE_RE.fullmatch(detected) or detected == "auto":
        raise ValueError(f"ASR result detected language is invalid for {lesson_id}")
    try:
        probability = float(language["detected_probability"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"ASR result language probability is invalid for {lesson_id}") from exc
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"ASR result language probability is invalid for {lesson_id}")
    model_runtime = result.get("model_runtime")
    if (
        not isinstance(model_runtime, dict)
        or model_runtime.get("model_snapshot_sha256_v1")
        != job["model"]["model_snapshot_sha256_v1"]
        or model_runtime.get("network_access_enabled") is not False
        or model_runtime.get("local_files_only") is not True
    ):
        raise ValueError(f"ASR model runtime binding mismatch for {lesson_id}")
    _validate_runtime(result.get("runtime"), job["runtime_contract"])
    duration = float(job["media_binding"]["media_duration_seconds"])
    maximum = float(job["decoding_config"]["maximum_segment_duration_seconds"])
    segments = result.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"ASR result segments are missing for {lesson_id}")
    postprocessing = result.get("segment_postprocessing")
    if not isinstance(postprocessing, dict):
        raise ValueError(
            f"ASR result segment postprocessing is missing for {lesson_id}"
        )
    try:
        postprocessing_maximum = float(
            postprocessing["maximum_output_segment_duration_seconds"]
        )
        maximum_attachment_distance = float(
            postprocessing[
                "zero_duration_word_maximum_attachment_distance_seconds"
            ]
        )
        maximum_observed_attachment_distance = float(
            postprocessing[
                "maximum_observed_zero_duration_word_attachment_distance_seconds"
            ]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"ASR result segment postprocessing is invalid for {lesson_id}"
        ) from exc
    source_segment_count = postprocessing.get("source_segment_count")
    source_segments_split = postprocessing.get("source_segments_split")
    output_segment_count = postprocessing.get("output_segment_count")
    overlong_source_word_count = postprocessing.get(
        "overlong_source_word_timestamp_count"
    )
    positive_duration_word_count = postprocessing.get(
        "positive_duration_word_anchor_count"
    )
    attached_zero_duration_word_count = postprocessing.get(
        "attached_zero_duration_word_count"
    )
    if (
        postprocessing.get("policy")
        != job["decoding_config"].get("long_segment_postprocessing_policy")
        or postprocessing.get("policy") != LONG_SEGMENT_POLICY
        or postprocessing_maximum != maximum
        or not isinstance(source_segment_count, int)
        or isinstance(source_segment_count, bool)
        or source_segment_count < 1
        or not isinstance(source_segments_split, int)
        or isinstance(source_segments_split, bool)
        or not 0 <= source_segments_split <= source_segment_count
        or not isinstance(output_segment_count, int)
        or isinstance(output_segment_count, bool)
        or output_segment_count != len(segments)
        or output_segment_count < source_segment_count
        or output_segment_count - source_segment_count < source_segments_split
        or (source_segments_split == 0)
        != (output_segment_count == source_segment_count)
        or not isinstance(overlong_source_word_count, int)
        or isinstance(overlong_source_word_count, bool)
        or overlong_source_word_count < 0
        or not isinstance(positive_duration_word_count, int)
        or isinstance(positive_duration_word_count, bool)
        or positive_duration_word_count < 0
        or not isinstance(attached_zero_duration_word_count, int)
        or isinstance(attached_zero_duration_word_count, bool)
        or attached_zero_duration_word_count < 0
        or overlong_source_word_count
        != positive_duration_word_count + attached_zero_duration_word_count
        or (source_segments_split == 0)
        != (overlong_source_word_count == 0)
        or (
            source_segments_split > 0
            and positive_duration_word_count < source_segments_split * 2
        )
        or postprocessing.get("zero_duration_word_attachment_policy")
        != job["decoding_config"].get(
            "zero_duration_word_attachment_policy"
        )
        or postprocessing.get("zero_duration_word_attachment_policy")
        != ZERO_DURATION_WORD_ATTACHMENT_POLICY
        or maximum_attachment_distance
        != job["decoding_config"].get(
            "zero_duration_word_maximum_attachment_distance_seconds"
        )
        or maximum_attachment_distance != maximum
        or not math.isfinite(maximum_observed_attachment_distance)
        or maximum_observed_attachment_distance < 0
        or maximum_observed_attachment_distance
        > maximum_attachment_distance
        or (
            attached_zero_duration_word_count == 0
            and maximum_observed_attachment_distance != 0
        )
        or postprocessing.get(
            "zero_duration_point_anchors_within_source_segment_verified"
        )
        is not True
        or postprocessing.get(
            "zero_duration_assignment_anchor_indices_monotonic_verified"
        )
        is not True
        or postprocessing.get("positive_word_boundaries_preserved") is not True
        or postprocessing.get(
            "split_text_trimmed_exact_equivalence_verified"
        )
        is not True
        or postprocessing.get("reference_or_label_used") is not False
        or postprocessing.get("boundary_selection_uses_text_content")
        is not False
    ):
        raise ValueError(
            f"ASR result segment postprocessing is invalid for {lesson_id}"
        )
    computed = _timeline_coverage(
        segments,
        duration=duration,
        maximum_segment_duration=maximum,
        policy=job["coverage_policy"],
    )
    declared = result.get("timeline_coverage")
    if declared != computed:
        raise ValueError(f"ASR result timeline coverage mismatch for {lesson_id}")
    if not computed["timeline_policy_passed"]:
        raise ValueError(f"ASR result timeline policy failed for {lesson_id}")
    return computed


def _platform_track(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("status") != "caption_timeline_audited":
        return None
    tracks = record.get("tracks")
    if not isinstance(tracks, list):
        return None
    candidates: list[dict[str, Any]] = []
    for track in tracks:
        timeline = track.get("timeline") if isinstance(track, dict) else None
        if (
            isinstance(timeline, dict)
            and timeline.get("timeline_audit_completed") is True
            and float(timeline.get("timeline_span_coverage_fraction", 0)) >= 0.90
        ):
            candidates.append(track)
    candidates.sort(
        key=lambda item: (
            item.get("track_type") != "manual_creator_provided",
            "original_language" not in item.get("roles", []),
            str(item.get("language") or ""),
        )
    )
    return candidates[0] if candidates else None


def build_teachobs_transcript_coverage_matrix(
    caption_audit: dict[str, Any],
    asr_import_audit: dict[str, Any] | None,
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    records = caption_audit.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_LESSON_COUNT:
        raise ValueError("TeachObs transcript coverage requires all caption records")
    asr_by_id: dict[str, dict[str, Any]] = {}
    if asr_import_audit is not None:
        if asr_import_audit.get("schema") != IMPORT_AUDIT_SCHEMA:
            raise ValueError("unsupported TeachObs ASR import audit")
        _verify_canonical_digest(
            asr_import_audit,
            digest_field="audit_sha256",
            description="TeachObs ASR import audit",
        )
        asr_by_id = {
            item["lesson_id"]: item
            for item in asr_import_audit.get("records", [])
            if isinstance(item, dict)
        }
    rows: list[dict[str, Any]] = []
    tier_counts = {
        "platform_creator_provided_caption": 0,
        "platform_automatic_caption": 0,
        "audited_asr_fallback": 0,
        "pending_no_audited_source": 0,
    }
    for record in sorted(records, key=lambda item: int(item["lesson_id"][1:])):
        lesson_id = _safe_lesson_id(record.get("lesson_id"))
        track = _platform_track(record)
        asr = asr_by_id.get(lesson_id)
        if track is not None:
            tier = (
                "platform_creator_provided_caption"
                if track.get("track_type") == "manual_creator_provided"
                else "platform_automatic_caption"
            )
            source_hash = track.get("caption_sha256")
            coverage = track["timeline"].get("timeline_span_coverage_fraction")
            asr_used = False
        elif asr is not None and asr.get("status") == "asr_provenance_and_timeline_audited":
            tier = "audited_asr_fallback"
            source_hash = asr.get("result_sha256")
            coverage = asr.get("timeline_coverage", {}).get("timeline_span_fraction")
            asr_used = True
        else:
            tier = "pending_no_audited_source"
            source_hash = None
            coverage = None
            asr_used = False
        tier_counts[tier] += 1
        rows.append(
            {
                "lesson_id": lesson_id,
                "selected_source_tier": tier,
                "selected_source_sha256": source_hash,
                "timeline_span_fraction": coverage,
                "platform_caption_preferred_over_asr": True,
                "asr_fallback_used": asr_used,
                "asr_is_official_caption": False,
                "content_accuracy_established": False,
                "word_error_rate_established": False,
                "independent_human_content_audit_completed": False,
            }
        )
    covered = EXPECTED_LESSON_COUNT - tier_counts["pending_no_audited_source"]
    matrix: dict[str, Any] = {
        "schema": COVERAGE_MATRIX_SCHEMA,
        "dataset_id": DATASET_ID,
        "generated_at_utc": generated_at_utc or _utc_now(),
        "private_artifact": True,
        "public_release_authorized": False,
        "source_priority": [
            "platform_creator_provided_caption",
            "platform_automatic_caption",
            "audited_asr_fallback",
            "pending_no_audited_source",
        ],
        "aggregate": {
            "expected_lesson_count": EXPECTED_LESSON_COUNT,
            "covered_lesson_count": covered,
            "pending_lesson_count": EXPECTED_LESSON_COUNT - covered,
            **{f"{key}_lesson_count": value for key, value in tier_counts.items()},
            "transcript_source_coverage_complete": covered == EXPECTED_LESSON_COUNT,
        },
        "claims": {
            "asr_is_official_caption": False,
            "asr_provenance_and_timeline_audit_completed": bool(
                tier_counts["audited_asr_fallback"]
            ),
            "content_accuracy_established": False,
            "word_error_rate_established": False,
            "independent_human_content_audit_completed": False,
        },
        "rows": rows,
    }
    matrix["matrix_sha256"] = _canonical_sha256(matrix)
    return matrix


def import_teachobs_asr_results(
    job_manifest_path: str | Path,
    media_manifest_path: str | Path,
    media_root: str | Path,
    caption_audit_path: str | Path,
    results_directory: str | Path,
    *,
    duration_probe: Callable[[Path], float] = _ffprobe_duration,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Strictly validate present result files; missing jobs remain pending."""

    manifest_path, manifest = _load_object(
        job_manifest_path, description="TeachObs ASR job manifest"
    )
    validation = validate_teachobs_asr_job_manifest(manifest)
    media_source, _, media_by_id = _media_records(media_manifest_path)
    if file_sha256(media_source) != manifest.get("source_media_manifest_file_sha256"):
        raise ValueError("TeachObs ASR job manifest is bound to another media manifest")
    caption_source, caption_audit = _validate_caption_audit(caption_audit_path)
    if file_sha256(caption_source) != manifest.get("source_caption_audit_file_sha256"):
        raise ValueError("TeachObs ASR job manifest is bound to another caption audit")
    results_candidate = Path(results_directory)
    if results_candidate.is_symlink() or not results_candidate.is_dir():
        raise ValueError("TeachObs ASR results directory is missing or unsafe")
    results_root = results_candidate.resolve()
    media_root_candidate = Path(media_root)
    if media_root_candidate.is_symlink() or not media_root_candidate.is_dir():
        raise ValueError("TeachObs private media root is missing or unsafe")
    media_root_path = media_root_candidate.resolve()
    expected_names = {
        PurePosixPath(job["expected_result_relative_path"]).name
        for job in manifest["jobs"]
    }
    actual_names = set()
    for path in results_root.rglob("*.json"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("TeachObs ASR results contain an unsafe JSON path")
        actual_names.add(path.relative_to(results_root).as_posix())
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        raise ValueError("TeachObs ASR results contain unbound JSON files")
    records: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        lesson_id = job["lesson_id"]
        current_binding = _validated_media_binding(
            media_by_id[lesson_id],
            media_root=media_root_path,
            duration_probe=duration_probe,
        )
        if current_binding != job["media_binding"]:
            raise ValueError(f"private media changed after ASR handoff for {lesson_id}")
        result_path = results_root / job["expected_result_relative_path"]
        if not result_path.exists():
            records.append(
                {
                    "lesson_id": lesson_id,
                    "status": "asr_result_pending",
                    "job_sha256": job["job_sha256"],
                    "result_sha256": None,
                    "timeline_coverage": None,
                }
            )
            continue
        if result_path.is_symlink() or not result_path.is_file():
            raise ValueError(f"unsafe TeachObs ASR result path for {lesson_id}")
        result = read_json(result_path)
        if not isinstance(result, dict):
            raise ValueError(f"TeachObs ASR result must be an object for {lesson_id}")
        coverage = validate_teachobs_asr_result(
            result,
            manifest_sha256=validation["manifest_sha256"],
            job=job,
        )
        records.append(
            {
                "lesson_id": lesson_id,
                "status": "asr_provenance_and_timeline_audited",
                "job_sha256": job["job_sha256"],
                "result_sha256": file_sha256(result_path),
                "result_canonical_sha256": result["result_sha256"],
                "timeline_coverage": coverage,
                "official_caption": False,
                "human_content_review_completed": False,
                "content_accuracy_established": False,
                "word_error_rate_established": False,
            }
        )
    completed = sum(
        item["status"] == "asr_provenance_and_timeline_audited" for item in records
    )
    timestamp = generated_at_utc or _utc_now()
    audit: dict[str, Any] = {
        "schema": IMPORT_AUDIT_SCHEMA,
        "dataset_id": DATASET_ID,
        "generated_at_utc": timestamp,
        "private_artifact": True,
        "public_release_authorized": False,
        "job_manifest_file_sha256": file_sha256(manifest_path),
        "job_manifest_canonical_sha256": validation["manifest_sha256"],
        "source_media_manifest_file_sha256": file_sha256(media_source),
        "source_caption_audit_file_sha256": file_sha256(caption_source),
        "model_contract_sha256": _canonical_sha256(manifest["model"]),
        "decoding_contract_sha256": _canonical_sha256(manifest["decoding_config"]),
        "runtime_contract_sha256": _canonical_sha256(manifest["runtime_contract"]),
        "aggregate": {
            "job_count": len(records),
            "valid_result_count": completed,
            "pending_result_count": len(records) - completed,
            "asr_job_set_complete": bool(records and completed == len(records)),
        },
        "claims": {
            "asr_provenance_and_timeline_audit_completed": bool(
                records and completed == len(records)
            ),
            "asr_is_official_caption": False,
            "independent_human_content_audit_completed": False,
            "content_accuracy_established": False,
            "word_error_rate_established": False,
        },
        "records": records,
    }
    audit["audit_sha256"] = _canonical_sha256(audit)
    matrix = build_teachobs_transcript_coverage_matrix(
        caption_audit, audit, generated_at_utc=timestamp
    )
    return {"audit": audit, "coverage_matrix": matrix}


def build_pending_teachobs_asr_receipt(
    media_manifest_path: str | Path,
    caption_audit_path: str | Path,
    *,
    job_manifest: dict[str, Any] | None = None,
    job_manifest_path: str | Path | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build an honest aggregate receipt before any ASR job has run."""

    media_source, _, media_by_id = _media_records(media_manifest_path)
    caption_source, caption_audit = _validate_caption_audit(caption_audit_path)
    matrix = build_teachobs_transcript_coverage_matrix(
        caption_audit, None, generated_at_utc=generated_at_utc
    )
    media_ready = sum(
        bool(item.get("media_sha256") and item.get("media_path"))
        for item in media_by_id.values()
    )
    media_ready_ids = {
        lesson_id
        for lesson_id, item in media_by_id.items()
        if item.get("media_sha256") and item.get("media_path")
    }
    fallback_needed_ids = {
        record["lesson_id"]
        for record in caption_audit["records"]
        if _platform_track(record) is None
    }
    fallback_media_ready = len(media_ready_ids & fallback_needed_ids)
    job_count = 0
    job_file_sha256: str | None = None
    model_contract_sha256: str | None = None
    decoding_contract_sha256: str | None = None
    runtime_contract_sha256: str | None = None
    if job_manifest is not None:
        validate_teachobs_asr_job_manifest(job_manifest)
        if (
            job_manifest.get("source_media_manifest_file_sha256")
            != file_sha256(media_source)
            or job_manifest.get("source_caption_audit_file_sha256")
            != file_sha256(caption_source)
        ):
            raise ValueError("pending ASR receipt job manifest source binding mismatch")
        job_count = len(job_manifest["jobs"])
        model_contract_sha256 = _canonical_sha256(job_manifest["model"])
        decoding_contract_sha256 = _canonical_sha256(
            job_manifest["decoding_config"]
        )
        runtime_contract_sha256 = _canonical_sha256(
            job_manifest["runtime_contract"]
        )
        if job_manifest_path is not None:
            candidate = Path(job_manifest_path)
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("pending ASR receipt job manifest path is unsafe")
            job_file_sha256 = file_sha256(candidate)
    return _public_receipt(
        aggregate={
            **matrix["aggregate"],
            "hash_bound_private_media_lesson_count": media_ready,
            "asr_fallback_needed_lesson_count": len(fallback_needed_ids),
            "asr_fallback_media_ready_lesson_count": fallback_media_ready,
            "asr_fallback_media_pending_lesson_count": (
                len(fallback_needed_ids) - fallback_media_ready
            ),
            "asr_job_count": job_count,
            "valid_asr_result_count": 0,
            "pending_asr_result_count": job_count,
        },
        source_hashes={
            "media_manifest_file_sha256": file_sha256(media_source),
            "caption_audit_file_sha256": file_sha256(caption_source),
            "job_manifest_file_sha256": job_file_sha256,
            "asr_import_audit_file_sha256": None,
            "coverage_matrix_file_sha256": None,
            "model_contract_sha256": model_contract_sha256,
            "decoding_contract_sha256": decoding_contract_sha256,
            "runtime_contract_sha256": runtime_contract_sha256,
        },
        completed=False,
        generated_at_utc=generated_at_utc or _utc_now(),
    )


def build_public_teachobs_asr_receipt(
    asr_import_audit: dict[str, Any],
    coverage_matrix: dict[str, Any],
    *,
    asr_import_audit_path: str | Path,
    coverage_matrix_path: str | Path,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    _verify_canonical_digest(
        asr_import_audit,
        digest_field="audit_sha256",
        description="TeachObs ASR import audit",
    )
    _verify_canonical_digest(
        coverage_matrix,
        digest_field="matrix_sha256",
        description="TeachObs transcript coverage matrix",
    )
    audit_path = Path(asr_import_audit_path)
    matrix_path = Path(coverage_matrix_path)
    if any(path.is_symlink() or not path.is_file() for path in (audit_path, matrix_path)):
        raise ValueError("private TeachObs ASR receipt inputs are missing or unsafe")
    aggregate = {
        **coverage_matrix["aggregate"],
        "hash_bound_private_media_lesson_count": None,
        "asr_job_count": asr_import_audit["aggregate"]["job_count"],
        "valid_asr_result_count": asr_import_audit["aggregate"][
            "valid_result_count"
        ],
        "pending_asr_result_count": asr_import_audit["aggregate"][
            "pending_result_count"
        ],
    }
    completed = bool(
        coverage_matrix["aggregate"]["transcript_source_coverage_complete"]
        and asr_import_audit["aggregate"]["asr_job_set_complete"]
    )
    return _public_receipt(
        aggregate=aggregate,
        source_hashes={
            "media_manifest_file_sha256": asr_import_audit[
                "source_media_manifest_file_sha256"
            ],
            "caption_audit_file_sha256": asr_import_audit[
                "source_caption_audit_file_sha256"
            ],
            "job_manifest_file_sha256": asr_import_audit[
                "job_manifest_file_sha256"
            ],
            "asr_import_audit_file_sha256": file_sha256(audit_path),
            "coverage_matrix_file_sha256": file_sha256(matrix_path),
            "model_contract_sha256": asr_import_audit["model_contract_sha256"],
            "decoding_contract_sha256": asr_import_audit[
                "decoding_contract_sha256"
            ],
            "runtime_contract_sha256": asr_import_audit["runtime_contract_sha256"],
        },
        completed=completed,
        generated_at_utc=generated_at_utc or _utc_now(),
    )


def _public_receipt(
    *,
    aggregate: dict[str, Any],
    source_hashes: dict[str, Any],
    completed: bool,
    generated_at_utc: str,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "artifact_kind": "teachobs_aggregate_audited_asr_handoff_receipt",
        "schema": PUBLIC_RECEIPT_SCHEMA,
        "generated_at_utc": generated_at_utc,
        "source_hashes": source_hashes,
        "aggregate": aggregate,
        "evidence_status": {
            "handoff_status": "completed" if completed else "pending",
            "transcript_source_coverage_complete": bool(
                aggregate.get("transcript_source_coverage_complete")
            ),
            "asr_provenance_and_timeline_audit_completed": bool(
                aggregate.get("valid_asr_result_count")
                and not aggregate.get("pending_asr_result_count")
            ),
            "asr_is_official_caption": False,
            "independent_human_content_audit_completed": False,
            "content_accuracy_established": False,
            "word_error_rate_established": False,
            "recognition_accuracy_established": False,
        },
        "source_priority": (
            "platform creator-provided caption, then platform automatic caption, "
            "then technically audited ASR fallback"
        ),
        "metric_boundary": (
            "Hash, runtime, model, decoding, timestamp, and aggregate timeline "
            "checks do not establish transcript content accuracy or WER."
        ),
        "content_exclusion": {
            "media_bytes_included": False,
            "transcript_text_included": False,
            "source_urls_included": False,
            "lesson_or_video_ids_included": False,
            "filesystem_paths_included": False,
            "per_lesson_records_included": False,
            "runtime_host_identity_included": False,
        },
    }
    serialized = json.dumps(receipt, ensure_ascii=False)
    if (
        "http://" in serialized.casefold()
        or "https://" in serialized.casefold()
        or re.search(
            r"(?<![A-Za-z0-9])S(?:[1-9]|[12][0-9]|30)(?![A-Za-z0-9])",
            serialized,
        )
    ):
        raise ValueError("public TeachObs ASR receipt leaked private identity data")
    return receipt


def write_teachobs_asr_import(
    result: dict[str, Any],
    *,
    audit_path: str | Path,
    coverage_matrix_path: str | Path,
    public_receipt_path: str | Path | None = None,
) -> dict[str, Path]:
    audit_target = write_json(audit_path, result["audit"])
    matrix_target = write_json(coverage_matrix_path, result["coverage_matrix"])
    outputs = {"audit": audit_target, "coverage_matrix": matrix_target}
    if public_receipt_path is not None:
        receipt = build_public_teachobs_asr_receipt(
            result["audit"],
            result["coverage_matrix"],
            asr_import_audit_path=audit_target,
            coverage_matrix_path=matrix_target,
        )
        outputs["public_receipt"] = write_json(public_receipt_path, receipt)
    return outputs


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"GPU ASR runner requires exact package `{name}`") from exc


def _gpu_runtime(container_image_digest: str, runner_source_sha256: str) -> dict[str, Any]:
    if not _CONTAINER_RE.fullmatch(container_image_digest):
        raise ValueError("container image digest must be sha256:<64 lowercase hex>")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not rows or "," not in rows[0]:
        raise RuntimeError("GPU ASR runner requires a visible NVIDIA CUDA device")
    device_name, driver_version = (part.strip() for part in rows[0].split(",", 1))
    return {
        "runner_name": RUNNER_NAME,
        "runner_source_sha256": _require_sha256(
            runner_source_sha256, field="runner_source_sha256"
        ),
        "python_version": platform.python_version(),
        "package_versions": {
            "faster-whisper": _package_version("faster-whisper"),
            "ctranslate2": _package_version("ctranslate2"),
        },
        "accelerator": {
            "device_type": "cuda",
            "device_count": len(rows),
            "device_name": device_name,
            "driver_version": driver_version,
        },
        "container_image_digest": container_image_digest,
        "executed_at_utc": _utc_now(),
        "hostname_recorded": False,
    }


def run_teachobs_asr_gpu_jobs(
    job_manifest_path: str | Path,
    media_root: str | Path,
    model_directory: str | Path,
    output_directory: str | Path,
    *,
    container_image_digest: str,
    runner_source_sha256: str,
    lesson_ids: Sequence[str] | None = None,
    duration_probe: Callable[[Path], float] = _ffprobe_duration,
) -> dict[str, Any]:
    """Run faster-whisper from a prepositioned local snapshot, never the network."""

    _, manifest = _load_object(job_manifest_path, description="TeachObs ASR job manifest")
    validation = validate_teachobs_asr_job_manifest(manifest)
    contract = manifest["runtime_contract"]
    if not _CONTAINER_RE.fullmatch(container_image_digest):
        raise ValueError("container image digest must be sha256:<64 lowercase hex>")
    if container_image_digest != contract["container_image_digest"]:
        raise ValueError("GPU container image differs from job manifest contract")
    model_hash = model_directory_sha256(model_directory)
    if model_hash != manifest["model"]["model_snapshot_sha256_v1"]:
        raise ValueError("local ASR model snapshot hash differs from job manifest")
    runtime = _gpu_runtime(container_image_digest, runner_source_sha256)
    if (
        runtime["package_versions"]["faster-whisper"]
        != contract["faster_whisper_version"]
        or runtime["package_versions"]["ctranslate2"]
        != contract["ctranslate2_version"]
    ):
        raise RuntimeError("installed GPU ASR packages differ from manifest contract")
    selected = None if lesson_ids is None else {_safe_lesson_id(value) for value in lesson_ids}
    if selected is not None and not selected:
        raise ValueError("GPU ASR lesson selection may not be empty")
    jobs = [
        job
        for job in manifest["jobs"]
        if selected is None or job["lesson_id"] in selected
    ]
    if selected is not None and selected != {job["lesson_id"] for job in jobs}:
        raise ValueError("GPU ASR lesson selection is absent from the job manifest")
    # Import only after all local/hash/runtime gates have passed.  A local path
    # plus offline environment variables prevents implicit model retrieval.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("GPU ASR runner requires the faster-whisper extra") from exc
    decoding = manifest["decoding_config"]
    model = WhisperModel(
        os.fspath(Path(model_directory).resolve()),
        device="cuda",
        compute_type=decoding["compute_type"],
        local_files_only=True,
    )
    media_root_candidate = Path(media_root)
    if media_root_candidate.is_symlink() or not media_root_candidate.is_dir():
        raise ValueError("GPU private media root is missing or unsafe")
    media_root_path = media_root_candidate.resolve()
    output_candidate = Path(output_directory)
    if output_candidate.is_symlink():
        raise ValueError("GPU ASR output directory may not be a symlink")
    output = ensure_private_directory(output_candidate).resolve()
    completed = 0
    for job in jobs:
        binding = job["media_binding"]
        media = media_root_path.joinpath(
            *PurePosixPath(binding["media_relative_path"]).parts
        )
        if media.is_symlink() or not media.is_file() or file_sha256(media) != binding["media_sha256"]:
            raise ValueError(f"GPU media hash binding failed for {job['lesson_id']}")
        actual_duration = duration_probe(media)
        if abs(actual_duration - float(binding["media_duration_seconds"])) > float(
            binding["duration_tolerance_seconds"]
        ):
            raise ValueError(f"GPU media duration binding failed for {job['lesson_id']}")
        requested = job["language"]["requested"]
        segment_iterator, info = model.transcribe(
            os.fspath(media),
            language=None if requested == "auto" else requested,
            task="transcribe",
            beam_size=decoding["beam_size"],
            best_of=decoding["best_of"],
            temperature=decoding["temperature"],
            condition_on_previous_text=decoding["condition_on_previous_text"],
            word_timestamps=decoding["word_timestamps"],
            vad_filter=decoding["vad_filter"],
            vad_parameters=decoding["vad_parameters"],
        )
        source_segments = list(segment_iterator)
        segments, segment_postprocessing = _postprocess_whisper_segments(
            source_segments,
            maximum_segment_duration=float(
                decoding["maximum_segment_duration_seconds"]
            ),
        )
        coverage = _timeline_coverage(
            segments,
            duration=float(binding["media_duration_seconds"]),
            maximum_segment_duration=float(
                decoding["maximum_segment_duration_seconds"]
            ),
            policy=job["coverage_policy"],
        )
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "job_manifest_sha256": validation["manifest_sha256"],
            "job_sha256": job["job_sha256"],
            "job_id": job["job_id"],
            "lesson_id": job["lesson_id"],
            "transcript_kind": "automatic_speech_recognition",
            "source_tier": "audited_asr_fallback_candidate",
            "official_caption": False,
            "human_content_review_completed": False,
            "content_accuracy_established": False,
            "word_error_rate_established": False,
            "media_binding": binding,
            "model": job["model"],
            "model_runtime": {
                "model_snapshot_sha256_v1": model_hash,
                "local_files_only": True,
                "network_access_enabled": False,
            },
            "decoding_config": decoding,
            "segment_postprocessing": segment_postprocessing,
            "language": {
                "requested": requested,
                "detected": str(info.language),
                "detected_probability": round(float(info.language_probability), 6),
            },
            "runtime": runtime,
            "segments": segments,
            "timeline_coverage": coverage,
        }
        result["result_sha256"] = _canonical_sha256(result)
        target = write_json(output / job["expected_result_relative_path"], result)
        ensure_private_file(target)
        completed += 1
    return {
        "job_manifest_sha256": validation["manifest_sha256"],
        "selected_job_count": len(jobs),
        "completed_result_count": completed,
        "official_caption_claimed": False,
        "model_or_media_download_performed": False,
    }


__all__ = [
    "COVERAGE_POLICY_NAME",
    "COVERAGE_MATRIX_SCHEMA",
    "IMPORT_AUDIT_SCHEMA",
    "JOB_MANIFEST_SCHEMA",
    "PUBLIC_RECEIPT_SCHEMA",
    "RESULT_SCHEMA",
    "RUNNER_NAME",
    "build_pending_teachobs_asr_receipt",
    "build_public_teachobs_asr_receipt",
    "build_teachobs_asr_job_manifest",
    "build_teachobs_transcript_coverage_matrix",
    "import_teachobs_asr_results",
    "model_directory_sha256",
    "run_teachobs_asr_gpu_jobs",
    "validate_teachobs_asr_job_manifest",
    "validate_teachobs_asr_result",
    "write_teachobs_asr_import",
]
