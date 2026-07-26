"""Materialize audited TeachObs transcript sources on official scene windows.

This module is intentionally label-blind.  It consumes the already-audited
private media plan, platform-caption audit, and (when needed) offline ASR
handoff artifacts, but it never opens the released per-scene transcripts or
Track A gold files.  Platform captions are preferred over ASR and empty
materialized scenes remain empty.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Sequence
import unicodedata

from .full_video_dataset import file_sha256
from .io_utils import (
    ensure_private_directory,
    read_json,
    write_json,
    write_text,
)
from .preprocess import parse_srt_or_vtt
from .teachobs_asr_handoff import (
    COVERAGE_MATRIX_SCHEMA,
    IMPORT_AUDIT_SCHEMA,
    JOB_MANIFEST_SCHEMA,
    RESULT_SCHEMA,
    build_teachobs_transcript_coverage_matrix,
    validate_teachobs_asr_job_manifest,
    validate_teachobs_asr_result,
)
from .teachobs_captions import PRIVATE_AUDIT_SCHEMA as CAPTION_AUDIT_SCHEMA
from .teachobs_media import (
    DATASET_ID,
    MEDIA_SCHEMA,
    PINNED_REPOSITORY_COMMIT,
    PLAN_SCHEMA,
)


MANIFEST_SCHEMA = (
    "teaching_skill_miner.teachobs_transcript_materialization_manifest.v2"
)
LESSON_JSONL_SCHEMA = "teaching_skill_miner.teachobs_materialized_lesson_scene.v1"
PUBLIC_RECEIPT_SCHEMA = (
    "teaching_skill_miner.teachobs_transcript_materialization_receipt.v2"
)
PLATFORM_CUE_TIMELINE_INTERSECTION_POLICY = (
    "positive_intersection_with_hash_bound_selected_scene_timeline_v1"
)
ASR_TIMELINE_INTERSECTION_POLICY = (
    "explicit_target_window_projection_from_hash_bound_media_timeline_v1"
)
PAPER_TRACK1_PROFILE = "paper_track1_23_train_6_test"
PAPER_TRACK1_TEST_LESSON_IDS = ("S2", "S5", "S19", "S24", "S28", "S30")
EXCLUDED_OFFICIAL_TEST_LESSON_IDS = ("S4",)
OFFICIAL_TEST_LESSON_IDS = (
    "S2",
    "S4",
    "S5",
    "S19",
    "S24",
    "S28",
    "S30",
)
PAPER_TRACK1_TRAIN_LESSON_IDS = tuple(
    f"S{index}"
    for index in range(1, 31)
    if f"S{index}" not in OFFICIAL_TEST_LESSON_IDS
)
PAPER_TRACK1_EXPECTED_TRAIN_SCENE_COUNT = 3_846
PAPER_TRACK1_EXPECTED_TEST_SCENE_COUNT = 1_099
PAPER_TRACK1_EXPECTED_SCENE_COUNT = 4_945
PINNED_RELEASE_SCENE_COUNT = 5_158
MINIMUM_PLATFORM_TIMELINE_SPAN_FRACTION = 0.90

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LESSON_RE = re.compile(r"^S(?:[1-9]|[12][0-9]|30)$")
_LANGUAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SOURCE_TIERS = (
    "platform_creator_provided_caption",
    "platform_automatic_caption",
    "audited_asr_fallback",
)
_PLATFORM_TRACK_TYPES = (
    "manual_creator_provided",
    "youtube_automatic_caption",
)
_MANIFEST_FIELDS = {
    "schema",
    "dataset_id",
    "profile_id",
    "generated_at_utc",
    "private_artifact",
    "public_release_authorized",
    "repository_binding",
    "input_hashes",
    "selection",
    "policy",
    "aggregate",
    "lessons",
    "ordered_sample_id_sha256",
    "ordered_scene_text_sha256",
    "claims",
    "materialization_fingerprint_sha256",
    "manifest_sha256",
}
_INPUT_HASH_FIELDS = {
    "media_plan_file_sha256",
    "media_plan_canonical_sha256",
    "media_manifest_file_sha256",
    "caption_audit_file_sha256",
    "caption_audit_canonical_sha256",
    "asr_import_audit_file_sha256",
    "asr_import_audit_canonical_sha256",
    "coverage_matrix_file_sha256",
    "coverage_matrix_canonical_sha256",
    "asr_job_manifest_file_sha256",
    "asr_job_manifest_canonical_sha256",
    "asr_result_set_sha256",
}
_LESSON_FIELDS = {
    "lesson_id",
    "split",
    "scene_count",
    "source_tier",
    "source_track_type",
    "source_language",
    "source_sha256",
    "timeline_span_fraction",
    "official_scene_manifest_sha256",
    "transcript_relative_path",
    "transcript_file_sha256",
    "ordered_sample_id_sha256",
    "ordered_scene_text_sha256",
    "source_timeline_adjustment",
    "labels_read_or_used",
}
_SCENE_FIELDS = {
    "schema",
    "sample_id",
    "lesson_id",
    "scene_no",
    "start_seconds",
    "end_seconds",
    "text",
    "text_sha256",
    "source_tier",
    "source_language",
    "source_sha256",
    "positive_overlap_item_count",
    "empty_transcript",
    "labels_read_or_used",
}
_TIMELINE_ADJUSTMENT_FIELDS = {
    "policy",
    "selected_timeline_start_seconds",
    "selected_timeline_end_seconds",
    "source_item_count",
    "retained_item_count",
    "endpoint_clamped_item_count",
    "total_endpoint_clipped_seconds",
    "maximum_endpoint_clipped_seconds",
    "all_source_items_have_positive_timeline_intersection",
    "source_text_items_silently_dropped",
    "text_or_labels_used_for_timing_adjustment",
}
_ASR_TIMELINE_ADJUSTMENT_FIELDS = _TIMELINE_ADJUSTMENT_FIELDS | {
    "hash_bound_media_end_seconds",
    "outside_selected_timeline_item_count",
    "outside_selected_timeline_item_duration_seconds",
    "all_source_items_within_hash_bound_media",
}


class TeachObsTranscriptMaterializationError(ValueError):
    """Raised when a materialization input or output is not fully bound."""


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


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip()
    if not _SHA256_RE.fullmatch(digest):
        raise TeachObsTranscriptMaterializationError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return digest


def _safe_lesson_id(value: Any) -> str:
    lesson_id = str(value or "").strip()
    if not _LESSON_RE.fullmatch(lesson_id):
        raise TeachObsTranscriptMaterializationError(
            f"unsafe TeachObs lesson id: {value!r}"
        )
    return lesson_id


def _safe_relative_path(
    value: Any,
    *,
    suffix: str | None = None,
) -> PurePosixPath:
    text = str(value or "").strip()
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or "\x00" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise TeachObsTranscriptMaterializationError(
            f"unsafe private relative path: {value!r}"
        )
    if suffix is not None and path.suffix.casefold() != suffix.casefold():
        raise TeachObsTranscriptMaterializationError(
            f"private relative path must end in {suffix}"
        )
    return path


def _safe_input_file(
    path: str | Path,
    *,
    description: str,
) -> Path:
    candidate = Path(path).absolute()
    if _has_symlink_component(candidate) or not candidate.is_file():
        raise TeachObsTranscriptMaterializationError(
            f"{description} is missing or unsafe"
        )
    return candidate.resolve()


def _safe_input_directory(
    path: str | Path,
    *,
    description: str,
) -> Path:
    candidate = Path(path).absolute()
    if _has_symlink_component(candidate) or not candidate.is_dir():
        raise TeachObsTranscriptMaterializationError(
            f"{description} is missing or unsafe"
        )
    return candidate.resolve()


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    components = [*reversed(absolute.parents), absolute]
    return any(component.is_symlink() for component in components)


def _load_json_object(
    path: str | Path,
    *,
    description: str,
) -> tuple[Path, dict[str, Any]]:
    source = _safe_input_file(path, description=description)
    try:
        value = read_json(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeachObsTranscriptMaterializationError(
            f"{description} is not valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise TeachObsTranscriptMaterializationError(
            f"{description} must be a JSON object"
        )
    return source, value


def _verify_canonical_digest(
    value: dict[str, Any],
    *,
    field: str,
    description: str,
) -> str:
    claimed = _require_sha256(value.get(field), field=field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if _canonical_sha256(unsigned) != claimed:
        raise TeachObsTranscriptMaterializationError(
            f"{description} canonical hash mismatch"
        )
    return claimed


def _bound_file(root: Path, relative: Any, *, suffix: str) -> Path:
    safe = _safe_relative_path(relative, suffix=suffix)
    current = root
    for part in safe.parts:
        current = current / part
        if current.is_symlink():
            raise TeachObsTranscriptMaterializationError(
                "private source path contains a symlink"
            )
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise TeachObsTranscriptMaterializationError(
            "private source path is missing or escapes its root"
        ) from exc
    if not resolved.is_file():
        raise TeachObsTranscriptMaterializationError(
            "private source path is not a regular file"
        )
    return resolved


def _validate_timestamp(value: Any) -> str:
    timestamp = str(value or "")
    if not _UTC_RE.fullmatch(timestamp):
        raise TeachObsTranscriptMaterializationError(
            "materialization timestamp must be UTC with whole seconds"
        )
    return timestamp


def _load_media_plan(
    media_plan_path: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    source, plan = _load_json_object(
        media_plan_path, description="TeachObs private media plan"
    )
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("dataset_id") != DATASET_ID
        or plan.get("private_artifact") is not True
        or plan.get("public_release_authorized") is not False
    ):
        raise TeachObsTranscriptMaterializationError(
            "unsupported TeachObs private media plan"
        )
    _verify_canonical_digest(
        plan,
        field="plan_sha256",
        description="TeachObs private media plan",
    )
    provenance = plan.get("repository_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("repository_commit") != PINNED_REPOSITORY_COMMIT
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs media plan is not bound to the pinned repository commit"
        )
    _require_sha256(
        provenance.get("repository_tree_sha256"),
        field="repository_tree_sha256",
    )
    _require_sha256(plan.get("lessons_csv_sha256"), field="lessons_csv_sha256")
    lessons = plan.get("lessons")
    if (
        not isinstance(lessons, list)
        or len(lessons) != 30
        or plan.get("lesson_count") != 30
        or plan.get("scene_count") != PINNED_RELEASE_SCENE_COUNT
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs media plan must contain the pinned 30-lesson release"
        )
    by_id: dict[str, dict[str, Any]] = {}
    total_scenes = 0
    for lesson in lessons:
        if not isinstance(lesson, dict):
            raise TeachObsTranscriptMaterializationError(
                "TeachObs media plan has a non-object lesson"
            )
        lesson_id = _safe_lesson_id(lesson.get("lesson_id"))
        if lesson_id in by_id:
            raise TeachObsTranscriptMaterializationError(
                "TeachObs media plan has duplicate lessons"
            )
        expected_split = (
            "test" if lesson_id in OFFICIAL_TEST_LESSON_IDS else "train"
        )
        scenes = lesson.get("scenes")
        if (
            lesson.get("split") != expected_split
            or not isinstance(scenes, list)
            or not scenes
            or lesson.get("scene_count") != len(scenes)
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs media-plan lesson contract is invalid for {lesson_id}"
            )
        _require_sha256(
            lesson.get("scene_manifest_sha256"),
            field="scene_manifest_sha256",
        )
        for expected_scene_no, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict) or scene.get("scene_no") != expected_scene_no:
                raise TeachObsTranscriptMaterializationError(
                    f"TeachObs scenes are not contiguous for {lesson_id}"
                )
            try:
                start = float(scene["start"])
                end = float(scene["end"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise TeachObsTranscriptMaterializationError(
                    f"TeachObs scene timing is invalid for {lesson_id}"
                ) from exc
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or not math.isclose(start, (expected_scene_no - 1) * 15.0)
                or not math.isclose(end, expected_scene_no * 15.0)
            ):
                raise TeachObsTranscriptMaterializationError(
                    f"TeachObs scenes are not fixed [start,end) windows for {lesson_id}"
                )
        total_scenes += len(scenes)
        by_id[lesson_id] = lesson
    if set(by_id) != {f"S{index}" for index in range(1, 31)}:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs media plan does not cover exactly S1-S30"
        )
    if total_scenes != PINNED_RELEASE_SCENE_COUNT:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs media-plan scene count differs from the pinned release"
        )
    return source, plan, by_id


def _selected_lessons(
    plan_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    ordered_ids = [
        *PAPER_TRACK1_TRAIN_LESSON_IDS,
        *PAPER_TRACK1_TEST_LESSON_IDS,
    ]
    selected = [plan_by_id[lesson_id] for lesson_id in ordered_ids]
    train_count = sum(
        len(plan_by_id[lesson_id]["scenes"])
        for lesson_id in PAPER_TRACK1_TRAIN_LESSON_IDS
    )
    test_count = sum(
        len(plan_by_id[lesson_id]["scenes"])
        for lesson_id in PAPER_TRACK1_TEST_LESSON_IDS
    )
    if (
        train_count != PAPER_TRACK1_EXPECTED_TRAIN_SCENE_COUNT
        or test_count != PAPER_TRACK1_EXPECTED_TEST_SCENE_COUNT
        or train_count + test_count != PAPER_TRACK1_EXPECTED_SCENE_COUNT
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs scenes differ from paper_track1_23_train_6_test"
        )
    sample_ids = [
        f"{lesson['lesson_id']}:{scene['scene_no']}"
        for lesson in selected
        for scene in lesson["scenes"]
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs selected profile has duplicate sample ids"
        )
    return selected, sample_ids


def _load_media_manifest(
    media_manifest_path: str | Path,
    *,
    plan: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    source, manifest = _load_json_object(
        media_manifest_path, description="TeachObs private media manifest"
    )
    provenance = plan["repository_provenance"]
    if (
        manifest.get("schema") != MEDIA_SCHEMA
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("private_artifact") is not True
        or manifest.get("public_release_authorized") is not False
        or manifest.get("selected_lesson_count") != 30
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or manifest.get("repository_commit") != PINNED_REPOSITORY_COMMIT
        or manifest.get("repository_tree_sha256")
        != provenance["repository_tree_sha256"]
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs media manifest is not bound to the selected media plan"
        )
    lessons = manifest.get("lessons")
    if not isinstance(lessons, list) or len(lessons) > 30:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs media manifest has an invalid lesson list"
        )
    seen: set[str] = set()
    for item in lessons:
        if not isinstance(item, dict):
            raise TeachObsTranscriptMaterializationError(
                "TeachObs media manifest has a non-object lesson"
            )
        lesson_id = _safe_lesson_id(item.get("lesson_id"))
        if lesson_id in seen:
            raise TeachObsTranscriptMaterializationError(
                "TeachObs media manifest has duplicate lessons"
            )
        seen.add(lesson_id)
    return source, manifest


def _load_caption_audit(
    caption_audit_path: str | Path,
    *,
    plan: dict[str, Any],
    plan_by_id: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    source, audit = _load_json_object(
        caption_audit_path, description="TeachObs private caption audit"
    )
    if (
        audit.get("schema") != CAPTION_AUDIT_SCHEMA
        or audit.get("dataset_id") != DATASET_ID
        or audit.get("private_artifact") is not True
        or audit.get("public_release_authorized") is not False
        or audit.get("repository_commit") != PINNED_REPOSITORY_COMMIT
        or audit.get("media_plan_sha256") != plan["plan_sha256"]
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs caption audit is not bound to the selected media plan"
        )
    canonical = _verify_canonical_digest(
        audit,
        field="audit_canonical_sha256",
        description="TeachObs private caption audit",
    )
    if not canonical:
        raise AssertionError("caption audit digest cannot be empty")
    claims = audit.get("claims")
    if (
        not isinstance(claims, dict)
        or claims.get("caption_content_accuracy_established") is not False
        or claims.get("word_error_rate_established") is not False
        or claims.get("independent_human_transcript_audit_completed") is not False
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs caption audit overstates transcript evidence"
        )
    records = audit.get("records")
    if not isinstance(records, list) or len(records) != 30:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs caption audit must contain all 30 lessons"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TeachObsTranscriptMaterializationError(
                "TeachObs caption audit has a non-object record"
            )
        lesson_id = _safe_lesson_id(record.get("lesson_id"))
        if lesson_id in by_id:
            raise TeachObsTranscriptMaterializationError(
                "TeachObs caption audit has duplicate lessons"
            )
        plan_lesson = plan_by_id[lesson_id]
        tracks = record.get("tracks")
        released = record.get("released_transcript_audit")
        if (
            record.get("split") != plan_lesson["split"]
            or record.get("scene_manifest_sha256")
            != plan_lesson["scene_manifest_sha256"]
            or not isinstance(tracks, list)
            or record.get("track_count") != len(tracks)
            or not isinstance(released, dict)
            or released.get("expected_file_count") != plan_lesson["scene_count"]
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs caption record is not scene-bound for {lesson_id}"
            )
        by_id[lesson_id] = record
    if set(by_id) != set(plan_by_id):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs caption audit lesson set differs from the media plan"
        )
    return source, audit, by_id


def _selected_timeline_bounds(
    plan_lesson: dict[str, Any],
) -> tuple[float, float]:
    scenes = plan_lesson.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs selected lesson has no hash-bound scene timeline"
        )
    try:
        start = float(scenes[0]["start"])
        end = float(scenes[-1]["end"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs selected scene timeline boundary is invalid"
        ) from exc
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end <= start
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs selected scene timeline boundary is invalid"
        )
    return start, end


def _intersect_platform_cues_with_selected_timeline(
    segments: Sequence[dict[str, Any]],
    *,
    plan_lesson: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Clip only timestamps to the selected scene timeline, never cue text."""

    timeline_start, timeline_end = _selected_timeline_bounds(plan_lesson)
    adjusted: list[dict[str, Any]] = []
    clipped_durations: list[float] = []
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        clipped_start = max(start, timeline_start)
        clipped_end = min(end, timeline_end)
        if clipped_end <= clipped_start:
            raise TeachObsTranscriptMaterializationError(
                "TeachObs platform cue has no positive overlap with the "
                f"hash-bound selected timeline at index {index}"
            )
        removed = max(0.0, end - clipped_end) + max(
            0.0, clipped_start - start
        )
        if removed > 0:
            clipped_durations.append(removed)
        adjusted.append(
            {
                "start": clipped_start,
                "end": clipped_end,
                "text": segment["text"],
                "_order": segment["_order"],
            }
        )
    adjustment = {
        "policy": PLATFORM_CUE_TIMELINE_INTERSECTION_POLICY,
        "selected_timeline_start_seconds": round(timeline_start, 6),
        "selected_timeline_end_seconds": round(timeline_end, 6),
        "source_item_count": len(segments),
        "retained_item_count": len(adjusted),
        "endpoint_clamped_item_count": len(clipped_durations),
        "total_endpoint_clipped_seconds": round(sum(clipped_durations), 6),
        "maximum_endpoint_clipped_seconds": round(
            max(clipped_durations, default=0.0), 6
        ),
        "all_source_items_have_positive_timeline_intersection": True,
        "source_text_items_silently_dropped": 0,
        "text_or_labels_used_for_timing_adjustment": False,
    }
    return adjusted, adjustment


def _platform_segments(
    caption_root: Path,
    *,
    record: dict[str, Any],
    track: dict[str, Any],
    plan_lesson: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    lesson_id = record["lesson_id"]
    relative = _safe_relative_path(
        track.get("private_caption_relative_path"), suffix=".vtt"
    )
    if (
        len(relative.parts) != 3
        or relative.parts[:2] != ("raw", lesson_id)
    ):
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs caption path is not lesson-bound for {lesson_id}"
        )
    source = _bound_file(caption_root, relative, suffix=".vtt")
    expected_digest = _require_sha256(
        track.get("caption_sha256"), field="caption_sha256"
    )
    if file_sha256(source) != expected_digest:
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs caption hash mismatch for {lesson_id}"
        )
    size = track.get("caption_size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs caption size is invalid for {lesson_id}"
        )
    if source.stat().st_size != size:
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs caption size mismatch for {lesson_id}"
        )
    try:
        text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs caption is not readable UTF-8 for {lesson_id}"
        ) from exc
    segments = parse_srt_or_vtt(text)
    if not segments:
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs caption has no timed cues for {lesson_id}"
        )
    validated: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs caption has invalid timing for {lesson_id}"
            ) from exc
        value = str(segment.get("text") or "").strip()
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or not value
            or "\x00" in value
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs caption has an invalid cue for {lesson_id}"
            )
        validated.append(
            {"start": start, "end": end, "text": value, "_order": index}
        )
    validated.sort(
        key=lambda item: (item["start"], item["end"], item["_order"])
    )
    timeline = track.get("timeline")
    if not isinstance(timeline, dict) or timeline.get("timeline_audit_completed") is not True:
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs caption timeline audit is missing for {lesson_id}"
        )
    duration = float(plan_lesson["reference_duration_seconds"])
    first = min(item["start"] for item in validated)
    last = max(item["end"] for item in validated)
    coverage = round(min(1.0, max(0.0, last - first) / duration), 6)
    expected_timeline = {
        "cue_count": len(validated),
        "first_cue_start_seconds": round(first, 3),
        "last_cue_end_seconds": round(last, 3),
        "timeline_span_seconds": round(max(0.0, last - first), 3),
        "timeline_span_coverage_fraction": coverage,
    }
    if any(timeline.get(key) != value for key, value in expected_timeline.items()):
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs caption timeline evidence changed for {lesson_id}"
        )
    adjusted, adjustment = _intersect_platform_cues_with_selected_timeline(
        validated,
        plan_lesson=plan_lesson,
    )
    return source, adjusted, adjustment


def _platform_source_candidates(
    caption_root: Path,
    *,
    record: dict[str, Any],
    plan_lesson: dict[str, Any],
) -> list[dict[str, Any]]:
    if record.get("status") != "caption_timeline_audited":
        if record.get("tracks"):
            raise TeachObsTranscriptMaterializationError(
                f"unaudited caption record contains tracks for {record['lesson_id']}"
            )
        return []
    candidates: list[dict[str, Any]] = []
    for track in record["tracks"]:
        if not isinstance(track, dict):
            raise TeachObsTranscriptMaterializationError(
                "TeachObs caption record contains a non-object track"
            )
        track_type = str(track.get("track_type") or "")
        language = str(track.get("language") or "")
        roles = track.get("roles")
        timeline = track.get("timeline")
        if (
            track_type not in _PLATFORM_TRACK_TYPES
            or not _LANGUAGE_RE.fullmatch(language)
            or not isinstance(roles, list)
            or any(
                not isinstance(role, str)
                or role not in {"original_language", "english"}
                for role in roles
            )
            or len(roles) != len(set(roles))
            or not isinstance(timeline, dict)
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs caption track metadata is invalid for {record['lesson_id']}"
            )
        caption_path, segments, timeline_adjustment = _platform_segments(
            caption_root,
            record=record,
            track=track,
            plan_lesson=plan_lesson,
        )
        coverage = float(timeline["timeline_span_coverage_fraction"])
        if coverage < MINIMUM_PLATFORM_TIMELINE_SPAN_FRACTION:
            continue
        tier = (
            "platform_creator_provided_caption"
            if track_type == "manual_creator_provided"
            else "platform_automatic_caption"
        )
        candidates.append(
            {
                "source_tier": tier,
                "source_track_type": track_type,
                "source_language": language,
                "original_language": "original_language" in roles,
                "source_sha256": track["caption_sha256"],
                "source_file_path": caption_path,
                "segments": segments,
                "timeline_span_fraction": coverage,
                "source_timeline_adjustment": timeline_adjustment,
            }
        )
    candidates.sort(
        key=lambda item: (
            _PLATFORM_TRACK_TYPES.index(item["source_track_type"]),
            not item["original_language"],
            item["source_language"],
            item["source_sha256"],
        )
    )
    return candidates


def _project_audited_asr_segments_to_selected_timeline(
    result: dict[str, Any],
    *,
    plan_lesson: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project full-media ASR onto the official target windows by time only.

    ASR is bound to the complete media duration, while the released target
    domain is the contiguous official scene timeline.  A segment wholly in a
    media-only tail is therefore explicitly excluded and counted.  A segment
    crossing the target endpoint is clipped.  Neither decision inspects text
    or labels.
    """

    timeline_start, timeline_end = _selected_timeline_bounds(plan_lesson)
    media_binding = result.get("media_binding")
    if not isinstance(media_binding, dict):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs audited ASR source has no hash-bound media duration"
        )
    try:
        media_end = float(media_binding["media_duration_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs audited ASR source has an invalid hash-bound media duration"
        ) from exc
    if (
        not math.isfinite(media_end)
        or media_end <= timeline_start
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs audited ASR media duration is incompatible with the "
            "selected scene timeline"
        )
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs audited ASR source has no segments"
        )
    adjusted: list[dict[str, Any]] = []
    clipped_durations: list[float] = []
    outside_durations: list[float] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs audited ASR source has a non-object segment at index {index}"
            )
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs audited ASR segment timing is invalid at index {index}"
            ) from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < timeline_start
            or end <= start
            or end > media_end
        ):
            raise TeachObsTranscriptMaterializationError(
                "TeachObs audited ASR segment is not within the hash-bound "
                f"media timeline at index {index}"
            )
        clipped_start = max(start, timeline_start)
        clipped_end = min(end, timeline_end)
        if clipped_end <= clipped_start:
            outside_durations.append(end - start)
            continue
        removed = max(0.0, end - clipped_end) + max(
            0.0, clipped_start - start
        )
        if removed > 0:
            clipped_durations.append(removed)
        adjusted.append(
            {
                "start": clipped_start,
                "end": clipped_end,
                "text": segment["text"],
            }
        )
    if not adjusted:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs audited ASR source has no segment in the selected target "
            "timeline"
        )
    adjustment = {
        "policy": ASR_TIMELINE_INTERSECTION_POLICY,
        "selected_timeline_start_seconds": round(timeline_start, 6),
        "selected_timeline_end_seconds": round(timeline_end, 6),
        "hash_bound_media_end_seconds": round(media_end, 6),
        "source_item_count": len(segments),
        "retained_item_count": len(adjusted),
        "outside_selected_timeline_item_count": len(outside_durations),
        "outside_selected_timeline_item_duration_seconds": round(
            sum(outside_durations), 6
        ),
        "endpoint_clamped_item_count": len(clipped_durations),
        "total_endpoint_clipped_seconds": round(sum(clipped_durations), 6),
        "maximum_endpoint_clipped_seconds": round(
            max(clipped_durations, default=0.0), 6
        ),
        "all_source_items_have_positive_timeline_intersection": (
            not outside_durations
        ),
        "all_source_items_within_hash_bound_media": True,
        "source_text_items_silently_dropped": 0,
        "text_or_labels_used_for_timing_adjustment": False,
    }
    return adjusted, adjustment


def _validate_asr_chain(
    *,
    media_manifest_path: Path,
    caption_audit_path: Path,
    caption_audit: dict[str, Any],
    asr_import_audit_path: str | Path | None,
    coverage_matrix_path: str | Path | None,
    asr_job_manifest_path: str | Path | None,
    asr_results_directory: str | Path | None,
) -> dict[str, Any]:
    supplied = (
        asr_import_audit_path,
        coverage_matrix_path,
        asr_job_manifest_path,
        asr_results_directory,
    )
    if not any(value is not None for value in supplied):
        matrix = build_teachobs_transcript_coverage_matrix(
            caption_audit,
            None,
            generated_at_utc=_utc_now(),
        )
        return {
            "import_path": None,
            "import_audit": None,
            "coverage_path": None,
            "coverage_matrix": matrix,
            "job_path": None,
            "job_manifest": None,
            "result_by_id": {},
            "result_file_by_id": {},
            "result_set_sha256": _canonical_sha256([]),
        }
    if any(value is None for value in supplied):
        raise TeachObsTranscriptMaterializationError(
            "ASR materialization requires import audit, coverage matrix, job "
            "manifest, and result directory together"
        )
    assert asr_import_audit_path is not None
    assert coverage_matrix_path is not None
    assert asr_job_manifest_path is not None
    assert asr_results_directory is not None
    job_path, manifest = _load_json_object(
        asr_job_manifest_path, description="TeachObs ASR job manifest"
    )
    if (
        manifest.get("schema") != JOB_MANIFEST_SCHEMA
        or manifest.get("source_media_manifest_file_sha256")
        != file_sha256(media_manifest_path)
        or manifest.get("source_caption_audit_file_sha256")
        != file_sha256(caption_audit_path)
        or manifest.get("repository_commit") != PINNED_REPOSITORY_COMMIT
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs ASR job manifest source binding mismatch"
        )
    try:
        job_validation = validate_teachobs_asr_job_manifest(manifest)
    except ValueError as exc:
        raise TeachObsTranscriptMaterializationError(str(exc)) from exc

    import_path, import_audit = _load_json_object(
        asr_import_audit_path, description="TeachObs ASR import audit"
    )
    if (
        import_audit.get("schema") != IMPORT_AUDIT_SCHEMA
        or import_audit.get("dataset_id") != DATASET_ID
        or import_audit.get("private_artifact") is not True
        or import_audit.get("public_release_authorized") is not False
        or import_audit.get("job_manifest_file_sha256") != file_sha256(job_path)
        or import_audit.get("job_manifest_canonical_sha256")
        != job_validation["manifest_sha256"]
        or import_audit.get("source_media_manifest_file_sha256")
        != file_sha256(media_manifest_path)
        or import_audit.get("source_caption_audit_file_sha256")
        != file_sha256(caption_audit_path)
        or import_audit.get("model_contract_sha256")
        != _canonical_sha256(manifest["model"])
        or import_audit.get("decoding_contract_sha256")
        != _canonical_sha256(manifest["decoding_config"])
        or import_audit.get("runtime_contract_sha256")
        != _canonical_sha256(manifest["runtime_contract"])
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs ASR import audit source or contract binding mismatch"
        )
    _verify_canonical_digest(
        import_audit,
        field="audit_sha256",
        description="TeachObs ASR import audit",
    )
    claims = import_audit.get("claims")
    if (
        not isinstance(claims, dict)
        or claims.get("asr_is_official_caption") is not False
        or claims.get("independent_human_content_audit_completed") is not False
        or claims.get("content_accuracy_established") is not False
        or claims.get("word_error_rate_established") is not False
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs ASR import audit overstates transcript evidence"
        )

    import_records = import_audit.get("records")
    jobs = manifest.get("jobs")
    if not isinstance(import_records, list) or not isinstance(jobs, list):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs ASR import/job records are malformed"
        )
    import_by_id: dict[str, dict[str, Any]] = {}
    for record in import_records:
        if not isinstance(record, dict):
            raise TeachObsTranscriptMaterializationError(
                "TeachObs ASR import audit has a non-object record"
            )
        lesson_id = _safe_lesson_id(record.get("lesson_id"))
        if lesson_id in import_by_id:
            raise TeachObsTranscriptMaterializationError(
                "TeachObs ASR import audit has duplicate lessons"
            )
        import_by_id[lesson_id] = record
    job_by_id = {job["lesson_id"]: job for job in jobs}
    if (
        len(job_by_id) != len(jobs)
        or set(import_by_id) != set(job_by_id)
        or import_audit.get("aggregate", {}).get("job_count") != len(jobs)
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs ASR import audit does not cover exactly the job set"
        )

    results_root = _safe_input_directory(
        asr_results_directory, description="TeachObs ASR results directory"
    )
    expected_relative_paths = {
        str(job["expected_result_relative_path"]) for job in jobs
    }
    actual_relative_paths: set[str] = set()
    for item in results_root.rglob("*"):
        if item.is_symlink():
            raise TeachObsTranscriptMaterializationError(
                "TeachObs ASR results contain a symlink"
            )
        if item.is_file():
            actual_relative_paths.add(item.relative_to(results_root).as_posix())
    if actual_relative_paths != expected_relative_paths:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs ASR result files differ from the bound job set"
        )

    result_by_id: dict[str, dict[str, Any]] = {}
    result_file_by_id: dict[str, Path] = {}
    result_set: list[dict[str, str]] = []
    for job in jobs:
        lesson_id = job["lesson_id"]
        record = import_by_id[lesson_id]
        if record.get("status") != "asr_provenance_and_timeline_audited":
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs ASR result is not fully audited for {lesson_id}"
            )
        result_path = _bound_file(
            results_root,
            job["expected_result_relative_path"],
            suffix=".json",
        )
        result = read_json(result_path)
        if not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA:
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs ASR result is malformed for {lesson_id}"
            )
        try:
            coverage = validate_teachobs_asr_result(
                result,
                manifest_sha256=job_validation["manifest_sha256"],
                job=job,
            )
        except ValueError as exc:
            raise TeachObsTranscriptMaterializationError(str(exc)) from exc
        result_file_digest = file_sha256(result_path)
        if (
            record.get("job_sha256") != job["job_sha256"]
            or record.get("result_sha256") != result_file_digest
            or record.get("result_canonical_sha256") != result["result_sha256"]
            or record.get("timeline_coverage") != coverage
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs ASR import/result binding mismatch for {lesson_id}"
            )
        result_by_id[lesson_id] = result
        result_file_by_id[lesson_id] = result_path
        result_set.append(
            {
                "lesson_id": lesson_id,
                "result_file_sha256": result_file_digest,
                "result_canonical_sha256": result["result_sha256"],
            }
        )

    aggregate = import_audit.get("aggregate")
    if (
        not isinstance(aggregate, dict)
        or aggregate.get("valid_result_count") != len(jobs)
        or aggregate.get("pending_result_count") != 0
        or aggregate.get("asr_job_set_complete") is not bool(jobs)
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs ASR import aggregate differs from the audited results"
        )

    coverage_path, coverage_matrix = _load_json_object(
        coverage_matrix_path, description="TeachObs transcript coverage matrix"
    )
    if (
        coverage_matrix.get("schema") != COVERAGE_MATRIX_SCHEMA
        or coverage_matrix.get("dataset_id") != DATASET_ID
        or coverage_matrix.get("private_artifact") is not True
        or coverage_matrix.get("public_release_authorized") is not False
    ):
        raise TeachObsTranscriptMaterializationError(
            "unsupported TeachObs transcript coverage matrix"
        )
    _verify_canonical_digest(
        coverage_matrix,
        field="matrix_sha256",
        description="TeachObs transcript coverage matrix",
    )
    regenerated = build_teachobs_transcript_coverage_matrix(
        caption_audit,
        import_audit,
        generated_at_utc=coverage_matrix.get("generated_at_utc"),
    )
    if regenerated != coverage_matrix:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs transcript coverage matrix is not reproducible from its inputs"
        )
    return {
        "import_path": import_path,
        "import_audit": import_audit,
        "coverage_path": coverage_path,
        "coverage_matrix": coverage_matrix,
        "job_path": job_path,
        "job_manifest": manifest,
        "result_by_id": result_by_id,
        "result_file_by_id": result_file_by_id,
        "result_set_sha256": _canonical_sha256(result_set),
    }


def _normalized_token(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _deduplicate_rolling_caption_text(values: Sequence[str]) -> str:
    """Keep only deterministic novel token suffixes from rolling cue snapshots."""

    emitted: list[str] = []
    emitted_normalized: list[str] = []
    seen_snapshots: set[tuple[str, ...]] = set()
    for raw_value in values:
        tokens = str(raw_value).strip().split()
        if not tokens:
            continue
        normalized = tuple(_normalized_token(token) for token in tokens)
        if normalized in seen_snapshots:
            continue
        seen_snapshots.add(normalized)
        maximum = min(len(emitted_normalized), len(normalized))
        overlap = 0
        for size in range(maximum, 0, -1):
            if tuple(emitted_normalized[-size:]) == normalized[:size]:
                overlap = size
                break
        emitted.extend(tokens[overlap:])
        emitted_normalized.extend(normalized[overlap:])
    return " ".join(emitted)


def _scene_text(
    segments: Sequence[dict[str, Any]],
    *,
    start: float,
    end: float,
    rolling_caption_deduplication: bool,
) -> tuple[str, int]:
    overlapping = [
        segment
        for segment in segments
        if float(segment["start"]) < end and float(segment["end"]) > start
    ]
    values = [str(segment["text"]).strip() for segment in overlapping]
    if rolling_caption_deduplication:
        text = _deduplicate_rolling_caption_text(values)
    else:
        text = " ".join(value for value in values if value)
    return text, len(overlapping)


def _manifest_fingerprint_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "schema",
        "dataset_id",
        "profile_id",
        "private_artifact",
        "public_release_authorized",
        "repository_binding",
        "input_hashes",
        "selection",
        "policy",
        "aggregate",
        "lessons",
        "ordered_sample_id_sha256",
        "ordered_scene_text_sha256",
        "claims",
    )
    return {field: manifest[field] for field in fields}


def _claims() -> dict[str, bool]:
    return {
        "source_selection_and_scene_alignment_audited": True,
        "platform_cue_timeline_intersection_provenance_recorded": True,
        "platform_cue_text_or_labels_used_for_timing_adjustment": False,
        "platform_cue_text_silently_dropped_during_timing_adjustment": False,
        "asr_target_window_projection_provenance_recorded": True,
        "asr_outside_target_window_items_explicitly_counted": True,
        "asr_text_or_labels_used_for_timing_adjustment": False,
        "asr_text_silently_dropped_during_timing_adjustment": False,
        "labels_read_or_used": False,
        "released_transcript_fallback_used": False,
        "empty_scenes_filled_from_released_transcript": False,
        "asr_is_official_caption": False,
        "content_accuracy_established": False,
        "word_error_rate_established": False,
        "independent_human_content_audit_completed": False,
        "confirmatory_multimodal_gain_established": False,
        "deployment_accuracy_established": False,
        "learner_effect_established": False,
    }


def _policy() -> dict[str, Any]:
    return {
        "source_priority": [
            "manual_creator_provided",
            "youtube_automatic_caption",
            "audited_asr_fallback",
        ],
        "minimum_platform_timeline_span_fraction": (
            MINIMUM_PLATFORM_TIMELINE_SPAN_FRACTION
        ),
        "same_tier_order": [
            "original_language_role_first",
            "language_ascending",
            "source_sha256_ascending",
        ],
        "scene_interval_semantics": "[start_seconds,end_seconds)",
        "positive_overlap_required": True,
        "boundary_touch_included": False,
        "platform_cue_timeline_intersection_policy": (
            PLATFORM_CUE_TIMELINE_INTERSECTION_POLICY
        ),
        "platform_cue_timeline_intersection_boundary": (
            "hash_bound_selected_official_scene_timeline"
        ),
        "platform_cue_without_positive_timeline_intersection": "reject",
        "platform_cue_text_or_labels_used_for_timing_adjustment": False,
        "platform_cue_text_silently_dropped_during_timing_adjustment": False,
        "asr_target_window_projection_policy": (
            ASR_TIMELINE_INTERSECTION_POLICY
        ),
        "asr_target_window_projection_boundary": (
            "hash_bound_selected_official_scene_timeline"
        ),
        "asr_segment_wholly_outside_target_window": (
            "explicitly_exclude_and_count"
        ),
        "asr_segment_outside_hash_bound_media": "reject",
        "asr_text_or_labels_used_for_timing_adjustment": False,
        "asr_text_silently_dropped_during_timing_adjustment": False,
        "rolling_caption_deduplication": (
            "deterministic_longest_emitted_suffix_to_current_prefix_tokens_v1"
        ),
        "empty_scene_policy": "retain_empty_string",
        "released_transcript_fallback_used": False,
        "labels_read_or_used": False,
    }


def materialize_teachobs_transcripts(
    media_plan_path: str | Path,
    media_manifest_path: str | Path,
    caption_audit_path: str | Path,
    asr_import_audit_path: str | Path | None,
    coverage_matrix_path: str | Path | None,
    asr_job_manifest_path: str | Path | None,
    asr_results_directory: str | Path | None,
    output_directory: str | Path,
    *,
    public_receipt_path: str | Path | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Write the fixed 29-lesson/4,945-scene audited transcript materialization."""

    timestamp = _validate_timestamp(generated_at_utc or _utc_now())
    plan_path, plan, plan_by_id = _load_media_plan(media_plan_path)
    selected_lessons, ordered_sample_ids = _selected_lessons(plan_by_id)
    media_manifest_source, _ = _load_media_manifest(
        media_manifest_path, plan=plan
    )
    caption_path, caption_audit, caption_by_id = _load_caption_audit(
        caption_audit_path,
        plan=plan,
        plan_by_id=plan_by_id,
    )
    asr = _validate_asr_chain(
        media_manifest_path=media_manifest_source,
        caption_audit_path=caption_path,
        caption_audit=caption_audit,
        asr_import_audit_path=asr_import_audit_path,
        coverage_matrix_path=coverage_matrix_path,
        asr_job_manifest_path=asr_job_manifest_path,
        asr_results_directory=asr_results_directory,
    )
    coverage_rows = asr["coverage_matrix"].get("rows")
    if not isinstance(coverage_rows, list) or len(coverage_rows) != 30:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs transcript coverage matrix must contain all 30 lessons"
        )
    coverage_by_id = {
        _safe_lesson_id(row.get("lesson_id")): row
        for row in coverage_rows
        if isinstance(row, dict)
    }
    if len(coverage_by_id) != 30:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs transcript coverage matrix has duplicate or missing lessons"
        )

    caption_root = caption_path.parent
    selected_sources: dict[str, dict[str, Any]] = {}
    for lesson in selected_lessons:
        lesson_id = lesson["lesson_id"]
        candidates = _platform_source_candidates(
            caption_root,
            record=caption_by_id[lesson_id],
            plan_lesson=lesson,
        )
        if candidates:
            source = candidates[0]
        else:
            result = asr["result_by_id"].get(lesson_id)
            result_path = asr["result_file_by_id"].get(lesson_id)
            if not isinstance(result, dict) or not isinstance(result_path, Path):
                raise TeachObsTranscriptMaterializationError(
                    f"no audited transcript source is available for {lesson_id}"
                )
            projected_segments, timeline_adjustment = (
                _project_audited_asr_segments_to_selected_timeline(
                    result,
                    plan_lesson=lesson,
                )
            )
            source = {
                "source_tier": "audited_asr_fallback",
                "source_track_type": "automatic_speech_recognition",
                "source_language": result["language"]["detected"],
                "original_language": False,
                "source_sha256": file_sha256(result_path),
                "source_file_path": result_path,
                "segments": projected_segments,
                "timeline_span_fraction": result["timeline_coverage"][
                    "timeline_span_fraction"
                ],
                "source_timeline_adjustment": timeline_adjustment,
            }
        coverage_row = coverage_by_id[lesson_id]
        if (
            coverage_row.get("selected_source_tier") != source["source_tier"]
            or coverage_row.get("selected_source_sha256")
            != source["source_sha256"]
            or coverage_row.get("content_accuracy_established") is not False
            or coverage_row.get("word_error_rate_established") is not False
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs selected source disagrees with coverage evidence for {lesson_id}"
            )
        selected_sources[lesson_id] = source

    output_candidate = Path(output_directory).absolute()
    if _has_symlink_component(output_candidate):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs materialization output directory may not be a symlink"
        )
    output = ensure_private_directory(output_candidate).resolve()
    lessons_directory = output / "lessons"
    if lessons_directory.is_symlink():
        raise TeachObsTranscriptMaterializationError(
            "TeachObs materialization lessons directory may not be a symlink"
        )
    ensure_private_directory(lessons_directory)
    expected_output_names = {
        f"{lesson['lesson_id']}.jsonl" for lesson in selected_lessons
    }
    for existing in lessons_directory.iterdir():
        if (
            existing.is_symlink()
            or not existing.is_file()
            or existing.name not in expected_output_names
        ):
            raise TeachObsTranscriptMaterializationError(
                "TeachObs materialization output contains an unbound lesson artifact"
            )

    lesson_records: list[dict[str, Any]] = []
    ordered_scene_text_bindings: list[dict[str, str]] = []
    all_rows: list[dict[str, Any]] = []
    for lesson in selected_lessons:
        lesson_id = lesson["lesson_id"]
        source = selected_sources[lesson_id]
        source_tier = source["source_tier"]
        rows: list[dict[str, Any]] = []
        lesson_text_bindings: list[dict[str, str]] = []
        lesson_sample_ids: list[str] = []
        for scene in lesson["scenes"]:
            scene_no = int(scene["scene_no"])
            sample_id = f"{lesson_id}:{scene_no}"
            text, overlap_count = _scene_text(
                source["segments"],
                start=float(scene["start"]),
                end=float(scene["end"]),
                rolling_caption_deduplication=source_tier.startswith("platform_"),
            )
            text_digest = _text_sha256(text)
            row = {
                "schema": LESSON_JSONL_SCHEMA,
                "sample_id": sample_id,
                "lesson_id": lesson_id,
                "scene_no": scene_no,
                "start_seconds": float(scene["start"]),
                "end_seconds": float(scene["end"]),
                "text": text,
                "text_sha256": text_digest,
                "source_tier": source_tier,
                "source_language": source["source_language"],
                "source_sha256": source["source_sha256"],
                "positive_overlap_item_count": overlap_count,
                "empty_transcript": not bool(text),
                "labels_read_or_used": False,
            }
            rows.append(row)
            all_rows.append(row)
            lesson_sample_ids.append(sample_id)
            binding = {"sample_id": sample_id, "text_sha256": text_digest}
            lesson_text_bindings.append(binding)
            ordered_scene_text_bindings.append(binding)
        payload = "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        )
        relative = PurePosixPath("lessons", f"{lesson_id}.jsonl")
        target = write_text(output.joinpath(*relative.parts), payload)
        lesson_records.append(
            {
                "lesson_id": lesson_id,
                "split": lesson["split"],
                "scene_count": len(rows),
                "source_tier": source_tier,
                "source_track_type": source["source_track_type"],
                "source_language": source["source_language"],
                "source_sha256": source["source_sha256"],
                "timeline_span_fraction": source["timeline_span_fraction"],
                "official_scene_manifest_sha256": lesson[
                    "scene_manifest_sha256"
                ],
                "transcript_relative_path": relative.as_posix(),
                "transcript_file_sha256": file_sha256(target),
                "ordered_sample_id_sha256": _canonical_sha256(
                    lesson_sample_ids
                ),
                "ordered_scene_text_sha256": _canonical_sha256(
                    lesson_text_bindings
                ),
                "source_timeline_adjustment": source[
                    "source_timeline_adjustment"
                ],
                "labels_read_or_used": False,
            }
        )

    if [row["sample_id"] for row in all_rows] != ordered_sample_ids:
        raise AssertionError("materialized sample order changed during writing")
    input_hashes = {
        "media_plan_file_sha256": file_sha256(plan_path),
        "media_plan_canonical_sha256": plan["plan_sha256"],
        "media_manifest_file_sha256": file_sha256(media_manifest_source),
        "caption_audit_file_sha256": file_sha256(caption_path),
        "caption_audit_canonical_sha256": caption_audit[
            "audit_canonical_sha256"
        ],
        "asr_import_audit_file_sha256": (
            file_sha256(asr["import_path"])
            if isinstance(asr["import_path"], Path)
            else None
        ),
        "asr_import_audit_canonical_sha256": (
            asr["import_audit"]["audit_sha256"]
            if isinstance(asr["import_audit"], dict)
            else None
        ),
        "coverage_matrix_file_sha256": (
            file_sha256(asr["coverage_path"])
            if isinstance(asr["coverage_path"], Path)
            else None
        ),
        "coverage_matrix_canonical_sha256": (
            asr["coverage_matrix"]["matrix_sha256"]
            if isinstance(asr["coverage_path"], Path)
            else None
        ),
        "asr_job_manifest_file_sha256": (
            file_sha256(asr["job_path"])
            if isinstance(asr["job_path"], Path)
            else None
        ),
        "asr_job_manifest_canonical_sha256": (
            asr["job_manifest"]["manifest_sha256"]
            if isinstance(asr["job_manifest"], dict)
            else None
        ),
        "asr_result_set_sha256": asr["result_set_sha256"],
    }
    ordered_sample_id_sha256 = _canonical_sha256(ordered_sample_ids)
    ordered_scene_text_sha256 = _canonical_sha256(
        ordered_scene_text_bindings
    )
    aggregate = _aggregate_rows(all_rows, lesson_records)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "dataset_id": DATASET_ID,
        "profile_id": PAPER_TRACK1_PROFILE,
        "generated_at_utc": timestamp,
        "private_artifact": True,
        "public_release_authorized": False,
        "repository_binding": {
            "repository_commit": PINNED_REPOSITORY_COMMIT,
            "repository_tree_sha256": plan["repository_provenance"][
                "repository_tree_sha256"
            ],
            "media_plan_sha256": plan["plan_sha256"],
        },
        "input_hashes": input_hashes,
        "selection": {
            "profile_id": PAPER_TRACK1_PROFILE,
            "selected_train_lesson_ids": list(
                PAPER_TRACK1_TRAIN_LESSON_IDS
            ),
            "selected_test_lesson_ids": list(PAPER_TRACK1_TEST_LESSON_IDS),
            "excluded_official_test_lesson_ids": list(
                EXCLUDED_OFFICIAL_TEST_LESSON_IDS
            ),
            "selected_lesson_count": len(lesson_records),
            "selected_train_scene_count": (
                PAPER_TRACK1_EXPECTED_TRAIN_SCENE_COUNT
            ),
            "selected_test_scene_count": (
                PAPER_TRACK1_EXPECTED_TEST_SCENE_COUNT
            ),
            "selected_scene_count": PAPER_TRACK1_EXPECTED_SCENE_COUNT,
            "ordered_sample_id_sha256": ordered_sample_id_sha256,
        },
        "policy": _policy(),
        "aggregate": aggregate,
        "lessons": lesson_records,
        "ordered_sample_id_sha256": ordered_sample_id_sha256,
        "ordered_scene_text_sha256": ordered_scene_text_sha256,
        "claims": _claims(),
    }
    manifest["materialization_fingerprint_sha256"] = _canonical_sha256(
        _manifest_fingerprint_payload(manifest)
    )
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_path = write_json(output / "manifest.json", manifest)
    receipt_path: Path | None = None
    if public_receipt_path is not None:
        receipt = build_public_teachobs_transcript_materialization_receipt(
            manifest,
            private_manifest_path=manifest_path,
        )
        receipt_path = write_json(public_receipt_path, receipt)
    return {
        "output_directory": str(output),
        "manifest_path": str(manifest_path.resolve()),
        "public_receipt_path": (
            str(receipt_path.resolve()) if receipt_path is not None else None
        ),
        "manifest": manifest,
    }


def _parse_lesson_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TeachObsTranscriptMaterializationError(
            "materialized transcript JSONL is not readable UTF-8"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise TeachObsTranscriptMaterializationError(
                f"materialized transcript JSONL has a blank line at {line_number}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TeachObsTranscriptMaterializationError(
                f"materialized transcript JSONL is invalid at line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise TeachObsTranscriptMaterializationError(
                "materialized transcript JSONL contains a non-object row"
            )
        rows.append(row)
    return rows


def _aggregate_rows(
    rows: Sequence[dict[str, Any]],
    lessons: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    tier_lesson_counts = {tier: 0 for tier in _SOURCE_TIERS}
    tier_scene_counts = {tier: 0 for tier in _SOURCE_TIERS}
    for lesson in lessons:
        tier_lesson_counts[lesson["source_tier"]] += 1
        tier_scene_counts[lesson["source_tier"]] += lesson["scene_count"]
    platform_adjustments = [
        lesson["source_timeline_adjustment"]
        for lesson in lessons
        if lesson["source_tier"].startswith("platform_")
    ]
    asr_adjustments = [
        lesson["source_timeline_adjustment"]
        for lesson in lessons
        if lesson["source_tier"] == "audited_asr_fallback"
    ]
    clipped_counts = [
        int(item["endpoint_clamped_item_count"])
        for item in platform_adjustments
    ]
    clipped_seconds = [
        float(item["total_endpoint_clipped_seconds"])
        for item in platform_adjustments
    ]
    maximum_clipped_seconds = [
        float(item["maximum_endpoint_clipped_seconds"])
        for item in platform_adjustments
    ]
    return {
        "lesson_count": len(lessons),
        "scene_count": len(rows),
        "nonempty_scene_count": sum(not row["empty_transcript"] for row in rows),
        "empty_scene_count": sum(row["empty_transcript"] for row in rows),
        "positive_overlap_item_count": sum(
            row["positive_overlap_item_count"] for row in rows
        ),
        "source_tier_lesson_counts": tier_lesson_counts,
        "source_tier_scene_counts": tier_scene_counts,
        "platform_cue_timeline_intersection_policy": (
            PLATFORM_CUE_TIMELINE_INTERSECTION_POLICY
        ),
        "platform_cue_source_item_count": sum(
            int(item["source_item_count"]) for item in platform_adjustments
        ),
        "platform_cue_retained_item_count": sum(
            int(item["retained_item_count"]) for item in platform_adjustments
        ),
        "platform_cue_endpoint_clamped_lesson_count": sum(
            count > 0 for count in clipped_counts
        ),
        "platform_cue_endpoint_clamped_item_count": sum(clipped_counts),
        "platform_cue_total_endpoint_clipped_seconds": round(
            sum(clipped_seconds), 6
        ),
        "platform_cue_maximum_endpoint_clipped_seconds": round(
            max(maximum_clipped_seconds, default=0.0), 6
        ),
        "platform_cue_source_text_items_silently_dropped": sum(
            int(item["source_text_items_silently_dropped"])
            for item in platform_adjustments
        ),
        "platform_cue_text_or_labels_used_for_timing_adjustment": False,
        "asr_target_window_projection_policy": (
            ASR_TIMELINE_INTERSECTION_POLICY
        ),
        "asr_source_item_count": sum(
            int(item["source_item_count"]) for item in asr_adjustments
        ),
        "asr_retained_item_count": sum(
            int(item["retained_item_count"]) for item in asr_adjustments
        ),
        "asr_outside_selected_timeline_item_count": sum(
            int(item["outside_selected_timeline_item_count"])
            for item in asr_adjustments
        ),
        "asr_outside_selected_timeline_item_duration_seconds": round(
            sum(
                float(item["outside_selected_timeline_item_duration_seconds"])
                for item in asr_adjustments
            ),
            6,
        ),
        "asr_endpoint_clamped_item_count": sum(
            int(item["endpoint_clamped_item_count"])
            for item in asr_adjustments
        ),
        "asr_total_endpoint_clipped_seconds": round(
            sum(
                float(item["total_endpoint_clipped_seconds"])
                for item in asr_adjustments
            ),
            6,
        ),
        "asr_all_source_items_within_hash_bound_media": all(
            item["all_source_items_within_hash_bound_media"]
            for item in asr_adjustments
        ),
        "asr_source_text_items_silently_dropped": sum(
            int(item["source_text_items_silently_dropped"])
            for item in asr_adjustments
        ),
        "asr_text_or_labels_used_for_timing_adjustment": False,
    }


def _validate_source_timeline_adjustment(
    value: Any,
    *,
    source_tier: str,
    scene_count: int,
    lesson_id: str,
) -> dict[str, Any]:
    is_asr = source_tier == "audited_asr_fallback"
    expected_fields = (
        _ASR_TIMELINE_ADJUSTMENT_FIELDS
        if is_asr
        else _TIMELINE_ADJUSTMENT_FIELDS
    )
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs source timeline adjustment is malformed for {lesson_id}"
        )
    expected_policy = (
        ASR_TIMELINE_INTERSECTION_POLICY
        if is_asr
        else PLATFORM_CUE_TIMELINE_INTERSECTION_POLICY
    )
    integer_fields = [
        "source_item_count",
        "retained_item_count",
        "endpoint_clamped_item_count",
        "source_text_items_silently_dropped",
    ]
    if is_asr:
        integer_fields.append("outside_selected_timeline_item_count")
    if any(
        not isinstance(value.get(field), int)
        or isinstance(value.get(field), bool)
        or int(value[field]) < 0
        for field in integer_fields
    ):
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs source timeline adjustment counts are invalid for {lesson_id}"
        )
    try:
        timeline_start = float(value["selected_timeline_start_seconds"])
        timeline_end = float(value["selected_timeline_end_seconds"])
        total_clipped = float(value["total_endpoint_clipped_seconds"])
        maximum_clipped = float(value["maximum_endpoint_clipped_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs source timeline adjustment values are invalid for {lesson_id}"
        ) from exc
    clamped_count = int(value["endpoint_clamped_item_count"])
    source_count = int(value["source_item_count"])
    retained_count = int(value["retained_item_count"])
    if (
        value.get("policy") != expected_policy
        or not math.isfinite(timeline_start)
        or not math.isfinite(timeline_end)
        or timeline_start != 0.0
        or not math.isclose(timeline_end, scene_count * 15.0)
        or source_count < 1
        or retained_count < 1
        or retained_count > source_count
        or not 0 <= clamped_count <= retained_count
        or not math.isfinite(total_clipped)
        or not math.isfinite(maximum_clipped)
        or total_clipped < 0
        or maximum_clipped < 0
        or maximum_clipped > total_clipped
        or (clamped_count == 0) != (total_clipped == 0)
        or (clamped_count == 0) != (maximum_clipped == 0)
        or value.get("source_text_items_silently_dropped") != 0
        or value.get("text_or_labels_used_for_timing_adjustment") is not False
    ):
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs source timeline adjustment is invalid for {lesson_id}"
        )
    if not is_asr:
        if (
            retained_count != source_count
            or value.get(
                "all_source_items_have_positive_timeline_intersection"
            )
            is not True
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs platform timeline adjustment is invalid for {lesson_id}"
            )
        return value
    try:
        media_end = float(value["hash_bound_media_end_seconds"])
        outside_duration = float(
            value["outside_selected_timeline_item_duration_seconds"]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs ASR target-window projection is invalid for {lesson_id}"
        ) from exc
    outside_count = int(value["outside_selected_timeline_item_count"])
    if (
        not math.isfinite(media_end)
        or media_end <= timeline_start
        or not math.isfinite(outside_duration)
        or outside_duration < 0
        or retained_count + outside_count != source_count
        or (outside_count == 0) != (outside_duration == 0)
        or value.get(
            "all_source_items_have_positive_timeline_intersection"
        )
        is not (outside_count == 0)
        or value.get("all_source_items_within_hash_bound_media") is not True
    ):
        raise TeachObsTranscriptMaterializationError(
            f"TeachObs ASR target-window projection is invalid for {lesson_id}"
        )
    return value


def validate_teachobs_transcript_materialization(
    manifest_path: str | Path,
    *,
    materialization_root: str | Path | None = None,
    expected_profile: str = PAPER_TRACK1_PROFILE,
) -> dict[str, Any]:
    """Load and fully verify every private materialized scene row."""

    if expected_profile != PAPER_TRACK1_PROFILE:
        raise TeachObsTranscriptMaterializationError(
            "only paper_track1_23_train_6_test is supported"
        )
    source, manifest = _load_json_object(
        manifest_path, description="TeachObs transcript materialization manifest"
    )
    if materialization_root is None:
        root = source.parent
    else:
        root = _safe_input_directory(
            materialization_root,
            description="TeachObs transcript materialization root",
        )
    if (
        set(manifest) != _MANIFEST_FIELDS
        or
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("profile_id") != expected_profile
        or manifest.get("private_artifact") is not True
        or manifest.get("public_release_authorized") is not False
    ):
        raise TeachObsTranscriptMaterializationError(
            "unsupported TeachObs transcript materialization manifest"
        )
    _validate_timestamp(manifest.get("generated_at_utc"))
    _verify_canonical_digest(
        manifest,
        field="manifest_sha256",
        description="TeachObs transcript materialization manifest",
    )
    repository = manifest.get("repository_binding")
    input_hashes = manifest.get("input_hashes")
    selection = manifest.get("selection")
    policy = manifest.get("policy")
    aggregate = manifest.get("aggregate")
    claims = manifest.get("claims")
    lessons = manifest.get("lessons")
    if not all(
        isinstance(value, dict)
        for value in (
            repository,
            input_hashes,
            selection,
            policy,
            aggregate,
            claims,
        )
    ) or not isinstance(lessons, list):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs materialization manifest has malformed contracts"
        )
    assert isinstance(repository, dict)
    assert isinstance(input_hashes, dict)
    assert isinstance(selection, dict)
    assert isinstance(policy, dict)
    assert isinstance(aggregate, dict)
    assert isinstance(claims, dict)
    if (
        set(repository)
        != {
            "repository_commit",
            "repository_tree_sha256",
            "media_plan_sha256",
        }
        or set(input_hashes) != _INPUT_HASH_FIELDS
        or
        repository.get("repository_commit") != PINNED_REPOSITORY_COMMIT
        or repository.get("media_plan_sha256")
        != input_hashes.get("media_plan_canonical_sha256")
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs materialization repository binding is invalid"
        )
    _require_sha256(
        repository.get("repository_tree_sha256"),
        field="repository_tree_sha256",
    )
    for field, value in input_hashes.items():
        if value is not None:
            _require_sha256(value, field=field)
    always_bound_hashes = (
        "media_plan_file_sha256",
        "media_plan_canonical_sha256",
        "media_manifest_file_sha256",
        "caption_audit_file_sha256",
        "caption_audit_canonical_sha256",
        "asr_result_set_sha256",
    )
    if any(input_hashes.get(field) is None for field in always_bound_hashes):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs materialization is missing a required input hash"
        )
    asr_hashes = (
        "asr_import_audit_file_sha256",
        "asr_import_audit_canonical_sha256",
        "coverage_matrix_file_sha256",
        "coverage_matrix_canonical_sha256",
        "asr_job_manifest_file_sha256",
        "asr_job_manifest_canonical_sha256",
    )
    if len({input_hashes.get(field) is None for field in asr_hashes}) != 1:
        raise TeachObsTranscriptMaterializationError(
            "TeachObs materialization has a partial ASR provenance chain"
        )
    expected_selection = {
        "profile_id": PAPER_TRACK1_PROFILE,
        "selected_train_lesson_ids": list(PAPER_TRACK1_TRAIN_LESSON_IDS),
        "selected_test_lesson_ids": list(PAPER_TRACK1_TEST_LESSON_IDS),
        "excluded_official_test_lesson_ids": list(
            EXCLUDED_OFFICIAL_TEST_LESSON_IDS
        ),
        "selected_lesson_count": 29,
        "selected_train_scene_count": PAPER_TRACK1_EXPECTED_TRAIN_SCENE_COUNT,
        "selected_test_scene_count": PAPER_TRACK1_EXPECTED_TEST_SCENE_COUNT,
        "selected_scene_count": PAPER_TRACK1_EXPECTED_SCENE_COUNT,
        "ordered_sample_id_sha256": manifest.get(
            "ordered_sample_id_sha256"
        ),
    }
    if selection != expected_selection or policy != _policy() or claims != _claims():
        raise TeachObsTranscriptMaterializationError(
            "TeachObs materialization profile, policy, or claim boundary changed"
        )
    expected_lesson_ids = [
        *PAPER_TRACK1_TRAIN_LESSON_IDS,
        *PAPER_TRACK1_TEST_LESSON_IDS,
    ]
    if (
        len(lessons) != 29
        or [item.get("lesson_id") for item in lessons] != expected_lesson_ids
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs materialization lesson order differs from the fixed profile"
        )

    rows: list[dict[str, Any]] = []
    text_by_sample_id: dict[str, str] = {}
    for lesson in lessons:
        if not isinstance(lesson, dict):
            raise TeachObsTranscriptMaterializationError(
                "TeachObs materialization has a non-object lesson"
            )
        if set(lesson) != _LESSON_FIELDS:
            raise TeachObsTranscriptMaterializationError(
                "TeachObs materialized lesson has unexpected fields"
            )
        lesson_id = _safe_lesson_id(lesson.get("lesson_id"))
        expected_split = (
            "test" if lesson_id in PAPER_TRACK1_TEST_LESSON_IDS else "train"
        )
        source_tier = str(lesson.get("source_tier") or "")
        source_language = str(lesson.get("source_language") or "")
        source_digest = _require_sha256(
            lesson.get("source_sha256"), field="source_sha256"
        )
        scene_count = lesson.get("scene_count")
        if (
            not isinstance(scene_count, int)
            or isinstance(scene_count, bool)
            or scene_count < 1
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs materialized scene count is invalid for {lesson_id}"
            )
        _require_sha256(
            lesson.get("official_scene_manifest_sha256"),
            field="official_scene_manifest_sha256",
        )
        try:
            timeline_span_fraction = float(
                lesson["timeline_span_fraction"]
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs materialized source coverage is invalid for {lesson_id}"
            ) from exc
        expected_track_type = {
            "platform_creator_provided_caption": "manual_creator_provided",
            "platform_automatic_caption": "youtube_automatic_caption",
            "audited_asr_fallback": "automatic_speech_recognition",
        }.get(source_tier)
        _validate_source_timeline_adjustment(
            lesson.get("source_timeline_adjustment"),
            source_tier=source_tier,
            scene_count=scene_count,
            lesson_id=lesson_id,
        )
        if (
            lesson.get("split") != expected_split
            or source_tier not in _SOURCE_TIERS
            or lesson.get("source_track_type") != expected_track_type
            or not _LANGUAGE_RE.fullmatch(source_language)
            or not math.isfinite(timeline_span_fraction)
            or not MINIMUM_PLATFORM_TIMELINE_SPAN_FRACTION
            <= timeline_span_fraction
            <= 1.0
            or lesson.get("labels_read_or_used") is not False
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs materialized lesson metadata is invalid for {lesson_id}"
            )
        relative = _safe_relative_path(
            lesson.get("transcript_relative_path"), suffix=".jsonl"
        )
        if relative != PurePosixPath("lessons", f"{lesson_id}.jsonl"):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs materialized lesson path is not identity-bound for {lesson_id}"
            )
        transcript_path = _bound_file(root, relative, suffix=".jsonl")
        if file_sha256(transcript_path) != _require_sha256(
            lesson.get("transcript_file_sha256"),
            field="transcript_file_sha256",
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs materialized transcript hash mismatch for {lesson_id}"
            )
        lesson_rows = _parse_lesson_jsonl(transcript_path)
        if (
            not lesson_rows
            or scene_count != len(lesson_rows)
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs materialized scene count mismatch for {lesson_id}"
            )
        sample_ids: list[str] = []
        text_bindings: list[dict[str, str]] = []
        previous_end = 0.0
        for expected_scene_no, row in enumerate(lesson_rows, start=1):
            if set(row) != _SCENE_FIELDS:
                raise TeachObsTranscriptMaterializationError(
                    "TeachObs materialized scene contains unexpected fields"
                )
            sample_id = f"{lesson_id}:{expected_scene_no}"
            text = row.get("text")
            overlap_count = row.get("positive_overlap_item_count")
            try:
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise TeachObsTranscriptMaterializationError(
                    f"TeachObs materialized scene timing is invalid for {sample_id}"
                ) from exc
            if (
                row.get("schema") != LESSON_JSONL_SCHEMA
                or row.get("sample_id") != sample_id
                or row.get("lesson_id") != lesson_id
                or row.get("scene_no") != expected_scene_no
                or not math.isclose(start, (expected_scene_no - 1) * 15.0)
                or not math.isclose(end, expected_scene_no * 15.0)
                or (expected_scene_no > 1 and not math.isclose(start, previous_end))
                or not isinstance(text, str)
                or "\x00" in text
                or row.get("text_sha256") != _text_sha256(text)
                or row.get("source_tier") != source_tier
                or row.get("source_language") != source_language
                or row.get("source_sha256") != source_digest
                or not isinstance(overlap_count, int)
                or isinstance(overlap_count, bool)
                or overlap_count < 0
                or row.get("empty_transcript") is not (not bool(text))
                or row.get("labels_read_or_used") is not False
            ):
                raise TeachObsTranscriptMaterializationError(
                    f"TeachObs materialized scene row is invalid for {sample_id}"
                )
            if sample_id in text_by_sample_id:
                raise TeachObsTranscriptMaterializationError(
                    "TeachObs materialization contains duplicate sample ids"
                )
            sample_ids.append(sample_id)
            text_bindings.append(
                {"sample_id": sample_id, "text_sha256": row["text_sha256"]}
            )
            text_by_sample_id[sample_id] = text
            rows.append(row)
            previous_end = end
        if (
            lesson.get("ordered_sample_id_sha256")
            != _canonical_sha256(sample_ids)
            or lesson.get("ordered_scene_text_sha256")
            != _canonical_sha256(text_bindings)
        ):
            raise TeachObsTranscriptMaterializationError(
                f"TeachObs materialized lesson order/text binding mismatch for {lesson_id}"
            )

    ordered_sample_ids = [row["sample_id"] for row in rows]
    text_bindings = [
        {"sample_id": row["sample_id"], "text_sha256": row["text_sha256"]}
        for row in rows
    ]
    ordered_sample_digest = _canonical_sha256(ordered_sample_ids)
    ordered_text_digest = _canonical_sha256(text_bindings)
    if (
        len(rows) != PAPER_TRACK1_EXPECTED_SCENE_COUNT
        or len(text_by_sample_id) != PAPER_TRACK1_EXPECTED_SCENE_COUNT
        or manifest.get("ordered_sample_id_sha256") != ordered_sample_digest
        or selection.get("ordered_sample_id_sha256") != ordered_sample_digest
        or manifest.get("ordered_scene_text_sha256") != ordered_text_digest
        or aggregate != _aggregate_rows(rows, lessons)
        or manifest.get("materialization_fingerprint_sha256")
        != _canonical_sha256(_manifest_fingerprint_payload(manifest))
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs materialization aggregate, order, text, or fingerprint mismatch"
        )
    return {
        "manifest": manifest,
        "manifest_path": source,
        "materialization_root": root,
        "rows": tuple(rows),
        "text_by_sample_id": text_by_sample_id,
    }


def build_public_teachobs_transcript_materialization_receipt(
    private_manifest: dict[str, Any],
    *,
    private_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build an allowlisted receipt containing no transcript or lesson identity."""

    if (
        not isinstance(private_manifest, dict)
        or private_manifest.get("schema") != MANIFEST_SCHEMA
    ):
        raise TeachObsTranscriptMaterializationError(
            "unsupported TeachObs transcript materialization manifest"
        )
    _verify_canonical_digest(
        private_manifest,
        field="manifest_sha256",
        description="TeachObs transcript materialization manifest",
    )
    if (
        set(private_manifest) != _MANIFEST_FIELDS
        or private_manifest.get("profile_id") != PAPER_TRACK1_PROFILE
        or private_manifest.get("private_artifact") is not True
        or private_manifest.get("public_release_authorized") is not False
        or private_manifest.get("policy") != _policy()
        or private_manifest.get("claims") != _claims()
        or private_manifest.get("materialization_fingerprint_sha256")
        != _canonical_sha256(
            _manifest_fingerprint_payload(private_manifest)
        )
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs materialization profile, fingerprint, or claim boundary changed"
        )
    aggregate = private_manifest.get("aggregate")
    if (
        not isinstance(aggregate, dict)
        or aggregate.get("lesson_count") != 29
        or aggregate.get("scene_count") != PAPER_TRACK1_EXPECTED_SCENE_COUNT
        or (
            aggregate.get("nonempty_scene_count", -1)
            + aggregate.get("empty_scene_count", -1)
            != PAPER_TRACK1_EXPECTED_SCENE_COUNT
        )
    ):
        raise TeachObsTranscriptMaterializationError(
            "TeachObs materialization aggregate is missing or invalid"
        )
    private_file_digest: str | None = None
    if private_manifest_path is not None:
        path = _safe_input_file(
            private_manifest_path,
            description="TeachObs transcript materialization manifest",
        )
        private_file_digest = file_sha256(path)
    receipt = {
        "artifact_kind": "teachobs_audited_transcript_materialization_receipt",
        "schema": PUBLIC_RECEIPT_SCHEMA,
        "generated_at_utc": private_manifest.get("generated_at_utc"),
        "profile_id": PAPER_TRACK1_PROFILE,
        "private_manifest_file_sha256": private_file_digest,
        "private_manifest_canonical_sha256": private_manifest[
            "manifest_sha256"
        ],
        "materialization_fingerprint_sha256": private_manifest[
            "materialization_fingerprint_sha256"
        ],
        "ordered_sample_id_sha256": private_manifest[
            "ordered_sample_id_sha256"
        ],
        "ordered_scene_text_sha256": private_manifest[
            "ordered_scene_text_sha256"
        ],
        "aggregate": {
            "lesson_count": aggregate["lesson_count"],
            "scene_count": aggregate["scene_count"],
            "nonempty_scene_count": aggregate["nonempty_scene_count"],
            "empty_scene_count": aggregate["empty_scene_count"],
            "source_tier_lesson_counts": aggregate[
                "source_tier_lesson_counts"
            ],
            "source_tier_scene_counts": aggregate[
                "source_tier_scene_counts"
            ],
            "platform_cue_timeline_intersection_policy": aggregate[
                "platform_cue_timeline_intersection_policy"
            ],
            "platform_cue_source_item_count": aggregate[
                "platform_cue_source_item_count"
            ],
            "platform_cue_retained_item_count": aggregate[
                "platform_cue_retained_item_count"
            ],
            "platform_cue_endpoint_clamped_lesson_count": aggregate[
                "platform_cue_endpoint_clamped_lesson_count"
            ],
            "platform_cue_endpoint_clamped_item_count": aggregate[
                "platform_cue_endpoint_clamped_item_count"
            ],
            "platform_cue_total_endpoint_clipped_seconds": aggregate[
                "platform_cue_total_endpoint_clipped_seconds"
            ],
            "platform_cue_maximum_endpoint_clipped_seconds": aggregate[
                "platform_cue_maximum_endpoint_clipped_seconds"
            ],
            "platform_cue_source_text_items_silently_dropped": aggregate[
                "platform_cue_source_text_items_silently_dropped"
            ],
            "platform_cue_text_or_labels_used_for_timing_adjustment": aggregate[
                "platform_cue_text_or_labels_used_for_timing_adjustment"
            ],
            "asr_target_window_projection_policy": aggregate[
                "asr_target_window_projection_policy"
            ],
            "asr_source_item_count": aggregate["asr_source_item_count"],
            "asr_retained_item_count": aggregate["asr_retained_item_count"],
            "asr_outside_selected_timeline_item_count": aggregate[
                "asr_outside_selected_timeline_item_count"
            ],
            "asr_outside_selected_timeline_item_duration_seconds": aggregate[
                "asr_outside_selected_timeline_item_duration_seconds"
            ],
            "asr_endpoint_clamped_item_count": aggregate[
                "asr_endpoint_clamped_item_count"
            ],
            "asr_total_endpoint_clipped_seconds": aggregate[
                "asr_total_endpoint_clipped_seconds"
            ],
            "asr_all_source_items_within_hash_bound_media": aggregate[
                "asr_all_source_items_within_hash_bound_media"
            ],
            "asr_source_text_items_silently_dropped": aggregate[
                "asr_source_text_items_silently_dropped"
            ],
            "asr_text_or_labels_used_for_timing_adjustment": aggregate[
                "asr_text_or_labels_used_for_timing_adjustment"
            ],
        },
        "evidence_status": {
            "source_selection_and_scene_alignment_audited": True,
            "platform_cue_timeline_intersection_provenance_recorded": True,
            "platform_cue_text_or_labels_used_for_timing_adjustment": False,
            "platform_cue_text_silently_dropped_during_timing_adjustment": False,
            "asr_target_window_projection_provenance_recorded": True,
            "asr_outside_target_window_items_explicitly_counted": True,
            "asr_all_source_items_within_hash_bound_media": aggregate[
                "asr_all_source_items_within_hash_bound_media"
            ],
            "asr_text_or_labels_used_for_timing_adjustment": False,
            "asr_text_silently_dropped_during_timing_adjustment": False,
            "labels_read_or_used": False,
            "released_transcript_fallback_used": False,
            "asr_is_official_caption": False,
            "content_accuracy_established": False,
            "word_error_rate_established": False,
            "independent_human_content_audit_completed": False,
            "confirmatory_multimodal_gain_established": False,
            "deployment_accuracy_established": False,
            "learner_effect_established": False,
        },
        "metric_boundary": (
            "Hash-bound source selection, deterministic positive timeline "
            "intersection, explicit full-media ASR target-window projection, "
            "endpoint-clipping provenance, and scene alignment do not "
            "establish transcript accuracy or WER."
        ),
        "content_exclusion": {
            "transcript_text_included": False,
            "caption_or_asr_segments_included": False,
            "lesson_or_video_ids_included": False,
            "source_urls_included": False,
            "filesystem_paths_included": False,
            "scene_level_records_included": False,
            "released_consensus_labels_included": False,
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
        raise TeachObsTranscriptMaterializationError(
            "public TeachObs materialization receipt leaked private identity data"
        )
    return receipt


__all__ = [
    "ASR_TIMELINE_INTERSECTION_POLICY",
    "LESSON_JSONL_SCHEMA",
    "MANIFEST_SCHEMA",
    "PAPER_TRACK1_PROFILE",
    "PLATFORM_CUE_TIMELINE_INTERSECTION_POLICY",
    "PUBLIC_RECEIPT_SCHEMA",
    "TeachObsTranscriptMaterializationError",
    "build_public_teachobs_transcript_materialization_receipt",
    "materialize_teachobs_transcripts",
    "validate_teachobs_transcript_materialization",
]
