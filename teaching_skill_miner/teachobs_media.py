"""Private, resumable TeachObs source-media and feature preparation.

TeachObs releases annotations under CC BY 4.0, but the linked classroom
videos retain their original rights and platform terms.  This module therefore
has no public-receipt builder: source URLs, lesson identifiers, frames, media,
audio-derived rows, and CLIP embeddings stay below a caller-supplied private
directory.  Execution is gated on an explicit source-terms acknowledgement.

The implementation binds every result to the fixed author-repository commit,
the audited extracted-tree hash, the lesson table, the per-lesson scene
manifest, and the locally observed media SHA-256.  Local media hashes detect
later mutation; they are not publisher-provided authenticity hashes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
import importlib.metadata
import importlib.util
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Sequence
from urllib.parse import parse_qs, urlsplit

from .full_video_dataset import (
    file_sha256,
    probe_local_media,
    validate_media_probe,
)
from .io_utils import (
    ensure_private_directory,
    ensure_private_file,
    read_json,
    write_json,
)


PINNED_REPOSITORY = "https://github.com/codingchild2424/teacherOps"
PINNED_REPOSITORY_COMMIT = "96c251ae09e79edd06a3a9bbaaa8b8f7fe99a15c"
DATASET_ID = "teachobs_v0_1_human_validated"
EXPECTED_LESSON_COUNT = 30
EXPECTED_TEST_IDS = {"S2", "S4", "S5", "S19", "S24", "S28", "S30"}
SCENE_SECONDS = 15.0
TAIL_FRAME_MARGIN_SECONDS = 0.25
MAXIMUM_TAIL_CLAMP_DELTA_SECONDS = SCENE_SECONDS
FRAME_TIMESTAMP_TOLERANCE_SECONDS = 0.1
FRAME_EXTRACTION_METHOD = (
    "ffmpeg_explicit_midpoint_select_with_showinfo_verification_v3"
)
AUDIO_SAMPLE_RATE_HZ = 16_000
AUDIO_SILENCE_THRESHOLD = 0.01
AUDIO_MEDIA_EOF_ALIGNMENT_TOLERANCE_SECONDS = 1.0

PLAN_SCHEMA = "teaching_skill_miner.teachobs_private_media_plan.v1"
MEDIA_SCHEMA = "teaching_skill_miner.teachobs_private_media_manifest.v1"
FRAME_TASK_SCHEMA = "teaching_skill_miner.visual_semantic_tasks.v1"
CLIP_RESULT_SCHEMA = "teaching_skill_miner.visual_semantic_results.v1"
AUDIO_SCHEMA = "teaching_skill_miner.teachobs_private_audio_features.v1"
FEATURE_SCHEMA = "teaching_skill_miner.teachobs_private_feature_manifest.v1"
FEATURE_FAILURE_SCHEMA = (
    "teaching_skill_miner.teachobs_private_feature_failures.v1"
)
_LEGACY_VISUAL_EVIDENCE_SCHEMA_V1 = (
    "teaching_skill_miner.teachobs_private_scene_visual_evidence.v1"
)
VISUAL_EVIDENCE_SCHEMA = (
    "teaching_skill_miner.teachobs_private_scene_visual_evidence.v2"
)
VISUAL_EVIDENCE_EVENT_TYPES = (
    "scene_change",
    "slide_change",
    "board_build_up",
    "code_or_formula_visible",
)
_VISUAL_EVIDENCE_TRANSITION_EVENT_TYPES = VISUAL_EVIDENCE_EVENT_TYPES[:3]
VISUAL_EVIDENCE_EVENT_COUNTING_POLICY = "all_events_exact_v2"
SOURCE_OVERRIDE_SCHEMA = (
    "teaching_skill_miner.teachobs_private_source_override_manifest.v1"
)

_LESSON_ID_RE = re.compile(r"^S(?:[1-9]|[12][0-9]|30)$")
_YOUTUBE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,32}$")
_COOKIE_BROWSER_FAMILIES = frozenset(
    {
        "brave",
        "chrome",
        "chromium",
        "edge",
        "firefox",
        "opera",
        "safari",
        "vivaldi",
        "whale",
    }
)
_COOKIE_BROWSER_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$")
_YTDLP_IMPERSONATION_TARGETS = frozenset({"chrome"})
_SHOWINFO_PTS_TIME_RE = re.compile(
    r"\bpts_time:(?P<timestamp>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_MEDIA_SUFFIXES = {".mp4"}
_LESSON_FIELDS = {
    "id",
    "week",
    "subject",
    "school_level",
    "country",
    "duration",
    "source",
    "youtube_url",
    "split",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
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


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _safe_lesson_id(value: Any) -> str:
    lesson_id = str(value).strip()
    if not _LESSON_ID_RE.fullmatch(lesson_id):
        raise ValueError(f"unsafe or unsupported TeachObs lesson id: {value!r}")
    return lesson_id


def _lesson_sort_key(value: str) -> int:
    return int(value[1:])


def _safe_manifest_filename(value: Any, *, field: str) -> str:
    text = str(value).strip()
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name in {"", ".", ".."}
        or "\\" in text
        or "\x00" in text
    ):
        raise ValueError(f"unsafe TeachObs {field}: {value!r}")
    return text


def _validate_youtube_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("TeachObs source-video URL must be non-empty")
    url = value.strip()
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("TeachObs source-video URL has an invalid port") from exc
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or port not in {None, 443}
        or parsed.fragment
    ):
        raise ValueError("TeachObs source-video URL must be a plain HTTPS URL")
    host = (parsed.hostname or "").lower().rstrip(".")
    video_id = ""
    if host in {"youtube.com", "www.youtube.com"}:
        if parsed.path != "/watch":
            raise ValueError("TeachObs YouTube URL must use the /watch endpoint")
        query = parse_qs(parsed.query, keep_blank_values=True)
        if set(query) != {"v"} or len(query["v"]) != 1:
            raise ValueError("TeachObs YouTube URL has unexpected query parameters")
        video_id = query["v"][0]
    elif host == "youtu.be":
        if parsed.query:
            raise ValueError("TeachObs youtu.be URL has unexpected query parameters")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 1:
            raise ValueError("TeachObs youtu.be URL has an invalid path")
        video_id = parts[0]
    else:
        raise ValueError("TeachObs source-video URL must use an approved YouTube host")
    if not _YOUTUBE_VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError("TeachObs source-video URL has an invalid video id")
    return url


def _youtube_video_id(value: str) -> str:
    url = _validate_youtube_url(value)
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold().rstrip(".")
    if host == "youtu.be":
        return parsed.path.strip("/")
    return parse_qs(parsed.query, keep_blank_values=True)["v"][0]


def _validate_override_https_url(value: Any) -> str:
    """Validate an explicit mirror URL without silently broadening plan URLs."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("TeachObs override source URL must be a non-empty exact string")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("TeachObs override source URL has an invalid port") from exc
    host = (parsed.hostname or "").casefold().rstrip(".")
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or port not in {None, 443}
        or parsed.fragment
        or not host
        or host == "localhost"
        or host.endswith(".localhost")
        or host.endswith(".local")
    ):
        raise ValueError(
            "TeachObs override source URL must be a public plain HTTPS URL"
        )
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("TeachObs override source URL cannot use a non-public IP")
    return value


def _parse_duration_seconds(value: Any) -> float:
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) not in {2, 3} or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid TeachObs duration: {value!r}")
    numbers = [int(part) for part in parts]
    if numbers[-1] >= 60 or (len(numbers) == 3 and numbers[-2] >= 60):
        raise ValueError(f"invalid TeachObs duration: {value!r}")
    if len(numbers) == 2:
        minutes, seconds = numbers
        duration = minutes * 60 + seconds
    else:
        hours, minutes, seconds = numbers
        duration = hours * 3600 + minutes * 60 + seconds
    if duration <= 0:
        raise ValueError("TeachObs duration must be positive")
    return float(duration)


def _tree_manifest_sha256(root: Path) -> tuple[str, int, int]:
    """Reproduce the tree digest recorded by the acquisition audit."""

    digest = sha256()
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return digest.hexdigest(), len(files), total_bytes


def _validate_repository_binding(
    repository_root: Path,
    *,
    acquisition_receipt_path: str | Path | None,
) -> dict[str, Any]:
    receipt_path = (
        Path(acquisition_receipt_path)
        if acquisition_receipt_path is not None
        else repository_root.parent / "acquisition_receipt.json"
    )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(
            "TeachObs media preparation requires the private acquisition receipt "
            "that pins the repository commit and extracted-tree hash"
        )
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError("TeachObs acquisition receipt must be a JSON object")
    identification = receipt.get("repository_identification")
    extracted = receipt.get("extracted_repository")
    licenses = receipt.get("licenses")
    if not all(isinstance(value, dict) for value in (identification, extracted, licenses)):
        raise ValueError("TeachObs acquisition receipt lacks repository provenance")
    assert isinstance(identification, dict)
    assert isinstance(extracted, dict)
    assert isinstance(licenses, dict)
    if identification.get("fixed_commit") != PINNED_REPOSITORY_COMMIT:
        raise ValueError("TeachObs acquisition receipt pins a different commit")
    if licenses.get("video_redistribution_authorized_by_dataset") is not False:
        raise ValueError("TeachObs receipt does not preserve the source-video boundary")
    expected_tree_digest = str(extracted.get("sha256_manifest_v1", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_tree_digest):
        raise ValueError("TeachObs acquisition receipt has no valid tree digest")
    actual_tree_digest, file_count, total_bytes = _tree_manifest_sha256(repository_root)
    if actual_tree_digest != expected_tree_digest:
        raise ValueError("TeachObs repository tree differs from the pinned acquisition")
    expected_files = extracted.get("files")
    expected_bytes = extracted.get("bytes")
    if expected_files is not None and int(expected_files) != file_count:
        raise ValueError("TeachObs repository file count differs from its receipt")
    if expected_bytes is not None and int(expected_bytes) != total_bytes:
        raise ValueError("TeachObs repository byte count differs from its receipt")
    return {
        "repository_url": PINNED_REPOSITORY,
        "repository_commit": PINNED_REPOSITORY_COMMIT,
        "repository_tree_sha256": actual_tree_digest,
        "repository_file_count": file_count,
        "repository_bytes": total_bytes,
        "acquisition_receipt_sha256": file_sha256(receipt_path),
        "identification_status": identification.get("identification_status"),
    }


def _load_scene_manifest(
    repository_root: Path,
    lesson_id: str,
    *,
    reference_duration_seconds: float,
) -> tuple[list[dict[str, Any]], str]:
    manifest_path = repository_root / "data" / "scenes" / lesson_id / "manifest.jsonl"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"TeachObs scene manifest is missing for {lesson_id}")
    raw = manifest_path.read_bytes()
    rows: list[dict[str, Any]] = []
    required = {
        "id",
        "scene_no",
        "start",
        "end",
        "mid_frame",
        "grid",
        "transcript_file",
    }
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid TeachObs scene JSON for {lesson_id}, line {line_number}"
            ) from exc
        if not isinstance(value, dict) or not required.issubset(value):
            raise ValueError(f"invalid TeachObs scene row for {lesson_id}")
        scene_no = value.get("scene_no")
        if isinstance(scene_no, bool) or not isinstance(scene_no, int):
            raise ValueError(f"invalid TeachObs scene number for {lesson_id}")
        expected_no = len(rows) + 1
        if value.get("id") != lesson_id or scene_no != expected_no:
            raise ValueError(f"non-contiguous TeachObs scene identity for {lesson_id}")
        try:
            start = float(value.get("start"))
            end = float(value.get("end"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid TeachObs scene interval for {lesson_id}") from exc
        expected_start = (scene_no - 1) * SCENE_SECONDS
        expected_end = scene_no * SCENE_SECONDS
        if (
            not math.isclose(start, expected_start, abs_tol=1e-6)
            or not math.isclose(end, expected_end, abs_tol=1e-6)
        ):
            raise ValueError(f"TeachObs scenes are not fixed 15-second windows: {lesson_id}")
        mid_frame = _safe_manifest_filename(value["mid_frame"], field="mid_frame")
        grid = _safe_manifest_filename(value["grid"], field="grid")
        transcript_name = _safe_manifest_filename(
            value["transcript_file"], field="transcript_file"
        )
        transcript_path = manifest_path.parent / transcript_name
        if transcript_path.is_symlink() or not transcript_path.is_file():
            raise ValueError(f"TeachObs scene transcript is missing for {lesson_id}")
        rows.append(
            {
                "scene_no": scene_no,
                "start": round(start, 6),
                "end": round(end, 6),
                "midpoint": round((start + end) / 2.0, 6),
                "declared_mid_frame": mid_frame,
                "declared_grid": grid,
            }
        )
    if not rows:
        raise ValueError(f"TeachObs scene manifest is empty for {lesson_id}")
    if abs(rows[-1]["end"] - reference_duration_seconds) > SCENE_SECONDS:
        raise ValueError(f"TeachObs scene coverage differs from lesson duration: {lesson_id}")
    return rows, _sha256_bytes(raw)


def build_teachobs_media_plan(
    repository_root: str | Path,
    *,
    lesson_ids: Sequence[str] | None = None,
    acquisition_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the pinned repository and construct a private source-media plan."""

    root = Path(repository_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("TeachObs repository root is missing or unsafe")
    provenance = _validate_repository_binding(
        root, acquisition_receipt_path=acquisition_receipt_path
    )
    lessons_path = root / "data" / "lessons.csv"
    if lessons_path.is_symlink() or not lessons_path.is_file():
        raise ValueError("TeachObs lessons.csv is missing")
    lessons_raw = lessons_path.read_bytes()
    text = lessons_raw.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    if set(reader.fieldnames or []) != _LESSON_FIELDS:
        raise ValueError("TeachObs lessons.csv has an unexpected schema")
    lesson_rows = list(reader)
    if len(lesson_rows) != EXPECTED_LESSON_COUNT:
        raise ValueError("TeachObs lessons.csv must contain exactly 30 lessons")
    all_ids = [_safe_lesson_id(row.get("id")) for row in lesson_rows]
    if len(set(all_ids)) != EXPECTED_LESSON_COUNT or set(all_ids) != {
        f"S{index}" for index in range(1, EXPECTED_LESSON_COUNT + 1)
    }:
        raise ValueError("TeachObs lessons.csv must contain unique S1-S30 ids")
    split_by_id: dict[str, str] = {}
    for row, lesson_id in zip(lesson_rows, all_ids):
        split = str(row.get("split", "")).strip()
        if split not in {"train", "test"}:
            raise ValueError(f"invalid TeachObs split for {lesson_id}")
        split_by_id[lesson_id] = split
    if {lesson_id for lesson_id, split in split_by_id.items() if split == "test"} != (
        EXPECTED_TEST_IDS
    ):
        raise ValueError("TeachObs test split differs from the pinned release")

    if lesson_ids is None:
        selected_ids = set(all_ids)
    else:
        selected_list = [_safe_lesson_id(value) for value in lesson_ids]
        if not selected_list or len(set(selected_list)) != len(selected_list):
            raise ValueError("TeachObs lesson selection must be non-empty and unique")
        selected_ids = set(selected_list)
        if not selected_ids.issubset(all_ids):
            raise ValueError("TeachObs lesson selection contains an unknown lesson")

    items: list[dict[str, Any]] = []
    for row, lesson_id in zip(lesson_rows, all_ids):
        if lesson_id not in selected_ids:
            continue
        duration = _parse_duration_seconds(row.get("duration"))
        scenes, scene_manifest_digest = _load_scene_manifest(
            root,
            lesson_id,
            reference_duration_seconds=duration,
        )
        items.append(
            {
                "lesson_id": lesson_id,
                "split": split_by_id[lesson_id],
                "source_url": _validate_youtube_url(row.get("youtube_url")),
                "reference_duration_seconds": duration,
                "scene_manifest_sha256": scene_manifest_digest,
                "scene_count": len(scenes),
                "last_scene_end_seconds": scenes[-1]["end"],
                "scenes": scenes,
            }
        )
    items.sort(key=lambda item: _lesson_sort_key(item["lesson_id"]))
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "dataset_id": DATASET_ID,
        "private_artifact": True,
        "public_release_authorized": False,
        "repository_provenance": provenance,
        "lessons_csv_sha256": _sha256_bytes(lessons_raw),
        "lesson_count": len(items),
        "scene_count": sum(item["scene_count"] for item in items),
        "reference_duration_seconds": sum(
            item["reference_duration_seconds"] for item in items
        ),
        "lessons": items,
        "terms_boundary": {
            "annotation_assets_license": "CC-BY-4.0",
            "source_videos_covered_by_annotation_license": False,
            "original_rights_and_platform_terms_apply": True,
            "redistribution_authorized_by_this_plan": False,
        },
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def _validate_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported TeachObs private-media plan")
    claimed_digest = plan.get("plan_sha256")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if claimed_digest != _canonical_sha256(unsigned):
        raise ValueError("TeachObs private-media plan hash mismatch")
    provenance = plan.get("repository_provenance")
    if not isinstance(provenance, dict) or provenance.get(
        "repository_commit"
    ) != PINNED_REPOSITORY_COMMIT:
        raise ValueError("TeachObs plan is not bound to the pinned commit")
    lessons = plan.get("lessons")
    if not isinstance(lessons, list) or not lessons:
        raise ValueError("TeachObs plan contains no lessons")
    seen: set[str] = set()
    for item in lessons:
        if not isinstance(item, dict):
            raise ValueError("TeachObs plan contains a non-object lesson")
        lesson_id = _safe_lesson_id(item.get("lesson_id"))
        if lesson_id in seen:
            raise ValueError("TeachObs plan contains duplicate lessons")
        seen.add(lesson_id)
        _validate_youtube_url(item.get("source_url"))
        scenes = item.get("scenes")
        if not isinstance(scenes, list) or len(scenes) != item.get("scene_count"):
            raise ValueError(f"TeachObs plan has invalid scenes for {lesson_id}")
    return lessons


def load_teachobs_source_override_manifest(
    source_override_manifest_path: str | Path | None,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Validate a private, explicit retrieval-mirror override manifest.

    Overrides never mutate the canonical URL in the repository-derived plan.
    They only supply a separately hash-bound retrieval URL for named lessons.
    """

    lessons = _validate_plan(plan)
    if source_override_manifest_path is None:
        return {
            "present": False,
            "manifest_file_sha256": None,
            "manifest_canonical_sha256": None,
            "overrides_by_lesson": {},
            "terms": None,
        }
    path = Path(source_override_manifest_path).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError("TeachObs source override manifest is missing or unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "TeachObs source override manifest must be UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("TeachObs source override manifest must be an object")
    required_top_level = {
        "schema",
        "dataset_id",
        "repository_commit",
        "private_artifact",
        "safe_to_publish",
        "terms",
        "overrides",
        "manifest_canonical_sha256",
    }
    if set(value) != required_top_level:
        raise ValueError("TeachObs source override manifest has an unexpected schema")
    if (
        value.get("schema") != SOURCE_OVERRIDE_SCHEMA
        or value.get("dataset_id") != DATASET_ID
        or value.get("repository_commit") != PINNED_REPOSITORY_COMMIT
        or value.get("private_artifact") is not True
        or value.get("safe_to_publish") is not False
    ):
        raise ValueError("TeachObs source override manifest provenance is invalid")
    unsigned = {
        key: item for key, item in value.items() if key != "manifest_canonical_sha256"
    }
    if value.get("manifest_canonical_sha256") != _canonical_sha256(unsigned):
        raise ValueError("TeachObs source override manifest hash mismatch")
    terms = value.get("terms")
    required_terms = {
        "canonical_source_terms_acknowledgement_required",
        "override_source_terms_acknowledgement_required",
        "override_redistribution_authorized",
    }
    if (
        not isinstance(terms, dict)
        or set(terms) != required_terms
        or terms.get("canonical_source_terms_acknowledgement_required") is not True
        or terms.get("override_source_terms_acknowledgement_required") is not True
        or terms.get("override_redistribution_authorized") is not False
    ):
        raise ValueError(
            "TeachObs source override manifest must preserve both terms gates"
        )
    rows = value.get("overrides")
    if not isinstance(rows, list) or not rows:
        raise ValueError("TeachObs source override manifest contains no overrides")
    items_by_id = {item["lesson_id"]: item for item in lessons}
    required_override_fields = {
        "lesson_id",
        "canonical_source_url_sha256",
        "override_source_url",
        "override_source_url_sha256",
        "reason",
        "evidence_metadata",
        "evidence_metadata_sha256",
        "candidate_same_content_mirror",
        "publisher_byte_identity_established",
    }
    required_evidence_fields = {
        "verification_method",
        "observed_title",
        "observed_duration_seconds",
        "official_reference_duration_seconds",
        "duration_absolute_difference_seconds",
        "canonical_source_identifier_reference_observed",
        "canonical_source_identifier_reference",
        "evidence_retrieved_at_utc",
    }
    overrides: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required_override_fields:
            raise ValueError("TeachObs source override row has an unexpected schema")
        lesson_id = _safe_lesson_id(row.get("lesson_id"))
        if lesson_id in overrides:
            raise ValueError("TeachObs source override manifest has duplicate lessons")
        item = items_by_id.get(lesson_id)
        if item is None:
            raise ValueError(
                "TeachObs source override targets a lesson outside the selected plan"
            )
        canonical_digest = _sha256_bytes(item["source_url"].encode("utf-8"))
        if row.get("canonical_source_url_sha256") != canonical_digest:
            raise ValueError(
                f"TeachObs canonical source URL binding differs for {lesson_id}"
            )
        override_url = _validate_override_https_url(row.get("override_source_url"))
        override_digest = _sha256_bytes(override_url.encode("utf-8"))
        if (
            row.get("override_source_url_sha256") != override_digest
            or override_digest == canonical_digest
        ):
            raise ValueError(
                f"TeachObs override source URL binding is invalid for {lesson_id}"
            )
        reason = row.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 12:
            raise ValueError(f"TeachObs source override reason is missing for {lesson_id}")
        evidence = row.get("evidence_metadata")
        if not isinstance(evidence, dict) or set(evidence) != required_evidence_fields:
            raise ValueError(
                f"TeachObs source override evidence has an unexpected schema for {lesson_id}"
            )
        if row.get("evidence_metadata_sha256") != _canonical_sha256(evidence):
            raise ValueError(
                f"TeachObs source override evidence hash differs for {lesson_id}"
            )
        if (
            not isinstance(evidence.get("verification_method"), str)
            or not evidence["verification_method"].strip()
            or not isinstance(evidence.get("observed_title"), str)
            or not evidence["observed_title"].strip()
            or evidence.get("canonical_source_identifier_reference_observed")
            is not True
            or evidence.get("canonical_source_identifier_reference")
            != _youtube_video_id(item["source_url"])
            or not isinstance(evidence.get("evidence_retrieved_at_utc"), str)
            or not evidence["evidence_retrieved_at_utc"].endswith("Z")
        ):
            raise ValueError(
                f"TeachObs source override evidence is incomplete for {lesson_id}"
            )
        try:
            observed_duration = float(evidence["observed_duration_seconds"])
            reference_duration = float(
                evidence["official_reference_duration_seconds"]
            )
            claimed_difference = float(
                evidence["duration_absolute_difference_seconds"]
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"TeachObs source override duration evidence is invalid for {lesson_id}"
            ) from exc
        actual_difference = abs(observed_duration - reference_duration)
        if (
            not all(
                math.isfinite(number)
                for number in (
                    observed_duration,
                    reference_duration,
                    claimed_difference,
                )
            )
            or observed_duration <= 0
            or reference_duration != float(item["reference_duration_seconds"])
            or not math.isclose(claimed_difference, actual_difference, abs_tol=1e-6)
            or actual_difference > max(20.0, 0.01 * reference_duration)
        ):
            raise ValueError(
                f"TeachObs source override duration evidence differs for {lesson_id}"
            )
        if (
            row.get("candidate_same_content_mirror") is not True
            or row.get("publisher_byte_identity_established") is not False
        ):
            raise ValueError(
                "TeachObs source override must remain a candidate mirror without "
                "claiming publisher byte identity"
            )
        overrides[lesson_id] = row
    return {
        "present": True,
        "manifest_file_sha256": _sha256_bytes(raw),
        "manifest_canonical_sha256": value["manifest_canonical_sha256"],
        "overrides_by_lesson": overrides,
        "terms": terms,
    }


def _command_prefix(command: Sequence[str] | str | None) -> list[str]:
    if command is None:
        executable = shutil.which("yt-dlp")
        if executable:
            return [executable]
        if importlib.util.find_spec("yt_dlp") is not None:
            return [sys.executable, "-m", "yt_dlp"]
        raise RuntimeError(
            "TeachObs media download requires yt-dlp; install it privately or pass "
            "yt_dlp_command=(python, '-m', 'yt_dlp')"
        )
    parts = [command] if isinstance(command, str) else list(command)
    if not parts or any(not isinstance(part, str) or not part for part in parts):
        raise ValueError("yt_dlp_command must contain non-empty strings")
    executable = shutil.which(parts[0])
    if not executable:
        raise RuntimeError(f"TeachObs media download cannot find `{parts[0]}`")
    parts[0] = executable
    return parts


def validate_teachobs_cookies_from_browser(
    value: str | None,
) -> tuple[str | None, str | None]:
    """Validate the deliberately narrow yt-dlp browser-cookie opt-in.

    Only a supported browser family and, optionally, a local profile *name* are
    accepted.  Keyring/container syntax and profile paths are intentionally
    excluded: they are unnecessary for the current recovery path and are much
    easier to disclose accidentally in logs or receipts.
    """

    if value is None:
        return None, None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            "cookies_from_browser must be a non-empty, unpadded browser spec"
        )
    if len(value) > 160 or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError("cookies_from_browser contains an unsafe character")
    family, separator, profile = value.partition(":")
    if family not in _COOKIE_BROWSER_FAMILIES:
        raise ValueError("cookies_from_browser uses an unsupported browser family")
    if separator:
        if (
            not profile
            or ":" in profile
            or "/" in profile
            or "\\" in profile
            or ".." in profile
            or _COOKIE_BROWSER_PROFILE_RE.fullmatch(profile) is None
        ):
            raise ValueError(
                "cookies_from_browser profile must be a safe local profile name"
            )
    return value, family


def validate_teachobs_ytdlp_transport(
    *,
    direct: bool = False,
    impersonate: str | None = None,
) -> tuple[bool, str | None]:
    """Validate the opt-in proxy bypass and HTTP impersonation request."""

    if not isinstance(direct, bool):
        raise ValueError("yt_dlp_direct must be a boolean")
    if impersonate is not None:
        if (
            not isinstance(impersonate, str)
            or impersonate not in _YTDLP_IMPERSONATION_TARGETS
        ):
            raise ValueError("yt_dlp_impersonate uses an unsupported target")
    return direct, impersonate


def _local_ytdlp_transport_provenance(
    command_prefix: Sequence[str],
    *,
    js_runtime: str | None,
    impersonation_target: str | None,
) -> dict[str, Any]:
    """Describe only packages loaded by this exact Python ``-m yt_dlp`` path."""

    if (
        len(command_prefix) != 3
        or list(command_prefix[1:]) != ["-m", "yt_dlp"]
        or Path(command_prefix[0]).resolve() != Path(sys.executable).resolve()
    ):
        return {}

    def package_version(name: str) -> str | None:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
        if (
            not version
            or len(version) > 80
            or any(ord(character) < 32 or ord(character) == 127 for character in version)
        ):
            return None
        return version

    provenance: dict[str, Any] = {}
    yt_dlp_version = package_version("yt-dlp")
    if yt_dlp_version is not None:
        provenance["yt_dlp_package_version"] = yt_dlp_version
    if js_runtime:
        runtime_family = js_runtime.partition(":")[0]
        if re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", runtime_family):
            provenance["javascript_runtime_family"] = runtime_family
        ejs_version = package_version("yt-dlp-ejs")
        if ejs_version is not None:
            provenance["local_ejs_package_version"] = ejs_version
            try:
                distribution = importlib.metadata.distribution("yt-dlp-ejs")
                direct_url = json.loads(
                    distribution.read_text("direct_url.json") or "{}"
                )
                archive_digest = (
                    direct_url.get("archive_info", {})
                    .get("hashes", {})
                    .get("sha256")
                )
            except (
                importlib.metadata.PackageNotFoundError,
                json.JSONDecodeError,
                AttributeError,
            ):
                archive_digest = None
            if isinstance(archive_digest, str) and re.fullmatch(
                r"[0-9a-f]{64}", archive_digest
            ):
                provenance["local_ejs_archive_sha256"] = archive_digest
    if impersonation_target is not None:
        curl_cffi_version = package_version("curl-cffi")
        if curl_cffi_version is not None:
            provenance["curl_cffi_package_version"] = curl_cffi_version
    return provenance


def _resolve_executable(command: str, *, purpose: str) -> str:
    executable = shutil.which(command)
    if not executable:
        raise RuntimeError(f"TeachObs {purpose} requires `{command}` on PATH")
    return executable


def preflight_teachobs_media_tools(
    *,
    yt_dlp_command: Sequence[str] | str | None = None,
    ffmpeg_command: str = "ffmpeg",
    ffprobe_command: str = "ffprobe",
    clip_model: str | Path | None = None,
    clip_device: str = "cpu",
) -> dict[str, Any]:
    """Return a side-effect-free readiness report for a dry run."""

    report: dict[str, Any] = {}
    try:
        prefix = _command_prefix(yt_dlp_command)
        report["yt_dlp"] = {"ready": True, "command_prefix": prefix}
    except (RuntimeError, ValueError) as exc:
        report["yt_dlp"] = {"ready": False, "reason": str(exc)}
    for key, command in (("ffmpeg", ffmpeg_command), ("ffprobe", ffprobe_command)):
        executable = shutil.which(command)
        report[key] = (
            {"ready": True, "executable": executable}
            if executable
            else {"ready": False, "reason": f"`{command}` was not found on PATH"}
        )
    report["numpy"] = {
        "ready": importlib.util.find_spec("numpy") is not None,
        "purpose": "deterministic per-scene PCM statistics",
    }
    report["clip"] = clip_preflight(clip_model, device=clip_device)
    report["ready_without_clip"] = all(
        report[key]["ready"] for key in ("yt_dlp", "ffmpeg", "ffprobe", "numpy")
    )
    report["ready_with_requested_clip"] = report["ready_without_clip"] and (
        clip_model is None or report["clip"]["ready"]
    )
    return report


def _fchmod_private_path(path: Path, *, directory: bool) -> None:
    """Open without following symlinks, verify type, then chmod by descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return  # yt-dlp may atomically rename a temporary between scan and open
    try:
        metadata = os.fstat(descriptor)
        expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(
            metadata.st_mode
        )
        if not expected:
            raise ValueError(f"unsafe non-regular yt-dlp staging entry: {path.name}")
        os.fchmod(descriptor, 0o700 if directory else 0o600)
    finally:
        os.close(descriptor)


def _harden_private_staging_tree(staging_directory: Path) -> None:
    """Recursively enforce 0700/0600 without following any symlink."""

    root = staging_directory.resolve()
    if staging_directory.is_symlink() or not root.is_dir():
        raise ValueError("TeachObs yt-dlp staging root is missing or unsafe")
    _fchmod_private_path(root, directory=True)
    for current, directory_names, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        if current_path.is_symlink() or not current_path.resolve().is_relative_to(root):
            raise ValueError("TeachObs yt-dlp staging traversal escaped its root")
        _fchmod_private_path(current_path, directory=True)
        for name in list(directory_names):
            path = current_path / name
            if path.is_symlink():
                raise ValueError("TeachObs yt-dlp staging contains a symlink")
            _fchmod_private_path(path, directory=True)
        for name in filenames:
            path = current_path / name
            if path.is_symlink():
                raise ValueError("TeachObs yt-dlp staging contains a symlink")
            _fchmod_private_path(path, directory=False)


def _precreate_private_resume_target(staging_directory: Path) -> None:
    """Precreate yt-dlp's primary configured .part path without truncation."""

    path = staging_directory / "media.mp4.part"
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("TeachObs yt-dlp resume target is not a regular file")
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _private_child_command(command: Sequence[str]) -> list[str]:
    if os.name != "posix":
        return list(command)
    # A shell only establishes the child-local umask; "$@" preserves every
    # argument literally and avoids shell interpolation of URLs or paths.
    return [
        "/bin/sh",
        "-c",
        'umask 077; exec "$@"',
        "tsm-private-yt-dlp",
        *command,
    ]


def _run_private_ytdlp(
    command: list[str],
    *,
    staging_directory: Path,
    runner: Callable[..., Any],
    timeout_seconds: int,
) -> Any:
    """Run yt-dlp with child umask plus continuous no-symlink hardening."""

    _harden_private_staging_tree(staging_directory)
    _precreate_private_resume_target(staging_directory)
    _harden_private_staging_tree(staging_directory)
    stop = threading.Event()
    watcher_errors: list[BaseException] = []

    def watch() -> None:
        while not stop.wait(0.02):
            try:
                _harden_private_staging_tree(staging_directory)
            except BaseException as exc:  # surfaced after the child is reaped
                watcher_errors.append(exc)
                stop.set()
                return

    watcher = threading.Thread(
        target=watch,
        name="teachobs-private-staging-permissions",
        daemon=True,
    )
    watcher.start()
    result: Any = None
    runner_error: BaseException | None = None
    try:
        result = runner(
            _private_child_command(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except BaseException as exc:
        runner_error = exc
    finally:
        stop.set()
        watcher.join(timeout=2.0)
        try:
            _harden_private_staging_tree(staging_directory)
        except BaseException as exc:
            watcher_errors.append(exc)
    if watcher.is_alive():
        raise RuntimeError("TeachObs private staging permission watcher did not stop")
    if watcher_errors:
        raise RuntimeError(
            "TeachObs private staging permission enforcement failed: "
            f"{type(watcher_errors[0]).__name__}"
        ) from watcher_errors[0]
    if runner_error is not None:
        raise runner_error
    return result


def _download_with_ytdlp(
    source_url: str,
    staging_directory: Path,
    *,
    yt_dlp_command: Sequence[str] | str | None,
    js_runtime: str | None,
    timeout_seconds: int,
    runner: Callable[..., Any],
    cookies_from_browser: str | None = None,
    yt_dlp_direct: bool = False,
    yt_dlp_impersonate: str | None = None,
) -> dict[str, Any]:
    if timeout_seconds < 1:
        raise ValueError("yt-dlp timeout must be positive")
    cookie_spec, browser_family = validate_teachobs_cookies_from_browser(
        cookies_from_browser
    )
    direct, impersonation_target = validate_teachobs_ytdlp_transport(
        direct=yt_dlp_direct,
        impersonate=yt_dlp_impersonate,
    )
    parsed_source = urlsplit(str(source_url))
    source_host = (parsed_source.hostname or "").casefold().rstrip(".")
    youtube_source = source_host in {"youtube.com", "www.youtube.com", "youtu.be"}
    url = (
        _validate_youtube_url(source_url)
        if youtube_source
        else _validate_override_https_url(source_url)
    )
    prefix = _command_prefix(yt_dlp_command)
    ensure_private_directory(staging_directory)
    command = [
        *prefix,
        "--no-playlist",
        "--continue",
        "--part",
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--no-write-info-json",
        "--no-write-playlist-metafiles",
        "--no-write-thumbnail",
        "--no-write-subs",
        "--no-write-auto-subs",
        "--no-write-comments",
        "--no-cache-dir",
        "--no-remote-components",
    ]
    if youtube_source:
        # The anonymous Android VR client is deliberately retained for the
        # normal public-media path.  It does not support browser cookies, so an
        # explicitly credentialed recovery must use cookie-capable web clients
        # instead.  This switch is local to the opt-in credentialed path and
        # never exports cookie material.
        youtube_player_client = (
            "default,web_safari" if cookie_spec is not None else "android_vr"
        )
        command.extend(
            [
                "--extractor-args",
                f"youtube:player_client={youtube_player_client}",
                "--format",
                "134+140/18/160+139",
            ]
        )
        format_policy = "youtube_134+140_then_18_then_160+139_fallback"
    else:
        youtube_player_client = None
        command.extend(
            [
                "--format",
                (
                    "bestvideo[height<=480]+bestaudio/"
                    "best[height<=480]/bestvideo+bestaudio/best"
                ),
            ]
        )
        format_policy = (
            "explicit_mirror_best_height_lte_480_then_separate_av_then_best"
        )
    command.extend(
        [
        "--merge-output-format",
        "mp4",
        "--remux-video",
        "mp4",
        "--paths",
        str(staging_directory),
        "--output",
        "media.%(ext)s",
        "--print",
        "after_move:filepath",
        ]
    )
    if js_runtime:
        command.extend(["--js-runtimes", js_runtime])
    if cookie_spec is not None:
        # The browser store is read only by the yt-dlp child.  We never pass
        # --cookies, export a Netscape jar, or persist the profile selector.
        command.extend(["--cookies-from-browser", cookie_spec])
    if direct:
        # An explicit empty proxy tells yt-dlp not to inherit HTTP(S)_PROXY.
        # No proxy URL is accepted by this interface or written to a receipt.
        command.extend(["--proxy", ""])
    if impersonation_target is not None:
        command.extend(["--impersonate", impersonation_target])
    command.append(url)
    local_transport_provenance = _local_ytdlp_transport_provenance(
        prefix,
        js_runtime=js_runtime,
        impersonation_target=impersonation_target,
    )
    timed_out = False
    try:
        result = _run_private_ytdlp(
            command,
            staging_directory=staging_directory,
            runner=runner,
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        # Raise only after leaving the exception handler.  TimeoutExpired keeps
        # the complete child command, which can contain a private browser
        # profile selector; raising here would retain it as __context__ even
        # when the user-facing message is sanitized.
        timed_out = True
        result = None
    if timed_out:
        raise RuntimeError(
            "yt-dlp timed out; its private partial files were preserved"
        )
    if result.returncode != 0:
        if cookie_spec is not None:
            # yt-dlp diagnostics may include a browser profile or account hint.
            # Keep those diagnostics out of persistent failure summaries.
            raise RuntimeError(
                "yt-dlp failed while browser credentials were enabled; "
                "private partial files were preserved and raw diagnostics were "
                "not retained"
            )
        detail = (result.stderr or result.stdout or "").strip()[-1600:]
        raise RuntimeError(f"yt-dlp failed; private partial files were preserved: {detail}")
    candidates = [
        path
        for path in staging_directory.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in _MEDIA_SUFFIXES
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "yt-dlp did not produce exactly one private MP4 after remuxing"
        )
    media_path = candidates[0]
    ensure_private_file(media_path)
    if media_path.stat().st_size <= 0:
        raise RuntimeError("yt-dlp produced an empty media file")
    return {
        "download_path": str(media_path),
        "transport": "yt-dlp_https_source",
        "yt_dlp_command_prefix": prefix,
        "format_policy": format_policy,
        "youtube_player_client": youtube_player_client,
        "retrieval_host_class": "canonical_youtube" if youtube_source else "explicit_override",
        "remote_ejs_allowed": False,
        "partial_resume_enabled": True,
        "child_process_umask": "0077" if os.name == "posix" else None,
        "staging_permission_watcher_enabled": True,
        "staging_directory_mode": "0700" if os.name == "posix" else None,
        "staging_regular_file_mode": "0600" if os.name == "posix" else None,
        "symlink_following_allowed_in_staging_hardener": False,
        **local_transport_provenance,
        **(
            {
                "credentials_used": True,
                "browser_family": browser_family,
                "browser_profile_recorded": False,
                "cookie_file_export_requested": False,
                "cookie_material_recorded_in_receipt": False,
            }
            if cookie_spec is not None
            else {}
        ),
        **(
            {
                "transport_mode": "direct_environment_proxy_bypass",
                "proxy_url_recorded": False,
            }
            if direct
            else {}
        ),
        **(
            {
                "http_impersonation_requested": True,
                "http_impersonation_target": impersonation_target,
                "http_impersonation_backend": "yt_dlp_curl_cffi",
            }
            if impersonation_target is not None
            else {}
        ),
    }


def _binding_sidecar_path(videos_directory: Path, lesson_id: str) -> Path:
    return videos_directory / f"{lesson_id}.media-binding.json"


_PERSISTED_DOWNLOAD_RECEIPT_FIELDS = frozenset(
    {
        "transport",
        "yt_dlp_command_prefix",
        "format_policy",
        "youtube_player_client",
        "retrieval_host_class",
        "remote_ejs_allowed",
        "partial_resume_enabled",
        "child_process_umask",
        "staging_permission_watcher_enabled",
        "staging_directory_mode",
        "staging_regular_file_mode",
        "symlink_following_allowed_in_staging_hardener",
        "yt_dlp_package_version",
        "javascript_runtime_family",
        "local_ejs_package_version",
        "local_ejs_archive_sha256",
        "curl_cffi_package_version",
    }
)

_BINDING_DOWNLOAD_POLICY_FIELDS = frozenset(
    {
        "transport",
        "format_policy",
        "youtube_player_client",
        "retrieval_host_class",
        "remote_ejs_allowed",
        "partial_resume_enabled",
        "yt_dlp_package_version",
        "javascript_runtime_family",
        "local_ejs_package_version",
        "local_ejs_archive_sha256",
        "curl_cffi_package_version",
    }
)


def _validated_binding_download_policy(value: Any) -> dict[str, Any]:
    """Keep a small, non-secret download-policy subset across media reuse."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("TeachObs media binding download policy must be an object")
    if set(value) - _BINDING_DOWNLOAD_POLICY_FIELDS:
        raise ValueError("TeachObs media binding download policy has unknown fields")
    policy: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"remote_ejs_allowed", "partial_resume_enabled"}:
            if not isinstance(item, bool):
                raise ValueError(
                    f"TeachObs media binding download policy {key} must be boolean"
                )
        elif item is not None and (
            not isinstance(item, str)
            or not item
            or len(item) > 200
            or any(ord(character) < 32 or ord(character) == 127 for character in item)
        ):
            raise ValueError(
                f"TeachObs media binding download policy {key} is invalid"
            )
        policy[key] = item
    if policy.get("remote_ejs_allowed") not in {None, False}:
        raise ValueError("TeachObs media binding cannot enable remote EJS")
    if policy.get("youtube_player_client") not in {
        None,
        "android_vr",
        "default,web_safari",
    }:
        raise ValueError("TeachObs media binding has an unsupported YouTube client")
    return policy


def _sanitize_download_receipt_for_manifest(
    receipt: dict[str, Any],
    *,
    credentials_used: bool,
    browser_family: str | None,
    direct: bool = False,
    impersonation_target: str | None = None,
) -> dict[str, Any]:
    """Persist an allowlisted receipt without cookie/profile/account material."""

    sanitized = {
        key: receipt[key]
        for key in _PERSISTED_DOWNLOAD_RECEIPT_FIELDS
        if key in receipt
    }
    if credentials_used:
        if browser_family not in _COOKIE_BROWSER_FAMILIES:
            raise ValueError("credentialed download lacks a safe browser family")
        sanitized.update(
            {
                "credentials_used": True,
                "browser_family": browser_family,
                "browser_profile_recorded": False,
                "cookie_file_export_requested": False,
                "cookie_material_recorded_in_receipt": False,
            }
        )
    direct, impersonation_target = validate_teachobs_ytdlp_transport(
        direct=direct,
        impersonate=impersonation_target,
    )
    if direct:
        sanitized.update(
            {
                "transport_mode": "direct_environment_proxy_bypass",
                "proxy_url_recorded": False,
            }
        )
    if impersonation_target is not None:
        sanitized.update(
            {
                "http_impersonation_requested": True,
                "http_impersonation_target": impersonation_target,
                "http_impersonation_backend": "yt_dlp_curl_cffi",
            }
        )
    return sanitized


def _resolved_source_binding(
    item: dict[str, Any],
    *,
    override_context: dict[str, Any],
    override_terms_acknowledged: bool,
) -> dict[str, Any]:
    canonical_url = item["source_url"]
    canonical_digest = _sha256_bytes(canonical_url.encode("utf-8"))
    override = override_context["overrides_by_lesson"].get(item["lesson_id"])
    if override is None:
        return {
            "canonical_source_url": canonical_url,
            "canonical_source_url_sha256": canonical_digest,
            "retrieval_source_url": canonical_url,
            "retrieval_source_url_sha256": canonical_digest,
            "retrieval_source_kind": "canonical_repository_source",
            "source_override_manifest_sha256": None,
            "source_override_evidence_metadata_sha256": None,
            "source_override_reason": None,
            "source_override_evidence_metadata": None,
            "candidate_same_content_mirror": False,
            "publisher_byte_identity_established": False,
            "override_source_terms_acknowledged": False,
        }
    return {
        "canonical_source_url": canonical_url,
        "canonical_source_url_sha256": canonical_digest,
        "retrieval_source_url": override["override_source_url"],
        "retrieval_source_url_sha256": override["override_source_url_sha256"],
        "retrieval_source_kind": "explicit_candidate_same_content_mirror",
        "source_override_manifest_sha256": override_context[
            "manifest_file_sha256"
        ],
        "source_override_evidence_metadata_sha256": override[
            "evidence_metadata_sha256"
        ],
        "source_override_reason": override["reason"],
        "source_override_evidence_metadata": override["evidence_metadata"],
        "candidate_same_content_mirror": True,
        "publisher_byte_identity_established": False,
        "override_source_terms_acknowledged": override_terms_acknowledged,
    }


def _validate_media_binding(
    binding: dict[str, Any],
    *,
    item: dict[str, Any],
    source_binding: dict[str, Any],
    media_path: Path,
) -> dict[str, Any]:
    expected_url_digest = source_binding["canonical_source_url_sha256"]
    if (
        binding.get("lesson_id") != item["lesson_id"]
        or binding.get("source_url_sha256") != expected_url_digest
        or binding.get("scene_manifest_sha256") != item["scene_manifest_sha256"]
    ):
        raise ValueError(f"existing media binding differs for {item['lesson_id']}")
    mirror = source_binding["candidate_same_content_mirror"]
    if mirror:
        required = {
            "canonical_source_url_sha256": source_binding[
                "canonical_source_url_sha256"
            ],
            "retrieval_source_url_sha256": source_binding[
                "retrieval_source_url_sha256"
            ],
            "retrieval_source_kind": source_binding["retrieval_source_kind"],
            "source_override_manifest_sha256": source_binding[
                "source_override_manifest_sha256"
            ],
            "source_override_evidence_metadata_sha256": source_binding[
                "source_override_evidence_metadata_sha256"
            ],
            "candidate_same_content_mirror": True,
            "publisher_byte_identity_established": False,
            "override_source_terms_acknowledged": True,
        }
        if any(binding.get(key) != expected for key, expected in required.items()):
            raise ValueError(
                f"existing mirror media binding differs for {item['lesson_id']}"
            )
    else:
        # v1 canonical sidecars remain reusable.  If newer source fields are
        # present, however, they must still agree with the canonical plan.
        optional_expected = {
            "canonical_source_url_sha256": expected_url_digest,
            "retrieval_source_url_sha256": expected_url_digest,
            "candidate_same_content_mirror": False,
            "publisher_byte_identity_established": False,
        }
        for key, expected in optional_expected.items():
            if key in binding and binding[key] != expected:
                raise ValueError(
                    f"existing canonical media binding differs for {item['lesson_id']}"
                )
    if binding.get("credentials_used") is True:
        if (
            binding.get("browser_family") not in _COOKIE_BROWSER_FAMILIES
            or binding.get("browser_profile_recorded") is not False
            or binding.get("cookie_file_export_requested") is not False
            or binding.get("cookie_material_recorded") is not False
        ):
            raise ValueError(
                f"existing credential provenance differs for {item['lesson_id']}"
            )
    elif any(
        key in binding
        for key in (
            "browser_family",
            "browser_profile_recorded",
            "cookie_file_export_requested",
            "cookie_material_recorded",
        )
    ):
        raise ValueError(
            f"existing credential provenance is incomplete for {item['lesson_id']}"
        )
    if "transport_mode" in binding and (
        binding.get("transport_mode") != "direct_environment_proxy_bypass"
        or binding.get("proxy_url_recorded") is not False
    ):
        raise ValueError(
            f"existing transport provenance differs for {item['lesson_id']}"
        )
    if binding.get("http_impersonation_requested") is True:
        if (
            binding.get("http_impersonation_target")
            not in _YTDLP_IMPERSONATION_TARGETS
            or binding.get("http_impersonation_backend") != "yt_dlp_curl_cffi"
        ):
            raise ValueError(
                f"existing HTTP impersonation provenance differs for {item['lesson_id']}"
            )
    elif any(
        key in binding
        for key in (
            "http_impersonation_target",
            "http_impersonation_backend",
        )
    ):
        raise ValueError(
            f"existing HTTP impersonation provenance is incomplete for {item['lesson_id']}"
        )
    actual_digest = file_sha256(media_path)
    if binding.get("media_sha256") != actual_digest:
        raise ValueError(f"existing TeachObs media hash differs for {item['lesson_id']}")
    return binding


def _load_previous_media_records(
    manifest_path: Path,
    *,
    plan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("existing TeachObs media manifest target is unsafe")
    value = read_json(manifest_path)
    if not isinstance(value, dict) or value.get("schema") != MEDIA_SCHEMA:
        raise ValueError("existing TeachObs media manifest has an unsupported schema")
    provenance = plan["repository_provenance"]
    if (
        value.get("repository_commit") != provenance["repository_commit"]
        or value.get("repository_tree_sha256")
        != provenance["repository_tree_sha256"]
    ):
        raise ValueError("existing TeachObs media manifest is bound to another source")
    records = value.get("lessons")
    if not isinstance(records, list):
        raise ValueError("existing TeachObs media manifest lessons must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("existing TeachObs media manifest has an invalid record")
        lesson_id = _safe_lesson_id(record.get("lesson_id"))
        if lesson_id in by_id:
            raise ValueError("existing TeachObs media manifest has duplicate lessons")
        by_id[lesson_id] = record
    return by_id


def _media_manifest_value(
    *,
    plan: dict[str, Any],
    records: dict[str, dict[str, Any]],
    selected_ids: set[str],
    generated_at_utc: str,
    override_context: dict[str, Any],
    override_terms_acknowledged: bool,
) -> dict[str, Any]:
    provenance = plan["repository_provenance"]
    selected_complete = selected_ids.issubset(records)
    mirror_record_count = sum(
        record.get("candidate_same_content_mirror") is True
        for record in records.values()
    )
    credentialed_records = [
        record
        for record in records.values()
        if isinstance(record.get("download_receipt"), dict)
        and record["download_receipt"].get("credentials_used") is True
    ]
    browser_families = sorted(
        {
            record["download_receipt"].get("browser_family")
            for record in credentialed_records
            if record["download_receipt"].get("browser_family")
            in _COOKIE_BROWSER_FAMILIES
        }
    )
    direct_transport_records = [
        record
        for record in records.values()
        if isinstance(record.get("download_receipt"), dict)
        and record["download_receipt"].get("transport_mode")
        == "direct_environment_proxy_bypass"
    ]
    impersonated_records = [
        record
        for record in records.values()
        if isinstance(record.get("download_receipt"), dict)
        and record["download_receipt"].get("http_impersonation_requested")
        is True
    ]
    return {
        "schema": MEDIA_SCHEMA,
        "dataset_id": DATASET_ID,
        "private_artifact": True,
        "public_release_authorized": False,
        "generated_at_utc": generated_at_utc,
        "plan_sha256": plan["plan_sha256"],
        "repository_commit": provenance["repository_commit"],
        "repository_tree_sha256": provenance["repository_tree_sha256"],
        "active_source_override_manifest_sha256": override_context[
            "manifest_file_sha256"
        ],
        "active_source_override_count": len(
            override_context["overrides_by_lesson"]
        ),
        "override_source_terms_acknowledged": bool(
            override_context["present"] and override_terms_acknowledged
        ),
        "selected_lesson_count": len(selected_ids),
        "selected_complete": selected_complete,
        "downloaded_lesson_count_total": len(records),
        **(
            {
                "browser_cookie_credentials": {
                    "credentials_used": True,
                    "credentialed_lesson_count": len(credentialed_records),
                    "browser_families": browser_families,
                    "browser_profile_recorded": False,
                    "cookie_file_export_requested": False,
                    "cookie_material_recorded": False,
                }
            }
            if credentialed_records
            else {}
        ),
        **(
            {
                "download_transport": {
                    "direct_proxy_bypass_lesson_count": len(
                        direct_transport_records
                    ),
                    "http_impersonation_lesson_count": len(
                        impersonated_records
                    ),
                    "http_impersonation_targets": sorted(
                        {
                            record["download_receipt"][
                                "http_impersonation_target"
                            ]
                            for record in impersonated_records
                        }
                    ),
                    "http_impersonation_backends": sorted(
                        {
                            record["download_receipt"][
                                "http_impersonation_backend"
                            ]
                            for record in impersonated_records
                        }
                    ),
                    "proxy_url_recorded": False,
                }
            }
            if direct_transport_records or impersonated_records
            else {}
        ),
        "lessons": sorted(
            records.values(), key=lambda record: _lesson_sort_key(record["lesson_id"])
        ),
        "privacy": {
            "contains_source_urls_and_lesson_ids": True,
            "contains_private_retrieval_mirror_urls": mirror_record_count > 0,
            "contains_media": False,
            "references_private_media_files": True,
            "safe_to_publish": False,
        },
        "claim_boundary": {
            "local_media_hashes_computed": True,
            "publisher_media_hashes_pinned": False,
            "publisher_authenticity_established_by_local_hash": False,
            "publisher_byte_identity_established": False,
            "candidate_same_content_mirror_count": mirror_record_count,
            "candidate_same_content_mirror_used": mirror_record_count > 0,
            "redistribution_authorized": False,
        },
    }


def _validate_media_scene_tail(
    item: dict[str, Any],
    *,
    media_duration_seconds: float,
) -> dict[str, Any]:
    """Allow only a bounded final-scene midpoint clamp near media EOF."""

    if not math.isfinite(media_duration_seconds) or media_duration_seconds <= 0:
        raise ValueError("TeachObs media duration must be finite and positive")
    safe_tail_timestamp = max(0.0, media_duration_seconds - TAIL_FRAME_MARGIN_SECONDS)
    requested = [float(scene["midpoint"]) for scene in item["scenes"]]
    clamped_indices = [
        index for index, timestamp in enumerate(requested) if timestamp > safe_tail_timestamp
    ]
    if not clamped_indices:
        return {
            "tail_frame_clamp_required": False,
            "media_duration_seconds": round(media_duration_seconds, 6),
            "safe_tail_timestamp_seconds": round(safe_tail_timestamp, 6),
            "tail_frame_clamp_delta_seconds": 0.0,
            "maximum_tail_clamp_delta_seconds": MAXIMUM_TAIL_CLAMP_DELTA_SECONDS,
        }
    final_index = len(item["scenes"]) - 1
    if clamped_indices != [final_index]:
        raise ValueError(
            f"TeachObs media would require a non-tail scene clamp: {item['lesson_id']}"
        )
    final_scene = item["scenes"][final_index]
    clamp_delta = requested[final_index] - safe_tail_timestamp
    if (
        clamp_delta <= 0
        or clamp_delta > MAXIMUM_TAIL_CLAMP_DELTA_SECONDS
        or safe_tail_timestamp < float(final_scene["start"])
        or safe_tail_timestamp >= float(final_scene["end"])
    ):
        raise ValueError(
            f"TeachObs final-scene clamp exceeds the safe tail boundary: {item['lesson_id']}"
        )
    return {
        "tail_frame_clamp_required": True,
        "media_duration_seconds": round(media_duration_seconds, 6),
        "safe_tail_timestamp_seconds": round(safe_tail_timestamp, 6),
        "tail_frame_clamp_delta_seconds": round(clamp_delta, 6),
        "maximum_tail_clamp_delta_seconds": MAXIMUM_TAIL_CLAMP_DELTA_SECONDS,
    }


def _private_creation_umask(function: Callable[..., Any]) -> Callable[..., Any]:
    """Keep downloader-created media and partials private from first creation."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if os.name != "posix":
            return function(*args, **kwargs)
        previous_umask = os.umask(0o077)
        try:
            return function(*args, **kwargs)
        finally:
            os.umask(previous_umask)

    return wrapped


@_private_creation_umask
def download_teachobs_media(
    plan: dict[str, Any],
    output_directory: str | Path,
    *,
    acknowledge_source_terms: bool = False,
    source_override_manifest_path: str | Path | None = None,
    acknowledge_override_source_terms: bool = False,
    dry_run: bool = False,
    yt_dlp_command: Sequence[str] | str | None = None,
    js_runtime: str | None = None,
    cookies_from_browser: str | None = None,
    yt_dlp_direct: bool = False,
    yt_dlp_impersonate: str | None = None,
    ffprobe_command: str = "ffprobe",
    download_timeout_seconds: int = 14_400,
    ffprobe_timeout_seconds: int = 180,
    absolute_duration_tolerance_seconds: float = 20.0,
    relative_duration_tolerance: float = 0.01,
    downloader: Callable[..., dict[str, Any]] = _download_with_ytdlp,
    media_probe: Callable[..., dict[str, Any]] = probe_local_media,
    runner: Callable[..., Any] = subprocess.run,
    jobs: int = 1,
    fail_on_incomplete: bool = True,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Download, hash, probe, and incrementally receipt selected source videos."""

    cookie_spec, browser_family = validate_teachobs_cookies_from_browser(
        cookies_from_browser
    )
    direct, impersonation_target = validate_teachobs_ytdlp_transport(
        direct=yt_dlp_direct,
        impersonate=yt_dlp_impersonate,
    )
    lessons = _validate_plan(plan)
    override_context = load_teachobs_source_override_manifest(
        source_override_manifest_path,
        plan=plan,
    )
    if dry_run:
        return {
            "mode": "dry_run",
            "output_created": False,
            "source_terms_acknowledged": acknowledge_source_terms,
            "source_override_manifest_present": override_context["present"],
            "source_override_count": len(
                override_context["overrides_by_lesson"]
            ),
            "override_source_terms_acknowledged": (
                acknowledge_override_source_terms
            ),
            "override_acknowledgement_required_for_execution": override_context[
                "present"
            ],
            **(
                {
                    "browser_cookie_credentials_enabled": True,
                    "browser_family": browser_family,
                    "browser_profile_recorded": False,
                }
                if cookie_spec is not None
                else {}
            ),
            **(
                {
                    "yt_dlp_direct_enabled": direct,
                    "yt_dlp_impersonation_target": impersonation_target,
                    "proxy_url_recorded": False,
                }
                if direct or impersonation_target is not None
                else {}
            ),
            "acknowledgement_required_for_execution": True,
            "plan_sha256": plan["plan_sha256"],
            "lesson_count": len(lessons),
            "scene_count": sum(item["scene_count"] for item in lessons),
        }
    if not acknowledge_source_terms:
        raise ValueError(
            "TeachObs source-media download requires "
            "acknowledge_source_terms=True; source videos retain their original "
            "rights and platform terms"
        )
    if override_context["present"] and not acknowledge_override_source_terms:
        raise ValueError(
            "TeachObs mirror retrieval requires the separate "
            "acknowledge_override_source_terms=True gate in addition to "
            "acknowledge_source_terms=True"
        )
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 32:
        raise ValueError("TeachObs download jobs must be an integer from 1 to 32")
    generated = generated_at_utc or _utc_now()
    output = ensure_private_directory(output_directory).resolve()
    videos_directory = ensure_private_directory(output / "videos")
    staging_root = ensure_private_directory(output / ".yt-dlp-staging")
    manifest_path = output / "media_manifest.json"
    records = _load_previous_media_records(manifest_path, plan=plan)
    selected_ids = {item["lesson_id"] for item in lessons}
    # Receipt the empty/resumed state before workers start so an all-failure
    # attempt still has a valid private manifest alongside retained partials.
    write_json(
        manifest_path,
        _media_manifest_value(
            plan=plan,
            records=records,
            selected_ids=selected_ids,
            generated_at_utc=generated,
            override_context=override_context,
            override_terms_acknowledged=acknowledge_override_source_terms,
        ),
    )

    def process_item(item: dict[str, Any]) -> dict[str, Any]:
        lesson_id = item["lesson_id"]
        source_binding = _resolved_source_binding(
            item,
            override_context=override_context,
            override_terms_acknowledged=acknowledge_override_source_terms,
        )
        final_path = videos_directory / f"{lesson_id}.mp4"
        sidecar_path = _binding_sidecar_path(videos_directory, lesson_id)
        if final_path.is_symlink() or (final_path.exists() and not final_path.is_file()):
            raise ValueError(f"unsafe TeachObs final media target for {lesson_id}")
        if final_path.exists():
            ensure_private_file(final_path)
            if sidecar_path.is_symlink() or not sidecar_path.is_file():
                raise ValueError(
                    f"existing TeachObs media lacks its recovery binding: {lesson_id}"
                )
            binding = _validate_media_binding(
                read_json(sidecar_path),
                item=item,
                source_binding=source_binding,
                media_path=final_path,
            )
            probe_summary = binding.get("media_probe")
            if not isinstance(probe_summary, dict):
                raise ValueError(f"existing TeachObs media probe is missing: {lesson_id}")
            tail_frame_timing = _validate_media_scene_tail(
                item,
                media_duration_seconds=float(probe_summary["duration_seconds"]),
            )
            status = "reused_hash_bound_private_media"
            media_digest = str(binding["media_sha256"])
            file_size = final_path.stat().st_size
            prior_credentials_used = binding.get("credentials_used") is True
            prior_browser_family = binding.get("browser_family")
            prior_transport_mode = binding.get("transport_mode")
            prior_impersonation_target = binding.get(
                "http_impersonation_target"
            )
            prior_download_policy = _validated_binding_download_policy(
                binding.get("download_policy")
            )
            download_receipt = _sanitize_download_receipt_for_manifest(
                {
                    **prior_download_policy,
                    "partial_resume_enabled": True,
                },
                credentials_used=prior_credentials_used,
                browser_family=(
                    str(prior_browser_family)
                    if prior_credentials_used
                    else None
                ),
                direct=prior_transport_mode
                == "direct_environment_proxy_bypass",
                impersonation_target=(
                    str(prior_impersonation_target)
                    if prior_impersonation_target is not None
                    else None
                ),
            )
            download_receipt["reused"] = True
        else:
            staging_directory = ensure_private_directory(staging_root / lesson_id)
            downloader_options: dict[str, Any] = {
                "yt_dlp_command": yt_dlp_command,
                "js_runtime": js_runtime,
                "timeout_seconds": download_timeout_seconds,
                "runner": runner,
            }
            if cookie_spec is not None:
                downloader_options["cookies_from_browser"] = cookie_spec
            if direct:
                downloader_options["yt_dlp_direct"] = True
            if impersonation_target is not None:
                downloader_options["yt_dlp_impersonate"] = impersonation_target
            raw_download_receipt = downloader(
                source_binding["retrieval_source_url"],
                staging_directory,
                **downloader_options,
            )
            if not isinstance(raw_download_receipt, dict):
                raise RuntimeError("TeachObs downloader returned an invalid receipt")
            candidate = Path(
                str(raw_download_receipt.get("download_path", ""))
            ).resolve()
            download_receipt = _sanitize_download_receipt_for_manifest(
                raw_download_receipt,
                credentials_used=cookie_spec is not None,
                browser_family=browser_family,
                direct=direct,
                impersonation_target=impersonation_target,
            )
            try:
                candidate.relative_to(staging_directory.resolve())
            except ValueError as exc:
                raise RuntimeError("TeachObs downloader returned a path outside staging") from exc
            if candidate.is_symlink() or not candidate.is_file():
                raise RuntimeError("TeachObs downloader did not create a safe media file")
            ensure_private_file(candidate)
            file_size = candidate.stat().st_size
            if file_size <= 0:
                raise RuntimeError("TeachObs downloader created an empty media file")
            raw_probe = media_probe(
                candidate,
                ffprobe_command=ffprobe_command,
                timeout_seconds=ffprobe_timeout_seconds,
            )
            probe_summary = validate_media_probe(
                raw_probe,
                reference_duration_seconds=item["reference_duration_seconds"],
                actual_file_size_bytes=file_size,
                absolute_duration_tolerance_seconds=(
                    absolute_duration_tolerance_seconds
                ),
                relative_duration_tolerance=relative_duration_tolerance,
            )
            tail_frame_timing = _validate_media_scene_tail(
                item,
                media_duration_seconds=float(probe_summary["duration_seconds"]),
            )
            media_digest = file_sha256(candidate)
            binding = {
                "schema": "teaching_skill_miner.teachobs_private_media_binding.v2",
                "lesson_id": lesson_id,
                # source_url_sha256 is retained as the canonical-v1 alias.
                "source_url_sha256": source_binding[
                    "canonical_source_url_sha256"
                ],
                "canonical_source_url_sha256": source_binding[
                    "canonical_source_url_sha256"
                ],
                "retrieval_source_url_sha256": source_binding[
                    "retrieval_source_url_sha256"
                ],
                "retrieval_source_kind": source_binding[
                    "retrieval_source_kind"
                ],
                "source_override_manifest_sha256": source_binding[
                    "source_override_manifest_sha256"
                ],
                "source_override_evidence_metadata_sha256": source_binding[
                    "source_override_evidence_metadata_sha256"
                ],
                "candidate_same_content_mirror": source_binding[
                    "candidate_same_content_mirror"
                ],
                "publisher_byte_identity_established": False,
                "canonical_source_terms_acknowledged": True,
                "override_source_terms_acknowledged": source_binding[
                    "override_source_terms_acknowledged"
                ],
                "scene_manifest_sha256": item["scene_manifest_sha256"],
                "media_sha256": media_digest,
                "media_size_bytes": file_size,
                "media_probe": probe_summary,
                "download_policy": _validated_binding_download_policy(
                    {
                        key: value
                        for key, value in download_receipt.items()
                        if key in _BINDING_DOWNLOAD_POLICY_FIELDS
                    }
                ),
                **(
                    {
                        "credentials_used": True,
                        "browser_family": download_receipt["browser_family"],
                        "browser_profile_recorded": False,
                        "cookie_file_export_requested": False,
                        "cookie_material_recorded": False,
                    }
                    if download_receipt.get("credentials_used") is True
                    else {}
                ),
                **(
                    {
                        "transport_mode": download_receipt["transport_mode"],
                        "proxy_url_recorded": False,
                    }
                    if "transport_mode" in download_receipt
                    else {}
                ),
                **(
                    {
                        "http_impersonation_requested": True,
                        "http_impersonation_target": download_receipt[
                            "http_impersonation_target"
                        ],
                        "http_impersonation_backend": download_receipt[
                            "http_impersonation_backend"
                        ],
                    }
                    if download_receipt.get("http_impersonation_requested")
                    is True
                    else {}
                ),
                "private_artifact": True,
                "safe_to_publish": False,
            }
            write_json(sidecar_path, binding)
            with candidate.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(candidate, final_path)
            ensure_private_file(final_path)
            status = "downloaded_hashed_probed_and_atomically_committed"
            # Only the dedicated private staging directory is removed.  Failed
            # downloads keep their .part files so yt-dlp can resume later.
            shutil.rmtree(staging_directory)

        return {
            "lesson_id": lesson_id,
            "split": item["split"],
            # Keep the legacy canonical fields and add explicit retrieval fields.
            "source_url": item["source_url"],
            "source_url_sha256": source_binding["canonical_source_url_sha256"],
            "canonical_source_url": source_binding["canonical_source_url"],
            "canonical_source_url_sha256": source_binding[
                "canonical_source_url_sha256"
            ],
            "retrieval_source_url": source_binding["retrieval_source_url"],
            "retrieval_source_url_sha256": source_binding[
                "retrieval_source_url_sha256"
            ],
            "retrieval_source_kind": source_binding["retrieval_source_kind"],
            "source_override_manifest_sha256": source_binding[
                "source_override_manifest_sha256"
            ],
            "source_override_reason": source_binding["source_override_reason"],
            "source_override_evidence_metadata": source_binding[
                "source_override_evidence_metadata"
            ],
            "source_override_evidence_metadata_sha256": source_binding[
                "source_override_evidence_metadata_sha256"
            ],
            "candidate_same_content_mirror": source_binding[
                "candidate_same_content_mirror"
            ],
            "publisher_byte_identity_established": False,
            "canonical_source_terms_acknowledged": True,
            "override_source_terms_acknowledged": source_binding[
                "override_source_terms_acknowledged"
            ],
            "scene_manifest_sha256": item["scene_manifest_sha256"],
            "scene_count": item["scene_count"],
            "reference_duration_seconds": item["reference_duration_seconds"],
            "media_path": f"videos/{lesson_id}.mp4",
            "media_size_bytes": file_size,
            "media_sha256": media_digest,
            "upstream_media_sha256_pinned": False,
            "media_probe": probe_summary,
            "tail_frame_timing": tail_frame_timing,
            "status": status,
            "download_receipt": {
                key: value
                for key, value in download_receipt.items()
                if key != "download_path"
            },
        }
    if jobs == 1:
        completed_records = (process_item(item) for item in lessons)
        for record in completed_records:
            records[record["lesson_id"]] = record
            manifest = _media_manifest_value(
                plan=plan,
                records=records,
                selected_ids=selected_ids,
                generated_at_utc=generated,
                override_context=override_context,
                override_terms_acknowledged=(
                    acknowledge_override_source_terms
                ),
            )
            write_json(manifest_path, manifest)
    else:
        failures: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="teachobs") as pool:
            futures = {pool.submit(process_item, item): item for item in lessons}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception:  # keep independent downloads auditable
                    failures.append(
                        {
                            "lesson_id": item["lesson_id"],
                            "reason": "download_or_validation_failed",
                        }
                    )
                    continue
                records[record["lesson_id"]] = record
                # Only the coordinator writes the atomic manifest.  Per-video
                # workers write distinct media/binding/staging paths.
                manifest = _media_manifest_value(
                    plan=plan,
                    records=records,
                    selected_ids=selected_ids,
                    generated_at_utc=generated,
                    override_context=override_context,
                    override_terms_acknowledged=(
                        acknowledge_override_source_terms
                    ),
                )
                write_json(manifest_path, manifest)
        failure_path = output / "media_failures.json"
        if failures:
            write_json(
                failure_path,
                {
                    "schema": "teaching_skill_miner.teachobs_private_media_failures.v1",
                    "private_artifact": True,
                    "safe_to_publish": False,
                    "plan_sha256": plan["plan_sha256"],
                    "source_override_manifest_sha256": override_context[
                        "manifest_file_sha256"
                    ],
                    "failure_count": len(failures),
                    "failures": sorted(
                        failures, key=lambda value: _lesson_sort_key(value["lesson_id"])
                    ),
                    "privacy": {
                        "contains_raw_exception_messages": False,
                        "contains_cookie_material": False,
                        "contains_browser_profile": False,
                        "contains_account_identifiers": False,
                        "contains_proxy_url": False,
                    },
                },
            )
            if fail_on_incomplete:
                raise RuntimeError(
                    "TeachObs media download remained incomplete after independent "
                    f"attempts: {len(failures)} lesson(s); successful lessons were "
                    "atomically receipted and failed partial files were preserved"
                )
        else:
            failure_path.unlink(missing_ok=True)
    return {
        "manifest": read_json(manifest_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def _frame_schedule(
    item: dict[str, Any], *, media_duration_seconds: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tail = _validate_media_scene_tail(
        item, media_duration_seconds=media_duration_seconds
    )
    schedule: list[dict[str, Any]] = []
    for index, scene in enumerate(item["scenes"]):
        requested = float(scene["midpoint"])
        clamped = bool(
            tail["tail_frame_clamp_required"]
            and index == len(item["scenes"]) - 1
        )
        actual = (
            float(tail["safe_tail_timestamp_seconds"])
            if clamped
            else requested
        )
        schedule.append(
            {
                "scene_no": scene["scene_no"],
                "requested_timestamp": round(requested, 6),
                "timestamp": round(actual, 6),
                "timestamp_clamped": clamped,
                "timestamp_clamp_delta_seconds": round(requested - actual, 6),
            }
        )
    return schedule, tail


def _showinfo_timestamps(stderr: str) -> list[float]:
    timestamps = [
        float(match.group("timestamp"))
        for match in _SHOWINFO_PTS_TIME_RE.finditer(stderr)
    ]
    if any(not math.isfinite(value) or value < 0 for value in timestamps):
        raise RuntimeError("TeachObs FFmpeg showinfo emitted an invalid timestamp")
    return timestamps


def _explicit_midpoint_select_filter(
    schedule: Sequence[dict[str, Any]],
) -> str:
    """Select the first decoded frame at or after each explicit target."""

    if not schedule:
        raise ValueError("TeachObs midpoint schedule cannot be empty")
    first = float(schedule[0]["timestamp"])
    target_expression = f"{first:.6f}+selected_n*{SCENE_SECONDS:.6f}"
    if schedule[-1]["timestamp_clamped"]:
        final_index = len(schedule) - 1
        final_target = float(schedule[-1]["timestamp"])
        target_expression = (
            f"if(eq(selected_n\\,{final_index})\\,{final_target:.6f}\\,"
            f"{target_expression})"
        )
    return (
        "setpts=PTS-STARTPTS,"
        f"select='gte(t,{target_expression})',"
        "showinfo"
    )


def _verify_frame_task(
    task_path: Path,
    *,
    item: dict[str, Any],
    media_sha256: str,
    media_duration_seconds: float,
    require_timestamp_evidence: bool = True,
) -> dict[str, Any]:
    if task_path.is_symlink() or not task_path.is_file():
        raise ValueError(f"TeachObs frame task is missing for {item['lesson_id']}")
    task = read_json(task_path)
    if (
        not isinstance(task, dict)
        or task.get("schema") != FRAME_TASK_SCHEMA
        or task.get("video_id") != item["lesson_id"]
        or task.get("media_sha256") != media_sha256
        or task.get("scene_manifest_sha256") != item["scene_manifest_sha256"]
        or task.get("frame_count") != item["scene_count"]
    ):
        raise ValueError(f"TeachObs frame task binding differs for {item['lesson_id']}")
    frames = task.get("frames")
    if not isinstance(frames, list) or len(frames) != item["scene_count"]:
        raise ValueError(f"TeachObs frame task count differs for {item['lesson_id']}")
    schedule, tail = _frame_schedule(
        item, media_duration_seconds=media_duration_seconds
    )
    declared_duration = task.get("media_duration_seconds")
    if declared_duration is not None and not math.isclose(
        float(declared_duration), media_duration_seconds, abs_tol=1e-6
    ):
        raise ValueError(
            f"TeachObs frame task media duration differs for {item['lesson_id']}"
        )
    if tail["tail_frame_clamp_required"] and (
        declared_duration is None
        or task.get("tail_clamp_margin_seconds") != TAIL_FRAME_MARGIN_SECONDS
        or task.get("maximum_tail_clamp_delta_seconds")
        != MAXIMUM_TAIL_CLAMP_DELTA_SECONDS
    ):
        raise ValueError(
            f"TeachObs clamped frame task lacks tail policy for {item['lesson_id']}"
        )
    if require_timestamp_evidence and (
        task.get("extraction_method") != FRAME_EXTRACTION_METHOD
        or task.get("timestamp_selection_policy")
        != "first_decoded_frame_at_or_after_target"
        or task.get("timestamp_evidence_source")
        != "ffmpeg_showinfo_pts_time_after_pts_start_normalization"
        or task.get("timestamp_tolerance_seconds")
        != FRAME_TIMESTAMP_TOLERANCE_SECONDS
    ):
        raise ValueError(
            f"TeachObs frame task lacks verified midpoint timing for "
            f"{item['lesson_id']}"
        )
    root = task_path.parent.resolve()
    for scene, expected, frame in zip(item["scenes"], schedule, frames, strict=True):
        if not isinstance(frame, dict):
            raise ValueError("TeachObs frame task contains a non-object row")
        expected_id = f"{item['lesson_id']}_scene_{scene['scene_no']:04d}"
        requested = frame.get("requested_timestamp", frame.get("timestamp"))
        clamped = frame.get("timestamp_clamped", False)
        delta = frame.get("timestamp_clamp_delta_seconds", 0.0)
        if (
            frame.get("frame_id") != expected_id
            or frame.get("scene_no") != scene["scene_no"]
            or not math.isclose(
                float(requested), expected["requested_timestamp"], abs_tol=1e-6
            )
            or not math.isclose(
                float(frame.get("timestamp", -1)),
                expected["timestamp"],
                abs_tol=1e-6,
            )
            or clamped is not expected["timestamp_clamped"]
            or not math.isclose(
                float(delta),
                expected["timestamp_clamp_delta_seconds"],
                abs_tol=1e-6,
            )
        ):
            raise ValueError(f"TeachObs frame identity differs for {expected_id}")
        if require_timestamp_evidence:
            source_timestamp = frame.get("source_frame_timestamp")
            selection_error = frame.get("timestamp_selection_error_seconds")
            try:
                source_timestamp_value = float(source_timestamp)
                selection_error_value = float(selection_error)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"TeachObs frame timing evidence is invalid for {expected_id}"
                ) from exc
            computed_error = source_timestamp_value - expected["timestamp"]
            if (
                not math.isfinite(source_timestamp_value)
                or not math.isfinite(selection_error_value)
                or computed_error < -1e-6
                or computed_error > FRAME_TIMESTAMP_TOLERANCE_SECONDS
                or not math.isclose(
                    selection_error_value, computed_error, abs_tol=1e-6
                )
                or frame.get("timestamp_verified_within_tolerance") is not True
            ):
                raise ValueError(
                    f"TeachObs frame timing evidence differs for {expected_id}"
                )
        relative = PurePosixPath(str(frame.get("path", "")))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("TeachObs frame path is unsafe")
        path = (root / relative.name).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("TeachObs frame path escapes its private root") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"TeachObs frame is missing: {expected_id}")
        if file_sha256(path) != frame.get("sha256"):
            raise ValueError(f"TeachObs frame hash differs: {expected_id}")
    return task


def extract_teachobs_midpoint_frames(
    media_path: str | Path,
    item: dict[str, Any],
    output_root: str | Path,
    *,
    media_sha256: str,
    media_duration_seconds: float,
    ffmpeg_command: str = "ffmpeg",
    timeout_seconds: int = 7_200,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Extract and timestamp-verify every explicit scene midpoint."""

    if timeout_seconds < 1:
        raise ValueError("TeachObs frame-extraction timeout must be positive")
    lesson_id = _safe_lesson_id(item.get("lesson_id"))
    media = Path(media_path)
    if media.is_symlink() or not media.is_file():
        raise ValueError(f"TeachObs private media is missing for {lesson_id}")
    if file_sha256(media) != media_sha256:
        raise ValueError(f"TeachObs private media hash differs for {lesson_id}")
    schedule, tail = _frame_schedule(
        item, media_duration_seconds=media_duration_seconds
    )
    root = ensure_private_directory(output_root)
    final_directory = root / lesson_id
    final_task_path = final_directory / "task.json"
    replace_unverified_cache = False
    if final_directory.exists():
        if final_directory.is_symlink() or not final_directory.is_dir():
            raise ValueError(f"TeachObs frame cache is unsafe for {lesson_id}")
        try:
            task = _verify_frame_task(
                final_task_path,
                item=item,
                media_sha256=media_sha256,
                media_duration_seconds=media_duration_seconds,
            )
        except ValueError:
            # A hash-bound v1/v2 cache is safe to replace, but it cannot prove
            # which decoded source PTS produced each JPEG.  Validate all old
            # bindings and hashes before scheduling an atomic v3 replacement.
            _verify_frame_task(
                final_task_path,
                item=item,
                media_sha256=media_sha256,
                media_duration_seconds=media_duration_seconds,
                require_timestamp_evidence=False,
            )
            replace_unverified_cache = True
        else:
            return {"task": task, "task_path": str(final_task_path), "reused": True}

    executable = _resolve_executable(ffmpeg_command, purpose="midpoint extraction")
    staging = Path(tempfile.mkdtemp(prefix=f".{lesson_id}-", dir=root))
    staging.chmod(0o700)
    pattern = staging / "frame_%04d.jpg"
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "info",
        "-y",
        "-i",
        str(media),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        _explicit_midpoint_select_filter(schedule),
        "-fps_mode",
        "vfr",
        "-frames:v",
        str(item["scene_count"]),
        "-q:v",
        "3",
        str(pattern),
    ]
    committed = False
    try:
        try:
            result = runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"TeachObs midpoint extraction timed out: {lesson_id}"
            ) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-1600:]
            raise RuntimeError(
                f"TeachObs midpoint extraction failed for {lesson_id}: {detail}"
            )
        observed_timestamps = _showinfo_timestamps(result.stderr or "")
        if len(observed_timestamps) < item["scene_count"]:
            raise RuntimeError(
                f"TeachObs midpoint timestamp evidence count mismatch for {lesson_id}: "
                f"{len(observed_timestamps)} < {item['scene_count']}"
            )
        observed_timestamps = observed_timestamps[: item["scene_count"]]
        selection_errors: list[float] = []
        for timing, observed_timestamp in zip(
            schedule, observed_timestamps, strict=True
        ):
            selection_error = observed_timestamp - float(timing["timestamp"])
            if (
                selection_error < -1e-6
                or selection_error > FRAME_TIMESTAMP_TOLERANCE_SECONDS
            ):
                raise RuntimeError(
                    f"TeachObs midpoint timestamp tolerance exceeded for "
                    f"{lesson_id} scene {timing['scene_no']}"
                )
            selection_errors.append(selection_error)
        paths = sorted(staging.glob("frame_*.jpg"))
        if len(paths) != item["scene_count"]:
            raise RuntimeError(
                f"TeachObs midpoint extraction count mismatch for {lesson_id}: "
                f"{len(paths)} != {item['scene_count']}"
            )
        frames: list[dict[str, Any]] = []
        for scene, timing, observed_timestamp, selection_error, path in zip(
            item["scenes"],
            schedule,
            observed_timestamps,
            selection_errors,
            paths,
            strict=True,
        ):
            if path.is_symlink() or path.stat().st_size <= 0:
                raise RuntimeError(f"TeachObs FFmpeg produced an unsafe frame: {path.name}")
            ensure_private_file(path)
            frames.append(
                {
                    "frame_id": f"{lesson_id}_scene_{scene['scene_no']:04d}",
                    "scene_no": scene["scene_no"],
                    "requested_timestamp": timing["requested_timestamp"],
                    "timestamp": timing["timestamp"],
                    "timestamp_clamped": timing["timestamp_clamped"],
                    "timestamp_clamp_delta_seconds": timing[
                        "timestamp_clamp_delta_seconds"
                    ],
                    "source_frame_timestamp": round(observed_timestamp, 6),
                    "timestamp_selection_error_seconds": round(
                        selection_error, 6
                    ),
                    "timestamp_verified_within_tolerance": True,
                    "path": path.name,
                    "sha256": file_sha256(path),
                }
            )
        task = {
            "schema": FRAME_TASK_SCHEMA,
            "video_id": lesson_id,
            "media_sha256": media_sha256,
            "media_duration_seconds": round(media_duration_seconds, 6),
            "scene_manifest_sha256": item["scene_manifest_sha256"],
            "sampling_method": FRAME_EXTRACTION_METHOD,
            "extraction_method": FRAME_EXTRACTION_METHOD,
            "timestamp_selection_policy": (
                "first_decoded_frame_at_or_after_target"
            ),
            "timestamp_evidence_source": (
                "ffmpeg_showinfo_pts_time_after_pts_start_normalization"
            ),
            "timestamp_tolerance_seconds": FRAME_TIMESTAMP_TOLERANCE_SECONDS,
            "maximum_observed_timestamp_error_seconds": round(
                max(selection_errors), 6
            ),
            "showinfo_timestamp_count": len(_showinfo_timestamps(result.stderr or "")),
            "showinfo_timestamps_used": len(observed_timestamps),
            "tail_clamp_margin_seconds": TAIL_FRAME_MARGIN_SECONDS,
            "maximum_tail_clamp_delta_seconds": (
                MAXIMUM_TAIL_CLAMP_DELTA_SECONDS
            ),
            "tail_frame_clamp_applied": tail["tail_frame_clamp_required"],
            "frame_count": len(frames),
            "private_artifact": True,
            "public_release_authorized": False,
            "frames": frames,
        }
        write_json(staging / "task.json", task)
        if replace_unverified_cache:
            backup = Path(
                tempfile.mkdtemp(prefix=f".{lesson_id}-unverified-", dir=root)
            )
            backup.rmdir()
            os.replace(final_directory, backup)
            try:
                os.replace(staging, final_directory)
            except BaseException:
                os.replace(backup, final_directory)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, final_directory)
        committed = True
        ensure_private_directory(final_directory)
        for path in final_directory.iterdir():
            if path.is_file():
                ensure_private_file(path)
        return {
            "task": task,
            "task_path": str(final_task_path),
            "reused": False,
        }
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def _verify_audio_features(
    path: Path,
    *,
    item: dict[str, Any],
    media_sha256: str,
    media_duration_seconds: float,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"TeachObs audio features are missing for {item['lesson_id']}")
    value = read_json(path)
    if (
        not isinstance(value, dict)
        or value.get("schema") != AUDIO_SCHEMA
        or value.get("lesson_id") != item["lesson_id"]
        or value.get("media_sha256") != media_sha256
        or value.get("scene_manifest_sha256") != item["scene_manifest_sha256"]
        or value.get("scene_count") != item["scene_count"]
        or value.get("sample_rate_hz") != AUDIO_SAMPLE_RATE_HZ
    ):
        raise ValueError(f"TeachObs audio-feature binding differs for {item['lesson_id']}")
    rows = value.get("scenes")
    if not isinstance(rows, list) or len(rows) != item["scene_count"]:
        raise ValueError(f"TeachObs audio-feature count differs for {item['lesson_id']}")
    algorithm = value.get("algorithm")
    if algorithm not in {
        "teaching_skill_miner.scene_pcm_statistics.v1",
        "teaching_skill_miner.scene_pcm_statistics.v2",
    }:
        raise ValueError(
            f"TeachObs audio-feature algorithm differs for {item['lesson_id']}"
        )
    strict_tail_audit = algorithm == "teaching_skill_miner.scene_pcm_statistics.v2"
    total_padding_samples = 0
    tail_observed_end_sample: int | None = None
    recorded_tail_alignment_delta: float | None = None
    for index, (scene, row) in enumerate(zip(item["scenes"], rows, strict=True)):
        if not isinstance(row, dict) or row.get("scene_no") != scene["scene_no"]:
            raise ValueError(
                f"TeachObs audio-feature scene alignment differs for "
                f"{item['lesson_id']}"
            )
        expected_samples = int(
            round((float(scene["end"]) - float(scene["start"])) * AUDIO_SAMPLE_RATE_HZ)
        )
        if row.get("start") != scene["start"] or row.get("end") != scene["end"]:
            raise ValueError(
                f"TeachObs audio-feature interval differs for {item['lesson_id']}"
            )
        observed_samples = row.get("observed_sample_count")
        if (
            isinstance(observed_samples, bool)
            or not isinstance(observed_samples, int)
            or not 0 < observed_samples <= expected_samples
        ):
            raise ValueError(
                f"TeachObs audio-feature sample count differs for {item['lesson_id']}"
            )
        expected_coverage = round(observed_samples / expected_samples, 8)
        try:
            observed_coverage = float(row.get("coverage_fraction"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TeachObs audio-feature coverage differs for {item['lesson_id']}"
            ) from exc
        if not math.isclose(
            observed_coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(
                f"TeachObs audio-feature coverage differs for {item['lesson_id']}"
            )
        missing_samples = expected_samples - observed_samples
        if missing_samples and index != len(rows) - 1:
            raise ValueError(
                f"TeachObs cached audio has a non-tail gap for {item['lesson_id']}"
            )
        if not strict_tail_audit:
            # Legacy cached rows are reusable only when they contain complete real
            # audio.  Partial legacy rows did not prove that missing samples were a
            # final-scene EOF condition or that statistics excluded padding.
            if missing_samples:
                raise ValueError(
                    f"TeachObs legacy audio tail is unauditable for "
                    f"{item['lesson_id']}"
                )
            continue
        padding_samples = row.get("padding_sample_count")
        if (
            row.get("requested_sample_count") != expected_samples
            or padding_samples != missing_samples
            or row.get("analysis_window_sample_count") != expected_samples
            or row.get("statistics_sample_count") != observed_samples
            or row.get("statistics_exclude_padding") is not True
            or row.get("padding_applied") is not bool(missing_samples)
            or row.get("real_audio_coverage_fraction") != expected_coverage
            or row.get("coverage_basis")
            != "decoded_real_samples_excluding_zero_padding"
        ):
            raise ValueError(
                f"TeachObs audio tail audit differs for {item['lesson_id']}"
            )
        expected_observed_seconds = round(observed_samples / AUDIO_SAMPLE_RATE_HZ, 8)
        expected_missing_seconds = round(missing_samples / AUDIO_SAMPLE_RATE_HZ, 8)
        if (
            row.get("observed_duration_seconds") != expected_observed_seconds
            or row.get("missing_duration_seconds") != expected_missing_seconds
            or row.get("padding_duration_seconds") != expected_missing_seconds
            or row.get("padding_value_normalized") != 0.0
        ):
            raise ValueError(
                f"TeachObs audio duration audit differs for {item['lesson_id']}"
            )
        if missing_samples:
            alignment_delta = row.get("media_eof_alignment_delta_seconds")
            if (
                isinstance(alignment_delta, bool)
                or not isinstance(alignment_delta, (int, float))
                or not math.isfinite(float(alignment_delta))
                or not 0
                <= float(alignment_delta)
                <= AUDIO_MEDIA_EOF_ALIGNMENT_TOLERANCE_SECONDS
                or row.get("tail_eof_exception_applied") is not True
            ):
                raise ValueError(
                    f"TeachObs audio EOF audit differs for {item['lesson_id']}"
                )
            tail_observed_end_sample = int(
                round(float(scene["start"]) * AUDIO_SAMPLE_RATE_HZ)
            ) + observed_samples
            recorded_tail_alignment_delta = float(alignment_delta)
        elif (
            row.get("media_eof_alignment_delta_seconds") is not None
            or row.get("tail_eof_exception_applied") is not False
        ):
            raise ValueError(
                f"TeachObs audio EOF audit differs for {item['lesson_id']}"
            )
        total_padding_samples += missing_samples
    if strict_tail_audit:
        decoded_sample_count = value.get("decoded_real_sample_count")
        tail_padding_scene_count = value.get("tail_padding_scene_count")
        tail_padding_sample_count = value.get("tail_padding_sample_count")
        if (
            isinstance(decoded_sample_count, bool)
            or not isinstance(decoded_sample_count, int)
            or decoded_sample_count <= 0
            or value.get("media_duration_seconds")
            != round(float(media_duration_seconds), 6)
            or isinstance(tail_padding_sample_count, bool)
            or not isinstance(tail_padding_sample_count, int)
            or tail_padding_sample_count != total_padding_samples
            or isinstance(tail_padding_scene_count, bool)
            or tail_padding_scene_count != int(bool(total_padding_samples))
            or value.get("statistics_exclude_padding") is not True
        ):
            raise ValueError(
                f"TeachObs audio audit binding differs for {item['lesson_id']}"
            )
        decoded_duration = decoded_sample_count / AUDIO_SAMPLE_RATE_HZ
        if value.get("decoded_real_duration_seconds") != round(decoded_duration, 8):
            raise ValueError(
                f"TeachObs decoded-audio duration differs for {item['lesson_id']}"
            )
        final_requested_end = int(
            round(float(item["scenes"][-1]["end"]) * AUDIO_SAMPLE_RATE_HZ)
        )
        if total_padding_samples:
            expected_alignment = round(
                abs(decoded_duration - float(media_duration_seconds)), 8
            )
            if (
                tail_observed_end_sample != decoded_sample_count
                or recorded_tail_alignment_delta != expected_alignment
            ):
                raise ValueError(
                    f"TeachObs decoded-audio EOF differs for {item['lesson_id']}"
                )
        elif decoded_sample_count < final_requested_end:
            raise ValueError(
                f"TeachObs decoded-audio coverage differs for {item['lesson_id']}"
            )
        if value.get("tail_padding_duration_seconds") != round(
            total_padding_samples / AUDIO_SAMPLE_RATE_HZ, 8
        ):
            raise ValueError(
                f"TeachObs audio padding audit differs for {item['lesson_id']}"
            )
    return value


def extract_teachobs_audio_statistics(
    media_path: str | Path,
    item: dict[str, Any],
    output_path: str | Path,
    *,
    media_sha256: str,
    media_duration_seconds: float,
    ffmpeg_command: str = "ffmpeg",
    timeout_seconds: int = 7_200,
    runner: Callable[..., Any] = subprocess.run,
    minimum_scene_coverage: float = 0.5,
) -> dict[str, Any]:
    """Decode 16-kHz mono PCM and compute deterministic statistics per scene."""

    if (
        timeout_seconds < 1
        or not 0 < minimum_scene_coverage <= 1
        or not math.isfinite(float(media_duration_seconds))
        or media_duration_seconds <= 0
    ):
        raise ValueError("invalid TeachObs audio extraction configuration")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "TeachObs audio statistics require NumPy; install the recognition extra"
        ) from exc
    lesson_id = _safe_lesson_id(item.get("lesson_id"))
    media = Path(media_path)
    if media.is_symlink() or not media.is_file():
        raise ValueError(f"TeachObs private media is missing for {lesson_id}")
    if file_sha256(media) != media_sha256:
        raise ValueError(f"TeachObs private media hash differs for {lesson_id}")
    output = Path(output_path)
    ensure_private_directory(output.parent)
    upgraded_from_algorithm: str | None = None
    if output.exists():
        value = _verify_audio_features(
            output,
            item=item,
            media_sha256=media_sha256,
            media_duration_seconds=media_duration_seconds,
        )
        if value.get("algorithm") == "teaching_skill_miner.scene_pcm_statistics.v2":
            return {
                "features": value,
                "output_path": str(output),
                "reused": True,
                "upgraded_legacy_cache": False,
            }
        # A complete v1 cache has just passed its media/scene/row binding and
        # real-coverage checks.  Re-decode and atomically replace it so future
        # reuse also proves EOF alignment and padding exclusion.
        upgraded_from_algorithm = str(value["algorithm"])

    executable = _resolve_executable(ffmpeg_command, purpose="audio decoding")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{lesson_id}-", suffix=".s16le.partial", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    ensure_private_file(temporary)
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media),
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE_HZ),
        "-acodec",
        "pcm_s16le",
        "-f",
        "s16le",
        str(temporary),
    ]
    samples: Any = None
    try:
        try:
            result = runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"TeachObs audio decoding timed out: {lesson_id}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-1600:]
            raise RuntimeError(f"TeachObs audio decoding failed for {lesson_id}: {detail}")
        ensure_private_file(temporary)
        byte_count = temporary.stat().st_size
        if byte_count <= 0 or byte_count % 2:
            raise RuntimeError(f"TeachObs decoded PCM is empty or truncated: {lesson_id}")
        samples = np.memmap(temporary, dtype="<i2", mode="r")
        total_samples = int(samples.shape[0])
        decoded_real_duration_seconds = total_samples / AUDIO_SAMPLE_RATE_HZ
        rows: list[dict[str, Any]] = []
        tail_padding_sample_count = 0
        for scene_index, scene in enumerate(item["scenes"]):
            start_index = int(round(scene["start"] * AUDIO_SAMPLE_RATE_HZ))
            requested_end = int(round(scene["end"] * AUDIO_SAMPLE_RATE_HZ))
            end_index = min(requested_end, total_samples)
            expected_samples = requested_end - start_index
            observed_samples = max(0, end_index - start_index)
            coverage = observed_samples / expected_samples if expected_samples else 0.0
            missing_samples = expected_samples - observed_samples
            is_final_scene = scene_index == len(item["scenes"]) - 1
            if missing_samples and not is_final_scene:
                raise RuntimeError(
                    f"TeachObs decoded audio has a non-tail gap for {lesson_id} "
                    f"scene {scene['scene_no']}"
                )
            if observed_samples <= 0:
                raise RuntimeError(
                    f"TeachObs audio coverage is insufficient for {lesson_id} "
                    f"scene {scene['scene_no']}"
                )
            eof_alignment_delta: float | None = None
            if missing_samples:
                eof_alignment_delta = abs(
                    decoded_real_duration_seconds - float(media_duration_seconds)
                )
                if (
                    eof_alignment_delta
                    > AUDIO_MEDIA_EOF_ALIGNMENT_TOLERANCE_SECONDS
                    or decoded_real_duration_seconds < float(scene["start"])
                    or decoded_real_duration_seconds
                    > float(scene["end"])
                    + AUDIO_MEDIA_EOF_ALIGNMENT_TOLERANCE_SECONDS
                ):
                    raise RuntimeError(
                        f"TeachObs decoded audio tail does not align with media EOF "
                        f"for {lesson_id}"
                    )
            values = np.asarray(samples[start_index:end_index], dtype=np.float32)
            normalized = values / 32768.0
            # A fixed-length logical scene window is formed for downstream shape
            # consistency, but every statistic below is deliberately calculated
            # from ``normalized`` (the real decoded prefix), never this padded view.
            analysis_window = np.pad(
                normalized, (0, missing_samples), mode="constant", constant_values=0.0
            )
            if int(analysis_window.shape[0]) != expected_samples:
                raise RuntimeError(
                    f"TeachObs audio padding failed for {lesson_id} "
                    f"scene {scene['scene_no']}"
                )
            signs = np.signbit(normalized)
            crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
            crossing_rate = crossings / max(1, observed_samples - 1)
            missing_duration_seconds = missing_samples / AUDIO_SAMPLE_RATE_HZ
            tail_padding_sample_count += missing_samples
            rows.append(
                {
                    "scene_no": scene["scene_no"],
                    "start": scene["start"],
                    "end": scene["end"],
                    "requested_sample_count": expected_samples,
                    "observed_sample_count": observed_samples,
                    "observed_duration_seconds": round(
                        observed_samples / AUDIO_SAMPLE_RATE_HZ, 8
                    ),
                    "coverage_fraction": round(coverage, 8),
                    "real_audio_coverage_fraction": round(coverage, 8),
                    "coverage_basis": (
                        "decoded_real_samples_excluding_zero_padding"
                    ),
                    "missing_duration_seconds": round(
                        missing_duration_seconds, 8
                    ),
                    "padding_applied": bool(missing_samples),
                    "padding_sample_count": missing_samples,
                    "padding_duration_seconds": round(
                        missing_duration_seconds, 8
                    ),
                    "padding_value_normalized": 0.0,
                    "analysis_window_sample_count": expected_samples,
                    "statistics_sample_count": observed_samples,
                    "statistics_exclude_padding": True,
                    "minimum_scene_coverage_met": (
                        coverage >= minimum_scene_coverage
                    ),
                    "tail_eof_exception_applied": bool(missing_samples),
                    "media_eof_alignment_delta_seconds": (
                        round(eof_alignment_delta, 8)
                        if eof_alignment_delta is not None
                        else None
                    ),
                    "rms_amplitude": round(
                        float(np.sqrt(np.mean(np.square(normalized, dtype=np.float64)))),
                        8,
                    ),
                    "mean_absolute_amplitude": round(
                        float(np.mean(np.abs(normalized), dtype=np.float64)), 8
                    ),
                    "peak_absolute_amplitude": round(
                        float(np.max(np.abs(normalized))), 8
                    ),
                    "dc_offset": round(float(np.mean(normalized, dtype=np.float64)), 8),
                    "zero_crossing_rate": round(float(crossing_rate), 8),
                    "silence_fraction": round(
                        float(np.mean(np.abs(normalized) <= AUDIO_SILENCE_THRESHOLD)),
                        8,
                    ),
                }
            )
        features = {
            "schema": AUDIO_SCHEMA,
            "lesson_id": lesson_id,
            "media_sha256": media_sha256,
            "scene_manifest_sha256": item["scene_manifest_sha256"],
            "scene_count": len(rows),
            "decoder": "ffmpeg_pcm_s16le_mono",
            "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
            "media_duration_seconds": round(float(media_duration_seconds), 6),
            "decoded_real_sample_count": total_samples,
            "decoded_real_duration_seconds": round(
                decoded_real_duration_seconds, 8
            ),
            "media_eof_alignment_tolerance_seconds": (
                AUDIO_MEDIA_EOF_ALIGNMENT_TOLERANCE_SECONDS
            ),
            "minimum_scene_coverage": minimum_scene_coverage,
            "tail_padding_scene_count": int(bool(tail_padding_sample_count)),
            "tail_padding_sample_count": tail_padding_sample_count,
            "tail_padding_duration_seconds": round(
                tail_padding_sample_count / AUDIO_SAMPLE_RATE_HZ, 8
            ),
            "statistics_exclude_padding": True,
            "silence_absolute_amplitude_threshold": AUDIO_SILENCE_THRESHOLD,
            "algorithm": "teaching_skill_miner.scene_pcm_statistics.v2",
            "cache_upgrade_from_algorithm": upgraded_from_algorithm,
            "numpy_version": np.__version__,
            "feature_names": [
                "coverage_fraction",
                "rms_amplitude",
                "mean_absolute_amplitude",
                "peak_absolute_amplitude",
                "dc_offset",
                "zero_crossing_rate",
                "silence_fraction",
            ],
            "private_artifact": True,
            "public_release_authorized": False,
            "scenes": rows,
            "claim_boundary": {
                "audio_statistics_computed": True,
                "only_final_scene_media_eof_padding_allowed": True,
                "padding_excluded_from_statistics": True,
                "speech_content_transcribed": False,
                "speaker_identity_inferred": False,
                "recognition_accuracy_established": False,
            },
        }
        write_json(output, features)
        return {
            "features": features,
            "output_path": str(output),
            "reused": False,
            "upgraded_legacy_cache": upgraded_from_algorithm is not None,
        }
    finally:
        if samples is not None:
            del samples
        temporary.unlink(missing_ok=True)


def clip_preflight(
    model_path: str | Path | None,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Check a local CLIP snapshot and imports without loading model weights."""

    if model_path is None:
        return {"requested": False, "ready": True, "status": "not_requested"}
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        return {
            "requested": True,
            "ready": False,
            "reason": f"local CLIP model directory does not exist: {path}",
        }
    missing_dependencies = [
        name
        for name in ("PIL", "torch", "transformers")
        if importlib.util.find_spec(name) is None
    ]
    if missing_dependencies:
        return {
            "requested": True,
            "ready": False,
            "reason": "missing visual dependencies: " + ", ".join(missing_dependencies),
        }
    weights = sorted(
        path.rglob("*.safetensors")
    ) + sorted(path.rglob("pytorch_model*.bin"))
    if not weights:
        return {
            "requested": True,
            "ready": False,
            "reason": "local CLIP snapshot contains no model weight file",
        }
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            return {
                "requested": True,
                "ready": False,
                "reason": "CUDA was requested but torch.cuda.is_available() is false",
            }
    return {
        "requested": True,
        "ready": True,
        "model_path": str(path),
        "weight_file_count": len(weights),
        "device": device,
    }


def extract_teachobs_clip_embeddings(
    task_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path,
    source_revision: str,
    device: str = "cpu",
    batch_size: int = 32,
    expected_output_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the existing hash-bound CLIP extractor on private scene frames."""

    if batch_size < 1 or not source_revision.strip():
        raise ValueError("CLIP batch size and source revision must be explicit")
    preflight = clip_preflight(model_path, device=device)
    if not preflight["ready"]:
        raise RuntimeError(f"TeachObs CLIP preflight failed: {preflight['reason']}")
    task_input = Path(task_path)
    if task_input.is_symlink() or not task_input.is_file():
        raise ValueError("TeachObs CLIP frame task is missing or unsafe")
    task_file = task_input.resolve()
    task = read_json(task_file)
    if not isinstance(task, dict):
        raise ValueError("TeachObs CLIP frame task is not an object")
    output = Path(output_path)
    ensure_private_directory(output.parent)
    task_file_digest = file_sha256(task_file)
    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise ValueError("existing TeachObs CLIP result is unsafe")
        if expected_output_sha256 is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", expected_output_sha256):
                raise ValueError("expected TeachObs CLIP result hash is invalid")
            if file_sha256(output) != expected_output_sha256:
                raise ValueError("existing TeachObs CLIP result hash differs")
        existing = read_json(output)
        _validate_teachobs_clip_result(
            existing,
            task=task,
            task_file_sha256=task_file_digest,
            model_path=model_path,
            source_revision=source_revision,
            device=device,
            batch_size=batch_size,
        )
        ensure_private_file(output)
        return {"result": existing, "output_path": str(output), "reused": True}

    from .visual_semantics import run_inference

    arguments = argparse.Namespace(
        tasks=task_file,
        frame_root=task_file.parent,
        output=output,
        model=str(Path(model_path).resolve()),
        revision=None,
        source_model_id="openai/clip-vit-base-patch32",
        source_revision=source_revision,
        device=device,
        batch_size=batch_size,
    )
    result = run_inference(arguments)
    result["teachobs_task_file_sha256"] = task_file_digest
    result["teachobs_source_revision"] = source_revision
    result["private_artifact"] = True
    result["public_release_authorized"] = False
    privacy = result.get("privacy")
    if not isinstance(privacy, dict):
        raise ValueError("TeachObs CLIP result has no privacy boundary")
    privacy["safe_to_publish"] = False
    _validate_teachobs_clip_result(
        result,
        task=task,
        task_file_sha256=task_file_digest,
        model_path=model_path,
        source_revision=source_revision,
        device=device,
        batch_size=batch_size,
    )
    write_json(output, result)
    return {"result": result, "output_path": str(output), "reused": False}


def _validate_teachobs_clip_result(
    result: Any,
    *,
    task: dict[str, Any],
    task_file_sha256: str,
    model_path: str | Path,
    source_revision: str,
    device: str,
    batch_size: int,
) -> None:
    """Fail closed on a newly generated or cached private CLIP artifact."""

    task_frames = task.get("frames")
    result_frames = result.get("frames") if isinstance(result, dict) else None
    expected_model_path = str(Path(model_path).resolve())
    provenance = result.get("model_provenance") if isinstance(result, dict) else None
    privacy = result.get("privacy") if isinstance(result, dict) else None
    claims = result.get("claim_boundary") if isinstance(result, dict) else None
    if (
        not isinstance(result, dict)
        or result.get("schema") != CLIP_RESULT_SCHEMA
        or result.get("video_id") != task.get("video_id")
        or result.get("media_sha256") != task.get("media_sha256")
        or result.get("task_manifest_sha256") != _canonical_sha256(task)
        or result.get("teachobs_task_file_sha256") != task_file_sha256
        or result.get("teachobs_source_revision") != source_revision
        or result.get("frame_count") != task.get("frame_count")
        or result.get("private_artifact") is not True
        or result.get("public_release_authorized") is not False
        or not isinstance(task_frames, list)
        or not task_frames
        or not isinstance(result_frames, list)
        or len(result_frames) != len(task_frames)
        or not isinstance(provenance, dict)
        or provenance.get("loaded_model_path") != expected_model_path
        or provenance.get("requested_revision") != source_revision
        or provenance.get("device") != device
        or provenance.get("batch_size") != batch_size
        or not isinstance(privacy, dict)
        or privacy.get("face_recognition_performed") is not False
        or privacy.get("identity_recognition_performed") is not False
        or privacy.get("safe_to_publish") is not False
        or not isinstance(claims, dict)
        or claims.get("recognition_accuracy_established") is not False
    ):
        raise ValueError("TeachObs CLIP result has a different or invalid binding")

    weight_manifest = provenance.get("weight_manifest")
    weight_files = (
        weight_manifest.get("files") if isinstance(weight_manifest, dict) else None
    )
    if (
        not isinstance(weight_manifest, dict)
        or not isinstance(weight_files, list)
        or not weight_files
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(weight_manifest.get("manifest_sha256", ""))
        )
        or _canonical_sha256(weight_files) != weight_manifest["manifest_sha256"]
    ):
        raise ValueError("TeachObs CLIP result has invalid model-weight provenance")

    embedding_dimension: int | None = None
    for task_frame, result_frame in zip(task_frames, result_frames, strict=True):
        if (
            not isinstance(task_frame, dict)
            or not isinstance(result_frame, dict)
            or any(
                result_frame.get(field) != task_frame.get(field)
                for field in ("frame_id", "timestamp", "path", "sha256")
            )
        ):
            raise ValueError("TeachObs CLIP result frame mapping differs from its task")
        embedding = result_frame.get("embedding")
        if (
            not isinstance(embedding, list)
            or not embedding
            or len(embedding) > 4096
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in embedding
            )
            or result_frame.get("embedding_sha256")
            != _canonical_sha256(embedding)
        ):
            raise ValueError("TeachObs CLIP result contains an invalid embedding")
        if embedding_dimension is None:
            embedding_dimension = len(embedding)
        elif len(embedding) != embedding_dimension:
            raise ValueError("TeachObs CLIP embedding dimensions are inconsistent")


def _dhash_distance(before: Any, after: Any) -> tuple[int | None, float | None]:
    if not isinstance(before, str) or not isinstance(after, str):
        return None, None
    if not re.fullmatch(r"[0-9a-fA-F]{16}", before) or not re.fullmatch(
        r"[0-9a-fA-F]{16}", after
    ):
        return None, None
    distance = (int(before, 16) ^ int(after, 16)).bit_count()
    return distance, round(distance / 64.0, 6)


def _visual_evidence_configuration(
    *,
    use_ocr: bool,
    ocr_language: str,
    minimum_word_confidence: float,
    scene_change_threshold: float,
    slide_change_threshold: float,
) -> dict[str, Any]:
    return {
        "use_ocr": use_ocr,
        "ocr_language": ocr_language if use_ocr else None,
        "minimum_word_confidence": minimum_word_confidence,
        "scene_change_threshold": scene_change_threshold,
        "slide_change_threshold": slide_change_threshold,
        "event_type_counting_policy": VISUAL_EVIDENCE_EVENT_COUNTING_POLICY,
    }


def _legacy_visual_evidence_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    legacy = dict(configuration)
    if (
        legacy.pop("event_type_counting_policy", None)
        != VISUAL_EVIDENCE_EVENT_COUNTING_POLICY
    ):
        raise ValueError("TeachObs visual-evidence counting policy differs")
    return legacy


def _computed_visual_event_type_counts(
    events: Any,
) -> dict[str, int]:
    if not isinstance(events, list):
        raise ValueError("TeachObs visual evidence events are invalid")
    counts = {event_type: 0 for event_type in VISUAL_EVIDENCE_EVENT_TYPES}
    for event in events:
        event_type = event.get("type") if isinstance(event, dict) else None
        if event_type not in counts:
            raise ValueError("TeachObs visual evidence event type is invalid")
        counts[event_type] += 1
    return counts


def _validate_visual_event_type_counts(
    value: dict[str, Any],
    *,
    legacy_v1: bool,
) -> None:
    events = value.get("events")
    event_count = value.get("event_count")
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 0
        or not isinstance(events, list)
        or len(events) != event_count
    ):
        raise ValueError("TeachObs visual evidence event count differs")
    computed = _computed_visual_event_type_counts(events)
    declared = value.get("event_type_counts")
    if legacy_v1:
        expected = {
            event_type: computed[event_type]
            for event_type in _VISUAL_EVIDENCE_TRANSITION_EVENT_TYPES
        }
    else:
        expected = computed
    if declared != expected:
        raise ValueError("TeachObs visual evidence event type counts differ")
    if not legacy_v1 and sum(declared.values()) != event_count:
        raise ValueError("TeachObs visual evidence event type counts do not sum")


def _validate_visual_evidence_cache_document(
    value: Any,
    *,
    task: dict[str, Any],
    task_file_sha256: str,
    configuration: dict[str, Any],
    schema: str,
) -> None:
    legacy_v1 = schema == _LEGACY_VISUAL_EVIDENCE_SCHEMA_V1
    expected_configuration = (
        _legacy_visual_evidence_configuration(configuration)
        if legacy_v1
        else configuration
    )
    scenes = value.get("scenes") if isinstance(value, dict) else None
    transitions = value.get("transitions") if isinstance(value, dict) else None
    frames = task.get("frames")
    if (
        schema not in {VISUAL_EVIDENCE_SCHEMA, _LEGACY_VISUAL_EVIDENCE_SCHEMA_V1}
        or not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("lesson_id") != task.get("video_id")
        or value.get("media_sha256") != task.get("media_sha256")
        or value.get("scene_manifest_sha256")
        != task.get("scene_manifest_sha256")
        or value.get("task_file_sha256") != task_file_sha256
        or value.get("configuration") != expected_configuration
        or value.get("configuration_sha256")
        != _canonical_sha256(expected_configuration)
        or value.get("scene_count") != task.get("frame_count")
        or not isinstance(frames, list)
        or not isinstance(scenes, list)
        or len(scenes) != len(frames)
        or not isinstance(transitions, list)
        or value.get("transition_count") != len(transitions)
        or value.get("private_artifact") is not True
        or value.get("public_release_authorized") is not False
    ):
        raise ValueError("TeachObs visual evidence identity differs")
    _validate_visual_event_type_counts(value, legacy_v1=legacy_v1)
    for frame, scene in zip(frames, scenes, strict=True):
        if not isinstance(frame, dict) or not isinstance(scene, dict):
            raise ValueError("TeachObs visual evidence scene binding differs")
        try:
            timestamps_match = math.isclose(
                float(scene.get("timestamp", -1)),
                float(frame.get("timestamp", -2)),
                abs_tol=1e-6,
            )
        except (TypeError, ValueError, OverflowError):
            timestamps_match = False
        if (
            scene.get("frame_id") != frame.get("frame_id")
            or scene.get("scene_no") != frame.get("scene_no")
            or scene.get("path") != frame.get("path")
            or scene.get("sha256") != frame.get("sha256")
            or not timestamps_match
        ):
            raise ValueError("TeachObs visual evidence scene binding differs")


def _validate_obsolete_visual_evidence_binding(
    task_path: Path,
    evidence_path: Path,
    *,
    item: dict[str, Any],
    media_sha256: str,
    media_duration_seconds: float,
    configuration: dict[str, Any],
    expected_evidence_sha256: str | None = None,
) -> dict[str, str] | None:
    """Validate an obsolete task or v1 evidence cache before replacement."""

    if not evidence_path.exists():
        return None
    if evidence_path.is_symlink() or not evidence_path.is_file():
        raise ValueError("existing TeachObs visual evidence is unsafe")
    old_task = _verify_frame_task(
        task_path,
        item=item,
        media_sha256=media_sha256,
        media_duration_seconds=media_duration_seconds,
        require_timestamp_evidence=False,
    )
    old_task_digest = file_sha256(task_path)
    evidence_digest = file_sha256(evidence_path)
    if expected_evidence_sha256 is not None and (
        not re.fullmatch(r"[0-9a-f]{64}", expected_evidence_sha256)
        or evidence_digest != expected_evidence_sha256
    ):
        raise ValueError("existing TeachObs visual evidence hash differs")
    existing = read_json(evidence_path)
    schema = existing.get("schema") if isinstance(existing, dict) else None
    if schema not in {VISUAL_EVIDENCE_SCHEMA, _LEGACY_VISUAL_EVIDENCE_SCHEMA_V1}:
        raise ValueError("existing TeachObs visual evidence schema differs")
    _validate_visual_evidence_cache_document(
        existing,
        task=old_task,
        task_file_sha256=old_task_digest,
        configuration=configuration,
        schema=schema,
    )
    return {
        "task_file_sha256": old_task_digest,
        "evidence_file_sha256": evidence_digest,
        "schema": schema,
    }


def extract_teachobs_scene_visual_evidence(
    task_path: str | Path,
    output_path: str | Path,
    *,
    use_ocr: bool = True,
    ocr_language: str = "eng",
    minimum_word_confidence: float = 35.0,
    scene_change_threshold: float = 0.25,
    slide_change_threshold: float = 0.45,
    visual_jobs: int = 1,
    ocr_jobs: int = 1,
    jobs: int | None = None,
    image_metric_extractor: Callable[[Path], dict[str, Any]] | None = None,
    ocr_extractor: Callable[..., tuple[str, dict[str, Any]]] | None = None,
    obsolete_binding: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute private image/OCR evidence and adjacent-frame visual events.

    The event thresholds are deterministic heuristics, not calibrated
    probabilities or human labels. OCR text is deliberately retained only in
    this private artifact.
    """

    if jobs is not None:
        if visual_jobs != 1 or ocr_jobs != 1:
            raise ValueError(
                "legacy jobs cannot be combined with visual_jobs or ocr_jobs"
            )
        visual_jobs = jobs
        ocr_jobs = jobs
    invalid_jobs = any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 8
        for value in (visual_jobs, ocr_jobs)
    )
    if (
        not 0 <= minimum_word_confidence <= 100
        or not 0 <= scene_change_threshold <= 1
        or not 0 <= slide_change_threshold <= 1
        or scene_change_threshold > slide_change_threshold
        or invalid_jobs
    ):
        raise ValueError("invalid TeachObs visual-evidence thresholds")
    task_file = Path(task_path).resolve()
    if task_file.is_symlink() or not task_file.is_file():
        raise ValueError("TeachObs visual-evidence frame task is missing or unsafe")
    task = read_json(task_file)
    if not isinstance(task, dict) or task.get("schema") != FRAME_TASK_SCHEMA:
        raise ValueError("TeachObs visual-evidence task has an unsupported schema")
    frames = task.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("TeachObs visual-evidence task contains no frames")
    output = Path(output_path)
    ensure_private_directory(output.parent)
    task_digest = file_sha256(task_file)
    config = _visual_evidence_configuration(
        use_ocr=use_ocr,
        ocr_language=ocr_language,
        minimum_word_confidence=minimum_word_confidence,
        scene_change_threshold=scene_change_threshold,
        slide_change_threshold=slide_change_threshold,
    )
    config_digest = _canonical_sha256(config)
    replacing_obsolete_binding = False
    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise ValueError("existing TeachObs visual evidence is unsafe")
        existing = read_json(output)
        exact_binding = (
            isinstance(existing, dict)
            and existing.get("schema") == VISUAL_EVIDENCE_SCHEMA
            and existing.get("task_file_sha256") == task_digest
            and existing.get("configuration_sha256") == config_digest
        )
        if exact_binding:
            _validate_visual_evidence_cache_document(
                existing,
                task=task,
                task_file_sha256=task_digest,
                configuration=config,
                schema=VISUAL_EVIDENCE_SCHEMA,
            )
            return {
                "result": existing,
                "output_path": str(output),
                "reused": True,
                "replaced_obsolete_binding": False,
                "upgraded_legacy_event_counts": False,
            }
        legacy_config = _legacy_visual_evidence_configuration(config)
        legacy_same_task = (
            isinstance(existing, dict)
            and existing.get("schema") == _LEGACY_VISUAL_EVIDENCE_SCHEMA_V1
            and existing.get("task_file_sha256") == task_digest
            and existing.get("configuration_sha256")
            == _canonical_sha256(legacy_config)
        )
        if legacy_same_task:
            _validate_visual_evidence_cache_document(
                existing,
                task=task,
                task_file_sha256=task_digest,
                configuration=config,
                schema=_LEGACY_VISUAL_EVIDENCE_SCHEMA_V1,
            )
            upgraded = {
                **existing,
                "schema": VISUAL_EVIDENCE_SCHEMA,
                "configuration": config,
                "configuration_sha256": config_digest,
                "event_type_counts": _computed_visual_event_type_counts(
                    existing["events"]
                ),
                "cache_upgrade": {
                    "from_schema": _LEGACY_VISUAL_EVIDENCE_SCHEMA_V1,
                    "metadata_only": True,
                    "events_reused_after_hash_and_binding_validation": True,
                    "event_type_counts_recomputed_from_events": True,
                },
            }
            _validate_visual_evidence_cache_document(
                upgraded,
                task=task,
                task_file_sha256=task_digest,
                configuration=config,
                schema=VISUAL_EVIDENCE_SCHEMA,
            )
            write_json(output, upgraded)
            return {
                "result": upgraded,
                "output_path": str(output),
                "reused": False,
                "replaced_obsolete_binding": True,
                "upgraded_legacy_event_counts": True,
            }
        else:
            binding_schema = (
                obsolete_binding.get("schema")
                if isinstance(obsolete_binding, dict)
                else None
            )
            expected_config = (
                legacy_config
                if binding_schema == _LEGACY_VISUAL_EVIDENCE_SCHEMA_V1
                else config
            )
            if (
                not isinstance(obsolete_binding, dict)
                or set(obsolete_binding)
                != {"task_file_sha256", "evidence_file_sha256", "schema"}
                or binding_schema
                not in {
                    VISUAL_EVIDENCE_SCHEMA,
                    _LEGACY_VISUAL_EVIDENCE_SCHEMA_V1,
                }
                or (
                    obsolete_binding["task_file_sha256"] == task_digest
                    and binding_schema != _LEGACY_VISUAL_EVIDENCE_SCHEMA_V1
                )
                or existing.get("task_file_sha256")
                != obsolete_binding["task_file_sha256"]
                or file_sha256(output)
                != obsolete_binding["evidence_file_sha256"]
                or existing.get("schema") != binding_schema
                or existing.get("lesson_id") != task.get("video_id")
                or existing.get("media_sha256") != task.get("media_sha256")
                or existing.get("scene_manifest_sha256")
                != task.get("scene_manifest_sha256")
                or existing.get("configuration") != expected_config
                or existing.get("configuration_sha256")
                != _canonical_sha256(expected_config)
            ):
                raise ValueError(
                    "existing TeachObs visual evidence has a different binding"
                )
            replacing_obsolete_binding = True

    if image_metric_extractor is None or ocr_extractor is None:
        from .longform_multimodal import (
            _dhash_and_image_metrics,
            _ocr_frame_audited,
        )

        image_metric_extractor = image_metric_extractor or _dhash_and_image_metrics
        ocr_extractor = ocr_extractor or _ocr_frame_audited
    assert image_metric_extractor is not None
    assert ocr_extractor is not None

    frame_root = task_file.parent.resolve()

    def bind_frame(frame: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(frame, dict):
            raise ValueError("TeachObs visual-evidence task has a non-object frame")
        relative = PurePosixPath(str(frame.get("path", "")))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("TeachObs visual-evidence frame path is unsafe")
        path = (frame_root / relative.name).resolve()
        try:
            path.relative_to(frame_root)
        except ValueError as exc:
            raise ValueError("TeachObs visual-evidence frame escapes its root") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError("TeachObs visual-evidence frame is missing")
        if file_sha256(path) != frame.get("sha256"):
            raise ValueError("TeachObs visual-evidence frame hash differs")
        return {**frame, "absolute_path": path}

    bound_frames = [bind_frame(frame) for frame in frames]

    def analyze_visual(frame: dict[str, Any]) -> dict[str, Any]:
        metrics = image_metric_extractor(frame["absolute_path"])
        if not isinstance(metrics, dict):
            raise RuntimeError("TeachObs image metric extractor returned invalid data")
        return metrics

    def analyze_ocr(frame: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        ocr_text, ocr_audit = ocr_extractor(
            frame["absolute_path"],
            ocr_language,
            minimum_word_confidence=minimum_word_confidence,
        )
        if not isinstance(ocr_text, str) or not isinstance(ocr_audit, dict):
            raise RuntimeError("TeachObs OCR extractor returned invalid data")
        return ocr_text, ocr_audit

    def ordered_map(function: Callable[[dict[str, Any]], Any], workers: int, prefix: str) -> list[Any]:
        if workers == 1:
            return [function(frame) for frame in bound_frames]
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=prefix
        ) as pool:
            return list(pool.map(function, bound_frames))

    visual_values = ordered_map(analyze_visual, visual_jobs, "teachobs-visual")
    if use_ocr:
        ocr_values = ordered_map(analyze_ocr, ocr_jobs, "teachobs-ocr")
    else:
        ocr_values = [
            (
                "",
                {
                    "status": "disabled",
                    "language": None,
                    "minimum_word_confidence": minimum_word_confidence,
                    "raw_word_count": 0,
                    "accepted_word_count": 0,
                    "accepted_mean_confidence": None,
                    "confidence_is_calibrated_probability": False,
                },
            )
            for _ in bound_frames
        ]
    private_rows = [
        {
            "frame_id": frame.get("frame_id"),
            "scene_no": frame.get("scene_no"),
            "requested_timestamp": frame.get(
                "requested_timestamp", frame.get("timestamp")
            ),
            "timestamp": frame.get("timestamp"),
            "timestamp_clamped": frame.get("timestamp_clamped", False),
            "timestamp_clamp_delta_seconds": frame.get(
                "timestamp_clamp_delta_seconds", 0.0
            ),
            "source_frame_timestamp": frame.get("source_frame_timestamp"),
            "timestamp_selection_error_seconds": frame.get(
                "timestamp_selection_error_seconds"
            ),
            "timestamp_verified_within_tolerance": frame.get(
                "timestamp_verified_within_tolerance", False
            ),
            "path": frame.get("path"),
            "sha256": frame.get("sha256"),
            "image_metrics": metrics,
            "ocr_text": ocr_value[0],
            "ocr_audit": ocr_value[1],
        }
        for frame, metrics, ocr_value in zip(
            bound_frames, visual_values, ocr_values, strict=True
        )
    ]
    inference_rows = [
        {
            "frame_id": row["frame_id"],
            "timestamp": row["timestamp"],
            "path": row["path"],
            "sampling_source": "teachobs_midpoint",
            "ocr_text": row["ocr_text"],
        }
        for row in private_rows
    ]

    from .multimodal import infer_visual_events

    ocr_events = infer_visual_events(inference_rows)
    events = [event for event in ocr_events if event.get("type") != "scene_change"]
    transitions: list[dict[str, Any]] = []
    for before, after in zip(private_rows, private_rows[1:]):
        before_metrics = before["image_metrics"]
        after_metrics = after["image_metrics"]
        distance, distance_fraction = _dhash_distance(
            before_metrics.get("dhash64"), after_metrics.get("dhash64")
        )
        edge_before = before_metrics.get("edge_difference_mean")
        edge_after = after_metrics.get("edge_difference_mean")
        edge_delta = (
            round(float(edge_after) - float(edge_before), 6)
            if isinstance(edge_before, (int, float))
            and isinstance(edge_after, (int, float))
            else None
        )
        transition_types: list[str] = []
        start = float(before["timestamp"])
        end = float(after["timestamp"])
        if distance_fraction is not None and distance_fraction >= scene_change_threshold:
            transition_types.append("scene_change")
            events.append(
                {
                    "type": "scene_change",
                    "start": start,
                    "end": end,
                    "modalities": ["visual"],
                    "evidence": {
                        "before_frame": before["path"],
                        "after_frame": after["path"],
                        "dhash_hamming_distance": distance,
                        "dhash_distance_fraction": distance_fraction,
                    },
                    "confidence": round(
                        min(0.9, 0.5 + float(distance_fraction)), 3
                    ),
                    "heuristic_not_human_ground_truth": True,
                }
            )
        existing_types = {
            event["type"]
            for event in events
            if math.isclose(float(event["start"]), start, abs_tol=1e-6)
            and math.isclose(float(event["end"]), end, abs_tol=1e-6)
        }
        transition_types.extend(sorted(existing_types - {"scene_change"}))
        if (
            distance_fraction is not None
            and distance_fraction >= slide_change_threshold
            and "slide_change" not in existing_types
        ):
            transition_types.append("slide_change")
            events.append(
                {
                    "type": "slide_change",
                    "start": start,
                    "end": end,
                    "modalities": ["visual"],
                    "evidence": {
                        "before_frame": before["path"],
                        "after_frame": after["path"],
                        "dhash_hamming_distance": distance,
                        "dhash_distance_fraction": distance_fraction,
                    },
                    "confidence": round(
                        min(0.88, 0.42 + float(distance_fraction)), 3
                    ),
                    "heuristic_not_human_ground_truth": True,
                }
            )
        if (
            "board_build_up" not in existing_types
            and edge_delta is not None
            and edge_delta >= 0.75
            and distance_fraction is not None
            and 0.015 <= distance_fraction < scene_change_threshold
        ):
            transition_types.append("board_build_up")
            events.append(
                {
                    "type": "board_build_up",
                    "start": start,
                    "end": end,
                    "modalities": ["visual"],
                    "evidence": {
                        "before_frame": before["path"],
                        "after_frame": after["path"],
                        "dhash_hamming_distance": distance,
                        "dhash_distance_fraction": distance_fraction,
                        "edge_difference_delta": edge_delta,
                    },
                    "confidence": 0.55,
                    "heuristic_not_human_ground_truth": True,
                }
            )
        transitions.append(
            {
                "before_frame_id": before["frame_id"],
                "after_frame_id": after["frame_id"],
                "start": start,
                "end": end,
                "dhash_hamming_distance": distance,
                "dhash_distance_fraction": distance_fraction,
                "edge_difference_delta": edge_delta,
                "event_types": sorted(set(transition_types)),
            }
        )
    events.sort(key=lambda event: (float(event["start"]), str(event["type"])))
    status_counts: dict[str, int] = {}
    for row in private_rows:
        status = str(row["ocr_audit"].get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    event_type_counts = _computed_visual_event_type_counts(events)
    result = {
        "schema": VISUAL_EVIDENCE_SCHEMA,
        "lesson_id": task.get("video_id"),
        "media_sha256": task.get("media_sha256"),
        "scene_manifest_sha256": task.get("scene_manifest_sha256"),
        "task_file_sha256": task_digest,
        "configuration": config,
        "configuration_sha256": config_digest,
        "scene_count": len(private_rows),
        "transition_count": len(transitions),
        "event_count": len(events),
        "event_type_counts": event_type_counts,
        "ocr_status_counts": status_counts,
        "private_artifact": True,
        "public_release_authorized": False,
        "scenes": private_rows,
        "transitions": transitions,
        "events": events,
        "privacy": {
            "contains_frame_paths": True,
            "contains_ocr_text": True,
            "contains_image_metrics": True,
            "safe_to_publish": False,
        },
        "claim_boundary": {
            "per_scene_image_metrics_computed": True,
            "per_scene_ocr_requested": use_ocr,
            "ocr_complete_for_every_scene": status_counts == {"completed": len(private_rows)},
            "adjacent_visual_events_inferred": True,
            "events_are_deterministic_heuristics": True,
            "event_accuracy_established": False,
            "human_ground_truth_used": False,
        },
    }
    write_json(output, result)
    return {
        "result": result,
        "output_path": str(output),
        "reused": False,
        "replaced_obsolete_binding": replacing_obsolete_binding,
        "upgraded_legacy_event_counts": False,
    }


def _media_records_by_id(
    media_manifest: dict[str, Any],
    *,
    plan: dict[str, Any],
    output_directory: Path,
) -> dict[str, dict[str, Any]]:
    if media_manifest.get("schema") != MEDIA_SCHEMA:
        raise ValueError("unsupported TeachObs private-media manifest")
    if media_manifest.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("TeachObs media manifest is bound to a different plan")
    rows = media_manifest.get("lessons")
    if not isinstance(rows, list):
        raise ValueError("TeachObs media manifest contains no lesson records")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("TeachObs media manifest contains an invalid lesson")
        lesson_id = _safe_lesson_id(row.get("lesson_id"))
        relative = PurePosixPath(str(row.get("media_path", "")))
        if relative != PurePosixPath("videos") / f"{lesson_id}.mp4":
            raise ValueError(f"TeachObs media path is unsafe for {lesson_id}")
        path = output_directory.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"TeachObs private media is missing for {lesson_id}")
        if file_sha256(path) != row.get("media_sha256"):
            raise ValueError(f"TeachObs private media hash differs for {lesson_id}")
        by_id[lesson_id] = row
    return by_id


_FEATURE_FAILURE_SAFE_REASONS = {
    "media_binding": "Private media binding validation did not complete.",
    "frame_extraction": "Private scene-frame extraction did not complete.",
    "visual_evidence": "Private visual-evidence extraction did not complete.",
    "audio_statistics": "Private audio-statistics extraction did not complete.",
    "lesson_pipeline": "Private per-lesson feature extraction did not complete.",
}


def _safe_exception_type(exception: Exception) -> str:
    if isinstance(exception, subprocess.TimeoutExpired):
        return "TimeoutExpired"
    if isinstance(exception, ValueError):
        return "ValueError"
    if isinstance(exception, RuntimeError):
        return "RuntimeError"
    if isinstance(exception, OSError):
        return "OSError"
    return "Exception"


class _TeachObsFeatureStageFailure(RuntimeError):
    """Carry a safe stage code without persisting the underlying message."""

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = (
            stage if stage in _FEATURE_FAILURE_SAFE_REASONS else "lesson_pipeline"
        )
        self.cause_type = _safe_exception_type(cause)
        super().__init__(_FEATURE_FAILURE_SAFE_REASONS[self.stage])


def extract_teachobs_multimodal_features(
    plan: dict[str, Any],
    media_manifest: dict[str, Any],
    output_directory: str | Path,
    *,
    acknowledge_source_terms: bool = False,
    include_audio_statistics: bool = True,
    include_visual_evidence: bool = True,
    use_ocr: bool = True,
    ocr_language: str = "eng",
    minimum_ocr_word_confidence: float = 35.0,
    clip_model: str | Path | None = None,
    clip_source_revision: str | None = None,
    clip_device: str = "cpu",
    clip_batch_size: int = 32,
    ffmpeg_command: str = "ffmpeg",
    frame_timeout_seconds: int = 7_200,
    audio_timeout_seconds: int = 7_200,
    frame_runner: Callable[..., Any] = subprocess.run,
    audio_runner: Callable[..., Any] = subprocess.run,
    image_metric_extractor: Callable[[Path], dict[str, Any]] | None = None,
    ocr_extractor: Callable[..., tuple[str, dict[str, Any]]] | None = None,
    feature_jobs: int = 2,
    visual_jobs: int = 1,
    ocr_jobs: int = 1,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Extract hash-bound frames, audio statistics, and optional CLIP vectors."""

    lessons = _validate_plan(plan)
    if not acknowledge_source_terms:
        raise ValueError(
            "TeachObs source-media feature extraction requires "
            "acknowledge_source_terms=True"
        )
    job_values = (feature_jobs, visual_jobs, ocr_jobs)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 8
        for value in job_values
    ):
        raise ValueError("TeachObs feature/visual/OCR jobs must each be in 1-8")
    inner_jobs = (
        max(visual_jobs, ocr_jobs if use_ocr else 1)
        if include_visual_evidence
        else 1
    )
    if feature_jobs * inner_jobs > 8:
        raise ValueError(
            "TeachObs feature_jobs * active inner visual/OCR jobs must not exceed 8"
        )
    if clip_model is not None:
        if not clip_source_revision:
            raise ValueError("a pinned CLIP source revision is required")
        preflight = clip_preflight(clip_model, device=clip_device)
        if not preflight["ready"]:
            raise RuntimeError(f"TeachObs CLIP preflight failed: {preflight['reason']}")
    output = ensure_private_directory(output_directory).resolve()
    media_by_id = _media_records_by_id(
        media_manifest, plan=plan, output_directory=output
    )
    frames_root = ensure_private_directory(output / "frames")
    audio_root = ensure_private_directory(output / "features" / "audio")
    visual_evidence_root = ensure_private_directory(
        output / "features" / "visual_evidence"
    )
    visual_root = ensure_private_directory(output / "features" / "clip")
    manifest_path = output / "feature_manifest.json"
    failure_records_path = output / "feature_failures.json"
    generated = generated_at_utc or _utc_now()
    media_manifest_digest = _canonical_sha256(media_manifest)

    previous_manifest: dict[str, Any] | None = None
    previous_records_by_id: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("existing TeachObs feature manifest is unsafe")
        previous_manifest = read_json(manifest_path)
        previous_rows = (
            previous_manifest.get("lessons")
            if isinstance(previous_manifest, dict)
            else None
        )
        if (
            not isinstance(previous_manifest, dict)
            or previous_manifest.get("schema") != FEATURE_SCHEMA
            or previous_manifest.get("plan_sha256") != plan["plan_sha256"]
            or not isinstance(previous_rows, list)
        ):
            raise ValueError("existing TeachObs feature manifest binding differs")
        for previous_record in previous_rows:
            if not isinstance(previous_record, dict):
                raise ValueError("existing TeachObs feature record is invalid")
            previous_id = _safe_lesson_id(previous_record.get("lesson_id"))
            if previous_id in previous_records_by_id:
                raise ValueError("existing TeachObs feature manifest has duplicates")
            previous_records_by_id[previous_id] = previous_record

    lesson_order = {
        item["lesson_id"]: index for index, item in enumerate(lessons)
    }
    lessons_by_id = {item["lesson_id"]: item for item in lessons}
    records_by_id: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}

    def persist_failure_records() -> None:
        if not failures:
            if failure_records_path.is_symlink() or failure_records_path.is_file():
                failure_records_path.unlink()
            elif failure_records_path.exists():
                raise ValueError("TeachObs feature-failure path is not a regular file")
            return
        ordered_failures = sorted(
            failures.values(), key=lambda row: lesson_order[row["lesson_id"]]
        )
        write_json(
            failure_records_path,
            {
                "schema": FEATURE_FAILURE_SCHEMA,
                "dataset_id": DATASET_ID,
                "private_artifact": True,
                "public_release_authorized": False,
                "generated_at_utc": generated,
                "plan_sha256": plan["plan_sha256"],
                "media_manifest_sha256": media_manifest_digest,
                "failure_count": len(ordered_failures),
                "failures": ordered_failures,
                "privacy": {
                    "contains_lesson_ids": True,
                    "contains_exception_messages": False,
                    "contains_transcript_or_ocr_text": False,
                    "safe_to_publish": False,
                },
                "claim_boundary": {
                    "reasons_are_sanitized_stage_summaries": True,
                    "raw_exception_messages_persisted": False,
                    "successful_peer_artifacts_retained": True,
                },
            },
        )
        ensure_private_file(failure_records_path)

    def manifest_value() -> dict[str, Any]:
        ordered_records = sorted(
            records_by_id.values(),
            key=lambda row: lesson_order[row["lesson_id"]],
        )
        all_lessons_complete = len(ordered_records) == len(lessons) and not failures
        return {
            "schema": FEATURE_SCHEMA,
            "dataset_id": DATASET_ID,
            "private_artifact": True,
            "public_release_authorized": False,
            "generated_at_utc": generated,
            "plan_sha256": plan["plan_sha256"],
            "media_manifest_sha256": media_manifest_digest,
            "lesson_count": len(ordered_records),
            "expected_lesson_count": len(lessons),
            "scene_count": sum(row["scene_count"] for row in ordered_records),
            "complete": all_lessons_complete and clip_model is None,
            "failed_lesson_count": len(failures),
            "failed_lesson_ids": sorted(
                failures, key=lambda lesson_id: lesson_order[lesson_id]
            ),
            "failure_records_path": (
                str(failure_records_path.relative_to(output)) if failures else None
            ),
            "failure_records_sha256": (
                file_sha256(failure_records_path) if failures else None
            ),
            "failure_reasons_are_sanitized": True,
            "partial_success_artifacts_are_resumable": True,
            "audio_statistics_included": include_audio_statistics,
            "visual_evidence_included": include_visual_evidence,
            "ocr_requested": include_visual_evidence and use_ocr,
            "clip_embeddings_requested": clip_model is not None,
            "clip_embeddings_included": False,
            "clip_embeddings_complete_for_all_lessons": False,
            "execution_parallelism": {
                "feature_jobs": feature_jobs,
                "visual_jobs_per_lesson": visual_jobs,
                "ocr_jobs_per_lesson": ocr_jobs if use_ocr else 0,
                "maximum_nested_worker_product": feature_jobs * inner_jobs,
                "combined_clip_runs_after_all_lessons": True,
                "combined_clip_accepts_successful_lesson_subset": True,
                "feature_manifest_single_coordinator_writer": True,
            },
            "lessons": ordered_records,
            "privacy": {
                "contains_lesson_ids": True,
                "references_private_frames": True,
                "contains_audio_derived_rows": include_audio_statistics,
                "contains_private_ocr_text": include_visual_evidence and use_ocr,
                "contains_visual_embeddings": False,
                "contains_transcript_text": False,
                "references_private_failure_records": bool(failures),
                "contains_raw_failure_messages": False,
                "safe_to_publish": False,
            },
            "claim_boundary": {
                "full_scene_midpoint_frames_extracted": all_lessons_complete,
                "audio_statistics_computed": (
                    all_lessons_complete and include_audio_statistics
                ),
                "image_metrics_computed": (
                    all_lessons_complete and include_visual_evidence
                ),
                "adjacent_visual_events_inferred": (
                    all_lessons_complete and include_visual_evidence
                ),
                "ocr_requested": include_visual_evidence and use_ocr,
                "clip_visual_embeddings_computed": False,
                "clip_visual_embeddings_complete_for_all_lessons": False,
                "clip_visual_embeddings_partial_success_only": False,
                "recognition_accuracy_established": False,
                "multimodal_gain_established": False,
            },
        }

    def process_lesson(item: dict[str, Any]) -> dict[str, Any]:
        lesson_id = item["lesson_id"]
        media_record = media_by_id.get(lesson_id)
        if media_record is None:
            raise _TeachObsFeatureStageFailure(
                "media_binding",
                ValueError(f"TeachObs media manifest is incomplete for {lesson_id}"),
            )
        media_path = output / media_record["media_path"]
        media_digest = media_record["media_sha256"]
        media_duration = float(media_record["media_probe"]["duration_seconds"])
        task_path = frames_root / lesson_id / "task.json"
        evidence_path = visual_evidence_root / f"{lesson_id}.json"
        obsolete_evidence_binding: dict[str, str] | None = None
        if include_visual_evidence and evidence_path.exists():
            prior_record = previous_records_by_id.get(lesson_id)
            expected_evidence_sha256: str | None = None
            if prior_record is not None:
                if (
                    prior_record.get("media_sha256") != media_digest
                    or prior_record.get("scene_manifest_sha256")
                    != item["scene_manifest_sha256"]
                    or prior_record.get("visual_evidence_path")
                    != str(evidence_path.relative_to(output))
                ):
                    raise _TeachObsFeatureStageFailure(
                        "visual_evidence",
                        ValueError("previous visual-evidence record binding differs"),
                    )
                expected_evidence_sha256 = prior_record.get(
                    "visual_evidence_sha256"
                )
                if not isinstance(expected_evidence_sha256, str):
                    raise _TeachObsFeatureStageFailure(
                        "visual_evidence",
                        ValueError("previous visual-evidence hash is missing"),
                    )
            visual_config = _visual_evidence_configuration(
                use_ocr=use_ocr,
                ocr_language=ocr_language,
                minimum_word_confidence=minimum_ocr_word_confidence,
                scene_change_threshold=0.25,
                slide_change_threshold=0.45,
            )
            try:
                obsolete_evidence_binding = (
                    _validate_obsolete_visual_evidence_binding(
                        task_path,
                        evidence_path,
                        item=item,
                        media_sha256=media_digest,
                        media_duration_seconds=media_duration,
                        configuration=visual_config,
                        expected_evidence_sha256=expected_evidence_sha256,
                    )
                )
            except Exception as exc:
                raise _TeachObsFeatureStageFailure("visual_evidence", exc) from exc
        try:
            frame_result = extract_teachobs_midpoint_frames(
                media_path,
                item,
                frames_root,
                media_sha256=media_digest,
                media_duration_seconds=media_duration,
                ffmpeg_command=ffmpeg_command,
                timeout_seconds=frame_timeout_seconds,
                runner=frame_runner,
            )
        except Exception as exc:
            raise _TeachObsFeatureStageFailure("frame_extraction", exc) from exc
        task_path = Path(frame_result["task_path"])
        record: dict[str, Any] = {
            "lesson_id": lesson_id,
            "scene_count": item["scene_count"],
            "media_sha256": media_digest,
            "scene_manifest_sha256": item["scene_manifest_sha256"],
            "frame_task_path": str(task_path.relative_to(output)),
            "frame_task_sha256": file_sha256(task_path),
            "frames_reused": frame_result["reused"],
        }
        if include_visual_evidence:
            try:
                evidence_result = extract_teachobs_scene_visual_evidence(
                    task_path,
                    evidence_path,
                    use_ocr=use_ocr,
                    ocr_language=ocr_language,
                    minimum_word_confidence=minimum_ocr_word_confidence,
                    visual_jobs=visual_jobs,
                    ocr_jobs=ocr_jobs,
                    image_metric_extractor=image_metric_extractor,
                    ocr_extractor=ocr_extractor,
                    obsolete_binding=obsolete_evidence_binding,
                )
            except Exception as exc:
                raise _TeachObsFeatureStageFailure("visual_evidence", exc) from exc
            evidence = evidence_result["result"]
            record.update(
                {
                    "visual_evidence_path": str(evidence_path.relative_to(output)),
                    "visual_evidence_sha256": file_sha256(evidence_path),
                    "visual_evidence_reused": evidence_result["reused"],
                    "visual_evidence_replaced_obsolete_binding": (
                        evidence_result["replaced_obsolete_binding"]
                    ),
                    "visual_evidence_upgraded_legacy_event_counts": (
                        evidence_result["upgraded_legacy_event_counts"]
                    ),
                    "visual_event_count": evidence["event_count"],
                    "visual_event_type_counts": evidence["event_type_counts"],
                    "ocr_status_counts": evidence["ocr_status_counts"],
                }
            )
        if include_audio_statistics:
            audio_path = audio_root / f"{lesson_id}.json"
            try:
                audio_result = extract_teachobs_audio_statistics(
                    media_path,
                    item,
                    audio_path,
                    media_sha256=media_digest,
                    media_duration_seconds=media_duration,
                    ffmpeg_command=ffmpeg_command,
                    timeout_seconds=audio_timeout_seconds,
                    runner=audio_runner,
                )
            except Exception as exc:
                raise _TeachObsFeatureStageFailure("audio_statistics", exc) from exc
            record.update(
                {
                    "audio_feature_path": str(audio_path.relative_to(output)),
                    "audio_feature_sha256": file_sha256(audio_path),
                    "audio_features_reused": audio_result["reused"],
                    "audio_features_upgraded_legacy_cache": audio_result[
                        "upgraded_legacy_cache"
                    ],
                }
            )
        return record
    
    # Only this coordinator writes feature_manifest.json. Workers write distinct,
    # content-bound per-lesson artifacts, which remain reusable after a peer fails.
    persist_failure_records()
    write_json(manifest_path, manifest_value())
    with ThreadPoolExecutor(
        max_workers=feature_jobs, thread_name_prefix="teachobs-feature"
    ) as pool:
        futures = {pool.submit(process_lesson, item): item for item in lessons}
        for future in as_completed(futures):
            item = futures[future]
            lesson_id = item["lesson_id"]
            try:
                records_by_id[lesson_id] = future.result()
            except Exception as exc:  # noqa: BLE001 - aggregate after peers finish
                if isinstance(exc, _TeachObsFeatureStageFailure):
                    stage = exc.stage
                    exception_type = exc.cause_type
                else:
                    stage = "lesson_pipeline"
                    exception_type = _safe_exception_type(exc)
                failures[lesson_id] = {
                    "lesson_id": lesson_id,
                    "stage": stage,
                    "reason_code": f"private_{stage}_failed",
                    "safe_reason": _FEATURE_FAILURE_SAFE_REASONS[stage],
                    "exception_type": exception_type,
                    "raw_exception_message_persisted": False,
                }
            persist_failure_records()
            write_json(manifest_path, manifest_value())
    records = sorted(
        records_by_id.values(), key=lambda row: lesson_order[row["lesson_id"]]
    )
    if clip_model is not None and records:
        combined_frames: list[dict[str, Any]] = []
        media_bindings: list[dict[str, str]] = []
        scene_manifest_bindings: list[dict[str, str]] = []
        successful_lesson_ids = [record["lesson_id"] for record in records]
        failed_lesson_ids = sorted(
            failures, key=lambda lesson_id: lesson_order[lesson_id]
        )
        expected_lesson_ids = [item["lesson_id"] for item in lessons]
        for lesson_id in successful_lesson_ids:
            record = records_by_id[lesson_id]
            item = lessons_by_id[lesson_id]
            task_path = output / record["frame_task_path"]
            if (
                task_path.is_symlink()
                or not task_path.is_file()
                or file_sha256(task_path) != record["frame_task_sha256"]
            ):
                raise ValueError(
                    f"TeachObs CLIP frame-task binding differs for {lesson_id}"
                )
            task = _verify_frame_task(
                task_path,
                item=item,
                media_sha256=record["media_sha256"],
                media_duration_seconds=float(
                    media_by_id[lesson_id]["media_probe"]["duration_seconds"]
                ),
            )
            media_bindings.append(
                {"lesson_id": lesson_id, "media_sha256": record["media_sha256"]}
            )
            scene_manifest_bindings.append(
                {
                    "lesson_id": lesson_id,
                    "scene_manifest_sha256": record["scene_manifest_sha256"],
                }
            )
            for frame in task["frames"]:
                combined_frames.append(
                    {
                        **frame,
                        "path": f"{lesson_id}/{frame['path']}",
                    }
                )
        combined_task = {
            "schema": FRAME_TASK_SCHEMA,
            "video_id": "teachobs_combined_scene_midpoints",
            "plan_sha256": plan["plan_sha256"],
            "complete": not failures and len(records) == len(lessons),
            "lesson_count": len(records),
            "expected_lesson_count": len(lessons),
            "included_lesson_ids": successful_lesson_ids,
            "expected_lesson_ids": expected_lesson_ids,
            "failed_lesson_ids": failed_lesson_ids,
            "media_sha256": _canonical_sha256(media_bindings),
            "media_set": media_bindings,
            "scene_manifest_set_sha256": _canonical_sha256(
                scene_manifest_bindings
            ),
            "frame_count": len(combined_frames),
            "private_artifact": True,
            "public_release_authorized": False,
            "frames": combined_frames,
        }
        combined_task_identity = _canonical_sha256(combined_task)
        combined_task_path = (
            frames_root / f"combined_{combined_task_identity}.task.json"
        )
        write_json(combined_task_path, combined_task)
        combined_task_digest = file_sha256(combined_task_path)
        clip_execution_binding = _canonical_sha256(
            {
                "task_file_sha256": combined_task_digest,
                "model_path": str(Path(clip_model).resolve()),
                "source_revision": str(clip_source_revision),
                "device": clip_device,
                "batch_size": clip_batch_size,
            }
        )
        visual_path = visual_root / f"combined_{clip_execution_binding}.json"
        expected_clip_sha256: str | None = None
        task_relative = str(combined_task_path.relative_to(output))
        result_relative = str(visual_path.relative_to(output))
        if (
            previous_manifest is not None
            and previous_manifest.get("combined_clip_task_path") == task_relative
            and previous_manifest.get("combined_clip_task_sha256")
            == combined_task_digest
            and previous_manifest.get("combined_clip_result_path")
            == result_relative
        ):
            previous_result_digest = previous_manifest.get(
                "combined_clip_result_sha256"
            )
            if not isinstance(previous_result_digest, str):
                raise ValueError(
                    "existing TeachObs feature manifest lacks its CLIP result hash"
                )
            expected_clip_sha256 = previous_result_digest
        visual_result = extract_teachobs_clip_embeddings(
            combined_task_path,
            visual_path,
            model_path=clip_model,
            source_revision=str(clip_source_revision),
            device=clip_device,
            batch_size=clip_batch_size,
            expected_output_sha256=expected_clip_sha256,
        )
        combined_result = visual_result["result"]
        combined_task_manifest_digest = _canonical_sha256(combined_task)
        combined_provenance = combined_result.get("model_provenance")
        if not isinstance(combined_provenance, dict):
            raise RuntimeError("combined TeachObs CLIP result has no model provenance")
        combined_complete = not failures and len(records) == len(lessons)
        manifest = read_json(manifest_path)
        manifest.update(
            {
                "complete": combined_complete,
                "clip_embeddings_included": True,
                "clip_embeddings_complete_for_all_lessons": combined_complete,
                "combined_clip_complete": combined_complete,
                "combined_clip_lesson_count": len(records),
                "combined_clip_expected_lesson_count": len(lessons),
                "combined_clip_lesson_ids": successful_lesson_ids,
                "combined_clip_failed_lesson_ids": failed_lesson_ids,
                "combined_clip_task_path": str(
                    combined_task_path.relative_to(output)
                ),
                "combined_clip_task_sha256": combined_task_digest,
                "combined_clip_result_path": str(visual_path.relative_to(output)),
                "combined_clip_result_sha256": file_sha256(visual_path),
                "combined_clip_frame_count": len(combined_frames),
                "combined_clip_reused": visual_result["reused"],
                "combined_clip_media_set_sha256": combined_task["media_sha256"],
                "combined_clip_scene_manifest_set_sha256": combined_task[
                    "scene_manifest_set_sha256"
                ],
                "combined_clip_task_manifest_sha256": (
                    combined_task_manifest_digest
                ),
                "combined_clip_model_provenance_sha256": _canonical_sha256(
                    combined_provenance
                ),
            }
        )
        manifest["privacy"]["contains_visual_embeddings"] = True
        manifest["claim_boundary"]["clip_visual_embeddings_computed"] = True
        manifest["claim_boundary"][
            "clip_visual_embeddings_complete_for_all_lessons"
        ] = combined_complete
        manifest["claim_boundary"][
            "clip_visual_embeddings_partial_success_only"
        ] = not combined_complete
        write_json(manifest_path, manifest)
    if failures:
        failed_ids = sorted(failures, key=lambda value: lesson_order[value])
        raise RuntimeError(
            "TeachObs feature extraction failed for "
            f"{len(failed_ids)} lesson(s): {', '.join(failed_ids)}; "
            "successful independent artifacts were retained for resume"
        )
    return {
        "manifest": read_json(manifest_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def prepare_teachobs_media_dataset(
    repository_root: str | Path,
    output_directory: str | Path,
    *,
    lesson_ids: Sequence[str] | None = None,
    acquisition_receipt_path: str | Path | None = None,
    acknowledge_source_terms: bool = False,
    source_override_manifest_path: str | Path | None = None,
    acknowledge_override_source_terms: bool = False,
    dry_run: bool = False,
    include_audio_statistics: bool = True,
    include_visual_evidence: bool = True,
    use_ocr: bool = True,
    ocr_language: str = "eng",
    minimum_ocr_word_confidence: float = 35.0,
    clip_model: str | Path | None = None,
    clip_source_revision: str | None = None,
    clip_device: str = "cpu",
    clip_batch_size: int = 32,
    yt_dlp_command: Sequence[str] | str | None = None,
    js_runtime: str | None = None,
    cookies_from_browser: str | None = None,
    yt_dlp_direct: bool = False,
    yt_dlp_impersonate: str | None = None,
    ffmpeg_command: str = "ffmpeg",
    ffprobe_command: str = "ffprobe",
    downloader: Callable[..., dict[str, Any]] = _download_with_ytdlp,
    media_probe: Callable[..., dict[str, Any]] = probe_local_media,
    command_runner: Callable[..., Any] = subprocess.run,
    frame_runner: Callable[..., Any] = subprocess.run,
    audio_runner: Callable[..., Any] = subprocess.run,
    download_jobs: int = 1,
    feature_jobs: int = 2,
    visual_jobs: int = 1,
    ocr_jobs: int = 1,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """End-to-end private preparation with a side-effect-free dry-run mode."""

    cookie_spec, browser_family = validate_teachobs_cookies_from_browser(
        cookies_from_browser
    )
    direct, impersonation_target = validate_teachobs_ytdlp_transport(
        direct=yt_dlp_direct,
        impersonate=yt_dlp_impersonate,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 8
        for value in (feature_jobs, visual_jobs, ocr_jobs)
    ):
        raise ValueError("TeachObs feature/visual/OCR jobs must each be in 1-8")
    active_inner_jobs = (
        max(visual_jobs, ocr_jobs if use_ocr else 1)
        if include_visual_evidence
        else 1
    )
    if feature_jobs * active_inner_jobs > 8:
        raise ValueError(
            "TeachObs feature_jobs * active inner visual/OCR jobs must not exceed 8"
        )
    if not dry_run and not acknowledge_source_terms:
        raise ValueError(
            "TeachObs media preparation requires acknowledge_source_terms=True"
        )
    plan = build_teachobs_media_plan(
        repository_root,
        lesson_ids=lesson_ids,
        acquisition_receipt_path=acquisition_receipt_path,
    )
    override_context = load_teachobs_source_override_manifest(
        source_override_manifest_path,
        plan=plan,
    )
    if (
        not dry_run
        and override_context["present"]
        and not acknowledge_override_source_terms
    ):
        raise ValueError(
            "TeachObs media preparation with a mirror requires "
            "acknowledge_override_source_terms=True in addition to the canonical "
            "source terms acknowledgement"
        )
    preflight = preflight_teachobs_media_tools(
        yt_dlp_command=yt_dlp_command,
        ffmpeg_command=ffmpeg_command,
        ffprobe_command=ffprobe_command,
        clip_model=clip_model,
        clip_device=clip_device,
    )
    if dry_run:
        return {
            "mode": "dry_run",
            "output_created": False,
            "plan": plan,
            "preflight": preflight,
            "source_terms_acknowledged": acknowledge_source_terms,
            "source_override_manifest_present": override_context["present"],
            "source_override_count": len(
                override_context["overrides_by_lesson"]
            ),
            "override_source_terms_acknowledged": (
                acknowledge_override_source_terms
            ),
            "override_acknowledgement_required_for_execution": override_context[
                "present"
            ],
            **(
                {
                    "browser_cookie_credentials_enabled": True,
                    "browser_family": browser_family,
                    "browser_profile_recorded": False,
                }
                if cookie_spec is not None
                else {}
            ),
            **(
                {
                    "yt_dlp_direct_enabled": direct,
                    "yt_dlp_impersonation_target": impersonation_target,
                    "proxy_url_recorded": False,
                }
                if direct or impersonation_target is not None
                else {}
            ),
            "acknowledgement_required_for_execution": True,
        }
    if not preflight["ready_with_requested_clip"]:
        raise RuntimeError(f"TeachObs media preflight failed: {preflight}")
    media_result = download_teachobs_media(
        plan,
        output_directory,
        acknowledge_source_terms=True,
        source_override_manifest_path=source_override_manifest_path,
        acknowledge_override_source_terms=acknowledge_override_source_terms,
        yt_dlp_command=yt_dlp_command,
        js_runtime=js_runtime,
        cookies_from_browser=cookie_spec,
        yt_dlp_direct=direct,
        yt_dlp_impersonate=impersonation_target,
        ffprobe_command=ffprobe_command,
        downloader=downloader,
        media_probe=media_probe,
        runner=command_runner,
        jobs=download_jobs,
        generated_at_utc=generated_at_utc,
    )
    feature_result = extract_teachobs_multimodal_features(
        plan,
        media_result["manifest"],
        output_directory,
        acknowledge_source_terms=True,
        include_audio_statistics=include_audio_statistics,
        include_visual_evidence=include_visual_evidence,
        use_ocr=use_ocr,
        ocr_language=ocr_language,
        minimum_ocr_word_confidence=minimum_ocr_word_confidence,
        clip_model=clip_model,
        clip_source_revision=clip_source_revision,
        clip_device=clip_device,
        clip_batch_size=clip_batch_size,
        ffmpeg_command=ffmpeg_command,
        frame_runner=frame_runner,
        audio_runner=audio_runner,
        feature_jobs=feature_jobs,
        visual_jobs=visual_jobs,
        ocr_jobs=ocr_jobs,
        generated_at_utc=generated_at_utc,
    )
    return {
        "mode": "executed_private_pipeline",
        "plan": plan,
        "preflight": preflight,
        "media": media_result,
        "features": feature_result,
    }


def _build_module_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or acquire the pinned TeachObs source videos into a private "
            "directory. No media or row-level artifact is public output."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser(
        "plan", description="Validate the pinned repository without downloading."
    )
    acquire_parser = subparsers.add_parser(
        "acquire", description="Download, hash, ffprobe, and privately receipt media."
    )
    for value in (plan_parser, acquire_parser):
        value.add_argument("--repository", type=Path, required=True)
        value.add_argument("--acquisition-receipt", type=Path)
        value.add_argument("--lesson-id", action="append", dest="lesson_ids")
    plan_parser.add_argument(
        "--private-plan-output",
        type=Path,
        help="Optional private 0600 JSON; includes source URLs and lesson IDs.",
    )
    acquire_parser.add_argument("--output", type=Path, required=True)
    acquire_parser.add_argument("--acknowledge-source-terms", action="store_true")
    acquire_parser.add_argument("--source-override-manifest", type=Path)
    acquire_parser.add_argument(
        "--acknowledge-override-source-terms", action="store_true"
    )
    acquire_parser.add_argument("--dry-run", action="store_true")
    acquire_parser.add_argument("--jobs", type=int, default=1)
    acquire_parser.add_argument("--yt-dlp-executable")
    acquire_parser.add_argument("--js-runtime")
    acquire_parser.add_argument("--cookies-from-browser")
    acquire_parser.add_argument("--yt-dlp-direct", action="store_true")
    acquire_parser.add_argument(
        "--yt-dlp-impersonate", choices=tuple(sorted(_YTDLP_IMPERSONATION_TARGETS))
    )
    acquire_parser.add_argument("--ffprobe-command", default="ffprobe")
    return parser


def _module_main(argv: Sequence[str] | None = None) -> int:
    args = _build_module_parser().parse_args(argv)
    plan = build_teachobs_media_plan(
        args.repository,
        lesson_ids=args.lesson_ids,
        acquisition_receipt_path=args.acquisition_receipt,
    )
    if args.command == "plan":
        if args.private_plan_output is not None:
            write_json(args.private_plan_output, plan)
        print(
            json.dumps(
                {
                    "mode": "private_plan_validated",
                    "plan_sha256": plan["plan_sha256"],
                    "lesson_count": plan["lesson_count"],
                    "scene_count": plan["scene_count"],
                    "reference_duration_seconds": plan[
                        "reference_duration_seconds"
                    ],
                    "private_plan_written": args.private_plan_output is not None,
                    "source_urls_or_lesson_ids_printed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    yt_dlp_command = (
        args.yt_dlp_executable if args.yt_dlp_executable is not None else None
    )
    if args.dry_run:
        result = download_teachobs_media(
            plan,
            args.output,
            acknowledge_source_terms=args.acknowledge_source_terms,
            source_override_manifest_path=args.source_override_manifest,
            acknowledge_override_source_terms=(
                args.acknowledge_override_source_terms
            ),
            dry_run=True,
            jobs=args.jobs,
            cookies_from_browser=args.cookies_from_browser,
            yt_dlp_direct=args.yt_dlp_direct,
            yt_dlp_impersonate=args.yt_dlp_impersonate,
        )
        result["preflight"] = preflight_teachobs_media_tools(
            yt_dlp_command=yt_dlp_command,
            ffprobe_command=args.ffprobe_command,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = download_teachobs_media(
        plan,
        args.output,
        acknowledge_source_terms=args.acknowledge_source_terms,
        source_override_manifest_path=args.source_override_manifest,
        acknowledge_override_source_terms=(
            args.acknowledge_override_source_terms
        ),
        yt_dlp_command=yt_dlp_command,
        js_runtime=args.js_runtime,
        cookies_from_browser=args.cookies_from_browser,
        yt_dlp_direct=args.yt_dlp_direct,
        yt_dlp_impersonate=args.yt_dlp_impersonate,
        ffprobe_command=args.ffprobe_command,
        jobs=args.jobs,
    )
    manifest = result["manifest"]
    print(
        json.dumps(
            {
                "mode": "private_media_acquisition",
                "manifest_path": result["manifest_path"],
                "manifest_sha256": result["manifest_sha256"],
                "selected_lesson_count": manifest["selected_lesson_count"],
                "selected_complete": manifest["selected_complete"],
                "source_urls_or_lesson_ids_printed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_module_main())
