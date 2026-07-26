"""Acquire and audit the public TeachObs human-annotation release.

The source classroom videos are deliberately outside this importer.  TeachObs
publishes its annotation assets under CC BY 4.0, while each linked source video
retains its own terms.  This module therefore imports only the released
metadata, coding scheme, fixed split, and consensus gold labels.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .io_utils import ensure_private_directory, write_json, write_text


TEACHOBS_REPOSITORY_ID = "codingchild2424/teacherOps"
TEACHOBS_ANONYMOUS_REPOSITORY_ID = "teacherOps-57D6"
TEACHOBS_SOURCE_COMMIT = "96c251ae09e79edd06a3a9bbaaa8b8f7fe99a15c"
TEACHOBS_RELEASE_VERSION = "0.1"
TEACHOBS_API_BASE = (
    "https://raw.githubusercontent.com/"
    f"{TEACHOBS_REPOSITORY_ID}/{TEACHOBS_SOURCE_COMMIT}"
)
TEACHOBS_DATASET_ID = "teachobs_v0_1_human_validated"
EXPECTED_LESSON_COUNT = 30
EXPECTED_TRAIN_LESSON_COUNT = 23
EXPECTED_TEST_LESSON_COUNT = 7
EXPECTED_SCENE_COUNT = 5158
EXPECTED_CODE_COUNT = 39
EXPECTED_VISUAL_CODE_COUNT = 20
EXPECTED_NONVISUAL_CODE_COUNT = 19
EXPECTED_TEST_IDS = {"S2", "S4", "S5", "S19", "S24", "S28", "S30"}

_STATIC_FILES = (
    "LICENSE-DATA",
    "README.md",
    "croissant.json",
    "data/lessons.csv",
    "data/track_a/README.md",
    "data/track_a/coding_scheme.json",
    "data/track_a/splits/train_ids.txt",
    "data/track_a/splits/test_ids.txt",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe TeachObs repository path: {value!r}")
    return path


def _source_url(relative_path: str) -> str:
    safe = _safe_relative_path(relative_path)
    encoded = "/".join(quote(part, safe="") for part in safe.parts)
    return f"{TEACHOBS_API_BASE}/{encoded}"


def _fetch_bytes(relative_path: str, *, timeout: int) -> tuple[bytes, str]:
    url = _source_url(relative_path)
    request = Request(
        url,
        headers={"User-Agent": "Teaching-Skill-Miner/1.2 TeachObs-audit"},
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS host
        final_url = response.geturl()
        parsed = urlparse(final_url)
        if parsed.scheme != "https" or parsed.hostname not in {
            "raw.githubusercontent.com",
            "github.com",
        }:
            raise ValueError(f"TeachObs download redirected to an untrusted host: {final_url}")
        payload = response.read()
    if not payload:
        raise ValueError(f"TeachObs source returned an empty file: {relative_path}")
    return payload, final_url


def _parse_lessons(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    expected_fields = {
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
    if not rows or set(rows[0]) != expected_fields:
        raise ValueError("TeachObs lessons.csv has an unexpected schema")
    if len(rows) != EXPECTED_LESSON_COUNT:
        raise ValueError(
            f"TeachObs lesson count mismatch: {len(rows)} != {EXPECTED_LESSON_COUNT}"
        )
    ids = [row["id"].strip() for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("TeachObs lessons.csv contains duplicate lesson ids")
    if set(ids) != {f"S{index}" for index in range(1, EXPECTED_LESSON_COUNT + 1)}:
        raise ValueError("TeachObs lessons.csv does not contain the expected S1-S30 ids")
    for row in rows:
        if row["split"] not in {"train", "test"}:
            raise ValueError(f"unsupported TeachObs split for {row['id']}")
        parsed = urlparse(row["youtube_url"])
        if parsed.scheme != "https" or parsed.hostname not in {
            "www.youtube.com",
            "youtube.com",
            "youtu.be",
        }:
            raise ValueError(f"invalid TeachObs source-video URL for {row['id']}")
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    if len(train) != EXPECTED_TRAIN_LESSON_COUNT or len(test) != EXPECTED_TEST_LESSON_COUNT:
        raise ValueError("TeachObs train/test lesson counts do not match the release")
    if {row["id"] for row in test} != EXPECTED_TEST_IDS:
        raise ValueError("TeachObs official test ids do not match the pinned release")
    return rows


def _parse_scheme(payload: bytes) -> tuple[list[str], dict[str, str], int]:
    scheme = json.loads(payload.decode("utf-8"))
    rows = scheme.get("codes")
    if scheme.get("n_codes") != EXPECTED_CODE_COUNT or not isinstance(rows, list):
        raise ValueError("TeachObs coding scheme does not declare 39 codes")
    names: list[str] = []
    groups: dict[str, str] = {}
    nonempty_definition_count = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("TeachObs coding scheme contains a non-object code")
        name = str(row.get("name", "")).strip()
        group = str(row.get("group", "")).strip()
        if not name or name in groups or group not in {"visual", "nonvisual"}:
            raise ValueError("TeachObs coding scheme has an invalid code entry")
        names.append(name)
        groups[name] = group
        nonempty_definition_count += bool(str(row.get("definition", "")).strip())
    if len(names) != EXPECTED_CODE_COUNT:
        raise ValueError("TeachObs coding scheme contains duplicate or missing codes")
    if sum(value == "visual" for value in groups.values()) != EXPECTED_VISUAL_CODE_COUNT:
        raise ValueError("TeachObs visual-code count mismatch")
    if sum(value == "nonvisual" for value in groups.values()) != EXPECTED_NONVISUAL_CODE_COUNT:
        raise ValueError("TeachObs nonvisual-code count mismatch")
    return names, groups, nonempty_definition_count


def _parse_id_list(payload: bytes) -> list[str]:
    return [line.strip() for line in payload.decode("utf-8-sig").splitlines() if line.strip()]


def _audit_gold_file(
    payload: bytes,
    *,
    expected_lesson_id: str,
    expected_codes: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid TeachObs gold JSON in {expected_lesson_id}, line {line_number}"
            ) from exc
        if row.get("lesson_id") != expected_lesson_id:
            raise ValueError(f"TeachObs gold lesson binding mismatch for {expected_lesson_id}")
        scene_no = row.get("scene_no")
        if isinstance(scene_no, bool) or not isinstance(scene_no, int) or scene_no < 1:
            raise ValueError(f"invalid TeachObs scene number for {expected_lesson_id}")
        if not isinstance(row.get("time"), str) or "-" not in row["time"]:
            raise ValueError(f"invalid TeachObs time range for {expected_lesson_id}")
        codes = row.get("codes")
        if not isinstance(codes, dict) or list(codes) != expected_codes:
            raise ValueError(f"TeachObs code vector mismatch for {expected_lesson_id}")
        if any(value not in {0, 1} or isinstance(value, bool) for value in codes.values()):
            raise ValueError(f"TeachObs labels must be integer 0/1 for {expected_lesson_id}")
        rows.append(row)
    expected_scene_numbers = list(range(1, len(rows) + 1))
    if [row["scene_no"] for row in rows] != expected_scene_numbers:
        raise ValueError(f"TeachObs scenes are not contiguous for {expected_lesson_id}")
    positives = {
        code: sum(int(row["codes"][code]) for row in rows) for code in expected_codes
    }
    return {
        "lesson_id": expected_lesson_id,
        "scene_count": len(rows),
        "positive_label_count": sum(positives.values()),
        "per_code_positive_count": positives,
    }


def download_and_audit_teachobs(
    output_dir: str | Path,
    *,
    acknowledge_source_terms: bool,
    timeout: int = 120,
) -> dict[str, Any]:
    """Download the released annotation assets and fail closed on drift."""

    if not acknowledge_source_terms:
        raise ValueError(
            "TeachObs import requires explicit acknowledgement of LICENSE-DATA and "
            "the separate source-video terms"
        )
    output = ensure_private_directory(output_dir).resolve()
    source_rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}

    def fetch(relative_path: str) -> bytes:
        payload, final_url = _fetch_bytes(relative_path, timeout=timeout)
        target = output.joinpath(*_safe_relative_path(relative_path).parts)
        ensure_private_directory(target.parent)
        write_text(target, payload.decode("utf-8"))
        actual = _sha256_file(target)
        expected = _sha256_bytes(payload)
        if actual != expected:
            raise ValueError(f"TeachObs local write hash mismatch: {relative_path}")
        payloads[relative_path] = payload
        source_rows.append(
            {
                "path": relative_path,
                "size_bytes": len(payload),
                "sha256": expected,
                "source_url": final_url,
            }
        )
        return payload

    for relative_path in _STATIC_FILES:
        fetch(relative_path)

    lessons = _parse_lessons(payloads["data/lessons.csv"])
    code_names, code_groups, nonempty_definition_count = _parse_scheme(
        payloads["data/track_a/coding_scheme.json"]
    )
    train_ids = _parse_id_list(payloads["data/track_a/splits/train_ids.txt"])
    test_ids = _parse_id_list(payloads["data/track_a/splits/test_ids.txt"])
    lesson_split_ids = {
        "train": [row["id"] for row in lessons if row["split"] == "train"],
        "test": [row["id"] for row in lessons if row["split"] == "test"],
    }
    if train_ids != lesson_split_ids["train"] or test_ids != lesson_split_ids["test"]:
        raise ValueError("TeachObs split files disagree with lessons.csv")

    gold_audits: list[dict[str, Any]] = []
    for split, ids in (("train", train_ids), ("test", test_ids)):
        for lesson_id in ids:
            relative_path = f"data/track_a/gold/{split}/{lesson_id}.jsonl"
            payload = fetch(relative_path)
            gold_audits.append(
                _audit_gold_file(
                    payload,
                    expected_lesson_id=lesson_id,
                    expected_codes=code_names,
                )
            )

    scene_count = sum(row["scene_count"] for row in gold_audits)
    if scene_count != EXPECTED_SCENE_COUNT:
        raise ValueError(
            f"TeachObs scene count mismatch: {scene_count} != {EXPECTED_SCENE_COUNT}"
        )
    license_text = payloads["LICENSE-DATA"].decode("utf-8")
    if "CC BY 4.0" not in license_text or "source video frames" not in license_text:
        raise ValueError("TeachObs data-license boundary is missing or changed")

    source_rows.sort(key=lambda row: row["path"])
    source_manifest = {
        "schema_version": "1.0",
        "artifact_kind": "teachobs_source_file_manifest",
        "dataset_id": TEACHOBS_DATASET_ID,
        "release_version": TEACHOBS_RELEASE_VERSION,
        "repository_id": TEACHOBS_REPOSITORY_ID,
        "source_commit": TEACHOBS_SOURCE_COMMIT,
        "retrieved_at_utc": _utc_now(),
        "files": source_rows,
    }
    source_manifest["file_set_sha256"] = _canonical_sha256(source_rows)
    source_manifest_path = write_json(output / "source_file_manifest.json", source_manifest)

    aggregate_positive_counts = {
        code: sum(row["per_code_positive_count"][code] for row in gold_audits)
        for code in code_names
    }
    audit = {
        "schema_version": "1.0",
        "artifact_kind": "teachobs_annotation_dataset_audit",
        "dataset_id": TEACHOBS_DATASET_ID,
        "release_version": TEACHOBS_RELEASE_VERSION,
        "repository_id": TEACHOBS_REPOSITORY_ID,
        "source_commit": TEACHOBS_SOURCE_COMMIT,
        "anonymous_repository_id_from_paper": TEACHOBS_ANONYMOUS_REPOSITORY_ID,
        "repository_linkage_status": (
            "author_owned_successor_inferred_not_cryptographically_linked_to_"
            "disconnected_anonymous_mirror"
        ),
        "import_complete": True,
        "confirmatory_validation_ready": False,
        "source_file_manifest_sha256": _sha256_file(source_manifest_path),
        "lesson_count": len(lessons),
        "train_lesson_count": len(train_ids),
        "test_lesson_count": len(test_ids),
        "scene_count": scene_count,
        "code_count": len(code_names),
        "visual_code_count": sum(value == "visual" for value in code_groups.values()),
        "nonvisual_code_count": sum(
            value == "nonvisual" for value in code_groups.values()
        ),
        "positive_label_count": sum(aggregate_positive_counts.values()),
        "per_code_positive_count": aggregate_positive_counts,
        "official_split_verified": True,
        "consensus_gold_schema_verified": True,
        "gold_status": "provisional_consensus_gold",
        "code_definition_count": nonempty_definition_count,
        "all_code_definitions_blank": nonempty_definition_count == 0,
        "annotation_protocol_from_source": {
            "independent_coder_count": 7,
            "all_scenes_annotated_by_all_coders": True,
            "gold_is_reliability_and_prevalence_aware_aggregation": True,
            "coder_level_files_in_imported_release": False,
            "coder_level_reliability_recomputable": False,
        },
        "license": {
            "annotation_assets": "CC BY 4.0",
            "source_video_terms_are_separate": True,
            "source_videos_or_frames_imported": False,
        },
        "claim_boundary": {
            "independent_human_consensus_labels_available": True,
            "independent_human_protocol_reported_by_source": True,
            "individual_coder_records_available": False,
            "inter_rater_reliability_independently_recomputed": False,
            "current_system_evaluated_on_labels": False,
            "confirmatory_multimodal_gain_established": False,
            "deployment_accuracy_established": False,
            "learner_effect_established": False,
        },
    }
    write_json(output / "dataset_audit.json", audit)
    return audit


def build_public_teachobs_receipt(
    audit: dict[str, Any],
    *,
    private_audit_path: str | Path,
) -> dict[str, Any]:
    """Build a content-free public receipt for the imported annotation release."""

    path = Path(private_audit_path).resolve()
    return {
        "schema_version": "1.0",
        "artifact_kind": "public_teachobs_annotation_receipt",
        "dataset_id": audit["dataset_id"],
        "release_version": audit["release_version"],
        "source_commit": audit["source_commit"],
        "private_audit_sha256": _sha256_file(path),
        "aggregate": {
            "lesson_count": audit["lesson_count"],
            "train_lesson_count": audit["train_lesson_count"],
            "test_lesson_count": audit["test_lesson_count"],
            "scene_count": audit["scene_count"],
            "code_count": audit["code_count"],
            "visual_code_count": audit["visual_code_count"],
            "nonvisual_code_count": audit["nonvisual_code_count"],
            "positive_label_count": audit["positive_label_count"],
        },
        "evidence": {
            "official_split_verified": audit["official_split_verified"],
            "consensus_gold_schema_verified": audit[
                "consensus_gold_schema_verified"
            ],
            "gold_status": audit["gold_status"],
            "independent_coder_count_from_source_protocol": audit[
                "annotation_protocol_from_source"
            ]["independent_coder_count"],
            "individual_coder_records_available": audit[
                "claim_boundary"
            ]["individual_coder_records_available"],
            "inter_rater_reliability_independently_recomputed": audit[
                "claim_boundary"
            ]["inter_rater_reliability_independently_recomputed"],
            "code_definition_count": audit["code_definition_count"],
            "annotation_asset_license": audit["license"]["annotation_assets"],
        },
        "content_exclusion": {
            "source_video_urls_included": False,
            "source_video_or_frame_content_included": False,
            "transcript_text_included": False,
            "scene_level_labels_included": False,
            "lesson_identifiers_included": False,
            "local_or_private_paths_included": False,
        },
        "claim_boundary": audit["claim_boundary"],
    }
