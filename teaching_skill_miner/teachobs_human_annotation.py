"""Blind, two-rater TeachObs scene annotation and agreement workflow.

Only TeachObs lesson identifiers, scene time intervals, and the public coding
scheme are read.  Consensus gold files, predictions, transcript bodies, frames,
and source-video bytes are intentionally outside this module.
"""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .io_utils import ensure_private_directory, ensure_private_file, read_json, write_json
from .teachobs import (
    EXPECTED_CODE_COUNT,
    EXPECTED_LESSON_COUNT,
    EXPECTED_NONVISUAL_CODE_COUNT,
    EXPECTED_SCENE_COUNT,
    EXPECTED_VISUAL_CODE_COUNT,
    TEACHOBS_DATASET_ID,
    TEACHOBS_SOURCE_COMMIT,
)


ASSIGNMENT_SCHEMA = "teaching_skill_miner.teachobs_blind_double_assignment.v1"
AGREEMENT_SCHEMA = "teaching_skill_miner.teachobs_double_annotation_agreement.v1"
PUBLIC_RECEIPT_SCHEMA = (
    "teaching_skill_miner.public_teachobs_double_annotation_receipt.v1"
)
OPERATIONAL_CODEBOOK_SCHEMA = (
    "teaching_skill_miner.teachobs_operational_codebook.v1"
)
EXPECTED_LESSONS_SHA256 = (
    "fe607eb503bea8f3175941232ff8f5ecbfaa8f3135e70436f4ce51f154217f48"
)
EXPECTED_CODING_SCHEME_SHA256 = (
    "ff90ef2abdd5103c828637b31814491e7656eb147888fa5f6ad3bb5d12e3386e"
)
EXPECTED_SCENE_MANIFEST_SET_SHA256 = (
    "9d29a30973696deb7bf5f993cd2ed1b193547efcd5d40187b86ea7be1ee190b8"
)

ASSIGNMENT_SLOTS = ("A", "B")
DEFAULT_SEEDS = {"A": 1729, "B": 2718}
LABEL_PREFIX = "label::"
ATTESTATION_VALUE = "YES"
RATER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}")
CODEBOOK_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}")
LESSON_ID_RE = re.compile(r"S(?:[1-9]|[12][0-9]|30)")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

IMMUTABLE_COLUMNS = (
    "assignment_slot",
    "assignment_position",
    "item_token",
    "start_seconds",
    "end_seconds",
    "private_media_reference",
    "operational_codebook_version",
    "operational_codebook_sha256",
)
ATTESTATION_COLUMNS = (
    "rater_id",
    "completed_by_real_human",
    "independent_without_gold_or_predictions",
    "signed_name",
    "attested_at_utc",
)
ADJUDICATION_COLUMNS = (
    "disagreement_no",
    "item_token",
    "start_seconds",
    "end_seconds",
    "private_media_reference",
    "code_name",
    "rater_A_value",
    "rater_B_value",
    "adjudicated_value",
    "adjudicator_id",
    "third_party_independence_attested",
    "signed_name",
    "attested_at_utc",
    "reason",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _safe_private_relative_path(value: str, *, field: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise ValueError(f"unsafe {field}: {value!r}")
    return path.as_posix()


def _safe_lesson_id(value: Any) -> str:
    lesson_id = str(value).strip()
    if not LESSON_ID_RE.fullmatch(lesson_id):
        raise ValueError(f"invalid TeachObs lesson id: {value!r}")
    return lesson_id


def _lesson_sort_key(value: str) -> int:
    return int(value[1:])


def _format_seconds(value: float) -> str:
    return f"{value:.3f}"


def _read_lesson_ids(root: Path) -> tuple[list[str], str]:
    path = root / "data" / "lessons.csv"
    if path.is_symlink() or not path.is_file():
        raise ValueError("TeachObs lessons.csv is missing or unsafe")
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    if digest != EXPECTED_LESSONS_SHA256:
        raise ValueError("TeachObs lessons.csv differs from the pinned source revision")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or []):
            raise ValueError("TeachObs lessons.csv has no id column")
        ids = [_safe_lesson_id(row.get("id")) for row in reader]
    expected = {f"S{index}" for index in range(1, EXPECTED_LESSON_COUNT + 1)}
    if len(ids) != EXPECTED_LESSON_COUNT or set(ids) != expected:
        raise ValueError("TeachObs lessons.csv does not contain unique S1-S30 ids")
    return sorted(ids, key=_lesson_sort_key), digest


def _read_codebook(root: Path) -> tuple[list[dict[str, str]], str, int]:
    path = root / "data" / "track_a" / "coding_scheme.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("TeachObs coding scheme is missing or unsafe")
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    if digest != EXPECTED_CODING_SCHEME_SHA256:
        raise ValueError("TeachObs coding scheme differs from the pinned source revision")
    value = json.loads(payload)
    rows = value.get("codes") if isinstance(value, dict) else None
    if value.get("n_codes") != EXPECTED_CODE_COUNT or not isinstance(rows, list):
        raise ValueError("TeachObs coding scheme does not declare 39 codes")
    codes: list[dict[str, str]] = []
    names: set[str] = set()
    nonempty_definition_count = 0
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("TeachObs coding scheme contains a non-object entry")
        name = str(row.get("name", "")).strip()
        group = str(row.get("group", "")).strip()
        if (
            not name
            or name in names
            or group not in {"visual", "nonvisual"}
            or "\n" in name
            or "\r" in name
        ):
            raise ValueError("TeachObs coding scheme contains an invalid code")
        names.add(name)
        codes.append({"code_id": f"code_{index:02d}", "name": name, "group": group})
        nonempty_definition_count += bool(str(row.get("definition", "")).strip())
    if len(codes) != EXPECTED_CODE_COUNT:
        raise ValueError("TeachObs coding scheme code count differs")
    if sum(row["group"] == "visual" for row in codes) != EXPECTED_VISUAL_CODE_COUNT:
        raise ValueError("TeachObs visual code count differs")
    if (
        sum(row["group"] == "nonvisual" for row in codes)
        != EXPECTED_NONVISUAL_CODE_COUNT
    ):
        raise ValueError("TeachObs nonvisual code count differs")
    return codes, digest, nonempty_definition_count


def _nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"operational codebook has invalid {field}")
    text = value.strip()
    if not text or len(text) > 10_000 or any(ord(character) < 32 for character in text):
        raise ValueError(f"operational codebook has invalid {field}")
    return text


def _nonempty_text_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"operational codebook {field} must be a non-empty array")
    return [_nonempty_text(item, field=field) for item in value]


def _operational_codebook_template(codes: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "schema": OPERATIONAL_CODEBOOK_SCHEMA,
        "version": "",
        "codes": [
            {
                "name": row["name"],
                "definition": "",
                "inclusion_criteria": [],
                "exclusion_criteria": [],
                "positive_examples": [],
                "negative_examples": [],
            }
            for row in codes
        ],
    }


def _load_operational_codebook(
    path: Path,
    codes: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("operational codebook is missing or unsafe")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("operational codebook is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != OPERATIONAL_CODEBOOK_SCHEMA:
        raise ValueError("operational codebook has an unsupported schema")
    version = str(value.get("version", "")).strip()
    if not CODEBOOK_VERSION_RE.fullmatch(version):
        raise ValueError("operational codebook has an invalid version")
    rows = value.get("codes")
    if not isinstance(rows, list) or len(rows) != len(codes):
        raise ValueError("operational codebook must contain exactly 39 code entries")
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("operational codebook contains a non-object code entry")
        name = str(raw.get("name", "")).strip()
        if not name or name in by_name:
            raise ValueError("operational codebook contains a missing/duplicate code name")
        by_name[name] = raw
    expected_names = [str(row["name"]) for row in codes]
    if set(by_name) != set(expected_names):
        raise ValueError("operational codebook names differ from the official 39 codes")

    normalized_rows: list[dict[str, Any]] = []
    for code in codes:
        name = str(code["name"])
        raw = by_name[name]
        normalized_rows.append(
            {
                "code_id": code["code_id"],
                "name": name,
                "group": code["group"],
                "definition": _nonempty_text(
                    raw.get("definition"), field=f"{name}.definition"
                ),
                "inclusion_criteria": _nonempty_text_list(
                    raw.get("inclusion_criteria"),
                    field=f"{name}.inclusion_criteria",
                ),
                "exclusion_criteria": _nonempty_text_list(
                    raw.get("exclusion_criteria"),
                    field=f"{name}.exclusion_criteria",
                ),
                "positive_examples": _nonempty_text_list(
                    raw.get("positive_examples"),
                    field=f"{name}.positive_examples",
                ),
                "negative_examples": _nonempty_text_list(
                    raw.get("negative_examples"),
                    field=f"{name}.negative_examples",
                ),
            }
        )
    normalized = {
        "schema": OPERATIONAL_CODEBOOK_SCHEMA,
        "version": version,
        "codes": normalized_rows,
    }
    return normalized, _sha256_bytes(payload), _canonical_sha256(normalized)


def _read_scene_metadata(
    root: Path,
    lesson_ids: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    by_lesson: dict[str, list[dict[str, Any]]] = {}
    manifest_receipts: list[dict[str, Any]] = []
    for lesson_id in lesson_ids:
        path = root / "data" / "scenes" / lesson_id / "manifest.jsonl"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"TeachObs scene manifest is missing or unsafe: {lesson_id}")
        payload = path.read_bytes()
        manifest_receipts.append(
            {
                "lesson_id": lesson_id,
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
        scenes: list[dict[str, Any]] = []
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid TeachObs scene metadata: {lesson_id}, line {line_number}"
                ) from exc
            if not isinstance(row, dict) or row.get("id") != lesson_id:
                raise ValueError(f"TeachObs scene lesson binding differs: {lesson_id}")
            scene_no = row.get("scene_no")
            if (
                isinstance(scene_no, bool)
                or not isinstance(scene_no, int)
                or scene_no != len(scenes) + 1
            ):
                raise ValueError(f"TeachObs scenes are not contiguous: {lesson_id}")
            try:
                start = float(row["start"])
                end = float(row["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid TeachObs scene interval: {lesson_id}") from exc
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end <= start
                or (scenes and start < float(scenes[-1]["end_seconds"]) - 0.001)
            ):
                raise ValueError(f"invalid TeachObs scene interval: {lesson_id}")
            scenes.append(
                {
                    "lesson_id": lesson_id,
                    "scene_no": scene_no,
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                }
            )
        if not scenes:
            raise ValueError(f"TeachObs scene manifest is empty: {lesson_id}")
        by_lesson[lesson_id] = scenes
    if sum(len(rows) for rows in by_lesson.values()) != EXPECTED_SCENE_COUNT:
        raise ValueError("TeachObs scene metadata does not contain 5158 scenes")
    set_digest = _canonical_sha256(manifest_receipts)
    if set_digest != EXPECTED_SCENE_MANIFEST_SET_SHA256:
        raise ValueError("TeachObs scene manifests differ from the pinned source revision")
    return by_lesson, set_digest


def _item_token(row: Mapping[str, Any]) -> str:
    identity = (
        f"{ASSIGNMENT_SCHEMA}\0{TEACHOBS_SOURCE_COMMIT}\0"
        f"{row['lesson_id']}\0{row['scene_no']}\0"
        f"{_format_seconds(float(row['start_seconds']))}\0"
        f"{_format_seconds(float(row['end_seconds']))}"
    )
    return _sha256_bytes(identity.encode("utf-8"))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return ensure_private_file(path)


def _immutable_assignment_digest(rows: Sequence[Mapping[str, str]]) -> str:
    return _canonical_sha256(
        [{field: row[field] for field in IMMUTABLE_COLUMNS} for row in rows]
    )


def _assignment_fieldnames(codes: Sequence[Mapping[str, str]]) -> list[str]:
    return [
        *IMMUTABLE_COLUMNS,
        *ATTESTATION_COLUMNS,
        *(f"{LABEL_PREFIX}{row['name']}" for row in codes),
    ]


def prepare_teachobs_double_annotation(
    repository_root: str | Path,
    output_directory: str | Path,
    *,
    lesson_ids: Sequence[str] | None = None,
    media_root: str | Path | None = None,
    media_reference_prefix: str = "videos",
    operational_codebook_path: str | Path | None = None,
    seed_a: int = DEFAULT_SEEDS["A"],
    seed_b: int = DEFAULT_SEEDS["B"],
    require_media: bool = False,
) -> dict[str, Any]:
    """Create two blinded, differently ordered private assignment CSV files."""

    if isinstance(seed_a, bool) or not isinstance(seed_a, int):
        raise ValueError("seed_a must be an integer")
    if isinstance(seed_b, bool) or not isinstance(seed_b, int):
        raise ValueError("seed_b must be an integer")
    if seed_a == seed_b:
        raise ValueError("A/B assignment seeds must differ")
    prefix = _safe_private_relative_path(
        media_reference_prefix, field="media reference prefix"
    ).rstrip("/")
    root = Path(repository_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("TeachObs repository root is missing or unsafe")

    all_lesson_ids, lessons_digest = _read_lesson_ids(root)
    codes, codebook_digest, official_definition_count = _read_codebook(root)
    scene_metadata, scene_set_digest = _read_scene_metadata(root, all_lesson_ids)
    if official_definition_count != 0:
        raise ValueError(
            "pinned TeachObs release unexpectedly contains operational definitions"
        )
    if operational_codebook_path is None:
        operational_codebook = _operational_codebook_template(codes)
        operational_source_digest = None
        operational_normalized_digest = None
        operational_version = ""
        operational_complete = False
    else:
        (
            operational_codebook,
            operational_source_digest,
            operational_normalized_digest,
        ) = _load_operational_codebook(Path(operational_codebook_path), codes)
        operational_version = str(operational_codebook["version"])
        operational_complete = True
    if lesson_ids is None:
        selected_ids = all_lesson_ids
    else:
        selected_ids = [_safe_lesson_id(value) for value in lesson_ids]
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("lesson selection must be non-empty and unique")
        if not set(selected_ids).issubset(all_lesson_ids):
            raise ValueError("lesson selection contains an unknown lesson")
        selected_ids.sort(key=_lesson_sort_key)

    resolved_media_root = Path(media_root).resolve() if media_root is not None else None
    if resolved_media_root is not None and (
        resolved_media_root.is_symlink() or not resolved_media_root.is_dir()
    ):
        raise ValueError("media root is missing or unsafe")

    media_available_by_lesson: dict[str, bool] = {}
    for lesson_id in selected_ids:
        reference = _safe_private_relative_path(
            f"{prefix}/{lesson_id}.mp4", field="private media reference"
        )
        if resolved_media_root is None:
            available = False
        else:
            target = resolved_media_root / reference
            try:
                target.resolve().relative_to(resolved_media_root)
            except ValueError as exc:
                raise ValueError("private media reference escapes its root") from exc
            available = target.is_file() and not target.is_symlink()
        media_available_by_lesson[lesson_id] = available
    if require_media and not all(media_available_by_lesson.values()):
        missing_count = sum(not value for value in media_available_by_lesson.values())
        raise ValueError(f"{missing_count} selected private media files are unavailable")

    items: list[dict[str, Any]] = []
    for lesson_id in selected_ids:
        reference = f"{prefix}/{lesson_id}.mp4"
        for scene in scene_metadata[lesson_id]:
            item = {
                **scene,
                "private_media_reference": reference,
            }
            item["item_token"] = _item_token(item)
            items.append(item)
    if len(items) < 2:
        raise ValueError("double-blind ordering requires at least two scene items")
    tokens = [str(item["item_token"]) for item in items]
    if len(set(tokens)) != len(tokens):
        raise ValueError("TeachObs item token collision")

    output = ensure_private_directory(output_directory).resolve()
    fieldnames = _assignment_fieldnames(codes)
    item_by_token = {str(item["item_token"]): item for item in items}
    assignment_records: dict[str, dict[str, Any]] = {}
    orders: dict[str, list[str]] = {}
    for slot, seed in (("A", seed_a), ("B", seed_b)):
        order = list(tokens)
        random.Random(seed).shuffle(order)
        orders[slot] = order
    if orders["A"] == orders["B"]:
        orders["B"] = orders["B"][1:] + orders["B"][:1]

    for slot, seed in (("A", seed_a), ("B", seed_b)):
        rows: list[dict[str, str]] = []
        for position, token in enumerate(orders[slot], 1):
            item = item_by_token[token]
            row = {
                "assignment_slot": slot,
                "assignment_position": str(position),
                "item_token": token,
                "start_seconds": _format_seconds(item["start_seconds"]),
                "end_seconds": _format_seconds(item["end_seconds"]),
                "private_media_reference": str(item["private_media_reference"]),
                "operational_codebook_version": operational_version,
                "operational_codebook_sha256": operational_normalized_digest or "",
                **{field: "" for field in ATTESTATION_COLUMNS},
                **{f"{LABEL_PREFIX}{code['name']}": "" for code in codes},
            }
            rows.append(row)
        target = _write_csv(output / f"assignment_{slot}.csv", fieldnames, rows)
        assignment_records[slot] = {
            "seed": seed,
            "row_count": len(rows),
            "template_file": target.name,
            "template_file_sha256": _sha256_file(target),
            "immutable_row_sequence_sha256": _immutable_assignment_digest(rows),
            "ordered_item_tokens": orders[slot],
        }

    codebook_rows = [
        {
            "code_id": row["code_id"],
            "code_name": row["name"],
            "modality_group": row["group"],
            "label_value_0": "absent",
            "label_value_1": "present",
        }
        for row in codes
    ]
    codebook_target = _write_csv(
        output / "codebook.csv",
        (
            "code_id",
            "code_name",
            "modality_group",
            "label_value_0",
            "label_value_1",
        ),
        codebook_rows,
    )
    operational_filename = (
        "operational_codebook.json"
        if operational_complete
        else "operational_codebook_template.json"
    )
    operational_target = write_json(output / operational_filename, operational_codebook)

    manifest: dict[str, Any] = {
        "schema": ASSIGNMENT_SCHEMA,
        "dataset_id": TEACHOBS_DATASET_ID,
        "source_revision": TEACHOBS_SOURCE_COMMIT,
        "private_artifact": True,
        "public_release_authorized": False,
        "source_metadata_binding": {
            "lessons_csv_sha256": lessons_digest,
            "coding_scheme_sha256": codebook_digest,
            "scene_manifest_set_sha256": scene_set_digest,
            "official_lesson_count": EXPECTED_LESSON_COUNT,
            "official_scene_count": EXPECTED_SCENE_COUNT,
            "official_code_count": EXPECTED_CODE_COUNT,
            "official_nonempty_operational_definition_count": (
                official_definition_count
            ),
        },
        "selection": {
            "lesson_ids": selected_ids,
            "lesson_count": len(selected_ids),
            "scene_item_count": len(items),
            "is_complete_official_scene_set": len(items) == EXPECTED_SCENE_COUNT,
        },
        "codes": codes,
        "items": items,
        "assignments": assignment_records,
        "codebook_file": codebook_target.name,
        "codebook_file_sha256": _sha256_file(codebook_target),
        "operational_codebook": {
            "required_for_annotation_execution": True,
            "schema": OPERATIONAL_CODEBOOK_SCHEMA,
            "operational_definitions_complete": operational_complete,
            "annotation_execution_ready": operational_complete
            and all(media_available_by_lesson.values()),
            "version": operational_version or None,
            "source_file_sha256": operational_source_digest,
            "normalized_sha256": operational_normalized_digest,
            "bound_code_count": len(codes) if operational_complete else 0,
            "private_file": operational_target.name,
            "private_file_sha256": _sha256_file(operational_target),
            "definitions_authored_by_this_tool": False,
        },
        "private_media_reference": {
            "relative_prefix": prefix,
            "source_media_copied": False,
            "selected_lesson_media_reference_count": len(selected_ids),
            "available_media_reference_count": sum(media_available_by_lesson.values()),
            "all_selected_media_available": all(media_available_by_lesson.values()),
            "media_root_embedded": False,
        },
        "blindness": {
            "assignment_orders_differ": orders["A"] != orders["B"],
            "gold_labels_included": False,
            "model_predictions_included": False,
            "transcript_text_included": False,
            "media_bytes_copied": False,
            "official_gold_files_read_by_this_workflow": False,
        },
        "attestation_requirements": {
            "distinct_rater_ids_required": True,
            "completed_by_real_human_value": ATTESTATION_VALUE,
            "independent_without_gold_or_predictions_value": ATTESTATION_VALUE,
            "signed_name_required": True,
            "utc_timestamp_required": True,
        },
        "human_completion": False,
        "claim_boundary": {
            "assignment_generation_completed": True,
            "operational_definitions_complete": operational_complete,
            "annotation_execution_ready": operational_complete
            and all(media_available_by_lesson.values()),
            "external_human_annotations_received": False,
            "two_independent_human_declarations_received": False,
            "inter_rater_reliability_computed": False,
            "adjudication_completed": False,
            "human_identity_independently_verified": False,
            "recognition_accuracy_established": False,
            "deployment_accuracy_established": False,
            "learner_effect_established": False,
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    write_json(output / "assignment_manifest.json", manifest)
    return manifest


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != ASSIGNMENT_SCHEMA:
        raise ValueError("unsupported TeachObs assignment manifest")
    claimed = str(manifest.get("manifest_sha256", ""))
    if not SHA256_RE.fullmatch(claimed):
        raise ValueError("assignment manifest has no valid SHA-256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if _canonical_sha256(unsigned) != claimed:
        raise ValueError("assignment manifest SHA-256 mismatch")
    if manifest.get("human_completion") is not False:
        raise ValueError("assignment manifest must not predeclare human completion")
    blindness = manifest.get("blindness")
    if not isinstance(blindness, dict) or any(
        blindness.get(field) is not False
        for field in (
            "gold_labels_included",
            "model_predictions_included",
            "transcript_text_included",
            "media_bytes_copied",
            "official_gold_files_read_by_this_workflow",
        )
    ):
        raise ValueError("assignment manifest does not preserve the blind-data boundary")
    operational = manifest.get("operational_codebook")
    if not isinstance(operational, dict):
        raise ValueError("assignment manifest lacks operational codebook status")
    complete = operational.get("operational_definitions_complete")
    ready = operational.get("annotation_execution_ready")
    if not isinstance(complete, bool) or not isinstance(ready, bool):
        raise ValueError("assignment manifest has invalid operational readiness flags")
    if ready and not complete:
        raise ValueError("annotation execution cannot be ready without definitions")
    if complete:
        if (
            operational.get("schema") != OPERATIONAL_CODEBOOK_SCHEMA
            or not CODEBOOK_VERSION_RE.fullmatch(str(operational.get("version", "")))
            or not SHA256_RE.fullmatch(
                str(operational.get("source_file_sha256", ""))
            )
            or not SHA256_RE.fullmatch(str(operational.get("normalized_sha256", "")))
            or operational.get("bound_code_count") != EXPECTED_CODE_COUNT
        ):
            raise ValueError("assignment manifest has an incomplete codebook binding")
    elif any(
        operational.get(field) is not None
        for field in ("version", "source_file_sha256", "normalized_sha256")
    ) or operational.get("bound_code_count") != 0:
        raise ValueError("pending operational codebook status contains false evidence")


def _validate_bound_operational_codebook(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    operational = manifest["operational_codebook"]
    if operational.get("operational_definitions_complete") is not True:
        raise ValueError(
            "39/39 external operational definitions are required before analysis"
        )
    if operational.get("annotation_execution_ready") is not True:
        raise ValueError(
            "annotation execution was not ready when the assignments were generated"
        )
    relative = _safe_private_relative_path(
        str(operational.get("private_file", "")),
        field="operational codebook private file",
    )
    path = manifest_path.parent / relative
    if _sha256_file(path) != operational.get("private_file_sha256"):
        raise ValueError("bound operational codebook file SHA-256 differs")
    normalized, _, normalized_digest = _load_operational_codebook(
        path, manifest["codes"]
    )
    if (
        normalized_digest != operational.get("normalized_sha256")
        or normalized.get("version") != operational.get("version")
    ):
        raise ValueError("bound operational codebook version/hash differs")


def _single_metadata_value(rows: Sequence[Mapping[str, str]], field: str) -> str:
    values = {str(row.get(field, "")).strip() for row in rows}
    values.discard("")
    if len(values) != 1:
        raise ValueError(f"completed assignment must contain one consistent {field}")
    return next(iter(values))


def _validate_utc_timestamp(value: str) -> str:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("attested_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("attested_at_utc must include a UTC offset")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError("attested_at_utc must be UTC")
    return parsed.isoformat().replace("+00:00", "Z")


def _load_completed_assignment(
    path: Path,
    *,
    slot: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    codes = manifest.get("codes")
    if not isinstance(codes, list) or len(codes) != EXPECTED_CODE_COUNT:
        raise ValueError("assignment manifest has an invalid code list")
    fieldnames = _assignment_fieldnames(codes)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise ValueError(f"completed assignment {slot} has an unexpected schema")
        rows = list(reader)
    assignment = manifest.get("assignments", {}).get(slot)
    if not isinstance(assignment, dict):
        raise ValueError(f"assignment manifest has no slot {slot}")
    expected_tokens = assignment.get("ordered_item_tokens")
    if not isinstance(expected_tokens, list) or len(rows) != len(expected_tokens):
        raise ValueError(f"completed assignment {slot} does not cover every item")
    observed_tokens = [str(row.get("item_token", "")) for row in rows]
    if len(set(observed_tokens)) != len(observed_tokens):
        raise ValueError(f"completed assignment {slot} contains duplicate item tokens")
    if observed_tokens != expected_tokens:
        raise ValueError(f"completed assignment {slot} item order/binding differs")
    if any(row.get("assignment_slot") != slot for row in rows):
        raise ValueError(f"completed assignment {slot} has a slot mismatch")
    if [row.get("assignment_position") for row in rows] != [
        str(index) for index in range(1, len(rows) + 1)
    ]:
        raise ValueError(f"completed assignment {slot} positions differ")
    if _immutable_assignment_digest(rows) != assignment.get(
        "immutable_row_sequence_sha256"
    ):
        raise ValueError(f"completed assignment {slot} immutable fields differ")

    rater_id = _single_metadata_value(rows, "rater_id")
    if not RATER_ID_RE.fullmatch(rater_id):
        raise ValueError(f"completed assignment {slot} has an invalid rater_id")
    if _single_metadata_value(rows, "completed_by_real_human").upper() != (
        ATTESTATION_VALUE
    ):
        raise ValueError(f"completed assignment {slot} lacks real-human attestation")
    if _single_metadata_value(
        rows, "independent_without_gold_or_predictions"
    ).upper() != ATTESTATION_VALUE:
        raise ValueError(f"completed assignment {slot} lacks independence attestation")
    signed_name = _single_metadata_value(rows, "signed_name")
    if len(signed_name) < 2 or len(signed_name) > 160:
        raise ValueError(f"completed assignment {slot} has an invalid signed_name")
    attested_at = _validate_utc_timestamp(
        _single_metadata_value(rows, "attested_at_utc")
    )

    labels: dict[str, dict[str, int]] = {}
    for row in rows:
        item_labels: dict[str, int] = {}
        for code in codes:
            name = str(code["name"])
            raw = str(row.get(f"{LABEL_PREFIX}{name}", "")).strip()
            if raw not in {"0", "1"}:
                raise ValueError(
                    f"completed assignment {slot} has a missing/invalid binary label"
                )
            item_labels[name] = int(raw)
        labels[str(row["item_token"])] = item_labels
    return {
        "slot": slot,
        "rater_id": rater_id,
        "signed_name": signed_name,
        "signed_declaration_sha256": _canonical_sha256(
            {
                "rater_id": rater_id,
                "signed_name": signed_name,
                "attested_at_utc": attested_at,
                "completed_by_real_human": True,
                "independent_without_gold_or_predictions": True,
            }
        ),
        "attested_at_utc": attested_at,
        "labels": labels,
        "file_sha256": _sha256_file(path),
        "row_count": len(rows),
    }


def _binary_agreement(a_values: Sequence[int], b_values: Sequence[int]) -> dict[str, Any]:
    if not a_values or len(a_values) != len(b_values):
        raise ValueError("binary agreement requires equally sized non-empty vectors")
    both_positive = sum(a == 1 and b == 1 for a, b in zip(a_values, b_values))
    a_only = sum(a == 1 and b == 0 for a, b in zip(a_values, b_values))
    b_only = sum(a == 0 and b == 1 for a, b in zip(a_values, b_values))
    both_negative = sum(a == 0 and b == 0 for a, b in zip(a_values, b_values))
    count = len(a_values)
    observed = (both_positive + both_negative) / count
    prevalence_a = (both_positive + a_only) / count
    prevalence_b = (both_positive + b_only) / count
    expected = prevalence_a * prevalence_b + (1 - prevalence_a) * (
        1 - prevalence_b
    )
    kappa = None if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)
    positive_denominator = 2 * both_positive + a_only + b_only
    negative_denominator = 2 * both_negative + a_only + b_only
    return {
        "decision_count": count,
        "both_positive_count": both_positive,
        "rater_A_positive_only_count": a_only,
        "rater_B_positive_only_count": b_only,
        "both_negative_count": both_negative,
        "observed_agreement": round(observed, 8),
        "expected_agreement": round(expected, 8),
        "cohen_kappa": round(kappa, 8) if kappa is not None else None,
        "rater_A_positive_prevalence": round(prevalence_a, 8),
        "rater_B_positive_prevalence": round(prevalence_b, 8),
        "mean_positive_prevalence": round((prevalence_a + prevalence_b) / 2, 8),
        "positive_agreement": (
            round(2 * both_positive / positive_denominator, 8)
            if positive_denominator
            else None
        ),
        "negative_agreement": (
            round(2 * both_negative / negative_denominator, 8)
            if negative_denominator
            else None
        ),
        "prevalence_index": round(abs(both_positive - both_negative) / count, 8),
        "bias_index": round(abs(a_only - b_only) / count, 8),
        "disagreement_count": a_only + b_only,
    }


def _mean_defined(values: Iterable[int | float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return round(mean(defined), 8) if defined else None


def _public_pending_receipt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    selection = manifest["selection"]
    media = manifest["private_media_reference"]
    operational = manifest["operational_codebook"]
    return {
        "schema": PUBLIC_RECEIPT_SCHEMA,
        "artifact_kind": "public_teachobs_double_annotation_receipt",
        "dataset_id": manifest["dataset_id"],
        "source_revision": manifest["source_revision"],
        "assignment_manifest_sha256": manifest["manifest_sha256"],
        "aggregate": {
            "official_scene_count": EXPECTED_SCENE_COUNT,
            "selected_lesson_count": selection["lesson_count"],
            "assigned_scene_count": selection["scene_item_count"],
            "code_count": EXPECTED_CODE_COUNT,
            "complete_operational_definition_count": operational[
                "bound_code_count"
            ],
            "assignment_count": 2,
            "available_private_media_reference_count": media[
                "available_media_reference_count"
            ],
            "validated_annotator_count": 0,
            "completed_annotation_row_count": 0,
            "disagreement_count": None,
        },
        "agreement": {
            "macro_label_cohen_kappa": None,
            "pooled_binary_cohen_kappa": None,
            "mean_positive_prevalence": None,
            "mean_positive_agreement": None,
            "exact_multilabel_scene_agreement": None,
        },
        "human_completion": False,
        "status": {
            "assignments_generated": True,
            "different_random_order_verified": True,
            "operational_definitions_complete": operational[
                "operational_definitions_complete"
            ],
            "all_selected_private_media_references_available": media[
                "all_selected_media_available"
            ],
            "annotation_execution_ready": operational[
                "annotation_execution_ready"
            ],
            "complete_binary_labels_received": False,
            "two_distinct_annotators_received": False,
            "real_human_declarations_received": False,
            "independence_declarations_received": False,
            "inter_rater_reliability_computed": False,
            "third_party_adjudication_completed": False,
            "human_identity_independently_verified": False,
        },
        "content_exclusion": {
            "source_media_or_frames_included": False,
            "private_media_references_included": False,
            "transcript_text_included": False,
            "gold_or_prediction_labels_included": False,
            "scene_level_human_labels_included": False,
            "scene_or_lesson_identifiers_included": False,
            "annotator_identifiers_or_names_included": False,
            "local_or_private_paths_included": False,
        },
        "claim_boundary": {
            "annotation_reliability_established": False,
            "recognition_accuracy_established": False,
            "deployment_accuracy_established": False,
            "learner_effect_established": False,
        },
    }


def build_pending_public_teachobs_annotation_receipt(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an aggregate public receipt before any human labels exist."""

    _validate_manifest(manifest)
    return _public_pending_receipt(manifest)


def analyze_teachobs_double_annotations(
    assignment_manifest_path: str | Path,
    completed_assignment_a_path: str | Path,
    completed_assignment_b_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Validate two completed assignments and compute private agreement artifacts."""

    manifest_path = Path(assignment_manifest_path).resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("assignment manifest must be a JSON object")
    _validate_manifest(manifest)
    _validate_bound_operational_codebook(manifest, manifest_path)
    path_a = Path(completed_assignment_a_path).resolve()
    path_b = Path(completed_assignment_b_path).resolve()
    if path_a == path_b:
        raise ValueError("A and B completed assignments must be different files")
    raters = {
        "A": _load_completed_assignment(path_a, slot="A", manifest=manifest),
        "B": _load_completed_assignment(path_b, slot="B", manifest=manifest),
    }
    if raters["A"]["rater_id"] == raters["B"]["rater_id"]:
        raise ValueError("A and B must have different rater_id values")
    if raters["A"]["signed_name"].casefold() == raters["B"][
        "signed_name"
    ].casefold():
        raise ValueError("A and B must have different signed_name values")

    codes = manifest["codes"]
    tokens = [str(row["item_token"]) for row in manifest["items"]]
    item_by_token = {str(row["item_token"]): row for row in manifest["items"]}
    if set(raters["A"]["labels"]) != set(tokens) or set(
        raters["B"]["labels"]
    ) != set(tokens):
        raise ValueError("completed assignment item coverage differs from manifest")

    per_label: dict[str, dict[str, Any]] = {}
    flattened_a: list[int] = []
    flattened_b: list[int] = []
    disagreements: list[dict[str, Any]] = []
    exact_scene_matches = 0
    for token in tokens:
        labels_a = raters["A"]["labels"][token]
        labels_b = raters["B"]["labels"][token]
        exact_scene_matches += labels_a == labels_b
    for code in codes:
        name = str(code["name"])
        a_values = [raters["A"]["labels"][token][name] for token in tokens]
        b_values = [raters["B"]["labels"][token][name] for token in tokens]
        flattened_a.extend(a_values)
        flattened_b.extend(b_values)
        per_label[name] = {
            "code_id": code["code_id"],
            "group": code["group"],
            **_binary_agreement(a_values, b_values),
        }
        for token, value_a, value_b in zip(tokens, a_values, b_values):
            if value_a == value_b:
                continue
            item = item_by_token[token]
            disagreements.append(
                {
                    "item_token": token,
                    "lesson_id": item["lesson_id"],
                    "scene_no": item["scene_no"],
                    "start_seconds": item["start_seconds"],
                    "end_seconds": item["end_seconds"],
                    "private_media_reference": item["private_media_reference"],
                    "code_id": code["code_id"],
                    "code_name": name,
                    "rater_A_value": value_a,
                    "rater_B_value": value_b,
                }
            )

    pooled = _binary_agreement(flattened_a, flattened_b)
    kappas = [value["cohen_kappa"] for value in per_label.values()]
    output = ensure_private_directory(output_directory).resolve()
    report: dict[str, Any] = {
        "schema": AGREEMENT_SCHEMA,
        "dataset_id": manifest["dataset_id"],
        "source_revision": manifest["source_revision"],
        "private_artifact": True,
        "safe_to_publish": False,
        "assignment_manifest_sha256": manifest["manifest_sha256"],
        "operational_codebook": {
            "schema": OPERATIONAL_CODEBOOK_SCHEMA,
            "version": manifest["operational_codebook"]["version"],
            "normalized_sha256": manifest["operational_codebook"][
                "normalized_sha256"
            ],
            "operational_definitions_complete": True,
            "bound_code_count": len(codes),
        },
        "completed_assignment_file_sha256": {
            "A": raters["A"]["file_sha256"],
            "B": raters["B"]["file_sha256"],
        },
        "annotation_validation": {
            "human_completion": True,
            "human_completion_basis": (
                "complete binary matrices plus two distinct signed self-declarations"
            ),
            "annotator_count": 2,
            "rater_ids": [raters["A"]["rater_id"], raters["B"]["rater_id"]],
            "rater_ids_distinct": True,
            "signed_declaration_sha256": {
                "A": raters["A"]["signed_declaration_sha256"],
                "B": raters["B"]["signed_declaration_sha256"],
            },
            "real_human_declarations_received": True,
            "independence_declarations_received": True,
            "complete_item_coverage": True,
            "all_labels_strict_binary": True,
            "gold_or_predictions_present_in_assignments": False,
            "human_identity_independently_verified": False,
        },
        "coverage": {
            "scene_item_count": len(tokens),
            "code_count": len(codes),
            "binary_decision_count_per_annotator": len(tokens) * len(codes),
            "completed_annotation_row_count": len(tokens) * 2,
        },
        "per_label": per_label,
        "overall": {
            "macro_label_cohen_kappa": _mean_defined(kappas),
            "defined_label_kappa_count": sum(value is not None for value in kappas),
            "undefined_label_kappa_count": sum(value is None for value in kappas),
            "macro_mean_positive_prevalence": _mean_defined(
                value["mean_positive_prevalence"] for value in per_label.values()
            ),
            "macro_mean_positive_agreement": _mean_defined(
                value["positive_agreement"] for value in per_label.values()
            ),
            "macro_mean_observed_agreement": _mean_defined(
                value["observed_agreement"] for value in per_label.values()
            ),
            "pooled_binary": pooled,
            "exact_multilabel_scene_agreement": round(
                exact_scene_matches / len(tokens), 8
            ),
            "exact_multilabel_scene_match_count": exact_scene_matches,
            "disagreement_count": len(disagreements),
            "scenes_with_any_disagreement_count": len(
                {row["item_token"] for row in disagreements}
            ),
        },
        "metric_semantics": {
            "per_label_cohen_kappa": (
                "Unweighted Cohen kappa for the two binary decisions on one code."
            ),
            "pooled_binary_cohen_kappa": (
                "Cohen kappa after flattening all scene-by-code binary decisions."
            ),
            "positive_agreement": "2a / (2a + b + c) for each binary code.",
            "mean_positive_prevalence": (
                "Mean of the two annotators' positive rates; not model accuracy."
            ),
        },
        "claim_boundary": {
            "human_annotation_completion_self_attested": True,
            "operational_definitions_complete": True,
            "annotation_reliability_threshold_pre_registered": False,
            "annotation_reliability_accepted": False,
            "human_identity_independently_verified": False,
            "adjudication_completed": False,
            "recognition_accuracy_established": False,
            "deployment_accuracy_established": False,
            "learner_effect_established": False,
        },
    }
    report["report_sha256"] = _canonical_sha256(report)
    report_target = write_json(output / "agreement_report.json", report)

    disagreement_target = _write_csv(
        output / "disagreements.csv",
        (
            "item_token",
            "lesson_id",
            "scene_no",
            "start_seconds",
            "end_seconds",
            "private_media_reference",
            "code_id",
            "code_name",
            "rater_A_value",
            "rater_B_value",
        ),
        disagreements,
    )
    adjudication_rows = [
        {
            "disagreement_no": index,
            "item_token": row["item_token"],
            "start_seconds": _format_seconds(float(row["start_seconds"])),
            "end_seconds": _format_seconds(float(row["end_seconds"])),
            "private_media_reference": row["private_media_reference"],
            "code_name": row["code_name"],
            "rater_A_value": row["rater_A_value"],
            "rater_B_value": row["rater_B_value"],
            "adjudicated_value": "",
            "adjudicator_id": "",
            "third_party_independence_attested": "",
            "signed_name": "",
            "attested_at_utc": "",
            "reason": "",
        }
        for index, row in enumerate(disagreements, 1)
    ]
    adjudication_target = _write_csv(
        output / "adjudication_template.csv",
        ADJUDICATION_COLUMNS,
        adjudication_rows,
    )
    return {
        "report": report,
        "report_path": str(report_target),
        "disagreement_path": str(disagreement_target),
        "adjudication_template_path": str(adjudication_target),
    }


def build_completed_public_teachobs_annotation_receipt(
    manifest: Mapping[str, Any],
    agreement_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an aggregate-only receipt after valid external annotation import."""

    _validate_manifest(manifest)
    if agreement_report.get("schema") != AGREEMENT_SCHEMA:
        raise ValueError("unsupported TeachObs agreement report")
    report_digest = str(agreement_report.get("report_sha256", ""))
    if not SHA256_RE.fullmatch(report_digest):
        raise ValueError("agreement report has no valid SHA-256")
    unsigned_report = dict(agreement_report)
    unsigned_report.pop("report_sha256", None)
    if _canonical_sha256(unsigned_report) != report_digest:
        raise ValueError("agreement report SHA-256 mismatch")
    if agreement_report.get("dataset_id") != manifest.get("dataset_id") or (
        agreement_report.get("source_revision") != manifest.get("source_revision")
    ):
        raise ValueError("agreement report source binding differs")
    operational = manifest["operational_codebook"]
    report_operational = agreement_report.get("operational_codebook")
    if (
        operational.get("operational_definitions_complete") is not True
        or operational.get("annotation_execution_ready") is not True
        or not isinstance(report_operational, dict)
        or report_operational.get("operational_definitions_complete") is not True
        or report_operational.get("version") != operational.get("version")
        or report_operational.get("normalized_sha256")
        != operational.get("normalized_sha256")
        or report_operational.get("bound_code_count") != EXPECTED_CODE_COUNT
    ):
        raise ValueError("agreement report operational codebook binding differs")
    validation = agreement_report.get("annotation_validation")
    if not isinstance(validation, dict) or validation.get("human_completion") is not True:
        raise ValueError("agreement report has no completed human annotation evidence")
    for field in (
        "rater_ids_distinct",
        "real_human_declarations_received",
        "independence_declarations_received",
        "complete_item_coverage",
        "all_labels_strict_binary",
    ):
        if validation.get(field) is not True:
            raise ValueError(f"agreement report is incomplete: {field}")
    if agreement_report.get("assignment_manifest_sha256") != manifest.get(
        "manifest_sha256"
    ):
        raise ValueError("agreement report is bound to another assignment manifest")
    overall = agreement_report.get("overall")
    coverage = agreement_report.get("coverage")
    if not isinstance(overall, dict) or not isinstance(coverage, dict):
        raise ValueError("agreement report lacks aggregate metrics")
    selection = manifest["selection"]
    expected_items = selection["scene_item_count"]
    if (
        coverage.get("scene_item_count") != expected_items
        or coverage.get("code_count") != len(manifest["codes"])
        or coverage.get("binary_decision_count_per_annotator")
        != expected_items * len(manifest["codes"])
        or coverage.get("completed_annotation_row_count") != expected_items * 2
    ):
        raise ValueError("agreement report coverage differs from the assignment")
    per_label = agreement_report.get("per_label")
    if not isinstance(per_label, dict) or len(per_label) != len(manifest["codes"]):
        raise ValueError("agreement report per-label coverage differs")
    claims = agreement_report.get("claim_boundary")
    if not isinstance(claims, dict) or any(
        claims.get(field) is not False
        for field in (
            "human_identity_independently_verified",
            "adjudication_completed",
            "annotation_reliability_threshold_pre_registered",
            "annotation_reliability_accepted",
            "recognition_accuracy_established",
            "deployment_accuracy_established",
            "learner_effect_established",
        )
    ):
        raise ValueError("agreement report contains an unsupported public claim")
    pooled = overall.get("pooled_binary")
    if not isinstance(pooled, dict):
        raise ValueError("agreement report lacks pooled binary metrics")
    receipt = _public_pending_receipt(manifest)
    receipt["agreement_report_sha256"] = report_digest
    receipt["operational_codebook"] = {
        "schema": OPERATIONAL_CODEBOOK_SCHEMA,
        "version_sha256": _sha256_bytes(
            str(operational["version"]).encode("utf-8")
        ),
        "normalized_sha256": operational["normalized_sha256"],
        "complete_code_count": operational["bound_code_count"],
        "operational_definitions_complete": True,
    }
    receipt["aggregate"].update(
        {
            "validated_annotator_count": 2,
            "completed_annotation_row_count": coverage[
                "completed_annotation_row_count"
            ],
            "binary_decision_count_per_annotator": coverage[
                "binary_decision_count_per_annotator"
            ],
            "disagreement_count": overall["disagreement_count"],
            "scenes_with_any_disagreement_count": overall[
                "scenes_with_any_disagreement_count"
            ],
            "defined_label_kappa_count": overall["defined_label_kappa_count"],
            "undefined_label_kappa_count": overall[
                "undefined_label_kappa_count"
            ],
        }
    )
    receipt["agreement"] = {
        "macro_label_cohen_kappa": overall["macro_label_cohen_kappa"],
        "pooled_binary_cohen_kappa": pooled["cohen_kappa"],
        "pooled_observed_agreement": pooled["observed_agreement"],
        "mean_positive_prevalence": overall["macro_mean_positive_prevalence"],
        "mean_positive_agreement": overall["macro_mean_positive_agreement"],
        "exact_multilabel_scene_agreement": overall[
            "exact_multilabel_scene_agreement"
        ],
    }
    receipt["human_completion"] = True
    receipt["status"].update(
        {
            "complete_binary_labels_received": True,
            "two_distinct_annotators_received": True,
            "real_human_declarations_received": True,
            "independence_declarations_received": True,
            "inter_rater_reliability_computed": True,
        }
    )
    return receipt
