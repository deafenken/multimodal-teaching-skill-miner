from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

try:
    import sklearn  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional recognition dependency
    SKLEARN_AVAILABLE = False

import teaching_skill_miner.teachobs_benchmark as text_benchmark
import teaching_skill_miner.teachobs_frozen_model as frozen_model
import teaching_skill_miner.teachobs_multimodal_benchmark as multimodal
from teaching_skill_miner.teachobs_media import (
    AUDIO_SCHEMA,
    FEATURE_FAILURE_SCHEMA,
    FEATURE_SCHEMA,
    FRAME_TASK_SCHEMA,
    VISUAL_EVIDENCE_EVENT_COUNTING_POLICY,
    VISUAL_EVIDENCE_SCHEMA,
)
from teaching_skill_miner.visual_semantics import SCHEMA_RESULT


TEST_IDS = {"S2", "S4", "S5", "S19", "S24", "S28", "S30"}
CODE_NAMES = [f"Code {index:02d}" for index in range(39)]
AUDIO_NAMES = (
    "coverage_fraction",
    "rms_amplitude",
    "mean_absolute_amplitude",
    "peak_absolute_amplitude",
    "dc_offset",
    "zero_crossing_rate",
    "silence_fraction",
)
FULL_TEST_IDS = ("S2", "S4", "S5", "S19", "S24", "S28", "S30")
FULL_TRAIN_IDS = tuple(
    f"S{index}" for index in range(1, 31) if f"S{index}" not in FULL_TEST_IDS
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_materialization_validation(
    manifest_path: str | Path,
    *,
    materialization_root: str | Path | None = None,
    expected_profile: str = multimodal.DEFAULT_BENCHMARK_PROFILE,
) -> dict:
    del materialization_root
    if expected_profile == multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE:
        test_ids = multimodal.PAPER_TRACK1_TEST_LESSON_IDS
    elif expected_profile == multimodal.FULL_23_TRAIN_7_TEST_PROFILE:
        test_ids = FULL_TEST_IDS
    else:
        raise ValueError("fixture profile mismatch")
    sample_ids = [
        f"{lesson_id}:{scene_no}"
        for lesson_id in (*FULL_TRAIN_IDS, *test_ids)
        for scene_no in (1, 2)
    ]
    text_by_sample_id = {
        sample_id: f"materialized transcript {sample_id}"
        for sample_id in sample_ids
    }
    rows = tuple(
        {"sample_id": sample_id, "text": text_by_sample_id[sample_id]}
        for sample_id in sample_ids
    )
    binding = [
        {
            "sample_id": sample_id,
            "text_sha256": hashlib.sha256(
                text_by_sample_id[sample_id].encode("utf-8")
            ).hexdigest(),
        }
        for sample_id in sample_ids
    ]
    profile_digest = _canonical_sha256(
        {"profile": expected_profile, "sample_ids": sample_ids}
    )
    source_tier_lesson_counts = {
        "platform_creator_provided_caption": len(FULL_TRAIN_IDS)
        + len(test_ids),
        "platform_automatic_caption": 0,
        "audited_asr_fallback": 0,
    }
    source_tier_scene_counts = {
        "platform_creator_provided_caption": len(sample_ids),
        "platform_automatic_caption": 0,
        "audited_asr_fallback": 0,
    }
    manifest = {
        "schema": multimodal.TRANSCRIPT_MATERIALIZATION_SCHEMA,
        "profile_id": expected_profile,
        "manifest_sha256": profile_digest,
        "materialization_fingerprint_sha256": profile_digest,
        "ordered_sample_id_sha256": _canonical_sha256(sample_ids),
        "ordered_scene_text_sha256": _canonical_sha256(binding),
        "repository_binding": {
            "repository_commit": "fixture",
            "repository_tree_sha256": "1" * 64,
            "media_plan_sha256": "2" * 64,
        },
        "input_hashes": {"fixture_sha256": "3" * 64},
        "policy": {
            "released_transcript_fallback_used": False,
            "labels_read_or_used": False,
        },
        "claims": {
            "released_transcript_fallback_used": False,
            "labels_read_or_used": False,
            "empty_scenes_filled_from_released_transcript": False,
        },
        "aggregate": {
            "lesson_count": len(FULL_TRAIN_IDS) + len(test_ids),
            "scene_count": len(sample_ids),
            "nonempty_scene_count": len(sample_ids),
            "empty_scene_count": 0,
            "source_tier_lesson_counts": source_tier_lesson_counts,
            "source_tier_scene_counts": source_tier_scene_counts,
        },
    }
    return {
        "manifest": manifest,
        "manifest_path": Path(manifest_path),
        "materialization_root": Path(manifest_path).parent,
        "rows": rows,
        "text_by_sample_id": text_by_sample_id,
    }


def _label_row(lesson_number: int, scene_no: int, *, flip_test: bool) -> dict[str, int]:
    lesson_id = f"S{lesson_number}"
    values: list[int] = []
    for code_index in range(39):
        if code_index == 0:
            value = int((lesson_number + scene_no) % 2 == 0)
            if flip_test and lesson_id in TEST_IDS:
                value = 1
        elif code_index == 1:
            value = 0
        elif code_index == 2:
            value = 1
        else:
            value = int((lesson_number + scene_no + code_index) % 3 == 0)
        values.append(value)
    return dict(zip(CODE_NAMES, values, strict=True))


def _source_value(lesson_number: int) -> str:
    if lesson_number in {1, 2, 6, 7, 11, 12, 16, 17, 21, 26}:
        return "TIMSS fixture"
    if lesson_number in {3, 4, 8, 13, 14, 18, 22, 23, 27, 28}:
        return "DESE fixture"
    if lesson_number in {9, 10, 19}:
        return "Gwangju fixture"
    return {
        5: "U-dang fixture",
        15: "Gangwon fixture",
        20: "Teacher fixture",
        24: "Yoonmo fixture",
        25: "Fun Math fixture",
        29: "Juhee fixture",
        30: "JJ fixture",
    }[lesson_number]


def _build_repository(root: Path, *, flip_test: bool = False) -> Path:
    repository = root / "repository"
    track_a = repository / "data" / "track_a"
    (track_a / "splits").mkdir(parents=True)
    _write_json(
        track_a / "coding_scheme.json",
        {
            "n_codes": 39,
            "codes": [
                {
                    "name": name,
                    "group": "visual" if index < 20 else "nonvisual",
                    "definition": "",
                }
                for index, name in enumerate(CODE_NAMES)
            ],
        },
    )
    train_ids = [f"S{index}" for index in range(1, 31) if f"S{index}" not in TEST_IDS]
    test_ids = [f"S{index}" for index in range(1, 31) if f"S{index}" in TEST_IDS]
    (track_a / "splits" / "train_ids.txt").write_text(
        "\n".join(train_ids) + "\n", encoding="utf-8"
    )
    (track_a / "splits" / "test_ids.txt").write_text(
        "\n".join(test_ids) + "\n", encoding="utf-8"
    )
    lessons = ["id,source,split"]
    for lesson_number in range(1, 31):
        lesson_id = f"S{lesson_number}"
        split = "test" if lesson_id in TEST_IDS else "train"
        lessons.append(f"{lesson_id},{_source_value(lesson_number)},{split}")
        scene_directory = repository / "data" / "scenes" / lesson_id
        scene_directory.mkdir(parents=True)
        manifest: list[dict] = []
        gold: list[dict] = []
        for scene_no in (1, 2):
            transcript_name = f"{lesson_id}_scene_{scene_no:04d}.txt"
            suffix = " test-only-never-fit-token" if lesson_id in TEST_IDS else ""
            (scene_directory / transcript_name).write_text(
                f"[Instructor] parity {lesson_number % 2} scene {scene_no}.{suffix}",
                encoding="utf-8",
            )
            manifest.append(
                {
                    "id": lesson_id,
                    "scene_no": scene_no,
                    "start": float((scene_no - 1) * 15),
                    "end": float(scene_no * 15),
                    "transcript_file": transcript_name,
                }
            )
            gold.append(
                {
                    "lesson_id": lesson_id,
                    "scene_no": scene_no,
                    "codes": _label_row(
                        lesson_number, scene_no, flip_test=flip_test
                    ),
                }
            )
        _write_jsonl(scene_directory / "manifest.jsonl", manifest)
        _write_jsonl(track_a / "gold" / split / f"{lesson_id}.jsonl", gold)
    (repository / "data" / "lessons.csv").write_text(
        "\n".join(lessons) + "\n", encoding="utf-8"
    )
    return repository


def _audio_value(lesson_number: int, scene_no: int, feature_index: int) -> float:
    return round(
        0.05
        + lesson_number * 0.002
        + scene_no * 0.01
        + feature_index * 0.003,
        8,
    )


def _visual_value(lesson_number: int, scene_no: int, feature_index: int) -> float:
    return round(
        (lesson_number % (feature_index + 3)) * 0.04
        + scene_no * 0.02
        + feature_index * 0.01,
        7,
    )


def _build_features(
    root: Path,
    repository: Path,
    *,
    combined: bool = False,
    ocr_available: bool = True,
    clamped_lesson_id: str | None = None,
    omitted_lesson_id: str | None = None,
) -> Path:
    feature_root = root / "private_features"
    records: list[dict] = []
    combined_frames: list[dict] = []
    combined_result_frames: list[dict] = []
    media_set: list[dict[str, str]] = []
    scene_manifest_set: list[dict[str, str]] = []
    weight_digest = "d" * 64
    source_revision = "3d74acf9-pinned-test"
    for lesson_number in range(1, 31):
        lesson_id = f"S{lesson_number}"
        media_digest = hashlib.sha256(f"media:{lesson_id}".encode()).hexdigest()
        scene_manifest_digest = _file_sha256(
            repository / "data" / "scenes" / lesson_id / "manifest.jsonl"
        )
        media_duration = 22.0 if lesson_id == clamped_lesson_id else 30.0
        lesson_frames: list[dict] = []
        for scene_no in (1, 2):
            requested_timestamp = (scene_no - 0.5) * 15.0
            timestamp_clamped = (
                scene_no == 2 and requested_timestamp > media_duration - 0.25
            )
            timestamp = (
                media_duration - 0.25 if timestamp_clamped else requested_timestamp
            )
            image_path = (
                feature_root / "frames" / lesson_id / f"frame_{scene_no:04d}.jpg"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(
                f"private-frame:{lesson_id}:{scene_no}".encode("ascii")
            )
            lesson_frames.append(
                {
                    "frame_id": f"{lesson_id}_scene_{scene_no:04d}",
                    "scene_no": scene_no,
                    "requested_timestamp": requested_timestamp,
                    "timestamp": timestamp,
                    "timestamp_clamped": timestamp_clamped,
                    "timestamp_clamp_delta_seconds": round(
                        requested_timestamp - timestamp, 6
                    ),
                    "path": image_path.name,
                    "sha256": _file_sha256(image_path),
                }
            )
        task_path = feature_root / "frames" / lesson_id / "task.json"
        _write_json(
            task_path,
            {
                "schema": FRAME_TASK_SCHEMA,
                "video_id": lesson_id,
                "media_sha256": media_digest,
                "scene_manifest_sha256": scene_manifest_digest,
                "media_duration_seconds": media_duration,
                "tail_clamp_margin_seconds": 0.25,
                "maximum_tail_clamp_delta_seconds": 15.0,
                "frame_count": 2,
                "frames": lesson_frames,
            },
        )
        task_digest = _file_sha256(task_path)
        audio_rows: list[dict] = []
        visual_rows: list[dict] = []
        evidence_scenes: list[dict] = []
        for scene_no, frame in enumerate(lesson_frames, start=1):
            audio = {
                "scene_no": scene_no,
                **{
                    name: _audio_value(lesson_number, scene_no, index)
                    for index, name in enumerate(AUDIO_NAMES)
                },
            }
            audio_rows.append(audio)
            embedding = [
                _visual_value(lesson_number, scene_no, index) for index in range(4)
            ]
            visual_rows.append(
                {
                    "frame_id": frame["frame_id"],
                    "timestamp": frame["timestamp"],
                    "path": frame["path"],
                    "sha256": frame["sha256"],
                    "embedding": embedding,
                    "embedding_sha256": _canonical_sha256(embedding),
                }
            )
            is_test = lesson_id in TEST_IDS
            ocr_text = (
                f"board parity {lesson_number % 2} scene {scene_no}"
                + (" test-only-ocr-never-fit-token" if is_test else "")
                if ocr_available
                else ""
            )
            evidence_scenes.append(
                {
                    "frame_id": frame["frame_id"],
                    "scene_no": scene_no,
                    "requested_timestamp": frame["requested_timestamp"],
                    "timestamp": frame["timestamp"],
                    "timestamp_clamped": frame["timestamp_clamped"],
                    "timestamp_clamp_delta_seconds": frame[
                        "timestamp_clamp_delta_seconds"
                    ],
                    "path": frame["path"],
                    "sha256": frame["sha256"],
                    "image_metrics": {
                        "backend": "Pillow",
                        "width": 1920,
                        "height": 1080,
                        "rgb_mean": [
                            40.0 + lesson_number,
                            50.0 + scene_no,
                            60.0,
                        ],
                        "rgb_stddev": [10.0, 11.0, 12.0],
                        "luminance_mean": 51.0 + scene_no,
                        "edge_difference_mean": 2.0 + scene_no,
                        "dark_pixel_fraction": 0.1,
                        "bright_pixel_fraction": 0.2,
                        "dhash64": f"{lesson_number * 2 + scene_no:016x}",
                    },
                    "ocr_text": ocr_text,
                    "ocr_audit": {
                        "status": "completed" if ocr_available else "unavailable",
                        "language": "eng",
                        "minimum_word_confidence": 35.0,
                        "raw_word_count": 5 if ocr_available else 0,
                        "accepted_word_count": 4 if ocr_available else 0,
                        "accepted_mean_confidence": 88.0 if ocr_available else None,
                        "confidence_is_calibrated_probability": False,
                    },
                }
            )
        audio = {
            "schema": AUDIO_SCHEMA,
            "lesson_id": lesson_id,
            "media_sha256": media_digest,
            "scene_manifest_sha256": scene_manifest_digest,
            "scene_count": 2,
            "feature_names": list(AUDIO_NAMES),
            "scenes": audio_rows,
        }
        visual = {
            "schema": SCHEMA_RESULT,
            "video_id": lesson_id,
            "media_sha256": media_digest,
            "teachobs_task_file_sha256": task_digest,
            "teachobs_source_revision": source_revision,
            "frame_count": 2,
            "model_provenance": {
                "weight_manifest": {"manifest_sha256": weight_digest}
            },
            "frames": visual_rows,
        }
        transition = {
            "before_frame_id": lesson_frames[0]["frame_id"],
            "after_frame_id": lesson_frames[1]["frame_id"],
            "start": lesson_frames[0]["timestamp"],
            "end": lesson_frames[1]["timestamp"],
            "dhash_hamming_distance": 16,
            "dhash_distance_fraction": 0.25,
            "edge_difference_delta": 1.0,
            "event_types": ["scene_change"],
        }
        evidence_configuration = {
            "use_ocr": ocr_available,
            "ocr_language": "eng" if ocr_available else None,
            "minimum_word_confidence": 35.0,
            "scene_change_threshold": 0.25,
            "slide_change_threshold": 0.45,
            "event_type_counting_policy": (
                VISUAL_EVIDENCE_EVENT_COUNTING_POLICY
            ),
        }
        evidence = {
            "schema": VISUAL_EVIDENCE_SCHEMA,
            "lesson_id": lesson_id,
            "media_sha256": media_digest,
            "scene_manifest_sha256": scene_manifest_digest,
            "task_file_sha256": task_digest,
            "configuration": evidence_configuration,
            "configuration_sha256": _canonical_sha256(evidence_configuration),
            "scene_count": 2,
            "transition_count": 1,
            "event_count": 1,
            "event_type_counts": {
                "scene_change": 1,
                "slide_change": 0,
                "board_build_up": 0,
                "code_or_formula_visible": 0,
            },
            "ocr_status_counts": {
                "completed" if ocr_available else "unavailable": 2
            },
            "private_artifact": True,
            "public_release_authorized": False,
            "scenes": evidence_scenes,
            "transitions": [transition],
            "events": [{"type": "scene_change"}],
            "privacy": {
                "contains_frame_paths": True,
                "contains_ocr_text": True,
                "contains_image_metrics": True,
                "safe_to_publish": False,
            },
            "claim_boundary": {
                "per_scene_image_metrics_computed": True,
                "per_scene_ocr_requested": ocr_available,
                "ocr_complete_for_every_scene": ocr_available,
                "adjacent_visual_events_inferred": True,
                "events_are_deterministic_heuristics": True,
                "event_accuracy_established": False,
                "human_ground_truth_used": False,
            },
        }
        audio_path = feature_root / "audio" / f"{lesson_id}.json"
        visual_path = feature_root / "visual" / f"{lesson_id}.json"
        evidence_path = feature_root / "visual_evidence" / f"{lesson_id}.json"
        _write_json(audio_path, audio)
        _write_json(visual_path, visual)
        _write_json(evidence_path, evidence)
        record = {
            "lesson_id": lesson_id,
            "scene_count": 2,
            "media_sha256": media_digest,
            "scene_manifest_sha256": scene_manifest_digest,
            "frame_task_path": f"frames/{lesson_id}/task.json",
            "frame_task_sha256": task_digest,
            "audio_feature_path": f"audio/{lesson_id}.json",
            "audio_feature_sha256": _file_sha256(audio_path),
            "visual_feature_path": f"visual/{lesson_id}.json",
            "visual_feature_sha256": _file_sha256(visual_path),
            "visual_evidence_path": f"visual_evidence/{lesson_id}.json",
            "visual_evidence_sha256": _file_sha256(evidence_path),
            "ocr_status_counts": evidence["ocr_status_counts"],
        }
        if combined:
            for frame, visual_row in zip(lesson_frames, visual_rows, strict=True):
                combined_frame = {**frame, "path": f"{lesson_id}/{frame['path']}"}
                combined_frames.append(combined_frame)
                combined_result_frames.append(
                    {
                        **combined_frame,
                        "embedding": visual_row["embedding"],
                        "embedding_sha256": visual_row["embedding_sha256"],
                    }
                )
            media_set.append(
                {"lesson_id": lesson_id, "media_sha256": media_digest}
            )
            scene_manifest_set.append(
                {
                    "lesson_id": lesson_id,
                    "scene_manifest_sha256": scene_manifest_digest,
                }
            )
        records.append(record)
    if omitted_lesson_id is not None:
        records = [
            record
            for record in records
            if record["lesson_id"] != omitted_lesson_id
        ]
        media_set = [
            row for row in media_set if row["lesson_id"] != omitted_lesson_id
        ]
        scene_manifest_set = [
            row
            for row in scene_manifest_set
            if row["lesson_id"] != omitted_lesson_id
        ]
        prefix = f"{omitted_lesson_id}_scene_"
        combined_frames = [
            row for row in combined_frames if not row["frame_id"].startswith(prefix)
        ]
        combined_result_frames = [
            row
            for row in combined_result_frames
            if not row["frame_id"].startswith(prefix)
        ]
    failed_ids = [omitted_lesson_id] if omitted_lesson_id is not None else []
    lesson_count = len(records)
    scene_count = lesson_count * 2
    manifest = {
        "schema": FEATURE_SCHEMA,
        "private_artifact": True,
        "public_release_authorized": False,
        "lesson_count": lesson_count,
        "expected_lesson_count": 30,
        "scene_count": scene_count,
        "complete": omitted_lesson_id is None,
        "failed_lesson_count": len(failed_ids),
        "failed_lesson_ids": failed_ids,
        "failure_records_path": (
            "feature_failures.json" if omitted_lesson_id is not None else None
        ),
        "failure_records_sha256": None,
        "audio_statistics_included": True,
        "visual_evidence_included": True,
        "ocr_requested": ocr_available,
        "clip_embeddings_included": True,
        "clip_embeddings_complete_for_all_lessons": omitted_lesson_id is None,
        "claim_boundary": {
            "full_scene_midpoint_frames_extracted": omitted_lesson_id is None,
            "audio_statistics_computed": omitted_lesson_id is None,
            "image_metrics_computed": omitted_lesson_id is None,
            "adjacent_visual_events_inferred": omitted_lesson_id is None,
            "clip_visual_embeddings_computed": True,
            "clip_visual_embeddings_complete_for_all_lessons": (
                omitted_lesson_id is None
            ),
            "clip_visual_embeddings_partial_success_only": (
                omitted_lesson_id is not None
            ),
        },
        "lessons": records,
    }
    if omitted_lesson_id is not None:
        failure_path = feature_root / "feature_failures.json"
        _write_json(
            failure_path,
            {
                "schema": FEATURE_FAILURE_SCHEMA,
                "private_artifact": True,
                "public_release_authorized": False,
                "failure_count": 1,
                "failures": [
                    {
                        "lesson_id": omitted_lesson_id,
                        "stage": "media_binding",
                        "reason": "source_media_unavailable",
                    }
                ],
            },
        )
        manifest["failure_records_sha256"] = _file_sha256(failure_path)
    manifest_path = feature_root / "feature_manifest.json"
    if combined:
        included_ids = [row["lesson_id"] for row in media_set]
        combined_task = {
            "schema": FRAME_TASK_SCHEMA,
            "video_id": "teachobs_combined_scene_midpoints",
            "complete": omitted_lesson_id is None,
            "lesson_count": lesson_count,
            "expected_lesson_count": 30,
            "included_lesson_ids": included_ids,
            "expected_lesson_ids": [f"S{index}" for index in range(1, 31)],
            "failed_lesson_ids": failed_ids,
            "media_sha256": _canonical_sha256(media_set),
            "media_set": media_set,
            "scene_manifest_set_sha256": _canonical_sha256(scene_manifest_set),
            "frame_count": scene_count,
            "private_artifact": True,
            "public_release_authorized": False,
            "frames": combined_frames,
        }
        combined_task_path = feature_root / "frames" / "combined_visual_task.json"
        _write_json(combined_task_path, combined_task)
        combined_task_file_sha256 = _file_sha256(combined_task_path)
        combined_result = {
            "schema": SCHEMA_RESULT,
            "video_id": "teachobs_combined_scene_midpoints",
            "media_sha256": combined_task["media_sha256"],
            "task_manifest_sha256": _canonical_sha256(combined_task),
            "teachobs_task_file_sha256": combined_task_file_sha256,
            "teachobs_source_revision": source_revision,
            "frame_count": scene_count,
            "private_artifact": True,
            "public_release_authorized": False,
            "model_provenance": {
                "weight_manifest": {"manifest_sha256": weight_digest}
            },
            "frames": combined_result_frames,
        }
        combined_result_path = feature_root / "visual" / "combined.json"
        _write_json(combined_result_path, combined_result)
        manifest.update(
            {
                "combined_clip_complete": omitted_lesson_id is None,
                "combined_clip_lesson_count": lesson_count,
                "combined_clip_expected_lesson_count": 30,
                "combined_clip_lesson_ids": included_ids,
                "combined_clip_failed_lesson_ids": failed_ids,
                "combined_clip_task_path": "frames/combined_visual_task.json",
                "combined_clip_task_sha256": combined_task_file_sha256,
                "combined_clip_result_path": "visual/combined.json",
                "combined_clip_result_sha256": _file_sha256(combined_result_path),
                "combined_clip_frame_count": scene_count,
                "combined_clip_media_set_sha256": combined_task["media_sha256"],
                "combined_clip_scene_manifest_set_sha256": combined_task[
                    "scene_manifest_set_sha256"
                ],
                "combined_clip_task_manifest_sha256": _canonical_sha256(
                    combined_task
                ),
                "combined_clip_model_provenance_sha256": _canonical_sha256(
                    combined_result["model_provenance"]
                ),
            }
        )
    _write_json(manifest_path, manifest)
    return manifest_path


def _update_manifest_hash(manifest_path: Path, lesson_id: str, field: str, path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(row for row in manifest["lessons"] if row["lesson_id"] == lesson_id)
    record[field] = _file_sha256(path)
    _write_json(manifest_path, manifest)


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is optional")
class TeachObsMultimodalBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.materialization_validator = patch.object(
            multimodal,
            "validate_teachobs_transcript_materialization",
            side_effect=_fixture_materialization_validation,
        )
        self.materialization_validator.start()
        self.addCleanup(self.materialization_validator.stop)

    def test_benchmark_requires_a_valid_materialization_manifest(self) -> None:
        with self.assertRaises(TypeError):
            multimodal.run_teachobs_multimodal_benchmark("repository", "features")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            feature_manifest = _build_features(
                root, repository, combined=True, omitted_lesson_id="S4"
            )
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(
                    multimodal,
                    "validate_teachobs_transcript_materialization",
                    side_effect=multimodal.TeachObsTranscriptMaterializationError(
                        "materialized transcript hash mismatch"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError,
                    "materialization failed validation",
                ):
                    multimodal.run_teachobs_multimodal_benchmark(
                        repository,
                        feature_manifest,
                        transcript_materialization_manifest_path=(
                            repository / "data" / "lessons.csv"
                        ),
                        profile=(
                            multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE
                        ),
                    )
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(
                    multimodal,
                    "validate_teachobs_transcript_materialization",
                    side_effect=multimodal.TeachObsTranscriptMaterializationError(
                        "only paper_track1_23_train_6_test is supported"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError,
                    "only paper_track1",
                ):
                    multimodal.run_teachobs_multimodal_benchmark(
                        repository,
                        feature_manifest,
                        transcript_materialization_manifest_path=(
                            repository / "data" / "lessons.csv"
                        ),
                    )

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is optional")
    def test_empty_materialized_scene_never_falls_back_to_released_canary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            released = (
                repository
                / "data"
                / "scenes"
                / "S1"
                / "S1_scene_0001.txt"
            )
            released.write_text(
                "released canary qzxcanarymustneverreachmodel",
                encoding="utf-8",
            )
            feature_manifest = _build_features(root, repository)
            materialization_path = repository / "data" / "lessons.csv"
            validated = _fixture_materialization_validation(
                materialization_path,
                expected_profile=multimodal.FULL_23_TRAIN_7_TEST_PROFILE,
            )
            validated["text_by_sample_id"]["S1:1"] = ""
            validated["manifest"]["aggregate"]["nonempty_scene_count"] -= 1
            validated["manifest"]["aggregate"]["empty_scene_count"] += 1
            frozen_output = root / "frozen"
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 2),
                patch.object(
                    multimodal,
                    "validate_teachobs_transcript_materialization",
                    return_value=validated,
                ),
            ):
                result = multimodal.run_teachobs_multimodal_benchmark(
                    repository,
                    feature_manifest,
                    transcript_materialization_manifest_path=(
                        materialization_path
                    ),
                    frozen_model_output=frozen_output,
                )
            bundle = frozen_model.load_teachobs_frozen_bundle(frozen_output)
            vocabulary = bundle.arms["transcript_only"].text_contracts[
                "transcript_tfidf"
            ]["vocabulary"]
            self.assertNotIn(
                "qzxc",
                "".join(str(token) for token in vocabulary),
            )
            transcript_audit = result["private_transcript_audit"]
            self.assertEqual(transcript_audit["empty_scene_count"], 1)
            self.assertFalse(
                transcript_audit["released_transcript_fallback_used"]
            )
            self.assertFalse(
                transcript_audit[
                    "released_repository_transcripts_used_for_model_input"
                ]
            )
            self.assertTrue(
                result["protocol"][
                    "same_materialized_transcript_scenes_used_for_all_arms"
                ]
            )

    def test_source_identity_audit_is_profile_bound_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            with patch.object(
                text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60
            ):
                full_dataset = text_benchmark.load_teachobs_text_dataset(repository)
            full_profile = multimodal.resolve_teachobs_benchmark_profile(
                full_dataset, multimodal.FULL_23_TRAIN_7_TEST_PROFILE
            )
            paper_profile = multimodal.resolve_teachobs_benchmark_profile(
                full_dataset,
                multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
            )
            full_audit = multimodal._load_source_identity_audit(
                repository, full_profile
            )
            paper_audit = multimodal._load_source_identity_audit(
                repository, paper_profile
            )
            self.assertEqual(full_audit["selected_test_lesson_count"], 7)
            self.assertEqual(paper_audit["selected_test_lesson_count"], 6)
            self.assertEqual(full_audit["overlapping_source_value_count"], 3)
            self.assertEqual(paper_audit["overlapping_source_value_count"], 3)
            self.assertEqual(full_audit["source_value_union_count"], 10)
            lessons = repository / "data" / "lessons.csv"
            lessons.write_text(
                lessons.read_text(encoding="utf-8").replace(
                    "S2,TIMSS fixture,test", "S2,unseen fixture,test"
                ),
                encoding="utf-8",
            )
            with patch.object(
                text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60
            ):
                changed_dataset = text_benchmark.load_teachobs_text_dataset(
                    repository
                )
            changed_profile = multimodal.resolve_teachobs_benchmark_profile(
                changed_dataset, multimodal.FULL_23_TRAIN_7_TEST_PROFILE
            )
            with self.assertRaisesRegex(
                multimodal.TeachObsMultimodalBenchmarkError,
                "source-value train/test overlap differs",
            ):
                multimodal._load_source_identity_audit(
                    repository, changed_profile
                )

    def test_paper_track1_profile_uses_exact_six_lesson_s4_hole(self) -> None:
        self.assertEqual(
            multimodal.PAPER_TRACK1_TEST_LESSON_IDS,
            ("S2", "S5", "S19", "S24", "S28", "S30"),
        )
        self.assertEqual(multimodal.PAPER_TRACK1_EXPECTED_TEST_SCENE_COUNT, 1_099)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(
                root, repository, combined=True, omitted_lesson_id="S4"
            )
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 10),
            ):
                private = multimodal.run_teachobs_multimodal_benchmark(
                    repository,
                    manifest,
                    transcript_materialization_manifest_path=(
                        repository / "data" / "lessons.csv"
                    ),
                    profile=multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                )
                receipt = multimodal.build_public_teachobs_multimodal_receipt(
                    private
                )

        audit = private["dataset_audit"]
        self.assertEqual(
            audit["benchmark_profile"],
            multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
        )
        self.assertEqual(
            (audit["lesson_count"], audit["train_lesson_count"], audit["test_lesson_count"]),
            (29, 23, 6),
        )
        self.assertEqual(
            (audit["scene_count"], audit["train_scene_count"], audit["test_scene_count"]),
            (58, 46, 12),
        )
        self.assertEqual(
            tuple(audit["selected_test_lesson_ids"]),
            multimodal.PAPER_TRACK1_TEST_LESSON_IDS,
        )
        self.assertEqual(audit["excluded_official_test_lesson_ids"], ["S4"])
        self.assertTrue(audit["published_six_lesson_intersection"])
        for arm in multimodal._ARMS:
            run = private["arms"][arm]
            self.assertEqual(run["prediction_shape"], [12, 39])
            self.assertEqual(len(run["per_lesson"]), 6)
            self.assertNotIn(
                "S4", {row["lesson_id"] for row in run["per_lesson"]}
            )
        self.assertEqual(private["paired_cluster_bootstrap"]["cluster_count"], 6)
        source_audit = private["source_identity_audit"]
        self.assertEqual(source_audit["selected_train_source_value_count"], 7)
        self.assertEqual(source_audit["selected_test_source_value_count"], 6)
        self.assertEqual(source_audit["overlapping_source_value_count"], 3)
        self.assertTrue(
            source_audit["train_test_source_value_overlap_detected"]
        )
        self.assertFalse(source_audit["source_value_disjoint_split"])
        self.assertFalse(source_audit["site_held_out_evaluation_performed"])
        self.assertFalse(source_audit["site_held_out_accuracy_established"])
        self.assertFalse(
            private["protocol"]["s4_labels_used_for_fitting_tuning_or_metrics"]
        )
        self.assertIn(
            "six-lesson Track 1 text/frame intersection",
            private["evidence_scope"]["valid_claim"],
        )
        self.assertEqual(
            receipt["dataset_aggregate"]["benchmark_profile"],
            multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
        )
        self.assertEqual(receipt["dataset_aggregate"]["lesson_count"], 29)
        self.assertEqual(receipt["dataset_aggregate"]["test_lesson_count"], 6)
        self.assertEqual(receipt["paired_cluster_bootstrap"]["cluster_count"], 6)
        self.assertEqual(
            receipt["source_identity_aggregate"][
                "overlapping_source_value_count"
            ],
            3,
        )
        self.assertFalse(
            receipt["claim_boundaries"]["site_disjointness_verified"]
        )

    def test_paper_track1_manifest_rejects_any_hole_except_exact_s4(self) -> None:
        for omitted in ("S1", "S5"):
            with self.subTest(omitted=omitted), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repository = _build_repository(root)
                manifest = _build_features(
                    root,
                    repository,
                    combined=True,
                    omitted_lesson_id=omitted,
                )
                with patch.object(
                    text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60
                ):
                    with self.assertRaisesRegex(
                        multimodal.TeachObsMultimodalBenchmarkError,
                        "incompatible with the selected profile",
                    ):
                        multimodal.load_teachobs_multimodal_features(
                            repository,
                            manifest,
                            profile=(
                                multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE
                            ),
                        )

        tampering = (
            ("scene_count", 57),
            (
                "combined_clip_lesson_ids",
                [f"S{index}" for index in range(1, 30) if index != 4],
            ),
            ("combined_clip_complete", True),
        )
        for field, replacement in tampering:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repository = _build_repository(root)
                manifest = _build_features(
                    root, repository, combined=True, omitted_lesson_id="S4"
                )
                value = json.loads(manifest.read_text(encoding="utf-8"))
                value[field] = replacement
                _write_json(manifest, value)
                with patch.object(
                    text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60
                ):
                    with self.assertRaisesRegex(
                        multimodal.TeachObsMultimodalBenchmarkError,
                        "incompatible with the selected profile",
                    ):
                        multimodal.load_teachobs_multimodal_features(
                            repository,
                            manifest,
                            profile=(
                                multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE
                            ),
                        )

    def test_paper_track1_frozen_bundle_binds_profile_and_sample_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(
                root, repository, combined=True, omitted_lesson_id="S4"
            )
            frozen_output = root / "paper_frozen"
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 5),
            ):
                private = multimodal.run_teachobs_multimodal_benchmark(
                    repository,
                    manifest,
                    transcript_materialization_manifest_path=(
                        repository / "data" / "lessons.csv"
                    ),
                    frozen_model_output=frozen_output,
                    profile=multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                )
                full_dataset = text_benchmark.load_teachobs_text_dataset(repository)
                resolved = multimodal.resolve_teachobs_benchmark_profile(
                    full_dataset,
                    multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                )
                features = multimodal.load_teachobs_multimodal_features(
                    repository,
                    manifest,
                    dataset=full_dataset,
                    profile=multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                )
            bundle = frozen_model.load_teachobs_frozen_bundle(frozen_output)
            self.assertEqual(
                bundle.benchmark_profile,
                multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
            )
            self.assertEqual(
                bundle.dataset_profile_fingerprint,
                resolved.dataset_profile_fingerprint,
            )
            self.assertEqual(
                bundle.selected_test_lesson_ids,
                multimodal.PAPER_TRACK1_TEST_LESSON_IDS,
            )
            self.assertEqual(len(bundle.selected_train_sample_ids), 46)
            self.assertEqual(len(bundle.selected_test_sample_ids), 12)
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError,
                "loaded frozen benchmark profile differs",
            ):
                frozen_model.load_teachobs_frozen_bundle(
                    frozen_output,
                    expected_benchmark_profile=(
                        multimodal.FULL_23_TRAIN_7_TEST_PROFILE
                    ),
                )
            prediction_arguments = {
                "transcripts": [
                    (
                        "materialized transcript "
                        f"{scene.lesson_id}:{scene.scene_no}"
                    )
                    for scene in resolved.dataset.test_scenes
                ],
                "transcript_input_materialization_fingerprint_sha256": (
                    bundle.transcript_materialization_fingerprint_sha256
                ),
                "ocr_texts": features.test_ocr_text,
                "audio": features.test_audio,
                "visual_numeric": features.test_visual_numeric,
                "clip": features.test_clip,
                "label_names": resolved.dataset.code_names,
                "audio_feature_names": features.audio_feature_names,
                "visual_numeric_feature_names": (
                    features.visual_numeric_feature_names
                ),
                "clip_source_revision": features.clip_source_revision,
                "clip_weight_manifest_sha256": (
                    features.clip_weight_manifest_sha256
                ),
                "audio_feature_schema": AUDIO_SCHEMA,
                "visual_evidence_schema": VISUAL_EVIDENCE_SCHEMA,
                "visual_evidence_configuration_sha256": (
                    features.visual_evidence_configuration_sha256_values[0]
                ),
                "sample_ids": resolved.selected_test_sample_ids,
                "benchmark_profile": resolved.spec.profile_id,
                "dataset_profile_fingerprint": (
                    resolved.dataset_profile_fingerprint
                ),
            }
            outputs = frozen_model.predict_teachobs_frozen_bundle(
                bundle, **prediction_arguments
            )
            for arm in multimodal._ARMS:
                self.assertEqual(
                    outputs[arm]["prediction_matrix_sha256"],
                    private["arms"][arm]["prediction_matrix_sha256"],
                )
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError,
                "benchmark profile differs",
            ):
                frozen_model.predict_teachobs_frozen_bundle(
                    bundle,
                    **{
                        **prediction_arguments,
                        "benchmark_profile": (
                            multimodal.FULL_23_TRAIN_7_TEST_PROFILE
                        ),
                    },
                )

            bundle_path = frozen_output / "bundle_manifest.json"
            bundle_value = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle_value["training_provenance"][
                "selected_test_lesson_ids"
            ][0] = "S4"
            bundle_value.pop("bundle_fingerprint")
            bundle_value["bundle_fingerprint"] = _canonical_sha256(bundle_value)
            _write_json(bundle_path, bundle_value)
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError,
                "profile binding is invalid",
            ):
                frozen_model.load_teachobs_frozen_bundle(frozen_output)

    def test_frozen_four_arm_bundle_roundtrip_is_exact_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository)
            frozen_output = root / "frozen_models"
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 5),
            ):
                private = multimodal.run_teachobs_multimodal_benchmark(
                    repository,
                    manifest,
                    transcript_materialization_manifest_path=(
                        repository / "data" / "lessons.csv"
                    ),
                    frozen_model_output=frozen_output,
                )
            export = private["frozen_model_export"]
            bundle = frozen_model.load_teachobs_frozen_bundle(
                frozen_output,
                expected_bundle_manifest_file_sha256=export[
                    "bundle_manifest_file_sha256"
                ],
            )
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                dataset = text_benchmark.load_teachobs_text_dataset(repository)
                features = multimodal.load_teachobs_multimodal_features(
                    repository, manifest, dataset=dataset
                )
            outputs = frozen_model.predict_teachobs_frozen_bundle(
                bundle,
                transcripts=[
                    (
                        "materialized transcript "
                        f"{scene.lesson_id}:{scene.scene_no}"
                    )
                    for scene in dataset.test_scenes
                ],
                transcript_input_materialization_fingerprint_sha256=(
                    bundle.transcript_materialization_fingerprint_sha256
                ),
                ocr_texts=features.test_ocr_text,
                audio=features.test_audio,
                visual_numeric=features.test_visual_numeric,
                clip=features.test_clip,
                label_names=dataset.code_names,
                audio_feature_names=features.audio_feature_names,
                visual_numeric_feature_names=features.visual_numeric_feature_names,
                clip_source_revision=features.clip_source_revision,
                clip_weight_manifest_sha256=features.clip_weight_manifest_sha256,
                audio_feature_schema=AUDIO_SCHEMA,
                visual_evidence_schema=VISUAL_EVIDENCE_SCHEMA,
                visual_evidence_configuration_sha256=(
                    features.visual_evidence_configuration_sha256_values[0]
                ),
                sample_ids=[
                    f"{scene.lesson_id}:{scene.scene_no}"
                    for scene in dataset.test_scenes
                ],
            )
            receipt = multimodal.build_public_teachobs_multimodal_receipt(private)
            second_output = root / "frozen_models_second"
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 5),
            ):
                multimodal.run_teachobs_multimodal_benchmark(
                    repository,
                    manifest,
                    transcript_materialization_manifest_path=(
                        repository / "data" / "lessons.csv"
                    ),
                    frozen_model_output=second_output,
                )

            self.assertEqual(tuple(bundle.arms), multimodal._ARMS)
            self.assertTrue(export["test_prediction_bitwise_parity_verified"])
            self.assertLessEqual(
                export["test_probability_maximum_absolute_delta"], 1e-12
            )
            input_fingerprints = {
                outputs[arm]["prediction_input_fingerprint"]
                for arm in multimodal._ARMS
            }
            self.assertEqual(len(input_fingerprints), 1)
            self.assertTrue(
                all(outputs[arm]["sample_identity_bound"] for arm in multimodal._ARMS)
            )
            for arm in multimodal._ARMS:
                artifact = export["arm_model_artifacts"][arm]
                standalone = frozen_model.load_teachobs_frozen_arm(
                    artifact["model_manifest_path"],
                    expected_manifest_file_sha256=artifact[
                        "model_manifest_file_sha256"
                    ],
                )
                self.assertEqual(standalone.arm, arm)
                self.assertEqual(
                    outputs[arm]["prediction_matrix_sha256"],
                    private["arms"][arm]["prediction_matrix_sha256"],
                )
                arm_directory = frozen_output / arm
                self.assertEqual(arm_directory.stat().st_mode & 0o777, 0o700)
                self.assertEqual(
                    (arm_directory / "manifest.json").stat().st_mode & 0o777,
                    0o600,
                )
                self.assertEqual(
                    (arm_directory / "arrays.npz").stat().st_mode & 0o777,
                    0o600,
                )
            public_payload = json.dumps(receipt, ensure_ascii=False)
            self.assertNotIn("vocabulary", public_payload)
            self.assertNotIn("model_coefficient", public_payload)
            self.assertNotIn(str(frozen_output), public_payload)
            self.assertNotIn("frozen_model_export", receipt)
            first_files = sorted(
                path.relative_to(frozen_output)
                for path in frozen_output.rglob("*")
                if path.is_file()
            )
            second_files = sorted(
                path.relative_to(second_output)
                for path in second_output.rglob("*")
                if path.is_file()
            )
            self.assertEqual(first_files, second_files)
            for relative in first_files:
                self.assertEqual(
                    (frozen_output / relative).read_bytes(),
                    (second_output / relative).read_bytes(),
                )

            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError,
                "prediction input fingerprint differs",
            ):
                frozen_model.predict_teachobs_frozen_bundle(
                    bundle,
                    transcripts=[
                        (
                            "materialized transcript "
                            f"{scene.lesson_id}:{scene.scene_no}"
                        )
                        for scene in dataset.test_scenes
                    ],
                    transcript_input_materialization_fingerprint_sha256=(
                        bundle.transcript_materialization_fingerprint_sha256
                    ),
                    ocr_texts=features.test_ocr_text,
                    audio=features.test_audio,
                    visual_numeric=features.test_visual_numeric,
                    clip=features.test_clip,
                    label_names=dataset.code_names,
                    audio_feature_names=features.audio_feature_names,
                    visual_numeric_feature_names=(
                        features.visual_numeric_feature_names
                    ),
                    clip_source_revision=features.clip_source_revision,
                    clip_weight_manifest_sha256=(
                        features.clip_weight_manifest_sha256
                    ),
                    audio_feature_schema=AUDIO_SCHEMA,
                    visual_evidence_schema=VISUAL_EVIDENCE_SCHEMA,
                    visual_evidence_configuration_sha256=(
                        features.visual_evidence_configuration_sha256_values[0]
                    ),
                    sample_ids=[
                        f"{scene.lesson_id}:{scene.scene_no}"
                        for scene in dataset.test_scenes
                    ],
                    expected_prediction_input_fingerprint="f" * 64,
                )

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is optional")
    def test_frozen_bundle_rejects_missing_or_weakened_transcript_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            feature_manifest = _build_features(root, repository)
            frozen_output = root / "frozen"
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 2),
            ):
                result = multimodal.run_teachobs_multimodal_benchmark(
                    repository,
                    feature_manifest,
                    transcript_materialization_manifest_path=(
                        repository / "data" / "lessons.csv"
                    ),
                    frozen_model_output=frozen_output,
                )
            transcript_audit = result["private_transcript_audit"]
            bundle = frozen_model.load_teachobs_frozen_bundle(
                frozen_output,
                expected_transcript_materialization_fingerprint=(
                    transcript_audit["materialization_fingerprint_sha256"]
                ),
                expected_benchmark_input_fingerprint=(
                    transcript_audit["benchmark_input_fingerprint"]
                ),
            )
            self.assertEqual(
                bundle.transcript_materialization_fingerprint_sha256,
                transcript_audit["materialization_fingerprint_sha256"],
            )
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError,
                "materialization differs",
            ):
                frozen_model.load_teachobs_frozen_bundle(
                    frozen_output,
                    expected_transcript_materialization_fingerprint="f" * 64,
                )

            bundle_path = frozen_output / "bundle_manifest.json"
            value = json.loads(bundle_path.read_text(encoding="utf-8"))
            del value["training_provenance"]["transcript_materialization"]
            value.pop("bundle_fingerprint")
            value["bundle_fingerprint"] = _canonical_sha256(value)
            _write_json(bundle_path, value)
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError,
                "training provenance",
            ):
                frozen_model.load_teachobs_frozen_bundle(frozen_output)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is optional")
    def test_public_receipt_binds_materialization_without_text_or_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            feature_manifest = _build_features(
                root, repository, combined=True, omitted_lesson_id="S4"
            )
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 2),
            ):
                private = multimodal.run_teachobs_multimodal_benchmark(
                    repository,
                    feature_manifest,
                    transcript_materialization_manifest_path=(
                        repository / "data" / "lessons.csv"
                    ),
                    profile=multimodal.PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                )
            receipt = multimodal.build_public_teachobs_multimodal_receipt(
                private
            )
            private_audit = private["private_transcript_audit"]
            public_audit = receipt["transcript_aggregate"]
            for field in (
                "manifest_file_sha256",
                "manifest_sha256",
                "materialization_fingerprint_sha256",
                "ordered_sample_id_sha256",
                "ordered_scene_text_sha256",
                "repository_binding_sha256",
                "input_hashes_sha256",
                "benchmark_input_fingerprint",
            ):
                self.assertEqual(public_audit[field], private_audit[field])
            self.assertFalse(public_audit["released_transcript_fallback_used"])
            self.assertFalse(public_audit["labels_read_or_used"])
            payload = json.dumps(receipt, ensure_ascii=False)
            self.assertNotIn("materialized transcript S1:1", payload)
            self.assertNotIn(str(repository), payload)

    def test_standalone_frozen_loader_rejects_semantic_and_resource_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            feature_manifest = _build_features(root, repository)
            frozen_output = root / "frozen_models"
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 2),
            ):
                multimodal.run_teachobs_multimodal_benchmark(
                    repository,
                    feature_manifest,
                    transcript_materialization_manifest_path=(
                        repository / "data" / "lessons.csv"
                    ),
                    frozen_model_output=frozen_output,
                )
            manifest_path = frozen_output / "transcript_only" / "manifest.json"
            arrays_path = manifest_path.with_name("arrays.npz")
            original_manifest = manifest_path.read_bytes()
            original_arrays = arrays_path.read_bytes()

            def write_manifest(value: dict) -> None:
                value.pop("manifest_sha256", None)
                value["manifest_sha256"] = _canonical_sha256(value)
                _write_json(manifest_path, value)

            weakened = json.loads(original_manifest)
            weakened["claim_boundary"][
                "confirmatory_lockbox_result_established"
            ] = True
            write_manifest(weakened)
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError, "claim boundary"
            ):
                frozen_model.load_teachobs_frozen_arm(manifest_path)

            manifest_path.write_bytes(original_manifest)
            wrong_runtime = json.loads(original_manifest)
            wrong_runtime["software_provenance"]["scikit_learn"] = "0.0"
            write_manifest(wrong_runtime)
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError, "runtime versions"
            ):
                frozen_model.load_teachobs_frozen_arm(manifest_path)

            manifest_path.write_bytes(original_manifest)
            import numpy as np

            with np.load(arrays_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            arrays["unreferenced_payload"] = np.ones(1, dtype=np.float64)
            frozen_model._write_deterministic_npz(arrays_path, arrays)
            extra_member = json.loads(original_manifest)
            extra_member["arrays_file_sha256"] = _file_sha256(arrays_path)
            extra_member["arrays"] = frozen_model._array_spec(arrays)
            write_manifest(extra_member)
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError, "array member set differs"
            ):
                frozen_model.load_teachobs_frozen_arm(manifest_path)

            arrays_path.write_bytes(original_arrays)
            manifest_path.write_bytes(original_manifest)
            oversized = json.loads(original_manifest)
            oversized["arrays"]["model_intercept"]["shape"] = [
                frozen_model._MAX_ARRAY_ELEMENTS + 1
            ]
            write_manifest(oversized)
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError, "array is too large"
            ):
                frozen_model.load_teachobs_frozen_arm(manifest_path)

    def test_frozen_npz_rejects_path_traversal_and_object_dtype_before_load(
        self,
    ) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            arrays = {
                "idf__transcript_tfidf": np.ones(1, dtype=np.float64),
                "model_coefficient": np.zeros((39, 1), dtype=np.float64),
                "model_intercept": np.zeros(39, dtype=np.float64),
                "model_thresholds": np.full(39, 0.5, dtype=np.float64),
            }
            specs = frozen_model._array_spec(arrays)

            def npy_bytes(value: object, *, allow_pickle: bool = False) -> bytes:
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(value), allow_pickle=allow_pickle
                )
                return buffer.getvalue()

            traversal = root / "traversal.npz"
            with zipfile.ZipFile(
                traversal, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for name, value in arrays.items():
                    member = f"{name}.npy"
                    if name == "model_intercept":
                        member = "../model_intercept.npy"
                    archive.writestr(member, npy_bytes(value))
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError, "unsafe frozen NPZ member"
            ):
                frozen_model._load_npz(
                    traversal,
                    arm="transcript_only",
                    root=root,
                    expected_file_sha256=_file_sha256(traversal),
                    specs=specs,
                )

            object_dtype = root / "object.npz"
            with zipfile.ZipFile(
                object_dtype, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for name, value in arrays.items():
                    if name == "model_intercept":
                        value = np.asarray([{"not": "safe"}] * 39, dtype=object)
                        archive.writestr(
                            f"{name}.npy", npy_bytes(value, allow_pickle=True)
                        )
                    else:
                        archive.writestr(f"{name}.npy", npy_bytes(value))
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError, "unsafe frozen NPY header"
            ):
                frozen_model._load_npz(
                    object_dtype,
                    arm="transcript_only",
                    root=root,
                    expected_file_sha256=_file_sha256(object_dtype),
                    specs=specs,
                )

    def test_frozen_bundle_hash_label_feature_and_shape_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository)
            frozen_output = root / "frozen_models"
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 2),
            ):
                private = multimodal.run_teachobs_multimodal_benchmark(
                    repository,
                    manifest,
                    transcript_materialization_manifest_path=(
                        repository / "data" / "lessons.csv"
                    ),
                    frozen_model_output=frozen_output,
                )
            pinned_hash = private["frozen_model_export"][
                "bundle_manifest_file_sha256"
            ]
            arrays_path = frozen_output / "full" / "arrays.npz"
            original_arrays = arrays_path.read_bytes()
            arrays_path.write_bytes(original_arrays + b"tamper")
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError, "integrity mismatch"
            ):
                frozen_model.load_teachobs_frozen_bundle(frozen_output)
            arrays_path.write_bytes(original_arrays)

            bundle_path = frozen_output / "bundle_manifest.json"
            original_bundle = bundle_path.read_bytes()
            bundle_value = json.loads(original_bundle)
            bundle_value["label_names"][0], bundle_value["label_names"][1] = (
                bundle_value["label_names"][1],
                bundle_value["label_names"][0],
            )
            _write_json(bundle_path, bundle_value)
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError, "pinned frozen bundle hash"
            ):
                frozen_model.load_teachobs_frozen_bundle(
                    frozen_output,
                    expected_bundle_manifest_file_sha256=pinned_hash,
                )
            bundle_path.write_bytes(original_bundle)

            full_manifest_path = frozen_output / "full" / "manifest.json"
            original_full_manifest = full_manifest_path.read_bytes()
            full_manifest = json.loads(original_full_manifest)
            full_manifest["feature_contract"]["ordered_blocks"][0] = "audio_numeric"
            _write_json(full_manifest_path, full_manifest)
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError, "manifest file hash"
            ):
                frozen_model.load_teachobs_frozen_bundle(frozen_output)
            full_manifest_path.write_bytes(original_full_manifest)

            import numpy as np

            with np.load(arrays_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            arrays["model_coefficient"] = arrays["model_coefficient"][:, :-1]
            frozen_model._write_deterministic_npz(arrays_path, arrays)
            full_manifest = json.loads(original_full_manifest)
            full_manifest["arrays_file_sha256"] = _file_sha256(arrays_path)
            full_manifest["arrays"] = frozen_model._array_spec(arrays)
            full_manifest.pop("manifest_sha256")
            full_manifest["manifest_sha256"] = _canonical_sha256(full_manifest)
            _write_json(full_manifest_path, full_manifest)
            bundle_value = json.loads(original_bundle)
            full_entry = bundle_value["arm_artifacts"]["full"]
            full_entry["arrays_file_sha256"] = _file_sha256(arrays_path)
            full_entry["manifest_sha256"] = full_manifest["manifest_sha256"]
            full_entry["manifest_file_sha256"] = _file_sha256(full_manifest_path)
            bundle_value.pop("bundle_fingerprint")
            bundle_value["bundle_fingerprint"] = _canonical_sha256(bundle_value)
            _write_json(bundle_path, bundle_value)
            with self.assertRaisesRegex(
                frozen_model.TeachObsFrozenModelError,
                "classifier dimension mismatch",
            ):
                frozen_model.load_teachobs_frozen_bundle(frozen_output)

    def test_combined_clip_result_binds_all_official_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository, combined=True)
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                features = multimodal.load_teachobs_multimodal_features(
                    repository, manifest
                )
        self.assertEqual(features.visual_feature_layout, "single_combined_clip_result")
        self.assertEqual(features.train_visual_numeric.shape, (46, 40))
        self.assertEqual(features.test_visual_numeric.shape, (14, 40))
        self.assertEqual(features.train_clip.shape, (46, 4))
        self.assertEqual(features.test_clip.shape, (14, 4))
        self.assertEqual(features.train_visual.shape, (46, 44))
        self.assertEqual(features.test_visual.shape, (14, 44))
        self.assertEqual(len(features.train_ocr_text), 46)
        self.assertEqual(len(features.test_ocr_text), 14)
        self.assertEqual(features.train_audio.shape, (46, 7))
        self.assertGreaterEqual(features.source_file_count, 152)

    def test_combined_clip_tamper_and_wrong_lesson_mapping_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository, combined=True)
            combined_task_path = manifest.parent / "frames" / "combined_visual_task.json"
            combined_task_path.write_text(
                combined_task_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError, "hash mismatch"
                ):
                    multimodal.load_teachobs_multimodal_features(repository, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository, combined=True)
            combined_result_path = manifest.parent / "visual" / "combined.json"
            combined_result = json.loads(
                combined_result_path.read_text(encoding="utf-8")
            )
            combined_result["frames"][0]["frame_id"] = "S2_scene_0001"
            _write_json(combined_result_path, combined_result)
            feature_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            feature_manifest["combined_clip_result_sha256"] = _file_sha256(
                combined_result_path
            )
            _write_json(manifest, feature_manifest)
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError,
                    "wrong lesson/scene/frame mapping",
                ):
                    multimodal.load_teachobs_multimodal_features(repository, manifest)

    def test_four_arm_protocol_is_train_only_and_receipt_is_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository)
            original_dependencies = multimodal._require_dependencies()
            fitted_documents: list[str] = []
            base_vectorizer = original_dependencies["TfidfVectorizer"]

            class RecordingVectorizer(base_vectorizer):
                def fit_transform(self, raw_documents, y=None):
                    documents = list(raw_documents)
                    fitted_documents.extend(documents)
                    return super().fit_transform(documents, y)

            dependencies = {
                **original_dependencies,
                "TfidfVectorizer": RecordingVectorizer,
            }
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 20),
                patch.object(multimodal, "_require_dependencies", return_value=dependencies),
            ):
                private = multimodal.run_teachobs_multimodal_benchmark(
                    repository,
                    manifest,
                    transcript_materialization_manifest_path=(
                        repository / "data" / "lessons.csv"
                    ),
                )
                receipt = multimodal.build_public_teachobs_multimodal_receipt(private)

        self.assertEqual(tuple(private["arms"]), multimodal._ARMS)
        # Three text transforms are fitted once on all 46 training scenes for
        # the final model and once inside each of five lesson-grouped folds.
        # Every training scene belongs to four fold-training partitions:
        # 3 * (46 + 4 * 46) = 690 fitted documents.
        self.assertEqual(len(fitted_documents), 690)
        self.assertFalse(any("test-only-never-fit-token" in text for text in fitted_documents))
        self.assertFalse(
            any("test-only-ocr-never-fit-token" in text for text in fitted_documents)
        )
        preprocessing = private["training_only_preprocessing_audit"]
        self.assertEqual(preprocessing["tfidf_fit_sample_count"], 46)
        self.assertEqual(preprocessing["ocr_char_tfidf"]["fit_sample_count"], 46)
        self.assertEqual(preprocessing["ocr_word_tfidf"]["fit_sample_count"], 46)
        self.assertEqual(preprocessing["audio_scaler_fit_sample_count"], 46)
        self.assertEqual(preprocessing["visual_scaler_fit_sample_count"], 46)
        self.assertEqual(preprocessing["clip_scaler_fit_sample_count"], 46)
        feature_audit = private["private_feature_audit"]
        self.assertEqual(feature_audit["visual_raw_numeric_feature_count"], 40)
        self.assertEqual(feature_audit["clip_embedding_feature_count"], 4)
        self.assertGreater(feature_audit["ocr_char_tfidf_feature_count"], 0)
        self.assertGreater(feature_audit["ocr_word_tfidf_feature_count"], 0)
        self.assertTrue(feature_audit["ocr_complete_for_every_scene"])
        self.assertEqual(
            private["protocol"]["decision_threshold_policy"],
            "per_label_official_training_lesson_grouped_oof",
        )
        self.assertEqual(
            private["protocol"]["decision_threshold_candidate_grid"],
            list(multimodal._ADVANCED_THRESHOLDS),
        )
        self.assertFalse(
            private["protocol"][
                "test_labels_used_for_feature_fitting_training_or_tuning"
            ]
        )
        selection = private["training_only_model_selection_audit"]
        self.assertTrue(selection["preprocessing_refit_inside_every_group_fold"])
        self.assertTrue(
            selection["each_training_scene_has_exactly_one_oof_prediction"]
        )
        self.assertFalse(selection["outer_test_labels_used_for_selection"])
        self.assertFalse(selection["outer_test_features_used_for_selection"])
        self.assertEqual(len(selection["folds"]), 5)
        for fold in selection["folds"]:
            self.assertFalse(
                set(fold["fit_lesson_ids"])
                & set(fold["validation_lesson_ids"])
            )
            self.assertEqual(
                fold["preprocessing"]["fit_scene_count"],
                fold["preprocessing"][
                    "transcript_tfidf_fit_sample_count"
                ],
            )
        for arm in multimodal._ARMS:
            arm_selection = selection["arms"][arm]
            self.assertEqual(
                sum(
                    bool(row["selected"])
                    for row in arm_selection["candidate_reports"]
                ),
                1,
            )
            self.assertEqual(
                len(arm_selection["selected_thresholds"]), 39
            )
        for arm in multimodal._ARMS:
            run = private["arms"][arm]
            self.assertEqual(run["prediction_shape"], [14, 39])
            self.assertEqual(len(run["per_label"]), 39)
            self.assertEqual(len(run["per_lesson"]), 7)
            self.assertEqual(set(run["metrics"]), set(multimodal._METRICS))
            self.assertEqual(
                run["feature_audit"]["feature_count"],
                sum(run["feature_audit"]["feature_blocks"].values()),
            )
        self.assertEqual(
            set(private["arms"]["transcript_visual"]["feature_audit"]["feature_blocks"]),
            {
                "transcript_tfidf",
                "ocr_char_tfidf",
                "ocr_word_tfidf",
                "visual_numeric",
                "clip_embedding",
            },
        )
        bootstrap = private["paired_cluster_bootstrap"]
        self.assertEqual(bootstrap["cluster_count"], 7)
        self.assertEqual(bootstrap["replicates"], 20)
        self.assertIn("full_minus_transcript_only", bootstrap["comparisons"])
        source_audit = private["source_identity_audit"]
        self.assertEqual(source_audit["selected_train_source_value_count"], 7)
        self.assertEqual(source_audit["selected_test_source_value_count"], 6)
        self.assertEqual(source_audit["overlapping_source_value_count"], 3)
        self.assertFalse(source_audit["canonical_site_id_field_present"])
        self.assertFalse(source_audit["source_field_used_as_canonical_site_id"])
        self.assertFalse(source_audit["site_disjointness_verified"])
        self.assertFalse(source_audit["teacher_disjointness_verified"])
        self.assertFalse(source_audit["classroom_disjointness_verified"])
        self.assertFalse(
            source_audit["source_family_sensitivity_metric_computed"]
        )
        self.assertTrue(private["evidence_scope"]["provisional_result"])
        self.assertTrue(
            private["evidence_scope"]["exploratory_multimodal_gain_estimate"]
        )
        self.assertFalse(
            private["evidence_scope"]["confirmatory_multimodal_gain_established"]
        )
        serialized = json.dumps(receipt, ensure_ascii=False)
        for forbidden in (
            '"lesson_id"',
            '"scene_no"',
            '"per_label"',
            '"per_lesson"',
            '"embedding"',
            '"ocr_text"',
            "TIMSS fixture",
            "test-only-never-fit-token",
            "test-only-ocr-never-fit-token",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertFalse(receipt["row_level_data_included"])
        self.assertFalse(receipt["visual_embeddings_included"])
        self.assertFalse(receipt["ocr_text_included"])
        self.assertEqual(
            receipt["aggregate_arms"]["full"]["metrics"],
            private["arms"]["full"]["metrics"],
        )
        self.assertEqual(
            receipt["source_identity_aggregate"][
                "overlapping_source_value_set_sha256"
            ],
            source_audit["overlapping_source_value_set_sha256"],
        )
        self.assertFalse(
            receipt["claim_boundaries"]["site_held_out_accuracy_established"]
        )
        weakened = json.loads(json.dumps(private))
        weakened["evidence_scope"]["site_disjointness_verified"] = True
        with self.assertRaisesRegex(
            multimodal.TeachObsMultimodalBenchmarkError,
            "source/site claim boundary is weakened",
        ):
            multimodal.build_public_teachobs_multimodal_receipt(weakened)

    def test_changing_only_test_gold_does_not_change_any_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as first_temp, tempfile.TemporaryDirectory() as second_temp:
            first_root = Path(first_temp)
            second_root = Path(second_temp)
            first_repository = _build_repository(first_root, flip_test=False)
            second_repository = _build_repository(second_root, flip_test=True)
            first_manifest = _build_features(first_root, first_repository)
            second_manifest = _build_features(second_root, second_repository)
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 10),
            ):
                first = multimodal.run_teachobs_multimodal_benchmark(
                    first_repository,
                    first_manifest,
                    transcript_materialization_manifest_path=(
                        first_repository / "data" / "lessons.csv"
                    ),
                )
                second = multimodal.run_teachobs_multimodal_benchmark(
                    second_repository,
                    second_manifest,
                    transcript_materialization_manifest_path=(
                        second_repository / "data" / "lessons.csv"
                    ),
                )
        for arm in multimodal._ARMS:
            self.assertEqual(
                first["arms"][arm]["prediction_matrix_sha256"],
                second["arms"][arm]["prediction_matrix_sha256"],
            )
        self.assertEqual(
            first["training_only_preprocessing_audit"],
            second["training_only_preprocessing_audit"],
        )
        self.assertEqual(
            first["training_only_model_selection_audit"],
            second["training_only_model_selection_audit"],
        )
        self.assertNotEqual(
            first["arms"]["full"]["per_label"][0]["test_positive_count"],
            second["arms"]["full"]["per_label"][0]["test_positive_count"],
        )

    def test_ocr_unavailable_is_explicit_and_does_not_fake_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository, ocr_available=False)
            frozen_output = root / "frozen_without_ocr"
            with (
                patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60),
                patch.object(multimodal, "_BOOTSTRAP_REPLICATES", 5),
            ):
                private = multimodal.run_teachobs_multimodal_benchmark(
                    repository,
                    manifest,
                    transcript_materialization_manifest_path=(
                        repository / "data" / "lessons.csv"
                    ),
                    frozen_model_output=frozen_output,
                )
                receipt = multimodal.build_public_teachobs_multimodal_receipt(private)
            bundle = frozen_model.load_teachobs_frozen_bundle(frozen_output)

        feature_audit = private["private_feature_audit"]
        self.assertEqual(feature_audit["ocr_status_counts"], {"unavailable": 60})
        self.assertEqual(feature_audit["ocr_completed_scene_count"], 0)
        self.assertEqual(feature_audit["ocr_nonempty_scene_count"], 0)
        self.assertFalse(feature_audit["ocr_complete_for_every_scene"])
        self.assertEqual(feature_audit["ocr_char_tfidf_feature_count"], 0)
        self.assertEqual(feature_audit["ocr_word_tfidf_feature_count"], 0)
        self.assertFalse(
            private["training_only_preprocessing_audit"]["ocr_char_tfidf"][
                "training_text_available"
            ]
        )
        self.assertFalse(
            receipt["feature_aggregate"]["ocr_complete_for_every_scene"]
        )
        self.assertFalse(receipt["ocr_text_included"])
        self.assertEqual(
            bundle.arms["transcript_visual"].block_dimensions[
                "ocr_char_tfidf"
            ],
            0,
        )
        self.assertEqual(
            bundle.arms["transcript_visual"].block_dimensions[
                "ocr_word_tfidf"
            ],
            0,
        )
        self.assertTrue(
            private["frozen_model_export"][
                "test_prediction_bitwise_parity_verified"
            ]
        )

    def test_visual_evidence_hash_scene_and_label_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository)
            evidence_path = manifest.parent / "visual_evidence" / "S1.json"
            evidence_path.write_text(
                evidence_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError, "hash mismatch"
                ):
                    multimodal.load_teachobs_multimodal_features(repository, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository)
            evidence_path = manifest.parent / "visual_evidence" / "S1.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["scenes"][0]["scene_no"] = 99
            _write_json(evidence_path, evidence)
            _update_manifest_hash(
                manifest, "S1", "visual_evidence_sha256", evidence_path
            )
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError,
                    "visual evidence scene alignment mismatch",
                ):
                    multimodal.load_teachobs_multimodal_features(repository, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository)
            evidence_path = manifest.parent / "visual_evidence" / "S1.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["gold"] = [1, 0, 1]
            _write_json(evidence_path, evidence)
            _update_manifest_hash(
                manifest, "S1", "visual_evidence_sha256", evidence_path
            )
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError,
                    "forbidden label field",
                ):
                    multimodal.load_teachobs_multimodal_features(repository, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository)
            evidence_path = manifest.parent / "visual_evidence" / "S1.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["event_type_counts"].pop("code_or_formula_visible")
            _write_json(evidence_path, evidence)
            _update_manifest_hash(
                manifest, "S1", "visual_evidence_sha256", evidence_path
            )
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError,
                    "visual evidence event type counts mismatch",
                ):
                    multimodal.load_teachobs_multimodal_features(repository, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository)
            evidence_path = manifest.parent / "visual_evidence" / "S1.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["event_type_counts"]["scene_change"] = 0
            evidence["event_type_counts"]["code_or_formula_visible"] = 1
            _write_json(evidence_path, evidence)
            _update_manifest_hash(
                manifest, "S1", "visual_evidence_sha256", evidence_path
            )
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError,
                    "visual evidence event type counts mismatch",
                ):
                    multimodal.load_teachobs_multimodal_features(repository, manifest)

    def test_tail_clamp_binds_actual_timestamp_and_invalid_clamp_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(
                root, repository, combined=True, clamped_lesson_id="S6"
            )
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                features = multimodal.load_teachobs_multimodal_features(
                    repository, manifest
                )
            self.assertEqual(features.train_clip.shape, (46, 4))
            task = json.loads(
                (manifest.parent / "frames" / "S6" / "task.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(task["frames"][-1]["requested_timestamp"], 22.5)
            self.assertEqual(task["frames"][-1]["timestamp"], 21.75)
            self.assertTrue(task["frames"][-1]["timestamp_clamped"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository, clamped_lesson_id="S6")
            task_path = manifest.parent / "frames" / "S6" / "task.json"
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task["frames"][-1]["timestamp"] = 22.5
            _write_json(task_path, task)
            _update_manifest_hash(manifest, "S6", "frame_task_sha256", task_path)
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError,
                    "clamp binding mismatch",
                ):
                    multimodal.load_teachobs_multimodal_features(repository, manifest)

    def test_file_hash_scene_alignment_and_embedding_hash_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository)
            audio_path = manifest.parent / "audio" / "S1.json"
            audio_path.write_text(audio_path.read_text() + " ", encoding="utf-8")
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError, "hash mismatch"
                ):
                    multimodal.load_teachobs_multimodal_features(repository, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository)
            audio_path = manifest.parent / "audio" / "S1.json"
            audio = json.loads(audio_path.read_text(encoding="utf-8"))
            audio["scenes"][0]["scene_no"] = 99
            _write_json(audio_path, audio)
            _update_manifest_hash(
                manifest, "S1", "audio_feature_sha256", audio_path
            )
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError,
                    "audio scene alignment mismatch",
                ):
                    multimodal.load_teachobs_multimodal_features(repository, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = _build_repository(root)
            manifest = _build_features(root, repository)
            visual_path = manifest.parent / "visual" / "S1.json"
            visual = json.loads(visual_path.read_text(encoding="utf-8"))
            visual["frames"][0]["embedding"][0] += 0.5
            _write_json(visual_path, visual)
            _update_manifest_hash(
                manifest, "S1", "visual_feature_sha256", visual_path
            )
            with patch.object(text_benchmark, "TEACHOBS_EXPECTED_SCENE_COUNT", 60):
                with self.assertRaisesRegex(
                    multimodal.TeachObsMultimodalBenchmarkError,
                    "visual embedding hash mismatch",
                ):
                    multimodal.load_teachobs_multimodal_features(repository, manifest)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
