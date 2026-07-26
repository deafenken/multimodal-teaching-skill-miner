from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

from teaching_skill_miner.full_video_dataset import file_sha256
from teaching_skill_miner.io_utils import write_json
from teaching_skill_miner.teachobs_asr_handoff import (
    LONG_SEGMENT_POLICY,
    RESULT_SCHEMA,
    RUNNER_NAME,
    ZERO_DURATION_WORD_ATTACHMENT_POLICY,
    _timeline_coverage,
    build_teachobs_asr_job_manifest,
    import_teachobs_asr_results,
    write_teachobs_asr_import,
)
from teaching_skill_miner.teachobs_captions import PRIVATE_AUDIT_SCHEMA
from teaching_skill_miner.teachobs_media import (
    DATASET_ID,
    MEDIA_SCHEMA,
    PINNED_REPOSITORY_COMMIT,
    PLAN_SCHEMA,
)
from teaching_skill_miner.teachobs_transcript_materialization import (
    MANIFEST_SCHEMA,
    PAPER_TRACK1_PROFILE,
    PAPER_TRACK1_TEST_LESSON_IDS,
    PAPER_TRACK1_TRAIN_LESSON_IDS,
    TeachObsTranscriptMaterializationError,
    _canonical_sha256,
    _project_audited_asr_segments_to_selected_timeline,
    _scene_text,
    build_public_teachobs_transcript_materialization_receipt,
    materialize_teachobs_transcripts,
    validate_teachobs_transcript_materialization,
)


_OFFICIAL_TEST = {"S2", "S4", "S5", "S19", "S24", "S28", "S30"}


def _timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, fraction = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{fraction:03d}"


def _vtt(cues: list[tuple[float, float, str]]) -> str:
    blocks = ["WEBVTT"]
    blocks.extend(
        f"{_timestamp(start)} --> {_timestamp(end)}\n{text}"
        for start, end, text in cues
    )
    return "\n\n".join(blocks) + "\n"


def _track(
    caption_root: Path,
    *,
    lesson_id: str,
    language: str,
    track_type: str,
    roles: list[str],
    duration: float,
    cues: list[tuple[float, float, str]],
) -> dict[str, object]:
    relative = Path("raw") / lesson_id / f"{language}.vtt"
    target = caption_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_vtt(cues), encoding="utf-8")
    os.chmod(target, 0o600)
    first = min(start for start, _, _ in cues)
    last = max(end for _, end, _ in cues)
    return {
        "language": language,
        "roles": roles,
        "original_language_declared_by_youtube": "original_language" in roles,
        "track_type": track_type,
        "human_authorship_independently_verified": False,
        "caption_sha256": file_sha256(target),
        "caption_size_bytes": target.stat().st_size,
        "private_caption_relative_path": relative.as_posix(),
        "timeline": {
            "cue_count": len(cues),
            "first_cue_start_seconds": round(first, 3),
            "last_cue_end_seconds": round(last, 3),
            "timeline_span_seconds": round(last - first, 3),
            "timeline_span_coverage_fraction": round(
                min(1.0, (last - first) / duration), 6
            ),
            "timeline_audit_completed": True,
        },
    }


def _scene_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for index, lesson_id in enumerate(PAPER_TRACK1_TRAIN_LESSON_IDS):
        counts[lesson_id] = 172 if index == 0 else 167
    for index, lesson_id in enumerate(PAPER_TRACK1_TEST_LESSON_IDS):
        counts[lesson_id] = 184 if index == 0 else 183
    counts["S4"] = 213
    assert sum(counts.values()) == 5_158
    return counts


def _build_inputs(
    root: Path,
    *,
    missing_selected_lesson: str | None = None,
    platform_endpoint_case: str | None = None,
    media_duration_extension_seconds: float = 0.0,
) -> dict[str, Path]:
    counts = _scene_counts()
    plan_lessons: list[dict[str, object]] = []
    for index in range(1, 31):
        lesson_id = f"S{index}"
        count = counts[lesson_id]
        scenes = [
            {
                "scene_no": scene_no,
                "start": float((scene_no - 1) * 15),
                "end": float(scene_no * 15),
            }
            for scene_no in range(1, count + 1)
        ]
        plan_lessons.append(
            {
                "lesson_id": lesson_id,
                "split": "test" if lesson_id in _OFFICIAL_TEST else "train",
                "reference_duration_seconds": float(count * 15),
                "scene_manifest_sha256": sha256(
                    f"scene-manifest:{lesson_id}".encode()
                ).hexdigest(),
                "scene_count": count,
                "scenes": scenes,
            }
        )
    plan: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "dataset_id": DATASET_ID,
        "private_artifact": True,
        "public_release_authorized": False,
        "repository_provenance": {
            "repository_commit": PINNED_REPOSITORY_COMMIT,
            "repository_tree_sha256": sha256(b"repository-tree").hexdigest(),
        },
        "lessons_csv_sha256": sha256(b"lessons.csv").hexdigest(),
        "lesson_count": 30,
        "scene_count": 5_158,
        "reference_duration_seconds": sum(
            float(item["reference_duration_seconds"]) for item in plan_lessons
        ),
        "lessons": plan_lessons,
        "terms_boundary": {
            "source_videos_covered_by_annotation_license": False,
        },
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    media_root = root / "media"
    media_root.mkdir(parents=True)
    plan_path = write_json(media_root / "media_plan.json", plan)
    media_lessons: list[dict[str, object]] = []
    if missing_selected_lesson is not None:
        media_path = media_root / "videos" / f"{missing_selected_lesson}.mp4"
        media_path.parent.mkdir()
        media_path.write_bytes(b"private-media-fixture")
        media_lesson = next(
            item
            for item in plan_lessons
            if item["lesson_id"] == missing_selected_lesson
        )
        media_lessons.append(
            {
                "lesson_id": missing_selected_lesson,
                "split": media_lesson["split"],
                "media_path": (
                    Path("videos") / f"{missing_selected_lesson}.mp4"
                ).as_posix(),
                "media_sha256": file_sha256(media_path),
                "media_size_bytes": media_path.stat().st_size,
                "media_probe": {
                    "duration_seconds": (
                        float(media_lesson["reference_duration_seconds"])
                        + media_duration_extension_seconds
                    )
                },
            }
        )
    media_manifest = {
        "schema": MEDIA_SCHEMA,
        "dataset_id": DATASET_ID,
        "private_artifact": True,
        "public_release_authorized": False,
        "selected_lesson_count": 30,
        "plan_sha256": plan["plan_sha256"],
        "repository_commit": PINNED_REPOSITORY_COMMIT,
        "repository_tree_sha256": plan["repository_provenance"][
            "repository_tree_sha256"
        ],
        "lessons": media_lessons,
    }
    media_manifest_path = write_json(
        media_root / "media_manifest.json", media_manifest
    )

    caption_root = root / "captions"
    caption_root.mkdir()
    records: list[dict[str, object]] = []
    for lesson in plan_lessons:
        lesson_id = str(lesson["lesson_id"])
        count = int(lesson["scene_count"])
        duration = float(lesson["reference_duration_seconds"])
        if lesson_id in {"S4", missing_selected_lesson}:
            tracks: list[dict[str, object]] = []
            status = "caption_unavailable_fallback_pending"
        elif lesson_id == "S1":
            manual_cues = [
                (0.0, 4.0, "hello world"),
                (4.0, 8.0, "world again"),
                (15.0, 20.0, "boundary cue"),
                (duration - 1.0, duration, "tail"),
            ]
            tracks = [
                _track(
                    caption_root,
                    lesson_id=lesson_id,
                    language="nl",
                    track_type="youtube_automatic_caption",
                    roles=["original_language"],
                    duration=duration,
                    cues=[(0.0, 1.0, "automatic"), (duration - 1.0, duration, "tail")],
                ),
                _track(
                    caption_root,
                    lesson_id=lesson_id,
                    language="en",
                    track_type="manual_creator_provided",
                    roles=["english"],
                    duration=duration,
                    cues=manual_cues,
                ),
            ]
            status = "caption_timeline_audited"
        elif lesson_id == "S2":
            tracks = [
                _track(
                    caption_root,
                    lesson_id=lesson_id,
                    language="en",
                    track_type="manual_creator_provided",
                    roles=["english"],
                    duration=duration,
                    cues=[(0.0, 1.0, "english"), (duration - 1.0, duration, "tail")],
                ),
                _track(
                    caption_root,
                    lesson_id=lesson_id,
                    language="nl",
                    track_type="manual_creator_provided",
                    roles=["original_language"],
                    duration=duration,
                    cues=[(0.0, 1.0, "original"), (duration - 1.0, duration, "tail")],
                ),
            ]
            status = "caption_timeline_audited"
        elif lesson_id == "S17" and platform_endpoint_case is not None:
            endpoint_cue = {
                "clamp": (
                    duration - 65.0,
                    duration + 35.522,
                    "long final cue",
                ),
                "starts_at_boundary": (
                    duration,
                    duration + 5.0,
                    "outside cue",
                ),
                "starts_after_boundary": (
                    duration + 1.0,
                    duration + 5.0,
                    "outside cue",
                ),
                "invalid_duration": (
                    duration - 1.0,
                    duration - 1.0,
                    "invalid cue",
                ),
            }[platform_endpoint_case]
            tracks = [
                _track(
                    caption_root,
                    lesson_id=lesson_id,
                    language="en",
                    track_type="manual_creator_provided",
                    roles=["original_language"],
                    duration=duration,
                    cues=[(0.0, 1.0, "head"), endpoint_cue],
                )
            ]
            status = "caption_timeline_audited"
        else:
            tracks = [
                _track(
                    caption_root,
                    lesson_id=lesson_id,
                    language="en",
                    track_type="manual_creator_provided",
                    roles=["original_language"],
                    duration=duration,
                    cues=[(0.0, 1.0, "head"), (duration - 1.0, duration, "tail")],
                )
            ]
            status = "caption_timeline_audited"
        records.append(
            {
                "lesson_id": lesson_id,
                "split": lesson["split"],
                "scene_manifest_sha256": lesson["scene_manifest_sha256"],
                "reference_duration_seconds": duration,
                "status": status,
                "track_count": len(tracks),
                "tracks": tracks,
                "released_transcript_audit": {
                    "expected_file_count": count,
                    "content_accuracy_established": False,
                    "human_review_established": False,
                },
            }
        )
    caption_audit: dict[str, object] = {
        "schema": PRIVATE_AUDIT_SCHEMA,
        "dataset_id": DATASET_ID,
        "generated_at_utc": "2026-07-23T00:00:00Z",
        "private_artifact": True,
        "public_release_authorized": False,
        "repository_commit": PINNED_REPOSITORY_COMMIT,
        "media_plan_sha256": plan["plan_sha256"],
        "claims": {
            "formal_caption_timeline_audit_completed": False,
            "caption_content_accuracy_established": False,
            "word_error_rate_established": False,
            "human_caption_authorship_independently_verified": False,
            "independent_human_transcript_audit_completed": False,
            "double_annotation_reliability_established": False,
        },
        "records": records,
    }
    caption_audit["audit_canonical_sha256"] = _canonical_sha256(caption_audit)
    caption_audit_path = write_json(
        caption_root / "caption_audit.json", caption_audit
    )
    return {
        "media_plan": plan_path,
        "media_manifest": media_manifest_path,
        "caption_audit": caption_audit_path,
        "media_root": media_root,
    }


@pytest.fixture
def materialization_inputs(tmp_path: Path) -> dict[str, Path]:
    return _build_inputs(tmp_path / "inputs")


def _run(
    inputs: dict[str, Path],
    output: Path,
    *,
    public_receipt: Path | None = None,
) -> dict[str, object]:
    return materialize_teachobs_transcripts(
        inputs["media_plan"],
        inputs["media_manifest"],
        inputs["caption_audit"],
        None,
        None,
        None,
        None,
        output,
        public_receipt_path=public_receipt,
        generated_at_utc="2026-07-23T01:02:03Z",
    )


def _build_asr_chain(
    root: Path,
    inputs: dict[str, Path],
    *,
    lesson_id: str,
    outside_selected_tail: bool = False,
) -> dict[str, Path]:
    plan = json.loads(inputs["media_plan"].read_text(encoding="utf-8"))
    selected_timeline_end = float(
        next(
            item["reference_duration_seconds"]
            for item in plan["lessons"]
            if item["lesson_id"] == lesson_id
        )
    )
    media_manifest = json.loads(
        inputs["media_manifest"].read_text(encoding="utf-8")
    )
    duration = float(
        next(
            item["media_probe"]["duration_seconds"]
            for item in media_manifest["lessons"]
            if item["lesson_id"] == lesson_id
        )
    )
    asr_root = root / "asr"
    asr_root.mkdir(parents=True)
    job = build_teachobs_asr_job_manifest(
        inputs["media_manifest"],
        inputs["media_root"],
        inputs["caption_audit"],
        model_id="Systran/faster-whisper-large-v3",
        model_revision="a" * 40,
        model_files_sha256=sha256(b"model").hexdigest(),
        faster_whisper_version="1.2.1",
        ctranslate2_version="4.8.1",
        container_image_digest=f"sha256:{'b' * 64}",
        lesson_ids=[lesson_id],
        require_all_selected_media=True,
        duration_probe=lambda _: duration,
        generated_at_utc="2026-07-23T00:10:00Z",
    )
    job_path = write_json(asr_root / "job_manifest.json", job)
    bound_job = job["jobs"][0]
    tail = (
        {
            "start": selected_timeline_end + 0.23,
            "end": selected_timeline_end + 2.69,
            "text": "asr media-only tail",
        }
        if outside_selected_tail
        else {
            "start": duration - 1.0,
            "end": duration,
            "text": "asr ending",
        }
    )
    segments = [
        {"start": 0.0, "end": 1.0, "text": "asr beginning"},
        tail,
    ]
    coverage = _timeline_coverage(
        segments,
        duration=duration,
        maximum_segment_duration=float(
            bound_job["decoding_config"]["maximum_segment_duration_seconds"]
        ),
        policy=bound_job["coverage_policy"],
    )
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "job_manifest_sha256": job["manifest_sha256"],
        "job_sha256": bound_job["job_sha256"],
        "job_id": bound_job["job_id"],
        "lesson_id": lesson_id,
        "transcript_kind": "automatic_speech_recognition",
        "source_tier": "audited_asr_fallback_candidate",
        "official_caption": False,
        "human_content_review_completed": False,
        "content_accuracy_established": False,
        "word_error_rate_established": False,
        "media_binding": bound_job["media_binding"],
        "model": bound_job["model"],
        "decoding_config": bound_job["decoding_config"],
        "language": {
            "requested": "auto",
            "detected": "en",
            "detected_probability": 0.9,
        },
        "model_runtime": {
            "model_snapshot_sha256_v1": bound_job["model"][
                "model_snapshot_sha256_v1"
            ],
            "network_access_enabled": False,
            "local_files_only": True,
        },
        "runtime": {
            "runner_name": RUNNER_NAME,
            "runner_source_sha256": bound_job["runtime_contract"][
                "runner_module_sha256"
            ],
            "package_versions": {
                "faster-whisper": "1.2.1",
                "ctranslate2": "4.8.1",
            },
            "python_version": "3.11.9",
            "accelerator": {
                "device_type": "cuda",
                "device_count": 1,
                "device_name": "NVIDIA L40",
                "driver_version": "550.54",
            },
            "container_image_digest": f"sha256:{'b' * 64}",
            "executed_at_utc": "2026-07-23T00:20:00Z",
        },
        "segments": segments,
        "segment_postprocessing": {
            "policy": LONG_SEGMENT_POLICY,
            "maximum_output_segment_duration_seconds": 30.5,
            "source_segment_count": 2,
            "source_segments_split": 0,
            "output_segment_count": 2,
            "overlong_source_word_timestamp_count": 0,
            "positive_duration_word_anchor_count": 0,
            "attached_zero_duration_word_count": 0,
            "zero_duration_word_attachment_policy": (
                ZERO_DURATION_WORD_ATTACHMENT_POLICY
            ),
            "zero_duration_word_maximum_attachment_distance_seconds": 30.5,
            "maximum_observed_zero_duration_word_attachment_distance_seconds": 0.0,
            "zero_duration_point_anchors_within_source_segment_verified": True,
            "zero_duration_assignment_anchor_indices_monotonic_verified": True,
            "positive_word_boundaries_preserved": True,
            "split_text_trimmed_exact_equivalence_verified": True,
            "reference_or_label_used": False,
            "boundary_selection_uses_text_content": False,
        },
        "timeline_coverage": coverage,
    }
    result["result_sha256"] = _canonical_sha256(result)
    results_root = asr_root / "results"
    results_root.mkdir()
    write_json(results_root / f"{lesson_id}.json", result)
    imported = import_teachobs_asr_results(
        job_path,
        inputs["media_manifest"],
        inputs["media_root"],
        inputs["caption_audit"],
        results_root,
        duration_probe=lambda _: duration,
        generated_at_utc="2026-07-23T00:30:00Z",
    )
    outputs = write_teachobs_asr_import(
        imported,
        audit_path=asr_root / "import_audit.json",
        coverage_matrix_path=asr_root / "coverage_matrix.json",
    )
    return {
        "job_manifest": job_path,
        "results": results_root,
        "import_audit": outputs["audit"],
        "coverage_matrix": outputs["coverage_matrix"],
    }


def test_materializes_fixed_profile_with_priority_and_exact_alignment(
    materialization_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "public" / "receipt.json"
    result = _run(
        materialization_inputs,
        tmp_path / "materialized",
        public_receipt=receipt_path,
    )
    manifest = result["manifest"]
    assert isinstance(manifest, dict)
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["profile_id"] == PAPER_TRACK1_PROFILE
    assert manifest["aggregate"]["lesson_count"] == 29
    assert manifest["aggregate"]["scene_count"] == 4_945
    assert "S4" not in [item["lesson_id"] for item in manifest["lessons"]]
    assert manifest["claims"]["labels_read_or_used"] is False
    assert manifest["claims"]["released_transcript_fallback_used"] is False

    validated = validate_teachobs_transcript_materialization(
        result["manifest_path"]
    )
    assert len(validated["rows"]) == 4_945
    first = validated["rows"][0]
    second = validated["rows"][1]
    assert first["sample_id"] == "S1:1"
    assert first["text"] == "hello world again"
    assert "boundary cue" not in first["text"]
    assert second["text"] == "boundary cue"
    assert validated["rows"][2]["text"] == ""
    assert validated["rows"][2]["empty_transcript"] is True

    lesson_by_id = {item["lesson_id"]: item for item in manifest["lessons"]}
    assert lesson_by_id["S1"]["source_track_type"] == "manual_creator_provided"
    assert lesson_by_id["S1"]["source_language"] == "en"
    assert lesson_by_id["S2"]["source_track_type"] == "manual_creator_provided"
    assert lesson_by_id["S2"]["source_language"] == "nl"
    assert validated["text_by_sample_id"]["S2:1"] == "original"

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    serialized = json.dumps(receipt)
    assert receipt["aggregate"]["scene_count"] == 4_945
    assert receipt["evidence_status"]["content_accuracy_established"] is False
    assert "hello world" not in serialized
    assert '"S1"' not in serialized
    if os.name == "posix":
        assert (Path(result["manifest_path"]).stat().st_mode & 0o777) == 0o600
        assert (
            (tmp_path / "materialized" / "lessons" / "S1.jsonl").stat().st_mode
            & 0o777
        ) == 0o600


def test_materialization_fingerprint_is_reproducible(
    materialization_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    first = _run(materialization_inputs, tmp_path / "first")["manifest"]
    second = _run(materialization_inputs, tmp_path / "second")["manifest"]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert (
        first["materialization_fingerprint_sha256"]
        == second["materialization_fingerprint_sha256"]
    )
    assert (
        first["ordered_scene_text_sha256"]
        == second["ordered_scene_text_sha256"]
    )
    assert [
        item["transcript_file_sha256"] for item in first["lessons"]
    ] == [item["transcript_file_sha256"] for item in second["lessons"]]


def test_clips_s17_like_final_cue_to_selected_timeline_without_text_loss(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(
        tmp_path / "inputs",
        platform_endpoint_case="clamp",
    )
    receipt_path = tmp_path / "public" / "receipt.json"
    result = _run(
        inputs,
        tmp_path / "materialized",
        public_receipt=receipt_path,
    )
    manifest = result["manifest"]
    lesson = next(
        item for item in manifest["lessons"] if item["lesson_id"] == "S17"
    )
    adjustment = lesson["source_timeline_adjustment"]
    assert adjustment["endpoint_clamped_item_count"] == 1
    assert adjustment["total_endpoint_clipped_seconds"] == 35.522
    assert adjustment["maximum_endpoint_clipped_seconds"] == 35.522
    assert adjustment["source_item_count"] == adjustment["retained_item_count"]
    assert adjustment["source_text_items_silently_dropped"] == 0
    assert adjustment["text_or_labels_used_for_timing_adjustment"] is False
    assert manifest["aggregate"][
        "platform_cue_endpoint_clamped_lesson_count"
    ] == 1
    assert manifest["aggregate"][
        "platform_cue_endpoint_clamped_item_count"
    ] == 1
    assert manifest["aggregate"][
        "platform_cue_total_endpoint_clipped_seconds"
    ] == 35.522
    validated = validate_teachobs_transcript_materialization(
        result["manifest_path"]
    )
    last_sample = f"S17:{lesson['scene_count']}"
    assert "long final cue" in validated["text_by_sample_id"][last_sample]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["aggregate"][
        "platform_cue_endpoint_clamped_item_count"
    ] == 1
    assert receipt["evidence_status"][
        "platform_cue_text_silently_dropped_during_timing_adjustment"
    ] is False


@pytest.mark.parametrize(
    ("endpoint_case", "message"),
    [
        ("starts_at_boundary", "no positive overlap"),
        ("starts_after_boundary", "no positive overlap"),
        ("invalid_duration", "invalid cue"),
    ],
)
def test_rejects_platform_cues_without_valid_positive_timeline_intersection(
    tmp_path: Path,
    endpoint_case: str,
    message: str,
) -> None:
    inputs = _build_inputs(
        tmp_path / "inputs",
        platform_endpoint_case=endpoint_case,
    )
    with pytest.raises(
        TeachObsTranscriptMaterializationError,
        match=message,
    ):
        _run(inputs, tmp_path / "materialized")


def test_validator_fails_closed_on_materialized_text_tamper(
    materialization_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    result = _run(materialization_inputs, tmp_path / "materialized")
    lesson_path = tmp_path / "materialized" / "lessons" / "S1.jsonl"
    lesson_path.write_text(
        lesson_path.read_text(encoding="utf-8").replace(
            "hello world again", "tampered text", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        TeachObsTranscriptMaterializationError,
        match="transcript hash mismatch",
    ):
        validate_teachobs_transcript_materialization(result["manifest_path"])


def test_materializer_fails_closed_on_caption_hash_tamper(
    materialization_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    source = materialization_inputs["caption_audit"].parent / "raw" / "S1" / "en.vtt"
    source.write_text(
        source.read_text(encoding="utf-8") + "\nNOTE changed\n",
        encoding="utf-8",
    )
    with pytest.raises(
        TeachObsTranscriptMaterializationError,
        match="caption hash mismatch",
    ):
        _run(materialization_inputs, tmp_path / "materialized")


def test_scene_alignment_uses_positive_overlap_and_deterministic_rolling_dedup() -> None:
    segments = [
        {"start": 0.0, "end": 4.0, "text": "alpha beta"},
        {"start": 4.0, "end": 8.0, "text": "beta gamma"},
        {"start": 15.0, "end": 20.0, "text": "boundary"},
    ]
    text, count = _scene_text(
        segments,
        start=0.0,
        end=15.0,
        rolling_caption_deduplication=True,
    )
    assert text == "alpha beta gamma"
    assert count == 2
    next_text, next_count = _scene_text(
        segments,
        start=15.0,
        end=30.0,
        rolling_caption_deduplication=True,
    )
    assert next_text == "boundary"
    assert next_count == 1


def test_materializes_hash_bound_audited_asr_only_as_fallback(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(
        tmp_path / "inputs",
        missing_selected_lesson="S9",
    )
    asr = _build_asr_chain(tmp_path, inputs, lesson_id="S9")
    result = materialize_teachobs_transcripts(
        inputs["media_plan"],
        inputs["media_manifest"],
        inputs["caption_audit"],
        asr["import_audit"],
        asr["coverage_matrix"],
        asr["job_manifest"],
        asr["results"],
        tmp_path / "materialized",
        generated_at_utc="2026-07-23T01:02:03Z",
    )
    manifest = result["manifest"]
    lesson = next(item for item in manifest["lessons"] if item["lesson_id"] == "S9")
    assert lesson["source_tier"] == "audited_asr_fallback"
    assert lesson["source_track_type"] == "automatic_speech_recognition"
    assert manifest["aggregate"]["source_tier_lesson_counts"][
        "audited_asr_fallback"
    ] == 1
    validated = validate_teachobs_transcript_materialization(
        result["manifest_path"]
    )
    assert validated["text_by_sample_id"]["S9:1"] == "asr beginning"
    assert manifest["claims"]["asr_is_official_caption"] is False
    assert manifest["input_hashes"]["asr_job_manifest_file_sha256"] == file_sha256(
        asr["job_manifest"]
    )


def test_explicitly_excludes_s29_like_asr_media_tail_from_target_scenes(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(
        tmp_path / "inputs",
        missing_selected_lesson="S29",
        media_duration_extension_seconds=3.14966,
    )
    asr = _build_asr_chain(
        tmp_path,
        inputs,
        lesson_id="S29",
        outside_selected_tail=True,
    )
    receipt_path = tmp_path / "public" / "receipt.json"
    result = materialize_teachobs_transcripts(
        inputs["media_plan"],
        inputs["media_manifest"],
        inputs["caption_audit"],
        asr["import_audit"],
        asr["coverage_matrix"],
        asr["job_manifest"],
        asr["results"],
        tmp_path / "materialized",
        public_receipt_path=receipt_path,
        generated_at_utc="2026-07-23T01:02:03Z",
    )
    manifest = result["manifest"]
    lesson = next(
        item for item in manifest["lessons"] if item["lesson_id"] == "S29"
    )
    adjustment = lesson["source_timeline_adjustment"]
    assert adjustment["source_item_count"] == 2
    assert adjustment["retained_item_count"] == 1
    assert adjustment["outside_selected_timeline_item_count"] == 1
    assert (
        adjustment["outside_selected_timeline_item_duration_seconds"] == 2.46
    )
    assert adjustment["endpoint_clamped_item_count"] == 0
    assert adjustment["all_source_items_within_hash_bound_media"] is True
    assert adjustment["source_text_items_silently_dropped"] == 0
    assert adjustment["text_or_labels_used_for_timing_adjustment"] is False
    assert manifest["aggregate"][
        "asr_outside_selected_timeline_item_count"
    ] == 1
    assert manifest["aggregate"][
        "asr_outside_selected_timeline_item_duration_seconds"
    ] == 2.46
    validated = validate_teachobs_transcript_materialization(
        result["manifest_path"]
    )
    assert validated["text_by_sample_id"]["S29:1"] == "asr beginning"
    assert "asr media-only tail" not in " ".join(
        validated["text_by_sample_id"].values()
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["aggregate"][
        "asr_outside_selected_timeline_item_count"
    ] == 1
    assert receipt["aggregate"][
        "asr_source_text_items_silently_dropped"
    ] == 0


def test_asr_target_window_projection_is_text_blind_and_media_bound() -> None:
    plan_lesson = {
        "scenes": [
            {"start": 0.0, "end": 15.0},
            {"start": 15.0, "end": 30.0},
        ]
    }
    first_result = {
        "media_binding": {"media_duration_seconds": 33.0},
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": 30.25, "end": 32.5, "text": "private tail A"},
        ],
    }
    second_result = {
        **first_result,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "different"},
            {"start": 30.25, "end": 32.5, "text": "private tail B"},
        ],
    }
    first_segments, first_adjustment = (
        _project_audited_asr_segments_to_selected_timeline(
            first_result,
            plan_lesson=plan_lesson,
        )
    )
    second_segments, second_adjustment = (
        _project_audited_asr_segments_to_selected_timeline(
            second_result,
            plan_lesson=plan_lesson,
        )
    )
    assert first_adjustment == second_adjustment
    assert [item["text"] for item in first_segments] == ["first"]
    assert [item["text"] for item in second_segments] == ["different"]

    outside_media = {
        **first_result,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "valid"},
            {"start": 32.5, "end": 33.1, "text": "outside media"},
        ],
    }
    with pytest.raises(
        TeachObsTranscriptMaterializationError,
        match="not within the hash-bound media timeline",
    ):
        _project_audited_asr_segments_to_selected_timeline(
            outside_media,
            plan_lesson=plan_lesson,
        )


def test_public_receipt_rejects_claim_tampering(
    materialization_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    manifest = _run(materialization_inputs, tmp_path / "materialized")["manifest"]
    assert isinstance(manifest, dict)
    manifest["claims"]["content_accuracy_established"] = True
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = _canonical_sha256(unsigned)
    with pytest.raises(
        TeachObsTranscriptMaterializationError,
        match="claim boundary",
    ):
        build_public_teachobs_transcript_materialization_receipt(manifest)
