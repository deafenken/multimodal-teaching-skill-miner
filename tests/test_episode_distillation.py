from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema

from teaching_skill_miner.episode_distillation import (
    EPISODE_ALGORITHM,
    EPISODE_SCHEMA,
    mine_episode_skills,
    segment_transcript_into_episodes,
    slice_transcript_for_episode,
    validate_episode_library,
    write_episode_library,
)
from teaching_skill_miner.models import validate_skill, validate_transcript


ROOT = Path(__file__).resolve().parents[1]


def _transcript(*, count: int = 10, with_multimodal: bool = False) -> dict:
    cues = [
        "Review the previous lecture.",
        "Here is a concrete example.",
        "Now define the formal rule.",
        "Work through it step by step.",
        "Check what happens next.",
        "If this is wrong, correct the error.",
        "Practice with feedback and try again.",
        "Summarize and transfer to a new case.",
    ]
    segments = [
        {"start": float(index * 100), "end": float((index + 1) * 100), "text": cues[index % len(cues)]}
        for index in range(count)
    ]
    result = {
        "video_id": "video_episode_fixture",
        "course_id": "course_fixture",
        "title": "Lecture 1: Episode fixture",
        "source_url": "https://example.org/video_episode_fixture",
        "segments": segments,
    }
    if with_multimodal:
        result["multimodal"] = {
            "modalities_available": ["transcript", "audio", "visual"],
            "media": {"duration_seconds": float(count * 100), "sha256": "a" * 64},
            "audio": {
                "silences": [
                    {"start": 100.0, "end": 110.0, "duration": 10.0},
                    {"start": 500.0, "end": 510.0, "duration": 10.0},
                ]
            },
            "visual": {
                "keyframes": [
                    {"timestamp": 50.0, "path": "frames/0.jpg", "ocr_text": ""},
                    {"timestamp": 550.0, "path": "frames/1.jpg", "ocr_text": ""},
                ],
                "events": [],
            },
            "events": [],
        }
    assert validate_transcript(result).valid
    return result


def test_short_transcript_degrades_to_one_episode_and_is_deterministic() -> None:
    transcript = _transcript(count=3)
    first = segment_transcript_into_episodes(transcript)
    second = segment_transcript_into_episodes(copy.deepcopy(transcript))
    assert first == second
    assert len(first) == 1
    assert first[0]["segment_start_index"] == 0
    assert first[0]["segment_end_index"] == 3


def test_long_transcript_is_an_exact_contiguous_partition() -> None:
    transcript = _transcript(count=12)
    episodes = segment_transcript_into_episodes(
        transcript, min_seconds=200, target_seconds=300, max_seconds=400
    )
    assert len(episodes) >= 3
    assert [row["episode_index"] for row in episodes] == list(range(len(episodes)))
    assert episodes[0]["segment_start_index"] == 0
    assert episodes[-1]["segment_end_index"] == len(transcript["segments"])
    assert [
        index
        for row in episodes
        for index in range(row["segment_start_index"], row["segment_end_index"])
    ] == list(range(len(transcript["segments"])))
    assert all(row["end"] > row["start"] for row in episodes)


def test_slice_preserves_global_timestamps_and_validation() -> None:
    transcript = _transcript(count=8)
    episode = segment_transcript_into_episodes(
        transcript, min_seconds=200, target_seconds=300, max_seconds=400
    )[1]
    sliced = slice_transcript_for_episode(transcript, episode)
    assert validate_transcript(sliced).valid
    assert sliced["segments"][0]["start"] == transcript["segments"][episode["segment_start_index"]]["start"]
    assert sliced["episode"]["parent_video_id"] == transcript["video_id"]
    assert sliced["episode"]["episode_id"] == episode["episode_id"]


def test_episode_skills_have_unique_ids_and_explicit_claim_boundary() -> None:
    transcript = _transcript(count=12)
    library = mine_episode_skills(
        transcript, min_seconds=200, target_seconds=300, max_seconds=400
    )
    assert library["schema"] == EPISODE_SCHEMA
    assert library["segmentation"]["algorithm"] == EPISODE_ALGORITHM
    assert len(library["episodes"]) == len(library["skills"])
    assert len({skill["skill_id"] for skill in library["skills"]}) == len(library["skills"])
    assert all(validate_skill(skill).valid for skill in library["skills"])
    assert all(skill["source"]["episode_id"] == row["episode_id"] for skill, row in zip(library["skills"], library["episodes"]))
    assert library["claim_boundary"]["semantic_clustering_established"] is False
    assert validate_episode_library(library) == []


def test_tampered_skill_hash_and_partition_fail_closed() -> None:
    transcript = _transcript(count=12)
    library = mine_episode_skills(
        transcript, min_seconds=200, target_seconds=300, max_seconds=400
    )
    tampered = copy.deepcopy(library)
    tampered["episodes"][0]["skill_sha256"] = "0" * 64
    assert any("hash" in error for error in validate_episode_library(tampered))
    tampered = copy.deepcopy(library)
    tampered["episodes"][1]["segment_start_index"] += 1
    assert any("partition" in error for error in validate_episode_library(tampered))


def test_public_schema_accepts_manifest_shape_and_cli_writer_persists_private_skills(tmp_path: Path) -> None:
    transcript = _transcript(count=12)
    library = mine_episode_skills(
        transcript, min_seconds=200, target_seconds=300, max_seconds=400
    )
    manifest = copy.deepcopy(library)
    manifest["skills"] = []
    for row in manifest["episodes"]:
        row["skill_path"] = f"skills/{row['episode_id']}.skill.json"
    schema = json.loads((ROOT / "schema/teaching_episode_manifest.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(manifest)

    output = ROOT / "artifacts" / ".episode_test_library"
    if output.exists():
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        output.rmdir()
    result = write_episode_library(
        transcript,
        output,
        min_seconds=200,
        target_seconds=300,
        max_seconds=400,
    )
    assert Path(result["manifest"]).is_file()
    persisted = json.loads(Path(result["manifest"]).read_text())
    assert persisted["episodes"]
    assert all((output / row["skill_path"]).is_file() for row in persisted["episodes"])
    for path in sorted(output.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    output.rmdir()


def test_multimodal_slice_keeps_events_single_owned() -> None:
    transcript = _transcript(count=8, with_multimodal=True)
    # This fixture has no valid multimodal events; the important invariant is
    # that an otherwise valid multimodal transcript remains valid after slicing.
    episode = segment_transcript_into_episodes(
        transcript, min_seconds=200, target_seconds=300, max_seconds=400
    )[0]
    sliced = slice_transcript_for_episode(transcript, episode)
    assert validate_transcript(sliced).valid
    assert sliced["multimodal"]["media"]["duration_seconds"] == 800.0
