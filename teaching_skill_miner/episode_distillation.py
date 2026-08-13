"""Deterministic long-video episode slicing and per-episode Skill mining.

The long-form extractor intentionally produces one hash-bound timeline for a
video.  This module is an optional projection on top of that timeline: it
partitions transcript segments at deterministic boundaries and mines one
evidence-aware Skill per slice.  The boundary detector is deliberately
interpretable and conservative; it is *not* a semantic topic clusterer and it
does not establish an expert gold set or recognition accuracy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .io_utils import ensure_private_directory, read_json, write_json
from .miner import mine_skill
from .models import validate_skill, validate_transcript


EPISODE_SCHEMA = "teaching_skill_miner.teaching_episode_manifest.v1"
EPISODE_ALGORITHM = "deterministic_boundary_segmentation_v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _validate_config(min_seconds: float, target_seconds: float, max_seconds: float) -> None:
    minimum = _number(min_seconds, "min_seconds")
    target = _number(target_seconds, "target_seconds")
    maximum = _number(max_seconds, "max_seconds")
    if minimum <= 0 or target <= 0 or maximum <= 0:
        raise ValueError("episode durations must be positive")
    if not minimum <= target <= maximum:
        raise ValueError("episode durations must satisfy min <= target <= max")


def _segment_rows(transcript: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = transcript.get("segments")
    if not isinstance(raw, list) or not raw:
        raise ValueError("transcript must contain at least one segment")
    rows: list[dict[str, Any]] = []
    previous_end = -math.inf
    for index, segment in enumerate(raw):
        if not isinstance(segment, Mapping):
            raise ValueError(f"segments[{index}] must be an object")
        start = _number(segment.get("start"), f"segments[{index}].start")
        end = _number(segment.get("end"), f"segments[{index}].end")
        if start < 0 or end <= start:
            raise ValueError(f"segments[{index}] must satisfy 0 <= start < end")
        if start < previous_end:
            raise ValueError("transcript segments must be ordered and non-overlapping")
        previous_end = end
        rows.append({"index": index, "start": start, "end": end})
    return rows


def _episode_id(
    video_id: str,
    start_index: int,
    end_index: int,
    start: float,
    end: float,
) -> str:
    material = "\x1f".join(
        (video_id, str(start_index), str(end_index), f"{start:.6f}", f"{end:.6f}")
    ).encode("utf-8")
    return "ep_" + hashlib.sha256(material).hexdigest()[:16]


def _choose_cut(
    rows: list[dict[str, Any]],
    start_index: int,
    *,
    min_seconds: float,
    target_seconds: float,
    max_seconds: float,
) -> int:
    """Choose the exclusive segment index for the next episode.

    Candidate boundaries are actual segment ends.  We prefer the boundary
    closest to the target while respecting max duration and, when possible,
    leaving a valid minimum-sized remainder.  This makes repeated runs and
    different input ordering deterministic without inventing timestamps.
    """

    start = rows[start_index]["start"]
    last = len(rows)
    candidates: list[tuple[float, float, int, float]] = []
    for exclusive in range(start_index + 1, last + 1):
        end = rows[exclusive - 1]["end"]
        duration = end - start
        if duration > max_seconds + 1e-9:
            break
        remainder = rows[-1]["end"] - end if exclusive < last else 0.0
        # A remainder smaller than min is only acceptable for the final slice;
        # otherwise the next slice would violate the requested lower bound.
        if exclusive < last and remainder + 1e-9 < min_seconds:
            continue
        candidates.append((abs(duration - target_seconds), -duration, exclusive, duration))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][2]

    # If the remaining tail is shorter than min, consume it as one final slice.
    if rows[-1]["end"] - start <= max_seconds + 1e-9:
        return last

    # A single source segment longer than max cannot be split without
    # fabricating a boundary.  Fail closed rather than violating the contract.
    raise ValueError(
        "cannot satisfy episode max_seconds at a source-segment boundary; "
        "lower max_seconds or provide finer transcript segmentation"
    )


def segment_transcript_into_episodes(
    transcript: Mapping[str, Any],
    *,
    min_seconds: float = 240.0,
    target_seconds: float = 600.0,
    max_seconds: float = 900.0,
) -> list[dict[str, Any]]:
    """Partition a transcript's segments into deterministic contiguous episodes.

    Returned rows use ``segment_start_index`` (inclusive) and
    ``segment_end_index`` (exclusive).  Every input segment appears exactly
    once.  A short transcript becomes one episode, even when it is shorter than
    ``min_seconds``; this is necessary to avoid dropping content.
    """

    _validate_config(min_seconds, target_seconds, max_seconds)
    rows = _segment_rows(transcript)
    video_id = str(transcript.get("video_id", ""))
    if not video_id:
        raise ValueError("transcript.video_id is required")
    total_duration = rows[-1]["end"] - rows[0]["start"]
    if total_duration <= max_seconds + 1e-9:
        boundaries = [(0, len(rows))]
    else:
        boundaries: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(rows):
            exclusive = _choose_cut(
                rows,
                cursor,
                min_seconds=float(min_seconds),
                target_seconds=float(target_seconds),
                max_seconds=float(max_seconds),
            )
            boundaries.append((cursor, exclusive))
            cursor = exclusive

    episodes: list[dict[str, Any]] = []
    for episode_index, (start_index, end_index) in enumerate(boundaries):
        start = rows[start_index]["start"]
        end = rows[end_index - 1]["end"]
        episode_id = _episode_id(video_id, start_index, end_index - 1, start, end)
        episodes.append(
            {
                "episode_id": episode_id,
                "episode_index": episode_index,
                "start": round(start, 6),
                "end": round(end, 6),
                "duration_seconds": round(end - start, 6),
                "segment_start_index": start_index,
                "segment_end_index": end_index,
                "segment_count": end_index - start_index,
            }
        )
    return episodes


def _event_anchor(event: Mapping[str, Any]) -> float | None:
    try:
        start = float(event.get("start"))
        end = float(event.get("end"))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    return (start + end) / 2.0


def _event_selected(event: Mapping[str, Any], start: float, end: float, final: bool) -> bool:
    anchor = _event_anchor(event)
    if anchor is None:
        return False
    return start <= anchor <= end if final else start <= anchor < end


def _referenced_paths(event: Mapping[str, Any]) -> set[str]:
    evidence = event.get("evidence")
    if not isinstance(evidence, Mapping):
        return set()
    return {
        str(evidence[key])
        for key in ("frame_path", "before_frame", "after_frame")
        if evidence.get(key)
    }


def _referenced_silences(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = event.get("evidence")
    silence = evidence.get("silence") if isinstance(evidence, Mapping) else None
    return [dict(silence)] if isinstance(silence, Mapping) else []


def slice_transcript_for_episode(
    transcript: Mapping[str, Any], episode: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a validation-preserving transcript projection for one episode."""

    rows = _segment_rows(transcript)
    start_index = int(episode["segment_start_index"])
    end_index = int(episode["segment_end_index"])
    if not 0 <= start_index < end_index <= len(rows):
        raise ValueError("episode segment range is invalid")
    result = copy.deepcopy(dict(transcript))
    result["segments"] = []
    for parent_index, raw_segment in enumerate(
        list(transcript["segments"])[start_index:end_index], start=start_index  # type: ignore[index]
    ):
        segment = copy.deepcopy(raw_segment)
        # ``mine_skill`` uses this marker only for stable evidence IDs.  It is
        # deliberately kept outside the public transcript contract and is
        # stripped by downstream serializers when a raw transcript is needed.
        segment["_parent_segment_index"] = parent_index
        result["segments"].append(segment)
    result["episode"] = {
        "schema": EPISODE_SCHEMA,
        "episode_id": str(episode["episode_id"]),
        "episode_index": int(episode["episode_index"]),
        "start": float(episode["start"]),
        "end": float(episode["end"]),
        "parent_video_id": str(transcript["video_id"]),
        "segment_start_index": start_index,
        "segment_end_index": end_index,
    }

    multimodal = result.get("multimodal")
    if not isinstance(multimodal, dict):
        validation = validate_transcript(result)
        if not validation.valid:
            raise ValueError("episode transcript is invalid: " + "; ".join(validation.errors))
        return result

    start = float(episode["start"])
    end = float(episode["end"])
    final = end >= max(row["end"] for row in rows) - 1e-9
    selected_events_by_id: dict[str, dict[str, Any]] = {}
    for collection in (
        multimodal.get("events", []),
        multimodal.get("visual", {}).get("events", [])
        if isinstance(multimodal.get("visual"), dict)
        else [],
    ):
        if not isinstance(collection, list):
            continue
        for event in collection:
            if isinstance(event, Mapping) and _event_selected(event, start, end, final):
                event_id = str(event.get("event_id", ""))
                if event_id:
                    selected_events_by_id[event_id] = copy.deepcopy(dict(event))

    selected_paths = {
        path
        for event in selected_events_by_id.values()
        for path in _referenced_paths(event)
    }
    visual = multimodal.get("visual")
    if isinstance(visual, dict) and isinstance(visual.get("keyframes"), list):
        visual["keyframes"] = [
            copy.deepcopy(frame)
            for frame in visual["keyframes"]
            if isinstance(frame, Mapping)
            and (
                str(frame.get("path", "")) in selected_paths
                or start <= _number(frame.get("timestamp", 0), "frame.timestamp") <= end
            )
        ]
        # Keep frame evidence referenced by an event even if the frame timestamp
        # is just outside the nominal interval at a boundary.
        known_paths = {str(frame.get("path", "")) for frame in visual["keyframes"]}
        for event in selected_events_by_id.values():
            for path in _referenced_paths(event) - known_paths:
                for frame in multimodal.get("visual", {}).get("keyframes", []):
                    if isinstance(frame, Mapping) and str(frame.get("path", "")) == path:
                        visual["keyframes"].append(copy.deepcopy(frame))
                        known_paths.add(path)
                        break
        visual["events"] = [
            event for event in visual.get("events", [])
            if isinstance(event, Mapping) and str(event.get("event_id", "")) in selected_events_by_id
        ]

    if isinstance(multimodal.get("events"), list):
        multimodal["events"] = [
            event for event in multimodal["events"]
            if isinstance(event, Mapping) and str(event.get("event_id", "")) in selected_events_by_id
        ]

    audio = multimodal.get("audio")
    if isinstance(audio, dict) and isinstance(audio.get("silences"), list):
        selected_silences = [
            dict(silence)
            for silence in audio["silences"]
            if isinstance(silence, Mapping)
            and _silence_overlaps(silence, start, end)
        ]
        for event in selected_events_by_id.values():
            for silence in _referenced_silences(event):
                if silence not in selected_silences:
                    selected_silences.append(silence)
        audio["silences"] = selected_silences

    validation = validate_transcript(result)
    if not validation.valid:
        raise ValueError("episode transcript is invalid: " + "; ".join(validation.errors))
    return result


def _silence_overlaps(silence: Mapping[str, Any], start: float, end: float) -> bool:
    try:
        silence_start = float(silence.get("start"))
        silence_end = float(silence.get("end"))
    except (TypeError, ValueError, OverflowError):
        return False
    return silence_end >= start and silence_start <= end


def mine_episode_skills(
    transcript: Mapping[str, Any],
    *,
    min_seconds: float = 240.0,
    target_seconds: float = 600.0,
    max_seconds: float = 900.0,
) -> dict[str, Any]:
    """Return an auditable per-video episode Skill library in memory."""

    source_validation = validate_transcript(dict(transcript))
    if not source_validation.valid:
        raise ValueError("invalid transcript: " + "; ".join(source_validation.errors))
    episodes = segment_transcript_into_episodes(
        transcript,
        min_seconds=min_seconds,
        target_seconds=target_seconds,
        max_seconds=max_seconds,
    )
    transcript_fingerprint = canonical_sha256(transcript)
    config = {
        "algorithm": EPISODE_ALGORITHM,
        "min_seconds": float(min_seconds),
        "target_seconds": float(target_seconds),
        "max_seconds": float(max_seconds),
    }
    config_fingerprint = canonical_sha256(config)
    skills: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        sliced = slice_transcript_for_episode(transcript, episode)
        skill = mine_skill(sliced)
        episode_id = str(episode["episode_id"])
        skill["skill_id"] = f"{skill['skill_id']}_{episode_id}"
        skill["source"].update(
            {
                "episode_id": episode_id,
                "parent_video_id": str(transcript["video_id"]),
                "episode_start": episode["start"],
                "episode_end": episode["end"],
                "parent_transcript_sha256": transcript_fingerprint,
                "segment_indices": list(
                    range(episode["segment_start_index"], episode["segment_end_index"])
                ),
                "event_ids": sorted(
                    str(event.get("event_id"))
                    for event in sliced.get("multimodal", {}).get("events", [])
                    if isinstance(event, Mapping) and event.get("event_id")
                ),
            }
        )
        skill["mining_metadata"]["episode_segmentation"] = {
            "algorithm": EPISODE_ALGORITHM,
            "config_sha256": config_fingerprint,
            "episode_input_sha256": canonical_sha256(sliced),
            "parent_transcript_sha256": transcript_fingerprint,
            "segment_start_index": episode["segment_start_index"],
            "segment_end_index": episode["segment_end_index"],
        }
        validation = validate_skill(skill)
        if not validation.valid:
            raise ValueError(
                f"episode {episode_id} produced invalid Skill: "
                + "; ".join(validation.errors)
            )
        skills.append(skill)
        rows.append(
            {
                **episode,
                "skill_id": skill["skill_id"],
                "skill_sha256": canonical_sha256(skill),
                "episode_input_sha256": canonical_sha256(sliced),
                "observed_phase_count": sum(
                    step.get("origin") == "observed_method"
                    for step in skill.get("procedure", [])
                ),
                "event_count": len(
                    sliced.get("multimodal", {}).get("events", [])
                    if isinstance(sliced.get("multimodal"), dict)
                    else []
                ),
            }
        )

    return {
        "schema": EPISODE_SCHEMA,
        "artifact_kind": "per_video_episode_skill_library",
        "status": "heuristic_boundary_slices",
        "video_id": str(transcript["video_id"]),
        "course_id": str(transcript["course_id"]),
        "media_sha256": (
            transcript.get("multimodal", {}).get("media", {}).get("sha256")
            if isinstance(transcript.get("multimodal"), dict)
            else None
        ),
        "parent_transcript_sha256": transcript_fingerprint,
        "segmentation": {**config, "config_sha256": config_fingerprint},
        "episodes": rows,
        "skills": skills,
        "claim_boundary": {
            "episode_boundaries_heuristic": True,
            "semantic_clustering_established": False,
            "episode_gold_established": False,
            "expert_skill_gold_established": False,
            "recognition_accuracy_established": False,
            "teaching_effectiveness_established": False,
        },
    }


def validate_episode_library(library: Mapping[str, Any]) -> list[str]:
    """Validate semantic invariants not expressible in JSON Schema."""

    errors: list[str] = []
    if library.get("schema") != EPISODE_SCHEMA:
        errors.append("unsupported episode library schema")
    episodes = library.get("episodes")
    skills = library.get("skills")
    if not isinstance(episodes, list) or not isinstance(skills, list):
        return errors + ["episodes and skills must be arrays"]
    compact = len(skills) == 0 and all(
        isinstance(episode, Mapping) and isinstance(episode.get("skill_path"), str)
        and bool(episode.get("skill_path"))
        for episode in episodes
    )
    if (not compact and len(episodes) != len(skills)) or not episodes:
        errors.append("episodes and skills must have the same non-zero length")
        return errors
    seen_segments: list[int] = []
    previous_end = 0
    video_id = str(library.get("video_id", ""))
    for index, episode in enumerate(episodes):
        skill = skills[index] if not compact else None
        if not isinstance(episode, Mapping) or (skill is not None and not isinstance(skill, Mapping)):
            errors.append(f"episode/skill row {index} must be objects")
            continue
        if episode.get("episode_index") != index:
            errors.append("episode indexes must be contiguous")
        start_index = episode.get("segment_start_index")
        end_index = episode.get("segment_end_index")
        if not isinstance(start_index, int) or not isinstance(end_index, int):
            errors.append(f"episode {index} segment indexes must be integers")
        elif start_index != previous_end or end_index <= start_index:
            errors.append(f"episode {index} segment partition is not contiguous")
        else:
            previous_end = end_index
            seen_segments.extend(range(start_index, end_index))
        if skill is not None:
            if skill.get("skill_id") != episode.get("skill_id"):
                errors.append(f"episode {index} skill_id does not match skill payload")
            source = skill.get("source")
            if not isinstance(source, Mapping) or source.get("video_id") != video_id:
                errors.append(f"episode {index} skill source video_id mismatch")
            if source.get("episode_id") != episode.get("episode_id"):
                errors.append(f"episode {index} source episode_id mismatch")
            if not validate_skill(dict(skill)).valid:
                errors.append(f"episode {index} skill payload is invalid")
            expected_hash = canonical_sha256(skill)
            if episode.get("skill_sha256") != expected_hash:
                errors.append(f"episode {index} skill hash mismatch")
    if seen_segments and seen_segments != list(range(previous_end)):
        errors.append("episode segment partition contains a gap or overlap")
    return errors


def write_episode_library(
    transcript: Mapping[str, Any],
    output_dir: str | Path,
    **options: Any,
) -> dict[str, Any]:
    """Mine and persist a private episode library."""

    library = mine_episode_skills(transcript, **options)
    errors = validate_episode_library(library)
    if errors:
        raise ValueError("invalid episode library: " + "; ".join(errors))
    output = ensure_private_directory(output_dir)
    skill_dir = ensure_private_directory(output / "skills")
    rows = []
    for row, skill in zip(library["episodes"], library["skills"]):
        path = write_json(skill_dir / f"{row['episode_id']}.skill.json", skill)
        row = dict(row)
        row["skill_path"] = str(path.relative_to(output))
        rows.append(row)
    library["episodes"] = rows
    library["skills"] = []
    library_path = write_json(output / "episode_manifest.json", library)
    # Keep the full Skill payloads private and separate from the compact
    # manifest.  Re-load them when validating the persisted form.
    persisted = read_json(library_path)
    persisted["skills"] = [
        read_json(output / str(row["skill_path"])) for row in persisted["episodes"]
    ]
    errors = validate_episode_library(persisted)
    if errors:
        raise ValueError("persisted episode library failed validation: " + "; ".join(errors))
    return {
        "manifest": str(library_path.resolve()),
        "video_id": library["video_id"],
        "episode_count": len(library["episodes"]),
        "claim_boundary": library["claim_boundary"],
    }


def _resolve_path(base: Path, value: str) -> Path:
    candidate = Path(value)
    for path in (candidate, base / candidate, Path.cwd() / candidate):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(value)


def write_episode_libraries_from_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    **options: Any,
) -> list[dict[str, Any]]:
    """Process every transcript listed by a dataset manifest."""

    manifest_file = Path(manifest_path).resolve()
    manifest = read_json(manifest_file)
    output = ensure_private_directory(output_dir)
    results: list[dict[str, Any]] = []
    for item in manifest.get("videos", []):
        if not isinstance(item, Mapping):
            raise ValueError("manifest videos must contain objects")
        transcript_path = _resolve_path(manifest_file.parent, str(item["transcript_path"]))
        target = output / str(item["video_id"])
        results.append(write_episode_library(read_json(transcript_path), target, **options))
    index = {
        "schema": EPISODE_SCHEMA,
        "artifact_kind": "per_video_episode_skill_library_index",
        "status": "index",
        "source_manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "video_count": len(results),
        "libraries": results,
        "claim_boundary": {
            "episode_boundaries_heuristic": True,
            "semantic_clustering_established": False,
            "episode_gold_established": False,
            "expert_skill_gold_established": False,
            "recognition_accuracy_established": False,
            "teaching_effectiveness_established": False,
        },
    }
    write_json(output / "episode_libraries.json", index)
    return results


__all__ = [
    "EPISODE_ALGORITHM",
    "EPISODE_SCHEMA",
    "canonical_sha256",
    "mine_episode_skills",
    "segment_transcript_into_episodes",
    "slice_transcript_for_episode",
    "validate_episode_library",
    "write_episode_library",
    "write_episode_libraries_from_manifest",
]
