"""Deterministic episode partitioning and provenance-safe multi-Skill mining.

The long-form pipeline produces one hash-bound transcript for each video.  This
module adds a deliberately conservative second layer: it proposes temporal
episode boundaries, slices the parent transcript without renumbering evidence,
and mines one ordinary Teaching Skill per proposed episode.  The partition is
an engineering heuristic, not an expert annotation or a recognition model.

All public helpers are deterministic.  In particular, multimodal events are
sorted by stable identifiers before they influence a boundary and every text
evidence id is generated from the parent transcript's global segment index.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .evaluator import evaluate_skill
from .io_utils import ensure_private_directory, read_json, write_json
from .miner import mine_skill
from .models import validate_skill, validate_transcript


EPISODE_PARTITION_SCHEMA = "teaching_skill_miner.episode_partition.v1"
EPISODE_SKILL_SCHEMA = "teaching_skill_miner.episode_skill.v1"
EPISODE_RECEIPT_SCHEMA = "teaching_skill_miner.episode_receipt.v1"
PARTITION_VERSION = "episode_partition_v1"

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_DEFAULT_MIN_EPISODE_SECONDS = 180.0
_DEFAULT_MAX_EPISODE_SECONDS = 900.0
_DEFAULT_TARGET_EPISODE_SECONDS = 480.0
_BOUNDARY_GAP_SECONDS = 8.0
_EVENT_BOUNDARY_TYPES = {"slide_change", "scene_change", "board_build_up"}


def _canonical_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    """Return the SHA-256 of a canonical JSON representation."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def transcript_sha256(transcript: dict[str, Any]) -> str:
    """Fingerprint a parent transcript, excluding internal slice markers."""

    value = copy.deepcopy(transcript)
    for segment in value.get("segments", []):
        if isinstance(segment, dict):
            segment.pop("_parent_segment_index", None)
    return sha256_json(value)


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _media_duration(transcript: dict[str, Any]) -> float:
    duration = _as_float(
        transcript.get("multimodal", {}).get("media", {}).get("duration_seconds")
    )
    if duration is not None and duration > 0:
        return duration
    segments = transcript.get("segments", [])
    ends = [_as_float(item.get("end"), 0.0) or 0.0 for item in segments if isinstance(item, dict)]
    return max(ends, default=0.0)


def _segment_tokens(text: Any) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(str(text or "")) if len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _window_tokens(segments: list[dict[str, Any]], start: int, end: int) -> set[str]:
    tokens: set[str] = set()
    for segment in segments[start:end]:
        tokens.update(_segment_tokens(segment.get("text")))
    return tokens


def _event_rows(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    events = transcript.get("multimodal", {}).get("events", [])
    if not isinstance(events, list):
        return []
    rows = [event for event in events if isinstance(event, dict)]
    return sorted(
        rows,
        key=lambda event: (
            _as_float(event.get("start"), 0.0) or 0.0,
            _as_float(event.get("end"), 0.0) or 0.0,
            str(event.get("event_id", "")),
        ),
    )


def _boundary_candidates(
    transcript: dict[str, Any],
    *,
    lexical_window: int = 4,
) -> list[dict[str, Any]]:
    """Score every legal boundary between adjacent transcript segments."""

    segments = [item for item in transcript.get("segments", []) if isinstance(item, dict)]
    events = _event_rows(transcript)
    candidates: list[dict[str, Any]] = []
    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]
        previous_end = _as_float(previous.get("end"), 0.0) or 0.0
        current_start = _as_float(current.get("start"), previous_end) or previous_end
        gap = max(0.0, current_start - previous_end)
        left = _window_tokens(segments, max(0, index - lexical_window), index)
        right = _window_tokens(segments, index, min(len(segments), index + lexical_window))
        lexical_shift = 1.0 - _jaccard(left, right)
        # A scene/slide event close to a transcript gap is useful corroboration,
        # but event order never affects the result because rows are sorted.
        nearby_events = [
            event
            for event in events
            if abs((_as_float(event.get("start"), 0.0) or 0.0) - current_start) <= 12.0
            or abs((_as_float(event.get("end"), 0.0) or 0.0) - previous_end) <= 12.0
        ]
        boundary_events = [
            str(event.get("event_id"))
            for event in nearby_events
            if str(event.get("type", "")) in _EVENT_BOUNDARY_TYPES
        ]
        event_signal = min(1.0, len(boundary_events) / 2.0)
        gap_signal = min(1.0, gap / _BOUNDARY_GAP_SECONDS)
        # Lexical change is the primary signal; temporal/media boundaries are
        # corroboration.  Keep the raw components for an auditable rationale.
        score = round(0.58 * lexical_shift + 0.27 * event_signal + 0.15 * gap_signal, 6)
        boundary_time = previous_end + gap / 2.0
        if gap <= 1e-6:
            boundary_time = previous_end
        candidates.append(
            {
                "after_segment_index": index - 1,
                "before_segment_index": index,
                "time": round(boundary_time, 6),
                "score": score,
                "lexical_shift": round(lexical_shift, 6),
                "event_signal": round(event_signal, 6),
                "gap_seconds": round(gap, 6),
                "event_ids": sorted(set(boundary_events)),
            }
        )
    return candidates


def _choose_cut_indices(
    transcript: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    min_episode_seconds: float,
    max_episode_seconds: float,
    target_episode_seconds: float,
) -> list[int]:
    """Choose segment-index cuts while enforcing duration bounds where possible."""

    segments = [item for item in transcript.get("segments", []) if isinstance(item, dict)]
    if len(segments) < 2:
        return []
    duration = _media_duration(transcript)
    starts = [_as_float(item.get("start"), 0.0) or 0.0 for item in segments]
    ends = [_as_float(item.get("end"), starts[i]) or starts[i] for i, item in enumerate(segments)]
    by_index = {int(row["before_segment_index"]): row for row in candidates}
    cuts: list[int] = []
    episode_start_time = 0.0
    episode_start_index = 0

    while episode_start_index < len(segments) - 1:
        if duration <= max_episode_seconds and not cuts:
            break
        # Candidate cuts must leave a minimum-duration prefix and respect the
        # maximum duration.  If none exists, use the nearest legal segment so a
        # pathological caption gap cannot create an uncovered tail.
        legal: list[tuple[float, int, dict[str, Any]]] = []
        for before_index, row in by_index.items():
            if before_index < episode_start_index or before_index >= len(segments) - 1:
                continue
            cut_time = float(row["time"])
            prefix_duration = cut_time - episode_start_time
            if prefix_duration + 1e-6 < min_episode_seconds:
                continue
            if prefix_duration > max_episode_seconds + 1e-6:
                continue
            distance = abs(prefix_duration - target_episode_seconds)
            # Prefer a strong natural boundary, then closeness to target.
            rank = (-float(row["score"]), distance, before_index)
            legal.append((rank[0] * -1 + distance * 0.001, before_index, row))
        if not legal:
            fallback = [
                (abs((float(row["time"]) - episode_start_time) - max_episode_seconds), idx, row)
                for idx, row in by_index.items()
                if episode_start_index <= idx < len(segments) - 1
            ]
            if not fallback:
                break
            _, before_index, row = min(fallback, key=lambda item: (item[0], item[1]))
        else:
            # Reconstruct the intended ordering explicitly; the odd-looking
            # tuple above keeps this branch independent of dict insertion order.
            _, before_index, row = min(
                legal,
                key=lambda item: (-float(item[2]["score"]), abs(float(item[2]["time"]) - episode_start_time - target_episode_seconds), item[1]),
            )
        if before_index < episode_start_index or before_index >= len(segments) - 1:
            break
        # ``before_segment_index`` is the first segment of the next episode;
        # the boundary lies between ``before_index - 1`` and ``before_index``.
        cuts.append(before_index)
        episode_start_index = before_index
        episode_start_time = float(row["time"])
        if len(cuts) >= len(segments) - 1:
            break
        # Avoid an accidental infinite loop in malformed timelines.
        if len(cuts) > len(segments):
            break

    # For a long tail, add cuts from the end backwards if the forward loop did
    # not need to run (e.g. no legal natural boundary before max duration).
    while True:
        starts_at = [0] + list(cuts)
        ends_at = [cut - 1 for cut in cuts] + [len(segments) - 1]
        oversized = [
            (ends_at[i] - starts_at[i], i)
            for i in range(len(starts_at))
            if ends[ends_at[i]] - (0.0 if i == 0 else starts[starts_at[i]]) > max_episode_seconds + 1e-6
        ]
        if not oversized:
            break
        _, episode_number = max(oversized)
        lo = starts_at[episode_number]
        hi = ends_at[episode_number] - 1
        possible = [
            (abs((float(by_index[idx]["time"]) - (0.0 if episode_number == 0 else starts[lo])) - target_episode_seconds), idx)
            for idx in range(lo, hi + 1)
            if idx in by_index
        ]
        if not possible:
            break
        cuts.append(min(possible)[1])
        cuts = sorted(set(cuts))
    return sorted(set(cuts))


def _merge_short_cuts(
    transcript: dict[str, Any],
    cuts: list[int],
    *,
    min_episode_seconds: float,
) -> list[int]:
    segments = [item for item in transcript.get("segments", []) if isinstance(item, dict)]
    if not cuts or not segments:
        return cuts
    starts = [_as_float(item.get("start"), 0.0) or 0.0 for item in segments]
    ends = [_as_float(item.get("end"), starts[i]) or starts[i] for i, item in enumerate(segments)]
    changed = True
    current = sorted(set(cuts))
    while changed and current:
        changed = False
        boundaries = [0] + current + [len(segments)]
        for pos in range(len(boundaries) - 1):
            first = boundaries[pos]
            last_exclusive = boundaries[pos + 1]
            last = last_exclusive - 1
            start_time = 0.0 if first == 0 else starts[first]
            end_time = ends[last]
            if end_time - start_time >= min_episode_seconds - 1e-6:
                continue
            # Remove the adjacent boundary that yields the larger merged
            # episode.  Ties resolve toward the earlier boundary.
            options: list[tuple[float, int]] = []
            if pos > 0:
                left_start = 0.0 if boundaries[pos - 1] == 0 else starts[boundaries[pos - 1]]
                options.append((ends[last] - left_start, current[pos - 1]))
            if pos < len(current):
                right_last = boundaries[pos + 2] - 1
                options.append((ends[right_last] - start_time, current[pos]))
            if options:
                current.remove(min(options, key=lambda item: (-item[0], item[1]))[1])
                changed = True
                break
    return current


def partition_episodes(
    transcript: dict[str, Any],
    *,
    min_episode_seconds: float = _DEFAULT_MIN_EPISODE_SECONDS,
    max_episode_seconds: float = _DEFAULT_MAX_EPISODE_SECONDS,
    target_episode_seconds: float = _DEFAULT_TARGET_EPISODE_SECONDS,
    lexical_window: int = 4,
) -> dict[str, Any]:
    """Create a deterministic candidate episode partition for one transcript."""

    validation = validate_transcript(transcript)
    if not validation.valid:
        raise ValueError("invalid transcript: " + "; ".join(validation.errors))
    if min_episode_seconds <= 0 or max_episode_seconds < min_episode_seconds:
        raise ValueError("episode duration bounds are invalid")
    target_episode_seconds = max(min_episode_seconds, min(max_episode_seconds, target_episode_seconds))
    segments = [item for item in transcript.get("segments", []) if isinstance(item, dict)]
    if not segments:
        raise ValueError("transcript has no segments")
    duration = _media_duration(transcript)
    candidates = _boundary_candidates(transcript, lexical_window=lexical_window)
    cuts = _choose_cut_indices(
        transcript,
        candidates,
        min_episode_seconds=min_episode_seconds,
        max_episode_seconds=max_episode_seconds,
        target_episode_seconds=target_episode_seconds,
    )
    cuts = _merge_short_cuts(transcript, cuts, min_episode_seconds=min_episode_seconds)

    starts = [_as_float(item.get("start"), 0.0) or 0.0 for item in segments]
    ends = [_as_float(item.get("end"), starts[i]) or starts[i] for i, item in enumerate(segments)]
    candidate_by_before = {int(item["before_segment_index"]): item for item in candidates}
    episode_ranges: list[tuple[int, int]] = []
    first = 0
    for cut in cuts:
        episode_ranges.append((first, cut - 1))
        first = cut
    episode_ranges.append((first, len(segments) - 1))
    episodes: list[dict[str, Any]] = []
    for number, (first_index, last_index) in enumerate(episode_ranges, 1):
        if first_index > last_index:
            continue
        start = 0.0 if first_index == 0 else float(candidate_by_before.get(first_index, {}).get("time", starts[first_index]))
        end = duration if last_index == len(segments) - 1 else float(candidate_by_before.get(last_index + 1, {}).get("time", ends[last_index]))
        boundary_rows = []
        if first_index > 0 and first_index in candidate_by_before:
            boundary_rows.append(candidate_by_before[first_index])
        if last_index < len(segments) - 1 and last_index + 1 in candidate_by_before:
            boundary_rows.append(candidate_by_before[last_index + 1])
        rationale = {
            "detector": PARTITION_VERSION,
            "signals": sorted(
                {
                    signal
                    for row in boundary_rows
                    for signal, present in (
                        ("lexical_shift", float(row.get("lexical_shift", 0)) > 0.25),
                        ("media_event_boundary", bool(row.get("event_ids"))),
                        ("temporal_gap", float(row.get("gap_seconds", 0)) >= _BOUNDARY_GAP_SECONDS),
                    )
                    if present
                }
            ),
            "boundary_candidates": boundary_rows,
        }
        episodes.append(
            {
                "episode_id": f"{transcript['video_id']}.ep{number:03d}",
                "video_id": str(transcript["video_id"]),
                "course_id": str(transcript["course_id"]),
                "title": str(transcript["title"]),
                "start": round(max(0.0, start), 6),
                "end": round(min(duration, max(start, end)), 6),
                "segment_indices": list(range(first_index, last_index + 1)),
                "boundary_rationale": rationale,
                "claim_boundary": {
                    "candidate_partition_only": True,
                    "episode_gold_established": False,
                    "expert_skill_quality_established": False,
                    "recognition_accuracy_established": False,
                    "teaching_effectiveness_established": False,
                },
            }
        )
    if not episodes:
        raise ValueError("episode partition produced no episodes")
    # Explicit coverage invariant: segment ranges are contiguous and each
    # episode's temporal interval is non-empty.
    flattened = [index for episode in episodes for index in episode["segment_indices"]]
    if flattened != list(range(len(segments))):
        raise AssertionError("episode partition does not cover transcript segments exactly once")
    if episodes[0]["start"] > 1e-6 or abs(episodes[-1]["end"] - duration) > 1e-5:
        raise AssertionError("episode partition does not cover the parent timeline")
    return {
        "schema": EPISODE_PARTITION_SCHEMA,
        "partition_version": PARTITION_VERSION,
        "video_id": str(transcript["video_id"]),
        "course_id": str(transcript["course_id"]),
        "duration_seconds": round(duration, 6),
        "parent_transcript_sha256": transcript_sha256(transcript),
        "parameters": {
            "min_episode_seconds": float(min_episode_seconds),
            "max_episode_seconds": float(max_episode_seconds),
            "target_episode_seconds": float(target_episode_seconds),
            "lexical_window": int(lexical_window),
        },
        "episodes": episodes,
        "claim_boundary": {
            "candidate_partition_only": True,
            "episode_gold_established": False,
            "expert_skill_quality_established": False,
            "recognition_accuracy_established": False,
            "teaching_effectiveness_established": False,
        },
    }


def slice_episode(transcript: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    """Return a provenance-preserving transcript slice for one episode."""

    expected_hash = str(episode.get("parent_transcript_sha256", ""))
    actual_hash = transcript_sha256(transcript)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("parent transcript SHA-256 does not match episode partition")
    indices = episode.get("segment_indices")
    if not isinstance(indices, list) or not indices or any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
        raise ValueError("episode.segment_indices must be a non-empty integer list")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("episode.segment_indices must be contiguous")
    segments = transcript.get("segments", [])
    if indices[0] < 0 or indices[-1] >= len(segments):
        raise ValueError("episode segment indices are outside parent transcript")
    result = copy.deepcopy(transcript)
    result["episode_id"] = str(episode.get("episode_id", ""))
    result["episode_start"] = float(episode["start"])
    result["episode_end"] = float(episode["end"])
    result["parent_transcript_sha256"] = actual_hash
    result["segments"] = []
    for parent_index in indices:
        segment = copy.deepcopy(segments[parent_index])
        segment["_parent_segment_index"] = parent_index
        result["segments"].append(segment)
    start = float(episode["start"])
    end = float(episode["end"])
    multimodal = result.get("multimodal")
    if isinstance(multimodal, dict):
        media_duration = _media_duration(transcript)

        final = end >= media_duration - 1e-9

        def owns_interval(item: Mapping[str, Any]) -> bool:
            """Assign an interval to one episode by its start anchor.

            Intervals are not clipped or rebased: an item crossing a cut stays
            whole in the episode that owns its start.  Event-linked evidence
            (handled below) is retained even when its referenced frame/silence
            starts outside that interval.
            """

            item_start = _finite_timestamp(item.get("start"))
            if item_start is None:
                return False
            if item_start < start - 1e-9:
                return False
            if item_start < end - 1e-9:
                return True
            return final and math.isclose(item_start, end, abs_tol=1e-9)

        events = multimodal.get("events", [])
        selected_events: list[dict[str, Any]] = []
        if isinstance(events, list):
            selected_events = [
                copy.deepcopy(event)
                for event in events
                if isinstance(event, dict)
                and _event_owned_by_episode(event, start=start, end=end, final=final)
            ]
            selected_events.sort(
                key=lambda event: (
                    _as_float(event.get("start"), 0.0) or 0.0,
                    _as_float(event.get("end"), 0.0) or 0.0,
                    str(event.get("event_id", "")),
                )
            )
            multimodal["events"] = selected_events

        event_identities = {_event_identity(event) for event in selected_events}
        event_paths: set[str] = set()
        event_silences: list[dict[str, Any]] = []
        for event in selected_events:
            evidence = event.get("evidence")
            if not isinstance(evidence, dict):
                continue
            for key in ("frame_path", "before_frame", "after_frame"):
                value = evidence.get(key)
                if value:
                    event_paths.add(str(value))
            silence = evidence.get("silence")
            if isinstance(silence, dict):
                event_silences.append(copy.deepcopy(silence))

        visual = multimodal.get("visual")
        if isinstance(visual, dict):
            keyframes = visual.get("keyframes", [])
            if isinstance(keyframes, list):
                selected_frames = []
                for frame in keyframes:
                    if not isinstance(frame, dict):
                        continue
                    path = str(frame.get("path", ""))
                    timestamp = _as_float(frame.get("timestamp"), None)
                    if path in event_paths or (
                        timestamp is not None
                        and start <= timestamp <= end
                        and (timestamp < end or math.isclose(end, media_duration))
                    ):
                        selected_frames.append(copy.deepcopy(frame))
                selected_frames.sort(
                    key=lambda frame: (
                        _as_float(frame.get("timestamp"), 0.0) or 0.0,
                        str(frame.get("path", "")),
                    )
                )
                visual["keyframes"] = selected_frames
            visual_events = visual.get("events")
            if isinstance(visual_events, list):
                visual["events"] = [
                    copy.deepcopy(event)
                    for event in visual_events
                    if isinstance(event, dict)
                    and _event_identity(event) in event_identities
                ]

        audio = multimodal.get("audio")
        if isinstance(audio, dict):
            silences = audio.get("silences", [])
            selected_silences: list[dict[str, Any]] = []
            if isinstance(silences, list):
                for silence in silences:
                    if isinstance(silence, dict) and owns_interval(silence):
                        selected_silences.append(copy.deepcopy(silence))
            for silence in event_silences:
                if silence not in selected_silences:
                    selected_silences.append(silence)
            selected_silences.sort(
                key=lambda silence: (
                    _as_float(silence.get("start"), 0.0) or 0.0,
                    _as_float(silence.get("end"), 0.0) or 0.0,
                )
            )
            audio["silences"] = selected_silences
        validation = validate_transcript(result)
        if not validation.valid:
            raise ValueError("episode transcript is invalid: " + "; ".join(validation.errors))
    return result


def _event_ids_in_episode(transcript: dict[str, Any], episode: dict[str, Any]) -> list[str]:
    sliced = slice_episode(transcript, episode)
    return sorted(
        str(event["event_id"])
        for event in sliced.get("multimodal", {}).get("events", [])
        if isinstance(event, dict) and event.get("event_id")
    )


def _finite_timestamp(value: Any) -> float | None:
    """Parse a finite timestamp without allowing malformed evidence to leak."""

    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _event_anchor(event: Mapping[str, Any]) -> float | None:
    """Return the stable ownership anchor for a multimodal event.

    ``question_and_wait`` spans a question and a following silence.  Assigning
    it by midpoint can put the event in a neighbouring episode and lose the
    question/silence pair.  Its immutable question segment start is therefore
    the owner anchor.  Other events use their declared start, which gives
    half-open, non-overlapping ownership at episode boundaries.
    """

    evidence = event.get("evidence")
    if str(event.get("type", "")) == "question_and_wait" and isinstance(evidence, Mapping):
        question_segment = evidence.get("question_segment")
        if isinstance(question_segment, Mapping):
            anchored = _finite_timestamp(question_segment.get("start"))
            if anchored is not None:
                return anchored
    return _finite_timestamp(event.get("start"))


def _event_identity(event: Mapping[str, Any]) -> str:
    """Build a stable identity for raw events that do not carry event_id."""

    event_id = str(event.get("event_id", "")).strip()
    if event_id:
        return "id:" + event_id
    material = {
        key: value
        for key, value in event.items()
        if key not in {"event_id", "event_fingerprint_sha256"}
    }
    return "hash:" + sha256_json(material)


def _event_owned_by_episode(
    event: Mapping[str, Any],
    *,
    start: float,
    end: float,
    final: bool,
) -> bool:
    anchor = _event_anchor(event)
    if anchor is None:
        return False
    if anchor < start - 1e-9:
        return False
    if anchor < end - 1e-9:
        return True
    # The final episode owns an event exactly at the media endpoint.  This is
    # deliberately not used for intermediate slices, so an event on a cut is
    # owned by exactly one (the following) episode.
    return final and math.isclose(anchor, end, abs_tol=1e-9)


def mine_episode_skill(
    transcript: dict[str, Any],
    episode: dict[str, Any],
    *,
    evaluate: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Mine and optionally internally evaluate one episode Skill artifact."""

    parent_hash = transcript_sha256(transcript)
    if str(episode.get("parent_transcript_sha256", parent_hash)) != parent_hash:
        raise ValueError("episode partition is bound to a different transcript")
    sliced = slice_episode(transcript, episode)
    skill = mine_skill(sliced)
    skill["mining_metadata"]["episode_partition"] = PARTITION_VERSION
    skill["mining_metadata"]["episode_id"] = str(episode["episode_id"])
    skill["mining_metadata"]["parent_transcript_sha256"] = parent_hash
    # Keep the underlying single-Skill contract intact while making the
    # episode scope explicit in source provenance.
    skill.setdefault("source", {})["episode_id"] = str(episode["episode_id"])
    skill["source"]["episode_bounds"] = {
        "start": float(episode["start"]),
        "end": float(episode["end"]),
    }
    skill["source"]["parent_transcript_sha256"] = parent_hash
    validation = validate_skill(skill)
    if not validation.valid:
        raise ValueError("generated invalid episode Skill: " + "; ".join(validation.errors))
    evidence = skill.get("source", {}).get("evidence", [])
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if not (float(episode["start"]) - 1e-6 <= float(item["start"]) and float(item["end"]) <= float(episode["end"]) + 1e-6):
            raise ValueError("episode Skill evidence falls outside episode bounds")
    event_ids = _event_ids_in_episode(transcript, episode)
    referenced_event_ids = sorted(
        str(item.get("event_id"))
        for item in skill.get("source", {}).get("multimodal_evidence", [])
        if isinstance(item, dict) and item.get("event_id")
    )
    if not set(referenced_event_ids) <= set(event_ids):
        raise ValueError("episode Skill references an event outside its parent episode")
    artifact: dict[str, Any] = {
        "schema": EPISODE_SKILL_SCHEMA,
        "artifact_kind": "episode_teaching_skill",
        "video_id": str(transcript["video_id"]),
        "course_id": str(transcript["course_id"]),
        "episode_id": str(episode["episode_id"]),
        "start": float(episode["start"]),
        "end": float(episode["end"]),
        "segment_indices": list(episode["segment_indices"]),
        "event_ids": event_ids,
        "skill": skill,
        "skill_sha256": sha256_json(skill),
        "parent_transcript_sha256": parent_hash,
        "partition_provenance": {
            "partition_version": PARTITION_VERSION,
            "boundary_rationale": copy.deepcopy(episode.get("boundary_rationale", {})),
            "candidate_partition_only": True,
        },
        "claim_boundary": copy.deepcopy(episode.get("claim_boundary", {})),
    }
    report = evaluate_skill(skill, sliced) if evaluate else None
    return artifact, report


def validate_episode_artifact(
    artifact: dict[str, Any],
    *,
    transcript: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed structural/provenance validation for an episode artifact."""

    errors: list[str] = []
    if not isinstance(artifact, dict):
        return {"valid": False, "errors": ["artifact must be an object"], "warnings": []}
    if artifact.get("schema") != EPISODE_SKILL_SCHEMA:
        errors.append("unsupported episode Skill schema")
    skill = artifact.get("skill")
    if not isinstance(skill, dict):
        errors.append("skill must be an object")
    else:
        result = validate_skill(skill)
        errors.extend(result.errors)
        if artifact.get("skill_sha256") != sha256_json(skill):
            errors.append("skill_sha256 does not match skill payload")
    boundary = artifact.get("claim_boundary")
    required_false = (
        "episode_gold_established",
        "expert_skill_quality_established",
        "recognition_accuracy_established",
        "teaching_effectiveness_established",
    )
    if not isinstance(boundary, dict):
        errors.append("claim_boundary is required")
    else:
        for key in required_false:
            if boundary.get(key) is not False:
                errors.append(f"claim_boundary.{key} must remain false")
    if transcript is not None:
        actual = transcript_sha256(transcript)
        if artifact.get("parent_transcript_sha256") != actual:
            errors.append("parent_transcript_sha256 does not match transcript")
        episode = {
            "episode_id": artifact.get("episode_id"),
            "start": artifact.get("start"),
            "end": artifact.get("end"),
            "segment_indices": artifact.get("segment_indices"),
            "parent_transcript_sha256": actual,
        }
        try:
            sliced = slice_episode(transcript, episode)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if isinstance(skill, dict):
                evidence_ids = {
                    str(item.get("evidence_id"))
                    for item in skill.get("source", {}).get("evidence", [])
                    if isinstance(item, dict)
                }
                parent_ids = set()
                for index in artifact.get("segment_indices", []):
                    segment = transcript.get("segments", [])[index]
                    parent_ids.add(
                        "evi_" + hashlib.sha256(
                            "\x1f".join((
                                str(transcript.get("video_id", "")),
                                str(index),
                                str(segment.get("start", "")),
                                str(segment.get("end", "")),
                                str(segment.get("text", "")),
                            )).encode("utf-8")
                        ).hexdigest()[:16]
                    )
                if not evidence_ids <= parent_ids:
                    errors.append("episode evidence id is not present in parent transcript")
                event_ids = {
                    str(event.get("event_id"))
                    for event in sliced.get("multimodal", {}).get("events", [])
                    if isinstance(event, dict)
                }
                refs = {
                    str(item.get("event_id"))
                    for item in skill.get("source", {}).get("multimodal_evidence", [])
                    if isinstance(item, dict)
                }
                if not refs <= event_ids:
                    errors.append("episode multimodal event reference is outside parent episode")
    return {"valid": not errors, "errors": errors, "warnings": []}


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    options = [candidate, manifest_path.parent / candidate, Path.cwd() / candidate]
    for option in options:
        if option.is_file():
            return option.resolve()
    raise FileNotFoundError(value)


def mine_episode_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    min_episode_seconds: float = _DEFAULT_MIN_EPISODE_SECONDS,
    max_episode_seconds: float = _DEFAULT_MAX_EPISODE_SECONDS,
    target_episode_seconds: float = _DEFAULT_TARGET_EPISODE_SECONDS,
    lexical_window: int = 4,
) -> dict[str, Any]:
    """Mine all videos in a long-form dataset manifest into episode artifacts."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = read_json(manifest_file)
    output = ensure_private_directory(output_dir).resolve()
    episodes_root = ensure_private_directory(output / "episodes")
    evaluations_root = ensure_private_directory(output / "evaluations")
    rows: list[dict[str, Any]] = []
    total_episodes = 0
    passed_evaluations = 0
    for video in manifest.get("videos", []):
        if not isinstance(video, dict):
            continue
        transcript_path = _resolve_manifest_path(manifest_file, str(video.get("transcript_path", "")))
        transcript = read_json(transcript_path)
        partition = partition_episodes(
            transcript,
            min_episode_seconds=min_episode_seconds,
            max_episode_seconds=max_episode_seconds,
            target_episode_seconds=target_episode_seconds,
            lexical_window=lexical_window,
        )
        video_dir = ensure_private_directory(episodes_root / str(transcript["video_id"]))
        video_eval_dir = ensure_private_directory(evaluations_root / str(transcript["video_id"]))
        episode_rows: list[dict[str, Any]] = []
        for episode in partition["episodes"]:
            artifact, evaluation = mine_episode_skill(transcript, episode)
            skill_path = write_json(video_dir / f"{episode['episode_id']}.skill.json", artifact)
            evaluation_path = write_json(video_eval_dir / f"{episode['episode_id']}.evaluation.json", evaluation or {})
            validation = validate_episode_artifact(artifact, transcript=transcript)
            if not validation["valid"]:
                raise ValueError("invalid generated episode artifact: " + "; ".join(validation["errors"]))
            passed_evaluations += bool(evaluation and evaluation.get("passed"))
            total_episodes += 1
            episode_rows.append(
                {
                    **copy.deepcopy(episode),
                    "skill_path": str(skill_path.relative_to(output)),
                    "evaluation_path": str(evaluation_path.relative_to(output)),
                    "skill_sha256": artifact["skill_sha256"],
                    "artifact_sha256": sha256_json(artifact),
                    "evaluation_passed": bool(evaluation and evaluation.get("passed")),
                }
            )
        rows.append(
            {
                "video_id": str(transcript["video_id"]),
                "course_id": str(transcript["course_id"]),
                "transcript_path": str(transcript_path),
                "parent_transcript_sha256": partition["parent_transcript_sha256"],
                "partition": partition,
                "episodes": episode_rows,
            }
        )
    episode_manifest = {
        "schema": EPISODE_PARTITION_SCHEMA,
        "artifact_kind": "episode_skill_dataset_manifest",
        "dataset_manifest_sha256": sha256_json(manifest),
        "video_count": len(rows),
        "episode_count": total_episodes,
        "videos": rows,
        "claim_boundary": {
            "candidate_partition_only": True,
            "episode_gold_established": False,
            "expert_skill_quality_established": False,
            "recognition_accuracy_established": False,
            "teaching_effectiveness_established": False,
        },
    }
    manifest_output = write_json(output / "episode_manifest.json", episode_manifest)
    receipt = {
        "schema": EPISODE_RECEIPT_SCHEMA,
        "artifact_kind": "episode_skill_mining_receipt",
        "episode_manifest_sha256": sha256_json(episode_manifest),
        "manifest_path": str(manifest_output),
        "video_count": len(rows),
        "episode_count": total_episodes,
        "evaluation_passed_count": passed_evaluations,
        "internal_evaluation_is_accuracy": False,
        "claim_boundary": {
            "candidate_partition_only": True,
            "episode_gold_established": False,
            "expert_skill_quality_established": False,
            "recognition_accuracy_established": False,
            "teaching_effectiveness_established": False,
        },
    }
    receipt_output = write_json(output / "episode_receipt.json", receipt)
    return {"manifest": episode_manifest, "receipt": receipt, "manifest_path": manifest_output, "receipt_path": receipt_output}
