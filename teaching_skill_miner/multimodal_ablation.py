from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from statistics import mean, median
from typing import Any, Iterable

from .evaluator import evaluate_skill
from .miner import mine_skill
from .models import validate_transcript


TRANSCRIPT_ONLY = "transcript_only"
TRANSCRIPT_AUDIO = "transcript_audio"
TRANSCRIPT_VISUAL = "transcript_visual"
FULL = "full"
ARM_ORDER = (TRANSCRIPT_ONLY, TRANSCRIPT_AUDIO, TRANSCRIPT_VISUAL, FULL)

ARM_LABELS = {
    TRANSCRIPT_ONLY: "transcript-only",
    TRANSCRIPT_AUDIO: "transcript + audio",
    TRANSCRIPT_VISUAL: "transcript + visual/OCR",
    FULL: "transcript + audio + visual/OCR",
}

_LANGUAGE_MODALITIES = {"transcript", "speech"}
_AUDIO_MODALITIES = {"audio"}
_VISUAL_MODALITIES = {"visual", "ocr"}
_EXCLUDED_MODALITIES = {"classroom_observation"}


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _event_is_in_arm(event: dict[str, Any], arm: str) -> bool:
    modalities = set(event.get("modalities", []))
    if not modalities or modalities & _EXCLUDED_MODALITIES:
        return False
    non_language = modalities - _LANGUAGE_MODALITIES
    if arm == TRANSCRIPT_AUDIO:
        return bool(non_language & _AUDIO_MODALITIES) and non_language <= _AUDIO_MODALITIES
    if arm == TRANSCRIPT_VISUAL:
        return bool(non_language & _VISUAL_MODALITIES) and non_language <= _VISUAL_MODALITIES
    if arm == FULL:
        allowed = _AUDIO_MODALITIES | _VISUAL_MODALITIES
        return bool(non_language & allowed) and non_language <= allowed
    return False


def build_ablation_transcript(
    transcript: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    """Return a validation-safe view containing only the modalities in one arm.

    The transcript segments and their provenance are identical in every arm.  Classroom
    observations are deliberately excluded because the four requested arms concern
    machine-readable audio and video evidence, not human observation notes.
    """

    if arm not in ARM_ORDER:
        raise ValueError(f"unknown ablation arm: {arm}")
    result = copy.deepcopy(transcript)
    if arm == TRANSCRIPT_ONLY:
        result.pop("multimodal", None)
        validation = validate_transcript(result)
        if not validation.valid:
            raise ValueError("invalid transcript-only view: " + "; ".join(validation.errors))
        return result

    source = transcript.get("multimodal")
    if not isinstance(source, dict):
        # The arm remains a valid no-added-signal comparator. Its zero event/record
        # counts make the missing modality explicit in the paired report.
        return result

    events = [
        copy.deepcopy(event)
        for event in source.get("events", [])
        if isinstance(event, dict) and _event_is_in_arm(event, arm)
    ]
    retained_modalities = {"transcript"}
    for event in events:
        retained_modalities.update(str(value) for value in event.get("modalities", []))

    source_modalities = set(source.get("modalities_available", []))
    if arm in {TRANSCRIPT_AUDIO, FULL} and "audio" in source_modalities:
        retained_modalities.add("audio")
    if arm in {TRANSCRIPT_VISUAL, FULL}:
        retained_modalities.update(source_modalities & _VISUAL_MODALITIES)

    filtered: dict[str, Any] = {
        "modalities_available": sorted(retained_modalities),
        "events": events,
    }
    if isinstance(source.get("media"), dict):
        filtered["media"] = copy.deepcopy(source["media"])
    if arm in {TRANSCRIPT_AUDIO, FULL} and isinstance(source.get("audio"), dict):
        filtered["audio"] = copy.deepcopy(source["audio"])
    if arm in {TRANSCRIPT_VISUAL, FULL} and isinstance(source.get("visual"), dict):
        filtered["visual"] = copy.deepcopy(source["visual"])
    result["multimodal"] = filtered

    validation = validate_transcript(result)
    if not validation.valid:
        raise ValueError(
            f"invalid {arm} view: " + "; ".join(validation.errors)
        )
    return result


def _covered_duration(events: list[dict[str, Any]]) -> float:
    intervals: list[tuple[float, float]] = []
    for event in events:
        try:
            start = max(0.0, float(event["start"]))
            end = max(start, float(event["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _duration_seconds(transcript: dict[str, Any]) -> float:
    raw_duration = transcript.get("multimodal", {}).get("media", {}).get(
        "duration_seconds"
    )
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0:
        return duration
    ends: list[float] = []
    for segment in transcript.get("segments", []):
        try:
            ends.append(float(segment["end"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(ends, default=0.0)


def _analysis_metrics(transcript: dict[str, Any]) -> dict[str, Any]:
    multimodal = transcript.get("multimodal", {})
    events = [
        item for item in multimodal.get("events", []) if isinstance(item, dict)
    ]
    duration = _duration_seconds(transcript)
    covered = _covered_duration(events)
    confidences = [
        float(item["confidence"])
        for item in events
        if isinstance(item.get("confidence"), (int, float))
        and not isinstance(item.get("confidence"), bool)
    ]
    keyframes = [
        item
        for item in multimodal.get("visual", {}).get("keyframes", [])
        if isinstance(item, dict)
    ]
    nonempty_ocr = sum(bool(str(item.get("ocr_text", "")).strip()) for item in keyframes)
    semantic_rows = [
        item.get("visual_semantics")
        for item in keyframes
        if isinstance(item.get("visual_semantics"), dict)
    ]
    semantic_labels = Counter(
        str(item.get("top_label", "unknown")) for item in semantic_rows
    )
    semantic_scores = [
        float(item["top_relative_score"])
        for item in semantic_rows
        if isinstance(item.get("top_relative_score"), (int, float))
        and not isinstance(item.get("top_relative_score"), bool)
    ]
    semantic_margins = [
        float(item["score_margin"])
        for item in semantic_rows
        if isinstance(item.get("score_margin"), (int, float))
        and not isinstance(item.get("score_margin"), bool)
    ]
    semantic_status = (
        multimodal.get("visual", {}).get("semantic_features", {}).get("status")
    )
    silences = [
        item
        for item in multimodal.get("audio", {}).get("silences", [])
        if isinstance(item, dict)
    ]
    event_types = Counter(str(item.get("type", "unknown")) for item in events)
    return {
        "duration_seconds": round(duration, 3),
        "event_count": len(events),
        "event_type_count": len(event_types),
        "event_type_frequency": dict(sorted(event_types.items())),
        "event_union_duration_seconds": round(covered, 3),
        "event_temporal_coverage_fraction": (
            round(min(1.0, covered / duration), 6) if duration else None
        ),
        "detector_reported_confidence_mean": (
            round(mean(confidences), 6) if confidences else None
        ),
        "detector_confidence_is_calibrated_probability": False,
        "silence_record_count": len(silences),
        "keyframe_count": len(keyframes),
        "nonempty_ocr_keyframe_count": nonempty_ocr,
        "nonempty_ocr_keyframe_fraction": (
            round(nonempty_ocr / len(keyframes), 6) if keyframes else None
        ),
        "visual_semantic_feature_status": semantic_status,
        "visual_semantic_keyframe_count": len(semantic_rows),
        "visual_semantic_keyframe_fraction": (
            round(len(semantic_rows) / len(keyframes), 6) if keyframes else None
        ),
        "visual_semantic_top_label_frequency": dict(sorted(semantic_labels.items())),
        "visual_semantic_top_relative_score_mean": (
            round(mean(semantic_scores), 6) if semantic_scores else None
        ),
        "visual_semantic_score_margin_mean": (
            round(mean(semantic_margins), 6) if semantic_margins else None
        ),
        "visual_semantic_scores_are_calibrated_probabilities": False,
    }


def _skill_metrics(skill: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    observed = [
        str(item.get("id"))
        for item in skill.get("strategies", [])
        if isinstance(item, dict) and item.get("origin") == "observed_method"
    ]
    multimodal = evaluation.get("multimodal_evaluation", {})
    method = evaluation.get("method_provenance", {})
    return {
        "skill_fingerprint_sha256": _fingerprint(skill),
        "primary_strategy_id": (
            skill.get("strategies", [{}])[0].get("id")
            if skill.get("strategies")
            else None
        ),
        "observed_strategy_ids": observed,
        "observed_strategy_count": len(observed),
        "multimodal_evidence_reference_count": multimodal.get(
            "source_reference_count", 0
        ),
        "matched_multimodal_evidence_reference_count": multimodal.get(
            "matched_reference_count", 0
        ),
        "observed_steps_with_valid_evidence_count": method.get(
            "observed_steps_with_valid_evidence_count", 0
        ),
        "internal_overall_score": evaluation.get("overall_score"),
        "internal_dimensions": evaluation.get("dimensions", {}),
        "internal_gates": evaluation.get("gates", {}),
        "internal_evidence_consistency_score": multimodal.get("score"),
        "skill_schema_valid": evaluation.get("validation", {}).get("valid", False),
        "pipeline_internal_evaluation_passed": evaluation.get("passed", False),
    }


def evaluate_lecture_ablation(transcript: dict[str, Any]) -> dict[str, Any]:
    """Compute four paired outputs for exactly one lecture."""

    source_validation = validate_transcript(transcript)
    if not source_validation.valid:
        raise ValueError("invalid source transcript: " + "; ".join(source_validation.errors))

    arm_payloads: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        arm_transcript = build_ablation_transcript(transcript, arm)
        skill = mine_skill(arm_transcript)
        evaluation = evaluate_skill(skill, arm_transcript)
        arm_payloads[arm] = (skill, evaluation)
        arms[arm] = {
            "label": ARM_LABELS[arm],
            "transcript_view_fingerprint_sha256": _fingerprint(arm_transcript),
            "analysis_metrics": _analysis_metrics(arm_transcript),
            "skill_metrics": _skill_metrics(skill, evaluation),
        }

    baseline = arms[TRANSCRIPT_ONLY]["skill_metrics"]
    baseline_strategies = set(baseline["observed_strategy_ids"])
    for arm in ARM_ORDER:
        metrics = arms[arm]["skill_metrics"]
        strategies = set(metrics["observed_strategy_ids"])
        metrics["paired_delta_from_transcript_only"] = {
            "internal_overall_score": round(
                float(metrics["internal_overall_score"])
                - float(baseline["internal_overall_score"]),
                6,
            ),
            "evidence_grounding": round(
                float(metrics["internal_dimensions"].get("evidence_grounding", 0))
                - float(
                    baseline["internal_dimensions"].get("evidence_grounding", 0)
                ),
                6,
            ),
            "observed_strategy_count": (
                metrics["observed_strategy_count"]
                - baseline["observed_strategy_count"]
            ),
            "new_observed_strategy_ids": sorted(strategies - baseline_strategies),
            "lost_observed_strategy_ids": sorted(baseline_strategies - strategies),
            "primary_strategy_changed": (
                metrics["primary_strategy_id"] != baseline["primary_strategy_id"]
            ),
        }

    return {
        "video_id": transcript.get("video_id"),
        "course_id": transcript.get("course_id"),
        "title": transcript.get("title"),
        "source_transcript_fingerprint_sha256": _fingerprint(transcript),
        "paired_on_identical_transcript_segments": all(
            [
                build_ablation_transcript(transcript, arm).get("segments")
                == transcript.get("segments")
                for arm in ARM_ORDER
            ]
        ),
        "arms": arms,
        "_payloads": arm_payloads,
    }


def _paired_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    scores = [
        float(row["arms"][arm]["skill_metrics"]["internal_overall_score"])
        for row in rows
    ]
    deltas = [
        float(
            row["arms"][arm]["skill_metrics"][
                "paired_delta_from_transcript_only"
            ]["internal_overall_score"]
        )
        for row in rows
    ]
    grounding_deltas = [
        float(
            row["arms"][arm]["skill_metrics"][
                "paired_delta_from_transcript_only"
            ]["evidence_grounding"]
        )
        for row in rows
    ]
    event_counts = [
        int(row["arms"][arm]["analysis_metrics"]["event_count"])
        for row in rows
    ]
    return {
        "lecture_count": len(rows),
        "mean_internal_overall_score": round(mean(scores), 6) if scores else None,
        "mean_paired_internal_score_delta": round(mean(deltas), 6) if deltas else None,
        "median_paired_internal_score_delta": round(median(deltas), 6)
        if deltas
        else None,
        "mean_paired_evidence_grounding_delta": round(mean(grounding_deltas), 6)
        if grounding_deltas
        else None,
        "lectures_with_positive_internal_score_delta": sum(value > 0 for value in deltas),
        "lectures_with_zero_internal_score_delta": sum(value == 0 for value in deltas),
        "lectures_with_negative_internal_score_delta": sum(value < 0 for value in deltas),
        "lectures_with_retained_events": sum(value > 0 for value in event_counts),
        "total_retained_event_count": sum(event_counts),
    }


def evaluate_multimodal_ablation(
    transcripts: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate paired internal pipeline changes without creating an accuracy claim."""

    rows = [evaluate_lecture_ablation(value) for value in transcripts]
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        video_id = str(row.get("video_id", ""))
        if video_id in seen:
            duplicates.append(video_id)
        seen.add(video_id)
    if duplicates:
        raise ValueError("duplicate video_id in ablation input: " + ", ".join(duplicates))

    # Payloads are returned separately so callers can write exact paired Skill and
    # evaluation artifacts while the summary report remains compact and auditable.
    payloads = {
        str(row["video_id"]): row.pop("_payloads")
        for row in rows
    }
    return {
        "evaluation_kind": "paired_internal_multimodal_pipeline_ablation",
        "design": {
            "unit_of_pairing": "lecture",
            "baseline_arm": TRANSCRIPT_ONLY,
            "segment_control": (
                "Every arm receives byte-equivalent transcript segment records from the "
                "same lecture; only machine-derived audio/visual analysis records vary."
            ),
            "classroom_observations_included": False,
            "aggregation": "unweighted macro mean over lecture-level paired differences",
            "model_fitting_performed": False,
        },
        "arm_order": list(ARM_ORDER),
        "arm_labels": ARM_LABELS,
        "paired_lecture_count": len(rows),
        "all_arms_use_identical_transcript_segments": all(
            row["paired_on_identical_transcript_segments"] for row in rows
        ),
        "aggregate_internal_metrics": {
            arm: _paired_summary(rows, arm) for arm in ARM_ORDER
        },
        "per_lecture": rows,
        "claim_boundary": {
            "independent_event_ground_truth_used": False,
            "human_skill_quality_labels_used": False,
            "learner_outcomes_used": False,
            "recognition_accuracy_established": False,
            "recognition_precision_recall_f1_established": False,
            "multimodal_gain_established": False,
            "causal_multimodal_gain_established": False,
            "independent_skill_quality_gain_established": False,
            "teaching_effectiveness_established": False,
            "deployment_accuracy_established": False,
            "recognition_metrics": {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
            },
            "allowed_claim": (
                "Paired changes in deterministic detector outputs, exact evidence-reference "
                "consistency, and the project's internal Skill rubric on these lectures."
            ),
            "prohibited_claim": (
                "The score deltas are not recognition accuracy, independently judged Skill "
                "quality, causal multimodal gain, learner benefit, or deployment performance."
            ),
        },
        "metric_semantics": {
            "internal_overall_score": (
                "Weighted structural/evidence/executability rubric computed by evaluate_skill; "
                "not an accuracy metric."
            ),
            "internal_evidence_consistency_score": (
                "Exact reference matching against the same analysis record; not detector "
                "precision or recall."
            ),
            "detector_reported_confidence_mean": (
                "Mean detector-emitted heuristic confidence; not calibrated probability."
            ),
            "visual_semantic_top_relative_score_mean": (
                "Mean CLIP closed-ontology relative prompt score; not a calibrated "
                "probability or recognition accuracy."
            ),
            "event_temporal_coverage_fraction": (
                "Union of retained event time spans divided by media duration; it measures "
                "output density, not correctness."
            ),
        },
        "_payloads": payloads,
    }
