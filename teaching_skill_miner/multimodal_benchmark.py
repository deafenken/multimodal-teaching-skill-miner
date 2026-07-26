from __future__ import annotations

import re
from typing import Any


def _round_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _event_matches(detected: dict[str, Any], expected: dict[str, Any]) -> bool:
    if detected.get("type") != expected.get("type"):
        return False
    tolerance = float(expected.get("tolerance_seconds", 0.25))
    if "anchor" in expected:
        anchor = float(expected["anchor"])
        return float(detected["start"]) - tolerance <= anchor <= float(detected["end"]) + tolerance
    return (
        abs(float(detected["start"]) - float(expected["start"])) <= tolerance
        and abs(float(detected["end"]) - float(expected["end"])) <= tolerance
    )


def _match_one_to_one(
    detected: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    unused = set(range(len(detected)))
    details: list[dict[str, Any]] = []
    matches = 0
    for expected_item in expected:
        matched_index = next(
            (index for index in sorted(unused) if _event_matches(detected[index], expected_item)),
            None,
        )
        matched = matched_index is not None
        if matched_index is not None:
            unused.remove(matched_index)
            matches += 1
        details.append(
            {
                "expected": expected_item,
                "matched": matched,
                "detected": detected[matched_index] if matched_index is not None else None,
            }
        )
    return matches, details


def _normal_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def evaluate_multimodal_fixture(
    analysis: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a deterministic fixture against labels authored independently of detector output.

    The result deliberately does not claim real-world accuracy. It verifies that a known
    silence, slide transition, OCR text, and aligned transcript event survive the pipeline.
    """

    expected_events = ground_truth.get("events", [])
    evaluated_types = {item.get("type") for item in expected_events if isinstance(item, dict)}
    detected_events = [
        item
        for item in analysis.get("events", [])
        if isinstance(item, dict)
        and item.get("type") in evaluated_types
        and "classroom_observation" not in item.get("modalities", [])
    ]
    true_positives, event_details = _match_one_to_one(detected_events, expected_events)
    event_precision = _round_ratio(true_positives, len(detected_events))
    event_recall = _round_ratio(true_positives, len(expected_events))
    event_f1 = (
        round(2 * event_precision * event_recall / (event_precision + event_recall), 4)
        if event_precision + event_recall
        else 0.0
    )

    expected_silences = ground_truth.get("silences", [])
    detected_silences = analysis.get("audio", {}).get("silences", [])
    silence_matches, silence_details = _match_one_to_one(
        [dict(item, type="silence") for item in detected_silences],
        [dict(item, type="silence") for item in expected_silences],
    )
    silence_recall = _round_ratio(silence_matches, len(expected_silences))

    frames = analysis.get("visual", {}).get("keyframes", [])
    ocr_details: list[dict[str, Any]] = []
    required_token_count = 0
    matched_token_count = 0
    for check in ground_truth.get("ocr_checks", []):
        timestamp = float(check["timestamp"])
        tolerance = float(check.get("tolerance_seconds", 0.6))
        nearby = [frame for frame in frames if abs(float(frame.get("timestamp", -999)) - timestamp) <= tolerance]
        frame = min(nearby, key=lambda item: abs(float(item["timestamp"]) - timestamp)) if nearby else None
        required_tokens = {str(item).lower() for item in check.get("required_tokens", [])}
        observed_tokens = _normal_tokens(str(frame.get("ocr_text", ""))) if frame else set()
        matched_tokens = sorted(required_tokens & observed_tokens)
        required_token_count += len(required_tokens)
        matched_token_count += len(matched_tokens)
        ocr_details.append(
            {
                "timestamp": timestamp,
                "required_tokens": sorted(required_tokens),
                "matched_tokens": matched_tokens,
                "frame": frame,
            }
        )
    ocr_token_recall = _round_ratio(matched_token_count, required_token_count)

    gates = {
        "all_labeled_events_matched": true_positives == len(expected_events),
        "no_extra_evaluated_events": true_positives == len(detected_events),
        "all_labeled_silences_matched": silence_matches == len(expected_silences),
        "ocr_required_token_recall_at_least_80_percent": ocr_token_recall >= 0.8,
    }
    return {
        "benchmark_kind": "deterministic_synthetic_fixture",
        "fixture_id": ground_truth.get("fixture_id"),
        "passed": all(gates.values()),
        "validity_scope": "Known synthetic media events only",
        "real_world_accuracy_established": False,
        "teaching_effectiveness_established": False,
        "automatic_speech_recognition_evaluated": False,
        "classroom_behavior_recognition_evaluated": False,
        "event_metrics": {
            "evaluated_types": sorted(str(item) for item in evaluated_types),
            "true_positives": true_positives,
            "detected_count": len(detected_events),
            "ground_truth_count": len(expected_events),
            "precision": event_precision,
            "recall": event_recall,
            "f1": event_f1,
            "details": event_details,
        },
        "silence_metrics": {
            "matched_count": silence_matches,
            "ground_truth_count": len(expected_silences),
            "recall": silence_recall,
            "details": silence_details,
        },
        "ocr_metrics": {
            "matched_token_count": matched_token_count,
            "required_token_count": required_token_count,
            "token_recall": ocr_token_recall,
            "details": ocr_details,
        },
        "gates": gates,
        "limitations": ground_truth.get("limitations", []),
    }
