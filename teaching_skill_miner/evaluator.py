from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from .miner import STRATEGY_PATTERNS
from .models import REQUIRED_SKILL_FIELDS, validate_skill, validate_transcript
from .teaching_phases import CANONICAL_PHASES


DIMENSION_WEIGHTS = {
    "structural_completeness": 0.12,
    "evidence_grounding": 0.18,
    "executability": 0.18,
    "method_fidelity": 0.22,
    "pedagogical_quality": 0.12,
    "generalizability": 0.09,
    "traceability": 0.09,
}

# Weights inside the ``method_fidelity`` dimension.  ``phase_coverage``,
# ``evidence_utilisation`` and ``evidence_density`` vary across honest skills:
# they measure how much of the teacher's own method the miner recovered.  The
# other three sit at 1.0 for an honest skill and only collapse when a step's
# provenance is fabricated, so they are the falsifiable half of the dimension.
METHOD_FIDELITY_WEIGHTS = {
    "phase_coverage": 0.30,
    "cue_verification": 0.20,
    "span_consistency": 0.15,
    "evidence_utilisation": 0.15,
    "temporal_monotonicity": 0.10,
    "evidence_density": 0.10,
}

# A procedure quoting two transcript segments per observed phase counts as
# fully dense; further quotes add breadth rather than method fidelity.
TARGET_EVIDENCE_PER_OBSERVED_STEP = 2.0


def _percent(parts: list[bool]) -> float:
    return round(100 * sum(parts) / len(parts), 1) if parts else 0.0


def _semantic_support_valid(item: dict[str, Any]) -> bool:
    quote = str(item.get("quote", "")).lower()
    supports = item.get("supports", [])
    if not quote or not isinstance(supports, list) or not supports:
        return False
    for strategy in supports:
        if strategy == "teaching_sequence":
            continue
        patterns = STRATEGY_PATTERNS.get(str(strategy))
        if not patterns or not any(pattern.lower() in quote for pattern in patterns):
            return False
    return True


def _strategy_origin(strategy: dict[str, Any]) -> str:
    origin = str(strategy.get("origin", ""))
    if origin in {"observed_method", "recommended_enrichment"}:
        return origin
    try:
        evidence_count = int(strategy.get("evidence_count", 0))
    except (TypeError, ValueError):
        evidence_count = 0
    return "observed_method" if evidence_count > 0 else "recommended_enrichment"


def _procedure_provenance_diagnostics(skill: dict[str, Any]) -> dict[str, Any]:
    source = skill.get("source", {}) if isinstance(skill.get("source"), dict) else {}
    text_evidence_ids = {
        str(item.get("evidence_id"))
        for item in source.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    multimodal_evidence_ids = {
        str(item.get("event_id"))
        for item in source.get("multimodal_evidence", [])
        if isinstance(item, dict) and item.get("event_id")
    }
    known_evidence_ids = text_evidence_ids | multimodal_evidence_ids
    procedure = skill.get("procedure", [])
    if not isinstance(procedure, list):
        procedure = []

    origin_counts: Counter[str] = Counter()
    no_evidence_step_count = 0
    observed_steps_with_valid_evidence = 0
    observed_steps_without_valid_evidence = 0
    invalid_evidence_references: list[dict[str, Any]] = []
    provenance_mismatch_steps: list[Any] = []
    step_audit: list[dict[str, Any]] = []
    for index, step in enumerate(procedure):
        if not isinstance(step, dict):
            origin_counts["unknown"] += 1
            no_evidence_step_count += 1
            step_audit.append(
                {
                    "step": index + 1,
                    "origin": "unknown",
                    "evidence_ids": [],
                    "valid_evidence_ids": [],
                    "invalid_evidence_ids": [],
                }
            )
            continue
        provenance = step.get("provenance", {})
        if not isinstance(provenance, dict):
            provenance = {}
        origin = str(step.get("origin") or provenance.get("origin") or "unknown")
        if origin not in {"observed_method", "recommended_enrichment"}:
            origin = "unknown"
        origin_counts[origin] += 1

        raw_evidence_ids = step.get("evidence_ids", provenance.get("evidence_ids", []))
        if not isinstance(raw_evidence_ids, list):
            raw_evidence_ids = []
        evidence_ids = list(dict.fromkeys(str(value) for value in raw_evidence_ids if value))
        valid_ids = [value for value in evidence_ids if value in known_evidence_ids]
        invalid_ids = [value for value in evidence_ids if value not in known_evidence_ids]
        if not evidence_ids:
            no_evidence_step_count += 1
        if invalid_ids:
            invalid_evidence_references.append(
                {
                    "step": step.get("step", index + 1),
                    "evidence_ids": invalid_ids,
                }
            )
        if (
            step.get("origin") is not None
            and provenance.get("origin") is not None
            and step.get("origin") != provenance.get("origin")
        ) or (
            step.get("evidence_ids") is not None
            and provenance.get("evidence_ids") is not None
            and list(step.get("evidence_ids", [])) != list(provenance.get("evidence_ids", []))
        ):
            provenance_mismatch_steps.append(step.get("step", index + 1))

        if origin == "observed_method":
            if valid_ids and not invalid_ids:
                observed_steps_with_valid_evidence += 1
            else:
                observed_steps_without_valid_evidence += 1
        step_audit.append(
            {
                "step": step.get("step", index + 1),
                "origin": origin,
                "evidence_ids": evidence_ids,
                "valid_evidence_ids": valid_ids,
                "invalid_evidence_ids": invalid_ids,
                "counts_as_video_method_evidence": (
                    origin == "observed_method" and bool(valid_ids) and not invalid_ids
                ),
            }
        )

    total = len(procedure)
    observed_count = origin_counts["observed_method"]
    recommended_count = origin_counts["recommended_enrichment"]
    unknown_count = origin_counts["unknown"]
    return {
        "total_step_count": total,
        "observed_method_step_count": observed_count,
        "recommended_enrichment_step_count": recommended_count,
        "unknown_origin_step_count": unknown_count,
        "observed_method_step_ratio": round(observed_count / total, 3) if total else 0.0,
        "recommended_enrichment_step_ratio": round(recommended_count / total, 3) if total else 0.0,
        "unknown_origin_step_ratio": round(unknown_count / total, 3) if total else 0.0,
        "no_evidence_step_count": no_evidence_step_count,
        "observed_steps_with_valid_evidence_count": observed_steps_with_valid_evidence,
        "observed_steps_without_valid_evidence_count": observed_steps_without_valid_evidence,
        "invalid_evidence_reference_count": sum(
            len(item["evidence_ids"]) for item in invalid_evidence_references
        ),
        "invalid_evidence_references": invalid_evidence_references,
        "provenance_mismatch_steps": provenance_mismatch_steps,
        "known_text_evidence_id_count": len(text_evidence_ids),
        "known_multimodal_evidence_id_count": len(multimodal_evidence_ids),
        "observed_method_established_from_step_evidence": observed_steps_with_valid_evidence > 0,
        "recommended_enrichment_counts_as_video_method_evidence": False,
        "step_audit": step_audit,
    }


def _method_fidelity(skill: dict[str, Any]) -> dict[str, Any]:
    """Score how faithfully the procedure reproduces the teacher's own method.

    Every other dimension checks fields the miner emits by construction, so they
    saturate.  This one is computed by re-deriving each ``observed_method``
    step's claims from the evidence records it cites: a step that claims a cue,
    a timespan or an evidence id it cannot support loses points.  A pure
    template with no observed steps scores near zero.
    """

    source = skill.get("source", {}) if isinstance(skill.get("source"), dict) else {}
    evidence_by_id = {
        str(item["evidence_id"]): item
        for item in source.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    multimodal_by_id = {
        str(item["event_id"]): item
        for item in source.get("multimodal_evidence", [])
        if isinstance(item, dict) and item.get("event_id")
    }
    procedure = skill.get("procedure", [])
    if not isinstance(procedure, list):
        procedure = []
    observed = [
        step
        for step in procedure
        if isinstance(step, dict) and step.get("origin") == "observed_method"
    ]

    def _records(step: dict[str, Any]) -> list[dict[str, Any]]:
        found = []
        for value in step.get("evidence_ids", []) or []:
            record = evidence_by_id.get(str(value)) or multimodal_by_id.get(str(value))
            if isinstance(record, dict):
                found.append(record)
        return found

    cue_verified = 0
    span_consistent = 0
    cited_ids: set[str] = set()
    quoted_segments = 0
    starts: list[float] = []
    for step in observed:
        records = _records(step)
        cited_ids.update(str(value) for value in step.get("evidence_ids", []) or [])
        quoted_segments += len(records)

        # A claimed cue must literally occur in the quoted evidence.
        haystack = " ".join(str(record.get("quote", "")) for record in records).lower()
        cues = [str(cue).lower() for cue in step.get("matched_cues", []) or [] if cue]
        if cues and haystack and all(cue in haystack for cue in cues):
            cue_verified += 1

        # The claimed span must actually contain every record it cites.
        span = step.get("observed_span") or {}
        try:
            low = float(span["start"])
            high = float(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        starts.append(low)
        if records and all(
            low - 1e-6 <= float(record.get("start", low - 1))
            and float(record.get("end", high + 1)) <= high + 1e-6
            for record in records
        ):
            span_consistent += 1

    observed_count = len(observed)
    canonical_total = len(CANONICAL_PHASES)
    distinct_phases = {
        step.get("teaching_phase") for step in observed if step.get("teaching_phase")
    }
    ordered_pairs = list(zip(starts, starts[1:]))
    components = {
        "phase_coverage": (
            len(distinct_phases) / canonical_total if canonical_total else 0.0
        ),
        "cue_verification": cue_verified / observed_count if observed_count else 0.0,
        "span_consistency": span_consistent / observed_count if observed_count else 0.0,
        "evidence_utilisation": (
            len(cited_ids & set(evidence_by_id)) / len(evidence_by_id)
            if evidence_by_id
            else 0.0
        ),
        "temporal_monotonicity": (
            sum(1 for earlier, later in ordered_pairs if earlier <= later)
            / len(ordered_pairs)
            if ordered_pairs
            else (1.0 if observed_count else 0.0)
        ),
        "evidence_density": (
            min(
                1.0,
                quoted_segments
                / (observed_count * TARGET_EVIDENCE_PER_OBSERVED_STEP),
            )
            if observed_count
            else 0.0
        ),
    }
    score = round(
        100
        * sum(components[key] * weight for key, weight in METHOD_FIDELITY_WEIGHTS.items()),
        1,
    )
    return {
        "score": min(100.0, max(0.0, score)),
        "components": {key: round(value, 3) for key, value in components.items()},
        "weights": METHOD_FIDELITY_WEIGHTS,
        "observed_step_count": observed_count,
        "distinct_observed_phase_count": len(distinct_phases),
        "canonical_phase_count": canonical_total,
        "cue_verified_step_count": cue_verified,
        "span_consistent_step_count": span_consistent,
        "cited_text_evidence_count": len(cited_ids & set(evidence_by_id)),
        "available_text_evidence_count": len(evidence_by_id),
        "score_semantics": (
            "衡量 procedure 中 observed_method 步骤能被其引用证据反推验证的程度："
            "线索、时间区间、证据利用率与时间顺序均由证据记录重新推导，"
            "不代表真实课堂教学效果。"
        ),
    }


def evaluate_skill(skill: dict[str, Any], transcript: dict[str, Any] | None = None) -> dict[str, Any]:
    validation = validate_skill(skill)
    transcript_validation = validate_transcript(transcript) if transcript is not None else None
    structural = _percent([skill.get(field) not in (None, "", []) for field in REQUIRED_SKILL_FIELDS])

    source = skill.get("source", {})
    evidence = source.get("evidence", [])
    if transcript is None:
        grounding_checks = [
            bool(item.get("quote"))
            and item.get("start") is not None
            and item.get("end") is not None
            and _semantic_support_valid(item)
            for item in evidence
            if isinstance(item, dict)
        ]
        grounding = min(_percent(grounding_checks), 75.0) if grounding_checks else 0.0
        grounding_note = "未提供原转写，证据只能检查格式，分数上限为 75。"
    else:
        grounding_checks = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote", ""))
            try:
                evidence_start = float(item["start"])
                evidence_end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                timestamp_match = False
            else:
                timestamp_match = any(
                    quote in str(segment.get("text", ""))
                    and abs(evidence_start - float(segment.get("start", -2))) < 0.001
                    and abs(evidence_end - float(segment.get("end", -2))) < 0.001
                    for segment in transcript.get("segments", [])
                )
            grounding_checks.append(bool(quote) and timestamp_match and _semantic_support_valid(item))
        count_factor = min(1.0, len(grounding_checks) / 2)
        grounding = round(_percent(grounding_checks) * count_factor, 1) if transcript_validation and transcript_validation.valid else 0.0
        grounding_note = "证据引文与带时间戳的原转写进行精确匹配。"
        if transcript.get("transcript_kind") == "curated_paraphrase_excerpt":
            grounding = min(grounding, 85.0)
            grounding_note += " 当前为人工释义节选，证据忠实度自动分上限为 85；正式实验应替换为完整字幕或 ASR。"

    multimodal_refs = source.get("multimodal_evidence", [])
    transcript_events = transcript.get("multimodal", {}).get("events", []) if transcript else []
    multimodal_expected = bool(transcript_events)
    multimodal_checks: list[bool] = []
    matched_modalities: set[str] = set()
    if multimodal_expected:
        event_by_id = {event.get("event_id"): event for event in transcript_events if isinstance(event, dict)}
        for reference in multimodal_refs:
            if not isinstance(reference, dict):
                continue
            event = event_by_id.get(reference.get("event_id"))
            matched = bool(
                event
                and reference.get("type") == event.get("type")
                and reference.get("start") == event.get("start")
                and reference.get("end") == event.get("end")
                and set(reference.get("modalities", [])) == set(event.get("modalities", []))
                and set(reference.get("supports", [])) <= set(event.get("supports_strategies", []))
                and reference.get("evidence") == event.get("evidence")
                and reference.get("confidence") == event.get("confidence")
            )
            multimodal_checks.append(matched)
            if matched:
                matched_modalities.update(reference.get("modalities", []))
        available_modalities = set(transcript.get("multimodal", {}).get("modalities_available", []))
        exact_score = _percent(multimodal_checks)
        modality_score = (
            min(100.0, round(100 * len(matched_modalities & available_modalities) / len(available_modalities), 1))
            if available_modalities
            else 0.0
        )
        multimodal_score = round(0.8 * exact_score + 0.2 * modality_score, 1) if multimodal_checks else 0.0
        multimodal_score = min(100.0, multimodal_score)
        grounding = min(100.0, round(0.7 * grounding + 0.3 * multimodal_score, 1))
        grounding_note += " 多模态证据按事件 ID、时间戳、模态集合和策略映射核对。"
    else:
        available_modalities = set()
        multimodal_score = None

    procedure = skill.get("procedure", []) if isinstance(skill.get("procedure"), list) else []
    method_provenance = _procedure_provenance_diagnostics(skill)
    strategy_origins = Counter(
        _strategy_origin(item)
        for item in skill.get("strategies", [])
        if isinstance(item, dict)
    )
    step_checks: list[bool] = []
    for step in procedure:
        step_checks.append(
            isinstance(step, dict)
            and all(step.get(field) not in (None, "") for field in ("step", "teacher_action", "instruction", "expected_signal", "fallback"))
        )
    executability = round(
        0.7 * _percent(step_checks)
        + 10 * (len(procedure) >= 4)
        + 10 * bool(skill.get("teacher_actions"))
        + 10 * bool(skill.get("parameters")),
        1,
    )
    executability = min(100.0, executability)

    verification = skill.get("verification", [])
    pedagogy_checks = [
        len(skill.get("trigger", [])) >= 1,
        len(skill.get("preconditions", [])) >= 1,
        len(skill.get("student_signals", [])) >= 2,
        len(skill.get("success_criteria", [])) >= 2,
        len(skill.get("failure_modes", [])) >= 2,
        isinstance(verification, list) and len(verification) >= 2,
        bool(skill.get("learning_objective", {}).get("observable")),
    ]
    pedagogical = _percent(pedagogy_checks)

    instructions = " ".join(str(step.get("instruction", "")) for step in procedure if isinstance(step, dict))
    general_checks = [
        "{concept}" in instructions,
        bool(skill.get("parameters", {}).get("concept")),
        any(item.get("type") == "near_transfer" for item in verification if isinstance(item, dict)),
        any("边界" in str(item) or "反例" in str(item) for item in skill.get("failure_modes", []) + verification),
    ]
    generalizability = _percent(general_checks)

    trace_checks = [
        bool(source.get("video_id")),
        bool(source.get("course_id")),
        str(source.get("source_url", "")).startswith(("https://", "http://")),
        bool(source.get("transcript_kind")),
        len(evidence) >= 2,
        bool(source.get("transcript_url")),
    ]
    if multimodal_expected:
        trace_checks.extend([bool(source.get("modalities_available")), bool(multimodal_refs)])
    traceability = _percent(trace_checks)
    if transcript and transcript.get("timestamps_are_approximate"):
        traceability = min(traceability, 85.0)

    method_fidelity = _method_fidelity(skill)
    dimensions = {
        "structural_completeness": structural,
        "evidence_grounding": grounding,
        "executability": executability,
        "method_fidelity": method_fidelity["score"],
        "pedagogical_quality": pedagogical,
        "generalizability": generalizability,
        "traceability": traceability,
    }
    overall = min(100.0, round(sum(dimensions[key] * weight for key, weight in DIMENSION_WEIGHTS.items()), 1))
    gates = {
        "schema_valid": validation.valid,
        "grounded": grounding >= 60,
        "executable": executability >= 70,
        "testable": isinstance(verification, list) and len(verification) >= 2,
        "multimodal_consistent": not multimodal_expected
        or (
            bool(multimodal_checks)
            and multimodal_score >= 60
            and bool(transcript_validation and transcript_validation.valid)
        ),
        "method_distilled_from_video": method_fidelity["score"] >= 40,
    }
    grounding_note += (
        " Procedure 中 observed_method 步骤只有在引用有效 evidence_id 时才计为视频中观察到的方法；"
        "recommended_enrichment 只作为可执行教学补充，不计作视频方法证据。"
    )
    return {
        "skill_id": skill.get("skill_id"),
        "overall_score": overall,
        "score_scope": "structural_quality_and_internal_evidence_consistency",
        "teaching_effectiveness_established": False,
        "real_world_recognition_accuracy_established": False,
        "grade": "A" if overall >= 90 else "B" if overall >= 80 else "C" if overall >= 70 else "D",
        "passed": overall >= 75 and all(gates.values()),
        "threshold": 75,
        "dimensions": dimensions,
        "weights": DIMENSION_WEIGHTS,
        "gates": gates,
        "validation": validation.as_dict(),
        "transcript_validation": transcript_validation.as_dict() if transcript_validation else None,
        "notes": [grounding_note],
        "method_fidelity": method_fidelity,
        "method_provenance": {
            **method_provenance,
            "observed_strategy_count": strategy_origins["observed_method"],
            "recommended_strategy_count": strategy_origins[
                "recommended_enrichment"
            ],
        },
        "grounding_diagnostics": {
            "evidence_count": len(evidence),
            "fully_matched_count": sum(grounding_checks),
            "observed_method_step_count": method_provenance[
                "observed_method_step_count"
            ],
            "recommended_enrichment_step_count": method_provenance[
                "recommended_enrichment_step_count"
            ],
            "observed_steps_with_valid_evidence_count": method_provenance[
                "observed_steps_with_valid_evidence_count"
            ],
            "no_evidence_step_count": method_provenance[
                "no_evidence_step_count"
            ],
            "checks_include": ["exact_quote", "exact_segment_timestamps", "strategy_semantic_support"],
        },
        "multimodal_evaluation": {
            "metric_name": "internal_evidence_consistency",
            "score_semantics": "Skill references are checked against validated analysis records; this is not detector accuracy.",
            "independent_ground_truth_used": False,
            "recognition_precision": None,
            "recognition_recall": None,
            "recognition_f1": None,
            "expected": multimodal_expected,
            "source_reference_count": len(multimodal_refs),
            "matched_reference_count": sum(multimodal_checks),
            "available_modalities": sorted(available_modalities),
            "matched_modalities": sorted(matched_modalities),
            "score": multimodal_score,
            "checks_include": [
                "event_id",
                "event_type",
                "timestamps",
                "modality_set",
                "strategy_mapping",
                "evidence_payload",
                "confidence",
                "referenced_frame_and_audio_records",
            ],
        },
    }


def evaluate_collection(
    skills: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strategy_counter: Counter[str] = Counter()
    recommended_strategy_counter: Counter[str] = Counter()
    video_ids: set[str] = set()
    course_ids: set[str] = set()
    for skill in skills:
        video_ids.add(str(skill.get("source", {}).get("video_id", "")))
        course_ids.add(str(skill.get("source", {}).get("course_id", "")))
        for item in skill.get("strategies", []):
            if not isinstance(item, dict) or not item.get("id"):
                continue
            if _strategy_origin(item) == "observed_method":
                strategy_counter.update([item["id"]])
            else:
                recommended_strategy_counter.update([item["id"]])
    expected_videos = set()
    expected_courses = set()
    if manifest:
        expected_videos = {str(item["video_id"]) for item in manifest.get("videos", [])}
        expected_courses = {str(item["course_id"]) for item in manifest.get("videos", [])}
    coverage = sorted(strategy_counter)
    checks = {
        "at_least_two_courses": len(course_ids) >= 2,
        "at_least_ten_videos": len(video_ids) >= 10,
        "skill_for_every_manifest_video": not expected_videos or expected_videos <= video_ids,
        "at_least_five_strategy_types": len(coverage) >= 5,
        "all_skills_pass": all(report.get("passed") for report in reports),
    }
    return {
        "skill_count": len(skills),
        "video_count": len(video_ids),
        "course_count": len(course_ids),
        "expected_course_count": len(expected_courses),
        "average_score": round(mean(report["overall_score"] for report in reports), 1) if reports else 0.0,
        "pass_rate": round(sum(report.get("passed", False) for report in reports) / len(reports), 3) if reports else 0.0,
        "strategy_type_count": len(coverage),
        "strategy_coverage": coverage,
        "strategy_frequency": dict(sorted(strategy_counter.items())),
        "recommended_strategy_frequency": dict(
            sorted(recommended_strategy_counter.items())
        ),
        "strategy_coverage_semantics": (
            "Only observed_method strategies count toward video-method coverage; "
            "recommended_enrichment is reported separately."
        ),
        "requirement_checks": checks,
        "passed": all(checks.values()),
    }
