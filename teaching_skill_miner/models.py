from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


REQUIRED_SKILL_FIELDS = (
    "skill_id",
    "name",
    "version",
    "source",
    "learning_objective",
    "trigger",
    "preconditions",
    "goal",
    "strategies",
    "procedure",
    "teacher_actions",
    "student_signals",
    "success_criteria",
    "failure_modes",
    "verification",
)

ALLOWED_ACTIONS = {
    "ask",
    "explain",
    "demonstrate",
    "contrast",
    "prompt",
    "check_understanding",
    "give_feedback",
    "summarize",
    "adapt",
}

ALLOWED_BLOOM_LEVELS = {
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
}
ALLOWED_METHOD_ORIGINS = {"observed_method", "recommended_enrichment"}
EVIDENCE_ID_RE = re.compile(r"(?:evi_[0-9a-f]{16}|mme_[0-9]{4,})")

ALLOWED_MULTIMODAL_MODALITIES = {
    "transcript",
    "speech",
    "audio",
    "visual",
    "ocr",
    "classroom_observation",
}

MULTIMODAL_REQUIRED_MODALITIES: dict[str, set[str]] = {
    "question_and_wait": {"audio"},
    "scene_change": {"visual"},
    "slide_change": {"visual", "ocr"},
    "board_build_up": {"visual", "ocr"},
    "code_or_formula_visible": {"visual", "ocr"},
    "code_formula_walkthrough": {"visual", "ocr"},
    "visual_example": {"visual"},
    "student_confusion": {"classroom_observation"},
    "teacher_adjustment": {"classroom_observation"},
    "student_answer": {"classroom_observation"},
}

MULTIMODAL_REQUIRED_EVIDENCE: dict[str, set[str]] = {
    "question_and_wait": {"speech_quote", "question_segment", "silence", "wait_seconds"},
    "scene_change": {"frame_path", "ocr_text"},
    "slide_change": {"before_frame", "after_frame", "before_text", "after_text", "text_similarity"},
    "board_build_up": {"before_frame", "after_frame", "added_token_count", "after_text"},
    "code_or_formula_visible": {"frame_path", "ocr_text"},
    "code_formula_walkthrough": {"speech_quote", "frame_path", "ocr_text"},
    "visual_example": {"speech_quote"},
    "student_confusion": {"anonymized_note", "evidence_origin"},
    "teacher_adjustment": {"anonymized_note", "evidence_origin"},
    "student_answer": {"anonymized_note", "evidence_origin"},
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}


def validate_transcript(data: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    for field in ("video_id", "course_id", "title", "source_url", "segments"):
        if not data.get(field):
            errors.append(f"missing transcript field: {field}")
    segments = data.get("segments", [])
    if not isinstance(segments, list):
        errors.append("segments must be a list")
        return ValidationResult(False, errors, warnings)
    previous_end = -1.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            errors.append(f"segments[{index}] must be an object")
            continue
        if not str(segment.get("text", "")).strip():
            errors.append(f"segments[{index}].text is empty")
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (TypeError, ValueError):
            errors.append(f"segments[{index}] has invalid timestamps")
            continue
        if start < 0 or end <= start:
            errors.append(f"segments[{index}] must satisfy 0 <= start < end")
        if start < previous_end:
            warnings.append(f"segments[{index}] overlaps the previous segment")
        previous_end = max(previous_end, end)
    if len(segments) < 3:
        warnings.append("fewer than 3 segments gives weak pedagogical evidence")
    multimodal = data.get("multimodal")
    if multimodal is not None:
        if not isinstance(multimodal, dict):
            errors.append("multimodal must be an object")
        else:
            modalities = multimodal.get("modalities_available", [])
            if not isinstance(modalities, list) or not all(isinstance(item, str) for item in modalities):
                errors.append("multimodal.modalities_available must be a list of strings")
                modality_set: set[str] = set()
            else:
                modality_set = set(modalities)
                unknown_modalities = sorted(modality_set - ALLOWED_MULTIMODAL_MODALITIES)
                if unknown_modalities:
                    errors.append(f"unknown multimodal modalities: {', '.join(unknown_modalities)}")

            keyframes = multimodal.get("visual", {}).get("keyframes", [])
            if not isinstance(keyframes, list):
                errors.append("multimodal.visual.keyframes must be a list")
                keyframes = []
            frame_by_path = {
                str(frame.get("path")): frame
                for frame in keyframes
                if isinstance(frame, dict) and frame.get("path")
            }
            silences = multimodal.get("audio", {}).get("silences", [])
            if not isinstance(silences, list):
                errors.append("multimodal.audio.silences must be a list")
                silences = []
            media_duration = multimodal.get("media", {}).get("duration_seconds")
            try:
                media_duration_value = float(media_duration) if media_duration is not None else None
            except (TypeError, ValueError):
                errors.append("multimodal.media.duration_seconds must be numeric")
                media_duration_value = None
            events = multimodal.get("events", [])
            if not isinstance(events, list):
                errors.append("multimodal.events must be a list")
            else:
                event_ids: set[str] = set()
                for index, event in enumerate(events):
                    if not isinstance(event, dict):
                        errors.append(f"multimodal.events[{index}] must be an object")
                        continue
                    for field in ("event_id", "type", "start", "end", "modalities", "evidence", "supports_strategies"):
                        if field not in event:
                            errors.append(f"multimodal.events[{index}].{field} is required")
                    event_id = str(event.get("event_id", ""))
                    if not re.fullmatch(r"mme_[0-9]{4,}", event_id):
                        errors.append(f"multimodal.events[{index}].event_id is invalid")
                    if event_id in event_ids:
                        errors.append(f"duplicate multimodal event_id: {event_id}")
                    event_ids.add(event_id)
                    event_type = str(event.get("type", ""))
                    if event_type not in MULTIMODAL_REQUIRED_MODALITIES:
                        errors.append(f"multimodal.events[{index}].type is unsupported")
                    try:
                        event_start = float(event.get("start"))
                        event_end = float(event.get("end"))
                    except (TypeError, ValueError):
                        errors.append(f"multimodal.events[{index}] has invalid timestamps")
                    else:
                        if event_start < 0 or event_end < event_start:
                            errors.append(f"multimodal.events[{index}] must satisfy 0 <= start <= end")
                        if media_duration_value is not None and event_end > media_duration_value + 0.001:
                            errors.append(f"multimodal.events[{index}].end exceeds media duration")
                    event_modalities = event.get("modalities")
                    if not isinstance(event_modalities, list) or not event_modalities:
                        errors.append(f"multimodal.events[{index}].modalities must be a non-empty list")
                        event_modality_set: set[str] = set()
                    else:
                        event_modality_set = set(event_modalities)
                        if len(event_modality_set) != len(event_modalities):
                            errors.append(f"multimodal.events[{index}].modalities contains duplicates")
                        if event_modality_set - ALLOWED_MULTIMODAL_MODALITIES:
                            errors.append(f"multimodal.events[{index}].modalities contains unsupported values")
                        if not event_modality_set <= modality_set:
                            errors.append(f"multimodal.events[{index}].modalities are not declared available")
                        required_modalities = MULTIMODAL_REQUIRED_MODALITIES.get(event_type, set())
                        if not required_modalities <= event_modality_set:
                            errors.append(f"multimodal.events[{index}] lacks required modalities for {event_type}")
                        if event_type in {"question_and_wait", "code_formula_walkthrough", "visual_example"}:
                            language_modalities = event_modality_set & {"speech", "transcript"}
                            if len(language_modalities) != 1:
                                errors.append(
                                    f"multimodal.events[{index}] requires exactly one of speech or transcript"
                                )

                    confidence = event.get("confidence")
                    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                        errors.append(f"multimodal.events[{index}].confidence must be between 0 and 1")

                    evidence = event.get("evidence")
                    if not isinstance(evidence, dict):
                        errors.append(f"multimodal.events[{index}].evidence must be an object")
                        evidence = {}
                    missing_evidence = MULTIMODAL_REQUIRED_EVIDENCE.get(event_type, set()) - set(evidence)
                    if missing_evidence:
                        errors.append(
                            f"multimodal.events[{index}].evidence missing: {', '.join(sorted(missing_evidence))}"
                        )

                    expected_strategies: set[str] = set()
                    if event_type:
                        # Imported lazily to keep the data-model module lightweight and avoid module-init cycles.
                        from .multimodal import EVENT_STRATEGY_MAP

                        expected_strategies = set(EVENT_STRATEGY_MAP.get(event_type, ()))
                    supports = event.get("supports_strategies")
                    if not isinstance(supports, list) or set(supports) != expected_strategies:
                        errors.append(f"multimodal.events[{index}].supports_strategies does not match event type")

                    if event_type in {"code_or_formula_visible", "code_formula_walkthrough"}:
                        frame = frame_by_path.get(str(evidence.get("frame_path", "")))
                        if frame is None:
                            errors.append(f"multimodal.events[{index}] references an unknown frame")
                        elif str(evidence.get("ocr_text", "")) != str(frame.get("ocr_text", "")):
                            errors.append(f"multimodal.events[{index}] OCR evidence does not match its frame")
                        else:
                            from .multimodal import CODE_OCR_RE

                            if not CODE_OCR_RE.search(str(evidence.get("ocr_text", ""))):
                                errors.append(f"multimodal.events[{index}] OCR evidence contains no code/formula marker")
                    elif event_type == "scene_change":
                        if str(evidence.get("frame_path", "")) not in frame_by_path:
                            errors.append(f"multimodal.events[{index}] references an unknown frame")
                    elif event_type in {"slide_change", "board_build_up"}:
                        before = frame_by_path.get(str(evidence.get("before_frame", "")))
                        after = frame_by_path.get(str(evidence.get("after_frame", "")))
                        if before is None or after is None:
                            errors.append(f"multimodal.events[{index}] references unknown before/after frames")
                        elif event_type == "slide_change" and (
                            str(evidence.get("before_text", "")) != str(before.get("ocr_text", ""))
                            or str(evidence.get("after_text", "")) != str(after.get("ocr_text", ""))
                        ):
                            errors.append(f"multimodal.events[{index}] slide OCR does not match referenced frames")

                    if event_type in {"question_and_wait", "code_formula_walkthrough", "visual_example"}:
                        speech_quote = str(evidence.get("speech_quote", ""))
                        if not speech_quote or not any(speech_quote == str(segment.get("text", "")) for segment in segments):
                            errors.append(f"multimodal.events[{index}] speech evidence is not in the transcript")
                    if event_type == "question_and_wait":
                        silence = evidence.get("silence")
                        if not isinstance(silence, dict) or silence not in silences:
                            errors.append(f"multimodal.events[{index}] silence evidence is not in audio analysis")
                        question_segment = evidence.get("question_segment")
                        matching_segment = next(
                            (
                                segment
                                for segment in segments
                                if isinstance(question_segment, dict)
                                and str(segment.get("text", ""))
                                == str(evidence.get("speech_quote", ""))
                                and question_segment.get("start")
                                == segment.get("start")
                                and question_segment.get("end")
                                == segment.get("end")
                            ),
                            None,
                        )
                        if not isinstance(question_segment, dict) or matching_segment is None or (
                            question_segment.get("start") != matching_segment.get("start")
                            or question_segment.get("end") != matching_segment.get("end")
                        ):
                            errors.append(f"multimodal.events[{index}] question segment does not match transcript")
                        if isinstance(silence, dict) and matching_segment is not None:
                            calculated_wait = max(
                                0.0,
                                float(silence.get("end", 0))
                                - max(float(silence.get("start", 0)), float(matching_segment.get("end", 0))),
                            )
                            try:
                                reported_wait = float(evidence.get("wait_seconds"))
                            except (TypeError, ValueError):
                                reported_wait = -1.0
                            if abs(round(calculated_wait, 3) - reported_wait) > 0.001:
                                errors.append(f"multimodal.events[{index}] wait_seconds is inconsistent")
                    if event_type in {"student_confusion", "teacher_adjustment", "student_answer"}:
                        if evidence.get("evidence_origin") != "provided_anonymized_annotation":
                            errors.append(f"multimodal.events[{index}] classroom observation origin is invalid")
                        if any(key in evidence for key in ("student_name", "face_id", "identity")):
                            errors.append(f"multimodal.events[{index}] contains identity data")
    return ValidationResult(not errors, errors, warnings)


def validate_skill(skill: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(skill, dict):
        return ValidationResult(False, ["skill must be an object"], warnings)
    for field in REQUIRED_SKILL_FIELDS:
        if field not in skill or skill[field] in (None, "", []):
            errors.append(f"missing or empty skill field: {field}")

    skill_id = str(skill.get("skill_id", ""))
    if skill_id and not re.fullmatch(r"[a-z0-9][a-z0-9_\-]*", skill_id):
        errors.append("skill_id must contain lowercase letters, digits, underscores, or hyphens")
    version = skill.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+", version):
        errors.append("version must use major.minor numeric form")

    objective = skill.get("learning_objective", {})
    if not isinstance(objective, dict):
        errors.append("learning_objective must be an object")
    elif objective.get("bloom_level") not in ALLOWED_BLOOM_LEVELS:
        errors.append("learning_objective.bloom_level is invalid")
    elif not objective.get("observable"):
        warnings.append("learning objective is not marked observable")

    actions = skill.get("teacher_actions", [])
    if not isinstance(actions, list):
        errors.append("teacher_actions must be a list")
    else:
        unknown = sorted(set(actions) - ALLOWED_ACTIONS)
        if unknown:
            errors.append(f"unknown teacher_actions: {', '.join(unknown)}")
        if len(actions) != len(set(actions)):
            errors.append("teacher_actions must not contain duplicates")

    for field in ("trigger", "preconditions", "student_signals", "success_criteria"):
        values = skill.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            errors.append(f"{field} must be a non-empty list of non-empty strings")

    strategies = skill.get("strategies", [])
    strategy_ids: set[str] = set()
    if not isinstance(strategies, list) or not strategies:
        errors.append("strategies must be a non-empty list")
    else:
        for index, strategy in enumerate(strategies):
            if not isinstance(strategy, dict):
                errors.append(f"strategies[{index}] must be an object")
                continue
            strategy_id = strategy.get("id")
            if not isinstance(strategy_id, str) or not strategy_id:
                errors.append(f"strategies[{index}].id is required")
            elif strategy_id in strategy_ids:
                errors.append(f"duplicate strategy id: {strategy_id}")
            else:
                strategy_ids.add(strategy_id)
            if not isinstance(strategy.get("name"), str) or not strategy["name"].strip():
                errors.append(f"strategies[{index}].name is required")
            evidence_count = strategy.get("evidence_count")
            if isinstance(evidence_count, bool) or not isinstance(evidence_count, int) or evidence_count < 0:
                errors.append(f"strategies[{index}].evidence_count must be a non-negative integer")
            confidence = strategy.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                errors.append(f"strategies[{index}].confidence must be between 0 and 1")
            if strategy.get("origin") not in ALLOWED_METHOD_ORIGINS:
                errors.append(f"strategies[{index}].origin is invalid")

    source = skill.get("source", {})
    source_evidence_ids: set[str] = set()
    multimodal_evidence_ids: set[str] = set()
    if not isinstance(source, dict) or not all(
        isinstance(source.get(key), str) and source.get(key).strip()
        for key in ("video_id", "course_id", "source_url")
    ):
        errors.append("source needs non-empty video_id, course_id, and source_url")
    else:
        evidence = source.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append("source.evidence must be a non-empty list")
        else:
            for index, item in enumerate(evidence):
                if not isinstance(item, dict):
                    errors.append(f"source.evidence[{index}] must be an object")
                    continue
                evidence_id = item.get("evidence_id")
                if not isinstance(evidence_id, str) or not re.fullmatch(
                    r"evi_[0-9a-f]{16}", evidence_id
                ):
                    errors.append(f"source.evidence[{index}].evidence_id is invalid")
                elif evidence_id in source_evidence_ids:
                    errors.append(f"duplicate source evidence_id: {evidence_id}")
                else:
                    source_evidence_ids.add(evidence_id)
                try:
                    start = float(item.get("start"))
                    end = float(item.get("end"))
                except (TypeError, ValueError, OverflowError):
                    errors.append(f"source.evidence[{index}] has invalid timestamps")
                else:
                    if start < 0 or end <= start:
                        errors.append(f"source.evidence[{index}] must satisfy 0 <= start < end")
                if not isinstance(item.get("quote"), str) or not item["quote"].strip():
                    errors.append(f"source.evidence[{index}].quote is required")
                supports = item.get("supports")
                if (
                    not isinstance(supports, list)
                    or not supports
                    or any(not isinstance(value, str) or not value for value in supports)
                    or len(set(supports)) != len(supports)
                ):
                    errors.append(f"source.evidence[{index}].supports is invalid")
        if "multimodal_evidence" in source:
            if not isinstance(source["multimodal_evidence"], list):
                errors.append("source.multimodal_evidence must be a list")
            else:
                for index, item in enumerate(source["multimodal_evidence"]):
                    if not isinstance(item, dict) or not all(
                        field in item
                        for field in (
                            "event_id", "type", "start", "end", "modalities", "supports", "evidence"
                        )
                    ):
                        errors.append(f"source.multimodal_evidence[{index}] is incomplete")
                        continue
                    event_id = item.get("event_id")
                    if not isinstance(event_id, str) or not re.fullmatch(r"mme_[0-9]{4,}", event_id):
                        errors.append(f"source.multimodal_evidence[{index}].event_id is invalid")
                    elif event_id in multimodal_evidence_ids:
                        errors.append(f"duplicate multimodal evidence event_id: {event_id}")
                    else:
                        multimodal_evidence_ids.add(event_id)

    procedure = skill.get("procedure", [])
    if not isinstance(procedure, list):
        errors.append("procedure must be a list")
    else:
        for index, step in enumerate(procedure):
            if not isinstance(step, dict):
                errors.append(f"procedure[{index}] must be an executable object")
                continue
            for field in ("step", "teacher_action", "instruction", "expected_signal", "fallback"):
                if step.get(field) in (None, ""):
                    errors.append(f"procedure[{index}].{field} is required")
            if step.get("teacher_action") not in ALLOWED_ACTIONS:
                errors.append(f"procedure[{index}].teacher_action is invalid")
            origin = step.get("origin")
            if origin not in ALLOWED_METHOD_ORIGINS:
                errors.append(f"procedure[{index}].origin is invalid")
            evidence_ids = step.get("evidence_ids")
            if (
                not isinstance(evidence_ids, list)
                or any(
                    not isinstance(value, str) or not EVIDENCE_ID_RE.fullmatch(value)
                    for value in (evidence_ids if isinstance(evidence_ids, list) else [])
                )
                or (
                    isinstance(evidence_ids, list)
                    and len(set(evidence_ids)) != len(evidence_ids)
                )
            ):
                errors.append(f"procedure[{index}].evidence_ids is invalid")
                evidence_ids = []
            known_evidence_ids = source_evidence_ids | multimodal_evidence_ids
            unknown_evidence = sorted(set(evidence_ids) - known_evidence_ids)
            if unknown_evidence:
                errors.append(
                    f"procedure[{index}] references unknown evidence: {', '.join(unknown_evidence)}"
                )
            if origin == "observed_method" and not evidence_ids:
                errors.append(f"procedure[{index}] observed_method requires evidence_ids")
            provenance = step.get("provenance")
            if not isinstance(provenance, dict):
                errors.append(f"procedure[{index}].provenance is required")
            else:
                expected_fields = {"origin", "strategy_id", "evidence_ids", "derivation"}
                if set(provenance) != expected_fields:
                    errors.append(f"procedure[{index}].provenance fields are invalid")
                if provenance.get("origin") != origin:
                    errors.append(f"procedure[{index}].provenance.origin must match origin")
                if provenance.get("evidence_ids") != evidence_ids:
                    errors.append(
                        f"procedure[{index}].provenance.evidence_ids must match evidence_ids"
                    )
                strategy_id = provenance.get("strategy_id")
                if strategy_id is not None and strategy_id not in strategy_ids:
                    errors.append(f"procedure[{index}].provenance.strategy_id is unknown")
                if origin == "observed_method" and strategy_id is None:
                    errors.append(
                        f"procedure[{index}] observed_method requires a strategy_id"
                    )
                if not isinstance(provenance.get("derivation"), str) or not provenance[
                    "derivation"
                ].strip():
                    errors.append(f"procedure[{index}].provenance.derivation is required")

    verification = skill.get("verification", [])
    if isinstance(verification, list):
        for index, check in enumerate(verification):
            if not isinstance(check, dict) or not all(check.get(k) for k in ("type", "prompt", "pass_condition")):
                errors.append(f"verification[{index}] needs type, prompt, and pass_condition")
    else:
        errors.append("verification must be a list")

    failure_modes = skill.get("failure_modes")
    if not isinstance(failure_modes, list) or not failure_modes:
        errors.append("failure_modes must be a non-empty list")
    else:
        for index, item in enumerate(failure_modes):
            if not isinstance(item, dict) or set(item) != {"mode", "mitigation"} or not all(
                isinstance(item.get(key), str) and item[key].strip()
                for key in ("mode", "mitigation")
            ):
                errors.append(f"failure_modes[{index}] needs mode and mitigation")
    return ValidationResult(not errors, errors, warnings)
