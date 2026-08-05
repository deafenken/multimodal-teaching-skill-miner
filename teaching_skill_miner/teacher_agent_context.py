"""Privacy-safe context construction for the task-two Teaching Agent.

The functions in this module are deliberately deterministic and dependency
free.  They prepare the smallest useful learner context for a remote language
model, but they do not perform a network request or claim that a model-derived
student state is correct.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import re
from typing import Any, Mapping, Sequence


CONTEXT_SCHEMA = "teaching_skill_miner.teacher_agent_context.v1"
LAYERED_CONTEXT_SCHEMA = "teaching_skill_miner.layered_teacher_context.v1"
GOAL_PLAN_SCHEMA = "teaching_skill_miner.teacher_agent_goal_plan.v1"

DEFAULT_LAYERED_CONTEXT_CHARS = 14_000
MINIMUM_LAYERED_CONTEXT_CHARS = 6_000

_REPLACEMENTS = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "cn_id": "[REDACTED_CN_ID]",
    "url": "[REDACTED_URL]",
    "local_path": "[REDACTED_LOCAL_PATH]",
}

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "url",
        re.compile(
            r"(?i)(?<![\w@])(?:https?://|ftp://|file://|www\.)"
            r"[^\s<>\"'，。！？；、（）【】《》]+"
        ),
    ),
    (
        "email",
        re.compile(
            r"(?i)(?<![A-Z0-9._%+\-])"
            r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}"
            r"(?![A-Z0-9._%+\-])"
        ),
    ),
    (
        "cn_id",
        re.compile(r"(?<!\d)\d{17}[0-9Xx](?![0-9Xx])"),
    ),
    (
        "phone",
        re.compile(r"(?<!\d)(?:\+?86[\s-]?)?1[3-9]\d(?:[\s-]?\d){8}(?!\d)"),
    ),
    (
        "local_path",
        re.compile(
            r"(?<![A-Za-z0-9])(?:"
            r"[A-Za-z]:\\(?:[^\\\r\n<>:\"|?*]+\\)*[^\\\r\n<>:\"|?*]*"
            r"|/(?!/)(?:[^/\s<>:\"|?*]+/)*[^/\s<>:\"|?*]+"
            r")"
        ),
    ),
)

_TRAILING_PATH_PUNCTUATION = ".,;!?)]}，。；！？）》】」』"
_TEXT_TRUNCATION_MARKER = "…[truncated]"


def _trim_candidate(kind: str, text: str, start: int, end: int) -> tuple[int, int]:
    if kind in {"url", "local_path"}:
        while end > start and text[end - 1] in _TRAILING_PATH_PUNCTUATION:
            end -= 1
    return start, end


def redact_remote_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Redact common direct identifiers before text leaves the machine.

    Findings contain only type, span and replacement metadata; the sensitive
    source value is intentionally never copied into the returned audit list.
    Overlapping matches are resolved by occurrence, then by the declared
    pattern priority (URL before path, for example).
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    candidates: list[tuple[int, int, int, str]] = []
    for priority, (kind, pattern) in enumerate(_PATTERNS):
        for match in pattern.finditer(text):
            start, end = _trim_candidate(kind, text, match.start(), match.end())
            if end > start:
                candidates.append((start, end, priority, kind))

    candidates.sort(key=lambda row: (row[0], row[2], -(row[1] - row[0])))
    accepted: list[tuple[int, int, str]] = []
    occupied_until = -1
    for start, end, _priority, kind in candidates:
        if start < occupied_until:
            continue
        accepted.append((start, end, kind))
        occupied_until = end

    if not accepted:
        return text, []

    parts: list[str] = []
    findings: list[dict[str, Any]] = []
    cursor = 0
    for start, end, kind in accepted:
        replacement = _REPLACEMENTS[kind]
        parts.extend((text[cursor:start], replacement))
        findings.append(
            {
                "kind": kind,
                "start": start,
                "end": end,
                "replacement": replacement,
            }
        )
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), findings


def _as_nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            if isinstance(item, bool) and not item:
                continue
            candidate = _as_nonempty_string(key)
            if candidate:
                result.append(candidate)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            if isinstance(item, Mapping):
                candidate = _as_nonempty_string(
                    item.get("knowledge_component")
                    or item.get("kc_id")
                    or item.get("id")
                    or item.get("name")
                )
            else:
                candidate = _as_nonempty_string(item)
            if candidate:
                result.append(candidate)
        return result
    return []


def _knowledge_components(value: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    keys = (
        "knowledge_component",
        "knowledge_components",
        "kc",
        "kc_id",
        "kc_ids",
        "target_knowledge_components",
    )
    for key in keys:
        result.extend(_string_values(value.get(key)))
    return list(dict.fromkeys(result))


def _current_focus_and_kcs(session: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    state = session.get("student_state", {})
    next_focus = state.get("next_focus", {}) if isinstance(state, Mapping) else {}
    action = session.get("current_action", {})
    primary = action.get("primary_skill", {}) if isinstance(action, Mapping) else {}
    goal = session.get("goal", {})

    focus = None
    for container in (next_focus, primary, action):
        if isinstance(container, Mapping):
            focus = _as_nonempty_string(
                container.get("dimension") or container.get("focus_dimension")
            )
            if focus:
                break

    kcs: list[str] = []
    for container in (next_focus, primary, action):
        if isinstance(container, Mapping):
            kcs.extend(_knowledge_components(container))
    if not kcs and isinstance(goal, Mapping):
        # Legacy sessions may not carry action-level KC tags.  Only then use
        # the complete goal as a conservative retrieval fallback.
        kcs.extend(_knowledge_components(goal))
    return focus, list(dict.fromkeys(kcs))


def _event_view(event: Mapping[str, Any], fallback_round: int) -> dict[str, Any]:
    action = event.get("action", {})
    primary = action.get("primary_skill", {}) if isinstance(action, Mapping) else {}
    teacher = action.get("teacher_action", {}) if isinstance(action, Mapping) else {}
    signal = event.get("structured_signal", {})
    if not isinstance(signal, Mapping):
        signal = {}
    focus = None
    if isinstance(primary, Mapping):
        focus = _as_nonempty_string(primary.get("focus_dimension"))
    if not focus:
        focus = _as_nonempty_string(event.get("focus_dimension"))

    kcs: list[str] = []
    for container in (event, action, primary):
        if isinstance(container, Mapping):
            kcs.extend(_knowledge_components(container))

    return {
        "round": event.get("round", fallback_round),
        "focus_dimension": focus,
        "knowledge_components": list(dict.fromkeys(kcs)),
        "skill_id": (
            _as_nonempty_string(primary.get("skill_id"))
            if isinstance(primary, Mapping)
            else None
        ),
        "teacher_message": (
            str(teacher.get("message", "")) if isinstance(teacher, Mapping) else ""
        ),
        "learner_response": str(event.get("learner_response", "")),
        "signal": _as_nonempty_string(signal.get("label")),
        "signal_confidence": signal.get("confidence"),
        "signal_source": _as_nonempty_string(signal.get("source")),
    }


def _redact_value(value: str, counts: Counter[str]) -> str:
    redacted, findings = redact_remote_text(value)
    counts.update(item["kind"] for item in findings)
    return redacted


def _redacted_event(event: dict[str, Any], counts: Counter[str]) -> dict[str, Any]:
    result = deepcopy(event)
    for field in (
        "focus_dimension",
        "skill_id",
        "teacher_message",
        "learner_response",
        "signal",
        "signal_source",
    ):
        value = result.get(field)
        if isinstance(value, str):
            result[field] = _redact_value(value, counts)
    result["knowledge_components"] = [
        _redact_value(str(item), counts)
        for item in result.get("knowledge_components", [])
    ]
    return result


def _history_summary(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rounds = [item["round"] for item in events if isinstance(item.get("round"), int)]
    signals = Counter(item.get("signal") or "not_recorded" for item in events)
    focuses = Counter(item.get("focus_dimension") or "not_recorded" for item in events)
    skills = Counter(item.get("skill_id") or "not_recorded" for item in events)
    sources = Counter(item.get("signal_source") or "not_recorded" for item in events)
    return {
        "turn_count": len(events),
        "round_range": [min(rounds), max(rounds)] if rounds else None,
        "signal_counts": dict(sorted(signals.items())),
        "focus_counts": dict(sorted(focuses.items())),
        "skill_counts": dict(sorted(skills.items())),
        "signal_source_counts": dict(sorted(sources.items())),
    }


def _serialized_length(value: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    if limit <= len(_TEXT_TRUNCATION_MARKER):
        return _TEXT_TRUNCATION_MARKER[:limit], True
    return value[: limit - len(_TEXT_TRUNCATION_MARKER)] + _TEXT_TRUNCATION_MARKER, True


def _rebuild_context(
    *,
    learner_response: str,
    selected: Sequence[dict[str, Any]],
    omitted: Sequence[dict[str, Any]],
    current_focus: str | None,
    current_kcs: Sequence[str],
    finding_counts: Counter[str],
    history_count: int,
    max_recent_turns: int,
    max_chars: int,
    text_limit: int,
    truncated: bool,
) -> dict[str, Any]:
    response, response_cut = _truncate_text(learner_response, text_limit * 2)
    turns: list[dict[str, Any]] = []
    any_cut = response_cut
    for item in selected:
        row = deepcopy(item)
        for field in ("teacher_message", "learner_response"):
            row[field], cut = _truncate_text(str(row.get(field, "")), text_limit)
            any_cut = any_cut or cut
        turns.append(row)
    return {
        "schema": CONTEXT_SCHEMA,
        "current_focus": current_focus,
        "current_knowledge_components": list(current_kcs),
        "current_learner_response": response,
        "recent_turns": turns,
        "earlier_summary": _history_summary(omitted),
        "privacy": {
            "remote_text_redacted": bool(finding_counts),
            "finding_counts": dict(sorted(finding_counts.items())),
            "original_values_retained": False,
        },
        "selection": {
            "priority": "same_knowledge_component_then_same_focus_then_recency",
            "history_turn_count": history_count,
            "selected_turn_count": len(turns),
            "max_recent_turns": max_recent_turns,
            "max_chars": max_chars,
            "truncated": truncated or any_cut,
        },
    }


def build_relevant_history(
    session: Mapping[str, Any],
    learner_response: str,
    max_recent_turns: int = 6,
    max_chars: int = 8000,
) -> dict[str, Any]:
    """Build bounded, redacted history for one remote learner assessment.

    Same-knowledge-component turns rank above same-focus turns, followed by
    recency.  At least the latest two turns are retained when capacity permits;
    all omitted turns are represented only by aggregate statistics.
    """

    if not isinstance(session, Mapping):
        raise TypeError("session must be a mapping")
    if not isinstance(learner_response, str):
        raise TypeError("learner_response must be a string")
    if isinstance(max_recent_turns, bool) or not isinstance(max_recent_turns, int):
        raise TypeError("max_recent_turns must be an integer")
    if not 0 <= max_recent_turns <= 100:
        raise ValueError("max_recent_turns must be in [0, 100]")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")
    if max_chars < 640:
        raise ValueError("max_chars must be at least 640")

    raw_history = session.get("history", [])
    if not isinstance(raw_history, list):
        raise ValueError("session.history must be a list")
    events: list[dict[str, Any]] = []
    for index, value in enumerate(raw_history):
        if not isinstance(value, Mapping):
            raise ValueError(f"session.history[{index}] must be an object")
        events.append(_event_view(value, index + 1))

    current_focus, current_kcs = _current_focus_and_kcs(session)
    normalized_kcs = {item.casefold() for item in current_kcs}

    latest_indices: set[int] = set()
    if max_recent_turns:
        latest_count = min(2, max_recent_turns, len(events))
        latest_indices.update(range(len(events) - latest_count, len(events)))

    def relevance(index: int) -> tuple[int, int, int]:
        event = events[index]
        event_kcs = {
            str(item).casefold() for item in event.get("knowledge_components", [])
        }
        kc_match = bool(normalized_kcs and normalized_kcs & event_kcs)
        focus_match = bool(
            current_focus
            and event.get("focus_dimension") == current_focus
        )
        return int(kc_match), int(focus_match), index

    available = [index for index in range(len(events)) if index not in latest_indices]
    available.sort(key=relevance, reverse=True)
    remaining = max(0, max_recent_turns - len(latest_indices))
    selected_indices = latest_indices | set(available[:remaining])
    selected = [events[index] for index in sorted(selected_indices)]
    omitted = [events[index] for index in range(len(events)) if index not in selected_indices]

    finding_counts: Counter[str] = Counter()
    redacted_response = _redact_value(learner_response, finding_counts)
    selected = [_redacted_event(item, finding_counts) for item in selected]
    omitted = [_redacted_event(item, finding_counts) for item in omitted]
    if current_focus:
        current_focus = _redact_value(current_focus, finding_counts)
    current_kcs = [_redact_value(item, finding_counts) for item in current_kcs]

    text_limits = (1200, 800, 480, 280, 160, 80, 32, 0)
    context: dict[str, Any] | None = None
    for text_limit in text_limits:
        context = _rebuild_context(
            learner_response=redacted_response,
            selected=selected,
            omitted=omitted,
            current_focus=current_focus,
            current_kcs=current_kcs,
            finding_counts=finding_counts,
            history_count=len(events),
            max_recent_turns=max_recent_turns,
            max_chars=max_chars,
            text_limit=text_limit,
            truncated=text_limit < text_limits[0],
        )
        if _serialized_length(context) <= max_chars:
            return context

    while selected:
        omitted.append(selected.pop(0))
        omitted.sort(key=lambda item: int(item.get("round", 0)))
        context = _rebuild_context(
            learner_response="",
            selected=selected,
            omitted=omitted,
            current_focus=current_focus,
            current_kcs=[],
            finding_counts=finding_counts,
            history_count=len(events),
            max_recent_turns=max_recent_turns,
            max_chars=max_chars,
            text_limit=0,
            truncated=True,
        )
        if _serialized_length(context) <= max_chars:
            return context

    minimal = {
        "schema": CONTEXT_SCHEMA,
        "current_focus": current_focus,
        "current_knowledge_components": [],
        "current_learner_response": "",
        "recent_turns": [],
        "earlier_summary": {
            "turn_count": len(events),
            "signal_counts": _history_summary(events)["signal_counts"],
        },
        "privacy": {
            "remote_text_redacted": bool(finding_counts),
            "finding_counts": dict(sorted(finding_counts.items())),
            "original_values_retained": False,
        },
        "selection": {
            "priority": "same_knowledge_component_then_same_focus_then_recency",
            "history_turn_count": len(events),
            "selected_turn_count": 0,
            "max_recent_turns": max_recent_turns,
            "max_chars": max_chars,
            "truncated": True,
        },
    }
    if _serialized_length(minimal) > max_chars:
        raise ValueError("max_chars is too small for the minimum context envelope")
    return minimal


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if not text or limit <= 0:
        return ""
    return _truncate_text(text, limit)[0]


def _redact_structure(value: Any, counts: Counter[str]) -> Any:
    """Deep-copy and redact every textual leaf of a JSON-compatible value."""

    if isinstance(value, str):
        return _redact_value(value, counts)
    if isinstance(value, Mapping):
        return {
            str(key): _redact_structure(item, counts)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_structure(item, counts) for item in value]
    return deepcopy(value)


def _safe_round(value: Any, fallback: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return fallback


def _evidence_record(
    evidence_id: str,
    *,
    source: str,
    field: str,
    round_number: int | None = None,
    excerpt: str = "",
    confidence: float | int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evidence_id": evidence_id,
        "source": source,
        "field": field,
        "round": round_number,
        "excerpt": excerpt,
    }
    if confidence is not None and not isinstance(confidence, bool):
        try:
            numeric = float(confidence)
        except (TypeError, ValueError):
            numeric = 0.0
        result["confidence"] = round(min(1.0, max(0.0, numeric)), 4)
    return result


def _teacher_profile_layer(
    profile: Mapping[str, Any],
    *,
    content_limit: int,
    prior_limit: int,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    initial_mastery = profile.get("initial_mastery", {})
    if not isinstance(initial_mastery, Mapping):
        initial_mastery = {}
    known = profile.get("known_misconceptions", [])
    if not isinstance(known, list):
        known = []
    known_view: list[dict[str, Any]] = []
    for index, item in enumerate(known[:8]):
        if not isinstance(item, Mapping):
            continue
        evidence_id = f"teacher_profile:known_misconception:{index}"
        description = _bounded_text(item.get("description", ""), min(300, content_limit))
        known_view.append(
            {
                "tag": _bounded_text(item.get("tag", ""), 120),
                "description": description,
                "confidence": item.get("confidence"),
                "evidence_refs": [evidence_id],
            }
        )
        evidence.append(
            _evidence_record(
                evidence_id,
                source="teacher_provided_student_profile",
                field=f"known_misconceptions[{index}]",
                excerpt=description,
                confidence=item.get("confidence"),
            )
        )

    prior = profile.get("conversation_history", [])
    if not isinstance(prior, list):
        prior = []
    retained_start = max(0, len(prior) - prior_limit)
    prior_view: list[dict[str, Any]] = []
    for index, item in enumerate(prior[retained_start:], retained_start):
        if not isinstance(item, Mapping):
            continue
        evidence_id = f"teacher_profile:conversation_history:{index}"
        excerpt = _bounded_text(item.get("response", ""), min(400, content_limit))
        prior_view.append(
            {
                "index": index,
                "response_excerpt": excerpt,
                "signal": item.get("signal"),
                "focus_dimension": item.get("focus_dimension"),
                "evidence_refs": [evidence_id],
            }
        )
        evidence.append(
            _evidence_record(
                evidence_id,
                source="teacher_provided_student_profile",
                field=f"conversation_history[{index}]",
                excerpt=excerpt,
            )
        )

    for dimension, value in initial_mastery.items():
        evidence.append(
            _evidence_record(
                f"teacher_profile:initial_mastery:{dimension}",
                source="teacher_provided_student_profile",
                field=f"initial_mastery.{dimension}",
            )
        )
    preferences = profile.get("preferences", [])
    accessibility = profile.get("accessibility_needs", [])
    return {
        "learner_level": _bounded_text(profile.get("learner_level", ""), 120),
        "preferences": [
            _bounded_text(item, min(200, content_limit))
            for item in preferences[:8]
            if _bounded_text(item, min(200, content_limit))
        ]
        if isinstance(preferences, list)
        else [],
        "accessibility_needs": [
            _bounded_text(item, min(200, content_limit))
            for item in accessibility[:8]
            if _bounded_text(item, min(200, content_limit))
        ]
        if isinstance(accessibility, list)
        else [],
        "initial_mastery": dict(initial_mastery),
        "declared_known_misconceptions": known_view,
        "provided_prior_context": {
            "total_turn_count": len(prior),
            "retained_turn_count": len(prior_view),
            "turns": prior_view,
        },
        "source": "teacher_input_normalized_locally",
        "mutable_by_model": False,
    }


def _current_plan_layer(
    session: Mapping[str, Any], *, content_limit: int
) -> dict[str, Any]:
    plan = session.get("goal_plan", {})
    if not isinstance(plan, Mapping):
        plan = {}
    steps = plan.get("intermediate_objectives", [])
    if not isinstance(steps, list):
        steps = []
    active_id = plan.get("active_step")
    active = next(
        (
            item
            for item in steps
            if isinstance(item, Mapping) and item.get("step_id") == active_id
        ),
        None,
    )
    active_view: dict[str, Any] | None = None
    if isinstance(active, Mapping):
        verification = active.get("verification", {})
        if not isinstance(verification, Mapping):
            verification = {}
        active_view = {
            "step_id": active.get("step_id"),
            "dimension": active.get("dimension"),
            "objective": _bounded_text(active.get("objective", ""), content_limit),
            "success_criterion": _bounded_text(
                verification.get("success_criterion", ""), content_limit
            ),
            "mastery_threshold": verification.get("mastery_threshold"),
            "status": active.get("status"),
        }
    action = session.get("current_action", {})
    if not isinstance(action, Mapping):
        action = {}
    primary = action.get("primary_skill", {})
    teacher_action = action.get("teacher_action", {})
    supporting = action.get("supporting_skills", [])
    if not isinstance(primary, Mapping):
        primary = {}
    if not isinstance(teacher_action, Mapping):
        teacher_action = {}
    if not isinstance(supporting, list):
        supporting = []
    return {
        "plan_schema": plan.get("schema"),
        "plan_status": plan.get("status"),
        "progress": deepcopy(plan.get("progress", {})),
        "active_step": active_view,
        "step_statuses": [
            {
                "step_id": item.get("step_id"),
                "dimension": item.get("dimension"),
                "status": item.get("status"),
            }
            for item in steps
            if isinstance(item, Mapping)
        ],
        "current_action": {
            "action_id": action.get("action_id"),
            "primary_skill_id": primary.get("skill_id"),
            "primary_skill_name": primary.get("name"),
            "focus_dimension": primary.get("focus_dimension"),
            "knowledge_components": deepcopy(
                primary.get(
                    "knowledge_components",
                    action.get("knowledge_components", []),
                )
            ),
            "supporting_skill_ids": [
                item.get("skill_id")
                for item in supporting
                if isinstance(item, Mapping) and item.get("skill_id")
            ],
            "selection_reason": _bounded_text(
                action.get("selection_reason", ""), content_limit
            ),
            "teacher_message": _bounded_text(
                teacher_action.get("message", ""), content_limit
            ),
            "expected_signal": _bounded_text(
                teacher_action.get("expected_signal", ""), content_limit
            ),
        },
        "source": "deterministic_session_plan_and_last_validated_action",
    }


def _latest_unresolved_events(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return unresolved observations without inventing a semantic diagnosis."""

    unresolved: list[dict[str, Any]] = []
    for event in events:
        focus = str(event.get("focus_dimension") or "unspecified")
        event_kcs = {
            str(item).casefold()
            for item in event.get("knowledge_components", [])
            if str(item).strip()
        }

        def same_issue(candidate: Mapping[str, Any]) -> bool:
            candidate_focus = str(
                candidate.get("focus_dimension") or "unspecified"
            )
            candidate_kcs = {
                str(item).casefold()
                for item in candidate.get("knowledge_components", [])
                if str(item).strip()
            }
            return candidate_focus == focus or bool(event_kcs & candidate_kcs)

        signal = event.get("signal")
        if signal == "correct":
            unresolved = [item for item in unresolved if not same_issue(item)]
        elif signal in {"partial", "misconception", "confused", "no_response"}:
            unresolved = [item for item in unresolved if not same_issue(item)]
            unresolved.append(event)
    return sorted(
        unresolved,
        key=lambda item: _safe_round(item.get("round"), 0),
    )[-4:]


def _knowledge_state_layer(
    session: Mapping[str, Any],
    *,
    events: Sequence[dict[str, Any]],
    content_limit: int,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    state = session.get("student_state", {})
    if not isinstance(state, Mapping):
        state = {}
    mastery = state.get("knowledge_mastery", {})
    if not isinstance(mastery, Mapping):
        mastery = {}
    thresholds = session.get("goal", {}).get("success_thresholds", {})
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    mastery_view: list[dict[str, Any]] = []
    for dimension, value in mastery.items():
        matching = [
            item for item in events if item.get("focus_dimension") == dimension
        ]
        if matching:
            latest_match = matching[-1]
            latest_round = _safe_round(latest_match.get("round"), 0)
            ref = f"session_history:r{latest_round}:structured_signal"
            evidence.append(
                _evidence_record(
                    ref,
                    source=latest_match.get("signal_source") or "not_recorded",
                    field="structured_signal",
                    round_number=latest_round,
                    confidence=latest_match.get("signal_confidence"),
                )
            )
        else:
            ref = f"teacher_profile:initial_mastery:{dimension}"
        mastery_view.append(
            {
                "dimension": dimension,
                "value": value,
                "success_threshold": thresholds.get(dimension),
                "evidence_refs": [ref],
                "interpretation": "deterministic_state_estimate_not_ground_truth",
            }
        )

    misconceptions_raw = state.get("misconceptions", [])
    if not isinstance(misconceptions_raw, list):
        misconceptions_raw = []
    misconceptions: list[dict[str, Any]] = []
    for index, item in enumerate(misconceptions_raw[:12]):
        if not isinstance(item, Mapping):
            continue
        last_round = _safe_round(item.get("last_observed_round"), 0)
        ref = (
            f"session_history:r{last_round}:structured_signal"
            if last_round > 0
            else f"teacher_profile:known_misconception:{index}"
        )
        if last_round > 0:
            matching_event = next(
                (
                    event
                    for event in reversed(events)
                    if _safe_round(event.get("round"), 0) == last_round
                ),
                {},
            )
            evidence.append(
                _evidence_record(
                    ref,
                    source=matching_event.get("signal_source") or "not_recorded",
                    field="structured_signal",
                    round_number=last_round,
                    confidence=matching_event.get("signal_confidence"),
                )
            )
        misconceptions.append(
            {
                "tag": _bounded_text(item.get("tag", ""), 120),
                "description": _bounded_text(
                    item.get("description", ""), min(300, content_limit)
                ),
                "confidence": item.get("confidence"),
                "status": item.get("status"),
                "first_observed_round": item.get("first_observed_round"),
                "last_observed_round": item.get("last_observed_round"),
                "resolved_round": item.get("resolved_round"),
                "evidence_refs": [ref],
            }
        )

    unresolved: list[dict[str, Any]] = []
    for event in _latest_unresolved_events(events):
        round_number = _safe_round(event.get("round"), 0)
        response_id = f"session_history:r{round_number}:learner_response"
        signal_id = f"session_history:r{round_number}:structured_signal"
        excerpt = _bounded_text(
            event.get("learner_response", ""), min(240, content_limit)
        )
        unresolved.append(
            {
                "issue_kind": "unresolved_observed_response",
                "focus_dimension": event.get("focus_dimension"),
                "knowledge_components": deepcopy(
                    event.get("knowledge_components", [])
                ),
                "observed_signal": event.get("signal"),
                "observation_source": event.get("signal_source") or "not_recorded",
                "learner_response_excerpt": excerpt,
                "evidence_refs": [response_id, signal_id],
            }
        )
        evidence.extend(
            [
                _evidence_record(
                    response_id,
                    source="session_history",
                    field="learner_response",
                    round_number=round_number,
                    excerpt=excerpt,
                ),
                _evidence_record(
                    signal_id,
                    source=event.get("signal_source") or "not_recorded",
                    field="structured_signal",
                    round_number=round_number,
                    confidence=event.get("signal_confidence"),
                ),
            ]
        )
    return {
        "concept_mastery": mastery_view,
        "misconceptions": misconceptions,
        "unresolved_issues": unresolved,
        "current_understanding_signal": deepcopy(state.get("understanding_signal", {})),
        "next_focus": deepcopy(state.get("next_focus", {})),
        "assessment_evidence": deepcopy(state.get("assessment_evidence", {})),
        "source": "deterministic_state_machine_over_labeled_observations",
    }


def _candidate_memory_layer(
    profile: Mapping[str, Any],
    *,
    observation_limit: int,
    content_limit: int,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    observations = profile.get("adaptive_observations", [])
    if not isinstance(observations, list):
        observations = []
    retained = observations[-observation_limit:] if observation_limit else []
    views: list[dict[str, Any]] = []
    for item in retained:
        if not isinstance(item, Mapping):
            continue
        round_number = _safe_round(item.get("round"), 0)
        candidate = item.get("candidate", {})
        evidence_value = item.get("evidence", {})
        if not isinstance(candidate, Mapping) or not isinstance(evidence_value, Mapping):
            continue
        evidence_id = f"adaptive_profile:r{round_number}:validated_diagnosis"
        excerpt = _bounded_text(
            evidence_value.get("excerpt", ""), min(240, content_limit)
        )
        views.append(
            {
                "round": round_number,
                "status": item.get("status"),
                "candidate": deepcopy(dict(candidate)),
                "confidence": evidence_value.get("confidence"),
                "needs_human_review": evidence_value.get("needs_human_review"),
                "evidence_refs": [evidence_id],
            }
        )
        evidence.append(
            _evidence_record(
                evidence_id,
                source=str(item.get("source") or "candidate_profile"),
                field="adaptive_observation.evidence",
                round_number=round_number,
                excerpt=excerpt,
                confidence=evidence_value.get("confidence"),
            )
        )
    return {
        "status": "candidate_unconfirmed",
        "summary": deepcopy(profile.get("adaptive_summary", {})),
        "retained_for_context": len(views),
        "observations": views,
        "may_override_teacher_profile": False,
        "eligible_sources": ["deepseek_v4_flash_validated_diagnosis"],
    }


def _deduplicate_evidence(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in records:
        evidence_id = str(item.get("evidence_id", ""))
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        result.append(item)
    return result


def _set_serialized_size(context: dict[str, Any]) -> int:
    """Reach the stable character count after writing the count itself."""

    for _ in range(8):
        measured = _serialized_length(context)
        if context["budget"].get("serialized_chars") == measured:
            return measured
        context["budget"]["serialized_chars"] = measured
    return _serialized_length(context)


def _layered_context_candidate(
    session: Mapping[str, Any],
    learner_response: str | None,
    *,
    max_chars: int,
    max_recent_turns: int,
    prior_limit: int,
    adaptive_limit: int,
    content_limit: int,
    evidence_excerpt_limit: int,
    degraded: bool,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    history_budget = max(640, min(8_000, max_chars // 2))
    relevant = build_relevant_history(
        session,
        learner_response or "",
        max_recent_turns=max_recent_turns,
        max_chars=history_budget,
    )
    raw_history = session.get("history", [])
    events = [
        _event_view(item, index + 1)
        for index, item in enumerate(raw_history)
        if isinstance(item, Mapping)
    ] if isinstance(raw_history, list) else []

    recent_turns: list[dict[str, Any]] = []
    for item in relevant["recent_turns"]:
        round_number = _safe_round(item.get("round"), 0)
        teacher_id = f"session_history:r{round_number}:teacher_action"
        response_id = f"session_history:r{round_number}:learner_response"
        signal_id = f"session_history:r{round_number}:structured_signal"
        row = deepcopy(item)
        row["evidence_refs"] = [teacher_id, response_id, signal_id]
        recent_turns.append(row)
        evidence.extend(
            [
                _evidence_record(
                    teacher_id,
                    source="session_history",
                    field="action.teacher_action.message",
                    round_number=round_number,
                    excerpt=_bounded_text(
                        item.get("teacher_message", ""), evidence_excerpt_limit
                    ),
                ),
                _evidence_record(
                    response_id,
                    source="session_history",
                    field="learner_response",
                    round_number=round_number,
                    excerpt=_bounded_text(
                        item.get("learner_response", ""), evidence_excerpt_limit
                    ),
                ),
                _evidence_record(
                    signal_id,
                    source=item.get("signal_source") or "not_recorded",
                    field="structured_signal",
                    round_number=round_number,
                    confidence=item.get("signal_confidence"),
                ),
            ]
        )

    current_response_id: str | None = None
    if learner_response is not None:
        current_response_id = f"current_response:r{int(session.get('round', 0)) + 1}"
        evidence.append(
            _evidence_record(
                current_response_id,
                source="current_learner_input",
                field="learner_response",
                round_number=int(session.get("round", 0)) + 1,
                # The actual current answer lives in working_memory exactly
                # once.  The ledger is a pointer, not a second text copy.
                excerpt="",
            )
        )

    selected_rounds = {
        _safe_round(item.get("round"), 0) for item in recent_turns
    }
    checkpoint_by_focus: dict[str, dict[str, Any]] = {}
    for event in events:
        round_number = _safe_round(event.get("round"), 0)
        if round_number in selected_rounds:
            continue
        focus = str(event.get("focus_dimension") or "unspecified")
        checkpoint_by_focus[focus] = event
    focus_checkpoints: list[dict[str, Any]] = []
    checkpoint_items = sorted(
        checkpoint_by_focus.items(),
        key=lambda item: _safe_round(item[1].get("round"), 0),
    )[-4:]
    for focus, event in checkpoint_items:
        round_number = _safe_round(event.get("round"), 0)
        response_id = f"session_history:r{round_number}:learner_response"
        signal_id = f"session_history:r{round_number}:structured_signal"
        excerpt = _bounded_text(
            event.get("learner_response", ""), min(160, content_limit)
        )
        focus_checkpoints.append(
            {
                "focus_dimension": focus,
                "latest_omitted_round": round_number,
                "signal": event.get("signal"),
                "signal_source": event.get("signal_source") or "not_recorded",
                "skill_id": event.get("skill_id"),
                "knowledge_components": deepcopy(
                    event.get("knowledge_components", [])
                ),
                "learner_response_excerpt": excerpt,
                "evidence_refs": [response_id, signal_id],
            }
        )
        evidence.extend(
            [
                _evidence_record(
                    response_id,
                    source="session_history",
                    field="learner_response",
                    round_number=round_number,
                    excerpt=_bounded_text(excerpt, evidence_excerpt_limit),
                ),
                _evidence_record(
                    signal_id,
                    source=event.get("signal_source") or "not_recorded",
                    field="structured_signal",
                    round_number=round_number,
                    confidence=event.get("signal_confidence"),
                ),
            ]
        )

    profile = session.get("student_profile", {})
    if not isinstance(profile, Mapping):
        profile = {}
    goal = session.get("goal", {})
    if not isinstance(goal, Mapping):
        goal = {}
    fixed_context = {
        "teaching_goal": {
            "concept": _bounded_text(goal.get("concept", ""), min(240, content_limit)),
            "objective": _bounded_text(goal.get("objective", ""), content_limit),
            "knowledge_components": [
                _bounded_text(item, min(240, content_limit))
                for item in goal.get("knowledge_components", [])[:12]
                if _bounded_text(item, min(240, content_limit))
            ]
            if isinstance(goal.get("knowledge_components", []), list)
            else [],
            "success_thresholds": deepcopy(goal.get("success_thresholds", {})),
            "max_rounds": goal.get("max_rounds"),
            "materials": {
                str(key): _bounded_text(value, content_limit)
                for key, value in list(goal.get("materials", {}).items())[:6]
            }
            if isinstance(goal.get("materials", {}), Mapping)
            else {},
            "source": "teacher_input_normalized_locally",
            "mutable_by_model": False,
        },
        "teacher_provided_student_profile": _teacher_profile_layer(
            profile,
            content_limit=content_limit,
            prior_limit=prior_limit,
            evidence=evidence,
        ),
    }
    knowledge_state = _knowledge_state_layer(
        session,
        events=events,
        content_limit=content_limit,
        evidence=evidence,
    )
    candidate_memory = _candidate_memory_layer(
        profile,
        observation_limit=adaptive_limit,
        content_limit=content_limit,
        evidence=evidence,
    )
    context: dict[str, Any] = {
        "schema": LAYERED_CONTEXT_SCHEMA,
        "snapshot": {
            "kind": "bounded_outbound_model_request_context",
            "session_round_before_request": int(session.get("round", 0)),
            "operation": (
                "initial_action" if learner_response is None else "assess_and_act"
            ),
            "current_response_present": learner_response is not None,
        },
        "fixed_context": fixed_context,
        "current_plan": _current_plan_layer(session, content_limit=content_limit),
        "working_memory": {
            "current_focus": relevant.get("current_focus"),
            "current_knowledge_components": deepcopy(
                relevant.get("current_knowledge_components", [])
            ),
            "current_learner_response": relevant.get("current_learner_response", ""),
            "current_response_evidence_id": current_response_id,
            "recent_turns": recent_turns,
        },
        "semantic_summary": {
            **deepcopy(relevant.get("earlier_summary", {})),
            "focus_checkpoints": focus_checkpoints,
            "compression_method": (
                "deterministic_aggregate_and_extractive_checkpoints_"
                "no_model_generation"
            ),
            "source": "omitted_local_session_history",
            "narrative_inference_added": False,
        },
        "knowledge_state": knowledge_state,
        "candidate_long_term_memory": candidate_memory,
        "evidence_ledger": _deduplicate_evidence(evidence),
        "retrieval": {
            **deepcopy(relevant.get("selection", {})),
            "selected_rounds": [item.get("round") for item in recent_turns],
            "fixed_context_always_included": True,
            "current_plan_always_included": True,
            "semantic_summary_covers_omitted_turns": True,
            "candidate_memory_policy": "latest_validated_candidates_only",
        },
        "privacy": {
            **deepcopy(relevant.get("privacy", {})),
            "raw_identity_fields_sent": False,
            "original_values_retained_in_context": False,
        },
        "budget": {
            "policy": "hard_serialized_character_cap",
            "max_chars": max_chars,
            "serialized_chars": 0,
            "max_recent_turns": max_recent_turns,
            "retained_recent_turns": len(recent_turns),
            "retained_teacher_prior_turns": fixed_context[
                "teacher_provided_student_profile"
            ]["provided_prior_context"]["retained_turn_count"],
            "retained_candidate_observations": candidate_memory[
                "retained_for_context"
            ],
            "truncated": bool(degraded or relevant["selection"]["truncated"]),
        },
        "claim_boundary": {
            "model_generated_history_summary": False,
            "candidate_long_term_memory_is_confirmed": False,
            "fallback_observations_are_labeled_by_actual_source": True,
            "context_is_evidence_not_learner_ground_truth": True,
            "model_may_not_mutate_fixed_context": True,
        },
    }
    counts: Counter[str] = Counter()
    redacted = _redact_structure(context, counts)
    redacted["privacy"]["remote_text_redacted"] = bool(
        counts or relevant.get("privacy", {}).get("remote_text_redacted")
    )
    combined_counts = Counter(relevant.get("privacy", {}).get("finding_counts", {}))
    combined_counts.update(counts)
    redacted["privacy"]["finding_counts"] = dict(sorted(combined_counts.items()))
    _set_serialized_size(redacted)
    return redacted


def build_minimal_layered_context(
    session: Mapping[str, Any],
    learner_response: str | None,
    *,
    max_chars: int = MINIMUM_LAYERED_CONTEXT_CHARS,
) -> dict[str, Any]:
    """Build the smallest schema-valid context while retaining current input.

    This path is deterministic and intentionally extractive.  It is used as
    the final budget degradation profile and as the local safety envelope when
    construction of the richer retrieval snapshot fails unexpectedly.
    """

    if not isinstance(session, Mapping):
        raise TypeError("session must be a mapping")
    if learner_response is not None and not isinstance(learner_response, str):
        raise TypeError("learner_response must be a string or None")
    if (
        isinstance(max_chars, bool)
        or not isinstance(max_chars, int)
        or max_chars < MINIMUM_LAYERED_CONTEXT_CHARS
    ):
        raise ValueError(
            f"max_chars must be at least {MINIMUM_LAYERED_CONTEXT_CHARS}"
        )

    goal = session.get("goal", {})
    if not isinstance(goal, Mapping):
        goal = {}
    profile = session.get("student_profile", {})
    if not isinstance(profile, Mapping):
        profile = {}
    state = session.get("student_state", {})
    if not isinstance(state, Mapping):
        state = {}
    plan = session.get("goal_plan", {})
    if not isinstance(plan, Mapping):
        plan = {}
    action = session.get("current_action", {})
    if not isinstance(action, Mapping):
        action = {}
    primary = action.get("primary_skill", {})
    if not isinstance(primary, Mapping):
        primary = {}
    mastery = state.get("knowledge_mastery", {})
    if not isinstance(mastery, Mapping):
        mastery = {}
    thresholds = goal.get("success_thresholds", {})
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    history = session.get("history", [])
    history_count = len(history) if isinstance(history, list) else 0

    raw_misconceptions = state.get("misconceptions", [])
    if not isinstance(raw_misconceptions, list):
        raw_misconceptions = []
    retained_misconception = next(
        (
            item
            for item in reversed(raw_misconceptions)
            if isinstance(item, Mapping) and item.get("status") == "active"
        ),
        None,
    )
    adaptive = profile.get("adaptive_observations", [])
    if not isinstance(adaptive, list):
        adaptive = []
    adaptive_summary = profile.get("adaptive_summary", {})
    if not isinstance(adaptive_summary, Mapping):
        adaptive_summary = {}

    for response_limit in (512, 256, 128, 64):
        evidence: list[dict[str, Any]] = []
        current_response_id: str | None = None
        current_response = _bounded_text(learner_response or "", response_limit)
        if learner_response is not None:
            current_response_id = (
                f"current_response:r{int(session.get('round', 0)) + 1}"
            )
            evidence.append(
                _evidence_record(
                    current_response_id,
                    source="current_learner_input",
                    field="learner_response",
                    round_number=int(session.get("round", 0)) + 1,
                    excerpt="",
                )
            )

        mastery_view: list[dict[str, Any]] = []
        for dimension, value in list(mastery.items())[:4]:
            evidence_id = f"student_state:knowledge_mastery:{dimension}"
            mastery_view.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "success_threshold": thresholds.get(dimension),
                    "evidence_refs": [evidence_id],
                }
            )
            evidence.append(
                _evidence_record(
                    evidence_id,
                    source="deterministic_session_state",
                    field=f"student_state.knowledge_mastery.{dimension}",
                )
            )

        misconception_view: list[dict[str, Any]] = []
        unresolved_view: list[dict[str, Any]] = []
        if isinstance(retained_misconception, Mapping):
            evidence_id = "student_state:latest_active_misconception"
            misconception_view.append(
                {
                    "tag": _bounded_text(retained_misconception.get("tag", ""), 64),
                    "description": _bounded_text(
                        retained_misconception.get("description", ""), 64
                    ),
                    "confidence": retained_misconception.get("confidence"),
                    "status": "active",
                    "evidence_refs": [evidence_id],
                }
            )
            unresolved_view.append(
                {
                    "issue_kind": "active_misconception",
                    "focus_dimension": "conceptual",
                    "observed_signal": "misconception",
                    "observation_source": "deterministic_session_state",
                    "evidence_refs": [evidence_id],
                }
            )
            evidence.append(
                _evidence_record(
                    evidence_id,
                    source="deterministic_session_state",
                    field="student_state.misconceptions.latest_active",
                    round_number=_safe_round(
                        retained_misconception.get("last_observed_round"), 0
                    ),
                )
            )

        signal = state.get("understanding_signal", {})
        if not isinstance(signal, Mapping):
            signal = {}
        assessment = state.get("assessment_evidence", {})
        if not isinstance(assessment, Mapping):
            assessment = {}
        next_focus = state.get("next_focus", {})
        if not isinstance(next_focus, Mapping):
            next_focus = {}
        provided_history = profile.get("conversation_history", [])
        known = profile.get("known_misconceptions", [])
        initial_mastery = profile.get("initial_mastery", {})
        preferences = profile.get("preferences", [])
        accessibility = profile.get("accessibility_needs", [])
        components = goal.get("knowledge_components", [])
        if not isinstance(components, list):
            components = []
        safe_components = [
            _bounded_text(item, 24)
            for item in components[:4]
            if _bounded_text(item, 24)
        ]
        current_components = safe_components or [
            _bounded_text(goal.get("concept", ""), 32)
        ]

        context: dict[str, Any] = {
            "schema": LAYERED_CONTEXT_SCHEMA,
            "snapshot": {
                "kind": "bounded_outbound_model_request_context",
                "session_round_before_request": int(session.get("round", 0)),
                "operation": (
                    "initial_action" if learner_response is None else "assess_and_act"
                ),
                "current_response_present": learner_response is not None,
            },
            "fixed_context": {
                "teaching_goal": {
                    "concept": _bounded_text(goal.get("concept", ""), 96),
                        "objective": _bounded_text(goal.get("objective", ""), 64),
                        "knowledge_components": safe_components,
                        "total_knowledge_component_count": len(components),
                    "success_thresholds": dict(thresholds),
                    "max_rounds": goal.get("max_rounds"),
                    "materials": {},
                    "source": "teacher_input_normalized_locally",
                    "mutable_by_model": False,
                },
                "teacher_provided_student_profile": {
                    "learner_level": _bounded_text(
                        profile.get("learner_level", ""), 64
                    ),
                        "preferences": [],
                        "total_preference_count": (
                            len(preferences) if isinstance(preferences, list) else 0
                        ),
                        "accessibility_needs": [],
                        "total_accessibility_need_count": (
                            len(accessibility)
                            if isinstance(accessibility, list)
                            else 0
                        ),
                    "initial_mastery": (
                        dict(initial_mastery)
                        if isinstance(initial_mastery, Mapping)
                        else {}
                    ),
                    "declared_known_misconceptions": [],
                    "total_declared_known_misconception_count": (
                        len(known) if isinstance(known, list) else 0
                    ),
                    "provided_prior_context": {
                        "total_turn_count": (
                            len(provided_history)
                            if isinstance(provided_history, list)
                            else 0
                        ),
                        "retained_turn_count": 0,
                        "turns": [],
                    },
                    "source": "teacher_input_normalized_locally",
                    "mutable_by_model": False,
                },
            },
            "current_plan": {
                "plan_schema": plan.get("schema"),
                "plan_status": plan.get("status"),
                    "progress": {},
                "active_step": plan.get("active_step"),
                "step_statuses": [],
                "current_action": {
                    "action_id": action.get("action_id"),
                    "primary_skill_id": primary.get("skill_id"),
                    "focus_dimension": primary.get("focus_dimension"),
                    "knowledge_components": current_components,
                        "teacher_message": "",
                        "expected_signal": "",
                },
                "source": "deterministic_session_plan_and_last_validated_action",
            },
            "working_memory": {
                "current_focus": next_focus.get("dimension")
                or primary.get("focus_dimension"),
                "current_knowledge_components": current_components,
                "current_learner_response": current_response,
                "current_response_evidence_id": current_response_id,
                "recent_turns": [],
            },
            "semantic_summary": {
                "turn_count": history_count,
                "focus_checkpoints": [],
                "compression_method": (
                    "deterministic_aggregate_and_extractive_checkpoints_"
                    "no_model_generation"
                ),
                "source": "omitted_local_session_history",
                "narrative_inference_added": False,
            },
            "knowledge_state": {
                "concept_mastery": mastery_view,
                "misconceptions": misconception_view,
                "total_misconception_count": len(raw_misconceptions),
                "unresolved_issues": unresolved_view,
                "current_understanding_signal": {
                    "label": signal.get("label"),
                    "confidence": signal.get("confidence"),
                    "source": signal.get("source"),
                },
                "next_focus": {
                    "dimension": next_focus.get("dimension"),
                    "selected_skill_id": next_focus.get("selected_skill_id"),
                },
                "assessment_evidence": {
                    "source": assessment.get("source"),
                    "needs_human_review": assessment.get("needs_human_review"),
                },
                "source": "deterministic_state_machine_over_labeled_observations",
            },
            "candidate_long_term_memory": {
                "status": "candidate_unconfirmed",
                "summary": {
                    "total_observation_count": adaptive_summary.get(
                        "total_observation_count", len(adaptive)
                    ),
                    "retained_observation_count": len(adaptive),
                    "needs_human_review": adaptive_summary.get(
                        "needs_human_review", False
                    ),
                },
                "retained_for_context": 0,
                "observations": [],
                "may_override_teacher_profile": False,
                "eligible_sources": [
                    "deepseek_v4_flash_validated_diagnosis"
                ],
            },
            "evidence_ledger": _deduplicate_evidence(evidence),
            "retrieval": {
                "priority": "minimum_safety_envelope_after_budget_degradation",
                "history_turn_count": history_count,
                "selected_turn_count": 0,
                "max_recent_turns": 0,
                "max_chars": max_chars,
                "truncated": True,
                "selected_rounds": [],
                "fixed_context_always_included": True,
                "current_plan_always_included": True,
                "semantic_summary_covers_omitted_turns": True,
                "candidate_memory_policy": "latest_validated_candidates_only",
            },
            "privacy": {
                "remote_text_redacted": False,
                "finding_counts": {},
                "original_values_retained": False,
                "raw_identity_fields_sent": False,
                "original_values_retained_in_context": False,
            },
            "budget": {
                "policy": "hard_serialized_character_cap",
                "max_chars": max_chars,
                "serialized_chars": 0,
                "max_recent_turns": 0,
                "retained_recent_turns": 0,
                "retained_teacher_prior_turns": 0,
                "retained_candidate_observations": 0,
                "truncated": True,
            },
            "claim_boundary": {
                "model_generated_history_summary": False,
                "candidate_long_term_memory_is_confirmed": False,
                "fallback_observations_are_labeled_by_actual_source": True,
                "context_is_evidence_not_learner_ground_truth": True,
                "model_may_not_mutate_fixed_context": True,
            },
        }
        counts: Counter[str] = Counter()
        redacted = _redact_structure(context, counts)
        redacted["privacy"]["remote_text_redacted"] = bool(counts)
        redacted["privacy"]["finding_counts"] = dict(sorted(counts.items()))
        _set_serialized_size(redacted)
        if _serialized_length(redacted) <= max_chars:
            validate_layered_context(redacted)
            return redacted
    raise ValueError("minimum layered context exceeds max_chars")


def build_layered_context(
    session: Mapping[str, Any],
    learner_response: str | None,
    *,
    max_chars: int = DEFAULT_LAYERED_CONTEXT_CHARS,
    max_recent_turns: int = 6,
) -> dict[str, Any]:
    """Build the single bounded, evidence-linked context sent to DeepSeek.

    The snapshot separates immutable teacher input, the current plan, recent
    working memory, deterministic older-history compression, explicit learner
    state, and unconfirmed long-term profile candidates.  No language model is
    called to create or repair memory.  When the full representation would
    exceed ``max_chars``, deterministic retention profiles are tried in order.
    """

    if not isinstance(session, Mapping):
        raise TypeError("session must be a mapping")
    if learner_response is not None and not isinstance(learner_response, str):
        raise TypeError("learner_response must be a string or None")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")
    if max_chars < MINIMUM_LAYERED_CONTEXT_CHARS:
        raise ValueError(
            f"max_chars must be at least {MINIMUM_LAYERED_CONTEXT_CHARS}"
        )
    if isinstance(max_recent_turns, bool) or not isinstance(max_recent_turns, int):
        raise TypeError("max_recent_turns must be an integer")
    if not 0 <= max_recent_turns <= 12:
        raise ValueError("max_recent_turns must be in [0, 12]")

    profiles = (
        (max_recent_turns, 3, 4, 800, 240),
        (min(max_recent_turns, 4), 2, 3, 480, 160),
        (min(max_recent_turns, 3), 1, 2, 300, 100),
        (min(max_recent_turns, 2), 0, 1, 180, 60),
        (min(max_recent_turns, 1), 0, 0, 100, 0),
        (0, 0, 0, 64, 0),
    )
    last: dict[str, Any] | None = None
    for index, (recent, prior, adaptive, content, excerpt) in enumerate(profiles):
        last = _layered_context_candidate(
            session,
            learner_response,
            max_chars=max_chars,
            max_recent_turns=recent,
            prior_limit=prior,
            adaptive_limit=adaptive,
            content_limit=content,
            evidence_excerpt_limit=excerpt,
            degraded=index > 0,
        )
        if _serialized_length(last) <= max_chars:
            validate_layered_context(last)
            return last
    return build_minimal_layered_context(
        session,
        learner_response,
        max_chars=max_chars,
    )


def _collect_evidence_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "evidence_refs" and isinstance(item, list):
                refs.extend(str(ref) for ref in item)
            elif key == "current_response_evidence_id" and isinstance(item, str):
                refs.append(item)
            else:
                refs.extend(_collect_evidence_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_collect_evidence_refs(item))
    return refs


def validate_layered_context(context: Mapping[str, Any]) -> None:
    """Fail closed on malformed budgets, provenance, or dangling evidence."""

    if not isinstance(context, Mapping) or context.get("schema") != LAYERED_CONTEXT_SCHEMA:
        raise ValueError(f"context schema must be {LAYERED_CONTEXT_SCHEMA}")
    required = {
        "snapshot",
        "fixed_context",
        "current_plan",
        "working_memory",
        "semantic_summary",
        "knowledge_state",
        "candidate_long_term_memory",
        "evidence_ledger",
        "retrieval",
        "privacy",
        "budget",
        "claim_boundary",
    }
    if not required <= set(context):
        raise ValueError("layered context sections are incomplete")
    snapshot = context.get("snapshot")
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("kind") != "bounded_outbound_model_request_context"
        or snapshot.get("operation") not in {"initial_action", "assess_and_act"}
        or not isinstance(snapshot.get("current_response_present"), bool)
    ):
        raise ValueError("layered context snapshot metadata is invalid")
    fixed = context.get("fixed_context")
    if not isinstance(fixed, Mapping) or set(fixed) != {
        "teaching_goal",
        "teacher_provided_student_profile",
    }:
        raise ValueError("layered context fixed context is invalid")
    fixed_goal = fixed.get("teaching_goal", {})
    fixed_profile = fixed.get("teacher_provided_student_profile", {})
    if (
        not isinstance(fixed_goal, Mapping)
        or fixed_goal.get("source") != "teacher_input_normalized_locally"
        or fixed_goal.get("mutable_by_model") is not False
        or not isinstance(fixed_profile, Mapping)
        or fixed_profile.get("source") != "teacher_input_normalized_locally"
        or fixed_profile.get("mutable_by_model") is not False
    ):
        raise ValueError("layered context immutable teacher input is invalid")
    current_plan = context.get("current_plan")
    if (
        not isinstance(current_plan, Mapping)
        or current_plan.get("source")
        != "deterministic_session_plan_and_last_validated_action"
    ):
        raise ValueError("layered context current plan provenance is invalid")
    working = context.get("working_memory")
    if (
        not isinstance(working, Mapping)
        or not isinstance(working.get("current_learner_response"), str)
        or not isinstance(working.get("recent_turns"), list)
        or not isinstance(working.get("current_knowledge_components"), list)
    ):
        raise ValueError("layered context working memory is invalid")
    response_id = working.get("current_response_evidence_id")
    if snapshot["current_response_present"] != isinstance(response_id, str):
        raise ValueError("layered context current-response provenance is inconsistent")
    semantic = context.get("semantic_summary")
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("compression_method")
        != "deterministic_aggregate_and_extractive_checkpoints_no_model_generation"
        or semantic.get("source") != "omitted_local_session_history"
        or semantic.get("narrative_inference_added") is not False
        or not isinstance(semantic.get("focus_checkpoints"), list)
        or len(semantic["focus_checkpoints"]) > 4
    ):
        raise ValueError("layered context semantic summary is invalid")
    knowledge = context.get("knowledge_state")
    if (
        not isinstance(knowledge, Mapping)
        or knowledge.get("source")
        != "deterministic_state_machine_over_labeled_observations"
        or not isinstance(knowledge.get("concept_mastery"), list)
        or not isinstance(knowledge.get("misconceptions"), list)
        or not isinstance(knowledge.get("unresolved_issues"), list)
    ):
        raise ValueError("layered context knowledge state is invalid")
    candidate = context.get("candidate_long_term_memory")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("status") != "candidate_unconfirmed"
        or candidate.get("may_override_teacher_profile") is not False
        or candidate.get("eligible_sources")
        != ["deepseek_v4_flash_validated_diagnosis"]
        or not isinstance(candidate.get("observations"), list)
        or len(candidate["observations"]) > 4
        or any(
            not isinstance(item, Mapping)
            or item.get("status") != "candidate_unconfirmed"
            for item in candidate["observations"]
        )
    ):
        raise ValueError("layered context candidate memory is invalid")
    retrieval = context.get("retrieval")
    if (
        not isinstance(retrieval, Mapping)
        or retrieval.get("fixed_context_always_included") is not True
        or retrieval.get("current_plan_always_included") is not True
        or retrieval.get("semantic_summary_covers_omitted_turns") is not True
        or retrieval.get("candidate_memory_policy")
        != "latest_validated_candidates_only"
    ):
        raise ValueError("layered context retrieval metadata is invalid")
    privacy = context.get("privacy")
    if (
        not isinstance(privacy, Mapping)
        or privacy.get("raw_identity_fields_sent") is not False
        or privacy.get("original_values_retained_in_context") is not False
    ):
        raise ValueError("layered context privacy metadata is invalid")
    budget = context.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError("layered context budget is missing")
    maximum = budget.get("max_chars")
    measured = _serialized_length(context)
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < MINIMUM_LAYERED_CONTEXT_CHARS
        or measured > maximum
        or budget.get("serialized_chars") != measured
        or budget.get("policy") != "hard_serialized_character_cap"
        or budget.get("retained_recent_turns") != len(working["recent_turns"])
        or budget.get("retained_candidate_observations")
        != len(candidate["observations"])
        or not isinstance(budget.get("truncated"), bool)
    ):
        raise ValueError("layered context character budget is invalid")
    ledger = context.get("evidence_ledger")
    if not isinstance(ledger, list) or not all(isinstance(item, Mapping) for item in ledger):
        raise ValueError("layered context evidence ledger is invalid")
    identifiers = [str(item.get("evidence_id", "")) for item in ledger]
    if any(not item for item in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("layered context evidence IDs must be unique and non-empty")
    dangling = set(_collect_evidence_refs(context)) - set(identifiers)
    if dangling:
        raise ValueError(f"layered context has dangling evidence refs: {sorted(dangling)}")
    boundary = context.get("claim_boundary", {})
    if not isinstance(boundary, Mapping) or any(
        boundary.get(field) is not expected
        for field, expected in {
            "model_generated_history_summary": False,
            "candidate_long_term_memory_is_confirmed": False,
            "fallback_observations_are_labeled_by_actual_source": True,
            "context_is_evidence_not_learner_ground_truth": True,
            "model_may_not_mutate_fixed_context": True,
        }.items()
    ):
        raise ValueError("layered context claim boundary is invalid")


def _goal_text(goal: Mapping[str, Any], key: str) -> str:
    value = goal.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"goal.{key} must be a non-empty string")
    return value.strip()


def _threshold(goal: Mapping[str, Any], dimension: str, default: float) -> float:
    thresholds = goal.get("success_thresholds", {})
    if not isinstance(thresholds, Mapping):
        raise ValueError("goal.success_thresholds must be an object")
    value = thresholds.get(dimension, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"goal.success_thresholds.{dimension} must be in [0, 1]")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"goal.success_thresholds.{dimension} must be in [0, 1]")
    return result


def build_goal_plan(goal: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a teaching goal into four observable intermediate objectives."""

    if not isinstance(goal, Mapping):
        raise TypeError("goal must be a mapping")
    concept = _goal_text(goal, "concept")
    objective = _goal_text(goal, "objective")
    thresholds = {
        "prerequisite": _threshold(goal, "prerequisite", 0.60),
        "conceptual": _threshold(goal, "conceptual", 0.65),
        "procedural": _threshold(goal, "procedural", 0.60),
        "transfer": _threshold(goal, "transfer", 0.55),
    }
    materials = goal.get("materials", {})
    if not isinstance(materials, Mapping):
        raise ValueError("goal.materials must be an object")
    practice = _as_nonempty_string(materials.get("practice"))
    transfer_task = _as_nonempty_string(materials.get("transfer_task"))

    specifications = (
        (
            "prerequisite",
            f"确认理解 {concept} 所需的前置知识",
            f"学习者能独立说出至少一个必要前置概念，并解释它与 {concept} 的关系。",
        ),
        (
            "conceptual",
            f"建立 {concept} 的概念表征与适用边界",
            f"学习者能用自己的话定义 {concept}，并给出一个适用例和一个不适用情形。",
        ),
        (
            "procedural",
            f"完成一次 {concept} 的关键操作或推导",
            (
                f"学习者能完成练习“{practice}”的关键步骤并逐步说明理由。"
                if practice
                else f"学习者能完成一道只考查 {concept} 的代表性练习，并逐步说明理由。"
            ),
        ),
        (
            "transfer",
            f"把 {concept} 迁移到表面不同的新情境",
            (
                f"学习者能处理迁移任务“{transfer_task}”，并说明可迁移与不可迁移的条件。"
                if transfer_task
                else f"学习者能在一个新情境中判断是否应使用 {concept}，并说明判断依据。"
            ),
        ),
    )

    steps: list[dict[str, Any]] = []
    for index, (dimension, step_objective, criterion) in enumerate(specifications, 1):
        steps.append(
            {
                "step_id": f"goal_step_{index:02d}",
                "order": index,
                "dimension": dimension,
                "objective": step_objective,
                "verification": {
                    "method": "observable_learner_response_or_task_performance",
                    "success_criterion": criterion,
                    "mastery_threshold": thresholds[dimension],
                },
                "status": "active" if index == 1 else "pending",
            }
        )

    return {
        "schema": GOAL_PLAN_SCHEMA,
        "goal": {"concept": concept, "objective": objective},
        "status": "active",
        "intermediate_objectives": steps,
        "active_step": steps[0]["step_id"],
        "progress": {
            "completed_steps": 0,
            "total_steps": len(steps),
            "fraction": 0.0,
        },
        "claim_boundary": {
            "objectives_are_deterministic_plan_scaffolds": True,
            "objective_completion_requires_observation": True,
            "plan_is_learning_effect_evidence": False,
        },
    }
