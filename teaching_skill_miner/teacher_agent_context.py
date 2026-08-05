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
GOAL_PLAN_SCHEMA = "teaching_skill_miner.teacher_agent_goal_plan.v1"

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
    for container in (next_focus, primary, action, goal):
        if isinstance(container, Mapping):
            kcs.extend(_knowledge_components(container))
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
    return {
        "turn_count": len(events),
        "round_range": [min(rounds), max(rounds)] if rounds else None,
        "signal_counts": dict(sorted(signals.items())),
        "focus_counts": dict(sorted(focuses.items())),
        "skill_counts": dict(sorted(skills.items())),
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
