"""Bounded, tool-using runtime for a real Teaching Agent.

The existing live agent is intentionally a one-turn decision service.  This
module adds the missing *agent loop*: DeepSeek proposes either an allowlisted
teaching-tool call or one final teaching action, the local runtime executes the
tool, and the result is fed back to the model.  The model never executes Python,
chooses an unknown Skill, or changes the learner record directly.

This is Codex-inspired orchestration (bounded steps, event traces, retries and
idempotency), not a claim of feature parity with Codex or Claude Code.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from .teacher_agent import PRIMARY_ROLES, canonical_sha256, validate_skill_library


LOOP_SCHEMA = "teaching_skill_miner.teacher_agent_loop.v1"
PLAN_SCHEMA = "teaching_skill_miner.teacher_agent_loop_plan.v1"
_FOCUS = frozenset({"prerequisite", "conceptual", "procedural", "transfer"})
_KINDS = frozenset({"tool_calls", "route_ready", "teaching_action", "terminate"})
_ALLOWED_TOOLS = frozenset(
    {
        "inspect_student_state",
        "inspect_recent_history",
        "search_skills",
        "select_skills",
        "set_next_focus",
        "evaluate_termination",
    }
)


class TeachingAgentLoopError(ValueError):
    """Raised when a model plan violates the loop contract."""


class StructuredModel(Protocol):
    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_kind: str,
        require_remote_consent: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class TeachingAgentLoopOptions:
    """Safety limits for one planning/execution run."""

    max_steps: int = 8
    max_tool_calls_per_step: int = 3
    max_repeated_tool_calls: int = 2
    model_retries: int = 1
    recent_history_limit: int = 6
    max_action_chars: int = 1400

    def validated(self) -> "TeachingAgentLoopOptions":
        if not 1 <= self.max_steps <= 32:
            raise TeachingAgentLoopError("max_steps must be in [1, 32]")
        if not 1 <= self.max_tool_calls_per_step <= 8:
            raise TeachingAgentLoopError("max_tool_calls_per_step must be in [1, 8]")
        if not 1 <= self.max_repeated_tool_calls <= 4:
            raise TeachingAgentLoopError("max_repeated_tool_calls must be in [1, 4]")
        if not 0 <= self.model_retries <= 4:
            raise TeachingAgentLoopError("model_retries must be in [0, 4]")
        if not 1 <= self.recent_history_limit <= 12:
            raise TeachingAgentLoopError("recent_history_limit must be in [1, 12]")
        if not 80 <= self.max_action_chars <= 3000:
            raise TeachingAgentLoopError("max_action_chars is outside the supported range")
        return self


def _short(value: Any, maximum: int = 400) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())[:maximum]


def _skill_index(library: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        validate_skill_library(library)
    except Exception as exc:  # expose one stable boundary to callers
        raise TeachingAgentLoopError("skill_library is invalid") from exc
    return {
        str(item["skill_id"]): deepcopy(dict(item))
        for item in library.get("skills", [])
        if isinstance(item, Mapping)
    }


def _session_view(
    session: Mapping[str, Any],
    *,
    history_limit: int,
    outbound_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a bounded model context without exposing integrity/private fields."""

    # Live DeepSeek turns already have a validated, redacted layered context.
    # Project that context into the loop rather than reading raw ``session``
    # history.  This keeps the tool-using path subject to the same privacy and
    # character budgets as the legacy planner.  Direct callers (and offline
    # tests) continue to use the normal session projection.
    source: Mapping[str, Any] = session
    current_response = ""
    current_response_evidence_id: str | None = None
    current_focus = ""
    if isinstance(outbound_context, Mapping):
        fixed = outbound_context.get("fixed_context", {})
        fixed_goal = (
            fixed.get("teaching_goal", {})
            if isinstance(fixed, Mapping)
            else {}
        )
        fixed_profile = (
            fixed.get("teacher_provided_student_profile", {})
            if isinstance(fixed, Mapping)
            else {}
        )
        knowledge_state = outbound_context.get("knowledge_state", {})
        if not isinstance(knowledge_state, Mapping):
            knowledge_state = {}
        mastery: dict[str, float] = {}
        for item in knowledge_state.get("concept_mastery", []) or []:
            if not isinstance(item, Mapping) or not _finite(item.get("value")):
                continue
            dimension = _short(item.get("dimension"), 40)
            if dimension:
                mastery[dimension] = float(item["value"])
        understanding = knowledge_state.get("current_understanding_signal", {})
        if not isinstance(understanding, Mapping):
            understanding = {}
        next_focus = knowledge_state.get("next_focus", {})
        next_focus_value = (
            next_focus.get("dimension", "")
            if isinstance(next_focus, Mapping)
            else next_focus
        )
        working = outbound_context.get("working_memory", {})
        if not isinstance(working, Mapping):
            working = {}
        # The live context is built before the controller commits the current
        # learner turn.  Preserve that bounded, already-redacted answer as an
        # explicit observation; otherwise the route loop would only see the
        # previous mastery/signal and could select a Skill for the wrong turn.
        current_response = _short(working.get("current_learner_response"), 720)
        response_id = working.get("current_response_evidence_id")
        current_response_evidence_id = (
            _short(response_id, 120) if isinstance(response_id, str) else None
        )
        current_focus = _short(working.get("current_focus"), 80)
        projected_history: list[dict[str, Any]] = []
        for row in working.get("recent_turns", []) or []:
            if not isinstance(row, Mapping):
                continue
            projected_history.append(
                {
                    "round": row.get("round"),
                    "action": {"type": _short(row.get("action_type"), 80)},
                    "learner_response": _short(
                        row.get("learner_response", row.get("learner_response_excerpt", "")),
                        360,
                    ),
                    "structured_signal": {
                        "label": _short(row.get("signal"), 40),
                    },
                }
            )
        current_plan = outbound_context.get("current_plan", {})
        if not isinstance(current_plan, Mapping):
            current_plan = {}
        current_action = current_plan.get("current_action", {})
        if not isinstance(current_action, Mapping):
            current_action = {}
        source = {
            **dict(session),
            "goal": deepcopy(dict(fixed_goal))
            if isinstance(fixed_goal, Mapping)
            else {},
            "student_profile": deepcopy(dict(fixed_profile))
            if isinstance(fixed_profile, Mapping)
            else {},
            "student_state": {
                "knowledge_mastery": mastery,
                "misconceptions": deepcopy(
                    list(knowledge_state.get("misconceptions", []) or [])
                ),
                "understanding_signal": deepcopy(dict(understanding)),
                "next_focus": _short(next_focus_value, 40),
            },
            "history": projected_history,
            "current_action": {
                "type": _short(current_action.get("action_type"), 80),
                "teacher_action": {
                    "message": _short(current_action.get("teacher_message"), 500)
                },
            },
            "current_observation": {
                "learner_response": current_response,
                "evidence_id": current_response_evidence_id,
                "focus": current_focus,
            },
            "round": int(session.get("round", 0) or 0),
        }

    goal = source.get("goal", {}) if isinstance(source.get("goal"), Mapping) else {}
    profile = (
        source.get("student_profile", {})
        if isinstance(source.get("student_profile"), Mapping)
        else {}
    )
    state = (
        source.get("student_state", {})
        if isinstance(source.get("student_state"), Mapping)
        else {}
    )
    mastery = state.get("knowledge_mastery", profile.get("initial_mastery", {}))
    if not isinstance(mastery, Mapping):
        mastery = {}
    history = source.get("history", [])
    compact_history: list[dict[str, Any]] = []
    if isinstance(history, list):
        for row in history[-history_limit:]:
            if not isinstance(row, Mapping):
                continue
            signal = row.get("structured_signal", {})
            if not isinstance(signal, Mapping):
                signal = {}
            compact_history.append(
                {
                    "round": row.get("round"),
                    "action_type": _short(
                        (row.get("action", {}) or {}).get("type")
                        if isinstance(row.get("action"), Mapping)
                        else "",
                        80,
                    ),
                    "learner_response": _short(
                        row.get("learner_response", row.get("learner_text", "")), 360
                    ),
                    "signal": _short(signal.get("label"), 40),
                }
            )
    current_action = source.get("current_action", {})
    if not isinstance(current_action, Mapping):
        current_action = {}
    teacher_action = current_action.get("teacher_action", {})
    if not isinstance(teacher_action, Mapping):
        teacher_action = {}
    if not current_response:
        current_response = _short(source.get("current_learner_response"), 720)
    if current_response_evidence_id is None:
        response_id = source.get("current_response_evidence_id")
        if isinstance(response_id, str):
            current_response_evidence_id = _short(response_id, 120)
    if not current_focus:
        current_focus = _short(source.get("current_focus"), 80)
    state_next_focus = state.get("next_focus", "")
    if isinstance(state_next_focus, Mapping):
        state_next_focus = state_next_focus.get("dimension", "")
    return {
        "goal": {
            "concept": _short(goal.get("concept"), 240),
            "objective": _short(goal.get("objective"), 500),
            "knowledge_components": [
                _short(item, 120)
                for item in (goal.get("knowledge_components", []) or [])[:24]
                if str(item).strip()
            ],
            "success_thresholds": {
                str(key): float(value)
                for key, value in (
                    goal.get("success_thresholds", {}) or {}
                ).items()
                if _finite(value)
            }
            if isinstance(goal.get("success_thresholds", {}), Mapping)
            else {},
            "max_rounds": int(goal.get("max_rounds", 0) or 0),
        },
        "student": {
            "learner_level": _short(profile.get("learner_level"), 64),
            "preferences": [_short(item, 120) for item in (profile.get("preferences", []) or [])[:8]],
            "mastery": {str(k): float(v) for k, v in mastery.items() if _finite(v)},
            "misconceptions": [
                {
                    "tag": _short(item.get("tag"), 100),
                    "status": _short(item.get("status"), 40),
                    "description": _short(item.get("description"), 240),
                }
                for item in (state.get("misconceptions", []) or [])[-8:]
                if isinstance(item, Mapping)
            ],
            "understanding_signal": deepcopy(state.get("understanding_signal", {})),
            "next_focus": _short(state_next_focus, 40),
        },
        "current_action": {
            "type": _short(current_action.get("type"), 80),
            "message": _short(teacher_action.get("message"), 500),
        },
        "current_observation": {
            "learner_response": current_response,
            "evidence_id": current_response_evidence_id,
            "focus": current_focus,
        },
        "history": compact_history,
        "round": int(source.get("round", 0) or 0),
        "status": _short(source.get("status", "active"), 32),
    }


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def _model_messages(
    context: Mapping[str, Any],
    library: Mapping[str, Any],
    previous_results: Sequence[Mapping[str, Any]],
    runtime_state: Mapping[str, Any],
) -> list[dict[str, str]]:
    skill_view = [
        {
            "skill_id": str(s.get("skill_id")),
            "name": _short(s.get("name"), 100),
            "role": _short(s.get("role"), 40),
            "focus_dimension": _short(s.get("focus_dimension"), 40),
            "action_type": _short(s.get("action_type"), 100),
            "applicable_signals": list(s.get("applicable_signals", []) or [])[:8],
            "message_template": _short(s.get("message_template"), 400),
        }
        for s in (library.get("skills", []) or [])
        if isinstance(s, Mapping)
    ]
    tool_results = [deepcopy(dict(item)) for item in previous_results[-8:]]
    system = (
        "你是一个实时 Teaching Agent 的规划器。每次只返回一个 JSON 对象，"
        "必须先调用 allowlisted 工具读取状态/查找并选择 Skill，再返回教学动作。"
        "不要输出思维链，不得调用未列出的工具，不得直接泄露完整答案。"
        f"可用工具={sorted(_ALLOWED_TOOLS)}。输出 schema={PLAN_SCHEMA}。"
        "kind 只能是 tool_calls、route_ready、teaching_action、terminate。"
        "tool_calls 的每项为 {call_id,name,arguments}；teaching_action 必须包含 "
        "selected_skill_id,supporting_skill_ids,next_focus,action_type,message,expected_signal,reason；"
        "完成 select_skills 与 set_next_focus 后，优先返回 {kind:route_ready,reason}，"
        "由受约束的最终动作规划器生成教师话语；terminate 必须包含 "
        "outcome(success|handoff),reason；若选择 success，先调用 evaluate_termination。"
        "teaching_context.student.understanding_signal 在 post-assessment 模式下是本轮已校验状态；"
        "必须优先使用它而不是上一轮 history signal。select_skills 必须逐字复制 skill_library 中的 "
        "skill_id；若工具返回 accepted=false，按 allowed_primary_skill_ids 修正一次，不要重复读取同一状态。"
    )
    user = json.dumps(
        {
            "teaching_context": context,
            "skill_library": skill_view,
            "previous_tool_results": tool_results,
            "runtime_state": {
                "selected_skill_id": _short(
                    runtime_state.get("selected_skill_id"), 120
                ),
                "supporting_skill_ids": [
                    _short(item, 120)
                    for item in (
                        runtime_state.get("supporting_skill_ids", []) or []
                    )[:2]
                ],
                "next_focus": _short(runtime_state.get("next_focus"), 40),
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _invoke_model(client: Any, messages: Sequence[Mapping[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(client, "chat_json"):
        result = client.chat_json(
            messages,
            request_kind="teacher_agent_loop",
            require_remote_consent=True,
        )
    elif callable(client):
        result = client(messages)
    else:
        raise TeachingAgentLoopError("model client must provide chat_json or be callable")
    if isinstance(result, tuple) and len(result) == 2:
        plan, trace = result
    else:
        plan, trace = result, {}
    if not isinstance(plan, Mapping):
        raise TeachingAgentLoopError("model returned a non-object plan")
    return dict(plan), deepcopy(dict(trace)) if isinstance(trace, Mapping) else {}


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") not in {None, PLAN_SCHEMA}:
        raise TeachingAgentLoopError("model plan schema is invalid")
    kind = plan.get("kind")
    if kind not in _KINDS:
        raise TeachingAgentLoopError("model plan kind is invalid")
    if kind == "tool_calls":
        calls = plan.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            raise TeachingAgentLoopError("tool_calls plan must contain at least one call")
        if len(calls) > 8:
            raise TeachingAgentLoopError("tool_calls plan contains too many calls")
        clean: list[dict[str, Any]] = []
        for index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                raise TeachingAgentLoopError(f"tool_calls[{index}] is not an object")
            name = str(call.get("name", "")).strip()
            if name not in _ALLOWED_TOOLS:
                raise TeachingAgentLoopError(f"tool is not allowlisted: {name}")
            arguments = call.get("arguments", {})
            if not isinstance(arguments, Mapping):
                raise TeachingAgentLoopError(f"tool_calls[{index}].arguments is invalid")
            clean.append(
                {
                    "call_id": _short(call.get("call_id"), 80) or f"call_{index + 1}",
                    "name": name,
                    "arguments": deepcopy(dict(arguments)),
                }
            )
        return {"schema": PLAN_SCHEMA, "kind": kind, "tool_calls": clean}
    if kind == "terminate":
        outcome = str(plan.get("outcome", "")).strip()
        if outcome not in {"success", "handoff"}:
            raise TeachingAgentLoopError("termination outcome is invalid")
        reason = _short(plan.get("reason"), 300)
        if not reason:
            raise TeachingAgentLoopError("termination reason is required")
        return {"schema": PLAN_SCHEMA, "kind": kind, "outcome": outcome, "reason": reason}
    if kind == "route_ready":
        reason = _short(plan.get("reason"), 600)
        if not reason:
            raise TeachingAgentLoopError("route_ready reason is required")
        return {"schema": PLAN_SCHEMA, "kind": kind, "reason": reason}
    required = (
        "selected_skill_id",
        "supporting_skill_ids",
        "next_focus",
        "action_type",
        "message",
        "expected_signal",
        "reason",
    )
    missing = [field for field in required if field not in plan]
    if missing:
        raise TeachingAgentLoopError("teaching_action missing: " + ", ".join(missing))
    supporting = plan.get("supporting_skill_ids")
    if not isinstance(supporting, list):
        raise TeachingAgentLoopError("supporting_skill_ids must be a list")
    message = _short(plan.get("message"), 3000)
    if not message:
        raise TeachingAgentLoopError("teaching_action.message is required")
    focus = str(plan.get("next_focus", "")).strip()
    if focus not in _FOCUS:
        raise TeachingAgentLoopError("next_focus is invalid")
    return {
        "schema": PLAN_SCHEMA,
        "kind": kind,
        "selected_skill_id": _short(plan.get("selected_skill_id"), 120),
        "supporting_skill_ids": [_short(item, 120) for item in supporting[:2]],
        "next_focus": focus,
        "action_type": _short(plan.get("action_type"), 120),
        "message": message,
        "expected_signal": _short(plan.get("expected_signal"), 600),
        "reason": _short(plan.get("reason"), 600),
        "question_contract": deepcopy(plan.get("question_contract", {}))
        if isinstance(plan.get("question_contract", {}), Mapping)
        else {},
    }


def _tool_result(
    name: str,
    arguments: Mapping[str, Any],
    context: Mapping[str, Any],
    library: Mapping[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute one local, deterministic and allowlisted teaching tool."""

    skills = _skill_index(library)
    if name == "inspect_student_state":
        return deepcopy(dict(context.get("student", {})))
    if name == "inspect_recent_history":
        rows = context.get("history", [])
        limit = arguments.get("limit", 4)
        try:
            limit = max(1, min(8, int(limit)))
        except (TypeError, ValueError):
            limit = 4
        return {"history": deepcopy(list(rows)[-limit:])}
    if name == "search_skills":
        query = _short(arguments.get("query"), 160).casefold()
        signals = {str(item) for item in arguments.get("signals", []) if str(item)}
        scored: list[tuple[int, dict[str, Any]]] = []
        for skill in skills.values():
            if skill.get("role") == "support":
                continue
            haystack = " ".join(
                str(skill.get(key, ""))
                for key in ("skill_id", "name", "role", "focus_dimension", "action_type", "selection_rationale")
            ).casefold()
            score = (3 if query and query in haystack else 0) + len(
                signals.intersection(set(skill.get("applicable_signals", []) or []))
            )
            scored.append(
                (
                    score,
                    {
                        "skill_id": skill["skill_id"],
                        "name": skill.get("name", ""),
                        "role": skill.get("role", ""),
                        "focus_dimension": skill.get("focus_dimension", ""),
                        "action_type": skill.get("action_type", ""),
                        "applicable_signals": list(skill.get("applicable_signals", []) or []),
                        "selection_rationale": _short(skill.get("selection_rationale"), 300),
                    },
                )
            )
        scored.sort(key=lambda item: (-item[0], str(item[1]["skill_id"])))
        max_results = arguments.get("max_results", 6)
        try:
            max_results = max(1, min(12, int(max_results)))
        except (TypeError, ValueError):
            max_results = 6
        return {"skills": [item[1] for item in scored[:max_results]], "query": query}
    if name == "select_skills":
        primary = _short(arguments.get("primary_skill_id"), 120)
        supports = arguments.get("supporting_skill_ids", [])
        if primary not in skills or skills[primary].get("role") not in PRIMARY_ROLES:
            allowed_primary_ids = sorted(
                skill_id
                for skill_id, skill in skills.items()
                if skill.get("role") in PRIMARY_ROLES
            )
            return {
                "accepted": False,
                "error_code": "invalid_primary_skill_id",
                "selected_skill_id": None,
                "requested_primary_skill_id": primary or None,
                "allowed_primary_skill_ids": allowed_primary_ids,
                "reason": "primary_skill_id must exactly match one allowlisted primary Skill",
            }
        rejected_support_ids: list[str] = []
        if not isinstance(supports, list):
            supports = []
            rejected_support_ids.append("non_list_supporting_skill_ids")
        support_ids: list[str] = []
        allowed_support_ids = set(
            skills[primary].get("supporting_skill_ids", []) or []
        )
        for skill_id in supports[:4]:
            skill_id = _short(skill_id, 120)
            if (
                skill_id not in skills
                or skills[skill_id].get("role") != "support"
                or skill_id not in allowed_support_ids
            ):
                if skill_id:
                    rejected_support_ids.append(skill_id)
                continue
            if skill_id not in support_ids:
                support_ids.append(skill_id)
        if len(supports) > 4:
            rejected_support_ids.append("supporting_skill_budget_exceeded")
        state["selected_skill_id"] = primary
        state["supporting_skill_ids"] = support_ids
        if state.get("next_focus") not in _FOCUS:
            inferred_focus = _short(skills[primary].get("focus_dimension"), 40)
            if inferred_focus in _FOCUS:
                state["next_focus"] = inferred_focus
        state["selection_reason"] = _short(arguments.get("reason"), 600)
        return {
            "accepted": True,
            "selected_skill_id": primary,
            "supporting_skill_ids": support_ids,
            "rejected_supporting_skill_ids": rejected_support_ids[:4],
            "reason": state["selection_reason"],
        }
    if name == "set_next_focus":
        focus = _short(arguments.get("next_focus"), 40)
        if focus not in _FOCUS:
            return {
                "accepted": False,
                "error_code": "invalid_next_focus",
                "next_focus": None,
                "allowed_focus_values": sorted(_FOCUS),
            }
        state["next_focus"] = focus
        return {"accepted": True, "next_focus": focus}
    if name == "evaluate_termination":
        goal = context.get("goal", {})
        source = context.get("student", {}).get("mastery", {})
        thresholds = goal.get("success_thresholds", {}) if isinstance(goal, Mapping) else {}
        unmet = [
            key
            for key, value in thresholds.items()
            if _finite(value) and float(source.get(key, 0.0)) < float(value)
        ]
        active_misconception = any(
            item.get("status") == "active"
            for item in context.get("student", {}).get("misconceptions", [])
            if isinstance(item, Mapping)
        )
        label = str(context.get("student", {}).get("understanding_signal", {}).get("label", ""))
        eligible = not unmet and not active_misconception and label == "correct"
        return {
            "eligible": eligible,
            "unmet_dimensions": unmet,
            "active_misconception": active_misconception,
            "latest_signal": label,
            "reason": "all thresholds and latest correctness checks pass" if eligible else "teaching evidence is incomplete",
        }
    raise TeachingAgentLoopError("tool is not allowlisted")


def _fallback_action(session: Mapping[str, Any], state: Mapping[str, Any], reason: str) -> dict[str, Any]:
    skills = _skill_index(session.get("skill_library", {}))
    current = session.get("current_action", {})
    current_primary = (
        current.get("primary_skill", {}).get("skill_id")
        if isinstance(current, Mapping) and isinstance(current.get("primary_skill"), Mapping)
        else None
    )
    chosen = skills.get(str(state.get("selected_skill_id"))) or skills.get(str(current_primary))
    if chosen is None:
        signal = str(session.get("student_state", {}).get("understanding_signal", {}).get("label", "not_observed"))
        candidates = [
            item for item in skills.values()
            if item.get("role") != "support" and signal in set(item.get("applicable_signals", []) or [])
        ]
        chosen = sorted(candidates or [item for item in skills.values() if item.get("role") != "support"], key=lambda x: str(x["skill_id"]))[0]
    goal = session.get("goal", {}) if isinstance(session.get("goal"), Mapping) else {}
    values = {
        "concept": _short(goal.get("concept"), 180),
        "example": _short((goal.get("materials", {}) or {}).get("example"), 300) if isinstance(goal.get("materials"), Mapping) else "",
        "practice": _short((goal.get("materials", {}) or {}).get("practice"), 300) if isinstance(goal.get("materials"), Mapping) else "",
        "transfer_task": _short((goal.get("materials", {}) or {}).get("transfer_task"), 300) if isinstance(goal.get("materials"), Mapping) else "",
        "misconception": "当前回答中的关键混淆点",
        "learner_level": _short((session.get("student_profile", {}) or {}).get("learner_level"), 40) if isinstance(session.get("student_profile"), Mapping) else "",
    }
    template = str(chosen.get("message_template", "请先用自己的话说明这个概念。"))
    message = template
    for key, value in values.items():
        message = message.replace("{" + key + "}", value)
    return {
        "selected_skill_id": chosen["skill_id"],
        "supporting_skill_ids": [],
        "next_focus": str(state.get("next_focus") or chosen.get("focus_dimension") or "conceptual"),
        "message": _short(message, 1400),
        "expected_signal": _short(chosen.get("expected_signal"), 500) or "学生给出可核验的解释或步骤。",
        "reason": "deterministic_safety_fallback: " + _short(reason, 260),
        "skill": {
            "skill_id": chosen["skill_id"],
            "name": chosen.get("name", ""),
            "action_type": chosen.get("action_type", ""),
        },
    }


def run_teaching_agent_loop(
    session: Mapping[str, Any],
    client: StructuredModel | Callable[[Sequence[Mapping[str, str]]], Any],
    *,
    options: TeachingAgentLoopOptions | None = None,
    outbound_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a bounded plan→tool→observe loop and return an auditable result.

    ``session`` is never mutated.  The returned ``events`` list contains model,
    tool, retry, guard and final-action events, making a UI or a benchmark able
    to show what the agent actually did.
    """

    options = (options or TeachingAgentLoopOptions()).validated()
    if not isinstance(session, Mapping):
        raise TeachingAgentLoopError("session must be an object")
    library = session.get("skill_library")
    if not isinstance(library, Mapping):
        raise TeachingAgentLoopError("session.skill_library is required")
    _skill_index(library)
    context = _session_view(
        session,
        history_limit=options.recent_history_limit,
        outbound_context=outbound_context,
    )
    state = {"selected_skill_id": None, "supporting_skill_ids": [], "next_focus": context["student"].get("next_focus")}
    events: list[dict[str, Any]] = []
    previous_results: list[dict[str, Any]] = []
    repeated_calls: dict[str, int] = {}
    readiness_checked = False
    terminal: dict[str, Any] | None = None
    for step in range(1, options.max_steps + 1):
        messages = _model_messages(context, library, previous_results, state)
        plan: dict[str, Any] | None = None
        trace: dict[str, Any] = {}
        last_error = ""
        for attempt in range(options.model_retries + 1):
            try:
                plan, trace = _invoke_model(client, messages)
                plan = _validate_plan(plan)
                events.append({"type": "model_plan", "step": step, "attempt": attempt + 1, "kind": plan["kind"], "trace": deepcopy(trace)})
                break
            except Exception as exc:  # model boundary is fail-closed and retried
                last_error = type(exc).__name__ + ": " + _short(exc, 180)
                events.append({"type": "model_error", "step": step, "attempt": attempt + 1, "error": last_error})
                if attempt >= options.model_retries:
                    break
        if plan is None:
            terminal = {"status": "action_ready", "action": _fallback_action(session, state, last_error or "model unavailable"), "reason": last_error or "model unavailable"}
            events.append({"type": "fallback", "step": step, "reason": terminal["reason"]})
            break
        if plan["kind"] == "tool_calls":
            calls = plan["tool_calls"][: options.max_tool_calls_per_step]
            if len(plan["tool_calls"]) > len(calls):
                events.append({"type": "guard", "step": step, "reason": "tool_call_budget_exceeded"})
            for call in calls:
                signature = hashlib.sha256(canonical_sha256({"name": call["name"], "arguments": call["arguments"]}).encode()).hexdigest()[:16]
                repeated_calls[signature] = repeated_calls.get(signature, 0) + 1
                if repeated_calls[signature] > options.max_repeated_tool_calls:
                    events.append({"type": "guard", "step": step, "reason": "repeated_tool_call", "tool": call["name"], "signature": signature})
                    if state.get("selected_skill_id") and state.get("next_focus") in _FOCUS:
                        terminal = {
                            "status": "route_ready",
                            "action": None,
                            "reason": "repeated tool call; using the last validated route",
                        }
                        events.append(
                            {
                                "type": "route_ready",
                                "step": step,
                                "skill_id": state["selected_skill_id"],
                                "next_focus": state["next_focus"],
                            }
                        )
                    else:
                        terminal = {
                            "status": "action_ready",
                            "action": _fallback_action(
                                session, state, "repeated tool call"
                            ),
                            "reason": "repeated tool call",
                        }
                        events.append(
                            {"type": "fallback", "step": step, "reason": "repeated tool call"}
                        )
                    break
                try:
                    result = _tool_result(call["name"], call["arguments"], context, library, state)
                    result = {"call_id": call["call_id"], "tool": call["name"], "ok": True, "result": result}
                    events.append({"type": "tool_result", "step": step, **deepcopy(result)})
                    if call["name"] == "evaluate_termination":
                        readiness_checked = bool(result.get("result", {}).get("eligible", False))
                except Exception as exc:
                    result = {"call_id": call["call_id"], "tool": call["name"], "ok": False, "error": type(exc).__name__ + ": " + _short(exc, 180)}
                    events.append({"type": "tool_error", "step": step, **deepcopy(result)})
                previous_results.append(result)
            if terminal:
                break
            continue
        if plan["kind"] == "terminate":
            if plan["outcome"] == "success" and not readiness_checked:
                events.append({"type": "guard", "step": step, "reason": "success_requires_evaluate_termination"})
                terminal = {"status": "action_ready", "action": _fallback_action(session, state, "success requested before readiness check"), "reason": "success requested before readiness check"}
                events.append(
                    {
                        "type": "fallback",
                        "step": step,
                        "reason": "success requested before readiness check",
                    }
                )
            else:
                terminal = {"status": "succeeded" if plan["outcome"] == "success" else "terminated_unable", "reason": plan["reason"], "action": None}
            break
        if plan["kind"] == "route_ready":
            if not state.get("selected_skill_id") or state.get("next_focus") not in _FOCUS:
                events.append(
                    {
                        "type": "guard",
                        "step": step,
                        "reason": "route_ready_requires_selected_skill_and_focus",
                    }
                )
                terminal = {
                    "status": "action_ready",
                    "action": _fallback_action(
                        session,
                        state,
                        "route emitted before required tools",
                    ),
                    "reason": "route emitted before required tools",
                }
                events.append(
                    {
                        "type": "fallback",
                        "step": step,
                        "reason": "route emitted before required tools",
                    }
                )
            else:
                terminal = {
                    "status": "route_ready",
                    "action": None,
                    "reason": plan["reason"],
                }
                events.append(
                    {
                        "type": "route_ready",
                        "step": step,
                        "skill_id": state["selected_skill_id"],
                        "next_focus": state["next_focus"],
                    }
                )
            break
        selected = state.get("selected_skill_id")
        planned_selected = plan.get("selected_skill_id")
        if not selected or not planned_selected or selected != planned_selected:
            events.append({"type": "guard", "step": step, "reason": "action_without_runtime_skill_selection"})
            terminal = {"status": "action_ready", "action": _fallback_action(session, state, "action emitted before select_skills"), "reason": "action emitted before select_skills"}
            events.append({"type": "fallback", "step": step, "reason": "action emitted before select_skills"})
            break
        planned_supporting = plan.get("supporting_skill_ids")
        if not isinstance(planned_supporting, list) or planned_supporting != state.get(
            "supporting_skill_ids", []
        ):
            events.append(
                {
                    "type": "guard",
                    "step": step,
                    "reason": "action_supports_do_not_match_runtime_selection",
                }
            )
            terminal = {
                "status": "action_ready",
                "action": _fallback_action(
                    session, state, "action support Skills changed after selection"
                ),
                "reason": "action support Skills changed after selection",
            }
            events.append(
                {
                    "type": "fallback",
                    "step": step,
                    "reason": "action support Skills changed after selection",
                }
            )
            break
        if plan.get("next_focus") != state.get("next_focus"):
            events.append(
                {
                    "type": "guard",
                    "step": step,
                    "reason": "action_focus_does_not_match_runtime_selection",
                }
            )
            terminal = {
                "status": "action_ready",
                "action": _fallback_action(
                    session, state, "action focus changed after selection"
                ),
                "reason": "action focus changed after selection",
            }
            events.append(
                {
                    "type": "fallback",
                    "step": step,
                    "reason": "action focus changed after selection",
                }
            )
            break
        message = plan.get("message")
        if not isinstance(message, str) or len(message) > options.max_action_chars:
            events.append(
                {
                    "type": "guard",
                    "step": step,
                    "reason": "action_message_budget_exceeded",
                }
            )
            terminal = {
                "status": "action_ready",
                "action": _fallback_action(
                    session, state, "action message exceeded the configured budget"
                ),
                "reason": "action message exceeded the configured budget",
            }
            events.append(
                {
                    "type": "fallback",
                    "step": step,
                    "reason": "action message exceeded the configured budget",
                }
            )
            break
        skills = _skill_index(library)
        skill = skills.get(selected)
        if (
            not skill
            or skill.get("role") == "support"
            or skill.get("action_type") != plan.get("action_type")
        ):
            events.append({"type": "guard", "step": step, "reason": "selected_skill_invalid_for_action"})
            terminal = {"status": "action_ready", "action": _fallback_action(session, state, "selected Skill invalid"), "reason": "selected Skill invalid"}
            events.append({"type": "fallback", "step": step, "reason": "selected Skill invalid"})
            break
        terminal = {
            "status": "action_ready",
            "action": {
                **plan,
                "skill": {"skill_id": skill["skill_id"], "name": skill.get("name", ""), "action_type": skill.get("action_type", "")},
            },
            "reason": "model produced a validated teaching action",
        }
        events.append(
            {
                "type": "teaching_action",
                "step": step,
                "skill_id": selected,
                "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
            }
        )
        break
    if terminal is None:
        if state.get("selected_skill_id") and state.get("next_focus") in _FOCUS:
            terminal = {
                "status": "route_ready",
                "action": None,
                "reason": "max_steps exceeded; using the last validated route",
            }
            events.append(
                {
                    "type": "route_ready",
                    "step": options.max_steps,
                    "skill_id": state["selected_skill_id"],
                    "next_focus": state["next_focus"],
                }
            )
        else:
            terminal = {
                "status": "action_ready",
                "action": _fallback_action(session, state, "max_steps exceeded"),
                "reason": "max_steps exceeded",
            }
            events.append(
                {"type": "fallback", "step": options.max_steps, "reason": "max_steps exceeded"}
            )
    return {
        "schema": LOOP_SCHEMA,
        "status": terminal["status"],
        "action": deepcopy(terminal.get("action")),
        "termination_reason": terminal.get("reason", ""),
        "loop_state": deepcopy(state),
        "steps": max((int(event.get("step", 0)) for event in events), default=0),
        "events": events,
        "session_fingerprint": canonical_sha256(dict(session)),
        "deterministic_fallback": any(
            event.get("type") == "fallback" for event in events
        ),
    }


def public_agent_loop_trace(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project a loop result into a UI/store-safe, non-chain-of-thought trace.

    Tool results can contain bounded learner excerpts because they are useful to
    the local executor, but exposing those payloads in the durable runtime makes
    the trace unnecessarily large and easy to misuse.  The public projection
    keeps event type, tool name, guards, hashes and transport metadata only.
    """

    if not isinstance(result, Mapping):
        raise TeachingAgentLoopError("loop result must be an object")
    events: list[dict[str, Any]] = []
    model_calls = 0
    tool_calls = 0
    retries = 0
    for raw_event in result.get("events", []) or []:
        if not isinstance(raw_event, Mapping):
            continue
        event_type = _short(raw_event.get("type"), 40)
        item: dict[str, Any] = {
            "type": event_type,
            "step": int(raw_event.get("step", 0) or 0),
        }
        if event_type == "model_plan":
            model_calls += 1
            attempt = int(raw_event.get("attempt", 1) or 1)
            retries += int(attempt > 1)
            item.update(
                {
                    "attempt": attempt,
                    "kind": _short(raw_event.get("kind"), 40),
                    "trace": _public_model_trace(raw_event.get("trace")),
                }
            )
        elif event_type == "model_error":
            model_calls += 1
            attempt = int(raw_event.get("attempt", 1) or 1)
            retries += int(attempt > 1)
            item.update(
                {
                    "attempt": attempt,
                    "error_type": _error_type(raw_event.get("error")),
                }
            )
        elif event_type in {"tool_result", "tool_error"}:
            tool_calls += 1
            item.update(
                {
                    "call_id": _short(raw_event.get("call_id"), 80),
                    "tool": _short(raw_event.get("tool"), 80),
                    "ok": event_type == "tool_result" and raw_event.get("ok") is True,
                }
            )
            if event_type == "tool_error":
                item["error_type"] = _error_type(raw_event.get("error"))
        elif event_type == "guard":
            item.update(
                {
                    "reason": _short(raw_event.get("reason"), 160),
                    "tool": _short(raw_event.get("tool"), 80),
                }
            )
        elif event_type == "teaching_action":
            item.update(
                {
                    "skill_id": _short(raw_event.get("skill_id"), 120),
                    "message_sha256": _short(raw_event.get("message_sha256"), 64),
                }
            )
        elif event_type == "route_ready":
            item.update(
                {
                    "skill_id": _short(raw_event.get("skill_id"), 120),
                    "next_focus": _short(raw_event.get("next_focus"), 40),
                }
            )
        elif event_type == "fallback":
            item["reason"] = _short(raw_event.get("reason"), 200)
        events.append(item)
    state = result.get("loop_state", {})
    if not isinstance(state, Mapping):
        state = {}
    bounded_route_completion = (
        result.get("termination_reason")
        == "max_steps exceeded; using the last validated route"
    )
    public = {
        "schema": LOOP_SCHEMA,
        "status": _short(result.get("status"), 40),
        "termination_reason": _short(result.get("termination_reason"), 300),
        "steps": int(result.get("steps", 0) or 0),
        "model_call_count": model_calls,
        "tool_call_count": tool_calls,
        "retry_count": retries,
        "selected_skill_id": _short(state.get("selected_skill_id"), 120),
        "supporting_skill_ids": [
            _short(item, 120)
            for item in (state.get("supporting_skill_ids", []) or [])[:2]
        ],
        "next_focus": _short(state.get("next_focus"), 40),
        "deterministic_fallback": bool(result.get("deterministic_fallback")),
        "bounded_route_completion": bounded_route_completion,
        "events": events[-32:],
    }
    public["trace_sha256"] = canonical_sha256(public)
    return public


def _public_model_trace(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    allowed = (
        "provider",
        "model",
        "request_kind",
        "latency_ms",
        "attempt_count",
        "http_status",
        "response_id",
        "usage",
    )
    result = {key: deepcopy(value[key]) for key in allowed if key in value}
    usage = result.get("usage")
    if not isinstance(usage, Mapping):
        result.pop("usage", None)
    else:
        result["usage"] = {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in usage and isinstance(usage[key], (int, float))
        }
    return result


def _error_type(value: Any) -> str:
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", str(value or ""))
    return match.group(1)[:80] if match else "ModelBoundaryError"
