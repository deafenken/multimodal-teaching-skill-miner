"""Recoverable Goal→Plan→Execute→Verify→Reflect Teaching Agent workflow.

``teacher_agent_loop`` provides the bounded model/tool loop.  This module adds
the product-level lifecycle around it: an explicit goal contract, a persisted
checkpoint after every phase, deterministic verification, uncertainty-aware
reflection, and bounded replanning.  Checkpoints contain hashes and public
receipts only; prompts, raw learner turns, credentials, and tool payloads are
never persisted here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Mapping, Sequence

from .teacher_agent import (
    PRIMARY_ROLES,
    SIGNALS,
    canonical_sha256,
    validate_skill_library,
)
from .teacher_agent_loop import (
    StructuredModel,
    TeachingAgentLoopOptions,
    public_agent_loop_trace,
    run_teaching_agent_loop,
)


ORCHESTRATION_SCHEMA = "teaching_skill_miner.teacher_agent_orchestration.v1"
ORCHESTRATION_STATE_VERSION = 1
_PHASES = frozenset({"goal", "plan", "execute", "verify", "reflect"})
_STATUSES = frozenset(
    {"running", "waiting_for_learner", "completed", "handoff_required", "blocked"}
)
_FOCUS = frozenset({"prerequisite", "conceptual", "procedural", "transfer"})
_TURN_LIFECYCLE_EVENTS = frozenset(
    {"observe", "assess", "route", "act", "commit", "abort"}
)
_TURN_OUTCOME_ALIASES = {
    "commit": "commit",
    "committed": "commit",
    "completed": "commit",
    "abort": "abort",
    "aborted": "abort",
    "cancel": "abort",
    "cancelled": "abort",
    "canceled": "abort",
    "pending": "pending",
    "": "pending",
}
_EVENT_ALIASES = {
    "observation": "observe",
    "observed": "observe",
    "assessment": "assess",
    "assessed": "assess",
    "routing": "route",
    "routed": "route",
    "action": "act",
    "acted": "act",
    "committed": "commit",
    "turn_committed": "commit",
    "aborted": "abort",
    "turn_aborted": "abort",
}


class TeacherAgentOrchestrationError(ValueError):
    """Raised when a workflow or checkpoint violates the durable contract."""


@dataclass(frozen=True, slots=True)
class TeacherAgentOrchestrationOptions:
    """Bounded replanning and uncertainty policy for one teaching turn."""

    max_replans: int = 1
    replan_on_deterministic_fallback: bool = True
    human_review_uncertainty_threshold: float = 0.55
    max_phase_transitions: int = 16

    def validated(self) -> "TeacherAgentOrchestrationOptions":
        if isinstance(self.max_replans, bool) or not 0 <= self.max_replans <= 4:
            raise TeacherAgentOrchestrationError("max_replans must be in [0, 4]")
        if not isinstance(self.replan_on_deterministic_fallback, bool):
            raise TeacherAgentOrchestrationError(
                "replan_on_deterministic_fallback must be a boolean"
            )
        if (
            isinstance(self.human_review_uncertainty_threshold, bool)
            or not isinstance(self.human_review_uncertainty_threshold, (int, float))
            or not math.isfinite(float(self.human_review_uncertainty_threshold))
            or not 0 <= float(self.human_review_uncertainty_threshold) <= 1
        ):
            raise TeacherAgentOrchestrationError(
                "human_review_uncertainty_threshold must be in [0, 1]"
            )
        if (
            isinstance(self.max_phase_transitions, bool)
            or not 5 <= self.max_phase_transitions <= 64
        ):
            raise TeacherAgentOrchestrationError(
                "max_phase_transitions must be in [5, 64]"
            )
        return self


def _short(value: Any, maximum: int = 400) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())[:maximum]


def _checkpoint_material(value: Mapping[str, Any]) -> dict[str, Any]:
    material = deepcopy(dict(value))
    material.pop("checkpoint_sha256", None)
    return material


def _seal(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(checkpoint))
    result["checkpoint_sha256"] = canonical_sha256(_checkpoint_material(result))
    return result


def _validate_checkpoint(
    checkpoint: Mapping[str, Any], session: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TeacherAgentOrchestrationError("checkpoint must be an object")
    current = deepcopy(dict(checkpoint))
    if current.get("schema") != ORCHESTRATION_SCHEMA:
        raise TeacherAgentOrchestrationError("checkpoint schema is invalid")
    if current.get("state_version") != ORCHESTRATION_STATE_VERSION:
        raise TeacherAgentOrchestrationError("checkpoint state_version is unsupported")
    if current.get("phase") not in _PHASES:
        raise TeacherAgentOrchestrationError("checkpoint phase is invalid")
    if current.get("status") not in _STATUSES:
        raise TeacherAgentOrchestrationError("checkpoint status is invalid")
    expected_sha = canonical_sha256(_checkpoint_material(current))
    if current.get("checkpoint_sha256") != expected_sha:
        raise TeacherAgentOrchestrationError("checkpoint integrity check failed")
    if current.get("session_fingerprint") != canonical_sha256(dict(session)):
        raise TeacherAgentOrchestrationError(
            "checkpoint belongs to a different teaching-session snapshot"
        )
    if not isinstance(current.get("events"), list):
        raise TeacherAgentOrchestrationError("checkpoint events are invalid")
    return current


def initialize_teacher_agent_orchestration(
    session: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a sealed checkpoint without sending any learner data remotely."""

    if not isinstance(session, Mapping):
        raise TeacherAgentOrchestrationError("session must be an object")
    library = session.get("skill_library")
    if not isinstance(library, Mapping):
        raise TeacherAgentOrchestrationError("session.skill_library is required")
    try:
        validate_skill_library(library)
    except Exception as exc:
        raise TeacherAgentOrchestrationError("skill_library is invalid") from exc
    return _seal(
        {
            "schema": ORCHESTRATION_SCHEMA,
            "state_version": ORCHESTRATION_STATE_VERSION,
            "status": "running",
            "phase": "goal",
            "cycle": 1,
            "replan_count": 0,
            "session_fingerprint": canonical_sha256(dict(session)),
            "goal_contract": None,
            "plan": None,
            "execution": None,
            "verification": None,
            "reflection": None,
            "output_action": None,
            "uncertainty": {
                "score": 1.0,
                "level": "high",
                "sources": ["workflow_not_started"],
                "needs_human_review": False,
            },
            "events": [],
            "claim_boundary": {
                "model_reasoning_persisted": False,
                "raw_learner_text_persisted": False,
                "tool_payloads_persisted": False,
                "public_loop_receipt_only": True,
                "learning_effect_established": False,
            },
        }
    )


def _event(checkpoint: dict[str, Any], phase: str, outcome: str, reason: str) -> None:
    checkpoint["events"].append(
        {
            "sequence": len(checkpoint["events"]) + 1,
            "phase": phase,
            "cycle": int(checkpoint.get("cycle", 1)),
            "outcome": _short(outcome, 80),
            "reason": _short(reason, 240),
        }
    )
    checkpoint["events"] = checkpoint["events"][-64:]


def _goal_contract(session: Mapping[str, Any]) -> dict[str, Any]:
    goal = session.get("goal", {})
    if not isinstance(goal, Mapping):
        raise TeacherAgentOrchestrationError("session goal is invalid")
    objective = _short(goal.get("objective"), 600)
    concept = _short(goal.get("concept"), 240)
    if not objective or not concept:
        raise TeacherAgentOrchestrationError("goal concept and objective are required")
    raw_thresholds = goal.get("success_thresholds", {})
    thresholds: dict[str, float] = {}
    if isinstance(raw_thresholds, Mapping):
        for key in _FOCUS:
            value = raw_thresholds.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0 <= float(value) <= 1
            ):
                thresholds[key] = float(value)
    return {
        "concept": concept,
        "objective": objective,
        "knowledge_components": [
            _short(item, 120)
            for item in (goal.get("knowledge_components", []) or [])[:24]
            if str(item).strip()
        ],
        "completion_conditions": {
            "mastery_thresholds": thresholds,
            "latest_signal_must_be_correct": True,
            "active_misconception_must_be_absent": True,
            "model_recommendation_alone_is_insufficient": True,
        },
        "maximum_rounds": int(goal.get("max_rounds", 0) or 0),
    }


def _skill_index(session: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["skill_id"]): deepcopy(dict(item))
        for item in session["skill_library"]["skills"]
        if isinstance(item, Mapping)
    }


def _render_route_action(
    session: Mapping[str, Any], route: Mapping[str, Any]
) -> dict[str, Any]:
    skills = _skill_index(session)
    skill_id = _short(route.get("selected_skill_id"), 120)
    skill = skills.get(skill_id)
    if skill is None or skill.get("role") not in PRIMARY_ROLES:
        raise TeacherAgentOrchestrationError("route selected an invalid primary Skill")
    goal = session.get("goal", {})
    materials = goal.get("materials", {}) if isinstance(goal, Mapping) else {}
    if not isinstance(materials, Mapping):
        materials = {}
    values = {
        "concept": _short(goal.get("concept"), 180),
        "example": _short(materials.get("example"), 300),
        "practice": _short(materials.get("practice"), 300),
        "transfer_task": _short(materials.get("transfer_task"), 300),
        "misconception": "当前回答中的关键混淆点",
        "learner_level": _short(
            (session.get("student_profile", {}) or {}).get("learner_level"), 60
        )
        if isinstance(session.get("student_profile"), Mapping)
        else "",
    }
    message = str(skill.get("message_template", "请用自己的话解释当前概念。"))
    for key, value in values.items():
        message = message.replace("{" + key + "}", value)
    return {
        "selected_skill_id": skill_id,
        "supporting_skill_ids": [
            _short(item, 120)
            for item in (route.get("supporting_skill_ids", []) or [])[:2]
        ],
        "next_focus": _short(route.get("next_focus"), 40),
        "action_type": _short(skill.get("action_type"), 120),
        "message": _short(message, 1400),
        "expected_signal": _short(skill.get("expected_signal"), 600),
        "reason": "validated route materialized from the selected Skill template",
        "skill": {
            "skill_id": skill_id,
            "name": _short(skill.get("name"), 120),
            "action_type": _short(skill.get("action_type"), 120),
        },
    }


def _verification(
    session: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    execution = checkpoint.get("execution", {})
    if not isinstance(execution, Mapping):
        return {
            "status": "replan_required",
            "checks": [],
            "failures": ["execution_missing"],
        }
    outcome = execution.get("outcome")
    if outcome == "completed":
        return {
            "status": "completed",
            "checks": ["local_completion_contract_passed"],
            "failures": [],
        }
    if outcome == "handoff_required":
        return {
            "status": "handoff_required",
            "checks": ["explicit_handoff_outcome"],
            "failures": [],
        }
    action = execution.get("action")
    if not isinstance(action, Mapping):
        return {
            "status": "replan_required",
            "checks": [],
            "failures": ["action_missing"],
        }
    skills = _skill_index(session)
    skill_id = _short(action.get("selected_skill_id"), 120)
    skill = skills.get(skill_id)
    failures: list[str] = []
    checks: list[str] = []
    if skill is None or skill.get("role") not in PRIMARY_ROLES:
        failures.append("primary_skill_invalid")
    else:
        checks.append("primary_skill_allowlisted")
        if action.get("action_type") != skill.get("action_type"):
            failures.append("action_type_does_not_match_skill")
        else:
            checks.append("action_type_matches_skill")
    if not _short(action.get("message"), 1401):
        failures.append("teacher_message_empty")
    elif len(str(action.get("message"))) > 1400:
        failures.append("teacher_message_budget_exceeded")
    else:
        checks.append("teacher_message_bounded")
    if action.get("next_focus") not in _FOCUS:
        failures.append("next_focus_invalid")
    else:
        checks.append("next_focus_valid")
    return {
        "status": "passed" if not failures else "replan_required",
        "checks": checks,
        "failures": failures,
    }


def _uncertainty(
    loop_receipt: Mapping[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    score = 0.08
    sources: list[str] = []
    if loop_receipt.get("deterministic_fallback"):
        score += 0.58
        sources.append("deterministic_fallback_used")
    error_count = sum(
        event.get("type") in {"model_error", "tool_error"}
        for event in (loop_receipt.get("events", []) or [])
        if isinstance(event, Mapping)
    )
    guard_count = sum(
        event.get("type") == "guard"
        for event in (loop_receipt.get("events", []) or [])
        if isinstance(event, Mapping)
    )
    if error_count:
        score += min(0.22, 0.08 * error_count)
        sources.append("model_or_tool_error")
    if guard_count:
        score += min(0.18, 0.06 * guard_count)
        sources.append("runtime_guard_triggered")
    if verification.get("status") == "replan_required":
        score += 0.4
        sources.append("verification_failed")
    if loop_receipt.get("status") == "route_ready":
        score += 0.08
        sources.append("route_template_materialization")
    score = round(min(1.0, score), 3)
    level = "low" if score < 0.3 else "medium" if score < 0.55 else "high"
    return {
        "score": score,
        "level": level,
        "sources": sources or ["validated_model_and_runtime_path"],
    }


def advance_teacher_agent_orchestration(
    checkpoint: Mapping[str, Any],
    session: Mapping[str, Any],
    client: StructuredModel | Callable[[Sequence[Mapping[str, str]]], Any],
    *,
    options: TeacherAgentOrchestrationOptions | None = None,
    loop_options: TeachingAgentLoopOptions | None = None,
    outbound_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance exactly one durable workflow phase."""

    options = (options or TeacherAgentOrchestrationOptions()).validated()
    current = _validate_checkpoint(checkpoint, session)
    if current["status"] != "running":
        return current
    phase = current["phase"]
    if phase == "goal":
        current["goal_contract"] = _goal_contract(session)
        _event(current, "goal", "accepted", "goal contract and completion conditions bound")
        current["phase"] = "plan"
    elif phase == "plan":
        result = run_teaching_agent_loop(
            session,
            client,
            options=loop_options,
            outbound_context=outbound_context,
        )
        receipt = public_agent_loop_trace(result)
        state = result.get("loop_state", {})
        if not isinstance(state, Mapping):
            state = {}
        current["plan"] = {
            "loop_receipt": receipt,
            "selected_skill_id": _short(state.get("selected_skill_id"), 120),
            "supporting_skill_ids": [
                _short(item, 120)
                for item in (state.get("supporting_skill_ids", []) or [])[:2]
            ],
            "next_focus": _short(state.get("next_focus"), 40),
            "candidate_action": deepcopy(result.get("action"))
            if isinstance(result.get("action"), Mapping)
            else None,
        }
        _event(current, "plan", receipt.get("status", "unknown"), "bounded agent loop completed")
        current["phase"] = "execute"
    elif phase == "execute":
        plan = current.get("plan", {})
        receipt = plan.get("loop_receipt", {}) if isinstance(plan, Mapping) else {}
        loop_status = receipt.get("status") if isinstance(receipt, Mapping) else None
        if loop_status == "succeeded":
            execution = {"outcome": "completed", "action": None, "source": "verified_loop_termination"}
        elif loop_status == "terminated_unable":
            execution = {"outcome": "handoff_required", "action": None, "source": "loop_handoff"}
        else:
            candidate = plan.get("candidate_action") if isinstance(plan, Mapping) else None
            action = (
                deepcopy(dict(candidate))
                if isinstance(candidate, Mapping)
                else _render_route_action(session, plan)
            )
            candidate_skill = action.get("skill", {})
            if (
                not action.get("action_type")
                and isinstance(candidate_skill, Mapping)
            ):
                action["action_type"] = _short(
                    candidate_skill.get("action_type"), 120
                )
            execution = {
                "outcome": "action_ready",
                "action": action,
                "source": "model_action" if isinstance(candidate, Mapping) else "validated_skill_template",
                "action_sha256": canonical_sha256(action),
            }
        current["execution"] = execution
        _event(current, "execute", execution["outcome"], execution["source"])
        current["phase"] = "verify"
    elif phase == "verify":
        verification = _verification(session, current)
        current["verification"] = verification
        _event(
            current,
            "verify",
            verification["status"],
            ",".join(verification.get("failures", [])) or "all deterministic checks passed",
        )
        current["phase"] = "reflect"
    elif phase == "reflect":
        plan = current.get("plan", {})
        receipt = plan.get("loop_receipt", {}) if isinstance(plan, Mapping) else {}
        verification = current.get("verification", {})
        uncertainty = _uncertainty(receipt, verification)
        needs_review = uncertainty["score"] >= float(
            options.human_review_uncertainty_threshold
        )
        uncertainty["needs_human_review"] = needs_review
        current["uncertainty"] = uncertainty
        verification_status = verification.get("status")
        replan_reason = ""
        if verification_status == "replan_required":
            replan_reason = "deterministic verification rejected the execution"
        elif (
            options.replan_on_deterministic_fallback
            and receipt.get("deterministic_fallback")
        ):
            replan_reason = "agent loop used a deterministic fallback"
        can_replan = int(current.get("replan_count", 0)) < options.max_replans
        if replan_reason and can_replan:
            current["replan_count"] = int(current.get("replan_count", 0)) + 1
            current["cycle"] = int(current.get("cycle", 1)) + 1
            current["reflection"] = {
                "outcome": "replan",
                "reason": replan_reason,
                "uncertainty": deepcopy(uncertainty),
            }
            _event(current, "reflect", "replan", replan_reason)
            current["plan"] = None
            current["execution"] = None
            current["verification"] = None
            current["phase"] = "plan"
        elif verification_status == "completed":
            current["status"] = "completed"
            current["reflection"] = {
                "outcome": "completed",
                "reason": "local completion contract and loop termination passed",
                "uncertainty": deepcopy(uncertainty),
            }
            _event(current, "reflect", "completed", current["reflection"]["reason"])
        elif verification_status == "handoff_required":
            current["status"] = "handoff_required"
            current["reflection"] = {
                "outcome": "handoff_required",
                "reason": "runtime requested bounded human handoff",
                "uncertainty": deepcopy(uncertainty),
            }
            _event(current, "reflect", "handoff_required", current["reflection"]["reason"])
        elif verification_status == "passed":
            current["status"] = "waiting_for_learner"
            current["output_action"] = deepcopy(current["execution"]["action"])
            current["reflection"] = {
                "outcome": "action_ready",
                "reason": (
                    "verified action available; human review recommended"
                    if needs_review
                    else "verified action available"
                ),
                "uncertainty": deepcopy(uncertainty),
                "replan_exhausted": bool(replan_reason and not can_replan),
            }
            _event(current, "reflect", "waiting_for_learner", current["reflection"]["reason"])
        else:
            current["status"] = "blocked"
            current["reflection"] = {
                "outcome": "blocked",
                "reason": replan_reason or "workflow reached an unsupported verification state",
                "uncertainty": deepcopy(uncertainty),
            }
            _event(current, "reflect", "blocked", current["reflection"]["reason"])
    return _seal(current)


def run_teacher_agent_orchestration(
    session: Mapping[str, Any],
    client: StructuredModel | Callable[[Sequence[Mapping[str, str]]], Any],
    *,
    checkpoint: Mapping[str, Any] | None = None,
    options: TeacherAgentOrchestrationOptions | None = None,
    loop_options: TeachingAgentLoopOptions | None = None,
    outbound_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run or resume a workflow until it waits, completes, hands off or blocks."""

    options = (options or TeacherAgentOrchestrationOptions()).validated()
    current = (
        initialize_teacher_agent_orchestration(session)
        if checkpoint is None
        else _validate_checkpoint(checkpoint, session)
    )
    for _ in range(options.max_phase_transitions):
        if current["status"] != "running":
            return current
        current = advance_teacher_agent_orchestration(
            current,
            session,
            client,
            options=options,
            loop_options=loop_options,
            outbound_context=outbound_context,
        )
    current["status"] = "blocked"
    current["reflection"] = {
        "outcome": "blocked",
        "reason": "maximum phase transitions exceeded",
        "uncertainty": deepcopy(current.get("uncertainty", {})),
    }
    _event(current, current["phase"], "blocked", "maximum phase transitions exceeded")
    return _seal(current)


def _lifecycle_code(value: Any, maximum: int = 120) -> str:
    """Return a bounded, non-prose identifier suitable for a public receipt."""

    candidate = str(value or "").strip()
    if not candidate or len(candidate) > maximum:
        return ""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", candidate) is None:
        return ""
    return candidate


def _lifecycle_codes(value: Any, maximum_items: int = 12) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[str] = []
    for item in value[:maximum_items]:
        code = _lifecycle_code(item, 80)
        if code and code not in result:
            result.append(code)
    return result


def _lifecycle_fact(event: Mapping[str, Any], key: str) -> Any:
    if key in event:
        return event[key]
    for container_key in ("facts", "data"):
        container = event.get(container_key)
        if isinstance(container, Mapping) and key in container:
            return container[key]
    return None


def _lifecycle_event_name(event: Mapping[str, Any]) -> str:
    raw = (
        event.get("event")
        or event.get("event_type")
        or event.get("type")
        or ""
    )
    name = str(raw).strip().lower()
    name = _EVENT_ALIASES.get(name, name)
    if name not in _TURN_LIFECYCLE_EVENTS:
        raise TeacherAgentOrchestrationError(
            "lifecycle event must be observe, assess, route, act, commit, or abort"
        )
    return name


def _nonnegative_lifecycle_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _lifecycle_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        return None
    return round(result, 4)


def _lifecycle_sha256(value: Any) -> str | None:
    candidate = str(value or "").strip().lower()
    return candidate if re.fullmatch(r"[0-9a-f]{64}", candidate) else None


def _turn_route_facts(
    loop: Mapping[str, Any],
    plan: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, Any]:
    decision = plan.get("decision", {})
    if not isinstance(decision, Mapping):
        decision = {}
    action_primary = action.get("primary_skill", {})
    if not isinstance(action_primary, Mapping):
        action_primary = {}
    selected = _short(
        action_primary.get("skill_id")
        or decision.get("primary_skill_id")
        or loop.get("selected_skill_id"),
        120,
    )
    supporting = decision.get(
        "supporting_skill_ids", loop.get("supporting_skill_ids", [])
    )
    if not isinstance(supporting, list):
        supporting = []
    if isinstance(action.get("supporting_skills"), list):
        action_supporting = [
            item.get("skill_id")
            for item in action["supporting_skills"]
            if isinstance(item, Mapping) and item.get("skill_id")
        ]
        supporting = action_supporting
    adjudication = decision.get("route_adjudication", {})
    if not isinstance(adjudication, Mapping):
        adjudication = {}
    provenance = decision.get("action_provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    if not adjudication and isinstance(provenance.get("route_adjudication"), Mapping):
        adjudication = provenance["route_adjudication"]
    if decision.get("manual_override_applied") is True:
        authority = "manual_override"
        authority_source = "validated_plan"
    elif adjudication.get("enabled") is True:
        authority = "state_first_adjudicator"
        authority_source = "validated_plan"
    elif decision.get("agent_loop_route_applied") is True:
        authority = "validated_agent_loop"
        authority_source = "validated_plan"
    elif (
        loop.get("status") == "route_ready"
        and selected
        and selected == _short(loop.get("selected_skill_id"), 120)
    ):
        authority = "validated_agent_loop"
        authority_source = "loop_receipt"
    elif plan:
        authority = "validated_plan"
        authority_source = "validated_plan"
    elif action.get("decision_origin") == "deterministic_safety_fallback":
        authority = "deterministic_safety_fallback"
        authority_source = "output_action"
    else:
        authority = "unresolved"
        authority_source = "unresolved"
    return {
        "selected_skill_id": selected or None,
        "supporting_skill_ids": [
            code
            for item in supporting[:2]
            if (code := _lifecycle_code(item, 120))
        ],
        "next_focus": (
            _short(
            action.get("next_focus")
            or decision.get("next_focus")
            or loop.get("next_focus"),
            40,
        )
            or None
        ),
        "authority": authority,
        "authority_source": authority_source,
        "route_changed": bool(
            adjudication.get("changed")
            or decision.get("primary_skill_was_retargeted")
        ),
        "reason_codes": _lifecycle_codes(adjudication.get("reason_codes", [])),
    }


def _build_event_turn_lifecycle_receipt(
    session: Mapping[str, Any],
    *,
    loop: Mapping[str, Any],
    plan: Mapping[str, Any],
    action: Mapping[str, Any],
    lifecycle_events: Sequence[Mapping[str, Any]],
    route_authority: str | None,
    turn_outcome: str | None,
    commit_round: int | None,
    fallback_reason: str,
) -> dict[str, Any]:
    """Build a fail-closed receipt from an allowlisted lifecycle event stream."""

    if isinstance(lifecycle_events, (str, bytes, bytearray, Mapping)):
        raise TeacherAgentOrchestrationError("lifecycle_events must be a sequence")
    raw_events = list(lifecycle_events)
    if any(not isinstance(item, Mapping) for item in raw_events):
        raise TeacherAgentOrchestrationError("each lifecycle event must be an object")
    requested_outcome = _TURN_OUTCOME_ALIASES.get(
        str(turn_outcome or "").strip().lower()
    )
    if requested_outcome is None:
        raise TeacherAgentOrchestrationError(
            "turn_outcome must be commit, abort, or pending"
        )
    explicit_authority = _lifecycle_code(route_authority, 80)
    if route_authority is not None and not explicit_authority:
        raise TeacherAgentOrchestrationError(
            "route_authority must be a bounded non-prose identifier"
        )
    explicit_commit_round = (
        _nonnegative_lifecycle_integer(commit_round)
        if commit_round is not None
        else None
    )
    if commit_round is not None and explicit_commit_round is None:
        raise TeacherAgentOrchestrationError(
            "commit_round must be a non-negative integer"
        )
    event_names = [_lifecycle_event_name(item) for item in raw_events]
    terminal_names = [
        name for name in event_names if name in {"commit", "abort"}
    ]
    if requested_outcome in {"commit", "abort"} and not terminal_names:
        terminal: dict[str, Any] = {"event": requested_outcome}
        if requested_outcome == "commit":
            terminal["committed"] = True
            if explicit_commit_round is not None:
                terminal["round"] = explicit_commit_round
        else:
            terminal["aborted"] = True
        raw_events.append(terminal)

    route_defaults = _turn_route_facts(loop, plan, action)
    skills = _skill_index(session)
    projected: list[dict[str, Any]] = []
    failures: list[str] = []
    checks: list[str] = []
    seen_observe = False
    seen_assess = False
    route_ready = False
    act_ready = False
    terminal_event: str | None = None
    terminal_valid = False
    route_records: list[dict[str, Any]] = []
    explicit_replan_counts: list[int] = []
    replan_reason_codes: list[str] = list(route_defaults["reason_codes"])
    final_action_sha256: str | None = None
    final_action_type: str | None = None
    final_commit_round: int | None = None

    def reject(event_receipt: dict[str, Any], reason: str) -> None:
        event_receipt["status"] = "rejected"
        event_receipt["failure_code"] = reason
        failures.append(reason)

    for raw_event in raw_events:
        name = _lifecycle_event_name(raw_event)
        event_receipt: dict[str, Any] = {
            "sequence": len(projected) + 1,
            "event": name,
        }
        if terminal_event is not None:
            reject(event_receipt, "event_after_terminal_outcome")
        elif name == "observe":
            observation = _lifecycle_fact(raw_event, "observation")
            observed_flags = [
                _lifecycle_fact(raw_event, key)
                for key in (
                    "observed",
                    "observation_present",
                    "learner_turn_observed",
                    "evidence_present",
                )
            ]
            counts = [
                _nonnegative_lifecycle_integer(_lifecycle_fact(raw_event, key))
                for key in ("observation_count", "evidence_count")
            ]
            observation_count = max((item for item in counts if item is not None), default=0)
            observation_present = bool(
                (isinstance(observation, Mapping) and bool(observation))
                or any(item is True for item in observed_flags)
                or observation_count > 0
            )
            event_receipt["observation_present"] = observation_present
            event_receipt["evidence_count"] = observation_count
            source = _lifecycle_code(_lifecycle_fact(raw_event, "source"), 80)
            if source:
                event_receipt["source"] = source
            if seen_observe:
                reject(event_receipt, "duplicate_observe_event")
            elif not observation_present:
                reject(event_receipt, "observation_fact_missing")
            else:
                seen_observe = True
                event_receipt["status"] = "observed"
                checks.append("observation_fact_recorded")
        elif name == "assess":
            assessment = _lifecycle_fact(raw_event, "assessment")
            if not isinstance(assessment, Mapping):
                assessment = {}
            signal = str(
                _lifecycle_fact(raw_event, "signal") or assessment.get("signal") or ""
            ).strip()
            confidence = _lifecycle_probability(
                _lifecycle_fact(raw_event, "confidence")
                if _lifecycle_fact(raw_event, "confidence") is not None
                else assessment.get("confidence")
            )
            event_receipt["signal"] = signal if signal in SIGNALS | {"not_observed"} else None
            event_receipt["confidence"] = confidence
            review_fact = _lifecycle_fact(raw_event, "needs_human_review")
            if review_fact is None:
                review_fact = assessment.get("needs_human_review", False)
            event_receipt["needs_human_review"] = (
                review_fact if isinstance(review_fact, bool) else False
            )
            source = _lifecycle_code(
                _lifecycle_fact(raw_event, "assessment_source")
                or assessment.get("assessment_source")
                or _lifecycle_fact(raw_event, "source"),
                80,
            )
            if source:
                event_receipt["source"] = source
            if not seen_observe:
                reject(event_receipt, "assessment_before_observation")
            elif seen_assess:
                reject(event_receipt, "duplicate_assess_event")
            elif signal not in SIGNALS | {"not_observed"} or confidence is None:
                reject(event_receipt, "assessment_fact_invalid")
            else:
                seen_assess = True
                event_receipt["status"] = "assessed"
                checks.append("assessment_signal_and_confidence_valid")
        elif name == "route":
            route = _lifecycle_fact(raw_event, "route")
            if not isinstance(route, Mapping):
                route = {}
            selected = _lifecycle_code(
                _lifecycle_fact(raw_event, "selected_skill_id")
                or route.get("selected_skill_id")
                or route.get("primary_skill_id")
                or route_defaults["selected_skill_id"],
                120,
            )
            event_authority = _lifecycle_code(
                _lifecycle_fact(raw_event, "route_authority")
                or route.get("authority")
                or explicit_authority
                or route_defaults["authority"],
                80,
            )
            supports = _lifecycle_fact(raw_event, "supporting_skill_ids")
            if supports is None:
                supports = route.get(
                    "supporting_skill_ids", route_defaults["supporting_skill_ids"]
                )
            if not isinstance(supports, list):
                supports = []
            raw_support_count = len(supports[:2])
            supports = [
                code
                for item in supports[:2]
                if (code := _lifecycle_code(item, 120))
            ]
            support_identifier_invalid = len(supports) != raw_support_count
            event_receipt.update(
                {
                    "selected_skill_id": selected or None,
                    "supporting_skill_ids": supports,
                    "route_authority": event_authority or None,
                }
            )
            event_reason_codes = _lifecycle_codes(
                _lifecycle_fact(raw_event, "reason_codes")
                or route.get("reason_codes", [])
            )
            if event_reason_codes:
                event_receipt["reason_codes"] = event_reason_codes
                replan_reason_codes.extend(event_reason_codes)
            event_replan_count = _nonnegative_lifecycle_integer(
                _lifecycle_fact(raw_event, "replan_count")
            )
            if event_replan_count is not None:
                explicit_replan_counts.append(event_replan_count)
            skill = skills.get(selected)
            authority_conflict = bool(
                explicit_authority
                and event_authority
                and explicit_authority != event_authority
            )
            supports_valid = all(
                item in skills
                and skills[item].get("role") == "support"
                and item in set(skill.get("supporting_skill_ids", []))
                for item in supports
            ) if isinstance(skill, Mapping) else False
            if not seen_assess:
                reject(event_receipt, "route_before_assessment")
            elif skill is None or skill.get("role") not in PRIMARY_ROLES:
                reject(event_receipt, "route_primary_skill_invalid")
            elif not event_authority or event_authority == "unresolved":
                reject(event_receipt, "route_authority_unresolved")
            elif authority_conflict:
                reject(event_receipt, "route_authority_conflict")
            elif support_identifier_invalid:
                reject(event_receipt, "route_support_identifier_invalid")
            elif not supports_valid:
                reject(event_receipt, "route_support_contract_invalid")
            else:
                route_ready = True
                act_ready = False
                event_receipt["status"] = "routed"
                event_receipt["contract_validated"] = True
                route_records.append(
                    {
                        "selected_skill_id": selected,
                        "authority": event_authority,
                        "supporting_skill_ids": supports,
                    }
                )
                checks.append("route_skill_and_authority_valid")
        elif name == "act":
            event_action = _lifecycle_fact(raw_event, "action")
            if not isinstance(event_action, Mapping):
                event_action = {}
            selected = _lifecycle_code(
                _lifecycle_fact(raw_event, "selected_skill_id")
                or event_action.get("selected_skill_id")
                or (
                    event_action.get("primary_skill", {}).get("skill_id")
                    if isinstance(event_action.get("primary_skill"), Mapping)
                    else None
                )
                or (route_records[-1]["selected_skill_id"] if route_records else None),
                120,
            )
            action_type = _lifecycle_code(
                _lifecycle_fact(raw_event, "action_type")
                or event_action.get("action_type")
                or event_action.get("type")
                or (
                    action.get("type")
                    if selected == route_defaults["selected_skill_id"]
                    else None
                ),
                100,
            )
            materialized = bool(
                event_action
                or _lifecycle_fact(raw_event, "action_materialized") is True
                or (
                    action
                    and selected == route_defaults["selected_skill_id"]
                    and action_type
                )
            )
            skill = skills.get(selected)
            event_receipt.update(
                {
                    "selected_skill_id": selected or None,
                    "action_type": action_type or None,
                    "action_materialized": materialized,
                }
            )
            if not route_ready:
                reject(event_receipt, "action_before_valid_route")
            elif not materialized or not action_type:
                reject(event_receipt, "action_fact_missing")
            elif skill is None or action_type != skill.get("action_type"):
                reject(event_receipt, "action_does_not_match_selected_skill")
            elif not route_records or selected != route_records[-1]["selected_skill_id"]:
                reject(event_receipt, "action_does_not_match_latest_route")
            else:
                action_material = event_action or {
                    "selected_skill_id": selected,
                    "action_type": action_type,
                }
                final_action_sha256 = canonical_sha256(action_material)
                final_action_type = action_type
                event_receipt["action_sha256"] = final_action_sha256
                event_receipt["status"] = "materialized"
                act_ready = True
                checks.append("action_matches_latest_route_and_skill")
        elif name == "commit":
            event_round = _nonnegative_lifecycle_integer(
                _lifecycle_fact(raw_event, "round")
            )
            selected_round = (
                explicit_commit_round
                if explicit_commit_round is not None
                else event_round
                if event_round is not None
                else _nonnegative_lifecycle_integer(session.get("round"))
            )
            session_round = _nonnegative_lifecycle_integer(session.get("round"))
            final_commit_round = selected_round
            event_receipt["commit_round"] = selected_round
            event_receipt["session_round_matches"] = bool(
                selected_round is not None and selected_round == session_round
            )
            commit_flag = _lifecycle_fact(raw_event, "committed")
            final_route_matches = bool(
                route_records
                and (
                    not route_defaults["selected_skill_id"]
                    or route_records[-1]["selected_skill_id"]
                    == route_defaults["selected_skill_id"]
                )
                and (
                    not action
                    or route_records[-1]["supporting_skill_ids"]
                    == route_defaults["supporting_skill_ids"]
                )
            )
            output_action_type = _lifecycle_code(
                action.get("action_type") or action.get("type"), 100
            )
            if requested_outcome == "abort":
                reject(event_receipt, "turn_outcome_conflicts_with_commit_event")
            elif commit_flag is False:
                reject(event_receipt, "commit_event_negated")
            elif not act_ready:
                reject(event_receipt, "commit_before_valid_action")
            elif failures:
                reject(event_receipt, "commit_has_prior_lifecycle_failures")
            elif selected_round is None or selected_round != session_round:
                reject(event_receipt, "commit_round_does_not_match_session")
            elif not final_route_matches:
                reject(event_receipt, "committed_route_does_not_match_output")
            elif output_action_type and output_action_type != final_action_type:
                reject(event_receipt, "committed_action_does_not_match_output")
            else:
                event_receipt["status"] = "committed"
                event_receipt["committed"] = True
                terminal_valid = True
                checks.extend(
                    [
                        "ordered_observe_assess_route_act_complete",
                        "commit_round_bound_to_session",
                    ]
                )
            terminal_event = "commit"
        else:
            abort_flag = _lifecycle_fact(raw_event, "aborted")
            reason_codes = _lifecycle_codes(
                _lifecycle_fact(raw_event, "reason_codes")
                or _lifecycle_fact(raw_event, "reason_code")
            )
            if reason_codes:
                event_receipt["reason_codes"] = reason_codes
            if requested_outcome == "commit":
                reject(event_receipt, "turn_outcome_conflicts_with_abort_event")
            elif abort_flag is False:
                reject(event_receipt, "abort_event_negated")
            else:
                event_receipt["status"] = "aborted"
                event_receipt["aborted"] = True
                terminal_valid = True
                checks.append("abort_terminal_event_recorded")
            terminal_event = "abort"
        event_receipt["event_sha256"] = canonical_sha256(event_receipt)
        projected.append(event_receipt)

    route_count = len(route_records)
    route_changed = any(
        route_records[index]["selected_skill_id"]
        != route_records[index - 1]["selected_skill_id"]
        for index in range(1, route_count)
    )
    replan_count = max(
        [max(0, route_count - 1), int(route_defaults["route_changed"]), *explicit_replan_counts],
        default=0,
    )
    replan_reason_codes = list(dict.fromkeys(replan_reason_codes))[:12]
    if terminal_event == "commit" and terminal_valid and not failures:
        final_status = "completed"
        derived_outcome = "commit"
        verification_status = "verified"
    elif (
        terminal_event == "abort"
        and terminal_valid
        and not {
            "event_after_terminal_outcome",
            "turn_outcome_conflicts_with_abort_event",
            "abort_event_negated",
        }
        & set(failures)
    ):
        final_status = "aborted"
        derived_outcome = "abort"
        verification_status = "aborted"
    elif failures:
        final_status = "blocked"
        derived_outcome = "invalid"
        verification_status = "rejected"
    else:
        final_status = "running"
        derived_outcome = "pending"
        verification_status = "incomplete"
        failures.append("terminal_commit_or_abort_missing")
    verification = {
        "status": verification_status,
        "checks": list(dict.fromkeys(checks)),
        "failures": list(dict.fromkeys(failures)),
    }
    uncertainty = _uncertainty(
        loop,
        {
            "status": (
                "replan_required"
                if verification_status in {"rejected", "incomplete"}
                else verification_status
            )
        },
    )
    if replan_count:
        uncertainty["score"] = round(min(1.0, uncertainty["score"] + 0.08), 3)
        uncertainty["sources"] = list(
            dict.fromkeys([*uncertainty["sources"], "route_replanned"])
        )
    if fallback_reason:
        uncertainty["sources"] = list(
            dict.fromkeys([*uncertainty["sources"], "fallback_reason_recorded"])
        )
    uncertainty["level"] = (
        "low"
        if uncertainty["score"] < 0.3
        else "medium"
        if uncertainty["score"] < 0.55
        else "high"
    )
    uncertainty["needs_human_review"] = bool(
        verification_status in {"rejected", "incomplete"}
        or terminal_event == "abort"
        or fallback_reason
    )
    final_route = route_records[-1] if route_records else {}
    receipt = {
        "schema": ORCHESTRATION_SCHEMA,
        "state_version": ORCHESTRATION_STATE_VERSION,
        "artifact_kind": "committed_turn_lifecycle_receipt",
        "lifecycle_mode": "event_derived",
        "status": final_status,
        "phase": terminal_event or (projected[-1]["event"] if projected else "observe"),
        "turn_outcome": derived_outcome,
        "commit_round": final_commit_round,
        "session_fingerprint": canonical_sha256(dict(session)),
        "cycle": replan_count + 1,
        "replan_count": replan_count,
        "replan": {
            "occurred": bool(replan_count),
            "count": replan_count,
            "route_event_count": route_count,
            "route_changed": route_changed or bool(route_defaults["route_changed"]),
            "initial_skill_id": (
                route_records[0]["selected_skill_id"] if route_records else None
            ),
            "final_skill_id": final_route.get("selected_skill_id"),
            "reason_codes": replan_reason_codes,
        },
        "route_authority": final_route.get("authority")
        or explicit_authority
        or route_defaults["authority"],
        "route": {
            "authority": final_route.get("authority")
            or explicit_authority
            or route_defaults["authority"],
            "authority_source": (
                "lifecycle_event"
                if final_route
                else "argument"
                if explicit_authority
                else route_defaults["authority_source"]
            ),
            "selected_skill_id": final_route.get("selected_skill_id"),
            "supporting_skill_ids": route_defaults["supporting_skill_ids"],
            "contract_validated": bool(final_route),
        },
        "selected_skill_id": final_route.get("selected_skill_id"),
        "supporting_skill_ids": route_defaults["supporting_skill_ids"],
        "next_focus": route_defaults["next_focus"],
        "action_type": final_action_type,
        "action_sha256": final_action_sha256,
        "loop_trace_sha256": _lifecycle_sha256(loop.get("trace_sha256")),
        "events": projected,
        "phases": [
            {
                "phase": item["event"],
                "status": item["status"],
                "sequence": item["sequence"],
            }
            for item in projected
        ],
        "verification": verification,
        "uncertainty": uncertainty,
        "claim_boundary": {
            "model_reasoning_persisted": False,
            "raw_learner_text_persisted": False,
            "assessment_excerpt_persisted": False,
            "tool_payloads_persisted": False,
            "teacher_message_persisted": False,
            "input_events_persisted_verbatim": False,
            "allowlisted_event_projection_only": True,
            "commit_established": derived_outcome == "commit",
            "learning_effect_established": False,
        },
    }
    receipt["checkpoint_sha256"] = canonical_sha256(receipt)
    return receipt


def build_turn_lifecycle_receipt(
    session: Mapping[str, Any],
    *,
    loop_trace: Mapping[str, Any] | None,
    plan: Mapping[str, Any] | None,
    output_action: Mapping[str, Any] | None = None,
    verification_status: str = "passed",
    fallback_reason: str = "",
    lifecycle_events: Sequence[Mapping[str, Any]] | None = None,
    route_authority: str | None = None,
    turn_outcome: str | None = None,
    commit_round: int | None = None,
) -> dict[str, Any]:
    """Project an already-run turn into a compact five-phase receipt.

    The live runtime already performs the model/tool loop and final contract
    validation.  This adapter makes those phases visible without issuing a
    second model request or persisting the model prompt, student text, tool
    payload, or teacher message.
    """

    if not isinstance(session, Mapping):
        raise TeacherAgentOrchestrationError("session must be an object")
    loop = dict(loop_trace) if isinstance(loop_trace, Mapping) else {}
    candidate = dict(plan) if isinstance(plan, Mapping) else {}
    action = dict(output_action) if isinstance(output_action, Mapping) else {}
    if lifecycle_events is not None:
        return _build_event_turn_lifecycle_receipt(
            session,
            loop=loop,
            plan=candidate,
            action=action,
            lifecycle_events=lifecycle_events,
            route_authority=route_authority,
            turn_outcome=turn_outcome,
            commit_round=commit_round,
            fallback_reason=fallback_reason,
        )
    route = candidate.get("decision", {})
    if not isinstance(route, Mapping):
        route = {}
    primary = action.get("primary_skill", {})
    if not isinstance(primary, Mapping):
        primary = {}
    selected = _short(
        primary.get("skill_id") or route.get("primary_skill_id") or loop.get("selected_skill_id"),
        120,
    )
    supports = [
        _short(item, 120)
        for item in (
            primary.get("supporting_skill_ids")
            if isinstance(primary.get("supporting_skill_ids"), list)
            else route.get("supporting_skill_ids", loop.get("supporting_skill_ids", []))
        )
        or []
    ][:2]
    if not supports and isinstance(action.get("supporting_skills"), list):
        supports = [
            _short(item.get("skill_id"), 120)
            for item in action["supporting_skills"]
            if isinstance(item, Mapping) and item.get("skill_id")
        ][:2]
    verification = {"status": verification_status}
    uncertainty = _uncertainty(loop, verification)
    if fallback_reason:
        uncertainty["sources"] = list(dict.fromkeys([*uncertainty["sources"], "fallback_reason_recorded"]))
    uncertainty["needs_human_review"] = bool(
        uncertainty["level"] == "high" or fallback_reason
    )
    session_status = str(session.get("status", "active"))
    if session_status == "succeeded":
        final_status = "completed"
    elif session_status == "terminated_unable":
        final_status = "handoff_required"
    else:
        final_status = (
            "waiting_for_learner" if output_action is not None else "handoff_required"
        )
    phases = [
        {"phase": "goal", "status": "accepted", "reason": "session goal is bound locally"},
        {
            "phase": "plan",
            "status": _short(loop.get("status"), 40) or "not_run",
            "selected_skill_id": selected or None,
            "supporting_skill_ids": supports,
        },
        {
            "phase": "execute",
            "status": "action_ready" if output_action is not None else "fallback_or_handoff",
            "action_type": _short(action.get("type"), 100) or None,
        },
        {
            "phase": "verify",
            "status": _short(verification_status, 40),
            "checks": ["server_contract_validation"] if verification_status == "passed" else [],
        },
        {
            "phase": "reflect",
            "status": final_status,
            "reason": _short(fallback_reason, 240)
            or ("verified action available" if output_action is not None else "no executable action"),
        },
    ]
    receipt = {
        "schema": ORCHESTRATION_SCHEMA,
        "state_version": ORCHESTRATION_STATE_VERSION,
        "status": final_status,
        "phase": "reflect",
        "session_fingerprint": canonical_sha256(dict(session)),
        "cycle": 1,
        "replan_count": 0,
        "phases": phases,
        "selected_skill_id": selected or None,
        "supporting_skill_ids": supports,
        "next_focus": _short(
            action.get("next_focus") or route.get("next_focus") or loop.get("next_focus"),
            40,
        )
        or None,
        "action_sha256": canonical_sha256(action) if action else None,
        "loop_trace_sha256": _short(loop.get("trace_sha256"), 64) or None,
        "uncertainty": uncertainty,
        "claim_boundary": {
            "model_reasoning_persisted": False,
            "raw_learner_text_persisted": False,
            "tool_payloads_persisted": False,
            "teacher_message_persisted": False,
            "learning_effect_established": False,
        },
    }
    receipt["checkpoint_sha256"] = canonical_sha256(receipt)
    return receipt
