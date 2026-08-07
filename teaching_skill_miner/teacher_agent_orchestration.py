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

from .teacher_agent import PRIMARY_ROLES, canonical_sha256, validate_skill_library
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


def build_turn_lifecycle_receipt(
    session: Mapping[str, Any],
    *,
    loop_trace: Mapping[str, Any] | None,
    plan: Mapping[str, Any] | None,
    output_action: Mapping[str, Any] | None = None,
    verification_status: str = "passed",
    fallback_reason: str = "",
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
    route = candidate.get("decision", {})
    if not isinstance(route, Mapping):
        route = {}
    action = dict(output_action) if isinstance(output_action, Mapping) else {}
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
