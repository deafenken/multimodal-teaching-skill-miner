"""Deterministic, stateful Teaching Agent for task two.

The module intentionally separates three responsibilities:

* a structured learner signal is supplied by a teacher or external judge;
* the Agent updates an explicit student model;
* a deterministic policy selects exactly one next teaching action.

It does not claim to grade arbitrary free-form answers or establish real learner
effects.  Those boundaries are written into every persisted session and report.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from typing import Any, Mapping


SESSION_SCHEMA = "teaching_skill_miner.teacher_agent_session.v1"
LIBRARY_SCHEMA = "teaching_skill_miner.teacher_agent_skill_library.v1"
LIBRARY_SCHEMA_V2 = "teaching_skill_miner.teacher_agent_skill_library.v2"
LIBRARY_SCHEMAS = frozenset({LIBRARY_SCHEMA, LIBRARY_SCHEMA_V2})
EVALUATION_SCHEMA = "teaching_skill_miner.teacher_agent_evaluation.v1"
POLICIES = {"adaptive_skill_library", "fixed_single_skill_baseline"}
SIGNALS = {"correct", "partial", "misconception", "confused", "no_response"}
MASTERY_DIMENSIONS = ("prerequisite", "conceptual", "procedural", "transfer")
TERMINAL_STATUSES = {"succeeded", "terminated_unable"}
ADAPTIVE_PROFILE_SCHEMA = "teaching_skill_miner.adaptive_student_profile_summary.v1"
ADAPTIVE_OBSERVATION_LIMIT = 12
ADAPTIVE_OBSERVATION_SOURCE = "deepseek_v4_flash_validated_diagnosis"
ADAPTIVE_OBSERVATION_STATUS = "candidate_unconfirmed"
PRIMARY_ROLES = {
    "diagnostic",
    "context",
    "example",
    "concept_mapping",
    "scaffolding",
    "assessment",
    "correction",
    "practice",
    "review",
    "metacognition",
    "engagement",
    "transfer",
    "summary",
}


class TeacherAgentError(ValueError):
    """Raised when a session or public fixture violates the task-two contract."""


class _TemplateValues(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _finite_probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeacherAgentError(f"{field} must be a number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise TeacherAgentError(f"{field} must be a finite number in [0, 1]")
    return result


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TeacherAgentError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, *, field: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise TeacherAgentError(f"{field} must be {qualifier}")
    result = [_nonempty_string(item, field=f"{field}[]") for item in value]
    if len(result) != len(set(result)):
        raise TeacherAgentError(f"{field} must not contain duplicates")
    return result


def validate_skill_library(library: Mapping[str, Any]) -> None:
    if not isinstance(library, Mapping) or library.get("schema") not in LIBRARY_SCHEMAS:
        raise TeacherAgentError(
            f"skill library schema must be one of {sorted(LIBRARY_SCHEMAS)}"
        )
    skills = library.get("skills")
    if not isinstance(skills, list) or len(skills) < 5:
        raise TeacherAgentError("skill library must contain at least five Skills")
    identifiers: set[str] = set()
    primary_count = 0
    is_v2 = library.get("schema") == LIBRARY_SCHEMA_V2
    if is_v2:
        derivation = library.get("derivation")
        if not isinstance(derivation, Mapping) or derivation.get(
            "general_skill_id"
        ) != "evidence_grounded_multimodal_teaching_neural_v1":
            raise TeacherAgentError("v2 library must bind the neural-v1 general Skill")
        boundaries = library.get("claim_boundary")
        if not isinstance(boundaries, Mapping) or boundaries.get(
            "neural_v1_materialization_gate_passed"
        ) is not False:
            raise TeacherAgentError(
                "v2 library must preserve the failed neural-v1 materialization gate"
            )
    for index, raw in enumerate(skills):
        if not isinstance(raw, Mapping):
            raise TeacherAgentError(f"skills[{index}] must be an object")
        skill_id = _nonempty_string(raw.get("skill_id"), field=f"skills[{index}].skill_id")
        if skill_id in identifiers:
            raise TeacherAgentError(f"duplicate skill_id: {skill_id}")
        identifiers.add(skill_id)
        role = _nonempty_string(raw.get("role"), field=f"skills[{index}].role")
        if role in PRIMARY_ROLES:
            primary_count += 1
        elif role != "support":
            raise TeacherAgentError(f"unsupported Skill role: {role}")
        _nonempty_string(raw.get("name"), field=f"skills[{index}].name")
        _nonempty_string(raw.get("selection_rationale"), field=f"skills[{index}].selection_rationale")
        _nonempty_string(raw.get("message_template"), field=f"skills[{index}].message_template")
        _nonempty_string(raw.get("expected_signal"), field=f"skills[{index}].expected_signal")
        focus = _nonempty_string(raw.get("focus_dimension"), field=f"skills[{index}].focus_dimension")
        if focus not in MASTERY_DIMENSIONS:
            raise TeacherAgentError(f"unsupported focus_dimension: {focus}")
        signals = _string_list(
            raw.get("applicable_signals", []),
            field=f"skills[{index}].applicable_signals",
            allow_empty=role == "support",
        )
        if not set(signals) <= SIGNALS | {"not_observed"}:
            raise TeacherAgentError(f"skills[{index}] has unsupported signal")
        _string_list(
            raw.get("supporting_skill_ids", []),
            field=f"skills[{index}].supporting_skill_ids",
            allow_empty=True,
        )
        source = raw.get("source")
        if not isinstance(source, Mapping):
            raise TeacherAgentError(f"skills[{index}].source must be an object")
        _nonempty_string(source.get("origin"), field=f"skills[{index}].source.origin")
        if is_v2:
            if source.get("general_skill_id") != (
                "evidence_grounded_multimodal_teaching_neural_v1"
            ):
                raise TeacherAgentError(
                    f"skills[{index}] must bind the neural-v1 general Skill"
                )
            contract = raw.get("execution_contract")
            if not isinstance(contract, Mapping):
                raise TeacherAgentError(
                    f"skills[{index}].execution_contract must be an object"
                )
            for contract_field in (
                "applicable_when",
                "failure_transition",
                "addition_reason",
            ):
                _nonempty_string(
                    contract.get(contract_field),
                    field=f"skills[{index}].execution_contract.{contract_field}",
                )
            for contract_field in (
                "contraindications",
                "preconditions",
                "postconditions",
            ):
                _string_list(
                    contract.get(contract_field, []),
                    field=f"skills[{index}].execution_contract.{contract_field}",
                    allow_empty=contract_field == "contraindications",
                )
            max_repeat = contract.get("max_repeat")
            if (
                isinstance(max_repeat, bool)
                or not isinstance(max_repeat, int)
                or not 1 <= max_repeat <= 50
            ):
                raise TeacherAgentError(
                    f"skills[{index}].execution_contract.max_repeat is invalid"
                )
    if primary_count < 5:
        raise TeacherAgentError("skill library must contain at least five primary Skills")
    for raw in skills:
        unknown = set(raw.get("supporting_skill_ids", [])) - identifiers
        if unknown:
            raise TeacherAgentError(
                f"{raw['skill_id']} references unknown supporting Skills: {sorted(unknown)}"
            )
    declared = library.get("content_sha256")
    if declared is not None:
        material = dict(library)
        material.pop("content_sha256", None)
        if declared != canonical_sha256(material):
            raise TeacherAgentError("skill library content_sha256 mismatch")


def _normalized_goal(goal: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(goal, Mapping):
        raise TeacherAgentError("goal must be an object")
    concept = _nonempty_string(goal.get("concept"), field="goal.concept")
    objective = _nonempty_string(goal.get("objective"), field="goal.objective")
    thresholds_raw = goal.get("success_thresholds", {})
    if not isinstance(thresholds_raw, Mapping):
        raise TeacherAgentError("goal.success_thresholds must be an object")
    default_thresholds = {
        "prerequisite": 0.60,
        "conceptual": 0.65,
        "procedural": 0.60,
        "transfer": 0.55,
    }
    thresholds = {
        dimension: _finite_probability(
            thresholds_raw.get(dimension, default),
            field=f"goal.success_thresholds.{dimension}",
        )
        for dimension, default in default_thresholds.items()
    }
    max_rounds = goal.get("max_rounds", 12)
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or not 3 <= max_rounds <= 50:
        raise TeacherAgentError("goal.max_rounds must be an integer in [3, 50]")
    materials = goal.get("materials", {})
    if not isinstance(materials, Mapping):
        raise TeacherAgentError("goal.materials must be an object")
    safe_materials: dict[str, str] = {}
    for key, value in materials.items():
        safe_materials[_nonempty_string(key, field="goal.materials key")] = _nonempty_string(
            value, field=f"goal.materials.{key}"
        )
    raw_components = goal.get("knowledge_components")
    if raw_components is None:
        # The teacher-provided concept is the minimum faithful retrieval anchor;
        # this is not a model-inferred curriculum decomposition.
        knowledge_components = [concept]
    else:
        knowledge_components = _string_list(
            raw_components,
            field="goal.knowledge_components",
        )
        if len(knowledge_components) > 12:
            raise TeacherAgentError("goal.knowledge_components must contain at most 12 items")
    return {
        "concept": concept,
        "objective": objective,
        "knowledge_components": knowledge_components,
        "success_thresholds": thresholds,
        "max_rounds": max_rounds,
        "materials": safe_materials,
    }


def _normalized_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, Mapping):
        raise TeacherAgentError("student_profile must be an object")
    learner_level = _nonempty_string(
        profile.get("learner_level", "beginner"), field="student_profile.learner_level"
    )
    initial = profile.get("initial_mastery", {})
    if not isinstance(initial, Mapping):
        raise TeacherAgentError("student_profile.initial_mastery must be an object")
    mastery = {
        dimension: _finite_probability(
            initial.get(dimension, 0.0),
            field=f"student_profile.initial_mastery.{dimension}",
        )
        for dimension in MASTERY_DIMENSIONS
    }
    preferences = profile.get("preferences", [])
    if not isinstance(preferences, list):
        raise TeacherAgentError("student_profile.preferences must be a list")
    preferences = [
        _nonempty_string(item, field="student_profile.preferences[]")
        for item in preferences
    ]
    misconceptions_raw = profile.get("known_misconceptions", [])
    if not isinstance(misconceptions_raw, list):
        raise TeacherAgentError("student_profile.known_misconceptions must be a list")
    misconceptions: list[dict[str, Any]] = []
    for index, item in enumerate(misconceptions_raw):
        if not isinstance(item, Mapping):
            raise TeacherAgentError(
                f"student_profile.known_misconceptions[{index}] must be an object"
            )
        misconceptions.append(
            {
                "tag": _nonempty_string(
                    item.get("tag"),
                    field=f"student_profile.known_misconceptions[{index}].tag",
                ),
                "description": _nonempty_string(
                    item.get("description", item.get("tag")),
                    field=f"student_profile.known_misconceptions[{index}].description",
                ),
                "confidence": _finite_probability(
                    item.get("confidence", 0.8),
                    field=f"student_profile.known_misconceptions[{index}].confidence",
                ),
                "status": "active",
                "first_observed_round": 0,
                "last_observed_round": 0,
            }
        )
    history = profile.get("conversation_history", [])
    if not isinstance(history, list):
        raise TeacherAgentError("student_profile.conversation_history must be a list")
    safe_history: list[dict[str, Any]] = []
    for index, item in enumerate(history):
        if not isinstance(item, Mapping):
            raise TeacherAgentError(
                f"student_profile.conversation_history[{index}] must be an object"
            )
        signal = _nonempty_string(
            item.get("signal"),
            field=f"student_profile.conversation_history[{index}].signal",
        )
        if signal not in SIGNALS:
            raise TeacherAgentError(f"unsupported history signal: {signal}")
        focus = str(item.get("focus_dimension", "conceptual"))
        if focus not in MASTERY_DIMENSIONS:
            raise TeacherAgentError(f"unsupported history focus_dimension: {focus}")
        safe_history.append(
            {
                "response": str(item.get("response", ""))[:500],
                "signal": signal,
                "focus_dimension": focus,
            }
        )
    return {
        "profile_ref": str(profile.get("profile_ref", "anonymous_student_profile"))[:80],
        "learner_level": learner_level,
        "preferences": preferences,
        "accessibility_needs": [
            _nonempty_string(item, field="student_profile.accessibility_needs[]")
            for item in profile.get("accessibility_needs", [])
        ]
        if isinstance(profile.get("accessibility_needs", []), list)
        else [],
        "initial_mastery": mastery,
        "known_misconceptions": misconceptions,
        "conversation_history": safe_history,
        "contains_direct_identity": bool(profile.get("contains_direct_identity", False)),
        "identity_assessment": (
            "user_declared_present"
            if bool(profile.get("contains_direct_identity", False))
            else "not_asserted_present_not_independently_verified"
        ),
    }


def _skill_index(library: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["skill_id"]): dict(item) for item in library["skills"]}


def _active_misconceptions(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in state.get("misconceptions", [])
        if item.get("status") == "active" and float(item.get("confidence", 0)) >= 0.5
    ]


def _lowest_mastery_dimension(
    mastery: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> str:
    return min(
        MASTERY_DIMENSIONS,
        key=lambda dimension: (
            float(mastery[dimension]) / max(float(thresholds[dimension]), 1e-9),
            MASTERY_DIMENSIONS.index(dimension),
        ),
    )


def _curriculum_focus(session: Mapping[str, Any]) -> str:
    """Choose the first unmet prerequisite in the learning dependency chain."""

    state = session["student_state"]
    mastery = state["knowledge_mastery"]
    thresholds = session["goal"]["success_thresholds"]
    if _active_misconceptions(state):
        return "conceptual"
    for dimension in MASTERY_DIMENSIONS:
        if float(mastery[dimension]) < float(thresholds[dimension]):
            return dimension
    return _lowest_mastery_dimension(mastery, thresholds)


def _active_knowledge_components(
    session: Mapping[str, Any],
    *,
    focus_dimension: str,
    action_text: str = "",
) -> list[str]:
    """Return the small ordered KC slice actually addressed by one action.

    Goal knowledge components are teacher-authored in dependency order.  A
    literal mention in the action wins; otherwise the current mastery phase is
    mapped to its corresponding slice.  This prevents every history turn from
    being tagged with the whole goal and makes same-KC retrieval meaningful.
    """

    goal = session.get("goal", {})
    raw_components = (
        goal.get("knowledge_components", []) if isinstance(goal, Mapping) else []
    )
    components = [
        str(item).strip()
        for item in raw_components
        if isinstance(item, str) and str(item).strip()
    ]
    if not components:
        concept = str(goal.get("concept", "")).strip() if isinstance(goal, Mapping) else ""
        return [concept] if concept else []

    folded_text = action_text.casefold()
    literal_matches = [
        component
        for component in components
        if component.casefold() in folded_text
    ]
    if literal_matches:
        return literal_matches[:3]

    try:
        phase_index = MASTERY_DIMENSIONS.index(focus_dimension)
    except ValueError:
        phase_index = 0
    start = phase_index * len(components) // len(MASTERY_DIMENSIONS)
    end = (phase_index + 1) * len(components) // len(MASTERY_DIMENSIONS)
    if end <= start:
        nearest = min(
            len(components) - 1,
            (phase_index * (len(components) - 1) + 1)
            // (len(MASTERY_DIMENSIONS) - 1),
        )
        return [components[nearest]]
    return components[start : min(end, start + 3)]


def _selection_scores(
    session: Mapping[str, Any], *, policy: str, fixed_skill_id: str | None
) -> list[dict[str, Any]]:
    library = session["skill_library"]
    state = session["student_state"]
    round_index = int(session["round"])
    history_present = bool(session["student_profile"]["conversation_history"])
    last_signal = str(state["understanding_signal"]["label"])
    active_misconceptions = _active_misconceptions(state)
    focus = _curriculum_focus(session)
    used_roles = [
        str(event.get("action", {}).get("primary_skill", {}).get("role", ""))
        for event in session.get("history", [])
    ]
    no_progress = int(session.get("control", {}).get("consecutive_no_progress", 0))
    engagement = str(
        state.get("interaction_statistics", {}).get("engagement_level", "unknown")
    )
    previous_id = session.get("current_action", {}).get("primary_skill", {}).get(
        "skill_id"
    )
    rows: list[dict[str, Any]] = []
    for skill in library["skills"]:
        if skill["role"] not in PRIMARY_ROLES:
            continue
        score = float(skill.get("base_priority", 0))
        reasons: list[str] = []
        if policy == "fixed_single_skill_baseline":
            score = 1000.0 if skill["skill_id"] == fixed_skill_id else -1000.0
            reasons.append("固定单一 Skill 基线不根据学生状态切换")
        else:
            if round_index == 0 and not history_present and skill["role"] == "diagnostic":
                score += 200
                reasons.append("首次教学先诊断前置知识")
            if last_signal in skill["applicable_signals"]:
                score += 35
                reasons.append(f"适配当前理解信号 {last_signal}")
            if skill["focus_dimension"] == focus:
                score += 120
                reasons.append(f"按学习依赖顺序，当前先补 {focus}")
            else:
                score -= 25
            if active_misconceptions and skill["role"] == "correction":
                score += 180
                reasons.append("存在高置信度活跃误解，优先对比纠错")
            elif not active_misconceptions and skill["role"] == "correction":
                score -= 80
            if skill["role"] == "transfer":
                mastery = state["knowledge_mastery"]
                thresholds = session["goal"]["success_thresholds"]
                ready = (
                    mastery["conceptual"] >= thresholds["conceptual"] * 0.75
                    and mastery["procedural"] >= thresholds["procedural"] * 0.75
                )
                score += 70 if ready else -90
                reasons.append("概念与过程达到迁移准备线" if ready else "尚未达到迁移准备线")
            if skill["role"] == "summary":
                score -= 40
            if skill["role"] == "diagnostic" and round_index > 0:
                if state["knowledge_mastery"]["prerequisite"] < session["goal"][
                    "success_thresholds"
                ]["prerequisite"]:
                    score += 35
                    reasons.append("前置知识仍低于成功门槛")
                else:
                    score -= 45
            if focus == "conceptual":
                if skill["role"] == "context" and "context" not in used_roles:
                    score += 48
                    reasons.append("已具备最低前置知识，先建立问题情境")
                if skill["role"] == "example" and last_signal in {
                    "not_observed",
                    "confused",
                    "no_response",
                }:
                    score += 55
                    reasons.append("先用直观例子建立概念表征")
                if skill["role"] == "concept_mapping" and last_signal in {
                    "correct",
                    "partial",
                }:
                    score += 50
                    reasons.append("已有初步表征，转入例子—定义映射")
                if skill["role"] == "assessment" and "concept_mapping" in used_roles:
                    score += 42
                    reasons.append("已有概念映射，进一步检查理由和边界")
            if focus == "procedural":
                if skill["role"] == "scaffolding" and "scaffolding" not in used_roles:
                    score += 65
                    reasons.append("先完成一次逐步支架示范")
                if skill["role"] == "practice" and "scaffolding" in used_roles:
                    score += 65
                    reasons.append("已有支架经历，转入独立练习反馈")
            if focus == "transfer" and skill["role"] == "transfer":
                score += 100
                reasons.append("前置、概念和过程已达标，开始变式迁移")
            if skill["role"] == "review" and last_signal in {
                "confused",
                "no_response",
            }:
                score += 45
                reasons.append("疑似回忆失败，优先主动检索而非完整重讲")
            if skill["role"] == "engagement":
                if no_progress >= 2 or engagement == "low":
                    score += 105
                    reasons.append("连续低进展或低参与，先恢复最小参与")
                else:
                    score -= 35
            if skill["role"] == "metacognition" and last_signal in {
                "correct",
                "partial",
            }:
                score += 28
                reasons.append("答案已有进展，要求学生解释依据与监控点")
            if previous_id == skill["skill_id"] and last_signal == "correct":
                score -= 25
                reasons.append("上一轮已正确，降低原 Skill 重复优先级")
        rows.append(
            {
                "skill_id": skill["skill_id"],
                "score": round(score, 3),
                "reasons": reasons or [skill["selection_rationale"]],
            }
        )
    return sorted(rows, key=lambda row: (-row["score"], row["skill_id"]))


def _render_action(
    session: dict[str, Any], *, policy: str, fixed_skill_id: str | None
) -> dict[str, Any]:
    scores = _selection_scores(session, policy=policy, fixed_skill_id=fixed_skill_id)
    selected_id = scores[0]["skill_id"]
    skills = _skill_index(session["skill_library"])
    selected = skills[selected_id]
    support_ids = list(selected.get("supporting_skill_ids", []))
    values = _TemplateValues(
        concept=session["goal"]["concept"],
        objective=session["goal"]["objective"],
        learner_level=session["student_profile"]["learner_level"],
        misconception=(
            _active_misconceptions(session["student_state"])[0]["description"]
            if _active_misconceptions(session["student_state"])
            else "当前尚未确认的误解"
        ),
        **session["goal"]["materials"],
    )
    message = str(selected["message_template"]).format_map(values)
    expected = str(selected["expected_signal"]).format_map(values)
    active_knowledge_components = _active_knowledge_components(
        session,
        focus_dimension=str(selected["focus_dimension"]),
        action_text=f"{message}\n{expected}",
    )
    previous_id = session.get("current_action", {}).get("primary_skill", {}).get(
        "skill_id"
    )
    switched = previous_id is not None and previous_id != selected_id
    reason = "；".join(scores[0]["reasons"] + [selected["selection_rationale"]])
    action = {
        "action_id": f"turn_{int(session['round']) + 1:03d}",
        "round": int(session["round"]) + 1,
        "primary_skill": {
            "skill_id": selected_id,
            "name": selected["name"],
            "role": selected["role"],
            "focus_dimension": selected["focus_dimension"],
            "knowledge_components": deepcopy(active_knowledge_components),
            "source": deepcopy(selected["source"]),
        },
        "supporting_skills": [
            {"skill_id": skill_id, "name": skills[skill_id]["name"]}
            for skill_id in support_ids
        ],
        "selection_reason": reason,
        "candidate_ranking": scores,
        "skill_switched": switched,
        "previous_primary_skill_id": previous_id,
        "teacher_action": {
            "type": selected["action_type"],
            "message": message,
            "expected_signal": expected,
            "direct_answer_prohibited": bool(
                selected.get("direct_answer_prohibited", True)
            ),
            "wait_for_student_before_next_action": True,
        },
        "knowledge_components": deepcopy(active_knowledge_components),
    }
    session["student_state"]["next_focus"] = {
        "dimension": selected["focus_dimension"],
        "reason": reason,
        "selected_skill_id": selected_id,
        "knowledge_components": deepcopy(active_knowledge_components),
    }
    return action


def _session_material(session: Mapping[str, Any]) -> dict[str, Any]:
    material = deepcopy(dict(session))
    material.pop("integrity", None)
    return material


def _refresh_integrity(session: dict[str, Any]) -> dict[str, Any]:
    session["integrity"] = {
        "algorithm": "sha256_canonical_json_without_integrity",
        "content_sha256": canonical_sha256(_session_material(session)),
    }
    return session


def _validate_adaptive_student_profile(
    profile: Mapping[str, Any], *, required: bool
) -> None:
    observations = profile.get("adaptive_observations")
    summary = profile.get("adaptive_summary")
    if observations is None and summary is None and not required:
        return
    if not isinstance(observations, list) or not isinstance(summary, Mapping):
        raise TeacherAgentError(
            "adaptive student profile requires observations and summary"
        )
    if len(observations) > ADAPTIVE_OBSERVATION_LIMIT:
        raise TeacherAgentError("adaptive_observations exceeds its retention limit")
    allowed_review_reasons = {"low_confidence", "model_requested_review"}
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping) or set(observation) != {
            "round",
            "source",
            "status",
            "candidate",
            "evidence",
        }:
            raise TeacherAgentError(f"adaptive_observations[{index}] is invalid")
        round_number = observation["round"]
        if (
            isinstance(round_number, bool)
            or not isinstance(round_number, int)
            or round_number < 1
        ):
            raise TeacherAgentError(
                f"adaptive_observations[{index}].round is invalid"
            )
        if (
            observation["source"] != ADAPTIVE_OBSERVATION_SOURCE
            or observation["status"] != ADAPTIVE_OBSERVATION_STATUS
        ):
            raise TeacherAgentError(
                f"adaptive_observations[{index}] provenance is invalid"
            )
        candidate = observation["candidate"]
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "response_quality",
            "engagement_level",
            "misconception_tag",
            "next_focus",
        }:
            raise TeacherAgentError(
                f"adaptive_observations[{index}].candidate is invalid"
            )
        if candidate["response_quality"] not in {
            "complete",
            "partial",
            "minimal",
            "off_topic",
            "empty",
        } or candidate["engagement_level"] not in {
            "high",
            "medium",
            "low",
            "unknown",
        }:
            raise TeacherAgentError(
                f"adaptive_observations[{index}] candidate labels are invalid"
            )
        tag = candidate["misconception_tag"]
        if tag is not None and (not isinstance(tag, str) or len(tag) > 120):
            raise TeacherAgentError(
                f"adaptive_observations[{index}].misconception_tag is invalid"
            )
        if candidate["next_focus"] not in MASTERY_DIMENSIONS:
            raise TeacherAgentError(
                f"adaptive_observations[{index}].next_focus is invalid"
            )
        evidence = observation["evidence"]
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "excerpt",
            "confidence",
            "grounding",
            "needs_human_review",
            "review_reasons",
            "privacy_status",
        }:
            raise TeacherAgentError(
                f"adaptive_observations[{index}].evidence is invalid"
            )
        if not isinstance(evidence["excerpt"], str) or len(evidence["excerpt"]) > 240:
            raise TeacherAgentError(
                f"adaptive_observations[{index}].evidence.excerpt is invalid"
            )
        _finite_probability(
            evidence["confidence"],
            field=f"adaptive_observations[{index}].evidence.confidence",
        )
        if evidence["grounding"] not in {
            "verified_current_response_substring",
            "empty_response_no_excerpt",
        } or evidence["privacy_status"] != "redacted_before_candidate_storage":
            raise TeacherAgentError(
                f"adaptive_observations[{index}].evidence provenance is invalid"
            )
        reasons = evidence["review_reasons"]
        if (
            not isinstance(evidence["needs_human_review"], bool)
            or not isinstance(reasons, list)
            or len(reasons) != len(set(reasons))
            or not set(reasons) <= allowed_review_reasons
            or bool(reasons) != evidence["needs_human_review"]
        ):
            raise TeacherAgentError(
                f"adaptive_observations[{index}].evidence review state is invalid"
            )
    expected_summary_fields = {
        "schema",
        "status",
        "source",
        "observation_limit",
        "total_observation_count",
        "retained_observation_count",
        "latest_round",
        "latest_response_quality",
        "latest_engagement_level",
        "candidate_misconception_tags",
        "latest_next_focus",
        "needs_human_review",
        "teacher_provided_fields_overwritten",
    }
    if set(summary) != expected_summary_fields:
        raise TeacherAgentError("adaptive_summary fields are invalid")
    total = summary["total_observation_count"]
    if (
        summary["schema"] != ADAPTIVE_PROFILE_SCHEMA
        or summary["status"] != ADAPTIVE_OBSERVATION_STATUS
        or summary["source"] != "validated_deepseek_diagnoses_only"
        or summary["observation_limit"] != ADAPTIVE_OBSERVATION_LIMIT
        or summary["retained_observation_count"] != len(observations)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total < len(observations)
        or not isinstance(summary["needs_human_review"], bool)
        or summary["teacher_provided_fields_overwritten"] is not False
    ):
        raise TeacherAgentError("adaptive_summary metadata is invalid")
    latest = observations[-1] if observations else None
    if latest is None:
        if any(
            summary[field] is not None
            for field in (
                "latest_round",
                "latest_response_quality",
                "latest_engagement_level",
                "latest_next_focus",
            )
        ) or (
            summary["candidate_misconception_tags"] != []
            or summary["total_observation_count"] != 0
            or summary["needs_human_review"]
        ):
            raise TeacherAgentError("empty adaptive_summary is inconsistent")
        return
    candidate = latest["candidate"]
    if (
        summary["latest_round"] != latest["round"]
        or summary["latest_response_quality"] != candidate["response_quality"]
        or summary["latest_engagement_level"] != candidate["engagement_level"]
        or summary["latest_next_focus"] != candidate["next_focus"]
        or summary["needs_human_review"]
        != any(item["evidence"]["needs_human_review"] for item in observations)
    ):
        raise TeacherAgentError("adaptive_summary latest candidate is inconsistent")
    expected_tags = list(
        dict.fromkeys(
            item["candidate"]["misconception_tag"]
            for item in observations
            if item["candidate"]["misconception_tag"]
        )
    )[-8:]
    if summary["candidate_misconception_tags"] != expected_tags:
        raise TeacherAgentError("adaptive_summary misconception tags are inconsistent")


def validate_session(session: Mapping[str, Any]) -> None:
    if not isinstance(session, Mapping) or session.get("schema") != SESSION_SCHEMA:
        raise TeacherAgentError(f"session schema must be {SESSION_SCHEMA}")
    if session.get("policy") not in POLICIES:
        raise TeacherAgentError("session policy is unsupported")
    validate_skill_library(session.get("skill_library", {}))
    if session.get("skill_library_fingerprint") != canonical_sha256(
        session.get("skill_library", {})
    ):
        raise TeacherAgentError("session skill_library_fingerprint mismatch")
    integrity = session.get("integrity")
    if not isinstance(integrity, Mapping) or integrity.get("content_sha256") != canonical_sha256(
        _session_material(session)
    ):
        raise TeacherAgentError("session integrity check failed")
    state = session.get("student_state")
    if not isinstance(state, Mapping):
        raise TeacherAgentError("session student_state is missing")
    if set(state.get("knowledge_mastery", {})) != set(MASTERY_DIMENSIONS):
        raise TeacherAgentError("student_state knowledge_mastery dimensions are incomplete")
    for dimension, value in state["knowledge_mastery"].items():
        _finite_probability(value, field=f"student_state.knowledge_mastery.{dimension}")
    for field in ("misconceptions", "understanding_signal", "next_focus"):
        if field not in state:
            raise TeacherAgentError(f"student_state.{field} is required")
    profile = session.get("student_profile")
    if not isinstance(profile, Mapping):
        raise TeacherAgentError("session student_profile is missing")
    _validate_adaptive_student_profile(
        profile,
        required=session.get("artifact_kind")
        == "real_time_deepseek_teaching_agent_session",
    )
    adaptive_rounds = [
        item["round"] for item in profile.get("adaptive_observations", [])
    ]
    if adaptive_rounds != sorted(set(adaptive_rounds)) or any(
        round_number > int(session.get("round", -1))
        for round_number in adaptive_rounds
    ):
        raise TeacherAgentError("adaptive_observations rounds are inconsistent")
    status = session.get("status")
    if status not in {"active", *TERMINAL_STATUSES}:
        raise TeacherAgentError("session status is unsupported")
    if status == "active" and not isinstance(session.get("current_action"), Mapping):
        raise TeacherAgentError("active session requires current_action")
    if session.get("artifact_kind") == "real_time_deepseek_teaching_agent_session":
        # Local import keeps the deterministic core reusable while making a
        # live session fail closed on malformed or over-budget model context.
        from .teacher_agent_context import (  # noqa: PLC0415
            LAYERED_CONTEXT_SCHEMA,
            validate_layered_context,
        )

        try:
            validate_layered_context(session.get("context_memory", {}))
        except (TypeError, ValueError) as exc:
            raise TeacherAgentError("live session context_memory is invalid") from exc
        runtime = session.get("agent_runtime", {})
        trace = runtime.get("last_context_trace") if isinstance(runtime, Mapping) else None
        if (
            not isinstance(trace, Mapping)
            or trace.get("schema") != LAYERED_CONTEXT_SCHEMA
            or trace.get("content_sha256")
            != canonical_sha256(session["context_memory"])
            or trace.get("serialized_chars")
            != session["context_memory"]["budget"]["serialized_chars"]
            or trace.get("model_generated_memory_written") is not False
        ):
            raise TeacherAgentError("live session context trace is invalid")


def start_teacher_agent_session(
    goal: Mapping[str, Any],
    student_profile: Mapping[str, Any],
    skill_library: Mapping[str, Any],
    *,
    policy: str = "adaptive_skill_library",
    fixed_skill_id: str | None = None,
) -> dict[str, Any]:
    """Create a session and emit only the first next action."""

    if policy not in POLICIES:
        raise TeacherAgentError(f"policy must be one of {sorted(POLICIES)}")
    library = deepcopy(dict(skill_library))
    validate_skill_library(library)
    goal_value = _normalized_goal(goal)
    profile = _normalized_profile(student_profile)
    skills = _skill_index(library)
    if policy == "fixed_single_skill_baseline":
        fixed_skill_id = fixed_skill_id or "skill_stepwise_scaffolding"
        if fixed_skill_id not in skills or skills[fixed_skill_id]["role"] not in PRIMARY_ROLES:
            raise TeacherAgentError("fixed baseline Skill is missing or not primary")
    mastery = deepcopy(profile["initial_mastery"])
    signal = {
        "label": "not_observed",
        "confidence": 0.0,
        "response_excerpt": "",
        "source": "no_current_turn_observation",
    }
    for prior in profile["conversation_history"]:
        signal = {
            "label": prior["signal"],
            "confidence": 1.0,
            "response_excerpt": prior["response"][:160],
            "source": "provided_conversation_history",
        }
        if prior["signal"] == "correct":
            mastery[prior["focus_dimension"]] = min(
                1.0, mastery[prior["focus_dimension"]] + 0.08
            )
    session: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "artifact_kind": "real_time_single_turn_teaching_agent_session",
        "status": "active",
        "policy": policy,
        "fixed_skill_id": fixed_skill_id,
        "goal": goal_value,
        "student_profile": profile,
        "skill_library": library,
        "skill_library_fingerprint": canonical_sha256(library),
        "round": 0,
        "student_state": {
            "knowledge_mastery": mastery,
            "misconceptions": deepcopy(profile["known_misconceptions"]),
            "understanding_signal": signal,
            "next_focus": {
                "dimension": "prerequisite",
                "reason": "等待首次 Skill 选择",
                "selected_skill_id": None,
            },
            "assessment_basis": "provided structured signals and deterministic updates",
        },
        "current_action": {},
        "history": [],
        "control": {
            "consecutive_no_progress": 0,
            "skill_switch_count": 0,
            "termination_reason": None,
        },
        "claim_boundary": {
            "free_text_answer_grading_established": False,
            "structured_signal_source": "teacher_or_external_judge",
            "student_identity_required": False,
            "simulated_evaluation_is_real_learning_effect": False,
            "real_learning_effectiveness_established": False,
            "deployment_quality_established": False,
        },
    }
    session["current_action"] = _render_action(
        session, policy=policy, fixed_skill_id=fixed_skill_id
    )
    return _refresh_integrity(session)


def _update_misconceptions(
    session: dict[str, Any],
    *,
    signal: str,
    response: str,
    misconception_tag: str | None,
    confidence: float,
) -> None:
    state = session["student_state"]
    action = session["current_action"]
    if signal == "misconception":
        tag = (misconception_tag or "unclassified_misconception").strip()
        if not tag:
            tag = "unclassified_misconception"
        existing = next(
            (item for item in state["misconceptions"] if item["tag"] == tag), None
        )
        if existing is None:
            state["misconceptions"].append(
                {
                    "tag": tag[:120],
                    "description": response[:240] or tag[:120],
                    "confidence": confidence,
                    "status": "active",
                    "first_observed_round": int(session["round"]) + 1,
                    "last_observed_round": int(session["round"]) + 1,
                }
            )
        else:
            existing["status"] = "active"
            existing["confidence"] = max(float(existing["confidence"]), confidence)
            existing["last_observed_round"] = int(session["round"]) + 1
    elif signal == "correct" and action["primary_skill"]["role"] == "correction":
        for item in state["misconceptions"]:
            if item["status"] == "active":
                item["status"] = "resolved"
                item["confidence"] = round(max(0.0, float(item["confidence"]) - 0.55), 3)
                item["resolved_round"] = int(session["round"]) + 1


def _apply_student_signal(
    session: dict[str, Any],
    *,
    response: str,
    signal: str,
    misconception_tag: str | None,
    confidence: float,
) -> None:
    action = session["current_action"]
    focus = action["primary_skill"]["focus_dimension"]
    state = session["student_state"]
    increments = {
        "correct": 0.28,
        "partial": 0.10,
        "misconception": 0.0,
        "confused": 0.0,
        "no_response": 0.0,
    }
    delta = increments[signal] * confidence
    state["knowledge_mastery"][focus] = round(
        min(1.0, float(state["knowledge_mastery"][focus]) + delta), 3
    )
    state["understanding_signal"] = {
        "label": signal,
        "confidence": confidence,
        "response_excerpt": response[:240],
        "source": "teacher_or_external_judge",
    }
    _update_misconceptions(
        session,
        signal=signal,
        response=response,
        misconception_tag=misconception_tag,
        confidence=confidence,
    )
    if signal in {"correct", "partial"} and confidence >= 0.5:
        session["control"]["consecutive_no_progress"] = 0
    else:
        session["control"]["consecutive_no_progress"] += 1


def _success_ready(session: Mapping[str, Any]) -> bool:
    if int(session["round"]) < 3:
        return False
    mastery = session["student_state"]["knowledge_mastery"]
    thresholds = session["goal"]["success_thresholds"]
    return (
        all(float(mastery[key]) >= float(thresholds[key]) for key in MASTERY_DIMENSIONS)
        and not _active_misconceptions(session["student_state"])
        and session["student_state"]["understanding_signal"]["label"] == "correct"
        and float(
            session["student_state"]["understanding_signal"].get("confidence", 0.0)
        )
        >= 0.5
    )


def advance_teacher_agent_session(
    session: Mapping[str, Any],
    *,
    learner_response: str,
    signal: str,
    misconception_tag: str | None = None,
    signal_confidence: float = 1.0,
) -> dict[str, Any]:
    """Consume one response and emit exactly one subsequent action or termination."""

    current = deepcopy(dict(session))
    validate_session(current)
    if current["status"] != "active":
        raise TeacherAgentError("cannot advance a terminal session")
    if signal not in SIGNALS:
        raise TeacherAgentError(f"signal must be one of {sorted(SIGNALS)}")
    confidence = _finite_probability(signal_confidence, field="signal_confidence")
    response = str(learner_response).strip()
    action_before = deepcopy(current["current_action"])
    state_before = deepcopy(current["student_state"])
    _apply_student_signal(
        current,
        response=response,
        signal=signal,
        misconception_tag=misconception_tag,
        confidence=confidence,
    )
    current["round"] += 1
    event = {
        "round": current["round"],
        "action": action_before,
        "learner_response": response[:1000],
        "structured_signal": {
            "label": signal,
            "confidence": confidence,
            "source": "teacher_or_external_judge",
        },
        "student_state_before": state_before,
        "student_state_after_observation": deepcopy(current["student_state"]),
    }
    current["history"].append(event)

    if _success_ready(current):
        current["status"] = "succeeded"
        current["control"]["termination_reason"] = (
            "all mastery thresholds reached, no active high-confidence misconception, "
            "and the latest structured signal is correct"
        )
        current["current_action"] = {
            "action_id": f"terminal_{current['round']:03d}",
            "round": current["round"],
            "type": "terminate_success",
            "teacher_action": {
                "type": "summarize_and_stop",
                "message": (
                    f"本轮关于 {current['goal']['concept']} 的目标已达到。"
                    "请学习者用自己的话总结关键条件，并安排一次延迟复习。"
                ),
                "wait_for_student_before_next_action": False,
            },
            "termination_reason": current["control"]["termination_reason"],
        }
    elif (
        current["round"] >= current["goal"]["max_rounds"]
        or current["control"]["consecutive_no_progress"] >= 3
    ):
        current["status"] = "terminated_unable"
        reason = (
            "maximum teaching rounds reached"
            if current["round"] >= current["goal"]["max_rounds"]
            else "three consecutive rounds without observable progress"
        )
        current["control"]["termination_reason"] = reason
        current["current_action"] = {
            "action_id": f"terminal_{current['round']:03d}",
            "round": current["round"],
            "type": "terminate_unable",
            "teacher_action": {
                "type": "stop_and_escalate",
                "message": (
                    "当前证据不足以安全继续。停止增加新内容，记录未掌握点，"
                    "建议教师人工诊断或更换前置材料。"
                ),
                "wait_for_student_before_next_action": False,
            },
            "termination_reason": reason,
        }
    else:
        next_action = _render_action(
            current,
            policy=current["policy"],
            fixed_skill_id=current.get("fixed_skill_id"),
        )
        if next_action["skill_switched"]:
            current["control"]["skill_switch_count"] += 1
        current["current_action"] = next_action
    return _refresh_integrity(current)


def session_turn_summary(session: Mapping[str, Any]) -> dict[str, Any]:
    """Return a concise, UI-safe view of the next action and explicit state."""

    validate_session(session)
    return {
        "schema": session["schema"],
        "status": session["status"],
        "rounds_completed": session["round"],
        "goal": deepcopy(session["goal"]),
        "student_state": deepcopy(session["student_state"]),
        "next_action": deepcopy(session["current_action"]),
        "skill_switch_count": session["control"]["skill_switch_count"],
        "termination_reason": session["control"]["termination_reason"],
        "claim_boundary": deepcopy(session["claim_boundary"]),
    }


def _mastery_score(session: Mapping[str, Any]) -> float:
    mastery = session["student_state"]["knowledge_mastery"]
    return round(
        100
        * sum(float(mastery[key]) for key in ("conceptual", "procedural", "transfer"))
        / 3,
        3,
    )


def _behavior_checks(action: Mapping[str, Any], concept: str) -> dict[str, bool]:
    teacher = action.get("teacher_action", {})
    message = str(teacher.get("message", ""))
    return {
        "message_nonempty": bool(message.strip()),
        "goal_targeted": concept in message,
        "selection_reason_present": bool(str(action.get("selection_reason", "")).strip()),
        "expected_signal_present": bool(str(teacher.get("expected_signal", "")).strip()),
        "waits_for_student": teacher.get("wait_for_student_before_next_action") is True,
        "does_not_directly_dump_answer": teacher.get("direct_answer_prohibited") is True,
    }


def _run_evaluation_policy(
    library: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    policy: str,
    fixed_skill_id: str | None = None,
) -> dict[str, Any]:
    session = start_teacher_agent_session(
        case["goal"],
        case["student_profile"],
        library,
        policy=policy,
        fixed_skill_id=fixed_skill_id,
    )
    pretest = _mastery_score(session)
    decisions: list[bool] = []
    state_checks: list[bool] = []
    behavior_checks: list[bool] = []
    selected: list[str] = []
    timeline: list[dict[str, Any]] = []
    for index, feedback in enumerate(case["feedback_sequence"]):
        if session["status"] != "active":
            break
        action = session["current_action"]
        selected_id = action["primary_skill"]["skill_id"]
        selected.append(selected_id)
        expected = feedback.get("expected_skill_ids", [])
        if expected:
            decisions.append(selected_id in expected)
        checks = _behavior_checks(action, session["goal"]["concept"])
        behavior_checks.extend(checks.values())
        state_before = deepcopy(session["student_state"])
        session = advance_teacher_agent_session(
            session,
            learner_response=str(feedback.get("response", "")),
            signal=str(feedback["signal"]),
            misconception_tag=feedback.get("misconception_tag"),
            signal_confidence=float(feedback.get("confidence", 1.0)),
        )
        timeline.append(
            {
                "round": index + 1,
                "teacher_action": {
                    "primary_skill_id": selected_id,
                    "primary_skill_name": action["primary_skill"]["name"],
                    "selection_reason": action["selection_reason"],
                    "message": action["teacher_action"]["message"],
                    "expected_signal": action["teacher_action"]["expected_signal"],
                },
                "learner_feedback": {
                    "response": str(feedback.get("response", ""))[:1000],
                    "declared_signal": feedback["signal"],
                    "signal_confidence": float(feedback.get("confidence", 1.0)),
                },
                "student_state_before": state_before,
                "student_state_after": deepcopy(session["student_state"]),
                "status_after": session["status"],
            }
        )
        observed = session["student_state"]["understanding_signal"]
        state_checks.append(observed["label"] == feedback["signal"])
        expected_focus = feedback.get("expected_next_focus")
        if expected_focus and session["status"] == "active":
            state_checks.append(
                session["student_state"]["next_focus"]["dimension"]
                == expected_focus
            )
    posttest = _mastery_score(session)
    transfer_threshold = float(session["goal"]["success_thresholds"]["transfer"])
    return {
        "policy": policy,
        "status": session["status"],
        "termination_reason": session["control"]["termination_reason"],
        "selected_skill_ids": selected,
        "unique_skill_count": len(set(selected)),
        "skill_switch_count": session["control"]["skill_switch_count"],
        "timeline": timeline,
        "structured_state_check_rate": (
            sum(state_checks) / len(state_checks) if state_checks else 0.0
        ),
        "decision_match_rate": (
            sum(decisions) / len(decisions) if decisions else None
        ),
        "behavior_quality_rate": (
            sum(behavior_checks) / len(behavior_checks) if behavior_checks else 0.0
        ),
        "simulated_pretest_score": pretest,
        "simulated_posttest_score": posttest,
        "simulated_gain": round(posttest - pretest, 3),
        "simulated_transfer_passed": (
            session["student_state"]["knowledge_mastery"]["transfer"]
            >= transfer_threshold
        ),
        "final_student_state": deepcopy(session["student_state"]),
        "session_content_sha256": session["integrity"]["content_sha256"],
    }


def evaluate_teacher_agent(
    skill_library: Mapping[str, Any], evaluation_cases: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate adaptive decisions against a fixed single-Skill baseline.

    The fixtures use supplied structured signals.  The reported pre/post and
    transfer values are deterministic simulation diagnostics, never real
    learner-effect evidence.
    """

    library = deepcopy(dict(skill_library))
    validate_skill_library(library)
    cases = evaluation_cases.get("cases") if isinstance(evaluation_cases, Mapping) else None
    if not isinstance(cases, list) or not cases:
        raise TeacherAgentError("evaluation cases must contain a non-empty cases list")
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise TeacherAgentError(f"cases[{index}] must be an object")
        case_id = _nonempty_string(case.get("case_id"), field=f"cases[{index}].case_id")
        expected_terminal_status = _nonempty_string(
            case.get("expected_terminal_status"),
            field=f"cases[{index}].expected_terminal_status",
        )
        if expected_terminal_status not in TERMINAL_STATUSES:
            raise TeacherAgentError(
                f"cases[{index}].expected_terminal_status must be one of "
                f"{sorted(TERMINAL_STATUSES)}"
            )
        feedback = case.get("feedback_sequence")
        if not isinstance(feedback, list) or not feedback:
            raise TeacherAgentError(f"cases[{index}].feedback_sequence must be non-empty")
        adaptive = _run_evaluation_policy(
            library, case, policy="adaptive_skill_library"
        )
        baseline = _run_evaluation_policy(
            library,
            case,
            policy="fixed_single_skill_baseline",
            fixed_skill_id="skill_stepwise_scaffolding",
        )
        results.append(
            {
                "case_id": case_id,
                "expected_terminal_status": expected_terminal_status,
                "adaptive_agent": adaptive,
                "fixed_single_skill_baseline": baseline,
                "terminal_decision_match": (
                    adaptive["status"] == expected_terminal_status
                ),
                "simulated_gain_delta": round(
                    adaptive["simulated_gain"] - baseline["simulated_gain"], 3
                ),
                "adaptive_used_multiple_skills": adaptive["unique_skill_count"] >= 2,
            }
        )
    adaptive_rows = [row["adaptive_agent"] for row in results]
    baseline_rows = [row["fixed_single_skill_baseline"] for row in results]

    def mean(rows: list[dict[str, Any]], key: str) -> float:
        return round(sum(float(row[key]) for row in rows) / len(rows), 6)

    decision_values = [
        float(row["decision_match_rate"])
        for row in adaptive_rows
        if row["decision_match_rate"] is not None
    ]
    aggregate = {
        "case_count": len(results),
        "expected_success_case_count": sum(
            row["expected_terminal_status"] == "succeeded" for row in results
        ),
        "expected_unable_case_count": sum(
            row["expected_terminal_status"] == "terminated_unable" for row in results
        ),
        "terminal_decision_match_rate": round(
            sum(row["terminal_decision_match"] for row in results) / len(results),
            6,
        ),
        "student_state_judgement_rate": mean(
            adaptive_rows, "structured_state_check_rate"
        ),
        "teaching_decision_match_rate": round(
            sum(decision_values) / len(decision_values), 6
        )
        if decision_values
        else 0.0,
        "teaching_behavior_quality_rate": mean(
            adaptive_rows, "behavior_quality_rate"
        ),
        "adaptive_multi_skill_case_rate": round(
            sum(row["adaptive_used_multiple_skills"] for row in results)
            / len(results),
            6,
        ),
        "adaptive_simulated_mean_gain": mean(adaptive_rows, "simulated_gain"),
        "baseline_simulated_mean_gain": mean(baseline_rows, "simulated_gain"),
        "simulated_mean_gain_delta": round(
            mean(adaptive_rows, "simulated_gain")
            - mean(baseline_rows, "simulated_gain"),
            6,
        ),
        "adaptive_simulated_transfer_pass_rate": round(
            sum(row["simulated_transfer_passed"] for row in adaptive_rows)
            / len(adaptive_rows),
            6,
        ),
        "baseline_simulated_transfer_pass_rate": round(
            sum(row["simulated_transfer_passed"] for row in baseline_rows)
            / len(baseline_rows),
            6,
        ),
    }
    gates = {
        "student_state_judgement": aggregate["student_state_judgement_rate"] >= 0.90,
        "teaching_decision_quality": aggregate["teaching_decision_match_rate"] >= 0.75,
        "teaching_behavior_quality": aggregate["teaching_behavior_quality_rate"] >= 0.95,
        "termination_quality": aggregate["terminal_decision_match_rate"] >= 0.95,
        "multi_skill_selection_and_switching": aggregate[
            "adaptive_multi_skill_case_rate"
        ]
        >= 0.80,
        "adaptive_beats_fixed_skill_in_simulation": aggregate[
            "simulated_mean_gain_delta"
        ]
        > 0,
    }
    report = {
        "schema": EVALUATION_SCHEMA,
        "artifact_kind": "reproducible_structured_fixture_teacher_agent_evaluation",
        "skill_library_fingerprint": canonical_sha256(library),
        "evaluation_cases_fingerprint": canonical_sha256(dict(evaluation_cases)),
        "baseline": {
            "policy": "fixed_single_skill_baseline",
            "skill_id": "skill_stepwise_scaffolding",
        },
        "aggregate": aggregate,
        "gates": gates,
        "passed": all(gates.values()),
        "cases": results,
        "claim_boundary": {
            "fixtures_are_synthetic": True,
            "responses_are_structured_test_inputs": True,
            "free_text_answer_grading_established": False,
            "simulated_pre_post_is_real_learning_effect": False,
            "real_learner_effectiveness_established": False,
            "expert_teaching_quality_established": False,
            "deployment_quality_established": False,
        },
        "metric_semantics": {
            "student_state_judgement_rate": "agreement with declared structured fixture state, not free-text diagnostic accuracy",
            "teaching_decision_match_rate": "match to fixture-allowed Skill choices",
            "teaching_behavior_quality_rate": "deterministic safety and completeness checks, not expert rating",
            "terminal_decision_match_rate": "agreement with each fixture's declared success-or-escalate outcome",
            "simulated_gain": "change in deterministic mastery state, not observed learner gain",
        },
    }
    report["content_sha256"] = canonical_sha256(report)
    return report
