"""Reproducible adversarial multi-turn evaluation for the Teaching Agent.

This benchmark is deliberately separate from the implementation tests.  It runs
complete scripted episodes against an executor, keeps every gold field outside
the model request, and reports semantic behavior independently from transport and
session-safety checks.

The bundled episodes are author-constructed development fixtures.  They are not
expert validated, do not involve real learners, and do not establish deployment
accuracy or a learning effect.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any, Mapping, Protocol, Sequence

from .deepseek_client import DeepSeekClient, DeepSeekConfig
from .io_utils import read_json, resolve_resource_path, write_json
from .teacher_agent import TeacherAgentError, start_teacher_agent_session
from .teacher_agent_live import (
    LIVE_PROMPT_VERSION,
    LiveAgentOptions,
    advance_live_teacher_agent_session,
    start_live_teacher_agent_session,
)


BENCHMARK_SCHEMA = "teaching_skill_miner.teacher_agent_multiturn_benchmark.v1"
REPORT_SCHEMA = "teaching_skill_miner.teacher_agent_multiturn_report.v1"

EXECUTOR_CURRENT = "current"
EXECUTOR_SAFE = "safe_generative_executor"
EXECUTOR_NAMES = frozenset({EXECUTOR_CURRENT, EXECUTOR_SAFE})

MINIMUM_LIVE_VALIDATED_MODEL_PLAN_RATE = 0.8
MINIMUM_LIVE_SAFE_EXECUTOR_SUCCESS_RATE = 0.5

_DEFAULT_BENCHMARK = Path("data/teacher_agent_multiturn_benchmark_v1.json")
_DEFAULT_LIBRARY = Path("data/teacher_agent_skill_library_v2.json")
_TERMINAL_STATUSES = frozenset({"succeeded", "terminated_unable"})
_MODEL_REQUEST_OUTCOMES = frozenset(
    {
        "validated_model_plan",
        "deterministic_safety_fallback",
        "prepared_not_yet_validated",
        "missing_context_trace",
        "invalid_context_trace",
        "not_attempted",
        "unknown",
    }
)
_MODEL_GOLD_KEYS = frozenset(
    {
        "gold",
        "episode_gold",
        "acceptable_signals",
        "allowed_primary_skill_ids",
        "must_address_term_groups",
        "recall_term_groups",
        "forbidden_output_terms",
        "direct_answer_terms",
        "profile_leakage_terms",
        "requires_visual_abstention",
        "expected_switch",
        "should_stop",
        "critical",
    }
)
class TeacherAgentMultiturnBenchmarkError(ValueError):
    """Raised when the benchmark fixture, executor, or report is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TeacherAgentMultiturnBenchmarkError(f"{field} must be an object")
    return value


def _require_string_list(
    value: Any,
    *,
    field: str,
    allow_empty: bool = True,
    maximum: int = 32,
) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > maximum
        or (not allow_empty and not value)
        or any(not _nonempty(item) for item in value)
    ):
        raise TeacherAgentMultiturnBenchmarkError(f"{field} is invalid")
    return [str(item) for item in value]


def _primary_skill_ids(library: Mapping[str, Any]) -> set[str]:
    skills = library.get("skills")
    if not isinstance(skills, list):
        raise TeacherAgentMultiturnBenchmarkError("skill library is invalid")
    result = {
        str(skill.get("skill_id"))
        for skill in skills
        if isinstance(skill, Mapping)
        and skill.get("role") != "support"
        and _nonempty(skill.get("skill_id"))
    }
    if not result:
        raise TeacherAgentMultiturnBenchmarkError("no primary Skills are available")
    return result


def _validate_term_groups(value: Any, *, field: str) -> list[list[str]]:
    if not isinstance(value, list) or len(value) > 16:
        raise TeacherAgentMultiturnBenchmarkError(f"{field} must be a bounded list")
    groups: list[list[str]] = []
    for index, group in enumerate(value):
        groups.append(
            _require_string_list(
                group,
                field=f"{field}[{index}]",
                allow_empty=False,
                maximum=12,
            )
        )
    return groups


def validate_multiturn_benchmark(
    dataset: Mapping[str, Any], library: Mapping[str, Any]
) -> None:
    """Fail closed on every field used by execution or scoring."""

    if dataset.get("schema") != BENCHMARK_SCHEMA:
        raise TeacherAgentMultiturnBenchmarkError(
            f"benchmark schema must be {BENCHMARK_SCHEMA}"
        )
    goals = _require_mapping(dataset.get("goals"), field="goals")
    profiles = _require_mapping(dataset.get("student_profiles"), field="student_profiles")
    evidence_catalog = _require_mapping(
        dataset.get("visual_evidence_catalog", {}), field="visual_evidence_catalog"
    )
    if not goals or not profiles:
        raise TeacherAgentMultiturnBenchmarkError(
            "at least one goal and student profile are required"
        )
    for goal_id, goal in goals.items():
        if not _nonempty(goal_id) or not isinstance(goal, Mapping):
            raise TeacherAgentMultiturnBenchmarkError("goal catalog is invalid")
    for profile_id, profile in profiles.items():
        if not _nonempty(profile_id) or not isinstance(profile, Mapping):
            raise TeacherAgentMultiturnBenchmarkError("profile catalog is invalid")
    representative_goal = next(iter(goals.values()))
    for profile_id, profile in profiles.items():
        try:
            start_teacher_agent_session(representative_goal, profile, library)
        except (TeacherAgentError, TypeError, ValueError) as exc:
            raise TeacherAgentMultiturnBenchmarkError(
                f"student profile {profile_id} is incompatible with the live runtime"
            ) from exc
    for evidence_id, evidence in evidence_catalog.items():
        if (
            not _nonempty(evidence_id)
            or not isinstance(evidence, Mapping)
            or evidence.get("schema")
            != "teaching_skill_miner.local_visual_evidence.v1"
        ):
            raise TeacherAgentMultiturnBenchmarkError(
                "visual evidence catalog is invalid"
            )

    boundary = _require_mapping(dataset.get("claim_boundary"), field="claim_boundary")
    required_false = (
        "expert_validated",
        "real_students_involved",
        "held_out_after_prompt_development",
        "deployment_accuracy_established",
        "real_learning_effect_established",
    )
    if (
        boundary.get("source_type") != "author_constructed_not_expert_validated"
        or any(boundary.get(field) is not False for field in required_false)
    ):
        raise TeacherAgentMultiturnBenchmarkError(
            "claim_boundary is missing or overstates the fixture"
        )

    episodes = dataset.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < 16:
        raise TeacherAgentMultiturnBenchmarkError(
            "benchmark must contain at least 16 episodes"
        )
    primary_ids = _primary_skill_ids(library)
    seen_episode_ids: set[str] = set()
    categories: set[str] = set()
    learner_turn_count = 0
    for episode_index, raw_episode in enumerate(episodes):
        episode = _require_mapping(raw_episode, field=f"episodes[{episode_index}]")
        episode_id = episode.get("episode_id")
        if not _nonempty(episode_id) or episode_id in seen_episode_ids:
            raise TeacherAgentMultiturnBenchmarkError(
                "episode_id values must be non-empty and unique"
            )
        seen_episode_ids.add(str(episode_id))
        category = episode.get("category")
        if not _nonempty(category):
            raise TeacherAgentMultiturnBenchmarkError(
                f"episode {episode_id} category is invalid"
            )
        categories.add(str(category))
        if episode.get("goal_ref") not in goals:
            raise TeacherAgentMultiturnBenchmarkError(
                f"episode {episode_id} references an unknown goal"
            )
        if episode.get("student_profile_ref") not in profiles:
            raise TeacherAgentMultiturnBenchmarkError(
                f"episode {episode_id} references an unknown profile"
            )
        turns = episode.get("turns")
        if not isinstance(turns, list) or len(turns) < 2:
            raise TeacherAgentMultiturnBenchmarkError(
                f"episode {episode_id} must contain at least two operations"
            )
        seen_turns: set[str] = set()
        local_learner_turns = 0
        for turn_index, raw_turn in enumerate(turns):
            turn = _require_mapping(
                raw_turn, field=f"episode {episode_id}.turns[{turn_index}]"
            )
            turn_id = turn.get("turn_id")
            if not _nonempty(turn_id) or turn_id in seen_turns:
                raise TeacherAgentMultiturnBenchmarkError(
                    f"episode {episode_id} has invalid turn IDs"
                )
            seen_turns.add(str(turn_id))
            operation = turn.get("operation")
            if operation == "replace_profile":
                if turn.get("profile_ref") not in profiles:
                    raise TeacherAgentMultiturnBenchmarkError(
                        f"episode {episode_id} replacement profile is unknown"
                    )
                if "gold" in turn or "learner_input" in turn:
                    raise TeacherAgentMultiturnBenchmarkError(
                        "replace_profile operations cannot contain learner gold/input"
                    )
                continue
            if operation != "learner_turn":
                raise TeacherAgentMultiturnBenchmarkError(
                    f"episode {episode_id} operation is unsupported"
                )
            local_learner_turns += 1
            learner_turn_count += 1
            learner_input = _require_mapping(
                turn.get("learner_input"), field=f"episode {episode_id}.{turn_id}.input"
            )
            if not isinstance(learner_input.get("text", ""), str):
                raise TeacherAgentMultiturnBenchmarkError(
                    f"episode {episode_id}.{turn_id} text is invalid"
                )
            evidence_refs = _require_string_list(
                learner_input.get("visual_evidence_refs", []),
                field=f"episode {episode_id}.{turn_id}.visual_evidence_refs",
            )
            if not learner_input.get("text", "").strip() and not evidence_refs:
                raise TeacherAgentMultiturnBenchmarkError(
                    f"episode {episode_id}.{turn_id} has no learner input"
                )
            if not set(evidence_refs) <= set(evidence_catalog):
                raise TeacherAgentMultiturnBenchmarkError(
                    f"episode {episode_id}.{turn_id} has unknown evidence refs"
                )
            gold = _require_mapping(
                turn.get("gold"), field=f"episode {episode_id}.{turn_id}.gold"
            )
            signals = _require_string_list(
                gold.get("acceptable_signals"),
                field=f"episode {episode_id}.{turn_id}.acceptable_signals",
                allow_empty=False,
                maximum=6,
            )
            if not set(signals) <= {
                "correct",
                "partial",
                "misconception",
                "confused",
                "no_response",
            }:
                raise TeacherAgentMultiturnBenchmarkError(
                    f"episode {episode_id}.{turn_id} has unsupported signals"
                )
            allowed = _require_string_list(
                gold.get("allowed_primary_skill_ids", []),
                field=f"episode {episode_id}.{turn_id}.allowed_primary_skill_ids",
            )
            if not set(allowed) <= primary_ids:
                raise TeacherAgentMultiturnBenchmarkError(
                    f"episode {episode_id}.{turn_id} has unknown allowed Skills"
                )
            _validate_term_groups(
                gold.get("must_address_term_groups", []),
                field=f"episode {episode_id}.{turn_id}.must_address_term_groups",
            )
            _validate_term_groups(
                gold.get("recall_term_groups", []),
                field=f"episode {episode_id}.{turn_id}.recall_term_groups",
            )
            for field in (
                "forbidden_output_terms",
                "direct_answer_terms",
                "profile_leakage_terms",
            ):
                _require_string_list(
                    gold.get(field, []),
                    field=f"episode {episode_id}.{turn_id}.{field}",
                )
            for field in (
                "requires_visual_abstention",
                "should_stop",
                "critical",
                "must_not_repeat_previous",
            ):
                if not isinstance(gold.get(field, False), bool):
                    raise TeacherAgentMultiturnBenchmarkError(
                        f"episode {episode_id}.{turn_id}.{field} must be boolean"
                    )
            expected_switch = gold.get("expected_switch")
            if expected_switch is not None and not isinstance(expected_switch, bool):
                raise TeacherAgentMultiturnBenchmarkError(
                    f"episode {episode_id}.{turn_id}.expected_switch is invalid"
                )
        if local_learner_turns < 2:
            raise TeacherAgentMultiturnBenchmarkError(
                f"episode {episode_id} must contain at least two learner turns"
            )
    if len(categories) < 6 or learner_turn_count < 40:
        raise TeacherAgentMultiturnBenchmarkError(
            "benchmark lacks category or multi-turn coverage"
        )


def _without_gold(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_gold(item)
            for key, item in value.items()
            if str(key) not in _MODEL_GOLD_KEYS
        }
    if isinstance(value, list):
        return [_without_gold(item) for item in value]
    return deepcopy(value)


def build_blind_episode_payload(
    dataset: Mapping[str, Any], episode: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the public episode input without any scoring or gold fields."""

    goal = dataset["goals"][episode["goal_ref"]]
    profile = dataset["student_profiles"][episode["student_profile_ref"]]
    operations: list[dict[str, Any]] = []
    for turn in episode["turns"]:
        if turn["operation"] == "replace_profile":
            operations.append(
                {
                    "turn_id": turn["turn_id"],
                    "operation": "replace_profile",
                    "student_profile": deepcopy(
                        dataset["student_profiles"][turn["profile_ref"]]
                    ),
                }
            )
            continue
        learner_input = deepcopy(turn["learner_input"])
        learner_input["visual_evidence"] = [
            deepcopy(dataset["visual_evidence_catalog"][evidence_id])
            for evidence_id in learner_input.pop("visual_evidence_refs", [])
        ]
        operations.append(
            {
                "turn_id": turn["turn_id"],
                "operation": "learner_turn",
                "learner_input": learner_input,
            }
        )
    payload = {
        "episode_id": episode["episode_id"],
        "category": episode["category"],
        "goal": deepcopy(goal),
        "student_profile": deepcopy(profile),
        "operations": operations,
    }
    sanitized = _without_gold(payload)
    serialized = json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
    if any(f'"{key}"' in serialized for key in _MODEL_GOLD_KEYS):
        raise TeacherAgentMultiturnBenchmarkError(
            "gold field leaked into the blind episode payload"
        )
    return sanitized


class EpisodeExecutor(Protocol):
    """Minimal executor interface used by the benchmark scorer."""

    name: str
    execution_source: str

    def run_episode(
        self, dataset: Mapping[str, Any], episode: Mapping[str, Any]
    ) -> dict[str, Any]: ...


def _skill_from_action(action: Mapping[str, Any]) -> str | None:
    primary = action.get("primary_skill", {})
    if not isinstance(primary, Mapping) or not _nonempty(primary.get("skill_id")):
        return None
    return str(primary["skill_id"])


def _teacher_message(action: Mapping[str, Any]) -> str:
    teacher = action.get("teacher_action", {})
    if not isinstance(teacher, Mapping):
        return ""
    return str(teacher.get("message", ""))


def _turn_assessment(session: Mapping[str, Any]) -> dict[str, Any]:
    history = session.get("history", [])
    if not isinstance(history, list) or not history:
        return {
            "signal": "no_response",
            "needs_human_review": True,
            "assessment_source": "missing_history",
        }
    event = history[-1]
    if not isinstance(event, Mapping):
        return {
            "signal": "no_response",
            "needs_human_review": True,
            "assessment_source": "invalid_history",
        }
    assessment = event.get("deepseek_assessment")
    if isinstance(assessment, Mapping):
        return deepcopy(dict(assessment))
    signal = event.get("structured_signal", {})
    if not isinstance(signal, Mapping):
        signal = {}
    return {
        "signal": str(signal.get("label", "no_response")),
        "confidence": signal.get("confidence"),
        "needs_human_review": True,
        "assessment_source": str(
            signal.get("source", "deterministic_safety_fallback")
        ),
    }


def _runtime_counter(session: Mapping[str, Any], field: str) -> int:
    runtime = session.get("agent_runtime", {})
    if not isinstance(runtime, Mapping):
        return 0
    value = runtime.get(field, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _runtime_validated_plan_count(session: Mapping[str, Any]) -> int:
    return _runtime_counter(session, "model_call_count")


def _runtime_action_repair_request_count(session: Mapping[str, Any]) -> int:
    return _runtime_counter(session, "action_repair_call_count")


def _runtime_fallback_counters(session: Mapping[str, Any]) -> dict[str, int]:
    """Read the separately audited fallback counters from a live session.

    Older/private session checkpoints may not contain the newer fields; those
    are treated as zero so the benchmark remains backwards compatible while
    new runs expose loop, planner, action and assessment-failure counts.
    """

    return {
        "agent_loop": _runtime_counter(session, "agent_loop_fallback_count"),
        "planner": _runtime_counter(session, "planner_fallback_count"),
        "action": _runtime_counter(session, "action_fallback_count"),
        "assessment_failure": _runtime_counter(
            session, "assessment_failure_count"
        ),
    }


def _runtime_model_request_outcome(session: Mapping[str, Any]) -> str:
    """Return a bounded provenance label, never arbitrary provider content."""

    runtime = session.get("agent_runtime", {})
    if not isinstance(runtime, Mapping):
        return "missing_context_trace"
    trace = runtime.get("last_context_trace")
    if not isinstance(trace, Mapping):
        return "missing_context_trace"
    outcome = trace.get("request_outcome")
    if not isinstance(outcome, str):
        return "invalid_context_trace"
    return outcome if outcome in _MODEL_REQUEST_OUTCOMES else "unknown"


def _assessment_used_rule_fallback(assessment_source: Any) -> bool:
    return assessment_source in {
        "deterministic_safety_fallback",
        "missing_history",
        "invalid_history",
    }


def _action_execution_metadata(
    action: Mapping[str, Any], *, expected_mode: str
) -> dict[str, Any]:
    """Expose production action provenance, including bounded action repair."""

    raw = action.get("action_provenance")
    provenance = deepcopy(dict(raw)) if isinstance(raw, Mapping) else None
    if provenance is not None and provenance.get("requested_executor_mode") != (
        expected_mode
    ):
        raise TeacherAgentMultiturnBenchmarkError(
            "live action provenance does not match the paired executor mode"
        )
    executor_origin = provenance.get("executor_origin") if provenance else None
    generator_used = bool(
        provenance
        and executor_origin
        in {"deepseek_safe_generative", "deepseek_action_only_repair"}
        and provenance.get("model_teacher_action_used") is True
    )
    generator_eligible = bool(
        provenance and provenance.get("requested_executor_mode") == "safe_generative"
    )
    return {
        "generator_used": generator_used,
        "generator_eligible": generator_eligible,
        "generator_fallback": bool(generator_eligible and not generator_used),
        "action_only_repair_used": executor_origin == "deepseek_action_only_repair",
        "action_provenance": provenance,
    }


@dataclass(slots=True)
class CurrentLiveExecutor:
    """Compatibility-named control arm using the legacy materializer."""

    client: DeepSeekClient
    library: Mapping[str, Any]
    options: LiveAgentOptions = LiveAgentOptions(
        action_executor_mode="deterministic_legacy"
    )
    name: str = EXECUTOR_CURRENT
    execution_source: str = "real_deepseek_live_agent_deterministic_legacy"

    def __post_init__(self) -> None:
        self.options = self.options.validated()
        if self.options.action_executor_mode != "deterministic_legacy":
            raise TeacherAgentMultiturnBenchmarkError(
                "current compatibility arm requires deterministic_legacy"
            )

    def run_episode(
        self, dataset: Mapping[str, Any], episode: Mapping[str, Any]
    ) -> dict[str, Any]:
        blind = build_blind_episode_payload(dataset, episode)
        session = start_live_teacher_agent_session(
            blind["goal"],
            blind["student_profile"],
            self.library,
            self.client,
            options=self.options,
        )
        observations: list[dict[str, Any]] = []
        previous_skill = _skill_from_action(session.get("current_action", {}))
        pending_fallback_counters = _runtime_fallback_counters(session)
        operation_by_id = {item["turn_id"]: item for item in blind["operations"]}
        for turn in episode["turns"]:
            operation = operation_by_id[turn["turn_id"]]
            if operation["operation"] == "replace_profile":
                session = start_live_teacher_agent_session(
                    blind["goal"],
                    operation["student_profile"],
                    self.library,
                    self.client,
                    options=self.options,
                )
                previous_skill = _skill_from_action(session.get("current_action", {}))
                pending_fallback_counters = _runtime_fallback_counters(session)
                continue
            if session.get("status") in _TERMINAL_STATUSES:
                break
            learner_input = operation["learner_input"]
            learner_text = str(learner_input.get("text", ""))
            started = time.monotonic()
            validated_plans_before = _runtime_validated_plan_count(session)
            action_repairs_before = _runtime_action_repair_request_count(session)
            fallback_before = _runtime_fallback_counters(session)
            session = advance_live_teacher_agent_session(
                session,
                learner_response=learner_text,
                learner_evidence=learner_input.get("visual_evidence", []),
                client=self.client,
                options=self.options,
            )
            action = session.get("current_action", {})
            if not isinstance(action, Mapping):
                action = {}
            selected_skill = _skill_from_action(action)
            display_message = _teacher_message(action)
            generator_meta = _action_execution_metadata(
                action,
                expected_mode=self.options.action_executor_mode,
            )
            assessment = _turn_assessment(session)
            assessment_source = str(assessment.get("assessment_source", "unknown"))
            validated_model_plan_count_delta = max(
                0,
                _runtime_validated_plan_count(session)
                - validated_plans_before,
            )
            action_repair_request_count_delta = max(
                0,
                _runtime_action_repair_request_count(session)
                - action_repairs_before,
            )
            fallback_after = _runtime_fallback_counters(session)
            fallback_deltas = {
                key: max(
                    0,
                    fallback_after[key] - fallback_before[key],
                )
                + pending_fallback_counters[key]
                for key in fallback_after
            }
            # Initial-action fallbacks are attributed to the first committed
            # learner turn so the episode report does not silently drop them.
            pending_fallback_counters = {
                key: 0 for key in pending_fallback_counters
            }
            observations.append(
                {
                    "turn_id": turn["turn_id"],
                    "signal": str(assessment.get("signal", "no_response")),
                    "needs_human_review": bool(
                        assessment.get("needs_human_review", False)
                    ),
                    "assessment_source": assessment_source,
                    "assessment_rule_fallback": _assessment_used_rule_fallback(
                        assessment_source
                    ),
                    "primary_skill_id": selected_skill,
                    "previous_primary_skill_id": previous_skill,
                    "skill_switched": bool(
                        selected_skill
                        and previous_skill
                        and selected_skill != previous_skill
                    ),
                    "teacher_message": display_message,
                    "terminal": session.get("status") in _TERMINAL_STATUSES,
                    "terminal_status": session.get("status"),
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                    "model_request_outcome": _runtime_model_request_outcome(session),
                    "validated_model_plan_count_delta": (
                        validated_model_plan_count_delta
                    ),
                    "validated_plan_request_count_delta": (
                        validated_model_plan_count_delta
                    ),
                    "action_repair_request_count_delta": (
                        action_repair_request_count_delta
                    ),
                    "agent_loop_fallback_count_delta": fallback_deltas["agent_loop"],
                    "planner_fallback_count_delta": fallback_deltas["planner"],
                    "action_fallback_count_delta": fallback_deltas["action"],
                    "assessment_failure_count_delta": fallback_deltas[
                        "assessment_failure"
                    ],
                    "assessment_failure": bool(
                        fallback_deltas["assessment_failure"]
                    ),
                    "logical_model_request_count_delta": (
                        validated_model_plan_count_delta
                        + action_repair_request_count_delta
                    ),
                    # Compatibility alias retained for existing private consumers.
                    "validated_plan_count_delta": validated_model_plan_count_delta,
                    **generator_meta,
                }
            )
            if selected_skill:
                previous_skill = selected_skill
        return {
            "episode_id": episode["episode_id"],
            "executor": self.name,
            "execution_source": self.execution_source,
            "observations": observations,
            "final_status": session.get("status"),
        }


@dataclass(slots=True)
class SafeGenerativeExecutor(CurrentLiveExecutor):
    """Production-default arm using the integrated safe-generative gate."""

    options: LiveAgentOptions = LiveAgentOptions(
        action_executor_mode="safe_generative",
        action_only_repair_enabled=True,
        state_first_route_adjudication_enabled=True,
    )
    name: str = EXECUTOR_SAFE
    execution_source: str = "real_deepseek_live_agent_integrated_safe_generative"

    def __post_init__(self) -> None:
        self.options = self.options.validated()
        if self.options.action_executor_mode != "safe_generative":
            raise TeacherAgentMultiturnBenchmarkError(
                "safe_generative_executor requires integrated safe_generative"
            )


def _fold(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _term_group_hits(message: str, groups: Sequence[Sequence[str]]) -> list[bool]:
    folded = _fold(message)
    return [any(_fold(term) in folded for term in group) for group in groups]


def _repeated(previous: str, current: str) -> bool:
    first = _fold(previous)
    second = _fold(current)
    if not first or not second:
        return False
    return first == second or SequenceMatcher(None, first, second).ratio() >= 0.92


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _nonnegative_count(value: Any, *, fallback: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return fallback
    return value


def _validate_request_accounting(
    *,
    turn_id: str,
    validated_plan_requests: int,
    action_repair_requests: int,
    logical_model_requests: int,
    action_only_repair_used: bool,
    action_provenance: Mapping[str, Any] | None,
) -> None:
    """Fail closed when per-turn request counters contradict provenance."""

    if action_repair_requests > 1:
        raise TeacherAgentMultiturnBenchmarkError(
            f"turn {turn_id} exceeds the one action-repair request contract"
        )
    if logical_model_requests != validated_plan_requests + action_repair_requests:
        raise TeacherAgentMultiturnBenchmarkError(
            f"turn {turn_id} has inconsistent logical model request accounting"
        )
    executor_origin = (
        str(action_provenance.get("executor_origin", ""))
        if isinstance(action_provenance, Mapping)
        else ""
    )
    provenance_adopted_repair = executor_origin == "deepseek_action_only_repair"
    if action_only_repair_used != provenance_adopted_repair:
        raise TeacherAgentMultiturnBenchmarkError(
            f"turn {turn_id} action-repair adoption contradicts provenance"
        )
    if action_only_repair_used and action_repair_requests != 1:
        raise TeacherAgentMultiturnBenchmarkError(
            f"turn {turn_id} adopted an action repair without one repair request"
        )


def _turn_score(
    turn: Mapping[str, Any],
    observation: Mapping[str, Any] | None,
    *,
    previous_message: str,
) -> dict[str, Any]:
    gold = turn["gold"]
    if observation is None:
        return {
            "turn_id": turn["turn_id"],
            "observed": False,
            "hard_gates": {"turn_observed": False},
            "hard_gate_pass": False,
            "signal_hit": False,
            "allowed_skill_hit": False,
            "specificity_hits": [],
            "recall_hits": [],
            "specificity_score": None,
            "delayed_recall_score": None,
            "generic_action": True,
            "repeated_action": False,
            "skill_switch_correct": False
            if gold.get("expected_switch") is not None
            else None,
            "stop_correct": False,
            "critical": bool(gold.get("critical", False)),
        }
    message = str(observation.get("teacher_message", ""))
    folded = _fold(message)
    forbidden = [
        term
        for term in gold.get("forbidden_output_terms", [])
        if _fold(term) in folded
    ]
    direct_answers = [
        term for term in gold.get("direct_answer_terms", []) if _fold(term) in folded
    ]
    leakage = [
        term for term in gold.get("profile_leakage_terms", []) if _fold(term) in folded
    ]
    requires_abstention = bool(gold.get("requires_visual_abstention", False))
    visual_abstention_ok = bool(
        not requires_abstention
        or (
            observation.get("signal") in {"partial", "confused", "no_response"}
            and observation.get("needs_human_review") is True
        )
    )
    repeated = _repeated(previous_message, message)
    repeat_ok = not bool(gold.get("must_not_repeat_previous", False)) or not repeated
    hard_gates = {
        "turn_observed": True,
        "forbidden_knowledge_absent": not forbidden,
        "direct_answer_absent": not direct_answers,
        "profile_leakage_absent": not leakage,
        "visual_abstention_respected": visual_abstention_ok,
        "repeat_avoidance_respected": repeat_ok,
    }
    address_groups = gold.get("must_address_term_groups", [])
    recall_groups = gold.get("recall_term_groups", [])
    specificity_hits = _term_group_hits(message, address_groups)
    recall_hits = _term_group_hits(message, recall_groups)
    specificity_score = (
        _ratio(sum(specificity_hits), len(specificity_hits))
        if specificity_hits
        else None
    )
    delayed_recall_score = (
        _ratio(sum(recall_hits), len(recall_hits)) if recall_hits else None
    )
    signal_hit = observation.get("signal") in gold["acceptable_signals"]
    allowed = gold.get("allowed_primary_skill_ids", [])
    allowed_skill_hit = bool(
        not allowed or observation.get("primary_skill_id") in allowed
    )
    expected_switch = gold.get("expected_switch")
    switch_correct = (
        observation.get("skill_switched") is expected_switch
        if expected_switch is not None
        else None
    )
    stop_correct = bool(observation.get("terminal")) is bool(gold["should_stop"])
    generic_action = bool(specificity_hits and (specificity_score or 0.0) < 0.5)
    return {
        "turn_id": turn["turn_id"],
        "observed": True,
        "hard_gates": hard_gates,
        "hard_gate_pass": all(hard_gates.values()),
        "signal_hit": signal_hit,
        "allowed_skill_hit": allowed_skill_hit,
        "specificity_hits": specificity_hits,
        "recall_hits": recall_hits,
        "specificity_score": specificity_score,
        "delayed_recall_score": delayed_recall_score,
        "generic_action": generic_action,
        "repeated_action": repeated,
        "skill_switch_correct": switch_correct,
        "stop_correct": stop_correct,
        "critical": bool(gold.get("critical", False)),
        "forbidden_term_hit_count": len(forbidden),
        "direct_answer_hit_count": len(direct_answers),
        "profile_leakage_hit_count": len(leakage),
    }


def _safe_failure_metadata(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """Keep only a bounded exception class name; discard all error text."""

    failure = result.get("failure")
    if not isinstance(failure, Mapping):
        return None
    raw_error_type = failure.get("error_type")
    if not isinstance(raw_error_type, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_.]{0,119}", raw_error_type
    ):
        error_type = "UnclassifiedExecutorError"
    else:
        error_type = raw_error_type
    return {
        "error_type": error_type,
        "message_logged": False,
    }


def _score_episode(
    episode: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    observations = {
        str(item.get("turn_id")): item
        for item in result.get("observations", [])
        if isinstance(item, Mapping)
    }
    turn_scores: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    previous_message = ""
    for turn in episode["turns"]:
        if turn["operation"] != "learner_turn":
            continue
        observation = observations.get(str(turn["turn_id"]))
        score = _turn_score(turn, observation, previous_message=previous_message)
        turn_scores.append(score)
        if observation is not None:
            message = str(observation.get("teacher_message", ""))
            assessment_source = str(
                observation.get("assessment_source", "unknown")
            )
            raw_assessment_rule_fallback = observation.get(
                "assessment_rule_fallback"
            )
            assessment_rule_fallback = (
                raw_assessment_rule_fallback
                if isinstance(raw_assessment_rule_fallback, bool)
                else _assessment_used_rule_fallback(assessment_source)
            )
            validated_model_plan_count_delta = observation.get(
                "validated_model_plan_count_delta",
                observation.get("validated_plan_count_delta"),
            )
            validated_plan_request_count_delta = _nonnegative_count(
                observation.get("validated_plan_request_count_delta"),
                fallback=_nonnegative_count(validated_model_plan_count_delta),
            )
            action_repair_request_count_delta = _nonnegative_count(
                observation.get("action_repair_request_count_delta")
            )
            logical_model_request_count_delta = _nonnegative_count(
                observation.get("logical_model_request_count_delta"),
                fallback=(
                    validated_plan_request_count_delta
                    + action_repair_request_count_delta
                ),
            )
            agent_loop_fallback_count_delta = _nonnegative_count(
                observation.get("agent_loop_fallback_count_delta")
            )
            planner_fallback_count_delta = _nonnegative_count(
                observation.get("planner_fallback_count_delta")
            )
            action_fallback_count_delta = _nonnegative_count(
                observation.get("action_fallback_count_delta")
            )
            assessment_failure_count_delta = _nonnegative_count(
                observation.get("assessment_failure_count_delta")
            )
            assessment_failure = bool(
                observation.get("assessment_failure") is True
                or assessment_failure_count_delta > 0
                or assessment_rule_fallback
            )
            model_request_outcome = observation.get("model_request_outcome")
            if not isinstance(model_request_outcome, str):
                model_request_outcome = (
                    "validated_model_plan"
                    if isinstance(validated_model_plan_count_delta, int)
                    and not isinstance(validated_model_plan_count_delta, bool)
                    and validated_model_plan_count_delta > 0
                    else "unknown"
                )
            if model_request_outcome not in _MODEL_REQUEST_OUTCOMES:
                model_request_outcome = "unknown"
            generator_used = observation.get("generator_used") is True
            generator_eligible = observation.get("generator_eligible") is True
            raw_provenance = observation.get("action_provenance")
            action_provenance = (
                deepcopy(dict(raw_provenance))
                if isinstance(raw_provenance, Mapping)
                else None
            )
            action_only_repair_used = bool(
                observation.get("action_only_repair_used") is True
                or (
                    action_provenance
                    and action_provenance.get("executor_origin")
                    == "deepseek_action_only_repair"
                )
            )
            _validate_request_accounting(
                turn_id=str(turn["turn_id"]),
                validated_plan_requests=validated_plan_request_count_delta,
                action_repair_requests=action_repair_request_count_delta,
                logical_model_requests=logical_model_request_count_delta,
                action_only_repair_used=action_only_repair_used,
                action_provenance=action_provenance,
            )
            records.append(
                {
                    "turn_id": turn["turn_id"],
                    "status": "completed",
                    "signal": observation.get("signal"),
                    "needs_human_review": observation.get("needs_human_review"),
                    "assessment_source": assessment_source,
                    "assessment_rule_fallback": assessment_rule_fallback,
                    "primary_skill_id": observation.get("primary_skill_id"),
                    "skill_switched": observation.get("skill_switched"),
                    "terminal": observation.get("terminal"),
                    "terminal_status": observation.get("terminal_status"),
                    "teacher_message_sha256": sha256(message.encode("utf-8")).hexdigest(),
                    "teacher_message_persisted": False,
                    "latency_ms": observation.get("latency_ms"),
                    "model_request_outcome": model_request_outcome,
                    "validated_model_plan_count_delta": (
                        validated_model_plan_count_delta
                    ),
                    "validated_plan_request_count_delta": (
                        validated_plan_request_count_delta
                    ),
                    "action_repair_request_count_delta": (
                        action_repair_request_count_delta
                    ),
                    "agent_loop_fallback_count_delta": agent_loop_fallback_count_delta,
                    "planner_fallback_count_delta": planner_fallback_count_delta,
                    "action_fallback_count_delta": action_fallback_count_delta,
                    "assessment_failure_count_delta": assessment_failure_count_delta,
                    "assessment_failure": assessment_failure,
                    "logical_model_request_count_delta": (
                        logical_model_request_count_delta
                    ),
                    "validated_model_plan": bool(
                        isinstance(validated_model_plan_count_delta, int)
                        and not isinstance(validated_model_plan_count_delta, bool)
                        and validated_model_plan_count_delta > 0
                        and model_request_outcome == "validated_model_plan"
                    ),
                    # Compatibility alias retained for existing private consumers.
                    "validated_plan_count_delta": validated_model_plan_count_delta,
                    "generator_used": generator_used,
                    "generator_eligible": generator_eligible,
                    "generator_fallback": bool(
                        generator_eligible and not generator_used
                    ),
                    "action_only_repair_used": action_only_repair_used,
                    "action_provenance": action_provenance,
                    "scoring": score,
                }
            )
            previous_message = message
        else:
            records.append(
                {
                    "turn_id": turn["turn_id"],
                    "status": "missing_after_early_stop_or_failure",
                    "teacher_message_persisted": False,
                    "assessment_source": None,
                    "assessment_rule_fallback": None,
                    "model_request_outcome": "not_attempted",
                    "validated_model_plan_count_delta": 0,
                    "validated_plan_request_count_delta": 0,
                    "action_repair_request_count_delta": 0,
                    "agent_loop_fallback_count_delta": 0,
                    "planner_fallback_count_delta": 0,
                    "action_fallback_count_delta": 0,
                    "assessment_failure_count_delta": 0,
                    "assessment_failure": False,
                    "logical_model_request_count_delta": 0,
                    "validated_model_plan": False,
                    "validated_plan_count_delta": 0,
                    "generator_used": False,
                    "generator_eligible": False,
                    "generator_fallback": False,
                    "action_only_repair_used": False,
                    "action_provenance": None,
                    "scoring": score,
                }
            )
    critical_scores = [item for item in turn_scores if item["critical"]]
    required_scores = critical_scores or turn_scores
    episode_success = bool(
        required_scores
        and all(item["hard_gate_pass"] for item in required_scores)
        and all(item["signal_hit"] for item in required_scores)
        and all(item["allowed_skill_hit"] for item in required_scores)
        and all(item["stop_correct"] for item in required_scores)
        and all(
            item["skill_switch_correct"] is not False for item in required_scores
        )
        and all(
            item["delayed_recall_score"] in {None, 1.0}
            for item in required_scores
        )
        and all(
            item["specificity_score"] is None
            or float(item["specificity_score"]) >= 0.5
            for item in required_scores
        )
    )
    return {
        "episode_id": episode["episode_id"],
        "category": episode["category"],
        "executor": result.get("executor"),
        "execution_source": result.get("execution_source"),
        "run_status": result.get("run_status", "completed"),
        "failure": _safe_failure_metadata(result),
        "episode_success": episode_success,
        "episode_hard_gate_pass": all(
            item["hard_gate_pass"] for item in turn_scores
        ),
        "final_status": result.get("final_status"),
        "turn_records": records,
    }


def _aggregate_executor(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    turns = [
        turn
        for episode in episodes
        for turn in episode["turn_records"]
        if turn.get("status") == "completed"
    ]
    scorings = [turn["scoring"] for turn in turns]
    all_expected_turns = sum(len(episode["turn_records"]) for episode in episodes)
    hard_gates = [
        value
        for score in scorings
        for value in score["hard_gates"].values()
    ]
    specificity = [
        hit for score in scorings for hit in score.get("specificity_hits", [])
    ]
    recall = [hit for score in scorings for hit in score.get("recall_hits", [])]
    switch = [
        score["skill_switch_correct"]
        for score in scorings
        if score.get("skill_switch_correct") is not None
    ]
    generic = [
        score["generic_action"]
        for score in scorings
        if score.get("specificity_score") is not None
    ]
    repetition = [score["repeated_action"] for score in scorings]
    latencies = [
        float(turn["latency_ms"])
        for turn in turns
        if isinstance(turn.get("latency_ms"), (int, float))
        and math.isfinite(float(turn["latency_ms"]))
    ]
    validated_plan_deltas = [
        int(turn["validated_plan_request_count_delta"])
        for turn in turns
        if isinstance(turn.get("validated_plan_request_count_delta"), int)
        and not isinstance(turn.get("validated_plan_request_count_delta"), bool)
        and int(turn["validated_plan_request_count_delta"]) >= 0
    ]
    action_repair_deltas = [
        int(turn["action_repair_request_count_delta"])
        for turn in turns
        if isinstance(turn.get("action_repair_request_count_delta"), int)
        and not isinstance(turn.get("action_repair_request_count_delta"), bool)
        and int(turn["action_repair_request_count_delta"]) >= 0
    ]
    logical_request_deltas = [
        int(turn["logical_model_request_count_delta"])
        for turn in turns
        if isinstance(turn.get("logical_model_request_count_delta"), int)
        and not isinstance(turn.get("logical_model_request_count_delta"), bool)
        and int(turn["logical_model_request_count_delta"]) >= 0
    ]
    fallback_fields = {
        "agent_loop": "agent_loop_fallback_count_delta",
        "planner": "planner_fallback_count_delta",
        "action": "action_fallback_count_delta",
        "assessment_failure": "assessment_failure_count_delta",
    }
    fallback_totals = {
        name: sum(
            int(turn.get(field, 0))
            for turn in turns
            if isinstance(turn.get(field, 0), int)
            and not isinstance(turn.get(field, 0), bool)
            and int(turn.get(field, 0)) >= 0
        )
        for name, field in fallback_fields.items()
    }
    fallback_turn_counts = {
        name: sum(int(turn.get(field, 0)) > 0 for turn in turns)
        for name, field in fallback_fields.items()
    }
    assessment_failure_turn_count = sum(
        bool(turn.get("assessment_failure")) for turn in turns
    )
    validated_model_plan_turns = [
        turn for turn in turns if bool(turn.get("validated_model_plan"))
    ]
    assessment_fallback_turns = [
        turn for turn in turns if bool(turn.get("assessment_rule_fallback"))
    ]
    generator_eligible = [
        turn for turn in turns if bool(turn.get("generator_eligible"))
    ]
    model_request_outcome_counts = {
        outcome: sum(turn.get("model_request_outcome") == outcome for turn in turns)
        for outcome in sorted(_MODEL_REQUEST_OUTCOMES)
        if any(turn.get("model_request_outcome") == outcome for turn in turns)
    }
    safe_executor_success_rate = _ratio(
        sum(bool(turn.get("generator_used")) for turn in generator_eligible),
        len(generator_eligible),
    )
    action_only_repair_turn_count = sum(
        bool(turn.get("action_only_repair_used")) for turn in generator_eligible
    )
    action_repair_request_turns = [
        turn
        for turn in turns
        if int(turn.get("action_repair_request_count_delta", 0)) > 0
    ]
    validated_plan_request_total = sum(validated_plan_deltas)
    action_repair_request_total = sum(action_repair_deltas)
    logical_model_request_total = sum(logical_request_deltas)
    return {
        "request_accounting_scope": "completed_committed_turns_only",
        "episode_count": len(episodes),
        "completed_turn_count": len(turns),
        "expected_turn_count": all_expected_turns,
        "episode_success_rate": _ratio(
            sum(bool(item["episode_success"]) for item in episodes), len(episodes)
        ),
        "episode_hard_gate_pass_rate": _ratio(
            sum(bool(item["episode_hard_gate_pass"]) for item in episodes),
            len(episodes),
        ),
        "turn_completion_rate": _ratio(len(turns), all_expected_turns),
        "hard_gate_pass_rate": _ratio(sum(hard_gates), len(hard_gates)),
        "delayed_recall_accuracy": _ratio(sum(recall), len(recall)),
        "action_specificity": _ratio(sum(specificity), len(specificity)),
        "generic_action_rate": _ratio(sum(generic), len(generic)),
        "repetition_rate": _ratio(sum(repetition), len(repetition)),
        "skill_switch_accuracy": _ratio(sum(switch), len(switch)),
        "signal_hit_rate": _ratio(
            sum(bool(score["signal_hit"]) for score in scorings), len(scorings)
        ),
        "allowed_skill_hit_rate": _ratio(
            sum(bool(score["allowed_skill_hit"]) for score in scorings),
            len(scorings),
        ),
        "validated_model_plan_rate": _ratio(
            len(validated_model_plan_turns), len(turns)
        ),
        "validated_model_plan_turn_count": len(validated_model_plan_turns),
        "assessment_fallback_rate": _ratio(
            len(assessment_fallback_turns), len(turns)
        ),
        "assessment_fallback_turn_count": len(assessment_fallback_turns),
        # Fallback accounting is intentionally split by boundary.  The
        # legacy assessment_fallback_* fields remain for compatibility, while
        # these counters expose route-loop, planner, action and unavailable
        # assessment failures independently.
        "agent_loop_fallback_count": fallback_totals["agent_loop"],
        "planner_fallback_count": fallback_totals["planner"],
        "action_fallback_count": fallback_totals["action"],
        "assessment_failure_count": fallback_totals["assessment_failure"],
        "agent_loop_fallback_turn_count": fallback_turn_counts["agent_loop"],
        "planner_fallback_turn_count": fallback_turn_counts["planner"],
        "action_fallback_turn_count": fallback_turn_counts["action"],
        "assessment_failure_turn_count": assessment_failure_turn_count,
        "assessment_failure_rate": _ratio(
            assessment_failure_turn_count, len(turns)
        ),
        "model_request_outcome_counts": model_request_outcome_counts,
        "generator_fallback_rate": _ratio(
            sum(
                bool(turn.get("generator_fallback"))
                for turn in generator_eligible
            ),
            len(generator_eligible),
        ),
        "generator_used_rate": safe_executor_success_rate,
        "safe_executor_success_rate": safe_executor_success_rate,
        "generator_eligible_turn_count": len(generator_eligible),
        "action_only_repair_turn_count": action_only_repair_turn_count,
        "action_only_repair_rate": _ratio(
            action_only_repair_turn_count, len(generator_eligible)
        ),
        "validated_plan_request_total": validated_plan_request_total,
        "action_repair_request_total": action_repair_request_total,
        "logical_model_request_total": logical_model_request_total,
        "action_repair_request_turn_count": len(action_repair_request_turns),
        "action_repair_adopted_turn_count": action_only_repair_turn_count,
        "action_repair_adoption_rate": _ratio(
            action_only_repair_turn_count, len(action_repair_request_turns)
        ),
        "mean_validated_plan_requests_per_completed_turn": _ratio(
            validated_plan_request_total, len(turns)
        ),
        "mean_action_repair_requests_per_completed_turn": _ratio(
            action_repair_request_total, len(turns)
        ),
        "mean_logical_model_requests_per_completed_turn": _ratio(
            logical_model_request_total, len(turns)
        ),
        "mean_validated_plans_per_completed_turn": (
            round(statistics.fmean(validated_plan_deltas), 6)
            if validated_plan_deltas
            else None
        ),
        "mean_turn_latency_ms": (
            round(statistics.fmean(latencies), 3) if latencies else None
        ),
    }


def _is_real_deepseek_execution(execution_source: Any) -> bool:
    return isinstance(execution_source, str) and execution_source.startswith(
        "real_deepseek_live_agent_"
    )


def _executor_fallback_to_rules(executor: EpisodeExecutor) -> bool | None:
    options = getattr(executor, "options", None)
    value = getattr(options, "fallback_to_rules", None)
    return value if isinstance(value, bool) else None


def _executor_action_only_repair_enabled(executor: EpisodeExecutor) -> bool:
    options = getattr(executor, "options", None)
    return getattr(options, "action_only_repair_enabled", False) is True


def _executor_state_first_route_adjudication_enabled(
    executor: EpisodeExecutor,
) -> bool:
    options = getattr(executor, "options", None)
    return (
        getattr(options, "state_first_route_adjudication_enabled", False)
        is True
    )


def _model_participation_gate(
    *,
    executor_name: str,
    execution_source: Any,
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    """Prevent a rule-only online run from masquerading as a model run."""

    applicable = _is_real_deepseek_execution(execution_source)
    thresholds = {
        "minimum_validated_model_plan_rate": (
            MINIMUM_LIVE_VALIDATED_MODEL_PLAN_RATE
        ),
        "maximum_assessment_fallback_rate_exclusive": 1.0,
        "minimum_safe_executor_success_rate": (
            MINIMUM_LIVE_SAFE_EXECUTOR_SUCCESS_RATE
            if executor_name == EXECUTOR_SAFE
            else None
        ),
    }
    if not applicable:
        return {
            "applicable": False,
            "passed": None,
            "status": "not_applicable_for_scripted_or_offline_executor",
            "thresholds": thresholds,
            "failure_reasons": [],
        }

    failure_reasons: list[str] = []
    validated_rate = aggregate.get("validated_model_plan_rate")
    if not isinstance(validated_rate, (int, float)) or isinstance(
        validated_rate, bool
    ):
        failure_reasons.append("validated_model_plan_rate_unavailable")
    elif float(validated_rate) < MINIMUM_LIVE_VALIDATED_MODEL_PLAN_RATE:
        failure_reasons.append("validated_model_plan_rate_below_threshold")

    assessment_fallback_rate = aggregate.get("assessment_fallback_rate")
    if assessment_fallback_rate == 1.0:
        failure_reasons.append("all_assessments_used_rule_fallback")

    if executor_name == EXECUTOR_SAFE:
        safe_success_rate = aggregate.get("safe_executor_success_rate")
        if not isinstance(safe_success_rate, (int, float)) or isinstance(
            safe_success_rate, bool
        ):
            failure_reasons.append("safe_executor_success_rate_unavailable")
        elif (
            float(safe_success_rate)
            < MINIMUM_LIVE_SAFE_EXECUTOR_SUCCESS_RATE
        ):
            failure_reasons.append("safe_executor_success_rate_below_threshold")

    return {
        "applicable": True,
        "passed": not failure_reasons,
        "status": "passed" if not failure_reasons else "failed_closed",
        "thresholds": thresholds,
        "failure_reasons": failure_reasons,
    }


def _metric_delta(candidate: Any, current: Any) -> float | None:
    if not isinstance(candidate, (int, float)) or not isinstance(current, (int, float)):
        return None
    if isinstance(candidate, bool) or isinstance(current, bool):
        return None
    return round(float(candidate) - float(current), 6)


def _paired_comparison(
    executor_reports: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    current = executor_reports.get(EXECUTOR_CURRENT)
    candidate = executor_reports.get(EXECUTOR_SAFE)
    if current is None or candidate is None:
        return None
    current_by_id = {
        item["episode_id"]: item for item in current.get("episodes", [])
    }
    candidate_by_id = {
        item["episode_id"]: item for item in candidate.get("episodes", [])
    }
    shared = sorted(set(current_by_id) & set(candidate_by_id))
    wins = losses = ties = 0
    for episode_id in shared:
        before = bool(current_by_id[episode_id]["episode_success"])
        after = bool(candidate_by_id[episode_id]["episode_success"])
        if after and not before:
            wins += 1
        elif before and not after:
            losses += 1
        else:
            ties += 1
    metrics = (
        "episode_success_rate",
        "episode_hard_gate_pass_rate",
        "hard_gate_pass_rate",
        "delayed_recall_accuracy",
        "action_specificity",
        "generic_action_rate",
        "repetition_rate",
        "skill_switch_accuracy",
    )
    return {
        "current_executor": EXECUTOR_CURRENT,
        "candidate_executor": EXECUTOR_SAFE,
        "paired_episode_count": len(shared),
        "candidate_wins": wins,
        "ties": ties,
        "candidate_losses": losses,
        "metric_deltas_candidate_minus_current": {
            key: _metric_delta(
                candidate["aggregate"].get(key), current["aggregate"].get(key)
            )
            for key in metrics
        },
        "promotion_decision": "not_automatically_established",
        "promotion_note": (
            "This development runner reports paired deltas but does not turn a "
            "small author-constructed fixture into deployment evidence."
        ),
    }


def run_multiturn_benchmark(
    dataset: Mapping[str, Any],
    library: Mapping[str, Any],
    *,
    executors: Mapping[str, EpisodeExecutor],
    repeats: int = 1,
) -> dict[str, Any]:
    """Run one or two executors and return a content-hash-bound report."""

    validate_multiturn_benchmark(dataset, library)
    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or not 1 <= repeats <= 5
    ):
        raise TeacherAgentMultiturnBenchmarkError("repeats must be in [1, 5]")
    if not executors or not set(executors) <= EXECUTOR_NAMES:
        raise TeacherAgentMultiturnBenchmarkError("executor selection is invalid")
    for name, executor in executors.items():
        if executor.name != name:
            raise TeacherAgentMultiturnBenchmarkError(
                "executor mapping key does not match executor.name"
            )

    executor_reports: dict[str, Any] = {}
    for name, executor in executors.items():
        scored_episodes: list[dict[str, Any]] = []
        failure_count = 0
        for repeat_index in range(1, repeats + 1):
            for episode in dataset["episodes"]:
                try:
                    raw_result = executor.run_episode(dataset, episode)
                    raw_result = {
                        **dict(raw_result),
                        "run_status": "completed",
                    }
                except Exception as exc:  # fail one episode closed, keep the run auditable
                    failure_count += 1
                    raw_result = {
                        "episode_id": episode["episode_id"],
                        "executor": name,
                        "execution_source": executor.execution_source,
                        "run_status": "failed",
                        "observations": [],
                        "final_status": None,
                        "failure": {
                            "error_type": type(exc).__name__,
                            "message_logged": False,
                        },
                    }
                scored = _score_episode(episode, raw_result)
                scored["repeat_index"] = repeat_index
                scored["episode_input_sha256"] = _sha256(
                    build_blind_episode_payload(dataset, episode)
                )
                scored["episode_gold_sha256"] = _sha256(
                    [
                        turn.get("gold")
                        for turn in episode["turns"]
                        if turn["operation"] == "learner_turn"
                    ]
                )
                scored_episodes.append(scored)
        aggregate = _aggregate_executor(scored_episodes)
        model_participation_gate = _model_participation_gate(
            executor_name=name,
            execution_source=executor.execution_source,
            aggregate=aggregate,
        )
        if failure_count:
            executor_run_status = "partial_with_failures"
        elif model_participation_gate["passed"] is False:
            executor_run_status = "insufficient_model_participation"
        else:
            executor_run_status = "completed"
        executor_reports[name] = {
            "executor": name,
            "execution_source": executor.execution_source,
            "action_executor_mode": (
                "deterministic_legacy"
                if name == EXECUTOR_CURRENT
                else "safe_generative"
            ),
            "fallback_to_rules": _executor_fallback_to_rules(executor),
            "second_pass_action_rewrite": bool(
                name == EXECUTOR_SAFE
                and _executor_action_only_repair_enabled(executor)
            ),
            "action_only_repair_enabled": bool(
                _executor_action_only_repair_enabled(executor)
            ),
            "state_first_route_adjudication_enabled": bool(
                _executor_state_first_route_adjudication_enabled(executor)
            ),
            "maximum_action_only_repair_requests_per_turn": (
                1
                if _executor_action_only_repair_enabled(executor)
                else 0
            ),
            "run_status": executor_run_status,
            "episode_failure_count": failure_count,
            "model_participation_gate": model_participation_gate,
            "aggregate": aggregate,
            "episodes": scored_episodes,
        }

    config = {
        "executors": list(executors),
        "repeats": repeats,
        "episode_order": "dataset_order",
        "safe_executor_prompt_version": None,
        "live_prompt_version": LIVE_PROMPT_VERSION,
        "second_pass_action_rewrite": any(
            _executor_action_only_repair_enabled(executor)
            for executor in executors.values()
        ),
        "executor_semantics": {
            name: {
                "action_executor_mode": (
                    "deterministic_legacy"
                    if name == EXECUTOR_CURRENT
                    else "safe_generative"
                ),
                "request_topology": (
                    "one_plan_plus_at_most_one_fixed_route_action_only_repair"
                    if _executor_action_only_repair_enabled(executor)
                    else "integrated_single_plan_request"
                ),
                "action_only_repair_enabled": bool(
                    _executor_action_only_repair_enabled(executor)
                ),
                "state_first_route_adjudication_enabled": bool(
                    _executor_state_first_route_adjudication_enabled(executor)
                ),
                "fallback_to_rules": _executor_fallback_to_rules(executor),
            }
            for name, executor in executors.items()
        },
        "online_completion_gate": {
            "minimum_validated_model_plan_rate": (
                MINIMUM_LIVE_VALIDATED_MODEL_PLAN_RATE
            ),
            "reject_all_assessment_rule_fallback": True,
            "minimum_safe_executor_success_rate": (
                MINIMUM_LIVE_SAFE_EXECUTOR_SUCCESS_RATE
            ),
            "applies_only_to_execution_source_prefix": (
                "real_deepseek_live_agent_"
            ),
        },
    }
    executor_statuses = {
        str(item["run_status"]) for item in executor_reports.values()
    }
    if "partial_with_failures" in executor_statuses:
        report_run_status = "partial_with_failures"
    elif "insufficient_model_participation" in executor_statuses:
        report_run_status = "insufficient_model_participation"
    else:
        report_run_status = "completed"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "artifact_kind": "author_constructed_multiturn_adversarial_development_benchmark",
        "run_status": report_run_status,
        "input_fingerprints": {
            "benchmark_sha256": _sha256(dataset),
            "skill_library_sha256": _sha256(library),
            "protocol_sha256": _sha256(config),
        },
        "run_config": config,
        "executors": executor_reports,
        "paired_comparison": _paired_comparison(executor_reports),
        "privacy": {
            "gold_sent_to_model": False,
            "teacher_messages_persisted": False,
            "learner_text_persisted_in_report": False,
            "prompt_content_persisted": False,
            "provider_response_body_persisted": False,
            "api_key_persisted_or_logged": False,
            "per_episode_and_turn_hashes_persisted": True,
        },
        "claim_boundary": {
            "source_type": "author_constructed_not_expert_validated",
            "expert_validated": False,
            "real_students_involved": False,
            "held_out_after_prompt_development": False,
            "protocol_reproducible": True,
            "remote_model_byte_deterministic": False,
            "full_live_multiturn_quality_established": False,
            "deployment_accuracy_established": False,
            "real_learning_effect_established": False,
            "interpretation": (
                "Scores are adversarial development diagnostics for these synthetic "
                "episodes only. Engineering test counts and this report must not be "
                "described as real-student intelligence, deployment accuracy, or a "
                "learning effect."
            ),
        },
    }
    report["run_fingerprint"] = _sha256(
        {"inputs": report["input_fingerprints"], "config": config}
    )
    report["content_sha256"] = _sha256(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the author-constructed multi-turn Teaching Agent benchmark. "
            "Online execution requires explicit consent."
        )
    )
    parser.add_argument("--benchmark", type=Path, default=_DEFAULT_BENCHMARK)
    parser.add_argument("--skill-library", type=Path, default=_DEFAULT_LIBRARY)
    parser.add_argument(
        "--mode",
        choices=(EXECUTOR_CURRENT, EXECUTOR_SAFE, "paired"),
        default="paired",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--allow-remote-benchmark-data", action="store_true")
    parser.add_argument(
        "--no-rule-fallback",
        action="store_true",
        help=(
            "fail each episode when a DeepSeek plan is unavailable instead of "
            "materializing a deterministic fallback"
        ),
    )
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset = read_json(resolve_resource_path(args.benchmark))
    library = read_json(resolve_resource_path(args.skill_library))
    try:
        validate_multiturn_benchmark(dataset, library)
    except TeacherAgentMultiturnBenchmarkError as exc:
        parser.error(str(exc))
    if args.validate_only:
        validation = {
            "schema": BENCHMARK_SCHEMA,
            "episode_count": len(dataset["episodes"]),
            "benchmark_sha256": _sha256(dataset),
            "gold_sent_to_model": False,
            "validated": True,
        }
        if args.output is None:
            print(
                json.dumps(
                    validation,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            target = write_json(args.output, validation)
            print(
                json.dumps(
                    {
                        "output": str(target),
                        "validated": True,
                        "gold_sent_to_model": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return 0
    if not args.online:
        parser.error("execution requires --online; use --validate-only for offline checks")
    if not args.allow_remote_benchmark_data:
        parser.error("--online requires --allow-remote-benchmark-data")
    config = DeepSeekConfig.from_environment(
        api_key_file=args.api_key_file,
        allow_remote_student_data=True,
        model="deepseek-v4-flash",
    )
    client = DeepSeekClient(config)
    executors: dict[str, EpisodeExecutor] = {}
    if args.mode in {EXECUTOR_CURRENT, "paired"}:
        executors[EXECUTOR_CURRENT] = CurrentLiveExecutor(
            client=client,
            library=library,
            options=LiveAgentOptions(
                action_executor_mode="deterministic_legacy",
                fallback_to_rules=not args.no_rule_fallback,
            ),
        )
    if args.mode in {EXECUTOR_SAFE, "paired"}:
        executors[EXECUTOR_SAFE] = SafeGenerativeExecutor(
            client=client,
            library=library,
            options=LiveAgentOptions(
                action_executor_mode="safe_generative",
                fallback_to_rules=not args.no_rule_fallback,
                action_only_repair_enabled=True,
                state_first_route_adjudication_enabled=True,
            ),
        )
    report = run_multiturn_benchmark(
        dataset,
        library,
        executors=executors,
        repeats=args.repeats,
    )
    if args.output is None:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        target = write_json(args.output, report)
        print(
            json.dumps(
                {
                    "output": str(target),
                    "run_status": report["run_status"],
                    "run_fingerprint": report["run_fingerprint"],
                    "content_sha256": report["content_sha256"],
                    "raw_text_printed": False,
                    "api_key_printed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0 if report["run_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
