"""Teaching-agent compatibility adapter for the domain-neutral harness.

The legacy teaching loop owns the teaching policy and its public receipt
contract.  This module only adapts that policy to the shared harness: model
plans are normalized into provider-neutral decisions, the seven deterministic
teaching tools are registered centrally, and the harness result is projected
back to ``teacher_agent_loop.v1`` with an additional content-free harness
trace.

Keeping the adapter separate lets the live controller migrate to the shared
runtime without changing ``run_teaching_agent_loop`` or duplicating the tool
implementations.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

from .deepseek_client import (
    DeepSeekClient,
    DeepSeekClientError,
    DeepSeekConfigurationError,
)
from .harness import (
    CancellationToken,
    HarnessCancelled,
    HarnessContractError,
    HarnessDeadlineExceeded,
    HarnessLimits,
    HarnessCheckpoint,
    HarnessJournal,
    HarnessModelRequest,
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
    RetryPolicy,
    ToolCall,
    ToolExecutionContext,
    ToolRegistry,
    ToolSpec,
    public_harness_trace,
    run_agent_harness,
)
from .teacher_agent import canonical_sha256
from .teacher_agent_safety import classify_assistant_output_safety
from .teacher_agent_resource_retrieval import (
    MAX_QUERY_EXPANSION_CHARS,
    MAX_QUERY_EXPANSIONS,
    MAX_RETRIEVAL_QUERY_CHARS,
    MAX_RETRIEVAL_RESULTS,
    MAX_RETRIEVAL_TOTAL_CHARS,
    LocalHashingVectorScoreProvider,
    ResourceRetrievalError,
    contextual_query_expansions,
    retrieve_teaching_resources,
)
from .teacher_agent_loop import (
    LOOP_SCHEMA,
    StructuredModel,
    TeachingAgentLoopError,
    TeachingAgentLoopOptions,
    _fallback_action,
    _invoke_model,
    _model_messages,
    _session_view,
    _short,
    _skill_index,
    _tool_result,
    _validate_plan,
)


TEACHER_AGENT_HARNESS_PERMISSION_STUDENT = "teaching.student.read"
TEACHER_AGENT_HARNESS_PERMISSION_HISTORY = "teaching.history.read"
TEACHER_AGENT_HARNESS_PERMISSION_SKILLS = "teaching.skills.read"
TEACHER_AGENT_HARNESS_PERMISSION_ROUTE = "teaching.route.write"
TEACHER_AGENT_HARNESS_PERMISSION_TERMINATION = "teaching.termination.read"
TEACHER_AGENT_HARNESS_PERMISSION_RESOURCES = "teaching.resources.retrieve"
TEACHER_AGENT_HARNESS_PERMISSIONS = frozenset(
    {
        TEACHER_AGENT_HARNESS_PERMISSION_STUDENT,
        TEACHER_AGENT_HARNESS_PERMISSION_HISTORY,
        TEACHER_AGENT_HARNESS_PERMISSION_SKILLS,
        TEACHER_AGENT_HARNESS_PERMISSION_ROUTE,
        TEACHER_AGENT_HARNESS_PERMISSION_TERMINATION,
        TEACHER_AGENT_HARNESS_PERMISSION_RESOURCES,
    }
)

_FOCUS = frozenset({"prerequisite", "conceptual", "procedural", "transfer"})
_OBJECT_OUTPUT_SCHEMA = {"type": "object"}
_EMPTY_INPUT_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class TeachingHarnessDurability:
    """Durability binding inherited by the nested route/tool Harness run."""

    parent_run_id: str
    parent_turn_id: str
    operation_id: str
    run_id: str
    turn_id: str
    journal: HarnessJournal
    checkpoint_sink: Callable[[HarnessCheckpoint], None] | None = None
    retrieval_resources: tuple[Mapping[str, Any], ...] = ()


_TEACHING_HARNESS_DURABILITY: ContextVar[TeachingHarnessDurability | None] = ContextVar(
    "teaching_harness_durability", default=None
)


@contextmanager
def bind_teaching_harness_durability(
    durability: TeachingHarnessDurability,
) -> Iterator[None]:
    """Bind an outer Teach operation to its subordinate route Harness.

    A context variable keeps the public live-policy APIs unchanged while the
    dashboard worker explicitly supplies durable identifiers and persistence.
    The binding is thread-local and is always reset when the domain operation
    returns or raises.
    """

    if not isinstance(durability, TeachingHarnessDurability):
        raise TeachingAgentLoopError("teaching Harness durability is invalid")
    durability_context_handle = _TEACHING_HARNESS_DURABILITY.set(durability)
    try:
        yield
    finally:
        _TEACHING_HARNESS_DURABILITY.reset(durability_context_handle)


def _tool_specs() -> tuple[ToolSpec, ...]:
    """Return the versioned contracts exposed to the planner."""

    common = {
        "version": "1.0.0",
        "timeout_seconds": 2.0,
        "replay_policy": "safe",
        "parallel_safe": False,
        "execution_isolation": "trusted_inline",
        "trusted_inline_reason": (
            "repository-owned bounded deterministic in-memory teaching policy"
        ),
        "retry_policy": RetryPolicy(max_attempts=1),
        "output_schema": _OBJECT_OUTPUT_SCHEMA,
    }
    read_tool = {**common, "risk": "low"}
    route_tool = {**common, "risk": "medium"}
    return (
        ToolSpec(
            name="inspect_student_state",
            description="Read the bounded learner-state projection for this teaching turn.",
            input_schema=_EMPTY_INPUT_SCHEMA,
            permission=TEACHER_AGENT_HARNESS_PERMISSION_STUDENT,
            data_scope="learner_profile",
            **read_tool,
        ),
        ToolSpec(
            name="inspect_recent_history",
            description="Read a bounded suffix of recent teaching turns.",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 8}
                },
                "additionalProperties": False,
            },
            permission=TEACHER_AGENT_HARNESS_PERMISSION_HISTORY,
            data_scope="learner_answer",
            **read_tool,
        ),
        ToolSpec(
            name="search_skills",
            description="Search allowlisted primary teaching Skills by query and signals.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 160},
                    "signals": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 80},
                        "maxItems": 8,
                        "uniqueItems": True,
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 12,
                    },
                },
                "additionalProperties": False,
            },
            permission=TEACHER_AGENT_HARNESS_PERMISSION_SKILLS,
            **read_tool,
        ),
        ToolSpec(
            name="select_skills",
            description="Commit one allowlisted primary Skill and bounded support Skills.",
            input_schema={
                "type": "object",
                "properties": {
                    "primary_skill_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 120,
                    },
                    "supporting_skill_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 120,
                        },
                        "maxItems": 4,
                        "uniqueItems": True,
                    },
                    "reason": {"type": "string", "maxLength": 600},
                },
                "required": ["primary_skill_id"],
                "additionalProperties": False,
            },
            permission=TEACHER_AGENT_HARNESS_PERMISSION_ROUTE,
            **route_tool,
        ),
        ToolSpec(
            name="set_next_focus",
            description="Commit the next bounded mastery dimension for the route.",
            input_schema={
                "type": "object",
                "properties": {
                    "next_focus": {"type": "string", "enum": sorted(_FOCUS)}
                },
                "required": ["next_focus"],
                "additionalProperties": False,
            },
            permission=TEACHER_AGENT_HARNESS_PERMISSION_ROUTE,
            **route_tool,
        ),
        ToolSpec(
            name="evaluate_termination",
            description="Check local mastery thresholds and current evidence before success.",
            input_schema=_EMPTY_INPUT_SCHEMA,
            permission=TEACHER_AGENT_HARNESS_PERMISSION_TERMINATION,
            **read_tool,
        ),
        ToolSpec(
            name="retrieve_resources",
            description=(
                "Retrieve bounded excerpts from teacher-imported resources by "
                "query. Results include source location and content hashes; they "
                "are teacher context and never learner-answer evidence."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_RETRIEVAL_QUERY_CHARS,
                    },
                    "query_expansions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_QUERY_EXPANSION_CHARS,
                        },
                        "maxItems": MAX_QUERY_EXPANSIONS,
                        "uniqueItems": True,
                    },
                    "resource_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^res_[0-9a-f]{20}$",
                        },
                        "maxItems": 6,
                        "uniqueItems": True,
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_RETRIEVAL_RESULTS,
                    },
                    "max_total_chars": {
                        "type": "integer",
                        "minimum": 200,
                        "maximum": MAX_RETRIEVAL_TOTAL_CHARS,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            data_scope="teacher_resource",
            permission=TEACHER_AGENT_HARNESS_PERMISSION_RESOURCES,
            **read_tool,
        ),
    )


def build_teacher_agent_tool_registry(
    context: Mapping[str, Any],
    library: Mapping[str, Any],
    state: dict[str, Any],
    *,
    runtime_meta: dict[str, Any] | None = None,
    teaching_resources: Sequence[Mapping[str, Any]] | None = None,
) -> ToolRegistry:
    """Register the legacy teaching tools without copying their domain logic."""

    if not isinstance(context, Mapping):
        raise TeachingAgentLoopError("teaching context must be an object")
    if not isinstance(library, Mapping):
        raise TeachingAgentLoopError("skill_library must be an object")
    if not isinstance(state, dict):
        raise TeachingAgentLoopError("teaching loop state must be mutable")
    _skill_index(library)
    meta = runtime_meta if runtime_meta is not None else {}
    resource_descriptors = tuple(
        deepcopy(dict(item))
        for item in (teaching_resources or ())
        if isinstance(item, Mapping)
    )
    local_vector_scorer = LocalHashingVectorScoreProvider()
    goal_context = context.get("goal", {})
    retrieval_context_terms = (
        [
            str(goal_context.get("concept", "")),
            str(goal_context.get("objective", "")),
            *[str(item) for item in goal_context.get("knowledge_components", [])],
        ]
        if isinstance(goal_context, Mapping)
        and isinstance(goal_context.get("knowledge_components", []), list)
        else []
    )
    registry = ToolRegistry()

    for spec in _tool_specs():
        name = spec.name

        def handler(
            arguments: Mapping[str, Any],
            _execution_context: ToolExecutionContext,
            *,
            _name: str = name,
        ) -> dict[str, Any]:
            if _name == "retrieve_resources":
                try:
                    query = str(arguments.get("query", ""))
                    supplied_expansions = (
                        [str(item) for item in arguments.get("query_expansions", [])]
                        if isinstance(arguments.get("query_expansions"), list)
                        else []
                    )
                    expansions = list(
                        dict.fromkeys(
                            [
                                *supplied_expansions,
                                *contextual_query_expansions(
                                    query, retrieval_context_terms
                                ),
                            ]
                        )
                    )[:MAX_QUERY_EXPANSIONS]
                    return retrieve_teaching_resources(
                        resource_descriptors,
                        query,
                        resource_ids=(
                            [str(item) for item in arguments.get("resource_ids", [])]
                            if isinstance(arguments.get("resource_ids"), list)
                            else None
                        ),
                        max_results=int(arguments.get("max_results", 4)),
                        max_total_chars=int(arguments.get("max_total_chars", 3_600)),
                        query_expansions=expansions,
                        vector_scorer=local_vector_scorer,
                    )
                except (ResourceRetrievalError, TypeError, ValueError) as exc:
                    raise TeachingAgentLoopError(
                        "resource retrieval request is invalid"
                    ) from exc
            result = _tool_result(_name, arguments, context, library, state)
            if _name == "evaluate_termination":
                meta["termination_checked"] = True
                meta["termination_eligible"] = result.get("eligible") is True
            return result

        registry.register(spec, handler)
    return registry


class _RetryablePlannerBoundaryError(RuntimeError):
    """Retain an error only until the adapter's typed classifier consumes it."""

    def __init__(self, message: str, *, original: BaseException) -> None:
        super().__init__(message)
        self.original = original


class TeachingAgentHarnessModelAdapter:
    """Adapt a legacy callable or ``DeepSeekClient`` to ``HarnessModel``."""

    def __init__(
        self,
        client: StructuredModel | Callable[[Sequence[Mapping[str, str]]], Any],
        *,
        context: Mapping[str, Any],
        library: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> None:
        self.client = client
        self.context = context
        self.library = library
        self.state = state
        self.records: list[dict[str, Any]] = []
        self.last_error = ""

    @property
    def model_spec(self) -> ProviderModelSpec:
        declared = getattr(self.client, "model_spec", None)
        if isinstance(declared, ProviderModelSpec):
            return declared.validated()
        if not isinstance(self.client, DeepSeekClient):
            raise HarnessContractError(
                "legacy structured model has no stable provider identity"
            )
        model = self.client.config.model
        return ProviderModelSpec(
            provider="deepseek",
            model=model,
            capabilities=ProviderCapabilities(
                provider="deepseek",
                model=model,
                structured_output=True,
                native_stream=self.client.native_stream_available,
                native_tools=False,
                vision=False,
                web_search=False,
                cancellation=self.client.native_stream_available,
            ),
            context_window_tokens=64_000,
            maximum_output_tokens=self.client.config.max_tokens,
        ).validated()

    def classify_error(self, error: BaseException) -> ProviderFailure:
        original = (
            error.original
            if isinstance(error, _RetryablePlannerBoundaryError)
            else error
        )
        if isinstance(original, DeepSeekConfigurationError):
            return ProviderFailure(
                kind=ProviderErrorKind.AUTHENTICATION,
                retryable=False,
                safe_code="deepseek_configuration",
            ).validated()
        if isinstance(original, (TeachingAgentLoopError, HarnessContractError)):
            return ProviderFailure(
                kind=ProviderErrorKind.MALFORMED_RESPONSE,
                retryable=False,
                safe_code="teaching_plan_malformed",
            ).validated()
        text = str(original).casefold()
        if "429" in text or "rate" in text:
            kind, retryable, code = (
                ProviderErrorKind.RATE_LIMIT,
                True,
                "deepseek_rate_limit",
            )
        elif "timeout" in text or "timed out" in text or "deadline" in text:
            kind, retryable, code = (
                ProviderErrorKind.TIMEOUT,
                True,
                "deepseek_timeout",
            )
        elif isinstance(original, (TimeoutError, ConnectionError, OSError)):
            kind, retryable, code = (
                ProviderErrorKind.UNAVAILABLE,
                True,
                "provider_unavailable",
            )
        elif isinstance(original, DeepSeekClientError):
            malformed_markers = (
                "malformed",
                "missing",
                "invalid",
                "unsupported",
                "truncated",
            )
            malformed = any(marker in text for marker in malformed_markers)
            kind, retryable, code = (
                (
                    ProviderErrorKind.MALFORMED_RESPONSE,
                    False,
                    "deepseek_malformed_response",
                )
                if malformed
                else (
                    ProviderErrorKind.UNAVAILABLE,
                    True,
                    "deepseek_unavailable",
                )
            )
        else:
            kind, retryable, code = (
                ProviderErrorKind.UNKNOWN,
                False,
                "provider_unknown",
            )
        return ProviderFailure(
            kind=kind,
            retryable=retryable,
            safe_code=code,
        ).validated()

    @staticmethod
    def _previous_results(
        observations: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in observations:
            if not isinstance(item, Mapping) or item.get("kind") != "tool_result":
                continue
            result: dict[str, Any] = {
                "call_id": _short(item.get("call_id"), 80),
                "tool": _short(item.get("tool_name"), 80),
                "ok": item.get("ok") is True,
            }
            if result["ok"]:
                result["result"] = deepcopy(item.get("result"))
            else:
                error = item.get("error", {})
                if isinstance(error, Mapping):
                    code = _short(error.get("code"), 80)
                    message = _short(error.get("message"), 300)
                    result["error"] = f"{code}: {message}".strip(": ")
                else:
                    result["error"] = _short(error, 300)
            results.append(result)
        return results

    def plan(
        self,
        request: HarnessModelRequest,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> HarnessModelResponse:
        cancellation_token.raise_if_cancelled()
        tool_definitions = tuple(
            deepcopy(dict(item)) for item in request.tools if isinstance(item, Mapping)
        )
        messages = _model_messages(
            self.context,
            self.library,
            self._previous_results(request.observations),
            self.state,
            tool_definitions=tool_definitions,
        )
        try:
            native_structured_stream = getattr(self.client, "chat_json_stream", None)
            if callable(native_structured_stream) and bool(
                getattr(self.client, "native_stream_available", True)
            ):
                raw_plan, trace = native_structured_stream(
                    messages,
                    request_kind="teacher_agent_loop",
                    cancellation_token=cancellation_token,
                    deadline_monotonic=deadline_monotonic,
                    require_remote_consent=True,
                    # Harness runtime is the single owner of model retries;
                    # nested client retries cannot observe central-tool effect
                    # fences or checkpoint policy.
                    transport_max_retries=0,
                )
            else:
                raw_plan, trace = _invoke_model(self.client, messages)
            cancellation_token.raise_if_cancelled()
            # The prompt is derived from ``ToolRegistry.definitions``.  The
            # executor still owns authorization and fails closed if a model
            # hallucinates a hidden or unknown tool despite that declaration.
            plan = _validate_plan(
                raw_plan,
                # Validation recognizes every centrally registered teaching
                # tool.  Authorization remains the executor's job so a model
                # hallucinating a hidden tool produces an auditable
                # permission_denied receipt rather than a planner retry.
                allowed_tools=frozenset(spec.name for spec in _tool_specs()),
            )
            if plan.get("kind") == "teaching_action":
                output_safety = classify_assistant_output_safety(
                    str(plan.get("message", ""))
                )
                if output_safety is not None:
                    raise TeachingAgentLoopError(
                        "generated teaching action failed the student-safety output boundary: "
                        + str(output_safety["category"])
                    )
        except (HarnessCancelled, HarnessDeadlineExceeded):
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {_short(exc, 300)}"
            self.last_error = error
            self.records.append({"ok": False, "error": error})
            raise _RetryablePlannerBoundaryError(error, original=exc) from exc

        self.records.append(
            {"ok": True, "plan": deepcopy(plan), "trace": deepcopy(trace)}
        )
        usage = trace.get("usage", {}) if isinstance(trace, Mapping) else {}
        if not isinstance(usage, Mapping):
            usage = {}
        response_id = trace.get("response_id") if isinstance(trace, Mapping) else None
        provider_request_id = (
            _short(response_id, 160)
            if isinstance(response_id, str) and response_id
            else None
        )
        if plan["kind"] == "tool_calls":
            return HarnessModelResponse(
                kind="tool_calls",
                tool_calls=tuple(
                    ToolCall(
                        call_id=call["call_id"],
                        name=call["name"],
                        arguments=call["arguments"],
                    )
                    for call in plan["tool_calls"]
                ),
                usage=deepcopy(dict(usage)),
                provider_request_id=provider_request_id,
            )
        if plan["kind"] == "terminate" and plan.get("outcome") == "handoff":
            return HarnessModelResponse(
                kind="handoff",
                output=deepcopy(plan),
                reason=plan["reason"],
                usage=deepcopy(dict(usage)),
                provider_request_id=provider_request_id,
            )
        return HarnessModelResponse(
            kind="final",
            output=deepcopy(plan),
            usage=deepcopy(dict(usage)),
            provider_request_id=provider_request_id,
        )


def _has_valid_route(state: Mapping[str, Any]) -> bool:
    return bool(
        str(state.get("selected_skill_id") or "").strip()
        and state.get("next_focus") in _FOCUS
    )


def _fallback_projection(
    session: Mapping[str, Any],
    state: Mapping[str, Any],
    reason: str,
    *,
    guard_reason: str = "",
) -> dict[str, Any]:
    return {
        "status": "action_ready",
        "action": _fallback_action(session, state, reason),
        "reason": reason,
        "deterministic_fallback": True,
        "terminal_event": "fallback",
        "guard_reason": guard_reason,
    }


def _project_terminal(
    harness_result: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    state: Mapping[str, Any],
    runtime_meta: Mapping[str, Any],
    options: TeachingAgentLoopOptions,
    model: TeachingAgentHarnessModelAdapter,
) -> dict[str, Any]:
    status = str(harness_result.get("status", "failed"))
    reason = _short(harness_result.get("reason"), 300) or "harness failed"

    if status == "cancelled":
        return {
            "status": "terminated_unable",
            "action": None,
            "reason": reason,
            "deterministic_fallback": False,
            "terminal_event": "",
            "guard_reason": "",
        }
    if status not in {"completed", "handoff"}:
        if reason in {
            "repeated_tool_call",
            "step_budget_exceeded",
        } and _has_valid_route(state):
            route_reason = (
                "repeated tool call; using the last validated route"
                if reason == "repeated_tool_call"
                else "max_steps exceeded; using the last validated route"
            )
            return {
                "status": "route_ready",
                "action": None,
                "reason": route_reason,
                "deterministic_fallback": False,
                "terminal_event": "route_ready",
                "guard_reason": reason,
            }
        fallback_reason = model.last_error or reason
        return _fallback_projection(
            session,
            state,
            fallback_reason,
            guard_reason=reason if reason != "model planning failed" else "",
        )

    plan = harness_result.get("output")
    if not isinstance(plan, Mapping):
        return _fallback_projection(
            session, state, "harness returned no terminal teaching plan"
        )
    kind = plan.get("kind")
    if kind == "route_ready":
        if not _has_valid_route(state):
            return _fallback_projection(
                session,
                state,
                "route emitted before required tools",
                guard_reason="route_ready_requires_selected_skill_and_focus",
            )
        return {
            "status": "route_ready",
            "action": None,
            "reason": _short(plan.get("reason"), 600),
            "deterministic_fallback": False,
            "terminal_event": "route_ready",
            "guard_reason": "",
        }
    if kind == "terminate":
        if plan.get("outcome") == "handoff":
            return {
                "status": "terminated_unable",
                "action": None,
                "reason": _short(plan.get("reason"), 300),
                "deterministic_fallback": False,
                "terminal_event": "",
                "guard_reason": "",
            }
        if runtime_meta.get("termination_eligible") is not True:
            return _fallback_projection(
                session,
                state,
                "success requested before readiness check",
                guard_reason="success_requires_evaluate_termination",
            )
        return {
            "status": "succeeded",
            "action": None,
            "reason": _short(plan.get("reason"), 300),
            "deterministic_fallback": False,
            "terminal_event": "",
            "guard_reason": "",
        }
    if kind != "teaching_action":
        return _fallback_projection(session, state, "terminal plan kind is invalid")

    selected = str(state.get("selected_skill_id") or "")
    if not selected or plan.get("selected_skill_id") != selected:
        return _fallback_projection(
            session,
            state,
            "action emitted before select_skills",
            guard_reason="action_without_runtime_skill_selection",
        )
    if list(plan.get("supporting_skill_ids", [])) != list(
        state.get("supporting_skill_ids", []) or []
    ):
        return _fallback_projection(
            session,
            state,
            "action support Skills changed after selection",
            guard_reason="action_supports_do_not_match_runtime_selection",
        )
    if plan.get("next_focus") != state.get("next_focus"):
        return _fallback_projection(
            session,
            state,
            "action focus changed after selection",
            guard_reason="action_focus_does_not_match_runtime_selection",
        )
    message = plan.get("message")
    if not isinstance(message, str) or len(message) > options.max_action_chars:
        return _fallback_projection(
            session,
            state,
            "action message exceeded the configured budget",
            guard_reason="action_message_budget_exceeded",
        )
    skill = _skill_index(session.get("skill_library", {})).get(selected)
    if (
        not skill
        or skill.get("role") == "support"
        or skill.get("action_type") != plan.get("action_type")
    ):
        return _fallback_projection(
            session,
            state,
            "selected Skill invalid",
            guard_reason="selected_skill_invalid_for_action",
        )
    return {
        "status": "action_ready",
        "action": {
            **deepcopy(dict(plan)),
            "skill": {
                "skill_id": skill["skill_id"],
                "name": skill.get("name", ""),
                "action_type": skill.get("action_type", ""),
            },
        },
        "reason": "model produced a validated teaching action",
        "deterministic_fallback": False,
        "terminal_event": "teaching_action",
        "guard_reason": "",
    }


def _legacy_events(
    harness_result: Mapping[str, Any],
    model: TeachingAgentHarnessModelAdapter,
    projection: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Keep the old receipt useful while the harness trace becomes authoritative."""

    events: list[dict[str, Any]] = []
    record_index = 0
    current_step = 1
    for raw in harness_result.get("events", []) or []:
        if not isinstance(raw, Mapping):
            continue
        event_type = raw.get("type")
        payload = raw.get("payload", {})
        if not isinstance(payload, Mapping):
            payload = {}
        if event_type in {"model.completed", "model.failed"}:
            current_step = int(payload.get("step", current_step) or current_step)
            record = (
                model.records[record_index] if record_index < len(model.records) else {}
            )
            record_index += 1
            if event_type == "model.completed":
                plan = record.get("plan", {}) if isinstance(record, Mapping) else {}
                trace = record.get("trace", {}) if isinstance(record, Mapping) else {}
                events.append(
                    {
                        "type": "model_plan",
                        "step": current_step,
                        "attempt": int(payload.get("attempt", 1) or 1),
                        "kind": _short(
                            plan.get("kind") if isinstance(plan, Mapping) else "", 40
                        ),
                        "trace": deepcopy(dict(trace))
                        if isinstance(trace, Mapping)
                        else {},
                    }
                )
            else:
                error = (
                    record.get("error")
                    if isinstance(record, Mapping)
                    else payload.get("error_type")
                )
                events.append(
                    {
                        "type": "model_error",
                        "step": current_step,
                        "attempt": int(payload.get("attempt", 1) or 1),
                        "error": _short(error, 400),
                    }
                )
        elif event_type == "tool.completed":
            events.append(
                {
                    "type": "tool_result",
                    "step": current_step,
                    "call_id": _short(payload.get("call_id"), 80),
                    "tool": _short(payload.get("tool_name"), 80),
                    "ok": True,
                    "result": deepcopy(payload.get("result")),
                }
            )
        elif event_type in {"tool.failed", "tool.rejected"}:
            error_code = _short(payload.get("error_code"), 100)
            events.append(
                {
                    "type": "tool_error",
                    "step": current_step,
                    "call_id": _short(payload.get("call_id"), 80),
                    "tool": _short(payload.get("tool_name"), 80),
                    "ok": False,
                    "error": f"HarnessToolError: {error_code or 'tool_failed'}",
                }
            )
        elif event_type == "guard.triggered":
            events.append(
                {
                    "type": "guard",
                    "step": current_step,
                    "reason": _short(payload.get("reason_code"), 160),
                    "tool": _short(payload.get("tool_name"), 80),
                }
            )

    guard_reason = _short(projection.get("guard_reason"), 160)
    if guard_reason and not any(
        event.get("type") == "guard" and event.get("reason") == guard_reason
        for event in events
    ):
        events.append({"type": "guard", "step": current_step, "reason": guard_reason})
    terminal_event = projection.get("terminal_event")
    if terminal_event == "fallback":
        events.append(
            {
                "type": "fallback",
                "step": current_step,
                "reason": _short(projection.get("reason"), 300),
            }
        )
    elif terminal_event == "route_ready":
        events.append(
            {
                "type": "route_ready",
                "step": current_step,
                "skill_id": _short(state.get("selected_skill_id"), 120),
                "next_focus": _short(state.get("next_focus"), 40),
            }
        )
    elif terminal_event == "teaching_action":
        message = str((projection.get("action") or {}).get("message", ""))
        events.append(
            {
                "type": "teaching_action",
                "step": current_step,
                "skill_id": _short(state.get("selected_skill_id"), 120),
                "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
            }
        )
    return events


def run_teaching_agent_harness(
    session: Mapping[str, Any],
    client: StructuredModel | Callable[[Sequence[Mapping[str, str]]], Any],
    *,
    options: TeachingAgentLoopOptions | None = None,
    outbound_context: Mapping[str, Any] | None = None,
    allowed_permissions: set[str] | None = None,
    run_id: str | None = None,
    turn_id: str | None = None,
    cancellation_token: CancellationToken | None = None,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    journal: HarnessJournal | None = None,
    checkpoint_sink: Callable[[HarnessCheckpoint], None] | None = None,
    retrieval_resources: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the shared harness and return a legacy-compatible loop receipt.

    ``retrieval_resources`` is the local-only integration point for longer
    projections loaded from ``TeachingResourceIndexStore``.  When omitted, the
    live session's 12k descriptors remain the backward-compatible source.
    """

    options = (options or TeachingAgentLoopOptions()).validated()
    durability = _TEACHING_HARNESS_DURABILITY.get()
    if durability is not None:
        if run_id is None:
            run_id = durability.run_id
        if turn_id is None:
            turn_id = durability.turn_id
        if journal is None:
            journal = durability.journal
        if checkpoint_sink is None:
            checkpoint_sink = durability.checkpoint_sink
        if retrieval_resources is None and durability.retrieval_resources:
            retrieval_resources = durability.retrieval_resources
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
    state: dict[str, Any] = {
        "selected_skill_id": None,
        "supporting_skill_ids": [],
        "next_focus": context["student"].get("next_focus"),
    }
    runtime_meta: dict[str, Any] = {
        "termination_checked": False,
        "termination_eligible": False,
    }
    registry = build_teacher_agent_tool_registry(
        context,
        library,
        state,
        runtime_meta=runtime_meta,
        teaching_resources=(
            retrieval_resources
            if retrieval_resources is not None
            else (
                session.get("teaching_resources", [])
                if isinstance(session.get("teaching_resources", []), list)
                else []
            )
        ),
    )
    model = TeachingAgentHarnessModelAdapter(
        client,
        context=context,
        library=library,
        state=state,
    )
    max_model_calls = options.max_steps * (options.model_retries + 1)
    if max_model_calls > 128:
        raise TeachingAgentLoopError(
            "max_steps and model_retries exceed the harness model-call budget"
        )
    limits = HarnessLimits(
        max_steps=options.max_steps,
        max_model_calls=max_model_calls,
        max_total_tool_calls=options.max_steps * options.max_tool_calls_per_step,
        max_tool_calls_per_step=options.max_tool_calls_per_step,
        max_repeated_tool_calls=options.max_repeated_tool_calls,
        deadline_seconds=90.0,
        max_tool_output_chars=12_000,
        max_event_payload_chars=24_000,
    )
    harness_result = run_agent_harness(
        model,
        registry,
        context,
        run_id=run_id,
        turn_id=turn_id,
        limits=limits,
        retry_policy=RetryPolicy(
            max_attempts=options.model_retries + 1,
            initial_backoff_seconds=0.0,
            backoff_multiplier=1.0,
            max_backoff_seconds=0.0,
        ),
        allowed_permissions=(
            set(TEACHER_AGENT_HARNESS_PERMISSIONS)
            if allowed_permissions is None
            else allowed_permissions
        ),
        cancellation_token=cancellation_token,
        event_sink=event_sink,
        journal=journal,
        checkpoint_sink=checkpoint_sink,
    )
    projection = _project_terminal(
        harness_result,
        session=session,
        state=state,
        runtime_meta=runtime_meta,
        options=options,
        model=model,
    )
    events = _legacy_events(harness_result, model, projection, state)
    receipt = {
        "schema": LOOP_SCHEMA,
        "status": projection["status"],
        "action": deepcopy(projection.get("action")),
        "termination_reason": projection.get("reason", ""),
        "loop_state": deepcopy(state),
        "steps": max((int(event.get("step", 0) or 0) for event in events), default=0),
        "events": events,
        "session_fingerprint": canonical_sha256(dict(session)),
        "deterministic_fallback": bool(projection["deterministic_fallback"]),
        "harness_trace": public_harness_trace(harness_result),
    }
    if durability is not None:
        receipt["harness_operation_link"] = {
            "parent_run_id": durability.parent_run_id,
            "parent_turn_id": durability.parent_turn_id,
            "operation_id": durability.operation_id,
            "inner_run_id": durability.run_id,
            "inner_turn_id": durability.turn_id,
        }
    return receipt


# Keep both natural spellings available while callers migrate from
# ``run_teaching_agent_loop``.
run_teacher_agent_harness = run_teaching_agent_harness


__all__ = [
    "TEACHER_AGENT_HARNESS_PERMISSIONS",
    "TEACHER_AGENT_HARNESS_PERMISSION_HISTORY",
    "TEACHER_AGENT_HARNESS_PERMISSION_ROUTE",
    "TEACHER_AGENT_HARNESS_PERMISSION_RESOURCES",
    "TEACHER_AGENT_HARNESS_PERMISSION_SKILLS",
    "TEACHER_AGENT_HARNESS_PERMISSION_STUDENT",
    "TEACHER_AGENT_HARNESS_PERMISSION_TERMINATION",
    "TeachingAgentHarnessModelAdapter",
    "TeachingHarnessDurability",
    "bind_teaching_harness_durability",
    "build_teacher_agent_tool_registry",
    "run_teacher_agent_harness",
    "run_teaching_agent_harness",
]
