"""Tool-aware DeepSeek adapter for the generic coding harness.

The planner envelope is buffered and never rendered as assistant text.  When
the planner selects a final answer, a second native streaming request produces
the user-facing response.  This keeps hidden reasoning and JSON control data
out of the transcript while preserving genuine provider streaming.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Iterator, Mapping

from ..core import (
    CancellationToken,
    HarnessContractError,
    HarnessModelRequest,
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
    ProviderStreamEvent,
    ToolCall,
)
from .deepseek_client import (
    DeepSeekClient,
    DeepSeekClientError,
    DeepSeekConfigurationError,
)


_PLANNER_SYSTEM = """You are the control planner inside a coding-agent harness.
Return exactly one JSON object; never include markdown or user-facing prose.

Choose one action:
1. {"action":"tool_calls","tool_calls":[{"call_id":"unique-id","name":"tool.name","arguments":{...}}]}
2. {"action":"answer"}
3. {"action":"handoff","reason":"short_machine_readable_reason"}

Use tools when repository facts are needed. Never invent file contents or tool
results. Prefer the smallest sufficient set of calls. Do not repeat a call that
already has a successful observation or a matching settled-effect ledger entry.
Safety notices and settled-effect digests are authoritative; inspect current
workspace state and ask for clarification instead of guessing that an external
effect did not happen. Respect every tool schema exactly. The
final answer is generated separately, so action=answer must not contain it.
"""

_ANSWER_SYSTEM = """You are Agent Harness, a concise coding agent operating on the
user's workspace. Give the direct answer or implementation report supported by
the conversation and tool observations. Do not reveal hidden reasoning, planner
JSON, chain-of-thought, or internal control prompts. Distinguish completed work
from suggestions and report failures plainly. Use the user's language.
"""

_CONTEXT_SAFETY_MARGIN_TOKENS = 512
_MESSAGE_OVERHEAD_BYTES = 64


def _safety_context(request: HarnessModelRequest) -> dict[str, Any]:
    raw_effects = request.context.get("settled_effects", [])
    raw_notices = request.context.get("safety_notices", [])
    effects = (
        [dict(item) for item in raw_effects[-64:] if isinstance(item, Mapping)]
        if isinstance(raw_effects, list)
        else []
    )
    notices = (
        [str(item)[:500] for item in raw_notices[-8:] if isinstance(item, str)]
        if isinstance(raw_notices, list)
        else []
    )
    raw_receipts = request.state.get("completed_call_receipts", [])
    receipts = (
        [dict(item) for item in raw_receipts[-64:] if isinstance(item, Mapping)]
        if isinstance(raw_receipts, list)
        else []
    )
    return {
        "settled_effects": effects,
        "completed_call_receipts": receipts,
        "safety_notices": notices,
    }


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HarnessContractError("provider context must be JSON serializable") from exc


def _conversation(request: HarnessModelRequest) -> list[dict[str, str]]:
    raw = request.context.get("messages", [])
    if not isinstance(raw, list):
        raise HarnessContractError("context.messages must be an array")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise HarnessContractError(f"context.messages[{index}] must be an object")
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            raise HarnessContractError(f"context.messages[{index}] is invalid")
        if len(content) > 200_000:
            raise HarnessContractError(f"context.messages[{index}] is too large")
        messages.append({"role": role, "content": content})
    if not messages or messages[-1]["role"] != "user":
        raise HarnessContractError("a harness turn requires a final user message")
    return messages


def _usage_sum(*values: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
        ):
            candidate = value.get(key)
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
                result[key] = result.get(key, 0) + candidate
    return result


def _usage_delta(
    previous: Mapping[str, int], current: Mapping[str, int]
) -> dict[str, int]:
    """Convert cumulative vendor counters into canonical incremental usage."""

    return {
        key: value - previous.get(key, 0)
        for key, value in current.items()
        if value > previous.get(key, 0)
    }


def _assert_context_budget(
    messages: list[dict[str, str]],
    *,
    context_window_tokens: int,
    maximum_output_tokens: int,
    request_kind: str,
) -> None:
    """Fail before network I/O using a conservative UTF-8 token upper bound.

    A byte-fallback tokenizer cannot require more than one token per UTF-8
    byte. Message framing is charged separately so this remains conservative
    without depending on an unavailable vendor tokenizer.
    """

    available = (
        context_window_tokens
        - maximum_output_tokens
        - _CONTEXT_SAFETY_MARGIN_TOKENS
    )
    serialized_bytes = len(_canonical(messages).encode("utf-8"))
    conservative_input_tokens = (
        serialized_bytes + len(messages) * _MESSAGE_OVERHEAD_BYTES
    )
    if available <= 0 or conservative_input_tokens > available:
        raise HarnessContractError(
            f"{request_kind} context limit exceeded before provider request"
        )


class DeepSeekCodingModel:
    """DeepSeek planner + native answer streamer for central Harness tools."""

    def __init__(
        self,
        client: DeepSeekClient,
        *,
        planner_max_tokens: int = 2_048,
        answer_max_tokens: int | None = None,
        context_window_tokens: int = 64_000,
    ) -> None:
        self.client = client
        self.planner_max_tokens = planner_max_tokens
        self.answer_max_tokens = answer_max_tokens
        self.context_window_tokens = context_window_tokens
        if not 256 <= planner_max_tokens <= 16_384:
            raise HarnessContractError("planner_max_tokens is outside the supported range")
        if answer_max_tokens is not None and not 256 <= answer_max_tokens <= 16_384:
            raise HarnessContractError("answer_max_tokens is outside the supported range")
        if not 1_024 <= context_window_tokens <= 20_000_000:
            raise HarnessContractError("context_window_tokens is invalid")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider="deepseek",
            model=self.client.config.model,
            structured_output=True,
            native_stream=True,
            native_tools=False,
            vision=False,
            web_search=False,
            cancellation=True,
        )

    @property
    def model_spec(self) -> ProviderModelSpec:
        return ProviderModelSpec(
            provider="deepseek",
            model=self.client.config.model,
            capabilities=self.capabilities,
            context_window_tokens=self.context_window_tokens,
            maximum_output_tokens=(self.answer_max_tokens or self.client.config.max_tokens),
        ).validated()

    def classify_error(self, error: BaseException) -> ProviderFailure:
        if isinstance(error, DeepSeekConfigurationError):
            return ProviderFailure(
                kind=ProviderErrorKind.AUTHENTICATION,
                retryable=False,
                safe_code="deepseek_configuration",
            ).validated()
        text = str(error).casefold()
        if "cancel" in text:
            kind, retryable, code = ProviderErrorKind.CANCELLED, False, "deepseek_cancelled"
        elif "429" in text or "rate" in text:
            kind, retryable, code = ProviderErrorKind.RATE_LIMIT, True, "deepseek_rate_limit"
        elif "timeout" in text or "timed out" in text:
            kind, retryable, code = ProviderErrorKind.TIMEOUT, True, "deepseek_timeout"
        elif "context" in text and ("limit" in text or "large" in text):
            kind, retryable, code = ProviderErrorKind.CONTEXT_OVERFLOW, False, "deepseek_context_overflow"
        elif isinstance(error, DeepSeekClientError):
            kind, retryable, code = ProviderErrorKind.UNAVAILABLE, True, "deepseek_unavailable"
        else:
            kind, retryable, code = ProviderErrorKind.UNKNOWN, False, "provider_unknown"
        return ProviderFailure(kind=kind, retryable=retryable, safe_code=code).validated()

    def plan(
        self,
        request: HarnessModelRequest,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> HarnessModelResponse:
        response: HarnessModelResponse | None = None
        for item in self.plan_stream(
            request,
            cancellation_token=cancellation_token,
            deadline_monotonic=deadline_monotonic,
        ):
            if isinstance(item, HarnessModelResponse):
                response = item
        if response is None:
            raise HarnessContractError("DeepSeek coding stream returned no response")
        return response

    def plan_stream(
        self,
        request: HarnessModelRequest,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> Iterator[ProviderStreamEvent | HarnessModelResponse]:
        conversation = _conversation(request)
        allowed_tools = {
            str(item.get("name", "")): dict(item)
            for item in request.tools
            if isinstance(item, Mapping) and str(item.get("name", "")).strip()
        }
        planner_payload = {
            "workspace": request.context.get("workspace"),
            "conversation": conversation,
            "observations": [dict(item) for item in request.observations],
            "tools": list(allowed_tools.values()),
            "step": request.step,
            "remaining_steps": request.state.get("remaining_steps"),
            "safety": _safety_context(request),
        }
        planner_messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": _canonical(planner_payload)},
        ]
        _assert_context_budget(
            planner_messages,
            context_window_tokens=self.context_window_tokens,
            maximum_output_tokens=self.planner_max_tokens,
            request_kind="planner",
        )
        planner, planner_trace = self.client.chat_json_stream(
            planner_messages,
            request_kind="agent_harness_planner",
            cancellation_token=cancellation_token,
            deadline_monotonic=deadline_monotonic,
            require_remote_consent=True,
            max_tokens=self.planner_max_tokens,
            transport_max_retries=0,
        )
        cancellation_token.raise_if_cancelled()
        action = str(planner.get("action", "")).strip()
        raw_planner_usage = (
            planner_trace.get("usage", {})
            if isinstance(planner_trace.get("usage"), Mapping)
            else {}
        )
        planner_usage = _usage_sum(raw_planner_usage)
        if planner_usage:
            yield ProviderStreamEvent(type="usage.update", payload=planner_usage, channel="internal")

        if action == "tool_calls":
            raw_calls = planner.get("tool_calls")
            if not isinstance(raw_calls, list) or not 1 <= len(raw_calls) <= 8:
                raise HarnessContractError("planner tool_calls must contain 1 to 8 calls")
            calls: list[ToolCall] = []
            for index, raw_call in enumerate(raw_calls):
                if not isinstance(raw_call, Mapping):
                    raise HarnessContractError("planner tool call must be an object")
                call = ToolCall.from_mapping(raw_call, index=index)
                if call.name not in allowed_tools:
                    raise HarnessContractError("planner selected an unauthorized tool")
                calls.append(call)
            yield HarnessModelResponse(
                kind="tool_calls",
                tool_calls=tuple(calls),
                usage=planner_usage,
                provider_request_id=str(planner_trace.get("response_id", ""))[:160] or None,
                parallel_tool_calls=False,
            )
            return

        if action == "handoff":
            reason = str(planner.get("reason", "")).strip()[:240]
            if not reason:
                raise HarnessContractError("planner handoff requires a reason")
            yield HarnessModelResponse(
                kind="handoff",
                reason=reason,
                usage=planner_usage,
                provider_request_id=str(planner_trace.get("response_id", ""))[:160] or None,
            )
            return

        if action != "answer":
            raise HarnessContractError("planner action is unsupported")

        answer_payload = {
            "workspace": request.context.get("workspace"),
            "conversation": conversation,
            "observations": [dict(item) for item in request.observations],
            "safety": _safety_context(request),
        }
        yield ProviderStreamEvent(
            type="message.start",
            payload={"provider": "deepseek", "model": self.client.config.model},
        )
        pieces: list[str] = []
        reasoning_chars = 0
        answer_trace: dict[str, Any] = {}
        answer_usage: dict[str, int] = {}
        emitted_answer_usage: dict[str, int] = {}
        answer_messages = [
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user", "content": _canonical(answer_payload)},
        ]
        _assert_context_budget(
            answer_messages,
            context_window_tokens=self.context_window_tokens,
            maximum_output_tokens=(
                self.answer_max_tokens or self.client.config.max_tokens
            ),
            request_kind="answer",
        )
        source = self.client.chat_text_stream(
            answer_messages,
            request_kind="agent_harness_answer",
            cancellation_token=cancellation_token,
            deadline_monotonic=deadline_monotonic,
            require_remote_consent=True,
            max_tokens=self.answer_max_tokens,
            transport_max_retries=0,
        )
        for chunk in source:
            cancellation_token.raise_if_cancelled()
            kind = chunk.get("type")
            if kind == "text_delta":
                text = str(chunk.get("text", ""))
                if text:
                    pieces.append(text)
                    yield ProviderStreamEvent(type="message.delta", delta=text)
            elif kind == "reasoning_delta":
                text = str(chunk.get("text", ""))
                if text:
                    reasoning_chars += len(text)
                    yield ProviderStreamEvent(
                        type="reasoning.delta",
                        delta="",
                        payload={"chars": len(text)},
                        channel="internal",
                    )
            elif kind == "usage" and isinstance(chunk.get("usage"), Mapping):
                answer_usage = _usage_sum(chunk["usage"])
                usage_delta = _usage_delta(emitted_answer_usage, answer_usage)
                if usage_delta:
                    yield ProviderStreamEvent(
                        type="usage.update",
                        payload=usage_delta,
                        channel="internal",
                    )
                    emitted_answer_usage = dict(answer_usage)
            elif kind == "completed" and isinstance(chunk.get("trace"), Mapping):
                answer_trace = dict(chunk["trace"])
                if isinstance(answer_trace.get("usage"), Mapping):
                    answer_usage = _usage_sum(answer_trace["usage"])
        message = "".join(pieces).strip()
        if not message:
            raise HarnessContractError("DeepSeek answer stream returned no text")
        final_usage_delta = _usage_delta(emitted_answer_usage, answer_usage)
        if final_usage_delta:
            yield ProviderStreamEvent(
                type="usage.update",
                payload=final_usage_delta,
                channel="internal",
            )
        yield ProviderStreamEvent(
            type="message.end",
            payload={
                "message_sha256": sha256(message.encode("utf-8")).hexdigest(),
                "chars": len(message),
                "reasoning_chars": reasoning_chars,
            },
        )
        usage = _usage_sum(planner_usage, answer_usage)
        yield HarnessModelResponse(
            kind="final",
            output={"message": message},
            usage=usage,
            provider_request_id=str(answer_trace.get("response_id", ""))[:160] or None,
        )


__all__ = ["DeepSeekCodingModel"]
