"""DeepSeek provider adapters for the generic harness."""

from __future__ import annotations

from hashlib import sha256
import math
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..deepseek_client import (
    DeepSeekClient,
    DeepSeekClientError,
    DeepSeekConfigurationError,
)
from .cancellation import CancellationToken
from .contracts import (
    HarnessContractError,
    HarnessModelRequest,
    HarnessModelResponse,
)
from .providers import ProviderCapabilities, ProviderStreamEvent
from .provider_registry import (
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
)


MessagesBuilder = Callable[
    [HarnessModelRequest], Sequence[Mapping[str, str]]
]


def _default_messages(request: HarnessModelRequest) -> Sequence[Mapping[str, str]]:
    raw = request.context.get("messages")
    if not isinstance(raw, list):
        raise HarnessContractError(
            "DeepSeek chat harness context.messages must be an array"
        )
    return raw


class DeepSeekChatHarnessModel:
    """Native streaming Chat profile with no fake character playback."""

    def __init__(
        self,
        client: DeepSeekClient,
        *,
        request_kind: str = "agent_harness_chat",
        messages_builder: MessagesBuilder | None = None,
        max_tokens: int | None = None,
        web_search: bool = False,
        web_search_system: str | None = None,
        web_search_max_uses: int = 3,
        delta_flush_interval_seconds: float = 0.04,
        delta_flush_chars: int = 512,
        monotonic: Callable[[], float] | None = None,
        context_window_tokens: int = 64_000,
    ) -> None:
        self.client = client
        self.request_kind = str(request_kind).strip()
        self.messages_builder = messages_builder or _default_messages
        self.max_tokens = max_tokens
        self.web_search = bool(web_search)
        self.web_search_system = str(web_search_system or "").strip()
        self.web_search_max_uses = web_search_max_uses
        self.delta_flush_interval_seconds = delta_flush_interval_seconds
        self.delta_flush_chars = delta_flush_chars
        self._monotonic = monotonic or time.monotonic
        self.context_window_tokens = context_window_tokens
        if not self.request_kind or len(self.request_kind) > 80:
            raise HarnessContractError("DeepSeek harness request_kind is invalid")
        if self.web_search and not self.web_search_system:
            raise HarnessContractError("web-search system prompt is required")
        if (
            isinstance(self.web_search_max_uses, bool)
            or not isinstance(self.web_search_max_uses, int)
            or not 1 <= self.web_search_max_uses <= 5
        ):
            raise HarnessContractError("web-search max uses is invalid")
        if (
            isinstance(self.delta_flush_interval_seconds, bool)
            or not isinstance(self.delta_flush_interval_seconds, (int, float))
            or not math.isfinite(float(self.delta_flush_interval_seconds))
            or not 0.005 <= float(self.delta_flush_interval_seconds) <= 1.0
        ):
            raise HarnessContractError("DeepSeek delta flush interval is invalid")
        if (
            isinstance(self.delta_flush_chars, bool)
            or not isinstance(self.delta_flush_chars, int)
            or not 32 <= self.delta_flush_chars <= 32_000
        ):
            raise HarnessContractError("DeepSeek delta flush size is invalid")
        if (
            isinstance(self.context_window_tokens, bool)
            or not isinstance(self.context_window_tokens, int)
            or not 1_024 <= self.context_window_tokens <= 20_000_000
        ):
            raise HarnessContractError("DeepSeek context window is invalid")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider="deepseek",
            model=self.client.config.model,
            structured_output=False,
            native_stream=True,
            native_tools=self.web_search,
            vision=False,
            web_search=self.web_search,
            cancellation=True,
        )

    @property
    def model_spec(self) -> ProviderModelSpec:
        maximum_output = self.max_tokens or self.client.config.max_tokens
        return ProviderModelSpec(
            provider="deepseek",
            model=self.client.config.model,
            capabilities=self.capabilities,
            context_window_tokens=self.context_window_tokens,
            maximum_output_tokens=maximum_output,
        ).validated()

    def classify_error(self, error: BaseException) -> ProviderFailure:
        """Map vendor/transport failures to a content-free stable taxonomy."""

        if isinstance(error, DeepSeekConfigurationError):
            return ProviderFailure(
                kind=ProviderErrorKind.AUTHENTICATION,
                retryable=False,
                safe_code="deepseek_configuration",
            ).validated()
        text = str(error).casefold()
        if "cancel" in text:
            kind, retryable, code = (
                ProviderErrorKind.CANCELLED,
                False,
                "deepseek_cancelled",
            )
        elif "429" in text or "rate" in text:
            kind, retryable, code = (
                ProviderErrorKind.RATE_LIMIT,
                True,
                "deepseek_rate_limit",
            )
        elif "timeout" in text or "timed out" in text:
            kind, retryable, code = (
                ProviderErrorKind.TIMEOUT,
                True,
                "deepseek_timeout",
            )
        elif "context" in text and ("limit" in text or "large" in text):
            kind, retryable, code = (
                ProviderErrorKind.CONTEXT_OVERFLOW,
                False,
                "deepseek_context_overflow",
            )
        elif isinstance(error, DeepSeekClientError):
            kind, retryable, code = (
                ProviderErrorKind.UNAVAILABLE,
                True,
                "deepseek_unavailable",
            )
        else:
            kind, retryable, code = (
                ProviderErrorKind.UNKNOWN,
                False,
                "provider_unknown",
            )
        return ProviderFailure(
            kind=kind, retryable=retryable, safe_code=code
        ).validated()

    def plan(
        self,
        request: HarnessModelRequest,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> HarnessModelResponse:
        """Blocking compatibility wrapper; the harness prefers plan_stream."""

        response: HarnessModelResponse | None = None
        for item in self.plan_stream(
            request,
            cancellation_token=cancellation_token,
            deadline_monotonic=deadline_monotonic,
        ):
            if isinstance(item, HarnessModelResponse):
                response = item
        if response is None:
            raise HarnessContractError("DeepSeek stream returned no final response")
        return response

    def plan_stream(
        self,
        request: HarnessModelRequest,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> Iterator[ProviderStreamEvent | HarnessModelResponse]:
        if self.web_search:
            matching = [
                item
                for item in request.tools
                if item.get("name") == "web_search"
            ]
            if len(matching) != 1 or not (
                matching[0].get("execution_mode") == "provider_managed"
                and matching[0].get("permission") == "chat.web_search"
                and matching[0].get("data_scope") == "public_web"
                and matching[0].get("requires_user_consent") is True
            ):
                raise HarnessContractError(
                    "web search requires one authorized provider-managed ToolSpec"
                )
        messages = self.messages_builder(request)
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise HarnessContractError("DeepSeek messages builder returned invalid data")
        yield ProviderStreamEvent(
            type="message.start",
            payload={
                "provider": "deepseek",
                "model": self.client.config.model,
            },
        )
        pieces: list[str] = []
        trace: dict[str, Any] = {}
        usage: dict[str, Any] = {}
        pending_text: list[str] = []
        pending_text_chars = 0
        pending_text_since: float | None = None
        pending_reasoning_chars = 0
        pending_reasoning_since: float | None = None

        def take_text(*, force: bool = False) -> ProviderStreamEvent | None:
            nonlocal pending_text, pending_text_chars, pending_text_since
            if not pending_text:
                return None
            now = self._monotonic()
            elapsed = (
                now - pending_text_since
                if pending_text_since is not None
                else 0.0
            )
            if (
                not force
                and pending_text_chars < self.delta_flush_chars
                and elapsed < self.delta_flush_interval_seconds
            ):
                return None
            delta = "".join(pending_text)
            pending_text = []
            pending_text_chars = 0
            pending_text_since = None
            return ProviderStreamEvent(type="message.delta", delta=delta)

        def take_reasoning(*, force: bool = False) -> ProviderStreamEvent | None:
            nonlocal pending_reasoning_chars, pending_reasoning_since
            if pending_reasoning_chars <= 0:
                return None
            now = self._monotonic()
            elapsed = (
                now - pending_reasoning_since
                if pending_reasoning_since is not None
                else 0.0
            )
            if (
                not force
                and pending_reasoning_chars < self.delta_flush_chars
                and elapsed < self.delta_flush_interval_seconds
            ):
                return None
            event = ProviderStreamEvent(
                type="reasoning.delta",
                delta="",
                payload={"chars": pending_reasoning_chars},
                channel="internal",
            )
            pending_reasoning_chars = 0
            pending_reasoning_since = None
            return event

        def take_pending() -> Iterator[ProviderStreamEvent]:
            reasoning_event = take_reasoning(force=True)
            if reasoning_event is not None:
                yield reasoning_event
            text_event = take_text(force=True)
            if text_event is not None:
                yield text_event

        if self.web_search:
            source = self.client.chat_web_stream(
                messages,
                system=self.web_search_system,
                request_kind=self.request_kind,
                cancellation_token=cancellation_token,
                deadline_monotonic=deadline_monotonic,
                max_uses=self.web_search_max_uses,
                require_remote_consent=True,
                # The harness owns typed classification, retry_after, effect
                # fences, cancellation and the run deadline.  Nested transport
                # retries would bypass those policy decisions.
                transport_max_retries=0,
            )
        else:
            source = self.client.chat_text_stream(
                messages,
                request_kind=self.request_kind,
                cancellation_token=cancellation_token,
                deadline_monotonic=deadline_monotonic,
                require_remote_consent=True,
                max_tokens=self.max_tokens,
                transport_max_retries=0,
            )
        final_result: dict[str, Any] = {}

        def source_with_failure_flush() -> Iterator[
            Mapping[str, Any] | ProviderStreamEvent
        ]:
            try:
                yield from source
            except DeepSeekClientError:
                # The client may have delivered vendor deltas smaller than the
                # coalescing threshold before detecting a truncated stream.
                # Publish that already-received model output before surfacing
                # the failure so runtime's no-retry-after-delta fence remains
                # authoritative.
                yield from take_pending()
                raise

        for chunk in source_with_failure_flush():
            if isinstance(chunk, ProviderStreamEvent):
                yield chunk
                continue
            kind = chunk.get("type")
            if kind == "text_delta":
                text = str(chunk.get("text", ""))
                if text:
                    reasoning_event = take_reasoning(force=True)
                    if reasoning_event is not None:
                        yield reasoning_event
                    pieces.append(text)
                    if pending_text_since is None:
                        pending_text_since = self._monotonic()
                    pending_text.append(text)
                    pending_text_chars += len(text)
                    text_event = take_text()
                    if text_event is not None:
                        yield text_event
            elif kind == "reasoning_delta":
                text = str(chunk.get("text", ""))
                if text:
                    text_event = take_text(force=True)
                    if text_event is not None:
                        yield text_event
                    if pending_reasoning_since is None:
                        pending_reasoning_since = self._monotonic()
                    pending_reasoning_chars += len(text)
                    reasoning_event = take_reasoning()
                    if reasoning_event is not None:
                        yield reasoning_event
            elif kind == "usage" and isinstance(chunk.get("usage"), Mapping):
                yield from take_pending()
                usage = dict(chunk["usage"])
                yield ProviderStreamEvent(
                    type="usage.update",
                    payload={
                        key: value
                        for key, value in usage.items()
                        if key in {
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                            "input_tokens",
                            "output_tokens",
                            "prompt_cache_hit_tokens",
                            "prompt_cache_miss_tokens",
                        }
                    },
                    channel="internal",
                )
            elif kind in {
                "tool_started",
                "tool_progress",
                "tool_completed",
                "tool_failed",
            }:
                yield from take_pending()
                phase = kind.removeprefix("tool_")
                call_id = str(
                    chunk.get("call_id", "provider_web_search")
                )[:160]
                payload: dict[str, Any] = {
                    "phase": phase,
                    "call_id": call_id or "provider_web_search",
                    "tool_name": "web_search",
                }
                if phase == "completed":
                    payload["source_count"] = int(
                        chunk.get("source_count", 0) or 0
                    )
                elif phase == "failed":
                    payload["error_code"] = str(
                        chunk.get("error_code", "provider_search_failed")
                    )[:80]
                yield ProviderStreamEvent(
                    type="tool_call.delta",
                    payload=payload,
                    channel="internal",
                )
            elif kind == "completed" and isinstance(chunk.get("trace"), Mapping):
                yield from take_pending()
                trace = dict(chunk["trace"])
                if isinstance(trace.get("usage"), Mapping):
                    usage = dict(trace["usage"])
                if isinstance(chunk.get("result"), Mapping):
                    final_result = dict(chunk["result"])
        yield from take_pending()
        message = "".join(pieces).strip()
        if not message:
            raise HarnessContractError("DeepSeek stream returned no assistant text")
        message_hash = sha256(message.encode("utf-8")).hexdigest()
        yield ProviderStreamEvent(
            type="message.end",
            payload={"message_sha256": message_hash, "chars": len(message)},
        )
        if not final_result:
            final_result = {"message": message}
        else:
            final_result["message"] = message
        yield HarnessModelResponse(
            kind="final",
            output=final_result,
            usage=usage,
            provider_request_id=str(trace.get("response_id", ""))[:160] or None,
        )
