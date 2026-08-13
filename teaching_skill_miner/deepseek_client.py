"""Minimal, auditable DeepSeek Chat Completions client.

The client deliberately has no third-party dependency.  Credentials are read
only at request time, never included in exceptions, request fingerprints, or
public status payloads.  Remote transmission is fail-closed: callers must
explicitly opt in before learner text can leave the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence
from urllib import error, parse, request


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
ALLOWED_MODELS = frozenset({DEFAULT_MODEL})
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_SSE_EVENT_BYTES = 1024 * 1024
_MAX_WEB_SOURCE_NODES = 4096
_WEB_SEARCH_ERROR_CODES = frozenset(
    {
        "too_many_requests",
        "invalid_tool_input",
        "max_uses_exceeded",
        "query_too_long",
        "request_too_large",
        "unavailable",
    }
)


class DeepSeekClientError(RuntimeError):
    """Raised when a safe structured DeepSeek request cannot be completed."""


class DeepSeekConfigurationError(DeepSeekClientError):
    """Raised when configuration or explicit remote-data consent is missing."""


Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]
ProbeTransport = Callable[[str, Mapping[str, str], float], tuple[int, bytes]]


class CancellationLike(Protocol):
    @property
    def cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None: ...

    def add_callback(self, callback: Callable[[], None]) -> Callable[[], None]: ...

    def wait(self, seconds: float) -> bool: ...


StreamTransport = Callable[
    [str, Mapping[str, str], bytes, float, CancellationLike],
    tuple[int, Iterable[bytes]],
]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _safe_base_url(value: str) -> str:
    raw = value.strip().rstrip("/")
    parsed = parse.urlsplit(raw)
    if parsed.scheme != "https" or not parsed.hostname:
        raise DeepSeekConfigurationError("DeepSeek base URL must be an HTTPS origin")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DeepSeekConfigurationError(
            "DeepSeek base URL must not contain credentials or query data"
        )
    return raw


def _safe_web_source(raw_source: Mapping[str, Any]) -> dict[str, str] | None:
    raw_url = raw_source.get("url")
    raw_title = raw_source.get("title")
    if not isinstance(raw_url, str) or not isinstance(raw_title, str):
        return None
    clean_input_url = raw_url.strip()
    if (
        not clean_input_url
        or len(clean_input_url) > 8192
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in clean_input_url
        )
    ):
        return None
    parsed_url = parse.urlsplit(clean_input_url)
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or parsed_url.username
        or parsed_url.password
    ):
        return None
    tracking_keys = {
        "dclid",
        "fbclid",
        "from",
        "from_source",
        "gclid",
        "gspk",
        "gsxid",
        "internal_id",
        "isappinstalled",
        "mc_cid",
        "mc_eid",
        "msclkid",
        "p_source",
        "p_type",
        "partner",
        "ps_partner_key",
        "ps_xid",
        "pscd",
        "ref",
        "referrer",
        "roistat_visit",
        "r",
        "source",
        "src",
        "srch_tag",
        "wxwork_userid",
    }
    clean_query = parse.urlencode(
        [
            (key_name, value)
            for key_name, value in parse.parse_qsl(
                parsed_url.query, keep_blank_values=True
            )
            if not key_name.casefold().startswith("utm_")
            and key_name.casefold() not in tracking_keys
        ],
        doseq=True,
    )
    clean_url = parse.urlunsplit(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            clean_query,
            "",
        )
    )
    title = " ".join(raw_title.split())[:240]
    if not title:
        return None
    return {"title": title, "url": clean_url[:2048]}


def _append_safe_web_source(
    sources: list[dict[str, str]], raw_source: Mapping[str, Any]
) -> None:
    source = _safe_web_source(raw_source)
    if (
        source is not None
        and all(existing["url"] != source["url"] for existing in sources)
        and len(sources) < 12
    ):
        sources.append(source)


def _collect_safe_web_sources(sources: list[dict[str, str]], value: Any) -> None:
    """Collect only bounded title/URL metadata from a provider envelope.

    Search bodies, encrypted continuation data, snippets, and model citations
    are intentionally never copied into the public result.  The node budget is
    independent of the transport byte budget so an adversarially nested JSON
    value cannot monopolize the parser.
    """

    pending = [value]
    inspected = 0
    while pending and inspected < _MAX_WEB_SOURCE_NODES:
        current = pending.pop()
        inspected += 1
        if isinstance(current, Mapping):
            if current.get("type") in {
                "web_search_result",
                "web_search_result_location",
            }:
                _append_safe_web_source(sources, current)
            citation = current.get("citation")
            if isinstance(citation, Mapping):
                _append_safe_web_source(sources, citation)
                pending.append(citation)
            citations = current.get("citations")
            if isinstance(citations, list):
                for item in citations:
                    if isinstance(item, Mapping):
                        _append_safe_web_source(sources, item)
                pending.extend(reversed(citations))
            content = current.get("content")
            if isinstance(content, (Mapping, list)):
                pending.append(content)
        elif isinstance(current, list):
            pending.extend(reversed(current))


def _iter_sse_events(
    chunks: Iterable[bytes], *, stream_name: str
) -> Iterator[tuple[str | None, str]]:
    """Parse a byte stream into SSE event/data pairs.

    Transport chunks are not assumed to align with UTF-8 code points, lines,
    or event boundaries.  Comment-only keep-alives are ignored, multiple data
    fields are joined according to the SSE specification, and both the whole
    stream and each event have explicit memory bounds.
    """

    buffer = bytearray()
    event_name: str | None = None
    data_lines: list[str] = []
    data_bytes = 0
    received = 0

    def process_line(raw_line: bytes) -> tuple[str | None, str] | None:
        nonlocal event_name, data_lines, data_bytes
        if raw_line.endswith(b"\r"):
            raw_line = raw_line[:-1]
        if not raw_line:
            if not data_lines:
                event_name = None
                data_bytes = 0
                return None
            event = (event_name, "\n".join(data_lines))
            event_name = None
            data_lines = []
            data_bytes = 0
            return event
        if raw_line.startswith(b":"):
            return None
        field, separator, raw_value = raw_line.partition(b":")
        if separator and raw_value.startswith(b" "):
            raw_value = raw_value[1:]
        try:
            field_name = field.decode("utf-8")
            value = raw_value.decode("utf-8") if separator else ""
        except UnicodeDecodeError as exc:
            raise DeepSeekClientError(f"{stream_name} contained invalid UTF-8") from exc
        if field_name == "event":
            if len(value) > 100:
                raise DeepSeekClientError(
                    f"{stream_name} event name exceeded the safety limit"
                )
            event_name = value
        elif field_name == "data":
            data_bytes += len(raw_value)
            if data_bytes > _MAX_SSE_EVENT_BYTES:
                raise DeepSeekClientError(
                    f"{stream_name} event exceeded the safety limit"
                )
            data_lines.append(value)
        return None

    for chunk in chunks:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise DeepSeekClientError(f"{stream_name} yielded a non-byte chunk")
        raw_chunk = bytes(chunk)
        received += len(raw_chunk)
        if received > _MAX_RESPONSE_BYTES:
            raise DeepSeekClientError(f"{stream_name} exceeded the safety limit")
        buffer.extend(raw_chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                if len(buffer) > _MAX_SSE_EVENT_BYTES:
                    raise DeepSeekClientError(
                        f"{stream_name} line exceeded the safety limit"
                    )
                break
            raw_line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            event = process_line(raw_line)
            if event is not None:
                yield event
    if buffer:
        event = process_line(bytes(buffer))
        if event is not None:
            yield event
    if data_lines:
        yield event_name, "\n".join(data_lines)


def _close_stream(chunks: Iterable[bytes]) -> None:
    close = getattr(chunks, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # Closing is best-effort after a terminal protocol/error decision;
            # cancellation remains monotonic and no response content is logged.
            pass


def _read_api_key(*, api_key: str | None, api_key_file: str | Path | None) -> str:
    direct = (api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()
    if direct:
        if any(character.isspace() for character in direct):
            raise DeepSeekConfigurationError("DeepSeek API key is malformed")
        return direct
    configured_file = api_key_file or os.getenv("TSM_DEEPSEEK_API_KEY_FILE")
    if not configured_file:
        raise DeepSeekConfigurationError(
            "DeepSeek API key is not configured; set DEEPSEEK_API_KEY or "
            "TSM_DEEPSEEK_API_KEY_FILE"
        )
    path = Path(configured_file).expanduser()
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise DeepSeekConfigurationError(
            "DeepSeek API key file cannot be read"
        ) from exc
    if not key or any(character.isspace() for character in key):
        raise DeepSeekConfigurationError("DeepSeek API key file is empty or malformed")
    return key


def _default_transport(
    url: str, headers: Mapping[str, str], payload: bytes, timeout: float
) -> tuple[int, bytes]:
    req = request.Request(url, data=payload, headers=dict(headers), method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(body) > _MAX_RESPONSE_BYTES:
                raise DeepSeekClientError("DeepSeek response exceeded the safety limit")
            return int(response.status), body
    except error.HTTPError as exc:
        body = exc.read(16_384)
        return int(exc.code), body


def _default_probe_transport(
    url: str, headers: Mapping[str, str], timeout: float
) -> tuple[int, bytes]:
    """Perform the provider's content-free authenticated model-list probe."""

    req = request.Request(url, headers=dict(headers), method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read(64 * 1024 + 1)
            if len(body) > 64 * 1024:
                raise DeepSeekClientError(
                    "DeepSeek readiness response exceeded the safety limit"
                )
            return int(response.status), body
    except error.HTTPError as exc:
        try:
            body = exc.read(16_384)
        finally:
            exc.close()
        return int(exc.code), body


def _default_stream_transport(
    url: str,
    headers: Mapping[str, str],
    payload: bytes,
    timeout: float,
    cancellation_token: CancellationLike,
) -> tuple[int, Iterable[bytes]]:
    """Open one SSE response and make response.close() cancellation-driven."""

    cancellation_token.raise_if_cancelled()
    req = request.Request(url, data=payload, headers=dict(headers), method="POST")
    try:
        response = request.urlopen(req, timeout=timeout)
    except error.HTTPError as exc:
        try:
            body = exc.read(16_384)
        finally:
            exc.close()
        return int(exc.code), (body,)
    unsubscribe = cancellation_token.add_callback(response.close)

    def lines() -> Iterator[bytes]:
        received = 0
        try:
            for line in response:
                cancellation_token.raise_if_cancelled()
                received += len(line)
                if received > _MAX_RESPONSE_BYTES:
                    raise DeepSeekClientError(
                        "DeepSeek stream exceeded the safety limit"
                    )
                yield bytes(line)
        finally:
            unsubscribe()
            response.close()

    return int(response.status), lines()


@dataclass(frozen=True, slots=True)
class DeepSeekConfig:
    """Server-side DeepSeek settings without credential material."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 60.0
    max_retries: int = 2
    max_tokens: int = 1800
    thinking_enabled: bool = False
    temperature: float = 0.0
    allow_remote_student_data: bool = False
    api_key_file: str | Path | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        api_key_file: str | Path | None = None,
        allow_remote_student_data: bool | None = None,
        model: str | None = None,
    ) -> "DeepSeekConfig":
        timeout = float(os.getenv("TSM_DEEPSEEK_TIMEOUT_SECONDS", "60"))
        retries = int(os.getenv("TSM_DEEPSEEK_MAX_RETRIES", "2"))
        temperature = float(os.getenv("TSM_DEEPSEEK_TEMPERATURE", "0"))
        return cls(
            base_url=os.getenv("TSM_DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            model=model or os.getenv("TSM_DEEPSEEK_MODEL", DEFAULT_MODEL),
            timeout_seconds=timeout,
            max_retries=retries,
            thinking_enabled=_truthy(os.getenv("TSM_DEEPSEEK_THINKING")),
            temperature=temperature,
            allow_remote_student_data=(
                _truthy(os.getenv("TSM_ALLOW_REMOTE_STUDENT_DATA"))
                if allow_remote_student_data is None
                else allow_remote_student_data
            ),
            api_key_file=api_key_file,
        )

    def validated(self) -> "DeepSeekConfig":
        _safe_base_url(self.base_url)
        if self.model not in ALLOWED_MODELS:
            raise DeepSeekConfigurationError(
                f"DeepSeek model must be one of {sorted(ALLOWED_MODELS)}"
            )
        if not 1 <= self.timeout_seconds <= 300:
            raise DeepSeekConfigurationError(
                "DeepSeek timeout must be in [1, 300] seconds"
            )
        if not 0 <= self.max_retries <= 4:
            raise DeepSeekConfigurationError("DeepSeek max_retries must be in [0, 4]")
        if not 256 <= self.max_tokens <= 16_384:
            raise DeepSeekConfigurationError(
                "DeepSeek max_tokens is outside the supported range"
            )
        if not 0 <= self.temperature <= 2:
            raise DeepSeekConfigurationError("DeepSeek temperature must be in [0, 2]")
        return self

    def public_status(self) -> dict[str, Any]:
        return {
            "provider": "deepseek",
            "model": self.model,
            "base_origin": _safe_base_url(self.base_url),
            "configured": bool(
                os.getenv("DEEPSEEK_API_KEY")
                or self.api_key_file
                or os.getenv("TSM_DEEPSEEK_API_KEY_FILE")
            ),
            "thinking_mode": "enabled" if self.thinking_enabled else "disabled",
            "temperature": None if self.thinking_enabled else self.temperature,
            "remote_student_data_opt_in": self.allow_remote_student_data,
            "api_key_exposed": False,
        }


class DeepSeekClient:
    """OpenAI-compatible `/chat/completions` JSON client for DeepSeek."""

    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        probe_transport: ProbeTransport | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config.validated()
        self._transport = transport or _default_transport
        self._stream_transport = stream_transport or _default_stream_transport
        self._probe_transport = probe_transport or _default_probe_transport
        # A caller that injects only the blocking transport normally expects
        # every request to remain inside that test/proxy boundary. Falling back
        # to the default network stream would bypass the injection.
        self._native_stream_available = (
            stream_transport is not None or transport is None
        )
        self._api_key = api_key

    @property
    def native_stream_available(self) -> bool:
        return self._native_stream_available

    def public_status(self) -> dict[str, Any]:
        status = self.config.public_status()
        status["web_search_supported"] = True
        status["web_search_transport"] = "anthropic_server_tool"
        return status

    def probe_model_availability(
        self, *, timeout_seconds: float = 5.0
    ) -> dict[str, Any]:
        """Validate the credential, network path, and configured model.

        The official ``GET /models`` endpoint receives no learner content and
        creates no model generation. Response bodies and model inventories are
        never returned or logged; callers receive only fixed aggregate booleans.
        """

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0.1 <= float(timeout_seconds) <= 30.0
        ):
            raise DeepSeekConfigurationError(
                "DeepSeek readiness timeout is outside the supported range"
            )
        key = _read_api_key(
            api_key=self._api_key, api_key_file=self.config.api_key_file
        )
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "TeachingSkillMiner-Readiness/1.0",
        }
        url = f"{_safe_base_url(self.config.base_url)}/models"
        try:
            status, body = self._probe_transport(url, headers, float(timeout_seconds))
        except (TimeoutError, socket.timeout, error.URLError, OSError) as exc:
            raise DeepSeekClientError("DeepSeek readiness probe failed") from exc
        if status != 200:
            raise DeepSeekClientError("DeepSeek readiness probe failed")
        try:
            envelope = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeepSeekClientError("DeepSeek readiness probe failed") from exc
        if (
            not isinstance(envelope, Mapping)
            or envelope.get("object") != "list"
            or not isinstance(envelope.get("data"), list)
            or not 1 <= len(envelope["data"]) <= 1024
        ):
            raise DeepSeekClientError("DeepSeek readiness probe failed")
        models: set[str] = set()
        for raw in envelope["data"]:
            if (
                not isinstance(raw, Mapping)
                or raw.get("object") != "model"
                or not isinstance(raw.get("id"), str)
                or not 1 <= len(raw["id"]) <= 160
                or not isinstance(raw.get("owned_by"), str)
                or not 1 <= len(raw["owned_by"]) <= 160
            ):
                raise DeepSeekClientError("DeepSeek readiness probe failed")
            models.add(str(raw["id"]))
        if self.config.model not in models:
            raise DeepSeekClientError("DeepSeek readiness probe failed")
        return {
            "schema": "teaching_skill_miner.deepseek_readiness.v1",
            "credential_validated": True,
            "provider_network_validated": True,
            "configured_model_available": True,
            "learner_content_sent": False,
            "generation_created": False,
        }

    def chat_web(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        system: str,
        request_kind: str,
        max_uses: int = 3,
        require_remote_consent: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return a text answer grounded by DeepSeek's server-side web search."""

        if require_remote_consent and not self.config.allow_remote_student_data:
            raise DeepSeekConfigurationError(
                "remote learner-text processing is disabled; pass explicit consent"
            )
        safe_kind = str(request_kind).strip()
        if not safe_kind or len(safe_kind) > 80:
            raise DeepSeekConfigurationError("request_kind is invalid")
        safe_system = str(system).strip()
        if not safe_system:
            raise DeepSeekConfigurationError("system prompt is required")
        if (
            isinstance(max_uses, bool)
            or not isinstance(max_uses, int)
            or not 1 <= max_uses <= 5
        ):
            raise DeepSeekConfigurationError("web search max_uses must be in [1, 5]")
        safe_messages: list[dict[str, str]] = []
        for index, item in enumerate(messages):
            role = str(item.get("role", ""))
            content = str(item.get("content", ""))
            if role not in {"user", "assistant"} or not content.strip():
                raise DeepSeekConfigurationError(f"messages[{index}] is invalid")
            safe_messages.append({"role": role, "content": content.strip()})
        if not safe_messages:
            raise DeepSeekConfigurationError("at least one message is required")
        request_body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": safe_system,
            "messages": safe_messages,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": max_uses,
                }
            ],
            "stream": False,
        }
        if self.config.thinking_enabled:
            request_body["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        else:
            request_body["temperature"] = self.config.temperature
        request_bytes = _canonical_json(request_body)
        request_sha = sha256(request_bytes).hexdigest()
        key = _read_api_key(
            api_key=self._api_key, api_key_file=self.config.api_key_file
        )
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "TeachingSkillMiner/2.0",
        }
        url = f"{_safe_base_url(self.config.base_url)}/anthropic/v1/messages"
        started = time.monotonic()
        last_status: int | None = None
        last_error = "request failed"
        for attempt in range(self.config.max_retries + 1):
            try:
                status, body = self._transport(
                    url, headers, request_bytes, self.config.timeout_seconds
                )
            except (TimeoutError, socket.timeout, error.URLError, OSError) as exc:
                last_error = type(exc).__name__
                if attempt >= self.config.max_retries:
                    raise DeepSeekClientError(
                        f"DeepSeek web search failed after {attempt + 1} attempt(s): {last_error}"
                    ) from exc
                time.sleep(0.2 * (attempt + 1))
                continue
            last_status = status
            if status != 200:
                last_error = f"HTTP {status}"
                if status in _RETRYABLE_STATUS and attempt < self.config.max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise DeepSeekClientError(
                    f"DeepSeek web search failed with HTTP {status}"
                )
            try:
                envelope = json.loads(body)
                content_blocks = envelope["content"]
            except (
                KeyError,
                TypeError,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ) as exc:
                raise DeepSeekClientError(
                    "DeepSeek returned malformed web search output"
                ) from exc
            if not isinstance(content_blocks, list):
                raise DeepSeekClientError(
                    "DeepSeek web search content must be an array"
                )
            text_parts: list[str] = []
            sources: list[dict[str, str]] = []
            searched = False

            for block in content_blocks:
                if not isinstance(block, Mapping):
                    continue
                block_type = block.get("type")
                if block_type == "text" and isinstance(block.get("text"), str):
                    text = str(block["text"]).strip()
                    if text:
                        text_parts.append(text)
                elif (
                    block_type == "server_tool_use"
                    and block.get("name") == "web_search"
                ):
                    searched = True
                elif block_type == "web_search_tool_result":
                    searched = True
                _collect_safe_web_sources(sources, block)
            message = "\n\n".join(text_parts).strip()
            if not message:
                raise DeepSeekClientError(
                    "DeepSeek web search response is missing text"
                )
            usage = envelope.get("usage") if isinstance(envelope, dict) else None
            trace = {
                "provider": "deepseek",
                "model": self.config.model,
                "thinking_mode": (
                    "enabled" if self.config.thinking_enabled else "disabled"
                ),
                "temperature": (
                    None if self.config.thinking_enabled else self.config.temperature
                ),
                "request_kind": safe_kind,
                "request_sha256": request_sha,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "attempt_count": attempt + 1,
                "http_status": status,
                "response_id": str(envelope.get("id", ""))[:160],
                "usage": usage if isinstance(usage, dict) else {},
                "credential_logged": False,
                "web_search_used": searched,
                "web_search_source_count": len(sources),
            }
            return {
                "message": message,
                "sources": sources,
                "web_search_used": searched,
            }, trace
        raise DeepSeekClientError(
            f"DeepSeek web search failed with {last_error} (last status {last_status})"
        )

    def chat_web_stream(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        system: str,
        request_kind: str,
        cancellation_token: CancellationLike,
        deadline_monotonic: float,
        max_uses: int = 3,
        require_remote_consent: bool = True,
        transport_max_retries: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield assistant deltas and bounded server-side web-search lifecycle."""

        if require_remote_consent and not self.config.allow_remote_student_data:
            raise DeepSeekConfigurationError(
                "remote learner-text processing is disabled; pass explicit consent"
            )
        safe_kind = str(request_kind).strip()
        safe_system = str(system).strip()
        if not safe_kind or len(safe_kind) > 80:
            raise DeepSeekConfigurationError("request_kind is invalid")
        if not safe_system:
            raise DeepSeekConfigurationError("system prompt is required")
        if (
            isinstance(max_uses, bool)
            or not isinstance(max_uses, int)
            or not 1 <= max_uses <= 5
        ):
            raise DeepSeekConfigurationError("web search max_uses must be in [1, 5]")
        safe_messages: list[dict[str, str]] = []
        for index, item in enumerate(messages):
            role = str(item.get("role", ""))
            content = str(item.get("content", ""))
            if role not in {"user", "assistant"} or not content.strip():
                raise DeepSeekConfigurationError(f"messages[{index}] is invalid")
            safe_messages.append({"role": role, "content": content.strip()})
        if not safe_messages:
            raise DeepSeekConfigurationError("at least one message is required")
        request_body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": safe_system,
            "messages": safe_messages,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": max_uses,
                }
            ],
            "stream": True,
        }
        if self.config.thinking_enabled:
            request_body["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        else:
            request_body["temperature"] = self.config.temperature
        request_bytes = _canonical_json(request_body)
        request_sha = sha256(request_bytes).hexdigest()
        key = _read_api_key(
            api_key=self._api_key, api_key_file=self.config.api_key_file
        )
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "TeachingSkillMiner/2.0",
        }
        url = f"{_safe_base_url(self.config.base_url)}/anthropic/v1/messages"
        started = time.monotonic()
        if transport_max_retries is None:
            effective_retries = self.config.max_retries
        elif (
            isinstance(transport_max_retries, bool)
            or not isinstance(transport_max_retries, int)
            or not 0 <= transport_max_retries <= 4
        ):
            raise DeepSeekConfigurationError("transport_max_retries must be in [0, 4]")
        else:
            effective_retries = transport_max_retries
        emitted_effect = False
        for attempt in range(effective_retries + 1):
            cancellation_token.raise_if_cancelled()
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise DeepSeekClientError(
                    "DeepSeek web search stream deadline exceeded"
                )
            timeout = max(0.05, min(self.config.timeout_seconds, remaining))
            try:
                status, lines = self._stream_transport(
                    url, headers, request_bytes, timeout, cancellation_token
                )
                if status != 200:
                    _close_stream(lines)
                    if (
                        status in _RETRYABLE_STATUS
                        and attempt < effective_retries
                        and not emitted_effect
                    ):
                        delay = min(0.2 * (attempt + 1), max(0.0, remaining))
                        if delay <= 0 or cancellation_token.wait(delay):
                            cancellation_token.raise_if_cancelled()
                        continue
                    raise DeepSeekClientError(
                        f"DeepSeek web search stream failed with HTTP {status}"
                    )
                response_id = ""
                usage: dict[str, Any] = {}
                sources: list[dict[str, str]] = []
                text_parts: list[str] = []
                searched = False
                block_types: dict[int, str] = {}
                block_call_ids: dict[int, str] = {}
                block_source_counts: dict[int, int] = {}
                block_errors: dict[int, str] = {}
                progressed_calls: set[str] = set()
                provider_call_ids: dict[str, str] = {}
                unsettled_calls: set[str] = set()
                settled_calls: set[str] = set()
                message_started = False
                stop_reason: str | None = None
                done = False

                def provider_key(raw_id: Any) -> str:
                    if not isinstance(raw_id, str) or not raw_id or len(raw_id) > 500:
                        raise DeepSeekClientError(
                            "DeepSeek web search tool identifier is malformed"
                        )
                    return sha256(raw_id.encode("utf-8")).hexdigest()

                def public_call_id(key: str) -> str:
                    return f"provider_web_search_{key[:20]}"

                try:
                    for event_name, data in _iter_sse_events(
                        lines,
                        stream_name="DeepSeek web search stream",
                    ):
                        cancellation_token.raise_if_cancelled()
                        if time.monotonic() >= deadline_monotonic:
                            raise DeepSeekClientError(
                                "DeepSeek web search stream deadline exceeded"
                            )
                        if data == "[DONE]":
                            raise DeepSeekClientError(
                                "DeepSeek web search stream used an invalid terminal marker"
                            )
                        try:
                            envelope = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise DeepSeekClientError(
                                "DeepSeek web search stream returned malformed JSON"
                            ) from exc
                        if not isinstance(envelope, Mapping):
                            raise DeepSeekClientError(
                                "DeepSeek web search stream chunk must be an object"
                            )
                        event_type = envelope.get("type")
                        if not isinstance(event_type, str) or not event_type:
                            raise DeepSeekClientError(
                                "DeepSeek web search stream event type is missing"
                            )
                        if event_name and event_name != event_type:
                            raise DeepSeekClientError(
                                "DeepSeek web search stream event name did not match its payload"
                            )
                        if event_type in {"ping", "keep_alive"}:
                            continue
                        if event_type == "error":
                            raise DeepSeekClientError(
                                "DeepSeek web search stream returned an error event"
                            )
                        if event_type == "message_start":
                            if message_started or block_types:
                                raise DeepSeekClientError(
                                    "DeepSeek web search stream repeated message_start"
                                )
                            message = envelope.get("message", {})
                            if not isinstance(message, Mapping):
                                raise DeepSeekClientError(
                                    "DeepSeek web search stream message_start is malformed"
                                )
                            initial_stop = message.get("stop_reason")
                            if initial_stop is not None:
                                raise DeepSeekClientError(
                                    "DeepSeek web search stream started terminal"
                                )
                            message_started = True
                            response_id = str(message.get("id", ""))[:160]
                            raw_usage = message.get("usage")
                            if isinstance(raw_usage, Mapping):
                                usage.update(raw_usage)
                            continue
                        if not message_started:
                            raise DeepSeekClientError(
                                "DeepSeek web search stream content preceded message_start"
                            )
                        if event_type == "content_block_start":
                            block = envelope.get("content_block", {})
                            raw_index = envelope.get("index")
                            if (
                                not isinstance(block, Mapping)
                                or isinstance(raw_index, bool)
                                or not isinstance(raw_index, int)
                                or raw_index < 0
                                or raw_index in block_types
                            ):
                                raise DeepSeekClientError(
                                    "DeepSeek web search stream content block is malformed"
                                )
                            block_type = block.get("type")
                            if not isinstance(block_type, str) or not block_type:
                                raise DeepSeekClientError(
                                    "DeepSeek web search stream content block type is missing"
                                )
                            block_types[raw_index] = block_type
                            if block_type != "web_search_tool_result":
                                _collect_safe_web_sources(sources, block)
                            if block_type == "text":
                                initial_text = block.get("text", "")
                                if not isinstance(initial_text, str):
                                    raise DeepSeekClientError(
                                        "DeepSeek web search text block is malformed"
                                    )
                                if initial_text:
                                    emitted_effect = True
                                    text_parts.append(initial_text)
                                    yield {
                                        "type": "text_delta",
                                        "text": initial_text,
                                    }
                            elif (
                                block_type == "server_tool_use"
                                and block.get("name") == "web_search"
                            ):
                                searched = True
                                emitted_effect = True
                                key = provider_key(block.get("id"))
                                call_id = public_call_id(key)
                                if key in provider_call_ids or call_id in settled_calls:
                                    raise DeepSeekClientError(
                                        "DeepSeek web search stream repeated a tool call"
                                    )
                                provider_call_ids[key] = call_id
                                block_call_ids[raw_index] = call_id
                                unsettled_calls.add(call_id)
                                yield {
                                    "type": "tool_started",
                                    "tool_name": "web_search",
                                    "call_id": call_id,
                                }
                            elif block_type == "server_tool_use":
                                raise DeepSeekClientError(
                                    "DeepSeek web search stream invoked an unsupported server tool"
                                )
                            elif block_type == "web_search_tool_result":
                                searched = True
                                emitted_effect = True
                                key = provider_key(block.get("tool_use_id"))
                                call_id = provider_call_ids.get(key)
                                if call_id is None or call_id not in unsettled_calls:
                                    raise DeepSeekClientError(
                                        "DeepSeek web search result did not match a tool call"
                                    )
                                block_call_ids[raw_index] = call_id
                                result_sources: list[dict[str, str]] = []
                                _collect_safe_web_sources(result_sources, block)
                                for source in result_sources:
                                    _append_safe_web_source(sources, source)
                                block_source_counts[raw_index] = len(result_sources)
                                raw_content = block.get("content")
                                if (
                                    isinstance(raw_content, Mapping)
                                    and raw_content.get("type")
                                    == "web_search_tool_result_error"
                                ):
                                    raw_code = raw_content.get("error_code")
                                    block_errors[raw_index] = (
                                        raw_code
                                        if raw_code in _WEB_SEARCH_ERROR_CODES
                                        else "provider_search_failed"
                                    )
                            continue
                        if event_type == "content_block_delta":
                            raw_index = envelope.get("index")
                            delta = envelope.get("delta", {})
                            if (
                                isinstance(raw_index, bool)
                                or not isinstance(raw_index, int)
                                or raw_index not in block_types
                                or not isinstance(delta, Mapping)
                            ):
                                raise DeepSeekClientError(
                                    "DeepSeek web search stream content delta is malformed"
                                )
                            delta_type = delta.get("type")
                            block_type = block_types[raw_index]
                            if delta_type == "text_delta":
                                text = delta.get("text")
                                if block_type != "text" or not isinstance(text, str):
                                    raise DeepSeekClientError(
                                        "DeepSeek web search text delta is malformed"
                                    )
                                if text:
                                    emitted_effect = True
                                    text_parts.append(text)
                                    yield {"type": "text_delta", "text": text}
                            elif (
                                delta_type == "input_json_delta"
                                and block_type == "server_tool_use"
                            ):
                                call_id = block_call_ids.get(raw_index)
                                if call_id and call_id not in progressed_calls:
                                    progressed_calls.add(call_id)
                                    yield {
                                        "type": "tool_progress",
                                        "tool_name": "web_search",
                                        "call_id": call_id,
                                    }
                            elif delta_type in {
                                "citations_delta",
                                "citation_delta",
                            }:
                                _collect_safe_web_sources(sources, delta)
                            # Thinking/signature deltas are intentionally not
                            # exposed to the assistant or public wire channel.
                            continue
                        if event_type == "content_block_stop":
                            raw_index = envelope.get("index")
                            if (
                                isinstance(raw_index, bool)
                                or not isinstance(raw_index, int)
                                or raw_index not in block_types
                            ):
                                raise DeepSeekClientError(
                                    "DeepSeek web search stream stopped an unknown block"
                                )
                            stopped_type = block_types.pop(raw_index)
                            call_id = block_call_ids.pop(raw_index, None)
                            if stopped_type == "web_search_tool_result":
                                if not call_id or call_id not in unsettled_calls:
                                    raise DeepSeekClientError(
                                        "DeepSeek web search tool settlement is malformed"
                                    )
                                unsettled_calls.remove(call_id)
                                settled_calls.add(call_id)
                                error_code = block_errors.pop(raw_index, None)
                                if error_code:
                                    yield {
                                        "type": "tool_failed",
                                        "tool_name": "web_search",
                                        "call_id": call_id,
                                        "error_code": error_code,
                                    }
                                else:
                                    yield {
                                        "type": "tool_completed",
                                        "tool_name": "web_search",
                                        "call_id": call_id,
                                        "source_count": block_source_counts.pop(
                                            raw_index, 0
                                        ),
                                    }
                            continue
                        if event_type == "message_delta":
                            delta = envelope.get("delta", {})
                            if not isinstance(delta, Mapping):
                                raise DeepSeekClientError(
                                    "DeepSeek web search message delta is malformed"
                                )
                            raw_stop = delta.get("stop_reason")
                            if raw_stop is not None:
                                if not isinstance(raw_stop, str) or not raw_stop:
                                    raise DeepSeekClientError(
                                        "DeepSeek web search stop reason is malformed"
                                    )
                                if stop_reason is not None and stop_reason != raw_stop:
                                    raise DeepSeekClientError(
                                        "DeepSeek web search stop reason changed"
                                    )
                                stop_reason = raw_stop
                            raw_usage = envelope.get("usage")
                            if isinstance(raw_usage, Mapping):
                                usage.update(raw_usage)
                                yield {"type": "usage", "usage": dict(usage)}
                            continue
                        if event_type == "message_stop":
                            if block_types or unsettled_calls:
                                raise DeepSeekClientError(
                                    "DeepSeek web search stream stopped with unfinished tools"
                                )
                            if stop_reason is None:
                                raise DeepSeekClientError(
                                    "DeepSeek web search stream omitted stop_reason"
                                )
                            done = True
                            break
                        # Anthropic may add event types over time. Unknown
                        # typed events are ignored, but cannot satisfy any
                        # required lifecycle or terminal condition.
                finally:
                    _close_stream(lines)
                cancellation_token.raise_if_cancelled()
                if not done:
                    raise DeepSeekClientError(
                        "DeepSeek web search stream ended before the terminal marker"
                    )
                if stop_reason == "max_tokens":
                    raise DeepSeekClientError(
                        "DeepSeek web search stream was truncated at the token limit"
                    )
                if stop_reason == "pause_turn":
                    raise DeepSeekClientError(
                        "DeepSeek web search stream requires an unsupported pause_turn continuation"
                    )
                if stop_reason not in {"end_turn", "stop_sequence", "refusal"}:
                    raise DeepSeekClientError(
                        "DeepSeek web search stream returned an unsupported stop_reason"
                    )
                message = "".join(text_parts).strip()
                if not message:
                    raise DeepSeekClientError(
                        "DeepSeek web search response is missing text"
                    )
                yield {
                    "type": "completed",
                    "result": {
                        "message": message,
                        "sources": sources,
                        "web_search_used": searched,
                    },
                    "trace": {
                        "provider": "deepseek",
                        "model": self.config.model,
                        "thinking_mode": (
                            "enabled" if self.config.thinking_enabled else "disabled"
                        ),
                        "temperature": (
                            None
                            if self.config.thinking_enabled
                            else self.config.temperature
                        ),
                        "request_kind": safe_kind,
                        "request_sha256": request_sha,
                        "latency_ms": round((time.monotonic() - started) * 1000, 1),
                        "attempt_count": attempt + 1,
                        "http_status": status,
                        "response_id": response_id,
                        "usage": usage,
                        "credential_logged": False,
                        "native_stream": True,
                        "transport_cancellation_supported": True,
                        "web_search_used": searched,
                        "web_search_source_count": len(sources),
                        "stop_reason": stop_reason,
                    },
                }
                return
            except DeepSeekClientError:
                raise
            except Exception as exc:
                cancellation_token.raise_if_cancelled()
                if emitted_effect or attempt >= effective_retries:
                    raise DeepSeekClientError(
                        "DeepSeek web search stream transport failed"
                    ) from exc
                delay = 0.2 * (attempt + 1)
                if time.monotonic() + delay >= deadline_monotonic:
                    raise DeepSeekClientError(
                        "DeepSeek web search stream retry would exceed the deadline"
                    ) from exc
                if cancellation_token.wait(delay):
                    cancellation_token.raise_if_cancelled()
        raise DeepSeekClientError("DeepSeek web search stream ended without a result")

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_kind: str,
        require_remote_consent: bool = True,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return parsed JSON content plus a secret-free request trace."""

        if require_remote_consent and not self.config.allow_remote_student_data:
            raise DeepSeekConfigurationError(
                "remote learner-text processing is disabled; pass explicit consent"
            )
        safe_kind = str(request_kind).strip()
        if not safe_kind or len(safe_kind) > 80:
            raise DeepSeekConfigurationError("request_kind is invalid")
        safe_messages: list[dict[str, str]] = []
        for index, item in enumerate(messages):
            role = str(item.get("role", ""))
            content = str(item.get("content", ""))
            if role not in {"system", "user", "assistant"} or not content.strip():
                raise DeepSeekConfigurationError(f"messages[{index}] is invalid")
            safe_messages.append({"role": role, "content": content})
        if not safe_messages:
            raise DeepSeekConfigurationError("at least one message is required")
        effective_max_tokens = (
            self.config.max_tokens if max_tokens is None else max_tokens
        )
        if (
            isinstance(effective_max_tokens, bool)
            or not isinstance(effective_max_tokens, int)
            or not 256 <= effective_max_tokens <= 16_384
        ):
            raise DeepSeekConfigurationError(
                "DeepSeek request max_tokens is outside the supported range"
            )
        request_body = {
            "model": self.config.model,
            "messages": safe_messages,
            "response_format": {"type": "json_object"},
            "thinking": {
                "type": "enabled" if self.config.thinking_enabled else "disabled"
            },
            "max_tokens": effective_max_tokens,
            "stream": False,
        }
        if not self.config.thinking_enabled:
            request_body["temperature"] = self.config.temperature
        request_bytes = _canonical_json(request_body)
        request_sha = sha256(request_bytes).hexdigest()
        key = _read_api_key(
            api_key=self._api_key, api_key_file=self.config.api_key_file
        )
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "TeachingSkillMiner/2.0",
        }
        url = f"{_safe_base_url(self.config.base_url)}/chat/completions"
        started = time.monotonic()
        last_status: int | None = None
        last_error = "request failed"
        for attempt in range(self.config.max_retries + 1):
            try:
                status, body = self._transport(
                    url, headers, request_bytes, self.config.timeout_seconds
                )
            except (TimeoutError, socket.timeout, error.URLError, OSError) as exc:
                last_error = type(exc).__name__
                if attempt >= self.config.max_retries:
                    raise DeepSeekClientError(
                        f"DeepSeek request failed after {attempt + 1} attempt(s): {last_error}"
                    ) from exc
                time.sleep(0.2 * (attempt + 1))
                continue
            last_status = status
            if status != 200:
                last_error = f"HTTP {status}"
                if status in _RETRYABLE_STATUS and attempt < self.config.max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise DeepSeekClientError(f"DeepSeek request failed with HTTP {status}")
            try:
                envelope = json.loads(body)
                choice = envelope["choices"][0]
                if (
                    isinstance(choice, Mapping)
                    and choice.get("finish_reason") == "length"
                ):
                    raise DeepSeekClientError(
                        "DeepSeek structured output was truncated at the token limit"
                    )
                content = choice["message"]["content"]
                parsed_content = json.loads(content)
            except DeepSeekClientError:
                raise
            except (
                KeyError,
                IndexError,
                TypeError,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ) as exc:
                raise DeepSeekClientError(
                    "DeepSeek returned malformed structured output"
                ) from exc
            if not isinstance(parsed_content, dict):
                raise DeepSeekClientError(
                    "DeepSeek structured output must be one JSON object"
                )
            usage = envelope.get("usage") if isinstance(envelope, dict) else None
            trace = {
                "provider": "deepseek",
                "model": self.config.model,
                "thinking_mode": (
                    "enabled" if self.config.thinking_enabled else "disabled"
                ),
                "temperature": (
                    None if self.config.thinking_enabled else self.config.temperature
                ),
                "request_kind": safe_kind,
                "request_sha256": request_sha,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "attempt_count": attempt + 1,
                "http_status": status,
                "response_id": str(envelope.get("id", ""))[:160],
                "usage": usage if isinstance(usage, dict) else {},
                "credential_logged": False,
            }
            return parsed_content, trace
        raise DeepSeekClientError(
            f"DeepSeek request failed with {last_error} (last status {last_status})"
        )

    def chat_text_stream(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_kind: str,
        cancellation_token: CancellationLike,
        deadline_monotonic: float,
        require_remote_consent: bool = True,
        max_tokens: int | None = None,
        response_format_json: bool = False,
        transport_max_retries: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield canonical text/reasoning/usage chunks from DeepSeek SSE.

        The method never retries after emitting content, preventing duplicated
        prefixes. Cancellation closes the active HTTP response through the
        token callback registered by the default stream transport.
        """

        if require_remote_consent and not self.config.allow_remote_student_data:
            raise DeepSeekConfigurationError(
                "remote learner-text processing is disabled; pass explicit consent"
            )
        safe_kind = str(request_kind).strip()
        if not safe_kind or len(safe_kind) > 80:
            raise DeepSeekConfigurationError("request_kind is invalid")
        safe_messages: list[dict[str, str]] = []
        for index, item in enumerate(messages):
            role = str(item.get("role", ""))
            content = str(item.get("content", ""))
            if role not in {"system", "user", "assistant"} or not content.strip():
                raise DeepSeekConfigurationError(f"messages[{index}] is invalid")
            safe_messages.append({"role": role, "content": content})
        if not safe_messages:
            raise DeepSeekConfigurationError("at least one message is required")
        effective_max_tokens = (
            self.config.max_tokens if max_tokens is None else max_tokens
        )
        if (
            isinstance(effective_max_tokens, bool)
            or not isinstance(effective_max_tokens, int)
            or not 256 <= effective_max_tokens <= 16_384
        ):
            raise DeepSeekConfigurationError(
                "DeepSeek request max_tokens is outside the supported range"
            )
        request_body: dict[str, Any] = {
            "model": self.config.model,
            "messages": safe_messages,
            "thinking": {
                "type": "enabled" if self.config.thinking_enabled else "disabled"
            },
            "max_tokens": effective_max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if response_format_json:
            request_body["response_format"] = {"type": "json_object"}
        if not self.config.thinking_enabled:
            request_body["temperature"] = self.config.temperature
        request_bytes = _canonical_json(request_body)
        request_sha = sha256(request_bytes).hexdigest()
        key = _read_api_key(
            api_key=self._api_key, api_key_file=self.config.api_key_file
        )
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "TeachingSkillMiner/2.0",
        }
        url = f"{_safe_base_url(self.config.base_url)}/chat/completions"
        if transport_max_retries is None:
            effective_retries = self.config.max_retries
        elif (
            isinstance(transport_max_retries, bool)
            or not isinstance(transport_max_retries, int)
            or not 0 <= transport_max_retries <= 4
        ):
            raise DeepSeekConfigurationError("transport_max_retries must be in [0, 4]")
        else:
            effective_retries = transport_max_retries
        started = time.monotonic()
        emitted_content = False
        for attempt in range(effective_retries + 1):
            cancellation_token.raise_if_cancelled()
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise DeepSeekClientError("DeepSeek stream deadline exceeded")
            timeout = max(0.05, min(self.config.timeout_seconds, remaining))
            try:
                status, lines = self._stream_transport(
                    url, headers, request_bytes, timeout, cancellation_token
                )
                if status != 200:
                    if (
                        status in _RETRYABLE_STATUS
                        and attempt < effective_retries
                        and not emitted_content
                    ):
                        delay = min(0.2 * (attempt + 1), max(0.0, remaining))
                        if delay <= 0 or cancellation_token.wait(delay):
                            cancellation_token.raise_if_cancelled()
                        continue
                    raise DeepSeekClientError(
                        f"DeepSeek stream failed with HTTP {status}"
                    )
                response_id = ""
                usage: dict[str, Any] = {}
                finish_reason: str | None = None
                done = False
                for raw_line in lines:
                    cancellation_token.raise_if_cancelled()
                    if time.monotonic() >= deadline_monotonic:
                        raise DeepSeekClientError("DeepSeek stream deadline exceeded")
                    try:
                        line = raw_line.decode("utf-8").strip()
                    except UnicodeDecodeError as exc:
                        raise DeepSeekClientError(
                            "DeepSeek stream contained invalid UTF-8"
                        ) from exc
                    if not line or line.startswith(":") or line.startswith("event:"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        envelope = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise DeepSeekClientError(
                            "DeepSeek stream returned malformed JSON"
                        ) from exc
                    if not isinstance(envelope, Mapping):
                        raise DeepSeekClientError(
                            "DeepSeek stream chunk must be an object"
                        )
                    response_id = str(envelope.get("id", response_id))[:160]
                    raw_usage = envelope.get("usage")
                    if isinstance(raw_usage, Mapping):
                        usage = dict(raw_usage)
                        yield {"type": "usage", "usage": usage}
                    choices = envelope.get("choices", [])
                    if not isinstance(choices, list):
                        raise DeepSeekClientError(
                            "DeepSeek stream choices must be an array"
                        )
                    for choice in choices:
                        if not isinstance(choice, Mapping):
                            continue
                        raw_finish = choice.get("finish_reason")
                        if isinstance(raw_finish, str) and raw_finish:
                            finish_reason = raw_finish
                        delta = choice.get("delta", {})
                        if not isinstance(delta, Mapping):
                            continue
                        reasoning = delta.get("reasoning_content")
                        if isinstance(reasoning, str) and reasoning:
                            yield {"type": "reasoning_delta", "text": reasoning}
                        text = delta.get("content")
                        if isinstance(text, str) and text:
                            emitted_content = True
                            yield {"type": "text_delta", "text": text}
                cancellation_token.raise_if_cancelled()
                if not done:
                    raise DeepSeekClientError(
                        "DeepSeek stream ended before the terminal marker"
                    )
                if finish_reason == "length":
                    raise DeepSeekClientError(
                        "DeepSeek stream was truncated at the token limit"
                    )
                yield {
                    "type": "completed",
                    "trace": {
                        "provider": "deepseek",
                        "model": self.config.model,
                        "thinking_mode": (
                            "enabled" if self.config.thinking_enabled else "disabled"
                        ),
                        "temperature": (
                            None
                            if self.config.thinking_enabled
                            else self.config.temperature
                        ),
                        "request_kind": safe_kind,
                        "request_sha256": request_sha,
                        "latency_ms": round((time.monotonic() - started) * 1000, 1),
                        "attempt_count": attempt + 1,
                        "http_status": status,
                        "response_id": response_id,
                        "usage": usage,
                        "credential_logged": False,
                        "native_stream": True,
                        "transport_cancellation_supported": True,
                    },
                }
                return
            except DeepSeekClientError:
                raise
            except Exception as exc:
                cancellation_token.raise_if_cancelled()
                if emitted_content or attempt >= effective_retries:
                    raise DeepSeekClientError(
                        "DeepSeek stream transport failed"
                    ) from exc
                delay = 0.2 * (attempt + 1)
                if time.monotonic() + delay >= deadline_monotonic:
                    raise DeepSeekClientError(
                        "DeepSeek stream retry would exceed the deadline"
                    ) from exc
                if cancellation_token.wait(delay):
                    cancellation_token.raise_if_cancelled()
        raise DeepSeekClientError("DeepSeek stream ended without a result")

    def chat_json_stream(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_kind: str,
        cancellation_token: CancellationLike,
        deadline_monotonic: float,
        require_remote_consent: bool = True,
        max_tokens: int | None = None,
        transport_max_retries: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return structured JSON through the cancellable native SSE transport.

        Structured tokens are deliberately buffered here rather than exposed as
        assistant deltas: they contain the planner envelope, not student-facing
        prose.  Callers may still publish content-free progress/tool events while
        this method gives cancellation ownership to the active HTTP response.
        """

        pieces: list[str] = []
        trace: dict[str, Any] = {}
        for chunk in self.chat_text_stream(
            messages,
            request_kind=request_kind,
            cancellation_token=cancellation_token,
            deadline_monotonic=deadline_monotonic,
            require_remote_consent=require_remote_consent,
            max_tokens=max_tokens,
            response_format_json=True,
            transport_max_retries=transport_max_retries,
        ):
            kind = chunk.get("type")
            if kind == "text_delta":
                text = chunk.get("text")
                if isinstance(text, str) and text:
                    pieces.append(text)
            elif kind == "completed" and isinstance(chunk.get("trace"), Mapping):
                trace = dict(chunk["trace"])
        cancellation_token.raise_if_cancelled()
        try:
            result = json.loads("".join(pieces))
        except json.JSONDecodeError as exc:
            raise DeepSeekClientError(
                "DeepSeek returned malformed streamed structured output"
            ) from exc
        if not isinstance(result, dict):
            raise DeepSeekClientError(
                "DeepSeek streamed structured output must be one JSON object"
            )
        return result, trace
