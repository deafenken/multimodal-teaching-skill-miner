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
from typing import Any, Callable, Mapping, Sequence
from urllib import error, parse, request


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
ALLOWED_MODELS = frozenset({DEFAULT_MODEL})
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class DeepSeekClientError(RuntimeError):
    """Raised when a safe structured DeepSeek request cannot be completed."""


class DeepSeekConfigurationError(DeepSeekClientError):
    """Raised when configuration or explicit remote-data consent is missing."""


Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


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
        raise DeepSeekConfigurationError("DeepSeek base URL must not contain credentials or query data")
    return raw


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
        raise DeepSeekConfigurationError("DeepSeek API key file cannot be read") from exc
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
            raise DeepSeekConfigurationError("DeepSeek timeout must be in [1, 300] seconds")
        if not 0 <= self.max_retries <= 4:
            raise DeepSeekConfigurationError("DeepSeek max_retries must be in [0, 4]")
        if not 256 <= self.max_tokens <= 16_384:
            raise DeepSeekConfigurationError("DeepSeek max_tokens is outside the supported range")
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
        api_key: str | None = None,
    ) -> None:
        self.config = config.validated()
        self._transport = transport or _default_transport
        self._api_key = api_key

    def public_status(self) -> dict[str, Any]:
        return self.config.public_status()

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        request_kind: str,
        require_remote_consent: bool = True,
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
        request_body = {
            "model": self.config.model,
            "messages": safe_messages,
            "response_format": {"type": "json_object"},
            "thinking": {
                "type": "enabled" if self.config.thinking_enabled else "disabled"
            },
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if not self.config.thinking_enabled:
            request_body["temperature"] = self.config.temperature
        request_bytes = _canonical_json(request_body)
        request_sha = sha256(request_bytes).hexdigest()
        key = _read_api_key(api_key=self._api_key, api_key_file=self.config.api_key_file)
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
                content = choice["message"]["content"]
                parsed_content = json.loads(content)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise DeepSeekClientError("DeepSeek returned malformed structured output") from exc
            if not isinstance(parsed_content, dict):
                raise DeepSeekClientError("DeepSeek structured output must be one JSON object")
            usage = envelope.get("usage") if isinstance(envelope, dict) else None
            trace = {
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
                "response_id": str(envelope.get("id", ""))[:160],
                "usage": usage if isinstance(usage, dict) else {},
                "credential_logged": False,
            }
            return parsed_content, trace
        raise DeepSeekClientError(
            f"DeepSeek request failed with {last_error} (last status {last_status})"
        )
