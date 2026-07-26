from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from .models import validate_skill


def _extract_output_text(payload: dict[str, Any]) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"])
    parts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(str(content["text"]))
    if parts:
        return "\n".join(parts)
    choices = payload.get("choices", [])
    if choices:
        return str(choices[0].get("message", {}).get("content", ""))
    raise RuntimeError("API response did not contain text output")


def _parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("model output must be one JSON object")
    return value


def refine_skill_with_api(
    transcript: dict[str, Any], baseline_skill: dict[str, Any]
) -> dict[str, Any]:
    base_url = os.getenv("TSM_API_BASE", "https://api.openai.com/v1").rstrip("/")
    api_key = os.getenv("TSM_API_KEY", "")
    model = os.getenv("TSM_MODEL", "gpt-4.1-mini")
    timeout = int(os.getenv("TSM_TIMEOUT_SECONDS", "90"))
    if not api_key:
        raise RuntimeError("TSM_API_KEY is required when --backend api is used")
    parsed = urlparse(base_url)
    localhost = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and localhost):
        raise RuntimeError(
            "TSM_API_BASE must use HTTPS; plain HTTP is only allowed for localhost"
        )
    if not localhost and os.getenv("TSM_ALLOW_REMOTE_TRANSCRIPT_UPLOAD") != "1":
        raise RuntimeError(
            "API refinement sends transcript excerpts to a remote service; set "
            "TSM_ALLOW_REMOTE_TRANSCRIPT_UPLOAD=1 only after authorization and privacy review"
        )
    if timeout < 1:
        raise ValueError("TSM_TIMEOUT_SECONDS must be positive")
    compact_transcript = {
        **{key: transcript.get(key) for key in ("video_id", "course_id", "title", "source_url", "transcript_kind")},
        "segments": transcript.get("segments", [])[:80],
    }
    prompt = (
        "You are a teaching-method mining system. Improve the baseline Teaching Skill using only "
        "evidence from the transcript. Preserve every top-level field and executable procedure shape. "
        "Every evidence quote must be an exact substring of one transcript segment. Return JSON only.\n\n"
        f"TRANSCRIPT:\n{json.dumps(compact_transcript, ensure_ascii=False)}\n\n"
        f"BASELINE_SKILL:\n{json.dumps(baseline_skill, ensure_ascii=False)}"
    )
    request_payload = {"model": model, "input": prompt}
    request = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"API request failed ({exc.code}): {detail}") from exc
    skill = _parse_json_text(_extract_output_text(response_payload))
    validation = validate_skill(skill)
    if not validation.valid:
        raise ValueError("API returned invalid Teaching Skill: " + "; ".join(validation.errors))
    return skill
