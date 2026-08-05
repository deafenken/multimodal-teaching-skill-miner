#!/usr/bin/env python3
"""Run a privacy-safe, black-box acceptance against the loopback Agent dashboard.

The runner intentionally exercises the same HTTP endpoints as the browser UI.  It
does not launch a browser, persist session content, or print the capability URL,
opaque session handles, profile contents, or learner messages.  Its only output is
an aggregate JSON receipt suitable for attaching to a local verification run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import secrets
import threading
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


RECEIPT_SCHEMA = "teaching_skill_miner.teacher_agent_system_acceptance.v1"
_CAPABILITY_PATH_PATTERN = r"/[A-Za-z0-9_-]{20,128}/"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MASTERY_DIMENSIONS = ("prerequisite", "conceptual", "procedural", "transfer")
_REQUIRED_INTERACTION_CONTRACT_FLAGS = (
    "start_requires_idempotency_key",
    "session_resume_supported",
    "step_requires_context_version",
    "independent_session_registry",
    "replacement_requires_expected_round",
    "replacement_requires_question_id",
    "replacement_requires_context_version",
    "replacement_requires_profile_revision",
)


class _AcceptanceFailure(RuntimeError):
    """One deliberately non-sensitive acceptance failure."""

    def __init__(self, stage: str, code: str) -> None:
        super().__init__(code)
        self.stage = stage
        self.code = code


class _NoRedirect(HTTPRedirectHandler):
    """Keep a loopback capability request from being redirected elsewhere."""

    def redirect_request(  # type: ignore[override]
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require(condition: bool, *, stage: str, code: str) -> None:
    if not condition:
        raise _AcceptanceFailure(stage, code)


def _bootstrap_contract_is_supported(bootstrap: Mapping[str, Any]) -> bool:
    """Keep the runner's bootstrap expectations tied to the public API names."""

    contract = bootstrap.get("interaction_contract")
    return bool(
        bootstrap.get("dashboard_kind")
        == "loopback_interactive_teacher_agent"
        and isinstance(contract, Mapping)
        and all(contract.get(key) is True for key in _REQUIRED_INTERACTION_CONTRACT_FLAGS)
    )


def _validated_base_url(raw: str) -> str:
    """Accept only the exact IPv4/IPv6 loopback capability URL shape."""

    import re

    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise _AcceptanceFailure("input", "invalid_base_url") from exc
    valid_host = parsed.hostname in {"127.0.0.1", "::1"}
    valid_path = bool(re.fullmatch(_CAPABILITY_PATH_PATTERN, parsed.path))
    if (
        parsed.scheme != "http"
        or not valid_host
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
        or not valid_path
    ):
        raise _AcceptanceFailure("input", "non_loopback_or_invalid_capability_url")
    return raw


class _DashboardHttpClient:
    def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
        self._base_url = _validated_base_url(base_url)
        self._timeout_seconds = timeout_seconds
        self._opener = build_opener(_NoRedirect())
        self._counter_lock = threading.Lock()
        self.request_count = 0
        self.success_count = 0
        self.rejection_count = 0

    def _record_status(self, status: int) -> None:
        with self._counter_lock:
            self.request_count += 1
            if 200 <= status < 300:
                self.success_count += 1
            elif 400 <= status < 500:
                self.rejection_count += 1

    @staticmethod
    def _decode_response(response: Any, *, stage: str) -> dict[str, Any]:
        payload = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(payload) > _MAX_RESPONSE_BYTES:
            raise _AcceptanceFailure(stage, "response_too_large")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _AcceptanceFailure(stage, "invalid_json_response") from exc
        if not isinstance(value, dict):
            raise _AcceptanceFailure(stage, "non_object_json_response")
        return value

    def request(
        self,
        method: str,
        route: str,
        *,
        stage: str,
        payload: Mapping[str, Any] | None = None,
        expected_status: int = 200,
    ) -> dict[str, Any]:
        url = urljoin(self._base_url, route)
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(
                dict(payload),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, headers=headers, method=method)
        try:
            response = self._opener.open(request, timeout=self._timeout_seconds)
        except HTTPError as exc:
            self._record_status(exc.code)
            if exc.code != expected_status:
                raise _AcceptanceFailure(stage, "unexpected_http_status") from exc
            return self._decode_response(exc, stage=stage)
        except (URLError, TimeoutError, OSError) as exc:
            raise _AcceptanceFailure(stage, "loopback_transport_failure") from exc
        with response:
            status = int(response.status)
            self._record_status(status)
            if status != expected_status:
                raise _AcceptanceFailure(stage, "unexpected_http_status")
            return self._decode_response(response, stage=stage)

    def get(self, route: str, *, stage: str) -> dict[str, Any]:
        return self.request("GET", route, stage=stage)

    def post(
        self,
        route: str,
        payload: Mapping[str, Any],
        *,
        stage: str,
        expected_status: int = 200,
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            route,
            stage=stage,
            payload=payload,
            expected_status=expected_status,
        )


def _profile_with_distinct_mastery(
    source: Mapping[str, Any], *, profile_ref: str, high: bool
) -> dict[str, Any]:
    profile = deepcopy(dict(source))
    source_mastery = source.get("initial_mastery", {})
    _require(
        isinstance(source_mastery, Mapping),
        stage="bootstrap",
        code="default_profile_mastery_missing",
    )
    target_mastery: dict[str, float] = {}
    for index, dimension in enumerate(_MASTERY_DIMENSIONS):
        raw = source_mastery.get(dimension, 0.0)
        _require(
            isinstance(raw, (int, float)) and not isinstance(raw, bool),
            stage="bootstrap",
            code="default_profile_mastery_invalid",
        )
        low_value = round(0.12 + index * 0.04, 2)
        high_value = round(0.82 - index * 0.05, 2)
        candidate = high_value if high else low_value
        if float(raw) == candidate:
            candidate = round(candidate - 0.07 if candidate > 0.5 else candidate + 0.07, 2)
        target_mastery[dimension] = candidate
    profile.update(
        {
            "profile_ref": profile_ref,
            "initial_mastery": target_mastery,
            "known_misconceptions": [],
            "conversation_history": [],
            "contains_direct_identity": False,
        }
    )
    return profile


def _start_payload(
    *,
    goal: Mapping[str, Any],
    profile: Mapping[str, Any],
    profile_revision: str,
    display_name: str,
    nonce: str,
    replace_session_id: str | None = None,
    replace_session: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "goal": deepcopy(dict(goal)),
        "student_profile": deepcopy(dict(profile)),
        "profile_revision": profile_revision,
        "profile_display_name": display_name,
        "start_idempotency_key": f"system-acceptance-{nonce}",
        "remote_processing_acknowledged": True,
    }
    if replace_session_id is not None:
        _require(
            isinstance(replace_session, Mapping),
            stage="replacement_binding",
            code="replacement_session_snapshot_missing",
        )
        profile_summary = replace_session.get("profile_summary", {})
        _require(
            isinstance(profile_summary, Mapping),
            stage="replacement_binding",
            code="replacement_profile_summary_missing",
        )
        replacement_round = replace_session.get("rounds_completed")
        replacement_question = replace_session.get("expected_question_id")
        replacement_context = replace_session.get("context_version")
        replacement_revision = profile_summary.get("profile_revision")
        _require(
            isinstance(replacement_round, int)
            and not isinstance(replacement_round, bool)
            and replacement_round >= 0
            and isinstance(replacement_question, str)
            and bool(replacement_question)
            and isinstance(replacement_context, int)
            and not isinstance(replacement_context, bool)
            and replacement_context >= 0
            and isinstance(replacement_revision, str)
            and bool(replacement_revision),
            stage="replacement_binding",
            code="replacement_session_snapshot_invalid",
        )
        payload["replace_session_id"] = replace_session_id
        payload.update(
            {
                "replace_expected_round": replacement_round,
                "replace_expected_question_id": replacement_question,
                "replace_expected_context_version": replacement_context,
                "replace_expected_profile_revision": replacement_revision,
            }
        )
    return payload


def _session_handle(value: Mapping[str, Any], *, stage: str) -> str:
    handle = value.get("session_id")
    _require(
        isinstance(handle, str) and bool(handle),
        stage=stage,
        code="session_handle_missing",
    )
    return handle


def _receipt(
    *,
    checks: Mapping[str, bool],
    client: _DashboardHttpClient | None,
    acknowledgement_present: bool = True,
    failure_stage: str | None = None,
    failure_code: str | None = None,
) -> dict[str, Any]:
    passed = bool(checks) and all(checks.values()) and failure_code is None
    result: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "dashboard_kind": "loopback_interactive_teacher_agent",
        "transport": "real_loopback_http",
        "passed": passed,
        "checks": dict(checks),
        "counts": {
            "synthetic_profiles_exercised": 3 if passed else 0,
            "sessions_started": 3 if passed else 0,
            "turns_committed": 1 if passed else 0,
            "expected_rejections_observed": 4 if passed else 0,
            "http_requests": client.request_count if client is not None else 0,
            "http_successes": client.success_count if client is not None else 0,
            "http_rejections": client.rejection_count if client is not None else 0,
        },
        "privacy": {
            "browser_launched": False,
            "capability_url_emitted": False,
            "session_handles_emitted": False,
            "profile_contents_emitted": False,
            "learner_messages_emitted": False,
        },
        "remote_demo_text_acknowledgement_present": acknowledgement_present,
    }
    if failure_code is not None:
        result["failure"] = {
            "stage": failure_stage or "internal",
            "code": failure_code,
        }
    return result


def run_system_acceptance(
    base_url: str,
    *,
    timeout_seconds: float = 90.0,
    acknowledge_remote_demo_text: bool,
) -> dict[str, Any]:
    """Exercise the full local session lifecycle and return an aggregate receipt."""

    if acknowledge_remote_demo_text is not True:
        return _receipt(
            checks={},
            client=None,
            acknowledgement_present=False,
            failure_stage="consent",
            failure_code="explicit_remote_demo_text_acknowledgement_required",
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 1 <= float(timeout_seconds) <= 300
    ):
        return _receipt(
            checks={},
            client=None,
            failure_stage="input",
            failure_code="invalid_timeout",
        )

    checks: dict[str, bool] = {}
    client: _DashboardHttpClient | None = None
    try:
        client = _DashboardHttpClient(
            base_url, timeout_seconds=float(timeout_seconds)
        )
        bootstrap = client.get("api/bootstrap", stage="bootstrap")
        _require(
            _bootstrap_contract_is_supported(bootstrap),
            stage="bootstrap",
            code="required_interaction_contract_missing",
        )
        goal = bootstrap.get("default_goal")
        profile_a = bootstrap.get("default_student_profile")
        _require(
            isinstance(goal, Mapping) and isinstance(profile_a, Mapping),
            stage="bootstrap",
            code="default_fixture_missing",
        )
        checks["bootstrap_contract"] = True

        nonce = secrets.token_hex(12)
        started_a = client.post(
            "api/start",
            _start_payload(
                goal=goal,
                profile=profile_a,
                profile_revision="system-acceptance-profile-a-v1",
                display_name="Synthetic A",
                nonce=f"{nonce}-start-a",
            ),
            stage="profile_a_start",
        )
        session_a = _session_handle(started_a, stage="profile_a_start")
        resumed_a = client.post(
            "api/session",
            {"session_id": session_a},
            stage="profile_a_resume",
        )
        _require(
            _canonical(started_a) == _canonical(resumed_a),
            stage="profile_a_resume",
            code="resume_did_not_restore_exact_session",
        )
        checks["profile_a_start_and_resume"] = True

        profile_b = _profile_with_distinct_mastery(
            profile_a, profile_ref="synthetic_system_acceptance_b", high=True
        )
        invalid_replace = _start_payload(
            goal=goal,
            profile=profile_b,
            profile_revision="system-acceptance-profile-b-v1",
            display_name="X" * 81,
            nonce=f"{nonce}-invalid-replace",
            replace_session_id=session_a,
            replace_session=resumed_a,
        )
        client.post(
            "api/start",
            invalid_replace,
            stage="transactional_replace_probe",
            expected_status=400,
        )
        still_a = client.post(
            "api/session",
            {"session_id": session_a},
            stage="transactional_replace_preservation",
        )
        _require(
            _canonical(still_a) == _canonical(resumed_a),
            stage="transactional_replace_preservation",
            code="failed_replace_mutated_original_session",
        )
        checks["failed_replace_preserves_original"] = True

        started_b = client.post(
            "api/start",
            _start_payload(
                goal=goal,
                profile=profile_b,
                profile_revision="system-acceptance-profile-b-v1",
                display_name="Synthetic B",
                nonce=f"{nonce}-start-b",
                replace_session_id=session_a,
                replace_session=resumed_a,
            ),
            stage="profile_b_replace",
        )
        session_b = _session_handle(started_b, stage="profile_b_replace")
        b_summary = started_b.get("profile_summary", {})
        b_state = started_b.get("student_state", {})
        b_mastery = (
            b_state.get("knowledge_mastery", {})
            if isinstance(b_state, Mapping)
            else {}
        )
        _require(
            session_b != session_a
            and started_b.get("rounds_completed") == 0
            and started_b.get("history") == []
            and isinstance(b_summary, Mapping)
            and b_summary.get("initial_mastery") == profile_b["initial_mastery"]
            and b_mastery == profile_b["initial_mastery"]
            and b_mastery != started_a.get("student_state", {}).get(
                "knowledge_mastery"
            ),
            stage="profile_b_replace",
            code="replacement_session_not_fresh_or_isolated",
        )
        client.post(
            "api/session",
            {"session_id": session_a},
            stage="profile_a_retirement",
            expected_status=400,
        )
        checks["profile_b_transactional_replace"] = True
        checks["profile_b_round_zero_empty_history"] = True
        checks["profile_b_distinct_mastery"] = True

        step_payload = {
            "session_id": session_b,
            "expected_round": started_b.get("rounds_completed"),
            "expected_question_id": started_b.get("expected_question_id"),
            "expected_context_version": started_b.get("context_version"),
            "profile_revision": "system-acceptance-profile-b-v1",
            "idempotency_key": f"system-acceptance-{nonce}-turn-b",
            "learner_response": "状态表示子问题的答案，转移关系连接相邻子问题。",
            "signal": "partial",
            "signal_confidence": 1.0,
        }
        stepped_b = client.post(
            "api/step", step_payload, stage="profile_b_step"
        )
        _require(
            stepped_b.get("rounds_completed") == 1
            and isinstance(stepped_b.get("history"), list)
            and len(stepped_b["history"]) == 1
            and isinstance(stepped_b.get("context_version"), int)
            and stepped_b["context_version"] > started_b.get("context_version", 0),
            stage="profile_b_step",
            code="turn_not_committed_once",
        )
        checks["one_turn_committed"] = True

        resumed_b = client.post(
            "api/session",
            {"session_id": session_b},
            stage="refresh_resume",
        )
        _require(
            _canonical(resumed_b) == _canonical(stepped_b),
            stage="refresh_resume",
            code="refresh_resume_state_mismatch",
        )
        checks["refresh_style_resume"] = True

        profile_c = _profile_with_distinct_mastery(
            profile_a, profile_ref="synthetic_system_acceptance_c", high=False
        )
        stale_start_c_payload = _start_payload(
            goal=goal,
            profile=profile_c,
            profile_revision="system-acceptance-profile-c-v1",
            display_name="Synthetic C",
            nonce=f"{nonce}-stale-start-c",
            replace_session_id=session_a,
            replace_session=resumed_a,
        )
        client.post(
            "api/start",
            stale_start_c_payload,
            stage="stale_profile_handle_rejection",
            expected_status=400,
        )
        start_c_payload = _start_payload(
            goal=goal,
            profile=profile_c,
            profile_revision="system-acceptance-profile-c-v1",
            display_name="Synthetic C",
            nonce=f"{nonce}-start-c",
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_c = pool.submit(
                client.post,
                "api/start",
                start_c_payload,
                stage="parallel_session_start",
            )
            future_b = pool.submit(
                client.post,
                "api/session",
                {"session_id": session_b},
                stage="parallel_original_resume",
            )
            started_c = future_c.result()
            parallel_b = future_b.result()
        session_c = _session_handle(started_c, stage="parallel_session_start")
        _require(
            session_c not in {session_a, session_b}
            and started_c.get("rounds_completed") == 0
            and started_c.get("history") == []
            and _canonical(parallel_b) == _canonical(resumed_b),
            stage="parallel_session_start",
            code="independent_session_isolation_failed",
        )
        checks["stale_profile_handle_retried_as_fresh_session"] = True
        checks["parallel_new_session_isolated"] = True

        before_rejected_step = parallel_b
        wrong_context = int(before_rejected_step["context_version"]) + 1
        client.post(
            "api/step",
            {
                "session_id": session_b,
                "expected_round": before_rejected_step.get("rounds_completed"),
                "expected_question_id": before_rejected_step.get(
                    "expected_question_id"
                ),
                "expected_context_version": wrong_context,
                "profile_revision": "system-acceptance-profile-b-v1",
                "idempotency_key": f"system-acceptance-{nonce}-stale-turn",
                "learner_response": "这条输入必须被过期上下文保护拒绝。",
                "signal": "correct",
                "signal_confidence": 1.0,
            },
            stage="stale_context_rejection",
            expected_status=400,
        )
        after_rejected_step = client.post(
            "api/session",
            {"session_id": session_b},
            stage="stale_context_preservation",
        )
        _require(
            _canonical(after_rejected_step) == _canonical(before_rejected_step),
            stage="stale_context_preservation",
            code="rejected_step_mutated_session",
        )
        checks["wrong_context_version_rejected"] = True
        checks["rejected_step_preserves_session"] = True

        _require(
            client.rejection_count == 4,
            stage="receipt",
            code="expected_rejection_count_mismatch",
        )
        return _receipt(checks=checks, client=client)
    except _AcceptanceFailure as exc:
        return _receipt(
            checks=checks,
            client=client,
            failure_stage=exc.stage,
            failure_code=exc.code,
        )
    except Exception:
        # A raw exception might embed a capability URL or request content.  Keep
        # stdout machine-readable and privacy-safe even for unforeseen failures.
        return _receipt(
            checks=checks,
            client=client,
            failure_stage="internal",
            failure_code="unexpected_internal_failure",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the loopback Teaching Agent system acceptance and emit only an "
            "aggregate JSON receipt."
        )
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="loopback dashboard capability base URL (never included in output)",
    )
    parser.add_argument(
        "--acknowledge-remote-demo-text",
        action="store_true",
        help=(
            "explicitly acknowledge that synthetic demo text may be sent to the "
            "configured remote model provider"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=90.0,
        help="per-request timeout in seconds (1-300; default: 90)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    receipt = run_system_acceptance(
        args.base_url,
        timeout_seconds=args.timeout_seconds,
        acknowledge_remote_demo_text=args.acknowledge_remote_demo_text,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    if receipt["passed"]:
        return 0
    return 2 if receipt.get("failure", {}).get("stage") == "consent" else 1


if __name__ == "__main__":
    raise SystemExit(main())
