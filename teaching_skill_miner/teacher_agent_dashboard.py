"""Loopback-only interactive dashboard for the task-two Teaching Agent."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import json
from pathlib import Path
import re
import secrets
import threading
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit
import webbrowser

from .deepseek_client import DeepSeekClient
from .io_utils import read_json
from .teacher_agent import (
    TeacherAgentError,
    advance_teacher_agent_session,
    evaluate_teacher_agent,
    session_turn_summary,
    start_teacher_agent_session,
    validate_skill_library,
)
from .teacher_agent_live import (
    LiveAgentOptions,
    advance_live_teacher_agent_session,
    live_session_view,
    parse_skill_command,
    start_live_teacher_agent_session,
    stop_live_teacher_agent_session,
)
from .teacher_agent_outcomes import evaluate_learning_observation


PACKAGE = "teaching_skill_miner.web"
HTML_RESOURCE = "teacher_agent_demo.html"
STYLE_RESOURCE = "teacher_agent_demo.css"
SCRIPT_RESOURCE = "teacher_agent_demo.js"
_SAFE_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_START_IDEMPOTENCY_ENTRIES = 16
_CSP = (
    "default-src 'none'; style-src 'self' 'unsafe-inline'; "
    "script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self'; "
    "base-uri 'none'; object-src 'none'; frame-ancestors 'none'; "
    "form-action 'none'; worker-src 'none'; manifest-src 'none'"
)


class TeacherAgentDashboardError(RuntimeError):
    """Raised when a task-two dashboard cannot be served safely."""


def _request_fingerprint(body: Mapping[str, Any]) -> str:
    """Return a stable, content-only fingerprint for one JSON request body."""

    try:
        encoded = json.dumps(
            dict(body),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeacherAgentDashboardError(
            "request body must be canonical JSON"
        ) from exc
    return sha256(encoded).hexdigest()


def _required_request_string(
    body: Mapping[str, Any], field_name: str, *, maximum: int = 200
) -> str:
    value = body.get(field_name)
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise TeacherAgentDashboardError(
            f"{field_name} must be a non-empty trimmed string"
        )
    return value


def _request_round(body: Mapping[str, Any], *, required: bool) -> int | None:
    if "expected_round" not in body:
        if required:
            raise TeacherAgentDashboardError("expected_round is required")
        return None
    value = body.get("expected_round")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TeacherAgentDashboardError(
            "expected_round must be a non-negative integer"
        )
    return value


def _resource_bytes(name: str) -> bytes:
    if name not in {HTML_RESOURCE, STYLE_RESOURCE, SCRIPT_RESOURCE}:
        raise TeacherAgentDashboardError("unknown teacher Agent dashboard resource")
    return resources.files(PACKAGE).joinpath(name).read_bytes()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _history_view(session: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in session.get("history", []):
        action = event["action"]
        state = event["student_state_after_observation"]
        rows.append(
            {
                "round": event["round"],
                "skill_id": action["primary_skill"]["skill_id"],
                "skill_name": action["primary_skill"]["name"],
                "skill_role": action["primary_skill"]["role"],
                "skill_switched": action["skill_switched"],
                "selection_reason": action["selection_reason"],
                "teacher_message": action["teacher_action"]["message"],
                "learner_response": event["learner_response"],
                "signal": event["structured_signal"],
                "mastery_after": deepcopy(state["knowledge_mastery"]),
            }
        )
    return rows


def _session_view(session: Mapping[str, Any]) -> dict[str, Any]:
    return {**session_turn_summary(session), "history": _history_view(session)}


def _library_view(library: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "skill_id": item["skill_id"],
            "name": item["name"],
            "role": item["role"],
            "focus_dimension": item["focus_dimension"],
            "selection_rationale": item["selection_rationale"],
            "source": deepcopy(item["source"]),
        }
        for item in library["skills"]
    ]


def _benchmark_receipt_view(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Expand the compact public receipt into the dashboard report shape."""

    if not receipt:
        return {}
    metrics = receipt["metrics"]
    configuration = receipt["configuration"]
    return {
        "schema": receipt["schema"],
        "run_status": receipt["run_status"],
        "run_config": {
            "prompt_version": configuration["prompt_version"],
            "case_count_per_repeat": configuration["case_count"],
            "repeats": configuration["repeat_count"],
            "temperature": configuration["temperature"],
            "thinking_mode": configuration["thinking_mode"],
        },
        "online_deepseek": {
            "status": receipt["run_status"],
            "provider": configuration["provider"],
            "model": configuration["model"],
            "metrics": {
                "signal": {
                    "end_to_end_all_attempts": {
                        "count": configuration["case_count"],
                        "accuracy": metrics["signal_accuracy"],
                        "macro_f1": metrics["signal_macro_f1"],
                    }
                },
                "misconception_tag": {
                    "end_to_end_exact_match_accuracy": metrics[
                        "misconception_tag_exact_match"
                    ]
                },
                "allowed_primary_skill": {
                    "end_to_end_hit_rate": metrics[
                        "allowed_primary_skill_hit_rate"
                    ]
                },
                "decision": {
                    "should_switch": {"f1": metrics["skill_switch_f1"]},
                    "should_terminate": {"f1": metrics["termination_f1"]},
                },
            },
            "operational": {
                "end_to_end_failure_rate": metrics[
                    "end_to_end_failure_rate"
                ],
                "all_attempt_wall_latency": {
                    "p50_ms": metrics["p50_latency_ms"],
                    "p95_ms": metrics["p95_latency_ms"],
                },
            },
        },
        "baselines": deepcopy(receipt["baselines"]),
        "claim_boundary": deepcopy(receipt["claim_boundary"]),
        "source_report": deepcopy(receipt["source_report"]),
    }


@dataclass(slots=True)
class TeacherAgentDashboardSnapshot:
    """Validated public fixtures plus one in-memory local session."""

    library: dict[str, Any]
    demo_input: dict[str, Any]
    evaluation: dict[str, Any]
    client: DeepSeekClient | None = None
    live_options: LiveAgentOptions = field(default_factory=LiveAgentOptions)
    neural_v1: dict[str, Any] = field(default_factory=dict)
    learning_outcome: dict[str, Any] = field(default_factory=dict)
    free_text_benchmark: dict[str, Any] = field(default_factory=dict)
    session: dict[str, Any] | None = None
    session_id: str | None = None
    pending_skill_id: str | None = None
    start_idempotency_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    step_idempotency_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    lock: threading.Lock = field(default_factory=threading.Lock)

    def bootstrap(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "dashboard_kind": "loopback_interactive_teacher_agent",
            "mode": "local_ephemeral_session",
            "provider_status": (
                self.client.public_status()
                if self.client is not None
                else {
                    "provider": "deterministic_fallback",
                    "model": None,
                    "configured": False,
                    "remote_student_data_opt_in": False,
                    "api_key_exposed": False,
                }
            ),
            "neural_v1": deepcopy(self.neural_v1),
            "default_goal": deepcopy(self.demo_input["goal"]),
            "default_student_profile": deepcopy(self.demo_input["student_profile"]),
            "skills": _library_view(self.library),
            "evaluation": {
                "aggregate": deepcopy(self.evaluation["aggregate"]),
                "gates": deepcopy(self.evaluation["gates"]),
                "passed": self.evaluation["passed"],
                "baseline": deepcopy(self.evaluation["baseline"]),
                "claim_boundary": deepcopy(self.evaluation["claim_boundary"]),
                "cases": deepcopy(self.evaluation.get("cases", [])),
                "learning_outcome": deepcopy(self.learning_outcome),
                "free_text_benchmark": _benchmark_receipt_view(
                    self.free_text_benchmark
                ),
            },
            "interaction_contract": {
                "one_action_per_turn": True,
                "structured_signal_required": False,
                "free_text_assessment_enabled": self.client is not None,
                "skill_selection_reason_exposed": True,
                "skill_switching_enabled": True,
                "success_and_unable_termination": True,
                "adaptive_profile_candidates_enabled": self.client is not None,
                "adaptive_profile_candidates_are_teacher_confirmed": False,
                "start_requires_idempotency_key": True,
                "active_session_replacement_requires_session_id": True,
                "step_requires_session_id": True,
                "step_requires_expected_round": True,
                "step_requires_idempotency_key": True,
                "command_requires_session_id": True,
                "command_accepts_expected_round": True,
                "remote_processing_acknowledgement_required": self.client
                is not None,
                "session_persisted_to_browser": False,
            },
        }

    def start(self, body: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            idempotency_key = _required_request_string(
                body, "start_idempotency_key"
            )
            request_fingerprint = _request_fingerprint(body)
            cached = self.start_idempotency_cache.get(idempotency_key)
            if cached is not None:
                if cached["request_fingerprint"] != request_fingerprint:
                    raise TeacherAgentDashboardError(
                        "start_idempotency_key was already used for a different request"
                    )
                if cached["session_id"] != self.session_id:
                    raise TeacherAgentDashboardError(
                        "start_idempotency_key belongs to an inactive session"
                    )
                return deepcopy(cached["response"])

            replacement_id = body.get("replace_session_id")
            if self.session is None:
                if replacement_id is not None:
                    _required_request_string(body, "replace_session_id")
                    raise TeacherAgentDashboardError(
                        "replace_session_id was provided but no session is active"
                    )
            else:
                replacement_id = _required_request_string(
                    body, "replace_session_id"
                )
                if replacement_id != self.session_id:
                    raise TeacherAgentDashboardError(
                        "replace_session_id does not match the active session"
                    )
            if self.client is not None:
                if body.get("remote_processing_acknowledged") is not True:
                    raise TeacherAgentDashboardError(
                        "online start requires remote_processing_acknowledged=true"
                    )
                new_session = start_live_teacher_agent_session(
                    body.get("goal", {}),
                    body.get("student_profile", {}),
                    self.library,
                    self.client,
                    options=self.live_options,
                    allowed_skill_ids=(
                        list(body["allowed_skill_ids"])
                        if isinstance(body.get("allowed_skill_ids"), list)
                        else None
                    ),
                )
            else:
                new_session = start_teacher_agent_session(
                    body.get("goal", {}),
                    body.get("student_profile", {}),
                    self.library,
                )
            self.session = new_session
            self.session_id = secrets.token_urlsafe(16)
            self.pending_skill_id = None
            self.step_idempotency_cache.clear()
            view = (
                live_session_view(self.session)
                if self.client is not None
                else _session_view(self.session)
            )
            response = {**view, "session_id": self.session_id}
            self.start_idempotency_cache[idempotency_key] = {
                "request_fingerprint": request_fingerprint,
                "session_id": self.session_id,
                "response": deepcopy(response),
            }
            while (
                len(self.start_idempotency_cache)
                > _MAX_START_IDEMPOTENCY_ENTRIES
            ):
                oldest_key = next(iter(self.start_idempotency_cache))
                del self.start_idempotency_cache[oldest_key]
            return response

    def step(self, body: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.session is None:
                raise TeacherAgentDashboardError("start a session before submitting a turn")
            request_session_id = _required_request_string(body, "session_id")
            if request_session_id != self.session_id:
                raise TeacherAgentDashboardError("session_id does not match the active session")
            expected_round = _request_round(body, required=True)
            idempotency_key = _required_request_string(body, "idempotency_key")
            request_fingerprint = _request_fingerprint(body)
            cached = self.step_idempotency_cache.get(idempotency_key)
            if cached is not None:
                if cached["request_fingerprint"] != request_fingerprint:
                    raise TeacherAgentDashboardError(
                        "idempotency_key was already used for a different request"
                    )
                return deepcopy(cached["response"])
            if expected_round != self.session["round"]:
                raise TeacherAgentDashboardError(
                    "expected_round does not match the active session"
                )
            if self.client is not None:
                requested_skill = (
                    str(body["manual_skill_id"])
                    if body.get("manual_skill_id")
                    else self.pending_skill_id
                )
                self.session = advance_live_teacher_agent_session(
                    self.session,
                    learner_response=str(body.get("learner_response", "")),
                    client=self.client,
                    manual_skill_id=requested_skill,
                    options=self.live_options,
                )
                self.pending_skill_id = None
                view = live_session_view(self.session)
            else:
                self.session = advance_teacher_agent_session(
                    self.session,
                    learner_response=str(body.get("learner_response", "")),
                    signal=str(body.get("signal", "confused")),
                    misconception_tag=(
                        str(body["misconception_tag"])
                        if body.get("misconception_tag")
                        else None
                    ),
                    signal_confidence=float(body.get("signal_confidence", 1.0)),
                )
                view = _session_view(self.session)
            response = {**view, "session_id": self.session_id}
            self.step_idempotency_cache[idempotency_key] = {
                "request_fingerprint": request_fingerprint,
                "response": deepcopy(response),
            }
            return response

    def command(self, body: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.session is None:
                raise TeacherAgentDashboardError("start a session before sending a command")
            request_session_id = _required_request_string(body, "session_id")
            if request_session_id != self.session_id:
                raise TeacherAgentDashboardError("session_id does not match the active session")
            expected_round = _request_round(body, required=False)
            if expected_round is not None and expected_round != self.session["round"]:
                raise TeacherAgentDashboardError(
                    "expected_round does not match the active session"
                )
            command = str(body.get("command", "")).strip()
            if command == "auto":
                self.pending_skill_id = None
            elif command == "select_skill":
                skill_id = str(body.get("skill_id", "")).strip()
                parsed = parse_skill_command(
                    f"/+skill {skill_id}", self.session["skill_library"]
                )
                self.pending_skill_id = str(parsed["skill_id"])
            elif command == "stop":
                if self.client is not None:
                    self.session = stop_live_teacher_agent_session(
                        self.session, reason="teacher requested stop from dashboard"
                    )
                else:
                    raise TeacherAgentDashboardError("manual stop requires a live session")
            else:
                raise TeacherAgentDashboardError("unsupported Agent command")
            view = (
                live_session_view(self.session)
                if self.client is not None
                else _session_view(self.session)
            )
            return {
                **view,
                "session_id": self.session_id,
                "pending_skill_id": self.pending_skill_id,
            }


def build_teacher_agent_dashboard_snapshot(
    library_path: str | Path,
    demo_input_path: str | Path,
    evaluation_cases_path: str | Path,
    *,
    client: DeepSeekClient | None = None,
    live_options: LiveAgentOptions | None = None,
    neural_v1_manifest_path: str | Path | None = None,
    learning_outcome_path: str | Path | None = None,
    free_text_benchmark_receipt_path: str | Path | None = None,
) -> TeacherAgentDashboardSnapshot:
    library = read_json(library_path)
    demo_input = read_json(demo_input_path)
    cases = read_json(evaluation_cases_path)
    if not isinstance(library, dict) or not isinstance(demo_input, dict):
        raise TeacherAgentDashboardError("teacher Agent fixtures must be JSON objects")
    validate_skill_library(library)
    if not isinstance(demo_input.get("goal"), dict) or not isinstance(
        demo_input.get("student_profile"), dict
    ):
        raise TeacherAgentDashboardError("teacher Agent demo input is incomplete")
    evaluation = evaluate_teacher_agent(library, cases)
    neural_v1: dict[str, Any] = {}
    if neural_v1_manifest_path is not None:
        loaded_manifest = read_json(neural_v1_manifest_path)
        if not isinstance(loaded_manifest, dict):
            raise TeacherAgentDashboardError("neural-v1 runtime manifest must be an object")
        neural_v1 = loaded_manifest
    learning_outcome: dict[str, Any] = {}
    if learning_outcome_path is not None:
        outcome_observation = read_json(learning_outcome_path)
        if not isinstance(outcome_observation, dict):
            raise TeacherAgentDashboardError(
                "learning outcome observation must be an object"
            )
        learning_outcome = evaluate_learning_observation(outcome_observation)
    free_text_benchmark: dict[str, Any] = {}
    if free_text_benchmark_receipt_path is not None:
        loaded_receipt = read_json(free_text_benchmark_receipt_path)
        if not isinstance(loaded_receipt, dict) or loaded_receipt.get("schema") != (
            "teaching_skill_miner.teacher_agent_free_text_benchmark_receipt.v1"
        ):
            raise TeacherAgentDashboardError(
                "free-text benchmark receipt is missing or has an invalid schema"
            )
        boundaries = loaded_receipt.get("claim_boundary", {})
        configuration = loaded_receipt.get("configuration", {})
        if (
            not isinstance(boundaries, Mapping)
            or boundaries.get("expert_validated") is not False
            or boundaries.get("deployment_accuracy_established") is not False
            or not isinstance(configuration, Mapping)
            or configuration.get("model") != "deepseek-v4-flash"
        ):
            raise TeacherAgentDashboardError(
                "free-text benchmark receipt overstates evidence or model identity"
            )
        free_text_benchmark = loaded_receipt
    return TeacherAgentDashboardSnapshot(
        library=library,
        demo_input=demo_input,
        evaluation=evaluation,
        client=client,
        live_options=(live_options or LiveAgentOptions()).validated(),
        neural_v1=neural_v1,
        learning_outcome=learning_outcome,
        free_text_benchmark=free_text_benchmark,
    )


def teacher_agent_dashboard_self_check(
    library_path: str | Path,
    demo_input_path: str | Path,
    evaluation_cases_path: str | Path,
) -> dict[str, Any]:
    snapshot = build_teacher_agent_dashboard_snapshot(
        library_path, demo_input_path, evaluation_cases_path
    )
    resources_by_name = {
        HTML_RESOURCE: _resource_bytes(HTML_RESOURCE),
        STYLE_RESOURCE: _resource_bytes(STYLE_RESOURCE),
        SCRIPT_RESOURCE: _resource_bytes(SCRIPT_RESOURCE),
    }
    text = "\n".join(value.decode("utf-8") for value in resources_by_name.values())
    required = (
        'data-screen-label="01 Session setup"',
        'data-screen-label="02 Live teaching loop"',
        'data-screen-label="03 Student state"',
        'data-screen-label="04 Reproducible evaluation"',
        "Skill 选择依据",
        "学生状态",
        "Agent 候选画像",
        "不会覆盖教师输入",
        "固定单 Skill 基线",
        "模拟增益，不是实际学习效果",
        "fetch(",
        ".textContent",
    )
    forbidden = (
        "http://example",
        "https://",
        "artifacts/private",
        "/Volumes/",
        "/Users/",
        ".innerHTML",
    )
    missing = [marker for marker in required if marker not in text]
    forbidden_matches = [marker for marker in forbidden if marker in text]
    return {
        "schema_version": "1.0",
        "dashboard_kind": "loopback_interactive_teacher_agent",
        "passed": not missing and not forbidden_matches and snapshot.evaluation["passed"],
        "missing_markers": missing,
        "forbidden_matches": forbidden_matches,
        "skill_count": len(snapshot.library["skills"]),
        "evaluation_passed": snapshot.evaluation["passed"],
        "one_action_per_turn": True,
        "real_time_skill_switching": True,
        "free_text_answer_grading_established": False,
        "real_learning_effectiveness_established": False,
    }


def create_teacher_agent_dashboard_server(
    snapshot: TeacherAgentDashboardSnapshot,
    *,
    port: int = 0,
    capability_token: str | None = None,
) -> tuple[ThreadingHTTPServer, str]:
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise TeacherAgentDashboardError("teacher Agent dashboard port is invalid")
    token = capability_token or secrets.token_urlsafe(24)
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,128}", token):
        raise TeacherAgentDashboardError("teacher Agent capability token is invalid")
    prefix = f"/{token}/"

    class TeacherAgentHandler(BaseHTTPRequestHandler):
        server_version = "TeachingSkillMinerAgent/1.0"
        sys_version = ""

        def log_message(self, _format: str, *args: Any) -> None:
            del args

        def _secure_headers(self) -> None:
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header(
                "Permissions-Policy",
                "camera=(), microphone=(), geolocation=(), payment=()",
            )
            self.send_header("Content-Security-Policy", _CSP)

        def _request_is_local(self) -> bool:
            host_header = self.headers.get("Host", "")
            try:
                authority = urlsplit(f"//{host_header}")
                host = (authority.hostname or "").casefold()
                request_port = authority.port
            except ValueError:
                return False
            if host not in _SAFE_HOSTS or request_port not in {
                None,
                self.server.server_port,
            }:
                return False
            if self.client_address[0] not in {"127.0.0.1", "::1"}:
                return False
            if self.headers.get("Sec-Fetch-Site", "").casefold() in {
                "cross-site",
                "cross-origin",
            }:
                return False
            origin = self.headers.get("Origin")
            if origin:
                parsed = urlsplit(origin)
                if parsed.scheme != "http" or (parsed.hostname or "").casefold() not in _SAFE_HOSTS:
                    return False
                if parsed.port not in {None, self.server.server_port}:
                    return False
            return True

        def _payload(
            self,
            payload: bytes,
            *,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self._secure_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _error(self, status: HTTPStatus, message: str) -> None:
            self._payload(
                _json_bytes({"error": message, "status": int(status)}),
                content_type="application/json; charset=utf-8",
                status=status,
            )

        def _route_name(self) -> str | None:
            if not self._request_is_local():
                self._error(HTTPStatus.FORBIDDEN, "loopback origin required")
                return None
            target = urlsplit(self.path)
            if target.query or target.fragment:
                self._error(HTTPStatus.BAD_REQUEST, "query strings are not supported")
                return None
            decoded = unquote(target.path)
            if ".." in decoded.split("/") or not decoded.startswith(prefix):
                self._error(HTTPStatus.FORBIDDEN, "valid capability path required")
                return None
            return decoded[len(prefix) :]

        def _read_body(self) -> dict[str, Any]:
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                raise TeacherAgentDashboardError("Content-Type must be application/json")
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length or "")
            except ValueError as exc:
                raise TeacherAgentDashboardError("valid Content-Length required") from exc
            if not 1 <= length <= _MAX_REQUEST_BYTES:
                raise TeacherAgentDashboardError("request body size is invalid")
            try:
                value = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise TeacherAgentDashboardError("request body is invalid JSON") from exc
            if not isinstance(value, dict):
                raise TeacherAgentDashboardError("request body must be one JSON object")
            return value

        def _get(self, route: str) -> None:
            if route in {"", "index.html"}:
                self._payload(_resource_bytes(HTML_RESOURCE), content_type="text/html; charset=utf-8")
            elif route == "assets/teacher_agent_demo.css":
                self._payload(_resource_bytes(STYLE_RESOURCE), content_type="text/css; charset=utf-8")
            elif route == "assets/teacher_agent_demo.js":
                self._payload(_resource_bytes(SCRIPT_RESOURCE), content_type="text/javascript; charset=utf-8")
            elif route == "api/bootstrap":
                self._payload(_json_bytes(snapshot.bootstrap()), content_type="application/json; charset=utf-8")
            else:
                self._error(HTTPStatus.NOT_FOUND, "resource not found")

        def _post(self, route: str) -> None:
            try:
                body = self._read_body()
                if route == "api/start":
                    result = snapshot.start(body)
                elif route == "api/step":
                    result = snapshot.step(body)
                elif route == "api/command":
                    result = snapshot.command(body)
                else:
                    self._error(HTTPStatus.NOT_FOUND, "resource not found")
                    return
            except (TeacherAgentError, TeacherAgentDashboardError, TypeError, ValueError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._payload(_json_bytes(result), content_type="application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            route = self._route_name()
            if route is not None:
                self._get(route)

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
            self.do_GET()

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            route = self._route_name()
            if route is not None:
                self._post(route)

    server = ThreadingHTTPServer(("127.0.0.1", port), TeacherAgentHandler)
    server.daemon_threads = True
    return server, f"http://127.0.0.1:{server.server_port}/{token}/"


def serve_teacher_agent_dashboard(
    library_path: str | Path,
    demo_input_path: str | Path,
    evaluation_cases_path: str | Path,
    *,
    port: int = 0,
    open_browser: bool = True,
    client: DeepSeekClient | None = None,
    live_options: LiveAgentOptions | None = None,
    neural_v1_manifest_path: str | Path | None = None,
    learning_outcome_path: str | Path | None = None,
    free_text_benchmark_receipt_path: str | Path | None = None,
) -> int:
    snapshot = build_teacher_agent_dashboard_snapshot(
        library_path,
        demo_input_path,
        evaluation_cases_path,
        client=client,
        live_options=live_options,
        neural_v1_manifest_path=neural_v1_manifest_path,
        learning_outcome_path=learning_outcome_path,
        free_text_benchmark_receipt_path=free_text_benchmark_receipt_path,
    )
    server, url = create_teacher_agent_dashboard_server(snapshot, port=port)
    status = {
        "schema_version": "1.0",
        "dashboard_kind": "loopback_interactive_teacher_agent",
        "dashboard_url": url,
        "loopback_only": True,
        "cache_control": "no-store",
        "skill_count": len(snapshot.library["skills"]),
        "evaluation_passed": snapshot.evaluation["passed"],
        "real_time_skill_switching": True,
        "free_text_answer_processing_enabled": client is not None,
        "free_text_answer_grading_established": False,
        "real_learning_effectiveness_established": False,
        "provider_status": (
            client.public_status()
            if client is not None
            else {"provider": "deterministic_fallback", "configured": False}
        ),
        "browser_open_requested": open_browser,
    }
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0
