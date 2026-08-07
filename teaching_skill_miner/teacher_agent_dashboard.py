"""Loopback-only interactive dashboard for the task-two Teaching Agent."""

from __future__ import annotations

import base64
import binascii
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
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlsplit
import webbrowser

from .deepseek_client import DeepSeekClient
from .io_utils import read_json
from .teacher_agent import (
    TeacherAgentError,
    _refresh_integrity,
    advance_teacher_agent_session,
    evaluate_teacher_agent,
    session_turn_summary,
    start_teacher_agent_session,
    validate_session,
    validate_skill_library,
)
from .teacher_agent_live import (
    LIVE_PROMPT_VERSION,
    LiveAgentOptions,
    LiveTeacherAgentError,
    advance_live_teacher_agent_session,
    live_session_view,
    parse_skill_command,
    start_live_teacher_agent_session,
    stop_live_teacher_agent_session,
    validate_live_runtime_policy_contract,
)
from .teacher_agent_outcomes import evaluate_learning_observation
from .teacher_agent_store import (
    TeacherAgentStore,
    TeacherAgentStoreError,
)
from .teacher_agent_vision import (
    LocalVisualEvidenceError,
    MAX_IMAGE_BYTES,
    extract_local_visual_evidence,
)


PACKAGE = "teaching_skill_miner.web"
HTML_RESOURCE = "teacher_agent_demo.html"
STYLE_RESOURCE = "teacher_agent_demo.css"
SCRIPT_RESOURCE = "teacher_agent_demo.js"
AVATAR_RESOURCES = frozenset(
    {
        "student-xiaoyu.png",
        "student-zimo.png",
        "student-zhixing.png",
    }
)
_SAFE_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_ATTACHMENT_REQUEST_BYTES = 6 * 1024 * 1024
_MAX_START_IDEMPOTENCY_ENTRIES = 16
_MAX_ACTIVE_SESSIONS = 16
_MAX_STEP_IDEMPOTENCY_ENTRIES = 64
_MAX_COMMAND_IDEMPOTENCY_ENTRIES = 64
_MAX_ATTACHMENT_IDEMPOTENCY_ENTRIES = 24
_MAX_PENDING_ATTACHMENTS = 4
_CSP = (
    "default-src 'none'; style-src 'self' 'unsafe-inline'; "
    "script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' blob:; "
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
        raise TeacherAgentDashboardError("request body must be canonical JSON") from exc
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


def _required_nonnegative_integer(body: Mapping[str, Any], field_name: str) -> int:
    value = body.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TeacherAgentDashboardError(f"{field_name} must be a non-negative integer")
    return value


def _record_question_id(record: "_DashboardSessionRecord") -> str | None:
    action = record.session.get("current_action", {})
    teacher_action = (
        action.get("teacher_action", {}) if isinstance(action, Mapping) else {}
    )
    question_id = (
        teacher_action.get("question_id")
        if isinstance(teacher_action, Mapping)
        else None
    )
    if not question_id and isinstance(action, Mapping):
        question_id = action.get("action_id")
    return str(question_id) if question_id else None


def _validate_replacement_guards(
    body: Mapping[str, Any], record: "_DashboardSessionRecord"
) -> None:
    expected_round = _required_nonnegative_integer(body, "replace_expected_round")
    if expected_round != record.session.get("round"):
        raise TeacherAgentDashboardError(
            "replace_expected_round does not match this session"
        )
    expected_question_id = _required_request_string(
        body, "replace_expected_question_id"
    )
    if expected_question_id != _record_question_id(record):
        raise TeacherAgentDashboardError(
            "replace_expected_question_id does not match this session"
        )
    expected_context_version = _required_nonnegative_integer(
        body, "replace_expected_context_version"
    )
    if expected_context_version != record.context_version:
        raise TeacherAgentDashboardError(
            "replace_expected_context_version does not match this session"
        )
    expected_profile_revision = _required_request_string(
        body, "replace_expected_profile_revision", maximum=120
    )
    if expected_profile_revision != record.profile_revision:
        raise TeacherAgentDashboardError(
            "replace_expected_profile_revision does not match this session"
        )


def _validate_session_turn_guards(
    body: Mapping[str, Any], record: "_DashboardSessionRecord"
) -> tuple[int, str, str]:
    """Validate the Codex-style identity/version tuple for one active turn."""

    expected_round = _request_round(body, required=True)
    if expected_round != record.session.get("round"):
        raise TeacherAgentDashboardError("expected_round does not match this session")
    expected_context_version = _required_nonnegative_integer(
        body, "expected_context_version"
    )
    if expected_context_version != record.context_version:
        raise TeacherAgentDashboardError(
            "expected_context_version does not match this session"
        )
    expected_question_id = _required_request_string(body, "expected_question_id")
    if expected_question_id != _record_question_id(record):
        raise TeacherAgentDashboardError(
            "expected_question_id does not match this session"
        )
    profile_revision = _required_request_string(body, "profile_revision", maximum=120)
    if profile_revision != record.profile_revision:
        raise TeacherAgentDashboardError("profile_revision does not match this session")
    return expected_round, expected_question_id, profile_revision


def _resource_bytes(name: str) -> bytes:
    if name not in {HTML_RESOURCE, STYLE_RESOURCE, SCRIPT_RESOURCE}:
        raise TeacherAgentDashboardError("unknown teacher Agent dashboard resource")
    return resources.files(PACKAGE).joinpath(name).read_bytes()


def _avatar_bytes(name: str) -> bytes:
    if name not in AVATAR_RESOURCES:
        raise TeacherAgentDashboardError("unknown student avatar resource")
    return resources.files(PACKAGE).joinpath("assets", name).read_bytes()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _history_view(session: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in session.get("history", []):
        action = event["action"]
        state = event["student_state_after_observation"]
        visual_evidence = []
        for item in event.get("multimodal_evidence", []):
            if not isinstance(item, Mapping):
                continue
            visual_evidence.append(
                {
                    "schema": item.get("schema"),
                    "source_modality": item.get("source_modality"),
                    "source_kind": item.get("source_kind"),
                    "display_name": item.get("display_name"),
                    "mime_type": item.get("mime_type"),
                    "byte_size": item.get("byte_size"),
                    "content_sha256": item.get("content_sha256"),
                    "engine": item.get("engine"),
                    "status": item.get("status"),
                    "recognized_text": item.get("recognized_text"),
                    "confidence": item.get("confidence"),
                    "confidence_semantics": item.get("confidence_semantics"),
                    "formula_like_text_detected": bool(
                        item.get("formula_like_text_detected")
                    ),
                    "formula_accuracy_established": False,
                    "extractor_fallback_used": bool(
                        item.get("extractor_fallback_used")
                    ),
                    "needs_student_confirmation": bool(
                        item.get("needs_student_confirmation")
                    ),
                    "student_confirmed_recognized_text": bool(
                        item.get("student_confirmed_recognized_text")
                    ),
                    "student_confirmation_method": item.get(
                        "student_confirmation_method"
                    ),
                    "ocr_confirmation_was_required": bool(
                        item.get("ocr_confirmation_was_required")
                    ),
                    "student_confirmation_establishes_answer_correctness": False,
                    "raw_media_retained": False,
                    "remote_media_sent": False,
                    "remote_representation": "bounded_redacted_ocr_text_only",
                }
            )
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
                "learner_text": str(event.get("learner_text", "")),
                "multimodal_evidence": visual_evidence,
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
                    "end_to_end_hit_rate": metrics["allowed_primary_skill_hit_rate"]
                },
                "decision": {
                    "should_switch": {"f1": metrics["skill_switch_f1"]},
                    "should_terminate": {"f1": metrics["termination_f1"]},
                },
            },
            "operational": {
                "end_to_end_failure_rate": metrics["end_to_end_failure_rate"],
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
class _DashboardSessionRecord:
    """One isolated local Teaching Session and its request controls."""

    session: dict[str, Any]
    profile_revision: str
    profile_display_name: str
    pending_skill_id: str | None = None
    control_notice: str | None = None
    context_version: int = 1
    step_idempotency_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    command_idempotency_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    attachment_idempotency_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    recovered_aborted_turns: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    attachments: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    # Active-turn fields are process-local concurrency controls.  They are not
    # checkpointed: a durable ``turn_started`` without a terminal event is
    # recovered as ``turn_aborted`` by the store replay path.
    active_turn_id: str | None = field(default=None, repr=False)
    active_turn_idempotency_key: str | None = field(default=None, repr=False)
    active_turn_request_fingerprint: str | None = field(default=None, repr=False)
    active_turn_generation: int = field(default=0, repr=False)
    active_turn_cancelled: bool = field(default=False, repr=False)
    active_turn_cancel_reason: str | None = field(default=None, repr=False)
    retiring: bool = field(default=False, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _begin_active_turn(
    record: _DashboardSessionRecord,
    *,
    turn_id: str,
    idempotency_key: str,
    request_fingerprint: str,
) -> int:
    if record.retiring:
        raise TeacherAgentDashboardError(
            "session replacement is in progress; this turn was not started"
        )
    if record.active_turn_id is not None:
        if record.active_turn_idempotency_key == idempotency_key:
            if record.active_turn_request_fingerprint != request_fingerprint:
                raise TeacherAgentDashboardError(
                    "idempotency_key is already running with a different request"
                )
            raise TeacherAgentDashboardError(
                "this idempotent turn is still running; retry after it completes"
            )
        raise TeacherAgentDashboardError(
            "another turn is already running for this session"
        )
    record.active_turn_generation += 1
    record.active_turn_id = turn_id
    record.active_turn_idempotency_key = idempotency_key
    record.active_turn_request_fingerprint = request_fingerprint
    record.active_turn_cancelled = False
    record.active_turn_cancel_reason = None
    return record.active_turn_generation


def _cancel_active_turn(
    record: _DashboardSessionRecord, *, reason: str
) -> bool:
    if record.active_turn_id is None:
        return False
    record.active_turn_generation += 1
    record.active_turn_cancelled = True
    record.active_turn_cancel_reason = reason
    return True


def _clear_active_turn(
    record: _DashboardSessionRecord, *, turn_id: str
) -> None:
    if record.active_turn_id != turn_id:
        return
    record.active_turn_id = None
    record.active_turn_idempotency_key = None
    record.active_turn_request_fingerprint = None
    record.active_turn_cancelled = False
    record.active_turn_cancel_reason = None


def _record_store_value(record: _DashboardSessionRecord) -> dict[str, Any]:
    """Return the complete JSON state required for exact cold recovery."""

    return {
        "schema": "teaching_skill_miner.teacher_agent_dashboard_record.v1",
        "session": deepcopy(record.session),
        "profile_revision": record.profile_revision,
        "profile_display_name": record.profile_display_name,
        "pending_skill_id": record.pending_skill_id,
        "control_notice": record.control_notice,
        "context_version": record.context_version,
        "step_idempotency_cache": deepcopy(record.step_idempotency_cache),
        "command_idempotency_cache": deepcopy(record.command_idempotency_cache),
        "attachment_idempotency_cache": deepcopy(
            record.attachment_idempotency_cache
        ),
        "recovered_aborted_turns": deepcopy(record.recovered_aborted_turns),
        "attachments": deepcopy(record.attachments),
    }


def _stored_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TeacherAgentDashboardError(
            f"durable session {field_name} must be an object"
        )
    return deepcopy(dict(value))


def _validate_durable_skill_library(
    stored_library: Mapping[str, Any],
    dashboard_library: Mapping[str, Any],
    *,
    allow_primary_subset: bool,
) -> None:
    """Bind a recovered session to the configured Skill Library contents.

    Live sessions may intentionally persist only the primary Skills selected by
    ``allowed_skill_ids``.  Such a session must remain an order-preserving,
    byte-content-equivalent subset and must retain every support Skill.  The
    deterministic backend has no subset option and therefore requires the full
    configured library exactly.
    """

    if not allow_primary_subset:
        if _request_fingerprint(stored_library) != _request_fingerprint(
            dashboard_library
        ):
            raise TeacherAgentDashboardError(
                "durable session Skill library does not match this dashboard"
            )
        return

    stored_material = deepcopy(dict(stored_library))
    dashboard_material = deepcopy(dict(dashboard_library))
    stored_skills = stored_material.pop("skills", None)
    dashboard_skills = dashboard_material.pop("skills", None)
    # A filtered library deliberately drops the full-library digest because it
    # no longer describes the subset.  All other library metadata is invariant.
    stored_material.pop("content_sha256", None)
    dashboard_material.pop("content_sha256", None)
    if (
        not isinstance(stored_skills, list)
        or not isinstance(dashboard_skills, list)
        or _request_fingerprint(stored_material)
        != _request_fingerprint(dashboard_material)
    ):
        raise TeacherAgentDashboardError(
            "durable session Skill library metadata does not match this dashboard"
        )

    expected_by_id = {
        str(skill["skill_id"]): skill
        for skill in dashboard_skills
        if isinstance(skill, Mapping) and isinstance(skill.get("skill_id"), str)
    }
    stored_by_id = {
        str(skill["skill_id"]): skill
        for skill in stored_skills
        if isinstance(skill, Mapping) and isinstance(skill.get("skill_id"), str)
    }
    stored_ids = [str(skill["skill_id"]) for skill in stored_skills]
    unknown_ids = sorted(set(stored_by_id) - set(expected_by_id))
    if unknown_ids:
        raise TeacherAgentDashboardError(
            "durable session Skill library contains unknown Skills: "
            + ", ".join(unknown_ids)
        )
    altered_ids = sorted(
        skill_id
        for skill_id, stored_skill in stored_by_id.items()
        if _request_fingerprint(stored_skill)
        != _request_fingerprint(expected_by_id[skill_id])
    )
    if altered_ids:
        raise TeacherAgentDashboardError(
            "durable session Skill definitions differ from this dashboard: "
            + ", ".join(altered_ids)
        )

    expected_support_ids = [
        str(skill["skill_id"])
        for skill in dashboard_skills
        if skill.get("role") == "support"
    ]
    stored_support_ids = [
        str(skill["skill_id"])
        for skill in stored_skills
        if skill.get("role") == "support"
    ]
    if stored_support_ids != expected_support_ids:
        raise TeacherAgentDashboardError(
            "durable session Skill subset does not preserve all support Skills"
        )
    expected_subset_order = [
        str(skill["skill_id"])
        for skill in dashboard_skills
        if str(skill["skill_id"]) in stored_by_id
    ]
    if stored_ids != expected_subset_order:
        raise TeacherAgentDashboardError(
            "durable session Skill subset does not preserve library order"
        )

    # When no primary filtering occurred, retain the exact full-library
    # contract, including any declared content digest.
    if len(stored_ids) == len(dashboard_skills) and _request_fingerprint(
        stored_library
    ) != _request_fingerprint(dashboard_library):
        raise TeacherAgentDashboardError(
            "durable session full Skill library does not match this dashboard"
        )


def _record_from_store(value: Mapping[str, Any]) -> _DashboardSessionRecord:
    """Validate one integrity-bound dashboard record before exposing it."""

    if value.get("schema") != (
        "teaching_skill_miner.teacher_agent_dashboard_record.v1"
    ):
        raise TeacherAgentDashboardError(
            "durable session record schema is unsupported"
        )
    session = _stored_mapping(value.get("session"), field_name="session")
    validate_session(session)
    profile_revision = value.get("profile_revision")
    profile_display_name = value.get("profile_display_name")
    context_version = value.get("context_version")
    if (
        not isinstance(profile_revision, str)
        or not profile_revision
        or profile_revision != profile_revision.strip()
        or len(profile_revision) > 120
    ):
        raise TeacherAgentDashboardError(
            "durable session profile_revision is invalid"
        )
    if (
        not isinstance(profile_display_name, str)
        or not profile_display_name
        or profile_display_name != profile_display_name.strip()
        or len(profile_display_name) > 80
    ):
        raise TeacherAgentDashboardError(
            "durable session profile_display_name is invalid"
        )
    if (
        isinstance(context_version, bool)
        or not isinstance(context_version, int)
        or context_version < 1
    ):
        raise TeacherAgentDashboardError(
            "durable session context_version is invalid"
        )
    pending_skill_id = value.get("pending_skill_id")
    control_notice = value.get("control_notice")
    if pending_skill_id is not None and (
        not isinstance(pending_skill_id, str) or not pending_skill_id
    ):
        raise TeacherAgentDashboardError(
            "durable session pending_skill_id is invalid"
        )
    if control_notice is not None and (
        not isinstance(control_notice, str) or not control_notice
    ):
        raise TeacherAgentDashboardError(
            "durable session control_notice is invalid"
        )
    return _DashboardSessionRecord(
        session=session,
        profile_revision=profile_revision,
        profile_display_name=profile_display_name,
        pending_skill_id=pending_skill_id,
        control_notice=control_notice,
        context_version=context_version,
        step_idempotency_cache=_stored_mapping(
            value.get("step_idempotency_cache", {}),
            field_name="step_idempotency_cache",
        ),
        command_idempotency_cache=_stored_mapping(
            value.get("command_idempotency_cache", {}),
            field_name="command_idempotency_cache",
        ),
        attachment_idempotency_cache=_stored_mapping(
            value.get("attachment_idempotency_cache", {}),
            field_name="attachment_idempotency_cache",
        ),
        recovered_aborted_turns=_stored_mapping(
            value.get("recovered_aborted_turns", {}),
            field_name="recovered_aborted_turns",
        ),
        attachments=_stored_mapping(
            value.get("attachments", {}), field_name="attachments"
        ),
    )


def _clone_record(record: _DashboardSessionRecord) -> _DashboardSessionRecord:
    return _record_from_store(_record_store_value(record))


def _install_record_state(
    target: _DashboardSessionRecord, candidate: _DashboardSessionRecord
) -> None:
    """Publish a fully flushed candidate without replacing its held lock."""

    target.session = candidate.session
    target.profile_revision = candidate.profile_revision
    target.profile_display_name = candidate.profile_display_name
    target.pending_skill_id = candidate.pending_skill_id
    target.control_notice = candidate.control_notice
    target.context_version = candidate.context_version
    target.step_idempotency_cache = candidate.step_idempotency_cache
    target.command_idempotency_cache = candidate.command_idempotency_cache
    target.attachment_idempotency_cache = candidate.attachment_idempotency_cache
    target.recovered_aborted_turns = candidate.recovered_aborted_turns
    target.attachments = candidate.attachments


@dataclass(slots=True)
class TeacherAgentDashboardSnapshot:
    """Validated public fixtures plus isolated in-memory local sessions."""

    library: dict[str, Any]
    demo_input: dict[str, Any]
    evaluation: dict[str, Any]
    client: DeepSeekClient | None = None
    live_options: LiveAgentOptions = field(default_factory=LiveAgentOptions)
    neural_v1: dict[str, Any] = field(default_factory=dict)
    learning_outcome: dict[str, Any] = field(default_factory=dict)
    free_text_benchmark: dict[str, Any] = field(default_factory=dict)
    vision_extractor: Callable[..., dict[str, Any]] = field(
        default=extract_local_visual_evidence, repr=False
    )
    store: TeacherAgentStore | None = field(default=None, repr=False)
    session: dict[str, Any] | None = None
    session_id: str | None = None
    pending_skill_id: str | None = None
    start_idempotency_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    step_idempotency_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    sessions: dict[str, _DashboardSessionRecord] = field(
        default_factory=dict, repr=False
    )
    start_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _event_specification(
        self,
        event_type: str,
        session_id: str,
        record: _DashboardSessionRecord,
        *,
        idempotency_key: str | None,
        request_fingerprint: str | None,
        turn_id: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "event_type": event_type,
            "session_id": session_id,
            "round": int(record.session.get("round", 0)),
            "question_id": _record_question_id(record),
            "context_version": record.context_version,
            "profile_revision": record.profile_revision,
            "idempotency_key": idempotency_key,
            "request_fingerprint": request_fingerprint,
            "turn_id": turn_id,
            "data": deepcopy(dict(data or {})),
        }

    def _checkpoint_specification(
        self,
        session_id: str,
        record: _DashboardSessionRecord,
        *,
        idempotency_key: str | None,
        request_fingerprint: str | None,
        turn_id: str | None = None,
        reason: str,
    ) -> dict[str, Any]:
        return self._event_specification(
            "context_checkpoint",
            session_id,
            record,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            turn_id=turn_id,
            data={"reason": reason, "record": _record_store_value(record)},
        )

    def _persist_events(self, events: list[dict[str, Any]]) -> None:
        if self.store is None or not events:
            return
        try:
            self.store.append_batch(events)
        except TeacherAgentStoreError as exc:
            raise TeacherAgentDashboardError(
                "durable teacher Agent session store write failed"
            ) from exc

    def _restore_from_store(self) -> None:
        """Install verified cold state and receipt any interrupted turn."""

        if self.store is None:
            return
        try:
            recovery = self.store.recover()
        except TeacherAgentStoreError as exc:
            raise TeacherAgentDashboardError(
                "durable teacher Agent session store recovery failed"
            ) from exc
        if len(recovery.session_records) > _MAX_ACTIVE_SESSIONS:
            raise TeacherAgentDashboardError(
                "durable session store exceeds active session capacity"
            )
        restored: dict[str, _DashboardSessionRecord] = {}
        expected_live = self.client is not None
        for session_id, value in recovery.session_records.items():
            record = _record_from_store(value)
            stored_live = record.session.get("artifact_kind") == (
                "real_time_deepseek_teaching_agent_session"
            )
            if stored_live != expected_live:
                raise TeacherAgentDashboardError(
                    "durable session backend does not match this dashboard"
                )
            _validate_durable_skill_library(
                record.session["skill_library"],
                self.library,
                allow_primary_subset=stored_live,
            )
            if stored_live:
                assert self.client is not None
                try:
                    validate_live_runtime_policy_contract(
                        record.session,
                        self.client,
                        self.live_options,
                    )
                except LiveTeacherAgentError as exc:
                    raise TeacherAgentDashboardError(
                        "durable session runtime policy does not match this dashboard"
                    ) from exc
            if (
                len(record.step_idempotency_cache)
                > _MAX_STEP_IDEMPOTENCY_ENTRIES
                or len(record.command_idempotency_cache)
                > _MAX_COMMAND_IDEMPOTENCY_ENTRIES
                or len(record.attachment_idempotency_cache)
                > _MAX_ATTACHMENT_IDEMPOTENCY_ENTRIES
                or len(record.recovered_aborted_turns)
                > _MAX_STEP_IDEMPOTENCY_ENTRIES
            ):
                raise TeacherAgentDashboardError(
                    "durable session idempotency cache exceeds its bound"
                )
            restored[session_id] = record

        abort_events: list[dict[str, Any]] = []
        checkpoint_sessions: set[str] = set()
        for interrupted in recovery.dangling_turns:
            session_id = str(interrupted["session_id"])
            idempotency_key = interrupted.get("idempotency_key")
            request_fingerprint = interrupted.get("request_fingerprint")
            turn_id = interrupted.get("turn_id")
            record = restored.get(session_id)
            if (
                record is not None
                and isinstance(idempotency_key, str)
                and isinstance(request_fingerprint, str)
            ):
                record.recovered_aborted_turns.pop(idempotency_key, None)
                record.recovered_aborted_turns[idempotency_key] = {
                    "request_fingerprint": request_fingerprint,
                    "turn_id": turn_id,
                    "reason": "process_restarted_before_turn_commit",
                }
                while (
                    len(record.recovered_aborted_turns)
                    > _MAX_STEP_IDEMPOTENCY_ENTRIES
                ):
                    oldest_key = next(iter(record.recovered_aborted_turns))
                    del record.recovered_aborted_turns[oldest_key]
                checkpoint_sessions.add(session_id)
                abort_record = record
            else:
                abort_record = _DashboardSessionRecord(
                    session={"round": int(interrupted["round"])},
                    profile_revision=str(interrupted["profile_revision"]),
                    profile_display_name="recovered interrupted session",
                    context_version=int(interrupted["context_version"]),
                )
            abort_event = self._event_specification(
                "turn_aborted",
                session_id,
                abort_record,
                idempotency_key=(
                    idempotency_key if isinstance(idempotency_key, str) else None
                ),
                request_fingerprint=(
                    request_fingerprint
                    if isinstance(request_fingerprint, str)
                    else None
                ),
                turn_id=turn_id if isinstance(turn_id, str) else None,
                data={
                    "reason": "process_restarted_before_turn_commit",
                    "recovered": True,
                },
            )
            for field_name in (
                "round",
                "question_id",
                "context_version",
                "profile_revision",
            ):
                abort_event[field_name] = interrupted[field_name]
            abort_events.append(abort_event)
        for session_id in sorted(checkpoint_sessions):
            record = restored[session_id]
            abort_events.append(
                self._checkpoint_specification(
                    session_id,
                    record,
                    idempotency_key=None,
                    request_fingerprint=None,
                    reason="recovered_interrupted_turns",
                )
            )
        self._persist_events(abort_events)

        start_cache: dict[str, dict[str, Any]] = {}
        for key, entry in recovery.start_idempotency_cache.items():
            if (
                not isinstance(key, str)
                or not isinstance(entry, Mapping)
                or not isinstance(entry.get("request_fingerprint"), str)
                or not isinstance(entry.get("session_id"), str)
                or not isinstance(entry.get("response"), Mapping)
            ):
                raise TeacherAgentDashboardError(
                    "durable start idempotency cache is invalid"
                )
            start_cache[key] = deepcopy(dict(entry))
            while len(start_cache) > _MAX_START_IDEMPOTENCY_ENTRIES:
                oldest_key = next(iter(start_cache))
                del start_cache[oldest_key]
        with self.lock:
            self.sessions = restored
            self.start_idempotency_cache = start_cache
            touched = recovery.last_touched_session_id
            if touched in restored:
                self._touch_aliases(touched, restored[touched])

    def _touch_aliases(self, session_id: str, record: _DashboardSessionRecord) -> None:
        """Keep legacy inspection attributes pointed at the last-touched session."""

        self.session_id = session_id
        self.session = record.session
        self.pending_skill_id = record.pending_skill_id
        self.step_idempotency_cache = record.step_idempotency_cache

    def _response(
        self, session_id: str, record: _DashboardSessionRecord
    ) -> dict[str, Any]:
        view = (
            live_session_view(record.session)
            if self.client is not None
            else _session_view(record.session)
        )
        profile = record.session.get("student_profile", {})
        goal = record.session.get("goal", {})
        question_id = _record_question_id(record)
        setup_misconceptions = []
        for item in profile.get("known_misconceptions", []):
            if not isinstance(item, Mapping):
                continue
            setup_misconceptions.append(
                {
                    "tag": item.get("tag"),
                    "description": item.get("description"),
                    "confidence": item.get("confidence"),
                }
            )
        return {
            **view,
            "session_id": session_id,
            "context_version": record.context_version,
            "expected_question_id": question_id,
            "pending_skill_id": record.pending_skill_id,
            "pending_skill_effective_from": (
                "next_learner_response" if record.pending_skill_id else None
            ),
            "control_notice": record.control_notice,
            "turn_runtime": {
                "active": record.active_turn_id is not None,
                "turn_id": record.active_turn_id,
                "cancellation_requested": record.active_turn_cancelled,
                "cancel_reason": record.active_turn_cancel_reason,
                "session_replacement_in_progress": record.retiring,
                "transport_cancellation_supported": False,
                "late_response_commit_fenced": True,
            },
            "profile_summary": {
                "profile_ref": profile.get("profile_ref"),
                "profile_revision": record.profile_revision,
                "display_name": record.profile_display_name,
                "learner_level": profile.get("learner_level"),
                "preferences": deepcopy(profile.get("preferences", [])),
                "initial_mastery": deepcopy(profile.get("initial_mastery", {})),
            },
            "setup_snapshot": {
                "goal": {
                    "concept": goal.get("concept"),
                    "objective": goal.get("objective"),
                    "knowledge_components": deepcopy(
                        goal.get("knowledge_components", [])
                    ),
                    "knowledge_spec": deepcopy(goal.get("knowledge_spec", {})),
                    "success_thresholds": deepcopy(goal.get("success_thresholds", {})),
                    "max_rounds": goal.get("max_rounds"),
                    "materials": deepcopy(goal.get("materials", {})),
                },
                "student_profile": {
                    "profile_ref": profile.get("profile_ref"),
                    "learner_level": profile.get("learner_level"),
                    "preferences": deepcopy(profile.get("preferences", [])),
                    "initial_mastery": deepcopy(profile.get("initial_mastery", {})),
                    "known_misconceptions": deepcopy(setup_misconceptions),
                    "background_history": deepcopy(
                        profile.get("background_history", [])
                    ),
                    "conversation_history": deepcopy(
                        profile.get("conversation_history", [])
                    ),
                    "accessibility_needs": deepcopy(
                        profile.get("accessibility_needs", [])
                    ),
                    "contains_direct_identity": (
                        profile.get("contains_direct_identity", False) is True
                    ),
                },
            },
        }

    def _reserve_start_capacity(
        self, *, replacement_id: str | None
    ) -> list[tuple[str, _DashboardSessionRecord]]:
        """Lock idle eviction candidates before any remote start work begins.

        ``start_lock`` serializes registry growth, so holding these record locks
        reserves enough capacity until the new session is either committed or
        abandoned.  A replacement reuses its target's slot and therefore does
        not evict an additional session at the normal capacity limit.
        """

        with self.lock:
            projected_count = len(self.sessions) + 1
            if replacement_id is not None:
                projected_count -= 1
            required_prunes = max(0, projected_count - _MAX_ACTIVE_SESSIONS)
            candidates: list[tuple[str, _DashboardSessionRecord]] = []
            for candidate_id, candidate_record in self.sessions.items():
                if len(candidates) == required_prunes:
                    break
                if candidate_id == replacement_id:
                    continue
                if not candidate_record.lock.acquire(blocking=False):
                    continue
                candidates.append((candidate_id, candidate_record))
            if len(candidates) == required_prunes:
                return candidates
            for _candidate_id, candidate_record in candidates:
                candidate_record.lock.release()
        raise TeacherAgentDashboardError(
            "active session capacity is busy; retry after an in-flight turn completes"
        )

    def bootstrap(self) -> dict[str, Any]:
        return {
            "schema_version": "1.1",
            "dashboard_kind": "loopback_interactive_teacher_agent",
            "mode": (
                "local_durable_opt_in_session"
                if self.store is not None
                else "local_ephemeral_session"
            ),
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
            "agent_runtime_policy": {
                "agent_loop_enabled": bool(self.live_options.agent_loop_enabled),
                "maximum_agent_steps": self.live_options.maximum_agent_steps,
                "maximum_agent_tool_calls_per_step": (
                    self.live_options.maximum_agent_tool_calls_per_step
                ),
                "recoverable_context": self.store is not None,
                "structured_tool_allowlist": True,
            },
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
                "independent_session_registry": True,
                "active_session_replacement_requires_session_id": False,
                "explicit_replacement_is_transactional": True,
                "validated_safety_fallback_can_commit_replacement": True,
                "replacement_requires_expected_round": True,
                "replacement_requires_question_id": True,
                "replacement_requires_context_version": True,
                "replacement_requires_profile_revision": True,
                "session_resume_supported": True,
                "cold_resume_enabled": self.store is not None,
                "cold_resume_requires_explicit_local_store_path": True,
                "step_requires_session_id": True,
                "step_requires_expected_round": True,
                "step_requires_idempotency_key": True,
                "step_requires_question_id": True,
                "step_requires_context_version": True,
                "step_requires_profile_revision": True,
                "single_active_turn_per_session": True,
                "active_turn_commit_fencing": True,
                "replacement_preempts_active_turn": True,
                "stop_preempts_active_turn": True,
                "cancel_generation_preempts_active_turn_without_ending_session": True,
                "remote_transport_cancellation_supported": False,
                "learner_image_attachment_enabled": True,
                "attachment_requires_session_guards": True,
                "attachment_confirmation_gate_when_flagged": True,
                "attachment_confirmation_methods": [
                    "confirmed_attachment_ids",
                    "non_empty_learner_response",
                ],
                "confirmed_ocr_is_student_attested_transcription_not_answer_correctness": True,
                "attachment_raw_media_is_ephemeral": True,
                "attachment_remote_representation": "bounded_redacted_ocr_text_only",
                "deepseek_raw_image_support": False,
                "attachment_max_bytes": MAX_IMAGE_BYTES,
                "attachment_max_per_turn": 2,
                "command_requires_session_id": True,
                "command_requires_expected_round": True,
                "command_requires_idempotency_key": True,
                "command_requires_question_id": True,
                "command_requires_context_version": True,
                "command_requires_profile_revision": True,
                "manual_skill_lock_effective_from_next_learner_response": True,
                "remote_processing_acknowledgement_required": self.client is not None,
                "session_content_persisted_to_browser": False,
                "opaque_session_handle_persisted_to_browser": True,
                "current_prompt_version": LIVE_PROMPT_VERSION,
            },
        }

    def start(self, body: Mapping[str, Any]) -> dict[str, Any]:
        with self.start_lock:
            if not isinstance(body.get("goal"), Mapping) or not isinstance(
                body.get("student_profile"), Mapping
            ):
                raise TeacherAgentDashboardError(
                    "goal and student_profile must be JSON objects"
                )
            if "allowed_skill_ids" in body and not isinstance(
                body.get("allowed_skill_ids"), list
            ):
                raise TeacherAgentDashboardError(
                    "allowed_skill_ids must be a list when provided"
                )
            manual_skill_id = None
            if body.get("manual_skill_id") is not None:
                if self.client is None:
                    raise TeacherAgentDashboardError(
                        "manual_skill_id requires a live DeepSeek session"
                    )
                raw_manual_skill_id = _required_request_string(body, "manual_skill_id")
                try:
                    parsed_manual = parse_skill_command(
                        f"/+skill {raw_manual_skill_id}", self.library
                    )
                except (LiveTeacherAgentError, KeyError, TypeError) as exc:
                    raise TeacherAgentDashboardError(
                        "manual_skill_id must name one available primary Skill"
                    ) from exc
                manual_skill_id = str(parsed_manual["skill_id"])
                if (
                    isinstance(body.get("allowed_skill_ids"), list)
                    and manual_skill_id not in body["allowed_skill_ids"]
                ):
                    raise TeacherAgentDashboardError(
                        "manual_skill_id must be included in allowed_skill_ids"
                    )
            idempotency_key = _required_request_string(body, "start_idempotency_key")
            request_fingerprint = _request_fingerprint(body)
            with self.lock:
                cached = self.start_idempotency_cache.get(idempotency_key)
                if cached is not None:
                    if cached["request_fingerprint"] != request_fingerprint:
                        raise TeacherAgentDashboardError(
                            "start_idempotency_key was already used for a different request"
                        )
                    cached_session_id = cached["session_id"]
                    cached_record = self.sessions.get(cached_session_id)
                    if cached_record is None:
                        raise TeacherAgentDashboardError(
                            "start_idempotency_key belongs to an inactive session"
                        )
                    self._touch_aliases(cached_session_id, cached_record)
                    return deepcopy(cached["response"])

                replacement_id = body.get("replace_session_id")
                replacement_record = None
                if replacement_id is not None:
                    replacement_id = _required_request_string(
                        body, "replace_session_id"
                    )
                    replacement_record = self.sessions.get(replacement_id)
                    if replacement_record is None:
                        raise TeacherAgentDashboardError(
                            "replace_session_id does not match an available session"
                        )
            profile_revision = str(body.get("profile_revision") or "").strip()
            if not profile_revision:
                profile_revision = (
                    "profile-"
                    + _request_fingerprint(
                        {"student_profile": body.get("student_profile", {})}
                    )[:16]
                )
            if len(profile_revision) > 120:
                raise TeacherAgentDashboardError(
                    "profile_revision must be at most 120 characters"
                )
            profile_display_name = str(
                body.get("profile_display_name")
                or body.get("student_profile", {}).get("learner_level", "学生")
            ).strip()
            if not profile_display_name or len(profile_display_name) > 80:
                raise TeacherAgentDashboardError(
                    "profile_display_name must be 1 to 80 characters"
                )

            def build_and_commit(
                prune_candidates: list[tuple[str, _DashboardSessionRecord]],
            ) -> dict[str, Any]:
                try:
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
                    new_session_id = secrets.token_urlsafe(16)
                    record = _DashboardSessionRecord(
                        session=new_session,
                        profile_revision=profile_revision,
                        profile_display_name=profile_display_name,
                        pending_skill_id=manual_skill_id,
                    )
                    response = self._response(new_session_id, record)
                    cached_response = deepcopy(response)
                    start_cache_entry = {
                        "request_fingerprint": request_fingerprint,
                        "session_id": new_session_id,
                        "response": cached_response,
                    }
                    durable_events: list[dict[str, Any]] = []
                    if replacement_id is not None and replacement_record is not None:
                        durable_events.append(
                            self._event_specification(
                                "session_stopped",
                                replacement_id,
                                replacement_record,
                                idempotency_key=idempotency_key,
                                request_fingerprint=request_fingerprint,
                                data={
                                    "reason": "explicit_session_replacement",
                                    "remove_session": True,
                                },
                            )
                        )
                    for candidate_id, candidate_record in prune_candidates:
                        durable_events.append(
                            self._event_specification(
                                "session_stopped",
                                candidate_id,
                                candidate_record,
                                idempotency_key=idempotency_key,
                                request_fingerprint=request_fingerprint,
                                data={
                                    "reason": "active_session_capacity_eviction",
                                    "remove_session": True,
                                },
                            )
                        )
                    durable_events.extend(
                        [
                            self._event_specification(
                                "session_started",
                                new_session_id,
                                record,
                                idempotency_key=idempotency_key,
                                request_fingerprint=request_fingerprint,
                                data={
                                    "record": _record_store_value(record),
                                    "start_cache_entry": start_cache_entry,
                                    "backend": (
                                        "deepseek"
                                        if self.client is not None
                                        else "deterministic"
                                    ),
                                },
                            ),
                            self._checkpoint_specification(
                                new_session_id,
                                record,
                                idempotency_key=idempotency_key,
                                request_fingerprint=request_fingerprint,
                                reason="session_started",
                            ),
                        ]
                    )

                    def persist_and_install() -> dict[str, Any]:
                        self._persist_events(durable_events)
                        with self.lock:
                            if replacement_id is not None:
                                if (
                                    self.sessions.get(replacement_id)
                                    is not replacement_record
                                ):
                                    raise TeacherAgentDashboardError(
                                        "replace_session_id changed while the new session was prepared"
                                    )
                                del self.sessions[replacement_id]
                            for candidate_id, candidate_record in prune_candidates:
                                if self.sessions.get(candidate_id) is candidate_record:
                                    del self.sessions[candidate_id]
                            self.sessions[new_session_id] = record
                            self._touch_aliases(new_session_id, record)
                            self.start_idempotency_cache[idempotency_key] = (
                                start_cache_entry
                            )
                            while (
                                len(self.start_idempotency_cache)
                                > _MAX_START_IDEMPOTENCY_ENTRIES
                            ):
                                oldest_key = next(iter(self.start_idempotency_cache))
                                del self.start_idempotency_cache[oldest_key]
                            return response

                    if replacement_id is None or replacement_record is None:
                        return persist_and_install()

                    # Preparing a replacement prevents new turns from entering,
                    # but an already-running turn remains valid until the new
                    # candidate has been fully constructed.  Revalidate the
                    # original replacement fence under the target lock, then
                    # cancel that old turn only as part of the final commit.
                    # This preserves the complete active-turn control state if
                    # candidate construction or validation fails.
                    with replacement_record.lock:
                        with self.lock:
                            if (
                                self.sessions.get(replacement_id)
                                is not replacement_record
                            ):
                                raise TeacherAgentDashboardError(
                                    "replace_session_id changed while the new session was prepared"
                                )
                        _validate_replacement_guards(body, replacement_record)
                        active_turn_control_state = (
                            replacement_record.active_turn_id,
                            replacement_record.active_turn_idempotency_key,
                            replacement_record.active_turn_request_fingerprint,
                            replacement_record.active_turn_generation,
                            replacement_record.active_turn_cancelled,
                            replacement_record.active_turn_cancel_reason,
                        )
                        _cancel_active_turn(
                            replacement_record,
                            reason="explicit_session_replacement",
                        )
                        try:
                            return persist_and_install()
                        except Exception:
                            (
                                replacement_record.active_turn_id,
                                replacement_record.active_turn_idempotency_key,
                                replacement_record.active_turn_request_fingerprint,
                                replacement_record.active_turn_generation,
                                replacement_record.active_turn_cancelled,
                                replacement_record.active_turn_cancel_reason,
                            ) = active_turn_control_state
                            raise
                finally:
                    for _candidate_id, candidate_record in prune_candidates:
                        candidate_record.lock.release()

            if replacement_record is None:
                prune_candidates = self._reserve_start_capacity(replacement_id=None)
                return build_and_commit(prune_candidates)
            with replacement_record.lock:
                with self.lock:
                    if self.sessions.get(replacement_id) is not replacement_record:
                        raise TeacherAgentDashboardError(
                            "replace_session_id is no longer available"
                        )
                if replacement_record.retiring:
                    raise TeacherAgentDashboardError(
                        "replace_session_id is already being replaced"
                    )
                _validate_replacement_guards(body, replacement_record)
                replacement_record.retiring = True
            try:
                prune_candidates = self._reserve_start_capacity(
                    replacement_id=replacement_id
                )
                return build_and_commit(prune_candidates)
            except Exception:
                with replacement_record.lock:
                    with self.lock:
                        still_installed = (
                            self.sessions.get(replacement_id) is replacement_record
                        )
                    if still_installed:
                        replacement_record.retiring = False
                raise

    def resume(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Resume one opaque local session without exposing another session."""

        request_session_id = _required_request_string(body, "session_id")
        with self.lock:
            record = self.sessions.get(request_session_id)
        if record is None:
            raise TeacherAgentDashboardError("session_id is no longer available")
        with record.lock:
            with self.lock:
                if self.sessions.get(request_session_id) is not record:
                    raise TeacherAgentDashboardError(
                        "session_id is no longer available"
                    )
            response = self._response(request_session_id, record)
        with self.lock:
            if self.sessions.get(request_session_id) is record:
                self._touch_aliases(request_session_id, record)
        return response

    def upload_attachment(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Extract local image evidence and retain only its bounded text record."""

        request_session_id = _required_request_string(body, "session_id")
        with self.lock:
            record = self.sessions.get(request_session_id)
        if record is None:
            raise TeacherAgentDashboardError("session_id is no longer available")
        with record.lock:
            with self.lock:
                if self.sessions.get(request_session_id) is not record:
                    raise TeacherAgentDashboardError(
                        "session_id is no longer available"
                    )
            if record.retiring:
                raise TeacherAgentDashboardError(
                    "session replacement is in progress; attachment was not accepted"
                )
            if record.active_turn_id is not None:
                raise TeacherAgentDashboardError(
                    "an active turn is already running; attachment was not accepted"
                )
            if record.session.get("status") != "active":
                raise TeacherAgentDashboardError(
                    "attachments are not allowed for a terminal session"
                )
            idempotency_key = _required_request_string(
                body, "attachment_idempotency_key"
            )
            request_fingerprint = _request_fingerprint(body)
            cached = record.attachment_idempotency_cache.get(idempotency_key)
            if cached is not None:
                if cached["request_fingerprint"] != request_fingerprint:
                    raise TeacherAgentDashboardError(
                        "attachment_idempotency_key was already used for a different request"
                    )
                return deepcopy(cached["response"])
            expected_round, expected_question_id, profile_revision = (
                _validate_session_turn_guards(body, record)
            )
            pending_count = sum(
                item.get("consumed") is not True and item.get("expired") is not True
                for item in record.attachments.values()
            )
            if pending_count >= _MAX_PENDING_ATTACHMENTS:
                raise TeacherAgentDashboardError("too many pending learner attachments")
            mime_type = _required_request_string(body, "mime_type", maximum=40)
            display_name = _required_request_string(body, "display_name", maximum=120)
            encoded = body.get("data_base64")
            if not isinstance(encoded, str) or not encoded:
                raise TeacherAgentDashboardError(
                    "data_base64 must be a non-empty base64 string"
                )
            if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 8:
                raise TeacherAgentDashboardError("learner attachment is too large")
            try:
                image_bytes = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise TeacherAgentDashboardError("data_base64 is invalid") from exc
            try:
                evidence = self.vision_extractor(
                    image_bytes,
                    mime_type,
                    display_name=display_name,
                )
            except (LocalVisualEvidenceError, OSError, TypeError, ValueError) as exc:
                raise TeacherAgentDashboardError(
                    "local learner-image extraction failed"
                ) from exc
            attachment_id = "att_" + secrets.token_urlsafe(12)
            candidate_record = _clone_record(record)
            candidate_record.attachments[attachment_id] = {
                "attachment_id": attachment_id,
                "question_id": expected_question_id,
                "round": expected_round,
                "profile_revision": profile_revision,
                "consumed": False,
                "expired": False,
                "evidence": deepcopy(evidence),
            }
            candidate_record.context_version += 1
            response = {
                "session_id": request_session_id,
                "context_version": candidate_record.context_version,
                "expected_question_id": expected_question_id,
                "profile_revision": profile_revision,
                "attachment": {
                    "attachment_id": attachment_id,
                    **deepcopy(evidence),
                },
            }
            candidate_record.attachment_idempotency_cache[idempotency_key] = {
                "request_fingerprint": request_fingerprint,
                "response": deepcopy(response),
            }
            while (
                len(candidate_record.attachment_idempotency_cache)
                > _MAX_ATTACHMENT_IDEMPOTENCY_ENTRIES
            ):
                oldest_key = next(
                    iter(candidate_record.attachment_idempotency_cache)
                )
                del candidate_record.attachment_idempotency_cache[oldest_key]
            self._persist_events(
                [
                    self._checkpoint_specification(
                        request_session_id,
                        candidate_record,
                        idempotency_key=idempotency_key,
                        request_fingerprint=request_fingerprint,
                        reason="attachment_uploaded",
                    )
                ]
            )
            _install_record_state(record, candidate_record)
        with self.lock:
            if self.sessions.get(request_session_id) is record:
                self._touch_aliases(request_session_id, record)
        return response

    def step(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request_session_id = _required_request_string(body, "session_id")
        with self.lock:
            record = self.sessions.get(request_session_id)
        if record is None:
            raise TeacherAgentDashboardError("session_id is no longer available")

        turn_was_receipted = False
        turn_id = ""
        turn_generation = -1
        idempotency_key = ""
        request_fingerprint = ""
        base_context_version = -1
        base_profile_revision = ""
        base_question_id: str | None = None
        aborted_persisted = False
        with record.lock:
            with self.lock:
                if self.sessions.get(request_session_id) is not record:
                    raise TeacherAgentDashboardError(
                        "session_id is no longer available"
                    )
            idempotency_key = _required_request_string(body, "idempotency_key")
            request_fingerprint = _request_fingerprint(body)
            cached = record.step_idempotency_cache.get(idempotency_key)
            if cached is not None:
                if cached["request_fingerprint"] != request_fingerprint:
                    raise TeacherAgentDashboardError(
                        "idempotency_key was already used for a different request"
                    )
                return deepcopy(cached["response"])
            recovered_abort = record.recovered_aborted_turns.get(idempotency_key)
            if recovered_abort is not None:
                if recovered_abort.get("request_fingerprint") != request_fingerprint:
                    raise TeacherAgentDashboardError(
                        "idempotency_key was already used for a different request"
                    )
                raise TeacherAgentDashboardError(
                    "this idempotency_key belongs to a turn aborted during cold "
                    "recovery; submit the answer with a new idempotency_key"
                )
            if record.session.get("status") != "active":
                raise TeacherAgentDashboardError(
                    "steps are not allowed for a terminal session"
                )
            _validate_session_turn_guards(body, record)
            attachment_ids = body.get("attachment_ids", [])
            if not isinstance(attachment_ids, list) or len(attachment_ids) > 2:
                raise TeacherAgentDashboardError(
                    "attachment_ids must be a list with at most two items"
                )
            if any(
                not isinstance(item, str) or not item or item.strip() != item
                for item in attachment_ids
            ) or len(attachment_ids) != len(set(attachment_ids)):
                raise TeacherAgentDashboardError(
                    "attachment_ids must contain unique non-empty strings"
                )
            confirmed_attachment_ids = body.get("confirmed_attachment_ids", [])
            if (
                not isinstance(confirmed_attachment_ids, list)
                or len(confirmed_attachment_ids) > len(attachment_ids)
                or any(
                    not isinstance(item, str) or not item or item.strip() != item
                    for item in confirmed_attachment_ids
                )
                or len(confirmed_attachment_ids) != len(set(confirmed_attachment_ids))
                or not set(confirmed_attachment_ids).issubset(attachment_ids)
            ):
                raise TeacherAgentDashboardError(
                    "confirmed_attachment_ids must be unique attachment_ids from this request"
                )
            learner_response = str(body.get("learner_response", "")).strip()
            if not learner_response and not attachment_ids:
                raise TeacherAgentDashboardError(
                    "learner_response or one learner attachment is required"
                )
            learner_evidence: list[dict[str, Any]] = []
            current_question_id = _record_question_id(record)
            for attachment_id in attachment_ids:
                attachment = record.attachments.get(attachment_id)
                if (
                    attachment is None
                    or attachment.get("consumed") is True
                    or attachment.get("expired") is True
                ):
                    raise TeacherAgentDashboardError(
                        "attachment_id is unavailable or already consumed"
                    )
                if (
                    attachment.get("question_id") != current_question_id
                    or attachment.get("round") != record.session.get("round")
                    or attachment.get("profile_revision") != record.profile_revision
                ):
                    raise TeacherAgentDashboardError(
                        "attachment_id does not belong to the active turn"
                    )
                evidence = attachment.get("evidence", {})
                needs_confirmation = isinstance(evidence, Mapping) and bool(
                    evidence.get("needs_student_confirmation")
                )
                if (
                    needs_confirmation
                    and attachment_id not in confirmed_attachment_ids
                    and not learner_response
                ):
                    raise TeacherAgentDashboardError(
                        "this OCR result needs student confirmation; confirm the recognized text or enter a corrected learner_response before sending"
                    )
                evidence_record = deepcopy(attachment["evidence"])
                if attachment_id in confirmed_attachment_ids:
                    recognized_text = str(
                        evidence_record.get("recognized_text", "")
                    ).strip()
                    if not recognized_text:
                        raise TeacherAgentDashboardError(
                            "an OCR result with no recognized text cannot be confirmed; enter a corrected learner_response instead"
                        )
                    evidence_record["student_confirmed_recognized_text"] = True
                    evidence_record["ocr_confirmation_was_required"] = bool(
                        needs_confirmation
                    )
                    evidence_record["needs_student_confirmation"] = False
                    evidence_record["student_confirmation_method"] = (
                        "confirmed_attachment_ids"
                    )
                    evidence_record[
                        "student_confirmation_establishes_answer_correctness"
                    ] = False
                else:
                    evidence_record["student_confirmed_recognized_text"] = False
                    evidence_record["ocr_confirmation_was_required"] = bool(
                        needs_confirmation
                    )
                    evidence_record["student_confirmation_method"] = None
                    evidence_record[
                        "student_confirmation_establishes_answer_correctness"
                    ] = False
                learner_evidence.append(evidence_record)
            body_manual_skill = (
                str(body["manual_skill_id"])
                if self.client is not None and body.get("manual_skill_id")
                else None
            )
            if (
                body_manual_skill
                and record.pending_skill_id
                and body_manual_skill != record.pending_skill_id
            ):
                raise TeacherAgentDashboardError(
                    "manual_skill_id does not match the active Skill lock; "
                    "use the command endpoint to change it"
                )
            turn_id = sha256(
                (
                    request_session_id
                    + "\x00"
                    + idempotency_key
                    + "\x00"
                    + request_fingerprint
                ).encode("utf-8")
            ).hexdigest()
            turn_generation = _begin_active_turn(
                record,
                turn_id=turn_id,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
            )
            base_context_version = record.context_version
            base_profile_revision = record.profile_revision
            base_question_id = _record_question_id(record)
            turn_started = self._event_specification(
                "turn_started",
                request_session_id,
                record,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                turn_id=turn_id,
                data={
                    "attachment_ids": deepcopy(attachment_ids),
                    "confirmed_attachment_ids": deepcopy(confirmed_attachment_ids),
                    "remote_model_call_may_follow": self.client is not None,
                },
            )
            try:
                self._persist_events([turn_started])
                turn_was_receipted = self.store is not None
                candidate_record = _clone_record(record)
            except Exception as exc:
                if turn_was_receipted:
                    try:
                        self._persist_events(
                            [
                                self._event_specification(
                                    "turn_aborted",
                                    request_session_id,
                                    record,
                                    idempotency_key=idempotency_key,
                                    request_fingerprint=request_fingerprint,
                                    turn_id=turn_id,
                                    data={
                                        "reason": "turn_failed_before_commit",
                                        "error_type": type(exc).__name__,
                                        "recovered": False,
                                    },
                                )
                            ]
                        )
                        aborted_persisted = True
                    except TeacherAgentDashboardError as abort_exc:
                        _clear_active_turn(record, turn_id=turn_id)
                        raise abort_exc from exc
                _clear_active_turn(record, turn_id=turn_id)
                raise

        # The potentially slow remote request deliberately runs without the
        # per-session lock.  Stop/replacement can therefore invalidate this
        # generation; the candidate is published only after the commit fence
        # below revalidates every session identity guard.
        try:
            candidate_pending_skill_id = candidate_record.pending_skill_id
            candidate_control_notice: str | None = None
            if self.client is not None:
                requested_skill = body_manual_skill or candidate_record.pending_skill_id
                persistent_skill = candidate_record.pending_skill_id
                prior_fallback_count = int(
                    candidate_record.session.get("agent_runtime", {}).get(
                        "fallback_count", 0
                    )
                )
                candidate_session = advance_live_teacher_agent_session(
                    candidate_record.session,
                    learner_response=learner_response,
                    client=self.client,
                    learner_evidence=learner_evidence,
                    manual_skill_id=requested_skill,
                    options=self.live_options,
                )
                current_fallback_count = int(
                    candidate_session.get("agent_runtime", {}).get(
                        "fallback_count", 0
                    )
                )
                action = candidate_session.get("current_action", {})
                manual_applied = (
                    action.get("manual_override_applied")
                    if isinstance(action, Mapping)
                    else None
                )
                if persistent_skill and current_fallback_count > prior_fallback_count:
                    candidate_pending_skill_id = None
                    candidate_control_notice = (
                        "manual_skill_released_after_safety_fallback"
                    )
                elif persistent_skill and manual_applied is False:
                    candidate_pending_skill_id = None
                    candidate_control_notice = (
                        "manual_skill_released_by_skill_contract_guard"
                    )
            else:
                evidence_text = "\n".join(
                    str(item.get("recognized_text", ""))
                    for item in learner_evidence
                    if item.get("recognized_text")
                ).strip()
                candidate_session = advance_teacher_agent_session(
                    candidate_record.session,
                    learner_response=(
                        learner_response
                        or evidence_text
                        or "（学生提交了图片，但本机未识别出可靠文字）"
                    ),
                    signal=str(body.get("signal", "confused")),
                    misconception_tag=(
                        str(body["misconception_tag"])
                        if body.get("misconception_tag")
                        else None
                    ),
                    signal_confidence=float(body.get("signal_confidence", 1.0)),
                )
                if candidate_session.get("history"):
                    event = candidate_session["history"][-1]
                    event["learner_text"] = learner_response
                    event["multimodal_evidence"] = deepcopy(learner_evidence)
                    _refresh_integrity(candidate_session)

            candidate_record.session = candidate_session
            candidate_record.pending_skill_id = candidate_pending_skill_id
            candidate_record.control_notice = candidate_control_notice
            candidate_record.context_version += 1
            for attachment in candidate_record.attachments.values():
                if attachment.get("question_id") == current_question_id:
                    attachment["consumed"] = (
                        attachment.get("attachment_id") in attachment_ids
                    )
                    attachment["expired"] = (
                        attachment.get("attachment_id") not in attachment_ids
                    )
            response = self._response(request_session_id, candidate_record)
            cached_response = deepcopy(response)
            candidate_record.step_idempotency_cache[idempotency_key] = {
                "request_fingerprint": request_fingerprint,
                "response": cached_response,
            }
            while (
                len(candidate_record.step_idempotency_cache)
                > _MAX_STEP_IDEMPOTENCY_ENTRIES
            ):
                oldest_key = next(iter(candidate_record.step_idempotency_cache))
                del candidate_record.step_idempotency_cache[oldest_key]
            durable_events = [
                self._event_specification(
                    "turn_committed",
                    request_session_id,
                    candidate_record,
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
                    turn_id=turn_id,
                    data={
                        "record": _record_store_value(candidate_record),
                        "response": cached_response,
                    },
                )
            ]
            if candidate_record.session.get("status") != "active":
                durable_events.append(
                    self._event_specification(
                        "session_stopped",
                        request_session_id,
                        candidate_record,
                        idempotency_key=idempotency_key,
                        request_fingerprint=request_fingerprint,
                        turn_id=turn_id,
                        data={
                            "reason": candidate_record.session.get(
                                "termination_reason", "terminal_turn"
                            ),
                            "remove_session": False,
                            "record": _record_store_value(candidate_record),
                        },
                    )
                )
            durable_events.append(
                self._checkpoint_specification(
                    request_session_id,
                    candidate_record,
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
                    turn_id=turn_id,
                    reason="turn_committed",
                )
            )

            with record.lock:
                with self.lock:
                    still_installed = self.sessions.get(request_session_id) is record
                cancellation_reason = record.active_turn_cancel_reason
                commit_allowed = (
                    still_installed
                    and record.active_turn_id == turn_id
                    and record.active_turn_generation == turn_generation
                    and not record.active_turn_cancelled
                    and record.context_version == base_context_version
                    and record.profile_revision == base_profile_revision
                    and _record_question_id(record) == base_question_id
                )
                if not commit_allowed:
                    reason = cancellation_reason or (
                        "session_replaced_during_turn"
                        if not still_installed
                        else "active_turn_commit_guard_changed"
                    )
                    if turn_was_receipted:
                        self._persist_events(
                            [
                                self._event_specification(
                                    "turn_aborted",
                                    request_session_id,
                                    record,
                                    idempotency_key=idempotency_key,
                                    request_fingerprint=request_fingerprint,
                                    turn_id=turn_id,
                                    data={
                                        "reason": reason,
                                        "error_type": "TurnCommitCancelled",
                                        "recovered": False,
                                    },
                                )
                            ]
                        )
                        aborted_persisted = True
                    _clear_active_turn(record, turn_id=turn_id)
                    raise TeacherAgentDashboardError(
                        f"active turn was cancelled before commit: {reason}"
                    )
                self._persist_events(durable_events)
                _install_record_state(record, candidate_record)
                _clear_active_turn(record, turn_id=turn_id)
        except Exception as exc:
            with record.lock:
                if turn_was_receipted and not aborted_persisted:
                    try:
                        self._persist_events(
                            [
                                self._event_specification(
                                    "turn_aborted",
                                    request_session_id,
                                    record,
                                    idempotency_key=idempotency_key,
                                    request_fingerprint=request_fingerprint,
                                    turn_id=turn_id,
                                    data={
                                        "reason": "turn_failed_before_commit",
                                        "error_type": type(exc).__name__,
                                        "recovered": False,
                                    },
                                )
                            ]
                        )
                    except TeacherAgentDashboardError as abort_exc:
                        _clear_active_turn(record, turn_id=turn_id)
                        raise abort_exc from exc
                _clear_active_turn(record, turn_id=turn_id)
            raise
        with self.lock:
            if self.sessions.get(request_session_id) is record:
                self._touch_aliases(request_session_id, record)
        return response

    def command(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request_session_id = _required_request_string(body, "session_id")
        with self.lock:
            record = self.sessions.get(request_session_id)
        if record is None:
            raise TeacherAgentDashboardError("session_id is no longer available")
        with record.lock:
            with self.lock:
                if self.sessions.get(request_session_id) is not record:
                    raise TeacherAgentDashboardError(
                        "session_id is no longer available"
                    )
            idempotency_key = _required_request_string(body, "command_idempotency_key")
            request_fingerprint = _request_fingerprint(body)
            cached = record.command_idempotency_cache.get(idempotency_key)
            if cached is not None:
                if cached["request_fingerprint"] != request_fingerprint:
                    raise TeacherAgentDashboardError(
                        "command_idempotency_key was already used for a different request"
                    )
                return deepcopy(cached["response"])
            if record.session.get("status") != "active":
                raise TeacherAgentDashboardError(
                    "commands are not allowed for a terminal session"
                )
            expected_round = _request_round(body, required=True)
            if expected_round != record.session["round"]:
                raise TeacherAgentDashboardError(
                    "expected_round does not match this session"
                )
            expected_context_version = body.get("expected_context_version")
            if (
                isinstance(expected_context_version, bool)
                or not isinstance(expected_context_version, int)
                or expected_context_version != record.context_version
            ):
                raise TeacherAgentDashboardError(
                    "expected_context_version does not match this session"
                )
            expected_question_id = _required_request_string(
                body, "expected_question_id"
            )
            current_action = record.session.get("current_action", {})
            current_teacher_action = (
                current_action.get("teacher_action", {})
                if isinstance(current_action, Mapping)
                else {}
            )
            current_question_id = (
                current_teacher_action.get("question_id")
                if isinstance(current_teacher_action, Mapping)
                else None
            ) or (
                current_action.get("action_id")
                if isinstance(current_action, Mapping)
                else None
            )
            if expected_question_id != current_question_id:
                raise TeacherAgentDashboardError(
                    "expected_question_id does not match this session"
                )
            request_profile_revision = _required_request_string(
                body, "profile_revision", maximum=120
            )
            if request_profile_revision != record.profile_revision:
                raise TeacherAgentDashboardError(
                    "profile_revision does not match this session"
                )
            command = str(body.get("command", "")).strip()
            if record.retiring:
                raise TeacherAgentDashboardError(
                    "session replacement is in progress; commands are unavailable"
                )
            if record.active_turn_id is not None and command not in {
                "stop",
                "cancel_turn",
            }:
                raise TeacherAgentDashboardError(
                    "an active turn is running; only stop or cancel_turn may preempt it"
                )
            candidate_record = _clone_record(record)
            candidate_session = candidate_record.session
            candidate_pending_skill_id = candidate_record.pending_skill_id
            candidate_control_notice = candidate_record.control_notice
            if command == "auto":
                candidate_pending_skill_id = None
                candidate_control_notice = None
            elif command == "select_skill":
                skill_id = str(body.get("skill_id", "")).strip()
                parsed = parse_skill_command(
                    f"/+skill {skill_id}", candidate_record.session["skill_library"]
                )
                candidate_pending_skill_id = str(parsed["skill_id"])
                candidate_control_notice = None
            elif command == "stop":
                if self.client is not None:
                    candidate_session = stop_live_teacher_agent_session(
                        candidate_record.session,
                        reason="teacher requested stop from dashboard",
                    )
                else:
                    raise TeacherAgentDashboardError(
                        "manual stop requires a live session"
                    )
            elif command == "cancel_turn":
                if self.client is None:
                    raise TeacherAgentDashboardError(
                        "turn cancellation requires a live session"
                    )
                if record.active_turn_id is None:
                    raise TeacherAgentDashboardError(
                        "cancel_turn requires an active turn"
                    )
            else:
                raise TeacherAgentDashboardError("unsupported Agent command")
            candidate_record.session = candidate_session
            candidate_record.pending_skill_id = candidate_pending_skill_id
            candidate_record.control_notice = candidate_control_notice
            candidate_record.context_version += 1
            if command in {"stop", "cancel_turn"} and record.active_turn_id is not None:
                candidate_record.active_turn_id = record.active_turn_id
                candidate_record.active_turn_idempotency_key = (
                    record.active_turn_idempotency_key
                )
                candidate_record.active_turn_request_fingerprint = (
                    record.active_turn_request_fingerprint
                )
                candidate_record.active_turn_generation = (
                    record.active_turn_generation + 1
                )
                candidate_record.active_turn_cancelled = True
                candidate_record.active_turn_cancel_reason = (
                    "teacher_requested_stop"
                    if command == "stop"
                    else "teacher_requested_turn_cancel"
                )
            response = self._response(request_session_id, candidate_record)
            cached_response = deepcopy(response)
            candidate_record.command_idempotency_cache[idempotency_key] = {
                "request_fingerprint": request_fingerprint,
                "response": cached_response,
            }
            while (
                len(candidate_record.command_idempotency_cache)
                > _MAX_COMMAND_IDEMPOTENCY_ENTRIES
            ):
                oldest_key = next(iter(candidate_record.command_idempotency_cache))
                del candidate_record.command_idempotency_cache[oldest_key]
            durable_events: list[dict[str, Any]] = []
            if command == "stop":
                durable_events.append(
                    self._event_specification(
                        "session_stopped",
                        request_session_id,
                        candidate_record,
                        idempotency_key=idempotency_key,
                        request_fingerprint=request_fingerprint,
                        data={
                            "reason": "teacher_requested_stop",
                            "remove_session": False,
                            "record": _record_store_value(candidate_record),
                        },
                    )
                )
            durable_events.append(
                self._checkpoint_specification(
                    request_session_id,
                    candidate_record,
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
                    reason=f"command_{command}",
                )
            )
            self._persist_events(durable_events)
            if command in {"stop", "cancel_turn"}:
                _cancel_active_turn(
                    record,
                    reason=(
                        "teacher_requested_stop"
                        if command == "stop"
                        else "teacher_requested_turn_cancel"
                    ),
                )
            _install_record_state(record, candidate_record)
        with self.lock:
            if self.sessions.get(request_session_id) is record:
                self._touch_aliases(request_session_id, record)
        return response


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
    vision_extractor: Callable[..., dict[str, Any]] = extract_local_visual_evidence,
    store_path: str | Path | None = None,
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
            raise TeacherAgentDashboardError(
                "neural-v1 runtime manifest must be an object"
            )
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
    store = None
    if store_path is not None:
        try:
            store = TeacherAgentStore(store_path)
        except TeacherAgentStoreError as exc:
            raise TeacherAgentDashboardError(
                "durable teacher Agent session store cannot be opened"
            ) from exc
    snapshot = TeacherAgentDashboardSnapshot(
        library=library,
        demo_input=demo_input,
        evaluation=evaluation,
        client=client,
        live_options=(live_options or LiveAgentOptions()).validated(),
        neural_v1=neural_v1,
        learning_outcome=learning_outcome,
        free_text_benchmark=free_text_benchmark,
        vision_extractor=vision_extractor,
        store=store,
    )
    snapshot._restore_from_store()
    return snapshot


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
        "passed": not missing
        and not forbidden_matches
        and snapshot.evaluation["passed"],
        "missing_markers": missing,
        "forbidden_matches": forbidden_matches,
        "skill_count": len(snapshot.library["skills"]),
        "evaluation_passed": snapshot.evaluation["passed"],
        "one_action_per_turn": True,
        "real_time_skill_switching": True,
        "structured_agent_loop_enabled": bool(
            snapshot.live_options.agent_loop_enabled and snapshot.client is not None
        ),
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
                if (
                    parsed.scheme != "http"
                    or (parsed.hostname or "").casefold() not in _SAFE_HOSTS
                ):
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

        def _read_body(
            self, *, maximum_bytes: int = _MAX_REQUEST_BYTES
        ) -> dict[str, Any]:
            if (
                self.headers.get("Content-Type", "").split(";", 1)[0].strip()
                != "application/json"
            ):
                raise TeacherAgentDashboardError(
                    "Content-Type must be application/json"
                )
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length or "")
            except ValueError as exc:
                raise TeacherAgentDashboardError(
                    "valid Content-Length required"
                ) from exc
            if not 1 <= length <= maximum_bytes:
                raise TeacherAgentDashboardError("request body size is invalid")
            try:
                value = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise TeacherAgentDashboardError(
                    "request body is invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise TeacherAgentDashboardError("request body must be one JSON object")
            return value

        def _get(self, route: str) -> None:
            if route in {"", "index.html"}:
                self._payload(
                    _resource_bytes(HTML_RESOURCE),
                    content_type="text/html; charset=utf-8",
                )
            elif route == "assets/teacher_agent_demo.css":
                self._payload(
                    _resource_bytes(STYLE_RESOURCE),
                    content_type="text/css; charset=utf-8",
                )
            elif route == "assets/teacher_agent_demo.js":
                self._payload(
                    _resource_bytes(SCRIPT_RESOURCE),
                    content_type="text/javascript; charset=utf-8",
                )
            elif (
                route.startswith("assets/")
                and route.removeprefix("assets/") in AVATAR_RESOURCES
            ):
                self._payload(
                    _avatar_bytes(route.removeprefix("assets/")),
                    content_type="image/png",
                )
            elif route == "api/bootstrap":
                self._payload(
                    _json_bytes(snapshot.bootstrap()),
                    content_type="application/json; charset=utf-8",
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, "resource not found")

        def _post(self, route: str) -> None:
            try:
                body = self._read_body(
                    maximum_bytes=(
                        _MAX_ATTACHMENT_REQUEST_BYTES
                        if route == "api/attachment"
                        else _MAX_REQUEST_BYTES
                    )
                )
                if route == "api/start":
                    result = snapshot.start(body)
                elif route == "api/session":
                    result = snapshot.resume(body)
                elif route == "api/attachment":
                    result = snapshot.upload_attachment(body)
                elif route == "api/step":
                    result = snapshot.step(body)
                elif route == "api/command":
                    result = snapshot.command(body)
                else:
                    self._error(HTTPStatus.NOT_FOUND, "resource not found")
                    return
            except (
                TeacherAgentError,
                TeacherAgentDashboardError,
                TypeError,
                ValueError,
            ) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._payload(
                _json_bytes(result), content_type="application/json; charset=utf-8"
            )

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
    store_path: str | Path | None = None,
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
        store_path=store_path,
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
        "durable_session_store_enabled": snapshot.store is not None,
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
