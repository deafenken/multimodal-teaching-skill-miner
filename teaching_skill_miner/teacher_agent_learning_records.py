"""Durable, privacy-bounded spaced-review records for the Teaching Agent.

This module is deliberately independent from the dashboard and live-agent
runtime.  A committed Teaching turn can place one validated outbox event in
its own durable checkpoint and later hand that event to
``LearningRecordStore.apply_outbox_event``.  Event IDs are strictly
idempotent, so replaying that outbox after a crash cannot schedule a review
twice.

The public contract is intentionally narrow:

* learner keys are opaque HMAC-derived server identifiers; anonymous/default
  profiles cannot be minted;
* a target is exactly one ``(learner_key, curriculum_namespace, kc_id)``;
* review outcomes are derived from an authoritative KC-v2 evidence entry, not
  from a client-provided score or mastery delta;
* persisted values contain IDs, hashes, timestamps, and scheduler state only --
  never learner responses, excerpts, notes, or transcripts;
* due-ness and expired leases are projections of an injected UTC clock, while
  every due date and lease boundary remains durable across process restarts.

Adjudication is intentionally out of scope here.  A later adjudication module
may emit the same strict outcome event after it has produced a versioned,
teacher-authoritative evidence entry.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence
import unicodedata

from .io_utils import ensure_private_directory, ensure_private_file

try:  # Durable multi-process mode is intentionally POSIX-only.
    import fcntl
except ImportError:  # pragma: no cover - CI and supported deployments are POSIX.
    fcntl = None  # type: ignore[assignment]


LEARNING_OUTBOX_EVENT_SCHEMA = (
    "teaching_skill_miner.teacher_agent_learning_outbox_event.v1"
)
LEARNING_STORE_EVENT_SCHEMA = (
    "teaching_skill_miner.teacher_agent_learning_store_event.v1"
)
LEARNING_RECORD_SCHEMA = "teaching_skill_miner.learner_learning_record.v1"
LEARNING_ERASURE_TOMBSTONES_SCHEMA = (
    "teaching_skill_miner.teacher_agent_learning_erasure_tombstones.v1"
)

REVIEW_INTERVAL_DAYS = (1, 3, 7, 14, 30, 60)
REVIEW_LEASE_SECONDS = 15 * 60

_EVENT_TYPES = frozenset(
    {
        "evidence_outcome_recorded",
        "review_outcome_recorded",
        "review_claimed",
        "review_lease_released",
        "source_cancelled",
        "curriculum_retired",
    }
)
_OUTCOMES = frozenset({"correct", "partial", "incorrect"})
_SCHEDULE_STATES = frozenset(
    {"scheduled", "in_progress", "suspended", "retired"}
)
_AUTHORITIES = frozenset({"validated_rubric", "teacher_adjudicated"})
_CANCELLATION_REASONS = frozenset(
    {"source_deleted", "authority_revoked", "pending_adjudication"}
)
_LEASE_RELEASE_REASONS = frozenset({"cancelled", "abandoned"})
_SAFE_KC_ID = re.compile(r"^kc_[a-z0-9][a-z0-9_-]{2,80}$")
_SAFE_REFERENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_LEARNER_KEY = re.compile(r"^learner_[0-9a-f]{64}$")
_CURRICULUM_NAMESPACE = re.compile(r"^curriculum_[0-9a-f]{64}$")
_EVENT_ID = re.compile(r"^lre_[0-9a-f]{64}$")
_REVIEW_ID = re.compile(r"^review_[0-9a-f]{64}$")
_LEASE_ID = re.compile(r"^lease_[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_ANONYMOUS_SUBJECTS = frozenset(
    {
        "anonymous",
        "anon",
        "default",
        "guest",
        "unknown",
        "unauthenticated",
        "profile_default",
        "匿名",
        "访客",
        "游客",
        "未知",
        "默认",
        "默认用户",
        "未登录",
    }
)
_ENVELOPE_KEYS = frozenset(
    {
        "schema",
        "seq",
        "recorded_at_utc",
        "previous_hash",
        "outbox_event",
        "outbox_fingerprint",
        "target_version_after",
        "learner_version_after",
        "hash",
    }
)
_OUTBOX_KEYS = frozenset(
    {
        "schema",
        "event_id",
        "event_type",
        "target",
        "expected_version",
        "occurred_at_utc",
        "data",
    }
)
_TARGET_KEYS = frozenset(
    {
        "learner_key",
        "curriculum_namespace",
        "knowledge_component_id",
        "source_ref_sha256",
    }
)


class LearningRecordError(ValueError):
    """Raised when a learning-record value violates the strict contract."""


class LearningRecordStoreError(RuntimeError):
    """Raised when durable learning records cannot be trusted or committed."""


class LearningRecordConflictError(LearningRecordStoreError):
    """Raised for stale versions, active leases, or conflicting idempotency."""


@dataclass(frozen=True, slots=True)
class LearningRecordApplyResult:
    """Receipt for one applied or idempotently replayed outbox event."""

    event_id: str
    applied: bool
    committed_target_version: int
    current_record: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LearningRecordStoreRecovery:
    """Defensive replay result for diagnostics, export, and recovery tests."""

    records: dict[str, dict[str, Any]]
    event_count: int
    last_hash: str | None


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LearningRecordError("learning records must contain canonical JSON") from exc


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _canonical_subject(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _required_text(value: Any, *, field: str, maximum: int = 160) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise LearningRecordError(
            f"{field} must be a non-empty trimmed string <= {maximum} chars"
        )
    return value


def _safe_reference_id(value: Any, *, field: str) -> str:
    result = _required_text(value, field=field)
    if _SAFE_REFERENCE_ID.fullmatch(result) is None:
        raise LearningRecordError(f"{field} must be an opaque safe identifier")
    return result


def _bounded_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LearningRecordError(f"{field} must be an integer >= {minimum}")
    return value


def _parse_utc(value: Any, *, field: str) -> datetime:
    text = _required_text(value, field=field, maximum=40)
    if _UTC_TIMESTAMP.fullmatch(text) is None:
        raise LearningRecordError(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise LearningRecordError(f"{field} is not a real UTC timestamp") from exc
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise LearningRecordError("clock must return a timezone-aware datetime")
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat(timespec=(
        "microseconds" if normalized.microsecond else "seconds"
    )).replace("+00:00", "Z")


def _system_clock() -> datetime:
    return datetime.now(timezone.utc)


def mint_learner_key(
    subject_id: str,
    *,
    tenant_id: str,
    secret: bytes,
) -> str:
    """Mint a stable opaque learner key on the trusted server boundary.

    Neither the tenant nor subject identifier is persisted.  A deployment must
    keep ``secret`` outside browser-visible configuration and use at least 256
    bits.  Placeholder identities are rejected instead of being allowed to
    merge unrelated anonymous learners into one durable record.
    """

    subject = _canonical_subject(
        _required_text(subject_id, field="subject_id", maximum=240)
    )
    tenant = _canonical_subject(
        _required_text(tenant_id, field="tenant_id", maximum=160)
    )
    anonymous_tokens = {
        token
        for token in re.split(r"[^\w]+|_+", subject, flags=re.UNICODE)
        if token
    }
    if subject in _ANONYMOUS_SUBJECTS or anonymous_tokens.intersection(
        _ANONYMOUS_SUBJECTS
    ):
        raise LearningRecordError(
            "anonymous/default profiles cannot have cross-session learning records"
        )
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise LearningRecordError("learner-key secret must contain at least 32 bytes")
    material = _canonical_bytes({"tenant": tenant, "subject": subject})
    return "learner_" + hmac.new(secret, material, "sha256").hexdigest()


def curriculum_namespace_for_source_ref(
    source_ref: str, knowledge_component_id: str
) -> str:
    """Derive a curriculum namespace from a KC-v2 ``source_ref``.

    KC-v2 references end in ``#<kc_id>``.  Hashing the preceding scope prevents
    two unrelated curricula that happen to use the same label-derived KC ID
    from sharing a schedule.
    """

    reference = _required_text(source_ref, field="source_ref", maximum=300)
    kc_id = _required_text(
        knowledge_component_id, field="knowledge_component_id", maximum=83
    )
    if _SAFE_KC_ID.fullmatch(kc_id) is None:
        raise LearningRecordError("knowledge_component_id is invalid")
    scope, separator, suffix = reference.rpartition("#")
    if not separator or not scope or suffix != kc_id:
        raise LearningRecordError(
            "source_ref must end in the exact target knowledge_component_id"
        )
    return "curriculum_" + sha256(scope.encode("utf-8")).hexdigest()


def _validate_learner_key(value: Any) -> str:
    result = _required_text(value, field="learner_key", maximum=72)
    if _LEARNER_KEY.fullmatch(result) is None:
        raise LearningRecordError("learner_key must be a server-minted opaque key")
    return result


def _validate_target(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _TARGET_KEYS:
        raise LearningRecordError("learning event target must be a strict object")
    learner_key = _validate_learner_key(value.get("learner_key"))
    curriculum_namespace = _required_text(
        value.get("curriculum_namespace"),
        field="curriculum_namespace",
        maximum=75,
    )
    if _CURRICULUM_NAMESPACE.fullmatch(curriculum_namespace) is None:
        raise LearningRecordError("curriculum_namespace is invalid")
    kc_id = _required_text(
        value.get("knowledge_component_id"),
        field="knowledge_component_id",
        maximum=83,
    )
    if _SAFE_KC_ID.fullmatch(kc_id) is None:
        raise LearningRecordError("knowledge_component_id is invalid")
    source_ref_sha256 = _required_text(
        value.get("source_ref_sha256"), field="source_ref_sha256", maximum=64
    )
    if _SHA256.fullmatch(source_ref_sha256) is None:
        raise LearningRecordError("source_ref_sha256 is invalid")
    return {
        "learner_key": learner_key,
        "curriculum_namespace": curriculum_namespace,
        "knowledge_component_id": kc_id,
        "source_ref_sha256": source_ref_sha256,
    }


def learning_record_target(
    *, learner_key: str, knowledge_component: Mapping[str, Any]
) -> dict[str, str]:
    """Build the only persisted target shape from one KC-v2 component."""

    if not isinstance(knowledge_component, Mapping):
        raise LearningRecordError("knowledge_component must be an object")
    kc_id = _required_text(
        knowledge_component.get("kc_id"),
        field="knowledge_component.kc_id",
        maximum=83,
    )
    source_ref = _required_text(
        knowledge_component.get("source_ref"),
        field="knowledge_component.source_ref",
        maximum=300,
    )
    return _validate_target(
        {
            "learner_key": _validate_learner_key(learner_key),
            "curriculum_namespace": curriculum_namespace_for_source_ref(
                source_ref, kc_id
            ),
            "knowledge_component_id": kc_id,
            "source_ref_sha256": sha256(source_ref.encode("utf-8")).hexdigest(),
        }
    )


def _event_id(material: Mapping[str, Any]) -> str:
    return "lre_" + _canonical_sha256(material)


def _review_id(component: Mapping[str, Any]) -> str:
    schedule = component["schedule"]
    return "review_" + _canonical_sha256(
        {
            "learner_key": component["learner_key"],
            "curriculum_namespace": component["curriculum_namespace"],
            "knowledge_component_id": component["knowledge_component_id"],
            "due_at_utc": schedule["due_at_utc"],
            "source_observation_id": schedule["source_observation_id"],
        }
    )


def _outcome_from_evidence(evidence: Mapping[str, Any]) -> str:
    signal = evidence.get("signal")
    alignment = evidence.get("answer_alignment")
    if signal == "correct" and alignment == "aligned":
        return "correct"
    if signal == "partial" and alignment == "partially_aligned":
        return "partial"
    if signal == "misconception" and alignment == "contradicted":
        return "incorrect"
    raise LearningRecordError(
        "KC evidence signal/alignment is not a schedulable assessment outcome"
    )


def build_learning_evidence_outbox_event(
    *,
    learner_key: str,
    knowledge_component: Mapping[str, Any],
    evidence: Mapping[str, Any],
    expected_version: int,
    committed_at_utc: str,
    commit_receipt_id: str,
    review_id: str | None = None,
    lease_id: str | None = None,
) -> dict[str, Any]:
    """Build an outcome event from an applied authoritative KC-v2 ledger row.

    ``outcome`` is deliberately not an argument.  It is derived from the
    immutable evidence row, which must already be rubric-bound, authoritative,
    assessment-eligible, and assigned to this exact KC.  For a due review the
    caller supplies the active server-issued review and lease IDs; the reducer
    checks both against durable state.  ``committed_at_utc`` and
    ``commit_receipt_id`` must come from the server's durable turn-commit
    receipt; session-logical evidence time is provenance and is never used as
    the cross-session scheduling clock.
    """

    if not isinstance(knowledge_component, Mapping) or not isinstance(
        evidence, Mapping
    ):
        raise LearningRecordError("knowledge_component and evidence must be objects")
    if knowledge_component.get("teacher_grading_authority_available") is not True:
        raise LearningRecordError(
            "a KC without teacher grading authority cannot schedule a review"
        )
    target = learning_record_target(
        learner_key=learner_key, knowledge_component=knowledge_component
    )
    kc_id = target["knowledge_component_id"]
    if evidence.get("knowledge_component_id") != kc_id:
        raise LearningRecordError("evidence must target exactly the selected KC")
    if (
        evidence.get("assessment_eligible") is not True
        or evidence.get("authoritative") is not True
    ):
        raise LearningRecordError(
            "only authoritative assessment-eligible evidence can schedule review"
        )
    outcome = _outcome_from_evidence(evidence)
    observation_id = _safe_reference_id(
        evidence.get("evidence_id"), field="evidence.evidence_id"
    )
    item_id = _safe_reference_id(evidence.get("item_id"), field="evidence.item_id")
    question_id = _safe_reference_id(
        evidence.get("question_id"), field="evidence.question_id"
    )
    rubric_id = _safe_reference_id(
        evidence.get("rubric_id"), field="evidence.rubric_id"
    )
    evidence_sha256 = _required_text(
        evidence.get("evidence_fingerprint"),
        field="evidence.evidence_fingerprint",
        maximum=64,
    )
    if _SHA256.fullmatch(evidence_sha256) is None:
        raise LearningRecordError("evidence.evidence_fingerprint is invalid")
    evidence_observed_at = _utc_text(
        _parse_utc(evidence.get("observed_at"), field="evidence.observed_at")
    )
    evidence_time_basis = _required_text(
        evidence.get("time_basis"), field="evidence.time_basis", maximum=40
    )
    if evidence_time_basis not in {"session_logical", "wall_clock_utc"}:
        raise LearningRecordError("evidence.time_basis is invalid")
    occurred_at_utc = _utc_text(
        _parse_utc(committed_at_utc, field="committed_at_utc")
    )
    if evidence_time_basis == "wall_clock_utc" and _parse_utc(
        evidence_observed_at, field="evidence.observed_at"
    ) > _parse_utc(occurred_at_utc, field="committed_at_utc"):
        raise LearningRecordError("wall-clock evidence cannot postdate its commit")
    receipt_id = _safe_reference_id(
        commit_receipt_id, field="commit_receipt_id"
    )
    expected = _bounded_integer(
        expected_version, field="expected_version", minimum=0
    )

    if (review_id is None) != (lease_id is None):
        raise LearningRecordError("review_id and lease_id must be supplied together")
    event_type = "evidence_outcome_recorded"
    if review_id is not None:
        if _REVIEW_ID.fullmatch(review_id) is None:
            raise LearningRecordError("review_id is invalid")
        if not isinstance(lease_id, str) or _LEASE_ID.fullmatch(lease_id) is None:
            raise LearningRecordError("lease_id is invalid")
        event_type = "review_outcome_recorded"

    data = {
        "outcome": outcome,
        "source_observation_id": observation_id,
        "evidence_sha256": evidence_sha256,
        "item_id": item_id,
        "question_id": question_id,
        "rubric_id": rubric_id,
        "authority": "validated_rubric",
        "assessment_eligible": True,
        "commit_receipt_id": receipt_id,
        "evidence_observed_at": evidence_observed_at,
        "evidence_time_basis": evidence_time_basis,
        "review_id": review_id,
        "lease_id": lease_id,
    }
    event = {
        "schema": LEARNING_OUTBOX_EVENT_SCHEMA,
        "event_id": _event_id(
            {
                "event_type": event_type,
                "target": target,
                "source_observation_id": observation_id,
                "evidence_sha256": evidence_sha256,
                "commit_receipt_id": receipt_id,
                "committed_at_utc": occurred_at_utc,
                "review_id": review_id,
            }
        ),
        "event_type": event_type,
        "target": target,
        "expected_version": expected,
        "occurred_at_utc": occurred_at_utc,
        "data": data,
    }
    validate_learning_outbox_event(event)
    return event


def _validate_outcome_data(
    data: Mapping[str, Any], *, review: bool
) -> None:
    required = {
        "outcome",
        "source_observation_id",
        "evidence_sha256",
        "item_id",
        "question_id",
        "rubric_id",
        "authority",
        "assessment_eligible",
        "commit_receipt_id",
        "evidence_observed_at",
        "evidence_time_basis",
        "review_id",
        "lease_id",
    }
    if set(data) != required:
        raise LearningRecordError("learning outcome data must be a strict object")
    if data.get("outcome") not in _OUTCOMES:
        raise LearningRecordError("learning outcome is invalid")
    for field in ("source_observation_id", "item_id", "question_id", "rubric_id"):
        _safe_reference_id(data.get(field), field=f"data.{field}")
    evidence_sha256 = _required_text(
        data.get("evidence_sha256"), field="data.evidence_sha256", maximum=64
    )
    if _SHA256.fullmatch(evidence_sha256) is None:
        raise LearningRecordError("data.evidence_sha256 is invalid")
    if data.get("authority") not in _AUTHORITIES:
        raise LearningRecordError("data.authority is invalid")
    if data.get("assessment_eligible") is not True:
        raise LearningRecordError("outcome evidence must be assessment-eligible")
    _safe_reference_id(
        data.get("commit_receipt_id"), field="data.commit_receipt_id"
    )
    _parse_utc(data.get("evidence_observed_at"), field="data.evidence_observed_at")
    evidence_time_basis = data.get("evidence_time_basis")
    if evidence_time_basis not in {"session_logical", "wall_clock_utc"}:
        raise LearningRecordError("data.evidence_time_basis is invalid")
    if review:
        review_id = _required_text(data.get("review_id"), field="data.review_id")
        lease_id = _required_text(data.get("lease_id"), field="data.lease_id")
        if _REVIEW_ID.fullmatch(review_id) is None or _LEASE_ID.fullmatch(
            lease_id
        ) is None:
            raise LearningRecordError("review outcome lease identity is invalid")
    elif data.get("review_id") is not None or data.get("lease_id") is not None:
        raise LearningRecordError("ordinary evidence cannot self-certify a review lease")


def validate_learning_outbox_event(event: Mapping[str, Any]) -> None:
    """Validate one strict, text-free event suitable for a session outbox."""

    if not isinstance(event, Mapping) or set(event) != _OUTBOX_KEYS:
        raise LearningRecordError("learning outbox event must be a strict object")
    if event.get("schema") != LEARNING_OUTBOX_EVENT_SCHEMA:
        raise LearningRecordError("learning outbox event schema is unsupported")
    event_id = _required_text(event.get("event_id"), field="event_id", maximum=68)
    if _EVENT_ID.fullmatch(event_id) is None:
        raise LearningRecordError("event_id is invalid")
    event_type = event.get("event_type")
    if event_type not in _EVENT_TYPES:
        raise LearningRecordError("learning outbox event type is unsupported")
    _validate_target(event.get("target"))
    _bounded_integer(event.get("expected_version"), field="expected_version")
    occurred = _parse_utc(event.get("occurred_at_utc"), field="occurred_at_utc")
    data = event.get("data")
    if not isinstance(data, Mapping):
        raise LearningRecordError("learning outbox event data must be an object")

    if event_type in {"evidence_outcome_recorded", "review_outcome_recorded"}:
        _validate_outcome_data(
            data, review=event_type == "review_outcome_recorded"
        )
        return
    if event_type == "review_claimed":
        if set(data) != {"review_id", "lease_id", "lease_expires_at_utc"}:
            raise LearningRecordError("review claim data must be a strict object")
        review_id = _required_text(data.get("review_id"), field="data.review_id")
        lease_id = _required_text(data.get("lease_id"), field="data.lease_id")
        if _REVIEW_ID.fullmatch(review_id) is None or _LEASE_ID.fullmatch(
            lease_id
        ) is None:
            raise LearningRecordError("review claim identity is invalid")
        expires = _parse_utc(
            data.get("lease_expires_at_utc"), field="data.lease_expires_at_utc"
        )
        if expires <= occurred or expires - occurred > timedelta(hours=1):
            raise LearningRecordError("review lease duration is invalid")
        return
    if event_type == "review_lease_released":
        if set(data) != {"review_id", "lease_id", "reason"}:
            raise LearningRecordError("review release data must be a strict object")
        review_id = _required_text(data.get("review_id"), field="data.review_id")
        lease_id = _required_text(data.get("lease_id"), field="data.lease_id")
        if _REVIEW_ID.fullmatch(review_id) is None or _LEASE_ID.fullmatch(
            lease_id
        ) is None:
            raise LearningRecordError("review release identity is invalid")
        if data.get("reason") not in _LEASE_RELEASE_REASONS:
            raise LearningRecordError("review lease release reason is invalid")
        return
    if event_type == "source_cancelled":
        if set(data) != {"source_observation_id", "reason"}:
            raise LearningRecordError("source cancellation data must be strict")
        _safe_reference_id(
            data.get("source_observation_id"),
            field="data.source_observation_id",
        )
        if data.get("reason") not in _CANCELLATION_REASONS:
            raise LearningRecordError("source cancellation reason is invalid")
        return
    if event_type == "curriculum_retired":
        if set(data) != {"reason"} or data.get("reason") != "curriculum_retired":
            raise LearningRecordError("curriculum retirement data is invalid")
        return
    raise LearningRecordError("learning outbox event type is unsupported")


def _empty_schedule() -> dict[str, Any]:
    return {
        "state": "scheduled",
        "due_at_utc": None,
        "interval_days": None,
        "repetition": 0,
        "lapse_count": 0,
        "last_reviewed_at_utc": None,
        "last_outcome": None,
        "source_observation_id": None,
        "source_evidence_sha256": None,
        "active_review_id": None,
        "active_lease_id": None,
        "lease_expires_at_utc": None,
        "suspension_reason": None,
    }


def _new_component(target: Mapping[str, str]) -> dict[str, Any]:
    return {
        "learner_key": target["learner_key"],
        "curriculum_namespace": target["curriculum_namespace"],
        "knowledge_component_id": target["knowledge_component_id"],
        "source_ref_sha256": target["source_ref_sha256"],
        "version": 0,
        "schedule": _empty_schedule(),
    }


def _component_from_records(
    records: Mapping[str, Any], target: Mapping[str, str]
) -> Mapping[str, Any] | None:
    learner = records.get(target["learner_key"])
    if not isinstance(learner, Mapping):
        return None
    namespaces = learner.get("knowledge_components")
    if not isinstance(namespaces, Mapping):
        return None
    components = namespaces.get(target["curriculum_namespace"])
    if not isinstance(components, Mapping):
        return None
    component = components.get(target["knowledge_component_id"])
    return component if isinstance(component, Mapping) else None


def _mutable_component(
    records: dict[str, dict[str, Any]],
    target: Mapping[str, str],
    *,
    create: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    learner = records.get(target["learner_key"])
    if learner is None:
        if not create:
            raise LearningRecordConflictError("learning record target does not exist")
        learner = {
            "schema": LEARNING_RECORD_SCHEMA,
            "learner_key": target["learner_key"],
            "version": 0,
            "updated_at_utc": None,
            "knowledge_components": {},
        }
        records[target["learner_key"]] = learner
    namespaces = learner["knowledge_components"]
    components = namespaces.setdefault(target["curriculum_namespace"], {})
    component = components.get(target["knowledge_component_id"])
    if component is None:
        if not create:
            raise LearningRecordConflictError("learning record target does not exist")
        component = _new_component(target)
        components[target["knowledge_component_id"]] = component
    if component["source_ref_sha256"] != target["source_ref_sha256"]:
        raise LearningRecordConflictError(
            "KC target source_ref changed inside an existing curriculum namespace"
        )
    return learner, component


def _clear_lease(schedule: dict[str, Any]) -> None:
    schedule["active_review_id"] = None
    schedule["active_lease_id"] = None
    schedule["lease_expires_at_utc"] = None


def _apply_outcome(schedule: dict[str, Any], data: Mapping[str, Any], at: str) -> None:
    outcome = str(data["outcome"])
    if outcome == "correct":
        repetition = min(
            int(schedule["repetition"]) + 1, len(REVIEW_INTERVAL_DAYS)
        )
        interval_days = REVIEW_INTERVAL_DAYS[repetition - 1]
    else:
        repetition = 0
        interval_days = REVIEW_INTERVAL_DAYS[0]
        schedule["lapse_count"] = int(schedule["lapse_count"]) + 1
    due = _parse_utc(at, field="occurred_at_utc") + timedelta(
        days=interval_days
    )
    schedule.update(
        {
            "state": "scheduled",
            "due_at_utc": _utc_text(due),
            "interval_days": interval_days,
            "repetition": repetition,
            "last_reviewed_at_utc": at,
            "last_outcome": outcome,
            "source_observation_id": data["source_observation_id"],
            "source_evidence_sha256": data["evidence_sha256"],
            "suspension_reason": None,
        }
    )
    _clear_lease(schedule)


def _reduce_event(
    records: dict[str, dict[str, Any]], event: Mapping[str, Any]
) -> tuple[int, int]:
    validate_learning_outbox_event(event)
    target = _validate_target(event["target"])
    event_type = str(event["event_type"])
    create = event_type == "evidence_outcome_recorded"
    learner, component = _mutable_component(records, target, create=create)
    expected_version = int(event["expected_version"])
    if component["version"] != expected_version:
        raise LearningRecordConflictError(
            "learning record expected_version is stale"
        )
    schedule = component["schedule"]
    occurred_text = _utc_text(
        _parse_utc(event["occurred_at_utc"], field="occurred_at_utc")
    )
    occurred = _parse_utc(occurred_text, field="occurred_at_utc")

    if event_type in {"evidence_outcome_recorded", "review_outcome_recorded"}:
        last_reviewed = schedule.get("last_reviewed_at_utc")
        precedes_latest = last_reviewed is not None and occurred < _parse_utc(
            last_reviewed, field="last_reviewed_at_utc"
        )
        if precedes_latest and event_type == "review_outcome_recorded":
            raise LearningRecordConflictError(
                "outcome precedes the target KC's latest durable outcome"
            )
        if schedule["state"] == "retired":
            raise LearningRecordConflictError("a retired KC cannot receive outcomes")
        if event_type == "review_outcome_recorded":
            if schedule["state"] != "in_progress":
                raise LearningRecordConflictError("review outcome has no active lease")
            if (
                event["data"]["review_id"] != schedule["active_review_id"]
                or event["data"]["lease_id"] != schedule["active_lease_id"]
            ):
                raise LearningRecordConflictError(
                    "review outcome does not match the active lease"
                )
            expires = _parse_utc(
                schedule["lease_expires_at_utc"], field="lease_expires_at_utc"
            )
            if occurred >= expires:
                raise LearningRecordConflictError("review outcome lease has expired")
        elif schedule["state"] == "in_progress":
            raise LearningRecordConflictError(
                "ordinary evidence cannot bypass an active review lease"
            )
        # An ordinary committed teaching observation may arrive after a newer
        # observation from another session/process.  It remains immutable audit
        # evidence and advances the target version exactly once, but it must not
        # roll the current due date/source backward.  Review outcomes retain
        # strict lease and temporal ordering above.
        if not precedes_latest:
            _apply_outcome(schedule, event["data"], occurred_text)
    elif event_type == "review_claimed":
        due = schedule.get("due_at_utc")
        if due is None:
            raise LearningRecordConflictError("review target has no due date")
        current_review_id = _review_id(component)
        if schedule["state"] == "scheduled":
            if occurred < _parse_utc(due, field="due_at_utc"):
                raise LearningRecordConflictError("review target is not due")
        elif schedule["state"] == "in_progress":
            if occurred < _parse_utc(
                schedule["lease_expires_at_utc"], field="lease_expires_at_utc"
            ):
                raise LearningRecordConflictError("review target has an active lease")
            current_review_id = str(schedule["active_review_id"])
        else:
            raise LearningRecordConflictError(
                "suspended or retired KCs cannot be claimed"
            )
        if event["data"]["review_id"] != current_review_id:
            raise LearningRecordConflictError("review_id is stale or targets another KC")
        schedule["state"] = "in_progress"
        schedule["active_review_id"] = current_review_id
        schedule["active_lease_id"] = event["data"]["lease_id"]
        schedule["lease_expires_at_utc"] = event["data"][
            "lease_expires_at_utc"
        ]
        schedule["suspension_reason"] = None
    elif event_type == "review_lease_released":
        if (
            schedule["state"] != "in_progress"
            or event["data"]["review_id"] != schedule["active_review_id"]
            or event["data"]["lease_id"] != schedule["active_lease_id"]
        ):
            raise LearningRecordConflictError(
                "review release does not match the active lease"
            )
        schedule["state"] = "scheduled"
        _clear_lease(schedule)
    elif event_type == "source_cancelled":
        if event["data"]["source_observation_id"] != schedule.get(
            "source_observation_id"
        ):
            raise LearningRecordConflictError(
                "source cancellation does not target the current schedule source"
            )
        if schedule["state"] == "retired":
            raise LearningRecordConflictError("retired KC cannot be suspended")
        schedule["state"] = "suspended"
        schedule["suspension_reason"] = event["data"]["reason"]
        _clear_lease(schedule)
    elif event_type == "curriculum_retired":
        schedule["state"] = "retired"
        schedule["suspension_reason"] = "curriculum_retired"
        _clear_lease(schedule)
    else:  # pragma: no cover - validator owns the exhaustive event set.
        raise LearningRecordError("learning event reducer is incomplete")

    component["version"] = int(component["version"]) + 1
    learner["version"] = int(learner["version"]) + 1
    previous_update = learner.get("updated_at_utc")
    if previous_update is None or occurred >= _parse_utc(
        previous_update, field="updated_at_utc"
    ):
        learner["updated_at_utc"] = occurred_text
    validate_learning_record(learner)
    return int(component["version"]), int(learner["version"])


def validate_learning_record(record: Mapping[str, Any]) -> None:
    """Validate a fully replayed learner projection and scheduler invariants."""

    if not isinstance(record, Mapping) or set(record) != {
        "schema",
        "learner_key",
        "version",
        "updated_at_utc",
        "knowledge_components",
    }:
        raise LearningRecordError("learner learning record must be a strict object")
    if record.get("schema") != LEARNING_RECORD_SCHEMA:
        raise LearningRecordError("learner learning record schema is unsupported")
    learner_key = _validate_learner_key(record.get("learner_key"))
    version = _bounded_integer(record.get("version"), field="record.version", minimum=1)
    _parse_utc(record.get("updated_at_utc"), field="record.updated_at_utc")
    namespaces = record.get("knowledge_components")
    if not isinstance(namespaces, Mapping) or not namespaces:
        raise LearningRecordError("learning record must contain knowledge components")
    component_versions = 0
    for namespace, components in namespaces.items():
        if not isinstance(namespace, str) or _CURRICULUM_NAMESPACE.fullmatch(
            namespace
        ) is None:
            raise LearningRecordError("learning record curriculum namespace is invalid")
        if not isinstance(components, Mapping) or not components:
            raise LearningRecordError("curriculum namespace must contain KCs")
        for kc_id, component in components.items():
            if not isinstance(kc_id, str) or _SAFE_KC_ID.fullmatch(kc_id) is None:
                raise LearningRecordError("learning record KC key is invalid")
            if not isinstance(component, Mapping) or set(component) != {
                "learner_key",
                "curriculum_namespace",
                "knowledge_component_id",
                "source_ref_sha256",
                "version",
                "schedule",
            }:
                raise LearningRecordError("learning record KC must be a strict object")
            if (
                component.get("learner_key") != learner_key
                or component.get("curriculum_namespace") != namespace
                or component.get("knowledge_component_id") != kc_id
            ):
                raise LearningRecordError("learning record KC identity is inconsistent")
            digest = component.get("source_ref_sha256")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise LearningRecordError("learning record source hash is invalid")
            component_version = _bounded_integer(
                component.get("version"), field="component.version", minimum=1
            )
            component_versions += component_version
            schedule = component.get("schedule")
            _validate_schedule(schedule)
    if component_versions != version:
        raise LearningRecordError(
            "learner record version must equal the sum of target revisions"
        )


def _validate_schedule(value: Any) -> None:
    keys = {
        "state",
        "due_at_utc",
        "interval_days",
        "repetition",
        "lapse_count",
        "last_reviewed_at_utc",
        "last_outcome",
        "source_observation_id",
        "source_evidence_sha256",
        "active_review_id",
        "active_lease_id",
        "lease_expires_at_utc",
        "suspension_reason",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise LearningRecordError("KC review schedule must be a strict object")
    state = value.get("state")
    if state not in _SCHEDULE_STATES:
        raise LearningRecordError("KC review schedule state is invalid")
    due_at = value.get("due_at_utc")
    if due_at is None:
        raise LearningRecordError("persisted KC review schedule must have a due date")
    _parse_utc(due_at, field="schedule.due_at_utc")
    interval = value.get("interval_days")
    if interval not in REVIEW_INTERVAL_DAYS:
        raise LearningRecordError("KC review interval is outside the fixed ladder")
    repetition = _bounded_integer(
        value.get("repetition"), field="schedule.repetition"
    )
    if repetition > len(REVIEW_INTERVAL_DAYS):
        raise LearningRecordError("KC review repetition exceeds the ladder")
    _bounded_integer(value.get("lapse_count"), field="schedule.lapse_count")
    _parse_utc(
        value.get("last_reviewed_at_utc"), field="schedule.last_reviewed_at_utc"
    )
    if value.get("last_outcome") not in _OUTCOMES:
        raise LearningRecordError("KC review last outcome is invalid")
    _safe_reference_id(
        value.get("source_observation_id"),
        field="schedule.source_observation_id",
    )
    evidence_hash = value.get("source_evidence_sha256")
    if not isinstance(evidence_hash, str) or _SHA256.fullmatch(evidence_hash) is None:
        raise LearningRecordError("KC review source evidence hash is invalid")
    active_values = (
        value.get("active_review_id"),
        value.get("active_lease_id"),
        value.get("lease_expires_at_utc"),
    )
    if state == "in_progress":
        if not all(isinstance(item, str) for item in active_values):
            raise LearningRecordError("in-progress review must have a complete lease")
        if _REVIEW_ID.fullmatch(str(active_values[0])) is None or _LEASE_ID.fullmatch(
            str(active_values[1])
        ) is None:
            raise LearningRecordError("in-progress review identity is invalid")
        _parse_utc(active_values[2], field="schedule.lease_expires_at_utc")
    elif any(item is not None for item in active_values):
        raise LearningRecordError("inactive review schedule cannot retain a lease")
    suspension_reason = value.get("suspension_reason")
    if state in {"suspended", "retired"}:
        _required_text(
            suspension_reason, field="schedule.suspension_reason", maximum=80
        )
    elif suspension_reason is not None:
        raise LearningRecordError("active schedule cannot retain a suspension reason")


def _envelope_hash(value: Mapping[str, Any]) -> str:
    material = dict(value)
    material.pop("hash", None)
    return _canonical_sha256(material)


class LearningRecordStore:
    """Private append-only hash-chain store with deterministic replay and CAS."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.tombstone_path = self.path.with_name(
            f".{self.path.name}.erasure_tombstones.json"
        )
        self._clock = clock or _system_clock
        self._thread_lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._event_index: dict[str, dict[str, Any]] = {}
        self._last_hash: str | None = None
        self._file_size = 0
        self._erasure_tombstones: dict[str, dict[str, Any]] = {}
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            self._refresh_locked(repair_truncated_tail=True)

    def _now(self) -> datetime:
        return _parse_utc(_utc_text(self._clock()), field="clock")

    @contextmanager
    def _process_lock(self) -> Iterator[None]:
        if fcntl is None:
            raise LearningRecordStoreError(
                "durable learning records require POSIX process locking"
            )
        try:
            if self.path.parent.exists() and self.path.parent.is_symlink():
                raise LearningRecordStoreError(
                    "learning record store directory must not be a symlink"
                )
            ensure_private_directory(self.path.parent)
            parent_stat = self.path.parent.lstat()
            if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(
                parent_stat.st_mode
            ):
                raise LearningRecordStoreError(
                    "learning record store directory must be a private directory"
                )
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.lock_path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
        except OSError as exc:
            raise LearningRecordStoreError(
                "learning record process lock cannot be opened"
            ) from exc
        try:
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise LearningRecordStoreError(
                    "learning record process lock must be a regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except OSError as exc:
            raise LearningRecordStoreError(
                "learning record process lock cannot be acquired"
            ) from exc
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(descriptor)

    def _refresh_locked(self, *, repair_truncated_tail: bool) -> None:
        if not self.path.exists():
            self._publish_replay([], {}, {}, None, 0)
            return
        try:
            file_stat = self.path.lstat()
        except OSError as exc:
            raise LearningRecordStoreError(
                "learning record store metadata cannot be read"
            ) from exc
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise LearningRecordStoreError(
                "learning record store must be a regular non-symlink file"
            )
        try:
            with self.path.open("r+b") as stream:
                raw = stream.read()
                final_newline = raw.rfind(b"\n")
                valid_size = final_newline + 1 if final_newline >= 0 else 0
                if valid_size != len(raw):
                    if not repair_truncated_tail:
                        raise LearningRecordStoreError(
                            "learning record store has a truncated tail"
                        )
                    stream.seek(valid_size)
                    stream.truncate(valid_size)
                    stream.flush()
                    os.fsync(stream.fileno())
                complete = raw[:valid_size]
        except OSError as exc:
            raise LearningRecordStoreError(
                "learning record store cannot be read"
            ) from exc
        events: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(complete.splitlines(), 1):
            if not raw_line:
                raise LearningRecordStoreError(
                    f"learning record line {line_number} is empty"
                )
            try:
                envelope = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise LearningRecordStoreError(
                    f"learning record line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(envelope, dict) or raw_line != _canonical_bytes(envelope):
                raise LearningRecordStoreError(
                    f"learning record line {line_number} is not canonical JSON"
                )
            events.append(envelope)
        records, index, last_hash = self._replay(events)
        self._publish_replay(events, records, index, last_hash, valid_size)
        ensure_private_file(self.path)

    def _refresh_tombstones_locked(self) -> None:
        """Load the content-free erasure fence before accepting any append."""

        if not self.tombstone_path.exists():
            self._erasure_tombstones = {}
            return
        try:
            file_stat = self.tombstone_path.lstat()
        except OSError as exc:
            raise LearningRecordStoreError(
                "learning erasure tombstone metadata cannot be read"
            ) from exc
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise LearningRecordStoreError(
                "learning erasure tombstones must be a regular non-symlink file"
            )
        try:
            raw = self.tombstone_path.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LearningRecordStoreError(
                "learning erasure tombstones cannot be read"
            ) from exc
        if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
            raise LearningRecordStoreError(
                "learning erasure tombstones are not canonical JSON"
            )
        if set(value) != {"schema", "version", "tombstones"} or value.get(
            "schema"
        ) != LEARNING_ERASURE_TOMBSTONES_SCHEMA:
            raise LearningRecordStoreError(
                "learning erasure tombstone schema is unsupported"
            )
        try:
            version = _bounded_integer(
                value.get("version"), field="erasure_tombstones.version"
            )
            tombstones = value.get("tombstones")
            if not isinstance(tombstones, Mapping):
                raise LearningRecordError("erasure tombstones must be an object")
            normalized: dict[str, dict[str, Any]] = {}
            for learner_key, tombstone in tombstones.items():
                key = _validate_learner_key(learner_key)
                if not isinstance(tombstone, Mapping) or set(tombstone) != {
                    "purged_at_utc",
                    "generation",
                }:
                    raise LearningRecordError(
                        "learning erasure tombstone must be a strict object"
                    )
                normalized[key] = {
                    "purged_at_utc": _utc_text(
                        _parse_utc(
                            tombstone.get("purged_at_utc"),
                            field="tombstone.purged_at_utc",
                        )
                    ),
                    "generation": _bounded_integer(
                        tombstone.get("generation"),
                        field="tombstone.generation",
                        minimum=1,
                    ),
                }
            if version != len(normalized):
                raise LearningRecordError(
                    "erasure tombstone version must equal its learner count"
                )
        except LearningRecordError as exc:
            raise LearningRecordStoreError(
                "learning erasure tombstones failed validation"
            ) from exc
        self._erasure_tombstones = normalized
        ensure_private_file(self.tombstone_path)

    def _write_tombstones_locked(
        self, tombstones: Mapping[str, Mapping[str, Any]]
    ) -> None:
        document = {
            "schema": LEARNING_ERASURE_TOMBSTONES_SCHEMA,
            "version": len(tombstones),
            "tombstones": deepcopy(dict(tombstones)),
        }
        payload = _canonical_bytes(document) + b"\n"
        self._atomic_replace_path(self.tombstone_path, payload)
        self._erasure_tombstones = deepcopy(dict(tombstones))

    def _reject_erased_target(self, event: Mapping[str, Any]) -> None:
        learner_key = str(event["target"]["learner_key"])
        if learner_key in self._erasure_tombstones:
            raise LearningRecordConflictError(
                "learner key was permanently erased; server must mint a new key for re-enrollment"
            )

    def _publish_replay(
        self,
        events: Sequence[Mapping[str, Any]],
        records: Mapping[str, Mapping[str, Any]],
        index: Mapping[str, Mapping[str, Any]],
        last_hash: str | None,
        file_size: int,
    ) -> None:
        self._events = deepcopy([dict(item) for item in events])
        self._records = deepcopy({key: dict(value) for key, value in records.items()})
        self._event_index = deepcopy({key: dict(value) for key, value in index.items()})
        self._last_hash = last_hash
        self._file_size = file_size

    def _replay(
        self, events: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str | None]:
        records: dict[str, dict[str, Any]] = {}
        index: dict[str, dict[str, Any]] = {}
        previous_hash: str | None = None
        for expected_seq, envelope in enumerate(events, 1):
            try:
                if set(envelope) != _ENVELOPE_KEYS:
                    raise LearningRecordError("store event must be a strict object")
                if envelope.get("schema") != LEARNING_STORE_EVENT_SCHEMA:
                    raise LearningRecordError("store event schema is unsupported")
                if envelope.get("seq") != expected_seq:
                    raise LearningRecordError("store event sequence is non-contiguous")
                if envelope.get("previous_hash") != previous_hash:
                    raise LearningRecordError("store event hash chain is broken")
                claimed_hash = envelope.get("hash")
                if (
                    not isinstance(claimed_hash, str)
                    or _SHA256.fullmatch(claimed_hash) is None
                    or claimed_hash != _envelope_hash(envelope)
                ):
                    raise LearningRecordError("store event hash is invalid")
                recorded = _parse_utc(
                    envelope.get("recorded_at_utc"), field="recorded_at_utc"
                )
                outbox_event = envelope.get("outbox_event")
                if not isinstance(outbox_event, Mapping):
                    raise LearningRecordError("store event outbox payload is invalid")
                validate_learning_outbox_event(outbox_event)
                if _parse_utc(
                    outbox_event["occurred_at_utc"], field="occurred_at_utc"
                ) > recorded:
                    raise LearningRecordError("outbox event occurs after its commit")
                fingerprint = _canonical_sha256(outbox_event)
                if envelope.get("outbox_fingerprint") != fingerprint:
                    raise LearningRecordError("outbox event fingerprint is invalid")
                event_id = str(outbox_event["event_id"])
                if event_id in index:
                    raise LearningRecordError("store contains a duplicate event_id")
                target_version, learner_version = _reduce_event(
                    records, outbox_event
                )
                if (
                    envelope.get("target_version_after") != target_version
                    or envelope.get("learner_version_after") != learner_version
                ):
                    raise LearningRecordError("store event reducer receipt is invalid")
            except (LearningRecordError, LearningRecordConflictError) as exc:
                raise LearningRecordStoreError(
                    f"learning record line {expected_seq} failed replay validation"
                ) from exc
            normalized = deepcopy(dict(envelope))
            index[event_id] = normalized
            previous_hash = str(envelope["hash"])
        return records, index, previous_hash

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            self._refresh_locked(repair_truncated_tail=True)
            return tuple(
                deepcopy(event)
                for event in self._events
                if event["outbox_event"]["target"]["learner_key"]
                not in self._erasure_tombstones
            )

    def recover(self) -> LearningRecordStoreRecovery:
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            self._refresh_locked(repair_truncated_tail=True)
            return LearningRecordStoreRecovery(
                records={
                    key: deepcopy(record)
                    for key, record in self._records.items()
                    if key not in self._erasure_tombstones
                },
                event_count=sum(
                    event["outbox_event"]["target"]["learner_key"]
                    not in self._erasure_tombstones
                    for event in self._events
                ),
                last_hash=self._last_hash,
            )

    def get_learner_record(self, learner_key: str) -> dict[str, Any] | None:
        key = _validate_learner_key(learner_key)
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            self._refresh_locked(repair_truncated_tail=True)
            if key in self._erasure_tombstones:
                return None
            record = self._records.get(key)
            return None if record is None else deepcopy(record)

    def is_learner_erased(self, learner_key: str) -> bool:
        """Return whether a content-free permanent-erasure fence exists."""

        key = _validate_learner_key(learner_key)
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            return key in self._erasure_tombstones

    def get_knowledge_component_record(
        self,
        *,
        learner_key: str,
        curriculum_namespace: str,
        knowledge_component_id: str,
    ) -> dict[str, Any] | None:
        target_stub = {
            "learner_key": _validate_learner_key(learner_key),
            "curriculum_namespace": curriculum_namespace,
            "knowledge_component_id": knowledge_component_id,
            "source_ref_sha256": "0" * 64,
        }
        normalized = _validate_target(target_stub)
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            self._refresh_locked(repair_truncated_tail=True)
            if normalized["learner_key"] in self._erasure_tombstones:
                return None
            component = _component_from_records(self._records, normalized)
            return None if component is None else deepcopy(dict(component))

    def _build_envelopes(
        self,
        outbox_events: Sequence[Mapping[str, Any]],
        *,
        recorded_at: str,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        candidate_records = deepcopy(self._records)
        candidate_index = deepcopy(self._event_index)
        envelopes: list[dict[str, Any]] = []
        previous_hash = self._last_hash
        next_seq = len(self._events) + 1
        for raw_event in outbox_events:
            event = deepcopy(dict(raw_event))
            validate_learning_outbox_event(event)
            self._reject_erased_target(event)
            if _parse_utc(event["occurred_at_utc"], field="occurred_at_utc") > _parse_utc(
                recorded_at, field="recorded_at_utc"
            ):
                raise LearningRecordConflictError(
                    "learning outbox event cannot occur in the future"
                )
            event_id = str(event["event_id"])
            existing = candidate_index.get(event_id)
            if existing is not None:
                if existing["outbox_fingerprint"] != _canonical_sha256(event):
                    raise LearningRecordConflictError(
                        "event_id was replayed with different learning content"
                    )
                continue
            target_version, learner_version = _reduce_event(candidate_records, event)
            envelope: dict[str, Any] = {
                "schema": LEARNING_STORE_EVENT_SCHEMA,
                "seq": next_seq,
                "recorded_at_utc": recorded_at,
                "previous_hash": previous_hash,
                "outbox_event": event,
                "outbox_fingerprint": _canonical_sha256(event),
                "target_version_after": target_version,
                "learner_version_after": learner_version,
            }
            envelope["hash"] = _envelope_hash(envelope)
            envelopes.append(envelope)
            candidate_index[event_id] = deepcopy(envelope)
            previous_hash = envelope["hash"]
            next_seq += 1
        return envelopes, candidate_records, candidate_index

    def _write_envelopes_locked(self, envelopes: Sequence[Mapping[str, Any]]) -> None:
        if not envelopes:
            return
        payload = b"".join(_canonical_bytes(item) + b"\n" for item in envelopes)
        existed = self.path.exists()
        expected_identity: tuple[int, int] | None = None
        if existed:
            current_stat = self.path.lstat()
            if stat.S_ISLNK(current_stat.st_mode) or not stat.S_ISREG(
                current_stat.st_mode
            ):
                raise LearningRecordStoreError(
                    "learning record store must be a regular non-symlink file"
                )
            expected_identity = (current_stat.st_dev, current_stat.st_ino)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "r+b") as stream:
                opened_stat = os.fstat(stream.fileno())
                if expected_identity is not None and expected_identity != (
                    opened_stat.st_dev,
                    opened_stat.st_ino,
                ):
                    raise LearningRecordStoreError(
                        "learning record store changed during a locked append"
                    )
                stream.seek(0, os.SEEK_END)
                original_size = stream.tell()
                if original_size != self._file_size:
                    raise LearningRecordStoreError(
                        "learning record store changed during a locked append"
                    )
                try:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                except OSError as exc:
                    try:
                        stream.seek(original_size)
                        stream.truncate(original_size)
                        stream.flush()
                        os.fsync(stream.fileno())
                    except OSError as rollback_exc:
                        raise LearningRecordStoreError(
                            "learning record append and rollback both failed"
                        ) from rollback_exc
                    raise LearningRecordStoreError(
                        "learning record append could not be committed"
                    ) from exc
            if not existed:
                self._fsync_parent()
        except LearningRecordStoreError:
            raise
        except OSError as exc:
            raise LearningRecordStoreError(
                "learning record store cannot be appended"
            ) from exc

    def _apply_locked(
        self, outbox_events: Sequence[Mapping[str, Any]]
    ) -> tuple[LearningRecordApplyResult, ...]:
        self._refresh_locked(repair_truncated_tail=True)
        recorded_at = _utc_text(self._now())
        original_index = deepcopy(self._event_index)
        envelopes, candidate_records, candidate_index = self._build_envelopes(
            outbox_events, recorded_at=recorded_at
        )
        self._write_envelopes_locked(envelopes)
        if envelopes:
            self._publish_replay(
                [*self._events, *envelopes],
                candidate_records,
                candidate_index,
                str(envelopes[-1]["hash"]),
                self._file_size
                + sum(len(_canonical_bytes(item)) + 1 for item in envelopes),
            )
        results: list[LearningRecordApplyResult] = []
        for raw_event in outbox_events:
            event = dict(raw_event)
            event_id = str(event["event_id"])
            envelope = self._event_index[event_id]
            target = _validate_target(event["target"])
            current = _component_from_records(self._records, target)
            if current is None:  # pragma: no cover - reducer guarantees creation.
                raise LearningRecordStoreError("applied learning target disappeared")
            results.append(
                LearningRecordApplyResult(
                    event_id=event_id,
                    applied=event_id not in original_index,
                    committed_target_version=int(envelope["target_version_after"]),
                    current_record=deepcopy(dict(current)),
                )
            )
        return tuple(results)

    def apply_outbox_event(
        self, event: Mapping[str, Any]
    ) -> LearningRecordApplyResult:
        """Atomically apply one session outbox event, with strict idempotency."""

        return self.apply_outbox_batch([event])[0]

    def apply_committed_evidence_event(
        self, event: Mapping[str, Any]
    ) -> LearningRecordApplyResult:
        """Atomically sequence one already-committed ordinary evidence event.

        A session outbox is committed before this call, so its CAS version can
        be stale when two sessions for the same learner/KC finish concurrently.
        This method allocates the current target version under the store's
        process lock.  The deterministic ``event_id`` and every authority/
        provenance field remain unchanged; only the transport CAS version is
        rebound.  Claimed-review outcomes deliberately cannot use this path.
        """

        validate_learning_outbox_event(event)
        if event.get("event_type") != "evidence_outcome_recorded":
            raise LearningRecordError(
                "committed evidence sequencing accepts ordinary evidence only"
            )
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            self._refresh_locked(repair_truncated_tail=True)
            self._reject_erased_target(event)
            event_id = str(event["event_id"])
            existing = self._event_index.get(event_id)
            if existing is not None:
                persisted = deepcopy(dict(existing["outbox_event"]))
                replayed = deepcopy(dict(event))
                persisted.pop("expected_version", None)
                replayed.pop("expected_version", None)
                if persisted != replayed:
                    raise LearningRecordConflictError(
                        "event_id was replayed with different learning content"
                    )
                target = _validate_target(event["target"])
                current = _component_from_records(self._records, target)
                if current is None:  # pragma: no cover - replay created it.
                    raise LearningRecordStoreError(
                        "idempotent committed evidence target disappeared"
                    )
                return LearningRecordApplyResult(
                    event_id=event_id,
                    applied=False,
                    committed_target_version=int(
                        existing["target_version_after"]
                    ),
                    current_record=deepcopy(dict(current)),
                )
            target = _validate_target(event["target"])
            current = _component_from_records(self._records, target)
            sequenced = deepcopy(dict(event))
            sequenced["expected_version"] = (
                0 if current is None else int(current["version"])
            )
            validate_learning_outbox_event(sequenced)
            return self._apply_locked([sequenced])[0]

    def apply_outbox_batch(
        self, events: Sequence[Mapping[str, Any]]
    ) -> tuple[LearningRecordApplyResult, ...]:
        """Validate and append a whole batch or leave the log unchanged."""

        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            raise LearningRecordError("learning outbox batch must be a sequence")
        if not events:
            return ()
        event_ids: list[str] = []
        for event in events:
            validate_learning_outbox_event(event)
            event_ids.append(str(event["event_id"]))
        if len(event_ids) != len(set(event_ids)):
            raise LearningRecordError(
                "learning outbox batch must not contain duplicate event_ids"
            )
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            return self._apply_locked(events)

    @staticmethod
    def _due_items(
        records: Mapping[str, Any], learner_key: str, now: datetime
    ) -> list[dict[str, Any]]:
        learner = records.get(learner_key)
        if not isinstance(learner, Mapping):
            return []
        result: list[dict[str, Any]] = []
        for namespace, components in learner["knowledge_components"].items():
            for kc_id, component in components.items():
                schedule = component["schedule"]
                claim_state: str | None = None
                review_id: str | None = None
                if schedule["state"] == "scheduled" and _parse_utc(
                    schedule["due_at_utc"], field="due_at_utc"
                ) <= now:
                    claim_state = "unclaimed"
                    review_id = _review_id(component)
                elif schedule["state"] == "in_progress" and _parse_utc(
                    schedule["lease_expires_at_utc"], field="lease_expires_at_utc"
                ) <= now:
                    claim_state = "expired"
                    review_id = str(schedule["active_review_id"])
                if review_id is None:
                    continue
                result.append(
                    {
                        "review_id": review_id,
                        "learner_key": learner_key,
                        "curriculum_namespace": namespace,
                        "knowledge_component_id": kc_id,
                        "source_ref_sha256": component["source_ref_sha256"],
                        "due_at_utc": schedule["due_at_utc"],
                        "expected_version": component["version"],
                        "claim_state": claim_state,
                    }
                )
        return sorted(
            result,
            key=lambda item: (
                item["due_at_utc"],
                item["curriculum_namespace"],
                item["knowledge_component_id"],
            ),
        )

    def list_due_reviews(
        self, learner_key: str, *, now_utc: datetime | None = None
    ) -> tuple[dict[str, Any], ...]:
        """Return due or lease-expired single-KC items; due-ness is derived."""

        key = _validate_learner_key(learner_key)
        now = self._now() if now_utc is None else _parse_utc(
            _utc_text(now_utc), field="now_utc"
        )
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            self._refresh_locked(repair_truncated_tail=True)
            if key in self._erasure_tombstones:
                return ()
            return tuple(deepcopy(self._due_items(self._records, key, now)))

    def _idempotent_special_result(
        self,
        *,
        event_id: str,
        learner_key: str,
        review_id: str,
        event_type: str,
    ) -> LearningRecordApplyResult | None:
        envelope = self._event_index.get(event_id)
        if envelope is None:
            return None
        event = envelope["outbox_event"]
        if (
            event["event_type"] != event_type
            or event["target"]["learner_key"] != learner_key
            or event["data"]["review_id"] != review_id
        ):
            raise LearningRecordConflictError(
                "idempotency key is already bound to another learning operation"
            )
        current = _component_from_records(self._records, event["target"])
        if current is None:  # pragma: no cover - a non-purged event has a target.
            raise LearningRecordStoreError("idempotent learning target disappeared")
        return LearningRecordApplyResult(
            event_id=event_id,
            applied=False,
            committed_target_version=int(envelope["target_version_after"]),
            current_record=deepcopy(dict(current)),
        )

    def claim_due_review(
        self,
        *,
        learner_key: str,
        review_id: str,
        expected_version: int,
        idempotency_key: str,
    ) -> LearningRecordApplyResult:
        """CAS-claim one due review using a fixed 15-minute durable lease."""

        key = _validate_learner_key(learner_key)
        if not isinstance(review_id, str) or _REVIEW_ID.fullmatch(review_id) is None:
            raise LearningRecordError("review_id is invalid")
        expected = _bounded_integer(expected_version, field="expected_version")
        idem = _safe_reference_id(idempotency_key, field="idempotency_key")
        event_id = _event_id(
            {
                "operation": "review_claimed",
                "learner_key": key,
                "review_id": review_id,
                "expected_version": expected,
                "idempotency_key": idem,
            }
        )
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            self._refresh_locked(repair_truncated_tail=True)
            if key in self._erasure_tombstones:
                raise LearningRecordConflictError(
                    "learner key was permanently erased"
                )
            replay = self._idempotent_special_result(
                event_id=event_id,
                learner_key=key,
                review_id=review_id,
                event_type="review_claimed",
            )
            if replay is not None:
                return replay
            now = self._now()
            item = next(
                (
                    candidate
                    for candidate in self._due_items(self._records, key, now)
                    if candidate["review_id"] == review_id
                ),
                None,
            )
            if item is None:
                raise LearningRecordConflictError(
                    "review_id is not currently due for this learner"
                )
            if item["expected_version"] != expected:
                raise LearningRecordConflictError(
                    "review claim expected_version is stale"
                )
            target = {
                field: item[field]
                for field in (
                    "learner_key",
                    "curriculum_namespace",
                    "knowledge_component_id",
                    "source_ref_sha256",
                )
            }
            lease_id = "lease_" + _canonical_sha256(
                {"event_id": event_id, "review_id": review_id}
            )
            event = {
                "schema": LEARNING_OUTBOX_EVENT_SCHEMA,
                "event_id": event_id,
                "event_type": "review_claimed",
                "target": target,
                "expected_version": expected,
                "occurred_at_utc": _utc_text(now),
                "data": {
                    "review_id": review_id,
                    "lease_id": lease_id,
                    "lease_expires_at_utc": _utc_text(
                        now + timedelta(seconds=REVIEW_LEASE_SECONDS)
                    ),
                },
            }
            return self._apply_locked([event])[0]

    def release_review_lease(
        self,
        *,
        learner_key: str,
        review_id: str,
        expected_version: int,
        idempotency_key: str,
        reason: str = "cancelled",
    ) -> LearningRecordApplyResult:
        """Release exactly the active server lease; stale releases fail CAS."""

        key = _validate_learner_key(learner_key)
        if not isinstance(review_id, str) or _REVIEW_ID.fullmatch(review_id) is None:
            raise LearningRecordError("review_id is invalid")
        if reason not in _LEASE_RELEASE_REASONS:
            raise LearningRecordError("review lease release reason is invalid")
        expected = _bounded_integer(expected_version, field="expected_version")
        idem = _safe_reference_id(idempotency_key, field="idempotency_key")
        event_id = _event_id(
            {
                "operation": "review_lease_released",
                "learner_key": key,
                "review_id": review_id,
                "expected_version": expected,
                "idempotency_key": idem,
            }
        )
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            self._refresh_locked(repair_truncated_tail=True)
            if key in self._erasure_tombstones:
                raise LearningRecordConflictError(
                    "learner key was permanently erased"
                )
            replay = self._idempotent_special_result(
                event_id=event_id,
                learner_key=key,
                review_id=review_id,
                event_type="review_lease_released",
            )
            if replay is not None:
                return replay
            active: dict[str, Any] | None = None
            for components in self._records.get(key, {}).get(
                "knowledge_components", {}
            ).values():
                for component in components.values():
                    if component["schedule"]["active_review_id"] == review_id:
                        active = component
                        break
                if active is not None:
                    break
            if active is None:
                raise LearningRecordConflictError("review_id has no active lease")
            if active["version"] != expected:
                raise LearningRecordConflictError(
                    "review release expected_version is stale"
                )
            schedule = active["schedule"]
            event = {
                "schema": LEARNING_OUTBOX_EVENT_SCHEMA,
                "event_id": event_id,
                "event_type": "review_lease_released",
                "target": {
                    field: active[field]
                    for field in (
                        "learner_key",
                        "curriculum_namespace",
                        "knowledge_component_id",
                        "source_ref_sha256",
                    )
                },
                "expected_version": expected,
                "occurred_at_utc": _utc_text(self._now()),
                "data": {
                    "review_id": review_id,
                    "lease_id": schedule["active_lease_id"],
                    "reason": reason,
                },
            }
            return self._apply_locked([event])[0]

    def cancel_source_observations(
        self,
        source_observation_ids: Sequence[str],
        *,
        reason: str = "source_deleted",
    ) -> tuple[LearningRecordApplyResult, ...]:
        """Suspend every current schedule derived from deleted/revoked evidence."""

        if reason not in _CANCELLATION_REASONS:
            raise LearningRecordError("source cancellation reason is invalid")
        if isinstance(source_observation_ids, (str, bytes)) or not isinstance(
            source_observation_ids, Sequence
        ):
            raise LearningRecordError("source_observation_ids must be a sequence")
        source_ids = {
            _safe_reference_id(value, field="source_observation_id")
            for value in source_observation_ids
        }
        if not source_ids:
            return ()
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            self._refresh_locked(repair_truncated_tail=True)
            now = _utc_text(self._now())
            events: list[dict[str, Any]] = []
            for learner in self._records.values():
                if learner["learner_key"] in self._erasure_tombstones:
                    continue
                for components in learner["knowledge_components"].values():
                    for component in components.values():
                        schedule = component["schedule"]
                        source_id = schedule["source_observation_id"]
                        if source_id not in source_ids or schedule["state"] in {
                            "suspended",
                            "retired",
                        }:
                            continue
                        target = {
                            field: component[field]
                            for field in (
                                "learner_key",
                                "curriculum_namespace",
                                "knowledge_component_id",
                                "source_ref_sha256",
                            )
                        }
                        event = {
                            "schema": LEARNING_OUTBOX_EVENT_SCHEMA,
                            "event_id": _event_id(
                                {
                                    "operation": "source_cancelled",
                                    "target": target,
                                    "source_observation_id": source_id,
                                    "reason": reason,
                                }
                            ),
                            "event_type": "source_cancelled",
                            "target": target,
                            "expected_version": component["version"],
                            "occurred_at_utc": now,
                            "data": {
                                "source_observation_id": source_id,
                                "reason": reason,
                            },
                        }
                        events.append(event)
            return self._apply_locked(events) if events else ()

    def purge_learner(self, learner_key: str) -> dict[str, int]:
        """Physically compact one learner's events for a data-rights purge."""

        key = _validate_learner_key(learner_key)
        with self._thread_lock, self._process_lock():
            self._refresh_tombstones_locked()
            self._refresh_locked(repair_truncated_tail=True)
            existing_tombstone = self._erasure_tombstones.get(key)
            had_record = key in self._records
            if existing_tombstone is None:
                tombstones = deepcopy(self._erasure_tombstones)
                tombstones[key] = {
                    "purged_at_utc": _utc_text(self._now()),
                    "generation": 1,
                }
                # The erasure fence is committed first.  If compaction fails,
                # no read or delayed outbox can expose/recreate this learner;
                # a retry can safely finish physical compaction.
                self._write_tombstones_locked(tombstones)
            retained_outbox = [
                deepcopy(event["outbox_event"])
                for event in self._events
                if event["outbox_event"]["target"]["learner_key"] != key
            ]
            removed = len(self._events) - len(retained_outbox)
            if removed == 0:
                return {
                    "learners": 1 if had_record or existing_tombstone is None else 0,
                    "events": 0,
                    "erasure_fence": 1,
                }
            recorded_times = [
                str(event["recorded_at_utc"])
                for event in self._events
                if event["outbox_event"]["target"]["learner_key"] != key
            ]
            records: dict[str, dict[str, Any]] = {}
            index: dict[str, dict[str, Any]] = {}
            envelopes: list[dict[str, Any]] = []
            previous_hash: str | None = None
            for seq, (event, recorded_at) in enumerate(
                zip(retained_outbox, recorded_times), 1
            ):
                target_version, learner_version = _reduce_event(records, event)
                envelope: dict[str, Any] = {
                    "schema": LEARNING_STORE_EVENT_SCHEMA,
                    "seq": seq,
                    "recorded_at_utc": recorded_at,
                    "previous_hash": previous_hash,
                    "outbox_event": event,
                    "outbox_fingerprint": _canonical_sha256(event),
                    "target_version_after": target_version,
                    "learner_version_after": learner_version,
                }
                envelope["hash"] = _envelope_hash(envelope)
                envelopes.append(envelope)
                index[event["event_id"]] = deepcopy(envelope)
                previous_hash = envelope["hash"]
            payload = b"".join(_canonical_bytes(item) + b"\n" for item in envelopes)
            self._atomic_replace(payload)
            self._publish_replay(
                envelopes,
                records,
                index,
                previous_hash,
                len(payload),
            )
            return {"learners": 1, "events": removed, "erasure_fence": 1}

    def _atomic_replace(self, payload: bytes) -> None:
        self._atomic_replace_path(self.path, payload)

    def _atomic_replace_path(self, target: Path, payload: bytes) -> None:
        ensure_private_directory(self.path.parent)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.replace-",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            ensure_private_file(target)
            self._fsync_parent()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _fsync_parent(self) -> None:
        try:
            descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise LearningRecordStoreError(
                "learning record parent directory cannot be flushed"
            ) from exc


__all__ = [
    "LEARNING_OUTBOX_EVENT_SCHEMA",
    "LEARNING_RECORD_SCHEMA",
    "LEARNING_STORE_EVENT_SCHEMA",
    "REVIEW_INTERVAL_DAYS",
    "REVIEW_LEASE_SECONDS",
    "LearningRecordApplyResult",
    "LearningRecordConflictError",
    "LearningRecordError",
    "LearningRecordStore",
    "LearningRecordStoreError",
    "LearningRecordStoreRecovery",
    "build_learning_evidence_outbox_event",
    "curriculum_namespace_for_source_ref",
    "learning_record_target",
    "mint_learner_key",
    "validate_learning_outbox_event",
    "validate_learning_record",
]
