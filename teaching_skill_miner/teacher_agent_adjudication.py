"""Bounded local adjudication core for assessment evidence.

This module is intentionally independent from the dashboard and student model.
Review items can only be materialized from a server-owned evidence resolver; a
client may name and hash an evidence record, but it cannot supply the evidence
payload itself.  Every state transition appends an immutable item version and a
hash-chained audit receipt.

The queue is an in-memory concurrency/reducer core, not a durable service.  An
integration must persist and recover its versions and audit receipts before it
can claim restart durability; this module deliberately makes no such claim.

The standalone desktop product has no authenticated teacher identity and keeps
using the explicit ``local_operator_not_authenticated`` placeholder.  A
gateway-hosted integration may opt into a verifier-backed, hash-only
``authenticated_teacher_server_authorized`` actor; raw identity claims are
never part of this contract.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import re
import secrets
import threading
from typing import Any, Callable, Mapping, Sequence

from .teacher_agent_authority import (
    AUTHENTICATED_TEACHER_ACTOR,
    AUTHORITY_ASSURANCE,
    TEACHER_AUTHORITY_RECEIPT_SCHEMA,
    authenticated_teacher_actor,
    validate_teacher_authority_verification_receipt,
)


ADJUDICATION_ITEM_SCHEMA = "teaching_skill_miner.teacher_agent_adjudication_item.v1"
ADJUDICATION_AUDIT_SCHEMA = "teaching_skill_miner.teacher_agent_adjudication_audit.v1"
ADJUDICATION_INSTRUCTION_SCHEMA = (
    "teaching_skill_miner.student_model_adjudication_instruction.v1"
)
LOCAL_OPERATOR_IDENTITY = "local_operator_not_authenticated"
CLAIM_LEASE_SECONDS = 15 * 60
MAX_TARGET_KNOWLEDGE_COMPONENTS = 4

DECISIONS = frozenset({"approve", "correct", "abstain"})
REVIEW_REASONS = frozenset(
    {
        "automatic_low_confidence",
        "authority_required",
        "learner_dispute",
        "rubric_conflict",
        "manual_quality_review",
    }
)
DECISION_REASONS = frozenset(
    {
        "assessment_confirmed",
        "signal_misclassified",
        "alignment_misclassified",
        "knowledge_component_misaligned",
        "focus_dimension_misaligned",
        "insufficient_evidence",
        "rubric_ambiguous",
        "authority_unavailable",
        "conflict_unresolved",
    }
)
CORRECTION_FIELDS = frozenset(
    {"signal", "answer_alignment", "focus_dimension", "target_kc_ids"}
)
CORRECTION_SIGNALS = frozenset({"correct", "partial", "misconception"})
CORRECTION_ALIGNMENTS = frozenset(
    {"aligned", "partially_aligned", "contradicted", "ambiguous"}
)
CORRECTION_DIMENSIONS = frozenset(
    {"prerequisite", "conceptual", "procedural", "transfer"}
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SAFE_KC_ID = re.compile(r"^kc_[a-z0-9][a-z0-9_-]{2,80}$")
_SAFE_IDEMPOTENCY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$")
_AUTHORITY_PATHS = {
    "claim": "api/adjudication/claim",
    "decide": "api/adjudication/decide",
}


class TeacherAgentAdjudicationError(ValueError):
    """Raised when an adjudication request violates its public contract."""


class AdjudicationConflictError(TeacherAgentAdjudicationError):
    """Raised when a lease, CAS, or terminal decision conflicts."""


class AdjudicationEvidenceError(TeacherAgentAdjudicationError):
    """Raised when authoritative evidence is absent or fails integrity checks."""


class AdjudicationEvidenceDeletedError(AdjudicationEvidenceError):
    """Raised when the authoritative evidence record no longer exists."""


class AdjudicationNotFoundError(TeacherAgentAdjudicationError):
    """Raised when a review item does not exist."""


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
        raise TeacherAgentAdjudicationError(
            "adjudication values must contain canonical JSON"
        ) from exc


def canonical_sha256(value: Any) -> str:
    """Return the canonical SHA-256 used by evidence and audit bindings."""

    return sha256(_canonical_bytes(value)).hexdigest()


def authoritative_evidence_sha256(evidence: Mapping[str, Any]) -> str:
    """Hash an authoritative evidence record without its self-hash field."""

    if not isinstance(evidence, Mapping):
        raise AdjudicationEvidenceError("authoritative evidence must be an object")
    material = deepcopy(dict(evidence))
    material.pop("evidence_sha256", None)
    return canonical_sha256(material)


def _required_string(value: Any, *, field: str, maximum: int = 160) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise TeacherAgentAdjudicationError(
            f"{field} must be a non-empty trimmed string <= {maximum} chars"
        )
    return value


def _safe_id(value: Any, *, field: str) -> str:
    result = _required_string(value, field=field)
    if _SAFE_ID.fullmatch(result) is None:
        raise TeacherAgentAdjudicationError(f"{field} is invalid")
    return result


def _digest(value: Any, *, field: str) -> str:
    result = _required_string(value, field=field, maximum=64)
    if _DIGEST.fullmatch(result) is None:
        raise TeacherAgentAdjudicationError(f"{field} must be a lowercase SHA-256")
    return result


def _version(value: Any, *, field: str = "expected_version") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TeacherAgentAdjudicationError(f"{field} must be an integer >= 1")
    return value


def _round_number(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdjudicationEvidenceError(
            "authoritative evidence source.round_number must be an integer >= 0"
        )
    return value


def _idempotency_key(value: Any) -> str:
    result = _required_string(value, field="idempotency_key")
    if _SAFE_IDEMPOTENCY.fullmatch(result) is None:
        raise TeacherAgentAdjudicationError(
            "idempotency_key must be 8-160 safe ASCII characters"
        )
    return result


def _format_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TeacherAgentAdjudicationError(
            "the adjudication clock must return a timezone-aware datetime"
        )
    utc = value.astimezone(timezone.utc)
    if utc.utcoffset() != timedelta(0):  # pragma: no cover - astimezone guarantees it.
        raise TeacherAgentAdjudicationError(
            "the adjudication clock must resolve to UTC"
        )
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - only internally generated values.
        raise TeacherAgentAdjudicationError(
            "stored lease timestamp is invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise TeacherAgentAdjudicationError("stored lease timestamp is not UTC")
    return parsed.astimezone(timezone.utc)


def _actor(identity: str) -> dict[str, Any]:
    return {
        "identity": identity,
        "authenticated": False,
        "teacher_identity_claimed": False,
    }


def _authority_actor_and_receipt(
    value: Mapping[str, Any] | None,
    *,
    operation: str,
    idempotency_key: str,
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if value is None:
        return _actor(LOCAL_OPERATOR_IDENTITY), None
    if validator is None:
        raise TeacherAgentAdjudicationError(
            "authenticated teacher authority is unavailable in standalone mode"
        )
    try:
        receipt = validate_teacher_authority_verification_receipt(validator(value))
        actor = authenticated_teacher_actor(receipt)
    except Exception as exc:
        raise TeacherAgentAdjudicationError(
            "authenticated teacher authority receipt is invalid"
        ) from exc
    if (
        receipt.get("schema") != TEACHER_AUTHORITY_RECEIPT_SCHEMA
        or receipt.get("method") != "POST"
        or receipt.get("path") != _AUTHORITY_PATHS.get(operation)
        or receipt.get("idempotency_key_sha256")
        != sha256(idempotency_key.encode("utf-8")).hexdigest()
    ):
        raise TeacherAgentAdjudicationError(
            "authenticated teacher authority request binding is invalid"
        )
    return actor, receipt


def _stable_actor_binding(actor: Mapping[str, Any]) -> dict[str, Any]:
    if actor.get("identity") == AUTHENTICATED_TEACHER_ACTOR:
        return {
            "identity": AUTHENTICATED_TEACHER_ACTOR,
            "authenticated": True,
            "teacher_identity_claimed": True,
            "principal_sha256": actor.get("principal_sha256"),
            "roles_sha256": actor.get("roles_sha256"),
            "assurance": AUTHORITY_ASSURANCE,
        }
    return deepcopy(dict(actor))


def authenticated_instruction_authority_basis(
    instruction: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact receipt-signing basis for an authenticated correction."""

    result = deepcopy(dict(instruction))
    result.pop("instruction_sha256", None)
    authority = result.get("authority")
    if not isinstance(authority, Mapping):
        raise TeacherAgentAdjudicationError("instruction authority is missing")
    normalized = deepcopy(dict(authority))
    normalized["authoritative_for_mastery_update"] = False
    normalized["requires_downstream_authority_revalidation"] = True
    normalized["server_revalidation_receipt"] = None
    result["authority"] = normalized
    result["mutates_student_model"] = False
    result["mastery_update_authorized"] = False
    return result


def authorize_authenticated_instruction(
    instruction: Mapping[str, Any],
    *,
    revalidation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the server revalidation receipt into a mastery-authorized instruction."""

    result = deepcopy(dict(instruction))
    if result.get("operation") != "supersede_and_replay":
        raise TeacherAgentAdjudicationError(
            "only a corrected instruction can receive teacher revalidation"
        )
    authority = result.get("authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("identity") != AUTHENTICATED_TEACHER_ACTOR
    ):
        raise TeacherAgentAdjudicationError(
            "authenticated instruction authority is missing"
        )
    normalized = deepcopy(dict(authority))
    normalized["authoritative_for_mastery_update"] = True
    normalized["requires_downstream_authority_revalidation"] = False
    normalized["server_revalidation_receipt"] = deepcopy(dict(revalidation_receipt))
    result["authority"] = normalized
    result["mutates_student_model"] = True
    result["mastery_update_authorized"] = True
    result["requires_explicit_apply"] = True
    result.pop("instruction_sha256", None)
    result["instruction_sha256"] = canonical_sha256(result)
    return result


def _snapshot_seal(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdjudicationEvidenceError(
            f"authoritative evidence {field} must be an object"
        )
    # Only the content address and non-sensitive structural coordinates enter a
    # review item.  The full learner snapshot remains in its authoritative
    # store, so public queue projections cannot disclose free-form responses.
    result: dict[str, Any] = {"content_sha256": canonical_sha256(dict(value))}
    schema = value.get("schema")
    if isinstance(schema, str) and schema and len(schema) <= 160:
        result["schema"] = schema
    state_version = value.get("version", value.get("model_version"))
    if isinstance(state_version, int) and not isinstance(state_version, bool):
        if state_version >= 0:
            result["state_version"] = state_version
    return result


def _target_kcs(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TeacherAgentAdjudicationError(f"{field} must be an array")
    if not 1 <= len(value) <= MAX_TARGET_KNOWLEDGE_COMPONENTS:
        raise TeacherAgentAdjudicationError(
            f"{field} must contain 1-{MAX_TARGET_KNOWLEDGE_COMPONENTS} IDs"
        )
    result: list[str] = []
    for index, raw in enumerate(value):
        kc_id = _required_string(raw, field=f"{field}[{index}]", maximum=84)
        if _SAFE_KC_ID.fullmatch(kc_id) is None:
            raise TeacherAgentAdjudicationError(f"{field}[{index}] is invalid")
        if kc_id in result:
            raise TeacherAgentAdjudicationError(f"{field} must contain unique IDs")
        result.append(kc_id)
    return tuple(result)


def _validated_evidence(
    raw: Any, *, evidence_id: str, evidence_sha256: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AdjudicationEvidenceError("authoritative evidence was not found")
    evidence = deepcopy(dict(raw))
    resolved_id = _safe_id(evidence.get("evidence_id"), field="evidence.evidence_id")
    if resolved_id != evidence_id:
        raise AdjudicationEvidenceError("authoritative evidence ID does not match")
    declared_hash = _digest(
        evidence.get("evidence_sha256"), field="evidence.evidence_sha256"
    )
    computed_hash = authoritative_evidence_sha256(evidence)
    if declared_hash != computed_hash:
        raise AdjudicationEvidenceError(
            "authoritative evidence failed its self-hash integrity check"
        )
    if evidence_sha256 != declared_hash:
        raise AdjudicationEvidenceError(
            "requested evidence_sha256 does not match authoritative evidence"
        )

    source = evidence.get("source")
    if not isinstance(source, Mapping):
        raise AdjudicationEvidenceError(
            "authoritative evidence source must be an object"
        )
    normalized_source = {
        "session_id": _safe_id(source.get("session_id"), field="source.session_id"),
        "round_number": _round_number(source.get("round_number")),
        "action_id": _safe_id(source.get("action_id"), field="source.action_id"),
        "question_id": _safe_id(source.get("question_id"), field="source.question_id"),
        "history_event_sha256": _digest(
            source.get("history_event_sha256"),
            field="source.history_event_sha256",
        ),
    }
    target_kcs = _target_kcs(
        evidence.get("target_kc_ids"), field="evidence.target_kc_ids"
    )
    return {
        "source": normalized_source,
        "target_kc_ids": list(target_kcs),
        "original": {
            "assessment_id": _safe_id(
                evidence.get("original_assessment_id"),
                field="evidence.original_assessment_id",
            ),
            "assessment_sha256": _digest(
                evidence.get("original_assessment_sha256"),
                field="evidence.original_assessment_sha256",
            ),
            "evidence_id": resolved_id,
            "evidence_sha256": declared_hash,
            "rubric_id": _safe_id(
                evidence.get("rubric_id"), field="evidence.rubric_id"
            ),
            "rubric_authority_sha256": _digest(
                evidence.get("rubric_authority_sha256"),
                field="evidence.rubric_authority_sha256",
            ),
        },
        "before_snapshot": _snapshot_seal(
            evidence.get("before_snapshot"), field="before_snapshot"
        ),
        "after_snapshot": _snapshot_seal(
            evidence.get("after_snapshot"), field="after_snapshot"
        ),
    }


def _validate_correction(
    value: Any, *, item_target_kcs: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise TeacherAgentAdjudicationError(
            "correct decisions require a non-empty correction object"
        )
    unexpected = set(value) - CORRECTION_FIELDS
    if unexpected:
        raise TeacherAgentAdjudicationError(
            "correction contains unsupported fields: " + ", ".join(sorted(unexpected))
        )
    result: dict[str, Any] = {}
    if "signal" in value:
        signal = value["signal"]
        if signal not in CORRECTION_SIGNALS:
            raise TeacherAgentAdjudicationError("correction.signal is invalid")
        result["signal"] = signal
    if "answer_alignment" in value:
        alignment = value["answer_alignment"]
        if alignment not in CORRECTION_ALIGNMENTS:
            raise TeacherAgentAdjudicationError(
                "correction.answer_alignment is invalid"
            )
        result["answer_alignment"] = alignment
    if "focus_dimension" in value:
        dimension = value["focus_dimension"]
        if dimension not in CORRECTION_DIMENSIONS:
            raise TeacherAgentAdjudicationError("correction.focus_dimension is invalid")
        result["focus_dimension"] = dimension
    if "target_kc_ids" in value:
        targets = _target_kcs(value["target_kc_ids"], field="correction.target_kc_ids")
        if not set(targets).issubset(set(item_target_kcs)):
            raise TeacherAgentAdjudicationError(
                "correction.target_kc_ids must be a subset of the review targets"
            )
        result["target_kc_ids"] = list(targets)
    if not result:
        raise TeacherAgentAdjudicationError(
            "correct decisions require at least one supported correction"
        )
    return result


def public_review_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded public projection of an already sealed item version."""

    if not isinstance(item, Mapping):
        raise TeacherAgentAdjudicationError("review item must be an object")
    allowed = {
        "schema",
        "item_id",
        "version",
        "status",
        "created_at",
        "updated_at",
        "source",
        "target_kc_ids",
        "target_scope",
        "original",
        "before_snapshot",
        "after_snapshot",
        "review_reason",
        "lease",
        "decision",
        "cancellation",
        "previous_version_sha256",
        "version_sha256",
    }
    return deepcopy({key: item[key] for key in allowed if key in item})


def reduce_adjudication(item: Mapping[str, Any]) -> dict[str, Any]:
    """Produce, but never apply, a student-model replay/supersede instruction."""

    projected = public_review_item(item)
    status = projected.get("status")
    decision = projected.get("decision")
    kind: Any = None
    if status == "decided" and isinstance(decision, Mapping):
        kind = decision.get("kind")
        if kind == "approve":
            operation = "replay_original_assessment"
        elif kind == "correct":
            operation = "supersede_and_replay"
        elif kind == "abstain":
            operation = "do_not_replay"
        else:  # pragma: no cover - queue construction prevents this.
            raise TeacherAgentAdjudicationError("stored decision kind is invalid")
    elif status == "cancelled":
        operation = "cancel_adjudication"
    else:
        raise TeacherAgentAdjudicationError(
            "only decided or cancelled items can produce an instruction"
        )

    original = projected.get("original")
    if not isinstance(original, Mapping):
        raise TeacherAgentAdjudicationError("review item original reference is missing")
    decision_actor = decision.get("actor") if isinstance(decision, Mapping) else None
    authenticated = (
        isinstance(decision_actor, Mapping)
        and decision_actor.get("identity") == AUTHENTICATED_TEACHER_ACTOR
        and decision_actor.get("authenticated") is True
    )
    if authenticated:
        authority_receipt = decision.get("authority_receipt")
        if not isinstance(authority_receipt, Mapping):
            raise TeacherAgentAdjudicationError(
                "authenticated decision authority receipt is missing"
            )
        authority: dict[str, Any] = {
            **deepcopy(dict(decision_actor)),
            "gateway_verification_receipt_sha256": authority_receipt.get(
                "receipt_sha256"
            ),
            "personal_non_repudiation": False,
            "authoritative_for_mastery_update": False,
            "requires_downstream_authority_revalidation": True,
            "server_revalidation_receipt": None,
        }
    else:
        authority = {
            "identity": LOCAL_OPERATOR_IDENTITY,
            "authenticated": False,
            "teacher_identity_claimed": False,
            "authoritative_for_mastery_update": False,
            "requires_downstream_authority_revalidation": True,
        }
    instruction: dict[str, Any] = {
        "schema": ADJUDICATION_INSTRUCTION_SCHEMA,
        "instruction_id": "adjinst_"
        + canonical_sha256(
            {
                "item_id": projected.get("item_id"),
                "version_sha256": projected.get("version_sha256"),
            }
        )[:24],
        "review_item_id": projected.get("item_id"),
        "review_item_version": projected.get("version"),
        "review_item_version_sha256": projected.get("version_sha256"),
        "operation": operation,
        "supersedes_evidence_id": original.get("evidence_id"),
        "evidence_sha256": original.get("evidence_sha256"),
        "target_kc_ids": deepcopy(projected.get("target_kc_ids")),
        "correction": (
            deepcopy(decision.get("correction"))
            if isinstance(decision, Mapping) and kind == "correct"
            else None
        ),
        "authority": authority,
        "mutates_student_model": False,
        "mastery_update_authorized": False,
        "requires_explicit_apply": True,
    }
    instruction["instruction_sha256"] = canonical_sha256(instruction)
    return instruction


class TeacherAgentAdjudicationQueue:
    """Thread-safe, non-durable core with immutable versions and CAS transitions."""

    def __init__(
        self,
        *,
        evidence_resolver: Callable[[str], Mapping[str, Any] | None],
        clock: Callable[[], datetime] | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if not callable(evidence_resolver):
            raise TeacherAgentAdjudicationError("evidence_resolver must be callable")
        self._evidence_resolver = evidence_resolver
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(32))
        self._lock = threading.RLock()
        self._versions: dict[str, list[dict[str, Any]]] = {}
        self._evidence_items: dict[str, str] = {}
        self._audit: list[dict[str, Any]] = []
        self._idempotency: dict[tuple[str, str], dict[str, Any]] = {}

    def _now(self) -> tuple[datetime, str]:
        raw = self._clock()
        timestamp = _format_timestamp(raw)
        return raw.astimezone(timezone.utc), timestamp

    def _resolve(self, *, evidence_id: str, evidence_sha256: str) -> dict[str, Any]:
        try:
            raw = self._evidence_resolver(evidence_id)
        except Exception as exc:
            raise AdjudicationEvidenceError(
                "authoritative evidence resolver failed closed"
            ) from exc
        if raw is None:
            raise AdjudicationEvidenceDeletedError("authoritative evidence was deleted")
        return _validated_evidence(
            raw, evidence_id=evidence_id, evidence_sha256=evidence_sha256
        )

    def _latest(self, item_id: str) -> dict[str, Any]:
        versions = self._versions.get(item_id)
        if not versions:
            raise AdjudicationNotFoundError("adjudication item was not found")
        return versions[-1]

    def _check_cas(self, item: Mapping[str, Any], expected_version: int) -> None:
        expected = _version(expected_version)
        if item.get("version") != expected:
            raise AdjudicationConflictError(
                f"expected_version {expected} does not match current version "
                f"{item.get('version')}"
            )

    def _idempotent_replay(
        self, *, operation: str, key: str, request_sha256: str
    ) -> dict[str, Any] | None:
        cached = self._idempotency.get((operation, key))
        if cached is None:
            return None
        if cached["request_sha256"] != request_sha256:
            raise AdjudicationConflictError(
                "idempotency_key was already used with different adjudication input"
            )
        return deepcopy(cached["response"])

    def _remember_idempotency(
        self,
        *,
        operation: str,
        key: str,
        request_sha256: str,
        response: Mapping[str, Any],
    ) -> None:
        self._idempotency[(operation, key)] = {
            "request_sha256": request_sha256,
            "response": deepcopy(dict(response)),
        }

    def _append_audit(
        self,
        *,
        event_type: str,
        item: Mapping[str, Any],
        from_version: int | None,
        actor: Mapping[str, Any],
        request_sha256: str,
        occurred_at: str,
    ) -> dict[str, Any]:
        prior = self._audit[-1]["receipt_sha256"] if self._audit else None
        receipt: dict[str, Any] = {
            "schema": ADJUDICATION_AUDIT_SCHEMA,
            "seq": len(self._audit) + 1,
            "event_type": event_type,
            "item_id": item["item_id"],
            "from_version": from_version,
            "to_version": item["version"],
            "result_version_sha256": item["version_sha256"],
            "actor": deepcopy(dict(actor)),
            "occurred_at": occurred_at,
            "request_sha256": request_sha256,
            "prior_receipt_sha256": prior,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        self._audit.append(receipt)
        return receipt

    def _append_version(
        self,
        *,
        item_id: str,
        base: Mapping[str, Any],
        changes: Mapping[str, Any],
        occurred_at: str,
        event_type: str,
        actor: Mapping[str, Any],
        request_sha256: str,
    ) -> dict[str, Any]:
        previous = self._versions.get(item_id, [])
        from_version = int(base["version"]) if previous else None
        version = deepcopy(dict(base))
        version.update(deepcopy(dict(changes)))
        if previous:
            version["version"] = int(previous[-1]["version"]) + 1
            version["previous_version_sha256"] = previous[-1]["version_sha256"]
        else:
            version["version"] = 1
            version["previous_version_sha256"] = None
        version["updated_at"] = occurred_at
        version.pop("version_sha256", None)
        version["version_sha256"] = canonical_sha256(version)
        self._versions.setdefault(item_id, []).append(version)
        self._append_audit(
            event_type=event_type,
            item=version,
            from_version=from_version,
            actor=actor,
            request_sha256=request_sha256,
            occurred_at=occurred_at,
        )
        return version

    def enqueue(
        self,
        *,
        evidence_id: str,
        evidence_sha256: str,
        idempotency_key: str,
        review_reason: str = "automatic_low_confidence",
    ) -> dict[str, Any]:
        """Create a pending item from server-resolved evidence only."""

        safe_evidence_id = _safe_id(evidence_id, field="evidence_id")
        safe_evidence_hash = _digest(evidence_sha256, field="evidence_sha256")
        key = _idempotency_key(idempotency_key)
        if review_reason not in REVIEW_REASONS:
            raise TeacherAgentAdjudicationError("review_reason is invalid")
        request = {
            "evidence_id": safe_evidence_id,
            "evidence_sha256": safe_evidence_hash,
            "review_reason": review_reason,
        }
        request_hash = canonical_sha256(request)
        with self._lock:
            replay = self._idempotent_replay(
                operation="enqueue", key=key, request_sha256=request_hash
            )
            if replay is not None:
                return replay
            evidence = self._resolve(
                evidence_id=safe_evidence_id,
                evidence_sha256=safe_evidence_hash,
            )
            if safe_evidence_id in self._evidence_items:
                raise AdjudicationConflictError(
                    "authoritative evidence already has an adjudication item"
                )
            _now, occurred_at = self._now()
            item_id = (
                "adj_"
                + canonical_sha256(
                    {
                        "evidence_id": safe_evidence_id,
                        "evidence_sha256": safe_evidence_hash,
                        "source": evidence["source"],
                    }
                )[:24]
            )
            base: dict[str, Any] = {
                "schema": ADJUDICATION_ITEM_SCHEMA,
                "item_id": item_id,
                "version": 0,
                "status": "pending",
                "created_at": occurred_at,
                "updated_at": occurred_at,
                "source": evidence["source"],
                "target_kc_ids": evidence["target_kc_ids"],
                "target_scope": (
                    "exact_one"
                    if len(evidence["target_kc_ids"]) == 1
                    else "bounded_set"
                ),
                "original": evidence["original"],
                "before_snapshot": evidence["before_snapshot"],
                "after_snapshot": evidence["after_snapshot"],
                "review_reason": review_reason,
                "lease": None,
                "decision": None,
                "cancellation": None,
            }
            version = self._append_version(
                item_id=item_id,
                base=base,
                changes={},
                occurred_at=occurred_at,
                event_type="review.enqueued",
                actor=_actor("server_evidence_registry"),
                request_sha256=request_hash,
            )
            self._evidence_items[safe_evidence_id] = item_id
            response = public_review_item(version)
            self._remember_idempotency(
                operation="enqueue",
                key=key,
                request_sha256=request_hash,
                response=response,
            )
            return deepcopy(response)

    def get(self, item_id: str) -> dict[str, Any]:
        """Return the latest public item projection."""

        safe_item_id = _safe_id(item_id, field="item_id")
        with self._lock:
            return public_review_item(self._latest(safe_item_id))

    def history(self, item_id: str) -> tuple[dict[str, Any], ...]:
        """Return immutable public versions in append order."""

        safe_item_id = _safe_id(item_id, field="item_id")
        with self._lock:
            self._latest(safe_item_id)
            return tuple(
                public_review_item(value) for value in self._versions[safe_item_id]
            )

    def list_items(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(
                public_review_item(versions[-1])
                for _item_id, versions in sorted(self._versions.items())
            )

    @property
    def audit_receipts(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(deepcopy(self._audit))

    def claim(
        self,
        item_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
        authority_receipt: Mapping[str, Any] | None = None,
        authority_validator: (
            Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
        ) = None,
    ) -> dict[str, Any]:
        """Claim an item for exactly fifteen minutes using optimistic CAS."""

        safe_item_id = _safe_id(item_id, field="item_id")
        expected = _version(expected_version)
        key = _idempotency_key(idempotency_key)
        actor, _verified_authority = _authority_actor_and_receipt(
            authority_receipt,
            operation="claim",
            idempotency_key=key,
            validator=authority_validator,
        )
        request = {
            "item_id": safe_item_id,
            "expected_version": expected,
            "actor": _stable_actor_binding(actor),
        }
        request_hash = canonical_sha256(request)
        with self._lock:
            replay = self._idempotent_replay(
                operation="claim", key=key, request_sha256=request_hash
            )
            if replay is not None:
                return replay
            item = self._latest(safe_item_id)
            self._check_cas(item, expected)
            if item["status"] in {"decided", "cancelled"}:
                raise AdjudicationConflictError(
                    "terminal adjudication item cannot be claimed"
                )
            now, occurred_at = self._now()
            lease = item.get("lease")
            if (
                isinstance(lease, Mapping)
                and _parse_timestamp(str(lease["expires_at"])) > now
            ):
                raise AdjudicationConflictError("adjudication item is already claimed")
            token = _required_string(
                self._token_factory(), field="generated claim token", maximum=512
            )
            token_hash = sha256(token.encode("utf-8")).hexdigest()
            expires_at = _format_timestamp(now + timedelta(seconds=CLAIM_LEASE_SECONDS))
            version = self._append_version(
                item_id=safe_item_id,
                base=item,
                changes={
                    "status": "claimed",
                    "lease": {
                        "actor": actor,
                        "claimed_at": occurred_at,
                        "expires_at": expires_at,
                        "claim_token_sha256": token_hash,
                    },
                },
                occurred_at=occurred_at,
                event_type="review.claimed",
                actor=actor,
                request_sha256=request_hash,
            )
            response = {"item": public_review_item(version), "claim_token": token}
            self._remember_idempotency(
                operation="claim",
                key=key,
                request_sha256=request_hash,
                response=response,
            )
            return deepcopy(response)

    def decide(
        self,
        item_id: str,
        *,
        expected_version: int,
        claim_token: str,
        evidence_id: str,
        evidence_sha256: str,
        idempotency_key: str,
        decision: str,
        reason_code: str,
        correction: Mapping[str, Any] | None = None,
        authority_receipt: Mapping[str, Any] | None = None,
        authority_validator: (
            Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
        ) = None,
        instruction_authorizer: (
            Callable[
                [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
                Mapping[str, Any],
            ]
            | None
        ) = None,
    ) -> dict[str, Any]:
        """Append one approve/correct/abstain decision, never mutate a model."""

        safe_item_id = _safe_id(item_id, field="item_id")
        expected = _version(expected_version)
        token = _required_string(claim_token, field="claim_token", maximum=512)
        safe_evidence_id = _safe_id(evidence_id, field="evidence_id")
        safe_evidence_hash = _digest(evidence_sha256, field="evidence_sha256")
        key = _idempotency_key(idempotency_key)
        actor, verified_authority = _authority_actor_and_receipt(
            authority_receipt,
            operation="decide",
            idempotency_key=key,
            validator=authority_validator,
        )
        if decision not in DECISIONS:
            raise TeacherAgentAdjudicationError(
                "decision must be approve, correct, or abstain"
            )
        if reason_code not in DECISION_REASONS:
            raise TeacherAgentAdjudicationError("reason_code is invalid")
        if decision == "approve" and reason_code != "assessment_confirmed":
            raise TeacherAgentAdjudicationError(
                "approve decisions require reason_code assessment_confirmed"
            )
        if decision == "correct" and reason_code in {
            "assessment_confirmed",
            "insufficient_evidence",
            "rubric_ambiguous",
            "authority_unavailable",
            "conflict_unresolved",
        }:
            raise TeacherAgentAdjudicationError(
                "correct decisions require a bounded correction reason"
            )
        if decision == "abstain" and reason_code not in {
            "insufficient_evidence",
            "rubric_ambiguous",
            "authority_unavailable",
            "conflict_unresolved",
        }:
            raise TeacherAgentAdjudicationError(
                "abstain decisions require an abstention reason"
            )

        normalized_correction: dict[str, Any] | None = None
        # Validate against current targets while holding the lock below; include
        # the raw bounded object in the request fingerprint first.
        correction_input = (
            deepcopy(dict(correction))
            if isinstance(correction, Mapping)
            else correction
        )
        request = {
            "item_id": safe_item_id,
            "expected_version": expected,
            "claim_token_sha256": sha256(token.encode("utf-8")).hexdigest(),
            "evidence_id": safe_evidence_id,
            "evidence_sha256": safe_evidence_hash,
            "decision": decision,
            "reason_code": reason_code,
            "correction": correction_input,
            "actor": _stable_actor_binding(actor),
        }
        request_hash = canonical_sha256(request)
        with self._lock:
            replay = self._idempotent_replay(
                operation="decide", key=key, request_sha256=request_hash
            )
            if replay is not None:
                return replay
            item = self._latest(safe_item_id)
            if item["status"] == "decided":
                raise AdjudicationConflictError(
                    "adjudication item already has a terminal decision"
                )
            if item["status"] == "cancelled":
                raise AdjudicationConflictError(
                    "cancelled adjudication item cannot be decided"
                )
            self._check_cas(item, expected)
            original = item["original"]
            if (
                safe_evidence_id != original["evidence_id"]
                or safe_evidence_hash != original["evidence_sha256"]
            ):
                raise AdjudicationEvidenceError(
                    "decision evidence binding does not match the review item"
                )
            try:
                self._resolve(
                    evidence_id=safe_evidence_id,
                    evidence_sha256=safe_evidence_hash,
                )
            except AdjudicationEvidenceDeletedError:
                # The deletion transition is append-only and makes a concurrent
                # purge visible even when the caller attempted a decision.
                _now, occurred_at = self._now()
                self._cancel_locked(
                    item=item,
                    occurred_at=occurred_at,
                    request_sha256=canonical_sha256(
                        {
                            "evidence_id": safe_evidence_id,
                            "reason": "evidence_deleted",
                        }
                    ),
                )
                raise
            if item["status"] != "claimed" or not isinstance(
                item.get("lease"), Mapping
            ):
                raise AdjudicationConflictError("adjudication item is not claimed")
            now, occurred_at = self._now()
            lease = item["lease"]
            if _parse_timestamp(str(lease["expires_at"])) <= now:
                raise AdjudicationConflictError("adjudication claim lease expired")
            if lease["claim_token_sha256"] != sha256(token.encode("utf-8")).hexdigest():
                raise AdjudicationConflictError("claim_token does not own this lease")
            lease_actor = lease.get("actor")
            if not isinstance(lease_actor, Mapping) or _stable_actor_binding(
                lease_actor
            ) != _stable_actor_binding(actor):
                raise AdjudicationConflictError(
                    "authenticated actor does not own this adjudication lease"
                )

            if decision == "correct":
                normalized_correction = _validate_correction(
                    correction, item_target_kcs=item["target_kc_ids"]
                )
            elif correction is not None:
                raise TeacherAgentAdjudicationError(
                    "only correct decisions may include correction fields"
                )
            decision_record: dict[str, Any] = {
                "kind": decision,
                "reason_code": reason_code,
                "actor": actor,
                "evidence_id": safe_evidence_id,
                "evidence_sha256": safe_evidence_hash,
                "correction": normalized_correction,
                "decided_at": occurred_at,
                "idempotency_key_sha256": sha256(key.encode("utf-8")).hexdigest(),
            }
            if verified_authority is not None:
                decision_record["authority_receipt"] = verified_authority
            changes = {
                "status": "decided",
                "lease": None,
                "decision": decision_record,
            }
            # Build the immutable next version before mutating queue history so
            # a failed rubric-authority callback cannot leave a terminal item.
            preview = deepcopy(dict(item))
            preview.update(deepcopy(changes))
            preview["version"] = int(item["version"]) + 1
            preview["previous_version_sha256"] = item["version_sha256"]
            preview["updated_at"] = occurred_at
            preview.pop("version_sha256", None)
            preview["version_sha256"] = canonical_sha256(preview)
            instruction = reduce_adjudication(preview)
            if decision == "correct" and verified_authority is not None:
                if instruction_authorizer is None:
                    raise TeacherAgentAdjudicationError(
                        "authenticated corrections require rubric revalidation"
                    )
                try:
                    instruction = deepcopy(
                        dict(
                            instruction_authorizer(
                                instruction,
                                preview,
                                verified_authority,
                            )
                        )
                    )
                except TeacherAgentAdjudicationError:
                    raise
                except Exception as exc:
                    raise TeacherAgentAdjudicationError(
                        "authenticated correction revalidation failed"
                    ) from exc
                material = deepcopy(instruction)
                declared = material.pop("instruction_sha256", None)
                if (
                    instruction.get("mastery_update_authorized") is not True
                    or instruction.get("mutates_student_model") is not True
                    or declared != canonical_sha256(material)
                ):
                    raise TeacherAgentAdjudicationError(
                        "authenticated correction revalidation is invalid"
                    )
            version = self._append_version(
                item_id=safe_item_id,
                base=item,
                changes=changes,
                occurred_at=occurred_at,
                event_type=f"review.{decision}",
                actor=actor,
                request_sha256=request_hash,
            )
            if version != preview:  # pragma: no cover - deterministic reducer guard.
                raise TeacherAgentAdjudicationError(
                    "adjudication version preview diverged during commit"
                )
            response = {
                "item": public_review_item(version),
                "instruction": instruction,
            }
            self._remember_idempotency(
                operation="decide",
                key=key,
                request_sha256=request_hash,
                response=response,
            )
            return deepcopy(response)

    def _cancel_locked(
        self,
        *,
        item: Mapping[str, Any],
        occurred_at: str,
        request_sha256: str,
    ) -> dict[str, Any]:
        if item["status"] == "cancelled":
            return deepcopy(dict(item))
        return self._append_version(
            item_id=str(item["item_id"]),
            base=item,
            changes={
                "status": "cancelled",
                "lease": None,
                "cancellation": {
                    "reason": "evidence_deleted",
                    "cancelled_at": occurred_at,
                    "actor": _actor("system_evidence_purge"),
                },
            },
            occurred_at=occurred_at,
            event_type="review.cancelled_evidence_deleted",
            actor=_actor("system_evidence_purge"),
            request_sha256=request_sha256,
        )

    def cancel_deleted_evidence(self, evidence_id: str) -> tuple[dict[str, Any], ...]:
        """Cancel every matching item after the authoritative record is deleted."""

        safe_evidence_id = _safe_id(evidence_id, field="evidence_id")
        with self._lock:
            try:
                resolved = self._evidence_resolver(safe_evidence_id)
            except Exception as exc:
                raise AdjudicationEvidenceError(
                    "authoritative evidence resolver failed closed"
                ) from exc
            if resolved is not None:
                raise AdjudicationEvidenceError(
                    "evidence must be deleted from its authoritative store before cancellation"
                )
            item_id = self._evidence_items.get(safe_evidence_id)
            if item_id is None:
                return ()
            item = self._latest(item_id)
            if item["status"] == "cancelled":
                return (public_review_item(item),)
            _now, occurred_at = self._now()
            request_hash = canonical_sha256(
                {"evidence_id": safe_evidence_id, "reason": "evidence_deleted"}
            )
            version = self._cancel_locked(
                item=item,
                occurred_at=occurred_at,
                request_sha256=request_hash,
            )
            return (public_review_item(version),)


# A concise alias for callers that do not need the product-name prefix.
AdjudicationQueue = TeacherAgentAdjudicationQueue


__all__ = [
    "ADJUDICATION_AUDIT_SCHEMA",
    "ADJUDICATION_INSTRUCTION_SCHEMA",
    "ADJUDICATION_ITEM_SCHEMA",
    "CLAIM_LEASE_SECONDS",
    "LOCAL_OPERATOR_IDENTITY",
    "MAX_TARGET_KNOWLEDGE_COMPONENTS",
    "AdjudicationConflictError",
    "AdjudicationEvidenceDeletedError",
    "AdjudicationEvidenceError",
    "AdjudicationNotFoundError",
    "AdjudicationQueue",
    "TeacherAgentAdjudicationError",
    "TeacherAgentAdjudicationQueue",
    "authoritative_evidence_sha256",
    "canonical_sha256",
    "public_review_item",
    "reduce_adjudication",
]
