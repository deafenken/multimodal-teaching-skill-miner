"""Durable, content-minimizing safeguarding case and escalation core.

This module is deliberately an identity *consumer*, never an identity issuer.
Every mutation requires a caller-supplied verifier for a short-lived,
server-authenticated, hash-only authorization receipt proving the safeguarding
role.  The journal never accepts or persists learner text: callers provide only
an already-computed SHA-256 digest.

The authoritative store is an atomically replaced JSON document containing a
strict hash chain of immutable case versions.  A separate private lock file is
used so compaction can safely replace the journal while other processes wait on
the same inode.  Both files are opened without following symlinks and are
required to be regular, single-link, current-user-owned 0600 files.  Every
commit fsyncs the new file and its parent directory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from types import MappingProxyType
from typing import Any

try:  # pragma: no cover - exercised on the supported POSIX deployment target.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


SAFEGUARDING_STORE_SCHEMA_V1 = "teaching_skill_miner.safeguarding_store.v1"
SAFEGUARDING_STORE_SCHEMA = "teaching_skill_miner.safeguarding_store.v2"
SAFEGUARDING_EVENT_SCHEMA = "teaching_skill_miner.safeguarding_event.v1"
SAFEGUARDING_CASE_SCHEMA = "teaching_skill_miner.safeguarding_case.v1"
SAFEGUARDING_AUTHORIZATION_RECEIPT_SCHEMA = (
    "teaching_skill_miner.safeguarding_authorization_receipt.v1"
)
SAFEGUARDING_ACTOR_RECEIPT_SCHEMA = "teaching_skill_miner.safeguarding_actor_receipt.v1"
SAFEGUARDING_ERASURE_TOMBSTONE_SCHEMA = (
    "teaching_skill_miner.safeguarding_erasure_tombstone.v1"
)
EMERGENCY_RESOURCE_BUNDLE_SCHEMA = "teaching_skill_miner.emergency_resource_bundle.v1"
SAFEGUARDING_ERASURE_FENCE_SCHEMA = "teaching_skill_miner.safeguarding_erasure_fence.v2"
SAFEGUARDING_RETENTION_COMPACTION_SCHEMA = (
    "teaching_skill_miner.safeguarding_retention_compaction.v1"
)
SAFEGUARDING_RETENTION_AUTHORIZATION_RECEIPT_SCHEMA = (
    "teaching_skill_miner.safeguarding_retention_authorization_receipt.v1"
)
SAFEGUARDING_RETENTION_PLAN_SCHEMA = (
    "teaching_skill_miner.safeguarding_retention_plan.v1"
)
SAFEGUARDING_CAPACITY_STATUS_SCHEMA = (
    "teaching_skill_miner.safeguarding_capacity_status.v1"
)

# The role name itself never enters a receipt or the durable store.
SAFEGUARDING_ROLE_SHA256 = sha256(b"safeguarding").hexdigest()
SYSTEM_SAFETY_CLASSIFIER_ROLE_SHA256 = sha256(b"system_safety_classifier").hexdigest()
SAFEGUARDING_RETENTION_ROLE_SHA256 = sha256(b"safeguarding_retention").hexdigest()

SAFEGUARDING_CATEGORIES = frozenset(
    {
        "self_harm",
        "abuse_disclosure",
        "bullying_disclosure",
        "minor_sexual_content",
        "urgent_medical",
        "harm_to_others",
        "other_safeguarding",
    }
)
SAFEGUARDING_SEVERITIES = frozenset({"elevated", "high", "urgent"})
SAFEGUARDING_CASE_STATUSES = frozenset({"open", "acknowledged", "closed"})
ESCALATION_DELIVERY_STATUSES = frozenset(
    {
        "escalation_unavailable",
        "pending",
        "overdue",
        "acknowledged",
    }
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CASE_ID = re.compile(r"^sgc_[0-9a-f]{24}$")
_DELIVERY_ID = re.compile(r"^sge_[0-9a-f]{24}$")
_EVENT_ID = re.compile(r"^sgev_[0-9a-f]{24}$")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_LOCALE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8}){0,3}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_MAX_EVENTS = 32_768
_MAX_TOMBSTONES = 32_768
_MAX_STORE_BYTES = 32 * 1024 * 1024
_ERASURE_FENCE_BYTES = 8 * 1024 * 1024
_ERASURE_FENCE_BITS = _ERASURE_FENCE_BYTES * 8
_ERASURE_FENCE_HASHES = 7
_ERASURE_FALSE_POSITIVE_TARGET = 1e-6
_RECENT_TOMBSTONES = 1024
_MAX_AUTHORIZATION_TTL_SECONDS = 15 * 60
_CLOCK_SKEW_SECONDS = 60
_STAFF_ACTOR_KIND = "server_authenticated_safeguarding"
_SYSTEM_ACTOR_KIND = "system_safety_classifier"
_RETENTION_ACTOR_KIND = "server_safeguarding_retention"
_RETENTION_OPERATION = "case.retention_compacted"
_MUTATING_OPERATIONS = frozenset(
    {
        "case.opened",
        "case.acknowledged",
        "case.closed",
        "escalation.overdue",
        "escalation.acknowledged",
        "case.purged",
    }
)

_AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "actor_kind",
        "server_authenticated",
        "principal_sha256",
        "role_sha256",
        "authorization_context_sha256",
        "operation",
        "body_sha256",
        "issued_at_utc",
        "expires_at_utc",
        "receipt_sha256",
    }
)
_ACTOR_KEYS = frozenset(
    {
        "schema",
        "actor_kind",
        "principal_sha256",
        "role_sha256",
        "authorization_context_sha256",
        "operation",
        "body_sha256",
        "authorization_receipt_sha256",
        "verified_at_utc",
    }
)
_CASE_KEYS = frozenset(
    {
        "schema",
        "case_id",
        "version",
        "status",
        "scope_sha256",
        "category",
        "severity",
        "observed_at_utc",
        "content_sha256",
        "created_at_utc",
        "updated_at_utc",
        "opened_by",
        "last_updated_by",
        "escalation",
        "emergency_resource_receipt",
        "previous_version_sha256",
        "case_sha256",
    }
)
_ESCALATION_KEYS = frozenset(
    {
        "delivery_status",
        "reason_code",
        "policy_version",
        "queue_sha256",
        "delivery_id",
        "sla_seconds",
        "enqueued_at_utc",
        "sla_due_at_utc",
        "overdue_recorded_at_utc",
        "acknowledged_at_utc",
        "acknowledged_by",
    }
)
_RESOURCE_RECEIPT_KEYS = frozenset(
    {
        "policy_version",
        "locale_selection_source",
        "configured_locale_sha256",
        "effective_locale",
        "localization_status",
        "localization_unavailable",
        "resource_set_sha256",
    }
)
_EVENT_KEYS = frozenset(
    {
        "schema",
        "sequence",
        "event_id",
        "event_type",
        "occurred_at_utc",
        "idempotency_key_sha256",
        "request_sha256",
        "case_snapshot",
        "actor",
        "previous_event_sha256",
        "event_sha256",
    }
)
_TOMBSTONE_KEYS = frozenset(
    {
        "schema",
        "case_id_sha256",
        "purged_at_utc",
        "generation",
        "expected_version",
        "idempotency_key_sha256",
        "request_sha256",
        "authorization_receipt_sha256",
        "actor",
        "tombstone_sha256",
    }
)
_RESOURCE_ITEM_KEYS = frozenset({"kind", "label", "instruction"})
_RESOURCE_KINDS = frozenset(
    {
        "emergency_services",
        "crisis_support",
        "trusted_adult",
        "institution_safeguarding",
    }
)

_ERASURE_FENCE_KEYS = frozenset(
    {
        "schema",
        "algorithm",
        "generation",
        "bit_count",
        "hash_count",
        "inserted_count",
        "bits_sha256",
        "accumulator_sha256",
        "false_negative_possible",
        "false_positive_policy",
    }
)
_RETENTION_COMPACTION_KEYS = frozenset(
    {
        "schema",
        "generation",
        "cases_compacted_total",
        "events_compacted_total",
        "last_compacted_at_utc",
        "last_authorization_receipt_sha256",
        "last_operation_body_sha256",
        "policy_version_sha256",
        "minimum_closed_age_seconds",
        "previous_compaction_sha256",
        "compaction_sha256",
    }
)
_RETENTION_AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "version",
        "actor_kind",
        "server_authenticated",
        "principal_sha256",
        "role_sha256",
        "authorization_context_sha256",
        "operation",
        "body_sha256",
        "issued_at_utc",
        "expires_at_utc",
        "authority_proof_sha256",
        "receipt_sha256",
    }
)


class TeacherAgentSafeguardingError(RuntimeError):
    """Base error for safeguarding integrity, policy, and transition failures."""


class SafeguardingIntegrityError(TeacherAgentSafeguardingError):
    """Raised when durable safeguarding state cannot be trusted."""


class SafeguardingAuthorizationError(TeacherAgentSafeguardingError):
    """Raised when server authorization cannot be proven."""


class SafeguardingConflictError(TeacherAgentSafeguardingError):
    """Raised for stale expected versions or conflicting idempotency reuse."""


class SafeguardingNotFoundError(TeacherAgentSafeguardingError):
    """Raised when a case does not exist in the active projection."""


class SafeguardingConfigurationError(TeacherAgentSafeguardingError):
    """Raised for an unsafe escalation or localization configuration."""


class SafeguardingErasedError(TeacherAgentSafeguardingError):
    """Raised when a purged case identity is reused."""


AuthorizationVerifier = Callable[[Mapping[str, Any]], Mapping[str, Any]]
RetentionAuthorizationVerifier = Callable[[Mapping[str, Any]], Mapping[str, Any]]
Clock = Callable[[], datetime]


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
        raise SafeguardingIntegrityError(
            "safeguarding values must be canonical JSON"
        ) from exc


def canonical_sha256(value: Any) -> str:
    """Return the canonical JSON SHA-256 used by public core receipts."""

    return sha256(_canonical_bytes(value)).hexdigest()


def authorization_receipt_sha256(value: Mapping[str, Any]) -> str:
    """Compute an external receipt seal without minting an identity assertion."""

    material = deepcopy(dict(value))
    material.pop("receipt_sha256", None)
    return canonical_sha256(material)


def retention_authorization_receipt_sha256(value: Mapping[str, Any]) -> str:
    """Seal the normalized, hash-only retention receipt envelope."""

    material = deepcopy(dict(value))
    material.pop("receipt_sha256", None)
    return canonical_sha256(material)


@dataclass(frozen=True, slots=True)
class SafeguardingCapacityLimits:
    """Injectable bounded capacities; production defaults stay fail-closed."""

    maximum_events: int = _MAX_EVENTS
    maximum_store_bytes: int = _MAX_STORE_BYTES
    recent_tombstones: int = _RECENT_TOMBSTONES
    near_event_headroom: int = 128
    near_byte_headroom: int = 512 * 1024

    def __post_init__(self) -> None:
        values = (
            self.maximum_events,
            self.maximum_store_bytes,
            self.recent_tombstones,
            self.near_event_headroom,
            self.near_byte_headroom,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise SafeguardingConfigurationError(
                "safeguarding capacity limits must be integers"
            )
        if (
            not 4 <= self.maximum_events <= _MAX_EVENTS
            or not 16 * 1024 <= self.maximum_store_bytes <= _MAX_STORE_BYTES
            or not 1 <= self.recent_tombstones <= _RECENT_TOMBSTONES
            or not 0 <= self.near_event_headroom < self.maximum_events
            or not 0 <= self.near_byte_headroom < self.maximum_store_bytes
        ):
            raise SafeguardingConfigurationError(
                "safeguarding capacity limits are invalid"
            )


@dataclass(frozen=True, slots=True)
class SafeguardingRetentionPolicy:
    """Deployment-owned minimum retention and bounded compaction batch."""

    policy_version: str
    minimum_closed_age_seconds: int
    maximum_cases_per_run: int = 128

    def __post_init__(self) -> None:
        version = _require_token(self.policy_version, field="retention.policy_version")
        if (
            isinstance(self.minimum_closed_age_seconds, bool)
            or not isinstance(self.minimum_closed_age_seconds, int)
            or not 1 <= self.minimum_closed_age_seconds <= 10 * 365 * 24 * 60 * 60
            or isinstance(self.maximum_cases_per_run, bool)
            or not isinstance(self.maximum_cases_per_run, int)
            or not 1 <= self.maximum_cases_per_run <= 1024
        ):
            raise SafeguardingConfigurationError(
                "safeguarding retention policy is invalid"
            )
        object.__setattr__(self, "policy_version", version)

    @property
    def policy_version_sha256(self) -> str:
        return sha256(self.policy_version.encode("utf-8")).hexdigest()


def safeguarding_open_authorization_body_sha256(
    *,
    scope_sha256: str,
    category: str,
    severity: str,
    observed_at_utc: str,
    content_sha256: str,
) -> str:
    """Hash the exact minimized body a system classifier may authorize."""

    _require_digest(scope_sha256, field="scope_sha256")
    _require_digest(content_sha256, field="content_sha256")
    if category not in SAFEGUARDING_CATEGORIES:
        raise SafeguardingIntegrityError("safeguarding category is unsupported")
    if severity not in SAFEGUARDING_SEVERITIES:
        raise SafeguardingIntegrityError("safeguarding severity is unsupported")
    _parse_utc(observed_at_utc, field="observed_at_utc")
    return canonical_sha256(
        {
            "operation": "case.opened",
            "scope_sha256": scope_sha256,
            "category": category,
            "severity": severity,
            "observed_at_utc": observed_at_utc,
            "content_sha256": content_sha256,
        }
    )


def safeguarding_case_authorization_body_sha256(
    *, operation: str, case_id: str, expected_version: int
) -> str:
    """Hash a staff-only case mutation without retaining the raw case ID."""

    if operation not in _MUTATING_OPERATIONS.difference({"case.opened"}):
        raise SafeguardingIntegrityError(
            "safeguarding authorization operation is unsupported"
        )
    if not isinstance(case_id, str) or _CASE_ID.fullmatch(case_id) is None:
        raise SafeguardingIntegrityError("case_id is invalid")
    if (
        isinstance(expected_version, bool)
        or not isinstance(expected_version, int)
        or expected_version < 1
    ):
        raise SafeguardingIntegrityError("expected_version is invalid")
    return canonical_sha256(
        {
            "operation": operation,
            "case_id_sha256": sha256(case_id.encode("utf-8")).hexdigest(),
            "expected_version": expected_version,
        }
    )


def _require_digest(value: Any, *, field: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise SafeguardingIntegrityError(f"{field} must be a lowercase SHA-256")
    return value


def _require_token(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_TOKEN.fullmatch(value) is None:
        raise SafeguardingIntegrityError(f"{field} is invalid")
    return value


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise SafeguardingIntegrityError(f"{field} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SafeguardingIntegrityError(f"{field} is not a real timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise SafeguardingIntegrityError(f"{field} must be UTC")
    return parsed


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SafeguardingIntegrityError("safeguarding clock must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _safe_config_text(value: Any, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value) > maximum
        or "\x00" in value
    ):
        raise SafeguardingConfigurationError(
            f"{field} must be a trimmed non-empty configuration string"
        )
    return value


def _validate_resource_items(
    values: Sequence[Mapping[str, Any]], *, field: str
) -> tuple[Mapping[str, str], ...]:
    if isinstance(values, (str, bytes)) or not 1 <= len(values) <= 8:
        raise SafeguardingConfigurationError(f"{field} must contain 1..8 resources")
    normalized: list[Mapping[str, str]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or frozenset(value) != _RESOURCE_ITEM_KEYS:
            raise SafeguardingConfigurationError(
                f"{field}[{index}] has an invalid resource shape"
            )
        kind = str(value.get("kind", ""))
        if kind not in _RESOURCE_KINDS:
            raise SafeguardingConfigurationError(
                f"{field}[{index}].kind is unsupported"
            )
        normalized.append(
            MappingProxyType(
                {
                    "kind": kind,
                    "label": _safe_config_text(
                        value.get("label"), field=f"{field}[{index}].label", maximum=160
                    ),
                    "instruction": _safe_config_text(
                        value.get("instruction"),
                        field=f"{field}[{index}].instruction",
                        maximum=800,
                    ),
                }
            )
        )
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class EscalationDeliveryConfig:
    """Trusted deployment configuration for the durable escalation outbox."""

    policy_version: str
    queue_sha256: str
    sla_seconds_by_severity: Mapping[str, int]

    def __post_init__(self) -> None:
        version = _require_token(self.policy_version, field="escalation.policy_version")
        try:
            queue = _require_digest(self.queue_sha256, field="escalation.queue_sha256")
        except SafeguardingIntegrityError as exc:
            raise SafeguardingConfigurationError(str(exc)) from exc
        values = dict(self.sla_seconds_by_severity)
        if frozenset(values) != SAFEGUARDING_SEVERITIES:
            raise SafeguardingConfigurationError(
                "escalation SLA configuration must cover every safeguarding severity"
            )
        normalized: dict[str, int] = {}
        for severity, seconds in values.items():
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, int)
                or not 30 <= seconds <= 7 * 24 * 60 * 60
            ):
                raise SafeguardingConfigurationError(
                    f"escalation SLA for {severity} must be 30..604800 seconds"
                )
            normalized[severity] = seconds
        object.__setattr__(self, "policy_version", version)
        object.__setattr__(self, "queue_sha256", queue)
        object.__setattr__(
            self, "sla_seconds_by_severity", MappingProxyType(normalized)
        )


@dataclass(frozen=True, slots=True)
class EmergencyResourcePolicy:
    """Versioned emergency resources selected only by trusted deployment scope.

    Locale is intentionally absent from every case-mutation method.  The only
    selector is ``trusted_locale_by_scope_sha256``, which must be assembled by
    the authenticated server from deployment configuration rather than learner
    text or browser fields.
    """

    policy_version: str
    trusted_locale_by_scope_sha256: Mapping[str, str]
    resources_by_locale: Mapping[str, Sequence[Mapping[str, Any]]]
    generic_fallback: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        version = _require_token(
            self.policy_version, field="emergency_resources.policy_version"
        )
        locale_map: dict[str, str] = {}
        for scope_digest, locale in dict(self.trusted_locale_by_scope_sha256).items():
            try:
                digest = _require_digest(
                    scope_digest, field="emergency_resources.scope_sha256"
                )
            except SafeguardingIntegrityError as exc:
                raise SafeguardingConfigurationError(str(exc)) from exc
            if not isinstance(locale, str) or _LOCALE.fullmatch(locale) is None:
                raise SafeguardingConfigurationError(
                    "trusted deployment locale mapping contains an invalid locale"
                )
            locale_map[digest] = locale
        catalogs: dict[str, tuple[Mapping[str, str], ...]] = {}
        for locale, resources in dict(self.resources_by_locale).items():
            if not isinstance(locale, str) or _LOCALE.fullmatch(locale) is None:
                raise SafeguardingConfigurationError(
                    "emergency resource catalog contains an invalid locale"
                )
            catalogs[locale] = _validate_resource_items(
                resources, field=f"resources_by_locale.{locale}"
            )
        generic = _validate_resource_items(
            self.generic_fallback, field="generic_fallback"
        )
        object.__setattr__(self, "policy_version", version)
        object.__setattr__(
            self, "trusted_locale_by_scope_sha256", MappingProxyType(locale_map)
        )
        object.__setattr__(self, "resources_by_locale", MappingProxyType(catalogs))
        object.__setattr__(self, "generic_fallback", generic)

    def resolve(self, scope_sha256: str) -> dict[str, Any]:
        """Resolve resources without accepting any user-selectable locale."""

        _require_digest(scope_sha256, field="scope_sha256")
        configured = self.trusted_locale_by_scope_sha256.get(scope_sha256)
        resources = self.resources_by_locale.get(configured) if configured else None
        available = resources is not None
        selected = resources if available else self.generic_fallback
        return {
            "schema": EMERGENCY_RESOURCE_BUNDLE_SCHEMA,
            "policy_version": self.policy_version,
            "locale_selection_source": "trusted_deployment_scope_mapping",
            "configured_locale": configured,
            "effective_locale": configured if available else "generic",
            "localization_status": (
                "localized" if available else "localization_unavailable"
            ),
            "localization_unavailable": not available,
            "resources": [dict(item) for item in selected],
        }


def default_emergency_resource_policy(
    trusted_locale_by_scope_sha256: Mapping[str, str],
) -> EmergencyResourcePolicy:
    """Return the versioned English/Chinese generic-local-services policy."""

    return EmergencyResourcePolicy(
        policy_version="institutional_emergency_resources_v1",
        trusted_locale_by_scope_sha256=trusted_locale_by_scope_sha256,
        resources_by_locale={
            "en-US": (
                {
                    "kind": "emergency_services",
                    "label": "Immediate danger",
                    "instruction": (
                        "Contact your local emergency services now if anyone is in "
                        "immediate danger."
                    ),
                },
                {
                    "kind": "trusted_adult",
                    "label": "Trusted support",
                    "instruction": (
                        "Move to a safer place and tell a trusted adult or "
                        "safeguarding professional now."
                    ),
                },
            ),
            "zh-CN": (
                {
                    "kind": "emergency_services",
                    "label": "紧急危险",
                    "instruction": "如任何人正面临紧急危险，请立即联系所在地紧急服务。",
                },
                {
                    "kind": "trusted_adult",
                    "label": "可信支持",
                    "instruction": (
                        "请前往安全地点，并立即告知可信任的成年人或安全保护专业人员。"
                    ),
                },
            ),
        },
        generic_fallback=(
            {
                "kind": "emergency_services",
                "label": "Immediate danger",
                "instruction": "Contact local emergency services now.",
            },
            {
                "kind": "trusted_adult",
                "label": "Trusted support",
                "instruction": (
                    "Move to a safer place and contact a trusted safeguarding "
                    "adult now."
                ),
            },
        ),
    )


def _empty_retention_compaction() -> dict[str, Any]:
    value = {
        "schema": SAFEGUARDING_RETENTION_COMPACTION_SCHEMA,
        "generation": 0,
        "cases_compacted_total": 0,
        "events_compacted_total": 0,
        "last_compacted_at_utc": None,
        "last_authorization_receipt_sha256": None,
        "last_operation_body_sha256": None,
        "policy_version_sha256": None,
        "minimum_closed_age_seconds": None,
        "previous_compaction_sha256": None,
        "compaction_sha256": None,
    }
    return value


def _empty_erasure_fence(*, generation: int = 1) -> dict[str, Any]:
    bits = bytes(_ERASURE_FENCE_BYTES)
    return {
        "schema": SAFEGUARDING_ERASURE_FENCE_SCHEMA,
        "algorithm": "bloom_sha256_double_hash_v1",
        "generation": generation,
        "bit_count": _ERASURE_FENCE_BITS,
        "hash_count": _ERASURE_FENCE_HASHES,
        "inserted_count": 0,
        "bits_sha256": sha256(bits).hexdigest(),
        "accumulator_sha256": sha256(
            b"safeguarding-erasure-accumulator-v2"
        ).hexdigest(),
        "false_negative_possible": False,
        "false_positive_policy": "fail_closed_as_erased",
    }


def _empty_state() -> dict[str, Any]:
    """Return the historical v1 empty state; the first commit migrates to v2."""

    return {
        "schema": SAFEGUARDING_STORE_SCHEMA_V1,
        "version": 1,
        "event_sequence": 0,
        "events": [],
        "erasure_tombstones": {},
    }


def _bloom_positions(case_id_sha256: str) -> tuple[int, ...]:
    raw = bytes.fromhex(case_id_sha256)
    first = int.from_bytes(sha256(b"\x00" + raw).digest()[:8], "big")
    step = int.from_bytes(sha256(b"\x01" + raw).digest()[:8], "big") | 1
    return tuple(
        (first + index * step) % _ERASURE_FENCE_BITS
        for index in range(_ERASURE_FENCE_HASHES)
    )


def _bloom_contains(bits: bytes | bytearray, case_id_sha256: str) -> bool:
    return all(
        bits[position // 8] & (1 << (position % 8))
        for position in _bloom_positions(case_id_sha256)
    )


def _bloom_add(bits: bytearray, case_id_sha256: str) -> None:
    for position in _bloom_positions(case_id_sha256):
        bits[position // 8] |= 1 << (position % 8)


def _erasure_false_positive_upper_bound(inserted_count: int) -> float:
    if inserted_count <= 0:
        return 0.0
    exponent = -(_ERASURE_FENCE_HASHES * inserted_count / (_ERASURE_FENCE_BITS - 1))
    return min(1.0, (1.0 - math.exp(exponent)) ** _ERASURE_FENCE_HASHES)


def safeguarding_erasure_false_positive_upper_bound(inserted_count: int) -> float:
    """Return the configured 64-Mibit/7-hash Bloom false-positive bound."""

    if (
        isinstance(inserted_count, bool)
        or not isinstance(inserted_count, int)
        or inserted_count < 0
    ):
        raise SafeguardingIntegrityError(
            "erasure fence inserted_count must be a non-negative integer"
        )
    return _erasure_false_positive_upper_bound(inserted_count)


def _advance_erasure_accumulator(previous: str, case_id_sha256: str) -> str:
    return sha256(bytes.fromhex(previous) + bytes.fromhex(case_id_sha256)).hexdigest()


def _retention_compaction_hash(value: Mapping[str, Any]) -> str:
    material = deepcopy(dict(value))
    material.pop("compaction_sha256", None)
    return canonical_sha256(material)


def _actor_from_authorization(
    receipt: Mapping[str, Any], *, verified_at_utc: str
) -> dict[str, Any]:
    return {
        "schema": SAFEGUARDING_ACTOR_RECEIPT_SCHEMA,
        "actor_kind": receipt["actor_kind"],
        "principal_sha256": receipt["principal_sha256"],
        "role_sha256": receipt["role_sha256"],
        "authorization_context_sha256": receipt["authorization_context_sha256"],
        "operation": receipt["operation"],
        "body_sha256": receipt["body_sha256"],
        "authorization_receipt_sha256": receipt["receipt_sha256"],
        "verified_at_utc": verified_at_utc,
    }


def _validate_actor(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _ACTOR_KEYS:
        raise SafeguardingIntegrityError(f"{field} has an invalid actor shape")
    actor = deepcopy(dict(value))
    if actor.get("schema") != SAFEGUARDING_ACTOR_RECEIPT_SCHEMA:
        raise SafeguardingIntegrityError(f"{field} actor schema is unsupported")
    for key in (
        "principal_sha256",
        "authorization_context_sha256",
        "authorization_receipt_sha256",
    ):
        _require_digest(actor.get(key), field=f"{field}.{key}")
    actor_kind = actor.get("actor_kind")
    role_sha256 = actor.get("role_sha256")
    if actor_kind == _STAFF_ACTOR_KIND:
        if role_sha256 != SAFEGUARDING_ROLE_SHA256:
            raise SafeguardingIntegrityError(f"{field} actor lacks safeguarding role")
    elif actor_kind == _SYSTEM_ACTOR_KIND:
        if role_sha256 != SYSTEM_SAFETY_CLASSIFIER_ROLE_SHA256:
            raise SafeguardingIntegrityError(
                f"{field} system actor is not the safety classifier"
            )
    else:
        raise SafeguardingIntegrityError(f"{field}.actor_kind is unsupported")
    operation = actor.get("operation")
    if operation not in _MUTATING_OPERATIONS:
        raise SafeguardingIntegrityError(f"{field}.operation is unsupported")
    _require_digest(actor.get("body_sha256"), field=f"{field}.body_sha256")
    _parse_utc(actor.get("verified_at_utc"), field=f"{field}.verified_at_utc")
    return actor


def _resource_receipt(bundle: Mapping[str, Any]) -> dict[str, Any]:
    configured = bundle.get("configured_locale")
    resources = bundle.get("resources")
    return {
        "policy_version": bundle["policy_version"],
        "locale_selection_source": bundle["locale_selection_source"],
        "configured_locale_sha256": (
            sha256(str(configured).encode("utf-8")).hexdigest()
            if configured is not None
            else None
        ),
        "effective_locale": bundle["effective_locale"],
        "localization_status": bundle["localization_status"],
        "localization_unavailable": bundle["localization_unavailable"],
        "resource_set_sha256": canonical_sha256(resources),
    }


def _validate_resource_receipt(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _RESOURCE_RECEIPT_KEYS:
        raise SafeguardingIntegrityError(f"{field} resource receipt shape is invalid")
    result = deepcopy(dict(value))
    _require_token(result.get("policy_version"), field=f"{field}.policy_version")
    if result.get("locale_selection_source") != "trusted_deployment_scope_mapping":
        raise SafeguardingIntegrityError(
            f"{field} locale source is not trusted deployment configuration"
        )
    _require_digest(
        result.get("configured_locale_sha256"),
        field=f"{field}.configured_locale_sha256",
        nullable=True,
    )
    effective = result.get("effective_locale")
    if effective != "generic" and (
        not isinstance(effective, str) or _LOCALE.fullmatch(effective) is None
    ):
        raise SafeguardingIntegrityError(f"{field}.effective_locale is invalid")
    unavailable = result.get("localization_unavailable")
    status = result.get("localization_status")
    if unavailable is True:
        if status != "localization_unavailable" or effective != "generic":
            raise SafeguardingIntegrityError(
                f"{field} unavailable localization state is inconsistent"
            )
    elif unavailable is False:
        if status != "localized" or effective == "generic":
            raise SafeguardingIntegrityError(f"{field} localized state is inconsistent")
    else:
        raise SafeguardingIntegrityError(
            f"{field}.localization_unavailable must be boolean"
        )
    _require_digest(
        result.get("resource_set_sha256"), field=f"{field}.resource_set_sha256"
    )
    return result


def _unavailable_escalation() -> dict[str, Any]:
    return {
        "delivery_status": "escalation_unavailable",
        "reason_code": "delivery_not_configured",
        "policy_version": None,
        "queue_sha256": None,
        "delivery_id": None,
        "sla_seconds": None,
        "enqueued_at_utc": None,
        "sla_due_at_utc": None,
        "overdue_recorded_at_utc": None,
        "acknowledged_at_utc": None,
        "acknowledged_by": None,
    }


def _validate_escalation(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _ESCALATION_KEYS:
        raise SafeguardingIntegrityError(f"{field} escalation shape is invalid")
    result = deepcopy(dict(value))
    delivery_status = result.get("delivery_status")
    if delivery_status not in ESCALATION_DELIVERY_STATUSES:
        raise SafeguardingIntegrityError(f"{field}.delivery_status is invalid")
    if delivery_status == "escalation_unavailable":
        if result != _unavailable_escalation():
            raise SafeguardingIntegrityError(
                f"{field} unavailable escalation state is inconsistent"
            )
        return result

    _require_token(result.get("policy_version"), field=f"{field}.policy_version")
    _require_digest(result.get("queue_sha256"), field=f"{field}.queue_sha256")
    delivery_id = result.get("delivery_id")
    if not isinstance(delivery_id, str) or _DELIVERY_ID.fullmatch(delivery_id) is None:
        raise SafeguardingIntegrityError(f"{field}.delivery_id is invalid")
    seconds = result.get("sla_seconds")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, int)
        or not 30 <= seconds <= 7 * 24 * 60 * 60
    ):
        raise SafeguardingIntegrityError(f"{field}.sla_seconds is invalid")
    enqueued = _parse_utc(
        result.get("enqueued_at_utc"), field=f"{field}.enqueued_at_utc"
    )
    due = _parse_utc(result.get("sla_due_at_utc"), field=f"{field}.sla_due_at_utc")
    if due != enqueued + timedelta(seconds=seconds):
        raise SafeguardingIntegrityError(f"{field} SLA deadline is inconsistent")
    overdue_at = result.get("overdue_recorded_at_utc")
    acknowledged_at = result.get("acknowledged_at_utc")
    acknowledged_by = result.get("acknowledged_by")
    if delivery_status == "pending":
        if any(
            item is not None for item in (overdue_at, acknowledged_at, acknowledged_by)
        ):
            raise SafeguardingIntegrityError(
                f"{field} pending escalation state is inconsistent"
            )
    elif delivery_status == "overdue":
        recorded = _parse_utc(overdue_at, field=f"{field}.overdue_recorded_at_utc")
        if recorded < due or acknowledged_at is not None or acknowledged_by is not None:
            raise SafeguardingIntegrityError(
                f"{field} overdue escalation state is inconsistent"
            )
    else:
        acked = _parse_utc(acknowledged_at, field=f"{field}.acknowledged_at_utc")
        if acked < enqueued:
            raise SafeguardingIntegrityError(
                f"{field} acknowledgement predates enqueue"
            )
        _validate_actor(acknowledged_by, field=f"{field}.acknowledged_by")
        if overdue_at is not None:
            recorded = _parse_utc(overdue_at, field=f"{field}.overdue_recorded_at_utc")
            if recorded < due or acked < recorded:
                raise SafeguardingIntegrityError(
                    f"{field} acknowledgement/overdue ordering is invalid"
                )
    if result.get("reason_code") is not None:
        raise SafeguardingIntegrityError(
            f"{field} configured escalation must not have an unavailable reason"
        )
    return result


def _case_hash(value: Mapping[str, Any]) -> str:
    material = deepcopy(dict(value))
    material.pop("case_sha256", None)
    return canonical_sha256(material)


def _validate_case(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _CASE_KEYS:
        raise SafeguardingIntegrityError(f"{field} case shape is invalid")
    result = deepcopy(dict(value))
    if result.get("schema") != SAFEGUARDING_CASE_SCHEMA:
        raise SafeguardingIntegrityError(f"{field} case schema is unsupported")
    case_id = result.get("case_id")
    if not isinstance(case_id, str) or _CASE_ID.fullmatch(case_id) is None:
        raise SafeguardingIntegrityError(f"{field}.case_id is invalid")
    version = result.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise SafeguardingIntegrityError(f"{field}.version is invalid")
    if result.get("status") not in SAFEGUARDING_CASE_STATUSES:
        raise SafeguardingIntegrityError(f"{field}.status is invalid")
    _require_digest(result.get("scope_sha256"), field=f"{field}.scope_sha256")
    _require_digest(result.get("content_sha256"), field=f"{field}.content_sha256")
    if result.get("category") not in SAFEGUARDING_CATEGORIES:
        raise SafeguardingIntegrityError(f"{field}.category is invalid")
    if result.get("severity") not in SAFEGUARDING_SEVERITIES:
        raise SafeguardingIntegrityError(f"{field}.severity is invalid")
    observed = _parse_utc(
        result.get("observed_at_utc"), field=f"{field}.observed_at_utc"
    )
    created = _parse_utc(result.get("created_at_utc"), field=f"{field}.created_at_utc")
    updated = _parse_utc(result.get("updated_at_utc"), field=f"{field}.updated_at_utc")
    if observed > created + timedelta(seconds=_CLOCK_SKEW_SECONDS) or updated < created:
        raise SafeguardingIntegrityError(f"{field} timestamps are inconsistent")
    _validate_actor(result.get("opened_by"), field=f"{field}.opened_by")
    _validate_actor(result.get("last_updated_by"), field=f"{field}.last_updated_by")
    escalation = _validate_escalation(
        result.get("escalation"), field=f"{field}.escalation"
    )
    if escalation["delivery_status"] != "escalation_unavailable":
        expected_delivery_id = (
            "sge_"
            + canonical_sha256(
                {
                    "case_id": case_id,
                    "queue_sha256": escalation["queue_sha256"],
                }
            )[:24]
        )
        if escalation["delivery_id"] != expected_delivery_id:
            raise SafeguardingIntegrityError(
                f"{field} escalation delivery identity is invalid"
            )
    _validate_resource_receipt(
        result.get("emergency_resource_receipt"),
        field=f"{field}.emergency_resource_receipt",
    )
    _require_digest(
        result.get("previous_version_sha256"),
        field=f"{field}.previous_version_sha256",
        nullable=True,
    )
    declared = _require_digest(result.get("case_sha256"), field=f"{field}.case_sha256")
    if declared != _case_hash(result):
        raise SafeguardingIntegrityError(f"{field} case integrity check failed")
    return result


def _event_hash(value: Mapping[str, Any]) -> str:
    material = deepcopy(dict(value))
    material.pop("event_sha256", None)
    return canonical_sha256(material)


def _tombstone_hash(value: Mapping[str, Any]) -> str:
    material = deepcopy(dict(value))
    material.pop("tombstone_sha256", None)
    return canonical_sha256(material)


def _same_actor(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return dict(left) == dict(right)


def _validate_transition(
    *,
    event_type: str,
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    event_actor: Mapping[str, Any],
    occurred_at_utc: str,
) -> None:
    if event_actor["operation"] != event_type:
        raise SafeguardingIntegrityError(
            "authorization operation is not bound to its durable event"
        )
    if current["updated_at_utc"] != occurred_at_utc:
        raise SafeguardingIntegrityError(
            "case update timestamp is not bound to its durable event"
        )
    if event_actor["verified_at_utc"] != occurred_at_utc:
        raise SafeguardingIntegrityError(
            "case actor verification is not bound to its durable event"
        )
    if previous is None:
        if event_type != "case.opened" or current["version"] != 1:
            raise SafeguardingIntegrityError("case history must begin with case.opened")
        if (
            current["status"] != "open"
            or current["previous_version_sha256"] is not None
        ):
            raise SafeguardingIntegrityError("new safeguarding case state is invalid")
        if not _same_actor(current["opened_by"], event_actor) or not _same_actor(
            current["last_updated_by"], event_actor
        ):
            raise SafeguardingIntegrityError("case opener actor binding is invalid")
        if event_actor["actor_kind"] not in {_SYSTEM_ACTOR_KIND, _STAFF_ACTOR_KIND}:
            raise SafeguardingIntegrityError("case opener authority is invalid")
        expected_body = safeguarding_open_authorization_body_sha256(
            scope_sha256=current["scope_sha256"],
            category=current["category"],
            severity=current["severity"],
            observed_at_utc=current["observed_at_utc"],
            content_sha256=current["content_sha256"],
        )
        if event_actor["body_sha256"] != expected_body:
            raise SafeguardingIntegrityError(
                "case opener authorization is not bound to scope/content"
            )
        if current["created_at_utc"] != occurred_at_utc:
            raise SafeguardingIntegrityError(
                "case creation timestamp is not bound to its durable event"
            )
        escalation = current["escalation"]
        if (
            escalation["delivery_status"] != "escalation_unavailable"
            and escalation["enqueued_at_utc"] != occurred_at_utc
        ):
            raise SafeguardingIntegrityError(
                "escalation enqueue is not atomic with case creation"
            )
        return

    if current["version"] != previous["version"] + 1:
        raise SafeguardingIntegrityError("case version sequence is invalid")
    if event_actor["actor_kind"] != _STAFF_ACTOR_KIND:
        raise SafeguardingIntegrityError(
            "post-open safeguarding mutation requires safeguarding role"
        )
    expected_body = safeguarding_case_authorization_body_sha256(
        operation=event_type,
        case_id=current["case_id"],
        expected_version=previous["version"],
    )
    if event_actor["body_sha256"] != expected_body:
        raise SafeguardingIntegrityError(
            "case mutation authorization is not bound to operation/body"
        )
    if current["previous_version_sha256"] != previous["case_sha256"]:
        raise SafeguardingIntegrityError("case version hash chain is invalid")
    immutable = (
        "schema",
        "case_id",
        "scope_sha256",
        "category",
        "severity",
        "observed_at_utc",
        "content_sha256",
        "created_at_utc",
        "opened_by",
        "emergency_resource_receipt",
    )
    if any(current[key] != previous[key] for key in immutable):
        raise SafeguardingIntegrityError("immutable safeguarding case fields changed")
    if not _same_actor(current["last_updated_by"], event_actor):
        raise SafeguardingIntegrityError("case update actor binding is invalid")
    if _parse_utc(current["updated_at_utc"], field="case.updated_at_utc") < _parse_utc(
        previous["updated_at_utc"], field="case.updated_at_utc"
    ):
        raise SafeguardingIntegrityError("case update timestamp moved backwards")

    prior_escalation = previous["escalation"]
    next_escalation = current["escalation"]
    if event_type == "case.acknowledged":
        if previous["status"] != "open" or current["status"] != "acknowledged":
            raise SafeguardingIntegrityError(
                "case acknowledgement transition is invalid"
            )
        if next_escalation != prior_escalation:
            raise SafeguardingIntegrityError("case acknowledgement changed escalation")
    elif event_type == "case.closed":
        if previous["status"] != "acknowledged" or current["status"] != "closed":
            raise SafeguardingIntegrityError("case close transition is invalid")
        if next_escalation != prior_escalation:
            raise SafeguardingIntegrityError("case close changed escalation")
    elif event_type == "escalation.overdue":
        if current["status"] != previous["status"]:
            raise SafeguardingIntegrityError("overdue transition changed case status")
        if (
            prior_escalation["delivery_status"] != "pending"
            or next_escalation["delivery_status"] != "overdue"
        ):
            raise SafeguardingIntegrityError("escalation overdue transition is invalid")
        unchanged = set(_ESCALATION_KEYS).difference(
            {"delivery_status", "overdue_recorded_at_utc"}
        )
        if any(next_escalation[key] != prior_escalation[key] for key in unchanged):
            raise SafeguardingIntegrityError(
                "escalation overdue transition changed immutable delivery fields"
            )
        if next_escalation["overdue_recorded_at_utc"] != occurred_at_utc:
            raise SafeguardingIntegrityError(
                "escalation overdue time is not bound to its durable event"
            )
    elif event_type == "escalation.acknowledged":
        if current["status"] != previous["status"]:
            raise SafeguardingIntegrityError("delivery ack changed case status")
        if (
            prior_escalation["delivery_status"] not in {"pending", "overdue"}
            or next_escalation["delivery_status"] != "acknowledged"
        ):
            raise SafeguardingIntegrityError(
                "escalation acknowledgement transition is invalid"
            )
        unchanged = set(_ESCALATION_KEYS).difference(
            {"delivery_status", "acknowledged_at_utc", "acknowledged_by"}
        )
        if any(next_escalation[key] != prior_escalation[key] for key in unchanged):
            raise SafeguardingIntegrityError(
                "escalation acknowledgement changed immutable delivery fields"
            )
        if next_escalation["acknowledged_at_utc"] != occurred_at_utc or not _same_actor(
            next_escalation["acknowledged_by"], event_actor
        ):
            raise SafeguardingIntegrityError(
                "escalation acknowledgement actor/time binding is invalid"
            )
    else:
        raise SafeguardingIntegrityError("safeguarding event type is unsupported")


def _validate_event(
    value: Any,
    *,
    sequence: int,
    previous_event_sha256: str | None,
    case_projection: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _EVENT_KEYS:
        raise SafeguardingIntegrityError("safeguarding event shape is invalid")
    event = deepcopy(dict(value))
    if event.get("schema") != SAFEGUARDING_EVENT_SCHEMA:
        raise SafeguardingIntegrityError("safeguarding event schema is unsupported")
    if event.get("sequence") != sequence:
        raise SafeguardingIntegrityError("safeguarding event sequence is invalid")
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or _EVENT_ID.fullmatch(event_id) is None:
        raise SafeguardingIntegrityError("safeguarding event ID is invalid")
    if event.get("event_type") not in {
        "case.opened",
        "case.acknowledged",
        "case.closed",
        "escalation.overdue",
        "escalation.acknowledged",
    }:
        raise SafeguardingIntegrityError("safeguarding event type is unsupported")
    _parse_utc(event.get("occurred_at_utc"), field="event.occurred_at_utc")
    _require_digest(
        event.get("idempotency_key_sha256"), field="event.idempotency_key_sha256"
    )
    _require_digest(event.get("request_sha256"), field="event.request_sha256")
    actor = _validate_actor(event.get("actor"), field="event.actor")
    if event.get("previous_event_sha256") != previous_event_sha256:
        raise SafeguardingIntegrityError("safeguarding event hash chain is invalid")
    declared = _require_digest(event.get("event_sha256"), field="event.event_sha256")
    if declared != _event_hash(event):
        raise SafeguardingIntegrityError("safeguarding event integrity check failed")
    case = _validate_case(event.get("case_snapshot"), field="event.case_snapshot")
    prior = case_projection.get(case["case_id"])
    expected_event_id = (
        "sgev_"
        + canonical_sha256(
            {
                "sequence": sequence,
                "event_type": event["event_type"],
                "request_sha256": event["request_sha256"],
                "previous": previous_event_sha256,
            }
        )[:24]
    )
    if event_id != expected_event_id:
        raise SafeguardingIntegrityError("safeguarding event identity is invalid")
    if prior is None:
        expected_case_id = (
            "sgc_"
            + canonical_sha256(
                {
                    "idempotency_key_sha256": event["idempotency_key_sha256"],
                    "request_sha256": event["request_sha256"],
                }
            )[:24]
        )
        expected_request = canonical_sha256(
            {
                "operation": "case.opened",
                "scope_sha256": case["scope_sha256"],
                "category": case["category"],
                "severity": case["severity"],
                "observed_at_utc": case["observed_at_utc"],
                "content_sha256": case["content_sha256"],
                "actor_principal_sha256": actor["principal_sha256"],
                "actor_role_sha256": actor["role_sha256"],
            }
        )
        if case["case_id"] != expected_case_id:
            raise SafeguardingIntegrityError("safeguarding case identity is invalid")
    else:
        expected_request = canonical_sha256(
            {
                "operation": event["event_type"],
                "case_id": case["case_id"],
                "expected_version": prior["version"],
                "actor_principal_sha256": actor["principal_sha256"],
                "actor_role_sha256": actor["role_sha256"],
            }
        )
    if event["request_sha256"] != expected_request:
        raise SafeguardingIntegrityError(
            "safeguarding event request binding is invalid"
        )
    _validate_transition(
        event_type=event["event_type"],
        previous=prior,
        current=case,
        event_actor=actor,
        occurred_at_utc=event["occurred_at_utc"],
    )
    case_projection[case["case_id"]] = case
    return event


def _validate_tombstone(value: Any, *, key: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _TOMBSTONE_KEYS:
        raise SafeguardingIntegrityError("safeguarding erasure tombstone is invalid")
    result = deepcopy(dict(value))
    if result.get("schema") != SAFEGUARDING_ERASURE_TOMBSTONE_SCHEMA:
        raise SafeguardingIntegrityError("safeguarding tombstone schema is unsupported")
    if (
        _require_digest(result.get("case_id_sha256"), field="tombstone.case_id_sha256")
        != key
    ):
        raise SafeguardingIntegrityError("safeguarding tombstone key is inconsistent")
    _parse_utc(result.get("purged_at_utc"), field="tombstone.purged_at_utc")
    generation = result.get("generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
    ):
        raise SafeguardingIntegrityError("safeguarding tombstone generation is invalid")
    expected_version = result.get("expected_version")
    if (
        isinstance(expected_version, bool)
        or not isinstance(expected_version, int)
        or expected_version < 1
    ):
        raise SafeguardingIntegrityError(
            "safeguarding tombstone expected_version is invalid"
        )
    for field in (
        "idempotency_key_sha256",
        "request_sha256",
        "authorization_receipt_sha256",
    ):
        _require_digest(result.get(field), field=f"tombstone.{field}")
    declared = _require_digest(
        result.get("tombstone_sha256"), field="tombstone.tombstone_sha256"
    )
    actor = _validate_actor(result.get("actor"), field="tombstone.actor")
    if (
        actor["actor_kind"] != _STAFF_ACTOR_KIND
        or actor["operation"] != "case.purged"
        or actor["authorization_receipt_sha256"]
        != result["authorization_receipt_sha256"]
    ):
        raise SafeguardingIntegrityError(
            "safeguarding purge actor authority is invalid"
        )
    expected_body = canonical_sha256(
        {
            "operation": "case.purged",
            "case_id_sha256": key,
            "expected_version": expected_version,
        }
    )
    if actor["body_sha256"] != expected_body:
        raise SafeguardingIntegrityError(
            "safeguarding purge authority is not bound to the erased case"
        )
    expected_request = canonical_sha256(
        {
            "operation": "case.purged",
            "case_id_sha256": key,
            "expected_version": expected_version,
            "actor_principal_sha256": actor["principal_sha256"],
            "actor_role_sha256": actor["role_sha256"],
        }
    )
    if result["request_sha256"] != expected_request:
        raise SafeguardingIntegrityError(
            "safeguarding purge request binding is invalid"
        )
    if declared != _tombstone_hash(result):
        raise SafeguardingIntegrityError(
            "safeguarding tombstone integrity check failed"
        )
    return result


def _validate_erasure_fence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _ERASURE_FENCE_KEYS:
        raise SafeguardingIntegrityError("safeguarding erasure fence is invalid")
    result = deepcopy(dict(value))
    if (
        result.get("schema") != SAFEGUARDING_ERASURE_FENCE_SCHEMA
        or result.get("algorithm") != "bloom_sha256_double_hash_v1"
        or result.get("bit_count") != _ERASURE_FENCE_BITS
        or result.get("hash_count") != _ERASURE_FENCE_HASHES
        or result.get("false_negative_possible") is not False
        or result.get("false_positive_policy") != "fail_closed_as_erased"
    ):
        raise SafeguardingIntegrityError("safeguarding erasure fence policy is invalid")
    for field in ("generation", "inserted_count"):
        value_int = result.get(field)
        minimum = 1 if field == "generation" else 0
        if (
            isinstance(value_int, bool)
            or not isinstance(value_int, int)
            or value_int < minimum
        ):
            raise SafeguardingIntegrityError(
                f"safeguarding erasure fence {field} is invalid"
            )
    _require_digest(result.get("bits_sha256"), field="erasure_fence.bits_sha256")
    _require_digest(
        result.get("accumulator_sha256"),
        field="erasure_fence.accumulator_sha256",
    )
    return result


def _validate_retention_compaction(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _RETENTION_COMPACTION_KEYS:
        raise SafeguardingIntegrityError(
            "safeguarding retention compaction record is invalid"
        )
    result = deepcopy(dict(value))
    if result.get("schema") != SAFEGUARDING_RETENTION_COMPACTION_SCHEMA:
        raise SafeguardingIntegrityError(
            "safeguarding retention compaction schema is invalid"
        )
    generation = result.get("generation")
    cases_total = result.get("cases_compacted_total")
    events_total = result.get("events_compacted_total")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in (generation, cases_total, events_total)
    ):
        raise SafeguardingIntegrityError(
            "safeguarding retention compaction counters are invalid"
        )
    nullable_digests = (
        "last_authorization_receipt_sha256",
        "last_operation_body_sha256",
        "policy_version_sha256",
        "previous_compaction_sha256",
        "compaction_sha256",
    )
    for field in nullable_digests:
        _require_digest(result.get(field), field=f"retention.{field}", nullable=True)
    if generation == 0:
        if result != _empty_retention_compaction():
            raise SafeguardingIntegrityError(
                "empty safeguarding retention compaction record is inconsistent"
            )
        return result
    if cases_total < generation or events_total < cases_total:
        raise SafeguardingIntegrityError(
            "safeguarding retention compaction totals are inconsistent"
        )
    _parse_utc(
        result.get("last_compacted_at_utc"), field="retention.last_compacted_at_utc"
    )
    minimum = result.get("minimum_closed_age_seconds")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or not 1 <= minimum <= 10 * 365 * 24 * 60 * 60
        or any(result.get(field) is None for field in nullable_digests)
        or result["compaction_sha256"] != _retention_compaction_hash(result)
    ):
        raise SafeguardingIntegrityError(
            "safeguarding retention compaction binding is invalid"
        )
    return result


def _validate_state(
    value: Any,
    *,
    capacity: SafeguardingCapacityLimits | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    limits = capacity or SafeguardingCapacityLimits()
    if not isinstance(value, Mapping):
        raise SafeguardingIntegrityError("safeguarding store shape is invalid")
    version = value.get("version")
    expected_keys = {
        "schema",
        "version",
        "event_sequence",
        "events",
        "erasure_tombstones",
    }
    if version == 2:
        expected_keys.update({"erasure_fence", "retention_compaction"})
    if frozenset(value) != expected_keys:
        raise SafeguardingIntegrityError("safeguarding store shape is invalid")
    state = deepcopy(dict(value))
    if (
        (version == 1 and state.get("schema") != SAFEGUARDING_STORE_SCHEMA_V1)
        or (version == 2 and state.get("schema") != SAFEGUARDING_STORE_SCHEMA)
        or version not in {1, 2}
    ):
        raise SafeguardingIntegrityError("safeguarding store schema is unsupported")
    events = state.get("events")
    tombstones = state.get("erasure_tombstones")
    if not isinstance(events, list) or len(events) > limits.maximum_events:
        raise SafeguardingIntegrityError("safeguarding event capacity is invalid")
    if state.get("event_sequence") != len(events):
        raise SafeguardingIntegrityError("safeguarding event head is invalid")
    maximum_tombstones = _MAX_TOMBSTONES if version == 1 else limits.recent_tombstones
    if not isinstance(tombstones, Mapping) or len(tombstones) > maximum_tombstones:
        raise SafeguardingIntegrityError("safeguarding tombstone capacity is invalid")
    normalized_tombstones: dict[str, dict[str, Any]] = {}
    for key, tombstone in tombstones.items():
        _require_digest(key, field="erasure_tombstones key")
        normalized_tombstones[key] = _validate_tombstone(tombstone, key=key)

    projection: dict[str, dict[str, Any]] = {}
    idempotency: dict[str, str] = {}
    previous_hash: str | None = None
    for sequence, raw_event in enumerate(events, start=1):
        event = _validate_event(
            raw_event,
            sequence=sequence,
            previous_event_sha256=previous_hash,
            case_projection=projection,
        )
        key = event["idempotency_key_sha256"]
        request_hash = event["request_sha256"]
        if key in idempotency:
            raise SafeguardingIntegrityError(
                "safeguarding idempotency key has multiple durable events"
            )
        idempotency[key] = request_hash
        previous_hash = event["event_sha256"]

    for case_id in projection:
        if sha256(case_id.encode("utf-8")).hexdigest() in normalized_tombstones:
            raise SafeguardingIntegrityError(
                "purged safeguarding case was resurrected in the active ledger"
            )
    if version == 2:
        _validate_erasure_fence(state.get("erasure_fence"))
        _validate_retention_compaction(state.get("retention_compaction"))
    if len(_canonical_bytes(state)) > limits.maximum_store_bytes:
        raise SafeguardingIntegrityError("safeguarding store exceeds byte capacity")
    return state, projection


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock_for(path: Path) -> threading.RLock:
    identity = str(path)
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(identity)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[identity] = lock
        return lock


class TeacherAgentSafeguardingStore:
    """Cross-process durable safeguarding case/outbox store."""

    def __init__(
        self,
        root: str | Path,
        *,
        authorization_verifier: AuthorizationVerifier,
        emergency_resource_policy: EmergencyResourcePolicy,
        escalation_delivery: EscalationDeliveryConfig | None = None,
        retention_authorization_verifier: RetentionAuthorizationVerifier | None = None,
        capacity_limits: SafeguardingCapacityLimits | None = None,
        clock: Clock = _default_clock,
    ) -> None:
        if not callable(authorization_verifier):
            raise SafeguardingConfigurationError(
                "a server authorization verifier is required"
            )
        if not isinstance(emergency_resource_policy, EmergencyResourcePolicy):
            raise SafeguardingConfigurationError(
                "a versioned emergency resource policy is required"
            )
        if escalation_delivery is not None and not isinstance(
            escalation_delivery, EscalationDeliveryConfig
        ):
            raise SafeguardingConfigurationError(
                "escalation_delivery configuration is invalid"
            )
        if not callable(clock):
            raise SafeguardingConfigurationError("safeguarding clock is invalid")
        if retention_authorization_verifier is not None and not callable(
            retention_authorization_verifier
        ):
            raise SafeguardingConfigurationError(
                "retention authorization verifier is invalid"
            )
        if capacity_limits is not None and not isinstance(
            capacity_limits, SafeguardingCapacityLimits
        ):
            raise SafeguardingConfigurationError(
                "safeguarding capacity limits are invalid"
            )
        candidate = Path(root).expanduser()
        if candidate.exists() and candidate.is_symlink():
            raise SafeguardingIntegrityError("safeguarding root cannot be a symlink")
        candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
        if candidate.is_symlink() or not candidate.is_dir():
            raise SafeguardingIntegrityError("safeguarding root is unsafe")
        if os.name == "posix":
            candidate.chmod(0o700)
        self.root = candidate.resolve()
        self.path = self.root / "safeguarding_cases.json"
        self.lock_path = self.root / ".safeguarding_cases.lock"
        self._authorization_verifier = authorization_verifier
        self._emergency_resource_policy = emergency_resource_policy
        self._escalation_delivery = escalation_delivery
        self._retention_authorization_verifier = retention_authorization_verifier
        self._capacity = capacity_limits or SafeguardingCapacityLimits()
        self._clock = clock
        self._thread_lock = _thread_lock_for(self.lock_path)

    @staticmethod
    def _validate_private_descriptor(descriptor: int, *, field: str) -> None:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SafeguardingIntegrityError(
                f"{field} must be a single-link regular file"
            )
        if os.name == "posix":
            if metadata.st_uid != os.getuid():
                raise SafeguardingIntegrityError(
                    f"{field} must be owned by current user"
                )
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise SafeguardingIntegrityError(f"{field} must have mode 0600")

    @contextmanager
    def _guard(self) -> Iterator[None]:
        if fcntl is None:
            raise SafeguardingIntegrityError(
                "durable safeguarding storage requires POSIX flock"
            )
        with self._thread_lock:
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(self.lock_path, flags, 0o600)
            except OSError as exc:
                raise SafeguardingIntegrityError(
                    "safeguarding lock file cannot be opened safely"
                ) from exc
            try:
                self._validate_private_descriptor(
                    descriptor, field="safeguarding lock file"
                )
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                self._validate_private_descriptor(
                    descriptor, field="safeguarding lock file"
                )
                yield
            finally:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)

    def _fence_path(self, generation: int) -> Path:
        return self.root / f".safeguarding_erasure_fence.{generation}.bin"

    def _next_fence_generation_locked(self) -> int:
        """Return an unused generation, including after a pre-ledger crash."""

        maximum = 0
        for path in self.root.glob(".safeguarding_erasure_fence.*.bin"):
            match = re.fullmatch(
                r"\.safeguarding_erasure_fence\.([1-9][0-9]*)\.bin",
                path.name,
            )
            if match is not None:
                maximum = max(maximum, int(match.group(1)))
        return maximum + 1

    def _read_fence_locked(self, metadata: Mapping[str, Any]) -> bytes:
        fence = _validate_erasure_fence(metadata)
        path = self._fence_path(int(fence["generation"]))
        if path.is_symlink():
            raise SafeguardingIntegrityError("safeguarding erasure fence is unsafe")
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
        except OSError as exc:
            raise SafeguardingIntegrityError(
                "safeguarding erasure fence is unavailable"
            ) from exc
        try:
            self._validate_private_descriptor(
                descriptor, field="safeguarding erasure fence"
            )
            metadata_stat = os.fstat(descriptor)
            if metadata_stat.st_size != _ERASURE_FENCE_BYTES:
                raise SafeguardingIntegrityError(
                    "safeguarding erasure fence size is invalid"
                )
            chunks: list[bytes] = []
            remaining = _ERASURE_FENCE_BYTES
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise SafeguardingIntegrityError(
                        "safeguarding erasure fence is incomplete"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise SafeguardingIntegrityError(
                    "safeguarding erasure fence size is invalid"
                )
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
        if sha256(payload).hexdigest() != fence["bits_sha256"]:
            raise SafeguardingIntegrityError(
                "safeguarding erasure fence integrity check failed"
            )
        return payload

    def _write_fence_locked(self, bits: bytes, *, generation: int) -> dict[str, Any]:
        if len(bits) != _ERASURE_FENCE_BYTES or generation < 1:
            raise SafeguardingIntegrityError("safeguarding erasure fence is invalid")
        target = self._fence_path(generation)
        expected_sha256 = sha256(bits).hexdigest()
        if target.exists() or target.is_symlink():
            existing = self._read_fence_locked(
                {
                    **_empty_erasure_fence(generation=generation),
                    "bits_sha256": expected_sha256,
                }
            )
            if existing != bits:
                raise SafeguardingIntegrityError(
                    "safeguarding erasure fence generation conflicts"
                )
            return {"bits_sha256": expected_sha256}
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".safeguarding_erasure_fence.", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(bits):
                written = os.write(descriptor, bits[offset:])
                if written < 1:
                    raise OSError("short safeguarding erasure fence write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.link(temporary, target)
            temporary.unlink()
            parent = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise
        return {"bits_sha256": expected_sha256}

    def _ensure_v2_locked(
        self, state: Mapping[str, Any]
    ) -> tuple[dict[str, Any], bytes]:
        normalized, _ = _validate_state(state, capacity=self._capacity)
        if normalized["version"] == 2:
            return normalized, self._read_fence_locked(normalized["erasure_fence"])
        bits = bytearray(_ERASURE_FENCE_BYTES)
        accumulator = sha256(b"safeguarding-erasure-accumulator-v2").hexdigest()
        tombstones = dict(normalized["erasure_tombstones"])
        for case_digest in sorted(tombstones):
            _bloom_add(bits, case_digest)
            accumulator = _advance_erasure_accumulator(accumulator, case_digest)
        generation = self._next_fence_generation_locked()
        fence = _empty_erasure_fence(generation=generation)
        fence.update(
            {
                "inserted_count": len(tombstones),
                "bits_sha256": sha256(bits).hexdigest(),
                "accumulator_sha256": accumulator,
            }
        )
        self._write_fence_locked(bytes(bits), generation=generation)
        recent = sorted(
            tombstones.items(),
            key=lambda item: (str(item[1]["purged_at_utc"]), item[0]),
        )[-self._capacity.recent_tombstones :]
        migrated = {
            "schema": SAFEGUARDING_STORE_SCHEMA,
            "version": 2,
            "event_sequence": normalized["event_sequence"],
            "events": deepcopy(normalized["events"]),
            "erasure_tombstones": dict(recent),
            "erasure_fence": fence,
            "retention_compaction": _empty_retention_compaction(),
        }
        _validate_state(migrated, capacity=self._capacity)
        return migrated, bytes(bits)

    @staticmethod
    def _assert_active_cases_outside_fence(
        projection: Mapping[str, Mapping[str, Any]], bits: bytes
    ) -> None:
        if any(
            _bloom_contains(bits, sha256(case_id.encode("utf-8")).hexdigest())
            for case_id in projection
        ):
            raise SafeguardingIntegrityError(
                "safeguarding erasure fence collides with an active case"
            )

    def _cleanup_old_fences_locked(self, *, current_generation: int) -> None:
        for path in self.root.glob(".safeguarding_erasure_fence.*.bin"):
            if path == self._fence_path(current_generation) or path.is_symlink():
                continue
            match = re.fullmatch(
                r"\.safeguarding_erasure_fence\.([1-9][0-9]*)\.bin", path.name
            )
            if match is not None and path.is_file():
                path.unlink(missing_ok=True)

    def _read_locked(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        if self.path.is_symlink():
            raise SafeguardingIntegrityError(
                "safeguarding store cannot be opened safely"
            )
        if not self.path.exists():
            return _validate_state(_empty_state(), capacity=self._capacity)
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(self.path, flags)
        except OSError as exc:
            raise SafeguardingIntegrityError(
                "safeguarding store cannot be opened safely"
            ) from exc
        try:
            self._validate_private_descriptor(descriptor, field="safeguarding store")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(
                        1024 * 1024,
                        self._capacity.maximum_store_bytes + 1 - total,
                    ),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > self._capacity.maximum_store_bytes:
                    raise SafeguardingIntegrityError(
                        "safeguarding store exceeds byte capacity"
                    )
            raw = b"".join(chunks)
        finally:
            os.close(descriptor)
        try:
            if not raw.endswith(b"\n"):
                raise SafeguardingIntegrityError(
                    "safeguarding store commit is incomplete"
                )
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafeguardingIntegrityError(
                "safeguarding store is not valid JSON"
            ) from exc
        normalized = _validate_state(value, capacity=self._capacity)
        if raw != _canonical_bytes(normalized[0]) + b"\n":
            raise SafeguardingIntegrityError(
                "safeguarding store encoding is not canonical"
            )
        if normalized[0]["version"] == 2:
            bits = self._read_fence_locked(normalized[0]["erasure_fence"])
            self._assert_active_cases_outside_fence(normalized[1], bits)
        return normalized

    def _write_locked(self, state: Mapping[str, Any]) -> None:
        prepared, bits = self._ensure_v2_locked(state)
        normalized, projection = _validate_state(prepared, capacity=self._capacity)
        self._assert_active_cases_outside_fence(projection, bits)
        payload = _canonical_bytes(normalized) + b"\n"
        if len(payload) > self._capacity.maximum_store_bytes:
            raise SafeguardingIntegrityError("safeguarding store exceeds byte capacity")
        if self.path.is_symlink():
            raise SafeguardingIntegrityError("safeguarding store target is unsafe")
        if self.path.exists():
            flags = (
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                existing = os.open(self.path, flags)
            except OSError as exc:
                raise SafeguardingIntegrityError(
                    "safeguarding store target is unsafe"
                ) from exc
            try:
                self._validate_private_descriptor(existing, field="safeguarding store")
            finally:
                os.close(existing)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".safeguarding_cases.", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("short safeguarding journal write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, self.path)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self._cleanup_old_fences_locked(
                current_generation=int(normalized["erasure_fence"]["generation"])
            )
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise

    def _now(self) -> tuple[datetime, str]:
        value = self._clock()
        if not isinstance(value, datetime):
            raise SafeguardingIntegrityError("safeguarding clock must return datetime")
        timestamp = _format_utc(value)
        return _parse_utc(timestamp, field="clock"), timestamp

    def _authorize(
        self,
        value: Mapping[str, Any],
        *,
        now: datetime,
        now_utc: str,
        operation: str,
        body_sha256: str,
        allowed_actor_kinds: frozenset[str],
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise SafeguardingAuthorizationError(
                "server safeguarding authorization receipt is required"
            )
        try:
            verified = self._authorization_verifier(deepcopy(dict(value)))
        except Exception as exc:
            raise SafeguardingAuthorizationError(
                "server rejected safeguarding authorization receipt"
            ) from exc
        if (
            not isinstance(verified, Mapping)
            or frozenset(verified) != _AUTHORIZATION_KEYS
        ):
            raise SafeguardingAuthorizationError(
                "server verifier did not return a normalized hash-only receipt"
            )
        receipt = deepcopy(dict(verified))
        if (
            receipt.get("schema") != SAFEGUARDING_AUTHORIZATION_RECEIPT_SCHEMA
            or receipt.get("server_authenticated") is not True
        ):
            raise SafeguardingAuthorizationError(
                "safeguarding actor is not server-authenticated"
            )
        try:
            for field in (
                "principal_sha256",
                "authorization_context_sha256",
                "role_sha256",
                "body_sha256",
            ):
                _require_digest(receipt.get(field), field=f"authorization.{field}")
            actor_kind = receipt.get("actor_kind")
            if actor_kind not in allowed_actor_kinds:
                raise SafeguardingAuthorizationError(
                    "actor kind is not authorized for this safeguarding operation"
                )
            expected_role = (
                SYSTEM_SAFETY_CLASSIFIER_ROLE_SHA256
                if actor_kind == _SYSTEM_ACTOR_KIND
                else SAFEGUARDING_ROLE_SHA256
            )
            if receipt.get("role_sha256") != expected_role:
                raise SafeguardingAuthorizationError(
                    "server-authenticated actor lacks the required safeguarding authority"
                )
            if receipt.get("operation") != operation:
                raise SafeguardingAuthorizationError(
                    "safeguarding authorization receipt operation does not match"
                )
            if receipt.get("body_sha256") != body_sha256:
                raise SafeguardingAuthorizationError(
                    "safeguarding authorization receipt body does not match"
                )
            issued = _parse_utc(
                receipt.get("issued_at_utc"), field="authorization.issued_at_utc"
            )
            expires = _parse_utc(
                receipt.get("expires_at_utc"), field="authorization.expires_at_utc"
            )
            declared = _require_digest(
                receipt.get("receipt_sha256"), field="authorization.receipt_sha256"
            )
        except SafeguardingIntegrityError as exc:
            raise SafeguardingAuthorizationError(str(exc)) from exc
        if declared != authorization_receipt_sha256(receipt):
            raise SafeguardingAuthorizationError(
                "safeguarding authorization receipt integrity check failed"
            )
        if issued > now + timedelta(seconds=_CLOCK_SKEW_SECONDS):
            raise SafeguardingAuthorizationError(
                "safeguarding authorization receipt is not yet valid"
            )
        if expires <= now:
            raise SafeguardingAuthorizationError(
                "safeguarding authorization receipt has expired"
            )
        if (
            expires <= issued
            or (expires - issued).total_seconds() > _MAX_AUTHORIZATION_TTL_SECONDS
        ):
            raise SafeguardingAuthorizationError(
                "safeguarding authorization receipt lifetime is invalid"
            )
        return _actor_from_authorization(receipt, verified_at_utc=now_utc)

    def _authorize_retention(
        self,
        value: Mapping[str, Any],
        *,
        now: datetime,
        body_sha256: str,
    ) -> dict[str, Any]:
        verifier = self._retention_authorization_verifier
        if verifier is None:
            raise SafeguardingConfigurationError(
                "safeguarding retention authority is not configured"
            )
        if not isinstance(value, Mapping):
            raise SafeguardingAuthorizationError(
                "server retention authorization receipt is required"
            )
        try:
            verified = verifier(deepcopy(dict(value)))
        except Exception as exc:
            raise SafeguardingAuthorizationError(
                "server rejected safeguarding retention authorization receipt"
            ) from exc
        if (
            not isinstance(verified, Mapping)
            or frozenset(verified) != _RETENTION_AUTHORIZATION_KEYS
        ):
            raise SafeguardingAuthorizationError(
                "retention verifier did not return an exact hash-only receipt"
            )
        receipt = deepcopy(dict(verified))
        if (
            receipt.get("schema") != SAFEGUARDING_RETENTION_AUTHORIZATION_RECEIPT_SCHEMA
            or type(receipt.get("version")) is not int
            or receipt.get("version") != 1
            or receipt.get("actor_kind") != _RETENTION_ACTOR_KIND
            or receipt.get("server_authenticated") is not True
            or receipt.get("role_sha256") != SAFEGUARDING_RETENTION_ROLE_SHA256
            or receipt.get("operation") != _RETENTION_OPERATION
            or receipt.get("body_sha256") != body_sha256
        ):
            raise SafeguardingAuthorizationError(
                "retention authorization is not bound to the exact operation"
            )
        try:
            for field in (
                "principal_sha256",
                "role_sha256",
                "authorization_context_sha256",
                "body_sha256",
                "authority_proof_sha256",
            ):
                _require_digest(receipt.get(field), field=f"retention.{field}")
            issued = _parse_utc(
                receipt.get("issued_at_utc"), field="retention.issued_at_utc"
            )
            expires = _parse_utc(
                receipt.get("expires_at_utc"), field="retention.expires_at_utc"
            )
            declared = _require_digest(
                receipt.get("receipt_sha256"), field="retention.receipt_sha256"
            )
        except SafeguardingIntegrityError as exc:
            raise SafeguardingAuthorizationError(str(exc)) from exc
        if declared != retention_authorization_receipt_sha256(receipt):
            raise SafeguardingAuthorizationError(
                "retention authorization receipt integrity check failed"
            )
        if issued > now + timedelta(seconds=_CLOCK_SKEW_SECONDS):
            raise SafeguardingAuthorizationError(
                "retention authorization receipt is not yet valid"
            )
        if (
            expires <= now
            or expires <= issued
            or (expires - issued).total_seconds() > _MAX_AUTHORIZATION_TTL_SECONDS
        ):
            raise SafeguardingAuthorizationError(
                "retention authorization receipt lifetime is invalid"
            )
        return receipt

    def _case_is_erased_locked(
        self,
        state: Mapping[str, Any],
        case_id_sha256: str,
        *,
        fence_bits: bytes | None = None,
    ) -> bool:
        if case_id_sha256 in state["erasure_tombstones"]:
            return True
        if state["version"] == 1:
            return False
        bits = (
            fence_bits
            if fence_bits is not None
            else self._read_fence_locked(state["erasure_fence"])
        )
        # A Bloom hit is deliberately treated as erased.  This can deny a new
        # identity on a false positive, but can never resurrect an erased case.
        return _bloom_contains(bits, case_id_sha256)

    @staticmethod
    def _rebuild_events(
        events: Sequence[Mapping[str, Any]], *, removed_case_ids: frozenset[str]
    ) -> tuple[list[dict[str, Any]], int]:
        retained = [
            deepcopy(dict(event))
            for event in events
            if event["case_snapshot"]["case_id"] not in removed_case_ids
        ]
        rebuilt: list[dict[str, Any]] = []
        previous: str | None = None
        for sequence, source in enumerate(retained, start=1):
            event = deepcopy(source)
            event["sequence"] = sequence
            event["previous_event_sha256"] = previous
            event["event_id"] = (
                "sgev_"
                + canonical_sha256(
                    {
                        "sequence": sequence,
                        "event_type": event["event_type"],
                        "request_sha256": event["request_sha256"],
                        "previous": previous,
                    }
                )[:24]
            )
            event["event_sha256"] = _event_hash(event)
            rebuilt.append(event)
            previous = event["event_sha256"]
        return rebuilt, len(events) - len(retained)

    def _retention_plan_locked(
        self,
        state: Mapping[str, Any],
        projection: Mapping[str, Mapping[str, Any]],
        *,
        policy: SafeguardingRetentionPolicy,
        as_of: datetime,
        as_of_utc: str,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        candidates: list[tuple[datetime, str, Mapping[str, Any]]] = []
        for case_id, case in projection.items():
            closed_at = _parse_utc(
                case["updated_at_utc"], field="retention.closed_at_utc"
            )
            if (
                case["status"] == "closed"
                and case["escalation"]["delivery_status"] == "acknowledged"
                and as_of
                >= closed_at + timedelta(seconds=policy.minimum_closed_age_seconds)
            ):
                candidates.append(
                    (
                        closed_at,
                        sha256(case_id.encode("utf-8")).hexdigest(),
                        case,
                    )
                )
        candidates.sort(key=lambda item: (item[0], item[1]))
        selected = candidates[: policy.maximum_cases_per_run]
        selected_ids = tuple(str(item[2]["case_id"]) for item in selected)
        selected_set = frozenset(selected_ids)
        selected_event_count = sum(
            event["case_snapshot"]["case_id"] in selected_set
            for event in state["events"]
        )
        selection_sha256 = canonical_sha256(
            [
                {
                    "case_id_sha256": case_digest,
                    "expected_version": int(case["version"]),
                    "closed_at_utc": case["updated_at_utc"],
                }
                for _, case_digest, case in selected
            ]
        )
        body = {
            "operation": _RETENTION_OPERATION,
            "store_schema": SAFEGUARDING_STORE_SCHEMA,
            "store_version": 2,
            "expected_event_sequence": int(state["event_sequence"]),
            "expected_event_head_sha256": (
                state["events"][-1]["event_sha256"] if state["events"] else None
            ),
            "expected_erasure_fence_generation": int(
                state["erasure_fence"]["generation"]
            ),
            "expected_retention_compaction_generation": int(
                state["retention_compaction"]["generation"]
            ),
            "as_of_utc": as_of_utc,
            "policy_version_sha256": policy.policy_version_sha256,
            "minimum_closed_age_seconds": policy.minimum_closed_age_seconds,
            "eligible_case_count": len(candidates),
            "selected_case_count": len(selected),
            "selected_event_count": int(selected_event_count),
            "selection_sha256": selection_sha256,
        }
        return (
            {
                "schema": SAFEGUARDING_RETENTION_PLAN_SCHEMA,
                **body,
                "body_sha256": canonical_sha256(body),
            },
            selected_ids,
        )

    @staticmethod
    def _find_idempotency_event(
        state: Mapping[str, Any], *, key_sha256: str, request_sha256: str
    ) -> dict[str, Any] | None:
        matches = [
            event
            for event in state["events"]
            if event["idempotency_key_sha256"] == key_sha256
        ]
        if len(matches) > 1:
            raise SafeguardingIntegrityError(
                "idempotency key has multiple durable safeguarding events"
            )
        if not matches:
            return None
        event = matches[0]
        if event["request_sha256"] != request_sha256:
            raise SafeguardingConflictError(
                "idempotency_key was reused with different content"
            )
        return deepcopy(event)

    @staticmethod
    def _idempotency_digest(value: Any) -> str:
        if not isinstance(value, str) or not 8 <= len(value) <= 160:
            raise SafeguardingIntegrityError(
                "idempotency_key must contain 8..160 safe characters"
            )
        if _SAFE_TOKEN.fullmatch(value) is None:
            raise SafeguardingIntegrityError("idempotency_key is invalid")
        return sha256(value.encode("utf-8")).hexdigest()

    def _append_event_locked(
        self,
        state: dict[str, Any],
        *,
        event_type: str,
        now_utc: str,
        idempotency_key_sha256: str,
        request_sha256: str,
        case_snapshot: Mapping[str, Any],
        actor: Mapping[str, Any],
    ) -> dict[str, Any]:
        if len(state["events"]) >= self._capacity.maximum_events:
            raise SafeguardingIntegrityError("safeguarding event capacity is exhausted")
        sequence = len(state["events"]) + 1
        previous = state["events"][-1]["event_sha256"] if state["events"] else None
        event = {
            "schema": SAFEGUARDING_EVENT_SCHEMA,
            "sequence": sequence,
            "event_id": "sgev_"
            + canonical_sha256(
                {
                    "sequence": sequence,
                    "event_type": event_type,
                    "request_sha256": request_sha256,
                    "previous": previous,
                }
            )[:24],
            "event_type": event_type,
            "occurred_at_utc": now_utc,
            "idempotency_key_sha256": idempotency_key_sha256,
            "request_sha256": request_sha256,
            "case_snapshot": deepcopy(dict(case_snapshot)),
            "actor": deepcopy(dict(actor)),
            "previous_event_sha256": previous,
        }
        event["event_sha256"] = _event_hash(event)
        state["events"].append(event)
        state["event_sequence"] = sequence
        self._write_locked(state)
        return deepcopy(event)

    def open_case(
        self,
        *,
        scope_sha256: str,
        category: str,
        severity: str,
        observed_at_utc: str,
        content_sha256: str,
        idempotency_key: str,
        authorization_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Open and, when configured, atomically enqueue a minimized case.

        ``scope_sha256`` is an opaque scope derived by the authenticated server;
        it must never be copied from a learner-controlled browser field.
        """

        _require_digest(scope_sha256, field="scope_sha256")
        _require_digest(content_sha256, field="content_sha256")
        if category not in SAFEGUARDING_CATEGORIES:
            raise SafeguardingIntegrityError("safeguarding category is unsupported")
        if severity not in SAFEGUARDING_SEVERITIES:
            raise SafeguardingIntegrityError("safeguarding severity is unsupported")
        observed = _parse_utc(observed_at_utc, field="observed_at_utc")
        key_digest = self._idempotency_digest(idempotency_key)
        now, now_utc = self._now()
        if observed > now + timedelta(seconds=_CLOCK_SKEW_SECONDS):
            raise SafeguardingIntegrityError(
                "safeguarding observation cannot be in the future"
            )
        authorization_body_sha256 = safeguarding_open_authorization_body_sha256(
            scope_sha256=scope_sha256,
            category=category,
            severity=severity,
            observed_at_utc=observed_at_utc,
            content_sha256=content_sha256,
        )
        actor = self._authorize(
            authorization_receipt,
            now=now,
            now_utc=now_utc,
            operation="case.opened",
            body_sha256=authorization_body_sha256,
            allowed_actor_kinds=frozenset({_SYSTEM_ACTOR_KIND}),
        )
        request_material = {
            "operation": "case.opened",
            "scope_sha256": scope_sha256,
            "category": category,
            "severity": severity,
            "observed_at_utc": observed_at_utc,
            "content_sha256": content_sha256,
            "actor_principal_sha256": actor["principal_sha256"],
            "actor_role_sha256": actor["role_sha256"],
        }
        request_hash = canonical_sha256(request_material)
        case_id = (
            "sgc_"
            + canonical_sha256(
                {
                    "idempotency_key_sha256": key_digest,
                    "request_sha256": request_hash,
                }
            )[:24]
        )
        with self._guard():
            state, _ = self._read_locked()
            replay = self._find_idempotency_event(
                state, key_sha256=key_digest, request_sha256=request_hash
            )
            if replay is not None:
                if replay["event_type"] != "case.opened":
                    raise SafeguardingConflictError(
                        "idempotency_key was reused for another operation"
                    )
                return deepcopy(replay["case_snapshot"])
            tombstone_key = sha256(case_id.encode("utf-8")).hexdigest()
            if self._case_is_erased_locked(state, tombstone_key):
                raise SafeguardingErasedError(
                    "safeguarding case identity was permanently erased"
                )
            bundle = self._emergency_resource_policy.resolve(scope_sha256)
            if self._escalation_delivery is None:
                escalation = _unavailable_escalation()
            else:
                sla_seconds = self._escalation_delivery.sla_seconds_by_severity[
                    severity
                ]
                escalation = {
                    "delivery_status": "pending",
                    "reason_code": None,
                    "policy_version": self._escalation_delivery.policy_version,
                    "queue_sha256": self._escalation_delivery.queue_sha256,
                    "delivery_id": "sge_"
                    + canonical_sha256(
                        {
                            "case_id": case_id,
                            "queue_sha256": self._escalation_delivery.queue_sha256,
                        }
                    )[:24],
                    "sla_seconds": sla_seconds,
                    "enqueued_at_utc": now_utc,
                    "sla_due_at_utc": _format_utc(now + timedelta(seconds=sla_seconds)),
                    "overdue_recorded_at_utc": None,
                    "acknowledged_at_utc": None,
                    "acknowledged_by": None,
                }
            case = {
                "schema": SAFEGUARDING_CASE_SCHEMA,
                "case_id": case_id,
                "version": 1,
                "status": "open",
                "scope_sha256": scope_sha256,
                "category": category,
                "severity": severity,
                "observed_at_utc": observed_at_utc,
                "content_sha256": content_sha256,
                "created_at_utc": now_utc,
                "updated_at_utc": now_utc,
                "opened_by": actor,
                "last_updated_by": actor,
                "escalation": escalation,
                "emergency_resource_receipt": _resource_receipt(bundle),
                "previous_version_sha256": None,
            }
            case["case_sha256"] = _case_hash(case)
            event = self._append_event_locked(
                state,
                event_type="case.opened",
                now_utc=now_utc,
                idempotency_key_sha256=key_digest,
                request_sha256=request_hash,
                case_snapshot=case,
                actor=actor,
            )
            return deepcopy(event["case_snapshot"])

    def _transition_case(
        self,
        *,
        case_id: str,
        expected_version: int,
        event_type: str,
        idempotency_key: str,
        authorization_receipt: Mapping[str, Any],
        apply: Callable[[dict[str, Any], dict[str, Any], datetime, str], None],
    ) -> dict[str, Any]:
        if not isinstance(case_id, str) or _CASE_ID.fullmatch(case_id) is None:
            raise SafeguardingIntegrityError("case_id is invalid")
        if (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version < 1
        ):
            raise SafeguardingIntegrityError("expected_version is invalid")
        key_digest = self._idempotency_digest(idempotency_key)
        now, now_utc = self._now()
        authorization_body_sha256 = safeguarding_case_authorization_body_sha256(
            operation=event_type,
            case_id=case_id,
            expected_version=expected_version,
        )
        actor = self._authorize(
            authorization_receipt,
            now=now,
            now_utc=now_utc,
            operation=event_type,
            body_sha256=authorization_body_sha256,
            allowed_actor_kinds=frozenset({_STAFF_ACTOR_KIND}),
        )
        request_hash = canonical_sha256(
            {
                "operation": event_type,
                "case_id": case_id,
                "expected_version": expected_version,
                "actor_principal_sha256": actor["principal_sha256"],
                "actor_role_sha256": actor["role_sha256"],
            }
        )
        with self._guard():
            state, projection = self._read_locked()
            replay = self._find_idempotency_event(
                state, key_sha256=key_digest, request_sha256=request_hash
            )
            if replay is not None:
                if replay["event_type"] != event_type:
                    raise SafeguardingConflictError(
                        "idempotency_key was reused for another operation"
                    )
                return deepcopy(replay["case_snapshot"])
            current = projection.get(case_id)
            if current is None:
                if self._case_is_erased_locked(
                    state, sha256(case_id.encode("utf-8")).hexdigest()
                ):
                    raise SafeguardingErasedError(
                        "safeguarding case was permanently erased"
                    )
                raise SafeguardingNotFoundError("safeguarding case was not found")
            if current["version"] != expected_version:
                raise SafeguardingConflictError(
                    "safeguarding case expected_version conflict"
                )
            updated = deepcopy(current)
            apply(updated, actor, now, now_utc)
            updated["version"] = current["version"] + 1
            updated["updated_at_utc"] = now_utc
            updated["last_updated_by"] = actor
            updated["previous_version_sha256"] = current["case_sha256"]
            updated["case_sha256"] = _case_hash(updated)
            event = self._append_event_locked(
                state,
                event_type=event_type,
                now_utc=now_utc,
                idempotency_key_sha256=key_digest,
                request_sha256=request_hash,
                case_snapshot=updated,
                actor=actor,
            )
            return deepcopy(event["case_snapshot"])

    def acknowledge_case(
        self,
        *,
        case_id: str,
        expected_version: int,
        idempotency_key: str,
        authorization_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        def apply(
            case: dict[str, Any],
            _actor: dict[str, Any],
            _now: datetime,
            _now_utc: str,
        ) -> None:
            if case["status"] != "open":
                raise SafeguardingConflictError(
                    "only an open safeguarding case can be acknowledged"
                )
            case["status"] = "acknowledged"

        return self._transition_case(
            case_id=case_id,
            expected_version=expected_version,
            event_type="case.acknowledged",
            idempotency_key=idempotency_key,
            authorization_receipt=authorization_receipt,
            apply=apply,
        )

    def close_case(
        self,
        *,
        case_id: str,
        expected_version: int,
        idempotency_key: str,
        authorization_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        def apply(
            case: dict[str, Any],
            _actor: dict[str, Any],
            _now: datetime,
            _now_utc: str,
        ) -> None:
            if case["status"] != "acknowledged":
                raise SafeguardingConflictError(
                    "only an acknowledged safeguarding case can be closed"
                )
            if case["escalation"]["delivery_status"] not in {
                "escalation_unavailable",
                "acknowledged",
            }:
                raise SafeguardingConflictError(
                    "configured escalation delivery must be acknowledged before close"
                )
            case["status"] = "closed"

        return self._transition_case(
            case_id=case_id,
            expected_version=expected_version,
            event_type="case.closed",
            idempotency_key=idempotency_key,
            authorization_receipt=authorization_receipt,
            apply=apply,
        )

    def record_escalation_overdue(
        self,
        *,
        case_id: str,
        expected_version: int,
        idempotency_key: str,
        authorization_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Durably record an SLA breach after the trusted server clock reaches it."""

        def apply(
            case: dict[str, Any],
            _actor: dict[str, Any],
            now: datetime,
            now_utc: str,
        ) -> None:
            escalation = case["escalation"]
            if escalation["delivery_status"] == "escalation_unavailable":
                raise SafeguardingConfigurationError(
                    "escalation_unavailable: delivery is not configured"
                )
            if escalation["delivery_status"] != "pending":
                raise SafeguardingConflictError(
                    "only a pending escalation can become overdue"
                )
            due = _parse_utc(
                escalation["sla_due_at_utc"], field="escalation.sla_due_at_utc"
            )
            if now < due:
                raise SafeguardingConflictError(
                    "escalation SLA deadline has not been reached"
                )
            escalation["delivery_status"] = "overdue"
            escalation["overdue_recorded_at_utc"] = now_utc

        return self._transition_case(
            case_id=case_id,
            expected_version=expected_version,
            event_type="escalation.overdue",
            idempotency_key=idempotency_key,
            authorization_receipt=authorization_receipt,
            apply=apply,
        )

    def acknowledge_escalation(
        self,
        *,
        case_id: str,
        expected_version: int,
        idempotency_key: str,
        authorization_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Durably acknowledge outbox delivery, including after an SLA breach."""

        def apply(
            case: dict[str, Any],
            actor: dict[str, Any],
            _now: datetime,
            now_utc: str,
        ) -> None:
            escalation = case["escalation"]
            if escalation["delivery_status"] == "escalation_unavailable":
                raise SafeguardingConfigurationError(
                    "escalation_unavailable: delivery is not configured"
                )
            if escalation["delivery_status"] not in {"pending", "overdue"}:
                raise SafeguardingConflictError(
                    "escalation delivery is already acknowledged"
                )
            escalation["delivery_status"] = "acknowledged"
            escalation["acknowledged_at_utc"] = now_utc
            escalation["acknowledged_by"] = actor

        return self._transition_case(
            case_id=case_id,
            expected_version=expected_version,
            event_type="escalation.acknowledged",
            idempotency_key=idempotency_key,
            authorization_receipt=authorization_receipt,
            apply=apply,
        )

    def get_case(self, case_id: str) -> dict[str, Any]:
        if not isinstance(case_id, str) or _CASE_ID.fullmatch(case_id) is None:
            raise SafeguardingIntegrityError("case_id is invalid")
        with self._guard():
            state, projection = self._read_locked()
            case = projection.get(case_id)
            if case is None:
                if self._case_is_erased_locked(
                    state, sha256(case_id.encode("utf-8")).hexdigest()
                ):
                    raise SafeguardingErasedError(
                        "safeguarding case was permanently erased"
                    )
                raise SafeguardingNotFoundError("safeguarding case was not found")
            return deepcopy(case)

    def case_for_open_idempotency_key(
        self,
        idempotency_key: str,
        *,
        scope_sha256: str,
        category: str,
        severity: str,
        content_sha256: str,
    ) -> dict[str, Any] | None:
        """Return the current case only for an exact durable open replay.

        Content hashes intentionally are not deduplication identities: the same
        disclosure can recur after a case is closed.  The server-owned request
        idempotency key is the occurrence identity, so an exact retry (including
        after restart) resolves to its current projection while a new request
        opens a new case even when category/content/severity are identical.
        """

        key_digest = self._idempotency_digest(idempotency_key)
        _require_digest(scope_sha256, field="scope_sha256")
        _require_digest(content_sha256, field="content_sha256")
        if category not in SAFEGUARDING_CATEGORIES:
            raise SafeguardingIntegrityError("safeguarding category is unsupported")
        if severity not in SAFEGUARDING_SEVERITIES:
            raise SafeguardingIntegrityError("safeguarding severity is unsupported")
        with self._guard():
            state, projection = self._read_locked()
            matches = [
                event
                for event in state["events"]
                if event["idempotency_key_sha256"] == key_digest
            ]
            if len(matches) > 1:
                raise SafeguardingIntegrityError(
                    "idempotency key has multiple durable safeguarding events"
                )
            if not matches:
                return None
            event = matches[0]
            if event["event_type"] != "case.opened":
                raise SafeguardingConflictError(
                    "idempotency_key was reused for another operation"
                )
            case_id = event["case_snapshot"]["case_id"]
            current = projection.get(case_id)
            if current is None:
                raise SafeguardingErasedError(
                    "safeguarding case identity was permanently erased"
                )
            if any(
                current.get(field) != expected
                for field, expected in (
                    ("scope_sha256", scope_sha256),
                    ("category", category),
                    ("severity", severity),
                    ("content_sha256", content_sha256),
                )
            ):
                raise SafeguardingConflictError(
                    "idempotency_key was reused for a different safeguarding observation"
                )
            return deepcopy(current)

    def list_cases(self, *, status: str | None = None) -> list[dict[str, Any]]:
        if status is not None and status not in SAFEGUARDING_CASE_STATUSES:
            raise SafeguardingIntegrityError("safeguarding case status is invalid")
        with self._guard():
            _, projection = self._read_locked()
        return [
            deepcopy(projection[case_id])
            for case_id in sorted(projection)
            if status is None or projection[case_id]["status"] == status
        ]

    def list_escalations_due(
        self, *, as_of_utc: str | None = None
    ) -> list[dict[str, Any]]:
        """Return pending deliveries whose SLA has elapsed; no state is inferred away."""

        if as_of_utc is None:
            now, _ = self._now()
        else:
            now = _parse_utc(as_of_utc, field="as_of_utc")
        with self._guard():
            _, projection = self._read_locked()
        due: list[dict[str, Any]] = []
        for case_id in sorted(projection):
            case = projection[case_id]
            escalation = case["escalation"]
            if escalation["delivery_status"] != "pending":
                continue
            deadline = _parse_utc(
                escalation["sla_due_at_utc"], field="escalation.sla_due_at_utc"
            )
            if now >= deadline:
                due.append(
                    {
                        "case_id": case_id,
                        "version": case["version"],
                        "delivery_id": escalation["delivery_id"],
                        "sla_due_at_utc": escalation["sla_due_at_utc"],
                        "overdue": True,
                    }
                )
        return due

    def pending_escalations(self) -> list[dict[str, Any]]:
        """Return the content-free durable outbox for an external dispatcher."""

        with self._guard():
            _, projection = self._read_locked()
        rows: list[dict[str, Any]] = []
        for case_id in sorted(projection):
            case = projection[case_id]
            escalation = case["escalation"]
            if escalation["delivery_status"] not in {"pending", "overdue"}:
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "case_version": case["version"],
                    "scope_sha256": case["scope_sha256"],
                    "category": case["category"],
                    "severity": case["severity"],
                    "observed_at_utc": case["observed_at_utc"],
                    "content_sha256": case["content_sha256"],
                    "delivery_id": escalation["delivery_id"],
                    "delivery_status": escalation["delivery_status"],
                    "queue_sha256": escalation["queue_sha256"],
                    "sla_due_at_utc": escalation["sla_due_at_utc"],
                }
            )
        return rows

    def emergency_resources_for_case(self, case_id: str) -> dict[str, Any]:
        """Materialize current configured text only when its stored receipt matches."""

        case = self.get_case(case_id)
        bundle = self._emergency_resource_policy.resolve(case["scope_sha256"])
        if _resource_receipt(bundle) != case["emergency_resource_receipt"]:
            raise SafeguardingConfigurationError(
                "emergency resource policy changed without a new policy version"
            )
        return bundle

    def retention_plan(
        self,
        policy: SafeguardingRetentionPolicy,
        *,
        as_of_utc: str | None = None,
    ) -> dict[str, Any]:
        """Return a hash-only, state-bound plan for the retention authority."""

        if not isinstance(policy, SafeguardingRetentionPolicy):
            raise SafeguardingConfigurationError(
                "safeguarding retention policy is required"
            )
        now, now_utc = self._now()
        if as_of_utc is None:
            as_of = now
            effective_as_of = now_utc
        else:
            as_of = _parse_utc(as_of_utc, field="retention.as_of_utc")
            effective_as_of = _format_utc(as_of)
            if as_of > now:
                raise SafeguardingIntegrityError(
                    "retention as_of_utc cannot be in the future"
                )
        with self._guard():
            state, projection = self._read_locked()
            if state["version"] == 1:
                self._write_locked(state)
                state, projection = self._read_locked()
            plan, _ = self._retention_plan_locked(
                state,
                projection,
                policy=policy,
                as_of=as_of,
                as_of_utc=effective_as_of,
            )
            return deepcopy(plan)

    def compact_retained_cases(
        self,
        policy: SafeguardingRetentionPolicy,
        *,
        as_of_utc: str,
        authorization_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compact only fully delivered closed cases after the configured minimum."""

        if not isinstance(policy, SafeguardingRetentionPolicy):
            raise SafeguardingConfigurationError(
                "safeguarding retention policy is required"
            )
        now, now_utc = self._now()
        as_of = _parse_utc(as_of_utc, field="retention.as_of_utc")
        effective_as_of = _format_utc(as_of)
        if as_of > now:
            raise SafeguardingIntegrityError(
                "retention as_of_utc cannot be in the future"
            )
        with self._guard():
            state, projection = self._read_locked()
            if state["version"] == 1:
                self._write_locked(state)
                state, projection = self._read_locked()
            plan, selected_ids = self._retention_plan_locked(
                state,
                projection,
                policy=policy,
                as_of=as_of,
                as_of_utc=effective_as_of,
            )
            if not selected_ids:
                return {
                    "compacted": False,
                    "cases_compacted": 0,
                    "events_compacted": 0,
                    "eligible_case_count": int(plan["eligible_case_count"]),
                }
            receipt = self._authorize_retention(
                authorization_receipt,
                now=now,
                body_sha256=str(plan["body_sha256"]),
            )
            selected_set = frozenset(selected_ids)
            selected_digests = tuple(
                sorted(
                    sha256(case_id.encode("utf-8")).hexdigest()
                    for case_id in selected_ids
                )
            )
            current_bits = bytearray(self._read_fence_locked(state["erasure_fence"]))
            accumulator = str(state["erasure_fence"]["accumulator_sha256"])
            for case_digest in selected_digests:
                if _bloom_contains(current_bits, case_digest):
                    raise SafeguardingIntegrityError(
                        "active safeguarding case already intersects the erasure fence"
                    )
                _bloom_add(current_bits, case_digest)
                accumulator = _advance_erasure_accumulator(accumulator, case_digest)
            remaining_projection = {
                case_id: case
                for case_id, case in projection.items()
                if case_id not in selected_set
            }
            self._assert_active_cases_outside_fence(
                remaining_projection, bytes(current_bits)
            )
            rebuilt, removed_count = self._rebuild_events(
                state["events"], removed_case_ids=selected_set
            )
            if removed_count != int(plan["selected_event_count"]):
                raise SafeguardingIntegrityError(
                    "retention event selection changed before compaction"
                )
            generation = self._next_fence_generation_locked()
            self._write_fence_locked(bytes(current_bits), generation=generation)
            state["events"] = rebuilt
            state["event_sequence"] = len(rebuilt)
            state["erasure_fence"] = {
                **deepcopy(dict(state["erasure_fence"])),
                "generation": generation,
                "inserted_count": int(state["erasure_fence"]["inserted_count"])
                + len(selected_ids),
                "bits_sha256": sha256(current_bits).hexdigest(),
                "accumulator_sha256": accumulator,
            }
            previous = deepcopy(dict(state["retention_compaction"]))
            previous_hash = (
                previous["compaction_sha256"]
                or sha256(b"safeguarding-retention-compaction-genesis-v1").hexdigest()
            )
            compaction = {
                "schema": SAFEGUARDING_RETENTION_COMPACTION_SCHEMA,
                "generation": int(previous["generation"]) + 1,
                "cases_compacted_total": int(previous["cases_compacted_total"])
                + len(selected_ids),
                "events_compacted_total": int(previous["events_compacted_total"])
                + removed_count,
                "last_compacted_at_utc": now_utc,
                "last_authorization_receipt_sha256": receipt["receipt_sha256"],
                "last_operation_body_sha256": plan["body_sha256"],
                "policy_version_sha256": policy.policy_version_sha256,
                "minimum_closed_age_seconds": policy.minimum_closed_age_seconds,
                "previous_compaction_sha256": previous_hash,
                "compaction_sha256": None,
            }
            compaction["compaction_sha256"] = _retention_compaction_hash(compaction)
            state["retention_compaction"] = compaction
            self._write_locked(state)
            return {
                "compacted": True,
                "cases_compacted": len(selected_ids),
                "events_compacted": removed_count,
                "eligible_case_count": int(plan["eligible_case_count"]),
            }

    def capacity_status(
        self,
        policy: SafeguardingRetentionPolicy | None = None,
        *,
        as_of_utc: str | None = None,
    ) -> dict[str, Any]:
        """Project aggregate capacity and erasure guarantees without case data."""

        if policy is not None and not isinstance(policy, SafeguardingRetentionPolicy):
            raise SafeguardingConfigurationError(
                "safeguarding retention policy is invalid"
            )
        now, now_utc = self._now()
        if as_of_utc is None:
            as_of = now
            effective_as_of = now_utc
        else:
            as_of = _parse_utc(as_of_utc, field="capacity.as_of_utc")
            effective_as_of = _format_utc(as_of)
            if as_of > now:
                raise SafeguardingIntegrityError(
                    "capacity as_of_utc cannot be in the future"
                )
        with self._guard():
            state, projection = self._read_locked()
            store_bytes = len(_canonical_bytes(state)) + 1
            if state["version"] == 2:
                fence = state["erasure_fence"]
                inserted_count = int(fence["inserted_count"])
                recent_limit = self._capacity.recent_tombstones
                retention = state["retention_compaction"]
            else:
                inserted_count = len(state["erasure_tombstones"])
                recent_limit = _MAX_TOMBSTONES
                retention = _empty_retention_compaction()
            eligible_cases = eligible_events = 0
            if policy is not None:
                if state["version"] == 2:
                    plan, _ = self._retention_plan_locked(
                        state,
                        projection,
                        policy=policy,
                        as_of=as_of,
                        as_of_utc=effective_as_of,
                    )
                    eligible_cases = int(plan["eligible_case_count"])
                    eligible_events = int(plan["selected_event_count"])
                else:
                    eligible_ids = [
                        case_id
                        for case_id, case in projection.items()
                        if case["status"] == "closed"
                        and case["escalation"]["delivery_status"] == "acknowledged"
                        and as_of
                        >= _parse_utc(
                            case["updated_at_utc"],
                            field="retention.closed_at_utc",
                        )
                        + timedelta(seconds=policy.minimum_closed_age_seconds)
                    ]
                    eligible_cases = len(eligible_ids)
                    selected_ids = frozenset(
                        sorted(eligible_ids)[: policy.maximum_cases_per_run]
                    )
                    eligible_events = sum(
                        event["case_snapshot"]["case_id"] in selected_ids
                        for event in state["events"]
                    )
            event_headroom = self._capacity.maximum_events - len(state["events"])
            byte_headroom = self._capacity.maximum_store_bytes - store_bytes
            tombstone_headroom = recent_limit - len(state["erasure_tombstones"])
            near_capacity = (
                event_headroom <= self._capacity.near_event_headroom
                or byte_headroom <= self._capacity.near_byte_headroom
            )
            blocked = near_capacity and (policy is None or eligible_cases == 0)
            return {
                "schema": SAFEGUARDING_CAPACITY_STATUS_SCHEMA,
                "store_version": int(state["version"]),
                "events": len(state["events"]),
                "event_capacity": self._capacity.maximum_events,
                "event_headroom": event_headroom,
                "store_bytes": store_bytes,
                "store_byte_capacity": self._capacity.maximum_store_bytes,
                "store_byte_headroom": byte_headroom,
                "recent_erasure_tombstones": len(state["erasure_tombstones"]),
                "recent_erasure_tombstone_capacity": recent_limit,
                "recent_erasure_tombstone_headroom": tombstone_headroom,
                "near_capacity": near_capacity,
                "retention_compaction_blocked": blocked,
                "retention_eligible_cases": eligible_cases,
                "retention_eligible_events": eligible_events,
                "cases_compacted_total": int(retention["cases_compacted_total"]),
                "events_compacted_total": int(retention["events_compacted_total"]),
                "erasure_fence_bit_count": _ERASURE_FENCE_BITS,
                "erasure_fence_hash_count": _ERASURE_FENCE_HASHES,
                "erasure_fence_inserted_count": inserted_count,
                "erasure_fence_estimated_false_positive_upper_bound": (
                    _erasure_false_positive_upper_bound(inserted_count)
                ),
                "erasure_fence_false_positive_target_upper_bound": (
                    _ERASURE_FALSE_POSITIVE_TARGET
                ),
                "erasure_fence_false_positive_within_target": (
                    _erasure_false_positive_upper_bound(inserted_count)
                    <= _ERASURE_FALSE_POSITIVE_TARGET
                ),
                "erasure_fence_false_negative_possible": False,
                "erasure_fence_false_positive_policy": "fail_closed_as_erased",
            }

    def purge_case(
        self,
        *,
        case_id: str,
        expected_version: int,
        idempotency_key: str,
        authorization_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Physically compact a closed case and retain only a hash-only fence."""

        if not isinstance(case_id, str) or _CASE_ID.fullmatch(case_id) is None:
            raise SafeguardingIntegrityError("case_id is invalid")
        if (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version < 1
        ):
            raise SafeguardingIntegrityError("expected_version is invalid")
        key_digest = self._idempotency_digest(idempotency_key)
        now, now_utc = self._now()
        authorization_body_sha256 = safeguarding_case_authorization_body_sha256(
            operation="case.purged",
            case_id=case_id,
            expected_version=expected_version,
        )
        actor = self._authorize(
            authorization_receipt,
            now=now,
            now_utc=now_utc,
            operation="case.purged",
            body_sha256=authorization_body_sha256,
            allowed_actor_kinds=frozenset({_STAFF_ACTOR_KIND}),
        )
        request_hash = canonical_sha256(
            {
                "operation": "case.purged",
                "case_id_sha256": sha256(case_id.encode("utf-8")).hexdigest(),
                "expected_version": expected_version,
                "actor_principal_sha256": actor["principal_sha256"],
                "actor_role_sha256": actor["role_sha256"],
            }
        )
        tombstone_key = sha256(case_id.encode("utf-8")).hexdigest()
        with self._guard():
            state, projection = self._read_locked()
            state, fence_payload = self._ensure_v2_locked(state)
            projection = _validate_state(state, capacity=self._capacity)[1]
            existing_tombstone = state["erasure_tombstones"].get(tombstone_key)
            if existing_tombstone is not None:
                if (
                    existing_tombstone["idempotency_key_sha256"] == key_digest
                    and existing_tombstone["request_sha256"] == request_hash
                ):
                    return {
                        "purged": False,
                        "case_id_sha256": tombstone_key,
                        "purged_at_utc": existing_tombstone["purged_at_utc"],
                    }
                raise SafeguardingErasedError(
                    "safeguarding case was permanently erased"
                )
            case = projection.get(case_id)
            if case is None:
                if self._case_is_erased_locked(
                    state, tombstone_key, fence_bits=fence_payload
                ):
                    raise SafeguardingErasedError(
                        "safeguarding case was permanently erased"
                    )
                raise SafeguardingNotFoundError("safeguarding case was not found")
            if case["version"] != expected_version:
                raise SafeguardingConflictError(
                    "safeguarding case expected_version conflict"
                )
            if case["status"] != "closed":
                raise SafeguardingConflictError(
                    "only a closed safeguarding case can be purged"
                )
            bits = bytearray(fence_payload)
            if _bloom_contains(bits, tombstone_key):
                raise SafeguardingIntegrityError(
                    "active safeguarding case already intersects the erasure fence"
                )
            _bloom_add(bits, tombstone_key)
            remaining_projection = {
                active_id: active_case
                for active_id, active_case in projection.items()
                if active_id != case_id
            }
            self._assert_active_cases_outside_fence(remaining_projection, bytes(bits))
            rebuilt, removed_count = self._rebuild_events(
                state["events"], removed_case_ids=frozenset({case_id})
            )
            tombstone = {
                "schema": SAFEGUARDING_ERASURE_TOMBSTONE_SCHEMA,
                "case_id_sha256": tombstone_key,
                "purged_at_utc": now_utc,
                "generation": 1,
                "expected_version": expected_version,
                "idempotency_key_sha256": key_digest,
                "request_sha256": request_hash,
                "authorization_receipt_sha256": actor["authorization_receipt_sha256"],
                "actor": actor,
            }
            tombstone["tombstone_sha256"] = _tombstone_hash(tombstone)
            state["events"] = rebuilt
            state["event_sequence"] = len(rebuilt)
            state["erasure_tombstones"][tombstone_key] = tombstone
            recent = sorted(
                state["erasure_tombstones"].items(),
                key=lambda item: (str(item[1]["purged_at_utc"]), item[0]),
            )[-self._capacity.recent_tombstones :]
            state["erasure_tombstones"] = dict(recent)
            generation = self._next_fence_generation_locked()
            self._write_fence_locked(bytes(bits), generation=generation)
            state["erasure_fence"] = {
                **deepcopy(dict(state["erasure_fence"])),
                "generation": generation,
                "inserted_count": int(state["erasure_fence"]["inserted_count"]) + 1,
                "bits_sha256": sha256(bits).hexdigest(),
                "accumulator_sha256": _advance_erasure_accumulator(
                    str(state["erasure_fence"]["accumulator_sha256"]),
                    tombstone_key,
                ),
            }
            self._write_locked(state)
            return {
                "purged": True,
                "events_removed": removed_count,
                "case_id_sha256": tombstone_key,
                "purged_at_utc": now_utc,
            }

    def recover(self) -> dict[str, Any]:
        """Strictly replay the journal and return a defensive content-free snapshot."""

        with self._guard():
            state, projection = self._read_locked()
        return {
            "event_sequence": state["event_sequence"],
            "event_head_sha256": (
                state["events"][-1]["event_sha256"] if state["events"] else None
            ),
            "cases": {key: deepcopy(value) for key, value in projection.items()},
            "erasure_tombstones": deepcopy(dict(state["erasure_tombstones"])),
        }


__all__ = [
    "EMERGENCY_RESOURCE_BUNDLE_SCHEMA",
    "SAFEGUARDING_AUTHORIZATION_RECEIPT_SCHEMA",
    "SAFEGUARDING_CAPACITY_STATUS_SCHEMA",
    "SAFEGUARDING_CASE_SCHEMA",
    "SAFEGUARDING_CASE_STATUSES",
    "SAFEGUARDING_CATEGORIES",
    "SAFEGUARDING_ERASURE_TOMBSTONE_SCHEMA",
    "SAFEGUARDING_ERASURE_FENCE_SCHEMA",
    "SAFEGUARDING_EVENT_SCHEMA",
    "SAFEGUARDING_ROLE_SHA256",
    "SAFEGUARDING_RETENTION_AUTHORIZATION_RECEIPT_SCHEMA",
    "SAFEGUARDING_RETENTION_COMPACTION_SCHEMA",
    "SAFEGUARDING_RETENTION_PLAN_SCHEMA",
    "SAFEGUARDING_RETENTION_ROLE_SHA256",
    "SAFEGUARDING_SEVERITIES",
    "SAFEGUARDING_STORE_SCHEMA",
    "SAFEGUARDING_STORE_SCHEMA_V1",
    "SYSTEM_SAFETY_CLASSIFIER_ROLE_SHA256",
    "EmergencyResourcePolicy",
    "EscalationDeliveryConfig",
    "SafeguardingAuthorizationError",
    "SafeguardingCapacityLimits",
    "SafeguardingConfigurationError",
    "SafeguardingConflictError",
    "SafeguardingErasedError",
    "SafeguardingIntegrityError",
    "SafeguardingNotFoundError",
    "SafeguardingRetentionPolicy",
    "TeacherAgentSafeguardingError",
    "TeacherAgentSafeguardingStore",
    "authorization_receipt_sha256",
    "canonical_sha256",
    "default_emergency_resource_policy",
    "retention_authorization_receipt_sha256",
    "safeguarding_erasure_false_positive_upper_bound",
    "safeguarding_case_authorization_body_sha256",
    "safeguarding_open_authorization_body_sha256",
]
