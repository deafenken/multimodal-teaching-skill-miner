from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta
from hashlib import sha256
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import stat
import threading

from jsonschema import Draft202012Validator
import pytest

from teaching_skill_miner.teacher_agent_safeguarding import (
    EMERGENCY_RESOURCE_BUNDLE_SCHEMA,
    SAFEGUARDING_AUTHORIZATION_RECEIPT_SCHEMA,
    SAFEGUARDING_RETENTION_AUTHORIZATION_RECEIPT_SCHEMA,
    SAFEGUARDING_ROLE_SHA256,
    SAFEGUARDING_STORE_SCHEMA,
    SAFEGUARDING_STORE_SCHEMA_V1,
    SYSTEM_SAFETY_CLASSIFIER_ROLE_SHA256,
    EmergencyResourcePolicy,
    EscalationDeliveryConfig,
    SafeguardingAuthorizationError,
    SafeguardingCapacityLimits,
    SafeguardingConfigurationError,
    SafeguardingConflictError,
    SafeguardingErasedError,
    SafeguardingIntegrityError,
    SafeguardingRetentionPolicy,
    TeacherAgentSafeguardingStore,
    authorization_receipt_sha256,
    canonical_sha256,
    default_emergency_resource_policy,
    safeguarding_erasure_false_positive_upper_bound,
    safeguarding_case_authorization_body_sha256,
    safeguarding_open_authorization_body_sha256,
)
from teaching_skill_miner.teacher_agent_safeguarding_retention_authority import (
    InternalSafeguardingRetentionAuthority,
    SafeguardingRetentionAuthorityError,
)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


class FrozenClock:
    def __init__(self, value: str = "2026-08-12T12:00:00Z") -> None:
        self._lock = threading.Lock()
        self._value = datetime.fromisoformat(value.replace("Z", "+00:00"))

    def __call__(self) -> datetime:
        with self._lock:
            return self._value

    def advance(self, *, seconds: int) -> None:
        with self._lock:
            self._value += timedelta(seconds=seconds)


class ExternalServerAuthority:
    """Test double for the server identity layer, not a core identity issuer."""

    def __init__(self, clock: FrozenClock) -> None:
        self.clock = clock
        self._trusted: set[str] = set()

    def issue(
        self,
        *,
        principal: str = "institution-safeguarding-officer-7",
        actor_kind: str = "server_authenticated_safeguarding",
        role_sha256: str | None = None,
        operation: str,
        body_sha256: str,
        lifetime_seconds: int = 600,
    ) -> dict:
        now = self.clock()
        effective_role = role_sha256 or (
            SYSTEM_SAFETY_CLASSIFIER_ROLE_SHA256
            if actor_kind == "system_safety_classifier"
            else SAFEGUARDING_ROLE_SHA256
        )
        value = {
            "schema": SAFEGUARDING_AUTHORIZATION_RECEIPT_SCHEMA,
            "actor_kind": actor_kind,
            "server_authenticated": True,
            "principal_sha256": _digest(principal),
            "role_sha256": effective_role,
            "authorization_context_sha256": _digest("oidc-session-context-91"),
            "operation": operation,
            "body_sha256": body_sha256,
            "issued_at_utc": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "expires_at_utc": (now + timedelta(seconds=lifetime_seconds))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        }
        value["receipt_sha256"] = authorization_receipt_sha256(value)
        self._trusted.add(value["receipt_sha256"])
        return value

    def verify(self, value: dict) -> dict:
        if value.get("receipt_sha256") not in self._trusted:
            raise PermissionError("receipt was not issued by the server")
        return deepcopy(value)


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def authority(clock: FrozenClock) -> ExternalServerAuthority:
    return ExternalServerAuthority(clock)


@pytest.fixture
def scope_zh() -> str:
    return _digest("tenant/school-zh/learner-scope-opaque")


@pytest.fixture
def policy(scope_zh: str) -> EmergencyResourcePolicy:
    return default_emergency_resource_policy({scope_zh: "zh-CN"})


def _store(
    root: Path,
    *,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    delivery: EscalationDeliveryConfig | None = None,
    retention_authority: InternalSafeguardingRetentionAuthority | None = None,
    capacity: SafeguardingCapacityLimits | None = None,
) -> TeacherAgentSafeguardingStore:
    return TeacherAgentSafeguardingStore(
        root,
        authorization_verifier=authority.verify,
        emergency_resource_policy=policy,
        escalation_delivery=delivery,
        retention_authorization_verifier=(
            retention_authority.verify if retention_authority is not None else None
        ),
        capacity_limits=capacity,
        clock=clock,
    )


def _open(
    store: TeacherAgentSafeguardingStore,
    authority: ExternalServerAuthority,
    *,
    scope: str,
    key: str = "open-safeguarding-case-001",
    content: str = "learner-safety-content-001",
    category: str = "self_harm",
    severity: str = "urgent",
    authorization_receipt: dict | None = None,
) -> dict:
    observed_at_utc = "2026-08-12T11:59:00Z"
    content_sha256 = _digest(content)
    body_sha256 = safeguarding_open_authorization_body_sha256(
        scope_sha256=scope,
        category=category,
        severity=severity,
        observed_at_utc=observed_at_utc,
        content_sha256=content_sha256,
    )
    receipt = authorization_receipt or authority.issue(
        principal="internal-safety-gate",
        actor_kind="system_safety_classifier",
        operation="case.opened",
        body_sha256=body_sha256,
    )
    return store.open_case(
        scope_sha256=scope,
        category=category,
        severity=severity,
        observed_at_utc=observed_at_utc,
        content_sha256=content_sha256,
        idempotency_key=key,
        authorization_receipt=receipt,
    )


def _staff_receipt(
    authority: ExternalServerAuthority,
    *,
    operation: str,
    case_id: str,
    expected_version: int,
    **kwargs: object,
) -> dict:
    return authority.issue(
        operation=operation,
        body_sha256=safeguarding_case_authorization_body_sha256(
            operation=operation,
            case_id=case_id,
            expected_version=expected_version,
        ),
        **kwargs,
    )


def _process_open_worker(
    root: str,
    scope_sha256: str,
    index: int,
    start: object,
    results: object,
) -> None:
    process_clock = FrozenClock()
    authority = ExternalServerAuthority(process_clock)
    policy = default_emergency_resource_policy({scope_sha256: "en-US"})
    store = _store(Path(root), clock=process_clock, authority=authority, policy=policy)
    start.wait(10)  # type: ignore[attr-defined]
    try:
        case = _open(
            store,
            authority,
            scope=scope_sha256,
            key=f"process-open-{index:03d}",
            content=f"process-content-{index:03d}",
        )
    except Exception as exc:  # pragma: no cover - asserted by the parent process.
        results.put(("error", repr(exc)))  # type: ignore[attr-defined]
    else:
        results.put(("ok", case["case_id"]))  # type: ignore[attr-defined]


def test_minimized_case_restart_idempotency_and_hash_only_actor(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    first = _open(store, authority, scope=scope_zh)
    assert first["status"] == "open"
    assert first["version"] == 1
    assert first["scope_sha256"] == scope_zh
    assert first["opened_by"]["actor_kind"] == "system_safety_classifier"
    assert first["opened_by"]["operation"] == "case.opened"
    assert first["escalation"] == {
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
    assert set(first["opened_by"]) == {
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

    restarted = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    assert restarted.get_case(first["case_id"]) == first
    assert _open(restarted, authority, scope=scope_zh) == first
    clock.advance(seconds=601)
    assert _open(restarted, authority, scope=scope_zh) == first
    assert restarted.recover()["event_sequence"] == 1

    raw = restarted.path.read_text(encoding="utf-8")
    assert "institution-safeguarding-officer-7" not in raw
    assert "oidc-session-context-91" not in raw
    assert "open-safeguarding-case-001" not in raw
    assert "learner-safety-content-001" not in raw


def test_occurrence_identity_replays_exactly_but_same_content_reopens_after_close(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    content = "same-disclosure-may-recur"
    first = _open(
        store,
        authority,
        scope=scope_zh,
        key="safety-occurrence-first-001",
        content=content,
    )
    exact = store.case_for_open_idempotency_key(
        "safety-occurrence-first-001",
        scope_sha256=scope_zh,
        category="self_harm",
        severity="urgent",
        content_sha256=_digest(content),
    )
    assert exact is not None and exact["case_id"] == first["case_id"]
    with pytest.raises(SafeguardingConflictError, match="different safeguarding"):
        store.case_for_open_idempotency_key(
            "safety-occurrence-first-001",
            scope_sha256=scope_zh,
            category="self_harm",
            severity="urgent",
            content_sha256=_digest("different-disclosure"),
        )

    acknowledged = store.acknowledge_case(
        case_id=first["case_id"],
        expected_version=1,
        idempotency_key="safety-occurrence-ack-001",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.acknowledged",
            case_id=first["case_id"],
            expected_version=1,
        ),
    )
    closed = store.close_case(
        case_id=first["case_id"],
        expected_version=acknowledged["version"],
        idempotency_key="safety-occurrence-close-001",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.closed",
            case_id=first["case_id"],
            expected_version=acknowledged["version"],
        ),
    )
    assert closed["status"] == "closed"

    second = _open(
        store,
        authority,
        scope=scope_zh,
        key="safety-occurrence-second-001",
        content=content,
    )
    assert second["status"] == "open"
    assert second["case_id"] != first["case_id"]
    assert {case["case_id"]: case["status"] for case in store.list_cases()} == {
        first["case_id"]: "closed",
        second["case_id"]: "open",
    }


def test_case_status_cas_and_idempotency_survive_restart(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    opened = _open(store, authority, scope=scope_zh)
    acknowledged = store.acknowledge_case(
        case_id=opened["case_id"],
        expected_version=1,
        idempotency_key="acknowledge-case-001",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.acknowledged",
            case_id=opened["case_id"],
            expected_version=1,
        ),
    )
    assert (acknowledged["status"], acknowledged["version"]) == (
        "acknowledged",
        2,
    )
    with pytest.raises(SafeguardingConflictError, match="expected_version"):
        store.close_case(
            case_id=opened["case_id"],
            expected_version=1,
            idempotency_key="close-with-stale-version",
            authorization_receipt=_staff_receipt(
                authority,
                operation="case.closed",
                case_id=opened["case_id"],
                expected_version=1,
            ),
        )

    restarted = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    replay = restarted.acknowledge_case(
        case_id=opened["case_id"],
        expected_version=1,
        idempotency_key="acknowledge-case-001",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.acknowledged",
            case_id=opened["case_id"],
            expected_version=1,
        ),
    )
    assert replay == acknowledged
    with pytest.raises(SafeguardingConflictError, match="different content"):
        restarted.acknowledge_case(
            case_id=opened["case_id"],
            expected_version=2,
            idempotency_key="acknowledge-case-001",
            authorization_receipt=_staff_receipt(
                authority,
                operation="case.acknowledged",
                case_id=opened["case_id"],
                expected_version=2,
            ),
        )

    closed = restarted.close_case(
        case_id=opened["case_id"],
        expected_version=2,
        idempotency_key="close-case-001",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.closed",
            case_id=opened["case_id"],
            expected_version=2,
        ),
    )
    assert (closed["status"], closed["version"]) == ("closed", 3)


def test_mutations_require_external_server_verified_safeguarding_role(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    body_sha256 = safeguarding_open_authorization_body_sha256(
        scope_sha256=scope_zh,
        category="self_harm",
        severity="urgent",
        observed_at_utc="2026-08-12T11:59:00Z",
        content_sha256=_digest("learner-safety-content-001"),
    )
    forged = authority.issue(
        actor_kind="system_safety_classifier",
        operation="case.opened",
        body_sha256=body_sha256,
    )
    authority._trusted.remove(forged["receipt_sha256"])
    with pytest.raises(SafeguardingAuthorizationError, match="server rejected"):
        _open(
            store,
            authority,
            scope=scope_zh,
            key="forged-authorization-001",
            authorization_receipt=forged,
        )

    wrong_role = authority.issue(
        actor_kind="system_safety_classifier",
        role_sha256=_digest("teacher"),
        operation="case.opened",
        body_sha256=body_sha256,
    )
    with pytest.raises(SafeguardingAuthorizationError, match="required"):
        _open(
            store,
            authority,
            scope=scope_zh,
            key="wrong-role-receipt-001",
            authorization_receipt=wrong_role,
        )

    unauthenticated = authority.issue(
        actor_kind="system_safety_classifier",
        operation="case.opened",
        body_sha256=body_sha256,
    )
    unauthenticated["server_authenticated"] = False
    unauthenticated["receipt_sha256"] = authorization_receipt_sha256(unauthenticated)
    authority._trusted.add(unauthenticated["receipt_sha256"])
    with pytest.raises(
        SafeguardingAuthorizationError, match="not server-authenticated"
    ):
        _open(
            store,
            authority,
            scope=scope_zh,
            key="unauthenticated-001",
            authorization_receipt=unauthenticated,
        )

    expired = authority.issue(
        actor_kind="system_safety_classifier",
        operation="case.opened",
        body_sha256=body_sha256,
        lifetime_seconds=30,
    )
    clock.advance(seconds=31)
    with pytest.raises(SafeguardingAuthorizationError, match="expired"):
        _open(
            store,
            authority,
            scope=scope_zh,
            key="expired-receipt-001",
            authorization_receipt=expired,
        )
    assert store.list_cases() == []


def test_system_classifier_is_body_bound_and_cannot_manage_a_case(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    wrong_body = authority.issue(
        principal="internal-safety-gate",
        actor_kind="system_safety_classifier",
        operation="case.opened",
        body_sha256=safeguarding_open_authorization_body_sha256(
            scope_sha256=scope_zh,
            category="abuse_disclosure",
            severity="high",
            observed_at_utc="2026-08-12T11:59:00Z",
            content_sha256=_digest("different-classifier-observation"),
        ),
    )
    with pytest.raises(SafeguardingAuthorizationError, match="body does not match"):
        _open(
            store,
            authority,
            scope=scope_zh,
            key="classifier-body-reuse-001",
            authorization_receipt=wrong_body,
        )

    case = _open(store, authority, scope=scope_zh)
    management_body = safeguarding_case_authorization_body_sha256(
        operation="case.acknowledged",
        case_id=case["case_id"],
        expected_version=1,
    )
    system_management_receipt = authority.issue(
        principal="internal-safety-gate",
        actor_kind="system_safety_classifier",
        operation="case.acknowledged",
        body_sha256=management_body,
    )
    with pytest.raises(SafeguardingAuthorizationError, match="actor kind"):
        store.acknowledge_case(
            case_id=case["case_id"],
            expected_version=1,
            idempotency_key="system-cannot-ack-case",
            authorization_receipt=system_management_receipt,
        )

    for operation, mutate in (
        (
            "case.closed",
            lambda system_receipt: store.close_case(
                case_id=case["case_id"],
                expected_version=1,
                idempotency_key="system-cannot-close-case",
                authorization_receipt=system_receipt,
            ),
        ),
        (
            "case.purged",
            lambda system_receipt: store.purge_case(
                case_id=case["case_id"],
                expected_version=1,
                idempotency_key="system-cannot-purge-case",
                authorization_receipt=system_receipt,
            ),
        ),
    ):
        system_receipt = authority.issue(
            principal="internal-safety-gate",
            actor_kind="system_safety_classifier",
            operation=operation,
            body_sha256=safeguarding_case_authorization_body_sha256(
                operation=operation,
                case_id=case["case_id"],
                expected_version=1,
            ),
        )
        with pytest.raises(SafeguardingAuthorizationError, match="actor kind"):
            mutate(system_receipt)

    staff_open_receipt = authority.issue(
        actor_kind="server_authenticated_safeguarding",
        operation="case.opened",
        body_sha256=safeguarding_open_authorization_body_sha256(
            scope_sha256=scope_zh,
            category="self_harm",
            severity="urgent",
            observed_at_utc="2026-08-12T11:59:00Z",
            content_sha256=_digest("staff-must-not-open-content"),
        ),
    )
    with pytest.raises(SafeguardingAuthorizationError, match="actor kind"):
        _open(
            store,
            authority,
            scope=scope_zh,
            key="staff-cannot-open-case",
            content="staff-must-not-open-content",
            authorization_receipt=staff_open_receipt,
        )

    close_receipt = _staff_receipt(
        authority,
        operation="case.closed",
        case_id=case["case_id"],
        expected_version=1,
    )
    with pytest.raises(
        SafeguardingAuthorizationError, match="operation does not match"
    ):
        store.acknowledge_case(
            case_id=case["case_id"],
            expected_version=1,
            idempotency_key="staff-cross-operation-reuse",
            authorization_receipt=close_receipt,
        )
    assert store.get_case(case["case_id"])["version"] == 1


def test_missing_delivery_is_explicit_and_cannot_be_acknowledged(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    case = _open(store, authority, scope=scope_zh)
    assert case["escalation"]["delivery_status"] == "escalation_unavailable"
    assert store.pending_escalations() == []
    assert store.list_escalations_due(as_of_utc="2030-01-01T00:00:00Z") == []
    with pytest.raises(SafeguardingConfigurationError, match="escalation_unavailable"):
        store.acknowledge_escalation(
            case_id=case["case_id"],
            expected_version=1,
            idempotency_key="ack-unavailable-delivery",
            authorization_receipt=_staff_receipt(
                authority,
                operation="escalation.acknowledged",
                case_id=case["case_id"],
                expected_version=1,
            ),
        )


def test_configured_delivery_enqueue_overdue_and_ack_are_durable(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    delivery = EscalationDeliveryConfig(
        policy_version="institutional_escalation_v3",
        queue_sha256=_digest("staffed-safeguarding-queue"),
        sla_seconds_by_severity={"elevated": 600, "high": 300, "urgent": 60},
    )
    store = _store(
        tmp_path,
        clock=clock,
        authority=authority,
        policy=policy,
        delivery=delivery,
    )
    case = _open(store, authority, scope=scope_zh)
    escalation = case["escalation"]
    assert escalation["delivery_status"] == "pending"
    assert escalation["policy_version"] == "institutional_escalation_v3"
    assert escalation["sla_due_at_utc"] == "2026-08-12T12:01:00Z"
    assert store.pending_escalations()[0]["content_sha256"] == case["content_sha256"]
    assert store.list_escalations_due() == []

    clock.advance(seconds=61)
    due = store.list_escalations_due()
    assert due == [
        {
            "case_id": case["case_id"],
            "version": 1,
            "delivery_id": escalation["delivery_id"],
            "sla_due_at_utc": "2026-08-12T12:01:00Z",
            "overdue": True,
        }
    ]
    overdue = store.record_escalation_overdue(
        case_id=case["case_id"],
        expected_version=1,
        idempotency_key="record-overdue-001",
        authorization_receipt=_staff_receipt(
            authority,
            operation="escalation.overdue",
            case_id=case["case_id"],
            expected_version=1,
        ),
    )
    assert overdue["escalation"]["delivery_status"] == "overdue"

    restarted = _store(
        tmp_path,
        clock=clock,
        authority=authority,
        policy=policy,
        delivery=delivery,
    )
    assert restarted.get_case(case["case_id"]) == overdue
    acknowledged = restarted.acknowledge_escalation(
        case_id=case["case_id"],
        expected_version=2,
        idempotency_key="ack-delivery-001",
        authorization_receipt=_staff_receipt(
            authority,
            operation="escalation.acknowledged",
            case_id=case["case_id"],
            expected_version=2,
        ),
    )
    assert acknowledged["escalation"]["delivery_status"] == "acknowledged"
    assert acknowledged["escalation"]["overdue_recorded_at_utc"] is not None
    assert restarted.pending_escalations() == []


def test_localized_resource_policy_english_chinese_and_unknown_golden(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
) -> None:
    en_scope = _digest("deployment-scope-en")
    zh_scope = _digest("deployment-scope-zh")
    unknown_scope = _digest("deployment-scope-unknown")
    policy = default_emergency_resource_policy(
        {en_scope: "en-US", zh_scope: "zh-CN", unknown_scope: "fr-FR"}
    )
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    cases = {
        "en": _open(store, authority, scope=en_scope, key="localized-case-en-001"),
        "zh": _open(store, authority, scope=zh_scope, key="localized-case-zh-001"),
        "unknown": _open(
            store, authority, scope=unknown_scope, key="localized-case-unknown-001"
        ),
    }

    english = store.emergency_resources_for_case(cases["en"]["case_id"])
    chinese = store.emergency_resources_for_case(cases["zh"]["case_id"])
    fallback = store.emergency_resources_for_case(cases["unknown"]["case_id"])
    assert english == {
        "schema": EMERGENCY_RESOURCE_BUNDLE_SCHEMA,
        "policy_version": "institutional_emergency_resources_v1",
        "locale_selection_source": "trusted_deployment_scope_mapping",
        "configured_locale": "en-US",
        "effective_locale": "en-US",
        "localization_status": "localized",
        "localization_unavailable": False,
        "resources": [
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
        ],
    }
    assert chinese["resources"] == [
        {
            "kind": "emergency_services",
            "label": "紧急危险",
            "instruction": "如任何人正面临紧急危险，请立即联系所在地紧急服务。",
        },
        {
            "kind": "trusted_adult",
            "label": "可信支持",
            "instruction": "请前往安全地点，并立即告知可信任的成年人或安全保护专业人员。",
        },
    ]
    assert fallback == {
        "schema": EMERGENCY_RESOURCE_BUNDLE_SCHEMA,
        "policy_version": "institutional_emergency_resources_v1",
        "locale_selection_source": "trusted_deployment_scope_mapping",
        "configured_locale": "fr-FR",
        "effective_locale": "generic",
        "localization_status": "localization_unavailable",
        "localization_unavailable": True,
        "resources": [
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
        ],
    }
    assert chinese["policy_version"] == "institutional_emergency_resources_v1"
    assert chinese["localization_unavailable"] is False

    # No locale parameter exists at the mutation boundary; browser/free text
    # cannot select a jurisdiction by adding one.
    assert "locale" not in inspect.signature(store.open_case).parameters
    with pytest.raises(TypeError, match="locale"):
        store.open_case(  # type: ignore[call-arg]
            scope_sha256=en_scope,
            category="self_harm",
            severity="urgent",
            observed_at_utc="2026-08-12T11:59:00Z",
            content_sha256=_digest("free-text-cannot-select-locale"),
            idempotency_key="locale-injection-attempt",
            authorization_receipt={},
            locale="zh-CN",
        )


def test_journal_never_persists_raw_content_or_resource_copy(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    sentinel = "我今晚准备吞下整瓶药；这段原文绝不能进入日志"
    case = _open(
        store,
        authority,
        scope=scope_zh,
        content=sentinel,
        key="content-minimization-001",
    )
    raw = store.path.read_text(encoding="utf-8")
    assert sentinel not in raw
    assert "learner_text" not in raw
    assert "raw_content" not in raw
    assert "如任何人正面临紧急危险" not in raw
    assert "Contact local emergency services now" not in raw
    assert case["content_sha256"] == _digest(sentinel)
    assert set(inspect.signature(store.open_case).parameters).isdisjoint(
        {"content", "text", "learner_text", "raw_content"}
    )


def test_hash_chain_and_semantic_tampering_fail_closed_without_repair(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    case = _open(store, authority, scope=scope_zh)
    state = json.loads(store.path.read_text(encoding="utf-8"))
    state["events"][0]["case_snapshot"]["severity"] = "elevated"
    tampered = json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n"
    store.path.write_text(tampered, encoding="utf-8")
    before = store.path.read_bytes()
    with pytest.raises(SafeguardingIntegrityError, match="integrity|hash|canonical"):
        store.get_case(case["case_id"])
    assert store.path.read_bytes() == before


def test_resealed_semantic_identity_tampering_still_fails_closed(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    import teaching_skill_miner.teacher_agent_safeguarding as module

    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    _open(store, authority, scope=scope_zh)
    state = json.loads(store.path.read_text(encoding="utf-8"))
    event = state["events"][0]
    event["case_snapshot"]["case_id"] = "sgc_" + "0" * 24
    event["case_snapshot"]["case_sha256"] = module._case_hash(event["case_snapshot"])
    event["event_sha256"] = module._event_hash(event)
    store.path.write_bytes(module._canonical_bytes(state) + b"\n")
    before = store.path.read_bytes()
    with pytest.raises(SafeguardingIntegrityError, match="identity"):
        store.recover()
    assert store.path.read_bytes() == before


@pytest.mark.skipif(os.name != "posix", reason="private flock store is POSIX-only")
def test_store_enforces_0600_regular_non_symlink_files(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path / "mode", clock=clock, authority=authority, policy=policy)
    _open(store, authority, scope=scope_zh)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.lock_path.stat().st_mode) == 0o600
    store.path.chmod(0o644)
    with pytest.raises(SafeguardingIntegrityError, match="0600"):
        store.recover()

    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    target = symlink_root / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    target.chmod(0o600)
    (symlink_root / "safeguarding_cases.json").symlink_to(target)
    unsafe = _store(symlink_root, clock=clock, authority=authority, policy=policy)
    with pytest.raises(SafeguardingIntegrityError, match="opened safely"):
        unsafe.recover()


def test_concurrent_writers_are_serialized_and_cas_has_one_winner(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    stores = [
        _store(tmp_path, clock=clock, authority=authority, policy=policy)
        for _ in range(8)
    ]

    def open_number(index: int) -> dict:
        return _open(
            stores[index % len(stores)],
            authority,
            scope=scope_zh,
            key=f"concurrent-open-{index:03d}",
            content=f"concurrent-content-{index:03d}",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        cases = list(pool.map(open_number, range(24)))
    assert len({case["case_id"] for case in cases}) == 24
    assert stores[0].recover()["event_sequence"] == 24

    target = cases[0]

    def acknowledge(index: int) -> str:
        try:
            stores[index].acknowledge_case(
                case_id=target["case_id"],
                expected_version=1,
                idempotency_key=f"concurrent-cas-{index:03d}",
                authorization_receipt=_staff_receipt(
                    authority,
                    operation="case.acknowledged",
                    case_id=target["case_id"],
                    expected_version=1,
                ),
            )
        except SafeguardingConflictError:
            return "conflict"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(acknowledge, range(2)))
    assert sorted(outcomes) == ["committed", "conflict"]
    assert stores[0].get_case(target["case_id"])["version"] == 2
    assert stores[0].recover()["event_sequence"] == 25


@pytest.mark.skipif(os.name != "posix", reason="cross-process flock requires POSIX")
def test_cross_process_flock_keeps_every_committed_case(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_process_open_worker,
            args=(str(tmp_path), scope_zh, index, start, results),
        )
        for index in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    rows = [results.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert all(status == "ok" for status, _ in rows), rows
    assert len({case_id for _, case_id in rows}) == len(processes)

    restarted = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    recovery = restarted.recover()
    assert recovery["event_sequence"] == len(processes)
    assert len(recovery["cases"]) == len(processes)


def test_closed_case_purge_physically_compacts_and_restart_fences_identity(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    opened = _open(store, authority, scope=scope_zh)
    acknowledged = store.acknowledge_case(
        case_id=opened["case_id"],
        expected_version=1,
        idempotency_key="purge-ack-case-001",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.acknowledged",
            case_id=opened["case_id"],
            expected_version=1,
        ),
    )
    closed = store.close_case(
        case_id=opened["case_id"],
        expected_version=acknowledged["version"],
        idempotency_key="purge-close-case-001",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.closed",
            case_id=opened["case_id"],
            expected_version=acknowledged["version"],
        ),
    )
    result = store.purge_case(
        case_id=opened["case_id"],
        expected_version=closed["version"],
        idempotency_key="purge-case-final-001",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.purged",
            case_id=opened["case_id"],
            expected_version=closed["version"],
        ),
    )
    assert result["purged"] is True
    assert result["events_removed"] == 3
    raw = store.path.read_text(encoding="utf-8")
    assert opened["case_id"] not in raw
    assert opened["scope_sha256"] not in raw
    assert opened["content_sha256"] not in raw
    assert json.loads(raw)["events"] == []

    restarted = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    with pytest.raises(SafeguardingErasedError, match="erased"):
        restarted.get_case(opened["case_id"])
    replay = restarted.purge_case(
        case_id=opened["case_id"],
        expected_version=closed["version"],
        idempotency_key="purge-case-final-001",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.purged",
            case_id=opened["case_id"],
            expected_version=closed["version"],
        ),
    )
    assert replay["purged"] is False
    assert len(restarted.recover()["erasure_tombstones"]) == 1


def _retention_authority(
    clock: FrozenClock,
) -> InternalSafeguardingRetentionAuthority:
    return InternalSafeguardingRetentionAuthority(
        key=b"dedicated-retention-authority-key-32-bytes-minimum",
        deployment_context_sha256=_digest("production-retention-deployment"),
        clock=clock,
    )


def _delivery() -> EscalationDeliveryConfig:
    return EscalationDeliveryConfig(
        policy_version="institution-safeguarding-routing-v1",
        queue_sha256=_digest("safeguarding-retention-queue"),
        sla_seconds_by_severity={
            "elevated": 86_400,
            "high": 14_400,
            "urgent": 900,
        },
    )


def _fully_deliver_and_close(
    store: TeacherAgentSafeguardingStore,
    authority: ExternalServerAuthority,
    case: dict,
    *,
    prefix: str,
) -> dict:
    delivered = store.acknowledge_escalation(
        case_id=case["case_id"],
        expected_version=case["version"],
        idempotency_key=f"{prefix}-delivery-acknowledged",
        authorization_receipt=_staff_receipt(
            authority,
            operation="escalation.acknowledged",
            case_id=case["case_id"],
            expected_version=case["version"],
        ),
    )
    acknowledged = store.acknowledge_case(
        case_id=case["case_id"],
        expected_version=delivered["version"],
        idempotency_key=f"{prefix}-case-acknowledged",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.acknowledged",
            case_id=case["case_id"],
            expected_version=delivered["version"],
        ),
    )
    return store.close_case(
        case_id=case["case_id"],
        expected_version=acknowledged["version"],
        idempotency_key=f"{prefix}-case-closed",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.closed",
            case_id=case["case_id"],
            expected_version=acknowledged["version"],
        ),
    )


def test_retention_authority_is_independent_restart_safe_and_exact(
    clock: FrozenClock,
) -> None:
    first = _retention_authority(clock)
    receipt = first.issue(
        operation="case.retention_compacted",
        body_sha256=_digest("exact-retention-plan-body"),
    )
    restarted = _retention_authority(clock)
    assert restarted.verify(receipt) == receipt
    assert receipt["schema"] == SAFEGUARDING_RETENTION_AUTHORIZATION_RECEIPT_SCHEMA
    assert receipt["actor_kind"] == "server_safeguarding_retention"
    assert receipt["role_sha256"] not in {
        SAFEGUARDING_ROLE_SHA256,
        SYSTEM_SAFETY_CLASSIFIER_ROLE_SHA256,
    }
    assert "case_id" not in json.dumps(receipt)

    tampered = deepcopy(receipt)
    tampered["body_sha256"] = _digest("tampered-retention-plan-body")
    with pytest.raises(SafeguardingRetentionAuthorityError, match="verification"):
        restarted.verify(tampered)
    with pytest.raises(SafeguardingRetentionAuthorityError, match="operation"):
        first.issue(operation="case.purged", body_sha256=_digest("wrong-operation"))
    clock.advance(seconds=301)
    with pytest.raises(SafeguardingRetentionAuthorityError, match="verification"):
        restarted.verify(receipt)


def test_automatic_retention_only_compacts_delivered_closed_min_age_cases(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    retention_authority = _retention_authority(clock)
    store = _store(
        tmp_path,
        clock=clock,
        authority=authority,
        policy=policy,
        delivery=_delivery(),
        retention_authority=retention_authority,
    )
    eligible = _open(
        store,
        authority,
        scope=scope_zh,
        key="retention-eligible-open-001",
        content="retention-eligible-content",
    )
    closed = _fully_deliver_and_close(
        store, authority, eligible, prefix="retention-eligible"
    )
    pending = _open(
        store,
        authority,
        scope=scope_zh,
        key="retention-pending-open-001",
        content="retention-pending-content",
    )
    acknowledged_pending = _open(
        store,
        authority,
        scope=scope_zh,
        key="retention-acknowledged-pending-open-001",
        content="retention-acknowledged-pending-content",
    )
    acknowledged_pending = store.acknowledge_case(
        case_id=acknowledged_pending["case_id"],
        expected_version=1,
        idempotency_key="retention-case-ack-pending-delivery",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.acknowledged",
            case_id=acknowledged_pending["case_id"],
            expected_version=1,
        ),
    )
    delivered_open = _open(
        store,
        authority,
        scope=scope_zh,
        key="retention-delivered-open-001",
        content="retention-delivered-open-content",
    )
    delivered_open = store.acknowledge_escalation(
        case_id=delivered_open["case_id"],
        expected_version=1,
        idempotency_key="retention-delivery-ack-open-case",
        authorization_receipt=_staff_receipt(
            authority,
            operation="escalation.acknowledged",
            case_id=delivered_open["case_id"],
            expected_version=1,
        ),
    )
    unavailable_store = _store(
        tmp_path / "delivery-unavailable",
        clock=clock,
        authority=authority,
        policy=policy,
        retention_authority=retention_authority,
    )
    unavailable_open = _open(
        unavailable_store,
        authority,
        scope=scope_zh,
        key="retention-unavailable-open-001",
        content="retention-unavailable-content",
    )
    unavailable_acknowledged = unavailable_store.acknowledge_case(
        case_id=unavailable_open["case_id"],
        expected_version=1,
        idempotency_key="retention-unavailable-case-ack",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.acknowledged",
            case_id=unavailable_open["case_id"],
            expected_version=1,
        ),
    )
    unavailable_closed = unavailable_store.close_case(
        case_id=unavailable_open["case_id"],
        expected_version=unavailable_acknowledged["version"],
        idempotency_key="retention-unavailable-case-close",
        authorization_receipt=_staff_receipt(
            authority,
            operation="case.closed",
            case_id=unavailable_open["case_id"],
            expected_version=unavailable_acknowledged["version"],
        ),
    )
    retention_policy = SafeguardingRetentionPolicy(
        policy_version="closed-delivered-retention-v1",
        minimum_closed_age_seconds=60,
    )
    assert store.retention_plan(retention_policy)["eligible_case_count"] == 0
    future = (clock() + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    with pytest.raises(SafeguardingIntegrityError, match="future"):
        store.retention_plan(retention_policy, as_of_utc=future)

    clock.advance(seconds=61)
    assert (
        unavailable_store.retention_plan(retention_policy)["eligible_case_count"] == 0
    )
    assert (
        unavailable_store.get_case(unavailable_closed["case_id"])["status"] == "closed"
    )
    plan = store.retention_plan(retention_policy)
    assert plan["eligible_case_count"] == 1
    assert plan["selected_case_count"] == 1
    assert plan["selected_event_count"] == 4
    assert eligible["case_id"] not in json.dumps(plan)
    wrong_receipt = retention_authority.issue(
        operation="case.retention_compacted",
        body_sha256=_digest("different-valid-authority-body"),
    )
    with pytest.raises(SafeguardingAuthorizationError, match="exact operation"):
        store.compact_retained_cases(
            retention_policy,
            as_of_utc=plan["as_of_utc"],
            authorization_receipt=wrong_receipt,
        )
    receipt = retention_authority.issue(
        operation="case.retention_compacted",
        body_sha256=plan["body_sha256"],
    )
    compacted = store.compact_retained_cases(
        retention_policy,
        as_of_utc=plan["as_of_utc"],
        authorization_receipt=receipt,
    )
    assert compacted == {
        "compacted": True,
        "cases_compacted": 1,
        "events_compacted": 4,
        "eligible_case_count": 1,
    }
    raw = store.path.read_text(encoding="utf-8")
    assert eligible["case_id"] not in raw
    assert eligible["content_sha256"] not in raw
    assert receipt["receipt_sha256"] in raw
    assert receipt["authority_proof_sha256"] not in raw
    assert store.get_case(pending["case_id"])["status"] == "open"
    assert store.get_case(acknowledged_pending["case_id"])["status"] == "acknowledged"
    assert store.get_case(delivered_open["case_id"])["status"] == "open"

    restarted_authority = _retention_authority(clock)
    restarted = _store(
        tmp_path,
        clock=clock,
        authority=authority,
        policy=policy,
        delivery=_delivery(),
        retention_authority=restarted_authority,
    )
    with pytest.raises(SafeguardingErasedError, match="erased"):
        restarted.get_case(closed["case_id"])
    capacity = restarted.capacity_status(retention_policy)
    assert capacity["cases_compacted_total"] == 1
    assert capacity["events_compacted_total"] == 4
    assert capacity["erasure_fence_inserted_count"] == 1
    assert capacity["erasure_fence_false_negative_possible"] is False
    assert capacity["erasure_fence_false_positive_policy"] == "fail_closed_as_erased"
    tampered_state = json.loads(restarted.path.read_text(encoding="utf-8"))
    tampered_state["retention_compaction"]["cases_compacted_total"] += 1
    restarted.path.write_text(
        json.dumps(
            tampered_state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SafeguardingIntegrityError, match="compaction"):
        restarted.recover()


def test_v1_first_write_migrates_and_erasure_fence_tamper_fails_closed(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    first = _open(
        store,
        authority,
        scope=scope_zh,
        key="v1-migration-first-open",
        content="v1-migration-first-content",
    )
    historical = json.loads(store.path.read_text(encoding="utf-8"))
    historical["schema"] = SAFEGUARDING_STORE_SCHEMA_V1
    historical["version"] = 1
    historical.pop("erasure_fence")
    historical.pop("retention_compaction")
    store.path.write_text(
        json.dumps(
            historical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    second = _open(
        store,
        authority,
        scope=scope_zh,
        key="v1-migration-second-open",
        content="v1-migration-second-content",
    )
    migrated = json.loads(store.path.read_text(encoding="utf-8"))
    assert migrated["schema"] == SAFEGUARDING_STORE_SCHEMA
    assert migrated["version"] == 2
    assert {first["case_id"], second["case_id"]} == set(store.recover()["cases"])
    fence_path = tmp_path / (
        f".safeguarding_erasure_fence.{migrated['erasure_fence']['generation']}.bin"
    )
    fence = bytearray(fence_path.read_bytes())
    assert len(fence) == 8 * 1024 * 1024
    fence[0] ^= 1
    fence_path.write_bytes(fence)
    with pytest.raises(SafeguardingIntegrityError, match="integrity check"):
        store.recover()


def test_bounded_exact_tombstones_keep_permanent_bloom_fence(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    capacity = SafeguardingCapacityLimits(
        maximum_events=16,
        maximum_store_bytes=32 * 1024 * 1024,
        recent_tombstones=2,
        near_event_headroom=1,
        near_byte_headroom=1024,
    )
    store = _store(
        tmp_path,
        clock=clock,
        authority=authority,
        policy=policy,
        capacity=capacity,
    )
    erased_ids: list[str] = []
    for index in range(4):
        opened = _open(
            store,
            authority,
            scope=scope_zh,
            key=f"bounded-tombstone-open-{index:03d}",
            content=f"bounded-tombstone-content-{index:03d}",
        )
        acknowledged = store.acknowledge_case(
            case_id=opened["case_id"],
            expected_version=1,
            idempotency_key=f"bounded-tombstone-ack-{index:03d}",
            authorization_receipt=_staff_receipt(
                authority,
                operation="case.acknowledged",
                case_id=opened["case_id"],
                expected_version=1,
            ),
        )
        closed = store.close_case(
            case_id=opened["case_id"],
            expected_version=acknowledged["version"],
            idempotency_key=f"bounded-tombstone-close-{index:03d}",
            authorization_receipt=_staff_receipt(
                authority,
                operation="case.closed",
                case_id=opened["case_id"],
                expected_version=acknowledged["version"],
            ),
        )
        store.purge_case(
            case_id=opened["case_id"],
            expected_version=closed["version"],
            idempotency_key=f"bounded-tombstone-purge-{index:03d}",
            authorization_receipt=_staff_receipt(
                authority,
                operation="case.purged",
                case_id=opened["case_id"],
                expected_version=closed["version"],
            ),
        )
        erased_ids.append(opened["case_id"])
        clock.advance(seconds=1)
    assert len(store.recover()["erasure_tombstones"]) == 2
    status = store.capacity_status()
    assert status["erasure_fence_inserted_count"] == 4
    assert status["recent_erasure_tombstone_headroom"] == 0
    with pytest.raises(SafeguardingErasedError, match="erased"):
        store.get_case(erased_ids[0])
    assert safeguarding_erasure_false_positive_upper_bound(500_000) < 1e-6


def test_bloom_membership_false_positive_is_fail_closed(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    import teaching_skill_miner.teacher_agent_safeguarding as safeguarding_module

    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    _open(store, authority, scope=scope_zh)
    state = json.loads(store.path.read_text(encoding="utf-8"))
    generation = state["erasure_fence"]["generation"]
    fence_path = tmp_path / f".safeguarding_erasure_fence.{generation}.bin"
    bits = bytearray(fence_path.read_bytes())
    absent_case_id = "sgc_" + "f" * 24
    absent_digest = _digest(absent_case_id)
    safeguarding_module._bloom_add(bits, absent_digest)
    fence_path.write_bytes(bits)
    state["erasure_fence"]["inserted_count"] = 1
    state["erasure_fence"]["bits_sha256"] = sha256(bits).hexdigest()
    state["erasure_fence"]["accumulator_sha256"] = _digest(
        "simulated-false-positive-boundary"
    )
    store.path.write_text(
        json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SafeguardingErasedError, match="erased"):
        store.get_case(absent_case_id)


def test_supervisor_near_capacity_compacts_then_new_urgent_survives_restart(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    scope_zh: str,
) -> None:
    from teaching_skill_miner.teacher_agent_safeguarding_supervisor import (
        SafeguardingDispatchSupervisor,
        write_safeguarding_route_record,
    )

    supervisor_root = tmp_path / "supervised"
    store_root = supervisor_root / "k1" / ("scope_" + "a" * 48) / "safeguarding"
    emergency_policy = default_emergency_resource_policy({scope_zh: "zh-CN"})
    capacity = SafeguardingCapacityLimits(
        maximum_events=6,
        maximum_store_bytes=32 * 1024 * 1024,
        recent_tombstones=2,
        near_event_headroom=2,
        near_byte_headroom=1024,
    )
    retention_authority = _retention_authority(clock)
    retention_policy = SafeguardingRetentionPolicy(
        policy_version="near-capacity-retention-v1",
        minimum_closed_age_seconds=1,
    )
    store = _store(
        store_root,
        clock=clock,
        authority=authority,
        policy=emergency_policy,
        delivery=_delivery(),
        retention_authority=retention_authority,
        capacity=capacity,
    )
    old = _open(
        store,
        authority,
        scope=scope_zh,
        key="near-capacity-old-case-open",
        content="near-capacity-old-case-content",
    )
    _fully_deliver_and_close(store, authority, old, prefix="near-capacity-old")
    pending = _open(
        store,
        authority,
        scope=scope_zh,
        key="near-capacity-pending-case-open",
        content="near-capacity-pending-case-content",
    )
    before = store.capacity_status(retention_policy)
    assert before["events"] == 5
    assert before["near_capacity"] is True
    clock.advance(seconds=2)
    write_safeguarding_route_record(store_root, "route_" + "b" * 48)

    class AcceptingReceiver:
        configured = True
        queue_sha256 = _delivery().queue_sha256

        def dispatch(self, row: dict) -> dict:
            return {"status": "accepted", "delivery_id": row["delivery_id"]}

    supervisor = SafeguardingDispatchSupervisor(
        supervisor_root,
        dispatcher_factory=lambda _route: AcceptingReceiver(),
        retention_policy=retention_policy,
        retention_authority=retention_authority,
        capacity_limits=capacity,
        wall_clock=clock,
    )
    compacted = supervisor.run_once()
    assert compacted["status"] == "ready"
    assert compacted["retention_status"] == "compacted"
    assert compacted["retention_cases_compacted"] == 1
    assert compacted["retention_events_compacted"] == 4
    assert compacted["retention_maximum_cases_per_run"] == 128
    assert compacted["capacity_event_headroom_min"] == 5
    assert compacted["retention_blocked_stores"] == 0
    assert compacted["erasure_fence_false_negative_possible"] is False
    assert compacted["erasure_fence_false_positive_within_target"] is True
    assert compacted["erasure_fence_false_positive_target_upper_bound"] == 1e-6
    assert ("scope_" + "a" * 48) not in json.dumps(compacted)

    restarted = _store(
        store_root,
        clock=clock,
        authority=authority,
        policy=emergency_policy,
        delivery=_delivery(),
        retention_authority=_retention_authority(clock),
        capacity=capacity,
    )
    assert restarted.get_case(pending["case_id"])["status"] == "open"
    new_urgent = _open(
        restarted,
        authority,
        scope=scope_zh,
        key="near-capacity-new-urgent-after-restart",
        content="near-capacity-new-urgent-content",
    )
    assert new_urgent["severity"] == "urgent"
    for index in range(3):
        _open(
            restarted,
            authority,
            scope=scope_zh,
            key=f"near-capacity-noneligible-{index:03d}",
            content=f"near-capacity-noneligible-content-{index:03d}",
        )
    blocked = supervisor.run_once()
    assert blocked["status"] == "degraded"
    assert blocked["retention_status"] == "blocked"
    assert blocked["capacity_near_limit_stores"] == 1
    assert blocked["retention_blocked_stores"] == 1
    assert blocked["retention_eligible_cases"] == 0


def test_store_and_public_receipts_match_json_schemas(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    case = _open(store, authority, scope=scope_zh)
    root = Path(__file__).resolve().parents[1] / "schema"
    store_schema = json.loads(
        (root / "teacher_agent_safeguarding_store.schema.json").read_text()
    )
    receipt_schema = json.loads(
        (
            root / "teacher_agent_safeguarding_authorization_receipt.schema.json"
        ).read_text()
    )
    retention_receipt_schema = json.loads(
        (
            root
            / "teacher_agent_safeguarding_retention_authorization_receipt.schema.json"
        ).read_text()
    )
    resource_schema = json.loads(
        (root / "teacher_agent_emergency_resource_bundle.schema.json").read_text()
    )
    Draft202012Validator.check_schema(store_schema)
    Draft202012Validator.check_schema(receipt_schema)
    Draft202012Validator.check_schema(retention_receipt_schema)
    Draft202012Validator.check_schema(resource_schema)
    Draft202012Validator(store_schema).validate(
        json.loads(store.path.read_text(encoding="utf-8"))
    )
    authorization = authority.issue(
        actor_kind="system_safety_classifier",
        operation="case.opened",
        body_sha256=safeguarding_open_authorization_body_sha256(
            scope_sha256=scope_zh,
            category="self_harm",
            severity="urgent",
            observed_at_utc="2026-08-12T11:59:00Z",
            content_sha256=_digest("learner-safety-content-001"),
        ),
    )
    Draft202012Validator(receipt_schema).validate(authorization)
    retention_receipt = _retention_authority(clock).issue(
        operation="case.retention_compacted",
        body_sha256=_digest("schema-retention-body"),
    )
    Draft202012Validator(retention_receipt_schema).validate(retention_receipt)
    Draft202012Validator(resource_schema).validate(
        store.emergency_resources_for_case(case["case_id"])
    )
    unknown_scope = _digest("schema-unknown-locale-scope")
    fallback_policy = default_emergency_resource_policy({unknown_scope: "fr-FR"})
    Draft202012Validator(resource_schema).validate(
        fallback_policy.resolve(unknown_scope)
    )
    repository = root.parent
    schema_path = (
        "schema/teacher_agent_safeguarding_retention_authorization_receipt.schema.json"
    )
    assert schema_path in (repository / "release/public_json_resources.txt").read_text()
    assert schema_path in (repository / "pyproject.toml").read_text()


def test_commit_fsyncs_file_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    policy: EmergencyResourcePolicy,
    scope_zh: str,
) -> None:
    import teaching_skill_miner.teacher_agent_safeguarding as module

    calls: list[int] = []
    real_fsync = module.os.fsync

    def recording_fsync(descriptor: int) -> None:
        calls.append(stat.S_IFMT(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", recording_fsync)
    store = _store(tmp_path, clock=clock, authority=authority, policy=policy)
    _open(store, authority, scope=scope_zh)
    assert stat.S_IFREG in calls
    assert stat.S_IFDIR in calls


def test_resource_policy_change_without_version_bump_fails_closed(
    tmp_path: Path,
    clock: FrozenClock,
    authority: ExternalServerAuthority,
    scope_zh: str,
) -> None:
    original = default_emergency_resource_policy({scope_zh: "zh-CN"})
    store = _store(tmp_path, clock=clock, authority=authority, policy=original)
    case = _open(store, authority, scope=scope_zh)
    changed = EmergencyResourcePolicy(
        policy_version=original.policy_version,
        trusted_locale_by_scope_sha256={scope_zh: "zh-CN"},
        resources_by_locale={
            "zh-CN": (
                {
                    "kind": "emergency_services",
                    "label": "紧急危险",
                    "instruction": "被错误静默修改的资源文案。",
                },
            )
        },
        generic_fallback=original.generic_fallback,
    )
    restarted = _store(tmp_path, clock=clock, authority=authority, policy=changed)
    with pytest.raises(SafeguardingConfigurationError, match="policy changed"):
        restarted.emergency_resources_for_case(case["case_id"])


def test_canonical_receipt_helper_does_not_accept_non_json_values() -> None:
    with pytest.raises(SafeguardingIntegrityError, match="canonical JSON"):
        canonical_sha256({"not_json": object()})
