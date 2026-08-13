from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import json
from pathlib import Path

import pytest

from teaching_skill_miner.teacher_agent_authority import (
    AUTHENTICATED_TEACHER_ACTOR,
    AUTHORITY_ASSURANCE,
    TEACHER_AUTHORITY_SCHEMA,
    TeacherAuthorityVerifier,
    canonical_bytes,
    canonical_sha256,
)
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardError,
    build_teacher_agent_dashboard_snapshot,
)
from teaching_skill_miner.teacher_agent_safeguarding import (
    EscalationDeliveryConfig,
    SafeguardingConflictError,
    TeacherAgentSafeguardingStore,
    default_emergency_resource_policy,
)
from teaching_skill_miner.teacher_agent_safeguarding_authority import (
    CompositeSafeguardingAuthorizationVerifier,
    InternalSafeguardingStaffAuthority,
    InternalSafeguardingSystemAuthority,
    SAFEGUARDING_GATEWAY_ROLE_POLICY_SHA256,
)
from teaching_skill_miner.teacher_agent_safeguarding_dispatch import (
    SAFEGUARDING_DISPATCH_ACK_SCHEMA,
    SafeguardingDispatchError,
)


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCOPE_ID = "scope_" + "a" * 48
SCOPE_SHA256 = sha256(b"opaque-learner-scope").hexdigest()
AUTHORITY_KEY = b"g" * 32
SAFEGUARDING_KEY = b"safeguarding-staff-test-key-32bytes"


class Clock:
    def __init__(self) -> None:
        self.value = datetime.now(timezone.utc).replace(microsecond=0)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


class FakeDispatcher:
    configured = True
    route_locator_sha256 = sha256(b"server-route-locator").hexdigest()
    queue_sha256 = sha256(b"configured-safeguarding-queue").hexdigest()

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    def dispatch(self, row):
        self.calls.append(deepcopy(dict(row)))
        if self.fail:
            raise SafeguardingDispatchError("safeguarding dispatch failed")
        return {
            "schema": SAFEGUARDING_DISPATCH_ACK_SCHEMA,
            "status": "accepted",
            "delivery_id": row["delivery_id"],
            "route_locator_sha256": self.route_locator_sha256,
            "raw_learner_text_sent": False,
            "durable_delivery_acknowledged": False,
        }


def _signature(value: dict) -> str:
    return (
        base64.urlsafe_b64encode(
            hmac.new(AUTHORITY_KEY, canonical_bytes(value), sha256).digest()
        )
        .decode("ascii")
        .rstrip("=")
    )


def _signed_request(
    body: dict,
    *,
    path: str,
    nonce: int,
    clock: Clock,
    role_policy_sha256: str = SAFEGUARDING_GATEWAY_ROLE_POLICY_SHA256,
) -> dict:
    idempotency_key = body["safeguarding_idempotency_key"]
    issued_at = clock()
    envelope = {
        "schema": TEACHER_AUTHORITY_SCHEMA,
        "authority_kind": AUTHENTICATED_TEACHER_ACTOR,
        "assurance": AUTHORITY_ASSURANCE,
        "scope_id": SCOPE_ID,
        "scope_key_version": "k1",
        "actor_principal_sha256": sha256(b"safeguarding-officer-7").hexdigest(),
        "roles_sha256": canonical_sha256(["safeguarding"]),
        "role_policy_sha256": role_policy_sha256,
        "method": "POST",
        "path": path,
        "body_sha256": canonical_sha256(body),
        "idempotency_key_sha256": sha256(idempotency_key.encode()).hexdigest(),
        "issued_at": issued_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "expires_at": (issued_at + timedelta(minutes=2))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "nonce": "tan_" + f"{nonce:048x}",
    }
    envelope["authority_id"] = "tauth_" + canonical_sha256(envelope)[:24]
    envelope["signature"] = _signature(envelope)
    return {**deepcopy(body), "_teacher_authority": envelope}


def _snapshot(root: Path, clock: Clock, dispatcher: FakeDispatcher):
    teacher = TeacherAuthorityVerifier(
        key=AUTHORITY_KEY,
        scope_id=SCOPE_ID,
        scope_key_version="k1",
        replay_store_path=root / "teacher-authority.jsonl",
        clock=clock,
    )
    system = InternalSafeguardingSystemAuthority(
        key=b"system-safeguarding-test-key-32bytes!",
        scope_sha256=SCOPE_SHA256,
        clock=clock,
    )
    staff = InternalSafeguardingStaffAuthority(
        key=SAFEGUARDING_KEY,
        scope_sha256=SCOPE_SHA256,
        gateway_receipt_verifier=teacher.verify_verification_receipt,
        clock=clock,
    )
    composite = CompositeSafeguardingAuthorizationVerifier(
        system=system,
        staff=staff,
    )
    store = TeacherAgentSafeguardingStore(
        root / "safeguarding",
        authorization_verifier=composite.verify,
        emergency_resource_policy=default_emergency_resource_policy(
            {SCOPE_SHA256: "zh-CN"}
        ),
        escalation_delivery=EscalationDeliveryConfig(
            policy_version="school-safeguarding-v1",
            queue_sha256=dispatcher.queue_sha256,
            sla_seconds_by_severity={"elevated": 30, "high": 30, "urgent": 30},
        ),
        clock=clock,
    )
    snapshot = build_teacher_agent_dashboard_snapshot(
        DATA / "teacher_agent_skill_library_v2.json",
        DATA / "teacher_agent_demo_input.json",
        DATA / "teacher_agent_evaluation_cases.json",
        store_path=root / "sessions.jsonl",
        teacher_authority_verifier=teacher,
        safeguarding_store=store,
        safeguarding_scope_sha256=SCOPE_SHA256,
        safeguarding_system_authority_issuer=system.issue,
        safeguarding_system_authority_verifier=system.verify,
        safeguarding_staff_authority_issuer=staff.issue,
        safeguarding_dispatcher=dispatcher,
    )
    return snapshot, store


def _open_case(snapshot) -> dict:
    return snapshot.chat(
        {"messages": [{"role": "user", "content": "我现在就要自杀"}], "web_search": False}
    )["safety_obligation"]["safeguarding"]


def test_staff_workflow_dispatch_cas_sla_idempotency_and_restart(tmp_path: Path) -> None:
    clock = Clock()
    dispatcher = FakeDispatcher()
    snapshot, store = _snapshot(tmp_path, clock, dispatcher)
    opened = _open_case(snapshot)
    case_id = opened["case_id"]

    listed = snapshot.list_safeguarding_cases(
        _signed_request(
            {"safeguarding_idempotency_key": "staff-list-0001"},
            path="api/safeguarding/list",
            nonce=1,
            clock=clock,
        )
    )
    assert listed["cases"][0]["case_id"] == case_id
    assert listed["cases"][0]["content_sha256"]
    assert "content" not in listed["cases"][0]

    dispatched = snapshot.dispatch_safeguarding_case(
        _signed_request(
            {
                "case_id": case_id,
                "expected_version": 1,
                "safeguarding_idempotency_key": "staff-dispatch-0001",
            },
            path="api/safeguarding/dispatch",
            nonce=2,
            clock=clock,
        )
    )
    assert dispatched["durable_delivery_acknowledged"] is False
    assert store.get_case(case_id)["escalation"]["delivery_status"] == "pending"
    assert dispatcher.calls[0]["content_sha256"]
    assert "我现在就要自杀" not in json.dumps(dispatcher.calls, ensure_ascii=False)

    clock.advance(31)
    overdue = snapshot.record_safeguarding_escalation_overdue(
        _signed_request(
            {
                "case_id": case_id,
                "expected_version": 1,
                "safeguarding_idempotency_key": "staff-overdue-0001",
            },
            path="api/safeguarding/escalation/overdue",
            nonce=3,
            clock=clock,
        )
    )["case"]
    assert overdue["version"] == 2
    assert overdue["delivery_status"] == "overdue"

    delivery = snapshot.acknowledge_safeguarding_escalation(
        _signed_request(
            {
                "case_id": case_id,
                "expected_version": 2,
                "safeguarding_idempotency_key": "staff-delivery-ack-0001",
            },
            path="api/safeguarding/escalation/acknowledge",
            nonce=4,
            clock=clock,
        )
    )["case"]
    assert delivery["version"] == 3
    assert delivery["delivery_status"] == "acknowledged"

    ack_body = {
        "case_id": case_id,
        "expected_version": 3,
        "safeguarding_idempotency_key": "staff-case-ack-0001",
    }
    acknowledged = snapshot.acknowledge_safeguarding_case(
        _signed_request(
            ack_body,
            path="api/safeguarding/case/acknowledge",
            nonce=5,
            clock=clock,
        )
    )["case"]
    replay = snapshot.acknowledge_safeguarding_case(
        _signed_request(
            ack_body,
            path="api/safeguarding/case/acknowledge",
            nonce=6,
            clock=clock,
        )
    )["case"]
    assert acknowledged == replay
    assert acknowledged["version"] == 4

    with pytest.raises(SafeguardingConflictError, match="expected_version"):
        snapshot.close_safeguarding_case(
            _signed_request(
                {
                    "case_id": case_id,
                    "expected_version": 3,
                    "safeguarding_idempotency_key": "staff-close-stale-0001",
                },
                path="api/safeguarding/case/close",
                nonce=7,
                clock=clock,
            )
        )

    restarted, restarted_store = _snapshot(tmp_path, clock, dispatcher)
    closed = restarted.close_safeguarding_case(
        _signed_request(
            {
                "case_id": case_id,
                "expected_version": 4,
                "safeguarding_idempotency_key": "staff-close-0001",
            },
            path="api/safeguarding/case/close",
            nonce=8,
            clock=clock,
        )
    )["case"]
    assert closed["status"] == "closed"
    assert restarted_store.get_case(case_id)["version"] == 5


def test_wrong_role_and_dispatch_outage_fail_before_mutation(tmp_path: Path) -> None:
    clock = Clock()
    dispatcher = FakeDispatcher(fail=True)
    snapshot, store = _snapshot(tmp_path, clock, dispatcher)
    opened = _open_case(snapshot)
    body = {
        "case_id": opened["case_id"],
        "expected_version": 1,
        "safeguarding_idempotency_key": "wrong-role-list-0001",
    }
    with pytest.raises(TeacherAgentDashboardError, match="safeguarding role"):
        snapshot.dispatch_safeguarding_case(
            _signed_request(
                body,
                path="api/safeguarding/dispatch",
                nonce=9,
                clock=clock,
                role_policy_sha256=canonical_sha256(["teacher"]),
            )
        )
    assert dispatcher.calls == []
    assert store.get_case(opened["case_id"])["version"] == 1

    with pytest.raises(SafeguardingDispatchError, match="dispatch failed"):
        snapshot.dispatch_safeguarding_case(
            _signed_request(
                body,
                path="api/safeguarding/dispatch",
                nonce=10,
                clock=clock,
            )
        )
    assert len(dispatcher.calls) == 1
    assert store.get_case(opened["case_id"])["version"] == 1
