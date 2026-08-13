from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import stat
import subprocess
import sys
import threading

from teaching_skill_miner.teacher_agent_safeguarding import (
    EscalationDeliveryConfig,
    TeacherAgentSafeguardingStore,
    default_emergency_resource_policy,
    safeguarding_open_authorization_body_sha256,
)
from teaching_skill_miner.teacher_agent_safeguarding_authority import (
    InternalSafeguardingSystemAuthority,
)
from teaching_skill_miner.teacher_agent_safeguarding_supervisor import (
    ROUTE_RECORD_SCHEMA,
    SafeguardingDispatchSupervisor,
    write_safeguarding_route_record,
)


class MonotonicClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class Receiver:
    configured = True
    queue_sha256 = sha256(b"supervised-safeguarding-queue-v1").hexdigest()

    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict] = []

    def dispatch(self, row):
        self.calls.append(deepcopy(dict(row)))
        if self.outcomes and not self.outcomes.pop(0):
            raise RuntimeError("private receiver outage")
        return {"status": "accepted", "delivery_id": row["delivery_id"]}


def test_supervisor_self_check_is_side_effect_free() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "teaching_skill_miner.teacher_agent_safeguarding_supervisor",
            "--self-check",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert (
        completed.stdout == "teacher_agent_safeguarding_supervisor self-check passed\n"
    )


def _seed_pending_case(root: Path, receiver: Receiver) -> tuple[Path, str]:
    safeguarding_root = root / "k1" / ("scope_" + "a" * 48) / "safeguarding"
    scope_sha256 = sha256(b"opaque-learner-scope").hexdigest()
    fixed_now = datetime(2026, 8, 12, 12, 20, tzinfo=timezone.utc)
    authority = InternalSafeguardingSystemAuthority(
        key=b"system-safeguarding-authority-key-32bytes",
        scope_sha256=scope_sha256,
        clock=lambda: fixed_now,
    )
    store = TeacherAgentSafeguardingStore(
        safeguarding_root,
        authorization_verifier=authority.verify,
        emergency_resource_policy=default_emergency_resource_policy(
            {scope_sha256: "zh-CN"}
        ),
        escalation_delivery=EscalationDeliveryConfig(
            policy_version="institution-safeguarding-routing-v1",
            queue_sha256=receiver.queue_sha256,
            sla_seconds_by_severity={
                "elevated": 86_400,
                "high": 14_400,
                "urgent": 900,
            },
        ),
        clock=lambda: fixed_now,
    )
    observed = "2026-08-12T12:00:00Z"
    content_sha256 = sha256(b"learner-disclosure-never-stored-here").hexdigest()
    body_sha256 = safeguarding_open_authorization_body_sha256(
        scope_sha256=scope_sha256,
        category="self_harm",
        severity="urgent",
        observed_at_utc=observed,
        content_sha256=content_sha256,
    )
    receipt = authority.issue(operation="case.opened", body_sha256=body_sha256)
    case = store.open_case(
        scope_sha256=scope_sha256,
        category="self_harm",
        severity="urgent",
        observed_at_utc=observed,
        content_sha256=content_sha256,
        idempotency_key="supervisor-observation-0001",
        authorization_receipt=receipt,
    )
    route = "route_" + "b" * 48
    route_path = write_safeguarding_route_record(safeguarding_root, route)
    assert stat.S_IMODE(route_path.stat().st_mode) == 0o600
    assert json.loads(route_path.read_text())["schema"] == ROUTE_RECORD_SCHEMA
    return safeguarding_root, case["escalation"]["delivery_id"]


def test_restart_retries_pending_delivery_without_learner_traffic(
    tmp_path: Path,
) -> None:
    receiver = Receiver([False, True])
    _store_root, delivery_id = _seed_pending_case(tmp_path, receiver)
    monotonic = MonotonicClock()

    def wall() -> datetime:
        return datetime(2026, 8, 12, 12, 40, tzinfo=timezone.utc)

    # First API lifecycle observes the receiver outage. The status is bounded,
    # aggregate-only, and marks the dependency degraded.
    first = SafeguardingDispatchSupervisor(
        tmp_path,
        dispatcher_factory=lambda _route: receiver,
        monotonic_clock=monotonic,
        wall_clock=wall,
    ).run_once()
    assert first["status"] == "degraded"
    assert first["failed"] == 1
    assert first["pending"] == 1
    assert first["overdue"] == 1

    # The learner worker is gone. A newly constructed central supervisor scans
    # the durable opaque root and immediately retries the same delivery ID,
    # without any learner/session request to recreate the worker.
    restarted = SafeguardingDispatchSupervisor(
        tmp_path,
        dispatcher_factory=lambda _route: receiver,
        monotonic_clock=monotonic,
        wall_clock=wall,
    ).run_once()
    assert restarted["status"] == "ready"
    assert restarted["accepted"] == 1
    assert [row["delivery_id"] for row in receiver.calls] == [delivery_id, delivery_id]
    serialized = json.dumps({"status": restarted, "calls": receiver.calls})
    assert "learner-disclosure-never-stored-here" not in serialized
    assert ("scope_" + "a" * 48) not in json.dumps(restarted)
    assert restarted["raw_learner_text_read_or_sent"] is False
    assert restarted["scope_identity_labels_exposed"] is False
    assert restarted["retention_enabled"] is False
    assert restarted["retention_status"] == "disabled"
    assert restarted["retention_policy_version_sha256"] is None
    assert restarted["retention_minimum_closed_age_seconds"] is None
    assert restarted["retention_maximum_cases_per_run"] is None
    assert restarted["erasure_fence_false_negative_possible"] is False
    assert restarted["erasure_fence_false_positive_within_target"] is True


def test_receiver_readiness_is_cached_and_failure_degrades_without_cases(
    tmp_path: Path,
) -> None:
    monotonic = MonotonicClock()
    outcomes = [False, True]
    attempts = 0

    def probe() -> dict:
        nonlocal attempts
        attempts += 1
        if not outcomes.pop(0):
            raise RuntimeError("private receiver credential detail")
        return {
            "status": "ready",
            "policy_version_validated": True,
            "receiver_network_validated": True,
            "credential_validated": True,
            "learner_content_sent": False,
            "case_created": False,
        }

    supervisor = SafeguardingDispatchSupervisor(
        tmp_path,
        dispatcher_factory=lambda _route: Receiver([]),
        readiness_probe=probe,
        monotonic_clock=monotonic,
    )
    failed = supervisor.run_once()
    assert failed["status"] == "degraded"
    assert failed["receiver_readiness_status"] == "failed"
    assert failed["receiver_network_validated"] is False
    assert failed["receiver_readiness_failures_total"] == 1
    assert attempts == 1
    assert supervisor.run_once()["receiver_readiness_status"] == "failed"
    assert attempts == 1

    monotonic.value = 1.0
    ready = supervisor.run_once()
    assert ready["status"] == "ready"
    assert ready["receiver_readiness_status"] == "ready"
    assert ready["receiver_network_validated"] is True
    assert ready["receiver_credential_validated"] is True
    assert ready["receiver_readiness_attempts_total"] == 2
    assert ready["receiver_readiness_successes_total"] == 1
    assert attempts == 2
    assert "private receiver" not in json.dumps(ready)


def test_route_record_and_scan_fail_closed_without_leaking_paths(
    tmp_path: Path,
) -> None:
    receiver = Receiver([True])
    safeguarding_root, _delivery_id = _seed_pending_case(tmp_path, receiver)
    route_path = safeguarding_root / ".safeguarding_dispatch_route_v1.json"
    route_path.write_text('{"schema":"wrong","route_locator":"route_aaaaaaaa"}\n')
    route_path.chmod(0o600)

    status = SafeguardingDispatchSupervisor(
        tmp_path,
        dispatcher_factory=lambda _route: receiver,
    ).run_once()
    assert status["status"] == "degraded"
    assert status["stores_unavailable"] == 1
    assert receiver.calls == []
    serialized = json.dumps(status)
    assert str(tmp_path) not in serialized
    assert "route_aaaaaaaa" not in serialized


def test_two_supervisors_never_dispatch_same_store_concurrently(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingReceiver(Receiver):
        def dispatch(self, row):
            self.calls.append(deepcopy(dict(row)))
            entered.set()
            assert release.wait(timeout=2)
            return {"status": "accepted", "delivery_id": row["delivery_id"]}

    receiver = BlockingReceiver([])
    _seed_pending_case(tmp_path, receiver)
    first = SafeguardingDispatchSupervisor(
        tmp_path, dispatcher_factory=lambda _route: receiver
    )
    second = SafeguardingDispatchSupervisor(
        tmp_path, dispatcher_factory=lambda _route: receiver
    )
    first_result: list[dict] = []
    thread = threading.Thread(target=lambda: first_result.append(first.run_once()))
    thread.start()
    assert entered.wait(timeout=2)
    coalesced = second.run_once()
    assert coalesced["attempted"] == 0
    assert len(receiver.calls) == 1
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert first_result[0]["accepted"] == 1
