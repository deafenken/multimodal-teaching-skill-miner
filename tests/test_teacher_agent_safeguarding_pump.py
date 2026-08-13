from __future__ import annotations

from copy import deepcopy
import json
import threading

from teaching_skill_miner.teacher_agent_safeguarding_pump import (
    SafeguardingDispatchPump,
)


DELIVERY_ID = "sge_" + "1" * 24


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class Store:
    def __init__(self) -> None:
        self.rows = [
            {
                "case_id": "sgc_" + "2" * 24,
                "case_version": 1,
                "scope_sha256": "3" * 64,
                "category": "self_harm",
                "severity": "urgent",
                "observed_at_utc": "2026-08-12T01:02:03Z",
                "content_sha256": "4" * 64,
                "delivery_id": DELIVERY_ID,
                "delivery_status": "pending",
                "queue_sha256": "5" * 64,
                "sla_due_at_utc": "2026-08-12T01:17:03Z",
            }
        ]

    def pending_escalations(self):
        return deepcopy(self.rows)


class Dispatcher:
    configured = True
    route_locator_sha256 = "6" * 64
    queue_sha256 = "5" * 64

    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict] = []

    def dispatch(self, row):
        self.calls.append(deepcopy(dict(row)))
        if self.outcomes and not self.outcomes.pop(0):
            raise RuntimeError("private receiver outage detail")
        return {"status": "accepted", "delivery_id": row["delivery_id"]}


def test_outage_backoff_accepted_retry_ack_and_restart_are_idempotent() -> None:
    clock = Clock()
    store = Store()
    dispatcher = Dispatcher([False, False, True, True])
    pump = SafeguardingDispatchPump(
        store=store,
        dispatcher=dispatcher,
        clock=clock,
        failure_retry_seconds=2,
        maximum_retry_seconds=60,
        accepted_retry_seconds=60,
    )

    first = pump.run_once()
    assert first["failed"] == 1
    assert pump.run_once()["attempted"] == 0
    clock.advance(2)
    assert pump.run_once()["failed"] == 1
    clock.advance(3.9)
    assert pump.run_once()["attempted"] == 0
    clock.advance(0.1)
    accepted = pump.run_once()
    assert accepted["accepted"] == 1
    assert accepted["durable_delivery_acknowledged"] is False
    assert pump.run_once()["attempted"] == 0

    # A process restart loses only the in-memory delay. The same durable row is
    # retried immediately with its stable delivery ID, so receiver idempotency
    # covers the crash window after HTTP accepted and before its separate ack.
    restarted = SafeguardingDispatchPump(
        store=store,
        dispatcher=dispatcher,
        clock=clock,
    )
    assert restarted.run_once()["accepted"] == 1
    assert [call["delivery_id"] for call in dispatcher.calls] == [DELIVERY_ID] * 4

    store.rows = []
    assert restarted.run_once()["pending"] == 0
    serialized = json.dumps(dispatcher.calls, ensure_ascii=False)
    assert "learner_text" not in serialized
    assert "message" not in serialized


def test_forever_loop_attempts_immediately_and_collapses_store_failure() -> None:
    store = Store()
    dispatcher = Dispatcher([True])
    stopping = threading.Event()
    observed: list[dict] = []

    def observe(result):
        observed.append(dict(result))
        stopping.set()

    SafeguardingDispatchPump(
        store=store,
        dispatcher=dispatcher,
        observer=observe,
    ).run_forever(stopping)
    assert observed[0]["accepted"] == 1
    assert len(dispatcher.calls) == 1

    class BrokenStore:
        def pending_escalations(self):
            raise RuntimeError("private durable path")

    failed = SafeguardingDispatchPump(
        store=BrokenStore(),
        dispatcher=dispatcher,
    ).run_once()
    assert failed["store_available"] is False
    assert "private" not in json.dumps(failed)
