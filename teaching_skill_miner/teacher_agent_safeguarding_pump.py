"""Restart-safe delivery pump for the content-free safeguarding outbox.

The durable store remains the only source of pending/acknowledged truth.  A
receiver's HTTP ``accepted`` response merely schedules a slower idempotent
retry; only the separately authorized escalation acknowledgement operation
removes the row from ``pending_escalations()``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import re
import threading
import time
from typing import Any, Protocol

from .teacher_agent_safeguarding_dispatch import SafeguardingDispatcher


_DELIVERY_ID = re.compile(r"^sge_[0-9a-f]{24}$")


class SafeguardingOutbox(Protocol):
    def pending_escalations(self) -> list[dict[str, Any]]: ...


class SafeguardingDispatchPumpError(RuntimeError):
    """Raised only for invalid local pump configuration."""


class SafeguardingDispatchPump:
    """Deliver durable rows with bounded retry and no inferred acknowledgement."""

    def __init__(
        self,
        *,
        store: SafeguardingOutbox,
        dispatcher: SafeguardingDispatcher,
        clock: Callable[[], float] = time.monotonic,
        poll_seconds: float = 1.0,
        failure_retry_seconds: float = 2.0,
        maximum_retry_seconds: float = 60.0,
        accepted_retry_seconds: float = 60.0,
        maximum_batch: int = 8,
        observer: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if (
            not callable(getattr(store, "pending_escalations", None))
            or not getattr(dispatcher, "configured", False)
            or not callable(getattr(dispatcher, "dispatch", None))
            or not callable(clock)
            or not isinstance(poll_seconds, (int, float))
            or isinstance(poll_seconds, bool)
            or not 0.1 <= float(poll_seconds) <= 60.0
            or not isinstance(failure_retry_seconds, (int, float))
            or isinstance(failure_retry_seconds, bool)
            or not 0.1 <= float(failure_retry_seconds) <= 60.0
            or not isinstance(maximum_retry_seconds, (int, float))
            or isinstance(maximum_retry_seconds, bool)
            or not float(failure_retry_seconds)
            <= float(maximum_retry_seconds)
            <= 3600.0
            or not isinstance(accepted_retry_seconds, (int, float))
            or isinstance(accepted_retry_seconds, bool)
            or not 1.0 <= float(accepted_retry_seconds) <= 3600.0
            or isinstance(maximum_batch, bool)
            or not isinstance(maximum_batch, int)
            or not 1 <= maximum_batch <= 128
            or (observer is not None and not callable(observer))
        ):
            raise SafeguardingDispatchPumpError(
                "safeguarding dispatch pump configuration is invalid"
            )
        self._store = store
        self._dispatcher = dispatcher
        self._clock = clock
        self.poll_seconds = float(poll_seconds)
        self.failure_retry_seconds = float(failure_retry_seconds)
        self.maximum_retry_seconds = float(maximum_retry_seconds)
        self.accepted_retry_seconds = float(accepted_retry_seconds)
        self.maximum_batch = maximum_batch
        self._observer = observer
        self._next_attempt: dict[str, float] = {}
        self._failure_streak: dict[str, int] = {}
        self._run_lock = threading.Lock()

    @staticmethod
    def _result(
        *,
        pending: int,
        attempted: int,
        accepted: int,
        failed: int,
        store_available: bool,
        coalesced: bool = False,
    ) -> dict[str, Any]:
        return {
            "schema": "teaching_skill_miner.safeguarding_dispatch_pump.v1",
            "pending": pending,
            "attempted": attempted,
            "accepted": accepted,
            "failed": failed,
            "store_available": store_available,
            "coalesced": coalesced,
            "durable_delivery_acknowledged": False,
            "raw_learner_text_sent": False,
        }

    def run_once(self) -> dict[str, Any]:
        """Attempt a bounded batch; exceptions are collapsed for a durable loop."""

        if not self._run_lock.acquire(blocking=False):
            return self._result(
                pending=0,
                attempted=0,
                accepted=0,
                failed=0,
                store_available=True,
                coalesced=True,
            )
        try:
            try:
                now = float(self._clock())
                if not 0 <= now < float("inf"):
                    raise ValueError("invalid clock")
                rows = self._store.pending_escalations()
                if not isinstance(rows, list):
                    raise TypeError("invalid outbox")
            except Exception:
                return self._result(
                    pending=0,
                    attempted=0,
                    accepted=0,
                    failed=0,
                    store_available=False,
                )

            delivery_ids = {
                row.get("delivery_id")
                for row in rows
                if isinstance(row, Mapping)
                and isinstance(row.get("delivery_id"), str)
                and _DELIVERY_ID.fullmatch(str(row["delivery_id"])) is not None
            }
            self._next_attempt = {
                delivery_id: due
                for delivery_id, due in self._next_attempt.items()
                if delivery_id in delivery_ids
            }
            self._failure_streak = {
                delivery_id: count
                for delivery_id, count in self._failure_streak.items()
                if delivery_id in delivery_ids
            }
            attempted = accepted = failed = 0
            for row in rows:
                if attempted >= self.maximum_batch or not isinstance(row, Mapping):
                    continue
                delivery_id = row.get("delivery_id")
                if (
                    not isinstance(delivery_id, str)
                    or _DELIVERY_ID.fullmatch(delivery_id) is None
                    or self._next_attempt.get(delivery_id, 0.0) > now
                ):
                    continue
                attempted += 1
                try:
                    self._dispatcher.dispatch(row)
                except Exception:
                    failed += 1
                    streak = min(self._failure_streak.get(delivery_id, 0) + 1, 31)
                    self._failure_streak[delivery_id] = streak
                    delay = min(
                        self.maximum_retry_seconds,
                        self.failure_retry_seconds * (2 ** min(streak - 1, 20)),
                    )
                    self._next_attempt[delivery_id] = now + delay
                else:
                    accepted += 1
                    self._failure_streak.pop(delivery_id, None)
                    self._next_attempt[delivery_id] = (
                        now + self.accepted_retry_seconds
                    )
            return self._result(
                pending=len(rows),
                attempted=attempted,
                accepted=accepted,
                failed=failed,
                store_available=True,
            )
        finally:
            self._run_lock.release()

    def run_forever(self, stopping: threading.Event) -> None:
        """Run immediately, then poll until the worker lifecycle stops."""

        if not isinstance(stopping, threading.Event):
            raise SafeguardingDispatchPumpError(
                "safeguarding dispatch pump stop event is invalid"
            )
        while not stopping.is_set():
            result = self.run_once()
            if self._observer is not None:
                try:
                    self._observer(result)
                except Exception:
                    # Observability cannot stop delivery or expose its adapter.
                    pass
            stopping.wait(self.poll_seconds)


__all__ = [
    "SafeguardingDispatchPump",
    "SafeguardingDispatchPumpError",
    "SafeguardingOutbox",
]
