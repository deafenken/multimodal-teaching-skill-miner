"""Thread-safe run controls used by session actors and streaming transports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from threading import Condition, Lock, Thread
import time
from typing import Any, Callable, Iterator, Mapping

from .cancellation import CancellationToken
from .contracts import HARNESS_SCHEMA, HarnessContractError, TERMINAL_EVENT_TYPES
from .events import canonical_json
from .journal import HarnessJournal


class SteeringQueue:
    """A bounded queue for input that should steer the active operation."""

    def __init__(self, *, max_items: int = 8, max_chars: int = 4_000) -> None:
        if not 1 <= max_items <= 64 or not 80 <= max_chars <= 20_000:
            raise HarnessContractError("steering queue limits are invalid")
        self._max_items = max_items
        self._max_chars = max_chars
        self._items: list[dict[str, Any]] = []
        self._lock = Lock()

    def submit(self, content: str, *, input_id: str | None = None) -> str:
        clean = str(content).strip()
        if not clean or len(clean) > self._max_chars:
            raise HarnessContractError("steering input is empty or too long")
        identifier = str(input_id or f"steer_{time.time_ns():x}").strip()
        if not identifier or len(identifier) > 160:
            raise HarnessContractError("steering input_id is invalid")
        with self._lock:
            if len(self._items) >= self._max_items:
                raise HarnessContractError("steering queue is full")
            if any(item["input_id"] == identifier for item in self._items):
                raise HarnessContractError("steering input_id is already queued")
            self._items.append(
                {"kind": "steer", "input_id": identifier, "content": clean}
            )
        return identifier

    def drain(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            items = tuple(deepcopy(self._items))
            self._items.clear()
            return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


@dataclass(frozen=True, slots=True)
class FollowUpItem:
    input_id: str
    payload: Mapping[str, Any]


class FollowUpQueue:
    """Bounded FIFO for operations that begin after the active run settles."""

    def __init__(self, *, max_items: int = 16) -> None:
        if not 1 <= max_items <= 128:
            raise HarnessContractError("follow-up queue limit is invalid")
        self._max_items = max_items
        self._items: list[FollowUpItem] = []
        self._lock = Lock()

    def submit(
        self, payload: Mapping[str, Any], *, input_id: str | None = None
    ) -> str:
        if not isinstance(payload, Mapping):
            raise HarnessContractError("follow-up payload must be an object")
        identifier = str(input_id or f"followup_{time.time_ns():x}").strip()
        with self._lock:
            if len(self._items) >= self._max_items:
                raise HarnessContractError("follow-up queue is full")
            if any(item.input_id == identifier for item in self._items):
                raise HarnessContractError("follow-up input_id is already queued")
            self._items.append(FollowUpItem(identifier, deepcopy(dict(payload))))
        return identifier

    def pop(self) -> FollowUpItem | None:
        with self._lock:
            if not self._items:
                return None
            return self._items.pop(0)

    def remove(self, input_id: str) -> bool:
        with self._lock:
            for index, item in enumerate(self._items):
                if item.input_id == input_id:
                    del self._items[index]
                    return True
            return False

    def snapshot(self) -> tuple[FollowUpItem, ...]:
        with self._lock:
            return tuple(deepcopy(self._items))


class HarnessRunHandle:
    """One active run with replayable in-memory events and real cancellation."""

    def __init__(
        self,
        *,
        run_id: str,
        turn_id: str,
        cancellation_token: CancellationToken | None = None,
        steering_queue: SteeringQueue | None = None,
        journal: HarnessJournal | None = None,
        max_in_memory_events: int = 512,
        max_in_memory_event_bytes: int = 1_000_000,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 160:
            raise HarnessContractError("run handle run_id is invalid")
        if not isinstance(turn_id, str) or not turn_id.strip() or len(turn_id) > 160:
            raise HarnessContractError("run handle turn_id is invalid")
        if not 1 <= max_in_memory_events <= 20_000:
            raise HarnessContractError("run handle event count limit is invalid")
        if not 1_024 <= max_in_memory_event_bytes <= 100_000_000:
            raise HarnessContractError("run handle event byte limit is invalid")
        if journal is not None:
            if not isinstance(journal, HarnessJournal):
                raise HarnessContractError("run handle journal is invalid")
            if journal.run_id != run_id or journal.turn_id != turn_id:
                raise HarnessContractError(
                    "run handle journal identifiers do not match"
                )
        self.run_id = run_id
        self.turn_id = turn_id
        self.journal = journal
        self.cancellation_token = cancellation_token or CancellationToken()
        self.steering_queue = (
            steering_queue if steering_queue is not None else SteeringQueue()
        )
        self._max_in_memory_events = max_in_memory_events
        self._max_in_memory_event_bytes = max_in_memory_event_bytes
        self._condition = Condition()
        self._events: list[dict[str, Any]] = []
        self._events_bytes = 0
        self._dropped_through_sequence = 0
        self._last_observed_sequence = 0
        self._closed_from_journal = False
        self._result: dict[str, Any] | None = None
        self._error: BaseException | None = None
        self._thread: Thread | None = None
        if journal is not None:
            last_sequence = journal.last_sequence
            tail_after = max(0, last_sequence - max_in_memory_events)
            for event in journal.replay(after_sequence=tail_after):
                self._append_bounded(event)
            self._last_observed_sequence = last_sequence
            self._dropped_through_sequence = max(
                self._dropped_through_sequence,
                tail_after,
            )
            self._closed_from_journal = journal.terminal_type is not None

    @staticmethod
    def _event_size(event: Mapping[str, Any]) -> int:
        return len(canonical_json(dict(event)).encode("utf-8"))

    def _append_bounded(self, event: Mapping[str, Any]) -> None:
        normalized = deepcopy(dict(event))
        self._events.append(normalized)
        self._events_bytes += self._event_size(normalized)
        while self._events and (
            len(self._events) > self._max_in_memory_events
            or self._events_bytes > self._max_in_memory_event_bytes
        ):
            removed = self._events.pop(0)
            self._events_bytes -= self._event_size(removed)
            self._dropped_through_sequence = max(
                self._dropped_through_sequence,
                int(removed.get("sequence", 0)),
            )

    def event_sink(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping):
            raise HarnessContractError("run handle event must be an object")
        sequence = event.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise HarnessContractError("run handle event sequence is invalid")
        if event.get("run_id") != self.run_id or event.get("turn_id") != self.turn_id:
            raise HarnessContractError("run handle event identifiers do not match")
        if self.journal is not None:
            acknowledgement = self.journal.last_ack
            if (
                acknowledgement is None
                or acknowledgement.sequence != sequence
                or acknowledgement.event_id != event.get("event_id")
                or not acknowledgement.durable
            ):
                durable = self.journal.replay(after_sequence=sequence - 1)
                if not durable or durable[0] != dict(event):
                    raise HarnessContractError(
                        "run handle refused an event without a durable journal ack"
                    )
        with self._condition:
            if (
                self._last_observed_sequence
                and sequence != self._last_observed_sequence + 1
            ):
                raise HarnessContractError("run handle event sequence is not contiguous")
            self._append_bounded(event)
            self._last_observed_sequence = sequence
            if event.get("type") in TERMINAL_EVENT_TYPES:
                self._closed_from_journal = self.journal is not None
            self._condition.notify_all()

    def start(self, target: Callable[[], dict[str, Any]]) -> None:
        with self._condition:
            if self._thread is not None:
                raise HarnessContractError("run handle is already started")
            if self._closed_from_journal:
                raise HarnessContractError("cannot start a terminal journal handle")

            def execute() -> None:
                try:
                    result = target()
                    if not isinstance(result, Mapping) or result.get("schema") != HARNESS_SCHEMA:
                        raise HarnessContractError("run target returned an invalid result")
                    with self._condition:
                        self._result = deepcopy(dict(result))
                except BaseException as exc:
                    with self._condition:
                        self._error = exc
                finally:
                    with self._condition:
                        self._condition.notify_all()

            self._thread = Thread(
                target=execute,
                name=f"harness-run-{self.run_id[:32]}",
                daemon=True,
            )
            self._thread.start()

    @property
    def settled(self) -> bool:
        with self._condition:
            return (
                self._result is not None
                or self._error is not None
                or self._closed_from_journal
            )

    def cancel(self, reason: str = "user_requested") -> bool:
        return self.cancellation_token.cancel(reason)

    def steer(self, content: str, *, input_id: str | None = None) -> str:
        if self.settled:
            raise HarnessContractError("cannot steer a settled run")
        return self.steering_queue.submit(content, input_id=input_id)

    def events_after(self, sequence: int = 0) -> list[dict[str, Any]]:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise HarnessContractError("event cursor cannot be negative")
        if self.journal is not None:
            return self.journal.replay(after_sequence=sequence)
        with self._condition:
            if sequence < self._dropped_through_sequence:
                raise HarnessContractError(
                    "event cursor expired from the bounded in-memory window"
                )
            return [
                deepcopy(item)
                for item in self._events
                if int(item.get("sequence", 0)) > sequence
            ]

    def iter_events(
        self,
        *,
        after_sequence: int = 0,
        heartbeat_seconds: float = 10.0,
    ) -> Iterator[dict[str, Any] | None]:
        """Yield new events; yield ``None`` as a transport heartbeat."""

        cursor = after_sequence
        while True:
            available = self.events_after(cursor)
            with self._condition:
                settled = (
                    self._result is not None
                    or self._error is not None
                    or self._closed_from_journal
                )
                if not available and not settled:
                    self._condition.wait(timeout=heartbeat_seconds)
            if not available and not settled:
                available = self.events_after(cursor)
                with self._condition:
                    settled = (
                        self._result is not None
                        or self._error is not None
                        or self._closed_from_journal
                    )
            if not available:
                if settled:
                    return
                yield None
                continue
            for event in available:
                cursor = int(event["sequence"])
                yield event
                if event.get("type") in TERMINAL_EVENT_TYPES:
                    return

    def durable_status(self) -> dict[str, Any]:
        """Return a bounded recovery view for a journal-backed handle."""

        if self.journal is None:
            raise HarnessContractError("run handle does not have a durable journal")
        terminal_type = self.journal.terminal_type
        if terminal_type is None:
            raise HarnessContractError("durable journal is not terminal")
        last_sequence = self.journal.last_sequence
        terminal_events = self.journal.replay(after_sequence=last_sequence - 1)
        if len(terminal_events) != 1:
            raise HarnessContractError("durable terminal event is unavailable")
        return {
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "terminal_type": terminal_type,
            "last_sequence": last_sequence,
            "terminal_event": terminal_events[0],
            "checkpoint": self.journal.load_checkpoint(),
        }

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        with self._condition:
            if (
                self._closed_from_journal
                and self._thread is None
                and self._result is None
                and self._error is None
            ):
                raise HarnessContractError(
                    "terminal journal replay has no live result; use durable_status()"
                )
            if self._result is None and self._error is None:
                self._condition.wait_for(
                    lambda: self._result is not None or self._error is not None,
                    timeout=timeout,
                )
            if self._error is not None:
                raise self._error
            if self._result is None:
                raise TimeoutError("harness run is still active")
            return deepcopy(self._result)
