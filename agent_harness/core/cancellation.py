"""Cooperative cancellation and deadline helpers."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Event, Lock
import time
from typing import Callable, Iterator, Protocol

from .contracts import HarnessCancelled, HarnessDeadlineExceeded


class HarnessClock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class CancellationToken:
    """Thread-safe turn-scoped cancellation token."""

    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._reason = "cancelled"
        self._committed = False
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._next_callback_id = 1

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    @property
    def committed(self) -> bool:
        """Whether an authoritative side effect already won the cancel race."""

        with self._lock:
            return self._committed

    def cancel(self, reason: str = "cancelled") -> bool:
        clean = str(reason).strip()[:300] or "cancelled"
        with self._lock:
            if self._event.is_set() or self._committed:
                return False
            self._reason = clean
            self._event.set()
            callbacks = tuple(self._callbacks.values())
            self._callbacks.clear()
        fatal_callback_error: BaseException | None = None
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # Cancellation must remain monotonic even when one transport
                # cleanup hook has already lost its underlying connection.
                continue
            except BaseException as exc:
                # KeyboardInterrupt/SystemExit-like application callbacks are
                # still propagated, but only after every registered transport
                # cleanup has had its chance to settle.
                if fatal_callback_error is None:
                    fatal_callback_error = exc
        if fatal_callback_error is not None:
            raise fatal_callback_error
        return True

    @contextmanager
    def commit_guard(self) -> Iterator[None]:
        """Fence a durable commit against concurrent cancellation.

        Cancellation and commit share this lock.  A cancel that wins first
        prevents entry; once the guarded block returns normally, subsequent
        cancellation is rejected so callers cannot report a cancelled run
        whose domain state was already committed.
        """

        self._lock.acquire()
        try:
            if self._event.is_set():
                raise HarnessCancelled(self._reason)
            if self._committed:
                raise RuntimeError("cancellation token is already committed")
            yield
            self._committed = True
            # A committed operation will never execute transport cancellation
            # callbacks.  Drop any stale closures immediately rather than
            # retaining sockets or provider objects for the handle lifetime.
            self._callbacks.clear()
        finally:
            self._lock.release()

    def add_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register transport cleanup and return an idempotent unsubscribe."""

        if not callable(callback):
            raise TypeError("cancellation callback must be callable")
        with self._lock:
            if self._event.is_set():
                execute_now = True
                callback_id = 0
            elif self._committed:
                execute_now = False
                callback_id = 0
            else:
                execute_now = False
                callback_id = self._next_callback_id
                self._next_callback_id += 1
                self._callbacks[callback_id] = callback
        if execute_now:
            callback()

        def unsubscribe() -> None:
            if callback_id == 0:
                return
            with self._lock:
                self._callbacks.pop(callback_id, None)

        return unsubscribe

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise HarnessCancelled(self.reason)

    def wait(self, seconds: float) -> bool:
        """Wait up to ``seconds``; return true when cancelled."""

        return self._event.wait(max(0.0, seconds))


def remaining_seconds(*, clock: HarnessClock, deadline_monotonic: float) -> float:
    return max(0.0, deadline_monotonic - clock.monotonic())


def check_deadline(*, clock: HarnessClock, deadline_monotonic: float) -> None:
    if remaining_seconds(clock=clock, deadline_monotonic=deadline_monotonic) <= 0:
        raise HarnessDeadlineExceeded("harness run deadline exceeded")


def interruptible_sleep(
    seconds: float,
    *,
    clock: HarnessClock,
    cancellation_token: CancellationToken,
    deadline_monotonic: float,
) -> None:
    """Sleep in bounded slices so cancellation and fake clocks both work."""

    remaining = max(0.0, seconds)
    while remaining > 0:
        cancellation_token.raise_if_cancelled()
        check_deadline(clock=clock, deadline_monotonic=deadline_monotonic)
        slice_seconds = min(remaining, 0.05)
        # A real token wait wakes immediately.  Deterministic fake clocks need
        # their own sleep method to advance virtual time.
        if isinstance(clock, SystemClock):
            if cancellation_token.wait(slice_seconds):
                cancellation_token.raise_if_cancelled()
        else:
            clock.sleep(slice_seconds)
        remaining -= slice_seconds
