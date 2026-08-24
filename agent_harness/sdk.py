"""Stable in-process Python SDK for Agent Harness.

The SDK is intentionally a thin authority-preserving facade over
``AgentRunner``.  Every turn re-applies the immutable thread permission mode,
and one client-wide lock covers that update, attachment import, execution, and
failure cleanup.  This prevents two thread handles from racing the runner's
mutable permission field and accidentally borrowing each other's authority.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
import math
import os
from pathlib import Path
from queue import Queue
from threading import Event, Lock, RLock, Thread, get_ident
from types import MappingProxyType
from typing import Any, AsyncIterator, Callable, Iterator, Literal, Mapping, Sequence

from .attachments import AttachmentDescriptor
from .core import (
    HARNESS_EVENT_SCHEMA,
    HARNESS_EVENT_TYPES,
    ApprovalBroker,
    CancellationToken,
)
from .runner import AgentRunner, TurnOutcome
from .toolsets import PERMISSION_PROFILES


SDK_RUN_RESULT_SCHEMA = "agent_harness.sdk_run_result.v1"

PermissionMode = Literal["read-only", "workspace-write", "full-access"]
PathInput = str | os.PathLike[str]
EventCallback = Callable[["HarnessEvent"], None]


class HarnessSdkContractError(ValueError):
    """Raised when an SDK value violates its public contract."""


def _permission_mode(value: object) -> PermissionMode:
    if not isinstance(value, str) or value not in PERMISSION_PROFILES:
        raise HarnessSdkContractError("permission_mode is invalid")
    return value  # type: ignore[return-value]


def _path_tuple(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, os.PathLike)):
        raise HarnessSdkContractError("attachments must be a sequence of paths")
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError:
        raise HarnessSdkContractError("attachments must be a sequence of paths") from None
    result: list[str] = []
    for raw in raw_values:
        try:
            value = os.fspath(raw)
        except TypeError:
            raise HarnessSdkContractError("attachment path is invalid") from None
        if not isinstance(value, str) or not value or "\x00" in value:
            raise HarnessSdkContractError("attachment path is invalid")
        result.append(value)
    return tuple(result)


def _freeze_json(value: Any, *, field_name: str) -> Any:
    """Detach one canonical JSON-shaped value into recursively immutable data."""

    active: set[int] = set()

    def freeze(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise HarnessSdkContractError(f"{field_name} must be canonical JSON")
            return item
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                raise HarnessSdkContractError(f"{field_name} must not contain cycles")
            active.add(identity)
            try:
                frozen: dict[str, Any] = {}
                for key, nested in item.items():
                    if not isinstance(key, str):
                        raise HarnessSdkContractError(
                            f"{field_name} object keys must be strings"
                        )
                    frozen[key] = freeze(nested)
                return MappingProxyType(frozen)
            finally:
                active.remove(identity)
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            identity = id(item)
            if identity in active:
                raise HarnessSdkContractError(f"{field_name} must not contain cycles")
            active.add(identity)
            try:
                return tuple(freeze(nested) for nested in item)
            finally:
                active.remove(identity)
        raise HarnessSdkContractError(f"{field_name} must be canonical JSON")

    return freeze(value)


@dataclass(frozen=True, slots=True)
class HarnessClientOptions:
    """Immutable configuration used to create one workspace-bound client."""

    workspace: PathInput
    active_directory: PathInput | None = None
    state_home: PathInput | None = None
    api_key_file: PathInput | None = None
    model: str | None = None
    deadline_seconds: float = 180.0
    max_steps: int = 12
    subagents_enabled: bool = True
    worktree_home: PathInput | None = None
    project_extensions_enabled: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "workspace",
            "active_directory",
            "state_home",
            "api_key_file",
            "worktree_home",
        ):
            raw = getattr(self, field_name)
            if raw is None:
                continue
            try:
                value = os.fspath(raw)
            except TypeError:
                raise HarnessSdkContractError(f"{field_name} is invalid") from None
            if not isinstance(value, str) or not value or "\x00" in value:
                raise HarnessSdkContractError(f"{field_name} is invalid")
            object.__setattr__(self, field_name, value)
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip()
        ):
            raise HarnessSdkContractError("model is invalid")
        if type(self.subagents_enabled) is not bool:
            raise HarnessSdkContractError("subagents_enabled must be a boolean")
        if type(self.project_extensions_enabled) is not bool:
            raise HarnessSdkContractError(
                "project_extensions_enabled must be a boolean"
            )
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(self.deadline_seconds)
            or not 0.05 <= self.deadline_seconds <= 86_400
        ):
            raise HarnessSdkContractError(
                "deadline_seconds must be a finite number in [0.05, 86400]"
            )
        object.__setattr__(self, "deadline_seconds", float(self.deadline_seconds))
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or not 1 <= self.max_steps <= 64
        ):
            raise HarnessSdkContractError("max_steps must be an integer in [1, 64]")


@dataclass(frozen=True, slots=True)
class HarnessThreadOptions:
    """Immutable options for a newly created thread."""

    title: str = "New session"
    permission_mode: PermissionMode = "read-only"

    def __post_init__(self) -> None:
        if not isinstance(self.title, str):
            raise HarnessSdkContractError("thread title is invalid")
        title = " ".join(self.title.strip().split())
        if not title:
            raise HarnessSdkContractError("thread title is invalid")
        object.__setattr__(self, "title", title[:80])
        object.__setattr__(self, "permission_mode", _permission_mode(self.permission_mode))


@dataclass(frozen=True, slots=True)
class HarnessRunOptions:
    """Immutable options for one turn.

    ``approval_broker`` is the only SDK route that can answer an interactive
    approval.  Omitting it retains the Runner's fail-closed headless behavior:
    medium/high effects settle as a handoff rather than being auto-approved.
    """

    attachments: tuple[PathInput, ...] = ()
    cancellation_token: CancellationToken | None = None
    on_event: EventCallback | None = None
    approval_broker: ApprovalBroker | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "attachments", _path_tuple(self.attachments))
        if self.cancellation_token is not None and not isinstance(
            self.cancellation_token, CancellationToken
        ):
            raise HarnessSdkContractError("cancellation_token is invalid")
        if self.on_event is not None and not callable(self.on_event):
            raise HarnessSdkContractError("on_event must be callable")
        if self.approval_broker is not None and not callable(
            getattr(self.approval_broker, "decide", None)
        ):
            raise HarnessSdkContractError("approval_broker is invalid")


@dataclass(frozen=True, slots=True)
class HarnessEvent:
    """One detached, recursively immutable durable runtime event."""

    schema: str
    event_id: str
    run_id: str
    turn_id: str
    sequence: int
    type: str
    timestamp: str
    payload: Mapping[str, Any]
    causation_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HarnessEvent":
        if not isinstance(value, Mapping) or value.get("schema") != HARNESS_EVENT_SCHEMA:
            raise HarnessSdkContractError("event envelope is invalid")
        text_fields: dict[str, str] = {}
        for field_name in ("event_id", "run_id", "turn_id", "type", "timestamp"):
            raw = value.get(field_name)
            if not isinstance(raw, str) or not raw:
                raise HarnessSdkContractError("event envelope is invalid")
            text_fields[field_name] = raw
        if text_fields["type"] not in HARNESS_EVENT_TYPES:
            raise HarnessSdkContractError("event type is invalid")
        sequence = value.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise HarnessSdkContractError("event sequence is invalid")
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise HarnessSdkContractError("event payload is invalid")
        causation_id = value.get("causation_id")
        if causation_id is not None and (
            not isinstance(causation_id, str) or not causation_id
        ):
            raise HarnessSdkContractError("event causation_id is invalid")
        frozen_payload = _freeze_json(payload, field_name="event payload")
        return cls(
            schema=HARNESS_EVENT_SCHEMA,
            event_id=text_fields["event_id"],
            run_id=text_fields["run_id"],
            turn_id=text_fields["turn_id"],
            sequence=sequence,
            type=text_fields["type"],
            timestamp=text_fields["timestamp"],
            payload=frozen_payload,
            causation_id=causation_id,
        )


@dataclass(frozen=True, slots=True)
class HarnessRunResult:
    """Immutable terminal result for one thread turn."""

    schema: str
    session_id: str
    run_id: str
    turn_id: str
    status: str
    final_response: str | None
    reason: str
    usage: Mapping[str, int]

    @property
    def message(self) -> str | None:
        """Compatibility spelling for ``final_response``."""

        return self.final_response

    @classmethod
    def from_outcome(cls, outcome: TurnOutcome) -> "HarnessRunResult":
        if not isinstance(outcome, TurnOutcome):
            raise HarnessSdkContractError("runner outcome is invalid")
        for value in (outcome.session_id, outcome.run_id, outcome.turn_id):
            if not isinstance(value, str) or not value:
                raise HarnessSdkContractError("runner outcome identity is invalid")
        if outcome.status not in {
            "completed",
            "failed",
            "cancelled",
            "handoff",
        }:
            raise HarnessSdkContractError("runner outcome status is invalid")
        if outcome.message is not None and not isinstance(outcome.message, str):
            raise HarnessSdkContractError("runner outcome message is invalid")
        if not isinstance(outcome.reason, str):
            raise HarnessSdkContractError("runner outcome reason is invalid")
        usage: dict[str, int] = {}
        for key, value in outcome.usage.items():
            if (
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise HarnessSdkContractError("runner usage is invalid")
            usage[key] = value
        return cls(
            schema=SDK_RUN_RESULT_SCHEMA,
            session_id=outcome.session_id,
            run_id=outcome.run_id,
            turn_id=outcome.turn_id,
            status=outcome.status,
            final_response=outcome.message,
            reason=outcome.reason,
            usage=MappingProxyType(usage),
        )


def _referenced_attachment_ids(
    runner: AgentRunner,
    session_id: str,
) -> set[str] | None:
    """Return exact durable references, or ``None`` when safety is uncertain."""

    try:
        session = runner.store.load(session_id)
    except Exception:
        return None
    messages = session.get("messages")
    if not isinstance(messages, list):
        return None
    referenced: set[str] = set()
    for message in messages:
        if not isinstance(message, Mapping):
            return None
        manifest = message.get("attachments", [])
        if not isinstance(manifest, list):
            return None
        for item in manifest:
            if not isinstance(item, Mapping):
                return None
            attachment_id = item.get("attachment_id")
            if not isinstance(attachment_id, str):
                return None
            referenced.add(attachment_id)
    return referenced


def _discard_unreferenced(
    runner: AgentRunner,
    session_id: str,
    descriptors: Sequence[AttachmentDescriptor],
) -> None:
    referenced = _referenced_attachment_ids(runner, session_id)
    if referenced is None:
        return
    for descriptor in descriptors:
        if descriptor.attachment_id in referenced:
            continue
        try:
            runner.discard_attachment(descriptor)
        except Exception:
            # Cleanup is best-effort and cannot replace the authoritative turn
            # exception.  A failed integrity check deliberately preserves data.
            continue


def _cancel_and_capture(
    token: CancellationToken,
    reason: str,
) -> BaseException | None:
    """Request cancellation without letting a callback skip worker settlement."""

    try:
        token.cancel(reason)
    except BaseException as exc:
        return exc
    return None


class HarnessClient:
    """Synchronous workspace client with serialized authority transitions."""

    def __init__(self, options: HarnessClientOptions) -> None:
        if not isinstance(options, HarnessClientOptions):
            raise HarnessSdkContractError("options must be HarnessClientOptions")
        self.options = options
        self._runner = AgentRunner(
            options.workspace,
            active_directory=options.active_directory,
            state_home=options.state_home,
            api_key_file=options.api_key_file,
            model=options.model,
            permission_mode="read-only",
            deadline_seconds=options.deadline_seconds,
            max_steps=options.max_steps,
            subagents_enabled=options.subagents_enabled,
            worktree_home=options.worktree_home,
            project_extensions_enabled=options.project_extensions_enabled,
        )
        # This lock intentionally spans attachment import and conservative
        # cleanup as well as the turn itself.  AgentRunner owns a mutable
        # permission mode, so narrower per-thread locking would permit
        # cross-authorizing concurrent thread handles.
        self._operation_lock = RLock()
        self._operation_owner: int | None = None

    @property
    def workspace(self) -> Path:
        return self._runner.workspace

    def start_thread(
        self,
        options: HarnessThreadOptions | None = None,
    ) -> "HarnessThread":
        selected = options or HarnessThreadOptions()
        if not isinstance(selected, HarnessThreadOptions):
            raise HarnessSdkContractError("options must be HarnessThreadOptions")
        with self._serialized_operation():
            self._runner.permission_mode = selected.permission_mode
            session = self._runner.new_session(title=selected.title)
        return self._thread_from_session(session, selected.permission_mode)

    def resume_thread(
        self,
        session_id: str,
        *,
        permission_mode: PermissionMode = "read-only",
    ) -> "HarnessThread":
        mode = _permission_mode(permission_mode)
        if not isinstance(session_id, str) or not session_id.strip():
            raise HarnessSdkContractError("session_id is invalid")
        with self._serialized_operation():
            session = self._runner.resume_session(
                session_id.strip(),
                permission_override=mode,
            )
        return self._thread_from_session(session, mode)

    def fork_thread(
        self,
        session_id: str,
        *,
        permission_mode: PermissionMode = "read-only",
    ) -> "HarnessThread":
        mode = _permission_mode(permission_mode)
        if not isinstance(session_id, str) or not session_id.strip():
            raise HarnessSdkContractError("session_id is invalid")
        with self._serialized_operation():
            session = self._runner.store.fork(session_id.strip())
            # Stored permission metadata is not authority, but immediately
            # narrowing the fork keeps every user-facing view accurate too.
            session = self._runner.set_permission_mode(str(session["session_id"]), mode)
        return self._thread_from_session(session, mode)

    def _thread_from_session(
        self,
        session: Mapping[str, Any],
        permission_mode: PermissionMode,
    ) -> "HarnessThread":
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise HarnessSdkContractError("session identity is invalid")
        return HarnessThread(self, session_id, permission_mode)

    @contextmanager
    def _serialized_operation(self) -> Iterator[None]:
        """Serialize the Runner seam and reject callback-driven re-entry."""

        identity = get_ident()
        with self._operation_lock:
            if self._operation_owner == identity:
                raise RuntimeError("SDK client operations cannot be nested")
            self._operation_owner = identity
            try:
                yield
            finally:
                self._operation_owner = None

    def _execute_turn(
        self,
        thread: "HarnessThread",
        prompt: str,
        options: HarnessRunOptions,
        *,
        stream_sink: EventCallback | None = None,
    ) -> HarnessRunResult:
        if not isinstance(prompt, str):
            raise HarnessSdkContractError("prompt must be a string")
        if not isinstance(options, HarnessRunOptions):
            raise HarnessSdkContractError("options must be HarnessRunOptions")

        def event_sink(raw: Mapping[str, Any]) -> None:
            event = HarnessEvent.from_mapping(raw)
            if stream_sink is not None:
                stream_sink(event)
            if options.on_event is not None:
                options.on_event(event)

        descriptors: tuple[AttachmentDescriptor, ...] = ()
        with self._serialized_operation():
            # Reapply authority for every invocation.  Session metadata alone
            # is never trusted as an authorization source.
            self._runner.set_permission_mode(thread.session_id, thread.permission_mode)
            try:
                if options.attachments:
                    descriptors = self._runner.ingest_attachments(options.attachments)
                outcome = self._runner.run_turn(
                    thread.session_id,
                    prompt,
                    cancellation_token=options.cancellation_token,
                    event_sink=(
                        event_sink
                        if stream_sink is not None or options.on_event is not None
                        else None
                    ),
                    approval_broker=options.approval_broker,
                    attachments=descriptors,
                )
                if outcome.session_id != thread.session_id:
                    raise HarnessSdkContractError(
                        "runner outcome session identity does not match the thread"
                    )
                return HarnessRunResult.from_outcome(outcome)
            except BaseException:
                if descriptors:
                    _discard_unreferenced(
                        self._runner,
                        thread.session_id,
                        descriptors,
                    )
                raise


@dataclass(frozen=True, slots=True)
class HarnessThread:
    """Immutable handle for one persisted Agent Harness session."""

    _client: HarnessClient = field(repr=False, compare=False)
    session_id: str
    permission_mode: PermissionMode

    def __post_init__(self) -> None:
        if not isinstance(self._client, HarnessClient):
            raise HarnessSdkContractError("client is invalid")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise HarnessSdkContractError("session_id is invalid")
        object.__setattr__(self, "permission_mode", _permission_mode(self.permission_mode))

    def run(
        self,
        prompt: str,
        options: HarnessRunOptions | None = None,
    ) -> HarnessRunResult:
        return self._client._execute_turn(self, prompt, options or HarnessRunOptions())

    def run_stream(
        self,
        prompt: str,
        options: HarnessRunOptions | None = None,
    ) -> "HarnessEventStream":
        return HarnessEventStream(self, prompt, options or HarnessRunOptions())

    def fork(
        self,
        *,
        permission_mode: PermissionMode = "read-only",
    ) -> "HarnessThread":
        return self._client.fork_thread(
            self.session_id,
            permission_mode=permission_mode,
        )


_STREAM_END = object()


class HarnessEventStream(Iterator[HarnessEvent]):
    """Non-daemon streaming turn with cooperative cancel-and-join cleanup."""

    def __init__(
        self,
        thread: HarnessThread,
        prompt: str,
        options: HarnessRunOptions,
    ) -> None:
        if not isinstance(thread, HarnessThread):
            raise HarnessSdkContractError("thread is invalid")
        if not isinstance(prompt, str):
            raise HarnessSdkContractError("prompt must be a string")
        if not isinstance(options, HarnessRunOptions):
            raise HarnessSdkContractError("options must be HarnessRunOptions")
        token = options.cancellation_token or CancellationToken()
        if options.cancellation_token is None:
            options = HarnessRunOptions(
                attachments=options.attachments,
                cancellation_token=token,
                on_event=options.on_event,
                approval_broker=options.approval_broker,
            )
        self._harness_thread = thread
        self._prompt = prompt
        self._options = options
        self._token = token
        self._queue: Queue[HarnessEvent | object] = Queue()
        self._worker: Thread | None = None
        self._state_lock = Lock()
        self._consumer_lock = Lock()
        self._done = Event()
        self._close_done = Event()
        self._started = False
        self._closed = False
        self._closing = False
        self._closing_owner: int | None = None
        self._exhausted = False
        self._result: HarnessRunResult | None = None
        self._error: BaseException | None = None

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._token

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def _ensure_started(self) -> None:
        with self._state_lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("event stream is closed")
            self._started = True
            worker = Thread(
                target=self._run,
                name=f"agent-harness-sdk-{self._harness_thread.session_id[-12:]}",
                daemon=False,
            )
            self._worker = worker
            worker.start()

    def _run(self) -> None:
        try:
            self._result = self._harness_thread._client._execute_turn(
                self._harness_thread,
                self._prompt,
                self._options,
                stream_sink=self._queue.put,
            )
        except BaseException as exc:
            self._error = exc
        finally:
            self._done.set()
            self._queue.put(_STREAM_END)

    def __iter__(self) -> "HarnessEventStream":
        return self

    def __next__(self) -> HarnessEvent:
        with self._consumer_operation():
            return self._next_item()

    def _next_item(self, *, settle_after_close: bool = False) -> HarnessEvent:
        self._reject_worker_reentry()
        if (self._closed and not settle_after_close) or self._exhausted:
            raise StopIteration
        self._ensure_started()
        item = self._queue.get()
        if item is _STREAM_END:
            self._exhausted = True
            worker = self._worker
            if worker is not None:
                worker.join()
            if self._error is not None:
                raise self._error
            raise StopIteration
        if not isinstance(item, HarnessEvent):
            raise HarnessSdkContractError("event stream item is invalid")
        return item

    def cancel(self, reason: str = "sdk_cancelled") -> bool:
        return self._token.cancel(reason)

    def _reject_worker_reentry(self) -> None:
        worker = self._worker
        if worker is not None and worker.ident == get_ident():
            raise RuntimeError(
                "an event stream cannot be consumed or closed from its worker callback"
            )

    @contextmanager
    def _consumer_operation(self) -> Iterator[None]:
        """Allow one blocking consumer operation without starving a sentinel waiter."""

        self._reject_worker_reentry()
        if not self._consumer_lock.acquire(blocking=False):
            raise RuntimeError("an event stream cannot be consumed concurrently")
        try:
            yield
        finally:
            self._consumer_lock.release()

    def close(self) -> None:
        self._reject_worker_reentry()
        identity = get_ident()
        with self._state_lock:
            if self._close_done.is_set():
                return
            if self._closing:
                if self._closing_owner == identity:
                    # CancellationToken callbacks run synchronously. A close
                    # callback re-entering the active closer must not wait on
                    # the completion event that this same stack will publish.
                    return
                wait_for_close = True
                worker = None
            else:
                wait_for_close = False
                self._closing = True
                self._closing_owner = identity
                self._closed = True
                worker = self._worker
        if wait_for_close:
            self._close_done.wait()
            return
        cancellation_error: BaseException | None = None
        try:
            cancellation_error = _cancel_and_capture(
                self._token,
                "sdk_stream_closed",
            )
            if worker is not None:
                worker.join()
        finally:
            # Every concurrent close waiter observes worker settlement rather
            # than returning merely because close has started.
            with self._state_lock:
                self._closing_owner = None
            self._close_done.set()
        if cancellation_error is not None:
            raise cancellation_error

    def _wait_for_close_settlement(self) -> None:
        with self._state_lock:
            same_closer = (
                self._closing_owner == get_ident() and not self._close_done.is_set()
            )
        if same_closer:
            raise RuntimeError(
                "an event stream result cannot wait from its close callback"
            )
        self._close_done.wait()

    def result(self) -> HarnessRunResult:
        with self._consumer_operation():
            if self._closed:
                if not self._started:
                    raise RuntimeError("event stream was closed before it started")
                self._wait_for_close_settlement()
            else:
                while not self._exhausted:
                    try:
                        self._next_item(settle_after_close=True)
                    except StopIteration:
                        break
            if self._error is not None:
                raise self._error
            if self._result is None:
                raise RuntimeError("event stream did not produce a result")
            return self._result

    def __enter__(self) -> "HarnessEventStream":
        self._ensure_started()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


async def _await_future_settlement(future: asyncio.Future[Any]) -> None:
    """Wait through repeated task cancellation until a worker has stopped."""

    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    try:
        future.exception()
    except BaseException:
        pass


async def _run_lifecycle_operation(
    callback: Callable[..., HarnessThread],
    *args: Any,
    **kwargs: Any,
) -> HarnessThread:
    """Run one durable thread lifecycle call without abandoning its worker."""

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, partial(callback, *args, **kwargs))
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        # These synchronous lifecycle calls do not expose cooperative
        # cancellation. Wait until their durable mutation has definitively
        # settled before propagating cancellation to the embedding task.
        await _await_future_settlement(future)
        raise


class AsyncHarnessClient:
    """Async facade using non-abandoning worker-thread boundaries."""

    def __init__(self, options: HarnessClientOptions) -> None:
        self._client = HarnessClient(options)

    @property
    def workspace(self) -> Path:
        return self._client.workspace

    async def start_thread(
        self,
        options: HarnessThreadOptions | None = None,
    ) -> "AsyncHarnessThread":
        sync_thread = await _run_lifecycle_operation(
            self._client.start_thread,
            options,
        )
        return AsyncHarnessThread(sync_thread)

    async def resume_thread(
        self,
        session_id: str,
        *,
        permission_mode: PermissionMode = "read-only",
    ) -> "AsyncHarnessThread":
        sync_thread = await _run_lifecycle_operation(
            self._client.resume_thread,
            session_id,
            permission_mode=permission_mode,
        )
        return AsyncHarnessThread(sync_thread)

    async def fork_thread(
        self,
        session_id: str,
        *,
        permission_mode: PermissionMode = "read-only",
    ) -> "AsyncHarnessThread":
        sync_thread = await _run_lifecycle_operation(
            self._client.fork_thread,
            session_id,
            permission_mode=permission_mode,
        )
        return AsyncHarnessThread(sync_thread)


@dataclass(frozen=True, slots=True)
class AsyncHarnessThread:
    """Async handle for one persisted Agent Harness session."""

    _thread: HarnessThread = field(repr=False, compare=False)

    @property
    def session_id(self) -> str:
        return self._thread.session_id

    @property
    def permission_mode(self) -> PermissionMode:
        return self._thread.permission_mode

    async def run(
        self,
        prompt: str,
        options: HarnessRunOptions | None = None,
    ) -> HarnessRunResult:
        selected = options or HarnessRunOptions()
        token = selected.cancellation_token or CancellationToken()
        if selected.cancellation_token is None:
            selected = HarnessRunOptions(
                attachments=selected.attachments,
                cancellation_token=token,
                on_event=selected.on_event,
                approval_broker=selected.approval_broker,
            )
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, self._thread.run, prompt, selected)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            cancellation_error = _cancel_and_capture(
                token,
                "async_task_cancelled",
            )
            await _await_future_settlement(future)
            if cancellation_error is not None:
                raise cancellation_error
            raise

    def run_stream(
        self,
        prompt: str,
        options: HarnessRunOptions | None = None,
    ) -> "AsyncHarnessEventStream":
        return AsyncHarnessEventStream(
            self._thread.run_stream(prompt, options or HarnessRunOptions())
        )

    async def fork(
        self,
        *,
        permission_mode: PermissionMode = "read-only",
    ) -> "AsyncHarnessThread":
        sync_thread = await _run_lifecycle_operation(
            self._thread.fork,
            permission_mode=permission_mode,
        )
        return AsyncHarnessThread(sync_thread)


def _next_stream_item(
    stream: HarnessEventStream,
) -> tuple[bool, HarnessEvent | None]:
    try:
        return True, next(stream)
    except StopIteration:
        return False, None


class AsyncHarnessEventStream(AsyncIterator[HarnessEvent]):
    """Async iterator over a cooperatively cancellable synchronous stream."""

    def __init__(self, stream: HarnessEventStream) -> None:
        self._stream = stream
        self._closed = False

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._stream.cancellation_token

    @property
    def done(self) -> bool:
        return self._stream.done

    def __aiter__(self) -> "AsyncHarnessEventStream":
        return self

    async def __anext__(self) -> HarnessEvent:
        if self._closed:
            raise StopAsyncIteration
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, _next_stream_item, self._stream)
        try:
            available, item = await asyncio.shield(future)
        except asyncio.CancelledError:
            cancellation_error = _cancel_and_capture(
                self._stream.cancellation_token,
                "async_stream_cancelled",
            )
            close_future = loop.run_in_executor(None, self._stream.close)
            await _await_future_settlement(future)
            await _await_future_settlement(close_future)
            self._closed = True
            if cancellation_error is not None:
                raise cancellation_error
            raise
        if not available:
            self._closed = True
            raise StopAsyncIteration
        if item is None:
            raise HarnessSdkContractError("async event stream item is invalid")
        return item

    async def aclose(self) -> None:
        if self._closed:
            return
        cancellation_error = _cancel_and_capture(
            self._stream.cancellation_token,
            "async_stream_closed",
        )
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, self._stream.close)
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            await _await_future_settlement(future)
            self._closed = True
            if cancellation_error is not None:
                raise cancellation_error
            raise
        else:
            self._closed = True
            if cancellation_error is not None:
                raise cancellation_error

    async def result(self) -> HarnessRunResult:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, self._stream.result)
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            cancellation_error = _cancel_and_capture(
                self._stream.cancellation_token,
                "async_stream_result_cancelled",
            )
            close_future = loop.run_in_executor(None, self._stream.close)
            await _await_future_settlement(future)
            await _await_future_settlement(close_future)
            self._closed = True
            if cancellation_error is not None:
                raise cancellation_error
            raise
        self._closed = True
        return result

    async def __aenter__(self) -> "AsyncHarnessEventStream":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()


__all__ = [
    "SDK_RUN_RESULT_SCHEMA",
    "AsyncHarnessClient",
    "AsyncHarnessEventStream",
    "AsyncHarnessThread",
    "EventCallback",
    "HarnessClient",
    "HarnessClientOptions",
    "HarnessEvent",
    "HarnessEventStream",
    "HarnessRunOptions",
    "HarnessRunResult",
    "HarnessSdkContractError",
    "HarnessThread",
    "HarnessThreadOptions",
    "PathInput",
    "PermissionMode",
]
