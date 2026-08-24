from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event, Lock, Thread
import time
from types import MappingProxyType
from typing import Any, Iterator

import pytest

import agent_harness
import agent_harness.runner as runner_module

from agent_harness.attachments import MAX_ATTACHMENTS_TOTAL_BYTES, AttachmentError
from agent_harness.core import (
    ApprovalDecision,
    ApprovalRequest,
    CancellationToken,
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
    ProviderStreamEvent,
)
from agent_harness.providers.deepseek_client import DEFAULT_MODEL
from agent_harness.core.events import HarnessEventEmitter
from agent_harness.runner import TurnOutcome
from agent_harness.sdk import (
    SDK_RUN_RESULT_SCHEMA,
    AsyncHarnessClient,
    HarnessClient,
    HarnessClientOptions,
    HarnessEvent,
    HarnessRunOptions,
    HarnessRunResult,
    HarnessSdkContractError,
    HarnessThreadOptions,
)


class _FatalCancellationCallback(BaseException):
    pass


def test_sdk_surface_is_exported_from_the_root_package() -> None:
    assert agent_harness.HarnessClient is HarnessClient
    assert agent_harness.AsyncHarnessClient is AsyncHarnessClient
    assert agent_harness.SDK_RUN_RESULT_SCHEMA == SDK_RUN_RESULT_SCHEMA


class _StreamingFinalModel:
    def __init__(self, *, attachment_support: bool = True) -> None:
        self.attachment_support = attachment_support
        self.requests: list[Any] = []
        self._lock = Lock()

    @property
    def model_spec(self) -> ProviderModelSpec:
        attachment_kinds = ("text",) if self.attachment_support else ()
        attachment_mime_types = (
            (
                "text/plain; charset=utf-8",
                "text/markdown; charset=utf-8",
            )
            if self.attachment_support
            else ()
        )
        return ProviderModelSpec(
            provider="test",
            model="sdk-test-v1",
            capabilities=ProviderCapabilities(
                provider="test",
                model="sdk-test-v1",
                structured_output=False,
                native_stream=True,
                cancellation=True,
                attachment_kinds=attachment_kinds,
                attachment_mime_types=attachment_mime_types,
                max_attachment_count=16 if self.attachment_support else 0,
                max_attachment_bytes=(
                    MAX_ATTACHMENTS_TOTAL_BYTES if self.attachment_support else 0
                ),
            ),
            context_window_tokens=65_536,
            maximum_output_tokens=2_048,
        ).validated()

    def plan_stream(
        self,
        request: Any,
        **_kwargs: Any,
    ) -> Iterator[ProviderStreamEvent | HarnessModelResponse]:
        with self._lock:
            self.requests.append(request)
        yield ProviderStreamEvent(type="message.start", payload={})
        yield ProviderStreamEvent(type="message.delta", delta="do")
        yield ProviderStreamEvent(type="message.delta", delta="ne")
        yield ProviderStreamEvent(
            type="message.end",
            payload={"chars": 4},
        )
        yield HarnessModelResponse(
            kind="final",
            output={"message": "done"},
            usage={"total_tokens": 7},
        )

    def plan(self, _request: Any, **_kwargs: Any) -> HarnessModelResponse:
        return HarnessModelResponse(
            kind="final",
            output={"message": "done"},
            usage={"total_tokens": 7},
        )

    def classify_error(self, _error: BaseException) -> ProviderFailure:
        return ProviderFailure(
            kind=ProviderErrorKind.UNKNOWN,
            retryable=False,
            safe_code="sdk_test_error",
        )


class _BlockingModel(_StreamingFinalModel):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.finished = Event()

    def plan(
        self,
        _request: Any,
        *,
        cancellation_token: CancellationToken,
        **_kwargs: Any,
    ) -> HarnessModelResponse:
        self.started.set()
        try:
            while not cancellation_token.wait(0.01):
                pass
            cancellation_token.raise_if_cancelled()
            raise AssertionError("cancelled token must raise")
        finally:
            self.finished.set()

    # Hiding the inherited streaming implementation exercises the Runner's
    # synchronous cancellable-model seam.
    plan_stream = None  # type: ignore[assignment]


class _DelayedCancellationModel(_BlockingModel):
    def __init__(self) -> None:
        super().__init__()
        self.cancellation_seen = Event()
        self.release = Event()

    def plan(
        self,
        _request: Any,
        *,
        cancellation_token: CancellationToken,
        **_kwargs: Any,
    ) -> HarnessModelResponse:
        self.started.set()
        try:
            while not cancellation_token.wait(0.01):
                pass
            self.cancellation_seen.set()
            assert self.release.wait(5)
            cancellation_token.raise_if_cancelled()
            raise AssertionError("cancelled token must raise")
        finally:
            self.finished.set()


class _PermissionRecordingModel(_StreamingFinalModel):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()
        self.modes: list[str] = []

    def plan(self, request: Any, **_kwargs: Any) -> HarnessModelResponse:
        with self._lock:
            self.modes.append(str(request.context["permission_mode"]))
            first = len(self.modes) == 1
        if first:
            self.started.set()
            assert self.release.wait(5)
        return HarnessModelResponse(kind="final", output={"message": "done"})

    plan_stream = None  # type: ignore[assignment]


def _client(
    tmp_path: Path,
    *,
    model: Any | None = None,
) -> tuple[HarnessClient, Path]:
    tmp_path.chmod(0o700)
    workspace = tmp_path / "workspace"
    active = workspace / "active"
    active.mkdir(parents=True)
    key = tmp_path / "key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    client = HarnessClient(
        HarnessClientOptions(
            workspace=workspace,
            active_directory=active,
            state_home=tmp_path / "state",
            api_key_file=key,
            model=DEFAULT_MODEL,
            subagents_enabled=False,
            project_extensions_enabled=False,
        )
    )
    client._runner.model = model or _StreamingFinalModel()  # noqa: SLF001
    return client, active


def _async_client(
    tmp_path: Path,
    *,
    model: Any | None = None,
) -> AsyncHarnessClient:
    tmp_path.chmod(0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key = tmp_path / "key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    client = AsyncHarnessClient(
        HarnessClientOptions(
            workspace=workspace,
            state_home=tmp_path / "state",
            api_key_file=key,
            model=DEFAULT_MODEL,
            subagents_enabled=False,
            project_extensions_enabled=False,
        )
    )
    client._client._runner.model = model or _StreamingFinalModel()  # noqa: SLF001
    return client


def test_options_results_and_events_are_immutable(tmp_path: Path) -> None:
    options = HarnessThreadOptions(title="  SDK   test  ")
    assert options.title == "SDK test"
    with pytest.raises(FrozenInstanceError):
        options.title = "changed"  # type: ignore[misc]

    client, _active = _client(tmp_path)
    thread = client.start_thread(options)
    seen: list[HarnessEvent] = []
    result = thread.run("answer", HarnessRunOptions(on_event=seen.append))

    assert result.schema == SDK_RUN_RESULT_SCHEMA
    assert result.final_response == result.message == "done"
    assert isinstance(result.usage, MappingProxyType)
    with pytest.raises(TypeError):
        result.usage["total_tokens"] = 9  # type: ignore[index]
    delta = next(event for event in seen if event.type == "message.delta")
    assert isinstance(delta.payload, MappingProxyType)
    with pytest.raises(TypeError):
        delta.payload["delta"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        delta.type = "run.failed"  # type: ignore[misc]


def test_callback_failure_is_observer_only_and_reentrant_operation_is_rejected(
    tmp_path: Path,
) -> None:
    client, _active = _client(tmp_path)
    thread = client.start_thread()
    nested_errors: list[str] = []

    def observer(_event: HarnessEvent) -> None:
        try:
            client.start_thread()
        except RuntimeError as exc:
            nested_errors.append(str(exc))
        raise RuntimeError("observer failed")

    result = thread.run("answer", HarnessRunOptions(on_event=observer))

    assert result.status == "completed"
    assert nested_errors
    assert all("cannot be nested" in message for message in nested_errors)


def test_start_resume_and_fork_always_apply_explicit_permission_mode(
    tmp_path: Path,
) -> None:
    client, _active = _client(tmp_path)
    source = client.start_thread(
        HarnessThreadOptions(permission_mode="workspace-write")
    )
    source.run("first")

    forked = source.fork()
    resumed = client.resume_thread(source.session_id)

    assert forked.permission_mode == "read-only"
    assert resumed.permission_mode == "read-only"
    assert client._runner.store.load(forked.session_id)["permission_mode"] == "read-only"  # noqa: SLF001
    assert client._runner.store.load(source.session_id)["permission_mode"] == "read-only"  # noqa: SLF001
    assert [
        message["content"]
        for message in client._runner.store.load(forked.session_id)["messages"]  # noqa: SLF001
    ] == ["first", "done"]


def test_client_scope_lock_prevents_concurrent_permission_cross_authorization(
    tmp_path: Path,
) -> None:
    model = _PermissionRecordingModel()
    client, _active = _client(tmp_path, model=model)
    writable = client.start_thread(
        HarnessThreadOptions(permission_mode="workspace-write")
    )
    readonly = client.start_thread(HarnessThreadOptions(permission_mode="read-only"))
    outcomes: list[str] = []

    first = Thread(target=lambda: outcomes.append(writable.run("one").status))
    second = Thread(target=lambda: outcomes.append(readonly.run("two").status))
    first.start()
    assert model.started.wait(5)
    second.start()
    time.sleep(0.05)

    # The second handle cannot mutate the shared Runner while the first turn
    # owns the client-wide authority boundary.
    assert client._runner.permission_mode == "workspace-write"  # noqa: SLF001
    model.release.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert outcomes == ["completed", "completed"]
    assert model.modes == ["workspace-write", "read-only"]


def test_sync_event_stream_is_lazy_ordered_and_has_terminal_result(
    tmp_path: Path,
) -> None:
    client, _active = _client(tmp_path)
    thread = client.start_thread()
    stream = thread.run_stream("stream")
    assert client._runner.store.load(thread.session_id)["messages"] == []  # noqa: SLF001

    events = list(stream)
    result = stream.result()

    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert "".join(
        str(event.payload["delta"])
        for event in events
        if event.type == "message.delta"
    ) == "done"
    assert result.status == "completed"
    assert result.final_response == "done"
    assert stream.done is True


def test_stream_callback_cannot_reenter_blocking_consumer_apis(
    tmp_path: Path,
) -> None:
    client, _active = _client(tmp_path)
    thread = client.start_thread()
    failures: list[str] = []
    stream: Any = None

    def observer(_event: HarnessEvent) -> None:
        if failures:
            return
        for operation in (stream.result, lambda: next(stream), stream.close):
            try:
                operation()
            except RuntimeError as exc:
                failures.append(str(exc))

    stream = thread.run_stream("stream", HarnessRunOptions(on_event=observer))
    events = list(stream)

    assert events
    assert len(failures) == 3
    assert all("worker callback" in message for message in failures)
    assert stream.result().status == "completed"


def test_stream_rejects_concurrent_blocking_consumers_without_losing_settlement(
    tmp_path: Path,
) -> None:
    model = _BlockingModel()
    client, _active = _client(tmp_path, model=model)
    stream = client.start_thread().run_stream("wait")
    outcome: list[HarnessRunResult | BaseException] = []

    def consume_result() -> None:
        try:
            outcome.append(stream.result())
        except BaseException as exc:
            outcome.append(exc)

    consumer = Thread(target=consume_result)
    consumer.start()
    assert model.started.wait(5)

    with pytest.raises(RuntimeError, match="consumed concurrently"):
        stream.result()

    stream.close()
    consumer.join(5)
    assert not consumer.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], HarnessRunResult)
    assert outcome[0].status == "cancelled"


def test_stream_close_cancels_and_joins_the_turn_worker(tmp_path: Path) -> None:
    model = _BlockingModel()
    client, _active = _client(tmp_path, model=model)
    thread = client.start_thread()
    stream = thread.run_stream("wait")
    iterator = iter(stream)

    # Starting result consumption in a helper lets the model reach its
    # cooperative cancellation boundary without abandoning the SDK worker.
    consumed: list[BaseException | str] = []

    def consume() -> None:
        try:
            next(iterator)
            consumed.append("event")
        except (StopIteration, BaseException) as exc:
            consumed.append(exc)

    consumer = Thread(target=consume)
    consumer.start()
    assert model.started.wait(5)
    stream.close()
    consumer.join(5)

    assert model.finished.is_set()
    assert stream.done is True
    assert not consumer.is_alive()
    assert stream.result().status == "cancelled"


def test_concurrent_stream_close_waiters_all_observe_worker_settlement(
    tmp_path: Path,
) -> None:
    model = _DelayedCancellationModel()
    client, _active = _client(tmp_path, model=model)
    stream = client.start_thread().run_stream("wait")
    consumer = Thread(target=lambda: stream.result())
    consumer.start()
    assert model.started.wait(5)

    first = Thread(target=stream.close)
    second = Thread(target=stream.close)
    first.start()
    assert model.cancellation_seen.wait(5)
    second.start()
    time.sleep(0.05)
    assert first.is_alive()
    assert second.is_alive()

    model.release.set()
    first.join(5)
    second.join(5)
    consumer.join(5)
    assert not first.is_alive() and not second.is_alive() and not consumer.is_alive()
    assert model.finished.is_set()


def test_cancellation_callbacks_cannot_reenter_the_active_stream_closer(
    tmp_path: Path,
) -> None:
    client, _active = _client(tmp_path)
    token = CancellationToken()
    stream = client.start_thread().run_stream(
        "unused",
        HarnessRunOptions(cancellation_token=token),
    )
    callback_finished = Event()
    token.add_callback(lambda: (stream.close(), callback_finished.set()))

    closer = Thread(target=stream.close, daemon=True)
    closer.start()
    closer.join(2)

    assert not closer.is_alive()
    assert callback_finished.is_set()
    with pytest.raises(RuntimeError, match="closed before it started"):
        stream.result()


def test_close_callback_result_cannot_wait_on_its_own_close_settlement(
    tmp_path: Path,
) -> None:
    model = _DelayedCancellationModel()
    client, _active = _client(tmp_path, model=model)
    token = CancellationToken()
    stream = client.start_thread().run_stream(
        "wait",
        HarnessRunOptions(cancellation_token=token),
    )
    consumer = Thread(target=lambda: next(stream))
    consumer.start()
    assert model.started.wait(5)
    callback_errors: list[str] = []

    def inspect_result() -> None:
        try:
            stream.result()
        except RuntimeError as exc:
            callback_errors.append(str(exc))

    token.add_callback(inspect_result)
    closer = Thread(target=stream.close, daemon=True)
    closer.start()
    assert model.cancellation_seen.wait(5)
    model.release.set()
    closer.join(5)
    consumer.join(5)

    assert not closer.is_alive() and not consumer.is_alive()
    assert callback_errors == [
        "an event stream result cannot wait from its close callback"
    ]


def test_fatal_cancellation_callback_is_reraised_only_after_stream_worker_settles(
    tmp_path: Path,
) -> None:
    model = _DelayedCancellationModel()
    client, _active = _client(tmp_path, model=model)
    token = CancellationToken()
    cleanup_after_failure = Event()

    def fail_callback() -> None:
        raise _FatalCancellationCallback("fatal cancellation callback")

    token.add_callback(fail_callback)
    token.add_callback(cleanup_after_failure.set)
    stream = client.start_thread().run_stream(
        "wait",
        HarnessRunOptions(cancellation_token=token),
    )
    consumer = Thread(target=lambda: next(stream))
    consumer.start()
    assert model.started.wait(5)
    failures: list[BaseException] = []

    def close_stream() -> None:
        try:
            stream.close()
        except BaseException as exc:
            failures.append(exc)

    closer = Thread(target=close_stream, daemon=True)
    closer.start()
    assert model.cancellation_seen.wait(5)
    assert cleanup_after_failure.is_set()
    assert closer.is_alive()
    model.release.set()
    closer.join(5)
    consumer.join(5)

    assert not closer.is_alive() and not consumer.is_alive()
    assert stream.done is True
    assert len(failures) == 1
    assert isinstance(failures[0], _FatalCancellationCallback)
    # A later closer returns only after the first closer published settlement.
    stream.close()


def test_attachment_paths_are_relative_to_active_directory_and_persist_snapshot(
    tmp_path: Path,
) -> None:
    model = _StreamingFinalModel()
    client, active = _client(tmp_path, model=model)
    source = active / "notes.md"
    source.write_text("PRIVATE_SDK_ATTACHMENT", encoding="utf-8")
    (client.workspace / source.name).write_text("WRONG_ROOT", encoding="utf-8")
    thread = client.start_thread()

    result = thread.run(
        "inspect",
        HarnessRunOptions(attachments=(source.name,)),
    )
    source.write_text("MUTATED", encoding="utf-8")

    assert result.status == "completed"
    manifest = model.requests[0].context["messages"][-1]["attachments"]
    descriptor = manifest[0]
    assert descriptor["display_name"] == source.name
    stored = client._runner.store.load(thread.session_id)  # noqa: SLF001
    assert stored["messages"][0]["attachments"] == manifest
    imported = client._runner.attachment_store.read(  # noqa: SLF001
        client._runner._normalize_turn_attachments(manifest)[0]  # noqa: SLF001
    )
    assert imported == b"PRIVATE_SDK_ATTACHMENT"


def test_preflight_attachment_failure_discards_only_unreferenced_import(
    tmp_path: Path,
) -> None:
    client, active = _client(tmp_path)
    pdf = active / "unsupported.pdf"
    pdf.write_bytes(b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    thread = client.start_thread()

    with pytest.raises(AttachmentError, match="does not support PDF"):
        thread.run("inspect", HarnessRunOptions(attachments=(pdf.name,)))

    assert list(client._runner.attachment_store.root.glob("*.blob")) == []  # noqa: SLF001
    stored = client._runner.store.load(thread.session_id)  # noqa: SLF001
    assert stored["messages"] == []
    assert stored["runs"] == []


def test_uncertain_turn_failure_preserves_durably_referenced_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, active = _client(tmp_path)
    source = active / "uncertain.md"
    source.write_text("PRESERVE_ME", encoding="utf-8")
    thread = client.start_thread()

    def uncertain_run(
        session_id: str,
        prompt: str,
        **kwargs: Any,
    ) -> TurnOutcome:
        client._runner.store.begin_run(  # noqa: SLF001
            session_id,
            run_id="run_11111111111111111111111111111111",
            turn_id="turn_11111111111111111111111111111111",
            user_content=prompt,
            provider="test",
            model="sdk-test-v1",
            permission_mode="read-only",
            attachments=kwargs["attachments"],
        )
        raise RuntimeError("uncertain provider boundary")

    monkeypatch.setattr(client._runner, "run_turn", uncertain_run)  # noqa: SLF001

    with pytest.raises(RuntimeError, match="uncertain provider boundary"):
        thread.run("inspect", HarnessRunOptions(attachments=(source.name,)))

    stored = client._runner.store.load(thread.session_id)  # noqa: SLF001
    assert stored["messages"][0]["attachments"]
    assert stored["runs"][0]["status"] == "running"
    assert len(list(client._runner.attachment_store.root.glob("*.blob"))) == 1  # noqa: SLF001


def test_approval_broker_is_never_invented_and_only_explicit_value_is_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _active = _client(tmp_path)
    thread = client.start_thread(
        HarnessThreadOptions(permission_mode="workspace-write")
    )
    captured: list[Any] = []

    class Broker:
        def decide(
            self,
            request: ApprovalRequest,
            *,
            preview: str = "",
        ) -> ApprovalDecision:
            del preview
            return ApprovalDecision.for_request(
                request,
                verdict="deny",
                reason_code="sdk_test_deny",
            )

    broker = Broker()

    def fake_run_turn(
        session_id: str,
        _prompt: str,
        **kwargs: Any,
    ) -> TurnOutcome:
        captured.append(kwargs.get("approval_broker"))
        ordinal = len(captured)
        return TurnOutcome(
            session_id=session_id,
            run_id=f"run_{ordinal:032x}",
            turn_id=f"turn_{ordinal:032x}",
            status="completed",
            message="done",
            reason="final_output",
            usage={},
            result={},
        )

    monkeypatch.setattr(client._runner, "run_turn", fake_run_turn)  # noqa: SLF001

    thread.run("default")
    thread.run("explicit", HarnessRunOptions(approval_broker=broker))

    assert captured == [None, broker]


def test_runner_outcome_must_match_the_invoked_thread_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _active = _client(tmp_path)
    thread = client.start_thread()

    def mismatched_run(*_args: Any, **_kwargs: Any) -> TurnOutcome:
        return TurnOutcome(
            session_id="session_" + "f" * 32,
            run_id="run_" + "1" * 32,
            turn_id="turn_" + "2" * 32,
            status="completed",
            message="wrong identity",
            reason="final_output",
            usage={},
            result={},
        )

    monkeypatch.setattr(client._runner, "run_turn", mismatched_run)  # noqa: SLF001

    with pytest.raises(HarnessSdkContractError, match="session identity"):
        thread.run("must fail closed")


def test_runner_normalizes_deadline_to_public_failed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _active = _client(tmp_path)
    thread = client.start_thread()

    def deadline_runtime(
        *_args: Any,
        run_id: str,
        turn_id: str,
        event_sink: Any,
        journal: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        emitter = HarnessEventEmitter(
            run_id=run_id,
            turn_id=turn_id,
            durable_sink=journal.append,
            event_sink=event_sink,
        )
        emitter.emit("run.started", {"resumed": False})
        emitter.emit(
            "run.failed",
            {
                "error_code": "deadline_exceeded",
                "reason_code": "deadline_exceeded",
            },
        )
        return {
            "schema": "agent_harness.run.v1",
            "run_id": run_id,
            "turn_id": turn_id,
            "status": "deadline_exceeded",
            "output": None,
            "reason": "deadline_exceeded",
            "events": emitter.events,
            "checkpoint": None,
        }

    monkeypatch.setattr(runner_module, "run_agent_harness", deadline_runtime)

    result = thread.run("reach the deadline")

    assert result.status == "failed"
    assert result.reason == "deadline_exceeded"
    stored = client._runner.store.load(thread.session_id)  # noqa: SLF001
    assert stored["runs"][-1]["status"] == "failed"
    assert stored["runs"][-1]["requires_reconciliation"] is False


def test_async_client_run_and_stream_share_the_typed_contract(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = _async_client(tmp_path)
        thread = await client.start_thread()
        result = await thread.run("async")
        assert result.status == "completed"

        stream = thread.run_stream("stream")
        events = [event async for event in stream]
        streamed_result = await stream.result()
        assert any(event.type == "message.delta" for event in events)
        assert streamed_result.final_response == "done"

        forked = await thread.fork()
        resumed = await client.resume_thread(thread.session_id)
        assert forked.permission_mode == resumed.permission_mode == "read-only"

    asyncio.run(scenario())


def test_async_task_cancellation_waits_for_runner_worker_settlement(
    tmp_path: Path,
) -> None:
    model = _BlockingModel()

    async def scenario() -> None:
        client = _async_client(tmp_path, model=model)
        thread = await client.start_thread()
        task = asyncio.create_task(thread.run("wait"))
        assert await asyncio.to_thread(model.started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model.finished.is_set()
        stored = client._client._runner.store.load(thread.session_id)  # noqa: SLF001
        assert stored["runs"][-1]["status"] == "cancelled"
        assert stored["runs"][-1]["requires_reconciliation"] is False

    asyncio.run(scenario())


def test_async_task_reraises_fatal_cancel_callback_after_worker_settlement(
    tmp_path: Path,
) -> None:
    model = _DelayedCancellationModel()

    async def scenario() -> None:
        client = _async_client(tmp_path, model=model)
        thread = await client.start_thread()
        token = CancellationToken()
        cleanup_after_failure = Event()

        def fail_callback() -> None:
            raise _FatalCancellationCallback("fatal async cancellation callback")

        token.add_callback(fail_callback)
        token.add_callback(cleanup_after_failure.set)
        task = asyncio.create_task(
            thread.run(
                "wait",
                HarnessRunOptions(cancellation_token=token),
            )
        )
        assert await asyncio.to_thread(model.started.wait, 5)
        task.cancel()
        assert await asyncio.to_thread(model.cancellation_seen.wait, 5)
        assert cleanup_after_failure.is_set()
        await asyncio.sleep(0.05)
        assert not task.done()
        model.release.set()
        with pytest.raises(
            _FatalCancellationCallback,
            match="fatal async cancellation callback",
        ):
            await task
        assert model.finished.is_set()

    asyncio.run(scenario())


def test_async_stream_rejects_concurrent_result_consumers_and_close_waiters_settle(
    tmp_path: Path,
) -> None:
    model = _DelayedCancellationModel()

    async def scenario() -> None:
        client = _async_client(tmp_path, model=model)
        thread = await client.start_thread()
        stream = thread.run_stream("wait")
        result_task = asyncio.create_task(stream.result())
        assert await asyncio.to_thread(model.started.wait, 5)

        with pytest.raises(RuntimeError, match="consumed concurrently"):
            await stream.result()

        first_close = asyncio.create_task(stream.aclose())
        assert await asyncio.to_thread(model.cancellation_seen.wait, 5)
        second_close = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0.05)
        assert not first_close.done()
        assert not second_close.done()

        model.release.set()
        await asyncio.gather(first_close, second_close)
        result = await result_task
        assert result.status == "cancelled"
        assert model.finished.is_set()

    asyncio.run(scenario())


def test_async_lifecycle_cancellation_waits_for_durable_worker_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()
    settled = Event()

    async def scenario() -> None:
        client = _async_client(tmp_path)
        original = client._client.start_thread  # noqa: SLF001

        def blocked_start(options: HarnessThreadOptions | None = None) -> Any:
            started.set()
            assert release.wait(5)
            try:
                return original(options)
            finally:
                settled.set()

        monkeypatch.setattr(client._client, "start_thread", blocked_start)  # noqa: SLF001
        task = asyncio.create_task(client.start_thread())
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert settled.is_set()
        assert len(client._client._runner.store.list()) == 1  # noqa: SLF001

    asyncio.run(scenario())


def test_async_thread_fork_cancellation_waits_for_durable_worker_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()
    settled = Event()

    async def scenario() -> None:
        client = _async_client(tmp_path)
        source = await client.start_thread()
        sync_type = type(source._thread)  # noqa: SLF001
        original = sync_type.fork

        def blocked_fork(
            sync_thread: Any,
            *,
            permission_mode: str = "read-only",
        ) -> Any:
            started.set()
            assert release.wait(5)
            try:
                return original(sync_thread, permission_mode=permission_mode)
            finally:
                settled.set()

        monkeypatch.setattr(sync_type, "fork", blocked_fork)
        task = asyncio.create_task(source.fork())
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert settled.is_set()
        assert len(client._client._runner.store.list()) == 2  # noqa: SLF001

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HarnessThreadOptions(permission_mode="invalid"),
        lambda: HarnessRunOptions(attachments=(b"bytes-path",)),
        lambda: HarnessRunOptions(cancellation_token=object()),
        lambda: HarnessClientOptions(workspace=".", deadline_seconds=True),
        lambda: HarnessClientOptions(workspace=".", deadline_seconds=float("inf")),
        lambda: HarnessClientOptions(workspace=".", deadline_seconds=0.01),
        lambda: HarnessClientOptions(workspace=".", max_steps=True),
        lambda: HarnessClientOptions(workspace=".", max_steps=65),
    ],
)
def test_public_options_fail_closed_on_invalid_values(factory: Any) -> None:
    with pytest.raises(HarnessSdkContractError):
        factory()
