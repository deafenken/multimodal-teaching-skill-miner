from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import zlib

import pytest

import agent_harness.cli as cli_module
import agent_harness.tui as tui_module
from agent_harness.attachments import (
    MAX_ATTACHMENTS_TOTAL_BYTES,
    AttachmentDescriptor,
    AttachmentError,
)
from agent_harness.cli import main as cli_main
from agent_harness.core import (
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
)
from agent_harness.providers.deepseek_client import DEFAULT_MODEL
from agent_harness.runner import AgentRunner
from agent_harness.tui import HarnessTui


def _chunk(kind: bytes, content: bytes) -> bytes:
    checksum = zlib.crc32(kind + content) & 0xFFFFFFFF
    return (
        len(content).to_bytes(4, "big")
        + kind
        + content
        + checksum.to_bytes(4, "big")
    )


def _png() -> bytes:
    header = (1).to_bytes(4, "big") * 2 + bytes([8, 2, 0, 0, 0])
    pixels = zlib.compress(b"\x00\x00\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", pixels)
        + _chunk(b"IEND", b"")
    )


def _descriptor(
    body: bytes,
    *,
    ordinal: int,
    display_name: str = "notes.md",
) -> AttachmentDescriptor:
    return AttachmentDescriptor.from_value(
        {
            "schema": "agent_harness.attachment.v1",
            "attachment_id": f"att_{ordinal:032x}",
            "kind": "text",
            "media_type": "text/markdown; charset=utf-8",
            "display_name": display_name,
            "size_bytes": len(body),
            "sha256": sha256(body).hexdigest(),
            "estimated_tokens": len(body),
        }
    )


class _FinalAttachmentModel:
    def __init__(self, *, max_count: int = 16) -> None:
        self.max_count = max_count
        self.requests: list[Any] = []

    @property
    def model_spec(self) -> ProviderModelSpec:
        return ProviderModelSpec(
            provider="test",
            model="attachment-test-v1",
            capabilities=ProviderCapabilities(
                provider="test",
                model="attachment-test-v1",
                structured_output=False,
                native_stream=False,
                attachment_kinds=("text",),
                attachment_mime_types=(
                    "text/plain; charset=utf-8",
                    "text/markdown; charset=utf-8",
                ),
                max_attachment_count=self.max_count,
                max_attachment_bytes=MAX_ATTACHMENTS_TOTAL_BYTES,
            ),
            context_window_tokens=65_536,
            maximum_output_tokens=2_048,
        ).validated()

    def plan(self, request: Any, **_kwargs: Any) -> HarnessModelResponse:
        self.requests.append(request)
        return HarnessModelResponse(kind="final", output={"message": "done"})

    def classify_error(self, _error: BaseException) -> ProviderFailure:
        return ProviderFailure(
            kind=ProviderErrorKind.UNKNOWN,
            retryable=False,
            safe_code="attachment_test_error",
        )


def _runner(
    tmp_path: Path,
    *,
    model: _FinalAttachmentModel | None = None,
) -> tuple[AgentRunner, Path]:
    tmp_path.chmod(0o700)
    workspace = tmp_path / "workspace"
    active = workspace / "active"
    active.mkdir(parents=True)
    key = tmp_path / "key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    runner = AgentRunner(
        workspace,
        active_directory=active,
        state_home=tmp_path / "state",
        api_key_file=key,
        model=DEFAULT_MODEL,
        subagents_enabled=False,
        project_extensions_enabled=False,
    )
    if model is not None:
        runner.model = model  # type: ignore[assignment]
    return runner, active


def test_cli_relative_repeated_attachments_forward_immutable_manifests_without_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = _FinalAttachmentModel()
    runner, active = _runner(tmp_path, model=model)
    first_body = b"PRIVATE_ATTACHMENT_BODY_ONE"
    second_body = b"PRIVATE_ATTACHMENT_BODY_TWO"
    first = active / "first.md"
    second = active / "second.txt"
    first.write_bytes(first_body)
    second.write_bytes(second_body)
    # A same-named workspace-root file proves resolution uses active_directory.
    (runner.workspace / first.name).write_bytes(b"WRONG_ROOT_FILE")
    monkeypatch.setattr(cli_module, "_runner", lambda _args: runner)

    code = cli_main(
        [
            "--cwd",
            os.fspath(runner.workspace),
            "exec",
            "inspect",
            "--attach",
            first.name,
            "--attach",
            second.name,
            "--jsonl",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert len(model.requests) == 1
    manifest = model.requests[0].context["messages"][-1]["attachments"]
    assert [item["display_name"] for item in manifest] == [first.name, second.name]
    descriptors = [AttachmentDescriptor.from_value(item) for item in manifest]
    assert [runner.attachment_store.read(item) for item in descriptors] == [
        first_body,
        second_body,
    ]
    persisted = runner.store.list()[0]
    status = runner.context_status(str(persisted["session_id"]))
    assert status["active_attachment_count"] == 2
    assert status["active_attachment_bytes"] == len(first_body) + len(second_body)

    first.write_bytes(b"MUTATED_SOURCE")
    assert runner.attachment_store.read(descriptors[0]) == first_body
    serialized_status = json.dumps(status, sort_keys=True)
    for private_value in (
        os.fspath(first),
        os.fspath(second),
        first_body.decode(),
        second_body.decode(),
        "WRONG_ROOT_FILE",
        "base64",
    ):
        assert private_value not in output
        assert private_value not in serialized_status


@pytest.mark.parametrize(
    ("name", "body", "kind"),
    [
        ("source.pdf", b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n", "pdf"),
        ("screen.bin", _png(), "image"),
    ],
)
def test_unsupported_attachment_rejects_before_run_or_provider_and_keeps_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    body: bytes,
    kind: str,
) -> None:
    runner, active = _runner(tmp_path)
    source = active / name
    source.write_bytes(body)
    descriptor = runner.ingest_attachments([name])[0]
    assert descriptor.kind == kind
    session = runner.new_session()
    before = runner.store.load(str(session["session_id"]))

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("preflight must not begin a run or call the provider")

    monkeypatch.setattr(runner.store, "begin_run", forbidden)
    monkeypatch.setattr(runner.client, "chat_json_stream", forbidden)
    monkeypatch.setattr(runner.client, "chat_text_stream", forbidden)

    with pytest.raises(AttachmentError, match="does not support"):
        runner.run_turn(
            str(session["session_id"]),
            "inspect",
            attachments=(descriptor,),
        )

    after = runner.store.load(str(session["session_id"]))
    assert after["messages"] == before["messages"] == []
    assert after["runs"] == before["runs"] == []


def test_cli_unsupported_pdf_cleans_only_the_unreferenced_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner, active = _runner(tmp_path)
    source = active / "private-source.pdf"
    body = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
    source.write_bytes(body)
    monkeypatch.setattr(cli_module, "_runner", lambda _args: runner)

    code = cli_main(
        [
            "--cwd",
            os.fspath(runner.workspace),
            "exec",
            "inspect",
            "--attach",
            source.name,
        ]
    )

    captured = capsys.readouterr()
    assert code == 78
    assert captured.out == ""
    assert "does not support PDF" in captured.err
    assert os.fspath(source) not in captured.err
    assert body.decode() not in captured.err
    assert list(runner.attachment_store.root.glob("*.blob")) == []
    persisted = runner.store.list()[0]
    session = runner.store.load(str(persisted["session_id"]))
    assert session["messages"] == []
    assert session["runs"] == []


def test_active_attachment_count_and_provider_cap_are_enforced_before_begin_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FinalAttachmentModel(max_count=1)
    runner, active = _runner(tmp_path, model=model)
    (active / "first.md").write_text("first", encoding="utf-8")
    (active / "second.md").write_text("second", encoding="utf-8")
    first, second = runner.ingest_attachments(["first.md", "second.md"])
    session = runner.new_session()

    runner.run_turn(str(session["session_id"]), "first", attachments=(first,))
    status = runner.context_status(str(session["session_id"]))
    assert status["active_attachment_count"] == 1
    assert status["active_attachment_bytes"] == first.size_bytes
    before = runner.store.load(str(session["session_id"]))
    request_count = len(model.requests)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("attachment cap must fail before begin_run")

    monkeypatch.setattr(runner.store, "begin_run", forbidden)
    with pytest.raises(AttachmentError, match="count limit"):
        runner.run_turn(
            str(session["session_id"]),
            "second",
            attachments=(second,),
        )

    after = runner.store.load(str(session["session_id"]))
    assert after["messages"] == before["messages"]
    assert after["runs"] == before["runs"]
    assert len(model.requests) == request_count


def test_tui_relative_attach_uses_runner_active_directory(tmp_path: Path) -> None:
    runner, active = _runner(tmp_path, model=_FinalAttachmentModel())
    body = b"TUI_RELATIVE_ATTACHMENT"
    (active / "relative.md").write_bytes(body)
    (runner.workspace / "relative.md").write_bytes(b"WRONG_ROOT_FILE")
    session = runner.new_session()
    application = HarnessTui(runner, session)

    application.command("/attach relative.md")

    assert len(application.pending_attachments) == 1
    descriptor = application.pending_attachments[0]
    assert descriptor.display_name == "relative.md"
    assert runner.attachment_store.read(descriptor) == body


def test_tui_busy_followups_freeze_attachment_ownership_per_prompt() -> None:
    application = HarnessTui(
        SimpleNamespace(),  # type: ignore[arg-type]
        {"session_id": "session_12345678", "messages": []},
    )
    active = _descriptor(b"active", ordinal=1, display_name="active.md")
    first = _descriptor(b"first", ordinal=2, display_name="first.md")
    second = _descriptor(b"second", ordinal=3, display_name="second.md")
    application.active_turn = tui_module._PendingTurn("active", (active,))
    # A non-None worker is the settlement authority even if the thread object
    # itself has not started; this avoids introducing timing into the test.
    application.worker = SimpleNamespace()  # type: ignore[assignment]

    application.pending_attachments.append(first)
    application.input_buffer = "queued first"
    application.submit()
    application.pending_attachments.append(second)
    application.input_buffer = "queued second"
    application.submit()

    assert application.active_turn.attachments == (active,)
    assert [(item.prompt, item.attachments) for item in application.followups] == [
        ("queued first", (first,)),
        ("queued second", (second,)),
    ]
    assert application.pending_attachments == []


@pytest.mark.parametrize("persisted", [False, True])
def test_tui_failure_recovers_only_an_unpersisted_active_attachment(
    persisted: bool,
) -> None:
    descriptor = _descriptor(b"recover", ordinal=4, display_name="recover.md")
    message: dict[str, Any] = {"role": "user", "content": "failed"}
    if persisted:
        message["attachments"] = [descriptor.to_dict()]
    stored = {
        "session_id": "session_12345678",
        "permission_mode": "read-only",
        "messages": [message] if persisted else [],
    }
    runner = SimpleNamespace(store=SimpleNamespace(load=lambda _session_id: stored))
    application = HarnessTui(
        runner,  # type: ignore[arg-type]
        {"session_id": "session_12345678", "messages": []},
    )
    application.active_turn = tui_module._PendingTurn("failed", (descriptor,))
    application.worker = SimpleNamespace()  # type: ignore[assignment]
    application.queue.put(("error", RuntimeError("provider failed")))

    application.drain()

    assert application.pending_attachments == ([] if persisted else [descriptor])
    assert application.state.status == "failed"


def test_tui_failure_recovers_queued_attachment_without_cross_turn_drift() -> None:
    active = _descriptor(b"active", ordinal=6, display_name="active.md")
    queued = _descriptor(b"queued", ordinal=7, display_name="queued.md")
    stored = {
        "session_id": "session_12345678",
        "permission_mode": "read-only",
        "messages": [],
    }
    runner = SimpleNamespace(store=SimpleNamespace(load=lambda _session_id: stored))
    application = HarnessTui(
        runner,  # type: ignore[arg-type]
        {"session_id": "session_12345678", "messages": []},
    )
    application.active_turn = tui_module._PendingTurn("active", (active,))
    application.followups.append(tui_module._PendingTurn("queued", (queued,)))
    application.worker = SimpleNamespace()  # type: ignore[assignment]
    application.queue.put(("error", RuntimeError("provider failed")))

    application.drain()

    assert application.followups == []
    assert application.pending_attachments == [active, queued]
    assert any("已恢复到待发送区" in notice for notice in application.state.notices)


def test_tui_detach_is_exact_and_pending_attachments_block_session_switch() -> None:
    descriptor = _descriptor(b"pending", ordinal=5, display_name="pending.md")

    class _Runner:
        permission_mode = "read-only"

        def __init__(self) -> None:
            self.discarded: list[AttachmentDescriptor] = []
            self.new_calls = 0

        def discard_attachment(self, item: AttachmentDescriptor) -> None:
            self.discarded.append(item)

        def new_session(self) -> dict[str, Any]:
            self.new_calls += 1
            return {
                "session_id": "session_87654321",
                "permission_mode": "read-only",
                "messages": [],
            }

    runner = _Runner()
    application = HarnessTui(
        runner,  # type: ignore[arg-type]
        {"session_id": "session_12345678", "messages": []},
    )
    application.pending_attachments.append(descriptor)

    application.command("/new")
    assert runner.new_calls == 0
    assert application.state.session_id == "session_12345678"
    assert any("待发送附件" in notice for notice in application.state.notices)

    application.command(f"/detach {descriptor.attachment_id}")
    assert runner.discarded == [descriptor]
    assert application.pending_attachments == []
    application.command("/new")
    assert runner.new_calls == 1
    assert application.state.session_id == "session_87654321"
