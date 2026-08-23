from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from agent_harness.context import (
    ContextCompactionPlan,
    ContextCompactionResult,
    estimate_context_tokens,
    select_compaction_plan,
)
from agent_harness.core import (
    CancellationToken,
    HarnessCancelled,
    HarnessModelRequest,
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderFailure,
    ProviderModelSpec,
    ProviderErrorKind,
)
from agent_harness.cli import main as cli_main
from agent_harness.providers.deepseek import DeepSeekCodingModel
from agent_harness.runner import AgentRunner
from agent_harness.session import SessionStore, SessionStoreError
from agent_harness.tui import HarnessTui


def _store(tmp_path: Path) -> SessionStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SessionStore(workspace, state_home=tmp_path / "state")


def _session_with_turns(store: SessionStore, turns: int, *, width: int = 20) -> dict[str, Any]:
    session = store.create(
        provider="deepseek",
        model="deepseek-test",
        permission_mode="read-only",
    )
    for index in range(turns):
        store.append_message(
            session["session_id"],
            role="user",
            content=f"user-{index}-" + "u" * width,
        )
        store.append_message(
            session["session_id"],
            role="assistant",
            content=f"assistant-{index}-" + "a" * width,
        )
    return store.load(session["session_id"])


def test_message_ids_are_stable_unique_and_transcript_is_append_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    session = store.create(
        provider="deepseek",
        model="deepseek-test",
        permission_mode="read-only",
    )
    store.append_message(session["session_id"], role="user", content="same")
    stored = store.append_message(session["session_id"], role="user", content="same")

    ids = [item["message_id"] for item in stored["messages"]]
    assert len(set(ids)) == 2
    assert all(item["content_sha256"] for item in stored["messages"])
    assert [item["message_id"] for item in store.load(session["session_id"])["messages"]] == ids

    modified = deepcopy(stored)
    modified["messages"][0]["content"] = "changed"
    modified["messages"][0].pop("content_sha256")
    with pytest.raises(SessionStoreError, match="append-only"):
        store.save(modified)

    reordered = deepcopy(stored)
    reordered["messages"].reverse()
    with pytest.raises(SessionStoreError, match="append-only"):
        store.save(reordered)


def test_begin_and_finish_run_bind_messages_atomically(tmp_path: Path) -> None:
    store = _store(tmp_path)
    session = store.create(
        provider="deepseek",
        model="deepseek-test",
        permission_mode="read-only",
    )
    running = store.begin_run(
        session["session_id"],
        run_id="run_atomic_12345678",
        turn_id="turn_atomic_12345678",
        user_content="perform work",
        provider="deepseek",
        model="deepseek-test",
        permission_mode="read-only",
    )

    assert running["runs"][-1]["status"] == "running"
    assert running["messages"][-1]["run_id"] == "run_atomic_12345678"
    assert running["messages"][-1]["turn_id"] == "turn_atomic_12345678"

    completed = store.finish_run(
        session["session_id"],
        run_id="run_atomic_12345678",
        turn_id="turn_atomic_12345678",
        status="completed",
        assistant_content="finished",
        usage={"total_tokens": 7},
    )
    assert completed["runs"][-1]["status"] == "completed"
    assert completed["messages"][-1]["content"] == "finished"
    assert completed["messages"][-1]["run_id"] == "run_atomic_12345678"
    assert completed["messages"][-1]["turn_id"] == "turn_atomic_12345678"
    assert completed["usage"]["total_tokens"] == 7


def test_legacy_messages_receive_deterministic_ids(tmp_path: Path) -> None:
    store = _store(tmp_path)
    session = store.create(
        provider="deepseek",
        model="deepseek-test",
        permission_mode="read-only",
    )
    stored = store.append_message(session["session_id"], role="user", content="legacy")
    path = store.sessions_directory / f"{session['session_id']}.json"
    material = json.loads(path.read_text(encoding="utf-8"))
    material["messages"][0].pop("message_id")
    material["messages"][0].pop("content_sha256")
    path.write_text(json.dumps(material), encoding="utf-8")
    path.chmod(0o600)

    first = store.load(session["session_id"])
    second = store.load(session["session_id"])
    assert first["messages"][0]["message_id"] == second["messages"][0]["message_id"]
    assert first["messages"][0]["content"] == stored["messages"][0]["content"]


def test_compaction_lineage_preserves_raw_transcript_and_fork(tmp_path: Path) -> None:
    store = _store(tmp_path)
    session = _session_with_turns(store, 5)
    original = deepcopy(session["messages"])
    record = store.record_compaction(
        session["session_id"],
        summary="The user and agent completed the first two turns.",
        source_message_count=4,
        provider="deepseek",
        model="deepseek-test",
        trigger="manual",
        usage={"total_tokens": 12},
    )
    stored = store.load(session["session_id"])
    view = store.context_view(session["session_id"])

    assert stored["messages"] == original
    assert view["summary"] == record["summary"]
    assert view["compacted_message_count"] == 4
    assert view["messages"] == original[4:]
    assert view["lineage"]["summary_sha256"] == record["summary_sha256"]
    assert view["lineage"]["active_context_sha256"]

    changed = deepcopy(stored)
    changed["compactions"][0]["summary"] = "tampered"
    changed["compactions"][0]["summary_sha256"] = "0" * 64
    with pytest.raises(SessionStoreError, match="lineage|append-only"):
        store.save(changed)

    forked = store.fork(session["session_id"])
    assert forked["messages"] == original
    assert forked["compactions"] == stored["compactions"]
    after = store.append_message(forked["session_id"], role="user", content="new branch")
    assert after["messages"][-1]["message_id"] not in {
        item["message_id"] for item in original
    }


def test_compaction_plan_is_incremental_bounded_and_assistant_aligned(tmp_path: Path) -> None:
    store = _store(tmp_path)
    session = _session_with_turns(store, 8, width=100)
    plan = select_compaction_plan(session, retain_messages=4, max_source_bytes=1_500)

    assert plan is not None
    assert plan.source_messages[-1]["role"] == "assistant"
    assert plan.source_message_count <= len(session["messages"]) - 4
    assert len(json.dumps(plan.source_messages).encode("utf-8")) <= 1_500

    store.record_compaction(
        session["session_id"],
        summary="First incremental summary.",
        source_message_count=plan.source_message_count,
        provider="deepseek",
        model="deepseek-test",
        trigger="manual",
    )
    next_plan = select_compaction_plan(
        store.load(session["session_id"]),
        retain_messages=4,
        max_source_bytes=10_000,
    )
    assert next_plan is not None
    assert next_plan.parent_compaction_id
    assert next_plan.parent_summary == "First incremental summary."
    assert next_plan.source_message_count > plan.source_message_count


class _CompactionClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model="deepseek-test", max_tokens=4_096)
        self.requests: list[tuple[str, list[Mapping[str, str]]]] = []

    def chat_json_stream(self, messages: list[Mapping[str, str]], **kwargs: Any):
        self.requests.append((str(kwargs["request_kind"]), list(messages)))
        if kwargs["request_kind"] == "agent_harness_compaction":
            return {"summary": "bounded summary"}, {"response_id": "resp_compact", "usage": {"total_tokens": 9}}
        return {"action": "answer"}, {"response_id": "resp_plan", "usage": {}}

    def chat_text_stream(self, messages: list[Mapping[str, str]], **kwargs: Any):
        self.requests.append((str(kwargs["request_kind"]), list(messages)))
        yield {"type": "text_delta", "text": "done"}
        yield {"type": "completed", "trace": {"response_id": "resp_answer", "usage": {}}}


def test_provider_compaction_is_hidden_and_summary_is_user_data() -> None:
    client = _CompactionClient()
    model = DeepSeekCodingModel(client)  # type: ignore[arg-type]
    session = {
        "messages": [
            {
                "message_id": f"message_{index:032x}",
                "role": role,
                "content": content,
            }
            for index, (role, content) in enumerate(
                (("user", "question"), ("assistant", "answer")), start=1
            )
        ],
        "compactions": [],
    }
    real_plan = select_compaction_plan(session, retain_messages=0 + 2, max_source_bytes=4_096)
    assert real_plan is None
    real_plan = ContextCompactionPlan(
        parent_compaction_id=None,
        parent_summary="",
        parent_source_message_count=0,
        source_message_count=2,
        source_messages=tuple(session["messages"]),
    )
    result = model.compact_context(
        real_plan,
        cancellation_token=CancellationToken(),
        deadline_monotonic=10**9,
    )
    assert result.summary == "bounded summary"
    kind, compaction_messages = client.requests[0]
    assert kind == "agent_harness_compaction"
    assert "question" not in compaction_messages[0]["content"]
    assert "question" in compaction_messages[1]["content"]


def test_provider_planner_and_answer_receive_identical_summary_as_data() -> None:
    client = _CompactionClient()
    model = DeepSeekCodingModel(client)  # type: ignore[arg-type]
    digest = "a" * 64
    request = HarnessModelRequest(
        run_id="run_12345678",
        turn_id="turn_12345678",
        step=1,
        context={
            "messages": [{"role": "user", "content": "latest question"}],
            "history_summary": {
                "content": "PRIVATE_LOSSY_SUMMARY",
                "lineage": {
                    "compaction_id": "compaction_12345678",
                    "source_message_count": 4,
                    "source_messages_sha256": digest,
                    "summary_sha256": digest,
                    "active_context_sha256": digest,
                },
            },
        },
        observations=(),
        tools=(),
        state={},
    )

    response = model.plan(
        request,
        cancellation_token=CancellationToken(),
        deadline_monotonic=10**9,
    )

    assert response.kind == "final"
    planner_messages = next(
        messages for kind, messages in client.requests if kind == "agent_harness_planner"
    )
    answer_messages = next(
        messages for kind, messages in client.requests if kind == "agent_harness_answer"
    )
    planner_payload = json.loads(planner_messages[-1]["content"])
    answer_payload = json.loads(answer_messages[-1]["content"])
    assert planner_payload["history_summary"] == answer_payload["history_summary"]
    assert planner_payload["history_summary"]["content"] == "PRIVATE_LOSSY_SUMMARY"
    assert all(
        "PRIVATE_LOSSY_SUMMARY" not in item["content"]
        for item in planner_messages[:-1] + answer_messages[:-1]
    )


class _CompactableModel:
    def __init__(self, *, window: int = 2_048) -> None:
        self.compaction_calls = 0
        self.requests: list[Any] = []
        self.window = window

    @property
    def model_spec(self) -> ProviderModelSpec:
        return ProviderModelSpec(
            provider="test",
            model="test-compact-v1",
            capabilities=ProviderCapabilities(
                provider="test",
                model="test-compact-v1",
                structured_output=False,
                native_stream=False,
            ),
            context_window_tokens=self.window,
            maximum_output_tokens=256,
        ).validated()

    def compact_context(self, plan: Any, **_kwargs: Any) -> ContextCompactionResult:
        self.compaction_calls += 1
        return ContextCompactionResult(
            summary=f"summary through {plan.source_message_count}",
            usage={"total_tokens": 5},
            provider_request_id=f"compact-{self.compaction_calls}",
        )

    def plan(self, request: Any, **_kwargs: Any) -> HarnessModelResponse:
        self.requests.append(request)
        return HarnessModelResponse(kind="final", output={"message": "done"})

    def classify_error(self, _error: BaseException) -> ProviderFailure:
        return ProviderFailure(
            kind=ProviderErrorKind.UNKNOWN,
            retryable=False,
            safe_code="test_error",
        )


def _runner(tmp_path: Path, *, window: int = 2_048) -> tuple[AgentRunner, _CompactableModel]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key = tmp_path / "key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    tmp_path.chmod(0o700)
    runner = AgentRunner(
        workspace,
        state_home=tmp_path / "state",
        api_key_file=key,
    )
    model = _CompactableModel(window=window)
    runner.model = model  # type: ignore[assignment]
    return runner, model


def test_runner_manual_compaction_and_status_do_not_expose_summary(tmp_path: Path) -> None:
    runner, model = _runner(tmp_path, window=8_192)
    session = runner.new_session()
    for index in range(6):
        runner.store.append_message(
            session["session_id"], role="user", content=f"user {index}"
        )
        runner.store.append_message(
            session["session_id"], role="assistant", content=f"answer {index}"
        )
    before = runner.store.load(session["session_id"])["messages"]

    result = runner.compact_session(session["session_id"])
    status = runner.context_status(session["session_id"])

    assert result["status"] == "compacted"
    assert model.compaction_calls == 1
    assert runner.store.load(session["session_id"])["messages"] == before
    assert status["compacted_message_count"] == 6
    assert status["active_message_count"] == 6
    serialized = json.dumps(status)
    assert "summary through" not in serialized


def test_runner_auto_compacts_before_turn_and_projects_only_active_view(tmp_path: Path) -> None:
    runner, model = _runner(tmp_path, window=2_048)
    session = runner.new_session()
    for index in range(8):
        runner.store.append_message(
            session["session_id"],
            role="user",
            content=f"user {index} " + "u" * 120,
        )
        runner.store.append_message(
            session["session_id"],
            role="assistant",
            content=f"answer {index} " + "a" * 120,
        )
    original_count = len(runner.store.load(session["session_id"])["messages"])

    outcome = runner.run_turn(session["session_id"], "new question")

    assert outcome.status == "completed"
    assert model.compaction_calls >= 1
    assert model.requests
    context = model.requests[0].context
    assert context["history_summary"]["content"].startswith("summary through")
    assert len(context["messages"]) < original_count + 1
    assert context["messages"][-1] == {"role": "user", "content": "new question"}
    stored = runner.store.load(session["session_id"])
    assert len(stored["messages"]) == original_count + 2
    assert [item["content"] for item in stored["messages"][:2]] == [
        "user 0 " + "u" * 120,
        "answer 0 " + "a" * 120,
    ]


def test_cancelled_compaction_persists_no_summary(tmp_path: Path) -> None:
    runner, _model = _runner(tmp_path, window=8_192)
    session = runner.new_session()
    for index in range(5):
        runner.store.append_message(
            session["session_id"], role="user", content=f"user {index}"
        )
        runner.store.append_message(
            session["session_id"], role="assistant", content=f"answer {index}"
        )

    class CancellingModel(_CompactableModel):
        def compact_context(
            self,
            plan: Any,
            *,
            cancellation_token: CancellationToken,
            **_kwargs: Any,
        ) -> ContextCompactionResult:
            cancellation_token.cancel("test_cancel")
            return super().compact_context(plan)

    runner.model = CancellingModel(window=8_192)  # type: ignore[assignment]
    with pytest.raises(HarnessCancelled):
        runner.compact_session(session["session_id"])
    assert runner.store.load(session["session_id"])["compactions"] == []


def test_tui_compact_runs_in_background_and_context_uses_persisted_view(
    tmp_path: Path,
) -> None:
    runner, _model = _runner(tmp_path, window=8_192)
    session = runner.new_session()
    for index in range(6):
        runner.store.append_message(
            session["session_id"], role="user", content=f"user {index}"
        )
        runner.store.append_message(
            session["session_id"], role="assistant", content=f"answer {index}"
        )
    application = HarnessTui(runner, runner.store.load(session["session_id"]))

    application.command("/compact")
    assert application.worker_kind == "compaction"
    assert application.worker is not None
    application.worker.join(timeout=2)
    application.drain()
    assert application.busy is False
    assert any("上下文已压缩" in item for item in application.state.notices)

    application.command("/context")
    assert any("已压缩 6" in item for item in application.state.notices)
    assert all("summary through" not in item for item in application.state.notices)


def test_cli_context_json_contains_no_transcript_or_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner, _model = _runner(tmp_path, window=8_192)
    session = runner.new_session()
    runner.store.append_message(
        session["session_id"], role="user", content="PRIVATE_CONTEXT_MARKER"
    )
    args = [
        "--cwd",
        str(runner.workspace),
        "--state-home",
        str(tmp_path / "state"),
        "--api-key-file",
        str(tmp_path / "key"),
        "context",
        session["session_id"],
        "--json",
    ]

    assert cli_main(args) == 0
    output = capsys.readouterr().out
    assert "PRIVATE_CONTEXT_MARKER" not in output
    assert '"messages"' not in output
    assert '"summary"' not in output


def test_context_estimate_counts_summary_tools_instructions_and_prompt() -> None:
    baseline = estimate_context_tokens(
        summary="",
        messages=[{"role": "user", "content": "x"}],
    )
    expanded = estimate_context_tokens(
        summary="summary",
        messages=[{"role": "user", "content": "x"}],
        project_instruction_bytes=100,
        tool_definitions=[{"name": "tool", "description": "d" * 50}],
        prospective_prompt="next",
    )
    assert expanded > baseline + 100
