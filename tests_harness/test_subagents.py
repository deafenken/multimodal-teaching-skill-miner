from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from threading import Barrier, Event, Lock, Thread
import time
from typing import Any

import pytest

from agent_harness.core import (
    CancellationToken,
    HarnessCancelled,
    HarnessContractError,
    HarnessDeadlineExceeded,
    ToolExecutionContext,
    ToolExecutionError,
    ToolRegistry,
)
from agent_harness.core.schema import validate_schema
from agent_harness.subagents import (
    SUBAGENT_BATCH_SCHEMA,
    SubagentBudgetLedger,
    SubagentLimits,
    SubagentLineage,
    SubagentResult,
    SubagentScheduler,
    SubagentTask,
    build_subagent_registry,
    register_subagent_tool,
)


def _context(
    *,
    token: CancellationToken | None = None,
    effects: list[str] | None = None,
    progress: list[tuple[str, dict[str, Any]]] | None = None,
    call_id: str = "call-1",
) -> ToolExecutionContext:
    effect_sink = effects if effects is not None else []
    progress_sink = progress if progress is not None else []
    return ToolExecutionContext(
        run_id="run-1",
        turn_id="turn-1",
        call_id=call_id,
        tool_name="agent.delegate",
        cancellation_token=token or CancellationToken(),
        deadline_monotonic=time.monotonic() + 5.0,
        idempotency_key=None,
        emit_progress=lambda kind, payload=None: progress_sink.append((kind, dict(payload or {}))),
        begin_effect=lambda: effect_sink.append("effect"),
        session_id="session-1",
        trusted_data_scopes=frozenset({"internal", "workspace_write"}),
    )


def _result(task: SubagentTask, summary: str | None = None) -> SubagentResult:
    return SubagentResult(
        task_id=task.task_id,
        status="completed",
        summary=summary or f"finished-{task.task_id}",
    )


def _parent_lineage(*, depth: int = 1) -> SubagentLineage:
    return SubagentLineage(
        root_run_id="root-1",
        parent_session_id="session-0",
        parent_run_id="run-0",
        parent_turn_id="turn-0",
        parent_call_id="call-0",
        agent_id="agent-parent",
        depth=depth,
        ordinal=0,
        task_id="parent-task",
    )


def test_contracts_are_frozen_and_validate_bounded_values() -> None:
    limits = SubagentLimits()
    task = SubagentTask("task-1", "Inspect the repository.", "read-only")
    lineage = _parent_lineage()
    result = SubagentResult("task-1", "completed", "Done.", artifact_id="worktree-1")

    with pytest.raises(FrozenInstanceError):
        task.prompt = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        limits.max_depth = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        lineage.depth = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.summary = "mutated"  # type: ignore[misc]

    with pytest.raises(HarnessContractError, match="max_batch_size"):
        SubagentLimits(max_batch_size=5)
    with pytest.raises(HarnessContractError, match="permission_mode"):
        SubagentTask("task-1", "Do work.", "full-access")
    with pytest.raises(HarnessContractError, match="prompt"):
        SubagentTask("task-1", "\x00secret", "read-only")
    with pytest.raises(HarnessContractError, match="status"):
        SubagentResult("task-1", "running", "Still running.")
    with pytest.raises(HarnessContractError, match="error_code"):
        SubagentResult("task-1", "failed", "Failed.", error_code="Raw Secret")


def test_scheduler_runs_children_concurrently_but_returns_input_order() -> None:
    barrier = Barrier(2)
    state_lock = Lock()
    active = 0
    maximum_active = 0
    effects: list[str] = []
    progress: list[tuple[str, dict[str, Any]]] = []

    def execute(
        task: SubagentTask,
        lineage: SubagentLineage,
        token: CancellationToken,
        deadline: float,
    ) -> SubagentResult:
        nonlocal active, maximum_active
        assert deadline > time.monotonic()
        assert lineage.task_id == task.task_id
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        barrier.wait(timeout=2)
        if task.task_id == "first":
            time.sleep(0.04)
        token.raise_if_cancelled()
        with state_lock:
            active -= 1
        return _result(task, f"private-summary-{task.task_id}")

    scheduler = SubagentScheduler(execute, permission_authorizer=lambda _task: True)
    paired = scheduler.execute_batch(
        (
            SubagentTask("first", "private prompt alpha", "read-only"),
            SubagentTask("second", "private prompt beta", "workspace-write"),
        ),
        context=_context(effects=effects, progress=progress),
    )

    assert maximum_active == 2
    assert effects == ["effect"]
    assert [item[1].task_id for item in paired] == ["first", "second"]
    assert [item[0].ordinal for item in paired] == [0, 1]
    assert [item[0].depth for item in paired] == [1, 1]
    serialized_progress = json.dumps(progress, sort_keys=True)
    assert "private prompt" not in serialized_progress
    assert "private-summary" not in serialized_progress
    assert (
        set()
        .union(*(payload.keys() for _kind, payload in progress))
        .issubset({"agent_id", "count", "depth", "ordinal", "status"})
    )


def test_parent_cancellation_cascades_and_started_children_are_joined() -> None:
    parent_token = CancellationToken()
    both_started = Event()
    joined = Event()
    state_lock = Lock()
    started_count = 0
    joined_count = 0
    captured: list[BaseException] = []
    effects: list[str] = []

    def execute(
        task: SubagentTask,
        _lineage: SubagentLineage,
        token: CancellationToken,
        _deadline: float,
    ) -> SubagentResult:
        nonlocal started_count, joined_count
        with state_lock:
            started_count += 1
            if started_count == 2:
                both_started.set()
        token.wait(2.0)
        try:
            token.raise_if_cancelled()
        finally:
            with state_lock:
                joined_count += 1
                if joined_count == 2:
                    joined.set()
        return _result(task)

    scheduler = SubagentScheduler(execute, permission_authorizer=lambda _task: True)

    def invoke() -> None:
        try:
            scheduler.execute_batch(
                (
                    SubagentTask("one", "First child.", "read-only"),
                    SubagentTask("two", "Second child.", "read-only"),
                ),
                context=_context(token=parent_token, effects=effects),
            )
        except BaseException as exc:  # captured for the calling thread
            captured.append(exc)

    thread = Thread(target=invoke)
    thread.start()
    assert both_started.wait(2.0)
    assert parent_token.cancel("user stopped the run") is True
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert joined.is_set()
    assert joined_count == 2
    assert len(captured) == 1
    assert isinstance(captured[0], HarnessCancelled)
    assert effects == ["effect"]


def test_reconciliation_cancels_and_joins_siblings_then_fails_after_effect() -> None:
    barrier = Barrier(2)
    sibling_joined = Event()
    effects: list[str] = []

    def execute(
        task: SubagentTask,
        _lineage: SubagentLineage,
        token: CancellationToken,
        _deadline: float,
    ) -> SubagentResult:
        assert effects == ["effect"]
        barrier.wait(timeout=2.0)
        if task.task_id == "uncertain":
            return SubagentResult(
                task_id=task.task_id,
                status="failed",
                summary="The isolated child requires reconciliation.",
                changed=True,
                requires_reconciliation=True,
                error_code="child_reconciliation_required",
                artifact_id="worktree-uncertain",
            )
        token.wait(2.0)
        try:
            token.raise_if_cancelled()
        finally:
            sibling_joined.set()
        return _result(task)

    scheduler = SubagentScheduler(execute, permission_authorizer=lambda _task: True)
    with pytest.raises(ToolExecutionError) as captured:
        scheduler.execute_batch(
            (
                SubagentTask("sibling", "Wait for sibling.", "read-only"),
                SubagentTask("uncertain", "Create an uncertain effect.", "workspace-write"),
            ),
            context=_context(effects=effects),
        )

    assert captured.value.code == "subagent_reconciliation_required"
    assert sibling_joined.is_set()
    assert effects == ["effect"]
    snapshot = scheduler.budget_ledger.snapshot(root_run_id="run-1", parent_agent_id="run-1")
    assert snapshot.active_global == 0
    assert snapshot.active_for_parent == 0
    assert snapshot.total_for_root == 2


def test_shared_deadline_cancels_and_joins_cooperative_children() -> None:
    joined = Event()
    effects: list[str] = []

    def execute(
        task: SubagentTask,
        lineage: SubagentLineage,
        token: CancellationToken,
        deadline: float,
    ) -> SubagentResult:
        del task, lineage, deadline
        try:
            while True:
                token.raise_if_cancelled()
                token.wait(0.02)
        finally:
            joined.set()

    scheduler = SubagentScheduler(execute)
    context = replace(
        _context(effects=effects),
        deadline_monotonic=time.monotonic() + 0.1,
    )

    started = time.monotonic()
    with pytest.raises(HarnessDeadlineExceeded):
        scheduler.execute_batch(
            (SubagentTask("deadline", "Wait for the deadline.", "read-only"),),
            context=context,
        )

    assert time.monotonic() - started < 1.0
    assert joined.is_set()
    assert effects == ["effect"]
    snapshot = scheduler.budget_ledger.snapshot(root_run_id="run-1", parent_agent_id="run-1")
    assert snapshot.active_global == 0
    assert snapshot.total_for_root == 1


def test_budget_checks_are_atomic_and_total_is_not_released() -> None:
    limits = SubagentLimits(
        max_batch_size=2,
        max_global_active=2,
        max_parent_active=2,
        max_depth=2,
        max_total_per_root=3,
    )
    ledger = SubagentBudgetLedger(limits)
    start = Barrier(3)
    release = Event()
    outcomes: list[str] = []
    outcome_lock = Lock()

    def contend(parent_id: str) -> None:
        start.wait(timeout=2.0)
        try:
            reservation = ledger.reserve(
                root_run_id="root-atomic",
                parent_agent_id=parent_id,
                depth=1,
                count=2,
            )
        except ToolExecutionError as exc:
            with outcome_lock:
                outcomes.append(exc.code)
            return
        with outcome_lock:
            outcomes.append("reserved")
        release.wait(2.0)
        reservation.release()

    threads = [Thread(target=contend, args=(f"parent-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    start.wait(timeout=2.0)
    deadline = time.monotonic() + 2.0
    while len(outcomes) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    release.set()
    for thread in threads:
        thread.join(timeout=2.0)

    assert sorted(outcomes) == ["reserved", "subagent_global_limit"]
    assert ledger.snapshot(root_run_id="root-atomic", parent_agent_id="parent-0").active_global == 0
    # A released child still counts against the root lifetime budget.
    one_more = ledger.reserve(
        root_run_id="root-atomic",
        parent_agent_id="parent-final",
        depth=2,
        count=1,
    )
    one_more.release()
    with pytest.raises(ToolExecutionError) as captured:
        ledger.reserve(
            root_run_id="root-atomic",
            parent_agent_id="parent-final",
            depth=2,
            count=1,
        )
    assert captured.value.code == "subagent_total_limit"


def test_per_parent_capacity_is_independent_from_global_capacity() -> None:
    limits = SubagentLimits(
        max_batch_size=2,
        max_global_active=4,
        max_parent_active=2,
        max_depth=1,
        max_total_per_root=6,
    )
    ledger = SubagentBudgetLedger(limits)
    first = ledger.reserve(
        root_run_id="root-parent-budget",
        parent_agent_id="parent-one",
        depth=1,
        count=2,
    )
    with pytest.raises(ToolExecutionError) as captured:
        ledger.reserve(
            root_run_id="root-parent-budget",
            parent_agent_id="parent-one",
            depth=1,
            count=1,
        )
    assert captured.value.code == "subagent_parent_limit"
    second = ledger.reserve(
        root_run_id="root-parent-budget",
        parent_agent_id="parent-two",
        depth=1,
        count=2,
    )
    assert (
        ledger.snapshot(
            root_run_id="root-parent-budget", parent_agent_id="parent-one"
        ).active_global
        == 4
    )
    first.release()
    second.release()


def test_default_depth_limit_rejects_nested_batch_before_effect() -> None:
    effects: list[str] = []
    called = False

    def execute(
        task: SubagentTask,
        lineage: SubagentLineage,
        token: CancellationToken,
        deadline: float,
    ) -> SubagentResult:
        nonlocal called
        called = True
        return _result(task)

    scheduler = SubagentScheduler(execute)
    with pytest.raises(ToolExecutionError) as captured:
        scheduler.execute_batch(
            (SubagentTask("nested", "Nested task.", "read-only"),),
            context=_context(effects=effects, call_id="nested-call"),
            parent_lineage=_parent_lineage(),
        )

    assert captured.value.code == "subagent_depth_limit"
    assert called is False
    assert effects == []


def test_invalid_or_exceptional_child_output_is_bounded_and_redacted() -> None:
    secret = "raw-secret-exception-detail"

    def execute(
        task: SubagentTask,
        lineage: SubagentLineage,
        token: CancellationToken,
        deadline: float,
    ) -> SubagentResult:
        if task.task_id == "wrong-id":
            return SubagentResult("somebody-else", "completed", "Must not escape.")
        if task.task_id == "long":
            return _result(task, "x" * 100)
        raise RuntimeError(secret)

    limits = SubagentLimits(max_summary_chars=10)
    scheduler = SubagentScheduler(execute, limits=limits)
    paired = scheduler.execute_batch(
        (
            SubagentTask("wrong-id", "Wrong result contract.", "read-only"),
            SubagentTask("long", "Long result.", "read-only"),
            SubagentTask("exception", "Raise an exception.", "read-only"),
        ),
        context=_context(),
    )
    results = [result for _lineage, result in paired]

    assert results[0].error_code == "subagent_contract_violation"
    assert results[1].summary == "x" * 9 + "…"
    assert results[2].error_code == "subagent_execution_failed"
    assert secret not in json.dumps(
        [item.public_dict(agent_id="agent-1", depth=1, ordinal=0) for item in results]
    )


def test_model_visible_tool_contract_is_strict_high_risk_and_never_replay() -> None:
    def execute(
        task: SubagentTask,
        lineage: SubagentLineage,
        token: CancellationToken,
        deadline: float,
    ) -> SubagentResult:
        return SubagentResult(
            task_id=task.task_id,
            status="completed",
            summary="Final conclusion only.",
            changed=task.permission_mode == "workspace-write",
            artifact_id="worktree-1",
            session_id="child-session-1",
            run_id="child-run-1",
            turn_id="child-turn-1",
        )

    scheduler = SubagentScheduler(execute, permission_authorizer=lambda _task: True)
    registry = ToolRegistry()
    spec = register_subagent_tool(registry, scheduler)
    entry = registry.get("agent.delegate")
    assert entry is not None
    _registered_spec, handler = entry

    assert spec.risk == "high"
    assert spec.replay_policy == "never"
    assert spec.retry_policy.max_attempts == 1
    assert spec.permission == "agent.delegate"
    assert spec.data_scope == "workspace_read"
    assert spec.execution_isolation == "trusted_inline"
    assert spec.persistent_approval_allowed is False
    assert spec.parallel_safe is False
    assert spec.input_schema["additionalProperties"] is False
    assert set(spec.input_schema["properties"]) == {"tasks"}
    task_schema = spec.input_schema["properties"]["tasks"]["items"]
    assert task_schema["additionalProperties"] is False
    assert set(task_schema["properties"]) == {
        "task_id",
        "prompt",
        "permission_mode",
    }

    effects: list[str] = []
    output = handler(
        {
            "tasks": [
                {
                    "task_id": "task-1",
                    "prompt": "Implement a bounded change.",
                    "permission_mode": "workspace-write",
                }
            ]
        },
        _context(effects=effects),
    )
    validate_schema(output, spec.output_schema)
    assert effects == ["effect"]
    assert output["schema"] == SUBAGENT_BATCH_SCHEMA
    assert output["foreground"] is True
    assert output["results"][0]["artifact_id"] == "worktree-1"
    assert output["results"][0]["session_id"] == "child-session-1"
    assert output["results"][0]["run_id"] == "child-run-1"
    assert output["results"][0]["turn_id"] == "child-turn-1"
    assert "reasoning" not in json.dumps(output).lower()

    second_registry = build_subagent_registry(scheduler)
    assert [item.name for item in second_registry.specs()] == ["agent.delegate"]


def test_dynamic_permission_authorizer_fails_closed_after_parent_downgrade() -> None:
    authority = {"mode": "workspace-write"}
    calls: list[str] = []
    effects: list[str] = []

    def authorize(task: SubagentTask) -> bool:
        return task.permission_mode == "read-only" or authority["mode"] == "workspace-write"

    def execute(
        task: SubagentTask,
        lineage: SubagentLineage,
        token: CancellationToken,
        deadline: float,
    ) -> SubagentResult:
        calls.append(task.task_id)
        return _result(task)

    scheduler = SubagentScheduler(execute, permission_authorizer=authorize)
    scheduler.execute_batch(
        (SubagentTask("before", "Authorized before downgrade.", "workspace-write"),),
        context=_context(effects=effects, call_id="call-before"),
    )
    authority["mode"] = "read-only"

    with pytest.raises(ToolExecutionError) as captured:
        scheduler.execute_batch(
            (SubagentTask("after", "Must be rejected.", "workspace-write"),),
            context=_context(effects=effects, call_id="call-after"),
        )

    assert captured.value.code == "subagent_permission_denied"
    assert calls == ["before"]
    assert effects == ["effect"]


def test_default_permission_authority_is_read_only_and_checks_before_effect() -> None:
    calls: list[str] = []
    effects: list[str] = []

    def execute(
        task: SubagentTask,
        lineage: SubagentLineage,
        token: CancellationToken,
        deadline: float,
    ) -> SubagentResult:
        calls.append(task.task_id)
        return _result(task)

    scheduler = SubagentScheduler(execute)
    with pytest.raises(ToolExecutionError) as captured:
        scheduler.execute_batch(
            (SubagentTask("write", "Attempt a write.", "workspace-write"),),
            context=_context(effects=effects),
        )

    assert captured.value.code == "subagent_permission_denied"
    assert calls == []
    assert effects == []
    snapshot = scheduler.budget_ledger.snapshot(root_run_id="run-1", parent_agent_id="run-1")
    assert snapshot.total_for_root == 0


def test_tool_request_rejects_duplicates_and_unknown_fields_without_starting() -> None:
    calls: list[str] = []
    effects: list[str] = []

    def execute(
        task: SubagentTask,
        lineage: SubagentLineage,
        token: CancellationToken,
        deadline: float,
    ) -> SubagentResult:
        calls.append(task.task_id)
        return _result(task)

    scheduler = SubagentScheduler(execute)
    spec = build_subagent_registry(scheduler).specs()[0]
    with pytest.raises(HarnessContractError):
        validate_schema(
            {
                "tasks": [
                    {
                        "task_id": "one",
                        "prompt": "Do work.",
                        "permission_mode": "read-only",
                        "background": True,
                    }
                ]
            },
            spec.input_schema,
        )

    with pytest.raises(ToolExecutionError) as captured:
        scheduler.handle_tool(
            {
                "tasks": [
                    {
                        "task_id": "duplicate",
                        "prompt": "First.",
                        "permission_mode": "read-only",
                    },
                    {
                        "task_id": "duplicate",
                        "prompt": "Second.",
                        "permission_mode": "workspace-write",
                    },
                ]
            },
            _context(effects=effects),
        )
    assert captured.value.code == "subagent_duplicate_task"
    assert calls == []
    assert effects == []


def test_effect_boundary_failure_rolls_back_every_budget_dimension() -> None:
    limits = SubagentLimits(max_total_per_root=1)
    scheduler = SubagentScheduler(
        lambda task, lineage, token, deadline: _result(task), limits=limits
    )
    context = ToolExecutionContext(
        run_id="run-1",
        turn_id="turn-1",
        call_id="call-rollback",
        tool_name="agent.delegate",
        cancellation_token=CancellationToken(),
        deadline_monotonic=time.monotonic() + 5.0,
        idempotency_key=None,
        emit_progress=lambda _kind, _payload=None: None,
        begin_effect=lambda: (_ for _ in ()).throw(RuntimeError("journal unavailable")),
        session_id="session-1",
    )

    with pytest.raises(RuntimeError, match="journal unavailable"):
        scheduler.execute_batch(
            (SubagentTask("one", "Do work.", "read-only"),),
            context=context,
        )
    snapshot = scheduler.budget_ledger.snapshot(root_run_id="run-1", parent_agent_id="run-1")
    assert snapshot.active_global == 0
    assert snapshot.active_for_parent == 0
    assert snapshot.total_for_root == 0
