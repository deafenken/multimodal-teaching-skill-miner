"""Bounded, foreground-only subagent delegation contracts.

The scheduler in this module deliberately does not know how a child workspace,
model, or run is created.  Product adapters inject that effectful boundary as a
``ChildExecutor``.  This keeps delegation policy independently testable while
ensuring that every started child is cancelled and joined before the tool call
settles.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
import hashlib
import re
from threading import Lock
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .core import (
    CancellationToken,
    HarnessCancelled,
    HarnessContractError,
    HarnessDeadlineExceeded,
    RetryPolicy,
    ToolExecutionContext,
    ToolExecutionError,
    ToolRegistry,
    ToolSpec,
)


SUBAGENT_BATCH_SCHEMA = "agent_harness.subagent_batch.v1"
SUBAGENT_PERMISSION_MODES = frozenset({"read-only", "workspace-write"})
SUBAGENT_RESULT_STATUSES = frozenset({"completed", "failed", "cancelled"})

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HARD_MAX_PROMPT_CHARS = 20_000
_HARD_MAX_SUMMARY_CHARS = 16_000
_HARD_MAX_ARTIFACT_ID_CHARS = 128


def _require_int(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarnessContractError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise HarnessContractError(f"{name} is outside its allowed range")
    return value


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise HarnessContractError(f"{name} is invalid")
    return value


def _require_opaque_id(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 200
        or _CONTROL.search(value) is not None
    ):
        raise HarnessContractError(f"{name} is invalid")
    return value


def _require_text(name: str, value: object, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or _CONTROL.search(value) is not None
    ):
        raise HarnessContractError(f"{name} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class SubagentLimits:
    """Immutable fan-out and output limits for one scheduler authority."""

    max_batch_size: int = 4
    max_global_active: int = 4
    max_parent_active: int = 4
    max_depth: int = 1
    max_total_per_root: int = 4
    max_prompt_chars: int = 8_000
    max_summary_chars: int = 4_000

    def __post_init__(self) -> None:
        _require_int("max_batch_size", self.max_batch_size, minimum=1, maximum=4)
        _require_int("max_global_active", self.max_global_active, minimum=1, maximum=64)
        _require_int("max_parent_active", self.max_parent_active, minimum=1, maximum=64)
        _require_int("max_depth", self.max_depth, minimum=1, maximum=8)
        _require_int("max_total_per_root", self.max_total_per_root, minimum=1, maximum=256)
        _require_int(
            "max_prompt_chars",
            self.max_prompt_chars,
            minimum=1,
            maximum=_HARD_MAX_PROMPT_CHARS,
        )
        _require_int(
            "max_summary_chars",
            self.max_summary_chars,
            minimum=1,
            maximum=_HARD_MAX_SUMMARY_CHARS,
        )
        if self.max_parent_active > self.max_global_active:
            raise HarnessContractError("max_parent_active cannot exceed max_global_active")


@dataclass(frozen=True, slots=True)
class SubagentTask:
    """One bounded child assignment; ``prompt`` is never progress metadata."""

    task_id: str
    prompt: str
    permission_mode: str = "read-only"

    def __post_init__(self) -> None:
        _require_identifier("subagent task_id", self.task_id)
        _require_text("subagent prompt", self.prompt, maximum=_HARD_MAX_PROMPT_CHARS)
        if self.permission_mode not in SUBAGENT_PERMISSION_MODES:
            raise HarnessContractError("subagent permission_mode is invalid")


@dataclass(frozen=True, slots=True)
class SubagentLineage:
    """Opaque tree identity supplied to an injected child executor."""

    root_run_id: str
    parent_session_id: str | None
    parent_run_id: str
    parent_turn_id: str
    parent_call_id: str
    agent_id: str
    depth: int
    ordinal: int
    task_id: str

    def __post_init__(self) -> None:
        _require_opaque_id("root_run_id", self.root_run_id)
        if self.parent_session_id is not None:
            _require_opaque_id("parent_session_id", self.parent_session_id)
        _require_opaque_id("parent_run_id", self.parent_run_id)
        _require_opaque_id("parent_turn_id", self.parent_turn_id)
        _require_opaque_id("parent_call_id", self.parent_call_id)
        _require_identifier("agent_id", self.agent_id)
        _require_int("subagent depth", self.depth, minimum=1, maximum=8)
        _require_int("subagent ordinal", self.ordinal, minimum=0, maximum=255)
        _require_identifier("subagent lineage task_id", self.task_id)


@dataclass(frozen=True, slots=True)
class SubagentResult:
    """Bounded child conclusion.

    There is intentionally no reasoning, transcript, or raw exception field.
    A child returns only a final summary and an optional opaque artifact ID.
    """

    task_id: str
    status: str
    summary: str
    changed: bool = False
    requires_reconciliation: bool = False
    error_code: str | None = None
    artifact_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    turn_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier("subagent result task_id", self.task_id)
        if self.status not in SUBAGENT_RESULT_STATUSES:
            raise HarnessContractError("subagent result status is invalid")
        _require_text(
            "subagent result summary",
            self.summary,
            maximum=_HARD_MAX_SUMMARY_CHARS,
        )
        if not isinstance(self.changed, bool):
            raise HarnessContractError("subagent result changed flag is invalid")
        if not isinstance(self.requires_reconciliation, bool):
            raise HarnessContractError("subagent reconciliation flag is invalid")
        if self.error_code is not None and (
            not isinstance(self.error_code, str) or _ERROR_CODE.fullmatch(self.error_code) is None
        ):
            raise HarnessContractError("subagent result error_code is invalid")
        if self.artifact_id is not None:
            if (
                not isinstance(self.artifact_id, str)
                or len(self.artifact_id) > _HARD_MAX_ARTIFACT_ID_CHARS
                or _IDENTIFIER.fullmatch(self.artifact_id) is None
            ):
                raise HarnessContractError("subagent artifact_id is invalid")
        for name, value in (
            ("subagent session_id", self.session_id),
            ("subagent run_id", self.run_id),
            ("subagent turn_id", self.turn_id),
        ):
            if value is not None:
                _require_opaque_id(name, value)
        if self.status == "completed" and self.error_code is not None:
            raise HarnessContractError("completed subagent results cannot declare an error_code")

    def public_dict(self, *, agent_id: str, depth: int, ordinal: int) -> dict[str, Any]:
        """Return the complete provider-visible result without hidden reasoning."""

        value: dict[str, Any] = {
            "task_id": self.task_id,
            "agent_id": agent_id,
            "depth": depth,
            "ordinal": ordinal,
            "status": self.status,
            "summary": self.summary,
            "changed": self.changed,
        }
        if self.error_code is not None:
            value["error_code"] = self.error_code
        if self.artifact_id is not None:
            value["artifact_id"] = self.artifact_id
        if self.session_id is not None:
            value["session_id"] = self.session_id
        if self.run_id is not None:
            value["run_id"] = self.run_id
        if self.turn_id is not None:
            value["turn_id"] = self.turn_id
        return value


class ChildExecutor(Protocol):
    """Effectful adapter implemented by the product/worktree layer."""

    def __call__(
        self,
        task: SubagentTask,
        lineage: SubagentLineage,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> SubagentResult: ...


@dataclass(frozen=True, slots=True)
class SubagentBudgetSnapshot:
    active_global: int
    active_for_parent: int
    total_for_root: int


class _BudgetReservation:
    def __init__(
        self,
        ledger: "SubagentBudgetLedger",
        *,
        root_run_id: str,
        parent_key: tuple[str, str],
        count: int,
    ) -> None:
        self._ledger = ledger
        self._root_run_id = root_run_id
        self._parent_key = parent_key
        self._count = count
        self._lock = Lock()
        self._active = True
        self._charged = True

    def release(self) -> None:
        """Release active capacity but retain the root's total-spawn charge."""

        with self._lock:
            if not self._active:
                return
            self._active = False
        self._ledger._release(self._parent_key, self._count)

    def rollback(self) -> None:
        """Undo a reservation when the effect boundary was never crossed."""

        with self._lock:
            if not self._active and not self._charged:
                return
            release_active = self._active
            rollback_total = self._charged
            self._active = False
            self._charged = False
        self._ledger._rollback(
            self._root_run_id,
            self._parent_key,
            self._count,
            release_active=release_active,
            rollback_total=rollback_total,
        )


class SubagentBudgetLedger:
    """Thread-safe atomic authority for global, parent, depth, and total limits."""

    def __init__(self, limits: SubagentLimits | None = None) -> None:
        self.limits = limits or SubagentLimits()
        self._lock = Lock()
        self._active_global = 0
        self._active_by_parent: dict[tuple[str, str], int] = {}
        self._total_by_root: dict[str, int] = {}

    def reserve(
        self,
        *,
        root_run_id: str,
        parent_agent_id: str,
        depth: int,
        count: int,
    ) -> _BudgetReservation:
        _require_opaque_id("root_run_id", root_run_id)
        _require_opaque_id("parent_agent_id", parent_agent_id)
        _require_int("subagent batch count", count, minimum=1, maximum=4)
        _require_int("subagent depth", depth, minimum=1, maximum=8)
        parent_key = (root_run_id, parent_agent_id)
        with self._lock:
            parent_active = self._active_by_parent.get(parent_key, 0)
            root_total = self._total_by_root.get(root_run_id, 0)
            if count > self.limits.max_batch_size:
                raise ToolExecutionError(
                    "subagent batch exceeds the configured limit",
                    code="subagent_batch_limit",
                )
            if depth > self.limits.max_depth:
                raise ToolExecutionError(
                    "subagent nesting depth exceeds the configured limit",
                    code="subagent_depth_limit",
                )
            if self._active_global + count > self.limits.max_global_active:
                raise ToolExecutionError(
                    "global subagent capacity is exhausted",
                    code="subagent_global_limit",
                )
            if parent_active + count > self.limits.max_parent_active:
                raise ToolExecutionError(
                    "parent subagent capacity is exhausted",
                    code="subagent_parent_limit",
                )
            if root_total + count > self.limits.max_total_per_root:
                raise ToolExecutionError(
                    "root subagent budget is exhausted",
                    code="subagent_total_limit",
                )
            self._active_global += count
            self._active_by_parent[parent_key] = parent_active + count
            self._total_by_root[root_run_id] = root_total + count
        return _BudgetReservation(
            self,
            root_run_id=root_run_id,
            parent_key=parent_key,
            count=count,
        )

    def _release(self, parent_key: tuple[str, str], count: int) -> None:
        with self._lock:
            self._active_global -= count
            remaining = self._active_by_parent[parent_key] - count
            if remaining:
                self._active_by_parent[parent_key] = remaining
            else:
                self._active_by_parent.pop(parent_key, None)

    def _rollback(
        self,
        root_run_id: str,
        parent_key: tuple[str, str],
        count: int,
        *,
        release_active: bool,
        rollback_total: bool,
    ) -> None:
        with self._lock:
            if release_active:
                self._active_global -= count
                remaining = self._active_by_parent[parent_key] - count
                if remaining:
                    self._active_by_parent[parent_key] = remaining
                else:
                    self._active_by_parent.pop(parent_key, None)
            if rollback_total:
                total = self._total_by_root[root_run_id] - count
                if total:
                    self._total_by_root[root_run_id] = total
                else:
                    self._total_by_root.pop(root_run_id, None)

    def finish_root(self, root_run_id: str) -> None:
        """Forget a completed root only after every child has joined."""

        _require_opaque_id("root_run_id", root_run_id)
        with self._lock:
            if any(key[0] == root_run_id for key in self._active_by_parent):
                raise ToolExecutionError(
                    "cannot finish a root while subagents are active",
                    code="subagent_root_active",
                )
            self._total_by_root.pop(root_run_id, None)

    def snapshot(self, *, root_run_id: str, parent_agent_id: str) -> SubagentBudgetSnapshot:
        parent_key = (root_run_id, parent_agent_id)
        with self._lock:
            return SubagentBudgetSnapshot(
                active_global=self._active_global,
                active_for_parent=self._active_by_parent.get(parent_key, 0),
                total_for_root=self._total_by_root.get(root_run_id, 0),
            )


class SubagentScheduler:
    """Run a bounded batch concurrently and settle it in deterministic order."""

    def __init__(
        self,
        child_executor: ChildExecutor,
        *,
        limits: SubagentLimits | None = None,
        budget_ledger: SubagentBudgetLedger | None = None,
        permission_authorizer: Callable[[SubagentTask], bool] | None = None,
    ) -> None:
        if not callable(child_executor):
            raise HarnessContractError("subagent child_executor must be callable")
        self.limits = limits or SubagentLimits()
        self.budget_ledger = budget_ledger or SubagentBudgetLedger(self.limits)
        if self.budget_ledger.limits != self.limits:
            raise HarnessContractError("subagent scheduler and budget ledger limits must match")
        self._child_executor = child_executor
        if permission_authorizer is not None and not callable(permission_authorizer):
            raise HarnessContractError("subagent permission_authorizer must be callable")
        self._permission_authorizer = permission_authorizer or (
            lambda task: task.permission_mode == "read-only"
        )

    def _validated_tasks(self, tasks: Sequence[SubagentTask]) -> tuple[SubagentTask, ...]:
        if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
            raise HarnessContractError("subagent tasks must be a sequence")
        frozen = tuple(tasks)
        if not 1 <= len(frozen) <= self.limits.max_batch_size:
            raise ToolExecutionError(
                "subagent batch exceeds the configured limit",
                code="subagent_batch_limit",
            )
        if any(not isinstance(task, SubagentTask) for task in frozen):
            raise HarnessContractError("subagent tasks must use SubagentTask")
        identifiers = [task.task_id for task in frozen]
        if len(identifiers) != len(set(identifiers)):
            raise ToolExecutionError(
                "subagent task IDs must be unique",
                code="subagent_duplicate_task",
            )
        for task in frozen:
            if len(task.prompt) > self.limits.max_prompt_chars:
                raise ToolExecutionError(
                    "subagent prompt exceeds the configured limit",
                    code="subagent_prompt_limit",
                )
            try:
                authorized = self._permission_authorizer(task)
            except Exception as exc:
                raise ToolExecutionError(
                    "subagent permission could not be authorized",
                    code="subagent_permission_denied",
                ) from exc
            if authorized is not True:
                raise ToolExecutionError(
                    "subagent permission was not authorized",
                    code="subagent_permission_denied",
                )
        return frozen

    @staticmethod
    def _agent_id(
        *, root_run_id: str, parent_agent_id: str, call_id: str, ordinal: int, task_id: str
    ) -> str:
        material = "\0".join((root_run_id, parent_agent_id, call_id, str(ordinal), task_id)).encode(
            "utf-8"
        )
        return f"agent-{hashlib.sha256(material).hexdigest()[:24]}"

    def _lineages(
        self,
        tasks: tuple[SubagentTask, ...],
        context: ToolExecutionContext,
        parent_lineage: SubagentLineage | None,
    ) -> tuple[SubagentLineage, ...]:
        if parent_lineage is None:
            root_run_id = context.run_id
            parent_agent_id = context.run_id
            depth = 1
        else:
            if not isinstance(parent_lineage, SubagentLineage):
                raise HarnessContractError("parent_lineage is invalid")
            root_run_id = parent_lineage.root_run_id
            parent_agent_id = parent_lineage.agent_id
            depth = parent_lineage.depth + 1
        return tuple(
            SubagentLineage(
                root_run_id=root_run_id,
                parent_session_id=context.session_id,
                parent_run_id=context.run_id,
                parent_turn_id=context.turn_id,
                parent_call_id=context.call_id,
                agent_id=self._agent_id(
                    root_run_id=root_run_id,
                    parent_agent_id=parent_agent_id,
                    call_id=context.call_id,
                    ordinal=ordinal,
                    task_id=task.task_id,
                ),
                depth=depth,
                ordinal=ordinal,
                task_id=task.task_id,
            )
            for ordinal, task in enumerate(tasks)
        )

    def _settle_child_result(
        self,
        task: SubagentTask,
        result: object,
    ) -> SubagentResult:
        if not isinstance(result, SubagentResult) or result.task_id != task.task_id:
            return SubagentResult(
                task_id=task.task_id,
                status="failed",
                summary="Subagent returned an invalid result contract.",
                error_code="subagent_contract_violation",
            )
        if len(result.summary) <= self.limits.max_summary_chars:
            return result
        # Preserve bounded user-facing content without exposing an exception or
        # a hidden continuation channel.
        suffix = "…"
        return SubagentResult(
            task_id=result.task_id,
            status=result.status,
            summary=result.summary[: self.limits.max_summary_chars - len(suffix)] + suffix,
            changed=result.changed,
            requires_reconciliation=result.requires_reconciliation,
            error_code=result.error_code,
            artifact_id=result.artifact_id,
            session_id=result.session_id,
            run_id=result.run_id,
            turn_id=result.turn_id,
        )

    def execute_batch(
        self,
        tasks: Sequence[SubagentTask],
        *,
        context: ToolExecutionContext,
        parent_lineage: SubagentLineage | None = None,
    ) -> tuple[tuple[SubagentLineage, SubagentResult], ...]:
        """Execute every child in parallel, then return results in input order."""

        frozen_tasks = self._validated_tasks(tasks)
        if not isinstance(context, ToolExecutionContext):
            raise HarnessContractError("subagent execution context is invalid")
        context.cancellation_token.raise_if_cancelled()
        if context.deadline_monotonic <= time.monotonic():
            raise ToolExecutionError(
                "subagent deadline expired before execution",
                code="subagent_deadline_exceeded",
            )
        lineages = self._lineages(frozen_tasks, context, parent_lineage)
        depth = lineages[0].depth
        root_run_id = lineages[0].root_run_id
        parent_agent_id = context.run_id if parent_lineage is None else parent_lineage.agent_id
        reservation = self.budget_ledger.reserve(
            root_run_id=root_run_id,
            parent_agent_id=parent_agent_id,
            depth=depth,
            count=len(frozen_tasks),
        )
        try:
            context.begin_effect()
        except BaseException:
            reservation.rollback()
            raise

        child_tokens = tuple(CancellationToken() for _task in frozen_tasks)

        def cancel_children(reason: str) -> None:
            for token in child_tokens:
                token.cancel(reason)

        unsubscribe = context.cancellation_token.add_callback(
            lambda: cancel_children("parent_cancelled")
        )
        reconciliation_lock = Lock()
        reconciliation_required = False

        def mark_reconciliation() -> None:
            nonlocal reconciliation_required
            with reconciliation_lock:
                reconciliation_required = True
            cancel_children("sibling_requires_reconciliation")

        def run_one(index: int) -> SubagentResult:
            task = frozen_tasks[index]
            token = child_tokens[index]
            try:
                token.raise_if_cancelled()
                raw = self._child_executor(
                    task,
                    lineages[index],
                    token,
                    context.deadline_monotonic,
                )
                # An executor is not allowed to turn a cancellation/deadline
                # race into a successful child result.  The injected runner is
                # cooperative, but this postcondition keeps the scheduler's
                # boundary fail-closed even if an adapter returns just after
                # its authority expired.
                token.raise_if_cancelled()
                if time.monotonic() >= context.deadline_monotonic:
                    raise HarnessDeadlineExceeded("subagent exceeded the shared deadline")
                result = self._settle_child_result(task, raw)
            except HarnessCancelled:
                result = SubagentResult(
                    task_id=task.task_id,
                    status="cancelled",
                    summary="Subagent was cancelled.",
                    error_code="subagent_cancelled",
                )
            except HarnessDeadlineExceeded:
                result = SubagentResult(
                    task_id=task.task_id,
                    status="failed",
                    summary="Subagent exceeded the shared deadline.",
                    error_code="subagent_deadline_exceeded",
                )
            except Exception:
                # Raw child exceptions are intentionally not exposed to either
                # the model or progress observers.
                result = SubagentResult(
                    task_id=task.task_id,
                    status="failed",
                    summary="Subagent execution failed.",
                    error_code="subagent_execution_failed",
                )
            if result.requires_reconciliation:
                mark_reconciliation()
            return result

        futures: list[Future[SubagentResult]] = []
        results: list[SubagentResult | None] = [None] * len(frozen_tasks)
        try:
            context.emit_progress(
                "subagent.batch_started", {"count": len(frozen_tasks), "depth": depth}
            )
            pool = ThreadPoolExecutor(
                max_workers=len(frozen_tasks),
                thread_name_prefix="agent-harness-subagent",
            )
            try:
                for index, lineage in enumerate(lineages):
                    context.emit_progress(
                        "subagent.child_started",
                        {
                            "agent_id": lineage.agent_id,
                            "depth": lineage.depth,
                            "ordinal": lineage.ordinal,
                        },
                    )
                    futures.append(pool.submit(run_one, index))
                # Waiting in input order preserves deterministic observations;
                # all child work itself is already concurrent.  Poll in short
                # slices so parent cancellation and the shared deadline are
                # observed even while a future is still running.  The finally
                # block still joins every started child before settlement.
                for index, future in enumerate(futures):
                    while True:
                        context.cancellation_token.raise_if_cancelled()
                        remaining = context.deadline_monotonic - time.monotonic()
                        if remaining <= 0:
                            raise HarnessDeadlineExceeded(
                                "subagent batch exceeded the shared deadline"
                            )
                        try:
                            result = future.result(timeout=min(0.05, remaining))
                            break
                        except FutureTimeout:
                            continue
                    results[index] = result
                    context.emit_progress(
                        "subagent.child_finished",
                        {
                            "agent_id": lineages[index].agent_id,
                            "depth": lineages[index].depth,
                            "ordinal": lineages[index].ordinal,
                            "status": result.status,
                        },
                    )
            except BaseException:
                cancel_children("subagent_batch_aborted")
                for future in futures:
                    future.cancel()
                raise
            finally:
                # Cancellation happens before this blocking join, so an
                # injected executor which honors its token cannot be orphaned.
                pool.shutdown(wait=True, cancel_futures=True)
        except BaseException:
            cancel_children("subagent_batch_aborted")
            raise
        finally:
            unsubscribe()
            reservation.release()

        if reconciliation_required:
            raise ToolExecutionError(
                "a child requires reconciliation; all siblings were cancelled and joined",
                code="subagent_reconciliation_required",
            )
        context.cancellation_token.raise_if_cancelled()
        settled = tuple(result for result in results if result is not None)
        if len(settled) != len(frozen_tasks):
            raise ToolExecutionError(
                "subagent batch did not settle every child",
                code="subagent_batch_incomplete",
            )
        context.emit_progress("subagent.batch_finished", {"count": len(settled), "depth": depth})
        return tuple(zip(lineages, settled))

    def handle_tool(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        raw_tasks = arguments.get("tasks")
        if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
            raise ToolExecutionError(
                "tasks must be an array",
                code="subagent_invalid_request",
            )
        tasks: list[SubagentTask] = []
        try:
            for raw in raw_tasks:
                if not isinstance(raw, Mapping):
                    raise HarnessContractError("subagent task must be an object")
                tasks.append(
                    SubagentTask(
                        task_id=raw.get("task_id"),  # type: ignore[arg-type]
                        prompt=raw.get("prompt"),  # type: ignore[arg-type]
                        permission_mode=raw.get("permission_mode"),  # type: ignore[arg-type]
                    )
                )
        except HarnessContractError as exc:
            raise ToolExecutionError(
                "subagent request is invalid",
                code="subagent_invalid_request",
            ) from exc
        paired = self.execute_batch(tasks, context=context)
        return {
            "schema": SUBAGENT_BATCH_SCHEMA,
            "foreground": True,
            "results": [
                result.public_dict(
                    agent_id=lineage.agent_id,
                    depth=lineage.depth,
                    ordinal=lineage.ordinal,
                )
                for lineage, result in paired
            ],
        }


def _task_schema(limits: SubagentLimits) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
                "pattern": _IDENTIFIER.pattern,
            },
            "prompt": {
                "type": "string",
                "minLength": 1,
                "maxLength": limits.max_prompt_chars,
            },
            "permission_mode": {
                "type": "string",
                "enum": ["read-only", "workspace-write"],
            },
        },
        "required": ["task_id", "prompt", "permission_mode"],
        "additionalProperties": False,
    }


def _result_schema(limits: SubagentLimits) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "agent_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "depth": {"type": "integer", "minimum": 1, "maximum": limits.max_depth},
            "ordinal": {"type": "integer", "minimum": 0, "maximum": 3},
            "status": {
                "type": "string",
                "enum": ["completed", "failed", "cancelled"],
            },
            "summary": {
                "type": "string",
                "minLength": 1,
                "maxLength": limits.max_summary_chars,
            },
            "changed": {"type": "boolean"},
            "error_code": {"type": "string", "minLength": 1, "maxLength": 128},
            "artifact_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": _HARD_MAX_ARTIFACT_ID_CHARS,
            },
            "session_id": {"type": "string", "minLength": 1, "maxLength": 200},
            "run_id": {"type": "string", "minLength": 1, "maxLength": 200},
            "turn_id": {"type": "string", "minLength": 1, "maxLength": 200},
        },
        "required": [
            "task_id",
            "agent_id",
            "depth",
            "ordinal",
            "status",
            "summary",
            "changed",
        ],
        "additionalProperties": False,
    }


def register_subagent_tool(registry: ToolRegistry, scheduler: SubagentScheduler) -> ToolSpec:
    """Register the single foreground delegation capability."""

    if not isinstance(registry, ToolRegistry):
        raise HarnessContractError("subagent registry is invalid")
    if not isinstance(scheduler, SubagentScheduler):
        raise HarnessContractError("subagent scheduler is invalid")
    limits = scheduler.limits
    spec = ToolSpec(
        name="agent.delegate",
        version="1",
        description=(
            "Run one foreground batch of 1-4 isolated subagents concurrently and "
            "wait for every child; no background, resume, steer, or hidden reasoning."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": _task_schema(limits),
                    "minItems": 1,
                    "maxItems": limits.max_batch_size,
                }
            },
            "required": ["tasks"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "schema": {"type": "string", "const": SUBAGENT_BATCH_SCHEMA},
                "foreground": {"type": "boolean", "const": True},
                "results": {
                    "type": "array",
                    "items": _result_schema(limits),
                    "minItems": 1,
                    "maxItems": limits.max_batch_size,
                },
            },
            "required": ["schema", "foreground", "results"],
            "additionalProperties": False,
        },
        permission="agent.delegate",
        risk="high",
        timeout_seconds=1_800.0,
        replay_policy="never",
        parallel_safe=False,
        data_scope="workspace_read",
        execution_isolation="trusted_inline",
        trusted_inline_reason=(
            "Bounded foreground scheduler joins isolated children behind one effect fence"
        ),
        persistent_approval_allowed=False,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    registry.register(spec, scheduler.handle_tool)
    return spec


def build_subagent_registry(scheduler: SubagentScheduler) -> ToolRegistry:
    registry = ToolRegistry()
    register_subagent_tool(registry, scheduler)
    return registry


__all__ = [
    "SUBAGENT_BATCH_SCHEMA",
    "SUBAGENT_PERMISSION_MODES",
    "SUBAGENT_RESULT_STATUSES",
    "ChildExecutor",
    "SubagentBudgetLedger",
    "SubagentBudgetSnapshot",
    "SubagentLimits",
    "SubagentLineage",
    "SubagentResult",
    "SubagentScheduler",
    "SubagentTask",
    "build_subagent_registry",
    "register_subagent_tool",
]
