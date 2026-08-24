"""Standard-library curses TUI for Agent Harness."""

from __future__ import annotations

import argparse
import curses
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
import re
import sys
from threading import Event, Thread
import time
from typing import Any, Mapping, Sequence

from .attachments import (
    AttachmentDescriptor,
    AttachmentError,
    MAX_ATTACHMENTS_PER_TURN,
    MAX_ATTACHMENTS_TOTAL_BYTES,
)
from .core import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRule,
    CancellationToken,
)
from .hooks import load_project_hooks
from .runner import AgentRunner, TurnOutcome
from .session import SessionStoreError
from .toolsets import PERMISSION_PROFILES, permission_profile
from .tui_state import (
    TuiState,
    display_width,
    sanitize_terminal_text,
    truncate_display,
    wrap_display,
)


_MAX_FOLLOWUPS = 32
_MAX_FOLLOWUP_CHARS = 100_000
_OPAQUE_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ARTIFACT_PHASE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class _PendingTurn:
    prompt: str
    attachments: tuple[AttachmentDescriptor, ...] = ()


@dataclass(slots=True)
class _PendingApproval:
    request: ApprovalRequest
    preview: str
    event: Event = field(default_factory=Event)
    decision: ApprovalDecision | None = None

    def settle(self, *, verdict: str, reason_code: str) -> None:
        if self.event.is_set():
            return
        self.decision = ApprovalDecision.for_request(
            self.request,
            verdict=verdict,  # type: ignore[arg-type]
            reason_code=reason_code,
        )
        self.event.set()


class _TuiApprovalBroker:
    def __init__(
        self,
        queue: Queue[tuple[str, Any]],
        cancellation: CancellationToken,
    ) -> None:
        self.queue = queue
        self.cancellation = cancellation

    def decide(
        self,
        request: ApprovalRequest,
        *,
        preview: str,
    ) -> ApprovalDecision:
        pending = _PendingApproval(
            request=request,
            preview=sanitize_terminal_text(preview)[:20_000],
        )
        self.queue.put(("approval", pending))
        while not pending.event.wait(0.05):
            self.cancellation.raise_if_cancelled()
        if pending.decision is None:
            return ApprovalDecision.for_request(
                request,
                verdict="deny",
                reason_code="approval_broker_failed",
            )
        return pending.decision


class HarnessTui:
    def __init__(self, runner: AgentRunner, session: Mapping[str, Any]) -> None:
        self.runner = runner
        self.session = dict(session)
        self.state = TuiState(session_id=str(session["session_id"]))
        self.transcript = [dict(item) for item in session.get("messages", [])]
        self.input_buffer = ""
        self.scroll = 0
        self.queue: Queue[tuple[str, Any]] = Queue()
        self.worker: Thread | None = None
        self.worker_kind: str | None = None
        self.cancellation: CancellationToken | None = None
        self.followups: list[_PendingTurn] = []
        self.pending_attachments: list[AttachmentDescriptor] = []
        self.active_turn: _PendingTurn | None = None
        self.pending_approval: _PendingApproval | None = None
        self.approval_scroll = 0
        self.approval_preview_visible = False
        self.approval_broker: _TuiApprovalBroker | None = None
        # Keep an explicit record of exact rules granted while this TUI is
        # running. The durable store remains the authority for later runs;
        # this list makes the current interactive grant visible to the
        # in-process controller without broadening the frozen run policy.
        self.turn_approval_rules: list[ApprovalRule] = []
        self.running = True

    @property
    def busy(self) -> bool:
        # A finished thread remains authoritative until its queued outcome has
        # been reduced. Treat that settlement window as busy so a new run
        # cannot be overwritten by the previous outcome.
        return self.worker is not None

    def _event_sink(self, event: Mapping[str, Any]) -> None:
        self.queue.put(("event", dict(event)))

    def start_turn(
        self,
        prompt: str,
        attachments: Sequence[AttachmentDescriptor | Mapping[str, Any]] = (),
    ) -> bool:
        prompt = prompt.strip()
        if not prompt:
            return False
        try:
            frozen_attachments = tuple(
                item
                if isinstance(item, AttachmentDescriptor)
                else AttachmentDescriptor.from_value(item)
                for item in attachments
            )
        except (AttachmentError, TypeError):
            self.state.notice("附件描述符无效；本轮未发送")
            return False
        if len(frozen_attachments) > MAX_ATTACHMENTS_PER_TURN:
            self.state.notice("单轮附件不能超过 8 个；本轮未发送")
            return False
        if sum(item.size_bytes for item in frozen_attachments) > MAX_ATTACHMENTS_TOTAL_BYTES:
            self.state.notice("单轮附件总量不能超过 24 MiB；本轮未发送")
            return False
        pending_turn = _PendingTurn(prompt=prompt, attachments=frozen_attachments)
        if self.busy:
            if (
                len(self.followups) >= _MAX_FOLLOWUPS
                or sum(len(item.prompt) for item in self.followups) + len(prompt)
                > _MAX_FOLLOWUP_CHARS
            ):
                self.state.notice("后续队列已满；请等待当前运行完成")
                return False
            self.followups.append(pending_turn)
            self.state.notice(f"已加入后续队列（{len(self.followups)}）")
            return True
        local_message: dict[str, Any] = {"role": "user", "content": prompt}
        if frozen_attachments:
            local_message["attachments"] = [
                item.to_dict() for item in frozen_attachments
            ]
        self.transcript.append(local_message)
        self.active_turn = pending_turn
        self.state.begin_turn()
        self.cancellation = CancellationToken()
        self.approval_broker = _TuiApprovalBroker(
            self.queue,
            self.cancellation,
        )

        def work() -> None:
            try:
                outcome = self.runner.run_turn(
                    self.state.session_id,
                    prompt,
                    cancellation_token=self.cancellation,
                    event_sink=self._event_sink,
                    approval_broker=self.approval_broker,
                    attachments=pending_turn.attachments,
                )
                self.queue.put(("outcome", outcome))
            except BaseException as exc:
                self.queue.put(("error", exc))

        self.worker = Thread(target=work, name="agent-harness-turn", daemon=True)
        self.worker_kind = "turn"
        self.worker.start()
        return True

    def start_compaction(self) -> None:
        if self.busy:
            self.state.notice("运行完成后才能压缩上下文")
            return
        self.state.status = "compacting"
        self.cancellation = CancellationToken()

        def work() -> None:
            try:
                result = self.runner.compact_session(
                    self.state.session_id,
                    cancellation_token=self.cancellation,
                )
                self.queue.put(("compaction_outcome", result))
            except BaseException as exc:
                self.queue.put(("error", exc))

        self.worker = Thread(
            target=work,
            name="agent-harness-compaction",
            daemon=True,
        )
        self.worker_kind = "compaction"
        self.worker.start()

    def _complete_turn(self, outcome: TurnOutcome) -> None:
        if outcome.session_id != self.state.session_id:
            raise ValueError("outcome belongs to another session")
        if (
            self.state.active_run_id is not None
            and outcome.run_id != self.state.active_run_id
        ):
            raise ValueError("outcome belongs to another run")
        if (
            self.state.active_turn_id is not None
            and outcome.turn_id != self.state.active_turn_id
        ):
            raise ValueError("outcome belongs to another turn")
        if not outcome.message and outcome.status != "cancelled":
            self.state.notice(f"运行未生成回答：{outcome.status} · {outcome.reason}")
        persisted = self.runner.store.load(self.state.session_id)
        self.session = dict(persisted)
        self.transcript = [dict(item) for item in persisted.get("messages", [])]
        self.state.assistant_draft = ""
        self.worker = None
        self.worker_kind = None
        self.cancellation = None
        self.pending_approval = None
        self.approval_scroll = 0
        self.approval_preview_visible = False
        self.approval_broker = None
        self.active_turn = None
        if self.followups:
            next_turn = self.followups.pop(0)
            self.start_turn(next_turn.prompt, next_turn.attachments)

    def _complete_compaction(self, result: Mapping[str, Any]) -> None:
        persisted = self.runner.store.load(self.state.session_id)
        self.session = dict(persisted)
        self.transcript = [dict(item) for item in persisted.get("messages", [])]
        self.worker = None
        self.worker_kind = None
        self.cancellation = None
        self.state.status = "idle"
        if result.get("status") == "compacted":
            self.state.notice(
                f"上下文已压缩：覆盖 {result.get('source_message_count', 0)} 条原始消息 · "
                f"保留 {result.get('active_message_count', 0)} 条活动消息 · "
                f"摘要 {str(result.get('summary_sha256', ''))[:12]}"
            )
        else:
            self.state.notice("没有可压缩的已完成旧轮次")
        if self.followups:
            next_turn = self.followups.pop(0)
            self.start_turn(next_turn.prompt, next_turn.attachments)

    def drain(self) -> None:
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except Empty:
                return
            try:
                if kind == "event":
                    self.state.apply_event(payload)
                elif kind == "outcome":
                    self._complete_turn(payload)
                elif kind == "compaction_outcome":
                    if not isinstance(payload, Mapping):
                        raise ValueError("invalid compaction outcome")
                    self._complete_compaction(payload)
                elif kind == "approval":
                    if isinstance(payload, _PendingApproval) and not payload.event.is_set():
                        self.pending_approval = payload
                        self.approval_scroll = 0
                        self.approval_preview_visible = False
                        self.state.status = "approval"
                elif kind == "error":
                    failed_turn = self.active_turn
                    queued_turns = tuple(self.followups)
                    self.worker = None
                    self.worker_kind = None
                    self.cancellation = None
                    self.state.status = "failed"
                    self.state.assistant_draft = ""
                    self.active_turn = None
                    self.followups.clear()
                    self.pending_approval = None
                    self.approval_scroll = 0
                    self.approval_preview_visible = False
                    self.approval_broker = None
                    try:
                        persisted = self.runner.store.load(self.state.session_id)
                    except SessionStoreError as refresh_error:
                        self.state.notice(f"会话刷新失败：{refresh_error}")
                    else:
                        self.session = dict(persisted)
                        self.transcript = [
                            dict(item) for item in persisted.get("messages", [])
                        ]
                        referenced = {
                            str(item.get("attachment_id"))
                            for message in persisted.get("messages", [])
                            if isinstance(message, Mapping)
                            for item in message.get("attachments", [])
                            if isinstance(item, Mapping)
                            and isinstance(item.get("attachment_id"), str)
                        }
                        recovery_turns = (
                            ((failed_turn,) if failed_turn is not None else ())
                            + queued_turns
                        )
                        if recovery_turns:
                            pending_ids = {
                                item.attachment_id for item in self.pending_attachments
                            }
                            recovered = 0
                            for recovery_turn in recovery_turns:
                                for descriptor in recovery_turn.attachments:
                                    if (
                                        descriptor.attachment_id not in referenced
                                        and descriptor.attachment_id not in pending_ids
                                    ):
                                        self.pending_attachments.append(descriptor)
                                        pending_ids.add(descriptor.attachment_id)
                                        recovered += 1
                            if recovered:
                                self.state.notice(
                                    f"失败/排队轮次的 {recovered} 个未持久化附件"
                                    "已恢复到待发送区"
                                )
                    self.state.notice(f"运行失败：{type(payload).__name__}: {payload}")
            except Exception as exc:
                if self.cancellation is not None:
                    self.cancellation.cancel("tui_event_contract_violation")
                self.state.notice(f"事件流被拒绝：{exc}")

    def _replace_session(self, session: Mapping[str, Any]) -> None:
        self.session = dict(session)
        self.state = TuiState(session_id=str(session["session_id"]))
        self.transcript = [dict(item) for item in session.get("messages", [])]
        self.runner.permission_mode = str(session.get("permission_mode", "read-only"))
        self.scroll = 0
        self.pending_approval = None
        self.approval_scroll = 0
        self.approval_preview_visible = False
        self.approval_broker = None
        self.followups.clear()
        self.active_turn = None
        for notice in session.get("risk_notices", [])[-8:]:
            self.state.notice(str(notice))

    def _session_switch_blocked(self) -> bool:
        if not self.pending_attachments:
            return False
        self.state.notice("仍有待发送附件；请先发送或使用 /detach all")
        return True

    def _discard_pending_attachments(self) -> bool:
        failed: list[AttachmentDescriptor] = []
        for descriptor in self.pending_attachments:
            try:
                self.runner.discard_attachment(descriptor)
            except Exception:
                failed.append(descriptor)
        removed = len(self.pending_attachments) - len(failed)
        self.pending_attachments = failed
        if failed:
            self.state.notice("部分附件无法安全删除，已保留在待发送区")
            return False
        if removed:
            self.state.notice(f"已删除 {removed} 个待发送附件快照")
        return True

    def _hook_status(self) -> dict[str, Any]:
        status_loader = getattr(self.runner, "hook_status", None)
        if callable(status_loader):
            status = status_loader()
        else:
            snapshot = load_project_hooks(self.runner.workspace)
            status = snapshot.metadata(self.runner.store.hook_trust_state())
        if not isinstance(status, Mapping):
            raise ValueError("hook status must be an object")
        return dict(status)

    def _subagent_artifacts(self) -> tuple[tuple[str, str], ...]:
        """Load only opaque artifact IDs and phases through the public seam."""

        loader = getattr(self.runner, "subagent_artifacts", None)
        if not callable(loader):
            raise ValueError("subagent artifact inventory is unavailable")
        raw = loader(reveal_paths=False)
        if not isinstance(raw, Mapping):
            raise ValueError("subagent artifact inventory is invalid")
        entries = raw.get("artifacts")
        if not isinstance(entries, list) or len(entries) > 64:
            raise ValueError("subagent artifact inventory is invalid")
        artifacts: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in entries:
            if not isinstance(item, Mapping):
                raise ValueError("subagent artifact inventory is invalid")
            artifact_id = item.get("artifact_id", item.get("worktree_id"))
            phase = item.get("phase", item.get("status", "unknown"))
            if (
                not isinstance(artifact_id, str)
                or _OPAQUE_ARTIFACT_ID.fullmatch(artifact_id) is None
                or artifact_id in seen
                or not isinstance(phase, str)
                or _ARTIFACT_PHASE.fullmatch(phase) is None
            ):
                raise ValueError("subagent artifact inventory is invalid")
            # Unknown fields (including any accidental path/ref/content field)
            # are never copied, formatted, or retained by the TUI.
            seen.add(artifact_id)
            artifacts.append((artifact_id, phase))
        return tuple(sorted(artifacts))

    def command(self, raw: str) -> None:
        command, _, argument = raw.strip().partition(" ")
        command = command.casefold()
        argument = argument.strip()
        if command in {"/quit", "/exit"}:
            if self.busy:
                self.state.notice("运行中请先按 Ctrl+C 停止")
            elif self._discard_pending_attachments():
                self.running = False
        elif command == "/help":
            self.state.notice(
                "/new /sessions /resume ID /fork /archive /effects "
                "/reconcile RUN_ID /status /model /permissions MODE /tools "
                "/agents /instructions /hooks /mcp /context /compact "
                "/attach PATH /attachments /detach ID|all "
                "/approvals /clear /quit"
            )
        elif command == "/new":
            if self.busy:
                self.state.notice("运行完成后才能新建会话")
            elif self._session_switch_blocked():
                return
            else:
                self._replace_session(self.runner.new_session())
                self.state.notice("已新建会话")
        elif command == "/sessions":
            sessions = self.runner.store.list(include_archived=True)[:12]
            if not sessions:
                self.state.notice("没有保存的会话")
            for item in reversed(sessions):
                archive = " [archived]" if item.get("archived") else ""
                self.state.notice(f"{item['session_id']} · {item['title']}{archive}")
        elif command == "/resume":
            if self.busy:
                self.state.notice("运行完成后才能切换会话")
            elif self._session_switch_blocked():
                return
            else:
                try:
                    self._replace_session(
                        self.runner.resume_session(
                            argument or None,
                            permission_override="read-only",
                        )
                    )
                    self.state.notice(f"已恢复 {self.state.session_id}")
                except SessionStoreError as exc:
                    self.state.notice(str(exc))
        elif command == "/fork":
            if self.busy:
                self.state.notice("运行完成后才能 fork")
            elif self._session_switch_blocked():
                return
            else:
                try:
                    self._replace_session(self.runner.store.fork(self.state.session_id))
                    self.state.notice("已 fork 当前会话")
                except SessionStoreError as exc:
                    self.state.notice(str(exc))
        elif command == "/archive":
            if self.busy:
                self.state.notice("运行完成后才能归档")
            elif self._session_switch_blocked():
                return
            else:
                self.runner.store.archive(self.state.session_id)
                self._replace_session(self.runner.new_session())
                self.state.notice("原会话已归档")
        elif command == "/effects":
            unresolved = self.runner.store.unresolved_workspace_runs()
            if not unresolved:
                self.state.notice("当前工作区没有待核对运行")
            for item in unresolved[-12:]:
                self.state.notice(
                    f"{item['run_id']} · session={item['session_id']} · {item['status']}"
                )
        elif command == "/reconcile":
            if self.busy:
                self.state.notice("运行中不能核销待核对运行")
            elif not argument:
                self.state.notice("用法：/reconcile RUN_ID")
            else:
                try:
                    result = self.runner.store.acknowledge_run(argument)
                    self.state.notice(f"已人工确认 {result['run_id']}")
                except SessionStoreError as exc:
                    self.state.notice(str(exc))
        elif command == "/status":
            status = self.runner.provider_status
            sandbox = status.get("workspace_sandbox", {})
            sandbox_label = (
                sandbox.get("backend", "unknown")
                if isinstance(sandbox, Mapping)
                else "unknown"
            )
            self.state.notice(
                f"session={self.state.session_id} provider={status['provider']} "
                f"model={status['model']} permissions={self.runner.permission_mode} "
                f"sandbox={sandbox_label} cwd={status['workspace']}"
            )
        elif command == "/model":
            self.state.notice(f"当前模型：{self.runner.client.config.model}")
        elif command == "/permissions":
            if not argument:
                self.state.notice("权限模式：" + " / ".join(PERMISSION_PROFILES))
            elif self.busy:
                self.state.notice("运行中不能切换权限")
            elif argument not in PERMISSION_PROFILES:
                self.state.notice("未知权限模式：" + argument)
            else:
                self.session = self.runner.set_permission_mode(self.state.session_id, argument)
                self.state.notice(f"权限已切换为 {argument}")
        elif command == "/tools":
            allowed = permission_profile(self.runner.permission_mode)
            names = [spec.name for spec in self.runner.registry.specs() if spec.permission in allowed]
            self.state.notice("可用工具：" + ", ".join(names))
        elif command == "/attach":
            if not argument:
                self.state.notice("用法：/attach PATH")
                return
            if len(self.pending_attachments) >= MAX_ATTACHMENTS_PER_TURN:
                self.state.notice("待发送附件已达 8 个；请先发送或移除")
                return
            loader = getattr(self.runner, "ingest_attachments", None)
            if not callable(loader):
                self.state.notice("当前运行器不支持附件")
                return
            try:
                imported = loader([argument])
                if not isinstance(imported, tuple) or len(imported) != 1:
                    raise AttachmentError("attachment importer returned invalid data")
                descriptor = imported[0]
                if not isinstance(descriptor, AttachmentDescriptor):
                    raise AttachmentError("attachment importer returned invalid data")
                projected = sum(
                    item.size_bytes for item in self.pending_attachments
                ) + descriptor.size_bytes
                if projected > MAX_ATTACHMENTS_TOTAL_BYTES:
                    try:
                        self.runner.discard_attachment(descriptor)
                    except Exception:
                        self.pending_attachments.append(descriptor)
                        self.state.notice(
                            "附件超过单轮 24 MiB 且无法安全回收；已保留，请 /detach"
                        )
                    else:
                        self.state.notice("待发送附件总量不能超过 24 MiB")
                    return
                self.pending_attachments.append(descriptor)
                self.state.notice(
                    f"已附加 {descriptor.display_name} · {descriptor.kind} · "
                    f"{descriptor.size_bytes} bytes · {descriptor.attachment_id}"
                )
            except AttachmentError as exc:
                self.state.notice(f"附件导入失败：{exc}")
        elif command == "/attachments":
            if argument:
                self.state.notice("用法：/attachments")
                return
            total = sum(item.size_bytes for item in self.pending_attachments)
            self.state.notice(
                f"待发送附件：{len(self.pending_attachments)} 个 · {total} bytes"
            )
            for descriptor in self.pending_attachments:
                self.state.notice(
                    f"{descriptor.attachment_id} · {descriptor.kind} · "
                    f"{descriptor.size_bytes} bytes · {descriptor.display_name}"
                )
        elif command == "/detach":
            if not argument:
                self.state.notice("用法：/detach ID|all")
                return
            if argument.casefold() == "all":
                self._discard_pending_attachments()
                return
            matches = [
                item
                for item in self.pending_attachments
                if item.attachment_id == argument
            ]
            if len(matches) != 1:
                self.state.notice("未找到这个待发送附件 ID")
                return
            descriptor = matches[0]
            try:
                self.runner.discard_attachment(descriptor)
            except Exception:
                self.state.notice("附件无法安全删除，仍保留在待发送区")
                return
            self.pending_attachments.remove(descriptor)
            self.state.notice(f"已移除 {descriptor.attachment_id}")
        elif command == "/agents":
            if argument:
                self.state.notice("用法：/agents（仅显示不含路径的活动与 artifact 元数据）")
                return
            activities = self.state.ordered_agents()
            self.state.notice(
                f"前台 Agents：{self.state.running_agent_count} running · "
                f"{self.state.settled_agent_count} settled · {len(activities)} total"
            )
            for activity in activities[-12:]:
                artifact = activity.artifact_id or "none"
                changed = (
                    "unknown"
                    if activity.changed is None
                    else ("yes" if activity.changed else "no")
                )
                self.state.notice(
                    f"{activity.agent_id} · {activity.status} · "
                    f"depth={activity.depth} · ordinal={activity.ordinal} · "
                    f"artifact={artifact} · changed={changed}"
                )
            try:
                artifacts = self._subagent_artifacts()
            except Exception:
                # Exception text can contain a worktree path. Keep this boundary
                # content-free and bounded even when an adapter misbehaves.
                self.state.notice("Agent artifact 清单不可用")
            else:
                if not artifacts:
                    self.state.notice("保留 artifacts：0")
                else:
                    self.state.notice(f"保留 artifacts：{len(artifacts)}")
                    for artifact_id, phase in artifacts[-12:]:
                        self.state.notice(f"artifact={artifact_id} · phase={phase}")
        elif command == "/instructions":
            try:
                snapshot = self.runner.instruction_snapshot()
                self.state.notice(
                    f"项目指令：{len(snapshot.documents)} 个文件 · "
                    f"{snapshot.total_bytes} bytes · {snapshot.snapshot_sha256[:12]}"
                )
                for document in snapshot.documents[-12:]:
                    self.state.notice(
                        f"{document.relative_path} · {document.byte_length} bytes · "
                        f"{document.content_sha256[:12]}"
                    )
            except Exception as exc:
                self.state.notice(f"项目指令不可用：{exc}")
        elif command == "/hooks":
            if argument:
                self.state.notice("用法：/hooks（只读；信任或禁用请使用 harness hooks）")
                return
            try:
                status = self._hook_status()
                if not status.get("config_present"):
                    self.state.notice(
                        "未配置项目 hooks：.agent-harness/hooks.json"
                    )
                    return
                hooks = status.get("hooks", [])
                if not isinstance(hooks, list):
                    raise ValueError("hook status list is invalid")
                self.state.notice(
                    f"项目 hooks：{status.get('hook_count', 0)} configured · "
                    f"{status.get('trusted_hook_count', 0)} trusted · "
                    f"{status.get('disabled_hook_count', 0)} disabled · "
                    f"{status.get('pending_hook_count', 0)} review-required · "
                    f"snapshot {str(status.get('snapshot_sha256', ''))[:12]}"
                )
                for item in hooks[:5]:
                    if not isinstance(item, Mapping):
                        raise ValueError("hook status entry is invalid")
                    self.state.notice(
                        f"{item.get('hook_id', 'hook')} · "
                        f"{item.get('event_name', 'event')} · "
                        f"{item.get('trust_status', 'unknown')} · "
                        f"{str(item.get('definition_sha256', ''))[:12]}"
                    )
                if len(hooks) > 5:
                    self.state.notice(
                        f"另有 {len(hooks) - 5} 个；运行 harness hooks 查看完整列表"
                    )
                if int(status.get("pending_hook_count", 0)) > 0:
                    self.state.notice(
                        "信任或禁用仅通过显式 CLI：harness hooks"
                    )
            except Exception as exc:
                self.state.notice(f"项目 hooks 不可用：{exc}")
        elif command == "/mcp":
            if argument:
                self.state.notice("用法：/mcp（只读；信任与刷新请使用 harness mcp）")
                return
            try:
                status = self.runner.mcp_status()
                if not status.get("config_present"):
                    self.state.notice("未配置项目 MCP：.agent-harness/mcp.json")
                    return
                self.state.notice(
                    f"项目 MCP：{status.get('server_count', 0)} configured · "
                    f"{status.get('trusted_server_count', 0)} trusted · "
                    f"{status.get('ready_server_count', 0)} ready · "
                    f"{status.get('pending_server_count', 0)} review-required · "
                    f"snapshot {str(status.get('snapshot_sha256', ''))[:12]}"
                )
                servers = status.get("servers", [])
                if not isinstance(servers, list):
                    raise ValueError("MCP status list is invalid")
                for item in servers[:8]:
                    if not isinstance(item, Mapping):
                        raise ValueError("MCP status entry is invalid")
                    self.state.notice(
                        f"{item.get('server_id', 'server')} · "
                        f"{item.get('trust_status', 'unknown')}/"
                        f"{item.get('catalog_status', 'unknown')} · "
                        f"{item.get('tool_count', 0)} tools · "
                        f"{str(item.get('definition_sha256', ''))[:12]}"
                    )
                if len(servers) > 8:
                    self.state.notice(
                        f"另有 {len(servers) - 8} 个；运行 harness mcp 查看完整定义"
                    )
                if int(status.get("pending_server_count", 0)) > 0 or int(
                    status.get("refresh_required_count", 0)
                ) > 0:
                    self.state.notice("信任、禁用与 catalog 刷新仅通过显式 CLI：harness mcp")
            except Exception as exc:
                self.state.notice(f"项目 MCP 不可用：{exc}")
        elif command == "/context":
            try:
                status = self.runner.context_status(self.state.session_id)
                self.state.notice(
                    f"上下文：原始 {status['transcript_message_count']} · "
                    f"已压缩 {status['compacted_message_count']} · "
                    f"活动 {status['active_message_count']} messages · "
                    f"估算 {status['estimated_input_tokens_upper_bound']}/"
                    f"{status['available_input_tokens']} tokens"
                )
                if status["compaction_id"]:
                    self.state.notice(
                        f"摘要：{status['summary_chars']} chars · "
                        f"{status['summary_sha256'][:12]} · {status['compaction_id']}"
                    )
            except Exception as exc:
                self.state.notice(f"上下文诊断不可用：{exc}")
        elif command == "/compact":
            self.start_compaction()
        elif command == "/approvals":
            try:
                if argument.startswith("clear "):
                    scope = argument.partition(" ")[2].strip().casefold()
                    if scope not in {"session", "workspace"}:
                        self.state.notice(
                            "用法：/approvals 或 /approvals clear session|workspace"
                        )
                        return
                    if self.busy:
                        self.state.notice("运行中不能清除审批规则")
                        return
                    removed = self.runner.store.clear_approval_rules(
                        scope=scope,
                        session_id=self.state.session_id,
                    )
                    self.state.notice(f"已清除 {removed} 条 {scope} 审批规则")
                    return
                rules = self.runner.store.approval_rules(self.state.session_id)
                self.state.notice(
                    "审批：low=allow · medium/high=ask · "
                    f"persistent rules={len(rules)}"
                )
                for rule in rules[-12:]:
                    scope = "exact" if rule.exact else "tool"
                    self.state.notice(f"{rule.action} · {rule.tool_name} · {scope}")
            except SessionStoreError as exc:
                self.state.notice(f"审批规则不可用：{exc}")
        elif command == "/clear":
            self.state.notices.clear()
            self.state.tools.clear()
            self.scroll = 0
        else:
            self.state.notice(f"未知命令：{command}；输入 /help 查看命令")

    def submit(self) -> None:
        value = self.input_buffer.strip()
        self.input_buffer = ""
        if not value:
            return
        if value.startswith("/"):
            self.command(value)
        else:
            attachments = tuple(self.pending_attachments)
            if self.start_turn(value, attachments):
                self.pending_attachments.clear()

    def handle_key(self, key: Any) -> None:
        if key == curses.KEY_RESIZE:
            return
        if self.pending_approval is not None:
            self._handle_approval_key(key)
            return
        if key in ("\n", "\r", curses.KEY_ENTER):
            self.submit()
        elif key in ("\x03", 3):
            self._handle_interrupt()
        elif key in ("\x04", 4):
            if not self.busy and not self.input_buffer:
                if self._discard_pending_attachments():
                    self.running = False
        elif key in (curses.KEY_BACKSPACE, "\b", "\x7f", 127, 8):
            self.input_buffer = self.input_buffer[:-1]
        elif key == curses.KEY_PPAGE:
            self.scroll += 5
        elif key == curses.KEY_NPAGE:
            self.scroll = max(0, self.scroll - 5)
        elif isinstance(key, str) and key.isprintable():
            if len(self.input_buffer) < 20_000:
                self.input_buffer += key

    def _handle_interrupt(self) -> None:
        if self.busy and self.cancellation is not None:
            self.cancellation.cancel("user_requested")
            self.state.notice("正在停止当前运行…")
        else:
            self.input_buffer = ""

    def _handle_approval_key(self, key: Any) -> None:
        pending = self.pending_approval
        if pending is None:
            return
        normalized = key.casefold() if isinstance(key, str) else key
        if normalized in (curses.KEY_UP, curses.KEY_PPAGE, "k"):
            self.approval_scroll = max(0, self.approval_scroll - 3)
            return
        if normalized in (curses.KEY_DOWN, curses.KEY_NPAGE, "j"):
            self.approval_scroll += 3
            return
        if normalized in ("\x03", 3):
            pending.settle(verdict="deny", reason_code="user_denied_once")
            self.pending_approval = None
            self.approval_scroll = 0
            self.approval_preview_visible = False
            self._handle_interrupt()
            return
        if normalized == "n":
            pending.settle(verdict="deny", reason_code="user_denied_once")
            self.pending_approval = None
            self.approval_scroll = 0
            self.approval_preview_visible = False
            return
        if normalized in {"y", "s", "w"} and not self.approval_preview_visible:
            self.state.notice("审批预览尚不可见；请放大终端并检查完整参数后再批准")
            return
        if normalized == "y":
            pending.settle(verdict="allow", reason_code="user_allowed_once")
            self.pending_approval = None
            self.approval_scroll = 0
            self.approval_preview_visible = False
            return
        if normalized not in {"s", "w"}:
            return
        if not pending.request.persistent_scope_allowed:
            self.state.notice(
                "此工具要求逐次审批；本次不能保存会话或工作区放行规则"
            )
            return
        rule = ApprovalRule(
            action="allow",
            tool_name=pending.request.tool_name,
            tool_version=pending.request.tool_version,
            arguments_sha256=pending.request.arguments_sha256,
        )
        scope = "session" if normalized == "s" else "workspace"
        try:
            self.runner.store.add_approval_rule(
                rule,
                scope=scope,
                session_id=self.state.session_id,
            )
        except SessionStoreError as exc:
            self.state.notice(f"审批规则保存失败：{exc}")
            return
        self.turn_approval_rules.append(rule)
        pending.settle(
            verdict="allow",
            reason_code=(
                "user_allowed_session"
                if scope == "session"
                else "user_allowed_workspace"
            ),
        )
        self.pending_approval = None
        self.approval_scroll = 0
        self.approval_preview_visible = False
        self.state.notice(f"已保存 {scope} 精确审批规则")

    def _body_lines(self, width: int) -> list[str]:
        lines: list[str] = []
        for message in self.transcript:
            role = "You" if message.get("role") == "user" else "Agent"
            content = sanitize_terminal_text(message.get("content", ""))
            wrapped = wrap_display(content, max(10, width - 4))
            lines.append(f"{role} › {wrapped[0]}")
            lines.extend(f"    {line}" for line in wrapped[1:])
            attachments = message.get("attachments", [])
            if isinstance(attachments, list) and attachments:
                labels: list[str] = []
                for item in attachments[:8]:
                    if not isinstance(item, Mapping):
                        continue
                    name = sanitize_terminal_text(item.get("display_name", "attachment"))
                    kind = sanitize_terminal_text(item.get("kind", "file"))
                    labels.append(f"{name} ({kind})")
                if labels:
                    lines.append("    attachments › " + ", ".join(labels))
            lines.append("")
        if self.state.assistant_draft:
            wrapped = wrap_display(self.state.assistant_draft, max(10, width - 4))
            lines.append(f"Agent › {wrapped[0]}")
            lines.extend(f"    {line}" for line in wrapped[1:])
        if self.state.tools:
            lines.append("")
            for tool in list(self.state.tools.values())[-8:]:
                detail = f" · {tool.detail}" if tool.detail else ""
                lines.append(f"  [{tool.phase}] {tool.name}{detail}")
        if self.state.agents:
            lines.append("")
            lines.append(
                f"  Foreground Agents · {self.state.running_agent_count} running · "
                f"{self.state.settled_agent_count} settled"
            )
            for activity in self.state.ordered_agents()[-8:]:
                lines.append(
                    f"  [{activity.status}] {activity.agent_id} · "
                    f"depth {activity.depth} · #{activity.ordinal + 1}"
                )
        if self.state.notices:
            lines.append("")
            lines.extend(f"  ! {notice}" for notice in self.state.notices[-8:])
        return lines

    @staticmethod
    def _add(screen: Any, row: int, column: int, text: str, width: int, attribute: int = 0) -> None:
        try:
            screen.addstr(
                row,
                column,
                truncate_display(sanitize_terminal_text(text), max(0, width)),
                attribute,
            )
        except curses.error:
            pass

    def render(self, screen: Any) -> None:
        rows, columns = screen.getmaxyx()
        screen.erase()
        self.approval_preview_visible = False
        minimum_rows = 12 if self.pending_approval is not None else 8
        if rows < minimum_rows or columns < 40:
            size = f"40×{minimum_rows}"
            message = f"终端过小：至少需要 {size}"
            if self.pending_approval is not None:
                message += "；可按 n 拒绝，放大后才能批准"
            self._add(screen, 0, 0, message, columns - 1, curses.A_BOLD)
            screen.refresh()
            return
        header = (
            f" Agent Harness  fg-agents {self.state.running_agent_count}/"
            f"{len(self.state.agents)}  {self.runner.client.config.model}  "
            f"{self.runner.permission_mode}  att {len(self.pending_attachments)}  "
            f"{self.runner.workspace.name} "
        )
        self._add(screen, 0, 0, header.ljust(columns - 1), columns - 1, curses.A_REVERSE)
        self._add(screen, 1, 0, "─" * (columns - 1), columns - 1, curses.A_DIM)
        body_top = 2
        approval_height = (
            min(10, max(5, rows // 3)) if self.pending_approval is not None else 0
        )
        body_height = max(1, rows - 6 - approval_height)
        all_lines = self._body_lines(columns - 1)
        maximum_scroll = max(0, len(all_lines) - body_height)
        self.scroll = min(self.scroll, maximum_scroll)
        end = len(all_lines) - self.scroll
        start = max(0, end - body_height)
        for offset, line in enumerate(all_lines[start:end]):
            self._add(screen, body_top + offset, 0, line, columns - 1)
        if self.pending_approval is not None:
            pending = self.pending_approval
            modal_top = body_top + body_height
            preview_lines = wrap_display(pending.preview or "(no preview)", columns - 4)
            preview_height = max(1, approval_height - 2)
            maximum_approval_scroll = max(0, len(preview_lines) - preview_height)
            self.approval_scroll = min(
                max(0, self.approval_scroll), maximum_approval_scroll
            )
            preview_end = self.approval_scroll + preview_height
            position = (
                f" · lines {self.approval_scroll + 1}-"
                f"{min(preview_end, len(preview_lines))}/{len(preview_lines)}"
                if len(preview_lines) > preview_height
                else ""
            )
            self._add(
                screen,
                modal_top,
                0,
                (
                    f" APPROVAL · {pending.request.tool_name} · {pending.request.risk} · "
                    f"args {pending.request.arguments_sha256[:12]}{position} "
                ).ljust(columns - 1),
                columns - 1,
                curses.A_REVERSE,
            )
            for offset, preview in enumerate(
                preview_lines[self.approval_scroll:preview_end]
            ):
                self._add(
                    screen,
                    modal_top + 1 + offset,
                    2,
                    preview,
                    columns - 4,
                )
            approval_help = (
                "↑↓/PgUp/PgDn 查看 · y 本次 · n 拒绝"
                if not pending.request.persistent_scope_allowed
                else "↑↓/PgUp/PgDn 查看 · y 本次 · s 会话 · w 工作区 · n 拒绝"
            )
            self._add(
                screen,
                modal_top + approval_height - 1,
                2,
                approval_help,
                columns - 4,
                curses.A_BOLD,
            )
            self.approval_preview_visible = True
        total = self.state.usage.get("total_tokens", 0)
        cache_hit = self.state.usage.get("prompt_cache_hit_tokens", 0)
        cache_total = cache_hit + self.state.usage.get("prompt_cache_miss_tokens", 0)
        cache = f"cache {round(cache_hit * 100 / cache_total)}%" if cache_total else "cache —"
        queue_label = f" · queued {len(self.followups)}" if self.followups else ""
        attachment_label = (
            f" · attachments {len(self.pending_attachments)}"
            if self.pending_attachments
            else " · attachments 0"
        )
        approval_label = " · approval waiting" if self.pending_approval else ""
        agents_label = (
            f" · fg-agents {self.state.running_agent_count} running/"
            f"{self.state.settled_agent_count} settled"
            if self.state.agents
            else " · fg-agents 0"
        )
        status = (
            f" {self.state.status}{agents_label} · {total} tokens · {cache}"
            f"{queue_label}{attachment_label}{approval_label} "
        )
        self._add(screen, rows - 4, 0, status.ljust(columns - 1), columns - 1, curses.A_REVERSE)
        self._add(screen, rows - 3, 0, "─" * (columns - 1), columns - 1, curses.A_DIM)
        prompt = "> " + self.input_buffer
        self._add(screen, rows - 2, 0, prompt, columns - 1, curses.A_BOLD)
        self._add(
            screen,
            rows - 1,
            0,
            "Enter 发送 · /attach PATH · Ctrl+C 停止 · PgUp/PgDn 滚动 · /help",
            columns - 1,
            curses.A_DIM,
        )
        cursor_column = min(columns - 2, display_width(prompt))
        try:
            screen.move(rows - 2, cursor_column)
        except curses.error:
            pass
        screen.refresh()

    def run(self, screen: Any) -> int:
        try:
            curses.curs_set(1)
        except curses.error:
            # Dumb terminals and some PTY test harnesses do not advertise a
            # mutable cursor mode. The UI remains fully usable without it.
            pass
        screen.keypad(True)
        screen.nodelay(True)
        while self.running:
            try:
                self.drain()
                self.render(screen)
                try:
                    key = screen.get_wch()
                except curses.error:
                    key = None
                if key is not None:
                    self.handle_key(key)
                time.sleep(0.03)
            except KeyboardInterrupt:
                # curses cbreak mode commonly delivers Ctrl+C as SIGINT
                # instead of a key code. Convert it to the same monotonic
                # cancellation action and keep the TUI alive.
                self._handle_interrupt()
        if self.busy and self.cancellation is not None:
            self.cancellation.cancel("tui_exit")
        return 0


def run_tui(
    runner: AgentRunner,
    *,
    resume: str | None = None,
    resume_latest: bool = False,
    permission_override: str = "read-only",
) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("TUI requires an interactive terminal; use `harness exec` instead")
    session = (
        runner.resume_session(
            resume,
            permission_override=permission_override,
        )
        if resume is not None or resume_latest
        else runner.new_session()
    )
    application = HarnessTui(runner, session)
    return curses.wrapper(application.run)


class _TuiArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _TuiArgumentParser(prog="harness-tui", description="Agent Harness terminal UI")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--active-directory")
    parser.add_argument("--resume", nargs="?", const="")
    parser.add_argument("--model")
    parser.add_argument("--permissions", choices=tuple(PERMISSION_PROFILES), default=None)
    parser.add_argument("--api-key-file")
    parser.add_argument("--state-home")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        runner = AgentRunner(
            Path(args.cwd),
            active_directory=args.active_directory,
            state_home=args.state_home,
            api_key_file=args.api_key_file,
            model=args.model,
            permission_mode=args.permissions or "read-only",
        )
        resume = args.resume or None
        return run_tui(
            runner,
            resume=resume,
            resume_latest=args.resume is not None,
            permission_override=args.permissions or "read-only",
        )
    except (SessionStoreError, RuntimeError, ValueError) as exc:
        print(f"harness-tui: {exc}", file=sys.stderr)
        return 64


__all__ = ["HarnessTui", "main", "run_tui"]
