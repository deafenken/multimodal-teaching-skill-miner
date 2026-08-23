"""Standard-library curses TUI for Agent Harness."""

from __future__ import annotations

import argparse
import curses
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Thread
import time
from typing import Any, Mapping, Sequence

from .core import CancellationToken
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
        self.cancellation: CancellationToken | None = None
        self.followups: list[str] = []
        self.running = True

    @property
    def busy(self) -> bool:
        # A finished thread remains authoritative until its queued outcome has
        # been reduced. Treat that settlement window as busy so a new run
        # cannot be overwritten by the previous outcome.
        return self.worker is not None

    def _event_sink(self, event: Mapping[str, Any]) -> None:
        self.queue.put(("event", dict(event)))

    def start_turn(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            return
        if self.busy:
            if (
                len(self.followups) >= _MAX_FOLLOWUPS
                or sum(len(item) for item in self.followups) + len(prompt)
                > _MAX_FOLLOWUP_CHARS
            ):
                self.state.notice("后续队列已满；请等待当前运行完成")
                return
            self.followups.append(prompt)
            self.state.notice(f"已加入后续队列（{len(self.followups)}）")
            return
        self.transcript.append({"role": "user", "content": prompt})
        self.state.begin_turn()
        self.cancellation = CancellationToken()

        def work() -> None:
            try:
                outcome = self.runner.run_turn(
                    self.state.session_id,
                    prompt,
                    cancellation_token=self.cancellation,
                    event_sink=self._event_sink,
                )
                self.queue.put(("outcome", outcome))
            except BaseException as exc:
                self.queue.put(("error", exc))

        self.worker = Thread(target=work, name="agent-harness-turn", daemon=True)
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
        if outcome.message:
            self.transcript.append(
                {"role": "assistant", "content": outcome.message, "run_id": outcome.run_id}
            )
        elif outcome.status != "cancelled":
            self.state.notice(f"运行未生成回答：{outcome.status} · {outcome.reason}")
        self.state.assistant_draft = ""
        self.worker = None
        self.cancellation = None
        if self.followups:
            next_prompt = self.followups.pop(0)
            self.start_turn(next_prompt)

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
                elif kind == "error":
                    self.worker = None
                    self.cancellation = None
                    self.state.status = "failed"
                    self.state.assistant_draft = ""
                    self.followups.clear()
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
        for notice in session.get("risk_notices", [])[-8:]:
            self.state.notice(str(notice))

    def command(self, raw: str) -> None:
        command, _, argument = raw.strip().partition(" ")
        command = command.casefold()
        argument = argument.strip()
        if command in {"/quit", "/exit"}:
            if self.busy:
                self.state.notice("运行中请先按 Ctrl+C 停止")
            else:
                self.running = False
        elif command == "/help":
            self.state.notice(
                "/new /sessions /resume ID /fork /archive /effects "
                "/reconcile RUN_ID /status /model /permissions MODE /tools /clear /quit"
            )
        elif command == "/new":
            if self.busy:
                self.state.notice("运行完成后才能新建会话")
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
            else:
                try:
                    self._replace_session(self.runner.store.fork(self.state.session_id))
                    self.state.notice("已 fork 当前会话")
                except SessionStoreError as exc:
                    self.state.notice(str(exc))
        elif command == "/archive":
            if self.busy:
                self.state.notice("运行完成后才能归档")
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
            self.state.notice(
                f"session={self.state.session_id} provider={status['provider']} model={status['model']} permissions={self.runner.permission_mode} cwd={status['workspace']}"
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
            self.start_turn(value)

    def handle_key(self, key: Any) -> None:
        if key == curses.KEY_RESIZE:
            return
        if key in ("\n", "\r", curses.KEY_ENTER):
            self.submit()
        elif key in ("\x03", 3):
            self._handle_interrupt()
        elif key in ("\x04", 4):
            if not self.busy and not self.input_buffer:
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

    def _body_lines(self, width: int) -> list[str]:
        lines: list[str] = []
        for message in self.transcript:
            role = "You" if message.get("role") == "user" else "Agent"
            content = sanitize_terminal_text(message.get("content", ""))
            wrapped = wrap_display(content, max(10, width - 4))
            lines.append(f"{role} › {wrapped[0]}")
            lines.extend(f"    {line}" for line in wrapped[1:])
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
        if rows < 8 or columns < 40:
            self._add(screen, 0, 0, "终端过小：至少需要 40×8", columns - 1, curses.A_BOLD)
            screen.refresh()
            return
        header = (
            f" Agent Harness  {self.runner.client.config.model}  "
            f"{self.runner.permission_mode}  {self.runner.workspace.name} "
        )
        self._add(screen, 0, 0, header.ljust(columns - 1), columns - 1, curses.A_REVERSE)
        self._add(screen, 1, 0, "─" * (columns - 1), columns - 1, curses.A_DIM)
        body_top = 2
        body_height = rows - 6
        all_lines = self._body_lines(columns - 1)
        maximum_scroll = max(0, len(all_lines) - body_height)
        self.scroll = min(self.scroll, maximum_scroll)
        end = len(all_lines) - self.scroll
        start = max(0, end - body_height)
        for offset, line in enumerate(all_lines[start:end]):
            self._add(screen, body_top + offset, 0, line, columns - 1)
        total = self.state.usage.get("total_tokens", 0)
        cache_hit = self.state.usage.get("prompt_cache_hit_tokens", 0)
        cache_total = cache_hit + self.state.usage.get("prompt_cache_miss_tokens", 0)
        cache = f"cache {round(cache_hit * 100 / cache_total)}%" if cache_total else "cache —"
        queue_label = f" · queued {len(self.followups)}" if self.followups else ""
        status = f" {self.state.status} · {total} tokens · {cache}{queue_label} "
        self._add(screen, rows - 4, 0, status.ljust(columns - 1), columns - 1, curses.A_REVERSE)
        self._add(screen, rows - 3, 0, "─" * (columns - 1), columns - 1, curses.A_DIM)
        prompt = "> " + self.input_buffer
        self._add(screen, rows - 2, 0, prompt, columns - 1, curses.A_BOLD)
        self._add(screen, rows - 1, 0, "Enter 发送 · Ctrl+C 停止 · PgUp/PgDn 滚动 · /help", columns - 1, curses.A_DIM)
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
