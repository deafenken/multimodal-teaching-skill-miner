"""Command-line and headless interfaces for Agent Harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
from typing import Any, Mapping, Sequence

from .core import CancellationToken
from .providers import DeepSeekClientError, DeepSeekConfigurationError
from .runner import AgentRunner
from .session import SessionStore, SessionStoreError
from .toolsets import PERMISSION_PROFILES
from .tui import run_tui
from .tui_state import sanitize_terminal_text


EXIT_COMPLETED = 0
EXIT_CANCELLED = 2
EXIT_HANDOFF = 3
EXIT_FAILED = 4
EXIT_USAGE = 64
EXIT_CONFIG = 78


class CliUsageError(ValueError):
    """Raised for command-line syntax errors with a stable exit code."""


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _runner(args: argparse.Namespace) -> AgentRunner:
    return AgentRunner(
        Path(args.cwd),
        state_home=args.state_home,
        api_key_file=args.api_key_file,
        model=args.model,
        permission_mode=args.permissions or "read-only",
        deadline_seconds=args.deadline,
        max_steps=args.max_steps,
    )


def _exec(args: argparse.Namespace, runner: AgentRunner) -> int:
    if args.resume is not None and args.resume_latest:
        raise CliUsageError("--resume and --resume-latest are mutually exclusive")
    prompt = " ".join(args.prompt).strip()
    if prompt == "-":
        prompt = sys.stdin.read().strip()
    if not prompt:
        raise ValueError("exec requires a non-empty prompt")
    session = (
        runner.resume_session(
            args.resume,
            permission_override=args.permissions or "read-only",
        )
        if args.resume is not None or args.resume_latest
        else runner.new_session()
    )
    streamed = False
    output_closed = False
    cancellation = CancellationToken()

    def event_sink(event: Mapping[str, Any]) -> None:
        nonlocal output_closed, streamed
        try:
            if args.jsonl:
                print(_json(event), flush=True)
                return
            event_type = event.get("type")
            payload = event.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            if (
                event_type == "message.delta"
                and payload.get("channel", "assistant") != "internal"
            ):
                delta = str(payload.get("delta", ""))
                if delta:
                    if sys.stdout.isatty():
                        delta = sanitize_terminal_text(delta)
                    print(delta, end="", flush=True)
                    streamed = True
            elif event_type == "tool.started" and sys.stderr.isatty():
                print(
                    f"\n› {payload.get('tool_name', 'tool')}",
                    file=sys.stderr,
                    flush=True,
                )
        except BrokenPipeError:
            output_closed = True
            cancellation.cancel("output_closed")

    previous_interrupt: Any = None

    def interrupt(_signum: int, _frame: Any) -> None:
        if not cancellation.cancel("keyboard_interrupt"):
            raise KeyboardInterrupt

    try:
        previous_interrupt = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, interrupt)
    except (AttributeError, ValueError):
        previous_interrupt = None
    try:
        outcome = runner.run_turn(
            session["session_id"],
            prompt,
            cancellation_token=cancellation,
            event_sink=event_sink,
        )
    except KeyboardInterrupt:
        cancellation.cancel("keyboard_interrupt")
        return EXIT_CANCELLED
    finally:
        if previous_interrupt is not None:
            signal.signal(signal.SIGINT, previous_interrupt)
    if output_closed:
        return EXIT_CANCELLED
    if args.jsonl:
        try:
            print(
                _json(
                    {
                        "schema": "agent_harness.exec_result.v1",
                        "session_id": outcome.session_id,
                        "run_id": outcome.run_id,
                        "turn_id": outcome.turn_id,
                        "status": outcome.status,
                        "reason": outcome.reason,
                        "usage": dict(outcome.usage),
                    }
                ),
                flush=True,
            )
        except BrokenPipeError:
            return EXIT_CANCELLED
    else:
        try:
            if outcome.message and not streamed:
                print(outcome.message)
            elif streamed:
                print()
            if outcome.status != "completed" and outcome.reason:
                print(
                    f"harness: {outcome.status}: "
                    f"{sanitize_terminal_text(outcome.reason)}",
                    file=sys.stderr,
                )
        except BrokenPipeError:
            return EXIT_CANCELLED
    if args.print_session:
        print(outcome.session_id, file=sys.stderr)
    return {
        "completed": EXIT_COMPLETED,
        "cancelled": EXIT_CANCELLED,
        "handoff": EXIT_HANDOFF,
    }.get(outcome.status, EXIT_FAILED)


def _session_metadata(session: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        key: session[key]
        for key in (
            "schema",
            "session_id",
            "workspace",
            "provider",
            "model",
            "permission_mode",
            "created_at",
            "updated_at",
            "archived",
            "usage",
        )
        if key in session
    }
    metadata["requires_reconciliation"] = any(
        item.get("status") == "running"
        or item.get("requires_reconciliation") is True
        for item in session.get("runs", [])
        if isinstance(item, Mapping)
    )
    return metadata


def _sessions(args: argparse.Namespace, store: SessionStore) -> int:
    sessions = store.list(include_archived=args.all)
    if args.json:
        print(
            _json(
                {
                    "schema": "agent_harness.session_list.v1",
                    "sessions": [_session_metadata(item) for item in sessions],
                }
            )
        )
        return 0
    if not sessions:
        print("No sessions.")
        return 0
    for session in sessions:
        archive = " archived" if session.get("archived") else ""
        attention = (
            " attention-required"
            if any(
                item.get("status") == "running"
                or item.get("requires_reconciliation") is True
                for item in session.get("runs", [])
                if isinstance(item, Mapping)
            )
            else ""
        )
        print(
            f"{session['session_id']}  {session['updated_at']}  "
            f"{session['permission_mode']}{archive}{attention}  "
            f"{sanitize_terminal_text(session['title'])}"
        )
    return 0


def _status(runner: AgentRunner) -> int:
    print(_json({"schema": "agent_harness.status.v1", **runner.provider_status}))
    return 0


class _HarnessArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _HarnessArgumentParser(
        prog="harness",
        description="Auditable, provider-neutral Agent Harness",
    )
    parser.add_argument("--cwd", default=".", help="workspace directory")
    parser.add_argument("--state-home")
    parser.add_argument("--api-key-file")
    parser.add_argument("--model")
    parser.add_argument(
        "--permissions",
        choices=tuple(PERMISSION_PROFILES),
        default=None,
    )
    parser.add_argument("--deadline", type=float, default=180.0)
    parser.add_argument("--max-steps", type=int, default=12)
    subparsers = parser.add_subparsers(dest="command")

    execute = subparsers.add_parser("exec", help="run one headless turn")
    execute.add_argument("prompt", nargs="+", help="prompt text, or - for stdin")
    execute.add_argument("--resume", metavar="SESSION_ID")
    execute.add_argument("--resume-latest", action="store_true")
    execute.add_argument("--jsonl", action="store_true")
    execute.add_argument("--print-session", action="store_true")

    resume = subparsers.add_parser("resume", help="resume a session in the TUI")
    resume.add_argument("session_id", nargs="?")

    subparsers.add_parser("tui", help="start a new TUI session")

    sessions = subparsers.add_parser("sessions", help="list saved sessions")
    sessions.add_argument("--all", action="store_true")
    sessions.add_argument("--json", action="store_true")

    archive = subparsers.add_parser("archive", help="archive a saved session")
    archive.add_argument("session_id")

    fork = subparsers.add_parser("fork", help="fork a saved session")
    fork.add_argument("session_id")

    subparsers.add_parser("effects", help="list unresolved workspace runs")

    reconcile = subparsers.add_parser(
        "reconcile",
        help="acknowledge one inspected unresolved run",
    )
    reconcile.add_argument("run_id")

    subparsers.add_parser("status", help="print provider and workspace status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command in {"sessions", "archive", "fork", "effects", "reconcile"}:
            store = SessionStore(Path(args.cwd), state_home=args.state_home)
            if args.command == "sessions":
                return _sessions(args, store)
            if args.command == "archive":
                session = store.archive(args.session_id)
                print(session["session_id"])
                return 0
            if args.command == "fork":
                session = store.fork(args.session_id)
                print(session["session_id"])
                return 0
            if args.command == "effects":
                print(
                    _json(
                        {
                            "schema": "agent_harness.unresolved_runs.v1",
                            "runs": store.unresolved_workspace_runs(),
                        }
                    )
                )
                return 0
            result = store.acknowledge_run(args.run_id)
            print(_json({"schema": "agent_harness.reconciliation.v1", **result}))
            return 0
        runner = _runner(args)
        if args.command == "exec":
            return _exec(args, runner)
        if args.command == "status":
            return _status(runner)
        if args.command == "resume":
            return run_tui(
                runner,
                resume=args.session_id,
                resume_latest=True,
                permission_override=args.permissions or "read-only",
            )
        if args.command in {None, "tui"}:
            return run_tui(runner)
        parser.error("unsupported command")
    except CliUsageError as exc:
        print(f"harness: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (DeepSeekConfigurationError, SessionStoreError, ValueError) as exc:
        print(f"harness: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except DeepSeekClientError as exc:
        print(f"harness: provider request failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except BrokenPipeError:
        return EXIT_CANCELLED
    except RuntimeError as exc:
        print(f"harness: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_USAGE


__all__ = ["build_parser", "main"]
