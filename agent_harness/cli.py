"""Command-line and headless interfaces for Agent Harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
from typing import Any, Mapping, Sequence

from .attachments import AttachmentDescriptor, AttachmentError
from .core import CancellationToken, HarnessCancelled
from .hooks import HookDefinition, HookLoadError, HookSnapshot, load_project_hooks
from .instructions import InstructionLoadError, load_project_instructions
from .mcp import (
    McpLoadError,
    McpServerDefinition,
    McpSnapshot,
    load_project_mcp,
    mcp_status,
    refresh_mcp_catalog,
)
from .core.mcp_protocol import McpProtocolError
from .providers import DeepSeekClientError, DeepSeekConfigurationError
from .runner import AgentRunner, default_worktree_home
from .session import SessionStore, SessionStoreError
from .toolsets import PERMISSION_PROFILES
from .tui import run_tui
from .tui_state import sanitize_terminal_text
from .worktrees import WorktreeError, WorktreeManager, WorktreeRecord


EXIT_COMPLETED = 0
EXIT_CANCELLED = 2
EXIT_HANDOFF = 3
EXIT_FAILED = 4
EXIT_USAGE = 64
EXIT_CONFIG = 78
_MAX_AGENT_ARTIFACTS = 256


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
        active_directory=args.active_directory,
        state_home=args.state_home,
        api_key_file=args.api_key_file,
        model=args.model,
        permission_mode=args.permissions or "read-only",
        deadline_seconds=args.deadline,
        max_steps=args.max_steps,
        worktree_home=args.worktree_home,
    )


def _discard_unreferenced_attachments(
    runner: AgentRunner,
    descriptors: Sequence[AttachmentDescriptor],
    *,
    session_id: str | None,
) -> None:
    """Best-effort cleanup limited to blobs proven absent from durable state."""

    referenced: set[str] = set()
    if session_id is not None:
        try:
            persisted = runner.store.load(session_id)
        except Exception:
            # An unreadable store is an uncertain boundary. Preserve every blob.
            return
        for message in persisted.get("messages", []):
            if not isinstance(message, Mapping):
                return
            manifest = message.get("attachments", [])
            if not isinstance(manifest, list):
                return
            for item in manifest:
                if not isinstance(item, Mapping):
                    return
                attachment_id = item.get("attachment_id")
                if isinstance(attachment_id, str):
                    referenced.add(attachment_id)
    for descriptor in descriptors:
        if descriptor.attachment_id in referenced:
            continue
        try:
            runner.discard_attachment(descriptor)
        except Exception:
            # Cleanup must never replace the authoritative turn failure.
            continue


def _exec(args: argparse.Namespace, runner: AgentRunner) -> int:
    if args.resume is not None and args.resume_latest:
        raise CliUsageError("--resume and --resume-latest are mutually exclusive")
    prompt = " ".join(args.prompt).strip()
    if prompt == "-":
        prompt = sys.stdin.read().strip()
    if not prompt:
        raise ValueError("exec requires a non-empty prompt")
    attachments = runner.ingest_attachments(args.attach) if args.attach else ()
    session: Mapping[str, Any] | None = None
    try:
        session = (
            runner.resume_session(
                args.resume,
                permission_override=args.permissions or "read-only",
            )
            if args.resume is not None or args.resume_latest
            else runner.new_session()
        )
    except BaseException:
        _discard_unreferenced_attachments(
            runner,
            attachments,
            session_id=None,
        )
        raise
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
            attachments=attachments,
        )
    except KeyboardInterrupt:
        cancellation.cancel("keyboard_interrupt")
        _discard_unreferenced_attachments(
            runner,
            attachments,
            session_id=str(session["session_id"]),
        )
        return EXIT_CANCELLED
    except BaseException:
        _discard_unreferenced_attachments(
            runner,
            attachments,
            session_id=str(session["session_id"]),
        )
        raise
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


def _worktree_metadata(
    record: WorktreeRecord,
    *,
    include_path: bool = False,
) -> dict[str, Any]:
    """Return content-free lifecycle metadata, optionally with one child path."""

    metadata: dict[str, Any] = {
        "schema": record.schema,
        "worktree_id": record.worktree_id,
        "phase": record.phase,
        "baseline_commit": record.baseline_commit,
        "baseline_manifest_sha256": record.baseline_manifest_sha256,
        "created_at": record.created_at,
    }
    if include_path:
        metadata["worktree_path"] = record.worktree_path
    return metadata


def _agents(args: argparse.Namespace) -> int:
    """Inspect retained child artifacts without constructing a provider client."""

    if args.path and args.worktree_id is None:
        raise CliUsageError("--path requires WORKTREE_ID")
    store = SessionStore(Path(args.cwd), state_home=args.state_home)
    selected_home = (
        Path(args.worktree_home).expanduser()
        if args.worktree_home is not None
        else default_worktree_home()
    )
    manager = WorktreeManager(
        store.workspace,
        state_directory=store.root / "worktrees",
        worktree_home=selected_home,
    )

    if args.worktree_id is None:
        records = manager.list_records(limit=_MAX_AGENT_ARTIFACTS)
        metadata = [_worktree_metadata(record) for record in records]
        if args.json:
            print(
                _json(
                    {
                        "schema": "agent_harness.worktree_list.v1",
                        "worktrees": metadata,
                    }
                )
            )
            return 0
        if not metadata:
            print("No subagent worktrees.")
            return 0
        for item in metadata:
            manifest = item.get("baseline_manifest_sha256")
            manifest_text = str(manifest)[:12] if manifest else "pending"
            print(
                f"{item['worktree_id']}  {item['phase']}  "
                f"{str(item['baseline_commit'])[:12]}  {manifest_text}  "
                f"{sanitize_terminal_text(str(item['created_at']))}"
            )
        return 0

    record = manager.get(args.worktree_id)
    metadata = _worktree_metadata(record, include_path=args.path)
    if args.json:
        print(
            _json(
                {
                    "schema": "agent_harness.worktree_status.v1",
                    "worktree": metadata,
                }
            )
        )
    elif args.path:
        print(sanitize_terminal_text(record.worktree_path))
    else:
        manifest = metadata.get("baseline_manifest_sha256")
        manifest_text = str(manifest)[:12] if manifest else "pending"
        print(
            f"{metadata['worktree_id']}  {metadata['phase']}  "
            f"{str(metadata['baseline_commit'])[:12]}  {manifest_text}  "
            f"{sanitize_terminal_text(str(metadata['created_at']))}"
        )
    return 0


def _context(args: argparse.Namespace, runner: AgentRunner) -> int:
    status = runner.context_status(args.session_id)
    if args.json:
        print(_json(status))
        return 0
    print(
        f"context {status['active_context_sha256'][:12]} · "
        f"raw {status['transcript_message_count']} · "
        f"compacted {status['compacted_message_count']} · "
        f"active {status['active_message_count']} · "
        f"estimate {status['estimated_input_tokens_upper_bound']}/"
        f"{status['available_input_tokens']} tokens"
    )
    if status["compaction_id"]:
        print(
            f"summary {status['summary_sha256'][:12]} · "
            f"{status['summary_chars']} chars · {status['compaction_id']}"
        )
    return 0


def _compact(args: argparse.Namespace, runner: AgentRunner) -> int:
    cancellation = CancellationToken()
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
        try:
            result = runner.compact_session(
                args.session_id,
                cancellation_token=cancellation,
            )
        except (HarnessCancelled, KeyboardInterrupt):
            return EXIT_CANCELLED
    finally:
        if previous_interrupt is not None:
            signal.signal(signal.SIGINT, previous_interrupt)
    if args.json:
        print(_json(result))
    elif result["status"] == "compacted":
        print(
            f"compacted {result['source_message_count']} messages · "
            f"active {result['active_message_count']} · "
            f"summary {str(result['summary_sha256'])[:12]}"
        )
    else:
        print("No eligible completed turns to compact.")
    return 0


def _instructions(args: argparse.Namespace) -> int:
    workspace = Path(args.cwd).expanduser().resolve(strict=True)
    active = Path(args.active_directory).expanduser() if args.active_directory else workspace
    if not active.is_absolute():
        active = workspace / active
    snapshot = load_project_instructions(workspace, active)
    metadata = dict(snapshot.metadata())
    if args.json:
        print(_json(metadata))
        return 0
    if not snapshot.documents:
        print("No project instructions.")
        return 0
    print(
        f"instructions {snapshot.snapshot_sha256[:12]} · "
        f"{len(snapshot.documents)} file(s) · {snapshot.total_bytes} bytes"
    )
    for document in snapshot.documents:
        print(
            f"{document.relative_path}  {document.byte_length} bytes  "
            f"{document.content_sha256[:12]}"
        )
    return 0


def _hook_definition(snapshot: HookSnapshot, hook_id: str) -> HookDefinition:
    for definition in snapshot.definitions:
        if definition.hook_id == hook_id:
            return definition
    raise ValueError(f"project hook is not configured: {hook_id}")


def _hooks(args: argparse.Namespace) -> int:
    """Inspect or update project-hook trust without constructing a provider."""

    store = SessionStore(Path(args.cwd), state_home=args.state_home)
    action = args.hook_action
    hook_id = str(args.hook_id).strip() if action is not None else ""
    if action == "revoke":
        removed = store.revoke_hook_trust(hook_id)
        result = {
            "schema": "agent_harness.hook_trust_update.v1",
            "hook_id": hook_id,
            "action": "revoked",
            "removed": removed,
        }
        if args.json:
            print(_json(result))
        elif removed:
            print(f"revoked {sanitize_terminal_text(hook_id)}")
        else:
            print(f"no trust decision for {sanitize_terminal_text(hook_id)}")
        return 0

    supplied_digest = ""
    if action is not None:
        supplied_digest = str(args.sha256).strip().casefold()
        if (
            len(supplied_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in supplied_digest
            )
        ):
            raise CliUsageError("--sha256 must be a 64-character hexadecimal digest")

    snapshot = load_project_hooks(store.workspace)
    if action is None:
        metadata = dict(snapshot.metadata(store.hook_trust_state()))
        if args.json:
            print(_json(metadata))
            return 0
        if not snapshot.config_present:
            print("No project hooks.")
            return 0
        print(
            f"hooks {snapshot.snapshot_sha256[:12]} · "
            f"{metadata['hook_count']} configured · "
            f"{metadata['trusted_hook_count']} trusted · "
            f"{metadata['disabled_hook_count']} disabled · "
            f"{metadata['pending_hook_count']} review-required"
        )
        for item in metadata["hooks"]:
            matchers = ",".join(str(value) for value in item["matchers"])
            print(
                f"{sanitize_terminal_text(item['hook_id'])}  "
                f"{item['event_name']}  {item['trust_status']}  "
                f"{item['definition_sha256']}  "
                f"{sanitize_terminal_text(item['entrypoint'])}  {matchers}"
            )
        return 0

    definition = _hook_definition(snapshot, hook_id)
    if supplied_digest != definition.definition_sha256:
        raise ValueError(
            "hook digest does not match the current definition; run `harness hooks` again"
        )
    # Re-read immediately before persisting. A later change remains safe: the
    # digest-bound record will be reported as modified and cannot execute.
    confirmed = _hook_definition(load_project_hooks(store.workspace), hook_id)
    if confirmed.definition_sha256 != definition.definition_sha256:
        raise HookLoadError("hook definition changed while recording trust")
    trust_action = "trusted" if action == "trust" else "disabled"
    store.set_hook_trust(
        hook_id,
        confirmed.definition_sha256,
        action=trust_action,
    )
    result = {
        "schema": "agent_harness.hook_trust_update.v1",
        "hook_id": hook_id,
        "action": trust_action,
        "definition_sha256": confirmed.definition_sha256,
    }
    if args.json:
        print(_json(result))
    else:
        print(
            f"{trust_action} {sanitize_terminal_text(hook_id)} "
            f"{confirmed.definition_sha256}"
        )
    return 0


def _mcp_definition(snapshot: McpSnapshot, server_id: str) -> McpServerDefinition:
    return snapshot.server(server_id)


def _mcp(args: argparse.Namespace) -> int:
    """Inspect, trust and refresh project MCP servers without a provider."""

    store = SessionStore(Path(args.cwd), state_home=args.state_home)
    action = args.mcp_action
    server_id = str(args.server_id).strip() if action is not None else ""
    if action == "revoke":
        removed = store.revoke_mcp_trust(server_id)
        result = {
            "schema": "agent_harness.mcp_trust_update.v1",
            "server_id": server_id,
            "action": "revoked",
            "removed": removed,
        }
        if args.json:
            print(_json(result))
        elif removed:
            print(f"revoked {sanitize_terminal_text(server_id)}")
        else:
            print(f"no trust decision for {sanitize_terminal_text(server_id)}")
        return 0

    snapshot = load_project_mcp(store.workspace)
    if action is None:
        metadata = mcp_status(store, snapshot)
        if args.json:
            print(_json(metadata))
            return 0
        if not snapshot.config_present:
            print("No project MCP servers.")
            return 0
        print(
            f"mcp {snapshot.snapshot_sha256[:12]} · "
            f"{metadata['server_count']} configured · "
            f"{metadata['trusted_server_count']} trusted · "
            f"{metadata['disabled_server_count']} disabled · "
            f"{metadata['pending_server_count']} review-required · "
            f"{metadata['ready_server_count']} ready"
        )
        for item in metadata["servers"]:
            argv = json.dumps(
                [item["command"], *item["args"]],
                ensure_ascii=True,
                separators=(",", ":"),
            )
            env_names = ",".join(str(value) for value in item["pass_env"]) or "-"
            print(
                f"{sanitize_terminal_text(item['server_id'])}  "
                f"{item['trust_status']}/{item['catalog_status']}  "
                f"{item['definition_sha256']}  argv={argv}  "
                f"cwd={sanitize_terminal_text(item['cwd'])}  env={env_names}  "
                f"network={str(item['network_access']).lower()}  "
                f"fork={str(item['allow_process_fork']).lower()}"
            )
        return 0

    definition = _mcp_definition(snapshot, server_id)
    if action == "refresh":
        result = refresh_mcp_catalog(store, snapshot, server_id)
        if args.json:
            print(_json(result))
        else:
            print(
                f"refreshed {sanitize_terminal_text(server_id)} "
                f"{result['catalog_sha256']} · {len(result['tools'])} tools · "
                f"{len(result['rejected_tools'])} rejected"
            )
        return 0

    supplied_digest = str(args.sha256).strip().casefold()
    if len(supplied_digest) != 64 or any(
        character not in "0123456789abcdef" for character in supplied_digest
    ):
        raise CliUsageError("--sha256 must be a 64-character hexadecimal digest")
    if supplied_digest != definition.definition_sha256:
        raise ValueError(
            "MCP digest does not match the current definition; run `harness mcp` again"
        )
    confirmed = _mcp_definition(load_project_mcp(store.workspace), server_id)
    if confirmed.definition_sha256 != definition.definition_sha256:
        raise McpLoadError("MCP server definition changed while recording trust")
    trust_action = "trusted" if action == "trust" else "disabled"
    store.set_mcp_trust(
        server_id,
        confirmed.definition_sha256,
        action=trust_action,
    )
    result = {
        "schema": "agent_harness.mcp_trust_update.v1",
        "server_id": server_id,
        "action": trust_action,
        "definition_sha256": confirmed.definition_sha256,
    }
    if args.json:
        print(_json(result))
    else:
        print(f"{trust_action} {sanitize_terminal_text(server_id)} {confirmed.definition_sha256}")
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
    parser.add_argument(
        "--active-directory",
        help="active subdirectory inside the workspace for scoped instructions",
    )
    parser.add_argument("--state-home")
    parser.add_argument(
        "--worktree-home",
        help="private directory for isolated subagent worktrees",
    )
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
    execute.add_argument(
        "--attach",
        action="append",
        default=[],
        metavar="PATH",
        help="attach one UTF-8 text, PNG, JPEG, or PDF snapshot (repeatable)",
    )
    execute.add_argument("--jsonl", action="store_true")
    execute.add_argument("--print-session", action="store_true")

    resume = subparsers.add_parser("resume", help="resume a session in the TUI")
    resume.add_argument("session_id", nargs="?")

    subparsers.add_parser("tui", help="start a new TUI session")

    sessions = subparsers.add_parser("sessions", help="list saved sessions")
    sessions.add_argument("--all", action="store_true")
    sessions.add_argument("--json", action="store_true")

    agents = subparsers.add_parser(
        "agents",
        help="list retained foreground-subagent worktree artifacts",
    )
    agents.add_argument("worktree_id", nargs="?", metavar="WORKTREE_ID")
    agents.add_argument(
        "--path",
        action="store_true",
        help="include the selected artifact's isolated worktree path",
    )
    agents.add_argument("--json", action="store_true")

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
    instructions = subparsers.add_parser(
        "instructions",
        help="show the project instruction snapshot without file contents",
    )
    instructions.add_argument("--json", action="store_true")
    hooks = subparsers.add_parser(
        "hooks",
        help="inspect or update exact project-hook trust decisions",
    )
    hooks.add_argument("--json", action="store_true")
    hook_actions = hooks.add_subparsers(dest="hook_action")
    for action, help_text in (
        ("trust", "trust one exact current hook definition"),
        ("disable", "disable one exact current hook definition"),
    ):
        update = hook_actions.add_parser(action, help=help_text)
        update.add_argument("hook_id")
        update.add_argument(
            "--sha256",
            required=True,
            help="exact definition digest shown by `harness hooks`",
        )
    revoke = hook_actions.add_parser(
        "revoke",
        help="remove a saved trust or disable decision",
    )
    revoke.add_argument("hook_id")
    mcp = subparsers.add_parser(
        "mcp",
        help="inspect, trust or refresh project MCP stdio servers",
    )
    mcp.add_argument("--json", action="store_true")
    mcp_actions = mcp.add_subparsers(dest="mcp_action")
    for action, help_text in (
        ("trust", "trust one exact current MCP server definition"),
        ("disable", "disable one exact current MCP server definition"),
    ):
        update = mcp_actions.add_parser(action, help=help_text)
        update.add_argument("server_id")
        update.add_argument(
            "--sha256",
            required=True,
            help="exact definition digest shown by `harness mcp`",
        )
    refresh = mcp_actions.add_parser(
        "refresh",
        help="start one trusted server and freeze its bounded tool catalog",
    )
    refresh.add_argument("server_id")
    revoke_mcp = mcp_actions.add_parser(
        "revoke",
        help="remove a saved MCP trust or disable decision",
    )
    revoke_mcp.add_argument("server_id")
    context = subparsers.add_parser(
        "context",
        help="show active-context and compaction diagnostics without content",
    )
    context.add_argument("session_id")
    context.add_argument("--json", action="store_true")
    compact = subparsers.add_parser(
        "compact",
        help="summarize old completed turns without deleting the transcript",
    )
    compact.add_argument("session_id")
    compact.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "instructions":
            return _instructions(args)
        if args.command == "hooks":
            return _hooks(args)
        if args.command == "mcp":
            return _mcp(args)
        if args.command == "agents":
            return _agents(args)
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
        if args.command == "context":
            return _context(args, runner)
        if args.command == "compact":
            return _compact(args, runner)
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
    except (
        DeepSeekConfigurationError,
        AttachmentError,
        HookLoadError,
        InstructionLoadError,
        McpLoadError,
        McpProtocolError,
        SessionStoreError,
        ValueError,
    ) as exc:
        print(f"harness: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except WorktreeError:
        # Worktree failures may be caused by tampered records.  Do not echo the
        # original message because it can contain untrusted or private paths.
        print("harness: isolated worktree state is unavailable", file=sys.stderr)
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
