"""High-level multi-turn runner shared by the TUI and headless CLI."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from .core import (
    CancellationToken,
    HarnessJournal,
    HarnessLimits,
    RetryPolicy,
    run_agent_harness,
)
from .providers import DeepSeekClient, DeepSeekCodingModel, DeepSeekConfig
from .session import SessionStore, SessionStoreError
from .toolsets import PERMISSION_PROFILES, build_workspace_registry, permission_profile


EventSink = Callable[[Mapping[str, Any]], None]
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    session_id: str
    run_id: str
    turn_id: str
    status: str
    message: str | None
    reason: str
    usage: Mapping[str, int]
    result: Mapping[str, Any]


def _event_usage(events: Any) -> dict[str, int]:
    usage: dict[str, int] = {}
    if not isinstance(events, list):
        return usage
    for event in events:
        if not isinstance(event, Mapping) or event.get("type") != "model.completed":
            continue
        payload = event.get("payload")
        raw = payload.get("usage") if isinstance(payload, Mapping) else None
        if not isinstance(raw, Mapping):
            continue
        for key, value in raw.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[str(key)] = usage.get(str(key), 0) + value
    return usage


def _remote_content_enabled() -> bool:
    raw = os.getenv("HARNESS_ALLOW_REMOTE_CONTENT")
    if raw is None:
        # A provider-backed coding agent cannot operate without sending the
        # prompt. Keep the direct-launch path usable while allowing operators
        # to make remote execution fail closed explicitly.
        return True
    value = raw.strip().casefold()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        "HARNESS_ALLOW_REMOTE_CONTENT must be one of 1/0, true/false, yes/no, on/off"
    )


class AgentRunner:
    """One workspace-bound, provider-backed Agent Harness facade."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        state_home: str | Path | None = None,
        api_key_file: str | Path | None = None,
        model: str | None = None,
        permission_mode: str = "read-only",
        deadline_seconds: float = 180.0,
        max_steps: int = 12,
    ) -> None:
        if permission_mode not in PERMISSION_PROFILES:
            raise ValueError(f"unknown permission mode: {permission_mode}")
        self.store = SessionStore(workspace, state_home=state_home)
        self.workspace = self.store.workspace
        default_key_file = self.workspace / ".private" / "deepseek_api.txt"
        selected_key_file = api_key_file
        if selected_key_file is None and default_key_file.exists():
            selected_key_file = default_key_file
        config = DeepSeekConfig.from_environment(
            api_key_file=selected_key_file,
            allow_remote_content=_remote_content_enabled(),
            model=model,
        )
        self.client = DeepSeekClient(config)
        self.model = DeepSeekCodingModel(self.client)
        self.registry = build_workspace_registry(self.workspace)
        self.permission_mode = permission_mode
        self.limits = HarnessLimits(
            max_steps=max_steps,
            max_model_calls=max_steps,
            max_total_tool_calls=32,
            max_tool_calls_per_step=8,
            max_repeated_tool_calls=2,
            deadline_seconds=deadline_seconds,
            max_tool_output_chars=24_000,
            max_event_payload_chars=40_000,
        ).validated()
        self.retry_policy = RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0.25,
            backoff_multiplier=2.0,
            max_backoff_seconds=2.0,
        ).validated()

    @property
    def provider_status(self) -> dict[str, Any]:
        provider = self.client.public_status()
        # The low-level client can parse provider-managed web-search events, but
        # the coding adapter does not currently expose that capability as a
        # Harness tool. Report the active product surface, not latent transport
        # support.
        provider["web_search_supported"] = False
        provider["web_search_transport"] = "disabled"
        return {
            **provider,
            "workspace": str(self.workspace),
            "permission_mode": self.permission_mode,
            "available_permission_modes": list(PERMISSION_PROFILES),
        }

    def new_session(self, *, title: str = "New session") -> dict[str, Any]:
        return self.store.create(
            provider="deepseek",
            model=self.client.config.model,
            permission_mode=self.permission_mode,
            title=title,
        )

    def resume_session(
        self,
        session_id: str | None = None,
        *,
        permission_override: str = "read-only",
    ) -> dict[str, Any]:
        if permission_override not in PERMISSION_PROFILES:
            raise ValueError(f"unknown permission mode: {permission_override}")
        self.permission_mode = permission_override
        if session_id is None:
            sessions = self.store.list()
            if not sessions:
                return self.new_session()
            selected_session_id = str(sessions[0]["session_id"])
        else:
            selected_session_id = session_id
        with self.store.turn_lock(selected_session_id):
            session = self.store.load(selected_session_id)
            if session.get("archived"):
                raise SessionStoreError("session is archived; fork it before continuing")
            unfinished = [
                item
                for item in session.get("runs", [])
                if item.get("status") == "running"
                or item.get("requires_reconciliation") is True
            ]
            if unfinished:
                run_id = str(unfinished[-1].get("run_id", "unknown"))
                raise SessionStoreError(
                    f"session has an unfinished or uncertain run ({run_id}); "
                    "inspect its journal before archiving it or starting a new session"
                )
            if (
                session.get("permission_mode") != permission_override
                or session.get("provider") != "deepseek"
                or session.get("model") != self.client.config.model
            ):
                session = self.store.update_runtime(
                    session["session_id"],
                    permission_mode=permission_override,
                    provider="deepseek",
                    model=self.client.config.model,
                )
            return session

    def set_permission_mode(self, session_id: str, mode: str) -> dict[str, Any]:
        if mode not in PERMISSION_PROFILES:
            raise ValueError(f"unknown permission mode: {mode}")
        with self.store.turn_lock(session_id):
            self.permission_mode = mode
            return self.store.update_runtime(
                session_id,
                permission_mode=mode,
                provider="deepseek",
                model=self.client.config.model,
            )

    def run_turn(
        self,
        session_id: str,
        prompt: str,
        *,
        cancellation_token: CancellationToken | None = None,
        event_sink: EventSink | None = None,
    ) -> TurnOutcome:
        with self.store.workspace_run_lock():
            unresolved = self.store.unresolved_workspace_runs()
            if unresolved:
                run_id = unresolved[-1]["run_id"]
                raise SessionStoreError(
                    f"workspace has an unresolved run ({run_id}); inspect its journal "
                    "and explicitly reconcile it before any new run"
                )
            with self.store.turn_lock(session_id):
                return self._run_turn_locked(
                    session_id,
                    prompt,
                    cancellation_token=cancellation_token,
                    event_sink=event_sink,
                )

    def _run_turn_locked(
        self,
        session_id: str,
        prompt: str,
        *,
        cancellation_token: CancellationToken | None = None,
        event_sink: EventSink | None = None,
    ) -> TurnOutcome:
        content = str(prompt).strip()
        if not content or len(content) > 200_000:
            raise ValueError("prompt must contain 1 to 200000 characters")
        # Stored permission metadata is never an authority source. The current
        # invocation's explicit runner mode controls the grants, and the
        # per-session turn lock makes the displayed metadata follow it.
        mode = self.permission_mode
        if mode not in PERMISSION_PROFILES:
            mode = "read-only"
        existing = self.store.load(session_id)
        unresolved = [
            item
            for item in existing.get("runs", [])
            if item.get("status") == "running"
            or item.get("requires_reconciliation") is True
        ]
        if unresolved:
            raise SessionStoreError(
                "session has an unfinished or uncertain run; inspect its journal "
                "before archiving it or starting a new session"
            )
        self.store.update_runtime(
            session_id,
            permission_mode=mode,
            provider="deepseek",
            model=self.client.config.model,
        )
        session = self.store.append_message(
            session_id,
            role="user",
            content=content,
            reserve_messages=1,
        )
        run_id = f"run_{uuid4().hex}"
        turn_id = f"turn_{uuid4().hex}"
        event_path, checkpoint_path = self.store.run_paths(run_id)
        journal = HarnessJournal(
            event_path,
            run_id=run_id,
            turn_id=turn_id,
            checkpoint_path=checkpoint_path,
        )
        self.store.record_run(
            session_id,
            run_id=run_id,
            turn_id=turn_id,
            status="running",
        )
        permissions = permission_profile(mode)
        scopes = {"internal", "user_input", "workspace_read"}
        if "workspace.write" in permissions or "process.exec" in permissions:
            scopes.add("workspace_write")
        if "process.exec" in permissions:
            scopes.add("host_access")
        try:
            result = run_agent_harness(
                self.model,
                self.registry,
                {
                    "messages": [
                        {"role": item["role"], "content": item["content"]}
                        for item in session["messages"]
                    ],
                    "workspace": ".",
                    "permission_mode": mode,
                    "settled_effects": [
                        {
                            "run_id": run.get("run_id"),
                            **effect,
                        }
                        for run in session.get("runs", [])
                        for effect in run.get("effects", [])
                    ][-64:],
                    "safety_notices": list(session.get("risk_notices", []))[-8:],
                },
                run_id=run_id,
                turn_id=turn_id,
                limits=self.limits,
                retry_policy=self.retry_policy,
                allowed_permissions=permissions,
                cancellation_token=cancellation_token or CancellationToken(),
                event_sink=event_sink,
                journal=journal,
                principal_id="local-user",
                session_id=session_id,
                trusted_data_scopes=scopes,
            )
        except Exception:
            # Unknown failures may happen after an external effect but before
            # its settlement can be inspected. Leave the run as `running` so
            # every future entry point fails closed instead of replaying it.
            raise
        status = str(result.get("status", "failed"))
        output = result.get("output")
        message = (
            str(output.get("message", "")).strip()
            if isinstance(output, Mapping)
            else ""
        )
        durable_events = journal.replay()
        usage = _event_usage(durable_events)
        requested: dict[str, Mapping[str, Any]] = {}
        started_effects: dict[str, str] = {}
        settled_effects: set[str] = set()
        effects: list[dict[str, str]] = []
        terminal_type = ""
        terminal_reason_code = ""
        for event in durable_events:
            if not isinstance(event, Mapping):
                continue
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                continue
            event_type = str(event.get("type", ""))
            if event_type in {
                "run.completed",
                "run.cancelled",
                "run.failed",
                "run.handoff",
            }:
                terminal_type = event_type
                terminal_reason_code = str(payload.get("reason_code", ""))
                continue
            call_id = payload.get("call_id")
            if not isinstance(call_id, str):
                continue
            if event_type == "tool.requested":
                requested[call_id] = payload
                continue
            tool_name = payload.get("tool_name")
            if not isinstance(tool_name, str):
                continue
            entry = self.registry.get(tool_name)
            if entry is None or entry[0].replay_policy == "safe":
                continue
            if event_type == "tool.effect_started":
                started_effects[call_id] = tool_name
                continue
            if event_type != "tool.completed" or call_id not in started_effects:
                continue
            settled_effects.add(call_id)
            effect = {
                "call_id": call_id,
                "tool_name": tool_name,
                "result_sha256": str(payload.get("result_sha256", "")),
            }
            arguments_sha256 = requested.get(call_id, {}).get("arguments_sha256")
            if isinstance(arguments_sha256, str):
                effect["arguments_sha256"] = arguments_sha256
            effects.append(effect)
        unresolved_effects = set(started_effects) - settled_effects
        expected_terminal = {
            "completed": "run.completed",
            "cancelled": "run.cancelled",
            "handoff": "run.handoff",
            "failed": "run.failed",
            "deadline_exceeded": "run.failed",
        }.get(status)
        if expected_terminal != terminal_type:
            raise SessionStoreError(
                "runtime result does not match its authoritative journal terminal"
            )
        reconciliation_reasons = {
            "cancelled_after_external_effect",
            "deadline_after_external_effect",
            "external_effect_unsettled",
            "failure_after_external_effect",
        }
        requires_reconciliation = terminal_reason_code in reconciliation_reasons
        checkpoint = result.get("checkpoint")
        if isinstance(checkpoint, Mapping):
            pending_effect = checkpoint.get("pending_effect")
        else:
            pending_effect = getattr(checkpoint, "pending_effect", None)
        if bool(unresolved_effects) != isinstance(pending_effect, Mapping):
            raise SessionStoreError(
                "runtime effect ledger does not match its authoritative terminal"
            )
        if status == "completed" and message:
            self.store.append_message(
                session_id,
                role="assistant",
                content=message,
                run_id=run_id,
            )
        stored_status = (
            status
            if status in {"completed", "failed", "cancelled", "handoff"}
            else "failed"
        )
        self.store.record_run(
            session_id,
            run_id=run_id,
            turn_id=turn_id,
            status=stored_status,
            usage=usage,
            effects=effects,
            requires_reconciliation=requires_reconciliation,
        )
        return TurnOutcome(
            session_id=session_id,
            run_id=run_id,
            turn_id=turn_id,
            status=status,
            message=message or None,
            reason=str(result.get("reason", "")),
            usage=usage,
            result=result,
        )


__all__ = ["AgentRunner", "EventSink", "TurnOutcome"]
