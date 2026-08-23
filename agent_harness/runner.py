"""High-level multi-turn runner shared by the TUI and headless CLI."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from .core import (
    ApprovalBroker,
    ApprovalPolicy,
    CancellationToken,
    HarnessJournal,
    HarnessLimits,
    RetryPolicy,
    run_agent_harness,
)
from .context import estimate_context_tokens, select_compaction_plan
from .hooks import (
    HookLoadError,
    HookSnapshot,
    TrustedHookRunner,
    load_project_hooks,
)
from .instructions import InstructionSnapshot, load_project_instructions
from .providers import DeepSeekClient, DeepSeekCodingModel, DeepSeekConfig
from .session import SessionStore, SessionStoreError
from .toolsets import (
    PERMISSION_PROFILES,
    build_workspace_registry,
    permission_profile,
    workspace_sandbox_status,
)


EventSink = Callable[[Mapping[str, Any]], None]
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_CONTEXT_SAFETY_MARGIN_TOKENS = 512
_AUTO_COMPACTION_TRIGGER_RATIO = 0.80
_AUTO_COMPACTION_TARGET_RATIO = 0.60
_MAX_COMPACTION_PASSES = 8


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
        active_directory: str | Path | None = None,
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
        raw_active = (
            self.workspace
            if active_directory is None
            else Path(active_directory).expanduser()
        )
        if not raw_active.is_absolute():
            raw_active = self.workspace / raw_active
        selected_active = raw_active.resolve(strict=True)
        if not selected_active.is_dir():
            raise ValueError("active directory must be a directory")
        try:
            selected_active.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError("active directory must stay inside the workspace") from exc
        self.active_directory = selected_active
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
        try:
            hooks = self.hook_status()
        except (HookLoadError, SessionStoreError):
            hooks = {
                "schema": "agent_harness.hook_status.v1",
                "status": "invalid",
                "error_code": "hook_configuration_invalid",
            }
        return {
            **provider,
            "workspace": str(self.workspace),
            "active_directory": str(self.active_directory),
            "permission_mode": self.permission_mode,
            "available_permission_modes": list(PERMISSION_PROFILES),
            "workspace_sandbox": workspace_sandbox_status(),
            "project_hooks": hooks,
            "approval_defaults": {
                "low": "allow",
                "medium": "ask",
                "high": "ask",
                "headless_ask": "handoff",
            },
        }

    def hook_snapshot(self) -> HookSnapshot:
        """Freeze the exact project hook definitions proposed for the next run."""

        return load_project_hooks(self.workspace)

    def hook_status(self) -> dict[str, Any]:
        """Return content-free hook definitions and exact trust status."""

        snapshot = self.hook_snapshot()
        trust_state = self.store.hook_trust_state()
        metadata = snapshot.metadata(trust_state)
        if metadata["pending_hook_count"]:
            status = "review_required"
        elif metadata["trusted_hook_count"]:
            status = "ready" if metadata["sandbox"]["available"] else "blocked"
        elif metadata["hook_count"]:
            status = "disabled"
        else:
            status = "none"
        return {
            **metadata,
            "schema": "agent_harness.hook_status.v1",
            "status": status,
        }

    def set_hook_trust(
        self,
        hook_id: str,
        definition_sha256: str,
        *,
        action: str,
    ) -> dict[str, Any]:
        """Persist a digest-bound trust or disable decision for a current hook."""

        snapshot = self.hook_snapshot()
        definition = next(
            (item for item in snapshot.definitions if item.hook_id == hook_id),
            None,
        )
        if definition is None:
            raise HookLoadError("hook id is not present in the current project config")
        if definition.definition_sha256 != definition_sha256:
            raise HookLoadError("hook definition digest does not match current content")
        self.store.set_hook_trust(
            definition.hook_id,
            definition.definition_sha256,
            action=action,
        )
        return self.hook_status()

    def trust_hook(
        self,
        hook_id: str,
        definition_sha256: str,
        *,
        action: str = "trusted",
    ) -> dict[str, Any]:
        return self.set_hook_trust(
            hook_id,
            definition_sha256,
            action=action,
        )

    def revoke_hook(self, hook_id: str) -> dict[str, Any]:
        self.store.revoke_hook_trust(hook_id)
        return self.hook_status()

    def instruction_snapshot(self) -> InstructionSnapshot:
        """Load the workspace-scoped instruction set for the next turn."""

        return load_project_instructions(self.workspace, self.active_directory)

    def _context_budget(
        self,
        *,
        session: Mapping[str, Any],
        snapshot: InstructionSnapshot,
        prospective_prompt: str = "",
    ) -> dict[str, Any]:
        view = self.store.context_view(str(session["session_id"]))
        permissions = permission_profile(self.permission_mode)
        scopes = {"internal", "user_input", "workspace_read"}
        if (
            "workspace.write" in permissions
            or "process.exec.sandboxed" in permissions
            or "process.exec.host" in permissions
        ):
            scopes.add("workspace_write")
        if "process.exec.host" in permissions:
            scopes.add("host_access")
        definitions = self.registry.definitions(
            permissions,
            trusted_data_scopes=frozenset(scopes),
        )
        estimated = estimate_context_tokens(
            summary=str(view["summary"]),
            messages=view["messages"],
            project_instruction_bytes=snapshot.total_bytes,
            tool_definitions=definitions,
            prospective_prompt=prospective_prompt,
        )
        spec = self.model.model_spec
        available = max(
            1,
            spec.context_window_tokens
            - spec.maximum_output_tokens
            - _CONTEXT_SAFETY_MARGIN_TOKENS,
        )
        return {
            "estimated_input_tokens_upper_bound": estimated,
            "available_input_tokens": available,
            "automatic_trigger_tokens": max(
                1, int(available * _AUTO_COMPACTION_TRIGGER_RATIO)
            ),
            "automatic_target_tokens": max(
                1, int(available * _AUTO_COMPACTION_TARGET_RATIO)
            ),
            "context_window_tokens": spec.context_window_tokens,
            "maximum_output_tokens": spec.maximum_output_tokens,
            "view": view,
        }

    def context_status(self, session_id: str) -> dict[str, Any]:
        """Return private-content-free active-context diagnostics."""

        session = self.store.load(session_id)
        snapshot = self.instruction_snapshot()
        budget = self._context_budget(session=session, snapshot=snapshot)
        view = budget.pop("view")
        lineage = view["lineage"]
        return {
            "schema": "agent_harness.context_status.v1",
            "session_id": session_id,
            "transcript_message_count": view["transcript_message_count"],
            "compacted_message_count": view["compacted_message_count"],
            "active_message_count": view["active_message_count"],
            "summary_chars": len(str(view["summary"])),
            "compaction_count": len(session["compactions"]),
            "compaction_id": lineage["compaction_id"],
            "summary_sha256": lineage["summary_sha256"],
            "source_messages_sha256": lineage["source_messages_sha256"],
            "active_context_sha256": lineage["active_context_sha256"],
            "instruction_bytes": snapshot.total_bytes,
            "instruction_count": len(snapshot.documents),
            **budget,
        }

    def _compact_locked(
        self,
        session_id: str,
        *,
        trigger: str,
        snapshot: InstructionSnapshot,
        cancellation_token: CancellationToken,
        target_tokens: int | None = None,
        prospective_prompt: str = "",
    ) -> dict[str, Any]:
        compactor = getattr(self.model, "compact_context", None)
        if not callable(compactor):
            raise SessionStoreError("the selected model does not support context compaction")
        passes = 0
        latest_record: Mapping[str, Any] | None = None
        while passes < _MAX_COMPACTION_PASSES:
            cancellation_token.raise_if_cancelled()
            session = self.store.load(session_id)
            plan = select_compaction_plan(session)
            if plan is None:
                break
            result = compactor(
                plan,
                cancellation_token=cancellation_token,
                deadline_monotonic=time.monotonic() + self.limits.deadline_seconds,
            ).validated()
            cancellation_token.raise_if_cancelled()
            latest_record = self.store.record_compaction(
                session_id,
                summary=result.summary,
                source_message_count=plan.source_message_count,
                provider=self.model.model_spec.provider,
                model=self.model.model_spec.model,
                trigger=trigger,
                usage=result.usage,
                instructions_sha256=snapshot.snapshot_sha256,
                provider_request_id=result.provider_request_id,
            )
            passes += 1
            if target_tokens is not None:
                refreshed = self.store.load(session_id)
                budget = self._context_budget(
                    session=refreshed,
                    snapshot=snapshot,
                    prospective_prompt=prospective_prompt,
                )
                if budget["estimated_input_tokens_upper_bound"] <= target_tokens:
                    break
        status = self.context_status(session_id)
        return {
            "schema": "agent_harness.compaction_result.v1",
            "session_id": session_id,
            "status": "compacted" if latest_record is not None else "not_needed",
            "passes": passes,
            "compaction_id": (
                latest_record.get("compaction_id") if latest_record is not None else None
            ),
            "source_message_count": (
                latest_record.get("source_message_count")
                if latest_record is not None
                else status["compacted_message_count"]
            ),
            "summary_sha256": (
                latest_record.get("summary_sha256")
                if latest_record is not None
                else status["summary_sha256"]
            ),
            "summary_chars": status["summary_chars"],
            "active_message_count": status["active_message_count"],
            "transcript_message_count": status["transcript_message_count"],
            "estimated_input_tokens_upper_bound": status[
                "estimated_input_tokens_upper_bound"
            ],
        }

    def compact_session(
        self,
        session_id: str,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Manually compact old turns while preserving the original transcript."""

        token = cancellation_token or CancellationToken()
        with self.store.workspace_run_lock():
            unresolved = self.store.unresolved_workspace_runs()
            if unresolved:
                raise SessionStoreError(
                    "workspace has an unresolved run; reconcile it before compacting"
                )
            with self.store.turn_lock(session_id):
                snapshot = self.instruction_snapshot()
                return self._compact_locked(
                    session_id,
                    trigger="manual",
                    snapshot=snapshot,
                    cancellation_token=token,
                )

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
        approval_broker: ApprovalBroker | None = None,
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
                    approval_broker=approval_broker,
                )

    def _run_turn_locked(
        self,
        session_id: str,
        prompt: str,
        *,
        cancellation_token: CancellationToken | None = None,
        event_sink: EventSink | None = None,
        approval_broker: ApprovalBroker | None = None,
    ) -> TurnOutcome:
        content = str(prompt).strip()
        if not content or len(content) > 200_000:
            raise ValueError("prompt must contain 1 to 200000 characters")
        token = cancellation_token or CancellationToken()
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
        hook_snapshot = self.hook_snapshot()
        hook_trust_state = self.store.hook_trust_state()
        # This resolves every project proposal and verifies sandbox support
        # before automatic compaction can make a provider request.
        hook_runner = TrustedHookRunner(hook_snapshot, hook_trust_state)
        permissions = permission_profile(mode)
        scopes = {"internal", "user_input", "workspace_read"}
        if (
            "workspace.write" in permissions
            or "process.exec.sandboxed" in permissions
            or "process.exec.host" in permissions
        ):
            scopes.add("workspace_write")
        if "process.exec.host" in permissions:
            scopes.add("host_access")
        approval_policy = ApprovalPolicy(
            rules=(
                self.store.approval_rules(session_id)
                if mode != "read-only"
                else ()
            )
        )
        instruction_snapshot = self.instruction_snapshot()
        budget = self._context_budget(
            session=existing,
            snapshot=instruction_snapshot,
            prospective_prompt=content,
        )
        if (
            budget["estimated_input_tokens_upper_bound"]
            >= budget["automatic_trigger_tokens"]
        ):
            self._compact_locked(
                session_id,
                trigger="automatic",
                snapshot=instruction_snapshot,
                cancellation_token=token,
                target_tokens=budget["automatic_target_tokens"],
                prospective_prompt=content,
            )
        run_id = f"run_{uuid4().hex}"
        turn_id = f"turn_{uuid4().hex}"
        session = self.store.begin_run(
            session_id,
            run_id=run_id,
            turn_id=turn_id,
            user_content=content,
            provider="deepseek",
            model=self.client.config.model,
            permission_mode=mode,
        )
        context_view = self.store.context_view(session_id)
        context_lineage = dict(context_view["lineage"])
        history_summary = (
            {
                "content": context_view["summary"],
                "lineage": {
                    key: context_lineage[key]
                    for key in (
                        "compaction_id",
                        "source_message_count",
                        "source_messages_sha256",
                        "summary_sha256",
                        "active_context_sha256",
                    )
                },
            }
            if context_view["summary"]
            else None
        )
        event_path, checkpoint_path = self.store.run_paths(run_id)
        journal = HarnessJournal(
            event_path,
            run_id=run_id,
            turn_id=turn_id,
            checkpoint_path=checkpoint_path,
        )
        try:
            result = run_agent_harness(
                self.model,
                self.registry,
                {
                    "messages": [
                        {"role": item["role"], "content": item["content"]}
                        for item in context_view["messages"]
                    ],
                    "workspace": ".",
                    "active_directory": instruction_snapshot.active_relative_path,
                    "permission_mode": mode,
                    "project_instructions": (
                        instruction_snapshot.to_model_content()
                        if instruction_snapshot.documents
                        else ""
                    ),
                    "instruction_snapshot": instruction_snapshot.metadata(),
                    "history_summary": history_summary,
                    "context_lineage": context_lineage,
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
                cancellation_token=token,
                event_sink=event_sink,
                journal=journal,
                principal_id="local-user",
                session_id=session_id,
                trusted_data_scopes=scopes,
                approval_policy=approval_policy,
                approval_broker=approval_broker,
                tool_hook_broker=hook_runner,
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
        hook_effect_calls: set[str] = set()
        hook_guarded_settlements: set[str] = set()
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
            if event_type == "hook.effect_started":
                hook_effect_calls.add(call_id)
                continue
            if event_type in {
                "tool.completed",
                "tool.failed",
                "tool.rejected",
                "tool.replayed",
            } and call_id in hook_effect_calls:
                hook_guarded_settlements.add(call_id)
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
        unresolved_hook_effects = hook_effect_calls - hook_guarded_settlements
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
            "unsafe_hook_replay_blocked",
        }
        requires_reconciliation = terminal_reason_code in reconciliation_reasons
        checkpoint = result.get("checkpoint")
        if isinstance(checkpoint, Mapping):
            pending_effect = checkpoint.get("pending_effect")
        else:
            pending_effect = getattr(checkpoint, "pending_effect", None)
        if bool(unresolved_effects or unresolved_hook_effects) != isinstance(
            pending_effect, Mapping
        ):
            raise SessionStoreError(
                "runtime effect ledger does not match its authoritative terminal"
            )
        stored_status = (
            status
            if status in {"completed", "failed", "cancelled", "handoff"}
            else "failed"
        )
        self.store.finish_run(
            session_id,
            run_id=run_id,
            turn_id=turn_id,
            status=stored_status,
            assistant_content=message or None,
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
