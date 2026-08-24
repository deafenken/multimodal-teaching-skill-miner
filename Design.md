# Agent Harness design

## Objective

Provide a small, auditable execution layer for coding agents, usable through a Python API,
headless CLI and terminal UI. Provider adapters decide between tool calls, a final answer
or an explicit handoff; the Harness owns permissions, tool settlement, events, recovery
and persistence.

## Control flow

```text
prompt
  -> session append
  -> provider plan
  -> central tool validation/execution
  -> bounded observation
  -> provider continuation
  -> authoritative assistant message
  -> terminal journal record
```

The loop is bounded by deadline, model-call, step, repeated-call and total-tool budgets.
Cancellation propagates to provider streams and attempts to stop the initial command
process group. It cannot contain a `full-access` command that escapes with `setsid`, a
double fork or another daemonization mechanism.

## Product surfaces

- `harness`: curses TUI with streaming output, tool activity, follow-up queue and
  session commands.
- `harness exec`: headless execution, with optional canonical JSONL events.
- `agent_harness.AgentRunner`: Python facade for embedding.

## Security

The default is `read-only`. `workspace-write` adds validated unified patches and
`full-access` adds bounded process execution. Path confinement applies to the built-in
list/read/search/patch handlers, not to the shell. `full-access` retains the current OS
user's host authority, including the ability to read and print workspace secrets.

`data_scope` is an authorization label, not a filesystem or process sandbox. Session fork
copies transcript state only; it does not isolate environment variables, provider state or
heap secrets. An unresolved run creates a workspace-wide fence. Operators must inspect
the journal and real effects before using `effects`/`reconcile`; acknowledgement clears the
fence but does not automatically verify or undo an effect.

See `docs/agent_harness_architecture.md` for invariants and
`docs/harness_parity_matrix.md` for the Claude Code/Codex comparison.
