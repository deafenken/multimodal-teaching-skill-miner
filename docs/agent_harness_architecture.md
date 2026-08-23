# Agent Harness architecture

## Layers

```text
TUI / headless CLI
        │
Session Store
        │
Harness Runtime
 ┌──────┼───────────────┐
Provider Adapter   Tool Registry   Event/Journal
                       │
                Workspace Toolset
```

The core is domain-neutral. It knows runs, turns, model decisions, tools, permissions,
events, budgets and recovery; application-domain models stay outside the package.

## Run protocol

1. Append the user message to the private workspace-scoped session.
2. Create a fresh run/turn identity and 0600 hash-chain journal.
3. Build the provider-visible tool list from the active permission profile and data-scope
   metadata. A scope label authorizes a handler; it does not sandbox that handler.
4. Ask the provider adapter for either central tool calls, a final answer or a handoff.
5. Validate and settle tools centrally. Every effect has typed lifecycle events.
6. Feed bounded observations to the next model step.
7. Stream final assistant text, commit one terminal event and atomically checkpoint.
8. Append only the authoritative final message to the session transcript.

## Invariants

- `run.started` is the first event and sequence is continuous.
- Exactly one `run.completed|cancelled|failed|handoff` event exists.
- No event is exposed before its durable journal acknowledgement.
- Tool authorization is the intersection of permission and trusted data scope. This is a
  registry decision, not filesystem, network, process or memory isolation.
- Hidden planner/reasoning content is never assistant output.
- Retry is suppressed after visible provider output or an external effect.
- The core can classify `safe`, `idempotent` and `never` replay policies. At the current
  product boundary, an unfinished or uncertain effect creates a workspace-wide fence and
  requires operator inspection; no entry point automatically proves, rolls back or safely
  repeats the effect.
- Resume must match the original context and complete execution-policy digest.

## Interfaces

- `harness`: interactive TUI when stdin/stdout are terminals.
- `harness exec`: streaming headless output.
- `harness exec --jsonl`: canonical event envelopes plus one exec-result record.
- `harness sessions|resume|fork|archive`: local session lifecycle.
- `harness effects` and `harness reconcile RUN_ID`: list workspace-wide unresolved runs
  and record a manual acknowledgement after inspection.

## Permission profiles

- `read-only`: workspace list/read/search.
- `workspace-write`: read-only tools plus unified patch application.
- `full-access`: workspace-write plus shell execution.

`full-access` is a host-level grant, not a workspace sandbox. The command starts with the
workspace as its working directory and a minimal environment, but it retains the current
OS user's filesystem, network and process authority. It can read and output workspace
secrets. Killing the initial process group is best effort: `setsid`, double-fork and other
daemonization patterns can escape it and continue after the parent run settles.

Session resume restores settled transcript state and always defaults back to `read-only`.
Session fork copies transcript and metadata only; it does not isolate environment variables,
provider clients, process memory or worktrees. An unfinished/uncertain run in any session,
including an archived session, blocks new runs across the whole workspace. `/effects` or
`harness effects` lists it; `/reconcile RUN_ID` or `harness reconcile RUN_ID` records the
operator's acknowledgement and clears the fence as a handoff without verifying the effect.

The current profiles are run-level grants. Per-call approval brokers, MCP, hooks,
compaction and subagents are tracked as P1 in `docs/harness_parity_matrix.md`.
