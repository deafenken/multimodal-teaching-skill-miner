# Agent Harness architecture

## Layers

```text
TUI / headless CLI
        │
Session Store
        │
Harness Runtime
 ┌──────┼──────────────────────┐
Provider Adapter   Tool Registry       Event/Journal
                       │
              Trusted Hook Broker
                       │
                Workspace Toolset
```

The core is domain-neutral. It knows runs, turns, model decisions, tools, permissions,
events, budgets and recovery; application-domain models stay outside the package.

## Run protocol

1. Validate the session fence, securely snapshot project instructions and project hook
   definitions, and require an exact trust or disable decision for every hook digest.
2. Estimate the prospective active context; when it crosses the 80% threshold, append one
   or more bounded provider summaries until it approaches 60% or cannot advance safely.
3. Create fresh run/turn identities and atomically append both the user message and the
   unresolved run record. A crash after this commit therefore leaves a visible fence.
4. Create the 0600 hash-chain journal and bind instruction, hook-policy and context-lineage
   digests.
5. Build the provider-visible tool list from the active permission profile and data-scope
   metadata. A scope label authorizes a handler; OS isolation is enforced separately.
6. Ask the provider adapter for either central tool calls, a final answer or a handoff.
7. Validate schema/replay first, then synchronously run matching `PreToolUse` hooks. Their
   aggregate can preserve the central policy, require approval or deny; it cannot allow a
   call that central policy would otherwise ask for or deny.
8. Resolve approval before `tool.started` and the durable tool effect boundary. Approval
   events contain digests, not raw command or patch text.
9. Validate and settle tools centrally. After the final success or failure, synchronously
   run the observe-only `PostToolUse` or `PostToolUseFailure` hook before emitting the final
   tool settlement. Every tool and hook effect has typed lifecycle evidence.
10. Feed bounded observations to the next model step.
11. Stream final assistant text, commit one terminal event and atomically checkpoint.
12. Atomically append the authoritative assistant message and settle the session run.

## Invariants

- `run.started` is the first event and sequence is continuous.
- Exactly one `run.completed|cancelled|failed|handoff` event exists.
- No event is exposed before its durable journal acknowledgement.
- Tool authorization is the intersection of permission and trusted data scope. This
  registry decision is independent from the built-in path checks and command sandbox.
- Hidden planner/reasoning content is never assistant output.
- Retry is suppressed after visible provider output or an external effect.
- The core can classify `safe`, `idempotent` and `never` replay policies. At the current
  product boundary, an unfinished or uncertain effect creates a workspace-wide fence and
  requires operator inspection; no entry point automatically proves, rolls back or safely
  repeats the effect.
- Resume must match the original context and complete execution-policy digest.
- Project instruction text is model context only; it cannot mutate permissions, scopes,
  approval policy, tool manifests or sandbox behavior.
- Project hook configuration is executable policy only after a private trust record matches
  the exact definition digest. A modified definition blocks new runs until it is explicitly
  trusted or disabled again.
- Hooks are monotonic. `PreToolUse` can return only `pass`, `ask` or `deny`, ordered
  `deny > ask > pass`; post-tool hooks are observe-only. No hook can grant authority,
  replace tool input, weaken scope/approval rules or select a less restrictive sandbox.
  A hook `ask` requires a fresh decision and cannot be bypassed by an existing persistent
  allow rule. That approval is once-only: the TUI exposes yes/no but cannot save a
  session/workspace allow from a hook-triggered challenge.
- Hook stdin/stdout/stderr are not event payloads. Hook events bind the definition, input and
  output by SHA-256 and retain only content-free identity, timing, action and error metadata.
- Approval is bound to run/call/tool version/argument digest/policy digest. Rejection is
  always before `tool.started`; headless ask decisions become an explicit handoff.
- Original messages and context-compaction records are append-only. An active view contains
  the latest lossy summary and only the uncovered message suffix; the full transcript is
  retained and each summary is bound to a stable-ID prefix digest and parent digest.
- A history summary is user data inside the provider JSON envelope, never a system message
  or a new authorization source. Planner and answer receive the same summary and suffix.

## Interfaces

- `harness`: interactive TUI when stdin/stdout are terminals.
- `harness exec`: streaming headless output.
- `harness exec --jsonl`: canonical event envelopes plus one exec-result record.
- `harness sessions|resume|fork|archive`: local session lifecycle.
- `harness effects` and `harness reconcile RUN_ID`: list workspace-wide unresolved runs
  and record a manual acknowledgement after inspection.
- `harness instructions`: inspect the exact instruction file set and digests without
  requiring provider credentials.
- `harness context SESSION_ID`: inspect content-free active-context counts, budget and hashes.
- `harness compact SESSION_ID`: manually summarize eligible completed old turns without
  deleting the transcript. The TUI exposes the same operations as `/context` and `/compact`.
- `harness hooks [--json]`: inspect content-free hook definitions, exact digests and trust
  status. `hooks trust|disable ... --sha256 DIGEST` records an exact decision and
  `hooks revoke HOOK_ID` removes it. TUI `/hooks` is inspection-only.

## Trusted command-hook subset

The only executable project source is `.agent-harness/hooks.json`. It can declare synchronous
`PreToolUse`, `PostToolUse` and `PostToolUseFailure` commands whose entrypoints stay below
`.agent-harness/hooks/`. This is a narrow tool-lifecycle subset, not full Claude Code or Codex
hooks parity: there are no other run/session/model lifecycle hooks, asynchronous hooks,
argument rewrites or configuration-compatibility claim.

Config, directories and entrypoints are opened relative to a no-follow workspace descriptor.
Symlinked intermediates; symlinked, non-regular, multi-linked or foreign-owned files;
group/other-writable hook directories or files; oversize data; and inode/device/size/time
changes during the read are rejected. A definition digest binds the exact config bytes,
event, matcher, arguments, timeout and entrypoint bytes.
Trust and disable records live in the 0600 private workspace state outside the repository;
repository contents cannot write their own trust decision through the Harness interface.

The runner materializes the already-snapshotted entrypoint bytes and invokes them through a
dedicated macOS Seatbelt profile. The hook gets a read-only workspace view, no network or
outgoing signals, no process fork, no access to protected `.git`/`.private`/`.agent-harness`
paths, and writes only to its per-call private runtime directory and `/dev`. The profile is
still an allow-default macOS host policy rather than a container or complete host-read/IPC
boundary. There is no unsupported-platform or unsandboxed fallback: if any definition is
trusted and Seatbelt is unavailable, the run fails before compaction or any provider request.

Input is a bounded JSON object on stdin. It contains the raw tool input and, for successful
post-tool observation, the bounded tool result. Output is a bounded strict JSON decision;
post-tool decisions other than `pass` are contract failures. Ordinary pre-hook spawn,
sandbox, timeout/output or contract failures tighten to deny; post-hook failures are recorded
but do not change the tool settlement. Cancellation propagates; timeout and deadline failures
follow the same bounded, fail-closed event policy. A crash after `hook.effect_started` is
treated conservatively by the existing unresolved-effect fence; the Harness does not
automatically replay a hook that may already have acted.

## Permission profiles

- `read-only`: workspace list/read/search.
- `workspace-write`: read-only tools plus, when macOS Seatbelt is available, unified patch
  application and `process.exec`. Both patch phases and commands use the same policy class.
- `full-access`: workspace-write plus the separate unsandboxed `process.exec_host` tool.

`workspace.patch` and the sandboxed command tool are registered only when the macOS Seatbelt
backend is available; other platforms fail closed instead of substituting an unsafe writer.
The deny-overlay policy confines writes to the workspace, per-call runtime directory and
`/dev`; it also denies network, outgoing signals, reads below known user-data roots outside
the workspace, and lookup of known `com.apple.security*`/`SecurityServer` Mach services.
The resolved-target write rule is the final boundary for a patch path swapped to an external
symlink after preflight. Descendants inherit the policy, although process-group cleanup
remains best effort.

This Seatbelt profile begins with `allow default`. It is a macOS host policy, not a container,
VM, complete host-read boundary or proof that future credential-service names are covered.
`process.exec_host` starts in the workspace with a minimal environment but retains the current
OS user's filesystem, network and process authority. It can read and output workspace secrets,
and daemonization can escape the Harness process-group cleanup.

Session resume restores settled transcript state and always defaults back to `read-only`.
Session fork copies transcript, compaction lineage and metadata only; it does not isolate
environment variables, provider clients, process memory or worktrees. An unfinished/uncertain run in any session,
including an archived session, blocks new runs across the whole workspace. `/effects` or
`harness effects` lists it; `/reconcile RUN_ID` or `harness reconcile RUN_ID` records the
operator's acknowledgement and clears the fence as a handoff without verifying the effect.

The private approval policy can add tool-wide deny/ask rules and exact allow rules. TUI
decisions support once/session/workspace scopes; only the exact tool version and canonical
argument digest persist. Approval events omit raw arguments, but private checkpoints may
retain a pending call so safe/idempotent recovery can be evaluated; users should not place
secrets in command or patch arguments. Full hooks parity, MCP and subagents remain in the
staged backlog in `docs/harness_parity_matrix.md`.
