# Agent Harness architecture

## Layers

```text
TUI / headless CLI
        │
Session Store
        │
Harness Runtime
 ┌──────┼─────────────────────────────┐
Provider Adapter      Tool Registry          Event/Journal
                 ┌────────┼─────────────┐
        Trusted Hook   Frozen MCP   Foreground Subagents
           Broker       Catalog             │
                 └────────┼────────── isolated worktrees
                          │
                 Workspace Toolset / Seatbelt
```

The core is domain-neutral. It knows runs, turns, model decisions, tools, permissions,
events, budgets and recovery; application-domain models stay outside the package.

## Run protocol

1. Validate the session fence; securely snapshot project instructions, hook definitions and
   local MCP definitions; require an exact trust/disable decision for every executable
   proposal and a current explicitly refreshed catalog for every trusted MCP server.
2. Estimate the prospective active context; when it crosses the 80% threshold, append one
   or more bounded provider summaries until it approaches 60% or cannot advance safely.
3. Create fresh run/turn identities and atomically append both the user message and the
   unresolved run record. A crash after this commit therefore leaves a visible fence.
4. Create the 0600 hash-chain journal and bind instruction, hook-policy, frozen MCP-policy/
   catalog and context-lineage digests.
5. Build the provider-visible tool list from the active permission profile and data-scope
   metadata. Trusted frozen MCP tools join it only in `full-access`; a scope label authorizes
   a handler while OS isolation is enforced separately.
6. Ask the provider adapter for either central tool calls, a final answer or a handoff.
7. Validate schema/replay first, then synchronously run matching `PreToolUse` hooks. Their
   aggregate can preserve the central policy, require approval or deny; it cannot allow a
   call that central policy would otherwise ask for or deny.
8. Resolve approval before `tool.started` and the durable tool effect boundary. Approval
   events contain digests, not raw command or patch text. MCP calls are always high-risk,
   once-only approvals and cannot inherit or create a persistent allow.
9. Validate and settle tools centrally. After the final success or failure, synchronously
   run the observe-only `PostToolUse` or `PostToolUseFailure` hook before emitting the final
   tool settlement. Every tool and hook effect has typed lifecycle evidence.
10. If the selected tool is `agent.delegate`, approve the exact batch once, atomically reserve
    the in-process fan-out budget, create one clean-HEAD Git worktree per child behind the
    repository-common lock, run 1–4 child sessions concurrently, propagate cancellation and
    join them all. Return only bounded summaries/opaque IDs; never merge child changes.
11. Feed bounded observations to the next model step.
12. Stream final assistant text, commit one terminal event and atomically checkpoint.
13. Atomically append the authoritative assistant message and settle the session run.

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
- Project MCP configuration is executable only after a private record matches the exact
  server-definition digest and an explicit refresh has frozen a bounded `2025-06-18` tools
  catalog. The run policy binds both definition and catalog digests. A live catalog change is
  rejected; it never changes the active run's tool surface implicitly.
- MCP tools are `full-access`, external-service, high-risk, serial and `never` replay. They
  require fresh user consent for every call and cross the durable effect boundary before the
  stdio process starts.
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
- `agent.delegate` is high-risk, never replayed and never persistently approved. Its child
  authority is a dynamic subset of the parent and is capped at `workspace-write`; child
  runners have no host command, MCP, command hooks, persistent approval or nested delegation.
- One subagent batch is foreground wait-all. Every started child settles or is cancelled and
  joined before the parent tool settles. Budget counters are per runner process; only Git
  common-directory mutation locking is cross-process.
- Child summaries and lifecycle are separate contracts: the provider may see bounded summary
  evidence, while public progress contains only batch `count`/`depth` and child opaque
  `agent_id`/`depth`/`ordinal`/`status`. Neither channel contains hidden reasoning, raw exceptions
  or worktree paths.
- A worktree can be automatically removed only when clean Git status, no-follow manifest,
  administrative mapping, exact lock reason and baseline ref all agree. Otherwise any artifact
  that still exists is preserved. If checkout removal has already succeeded and only ref CAS
  deletion fails, the branch and `ref_preserved` record remain without a checkout. There is no
  force/reset/clean/prune or automatic merge/apply/commit/push/PR path.

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
- `harness mcp [--json]`: inspect exact stdio launch definitions, authority flags, trust and
  catalog state. `mcp trust|disable ... --sha256 DIGEST` records an exact decision,
  `mcp refresh SERVER_ID` freezes a catalog and `mcp revoke SERVER_ID` removes trust. TUI
  `/mcp` is inspection-only and never starts a server.
- `harness agents [WORKTREE_ID] [--path] [--json]`: inspect preserved child artifacts without
  constructing a provider. Lists and normal detail are path-free; only one exact ID plus
  `--path` reveals its worktree path. TUI `/agents` combines content-free live status and
  retained artifact metadata.

## Trusted command-hook subset

The only project hook source is `.agent-harness/hooks.json`. It can declare synchronous
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

## Exact-trust local MCP stdio tools subset

`.agent-harness/mcp.json` proposes local stdio server definitions; proposal discovery and
status inspection never start a process. The implementation is a tools-only MCP client subset
pinned to protocol `2025-06-18`. Every definition must be exactly trusted or disabled in the
0600 private workspace state. Trusted definitions are still inactive until an explicit
`harness mcp refresh SERVER_ID` starts the server, initializes it and stores a canonical,
bounded tool catalog outside the repository.

A definition digest binds the exact config bytes, server ID, stdio transport, direct
executable bytes and stat identity, argv, workspace-confined cwd, allowed environment names,
network/fork flags, startup/tool timeouts and sandbox-policy version. The direct executable is
revalidated before refresh and call. This does not attest transitive integrity: scripts named
only in interpreter arguments, imports, packages, shared libraries, runtime files,
environment values and files opened later are outside that digest.

The protocol engine accepts strict, bounded newline-delimited JSON-RPC objects. It implements
initialize/initialized, paginated `tools/list`, `tools/call`, cancellation, server `ping`
responses and `notifications/tools/list_changed` invalidation. It freezes canonical accepted
tool definitions and content-free rejected-tool records. At call time it performs a fresh
handshake/list and requires the live catalog digest to equal the stored catalog, so catalog
change fails closed instead of mutating an active registry.

Remote input/output schemas must be object-root schemas using only the Harness-supported
local validation keywords. Text and object-shaped `structuredContent` are normalized;
non-text content is reduced to type/length/MIME/digest metadata and is not rendered. Remote
`isError` is a settled tool result. Unsupported keywords/tools are excluded during explicit
refresh. HTTP, OAuth, resources, prompts, sampling, elicitation, tasks, input-required/task
results, active-run dynamic catalogs, full JSON Schema, binary rendering and MCP server mode
are outside this version.

The stdio process runs under a read-only macOS Seatbelt profile with protected project reads
denied and writes limited to a private runtime HOME/TMP and `/dev`. Network and process fork
default to denied and are enabled only by exact-digest-bound flags. There is no unsandboxed or
unsupported-host fallback. This remains an allow-default host policy, not a container or
complete host confidentiality boundary. With fork enabled, daemonization may escape
best-effort process-group cleanup while retaining the granted Seatbelt/network authority.

Accepted MCP tools are registered only in `full-access` with `mcp.external`, external-service
and remote-consent scope. Each is high-risk, nonparallel and `never` replay, and requires a
fresh once-only approval that cannot be replaced by a session/workspace allow rule. The
durable tool effect boundary is crossed before the stdio process starts.

Raw MCP arguments and normalized results use the ordinary owner-only checkpoint/journal
contract; the result may become a later provider observation. Raw stderr is drained and
hashed transiently but never persisted. Only byte count, truncation state and SHA-256 may be
returned to a local refresh caller. Server instructions/info are represented in the frozen
catalog by digests, not raw text.

## Foreground subagent and worktree subset

The model-visible surface is one `agent.delegate` tool whose schema accepts 1–4 unique task
IDs, bounded prompts and a requested `read-only`/`workspace-write` child mode. A thread-safe
ledger atomically enforces batch, global-active, parent-active, depth and total-per-root
limits. The shipped configuration has maximum depth one and maximum concurrency/total four.
The scheduler starts true concurrent workers but returns their results in input order. Parent
cancellation is fanned out, reconciliation in any child cancels its siblings, and executor
shutdown waits for every child before the tool can settle.

The runner adapter allocates a fresh child session in a dedicated Git worktree, using the same
provider/model without inheriting the parent's transcript. It receives the direct delegated
prompt, bounded prompt/output/path-free lineage metadata (including the validated caller task
ID), the standard Harness system context and the child's ordinary project-instruction snapshot.
Executable project hooks, MCP, host commands and further subagents are disabled. Persistent
approval is globally capped off in the child execution-policy hash. The once-only child broker
can approve only a patch or sandboxed command that already passed the child's registry,
permission, scope and schema checks.

`WorktreeManager` accepts only an exact clean committed `HEAD`. It writes a provisional 0600
record before Git mutation, creates opaque worktree/branch identities with `--no-checkout`,
performs a controlled checkout with project execution/config extensions disabled, verifies
the linked-worktree administrative mapping and exact lock reason, then stores a bounded
no-follow manifest. All Git calls use a trusted absolute executable, direct argv, scrubbed
environment and a common-Git-directory lock. The state directory, worktree home and source
repository must be canonical owner-controlled, disjoint paths.

On successful child completion the manager attempts only `remove_if_pristine`. Any modified
status, manifest, pre-remove ref, mapping, lock or phase preserves artifacts that still exist.
A post-remove ref race uses compare-and-swap deletion and preserves the branch plus a
`ref_preserved` record on mismatch, but the checkout has already been removed. The public runner
projection excludes repository, common Git directory, worktree/Git paths, branch refs and lock
reason. This is file-collision isolation, not process/credential/network security isolation,
and it intentionally omits background threads, resume/steer, custom agents, automatic
integration and teams.

## Permission profiles

- `read-only`: workspace list/read/search plus high-risk delegation to read-only children.
- `workspace-write`: read-only tools plus, when macOS Seatbelt is available, unified patch
  application and `process.exec`; it may delegate read-only or worktree-write children. Both
  patch phases and commands use the same policy class.
- `full-access`: workspace-write plus the separate unsandboxed `process.exec_host` tool and
  exact-trusted/frozen local MCP tools when their Seatbelt backend is available. Delegated
  children remain capped at `workspace-write` and never inherit those two full-access tools.

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
environment variables, provider clients, process memory or worktrees. An unfinished/uncertain
run in any session, including an archived session, blocks new runs across the whole workspace. `/effects` or
`harness effects` lists it; `/reconcile RUN_ID` or `harness reconcile RUN_ID` records the
operator's acknowledgement and clears the fence as a handoff without verifying the effect.

The private approval policy can add tool-wide deny/ask rules and exact allow rules. TUI
decisions support once/session/workspace scopes; only the exact tool version and canonical
argument digest persist. Approval events omit raw arguments, but private checkpoints may
retain a pending call so safe/idempotent recovery can be evaluated; users should not place
secrets in command, patch or MCP arguments. Full hooks parity, broader MCP transports/features
and broader background/configurable agent orchestration remain staged in
`docs/harness_parity_matrix.md`; the shipped subagent surface is only the bounded foreground
worktree subset described above.
