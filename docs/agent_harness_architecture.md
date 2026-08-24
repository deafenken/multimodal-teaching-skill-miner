# Agent Harness architecture

## Layers

```text
TUI / headless CLI ───── TypeScript SDK (bounded JSONL subprocess)
          │
Python SDK (in process)
          │
Session Store ────────── Immutable Attachment Store
          └──────┬──────┘
                 │
         Harness Runtime ─┬─ Provider Adapter
                          ├─ Event / Journal
                          └─ Tool Registry ─┬─ Trusted Hook Broker
                                           ├─ Frozen MCP Catalog
                                           └─ Foreground Subagents
                                                      │
                                               isolated worktrees
                                                      │
                                           Workspace Toolset / Seatbelt
```

The core is domain-neutral. It knows runs, turns, model decisions, tools, permissions,
events, budgets and recovery; application-domain models stay outside the package.

## Run protocol

1. Validate the session fence and reject an unfinished or uncertain prior run.
2. Validate every prospective attachment descriptor against its owner-private immutable blob,
   the per-turn/active-request bounds and the exact selected provider/model capability. A
   missing, modified or unsupported attachment fails before compaction, run creation or a
   network request.
3. Securely snapshot project instructions, hook definitions and local MCP definitions; require
   an exact trust/disable decision for every executable proposal and a current explicitly
   refreshed catalog for every trusted MCP server.
4. Estimate the prospective active context, including conservative attachment tokens; when it
   crosses the 80% threshold, append one
   or more bounded provider summaries until it approaches 60% or cannot advance safely.
5. Create fresh run/turn identities and atomically append both the user message, its content-free
   attachment manifest and the
   unresolved run record. A crash after this commit therefore leaves a visible fence.
6. Create the 0600 hash-chain journal and bind instruction, hook-policy, frozen MCP-policy/
   catalog and context-lineage digests.
7. Build the provider-visible tool list from the active permission profile and data-scope
   metadata. Trusted frozen MCP tools join it only in `full-access`; a scope label authorizes
   a handler while OS isolation is enforced separately.
8. Ask the provider adapter for either central tool calls, a final answer or a handoff.
9. Validate schema/replay first, then synchronously run matching `PreToolUse` hooks. Their
   aggregate can preserve the central policy, require approval or deny; it cannot allow a
   call that central policy would otherwise ask for or deny.
10. Resolve approval before `tool.started` and the durable tool effect boundary. Approval
   events contain digests, not raw command or patch text. MCP calls are always high-risk,
   once-only approvals and cannot inherit or create a persistent allow.
11. Validate and settle tools centrally. After the final success or failure, synchronously
   run the observe-only `PostToolUse` or `PostToolUseFailure` hook before emitting the final
   tool settlement. Every tool and hook effect has typed lifecycle evidence.
12. If the selected tool is `agent.delegate`, approve the exact batch once, atomically reserve
    the in-process fan-out budget, create one clean-HEAD Git worktree per child behind the
    repository-common lock, run 1–4 child sessions concurrently, propagate cancellation and
    join them all. Return only bounded summaries/opaque IDs; never merge child changes.
13. Feed bounded observations to the next model step.
14. Stream final assistant text, commit one terminal event and atomically checkpoint.
15. Atomically append the authoritative assistant message and settle the session run.

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
- Attachment descriptors bind an immutable local blob by opaque ID, exact size and SHA-256.
  Attachment manifests and attachment-specific event metadata do not copy the source path,
  raw body or base64. The sanitized basename is local display metadata, not a path or authority
  source. Assistant/tool output that quotes an attachment follows its ordinary persistence
  contract.
- A compaction range cannot cover or cross an attachment-bearing message. That message and
  its following active suffix remain live and may therefore reach the context limit rather
  than losing the attachment relationship through summarization.
- Attachment text, documents and images are untrusted user data. Instructions visible inside
  them cannot become system/project policy, tools, permissions, scopes or approval authority.
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
- A Python SDK thread carries one explicit immutable permission mode. One client-wide operation
  lock covers permission reapplication, attachment import, execution and conservative cleanup,
  so concurrently held thread objects cannot borrow each other's authority. Resume and logical
  fork default back to `read-only` unless the application explicitly selects another mode.
- The TypeScript SDK is a bounded consumer of the CLI JSONL contract, not another runtime
  implementation. It accepts a lazy new-thread identity only from a fully validated exec-result
  record, serializes runs on one thread, and requires contiguous events, one matching terminal,
  invariant run/turn/session identity and the matching process exit code. It never reconstructs
  hidden reasoning as an answer.

## Interfaces

- `agent_harness.sdk`: typed in-process Python sync/async clients. They support persisted
  start/resume/fork threads, run/event-stream methods, immutable attachment snapshots, cooperative
  cancellation and an optional caller-supplied approval broker while preserving `AgentRunner`
  storage and execution policy.
- `@agent-harness/sdk` below `sdk/typescript/`: dependency-free Node.js 18+ strict-ESM client
  for `harness exec --jsonl`. It supports lazy start, resume, run/event stream, repeated
  attachments, explicit permission mode and `AbortSignal`; it does not implement thread fork or
  an approval broker.
- `harness`: interactive TUI when stdin/stdout are terminals.
- `harness exec`: streaming headless output.
- `harness exec --jsonl`: canonical event envelopes plus one exec-result record.
- `harness exec --attach PATH`: repeatable immutable attachment input for one headless turn.
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

## SDK boundaries

The Python SDK is an authority-preserving facade over one in-process `AgentRunner`, not a
remote protocol. `HarnessClient.start_thread`, `resume_thread` and `fork_thread` return typed
handles; a handle can `run`, `run_stream` or perform its logical `fork`. `AsyncHarnessClient`
and `AsyncHarnessThread` place the same blocking runner behind non-abandoning worker-thread
boundaries. Async task cancellation first cancels the Harness token and waits for the worker to
settle, so it is not represented as proof that an already-started external effect was rolled
back. A sync/async event stream must be exhausted or explicitly closed; close cancels and joins
its non-daemon worker. Each stream permits one blocking consumer operation at a time and rejects
concurrent consumers instead of allowing them to compete for the terminal sentinel. Concurrent
close callers all wait for the same worker settlement. Cancellation runs every registered
cleanup callback; an application callback's fatal `BaseException` is retained and propagated
only after the synchronous or asynchronous worker boundary has settled.

One Python client deliberately serializes all thread operations because `AgentRunner` has a
mutable live permission field. Before every run it reapplies the thread's immutable explicit
permission, then imports that turn's attachments and performs conservative unreferenced-blob
cleanup on failure under the same lock. Start can select a permission; resume and fork default
to `read-only`. A supplied `ApprovalBroker` executes in the embedding process and is the only
SDK route that can answer an interactive approval. Without one, the ordinary headless policy
remains fail closed. Event callbacks receive detached immutable event values and are observers,
not an authorization channel.

The TypeScript SDK does not import Python or `AgentRunner`. It uses Node's `spawn()` with
`shell: false`, sends the prompt through stdin, passes repeated attachments as absolute
`--attach` values and consumes stdout as canonical bounded UTF-8 JSONL. Configurable line,
record, total-output and reconstructed-response limits are capped. It validates exact event
envelopes, contiguous sequence, invariant run/turn identity, one terminal event, the final
`agent_harness.exec_result.v1`, session identity and the expected exit code. `finalResponse`
concatenates only non-internal `message.delta` values. Stderr is drained and discarded; errors
retain only safe process/protocol metadata.

`startThread()` in TypeScript is intentionally lazy: its ID is `null` until the first valid
exec-result establishes a durable session. `resumeThread()` accepts a syntactically bounded ID;
the CLI/session store remains authoritative when the next run opens it. Runs on one handle are
serialized in submission order, while different handles can spawn concurrent processes. The
SDK has no `forkThread` and no headless approval broker. Per-run permission override and
`AbortSignal` are explicit application choices; SIGINT followed by bounded SIGKILL fallback is
lifecycle control, not effect rollback or containment. Signal listener registration completes
before spawn. After a syntactically valid exec-result, a separate bounded watchdog requires the
child to close; timeout terminates it and fails the run as a protocol error even if its result
record otherwise matched. Before the result, a capped transport timer is derived from the
Harness deadline unless explicitly narrowed; stdout EOF without a result makes further protocol
progress impossible and triggers bounded termination. Observation-only callback and listener
thenables have their rejections consumed so they cannot become unhandled host-process failures.
All of these lifecycle paths request cooperative CLI cancellation with SIGINT before the bounded
SIGKILL fallback; they do not use SIGTERM to bypass Runner child/effect settlement.

Public terminal status is the four-value contract `completed`, `cancelled`, `handoff` or
`failed`. Internal deadline exhaustion therefore appears publicly as `failed`, matching the
authoritative `run.failed` event and exit code, while `deadline_exceeded` remains available as
the detailed reason and internal runtime status.

The executable is part of the TypeScript trust boundary. A bare `harness` name is resolved by
the child-process environment's `PATH`; an absolute path is checked to resolve to an executable
file but is not content-digest pinned by the SDK. Deployments must review and version-pin the
binary/package and protect its path and `PATH` from less-trusted writers. The child inherits a
snapshot of the Node process environment plus explicit overrides. This interface is not the
Codex app-server JSON-RPC protocol and does not claim full API parity with the Claude Agent SDK
or either Codex SDK.

## Immutable attachment boundary

The attachment store treats the selected pathname as an untrusted import source. It walks
components without following user-controlled links, requires a stable regular file, reads
under a type-specific bound, derives the kind from bytes, and atomically publishes a mode-0600
blob below a mode-0700 workspace attachment directory. The public
`agent_harness.attachment.v1` descriptor contains an opaque ID, kind, MIME type, sanitized
basename, byte size, SHA-256, conservative token estimate and image dimensions where
applicable. It contains no source path or body. A session fork copies descriptors and shares
the same workspace blob store; it is not a new confidentiality boundary or physical copy.

One turn can ingest at most 8 attachments and 24 MiB total. Individual strict UTF-8 text,
PNG/JPEG and PDF snapshots are limited to 2, 8 and 16 MiB respectively. PNG/JPEG parsers
validate the actual signature, bounded structure and dimensions. PDF handling only checks a
`%PDF-` header, a bounded `%%EOF` terminator and size; it does not parse, render, OCR or extract
the document. These ingestion facts do not imply that the selected provider can consume the
kind.

The provider seam declares exact attachment kinds, MIME types, count and byte limits. The
current DeepSeek adapter accepts strict UTF-8 text on ordinary text models. It accepts
PNG/JPEG only for the exact, explicitly configured `deepseek-v4-flash-vision-exp` model, with
an active maximum of 16 attachments and 24 MiB. The Harness never changes models
automatically. Image blocks are inline user-message data for that request; the current path
does not call DeepSeek Files API. PDF is unsupported by the adapter and fails before run
creation or network access even though its immutable local representation is valid.

Planner and final-answer phases receive the same expanded attachment inputs. Inline image
bytes and base64 are not copied into provider traces, attachment manifests or attachment-event
metadata; assistant/tool output may still describe them under the normal persistence contract.
Text/image content is still sent to
the selected remote provider when a turn proceeds, so attachment selection is a disclosure
decision. Content is always marked as untrusted user data. The curses TUI provides explicit
`/attach PATH`, `/attachments` and `/detach ID|all` staging bound to a pending turn; graphical
drag-and-drop, image paste and clipboard ingestion are outside this version.

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
