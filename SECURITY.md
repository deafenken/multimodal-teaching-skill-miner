# Security policy

## Supported version

Security fixes target the current `2.x` Agent Harness line.

## Trust boundaries

- The default permission mode is `read-only`.
- `workspace-write` exposes validated unified patches and `process.exec` only when the macOS
  Seatbelt backend is available. Both run under a workspace-write, no-network policy.
- `full-access` additionally exposes the separately named `process.exec_host`; it must be an
  explicit user choice. This host command is not workspace-confined: it can exercise every
  permission already held by the current OS user, read workspace secrets or other files,
  use the network and place those bytes in bounded tool output that may be sent to the
  provider.
- The list/read/search/patch handlers reject absolute paths, `..`, symbolic links and
  protected `.git`, `.private` and `.agent-harness` roots. Those user-space checks do not
  constrain `process.exec_host`.
- Sandboxed patch and command execution use an `(allow default)` Seatbelt deny overlay. It
  denies external writes, network, outgoing signals, reads below known user-data roots
  outside the workspace, and lookup of known Keychain/securityd Mach services. It is a
  macOS host policy, not a container, VM, complete host-read boundary or proof that all
  present and future credential IPC names are covered. Unsupported hosts do not receive an
  unsandboxed replacement for these tools.
- Command execution uses a minimal environment, but this only removes named environment
  variables from the direct child. It does not protect secrets stored in files, inherited
  descriptors, subprocess state or the Harness process heap.
- Timeout and cancellation attempt to terminate the initial process group. A command can
  call `setsid`, double-fork or otherwise daemonize, escape that group and continue after
  the Harness reports completion, timeout or cancellation. Process-group cleanup is a
  best-effort lifecycle aid, not containment.
- Tool permission, data scope, risk, timeout, retry and replay policy are checked by the
  central executor, not by prompt wording. `data_scope` is authorization metadata only;
  it does not sandbox filesystem, network, process or memory access by a handler.
- Non-safe built-in handlers use a two-phase boundary: argument/path/preflight validation
  completes first, then a durable `tool.effect_started` event and checkpoint are written
  before the handler may patch or spawn a command. A crossed boundary without a successful
  settlement ends in the run's sole authoritative `run.handoff` terminal.
- `never` replay tools such as patch and command execution are not silently replayed after
  an uncertain crash boundary. Cancellation after such an effect begins settles as an
  explicit handoff.

### SDK embedding boundaries

The Python SDK is an in-process facade over `AgentRunner`. It preserves the Runner's session,
permission, approval, tool, sandbox, journal and unresolved-effect rules, but it does not
sandbox the embedding Python application. The application, its imports, event callbacks and
any caller-supplied `ApprovalBroker` already run with the host Python process's OS authority
and can read objects and credentials available in that process. Only pass a broker whose code
is trusted to inspect approval requests and make decisions; a broker can answer a central ask
but cannot turn a central schema, permission, scope or sandbox denial into authority.

A Python thread stores an explicit immutable permission mode. Resume and fork default to
`read-only`; selecting `workspace-write` or `full-access` is an application authority decision
and must not be derived directly from a prompt, attachment or other untrusted model content.
One client serializes its thread operations while it reapplies that mode, imports attachments,
runs and conservatively cleans up. This prevents authority races inside that client, not across
independent clients/processes and not against code already controlling the process. Event
callbacks receive detached event payloads that may contain model or ordinary tool output.
Streaming workers are non-daemon; callers must exhaust or close streams so cancellation and
join can finish. Application cancellation callbacks that raise `BaseException` are not treated
as successful cleanup: the SDK completes every registered cleanup and worker join first, then
re-raises the fatal callback exception to the caller.

The TypeScript SDK executes a separate `harness` binary. `harnessPath` is therefore executable
code trust, not data. The default bare name is resolved through the inherited `PATH`; an
absolute path is checked for an executable file but is not opened no-follow, digest-attested or
locked against later replacement. Production callers should use an absolute path to a
reviewed/version-pinned installation, protect every writable path component and avoid an
untrusted `PATH`. The SDK snapshots the Node process environment and applies requested
overrides before spawn; child-process credentials and loader controls present there remain part
of the launch authority.

Node uses direct argv and `shell: false`; the prompt is written to stdin rather than the process
argument list. Absolute attachment source paths are still passed as repeated `--attach`
arguments and can be visible to same-user process inspection before the CLI imports them. The
SDK validates bounded canonical UTF-8 JSONL, exact event types and identity, contiguous
sequence, one terminal, the exec-result record and exit-code agreement. These are protocol and
memory bounds, not confidentiality filtering: event callbacks and `finalResponse` can contain
model text or normal tool output, including sensitive content the run was authorized to read.
Child stderr is drained and discarded and only safe process metadata is exposed in SDK errors.

The TypeScript surface has no approval broker. It passes the explicit permission mode to the
headless CLI; an unresolved medium/high approval becomes handoff under the normal policy rather
than being silently accepted. Cancellation, transport and protocol watchdogs send cooperative
SIGINT first and, after a bounded grace period, SIGKILL. Abort listener registration is
completed before spawn so a rejected signal cannot
leave an untracked child. Protocol failure similarly terminates the child best-effort, and a
valid exec-result is followed by a bounded close watchdog so a lingering child cannot hold the
SDK promise indefinitely. A separate capped transport timeout bounds the period before that
record, and stdout EOF without a result triggers bounded termination immediately. Async event
callback and signal-listener rejections are consumed as observation failures rather than
becoming unhandled host-process rejections. Neither outcome proves
that an already-started external effect was undone; an unresolved session/workspace fence may
remain and must be inspected through the normal durable evidence. This subprocess/JSONL SDK is
not an app-server protocol or a containment boundary.

### Immutable attachment imports

An attachment pathname is an untrusted import source, not an authorization grant. The loader
requires an absolute or home-expanded path, walks components with no-follow descriptors,
rejects user-controlled symbolic links and non-regular sources, and checks file
identity/size/timestamps before and after a bounded read. It then publishes an immutable copy
as a mode-0600 blob inside a mode-0700 workspace attachment directory; published/read blobs
must have exactly one hard link. Descriptor size and SHA-256 are revalidated before provider
use. These owner-only modes protect against other OS
accounts under ordinary filesystem semantics; they do not isolate another process running as
the same user or an attacker who already controls the Harness state directory/process.

Type is derived from bytes rather than trusted from the suffix. Text must decode as strict
UTF-8. PNG/JPEG validation checks the actual signature, bounded structure and dimensions.
PDF validation only checks the `%PDF-`/`%%EOF` envelope and size limit; it is not a PDF parser,
renderer, malware scanner, OCR engine or content extractor. A turn is limited to 8 imports and
24 MiB total, with 2-MiB text, 8-MiB image and 16-MiB PDF item bounds. A selected provider's
active context has a separate exact capability/count/byte preflight; the DeepSeek adapter caps
active attachments at 16/24 MiB and rejects an unsupported, missing or changed blob before
compaction, run creation or network access.

Ordinary DeepSeek text models accept only strict UTF-8 text attachments. PNG/JPEG requires the
operator to select the exact `deepseek-v4-flash-vision-exp` model. There is no automatic model
switch and no Files API upload; supported image bytes are inlined in that provider request.
PDF is rejected by the current DeepSeek adapter before a run. Secure local ingestion therefore
does not imply provider support or semantic safety.

Attachment bodies, including text visible inside screenshots or documents, are untrusted user
data. They are delimited as such in provider input and cannot directly alter central tools,
permissions, scopes, approval rules, hooks, MCP trust or sandbox policy. They can still influence
the model to request an authorized tool, so normal schema, permission and approval enforcement
remains essential. Attachment manifests and attachment-specific event metadata never copy a
source path, raw body or base64; a sanitized basename is retained only as local display
metadata. Assistant or tool output may still quote attachment content and then follows the
ordinary transcript/event contract. The immutable blob itself remains sensitive local data
and may contain secrets.

### Trusted project command hooks

`.agent-harness/hooks.json` and files below `.agent-harness/hooks/` are untrusted repository
content until the operator makes an exact decision. Merely opening the workspace never
executes them. The loader uses descriptor-relative no-follow reads and rejects symlinks,
non-regular or multi-linked files, foreign-owned files, group/other-writable hook files or
intermediate hook directories, oversize data and files that change while being read.
Entrypoints are snapshotted as bytes before a run.

Each definition digest binds the exact config bytes, event, matchers, arguments, timeout and
entrypoint bytes. `harness hooks trust HOOK_ID --sha256 DIGEST` or `disable ... --sha256`
stores that exact decision in the 0600 private workspace state outside the repository. Any
bound change makes the definition `modified` and blocks new runs until the current digest is
explicitly trusted or disabled. `revoke` removes the decision. Digest binding detects change;
it is not a code signature or protection against an attacker who already controls both the
repository and the operator's private Harness state.

The implemented surface is only the synchronous `PreToolUse`, `PostToolUse` and
`PostToolUseFailure` subset. A pre hook can return `pass`, `ask` or `deny`, but `pass` cannot
turn a central ask/deny into allow and cannot change tool input. Multiple hooks aggregate as
`deny > ask > pass`; `ask` requires a fresh per-call decision and cannot be bypassed by an
existing persistent allow rule. A hook-triggered approval is once-only; the TUI does not
offer session/workspace persistence for that challenge. Post hooks are observe-only and must
return `pass`. Thus project hooks can only preserve or tighten permission, scope, approval
and sandbox decisions; they cannot grant authority.

Trusted hook snapshots execute under a dedicated Seatbelt profile with a read-only workspace,
no network, no outgoing signals, no process fork, protected `.git`/`.private`/`.agent-harness`
reads denied, and writes limited to a per-call private runtime directory and `/dev`. This
remains an allow-default macOS host policy rather than complete confidentiality isolation.
There is no cross-platform or unsandboxed fallback. If any definition is trusted and the
sandbox is unavailable, the run fails before compaction or any provider request. During
execution, a pre-hook command/contract failure tightens to denial; a post-hook failure is
recorded without changing the already determined tool settlement. Cancellation is propagated.

Hook stdin transiently contains the raw tool input and, for `PostToolUse`, the bounded tool
result. Only trust programs that may see that data. Raw hook stdin, stdout and merged stderr
are not written to hook events or journals; events retain identities, timing, safe outcome
codes and SHA-256 bindings. The bytes may still exist transiently in process memory, and the
underlying tool retains its independent standard event/persistence contract.

`hook.effect_started` is durably emitted before the trusted command may execute. A crash or
cancellation after that boundary is handled by the same conservative unresolved-effect fence
as other uncertain effects. The Harness does not automatically replay a hook that may already
have acted.

### Exact-trust local MCP stdio tools

`.agent-harness/mcp.json` is untrusted repository content. Merely opening a workspace or
running `harness mcp` does not start a server. Each configured server must first receive an
exact `trust` or `disable` decision in the 0600 state directory outside the repository. A
trusted server must then be started explicitly by `harness mcp refresh SERVER_ID` to negotiate
protocol `2025-06-18` and freeze a bounded tools catalog. New runs fail closed when a proposal
is unresolved or a trusted catalog is missing/stale; an active run never adopts a changed
catalog dynamically. Before every call the client revalidates the direct executable, performs
a fresh handshake/list, and rejects a changed or list-changed catalog.

The definition digest binds exact config bytes, server ID, transport, the resolved direct
executable's bytes and file identity, argv, cwd, allowed environment-variable names,
network/fork flags, timeouts and sandbox-policy version. This is **not transitive dependency
integrity** or a code signature. In particular, it does not bind a script named only in an
interpreter argument, imported packages, shared libraries, runtime configuration, environment
variable values or files the server reads later. Rechecking the direct executable cannot
detect those changes. Trust therefore means the operator reviewed and accepts the entire
server/dependency chain; the Harness only enforces the narrower digest it reports.

To close the path-replacement window between verification and `exec`, each user-owned direct
executable is copied byte-for-byte into a mode-0500 file inside that connection's mode-0700
private runtime and the copy is executed. Its `argv[0]` (and a script's usual `__file__`) thus
points at the transient runtime copy, so servers that locate resources relative to their
executable directory must account for that behavior. A macOS system executable is left at its
canonical platform path only when the executable and every ancestor are root-owned and both
mode and ACL-aware current-user checks show no write authority; copied platform binaries may
otherwise be non-executable on macOS.

MCP processes have a read-only workspace, protected `.git`/`.private`/`.agent-harness` reads
denied, writes limited to a private runtime HOME/TMP plus `/dev`, and known external user-data
and credential-service denials. Network and process fork are denied by default; setting
`network_access` or `allow_process_fork` grants that authority and changes the exact definition
digest. Reserved provider credentials and `HARNESS_*` names cannot be selected through
`pass_env`. Harness/sandbox environment controls and common runtime code-loading controls such
as `DYLD_*`, `LD_*`, `PYTHONPATH` and `NODE_OPTIONS` are also rejected. Other explicitly named
values are disclosed to the local server and are not digest bound.

This MCP boundary is an allow-default macOS Seatbelt host policy, **not a container, VM,
portable sandbox or complete host-read/IPC/confidentiality boundary**. There is no unsandboxed
or unsupported-host fallback. When fork is disabled the profile denies child creation. When
fork is explicitly enabled, descendants ordinarily inherit the Seatbelt profile, but
`setsid`, double-forking or daemonization can escape the Harness process group and survive its
best-effort timeout/cancellation cleanup. Such a descendant can continue exercising whatever
filesystem, IPC and network authority the exact server definition granted; process-group
reaping is not containment.

Frozen MCP tools are registered only in `full-access` with the separate `mcp.external`
permission and external-service/remote-consent scopes. Every call is high-risk, serial,
`never` replay and crosses a durable effect boundary before the server starts. It requires a
fresh once-only approval; existing or newly requested session/workspace persistent allow
rules cannot bypass that challenge. A crash or cancellation after the boundary is fenced for
manual inspection rather than replayed.

The protocol implementation is a strict bounded client subset pinned to `2025-06-18`:
initialize/initialized, paginated `tools/list`, `tools/call`, cancellation, server ping
responses and tools-list-changed invalidation. It does not implement HTTP, OAuth, resources,
prompts, sampling, elicitation, tasks, input-required/task results, active-run dynamic
catalogs, full JSON Schema, binary rendering or MCP server mode. Non-`ping` server requests
receive method-not-supported. Unsupported schema keywords/tools fail closed or are excluded
from the explicit frozen catalog rather than weakening local validation.

Tool arguments and normalized text/structured results follow the ordinary owner-only
journal/checkpoint persistence contract and results may later be sent to the model provider.
Raw server stderr is drained and hashed transiently but is not persisted; only byte count,
truncation state and SHA-256 metadata can be returned to the local CLI. Non-text content is
not rendered and is reduced to content-free type/length/MIME/digest metadata before normal
tool persistence.

### Foreground subagents and Git worktrees

`agent.delegate` is a deliberately narrow high-risk, `never`-replay operation. It accepts one
batch of 1–4 bounded tasks, requires a fresh once-only parent approval, crosses the durable
effect boundary before allocating children, and cannot create a session/workspace persistent
allow. A headless run without an approval broker hands off before creating a worktree. Parent
cancellation propagates to every child and the foreground call joins all started children
before settlement; background, detach, resume and steer are not implemented.

The requested child authority is only `read-only` or `workspace-write`. It must be no broader
than the parent's live permission, and a `full-access` parent cannot give a child host-command
or MCP authority. Child runners do not expose/authorize project command hooks, MCP, host
commands or nested delegation, and persistent approvals are disabled. After the parent approves the exact batch, the child approval broker
can allow only the child's centrally authorized `workspace.patch` and sandboxed
`process.exec`; every other approval request is denied. This once-only delegated authority is
bound into the child run policy and cannot survive resume as a broader policy.

Each child gets a separate Git worktree created from the exact committed local `HEAD`. The
source repository must be Git-backed, non-unborn and clean including non-ignored untracked
files; dirty parent state is rejected rather than copied, and ignored local files are not
copied. Git is invoked without a shell through a trusted
absolute executable and a scrubbed environment. Project hooks, config includes, executable
filters, fsmonitor and external diff are disabled/rejected. Repository-common Git mutations
are serialized by a cross-process lock keyed to the canonical common Git directory, while
subagent concurrency and total-spawn budgets are only in-process controls.

Worktree isolation prevents concurrent child file edits from directly sharing a checkout. It
is **not** a container, VM, separate OS principal, provider credential boundary, process-heap
boundary, complete host-read boundary or independent network sandbox. A child uses the same
configured provider/model and the ordinary workspace tool/Seatbelt boundary. The worktree
also shares the repository's Git object database and refs; the Harness therefore treats its
branch and administrative mapping as control-plane state, not as an adversarial Git tenancy
boundary.

Automatic removal is permitted only when the record is active, the `.git`/administrative
mapping and exact lock reason still match, Git status is clean, a no-follow bounded content
manifest equals the baseline, and the opaque branch still points to the baseline commit.
Removal uses normal unlock/remove plus compare-and-swap ref deletion. The manager exposes no
force, reset, clean or prune operation. Changed, structurally suspicious or uncertain artifacts
that still exist are preserved for operator inspection. If checkout removal succeeds but the
subsequent ref compare-and-swap loses a race, only the branch and a `ref_preserved` record remain;
the checkout is already gone. No child change is merged, applied, committed, pushed or opened as
a PR automatically.

Default `harness agents` and TUI `/agents` views expose only bounded lifecycle metadata and
opaque artifact identities. An absolute local worktree path is disclosed only by the explicit
`harness agents WORKTREE_ID --path` form. Child summaries are untrusted model/tool data; the
parent provider prompt explicitly forbids treating embedded instructions as authority. The
public lifecycle projection includes only typed batch `count`/`depth` and child opaque
`agent_id`/`depth`/`ordinal`/`status` metadata, never prompts, child output, raw exceptions,
worktree paths or hidden reasoning.

## Provider credentials

Use `DEEPSEEK_API_KEY` or `HARNESS_DEEPSEEK_API_KEY_FILE`. Key files must be private regular
files. A convenience symlink is accepted only when both its directory and final target
directory are owner-only; the final target is opened with `O_NOFOLLOW` and checked for
inode, device and size stability. Credentials are not emitted in status, events, errors or
session files.

The model is remote. Running the TUI or headless command sends the current conversation and
bounded tool observations to the configured provider.

## Local persistence

Sessions and journals live below `${AGENT_HARNESS_HOME:-~/.agent-harness}` in a workspace
hash namespace. Managed directories are mode 0700 and files mode 0600; symlinked managed
paths are rejected. Journals are append-only hash chains and checkpoints use fsync plus
atomic replace.

Normal sessions can be resumed and forked. A fork is a logical copy of transcript and
session metadata; it does not create a fresh process, container, worktree, environment or
heap, and therefore does not isolate credentials or other in-memory secrets.

Subagent child sessions use the same private workspace-hash session-store contract, while
worktree lifecycle records live below that workspace's private `worktrees/` state. The
separate worktree root must be owner-controlled and non-overlapping with the source
repository/state directory. Changed or uncertain artifacts are intentionally retained until
the operator inspects them; there is no automatic retention deadline or destructive cleanup
command in this version.

If any session, including an archived one, has a run still marked active or requiring
effect reconciliation, a workspace-wide fence blocks every new run in that workspace.
Use `harness effects` or `/effects` to list the records, then inspect the durable journal,
actual workspace and any external system. `harness reconcile RUN_ID` and
`/reconcile RUN_ID` merely record the operator's acknowledgement and clear the fence by
settling the record as a handoff. They do not prove whether an effect occurred, undo it,
or make repeating it safe. Automated cross-process continuation and automatic effect
reconciliation are not exposed by the CLI/TUI.

## Reporting

Report vulnerabilities through a private repository security advisory. Never include API
keys, private prompts, command output or session files in a public issue.
