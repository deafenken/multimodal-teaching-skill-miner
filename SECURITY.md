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
