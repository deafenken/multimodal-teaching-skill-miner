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
