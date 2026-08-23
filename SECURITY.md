# Security policy

## Supported version

Security fixes target the current `2.x` Agent Harness line.

## Trust boundaries

- The default permission mode is `read-only`.
- `workspace-write` exposes only validated unified patches inside the selected workspace.
- `full-access` additionally exposes `process.exec`; it must be an explicit user choice.
  The shell starts in the workspace but is not workspace-confined: it can exercise every
  host permission already held by the current OS user. It can read workspace secrets,
  read files outside the workspace, use the network and place those bytes in bounded tool
  output that may be sent to the provider. This is not a kernel/container sandbox.
- The list/read/search/patch handlers reject absolute paths, `..`, symbolic links and
  protected `.git`, `.private` and `.agent-harness` roots. Those checks do not constrain
  `process.exec`.
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
