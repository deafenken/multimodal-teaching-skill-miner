# Privacy and local data

Agent Harness stores only the data needed for local multi-turn operation and recovery.

| Data | Location | Remote transmission |
|---|---|---|
| Session messages | `~/.agent-harness/workspaces/<hash>/sessions/` | Sent to the selected model provider during a turn |
| Tool observations | Private per-run journal | Bounded observations may be sent on the next model step |
| Event/checkpoint state | `~/.agent-harness/workspaces/<hash>/runs/` | No, except fields included in the next provider request |
| API credentials | Environment or configured private file | Authorization header only; never session/journal content |
| Workspace identity | Local real path, hashed for the state namespace | The provider context receives only the literal workspace label `.`; the Harness does not send the absolute workspace path as that field |
| Workspace files | Original workspace | Built-in file tools and sandboxed `process.exec` can expose bounded workspace content; `process.exec_host` can read other current-user-accessible files and its bounded output may be sent |
| Hook definitions | `.agent-harness/hooks.json` and `.agent-harness/hooks/` in the workspace | Not sent by the hook subsystem; only exact digests and content-free metadata enter run policy/events |
| Hook trust decisions | Private workspace state below `${AGENT_HARNESS_HOME:-~/.agent-harness}` | Never; records contain hook ID, exact definition SHA-256 and trusted/disabled action |
| Raw hook stdin/stdout/stderr | Transient Harness/child-process memory and anonymous temporary input | Not sent by the hook subsystem and not persisted in hook events; the underlying tool input/result keeps its separate normal contract |

The TUI never renders hidden reasoning text. It records only a character count for
provider reasoning deltas. Planner JSON is also internal and never becomes an assistant
message.

`harness sessions`, `resume`, `fork` and `archive` operate only within the real-path hash of
the selected workspace. There is no browser cache or domain-specific user profile.

A session fork is a logical transcript copy, not a confidentiality boundary. It does not
isolate the parent and fork into separate processes, environments, heaps, provider clients
or credential stores. Likewise, `data_scope` labels describe Harness authorization; they
do not stop an authorized handler or `full-access` shell from reading data available to
its operating-system process.

The minimal command environment does not make command output safe. Sandboxed `process.exec`
can read permitted workspace content, while `process.exec_host` in `full-access` can
deliberately read a secret from the workspace or host and print it; the bounded observation
can then be transmitted to the provider. Review the workspace and command before granting
these modes.

## Project hook data

Repository hook files are not executed merely because they exist. The operator must bind a
trust decision to the exact definition digest in private state outside the workspace; changes
invalidate that decision. TUI `/hooks` and `harness hooks` expose only IDs, matchers, counts,
paths, timing limits, digests and trust status. Trust/disable/revoke is an explicit CLI action.

A trusted `PreToolUse` hook receives the raw bounded tool input on stdin. `PostToolUse` also
receives the bounded tool result, and `PostToolUseFailure` receives a safe error code. This
disclosure is local to the trusted command, but it can include source code, command arguments
or other sensitive workspace data. Exact-digest trust therefore authorizes that reviewed
executable to see those fields; it is not only permission to return a policy decision.

Hook stdout and merged stderr are parsed transiently as a bounded decision. Hook lifecycle
events and journals contain definition/input/output SHA-256 values plus content-free identity,
duration, action and error codes; they do not contain raw hook stdin, stdout or stderr. This
redaction does not remove the underlying tool's own journal fields—for example, a normal
bounded `tool.completed` result may still be retained and later sent to the provider.

Trusted hooks use a dedicated read-only/no-network macOS Seatbelt profile and cannot fall
back to unsandboxed execution on unsupported hosts. It also denies hook reads of the
workspace's `.git`, `.private` and `.agent-harness` roots. That policy reduces exfiltration
routes but is an allow-default host policy, not a container or complete protection against
local side channels. Input, output and snapshotted executable bytes also exist transiently
in the Harness and child-process memory before cleanup.

Private data created by pre-2.0 applications is not migrated, deleted or
uploaded by Agent Harness. It remains legacy operator-owned data outside the new session
namespace.
