# Privacy and local data

Agent Harness stores only the data needed for local multi-turn operation and recovery.

| Data | Location | Remote transmission |
|---|---|---|
| Session messages | `~/.agent-harness/workspaces/<hash>/sessions/` | Sent to the selected model provider during a turn |
| Tool observations | Private per-run journal | Bounded observations may be sent on the next model step |
| Event/checkpoint state | `~/.agent-harness/workspaces/<hash>/runs/` | No, except fields included in the next provider request |
| API credentials | Environment or configured private file | Authorization header only; never session/journal content |
| Workspace identity | Local real path, hashed for the state namespace | The provider context receives only the literal workspace label `.`; the Harness does not send the absolute workspace path as that field |
| Workspace files | Original workspace | Built-in file tools send selected bounded content; `full-access` commands can read other current-user-accessible files and their bounded output may be sent |

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

The minimal `process.exec` environment does not make command output safe. In
`full-access`, a command can deliberately read a secret from the workspace or host and
print it; the bounded observation can then be transmitted to the provider. Review the
workspace and command before granting this mode.

Private data created by pre-2.0 applications is not migrated, deleted or
uploaded by Agent Harness. It remains legacy operator-owned data outside the new session
namespace.
