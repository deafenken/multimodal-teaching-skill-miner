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
| MCP definitions and frozen catalog | `.agent-harness/mcp.json` plus 0600 trust/catalog files below the private workspace state | Definitions/catalog metadata are local; accepted tool schemas and digests may enter the provider-visible tool list/run policy, but server instructions and stderr text do not |
| MCP tool input and normalized result | Owner-only per-run journal/checkpoint; normalized result also becomes a bounded tool observation | Input is sent to the explicitly trusted local server; the normalized result may be sent to the selected model provider on the next step |
| Raw MCP stderr | Transient local drain only | Never persisted or sent; the CLI can receive only byte count, truncation state and SHA-256 metadata |

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

## Local MCP data

Repository MCP definitions are inert until the operator binds trust to the current exact
definition digest and explicitly refreshes the server catalog. `harness mcp` and TUI `/mcp`
show IDs, exact argv/cwd, selected environment-variable **names**, authority flags, digests,
counts and trust/catalog status. Environment-variable values are neither shown nor stored in
the trust/catalog record, but a value named by `pass_env` is copied transiently into the
trusted local server's environment when it starts.

An MCP call sends the raw bounded tool arguments over local stdio to that trusted server.
The ordinary tool recovery contract then retains the arguments and normalized result in the
workspace's owner-only journal/checkpoint state (0700 directories and 0600 managed files).
Text and object-shaped structured results may therefore be present locally and may become the
next model-step observation. If the exact server definition enables network, the local server
can independently transmit data under its granted Seatbelt/network authority; this is
separate from transmission to the configured model provider.

Raw MCP stdout frames exist transiently while the client validates and normalizes them. Text
and structured result fields enter the normal tool result; unsupported non-text/binary items
are not rendered and are replaced with type, encoded/decoded length where available, MIME and
SHA-256 metadata before persistence. This does not make the original bytes harmless—they were
still received into local process memory before normalization.

Server stderr is continuously drained to avoid blocking, hashed and counted. Its original
bytes are not written to a journal, checkpoint, session, trust file or frozen catalog and are
not sent to the provider. A local `harness mcp --json refresh SERVER_ID` response can include only
`byte_length`, `truncated` and `sha256`; no stderr text is retained. Server `instructions` and
`serverInfo` are likewise represented in the catalog by SHA-256 only, not stored as raw text.

Exact-digest trust is not a confidentiality boundary or dependency attestation. The digest
does not cover transitive packages/libraries, scripts named only as interpreter arguments,
runtime files or environment values. A trusted server can read permitted workspace data, and
explicit `network_access` or `allow_process_fork` expands its local authority. Seatbelt remains
an allow-default macOS host policy rather than a container; a daemonized child may escape
best-effort process-group cleanup while retaining its granted policy.

Private data created by pre-2.0 applications is not migrated, deleted or
uploaded by Agent Harness. It remains legacy operator-owned data outside the new session
namespace.
