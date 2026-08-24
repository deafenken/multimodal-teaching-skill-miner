# Privacy and local data

Agent Harness stores only the data needed for local multi-turn operation and recovery.

| Data | Location | Remote transmission |
|---|---|---|
| Session messages | `~/.agent-harness/workspaces/<hash>/sessions/` | Sent to the selected model provider during a turn |
| Python SDK values | Embedding Python process memory plus the ordinary session/journal/attachment locations below | No additional transport; prompts, attachments and tool observations follow the same selected-provider path as `AgentRunner` |
| TypeScript SDK subprocess data | Node process memory, transient child stdin/stdout pipes and CLI-owned private state | Prompt is sent locally to `harness` over stdin; validated JSONL/model/tool output returns to the Node caller, while the CLI uses the ordinary selected-provider path |
| Tool observations | Private per-run journal | Bounded observations may be sent on the next model step |
| Event/checkpoint state | `~/.agent-harness/workspaces/<hash>/runs/` | No, except fields included in the next provider request |
| API credentials | Environment or configured private file | Authorization header only; never session/journal content |
| Workspace identity | Local real path, hashed for the state namespace | The provider context receives only the literal workspace label `.`; the Harness does not send the absolute workspace path as that field |
| Workspace files | Original workspace | Built-in file tools and sandboxed `process.exec` can expose bounded workspace content; `process.exec_host` can read other current-user-accessible files and its bounded output may be sent |
| Attachment descriptors | Session messages below the private workspace state | Opaque ID, digest and MIME metadata may frame provider input; the sanitized basename remains local display metadata, and the source path is never retained or sent by the attachment subsystem |
| Immutable attachment blobs | `~/.agent-harness/workspaces/<hash>/attachments/` | Strict UTF-8 text is sent when attached to a supported turn; PNG/JPEG bytes are inlined only for an explicitly selected supported vision model; current DeepSeek PDF input is rejected before a run/network request |
| Hook definitions | `.agent-harness/hooks.json` and `.agent-harness/hooks/` in the workspace | Not sent by the hook subsystem; only exact digests and content-free metadata enter run policy/events |
| Hook trust decisions | Private workspace state below `${AGENT_HARNESS_HOME:-~/.agent-harness}` | Never; records contain hook ID, exact definition SHA-256 and trusted/disabled action |
| Raw hook stdin/stdout/stderr | Transient Harness/child-process memory and anonymous temporary input | Not sent by the hook subsystem and not persisted in hook events; the underlying tool input/result keeps its separate normal contract |
| MCP definitions and frozen catalog | `.agent-harness/mcp.json` plus 0600 trust/catalog files below the private workspace state | Definitions/catalog metadata are local; accepted tool schemas and digests may enter the provider-visible tool list/run policy, but server instructions and stderr text do not |
| MCP tool input and normalized result | Owner-only per-run journal/checkpoint; normalized result also becomes a bounded tool observation | Input is sent to the explicitly trusted local server; the normalized result may be sent to the selected model provider on the next step |
| Raw MCP stderr | Transient local drain only | Never persisted or sent; the CLI can receive only byte count, truncation state and SHA-256 metadata |
| Subagent task and child transcript | Parent run checkpoint plus the child workspace-hash session namespace | The bounded delegated prompt and child conversation are sent to the selected provider; the child's final bounded summary may be sent again as a parent tool observation |
| Subagent lifecycle metadata | Parent journal/public trace and private child/worktree records | Only typed count/depth/ordinal/status and opaque IDs enter public lifecycle views; no prompt, child output, hidden reasoning or local path is sent as lifecycle metadata |
| Isolated worktree content | Owner-controlled worktree root outside the repository/state tree | Child workspace tools can expose bounded file content to the provider under the same rules as a normal turn; changed or uncertain worktrees may remain locally after the run |
| Worktree control records | Private `worktrees/` directory below the parent workspace state | Never sent by the worktree subsystem; records locally contain absolute repository/worktree/Git paths, baseline identifiers and digests, while default CLI/TUI views redact paths |

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

## SDK data

The Python SDK runs `AgentRunner` inside the embedding process. It adds no separate wire
protocol or SDK-specific persistent database: session messages, journals, attachment blobs and
worktree records use the same private paths documented above. Prompts, imported attachment
content and bounded tool observations are sent to the selected model under the same turn rules.
Typed results and event objects remain in application memory for as long as the caller retains
them. An `on_event` callback receives public event payloads, which can include assistant text or
ordinary tool output. An explicitly supplied approval broker also receives a transient bounded
human preview in addition to digest-bound request metadata. Those callbacks and brokers are
application code; they may log, persist or transmit what they receive outside Harness control.

Python streaming uses in-memory queues and non-daemon worker threads. Closing or cancelling a
stream joins the worker but does not erase already delivered events, remove referenced
attachments, roll back effects or clear durable recovery evidence. The async facade uses worker
threads around the same in-process runner rather than a remote service.

The TypeScript SDK starts the configured `harness` executable with direct argv and
`shell: false`. Prompt text is written to child stdin and is not placed in argv. Attachment
source paths are normalized to absolute paths and passed as repeated `--attach` arguments, so
the path spellings can be visible transiently to same-user process inspection even though the
attachment subsystem does not retain them in descriptors/events after import. The child gets a
snapshot of the Node process environment plus explicit `env` overrides; configured credentials
therefore cross into the local CLI process.

Child stdout carries the canonical JSONL event stream and final exec-result record. The SDK
parses it under line/record/total/response bounds, exposes immutable events to `onEvent` and the
async iterator, and constructs `finalResponse` from non-internal message deltas. Protocol
validation is not content redaction: model text and normal tool output may include source,
attachment content, paths or secrets the run was authorized to read. The SDK itself does not
persist this stream, but the embedding application may do so. Child stderr is drained and
discarded; only generic errors, exit code and signal are exposed. Cancellation does not wipe
Node/child memory or CLI-owned durable state, and a run interrupted after an effect can leave the
ordinary unresolved-effect fence.

## Attachment data

`/attach PATH` and `harness exec --attach PATH` import an immutable local snapshot before a
turn. The original pathname is used transiently for a no-follow read and is not written to the
session, journal, event stream or attachment descriptor. A sanitized basename is stored as
local UI metadata and can itself reveal information to anyone who can read the private session,
but the current DeepSeek request expansion uses only opaque ID, SHA-256 and MIME framing. Blob
files are mode 0600 under a mode-0700 workspace attachment directory. Processes running as the
same OS user can still access that local state.

The descriptor and session manifest contain kind, MIME type, size, SHA-256, conservative token
estimate and image dimensions, not the body or base64. Provider traces and attachment-specific
public JSONL metadata do not copy body/base64/source path. Assistant or tool output can still
quote attachment content and then follows the ordinary transcript/JSONL contract. This does
not mean attachment content stays local: on a
supported turn, strict UTF-8 text is expanded into provider input, and PNG/JPEG bytes are
base64-inlined into the request only when the operator explicitly configured
`deepseek-v4-flash-vision-exp`. The Harness does not switch models or upload through DeepSeek
Files API automatically. PDF snapshots are represented locally but the current DeepSeek adapter
rejects them before run creation and does not parse, extract or transmit their content.

Each turn can import at most 8 attachments/24 MiB. The DeepSeek active request additionally
caps attachment history plus the prospective turn at 16 items/24 MiB. Attachments are frozen
with their pending prompt, so queued TUI follow-ups cannot silently acquire another turn's
files. A session fork copies descriptors and shares the workspace attachment blobs; it does not
create a separate copy or confidentiality boundary. Referenced blobs have no automatic expiry
or whole-store garbage collector in this version, so they can outlive an active session unless
the operator removes the private Harness state outside a running workflow.

All attachment content, including text visible in an image, is untrusted user data. Embedded
instructions do not become project/system instructions or authorization. They can still affect
the remote model's response and tool requests, which remain subject to the normal local policy
and approvals.

## Subagent and worktree data

One approved `agent.delegate` call stores its exact task array in the ordinary private parent
run/checkpoint contract. Each child then creates a normal local session under the state
namespace derived from that child's worktree path. Removing a pristine worktree does not
delete that child session record; this preserves audit/recovery evidence and means subagent
transcripts can outlive the checkout. There is no automatic child-session retention policy in
this version.

Child prompts, workspace reads, command output and final text use the same remote-model data
path as a normal Harness turn. A final bounded child summary becomes an untrusted parent tool
observation and may therefore be transmitted to the provider once more on the parent's next
model step. Hidden planner/reasoning content is never returned as the child result. The
child's local absolute workspace label remains `.` in provider context, but file contents or a
tool/command's textual output can still mention paths and sensitive data.

Worktree lifecycle records are local control-plane files and include the canonical source
repository, common Git directory, isolated path, opaque branch, baseline commit/manifest,
lock reason and phase. They are mode-0600 files below an owner-only state directory. The
worktree root is a separate owner-controlled directory; checked-out files retain normal Git
file modes but are enclosed by that private root. Default `harness agents` and TUI `/agents`
views project only opaque identity and bounded lifecycle data. The operator must provide one
exact artifact ID and `--path` to print its absolute local path.

Pristine worktrees and their lifecycle records are removed only after exact status/manifest/
ref checks. Changed, structurally suspicious or uncertain artifacts that still exist are kept
without an automatic expiry so an operator can inspect them. If checkout removal succeeds but
ref compare-and-swap deletion loses a race, the branch and a `ref_preserved` record remain while
the checkout does not. Harness never auto-merges or uploads those changes. Worktree separation
prevents direct checkout collisions; it does not separate process memory, provider credentials,
OS identity, shared Git objects/refs or every host read/IPC channel.

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
