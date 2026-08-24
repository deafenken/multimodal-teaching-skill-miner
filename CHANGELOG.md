# Changelog

## 2.5.0 — Isolated foreground subagents

- Added one provider-visible `agent.delegate` capability for a high-risk, never-replay,
  foreground wait-all batch of 1–4 child tasks. Child work is truly concurrent, results remain
  in input order, parent cancellation propagates, reconciliation cancels siblings, and every
  started child is joined before the parent tool settles.
- Added atomic in-process batch/global/parent/depth/total budgets. The default depth is one and
  total/concurrent child limit is four. This does not claim a cross-process global scheduler.
- Added child sessions with bounded prompt/output/path-free lineage metadata, untrusted-result
  prompt-injection boundaries and no hidden reasoning channel. Children receive at most
  `workspace-write` and have no host command, MCP, executable project hooks, nested delegation
  or persistent approvals.
- Added a run-level persistent-approval cap bound into the execution-policy/checkpoint digest.
  The exact parent batch needs one fresh approval; inside that authority, only centrally
  authorized child patch/sandbox-command requests can receive an automatic once-only allow.
- Added fail-closed Git worktree management from exact clean committed `HEAD`: opaque
  branch/path IDs, cross-process common-Git-directory lock, direct absolute Git argv, scrubbed
  environment/config, provisional durable records, linked-worktree/lock verification and a
  bounded no-follow baseline manifest.
- Added pristine-only normal worktree removal plus compare-and-swap ref deletion. Changed,
  committed, structurally suspicious or uncertain artifacts that still exist are preserved. A
  post-remove ref race retains the branch and record after the checkout is gone; there is no
  force/reset/clean/prune or automatic merge/apply/commit/push/PR behavior.
- Added `harness agents [WORKTREE_ID] [--path] [--json]`, TUI `/agents`, active-agent status and
  typed public lifecycle projection. Default views omit local paths; one explicit ID plus
  `--path` is required to reveal a retained worktree path.
- Explicitly do not claim background/resumable/steerable agent threads, custom agents/model
  routing, nested delegation, team coordination, automatic integration or full Claude
  Code/Codex subagent parity.

## 2.4.0 — Exact-trust local MCP stdio tools

- Added an intentionally narrow MCP tools client subset pinned to protocol version
  `2025-06-18`: local stdio transport, initialization, paginated `tools/list`,
  `tools/call`, cancellation, server ping responses and stale-catalog notification
  handling.
- Added secure `.agent-harness/mcp.json` discovery. A server definition must be accepted
  with its full exact digest, then have its bounded tool catalog explicitly refreshed and
  frozen in private state before it contributes tools to a run. The live catalog is checked
  again before each call; changes require another explicit refresh rather than changing an
  active run dynamically.
- Registered MCP tools only in `full-access`, as high-risk, external-service, never-replay
  operations that require a fresh once-only approval. Persistent session/workspace allow
  rules cannot bypass an MCP approval.
- Added strict, bounded newline-delimited JSON-RPC framing, a deliberately limited
  object-root JSON Schema validator, bounded text/structured results and hash/length-only
  projections for non-text content. Unsupported tool schemas are rejected from the frozen
  catalog.
- Added a dedicated read-only macOS Seatbelt launch with network and process fork denied by
  default and available only through exact-digest-bound configuration flags. There is no
  unsandboxed or unsupported-host fallback. Exact trust binds the direct executable and
  launch definition; it does not attest transitive libraries, interpreter arguments,
  packages, runtime configuration or environment values.
- Closed the verified-path-to-exec race for user-owned direct executables by running an exact
  mode-0500 copy from each connection's mode-0700 private runtime. This intentionally changes
  `argv[0]`/script `__file__`; ACL-nonwritable root-anchored macOS system executables retain
  their canonical platform path. Common loader/runtime code-loading variables are rejected
  from `pass_env` even when explicitly named.
- Added `harness mcp` inspection plus digest-bound `trust`, `disable`, `refresh` and
  `revoke` operations; TUI `/mcp` remains content-free and read-only.
- MCP inputs and normalized results follow the existing owner-only journal/checkpoint
  contract. Raw server stderr is drained transiently and is never persisted; only bounded
  byte-count, truncation and SHA-256 metadata can be returned to the local CLI.
- Explicitly do not claim HTTP transport, OAuth, resources, prompts, sampling, elicitation,
  tasks, active-run dynamic catalogs, full JSON Schema, binary rendering or MCP server mode.

## 2.3.0 — Trusted synchronous command hooks

- Added a deliberately limited synchronous hook surface for `PreToolUse`, `PostToolUse`
  and `PostToolUseFailure`; this is not full Claude Code or Codex hooks parity.
- Added secure `.agent-harness/hooks.json` discovery and exact definition-digest trust or
  disable decisions stored in the private workspace state. Changed definitions become
  untrusted and block runs until the operator makes a new exact decision.
- Made project hooks monotonic: pre-tool hooks may preserve the central decision, require
  approval or deny, while post-tool hooks are observe-only. A hook cannot grant authority,
  replace arguments or weaken permissions, scopes, approvals or the tool sandbox. Hook
  approval challenges are once-only and cannot be bypassed by or saved as a persistent rule.
- Run trusted entrypoint snapshots in a dedicated read-only, no-network macOS Seatbelt
  profile with no unsafe fallback on unsupported hosts.
- Added content-free hook lifecycle evidence. Journals contain identities, timing, outcome
  codes and SHA-256 bindings, not raw hook stdin, stdout or stderr.
- Added `harness hooks` inspection plus explicit digest-bound `trust`, `disable` and `revoke`
  operations; TUI `/hooks` is intentionally read-only.

## 2.2.0 — Traceable context compaction

- Added stable message IDs and content digests while retaining backwards-compatible reads of
  earlier Harness sessions.
- Made persisted transcript and compaction history append-only; session begin/finish now bind
  user/run and assistant/terminal state in atomic store updates.
- Added incremental provider summaries with source-prefix digests, parent lineage, bounded
  chunks and assistant-boundary cut points. Original messages are never replaced or deleted.
- Added summary+suffix active-context projection shared by planner and answer. Summaries remain
  user data and cannot grant tools, permissions, scopes or approval bypasses.
- Added `harness context`, `harness compact`, TUI `/context` and background `/compact`.
- Added automatic compaction at 80% of the conservatively estimated input budget, targeting
  60% and retaining the most recent six original messages.

## 2.1.0 — Workspace policy, approvals and project instructions

- Added a macOS Seatbelt workspace command boundary, a distinct full-access host command and
  fail-closed workspace writer registration on unsupported platforms.
- Added digest-bound per-call approvals, headless handoff and private exact session/workspace
  rules, with an interactive TUI approval surface.
- Added secure root-to-active-directory `AGENTS.override.md` / `AGENTS.md` discovery and a
  32 KiB instruction snapshot bound into every run.
- Added OS-sandboxed patch application, outgoing-signal denial and denial of known Keychain
  security-service lookups; documented that the allow-default profile is not a container.

## 2.0.0 — Agent Harness pivot

- Retired the previous domain-specific application, prompts, schemas and product surfaces.
- Added the standalone `agent_harness` package with generic schema identifiers and data
  scopes.
- Added a tool-aware DeepSeek coding adapter. Planner JSON and hidden reasoning remain
  internal; final prose uses the provider's native stream.
- Added workspace-confined list/read/search and patch tools, plus an explicit host-level
  command tool in `full-access`.
- Added real `read-only`, `workspace-write` and `full-access` tool surfaces.
- Added private workspace-scoped sessions with resume, fork and archive, backed by the
  existing durable hash-chain run journals and atomic checkpoints.
- Added a workspace-wide fence for unfinished or uncertain runs and manual
  `effects`/`reconcile` acknowledgement commands; acknowledgement is not automatic effect
  verification, rollback or replay.
- Added a standard-library curses TUI and `harness exec --jsonl` headless mode.
- Added truthful provider cache telemetry: hit/miss values are shown only when the
  provider reports them.
- Replaced the macOS launch entry with `打开Agent Harness.command`.
- Added an official-docs capability matrix against Claude Code and Codex CLI.

Existing pre-2.0 private runtime data is left untouched as legacy data and is not
loaded into the new Harness session namespace.
