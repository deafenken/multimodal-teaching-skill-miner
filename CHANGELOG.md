# Changelog

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
