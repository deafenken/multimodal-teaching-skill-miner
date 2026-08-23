# Changelog

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
