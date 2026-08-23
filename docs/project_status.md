# Project status

Status: **engineering preview**
Version: **2.0.0**

## Implemented

- independent `agent_harness` package with domain-neutral schemas;
- bounded provider/tool loop with cancellation, retries and explicit handoff;
- typed, durable-first event stream and tamper-evident per-run journal;
- private workspace-scoped sessions with resume, logical transcript fork and archive;
- read-only downgrade on resume, workspace-wide unresolved-run fencing and manual
  `effects`/`reconcile` acknowledgement;
- DeepSeek adapter with central tool calls and native final-answer streaming;
- list/read/search/patch/process tools behind explicit permission profiles;
- curses TUI and headless text/JSONL commands;
- macOS double-click launcher;
- unit, contract, security, persistence and reducer tests.

## Deliberately not claimed

- feature parity with Claude Code or Codex;
- a kernel/container-grade shell sandbox;
- containment of a `full-access` command that daemonizes outside its initial process group;
- confidentiality isolation between a session and its fork;
- per-call approval rules;
- MCP, hooks, instruction discovery, context compaction or subagents;
- exact cost estimates when the provider does not supply a stable price contract;
- production deployment or multi-user tenancy;
- automatic cross-process continuation of an in-flight run;
- automatic verification, rollback or safe replay of an unresolved effect.

The current comparison and implementation order are documented in
`docs/harness_parity_matrix.md`. The previous product-domain application, its public
fixtures, schemas, Console/API and active entry points have been removed. Private legacy
runtime data was not deleted and is outside the new session namespace.

## Verification

```bash
ruff check agent_harness tests_harness
pytest -q
python -m build --outdir "$(mktemp -d)"
python -m agent_harness --cwd . status
zsh -n '打开Agent Harness.command'
```

The PTY TUI launch, idle `Ctrl+C` and `/quit` checks are local interactive smoke tests;
the current CI job does not claim to execute them.
