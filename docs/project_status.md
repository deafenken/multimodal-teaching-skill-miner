# Project status

Status: **engineering preview**
Version: **2.5.0**

## Implemented

- independent `agent_harness` package with domain-neutral schemas;
- bounded provider/tool loop with cancellation, retries and explicit handoff;
- typed, durable-first event stream and tamper-evident per-run journal;
- private workspace-scoped sessions with resume, logical transcript fork and archive;
- read-only downgrade on resume, workspace-wide unresolved-run fencing and manual
  `effects`/`reconcile` acknowledgement;
- DeepSeek adapter with central tool calls and native final-answer streaming;
- list/read/search tools, macOS Seatbelt-enforced patch/command writers, and a separately
  named host command behind explicit permission profiles;
- per-call approval before sensitive effects, fail-closed headless behavior and private
  exact session/workspace rules;
- secure `AGENTS.override.md`/`AGENTS.md` snapshots, provider wiring and context diagnostics;
- exact-digest-trusted synchronous command hooks for `PreToolUse`, `PostToolUse` and
  `PostToolUseFailure`; project hooks are monotonic, post hooks are observe-only, execution
  uses a dedicated read-only/no-network macOS sandbox with no unsupported-host fallback,
  and events omit raw hook I/O;
- exact-digest trusted local stdio MCP tools client subset pinned to protocol `2025-06-18`:
  explicit trust/disable and catalog refresh, frozen/run-bound tool surface, live-catalog
  revalidation, strict bounded JSON-RPC/schema/result handling, `full-access` high-risk
  once-only approval, never replay, read-only macOS Seatbelt execution and digest-only stderr
  diagnostics;
- stable message identities, append-only transcripts, provider-generated summary lineage,
  active-context projection, manual `/compact` and bounded 80%→60% automatic compaction;
- one bounded foreground `agent.delegate` batch: 1–4 concurrent child sessions, parent
  cancellation and wait-all settlement, dynamic read-only/workspace-write child cap, no child
  host/MCP/hooks/nesting/persistent approval, and typed content-free lifecycle status;
- exact clean-HEAD Git worktree isolation with opaque durable records, cross-process
  common-directory mutation locking, direct trusted Git execution, pristine-only non-force
  cleanup, conservative artifact preservation (with an explicit post-remove ref-race exception),
  and path-redacted CLI/TUI artifact inspection;
- curses TUI and headless text/JSONL commands;
- macOS double-click launcher;
- unit, contract, security, persistence and reducer tests.

## Deliberately not claimed

- feature parity with Claude Code or Codex;
- a portable cross-platform or container-grade command sandbox (the enforced backend is
  currently macOS Seatbelt only);
- complete host-read, IPC or credential-service isolation: the Seatbelt profile is an
  allow-default deny overlay covering named boundaries, not a container or VM;
- cleanup or containment of an unsandboxed `process.exec_host` command that daemonizes
  outside its initial process group;
- confidentiality isolation between a session and its fork;
- MCP HTTP transport, OAuth, resources, prompts, sampling, elicitation, tasks, input-required/
  task results, active-run dynamic catalogs, full JSON Schema, binary rendering or MCP server
  mode; the implemented surface is only the exact-trusted local stdio tools client subset;
- transitive integrity for MCP dependencies: the exact digest binds the direct executable and
  launch definition, not interpreter-argument scripts, libraries, packages, runtime files or
  environment values;
- containment of an MCP server allowed to fork and then daemonize outside the Harness process
  group; descendants retain the Seatbelt/network authority explicitly granted to that server;
- `CLAUDE.md` compatibility;
- complete Claude Code/Codex subagent parity: no background/resume/steer or agent-thread
  switching, custom agent definitions/model routing, nested delegation, team coordination,
  automatic merge/apply/commit/push/PR, or cross-process global fan-out budget;
- worktree isolation as a process, credential, OS-principal, host-read, IPC or network security
  boundary; it isolates checkout writes and shares the Git object database/refs;
- full Claude Code/Codex hooks parity: only the three synchronous tool lifecycle events are
  implemented, with no async hooks, input rewriting or other run/session/model events;
- a provider-neutral compaction implementation beyond the current DeepSeek adapter;
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
