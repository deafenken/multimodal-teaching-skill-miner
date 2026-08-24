# Agent Harness parity progress

**Status**: 🔄 In Progress
**Started**: 2026-08-24 02:49 UTC
**Last updated**: 2026-08-24 03:54 UTC
**Current stage**: 6. Run full acceptance, package audit, and GitHub synchronization

## TODO
- [x] 1. Refresh Claude Code and Codex comparison baseline
- [x] 2. Harden sandbox, instructions, approvals, compaction, hooks, and MCP
- [x] 3. Add foreground subagents, worktrees, and TUI visibility
- [x] 4. Complete provider-neutral attachment release
- [x] 5. Add stable Python and TypeScript SDKs
- [ ] 6. Run full acceptance, package audit, and GitHub synchronization

---

## Detailed Log

### [02:49] 4. Provider-neutral attachments
Resumed after context compaction. Implementation and tests pass; validating built artifacts and a clean wheel installation before commit and push.

### [02:53] 4. Provider-neutral attachments complete
Validated wheel and sdist contents, installed the wheel in a fresh virtual environment, and passed 405 tests, Ruff, compileall, launcher syntax, and diff checks. Independent read-only audit found no release blocker.

### [02:54] 5. Stable Python and TypeScript SDKs
Starting a provider-neutral SDK layer that preserves session, streaming, attachment, cancellation, permission, and audit semantics across Python and TypeScript.
Implemented and adversarially audited stable Python and TypeScript SDKs; 439 Python tests plus 26 subtests and 19 Node tests pass, with fresh wheel and npm installs verified.

### [03:54] 6. Run full acceptance, package audit, and GitHub synchronization
Starting...
