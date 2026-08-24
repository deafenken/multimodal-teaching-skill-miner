# Pi Agent Harness reference notes

The earlier architecture review used the open-source Pi repository as a design reference
for event-driven agent execution. No Pi runtime code is embedded in this package.

The reusable ideas were:

- model a run as ordered lifecycle events instead of a `busy` flag;
- separate the current run from steering and follow-up input;
- separate durable operation records from the user-facing transcript;
- stream message deltas while treating the final message as authoritative;
- keep tool validation, execution and lifecycle observation in the Harness;
- persist enough identity and policy state to reject unsafe replay after a crash.

Those ideas now appear in `agent_harness/core`, the workspace-scoped session store and
the TUI reducer. Product-domain state and prompts are intentionally outside the core.

Claude Code and Codex CLI are the active product baselines. Their official-document
comparison and the remaining gaps are tracked in `docs/harness_parity_matrix.md`.
