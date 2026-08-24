# Requirements traceability

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| H-CORE-01 | Bounded model/tool loop | `agent_harness/core/runtime.py` | `tests_harness/test_agent_harness.py` |
| H-EVENT-01 | Continuous typed events and one terminal | `agent_harness/core/events.py` | `tests_harness/test_harness_controller.py`, `test_generic_product.py` |
| H-DUR-01 | Durable-first hash-chain journal and checkpoint | `agent_harness/core/journal.py` | `test_harness_journal.py`, `test_harness_runtime_journal.py` |
| H-PROV-01 | Provider-neutral contracts and tool-aware DeepSeek adapter | `agent_harness/core/providers.py`, `agent_harness/providers/deepseek.py` | `test_harness_provider_registry.py`, `test_generic_product.py` |
| H-SEC-01 | Central schema, permission, data-scope metadata and replay-policy checks; no claim that scopes sandbox handler authority | `agent_harness/core/tools.py` | `test_harness_tool_security.py` |
| H-WS-01 | Workspace-confined readers, Seatbelt-enforced patch/command writers with signal and known credential-IPC denial, and a separately authorized host shell | `agent_harness/toolsets/workspace.py` | `test_generic_product.py` |
| H-APP-01 | Digest-bound per-call approval before sensitive tool start/effect, fail-closed headless handoff and private exact rules | `agent_harness/core/approvals.py`, `core/tools.py`, `session.py`, `tui.py` | `test_approvals.py`, `test_generic_product.py` |
| H-INS-01 | Workspace-scoped, no-follow, bounded project instruction snapshot used by planner and answer | `agent_harness/instructions.py`, `runner.py`, `providers/deepseek.py` | `test_instructions.py`, `test_generic_product.py` |
| H-CTX-01 | Stable message IDs, append-only original transcript, prefix/parent summary lineage, summary+suffix active view, manual and bounded automatic compaction | `agent_harness/context.py`, `session.py`, `runner.py`, `providers/deepseek.py` | `test_context_compaction.py`, `test_generic_product.py` |
| H-HOOK-01 | Exact-digest trust for the synchronous `PreToolUse`/`PostToolUse`/`PostToolUseFailure` command-hook subset; project hooks can only tighten central authority, execute in a dedicated read-only/no-network macOS sandbox with no unsafe fallback, and emit no raw hook I/O | `agent_harness/hooks.py`, `core/hooks.py`, `core/tools.py`, `toolsets/workspace.py`, `session.py`, `runner.py`, `cli.py`, `tui.py` | `test_hooks.py`, `test_harness_runtime_journal.py`, `test_harness_schemas.py`, `test_generic_product.py` |
| H-MCP-01 | Exact-digest trusted local stdio MCP tools client subset pinned to `2025-06-18`; explicit frozen catalog, live-catalog equality, bounded schema/content normalization, full-access high-risk once-only approval, never replay, read-only Seatbelt with digest-bound network/fork flags, and no raw stderr persistence | `agent_harness/mcp.py`, `core/mcp_protocol.py`, `core/tools.py`, `toolsets/workspace.py`, `session.py`, `runner.py`, `cli.py`, `tui.py` | `test_mcp.py`, `test_mcp_protocol_edges.py`, `test_mcp_runner_integration.py`, `test_session_private_paths.py` |
| H-SES-01 | Private workspace-scoped sessions, atomic user/run and assistant/terminal commits, resume downgrade/logical fork/archive, workspace-wide unresolved-run fence and manual acknowledgement | `agent_harness/session.py`, `agent_harness/runner.py` | `test_generic_product.py`, `test_context_compaction.py` |
| H-TUI-01 | Streaming TUI without hidden reasoning; session/reconciliation commands, scrollable approvals, read-only `/hooks` and `/mcp`, `/context` and background `/compact` | `agent_harness/tui.py`, `tui_state.py` | `test_generic_product.py`, `test_context_compaction.py`, `test_hooks.py`, `test_mcp_runner_integration.py`; PTY smoke is run locally before release |
| H-CLI-01 | Text and canonical JSONL headless entry plus explicit digest-bound hook/MCP trust administration and MCP catalog refresh | `agent_harness/cli.py` | `test_hooks.py`, `test_mcp.py`, `test_mcp_runner_integration.py`, CLI smoke and package entry-point check |
| H-PKG-01 | Wheel contains only the active Harness package, schemas, license and governance documents | `pyproject.toml` | clean wheel content audit |

All listed verification proves repository engineering behavior only. It is not a
production-security certification.
