# Requirements traceability

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| H-CORE-01 | Bounded model/tool loop | `agent_harness/core/runtime.py` | `tests_harness/test_agent_harness.py` |
| H-EVENT-01 | Continuous typed events and one terminal | `agent_harness/core/events.py` | `tests_harness/test_harness_controller.py`, `test_generic_product.py` |
| H-DUR-01 | Durable-first hash-chain journal and checkpoint | `agent_harness/core/journal.py` | `test_harness_journal.py`, `test_harness_runtime_journal.py` |
| H-PROV-01 | Provider-neutral contracts and tool-aware DeepSeek adapter | `agent_harness/core/providers.py`, `agent_harness/providers/deepseek.py` | `test_harness_provider_registry.py`, `test_generic_product.py` |
| H-SEC-01 | Central schema, permission, data-scope metadata and replay-policy checks; no claim that scopes sandbox handler authority | `agent_harness/core/tools.py` | `test_harness_tool_security.py` |
| H-WS-01 | Workspace-confined built-in file/patch handlers plus an explicitly host-authorized shell | `agent_harness/toolsets/workspace.py` | `test_generic_product.py` |
| H-SES-01 | Private workspace-scoped sessions, serialized turns, resume downgrade/logical fork/archive, workspace-wide unresolved-run fence and manual acknowledgement | `agent_harness/session.py`, `agent_harness/runner.py` | `test_generic_product.py` |
| H-TUI-01 | Streaming TUI without hidden reasoning; session commands plus `/effects` and `/reconcile` | `agent_harness/tui.py`, `tui_state.py` | `test_generic_product.py`; PTY smoke is run locally before release |
| H-CLI-01 | Text and canonical JSONL headless entry | `agent_harness/cli.py` | CLI smoke and package entry-point check |
| H-PKG-01 | Wheel contains only the active Harness package and schemas | `pyproject.toml` | clean wheel content audit |

All listed verification proves repository engineering behavior only. It is not a
production-security certification.
