# Contributing

Agent Harness is a domain-neutral coding-agent runtime. New changes must preserve:

- one typed, continuous event stream with exactly one terminal event;
- durable-first journal publication and atomic checkpoints;
- central tool authorization, schema validation and bounded output;
- built-in file/patch path confinement and explicit permission profiles;
- no hidden reasoning, credentials or planner envelopes in user-visible output;
- no product-domain prompts or application-specific state in the core.

## Local checks

```bash
python -m pip install -e '.[dev]'
ruff check agent_harness tests_harness
pytest -q
python -m build
```

Add tests for all lifecycle, security and recovery changes. Provider integrations must
work through the generic contracts in `agent_harness/core`; do not bypass the tool
registry or write directly to session files.

Security documentation must distinguish authorization metadata from containment.
`data_scope` controls registry eligibility; it is not a filesystem, network, process or
memory sandbox. Likewise, session fork is a logical transcript fork, and `full-access`
shell execution retains the current OS user's host authority.

Do not commit API keys, session journals, private workspaces, build artifacts or legacy
runtime data.
