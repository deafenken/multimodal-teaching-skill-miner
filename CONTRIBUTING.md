# Contributing

## Development setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[recognition,dev]'
python -m pytest
```

Use Python 3.10–3.13. Keep the standard-library-only core usable without recognition extras.

## Change requirements

- Add tests for behavior changes and tamper tests for integrity/security boundaries.
- Preserve exact sample ordering, identity-disjoint evaluation, nested fitting, and frozen claim contracts.
- Mark post-selection, transductive, retrospective, synthetic, and human-provided evidence explicitly.
- Never convert an automatic structure score into a learning-effectiveness claim.
- Never convert an offline development score into a deployment claim.
- Do not commit identifiable media, participant-level artifacts, credentials, or absolute local paths.
- Use `write_json`/`write_text` for atomic generated artifacts.

Before proposing a change, run `sh scripts/verify_project.sh`. Changes to a data importer, feature schema, model, gate, or evaluation protocol must update the relevant schema, documentation, fingerprint, and validation report.
