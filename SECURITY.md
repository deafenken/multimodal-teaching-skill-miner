# Security policy

## Supported version

Security and privacy fixes target the current `1.2.x` line. This project is a research system, not a managed production service.

## Reporting

Do not publish credentials, identifiable classroom media, participant-level features, exploit details, or affected archives in a public issue. Contact the repository owner through a private channel or a private security advisory and include only the minimum reproducible information.

## Security boundaries

- `TSM_API_KEY` must stay in environment variables or a local secret manager. It is never printed by `tsm doctor`.
- A non-local API endpoint must use HTTPS and requires the explicit `TSM_ALLOW_REMOTE_TRANSCRIPT_UPLOAD=1` acknowledgement.
- Real classroom media and row-level DIPSER/OUC artifacts are private research data. They are excluded by `.gitignore` and rejected by `tsm release-audit`.
- `tsm dashboard-real` is a local capability-scoped viewer, not a network service: it must bind only to `127.0.0.1`, reject requests without the per-run random capability token, and send `Cache-Control: no-store` for the HTML, structured records, captions, frames, and video responses. OCR remains a private frozen-model input but its text is not returned to the browser. Raw Skill objects must be projected through the explicit presentation allowlist so local paths, source URLs, job directories and OCR evidence cannot cross the server boundary. Do not proxy, tunnel, port-forward, or rebind it to a LAN/public address.
- `tsm teacher-agent-dashboard` has the same loopback/capability/no-store boundary, additionally rejects non-JSON or oversized request bodies and keeps the teaching session in memory. It is a single-presenter defense demo, not a hardened multi-user service; do not reverse-proxy, tunnel, port-forward, or expose its port.
- The task-two DeepSeek client accepts only the explicitly allowlisted `deepseek-v4-flash` model and an HTTPS origin, requires an affirmative remote-student-data consent flag, limits response size, validates JSON output, and never forwards classroom media. Keep its credential in the ignored `.private/deepseek_api.txt` path, an external permission-restricted file, or a secret manager; never put the key value in `.env.example`, command history, screenshots, reports, browser state, or Git. A local dashboard URL does not mean model inference is local when the ONLINE backend is enabled.
- The capability URL/token grants access for the life of that local process. Treat it as sensitive, do not include it in screenshots, shell transcripts, issues, or telemetry, and terminate the process after use. Loopback plus a token reduces exposure but does not protect a compromised or shared host.
- Raw-to-feature outputs contain row-level hashes, windows, features, and provenance even though they omit media bytes. Keep `strict_feature_bundle.json` under a private `0700/0600` path and never treat successful extraction as an accuracy or deployment result.
- ZIP member paths, symlink/encryption flags, duplicate names, compression-resource limits, archive integrity, content hashes, feature matrices, checkpoint v3, identity/claim-cluster contracts, external claim contracts, coverage evidence, signed freeze registrations, one-time ledgers, and signed evaluation receipts are validated fail-closed.
- Ed25519 verification proves signature integrity against the supplied trust anchor; it does not prove that the signer is organizationally independent. Pin the custodian public key through an external governance channel before evaluation and never store the private key in this repository.
- Generated Skills and recognition predictions must not be the sole basis for grading, discipline, surveillance, diagnosis, hiring, or other high-impact decisions.

## Before a release

Run:

```bash
python3 scripts/audit_repository_privacy.py
sh scripts/build_release_acceptance.sh
```

The first command checks the Git-tracked public boundary and therefore requires a real Git checkout. The final command performs the full project checks, reproducible double-build, isolated exact-wheel smoke, and release audits before atomically writing the verified wheel and its machine-generated acceptance. The acceptance generator rejects stale receipts, artifact/hash/member mismatches, changed verification code, failed audits, and non-passing checks; it never carries fields forward from an older acceptance. If `.git` is absent, the workflow file may be inspected but the tracked-file scan has not actually run, and the generated acceptance records that limitation.

`tsm release-audit` and `audit_repository_privacy.py` are conservative publication guards for media/archive/model suffixes and signatures, unknown binary payloads, UTF-8/UTF-16 secrets, local paths, and parsed row-level identity fields. Passing them is **not** a complete de-identification proof: they cannot establish that free text, aggregate cells, rare combinations, embeddings, screenshots, or other novel artifacts are impossible to re-identify. Publication still requires a dataset-specific privacy review, disclosure-risk assessment, ethics/licensing approval, and human inspection of the exact release candidate.

The wheel may contain the generic `dashboard-real` viewer code and its data-free HTML/CSS/JavaScript templates. It must never contain the inputs or session outputs consumed by that viewer: no real video/audio, complete captions, OCR text, labels, sample identifiers, per-sample predictions, raw Skill artifacts, cited private records, embeddings, local paths, or capability tokens. Verify the exact wheel rather than assuming that a local demonstration remained outside the build context.
