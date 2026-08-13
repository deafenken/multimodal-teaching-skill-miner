# Privacy and data governance

Last storage-map review: 2026-08-12. Owner: repository maintainers. This map
must be reviewed whenever a browser storage key, remote recipient, persistent
store path, export field, deletion target, or retention default changes, and at
least once per release. CI tests assert the named browser keys and private
store defaults; a passing test is a freshness aid, not a legal compliance
attestation.

## Data classes

| Class | Examples | Public release |
|---|---|:---:|
| Public code and schemas | Python, tests, JSON Schema, claim-contract example | Yes |
| Synthetic demonstration data | generated slides, tones, anonymized fixture events | Source repository only; excluded from wheel media |
| Aggregate research results | metrics, confidence intervals, limitations, hashes | Candidate only after exporter, audit, and human disclosure-risk review |
| Identifiable classroom media | faces, voices, raw video/audio, archives | No |
| Frame- and row-level research data | frames, OCR text, CLIP embeddings/scores, timestamps, participant/sample IDs, labels, pose/watch features, predictions | No |
| Local real-dashboard inputs and browser responses | non-displayed OCR/features plus browser-served video, frames, captions, teacher-behavior projection, labels, per-sample four-arm predictions, capability URL/token | No |
| Task-two Agent sessions | locally entered goals, anonymous learner profiles, typed responses, transient answer images, bounded OCR evidence, misconception tags, state trajectories, capability URL/token | No |
| Background Teaching tasks | private request bodies, durable dispatch/cancel/resume commands, recovery checkpoints, task/error status codes, content-free purge tombstones | No |
| Cross-session learning schedules | opaque HMAC learner keys, KC/source hashes, outcome evidence, mastery/scheduling state, review leases, erasure tombstones | No |
| Secrets | API keys, credentials, private keys | No |

## Storage boundary

- Store authorized raw data under `data/real/` with directories at mode `0700` and files at mode `0600`.
- Store the complete MIT OCW media under `artifacts/private/full_videos/`; its `videos/`, private manifests, download receipts, and partial downloads are not public artifacts.
- Store extracted lecture frames, OCR text, silence records, CLIP embeddings/prompt scores, events, semantic results, and ablation artifacts under `artifacts/private/full_multimodal/`; a source being publicly viewable does not make these local derivatives part of this project's public release.
- Store TeachObs repository audits, source URLs, complete videos, subtitle tracks, ASR job/results, transcript coverage matrices, frames, audio rows, OCR, embeddings, scene labels, predictions, double-annotation assignments, and disagreement records under `artifacts/private/external_datasets/teachobs/`. Public TeachObs receipts must remain aggregate/hash-only and omit lesson/scene identifiers and row-level content.
- Store participant-level derivatives under `artifacts/dipser_credible/`, `artifacts/real_classroom/`, or other `artifacts/private/` paths; these paths are not public artifacts.
- Keep every input read by `tsm dashboard-real` in an ignored private path. OCR text remains a non-displayed frozen-model input. The TeachObs view may return video, frames, subtitles, behavior projections, labels, timestamps and predictions; the MIT view may return a field-allowlisted Skill projection, short cited subtitle evidence and sanitized candidate-event summaries. Raw Skill JSON fields such as local/job paths, source URLs and OCR text are not returned. Neither private inputs nor browser responses become eligible for GitHub, a wheel, `artifacts/public/`, CI, or a hosted dashboard.
- The Next Teaching Console sends `Cache-Control: no-store`, but browser
  storage is still used. `sessionStorage` keys
  `teachlab.teacher-agent.session-id`, `teachlab.teacher-agent.session-index`,
  and `teachlab.teacher-agent.start-idempotency-key` hold opaque execution
  handles. A legacy `teachlab.chat.threads` value may contain complete Chat
  transcripts, but current Console startup migrates it once into the durable
  default project and deletes it; current turns are not written back to this
  browser key. `localStorage` contains
  `teachlab.learning-project.active-id` and `teachlab.console.preferences`
  (including the web-search preference), plus display-theme/density
  preferences. The obsolete `teachlab.remote-consent.v1` flag is deleted and
  is never promoted into authority; current consent is a revocable,
  server-minted receipt. IndexedDB database `teachlab-console-runtime` stores
  composer drafts, a seven-day bounded recovery outbox, and up to twelve
  terminal-only project snapshots (each at most 5 MB, expiring after seven
  days) so an offline reload can display recent history read-only. These
  snapshots contain project notes and Chat text; they are never submitted as
  authoritative mutations, and running/queued messages are rejected from the
  cache. Project purge also removes its browser snapshot. Same-origin
  JavaScript and DevTools can read all of these values. Project Chat
  transcripts are also saved server-side in the private learning-project
  store; do not describe all browser state as opaque or content-free.
- Server-side Teaching Console state is split across process memory and the
  explicitly configured private stores: project JSON and trash under
  `.private/learning_projects/`; per-syllabus JSON under
  `.private/teaching_syllabi/`; extracted resource text plus chunk offsets
  under `.private/teaching_resource_index/`; and, when `--session-store` is
  supplied, learner profiles/goals/responses/state/idempotency records in an
  integrity-chained JSONL file with related Harness journals, checkpoints,
  leases and content-free tombstones in `<session-store>.harness_streams/`.
  That sibling directory also contains the private background-task registry:
  complete retry requests and its dispatch/cancel/resume command outbox are
  recovery data, not a browser-safe log. Task list/status responses use a
  content-free projection and never expose learner, model, or tool bodies.
  When cross-session scheduling is configured, authoritative assessment
  outcomes and review state are stored in the private append-only
  `--learning-record-store`; its sibling erasure-tombstone and process-lock
  files are private runtime state. The raw `--learner-key-secret-file` is a
  recoverability-critical server secret, must remain mode `0600`, and is never
  returned to the browser. The standalone loopback console derives its opaque
  key from a client-supplied `profile_ref`; that reference is explicitly
  unauthenticated and receives no production cross-user authority. In the
  authenticated Apps API topology, the gateway instead derives a server-only,
  versioned learner reference from the verified OIDC issuer + tenant + subject
  scope and passes it through the private worker bootstrap; the browser cannot
  select or enumerate it. Anonymous/default local profiles receive no
  cross-session learning record.
  When `--metacognition-store` is configured, pre-answer learner JOL values,
  controlled strategy codes, opaque assessment bindings, authoritative
  outcome hashes, and per-KC calibration aggregates are stored in a separate
  private hash-chained JSONL file. Raw questions, answers, reflections, and
  model assessment confidence are not stored there. Its sibling `.erased.json`
  erasure fence and process-lock file are also private runtime state. This
  descriptive pairing does not establish external calibration or learning
  effectiveness and never changes scoring thresholds or mastery by itself.
  Active sessions, pending attachments, staged resources, request queues and
  provider results may additionally exist in process memory. Raw uploaded
  resource/image bytes are processed ephemerally; bounded extracted text/OCR
  may persist in the stores above.
- The authenticated Apps API stores tenant-scoped Sessions, Tasks, Events and
  hash-only session-revocation authorities in PostgreSQL with FORCE RLS. It
  additionally stores account export/deletion operations and permanent
  hash-only deletion tombstones. Raw issuer, subject and tenant claims are not
  written to worker directories: a versioned HMAC namespace selects an opaque
  per-account root. Each root contains that account's Python session/project/
  resource/learning/consent/adjudication/metacognition/safeguarding stores and
  recovery journals. Account deletion fences database writes and workers
  before clearing live rows/root data, while retaining only anti-resurrection
  tombstones and a content-free receipt. PostgreSQL and `scope_data` backups
  remain operator-controlled copies subject to the separate caveat below.
  Browser session-status JSON is deliberately identity-free: it contains only
  authentication state plus the CSRF/expiry fields required by the same-origin
  BFF, never issuer, tenant, subject, email or roles, and is marked `no-store`.
  The HttpOnly session ticket is AES-256-GCM sealed with a domain-separated
  key derived from the active/previous session secret; its Base64 components
  do not expose those claims. Legacy plaintext signed v1/v2 tickets are
  deliberately rejected after this boundary change.
- A safeguarding case is deliberately content-minimized but still sensitive:
  the per-scope hash-chained store records category, severity, observation
  time, a SHA-256 of the blocked learner disclosure, case/version state, an
  emergency-resource policy receipt and (when configured) a content-free
  delivery outbox/SLA state. It never stores the disclosure, learner contact
  details or raw OIDC identity. Its purpose is to hold teaching, provide
  deployment-approved language guidance, and support an authorized staff
  workflow; a queue entry alone is not proof that a human received it. The
  browser receives only the owning scope's content-free projection. Project
  export/purge excludes cases because they are account-scoped; account export
  includes their private audit records, and account deletion purges their live
  store while retaining only hash-only anti-resurrection receipts. Automatic
  case retention is disabled in local/custom deployments unless the deployment
  supplies an explicit versioned policy, minimum closed-case age, bounded
  maximum cases per run, stable deployment context and independent server-only
  retention authority. The supported production Apps topology requires all
  five and refuses to become ready if retention is disabled. When
  enabled, the supervisor may compact only `closed` cases whose configured
  escalation delivery is already `acknowledged`; open, case-acknowledged,
  pending/overdue, and delivery-unavailable cases are never automatically
  cleared. A private 8 MiB, seven-hash Bloom fence permanently fails closed on
  possible erased identities while a bounded recent exact-tombstone cache
  prevents the JSON ledger from exhausting capacity. Its public projection is
  aggregate-only and reports capacity/headroom, compacted/blocked counts, the
  calculated false-positive upper bound plus
  `erasure_fence_false_negative_possible=false`. Near-capacity/blocked stores,
  retention failure, or an estimated false-positive bound above `1e-6` makes
  production readiness fail closed while liveness remains available. The controller still owns
  staff access, delivery routing, the minimum retention value, and backup
  expiry/deletion.
- Store only aggregate publication candidates that have passed dataset-specific disclosure review under `artifacts/public/`; the directory name does not itself certify anonymity.
- Do not upload raw classroom/research media or row-level research derivatives to CI, issue trackers, public artifact stores, model hubs, or remote LLM APIs. The only separate exception implemented here is an explicitly authorized task-two learner-answer image: the original image remains local, while a bounded OCR text envelope may be sent to DeepSeek under the consent and minimization controls described below.

The recorded 10-lecture visual-semantic run kept videos, frames, captions, OCR, embeddings, and events on the local workstation. A GPU server was used only to process publicly available CLIP model weights; no private project media or derivatives were uploaded to it. Final inference over all 2,553 frames ran locally on CPU. Future operators must not infer that server transfer is authorized merely because a model can run faster on GPU: moving any project frame, audio, caption, OCR, embedding, or event record requires a separate documented authorization and transfer-risk review.

The TeachObs ASR handoff is an executable protocol, not evidence that a transfer has occurred or is authorized. Before using its GPU runner, the controller must separately approve the server, region, encryption, access list, logging, retention/deletion, backup behavior, and source/platform terms. Media, job manifests, per-lesson ASR JSON and the private coverage matrix remain restricted even when the server is institutionally managed; only the aggregate/hash-only receipt is a public-release candidate.

The pipeline does not perform face recognition or student identity inference. CLIP's `instructor_talking` and `classroom_wide_view` prompt categories describe relative visual similarity within a fixed ontology; they do not identify a person. Embeddings can still carry information about their source images and must remain private unless a separate disclosure-risk assessment approves release.

`tsm dashboard-real` binds only to `127.0.0.1`, generates a fresh random capability token for each run, and returns private resources with `Cache-Control: no-store`. These controls reduce accidental network exposure and browser caching; they do not replace device access control, informed consent, source-license compliance, screen-sharing discipline, or deletion policy. Treat the printed capability URL/token as session-sensitive, do not paste it into messages or logs, stop the server after the demonstration, and close private browser tabs before screen sharing ends.

## Remote API use

The deterministic heuristic backend is the default. The optional API backend sends up to 80 transcript segments to the configured endpoint. It is blocked for a non-local endpoint until the operator explicitly sets `TSM_ALLOW_REMOTE_TRANSCRIPT_UPLOAD=1` after confirming authorization, data minimization, retention terms, regional transfer requirements, and institutional policy.

The task-two live Teaching Agent is a separate remote-processing path.  Its
DeepSeek backend is fail-closed until the presenter explicitly enables
`TSM_ALLOW_REMOTE_STUDENT_DATA=1` or supplies the equivalent dashboard flag.
Each turn sends only the redacted teaching goal, the minimum relevant anonymous
profile/history, the current structured state, the allowed Skill summary, and
the learner's current typed response. If the learner attaches an answer image,
the original image is processed locally and is never sent to DeepSeek; the
request may additionally contain a bounded, direct-identifier-pattern-redacted
`[LOCAL_VISUAL_EVIDENCE]` text envelope with OCR status, heuristic confidence,
confirmation requirement, and recognized text. Formula-like or low-confidence
OCR is marked for student confirmation and cannot be treated as trusted exact
answer evidence. Classroom video, audio, captions, research-dataset OCR,
embeddings, private evidence pointers, local paths, thumbnails, raw image bytes,
and capability tokens are never included in that request. The local controller
stores hashes, latency, token counts, the bounded visual-evidence record, and the
validated structured result in process memory for the active session; it does
not persist the provider response body. Remote processing is still a data
transfer: use synthetic or properly authorized learner text/OCR, remove identity
information before upload, review the provider's current retention and regional
terms, and do not describe the live DeepSeek mode as fully offline or fully
anonymous.

Chat file attachments use the private teaching-resource store rather than the
learner-answer image channel. Raw files, images, audio, and video are never put
in a Chat model request. The browser uploads a supported file only to the local
`api/resource` endpoint; the Chat request contains at most six project-bound
resource/stage identifier pairs. Immediately before the remote model effect,
the server revalidates project ownership, the exact staged pair, the local
multimodal evidence decision, and a current `remote_chat` consent receipt whose
categories include both `learner_message` and `teaching_resource_excerpt`.
Only query-related, character-bounded extracted text may then cross that
boundary as untrusted quoted context. Conflicting sources, pending visual
review, unverified semantics, missing categories, and revoked receipts fail
closed. Private resource text and excerpts are excluded from Harness SSE
journals and operation results; hash-bound retrieval/citation receipts contain
metadata only. This authority is distinct from `learner_image` consent and
cannot be inferred from it.

The optional web-search path may also send the minimized query/context to the
configured search provider and return source URLs. Authenticated production
also uses an operator-configured OIDC identity provider: the browser is
redirected to its authorization endpoint, and the API sends the authorization
code, PKCE verifier and confidential-client credential (when configured) to
its token endpoint. The IdP therefore receives authentication metadata and
governs its own account/claim retention. DeepSeek, the configured web-search
service, and the configured OIDC IdP are the implemented remote recipients;
there is no built-in cloud backup, product analytics or telemetry collector.
Enabling model/search processing requires the visible versioned consent and
server-owned subject/provider policy gates. Provider-side retention and
regional processing are deployment-supplied policy assertions and are not
erased by deleting this local copy. Operational Prometheus metrics are
content-free and served only behind the private bearer-protected metrics path;
this repository does not ship them to a third party.

## Retention and deletion

The data controller should define a study-specific retention deadline. At that deadline, remove raw media, extracted frames/audio, participant-level feature caches, checkpoints containing identity sets, and row-level predictions from every local and backup location. Aggregate reports may be retained only when they cannot reasonably re-identify participants and the source license permits it.

Learning projects first move to a recoverable private trash location. A
permanent project purge requires both its opaque recovery token and exact
project-bound confirmation text. Before mutation the controller resolves the
complete private target graph and preserves any syllabus, session, or resource
referenced by another active or trashed project. Exclusive project JSON,
session-store records, syllabus JSON, resource index documents, associated
Harness stream artifacts, and exclusively linked opaque learning records are
removed. Exclusively linked terminal background tasks are removed from the
private recovery registry and replaced by content-free identity tombstones;
active tasks block purge, and tasks shared by another project are retained.
A learning-record purge writes a content-free erasure tombstone so
the same opaque identity cannot be silently recreated; local re-enrollment
therefore requires a new profile reference. Shared learning records are
retained when another active or trashed project still references them. The
current review schedule is suspended when its source observation belonged to
an exclusively deleted session, so retained cross-session state cannot keep
serving deleted evidence as an active review source. The returned deletion receipt contains
only identifiers' hashes/counts and no learner-authored content. Repeating the
same completed request returns the stored content-free receipt. A failed or
uncertain operation fails closed and must not be represented as completed.

The project-private ZIP export contains the full validated private project and
associated records, not the browser-safe public projection. Its canonical
manifest lists every member's byte length, SHA-256 digest and data class; the
archive is private and is never suitable for public release merely because it
was exported. This includes full background-task recovery requests for tasks
in the project graph, while the ordinary task status API remains content-free.
Export ZIPs and encrypted backup files selected by the user are
offline, user-managed copies: the running application does not track their
locations and cannot verify or delete them during a project purge. The user or
data controller must separately locate and delete those copies.

`tsm teacher-agent-backup` creates an AES-256-GCM authenticated local backup
using a passphrase-derived scrypt key. It does not upload anywhere.
When a session store is supplied, its complete `harness_streams` sibling is
included, including task registry/outbox, task tombstones, checkpoints and
lock files required for deterministic recovery.
When the learning-record options are supplied, the encrypted payload includes
the learning JSONL store, its sibling erasure-tombstone/process-lock files when
present, and the learner-key secret. The store and secret options are
all-or-none because restoring one without the other cannot reproduce learner
keys. The secret is not a project-owned record and a single-project purge does
not delete it; rotating or destroying a deployment-wide secret is a separate
controller operation.
`tsm teacher-agent-restore-drill` decrypts into a new/empty drill directory,
checks the manifest and every file hash, and never overwrites a live store.
Operators remain responsible for backup location, retention, passphrase
custody, scheduled drills, secure deletion of superseded backups, and a local
RPO/RTO appropriate to their use; this repository makes no cloud-durability or
disaster-recovery SLA claim.

Outside the project purge and the independently authorized safeguarding
closed-case compaction described above, this repository does not run an
automatic retention scheduler. Destruction must be deliberate, scope-checked,
receipted, and performed by the authorized data controller. Browser
`sessionStorage` lasts until the tab/session is closed unless cleared sooner;
`localStorage` remains until cleared by the user/site. IndexedDB recovery data
is bounded to seven days by application checks, but browser storage eviction
and direct site-data deletion can occur sooner. Remote-provider copies and
independently created backups require their own deletion workflows.

In authenticated Apps API deployments, account-level data rights add a separate
scope-wide path. Full export and permanent deletion both require a recent
issuer-bound OIDC AAL2 step-up; a browser cannot submit tenant, subject, role or
scope. The export is a bounded private ZIP with a canonical manifest and entry
hashes covering PostgreSQL teaching/session/task/event/auth-audit data and every
non-secret file in that account's current or previous-version worker roots.
Scope-key envelopes, capabilities, raw tenant/subject claims and host paths are
excluded. This archive is still private learner data and must be protected and
deleted under the controller's own retention schedule.

Remote learner processing eligibility is likewise server-owned. The API reads
an exact state and policy-version pair only from the verified OIDC token, signs
the normalized policy into its HttpOnly session, and never returns it in public
session JSON. A missing, unknown, stale, or explicit-denied assertion becomes
`remote_processing_eligible=false`; browser-submitted age, guardian, provider,
or policy fields are rejected rather than treated as evidence. An authorized
state is bound together with the deployment's explicit provider region,
retention, deletion-status and documentation policy in each consent receipt.
Those provider terms are operator assertions about approved external policy,
not independent proof that remote copies were deleted.

Permanent account deletion first fences and drains the exact account scope,
durably seals all configured scope-key-version tombstones, quarantines and
removes every corresponding worker root, then transactionally deletes Events,
Tasks, Sessions and all device session authorities. The retained receipt and
tombstones contain only irreversible scope/operation hashes, counts, timestamps
and a receipt hash; they are permanent anti-resurrection controls, not learner
records. A separate short-lived HttpOnly capability permits content-free status
after the current session has been deleted. “Permanently deleted” refers only to
data controlled by this TeachLab deployment. It does not delete user-managed
exports/backups or copies already sent to DeepSeek/search providers, which remain
subject to those providers' retention/deletion processes. It also does not
delete the upstream organization identity-provider account; that requires the
organization IdP's own process. Operator backup copies remain pending until the
declared retention expiry or an operator-controlled cryptographic erasure. The
receipt therefore fixes all three caveats to `false`; it cannot be interpreted
as global erasure beyond the live TeachLab deployment.

## Public export procedure

```bash
python3 -m teaching_skill_miner export-public-dipser \
  --input artifacts/dipser_credible/PRIVATE_RUN/hierarchical_0_9_report.json \
  --output artifacts/public/dipser_summary.json

python3 scripts/build_multimodal_public_receipts.py

python3 -m teaching_skill_miner release-audit dist/teaching_skill_miner-*.whl
python3 -m teaching_skill_miner release-audit artifacts/public
```

The aggregate exporter intentionally removes sample IDs, participant IDs, paths, matrices, and row-level predictions. The full-video receipt omits media/caption/frame content and local paths but retains public course/video identifiers with per-video hashes, durations, and stream counts. The two multimodal receipts are stricter: they also omit OCR text, embeddings, per-lecture records, lecture identifiers, and personal identity fields.
This is data minimization, not a mathematical anonymity guarantee. Before publication, inspect rare cells, small groups, free text, timestamps, model outputs, and linkage risk against other available datasets; `tsm release-audit` cannot prove that re-identification is impossible.
