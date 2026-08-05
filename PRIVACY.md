# Privacy and data governance

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
| Secrets | API keys, credentials, private keys | No |

## Storage boundary

- Store authorized raw data under `data/real/` with directories at mode `0700` and files at mode `0600`.
- Store the complete MIT OCW media under `artifacts/private/full_videos/`; its `videos/`, private manifests, download receipts, and partial downloads are not public artifacts.
- Store extracted lecture frames, OCR text, silence records, CLIP embeddings/prompt scores, events, semantic results, and ablation artifacts under `artifacts/private/full_multimodal/`; a source being publicly viewable does not make these local derivatives part of this project's public release.
- Store TeachObs repository audits, source URLs, complete videos, subtitle tracks, ASR job/results, transcript coverage matrices, frames, audio rows, OCR, embeddings, scene labels, predictions, double-annotation assignments, and disagreement records under `artifacts/private/external_datasets/teachobs/`. Public TeachObs receipts must remain aggregate/hash-only and omit lesson/scene identifiers and row-level content.
- Store participant-level derivatives under `artifacts/dipser_credible/`, `artifacts/real_classroom/`, or other `artifacts/private/` paths; these paths are not public artifacts.
- Keep every input read by `tsm dashboard-real` in an ignored private path. OCR text remains a non-displayed frozen-model input. The TeachObs view may return video, frames, subtitles, behavior projections, labels, timestamps and predictions; the MIT view may return a field-allowlisted Skill projection, short cited subtitle evidence and sanitized candidate-event summaries. Raw Skill JSON fields such as local/job paths, source URLs and OCR text are not returned. Neither private inputs nor browser responses become eligible for GitHub, a wheel, `artifacts/public/`, CI, or a hosted dashboard.
- `tsm teacher-agent-dashboard` keeps its active teaching session in process memory and sends `Cache-Control: no-store`. It does not put goals, responses, profiles, misconceptions, or state trajectories in browser storage. `sessionStorage` holds only a random opaque session handle so the same tab can resume while the local process remains alive; that handle is not teaching content but can still be read by same-origin JavaScript or DevTools. `localStorage` is limited to paper/night and density display preferences. Command-line session files use user-only permissions and belong under `artifacts/private/`; the packaged task-two fixtures are synthetic.
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

## Retention and deletion

The data controller should define a study-specific retention deadline. At that deadline, remove raw media, extracted frames/audio, participant-level feature caches, checkpoints containing identity sets, and row-level predictions from every local and backup location. Aggregate reports may be retained only when they cannot reasonably re-identify participants and the source license permits it.

This repository does not automatically delete user data. Destruction must be deliberate, scope-checked, logged, and performed by the authorized data controller.

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
