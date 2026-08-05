# Third-party data and license boundaries

The project license covers this repository's original code and documentation only. It does not relicense classroom recordings, transcripts, labels, pose metadata, sensor streams, pretrained models, fonts, codecs, or external tools.

## Bundled teaching links

The 2×5 offline teaching corpus contains short curated paraphrase excerpts and source links for engineering demonstration. It is not a redistribution of full videos or official captions and is not formal empirical evidence.

`data/formal_caption_sources.json` contains only MIT OpenCourseWare lecture-page, WebVTT, and page-linked media URLs plus fixed caption hashes and timing references. `tsm fetch-formal-captions` verifies those live relationships and writes full caption text under `artifacts/private/formal_captions/`; full captions are deliberately excluded from the wheel and public receipt. The official captions retain the upstream CC BY-NC-SA terms, attribution requirements, and third-party-material exceptions. A successful hash/coverage audit establishes transcript provenance and completeness only, not learner effectiveness or deployment accuracy.

## MIT OpenCourseWare full videos

With explicit `--acknowledge-source-terms`, `tsm fetch-full-videos` downloads the page-linked MP4 files for the first five MIT 18.06 lectures and first five MIT 6.0001 lectures into `artifacts/private/full_videos/`. The completed private dataset contains 10 videos, 1,038,813,006 bytes, and 26,656.83 seconds of media. It is used for whole-timeline audio, frame, OCR, visual-semantic, and event analysis under `artifacts/private/full_multimodal/`.

The raw videos, caption text, extracted frames, OCR text, and embeddings are not bundled in the wheel and are not candidates for public redistribution. `artifacts/public/full_video_validation_receipt.json` contains only content-free validation metadata such as aggregate counts, local post-download hashes, durations, and stream counts. Those local SHA-256 values detect later mutation of the downloaded files; because the source index does not pin publisher-provided media hashes, they do not independently authenticate the publisher's original bytes.

`tsm dashboard-real` may replay an authorized local copy together with its local captions and frozen per-sample predictions. OCR can remain a private input to the frozen visual arms, but its text is not displayed by the current viewer. This is a private, same-device research display only. The viewer does not change the upstream license, create redistribution rights, or permit publishing the media or derivatives to GitHub, a wheel, a hosted page, or a public artifact directory.

The same rule applies to the MIT OCW Skill view. It may locally display a sanitized projection of a Full Skill and short evidence excerpts from the authorized local research artifacts, but it does not grant redistribution rights for the original media, complete captions, raw Skill JSON, frames, OCR text or event payloads. Only the generic data-free viewer assets may enter the wheel.

MIT OCW course content and linked media retain the upstream CC BY-NC-SA terms, required attribution, non-commercial/share-alike conditions, and any third-party-material exceptions shown by the source. The repository's own license does not replace or broaden those terms. Users must review the specific course and lecture pages before reuse or redistribution.

## OpenAI CLIP model

The visual-semantic stage uses `openai/clip-vit-base-patch32` revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. The locally used FP16 `model.safetensors` has SHA-256 `676093550c9e05bc3ba55256c278c89f0d15a1a1585f3b81d76454c33b852d5e`. Model weights and tokenizer/config files remain third-party artifacts: they are stored outside the release wheel and are not relicensed by this project.

Operators must review the upstream model card, license, training-data limitations, and intended-use restrictions before obtaining or using the model. The current closed-ontology prompt scores are uncalibrated on these lectures and must not be presented as validated classification probabilities, recognition accuracy, or suitability for high-impact teacher/student decisions.

## OUC-CGE

OUC-CGE contains identifiable adult classroom video. Its paper, OSF metadata, and code repository use different license statements. Until clarified, use the stricter non-commercial research interpretation and satisfy privacy and institutional requirements independently of copyright permission.

## DIPSER V5

DIPSER contains RGB-derived pose data, wearable sensor data, expert labels, and participant/session identities. The data page indicates CC BY 4.0 while the paper also describes academic/research restrictions and separate commercial approval. Until clarified, treat it as non-commercial research data and never redistribute row-level artifacts.

## TeachObs v0.1

The project imports the annotation repository at commit `96c251ae09e79edd06a3a9bbaaa8b8f7fe99a15c`. Its released annotation assets are marked CC BY 4.0 and contain 30 lessons, 5,158 15-second scenes, and 39 consensus teaching-behavior labels. The source paper reports seven independent coders, but the release does not include their individual coding files, so this project cannot independently recompute inter-rater reliability. The release's coding-scheme entries contain no non-empty definitions; users must not infer missing operational definitions from the label names alone.

Source classroom videos are linked separately and retain their original rights, consent/privacy conditions, and platform terms. They are downloaded only after explicit `--acknowledge-source-terms`, kept under `artifacts/private/external_datasets/teachobs/`, and excluded from the wheel and public receipts together with frames, captions, OCR, embeddings, URLs, lesson identifiers, and scene-level labels. Local SHA-256 values establish post-download integrity bindings, not publisher authenticity or redistribution permission.

If those private TeachObs artifacts are selected for `tsm dashboard-real`, the same exclusion applies to every item visible in the local page, including complete media, caption/ASR text, frames, ground-truth labels and transcript-only / +audio / +visual / full per-scene predictions. It also applies to non-displayed private inputs such as OCR text and embeddings. Local loopback delivery is not a public release and is not evidence that downstream redistribution is licensed.

## MM-TBA

The audited Figshare release for MM-TBA (DOI `10.6084/m9.figshare.28942487.v1`) is marked CC BY 4.0 and reports 3,173 clips and 19,020 final labels with at least two annotators. The inspected release contains adjudicated labels rather than separate annotator-A/annotator-B files and does not provide the original video pixels needed by this project's full visual pipeline. It is therefore documented as a candidate external dataset, not treated as completed independent double annotation or a raw-video deployment lockbox.

## Pri-MCCD

Pri-MCCD contains identifiable minors and is controlled-access. This project does not download it, accept its DUA, or bypass its access process.

Users are responsible for checking the current upstream terms, attribution, consent, ethics approval, data-use agreement, retention policy, and geographic/legal requirements before every collection or deployment.
