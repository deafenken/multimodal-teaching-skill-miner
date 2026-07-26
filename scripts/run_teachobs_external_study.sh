#!/usr/bin/env sh
set -eu
umask 077

usage() {
  cat >&2 <<'EOF'
usage: scripts/run_teachobs_external_study.sh --acknowledge-source-terms [OPTIONS]

Resume the real 30-lesson TeachObs external multimodal study. The runner:
  1. verifies or refreshes the commit-pinned annotation audit and public receipt;
  2. rebuilds the private 30-lesson media plan;
  3. reuses a valid caption audit, or runs/retries the privacy-safe audit;
  4. resumes downloads, then applies the selected exact media profile gate;
  5. requires the complete hash-bound six-lesson ASR import and materializes
     all 4,945 selected scene transcripts without released-text fallback;
  6. extracts the selected audio/OCR/image/change/CLIP scene features;
  7. runs all four arms and exports a deterministic private JSON+NPZ bundle;
  8. safely regenerates only the blank pending double-annotation skeleton;
  9. binds the frozen bundle into an externally unregistered lockbox draft;
 10. audits the aggregate-only public artifact set.

Required authorization:
  --acknowledge-source-terms
      May instead be set exactly as TSM_TEACHOBS_SOURCE_TERMS_ACKNOWLEDGED=true.

Important options:
  --evaluation-profile PROFILE  paper_track1_23_train_6_test (default).
                                The full profile remains unavailable until the
                                missing S4 media/transcript chain is complete.
  --clip-model PATH             Local CLIP snapshot (no model download).
  --clip-revision REV           Immutable CLIP source revision.
  --clip-device DEVICE          cpu, mps, cuda, or a backend-specific device.
  --clip-batch-size N           Default: 32.
  --retry-captions              Retry even when a valid prior audit exists.
  --refresh-annotations         Re-fetch the pinned annotation audit.
  --source-override-manifest P  Explicit private mirror manifest; disabled by default.
  --acknowledge-override-source-terms
      Required in addition to source acknowledgement when an override is supplied.
  --js-runtime SPEC             Optional local yt-dlp JS runtime; empty by default.
  --cookies-from-browser SPEC   Explicit opt-in: browser or browser:profile-name.
      Disabled by default. This exposes an authenticated browser session to the
      video platform and can trigger account challenges or suspension; use only
      with the account owner's explicit authorization. Cookies are not exported.
  --yt-dlp-direct               Explicitly bypass inherited HTTP(S) proxy variables.
  --yt-dlp-impersonate chrome  Request local yt-dlp/curl_cffi Chrome impersonation.
  --yt-dlp-youtube-client android_vr
      Use the fixed YouTube client for caption metadata and VTT retrieval.
  --jobs N                      Concurrent media/caption downloads (default: 4).
  --feature-jobs N              Concurrent lesson feature workers (default: 2).
  --visual-jobs N               Image workers per lesson (default: 1).
  --ocr-jobs N                  OCR workers per lesson (default: 1).

All paths and executable choices can also be configured with the TSM_TEACHOBS_*
variables documented in docs/teachobs_external_study_runner.md. Remote yt-dlp
EJS components are never installed or enabled by this runner.
EOF
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

tsm_acknowledged=${TSM_TEACHOBS_SOURCE_TERMS_ACKNOWLEDGED:-false}
tsm_override_acknowledged=${TSM_TEACHOBS_OVERRIDE_SOURCE_TERMS_ACKNOWLEDGED:-false}
tsm_retry_captions=false
tsm_refresh_annotations=false
tsm_source_override_manifest=${TSM_TEACHOBS_SOURCE_OVERRIDE_MANIFEST:-}
tsm_clip_model=${TSM_TEACHOBS_CLIP_MODEL:-artifacts/private/models/openai_clip_vit_b32_fp16_3d74acf9}
tsm_clip_revision=${TSM_TEACHOBS_CLIP_REVISION:-3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268}
tsm_clip_device=${TSM_TEACHOBS_CLIP_DEVICE:-cpu}
tsm_clip_batch_size=${TSM_TEACHOBS_CLIP_BATCH_SIZE:-32}
tsm_js_runtime=${TSM_TEACHOBS_JS_RUNTIME:-}
tsm_cookies_from_browser=${TSM_TEACHOBS_COOKIES_FROM_BROWSER:-}
tsm_yt_dlp_direct=${TSM_TEACHOBS_YT_DLP_DIRECT:-false}
tsm_yt_dlp_impersonate=${TSM_TEACHOBS_YT_DLP_IMPERSONATE:-}
tsm_yt_dlp_youtube_client=${TSM_TEACHOBS_YT_DLP_YOUTUBE_CLIENT:-}
tsm_jobs=${TSM_TEACHOBS_JOBS:-4}
tsm_feature_jobs=${TSM_TEACHOBS_FEATURE_JOBS:-2}
tsm_visual_jobs=${TSM_TEACHOBS_VISUAL_JOBS:-1}
tsm_ocr_jobs=${TSM_TEACHOBS_OCR_JOBS:-1}
tsm_evaluation_profile=${TSM_TEACHOBS_EVALUATION_PROFILE:-paper_track1_23_train_6_test}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --acknowledge-source-terms)
      tsm_acknowledged=true
      shift
      ;;
    --acknowledge-override-source-terms)
      tsm_override_acknowledged=true
      shift
      ;;
    --retry-captions)
      tsm_retry_captions=true
      shift
      ;;
    --refresh-annotations)
      tsm_refresh_annotations=true
      shift
      ;;
    --yt-dlp-direct)
      tsm_yt_dlp_direct=true
      shift
      ;;
    --source-override-manifest|--clip-model|--clip-revision|--clip-device|--clip-batch-size|--js-runtime|--cookies-from-browser|--yt-dlp-impersonate|--yt-dlp-youtube-client|--jobs|--feature-jobs|--visual-jobs|--ocr-jobs|--evaluation-profile)
      if [ "$#" -lt 2 ]; then
        echo "missing value for $1" >&2
        usage
        exit 2
      fi
      case "$1" in
        --source-override-manifest) tsm_source_override_manifest=$2 ;;
        --clip-model) tsm_clip_model=$2 ;;
        --clip-revision) tsm_clip_revision=$2 ;;
        --clip-device) tsm_clip_device=$2 ;;
        --clip-batch-size) tsm_clip_batch_size=$2 ;;
        --js-runtime) tsm_js_runtime=$2 ;;
        --cookies-from-browser) tsm_cookies_from_browser=$2 ;;
        --yt-dlp-impersonate) tsm_yt_dlp_impersonate=$2 ;;
        --yt-dlp-youtube-client) tsm_yt_dlp_youtube_client=$2 ;;
        --jobs) tsm_jobs=$2 ;;
        --feature-jobs) tsm_feature_jobs=$2 ;;
        --visual-jobs) tsm_visual_jobs=$2 ;;
        --ocr-jobs) tsm_ocr_jobs=$2 ;;
        --evaluation-profile) tsm_evaluation_profile=$2 ;;
      esac
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$tsm_acknowledged" in
  true) ;;
  false)
    echo "refusing third-party annotation/media work without --acknowledge-source-terms" >&2
    exit 2
    ;;
  *)
    echo "TSM_TEACHOBS_SOURCE_TERMS_ACKNOWLEDGED must be exactly true or false" >&2
    exit 2
    ;;
esac

case "$tsm_override_acknowledged" in
  true|false) ;;
  *)
    echo "TSM_TEACHOBS_OVERRIDE_SOURCE_TERMS_ACKNOWLEDGED must be exactly true or false" >&2
    exit 2
    ;;
esac
case "$tsm_yt_dlp_direct" in
  true|false) ;;
  *)
    echo "TSM_TEACHOBS_YT_DLP_DIRECT must be exactly true or false" >&2
    exit 2
    ;;
esac
case "$tsm_yt_dlp_youtube_client" in
  ''|android_vr) ;;
  *)
    echo "TSM_TEACHOBS_YT_DLP_YOUTUBE_CLIENT must be empty or android_vr" >&2
    exit 2
    ;;
esac
case "$tsm_evaluation_profile" in
  paper_track1_23_train_6_test) ;;
  full_23_train_7_test)
    echo "full_23_train_7_test is unavailable until the S4 media/transcript chain is complete" >&2
    exit 2
    ;;
  *)
    echo "evaluation profile must be paper_track1_23_train_6_test" >&2
    exit 2
    ;;
esac
if [ -n "$tsm_source_override_manifest" ]; then
  if [ "$tsm_override_acknowledged" != true ]; then
    echo "an explicit source override also requires --acknowledge-override-source-terms" >&2
    exit 2
  fi
  if [ ! -f "$tsm_source_override_manifest" ] || [ -L "$tsm_source_override_manifest" ]; then
    echo "source override manifest is not a regular non-symlink file: $tsm_source_override_manifest" >&2
    exit 2
  fi
elif [ "$tsm_override_acknowledged" = true ]; then
  echo "override terms acknowledgement was supplied without an override manifest" >&2
  exit 2
fi

for tsm_number in \
  "$tsm_clip_batch_size" "$tsm_jobs" "$tsm_feature_jobs" \
  "$tsm_visual_jobs" "$tsm_ocr_jobs"
do
  case "$tsm_number" in
    ''|*[!0-9]*|0)
      echo "worker counts and batch size must be positive integers" >&2
      exit 2
      ;;
  esac
done
if [ $((tsm_feature_jobs * tsm_visual_jobs)) -gt 8 ] || \
   [ $((tsm_feature_jobs * tsm_ocr_jobs)) -gt 8 ]; then
  echo "feature-jobs times either nested worker count must not exceed 8" >&2
  exit 2
fi
if [ -z "$tsm_clip_revision" ] || [ -z "$tsm_clip_device" ]; then
  echo "CLIP revision and device must be non-empty" >&2
  exit 2
fi
if [ ! -d "$tsm_clip_model" ]; then
  echo "local CLIP model directory not found: $tsm_clip_model" >&2
  exit 2
fi
if [ ! -f "$tsm_clip_model/model.safetensors" ] && \
   [ ! -f "$tsm_clip_model/pytorch_model.bin" ]; then
  echo "local CLIP model weights not found in: $tsm_clip_model" >&2
  exit 2
fi

tsm_python=${TSM_TEACHOBS_PYTHON:-python3}
tsm_yt_dlp_python=${TSM_TEACHOBS_YT_DLP_PYTHON:-$tsm_python}
tsm_yt_dlp_module_dir=${TSM_TEACHOBS_YT_DLP_MODULE_DIR:-artifacts/private/tools/yt_dlp}
tsm_ffmpeg=${TSM_TEACHOBS_FFMPEG:-ffmpeg}
tsm_ffprobe=${TSM_TEACHOBS_FFPROBE:-ffprobe}
tsm_private_root=${TSM_TEACHOBS_PRIVATE_ROOT:-artifacts/private/external_datasets/teachobs}
tsm_repository=${TSM_TEACHOBS_REPOSITORY:-$tsm_private_root/repository}
tsm_annotation_import=${TSM_TEACHOBS_ANNOTATION_IMPORT:-$tsm_private_root/imported_annotations}
tsm_acquisition_receipt=${TSM_TEACHOBS_ACQUISITION_RECEIPT:-$tsm_private_root/acquisition_receipt.json}
tsm_media_root=${TSM_TEACHOBS_MEDIA_ROOT:-$tsm_private_root/media}
tsm_caption_root=${TSM_TEACHOBS_CAPTION_ROOT:-$tsm_private_root/captions}
tsm_asr_root=${TSM_TEACHOBS_ASR_ROOT:-$tsm_private_root/asr}
tsm_asr_results=${TSM_TEACHOBS_ASR_RESULTS:-$tsm_asr_root/results}
tsm_transcript_root=${TSM_TEACHOBS_TRANSCRIPT_ROOT:-$tsm_private_root/materialized_transcripts}
tsm_human_root=${TSM_TEACHOBS_HUMAN_ROOT:-$tsm_private_root/human_annotation}
tsm_benchmark_result=${TSM_TEACHOBS_BENCHMARK_RESULT:-$tsm_private_root/multimodal_benchmark_result.json}
tsm_frozen_model_root=${TSM_TEACHOBS_FROZEN_MODEL_ROOT:-$tsm_private_root/frozen_models}
tsm_public_root=${TSM_TEACHOBS_PUBLIC_ROOT:-artifacts/public}
tsm_lockbox_study_id=${TSM_TEACHOBS_LOCKBOX_STUDY_ID:-teachobs-new-site-confirmatory-2026-draft}

tsm_annotation_audit=$tsm_annotation_import/dataset_audit.json
tsm_annotation_receipt=$tsm_public_root/teachobs_annotation_receipt.json
tsm_media_plan=$tsm_media_root/media_plan.json
tsm_media_manifest=$tsm_media_root/media_manifest.json
tsm_feature_manifest=$tsm_media_root/feature_manifest.json
tsm_caption_audit=$tsm_caption_root/caption_audit.json
tsm_caption_receipt=$tsm_public_root/teachobs_caption_receipt.json
tsm_asr_job_manifest=$tsm_asr_root/job_manifest.json
tsm_asr_import_audit=$tsm_asr_root/import_audit.json
tsm_asr_coverage_matrix=$tsm_asr_root/transcript_coverage_matrix.json
tsm_asr_receipt=$tsm_public_root/teachobs_asr_receipt.json
tsm_transcript_manifest=$tsm_transcript_root/manifest.json
tsm_transcript_receipt=$tsm_public_root/teachobs_transcript_materialization_receipt.json
tsm_benchmark_receipt=$tsm_public_root/teachobs_multimodal_benchmark_receipt.json
tsm_human_manifest=$tsm_human_root/assignment_manifest.json
tsm_human_receipt=${TSM_TEACHOBS_HUMAN_RECEIPT:-$tsm_public_root/teachobs_human_annotation_receipt.json}
tsm_lockbox_draft=${TSM_TEACHOBS_LOCKBOX_DRAFT:-$tsm_public_root/teachobs_new_site_lockbox_preregistration_draft.json}
tsm_checker=$repo_root/scripts/check_teachobs_external_study.py

mkdir -p "$tsm_private_root" "$tsm_annotation_import" "$tsm_media_root" \
  "$tsm_caption_root" "$tsm_asr_root" "$tsm_human_root" "$tsm_public_root"
chmod 700 "$tsm_private_root" "$tsm_annotation_import" "$tsm_media_root" \
  "$tsm_caption_root" "$tsm_asr_root" "$tsm_human_root"

run_asr_receipt_check() {
  tsm_asr_check_mode=$1
  set -- "$tsm_python" "$tsm_checker" asr \
    --receipt "$tsm_asr_receipt" \
    --media-manifest "$tsm_media_manifest" \
    --caption-audit "$tsm_caption_audit" \
    --evaluation-profile "$tsm_evaluation_profile"
  case "$tsm_asr_check_mode" in
    full)
      set -- "$@" \
        --job-manifest "$tsm_asr_job_manifest" \
        --import-audit "$tsm_asr_import_audit" \
        --coverage-matrix "$tsm_asr_coverage_matrix"
      ;;
    job)
      set -- "$@" --job-manifest "$tsm_asr_job_manifest"
      ;;
    none) ;;
    *)
      echo "invalid internal ASR receipt-check mode" >&2
      return 2
      ;;
  esac
  "$@"
}

if [ -n "$tsm_yt_dlp_module_dir" ] && [ -d "$tsm_yt_dlp_module_dir" ]; then
  case "$tsm_yt_dlp_module_dir" in
    /*) tsm_yt_dlp_import_root=$tsm_yt_dlp_module_dir ;;
    *) tsm_yt_dlp_import_root=$repo_root/$tsm_yt_dlp_module_dir ;;
  esac
  if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH=$tsm_yt_dlp_import_root:$PYTHONPATH
  else
    PYTHONPATH=$tsm_yt_dlp_import_root
  fi
  export PYTHONPATH
fi

command -v "$tsm_python" >/dev/null 2>&1 || {
  echo "Python executable not found: $tsm_python" >&2
  exit 2
}
"$tsm_python" -c \
  'import sys; from teaching_skill_miner.teachobs_media import validate_teachobs_cookies_from_browser, validate_teachobs_ytdlp_transport; validate_teachobs_cookies_from_browser(sys.argv[1] or None); validate_teachobs_ytdlp_transport(direct=sys.argv[2] == "true", impersonate=sys.argv[3] or None)' \
  "$tsm_cookies_from_browser" "$tsm_yt_dlp_direct" \
  "$tsm_yt_dlp_impersonate" || {
  echo "invalid browser-cookie or yt-dlp transport opt-in" >&2
  exit 2
}
command -v "$tsm_ffmpeg" >/dev/null 2>&1 || {
  echo "FFmpeg executable not found: $tsm_ffmpeg" >&2
  exit 2
}
command -v "$tsm_ffprobe" >/dev/null 2>&1 || {
  echo "FFprobe executable not found: $tsm_ffprobe" >&2
  exit 2
}
command -v tesseract >/dev/null 2>&1 || {
  echo "Tesseract is required for the complete OCR arm" >&2
  exit 2
}
"$tsm_yt_dlp_python" -c 'import yt_dlp' >/dev/null 2>&1 || {
  echo "yt-dlp is not importable by: $tsm_yt_dlp_python" >&2
  exit 2
}
"$tsm_python" -c \
  'import numpy, PIL, scipy, sklearn, torch, transformers' >/dev/null 2>&1 || {
  echo "NumPy/Pillow/SciPy/scikit-learn/Torch/Transformers are required" >&2
  exit 2
}

echo "[TeachObs 1/10] commit-pinned annotation audit and aggregate receipt"
if [ "$tsm_refresh_annotations" = false ] && \
   "$tsm_python" "$tsm_checker" annotations \
     --audit "$tsm_annotation_audit" \
     --receipt "$tsm_annotation_receipt" >/dev/null 2>&1; then
  echo "reusing verified pinned annotation audit"
else
  "$tsm_python" -m teaching_skill_miner fetch-teachobs \
    --output "$tsm_annotation_import" \
    --public-receipt "$tsm_annotation_receipt" \
    --acknowledge-source-terms
fi
"$tsm_python" "$tsm_checker" annotations \
  --audit "$tsm_annotation_audit" \
  --receipt "$tsm_annotation_receipt"

echo "[TeachObs 2/10] private 30-lesson media plan and repository provenance"
if [ ! -d "$tsm_repository" ] || [ ! -f "$tsm_acquisition_receipt" ]; then
  echo "the pinned full repository and acquisition receipt are prerequisites" >&2
  echo "repository: $tsm_repository" >&2
  echo "receipt: $tsm_acquisition_receipt" >&2
  exit 2
fi
"$tsm_python" scripts/run_teachobs_media_preparation.py \
  --repository "$tsm_repository" \
  --output "$tsm_media_root" \
  --stage plan

run_caption_audit() {
  set -- "$tsm_python" -m teaching_skill_miner audit-teachobs-captions \
    --repository "$tsm_repository" \
    --media-plan "$tsm_media_plan" \
    --acquisition-receipt "$tsm_acquisition_receipt" \
    --output "$tsm_caption_root" \
    --public-receipt "$tsm_caption_receipt" \
    --jobs "$tsm_jobs" \
    --yt-dlp-python "$tsm_yt_dlp_python" \
    --js-runtime "$tsm_js_runtime" \
    --acknowledge-source-terms
  if [ -n "$tsm_cookies_from_browser" ]; then
    set -- "$@" --cookies-from-browser "$tsm_cookies_from_browser"
  fi
  if [ "$tsm_yt_dlp_direct" = true ]; then
    set -- "$@" --yt-dlp-direct
  fi
  if [ -n "$tsm_yt_dlp_impersonate" ]; then
    set -- "$@" --yt-dlp-impersonate "$tsm_yt_dlp_impersonate"
  fi
  if [ -n "$tsm_yt_dlp_youtube_client" ]; then
    set -- "$@" --yt-dlp-youtube-client "$tsm_yt_dlp_youtube_client"
  fi
  "$@"
}

echo "[TeachObs 3/10] caption timeline audit (valid partial coverage stays pending)"
tsm_caption_needs_run=true
if [ "$tsm_retry_captions" = false ] && \
   "$tsm_python" "$tsm_checker" captions \
     --audit "$tsm_caption_audit" \
     --receipt "$tsm_caption_receipt" >/dev/null 2>&1; then
  tsm_caption_needs_run=false
  echo "reusing hash-bound caption audit; use --retry-captions to retry pending lessons"
fi
if [ "$tsm_caption_needs_run" = true ]; then
  tsm_caption_before=missing
  if [ -f "$tsm_caption_audit" ]; then
    tsm_caption_before=$(cksum "$tsm_caption_audit")
  fi
  if run_caption_audit; then
    tsm_caption_exit=0
  else
    tsm_caption_exit=$?
  fi
  if [ "$tsm_caption_exit" -ne 0 ] && [ "$tsm_caption_exit" -ne 2 ]; then
    echo "caption audit failed operationally with status $tsm_caption_exit" >&2
    exit "$tsm_caption_exit"
  fi
  "$tsm_python" "$tsm_checker" captions \
    --audit "$tsm_caption_audit" \
    --receipt "$tsm_caption_receipt"
  if [ "$tsm_caption_exit" -eq 2 ]; then
    tsm_caption_after=$(cksum "$tsm_caption_audit")
    if [ "$tsm_caption_before" = "$tsm_caption_after" ]; then
      echo "caption command failed without producing a fresh audit" >&2
      exit 2
    fi
    echo "caption coverage remains pending; this is not WER or a formal audit" >&2
  fi
else
  "$tsm_python" "$tsm_checker" captions \
    --audit "$tsm_caption_audit" \
    --receipt "$tsm_caption_receipt"
fi
tsm_caption_formal=$("$tsm_python" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1], encoding="utf-8"))["aggregate"]["formal_caption_timeline_audit_completed"]).lower())' \
  "$tsm_caption_audit")

run_media_download() {
  set -- "$tsm_python" scripts/run_teachobs_media_preparation.py \
    --repository "$tsm_repository" \
    --output "$tsm_media_root" \
    --stage download \
    --jobs "$tsm_jobs" \
    --yt-dlp-python "$tsm_yt_dlp_python" \
    --js-runtime "$tsm_js_runtime" \
    --ffprobe "$tsm_ffprobe" \
    --acknowledge-source-terms
  if [ -n "$tsm_source_override_manifest" ]; then
    set -- "$@" \
      --source-override-manifest "$tsm_source_override_manifest" \
      --acknowledge-override-source-terms
  fi
  if [ -n "$tsm_cookies_from_browser" ]; then
    set -- "$@" --cookies-from-browser "$tsm_cookies_from_browser"
  fi
  if [ "$tsm_yt_dlp_direct" = true ]; then
    set -- "$@" --yt-dlp-direct
  fi
  if [ -n "$tsm_yt_dlp_impersonate" ]; then
    set -- "$@" --yt-dlp-impersonate "$tsm_yt_dlp_impersonate"
  fi
  "$@"
}

echo "[TeachObs 4/10] resumable complete-video download and exact profile gate"
if run_media_download; then
  tsm_media_exit=0
else
  tsm_media_exit=$?
fi
set -- "$tsm_python" "$tsm_checker" media \
  --plan "$tsm_media_plan" \
  --manifest "$tsm_media_manifest" \
  --evaluation-profile "$tsm_evaluation_profile"
if [ -n "$tsm_source_override_manifest" ]; then
  set -- "$@" --source-override-manifest "$tsm_source_override_manifest"
fi
"$@"
if [ "$tsm_media_exit" -ne 0 ]; then
  if [ "$tsm_evaluation_profile" = paper_track1_23_train_6_test ]; then
    echo "download command was nonzero, but the checker proved the unique expected S4 hole"
  else
    echo "media download failed with status $tsm_media_exit" >&2
    exit "$tsm_media_exit"
  fi
fi

echo "[TeachObs ASR] refresh or verify receipt against current captions and media"
tsm_asr_receipt_mode=none
# Private ASR evidence is monotonic: once a deeper artifact exists, never hide
# a stale/corrupt import by silently falling back to a shallower pending receipt.
# Broken symlinks count as existing evidence and are rejected by the checker.
if [ -e "$tsm_asr_import_audit" ] || [ -L "$tsm_asr_import_audit" ] || \
   [ -e "$tsm_asr_coverage_matrix" ] || [ -L "$tsm_asr_coverage_matrix" ]; then
  tsm_asr_receipt_mode=full
  echo "private ASR import evidence exists; requiring the full hash-bound gate"
  run_asr_receipt_check full
elif [ -e "$tsm_asr_job_manifest" ] || [ -L "$tsm_asr_job_manifest" ]; then
  tsm_asr_receipt_mode=job
  echo "private ASR job manifest exists; requiring its hash-bound pending gate"
  run_asr_receipt_check job
elif run_asr_receipt_check none >/dev/null 2>&1; then
  echo "reusing pending receipt bound to the current captions and media"
else
  echo "ASR receipt is missing or stale; preserving private files and refreshing pending evidence"
  "$tsm_python" -m teaching_skill_miner prepare-teachobs-asr-handoff \
    --media-manifest "$tsm_media_manifest" \
    --caption-audit "$tsm_caption_audit" \
    --public-receipt "$tsm_asr_receipt" \
    --pending-only
fi
run_asr_receipt_check "$tsm_asr_receipt_mode"

if [ "$tsm_asr_receipt_mode" != full ]; then
  echo "the paper benchmark requires six imported ASR results, not only a pending receipt or job manifest" >&2
  echo "complete the offline jobs and run import-teachobs-asr-results before resuming" >&2
  exit 2
fi

echo "[TeachObs 5/10] audited 4,945-scene transcript materialization"
if "$tsm_python" "$tsm_checker" transcripts \
     --manifest "$tsm_transcript_manifest" \
     --receipt "$tsm_transcript_receipt" \
     --evaluation-profile "$tsm_evaluation_profile" >/dev/null 2>&1; then
  echo "reusing fully revalidated official-caption/audited-ASR materialization"
else
  if [ -e "$tsm_transcript_root" ] || [ -L "$tsm_transcript_root" ] || \
     [ -e "$tsm_transcript_receipt" ] || [ -L "$tsm_transcript_receipt" ]; then
    echo "existing transcript materialization is incomplete, stale, or unbound; refusing to overwrite" >&2
    echo "preserve it and choose a new TSM_TEACHOBS_TRANSCRIPT_ROOT/public root" >&2
    "$tsm_python" "$tsm_checker" transcripts \
      --manifest "$tsm_transcript_manifest" \
      --receipt "$tsm_transcript_receipt" \
      --evaluation-profile "$tsm_evaluation_profile"
    exit 2
  fi
  "$tsm_python" -m teaching_skill_miner materialize-teachobs-transcripts \
    --media-plan "$tsm_media_plan" \
    --media-manifest "$tsm_media_manifest" \
    --caption-audit "$tsm_caption_audit" \
    --asr-import-audit "$tsm_asr_import_audit" \
    --coverage-matrix "$tsm_asr_coverage_matrix" \
    --asr-job-manifest "$tsm_asr_job_manifest" \
    --asr-results "$tsm_asr_results" \
    --output "$tsm_transcript_root" \
    --public-receipt "$tsm_transcript_receipt"
fi
"$tsm_python" "$tsm_checker" transcripts \
  --manifest "$tsm_transcript_manifest" \
  --receipt "$tsm_transcript_receipt" \
  --evaluation-profile "$tsm_evaluation_profile"

echo "[TeachObs 6/10] exact-profile audio/OCR/image-change/CLIP features"
if "$tsm_python" scripts/run_teachobs_media_preparation.py \
  --repository "$tsm_repository" \
  --output "$tsm_media_root" \
  --stage features \
  --feature-jobs "$tsm_feature_jobs" \
  --visual-jobs "$tsm_visual_jobs" \
  --ocr-jobs "$tsm_ocr_jobs" \
  --ffmpeg "$tsm_ffmpeg" \
  --ffprobe "$tsm_ffprobe" \
  --clip-model "$tsm_clip_model" \
  --clip-source-revision "$tsm_clip_revision" \
  --clip-device "$tsm_clip_device" \
  --clip-batch-size "$tsm_clip_batch_size" \
  --acknowledge-source-terms; then
  tsm_feature_exit=0
else
  tsm_feature_exit=$?
fi
"$tsm_python" "$tsm_checker" features \
  --manifest "$tsm_feature_manifest" \
  --media-manifest "$tsm_media_manifest" \
  --evaluation-profile "$tsm_evaluation_profile"
if [ "$tsm_feature_exit" -ne 0 ]; then
  if [ "$tsm_evaluation_profile" = paper_track1_23_train_6_test ]; then
    echo "feature command was nonzero, but the checker proved the unique expected S4 hole"
  else
    echo "feature extraction failed with status $tsm_feature_exit" >&2
    exit "$tsm_feature_exit"
  fi
fi

echo "[TeachObs 7/10] four-arm benchmark and deterministic private frozen bundle"
if "$tsm_python" "$tsm_checker" benchmark \
     --result "$tsm_benchmark_result" \
     --receipt "$tsm_benchmark_receipt" \
     --feature-manifest "$tsm_feature_manifest" \
     --transcript-materialization-manifest "$tsm_transcript_manifest" \
     --frozen-model-output "$tsm_frozen_model_root" \
     --evaluation-profile "$tsm_evaluation_profile" >/dev/null 2>&1; then
  echo "reusing hash-verified frozen bundle and bound exploratory result"
else
  if [ -e "$tsm_frozen_model_root" ] || [ -L "$tsm_frozen_model_root" ]; then
    echo "existing frozen output is incomplete, stale, or unbound; refusing to overwrite" >&2
    echo "choose a new TSM_TEACHOBS_FROZEN_MODEL_ROOT after preserving the old directory" >&2
    "$tsm_python" "$tsm_checker" benchmark \
      --result "$tsm_benchmark_result" \
      --receipt "$tsm_benchmark_receipt" \
      --feature-manifest "$tsm_feature_manifest" \
      --transcript-materialization-manifest "$tsm_transcript_manifest" \
      --frozen-model-output "$tsm_frozen_model_root" \
      --evaluation-profile "$tsm_evaluation_profile"
    exit 2
  fi
  "$tsm_python" -m teaching_skill_miner benchmark-teachobs-multimodal \
    --repository "$tsm_repository" \
    --feature-manifest "$tsm_feature_manifest" \
    --transcript-materialization-manifest "$tsm_transcript_manifest" \
    --output "$tsm_benchmark_result" \
    --public-receipt "$tsm_benchmark_receipt" \
    --frozen-model-output "$tsm_frozen_model_root" \
    --evaluation-profile "$tsm_evaluation_profile"
fi
"$tsm_python" "$tsm_checker" benchmark \
  --result "$tsm_benchmark_result" \
  --receipt "$tsm_benchmark_receipt" \
  --feature-manifest "$tsm_feature_manifest" \
  --transcript-materialization-manifest "$tsm_transcript_manifest" \
  --frozen-model-output "$tsm_frozen_model_root" \
  --evaluation-profile "$tsm_evaluation_profile"

echo "[TeachObs 8/10] blank pending two-rater assignment skeleton"
if [ -f "$tsm_human_manifest" ]; then
  "$tsm_python" "$tsm_checker" human-template-reusable \
    --manifest "$tsm_human_manifest" >/dev/null
fi
set -- "$tsm_python" -m teaching_skill_miner prepare-teachobs-double-annotation \
  --repository "$tsm_repository" \
  --media-root "$tsm_media_root" \
  --output "$tsm_human_root" \
  --public-receipt "$tsm_human_receipt" \
  --require-media
if [ "$tsm_evaluation_profile" = paper_track1_23_train_6_test ]; then
  for tsm_lesson_id in \
    S1 S2 S3 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 S17 S18 \
    S19 S20 S21 S22 S23 S24 S25 S26 S27 S28 S29 S30
  do
    set -- "$@" --lesson-id "$tsm_lesson_id"
  done
fi
"$@"
"$tsm_python" "$tsm_checker" human \
  --manifest "$tsm_human_manifest" \
  --receipt "$tsm_human_receipt" \
  --evaluation-profile "$tsm_evaluation_profile"

echo "[TeachObs 9/10] artifact-complete but externally unregistered lockbox draft"
"$tsm_python" -m teaching_skill_miner prepare-teachobs-lockbox-preregistration \
  --study-id "$tsm_lockbox_study_id" \
  --system-artifact "$tsm_frozen_model_root/bundle_manifest.json" \
  --analysis-code teaching_skill_miner/teachobs_lockbox.py \
  --arm-model "transcript_only=$tsm_frozen_model_root/transcript_only/manifest.json" \
  --arm-model "transcript_audio=$tsm_frozen_model_root/transcript_audio/manifest.json" \
  --arm-model "transcript_visual=$tsm_frozen_model_root/transcript_visual/manifest.json" \
  --arm-model "full=$tsm_frozen_model_root/full/manifest.json" \
  --expected-development-profile "$tsm_evaluation_profile" \
  --output "$tsm_lockbox_draft"
"$tsm_python" "$tsm_checker" lockbox \
  --draft "$tsm_lockbox_draft" \
  --study-id "$tsm_lockbox_study_id" \
  --evaluation-profile "$tsm_evaluation_profile" \
  --frozen-model-output "$tsm_frozen_model_root" \
  --analysis-code teaching_skill_miner/teachobs_lockbox.py

echo "[TeachObs 10/10] aggregate-only public release audit"
run_asr_receipt_check "$tsm_asr_receipt_mode"
"$tsm_python" "$tsm_checker" transcripts \
  --manifest "$tsm_transcript_manifest" \
  --receipt "$tsm_transcript_receipt" \
  --evaluation-profile "$tsm_evaluation_profile"
"$tsm_python" -m teaching_skill_miner release-audit "$tsm_public_root"

echo "local_pipeline_postconditions_passed=true"
echo "formal_caption_timeline_audit_completed=$tsm_caption_formal"
echo "evaluation_profile=$tsm_evaluation_profile"
if [ "$tsm_evaluation_profile" = paper_track1_23_train_6_test ]; then
  echo "result_scope=provisional_exploratory_public_TeachObs_published_six_lesson_intersection"
else
  echo "result_scope=provisional_exploratory_public_TeachObs_official_23_7_split"
fi
echo "human_completion=false"
echo "audited_transcript_materialization_complete=true"
echo "released_transcript_fallback_used=false"
echo "frozen_artifact_set_complete=true"
echo "confirmatory_multimodal_gain_established=false"
echo "external_lockbox_established=false"
echo "deployment_accuracy_established=false"
echo "learner_effectiveness_established=false"
