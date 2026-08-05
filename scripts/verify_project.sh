#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

receipt_path=
if [ "$#" -ne 0 ] && [ "$#" -ne 2 ]; then
  echo "usage: $0 [--receipt /path/to/project-verification-receipt.json]" >&2
  exit 2
fi
if [ "$#" -eq 2 ]; then
  if [ "$1" != "--receipt" ] || [ -z "$2" ]; then
    echo "usage: $0 [--receipt /path/to/project-verification-receipt.json]" >&2
    exit 2
  fi
  receipt_path=$2
fi

python_command=${PYTHON:-python3}
verification_tmp=$(mktemp -d "${TMPDIR:-/tmp}/tsm-verify.XXXXXX")
trap 'rm -rf "$verification_tmp"' EXIT HUP INT TERM

"$python_command" -m pytest --junitxml="$verification_tmp/pytest.xml"
"$python_command" -m ruff check teaching_skill_miner tests scripts
"$python_command" -m compileall -q teaching_skill_miner scripts
sh -n scripts/*.sh
"$python_command" -m teaching_skill_miner fetch-full-videos --help >/dev/null
"$python_command" -m teaching_skill_miner audit-teachobs-captions --help >/dev/null
"$python_command" -m teaching_skill_miner fetch-teachobs-captions --help >/dev/null
"$python_command" -m teaching_skill_miner prepare-teachobs-asr-handoff --help >/dev/null
"$python_command" -m teaching_skill_miner import-teachobs-asr-results --help >/dev/null
"$python_command" -m teaching_skill_miner multimodal-longform-dataset --help >/dev/null
"$python_command" -m teaching_skill_miner multimodal-ablation --help >/dev/null
"$python_command" -m teaching_skill_miner visual-semantic-extract --help >/dev/null
"$python_command" -m teaching_skill_miner visual-semantic-dataset --help >/dev/null
"$python_command" -m teaching_skill_miner visual-semantic-apply --help >/dev/null
"$python_command" -m teaching_skill_miner dashboard --check >/dev/null
"$python_command" -m teaching_skill_miner teacher-agent-dashboard --check >/dev/null
"$python_command" -m teaching_skill_miner teacher-agent-start --help >/dev/null
"$python_command" -m teaching_skill_miner teacher-agent-step --help >/dev/null
"$python_command" -m teaching_skill_miner teacher-agent-benchmark --help >/dev/null
"$python_command" -m teaching_skill_miner teacher-agent-outcome-evaluate --help >/dev/null
"$python_command" -m teaching_skill_miner teacher-agent-evaluate \
  --output "$verification_tmp/teacher-agent-evaluation.json" >/dev/null
"$python_command" -m teaching_skill_miner teacher-agent-benchmark \
  --output "$verification_tmp/teacher-agent-benchmark-offline.json" >/dev/null
"$python_command" -m teaching_skill_miner teacher-agent-outcome-evaluate \
  --output "$verification_tmp/teacher-agent-learning-report.json" >/dev/null
"$python_command" -c \
  'import json,sys; benchmark=json.load(open(sys.argv[1], encoding="utf-8")); outcome=json.load(open(sys.argv[2], encoding="utf-8")); assert benchmark["run_status"] == "baseline_only"; assert benchmark["claim_boundary"]["free_text_diagnostic_accuracy_established"] is False; assert benchmark["privacy"]["api_key_persisted_or_logged"] is False; assert outcome["provenance"] == "author_constructed_demo_not_real"; assert outcome["claim_boundary"]["real_learner_effectiveness_established"] is False' \
  "$verification_tmp/teacher-agent-benchmark-offline.json" \
  "$verification_tmp/teacher-agent-learning-report.json"
"$python_command" -m teaching_skill_miner teacher-agent-demo \
  --output-dir "$verification_tmp/teacher-agent-demo" >/dev/null
for source_tool in \
  scripts/download_full_videos.py \
  scripts/run_full_multimodal_dataset.py \
  scripts/run_multimodal_ablation.py \
  scripts/extract_visual_semantics.py \
  scripts/run_visual_semantics_dataset.py \
  scripts/apply_visual_semantics.py \
  scripts/build_multimodal_public_receipts.py \
  scripts/run_teachobs_asr_gpu.py \
  scripts/generate_release_acceptance.py \
  scripts/verify_wheel_allowlist.py
do
  "$python_command" "$source_tool" --help >/dev/null
done
"$python_command" -m pip check
"$python_command" -m teaching_skill_miner doctor --output "$verification_tmp/doctor.json"
"$python_command" -m teaching_skill_miner demo --output "$verification_tmp/source-demo"
"$python_command" -m teaching_skill_miner verify-delivery \
  --output "$verification_tmp/delivery.json" \
  --markdown "$verification_tmp/delivery.md"
formal_caption_audit_run=false
if [ -f "$repo_root/artifacts/private/formal_captions/dataset_manifest.json" ]; then
  "$python_command" -m teaching_skill_miner audit \
    --manifest "$repo_root/artifacts/private/formal_captions/dataset_manifest.json" \
    --output "$verification_tmp/formal-caption-audit.json" \
    --require-formal >/dev/null
  "$python_command" -m teaching_skill_miner verify-delivery \
    --formal-manifest "$repo_root/artifacts/private/formal_captions/dataset_manifest.json" \
    --output "$verification_tmp/formal-delivery.json" \
    --markdown "$verification_tmp/formal-delivery.md" >/dev/null
  formal_caption_audit_run=true
else
  echo "Private formal-caption dataset not present; run fetch-formal-captions to verify live evidence."
fi

teachobs_private_receipt_audit_run=false
teachobs_governance_binding_ready=false
teachobs_media_manifest="$repo_root/artifacts/private/external_datasets/teachobs/media/media_manifest.json"
teachobs_caption_audit="$repo_root/artifacts/private/external_datasets/teachobs/captions/caption_audit.json"
teachobs_asr_root="$repo_root/artifacts/private/external_datasets/teachobs/asr"
teachobs_asr_receipt="$repo_root/artifacts/public/teachobs_asr_receipt.json"
teachobs_evaluation_profile=paper_track1_23_train_6_test
teachobs_transcript_manifest="$repo_root/artifacts/private/external_datasets/teachobs/materialized_transcripts/manifest.json"
teachobs_transcript_receipt="$repo_root/artifacts/public/teachobs_transcript_materialization_receipt.json"
teachobs_human_root=${TSM_TEACHOBS_HUMAN_ROOT:-"$repo_root/artifacts/private/external_datasets/teachobs/human_annotation"}
teachobs_human_manifest="$teachobs_human_root/assignment_manifest.json"
teachobs_human_receipt=${TSM_TEACHOBS_HUMAN_RECEIPT:-"$repo_root/artifacts/public/teachobs_human_annotation_receipt.json"}
teachobs_lockbox_draft=${TSM_TEACHOBS_LOCKBOX_DRAFT:-"$repo_root/artifacts/public/teachobs_new_site_lockbox_preregistration_draft.json"}
teachobs_lockbox_analysis="$repo_root/teaching_skill_miner/teachobs_lockbox.py"
if [ -f "$teachobs_media_manifest" ] && [ -f "$teachobs_caption_audit" ]; then
  teachobs_annotation_audit="$repo_root/artifacts/private/external_datasets/teachobs/imported_annotations/dataset_audit.json"
  teachobs_annotation_receipt="$repo_root/artifacts/public/teachobs_annotation_receipt.json"
  teachobs_caption_receipt="$repo_root/artifacts/public/teachobs_caption_receipt.json"
  if [ ! -f "$teachobs_annotation_audit" ] || \
     [ ! -f "$teachobs_annotation_receipt" ] || \
     [ ! -f "$teachobs_caption_receipt" ]; then
    echo "TeachObs private inputs are present but an annotation/caption receipt input is missing." >&2
    exit 2
  fi
  "$python_command" scripts/check_teachobs_external_study.py annotations \
    --audit "$teachobs_annotation_audit" \
    --receipt "$teachobs_annotation_receipt" >/dev/null
  "$python_command" scripts/check_teachobs_external_study.py captions \
    --audit "$teachobs_caption_audit" \
    --receipt "$teachobs_caption_receipt" >/dev/null
  if [ ! -f "$teachobs_asr_receipt" ]; then
    echo "TeachObs private inputs are present but the public ASR receipt is missing." >&2
    exit 2
  fi
  if [ -f "$teachobs_asr_root/job_manifest.json" ] && \
     [ -f "$teachobs_asr_root/import_audit.json" ] && \
     [ -f "$teachobs_asr_root/transcript_coverage_matrix.json" ] && \
     "$python_command" scripts/check_teachobs_external_study.py asr \
       --receipt "$teachobs_asr_receipt" \
       --media-manifest "$teachobs_media_manifest" \
       --caption-audit "$teachobs_caption_audit" \
       --job-manifest "$teachobs_asr_root/job_manifest.json" \
       --import-audit "$teachobs_asr_root/import_audit.json" \
       --coverage-matrix "$teachobs_asr_root/transcript_coverage_matrix.json" \
       --evaluation-profile "$teachobs_evaluation_profile" \
       >/dev/null 2>&1; then
    teachobs_asr_receipt_mode=completed_import
  elif [ -f "$teachobs_asr_root/job_manifest.json" ] && \
       "$python_command" scripts/check_teachobs_external_study.py asr \
         --receipt "$teachobs_asr_receipt" \
         --media-manifest "$teachobs_media_manifest" \
         --caption-audit "$teachobs_caption_audit" \
         --job-manifest "$teachobs_asr_root/job_manifest.json" \
         --evaluation-profile "$teachobs_evaluation_profile" \
         >/dev/null 2>&1; then
    teachobs_asr_receipt_mode=pending_job
  else
    "$python_command" scripts/check_teachobs_external_study.py asr \
      --receipt "$teachobs_asr_receipt" \
      --media-manifest "$teachobs_media_manifest" \
      --caption-audit "$teachobs_caption_audit" \
      --evaluation-profile "$teachobs_evaluation_profile" >/dev/null
    teachobs_asr_receipt_mode=pending_without_job
  fi
  teachobs_transcript_materialization_status=absent
  if [ -f "$teachobs_transcript_manifest" ] || \
     [ -f "$teachobs_transcript_receipt" ]; then
    if [ ! -f "$teachobs_transcript_manifest" ] || \
       [ ! -f "$teachobs_transcript_receipt" ]; then
      echo "TeachObs transcript materialization evidence is partial; private manifest and public receipt are both required." >&2
      exit 2
    fi
    "$python_command" scripts/check_teachobs_external_study.py transcripts \
      --manifest "$teachobs_transcript_manifest" \
      --receipt "$teachobs_transcript_receipt" \
      --evaluation-profile "$teachobs_evaluation_profile" >/dev/null
    teachobs_transcript_materialization_status=complete
  elif [ "$teachobs_asr_receipt_mode" = completed_import ]; then
    echo "TeachObs ASR import is complete but the required transcript materialization is missing." >&2
    exit 2
  fi
  teachobs_benchmark_result="$repo_root/artifacts/private/external_datasets/teachobs/multimodal_benchmark_result.json"
  teachobs_benchmark_receipt="$repo_root/artifacts/public/teachobs_multimodal_benchmark_receipt.json"
  teachobs_frozen_root=${TSM_TEACHOBS_FROZEN_MODEL_ROOT:-"$repo_root/artifacts/private/external_datasets/teachobs/frozen_models"}
  teachobs_benchmark_required=false
  if [ "$teachobs_asr_receipt_mode" = completed_import ] && \
     [ "$teachobs_transcript_materialization_status" = complete ]; then
    teachobs_benchmark_required=true
  fi
  teachobs_benchmark_evidence_present=false
  if [ -f "$teachobs_benchmark_result" ] || \
     [ -f "$teachobs_benchmark_receipt" ] || \
     [ -e "$teachobs_frozen_root" ] || \
     [ -L "$teachobs_frozen_root" ]; then
    teachobs_benchmark_evidence_present=true
  fi
  teachobs_benchmark_status=absent
  if [ "$teachobs_benchmark_required" = true ] || \
     [ "$teachobs_benchmark_evidence_present" = true ]; then
    if [ ! -f "$teachobs_benchmark_result" ] || \
       [ ! -f "$teachobs_benchmark_receipt" ] || \
       [ ! -f "$teachobs_frozen_root/bundle_manifest.json" ] || \
       [ "$teachobs_transcript_materialization_status" != complete ]; then
      echo "Completed TeachObs ASR/materialization requires a current four-arm result, public receipt, complete frozen bundle, and audited transcript materialization." >&2
      echo "Preserve partial outputs and rerun the benchmark with a fresh TSM_TEACHOBS_FROZEN_MODEL_ROOT when necessary." >&2
      exit 2
    fi
    teachobs_benchmark_profile=$("$python_command" -c \
      'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); profile=value.get("dataset_audit", {}).get("benchmark_profile"); allowed={"full_23_train_7_test","paper_track1_23_train_6_test"}; raise SystemExit(print(profile) if profile in allowed else 2)' \
      "$teachobs_benchmark_result")
    "$python_command" scripts/check_teachobs_external_study.py benchmark \
      --result "$teachobs_benchmark_result" \
      --receipt "$teachobs_benchmark_receipt" \
      --feature-manifest "$repo_root/artifacts/private/external_datasets/teachobs/media/feature_manifest.json" \
      --transcript-materialization-manifest "$teachobs_transcript_manifest" \
      --frozen-model-output "$teachobs_frozen_root" \
      --evaluation-profile "$teachobs_benchmark_profile" >/dev/null
    teachobs_benchmark_status=complete
    echo "TeachObs benchmark receipt and frozen four-arm bundle are current: $teachobs_benchmark_profile"
  fi
  teachobs_human_artifact_present=false
  teachobs_human_status=absent
  if [ -e "$teachobs_human_root" ] || \
     [ -L "$teachobs_human_root" ] || \
     [ -f "$teachobs_human_receipt" ]; then
    teachobs_human_artifact_present=true
  fi
  if [ "$teachobs_human_artifact_present" = true ]; then
    if [ ! -f "$teachobs_human_manifest" ] || \
       [ ! -f "$teachobs_human_receipt" ]; then
      echo "TeachObs human-review evidence is partial; manifest and public receipt are both required." >&2
      echo "Preserve prior work and select a fresh TSM_TEACHOBS_HUMAN_ROOT rather than overwriting it." >&2
      exit 2
    fi
    if ! "$python_command" scripts/check_teachobs_external_study.py human \
      --manifest "$teachobs_human_manifest" \
      --receipt "$teachobs_human_receipt" \
      --evaluation-profile "$teachobs_evaluation_profile" >/dev/null; then
      echo "TeachObs human-review evidence is stale or uses another profile." >&2
      echo "Preserve prior work and regenerate the blank package under a fresh TSM_TEACHOBS_HUMAN_ROOT." >&2
      exit 2
    fi
    teachobs_human_status=complete
    echo "TeachObs pending human-review skeleton is current for: $teachobs_evaluation_profile"
  fi
  if [ "$teachobs_benchmark_required" = true ] && \
     [ "$teachobs_human_status" != complete ]; then
    echo "Completed TeachObs benchmark evidence requires a current paper-profile human-review skeleton and receipt." >&2
    echo "Generate them under a fresh TSM_TEACHOBS_HUMAN_ROOT and set TSM_TEACHOBS_HUMAN_RECEIPT." >&2
    exit 2
  fi
  teachobs_lockbox_status=absent
  if [ -e "$teachobs_lockbox_draft" ] || [ -L "$teachobs_lockbox_draft" ]; then
    if [ ! -f "$teachobs_lockbox_draft" ]; then
      echo "TeachObs lockbox draft path is not a regular file." >&2
      exit 2
    fi
    if [ "$teachobs_benchmark_status" != complete ]; then
      echo "An existing TeachObs lockbox draft requires the current frozen four-arm benchmark bundle." >&2
      exit 2
    fi
    teachobs_lockbox_study_id=$(
      "$python_command" -c \
        'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); study_id=value.get("study_id"); raise SystemExit(print(study_id) if isinstance(study_id, str) and study_id else 2)' \
        "$teachobs_lockbox_draft"
    )
    if ! "$python_command" scripts/check_teachobs_external_study.py lockbox \
      --draft "$teachobs_lockbox_draft" \
      --study-id "$teachobs_lockbox_study_id" \
      --evaluation-profile "$teachobs_evaluation_profile" \
      --frozen-model-output "$teachobs_frozen_root" \
      --analysis-code "$teachobs_lockbox_analysis" >/dev/null; then
      echo "TeachObs lockbox draft is stale or not transitively bound to the current frozen bundle." >&2
      echo "Regenerate the pending draft after preserving the previous artifact." >&2
      exit 2
    fi
    teachobs_lockbox_status=complete
    echo "TeachObs pending lockbox draft is current and transitively bundle-bound."
  fi
  if [ "$teachobs_benchmark_required" = true ] && \
     [ "$teachobs_lockbox_status" != complete ]; then
    echo "Completed TeachObs benchmark evidence requires a current transitively bundle-bound lockbox draft." >&2
    echo "Generate it at TSM_TEACHOBS_LOCKBOX_DRAFT after freezing all four arms." >&2
    exit 2
  fi
  if [ "$teachobs_human_status" = complete ] && \
     [ "$teachobs_lockbox_status" = complete ]; then
    teachobs_governance_binding_ready=true
  fi
  teachobs_private_receipt_audit_run=true
  echo "TeachObs annotation/caption/ASR receipts are current and private-input-bound: $teachobs_asr_receipt_mode"
  echo "TeachObs transcript materialization status: $teachobs_transcript_materialization_status"
else
  echo "TeachObs private media/caption inputs not both present; semantic ASR receipt audit skipped."
fi
"$python_command" -m teaching_skill_miner release-audit artifacts/public \
  --output "$verification_tmp/public-release-audit.json" >/dev/null

tracked_file_privacy_scan_run=false
if [ -e "$repo_root/.git" ]; then
  "$python_command" scripts/audit_repository_privacy.py
  tracked_file_privacy_scan_run=true
else
  echo "Tracked-file privacy audit skipped: this directory has no .git metadata."
fi

PYTHON="$python_command" sh scripts/build_release_wheel.sh "$verification_tmp/dist-a"
PYTHON="$python_command" sh scripts/build_release_wheel.sh "$verification_tmp/dist-b"
set -- "$verification_tmp"/dist-a/*.whl
wheel=$1
set -- "$verification_tmp"/dist-b/*.whl
comparison_wheel=$1
cmp "$wheel" "$comparison_wheel"
PYTHON="$python_command" sh scripts/verify_release_wheel.sh "$wheel" \
  --receipt "$verification_tmp/exact-wheel-verification.json"

TSM_VERIFICATION_TMP="$verification_tmp" "$python_command" - <<'PY'
import json
import os
from pathlib import Path
import jsonschema

root = Path.cwd()
schema = json.loads((root / "schema/teaching_skill.schema.json").read_text(encoding="utf-8"))
generated = Path(os.environ["TSM_VERIFICATION_TMP"]) / "source-demo/skills"
for path in sorted(generated.glob("*.json")):
    jsonschema.Draft202012Validator(schema).validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
PY

if [ -n "$receipt_path" ]; then
  set -- record-project \
    --junit-xml "$verification_tmp/pytest.xml" \
    --first-wheel "$wheel" \
    --second-wheel "$comparison_wheel" \
    --exact-wheel-receipt "$verification_tmp/exact-wheel-verification.json" \
    --public-directory "$repo_root/artifacts/public" \
    --public-release-audit "$verification_tmp/public-release-audit.json" \
    --repository-root "$repo_root" \
    --output "$receipt_path"
  if [ "$formal_caption_audit_run" = true ]; then
    set -- "$@" --formal-caption-audit-run
  fi
  if [ "$teachobs_private_receipt_audit_run" = true ]; then
    set -- "$@" --teachobs-private-receipt-audit-run
  fi
  if [ "$teachobs_governance_binding_ready" = true ]; then
    set -- "$@" \
      --teachobs-human-manifest "$teachobs_human_manifest" \
      --teachobs-human-receipt "$teachobs_human_receipt" \
      --teachobs-lockbox-draft "$teachobs_lockbox_draft"
  fi
  if [ "$tracked_file_privacy_scan_run" = true ]; then
    set -- "$@" --tracked-file-privacy-scan-run
  fi
  "$python_command" scripts/generate_release_acceptance.py "$@" >/dev/null
fi

echo "Project verification passed: tests, lint, syntax, dependencies, new multimodal entrypoint smoke tests, public audit, reproducible clean builds, and exact-wheel verification."
