#!/bin/sh
set -eu

usage() {
  echo "usage: $0 MODEL_DIR --acknowledge-source-terms [--reuse-downloads]" >&2
  echo "" >&2
  echo "Runs the complete private 10-video multimodal study and public aggregate receipts." >&2
  echo "MODEL_DIR must be a local openai/clip-vit-base-patch32 snapshot." >&2
}

if [ "$#" -eq 1 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  usage
  exit 0
fi
if [ "$#" -lt 2 ]; then
  usage
  exit 2
fi

tsm_model_dir=$1
shift
tsm_acknowledged=false
tsm_reuse_downloads=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --acknowledge-source-terms)
      tsm_acknowledged=true
      ;;
    --reuse-downloads)
      tsm_reuse_downloads=true
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
  shift
done

if [ "$tsm_acknowledged" != true ]; then
  echo "refusing to download or process third-party course media without --acknowledge-source-terms" >&2
  exit 2
fi
if [ ! -f "$tsm_model_dir/model.safetensors" ] && [ ! -f "$tsm_model_dir/pytorch_model.bin" ]; then
  echo "no CLIP model weight file found in: $tsm_model_dir" >&2
  exit 2
fi

tsm_python=${TSM_PYTHON:-python3}
tsm_source_manifest=${TSM_SOURCE_MANIFEST:-data/formal_caption_sources.json}
tsm_caption_dir=${TSM_CAPTION_DIR:-artifacts/private/formal_captions}
tsm_video_dir=${TSM_VIDEO_DIR:-artifacts/private/full_videos}
tsm_multimodal_dir=${TSM_MULTIMODAL_DIR:-artifacts/private/full_multimodal}
tsm_semantic_dir=$tsm_multimodal_dir/semantic_results
tsm_clip_revision=${TSM_CLIP_REVISION:-3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268}

if [ "$tsm_reuse_downloads" != true ]; then
  "$tsm_python" -m teaching_skill_miner fetch-formal-captions \
    --source-manifest "$tsm_source_manifest" \
    --output "$tsm_caption_dir" \
    --public-receipt artifacts/public/formal_caption_retrieval_receipt.json \
    --acknowledge-source-terms

  "$tsm_python" -m teaching_skill_miner fetch-full-videos \
    --source-manifest "$tsm_source_manifest" \
    --formal-caption-manifest "$tsm_caption_dir/dataset_manifest.json" \
    --output "$tsm_video_dir" \
    --public-receipt artifacts/public/full_video_validation_receipt.json \
    --acknowledge-source-terms
else
  test -f "$tsm_caption_dir/dataset_manifest.json"
  test -f "$tsm_video_dir/media_manifest.json"
fi

"$tsm_python" -m teaching_skill_miner multimodal-longform-dataset \
  --media-manifest "$tsm_video_dir/media_manifest.json" \
  --transcript-manifest "$tsm_caption_dir/dataset_manifest.json" \
  --output "$tsm_multimodal_dir" \
  --chunk-seconds 300 \
  --overlap-seconds 2 \
  --frame-interval 15 \
  --scene-threshold 0.32 \
  --max-scenes-per-chunk 12 \
  --ocr-workers 4

"$tsm_python" -m teaching_skill_miner visual-semantic-dataset \
  --manifest "$tsm_multimodal_dir/dataset_manifest.json" \
  --model "$tsm_model_dir" \
  --output-dir "$tsm_semantic_dir" \
  --source-model-id openai/clip-vit-base-patch32 \
  --source-revision "$tsm_clip_revision" \
  --device "${TSM_VISUAL_DEVICE:-cpu}" \
  --batch-size "${TSM_VISUAL_BATCH_SIZE:-16}"

"$tsm_python" -m teaching_skill_miner visual-semantic-apply \
  --manifest "$tsm_multimodal_dir/dataset_manifest.json" \
  --semantic-results-dir "$tsm_semantic_dir" \
  --output-manifest "$tsm_multimodal_dir/dataset_manifest.semantic.json"

"$tsm_python" -m teaching_skill_miner audit \
  --manifest "$tsm_multimodal_dir/dataset_manifest.semantic.json" \
  --output "$tsm_multimodal_dir/data_audit.semantic.json" \
  --require-formal

"$tsm_python" -m teaching_skill_miner multimodal-ablation \
  --manifest "$tsm_multimodal_dir/dataset_manifest.semantic.json" \
  --output "$tsm_multimodal_dir/ablation"

"$tsm_python" scripts/build_multimodal_public_receipts.py \
  --manifest "$tsm_multimodal_dir/dataset_manifest.semantic.json" \
  --audit "$tsm_multimodal_dir/data_audit.semantic.json" \
  --semantic-batch "$tsm_semantic_dir/semantic_batch_receipt.json" \
  --ablation-report "$tsm_multimodal_dir/ablation/ablation_report.json" \
  --validation-output artifacts/public/full_multimodal_validation_receipt.json \
  --ablation-output artifacts/public/multimodal_ablation_receipt.json

"$tsm_python" -m teaching_skill_miner release-audit artifacts/public

echo "complete: $tsm_multimodal_dir/dataset_manifest.semantic.json"
echo "public receipts: artifacts/public/full_multimodal_validation_receipt.json and artifacts/public/multimodal_ablation_receipt.json"
