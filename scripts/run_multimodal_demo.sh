#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

mkdir -p data/demo artifacts/multimodal_demo

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg is required" >&2
  exit 1
}
command -v ffprobe >/dev/null 2>&1 || {
  echo "ffprobe is required" >&2
  exit 1
}
ocr_flag=
if ! command -v "${TSM_TESSERACT:-tesseract}" >/dev/null 2>&1; then
  ocr_flag=--no-ocr
fi

if command -v rsvg-convert >/dev/null 2>&1; then
  rsvg-convert -w 1280 -h 720 data/demo/slide_example.svg -o data/demo/slide_example.png
  rsvg-convert -w 1280 -h 720 data/demo/slide_code.svg -o data/demo/slide_code.png
fi

if [ ! -f data/demo/slide_example.png ] || [ ! -f data/demo/slide_code.png ]; then
  echo "missing demo PNG slides; install rsvg-convert or provide the pre-rendered assets" >&2
  exit 1
fi

ffmpeg -hide_banner -loglevel error -y \
  -loop 1 -framerate 10 -t 3.5 -i data/demo/slide_example.png \
  -loop 1 -framerate 10 -t 4.5 -i data/demo/slide_code.png \
  -f lavfi -i "sine=frequency=440:sample_rate=16000:duration=1.5" \
  -f lavfi -i "anullsrc=r=16000:cl=mono:d=2.0" \
  -f lavfi -i "sine=frequency=660:sample_rate=16000:duration=4.5" \
  -filter_complex "[0:v][1:v]concat=n=2:v=1:a=0[v];[2:a][3:a][4:a]concat=n=3:v=0:a=1[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \
  data/demo/synthetic_lesson.mp4

python3 -m teaching_skill_miner multimodal \
  --video data/demo/synthetic_lesson.mp4 \
  --transcript data/demo/synthetic_multimodal_transcript.json \
  --observations data/demo/anonymized_observations.json \
  --artifacts-dir artifacts/multimodal_demo/analysis \
  --frame-interval 2 \
  --max-frames 16 \
  --output artifacts/multimodal_demo/enriched_transcript.json \
  $ocr_flag

python3 -m teaching_skill_miner mine \
  --transcript artifacts/multimodal_demo/enriched_transcript.json \
  --output artifacts/multimodal_demo/skill.json

python3 -m teaching_skill_miner evaluate \
  --skill artifacts/multimodal_demo/skill.json \
  --transcript artifacts/multimodal_demo/enriched_transcript.json \
  --output artifacts/multimodal_demo/evaluation.json

python3 -m teaching_skill_miner multimodal-benchmark \
  --transcript artifacts/multimodal_demo/enriched_transcript.json \
  --ground-truth data/demo/synthetic_multimodal_ground_truth.json \
  --output artifacts/multimodal_demo/fixture_benchmark.json

python3 -m teaching_skill_miner teach \
  --skill artifacts/multimodal_demo/skill.json \
  --concept "递归调用栈" \
  --output artifacts/multimodal_demo/teaching_process.md

python3 -m teaching_skill_miner interact \
  --skill artifacts/multimodal_demo/skill.json \
  --concept "递归调用栈" \
  --script data/demo/scripted_learner_responses.json \
  --output artifacts/multimodal_demo/interactive_session.json

python3 -m teaching_skill_miner pipeline data/demo/synthetic_lesson.mp4 \
  --transcript data/demo/synthetic_multimodal_transcript.json \
  --observations data/demo/anonymized_observations.json \
  --concept "递归调用栈" \
  --frame-interval 2 \
  --max-frames 16 \
  --output artifacts/captioned_video_pipeline_demo \
  $ocr_flag

echo "One-command captioned-video pipeline artifacts: artifacts/captioned_video_pipeline_demo"
echo "The supplied transcript is not treated as audio-verified speech unless its ASR provenance hashes the same media."
