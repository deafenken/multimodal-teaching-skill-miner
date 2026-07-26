#!/usr/bin/env sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"
python3 -m teaching_skill_miner demo --output artifacts
if command -v ffmpeg >/dev/null 2>&1 \
  && command -v ffprobe >/dev/null 2>&1 \
  && command -v "${TSM_TESSERACT:-tesseract}" >/dev/null 2>&1; then
  sh scripts/run_multimodal_demo.sh
else
  echo "Skipping synthetic multimodal fixture: FFmpeg, FFprobe, and Tesseract are all required." >&2
fi
python3 -m unittest discover -s tests -v
