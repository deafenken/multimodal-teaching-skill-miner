#!/usr/bin/env sh
set -eu
umask 077

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

if [ "$#" -lt 6 ]; then
  echo "usage: $0 VIDEO OUTPUT_DIR VIDEO_ID COURSE_ID TITLE SOURCE_URL [CONCEPT] [TRANSCRIPT]" >&2
  exit 64
fi

video=$1
output_dir=$2
video_id=$3
course_id=$4
title=$5
source_url=$6
concept=${7:-课堂核心概念}
transcript=${8:-}

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

python3 -m teaching_skill_miner doctor --output "$output_dir/doctor.json"
if [ -z "$transcript" ]; then
  TSM_DEFENSE_DOCTOR="$output_dir/doctor.json" python3 -c 'import json, os; report=json.load(open(os.environ["TSM_DEFENSE_DOCTOR"], encoding="utf-8")); raise SystemExit(0 if report["capabilities"]["media_transcription_ready"] else "real-video ASR requires an OpenAI-Whisper-compatible whisper CLI; inspect doctor.json or pass an official/audited transcript as argument 8")'
fi
if [ -n "$transcript" ]; then
  python3 -m teaching_skill_miner pipeline "$video" \
    --video-id "$video_id" \
    --course-id "$course_id" \
    --title "$title" \
    --source-url "$source_url" \
    --transcript "$transcript" \
    --concept "$concept" \
    --output "$output_dir" \
    $ocr_flag
else
  python3 -m teaching_skill_miner pipeline "$video" \
    --video-id "$video_id" \
    --course-id "$course_id" \
    --title "$title" \
    --source-url "$source_url" \
    --concept "$concept" \
    --output "$output_dir" \
    $ocr_flag
fi

echo "Defense artifacts: $output_dir"
echo "The interactive session is scripted for deterministic demonstration; it is not learner-effectiveness evidence."
