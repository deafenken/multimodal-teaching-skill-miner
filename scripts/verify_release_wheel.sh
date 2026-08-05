#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ] && [ "$#" -ne 3 ]; then
  echo "usage: $0 /path/to/teaching_skill_miner-VERSION-py3-none-any.whl [--receipt /path/to/receipt.json]" >&2
  exit 2
fi
receipt_path=
if [ "$#" -eq 3 ]; then
  if [ "$2" != "--receipt" ] || [ -z "$3" ]; then
    echo "usage: $0 /path/to/teaching_skill_miner-VERSION-py3-none-any.whl [--receipt /path/to/receipt.json]" >&2
    exit 2
  fi
  receipt_path=$3
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
python_command=${PYTHON:-python3}
wheel_dir=$(CDPATH= cd -- "$(dirname -- "$1")" && pwd)
wheel="$wheel_dir/$(basename -- "$1")"
if [ ! -f "$wheel" ]; then
  echo "release wheel not found: $wheel" >&2
  exit 2
fi
command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg is required for exact multimodal release verification" >&2
  exit 2
}
command -v ffprobe >/dev/null 2>&1 || {
  echo "ffprobe is required for exact multimodal release verification" >&2
  exit 2
}

verification_tmp=$(mktemp -d "${TMPDIR:-/tmp}/tsm-release-verify.XXXXXX")
trap 'rm -rf "$verification_tmp"' EXIT HUP INT TERM
expected_version=$(cd "$repo_root" && "$python_command" -c 'from teaching_skill_miner import __version__; print(__version__)')

cd "$repo_root"
"$python_command" -m teaching_skill_miner release-audit "$wheel" \
  --output "$verification_tmp/source-release-audit.json" >/dev/null
"$python_command" scripts/verify_wheel_allowlist.py "$wheel" \
  --repository-root "$repo_root" \
  --output "$verification_tmp/wheel-allowlist-audit.json" >/dev/null

"$python_command" -m venv "$verification_tmp/core-venv"
"$verification_tmp/core-venv/bin/python" -m pip install --quiet --no-deps "$wheel"
actual_version=$(
  cd "$verification_tmp"
  "$verification_tmp/core-venv/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("teaching-skill-miner"))'
)
if [ "$actual_version" != "$expected_version" ]; then
  echo "wheel version mismatch: source=$expected_version wheel=$actual_version" >&2
  exit 2
fi
version_output=$(cd "$verification_tmp" && "$verification_tmp/core-venv/bin/tsm" --version)
if [ "$version_output" != "tsm $expected_version" ]; then
  echo "CLI version mismatch: $version_output" >&2
  exit 2
fi
(
  cd "$verification_tmp"
  "$verification_tmp/core-venv/bin/tsm" doctor --output "$verification_tmp/core-doctor.json" >/dev/null
  "$verification_tmp/core-venv/bin/tsm" audit --output "$verification_tmp/core-audit.json" >/dev/null
  "$verification_tmp/core-venv/bin/tsm" fetch-formal-captions --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" fetch-full-videos --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" fetch-teachobs --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" benchmark-teachobs-text --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" audit-teachobs-captions --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" fetch-teachobs-captions --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" prepare-teachobs-asr-handoff --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" import-teachobs-asr-results --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" hash-teachobs-asr-model --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" run-teachobs-asr-gpu --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" materialize-teachobs-transcripts --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" dashboard --check \
    >"$verification_tmp/dashboard-check.json"
  "$verification_tmp/core-venv/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); assert value["passed"] is True; assert value["private_media_embedded"] is False; assert value["row_level_private_data_embedded"] is False' \
    "$verification_tmp/dashboard-check.json"
  "$verification_tmp/core-venv/bin/tsm" dashboard-real --check-template \
    >"$verification_tmp/private-dashboard-template-check.json"
  "$verification_tmp/core-venv/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); assert value["passed"] is True; assert value["contains_private_data"] is False; assert value["dashboard_kind"] == "local_private_real_evidence_template"' \
    "$verification_tmp/private-dashboard-template-check.json"
  "$verification_tmp/core-venv/bin/tsm" teacher-agent-dashboard --check \
    >"$verification_tmp/teacher-agent-dashboard-check.json"
  "$verification_tmp/core-venv/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); assert value["passed"] is True; assert value["real_time_skill_switching"] is True; assert value["real_learning_effectiveness_established"] is False' \
    "$verification_tmp/teacher-agent-dashboard-check.json"
  "$verification_tmp/core-venv/bin/tsm" teacher-agent-evaluate \
    --output "$verification_tmp/teacher-agent-evaluation.json" >/dev/null
  "$verification_tmp/core-venv/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); assert value["passed"] is True; assert value["aggregate"]["simulated_mean_gain_delta"] > 0; assert value["claim_boundary"]["real_learner_effectiveness_established"] is False' \
    "$verification_tmp/teacher-agent-evaluation.json"
  "$verification_tmp/core-venv/bin/tsm" teacher-agent-start --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" teacher-agent-step --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" teacher-agent-benchmark --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" teacher-agent-outcome-evaluate --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" teacher-agent-benchmark \
    --output "$verification_tmp/teacher-agent-benchmark-offline.json" >/dev/null
  "$verification_tmp/core-venv/bin/python" -m \
    teaching_skill_miner.teacher_agent_benchmark \
    --output "$verification_tmp/teacher-agent-benchmark-standalone.json" >/dev/null
  "$verification_tmp/core-venv/bin/tsm" teacher-agent-outcome-evaluate \
    --output "$verification_tmp/teacher-agent-learning-report.json" >/dev/null
  "$verification_tmp/core-venv/bin/tsm" teacher-agent-demo \
    --output-dir "$verification_tmp/teacher-agent-demo" >/dev/null
  "$verification_tmp/core-venv/bin/python" -c \
    'import json,sys; reports=[json.load(open(path, encoding="utf-8")) for path in sys.argv[1:3]]; outcome=json.load(open(sys.argv[3], encoding="utf-8")); demo=json.load(open(sys.argv[4], encoding="utf-8")); assert all(report["run_status"] == "baseline_only" for report in reports); assert all(report["claim_boundary"]["free_text_diagnostic_accuracy_established"] is False for report in reports); assert all(report["privacy"]["api_key_persisted_or_logged"] is False for report in reports); assert outcome["provenance"] == "author_constructed_demo_not_real"; assert outcome["claim_boundary"]["real_learner_effectiveness_established"] is False; assert demo["real_learning_effectiveness_established"] is False' \
    "$verification_tmp/teacher-agent-benchmark-offline.json" \
    "$verification_tmp/teacher-agent-benchmark-standalone.json" \
    "$verification_tmp/teacher-agent-learning-report.json" \
    "$verification_tmp/teacher-agent-demo/summary.json"
  asr_model_hash=$(
    "$verification_tmp/core-venv/bin/tsm" hash-teachobs-asr-model \
      --model-directory "$repo_root/configs"
  )
  "$verification_tmp/core-venv/bin/python" -c \
    'import re,sys; assert re.fullmatch(r"[0-9a-f]{64}", sys.argv[1])' \
    "$asr_model_hash"
  "$verification_tmp/core-venv/bin/tsm" prepare-teachobs-double-annotation --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" analyze-teachobs-double-annotation --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" prepare-teachobs-media --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" benchmark-teachobs-multimodal --help \
    >"$verification_tmp/teachobs-multimodal-help.txt"
  "$verification_tmp/core-venv/bin/python" -c \
    'from pathlib import Path; import sys; value=Path(sys.argv[1]).read_text(encoding="utf-8"); assert "full_23_train_7_test" in value; assert "paper_track1_23_train_6_test" in value; assert "--evaluation-profile" in value; assert "--transcript-materialization-manifest" in value' \
    "$verification_tmp/teachobs-multimodal-help.txt"
  "$verification_tmp/core-venv/bin/python" -c \
    'from teaching_skill_miner.teachobs_multimodal_benchmark import FULL_23_TRAIN_7_TEST_PROFILE, PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE, PAPER_TRACK1_EXPECTED_TEST_SCENE_COUNT, PAPER_TRACK1_TEST_LESSON_IDS, TEACHOBS_BENCHMARK_PROFILES; assert FULL_23_TRAIN_7_TEST_PROFILE in TEACHOBS_BENCHMARK_PROFILES; assert PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE in TEACHOBS_BENCHMARK_PROFILES; assert PAPER_TRACK1_EXPECTED_TEST_SCENE_COUNT == 1099; assert PAPER_TRACK1_TEST_LESSON_IDS == ("S2","S5","S19","S24","S28","S30")'
  "$verification_tmp/core-venv/bin/tsm" prepare-teachobs-lockbox-preregistration --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" prepare-learner-effect-study --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" analyze-learner-effect-study --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" multimodal-longform-dataset --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" multimodal-ablation --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" visual-semantic-extract --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" visual-semantic-dataset --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" visual-semantic-apply --help >/dev/null
  "$verification_tmp/core-venv/bin/tsm" demo --output "$verification_tmp/core-demo" >/dev/null
  "$verification_tmp/core-venv/bin/tsm" distill-general-skill \
    --skill-root "$verification_tmp/core-demo/skills" \
    --pattern '*.skill.json' \
    --output-dir "$verification_tmp/general-skill" \
    --example-concept "dynamic programming" >/dev/null
  "$verification_tmp/core-venv/bin/tsm" validate general-skill \
    "$verification_tmp/general-skill/general_skill.json" >/dev/null
  "$verification_tmp/core-venv/bin/tsm" evaluate-general-skill \
    --skill "$verification_tmp/general-skill/general_skill.json" \
    --output "$verification_tmp/general-skill/reevaluation.json" >/dev/null
  "$verification_tmp/core-venv/bin/tsm" apply-general-skill \
    --skill "$verification_tmp/general-skill/general_skill.json" \
    --concept "Newton's method" \
    --output "$verification_tmp/general-skill/newton-process.md" >/dev/null
  "$verification_tmp/core-venv/bin/python" -c \
    'import json, pathlib, sys; root=pathlib.Path(sys.argv[1]); skill=json.loads((root/"general_skill.json").read_text(encoding="utf-8")); report=json.loads((root/"reevaluation.json").read_text(encoding="utf-8")); assert skill["status"] == "heuristic_provisional"; assert report["passed"] is True; assert report["claim_boundary"]["internal_score_is_accuracy"] is False; assert "Newton'"'"'s method" in (root/"newton-process.md").read_text(encoding="utf-8")' \
    "$verification_tmp/general-skill"
  "$verification_tmp/core-venv/bin/tsm" release-audit "$wheel" \
    --output "$verification_tmp/wheel-self-audit.json" >/dev/null
  "$verification_tmp/core-venv/bin/tsm" pipeline \
    "$repo_root/data/demo/synthetic_lesson.mp4" \
    --transcript "$repo_root/data/demo/synthetic_multimodal_transcript.json" \
    --observations "$repo_root/data/demo/anonymized_observations.json" \
    --concept "recursive call stack" \
    --no-ocr \
    --frame-interval 2 \
    --max-frames 8 \
    --output "$verification_tmp/exact-wheel-captioned-video-demo" >/dev/null
  "$verification_tmp/core-venv/bin/python" -c 'import importlib.metadata as metadata; value=metadata.metadata("teaching-skill-miner"); assert value["Version"] == "'"$expected_version"'"; assert value["License-Expression"] == "LicenseRef-Academic-Evaluation-1.0"; assert value["Author"] == "Teaching Skill Miner contributors"; assert "visual" in value.get_all("Provides-Extra", [])'
  "$verification_tmp/core-venv/bin/python" -c 'from teaching_skill_miner.io_utils import resource_root; root=resource_root(); required=("CHANGELOG.md","PRIVACY.md","SECURITY.md","THIRD_PARTY_DATA.md"); missing=[name for name in required if not (root / "governance" / name).is_file()]; assert not missing, missing'
  "$verification_tmp/core-venv/bin/python" -c 'from teaching_skill_miner.visual_semantics import ONTOLOGY, SCHEMA_RESULT, SCHEMA_TASK; from teaching_skill_miner.visual_semantics_apply import execute_visual_semantic_apply; from teaching_skill_miner.visual_semantics_dataset import execute_visual_semantic_dataset; assert len(ONTOLOGY) == 8; assert SCHEMA_TASK.endswith(".v1"); assert SCHEMA_RESULT.endswith(".v1"); assert callable(execute_visual_semantic_apply); assert callable(execute_visual_semantic_dataset)'
)

"$python_command" -m venv --system-site-packages "$verification_tmp/recognition-venv"
"$verification_tmp/recognition-venv/bin/python" -m pip install --quiet --no-deps "$wheel"
(
  cd "$verification_tmp"
  "$verification_tmp/recognition-venv/bin/tsm" doctor \
    --output "$verification_tmp/recognition-doctor.json" >/dev/null
  "$verification_tmp/recognition-venv/bin/tsm" extract-strict-features --help >/dev/null
  "$verification_tmp/recognition-venv/bin/tsm" freeze-recognition-model --help >/dev/null
  "$verification_tmp/recognition-venv/bin/tsm" verify-external-research-evidence --help >/dev/null
  TSM_RELEASE_DOCTOR="$verification_tmp/recognition-doctor.json" \
    "$verification_tmp/recognition-venv/bin/python" -c 'import json, os; report=json.load(open(os.environ["TSM_RELEASE_DOCTOR"], encoding="utf-8")); required=("recognition_experiments_ready","signed_external_evaluation_ready","external_research_evidence_attestation_ready"); missing=[name for name in required if not report["capabilities"][name]]; assert not missing, missing'
)

if [ -n "$receipt_path" ]; then
  "$python_command" scripts/generate_release_acceptance.py record-wheel \
    --wheel "$wheel" \
    --release-audit "$verification_tmp/source-release-audit.json" \
    --allowlist-audit "$verification_tmp/wheel-allowlist-audit.json" \
    --repository-root "$repo_root" \
    --output "$receipt_path" >/dev/null
fi

"$python_command" -c 'import hashlib, pathlib, sys, zipfile; path=pathlib.Path(sys.argv[1]); payload=path.read_bytes(); print(f"verified_release_wheel={path}"); print(f"size_bytes={len(payload)}"); print(f"sha256={hashlib.sha256(payload).hexdigest()}"); print(f"member_count={len(zipfile.ZipFile(path).infolist())}")' "$wheel"
echo "Exact release wheel verification passed."
