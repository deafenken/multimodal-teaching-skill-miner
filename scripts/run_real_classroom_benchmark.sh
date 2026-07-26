#!/usr/bin/env sh
set -eu
umask 077

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

dataset_root="${1:-data/real/ouc_cge/extracted}"
output_dir="${2:-artifacts/real_classroom}"
mkdir -p "$output_dir"
chmod 700 "$output_dir"

python3 -m teaching_skill_miner real-data-audit \
  --dataset-root "$dataset_root" \
  --output "$output_dir/dataset_audit.json"

python3 -m teaching_skill_miner real-recognition-benchmark \
  --dataset-root "$dataset_root" \
  --output-dir "$output_dir" \
  --folds 4 \
  --frames 12 \
  --seed 2026

echo "Benchmark report: $output_dir/benchmark_report.json"
echo "The included public-sample result uses filename-index surrogate groups, not verified sessions or a deployment accuracy claim."
