#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
python_command=${PYTHON:-python3}
output_dir=${1:-"$repo_root/dist"}
SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1784592000}
export SOURCE_DATE_EPOCH

mkdir -p "$output_dir"
output_dir=$(CDPATH= cd -- "$output_dir" && pwd)
build_tmp=$(mktemp -d "${TMPDIR:-/tmp}/tsm-clean-build.XXXXXX")
trap 'rm -rf "$build_tmp"' EXIT HUP INT TERM
source_dir="$build_tmp/source"

copy_release_file() {
  relative_path=$1
  destination="$source_dir/$relative_path"
  mkdir -p "$(dirname -- "$destination")"
  cp "$repo_root/$relative_path" "$destination"
}

"$python_command" "$repo_root/scripts/verify_wheel_allowlist.py" \
  --repository-root "$repo_root" \
  --check-public-json-source-only \
  --quiet

for relative_path in \
  pyproject.toml \
  README.md \
  LICENSE \
  CHANGELOG.md \
  PRIVACY.md \
  SECURITY.md \
  THIRD_PARTY_DATA.md \
  data/dataset_manifest.json \
  data/evaluation_cases.json \
  data/formal_caption_sources.json \
  data/transcripts/linear_algebra_l01.json \
  data/transcripts/linear_algebra_l02.json \
  data/transcripts/linear_algebra_l03.json \
  data/transcripts/linear_algebra_l04.json \
  data/transcripts/linear_algebra_l05.json \
  data/transcripts/python_l01.json \
  data/transcripts/python_l02.json \
  data/transcripts/python_l03.json \
  data/transcripts/python_l04.json \
  data/transcripts/python_l05.json \
  teaching_skill_miner/web/index.html
do
  copy_release_file "$relative_path"
done

find "$repo_root/teaching_skill_miner" -type f -name '*.py' -print |
while IFS= read -r source_path; do
  relative_path=${source_path#"$repo_root/"}
  copy_release_file "$relative_path"
done

while IFS= read -r relative_path; do
  copy_release_file "$relative_path"
done < "$repo_root/release/public_json_resources.txt"

(
  cd "$source_dir"
  "$python_command" -m build --wheel --no-isolation --outdir "$output_dir"
)
