#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

if [ "$#" -gt 2 ]; then
  echo "usage: $0 [--output artifacts/release_acceptance_VERSION.json]" >&2
  exit 2
fi

python_command=${PYTHON:-python3}
release_version=$(
  "$python_command" -c 'from teaching_skill_miner import __version__; print(__version__)'
)
output_path="$repo_root/artifacts/release_acceptance_${release_version}.json"
if [ "$#" -eq 2 ]; then
  if [ "$1" != "--output" ] || [ -z "$2" ]; then
    echo "usage: $0 [--output artifacts/release_acceptance_VERSION.json]" >&2
    exit 2
  fi
  output_path=$2
elif [ "$#" -eq 1 ]; then
  echo "usage: $0 [--output artifacts/release_acceptance_VERSION.json]" >&2
  exit 2
fi

if [ ! -d "$(dirname -- "$output_path")" ]; then
  echo "acceptance output parent does not exist: $(dirname -- "$output_path")" >&2
  exit 2
fi

release_tmp=$(mktemp -d "${TMPDIR:-/tmp}/tsm-final-release.XXXXXX")
staged_wheel=
staged_acceptance=
trap 'rm -rf "$release_tmp"; if [ -n "$staged_wheel" ]; then rm -f "$staged_wheel"; fi; if [ -n "$staged_acceptance" ]; then rm -f "$staged_acceptance"; fi' EXIT HUP INT TERM

"$python_command" scripts/generate_release_acceptance.py pending \
  --release-version "$release_version" \
  --output "$output_path" >/dev/null

PYTHON="$python_command" sh scripts/verify_project.sh \
  --receipt "$release_tmp/project-verification.json"

mkdir -p "$release_tmp/dist"
if [ -L "$repo_root/dist" ] || { [ -e "$repo_root/dist" ] && [ ! -d "$repo_root/dist" ]; }; then
  echo "release dist path must be a real directory: $repo_root/dist" >&2
  exit 2
fi
mkdir -p "$repo_root/dist"
PYTHON="$python_command" sh scripts/build_release_wheel.sh "$release_tmp/dist"
set -- "$release_tmp"/dist/*.whl
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
  echo "clean release build did not produce exactly one wheel" >&2
  exit 2
fi
candidate_wheel=$1

PYTHON="$python_command" sh scripts/verify_release_wheel.sh "$candidate_wheel" \
  --receipt "$release_tmp/exact-wheel-verification.json"

"$python_command" -m teaching_skill_miner release-audit "$candidate_wheel" \
  --output "$release_tmp/wheel-release-audit.json" >/dev/null
"$python_command" -m teaching_skill_miner release-audit "$repo_root/artifacts/public" \
  --output "$release_tmp/public-release-audit.json" >/dev/null

final_wheel="$repo_root/dist/$(basename -- "$candidate_wheel")"
staged_wheel=$(mktemp "$repo_root/dist/.$(basename -- "$candidate_wheel").verified.XXXXXX")
cp "$candidate_wheel" "$staged_wheel"
chmod 0644 "$staged_wheel"
cmp "$candidate_wheel" "$staged_wheel"
mv -f "$staged_wheel" "$final_wheel"
staged_wheel=

staged_acceptance=$(mktemp "$(dirname -- "$output_path")/.release-acceptance.XXXXXX")
"$python_command" scripts/generate_release_acceptance.py acceptance \
  --wheel "$final_wheel" \
  --project-verification-receipt "$release_tmp/project-verification.json" \
  --exact-wheel-receipt "$release_tmp/exact-wheel-verification.json" \
  --wheel-release-audit "$release_tmp/wheel-release-audit.json" \
  --public-directory "$repo_root/artifacts/public" \
  --public-release-audit "$release_tmp/public-release-audit.json" \
  --repository-root "$repo_root" \
  --output "$staged_acceptance" >/dev/null
"$python_command" -m teaching_skill_miner release-audit "$staged_acceptance" \
  --output "$release_tmp/acceptance-release-audit.json" >/dev/null
chmod 0644 "$staged_acceptance"
mv -f "$staged_acceptance" "$output_path"
staged_acceptance=

"$python_command" -c 'import hashlib, pathlib, sys; path=pathlib.Path(sys.argv[1]); payload=path.read_bytes(); print(f"release_wheel={path}"); print(f"wheel_sha256={hashlib.sha256(payload).hexdigest()}"); print(f"release_acceptance={sys.argv[2]}")' "$final_wheel" "$output_path"
