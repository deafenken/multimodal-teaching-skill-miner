#!/usr/bin/env sh
set -eu

# Internal runner.  The publication wrapper creates and enters the isolated
# source snapshot before invoking this script; do not call it to publish.

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

if [ "$#" -ne 2 ] || [ "$1" != "--output" ] || [ -z "$2" ]; then
  echo "usage: $0 --output /snapshot/artifacts/release_acceptance_VERSION.json" >&2
  exit 2
fi
output_path=$2
python_command=${PYTHON:-python3}
release_version=$(
  "$python_command" -c 'from teaching_skill_miner import __version__; print(__version__)'
)
release_tmp=$(mktemp -d "${TMPDIR:-/tmp}/tsm-snapshot-release.XXXXXX")
staged_wheel=
staged_acceptance=
proof_directory="$repo_root/artifacts/.release-verification-proofs"
cleanup() {
  rm -rf "$release_tmp"
  if [ -n "$staged_wheel" ]; then rm -f "$staged_wheel"; fi
  if [ -n "$staged_acceptance" ]; then rm -f "$staged_acceptance"; fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

"$python_command" scripts/generate_release_acceptance.py pending \
  --release-version "$release_version" \
  --output "$output_path" >/dev/null

PYTHON="$python_command" sh scripts/verify_project.sh \
  --receipt "$release_tmp/project-verification.json"

mkdir -p "$release_tmp/dist" "$repo_root/dist"
if [ -L "$repo_root/dist" ] || [ ! -d "$repo_root/dist" ]; then
  echo "snapshot release dist path must be a real directory" >&2
  exit 2
fi
PYTHON="$python_command" sh scripts/build_release_wheel.sh "$release_tmp/dist"
set -- "$release_tmp"/dist/*.whl
if [ "$#" -ne 1 ] || [ ! -f "$1" ] || [ -L "$1" ]; then
  echo "clean snapshot release build did not produce exactly one wheel" >&2
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
  --source-snapshot-manifest "$repo_root/.release-source-snapshot.json" \
  --output "$staged_acceptance" >/dev/null
"$python_command" -m teaching_skill_miner release-audit "$staged_acceptance" \
  --output "$release_tmp/acceptance-release-audit.json" >/dev/null
chmod 0644 "$staged_acceptance"
mv -f "$staged_acceptance" "$output_path"
staged_acceptance=

"$python_command" scripts/generate_release_acceptance.py verification-scope \
  --repository-root "$repo_root" \
  --output "$repo_root/artifacts/final-verification-scope.json" >/dev/null

"$python_command" scripts/verify_published_release_pair.py \
  --repository-root "$repo_root" \
  --wheel "$final_wheel" \
  --acceptance "$output_path" \
  --expected-source-scope "$repo_root/artifacts/final-verification-scope.json" \
  --source-snapshot-manifest "$repo_root/.release-source-snapshot.json" \
  --project-verification-receipt "$release_tmp/project-verification.json" \
  --exact-wheel-verification-receipt \
    "$release_tmp/exact-wheel-verification.json"

# Preserve the exact proof bytes only inside the short-lived source snapshot.
# The publication wrapper uses these for its own pre-publication verification;
# they are never copied into the live public artifacts tree.
if [ -L "$proof_directory" ] || \
   { [ -e "$proof_directory" ] && [ ! -d "$proof_directory" ]; }; then
  echo "snapshot release proof path must be a real directory" >&2
  exit 2
fi
mkdir -p "$proof_directory"
cp "$release_tmp/project-verification.json" "$proof_directory/project.json"
cp "$release_tmp/exact-wheel-verification.json" "$proof_directory/exact-wheel.json"
chmod 0600 "$proof_directory/project.json" "$proof_directory/exact-wheel.json"
