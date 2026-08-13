#!/usr/bin/env sh
set -eu

# This is the only release-publication entrypoint.  It owns a repository-scoped
# lock for the full operation, invalidates any previous positive receipt, takes
# one source/evidence snapshot, and publishes only the pair verified there.

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

if [ -L "$repo_root/artifacts" ] || [ ! -d "$repo_root/artifacts" ]; then
  echo "release artifacts path must be an existing real directory: $repo_root/artifacts" >&2
  exit 2
fi

release_lock="$repo_root/artifacts/.release-acceptance.lock"
lock_owner="$release_lock/owner"
lock_started_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
lock_token="$$:$lock_started_at:$repo_root"
release_lock_acquired=false
release_tmp=
staged_wheel=
staged_acceptance=
cleanup() {
  if [ -n "$release_tmp" ]; then
    rm -rf "$release_tmp"
  fi
  if [ -n "$staged_wheel" ]; then
    rm -f "$staged_wheel"
  fi
  if [ -n "$staged_acceptance" ]; then
    rm -f "$staged_acceptance"
  fi
  if [ "$release_lock_acquired" = true ]; then
    if [ -f "$lock_owner" ] && [ ! -L "$lock_owner" ] && \
       [ "$(sed -n '1p' "$lock_owner")" = "token=$lock_token" ]; then
      rm -f "$lock_owner"
      if ! rmdir "$release_lock"; then
        echo "release lock could not be removed safely: $release_lock" >&2
      fi
    elif [ ! -e "$lock_owner" ]; then
      if ! rmdir "$release_lock"; then
        echo "release lock could not be removed safely: $release_lock" >&2
      fi
    elif [ -e "$release_lock" ]; then
      echo "release lock ownership changed; refusing to remove it: $release_lock" >&2
    fi
  fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if ! mkdir "$release_lock" 2>/dev/null; then
  echo "release acceptance already running (lock: $release_lock)" >&2
  if [ -f "$lock_owner" ] && [ ! -L "$lock_owner" ]; then
    sed -n '1,8p' "$lock_owner" >&2
  fi
  echo "A SIGKILL can leave a stale lock. Confirm the recorded PID is no longer running, then remove only this lock directory manually." >&2
  exit 75
fi
release_lock_acquired=true
printf 'token=%s\npid=%s\nstarted_at_utc=%s\nrepository=%s\nsnapshot_scope=pending\n' \
  "$lock_token" "$$" "$lock_started_at" "$repo_root" >"$lock_owner"
chmod 0600 "$lock_owner"

if [ "$#" -gt 2 ]; then
  echo "usage: $0 [--output artifacts/release_acceptance_VERSION.json]" >&2
  exit 2
fi

python_command=${PYTHON:-python3}
case "$python_command" in
  */*)
    case "$python_command" in
      /*) ;;
      *) python_command="$repo_root/$python_command" ;;
    esac
    ;;
esac
release_version=$(
  "$python_command" -c 'from teaching_skill_miner import __version__; print(__version__)'
)
output_path="$repo_root/artifacts/release_acceptance_${release_version}.json"
if [ "$#" -eq 2 ]; then
  if [ "$1" != "--output" ] || [ -z "$2" ]; then
    echo "usage: $0 [--output artifacts/release_acceptance_VERSION.json]" >&2
    exit 2
  fi
  case "$2" in
    /*) output_path=$2 ;;
    *) output_path="$repo_root/$2" ;;
  esac
elif [ "$#" -eq 1 ]; then
  echo "usage: $0 [--output artifacts/release_acceptance_VERSION.json]" >&2
  exit 2
fi

expected_output_name="release_acceptance_${release_version}.json"
if [ "$(basename -- "$output_path")" != "$expected_output_name" ]; then
  echo "acceptance output filename must be $expected_output_name" >&2
  exit 2
fi
canonical_output="$repo_root/artifacts/$expected_output_name"
if [ "$output_path" != "$canonical_output" ]; then
  echo "acceptance output must be the canonical repository path: $canonical_output" >&2
  exit 2
fi

output_parent=$(dirname -- "$output_path")
if [ -L "$output_parent" ] || [ ! -d "$output_parent" ]; then
  echo "acceptance output parent must be an existing real directory: $output_parent" >&2
  exit 2
fi
if [ -L "$repo_root/dist" ] || \
   { [ -e "$repo_root/dist" ] && [ ! -d "$repo_root/dist" ]; }; then
  echo "release dist path must be a real directory: $repo_root/dist" >&2
  exit 2
fi
mkdir -p "$repo_root/dist"

# Fail closed before any expensive work.  If snapshot creation or a gate later
# fails, the old positive receipt cannot continue to look current.
"$python_command" scripts/generate_release_acceptance.py pending \
  --release-version "$release_version" \
  --output "$output_path" >/dev/null

release_tmp=$(mktemp -d "${TMPDIR:-/tmp}/tsm-final-release.XXXXXX")
release_tmp=$(CDPATH= cd -P -- "$release_tmp" && pwd)
source_snapshot="$release_tmp/source"
snapshot_manifest="$release_tmp/source-snapshot.json"
"$python_command" scripts/create_release_source_snapshot.py \
  --repository-root "$repo_root" \
  --destination "$source_snapshot" \
  --manifest-output "$snapshot_manifest" >/dev/null
snapshot_release_version=$(
  "$python_command" - "$source_snapshot/teaching_skill_miner/__init__.py" <<'PY'
import ast
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
tree = ast.parse(source, filename=sys.argv[1])
values = [
    node.value.value
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    and isinstance(node.value, ast.Constant)
    and isinstance(node.value.value, str)
]
if len(values) != 1 or not values[0]:
    raise SystemExit("snapshot package has no unique literal __version__")
print(values[0])
PY
)
if [ "$snapshot_release_version" != "$release_version" ]; then
  echo "release version changed while the source snapshot was being captured" >&2
  exit 2
fi
snapshot_scope=$(
  "$python_command" -c \
    'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value["snapshot_sha256"])' \
    "$snapshot_manifest"
)
printf 'token=%s\npid=%s\nstarted_at_utc=%s\nrepository=%s\nsnapshot_scope=%s\n' \
  "$lock_token" "$$" "$lock_started_at" "$repo_root" "$snapshot_scope" >"$lock_owner"
chmod 0600 "$lock_owner"

snapshot_output="$source_snapshot/artifacts/release_acceptance_${snapshot_release_version}.json"

# Environment-selected conditional evidence is copied at the same relative
# path.  Rebase those paths so every read performed by the gate stays inside
# the immutable snapshot.
snapshot_path() {
  "$python_command" - "$repo_root" "$source_snapshot" "$1" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
snapshot = Path(sys.argv[2]).resolve(strict=True)
requested = Path(sys.argv[3])
if not requested.is_absolute():
    requested = root / requested
try:
    relative = requested.resolve(strict=True).relative_to(root)
except (FileNotFoundError, ValueError) as exc:
    raise SystemExit("conditional release evidence path is missing or outside the repository") from exc
print(snapshot / relative)
PY
}

if [ "${TSM_TEACHOBS_HUMAN_ROOT+x}" = x ]; then
  TSM_TEACHOBS_HUMAN_ROOT=$(snapshot_path "$TSM_TEACHOBS_HUMAN_ROOT")
  export TSM_TEACHOBS_HUMAN_ROOT
fi
if [ "${TSM_TEACHOBS_HUMAN_RECEIPT+x}" = x ]; then
  TSM_TEACHOBS_HUMAN_RECEIPT=$(snapshot_path "$TSM_TEACHOBS_HUMAN_RECEIPT")
  export TSM_TEACHOBS_HUMAN_RECEIPT
fi
if [ "${TSM_TEACHOBS_LOCKBOX_DRAFT+x}" = x ]; then
  TSM_TEACHOBS_LOCKBOX_DRAFT=$(snapshot_path "$TSM_TEACHOBS_LOCKBOX_DRAFT")
  export TSM_TEACHOBS_LOCKBOX_DRAFT
fi
if [ "${TSM_TEACHOBS_FROZEN_MODEL_ROOT+x}" = x ]; then
  TSM_TEACHOBS_FROZEN_MODEL_ROOT=$(snapshot_path "$TSM_TEACHOBS_FROZEN_MODEL_ROOT")
  export TSM_TEACHOBS_FROZEN_MODEL_ROOT
fi

(
  cd "$source_snapshot"
  PYTHON="$python_command" sh scripts/build_release_acceptance_snapshot.sh \
    --output "$snapshot_output"
)

set -- "$source_snapshot"/dist/*.whl
if [ "$#" -ne 1 ] || [ ! -f "$1" ] || [ -L "$1" ]; then
  echo "verified source snapshot did not produce exactly one regular wheel" >&2
  exit 2
fi
snapshot_final_wheel=$1
snapshot_scope_receipt="$source_snapshot/artifacts/final-verification-scope.json"
snapshot_project_proof="$source_snapshot/artifacts/.release-verification-proofs/project.json"
snapshot_exact_proof="$source_snapshot/artifacts/.release-verification-proofs/exact-wheel.json"
"$python_command" "$source_snapshot/scripts/verify_published_release_pair.py" \
  --repository-root "$source_snapshot" \
  --wheel "$snapshot_final_wheel" \
  --acceptance "$snapshot_output" \
  --expected-source-scope "$snapshot_scope_receipt" \
  --source-snapshot-manifest "$source_snapshot/.release-source-snapshot.json" \
  --project-verification-receipt "$snapshot_project_proof" \
  --exact-wheel-verification-receipt "$snapshot_exact_proof"

final_wheel="$repo_root/dist/$(basename -- "$snapshot_final_wheel")"
staged_wheel=$(mktemp "$repo_root/dist/.$(basename -- "$snapshot_final_wheel").verified.XXXXXX")
cp "$snapshot_final_wheel" "$staged_wheel"
chmod 0644 "$staged_wheel"
cmp "$snapshot_final_wheel" "$staged_wheel"

# The positive receipt is published last.  Until this atomic rename, readers
# see the pending receipt written above rather than a mismatched positive pair.
staged_acceptance=$(mktemp "$output_parent/.release-acceptance.XXXXXX")
cp "$snapshot_output" "$staged_acceptance"
chmod 0644 "$staged_acceptance"
cmp "$snapshot_output" "$staged_acceptance"
published_wheel_sha256=$(
  "$python_command" -c \
    'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "$staged_wheel"
)
mv -f "$staged_wheel" "$final_wheel"
staged_wheel=
mv -f "$staged_acceptance" "$output_path"
staged_acceptance=

printf 'release_wheel=%s\nwheel_sha256=%s\nrelease_acceptance=%s\n' \
  "$final_wheel" "$published_wheel_sha256" "$output_path"
