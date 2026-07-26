#!/usr/bin/env sh
set -eu

umask 077

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
python_command=${PYTHON:-python3}

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/build_teachobs_asr_container.sh \
    --wheel PATH \
    --base-image LOCAL_IMAGE_REFERENCE \
    --expected-base-image-id sha256:<64-lowercase-hex> \
    [--ubuntu-mirror URL] \
    [--pip-index-url URL]

The CUDA base image must already exist in the local Docker image store. The
script never pulls a base image, model, or media. The only stdout on success is
a path-free JSON receipt; Docker build logs are written to stderr.
EOF
}

die() {
  echo "build_teachobs_asr_container.sh: $*" >&2
  exit 2
}

require_sha256_id() {
  value=$1
  case "$value" in
    sha256:*) value_hex=${value#sha256:} ;;
    *) return 1 ;;
  esac
  [ "${#value_hex}" -eq 64 ] || return 1
  case "$value_hex" in
    *[!0-9a-f]*) return 1 ;;
  esac
  return 0
}

wheel=
base_image=
expected_base_image_id=
ubuntu_mirror=http://archive.ubuntu.com/ubuntu
pip_index_url=https://pypi.org/simple

while [ "$#" -gt 0 ]; do
  case "$1" in
    --wheel)
      [ "$#" -ge 2 ] || die "--wheel requires a value"
      wheel=$2
      shift 2
      ;;
    --base-image)
      [ "$#" -ge 2 ] || die "--base-image requires a value"
      base_image=$2
      shift 2
      ;;
    --expected-base-image-id)
      [ "$#" -ge 2 ] || die "--expected-base-image-id requires a value"
      expected_base_image_id=$2
      shift 2
      ;;
    --ubuntu-mirror)
      [ "$#" -ge 2 ] || die "--ubuntu-mirror requires a value"
      ubuntu_mirror=$2
      shift 2
      ;;
    --pip-index-url)
      [ "$#" -ge 2 ] || die "--pip-index-url requires a value"
      pip_index_url=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[ -n "$wheel" ] || die "--wheel is required"
[ -n "$base_image" ] || die "--base-image is required"
[ -n "$expected_base_image_id" ] || die "--expected-base-image-id is required"

[ -f "$wheel" ] || die "release wheel is not a regular file"
[ ! -L "$wheel" ] || die "release wheel may not be a symbolic link"
case "$(basename -- "$wheel")" in
  teaching_skill_miner-*-py3-none-any.whl) ;;
  *) die "wheel must be the exact py3-none-any Teaching Skill Miner release wheel" ;;
esac
wheel_filename=$(basename -- "$wheel")
case "$wheel_filename" in
  *[!A-Za-z0-9_.-]*) die "wheel filename contains unsafe characters" ;;
esac
case "$base_image" in
  ""|-*|*[!A-Za-z0-9._/:@+-]*) die "unsafe local base image reference" ;;
esac
require_sha256_id "$expected_base_image_id" || {
  die "expected base image ID must be sha256:<64 lowercase hex>"
}
case "$ubuntu_mirror" in
  http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu) ;;
  *) die "unsupported Ubuntu package mirror" ;;
esac
case "$pip_index_url" in
  https://pypi.org/simple|https://mirrors.aliyun.com/pypi/simple) ;;
  *) die "unsupported Python package index" ;;
esac
command -v docker >/dev/null 2>&1 || die "docker is required"
command -v "$python_command" >/dev/null 2>&1 || die "Python is required to hash the wheel"

actual_base_image_id=$(
  docker image inspect --format '{{.Id}}' "$base_image"
) || die "CUDA base image is not present in the local Docker image store"
require_sha256_id "$actual_base_image_id" || die "Docker returned an invalid base image ID"
[ "$actual_base_image_id" = "$expected_base_image_id" ] || {
  die "local CUDA base image ID differs from --expected-base-image-id"
}

wheel_sha256=$(
  "$python_command" -c 'from hashlib import sha256; from pathlib import Path; import sys; print(sha256(Path(sys.argv[1]).read_bytes()).hexdigest())' "$wheel"
)
[ "${#wheel_sha256}" -eq 64 ] || die "failed to compute release wheel SHA-256"
case "$wheel_sha256" in
  *[!0-9a-f]*) die "failed to compute release wheel SHA-256" ;;
esac

build_tmp=$(mktemp -d "${TMPDIR:-/tmp}/tsm-teachobs-asr-build.XXXXXX")
trap 'rm -rf "$build_tmp"' EXIT HUP INT TERM
chmod 700 "$build_tmp"
build_context="$build_tmp/context"
mkdir "$build_context"
chmod 700 "$build_context"
cp "$repo_root/docker/teachobs_asr.Dockerfile" "$build_context/Dockerfile"
cp "$wheel" "$build_context/$wheel_filename"
chmod 600 "$build_context/Dockerfile" "$build_context/$wheel_filename"

iid_file="$build_tmp/image.id"
docker build \
  --pull=false \
  --iidfile "$iid_file" \
  --build-arg "BASE_IMAGE=$base_image" \
  --build-arg "BASE_IMAGE_ID=$actual_base_image_id" \
  --build-arg "UBUNTU_MIRROR=$ubuntu_mirror" \
  --build-arg "PIP_INDEX_URL=$pip_index_url" \
  --build-arg "PROJECT_WHEEL_SHA256=$wheel_sha256" \
  --build-arg "PROJECT_WHEEL_FILENAME=$wheel_filename" \
  --file "$build_context/Dockerfile" \
  "$build_context" 1>&2

[ -f "$iid_file" ] || die "Docker did not produce an image ID"
built_image_reference=$(sed -n '1p' "$iid_file")
image_id=$(
  docker image inspect --format '{{.Id}}' "$built_image_reference"
) || die "built image cannot be inspected"
require_sha256_id "$image_id" || die "Docker returned an invalid final image ID"

base_label=$(
  docker image inspect \
    --format '{{ index .Config.Labels "org.teaching-skill-miner.asr.base-image-id" }}' \
    "$image_id"
)
wheel_label=$(
  docker image inspect \
    --format '{{ index .Config.Labels "org.teaching-skill-miner.asr.wheel-sha256" }}' \
    "$image_id"
)
[ "$base_label" = "$actual_base_image_id" ] || die "built image base binding is missing"
[ "$wheel_label" = "$wheel_sha256" ] || die "built image wheel binding is missing"

docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --user 65534:65534 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 \
  "$image_id" \
  -c 'from importlib.metadata import version; expected={"ctranslate2":"4.8.1","faster-whisper":"1.2.1","nvidia-cublas-cu12":"12.9.2.10","nvidia-cudnn-cu12":"9.24.0.43"}; actual={name:version(name) for name in expected}; raise SystemExit(0 if actual == expected else f"ASR package mismatch: {actual!r}")' \
  >/dev/null

help_output="$build_tmp/gpu-entrypoint-help.txt"
docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --user 65534:65534 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  "$image_id" run-teachobs-asr-gpu --help >"$help_output"
grep -F -- '--job-manifest' "$help_output" >/dev/null
grep -F -- '--model-directory' "$help_output" >/dev/null
grep -F -- '--container-image-digest' "$help_output" >/dev/null

docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --user 65534:65534 \
  --entrypoint ffprobe \
  "$image_id" -version >/dev/null

cat <<EOF
{
  "schema": "teaching_skill_miner.teachobs_asr_container_receipt.v1",
  "base_image_id": "$actual_base_image_id",
  "image_id": "$image_id",
  "container_image_digest": "$image_id",
  "wheel_sha256": "$wheel_sha256",
  "ubuntu_package_mirror": "$ubuntu_mirror",
  "python_package_index": "$pip_index_url",
  "package_versions": {
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.1",
    "nvidia-cublas-cu12": "12.9.2.10",
    "nvidia-cudnn-cu12": "9.24.0.43"
  },
  "gpu_entrypoint_help_verified": true,
  "ffprobe_verified": true,
  "base_pull_allowed": false,
  "model_or_media_embedded": false,
  "private_paths_recorded": false
}
EOF
