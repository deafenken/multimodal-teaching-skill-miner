#!/usr/bin/env sh
set -eu
umask 077

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

dataset_root="${1:-data/real/ouc_cge}"
raw_dir="$dataset_root/raw"
extract_dir="$dataset_root/extracted"

command -v curl >/dev/null 2>&1 || {
  echo "curl is required" >&2
  exit 1
}
command -v unzip >/dev/null 2>&1 || {
  echo "unzip is required" >&2
  exit 1
}

mkdir -p "$raw_dir" "$extract_dir/low" "$extract_dir/mid" "$extract_dir/high"
chmod 700 "$dataset_root" "$raw_dir" "$extract_dir" "$extract_dir/low" "$extract_dir/mid" "$extract_dir/high"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

download_and_verify() {
  label="$1"
  url="$2"
  expected="$3"
  archive="$raw_dir/${label}_sample.zip"
  partial="$archive.partial"

  if [ -f "$archive" ] && [ "$(sha256_file "$archive")" = "$expected" ]; then
    :
  else
    curl -L --fail --retry 3 "$url" -o "$partial"
    partial_sha="$(sha256_file "$partial")"
    if [ "$partial_sha" != "$expected" ]; then
      echo "SHA-256 mismatch for downloaded $partial" >&2
      exit 1
    fi
    mv -f "$partial" "$archive"
  fi
  actual="$(sha256_file "$archive")"
  if [ "$actual" != "$expected" ]; then
    echo "SHA-256 mismatch for $archive" >&2
    exit 1
  fi
  # Each official archive already contains its own low/mid/high directory.
  unzip -oq "$archive" -d "$extract_dir"
  chmod 600 "$archive"
}

download_and_verify low "https://osf.io/download/xcwj8/" "4575e7fd9d433ffc1b6da1ba8a28eacd4feb95b7afdd0264cfd7a9f6599aa778"
download_and_verify mid "https://osf.io/download/dnex8/" "ec5f5db9868bf8e3ccfd72ec04f965e6adad767276112240aba96bf7a0f6ad42"
download_and_verify high "https://osf.io/download/aw2kp/" "f6cae1ce079b5e698cdc9db3654d163da6a7b03877d7701c7f0d816c2281ee2d"

echo "OUC-CGE public samples are available under $extract_dir"
echo "These recordings contain identifiable adults; keep them local and follow the dataset terms."
