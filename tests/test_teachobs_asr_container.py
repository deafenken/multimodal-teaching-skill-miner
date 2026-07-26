from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / "scripts/build_teachobs_asr_container.sh"
DOCKERFILE = PROJECT_ROOT / "docker/teachobs_asr.Dockerfile"
BASE_IMAGE_ID = "sha256:" + "b" * 64
FINAL_IMAGE_ID = "sha256:" + "c" * 64


class TeachObsAsrContainerTests(unittest.TestCase):
    def _fake_docker(self, root: Path) -> tuple[Path, Path, Path]:
        binary_directory = root / "bin"
        binary_directory.mkdir()
        state_directory = root / "state"
        state_directory.mkdir()
        log_path = root / "docker.log"
        docker = binary_directory / "docker"
        docker.write_text(
            """#!/usr/bin/env sh
set -eu
printf '%s\\n' "$*" >>"$FAKE_DOCKER_LOG"

if [ "$1" = image ] && [ "$2" = inspect ]; then
  format=$4
  reference=$5
  case "$format" in
    *base-image-id*) cat "$FAKE_DOCKER_STATE/base-label" ;;
    *wheel-sha256*) cat "$FAKE_DOCKER_STATE/wheel-label" ;;
    '{{.Id}}')
      if [ "$reference" = local/cuda:test ]; then
        printf '%s\\n' "$FAKE_BASE_IMAGE_ID"
      else
        printf '%s\\n' "$FAKE_FINAL_IMAGE_ID"
      fi
      ;;
    *) exit 81 ;;
  esac
  exit 0
fi

if [ "$1" = build ]; then
  shift
  iid_file=
  context=
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --iidfile)
        iid_file=$2
        shift 2
        ;;
      --build-arg)
        case "$2" in
          BASE_IMAGE=*) printf '%s\\n' "${2#BASE_IMAGE=}" >"$FAKE_DOCKER_STATE/base-reference" ;;
          BASE_IMAGE_ID=*) printf '%s\\n' "${2#BASE_IMAGE_ID=}" >"$FAKE_DOCKER_STATE/base-label" ;;
          PROJECT_WHEEL_SHA256=*) printf '%s\\n' "${2#PROJECT_WHEEL_SHA256=}" >"$FAKE_DOCKER_STATE/wheel-label" ;;
          PROJECT_WHEEL_FILENAME=*) printf '%s\\n' "${2#PROJECT_WHEEL_FILENAME=}" >"$FAKE_DOCKER_STATE/wheel-filename" ;;
        esac
        shift 2
        ;;
      --file)
        [ -f "$2" ] || exit 82
        shift 2
        ;;
      --pull=false)
        : >"$FAKE_DOCKER_STATE/pull-false"
        shift
        ;;
      *)
        context=$1
        shift
        ;;
    esac
  done
  [ -f "$FAKE_DOCKER_STATE/pull-false" ] || exit 83
  [ -f "$context/Dockerfile" ] || exit 84
  [ -f "$context/$(cat "$FAKE_DOCKER_STATE/wheel-filename")" ] || exit 85
  [ "$(find "$context" -mindepth 1 -maxdepth 1 -type f | wc -l | tr -d ' ')" = 2 ] || exit 86
  [ "$(cat "$FAKE_DOCKER_STATE/base-label")" = "$FAKE_BASE_IMAGE_ID" ] || exit 87
  [ "$(cat "$FAKE_DOCKER_STATE/base-reference")" = local/cuda:test ] || exit 89
  printf '%s\\n' "$FAKE_FINAL_IMAGE_ID" >"$iid_file"
  exit 0
fi

if [ "$1" = run ]; then
  case " $* " in
    *' run-teachobs-asr-gpu --help '*)
      printf '%s\\n' 'usage: tsm run-teachobs-asr-gpu --job-manifest FILE --model-directory DIR --container-image-digest SHA'
      ;;
  esac
  exit 0
fi

exit 88
""",
            encoding="utf-8",
        )
        docker.chmod(0o700)
        return binary_directory, state_directory, log_path

    def _environment(
        self, binary_directory: Path, state_directory: Path, log_path: Path
    ) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{binary_directory}{os.pathsep}{environment['PATH']}",
                "PYTHON": sys.executable,
                "FAKE_DOCKER_LOG": os.fspath(log_path),
                "FAKE_DOCKER_STATE": os.fspath(state_directory),
                "FAKE_BASE_IMAGE_ID": BASE_IMAGE_ID,
                "FAKE_FINAL_IMAGE_ID": FINAL_IMAGE_ID,
            }
        )
        return environment

    def test_build_uses_only_exact_wheel_and_emits_path_free_digest_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "teaching_skill_miner-1.2.0-py3-none-any.whl"
            wheel.write_bytes(b"exact release wheel fixture\n")
            binary_directory, state_directory, log_path = self._fake_docker(root)
            temporary_root = root / "private-tmp"
            temporary_root.mkdir(mode=0o700)
            environment = self._environment(
                binary_directory, state_directory, log_path
            )
            environment["TMPDIR"] = os.fspath(temporary_root)

            completed = subprocess.run(
                [
                    "sh",
                    os.fspath(BUILD_SCRIPT),
                    "--wheel",
                    os.fspath(wheel),
                    "--base-image",
                    "local/cuda:test",
                    "--expected-base-image-id",
                    BASE_IMAGE_ID,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads(completed.stdout)
            self.assertEqual(receipt["base_image_id"], BASE_IMAGE_ID)
            self.assertEqual(receipt["image_id"], FINAL_IMAGE_ID)
            self.assertEqual(receipt["container_image_digest"], FINAL_IMAGE_ID)
            self.assertEqual(
                receipt["ubuntu_package_mirror"],
                "http://archive.ubuntu.com/ubuntu",
            )
            self.assertEqual(
                receipt["python_package_index"], "https://pypi.org/simple"
            )
            self.assertEqual(receipt["wheel_sha256"], sha256(wheel.read_bytes()).hexdigest())
            self.assertEqual(
                receipt["package_versions"],
                {
                    "faster-whisper": "1.2.1",
                    "ctranslate2": "4.8.1",
                    "nvidia-cublas-cu12": "12.9.2.10",
                    "nvidia-cudnn-cu12": "9.24.0.43",
                },
            )
            self.assertTrue(receipt["gpu_entrypoint_help_verified"])
            self.assertTrue(receipt["ffprobe_verified"])
            self.assertFalse(receipt["base_pull_allowed"])
            self.assertFalse(receipt["model_or_media_embedded"])
            self.assertFalse(receipt["private_paths_recorded"])
            serialized = json.dumps(receipt, sort_keys=True)
            self.assertNotIn(os.fspath(root), serialized)
            self.assertNotIn("local/cuda:test", serialized)
            self.assertIn("--pull=false", log_path.read_text(encoding="utf-8"))
            self.assertFalse(any(temporary_root.iterdir()))

    def test_base_image_id_mismatch_fails_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "teaching_skill_miner-1.2.0-py3-none-any.whl"
            wheel.write_bytes(b"wheel\n")
            binary_directory, state_directory, log_path = self._fake_docker(root)
            completed = subprocess.run(
                [
                    "sh",
                    os.fspath(BUILD_SCRIPT),
                    "--wheel",
                    os.fspath(wheel),
                    "--base-image",
                    "local/cuda:test",
                    "--expected-base-image-id",
                    "sha256:" + "d" * 64,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=self._environment(binary_directory, state_directory, log_path),
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("differs", completed.stderr)
            self.assertNotIn("build", log_path.read_text(encoding="utf-8"))

    def test_symlink_wheel_and_unsafe_base_reference_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "teaching_skill_miner-1.2.0-py3-none-any.whl"
            wheel.write_bytes(b"wheel\n")
            symlink = root / "teaching_skill_miner-symlink-py3-none-any.whl"
            symlink.symlink_to(wheel)
            for candidate, expected_error in (
                (symlink, "symbolic link"),
                (wheel, "unsafe local base image reference"),
            ):
                arguments = [
                    "sh",
                    os.fspath(BUILD_SCRIPT),
                    "--wheel",
                    os.fspath(candidate),
                    "--base-image",
                    "local/cuda:test" if candidate == symlink else "--pull=true",
                    "--expected-base-image-id",
                    BASE_IMAGE_ID,
                ]
                with self.subTest(candidate=candidate.name):
                    completed = subprocess.run(
                        arguments,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 2)
                    self.assertIn(expected_error, completed.stderr)

    def test_unapproved_package_mirrors_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "teaching_skill_miner-1.2.0-py3-none-any.whl"
            wheel.write_bytes(b"wheel\n")
            for option, value, expected in (
                ("--ubuntu-mirror", "https://untrusted.invalid/ubuntu", "Ubuntu"),
                ("--pip-index-url", "https://untrusted.invalid/simple", "Python"),
            ):
                completed = subprocess.run(
                    [
                        "sh",
                        os.fspath(BUILD_SCRIPT),
                        "--wheel",
                        os.fspath(wheel),
                        "--base-image",
                        "local/cuda:test",
                        "--expected-base-image-id",
                        BASE_IMAGE_ID,
                        option,
                        value,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                with self.subTest(option=option):
                    self.assertEqual(completed.returncode, 2)
                    self.assertIn(expected, completed.stderr)

    def test_dockerfile_binds_base_wheel_and_exact_gpu_runtime(self) -> None:
        source = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("ARG BASE_IMAGE", source)
        self.assertEqual(source.count("ARG BASE_IMAGE"), 2)
        self.assertIn("FROM ${BASE_IMAGE}", source)
        self.assertIn("ARG BASE_IMAGE_ID", source)
        self.assertIn("ARG UBUNTU_MIRROR", source)
        self.assertIn("ARG PIP_INDEX_URL", source)
        self.assertIn(
            'org.teaching-skill-miner.asr.base-image-id="${BASE_IMAGE_ID}"',
            source,
        )
        self.assertIn("ARG PROJECT_WHEEL_SHA256", source)
        self.assertGreater(
            source.index(
                'LABEL org.teaching-skill-miner.asr.wheel-sha256='
            ),
            source.index("nvidia-cudnn-cu12==9.24.0.43"),
        )
        self.assertIn("ARG PROJECT_WHEEL_FILENAME", source)
        self.assertIn("COPY ${PROJECT_WHEEL_FILENAME}", source)
        self.assertNotIn("COPY .", source)
        self.assertIn("ffmpeg", source)
        self.assertIn("HF_HUB_OFFLINE=1", source)
        self.assertIn("TRANSFORMERS_OFFLINE=1", source)
        for requirement in (
            "faster-whisper==1.2.1",
            "ctranslate2==4.8.1",
            "nvidia-cublas-cu12==12.9.2.10",
            "nvidia-cudnn-cu12==9.24.0.43",
        ):
            self.assertIn(requirement, source)
        self.assertNotIn("artifacts/private", source)
        self.assertNotIn("/models/", source)
        self.assertNotIn("/media/", source)


if __name__ == "__main__":
    unittest.main()
