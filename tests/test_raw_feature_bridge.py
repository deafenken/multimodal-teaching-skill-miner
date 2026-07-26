from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import jsonschema

try:
    import numpy  # noqa: F401
    import sklearn  # noqa: F401

    RECOGNITION_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency environment
    RECOGNITION_DEPS_AVAILABLE = False

from teaching_skill_miner.cli import _load_strict_feature_bundle, main
from teaching_skill_miner.io_utils import project_root, read_json
from teaching_skill_miner.recognition.raw_feature_bridge import (
    extract_strict_feature_bundle,
    synchronized_content_sha256,
)
from teaching_skill_miner.recognition.strict_evaluation import strict_dataset_fingerprint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_manifest(
    records: list[dict], *, dataset_id: str = "raw-bridge-fixture"
) -> dict:
    audit = {
        "dataset_id": dataset_id,
        "provenance_verified": True,
        "real_classroom_recording": True,
        "independent_human_ground_truth": True,
        "synchronized_modalities_verified": True,
        "identity_metadata_verified": {
            "session_id": True,
            "teacher_id": True,
            "site_id": True,
        },
        "identity_metadata_source": "independent fixture roster",
        "ground_truth_source": "independent fixture labels",
    }
    audit["dataset_fingerprint"] = strict_dataset_fingerprint(records)
    return {"schema_version": "2.0", "audit": audit, "records": records}


@unittest.skipUnless(RECOGNITION_DEPS_AVAILABLE, "recognition dependencies are optional")
class RawFeatureBridgeTests(unittest.TestCase):
    def _sensor_fixture(self, root: Path) -> tuple[dict, dict]:
        records = []
        for index in range(2):
            path = root / f"sensor_{index}.json"
            path.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "timestamp": row,
                                "heart_rate": 60 + index * 10 + row,
                                "accel_x": index + row / 10,
                                "accel_y": index - row / 20,
                            }
                            for row in range(4)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            raw_inputs = {
                "sensor": {"path": path.name, "sha256": _sha256(path)}
            }
            sample_id = f"sample-{index}"
            records.append(
                {
                    "sample_id": sample_id,
                    "content_sha256": synchronized_content_sha256(raw_inputs),
                    "label": index,
                    "label_name": ("low", "high")[index],
                    "session_id": f"session-{index}",
                    "teacher_id": f"teacher-{index}",
                    "site_id": "site-a",
                    "modalities": {"heart": True, "motion": True},
                    "raw_inputs": raw_inputs,
                }
            )
        config = {
            "schema_version": "1.0",
            "extractor_suite_id": "sensor-fixture-v1",
            "modalities": {
                "heart": {
                    "kind": "builtin_numeric_sensor_summary_v1",
                    "input_key": "sensor",
                    "value_fields": ["heart_rate"],
                    "timestamp_field": "timestamp",
                    "minimum_rows": 2,
                },
                "motion": {
                    "kind": "builtin_numeric_sensor_summary_v1",
                    "input_key": "sensor",
                    "value_fields": ["accel_x", "accel_y"],
                    "timestamp_field": "timestamp",
                    "minimum_rows": 2,
                },
            },
        }
        return _strict_manifest(records), config

    def test_sensor_cli_builds_private_content_addressed_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, config = self._sensor_fixture(root)
            manifest_path = root / "manifest.json"
            config_path = root / "config.json"
            output = root / "private-output"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                return_code = main(
                    [
                        "extract-strict-features",
                        "--manifest",
                        str(manifest_path),
                        "--raw-root",
                        str(root),
                        "--extractor-config",
                        str(config_path),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(return_code, 0)
            bundle_path = output / "strict_feature_bundle.json"
            bundle = read_json(bundle_path)
            jsonschema.Draft202012Validator(
                read_json(project_root() / "schema/strict_feature_bundle.schema.json")
            ).validate(bundle)
            self.assertEqual(bundle["sample_ids"], ["sample-0", "sample-1"])
            self.assertEqual(set(bundle["matrices"]), {"heart", "motion"})
            self.assertEqual(len(bundle["matrices"]["heart"][0]), 9)
            self.assertEqual(len(bundle["matrices"]["motion"][0]), 18)
            self.assertFalse(
                bundle["raw_source_binding"]["claim_scope"][
                    "automatic_recognition_performed"
                ]
            )
            for provenance in bundle["feature_provenance"].values():
                self.assertTrue(provenance["implementation_artifacts"])
                self.assertEqual(provenance["weight_artifacts"], [])
                self.assertFalse(provenance["learned_weights_used"])
                self.assertTrue(
                    provenance["execution_entrypoint"].endswith(
                        ":_extract_sensor_features"
                    )
                )
                self.assertRegex(provenance["configuration_fingerprint"], r"^[0-9a-f]{64}$")
                self.assertRegex(provenance["runtime_fingerprint"], r"^[0-9a-f]{64}$")
            _, _, _, binding = _load_strict_feature_bundle(bundle_path, manifest)
            self.assertTrue(binding["raw_source_binding_verified"])
            self.assertTrue(binding["extractor_suite_binding_verified"])
            self.assertEqual(
                binding["raw_source_binding_fingerprint"],
                bundle["raw_source_binding"]["binding_fingerprint"],
            )
            if os.name == "posix":
                self.assertEqual(output.stat().st_mode & 0o777, 0o700)
                self.assertEqual(bundle_path.stat().st_mode & 0o777, 0o600)

    def test_raw_byte_or_binding_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, config = self._sensor_fixture(root)
            bundle = extract_strict_feature_bundle(
                manifest,
                root,
                config,
                required_identity_fields=["session_id", "teacher_id", "site_id"],
            )
            manifest_path = root / "manifest.json"
            config_path = root / "config.json"
            bundle_path = root / "bundle.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            tampered_bundle = json.loads(json.dumps(bundle))
            tampered_bundle["raw_source_binding"]["samples"][0]["inputs"][0][
                "file_sha256"
            ] = "0" * 64
            bundle_path.write_text(json.dumps(tampered_bundle), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw source"):
                _load_strict_feature_bundle(bundle_path, manifest)

            (root / "sensor_0.json").write_text('{"samples": [{"changed": 1}]}')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                return_code = main(
                    [
                        "extract-strict-features",
                        "--manifest",
                        str(manifest_path),
                        "--raw-root",
                        str(root),
                        "--extractor-config",
                        str(config_path),
                        "--output-dir",
                        str(root / "out"),
                    ]
                )
            self.assertEqual(return_code, 1)

    def test_raw_path_escape_and_window_rebinding_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, config = self._sensor_fixture(root)
            nested_root = root / "private-raw-root"
            nested_root.mkdir()
            manifest["records"][0]["raw_inputs"]["sensor"]["path"] = "../sensor_0.json"
            with self.assertRaisesRegex(ValueError, "escapes raw_root"):
                extract_strict_feature_bundle(
                    manifest,
                    nested_root,
                    config,
                    required_identity_fields=[
                        "session_id",
                        "teacher_id",
                        "site_id",
                    ],
                )

            raw_hash = "1" * 64
            first = synchronized_content_sha256(
                {
                    "video": {
                        "sha256": raw_hash,
                        "start_offset_seconds": 0.0,
                        "duration_seconds": 10.0,
                    }
                }
            )
            rebound = synchronized_content_sha256(
                {
                    "video": {
                        "sha256": raw_hash,
                        "start_offset_seconds": 1.0,
                        "duration_seconds": 10.0,
                    }
                }
            )
            self.assertNotEqual(first, rebound)

    def test_example_config_conforms_to_schema(self) -> None:
        root = project_root()
        schema = read_json(root / "schema/raw_feature_extractor_config.schema.json")
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(
            read_json(root / "configs/raw_feature_extractor.example.json")
        )

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg and ffprobe are required",
    )
    def test_real_video_and_audio_are_decoded_into_separate_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "classroom_fixture.mp4"
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=blue:s=64x36:r=8:d=2",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=8000:duration=2",
                    "-shortest",
                    "-c:v",
                    "mpeg4",
                    "-c:a",
                    "aac",
                    "-y",
                    str(media),
                ],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                self.skipTest("local ffmpeg lacks the fixture encoders")
            raw_inputs = {
                "video": {
                    "path": media.name,
                    "sha256": _sha256(media),
                    "start_offset_seconds": 0.25,
                    "duration_seconds": 1.0,
                }
            }
            records = [
                {
                    "sample_id": "video-sample",
                    "content_sha256": synchronized_content_sha256(raw_inputs),
                    "label": 0,
                    "label_name": "observed-class",
                    "session_id": "session-video",
                    "teacher_id": "teacher-video",
                    "site_id": "site-video",
                    "modalities": {"visual": True, "audio": True},
                    "raw_inputs": raw_inputs,
                }
            ]
            manifest = _strict_manifest(records, dataset_id="real-decoding-fixture")
            config = {
                "schema_version": "1.0",
                "extractor_suite_id": "decoded-av-fixture-v1",
                "modalities": {
                    "visual": {
                        "kind": "builtin_classroom_video_visual_v1",
                        "input_key": "video",
                        "frame_count": 4,
                        "width": 32,
                        "height": 18,
                        "clip_seconds": 0.8,
                    },
                    "audio": {
                        "kind": "builtin_classroom_video_audio_v1",
                        "input_key": "video",
                        "sample_rate": 8000,
                        "clip_seconds": 0.8,
                    },
                },
            }
            bundle = extract_strict_feature_bundle(
                manifest,
                root,
                config,
                required_identity_fields=["session_id", "teacher_id", "site_id"],
            )
            self.assertGreater(len(bundle["matrices"]["visual"][0]), 100)
            self.assertEqual(
                bundle["feature_names_by_modality"]["audio"][-1], "audio_present"
            )
            self.assertEqual(bundle["matrices"]["audio"][0][-1], 1.0)
            self.assertEqual(
                bundle["raw_source_binding"]["samples"][0]["inputs"][0][
                    "duration_seconds"
                ],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
