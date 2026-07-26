from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

try:
    import numpy  # noqa: F401
    import sklearn  # noqa: F401

    RECOGNITION_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency environment
    RECOGNITION_DEPS_AVAILABLE = False

from teaching_skill_miner.recognition.dipser_experiment import (
    feature_bundle_fingerprint,
)
from teaching_skill_miner.recognition.strict_evaluation import (
    strict_dataset_fingerprint,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_valid_bundle(root: Path) -> tuple[Path, Path]:
    records = [
        {
            "sample_id": "sample-0",
            "content_sha256": "0" * 64,
            "label": 0,
            "session_id": "session-0",
            "participant_id": "participant-0",
            "cohort_id": "cohort-0",
            "activity_id": "activity-0",
            "site_id": "site-0",
            "modalities": {"visual": True, "sensor": True},
        },
        {
            "sample_id": "sample-1",
            "content_sha256": "1" * 64,
            "label": 1,
            "session_id": "session-1",
            "participant_id": "participant-1",
            "cohort_id": "cohort-1",
            "activity_id": "activity-1",
            "site_id": "site-0",
            "modalities": {"visual": True, "sensor": True},
        },
    ]
    names = {"visual": ["visual_0", "visual_1"], "sensor": ["sensor_0"]}
    matrices = {
        "visual": [[0.1, 0.2], [1.1, 1.2]],
        "sensor": [[0.3], [1.3]],
    }
    dataset_fingerprint = strict_dataset_fingerprint(records)
    bundle_fingerprint = feature_bundle_fingerprint(records, names, matrices)
    manifest = {
        "schema_version": "2.0",
        "audit": {
            "dataset_fingerprint": dataset_fingerprint,
            "feature_bundle_fingerprint": bundle_fingerprint,
        },
        "records": records,
    }
    features = {
        "schema_version": "1.0",
        "dataset_fingerprint": dataset_fingerprint,
        "feature_bundle_fingerprint": bundle_fingerprint,
        "sample_ids": [record["sample_id"] for record in records],
        "feature_names_by_modality": names,
        "matrices": matrices,
    }
    manifest_path = root / "dataset_manifest.json"
    features_path = root / "features.json"
    _write_json(manifest_path, manifest)
    _write_json(features_path, features)
    return manifest_path, features_path


@unittest.skipUnless(
    RECOGNITION_DEPS_AVAILABLE, "recognition dependencies are optional"
)
class DipserRunnerArtifactIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from scripts.run_dipser_hierarchical_challenge import (
            _load_inputs as load_hierarchical_inputs,
        )
        from scripts.run_dipser_optimization import (
            _load_inputs as load_optimization_inputs,
        )

        cls.loaders: tuple[tuple[str, Callable[[Path], dict[str, Any]]], ...] = (
            ("optimization", load_optimization_inputs),
            ("hierarchical", load_hierarchical_inputs),
        )

    def _assert_all_loaders_reject(self, root: Path, message: str) -> None:
        for name, loader in self.loaders:
            with self.subTest(loader=name):
                with self.assertRaisesRegex(ValueError, message):
                    loader(root)

    def test_valid_bundle_is_accepted_by_both_runners(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_valid_bundle(root)

            for name, loader in self.loaders:
                with self.subTest(loader=name):
                    loaded = loader(root)
                    self.assertEqual(loaded["dataset_fingerprint"], strict_dataset_fingerprint(loaded["records"]))

    def test_tampered_label_fails_dataset_fingerprint_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _ = _write_valid_bundle(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["records"][0]["label"] = 2
            _write_json(manifest_path, manifest)

            self._assert_all_loaders_reject(root, "dataset fingerprint binding failed")

    def test_tampered_feature_value_fails_bundle_fingerprint_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, features_path = _write_valid_bundle(root)
            features = json.loads(features_path.read_text(encoding="utf-8"))
            features["matrices"]["visual"][0][0] += 999.0
            _write_json(features_path, features)

            self._assert_all_loaders_reject(
                root, "feature-bundle fingerprint binding failed"
            )

    def test_tampered_feature_name_fails_bundle_fingerprint_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, features_path = _write_valid_bundle(root)
            features = json.loads(features_path.read_text(encoding="utf-8"))
            features["feature_names_by_modality"]["sensor"][0] = "renamed_sensor"
            _write_json(features_path, features)

            self._assert_all_loaders_reject(
                root, "feature-bundle fingerprint binding failed"
            )


if __name__ == "__main__":
    unittest.main()
