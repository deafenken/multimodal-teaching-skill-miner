from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency environment
    np = None
    NUMPY_AVAILABLE = False

try:
    import sklearn  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency environment
    SKLEARN_AVAILABLE = False

from teaching_skill_miner.recognition.datasets import _filename_group, discover_ouc_cge
from teaching_skill_miner.recognition.experiment import (
    _grouped_oof,
    _provenance_scope,
    infer_classroom_engagement,
    run_real_classroom_benchmark,
)
from teaching_skill_miner.recognition.metrics import (
    classification_metrics,
    grouped_bootstrap_interval,
)


@unittest.skipUnless(NUMPY_AVAILABLE and SKLEARN_AVAILABLE, "recognition dependencies are optional")
class RecognitionMetricTests(unittest.TestCase):
    def test_classification_metrics_match_known_confusion_matrix(self) -> None:
        labels = np.asarray([0, 1, 2, 0])
        probabilities = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.2, 0.6, 0.2],
                [0.7, 0.2, 0.1],
            ]
        )

        metrics = classification_metrics(
            labels,
            probabilities,
            class_names=["low", "medium", "high"],
        )

        self.assertEqual(metrics["sample_count"], 4)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["balanced_accuracy"], 0.6667)
        self.assertEqual(metrics["macro_f1"], 0.5556)
        self.assertEqual(metrics["weighted_f1"], 0.6667)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0, 0], [0, 1, 0], [0, 1, 0]])
        self.assertEqual(metrics["per_class"]["low"]["support"], 2)
        self.assertEqual(metrics["per_class"]["high"]["recall"], 0.0)
        self.assertGreater(metrics["log_loss"], 0.0)
        self.assertGreaterEqual(metrics["expected_calibration_error_10_bins"], 0.0)

    def test_grouped_bootstrap_is_reproducible_and_names_its_unit(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        probabilities = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.1, 0.8],
                [0.1, 0.2, 0.7],
            ]
        )
        groups = np.asarray(["g0", "g0", "g1", "g1", "g2", "g2"])

        first = grouped_bootstrap_interval(
            labels,
            probabilities,
            groups,
            seed=17,
            replicates=40,
        )
        second = grouped_bootstrap_interval(
            labels,
            probabilities,
            groups,
            seed=17,
            replicates=40,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_unit"], "filename_index_surrogate_group")
        self.assertEqual(first["replicates"], 40)
        self.assertEqual(first["accuracy_95_ci"], [1.0, 1.0])


@unittest.skipUnless(NUMPY_AVAILABLE and SKLEARN_AVAILABLE, "recognition dependencies are optional")
class RecognitionSplitTests(unittest.TestCase):
    def test_grouped_oof_never_passes_a_group_to_train_and_test(self) -> None:
        labels = np.repeat(np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2]), 2)
        groups = np.repeat(np.arange(9), 2)
        matrix = np.column_stack([groups, np.arange(len(groups), dtype=float)])
        observed_splits: list[tuple[set[int], set[int]]] = []

        def fake_fit_predict(train_x, train_y, test_x):
            del train_y
            train_groups = {int(value) for value in train_x[:, 0]}
            test_groups = {int(value) for value in test_x[:, 0]}
            observed_splits.append((train_groups, test_groups))
            self.assertFalse(train_groups & test_groups)
            probabilities = np.tile(np.asarray([[0.34, 0.33, 0.33]]), (len(test_x), 1))
            return probabilities, {}

        with patch(
            "teaching_skill_miner.recognition.experiment._fit_predict",
            side_effect=fake_fit_predict,
        ):
            probabilities, folds = _grouped_oof(
                matrix,
                labels,
                groups,
                folds=3,
                seed=2026,
            )

        self.assertEqual(probabilities.shape, (18, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(len(observed_splits), 3)
        self.assertEqual(len(folds), 3)
        self.assertTrue(all(fold["group_overlap"] == [] for fold in folds))
        self.assertTrue(all(fold["train_count"] + fold["test_count"] == 18 for fold in folds))


class RecognitionDatasetAuditTests(unittest.TestCase):
    def test_filename_group_is_label_independent_and_normalized(self) -> None:
        self.assertEqual(_filename_group("view1"), "view_000001")
        self.assertEqual(_filename_group("anything_000012"), "view_000012")
        self.assertEqual(_filename_group("NoNumber"), "stem_nonumber")

    def test_discovery_removes_exact_duplicate_bytes_without_real_video_tools(self) -> None:
        fake_probe = {
            "valid": True,
            "duration_seconds": 10.0,
            "size_bytes": 8,
            "has_audio": True,
            "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ("low", "mid", "high"):
                (root / label).mkdir()
            (root / "low" / "view1.mp4").write_bytes(b"duplicate-video")
            (root / "low" / "view2.mp4").write_bytes(b"duplicate-video")
            (root / "mid" / "view3.mp4").write_bytes(b"unique-medium")
            (root / "high" / "view4.mp4").write_bytes(b"unique-high")

            with patch(
                "teaching_skill_miner.recognition.datasets.probe_video",
                return_value=fake_probe,
            ):
                manifest = discover_ouc_cge(root)

        audit = manifest["audit"]
        self.assertEqual(audit["raw_file_count"], 4)
        self.assertEqual(audit["usable_unique_count"], 3)
        self.assertEqual(audit["exact_duplicate_count"], 1)
        self.assertEqual(len(manifest["records"]), 3)
        self.assertEqual(audit["label_counts"], {"high": 1, "low": 1, "medium": 1})
        self.assertEqual(audit["surrogate_group_count"], 3)
        self.assertFalse(audit["provenance_verified"])
        self.assertFalse(audit["real_classroom_video"])
        self.assertFalse(audit["independent_human_ground_truth"])
        self.assertFalse(audit["sample_release_is_final_benchmark_split"])
        retained_hashes = [record["video_sha256"] for record in manifest["records"]]
        self.assertEqual(len(retained_hashes), len(set(retained_hashes)))


@unittest.skipUnless(NUMPY_AVAILABLE and SKLEARN_AVAILABLE, "recognition dependencies are optional")
class RecognitionReportSemanticsTests(unittest.TestCase):
    def test_unverified_directory_scope_never_claims_official_data(self) -> None:
        verified = _provenance_scope({"provenance_verified": True})
        unverified = _provenance_scope({"provenance_verified": False})

        self.assertIn("official public-sample", verified["training_scope"])
        self.assertIn("unverified", unverified["training_scope"])
        self.assertIn("unverified", unverified["metric_scope"])
        self.assertIn("unverified", unverified["dataset_limitation"])
        self.assertNotIn("verified OUC-CGE public sample", unverified["inference_warning"])

    def test_pilot_report_does_not_claim_deployment_or_causal_accuracy(self) -> None:
        records = [
            {
                "sample_id": f"sample-{index}",
                "group_id": f"group-{index // 3}",
                "label": index % 3,
                "label_name": ("low", "medium", "high")[index % 3],
                "relative_path": f"unused-{index}.mp4",
                "video_sha256": f"{index:064x}",
            }
            for index in range(9)
        ]
        manifest = {
            "schema_version": "1.0",
            "records": records,
            "audit": {
                "dataset_id": "OUC-CGE",
                "dataset_variant": "public_sample",
                "dataset_fingerprint": "a" * 64,
                "provenance_verified": True,
                "real_classroom_video": True,
                "independent_human_ground_truth": True,
                "invalid_video_count": 0,
                "exact_duplicate_count": 0,
                "raw_file_count": 9,
            },
        }
        matrix = np.arange(36, dtype=float).reshape(9, 4)

        def fake_grouped_oof(selected, selected_labels, selected_groups, *, folds, seed):
            del selected, selected_groups, seed
            probabilities = np.full((len(selected_labels), 3), 0.05)
            probabilities[np.arange(len(selected_labels)), selected_labels] = 0.9
            fold_reports = [
                {
                    "fold": fold,
                    "train_count": 6,
                    "test_count": 3,
                    "train_group_count": 2,
                    "test_group_count": 1,
                    "group_overlap": [],
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                }
                for fold in range(1, folds + 1)
            ]
            return probabilities, fold_reports

        final_state = {
            "scaler_mean": [0.0] * 4,
            "scaler_scale": [1.0] * 4,
            "coefficient": [[0.0] * 4 for _ in range(3)],
            "intercept": [0.0] * 3,
            "classes": [0, 1, 2],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            with (
                patch(
                    "teaching_skill_miner.recognition.experiment.discover_ouc_cge",
                    return_value=manifest,
                ),
                patch(
                    "teaching_skill_miner.recognition.experiment._feature_matrix",
                    return_value=(matrix, ["v0", "v1", "a0", "a1"], 2, []),
                ),
                patch(
                    "teaching_skill_miner.recognition.experiment._grouped_oof",
                    side_effect=fake_grouped_oof,
                ),
                patch(
                    "teaching_skill_miner.recognition.experiment._train_final_model",
                    return_value=final_state,
                ),
                patch(
                    "teaching_skill_miner.recognition.experiment.grouped_bootstrap_interval",
                    return_value={
                        "accuracy_95_ci": [1.0, 1.0],
                        "macro_f1_95_ci": [1.0, 1.0],
                        "bootstrap_unit": "filename_index_surrogate_group",
                        "replicates": 10,
                    },
                ),
            ):
                report = run_real_classroom_benchmark(
                    Path(directory) / "unused-dataset",
                    output,
                    folds=3,
                )
            persisted = json.loads((output / "benchmark_report.json").read_text(encoding="utf-8"))

        self.assertEqual(report, persisted)
        self.assertEqual(
            report["benchmark_kind"],
            "real_classroom_human_labeled_surrogate_group_oof_pilot",
        )
        self.assertTrue(report["real_classroom_video_used"])
        self.assertTrue(report["independent_human_ground_truth"])
        self.assertFalse(report["group_disjoint_evaluation"])
        self.assertTrue(report["surrogate_filename_group_disjoint_evaluation"])
        self.assertFalse(report["verified_source_group_disjoint_evaluation"])
        self.assertFalse(report["session_disjoint_evaluation"])
        self.assertFalse(report["full_official_test_set_used"])
        self.assertFalse(report["real_world_recognition_accuracy_established"])
        self.assertFalse(report["teaching_effectiveness_established"])
        self.assertTrue(report["automatic_model_inference_implemented"])
        self.assertEqual(report["modality_results"]["fusion"]["accuracy"], 1.0)
        self.assertFalse(report["multimodal_gain_established_on_pilot"])
        self.assertEqual(report["fusion_macro_f1_gain_over_visual"], 0.0)
        self.assertTrue(report["quality_gates"]["all_samples_have_oof_predictions"])
        self.assertTrue(report["quality_gates"]["zero_surrogate_group_overlap_each_fold"])
        self.assertEqual(set(report["modality_results"]), {"visual", "audio", "fusion"})
        self.assertTrue(any("production-calibrated" in item for item in report["limitations"]))

    def test_inference_validates_checkpoint_hash_and_feature_order(self) -> None:
        checkpoint = {
            "schema_version": "1.0",
            "task": "test",
            "classes": ["low", "medium", "high"],
            "model_classes": [0, 1, 2],
            "modality": "visual",
            "feature_names": ["visual_only"],
            "feature_configuration": {
                "frame_count": 1,
                "width": 1,
                "height": 1,
                "clip_seconds": 1.0,
                "sample_rate": 8000,
            },
            "scaler_mean": [0.0],
            "scaler_scale": [1.0],
            "coefficient": [[-1.0], [0.0], [1.0]],
            "intercept": [0.0, 0.0, 0.0],
            "training_video_sha256s": [],
        }
        checkpoint["checkpoint_sha256"] = hashlib.sha256(
            json.dumps(checkpoint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        extracted = {
            "features": np.asarray([2.0, 0.0]),
            "feature_names": ["visual_only", "audio_present"],
            "visual_dimension": 1,
            "audio_present": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"fixture")
            model = Path(directory) / "checkpoint.json"
            model.write_text(json.dumps(checkpoint), encoding="utf-8")
            with (
                patch(
                    "teaching_skill_miner.recognition.experiment.probe_video",
                    return_value={
                        "valid": True,
                        "duration_seconds": 1.0,
                        "video_start_time_seconds": 0.0,
                    },
                ),
                patch(
                    "teaching_skill_miner.recognition.experiment.extract_multimodal_features",
                    return_value=extracted,
                ),
            ):
                prediction = infer_classroom_engagement(video, model)
                self.assertEqual(prediction["predicted_label"], "high")

                tampered = dict(checkpoint)
                tampered["intercept"] = [3.0, 0.0, 0.0]
                model.write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                    infer_classroom_engagement(video, model)


if __name__ == "__main__":
    unittest.main()
