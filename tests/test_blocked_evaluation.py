from __future__ import annotations

import unittest

try:
    import numpy as np
    import sklearn  # noqa: F401

    RECOGNITION_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency environment
    np = None
    RECOGNITION_DEPS_AVAILABLE = False

from teaching_skill_miner.recognition.blocked_evaluation import (
    BlockedEvaluationError,
    blocked_descriptive_evaluation,
)


def _fixture(
    *,
    omit_cell: tuple[int, int] | None = None,
    class_two_only_in_cohort_one: bool = False,
) -> tuple[list[dict], "np.ndarray", "np.ndarray", list[str]]:
    records: list[dict] = []
    visual: list[list[float]] = []
    sensor: list[list[float]] = []
    for cohort in range(1, 4):
        for activity in range(1, 10):
            if omit_cell == (cohort, activity):
                continue
            classes = (0, 1, 2)
            if class_two_only_in_cohort_one and cohort != 1:
                classes = (0, 1)
            for label in classes:
                sample_id = f"c{cohort}-a{activity}-y{label}"
                record = {
                    "sample_id": sample_id,
                    "label": label,
                    "cohort_id": f"cohort-{cohort}",
                    "site_id": "alicante-site",
                    "session_id": f"cohort-{cohort}/activity-{activity:02d}",
                }
                if activity % 2:
                    record["activity_id"] = f"activity-{activity:02d}"
                else:
                    record["source"] = {"experiment_id": f"activity-{activity:02d}"}
                records.append(record)
                # Each branch is deliberately a valid, simple fixed feature set.
                visual.append([float(label == 0), float(label == 1), cohort / 100.0])
                sensor.append([float(label == 2), float(label == 1), activity / 100.0])
    return (
        records,
        np.asarray(visual, dtype=float),
        np.asarray(sensor, dtype=float),
        [record["sample_id"] for record in records],
    )


@unittest.skipUnless(RECOGNITION_DEPS_AVAILABLE, "recognition dependencies are optional")
class BlockedDescriptiveEvaluationTests(unittest.TestCase):
    def test_runs_all_fixed_folds_and_reports_paired_modalities(self) -> None:
        records, visual, sensor, sample_ids = _fixture()

        report = blocked_descriptive_evaluation(
            records,
            visual,
            sensor,
            sample_ids=sample_ids,
        )

        self.assertEqual(report["evaluation_kind"], "single_site_descriptive_blocked_oof")
        self.assertFalse(report["contains_inferential_statistics"])
        self.assertTrue(report["dataset_summary"]["feature_sample_order_verified"])
        self.assertEqual(report["fixed_model_protocol"]["C"], 1.0)
        self.assertEqual(report["fixed_model_protocol"]["hyperparameter_selection"], "none")
        expected = {
            "leave_one_cohort_out": 3,
            "leave_one_activity_out": 9,
            "double_blocked_cohort_activity": 27,
        }
        self.assertEqual(report["expected_fold_counts"], expected)
        for name, fold_count in expected.items():
            evaluation = report["evaluations"][name]
            self.assertEqual(evaluation["fold_count"], fold_count)
            self.assertEqual(evaluation["evaluated_fold_count"], fold_count)
            self.assertEqual(evaluation["unevaluable_folds"], [])
            self.assertEqual(set(evaluation["modalities"]), {"visual", "sensor", "fusion"})
            for modality in ("visual", "sensor", "fusion"):
                modality_report = evaluation["modalities"][modality]
                self.assertEqual(modality_report["oof_coverage"], 1.0)
                self.assertEqual(modality_report["oof_sample_ids"], sample_ids)
                self.assertIsNotNone(
                    modality_report["pooled_oof_sample_weighted"]["macro_f1"]
                )
                self.assertIsNotNone(
                    modality_report["test_cell_session_equal_weighted"]["macro_f1"]
                )
                self.assertEqual(
                    modality_report["test_cell_session_equal_weighted"][
                        "evaluated_cell_count"
                    ],
                    27,
                )
            self.assertTrue(
                all(fold["status"] == "evaluated" for fold in evaluation["folds"])
            )
            self.assertTrue(
                all(not fold["session_overlap_detected"] for fold in evaluation["folds"])
            )
        self.assertTrue(
            all(value is False for value in report["claim_status"].values())
        )

    def test_double_block_excludes_same_cohort_or_activity_from_training(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        report = blocked_descriptive_evaluation(
            records, visual, sensor, sample_ids=sample_ids
        )

        folds = report["evaluations"]["double_blocked_cohort_activity"]["folds"]
        self.assertEqual(len(folds), 27)
        for fold in folds:
            held_cohort = fold["test_block"]["cohort_id"]
            held_activity = fold["test_block"]["activity_id"]
            self.assertNotIn(held_cohort, fold["train_cohort_ids"])
            self.assertNotIn(held_activity, fold["train_activity_ids"])
            self.assertGreater(fold["embargo_sample_count"], 0)

    def test_training_missing_a_global_class_is_disclosed_not_scored(self) -> None:
        records, visual, sensor, sample_ids = _fixture(
            class_two_only_in_cohort_one=True
        )
        report = blocked_descriptive_evaluation(
            records, visual, sensor, sample_ids=sample_ids
        )

        evaluation = report["evaluations"]["leave_one_cohort_out"]
        unavailable = {
            item["fold_id"]: item for item in evaluation["unevaluable_folds"]
        }
        self.assertIn("cohort=cohort-1", unavailable)
        self.assertEqual(
            unavailable["cohort=cohort-1"]["reason"],
            "training_missing_global_classes",
        )
        self.assertEqual(unavailable["cohort=cohort-1"]["missing_training_classes"], [2])
        self.assertLess(evaluation["modalities"]["fusion"]["oof_coverage"], 1.0)

    def test_empty_cartesian_test_cell_remains_one_of_27_folds(self) -> None:
        records, visual, sensor, sample_ids = _fixture(omit_cell=(3, 9))
        report = blocked_descriptive_evaluation(
            records, visual, sensor, sample_ids=sample_ids
        )

        evaluation = report["evaluations"]["double_blocked_cohort_activity"]
        self.assertEqual(evaluation["fold_count"], 27)
        empty = [
            fold
            for fold in evaluation["unevaluable_folds"]
            if fold["reason"] == "empty_test_block"
        ]
        self.assertEqual([item["fold_id"] for item in empty], ["cohort=cohort-3|activity=activity-09"])

    def test_feature_sample_order_mismatch_fails_closed(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        sample_ids[0], sample_ids[1] = sample_ids[1], sample_ids[0]

        with self.assertRaisesRegex(BlockedEvaluationError, "record order"):
            blocked_descriptive_evaluation(
                records, visual, sensor, sample_ids=sample_ids
            )

    def test_feature_row_count_and_finite_values_are_checked(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        with self.assertRaisesRegex(BlockedEvaluationError, "sensor features must have shape"):
            blocked_descriptive_evaluation(
                records, visual, sensor[:-1], sample_ids=sample_ids
            )
        sensor[0, 0] = np.nan
        with self.assertRaisesRegex(BlockedEvaluationError, "NaN or infinity"):
            blocked_descriptive_evaluation(
                records, visual, sensor, sample_ids=sample_ids
            )

    def test_protocol_rejects_hyperparameter_tuning_and_multiple_sites(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        with self.assertRaisesRegex(BlockedEvaluationError, "protocol-fixed"):
            blocked_descriptive_evaluation(
                records,
                visual,
                sensor,
                sample_ids=sample_ids,
                logistic_c=0.1,
            )

        records[-1]["site_id"] = "another-site"
        with self.assertRaisesRegex(BlockedEvaluationError, "one documented site"):
            blocked_descriptive_evaluation(
                records, visual, sensor, sample_ids=sample_ids
            )

    def test_requires_exact_dipser_three_by_nine_design(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        keep = [
            index
            for index, record in enumerate(records)
            if record["cohort_id"] != "cohort-3"
        ]
        with self.assertRaisesRegex(BlockedEvaluationError, "exactly 3 cohorts"):
            blocked_descriptive_evaluation(
                [records[index] for index in keep],
                visual[keep],
                sensor[keep],
                sample_ids=[sample_ids[index] for index in keep],
            )

    def test_session_identity_must_map_one_to_one_with_observed_design_cells(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        records[0]["session_id"] = records[3]["session_id"]

        with self.assertRaisesRegex(BlockedEvaluationError, "multiple cohort/activity"):
            blocked_descriptive_evaluation(
                records, visual, sensor, sample_ids=sample_ids
            )


if __name__ == "__main__":
    unittest.main()
