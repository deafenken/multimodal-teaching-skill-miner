from __future__ import annotations

import unittest

try:
    import numpy as np
    import sklearn  # noqa: F401

    RECOGNITION_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency environment
    np = None
    RECOGNITION_DEPS_AVAILABLE = False

from teaching_skill_miner.recognition.optimized_evaluation import (
    DOUBLE_BLOCK_DESIGN,
    LOAO_DESIGN,
    LOCO_DESIGN,
    SESSION_DESIGN,
    OptimizedEvaluationError,
    candidate_registry,
    optimized_blocked_evaluation,
    optimized_multimodal_evaluation,
    optimized_session_grouped_evaluation,
)


TEST_CANDIDATES = ("majority", "logreg_unweighted_c0.3")


def _fixture() -> tuple[list[dict], "np.ndarray", "np.ndarray", list[str]]:
    records: list[dict] = []
    visual: list[list[float]] = []
    sensor: list[list[float]] = []
    for cohort in range(1, 3):
        for activity in range(1, 6):
            session = f"cohort-{cohort}/activity-{activity}"
            for label in range(3):
                sample_id = f"c{cohort}-a{activity}-y{label}"
                record = {
                    "sample_id": sample_id,
                    "label": label,
                    "session_id": session,
                    "participant_id": f"participant-{cohort}-{activity}",
                    "teacher_id": f"teacher-{cohort}",
                    "cohort_id": f"cohort-{cohort}",
                    "site_id": "single-site",
                    # These nuisance fields must never become model features.
                    "archive_path": f"/forbidden/label-{label}.zip",
                    "annotator_label": label,
                }
                if activity % 2:
                    record["activity_id"] = f"activity-{activity}"
                else:
                    record["source"] = {"experiment_id": f"activity-{activity}"}
                records.append(record)
                # Each modality collapses one pair; early fusion separates all 3.
                visual.append(
                    [4.0 if label == 0 else 0.0, cohort / 100.0]
                )
                sensor.append(
                    [4.0 if label == 2 else 0.0, activity / 100.0]
                )
    return (
        records,
        np.asarray(visual, dtype=float),
        np.asarray(sensor, dtype=float),
        [str(record["sample_id"]) for record in records],
    )


@unittest.skipUnless(RECOGNITION_DEPS_AVAILABLE, "recognition dependencies are optional")
class OptimizedEvaluationTests(unittest.TestCase):
    def test_session_nested_oof_beats_fold_training_majority_without_leakage(self) -> None:
        records, visual, sensor, sample_ids = _fixture()

        report = optimized_session_grouped_evaluation(
            records,
            visual,
            sensor,
            sample_ids,
            candidate_ids=TEST_CANDIDATES,
            inner_splits=3,
        )

        self.assertEqual(report["protocol"], "optimized_nested_grouped_multimodal_oof_v1")
        self.assertEqual(report["designs"], [SESSION_DESIGN])
        self.assertTrue(report["feature_sample_order_verified"])
        self.assertTrue(report["candidate_set_frozen_before_outer_scoring"])
        self.assertTrue(report["all_preprocessing_fitted_inside_training_partitions"])
        self.assertFalse(
            report["outer_test_labels_used_for_model_or_hyperparameter_selection"]
        )
        self.assertFalse(report["feature_policy"]["records_converted_to_features"])
        self.assertTrue(all(value is False for value in report["claim_status"].values()))

        evaluation = report["evaluations"][SESSION_DESIGN]
        self.assertEqual(evaluation["fold_count"], 5)
        self.assertEqual(evaluation["evaluated_fold_count"], 5)
        self.assertEqual(evaluation["oof_coverage"], 1.0)
        self.assertGreater(
            evaluation["metrics"]["fusion"]["accuracy"],
            evaluation["metrics"]["majority_baseline"]["accuracy"],
        )
        self.assertEqual(len(evaluation["oof_predictions"]), len(records))
        self.assertTrue(
            all(row["evaluated"] for row in evaluation["oof_predictions"])
        )
        for fold in evaluation["folds"]:
            self.assertEqual(fold["status"], "evaluated")
            self.assertEqual(fold["sample_id_overlap"], [])
            self.assertEqual(
                fold["identity_overlap_audit"]["session_id"]["overlap_ids"], []
            )
            self.assertFalse(
                fold["outer_test_labels_used_for_model_or_hyperparameter_selection"]
            )
            self.assertIn(
                fold["selected_unimodal_from_inner_training_scores"],
                {"visual", "sensor"},
            )
            for modality in ("visual", "sensor", "fusion"):
                selection = fold["modality_selection"][modality]
                selected = [
                    candidate
                    for candidate in selection["candidate_reports"]
                    if candidate["selected"]
                ]
                self.assertEqual(len(selected), 1)
                for candidate in selection["candidate_reports"]:
                    if candidate["status"] == "evaluated":
                        self.assertTrue(
                            all(
                                inner["session_overlap_ids"] == []
                                for inner in candidate["folds"]
                            )
                        )

    def test_all_fixed_blocked_designs_have_expected_disjoint_audits(self) -> None:
        records, visual, sensor, sample_ids = _fixture()

        report = optimized_blocked_evaluation(
            records,
            visual,
            sensor,
            sample_ids,
            candidate_ids=TEST_CANDIDATES,
            inner_splits=3,
        )

        expected_counts = {
            LOCO_DESIGN: 2,
            LOAO_DESIGN: 5,
            DOUBLE_BLOCK_DESIGN: 10,
        }
        self.assertEqual(set(report["evaluations"]), set(expected_counts))
        for design, count in expected_counts.items():
            evaluation = report["evaluations"][design]
            self.assertEqual(evaluation["fold_count"], count)
            self.assertEqual(evaluation["evaluated_fold_count"], count)
            self.assertEqual(evaluation["oof_coverage"], 1.0)
            self.assertTrue(
                all(value is False for value in evaluation["claim_status"].values())
            )
            for fold in evaluation["folds"]:
                self.assertEqual(fold["status"], "evaluated")
                for field in fold["expected_disjoint_fields"]:
                    self.assertEqual(
                        fold["identity_overlap_audit"][field]["overlap_ids"], []
                    )
                # Single-site overlap is disclosed rather than misrepresented.
                self.assertEqual(
                    fold["identity_overlap_audit"]["site_id"]["overlap_ids"],
                    ["single-site"],
                )

        for fold in report["evaluations"][DOUBLE_BLOCK_DESIGN]["folds"]:
            held = fold["test_block"]
            cohort_audit = fold["identity_overlap_audit"]["cohort_id"]
            activity_audit = fold["identity_overlap_audit"]["activity_id"]
            self.assertNotIn(held["cohort_id"], cohort_audit["train_values"])
            self.assertNotIn(held["activity_id"], activity_audit["train_values"])
            self.assertGreater(fold["embargo_sample_count"], 0)

    def test_main_interface_can_run_all_four_designs(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        report = optimized_multimodal_evaluation(
            records,
            visual,
            sensor,
            sample_ids,
            designs=(SESSION_DESIGN, LOCO_DESIGN, LOAO_DESIGN, DOUBLE_BLOCK_DESIGN),
            candidate_ids=TEST_CANDIDATES,
            inner_splits=3,
        )
        self.assertEqual(
            set(report["evaluations"]),
            {SESSION_DESIGN, LOCO_DESIGN, LOAO_DESIGN, DOUBLE_BLOCK_DESIGN},
        )
        self.assertFalse(report["contains_inferential_statistics"])

    def test_exact_feature_order_dimensions_and_finite_values_fail_closed(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        reversed_ids = list(reversed(sample_ids))
        with self.assertRaisesRegex(OptimizedEvaluationError, "record order"):
            optimized_session_grouped_evaluation(
                records,
                visual,
                sensor,
                reversed_ids,
                candidate_ids=TEST_CANDIDATES,
            )
        with self.assertRaisesRegex(OptimizedEvaluationError, "must have shape"):
            optimized_session_grouped_evaluation(
                records,
                visual[:-1],
                sensor,
                sample_ids,
                candidate_ids=TEST_CANDIDATES,
            )
        sensor[0, 0] = np.nan
        with self.assertRaisesRegex(OptimizedEvaluationError, "NaN or infinity"):
            optimized_session_grouped_evaluation(
                records,
                visual,
                sensor,
                sample_ids,
                candidate_ids=TEST_CANDIDATES,
            )

    def test_candidate_registry_and_fixed_session_fold_count_are_enforced(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        registry = candidate_registry()
        self.assertIn("rbf_svc_balanced_c2", registry)
        self.assertIn("extra_trees_balanced_leaf2", registry)
        self.assertEqual(
            registry["linear_svc_unweighted_c0.1"],
            {"family": "linear_svc", "C": 0.1, "class_weight": None},
        )
        with self.assertRaisesRegex(OptimizedEvaluationError, "unknown candidate"):
            optimized_session_grouped_evaluation(
                records,
                visual,
                sensor,
                sample_ids,
                candidate_ids=("invented_after_seeing_outer_scores",),
            )
        with self.assertRaisesRegex(OptimizedEvaluationError, "fixed at five"):
            optimized_session_grouped_evaluation(
                records,
                visual,
                sensor,
                sample_ids,
                outer_splits=4,
                candidate_ids=TEST_CANDIDATES,
            )

    def test_frozen_linear_svc_candidate_builds_and_scores_inside_nested_folds(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        candidate_id = "linear_svc_unweighted_c0.1"

        report = optimized_session_grouped_evaluation(
            records,
            visual,
            sensor,
            sample_ids,
            candidate_ids=(candidate_id,),
            inner_splits=3,
        )

        evaluation = report["evaluations"][SESSION_DESIGN]
        self.assertEqual(evaluation["evaluated_fold_count"], 5)
        for fold in evaluation["folds"]:
            for modality in ("visual", "sensor", "fusion"):
                selection = fold["modality_selection"][modality]
                self.assertEqual(selection["selected_candidate_id"], candidate_id)
                self.assertEqual(
                    selection["candidate_reports"][0]["status"], "evaluated"
                )

    def test_conflicting_session_design_identity_is_rejected(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        records[3]["session_id"] = records[0]["session_id"]
        with self.assertRaisesRegex(
            OptimizedEvaluationError, "maps to multiple cohort/activity/site"
        ):
            optimized_session_grouped_evaluation(
                records,
                visual,
                sensor,
                sample_ids,
                candidate_ids=TEST_CANDIDATES,
            )

    def test_optional_identity_missing_is_audited_not_invented(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        for record in records:
            record.pop("teacher_id")
        report = optimized_session_grouped_evaluation(
            records,
            visual,
            sensor,
            sample_ids,
            candidate_ids=TEST_CANDIDATES,
        )
        self.assertFalse(
            report["dataset_design_summary"]["optional_identity_completeness"][
                "teacher_id"
            ]
        )
        for fold in report["evaluations"][SESSION_DESIGN]["folds"]:
            self.assertFalse(
                fold["identity_overlap_audit"]["teacher_id"]["field_complete"]
            )
            self.assertEqual(
                fold["identity_overlap_audit"]["teacher_id"]["overlap_ids"], []
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
