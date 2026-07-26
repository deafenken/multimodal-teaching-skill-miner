from __future__ import annotations

import copy
import unittest

try:
    import numpy as np
    import sklearn  # noqa: F401

    RECOGNITION_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency environment
    np = None
    RECOGNITION_DEPS_AVAILABLE = False

from teaching_skill_miner.recognition.hierarchical_evaluation import (
    GATE_REGISTRY,
    HierarchicalEvaluationError,
    _apply_gate,
    _fit_component_predictions,
    _majority_label,
    _prepare_inputs,
    fixed_hierarchical_stress_evaluation,
    hierarchical_offline_evaluation,
)


def _fixture() -> tuple[list[dict], "np.ndarray", "np.ndarray", list[str]]:
    records: list[dict] = []
    visual: list[list[float]] = []
    sensor: list[list[float]] = []
    # Fifteen independent sessions provide five session groups per class.
    for session_index in range(15):
        label = session_index % 3
        for participant_index in range(2):
            participant = f"session-{session_index:02d}/p{participant_index}"
            for window_index in range(2):
                sample_id = (
                    f"s{session_index:02d}-p{participant_index}-w{window_index}"
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "label": label,
                        "session_id": f"session-{session_index:02d}",
                        "participant_id": participant,
                        "teacher_id": f"teacher-{session_index % 2}",
                        "cohort_id": f"cohort-{session_index % 3}",
                        "activity_id": f"activity-{session_index % 5}",
                        "site_id": "test-site",
                    }
                )
                jitter = (participant_index * 2 + window_index) / 1000.0
                visual.append(
                    [
                        float(label == 0) + jitter,
                        float(label == 1) - jitter,
                        float(label == 2) + jitter,
                    ]
                )
                sensor.append(
                    [
                        2.0 * float(label == 0) - jitter,
                        2.0 * float(label == 1) + jitter,
                        2.0 * float(label == 2) - jitter,
                    ]
                )
    return (
        records,
        np.asarray(visual, dtype=float),
        np.asarray(sensor, dtype=float),
        [record["sample_id"] for record in records],
    )


@unittest.skipUnless(RECOGNITION_DEPS_AVAILABLE, "recognition dependencies are optional")
class HierarchicalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records, cls.visual, cls.sensor, cls.sample_ids = _fixture()
        cls.report = hierarchical_offline_evaluation(
            cls.records,
            cls.visual,
            cls.sensor,
            sample_ids=cls.sample_ids,
        )

    def test_protocol_is_nested_but_all_claims_remain_exploratory(self) -> None:
        report = self.report
        self.assertEqual(
            report["evaluation_kind"],
            "nested_post_selected_offline_full_session_transductive_oof",
        )
        self.assertTrue(report["inner_components_refit_per_fold"])
        self.assertFalse(report["outer_test_labels_used_for_training_or_gate_selection"])
        self.assertTrue(report["outer_labels_used_for_stratified_split_construction"])
        self.assertTrue(report["offline_full_sequence_required"])
        self.assertTrue(report["offline_full_session_required"])
        self.assertTrue(report["uses_unlabeled_test_session_context"])
        self.assertTrue(report["uses_future_feature_rows"])
        self.assertTrue(report["cross_participant_context"])
        self.assertFalse(report["uses_future_labels"])
        self.assertFalse(report["real_time_compatible"])
        self.assertFalse(report["single_participant_inference_compatible"])
        self.assertFalse(report["sample_count_used_as_model_feature_or_gate"])
        self.assertFalse(report["identity_values_enter_numeric_estimator"])
        self.assertTrue(report["identities_used_as_aggregation_boundaries"])
        self.assertEqual(
            report["aggregation_boundaries"],
            [["session_id", "participant_id"], ["session_id"]],
        )
        self.assertTrue(all(value is False for value in report["claim_status"].values()))
        self.assertEqual(len(report["protocol"]["gate_registry"]), 10)

    def test_outer_and_inner_folds_are_session_disjoint(self) -> None:
        for fold in self.report["folds"]:
            self.assertEqual(fold["session_overlap_ids"], [])
            self.assertEqual(fold["participant_overlap_ids"], [])
            self.assertTrue(
                set(fold["train_session_ids"]).isdisjoint(fold["test_session_ids"])
            )
            selection = fold["inner_gate_selection"]
            self.assertTrue(selection["inner_components_refit_per_fold"])
            self.assertFalse(selection["outer_test_labels_observed_during_selection"])
            self.assertEqual(len(selection["candidate_reports"]), len(GATE_REGISTRY))
            self.assertEqual(
                sum(bool(row["selected"]) for row in selection["candidate_reports"]),
                1,
            )
            for inner_fold in selection["inner_folds"]:
                self.assertEqual(inner_fold["session_overlap_ids"], [])

    def test_reported_metrics_are_recomputed_from_oof_rows(self) -> None:
        labels = np.asarray([row["label"] for row in self.report["oof_predictions"]])
        predictions = np.asarray(
            [
                row["nested_hierarchical_prediction"]
                for row in self.report["oof_predictions"]
            ]
        )
        self.assertAlmostEqual(
            self.report["nested_hierarchical_metrics"]["accuracy"],
            float((labels == predictions).mean()),
            places=6,
        )
        self.assertEqual(len(self.report["oof_predictions"]), len(self.records))
        self.assertEqual(
            {row["sample_id"] for row in self.report["oof_predictions"]},
            set(self.sample_ids),
        )

    def test_outer_test_labels_do_not_change_fitted_component_predictions(self) -> None:
        data = _prepare_inputs(
            self.records, self.visual, self.sensor, self.sample_ids
        )
        sessions = data["identities"]["session_id"]
        train_mask = np.asarray(
            [int(str(value).split("-")[-1]) < 12 for value in sessions], dtype=bool
        )
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(~train_mask)
        first = _fit_component_predictions(data, train_indices, test_indices)

        changed = copy.deepcopy(data)
        changed["labels"] = data["labels"].copy()
        changed["labels"][test_indices] = (
            changed["labels"][test_indices] + 1
        ) % 3
        second = _fit_component_predictions(changed, train_indices, test_indices)
        for field in (
            "window_logistic_probabilities",
            "window_linear_svc_predictions",
            "sequence_base_predictions",
            "sequence_svc_votes",
            "session_probabilities",
            "session_predictions",
        ):
            np.testing.assert_allclose(first[field], second[field])

    def test_ties_and_gate_order_are_deterministic(self) -> None:
        self.assertEqual(_majority_label([0, 1]), 1)
        self.assertEqual(_majority_label([0, 2]), 0)
        components = {
            "sequence_base_predictions": np.asarray([0, 1, 2, 2]),
            "sequence_svc_votes": np.asarray([1, 1, 1, 2]),
            "session_predictions": np.asarray([1, 2, 1, 1]),
            "session_probabilities": np.asarray(
                [
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8],
                    [0.1, 0.6, 0.3],
                    [0.1, 0.6, 0.3],
                ]
            ),
        }
        candidate = {
            "low_confirmation": True,
            "session_high_override": True,
            "ambiguous_high_ratio_threshold": 0.5,
        }
        # low is reset, medium is raised by session context, only the high not
        # confirmed by linear SVC is reset by the final probability guard.
        np.testing.assert_array_equal(
            _apply_gate(components, candidate), np.asarray([1, 2, 1, 2])
        )

    def test_order_nonfinite_and_repeated_participant_fail_closed(self) -> None:
        wrong_order = list(self.sample_ids)
        wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
        with self.assertRaisesRegex(HierarchicalEvaluationError, "record order"):
            hierarchical_offline_evaluation(
                self.records,
                self.visual,
                self.sensor,
                sample_ids=wrong_order,
            )

        bad_sensor = self.sensor.copy()
        bad_sensor[0, 0] = np.nan
        with self.assertRaisesRegex(HierarchicalEvaluationError, "NaN or infinity"):
            hierarchical_offline_evaluation(
                self.records,
                self.visual,
                bad_sensor,
                sample_ids=self.sample_ids,
            )

        repeated = copy.deepcopy(self.records)
        repeated[-1]["participant_id"] = repeated[0]["participant_id"]
        with self.assertRaisesRegex(
            HierarchicalEvaluationError, "participants occur in multiple sessions"
        ):
            hierarchical_offline_evaluation(
                repeated,
                self.visual,
                self.sensor,
                sample_ids=self.sample_ids,
            )

    def test_loso_is_deterministic_and_complete(self) -> None:
        first = hierarchical_offline_evaluation(
            self.records,
            self.visual,
            self.sensor,
            sample_ids=self.sample_ids,
            outer_design="leave_one_session_out",
        )
        second = hierarchical_offline_evaluation(
            self.records,
            self.visual,
            self.sensor,
            sample_ids=self.sample_ids,
            outer_design="leave_one_session_out",
        )
        self.assertEqual(first["fold_count"], 15)
        self.assertEqual(first["oof_predictions"], second["oof_predictions"])
        self.assertEqual(
            [fold["test_session_ids"] for fold in first["folds"]],
            [[f"session-{index:02d}"] for index in range(15)],
        )

    def test_unimodal_ablation_uses_only_the_declared_model_input(self) -> None:
        report = hierarchical_offline_evaluation(
            self.records,
            self.visual,
            self.sensor,
            sample_ids=self.sample_ids,
            feature_mode="sensor",
        )
        self.assertEqual(report["feature_mode"], "sensor")
        self.assertEqual(report["protocol"]["feature_order"], ["sensor"])
        self.assertEqual(
            report["feature_dimensions"]["model_input"], self.sensor.shape[1]
        )
        self.assertEqual(
            report["feature_dimensions"]["available_fusion"],
            self.visual.shape[1] + self.sensor.shape[1],
        )
        with self.assertRaisesRegex(HierarchicalEvaluationError, "feature_mode"):
            hierarchical_offline_evaluation(
                self.records,
                self.visual,
                self.sensor,
                sample_ids=self.sample_ids,
                feature_mode="identity",
            )

    def test_fixed_activity_stress_has_complete_oof_coverage_and_false_claims(self) -> None:
        report = fixed_hierarchical_stress_evaluation(
            self.records,
            self.visual,
            self.sensor,
            sample_ids=self.sample_ids,
            design="leave_one_activity_out",
        )
        self.assertEqual(report["coverage_fraction"], 1.0)
        self.assertEqual(report["metrics"]["sample_count"], len(self.records))
        self.assertFalse(report["gate_selected_inside_stress_folds"])
        self.assertTrue(all(value is False for value in report["claim_status"].values()))
        for fold in report["folds"]:
            self.assertEqual(fold["session_overlap_ids"], [])
            self.assertEqual(fold["participant_overlap_ids"], [])
            self.assertEqual(fold["activity_overlap_ids"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
