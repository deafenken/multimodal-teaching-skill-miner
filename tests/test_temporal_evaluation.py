from __future__ import annotations

import unittest

try:
    import numpy as np
    import sklearn  # noqa: F401

    RECOGNITION_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency environment
    np = None
    RECOGNITION_DEPS_AVAILABLE = False

from teaching_skill_miner.recognition.temporal_evaluation import (
    CAUSAL_WINDOWS,
    FIXED_LINEAR_SVC_C,
    FIXED_SEED,
    LEAVE_ONE_SESSION_OUT,
    MAXIMUM_GAP_SECONDS,
    STRATIFIED_SESSION_5FOLD,
    TemporalEvaluationError,
    _causal_average,
    _evaluate_outer_fold,
    _outer_specs,
    _prepare_temporal_inputs,
    temporal_linear_svc_evaluation,
)


def _fixture() -> tuple[list[dict], "np.ndarray", "np.ndarray", list[str]]:
    records: list[dict] = []
    visual: list[list[float]] = []
    sensor: list[list[float]] = []
    label_sequence = (0, 1, 2, 1, 0, 2)
    for session_number in range(12):
        cohort = session_number % 3
        activity = session_number % 4
        session_id = f"cohort-{cohort}/activity-{activity}/session-{session_number:02d}"
        participant_id = f"participant-{session_number:02d}"
        start_time = 36_000.0 + session_number * 10.0
        for position, label in enumerate(label_sequence):
            sample_id = f"session-{session_number:02d}-time-{position:02d}"
            records.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "label_name": ("low", "medium", "high")[label],
                    "session_id": session_id,
                    "participant_id": participant_id,
                    "teacher_id": f"teacher-{cohort}",
                    "cohort_id": f"cohort-{cohort}",
                    "activity_id": f"activity-{activity}",
                    "site_id": "single-site",
                    "source": {
                        "metadata_median_seconds_of_day": start_time
                        + position * 15.0,
                    },
                }
            )
            visual.append(
                [5.0 if label == 0 else 0.0, session_number / 100.0]
            )
            sensor.append(
                [5.0 if label == 2 else 0.0, position / 100.0]
            )
    return (
        records,
        np.asarray(visual, dtype=float),
        np.asarray(sensor, dtype=float),
        [str(record["sample_id"]) for record in records],
    )


@unittest.skipUnless(RECOGNITION_DEPS_AVAILABLE, "recognition dependencies are optional")
class TemporalLinearSvcEvaluationTests(unittest.TestCase):
    def test_fixed_five_by_five_and_loso_protocols_are_fully_audited(self) -> None:
        records, visual, sensor, sample_ids = _fixture()

        report = temporal_linear_svc_evaluation(
            records, visual, sensor, sample_ids
        )

        self.assertTrue(report["post_selection_exploratory"])
        self.assertEqual(report["fixed_model"]["C"], FIXED_LINEAR_SVC_C)
        self.assertEqual(report["fixed_model"]["kernel"], "linear")
        self.assertEqual(report["nested_selection"]["inner_splits"], 5)
        self.assertEqual(report["nested_selection"]["seed"], FIXED_SEED)
        self.assertEqual(
            report["temporal_protocol"]["candidate_windows"],
            list(CAUSAL_WINDOWS),
        )
        self.assertEqual(
            report["temporal_protocol"]["maximum_gap_seconds"],
            MAXIMUM_GAP_SECONDS,
        )
        self.assertFalse(report["temporal_protocol"]["future_score_use"])
        self.assertFalse(report["temporal_protocol"]["cross_session_smoothing"])
        self.assertFalse(
            report["temporal_protocol"]["cross_participant_smoothing"]
        )
        self.assertTrue(all(value is False for value in report["claim_status"].values()))

        expected_fold_counts = {
            STRATIFIED_SESSION_5FOLD: 5,
            LEAVE_ONE_SESSION_OUT: 12,
        }
        by_id = {record["sample_id"]: record for record in records}
        for design, expected_count in expected_fold_counts.items():
            evaluation = report["evaluations"][design]
            self.assertEqual(evaluation["fold_count"], expected_count)
            self.assertEqual(evaluation["evaluated_fold_count"], expected_count)
            self.assertEqual(evaluation["oof_coverage"], 1.0)
            self.assertTrue(
                all(value is False for value in evaluation["claim_status"].values())
            )
            self.assertEqual(
                len(evaluation["metrics"]["raw"]["confusion_matrix"]), 3
            )
            self.assertEqual(
                len(evaluation["metrics"]["selected_causal"]["per_class"]), 3
            )
            self.assertEqual(
                evaluation["metrics"]["fold_training_majority"]["sample_count"],
                len(records),
            )
            for fold in evaluation["folds"]:
                self.assertEqual(fold["session_overlap_ids"], [])
                self.assertEqual(fold["participant_overlap_ids"], [])
                self.assertEqual(fold["sample_id_overlap"], [])
                self.assertIn(fold["selected_causal_window"], CAUSAL_WINDOWS)
                self.assertFalse(
                    fold[
                        "outer_test_labels_used_for_window_or_model_selection"
                    ]
                )
                self.assertEqual(
                    len(fold["inner_window_selection"]["inner_folds"]), 5
                )
                for inner in fold["inner_window_selection"]["inner_folds"]:
                    self.assertEqual(inner["session_overlap_ids"], [])
            for row in evaluation["oof_predictions"]:
                current = by_id[row["sample_id"]]
                current_time = current["source"][
                    "metadata_median_seconds_of_day"
                ]
                history_records = [by_id[value] for value in row["causal_history_sample_ids"]]
                self.assertLessEqual(
                    len(history_records), row["selected_causal_window"]
                )
                self.assertTrue(
                    all(
                        item["session_id"] == current["session_id"]
                        and item["participant_id"] == current["participant_id"]
                        and item["source"]["metadata_median_seconds_of_day"]
                        <= current_time
                        for item in history_records
                    )
                )

    def test_causal_smoothing_never_crosses_stream_or_uses_future_scores(self) -> None:
        sample_ids = ["a0", "a1", "a2", "b0", "c0"]
        data = {
            "sample_ids": sample_ids,
            "times": np.asarray([100.0, 115.0, 190.0, 110.0, 120.0]),
            "identities": {
                "session_id": np.asarray(["s1", "s1", "s1", "s1", "s2"], dtype=object),
                "participant_id": np.asarray(["p1", "p1", "p1", "p2", "p1"], dtype=object),
            },
        }
        scores = np.asarray(
            [[10.0, 0.0], [0.0, 10.0], [8.0, 2.0], [0.0, 20.0], [20.0, 0.0]]
        )
        indices = np.arange(len(sample_ids))

        smoothed, histories = _causal_average(
            scores, indices, data, window=3
        )

        np.testing.assert_allclose(smoothed[0], scores[0])
        np.testing.assert_allclose(smoothed[1], np.mean(scores[:2], axis=0))
        # The 75-second gap resets p1/s1 history.
        np.testing.assert_allclose(smoothed[2], scores[2])
        self.assertEqual(histories[2], ["a2"])
        # Different participant and different session are isolated.
        np.testing.assert_allclose(smoothed[3], scores[3])
        np.testing.assert_allclose(smoothed[4], scores[4])
        self.assertEqual(histories[3], ["b0"])
        self.assertEqual(histories[4], ["c0"])

        changed = scores.copy()
        changed[2] = [-1_000.0, 1_000.0]
        changed_smoothed, _ = _causal_average(
            changed, indices, data, window=3
        )
        # Changing a future score cannot alter already-issued predictions/scores.
        np.testing.assert_allclose(changed_smoothed[:2], smoothed[:2])

    def test_outer_test_label_permutation_cannot_change_selection_or_prediction(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        data = _prepare_temporal_inputs(
            records, visual, sensor, sample_ids
        )
        spec = _outer_specs(data, STRATIFIED_SESSION_5FOLD)[0]
        original = _evaluate_outer_fold(data, spec, fold_number=1)

        permuted_data = dict(data)
        permuted_labels = data["labels"].copy()
        test_indices = np.asarray(spec["test_indices"], dtype=int)
        permuted_labels[test_indices] = (permuted_labels[test_indices] + 1) % 3
        permuted_data["labels"] = permuted_labels
        permuted = _evaluate_outer_fold(permuted_data, spec, fold_number=1)

        self.assertEqual(
            original["selected_causal_window"],
            permuted["selected_causal_window"],
        )
        self.assertEqual(
            original["inner_window_selection"],
            permuted["inner_window_selection"],
        )
        self.assertEqual(original["raw_predictions"], permuted["raw_predictions"])
        self.assertEqual(
            original["selected_causal_predictions"],
            permuted["selected_causal_predictions"],
        )
        self.assertEqual(original["raw_scores"], permuted["raw_scores"])
        self.assertNotEqual(
            original["raw_metrics"]["accuracy"],
            permuted["raw_metrics"]["accuracy"],
        )

    def test_loso_fold_order_is_lexically_deterministic(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        report = temporal_linear_svc_evaluation(
            records,
            visual,
            sensor,
            sample_ids,
            designs=(LEAVE_ONE_SESSION_OUT,),
        )
        folds = report["evaluations"][LEAVE_ONE_SESSION_OUT]["folds"]
        held_sessions = [fold["test_session_ids"][0] for fold in folds]
        self.assertEqual(held_sessions, sorted(held_sessions))
        self.assertTrue(
            all(len(fold["test_session_ids"]) == 1 for fold in folds)
        )

    def test_order_timestamp_and_exact_feature_order_fail_closed(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        records[0]["source"].pop("metadata_median_seconds_of_day")
        with self.assertRaisesRegex(
            TemporalEvaluationError, "metadata_median_seconds_of_day"
        ):
            temporal_linear_svc_evaluation(
                records,
                visual,
                sensor,
                sample_ids,
                designs=(STRATIFIED_SESSION_5FOLD,),
            )

        records, visual, sensor, sample_ids = _fixture()
        sample_ids[0], sample_ids[1] = sample_ids[1], sample_ids[0]
        with self.assertRaisesRegex(TemporalEvaluationError, "record order"):
            temporal_linear_svc_evaluation(
                records,
                visual,
                sensor,
                sample_ids,
                designs=(STRATIFIED_SESSION_5FOLD,),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
