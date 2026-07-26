from __future__ import annotations

import copy
import unittest
from collections import defaultdict

try:
    import numpy as np
    import sklearn  # noqa: F401

    RECOGNITION_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency environment
    np = None
    RECOGNITION_DEPS_AVAILABLE = False

from teaching_skill_miner.recognition.offline_sequence_evaluation import (
    OfflineSequenceEvaluationError,
    offline_sequence_evaluation,
)


def _fixture() -> tuple[list[dict], "np.ndarray", "np.ndarray", list[str]]:
    records: list[dict] = []
    visual: list[list[float]] = []
    sensor: list[list[float]] = []
    # Five independent sessions per class support the fixed five-fold splitter.
    for session_index in range(15):
        label = session_index % 3
        for participant_index in range(2):
            participant_id = f"session-{session_index:02d}/p{participant_index}"
            for window_index in range(3):
                sample_id = f"s{session_index:02d}-p{participant_index}-w{window_index}"
                records.append(
                    {
                        "sample_id": sample_id,
                        "label": label,
                        "session_id": f"session-{session_index:02d}",
                        "participant_id": participant_id,
                        "cohort_id": f"cohort-{session_index % 3}",
                        "activity_id": f"activity-{session_index % 5}",
                        "site_id": "test-site",
                    }
                )
                jitter = (participant_index * 3 + window_index) / 1000.0
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
class OfflineSequenceEvaluationTests(unittest.TestCase):
    def test_fixed_protocol_and_all_claims_are_explicitly_exploratory(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        report = offline_sequence_evaluation(
            records, visual, sensor, sample_ids=sample_ids
        )

        self.assertEqual(
            report["evaluation_kind"],
            "post_selected_offline_participant_recording_oof",
        )
        self.assertTrue(report["offline_full_sequence_required"])
        self.assertFalse(report["real_time_compatible"])
        self.assertTrue(report["post_selected_on_current_dataset"])
        self.assertFalse(report["contains_inferential_statistics"])
        self.assertEqual(report["outer_design"], "sgkf5")
        self.assertEqual(report["fold_count"], 5)
        protocol = report["fixed_model_protocol"]
        self.assertEqual(protocol["C"], 0.1)
        self.assertEqual(protocol["class_weight"], "balanced")
        self.assertEqual(protocol["solver"], "lbfgs")
        self.assertEqual(protocol["max_iter"], 4000)
        self.assertEqual(protocol["minority_to_medium_ratio_threshold"], 3.0)
        self.assertEqual(protocol["hyperparameter_selection"], "none; post-selected constants are fixed")
        self.assertTrue(all(value is False for value in report["claim_status"].values()))

    def test_pooling_never_mixes_sequences_and_assigns_one_sequence_prediction(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        report = offline_sequence_evaluation(
            records, visual, sensor, sample_ids=sample_ids
        )

        expected_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
        for record in records:
            expected_ids[(record["session_id"], record["participant_id"])].add(
                record["sample_id"]
            )
        observed_predictions: dict[tuple[str, str], set[int]] = defaultdict(set)
        observed_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in report["oof_predictions"]:
            key = (row["session_id"], row["participant_id"])
            observed_ids[key].add(row["sample_id"])
            observed_predictions[key].add(row["pooled_sequence_prediction"])

        self.assertEqual(dict(observed_ids), dict(expected_ids))
        self.assertTrue(all(len(values) == 1 for values in observed_predictions.values()))
        self.assertEqual(report["sequence_count"], len(expected_ids))
        self.assertEqual(
            {(row["session_id"], row["participant_id"]) for row in report["sequence_predictions"]},
            set(expected_ids),
        )

    def test_outer_folds_have_no_session_or_participant_overlap(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        report = offline_sequence_evaluation(
            records, visual, sensor, sample_ids=sample_ids
        )

        for fold in report["folds"]:
            self.assertEqual(fold["session_overlap"], [])
            self.assertEqual(fold["participant_overlap"], [])
            self.assertEqual(fold["identity_overlap"]["session_id"], [])
            self.assertEqual(fold["identity_overlap"]["participant_id"], [])
            self.assertTrue(
                set(fold["train_session_ids"]).isdisjoint(fold["test_session_ids"])
            )
            self.assertTrue(
                set(fold["train_participant_ids"]).isdisjoint(
                    fold["test_participant_ids"]
                )
            )

    def test_metrics_are_computed_from_oof_predictions_not_hardcoded(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        changed_records = copy.deepcopy(records)
        changed_records[0]["label"] = 1
        report = offline_sequence_evaluation(
            changed_records, visual, sensor, sample_ids=sample_ids
        )

        labels = np.asarray([row["label"] for row in report["oof_predictions"]])
        predictions = np.asarray(
            [row["pooled_sequence_prediction"] for row in report["oof_predictions"]]
        )
        manual_accuracy = float((labels == predictions).mean())
        self.assertAlmostEqual(
            report["pooled_sequence_metrics"]["accuracy"], manual_accuracy, places=6
        )
        self.assertLess(report["pooled_sequence_metrics"]["accuracy"], 1.0)
        self.assertNotAlmostEqual(
            report["pooled_sequence_metrics"]["accuracy"], 0.848, places=3
        )

    def test_repeated_participant_across_sessions_fails_closed(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        records[-1]["participant_id"] = records[0]["participant_id"]
        with self.assertRaisesRegex(
            OfflineSequenceEvaluationError, "participants occur in multiple sessions"
        ):
            offline_sequence_evaluation(
                records, visual, sensor, sample_ids=sample_ids
            )

    def test_sample_order_and_supported_design_are_validated(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        sample_ids[0], sample_ids[1] = sample_ids[1], sample_ids[0]
        with self.assertRaisesRegex(OfflineSequenceEvaluationError, "record order"):
            offline_sequence_evaluation(
                records, visual, sensor, sample_ids=sample_ids
            )
        with self.assertRaisesRegex(OfflineSequenceEvaluationError, "outer_design"):
            offline_sequence_evaluation(
                records,
                visual,
                sensor,
                sample_ids=[record["sample_id"] for record in records],
                outer_design="random-window-cv",
            )

    def test_leave_one_session_out_is_deterministic_and_complete(self) -> None:
        records, visual, sensor, sample_ids = _fixture()
        first = offline_sequence_evaluation(
            records,
            visual,
            sensor,
            sample_ids=sample_ids,
            outer_design="leave_one_session_out",
        )
        second = offline_sequence_evaluation(
            records,
            visual,
            sensor,
            sample_ids=sample_ids,
            outer_design="leave_one_session_out",
        )
        self.assertEqual(first["fold_count"], 15)
        self.assertEqual(first["oof_predictions"], second["oof_predictions"])
        self.assertEqual(
            [fold["test_session_ids"] for fold in first["folds"]],
            [[f"session-{index:02d}"] for index in range(15)],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
