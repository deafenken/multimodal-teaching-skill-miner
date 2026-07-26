from __future__ import annotations

import unittest

try:
    import numpy as np
    import sklearn  # noqa: F401

    RECOGNITION_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency environment
    np = None
    RECOGNITION_DEPS_AVAILABLE = False

from teaching_skill_miner.recognition.hierarchical_counterfactual import (
    CAUSAL_PREFIX,
    FULL,
    LEAVE_CURRENT_OUT,
    _source_indices,
    hierarchical_counterfactual_evaluation,
)


def _fixture() -> tuple[list[dict], "np.ndarray", "np.ndarray", list[str]]:
    records: list[dict] = []
    visual: list[list[float]] = []
    sensor: list[list[float]] = []
    # Five independent session groups per class support both nested SGKF levels.
    for session_index in range(15):
        label = session_index % 3
        base_time = 1_000.0 + session_index * 300.0
        for participant_index in range(2):
            participant = f"session-{session_index:02d}/p{participant_index}"
            for window_index in range(2):
                sample_id = (
                    f"s{session_index:02d}-p{participant_index}-w{window_index}"
                )
                timestamp = (
                    base_time + window_index * 15.0 + participant_index * 0.1
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
                        "source": {
                            "metadata_median_seconds_of_day": timestamp,
                        },
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
class HierarchicalCounterfactualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records, cls.visual, cls.sensor, cls.sample_ids = _fixture()
        cls.report = hierarchical_counterfactual_evaluation(
            cls.records,
            cls.visual,
            cls.sensor,
            sample_ids=cls.sample_ids,
        )

    def test_source_index_modes_enforce_exclusion_and_prefix_gap(self) -> None:
        timestamps = np.asarray([0.0, 30.0, 100.0])
        group = np.asarray([0, 1, 2])
        np.testing.assert_array_equal(
            _source_indices(FULL, group, timestamps, 1), group
        )
        np.testing.assert_array_equal(
            _source_indices(LEAVE_CURRENT_OUT, group, timestamps, 1),
            np.asarray([0, 2]),
        )
        np.testing.assert_array_equal(
            _source_indices(CAUSAL_PREFIX, group, timestamps, 1),
            np.asarray([0, 1]),
        )
        # The 70-second discontinuity resets the causal context.
        np.testing.assert_array_equal(
            _source_indices(CAUSAL_PREFIX, group, timestamps, 2),
            np.asarray([2]),
        )

    def test_oof_audit_proves_loo_exclusion_and_no_prefix_future(self) -> None:
        by_id = {record["sample_id"]: record for record in self.records}
        full_future_seen = False
        cross_participant_seen = False
        for row in self.report["oof_predictions"]:
            sample_id = row["sample_id"]
            target = by_id[sample_id]
            target_time = float(
                target["source"]["metadata_median_seconds_of_day"]
            )
            loo = row["modes"][LEAVE_CURRENT_OUT]
            self.assertNotIn(sample_id, loo["sequence_source_sample_ids"])
            self.assertNotIn(sample_id, loo["session_source_sample_ids"])

            prefix = row["modes"][CAUSAL_PREFIX]
            for source_id in prefix["sequence_source_sample_ids"]:
                source = by_id[source_id]
                self.assertEqual(source["session_id"], target["session_id"])
                self.assertEqual(
                    source["participant_id"], target["participant_id"]
                )
                self.assertLessEqual(
                    float(source["source"]["metadata_median_seconds_of_day"]),
                    target_time,
                )
            for source_id in prefix["session_source_sample_ids"]:
                source = by_id[source_id]
                self.assertEqual(source["session_id"], target["session_id"])
                self.assertLessEqual(
                    float(source["source"]["metadata_median_seconds_of_day"]),
                    target_time,
                )
            self.assertFalse(prefix["future_sequence_source_used"])
            self.assertFalse(prefix["future_session_source_used"])

            full = row["modes"][FULL]
            full_future_seen |= bool(
                full["future_sequence_source_used"]
                or full["future_session_source_used"]
            )
            cross_participant_seen |= bool(
                full["other_participant_session_source_used"]
            )
        self.assertTrue(full_future_seen)
        self.assertTrue(cross_participant_seen)

    def test_all_claims_remain_false_and_semantics_are_explicit(self) -> None:
        report = self.report
        self.assertTrue(all(value is False for value in report["claim_status"].values()))
        self.assertFalse(
            report["mode_results"][CAUSAL_PREFIX]["semantics"][
                "future_feature_rows_may_be_used"
            ]
        )
        self.assertTrue(
            report["mode_results"][FULL]["semantics"][
                "future_feature_rows_may_be_used"
            ]
        )
        self.assertEqual(
            report["mode_results"][FULL]["changed_from_full_count"], 0
        )
        self.assertFalse(report["outer_test_labels_used_for_training_or_gate_selection"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
