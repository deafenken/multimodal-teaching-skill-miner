from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import jsonschema

from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.teacher_agent_outcomes import (
    LearningOutcomeError,
    evaluate_learning_observation,
)


ROOT = Path(__file__).resolve().parents[1]


class LearningOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = read_json(
            ROOT / "data/teacher_agent_learning_outcome_demo.json"
        )

    def test_demo_scores_pre_post_and_transfer_without_overclaiming(self) -> None:
        jsonschema.Draft202012Validator(
            read_json(ROOT / "schema/teacher_agent_learning_observation.schema.json")
        ).validate(self.observation)
        report = evaluate_learning_observation(self.observation)
        jsonschema.Draft202012Validator(
            read_json(ROOT / "schema/teacher_agent_learning_report.schema.json")
        ).validate(report)
        self.assertEqual(report["metrics"]["absolute_gain"], 0.4)
        self.assertEqual(report["metrics"]["normalized_gain"], 0.666667)
        self.assertEqual(report["metrics"]["transfer_proportion"], 0.666667)
        self.assertFalse(
            report["claim_boundary"]["real_learner_effectiveness_established"]
        )
        self.assertFalse(
            report["claim_boundary"]["record_declares_real_learner_data"]
        )

    def test_invalid_scores_fail_closed(self) -> None:
        invalid = deepcopy(self.observation)
        invalid["posttest"] = {"earned": 6, "possible": 5}
        with self.assertRaises(LearningOutcomeError):
            evaluate_learning_observation(invalid)


if __name__ == "__main__":
    unittest.main()
