from __future__ import annotations

import unittest

from teaching_skill_miner.student_model import (
    STUDENT_MODEL_SCHEMA,
    initialize_student_model,
    project_student_model,
    recommend_focus,
    update_student_model,
    validate_student_model,
)


class StudentModelTests(unittest.TestCase):
    def test_initial_model_is_bounded_and_evidence_free(self) -> None:
        model = initialize_student_model(
            {"prerequisite": 0.2, "conceptual": 0.4, "procedural": 0.1, "transfer": 0.0}
        )
        self.assertEqual(model["schema"], STUDENT_MODEL_SCHEMA)
        self.assertEqual(model["overall"]["evidence_count"], 0)
        self.assertFalse(model["claim_boundary"]["is_ground_truth"])
        validate_student_model(model)

    def test_aligned_correct_observation_increases_probability(self) -> None:
        model = initialize_student_model({"conceptual": 0.2})
        before = model["dimensions"]["conceptual"]["p_mastery"]
        updated = update_student_model(
            model,
            signal="correct",
            confidence=0.9,
            focus_dimension="conceptual",
            answer_alignment="aligned",
            round_number=1,
            evidence_id="session_history:r1:structured_signal",
        )
        row = updated["dimensions"]["conceptual"]
        self.assertGreater(row["p_mastery"], before)
        self.assertEqual(row["evidence_count"], 1)
        self.assertEqual(row["last_evidence_id"], "session_history:r1:structured_signal")
        self.assertTrue(updated["last_update"]["update_applied"])

    def test_ambiguous_or_review_turn_does_not_move_estimate(self) -> None:
        model = initialize_student_model({"conceptual": 0.4})
        before = model["dimensions"]["conceptual"]["p_mastery"]
        updated = update_student_model(
            model,
            signal="correct",
            confidence=0.99,
            focus_dimension="conceptual",
            answer_alignment="ambiguous",
            needs_human_review=True,
            round_number=1,
        )
        self.assertEqual(updated["dimensions"]["conceptual"]["p_mastery"], before)
        self.assertEqual(updated["dimensions"]["conceptual"]["evidence_count"], 0)
        self.assertFalse(updated["last_update"]["update_applied"])

    def test_grounded_misconception_decreases_estimate(self) -> None:
        model = initialize_student_model({"conceptual": 0.8})
        before = model["dimensions"]["conceptual"]["p_mastery"]
        updated = update_student_model(
            model,
            signal="misconception",
            confidence=0.95,
            focus_dimension="conceptual",
            answer_alignment="contradicted",
            round_number=2,
            evidence_id="session_history:r2:structured_signal",
        )
        self.assertLess(updated["dimensions"]["conceptual"]["p_mastery"], before)
        self.assertEqual(updated["dimensions"]["conceptual"]["negative_evidence"], 1)

    def test_focus_prefers_deficit_and_uncertainty(self) -> None:
        model = initialize_student_model(
            {"prerequisite": 0.8, "conceptual": 0.6, "procedural": 0.2, "transfer": 0.7}
        )
        recommendation = recommend_focus(
            model,
            {"prerequisite": 0.75, "conceptual": 0.8, "procedural": 0.75, "transfer": 0.8},
        )
        self.assertEqual(recommendation["dimension"], "procedural")
        self.assertGreaterEqual(recommendation["confidence"], 0.0)

    def test_projection_does_not_expose_beta_parameters(self) -> None:
        model = initialize_student_model({"conceptual": 0.2})
        projected = project_student_model(model)
        self.assertNotIn("alpha", projected["dimensions"]["conceptual"])
        self.assertNotIn("beta", projected["dimensions"]["conceptual"])
        validate_student_model(model)


if __name__ == "__main__":
    unittest.main()
