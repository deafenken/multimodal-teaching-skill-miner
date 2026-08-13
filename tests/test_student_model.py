from __future__ import annotations

from copy import deepcopy
import unittest

from teaching_skill_miner.student_model import (
    LEGACY_STUDENT_MODEL_SCHEMA,
    STUDENT_MODEL_SCHEMA,
    StudentModelError,
    initialize_student_model,
    knowledge_component_definitions,
    migrate_student_model,
    project_legacy_mastery,
    student_model_mastery_readiness,
    synchronize_runtime_state_from_kc_model,
    project_student_model,
    recommend_focus,
    stable_knowledge_component_id,
    update_student_model,
    validate_student_model,
)


GOAL = {
    "concept": "动态规划",
    "knowledge_components": ["状态定义", "状态转移"],
    "knowledge_spec": {
        "schema": "teaching_skill_miner.teacher_goal_knowledge_spec.v1",
        "status": "teacher_provided",
        "claim_boundary": {"authoritative_for_runtime_grading": True},
    },
}


def _authoritative_update(
    model: dict,
    *,
    kc_id: str,
    signal: str = "correct",
    evidence_id: str = "evidence-1",
    focus_dimension: str = "conceptual",
    round_number: int = 1,
    difficulty: float | None = None,
    discrimination: float | None = None,
    observed_at: str = "2026-08-12T10:00:00Z",
    time_basis: str = "wall_clock_utc",
) -> dict:
    return update_student_model(
        model,
        signal=signal,
        confidence=0.9,
        focus_dimension=focus_dimension,
        knowledge_component_ids=[kc_id],
        answer_alignment=("contradicted" if signal == "misconception" else "aligned"),
        assessment_eligible=True,
        authoritative=True,
        round_number=round_number,
        evidence_id=evidence_id,
        item_id="item-state-definition",
        question_id="question-state-definition",
        rubric_id="teacher-rubric-state-definition",
        difficulty=difficulty,
        discrimination=discrimination,
        observed_at=observed_at,
        time_basis=time_basis,
        source="teacher_knowledge_spec_exact_match",
    )


class StudentModelTests(unittest.TestCase):
    def test_stable_component_ids_do_not_depend_on_list_position(self) -> None:
        first = knowledge_component_definitions(GOAL)
        reordered = knowledge_component_definitions(
            {
                **GOAL,
                "knowledge_components": list(reversed(GOAL["knowledge_components"])),
            }
        )
        self.assertEqual(
            {item["label"]: item["kc_id"] for item in first},
            {item["label"]: item["kc_id"] for item in reordered},
        )
        self.assertEqual(first[0]["kc_id"], stable_knowledge_component_id("状态定义"))

    def test_initial_model_is_per_kc_bounded_and_evidence_free(self) -> None:
        model = initialize_student_model(
            {
                "prerequisite": 0.2,
                "conceptual": 0.4,
                "procedural": 0.1,
                "transfer": 0.0,
            },
            goal=GOAL,
        )
        self.assertEqual(model["schema"], STUDENT_MODEL_SCHEMA)
        self.assertEqual(len(model["knowledge_components"]), 2)
        self.assertEqual(model["overall"]["evidence_count"], 0)
        self.assertFalse(model["claim_boundary"]["is_ground_truth"])
        validate_student_model(model)

    def test_no_response_contains_no_evidence_and_does_not_update(self) -> None:
        model = initialize_student_model({"conceptual": 0.2}, goal=GOAL)
        kc_id = stable_knowledge_component_id("状态定义")
        updated = update_student_model(
            model,
            signal="no_response",
            confidence=1.0,
            focus_dimension="conceptual",
            knowledge_component_ids=[kc_id],
            answer_alignment="no_response",
            assessment_eligible=True,
            authoritative=True,
            round_number=1,
            evidence_id="blank-1",
            item_id="item-1",
            question_id="question-1",
            rubric_id="rubric-1",
            observed_at="2026-08-12T10:00:00Z",
        )
        self.assertEqual(updated["knowledge_components"][kc_id]["evidence_count"], 0)
        self.assertFalse(updated["last_update"]["update_applied"])
        self.assertEqual(
            updated["last_update"]["reason"], "signal_contains_no_assessment_outcome"
        )

    def test_correct_partial_and_recurrent_wrong_are_bounded_bkt_updates(self) -> None:
        model = initialize_student_model({"conceptual": 0.35}, goal=GOAL)
        kc_id = stable_knowledge_component_id("状态定义")
        before = model["knowledge_components"][kc_id]["dimensions"]["conceptual"][
            "p_mastery"
        ]
        correct = _authoritative_update(model, kc_id=kc_id)
        after_correct = correct["knowledge_components"][kc_id]["dimensions"][
            "conceptual"
        ]["p_mastery"]
        self.assertGreater(after_correct, before)
        partial = _authoritative_update(
            correct,
            kc_id=kc_id,
            signal="partial",
            evidence_id="evidence-2",
            observed_at="2026-08-12T10:01:00Z",
        )
        self.assertEqual(partial["knowledge_components"][kc_id]["evidence_count"], 2)
        wrong_once = _authoritative_update(
            partial,
            kc_id=kc_id,
            signal="misconception",
            evidence_id="evidence-3",
            observed_at="2026-08-12T10:02:00Z",
        )
        wrong_twice = _authoritative_update(
            wrong_once,
            kc_id=kc_id,
            signal="misconception",
            evidence_id="evidence-4",
            observed_at="2026-08-12T10:03:00Z",
        )
        first_wrong_p = wrong_once["knowledge_components"][kc_id]["dimensions"][
            "conceptual"
        ]["p_mastery"]
        second_wrong_p = wrong_twice["knowledge_components"][kc_id]["dimensions"][
            "conceptual"
        ]["p_mastery"]
        self.assertLess(second_wrong_p, first_wrong_p)
        self.assertGreaterEqual(second_wrong_p, 0.0)

    def test_item_difficulty_and_discrimination_change_posterior(self) -> None:
        model = initialize_student_model({"conceptual": 0.3}, goal=GOAL)
        kc_id = stable_knowledge_component_id("状态定义")
        easy = _authoritative_update(
            model,
            kc_id=kc_id,
            evidence_id="easy",
            difficulty=0.1,
            discrimination=0.9,
        )
        hard = _authoritative_update(
            model,
            kc_id=kc_id,
            evidence_id="hard",
            difficulty=0.9,
            discrimination=0.9,
        )
        easy_p = easy["knowledge_components"][kc_id]["dimensions"]["conceptual"][
            "p_mastery"
        ]
        hard_p = hard["knowledge_components"][kc_id]["dimensions"]["conceptual"][
            "p_mastery"
        ]
        self.assertNotEqual(easy_p, hard_p)
        low_discrimination = _authoritative_update(
            model,
            kc_id=kc_id,
            evidence_id="low-discrimination",
            difficulty=0.5,
            discrimination=0.1,
        )
        self.assertLess(
            low_discrimination["knowledge_components"][kc_id]["dimensions"][
                "conceptual"
            ]["p_mastery"],
            hard_p,
        )

    def test_missing_rubric_or_authority_fails_closed(self) -> None:
        model = initialize_student_model({"conceptual": 0.4}, goal=GOAL)
        kc_id = stable_knowledge_component_id("状态定义")
        updated = update_student_model(
            model,
            signal="correct",
            confidence=0.99,
            focus_dimension="conceptual",
            knowledge_component_ids=[kc_id],
            answer_alignment="aligned",
            assessment_eligible=True,
            authoritative=True,
            evidence_id="missing-rubric",
            item_id="item-1",
            question_id="question-1",
            rubric_id=None,
            observed_at="2026-08-12T10:00:00Z",
        )
        self.assertEqual(updated["knowledge_components"][kc_id]["evidence_count"], 0)
        self.assertEqual(
            updated["last_update"]["reason"],
            "complete_item_question_rubric_provenance_required",
        )
        untrusted = update_student_model(
            model,
            signal="correct",
            confidence=0.99,
            focus_dimension="conceptual",
            knowledge_component_ids=[kc_id],
            answer_alignment="aligned",
            assessment_eligible=True,
            authoritative=False,
            evidence_id="untrusted",
            item_id="item-1",
            question_id="question-1",
            rubric_id="rubric-1",
            observed_at="2026-08-12T10:00:00Z",
        )
        self.assertEqual(untrusted["knowledge_components"][kc_id]["evidence_count"], 0)

    def test_replay_is_idempotent_and_conflicting_id_is_rejected(self) -> None:
        model = initialize_student_model({"conceptual": 0.2}, goal=GOAL)
        kc_id = stable_knowledge_component_id("状态定义")
        once = _authoritative_update(model, kc_id=kc_id)
        replayed = _authoritative_update(once, kc_id=kc_id)
        self.assertEqual(replayed, once)
        with self.assertRaises(StudentModelError):
            _authoritative_update(
                once,
                kc_id=kc_id,
                signal="misconception",
            )

    def test_one_kc_update_does_not_leak_to_another_kc(self) -> None:
        model = initialize_student_model({"conceptual": 0.25}, goal=GOAL)
        state_id = stable_knowledge_component_id("状态定义")
        transition_id = stable_knowledge_component_id("状态转移")
        untouched = deepcopy(model["knowledge_components"][transition_id])
        updated = _authoritative_update(model, kc_id=state_id)
        self.assertEqual(updated["knowledge_components"][transition_id], untouched)
        ambiguous = update_student_model(
            model,
            signal="correct",
            confidence=1.0,
            focus_dimension="conceptual",
            knowledge_component_ids=[state_id, transition_id],
            answer_alignment="aligned",
            assessment_eligible=True,
            authoritative=True,
        )
        self.assertEqual(ambiguous["overall"]["evidence_count"], 0)
        self.assertEqual(
            ambiguous["last_update"]["reason"],
            "exactly_one_knowledge_component_required",
        )

    def test_forgetting_uses_elapsed_time_and_preserves_audit_timestamp(self) -> None:
        model = initialize_student_model({"conceptual": 0.3}, goal=GOAL)
        kc_id = stable_knowledge_component_id("状态定义")
        learned = _authoritative_update(model, kc_id=kc_id)
        later = _authoritative_update(
            learned,
            kc_id=kc_id,
            signal="partial",
            evidence_id="later",
            observed_at="2026-09-11T10:00:00Z",
        )
        entry = later["knowledge_components"][kc_id]["evidence_ledger"][-1]
        self.assertEqual(entry["elapsed_days"], 30.0)
        self.assertEqual(
            later["knowledge_components"][kc_id]["last_observed_at"],
            "2026-09-11T10:00:00Z",
        )

    def test_wall_clock_after_logical_time_does_not_apply_decades_of_forgetting(
        self,
    ) -> None:
        model = initialize_student_model({"conceptual": 0.3}, goal=GOAL)
        kc_id = stable_knowledge_component_id("状态定义")
        logical = _authoritative_update(
            model,
            kc_id=kc_id,
            observed_at="2000-01-01T00:00:01Z",
            time_basis="session_logical",
        )
        wall_clock = _authoritative_update(
            logical,
            kc_id=kc_id,
            signal="partial",
            evidence_id="wall-clock-after-logical",
            observed_at="2026-08-12T10:00:00Z",
            time_basis="wall_clock_utc",
        )
        entry = wall_clock["knowledge_components"][kc_id]["evidence_ledger"][-1]
        self.assertEqual(entry["elapsed_days"], 0.0)
        self.assertFalse(entry["forgetting_applied"])
        self.assertEqual(entry["time_basis"], "wall_clock_utc")

    def test_v1_migration_keeps_global_means_as_priors_not_fake_kc_evidence(
        self,
    ) -> None:
        legacy = {
            "schema": LEGACY_STUDENT_MODEL_SCHEMA,
            "version": 1,
            "dimensions": {
                dimension: {"p_mastery": 0.6, "evidence_count": 2}
                for dimension in (
                    "prerequisite",
                    "conceptual",
                    "procedural",
                    "transfer",
                )
            },
        }
        migrated = migrate_student_model(legacy, goal=GOAL)
        self.assertEqual(migrated["schema"], STUDENT_MODEL_SCHEMA)
        self.assertEqual(migrated["dimensions"]["conceptual"]["p_mastery"], 0.6)
        self.assertEqual(migrated["overall"]["evidence_count"], 0)
        self.assertEqual(migrated["migration"]["unattributed_legacy_evidence_count"], 8)
        self.assertFalse(
            migrated["migration"]["legacy_evidence_attributed_to_components"]
        )

    def test_legacy_projection_is_derived_and_public_projection_hides_parameters(
        self,
    ) -> None:
        model = initialize_student_model({"conceptual": 0.2}, goal=GOAL)
        kc_id = stable_knowledge_component_id("状态定义")
        updated = _authoritative_update(model, kc_id=kc_id)
        legacy = project_legacy_mastery(updated)
        self.assertEqual(
            legacy["conceptual"], updated["dimensions"]["conceptual"]["p_mastery"]
        )
        projected = project_student_model(updated)
        self.assertNotIn("alpha", projected["dimensions"]["conceptual"])
        self.assertNotIn("bkt", projected["knowledge_components"][kc_id])
        self.assertFalse(updated["compatibility"]["legacy_success_gate_uses_kc_model"])
        validate_student_model(updated)

    def test_live_success_readiness_rejects_unobserved_priors_and_requires_all_kcs(
        self,
    ) -> None:
        thresholds = {
            "prerequisite": 0.6,
            "conceptual": 0.6,
            "procedural": 0.6,
            "transfer": 0.6,
        }
        readiness_goal = {
            **GOAL,
            "knowledge_components": [
                "状态定义",
                "状态转移",
                "边界条件",
                "迁移判断",
            ],
        }
        model = synchronize_runtime_state_from_kc_model(
            initialize_student_model(
                {dimension: 0.99 for dimension in thresholds}, goal=readiness_goal
            )
        )

        prior_only = student_model_mastery_readiness(model, thresholds)

        self.assertFalse(prior_only["eligible"])
        self.assertIn(
            "target_knowledge_components_without_authoritative_evidence",
            prior_only["reasons"],
        )
        self.assertIn(
            "required_mastery_dimension_not_observed_or_below_threshold",
            prior_only["reasons"],
        )
        self.assertFalse(
            prior_only["claim_boundary"]["unobserved_prior_counts_as_mastery_evidence"]
        )

        dimensions = ("prerequisite", "conceptual", "procedural", "transfer")
        for index, kc_id in enumerate(sorted(model["knowledge_components"])):
            model = _authoritative_update(
                model,
                kc_id=kc_id,
                focus_dimension=dimensions[index % len(dimensions)],
                evidence_id=f"readiness:{kc_id}",
                round_number=index + 1,
            )
        model = synchronize_runtime_state_from_kc_model(model)

        ready = student_model_mastery_readiness(model, thresholds)

        self.assertTrue(ready["eligible"])
        self.assertTrue(all(row["ready"] for row in ready["component_coverage"]))
        self.assertTrue(all(row["ready"] for row in ready["dimensions"].values()))

    def test_focus_is_a_stable_kc_and_dimension_pair(self) -> None:
        model = initialize_student_model(
            {
                "prerequisite": 0.8,
                "conceptual": 0.6,
                "procedural": 0.2,
                "transfer": 0.7,
            },
            goal=GOAL,
        )
        recommendation = recommend_focus(
            model,
            {
                "prerequisite": 0.75,
                "conceptual": 0.8,
                "procedural": 0.75,
                "transfer": 0.8,
            },
        )
        self.assertEqual(recommendation["dimension"], "procedural")
        self.assertIn(
            recommendation["knowledge_component_id"], model["knowledge_components"]
        )


if __name__ == "__main__":
    unittest.main()
