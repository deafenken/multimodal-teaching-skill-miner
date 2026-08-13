from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from teaching_skill_miner.student_model import (
    STUDENT_MODEL_ADJUDICATION_RESULT_SCHEMA,
    STUDENT_MODEL_ADJUDICATION_REVISION_SCHEMA,
    StudentModelError,
    apply_student_model_adjudication,
    initialize_student_model,
    supersede_and_replay_student_model_evidence,
    update_student_model,
    validate_student_model,
)
from teaching_skill_miner.teacher_agent_adjudication import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
GOAL = {
    "concept": "动态规划",
    "knowledge_components": [
        {"kc_id": "kc_state_definition", "label": "状态定义"},
        {"kc_id": "kc_state_transition", "label": "状态转移"},
    ],
    "knowledge_spec": {
        "schema": "teaching_skill_miner.teacher_goal_knowledge_spec.v1",
        "status": "teacher_provided",
        "claim_boundary": {"authoritative_for_runtime_grading": True},
    },
}


def _instruction(
    *,
    operation: str,
    evidence_id: str,
    evidence_sha256: str,
    kc_id: str = "kc_state_definition",
) -> dict:
    item_id = "adj_" + "a" * 24
    item_version_sha256 = "4" * 64
    value = {
        "schema": "teaching_skill_miner.student_model_adjudication_instruction.v1",
        "instruction_id": "adjinst_"
        + canonical_sha256({"item_id": item_id, "version_sha256": item_version_sha256})[
            :24
        ],
        "review_item_id": item_id,
        "review_item_version": 3,
        "review_item_version_sha256": item_version_sha256,
        "operation": operation,
        "supersedes_evidence_id": evidence_id,
        "evidence_sha256": evidence_sha256,
        "target_kc_ids": [kc_id],
        "correction": (
            {
                "signal": "partial",
                "answer_alignment": "partially_aligned",
                "focus_dimension": "conceptual",
                "target_kc_ids": [kc_id],
            }
            if operation == "supersede_and_replay"
            else None
        ),
        "authority": {
            "identity": "local_operator_not_authenticated",
            "authenticated": False,
            "teacher_identity_claimed": False,
            "authoritative_for_mastery_update": False,
            "requires_downstream_authority_revalidation": True,
        },
        "mutates_student_model": False,
        "mastery_update_authorized": False,
        "requires_explicit_apply": True,
    }
    value["instruction_sha256"] = canonical_sha256(value)
    return value


def _apply(
    model: dict,
    *,
    evidence_id: str,
    kc_id: str = "kc_state_definition",
    signal: str = "correct",
    minute: int = 0,
    authority_hash: str | None = None,
) -> dict:
    alignment = "contradicted" if signal == "misconception" else "aligned"
    hour = 5 + minute // 60
    minute_of_hour = minute % 60
    return update_student_model(
        model,
        signal=signal,
        confidence=0.9,
        focus_dimension="conceptual",
        knowledge_component_ids=[kc_id],
        answer_alignment=alignment,
        assessment_eligible=True,
        authoritative=True,
        round_number=minute + 1,
        evidence_id=evidence_id,
        item_id=f"item.{evidence_id}",
        question_id=f"question.{evidence_id}",
        rubric_id="rubric.dp.1",
        difficulty=0.5,
        discrimination=0.8,
        observed_at=f"2026-08-12T{hour:02d}:{minute_of_hour:02d}:00Z",
        source="validated_teacher_rubric",
        authority_evidence_sha256=authority_hash,
    )


class StudentModelAdjudicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial = initialize_student_model(
            {
                "prerequisite": 0.2,
                "conceptual": 0.3,
                "procedural": 0.1,
                "transfer": 0.05,
            },
            goal=GOAL,
        )

    def test_approve_is_deeply_idempotent_and_does_not_append_receipt(self) -> None:
        authority_hash = "8" * 64
        model = _apply(
            self.initial,
            evidence_id="evidence.approve",
            authority_hash=authority_hash,
        )
        before = deepcopy(model)
        instruction = _instruction(
            operation="replay_original_assessment",
            evidence_id="evidence.approve",
            evidence_sha256=authority_hash,
        )
        first = apply_student_model_adjudication(
            model,
            target_kc_id="kc_state_definition",
            source_evidence_id="evidence.approve",
            instruction=instruction,
        )
        second = apply_student_model_adjudication(
            first["model"],
            target_kc_id="kc_state_definition",
            source_evidence_id="evidence.approve",
            instruction=instruction,
        )
        self.assertEqual(first["schema"], STUDENT_MODEL_ADJUDICATION_RESULT_SCHEMA)
        self.assertEqual(first["status"], "approved_no_change")
        self.assertEqual(second["status"], "approved_no_change")
        self.assertEqual(first["model"], before)
        self.assertEqual(second["model"], before)
        self.assertIsNone(first["revision_receipt"])
        self.assertEqual(model, before)

    def test_abstain_supersedes_then_replays_target_from_initial_prior(self) -> None:
        model = _apply(self.initial, evidence_id="evidence.keep.1", minute=1)
        source_hash = "9" * 64
        model = _apply(
            model,
            evidence_id="evidence.remove",
            signal="misconception",
            minute=2,
            authority_hash=source_hash,
        )
        model = _apply(model, evidence_id="evidence.keep.2", minute=3)
        model = _apply(
            model,
            evidence_id="evidence.other.kc",
            kc_id="kc_state_transition",
            minute=4,
        )
        other_before = deepcopy(model["knowledge_components"]["kc_state_transition"])
        instruction = _instruction(
            operation="do_not_replay",
            evidence_id="evidence.remove",
            evidence_sha256=source_hash,
        )
        result = supersede_and_replay_student_model_evidence(
            model,
            target_kc_id="kc_state_definition",
            source_evidence_id="evidence.remove",
            instruction=instruction,
        )
        self.assertEqual(result["status"], "applied_supersede_replay")
        revised = result["model"]
        target = revised["knowledge_components"]["kc_state_definition"]
        source = next(
            row
            for row in target["evidence_ledger"]
            if row["evidence_id"] == "evidence.remove"
        )
        self.assertEqual(source["lifecycle_status"], "superseded")
        self.assertEqual(
            source["superseded_by_revision_id"],
            result["revision_receipt"]["revision_id"],
        )
        self.assertEqual(target["evidence_count"], 2)
        self.assertEqual(target["evidence_ledger_state"]["active_evidence"], 2)
        self.assertEqual(target["evidence_ledger_state"]["superseded_evidence"], 1)
        self.assertTrue(target["evidence_ledger_state"]["complete_from_initial_prior"])
        self.assertEqual(
            revised["knowledge_components"]["kc_state_transition"], other_before
        )

        expected = _apply(self.initial, evidence_id="evidence.keep.1", minute=1)
        expected = _apply(expected, evidence_id="evidence.keep.2", minute=3)
        expected_target = expected["knowledge_components"]["kc_state_definition"]
        for field in (
            "p_mastery",
            "uncertainty",
            "evidence_count",
            "positive_evidence",
            "negative_evidence",
            "last_signal",
            "last_round",
            "last_observed_at",
            "last_time_basis",
            "last_evidence_id",
            "calibration",
            "dimensions",
        ):
            self.assertEqual(target[field], expected_target[field], field)
        receipt = result["revision_receipt"]
        self.assertEqual(receipt["schema"], STUDENT_MODEL_ADJUDICATION_REVISION_SCHEMA)
        self.assertFalse(receipt["negative_delta_applied"])
        self.assertEqual(receipt["active_evidence_replayed"], 2)
        validate_student_model(revised)

        replay = apply_student_model_adjudication(
            revised,
            target_kc_id="kc_state_definition",
            source_evidence_id="evidence.remove",
            instruction=instruction,
        )
        self.assertEqual(replay["status"], "already_applied_no_change")
        self.assertEqual(replay["model"], revised)

    def test_local_operator_correct_is_pending_and_never_changes_mastery(self) -> None:
        authority_hash = "a" * 64
        model = _apply(
            self.initial,
            evidence_id="evidence.correct",
            authority_hash=authority_hash,
        )
        instruction = _instruction(
            operation="supersede_and_replay",
            evidence_id="evidence.correct",
            evidence_sha256=authority_hash,
        )
        result = apply_student_model_adjudication(
            model,
            target_kc_id="kc_state_definition",
            source_evidence_id="evidence.correct",
            instruction=instruction,
        )
        self.assertEqual(result["status"], "pending_authority_revalidation")
        self.assertEqual(result["model"], model)
        self.assertFalse(result["pending"]["model_mutated"])
        self.assertFalse(result["pending"]["mastery_update_authorized"])
        self.assertEqual(model["adjudication_revisions"], [])
        entry = model["knowledge_components"]["kc_state_definition"]["evidence_ledger"][
            0
        ]
        self.assertEqual(entry["lifecycle_status"], "active")

    def test_local_operator_correct_stays_authority_pending_when_ledger_truncated(
        self,
    ) -> None:
        model = self.initial
        for index in range(65):
            model = _apply(
                model,
                evidence_id=f"evidence.correct.capacity.{index:02d}",
                minute=index,
            )
        component = model["knowledge_components"]["kc_state_definition"]
        self.assertFalse(
            component["evidence_ledger_state"]["complete_from_initial_prior"]
        )
        source = component["evidence_ledger"][-1]
        instruction = _instruction(
            operation="supersede_and_replay",
            evidence_id=source["evidence_id"],
            evidence_sha256=source["evidence_fingerprint"],
        )
        before = deepcopy(model)
        result = apply_student_model_adjudication(
            model,
            target_kc_id="kc_state_definition",
            source_evidence_id=source["evidence_id"],
            instruction=instruction,
        )
        self.assertEqual(result["status"], "pending_authority_revalidation")
        self.assertEqual(
            result["pending"]["reason"],
            "authenticated_teacher_and_rubric_authority_receipt_required",
        )
        self.assertFalse(result["pending"]["model_mutated"])
        self.assertEqual(result["model"], before)
        self.assertEqual(model, before)

    def test_missing_or_truncated_ledger_fails_closed_without_guessing(self) -> None:
        missing_instruction = _instruction(
            operation="do_not_replay",
            evidence_id="evidence.not.retained",
            evidence_sha256="b" * 64,
        )
        missing = apply_student_model_adjudication(
            self.initial,
            target_kc_id="kc_state_definition",
            source_evidence_id="evidence.not.retained",
            instruction=missing_instruction,
        )
        self.assertEqual(missing["status"], "pending_external_ledger")
        self.assertEqual(missing["model"], self.initial)

        model = self.initial
        for index in range(65):
            model = _apply(
                model,
                evidence_id=f"evidence.capacity.{index:02d}",
                minute=index,
            )
        target = model["knowledge_components"]["kc_state_definition"]
        self.assertEqual(len(target["evidence_ledger"]), 64)
        self.assertFalse(target["evidence_ledger_state"]["complete_from_initial_prior"])
        source = target["evidence_ledger"][-1]
        instruction = _instruction(
            operation="do_not_replay",
            evidence_id=source["evidence_id"],
            evidence_sha256=source["evidence_fingerprint"],
        )
        truncated = apply_student_model_adjudication(
            model,
            target_kc_id="kc_state_definition",
            source_evidence_id=source["evidence_id"],
            instruction=instruction,
        )
        self.assertEqual(truncated["status"], "pending_external_ledger")
        self.assertIn("truncated", truncated["pending"]["reason"])
        self.assertEqual(truncated["model"], model)

    def test_legacy_entry_without_round_is_pending_external_ledger(self) -> None:
        model = _apply(self.initial, evidence_id="evidence.legacy", minute=1)
        entry = model["knowledge_components"]["kc_state_definition"]["evidence_ledger"][
            0
        ]
        entry.pop("round_number")
        material_fields = (
            "evidence_id",
            "item_id",
            "question_id",
            "rubric_id",
            "knowledge_component_id",
            "focus_dimension",
            "signal",
            "confidence",
            "answer_alignment",
            "difficulty",
            "discrimination",
            "observed_at",
            "time_basis",
            "source",
            "assessment_eligible",
            "authoritative",
        )
        entry["evidence_fingerprint"] = canonical_sha256(
            {field: entry[field] for field in material_fields}
        )
        validate_student_model(model)
        instruction = _instruction(
            operation="do_not_replay",
            evidence_id="evidence.legacy",
            evidence_sha256=entry["evidence_fingerprint"],
        )
        result = apply_student_model_adjudication(
            model,
            target_kc_id="kc_state_definition",
            source_evidence_id="evidence.legacy",
            instruction=instruction,
        )
        self.assertEqual(result["status"], "pending_external_ledger")
        self.assertIn("round_number", result["pending"]["reason"])
        self.assertEqual(result["model"], model)

    def test_forged_instruction_or_evidence_binding_is_rejected(self) -> None:
        model = _apply(self.initial, evidence_id="evidence.secure", minute=1)
        entry = model["knowledge_components"]["kc_state_definition"]["evidence_ledger"][
            0
        ]
        instruction = _instruction(
            operation="do_not_replay",
            evidence_id="evidence.secure",
            evidence_sha256=entry["evidence_fingerprint"],
        )
        tampered = deepcopy(instruction)
        tampered["target_kc_ids"] = ["kc_state_transition"]
        with self.assertRaisesRegex(StudentModelError, "integrity"):
            apply_student_model_adjudication(
                model,
                target_kc_id="kc_state_definition",
                source_evidence_id="evidence.secure",
                instruction=tampered,
            )

        wrong_evidence = _instruction(
            operation="do_not_replay",
            evidence_id="evidence.secure",
            evidence_sha256="f" * 64,
        )
        with self.assertRaisesRegex(StudentModelError, "evidence hash"):
            apply_student_model_adjudication(
                model,
                target_kc_id="kc_state_definition",
                source_evidence_id="evidence.secure",
                instruction=wrong_evidence,
            )

    def test_revised_model_matches_live_session_json_schema(self) -> None:
        authority_hash = "c" * 64
        model = _apply(
            self.initial,
            evidence_id="evidence.schema",
            minute=1,
            authority_hash=authority_hash,
        )
        result = apply_student_model_adjudication(
            model,
            target_kc_id="kc_state_definition",
            source_evidence_id="evidence.schema",
            instruction=_instruction(
                operation="do_not_replay",
                evidence_id="evidence.schema",
                evidence_sha256=authority_hash,
            ),
        )
        schema = json.loads(
            (ROOT / "schema" / "teacher_agent_live_session.schema.json").read_text()
        )
        Draft202012Validator.check_schema(schema)
        student_model_schema = {
            "$schema": schema["$schema"],
            "$ref": "#/$defs/studentModel",
            "$defs": schema["$defs"],
        }
        Draft202012Validator(student_model_schema).validate(result["model"])


if __name__ == "__main__":
    unittest.main()
