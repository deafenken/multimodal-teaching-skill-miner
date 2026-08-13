from __future__ import annotations

import csv
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import jsonschema

from teaching_skill_miner.io_utils import read_json

from teaching_skill_miner.cli import main
from teaching_skill_miner.learner_effect_study import (
    ALLOCATION_FIELDS,
    OUTCOME_FIELDS,
    TEACHER_BURDEN_FIELDS,
    LearnerEffectStudyError,
    analyze_learner_effect_study,
    generate_learner_effect_study_package,
    sign_external_assessment_attestation,
    verify_external_assessment_attestation,
    write_learner_effect_analysis,
)


ROOT = Path(__file__).resolve().parents[1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _package(root: Path, *, data_origin: str = "synthetic", **kwargs: object) -> dict:
    return generate_learner_effect_study_package(
        root,
        study_id="synthetic-cluster-rct-2026",
        cluster_unit="teacher",
        cluster_count=6,
        participants_per_cluster=5,
        randomization_seed=71,
        bootstrap_seed=72,
        bootstrap_replicates=50,
        data_origin=data_origin,
        token_secret=b"test-only-private-token-secret-32",
        generated_at_utc="2026-07-23T00:00:00Z",
        **kwargs,
    )


def _fill_outcomes(root: Path, *, include_effect: bool = True) -> Path:
    allocation = _read_rows(root / "participant_allocation.csv")
    arm_by_token = {row["participant_token"]: row["assigned_arm"] for row in allocation}
    rows = _read_rows(root / "outcome_collection_template.csv")
    for index, row in enumerate(rows):
        pre = 30.0 + (index % 11) * 2.0 + (index // 11)
        effect = (
            10.0
            if include_effect and arm_by_token[row["participant_token"]] == "A"
            else 1.0
        )
        row.update(
            {
                "pre_score": f"{pre:.1f}",
                "post_score": f"{pre + effect:.1f}",
                "transfer_score": f"{pre + effect * 0.7:.1f}",
                "retention_score": f"{pre + effect * 0.8:.1f}",
                "primary_missing_reason": "not_missing",
                "transfer_missing_reason": "not_missing",
                "retention_missing_reason": "not_missing",
                "retention_assessment_day": "21",
                "subgroup_code": "sg01" if index % 2 == 0 else "sg02",
                "grader_blind": "true",
                "adverse_event_reported": "no",
                "adverse_event_severity": "none",
                "adverse_event_relatedness": "none",
                "protocol_deviation": "crossover" if index == 0 else "none",
            }
        )
    completed = root / "completed_outcomes.csv"
    _write_rows(completed, OUTCOME_FIELDS, rows)
    return completed


def _fill_teacher_burden(root: Path) -> Path:
    allocation = _read_rows(root / "participant_allocation.csv")
    arm_by_cluster = {row["cluster_token"]: row["assigned_arm"] for row in allocation}
    rows = _read_rows(root / "teacher_burden_collection_template.csv")
    for row in rows:
        intervention = arm_by_cluster[row["cluster_token"]] == "A"
        row.update(
            {
                "preparation_minutes": "40" if intervention else "25",
                "delivery_minutes": "55" if intervention else "50",
                "followup_minutes": "15" if intervention else "10",
                "workload_rating_1_to_5": "3" if intervention else "2",
                "burden_missing_reason": "not_missing",
            }
        )
    completed = root / "completed_teacher_burden.csv"
    _write_rows(completed, TEACHER_BURDEN_FIELDS, rows)
    return completed


class LearnerEffectStudyTests(unittest.TestCase):
    def test_templates_are_deterministic_token_only_and_never_positive(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            first_root = Path(first)
            second_root = Path(second)
            first_result = _package(first_root, data_origin="template")
            second_result = _package(second_root, data_origin="template")
            preregistration = json.loads(
                (first_root / "preregistration.json").read_text(encoding="utf-8")
            )
            jsonschema.Draft202012Validator(
                read_json(ROOT / "schema/learner_effect_preregistration.schema.json")
            ).validate(preregistration)
            allocation_text = (first_root / "participant_allocation.csv").read_text(
                encoding="utf-8"
            )
            outcomes_text = (first_root / "outcome_collection_template.csv").read_text(
                encoding="utf-8"
            )
            allocation = _read_rows(first_root / "participant_allocation.csv")
            outcomes = _read_rows(first_root / "outcome_collection_template.csv")

            self.assertEqual(
                allocation_text,
                (second_root / "participant_allocation.csv").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertEqual(
                first_result["preregistration_core_sha256"],
                second_result["preregistration_core_sha256"],
            )
            self.assertEqual(tuple(allocation[0]), ALLOCATION_FIELDS)
            self.assertEqual(tuple(outcomes[0]), OUTCOME_FIELDS)
            self.assertEqual(len(allocation), 30)
            self.assertTrue(
                all(
                    len(row["participant_token"]) == 64
                    and len(row["cluster_token"]) == 64
                    for row in allocation
                )
            )
            self.assertEqual({row["assigned_arm"] for row in allocation}, {"A", "B"})
            self.assertTrue(all(row["pre_score"] == "" for row in outcomes))
            self.assertTrue(
                (first_root / "teacher_burden_collection_template.csv").is_file()
            )
            forbidden = ("name", "email", "student_id", "teacher_id", "classroom_id")
            header_text = (
                allocation_text.splitlines()[0].casefold()
                + outcomes_text.splitlines()[0].casefold()
            )
            self.assertTrue(all(value not in header_text for value in forbidden))
            self.assertFalse(first_result["real_participant_data_generated"])
            self.assertFalse(first_result["learner_effectiveness_established"])
            self.assertEqual(
                first_result["external_validation_status"],
                "external_validation_pending",
            )
            self.assertFalse(preregistration["learner_effectiveness_established"])
            if os.name == "posix":
                self.assertEqual(first_root.stat().st_mode & 0o777, 0o700)
                self.assertEqual(
                    (first_root / "participant_allocation.csv").stat().st_mode & 0o777,
                    0o600,
                )

    def test_synthetic_positive_shaped_analysis_stays_unestablished(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _package(root)
            completed = _fill_outcomes(root)
            result = analyze_learner_effect_study(
                root / "preregistration.json",
                root / "participant_allocation.csv",
                completed,
            )
            output = write_learner_effect_analysis(root / "analysis.json", result)
            jsonschema.Draft202012Validator(
                read_json(ROOT / "schema/learner_effect_analysis.schema.json")
            ).validate(result)
            serialized = json.dumps(result, ensure_ascii=False)
            allocation = _read_rows(root / "participant_allocation.csv")

            self.assertGreater(
                result["primary_analysis"]["adjusted_standardized_effect"], 0
            )
            self.assertEqual(
                result["primary_analysis"]["cluster_bootstrap"]["replicates_requested"],
                50,
            )
            self.assertEqual(
                result["coverage_and_attrition"]["primary_outcome_coverage_fraction"],
                1.0,
            )
            self.assertTrue(result["design"]["analysis_is_intention_to_treat"])
            self.assertFalse(result["gates"]["real_participant_data_declared"])
            self.assertFalse(result["local_preregistered_positive_result_gate"])
            self.assertFalse(result["eligible_for_external_governance_review"])
            self.assertFalse(result["learner_effect_establishment_gate"])
            self.assertFalse(
                result["learner_effect_establishment_gates"][
                    "external_evidence_signature_verified"
                ]
            )
            self.assertFalse(result["learner_effectiveness_established"])
            self.assertFalse(
                result["primary_analysis"]["p_value_computed_by_this_minimum_tool"]
            )
            self.assertFalse(
                result["claim_boundary"][
                    "internal_skill_scores_are_learning_effectiveness"
                ]
            )
            self.assertTrue(output.is_file())
            self.assertNotIn(allocation[0]["participant_token"], serialized)

    def test_empty_extra_internal_score_and_wrong_mapping_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _package(root)
            with self.assertRaisesRegex(LearnerEffectStudyError, "pre_score must be"):
                analyze_learner_effect_study(
                    root / "preregistration.json",
                    root / "participant_allocation.csv",
                    root / "outcome_collection_template.csv",
                )

            completed = _fill_outcomes(root)
            rows = _read_rows(completed)
            extra = root / "outcomes_with_internal_skill_score.csv"
            extra_fields = OUTCOME_FIELDS + ("skill_score",)
            for row in rows:
                row["skill_score"] = "99"
            _write_rows(extra, extra_fields, rows)
            with self.assertRaisesRegex(LearnerEffectStudyError, "privacy-safe schema"):
                analyze_learner_effect_study(
                    root / "preregistration.json",
                    root / "participant_allocation.csv",
                    extra,
                )

            rows = _read_rows(completed)
            rows[0]["cluster_token"] = rows[5]["cluster_token"]
            wrong = root / "wrong_cluster.csv"
            _write_rows(wrong, OUTCOME_FIELDS, rows)
            with self.assertRaisesRegex(
                LearnerEffectStudyError, "token/cluster mapping"
            ):
                analyze_learner_effect_study(
                    root / "preregistration.json",
                    root / "participant_allocation.csv",
                    wrong,
                )

    def test_missing_post_is_counted_and_kept_in_fixed_itt_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _package(root)
            completed = _fill_outcomes(root)
            rows = _read_rows(completed)
            rows[-1]["post_score"] = ""
            rows[-1]["primary_missing_reason"] = "lost_to_followup"
            _write_rows(completed, OUTCOME_FIELDS, rows)
            result = analyze_learner_effect_study(
                root / "preregistration.json",
                root / "participant_allocation.csv",
                completed,
            )
        coverage = result["coverage_and_attrition"]
        self.assertEqual(coverage["observed_primary_outcome_count"], 29)
        self.assertEqual(coverage["baseline_carried_forward_count"], 1)
        self.assertEqual(coverage["primary_attrition_fraction"], 0.033333)
        self.assertTrue(result["design"]["analysis_is_intention_to_treat"])
        self.assertTrue(result["gates"]["all_randomized_participants_in_itt"])
        self.assertFalse(result["learner_effectiveness_established"])

    def test_transfer_retention_subgroup_safety_and_burden_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _package(root)
            completed = _fill_outcomes(root)
            burden = _fill_teacher_burden(root)
            result = analyze_learner_effect_study(
                root / "preregistration.json",
                root / "participant_allocation.csv",
                completed,
                burden,
            )
        self.assertEqual(
            result["secondary_analyses"]["retention_2_to_4_weeks"][
                "assessment_window_days"
            ],
            [14, 28],
        )
        self.assertEqual(
            result["coverage_and_attrition"]["transfer_outcome_coverage_fraction"],
            1.0,
        )
        self.assertEqual(
            result["subgroup_and_fairness"]["analysis_status"], "estimable"
        )
        self.assertEqual(
            result["teacher_burden"]["status"], "complete_for_external_review"
        )
        self.assertTrue(result["blinding_and_safety"]["automatic_safety_gate_passed"])
        self.assertFalse(result["learner_effectiveness_established"])

    def test_retention_outside_two_to_four_weeks_and_contradictory_ae_fail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _package(root)
            completed = _fill_outcomes(root)
            rows = _read_rows(completed)
            rows[0]["retention_assessment_day"] = "29"
            _write_rows(completed, OUTCOME_FIELDS, rows)
            with self.assertRaisesRegex(LearnerEffectStudyError, "2-4 week"):
                analyze_learner_effect_study(
                    root / "preregistration.json",
                    root / "participant_allocation.csv",
                    completed,
                )
            rows[0]["retention_assessment_day"] = "21"
            rows[0]["adverse_event_reported"] = "no"
            rows[0]["adverse_event_severity"] = "serious"
            rows[0]["adverse_event_relatedness"] = "probably"
            _write_rows(completed, OUTCOME_FIELDS, rows)
            with self.assertRaisesRegex(LearnerEffectStudyError, "contradict"):
                analyze_learner_effect_study(
                    root / "preregistration.json",
                    root / "participant_allocation.csv",
                    completed,
                )

    def test_serious_related_adverse_event_is_reported_and_fails_safety_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _package(root)
            completed = _fill_outcomes(root)
            rows = _read_rows(completed)
            rows[0]["adverse_event_reported"] = "yes"
            rows[0]["adverse_event_severity"] = "serious"
            rows[0]["adverse_event_relatedness"] = "probably"
            _write_rows(completed, OUTCOME_FIELDS, rows)
            result = analyze_learner_effect_study(
                root / "preregistration.json",
                root / "participant_allocation.csv",
                completed,
            )
        self.assertEqual(
            result["blinding_and_safety"]["serious_related_adverse_event_count"],
            1,
        )
        self.assertFalse(result["blinding_and_safety"]["automatic_safety_gate_passed"])
        self.assertFalse(result["gates"]["no_serious_related_adverse_event"])
        self.assertFalse(result["learner_effectiveness_established"])

    def test_independent_assessor_signature_binds_analysis_and_lockbox_but_stays_pending(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lockbox_path = root / "semantic-lockbox-report.json"
            lockbox = {
                "schema": "teaching_skill_miner.teacher_agent_semantic_lockbox_report.v1",
                "passed": True,
                "claim_boundary": {
                    "behavioral_thresholds_met": True,
                    "expert_gold_signature_verified": True,
                    "runtime_observations_derived_from_integrity_checked_sessions": True,
                    "real_students_involved": False,
                    "real_learning_effect_established": False,
                },
            }
            lockbox["content_sha256"] = hashlib.sha256(
                json.dumps(
                    lockbox,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            lockbox_path.write_text(
                json.dumps(lockbox, sort_keys=True), encoding="utf-8"
            )
            lockbox_sha256 = hashlib.sha256(lockbox_path.read_bytes()).hexdigest()
            _package(
                root,
                data_origin="real",
                ethics_approval_id="IRB-EXTERNAL-2026",
                informed_consent_or_approved_waiver=True,
                preregistration_frozen_before_allocation=True,
                allocation_concealment_procedure_declared=True,
                intervention_version="2.0.0",
                intervention_build_sha256="a" * 64,
                semantic_lockbox_report_sha256=lockbox_sha256,
                independent_assessor_organization="independent-assessment-lab",
                independent_assessor_key_id="assessor-key-2026",
                assessor_independence_declared=True,
                subgroup_definitions_registry_sha256="b" * 64,
            )
            completed = _fill_outcomes(root)
            burden = _fill_teacher_burden(root)
            analysis = analyze_learner_effect_study(
                root / "preregistration.json",
                root / "participant_allocation.csv",
                completed,
                burden,
            )
            private_key = Ed25519PrivateKey.generate()
            private_pem = private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            public_pem = private_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            attestation = sign_external_assessment_attestation(
                analysis,
                private_key_pem=private_pem,
                assessor_organization="independent-assessment-lab",
                assessor_key_id="assessor-key-2026",
                signed_at_utc="2026-08-01T00:00:00Z",
            )
            jsonschema.Draft202012Validator(
                read_json(
                    ROOT
                    / "schema/learner_effect_external_assessment_attestation.schema.json"
                )
            ).validate(attestation)
            receipt = verify_external_assessment_attestation(
                attestation,
                analysis,
                trusted_public_key_pem=public_pem,
                expected_assessor_key_id="assessor-key-2026",
                semantic_lockbox_report_path=lockbox_path,
            )
            self.assertTrue(receipt["signature_verified"])
            self.assertEqual(
                receipt["external_validation_status"], "external_validation_pending"
            )
            self.assertFalse(receipt["learner_effectiveness_established"])
            tampered = json.loads(json.dumps(analysis))
            tampered["primary_analysis"]["adjusted_standardized_effect"] = 99
            with self.assertRaisesRegex(LearnerEffectStudyError, "exact analysis"):
                verify_external_assessment_attestation(
                    attestation,
                    tampered,
                    trusted_public_key_pem=public_pem,
                    expected_assessor_key_id="assessor-key-2026",
                    semantic_lockbox_report_path=lockbox_path,
                )

    def test_cli_generates_only_unestablished_templates(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            redirect_stdout(io.StringIO()) as output,
        ):
            code = main(
                [
                    "prepare-learner-effect-study",
                    "--output-dir",
                    temporary,
                    "--study-id",
                    "cli-template-study-2026",
                    "--bootstrap-replicates",
                    "20",
                ]
            )
            summary = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertFalse(summary["identity_fields_included"])
        self.assertFalse(summary["real_participant_data_generated"])
        self.assertFalse(summary["learner_effectiveness_established"])


if __name__ == "__main__":
    unittest.main()
