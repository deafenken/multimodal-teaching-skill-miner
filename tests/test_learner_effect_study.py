from __future__ import annotations

import csv
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from teaching_skill_miner.cli import main
from teaching_skill_miner.learner_effect_study import (
    ALLOCATION_FIELDS,
    OUTCOME_FIELDS,
    LearnerEffectStudyError,
    analyze_learner_effect_study,
    generate_learner_effect_study_package,
    write_learner_effect_analysis,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _package(root: Path, *, data_origin: str = "synthetic") -> dict:
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
    )


def _fill_outcomes(root: Path, *, include_effect: bool = True) -> Path:
    allocation = _read_rows(root / "participant_allocation.csv")
    arm_by_token = {row["participant_token"]: row["assigned_arm"] for row in allocation}
    rows = _read_rows(root / "outcome_collection_template.csv")
    for index, row in enumerate(rows):
        pre = 30.0 + (index % 11) * 2.0 + (index // 11)
        effect = 10.0 if include_effect and arm_by_token[row["participant_token"]] == "A" else 1.0
        row.update(
            {
                "pre_score": f"{pre:.1f}",
                "post_score": f"{pre + effect:.1f}",
                "retention_score": f"{pre + effect * 0.8:.1f}",
                "primary_missing_reason": "not_missing",
                "retention_missing_reason": "not_missing",
                "grader_blind": "true",
                "adverse_event_reported": "no",
                "protocol_deviation": "crossover" if index == 0 else "none",
            }
        )
    completed = root / "completed_outcomes.csv"
    _write_rows(completed, OUTCOME_FIELDS, rows)
    return completed


class LearnerEffectStudyTests(unittest.TestCase):
    def test_templates_are_deterministic_token_only_and_never_positive(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            first_result = _package(first_root, data_origin="template")
            second_result = _package(second_root, data_origin="template")
            preregistration = json.loads(
                (first_root / "preregistration.json").read_text(encoding="utf-8")
            )
            allocation_text = (first_root / "participant_allocation.csv").read_text(
                encoding="utf-8"
            )
            outcomes_text = (
                first_root / "outcome_collection_template.csv"
            ).read_text(encoding="utf-8")
            allocation = _read_rows(first_root / "participant_allocation.csv")
            outcomes = _read_rows(first_root / "outcome_collection_template.csv")

            self.assertEqual(
                allocation_text,
                (second_root / "participant_allocation.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(first_result["preregistration_core_sha256"], second_result["preregistration_core_sha256"])
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
            self.assertEqual(
                {row["assigned_arm"] for row in allocation}, {"A", "B"}
            )
            self.assertTrue(all(row["pre_score"] == "" for row in outcomes))
            forbidden = ("name", "email", "student_id", "teacher_id", "classroom_id")
            header_text = allocation_text.splitlines()[0].casefold() + outcomes_text.splitlines()[0].casefold()
            self.assertTrue(all(value not in header_text for value in forbidden))
            self.assertFalse(first_result["real_participant_data_generated"])
            self.assertFalse(first_result["learner_effectiveness_established"])
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
                result["claim_boundary"]["internal_skill_scores_are_learning_effectiveness"]
            )
            self.assertTrue(output.is_file())
            self.assertNotIn(allocation[0]["participant_token"], serialized)

    def test_empty_extra_internal_score_and_wrong_mapping_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _package(root)
            with self.assertRaisesRegex(
                LearnerEffectStudyError, "pre_score must be"
            ):
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
            with self.assertRaisesRegex(
                LearnerEffectStudyError, "privacy-safe schema"
            ):
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

    def test_cli_generates_only_unestablished_templates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(
            io.StringIO()
        ) as output:
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
