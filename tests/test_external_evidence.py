from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional recognition dependency
    CRYPTOGRAPHY_AVAILABLE = False

from teaching_skill_miner.audit import audit_dataset
from teaching_skill_miner.cli import main
from teaching_skill_miner.delivery import verify_delivery
from teaching_skill_miner.external_evidence import (
    ExternalEvidenceError,
    finalize_external_research_evidence,
    sign_external_research_evidence,
    validate_external_research_evidence,
    verify_external_research_evidence_attestation,
    verify_external_research_evidence_files,
)
from teaching_skill_miner.io_utils import (
    project_root,
    read_json,
    resolve_manifest_root,
    write_json,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _common_draft(kind: str) -> dict:
    return {
        "schema_version": "1.0",
        "protocol": "externally_governed_research_evidence_v1",
        "evidence_id": f"synthetic-{kind}-evidence",
        "evidence_kind": kind,
        "study_id": f"synthetic-{kind}-study",
        "registered_at_utc": "2026-01-01T00:00:00Z",
        "completed_at_utc": "2026-02-01T00:00:00Z",
        "preregistration": {
            "registry_name": "synthetic test-only registry",
            "registration_id": f"test-only-{kind}",
            "protocol_sha256": _digest(f"{kind}-protocol"),
            "statistical_analysis_plan_sha256": _digest(f"{kind}-sap"),
            "frozen_before_outcome_access": True,
        },
        "governance": {
            "independent_external_governance": True,
            "data_or_analysis_independent_of_system_developers": True,
            "data_not_used_to_modify_system_or_claim_thresholds": True,
            "conflicts_of_interest_disclosed": True,
            "complete_registered_outcome_reporting": True,
            "trusted_key_provisioned_out_of_band": True,
        },
        "population": {},
        "artifact_bindings": {
            "dataset_or_study_data_fingerprint": _digest(f"{kind}-data"),
            "evaluated_system_fingerprint": _digest(f"{kind}-system"),
            "comparator_fingerprint": _digest(f"{kind}-comparator"),
            "analysis_code_sha256": _digest(f"{kind}-analysis"),
            "statistical_output_sha256": _digest(f"{kind}-output"),
            "aggregate_result_table_sha256": _digest(f"{kind}-table"),
        },
        "claim": {},
    }


def _multimodal_draft() -> dict:
    value = _common_draft("confirmatory_multimodal_gain")
    value["population"] = {
        "real_world_data": True,
        "data_context": "real_classroom",
        "site_count": 2,
        "independent_site_count": 2,
        "participant_count": 12,
        "eligible_unit_count": 100,
        "evaluated_unit_count": 90,
        "excluded_unit_count": 10,
        "exclusion_reason_counts": {"preregistered_quality_exclusion": 10},
        "coverage_fraction": 0.9,
        "coverage_evidence_sha256": _digest("multimodal-coverage"),
        "identity_disjoint_from_development": True,
    }
    value["claim"] = {
        "claim_type": "confirmatory_multimodal_gain",
        "evaluation_design": "paired_same_samples_external_lockbox",
        "baseline_modalities": ["visual"],
        "added_modalities": ["audio"],
        "primary_metric": "macro_f1",
        "baseline_estimate": 0.72,
        "multimodal_estimate": 0.80,
        "absolute_gain": 0.08,
        "confidence_level": 0.95,
        "confidence_interval_lower": 0.03,
        "confidence_interval_upper": 0.13,
        "p_value": 0.01,
        "alpha": 0.05,
        "minimum_confirmatory_gain": 0.02,
        "paired_cluster_aware_inference": True,
        "claim_cluster_field": "session_id",
        "claim_cluster_count": 12,
        "all_registered_primary_analyses_reported": True,
        "multimodal_gain_established": True,
    }
    return value


def _learner_draft() -> dict:
    value = _common_draft("real_learner_effectiveness")
    value["population"] = {
        "real_world_data": True,
        "data_context": "real_learner_study",
        "site_count": 2,
        "independent_site_count": 2,
        "participant_count": 60,
        "eligible_unit_count": 60,
        "evaluated_unit_count": 54,
        "excluded_unit_count": 6,
        "exclusion_reason_counts": {"missing_primary_outcome": 6},
        "coverage_fraction": 0.9,
        "coverage_evidence_sha256": _digest("learner-coverage"),
        "identity_disjoint_from_development": True,
    }
    value["claim"] = {
        "claim_type": "real_learner_effectiveness",
        "study_design": "individual_randomized_controlled_trial",
        "randomization_unit": "learner",
        "comparator_description": "preregistered active teaching comparator",
        "primary_outcome_name": "delayed transfer assessment",
        "effect_measure": "standardized_mean_difference",
        "effect_direction": "positive_favors_teaching_skill",
        "adjusted_effect_estimate": 0.35,
        "confidence_level": 0.95,
        "confidence_interval_lower": 0.12,
        "confidence_interval_upper": 0.58,
        "p_value": 0.01,
        "alpha": 0.05,
        "minimum_educationally_meaningful_effect": 0.10,
        "random_assignment": True,
        "allocation_concealment": True,
        "intention_to_treat": True,
        "independent_outcome_assessment": True,
        "randomized_unit_count": 60,
        "analysis_cluster_count": 0,
        "maximum_preregistered_attrition_fraction": 0.20,
        "observed_attrition_fraction": 0.10,
        "attrition_handled_under_preregistered_plan": True,
        "adverse_events_reported": True,
        "all_registered_primary_outcomes_reported": True,
        "ethics_approval_id": "TEST-ONLY-IRB-001",
        "informed_consent_or_approved_waiver": True,
        "learner_effectiveness_established": True,
    }
    return value


def _key_material() -> tuple[bytes, bytes]:
    private = Ed25519PrivateKey.generate()
    return (
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ),
    )


@unittest.skipUnless(CRYPTOGRAPHY_AVAILABLE, "cryptography is optional")
class ExternalResearchEvidenceTests(unittest.TestCase):
    def test_both_claim_kinds_are_recomputed_signed_and_verified(self) -> None:
        private_pem, public_pem = _key_material()
        for draft in (_multimodal_draft(), _learner_draft()):
            with self.subTest(kind=draft["evidence_kind"]):
                evidence = finalize_external_research_evidence(draft)
                validation = validate_external_research_evidence(evidence)
                self.assertTrue(validation["claim_established"])
                attestation = sign_external_research_evidence(
                    evidence,
                    private_key_pem=private_pem,
                    issuer="synthetic-test-only-governance",
                    key_id="test-key-2026",
                )
                verified = verify_external_research_evidence_attestation(
                    attestation,
                    trusted_public_key_pem=public_pem,
                    expected_kind=draft["evidence_kind"],
                    expected_evidence=evidence,
                )
                self.assertTrue(verified["signature_verified"])
                self.assertTrue(verified["claim_established"])
                self.assertFalse(
                    verified["signature_alone_proves_signer_independence"]
                )

    def test_negative_study_can_be_authentic_without_establishing_claim(self) -> None:
        draft = _multimodal_draft()
        draft["claim"]["p_value"] = 0.40
        draft["claim"]["multimodal_gain_established"] = False
        evidence = finalize_external_research_evidence(draft)
        private_pem, public_pem = _key_material()
        attestation = sign_external_research_evidence(
            evidence,
            private_key_pem=private_pem,
            issuer="synthetic-test-only-governance",
            key_id="test-key-2026",
        )
        verified = verify_external_research_evidence_attestation(
            attestation,
            trusted_public_key_pem=public_pem,
            expected_kind="confirmatory_multimodal_gain",
        )
        self.assertTrue(verified["signature_verified"])
        self.assertFalse(verified["claim_established"])
        self.assertFalse(verified["gates"]["confirmatory_p_value_passes"])

    def test_fewer_than_ten_claim_clusters_cannot_establish_gain(self) -> None:
        draft = _multimodal_draft()
        draft["claim"]["claim_cluster_count"] = 9
        draft["claim"]["multimodal_gain_established"] = False
        evidence = finalize_external_research_evidence(draft)
        validation = validate_external_research_evidence(evidence)
        self.assertFalse(validation["claim_established"])
        self.assertFalse(
            validation["gates"]["at_least_ten_independent_claim_clusters"]
        )
        draft["claim"]["multimodal_gain_established"] = True
        with self.assertRaisesRegex(
            ExternalEvidenceError, "does not equal the recomputed gates"
        ):
            finalize_external_research_evidence(draft)

    def test_teacher_cluster_rct_can_use_existing_external_evidence_chain(self) -> None:
        draft = _learner_draft()
        draft["claim"].update(
            {
                "study_design": "cluster_randomized_controlled_trial",
                "randomization_unit": "teacher",
                "randomized_unit_count": 6,
                "analysis_cluster_count": 6,
            }
        )
        evidence = finalize_external_research_evidence(draft)
        validation = validate_external_research_evidence(evidence)
        self.assertTrue(validation["claim_established"])
        self.assertEqual(validation["metrics"]["randomization_unit"], "teacher")

    def test_self_reported_true_cannot_override_weak_statistics(self) -> None:
        weak = _learner_draft()
        weak["claim"]["confidence_interval_lower"] = 0.05
        with self.assertRaisesRegex(
            ExternalEvidenceError, "does not equal the recomputed gates"
        ):
            finalize_external_research_evidence(weak)

    def test_coverage_and_exact_manifest_tampering_fail_closed(self) -> None:
        evidence = finalize_external_research_evidence(_multimodal_draft())
        private_pem, public_pem = _key_material()
        attestation = sign_external_research_evidence(
            evidence,
            private_key_pem=private_pem,
            issuer="synthetic-test-only-governance",
            key_id="test-key-2026",
        )
        bad_coverage = copy.deepcopy(evidence)
        bad_coverage["population"]["evaluated_unit_count"] = 89
        with self.assertRaisesRegex(ExternalEvidenceError, "eligible count"):
            validate_external_research_evidence(bad_coverage)
        altered_manifest = copy.deepcopy(evidence)
        altered_manifest["artifact_bindings"]["analysis_code_sha256"] = _digest(
            "altered-analysis"
        )
        with self.assertRaisesRegex(ExternalEvidenceError, "fingerprint mismatch"):
            validate_external_research_evidence(altered_manifest)
        with self.assertRaisesRegex(ExternalEvidenceError, "differs from signed"):
            verify_external_research_evidence_attestation(
                attestation,
                trusted_public_key_pem=public_pem,
                expected_evidence=altered_manifest,
            )

    def test_signature_tampering_and_wrong_trusted_key_fail(self) -> None:
        evidence = finalize_external_research_evidence(_multimodal_draft())
        private_pem, public_pem = _key_material()
        attestation = sign_external_research_evidence(
            evidence,
            private_key_pem=private_pem,
            issuer="synthetic-test-only-governance",
            key_id="test-key-2026",
        )
        tampered = copy.deepcopy(attestation)
        signature = tampered["signature_base64"]
        tampered["signature_base64"] = (
            ("A" if signature[0] != "A" else "B") + signature[1:]
        )
        with self.assertRaisesRegex(ExternalEvidenceError, "signature verification"):
            verify_external_research_evidence_attestation(
                tampered, trusted_public_key_pem=public_pem
            )
        tampered_statement = copy.deepcopy(attestation)
        tampered_statement["statement"]["issuer"] = "altered-test-only-governance"
        with self.assertRaisesRegex(ExternalEvidenceError, "signature verification"):
            verify_external_research_evidence_attestation(
                tampered_statement, trusted_public_key_pem=public_pem
            )
        _, wrong_public_pem = _key_material()
        with self.assertRaisesRegex(ExternalEvidenceError, "public-key fingerprint"):
            verify_external_research_evidence_attestation(
                attestation, trusted_public_key_pem=wrong_public_pem
            )

    def test_cli_prepare_sign_and_verify_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_pem, public_pem = _key_material()
            draft_path = root / "draft.json"
            evidence_path = root / "evidence.json"
            private_path = root / "private.pem"
            public_path = root / "public.pem"
            attestation_path = root / "attestation.json"
            report_path = root / "verification.json"
            system_artifact_path = root / "synthetic-test-only-system.whl"
            system_artifact_path.write_bytes(b"synthetic test-only system artifact")
            learner_draft = _learner_draft()
            learner_draft["artifact_bindings"]["evaluated_system_fingerprint"] = (
                hashlib.sha256(system_artifact_path.read_bytes()).hexdigest()
            )
            write_json(draft_path, learner_draft)
            private_path.write_bytes(private_pem)
            public_path.write_bytes(public_pem)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "prepare-external-research-evidence",
                            "--input",
                            str(draft_path),
                            "--output",
                            str(evidence_path),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "sign-external-research-evidence",
                            "--evidence",
                            str(evidence_path),
                            "--private-key",
                            str(private_path),
                            "--issuer",
                            "synthetic-test-only-governance",
                            "--key-id",
                            "test-key-2026",
                            "--output",
                            str(attestation_path),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "verify-external-research-evidence",
                            "--evidence",
                            str(evidence_path),
                            "--attestation",
                            str(attestation_path),
                            "--trusted-public-key",
                            str(public_path),
                            "--expected-system-artifact",
                            str(system_artifact_path),
                            "--expected-kind",
                            "real_learner_effectiveness",
                            "--require-claim-established",
                            "--output",
                            str(report_path),
                        ]
                    ),
                    0,
                )
            self.assertTrue(read_json(report_path)["claim_established"])

    def test_valid_future_external_evidence_can_complete_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            key_pairs = [_key_material(), _key_material()]
            paths: dict[str, Path] = {}
            system_artifact = temp / "synthetic-test-only-release.whl"
            system_artifact.write_bytes(b"synthetic test-only frozen release artifact")
            system_fingerprint = hashlib.sha256(
                system_artifact.read_bytes()
            ).hexdigest()
            for name, draft, keys in (
                ("multimodal", _multimodal_draft(), key_pairs[0]),
                ("learner", _learner_draft(), key_pairs[1]),
            ):
                draft["artifact_bindings"][
                    "evaluated_system_fingerprint"
                ] = system_fingerprint
                evidence = finalize_external_research_evidence(draft)
                attestation = sign_external_research_evidence(
                    evidence,
                    private_key_pem=keys[0],
                    issuer=f"synthetic-test-only-{name}-governance",
                    key_id=f"test-{name}-key-2026",
                )
                paths[f"{name}_evidence"] = write_json(
                    temp / f"{name}-evidence.json", evidence
                )
                paths[f"{name}_attestation"] = write_json(
                    temp / f"{name}-attestation.json", attestation
                )
                public_path = temp / f"{name}-public.pem"
                public_path.write_bytes(keys[1])
                paths[f"{name}_public"] = public_path

            root = project_root()
            manifest_path = root / "data/dataset_manifest.json"
            manifest = read_json(manifest_path)
            local_dataset_report = audit_dataset(
                manifest, resolve_manifest_root(manifest_path, manifest)
            )
            local_dataset_report["formal_empirical_ready"] = True
            dummy_review = temp / "human-review.csv"
            dummy_review.write_text("test-only\n", encoding="utf-8")
            with (
                patch(
                    "teaching_skill_miner.delivery.audit_dataset",
                    return_value=local_dataset_report,
                ),
                patch(
                    "teaching_skill_miner.delivery.summarize_human_review",
                    return_value={
                        "validation_status": "complete",
                        "passed": True,
                        "reviewed_skill_count": 10,
                    },
                ),
                patch(
                    "teaching_skill_miner.delivery._external_deployment_status",
                    return_value={"available": True, "passed": True},
                ),
            ):
                missing_artifact_report = verify_delivery(
                    manifest_path,
                    root / "data/evaluation_cases.json",
                    human_review_path=dummy_review,
                    external_multimodal_evidence_path=paths["multimodal_evidence"],
                    external_multimodal_attestation_path=paths[
                        "multimodal_attestation"
                    ],
                    trusted_multimodal_public_key_path=paths["multimodal_public"],
                    external_learner_evidence_path=paths["learner_evidence"],
                    external_learner_attestation_path=paths["learner_attestation"],
                    trusted_learner_public_key_path=paths["learner_public"],
                )
                report = verify_delivery(
                    manifest_path,
                    root / "data/evaluation_cases.json",
                    human_review_path=dummy_review,
                    external_multimodal_evidence_path=paths["multimodal_evidence"],
                    external_multimodal_attestation_path=paths[
                        "multimodal_attestation"
                    ],
                    trusted_multimodal_public_key_path=paths["multimodal_public"],
                    external_learner_evidence_path=paths["learner_evidence"],
                    external_learner_attestation_path=paths["learner_attestation"],
                    trusted_learner_public_key_path=paths["learner_public"],
                    external_evaluated_system_artifact_path=system_artifact,
                )
            self.assertFalse(
                missing_artifact_report[
                    "external_evidence_checks"
                ]["confirmatory_external_multimodal_gain"]
            )
            self.assertFalse(
                missing_artifact_report[
                    "external_evidence_checks"
                ]["real_learner_effectiveness_established"]
            )
            self.assertIn(
                "without the exact local evaluated-system artifact",
                missing_artifact_report[
                    "external_confirmatory_multimodal_validation"
                ]["reason"],
            )
            self.assertTrue(report["engineering_delivery_ready"])
            self.assertTrue(report["research_validation_complete"])
            self.assertEqual(report["overall_status"], "complete")
            self.assertEqual(report["external_blockers"], [])
            self.assertTrue(
                report["external_confirmatory_multimodal_validation"]["passed"]
            )
            self.assertTrue(
                report["external_learner_effectiveness_validation"]["passed"]
            )

    def test_file_verifier_requires_exact_three_file_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system_artifact = root / "synthetic-test-only-release.whl"
            system_artifact.write_bytes(b"synthetic test-only release artifact")
            draft = _multimodal_draft()
            draft["artifact_bindings"]["evaluated_system_fingerprint"] = (
                hashlib.sha256(system_artifact.read_bytes()).hexdigest()
            )
            evidence = finalize_external_research_evidence(draft)
            private_pem, public_pem = _key_material()
            attestation = sign_external_research_evidence(
                evidence,
                private_key_pem=private_pem,
                issuer="synthetic-test-only-governance",
                key_id="test-key-2026",
            )
            evidence_path = write_json(root / "evidence.json", evidence)
            attestation_path = write_json(root / "attestation.json", attestation)
            public_path = root / "public.pem"
            public_path.write_bytes(public_pem)
            verified = verify_external_research_evidence_files(
                evidence_path,
                attestation_path,
                public_path,
                expected_kind="confirmatory_multimodal_gain",
                expected_system_artifact_path=system_artifact,
            )
            self.assertTrue(verified["claim_established"])
            self.assertTrue(verified["system_artifact_binding_verified"])
            wrong_artifact = root / "different-system.whl"
            wrong_artifact.write_bytes(b"different system artifact")
            with self.assertRaisesRegex(
                ExternalEvidenceError, "evaluated-system fingerprint"
            ):
                verify_external_research_evidence_files(
                    evidence_path,
                    attestation_path,
                    public_path,
                    expected_system_artifact_path=wrong_artifact,
                )
            with self.assertRaisesRegex(ExternalEvidenceError, "does not exist"):
                verify_external_research_evidence_files(
                    evidence_path,
                    attestation_path,
                    root / "unprovisioned-public.pem",
                )
