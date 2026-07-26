from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import jsonschema

try:
    import numpy as np
    import sklearn  # noqa: F401
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    RECOGNITION_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency environment
    np = None
    RECOGNITION_DEPS_AVAILABLE = False

from teaching_skill_miner.recognition.strict_evaluation import (
    ClaimContract,
    CredibilityError,
    evaluate_frozen_external_deployment,
    fit_frozen_deployment_model,
    frozen_model_to_artifact,
    load_frozen_deployment_model,
    nested_grouped_multimodal_evaluation,
    strict_dataset_fingerprint,
    strict_coverage_evidence_fingerprint,
    strict_feature_bundle_fingerprint,
    validate_feature_provenance,
    validate_strict_manifest,
)
from teaching_skill_miner.attestation import (
    build_freeze_registration_request,
    sign_freeze_registration,
)
from teaching_skill_miner.io_utils import project_root, read_json


def _strict_fixture(
    *,
    prefix: str,
    session_count: int,
    site_id: str,
    external: bool = False,
) -> tuple[dict, dict[str, "np.ndarray"], dict[str, list[str]]]:
    records = []
    visual_rows = []
    audio_rows = []
    for session_index in range(session_count):
        for label in range(3):
            sample_id = f"{prefix}-s{session_index:02d}-y{label}"
            content_hash = hashlib.sha256(f"content:{sample_id}".encode()).hexdigest()
            records.append(
                {
                    "sample_id": sample_id,
                    "content_sha256": content_hash,
                    "label": label,
                    "label_name": ("low", "medium", "high")[label],
                    "session_id": f"{prefix}-session-{session_index:02d}",
                    "teacher_id": f"{prefix}-teacher-{session_index:02d}",
                    "site_id": site_id,
                    "modalities": {"visual": True, "audio": True},
                }
            )
            # Each single modality collapses two classes. Together they identify all three.
            visual_rows.append([-2.0 if label == 0 else 0.0, session_index / 1000.0])
            audio_rows.append([2.0 if label == 2 else 0.0, -session_index / 1000.0])
    audit = {
        "dataset_id": prefix,
        "provenance_verified": True,
        "real_classroom_recording": True,
        "independent_human_ground_truth": True,
        "synchronized_modalities_verified": True,
        "identity_metadata_verified": {
            "session_id": True,
            "teacher_id": True,
            "site_id": True,
        },
        "identity_metadata_source": "publisher metadata table",
        "ground_truth_source": "independent double annotation",
        "held_out_from_model_development": bool(external),
        "deployment_target_documented": bool(external),
        "prospective_deployment_collection": bool(external),
        "one_time_lockbox_evaluation": bool(external),
        "evaluation_coverage_fraction": 1.0 if external else None,
    }
    audit["dataset_fingerprint"] = strict_dataset_fingerprint(records)
    if external:
        coverage = {
            "eligible_sample_count": len(records),
            "evaluated_sample_count": len(records),
            "excluded_sample_count": 0,
            "eligible_population_fingerprint": hashlib.sha256(
                "\n".join(record["sample_id"] for record in records).encode()
            ).hexdigest(),
            "denominator_source": "frozen external test roster",
            "exclusion_reason_counts": {},
        }
        coverage["coverage_evidence_fingerprint"] = (
            strict_coverage_evidence_fingerprint(coverage)
        )
        audit["evaluation_coverage"] = coverage
    return (
        {"schema_version": "2.0", "audit": audit, "records": records},
        {
            "visual": np.asarray(visual_rows, dtype=float),
            "audio": np.asarray(audio_rows, dtype=float),
        },
        {"visual": ["v_class0", "v_session_noise"], "audio": ["a_class2", "a_session_noise"]},
    )


def _feature_provenance() -> dict[str, dict]:
    return {
        modality: {
            "extractor_id": f"fixed-{modality}-fixture-v1",
            "extractor_fingerprint": hashlib.sha256(
                f"fixed-{modality}-fixture-v1".encode()
            ).hexdigest(),
            "frozen_before_evaluation": True,
            "uses_ground_truth_labels": False,
            "fitted_on_evaluation_records": False,
        }
        for modality in ("visual", "audio")
    }


def _registered_evaluation_kwargs(
    model,
    external_manifest: dict,
    external_features: dict,
    external_names: dict,
    ledger_root: Path,
    *,
    registration_id: str,
) -> dict:
    artifact = frozen_model_to_artifact(model)
    request = build_freeze_registration_request(
        registration_id=registration_id,
        model_fingerprint=model.model_fingerprint,
        claim_contract_fingerprint=artifact["claim_contract_fingerprint"],
        training_dataset_fingerprint=model.training_dataset_fingerprint,
        training_feature_bundle_fingerprint=model.training_feature_bundle_fingerprint,
        external_dataset_fingerprint=external_manifest["audit"]["dataset_fingerprint"],
        external_feature_bundle_fingerprint=strict_feature_bundle_fingerprint(
            external_manifest["records"],
            external_names,
            external_features,
            _feature_provenance(),
        ),
        external_coverage_evidence_fingerprint=external_manifest["audit"][
            "evaluation_coverage"
        ]["coverage_evidence_fingerprint"],
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
    attestation = sign_freeze_registration(
        request,
        private_key_pem=private_pem,
        issuer="independent-test-custodian",
        key_id="test-key-2026",
    )
    return {
        "registration_attestation": attestation,
        "trusted_public_key_pem": public_pem,
        "one_time_ledger_dir": ledger_root,
    }


@unittest.skipUnless(RECOGNITION_DEPS_AVAILABLE, "recognition dependencies are optional")
class StrictManifestTests(unittest.TestCase):
    def test_malformed_record_and_modalities_fail_as_credibility_errors(self) -> None:
        with self.assertRaisesRegex(CredibilityError, "record 0"):
            strict_dataset_fingerprint(["not-an-object"])
        with self.assertRaisesRegex(CredibilityError, "modality"):
            strict_dataset_fingerprint(
                [
                    {
                        "sample_id": "sample-1",
                        "content_sha256": "a" * 64,
                        "label": 0,
                        "session_id": "session-1",
                        "modalities": ["visual"],
                    }
                ]
            )
    def test_missing_verified_session_metadata_fails_closed(self) -> None:
        manifest, _, _ = _strict_fixture(prefix="source", session_count=3, site_id="site-a")
        manifest["audit"]["identity_metadata_verified"]["session_id"] = False

        with self.assertRaisesRegex(CredibilityError, "session_id"):
            validate_strict_manifest(manifest, required_modalities=["visual", "audio"])

    def test_mutating_identity_after_fingerprinting_is_detected(self) -> None:
        manifest, _, _ = _strict_fixture(prefix="source", session_count=3, site_id="site-a")
        for record in manifest["records"]:
            if record["session_id"] == "source-session-00":
                record["teacher_id"] = "silently-changed"

        with self.assertRaisesRegex(CredibilityError, "dataset_fingerprint"):
            validate_strict_manifest(manifest, required_modalities=["visual", "audio"])

    def test_participant_task_does_not_invent_teacher_identity(self) -> None:
        manifest, _, _ = _strict_fixture(prefix="student", session_count=3, site_id="site-a")
        for index, record in enumerate(manifest["records"]):
            record["participant_id"] = f"participant-{index % 4:02d}"
            record["cohort_id"] = "cohort-a"
            record.pop("teacher_id")
        manifest["audit"]["identity_metadata_verified"] = {
            "session_id": True,
            "participant_id": True,
            "cohort_id": True,
            "site_id": True,
            "teacher_id": False,
        }
        manifest["audit"]["dataset_fingerprint"] = strict_dataset_fingerprint(
            manifest["records"]
        )

        summary = validate_strict_manifest(
            manifest,
            required_modalities=["visual", "audio"],
            required_identity_fields=[
                "session_id",
                "participant_id",
                "cohort_id",
                "site_id",
            ],
        )

        self.assertEqual(summary["participant_count"], 4)
        self.assertNotIn("teacher_count", summary)

    def test_feature_extractor_fitted_on_evaluation_records_fails_closed(self) -> None:
        provenance = _feature_provenance()
        provenance["audio"]["fitted_on_evaluation_records"] = True

        with self.assertRaisesRegex(CredibilityError, "fitted_on_evaluation_records"):
            validate_feature_provenance(provenance, ["visual", "audio"])


@unittest.skipUnless(RECOGNITION_DEPS_AVAILABLE, "recognition dependencies are optional")
class NestedStrictEvaluationTests(unittest.TestCase):
    def test_nested_session_evaluation_establishes_complementary_gain(self) -> None:
        manifest, features, _ = _strict_fixture(
            prefix="source", session_count=12, site_id="site-a"
        )

        report = nested_grouped_multimodal_evaluation(
            manifest,
            features,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            feature_sample_ids=[record["sample_id"] for record in manifest["records"]],
            outer_splits=3,
            inner_splits=2,
            c_grid=(0.1, 1.0),
            bootstrap_replicates=120,
            permutation_replicates=399,
            min_claim_groups=10,
            seed=41,
        )

        self.assertTrue(report["session_disjoint_accuracy_established"])
        self.assertTrue(report["multimodal_gain_established"])
        self.assertTrue(report["hyperparameters_selected_without_outer_test_labels"])
        self.assertTrue(report["best_unimodal_selected_without_outer_test_labels"])
        self.assertEqual(report["metrics"]["fusion"]["macro_f1"], 1.0)
        self.assertEqual(
            report["cluster_bootstrap_95_intervals"]["fusion"]["accuracy_95_ci"],
            [1.0, 1.0],
        )
        self.assertEqual(
            report["cluster_bootstrap_95_intervals"]["fusion"]["macro_f1_95_ci"],
            [1.0, 1.0],
        )
        self.assertGreater(
            report["fusion_vs_best_unimodal_nested"]["macro_f1"]["paired_group_bootstrap_95_ci"][0],
            0,
        )
        self.assertTrue(all(fold["group_overlap"] == [] for fold in report["folds"]))
        self.assertTrue(all(fold["site_overlap"] == ["site-a"] for fold in report["folds"]))

    def test_too_few_groups_reports_metrics_without_claim(self) -> None:
        manifest, features, _ = _strict_fixture(
            prefix="small", session_count=6, site_id="site-a"
        )

        report = nested_grouped_multimodal_evaluation(
            manifest,
            features,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            outer_splits=3,
            inner_splits=2,
            c_grid=(1.0,),
            bootstrap_replicates=30,
            permutation_replicates=39,
            min_claim_groups=10,
        )

        self.assertFalse(report["session_disjoint_accuracy_established"])
        self.assertFalse(report["multimodal_gain_established"])

    def test_session_split_audits_participant_overlap_without_teacher_ids(self) -> None:
        manifest, features, _ = _strict_fixture(
            prefix="student", session_count=12, site_id="site-a"
        )
        for session_index in range(12):
            session_id = f"student-session-{session_index:02d}"
            for record in manifest["records"]:
                if record["session_id"] == session_id:
                    record["participant_id"] = f"participant-{session_index:02d}"
                    record["cohort_id"] = f"cohort-{session_index % 3}"
                    record.pop("teacher_id")
        manifest["audit"]["identity_metadata_verified"] = {
            "session_id": True,
            "participant_id": True,
            "cohort_id": True,
            "site_id": True,
            "teacher_id": False,
        }
        manifest["audit"]["dataset_fingerprint"] = strict_dataset_fingerprint(
            manifest["records"]
        )

        report = nested_grouped_multimodal_evaluation(
            manifest,
            features,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            feature_sample_ids=[record["sample_id"] for record in manifest["records"]],
            group_field="session_id",
            statistical_group_field="session_id",
            required_identity_fields=(
                "session_id",
                "participant_id",
                "cohort_id",
                "site_id",
            ),
            outer_splits=3,
            inner_splits=2,
            c_grid=(1.0,),
            bootstrap_replicates=30,
            permutation_replicates=39,
            min_claim_groups=10,
        )

        self.assertTrue(report["session_disjoint_accuracy_established"])
        self.assertTrue(all(not fold["participant_overlap"] for fold in report["folds"]))

    def test_bad_model_and_unverified_feature_order_cannot_establish_accuracy(self) -> None:
        manifest, features, _ = _strict_fixture(
            prefix="bad", session_count=12, site_id="site-a"
        )
        zero_features = {
            name: np.zeros_like(matrix) for name, matrix in features.items()
        }

        report = nested_grouped_multimodal_evaluation(
            manifest,
            zero_features,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            outer_splits=3,
            inner_splits=2,
            c_grid=(1.0,),
            bootstrap_replicates=30,
            permutation_replicates=39,
            min_claim_groups=10,
        )

        self.assertLess(report["metrics"]["fusion"]["accuracy"], 0.75)
        self.assertFalse(report["feature_sample_order_verified"])
        self.assertFalse(report["accuracy_claim_gates"]["feature_sample_order"]["passed"])
        self.assertFalse(report["cross_group_accuracy_established"])
        self.assertFalse(report["session_disjoint_accuracy_established"])
        self.assertFalse(report["multimodal_gain_established"])


@unittest.skipUnless(RECOGNITION_DEPS_AVAILABLE, "recognition dependencies are optional")
class FrozenExternalEvaluationTests(unittest.TestCase):
    def test_duplicate_or_mismatched_class_names_fail_closed(self) -> None:
        manifest, features, feature_names = _strict_fixture(
            prefix="source", session_count=6, site_id="site-a"
        )
        with self.assertRaisesRegex(CredibilityError, "class_names"):
            fit_frozen_deployment_model(
                manifest,
                features,
                feature_names,
                feature_provenance=_feature_provenance(),
                class_names=["low", "low", "high"],
                inner_splits=2,
                c_grid=(1.0,),
            )
        manifest["records"][0]["label_name"] = "wrong"
        manifest["audit"]["dataset_fingerprint"] = strict_dataset_fingerprint(
            manifest["records"]
        )
        with self.assertRaisesRegex(CredibilityError, "label_name"):
            fit_frozen_deployment_model(
                manifest,
                features,
                feature_names,
                feature_provenance=_feature_provenance(),
                class_names=["low", "medium", "high"],
                inner_splits=2,
                c_grid=(1.0,),
            )

    def test_external_evaluation_requires_exact_schema_and_zero_identity_overlap(self) -> None:
        source_manifest, source_features, feature_names = _strict_fixture(
            prefix="source", session_count=12, site_id="site-a"
        )
        external_manifest, external_features, external_names = _strict_fixture(
            prefix="external", session_count=10, site_id="site-b", external=True
        )
        model = fit_frozen_deployment_model(
            source_manifest,
            source_features,
            feature_names,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            modality="fusion",
            inner_splits=3,
            c_grid=(0.1, 1.0),
        )
        artifact = frozen_model_to_artifact(model)
        self.assertEqual(
            artifact["training_feature_bundle_fingerprint"],
            strict_feature_bundle_fingerprint(
                source_manifest["records"],
                feature_names,
                source_features,
                _feature_provenance(),
            ),
        )
        self.assertEqual(
            artifact["training_feature_bundle_algorithm"],
            "strict_feature_bundle_v1",
        )
        model = load_frozen_deployment_model(json.loads(json.dumps(artifact)))
        self.assertEqual(frozen_model_to_artifact(model), artifact)

        with tempfile.TemporaryDirectory() as directory:
            registration_kwargs = _registered_evaluation_kwargs(
                model,
                external_manifest,
                external_features,
                external_names,
                Path(directory) / "ledger",
                registration_id="external-test-001",
            )
            report = evaluate_frozen_external_deployment(
                model,
                external_manifest,
                external_features,
                external_names,
                feature_provenance=_feature_provenance(),
                bootstrap_replicates=80,
                min_claim_sessions=10,
                **registration_kwargs,
            )
            with self.assertRaisesRegex(ValueError, "already consumed"):
                evaluate_frozen_external_deployment(
                    model,
                    external_manifest,
                    external_features,
                    external_names,
                    feature_provenance=_feature_provenance(),
                    bootstrap_replicates=10,
                    **registration_kwargs,
                )

        self.assertTrue(report["external_site_frozen_accuracy_established"])
        self.assertTrue(report["deployment_accuracy_established"])
        self.assertTrue(report["all_frozen_claim_gates_passed"])
        self.assertFalse(report["model_or_hyperparameter_fit_on_external_data"])
        self.assertEqual(report["metrics"]["accuracy"], 1.0)
        self.assertTrue(all(not values for values in report["training_external_overlap"].values()))

        mismatched_names = {**external_names, "audio": ["wrong", "order"]}
        with self.assertRaisesRegex(CredibilityError, "feature schema mismatch"):
            evaluate_frozen_external_deployment(
                model,
                external_manifest,
                external_features,
                mismatched_names,
                feature_provenance=_feature_provenance(),
                bootstrap_replicates=10,
            )

        tampered_contract = json.loads(json.dumps(artifact))
        tampered_contract["claim_contract"]["minimum_accuracy"] = 0.0
        with self.assertRaisesRegex(CredibilityError, "fingerprint mismatch"):
            load_frozen_deployment_model(tampered_contract)

        artifact["selected_C"] = 999.0
        with self.assertRaisesRegex(CredibilityError, "fingerprint mismatch"):
            load_frozen_deployment_model(artifact)

    def test_checkpoint_v3_schema_requires_all_integrity_bindings(self) -> None:
        manifest, features, feature_names = _strict_fixture(
            prefix="source", session_count=6, site_id="site-a"
        )
        model = fit_frozen_deployment_model(
            manifest,
            features,
            feature_names,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            inner_splits=2,
            c_grid=(1.0,),
        )
        artifact = frozen_model_to_artifact(model)
        validator = jsonschema.Draft202012Validator(
            read_json(project_root() / "schema/frozen_recognition_model.schema.json")
        )
        validator.validate(artifact)

        for required_field in (
            "claim_cluster_field",
            "claim_contract_fingerprint",
            "training_feature_bundle_fingerprint",
            "training_feature_bundle_algorithm",
            "model_fingerprint",
        ):
            with self.subTest(required_field=required_field):
                incomplete = json.loads(json.dumps(artifact))
                incomplete.pop(required_field)
                with self.assertRaises(jsonschema.ValidationError):
                    validator.validate(incomplete)

    def test_signed_coverage_binding_rejects_denominator_tampering(self) -> None:
        source_manifest, source_features, feature_names = _strict_fixture(
            prefix="source", session_count=12, site_id="site-a"
        )
        original_manifest, external_features, external_names = _strict_fixture(
            prefix="external", session_count=10, site_id="site-b", external=True
        )
        model = fit_frozen_deployment_model(
            source_manifest,
            source_features,
            feature_names,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            inner_splits=3,
            c_grid=(1.0,),
        )
        mutations = {
            "eligible denominator": lambda evidence, audit: (
                evidence.__setitem__(
                    "eligible_sample_count", evidence["eligible_sample_count"] + 5
                ),
                evidence.__setitem__("excluded_sample_count", 5),
                evidence.__setitem__("exclusion_reason_counts", {"not_available": 5}),
                audit.__setitem__(
                    "evaluation_coverage_fraction",
                    evidence["evaluated_sample_count"]
                    / evidence["eligible_sample_count"],
                ),
            ),
            "population fingerprint": lambda evidence, audit: evidence.__setitem__(
                "eligible_population_fingerprint", "0" * 64
            ),
            "denominator source": lambda evidence, audit: evidence.__setitem__(
                "denominator_source", "changed after custodian registration"
            ),
        }
        for index, (name, mutate) in enumerate(mutations.items()):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                external_manifest = json.loads(json.dumps(original_manifest))
                registration_kwargs = _registered_evaluation_kwargs(
                    model,
                    external_manifest,
                    external_features,
                    external_names,
                    Path(directory) / "ledger",
                    registration_id=f"coverage-binding-{index:02d}",
                )
                audit = external_manifest["audit"]
                evidence = audit["evaluation_coverage"]
                mutate(evidence, audit)
                evidence["coverage_evidence_fingerprint"] = (
                    strict_coverage_evidence_fingerprint(evidence)
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "binding mismatch: external_coverage_evidence_fingerprint",
                ):
                    evaluate_frozen_external_deployment(
                        model,
                        external_manifest,
                        external_features,
                        external_names,
                        feature_provenance=_feature_provenance(),
                        bootstrap_replicates=10,
                        **registration_kwargs,
                    )

    def test_tampered_registration_signature_fails_before_consumption(self) -> None:
        source_manifest, source_features, feature_names = _strict_fixture(
            prefix="source", session_count=12, site_id="site-a"
        )
        external_manifest, external_features, external_names = _strict_fixture(
            prefix="external", session_count=10, site_id="site-b", external=True
        )
        model = fit_frozen_deployment_model(
            source_manifest,
            source_features,
            feature_names,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            inner_splits=3,
            c_grid=(1.0,),
        )
        with tempfile.TemporaryDirectory() as directory:
            kwargs = _registered_evaluation_kwargs(
                model,
                external_manifest,
                external_features,
                external_names,
                Path(directory) / "ledger",
                registration_id="tamper-signature-001",
            )
            tampered = json.loads(json.dumps(kwargs["registration_attestation"]))
            tampered["request"]["external_dataset_fingerprint"] = "0" * 64
            unsigned_request = {
                key: value
                for key, value in tampered["request"].items()
                if key != "request_fingerprint"
            }
            tampered["request"]["request_fingerprint"] = hashlib.sha256(
                json.dumps(
                    unsigned_request,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            kwargs["registration_attestation"] = tampered

            with self.assertRaisesRegex(ValueError, "signature verification failed"):
                evaluate_frozen_external_deployment(
                    model,
                    external_manifest,
                    external_features,
                    external_names,
                    feature_provenance=_feature_provenance(),
                    bootstrap_replicates=10,
                    **kwargs,
                )
            self.assertFalse(any((Path(directory) / "ledger").glob("*.json")))

    def test_cross_session_person_dependency_blocks_deployment_claim(self) -> None:
        source_manifest, source_features, feature_names = _strict_fixture(
            prefix="source", session_count=12, site_id="site-a"
        )
        external_manifest, external_features, external_names = _strict_fixture(
            prefix="external", session_count=10, site_id="site-b", external=True
        )
        for manifest, prefix in (
            (source_manifest, "source"),
            (external_manifest, "external"),
        ):
            for record in manifest["records"]:
                session_suffix = record["session_id"].rsplit("-", 1)[-1]
                record["participant_id"] = (
                    f"{prefix}-participant-shared"
                    if prefix == "external"
                    else f"{prefix}-participant-{session_suffix}"
                )
            manifest["audit"]["identity_metadata_verified"]["participant_id"] = True
            manifest["audit"]["dataset_fingerprint"] = strict_dataset_fingerprint(
                manifest["records"]
            )
        identity_fields = ("session_id", "participant_id", "teacher_id", "site_id")
        model = fit_frozen_deployment_model(
            source_manifest,
            source_features,
            feature_names,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            required_identity_fields=identity_fields,
            claim_cluster_field="session_id",
            inner_splits=3,
            c_grid=(1.0,),
        )
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_frozen_external_deployment(
                model,
                external_manifest,
                external_features,
                external_names,
                feature_provenance=_feature_provenance(),
                bootstrap_replicates=80,
                **_registered_evaluation_kwargs(
                    model,
                    external_manifest,
                    external_features,
                    external_names,
                    Path(directory) / "ledger",
                    registration_id="cross-cluster-001",
                ),
            )

        self.assertEqual(report["metrics"]["accuracy"], 1.0)
        self.assertFalse(report["independent_cluster_structure_verified"])
        self.assertIn("external-participant-shared", report["cross_cluster_person_dependencies"]["participant_id"])
        self.assertFalse(
            report["claim_gate_results"]["independent_cluster_structure"]["passed"]
        )
        self.assertFalse(report["deployment_accuracy_established"])

    def test_overlap_with_training_site_is_rejected(self) -> None:
        source_manifest, source_features, feature_names = _strict_fixture(
            prefix="source", session_count=6, site_id="same-site"
        )
        external_manifest, external_features, external_names = _strict_fixture(
            prefix="external", session_count=6, site_id="same-site", external=True
        )
        model = fit_frozen_deployment_model(
            source_manifest,
            source_features,
            feature_names,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            inner_splits=2,
            c_grid=(1.0,),
        )

        with self.assertRaisesRegex(CredibilityError, "overlaps training"):
            evaluate_frozen_external_deployment(
                model,
                external_manifest,
                external_features,
                external_names,
                feature_provenance=_feature_provenance(),
                bootstrap_replicates=10,
            )

    def test_poor_external_performance_cannot_establish_accuracy(self) -> None:
        source_manifest, source_features, feature_names = _strict_fixture(
            prefix="source", session_count=12, site_id="site-a"
        )
        external_manifest, external_features, external_names = _strict_fixture(
            prefix="external", session_count=10, site_id="site-b", external=True
        )
        model = fit_frozen_deployment_model(
            source_manifest,
            source_features,
            feature_names,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            modality="fusion",
            inner_splits=3,
            c_grid=(0.1, 1.0),
        )
        poor_features = {
            name: np.zeros_like(matrix) for name, matrix in external_features.items()
        }

        report = evaluate_frozen_external_deployment(
            model,
            external_manifest,
            poor_features,
            external_names,
            feature_provenance=_feature_provenance(),
            bootstrap_replicates=80,
        )

        self.assertFalse(report["all_frozen_claim_gates_passed"])
        self.assertFalse(report["external_site_frozen_accuracy_established"])
        self.assertFalse(report["deployment_accuracy_established"])
        self.assertFalse(report["claim_gate_results"]["accuracy"]["passed"])
        self.assertFalse(
            report["claim_gate_results"]["minimum_per_class_recall"]["passed"]
        )

    def test_missing_one_time_lockbox_attestation_cannot_establish_accuracy(self) -> None:
        source_manifest, source_features, feature_names = _strict_fixture(
            prefix="source", session_count=12, site_id="site-a"
        )
        external_manifest, external_features, external_names = _strict_fixture(
            prefix="external", session_count=10, site_id="site-b", external=True
        )
        external_manifest["audit"]["one_time_lockbox_evaluation"] = False
        model = fit_frozen_deployment_model(
            source_manifest,
            source_features,
            feature_names,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            modality="fusion",
            inner_splits=3,
            c_grid=(0.1, 1.0),
        )

        report = evaluate_frozen_external_deployment(
            model,
            external_manifest,
            external_features,
            external_names,
            feature_provenance=_feature_provenance(),
            bootstrap_replicates=80,
        )

        self.assertFalse(report["claim_gate_results"]["one_time_lockbox"]["passed"])
        self.assertFalse(report["external_site_frozen_accuracy_established"])
        self.assertFalse(report["deployment_accuracy_established"])

    def test_student_identity_contract_freezes_activity_without_teacher_id(self) -> None:
        source_manifest, source_features, feature_names = _strict_fixture(
            prefix="source", session_count=12, site_id="site-a"
        )
        external_manifest, external_features, external_names = _strict_fixture(
            prefix="external", session_count=10, site_id="site-b", external=True
        )
        identity_fields = (
            "session_id",
            "participant_id",
            "cohort_id",
            "activity_id",
            "site_id",
        )
        for prefix, manifest in (
            ("source", source_manifest),
            ("external", external_manifest),
        ):
            for record in manifest["records"]:
                session_suffix = record["session_id"].rsplit("-", 1)[-1]
                record["participant_id"] = f"{prefix}-participant-{session_suffix}"
                record["cohort_id"] = f"{prefix}-cohort"
                record["activity_id"] = f"{prefix}-activity-{session_suffix}"
                record.pop("teacher_id")
            manifest["audit"]["identity_metadata_verified"] = {
                "session_id": True,
                "participant_id": True,
                "cohort_id": True,
                "activity_id": True,
                "site_id": True,
                "teacher_id": False,
            }
            manifest["audit"]["dataset_fingerprint"] = strict_dataset_fingerprint(
                manifest["records"]
            )

        model = fit_frozen_deployment_model(
            source_manifest,
            source_features,
            feature_names,
            feature_provenance=_feature_provenance(),
            class_names=["low", "medium", "high"],
            modality="fusion",
            required_identity_fields=identity_fields,
            inner_splits=3,
            c_grid=(0.1, 1.0),
            claim_contract=ClaimContract(minimum_per_class_support=10),
        )
        artifact = frozen_model_to_artifact(model)
        loaded = load_frozen_deployment_model(json.loads(json.dumps(artifact)))

        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_frozen_external_deployment(
                loaded,
                external_manifest,
                external_features,
                external_names,
                feature_provenance=_feature_provenance(),
                bootstrap_replicates=80,
                **_registered_evaluation_kwargs(
                    loaded,
                    external_manifest,
                    external_features,
                    external_names,
                    Path(directory) / "ledger",
                    registration_id="student-test-001",
                ),
            )

        self.assertEqual(loaded.required_identity_fields, identity_fields)
        self.assertNotIn("teacher_id", loaded.training_identity_values_by_field)
        self.assertIn("activity_id", report["training_external_overlap"])
        self.assertTrue(report["deployment_accuracy_established"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
