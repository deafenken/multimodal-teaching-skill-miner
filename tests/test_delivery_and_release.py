from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

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

from teaching_skill_miner.cli import _load_strict_feature_bundle, main
from teaching_skill_miner.delivery import verify_delivery
from teaching_skill_miner.io_utils import (
    ensure_private_directory,
    project_root,
    read_json,
    write_json,
)
from teaching_skill_miner.miner import mine_skill
from teaching_skill_miner.project_health import doctor_report
from teaching_skill_miner.recognition.strict_evaluation import (
    strict_dataset_fingerprint,
    strict_coverage_evidence_fingerprint,
    strict_feature_bundle_fingerprint,
)
from teaching_skill_miner.recognition.dipser_experiment import feature_bundle_fingerprint
from teaching_skill_miner.release_audit import audit_release_path
from scripts import audit_repository_privacy


class DeliveryTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "POSIX permission semantics required")
    def test_sensitive_artifact_helpers_enforce_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private = ensure_private_directory(Path(directory) / "classroom-output")
            artifact = write_json(private / "result.json", {"private": True})
            self.assertEqual(private.stat().st_mode & 0o777, 0o700)
            self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)

    def test_delivery_is_engineering_ready_but_external_evidence_pending(self) -> None:
        root = project_root()
        report = verify_delivery(
            root / "data/dataset_manifest.json",
            root / "data/evaluation_cases.json",
            human_review_path=root / "artifacts/human_review.csv",
            dipser_report_path=(
                root
                / "artifacts/dipser_credible/full_v5_52_complete_v5/hierarchical_0_9_report.json"
            ),
        )
        self.assertTrue(report["engineering_delivery_ready"])
        self.assertFalse(report["research_validation_complete"])
        self.assertEqual(
            report["overall_status"], "engineering_ready_external_validation_pending"
        )
        self.assertIn("bundled deterministic", report["engineering_delivery_scope"])
        self.assertFalse(
            report["claim_policy"][
                "engineering_delivery_ready_means_exact_release_verified"
            ]
        )
        self.assertEqual(report["human_validation"]["validation_status"], "incomplete")
        self.assertEqual(report["summary_metrics"]["human_reviewed_skill_count"], 0)
        self.assertTrue(
            all(row["procedure_evidence_audit"] for row in report["transcripts_and_skills"])
        )

    def test_demo_does_not_overwrite_existing_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["demo", "--output", str(output)]), 0)
            summary = read_json(output / "summary.json")
            self.assertEqual(
                summary["research_validation_pending"],
                [
                    "formal_full_transcripts_or_audited_asr",
                    "independent_human_review",
                    "confirmatory_external_multimodal_gain",
                    "cryptographically_registered_prospective_deployment_evaluation",
                    "real_learner_effectiveness_study",
                ],
            )
            review = output / "human_review.csv"
            original = review.read_text(encoding="utf-8-sig")
            self.assertIn("skill_fingerprint", original.splitlines()[0])
            review.write_text(original + "# human work must survive\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["demo", "--output", str(output)]), 0)
            self.assertTrue(review.read_text(encoding="utf-8").endswith("# human work must survive\n"))

    def test_doctor_never_exposes_api_secret(self) -> None:
        original = os.environ.get("TSM_API_KEY")
        os.environ["TSM_API_KEY"] = "super-secret-value-that-must-not-appear"
        try:
            report = doctor_report()
        finally:
            if original is None:
                os.environ.pop("TSM_API_KEY", None)
            else:
                os.environ["TSM_API_KEY"] = original
        serialized = json.dumps(report)
        self.assertNotIn("super-secret-value", serialized)
        self.assertTrue(report["api_backend"]["configured"])
        self.assertFalse(report["api_backend"]["key_value_exposed"])
        self.assertIn("cryptography", report["packages"])
        self.assertIn("offline_core", report["status_scope"])
        self.assertTrue(
            any("Top-level status=ready" in value for value in report["limitations"])
        )
        self.assertEqual(
            report["capabilities"]["signed_external_evaluation_ready"],
            report["capabilities"]["recognition_experiments_ready"]
            and report["packages"]["cryptography"]["available"]
            and report["workspace"]["writable"],
        )


class ReleaseAuditTests(unittest.TestCase):
    def test_safe_release_directory_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "safe.json").write_text('{"aggregate_accuracy": 0.8}\n', encoding="utf-8")
            report = audit_release_path(root)
            self.assertTrue(report["passed"], report["findings"])

    def test_media_secret_and_identity_payload_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_secret = "api_" + "key=abcdefghijklmnopqrstuvwxyz012345"
            (root / "classroom.mp4").write_bytes(b"not-real-media")
            (root / "secret.txt").write_text(
                fake_secret, encoding="utf-8"
            )
            (root / "rows.json").write_text(
                '{"participant_id": "p-1", "score": 0.9}', encoding="utf-8"
            )
            (root / "reviews.csv").write_text(
                "skill_id,reviewer_id,score\ns1,reviewer-1,5\n",
                encoding="utf-8",
            )
            report = audit_release_path(root)
            rules = {finding["rule"] for finding in report["findings"]}
            self.assertFalse(report["passed"])
            self.assertIn("forbidden_binary_or_archive", rules)
            self.assertIn("possible_secret", rules)
            self.assertIn("row_level_identity_field", rules)
            identity_details = {
                finding["detail"]
                for finding in report["findings"]
                if finding["rule"] == "row_level_identity_field"
            }
            self.assertIn("reviewer_id", identity_details)

    def test_face_frame_and_disguised_png_signature_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "face.jpg").write_bytes(b"\xff\xd8\xffnot-a-real-photo")
            (root / "renamed.bin").write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-image")
            report = audit_release_path(root)
            rules = {finding["rule"] for finding in report["findings"]}
            self.assertFalse(report["passed"])
            self.assertIn("forbidden_binary_or_archive", rules)
            self.assertIn("forbidden_binary_signature", rules)

    def test_renamed_mp4_and_example_named_identity_payload_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "renamed.bin").write_bytes(
                b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
            )
            (root / "real_results_example.json").write_text(
                '{"participant_id": "p-1", "score": 0.9}',
                encoding="utf-8",
            )
            report = audit_release_path(root)
            self.assertFalse(report["passed"])
            self.assertIn(
                "iso-base-media",
                {
                    finding["detail"]
                    for finding in report["findings"]
                    if finding["rule"] == "forbidden_binary_signature"
                },
            )
            self.assertIn(
                "participant_id",
                {
                    finding["detail"]
                    for finding in report["findings"]
                    if finding["rule"] == "row_level_identity_field"
                },
            )

    def test_schema_named_row_data_and_common_disguised_binaries_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "schema").mkdir()
            (root / "schema/private_rows.json").write_text(
                '{"participant_id": "p-1"}', encoding="utf-8"
            )
            (root / "archive.bin").write_bytes(b"\x1f\x8b\x08\x00")
            (root / "document.bin").write_bytes(b"%PDF-1.7\n")
            (root / "model.bin").write_bytes(b"\x80\x04N.")
            report = audit_release_path(root)
            details = {finding["detail"] for finding in report["findings"]}
            self.assertFalse(report["passed"])
            self.assertIn("participant_id", details)
            self.assertIn("gzip", details)
            self.assertIn("pdf", details)
            self.assertIn("pickle", details)

    def test_archive_traversal_and_duplicate_members_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../../outside.txt", "unsafe")
                archive.writestr("safe.txt", "first")
                with self.assertWarns(UserWarning):
                    archive.writestr("safe.txt", "second")
            report = audit_release_path(archive_path)
            rules = {finding["rule"] for finding in report["findings"]}
            self.assertFalse(report["passed"])
            self.assertIn("unsafe_member_path", rules)
            self.assertIn("duplicate_member_path", rules)

    def test_archive_symlink_and_zip_bomb_metadata_fail_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            symlink_archive = root / "symlink.zip"
            with zipfile.ZipFile(symlink_archive, "w") as archive:
                link = zipfile.ZipInfo("linked.txt")
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(link, "private-target.txt")
            symlink_report = audit_release_path(symlink_archive)
            self.assertFalse(symlink_report["passed"])
            self.assertIn(
                "symlink_member",
                {finding["rule"] for finding in symlink_report["findings"]},
            )

            bomb_archive = root / "bomb.zip"
            with zipfile.ZipFile(
                bomb_archive, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr("zeros.txt", b"0" * 2_000_000)
            bomb_report = audit_release_path(bomb_archive)
            self.assertFalse(bomb_report["passed"])
            self.assertEqual(bomb_report["member_count"], 0)
            self.assertIn(
                "archive_resource_limit",
                {finding["rule"] for finding in bomb_report["findings"]},
            )

    def test_tsv_columnar_json_logs_env_and_renamed_tar_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_secret = "api_" + "key=abcdefghijklmnopqrstuvwxyz012345"
            fake_local_path = "/" + "Users/private/classroom.json"
            (root / "participants.tsv").write_text(
                "participant_id\tscore\np-1\t0.9\n", encoding="utf-8"
            )
            (root / "columnar.json").write_text(
                '{"columns": ["participant_id", "score"], "rows": [["p-1", 0.9]]}',
                encoding="utf-8",
            )
            (root / "application.log").write_text(
                f"{fake_secret} {fake_local_path}\n",
                encoding="utf-8",
            )
            (root / ".env.production").write_text(
                f"{fake_secret}\n", encoding="utf-8"
            )
            tar_payload = bytearray(512)
            tar_payload[257:262] = b"ustar"
            (root / "archive.bin").write_bytes(tar_payload)
            report = audit_release_path(root)
            rules = {finding["rule"] for finding in report["findings"]}
            details = {finding["detail"] for finding in report["findings"]}
            self.assertFalse(report["passed"])
            self.assertIn("participant_id", details)
            self.assertIn("tar", details)
            self.assertIn("possible_secret", rules)
            self.assertIn("absolute_local_path", rules)
            self.assertIn("forbidden_private_path", rules)

    def test_json_string_tokens_and_bomless_utf16_fail_but_schema_definitions_pass(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_secret = "api_" + "key=abcdefghijklmnopqrstuvwxyz012345"
            fake_local_path = "/" + "Users/private/classroom.json"
            (root / "string-tokens.json").write_text(
                json.dumps(
                    {
                        "schema": ["participant_id", "score"],
                        "names": ["session_id"],
                        "variables": ["teacher_id"],
                    }
                ),
                encoding="utf-8",
            )
            (root / "utf16-rows.json").write_bytes(
                '{"columns":["reviewer_id"],"rows":[["r-1"]]}'.encode(
                    "utf-16-le"
                )
            )
            (root / "utf16-secret.txt").write_bytes(
                f"{fake_secret} {fake_local_path}".encode("utf-16-be")
            )
            report = audit_release_path(root)
            rules = {finding["rule"] for finding in report["findings"]}
            details = {finding["detail"] for finding in report["findings"]}
            self.assertFalse(report["passed"])
            self.assertTrue(
                {"participant_id", "session_id", "teacher_id", "reviewer_id"}
                <= details
            )
            self.assertIn("possible_secret", rules)
            self.assertIn("absolute_local_path", rules)

        with tempfile.TemporaryDirectory() as directory:
            schema_root = Path(directory)
            (schema_root / "safe.schema.json").write_text(
                json.dumps(
                    {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "participant_id": {"type": "string"},
                            "session_id": {"type": "string"},
                        },
                        "required": ["participant_id", "session_id"],
                        "dependentRequired": {
                            "participant_id": ["session_id"]
                        },
                        "$defs": {
                            "identity_field": {
                                "enum": ["participant_id", "session_id"]
                            }
                        },
                        "allOf": [
                            {
                                "properties": {
                                    "primary_identity": {
                                        "const": "participant_id"
                                    }
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = audit_release_path(schema_root)
            self.assertTrue(report["passed"], report["findings"])

    def test_unknown_wrapped_binary_payloads_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wrapped-model.bin").write_bytes(b"JUNK\x80\x04N.")
            (root / "weights.bin").write_bytes(
                (8).to_bytes(8, "little") + b'{"x":1}' + b"\x00" * 64
            )
            report = audit_release_path(root)
            self.assertFalse(report["passed"])
            self.assertEqual(
                {
                    finding["path"]
                    for finding in report["findings"]
                    if finding["rule"] == "unknown_binary_payload"
                },
                {"weights.bin", "wrapped-model.bin"},
            )


class RepositoryPrivacyAuditTests(unittest.TestCase):
    def test_tracked_disguised_media_and_example_identity_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "renamed.bin").write_bytes(
                b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
            )
            (root / "real_results_example.json").write_text(
                '{"participant_id": "p-1", "score": 0.9}',
                encoding="utf-8",
            )
            with (
                patch.object(audit_repository_privacy, "ROOT", root),
                patch.object(
                    audit_repository_privacy,
                    "tracked_files",
                    return_value=["real_results_example.json", "renamed.bin"],
                ),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(audit_repository_privacy.main(), 2)

    def test_non_git_checkout_fails_cleanly(self) -> None:
        with (
            patch.object(
                audit_repository_privacy,
                "tracked_files",
                side_effect=RuntimeError("repository privacy audit requires a Git checkout"),
            ),
            redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(audit_repository_privacy.main(), 2)
        self.assertIn("unavailable", stderr.getvalue())

    def test_oversized_tracked_file_requires_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "large.txt").write_bytes(b"x" * 2_000_001)
            with (
                patch.object(audit_repository_privacy, "ROOT", root),
                patch.object(
                    audit_repository_privacy,
                    "tracked_files",
                    return_value=["large.txt"],
                ),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(audit_repository_privacy.main(), 2)

    def test_tracked_utf16_secret_and_unknown_binary_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_secret = "api_" + "key=abcdefghijklmnopqrstuvwxyz012345"
            (root / "secret.txt").write_bytes(
                fake_secret.encode("utf-16-le")
            )
            (root / "wrapped.bin").write_bytes(b"JUNK\x80\x04N.")
            with (
                patch.object(audit_repository_privacy, "ROOT", root),
                patch.object(
                    audit_repository_privacy,
                    "tracked_files",
                    return_value=["secret.txt", "wrapped.bin"],
                ),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(audit_repository_privacy.main(), 2)


class SchemaTests(unittest.TestCase):
    def test_all_schemas_are_valid_and_generated_skills_conform(self) -> None:
        root = project_root()
        schemas = {}
        for path in sorted((root / "schema").glob("*.json")):
            schema = read_json(path)
            jsonschema.Draft202012Validator.check_schema(schema)
            schemas[path.name] = schema
        manifest = read_json(root / "data/dataset_manifest.json")
        validator = jsonschema.Draft202012Validator(
            schemas["teaching_skill.schema.json"],
            format_checker=jsonschema.FormatChecker(),
        )
        for item in manifest["videos"]:
            skill = mine_skill(read_json(root / item["transcript_path"]))
            validator.validate(skill)
        jsonschema.Draft202012Validator(
            schemas["external_claim_contract.schema.json"]
        ).validate(read_json(root / "configs/external_claim_contract.example.json"))
        jsonschema.Draft202012Validator(
            schemas["formal_caption_sources.schema.json"],
            format_checker=jsonschema.FormatChecker(),
        ).validate(read_json(root / "data/formal_caption_sources.json"))


@unittest.skipUnless(RECOGNITION_DEPS_AVAILABLE, "recognition dependencies are optional")
class FrozenRecognitionCliTests(unittest.TestCase):
    @staticmethod
    def _fixture(prefix: str, sessions: int, site: str, external: bool) -> tuple[dict, dict]:
        records = []
        visual = []
        audio = []
        for session in range(sessions):
            for label in range(3):
                sample_id = f"{prefix}-s{session}-y{label}"
                records.append(
                    {
                        "sample_id": sample_id,
                        "content_sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
                        "label": label,
                        "label_name": ("low", "medium", "high")[label],
                        "session_id": f"{prefix}-session-{session}",
                        "teacher_id": f"{prefix}-teacher-{session}",
                        "site_id": site,
                        "modalities": {"visual": True, "audio": True},
                    }
                )
                visual.append([-2.0 if label == 0 else 0.0, session / 1000])
                audio.append([2.0 if label == 2 else 0.0, -session / 1000])
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
            "identity_metadata_source": "test fixture",
            "ground_truth_source": "test fixture",
            "held_out_from_model_development": external,
            "deployment_target_documented": external,
            "prospective_deployment_collection": external,
            "one_time_lockbox_evaluation": external,
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
                "denominator_source": "frozen external CLI test roster",
                "exclusion_reason_counts": {},
            }
            coverage["coverage_evidence_fingerprint"] = (
                strict_coverage_evidence_fingerprint(coverage)
            )
            audit["evaluation_coverage"] = coverage
        provenance = {
            name: {
                "extractor_id": f"fixed-{name}",
                "extractor_fingerprint": hashlib.sha256(f"fixed-{name}".encode()).hexdigest(),
                "frozen_before_evaluation": True,
                "uses_ground_truth_labels": False,
                "fitted_on_evaluation_records": False,
            }
            for name in ("visual", "audio")
        }
        feature_bundle = {
            "schema_version": "1.0",
            "dataset_fingerprint": audit["dataset_fingerprint"],
            "fingerprint_algorithm": "strict_feature_bundle_v1",
            "sample_ids": [record["sample_id"] for record in records],
            "feature_names_by_modality": {
                "visual": ["v_class0", "v_noise"],
                "audio": ["a_class2", "a_noise"],
            },
            "feature_provenance": provenance,
            "matrices": {"visual": visual, "audio": audio},
        }
        feature_bundle["feature_bundle_fingerprint"] = strict_feature_bundle_fingerprint(
            records,
            feature_bundle["feature_names_by_modality"],
            feature_bundle["matrices"],
            provenance,
        )
        return {"schema_version": "2.0", "audit": audit, "records": records}, feature_bundle

    def test_freeze_and_external_evaluate_cli_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest, source_features = self._fixture("source", 6, "site-a", False)
            external_manifest, external_features = self._fixture("external", 3, "site-b", True)
            contract = {
                "schema_version": "1.0",
                "primary_metric": "macro_f1",
                "minimum_primary_metric": 0.5,
                "minimum_accuracy": 0.5,
                "minimum_macro_f1": 0.5,
                "minimum_accuracy_ci_lower": 0.5,
                "minimum_macro_f1_ci_lower": 0.5,
                "minimum_per_class_recall": 0.5,
                "minimum_per_class_support": 1,
                "minimum_sessions": 3,
                "minimum_coverage_fraction": 1.0,
                "require_prospective": True,
                "require_one_time_lockbox": True,
            }
            paths = {}
            for name, value in (
                ("source_manifest", source_manifest),
                ("source_features", source_features),
                ("external_manifest", external_manifest),
                ("external_features", external_features),
                ("contract", contract),
            ):
                path = root / f"{name}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths[name] = path
            model_path = root / "model.json"
            with redirect_stdout(io.StringIO()):
                freeze_code = main(
                    [
                        "freeze-recognition-model",
                        "--manifest", str(paths["source_manifest"]),
                        "--features", str(paths["source_features"]),
                        "--claim-contract", str(paths["contract"]),
                        "--class-name", "low",
                        "--class-name", "medium",
                        "--class-name", "high",
                        "--inner-splits", "2",
                        "--c-grid", "0.1", "1.0",
                        "--output", str(model_path),
                    ]
                )
            self.assertEqual(freeze_code, 0)
            private_key = Ed25519PrivateKey.generate()
            private_key_path = root / "custodian-private.pem"
            public_key_path = root / "custodian-public.pem"
            private_key_path.write_bytes(
                private_key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
            public_key_path.write_bytes(
                private_key.public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            request_path = root / "freeze-registration-request.json"
            attestation_path = root / "freeze-registration-attestation.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "create-freeze-registration",
                            "--model", str(model_path),
                            "--manifest", str(paths["external_manifest"]),
                            "--features", str(paths["external_features"]),
                            "--registration-id", "cli-lockbox-001",
                            "--output", str(request_path),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "sign-freeze-registration",
                            "--request", str(request_path),
                            "--private-key", str(private_key_path),
                            "--issuer", "independent-cli-test-custodian",
                            "--key-id", "cli-key-2026",
                            "--output", str(attestation_path),
                        ]
                    ),
                    0,
                )
            report_path = root / "external_report.json"
            with redirect_stdout(io.StringIO()):
                evaluate_code = main(
                    [
                        "evaluate-frozen-recognition",
                        "--model", str(model_path),
                        "--manifest", str(paths["external_manifest"]),
                        "--features", str(paths["external_features"]),
                        "--bootstrap-replicates", "30",
                        "--registration-attestation", str(attestation_path),
                        "--trusted-public-key", str(public_key_path),
                        "--one-time-ledger", str(root / "one-time-ledger"),
                        "--output", str(report_path),
                    ]
                )
            self.assertEqual(evaluate_code, 0)
            report = read_json(report_path)
            self.assertTrue(report["all_frozen_claim_gates_passed"])
            self.assertTrue(report["deployment_accuracy_established"])
            signed_receipt_path = root / "signed-evaluation-receipt.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "sign-evaluation-report",
                            "--report", str(report_path),
                            "--private-key", str(private_key_path),
                            "--issuer", "independent-cli-test-custodian",
                            "--key-id", "cli-key-2026",
                            "--output", str(signed_receipt_path),
                        ]
                    ),
                    0,
                )
            delivery = verify_delivery(
                project_root() / "data/dataset_manifest.json",
                project_root() / "data/evaluation_cases.json",
                external_deployment_report_path=report_path,
                external_evaluation_receipt_path=signed_receipt_path,
                trusted_attestation_public_key_path=public_key_path,
            )
            self.assertTrue(
                delivery["external_evidence_checks"][
                    "prospective_deployment_accuracy"
                ]
            )
            self.assertTrue(delivery["external_deployment_validation"]["passed"])

            tampered_report = json.loads(json.dumps(report))
            tampered_report["metrics"]["accuracy"] = 0.0
            tampered_report_path = root / "tampered-external-report.json"
            write_json(tampered_report_path, tampered_report)
            tampered_delivery = verify_delivery(
                project_root() / "data/dataset_manifest.json",
                project_root() / "data/evaluation_cases.json",
                external_deployment_report_path=tampered_report_path,
                external_evaluation_receipt_path=signed_receipt_path,
                trusted_attestation_public_key_path=public_key_path,
            )
            self.assertFalse(tampered_delivery["external_deployment_validation"]["passed"])
            self.assertIn(
                "report fingerprint mismatch",
                tampered_delivery["external_deployment_validation"]["reason"],
            )

            tampered_receipt = read_json(signed_receipt_path)
            signature = tampered_receipt["signature_base64"]
            tampered_receipt["signature_base64"] = (
                ("A" if signature[0] != "A" else "B") + signature[1:]
            )
            tampered_receipt_path = root / "tampered-evaluation-receipt.json"
            write_json(tampered_receipt_path, tampered_receipt)
            tampered_delivery = verify_delivery(
                project_root() / "data/dataset_manifest.json",
                project_root() / "data/evaluation_cases.json",
                external_deployment_report_path=report_path,
                external_evaluation_receipt_path=tampered_receipt_path,
                trusted_attestation_public_key_path=public_key_path,
            )
            self.assertFalse(tampered_delivery["external_deployment_validation"]["passed"])
            self.assertIn(
                "receipt signature failed",
                tampered_delivery["external_deployment_validation"]["reason"],
            )

    def test_feature_bundle_tampering_fails_closed_before_training(self) -> None:
        mutations = {
            "matrix": lambda bundle: bundle["matrices"]["visual"][0].__setitem__(0, -1.75),
            "provenance": lambda bundle: bundle["feature_provenance"]["visual"].__setitem__(
                "extractor_id", "changed-after-freeze"
            ),
            "dataset fingerprint": lambda bundle: bundle.__setitem__(
                "dataset_fingerprint", "0" * 64
            ),
            "missing bundle fingerprint": lambda bundle: bundle.pop(
                "feature_bundle_fingerprint"
            ),
            "row order": lambda bundle: bundle["sample_ids"].reverse(),
            "boolean matrix value": lambda bundle: bundle["matrices"]["audio"][0].__setitem__(
                0, True
            ),
            "numeric feature name": lambda bundle: bundle[
                "feature_names_by_modality"
            ]["visual"].__setitem__(0, 1),
            "numeric sample id": lambda bundle: bundle["sample_ids"].__setitem__(0, 1),
            "overflowing feature": lambda bundle: bundle["matrices"]["audio"][0].__setitem__(
                0, 10 ** 4000
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                manifest, bundle = self._fixture("source", 6, "site-a", False)
                mutate(bundle)
                path = Path(directory) / "features.json"
                path.write_text(json.dumps(bundle), encoding="utf-8")
                with self.assertRaises(ValueError):
                    _load_strict_feature_bundle(path, manifest)

    def test_unmarked_legacy_dipser_fingerprint_remains_narrowly_compatible(self) -> None:
        manifest, bundle = self._fixture("dipser", 6, "site-a", False)
        manifest["audit"]["dataset_id"] = "DIPSER"
        manifest["audit"]["archive_catalog_fingerprint"] = hashlib.sha256(
            b"synthetic-dipser-catalog"
        ).hexdigest()
        for record in manifest["records"]:
            record["modalities"]["sensor"] = record["modalities"].pop("audio")
        manifest["audit"]["dataset_fingerprint"] = strict_dataset_fingerprint(
            manifest["records"]
        )
        bundle["dataset_fingerprint"] = manifest["audit"]["dataset_fingerprint"]
        bundle["feature_names_by_modality"]["sensor"] = bundle[
            "feature_names_by_modality"
        ].pop("audio")
        bundle["feature_provenance"]["sensor"] = bundle["feature_provenance"].pop("audio")
        bundle["matrices"]["sensor"] = bundle["matrices"].pop("audio")
        bundle.pop("fingerprint_algorithm")
        bundle["feature_bundle_fingerprint"] = feature_bundle_fingerprint(
            manifest["records"],
            bundle["feature_names_by_modality"],
            bundle["matrices"],
        )
        manifest["audit"]["feature_bundle_fingerprint"] = bundle[
            "feature_bundle_fingerprint"
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.json"
            path.write_text(json.dumps(bundle), encoding="utf-8")
            matrices, names, provenance, binding = _load_strict_feature_bundle(path, manifest)
            self.assertEqual(set(matrices), {"visual", "sensor"})
            self.assertEqual(set(names), set(provenance))
            self.assertTrue(binding["legacy_input_upgraded_to_strict_binding"])

            manifest["audit"]["dataset_id"] = "not-dipser"
            with self.assertRaisesRegex(ValueError, "fingerprint_algorithm"):
                _load_strict_feature_bundle(path, manifest)


if __name__ == "__main__":
    unittest.main()
