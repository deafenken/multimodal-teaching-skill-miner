from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.generate_release_acceptance import (
    ACCEPTANCE_SCHEMA,
    TASK_TWO_RELEASE_FILES,
    AcceptanceError,
    _recompute_teachobs_governance_binding,
    _task_two_release_binding,
    _teachobs_governance_binding,
    build_acceptance,
    record_project,
    record_wheel,
    write_pending,
)
from teaching_skill_miner.release_audit import audit_release_path


class ReleaseAcceptanceTests(unittest.TestCase):
    def _write_json(self, path: Path, value: dict[str, object]) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _fixture(self, root: Path) -> dict[str, Path]:
        (root / "teaching_skill_miner/web").mkdir(parents=True)
        (root / "scripts").mkdir()
        (root / "tests").mkdir()
        (root / "release").mkdir()
        (root / "docs").mkdir()
        (root / "dist").mkdir()
        public = root / "artifacts/public"
        public.mkdir(parents=True)
        (root / "pyproject.toml").write_text(
            '[project]\nname="teaching-skill-miner"\nversion="1.2.0"\n',
            encoding="utf-8",
        )
        (root / "scripts/verify_project.sh").write_text("exit 0\n", encoding="utf-8")
        (root / "scripts/verify_release_wheel.sh").write_text(
            "exit 0\n", encoding="utf-8"
        )
        (root / "scripts/generate_release_acceptance.py").write_text(
            "# fixture\n", encoding="utf-8"
        )
        (root / "tests/test_fixture.py").write_text(
            "def test_ok(): pass\n", encoding="utf-8"
        )
        (root / "release/public_json_resources.txt").write_text(
            "schema/example.json\n", encoding="utf-8"
        )
        (root / "teaching_skill_miner/__init__.py").write_text(
            '__version__ = "1.2.0"\n', encoding="utf-8"
        )
        (root / "teaching_skill_miner/web/index.html").write_text(
            "<!doctype html><title>fixture</title>\n", encoding="utf-8"
        )
        (root / "README.md").write_text("# Fixture\n", encoding="utf-8")
        (root / "docs/status.md").write_text("# Status\n", encoding="utf-8")
        (public / "README.md").write_text(
            "Aggregate public evidence only.\n", encoding="utf-8"
        )

        wheel = root / "dist/teaching_skill_miner-1.2.0-py3-none-any.whl"
        metadata = (
            "Metadata-Version: 2.4\n"
            "Name: teaching-skill-miner\n"
            "Version: 1.2.0\n"
            "License-Expression: LicenseRef-Academic-Evaluation-1.0\n"
            "Requires-Python: >=3.10\n\n"
        )
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "teaching_skill_miner/__init__.py", '__version__ = "1.2.0"\n'
            )
            archive.writestr("teaching_skill_miner-1.2.0.dist-info/METADATA", metadata)
        (root / "second-dist").mkdir()
        second_wheel = root / "second-dist" / wheel.name
        shutil.copyfile(wheel, second_wheel)

        wheel_audit = root / "wheel-audit.json"
        public_audit = root / "public-audit.json"
        self._write_json(wheel_audit, audit_release_path(wheel))
        self._write_json(public_audit, audit_release_path(public))
        allowlist = root / "allowlist.json"
        self._write_json(
            allowlist,
            {
                "schema_version": "1.0",
                "audit_kind": "exact_release_wheel_allowlist",
                "member_count": 2,
                "wheel_size_bytes": wheel.stat().st_size,
                "wheel_sha256": sha256(wheel.read_bytes()).hexdigest(),
                "passed": True,
                "errors": [],
                "missing_members": [],
                "unexpected_members": [],
                "content_mismatches": [],
                "duplicate_members": [],
            },
        )
        junit = root / "pytest.xml"
        junit.write_text(
            "<?xml version='1.0'?>"
            "<testsuites><testsuite tests='3' errors='0' failures='0' skipped='0'>"
            "<testcase classname='tests.test_fixture' name='test_ok'/>"
            "<testcase classname='tests.test_fixture' name='test_subtests'/>"
            "</testsuite></testsuites>",
            encoding="utf-8",
        )
        return {
            "wheel": wheel,
            "second_wheel": second_wheel,
            "public": public,
            "wheel_audit": wheel_audit,
            "public_audit": public_audit,
            "allowlist": allowlist,
            "junit": junit,
        }

    def _receipts(self, root: Path, paths: dict[str, Path]) -> tuple[Path, Path]:
        exact = root / "exact.json"
        record_wheel(
            argparse.Namespace(
                repository_root=root,
                wheel=paths["wheel"],
                release_audit=paths["wheel_audit"],
                allowlist_audit=paths["allowlist"],
                output=exact,
            )
        )
        project = root / "project.json"
        with patch(
            "scripts.generate_release_acceptance._tool_version",
            return_value="fixture-version",
        ):
            record_project(
                argparse.Namespace(
                    repository_root=root,
                    junit_xml=paths["junit"],
                    first_wheel=paths["wheel"],
                    second_wheel=paths["second_wheel"],
                    exact_wheel_receipt=exact,
                    public_directory=paths["public"],
                    public_release_audit=paths["public_audit"],
                    formal_caption_audit_run=False,
                    teachobs_private_receipt_audit_run=False,
                    tracked_file_privacy_scan_run=False,
                    output=project,
                )
            )
        return exact, project

    def _build(
        self,
        root: Path,
        paths: dict[str, Path],
        exact: Path,
        project: Path,
        output: Path,
    ) -> dict[str, object]:
        return build_acceptance(
            argparse.Namespace(
                repository_root=root,
                wheel=paths["wheel"],
                project_verification_receipt=project,
                exact_wheel_receipt=exact,
                wheel_release_audit=paths["wheel_audit"],
                public_directory=paths["public"],
                public_release_audit=paths["public_audit"],
                output=output,
            )
        )

    def test_acceptance_is_deterministic_and_contains_only_bound_positive_claims(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact, project = self._receipts(root, paths)
            first = root / "acceptance-a.json"
            second = root / "acceptance-b.json"
            result = self._build(root, paths, exact, project, first)
            self._build(root, paths, exact, project, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(result["schema_version"], ACCEPTANCE_SCHEMA)
            self.assertEqual(result["release_version"], "1.2.0")
            self.assertEqual(result["validation"]["pytest"]["testcase_count"], 2)
            self.assertEqual(result["validation"]["pytest"]["subtest_count"], 1)
            self.assertEqual(result["release_artifact"]["member_count"], 2)
            self.assertEqual(result["public_artifacts"]["member_count"], 1)
            self.assertIs(
                result["conditional_validation"]["tracked_file_privacy_scan_run"],
                False,
            )
            self.assertIs(
                result["conditional_validation"]["teachobs_private_receipt_audit_run"],
                False,
            )
            self.assertTrue(
                result["generation_policy"]["timestamp_omitted_for_reproducibility"]
            )
            boundaries = result["research_claim_boundaries"]
            self.assertFalse(boundaries["independent_human_review_complete"])
            self.assertFalse(boundaries["confirmatory_multimodal_gain_established"])
            self.assertFalse(boundaries["prospective_deployment_accuracy_established"])
            self.assertFalse(boundaries["real_learner_effectiveness_established"])
            self.assertFalse(
                boundaries["teacher_agent_free_text_diagnostic_accuracy_established"]
            )
            self.assertIsNone(result["task_two_evidence"])

    def test_task_two_release_binding_is_complete_safe_and_stale_sensitive(
        self,
    ) -> None:
        project = Path(__file__).resolve().parents[1]
        current = _task_two_release_binding(project)
        self.assertIsNotNone(current)
        assert current is not None
        self.assertEqual(current["file_count"], len(TASK_TWO_RELEASE_FILES))
        self.assertEqual(current["evidence_summary"]["runtime_skill_count"], 16)
        self.assertEqual(
            current["evidence_summary"]["free_text_benchmark_case_count"], 28
        )
        self.assertTrue(
            current["evidence_summary"]["free_text_online_development_run_completed"]
        )
        self.assertEqual(
            current["evidence_summary"]["multiturn_benchmark_episode_count"], 20
        )
        self.assertEqual(
            current["evidence_summary"]["multiturn_benchmark_learner_turn_count"], 65
        )
        self.assertEqual(
            current["evidence_summary"]["multiturn_benchmark_profile_replacement_count"],
            1,
        )
        self.assertFalse(
            current["evidence_summary"]["multiturn_benchmark_online_run_completed"]
        )
        self.assertFalse(
            current["evidence_summary"]["neural_v1_materialization_gate_passed"]
        )
        self.assertFalse(current["claim_boundaries"]["deployment_accuracy_established"])
        self.assertFalse(current["claim_boundaries"]["full_live_session_quality_established"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in TASK_TWO_RELEASE_FILES.values():
                source = project / relative
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            copied = _task_two_release_binding(root)
            self.assertEqual(copied, current)

            receipt_path = (
                root
                / TASK_TWO_RELEASE_FILES["teacher_agent_free_text_benchmark_receipt"]
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["claim_boundary"]["deployment_accuracy_established"] = True
            self._write_json(receipt_path, receipt)
            with self.assertRaisesRegex(AcceptanceError, "stale or overclaimed"):
                _task_two_release_binding(root)

            shutil.copyfile(
                project
                / TASK_TWO_RELEASE_FILES["teacher_agent_multiturn_benchmark"],
                root / TASK_TWO_RELEASE_FILES["teacher_agent_multiturn_benchmark"],
            )
            multiturn_path = (
                root / TASK_TWO_RELEASE_FILES["teacher_agent_multiturn_benchmark"]
            )
            multiturn = json.loads(multiturn_path.read_text(encoding="utf-8"))
            multiturn["claim_boundary"]["deployment_accuracy_established"] = True
            self._write_json(multiturn_path, multiturn)
            with self.assertRaisesRegex(AcceptanceError, "multi-turn benchmark"):
                _task_two_release_binding(root)

            shutil.copyfile(
                project
                / TASK_TWO_RELEASE_FILES["teacher_agent_free_text_benchmark_receipt"],
                receipt_path,
            )
            (root / TASK_TWO_RELEASE_FILES["teacher_agent_dashboard_js"]).unlink()
            with self.assertRaisesRegex(AcceptanceError, "partial"):
                _task_two_release_binding(root)

    def test_acceptance_rejects_stale_wheel_public_audit_and_verification_scope(
        self,
    ) -> None:
        for tamper in (
            "wheel",
            "public",
            "verification_script",
            "package_source",
            "dashboard",
            "documentation",
        ):
            with (
                self.subTest(tamper=tamper),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                paths = self._fixture(root)
                exact, project = self._receipts(root, paths)
                if tamper == "wheel":
                    with paths["wheel"].open("ab") as handle:
                        handle.write(b"stale")
                elif tamper == "public":
                    (paths["public"] / "README.md").write_text(
                        "changed after audit\n", encoding="utf-8"
                    )
                elif tamper == "verification_script":
                    (root / "scripts/verify_project.sh").write_text(
                        "exit 1\n", encoding="utf-8"
                    )
                elif tamper == "package_source":
                    (root / "teaching_skill_miner/__init__.py").write_text(
                        '__version__ = "tampered"\n', encoding="utf-8"
                    )
                elif tamper == "dashboard":
                    (root / "teaching_skill_miner/web/index.html").write_text(
                        "<!doctype html><title>changed</title>\n",
                        encoding="utf-8",
                    )
                else:
                    (root / "docs/status.md").write_text(
                        "# Changed after verification\n", encoding="utf-8"
                    )
                with self.assertRaises(AcceptanceError):
                    self._build(
                        root,
                        paths,
                        exact,
                        project,
                        root / "acceptance.json",
                    )

    def test_failed_or_incomplete_evidence_cannot_create_positive_receipts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            allowlist = json.loads(paths["allowlist"].read_text(encoding="utf-8"))
            allowlist["passed"] = False
            self._write_json(paths["allowlist"], allowlist)
            with self.assertRaises(AcceptanceError):
                record_wheel(
                    argparse.Namespace(
                        repository_root=root,
                        wheel=paths["wheel"],
                        release_audit=paths["wheel_audit"],
                        allowlist_audit=paths["allowlist"],
                        output=root / "exact.json",
                    )
                )

            junit = (
                paths["junit"]
                .read_text(encoding="utf-8")
                .replace("failures='0'", "failures='1'")
            )
            paths["junit"].write_text(junit, encoding="utf-8")
            with self.assertRaises(AcceptanceError):
                from scripts.generate_release_acceptance import _pytest_summary

                _pytest_summary(paths["junit"])

    def test_project_receipt_records_conditional_teachobs_private_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact = root / "exact.json"
            record_wheel(
                argparse.Namespace(
                    repository_root=root,
                    wheel=paths["wheel"],
                    release_audit=paths["wheel_audit"],
                    allowlist_audit=paths["allowlist"],
                    output=exact,
                )
            )
            media = (
                root
                / "artifacts/private/external_datasets/teachobs/media/media_manifest.json"
            )
            captions = (
                root
                / "artifacts/private/external_datasets/teachobs/captions/caption_audit.json"
            )
            media.parent.mkdir(parents=True)
            captions.parent.mkdir(parents=True)
            media.write_text("{}\n", encoding="utf-8")
            captions.write_text("{}\n", encoding="utf-8")

            def arguments(*, audit_run: bool, output: Path) -> argparse.Namespace:
                return argparse.Namespace(
                    repository_root=root,
                    junit_xml=paths["junit"],
                    first_wheel=paths["wheel"],
                    second_wheel=paths["second_wheel"],
                    exact_wheel_receipt=exact,
                    public_directory=paths["public"],
                    public_release_audit=paths["public_audit"],
                    formal_caption_audit_run=False,
                    teachobs_private_receipt_audit_run=audit_run,
                    tracked_file_privacy_scan_run=False,
                    output=output,
                )

            with self.assertRaisesRegex(
                AcceptanceError, "TeachObs private receipt-audit state"
            ):
                record_project(
                    arguments(audit_run=False, output=root / "rejected.json")
                )

            annotation = (
                root / "artifacts/private/external_datasets/teachobs/"
                "imported_annotations/dataset_audit.json"
            )
            annotation.parent.mkdir(parents=True)
            annotation.write_text("{}\n", encoding="utf-8")
            self._write_json(
                paths["public"] / "teachobs_annotation_receipt.json",
                {"private_audit_sha256": sha256(annotation.read_bytes()).hexdigest()},
            )
            self._write_json(
                paths["public"] / "teachobs_caption_receipt.json",
                {"private_audit_sha256": sha256(captions.read_bytes()).hexdigest()},
            )
            self._write_json(
                paths["public"] / "teachobs_asr_receipt.json",
                {
                    "source_hashes": {
                        "media_manifest_file_sha256": sha256(
                            media.read_bytes()
                        ).hexdigest(),
                        "caption_audit_file_sha256": sha256(
                            captions.read_bytes()
                        ).hexdigest(),
                        "job_manifest_file_sha256": None,
                        "asr_import_audit_file_sha256": None,
                        "coverage_matrix_file_sha256": None,
                    }
                },
            )
            self._write_json(paths["public_audit"], audit_release_path(paths["public"]))
            with patch(
                "scripts.generate_release_acceptance._tool_version",
                return_value="fixture-version",
            ):
                project = root / "accepted.json"
                result = record_project(arguments(audit_run=True, output=project))
            self.assertIs(
                result["conditional_checks"]["teachobs_private_receipt_audit_passed"],
                True,
            )
            binding = result["conditional_checks"]["teachobs_private_receipt_binding"]
            self.assertEqual(binding["file_count"], 6)
            self.assertEqual(len(binding["binding_sha256"]), 64)
            self._build(
                root,
                paths,
                exact,
                project,
                root / "accepted-release.json",
            )
            self.assertTrue(
                audit_release_path(root / "accepted-release.json")["passed"]
            )
            media.write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                AcceptanceError, "TeachObs ASR receipt is stale"
            ):
                self._build(
                    root,
                    paths,
                    exact,
                    project,
                    root / "stale-release.json",
                )

    def test_teachobs_governance_binding_is_complete_and_stale_sensitive(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            human_manifest = (
                root / "artifacts/private/external_datasets/teachobs/human_fresh/"
                "assignment_manifest.json"
            )
            human_receipt = root / "artifacts/public/teachobs_human_fresh_receipt.json"
            lockbox_draft = root / "artifacts/public/teachobs_lockbox_fresh_draft.json"
            human_manifest.parent.mkdir(parents=True)
            human_receipt.parent.mkdir(parents=True)
            for path in (human_manifest, human_receipt, lockbox_draft):
                path.write_text("{}\n", encoding="utf-8")
            binding = _teachobs_governance_binding(
                root,
                human_manifest=human_manifest,
                human_receipt=human_receipt,
                lockbox_draft=lockbox_draft,
            )
            self.assertIsNotNone(binding)
            assert binding is not None
            self.assertEqual(binding["file_count"], 3)
            self.assertEqual(
                _recompute_teachobs_governance_binding(root, binding),
                binding,
            )
            human_receipt.write_text('{"changed":true}\n', encoding="utf-8")
            self.assertNotEqual(
                _recompute_teachobs_governance_binding(root, binding),
                binding,
            )
            with self.assertRaisesRegex(AcceptanceError, "requires human manifest"):
                _teachobs_governance_binding(
                    root,
                    human_manifest=human_manifest,
                    human_receipt=None,
                    lockbox_draft=lockbox_draft,
                )

    def test_release_shells_emit_and_consume_bound_receipts(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        verify_project = (project_root / "scripts/verify_project.sh").read_text(
            encoding="utf-8"
        )
        verify_wheel = (project_root / "scripts/verify_release_wheel.sh").read_text(
            encoding="utf-8"
        )
        finalizer = (project_root / "scripts/build_release_acceptance.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--junitxml=", verify_project)
        self.assertIn("record-project", verify_project)
        self.assertIn("check_teachobs_external_study.py asr", verify_project)
        self.assertIn("check_teachobs_external_study.py benchmark", verify_project)
        self.assertIn("check_teachobs_external_study.py human", verify_project)
        self.assertIn("check_teachobs_external_study.py lockbox", verify_project)
        self.assertIn("teachobs_benchmark_required=true", verify_project)
        self.assertIn(
            '"$teachobs_benchmark_required" = true',
            verify_project,
        )
        self.assertIn(
            "Completed TeachObs ASR/materialization requires a current four-arm",
            verify_project,
        )
        self.assertIn("TSM_TEACHOBS_HUMAN_ROOT", verify_project)
        self.assertIn("TSM_TEACHOBS_HUMAN_RECEIPT", verify_project)
        self.assertIn("TSM_TEACHOBS_LOCKBOX_DRAFT", verify_project)
        self.assertIn("TSM_TEACHOBS_FROZEN_MODEL_ROOT", verify_project)
        self.assertIn("--teachobs-human-manifest", verify_project)
        self.assertIn("--teachobs-human-receipt", verify_project)
        self.assertIn("--teachobs-lockbox-draft", verify_project)
        benchmark_gate = verify_project.split(
            'teachobs_benchmark_result="$repo_root/', 1
        )[1].split("teachobs_private_receipt_audit_run=true", 1)[0]
        for required in (
            '[ ! -f "$teachobs_benchmark_result" ]',
            '[ ! -f "$teachobs_benchmark_receipt" ]',
            '[ ! -f "$teachobs_frozen_root/bundle_manifest.json" ]',
            '[ "$teachobs_transcript_materialization_status" != complete ]',
        ):
            self.assertIn(required, benchmark_gate)
        self.assertLess(
            benchmark_gate.index("teachobs_benchmark_required=true"),
            benchmark_gate.index("check_teachobs_external_study.py benchmark"),
        )
        self.assertLess(
            benchmark_gate.index("check_teachobs_external_study.py benchmark"),
            benchmark_gate.index("check_teachobs_external_study.py human"),
        )
        self.assertLess(
            benchmark_gate.index("check_teachobs_external_study.py human"),
            benchmark_gate.index("check_teachobs_external_study.py lockbox"),
        )
        self.assertIn("--teachobs-private-receipt-audit-run", verify_project)
        self.assertIn("record-wheel", verify_wheel)
        self.assertIn(
            "materialize-teachobs-transcripts --help",
            verify_wheel,
        )
        self.assertIn(
            "--transcript-materialization-manifest",
            verify_wheel,
        )
        self.assertIn('cmp "$wheel" "$comparison_wheel"', verify_project)
        self.assertIn("scripts/verify_project.sh", finalizer)
        self.assertIn("scripts/verify_release_wheel.sh", finalizer)
        self.assertIn('release-audit "$candidate_wheel"', finalizer)
        self.assertIn('release-audit "$staged_acceptance"', finalizer)
        self.assertIn("generate_release_acceptance.py acceptance", finalizer)
        self.assertLess(
            finalizer.index("generate_release_acceptance.py pending"),
            finalizer.index("scripts/verify_project.sh"),
        )

    def test_pending_mode_invalidates_prior_positive_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "acceptance.json"
            output.write_text('{"overall_status":"passed"}\n', encoding="utf-8")
            result = write_pending(
                argparse.Namespace(release_version="1.2.0", output=output)
            )
            self.assertEqual(result["overall_status"], "stale_not_accepted")
            self.assertIs(result["engineering_release_accepted"], False)
            self.assertNotIn(
                '"overall_status":"passed"', output.read_text(encoding="utf-8")
            )

    def test_checked_in_acceptance_is_explicitly_pending_or_machine_generated(
        self,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        receipt = json.loads(
            (project_root / "artifacts/release_acceptance_1.2.0.json").read_text(
                encoding="utf-8"
            )
        )
        boundaries = receipt["research_claim_boundaries"]
        for field in (
            "independent_human_review_complete",
            "confirmatory_multimodal_gain_established",
            "prospective_deployment_accuracy_established",
            "real_learner_effectiveness_established",
            "stable_cross_session_accuracy_0_9_established",
        ):
            self.assertIs(boundaries[field], False)
        if receipt["schema_version"].endswith(".pending.v1"):
            self.assertEqual(receipt["overall_status"], "stale_not_accepted")
            self.assertIs(receipt["engineering_release_accepted"], False)
            self.assertIs(
                receipt["prior_acceptance_values_valid_for_current_source"], False
            )
        else:
            self.assertEqual(receipt["schema_version"], ACCEPTANCE_SCHEMA)
            self.assertEqual(
                receipt["generation_policy"]["method"],
                "fresh_receipts_and_exact_artifact_recomputation",
            )


if __name__ == "__main__":
    unittest.main()
