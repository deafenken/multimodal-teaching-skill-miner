from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from scripts.generate_release_acceptance import (
    ACCEPTANCE_SCHEMA,
    TASK_TWO_RELEASE_FILES,
    AcceptanceError,
    _recompute_teachobs_governance_binding,
    _release_source_snapshot_binding,
    _task_two_release_binding,
    _teachobs_governance_binding,
    build_acceptance,
    record_project,
    record_wheel,
    write_verification_scope,
    write_pending,
)
from scripts.create_release_source_snapshot import create_snapshot
from scripts.verify_published_release_pair import (
    PublishedPairError,
    verify_published_pair,
)
from teaching_skill_miner.release_audit import audit_release_path


class ReleaseAcceptanceTests(unittest.TestCase):
    def _write_json(self, path: Path, value: dict[str, object]) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _scope_receipt(self, root: Path) -> Path:
        receipt = root / "initial-verification-scope.json"
        write_verification_scope(
            argparse.Namespace(repository_root=root, output=receipt)
        )
        return receipt

    def _final_scope_receipt(self, root: Path) -> Path:
        receipt = root / "artifacts/final-verification-scope.json"
        write_verification_scope(
            argparse.Namespace(repository_root=root, output=receipt)
        )
        return receipt

    def _verify_pair(
        self,
        *,
        root: Path,
        paths: dict[str, Path],
        exact: Path,
        project: Path,
        acceptance: Path,
        scope: Path,
        allowlist_override: dict[str, object] | None = None,
    ) -> dict[str, object]:
        wheel = paths["wheel"]
        allowlist = {
            "passed": True,
            "errors": [],
            "missing_members": [],
            "unexpected_members": [],
            "content_mismatches": [],
            "duplicate_members": [],
            "member_count": 2,
            "wheel_size_bytes": wheel.stat().st_size,
            "wheel_sha256": sha256(wheel.read_bytes()).hexdigest(),
        }
        if allowlist_override is not None:
            allowlist = allowlist_override
        with patch(
            "scripts.verify_published_release_pair.verify_wheel_allowlist",
            return_value=allowlist,
        ):
            return verify_published_pair(
                repository_root=root,
                wheel=wheel,
                acceptance=acceptance,
                expected_source_scope=scope,
                source_snapshot_manifest=root / ".release-source-snapshot.json",
                project_verification_receipt=project,
                exact_wheel_verification_receipt=exact,
            )

    def _write_snapshot_manifest(self, root: Path) -> Path:
        marker = root / ".release-snapshot-marker.bin"
        marker.write_text("sealed fixture source\n", encoding="utf-8")
        row = {
            "path": marker.relative_to(root).as_posix(),
            "size_bytes": marker.stat().st_size,
            "sha256": sha256(marker.read_bytes()).hexdigest(),
            "source_kind": "release_source_or_public_evidence",
            "executable": False,
        }
        compact = {key: row[key] for key in ("path", "size_bytes", "sha256")}
        binding = {
            "file_count": 1,
            "total_size_bytes": row["size_bytes"],
            "sha256": sha256(
                json.dumps(
                    [compact], sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
        }
        empty_binding = {
            "file_count": 0,
            "total_size_bytes": 0,
            "sha256": sha256(b"[]").hexdigest(),
        }
        manifest = root / ".release-source-snapshot.json"
        self._write_json(
            manifest,
            {
                "schema_version": "teaching_skill_miner.release_source_snapshot.v1",
                "snapshot_sha256": binding["sha256"],
                "source_snapshot_binding": binding,
                "conditional_private_evidence_binding": empty_binding,
                "files": [row],
            },
        )
        return manifest

    def _fixture(self, root: Path) -> dict[str, Path]:
        (root / "teaching_skill_miner/web").mkdir(parents=True)
        (root / "scripts").mkdir()
        (root / "tests").mkdir()
        (root / "release").mkdir()
        (root / "docs").mkdir()
        (root / "apps/api/src").mkdir(parents=True)
        (root / "deploy/production").mkdir(parents=True)
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
        self._write_snapshot_manifest(root)
        (root / "teaching_skill_miner/__init__.py").write_text(
            '__version__ = "1.2.0"\n', encoding="utf-8"
        )
        (root / "teaching_skill_miner/web/index.html").write_text(
            "<!doctype html><title>fixture</title>\n", encoding="utf-8"
        )
        (root / "README.md").write_text("# Fixture\n", encoding="utf-8")
        (root / "docs/status.md").write_text("# Status\n", encoding="utf-8")
        (root / "apps/api/src/main.ts").write_text(
            "export const fixture = true;\n", encoding="utf-8"
        )
        (root / "deploy/production/Caddyfile").write_text(
            ':443 { respond "fixture" }\n', encoding="utf-8"
        )
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
                    expected_verification_scope=self._scope_receipt(root),
                    source_snapshot_manifest=root / ".release-source-snapshot.json",
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
                source_snapshot_manifest=root / ".release-source-snapshot.json",
                output=output,
            )
        )

    def _release_wrapper_fixture(self, root: Path) -> None:
        project_root = Path(__file__).resolve().parents[1]
        scripts = root / "scripts"
        package = root / "teaching_skill_miner"
        public = root / "artifacts/public"
        scripts.mkdir()
        package.mkdir()
        public.mkdir(parents=True)
        for name in (
            "build_release_acceptance.sh",
            "create_release_source_snapshot.py",
            "verify_published_release_pair.py",
        ):
            # Release snapshots deliberately make source files read-only.
            # These fixture copies are subsequently replaced with injected
            # test doubles, so do not preserve the snapshot's 0444 mode.
            shutil.copyfile(project_root / "scripts" / name, scripts / name)
        (package / "__init__.py").write_text(
            '__version__ = "1.2.0"\n', encoding="utf-8"
        )
        (root / "release_marker.txt").write_text("A\n", encoding="utf-8")
        (scripts / "generate_release_acceptance.py").write_text(
            """import json
import os
from pathlib import Path
import sys
import time

def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]

if __name__ == "__main__":
    output = Path(argument("--output"))
    output.write_text(json.dumps({
        "schema_version": "teaching_skill_miner.release_acceptance.pending.v1",
        "overall_status": "stale_not_accepted",
        "engineering_release_accepted": False,
    }) + "\\n", encoding="utf-8")
    time.sleep(float(os.environ.get("TSM_RELEASE_TEST_PRE_SNAPSHOT_HOLD_SECONDS", "0")))
""",
            encoding="utf-8",
        )
        # These wrapper tests exercise publication ordering and locking.  The
        # full verifier is covered independently below, so use a tiny injected
        # gate that can deterministically fail before publication.
        (scripts / "verify_published_release_pair.py").write_text(
            """import os
from pathlib import Path
import sys

def argument(name: str) -> Path:
    return Path(sys.argv[sys.argv.index(name) + 1])

if os.environ.get("TSM_RELEASE_TEST_FINAL_VERIFY_FAIL") == "true":
    raise SystemExit(11)
for flag in ("--wheel", "--acceptance", "--expected-source-scope", "--source-snapshot-manifest"):
    if not argument(flag).is_file():
        raise SystemExit(f"missing fixture verifier input: {flag}")
print("published_release_pair_verified=fixture")
""",
            encoding="utf-8",
        )
        (scripts / "build_release_acceptance_snapshot.sh").write_text(
            """#!/usr/bin/env sh
set -eu
if [ "${TSM_RELEASE_TEST_FAIL:-false}" = true ]; then exit 9; fi
sleep "${TSM_RELEASE_TEST_HOLD_SECONDS:-0}"
output=$2
mkdir -p dist artifacts/public
"$PYTHON" - "$output" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys
import zipfile

root = Path.cwd()
wheel = root / "dist/teaching_skill_miner-1.2.0-py3-none-any.whl"
metadata = (
    "Metadata-Version: 2.4\\n"
    "Name: teaching-skill-miner\\n"
    "Version: 1.2.0\\n"
    "License-Expression: LicenseRef-Academic-Evaluation-1.0\\n"
    "Requires-Python: >=3.10\\n\\n"
)
with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("teaching_skill_miner/__init__.py", '__version__ = "1.2.0"\\n')
    archive.writestr("teaching_skill_miner-1.2.0.dist-info/METADATA", metadata)
body = wheel.read_bytes()
marker = (root / "release_marker.txt").read_bytes()
scope = {"file_count": 1, "sha256": sha256(marker).hexdigest()}
snapshot = json.loads((root / ".release-source-snapshot.json").read_text())

def members_digest(rows):
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()

with zipfile.ZipFile(wheel) as archive:
    wheel_rows = [
        {
            "path": info.filename,
            "size_bytes": info.file_size,
            "sha256": sha256(archive.read(info)).hexdigest(),
        }
        for info in sorted(archive.infolist(), key=lambda item: item.filename)
        if not info.is_dir()
    ]
public_root = root / "artifacts/public"
public_rows = [
    {
        "path": path.relative_to(public_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }
    for path in sorted(public_root.rglob("*"))
    if path.is_file()
]
(root / "artifacts/final-verification-scope.json").write_text(json.dumps({
    "schema_version": "teaching_skill_miner.verification_scope.v1",
    "verification_scope": scope,
}) + "\\n")
receipt = {
    "schema_version": "teaching_skill_miner.release_acceptance.v2",
    "release_version": "1.2.0",
    "overall_status": "engineering_ready_external_validation_pending",
    "generation_policy": {
        "method": "fresh_receipts_and_exact_artifact_recomputation",
        "copied_from_previous_acceptance": False,
        "timestamp_omitted_for_reproducibility": True,
    },
    "validation": {
        "pytest": {
            "passed": True,
            "testcase_count": 1,
            "subtest_count": 0,
            "total_count": 1,
            "skipped_count": 0,
            "passed_count": 1,
        },
        "ruff_passed": True,
        "compileall_passed": True,
        "shell_syntax_passed": True,
        "source_entrypoint_smoke_passed": True,
        "pip_check_passed": True,
        "doctor_passed": True,
        "demo_passed": True,
        "delivery_verification_passed": True,
        "public_release_audit_passed": True,
        "byte_identical_double_build_passed": True,
        "exact_release_wheel_verification_passed": True,
        "generated_skill_schema_validation_passed": True,
        "task_two_entrypoints_and_claim_boundaries_passed": True,
        "api_locked_install_audit_tests_typecheck_build_passed": True,
        "console_locked_install_audit_tests_typecheck_sealed_build_passed": True,
        "deployment_sources_bound_to_receipt": True,
        "wheel_release_audit_passed": True,
    },
    "conditional_validation": {
        "formal_caption_private_audit_run": False,
        "formal_caption_private_audit_passed": None,
        "teachobs_private_receipt_audit_run": False,
        "teachobs_private_receipt_audit_passed": None,
        "teachobs_private_receipt_binding": None,
        "teachobs_governance_artifacts_checked": False,
        "teachobs_governance_binding": None,
        "tracked_file_privacy_scan_run": True,
        "tracked_file_privacy_scan_passed": True,
        "task_two_public_evidence_checked": False,
        "task_two_public_evidence_binding": None,
    },
    "release_artifact": {
        "artifact": "dist/" + wheel.name,
        "distribution": "teaching-skill-miner",
        "version": "1.2.0",
        "size_bytes": len(body),
        "uncompressed_bytes": sum(row["size_bytes"] for row in wheel_rows),
        "member_count": len(wheel_rows),
        "sha256": sha256(body).hexdigest(),
        "license_expression": "LicenseRef-Academic-Evaluation-1.0",
        "requires_python": ">=3.10",
        "release_audit_finding_count": 0,
        "members_sha256": members_digest(wheel_rows),
    },
    "public_artifacts": {
        "directory": "artifacts/public",
        "member_count": len(public_rows),
        "total_size_bytes": sum(row["size_bytes"] for row in public_rows),
        "release_audit_finding_count": 0,
        "members_sha256": members_digest(public_rows),
    },
    "environment": {
        "python": "fixture",
        "python_implementation": "fixture",
        "platform": "fixture",
        "ruff": "fixture",
        "ffmpeg": None,
        "ffprobe": None,
        "tesseract": None,
    },
    "verification_binding": {
        "verification_scope": scope,
        "project_verification_receipt_sha256": "0" * 64,
        "exact_wheel_verification_receipt_sha256": "1" * 64,
        "release_source_snapshot": {
            "source_snapshot": snapshot["source_snapshot_binding"],
            "conditional_private_evidence": snapshot[
                "conditional_private_evidence_binding"
            ],
        },
    },
    "task_two_evidence": None,
    "research_claim_boundaries": {
        "independent_human_review_complete": False,
        "confirmatory_multimodal_gain_established": False,
        "prospective_deployment_accuracy_established": False,
        "real_learner_effectiveness_established": False,
        "stable_cross_session_accuracy_0_9_established": False,
        "teacher_agent_free_text_diagnostic_accuracy_established": False,
        "teacher_agent_skill_routing_quality_established": False,
        "teacher_agent_deployment_accuracy_established": False,
        "note": (
            "Release engineering checks do not establish annotation validity, "
            "confirmatory multimodal gain, deployment accuracy, or learner effects."
        ),
    },
    "unverified_external_state": {
        "git_checkout_available": True,
        "tracked_file_privacy_scan_run": True,
        "remote_github_actions_run_verified": False,
    },
}
Path(sys.argv[1]).write_text(json.dumps(receipt) + "\\n")
PY
""",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "."], cwd=root, check=True)

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
            current["evidence_summary"][
                "multiturn_benchmark_profile_replacement_count"
            ],
            1,
        )
        self.assertFalse(
            current["evidence_summary"]["multiturn_benchmark_online_run_completed"]
        )
        self.assertFalse(
            current["evidence_summary"]["neural_v1_materialization_gate_passed"]
        )
        self.assertFalse(current["claim_boundaries"]["deployment_accuracy_established"])
        self.assertFalse(
            current["claim_boundaries"]["full_live_session_quality_established"]
        )

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
                project / TASK_TWO_RELEASE_FILES["teacher_agent_multiturn_benchmark"],
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
            "typescript_api",
            "deployment",
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
                elif tamper == "documentation":
                    (root / "docs/status.md").write_text(
                        "# Changed after verification\n", encoding="utf-8"
                    )
                elif tamper == "typescript_api":
                    (root / "apps/api/src/main.ts").write_text(
                        "export const fixture = false;\n", encoding="utf-8"
                    )
                else:
                    (root / "deploy/production/Caddyfile").write_text(
                        ':443 { respond "changed" }\n', encoding="utf-8"
                    )
                with self.assertRaises(AcceptanceError):
                    self._build(
                        root,
                        paths,
                        exact,
                        project,
                        root / "acceptance.json",
                    )

    def test_verification_scope_ignores_generated_node_artifacts_but_binds_lockfiles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact, project = self._receipts(root, paths)
            generated = root / "apps/api/node_modules/vendor"
            generated.mkdir(parents=True)
            (generated / "generated.js").write_text(
                "throw new Error('vendor mutation');\n", encoding="utf-8"
            )
            # Generated/vendor output must not make a source-bound receipt
            # stale merely because npm ci/build materialized it.
            self._build(root, paths, exact, project, root / "accepted.json")

            lockfile = root / "apps/api/package-lock.json"
            lockfile.write_text('{"lockfileVersion":3}\n', encoding="utf-8")
            with self.assertRaises(AcceptanceError):
                self._build(root, paths, exact, project, root / "stale.json")
            lockfile.unlink()

            # Non-JSON supply-chain locks under deploy/ are source too.  In
            # particular the hashed Playwright runner lock must invalidate a
            # receipt if its wheel selection changes.
            deploy_lock = root / "deploy/production/playwright-requirements.lock"
            deploy_lock.parent.mkdir(parents=True, exist_ok=True)
            deploy_lock.write_text(
                "playwright==1.60.0 --hash=sha256:" + "a" * 64 + "\n"
            )
            with self.assertRaises(AcceptanceError):
                self._build(root, paths, exact, project, root / "stale-lock.json")

    def test_verification_scope_binds_all_top_level_data_benchmarks(self) -> None:
        benchmark_names = (
            "teacher_agent_benchmark_v2_development.json",
            "teacher_agent_benchmark_v2_development_gold.json",
            "teacher_agent_dialogue_quality_benchmark_v1.json",
            "teacher_agent_dialogue_quality_predictions_fixture_v1.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            data = root / "data"
            data.mkdir()
            for name in benchmark_names:
                (data / name).write_text('{"version":1}\n', encoding="utf-8")
            exact, project = self._receipts(root, paths)
            for name in benchmark_names:
                with self.subTest(name=name):
                    target = data / name
                    original = target.read_text(encoding="utf-8")
                    target.write_text('{"version":2}\n', encoding="utf-8")
                    with self.assertRaises(AcceptanceError):
                        self._build(
                            root,
                            paths,
                            exact,
                            project,
                            root / f"stale-{name}",
                        )
                    target.write_text(original, encoding="utf-8")

            private_data = data / "real/private-benchmark.json"
            private_data.parent.mkdir()
            private_data.write_text('{"private":true}\n', encoding="utf-8")
            self._build(root, paths, exact, project, root / "private-excluded.json")

    def test_release_source_snapshot_is_minimal_and_aba_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            root.mkdir()
            (root / ".gitignore").write_text("artifacts/*\n", encoding="utf-8")
            (root / "tracked.py").write_text("VALUE = 'A'\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "add", ".gitignore", "tracked.py"], cwd=root, check=True
            )
            (root / "untracked.py").write_text("PENDING = True\n", encoding="utf-8")
            public = root / "artifacts/public"
            public.mkdir(parents=True)
            (public / "receipt.json").write_text("{}\n", encoding="utf-8")
            formal = root / "artifacts/private/formal_captions"
            (formal / "raw").mkdir(parents=True)
            (formal / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
            (formal / "raw/lesson.vtt").write_text("WEBVTT\n", encoding="utf-8")
            teachobs = root / "artifacts/private/external_datasets/teachobs"
            for relative in (
                "media/media_manifest.json",
                "media/feature_manifest.json",
                "captions/caption_audit.json",
                "frozen_models/bundle_manifest.json",
                "human_annotation/assignment_manifest.json",
            ):
                target = teachobs / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{}\n", encoding="utf-8")
            videos = teachobs / "media/videos"
            videos.mkdir()
            (videos / "large-private.mp4").write_bytes(b"private-video")

            destination = Path(directory) / "snapshot"
            with patch.dict(
                os.environ,
                {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith("TSM_TEACHOBS_")
                },
                clear=True,
            ):
                result = create_snapshot(root, destination)
            self.assertLess(
                result["source_snapshot_binding"]["total_size_bytes"],
                128 * 1024 * 1024,
            )
            self.assertTrue((destination / "tracked.py").is_file())
            self.assertTrue((destination / "untracked.py").is_file())
            self.assertTrue((destination / "artifacts/public/receipt.json").is_file())
            self.assertTrue(
                (
                    destination / "artifacts/private/formal_captions/raw/lesson.vtt"
                ).is_file()
            )
            self.assertFalse(
                (
                    destination
                    / "artifacts/private/external_datasets/teachobs/media/videos/large-private.mp4"
                ).exists()
            )
            git_files = subprocess.run(
                ["git", "ls-files"],
                cwd=destination,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
            self.assertEqual(git_files, [".gitignore", "tracked.py"])

            # Live A→B→A writes after capture cannot alter gate input bytes.
            (root / "tracked.py").write_text("VALUE = 'B'\n", encoding="utf-8")
            (root / "tracked.py").write_text("VALUE = 'A'\n", encoding="utf-8")
            self.assertEqual(
                (destination / "tracked.py").read_text(encoding="utf-8"),
                "VALUE = 'A'\n",
            )
            sealed = _release_source_snapshot_binding(
                destination, destination / ".release-source-snapshot.json"
            )
            assert sealed is not None
            self.assertGreater(sealed["conditional_private_evidence"]["file_count"], 0)
            private_copy = (
                destination / "artifacts/private/formal_captions/raw/lesson.vtt"
            )
            private_copy.chmod(0o600)
            private_copy.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(AcceptanceError, "sealed snapshot"):
                _release_source_snapshot_binding(
                    destination, destination / ".release-source-snapshot.json"
                )

    def test_release_snapshot_rejects_partial_private_evidence_trees(self) -> None:
        partial_paths = (
            "media/media_manifest.json",
            "captions/caption_audit.json",
            "asr/job_manifest.json",
            "frozen_models/bundle_manifest.json",
            "human_annotation/assignment_manifest.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for index, relative in enumerate(partial_paths):
                with self.subTest(relative=relative):
                    root = base / f"repository-{index}"
                    root.mkdir()
                    (root / ".gitignore").write_text("artifacts/*\n", encoding="utf-8")
                    (root / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
                    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
                    subprocess.run(
                        ["git", "add", ".gitignore", "tracked.py"],
                        cwd=root,
                        check=True,
                    )
                    target = (
                        root / "artifacts/private/external_datasets/teachobs" / relative
                    )
                    target.parent.mkdir(parents=True)
                    target.write_text("{}\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        RuntimeError, "partial TeachObs private evidence"
                    ):
                        create_snapshot(root, base / f"snapshot-{index}")

            formal_root = base / "formal-repository"
            formal_root.mkdir()
            (formal_root / ".gitignore").write_text("artifacts/*\n", encoding="utf-8")
            (formal_root / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=formal_root, check=True)
            subprocess.run(
                ["git", "add", ".gitignore", "tracked.py"],
                cwd=formal_root,
                check=True,
            )
            partial_formal = (
                formal_root / "artifacts/private/formal_captions/raw/lesson.vtt"
            )
            partial_formal.parent.mkdir(parents=True)
            partial_formal.write_text("WEBVTT\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "partial formal-caption"):
                create_snapshot(formal_root, base / "formal-snapshot")

    def test_release_snapshot_captures_pending_media_caption_pair_with_optional_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "repository"
            root.mkdir()
            (root / ".gitignore").write_text("artifacts/*\n", encoding="utf-8")
            (root / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "add", ".gitignore", "tracked.py"], cwd=root, check=True
            )
            teachobs = root / "artifacts/private/external_datasets/teachobs"
            media = teachobs / "media/media_manifest.json"
            caption = teachobs / "captions/caption_audit.json"
            media.parent.mkdir(parents=True)
            caption.parent.mkdir(parents=True)
            media.write_text("{}\n", encoding="utf-8")
            caption.write_text("{}\n", encoding="utf-8")
            environment = {
                "TSM_TEACHOBS_FROZEN_MODEL_ROOT": (
                    "artifacts/private/external_datasets/teachobs/not_started_frozen"
                ),
                "TSM_TEACHOBS_HUMAN_ROOT": (
                    "artifacts/private/external_datasets/teachobs/not_started_human"
                ),
                "TSM_TEACHOBS_HUMAN_RECEIPT": (
                    "artifacts/public/not_started_human_receipt.json"
                ),
                "TSM_TEACHOBS_LOCKBOX_DRAFT": (
                    "artifacts/public/not_started_lockbox.json"
                ),
            }
            with patch.dict(os.environ, environment, clear=False):
                result = create_snapshot(root, base / "snapshot")
            self.assertGreater(
                result["conditional_private_evidence_binding"]["file_count"], 0
            )
            self.assertTrue(
                (base / "snapshot" / media.relative_to(root)).is_file()
            )
            self.assertTrue(
                (base / "snapshot" / caption.relative_to(root)).is_file()
            )

            with patch.dict(
                os.environ,
                {"TSM_TEACHOBS_FROZEN_MODEL_ROOT": os.fspath(base / "outside")},
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "outside the repository"):
                    create_snapshot(root, base / "outside-path-snapshot")

    def test_snapshot_privacy_gate_scans_untracked_release_source(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            scripts = root / "scripts"
            secret = root / "apps/console/src/untracked-secret.ts"
            scripts.mkdir(parents=True)
            secret.parent.mkdir(parents=True)
            shutil.copy2(
                project_root / "scripts/audit_repository_privacy.py",
                scripts / "audit_repository_privacy.py",
            )
            secret.write_text(
                'export const token = "'
                + "sk-"
                + 'untracked-release-secret-123456";\n',
                encoding="utf-8",
            )
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "add", "scripts/audit_repository_privacy.py"],
                cwd=root,
                check=True,
            )
            destination = Path(directory) / "snapshot"
            create_snapshot(root, destination)
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.fspath(project_root)
            result = subprocess.run(
                [sys.executable, "scripts/audit_repository_privacy.py"],
                cwd=destination,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("possible secret", result.stderr)
            self.assertIn("apps/console/src/untracked-secret.ts", result.stderr)

    def test_acceptance_binds_private_snapshot_bytes_without_leaking_paths(
        self,
    ) -> None:
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
            private = root / "artifacts/private/formal_captions/audit-input.txt"
            private.parent.mkdir(parents=True)
            private.write_text("private aggregate evidence\n", encoding="utf-8")
            compact = {
                "path": "artifacts/private/formal_captions/audit-input.txt",
                "size_bytes": private.stat().st_size,
                "sha256": sha256(private.read_bytes()).hexdigest(),
            }
            digest = sha256(
                json.dumps(
                    [compact],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            binding = {
                "file_count": 1,
                "total_size_bytes": private.stat().st_size,
                "sha256": digest,
            }
            manifest = root / ".release-source-snapshot.json"
            self._write_json(
                manifest,
                {
                    "schema_version": (
                        "teaching_skill_miner.release_source_snapshot.v1"
                    ),
                    "snapshot_sha256": digest,
                    "source_snapshot_binding": binding,
                    "conditional_private_evidence_binding": binding,
                    "files": [
                        {
                            **compact,
                            "source_kind": "conditional_private_evidence",
                            "executable": False,
                        }
                    ],
                },
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
                        expected_verification_scope=self._scope_receipt(root),
                        source_snapshot_manifest=manifest,
                        formal_caption_audit_run=False,
                        teachobs_private_receipt_audit_run=False,
                        tracked_file_privacy_scan_run=False,
                        output=project,
                    )
                )
            acceptance = root / "release_acceptance_1.2.0.json"
            result = self._build(root, paths, exact, project, acceptance)
            snapshot_binding = result["verification_binding"]["release_source_snapshot"]
            self.assertEqual(snapshot_binding["conditional_private_evidence"], binding)
            self.assertNotIn(
                "artifacts/private", acceptance.read_text(encoding="utf-8")
            )
            self.assertTrue(audit_release_path(acceptance)["passed"])

    def test_published_pair_verifier_rejects_wheel_or_scope_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact, project = self._receipts(root, paths)
            acceptance = root / "artifacts/release_acceptance_1.2.0.json"
            self._build(root, paths, exact, project, acceptance)
            scope = self._final_scope_receipt(root)
            verified = self._verify_pair(
                root=root,
                paths=paths,
                exact=exact,
                project=project,
                acceptance=acceptance,
                scope=scope,
            )
            self.assertEqual(
                verified["sha256"], sha256(paths["wheel"].read_bytes()).hexdigest()
            )
            wrong_name = root / "artifacts/custom-name.json"
            shutil.copyfile(acceptance, wrong_name)
            with self.assertRaisesRegex(PublishedPairError, "filename/version"):
                self._verify_pair(
                    root=root,
                    paths=paths,
                    exact=exact,
                    project=project,
                    acceptance=wrong_name,
                    scope=scope,
                )
            with paths["wheel"].open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(PublishedPairError, "size does not match"):
                self._verify_pair(
                    root=root,
                    paths=paths,
                    exact=exact,
                    project=project,
                    acceptance=acceptance,
                    scope=scope,
                )

    def test_published_pair_verifier_rejects_non_positive_receipt_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact, project = self._receipts(root, paths)
            acceptance = root / "artifacts/release_acceptance_1.2.0.json"
            self._build(root, paths, exact, project, acceptance)
            scope = self._final_scope_receipt(root)
            original = json.loads(acceptance.read_text(encoding="utf-8"))

            mutations = (
                (
                    "schema_version",
                    "attacker.release_acceptance.v999",
                    "unexpected positive schema",
                ),
                ("overall_status", "stale_not_accepted", "not in the accepted"),
                ("engineering_release_accepted", False, "mixes pending"),
            )
            for field, value, message in mutations:
                with self.subTest(field=field):
                    tampered = dict(original)
                    tampered[field] = value
                    acceptance.write_text(
                        json.dumps(tampered, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(PublishedPairError, message):
                        self._verify_pair(
                            root=root,
                            paths=paths,
                            exact=exact,
                            project=project,
                            acceptance=acceptance,
                            scope=scope,
                        )

    def test_published_pair_verifier_rejects_false_positive_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact, project = self._receipts(root, paths)
            acceptance = root / "artifacts/release_acceptance_1.2.0.json"
            self._build(root, paths, exact, project, acceptance)
            scope = self._final_scope_receipt(root)
            original = json.loads(acceptance.read_text(encoding="utf-8"))

            mutations = (
                (
                    "validation",
                    lambda value: value.__setitem__(
                        "exact_release_wheel_verification_passed", False
                    ),
                    "validation is not passing",
                ),
                (
                    "research_claim_boundaries",
                    lambda value: value.__setitem__(
                        "real_learner_effectiveness_established", True
                    ),
                    "research claim boundaries",
                ),
                (
                    "public_artifacts",
                    lambda value: value.__setitem__("members_sha256", "f" * 64),
                    "public-artifact binding",
                ),
                (
                    "release_artifact",
                    lambda value: value.__setitem__("member_count", 999),
                    "wheel summary is stale",
                ),
            )
            for field, mutate, message in mutations:
                with self.subTest(field=field):
                    tampered = json.loads(json.dumps(original))
                    mutate(tampered[field])
                    acceptance.write_text(
                        json.dumps(tampered, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(PublishedPairError, message):
                        self._verify_pair(
                            root=root,
                            paths=paths,
                            exact=exact,
                            project=project,
                            acceptance=acceptance,
                            scope=scope,
                        )

    def test_published_pair_verifier_recomputes_snapshot_proofs_and_gates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact, project = self._receipts(root, paths)
            acceptance = root / "artifacts/release_acceptance_1.2.0.json"
            self._build(root, paths, exact, project, acceptance)
            scope = self._final_scope_receipt(root)

            marker = root / ".release-snapshot-marker.bin"
            original_marker = marker.read_bytes()
            marker.write_bytes(b"tampered snapshot byte\n")
            with self.assertRaisesRegex(
                PublishedPairError, "snapshot manifest member binding is stale"
            ):
                self._verify_pair(
                    root=root,
                    paths=paths,
                    exact=exact,
                    project=project,
                    acceptance=acceptance,
                    scope=scope,
                )
            marker.write_bytes(original_marker)

            project.write_bytes(project.read_bytes() + b" \n")
            with self.assertRaisesRegex(
                PublishedPairError, "project verification proof hash is stale"
            ):
                self._verify_pair(
                    root=root,
                    paths=paths,
                    exact=exact,
                    project=project,
                    acceptance=acceptance,
                    scope=scope,
                )

    def test_published_pair_verifier_validates_proof_content_not_only_hash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact, project = self._receipts(root, paths)
            acceptance = root / "artifacts/release_acceptance_1.2.0.json"
            self._build(root, paths, exact, project, acceptance)
            scope = self._final_scope_receipt(root)

            project_value = json.loads(project.read_text(encoding="utf-8"))
            project_value["attacker_field"] = True
            self._write_json(project, project_value)
            acceptance_value = json.loads(acceptance.read_text(encoding="utf-8"))
            acceptance_value["verification_binding"][
                "project_verification_receipt_sha256"
            ] = sha256(project.read_bytes()).hexdigest()
            self._write_json(acceptance, acceptance_value)
            with self.assertRaisesRegex(
                PublishedPairError, "project verification proof has an unexpected"
            ):
                self._verify_pair(
                    root=root,
                    paths=paths,
                    exact=exact,
                    project=project,
                    acceptance=acceptance,
                    scope=scope,
                )

    def test_published_pair_verifier_recomputes_task_two_and_wheel_allowlist(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact, project = self._receipts(root, paths)
            acceptance = root / "artifacts/release_acceptance_1.2.0.json"
            self._build(root, paths, exact, project, acceptance)
            scope = self._final_scope_receipt(root)

            with (
                patch(
                    "scripts.verify_published_release_pair._task_two_release_binding",
                    return_value={"current": "fixed-source-evidence"},
                ),
                self.assertRaisesRegex(PublishedPairError, "Task 2 evidence"),
            ):
                self._verify_pair(
                    root=root,
                    paths=paths,
                    exact=exact,
                    project=project,
                    acceptance=acceptance,
                    scope=scope,
                )

            failed_allowlist = {
                "passed": False,
                "errors": ["unexpected source projection"],
                "missing_members": [],
                "unexpected_members": ["private.bin"],
                "content_mismatches": [],
                "duplicate_members": [],
                "member_count": 3,
                "wheel_size_bytes": paths["wheel"].stat().st_size,
                "wheel_sha256": sha256(paths["wheel"].read_bytes()).hexdigest(),
            }
            with self.assertRaisesRegex(PublishedPairError, "exact allowlist"):
                self._verify_pair(
                    root=root,
                    paths=paths,
                    exact=exact,
                    project=project,
                    acceptance=acceptance,
                    scope=scope,
                    allowlist_override=failed_allowlist,
                )

    def test_published_pair_verifier_reruns_public_privacy_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact, project = self._receipts(root, paths)
            acceptance = root / "artifacts/release_acceptance_1.2.0.json"
            self._build(root, paths, exact, project, acceptance)
            scope = self._final_scope_receipt(root)

            secret = paths["public"] / "forged-proof.txt"
            secret.write_text(
                'api_key="this-is-a-forged-secret-value-123456789"\n',
                encoding="utf-8",
            )
            rows = [
                {
                    "path": path.relative_to(paths["public"]).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(paths["public"].rglob("*"))
                if path.is_file()
            ]
            member_digest = sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            acceptance_value = json.loads(acceptance.read_text(encoding="utf-8"))
            acceptance_value["public_artifacts"].update(
                {
                    "member_count": len(rows),
                    "total_size_bytes": sum(row["size_bytes"] for row in rows),
                    "members_sha256": member_digest,
                }
            )
            project_value = json.loads(project.read_text(encoding="utf-8"))
            project_value["public_release_audit"] = {
                "passed": True,
                "finding_count": 0,
                "member_count": len(rows),
                "total_size_bytes": sum(row["size_bytes"] for row in rows),
                "members_sha256": member_digest,
            }
            self._write_json(project, project_value)
            acceptance_value["verification_binding"][
                "project_verification_receipt_sha256"
            ] = sha256(project.read_bytes()).hexdigest()
            self._write_json(acceptance, acceptance_value)
            with self.assertRaisesRegex(PublishedPairError, "privacy audit did not pass"):
                self._verify_pair(
                    root=root,
                    paths=paths,
                    exact=exact,
                    project=project,
                    acceptance=acceptance,
                    scope=scope,
                )

    def test_published_pair_verifier_recomputes_conditional_evidence_presence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact, project = self._receipts(root, paths)
            acceptance = root / "artifacts/release_acceptance_1.2.0.json"
            self._build(root, paths, exact, project, acceptance)
            scope = self._final_scope_receipt(root)
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
            with self.assertRaisesRegex(
                PublishedPairError, "TeachObs audit state differs"
            ):
                self._verify_pair(
                    root=root,
                    paths=paths,
                    exact=exact,
                    project=project,
                    acceptance=acceptance,
                    scope=scope,
                )

    def test_positive_acceptance_requires_snapshot_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            exact, project = self._receipts(root, paths)
            (root / ".release-source-snapshot.json").unlink()
            with self.assertRaisesRegex(
                AcceptanceError, "requires the source snapshot manifest"
            ):
                self._build(
                    root,
                    paths,
                    exact,
                    project,
                    root / "artifacts/release_acceptance_1.2.0.json",
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
                    expected_verification_scope=self._scope_receipt(root),
                    source_snapshot_manifest=root / ".release-source-snapshot.json",
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
        snapshot_runner = (
            project_root / "scripts/build_release_acceptance_snapshot.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--junitxml=", verify_project)
        self.assertIn("npm --prefix apps/api ci", verify_project)
        self.assertIn("npm --prefix apps/api run build", verify_project)
        self.assertIn("npm --prefix apps/console ci", verify_project)
        self.assertIn("npm --prefix apps/console run build:runtime", verify_project)
        for variable in (
            "TSM_TEACHOBS_HUMAN_ROOT",
            "TSM_TEACHOBS_HUMAN_RECEIPT",
            "TSM_TEACHOBS_LOCKBOX_DRAFT",
            "TSM_TEACHOBS_FROZEN_MODEL_ROOT",
        ):
            self.assertIn(f"unset {variable}", verify_project)
            self.assertLess(
                verify_project.index(f"unset {variable}"),
                verify_project.index('-m pytest --junitxml='),
            )
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
        self.assertLess(
            verify_project.index("teachobs_frozen_root="),
            verify_project.index('[ -e "$teachobs_frozen_root" ]'),
        )
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
        self.assertIn("scripts/verify_project.sh", snapshot_runner)
        self.assertIn("scripts/verify_release_wheel.sh", snapshot_runner)
        self.assertIn('release-audit "$candidate_wheel"', snapshot_runner)
        self.assertIn('release-audit "$staged_acceptance"', snapshot_runner)
        self.assertIn("generate_release_acceptance.py acceptance", snapshot_runner)
        self.assertIn("--project-verification-receipt", snapshot_runner)
        self.assertIn("--exact-wheel-verification-receipt", snapshot_runner)
        self.assertIn(".release-verification-proofs", snapshot_runner)
        self.assertLess(
            finalizer.index("generate_release_acceptance.py pending"),
            finalizer.index("create_release_source_snapshot.py"),
        )
        self.assertIn('mkdir "$release_lock"', finalizer)
        self.assertIn('rmdir "$release_lock"', finalizer)
        self.assertIn("trap cleanup EXIT", finalizer)
        self.assertIn("trap 'exit 143' TERM", finalizer)
        self.assertIn("build_release_acceptance_snapshot.sh", finalizer)
        self.assertIn("verify_published_release_pair.py", finalizer)
        self.assertIn("--project-verification-receipt", finalizer)
        self.assertIn("--exact-wheel-verification-receipt", finalizer)
        self.assertLess(
            finalizer.index("verify_published_release_pair.py"),
            finalizer.index('mv -f "$staged_wheel" "$final_wheel"'),
        )
        positive_publication = finalizer.index(
            'mv -f "$staged_acceptance" "$output_path"'
        )
        self.assertNotIn(
            "verify_published_release_pair.py", finalizer[positive_publication:]
        )
        self.assertIn("Confirm the recorded PID is no longer running", finalizer)
        self.assertNotIn("\nexit 0\n", finalizer)

    def test_release_finalizer_rejects_concurrent_publisher_and_seals_pair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release_wrapper_fixture(root)

            command = ["sh", str(root / "scripts/build_release_acceptance.sh")]
            environment = dict(os.environ)
            environment["PYTHON"] = sys.executable
            environment["TSM_RELEASE_TEST_HOLD_SECONDS"] = "1"
            first = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            lock = root / "artifacts/.release-acceptance.lock"
            deadline = time.monotonic() + 5
            while not lock.is_dir() and first.poll() is None:
                if time.monotonic() >= deadline:
                    first.kill()
                    self.fail("first release publisher did not acquire its lock")
                time.sleep(0.01)
            owner = lock / "owner"
            deadline = time.monotonic() + 5
            while (
                not owner.is_file()
                or "snapshot_scope=pending" in owner.read_text(encoding="utf-8")
            ) and first.poll() is None:
                if time.monotonic() >= deadline:
                    first.kill()
                    self.fail("first release publisher did not seal its snapshot")
                time.sleep(0.01)

            # Exercise the ABA case while the gate is sleeping.  The published
            # receipt must remain bound to the already-captured A snapshot.
            marker = root / "release_marker.txt"
            marker.write_text("B\n", encoding="utf-8")
            marker.write_text("A\n", encoding="utf-8")

            second = subprocess.run(
                command,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            first_stdout, first_stderr = first.communicate(timeout=10)
            self.assertEqual(
                second.returncode,
                75,
                msg=f"stdout={second.stdout}\nstderr={second.stderr}",
            )
            self.assertIn("release acceptance already running", second.stderr)
            self.assertIn("pid=", second.stderr)
            self.assertIn("Confirm the recorded PID", second.stderr)
            self.assertEqual(
                first.returncode,
                0,
                msg=f"stdout={first_stdout}\nstderr={first_stderr}",
            )
            self.assertFalse(lock.exists())
            wheel = root / "dist/teaching_skill_miner-1.2.0-py3-none-any.whl"
            receipt = json.loads(
                (root / "artifacts/release_acceptance_1.2.0.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                receipt["release_artifact"]["sha256"],
                sha256(wheel.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                receipt["verification_binding"]["verification_scope"]["sha256"],
                sha256(b"A\n").hexdigest(),
            )

    def test_release_finalizer_leaves_pending_receipt_after_snapshot_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release_wrapper_fixture(root)
            acceptance = root / "artifacts/release_acceptance_1.2.0.json"
            acceptance.write_text('{"overall_status":"passed"}\n', encoding="utf-8")
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHON": sys.executable,
                    "TSM_RELEASE_TEST_FAIL": "true",
                }
            )
            result = subprocess.run(
                ["sh", str(root / "scripts/build_release_acceptance.sh")],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 9)
            pending = json.loads(acceptance.read_text(encoding="utf-8"))
            self.assertEqual(pending["overall_status"], "stale_not_accepted")
            self.assertIs(pending["engineering_release_accepted"], False)
            self.assertFalse((root / "artifacts/.release-acceptance.lock").exists())

    def test_release_finalizer_keeps_pending_when_final_prepublication_gate_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release_wrapper_fixture(root)
            acceptance = root / "artifacts/release_acceptance_1.2.0.json"
            acceptance.write_text('{"overall_status":"passed"}\n', encoding="utf-8")
            result = subprocess.run(
                ["sh", str(root / "scripts/build_release_acceptance.sh")],
                cwd=root,
                env={
                    **os.environ,
                    "PYTHON": sys.executable,
                    "TSM_RELEASE_TEST_FINAL_VERIFY_FAIL": "true",
                },
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 11)
            pending = json.loads(acceptance.read_text(encoding="utf-8"))
            self.assertEqual(pending["overall_status"], "stale_not_accepted")
            self.assertIs(pending["engineering_release_accepted"], False)
            self.assertEqual(list((root / "dist").glob("*.whl")), [])

    def test_release_finalizer_rejects_version_change_before_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release_wrapper_fixture(root)
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHON": sys.executable,
                    "TSM_RELEASE_TEST_PRE_SNAPSHOT_HOLD_SECONDS": "1",
                }
            )
            process = subprocess.Popen(
                ["sh", str(root / "scripts/build_release_acceptance.sh")],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            pending_path = root / "artifacts/release_acceptance_1.2.0.json"
            deadline = time.monotonic() + 5
            while not pending_path.is_file() and process.poll() is None:
                if time.monotonic() >= deadline:
                    process.kill()
                    self.fail("release publisher did not invalidate the old receipt")
                time.sleep(0.01)
            (root / "teaching_skill_miner/__init__.py").write_text(
                '__version__ = "1.2.1"\n', encoding="utf-8"
            )
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(
                process.returncode, 2, msg=f"stdout={stdout}\nstderr={stderr}"
            )
            self.assertIn("release version changed", stderr)
            receipt = json.loads(pending_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["overall_status"], "stale_not_accepted")
            self.assertEqual(list((root / "dist").glob("*.whl")), [])

    def test_release_finalizer_rejects_noncanonical_custom_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release_wrapper_fixture(root)
            alternate = root / "alternate"
            alternate.mkdir()
            result = subprocess.run(
                [
                    "sh",
                    str(root / "scripts/build_release_acceptance.sh"),
                    "--output",
                    str(alternate / "release_acceptance_1.2.0.json"),
                ],
                cwd=root,
                env={**os.environ, "PYTHON": sys.executable},
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("canonical repository path", result.stderr)
            self.assertFalse((root / "artifacts/.release-acceptance.lock").exists())

    def test_project_verification_rejects_source_changed_after_checks_started(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            initial_scope = self._scope_receipt(root)
            (root / "README.md").write_text(
                "# Changed during checks\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                AcceptanceError,
                "verification source scope changed while project checks were running",
            ):
                record_project(
                    argparse.Namespace(
                        repository_root=root,
                        junit_xml=paths["junit"],
                        first_wheel=paths["wheel"],
                        second_wheel=paths["second_wheel"],
                        exact_wheel_receipt=root / "unused-exact.json",
                        public_directory=paths["public"],
                        public_release_audit=paths["public_audit"],
                        expected_verification_scope=initial_scope,
                        formal_caption_audit_run=False,
                        teachobs_private_receipt_audit_run=False,
                        tracked_file_privacy_scan_run=False,
                        output=root / "must-not-exist.json",
                    )
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
