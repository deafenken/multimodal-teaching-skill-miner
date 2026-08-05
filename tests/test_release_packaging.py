from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from teaching_skill_miner.cli import main
from teaching_skill_miner.release_audit import audit_release_path
from scripts.verify_wheel_allowlist import (
    BUNDLED_TRANSCRIPTS,
    DIST_INFO_FILES,
    GOVERNANCE_FILES,
    PACKAGE_RESOURCE_FILES,
    PUBLIC_JSON_ALLOWLIST,
    PUBLIC_DATA_FILES,
    _expected_payloads,
    verify_public_json_resource_allowlist,
    verify_wheel_allowlist,
)


class ReleasePackagingTests(unittest.TestCase):
    def _fixture_repository(self, root: Path) -> None:
        project = Path(__file__).resolve().parents[1]
        files: dict[str, str | bytes] = {
            "teaching_skill_miner/__init__.py": '__version__ = "1.2.0"\n',
            "LICENSE": "test-only license\n",
            "schema/example.json": "{}\n",
            "configs/example.json": "{}\n",
            PUBLIC_JSON_ALLOWLIST.as_posix(): (
                "configs/example.json\nschema/example.json\n"
            ),
        }
        files.update({
            name: (
                (project / name).read_bytes()
                if Path(name).suffix == ".png"
                else "<!doctype html><title>fixture</title>\n"
            )
            for name in PACKAGE_RESOURCE_FILES
        })
        files.update({f"data/{name}": "{}\n" for name in PUBLIC_DATA_FILES})
        files.update(
            {
                f"data/transcripts/{name}": '{"fixture": true}\n'
                for name in BUNDLED_TRANSCRIPTS
            }
        )
        files.update({name: f"public {name}\n" for name in GOVERNANCE_FILES})
        for relative, payload in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, bytes):
                path.write_bytes(payload)
            else:
                path.write_text(payload, encoding="utf-8")

    def _fixture_wheel(
        self,
        root: Path,
        *,
        unexpected_member: tuple[str, bytes] | None = None,
        corrupt_member: str | None = None,
    ) -> Path:
        distribution_stem = "teaching_skill_miner-1.2.0"
        dist_info = f"{distribution_stem}.dist-info"
        expected, errors = _expected_payloads(
            root,
            distribution_stem,
            dist_info,
        )
        self.assertEqual(errors, [])
        wheel = root / f"{distribution_stem}-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for member, source in expected.items():
                payload = source.read_bytes()
                if member == corrupt_member:
                    payload = b"not-the-public-source\n"
                archive.writestr(member, payload)
            for filename in DIST_INFO_FILES:
                archive.writestr(f"{dist_info}/{filename}", "generated\n")
            if unexpected_member:
                archive.writestr(*unexpected_member)
        return wheel

    def test_exact_wheel_allowlist_accepts_only_hash_matched_public_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture_repository(root)
            wheel = self._fixture_wheel(root)
            report = verify_wheel_allowlist(wheel, root)
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["unexpected_members"], [])
            self.assertEqual(report["content_mismatches"], [])

    def test_exact_wheel_allowlist_rejects_private_or_changed_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture_repository(root)
            private_wheel = self._fixture_wheel(
                root,
                unexpected_member=(
                    "artifacts/private/formal_captions/transcript.json",
                    b'{"caption": "private full text"}\n',
                ),
            )
            private_report = verify_wheel_allowlist(private_wheel, root)
            self.assertFalse(private_report["passed"])
            self.assertEqual(
                private_report["unexpected_members"],
                ["artifacts/private/formal_captions/transcript.json"],
            )

            transcript_member = (
                "teaching_skill_miner-1.2.0.data/data/share/"
                "teaching-skill-miner/data/transcripts/linear_algebra_l01.json"
            )
            changed_wheel = self._fixture_wheel(
                root,
                corrupt_member=transcript_member,
            )
            changed_report = verify_wheel_allowlist(changed_wheel, root)
            self.assertFalse(changed_report["passed"])
            self.assertEqual(
                [item["member"] for item in changed_report["content_mismatches"]],
                [transcript_member],
            )

    def test_unreviewed_json_fails_build_gate_and_wheel_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture_repository(root)
            wheel = self._fixture_wheel(root)
            unreviewed = root / "schema/unreviewed_private_payload.json"
            unreviewed.write_text('{"must_not_ship": true}\n', encoding="utf-8")

            source_report = verify_public_json_resource_allowlist(root)
            self.assertFalse(source_report["passed"])
            self.assertEqual(
                source_report["unlisted_resources"],
                ["schema/unreviewed_private_payload.json"],
            )

            wheel_report = verify_wheel_allowlist(wheel, root)
            self.assertFalse(wheel_report["passed"])
            self.assertTrue(
                any(
                    "unreviewed public JSON resources" in error
                    for error in wheel_report["errors"]
                ),
                wheel_report,
            )

            project_root = Path(__file__).resolve().parents[1]
            scripts = root / "scripts"
            scripts.mkdir()
            build_script = scripts / "build_release_wheel.sh"
            verifier = scripts / "verify_wheel_allowlist.py"
            build_script.write_bytes(
                (project_root / "scripts/build_release_wheel.sh").read_bytes()
            )
            verifier.write_bytes(
                (project_root / "scripts/verify_wheel_allowlist.py").read_bytes()
            )
            completed = subprocess.run(
                ["sh", str(build_script), str(root / "dist")],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn(
                "unreviewed public JSON resources",
                completed.stdout + completed.stderr,
            )

    def test_release_audit_rejects_private_caption_and_model_weight_members(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "unsafe.whl"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "artifacts/private/formal_captions/transcript.json",
                    "{}\n",
                )
                archive.writestr(
                    "models/openai_clip/model.safetensors",
                    b"not-a-real-weight",
                )
            report = audit_release_path(archive_path)
            rules = {item["rule"] for item in report["findings"]}
            self.assertFalse(report["passed"])
            self.assertIn("forbidden_private_path", rules)
            self.assertIn("forbidden_binary_or_archive", rules)

    def test_packaging_metadata_disables_implicit_data_and_wildcard_transcripts(
        self,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
        build_script = (project_root / "scripts/build_release_wheel.sh").read_text(
            encoding="utf-8"
        )
        dashboard_script = (project_root / "scripts/run_dashboard.sh").read_text(
            encoding="utf-8"
        )
        allowlist_script = (
            project_root / "scripts/verify_wheel_allowlist.py"
        ).read_text(encoding="utf-8")
        self.assertIn("include-package-data = false", pyproject)
        self.assertNotIn('"data/transcripts/*.json"', pyproject)
        self.assertNotIn("data/transcripts/*.json", build_script)
        self.assertNotIn('"schema/*.json"', pyproject)
        self.assertNotIn('"configs/*.json"', pyproject)
        self.assertNotIn('find "$repo_root/$public_directory"', build_script)
        self.assertIn('"private_demo.html"', pyproject)
        self.assertIn('"private_skill_demo.css"', pyproject)
        self.assertIn('"private_skill_demo.js"', pyproject)
        for resource in PACKAGE_RESOURCE_FILES:
            self.assertIn(resource, build_script)
            package_resource = Path(resource).relative_to(
                "teaching_skill_miner/web"
            )
            self.assertIn(f'"{package_resource.as_posix()}"', pyproject)
        for filename in PUBLIC_DATA_FILES:
            self.assertIn(f"data/{filename}", pyproject)
            self.assertIn(f"data/{filename}", build_script)
        self.assertIn("python_command=${PYTHON:-python3}", dashboard_script)
        self.assertIn(
            '"$python_command" -m teaching_skill_miner dashboard', dashboard_script
        )
        self.assertIn(
            '"package_resource_files": list(PACKAGE_RESOURCE_FILES)',
            allowlist_script,
        )
        for filename in BUNDLED_TRANSCRIPTS:
            self.assertIn(f"data/transcripts/{filename}", pyproject)
            self.assertIn(f"data/transcripts/{filename}", build_script)
        reviewed = (
            (project_root / PUBLIC_JSON_ALLOWLIST)
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertEqual(reviewed, sorted(set(reviewed)))
        for relative in reviewed:
            self.assertIn(f'"{relative}"', pyproject)
        self.assertIn("--check-public-json-source-only", build_script)
        self.assertIn(PUBLIC_JSON_ALLOWLIST.as_posix(), build_script)

    def test_release_audit_rejects_complete_subtitle_tracks(self) -> None:
        subtitle_suffixes = (".vtt", ".srt", ".ass", ".ssa", ".ttml")
        for suffix in subtitle_suffixes:
            with (
                self.subTest(suffix=suffix),
                tempfile.TemporaryDirectory() as directory,
            ):
                subtitle = Path(directory) / f"private-caption{suffix}"
                subtitle.write_text(
                    "WEBVTT\n\n00:00.000 --> 00:01.000\nprivate classroom text\n",
                    encoding="utf-8",
                )
                report = audit_release_path(subtitle)
                self.assertFalse(report["passed"], report)
                self.assertIn(
                    "forbidden_binary_or_archive",
                    {item["rule"] for item in report["findings"]},
                )

    def test_new_wheel_cli_entrypoints_have_safe_help_paths(self) -> None:
        for command in (
            "fetch-full-videos",
            "fetch-teachobs",
            "benchmark-teachobs-text",
            "audit-teachobs-captions",
            "fetch-teachobs-captions",
            "prepare-teachobs-asr-handoff",
            "import-teachobs-asr-results",
            "materialize-teachobs-transcripts",
            "prepare-teachobs-double-annotation",
            "analyze-teachobs-double-annotation",
            "prepare-teachobs-media",
            "benchmark-teachobs-multimodal",
            "prepare-teachobs-lockbox-preregistration",
            "prepare-learner-effect-study",
            "analyze-learner-effect-study",
            "multimodal-longform-dataset",
            "multimodal-ablation",
            "visual-semantic-extract",
            "visual-semantic-dataset",
            "visual-semantic-apply",
            "dashboard",
            "teacher-agent-start",
            "teacher-agent-step",
            "teacher-agent-evaluate",
            "teacher-agent-benchmark",
            "teacher-agent-outcome-evaluate",
            "teacher-agent-demo",
            "teacher-agent-dashboard",
        ):
            with (
                self.subTest(command=command),
                redirect_stdout(io.StringIO()) as output,
            ):
                with self.assertRaises(SystemExit) as raised:
                    main([command, "--help"])
                self.assertEqual(raised.exception.code, 0)
                self.assertIn("usage:", output.getvalue())

        with redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as raised:
                main(["benchmark-teachobs-multimodal", "--help"])
            self.assertEqual(raised.exception.code, 0)
            help_text = output.getvalue()
            self.assertIn("--evaluation-profile", help_text)
            self.assertIn("full_23_train_7_test", help_text)
            self.assertIn("paper_track1_23_train_6_test", help_text)
            self.assertIn("--transcript-materialization-manifest", help_text)

    def test_materialize_teachobs_cli_maps_every_required_private_input(
        self,
    ) -> None:
        manifest = {
            "profile_id": "paper_track1_23_train_6_test",
            "aggregate": {"lesson_count": 29, "scene_count": 4945},
            "materialization_fingerprint_sha256": "a" * 64,
        }
        paths = {
            "media_plan": "media-plan.json",
            "media_manifest": "media-manifest.json",
            "caption_audit": "caption-audit.json",
            "asr_import_audit": "import-audit.json",
            "coverage_matrix": "coverage.json",
            "asr_job_manifest": "jobs.json",
            "asr_results": "results",
            "output": "materialized",
            "public_receipt": "receipt.json",
        }
        arguments = ["materialize-teachobs-transcripts"]
        for name, value in paths.items():
            arguments.extend(("--" + name.replace("_", "-"), value))
        with (
            patch(
                "teaching_skill_miner.cli.materialize_teachobs_transcripts",
                return_value={
                    "manifest_path": "materialized/manifest.json",
                    "public_receipt_path": "receipt.json",
                    "manifest": manifest,
                },
            ) as mocked,
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(main(arguments), 0)
        mocked.assert_called_once_with(
            paths["media_plan"],
            paths["media_manifest"],
            paths["caption_audit"],
            paths["asr_import_audit"],
            paths["coverage_matrix"],
            paths["asr_job_manifest"],
            paths["asr_results"],
            paths["output"],
            public_receipt_path=paths["public_receipt"],
        )
        rendered = output.getvalue()
        self.assertIn('"lesson_count": 29', rendered)
        self.assertIn('"scene_count": 4945', rendered)
        self.assertIn('"content_accuracy_established": false', rendered)
        self.assertIn('"word_error_rate_established": false', rendered)


if __name__ == "__main__":
    unittest.main()
