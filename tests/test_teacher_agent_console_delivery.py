from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TeacherAgentConsoleDeliveryTests(unittest.TestCase):
    def test_launcher_lifecycle_and_shell_syntax(self) -> None:
        # The lifecycle probe must remain hermetic: Python-only package checks
        # do not install apps/console/node_modules before invoking it.
        with tempfile.TemporaryDirectory(prefix="teachlab-launcher-self-test-") as tmp:
            isolated_root = pathlib.Path(tmp)
            isolated_scripts = isolated_root / "scripts"
            isolated_scripts.mkdir()
            isolated_launcher = isolated_scripts / "start_teacher_agent_console.mjs"
            shutil.copy2(
                ROOT / "scripts/start_teacher_agent_console.mjs", isolated_launcher
            )
            subprocess.run(
                ["node", str(isolated_launcher), "--lifecycle-self-test"],
                cwd=isolated_root,
                check=True,
                capture_output=True,
                text=True,
            )
        runtime_builder = (
            ROOT / "scripts/build_teacher_agent_console_runtime.mjs"
        ).read_text(encoding="utf-8")
        self.assertIn("copyBuildInput", runtime_builder)
        self.assertIn("fs.chmodSync(target, 0o600)", runtime_builder)
        self.assertIn("read-only snapshot input", runtime_builder)
        command_path = ROOT / "打开题目二教学Agent.command"
        command_source = command_path.read_text(encoding="utf-8")
        self.assertTrue(command_source.startswith("#!/bin/zsh\n"))
        self.assertTrue(os.access(command_path, os.X_OK))
        zsh = shutil.which("zsh")
        if zsh is not None:
            subprocess.run([zsh, "-n", str(command_path)], check=True)
        subprocess.run(
            [
                "node",
                str(ROOT / "scripts/build_teacher_agent_console_runtime.mjs"),
                "--self-test",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "node",
                str(ROOT / "scripts/package_teacher_agent_macos.mjs"),
                "--embedded-runtime-self-test",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "python",
                str(ROOT / "scripts/run_teacher_agent_console_cross_browser_smoke.py"),
                "--self-test",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_launcher_uses_production_runtime_and_strict_readiness(self) -> None:
        source = (ROOT / "scripts/start_teacher_agent_console.mjs").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('["run", "dev"', source)
        self.assertIn("`${url}/health`", source)
        self.assertIn("`${url}/ready`", source)
        self.assertIn("ready.status === 200", source)
        self.assertIn("verifyProductionRuntime", source)
        self.assertIn("TEACHLAB_RUNTIME_STOP_REQUEST", source)
        self.assertIn("TEACHLAB_IDLE_SHUTDOWN_MINUTES", source)
        self.assertIn("TEACHLAB_RESOURCE_REVIEW_STORE", source)
        self.assertIn('"--resource-review-store", resourceReviewStore', source)

    def test_double_click_secret_and_browser_contract(self) -> None:
        """Keep the user-facing macOS entry point aligned with launcher safeguards."""
        launcher = (ROOT / "scripts/start_teacher_agent_console.mjs").read_text(
            encoding="utf-8"
        )
        command = (ROOT / "打开题目二教学Agent.command").read_text(encoding="utf-8")
        self.assertIn('TEACHLAB_OPEN_BROWSER:=1', command)
        self.assertIn('export TEACHLAB_OPEN_BROWSER', command)
        self.assertIn('TEACHLAB_OPEN_BROWSER=0', command)
        self.assertIn('allowSymlinkFallback: true', launcher)
        self.assertEqual(
            launcher.count('allowSymlinkFallback: true'),
            1,
            "only the DeepSeek API-key resolver may accept a symlink fallback",
        )
        self.assertIn('O_NOFOLLOW', launcher)
        self.assertIn('opened.dev !== inspected.target.dev', launcher)
        self.assertIn('opened.ino !== inspected.target.ino', launcher)
        self.assertIn('fs.openSync(target, "wx", 0o600)', launcher)
        self.assertIn('cleanupMaterializedSecrets()', launcher)

        runtime_doc = (
            ROOT / "docs/teacher_agent_console_production_runtime.md"
        ).read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("TEACHLAB_OPEN_BROWSER=0", runtime_doc)
        self.assertIn("owner-only", runtime_doc)
        self.assertIn("runtime-secrets", runtime_doc)
        self.assertIn("TEACHLAB_OPEN_BROWSER=0", readme)
        self.assertIn("owner-only", readme)

    def test_cross_browser_smoke_is_production_only_and_avoids_dev_port(self) -> None:
        source = (
            ROOT / "scripts/run_teacher_agent_console_cross_browser_smoke.py"
        ).read_text(encoding="utf-8")
        self.assertIn('SUPPORTED_BROWSERS = ("chromium", "firefox", "webkit")', source)
        self.assertIn('"NODE_ENV": "production"', source)
        self.assertIn('"TEACHLAB_RELEASE_ID": runtime.release_id', source)
        self.assertIn('runtime.directory / "server.js"', source)
        self.assertIn("if port != 3030", source)
        self.assertNotIn("next dev", source.lower())
        self.assertIn("session_cookie_attributes_failed", source)
        self.assertIn("critical_a11y_structure", source)
        self.assertIn("globalThis.axe.run(document", source)
        self.assertIn("axe_serious_or_critical_", source)
        self.assertIn("offline_scope_bound_shell", source)
        self.assertIn("equivalent_400_percent", source)
        self.assertIn("service_worker_not_controlling", source)
        self.assertNotIn("static_a11y_passed", source)

    def test_packager_only_claims_a_pinned_self_contained_notarized_artifact(
        self,
    ) -> None:
        source = (ROOT / "scripts/package_teacher_agent_macos.mjs").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "distributable: Boolean(releaseRequested && signed && notarized && embeddedNode && embeddedPython)",
            source,
        )
        self.assertIn("unsigned_developer_artifact", source)
        self.assertIn("signed_notarized_self_contained_release", source)
        self.assertIn("notarytool", source)
        self.assertIn("rollback_policy", source)
        self.assertIn("TEACHLAB_EMBEDDED_NODE", source)
        self.assertIn("TEACHLAB_EMBEDDED_NODE_MANIFEST_SHA256", source)
        self.assertIn("TEACHLAB_EMBEDDED_PYTHON_MANIFEST_SHA256", source)
        self.assertIn("runtimeTreeSha256(destination)", source)
        self.assertIn("manifest.self_contained !== true", source)
        self.assertIn("copyPublishedRuntimeResources", source)
        self.assertIn("embedded-python-app-smoke-ok", source)
        self.assertIn("TEACHLAB_CONSOLE_RUNTIME_ROOT: runtimeRoot", source)
        self.assertIn("required_external_runtime: releaseRequested ? null", source)
        self.assertIn('"--runtime-self-test"', source)
        subprocess.run(
            ["node", "--check", str(ROOT / "scripts/package_teacher_agent_macos.mjs")],
            check=True,
        )
        refused = subprocess.run(
            ["node", str(ROOT / "scripts/package_teacher_agent_macos.mjs")],
            cwd=ROOT,
            env={**os.environ, "TEACHLAB_DISTRIBUTABLE": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("requires codesign identity", refused.stderr)


if __name__ == "__main__":
    unittest.main()
