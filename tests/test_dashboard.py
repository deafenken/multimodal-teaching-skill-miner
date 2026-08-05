from __future__ import annotations

from contextlib import redirect_stdout
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from teaching_skill_miner.cli import main
from teaching_skill_miner.dashboard import (
    dashboard_html_bytes,
    dashboard_self_check,
    materialize_dashboard,
    open_dashboard,
)
from teaching_skill_miner.io_utils import project_root


class _DashboardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.screen_labels: list[str] = []
        self.external_assets: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = dict(attrs)
        label = values.get("data-screen-label")
        if label:
            self.screen_labels.append(label)
        for name in ("src", "href"):
            value = values.get(name)
            if value and value.startswith(("http://", "https://", "//")):
                self.external_assets.append(value)


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = dashboard_html_bytes()
        self.html = self.payload.decode("utf-8")

    def test_dashboard_is_single_file_public_aggregate_ui(self) -> None:
        report = dashboard_self_check()
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["release_audit_passed"], report)
        self.assertEqual(report["release_audit_finding_count"], 0)
        self.assertEqual(report["release_audit_findings"], [])
        self.assertEqual(report["embedded_media_matches"], [])
        self.assertEqual(report["row_level_marker_matches"], [])
        self.assertGreater(report["size_bytes"], 20_000)
        self.assertEqual(len(report["sha256"]), 64)
        parser = _DashboardParser()
        parser.feed(self.html)
        self.assertEqual(
            parser.screen_labels,
            [
                "01 Overview",
                "02 Multimodal timeline",
                "03 Evidence",
                "04 Skill runtime",
                "05 Claim boundaries",
            ],
        )
        self.assertEqual(parser.external_assets, [])
        self.assertNotIn("fetch(", self.html)
        for marker in (
            'id="main" tabindex="-1"',
            'aria-controls="tweaks" aria-expanded="false"',
            'id="eventStatus" aria-live="polite"',
            'role="dialog" aria-labelledby="tweaksTitle"',
            'button.setAttribute("aria-pressed", String(active))',
            'historyRoot.replaceChildren()',
        ):
            self.assertIn(marker, self.html)
        self.assertNotIn('id="classroomFrame" aria-live=', self.html)
        self.assertNotIn('qs("#runtimeHistory").innerHTML', self.html)

    def test_dashboard_embeds_current_public_aggregate_metrics(self) -> None:
        root = project_root()
        receipt = json.loads(
            (
                root
                / "artifacts/public/teachobs_multimodal_benchmark_receipt.json"
            ).read_text(encoding="utf-8")
        )
        for arm in receipt["aggregate_arms"].values():
            for metric in (
                "micro_f1",
                "macro_f1",
                "hamming_accuracy",
                "visual_macro_f1",
            ):
                rendered = f"{arm['metrics'][metric]:.6f}".lstrip("0")
                self.assertIn(rendered, self.html)
        materialization = json.loads(
            (
                root
                / "artifacts/public/teachobs_transcript_materialization_receipt.json"
            ).read_text(encoding="utf-8")
        )
        aggregate = materialization["aggregate"]
        self.assertIn(f"{aggregate['lesson_count']}/30", self.html)
        self.assertIn(f"{aggregate['scene_count']:,}", self.html)
        self.assertIn(f"{aggregate['nonempty_scene_count']:,} transcript non-empty", self.html)
        source_counts = aggregate["source_tier_lesson_counts"]
        self.assertIn(
            f"{source_counts['platform_creator_provided_caption']} "
            "platform creator-provided",
            self.html,
        )
        self.assertIn(
            f"{source_counts['platform_automatic_caption']} automatic",
            self.html,
        )
        self.assertIn(
            f"{source_counts['audited_asr_fallback']} technically audited ASR",
            self.html,
        )

    def test_dashboard_preserves_claim_and_privacy_boundaries(self) -> None:
        for forbidden in (
            "artifacts/private",
            "data/real",
            "/Volumes/",
            "/data/winbeau_zhao",
        ):
            self.assertNotIn(forbidden, self.html)
        for required in (
            "engineering_ready_external_validation_pending",
            "0.9020 ≠ 部署 0.9",
            "ASR 内容准确率与 WER",
            "双人独立标注完成",
            "外部锁箱与部署准确率",
            "真实学习效果",
            "状态机执行完成，不证明学习效果",
            "确认性增益尚未建立",
            "post-selection exploratory",
        ):
            self.assertIn(required, self.html)

    def test_dashboard_cli_check_is_nonblocking(self) -> None:
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["dashboard", "--check"]), 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report["passed"])
        self.assertFalse(report["private_media_embedded"])
        self.assertFalse(report["row_level_private_data_embedded"])
        self.assertTrue(report["release_audit_passed"])

    def test_dashboard_self_check_derives_privacy_result_from_release_audit(
        self,
    ) -> None:
        contaminated = self.payload + b"\nparticipant_id\n"
        with patch(
            "teaching_skill_miner.dashboard.dashboard_html_bytes",
            return_value=contaminated,
        ):
            report = dashboard_self_check()
        self.assertFalse(report["passed"])
        self.assertTrue(report["release_audit_passed"])
        self.assertTrue(report["row_level_private_data_embedded"])
        self.assertEqual(report["row_level_marker_matches"], ["participant_id"])

        with patch(
            "teaching_skill_miner.dashboard.dashboard_html_bytes",
            return_value=self.payload + b"\ndata:image/png;base64,AAAA\n",
        ):
            media_report = dashboard_self_check()
        self.assertFalse(media_report["passed"])
        self.assertTrue(media_report["private_media_embedded"])
        self.assertEqual(media_report["embedded_media_matches"], ["data:image/"])

    def test_dashboard_materializes_exact_reviewed_html(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = materialize_dashboard(Path(directory) / "dashboard.html")
            self.assertEqual(target.read_bytes(), self.payload)

    def test_dashboard_default_materialization_uses_private_random_directory(
        self,
    ) -> None:
        target = materialize_dashboard()
        self.addCleanup(shutil.rmtree, target.parent)
        self.assertEqual(target.name, "index.html")
        self.assertTrue(target.parent.name.startswith("tsm-evidence-dashboard-"))
        self.assertEqual(target.parent.stat().st_mode & 0o077, 0)
        self.assertEqual(target.read_bytes(), self.payload)

    def test_dashboard_materialization_replaces_output_symlink_atomically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "sentinel.txt"
            sentinel.write_text("do not overwrite", encoding="utf-8")
            target = root / "dashboard.html"
            try:
                os.symlink(sentinel, target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            materialized = materialize_dashboard(target)
            self.assertEqual(materialized, target.resolve())
            self.assertFalse(target.is_symlink())
            self.assertEqual(target.read_bytes(), self.payload)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not overwrite")

    def test_dashboard_open_uses_file_uri_and_returns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dashboard.html"
            with patch(
                "teaching_skill_miner.dashboard.webbrowser.open",
                return_value=True,
            ) as browser, redirect_stdout(io.StringIO()) as output:
                self.assertEqual(open_dashboard(output=target), 0)
            browser.assert_called_once_with(target.resolve().as_uri())
            report = json.loads(output.getvalue())
            self.assertEqual(report["dashboard_path"], str(target.resolve()))
            self.assertTrue(report["browser_open_result"])


if __name__ == "__main__":
    unittest.main()
