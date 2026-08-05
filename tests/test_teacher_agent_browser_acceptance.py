from __future__ import annotations

from contextlib import redirect_stdout
from io import BytesIO
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.run_teacher_agent_browser_acceptance import (
    RECEIPT_SCHEMA,
    _Telemetry,
    _image_answer_png,
    _receipt,
    _request_within_capability,
    _validated_base_url,
    main,
    run_browser_acceptance,
)
from teaching_skill_miner.io_utils import project_root


class TeacherAgentBrowserAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.safe_url = "http://127.0.0.1:8765/browseracceptancecapability01/"

    def test_loopback_capability_validation_is_exact(self) -> None:
        self.assertEqual(_validated_base_url(self.safe_url), self.safe_url)
        ipv6 = "http://[::1]:8765/browseracceptancecapability01/"
        self.assertEqual(_validated_base_url(ipv6), ipv6)

        invalid = (
            "https://127.0.0.1:8765/browseracceptancecapability01/",
            "http://localhost:8765/browseracceptancecapability01/",
            "http://127.0.0.1/browseracceptancecapability01/",
            "http://127.0.0.1:8765/too-short/",
            "http://127.0.0.1:8765/browseracceptancecapability01/extra/",
            "http://127.0.0.1:8765/browseracceptancecapability01/?debug=1",
            "http://user@127.0.0.1:8765/browseracceptancecapability01/",
            "https://example.invalid/browseracceptancecapability01/",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(Exception):
                    _validated_base_url(value)

    def test_browser_request_guard_stays_under_one_capability(self) -> None:
        allowed = (
            self.safe_url,
            self.safe_url + "api/bootstrap",
            self.safe_url + "assets/teacher_agent_demo.css",
            self.safe_url + "assets/student-xiaoyu.png?v=1",
        )
        for value in allowed:
            with self.subTest(value=value):
                self.assertTrue(_request_within_capability(self.safe_url, value))

        blocked = (
            "http://127.0.0.1:8766/browseracceptancecapability01/api/bootstrap",
            "http://127.0.0.1:8765/anothercapability00000001/api/bootstrap",
            self.safe_url + "../outside",
            self.safe_url + "%2e%2e/outside",
            "https://example.invalid/asset.js",
        )
        for value in blocked:
            with self.subTest(value=value):
                self.assertFalse(_request_within_capability(self.safe_url, value))

    def test_missing_playwright_is_an_explicit_dependency_exit(self) -> None:
        with patch(
            "scripts.run_teacher_agent_browser_acceptance._load_playwright",
            side_effect=ModuleNotFoundError("playwright"),
        ):
            receipt = run_browser_acceptance(
                self.safe_url,
                acknowledge_remote_demo_text=True,
            )
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["failure"]["stage"], "dependency")
        self.assertEqual(
            receipt["failure"]["code"],
            "python_playwright_not_installed",
        )
        self.assertFalse(receipt["browser"]["launched"])

        stdout = io.StringIO()
        with (
            patch(
                "scripts.run_teacher_agent_browser_acceptance._load_playwright",
                side_effect=ModuleNotFoundError("playwright"),
            ),
            redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--base-url",
                    self.safe_url,
                    "--acknowledge-remote-demo-text",
                ]
            )
        self.assertEqual(exit_code, 3)
        parsed = json.loads(stdout.getvalue())
        self.assertEqual(parsed["schema"], RECEIPT_SCHEMA)
        self.assertNotIn(self.safe_url, stdout.getvalue())

    def test_consent_and_url_rejection_happen_before_playwright_load(self) -> None:
        with patch(
            "scripts.run_teacher_agent_browser_acceptance._load_playwright"
        ) as loader:
            missing_consent = run_browser_acceptance(
                self.safe_url,
                acknowledge_remote_demo_text=False,
            )
            invalid_target = run_browser_acceptance(
                "https://example.invalid/not-a-capability/",
                acknowledge_remote_demo_text=True,
            )
        loader.assert_not_called()
        self.assertEqual(missing_consent["failure"]["stage"], "consent")
        self.assertEqual(invalid_target["failure"]["stage"], "input")
        self.assertEqual(invalid_target["counts"]["start_requests"], 0)
        self.assertNotIn(
            "example.invalid", json.dumps(invalid_target, ensure_ascii=False)
        )

    def test_receipt_is_aggregate_and_optional_skip_can_pass(self) -> None:
        telemetry = _Telemetry(
            start_requests=2,
            replacement_start_requests=1,
            attachment_requests=1,
            step_requests=2,
            image_step_requests=1,
            overflow_by_viewport={390: True, 768: True, 1440: True},
        )
        receipt = _receipt(
            checks={
                "profile_a_turn_committed": True,
                "profile_b_replacement_started": True,
            },
            optional_checks={
                "manual_skill_not_inherited": "skipped_provider_unavailable"
            },
            telemetry=telemetry,
            browser_name="chromium",
            headless=True,
            duration_ms=125,
            browser_launched=True,
        )
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["counts"]["profiles_exercised"], 2)
        self.assertEqual(receipt["counts"]["viewports_checked"], 3)
        self.assertEqual(receipt["counts"]["attachment_requests"], 1)
        self.assertEqual(receipt["counts"]["image_step_requests"], 1)
        serialized = json.dumps(receipt, ensure_ascii=False)
        for forbidden in (
            self.safe_url,
            "session_id",
            "profile_ref",
            "learner_response",
            "Recursion is a prerequisite concept.",
            "OCR：Recursion",
            "synthetic console error detail",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertFalse(receipt["privacy"]["raw_image_bytes_emitted"])
        self.assertFalse(receipt["privacy"]["ocr_text_emitted"])

        failed = _receipt(
            checks={"required": True},
            optional_checks={"manual_skill_not_inherited": "failed"},
            telemetry=_Telemetry(),
            browser_name="chrome",
            headless=False,
            duration_ms=0,
            browser_launched=True,
        )
        self.assertFalse(failed["passed"])

    def test_image_answer_fixture_is_a_memory_only_clear_png(self) -> None:
        payload = _image_answer_png()
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        with Image.open(BytesIO(payload)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.mode, "RGB")
            self.assertGreaterEqual(image.width, 1200)
            self.assertGreaterEqual(image.height, 200)

    def test_source_contains_real_browser_lifecycle_and_no_artifact_capture(
        self,
    ) -> None:
        source = (
            Path(project_root()) / "scripts/run_teacher_agent_browser_acceptance.py"
        ).read_text(encoding="utf-8")
        for required in (
            "page.goto(",
            "page.reload(",
            'page.on("console"',
            'page.on("pageerror"',
            "page.set_viewport_size(",
            "sessionStorage.getItem(",
            '"#startButton"',
            '"#stepButton"',
            "\"[data-profile-id='zimo']\"",
            '"#initialConceptual"',
            "\"[data-profile-id='zimo'] .profile-mini-ring\"",
            '"#evaluationViewButton"',
            "replacement_request_had_manual_skill",
            "replacement_request_had_version_guards",
            "replacement_request_had_custom_mastery",
            'checks["profile_b_mastery_ring_updated_live"]',
            'checks["profile_b_custom_mastery_materialized"]',
            'checks["profile_a_image_only_turn_committed"]',
            'checks["profile_a_image_attachment_uploaded"]',
            'checks["profile_a_image_step_bound"]',
            'checks["profile_a_visual_evidence_summary_rendered"]',
            'checks["profile_a_raw_image_not_sent"]',
            'checks["draft_edit_kept_active_goal_titles"]',
            'checks["draft_cancel_preserved_session_identity"]',
            'checks["draft_cancel_emitted_no_request"]',
            'checks["draft_cancel_restored_active_composer"]',
            'checks["external_session_advanced_before_replacement"]',
            'checks["stale_replacement_retried_exactly_once"]',
            'checks["stale_replacement_rejection_observed"]',
            'checks["stale_replacement_used_synchronized_guards"]',
            'checks["stale_replacement_used_fresh_idempotency_key"]',
            "_advance_active_session_outside_ui",
            "replacement_guard_snapshots",
            "replacement_idempotency_keys",
            "expected_stale_replacement_rejections",
            'page.on("response"',
            "取消草稿不应成为活动目标",
            '"replace_expected_profile_revision"',
            '"#answerImageInput"',
            "set_input_files(",
            '"api/attachment"',
            '"attachment_ids"',
            "Recursion is a prerequisite concept.",
            "local_ocr_unavailable",
            "原图未发送",
            'responsive_sidebar_modal_',
            'responsive_inspector_modal_',
            'responsive_desktop_inert_cleared_',
            "sidebar_drawer_modal_contract_failed",
            "inspector_drawer_modal_contract_failed",
            "desktop_drawer_inert_not_cleared",
            "document.activeElement?.id === 'sidebarClose'",
            "document.activeElement?.id === 'inspectorClose'",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        for forbidden in (
            "page.screenshot",
            "context.tracing",
            "print(base_url",
            "print(safe_base_url",
            "str(exc)",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
