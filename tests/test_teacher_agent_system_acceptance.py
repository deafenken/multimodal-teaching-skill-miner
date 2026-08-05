from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from scripts.run_teacher_agent_system_acceptance import (
    _bootstrap_contract_is_supported,
    _start_payload,
    main,
    run_system_acceptance,
)
from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.teacher_agent_dashboard import (
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
)


class TeacherAgentSystemAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = project_root()
        cls.safe_unused_url = (
            "http://127.0.0.1:9/systemacceptancecapability01/"
        )

    def test_real_http_lifecycle_and_private_aggregate_receipt(self) -> None:
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.root / "data/teacher_agent_skill_library.json",
            self.root / "data/teacher_agent_demo_input.json",
            self.root / "data/teacher_agent_evaluation_cases.json",
        )
        try:
            safe_capability = "a" * 24
            server, url = create_teacher_agent_dashboard_server(
                snapshot,
                capability_token=safe_capability,
            )
        except PermissionError:
            self.skipTest("sandbox does not permit loopback socket binding")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        output = io.StringIO()
        try:
            with patch("webbrowser.open") as browser_open, redirect_stdout(output):
                exit_code = main(
                    [
                        "--base-url",
                        url,
                        "--acknowledge-remote-demo-text",
                        "--timeout-seconds",
                        "10",
                    ]
                )
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

        self.assertEqual(exit_code, 0)
        browser_open.assert_not_called()
        receipt = json.loads(output.getvalue())
        self.assertTrue(receipt["passed"])
        self.assertTrue(all(receipt["checks"].values()))
        self.assertEqual(receipt["counts"]["sessions_started"], 3)
        self.assertEqual(receipt["counts"]["turns_committed"], 1)
        self.assertEqual(receipt["counts"]["expected_rejections_observed"], 4)
        self.assertEqual(receipt["counts"]["http_rejections"], 4)
        self.assertGreaterEqual(receipt["counts"]["http_requests"], 14)
        self.assertTrue(
            receipt["checks"]["stale_profile_handle_retried_as_fresh_session"]
        )

        serialized = output.getvalue()
        capability_token = urlsplit(url).path.strip("/")
        for private_value in (
            url,
            capability_token,
            "状态表示子问题的答案",
            "这条输入必须被过期上下文保护拒绝",
        ):
            self.assertNotIn(private_value, serialized)
        self.assertNotIn("session_id", serialized)
        self.assertNotIn("profile_ref", serialized)

    def test_runner_accepts_the_dashboard_bootstrap_contract_without_a_socket(
        self,
    ) -> None:
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.root / "data/teacher_agent_skill_library.json",
            self.root / "data/teacher_agent_demo_input.json",
            self.root / "data/teacher_agent_evaluation_cases.json",
        )
        bootstrap = snapshot.bootstrap()

        self.assertTrue(_bootstrap_contract_is_supported(bootstrap))
        self.assertTrue(
            bootstrap["interaction_contract"]["step_requires_context_version"]
        )
        self.assertTrue(
            bootstrap["interaction_contract"]["replacement_requires_context_version"]
        )

        renamed = json.loads(json.dumps(bootstrap))
        renamed["interaction_contract"].pop("step_requires_context_version")
        renamed["interaction_contract"][
            "step_context_version_binding_enabled"
        ] = True
        self.assertFalse(_bootstrap_contract_is_supported(renamed))

    def test_replacement_payload_is_bound_to_the_exact_old_session_version(
        self,
    ) -> None:
        session = {
            "rounds_completed": 3,
            "expected_question_id": "question-004",
            "context_version": 5,
            "profile_summary": {"profile_revision": "profile-a-v4"},
        }
        payload = _start_payload(
            goal={},
            profile={},
            profile_revision="profile-b-v1",
            display_name="Synthetic B",
            nonce="replacement-binding",
            replace_session_id="opaque-old-session",
            replace_session=session,
        )
        self.assertEqual(payload["replace_expected_round"], 3)
        self.assertEqual(payload["replace_expected_question_id"], "question-004")
        self.assertEqual(payload["replace_expected_context_version"], 5)
        self.assertEqual(
            payload["replace_expected_profile_revision"], "profile-a-v4"
        )
        with self.assertRaises(RuntimeError):
            _start_payload(
                goal={},
                profile={},
                profile_revision="profile-b-v1",
                display_name="Synthetic B",
                nonce="missing-replacement-binding",
                replace_session_id="opaque-old-session",
            )

    def test_acknowledgement_flag_is_mandatory_before_any_request(self) -> None:
        stdout = io.StringIO()
        with (
            patch(
                "scripts.run_teacher_agent_system_acceptance._DashboardHttpClient"
            ) as client,
            redirect_stdout(stdout),
        ):
            exit_code = main(["--base-url", self.safe_unused_url])
        self.assertEqual(exit_code, 2)
        client.assert_not_called()
        receipt = json.loads(stdout.getvalue())
        self.assertFalse(receipt["passed"])
        self.assertFalse(receipt["remote_demo_text_acknowledgement_present"])
        self.assertEqual(receipt["failure"]["stage"], "consent")
        self.assertNotIn(self.safe_unused_url, stdout.getvalue())

    def test_non_loopback_target_is_rejected_without_network_access(self) -> None:
        with patch(
            "scripts.run_teacher_agent_system_acceptance.build_opener"
        ) as opener:
            receipt = run_system_acceptance(
                "https://example.invalid/not-a-capability/",
                acknowledge_remote_demo_text=True,
            )
        opener.assert_not_called()
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["failure"]["stage"], "input")
        self.assertEqual(receipt["counts"]["http_requests"], 0)
        self.assertNotIn("example.invalid", json.dumps(receipt))

    def test_script_source_has_no_browser_or_capability_logging_path(self) -> None:
        source = (
            Path(project_root())
            / "scripts/run_teacher_agent_system_acceptance.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("import webbrowser", source)
        self.assertNotIn("print(base_url", source)


if __name__ == "__main__":
    unittest.main()
