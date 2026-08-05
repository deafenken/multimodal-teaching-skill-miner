from __future__ import annotations

from contextlib import redirect_stdout
from html.parser import HTMLParser
import http.client
import io
import json
import threading
import unittest
from urllib.parse import urlsplit

from teaching_skill_miner.cli import main
from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.teacher_agent_dashboard import (
    HTML_RESOURCE,
    SCRIPT_RESOURCE,
    STYLE_RESOURCE,
    _resource_bytes,
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
    teacher_agent_dashboard_self_check,
)


class _AgentDashboardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.screens: list[str] = []
        self.external_assets: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del tag
        values = dict(attrs)
        if values.get("data-screen-label"):
            self.screens.append(str(values["data-screen-label"]))
        for key in ("src", "href"):
            value = values.get(key)
            if value and value.startswith(("http://", "https://", "//")):
                self.external_assets.append(value)


class TeacherAgentDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = project_root()
        cls.library_path = cls.root / "data/teacher_agent_skill_library.json"
        cls.input_path = cls.root / "data/teacher_agent_demo_input.json"
        cls.cases_path = cls.root / "data/teacher_agent_evaluation_cases.json"
        cls.v2_library_path = cls.root / "data/teacher_agent_skill_library_v2.json"
        cls.neural_manifest_path = cls.root / "data/neural_v1_runtime_manifest.json"
        cls.outcome_path = cls.root / "data/teacher_agent_learning_outcome_demo.json"
        cls.benchmark_receipt_path = (
            cls.root / "data/teacher_agent_free_text_benchmark_receipt.json"
        )

    def test_generic_frontend_has_four_clear_screens_and_no_external_assets(self) -> None:
        html = _resource_bytes(HTML_RESOURCE).decode("utf-8")
        parser = _AgentDashboardParser()
        parser.feed(html)
        self.assertEqual(
            parser.screens,
            [
                "01 Session setup",
                "02 Live teaching loop",
                "03 Student state",
                "04 Reproducible evaluation",
            ],
        )
        self.assertEqual(parser.external_assets, [])
        self.assertIn("teacher_agent_demo.css", html)
        self.assertIn("teacher_agent_demo.js", html)
        script = _resource_bytes(SCRIPT_RESOURCE).decode("utf-8")
        self.assertIn("fetch(", script)
        self.assertIn("replaceChildren", script)
        self.assertNotIn(".innerHTML", script)
        self.assertIn('id="adaptiveProfileStatus"', html)
        self.assertIn('id="adaptiveConfidenceBar"', html)
        self.assertIn("Agent 候选画像", html)
        self.assertIn("renderAdaptiveStudentProfile(session)", script)
        self.assertIn("不会覆盖教师输入", script)
        self.assertIn('needsReview ? "需人工复核" : "未确认"', script)
        self.assertGreater(len(_resource_bytes(STYLE_RESOURCE)), 10_000)

    def test_self_check_and_cli_check_pass(self) -> None:
        report = teacher_agent_dashboard_self_check(
            self.library_path, self.input_path, self.cases_path
        )
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["real_time_skill_switching"])
        self.assertFalse(report["real_learning_effectiveness_established"])
        with redirect_stdout(io.StringIO()) as output:
            code = main(["teacher-agent-dashboard", "--check"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())["passed"])

    def test_v2_bootstrap_exposes_live_model_evidence_without_overclaiming(self) -> None:
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.v2_library_path,
            self.input_path,
            self.cases_path,
            neural_v1_manifest_path=self.neural_manifest_path,
            learning_outcome_path=self.outcome_path,
            free_text_benchmark_receipt_path=self.benchmark_receipt_path,
        )
        bootstrap = snapshot.bootstrap()
        receipt = bootstrap["evaluation"]["free_text_benchmark"]
        outcome = bootstrap["evaluation"]["learning_outcome"]
        self.assertEqual(
            receipt["online_deepseek"]["model"], "deepseek-v4-flash"
        )
        self.assertEqual(
            receipt["online_deepseek"]["metrics"]["signal"][
                "end_to_end_all_attempts"
            ]["macro_f1"],
            0.875325,
        )
        self.assertFalse(receipt["claim_boundary"]["expert_validated"])
        self.assertFalse(
            receipt["claim_boundary"]["deployment_accuracy_established"]
        )
        self.assertEqual(outcome["metrics"]["absolute_gain"], 0.4)
        self.assertFalse(
            outcome["claim_boundary"]["real_learner_effectiveness_established"]
        )
        self.assertFalse(
            bootstrap["neural_v1"]["materialization_gate"]["passed"]
        )
        self.assertFalse(
            bootstrap["interaction_contract"][
                "adaptive_profile_candidates_are_teacher_confirmed"
            ]
        )

    def test_loopback_server_runs_real_start_and_step_policy(self) -> None:
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.library_path, self.input_path, self.cases_path
        )
        try:
            server, url = create_teacher_agent_dashboard_server(
                snapshot, capability_token="a" * 24
            )
        except PermissionError:
            self.skipTest("sandbox does not permit loopback socket binding")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        parsed = urlsplit(url)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        try:
            connection.request("GET", f"{parsed.path}api/bootstrap")
            response = connection.getresponse()
            bootstrap = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Cache-Control"), "no-store, max-age=0")
            self.assertIn("default-src 'none'", response.getheader("Content-Security-Policy"))
            self.assertTrue(bootstrap["interaction_contract"]["one_action_per_turn"])

            start_body = json.dumps(
                {
                    "goal": snapshot.demo_input["goal"],
                    "student_profile": snapshot.demo_input["student_profile"],
                },
                ensure_ascii=False,
            ).encode("utf-8")
            connection.request(
                "POST",
                f"{parsed.path}api/start",
                body=start_body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            started = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(started["rounds_completed"], 0)
            self.assertEqual(started["history"], [])
            self.assertEqual(
                started["next_action"]["primary_skill"]["skill_id"],
                "skill_diagnostic_questioning",
            )

            step_body = json.dumps(
                {
                    "learner_response": "状态只需要看前一个位置。",
                    "signal": "misconception",
                    "misconception_tag": "missing_transition",
                },
                ensure_ascii=False,
            ).encode("utf-8")
            connection.request(
                "POST",
                f"{parsed.path}api/step",
                body=step_body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            stepped = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(stepped["rounds_completed"], 1)
            self.assertEqual(len(stepped["history"]), 1)
            self.assertEqual(
                stepped["next_action"]["primary_skill"]["skill_id"],
                "skill_misconception_contrast",
            )
            self.assertEqual(
                stepped["student_state"]["misconceptions"][0]["status"],
                "active",
            )
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

    def test_server_rejects_invalid_capability_and_non_json_post(self) -> None:
        snapshot = build_teacher_agent_dashboard_snapshot(
            self.library_path, self.input_path, self.cases_path
        )
        try:
            server, url = create_teacher_agent_dashboard_server(
                snapshot, capability_token="b" * 24
            )
        except PermissionError:
            self.skipTest("sandbox does not permit loopback socket binding")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        parsed = urlsplit(url)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        try:
            connection.request("GET", "/wrong/api/bootstrap")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 403)

            connection.request(
                "POST",
                f"{parsed.path}api/start",
                body=b"not-json",
                headers={"Content-Type": "text/plain"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 400)
            self.assertIn("Content-Type", payload["error"])
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
