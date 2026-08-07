from __future__ import annotations

import json
from pathlib import Path
import threading
import unittest

from scripts.run_teacher_agent_cancel_browser_acceptance import (
    RECEIPT_SCHEMA,
    _BlockingClient,
    _receipt,
)
from teaching_skill_miner.io_utils import project_root


class TeacherAgentCancelBrowserAcceptanceTests(unittest.TestCase):
    def test_blocking_client_pauses_only_the_second_model_call(self) -> None:
        client = _BlockingClient()
        first, first_trace = client.chat_json([], request_kind="teacher_agent_initial")
        self.assertEqual(first["schema"], "teaching_skill_miner.deepseek_turn_plan.v1")
        self.assertEqual(first_trace["provider"], "deepseek")
        self.assertFalse(client.blocked_call_entered.is_set())

        result: list[object] = []

        def second_call() -> None:
            result.append(client.chat_json([], request_kind="teacher_agent_turn"))

        thread = threading.Thread(target=second_call)
        thread.start()
        self.assertTrue(client.blocked_call_entered.wait(timeout=1))
        self.assertTrue(thread.is_alive())
        client.release_blocked_call.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(result), 1)
        self.assertEqual(client.chat_json_call_count, 2)

    def test_receipt_is_aggregate_and_never_exposes_session_content(self) -> None:
        receipt = _receipt(
            passed=True,
            browser="chrome",
            launched=True,
            checks={"late_response_fenced_and_draft_preserved": True},
            duration_ms=123,
        )
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["schema"], RECEIPT_SCHEMA)
        serialized = json.dumps(receipt, ensure_ascii=False)
        for forbidden in (
            "session_id",
            "learner_response",
            "capability_token",
            "我准备提交一条需要取消的回答",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_frontend_resynchronizes_controls_after_turn_becomes_cancellable(
        self,
    ) -> None:
        source = (
            Path(project_root())
            / "teaching_skill_miner/web/teacher_agent_demo.js"
        ).read_text(encoding="utf-8")
        marker = (
            "app.pendingTurn = {fingerprint: requestFingerprint, key: "
            "idempotencyKey, active: true};"
        )
        tail = source.split(marker, 1)[1].split(
            'const requestAnchor = beginRequestAnchor("step");', 1
        )[0]
        self.assertIn("syncControls();", tail)

    def test_runner_contains_real_dom_cancel_and_resume_assertions(self) -> None:
        source = (
            Path(project_root())
            / "scripts/run_teacher_agent_cancel_browser_acceptance.py"
        ).read_text(encoding="utf-8")
        for required in (
            'page.locator("#cancelTurnButton").click()',
            "stop_button_visible_while_busy",
            "late_response_fenced_and_draft_preserved",
            "session_active_without_committed_turn",
            "resume_after_cancel_commits_next_turn",
            "client.release_blocked_call.set()",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        for forbidden in (
            "page.screenshot",
            "context.tracing",
            "print(url",
            "print(session",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
