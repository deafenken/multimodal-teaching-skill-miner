#!/usr/bin/env python3
"""Exercise the real browser ``停止生成`` path with a blocking model stub.

This is deliberately separate from the paid/remote DeepSeek browser receipt.  It
starts the same local dashboard HTTP handler, but injects a deterministic client
whose second model call pauses.  The browser must be able to send ``cancel_turn``
while the step request is still in flight; the late model result is then released
and must be rejected by the server's commit fence.  Stdout contains only a fixed,
aggregate receipt (never session IDs, learner text, or the capability URL).

The script is an engineering interaction check, not evidence of model quality.
Playwright and an installed Chromium/Chrome are optional runtime dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlsplit

# When invoked as ``python scripts/...`` Python puts ``scripts/`` first on
# sys.path.  Prefer the checked-out source tree over an older installed wheel,
# otherwise the browser test can accidentally exercise stale code or fixtures.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from teaching_skill_miner.deepseek_client import DeepSeekClientError  # noqa: E402
from teaching_skill_miner.io_utils import project_root  # noqa: E402
from teaching_skill_miner.teacher_agent_dashboard import (  # noqa: E402
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
)  # noqa: E402


RECEIPT_SCHEMA = "teaching_skill_miner.teacher_agent_cancel_browser_acceptance.v1"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
_CAPABILITY_PATH_RE = re.compile(r"/[A-Za-z0-9_-]{20,128}/")


class _BlockingClient:
    """A small valid live client which blocks exactly one call."""

    def __init__(self) -> None:
        self.chat_json_call_count = 0
        self.blocked_call_entered = threading.Event()
        self.release_blocked_call = threading.Event()
        self._blocked_once = False

    def public_status(self) -> dict[str, object]:
        return {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_origin": "https://api.deepseek.com",
            "configured": True,
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "remote_student_data_opt_in": True,
            "api_key_exposed": False,
        }

    def chat_json(
        self,
        messages: object,
        *,
        request_kind: str,
        require_remote_consent: bool = True,
    ) -> tuple[dict[str, object], dict[str, object]]:
        del messages, require_remote_consent
        self.chat_json_call_count += 1
        plan = {
            "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
            "diagnosis": {
                "signal": "not_observed",
                "confidence": 0.0,
                "answer_alignment": "not_applicable",
                "matched_concepts": [],
                "missing_concepts": [],
                "diagnosis_reason": "等待学生作答",
                "evidence_excerpt": "",
                "misconception_tag": None,
                "misconception_description": "",
                "resolved_misconception_tags": [],
                "response_quality": "empty",
                "engagement_level": "unknown",
                "needs_human_review": False,
            },
            "decision": {
                "primary_skill_id": "skill_diagnostic_questioning",
                "supporting_skill_ids": ["skill_wait_and_elicit"],
                "selection_reason": "先确认学生的前置知识。",
                "next_focus": "prerequisite",
            },
            "teacher_action": {
                "type": "ask_one_question",
                "message": "请先说出一个完成当前目标所需的前置概念。",
                "expected_signal": "学生给出一个相关的前置概念。",
            },
            "stop_recommendation": {"should_stop": False, "reason": ""},
        }
        trace = {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "request_kind": request_kind,
            "request_sha256": "b" * 64,
            "latency_ms": 1.0,
            "attempt_count": 1,
            "http_status": 200,
            "response_id": "browser-cancel-stub",
            "usage": {},
            "credential_logged": False,
        }
        if request_kind == "teacher_agent_turn" and not self._blocked_once:
            self._blocked_once = True
            self.blocked_call_entered.set()
            if not self.release_blocked_call.wait(timeout=10):
                raise DeepSeekClientError("blocking browser test timed out")
        return plan, trace


def _load_playwright() -> tuple[Any, type[BaseException]]:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    return sync_playwright, PlaywrightError


def _valid_target(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return bool(
            parsed.scheme == "http"
            and parsed.hostname in _LOOPBACK_HOSTS
            and parsed.port
            and _CAPABILITY_PATH_RE.fullmatch(parsed.path)
            and not parsed.query
            and not parsed.fragment
        )
    except (TypeError, ValueError):
        return False


def _receipt(
    *,
    passed: bool,
    browser: str,
    launched: bool,
    checks: Mapping[str, bool],
    failure: str | None = None,
    duration_ms: int = 0,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "dashboard_kind": "loopback_interactive_teacher_agent",
        "transport": "real_loopback_playwright_browser",
        "browser": {"engine": browser, "headless": True, "launched": launched},
        "passed": bool(passed and not failure),
        "checks": dict(checks),
        "counts": {
            "model_calls": 0,
            "turns_committed": 0,
            "cancel_commands": 0,
            "console_errors": 0,
            "expected_cancel_console_errors": 0,
            "page_errors": 0,
            "request_failures": 0,
            "blocked_external_requests": 0,
            "duration_ms": max(0, int(duration_ms)),
        },
        "privacy": {
            "capability_url_emitted": False,
            "session_handles_emitted": False,
            "student_text_emitted": False,
            "raw_exceptions_emitted": False,
        },
    }
    if failure:
        result["failure"] = {"stage": "verification", "code": failure}
    return result


def run_cancel_acceptance(
    *, browser_name: str = "chrome", timeout_seconds: float = 30.0
) -> dict[str, Any]:
    started_at = time.monotonic()
    checks: dict[str, bool] = {}
    client = _BlockingClient()
    browser_launched = False
    try:
        sync_playwright, playwright_error = _load_playwright()
    except (ImportError, ModuleNotFoundError):
        return _receipt(
            passed=False,
            browser=browser_name,
            launched=False,
            checks=checks,
            failure="python_playwright_not_installed",
            duration_ms=round((time.monotonic() - started_at) * 1000),
        )

    root = project_root()
    snapshot = build_teacher_agent_dashboard_snapshot(
        root / "data/teacher_agent_skill_library_v2.json",
        root / "data/teacher_agent_demo_input.json",
        root / "data/teacher_agent_evaluation_cases.json",
        client=client,
    )
    server, url = create_teacher_agent_dashboard_server(snapshot, port=0)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    browser = None
    context = None
    try:
        if not _valid_target(url):
            return _receipt(
                passed=False,
                browser=browser_name,
                launched=False,
                checks=checks,
                failure="invalid_loopback_capability_url",
            )
        manager = sync_playwright()
        playwright = manager.__enter__()
        try:
            options: dict[str, Any] = {"headless": True}
            if browser_name == "chrome":
                options["channel"] = "chrome"
            try:
                browser = playwright.chromium.launch(**options)
            except playwright_error:
                return _receipt(
                    passed=False,
                    browser=browser_name,
                    launched=False,
                    checks=checks,
                    failure="browser_executable_unavailable",
                )
            browser_launched = True
            context = browser.new_context(
                viewport={"width": 1440, "height": 900}, service_workers="block"
            )
            external_requests = 0
            console_errors = 0
            expected_cancel_console_errors = 0
            page_errors = 0
            request_failures = 0
            cancel_requested = False

            def guard(route: Any) -> None:
                nonlocal external_requests
                target = urlsplit(route.request.url)
                base = urlsplit(url)
                if (
                    target.scheme == base.scheme == "http"
                    and target.hostname == base.hostname
                    and target.port == base.port
                    and target.path.startswith(base.path)
                ):
                    route.continue_()
                else:
                    external_requests += 1
                    route.abort("blockedbyclient")

            context.route("**/*", guard)
            page = context.new_page()
            # Avoid retaining console text; count only errors.
            def on_console(message: Any) -> None:
                nonlocal console_errors, expected_cancel_console_errors
                if getattr(message, "type", "") == "error":
                    # A fetch rejected by the server after commit-fence
                    # cancellation is surfaced by Chromium as a resource
                    # 400.  It is the expected HTTP representation of the
                    # cancelled turn, not a JavaScript/runtime error.
                    message_text = str(getattr(message, "text", ""))
                    if cancel_requested and re.search(
                        r"(?:status of )?400", message_text, re.I
                    ):
                        expected_cancel_console_errors += 1
                    else:
                        console_errors += 1

            page.on("console", on_console)

            def on_page_error(_error: Any) -> None:
                nonlocal page_errors
                page_errors += 1

            page.on("pageerror", on_page_error)

            def on_request_failed(_request: Any) -> None:
                nonlocal request_failures
                request_failures += 1

            page.on("requestfailed", on_request_failed)
            page.goto(url, wait_until="domcontentloaded", timeout=round(timeout_seconds * 1000))
            page.wait_for_function(
                "() => document.querySelector('#startButton')?.disabled === false",
                timeout=round(timeout_seconds * 1000),
            )
            page.locator("#remoteConsent").check()
            page.locator("#startButton").click()
            page.wait_for_function(
                "() => document.querySelector('#activeSession')?.hidden === false",
                timeout=round(timeout_seconds * 1000),
            )
            checks["session_started"] = True
            page.wait_for_function(
                "() => document.querySelector('#stepButton')?.disabled === false",
                timeout=round(timeout_seconds * 1000),
            )
            page.locator("#learnerResponse").fill("我准备提交一条需要取消的回答。")
            page.locator("#stepButton").click()
            if not client.blocked_call_entered.wait(timeout=timeout_seconds):
                return _receipt(
                    passed=False,
                    browser=browser_name,
                    launched=browser_launched,
                    checks=checks,
                    failure="blocking_model_call_not_entered",
                )
            page.wait_for_function(
                "() => { const b=document.querySelector('#cancelTurnButton'); return b && !b.hidden && !b.disabled && b.textContent.trim() === '停止生成'; }",
                timeout=round(timeout_seconds * 1000),
            )
            checks["stop_button_visible_while_busy"] = True
            cancel_requested = True
            page.locator("#cancelTurnButton").click()
            page.wait_for_function(
                "() => document.querySelector('#cancelTurnButton')?.textContent.trim() === '正在停止…'",
                timeout=round(timeout_seconds * 1000),
            )
            checks["stop_button_acknowledged"] = True
            page.wait_for_function(
                "() => document.querySelector('#toast')?.textContent.includes('会话仍可继续')",
                timeout=round(timeout_seconds * 1000),
            )
            checks["cancel_command_confirmed_in_ui"] = True
            client.release_blocked_call.set()
            page.wait_for_function(
                "() => { const e=document.querySelector('#turnError'); const b=document.querySelector('#cancelTurnButton'); const input=document.querySelector('#learnerResponse'); return Boolean(e && !e.hidden && e.textContent.includes('生成已停止') && b && b.hidden && input && input.value.includes('需要取消')); }",
                timeout=round(timeout_seconds * 1000),
            )
            checks["late_response_fenced_and_draft_preserved"] = True
            page.wait_for_function(
                "() => document.querySelector('#roundCounter')?.textContent.trim() === 'R0' && document.querySelector('#historyCount')?.textContent.trim() === '0 轮' && document.querySelector('#stepButton')?.disabled === false",
                timeout=round(timeout_seconds * 1000),
            )
            checks["session_active_without_committed_turn"] = True
            # A second normal turn must still be possible after cancellation.
            page.locator("#learnerResponse").fill("取消后继续教学的一条新回答。")
            page.locator("#stepButton").click()
            page.wait_for_function(
                "() => document.querySelector('#roundCounter')?.textContent.trim() === 'R1' && document.querySelector('#historyCount')?.textContent.trim() === '1 轮'",
                timeout=round(timeout_seconds * 1000),
            )
            checks["resume_after_cancel_commits_next_turn"] = True
            checks["no_console_errors"] = console_errors == 0
            checks["no_page_errors"] = page_errors == 0
            checks["no_failed_browser_requests"] = request_failures == 0
            checks["all_requests_loopback_only"] = external_requests == 0
            result = _receipt(
                passed=all(checks.values()),
                browser=browser_name,
                launched=browser_launched,
                checks=checks,
                duration_ms=round((time.monotonic() - started_at) * 1000),
            )
            result["counts"].update(
                {
                    "model_calls": client.chat_json_call_count,
                    "turns_committed": 1,
                    "cancel_commands": 1,
                    "console_errors": console_errors,
                    "expected_cancel_console_errors": expected_cancel_console_errors,
                    "page_errors": page_errors,
                    "request_failures": request_failures,
                    "blocked_external_requests": external_requests,
                }
            )
            return result
        finally:
            if context is not None:
                try:
                    context.close()
                except Exception:
                    pass
            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    pass
            manager.__exit__(None, None, None)
    except Exception:
        return _receipt(
            passed=False,
            browser=browser_name,
            launched=browser_launched,
            checks=checks,
            failure="unexpected_browser_cancel_acceptance_failure",
            duration_ms=round((time.monotonic() - started_at) * 1000),
        )
    finally:
        client.release_blocked_call.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", choices=("chromium", "chrome"), default="chrome")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    receipt = run_cancel_acceptance(
        browser_name=args.browser, timeout_seconds=args.timeout_seconds
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
