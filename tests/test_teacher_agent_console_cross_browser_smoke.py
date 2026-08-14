from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_teacher_agent_console_cross_browser_smoke import (
    SmokeFailure,
    _BrowserErrorGate,
    _BROWSER_INTERACTION_STAGES,
    _MAX_FIREFOX_OFFLINE_NETWORK_DIAGNOSTICS,
)


ORIGIN = "http://127.0.0.1:8765"
EXPECTED_FIREFOX_TEXT = (
    '[JavaScript Error: "Failed to load. A ServiceWorker intercepted the request '
    'and encountered an unexpected error."]'
)
SERVICE_WORKER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "console"
    / "public"
    / "teachlab-sw-v1.js"
).read_text(encoding="utf-8")
RUNNER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_teacher_agent_console_cross_browser_smoke.py"
).read_text(encoding="utf-8")


def _console_error(
    *,
    text: str = EXPECTED_FIREFOX_TEXT,
    url: str = f"{ORIGIN}/teachlab-sw-v1.js",
) -> SimpleNamespace:
    return SimpleNamespace(type="error", text=text, location={"url": url})


def test_scoped_firefox_allowlist_source_never_logs_console_errors() -> None:
    assert "console.error" not in SERVICE_WORKER_SOURCE
    assert 'console["error"]' not in SERVICE_WORKER_SOURCE
    assert "console['error']" not in SERVICE_WORKER_SOURCE
    assert "event.respondWith(fetch(request).catch(() => offlineDocument()))" in SERVICE_WORKER_SOURCE


def test_other_origin_diagnostic_quarantine_keeps_external_request_gate() -> None:
    assert 'page.on("request", observe_request)' in RUNNER_SOURCE
    assert "external_requests += 1" in RUNNER_SOURCE
    assert "_require(external_requests == 0" in RUNNER_SOURCE


def test_generic_browser_failures_emit_only_fixed_interaction_stages() -> None:
    assert _BROWSER_INTERACTION_STAGES == {
        "root_navigation",
        "workspace_controls",
        "security_contract",
        "teach_mode_controls",
        "workspace_axe",
        "project_menu_focus",
        "deletion_prompt_focus",
        "command_dialog_focus",
        "mobile_inspector_focus",
        "reflow",
        "service_worker_control",
        "static_cache_policy",
        "offline_snapshot_seed",
        "offline_navigation",
        "offline_shell_focus",
        "offline_shell_axe",
        "offline_final_gates",
    }
    assert "browser_interaction_failed_{safe_stage}" in RUNNER_SOURCE


def test_online_errors_fail_before_the_offline_transition_without_raw_text() -> None:
    raw_text = "private console payload"
    gate = _BrowserErrorGate("firefox", ORIGIN)
    gate.on_console(_console_error(text=raw_text))

    with pytest.raises(SmokeFailure) as caught:
        gate.require_online_clean()

    assert caught.value.stage == "browser_online"
    assert caught.value.code == "console_error"
    assert raw_text not in str(caught.value)
    assert raw_text not in repr(gate)
    assert gate.accepted_firefox_offline_diagnostics == 0


def test_scoped_firefox_offline_network_diagnostics_are_accepted() -> None:
    gate = _BrowserErrorGate("firefox", ORIGIN)
    gate.require_online_clean()
    gate.begin_intentional_offline_navigation()

    localized_diagnostic = _console_error(text="本地化的 Firefox 离线导航诊断")
    for _ in range(_MAX_FIREFOX_OFFLINE_NETWORK_DIAGNOSTICS):
        gate.on_console(localized_diagnostic)
    gate.complete_intentional_offline_navigation()
    gate.require_offline_clean()

    assert (
        gate.accepted_firefox_offline_diagnostics
        == _MAX_FIREFOX_OFFLINE_NETWORK_DIAGNOSTICS
    )
    assert gate.offline_console_errors == 0
    assert localized_diagnostic.text not in repr(gate)


@pytest.mark.parametrize(
    ("browser", "message", "expected_code"),
    (
        (
            "chromium",
            _console_error(),
            "navigation_console_error_sealed_worker",
        ),
        ("webkit", _console_error(), "navigation_console_error_sealed_worker"),
        (
            "chromium",
            _console_error(url="http://127.0.0.1:8766/teachlab-sw-v1.js"),
            "navigation_console_error_loopback_other_port",
        ),
        (
            "firefox",
            _console_error(url="http://127.0.0.1:not-a-port/teachlab-sw-v1.js"),
            "navigation_console_error_invalid_location",
        ),
        (
            "firefox",
            _console_error(url=f"{ORIGIN}/application.js"),
            "navigation_console_error_same_origin_other",
        ),
        (
            "firefox",
            _console_error(url=f"{ORIGIN}/teachlab-offline-shell-v1.js"),
            "navigation_console_error_offline_shell_script",
        ),
        (
            "firefox",
            _console_error(url=f"{ORIGIN}/_next/static/chunks/runtime.js"),
            "navigation_console_error_next_static_asset",
        ),
        (
            "firefox",
            _console_error(url=f"{ORIGIN}/?offline-smoke=fixture"),
            "navigation_console_error_navigation_document",
        ),
        (
            "firefox",
            SimpleNamespace(type="error", text="diagnostic", location={}),
            "navigation_console_error_missing_location",
        ),
        (
            "firefox",
            _console_error(url="https://example.invalid/application.js"),
            "navigation_console_error_web_other_origin",
        ),
        (
            "firefox",
            _console_error(url="custom-scheme:diagnostic"),
            "navigation_console_error_other_scheme",
        ),
    ),
)
def test_offline_allowlist_rejects_other_engines_sources_and_messages(
    browser: str,
    message: SimpleNamespace,
    expected_code: str,
) -> None:
    gate = _BrowserErrorGate(browser, ORIGIN)
    gate.begin_intentional_offline_navigation()
    gate.on_console(message)

    with pytest.raises(SmokeFailure) as caught:
        gate.complete_intentional_offline_navigation()

    assert caught.value.stage == "browser_offline"
    assert caught.value.code == expected_code
    assert gate.accepted_firefox_offline_diagnostics == 0


def test_offline_allowlist_is_bounded_and_never_accepts_page_errors() -> None:
    diagnostic_storm_gate = _BrowserErrorGate("firefox", ORIGIN)
    diagnostic_storm_gate.begin_intentional_offline_navigation()
    for _ in range(_MAX_FIREFOX_OFFLINE_NETWORK_DIAGNOSTICS + 1):
        diagnostic_storm_gate.on_console(_console_error())
    with pytest.raises(
        SmokeFailure,
        match="^navigation_console_error_diagnostic_limit$",
    ):
        diagnostic_storm_gate.complete_intentional_offline_navigation()
    assert (
        diagnostic_storm_gate.accepted_firefox_offline_diagnostics
        == _MAX_FIREFOX_OFFLINE_NETWORK_DIAGNOSTICS
    )
    assert diagnostic_storm_gate.offline_console_errors == 1

    page_error_gate = _BrowserErrorGate("firefox", ORIGIN)
    page_error_gate.begin_intentional_offline_navigation()
    page_error_gate.on_page_error(RuntimeError("private page payload"))
    with pytest.raises(SmokeFailure, match="^page_error$"):
        page_error_gate.complete_intentional_offline_navigation()


def test_firefox_browser_internal_diagnostic_is_scoped_to_navigation_window() -> None:
    gate = _BrowserErrorGate("firefox", ORIGIN)
    other_origin = _console_error(
        url="resource://gre/modules/ServiceWorkerManager.sys.mjs"
    )
    gate.begin_intentional_offline_navigation()
    gate.on_console(other_origin)
    gate.complete_intentional_offline_navigation()
    gate.require_offline_clean()

    assert gate.accepted_firefox_offline_diagnostics == 1

    gate.on_console(other_origin)
    with pytest.raises(
        SmokeFailure,
        match="^verified_shell_console_error_browser_internal$",
    ):
        gate.require_offline_clean()


def test_firefox_worker_diagnostic_fails_after_offline_shell_is_verified() -> None:
    gate = _BrowserErrorGate("firefox", ORIGIN)
    gate.begin_intentional_offline_navigation()
    gate.complete_intentional_offline_navigation()
    gate.on_console(_console_error())

    with pytest.raises(
        SmokeFailure,
        match="^verified_shell_console_error_sealed_worker$",
    ):
        gate.require_offline_clean()

    assert gate.accepted_firefox_offline_diagnostics == 0
    assert gate.offline_console_errors == 1
