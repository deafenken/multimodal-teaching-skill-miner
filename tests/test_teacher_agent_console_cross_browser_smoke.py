from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_teacher_agent_console_cross_browser_smoke import (
    SmokeFailure,
    _BrowserErrorGate,
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
            "firefox",
            _console_error(url="http://127.0.0.1:8766/teachlab-sw-v1.js"),
            "navigation_console_error_other_origin",
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
