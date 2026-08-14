from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.run_teacher_agent_console_cross_browser_smoke import (
    SmokeFailure,
    _BrowserErrorGate,
)


ORIGIN = "http://127.0.0.1:8765"
EXPECTED_FIREFOX_TEXT = (
    '[JavaScript Error: "Failed to load. A ServiceWorker intercepted the request '
    'and encountered an unexpected error."]'
)


def _console_error(
    *,
    text: str = EXPECTED_FIREFOX_TEXT,
    url: str = f"{ORIGIN}/teachlab-sw-v1.js",
) -> SimpleNamespace:
    return SimpleNamespace(type="error", text=text, location={"url": url})


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


def test_one_scoped_firefox_offline_network_diagnostic_is_accepted() -> None:
    gate = _BrowserErrorGate("firefox", ORIGIN)
    gate.require_online_clean()
    gate.begin_intentional_offline_navigation()

    gate.on_console(_console_error())
    gate.require_offline_clean()

    assert gate.accepted_firefox_offline_diagnostics == 1
    assert gate.offline_console_errors == 0
    assert EXPECTED_FIREFOX_TEXT not in repr(gate)


@pytest.mark.parametrize(
    ("browser", "message"),
    (
        ("chromium", _console_error()),
        ("webkit", _console_error()),
        (
            "firefox",
            _console_error(url="http://127.0.0.1:8766/teachlab-sw-v1.js"),
        ),
        (
            "firefox",
            _console_error(url="http://127.0.0.1:not-a-port/teachlab-sw-v1.js"),
        ),
        ("firefox", _console_error(url=f"{ORIGIN}/application.js")),
        ("firefox", _console_error(text="Failed to load")),
    ),
)
def test_offline_allowlist_rejects_other_engines_sources_and_messages(
    browser: str,
    message: SimpleNamespace,
) -> None:
    gate = _BrowserErrorGate(browser, ORIGIN)
    gate.begin_intentional_offline_navigation()
    gate.on_console(message)

    with pytest.raises(SmokeFailure) as caught:
        gate.require_offline_clean()

    assert caught.value.stage == "browser_offline"
    assert caught.value.code == "console_error"
    assert gate.accepted_firefox_offline_diagnostics == 0


def test_offline_allowlist_is_bounded_and_never_accepts_page_errors() -> None:
    duplicate_gate = _BrowserErrorGate("firefox", ORIGIN)
    duplicate_gate.begin_intentional_offline_navigation()
    duplicate_gate.on_console(_console_error())
    duplicate_gate.on_console(_console_error())
    with pytest.raises(SmokeFailure, match="^console_error$"):
        duplicate_gate.require_offline_clean()
    assert duplicate_gate.accepted_firefox_offline_diagnostics == 1
    assert duplicate_gate.offline_console_errors == 1

    page_error_gate = _BrowserErrorGate("firefox", ORIGIN)
    page_error_gate.begin_intentional_offline_navigation()
    page_error_gate.on_page_error(RuntimeError("private page payload"))
    with pytest.raises(SmokeFailure, match="^page_error$"):
        page_error_gate.require_offline_clean()
