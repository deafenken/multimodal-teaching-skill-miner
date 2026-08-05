#!/usr/bin/env python3
"""Drive the local Teaching Agent UI with a real Chromium browser.

The target must be an already-running, capability-token-protected loopback URL.
This runner never starts the dashboard server and never emits the target URL,
opaque session handles, profile payloads, learner text, console messages, or raw
exceptions.  Its stdout is one aggregate JSON receipt.

Pillow and Playwright are intentionally optional runtime dependencies. Install
the light browser-test extra and a browser separately before running this
acceptance, for example::

    python -m pip install -e '.[browser-test]'
    python -m playwright install chromium

The second command is an explicit operator action because it downloads a browser.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from io import BytesIO
import json
import re
import time
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageDraw, ImageFont


RECEIPT_SCHEMA = "teaching_skill_miner.teacher_agent_browser_acceptance.v1"
_CAPABILITY_PATH_RE = re.compile(r"/[A-Za-z0-9_-]{20,128}/")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
_BROWSERS = frozenset({"chromium", "chrome"})
_VIEWPORTS = (390, 768, 1440)
_MASTERY_FIELDS = {
    "prerequisite": "#initialPrerequisite",
    "conceptual": "#initialConceptual",
    "procedural": "#initialProcedural",
    "transfer": "#initialTransfer",
}
_MASTERY_BARS = {
    "prerequisite": "#prerequisiteBar",
    "conceptual": "#conceptualBar",
    "procedural": "#proceduralBar",
    "transfer": "#transferBar",
}
_PROFILE_B_CUSTOM_MASTERY_FIELD = "conceptual"
_PROFILE_B_CUSTOM_MASTERY_PERCENT = 65
_PROFILE_B_CUSTOM_MASTERY_VALUE = _PROFILE_B_CUSTOM_MASTERY_PERCENT / 100.0
_IMAGE_ANSWER_TEXT = (
    "Recursion is a prerequisite concept.\n"
    "Fibonacci contains repeated smaller subproblems.\n"
    "Dynamic programming stores and reuses each state result."
)


class _AcceptanceFailure(RuntimeError):
    """A fixed-code failure safe to include in the aggregate receipt."""

    def __init__(self, stage: str, code: str) -> None:
        super().__init__(code)
        self.stage = stage
        self.code = code


@dataclass(slots=True)
class _Telemetry:
    console_errors: int = 0
    expected_stale_console_errors: int = 0
    page_errors: int = 0
    request_failures: int = 0
    blocked_external_requests: int = 0
    start_requests: int = 0
    replacement_start_requests: int = 0
    attachment_requests: int = 0
    step_requests: int = 0
    image_step_requests: int = 0
    request_capture_failures: int = 0
    attachment_request_had_expected_payload: bool = False
    step_request_had_attachment_ids: bool = False
    step_request_was_image_only: bool = False
    replacement_request_had_manual_skill: bool = False
    replacement_request_had_version_guards: bool = False
    replacement_request_had_profile_roundtrip_fields: bool = False
    replacement_request_had_custom_mastery: bool = False
    replacement_guard_snapshots: list[tuple[int, str, int, str]] = field(
        default_factory=list
    )
    replacement_idempotency_keys: list[str] = field(default_factory=list)
    external_session_advance_completed: bool = False
    expected_stale_replacement_rejections: int = 0
    overflow_failures: int = 0
    overflow_by_viewport: dict[int, bool] = field(default_factory=dict)


def _image_answer_png() -> bytes:
    """Create a high-contrast answer image entirely in memory."""

    width, height = 1800, 430
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=54)
    bounds = draw.multiline_textbbox((0, 0), _IMAGE_ANSWER_TEXT, font=font, spacing=22)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    origin = (
        max(48, (width - text_width) // 2),
        max(48, (height - text_height) // 2 - bounds[1]),
    )
    draw.multiline_text(
        origin,
        _IMAGE_ANSWER_TEXT,
        fill="black",
        font=font,
        spacing=22,
    )
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _validated_base_url(raw: str) -> str:
    """Accept only an exact IPv4/IPv6 loopback capability URL."""

    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise _AcceptanceFailure("input", "invalid_base_url") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
        or _CAPABILITY_PATH_RE.fullmatch(parsed.path) is None
    ):
        raise _AcceptanceFailure("input", "non_loopback_or_invalid_capability_url")
    return raw


def _request_within_capability(base_url: str, target_url: str) -> bool:
    """Return whether a browser request stays under the selected capability."""

    try:
        base = urlsplit(base_url)
        target = urlsplit(target_url)
        base_port = base.port
        target_port = target.port
    except (TypeError, ValueError):
        return False
    decoded_path = unquote(target.path)
    if any(part in {".", ".."} for part in decoded_path.split("/")):
        return False
    return bool(
        target.scheme == base.scheme == "http"
        and target.hostname == base.hostname
        and target.hostname in _LOOPBACK_HOSTS
        and target_port == base_port
        and target.username is None
        and target.password is None
        and target.path.startswith(base.path)
    )


def _load_playwright() -> tuple[Callable[[], Any], type[BaseException]]:
    """Load Playwright lazily so normal project installs remain dependency-free."""

    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    return sync_playwright, PlaywrightError


def _require(condition: bool, *, stage: str, code: str) -> None:
    if not condition:
        raise _AcceptanceFailure(stage, code)


def _perform(callback: Callable[[], Any], *, stage: str, code: str) -> Any:
    """Convert browser-library details into one fixed, non-sensitive code."""

    try:
        return callback()
    except _AcceptanceFailure:
        raise
    except Exception:
        raise _AcceptanceFailure(stage, code) from None


def _wait_for_js(
    page: Any,
    expression: str,
    *,
    stage: str,
    code: str,
    timeout_ms: int,
    arg: Any = None,
) -> None:
    _perform(
        lambda: page.wait_for_function(expression, arg=arg, timeout=timeout_ms),
        stage=stage,
        code=code,
    )


def _form_mastery(page: Any) -> dict[str, float]:
    return {
        dimension: float(page.locator(selector).input_value()) / 100.0
        for dimension, selector in _MASTERY_FIELDS.items()
    }


def _rendered_mastery(page: Any) -> dict[str, float]:
    return {
        dimension: float(
            page.locator(selector).evaluate("element => Number(element.value)")
        )
        for dimension, selector in _MASTERY_BARS.items()
    }


def _same_mastery(first: Mapping[str, float], second: Mapping[str, float]) -> bool:
    return all(
        dimension in first
        and dimension in second
        and abs(float(first[dimension]) - float(second[dimension])) <= 0.001
        for dimension in _MASTERY_FIELDS
    )


def _page_has_no_horizontal_overflow(page: Any) -> bool:
    return bool(
        page.evaluate(
            """() => {
                const root = document.documentElement;
                const body = document.body;
                const scrolling = document.scrollingElement || root;
                const viewport = window.innerWidth;
                return Math.max(
                    root.scrollWidth,
                    body ? body.scrollWidth : 0,
                    scrolling.scrollWidth
                ) <= viewport + 1;
            }"""
        )
    )


def _advance_active_session_outside_ui(page: Any, *, timeout_ms: int) -> bool:
    """Advance the authoritative Session without updating this tab's UI state."""

    return bool(
        _perform(
            lambda: page.evaluate(
                """async () => {
                    const sessionId = sessionStorage.getItem(
                        'teachlab_opaque_session_handle_v2'
                    );
                    if (!sessionId) return false;
                    const post = async (relative, payload) => {
                        const response = await fetch(
                            new URL(relative, window.location.href).href,
                            {
                                method: 'POST',
                                cache: 'no-store',
                                credentials: 'same-origin',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify(payload)
                            }
                        );
                        let body = null;
                        try { body = await response.json(); } catch (_error) {}
                        return {ok: response.ok, body};
                    };
                    const resumed = await post('api/session', {
                        session_id: sessionId
                    });
                    if (!resumed.ok || !resumed.body) return false;
                    const session = resumed.body;
                    const advanced = await post('api/step', {
                        session_id: sessionId,
                        learner_response:
                            '我补充一个当前问题相关的解释，但仍需要逐步核对。',
                        attachment_ids: [],
                        expected_round: session.rounds_completed,
                        expected_question_id: session.expected_question_id,
                        expected_context_version: session.context_version,
                        profile_revision: session.profile_summary?.profile_revision,
                        signal: 'partial',
                        signal_confidence: 1.0,
                        idempotency_key: `browser-external-${Date.now()}-${Math.random()}`
                    });
                    return advanced.ok
                        && advanced.body?.rounds_completed
                            === session.rounds_completed + 1;
                }"""
            ),
            stage="profile_b_stale_retry",
            code="external_session_advance_failed",
        )
    )


def _receipt(
    *,
    checks: Mapping[str, bool],
    optional_checks: Mapping[str, str],
    telemetry: _Telemetry,
    browser_name: str,
    headless: bool,
    duration_ms: int,
    browser_launched: bool,
    acknowledgement_present: bool = True,
    failure_stage: str | None = None,
    failure_code: str | None = None,
) -> dict[str, Any]:
    optional_failed = any(status == "failed" for status in optional_checks.values())
    passed = (
        bool(checks)
        and all(checks.values())
        and not optional_failed
        and failure_code is None
    )
    result: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "dashboard_kind": "loopback_interactive_teacher_agent",
        "transport": "real_loopback_playwright_browser",
        "browser": {
            "engine": browser_name,
            "headless": headless,
            "launched": browser_launched,
        },
        "passed": passed,
        "checks": dict(checks),
        "optional_checks": dict(optional_checks),
        "counts": {
            "profiles_exercised": (
                2 if checks.get("profile_b_replacement_started") else 0
            ),
            "turns_committed": sum(
                bool(checks.get(key))
                for key in (
                    "profile_a_turn_committed",
                    "profile_b_turn_committed_after_replacement",
                )
            ),
            "viewports_checked": len(telemetry.overflow_by_viewport),
            "console_errors": telemetry.console_errors,
            "expected_stale_console_errors": (
                telemetry.expected_stale_console_errors
            ),
            "expected_stale_replacement_rejections": (
                telemetry.expected_stale_replacement_rejections
            ),
            "page_errors": telemetry.page_errors,
            "request_failures": telemetry.request_failures,
            "blocked_external_requests": telemetry.blocked_external_requests,
            "start_requests": telemetry.start_requests,
            "replacement_start_requests": (telemetry.replacement_start_requests),
            "attachment_requests": telemetry.attachment_requests,
            "step_requests": telemetry.step_requests,
            "image_step_requests": telemetry.image_step_requests,
            "request_capture_failures": telemetry.request_capture_failures,
            "overflow_failures": telemetry.overflow_failures,
            "duration_ms": max(0, int(duration_ms)),
        },
        "privacy": {
            "capability_url_emitted": False,
            "session_handles_emitted": False,
            "student_text_emitted": False,
            "raw_image_bytes_emitted": False,
            "ocr_text_emitted": False,
            "attachment_handles_emitted": False,
            "console_messages_emitted": False,
            "raw_exceptions_emitted": False,
            "screenshots_written": False,
            "traces_written": False,
        },
        "remote_demo_text_acknowledgement_present": acknowledgement_present,
    }
    if failure_code is not None:
        result["failure"] = {
            "stage": failure_stage or "internal",
            "code": failure_code,
        }
    return result


def _exercise_browser(
    page: Any,
    *,
    timeout_ms: int,
    telemetry: _Telemetry,
) -> tuple[dict[str, bool], dict[str, str]]:
    """Exercise the browser UI without returning any user/session content."""

    checks: dict[str, bool] = {}
    optional_checks: dict[str, str] = {}

    _wait_for_js(
        page,
        """() => {
            const start = document.querySelector('#startButton');
            const cards = document.querySelector('#profileCards');
            const provider = document.querySelector('#providerState');
            return Boolean(start && cards && provider && !start.disabled
                && provider.textContent.trim() !== '检查中');
        }""",
        stage="page_load",
        code="dashboard_did_not_become_ready",
        timeout_ms=timeout_ms,
    )
    checks["page_loaded"] = True
    _require(
        page.locator("#learningView").is_visible()
        and page.locator("#setupForm").is_visible(),
        stage="page_load",
        code="initial_learning_setup_not_visible",
    )
    checks["initial_setup_visible"] = True

    profile_a_mastery = _perform(
        lambda: _form_mastery(page),
        stage="profile_a_start",
        code="profile_a_mastery_unreadable",
    )
    concept_value = _perform(
        lambda: page.locator("#conceptInput").input_value(),
        stage="profile_a_start",
        code="goal_form_unreadable",
    )
    objective_value = _perform(
        lambda: page.locator("#objectiveInput").input_value(),
        stage="profile_a_start",
        code="goal_form_unreadable",
    )
    _require(
        bool(str(concept_value).strip()) and bool(str(objective_value).strip()),
        stage="profile_a_start",
        code="default_goal_not_materialized",
    )
    _perform(
        lambda: page.locator("#remoteConsent").check(),
        stage="profile_a_start",
        code="remote_consent_control_unusable",
    )
    _perform(
        lambda: page.locator("#startButton").click(),
        stage="profile_a_start",
        code="profile_a_start_click_failed",
    )
    _wait_for_js(
        page,
        """({name, roundText}) => {
            const active = document.querySelector('#activeSession');
            const setup = document.querySelector('#setupForm');
            const student = document.querySelector('#conversationStudentName');
            const round = document.querySelector('#roundCounter');
            const historyCount = document.querySelector('#historyCount');
            return Boolean(active && !active.hidden && setup && setup.hidden
                && student && student.textContent.includes(name)
                && round && round.textContent.trim() === roundText
                && historyCount && historyCount.textContent.trim() === '0 轮');
        }""",
        arg={"name": "小雨", "roundText": "R0"},
        stage="profile_a_start",
        code="profile_a_session_not_started",
        timeout_ms=timeout_ms,
    )
    profile_a_rendered = _perform(
        lambda: _rendered_mastery(page),
        stage="profile_a_start",
        code="profile_a_state_unreadable",
    )
    _require(
        _same_mastery(profile_a_mastery, profile_a_rendered),
        stage="profile_a_start",
        code="profile_a_initial_state_mismatch",
    )
    profile_a_handle = _perform(
        lambda: page.evaluate(
            """() => sessionStorage.getItem(
                'teachlab_opaque_session_handle_v2'
            ) || ''"""
        ),
        stage="profile_a_start",
        code="profile_a_session_handle_unreadable",
    )
    _require(
        isinstance(profile_a_handle, str) and bool(profile_a_handle),
        stage="profile_a_start",
        code="profile_a_session_handle_missing",
    )
    checks["profile_a_started"] = True

    image_payload = _image_answer_png()
    _perform(
        lambda: page.locator("#answerImageInput").set_input_files(
            {
                "name": "answer.png",
                "mimeType": "image/png",
                "buffer": image_payload,
            }
        ),
        stage="profile_a_image_turn",
        code="answer_image_selection_failed",
    )
    _wait_for_js(
        page,
        """() => {
            const preview = document.querySelector('#attachmentPreview');
            const input = document.querySelector('#learnerResponse');
            return Boolean(preview && !preview.hidden && input && input.value === '');
        }""",
        stage="profile_a_image_turn",
        code="image_only_answer_not_staged",
        timeout_ms=timeout_ms,
    )
    if page.locator("#fallbackSignalField").is_visible():
        _perform(
            lambda: page.locator("#fallbackSignalInput").select_option("correct"),
            stage="profile_a_image_turn",
            code="fallback_signal_selection_failed",
        )
    _perform(
        lambda: page.locator("#stepButton").click(),
        stage="profile_a_image_turn",
        code="image_turn_submit_click_failed",
    )
    _wait_for_js(
        page,
        """() => {
            const round = document.querySelector('#roundCounter');
            const history = document.querySelector('#historyList');
            const visual = history?.querySelector('.visual-evidence-summary');
            return Boolean(round && round.textContent.trim() === 'R1'
                && history && history.children.length >= 1
                && visual);
        }""",
        stage="profile_a_image_turn",
        code="profile_a_image_turn_not_committed",
        timeout_ms=timeout_ms,
    )
    visual_summary = _perform(
        lambda: page.locator(
            "#historyList .visual-evidence-summary"
        ).first.text_content(),
        stage="profile_a_image_turn",
        code="profile_a_visual_evidence_summary_unreadable",
    )
    visual_summary_text = str(visual_summary or "")
    _require(
        "本机识别器不可用" not in visual_summary_text,
        stage="profile_a_image_turn",
        code="local_ocr_unavailable",
    )
    _require(
        "未识别到可靠文字" not in visual_summary_text,
        stage="profile_a_image_turn",
        code="local_ocr_no_text_recognized",
    )
    _require(
        "recursion" in visual_summary_text.casefold(),
        stage="profile_a_image_turn",
        code="local_ocr_expected_text_missing",
    )
    _require(
        "原图未发送" in visual_summary_text,
        stage="profile_a_image_turn",
        code="visual_privacy_boundary_missing",
    )
    _require(
        page.locator("#assessmentLabel").text_content() == "理解正确",
        stage="profile_a_image_turn",
        code="profile_a_image_answer_not_recognized",
    )
    _require(
        telemetry.attachment_requests >= 1
        and telemetry.attachment_request_had_expected_payload,
        stage="profile_a_image_turn",
        code="attachment_request_not_observed",
    )
    _require(
        telemetry.image_step_requests >= 1
        and telemetry.step_request_had_attachment_ids
        and telemetry.step_request_was_image_only,
        stage="profile_a_image_turn",
        code="image_only_step_request_not_observed",
    )
    checks["profile_a_turn_committed"] = True
    checks["profile_a_correct_answer_recognized"] = True
    checks["profile_a_image_only_turn_committed"] = True
    checks["profile_a_image_attachment_uploaded"] = True
    checks["profile_a_image_step_bound"] = True
    checks["profile_a_visual_evidence_summary_rendered"] = True
    checks["profile_a_raw_image_not_sent"] = True

    draft_cancel_sentinel = "取消草稿不应成为活动目标"
    active_identity_before = _perform(
        lambda: (
            page.locator("#workspaceGoalTitle").text_content(),
            page.locator("#conversationTitle").text_content(),
            page.locator("#teacherMessage").text_content(),
            page.locator("#roundCounter").text_content(),
            page.evaluate(
                """() => sessionStorage.getItem(
                    'teachlab_opaque_session_handle_v2'
                ) || ''"""
            ),
        ),
        stage="draft_cancel",
        code="active_session_identity_unreadable_before_draft",
    )
    requests_before_draft = (
        telemetry.start_requests,
        telemetry.step_requests,
        telemetry.attachment_requests,
    )
    _perform(
        lambda: page.locator("#presetButton").click(),
        stage="draft_cancel",
        code="replacement_draft_open_failed",
    )
    _wait_for_js(
        page,
        "() => !document.querySelector('#setupForm')?.hidden",
        stage="draft_cancel",
        code="replacement_draft_not_visible",
        timeout_ms=timeout_ms,
    )
    _perform(
        lambda: page.locator("#conceptInput").fill(draft_cancel_sentinel),
        stage="draft_cancel",
        code="replacement_draft_goal_edit_failed",
    )
    _wait_for_js(
        page,
        """({concept, sentinel}) => {
            const setup = document.querySelector('#setupForm');
            const input = document.querySelector('#conceptInput');
            const workspace = document.querySelector('#workspaceGoalTitle');
            const conversation = document.querySelector('#conversationTitle');
            return Boolean(setup && !setup.hidden
                && input?.value === sentinel
                && workspace?.textContent.trim() === concept
                && conversation?.textContent.trim() === concept);
        }""",
        arg={"concept": str(concept_value), "sentinel": draft_cancel_sentinel},
        stage="draft_cancel",
        code="draft_edit_overwrote_active_goal_titles",
        timeout_ms=timeout_ms,
    )
    checks["draft_edit_kept_active_goal_titles"] = True
    _perform(
        lambda: page.locator("#presetButton").click(),
        stage="draft_cancel",
        code="replacement_draft_close_failed",
    )
    _wait_for_js(
        page,
        """({concept, sentinel}) => {
            const setup = document.querySelector('#setupForm');
            const workspace = document.querySelector('#workspaceGoalTitle');
            const conversation = document.querySelector('#conversationTitle');
            return Boolean(setup && setup.hidden
                && workspace?.textContent.trim() === concept
                && conversation?.textContent.trim() === concept
                && !workspace.textContent.includes(sentinel)
                && !conversation.textContent.includes(sentinel));
        }""",
        arg={"concept": str(concept_value), "sentinel": draft_cancel_sentinel},
        stage="draft_cancel",
        code="active_goal_titles_not_restored_after_draft_cancel",
        timeout_ms=timeout_ms,
    )
    active_identity_after = _perform(
        lambda: (
            page.locator("#workspaceGoalTitle").text_content(),
            page.locator("#conversationTitle").text_content(),
            page.locator("#teacherMessage").text_content(),
            page.locator("#roundCounter").text_content(),
            page.evaluate(
                """() => sessionStorage.getItem(
                    'teachlab_opaque_session_handle_v2'
                ) || ''"""
            ),
        ),
        stage="draft_cancel",
        code="active_session_identity_unreadable_after_draft",
    )
    _require(
        active_identity_after == active_identity_before,
        stage="draft_cancel",
        code="active_session_changed_after_draft_cancel",
    )
    checks["draft_cancel_preserved_session_identity"] = True
    _require(
        (
            telemetry.start_requests,
            telemetry.step_requests,
            telemetry.attachment_requests,
        )
        == requests_before_draft,
        stage="draft_cancel",
        code="draft_cancel_emitted_request",
    )
    checks["draft_cancel_emitted_no_request"] = True
    _require(
        page.locator("#learnerResponse").is_enabled()
        and page.locator("#stepButton").is_enabled(),
        stage="draft_cancel",
        code="active_composer_not_restored_after_draft_cancel",
    )
    checks["draft_cancel_restored_active_composer"] = True

    if page.locator("#manualModeButton").is_hidden():
        _perform(
            lambda: page.locator("#methodTab").click(),
            stage="manual_skill",
            code="method_inspector_tab_open_failed",
        )
        _wait_for_js(
            page,
            "() => Boolean(document.querySelector('#manualModeButton')?.offsetParent)",
            stage="manual_skill",
            code="manual_skill_controls_not_visible",
            timeout_ms=timeout_ms,
        )
    manual_supported = _perform(
        lambda: (
            page.locator("#manualModeButton").is_visible()
            and page.locator("#manualModeButton").is_enabled()
        ),
        stage="manual_skill",
        code="manual_skill_availability_unreadable",
    )
    manual_applied = False
    if manual_supported:
        _perform(
            lambda: page.locator("#manualModeButton").click(),
            stage="manual_skill",
            code="manual_mode_open_failed",
        )
        option_values = _perform(
            lambda: page.locator("#skillOverrideSelect option").evaluate_all(
                "elements => elements.map(element => element.value).filter(Boolean)"
            ),
            stage="manual_skill",
            code="manual_skill_options_unreadable",
        )
        _require(
            isinstance(option_values, list) and bool(option_values),
            stage="manual_skill",
            code="manual_skill_options_missing",
        )
        _perform(
            lambda: page.locator("#skillOverrideSelect").select_option(
                str(option_values[0])
            ),
            stage="manual_skill",
            code="manual_skill_selection_failed",
        )
        _perform(
            lambda: page.locator("#applySkillButton").click(),
            stage="manual_skill",
            code="manual_skill_apply_failed",
        )
        _wait_for_js(
            page,
            """() => document.querySelector('#manualModeButton')
                ?.getAttribute('aria-pressed') === 'true'""",
            stage="manual_skill",
            code="manual_skill_not_confirmed",
            timeout_ms=timeout_ms,
        )
        manual_applied = True
    else:
        optional_checks["manual_skill_not_inherited"] = "skipped_provider_unavailable"

    if page.locator("#setupForm").is_hidden():
        _perform(
            lambda: page.locator("#presetButton").click(),
            stage="profile_b_replace",
            code="replacement_form_open_failed",
        )
        _wait_for_js(
            page,
            "() => !document.querySelector('#setupForm')?.hidden",
            stage="profile_b_replace",
            code="replacement_form_not_visible",
            timeout_ms=timeout_ms,
        )
    _perform(
        lambda: page.locator("#conceptInput").fill(str(concept_value)),
        stage="profile_b_replace",
        code="replacement_goal_restore_failed",
    )
    _perform(
        lambda: page.locator("[data-profile-id='zimo']").click(),
        stage="profile_b_replace",
        code="profile_b_selection_failed",
    )
    _wait_for_js(
        page,
        """() => document.querySelector("[data-profile-id='zimo']")
            ?.getAttribute('aria-checked') === 'true'""",
        stage="profile_b_replace",
        code="profile_b_selection_not_applied",
        timeout_ms=timeout_ms,
    )
    profile_b_preset_mastery = _perform(
        lambda: _form_mastery(page),
        stage="profile_b_replace",
        code="profile_b_mastery_unreadable",
    )
    _require(
        not _same_mastery(profile_a_mastery, profile_b_preset_mastery),
        stage="profile_b_replace",
        code="profile_b_mastery_not_distinct",
    )
    if page.locator("#preferencesInput").is_hidden():
        _perform(
            lambda: page.locator("details:has(#preferencesInput) > summary").click(),
            stage="profile_b_replace",
            code="profile_b_state_group_open_failed",
        )
        _wait_for_js(
            page,
            "() => Boolean(document.querySelector('#preferencesInput')?.offsetParent)",
            stage="profile_b_replace",
            code="profile_b_state_group_not_visible",
            timeout_ms=timeout_ms,
        )
    profile_b_ring_before = _perform(
        lambda: (
            page.locator(
                "[data-profile-id='zimo'] .profile-mini-ring b"
            ).text_content(),
            page.locator("[data-profile-id='zimo'] .profile-mini-ring").get_attribute(
                "aria-label"
            ),
        ),
        stage="profile_b_mastery_edit",
        code="profile_b_mastery_ring_unreadable",
    )
    conceptual_range = page.locator("#initialConceptual")

    def customize_profile_b_mastery() -> None:
        conceptual_range.focus()
        conceptual_range.press("Home")
        for _ in range(_PROFILE_B_CUSTOM_MASTERY_PERCENT // 5):
            conceptual_range.press("ArrowRight")

    _perform(
        customize_profile_b_mastery,
        stage="profile_b_mastery_edit",
        code="profile_b_mastery_range_interaction_failed",
    )
    profile_b_mastery = _perform(
        lambda: _form_mastery(page),
        stage="profile_b_mastery_edit",
        code="profile_b_custom_mastery_unreadable",
    )
    _require(
        abs(
            profile_b_mastery[_PROFILE_B_CUSTOM_MASTERY_FIELD]
            - _PROFILE_B_CUSTOM_MASTERY_VALUE
        )
        <= 0.001
        and not _same_mastery(profile_b_preset_mastery, profile_b_mastery),
        stage="profile_b_mastery_edit",
        code="profile_b_mastery_range_did_not_change",
    )
    expected_profile_b_ring_score = int(
        sum(profile_b_mastery.values()) * 100 / max(1, len(profile_b_mastery)) + 0.5
    )
    _wait_for_js(
        page,
        """({score}) => {
            const ring = document.querySelector(
                "[data-profile-id='zimo'] .profile-mini-ring"
            );
            return Boolean(ring
                && ring.querySelector('b')?.textContent.trim() === `${score}%`
                && ring.getAttribute('aria-label')
                    === `综合初始掌握度 ${score}%`);
        }""",
        arg={"score": expected_profile_b_ring_score},
        stage="profile_b_mastery_edit",
        code="profile_b_mastery_ring_did_not_update_live",
        timeout_ms=timeout_ms,
    )
    profile_b_ring_after = _perform(
        lambda: (
            page.locator(
                "[data-profile-id='zimo'] .profile-mini-ring b"
            ).text_content(),
            page.locator("[data-profile-id='zimo'] .profile-mini-ring").get_attribute(
                "aria-label"
            ),
        ),
        stage="profile_b_mastery_edit",
        code="profile_b_updated_mastery_ring_unreadable",
    )
    _require(
        profile_b_ring_after
        == (
            f"{expected_profile_b_ring_score}%",
            f"综合初始掌握度 {expected_profile_b_ring_score}%",
        )
        and profile_b_ring_after != profile_b_ring_before,
        stage="profile_b_mastery_edit",
        code="profile_b_mastery_ring_or_aria_label_stale",
    )
    checks["profile_b_mastery_range_customized"] = True
    checks["profile_b_mastery_ring_updated_live"] = True
    profile_b_preferences = "先分步检查，再让我用自己的话复述"
    profile_b_misconception = "容易把相关术语当成当前问题的直接答案"
    profile_b_background = "曾完成过一个相邻概念的入门练习"
    _perform(
        lambda: page.locator("#preferencesInput").fill(profile_b_preferences),
        stage="profile_b_replace",
        code="profile_b_preferences_input_failed",
    )
    _perform(
        lambda: page.locator("#knownMisconceptionsInput").fill(profile_b_misconception),
        stage="profile_b_replace",
        code="profile_b_misconception_input_failed",
    )
    _perform(
        lambda: page.locator("#historyInput").fill(profile_b_background),
        stage="profile_b_replace",
        code="profile_b_background_input_failed",
    )
    auto_before_replacement = (
        page.locator("#autoModeButton").get_attribute("aria-pressed") == "true"
    )
    telemetry.external_session_advance_completed = _advance_active_session_outside_ui(
        page,
        timeout_ms=timeout_ms,
    )
    _require(
        telemetry.external_session_advance_completed,
        stage="profile_b_stale_retry",
        code="external_session_advance_not_confirmed",
    )
    checks["external_session_advanced_before_replacement"] = True
    _perform(
        lambda: page.locator("#startButton").click(),
        stage="profile_b_replace",
        code="profile_b_start_click_failed",
    )
    _wait_for_js(
        page,
        """({name, roundText}) => {
            const active = document.querySelector('#activeSession');
            const setup = document.querySelector('#setupForm');
            const student = document.querySelector('#conversationStudentName');
            const round = document.querySelector('#roundCounter');
            const historyCount = document.querySelector('#historyCount');
            return Boolean(active && !active.hidden && setup && setup.hidden
                && student && student.textContent.includes(name)
                && round && round.textContent.trim() === roundText
                && historyCount && historyCount.textContent.trim() === '0 轮');
        }""",
        arg={"name": "子墨", "roundText": "R0"},
        stage="profile_b_replace",
        code="profile_b_replacement_not_started",
        timeout_ms=timeout_ms,
    )
    checks["profile_b_replacement_started"] = True
    profile_b_rendered = _perform(
        lambda: _rendered_mastery(page),
        stage="profile_b_replace",
        code="profile_b_state_unreadable",
    )
    _require(
        _same_mastery(profile_b_mastery, profile_b_rendered),
        stage="profile_b_replace",
        code="profile_b_initial_state_mismatch",
    )
    checks["profile_b_custom_mastery_materialized"] = True
    profile_b_handle = _perform(
        lambda: page.evaluate(
            """() => sessionStorage.getItem(
                'teachlab_opaque_session_handle_v2'
            ) || ''"""
        ),
        stage="profile_b_replace",
        code="profile_b_session_handle_unreadable",
    )
    _require(
        isinstance(profile_b_handle, str)
        and bool(profile_b_handle)
        and profile_b_handle != profile_a_handle,
        stage="profile_b_replace",
        code="profile_b_session_handle_not_isolated",
    )
    checks["profile_b_identity_and_mastery_isolated"] = True
    _require(
        telemetry.replacement_start_requests >= 1
        and telemetry.request_capture_failures == 0,
        stage="profile_b_replace",
        code="replacement_request_not_observed",
    )
    checks["replacement_request_observed"] = True
    _require(
        telemetry.replacement_start_requests == 2,
        stage="profile_b_stale_retry",
        code="replacement_retry_count_was_not_exactly_one",
    )
    checks["stale_replacement_retried_exactly_once"] = True
    _require(
        telemetry.expected_stale_replacement_rejections == 1,
        stage="profile_b_stale_retry",
        code="expected_stale_replacement_rejection_not_observed",
    )
    checks["stale_replacement_rejection_observed"] = True
    _require(
        len(telemetry.replacement_guard_snapshots) == 2
        and telemetry.replacement_guard_snapshots[1][0]
        == telemetry.replacement_guard_snapshots[0][0] + 1
        and telemetry.replacement_guard_snapshots[1][2]
        == telemetry.replacement_guard_snapshots[0][2] + 1
        and telemetry.replacement_guard_snapshots[1][1]
        != telemetry.replacement_guard_snapshots[0][1],
        stage="profile_b_stale_retry",
        code="replacement_retry_did_not_use_synchronized_guards",
    )
    checks["stale_replacement_used_synchronized_guards"] = True
    _require(
        len(telemetry.replacement_idempotency_keys) == 2
        and len(set(telemetry.replacement_idempotency_keys)) == 2,
        stage="profile_b_stale_retry",
        code="replacement_retry_reused_idempotency_key",
    )
    checks["stale_replacement_used_fresh_idempotency_key"] = True
    _require(
        not telemetry.replacement_request_had_manual_skill,
        stage="profile_b_replace",
        code="manual_skill_leaked_into_replacement_request",
    )
    checks["replacement_request_has_no_manual_skill"] = True
    _require(
        telemetry.replacement_request_had_version_guards,
        stage="profile_b_replace",
        code="replacement_request_missing_version_guards",
    )
    checks["replacement_request_has_version_guards"] = True
    _require(
        telemetry.replacement_request_had_profile_roundtrip_fields,
        stage="profile_b_replace",
        code="replacement_request_missing_profile_roundtrip_fields",
    )
    checks["replacement_request_has_profile_roundtrip_fields"] = True
    _require(
        telemetry.replacement_request_had_custom_mastery,
        stage="profile_b_replace",
        code="replacement_request_missing_custom_mastery",
    )
    checks["replacement_request_has_custom_mastery"] = True
    if manual_applied:
        optional_checks["manual_skill_not_inherited"] = (
            "passed"
            if auto_before_replacement
            and page.locator("#autoModeButton").get_attribute("aria-pressed") == "true"
            else "failed"
        )

    _perform(
        lambda: page.locator("#learnerResponse").fill(
            "我先指出当前问题里的关键信息，再说明它们与教学目标的关系。"
        ),
        stage="profile_b_turn",
        code="profile_b_learner_response_input_failed",
    )
    if page.locator("#fallbackSignalField").is_visible():
        _perform(
            lambda: page.locator("#fallbackSignalInput").select_option("partial"),
            stage="profile_b_turn",
            code="profile_b_fallback_signal_selection_failed",
        )
    _perform(
        lambda: page.locator("#stepButton").click(),
        stage="profile_b_turn",
        code="profile_b_turn_submit_click_failed",
    )
    _wait_for_js(
        page,
        """() => {
            const round = document.querySelector('#roundCounter');
            const historyCount = document.querySelector('#historyCount');
            return Boolean(round && round.textContent.trim() === 'R1'
                && historyCount && historyCount.textContent.trim() === '1 轮');
        }""",
        stage="profile_b_turn",
        code="profile_b_turn_not_committed_after_replacement",
        timeout_ms=timeout_ms,
    )
    checks["profile_b_turn_committed_after_replacement"] = True

    _perform(
        lambda: page.reload(wait_until="domcontentloaded", timeout=timeout_ms),
        stage="refresh_resume",
        code="page_refresh_failed",
    )
    _wait_for_js(
        page,
        """({name, roundText}) => {
            const active = document.querySelector('#activeSession');
            const setup = document.querySelector('#setupForm');
            const student = document.querySelector('#conversationStudentName');
            const round = document.querySelector('#roundCounter');
            const historyCount = document.querySelector('#historyCount');
            return Boolean(active && !active.hidden && setup && setup.hidden
                && student && student.textContent.includes(name)
                && round && round.textContent.trim() === roundText
                && historyCount && historyCount.textContent.trim() === '1 轮');
        }""",
        arg={"name": "子墨", "roundText": "R1"},
        stage="refresh_resume",
        code="profile_b_session_not_resumed",
        timeout_ms=timeout_ms,
    )
    checks["refresh_resumed_profile_b"] = True
    refreshed_handle = _perform(
        lambda: page.evaluate(
            """() => sessionStorage.getItem(
                'teachlab_opaque_session_handle_v2'
            ) || ''"""
        ),
        stage="refresh_resume",
        code="resumed_session_handle_unreadable",
    )
    _require(
        refreshed_handle == profile_b_handle,
        stage="refresh_resume",
        code="refresh_changed_opaque_session_handle",
    )
    checks["refresh_preserved_opaque_session_handle"] = True
    _perform(
        lambda: page.locator("#presetButton").click(),
        stage="refresh_resume",
        code="resumed_setup_open_failed",
    )
    _wait_for_js(
        page,
        "() => !document.querySelector('#setupForm')?.hidden",
        stage="refresh_resume",
        code="resumed_setup_not_visible",
        timeout_ms=timeout_ms,
    )
    restored_mastery = _perform(
        lambda: _form_mastery(page),
        stage="refresh_resume",
        code="resumed_profile_mastery_unreadable",
    )
    restored_goal = _perform(
        lambda: (
            page.locator("#conceptInput").input_value(),
            page.locator("#objectiveInput").input_value(),
        ),
        stage="refresh_resume",
        code="resumed_goal_form_unreadable",
    )
    _require(
        page.locator("[data-profile-id='zimo']").get_attribute("aria-checked") == "true"
        and _same_mastery(profile_b_mastery, restored_mastery)
        and restored_goal == (concept_value, objective_value),
        stage="refresh_resume",
        code="resumed_setup_not_rehydrated",
    )
    checks["refresh_setup_form_rehydrated"] = True
    restored_profile_text = _perform(
        lambda: (
            page.locator("#preferencesInput").input_value(),
            page.locator("#knownMisconceptionsInput").input_value(),
            page.locator("#historyInput").input_value(),
        ),
        stage="refresh_resume",
        code="resumed_profile_text_unreadable",
    )
    _require(
        restored_profile_text
        == (
            profile_b_preferences,
            profile_b_misconception,
            profile_b_background,
        ),
        stage="refresh_resume",
        code="resumed_profile_text_not_rehydrated",
    )
    checks["refresh_profile_text_rehydrated"] = True

    _perform(
        lambda: page.locator("#evaluationViewButton").click(),
        stage="evaluation_view",
        code="evaluation_switch_click_failed",
    )
    _wait_for_js(
        page,
        """() => {
            const evaluation = document.querySelector('#evaluationView');
            const learning = document.querySelector('#learningView');
            const title = document.querySelector('#evaluationTitle');
            return Boolean(evaluation && !evaluation.hidden
                && learning && learning.hidden
                && title && title.textContent.trim());
        }""",
        stage="evaluation_view",
        code="evaluation_view_not_visible",
        timeout_ms=timeout_ms,
    )
    checks["evaluation_view_switch"] = True

    for width in _VIEWPORTS:
        _perform(
            lambda width=width: page.set_viewport_size({"width": width, "height": 900}),
            stage="responsive_layout",
            code="viewport_resize_failed",
        )
        _perform(
            lambda: page.keyboard.press("Escape"),
            stage="responsive_layout",
            code="drawer_close_failed",
        )
        _perform(
            lambda: page.locator("#learningViewButton").click(),
            stage="responsive_layout",
            code="learning_view_switch_failed",
        )
        _perform(
            lambda: page.wait_for_timeout(120),
            stage="responsive_layout",
            code="layout_settle_failed",
        )
        if width <= 860:
            _perform(
                lambda: page.locator("#sidebarToggle").click(),
                stage="responsive_accessibility",
                code="sidebar_drawer_open_failed",
            )
            _wait_for_js(
                page,
                """() => {
                    const panel = document.querySelector('#setupPanel');
                    return Boolean(panel
                        && !panel.hasAttribute('inert')
                        && panel.getAttribute('aria-hidden') === 'false'
                        && panel.getAttribute('role') === 'dialog'
                        && panel.getAttribute('aria-modal') === 'true'
                        && document.querySelector('.topbar')?.hasAttribute('inert')
                        && document.querySelector('#liveLoop')?.hasAttribute('inert')
                        && document.activeElement?.id === 'sidebarClose');
                }""",
                stage="responsive_accessibility",
                code="sidebar_drawer_modal_contract_failed",
                timeout_ms=timeout_ms,
            )
            _perform(
                lambda: page.keyboard.press("Tab"),
                stage="responsive_accessibility",
                code="sidebar_drawer_tab_failed",
            )
            _require(
                _perform(
                    lambda: page.evaluate(
                        "() => document.querySelector('#setupPanel')?.contains(document.activeElement) === true"
                    ),
                    stage="responsive_accessibility",
                    code="sidebar_drawer_focus_unreadable",
                ),
                stage="responsive_accessibility",
                code="sidebar_drawer_focus_escaped",
            )
            _perform(
                lambda: page.keyboard.press("Escape"),
                stage="responsive_accessibility",
                code="sidebar_drawer_escape_failed",
            )
            _wait_for_js(
                page,
                """() => {
                    const panel = document.querySelector('#setupPanel');
                    return Boolean(panel
                        && panel.hasAttribute('inert')
                        && panel.getAttribute('aria-hidden') === 'true'
                        && !panel.hasAttribute('role')
                        && !document.querySelector('.topbar')?.hasAttribute('inert')
                        && !document.querySelector('#liveLoop')?.hasAttribute('inert')
                        && document.activeElement?.id === 'sidebarToggle');
                }""",
                stage="responsive_accessibility",
                code="sidebar_drawer_close_contract_failed",
                timeout_ms=timeout_ms,
            )
            checks[f"responsive_sidebar_modal_{width}"] = True

        if width <= 1260:
            _perform(
                lambda: page.locator("#inspectorToggle").click(),
                stage="responsive_accessibility",
                code="inspector_drawer_open_failed",
            )
            _wait_for_js(
                page,
                """() => {
                    const panel = document.querySelector('#statePanel');
                    return Boolean(panel
                        && !panel.hasAttribute('inert')
                        && panel.getAttribute('aria-hidden') === 'false'
                        && panel.getAttribute('role') === 'dialog'
                        && panel.getAttribute('aria-modal') === 'true'
                        && document.querySelector('.topbar')?.hasAttribute('inert')
                        && document.querySelector('#liveLoop')?.hasAttribute('inert')
                        && document.activeElement?.id === 'inspectorClose');
                }""",
                stage="responsive_accessibility",
                code="inspector_drawer_modal_contract_failed",
                timeout_ms=timeout_ms,
            )
            _perform(
                lambda: page.locator("#drawerBackdrop").click(position={"x": 4, "y": 4}),
                stage="responsive_accessibility",
                code="inspector_drawer_backdrop_close_failed",
            )
            _wait_for_js(
                page,
                """() => {
                    const panel = document.querySelector('#statePanel');
                    return Boolean(panel
                        && panel.hasAttribute('inert')
                        && panel.getAttribute('aria-hidden') === 'true'
                        && !panel.hasAttribute('role')
                        && !document.querySelector('.topbar')?.hasAttribute('inert')
                        && !document.querySelector('#liveLoop')?.hasAttribute('inert')
                        && document.activeElement?.id === 'inspectorToggle');
                }""",
                stage="responsive_accessibility",
                code="inspector_drawer_close_contract_failed",
                timeout_ms=timeout_ms,
            )
            checks[f"responsive_inspector_modal_{width}"] = True
        else:
            _require(
                _perform(
                    lambda: page.evaluate(
                        """() => ['setupPanel', 'statePanel'].every((id) => {
                            const panel = document.getElementById(id);
                            return panel && !panel.hasAttribute('inert')
                                && !panel.hasAttribute('aria-hidden')
                                && !panel.hasAttribute('aria-modal');
                        })"""
                    ),
                    stage="responsive_accessibility",
                    code="desktop_drawer_state_unreadable",
                ),
                stage="responsive_accessibility",
                code="desktop_drawer_inert_not_cleared",
            )
            checks[f"responsive_desktop_inert_cleared_{width}"] = True

        learning_ok = _perform(
            lambda: _page_has_no_horizontal_overflow(page),
            stage="responsive_layout",
            code="learning_overflow_check_failed",
        )
        _perform(
            lambda: page.locator("#evaluationViewButton").click(),
            stage="responsive_layout",
            code="evaluation_view_switch_failed",
        )
        _perform(
            lambda: page.wait_for_timeout(120),
            stage="responsive_layout",
            code="layout_settle_failed",
        )
        evaluation_ok = _perform(
            lambda: _page_has_no_horizontal_overflow(page),
            stage="responsive_layout",
            code="evaluation_overflow_check_failed",
        )
        viewport_ok = bool(learning_ok and evaluation_ok)
        telemetry.overflow_by_viewport[width] = viewport_ok
        if not viewport_ok:
            telemetry.overflow_failures += 1
        checks[f"no_horizontal_overflow_{width}"] = viewport_ok

    checks["no_console_errors"] = (
        telemetry.console_errors == 0
        and telemetry.expected_stale_console_errors
        <= telemetry.expected_stale_replacement_rejections
    )
    checks["no_page_errors"] = telemetry.page_errors == 0
    checks["no_failed_browser_requests"] = telemetry.request_failures == 0
    checks["all_browser_requests_stayed_in_capability"] = (
        telemetry.blocked_external_requests == 0
    )
    checks["request_observation_succeeded"] = telemetry.request_capture_failures == 0
    return checks, optional_checks


def run_browser_acceptance(
    base_url: str,
    *,
    browser_name: str = "chromium",
    headless: bool = True,
    timeout_seconds: float = 120.0,
    acknowledge_remote_demo_text: bool,
) -> dict[str, Any]:
    """Run the browser acceptance and return only an aggregate receipt."""

    started_at = time.monotonic()
    telemetry = _Telemetry()
    checks: dict[str, bool] = {}
    optional_checks: dict[str, str] = {}
    browser_launched = False

    def receipt(
        *, failure_stage: str | None = None, failure_code: str | None = None
    ) -> dict[str, Any]:
        return _receipt(
            checks=checks,
            optional_checks=optional_checks,
            telemetry=telemetry,
            browser_name=browser_name,
            headless=headless,
            duration_ms=round((time.monotonic() - started_at) * 1000),
            browser_launched=browser_launched,
            acknowledgement_present=acknowledge_remote_demo_text is True,
            failure_stage=failure_stage,
            failure_code=failure_code,
        )

    if acknowledge_remote_demo_text is not True:
        return receipt(
            failure_stage="consent",
            failure_code="explicit_remote_demo_text_acknowledgement_required",
        )
    if browser_name not in _BROWSERS:
        return receipt(failure_stage="input", failure_code="invalid_browser")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 5 <= float(timeout_seconds) <= 300
    ):
        return receipt(failure_stage="input", failure_code="invalid_timeout")
    try:
        safe_base_url = _validated_base_url(base_url)
    except _AcceptanceFailure as exc:
        return receipt(failure_stage=exc.stage, failure_code=exc.code)

    try:
        sync_playwright, playwright_error = _load_playwright()
    except (ImportError, ModuleNotFoundError):
        return receipt(
            failure_stage="dependency",
            failure_code="python_playwright_not_installed",
        )

    timeout_ms = round(float(timeout_seconds) * 1000)
    browser: Any = None
    context: Any = None
    try:
        try:
            playwright_manager = sync_playwright()
            playwright = playwright_manager.__enter__()
        except Exception:
            return receipt(
                failure_stage="dependency",
                failure_code="playwright_driver_unavailable",
            )
        try:
            launch_options: dict[str, Any] = {"headless": headless}
            if browser_name == "chrome":
                launch_options["channel"] = "chrome"
            try:
                browser = playwright.chromium.launch(**launch_options)
            except playwright_error:
                return receipt(
                    failure_stage="dependency",
                    failure_code="browser_executable_unavailable",
                )
            browser_launched = True
            context = browser.new_context(
                viewport={"width": 1440, "height": 900},
                service_workers="block",
            )

            def guard_route(route: Any) -> None:
                try:
                    if _request_within_capability(safe_base_url, route.request.url):
                        route.continue_()
                        return
                except Exception:
                    pass
                telemetry.blocked_external_requests += 1
                route.abort("blockedbyclient")

            context.route("**/*", guard_route)
            page = context.new_page()

            capability_path = urlsplit(safe_base_url).path
            expected_start_path = capability_path + "api/start"
            expected_attachment_path = capability_path + "api/attachment"
            expected_step_path = capability_path + "api/step"

            def observe_console(message: Any) -> None:
                if getattr(message, "type", "") == "error":
                    message_text = str(getattr(message, "text", ""))
                    if re.search(
                        r"Failed to load resource:.*(?:status of )?400",
                        message_text,
                        re.IGNORECASE,
                    ):
                        telemetry.expected_stale_console_errors += 1
                        return
                    telemetry.console_errors += 1

            def observe_page_error(_error: Any) -> None:
                telemetry.page_errors += 1

            def observe_request_failure(_request: Any) -> None:
                telemetry.request_failures += 1

            def observe_response(response: Any) -> None:
                try:
                    request = response.request
                    parsed = urlsplit(response.url)
                    if (
                        request.method == "POST"
                        and parsed.path == expected_start_path
                        and response.status == 400
                    ):
                        telemetry.expected_stale_replacement_rejections += 1
                except Exception:
                    telemetry.request_capture_failures += 1

            def observe_request(request: Any) -> None:
                try:
                    parsed = urlsplit(request.url)
                    if request.method != "POST":
                        return
                    payload = request.post_data_json
                    if not isinstance(payload, Mapping):
                        telemetry.request_capture_failures += 1
                        return
                    if parsed.path == expected_attachment_path:
                        telemetry.attachment_requests += 1
                        telemetry.attachment_request_had_expected_payload = bool(
                            payload.get("mime_type") == "image/png"
                            and isinstance(payload.get("data_base64"), str)
                            and bool(payload.get("data_base64"))
                            and isinstance(payload.get("expected_round"), int)
                            and isinstance(payload.get("expected_question_id"), str)
                            and bool(payload.get("expected_question_id"))
                            and isinstance(payload.get("expected_context_version"), int)
                            and isinstance(payload.get("profile_revision"), str)
                            and bool(payload.get("profile_revision"))
                        )
                        return
                    if parsed.path == expected_step_path:
                        telemetry.step_requests += 1
                        attachment_ids = payload.get("attachment_ids")
                        if (
                            isinstance(attachment_ids, list)
                            and len(attachment_ids) == 1
                            and isinstance(attachment_ids[0], str)
                            and bool(attachment_ids[0])
                        ):
                            telemetry.image_step_requests += 1
                            telemetry.step_request_had_attachment_ids = True
                            telemetry.step_request_was_image_only = (
                                payload.get("learner_response") == ""
                            )
                        return
                    if parsed.path != expected_start_path:
                        return
                    telemetry.start_requests += 1
                    if payload.get("replace_session_id"):
                        telemetry.replacement_start_requests += 1
                        if payload.get("manual_skill_id"):
                            telemetry.replacement_request_had_manual_skill = True
                        required_guards = (
                            "replace_expected_round",
                            "replace_expected_question_id",
                            "replace_expected_context_version",
                            "replace_expected_profile_revision",
                        )
                        guards_valid = bool(
                            isinstance(payload.get("replace_expected_round"), int)
                            and not isinstance(
                                payload.get("replace_expected_round"), bool
                            )
                            and isinstance(
                                payload.get("replace_expected_question_id"), str
                            )
                            and bool(payload.get("replace_expected_question_id"))
                            and isinstance(
                                payload.get("replace_expected_context_version"), int
                            )
                            and not isinstance(
                                payload.get("replace_expected_context_version"), bool
                            )
                            and isinstance(
                                payload.get("replace_expected_profile_revision"), str
                            )
                            and bool(payload.get("replace_expected_profile_revision"))
                            and all(field in payload for field in required_guards)
                        )
                        telemetry.replacement_request_had_version_guards = (
                            guards_valid
                            if telemetry.replacement_start_requests == 1
                            else telemetry.replacement_request_had_version_guards
                            and guards_valid
                        )
                        if guards_valid:
                            telemetry.replacement_guard_snapshots.append(
                                (
                                    int(payload["replace_expected_round"]),
                                    str(payload["replace_expected_question_id"]),
                                    int(payload["replace_expected_context_version"]),
                                    str(payload["replace_expected_profile_revision"]),
                                )
                            )
                        replacement_key = payload.get("start_idempotency_key")
                        if isinstance(replacement_key, str) and replacement_key:
                            telemetry.replacement_idempotency_keys.append(
                                replacement_key
                            )
                        profile = payload.get("student_profile")
                        profile_roundtrip_valid = bool(
                                isinstance(profile, Mapping)
                                and isinstance(profile.get("preferences"), list)
                                and bool(profile.get("preferences"))
                                and isinstance(
                                    profile.get("known_misconceptions"), list
                                )
                                and bool(profile.get("known_misconceptions"))
                                and isinstance(profile.get("background_history"), list)
                                and bool(profile.get("background_history"))
                                and isinstance(
                                    profile.get("conversation_history"), list
                                )
                                and isinstance(profile.get("accessibility_needs"), list)
                                and isinstance(
                                    profile.get("contains_direct_identity"), bool
                                )
                        )
                        telemetry.replacement_request_had_profile_roundtrip_fields = (
                            profile_roundtrip_valid
                            if telemetry.replacement_start_requests == 1
                            else telemetry.replacement_request_had_profile_roundtrip_fields
                            and profile_roundtrip_valid
                        )
                        initial_mastery = (
                            profile.get("initial_mastery", {})
                            if isinstance(profile, Mapping)
                            else {}
                        )
                        custom_mastery_valid = bool(
                            isinstance(initial_mastery, Mapping)
                            and abs(
                                float(
                                    initial_mastery.get(
                                        _PROFILE_B_CUSTOM_MASTERY_FIELD, -1.0
                                    )
                                )
                                - _PROFILE_B_CUSTOM_MASTERY_VALUE
                            )
                            <= 0.001
                        )
                        telemetry.replacement_request_had_custom_mastery = (
                            custom_mastery_valid
                            if telemetry.replacement_start_requests == 1
                            else telemetry.replacement_request_had_custom_mastery
                            and custom_mastery_valid
                        )
                except Exception:
                    telemetry.request_capture_failures += 1

            page.on("console", observe_console)
            page.on("pageerror", observe_page_error)
            page.on("requestfailed", observe_request_failure)
            page.on("request", observe_request)
            page.on("response", observe_response)
            _perform(
                lambda: page.goto(
                    safe_base_url,
                    wait_until="domcontentloaded",
                    timeout=timeout_ms,
                ),
                stage="navigation",
                code="loopback_dashboard_unreachable",
            )
            _require(
                _request_within_capability(safe_base_url, page.url),
                stage="navigation",
                code="navigation_left_capability",
            )
            checks, optional_checks = _exercise_browser(
                page, timeout_ms=timeout_ms, telemetry=telemetry
            )
            if not all(checks.values()):
                return receipt(
                    failure_stage="verification",
                    failure_code="one_or_more_browser_checks_failed",
                )
            if any(status == "failed" for status in optional_checks.values()):
                return receipt(
                    failure_stage="verification",
                    failure_code="manual_skill_isolation_failed",
                )
            return receipt()
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
            try:
                playwright_manager.__exit__(None, None, None)
            except Exception:
                pass
    except _AcceptanceFailure as exc:
        return receipt(failure_stage=exc.stage, failure_code=exc.code)
    except Exception:
        return receipt(
            failure_stage="internal",
            failure_code="unexpected_browser_acceptance_failure",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run real-browser acceptance against an existing loopback Teaching "
            "Agent capability URL and emit only an aggregate JSON receipt."
        )
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="existing loopback capability URL (never included in output)",
    )
    parser.add_argument(
        "--browser",
        choices=sorted(_BROWSERS),
        default="chromium",
        help="Playwright Chromium or installed Google Chrome channel",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="show the browser window instead of running headless",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="per-stage timeout in seconds (5-300; default: 120)",
    )
    parser.add_argument(
        "--acknowledge-remote-demo-text",
        action="store_true",
        help=(
            "acknowledge that fixed synthetic test text may be processed by the "
            "dashboard's configured remote model provider"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = run_browser_acceptance(
        args.base_url,
        browser_name=args.browser,
        headless=not args.headed,
        timeout_seconds=args.timeout_seconds,
        acknowledge_remote_demo_text=args.acknowledge_remote_demo_text,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    if receipt["passed"]:
        return 0
    stage = receipt.get("failure", {}).get("stage")
    if stage == "consent":
        return 2
    if stage == "dependency":
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
