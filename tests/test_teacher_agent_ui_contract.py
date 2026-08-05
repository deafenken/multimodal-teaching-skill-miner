from __future__ import annotations

from html.parser import HTMLParser
import re
import struct
import unittest
from urllib.parse import urlsplit

from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.teacher_agent_dashboard import (
    AVATAR_RESOURCES,
    build_teacher_agent_dashboard_snapshot,
)


ROOT = project_root()
WEB_ROOT = ROOT / "teaching_skill_miner" / "web"
HTML_PATH = WEB_ROOT / "teacher_agent_demo.html"
STYLE_PATH = WEB_ROOT / "teacher_agent_demo.css"
SCRIPT_PATH = WEB_ROOT / "teacher_agent_demo.js"
AVATAR_ROOT = WEB_ROOT / "assets"


class _UiContractParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self.text_parts.append(value)

    def with_attr(self, name: str, value: str | None = None):
        return [
            (tag, attrs)
            for tag, attrs in self.elements
            if name in attrs and (value is None or attrs[name] == value)
        ]

    def by_id(self, element_id: str) -> tuple[str, dict[str, str | None]]:
        matches = [
            (tag, attrs)
            for tag, attrs in self.elements
            if attrs.get("id") == element_id
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"expected exactly one element with id={element_id!r}, got {len(matches)}"
            )
        return matches[0]


class TeacherAgentUiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = HTML_PATH.read_text(encoding="utf-8")
        cls.style = STYLE_PATH.read_text(encoding="utf-8")
        cls.script = SCRIPT_PATH.read_text(encoding="utf-8")
        cls.dashboard_source = (
            ROOT / "teaching_skill_miner" / "teacher_agent_dashboard.py"
        ).read_text(encoding="utf-8")
        cls.parser = _UiContractParser()
        cls.parser.feed(cls.html)

    def test_three_local_avatar_assets_are_valid_and_no_external_assets_exist(
        self,
    ) -> None:
        expected_names = {
            "student-xiaoyu.png",
            "student-zimo.png",
            "student-zhixing.png",
        }
        self.assertEqual(set(AVATAR_RESOURCES), expected_names)
        self.assertEqual(
            {path.name for path in AVATAR_ROOT.glob("student-*.png")},
            expected_names,
        )

        for name in expected_names:
            with self.subTest(avatar=name):
                payload = (AVATAR_ROOT / name).read_bytes()
                self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
                width, height = struct.unpack(">II", payload[16:24])
                self.assertGreaterEqual(width, 96)
                self.assertGreaterEqual(height, 96)

        referenced_avatars = {
            str(attrs["src"]).removeprefix("assets/")
            for tag, attrs in self.parser.elements
            if tag == "img" and str(attrs.get("src", "")).startswith("assets/student-")
        }
        self.assertEqual(referenced_avatars, expected_names)

        external_refs: list[str] = []
        for _tag, attrs in self.parser.elements:
            for attr_name in ("src", "href"):
                raw = attrs.get(attr_name)
                if not raw or str(raw).startswith("#"):
                    continue
                parsed = urlsplit(str(raw))
                if parsed.scheme or parsed.netloc or str(raw).startswith("//"):
                    external_refs.append(str(raw))
        external_refs.extend(
            value
            for value in re.findall(r"url\(\s*['\"]?([^)'\"]+)", self.style)
            if urlsplit(value).scheme or value.startswith("//")
        )
        self.assertEqual(external_refs, [])
        self.assertIn("img-src 'self'", self.html)
        self.assertIn("img-src 'self' blob:", self.html)
        self.assertIn("img-src 'self'", self.dashboard_source)

    def test_csp_meta_avoids_unsupported_frame_ancestors_and_has_local_icon(
        self,
    ) -> None:
        csp_meta = [
            attrs
            for tag, attrs in self.parser.elements
            if tag == "meta" and attrs.get("http-equiv") == "Content-Security-Policy"
        ]
        self.assertEqual(len(csp_meta), 1)
        self.assertNotIn("frame-ancestors", csp_meta[0].get("content", ""))
        icons = [
            attrs
            for tag, attrs in self.parser.elements
            if tag == "link" and attrs.get("rel") == "icon"
        ]
        self.assertEqual(len(icons), 1)
        self.assertEqual(icons[0].get("href"), "assets/student-xiaoyu.png")
        self.assertIn("frame-ancestors 'none'", self.dashboard_source)

    def test_narrow_toast_does_not_cover_the_fixed_response_composer(self) -> None:
        narrow = self.style.split("@media (max-width: 1260px)", 1)[1].split(
            "@media (max-width: 860px)", 1
        )[0]
        toast = re.search(r"\.toast\s*\{([^}]*)\}", narrow)
        self.assertIsNotNone(toast)
        rules = toast.group(1) if toast else ""
        self.assertIn("top:", rules)
        self.assertIn("bottom: auto", rules)

    def test_answer_image_controls_are_wired_to_local_evidence_api(self) -> None:
        for element_id in (
            "attachImageButton",
            "answerImageInput",
            "attachmentPreview",
            "attachmentThumbnail",
            "attachmentStatus",
            "attachmentEvidencePreview",
            "removeAttachmentButton",
        ):
            with self.subTest(element_id=element_id):
                self.parser.by_id(element_id)

        _tag, file_input = self.parser.by_id("answerImageInput")
        self.assertEqual(
            file_input.get("accept"),
            "image/png,image/jpeg,image/webp",
        )
        for required in (
            'postJson("api/attachment"',
            "attachment_idempotency_key",
            "attachment_ids: attachmentIds",
            "await file.arrayBuffer()",
            "function clearPendingAttachment()",
            "function uploadPendingAttachment()",
            "event.multimodal_evidence",
            "event.learner_text",
            "原图未发送",
            "需学生核对",
            "OCR（待核对）",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.script)
        submit = self.script.split("async function submitTurn(event)", 1)[1].split(
            "async function chooseAutoMode", 1
        )[0]
        self.assertIn("if (!learnerResponse && !hasAttachment)", submit)
        self.assertIn("const attachmentIds = await uploadPendingAttachment()", submit)
        self.assertIn("clearPendingAttachment()", submit)
        self.assertIn(".attachment-preview", self.style)
        self.assertIn(".visual-evidence-summary", self.style)
        rendered_text = " ".join(self.parser.text_parts)
        self.assertIn("原图仅本机短暂处理", rendered_text)
        self.assertIn("原图不会交给 DeepSeek", rendered_text)

    def test_mobile_header_keeps_the_student_profile_visible(self) -> None:
        mobile = self.style.split("@media (max-width: 560px)", 1)[1].split(
            "@media (prefers-reduced-motion", 1
        )[0]
        self.assertNotRegex(mobile, r"\.conversation-student\s*\{[^}]*display:\s*none")
        self.assertRegex(mobile, r"\.conversation-student img\s*\{[^}]*width:\s*30px")
        self.assertRegex(mobile, r"\.session-state\s*\{[^}]*display:\s*none")

    def test_profile_cards_form_one_explicit_synthetic_radio_group(self) -> None:
        groups = self.parser.with_attr("role", "radiogroup")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][1].get("id"), "profileCards")

        cards = [
            (tag, attrs)
            for tag, attrs in self.parser.with_attr("role", "radio")
            if attrs.get("data-profile-id")
        ]
        self.assertEqual(len(cards), 3)
        self.assertEqual({tag for tag, _attrs in cards}, {"button"})
        self.assertEqual({attrs.get("type") for _tag, attrs in cards}, {"button"})
        self.assertEqual(
            {attrs["data-profile-id"] for _tag, attrs in cards},
            {"xiaoyu", "zimo", "zhixing"},
        )
        self.assertEqual(
            sum(attrs.get("aria-checked") == "true" for _tag, attrs in cards), 1
        )
        self.assertEqual(sum(attrs.get("tabindex") == "0" for _tag, attrs in cards), 1)
        self.assertEqual(sum(attrs.get("tabindex") == "-1" for _tag, attrs in cards), 2)

        rendered_text = " ".join(self.parser.text_parts)
        for claim in ("头像由 AI 合成", "不对应真实学生", "不参与能力判断"):
            with self.subTest(claim=claim):
                self.assertIn(claim, rendered_text)
        for profile_id in ("xiaoyu", "zimo", "zhixing"):
            self.assertIn(f"{profile_id}: {{", self.script)
            self.assertRegex(
                self.script,
                rf'{profile_id}: \{{[\s\S]*?ref: "synthetic_profile_{profile_id}_v1"',
            )
        self.assertIn("function applyProfile(profileId", self.script)
        self.assertIn('setControlMode("auto")', self.script)
        apply_profile = self.script.split("function applyProfile(profileId", 1)[
            1
        ].split("function markProfileEdited", 1)[0]
        self.assertIn("card.tabIndex = selected ? 0 : -1", apply_profile)

        keyboard_handler = self.script.split("const profileCards = [", 1)[1].split(
            'for (const input of ["#preferencesInput"', 1
        )[0]
        for key in ("ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"):
            with self.subTest(profile_key=key):
                self.assertIn(key, keyboard_handler)
        self.assertIn("event.preventDefault()", keyboard_handler)
        self.assertIn("applyProfile(target.dataset.profileId", keyboard_handler)
        self.assertIn("target.focus()", keyboard_handler)

    def test_ungrounded_profile_candidate_is_visibly_marked_for_review(self) -> None:
        renderer = self.script.split(
            "function renderAdaptiveStudentProfile(session)", 1
        )[1].split("function renderRanking", 1)[0]
        self.assertIn('evidence.grounding === "no_grounded_excerpt"', renderer)
        self.assertIn("本轮未绑定到学生原话片段，必须人工复核", renderer)

    def test_profile_switching_preserves_each_in_memory_draft(self) -> None:
        capture = self.script.split("function captureProfileDraft()", 1)[1].split(
            "function restoreProfileDraft", 1
        )[0]
        restore = self.script.split("function restoreProfileDraft(profile)", 1)[
            1
        ].split("function renderDraftProfileBaseline", 1)[0]
        apply_profile = self.script.split("function applyProfile(profileId", 1)[
            1
        ].split("function markProfileEdited", 1)[0]
        for field in (
            "learnerLevel",
            "mastery",
            "preferences",
            "misconceptions",
            "backgroundHistory",
            "revision",
            "editCounter",
        ):
            with self.subTest(draft_field=field):
                self.assertIn(field, capture)
                self.assertIn(field, restore)
        self.assertIn("captureProfileDraft()", apply_profile)
        self.assertIn("restoreProfileDraft(profile)", apply_profile)
        self.assertIn("renderDraftProfileBaseline()", apply_profile)
        self.assertIn('applyProfile("xiaoyu", {resetDrafts: true})', self.script)

    def test_profile_card_ring_tracks_the_current_mastery_draft(self) -> None:
        renderer = self.script.split("function renderProfileMiniRing", 1)[1].split(
            "function splitList", 1
        )[0]
        for required in (
            'ring.style.setProperty("--profile-score", String(score))',
            'ring.setAttribute("aria-label", `综合初始掌握度 ${score}%`)',
            "label.textContent = `${score}%`",
        ):
            with self.subTest(required=required):
                self.assertIn(required, renderer)
        capture = self.script.split("function captureProfileDraft()", 1)[1].split(
            "function restoreProfileDraft", 1
        )[0]
        restore = self.script.split("function restoreProfileDraft(profile)", 1)[
            1
        ].split("function renderDraftProfileBaseline", 1)[0]
        self.assertIn("renderProfileMiniRing(profile.id, mastery)", capture)
        self.assertIn("renderProfileMiniRing(profile.id, mastery)", restore)

    def test_replacement_draft_makes_start_commit_semantics_explicit(self) -> None:
        copy_sync = self.script.split("function syncStartButtonCopy()", 1)[1].split(
            "function syncControls", 1
        )[0]
        self.assertIn('label.textContent = "开始学习"', copy_sync)
        self.assertIn('hint.textContent = "先生成一个教学动作"', copy_sync)
        self.assertIn("if (app.draftingReplacement)", copy_sync)
        self.assertIn("`切换到${selectedProfile().name}并开始`", copy_sync)
        self.assertIn(
            'hint.textContent = "成功后替换当前会话；失败仍保留旧会话"',
            copy_sync,
        )
        self.assertIn('label.textContent = "开始新会话"', copy_sync)
        sync_controls = self.script.split("function syncControls()", 1)[1].split(
            "function setRange", 1
        )[0]
        self.assertIn("syncStartButtonCopy()", sync_controls)

    def test_mastery_rings_state_their_math_and_match_profile_values(self) -> None:
        _tag, ring = self.parser.by_id("masteryRing")
        self.assertEqual(ring.get("role"), "img")
        self.assertIn("综合掌握度", str(ring.get("aria-label")))
        rendered_text = " ".join(self.parser.text_parts)
        self.assertIn("四项掌握估计的等权平均", rendered_text)
        self.assertIn("细分值不是总体的组成占比", rendered_text)
        self.assertIn("conic-gradient", self.style)
        for element_id, value, label in (
            ("prerequisiteBar", "0.25", "前置知识 25%"),
            ("conceptualBar", "0.15", "概念理解 15%"),
            ("proceduralBar", "0.1", "操作过程 10%"),
            ("transferBar", "0.05", "迁移能力 5%"),
        ):
            with self.subTest(initial_mastery_bar=element_id):
                _tag, attrs = self.parser.by_id(element_id)
                self.assertEqual(attrs.get("value"), value)
                self.assertEqual(attrs.get("aria-label"), label)

        renderer = self.script.split("function renderMastery(state", 1)[1].split(
            "function renderMisconceptions", 1
        )[0]
        self.assertIn("masteryValues.reduce", renderer)
        self.assertIn("/ Math.max(1, masteryValues.length)", renderer)
        self.assertIn("Math.round(average * 100)", renderer)
        self.assertIn("四项等权平均", renderer)

        expected_scores = {"xiaoyu": 14, "zimo": 33, "zhixing": 55}
        for profile_id, expected_score in expected_scores.items():
            with self.subTest(profile=profile_id):
                card_pattern = re.compile(
                    rf'data-profile-id="{profile_id}"[\s\S]*?'
                    rf"--profile-score:\s*{expected_score}\b[\s\S]*?"
                    rf"综合初始掌握度\s+{expected_score}%[\s\S]*?"
                    rf"<b>{expected_score}%</b>",
                )
                self.assertRegex(self.html, card_pattern)

    def test_inspector_has_state_method_and_evidence_tabs_and_panels(self) -> None:
        tabs = self.parser.with_attr("role", "tab")
        self.assertEqual(len(tabs), 3)
        self.assertEqual(
            {attrs.get("data-inspector-tab") for _tag, attrs in tabs},
            {"state", "method", "evidence"},
        )
        self.assertEqual(
            sum(attrs.get("aria-selected") == "true" for _tag, attrs in tabs), 1
        )
        self.assertEqual(sum(attrs.get("tabindex") == "0" for _tag, attrs in tabs), 1)
        self.assertEqual(sum(attrs.get("tabindex") == "-1" for _tag, attrs in tabs), 2)
        panels = self.parser.with_attr("data-inspector-panel")
        self.assertEqual(
            {attrs.get("data-inspector-panel") for _tag, attrs in panels},
            {"state", "method", "evidence"},
        )
        panel_ids = {str(attrs.get("id")) for _tag, attrs in panels}
        self.assertNotIn("None", panel_ids)
        self.assertEqual(len(panel_ids), len(panels))
        for _tag, panel in panels:
            with self.subTest(panel=panel.get("id")):
                self.assertEqual(panel.get("role"), "tabpanel")
                labelled_by = str(panel.get("aria-labelledby"))
                controlling_tab = self.parser.by_id(labelled_by)[1]
                self.assertEqual(controlling_tab.get("role"), "tab")
                self.assertEqual(
                    panel.get("data-inspector-panel"),
                    controlling_tab.get("data-inspector-tab"),
                )
                self.assertIn(
                    str(panel.get("id")),
                    str(controlling_tab.get("aria-controls", "")).split(),
                )
        controlled_ids = {
            controlled_id
            for _tag, tab in tabs
            for controlled_id in str(tab.get("aria-controls", "")).split()
        }
        self.assertEqual(controlled_ids, panel_ids)
        self.assertIn("function setInspectorTab(tabName)", self.script)
        self.assertIn(
            "panel.hidden = panel.dataset.inspectorPanel !== selectedTab", self.script
        )
        tab_keyboard_handler = self.script.split("const inspectorTabs = [", 1)[1].split(
            'select("#setupForm").addEventListener', 1
        )[0]
        for key in ("ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"):
            with self.subTest(tab_key=key):
                self.assertIn(key, tab_keyboard_handler)
        self.assertIn("event.preventDefault()", tab_keyboard_handler)
        self.assertIn(
            "setInspectorTab(target.dataset.inspectorTab)", tab_keyboard_handler
        )
        self.assertIn("target.focus()", tab_keyboard_handler)

    def test_quick_feedback_is_visible_and_populates_the_composer(self) -> None:
        quick_buttons = self.parser.with_attr("data-quick-response")
        self.assertEqual(len(quick_buttons), 4)
        values = {attrs["data-quick-response"] for _tag, attrs in quick_buttons}
        for expected in ("换一个", "只提示", "没有听懂", "误解了我的意思"):
            with self.subTest(expected=expected):
                self.assertTrue(any(expected in str(value) for value in values))
        self.assertIn(
            'for (const button of document.querySelectorAll("[data-quick-response]"))',
            self.script,
        )
        self.assertIn("textarea.value = button.dataset.quickResponse", self.script)
        self.assertIn("textarea.focus()", self.script)
        self.assertRegex(
            self.style,
            r"\.quick-prompts button\s*\{[^}]*min-height:\s*44px;[^}]*font-size:\s*12px;",
        )

    def test_busy_state_locks_every_setup_control_and_restores_prior_state(
        self,
    ) -> None:
        controls = self.script.split("function syncControls()", 1)[1].split(
            "function setRange", 1
        )[0]
        self.assertIn(
            'select("#setupForm").querySelectorAll("input, textarea, select, button")',
            controls,
        )
        self.assertIn(
            "control.dataset.busyDisabled = String(control.disabled)", controls
        )
        self.assertIn("control.disabled = true", controls)
        self.assertIn(
            'control.disabled = control.dataset.busyDisabled === "true"', controls
        )
        self.assertIn("delete control.dataset.busyDisabled", controls)

    def test_manual_skill_start_and_resume_follow_server_control_state(self) -> None:
        start_session = self.script.split("async function startSession(event)", 1)[
            1
        ].split("async function sendCommand", 1)[0]
        manual_payload = (
            'if (commandControlsSupported() && app.controlMode === "manual" '
            "&& app.manualSkillId)"
        )
        self.assertIn(manual_payload, start_session)
        self.assertIn("startPayload.manual_skill_id = app.manualSkillId", start_session)
        self.assertLess(
            start_session.index("startPayload.manual_skill_id = app.manualSkillId"),
            start_session.index("const startSignature = JSON.stringify(startPayload)"),
        )
        self.assertIn("synchronizeControlModeFromSession()", start_session)
        self.assertNotIn('setControlMode("auto")', start_session)

        synchronize = self.script.split(
            "function synchronizeControlModeFromSession()", 1
        )[1].split("async function synchronizeCurrentSession", 1)[0]
        self.assertIn("app.manualDraftOpen = false", synchronize)
        self.assertIn("app.session?.pending_skill_id", synchronize)
        self.assertIn(
            'setControlMode("manual", app.session.pending_skill_id)', synchronize
        )
        self.assertIn('setControlMode("auto")', synchronize)

        current_session = self.script.split(
            "async function synchronizeCurrentSession(", 1
        )[1].split("async function submitTurn", 1)[0]
        self.assertIn(
            'app.session = await postJson("api/session", {session_id: sessionId})',
            current_session,
        )
        self.assertIn("synchronizeControlModeFromSession()", current_session)

        restore_session = self.script.split("async function restoreSession()", 1)[
            1
        ].split("async function init()", 1)[0]
        self.assertIn(
            'app.session = await postJson("api/session", {session_id: sessionId})',
            restore_session,
        )
        self.assertIn("synchronizeControlModeFromSession()", restore_session)

    def test_active_skill_commands_commit_local_state_only_after_server_success(
        self,
    ) -> None:
        send_command = self.script.split("async function sendCommand", 1)[1].split(
            "async function applySkillOverride", 1
        )[0]
        request = 'updatedSession = await postJson("api/command"'
        commit = "app.session = updatedSession"
        synchronize = "synchronizeControlModeFromSession()"
        self.assertIn(request, send_command)
        self.assertIn(commit, send_command)
        self.assertLess(send_command.index(request), send_command.index(commit))
        self.assertLess(send_command.index(commit), send_command.index(synchronize))
        self.assertNotIn("setControlMode(", send_command)
        self.assertIn(
            "await synchronizeCurrentSession({preserveOnFailure: true})",
            send_command,
        )
        current_session = self.script.split(
            "async function synchronizeCurrentSession(", 1
        )[1].split("async function submitTurn", 1)[0]
        self.assertIn("isUnavailableSessionError(error)", current_session)
        self.assertIn("preserveOnFailure && !unavailable", current_session)
        self.assertIn("discardUnavailableSession()", current_session)

        skill_override = self.script.split("async function applySkillOverride()", 1)[
            1
        ].split("function findCommandSkill", 1)[0]
        no_session, active_session = skill_override.split("setBusy(true);", 1)
        self.assertIn('setControlMode("manual", skillId)', no_session)
        before_request = active_session.split(
            'await sendCommand("select_skill", skillId)', 1
        )[0]
        self.assertNotIn("setControlMode(", before_request)
        self.assertIn("synchronizeControlModeFromSession()", active_session)

    def test_assessment_provenance_distinguishes_model_contract_and_fallback(
        self,
    ) -> None:
        renderer = self.script.split("function renderAssessment(session)", 1)[1].split(
            "function renderGoalPlan", 1
        )[0]
        for source in (
            "deepseek_v4_flash",
            "deepseek_v4_flash_constrained_by_deterministic_contract",
            "active_question_contract_exact_match",
            "deterministic_safety_fallback",
        ):
            with self.subTest(source=source):
                self.assertIn(source, renderer)
        for label in (
            "DEEPSEEK ASSESSMENT",
            "DEEPSEEK + CONTRACT GUARD",
            "ACTIVE CONTRACT EXACT MATCH",
            "SAFETY FALLBACK SIGNAL",
        ):
            with self.subTest(label=label):
                self.assertIn(label, renderer)
        self.assertIn("本问契约精确命中", renderer)
        self.assertIn("约束层修正 · 建议人工确认", renderer)
        self.assertIn("lowConfidence", renderer)
        self.assertIn("modelBackedAssessment ? probability", renderer)
        self.assertIn("等待诊断来源", " ".join(self.parser.text_parts))
        for label in (
            "等待学生回答 · 尚无诊断",
            "在线模型诊断 · DeepSeek",
            "在线模型诊断 · 契约约束",
            "确定性契约 · 精确命中",
            "安全规则回退 · 非模型",
            "结构化演示信号 · 非模型",
        ):
            with self.subTest(displayed_source_label=label):
                self.assertIn(label, self.script)
        self.assertIn('!history.length\n      ? "AWAITING STUDENT RESPONSE"', renderer)

    def test_history_audit_distinguishes_model_proposal_and_final_execution(
        self,
    ) -> None:
        renderer = self.script.split("function renderHistory(history)", 1)[1].split(
            "function renderRuntime", 1
        )[0]
        for required in (
            "model_proposed_primary_skill_id",
            "primary_skill_was_retargeted",
            "decision_origin",
            "manual_override_applied",
            "模型提议",
            "最终执行",
            "路由审计：",
        ):
            with self.subTest(required=required):
                self.assertIn(required, renderer)

    def test_context_inspector_surfaces_evidence_linked_teaching_checkpoints(
        self,
    ) -> None:
        renderer = self.script.split("function renderContextMemory", 1)[1].split(
            "function masteryDeltaNodes", 1
        )[0]
        for required in (
            "teaching_checkpoints",
            "unresolved_learning_signal",
            "explicit_learner_question",
            "explicit_learner_preference_or_constraint",
            "verified_prerequisite",
            "teacher_next_step_statement",
            "证据检查点：",
        ):
            with self.subTest(required=required):
                self.assertIn(required, renderer)

    def test_benchmark_copy_distinguishes_shared_taxonomy_from_live_prompt(
        self,
    ) -> None:
        renderer = self.script.split("function renderFreeTextBenchmark", 1)[1].split(
            "function outcomeProportion", 1
        )[0]
        self.assertIn("DEV EVAL", renderer)
        self.assertIn("单轮评测", renderer)
        self.assertIn("实时会话", renderer)
        self.assertIn("共享诊断语义量表", renderer)
        self.assertIn("不是完整 live Session 评测", renderer)
        self.assertNotIn("HISTORICAL", renderer)
        self.assertNotIn("尚未重跑", renderer)
        self.assertIn("在线单轮开发评测", " ".join(self.parser.text_parts))

    def test_profile_replacement_surfaces_validated_safety_fallback(self) -> None:
        start = self.script.split("async function startSession", 1)[1].split(
            "async function sendCommand", 1
        )[0]
        self.assertIn("startedWithFallback", start)
        self.assertIn("新画像会话已建立", start)
        self.assertIn("可审计的安全规则动作", start)

    def test_free_text_prior_context_is_unlabeled_and_split_only_by_line(
        self,
    ) -> None:
        setup = self.script.split("function setupPayload()", 1)[1].split(
            "function populateSkillSelect", 1
        )[0]
        split_lines = self.script.split("function splitLines(value)", 1)[1].split(
            "function profileByReference", 1
        )[0]
        self.assertIn('splitLines(select("#historyInput").value)', setup)
        self.assertIn("conversation_history: extras.conversationHistory", setup)
        self.assertIn("background_history: backgroundHistory", setup)
        self.assertIn("accessibility_needs: extras.accessibilityNeeds", setup)
        self.assertIn("contains_direct_identity: extras.containsDirectIdentity", setup)
        self.assertNotIn('signal: "partial"', setup)
        self.assertNotIn('focus_dimension: "conceptual"', setup)
        self.assertIn(".split(/\\r?\\n/)", split_lines)
        self.assertNotIn("，", split_lines)
        rendered_text = " ".join(self.parser.text_parts)
        self.assertIn("既往上下文（未标注）", rendered_text)

        text_commands = self.script.split("async function maybeRunTextCommand", 1)[
            1
        ].split("function synchronizeControlModeFromSession", 1)[0]
        auto_command = text_commands.split('if (commandText === "/auto")', 1)[1].split(
            'if (commandText === "/stop")', 1
        )[0]
        self.assertIn('await sendCommand("auto")', auto_command)
        self.assertNotIn("setControlMode(", auto_command)
        skill_command = text_commands.split(
            'if (commandText.startsWith("/+skill"))', 1
        )[1]
        self.assertIn(
            'await sendCommand("select_skill", match.skill_id)', skill_command
        )
        self.assertNotIn("setControlMode(", skill_command)
        self.assertNotIn("app.manualSkillId =", skill_command)

        auto_mode = self.script.split("async function chooseAutoMode()", 1)[1].split(
            "function applyTweaks", 1
        )[0]
        active_auto_mode = auto_mode.split("setBusy(true);", 1)[1]
        self.assertIn('await sendCommand("auto")', active_auto_mode)
        self.assertNotIn("setControlMode(", active_auto_mode)

    def test_active_manual_editor_is_a_draft_until_apply_succeeds(self) -> None:
        bindings = self.script.split(
            'select("#manualModeButton").addEventListener("click"', 1
        )[1].split('select("#fallbackSignalInput").addEventListener', 1)[0]
        active_manual = bindings.split(
            'if (app.session?.status === "active" && !app.draftingReplacement)',
            1,
        )[1].split("} else {", 1)[0]
        self.assertIn("app.manualDraftOpen = true", active_manual)
        self.assertNotIn("setControlMode(", active_manual)
        self.assertIn("服务端确认前仍保持当前控制模式", active_manual)

        select_change = bindings.split(
            'select("#skillOverrideSelect").addEventListener("change"', 1
        )[1]
        active_change = select_change.split(
            'if (app.session?.status === "active" && !app.draftingReplacement)',
            1,
        )[1].split("} else {", 1)[0]
        self.assertIn("app.manualDraftOpen = true", active_change)
        self.assertNotIn("setControlMode(", active_change)
        self.assertNotIn("app.manualSkillId =", active_change)

    def test_preselected_manual_skill_copy_matches_first_answer_semantics(self) -> None:
        self.assertIn(
            "首个学生回答后开始锁定，该选择不会改动旧会话",
            self.script,
        )
        self.assertIn(
            "开始会话后，首个学生回答将按该 Skill 路由",
            self.script,
        )
        self.assertNotIn("新会话将从", self.script)

    def test_start_and_turn_errors_are_persistent_inline_alerts(self) -> None:
        for element_id in ("setupError", "turnError"):
            with self.subTest(element_id=element_id):
                _tag, attrs = self.parser.by_id(element_id)
                self.assertEqual(attrs.get("role"), "alert")
                self.assertIn("hidden", attrs)
        show_error = self.script.split("function showInlineError", 1)[1].split(
            "function clearInlineError", 1
        )[0]
        self.assertIn("error.textContent = message", show_error)
        self.assertIn("error.hidden = false", show_error)
        self.assertNotIn("setTimeout", show_error)
        self.assertIn('showInlineError("#setupError", `无法开始：', self.script)
        self.assertIn('showInlineError("#turnError", `无法提交：', self.script)

    def test_unavailable_session_error_moves_to_the_visible_setup_surface(self) -> None:
        helper = self.script.split("function showUnavailableSessionNotice", 1)[1].split(
            "function persistSessionHandle", 1
        )[0]
        self.assertIn("showSetupForm(true)", helper)
        self.assertIn('showInlineError("#setupError", message)', helper)
        self.assertIn("showToast(message)", helper)
        self.assertIn("error.focus()", helper)
        submit = self.script.split("async function submitTurn", 1)[1].split(
            "async function chooseAutoMode", 1
        )[0]
        self.assertIn("showUnavailableSessionNotice(", submit)
        self.assertNotIn('showInlineError("#turnError", "原会话已结束或被替换', submit)

    def test_stale_profile_replacement_is_cleared_and_retried_once(self) -> None:
        helper = self.script.split("function isUnavailableSessionError", 1)[1].split(
            "function probability", 1
        )[0]
        self.assertIn("replace_session_id", helper)
        self.assertIn("does not match an available session", helper)
        self.assertIn("changed while the new session was prepared", helper)
        self.assertIn("app.session = null", helper)
        self.assertIn("clearSessionHandle()", helper)
        self.assertIn('setControlMode("auto")', helper)

        start_session = self.script.split("async function startSession(event)", 1)[
            1
        ].split("async function sendCommand", 1)[0]
        self.assertIn("for (let attempt = 0; attempt < 2; attempt += 1)", start_session)
        self.assertIn(
            "isUnavailableSessionError(error, {replacement: true})", start_session
        )
        self.assertIn("discardUnavailableSession({render: false})", start_session)
        self.assertIn("delete startPayload.replace_session_id", start_session)
        for guard in (
            "replace_expected_round",
            "replace_expected_question_id",
            "replace_expected_context_version",
            "replace_expected_profile_revision",
        ):
            with self.subTest(replacement_guard=guard):
                self.assertIn(f"{guard}:", start_session)
                self.assertIn(f"delete startPayload.{guard}", start_session)
        self.assertIn("delete startPayload.start_idempotency_key", start_session)
        self.assertIn(
            "if (attempt > 0 || (!staleReplacement && !staleReplacementGuards))",
            start_session,
        )
        self.assertIn(
            "if (recoveredFromUnavailableSession) discardUnavailableSession()",
            start_session,
        )
        self.assertLess(
            start_session.index("discardUnavailableSession({render: false})"),
            start_session.index("delete startPayload.replace_session_id"),
        )

    def test_replacement_guard_mismatch_resumes_and_rebinds_once(self) -> None:
        helper = self.script.split("function isReplacementGuardMismatch", 1)[1].split(
            "function discardUnavailableSession", 1
        )[0]
        for guard in (
            "round",
            "question_id",
            "context_version",
            "profile_revision",
        ):
            with self.subTest(replacement_guard=guard):
                self.assertIn(guard, helper)
        self.assertIn("does not match this session", helper)

        start_session = self.script.split("async function startSession(event)", 1)[
            1
        ].split("async function sendCommand", 1)[0]
        self.assertIn("staleReplacementGuards", start_session)
        self.assertIn(
            "await synchronizeCurrentSession({preserveOnFailure: true})",
            start_session,
        )
        self.assertIn("captureProfileDraft()", start_session)
        self.assertIn("app.pendingStart = null", start_session)
        self.assertIn("delete startPayload.start_idempotency_key", start_session)
        self.assertIn("replace_session_id: app.session.session_id", start_session)
        self.assertIn("replace_expected_round:", start_session)
        self.assertIn("replace_expected_question_id: synchronizedQuestionId", start_session)
        self.assertIn("replace_expected_context_version:", start_session)
        self.assertIn(
            "replace_expected_profile_revision: synchronizedRevision",
            start_session,
        )
        self.assertIn("recoveredFromConcurrentUpdate = true", start_session)
        self.assertIn(
            "if (attempt > 0 || (!staleReplacement && !staleReplacementGuards))",
            start_session,
        )
        self.assertLess(
            start_session.index("captureProfileDraft()"),
            start_session.index(
                "await synchronizeCurrentSession({preserveOnFailure: true})"
            ),
        )
        self.assertLess(
            start_session.index("delete startPayload.start_idempotency_key"),
            start_session.index("recoveredFromConcurrentUpdate = true"),
        )

    def test_setup_snapshot_round_trips_non_visual_profile_fields_per_student(
        self,
    ) -> None:
        draft_helpers = self.script.split("function profileDraftExtras", 1)[1].split(
            "function renderProfileIdentity", 1
        )[0]
        snapshot = self.script.split("function applySetupSnapshot", 1)[1].split(
            "function fillSetupForm", 1
        )[0]
        for camel, snake in (
            ("conversationHistory", "conversation_history"),
            ("accessibilityNeeds", "accessibility_needs"),
            ("containsDirectIdentity", "contains_direct_identity"),
        ):
            with self.subTest(profile_extra=camel):
                self.assertIn(camel, draft_helpers)
                self.assertIn(camel, snapshot)
                self.assertIn(snake, snapshot)
        self.assertIn("profile.containsDirectIdentity === true", draft_helpers)
        self.assertIn("profile.contains_direct_identity === true", snapshot)

    def test_session_storage_contains_only_one_opaque_session_handle(self) -> None:
        accesses = re.findall(
            r"window\.sessionStorage\.(setItem|getItem|removeItem)\(([^;\n]+)\)",
            self.script,
        )
        self.assertEqual(len(accesses), 4)
        for operation, arguments in accesses:
            with self.subTest(operation=operation, arguments=arguments):
                self.assertTrue(arguments.strip().startswith("sessionHandleKey"))
        set_calls = [
            arguments for operation, arguments in accesses if operation == "setItem"
        ]
        self.assertEqual(set_calls, ["sessionHandleKey, app.session.session_id"])
        self.assertIn(
            'const sessionHandleKey = "teachlab_opaque_session_handle_v2"',
            self.script,
        )
        self.assertNotRegex(
            self.script, r"sessionStorage[^\n]*(?:JSON|stringify|history|goal|profile)"
        )

        local_storage_keys = set(
            re.findall(r'localStorage\.(?:setItem|getItem)\("([^"]+)"', self.script)
        )
        self.assertEqual(local_storage_keys, {"teacher_agent_display_v1"})

    def test_api_session_resumes_the_exact_opaque_handle(self) -> None:
        self.assertIn("async function restoreSession()", self.script)
        self.assertIn(
            'app.session = await postJson("api/session", {session_id: sessionId})',
            self.script,
        )
        self.assertIn("const restored = await restoreSession()", self.script)
        self.assertIn('elif route == "api/session":', self.dashboard_source)
        self.assertIn("result = snapshot.resume(body)", self.dashboard_source)

        snapshot = build_teacher_agent_dashboard_snapshot(
            ROOT / "data" / "teacher_agent_skill_library.json",
            ROOT / "data" / "teacher_agent_demo_input.json",
            ROOT / "data" / "teacher_agent_evaluation_cases.json",
        )
        started = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": snapshot.demo_input["student_profile"],
                "start_idempotency_key": "ui-contract-resume-001",
                "profile_revision": "ui-contract-profile-v1",
                "profile_display_name": "UI Contract Student",
            }
        )
        resumed = snapshot.resume({"session_id": started["session_id"]})
        self.assertEqual(resumed["session_id"], started["session_id"])
        self.assertEqual(
            resumed["expected_question_id"], started["expected_question_id"]
        )
        self.assertEqual(resumed["context_version"], started["context_version"])
        self.assertEqual(
            resumed["profile_summary"]["profile_revision"],
            "ui-contract-profile-v1",
        )

    def test_step_payload_binds_round_question_context_and_profile(self) -> None:
        submit_turn = self.script.split("async function submitTurn(event)", 1)[1].split(
            "async function chooseAutoMode", 1
        )[0]
        expected_fields = {
            "expected_round": "round",
            "expected_question_id": "app.session.expected_question_id",
            "expected_context_version": "finite(app.session.context_version, 1)",
            "profile_revision": "textValue(app.session.profile_summary?.profile_revision, app.profileRevision)",
        }
        for field, expression in expected_fields.items():
            with self.subTest(field=field):
                self.assertIn(f"{field}: {expression}", submit_turn)

        fingerprint = self.script.split("function turnRequestFingerprint(payload)", 1)[
            1
        ].split("function clearPendingTurn", 1)[0]
        for field in expected_fields:
            with self.subTest(fingerprint_field=field):
                self.assertIn(f"{field}: payload.{field}", fingerprint)

    def test_js_never_uses_inner_html_and_all_literal_select_ids_exist(self) -> None:
        self.assertNotIn(".innerHTML", self.script)
        html_ids = [
            str(attrs["id"]) for _tag, attrs in self.parser.elements if attrs.get("id")
        ]
        self.assertEqual(len(html_ids), len(set(html_ids)), "HTML IDs must be unique")
        selected_ids = set(
            re.findall(
                r"""select\(\s*["']#([A-Za-z][\w:.-]*)["']\s*\)""",
                self.script,
            )
        )
        self.assertEqual(selected_ids - set(html_ids), set())

    def test_css_keeps_responsive_drawers_focus_and_reduced_motion_contracts(
        self,
    ) -> None:
        for media_query in (
            "@media (max-width: 1260px)",
            "@media (max-width: 860px)",
            "@media (max-width: 560px)",
            "@media (prefers-reduced-motion: reduce)",
        ):
            with self.subTest(media_query=media_query):
                self.assertIn(media_query, self.style)
        for selector in (
            ".app-shell.inspector-open .state-panel",
            ".app-shell.sidebar-open .setup-panel",
            ".drawer-backdrop",
            ".setup-group > summary:focus-visible",
            ".inspector-section > summary:focus-visible",
            ".turn-audit > summary:focus-visible",
            ".teacher-audit-hint > summary:focus-visible",
        ):
            with self.subTest(selector=selector):
                self.assertIn(selector, self.style)
        self.assertIn("box-shadow: inset 0 0 0 2px", self.style)
        self.assertIn("transition-duration: .01ms !important", self.style)
        self.assertIn("[inert]", self.style)
        self.assertRegex(
            self.style,
            r"\.inspector-tabs button\s*\{[^}]*min-height:\s*44px;[^}]*font-size:\s*12px;",
        )
        self.assertRegex(
            self.style,
            r"\.icon-button\s*\{[^}]*min-width:\s*44px;[^}]*min-height:\s*44px;",
        )
        self.assertRegex(
            self.style,
            r"\.send-button\s*\{[^}]*min-width:\s*78px;[^}]*min-height:\s*44px;",
        )
        for selector, minimum_size in (
            (r"\.profile-picker-head > small", "11px"),
            (r"\.profile-card-copy small", "11px"),
            (r"\.synthetic-note", "11px"),
            (r"\.mastery-overview > p", "11px"),
        ):
            with self.subTest(font_selector=selector):
                self.assertRegex(
                    self.style,
                    rf"{selector}\s*\{{[^}}]*font-size:\s*{minimum_size};",
                )

    def test_responsive_drawers_are_modal_inert_and_focus_contained(self) -> None:
        for element_id in ("setupPanel", "statePanel"):
            with self.subTest(panel=element_id):
                _tag, attrs = self.parser.by_id(element_id)
                self.assertEqual(attrs.get("tabindex"), "-1")

        responsive = self.script.split("function responsiveDrawerState()", 1)[
            1
        ].split("function syncDrawerBackdrop", 1)[0]
        for required in (
            'window.matchMedia("(max-width: 860px)").matches',
            'window.matchMedia("(max-width: 1260px)").matches',
            'panel.toggleAttribute("inert", !open)',
            'panel.setAttribute("aria-hidden", String(!open))',
            'panel.setAttribute("role", "dialog")',
            'panel.setAttribute("aria-modal", "true")',
            'panel.removeAttribute("inert")',
            'select(".topbar").toggleAttribute("inert", modalOpen)',
            'select("#liveLoop").toggleAttribute("inert", modalOpen)',
            'focusResponsiveDrawer(activeDrawer)',
            'if (event.key !== "Tab") return',
            'document.activeElement === first',
            'document.activeElement === last',
        ):
            with self.subTest(required=required):
                self.assertIn(required, responsive)

        bindings = self.script.split("function bindEvents()", 1)[1].split(
            "async function restoreSession", 1
        )[0]
        self.assertIn("trapResponsiveDrawerFocus(event)", bindings)
        self.assertIn('if (event.key !== "Escape") return', bindings)
        self.assertIn(
            'select("#drawerBackdrop").addEventListener("click", () => closeDrawers({restoreFocus: true}))',
            bindings,
        )
        self.assertGreaterEqual(
            bindings.count("closeDrawers({restoreFocus: true})"),
            4,
        )

        close = self.script.split("function closeDrawers", 1)[1].split(
            "function showSetupForm", 1
        )[0]
        self.assertIn("const focusTarget", close)
        self.assertIn("focusTarget.focus()", close)
        self.assertIn("syncResponsiveA11y()", bindings)


if __name__ == "__main__":
    unittest.main()
