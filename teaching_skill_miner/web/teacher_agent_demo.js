(() => {
  "use strict";

  const app = {
    bootstrap: null,
    session: null,
    busy: false,
    controlMode: "auto",
    manualSkillId: "",
    manualDraftOpen: false,
    sessionInitialMastery: {},
    lastSetupPayload: null,
    pendingStart: null,
    pendingTurn: null,
    pendingCommand: null,
    pendingAttachment: null,
    stopRequested: false,
    draftingReplacement: false,
    activeView: "learning",
    toastTimer: null,
    selectedProfileId: "xiaoyu",
    activeProfileId: "",
    profileRevision: "profile-xiaoyu-v1",
    profileEditCounter: 0,
    profileDraftReady: false,
    profileDrafts: {},
    knowledgeSpecDraft: null,
    inspectorTab: "state",
    lastDrawerTrigger: null,
    stateEpoch: 0,
    profileEpoch: 0,
    requestSequence: 0,
    requestEpochs: {}
  };

  const sessionHandleKey = "teachlab_opaque_session_handle_v2";
  const maximumImageBytes = 4 * 1024 * 1024;
  const supportedImageTypes = new Set(["image/png", "image/jpeg", "image/webp"]);
  const studentProfiles = {
    xiaoyu: {
      id: "xiaoyu",
      ref: "synthetic_profile_xiaoyu_v1",
      revision: "profile-xiaoyu-v1",
      name: "小雨",
      type: "例子驱动型",
      level: "beginner",
      avatar: "assets/student-xiaoyu.png",
      preferences: ["先看直观例子", "再用自己的话解释"],
      mastery: {prerequisite: 0.25, conceptual: 0.15, procedural: 0.1, transfer: 0.05},
      misconceptions: [],
      conversationHistory: [],
      accessibilityNeeds: [],
      containsDirectIdentity: false
    },
    zimo: {
      id: "zimo",
      ref: "synthetic_profile_zimo_v1",
      revision: "profile-zimo-v1",
      name: "子墨",
      type: "稳步练习型",
      level: "intermediate",
      avatar: "assets/student-zimo.png",
      preferences: ["逐步提示", "每个步骤后进行即时练习"],
      mastery: {prerequisite: 0.45, conceptual: 0.35, procedural: 0.3, transfer: 0.2},
      misconceptions: [],
      conversationHistory: [],
      accessibilityNeeds: [],
      containsDirectIdentity: false
    },
    zhixing: {
      id: "zhixing",
      ref: "synthetic_profile_zhixing_v1",
      revision: "profile-zhixing-v1",
      name: "知行",
      type: "迁移挑战型",
      level: "advanced",
      avatar: "assets/student-zhixing.png",
      preferences: ["比较相近概念", "使用反例和新情境挑战"],
      mastery: {prerequisite: 0.7, conceptual: 0.58, procedural: 0.52, transfer: 0.4},
      misconceptions: [],
      conversationHistory: [],
      accessibilityNeeds: [],
      containsDirectIdentity: false
    }
  };

  const signalLabels = {
    not_observed: "尚未观察",
    correct: "理解正确",
    partial: "部分理解",
    misconception: "存在误解",
    confused: "仍然困惑",
    no_response: "没有回应"
  };
  const answerAlignmentLabels = {
    not_applicable: "尚未作答",
    aligned: "直接回答本问",
    partially_aligned: "部分满足本问",
    related_but_not_answer: "相关，但未回答本问",
    contradicted: "与目标概念冲突",
    ambiguous: "证据不足，需澄清",
    no_response: "没有可判断回答"
  };
  const assessmentSourceLabels = {
    "AWAITING STUDENT RESPONSE": "等待学生回答 · 尚无诊断",
    "DEEPSEEK ASSESSMENT": "在线模型诊断 · DeepSeek",
    "DEEPSEEK + CONTRACT GUARD": "在线模型诊断 · 契约约束",
    "ACTIVE CONTRACT EXACT MATCH": "确定性契约 · 精确命中",
    "TEACHER KNOWLEDGE EXACT MATCH": "教师知识标准 · 精确命中",
    "TEACHER GOAL BOUNDED MATCH": "教师目标知识点 · 有界命中",
    "SAFETY FALLBACK SIGNAL": "安全规则回退 · 非模型",
    "STRUCTURED DEMO SIGNAL": "结构化演示信号 · 非模型"
  };
  const provenanceReasonLabels = {
    deterministic_legacy_mode: "配置为确定性执行",
    validated_model_plan_unavailable: "模型计划不可用",
    visual_confirmation_requires_materializer: "图片文字需要学生确认",
    model_teacher_action_type_mismatch: "动作类型与 Skill 不一致",
    model_teacher_action_message_invalid: "教师话语格式无效",
    model_teacher_action_expected_signal_invalid: "观察目标格式无效",
    model_teacher_action_final_answer_pattern: "候选话语可能直接泄露答案",
    model_teacher_action_policy_or_answer_violation: "候选话语触发安全边界",
    model_teacher_action_does_not_elicit_response: "候选话语没有等待学生回应",
    action_only_repair_applied: "固定 Skill 后已由 DeepSeek 定向修复话语",
    model_teacher_action_does_not_execute_selected_skill: "候选话语没有落实所选 Skill",
    model_question_contract_not_object: "本轮评分契约缺失",
    model_question_contract_requires_server_repair: "本轮评分契约需要校正",
    primary_skill_selection_constrained: "主 Skill 已按契约校正",
    supporting_skill_selection_constrained: "辅助 Skill 已按契约校正",
    next_focus_constrained_to_primary_skill: "下一重点已对齐主 Skill"
  };
  const dimensionLabels = {
    prerequisite: "前置知识",
    conceptual: "概念理解",
    procedural: "操作过程",
    transfer: "迁移能力"
  };
  const responseQualityLabels = {
    complete: "完整回答",
    partial: "部分回答",
    minimal: "信息较少",
    off_topic: "偏离主题",
    empty: "未作答"
  };
  const engagementLabels = {
    high: "高",
    medium: "中",
    low: "低",
    unknown: "未知"
  };
  const roleLabels = {
    diagnostic: "诊断",
    context: "情境建立",
    example: "直观例子",
    concept_mapping: "概念映射",
    scaffolding: "逐步支架",
    assessment: "理解检查",
    correction: "误解纠错",
    practice: "练习反馈",
    review: "主动复习",
    metacognition: "元认知",
    engagement: "参与恢复",
    transfer: "迁移检查",
    summary: "学习者总结",
    support: "支持约束"
  };
  const statusLabels = {
    active: "进行中",
    succeeded: "目标达标",
    terminated_unable: "停止并转人工"
  };
  const primaryRoles = new Set(Object.keys(roleLabels).filter((role) => role !== "support"));

  const select = (query) => document.querySelector(query);
  const object = (value) => value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const array = (value) => Array.isArray(value) ? value : [];
  const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
  const textValue = (value, fallback = "—") => {
    const valueText = value === null || value === undefined ? "" : String(value).trim();
    return valueText || fallback;
  };

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function route(relative) {
    return new URL(relative, window.location.href).href;
  }

  async function requestJson(relative, options = {}) {
    const response = await fetch(route(relative), {
      cache: "no-store",
      credentials: "same-origin",
      ...options
    });
    let payload;
    try {
      payload = await response.json();
    } catch (_error) {
      throw new Error(`服务返回了无法解析的结果（HTTP ${response.status}）`);
    }
    if (!response.ok) {
      throw new Error(textValue(payload.error, `请求失败（HTTP ${response.status}）`));
    }
    return payload;
  }

  async function postJson(relative, payload) {
    return requestJson(relative, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
  }

  class StaleResponseError extends Error {
    constructor(message = "已忽略过期的异步响应") {
      super(message);
      this.name = "StaleResponseError";
    }
  }

  function isStaleResponseError(error) {
    return error instanceof StaleResponseError
      || error?.name === "StaleResponseError";
  }

  function sessionProfileRevision(session = app.session) {
    return textValue(object(session?.profile_summary).profile_revision, "");
  }

  function sessionContextVersion(session = app.session) {
    return Math.max(0, Math.round(finite(session?.context_version, 0)));
  }

  function beginRequestAnchor(kind, {
    session = app.session,
    targetProfileId = app.selectedProfileId,
    targetProfileRevision = app.profileRevision
  } = {}) {
    const requestSerial = ++app.requestSequence;
    app.requestEpochs[kind] = requestSerial;
    return {
      kind,
      requestSerial,
      stateEpoch: app.stateEpoch,
      profileEpoch: app.profileEpoch,
      sessionId: textValue(session?.session_id, ""),
      profileRevision: sessionProfileRevision(session),
      contextVersion: sessionContextVersion(session),
      targetProfileId,
      targetProfileRevision
    };
  }

  function assertRequestAnchorCurrent(anchor, {
    requireSameSession = true,
    requireTargetProfile = false
  } = {}) {
    if (
      !anchor
      || app.requestEpochs[anchor.kind] !== anchor.requestSerial
      || app.stateEpoch !== anchor.stateEpoch
    ) {
      throw new StaleResponseError();
    }
    if (requireSameSession) {
      if (
        textValue(app.session?.session_id, "") !== anchor.sessionId
        || sessionProfileRevision() !== anchor.profileRevision
        || sessionContextVersion() !== anchor.contextVersion
      ) {
        throw new StaleResponseError();
      }
    }
    if (
      requireTargetProfile
      && (
        app.profileEpoch !== anchor.profileEpoch
        || app.selectedProfileId !== anchor.targetProfileId
        || app.profileRevision !== anchor.targetProfileRevision
      )
    ) {
      throw new StaleResponseError("画像设置已变化，旧响应未应用");
    }
  }

  function sessionResponseIdentity(response) {
    const session = object(response);
    const sessionId = textValue(session.session_id, "");
    const profileRevision = sessionProfileRevision(session);
    const rawContextVersion = Number(session.context_version);
    if (
      !sessionId
      || !profileRevision
      || !Number.isInteger(rawContextVersion)
      || rawContextVersion < 1
    ) {
      throw new Error("服务返回的会话身份或上下文版本无效");
    }
    return {sessionId, profileRevision, contextVersion: rawContextVersion};
  }

  function commitSessionResponse(anchor, response, {
    newSession = false,
    expectedSessionId = "",
    targetProfileRevision = "",
    requireDifferentSession = false,
    requireTargetProfile = false
  } = {}) {
    assertRequestAnchorCurrent(anchor, {
      requireSameSession: !newSession || Boolean(anchor.sessionId),
      requireTargetProfile
    });
    const identity = sessionResponseIdentity(response);
    if (newSession) {
      if (expectedSessionId && identity.sessionId !== expectedSessionId) {
        throw new Error("恢复响应没有返回请求的会话");
      }
      if (targetProfileRevision && identity.profileRevision !== targetProfileRevision) {
        throw new Error("新画像会话没有返回目标 profile_revision");
      }
      if (requireDifferentSession && anchor.sessionId === identity.sessionId) {
        throw new Error("画像替换必须创建不同的 session_id");
      }
    } else {
      if (
        identity.sessionId !== anchor.sessionId
        || identity.profileRevision !== anchor.profileRevision
      ) {
        throw new Error("服务响应与当前会话或画像不匹配");
      }
      if (
        identity.contextVersion < anchor.contextVersion
        || identity.contextVersion < sessionContextVersion()
      ) {
        throw new StaleResponseError("服务返回了较旧的 context_version");
      }
    }
    app.session = response;
    app.stateEpoch += 1;
    return response;
  }

  function commitAttachmentResponse(anchor, response) {
    assertRequestAnchorCurrent(anchor);
    const sessionId = textValue(response?.session_id, "");
    const profileRevision = textValue(response?.profile_revision, "");
    const contextVersion = Number(response?.context_version);
    if (
      sessionId !== anchor.sessionId
      || profileRevision !== anchor.profileRevision
    ) {
      throw new Error("图片证据响应与当前会话或画像不匹配");
    }
    if (
      !Number.isInteger(contextVersion)
      || contextVersion < anchor.contextVersion
      || contextVersion < sessionContextVersion()
    ) {
      throw new StaleResponseError("图片证据返回了较旧的 context_version");
    }
    app.session.context_version = contextVersion;
    app.stateEpoch += 1;
    return response;
  }

  function showToast(message) {
    const toast = select("#toast");
    toast.textContent = message;
    toast.hidden = false;
    window.clearTimeout(app.toastTimer);
    app.toastTimer = window.setTimeout(() => {
      toast.hidden = true;
    }, 4600);
  }

  function showInlineError(target, message) {
    const error = select(target);
    error.textContent = message;
    error.hidden = false;
  }

  function clearInlineError(target) {
    const error = select(target);
    error.textContent = "";
    error.hidden = true;
  }

  function showUnavailableSessionNotice(message) {
    showSetupForm(true);
    showInlineError("#setupError", message);
    showToast(message);
    const error = select("#setupError");
    error.tabIndex = -1;
    error.focus();
  }

  function persistSessionHandle() {
    try {
      if (app.session?.session_id) {
        window.sessionStorage.setItem(sessionHandleKey, app.session.session_id);
      } else {
        window.sessionStorage.removeItem(sessionHandleKey);
      }
    } catch (_error) {
      // The opaque handle is optional; no teaching content is stored in the browser.
    }
  }

  function readSessionHandle() {
    try {
      return window.sessionStorage.getItem(sessionHandleKey) || "";
    } catch (_error) {
      return "";
    }
  }

  function clearSessionHandle() {
    try {
      window.sessionStorage.removeItem(sessionHandleKey);
    } catch (_error) {
      // Ignore browsers that disable sessionStorage.
    }
  }

  function isUnavailableSessionError(error, {replacement = false} = {}) {
    const message = String(error?.message || error);
    if (/session_id is no longer available/.test(message)) return true;
    return replacement && /replace_session_id (?:does not match an available session|is no longer available|changed while the new session was prepared)/.test(message);
  }

  function isReplacementGuardMismatch(error) {
    return /replace_expected_(?:round|question_id|context_version|profile_revision) does not match this session/.test(
      String(error?.message || error)
    );
  }

  function discardUnavailableSession({render = true} = {}) {
    app.stateEpoch += 1;
    clearPendingAttachment();
    app.session = null;
    syncNewMessageAnnouncer(null);
    app.activeProfileId = "";
    app.pendingStart = null;
    app.pendingTurn = null;
    app.pendingCommand = null;
    app.draftingReplacement = false;
    clearSessionHandle();
    setControlMode("auto");
    syncReplacementDraftUi();
    if (!render) return;
    select("#activeSession").hidden = true;
    select("#emptySession").hidden = false;
    renderPhase(null);
    showSetupForm(true);
    renderProfileIdentity();
    renderDraftProfileBaseline();
    syncControls();
  }

  function probability(value, digits = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
  }

  function numberScore(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : "—";
  }

  function signedPoints(value, digits = 2) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return `${number > 0 ? "+" : ""}${number.toFixed(digits)} pt`;
  }

  function compactText(value, maximum = 180) {
    const raw = textValue(value, "");
    if (!raw) return "—";
    return raw.length > maximum ? `${raw.slice(0, maximum - 1)}…` : raw;
  }

  function makeIdempotencyKey() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return `turn-${Date.now()}-${Math.random().toString(36).slice(2, 12)}`;
  }

  function turnRequestFingerprint(payload) {
    return JSON.stringify({
      learner_response: payload.learner_response,
      session_id: payload.session_id,
      expected_round: payload.expected_round,
      expected_question_id: payload.expected_question_id ?? null,
      expected_context_version: payload.expected_context_version ?? null,
      profile_revision: payload.profile_revision ?? null,
      signal: payload.signal ?? null,
      signal_confidence: payload.signal_confidence ?? null,
      misconception_tag: payload.misconception_tag ?? null,
      manual_skill_id: payload.manual_skill_id ?? null,
      attachment_ids: array(payload.attachment_ids),
      confirmed_attachment_ids: array(payload.confirmed_attachment_ids)
    });
  }

  function clearPendingTurn() {
    app.pendingTurn = null;
  }

  function attachmentBindingFingerprint(file = app.pendingAttachment?.file) {
    if (!file || !app.session) return "";
    return JSON.stringify({
      session_id: app.session.session_id,
      round: finite(app.session.rounds_completed, 0),
      question_id: app.session.expected_question_id,
      profile_revision: textValue(
        app.session.profile_summary?.profile_revision,
        app.profileRevision
      ),
      file_name: file.name,
      file_type: file.type,
      file_size: file.size,
      file_modified: file.lastModified
    });
  }

  function renderPendingAttachment({status, evidence = ""} = {}) {
    const pending = app.pendingAttachment;
    const preview = select("#attachmentPreview");
    if (!pending) {
      preview.hidden = true;
      select("#attachmentThumbnail").removeAttribute("src");
      select("#attachmentName").textContent = "答案图片";
      select("#attachmentStatus").textContent = "等待本机识别";
      select("#attachmentStatus").removeAttribute("data-state");
      select("#attachmentEvidencePreview").textContent = "发送时提取 OCR，原图不会交给 DeepSeek。";
      renderAttachmentConfirmation();
      return;
    }
    preview.hidden = false;
    select("#attachmentThumbnail").src = pending.previewUrl;
    select("#attachmentName").textContent = pending.file.name;
    const statusNode = select("#attachmentStatus");
    statusNode.textContent = status || "已选择 · 发送时本机识别";
    statusNode.dataset.state = pending.state || "selected";
    select("#attachmentEvidencePreview").textContent = evidence
      || "原图只在本机内存和临时目录短暂处理；识别文字会随本轮回答送入 Agent。";
    renderAttachmentConfirmation();
  }

  function attachmentNeedsConfirmation(pending = app.pendingAttachment) {
    return Boolean(
      pending?.uploaded?.needs_student_confirmation
      && pending.ocrTextConfirmed !== true
    );
  }

  function renderAttachmentConfirmation() {
    const panel = select("#attachmentConfirmation");
    const pending = app.pendingAttachment;
    const needsConfirmation = attachmentNeedsConfirmation(pending);
    panel.hidden = !needsConfirmation;
    if (!needsConfirmation) return;
    select("#attachmentConfirmationText").textContent = textValue(
      pending.uploaded?.recognized_text,
      "本机没有识别出可靠文字；请在输入框手动填写答案"
    );
    select("#confirmAttachmentTextButton").disabled = app.busy
      || app.session?.status !== "active"
      || app.draftingReplacement
      || !textValue(pending.uploaded?.recognized_text, "");
  }

  function clearPendingAttachment() {
    const previewUrl = app.pendingAttachment?.previewUrl;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    app.pendingAttachment = null;
    const input = select("#answerImageInput");
    if (input) input.value = "";
    const preview = select("#attachmentPreview");
    if (preview) renderPendingAttachment();
    clearPendingTurn();
  }

  function chooseAnswerImage(file) {
    clearInlineError("#turnError");
    if (!app.session || app.session.status !== "active" || app.draftingReplacement) {
      showInlineError("#turnError", "请先进入一个正在进行的教学会话，再添加本轮答案图片。");
      return;
    }
    if (!file || !supportedImageTypes.has(file.type)) {
      showInlineError("#turnError", "请选择 PNG、JPEG 或 WebP 格式的答案图片。");
      return;
    }
    if (!Number.isFinite(file.size) || file.size < 1 || file.size > maximumImageBytes) {
      showInlineError("#turnError", "答案图片必须小于或等于 4 MB。");
      return;
    }
    clearPendingAttachment();
    app.pendingAttachment = {
      file,
      previewUrl: URL.createObjectURL(file),
      state: "selected",
      uploadFingerprint: "",
      uploadIdempotencyKey: "",
      uploaded: null,
      ocrTextConfirmed: false
    };
    renderPendingAttachment();
    syncControls();
  }

  async function fileToBase64(file) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const parts = [];
    for (let offset = 0; offset < bytes.length; offset += 0x8000) {
      parts.push(String.fromCharCode(...bytes.subarray(offset, offset + 0x8000)));
    }
    return window.btoa(parts.join(""));
  }

  async function uploadPendingAttachment() {
    const pending = app.pendingAttachment;
    if (!pending) return [];
    const bindingFingerprint = attachmentBindingFingerprint(pending.file);
    if (!bindingFingerprint) throw new Error("当前会话无法安全绑定答案图片");
    if (
      pending.uploaded?.attachment_id
      && pending.uploadFingerprint === bindingFingerprint
    ) {
      return [pending.uploaded.attachment_id];
    }
    if (pending.uploadFingerprint !== bindingFingerprint) {
      pending.uploadFingerprint = bindingFingerprint;
      pending.uploadIdempotencyKey = makeIdempotencyKey();
      pending.uploaded = null;
      pending.ocrTextConfirmed = false;
    }
    pending.state = "uploading";
    renderPendingAttachment({status: "正在本机提取文字证据……"});
    const dataBase64 = await fileToBase64(pending.file);
    if (app.pendingAttachment !== pending || attachmentBindingFingerprint(pending.file) !== bindingFingerprint) {
      throw new Error("图片处理期间会话已变化，请重新选择本轮答案图片");
    }
    const requestAnchor = beginRequestAnchor("attachment");
    const response = await postJson("api/attachment", {
      session_id: app.session.session_id,
      expected_round: finite(app.session.rounds_completed, 0),
      expected_question_id: app.session.expected_question_id,
      expected_context_version: finite(app.session.context_version, 1),
      profile_revision: textValue(
        app.session.profile_summary?.profile_revision,
        app.profileRevision
      ),
      attachment_idempotency_key: pending.uploadIdempotencyKey,
      mime_type: pending.file.type,
      display_name: pending.file.name,
      data_base64: dataBase64
    });
    if (app.pendingAttachment !== pending) {
      throw new StaleResponseError("图片上传完成前已被移除或替换");
    }
    if (attachmentBindingFingerprint(pending.file) !== bindingFingerprint) {
      throw new StaleResponseError("图片处理期间会话绑定已变化");
    }
    commitAttachmentResponse(requestAnchor, response);
    const attachment = object(response.attachment);
    if (!textValue(attachment.attachment_id, "")) {
      throw new Error("本机图片识别没有返回可绑定的证据编号");
    }
    pending.uploaded = attachment;
    pending.ocrTextConfirmed = false;
    pending.state = attachment.needs_student_confirmation ? "review" : "ready";
    const recognized = compactText(attachment.recognized_text, 220);
    const confidence = probability(attachment.confidence, 0);
    const confirmationReason = attachment.formula_like_text_detected
      ? "检测到公式或数学符号；OCR 置信度不代表公式正确，请按原图核对。"
      : "本机识别结果可能有误，请按原图核对或在输入框修正。";
    renderPendingAttachment({
      status: attachment.needs_student_confirmation
        ? `识别结果需核对 · 置信度 ${confidence}`
        : `本机识别完成 · 置信度 ${confidence}`,
      evidence: recognized === "—"
        ? "没有识别出可靠文字；请在输入框补充或修正图片中的答案。"
        : attachment.needs_student_confirmation
          ? `${confirmationReason} 识别文字：${recognized}`
          : `识别文字：${recognized}`
    });
    return [attachment.attachment_id];
  }

  function clearPendingCommand() {
    app.pendingCommand = null;
  }

  function commandRequestFingerprint(payload) {
    return JSON.stringify({
      command: payload.command,
      skill_id: payload.skill_id ?? null,
      session_id: payload.session_id,
      expected_round: payload.expected_round,
      expected_question_id: payload.expected_question_id ?? null,
      expected_context_version: payload.expected_context_version ?? null,
      profile_revision: payload.profile_revision ?? null
    });
  }

  function setAppView(view) {
    const evaluation = view === "evaluation";
    app.activeView = evaluation ? "evaluation" : "learning";
    select("#learningView").hidden = evaluation;
    select("#evaluationView").hidden = !evaluation;
    select("#learningViewButton").classList.toggle("active", !evaluation);
    select("#evaluationViewButton").classList.toggle("active", evaluation);
    select("#learningViewButton").setAttribute("aria-pressed", String(!evaluation));
    select("#evaluationViewButton").setAttribute("aria-pressed", String(evaluation));
  }

  function setInspectorTab(tabName) {
    const selectedTab = ["state", "method", "evidence"].includes(tabName) ? tabName : "state";
    app.inspectorTab = selectedTab;
    for (const button of document.querySelectorAll("[data-inspector-tab]")) {
      const selected = button.dataset.inspectorTab === selectedTab;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    }
    for (const panel of document.querySelectorAll("[data-inspector-panel]")) {
      panel.hidden = panel.dataset.inspectorPanel !== selectedTab;
    }
  }

  function setInspector(open) {
    const shell = select("#appShell");
    shell.classList.toggle("inspector-open", open);
    shell.classList.toggle("inspector-collapsed", !open);
    select("#inspectorToggle").setAttribute("aria-expanded", String(open));
    select("#inspectorToggle").setAttribute("aria-label", open ? "收起学习状态" : "打开学习状态");
    syncDrawerBackdrop();
    syncResponsiveA11y({focusOpen: open});
  }

  function setSidebar(open) {
    select("#appShell").classList.toggle("sidebar-open", open);
    select("#sidebarToggle").setAttribute("aria-expanded", String(open));
    syncDrawerBackdrop();
    syncResponsiveA11y({focusOpen: open});
  }

  function responsiveDrawerState() {
    const shell = select("#appShell");
    const sidebarDrawer = window.matchMedia("(max-width: 860px)").matches;
    const inspectorDrawer = window.matchMedia("(max-width: 1260px)").matches;
    return {
      sidebarDrawer,
      inspectorDrawer,
      sidebarOpen: sidebarDrawer && shell.classList.contains("sidebar-open"),
      inspectorOpen: inspectorDrawer && shell.classList.contains("inspector-open")
    };
  }

  function setDrawerSemantics(panel, {drawer, open}) {
    if (drawer) {
      panel.toggleAttribute("inert", !open);
      panel.setAttribute("aria-hidden", String(!open));
      if (open) {
        panel.setAttribute("role", "dialog");
        panel.setAttribute("aria-modal", "true");
      } else {
        panel.removeAttribute("role");
        panel.removeAttribute("aria-modal");
      }
      return;
    }
    panel.removeAttribute("inert");
    panel.removeAttribute("aria-hidden");
    panel.removeAttribute("role");
    panel.removeAttribute("aria-modal");
  }

  function activeResponsiveDrawer(state = responsiveDrawerState()) {
    if (state.sidebarOpen) return select("#setupPanel");
    if (state.inspectorOpen) return select("#statePanel");
    return null;
  }

  function drawerFocusableElements(panel) {
    return [...panel.querySelectorAll(
      "button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, a[href], [tabindex]:not([tabindex='-1'])"
    )].filter((element) => !element.hidden && element.getClientRects().length > 0);
  }

  function focusResponsiveDrawer(panel) {
    const preferred = panel === select("#setupPanel")
      ? select("#sidebarClose")
      : select("#inspectorClose");
    const target = preferred.getClientRects().length > 0
      ? preferred
      : drawerFocusableElements(panel)[0] || panel;
    window.requestAnimationFrame(() => target.focus());
  }

  function syncResponsiveA11y({focusOpen = false} = {}) {
    const state = responsiveDrawerState();
    const setupPanel = select("#setupPanel");
    const statePanel = select("#statePanel");
    const activeDrawer = activeResponsiveDrawer(state);
    const modalOpen = Boolean(activeDrawer);

    setDrawerSemantics(setupPanel, {drawer: state.sidebarDrawer, open: state.sidebarOpen});
    setDrawerSemantics(statePanel, {drawer: state.inspectorDrawer, open: state.inspectorOpen});

    select(".topbar").toggleAttribute("inert", modalOpen);
    select("#liveLoop").toggleAttribute("inert", modalOpen);
    if (state.sidebarOpen) statePanel.setAttribute("inert", "");
    if (state.inspectorOpen) setupPanel.setAttribute("inert", "");

    document.body.classList.toggle("drawer-modal-open", modalOpen);
    if (modalOpen && (focusOpen || !activeDrawer.contains(document.activeElement))) {
      focusResponsiveDrawer(activeDrawer);
    }
  }

  function trapResponsiveDrawerFocus(event) {
    if (event.key !== "Tab") return;
    const panel = activeResponsiveDrawer();
    if (!panel) return;
    const focusable = drawerFocusableElements(panel);
    if (!focusable.length) {
      event.preventDefault();
      panel.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable.at(-1);
    if (!panel.contains(document.activeElement)) {
      event.preventDefault();
      first.focus();
      return;
    }
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function syncDrawerBackdrop() {
    const shell = select("#appShell");
    const drawerOpen = (window.matchMedia("(max-width: 860px)").matches && shell.classList.contains("sidebar-open"))
      || (window.matchMedia("(max-width: 1260px)").matches && shell.classList.contains("inspector-open"));
    select("#drawerBackdrop").hidden = !drawerOpen;
  }

  function closeDrawers({restoreFocus = false} = {}) {
    const focusTarget = restoreFocus ? app.lastDrawerTrigger : null;
    setSidebar(false);
    if (window.matchMedia("(max-width: 1260px)").matches) setInspector(false);
    if (focusTarget) window.requestAnimationFrame(() => focusTarget.focus());
    app.lastDrawerTrigger = null;
  }

  function showSetupForm(visible) {
    select("#setupForm").hidden = !visible;
    const hasSession = Boolean(app.session);
    select("#presetButtonLabel").textContent = hasSession
      ? (visible ? "收起任务设置" : "新建 / 编辑任务")
      : "载入演示任务";
    select("#presetButtonHint").textContent = hasSession
      ? (visible ? "当前会话仍保留，提交后新建会话" : "查看目标、画像与达标条件")
      : "填写一组可直接运行的样例";
  }

  function syncReplacementDraftUi() {
    const notice = select("#replacementDraftNotice");
    const hint = select("#profileSwitchHint");
    const replacing = Boolean(app.session && app.draftingReplacement);
    notice.hidden = !replacing;
    select("#setupForm").classList.toggle("replacement-draft", replacing);
    if (!replacing) {
      hint.textContent = "切换后新建独立会话";
      return;
    }
    const activeName = activeProfile().name;
    const nextName = selectedProfile().name;
    hint.textContent = `待切换：${nextName}`;
    notice.textContent = activeName === nextName
      ? `当前“${activeName}”会话仍完整保留；确认设置并开始后，才会创建新的独立会话。`
      : `当前对话仍属于“${activeName}”；点击“切换到${nextName}并开始”成功后，才会替换为新画像会话。`;
  }

  function beginReplacementDraft({resetControl = true} = {}) {
    if (!app.session) return;
    clearPendingAttachment();
    app.draftingReplacement = true;
    app.pendingStart = null;
    clearPendingTurn();
    clearPendingCommand();
    if (resetControl) setControlMode("auto");
    showSetupForm(true);
    syncReplacementDraftUi();
    if (window.matchMedia("(max-width: 860px)").matches) setSidebar(true);
    syncControls();
  }

  function endReplacementDraft() {
    if (!app.draftingReplacement) return;
    app.draftingReplacement = false;
    synchronizeControlModeFromSession();
    syncReplacementDraftUi();
    syncControls();
  }

  function handleSetupButton() {
    if (!app.session) {
      fillSetupForm();
      showToast("演示任务已填入，可以直接检查后开始学习。 ");
      return;
    }
    const nextVisible = select("#setupForm").hidden;
    if (nextVisible) {
      beginReplacementDraft({resetControl: true});
    } else {
      showSetupForm(false);
      endReplacementDraft();
    }
    if (nextVisible) select("#conceptInput").focus();
  }

  function updateWorkspaceIdentity(goal = null) {
    const currentGoal = object(goal || app.session?.goal || app.lastSetupPayload?.goal);
    const concept = textValue(currentGoal.concept || select("#conceptInput").value, "尚未开始学习");
    const objective = textValue(currentGoal.objective || select("#objectiveInput").value, "先告诉系统“要学什么”和“学到什么程度”。");
    select("#workspaceGoalTitle").textContent = concept;
    select("#sidebarConceptTitle").textContent = concept;
    select("#sidebarObjectiveText").textContent = compactText(objective, 110);
    select("#conversationTitle").textContent = app.session ? concept : "准备开始";
  }

  function renderPhase(session) {
    const active = session?.status === "active";
    const terminal = session && !active;
    const setup = select("#phaseSetup");
    const learn = select("#phaseLearn");
    const finish = select("#phaseFinish");
    setup.classList.toggle("current", !session);
    setup.classList.toggle("completed", Boolean(session));
    learn.classList.toggle("current", Boolean(active));
    learn.classList.toggle("completed", Boolean(terminal));
    finish.classList.toggle("current", Boolean(terminal));
  }

  function providerReady() {
    const provider = object(app.bootstrap?.provider_status);
    return provider.provider === "deepseek"
      && provider.configured === true
      && provider.remote_student_data_opt_in === true;
  }

  function commandControlsSupported() {
    return app.bootstrap?.provider_status?.provider === "deepseek";
  }

  function latestTrace() {
    const action = currentAction();
    const runtime = object(app.session?.agent_runtime);
    const history = array(app.session?.history);
    const lastEvent = object(history.at(-1));
    return object(action.model_trace || lastEvent.model_trace || runtime.last_model_trace);
  }

  function renderProviderStatus() {
    const provider = object(app.bootstrap?.provider_status);
    const trace = latestTrace();
    const badge = select("#providerBadge");
    const model = textValue(trace.model || provider.model, "DeepSeek V4 Flash");
    const fallback = trace.fallback_used === true
      || app.session?.agent_runtime?.last_error
      || provider.provider === "deterministic_fallback";
    const confirmedOnline = trace.http_status === 200 && !fallback;
    const ready = providerReady();

    badge.classList.remove("pending", "online", "degraded", "offline");
    if (confirmedOnline) {
      badge.classList.add("online");
      select("#providerState").textContent = "ONLINE";
      select("#providerHeadline").textContent = `${model} 已完成本轮调用`;
      select("#providerDetail").textContent = "本轮诊断、Skill 提议和候选教师话语来自在线模型；服务端校验通过才直接展示，否则明确标记修复或回退。";
    } else if (ready && !fallback) {
      badge.classList.add("online");
      select("#providerState").textContent = "ONLINE READY";
      select("#providerHeadline").textContent = `${model} 在线模式已就绪`;
      select("#providerDetail").textContent = "开始会话或提交学生回答后，页面会同时确认在线调用和最终教师话语来源。";
    } else {
      badge.classList.add("degraded");
      select("#providerState").textContent = "DEGRADED";
      select("#providerHeadline").textContent = "确定性规则降级模式";
      select("#providerDetail").textContent = "当前不能建立自由文本语义诊断；系统仍可演示状态机，但会明确标记 fallback，不能当作模型准确率。";
    }
    const commandSupport = commandControlsSupported();
    select("#fallbackSignalField").hidden = provider.provider !== "deterministic_fallback";
    select("#commandAvailability").textContent = commandSupport
      ? "在线会话可用"
      : "降级模式仅自动策略；手动 Skill 与 /stop 不可用";
    select("#manualModeButton").title = commandSupport
      ? "用指定 Skill 覆盖下一轮自动决策"
      : "确定性后端未实现手动 Skill 覆盖";
    if (!commandSupport && app.controlMode === "manual") setControlMode("auto");
    syncControls();
  }

  function renderNeuralGate() {
    const manifest = object(app.bootstrap?.neural_v1);
    const gate = object(manifest.materialization_gate);
    const passed = gate.passed === true;
    const badge = select("#neuralBadge");
    badge.classList.remove("provisional", "online", "offline");
    badge.classList.add(passed ? "online" : "provisional");
    select("#neuralState").textContent = passed ? "GATE PASSED" : "PROVISIONAL";
    select("#neuralHeadline").textContent = passed
      ? "neural-v1 已通过物化门槛"
      : "neural-v1 仅作暂定执行本体";
    if (Object.keys(gate).length) {
      const eligible = finite(gate.eligible_prediction_count, 0);
      const excluded = finite(gate.excluded_prediction_count, 0);
      select("#neuralDetail").textContent = passed
        ? `门槛记录：${eligible} 条合格预测；运行时可按正式 materialized Skill 使用。`
        : `门槛未通过：合格 ${eligible} 条，排除 ${excluded} 条。当前只能称为 provisional，不能冒充已训练定稿。`;
    } else {
      select("#neuralDetail").textContent = "未载入 neural-v1 门槛清单；页面按 provisional 处理，不作正式物化声明。";
    }
  }

  function setBusy(busy) {
    app.busy = busy;
    syncControls();
  }

  function syncStartButtonCopy() {
    const button = select("#startButton");
    const label = button.querySelector("span");
    const hint = button.querySelector("small");
    if (!app.session) {
      label.textContent = "开始学习";
      hint.textContent = "先生成一个教学动作";
      return;
    }
    if (app.draftingReplacement) {
      label.textContent = `切换到${selectedProfile().name}并开始`;
      hint.textContent = "成功后替换当前会话；失败仍保留旧会话";
      return;
    }
    label.textContent = "开始新会话";
    hint.textContent = "展开任务设置后可更换目标或画像";
  }

  function syncControls() {
    syncStartButtonCopy();
    const active = app.session?.status === "active" && !app.draftingReplacement;
    const commandSupport = commandControlsSupported();
    for (const control of select("#setupForm").querySelectorAll("input, textarea, select, button")) {
      if (app.busy) {
        if (!("busyDisabled" in control.dataset)) {
          control.dataset.busyDisabled = String(control.disabled);
        }
        control.disabled = true;
      } else if ("busyDisabled" in control.dataset) {
        control.disabled = control.dataset.busyDisabled === "true";
        delete control.dataset.busyDisabled;
      }
    }
    select("#startButton").disabled = app.busy;
    select("#stepButton").disabled = app.busy || !active;
    select("#autoModeButton").disabled = app.busy;
    select("#manualModeButton").disabled = app.busy || !commandSupport;
    const manualEditorOpen = app.controlMode === "manual" || app.manualDraftOpen;
    select("#skillOverrideSelect").disabled = app.busy || !commandSupport || !manualEditorOpen;
    select("#applySkillButton").disabled = app.busy
      || !commandSupport
      || (app.session && !active && !app.draftingReplacement)
      || !manualEditorOpen
      || !select("#skillOverrideSelect").value;
    select("#learnerResponse").disabled = app.busy || !active;
    select("#attachImageButton").disabled = app.busy || !active;
    select("#answerImageInput").disabled = app.busy || !active;
    select("#removeAttachmentButton").disabled = app.busy || !active;
    select("#fallbackSignalInput").disabled = app.busy
      || !active
      || select("#fallbackSignalField").hidden;
    const cancelButton = select("#cancelTurnButton");
    const cancellableTurn = app.busy
      && Boolean(app.pendingTurn?.active)
      && active
      && commandSupport;
    cancelButton.hidden = !cancellableTurn;
    cancelButton.disabled = !cancellableTurn || app.stopRequested;
    cancelButton.textContent = app.stopRequested ? "正在停止…" : "停止生成";
    renderAttachmentConfirmation();
    for (const card of document.querySelectorAll("[data-profile-id]")) {
      card.disabled = app.busy;
    }
  }

  function setRange(input, value) {
    input.value = String(Math.round(finite(value, 0) * 100));
    if (input.nextElementSibling) input.nextElementSibling.value = input.value;
  }

  function readRange(id) {
    return Math.max(0, Math.min(1, finite(select(`#${id}`).value, 0) / 100));
  }

  function formMastery() {
    return {
      prerequisite: readRange("initialPrerequisite"),
      conceptual: readRange("initialConceptual"),
      procedural: readRange("initialProcedural"),
      transfer: readRange("initialTransfer")
    };
  }

  function renderProfileMiniRing(profileId, mastery) {
    const values = Object.values(object(mastery)).map((value) => finite(value, 0));
    const score = Math.round(
      values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length) * 100
    );
    const ring = document.querySelector(
      `[data-profile-id="${profileId}"] .profile-mini-ring`
    );
    if (!ring) return;
    ring.style.setProperty("--profile-score", String(score));
    ring.setAttribute("aria-label", `综合初始掌握度 ${score}%`);
    const label = ring.querySelector("b");
    if (label) label.textContent = `${score}%`;
  }

  function splitList(value) {
    return String(value || "")
      .split(/[\n,，;；]+/)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function splitLines(value) {
    return String(value || "")
      .split(/\r?\n/)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function knowledgeRowText(value, preferredKeys) {
    if (typeof value === "string") return value.trim();
    const row = object(value);
    for (const key of preferredKeys) {
      const rendered = textValue(row[key], "");
      if (rendered) return rendered;
    }
    return "";
  }

  function knowledgeRows(value, preferredKeys) {
    return array(value)
      .map((item) => knowledgeRowText(item, preferredKeys))
      .filter(Boolean);
  }

  function copyJSON(value) {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (_error) {
      return {};
    }
  }

  function rowKey(value, preferredKeys) {
    return knowledgeRowText(value, preferredKeys).replace(/\s+/g, " ").trim();
  }

  function preserveKnowledgeRows(originalRows, lines, preferredKeys, makeNew, sanitize) {
    const unused = array(originalRows).map((item) => object(copyJSON(item)));
    let newCount = 0;
    const rows = lines.map((line, index) => {
      const matchIndex = unused.findIndex(
        (item) => rowKey(item, preferredKeys) === line.replace(/\s+/g, " ").trim()
      );
      if (matchIndex >= 0) {
        const [original] = unused.splice(matchIndex, 1);
        return sanitize({...original}, line, index, false);
      }
      newCount += 1;
      return sanitize(makeNew(line, index), line, index, true);
    });
    return {rows, newCount};
  }

  function parseScopedKnowledgeLine(line, activeComponents) {
    const raw = String(line || "").trim();
    const separator = raw.indexOf("::");
    if (separator <= 0) return {text: raw, components: []};
    const labels = raw.slice(0, separator)
      .split(/[,|]/)
      .map((item) => item.trim())
      .filter((item) => activeComponents.has(item));
    const text = raw.slice(separator + 2).trim();
    return labels.length && text ? {text, components: labels} : {text: raw, components: []};
  }

  function updateKnowledgeSpecStatus() {
    const components = splitLines(select("#knowledgeComponentsInput").value);
    const evidenceRows = [
      "#canonicalClaimsInput",
      "#rubricCriteriaInput",
      "#acceptedAlternativesInput",
      "#misconceptionCatalogInput"
    ].reduce((count, id) => count + splitLines(select(id).value).length, 0);
    const status = select("#knowledgeSpecStatus");
    status.textContent = evidenceRows
      ? `${components.length} 个知识点 · ${evidenceRows} 条依据`
      : (components.length ? `${components.length} 个知识点 · 未提供评分依据` : "未提供");
    status.classList.toggle("provided", evidenceRows > 0);
  }

  function writeKnowledgeSpecForm(goal) {
    const normalizedGoal = object(goal);
    const spec = object(normalizedGoal.knowledge_spec);
    app.knowledgeSpecDraft = copyJSON(spec);
    select("#knowledgeComponentsInput").value = array(normalizedGoal.knowledge_components)
      .map((item) => String(item || "").trim())
      .filter(Boolean)
      .join("\n");
    select("#canonicalClaimsInput").value = knowledgeRows(
      spec.canonical_claims,
      ["statement", "claim"]
    ).join("\n");
    select("#rubricCriteriaInput").value = knowledgeRows(
      spec.rubric_criteria,
      ["description", "criterion"]
    ).join("\n");
    select("#acceptedAlternativesInput").value = knowledgeRows(
      spec.accepted_alternatives,
      ["description", "alternative"]
    ).join("\n");
    select("#misconceptionCatalogInput").value = knowledgeRows(
      spec.misconception_catalog,
      ["description", "tag"]
    ).join("\n");
    updateKnowledgeSpecStatus();
  }

  function formKnowledgeSpec() {
    const knowledgeComponents = splitLines(select("#knowledgeComponentsInput").value);
    const canonicalClaims = splitLines(select("#canonicalClaimsInput").value);
    const rubricCriteria = splitLines(select("#rubricCriteriaInput").value);
    const acceptedAlternatives = splitLines(select("#acceptedAlternativesInput").value);
    const misconceptionCatalog = splitLines(select("#misconceptionCatalogInput").value);
    const original = object(app.knowledgeSpecDraft);
    const activeComponents = new Set(knowledgeComponents);
    const sourceId = "source_teacher_workbench";
    const canonical = preserveKnowledgeRows(
      original.canonical_claims,
      canonicalClaims,
      ["statement", "claim"],
      (statement, index) => ({
        ...(() => {
          const parsed = parseScopedKnowledgeLine(statement, activeComponents);
          return {statement: parsed.text, knowledge_components: parsed.components};
        })(),
        claim_id: `workbench_claim_${String(index + 1).padStart(2, "0")}`,
        required: true,
        source_ids: [sourceId]
      }),
      (row, statement, _index, isNew) => {
        const parsed = parseScopedKnowledgeLine(statement, activeComponents);
        return {
          ...row,
          statement: parsed.text,
        ...(isNew ? {source_ids: [sourceId]} : {}),
          knowledge_components: isNew
            ? parsed.components
            : array(row.knowledge_components)
              .map((item) => String(item || "").trim())
              .filter((item) => activeComponents.has(item))
        };
      }
    );
    const rubric = preserveKnowledgeRows(
      original.rubric_criteria,
      rubricCriteria,
      ["description", "criterion"],
      (description, index) => ({
        ...(() => {
          const parsed = parseScopedKnowledgeLine(description, activeComponents);
          return {description: parsed.text, knowledge_component: parsed.components[0] || ""};
        })(),
        criterion_id: `workbench_criterion_${String(index + 1).padStart(2, "0")}`,
        required: true,
        acceptable_evidence: []
      }),
      (row, description) => ({...row, description})
    );
    const alternatives = preserveKnowledgeRows(
      original.accepted_alternatives,
      acceptedAlternatives,
      ["description", "alternative"],
      (description, index) => ({
        alternative_id: `workbench_alternative_${String(index + 1).padStart(2, "0")}`,
        description,
        equivalent_claim_ids: [],
        conditions: []
      }),
      (row, description) => ({...row, description})
    );
    const misconceptions = preserveKnowledgeRows(
      original.misconception_catalog,
      misconceptionCatalog,
      ["description", "tag"],
      (description, index) => ({
        tag: `workbench_misconception_${String(index + 1).padStart(2, "0")}`,
        description,
        contradicts_claim_ids: [],
        corrective_principle: ""
      }),
      (row, description) => ({...row, description})
    );
    const claimIds = new Set(canonical.rows.map((row) => String(row.claim_id || "")));
    const sanitizedAlternatives = alternatives.rows.map((row) => ({
      ...row,
      equivalent_claim_ids: array(row.equivalent_claim_ids)
        .map((item) => String(item || "").trim())
        .filter((item) => claimIds.has(item))
    }));
    const sanitizedMisconceptions = misconceptions.rows.map((row) => ({
      ...row,
      contradicts_claim_ids: array(row.contradicts_claim_ids)
        .map((item) => String(item || "").trim())
        .filter((item) => claimIds.has(item))
    }));
    const provided = canonical.rows.length
      + rubricCriteria.length
      + acceptedAlternatives.length
      + misconceptionCatalog.length > 0;
    const sources = array(original.sources).map((item) => object(copyJSON(item)));
    const needsWorkbenchSource = canonical.newCount + rubric.newCount
      + alternatives.newCount + misconceptions.newCount > 0;
    if (needsWorkbenchSource && !sources.some((item) => item.source_id === sourceId)) {
      sources.push({
        source_id: sourceId,
        title: "当前工作台中的教师知识规格",
        citation: "由当前操作者在本次会话设置中提供；系统未独立验证其学科正确性。",
        kind: "teacher_authored_runtime_input"
      });
    }
    return {
      knowledgeComponents,
      knowledgeSpec: {
        canonical_claims: canonical.rows,
        rubric_criteria: rubric.rows,
        accepted_alternatives: sanitizedAlternatives,
        reference_steps: array(original.reference_steps).map((item) => ({
          ...object(copyJSON(item)),
          knowledge_components: array(item.knowledge_components)
            .map((component) => String(component || "").trim())
            .filter((component) => activeComponents.has(component))
        })),
        misconception_catalog: sanitizedMisconceptions,
        sources: provided ? sources : []
      }
    };
  }

  function profileByReference(reference) {
    return Object.values(studentProfiles).find((profile) => profile.ref === reference) || null;
  }

  function selectedProfile() {
    return studentProfiles[app.selectedProfileId] || studentProfiles.xiaoyu;
  }

  function activeProfile() {
    return studentProfiles[app.activeProfileId] || selectedProfile();
  }

  function profileDraftExtras(profile = selectedProfile()) {
    const draft = object(app.profileDrafts[profile.id]);
    return {
      conversationHistory: draft.conversationHistory === undefined
        ? [...array(profile.conversationHistory)]
        : [...array(draft.conversationHistory)],
      accessibilityNeeds: draft.accessibilityNeeds === undefined
        ? [...array(profile.accessibilityNeeds)]
        : [...array(draft.accessibilityNeeds)],
      containsDirectIdentity: draft.containsDirectIdentity === undefined
        ? profile.containsDirectIdentity === true
        : draft.containsDirectIdentity === true
    };
  }

  function renderProfileIdentity() {
    const profile = app.session ? activeProfile() : selectedProfile();
    select("#conversationStudentAvatar").src = profile.avatar;
    select("#conversationStudentAvatar").alt = `${profile.name}的合成插画头像`;
    select("#conversationStudentName").textContent = `${profile.name} · ${profile.type}`;
  }

  function captureProfileDraft() {
    if (!app.profileDraftReady) return;
    const profile = selectedProfile();
    const extras = profileDraftExtras(profile);
    const mastery = formMastery();
    app.profileDrafts[profile.id] = {
      learnerLevel: select("#learnerLevelInput").value,
      mastery,
      preferences: select("#preferencesInput").value,
      misconceptions: select("#knownMisconceptionsInput").value,
      backgroundHistory: select("#historyInput").value,
      ...extras,
      revision: app.profileRevision,
      editCounter: app.profileEditCounter
    };
    renderProfileMiniRing(profile.id, mastery);
  }

  function restoreProfileDraft(profile) {
    const draft = object(app.profileDrafts[profile.id]);
    const storedMastery = object(draft.mastery);
    const mastery = Object.keys(storedMastery).length ? storedMastery : profile.mastery;
    select("#learnerLevelInput").value = textValue(draft.learnerLevel, profile.level);
    setRange(select("#initialPrerequisite"), mastery.prerequisite);
    setRange(select("#initialConceptual"), mastery.conceptual);
    setRange(select("#initialProcedural"), mastery.procedural);
    setRange(select("#initialTransfer"), mastery.transfer);
    renderProfileMiniRing(profile.id, mastery);
    select("#preferencesInput").value = draft.preferences === undefined
      ? profile.preferences.join("，")
      : String(draft.preferences);
    select("#knownMisconceptionsInput").value = draft.misconceptions === undefined
      ? profile.misconceptions.join("\n")
      : String(draft.misconceptions);
    select("#historyInput").value = draft.backgroundHistory === undefined ? "" : String(draft.backgroundHistory);
    const extras = profileDraftExtras(profile);
    app.profileDrafts[profile.id] = {...draft, ...extras};
    app.profileRevision = textValue(draft.revision, profile.revision);
    app.profileEditCounter = Math.max(0, Math.round(finite(draft.editCounter, 0)));
    app.profileDraftReady = true;
  }

  function renderDraftProfileBaseline() {
    const mastery = formMastery();
    renderProfileMiniRing(selectedProfile().id, mastery);
    renderMastery({knowledge_mastery: mastery});
    const values = Object.values(mastery).map((value) => finite(value, 0));
    const score = Math.round(values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length) * 100);
    select("#masteryRing").setAttribute(
      "aria-label",
      `综合掌握度 ${score}%，选中画像的四项初始估计等权平均`
    );
  }

  function applyProfile(profileId, {announce = false, resetDrafts = false} = {}) {
    const profile = studentProfiles[profileId];
    if (!profile || app.busy) return;
    if (resetDrafts) {
      app.profileDrafts = {};
      app.profileDraftReady = false;
    } else {
      captureProfileDraft();
    }
    app.selectedProfileId = profile.id;
    app.profileEpoch += 1;
    restoreProfileDraft(profile);
    app.pendingStart = null;
    clearPendingTurn();
    clearPendingCommand();
    for (const card of document.querySelectorAll("[data-profile-id]")) {
      const selected = card.dataset.profileId === profile.id;
      card.classList.toggle("selected", selected);
      card.setAttribute("aria-checked", String(selected));
      card.tabIndex = selected ? 0 : -1;
    }
    if (!app.session) setControlMode("auto");
    if (!app.session) {
      renderProfileIdentity();
      renderDraftProfileBaseline();
    } else {
      beginReplacementDraft({resetControl: true});
    }
    syncReplacementDraftUi();
    if (announce) {
      showToast(app.session
        ? `已选择“${profile.name} · ${profile.type}”；新会话默认 AUTO，旧学生的手动 Skill 不会继承。提交成功前，当前会话仍完整保留。`
        : `已选择“${profile.name} · ${profile.type}”。`);
    }
  }

  function markProfileEdited() {
    if (app.busy) return;
    app.profileEpoch += 1;
    app.profileEditCounter += 1;
    app.profileRevision = `${selectedProfile().revision}-custom-${app.profileEditCounter}`;
    app.pendingStart = null;
  }

  function applySessionProfile(session, {syncDraft = false} = {}) {
    const summary = object(session.profile_summary);
    const profile = profileByReference(summary.profile_ref);
    if (profile) {
      app.activeProfileId = profile.id;
      if (syncDraft) {
        app.profileEpoch += 1;
        app.selectedProfileId = profile.id;
        app.profileRevision = textValue(summary.profile_revision, profile.revision);
        const revisionCounter = app.profileRevision.match(/-custom-(\d+)$/);
        app.profileEditCounter = revisionCounter ? Math.max(0, finite(revisionCounter[1], 0)) : 0;
        app.profileDraftReady = true;
        select("#learnerLevelInput").value = textValue(summary.learner_level, profile.level);
        const mastery = object(summary.initial_mastery);
        setRange(select("#initialPrerequisite"), mastery.prerequisite ?? profile.mastery.prerequisite);
        setRange(select("#initialConceptual"), mastery.conceptual ?? profile.mastery.conceptual);
        setRange(select("#initialProcedural"), mastery.procedural ?? profile.mastery.procedural);
        setRange(select("#initialTransfer"), mastery.transfer ?? profile.mastery.transfer);
        select("#preferencesInput").value = array(summary.preferences).join("，");
        for (const card of document.querySelectorAll("[data-profile-id]")) {
          const selected = card.dataset.profileId === profile.id;
          card.classList.toggle("selected", selected);
          card.setAttribute("aria-checked", String(selected));
          card.tabIndex = selected ? 0 : -1;
        }
      }
    }
    renderProfileIdentity();
  }

  function applySetupSnapshot(session) {
    const snapshot = object(session.setup_snapshot);
    const goal = object(snapshot.goal);
    const profile = object(snapshot.student_profile);
    if (!Object.keys(goal).length || !Object.keys(profile).length) return false;

    const materials = object(goal.materials);
    const thresholds = object(goal.success_thresholds);
    select("#conceptInput").value = textValue(goal.concept, "");
    select("#objectiveInput").value = textValue(goal.objective, "");
    select("#maxRoundsInput").value = String(
      Math.max(3, Math.min(50, Math.round(finite(goal.max_rounds, 12))))
    );
    select("#exampleInput").value = textValue(materials.example, "");
    select("#practiceInput").value = textValue(materials.practice, "");
    select("#transferInput").value = textValue(materials.transfer_task, "");
    writeKnowledgeSpecForm(goal);
    setRange(select("#thresholdPrerequisite"), thresholds.prerequisite ?? 0.6);
    setRange(select("#thresholdConceptual"), thresholds.conceptual ?? 0.65);
    setRange(select("#thresholdProcedural"), thresholds.procedural ?? 0.6);
    setRange(select("#thresholdTransfer"), thresholds.transfer ?? 0.55);

    select("#learnerLevelInput").value = textValue(
      profile.learner_level,
      selectedProfile().level
    );
    const mastery = object(profile.initial_mastery);
    setRange(select("#initialPrerequisite"), mastery.prerequisite ?? selectedProfile().mastery.prerequisite);
    setRange(select("#initialConceptual"), mastery.conceptual ?? selectedProfile().mastery.conceptual);
    setRange(select("#initialProcedural"), mastery.procedural ?? selectedProfile().mastery.procedural);
    setRange(select("#initialTransfer"), mastery.transfer ?? selectedProfile().mastery.transfer);
    select("#preferencesInput").value = array(profile.preferences).join("，");
    select("#knownMisconceptionsInput").value = array(profile.known_misconceptions)
      .map((item) => {
        if (typeof item === "string") return item.trim();
        const misconception = object(item);
        return textValue(misconception.description || misconception.tag, "");
      })
      .filter(Boolean)
      .join("\n");
    select("#historyInput").value = array(profile.background_history)
      .map((item) => String(item || "").trim())
      .filter(Boolean)
      .join("\n");
    app.profileDrafts[selectedProfile().id] = {
      ...object(app.profileDrafts[selectedProfile().id]),
      conversationHistory: [...array(profile.conversation_history)],
      accessibilityNeeds: [...array(profile.accessibility_needs)],
      containsDirectIdentity: profile.contains_direct_identity === true
    };
    app.lastSetupPayload = {goal};
    app.profileDraftReady = true;
    captureProfileDraft();
    updateWorkspaceIdentity(goal);
    return true;
  }

  function fillSetupForm() {
    if (!app.bootstrap) return;
    const goal = object(app.bootstrap.default_goal);
    const materials = object(goal.materials);
    const thresholds = object(goal.success_thresholds);
    select("#conceptInput").value = textValue(goal.concept, "动态规划的状态与转移");
    select("#objectiveInput").value = textValue(goal.objective, "学习者能够解释概念并完成一个迁移任务。");
    select("#maxRoundsInput").value = String(Math.max(3, Math.min(50, finite(goal.max_rounds, 12))));
    select("#exampleInput").value = textValue(materials.example, "");
    select("#practiceInput").value = textValue(materials.practice, "");
    select("#transferInput").value = textValue(materials.transfer_task, "");
    writeKnowledgeSpecForm(goal);
    applyProfile("xiaoyu", {resetDrafts: true});
    setRange(select("#thresholdPrerequisite"), thresholds.prerequisite ?? 0.6);
    setRange(select("#thresholdConceptual"), thresholds.conceptual ?? 0.65);
    setRange(select("#thresholdProcedural"), thresholds.procedural ?? 0.6);
    setRange(select("#thresholdTransfer"), thresholds.transfer ?? 0.55);
    select("#remoteConsent").checked = false;
    updateWorkspaceIdentity(goal);
    syncReplacementDraftUi();
  }

  function setupPayload() {
    const misconceptions = splitList(select("#knownMisconceptionsInput").value)
      .map((description, index) => ({
        tag: `prior_${String(index + 1).padStart(2, "0")}`,
        description,
        confidence: 0.8
      }));
    const backgroundHistory = splitLines(select("#historyInput").value);
    const extras = profileDraftExtras();
    const {knowledgeComponents, knowledgeSpec} = formKnowledgeSpec();
    return {
      goal: {
        concept: select("#conceptInput").value.trim(),
        objective: select("#objectiveInput").value.trim(),
        ...(knowledgeComponents.length ? {knowledge_components: knowledgeComponents} : {}),
        knowledge_spec: knowledgeSpec,
        success_thresholds: {
          prerequisite: readRange("thresholdPrerequisite"),
          conceptual: readRange("thresholdConceptual"),
          procedural: readRange("thresholdProcedural"),
          transfer: readRange("thresholdTransfer")
        },
        max_rounds: Math.max(3, Math.min(50, Math.round(finite(select("#maxRoundsInput").value, 12)))),
        materials: {
          example: select("#exampleInput").value.trim() || "请教师补充一个最小例子",
          practice: select("#practiceInput").value.trim() || "请教师补充一道单步练习",
          transfer_task: select("#transferInput").value.trim() || "请教师补充一个新情境"
        }
      },
      student_profile: {
        profile_ref: selectedProfile().ref,
        learner_level: select("#learnerLevelInput").value,
        preferences: splitList(select("#preferencesInput").value),
        initial_mastery: {
          prerequisite: readRange("initialPrerequisite"),
          conceptual: readRange("initialConceptual"),
          procedural: readRange("initialProcedural"),
          transfer: readRange("initialTransfer")
        },
        known_misconceptions: misconceptions,
        conversation_history: extras.conversationHistory,
        background_history: backgroundHistory,
        accessibility_needs: extras.accessibilityNeeds,
        contains_direct_identity: extras.containsDirectIdentity
      },
      allowed_skill_ids: array(app.bootstrap?.skills).map((skill) => skill.skill_id),
      remote_processing_acknowledged: select("#remoteConsent").checked,
      profile_revision: app.profileRevision,
      profile_display_name: selectedProfile().name
    };
  }

  function populateSkillSelect() {
    const dropdown = select("#skillOverrideSelect");
    const placeholder = node("option", "", "选择一个主 Skill");
    placeholder.value = "";
    const options = array(app.bootstrap?.skills)
      .filter((skill) => primaryRoles.has(textValue(skill.role, "")))
      .map((skill) => {
        const option = node("option", "", `${textValue(skill.name)} · ${roleLabels[skill.role] || skill.role}`);
        option.value = skill.skill_id;
        return option;
      });
    dropdown.replaceChildren(placeholder, ...options);
  }

  function setControlMode(mode, skillId = app.manualSkillId) {
    const nextMode = mode === "manual" ? "manual" : "auto";
    const nextSkillId = nextMode === "manual" ? textValue(skillId, "") : "";
    if (app.controlMode !== nextMode || app.manualSkillId !== nextSkillId) clearPendingTurn();
    app.controlMode = nextMode;
    app.manualSkillId = nextSkillId;
    select("#autoModeButton").classList.toggle("active", app.controlMode === "auto");
    select("#manualModeButton").classList.toggle("active", app.controlMode === "manual");
    select("#autoModeButton").setAttribute("aria-pressed", String(app.controlMode === "auto"));
    select("#manualModeButton").setAttribute("aria-pressed", String(app.controlMode === "manual"));
    select("#skillOverrideSelect").value = app.manualSkillId;
    select("#runtimeMode").textContent = app.controlMode === "auto" ? "AUTO" : "MANUAL";
    syncControls();
  }

  function currentAction(session = app.session) {
    return object(session?.next_action || session?.current_action);
  }

  function syncNewMessageAnnouncer(session = app.session) {
    const announcer = select("#newMessageAnnouncer");
    if (!session) {
      announcer.textContent = "";
      return;
    }
    if (session.status !== "active") {
      announcer.textContent = "本次教学会话已经结束。";
      return;
    }
    const message = textValue(object(currentAction(session)).teacher_action?.message, "");
    announcer.textContent = message ? `老师的新问题：${message}` : "";
  }

  function skillName(skillId) {
    const match = array(app.bootstrap?.skills).find((item) => item.skill_id === skillId);
    return match ? match.name : textValue(skillId);
  }

  function actionProvenanceSummary(actionValue) {
    const action = object(actionValue);
    const provenance = object(action.action_provenance);
    const trace = object(action.model_trace);
    const origin = textValue(provenance.executor_origin, "");
    const requestedMode = textValue(provenance.requested_executor_mode, "");
    const reasons = [
      ...array(provenance.model_action_validation_reasons),
      ...array(provenance.normalization_reasons)
    ].map((reason) => textValue(reason, "")).filter(Boolean);
    const fallback = trace.fallback_used === true
      || action.decision_origin === "deterministic_safety_fallback"
      || origin === "deterministic_safety_fallback"
      || app.bootstrap?.provider_status?.provider === "deterministic_fallback";
    if (fallback) {
      return {
        key: "fallback",
        label: "确定性回退",
        detail: "在线模型计划不可用或未通过安全门禁，本轮由可复现的规则动作接管；不能把该话语算作模型生成。"
      };
    }
    if (provenance.model_teacher_action_used === true || origin === "deepseek_safe_generative") {
      return {
        key: "model",
        label: "模型生成",
        detail: provenance.message_preserved_verbatim === true
          ? "DeepSeek 候选教师话语通过 Skill、安全和问题契约校验，本轮按原文展示。"
          : "DeepSeek 候选教师话语通过校验；仅叠加了已公开的辅助 Skill 约束。"
      };
    }
    if (origin.includes("repair") || (origin === "deterministic_materializer" && requestedMode === "safe_generative")) {
      const readableReasons = reasons
        .map((reason) => provenanceReasonLabels[reason] || (reason.startsWith("diagnosis_or_routing_normalized:") ? "诊断或路由已按契约校正" : "候选动作未通过一项执行约束"))
        .filter((reason, index, values) => values.indexOf(reason) === index)
        .slice(0, 2);
      return {
        key: "repaired",
        label: "契约修复",
        detail: `模型参与了诊断与路由，但教师话语由服务端按所选 Skill 重新生成${readableReasons.length ? `：${readableReasons.join("；")}` : "。"}`
      };
    }
    if (origin === "deterministic_materializer" || requestedMode === "deterministic_legacy") {
      return {
        key: "deterministic",
        label: "规则生成",
        detail: "当前配置使用确定性 Skill 执行器生成教师话语，不是在线模型原样输出。"
      };
    }
    if (action.type === "terminate" || action.termination_reason) {
      return {
        key: "deterministic",
        label: "终止判定",
        detail: "该动作来自显式停止条件，系统不会继续生成教学内容。"
      };
    }
    return {
      key: "unknown",
      label: "来源未记录",
      detail: "当前会话响应没有提供 action_provenance；不能推断该话语由模型还是规则生成。"
    };
  }

  function applyOriginBadge(element, summary) {
    element.textContent = summary.label;
    element.dataset.origin = summary.key;
    element.title = summary.detail;
  }

  function renderActionProvenance(action) {
    const summary = actionProvenanceSummary(action);
    applyOriginBadge(select("#currentActionOrigin"), summary);
    applyOriginBadge(select("#actionProvenanceLabel"), summary);
    select("#actionProvenanceDetail").textContent = summary.detail;
  }

  function normalizeSignal(value) {
    const raw = typeof value === "string" ? value : object(value).label || object(value).signal;
    return textValue(raw, "not_observed");
  }

  function historyAction(event) {
    const raw = object(event.action);
    if (Object.keys(raw).length) return raw;
    return {
      primary_skill: {
        skill_id: event.skill_id,
        name: event.skill_name,
        role: event.skill_role
      },
      skill_switched: event.skill_switched,
      selection_reason: event.selection_reason,
      teacher_action: {message: event.teacher_message},
      action_provenance: event.action_provenance,
      decision_origin: event.decision_origin,
      model_trace: event.model_trace
    };
  }

  function historyStateAfter(event) {
    const full = object(event.student_state_after_observation || event.student_state_after);
    if (Object.keys(full).length) return full;
    return {knowledge_mastery: object(event.mastery_after)};
  }

  function historyStateBefore(event, previousMastery) {
    const full = object(event.student_state_before);
    if (Object.keys(full).length) return full;
    return {knowledge_mastery: object(previousMastery)};
  }

  function latestHistoryStates() {
    const history = array(app.session?.history);
    if (!history.length) return {before: null, after: object(app.session?.student_state)};
    const event = object(history.at(-1));
    let priorMastery = object(app.sessionInitialMastery);
    if (history.length > 1) {
      priorMastery = object(historyStateAfter(object(history.at(-2))).knowledge_mastery);
    }
    return {
      before: historyStateBefore(event, priorMastery),
      after: historyStateAfter(event)
    };
  }

  function renderMastery(state, beforeState = null) {
    const mastery = object(state.knowledge_mastery);
    const before = object(object(beforeState).knowledge_mastery);
    const masteryValues = [];
    for (const dimension of Object.keys(dimensionLabels)) {
      const value = Math.max(0, Math.min(1, finite(mastery[dimension], 0)));
      masteryValues.push(value);
      const bar = select(`#${dimension}Bar`);
      const valueNode = select(`#${dimension}Value`);
      const deltaNode = select(`#${dimension}Delta`);
      bar.value = value;
      bar.setAttribute("aria-label", `${dimensionLabels[dimension]} ${probability(value)}`);
      valueNode.textContent = probability(value);
      deltaNode.classList.remove("positive", "negative");
      if (Object.prototype.hasOwnProperty.call(before, dimension)) {
        const delta = value - finite(before[dimension], value);
        deltaNode.textContent = `${delta > 0 ? "+" : ""}${(delta * 100).toFixed(1)} pt`;
        if (delta > 0) deltaNode.classList.add("positive");
        if (delta < 0) deltaNode.classList.add("negative");
      } else {
        deltaNode.textContent = "初始";
      }
    }
    const average = masteryValues.reduce((sum, value) => sum + value, 0) / Math.max(1, masteryValues.length);
    const score = Math.round(average * 100);
    select("#masteryAverage").textContent = `${score}%`;
    select("#masteryRing").style.setProperty("--mastery-score", String(score));
    select("#masteryRing").setAttribute("aria-label", `当前综合掌握度 ${score}%，四项等权平均`);
  }

  function renderMisconceptions(state) {
    const rows = array(state.misconceptions);
    const active = rows.filter((item) => item.status !== "resolved");
    select("#misconceptionCount").textContent = `${active.length} active`;
    const root = select("#misconceptionList");
    if (!rows.length) {
      root.replaceChildren(node("p", "", "尚未记录明确误解。"));
      return;
    }
    root.replaceChildren(...rows.map((raw) => {
      const item = object(raw);
      const resolved = item.status === "resolved";
      const card = node("div", `misconception-item${resolved ? " resolved" : ""}`);
      card.append(
        node("strong", "", `${resolved ? "RESOLVED" : "ACTIVE"} · ${textValue(item.tag, "未分类")}`),
        node("p", "", textValue(item.description, "未提供误解描述"))
      );
      return card;
    }));
  }

  function renderAdaptiveStudentProfile(session) {
    const profile = object(session.adaptive_student_profile);
    const observations = array(profile.observations);
    const latest = object(observations.at(-1));
    const candidate = object(latest.candidate);
    const evidence = object(latest.evidence);
    const status = select("#adaptiveProfileStatus");
    const hasCandidate = Object.keys(candidate).length > 0;
    const needsReview = evidence.needs_human_review === true;
    const lacksGroundedExcerpt = evidence.grounding === "no_grounded_excerpt";

    status.classList.toggle("pending", hasCandidate && !needsReview);
    status.classList.toggle("review", hasCandidate && needsReview);
    status.textContent = hasCandidate ? (needsReview ? "需人工复核" : "未确认") : "等待观察";
    select("#adaptiveResponseQuality").textContent = hasCandidate
      ? (responseQualityLabels[candidate.response_quality] || textValue(candidate.response_quality))
      : "—";
    select("#adaptiveEngagement").textContent = hasCandidate
      ? (engagementLabels[candidate.engagement_level] || textValue(candidate.engagement_level))
      : "—";
    select("#adaptiveNextFocus").textContent = hasCandidate
      ? (dimensionLabels[candidate.next_focus] || textValue(candidate.next_focus))
      : "—";
    select("#adaptiveConfidence").textContent = hasCandidate ? probability(evidence.confidence) : "—";
    select("#adaptiveConfidenceBar").value = hasCandidate
      ? Math.max(0, Math.min(1, finite(evidence.confidence, 0)))
      : 0;
    select("#adaptiveProfileNotice").textContent = hasCandidate
      ? lacksGroundedExcerpt
        ? "本轮未绑定到学生原话片段，必须人工复核；不会覆盖教师输入。"
        : "仅作下一轮决策候选，不会覆盖教师输入。"
      : "有效 DeepSeek 回合后生成；教师输入保持原值。";
    select("#adaptiveProfileRound").textContent = hasCandidate
      ? `R${finite(latest.round, 0)} · candidate_unconfirmed`
      : "尚无候选观察";
  }

  function renderRanking(action) {
    const root = select("#candidateRanking");
    const rows = array(action.candidate_ranking).slice(0, 4);
    if (!rows.length) {
      root.replaceChildren(node("p", "", app.session?.status === "active" ? "当前动作没有返回候选排序。" : "会话已停止，无下一步候选。"));
      return;
    }
    root.replaceChildren(...rows.map((raw, index) => {
      const item = object(raw);
      const row = node("div", "candidate-row");
      row.title = array(item.reasons).join("；");
      row.append(
        node("span", "", String(index + 1).padStart(2, "0")),
        node("strong", "", skillName(item.skill_id)),
        node("code", "", numberScore(item.score, 1))
      );
      return row;
    }));
  }

  function renderSupportingSkills(action) {
    const root = select("#supportingSkills");
    const skills = array(action.supporting_skills);
    if (!skills.length) {
      root.replaceChildren(node("span", "", "本轮未组合支持 Skill"));
      return;
    }
    root.replaceChildren(...skills.map((raw) => {
      const skill = object(raw);
      const chip = node("span", "", textValue(skill.name || skillName(skill.skill_id)));
      chip.title = textValue(skill.executed_as, "作为动作约束执行");
      return chip;
    }));
  }

  function renderAssessment(session) {
    const history = array(session.history);
    const latest = object(history.at(-1));
    const state = object(session.student_state);
    const stateSignal = object(state.understanding_signal);
    const diagnosis = object(latest.deepseek_assessment);
    const assessmentEvidence = object(state.assessment_evidence);
    const structuredSignal = object(latest.structured_signal || latest.signal);
    const signalSource = textValue(structuredSignal.source || stateSignal.source, "no_current_turn_observation");
    const assessmentSource = textValue(diagnosis.assessment_source || signalSource, signalSource);
    const rawDeepseekAssessment = assessmentSource === "deepseek_v4_flash";
    const contractConstrainedAssessment = assessmentSource === "deepseek_v4_flash_constrained_by_deterministic_contract";
    const activeContractExactAssessment = assessmentSource === "active_question_contract_exact_match";
    const teacherKnowledgeExactAssessment = assessmentSource === "teacher_knowledge_spec_exact_match";
    const teacherGoalBoundedAssessment = assessmentSource === "teacher_goal_knowledge_component_bounded_match";
    const exactGroundedAssessment = activeContractExactAssessment
      || teacherKnowledgeExactAssessment
      || teacherGoalBoundedAssessment;
    const confidenceBearingAssessment = Object.keys(diagnosis).length > 0
      && (rawDeepseekAssessment || contractConstrainedAssessment || exactGroundedAssessment);
    const safetyFallback = signalSource === "deterministic_safety_fallback" || Boolean(latest.model_error);
    const signal = textValue(diagnosis.signal || stateSignal.label, "not_observed");
    const confidence = diagnosis.confidence ?? state.assessment_confidence ?? stateSignal.confidence;
    const evidence = textValue(diagnosis.evidence_excerpt || assessmentEvidence.excerpt || stateSignal.response_excerpt, "等待第一条学生回答。");
    const reason = textValue(diagnosis.diagnosis_reason || assessmentEvidence.reason, "尚无本轮诊断依据。");
    const answerAlignment = textValue(
      diagnosis.answer_alignment || assessmentEvidence.answer_alignment,
      history.length ? "ambiguous" : "not_applicable"
    );
    const review = diagnosis.needs_human_review === true || assessmentEvidence.needs_human_review === true;
    const lowConfidence = finite(confidence, 0) < 0.35;

    const sourceLabelKey = !history.length
      ? "AWAITING STUDENT RESPONSE"
      : (rawDeepseekAssessment
        ? "DEEPSEEK ASSESSMENT"
        : (contractConstrainedAssessment
          ? "DEEPSEEK + CONTRACT GUARD"
          : (activeContractExactAssessment
            ? "ACTIVE CONTRACT EXACT MATCH"
            : (teacherKnowledgeExactAssessment
              ? "TEACHER KNOWLEDGE EXACT MATCH"
              : (teacherGoalBoundedAssessment
                ? "TEACHER GOAL BOUNDED MATCH"
                : (safetyFallback ? "SAFETY FALLBACK SIGNAL" : "STRUCTURED DEMO SIGNAL"))))));
    select("#assessmentSourceLabel").textContent = assessmentSourceLabels[sourceLabelKey] || sourceLabelKey;
    select("#assessmentStatus").textContent = history.length
      ? (confidenceBearingAssessment
        ? (exactGroundedAssessment
          ? (teacherKnowledgeExactAssessment
            ? (review ? "教师知识标准精确命中 · 仍建议复核" : "教师知识标准精确命中")
            : (teacherGoalBoundedAssessment
              ? (review ? "教师目标知识点有界命中 · 仍建议复核" : "教师目标知识点有界命中")
              : (review ? "本问契约精确命中 · 仍建议复核" : "本问契约精确命中")))
          : (contractConstrainedAssessment
            ? (review ? "约束层修正 · 建议人工确认" : "约束层修正已记录")
            : (review
              ? (lowConfidence ? "低置信度 · 建议人工确认" : "模型建议人工确认")
              : "本轮在线诊断已记录")))
        : (safetyFallback ? "规则回退 · 非自由文本识别" : "人工标签 · 非自由文本识别"))
      : (commandControlsSupported() ? "等待第一条学生回答" : "等待人工降级演示信号");
    select("#assessmentLabel").textContent = signalLabels[signal] || signal;
    select("#answerAlignmentLabel").textContent = answerAlignmentLabels[answerAlignment] || answerAlignment;
    select("#assessmentConfidence").textContent = signal === "not_observed"
      ? "—"
      : (confidenceBearingAssessment ? probability(confidence, 0) : (safetyFallback ? "规则回退" : "人工输入"));
    select("#assessmentEvidence").textContent = history.length
      ? (confidenceBearingAssessment
        ? (exactGroundedAssessment
          ? (teacherKnowledgeExactAssessment
            ? `“${evidence}”——当前回答精确命中教师提供的知识标准；${reason}`
            : (teacherGoalBoundedAssessment
              ? `“${evidence}”——当前回答在教师目标限定的知识点范围内形成有界命中；${reason}`
              : `“${evidence}”——当前回答精确命中服务端绑定的本问契约；${reason}`))
          : (contractConstrainedAssessment
            ? `“${evidence}”——DeepSeek 诊断经确定性契约修正；${reason}`
            : `“${evidence}”——${reason}`))
        : `“${evidence}”——${safetyFallback ? "在线调用失败后由安全规则给出该标签" : "该标签由演示者选择，系统没有执行自由文本识别"}`)
      : reason;
  }

  function renderGoalPlan(session) {
    const plan = object(session.goal_plan);
    let steps = array(plan.intermediate_objectives);
    let progress = object(plan.progress);
    if (!steps.length) {
      const mastery = object(session.student_state?.knowledge_mastery);
      const thresholds = object(session.goal?.success_thresholds);
      let activeAssigned = false;
      steps = Object.keys(dimensionLabels).map((dimension, index) => {
        const completed = finite(mastery[dimension], 0) >= finite(thresholds[dimension], 1);
        const current = !completed && !activeAssigned;
        if (current) activeAssigned = true;
        return {
          step_id: `goal_${index + 1}`,
          dimension,
          description: `${dimensionLabels[dimension]}达到 ${probability(thresholds[dimension])}`,
          status: completed ? "completed" : (current ? "active" : "pending"),
          progress: Math.min(1, finite(mastery[dimension], 0) / Math.max(finite(thresholds[dimension], 1), 0.001))
        };
      });
      const completed = steps.filter((step) => step.status === "completed").length;
      progress = {completed_steps: completed, total_steps: steps.length, fraction: completed / Math.max(1, steps.length)};
    }
    const completed = finite(progress.completed_steps, steps.filter((step) => step.status === "completed").length);
    const total = finite(progress.total_steps, steps.length);
    const fraction = Math.max(0, Math.min(1, finite(progress.fraction, total ? completed / total : 0)));
    select("#goalProgressText").textContent = `${completed} / ${total}`;
    select("#goalProgressBar").value = fraction;
    const root = select("#goalSteps");
    if (!steps.length) {
      root.replaceChildren(node("p", "", "当前后端没有返回 Goal 分解。"));
      return;
    }
    root.replaceChildren(...steps.map((raw, index) => {
      const step = object(raw);
      const status = textValue(step.status, "pending");
      const row = node("div", `goal-step${status === "completed" ? " completed" : ""}${status === "active" ? " current" : ""}`);
      row.append(
        node("i", ""),
        node("span", "", textValue(step.description || step.objective, `${index + 1}. ${dimensionLabels[step.dimension] || step.dimension}`))
      );
      return row;
    }));
  }

  function contextText(value, preferredKeys = []) {
    if (typeof value === "string" || typeof value === "number") {
      return compactText(value, 180);
    }
    if (Array.isArray(value)) {
      return value.map((item) => contextText(item, preferredKeys)).filter((item) => item && item !== "—").join("；");
    }
    const record = object(value);
    for (const key of preferredKeys) {
      if (record[key] !== undefined && record[key] !== null) {
        const rendered = contextText(record[key]);
        if (rendered && rendered !== "—") return rendered;
      }
    }
    return "";
  }

  function contextItems(value) {
    if (Array.isArray(value)) {
      return value.map((item) => {
        if (typeof item === "string") return compactText(item, 100);
        const record = object(item);
        return contextText(record, ["description", "question", "learner_response_excerpt", "objective", "name", "tag", "dimension", "focus_dimension", "issue_kind", "skill_id", "text"]);
      }).filter(Boolean);
    }
    const record = object(value);
    return Object.entries(record).filter(([, item]) => item !== false && item !== null && item !== undefined)
      .map(([key, item]) => {
        const label = dimensionLabels[key] || key;
        if (typeof item === "boolean") return label;
        if (typeof item === "number") return `${label} ${item <= 1 ? probability(item) : numberScore(item, 0)}`;
        return contextText(item, ["description", "name", "tag", "text"]) || label;
      });
  }

  function renderContextMemory(session) {
    const memory = object(session.context_memory);
    const snapshot = object(memory.snapshot);
    const snapshotRoundPresent = Object.prototype.hasOwnProperty.call(snapshot, "session_round_before_request");
    const snapshotRound = Number(snapshot.session_round_before_request);
    const operation = textValue(snapshot.operation, "");
    const hasSnapshot = snapshot.kind === "bounded_outbound_model_request_context"
      && ["initial_action", "assess_and_act"].includes(operation)
      && snapshotRoundPresent
      && Number.isInteger(snapshotRound)
      && snapshotRound >= 0;
    if (!hasSnapshot) {
      select("#contextWindowText").textContent = "暂无请求快照";
      select("#contextOperationText").textContent = "—";
      select("#contextSnapshotRound").textContent = "—";
      select("#contextBudgetText").textContent = "—";
      select("#contextHistoryText").textContent = "—";
      select("#contextSummary").textContent = "暂无请求快照。最新学生状态请看上方“知识掌握度”“这轮理解信号”和“接下来重点”；本区不会混入这些处理后结果。";
      select("#contextMemoryList").replaceChildren(node("li", "", "暂无请求快照。"));
      select("#contextBoundaryText").textContent = "只有后端返回带轮次与操作类型的模型请求快照后，本区才会展示内容。";
      return;
    }
    const working = object(memory.working_memory);
    const semantic = object(memory.semantic_memory || memory.semantic_summary);
    const knowledge = object(memory.knowledge_state);
    const fixedContext = object(memory.fixed_context);
    const teacherProfile = object(fixedContext.teacher_provided_student_profile);
    const retrieval = object(memory.retrieval);
    const budget = object(memory.budget);
    const selection = object(memory.selection);
    const providedHistory = object(teacherProfile.provided_prior_context);
    const providedHistoryCount = Math.max(0, finite(providedHistory.total_turn_count, 0));
    const recent = array(
      working.recent_turns
      || retrieval.recent_turns
      || retrieval.selected_turns
      || memory.recent_turns
    );
    const visibleRecent = recent;
    const earlier = object(
      semantic.earlier_summary
      || retrieval.earlier_summary
      || memory.earlier_summary
      || semantic
    );
    const maxRecent = Math.max(0, finite(
      budget.max_recent_turns
      ?? retrieval.max_recent_turns
      ?? memory.max_recent_turns,
      6
    ));
    const selectedCount = Math.max(0, finite(
      retrieval.selected_turn_count
      ?? selection.selected_turn_count,
      Math.min(visibleRecent.length, maxRecent)
    ));
    const maxChars = Math.max(1, finite(
      budget.max_chars
      ?? budget.character_limit
      ?? memory.max_chars,
      14000
    ));
    const serializedChars = Number.isFinite(Number(budget.serialized_chars))
      ? Math.max(0, Number(budget.serialized_chars))
      : null;
    const earlierCount = Math.max(0, finite(earlier.turn_count, 0));
    const teachingCheckpoints = array(earlier.teaching_checkpoints);
    const goalAnchor = contextText(
      memory.goal_anchor || fixedContext.teaching_goal,
      ["objective", "concept", "summary", "text"]
    ) || "请求中未提供目标锚点";
    const plan = object(memory.current_plan);
    const currentPlan = contextText(
      plan.active_step || plan,
      ["active_step_description", "description", "objective", "next_focus", "step_id"]
    );
    const knowledgeMisconceptions = array(knowledge.misconceptions)
      .filter((item) => object(item).status !== "resolved");
    let activeMisconceptions = contextItems(
      working.active_misconceptions
      || semantic.active_misconceptions
      || memory.active_misconceptions
      || knowledgeMisconceptions
    );
    let confirmedMastery = contextItems(
      semantic.confirmed_mastery
      || working.confirmed_mastery
      || memory.confirmed_mastery
    );
    let masteryMemoryLabel = "已确认掌握";
    if (!confirmedMastery.length) {
      masteryMemoryLabel = "达到阈值的掌握估计";
      confirmedMastery = array(knowledge.concept_mastery)
        .filter((item) => {
          const row = object(item);
          return Number.isFinite(Number(row.value))
            && Number.isFinite(Number(row.success_threshold))
            && Number(row.value) >= Number(row.success_threshold);
        })
        .map((item) => {
          const row = object(item);
          return `${dimensionLabels[row.dimension] || textValue(row.dimension, "掌握项")}达到当前阈值（${probability(row.value)}）`;
        });
    }
    const knowledgeUnresolved = array(knowledge.unresolved_issues);
    let unresolved = contextItems(
      working.unresolved_questions
      || semantic.unresolved_questions
      || memory.unresolved_questions
      || knowledgeUnresolved
    );
    const operationLabel = operation === "initial_action"
      ? "initial_action · 生成首个动作"
      : "assess_and_act · 评估并生成下一步";
    select("#contextWindowText").textContent = `请求保留 ${selectedCount} / ${maxRecent} 轮`;
    select("#contextOperationText").textContent = operationLabel;
    select("#contextSnapshotRound").textContent = `R${snapshotRound}`;
    select("#contextBudgetText").textContent = `${serializedChars === null ? "—" : serializedChars.toLocaleString("zh-CN")} / ${maxChars.toLocaleString("zh-CN")}`;
    select("#contextHistoryText").textContent = String(providedHistoryCount + earlierCount);
    select("#contextSummary").textContent = `这是提交前 R${snapshotRound}、执行 ${operation} 时模型实际读取的请求快照：围绕“${compactText(goalAnchor, 86)}”组织内容，保留近期原始教学语义，并把更早历史压缩为确定性统计与原文抽取检查点。`;

    const items = [];
    if (currentPlan) items.push(`当前计划：${compactText(currentPlan, 120)}`);
    if (visibleRecent.length) {
      const rounds = visibleRecent
        .map((item) => Number(object(item).round))
        .filter((round) => Number.isInteger(round) && round >= 0);
      if (rounds.length) items.push(`最近相关回合：${rounds.map((round) => `R${round}`).join("、")}`);
    }
    if (earlierCount) {
      const signalCounts = Object.entries(object(earlier.signal_counts))
        .map(([signal, count]) => `${signalLabels[signal] || signal} ${finite(count, 0)}`)
        .join("、");
      items.push(`较早摘要：${earlierCount} 轮已压缩为计数${signalCounts ? `（${signalCounts}）` : ""}`);
    }
    if (teachingCheckpoints.length) {
      const checkpointLabels = {
        unresolved_learning_signal: "待解决理解信号",
        explicit_learner_question: "学生明确提问",
        explicit_learner_preference_or_constraint: "学习偏好或约束",
        verified_prerequisite: "已核验前置知识",
        teacher_next_step_statement: "教师承诺的下一步"
      };
      const checkpointText = teachingCheckpoints.slice(0, 3).map((raw) => {
        const checkpoint = object(raw);
        const label = checkpointLabels[checkpoint.kind] || textValue(checkpoint.kind, "教学事实");
        const excerpt = compactText(checkpoint.excerpt, 54);
        return `${label}${excerpt === "—" ? "" : `：“${excerpt}”`}`;
      }).join("；");
      items.push(`证据检查点：${checkpointText}`);
    }
    if (activeMisconceptions.length) items.push(`活跃误解：${activeMisconceptions.slice(0, 3).join("；")}`);
    if (confirmedMastery.length) items.push(`${masteryMemoryLabel}：${confirmedMastery.slice(0, 3).join("；")}`);
    if (unresolved.length) items.push(`未解决问题：${unresolved.slice(0, 2).join("；")}`);
    if (!items.length) items.push("该请求快照未携带可展示的教学记忆。 ");
    const root = select("#contextMemoryList");
    root.replaceChildren(...items.slice(0, 7).map((item) => node("li", "", item)));

    const truncated = budget.truncated === true || retrieval.truncated === true || selection.truncated === true || memory.truncated === true;
    select("#contextBoundaryText").textContent = truncated
      ? `该请求快照已按预算裁剪，只反映提交前 R${snapshotRound} 的状态；最新结果以上方学生状态区为准。这里只展示可审计范围，不展示模型私有推理。`
      : `该请求快照只反映提交前 R${snapshotRound} 的状态；最新结果以上方学生状态区为准。这里只展示可审计范围，不展示模型私有推理。`;
  }

  function visibleTeachingMemory(session) {
    const directProjection = object(session.teaching_memory_projection);
    if (Object.keys(directProjection).length) return directProjection;
    const direct = object(session.teaching_memory);
    if (Object.keys(direct).length) return direct;
    const contextMemory = object(session.context_memory);
    const semantic = object(contextMemory.semantic_memory || contextMemory.semantic_summary);
    const semanticProjection = object(semantic.teaching_memory);
    if (Object.keys(semanticProjection).length) return semanticProjection;
    return object(object(contextMemory.snapshot).teaching_memory);
  }

  function teachingMemoryRows(memory, projectionField, fullField) {
    const projected = array(memory[projectionField]);
    return projected.length ? projected : array(memory[fullField]);
  }

  function teachingMemoryEvidence(item) {
    const refs = array(item.evidence_refs).map((ref) => textValue(ref, "")).filter(Boolean);
    const answerRefs = array(item.answer_evidence_refs).map((ref) => textValue(ref, "")).filter(Boolean);
    const combined = [...refs, ...answerRefs].filter((ref, index, values) => values.indexOf(ref) === index);
    return combined.length ? combined.join(" · ") : "证据指针未公开";
  }

  function renderTeachingMemoryList(rootId, countId, rows, preferredKeys, emptyText) {
    select(`#${countId}`).textContent = String(rows.length);
    const root = select(`#${rootId}`);
    if (!rows.length) {
      root.replaceChildren(node("li", "", emptyText));
      return;
    }
    root.replaceChildren(...rows.map((raw) => {
      const item = object(raw);
      const entry = node("li", "");
      entry.append(node("span", "", compactText(contextText(item, preferredKeys), 130)));
      const evidence = node("small", "", teachingMemoryEvidence(item));
      evidence.title = teachingMemoryEvidence(item);
      entry.append(evidence);
      return entry;
    }));
  }

  function renderTeachingMemory(session) {
    const memory = visibleTeachingMemory(session);
    const preferences = teachingMemoryRows(memory, "active_preferences", "preferences")
      .filter((item) => object(item).status !== "superseded");
    const questions = teachingMemoryRows(memory, "unresolved_questions", "open_questions")
      .filter((item) => object(item).status !== "resolved");
    const commitments = teachingMemoryRows(memory, "pending_teacher_commitments", "commitments")
      .filter((item) => !object(item).status || object(item).status === "pending");
    const referents = teachingMemoryRows(memory, "active_referents", "referents")
      .filter((item) => !object(item).status || object(item).status === "active");
    const itemCount = preferences.length + questions.length + commitments.length + referents.length;
    const version = Math.max(0, Math.round(finite(memory.history_version, 0)));
    const generation = Math.max(0, Math.round(finite(memory.compaction_generation, 0)));
    select("#teachingMemoryStatus").textContent = `V${version} · ${itemCount} 项`;
    renderTeachingMemoryList("memoryPreferences", "memoryPreferenceCount", preferences, ["statement", "description"], "尚无明确偏好。");
    renderTeachingMemoryList("memoryQuestions", "memoryQuestionCount", questions, ["question", "description"], "尚无待解决问题。");
    renderTeachingMemoryList("memoryCommitments", "memoryCommitmentCount", commitments, ["statement", "description"], "尚无待兑现承诺。");
    renderTeachingMemoryList("memoryReferents", "memoryReferentCount", referents, ["description", "statement"], "尚无跨轮指代。");
    select("#teachingMemoryBoundary").textContent = Object.keys(memory).length
      ? `历史版本 V${version} · 压缩代次 ${generation}。每项必须指向可观察原话；它们只维持对话连续性，不充当学科标准答案，也不允许模型直接改写。`
      : "当前后端尚未返回长期教学记忆；页面不会用前端猜测补齐偏好、问题、承诺或指代。";
  }

  function masteryDeltaNodes(beforeState, afterState) {
    const before = object(object(beforeState).knowledge_mastery);
    const after = object(object(afterState).knowledge_mastery);
    const nodes = [];
    for (const dimension of Object.keys(dimensionLabels)) {
      if (!Object.prototype.hasOwnProperty.call(after, dimension)) continue;
      const delta = finite(after[dimension], 0) - finite(before[dimension], after[dimension]);
      const item = node("span", delta > 0 ? "positive" : (delta < 0 ? "negative" : ""), `${dimensionLabels[dimension]} ${delta > 0 ? "+" : ""}${(delta * 100).toFixed(1)} pt`);
      nodes.push(item);
    }
    return nodes.length ? nodes : [node("span", "", "本轮无可比状态")];
  }

  function detailCell(label, content, wide = false) {
    const cell = node("div", wide ? "wide" : "");
    cell.append(node("span", "", label), node("p", "", textValue(content)));
    return cell;
  }

  function renderHistory(history) {
    select("#historyCount").textContent = `${history.length} 轮`;
    const root = select("#historyList");
    if (!history.length) {
      root.replaceChildren(node("p", "", "老师已经准备好第一步，正在等待你的回答。"));
      return;
    }
    let previousMastery = object(app.sessionInitialMastery);
    const normalized = history.map((raw) => {
      const event = object(raw);
      const before = historyStateBefore(event, previousMastery);
      const after = historyStateAfter(event);
      previousMastery = object(after.knowledge_mastery);
      return {event, before, after};
    });
    const turns = normalized.map(({event, before, after}, index) => {
      const action = historyAction(event);
      const skill = object(action.primary_skill);
      const diagnosis = object(event.deepseek_assessment);
      const signal = normalizeSignal(event.structured_signal || event.signal || diagnosis.signal);
      const turn = node("section", "chat-turn");
      const teacherRow = node("article", "chat-message teacher");
      const teacherAvatar = node("div", "message-avatar", "T");
      teacherAvatar.setAttribute("aria-hidden", "true");
      const teacherBubble = node("div", "chat-bubble");
      const teacherMeta = node("div", "chat-meta");
      const provenanceSummary = actionProvenanceSummary(action);
      const provenanceChip = node("span", "action-origin-chip", provenanceSummary.label);
      provenanceChip.dataset.origin = provenanceSummary.key;
      provenanceChip.title = provenanceSummary.detail;
      teacherMeta.append(
        node("span", "", `老师 · R${finite(action.round, Math.max(0, finite(event.round, index + 1) - 1))}`),
        node("strong", "", textValue(skill.name || event.skill_name, "教学动作")),
        ...(action.skill_switched || event.skill_switched ? [node("span", "", "已切换方法")] : []),
        provenanceChip
      );
      teacherBubble.append(
        teacherMeta,
        node("p", "", object(action.teacher_action).message || event.teacher_message || "（该轮教师动作未公开）")
      );
      teacherRow.append(teacherAvatar, teacherBubble);

      const studentRow = node("article", "chat-message student");
      const studentAvatar = node("div", "message-avatar student-photo");
      studentAvatar.setAttribute("aria-hidden", "true");
      const studentImage = document.createElement("img");
      studentImage.src = activeProfile().avatar;
      studentImage.alt = "";
      studentAvatar.append(studentImage);
      const studentBubble = node("div", "chat-bubble");
      const studentMeta = node("div", "chat-meta");
      studentMeta.append(
        node("span", "", `学生 · R${finite(event.round, index + 1)}`),
        node("strong", "", signalLabels[signal] || signal)
      );
      const visualEvidence = array(event.multimodal_evidence);
      const typedLearnerText = textValue(event.learner_text, "");
      studentBubble.append(
        studentMeta,
        node(
          "p",
          "",
          typedLearnerText
            || (visualEvidence.length ? `已提交 ${visualEvidence.length} 张答案图片。` : "")
            || event.learner_response
            || object(event.learner_feedback).response
            || "（没有文字回应）"
        )
      );
      if (visualEvidence.length) {
        const visualSummary = node("div", "visual-evidence-summary");
        for (const evidenceItem of visualEvidence) {
          const item = object(evidenceItem);
          const evidenceStatus = item.status === "recognized"
            ? "本机文字识别完成"
            : item.status === "low_confidence"
              ? "本机识别置信度较低"
              : item.status === "no_text_recognized"
                ? "未识别到可靠文字"
                : "本机识别器不可用";
          const confirmationLabel = item.needs_student_confirmation
            ? " · 需学生核对"
            : item.student_confirmed_recognized_text
              ? " · 学生已核对转写"
              : "";
          const visualItem = node("div", "visual-evidence-item");
          const visualMeta = node("span", "");
          visualMeta.append(
            node("strong", "", `${evidenceStatus}${confirmationLabel}`),
            document.createTextNode(` · ${probability(item.confidence, 0)} · 原图未发送`)
          );
          visualItem.append(
            visualMeta,
            node("p", "", compactText(item.recognized_text, 240) === "—"
              ? "需要学生用文字补充或确认图片内容。"
              : item.needs_student_confirmation
                ? `OCR（待核对）：${compactText(item.recognized_text, 240)}`
                : `OCR：${compactText(item.recognized_text, 240)}`)
          );
          visualSummary.append(visualItem);
        }
        studentBubble.append(visualSummary);
      }
      studentRow.append(studentAvatar, studentBubble);

      const details = node("details", "turn-audit");
      const answerAlignment = textValue(
        diagnosis.answer_alignment,
        Object.keys(diagnosis).length ? "ambiguous" : "not_applicable"
      );
      const summary = node("summary", "", "查看诊断依据");
      const evidence = textValue(diagnosis.evidence_excerpt, object(after.understanding_signal).response_excerpt || "降级路径未返回语义证据");
      const diagnosisReason = textValue(diagnosis.diagnosis_reason, "诊断依据缺失，需要重新判断。");
      const trace = object(event.model_trace || action.model_trace);
      const privacy = object(event.privacy_trace || action.privacy_trace);
      const receipt = node("div", "turn-receipt");
      const diagnosisReceipt = node("span", "");
      diagnosisReceipt.append(node("strong", "", "诊断："), document.createTextNode(answerAlignmentLabels[answerAlignment] || signalLabels[signal] || signal));
      const skillReceipt = node("span", "");
      skillReceipt.append(node("strong", "", "Skill："), document.createTextNode(textValue(skill.name || event.skill_name)));
      const provenanceReceipt = node("span", "");
      provenanceReceipt.append(node("strong", "", "话语："), document.createTextNode(provenanceSummary.label));
      const deltaList = node("div", "delta-list");
      deltaList.append(...masteryDeltaNodes(before, after));
      receipt.append(diagnosisReceipt, skillReceipt, provenanceReceipt, deltaList);
      const detail = node("div", "turn-evidence");
      const diagnosisLine = node("p", "");
      diagnosisLine.append(node("b", "", "依据："), document.createTextNode(`“${evidence}”——${diagnosisReason}`));
      const skillLine = node("p", "");
      skillLine.append(node("b", "", "选择理由："), document.createTextNode(textValue(action.selection_reason || event.selection_reason, "未返回选择理由")));
      const proposedSkillId = textValue(action.model_proposed_primary_skill_id, "");
      const routingLine = node("p", "");
      const routingSummary = proposedSkillId
        ? `模型提议 ${skillName(proposedSkillId)} → 最终执行 ${textValue(skill.name || event.skill_name, "未命名 Skill")}${action.primary_skill_was_retargeted ? "（已重定向）" : ""}`
        : `最终执行 ${textValue(skill.name || event.skill_name, "未命名 Skill")}`;
      routingLine.append(
        node("b", "", "路由审计："),
        document.createTextNode(
          `${routingSummary} · ${textValue(action.decision_origin, "来源未记录")}${action.manual_override_applied ? " · 人工锁定已执行" : ""}`
        )
      );
      const provenanceLine = node("p", "");
      provenanceLine.append(
        node("b", "", "话语来源："),
        document.createTextNode(`${provenanceSummary.label}——${provenanceSummary.detail}`)
      );
      const traceLine = node("p", "");
      traceLine.append(node("b", "", "运行："), document.createTextNode(trace.model
        ? `${trace.model} · ${Number.isFinite(Number(trace.latency_ms)) ? `${trace.latency_ms} ms` : "延迟未记录"}${trace.fallback_used ? " · FALLBACK" : ""} · 媒体发送 ${privacy.media_sent === false ? "否" : "未确认"}`
        : "确定性规则路径"));
      detail.append(diagnosisLine, skillLine, routingLine, provenanceLine, traceLine);
      details.append(summary, detail);
      turn.append(teacherRow, studentRow, receipt, details);
      return turn;
    });
    root.replaceChildren(...turns);
  }

  function renderRuntime(session, action) {
    const runtime = object(session.agent_runtime);
    const trace = object(action.model_trace || runtime.last_model_trace || latestTrace());
    const model = textValue(trace.model || runtime.model || app.bootstrap?.provider_status?.model, "deterministic-policy");
    const currentFallback = trace.fallback_used === true
      || action.decision_origin === "deterministic_safety_fallback"
      || Boolean(action.model_error);
    const deterministicMode = app.bootstrap?.provider_status?.provider === "deterministic_fallback";
    const fallbackCount = finite(runtime.fallback_count, currentFallback ? 1 : 0);
    select("#traceModel").textContent = model;
    select("#traceLatency").textContent = Number.isFinite(Number(trace.latency_ms)) ? `${Number(trace.latency_ms).toFixed(0)} ms` : "未记录";
    select("#traceFallback").textContent = deterministicMode ? "RULE MODE" : (currentFallback ? "YES" : "NO");
    select("#runtimeMode").textContent = app.controlMode === "auto" ? "AUTO" : "MANUAL";
    select("#runtimeModel").textContent = model;
    select("#runtimeFallback").textContent = deterministicMode
      ? "RULE MODE"
      : (fallbackCount > 0 ? `TOTAL ${fallbackCount}` : "NO");
    renderProviderStatus();
  }

  function renderTerminal(session, action) {
    clearPendingAttachment();
    const succeeded = session.status === "succeeded";
    select("#selectedSkillName").textContent = succeeded ? "教学目标达标" : "停止并转人工";
    select("#selectedSkillRole").textContent = succeeded ? "SUCCESS" : "UNABLE";
    select("#selectedSkillId").textContent = textValue(action.action_id, "terminal");
    select("#switchBadge").textContent = "终止动作";
    select("#switchBadge").classList.add("switched");
    const terminalSkillLabel = succeeded ? "教学目标达标" : "停止并转人工";
    select("#liveSkillName").textContent = terminalSkillLabel;
    select("#liveSkillName").title = terminalSkillLabel;
    select("#liveSwitchLabel").textContent = "会话终止";
    select("#liveSwitchLabel").title = "会话终止";
    select("#liveSwitchLabel").dataset.switched = "true";
    select("#selectionReason").textContent = textValue(action.termination_reason || session.termination_reason, "会话已进入停止状态。");
    renderSupportingSkills({});
    select("#actionType").textContent = textValue(object(action.teacher_action).type || action.type, "stop");
    select("#teacherMessage").textContent = textValue(object(action.teacher_action).message, "会话已停止。");
    select("#expectedSignal").textContent = "会话已停止，不再自动生成下一步。";
    select("#turnForm").hidden = true;
  }

  function renderActive(session, action) {
    const skill = object(action.primary_skill);
    select("#selectedSkillName").textContent = textValue(skill.name || skillName(skill.skill_id));
    select("#selectedSkillRole").textContent = roleLabels[skill.role] || textValue(skill.role);
    select("#selectedSkillId").textContent = textValue(skill.skill_id);
    const liveSkillLabel = textValue(skill.name || skillName(skill.skill_id));
    select("#liveSkillName").textContent = liveSkillLabel;
    select("#liveSkillName").title = liveSkillLabel;
    const switchBadge = select("#switchBadge");
    if (action.skill_switched) {
      switchBadge.textContent = `从 ${skillName(action.previous_primary_skill_id)} 切换`;
    } else if (action.previous_primary_skill_id) {
      switchBadge.textContent = "继续当前 Skill";
    } else {
      switchBadge.textContent = "首个 Skill";
    }
    switchBadge.classList.toggle("switched", action.skill_switched === true);
    select("#liveSwitchLabel").textContent = switchBadge.textContent;
    select("#liveSwitchLabel").title = switchBadge.textContent;
    select("#liveSwitchLabel").dataset.switched = String(action.skill_switched === true);
    renderSupportingSkills(action);
    const modelProposedSkill = textValue(action.model_proposed_primary_skill_id, "");
    const finalReason = textValue(action.selection_reason, "后端未返回最终执行理由。");
    const modelReason = textValue(action.model_selection_reason, "未返回模型提议理由");
    select("#selectionReason").textContent = modelProposedSkill
      ? `模型提议：${skillName(modelProposedSkill)}（${modelReason}）；最终执行：${textValue(skill.name || skillName(skill.skill_id))}（${finalReason}）`
      : finalReason;
    const teacher = object(action.teacher_action);
    select("#actionType").textContent = textValue(teacher.type, "one_action");
    select("#teacherMessage").textContent = textValue(teacher.message);
    select("#expectedSignal").textContent = textValue(teacher.expected_signal, "等待学生作答后再判断。");
    select("#turnForm").hidden = false;
  }

  function renderSession() {
    const session = app.session;
    if (!session) return;
    const scroller = select("#conversationScroll");
    const followLatest = !scroller || scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 96;
    applySessionProfile(session);
    const action = currentAction(session);
    const active = session.status === "active";
    select("#emptySession").hidden = true;
    select("#activeSession").hidden = false;
    select("#liveDecisionStrip").hidden = false;
    const status = select("#sessionStatus");
    status.textContent = active
      ? `进行中 · 已完成 ${finite(session.rounds_completed, 0)} 轮`
      : statusLabels[session.status] || textValue(session.status);
    status.parentElement.classList.toggle("active", active);
    status.parentElement.classList.toggle("terminal", !active);
    select("#roundCounter").textContent = `R${finite(session.rounds_completed, 0)}`;
    updateWorkspaceIdentity(session.goal);
    renderPhase(session);
    if (active) renderActive(session, action); else renderTerminal(session, action);
    renderActionProvenance(action);

    renderAssessment(session);
    const states = latestHistoryStates();
    const studentState = object(session.student_state);
    renderMastery(studentState, states.before);
    const understanding = object(studentState.understanding_signal);
    select("#understandingSignal").textContent = signalLabels[understanding.label] || textValue(understanding.label, "尚未观察");
    select("#responseExcerpt").textContent = textValue(understanding.response_excerpt, "等待第一轮作答。");
    const focus = object(studentState.next_focus);
    select("#nextFocus").textContent = dimensionLabels[focus.dimension] || textValue(focus.dimension);
    select("#nextFocusReason").textContent = textValue(focus.reason, "后端未返回下一重点依据。");
    const liveFocusLabel = dimensionLabels[focus.dimension] || textValue(focus.dimension, "等待下一轮判断");
    select("#liveNextFocus").textContent = liveFocusLabel;
    select("#liveNextFocus").title = liveFocusLabel;
    renderMisconceptions(studentState);
    renderAdaptiveStudentProfile(session);
    renderRanking(active ? action : {});
    renderGoalPlan(session);
    renderContextMemory(session);
    renderTeachingMemory(session);
    renderHistory(array(session.history));
    renderRuntime(session, action);
    syncControls();
    if (followLatest) {
      window.requestAnimationFrame(() => {
        scroller.scrollTop = scroller.scrollHeight;
      });
    }
  }

  function evaluationTimeline(result) {
    const direct = array(result.timeline || result.history || result.turns);
    if (direct.length) return direct;
    return array(result.selected_skill_ids).map((skillId, index) => ({
      round: index + 1,
      teacher_action: {primary_skill_id: skillId, primary_skill_name: skillName(skillId)},
      learner_feedback: {},
      status_after: index === array(result.selected_skill_ids).length - 1 ? result.status : "active"
    }));
  }

  function caseTurn(raw, previousSkillId) {
    const item = object(raw);
    const teacher = object(item.teacher_action || item.action);
    const primary = object(teacher.primary_skill);
    const skillId = textValue(teacher.primary_skill_id || primary.skill_id, "unknown_skill");
    const skill = textValue(teacher.primary_skill_name || primary.name || skillName(skillId));
    const feedback = object(item.learner_feedback);
    const response = textValue(feedback.response || item.learner_response, "未展示学生回答");
    const signal = textValue(feedback.declared_signal || object(item.structured_signal).label, "—");
    const before = object(object(item.student_state_before).knowledge_mastery);
    const after = object(object(item.student_state_after || item.student_state_after_observation).knowledge_mastery);
    const stateDeltas = Object.keys(dimensionLabels)
      .filter((dimension) => Object.prototype.hasOwnProperty.call(after, dimension))
      .map((dimension) => {
        const delta = finite(after[dimension], 0) - finite(before[dimension], after[dimension]);
        return `${dimensionLabels[dimension]} ${delta > 0 ? "+" : ""}${(delta * 100).toFixed(0)}pt`;
      })
      .filter((itemText) => !itemText.endsWith(" 0pt"));
    const changed = previousSkillId && previousSkillId !== skillId;
    const row = node("div", "case-turn");
    const copy = node("div", "");
    copy.append(
      node("strong", "", `${skill}${changed ? " · 切换" : ""}`),
      node("p", "", `学生：${compactText(response, 120)}`),
      node("p", "", `教师：${compactText(teacher.message, 150)}`),
      node("p", "", `依据：${compactText(teacher.selection_reason, 150)}`),
      node("p", "", `状态：${stateDeltas.length ? stateDeltas.join("；") : "本轮无掌握度变化"}`)
    );
    row.title = textValue(teacher.selection_reason, "未返回该轮选择理由");
    row.append(
      node("span", "", `R${finite(item.round, 0)}`),
      copy,
      node("code", "", `${signalLabels[signal] || signal}${item.status_after && item.status_after !== "active" ? ` · ${statusLabels[item.status_after] || item.status_after}` : ""}`)
    );
    return {row, skillId};
  }

  function renderCaseTimeline(rootId, result) {
    const root = select(rootId);
    const timeline = evaluationTimeline(result);
    if (!timeline.length) {
      root.replaceChildren(node("p", "", "该案例没有公开逐轮轨迹。"));
      return;
    }
    let previousSkillId = "";
    const rows = timeline.map((item) => {
      const built = caseTurn(item, previousSkillId);
      previousSkillId = built.skillId;
      return built.row;
    });
    root.replaceChildren(...rows);
  }

  function renderEvaluationCase() {
    const cases = array(app.bootstrap?.evaluation?.cases);
    const selectedId = select("#evaluationCaseSelect").value;
    const record = object(cases.find((item) => item.case_id === selectedId) || cases[0]);
    if (!Object.keys(record).length) {
      select("#caseOutcome").textContent = "无案例";
      renderCaseTimeline("#adaptiveCaseTimeline", {});
      renderCaseTimeline("#baselineCaseTimeline", {});
      return;
    }
    const adaptive = object(record.adaptive_agent);
    const baseline = object(record.fixed_single_skill_baseline);
    const match = record.terminal_decision_match === true;
    const delta = finite(record.simulated_gain_delta, finite(adaptive.simulated_gain, 0) - finite(baseline.simulated_gain, 0));
    select("#caseOutcome").textContent = `${match ? "终止匹配" : "终止不匹配"} · ${signedPoints(delta)}`;
    renderCaseTimeline("#adaptiveCaseTimeline", adaptive);
    renderCaseTimeline("#baselineCaseTimeline", baseline);
  }

  function renderEvaluation() {
    const evaluation = object(app.bootstrap?.evaluation);
    const aggregate = object(evaluation.aggregate);
    select("#evaluationStatus").textContent = evaluation.passed ? "全部通过" : "存在失败";
    select("#stateRate").textContent = probability(aggregate.student_state_judgement_rate);
    select("#decisionRate").textContent = probability(aggregate.teaching_decision_match_rate);
    select("#behaviorRate").textContent = probability(aggregate.teaching_behavior_quality_rate);
    select("#effectRate").textContent = signedPoints(aggregate.simulated_mean_gain_delta);
    select("#multiSkillRate").textContent = probability(aggregate.adaptive_multi_skill_case_rate);
    select("#terminationRate").textContent = probability(aggregate.terminal_decision_match_rate);
    select("#caseComposition").textContent = `${finite(aggregate.case_count, 0)} 条（${finite(aggregate.expected_success_case_count, 0)} 成功 / ${finite(aggregate.expected_unable_case_count, 0)} 转人工）`;
    select("#adaptiveGain").textContent = signedPoints(aggregate.adaptive_simulated_mean_gain);
    select("#baselineGain").textContent = signedPoints(aggregate.baseline_simulated_mean_gain);
    select("#adaptiveTransfer").textContent = probability(aggregate.adaptive_simulated_transfer_pass_rate);
    select("#baselineTransfer").textContent = probability(aggregate.baseline_simulated_transfer_pass_rate);

    const cases = array(evaluation.cases);
    const dropdown = select("#evaluationCaseSelect");
    if (!cases.length) {
      const empty = node("option", "", "没有公开评测案例");
      empty.value = "";
      dropdown.replaceChildren(empty);
    } else {
      dropdown.replaceChildren(...cases.map((item, index) => {
        const option = node("option", "", `${String(index + 1).padStart(2, "0")} · ${textValue(item.case_id)}`);
        option.value = item.case_id;
        return option;
      }));
    }
    renderOutcomeDemo(evaluation);
    renderFreeTextBenchmark(evaluation);
    renderEvaluationCase();
  }

  function renderFreeTextBenchmark(evaluation) {
    const report = object(evaluation.free_text_benchmark);
    const online = object(report.online_deepseek);
    const root = select("#freeTextBenchmark");
    if (!Object.keys(report).length || !Object.keys(online).length) {
      root.hidden = true;
      return;
    }
    root.hidden = false;
    const metrics = object(online.metrics);
    const signal = object(object(metrics.signal).end_to_end_all_attempts);
    const skill = object(metrics.allowed_primary_skill);
    const decision = object(metrics.decision);
    const operational = object(online.operational);
    const latency = object(operational.all_attempt_wall_latency);
    const count = finite(signal.count, report.run_config?.case_count_per_repeat || 0);
    const receiptPrompt = textValue(report.run_config?.prompt_version, "unknown");
    const currentPrompt = textValue(
      app.bootstrap?.interaction_contract?.current_prompt_version,
      "unknown"
    );
    select("#freeTextBenchmarkStatus").textContent = `DEV EVAL · ${textValue(online.model, "online model")} · n=${count}`;
    select("#benchmarkVersionNotice").textContent = `单轮评测: ${receiptPrompt}；实时会话: ${currentPrompt}`;
    select("#benchmarkSignalAccuracy").textContent = probability(signal.accuracy, 2);
    select("#benchmarkSignalMacroF1").textContent = probability(signal.macro_f1, 2);
    select("#benchmarkSkillHit").textContent = probability(skill.end_to_end_hit_rate, 2);
    select("#benchmarkSwitchF1").textContent = probability(object(decision.should_switch).f1, 2);
    select("#benchmarkTerminationF1").textContent = probability(object(decision.should_terminate).f1, 2);
    select("#benchmarkFailureRate").textContent = probability(operational.end_to_end_failure_rate, 2);
    select("#benchmarkLatency").textContent = Number.isFinite(Number(latency.p50_ms)) && Number.isFinite(Number(latency.p95_ms))
      ? `${Number(latency.p50_ms).toFixed(0)} / ${Number(latency.p95_ms).toFixed(0)} ms`
      : "—";
    const boundary = object(report.claim_boundary);
    select("#freeTextBenchmarkBoundary").textContent = boundary.source_type === "author_constructed_not_expert_validated"
      ? "与实时 Agent 共享诊断语义量表，但使用独立的单轮诊断/路由 prompt；不是完整 live Session 评测。作者构造 development set；非专家验证、非锁箱测试、非真实学生、非部署准确率。"
      : "开发集结果只说明当前报告所覆盖的样例；不自动建立专家效度、锁箱表现、部署准确率或真实学习效果。";
  }

  function outcomeProportion(block) {
    const value = object(block);
    if (Number.isFinite(Number(value.proportion))) return Number(value.proportion);
    if (Number.isFinite(Number(value.earned)) && Number(value.possible) > 0) {
      return Number(value.earned) / Number(value.possible);
    }
    return null;
  }

  function renderOutcomeScore(prefix, block, digits = 0) {
    const scoreBlock = object(block);
    const proportion = outcomeProportion(scoreBlock);
    select(`#${prefix}`).textContent = proportion === null ? "—" : probability(proportion, digits);
    select(`#${prefix}Detail`).textContent = proportion === null
      ? "未载入"
      : `${numberScore(scoreBlock.earned, 0)} / ${numberScore(scoreBlock.possible, 0)}`;
  }

  function renderOutcomeDemo(evaluation) {
    const report = object(evaluation.learning_outcome);
    const scores = object(report.scores);
    renderOutcomeScore("demoPretest", scores.pretest);
    renderOutcomeScore("demoPosttest", scores.posttest);
    renderOutcomeScore("demoTransfer", scores.transfer_test, 2);
    const provenance = textValue(report.provenance, "");
    const labels = {
      author_constructed_demo_not_real: "作者构造演示记录 · 非真实学生",
      teacher_provided_test_record: "教师提供测试记录 · 未独立核验",
      authorized_real_learner_observation: "授权真实记录 · 单例不建立因果效果"
    };
    select("#outcomeProvenance").textContent = labels[provenance] || "未载入学习测量记录";
  }

  async function startSession(event) {
    event.preventDefault();
    if (app.busy) return;
    clearInlineError("#setupError");
    if (providerReady() && !select("#remoteConsent").checked) {
      showInlineError("#setupError", "在线模式需要先确认脱敏文本的远程处理边界；课堂媒体永远不会发送。");
      select("#remoteConsent").focus();
      return;
    }
    setBusy(true);
    let recoveredFromUnavailableSession = false;
    let recoveredFromConcurrentUpdate = false;
    try {
      const startPayload = setupPayload();
      const replacingActiveSession = app.session?.status === "active"
        && Boolean(app.session?.session_id);
      if (replacingActiveSession) {
        const priorRevision = textValue(
          object(app.session.profile_summary).profile_revision,
          ""
        );
        const priorQuestionId = textValue(app.session.expected_question_id, "");
        if (!priorRevision || !priorQuestionId) {
          throw new Error("当前会话缺少安全切换所需的版本信息，请刷新恢复后再试");
        }
        Object.assign(startPayload, {
          replace_session_id: app.session.session_id,
          replace_expected_round: Math.max(0, Math.round(finite(app.session.rounds_completed, 0))),
          replace_expected_question_id: priorQuestionId,
          replace_expected_context_version: Math.max(0, Math.round(finite(app.session.context_version, 1))),
          replace_expected_profile_revision: priorRevision
        });
      }
      if (commandControlsSupported() && app.controlMode === "manual" && app.manualSkillId) {
        startPayload.manual_skill_id = app.manualSkillId;
      }
      let startedSession;
      for (let attempt = 0; attempt < 2; attempt += 1) {
        const startSignature = JSON.stringify(startPayload);
        const reusable = app.pendingStart?.signature === startSignature;
        const startKey = reusable ? app.pendingStart.key : makeIdempotencyKey();
        app.pendingStart = {signature: startSignature, key: startKey};
        startPayload.start_idempotency_key = startKey;
        const requestAnchor = beginRequestAnchor("start", {
          targetProfileId: app.selectedProfileId,
          targetProfileRevision: textValue(startPayload.profile_revision, "")
        });
        try {
          startedSession = await postJson("api/start", startPayload);
          commitSessionResponse(requestAnchor, startedSession, {
            newSession: true,
            targetProfileRevision: textValue(startPayload.profile_revision, ""),
            requireDifferentSession: Boolean(requestAnchor.sessionId),
            requireTargetProfile: true
          });
          break;
        } catch (error) {
          if (isStaleResponseError(error)) throw error;
          const staleReplacement = Boolean(startPayload.replace_session_id)
            && isUnavailableSessionError(error, {replacement: true});
          const staleReplacementGuards = Boolean(startPayload.replace_session_id)
            && isReplacementGuardMismatch(error);
          if (attempt > 0 || (!staleReplacement && !staleReplacementGuards)) {
            throw error;
          }
          delete startPayload.start_idempotency_key;
          app.pendingStart = null;
          if (staleReplacementGuards) {
            captureProfileDraft();
            const synchronized = await synchronizeCurrentSession({preserveOnFailure: true});
            if (synchronized) {
              const synchronizedRevision = textValue(
                object(app.session.profile_summary).profile_revision,
                ""
              );
              const synchronizedQuestionId = textValue(
                app.session.expected_question_id,
                ""
              );
              if (!synchronizedRevision || !synchronizedQuestionId) {
                throw new Error("同步后的会话缺少安全切换所需的版本信息，请刷新恢复后再试");
              }
              Object.assign(startPayload, {
                replace_session_id: app.session.session_id,
                replace_expected_round: Math.max(
                  0,
                  Math.round(finite(app.session.rounds_completed, 0))
                ),
                replace_expected_question_id: synchronizedQuestionId,
                replace_expected_context_version: Math.max(
                  0,
                  Math.round(finite(app.session.context_version, 1))
                ),
                replace_expected_profile_revision: synchronizedRevision
              });
              recoveredFromConcurrentUpdate = true;
              continue;
            }
            if (app.session) throw error;
          }
          recoveredFromUnavailableSession = true;
          discardUnavailableSession({render: false});
          delete startPayload.replace_session_id;
          delete startPayload.replace_expected_round;
          delete startPayload.replace_expected_question_id;
          delete startPayload.replace_expected_context_version;
          delete startPayload.replace_expected_profile_revision;
        }
      }
      if (!startedSession) throw new Error("新会话未能建立");
      app.sessionInitialMastery = {...startPayload.student_profile.initial_mastery};
      app.activeProfileId = app.selectedProfileId;
      app.draftingReplacement = false;
      syncReplacementDraftUi();
      applySessionProfile(app.session, {syncDraft: true});
      applySetupSnapshot(app.session);
      persistSessionHandle();
      app.pendingStart = null;
      app.lastSetupPayload = startPayload;
      app.pendingTurn = null;
      app.pendingCommand = null;
      clearPendingAttachment();
      select("#learnerResponse").value = "";
      synchronizeControlModeFromSession();
      renderSession();
      syncNewMessageAnnouncer();
      showSetupForm(false);
      setSidebar(false);
      const startedWithFallback = finite(
        app.session?.agent_runtime?.fallback_count,
        0
      ) > 0;
      showToast(recoveredFromConcurrentUpdate
        ? "旧会话已在另一标签页推进；系统同步最新状态后完成了画像切换。"
        : recoveredFromUnavailableSession
        ? "旧会话已在其他标签页失效，系统已自动建立新的画像会话。"
        : startedWithFallback
          ? "新画像会话已建立；在线模型暂时不可用，本轮使用可审计的安全规则动作。"
          : "学习已开始：老师只给出了第一步，现在正在等待你的回答。 ");
    } catch (error) {
      if (isStaleResponseError(error)) {
        showInlineError("#setupError", "较早的开始/画像切换响应已被忽略；当前会话和设置均未被覆盖。");
        return;
      }
      if (recoveredFromUnavailableSession) discardUnavailableSession();
      showInlineError("#setupError", `无法开始：${String(error.message || error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function sendCommand(command, skillId = "") {
    if (!app.session || app.session.status !== "active") {
      throw new Error("请先开始一个仍在进行的教学会话");
    }
    if (app.draftingReplacement) {
      throw new Error("正在编辑新画像会话；请先开始新会话，或收起设置后再控制当前会话");
    }
    const payload = {
      command,
      skill_id: skillId || undefined,
      session_id: app.session.session_id,
      expected_round: finite(app.session.rounds_completed, 0),
      expected_question_id: app.session.expected_question_id,
      expected_context_version: finite(app.session.context_version, 1),
      profile_revision: textValue(app.session.profile_summary?.profile_revision, app.profileRevision)
    };
    const requestFingerprint = commandRequestFingerprint(payload);
    const reusable = app.pendingCommand?.fingerprint === requestFingerprint;
    const idempotencyKey = reusable ? app.pendingCommand.key : makeIdempotencyKey();
    payload.command_idempotency_key = idempotencyKey;
    app.pendingCommand = {fingerprint: requestFingerprint, key: idempotencyKey};
    const priorContextVersion = finite(app.session.context_version, 1);
    const requestAnchor = beginRequestAnchor("command");
    let updatedSession;
    try {
      updatedSession = await postJson("api/command", payload);
      commitSessionResponse(requestAnchor, updatedSession);
    } catch (error) {
      if (isStaleResponseError(error)) throw error;
      const synchronized = await synchronizeCurrentSession({preserveOnFailure: true});
      if (synchronized && finite(app.session?.context_version, 1) !== priorContextVersion) {
        clearPendingCommand();
      }
      throw error;
    }
    clearPendingCommand();
    persistSessionHandle();
    synchronizeControlModeFromSession();
    renderSession();
    syncNewMessageAnnouncer();
  }

  async function requestStopGeneration() {
    if (
      !app.busy
      || !app.pendingTurn?.active
      || !app.session
      || app.session.status !== "active"
      || !commandControlsSupported()
      || app.stopRequested
    ) return;
    app.stopRequested = true;
    syncControls();
    try {
      await sendCommand("cancel_turn");
      showToast("已请求停止本轮生成：会话仍可继续，当前文字和图片草稿保持不变。 ");
    } catch (error) {
      app.stopRequested = false;
      syncControls();
      showInlineError("#turnError", `停止生成失败：${String(error.message || error)}`);
    }
  }

  async function applySkillOverride() {
    const skillId = select("#skillOverrideSelect").value;
    if (!skillId || app.busy) return;
    if (!commandControlsSupported()) {
      showToast("确定性降级后端不支持手动 Skill；请启用 DeepSeek 在线会话。 ");
      return;
    }
    if (!app.session || app.draftingReplacement) {
      app.manualDraftOpen = false;
      setControlMode("manual", skillId);
      showToast(`已为新会话预选“${skillName(skillId)}”；首个学生回答后开始锁定，该选择不会改动旧会话。`);
      return;
    }
    setBusy(true);
    try {
      await sendCommand("select_skill", skillId);
      showToast(`已锁定“${skillName(skillId)}”；后续每轮持续使用，输入 /auto 才恢复自动。`);
    } catch (error) {
      app.manualDraftOpen = false;
      synchronizeControlModeFromSession();
      showToast(`无法应用 Skill：${String(error.message || error)}`);
    } finally {
      setBusy(false);
    }
  }

  function findCommandSkill(query) {
    const normalized = query.trim().toLocaleLowerCase("zh-CN");
    return array(app.bootstrap?.skills).find((skill) => primaryRoles.has(skill.role)
      && [skill.skill_id, skill.name].some((value) => String(value).trim().toLocaleLowerCase("zh-CN") === normalized));
  }

  async function maybeRunTextCommand(raw) {
    const commandText = raw.trim();
    if (!commandText.startsWith("/")) return false;
    if (app.draftingReplacement) {
      throw new Error("正在编辑新画像会话；请先开始新会话，或收起设置后再输入教学控制命令");
    }
    if (commandText === "/auto") {
      setBusy(true);
      try {
        await sendCommand("auto");
        select("#learnerResponse").value = "";
        showToast("已恢复自动 Skill 选择。 ");
      } finally {
        setBusy(false);
      }
      return true;
    }
    if (commandText === "/stop") {
      if (!commandControlsSupported()) throw new Error("/stop 仅由 DeepSeek 在线会话后端支持");
      setBusy(true);
      try {
        await sendCommand("stop");
        select("#learnerResponse").value = "";
        showToast("会话已停止，并保留当前学生状态供人工接管。 ");
      } finally {
        setBusy(false);
      }
      return true;
    }
    if (commandText.startsWith("/+skill")) {
      if (!commandControlsSupported()) throw new Error("手动 Skill 命令仅由 DeepSeek 在线会话后端支持");
      const match = findCommandSkill(commandText.slice("/+skill".length));
      if (!match) throw new Error("/+skill 后必须填写一个完整、可用的主 Skill 名称或 ID");
      setBusy(true);
      try {
        await sendCommand("select_skill", match.skill_id);
        select("#learnerResponse").value = "";
        showToast(`已锁定“${match.name}”；后续每轮持续使用，输入 /auto 才恢复自动。`);
      } finally {
        setBusy(false);
      }
      return true;
    }
    throw new Error("未知命令；可用 /+skill 名称、/auto 或 /stop");
  }

  function synchronizeControlModeFromSession() {
    app.manualDraftOpen = false;
    if (app.session?.pending_skill_id) {
      setControlMode("manual", app.session.pending_skill_id);
    } else {
      setControlMode("auto");
    }
  }

  async function synchronizeCurrentSession({preserveOnFailure = false} = {}) {
    const sessionId = app.session?.session_id;
    if (!sessionId) return false;
    const requestAnchor = beginRequestAnchor("sync");
    try {
      const synchronized = await postJson("api/session", {session_id: sessionId});
      commitSessionResponse(requestAnchor, synchronized);
      persistSessionHandle();
      synchronizeControlModeFromSession();
      renderSession();
      syncNewMessageAnnouncer();
      return true;
    } catch (error) {
      if (isStaleResponseError(error)) return false;
      const unavailable = isUnavailableSessionError(error);
      if (preserveOnFailure && !unavailable) return false;
      discardUnavailableSession();
      return false;
    }
  }

  async function submitTurn(event) {
    event.preventDefault();
    if (app.busy || !app.session || app.session.status !== "active") return;
    clearInlineError("#turnError");
    const learnerResponse = select("#learnerResponse").value.trim();
    const hasAttachment = Boolean(app.pendingAttachment);
    if (!learnerResponse && !hasAttachment) {
      showInlineError("#turnError", "请输入自然语言回答，或添加一张答案图片。");
      return;
    }
    if (hasAttachment && learnerResponse.startsWith("/")) {
      showInlineError("#turnError", "教学控制命令不能与答案图片同时提交；请先移除图片。");
      return;
    }
    try {
      if (!hasAttachment && await maybeRunTextCommand(learnerResponse)) return;
    } catch (error) {
      showInlineError("#turnError", `命令未执行：${String(error.message || error)}`);
      return;
    }
    setBusy(true);
    try {
      const attachmentIds = await uploadPendingAttachment();
      if (attachmentNeedsConfirmation() && !learnerResponse) {
        showInlineError(
          "#turnError",
          "这张图片的 OCR 需要你确认。请点击“确认识别文字”，或在输入框写出正确答案；本次不会消耗教学轮。"
        );
        return;
      }
      const round = finite(app.session.rounds_completed, 0);
      const payload = {
        learner_response: learnerResponse,
        attachment_ids: attachmentIds,
        session_id: app.session.session_id,
        expected_round: round,
        expected_question_id: app.session.expected_question_id,
        expected_context_version: finite(app.session.context_version, 1),
        profile_revision: textValue(app.session.profile_summary?.profile_revision, app.profileRevision)
      };
      payload.confirmed_attachment_ids = attachmentIds.filter(
        (attachmentId) => app.pendingAttachment?.uploaded?.attachment_id === attachmentId
          && app.pendingAttachment.ocrTextConfirmed === true
      );
      if (!select("#fallbackSignalField").hidden) {
        payload.signal = select("#fallbackSignalInput").value;
        payload.signal_confidence = 1.0;
        if (payload.signal === "misconception") payload.misconception_tag = "manual_demo_misconception";
      }
      if (commandControlsSupported() && app.controlMode === "manual" && app.manualSkillId) {
        payload.manual_skill_id = app.manualSkillId;
      }
      const requestFingerprint = turnRequestFingerprint(payload);
      const reusable = app.pendingTurn?.fingerprint === requestFingerprint;
      const idempotencyKey = reusable ? app.pendingTurn.key : makeIdempotencyKey();
      payload.idempotency_key = idempotencyKey;
      app.pendingTurn = {fingerprint: requestFingerprint, key: idempotencyKey, active: true};
      // setBusy(true) runs before attachment preparation and therefore cannot
      // yet know that a cancellable model turn exists.  Re-sync after marking
      // the turn active so the real DOM exposes “停止生成” while fetch is in
      // flight; without this, cancellation worked only through direct API
      // calls and the browser button remained hidden for the whole request.
      syncControls();
      const requestAnchor = beginRequestAnchor("step");
      const updatedSession = await postJson("api/step", payload);
      commitSessionResponse(requestAnchor, updatedSession);
      persistSessionHandle();
      app.pendingTurn = null;
      app.pendingCommand = null;
      select("#learnerResponse").value = "";
      clearPendingAttachment();
      synchronizeControlModeFromSession();
      renderSession();
      syncNewMessageAnnouncer();
      const controlNotice = textValue(app.session.control_notice, "");
      showToast(controlNotice
        ? "人工 Skill 与本轮证据或执行边界冲突，系统已安全释放锁定并恢复 AUTO。"
        : app.session.status === "active"
          ? "本轮诊断、状态变化与下一 Skill 已更新。"
          : "系统已依据停止条件结束本次会话。 ");
    } catch (error) {
      const message = String(error.message || error);
      if (app.stopRequested || app.session?.status !== "active") {
        showInlineError(
          "#turnError",
          "生成已停止；当前教学轮没有提交。你的文字输入和答案图片仍保留在页面中。"
        );
        return;
      }
      if (isStaleResponseError(error)) {
        showInlineError(
          "#turnError",
          "较早的教学响应已被忽略；当前文字和答案图片均已保留，请核对最新问题后重试。"
        );
        return;
      }
      if (/expected_(?:round|question_id|context_version)/.test(message)) {
        const synchronized = await synchronizeCurrentSession();
        if (synchronized) {
          clearPendingAttachment();
          showInlineError(
            "#turnError",
            "会话已在另一处更新，页面已同步到最新轮次。文字输入仍保留；图片已解除绑定，请核对当前问题后重新选择。"
          );
        } else {
          showUnavailableSessionNotice("原会话已不可恢复，请检查左侧设置后重新开始。");
        }
      } else if (/no longer available/.test(message)) {
        const synchronized = await synchronizeCurrentSession();
        if (synchronized) {
          clearPendingAttachment();
          showInlineError("#turnError", "会话已更新，请核对当前问题后重新发送。");
        } else {
          showUnavailableSessionNotice("原会话已结束或被替换，请检查左侧设置后重新开始。");
        }
      } else {
        if (app.pendingAttachment) {
          app.pendingAttachment.state = app.pendingAttachment.uploaded ? "ready" : "error";
          renderPendingAttachment({
            status: app.pendingAttachment.uploaded ? "本机证据已就绪 · 可重试发送" : "图片处理失败 · 请重试",
            evidence: app.pendingAttachment.uploaded
              ? `识别文字：${compactText(app.pendingAttachment.uploaded.recognized_text, 220)}`
              : "原图仍只保留在当前页面内存中，没有发送给远程模型。"
          });
        }
        showInlineError("#turnError", `无法提交：${message}`);
      }
    } finally {
      setBusy(false);
      app.stopRequested = false;
      syncControls();
    }
  }

  async function chooseAutoMode() {
    if (app.busy) return;
    if (!app.session || app.session.status !== "active" || app.draftingReplacement) {
      setControlMode("auto");
      return;
    }
    setBusy(true);
    try {
      await sendCommand("auto");
      showToast("下一轮恢复由 Agent 自动选择 Skill。 ");
    } catch (error) {
      showToast(`无法切换自动模式：${String(error.message || error)}`);
    } finally {
      setBusy(false);
    }
  }

  function applyTweaks(values, notifyHost = false) {
    const safe = {
      theme: values.theme === "night" ? "night" : "paper",
      density: values.density === "compact" ? "compact" : "comfortable"
    };
    document.documentElement.dataset.theme = safe.theme;
    document.documentElement.dataset.density = safe.density;
    select("#themeTweak").value = safe.theme;
    select("#densityTweak").value = safe.density;
    try {
      window.localStorage.setItem("teacher_agent_display_v1", JSON.stringify(safe));
    } catch (_error) {
      // Only display preferences are optional; teaching-session content is never stored.
    }
    if (notifyHost) {
      window.parent.postMessage({type: "__edit_mode_set_keys", edits: safe}, "*");
    }
  }

  function initTweaks() {
    let values = window.TEACHER_AGENT_TWEAKS || {theme: "paper", density: "comfortable"};
    try {
      values = {...values, ...JSON.parse(window.localStorage.getItem("teacher_agent_display_v1") || "{}")};
    } catch (_error) {
      // Ignore malformed optional display preferences.
    }
    applyTweaks(values);
    const panel = select("#tweaksPanel");
    const toggle = (visible) => {
      panel.hidden = !visible;
      select("#tweaksButton").setAttribute("aria-expanded", String(visible));
    };
    select("#tweaksButton").addEventListener("click", () => toggle(panel.hidden));
    select("#closeTweaks").addEventListener("click", () => toggle(false));
    select("#themeTweak").addEventListener("change", () => applyTweaks({
      theme: select("#themeTweak").value,
      density: select("#densityTweak").value
    }, true));
    select("#densityTweak").addEventListener("change", () => applyTweaks({
      theme: select("#themeTweak").value,
      density: select("#densityTweak").value
    }, true));
    window.addEventListener("message", (event) => {
      if (event.data?.type === "__activate_edit_mode") toggle(true);
      if (event.data?.type === "__deactivate_edit_mode") toggle(false);
    });
    window.parent.postMessage({type: "__edit_mode_available"}, "*");
  }

  function bindEvents() {
    for (const input of document.querySelectorAll("input[type='range']")) {
      input.addEventListener("input", () => {
        if (input.nextElementSibling) input.nextElementSibling.value = input.value;
        if (input.closest(".mastery-inputs")) {
          markProfileEdited();
          if (!app.session) {
            renderDraftProfileBaseline();
          } else {
            renderProfileMiniRing(selectedProfile().id, formMastery());
          }
        }
      });
    }
    const profileCards = [...document.querySelectorAll("[data-profile-id]")];
    for (const card of profileCards) {
      card.addEventListener("click", () => applyProfile(card.dataset.profileId, {announce: true}));
      card.addEventListener("keydown", (event) => {
        const current = profileCards.indexOf(card);
        let next = null;
        if (["ArrowRight", "ArrowDown"].includes(event.key)) next = (current + 1) % profileCards.length;
        if (["ArrowLeft", "ArrowUp"].includes(event.key)) next = (current - 1 + profileCards.length) % profileCards.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = profileCards.length - 1;
        if (next === null) return;
        event.preventDefault();
        const target = profileCards[next];
        applyProfile(target.dataset.profileId, {announce: true});
        target.focus();
      });
    }
    for (const input of ["#preferencesInput", "#knownMisconceptionsInput", "#historyInput", "#learnerLevelInput"]) {
      select(input).addEventListener("input", markProfileEdited);
    }
    for (const input of [
      "#knowledgeComponentsInput",
      "#canonicalClaimsInput",
      "#rubricCriteriaInput",
      "#acceptedAlternativesInput",
      "#misconceptionCatalogInput"
    ]) {
      select(input).addEventListener("input", () => {
        app.profileEpoch += 1;
        app.pendingStart = null;
        updateKnowledgeSpecStatus();
      });
    }
    for (const button of document.querySelectorAll("[data-quick-response]")) {
      button.addEventListener("click", () => {
        const textarea = select("#learnerResponse");
        textarea.value = button.dataset.quickResponse || "";
        clearInlineError("#turnError");
        textarea.focus();
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);
      });
    }
    const inspectorTabs = [...document.querySelectorAll("[data-inspector-tab]")];
    for (const button of inspectorTabs) {
      button.addEventListener("click", () => setInspectorTab(button.dataset.inspectorTab));
      button.addEventListener("keydown", (event) => {
        const current = inspectorTabs.indexOf(button);
        let next = null;
        if (["ArrowRight", "ArrowDown"].includes(event.key)) next = (current + 1) % inspectorTabs.length;
        if (["ArrowLeft", "ArrowUp"].includes(event.key)) next = (current - 1 + inspectorTabs.length) % inspectorTabs.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = inspectorTabs.length - 1;
        if (next === null) return;
        event.preventDefault();
        const target = inspectorTabs[next];
        setInspectorTab(target.dataset.inspectorTab);
        target.focus();
      });
    }
    select("#setupForm").addEventListener("submit", startSession);
    select("#turnForm").addEventListener("submit", submitTurn);
    select("#attachImageButton").addEventListener("click", () => {
      clearInlineError("#turnError");
      select("#answerImageInput").click();
    });
    select("#answerImageInput").addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (file) chooseAnswerImage(file);
    });
    select("#removeAttachmentButton").addEventListener("click", () => {
      clearPendingAttachment();
      clearInlineError("#turnError");
      select("#learnerResponse").focus();
      syncControls();
    });
    select("#confirmAttachmentTextButton").addEventListener("click", () => {
      const pending = app.pendingAttachment;
      if (!attachmentNeedsConfirmation(pending) || app.busy) return;
      if (!textValue(pending.uploaded?.recognized_text, "")) {
        showInlineError(
          "#turnError",
          "本机没有识别出可核对的文字，请在输入框手动填写图片中的答案。"
        );
        select("#learnerResponse").focus();
        return;
      }
      pending.ocrTextConfirmed = true;
      pending.state = "confirmed";
      clearPendingTurn();
      renderPendingAttachment({
        status: "学生已确认识别文字 · 可发送",
        evidence: `已确认文字：${compactText(pending.uploaded?.recognized_text, 220)}`
      });
      syncControls();
      select("#learnerResponse").focus();
      showToast("已记录你对 OCR 转写的核对；系统仍会依据当前问题契约判断答案。 ");
    });
    select("#cancelTurnButton").addEventListener("click", requestStopGeneration);
    select("#presetButton").addEventListener("click", handleSetupButton);
    select("#conceptInput").addEventListener("input", () => updateWorkspaceIdentity());
    select("#objectiveInput").addEventListener("input", () => updateWorkspaceIdentity());
    select("#autoModeButton").addEventListener("click", chooseAutoMode);
    select("#manualModeButton").addEventListener("click", () => {
      if (!commandControlsSupported()) {
        showToast("确定性降级后端不支持手动 Skill；当前只演示自动规则策略。 ");
        return;
      }
      const selected = select("#skillOverrideSelect").value || app.manualSkillId;
      if (app.session?.status === "active" && !app.draftingReplacement) {
        app.manualDraftOpen = true;
        syncControls();
        select("#skillOverrideSelect").focus();
        showToast("请选择 Skill 并点击应用；服务端确认前仍保持当前控制模式。");
      } else {
        setControlMode("manual", selected);
        showToast(app.draftingReplacement
          ? "这是新画像会话的独立 Skill 草稿；旧会话控制状态保持不变。"
          : "已预选手动 Skill；开始会话后，首个学生回答将按该 Skill 路由。");
      }
    });
    select("#skillOverrideSelect").addEventListener("change", () => {
      const nextSkillId = select("#skillOverrideSelect").value;
      if (app.session?.status === "active" && !app.draftingReplacement) {
        app.manualDraftOpen = true;
        clearPendingTurn();
        clearPendingCommand();
        syncControls();
      } else {
        setControlMode("manual", nextSkillId);
      }
    });
    select("#fallbackSignalInput").addEventListener("change", clearPendingTurn);
    select("#applySkillButton").addEventListener("click", applySkillOverride);
    select("#evaluationCaseSelect").addEventListener("change", renderEvaluationCase);
    select("#learningViewButton").addEventListener("click", () => setAppView("learning"));
    select("#evaluationViewButton").addEventListener("click", () => setAppView("evaluation"));
    select("#backToLearningButton").addEventListener("click", () => setAppView("learning"));
    select("#inspectorToggle").addEventListener("click", () => {
      const open = select("#inspectorToggle").getAttribute("aria-expanded") !== "true";
      if (open && window.matchMedia("(max-width: 860px)").matches) setSidebar(false);
      if (open) app.lastDrawerTrigger = select("#inspectorToggle");
      setInspector(open);
    });
    select("#openMethodInspectorButton").addEventListener("click", () => {
      if (window.matchMedia("(max-width: 860px)").matches) setSidebar(false);
      app.lastDrawerTrigger = select("#openMethodInspectorButton");
      setInspectorTab("method");
      setInspector(true);
      select("#methodTab").focus();
    });
    select("#inspectorClose").addEventListener("click", () => {
      closeDrawers({restoreFocus: true});
    });
    select("#sidebarToggle").addEventListener("click", () => {
      const open = select("#sidebarToggle").getAttribute("aria-expanded") !== "true";
      if (open) setInspector(false);
      if (open) app.lastDrawerTrigger = select("#sidebarToggle");
      setSidebar(open);
    });
    select("#sidebarClose").addEventListener("click", () => {
      closeDrawers({restoreFocus: true});
    });
    select("#drawerBackdrop").addEventListener("click", () => closeDrawers({restoreFocus: true}));
    select("#commandHintButton").addEventListener("click", () => {
      const hints = select("#commandHints");
      hints.hidden = !hints.hidden;
      select("#commandHintButton").setAttribute("aria-expanded", String(!hints.hidden));
    });
    for (const button of document.querySelectorAll("[data-command]")) {
      button.addEventListener("click", () => {
        select("#learnerResponse").value = button.dataset.command || "";
        select("#commandHints").hidden = true;
        select("#commandHintButton").setAttribute("aria-expanded", "false");
        select("#learnerResponse").focus();
      });
    }
    select("#learnerResponse").addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        select("#turnForm").requestSubmit();
      }
    });
    document.addEventListener("keydown", (event) => {
      trapResponsiveDrawerFocus(event);
      if (event.key !== "Escape") return;
      closeDrawers({restoreFocus: true});
      select("#commandHints").hidden = true;
      select("#commandHintButton").setAttribute("aria-expanded", "false");
    });
    const desktopLayout = window.matchMedia("(min-width: 1261px)");
    desktopLayout.addEventListener("change", (event) => {
      setSidebar(false);
      setInspector(event.matches);
    });
    window.addEventListener("resize", () => {
      syncDrawerBackdrop();
      syncResponsiveA11y();
    });
  }

  async function restoreSession() {
    const sessionId = readSessionHandle();
    if (!sessionId) return false;
    const requestAnchor = beginRequestAnchor("resume", {session: null});
    try {
      const restoredSession = await postJson("api/session", {session_id: sessionId});
      commitSessionResponse(requestAnchor, restoredSession, {
        newSession: true,
        expectedSessionId: sessionId
      });
      const initial = object(app.session.profile_summary).initial_mastery;
      app.sessionInitialMastery = {...object(initial)};
      app.pendingStart = null;
      app.pendingTurn = null;
      app.pendingCommand = null;
      clearPendingAttachment();
      app.draftingReplacement = false;
      syncReplacementDraftUi();
      applySessionProfile(app.session, {syncDraft: true});
      applySetupSnapshot(app.session);
      synchronizeControlModeFromSession();
      renderSession();
      syncNewMessageAnnouncer();
      showSetupForm(false);
      return true;
    } catch (error) {
      if (isStaleResponseError(error)) return false;
      app.stateEpoch += 1;
      app.session = null;
      syncNewMessageAnnouncer(null);
      clearSessionHandle();
      return false;
    }
  }

  async function init() {
    initTweaks();
    bindEvents();
    setAppView("learning");
    setInspector(window.matchMedia("(min-width: 1261px)").matches);
    setSidebar(false);
    setInspectorTab("state");
    renderPhase(null);
    showSetupForm(true);
    setControlMode("auto");
    syncReplacementDraftUi();
    try {
      app.bootstrap = await requestJson("api/bootstrap");
      fillSetupForm();
      populateSkillSelect();
      renderProviderStatus();
      renderNeuralGate();
      renderEvaluation();
      const restored = await restoreSession();
      if (restored) showToast("已恢复本机会话；浏览器只保存随机、无业务语义的 opaque 会话句柄。 ");
      syncControls();
    } catch (error) {
      showToast(`初始化失败：${String(error.message || error)}`);
      select("#startButton").disabled = true;
      select("#providerBadge").classList.remove("pending");
      select("#providerBadge").classList.add("offline");
      select("#providerState").textContent = "OFFLINE";
      select("#providerHeadline").textContent = "本机教学服务未连接";
      select("#providerDetail").textContent = "请用项目提供的启动命令打开页面；直接双击 HTML 无法调用本机 API。";
    }
  }

  init();
})();
