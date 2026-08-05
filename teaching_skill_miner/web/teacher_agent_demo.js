(() => {
  "use strict";

  const app = {
    bootstrap: null,
    session: null,
    busy: false,
    controlMode: "auto",
    manualSkillId: "",
    sessionInitialMastery: {},
    toastTimer: null
  };

  const signalLabels = {
    not_observed: "尚未观察",
    correct: "理解正确",
    partial: "部分理解",
    misconception: "存在误解",
    confused: "仍然困惑",
    no_response: "没有回应"
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

  function showToast(message) {
    const toast = select("#toast");
    toast.textContent = message;
    toast.hidden = false;
    window.clearTimeout(app.toastTimer);
    app.toastTimer = window.setTimeout(() => {
      toast.hidden = true;
    }, 4600);
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
      select("#providerDetail").textContent = "本轮语义诊断与教学动作来自在线模型；页面同时保留延迟和 fallback 记录。";
    } else if (ready && !fallback) {
      badge.classList.add("online");
      select("#providerState").textContent = "ONLINE READY";
      select("#providerHeadline").textContent = `${model} 在线模式已就绪`;
      select("#providerDetail").textContent = "只有开始会话或提交学生回答后，才能由成功响应确认本轮在线调用。";
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

  function syncControls() {
    const active = app.session?.status === "active";
    const commandSupport = commandControlsSupported();
    select("#startButton").disabled = app.busy;
    select("#stepButton").disabled = app.busy || !active;
    select("#autoModeButton").disabled = app.busy;
    select("#manualModeButton").disabled = app.busy || !commandSupport;
    select("#skillOverrideSelect").disabled = app.busy || !commandSupport || app.controlMode !== "manual";
    select("#applySkillButton").disabled = app.busy
      || !commandSupport
      || (app.session && !active)
      || app.controlMode !== "manual"
      || !select("#skillOverrideSelect").value;
    select("#learnerResponse").disabled = app.busy || !active;
    select("#fallbackSignalInput").disabled = app.busy
      || !active
      || select("#fallbackSignalField").hidden;
  }

  function setRange(input, value) {
    input.value = String(Math.round(finite(value, 0) * 100));
    if (input.nextElementSibling) input.nextElementSibling.value = input.value;
  }

  function readRange(id) {
    return Math.max(0, Math.min(1, finite(select(`#${id}`).value, 0) / 100));
  }

  function splitList(value) {
    return String(value || "")
      .split(/[\n,，;；]+/)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function fillSetupForm() {
    if (!app.bootstrap) return;
    const goal = object(app.bootstrap.default_goal);
    const profile = object(app.bootstrap.default_student_profile);
    const materials = object(goal.materials);
    const initial = object(profile.initial_mastery);
    const thresholds = object(goal.success_thresholds);
    select("#conceptInput").value = textValue(goal.concept, "动态规划的状态与转移");
    select("#objectiveInput").value = textValue(goal.objective, "学习者能够解释概念并完成一个迁移任务。");
    select("#learnerLevelInput").value = textValue(profile.learner_level, "beginner");
    select("#maxRoundsInput").value = String(Math.max(3, Math.min(50, finite(goal.max_rounds, 12))));
    select("#exampleInput").value = textValue(materials.example, "");
    select("#practiceInput").value = textValue(materials.practice, "");
    select("#transferInput").value = textValue(materials.transfer_task, "");
    setRange(select("#initialPrerequisite"), initial.prerequisite);
    setRange(select("#initialConceptual"), initial.conceptual);
    setRange(select("#initialProcedural"), initial.procedural);
    setRange(select("#initialTransfer"), initial.transfer);
    setRange(select("#thresholdPrerequisite"), thresholds.prerequisite ?? 0.6);
    setRange(select("#thresholdConceptual"), thresholds.conceptual ?? 0.65);
    setRange(select("#thresholdProcedural"), thresholds.procedural ?? 0.6);
    setRange(select("#thresholdTransfer"), thresholds.transfer ?? 0.55);
    select("#preferencesInput").value = array(profile.preferences).join("，");
    select("#knownMisconceptionsInput").value = array(profile.known_misconceptions)
      .map((item) => textValue(object(item).description || object(item).tag, ""))
      .filter(Boolean)
      .join("\n");
    select("#historyInput").value = array(profile.conversation_history)
      .map((item) => textValue(object(item).response, ""))
      .filter(Boolean)
      .join("\n");
    select("#remoteConsent").checked = false;
  }

  function setupPayload() {
    const misconceptions = splitList(select("#knownMisconceptionsInput").value)
      .map((description, index) => ({
        tag: `prior_${String(index + 1).padStart(2, "0")}`,
        description,
        confidence: 0.8
      }));
    const conversationHistory = splitList(select("#historyInput").value)
      .map((response) => ({
        response,
        signal: "partial",
        focus_dimension: "conceptual"
      }));
    return {
      goal: {
        concept: select("#conceptInput").value.trim(),
        objective: select("#objectiveInput").value.trim(),
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
        profile_ref: "local_ephemeral_demo_profile",
        learner_level: select("#learnerLevelInput").value,
        preferences: splitList(select("#preferencesInput").value),
        initial_mastery: {
          prerequisite: readRange("initialPrerequisite"),
          conceptual: readRange("initialConceptual"),
          procedural: readRange("initialProcedural"),
          transfer: readRange("initialTransfer")
        },
        known_misconceptions: misconceptions,
        conversation_history: conversationHistory
      },
      allowed_skill_ids: array(app.bootstrap?.skills).map((skill) => skill.skill_id),
      remote_processing_acknowledged: select("#remoteConsent").checked
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
    app.controlMode = mode === "manual" ? "manual" : "auto";
    app.manualSkillId = app.controlMode === "manual" ? textValue(skillId, "") : "";
    select("#autoModeButton").classList.toggle("active", app.controlMode === "auto");
    select("#manualModeButton").classList.toggle("active", app.controlMode === "manual");
    select("#autoModeButton").setAttribute("aria-pressed", String(app.controlMode === "auto"));
    select("#manualModeButton").setAttribute("aria-pressed", String(app.controlMode === "manual"));
    if (app.manualSkillId) select("#skillOverrideSelect").value = app.manualSkillId;
    select("#runtimeMode").textContent = app.controlMode === "auto" ? "AUTO" : "MANUAL";
    syncControls();
  }

  function currentAction(session = app.session) {
    return object(session?.next_action || session?.current_action);
  }

  function skillName(skillId) {
    const match = array(app.bootstrap?.skills).find((item) => item.skill_id === skillId);
    return match ? match.name : textValue(skillId);
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
      teacher_action: {message: event.teacher_message}
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
    for (const dimension of Object.keys(dimensionLabels)) {
      const value = Math.max(0, Math.min(1, finite(mastery[dimension], 0)));
      const bar = select(`#${dimension}Bar`);
      const valueNode = select(`#${dimension}Value`);
      const deltaNode = select(`#${dimension}Delta`);
      bar.value = value;
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
      ? "仅作下一轮决策候选，不会覆盖教师输入。"
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
    const deepseekAssessment = Object.keys(diagnosis).length > 0 && signalSource === "deepseek_v4_flash";
    const safetyFallback = signalSource === "deterministic_safety_fallback" || Boolean(latest.model_error);
    const signal = textValue(diagnosis.signal || stateSignal.label, "not_observed");
    const confidence = diagnosis.confidence ?? state.assessment_confidence ?? stateSignal.confidence;
    const evidence = textValue(diagnosis.evidence_excerpt || assessmentEvidence.excerpt || stateSignal.response_excerpt, "等待第一条学生回答。");
    const reason = textValue(diagnosis.diagnosis_reason || assessmentEvidence.reason, "尚无本轮诊断依据。");
    const review = diagnosis.needs_human_review === true || assessmentEvidence.needs_human_review === true;

    select("#assessmentSourceLabel").textContent = deepseekAssessment || (!history.length && commandControlsSupported())
      ? "DEEPSEEK ASSESSMENT"
      : (safetyFallback ? "SAFETY FALLBACK SIGNAL" : "STRUCTURED DEMO SIGNAL");
    select("#assessmentStatus").textContent = history.length
      ? (deepseekAssessment
        ? (review ? "低置信度 · 建议人工确认" : "本轮在线诊断已记录")
        : (safetyFallback ? "规则回退 · 非自由文本识别" : "人工标签 · 非自由文本识别"))
      : (commandControlsSupported() ? "等待第一条学生回答" : "等待人工降级演示信号");
    select("#assessmentLabel").textContent = signalLabels[signal] || signal;
    select("#assessmentConfidence").textContent = signal === "not_observed"
      ? "—"
      : (deepseekAssessment ? probability(confidence, 0) : (safetyFallback ? "规则回退" : "人工输入"));
    select("#assessmentEvidence").textContent = history.length
      ? (deepseekAssessment
        ? `“${evidence}”——${reason}`
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
    select("#historyCount").textContent = `${history.length} rounds`;
    const root = select("#historyList");
    if (!history.length) {
      root.replaceChildren(node("p", "", "尚无已完成回合；当前教师动作正在等待学生回答。"));
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
    const cards = normalized.reverse().map(({event, before, after}, reverseIndex) => {
      const action = historyAction(event);
      const skill = object(action.primary_skill);
      const diagnosis = object(event.deepseek_assessment);
      const signal = normalizeSignal(event.structured_signal || event.signal || diagnosis.signal);
      const details = node("details", "history-item");
      details.open = reverseIndex === 0;
      const summary = node("summary", "");
      const copy = node("div", "history-summary");
      copy.append(
        node("strong", "", `${textValue(skill.name || event.skill_name)}${action.skill_switched || event.skill_switched ? " · 已切换" : ""}`),
        node("span", "", compactText(event.learner_response || object(event.learner_feedback).response, 88))
      );
      summary.append(
        node("span", "history-round", `R${finite(event.round, history.length - reverseIndex)}`),
        copy,
        node("span", "history-signal", signalLabels[signal] || signal)
      );
      const detail = node("div", "history-detail");
      const evidence = textValue(diagnosis.evidence_excerpt, object(after.understanding_signal).response_excerpt || "降级路径未返回语义证据");
      const diagnosisReason = textValue(diagnosis.diagnosis_reason, `结构化状态记录为 ${signalLabels[signal] || signal}`);
      const trace = object(event.model_trace || action.model_trace);
      const privacy = object(event.privacy_trace || action.privacy_trace);
      detail.append(
        detailCell("STUDENT RESPONSE", event.learner_response || object(event.learner_feedback).response || "（无文字回应）"),
        detailCell("DIAGNOSIS / EVIDENCE", `“${evidence}”——${diagnosisReason}`),
        detailCell("SKILL / REASON", `${textValue(skill.name || event.skill_name)}：${textValue(action.selection_reason || event.selection_reason)}`),
        detailCell("TEACHER ACTION", object(action.teacher_action).message || event.teacher_message),
        detailCell("MODEL TRACE", trace.model ? `${trace.model} · ${Number.isFinite(Number(trace.latency_ms)) ? `${trace.latency_ms} ms` : "延迟未记录"}${trace.fallback_used ? " · FALLBACK" : ""}` : "确定性规则路径", true),
        detailCell("PRIVACY TRACE", Object.keys(privacy).length
          ? `媒体发送：${privacy.media_sent === false ? "否" : "未确认"}；自动遮盖：${privacy.redaction_applied ? "发现并处理了常见格式" : "未触发规则（不代表已完全去标识）"}`
          : "该路径未返回隐私跟踪记录。", true)
      );
      const deltaCell = node("div", "wide");
      deltaCell.append(node("span", "", "STATE DELTA"));
      const deltaList = node("div", "delta-list");
      deltaList.append(...masteryDeltaNodes(before, after));
      deltaCell.append(deltaList);
      detail.append(deltaCell);
      details.append(summary, detail);
      return details;
    });
    root.replaceChildren(...cards);
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
    const succeeded = session.status === "succeeded";
    select("#selectedSkillName").textContent = succeeded ? "教学目标达标" : "停止并转人工";
    select("#selectedSkillRole").textContent = succeeded ? "SUCCESS" : "UNABLE";
    select("#selectedSkillId").textContent = textValue(action.action_id, "terminal");
    select("#switchBadge").textContent = "终止动作";
    select("#switchBadge").classList.add("switched");
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
    const switchBadge = select("#switchBadge");
    if (action.skill_switched) {
      switchBadge.textContent = `从 ${skillName(action.previous_primary_skill_id)} 切换`;
    } else if (action.previous_primary_skill_id) {
      switchBadge.textContent = "继续当前 Skill";
    } else {
      switchBadge.textContent = "首个 Skill";
    }
    switchBadge.classList.toggle("switched", action.skill_switched === true);
    renderSupportingSkills(action);
    select("#selectionReason").textContent = textValue(action.selection_reason, "后端未返回选择理由。");
    const teacher = object(action.teacher_action);
    select("#actionType").textContent = textValue(teacher.type, "one_action");
    select("#teacherMessage").textContent = textValue(teacher.message);
    select("#expectedSignal").textContent = textValue(teacher.expected_signal, "等待学生作答后再判断。");
    select("#turnForm").hidden = false;
  }

  function renderSession() {
    const session = app.session;
    if (!session) return;
    const action = currentAction(session);
    const active = session.status === "active";
    select("#emptySession").hidden = true;
    select("#activeSession").hidden = false;
    const status = select("#sessionStatus");
    status.textContent = active
      ? `进行中 · 已完成 ${finite(session.rounds_completed, 0)} 轮`
      : statusLabels[session.status] || textValue(session.status);
    status.parentElement.classList.toggle("active", active);
    status.parentElement.classList.toggle("terminal", !active);
    select("#roundCounter").textContent = `R${finite(session.rounds_completed, 0)}`;
    if (active) renderActive(session, action); else renderTerminal(session, action);

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
    renderMisconceptions(studentState);
    renderAdaptiveStudentProfile(session);
    renderRanking(active ? action : {});
    renderGoalPlan(session);
    renderHistory(array(session.history));
    renderRuntime(session, action);
    syncControls();
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
    select("#freeTextBenchmarkStatus").textContent = `${textValue(online.model, "online model")} · n=${count}`;
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
      ? "作者构造 development set；非专家验证、非锁箱测试、非真实学生、非部署准确率。"
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
    if (providerReady() && !select("#remoteConsent").checked) {
      showToast("在线模式需要先确认脱敏文本的远程处理边界；课堂媒体永远不会发送。 ");
      select("#remoteConsent").focus();
      return;
    }
    setBusy(true);
    try {
      const startPayload = setupPayload();
      app.sessionInitialMastery = {...startPayload.student_profile.initial_mastery};
      app.session = await postJson("api/start", startPayload);
      select("#learnerResponse").value = "";
      renderSession();
      if (commandControlsSupported() && app.controlMode === "manual" && app.manualSkillId && app.session.status === "active") {
        app.session = await postJson("api/command", {
          command: "select_skill",
          skill_id: app.manualSkillId,
          session_id: app.session.session_id
        });
        renderSession();
      }
      showToast("新会话已开始：系统只生成了第一个教师动作，并正在等待学生回答。 ");
    } catch (error) {
      showToast(`无法开始：${String(error.message || error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function sendCommand(command, skillId = "") {
    if (!app.session || app.session.status !== "active") {
      throw new Error("请先开始一个仍在进行的教学会话");
    }
    app.session = await postJson("api/command", {
      command,
      skill_id: skillId || undefined,
      session_id: app.session.session_id
    });
    renderSession();
  }

  async function applySkillOverride() {
    const skillId = select("#skillOverrideSelect").value;
    if (!skillId || app.busy) return;
    if (!commandControlsSupported()) {
      showToast("确定性降级后端不支持手动 Skill；请启用 DeepSeek 在线会话。 ");
      return;
    }
    setControlMode("manual", skillId);
    if (!app.session) {
      showToast(`已选择“${skillName(skillId)}”；开始会话后用于下一次决策。`);
      return;
    }
    setBusy(true);
    try {
      await sendCommand("select_skill", skillId);
      showToast(`手动覆盖已排队：下一轮主 Skill 为“${skillName(skillId)}”。`);
    } catch (error) {
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
    if (commandText === "/auto") {
      setBusy(true);
      try {
        await sendCommand("auto");
        setControlMode("auto");
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
      select("#skillOverrideSelect").value = match.skill_id;
      setControlMode("manual", match.skill_id);
      setBusy(true);
      try {
        await sendCommand("select_skill", match.skill_id);
        select("#learnerResponse").value = "";
        showToast(`已排队“${match.name}”，下一次学生回答后执行。`);
      } finally {
        setBusy(false);
      }
      return true;
    }
    throw new Error("未知命令；可用 /+skill 名称、/auto 或 /stop");
  }

  async function submitTurn(event) {
    event.preventDefault();
    if (app.busy || !app.session || app.session.status !== "active") return;
    const learnerResponse = select("#learnerResponse").value.trim();
    if (!learnerResponse) {
      showToast("请输入学生刚才的自然语言回答。 ");
      return;
    }
    try {
      if (await maybeRunTextCommand(learnerResponse)) return;
    } catch (error) {
      showToast(`命令未执行：${String(error.message || error)}`);
      return;
    }
    setBusy(true);
    try {
      const payload = {
        learner_response: learnerResponse,
        session_id: app.session.session_id
      };
      if (!select("#fallbackSignalField").hidden) {
        payload.signal = select("#fallbackSignalInput").value;
        payload.signal_confidence = 1.0;
        if (payload.signal === "misconception") payload.misconception_tag = "manual_demo_misconception";
      }
      if (commandControlsSupported() && app.controlMode === "manual" && app.manualSkillId) {
        payload.manual_skill_id = app.manualSkillId;
      }
      app.session = await postJson("api/step", payload);
      select("#learnerResponse").value = "";
      renderSession();
      showToast(app.session.status === "active"
        ? "本轮诊断、状态变化与下一 Skill 已更新。"
        : "系统已依据停止条件结束本次会话。 ");
    } catch (error) {
      showToast(`无法提交：${String(error.message || error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function chooseAutoMode() {
    if (app.busy) return;
    if (!app.session || app.session.status !== "active") {
      setControlMode("auto");
      return;
    }
    setBusy(true);
    try {
      await sendCommand("auto");
      setControlMode("auto");
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
      });
    }
    select("#setupForm").addEventListener("submit", startSession);
    select("#turnForm").addEventListener("submit", submitTurn);
    select("#presetButton").addEventListener("click", fillSetupForm);
    select("#autoModeButton").addEventListener("click", chooseAutoMode);
    select("#manualModeButton").addEventListener("click", () => {
      if (!commandControlsSupported()) {
        showToast("确定性降级后端不支持手动 Skill；当前只演示自动规则策略。 ");
        return;
      }
      const selected = select("#skillOverrideSelect").value || app.manualSkillId;
      setControlMode("manual", selected);
      showToast("手动覆盖模式已开启；选择一个主 Skill 后点击“应用”。");
    });
    select("#skillOverrideSelect").addEventListener("change", () => {
      app.manualSkillId = select("#skillOverrideSelect").value;
      syncControls();
    });
    select("#applySkillButton").addEventListener("click", applySkillOverride);
    select("#evaluationCaseSelect").addEventListener("change", renderEvaluationCase);
    select("#learnerResponse").addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        select("#turnForm").requestSubmit();
      }
    });
  }

  async function init() {
    initTweaks();
    bindEvents();
    setControlMode("auto");
    try {
      app.bootstrap = await requestJson("api/bootstrap");
      fillSetupForm();
      populateSkillSelect();
      renderProviderStatus();
      renderNeuralGate();
      renderEvaluation();
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
