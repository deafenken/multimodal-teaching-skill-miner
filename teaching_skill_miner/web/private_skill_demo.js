(() => {
  "use strict";

  const skillState = {
    catalog: null,
    catalogRow: null,
    payload: null,
    selectedStep: 0,
    requestGeneration: 0,
    result: "recognition",
    requestedResult: window.location.hash === "#skill" ? "skill" : "recognition",
    execution: {
      concept: "",
      learnerLevel: "beginner"
    },
    runtime: {
      phase: "ready",
      procedureIndex: 0,
      verificationIndex: 0,
      attempts: 0,
      fallback: ""
    }
  };

  const eventNames = {
    question_and_wait: "提问后等待",
    code_formula_walkthrough: "板书 / 公式推演",
    visual_example: "视觉例子"
  };
  const visualNames = {
    handwritten_blackboard: "手写板书",
    mathematical_formula: "数学公式",
    programming_code: "程序代码",
    diagram_or_graph: "图表",
    instructor_talking: "教师讲解",
    other_lecture_visual: "其他课堂画面"
  };
  const learnerLevelNames = {
    beginner: "初学者",
    intermediate: "中等基础",
    advanced: "进阶学习者"
  };
  const evaluationDimensions = {
    structural_completeness: "结构完整性",
    evidence_grounding: "证据落地性",
    executability: "可执行性",
    method_fidelity: "方法忠实度",
    pedagogical_quality: "教学设计质量",
    generalizability: "可迁移性",
    traceability: "可追溯性"
  };
  const evaluationGates = {
    schema_valid: "Schema 合法",
    grounded: "证据有效",
    executable: "过程可执行",
    testable: "结果可验证",
    multimodal_consistent: "多模态证据一致",
    method_distilled_from_video: "方法来自视频证据"
  };

  const select = (selector) => document.querySelector(selector);

  function node(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined) value.textContent = text;
    return value;
  }

  function route(relative) {
    return new URL(relative, window.location.href).href;
  }

  async function requestJson(relative) {
    const response = await fetch(route(relative), {
      cache: "no-store",
      credentials: "same-origin"
    });
    if (!response.ok) throw new Error(`request failed (${response.status})`);
    return response.json();
  }

  function formatTime(seconds) {
    const value = Math.max(0, Math.round(Number(seconds) || 0));
    const hours = Math.floor(value / 3600);
    const minutes = Math.floor((value % 3600) / 60);
    const remainder = value % 60;
    const clock = `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
    return hours ? `${String(hours).padStart(2, "0")}:${clock}` : clock;
  }

  function formatCount(value) {
    return Number(value || 0).toLocaleString("en-US");
  }

  function formatScore(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(1) : "—";
  }

  function fillTemplate(value, parameters) {
    let rendered = String(value || "");
    for (const [key, replacement] of Object.entries(parameters)) {
      rendered = rendered.replaceAll(`{${key}}`, String(replacement));
    }
    return rendered;
  }

  function updateResultActions() {
    select("#recognitionResultAction").textContent = skillState.result === "recognition"
      ? "正在查看此成果"
      : "切换查看 →";
    const skillButton = select("#resultSkillButton");
    const skillAction = select("#skillResultAction");
    if (skillButton.dataset.loadState === "loading") {
      skillAction.textContent = "正在加载真实 Skill…";
    } else if (skillButton.dataset.loadState === "error") {
      skillAction.textContent = "Skill 载入失败";
    } else {
      skillAction.textContent = skillState.result === "skill"
        ? "正在查看此成果"
        : "点击进入 Skill Demo →";
    }
  }

  function setResult(result, updateUrl = true) {
    if (result === "skill" && !skillState.catalog?.skill_distillation?.available) return;
    skillState.result = result;
    const recognitionActive = result === "recognition";
    select("#recognitionOutcome").hidden = !recognitionActive;
    select("#skillOutcome").hidden = recognitionActive;
    select("#privacyToggle").hidden = !recognitionActive;
    for (const button of document.querySelectorAll(".result-switch-button")) {
      const active = button.dataset.result === result;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    }
    document.documentElement.dataset.result = result;
    updateResultActions();
    if (updateUrl) window.history.replaceState(null, "", result === "skill" ? "#skill" : "#recognition");
    if (!recognitionActive) select("#classroomVideo").pause();
  }

  function renderAggregate() {
    const track = skillState.catalog.skill_distillation;
    const aggregate = track.aggregate;
    const sceneCount = skillState.catalog.lessons.reduce(
      (total, lesson) => total + Number(lesson.scene_count || 0),
      0
    );
    select("#recognitionSceneCount").textContent = formatCount(sceneCount);
    select("#skillResultCount").textContent = formatCount(track.skills.length);
    select("#aggregateLectureCount").textContent = formatCount(aggregate.complete_lecture_count);
    select("#aggregateCandidateEvents").textContent = formatCount(aggregate.candidate_multimodal_event_count);
    select("#aggregateProcedureSteps").textContent = formatCount(aggregate.procedure_step_count);
    select("#aggregateObservedSteps").textContent = formatCount(aggregate.observed_method_step_count);
    select("#aggregateRecommendedSteps").textContent = formatCount(aggregate.recommended_enrichment_step_count);
    select("#aggregateSelectedEvents").textContent = formatCount(aggregate.selected_multimodal_event_count);
    select("#directMmeCount").textContent = formatCount(aggregate.procedure_direct_mme_reference_count);
  }

  function renderSkillOptions() {
    const track = skillState.catalog.skill_distillation;
    const options = track.skills.map((row) => {
      const option = document.createElement("option");
      option.value = row.id;
      option.textContent = `${row.id} · ${row.title}`;
      return option;
    });
    select("#skillSelect").replaceChildren(...options);
    select("#skillSelect").value = track.default_skill;
  }

  function resetRuntime() {
    skillState.runtime = {
      phase: "ready",
      procedureIndex: 0,
      verificationIndex: 0,
      attempts: 0,
      fallback: ""
    };
    renderRuntime();
  }

  async function loadSkill(videoId) {
    const track = skillState.catalog.skill_distillation;
    const catalogRow = track.skills.find((row) => row.id === videoId);
    if (!catalogRow) return;
    const generation = ++skillState.requestGeneration;
    select("#skillTitle").textContent = "正在读取真实 Skill…";
    try {
      const payload = await requestJson(catalogRow.detail);
      if (generation !== skillState.requestGeneration || payload.video_id !== videoId) return;
      skillState.catalogRow = catalogRow;
      skillState.payload = payload;
      skillState.selectedStep = 0;
      select("#skillSelect").value = videoId;
      renderSkill();
      resetRuntime();
      return true;
    } catch (error) {
      select("#skillTitle").textContent = "Skill 读取失败";
      select("#skillName").textContent = String(error.message || error);
      return false;
    }
  }

  function renderSkillMeta() {
    const row = skillState.catalogRow;
    const payload = skillState.payload;
    const items = [
      `${formatTime(row.duration_seconds)} 完整视频`,
      `${formatCount(row.candidate_event_count)} 候选事件`,
      `${row.observed_phase_count} observed + ${row.recommended_phase_count} recommended`,
      `${payload.source.caption_coverage_fraction.toFixed(3)} 字幕覆盖`
    ].map((text) => node("span", "", text));
    select("#skillMeta").replaceChildren(...items);
    select("#skillCourse").textContent = `${payload.source.course_id} · FULL MULTIMODAL SKILL`;
    select("#skillTitle").textContent = payload.source.title;
    select("#skillName").textContent = payload.skill.name;
    select("#skillObjective").textContent = payload.skill.learning_objective.statement;
    select("#methodFidelity").textContent = Number(payload.evaluation.method_fidelity).toFixed(1);
  }

  function renderProcedure() {
    const steps = skillState.payload.skill.procedure;
    const buttons = steps.map((step, index) => {
      const observed = step.origin === "observed_method";
      const button = node(
        "button",
        `procedure-step${observed ? " observed" : " recommended"}${index === skillState.selectedStep ? " active" : ""}`
      );
      button.type = "button";
      button.setAttribute("aria-pressed", String(index === skillState.selectedStep));
      const number = node("span", "step-number", String(step.step).padStart(2, "0"));
      const name = node("span", "step-name");
      name.append(
        node("strong", "", step.teaching_phase_name),
        node(
          "small",
          "",
          observed && step.observed_span
            ? `首末证据 ${formatTime(step.observed_span.start)}–${formatTime(step.observed_span.end)}`
            : "原视频无直接证据"
        )
      );
      const origin = node("span", "origin-pill", observed ? "视频中观察到" : "建议补全");
      button.append(number, name, origin);
      button.addEventListener("click", () => {
        skillState.selectedStep = index;
        renderProcedure();
        renderStepDetail();
      });
      return button;
    });
    select("#skillProcedureList").replaceChildren(...buttons);
  }

  function renderStepDetail() {
    const payload = skillState.payload;
    const step = payload.skill.procedure[skillState.selectedStep];
    const observed = step.origin === "observed_method";
    const origin = select("#stepOrigin");
    origin.textContent = observed ? "OBSERVED METHOD · 视频中观察到" : "RECOMMENDED ENRICHMENT · 建议补全";
    origin.classList.toggle("recommended", !observed);
    select("#stepPhase").textContent = `${String(step.step).padStart(2, "0")} · ${step.teaching_phase_name}`;
    select("#stepAction").textContent = step.teacher_action.toUpperCase();
    select("#stepInstruction").textContent = step.instruction;
    select("#stepExpected").textContent = step.expected_signal;
    select("#stepFallback").textContent = step.fallback;
    const evidenceById = new Map(
      payload.text_evidence.map((record) => [record.evidence_id, record])
    );
    const evidence = step.evidence_ids
      .map((evidenceId) => evidenceById.get(evidenceId))
      .filter(Boolean);
    const records = evidence.map((record) => {
      const item = node("div", "evidence-record");
      item.append(
        node("code", "", `${formatTime(record.start)}–${formatTime(record.end)}\n${record.evidence_id}`),
        node("q", "", record.quote)
      );
      return item;
    });
    if (!records.length) {
      records.push(
        node(
          "p",
          "evidence-empty",
          "视频中未观察到该环节。本步骤来自九环节 canonical scaffold，不作为原教师行为声明。"
        )
      );
    }
    select("#stepEvidenceList").replaceChildren(...records);
    select("#stepEvidenceBoundary").textContent = observed
      ? `${evidence.length} 条直接字幕证据 · mme_* 直接引用 0`
      : "建议补全不计入视频方法证据";
  }

  function eventDescription(event) {
    const parts = [];
    if (event.wait_seconds !== null) parts.push(`提问后等待 ${event.wait_seconds.toFixed(3)} 秒`);
    if (event.visual_semantic_labels.length) {
      const labels = event.visual_semantic_labels.map(
        (item) => visualNames[item.label] || item.label.replaceAll("_", " ")
      );
      parts.push(`视觉语义：${labels.join(" / ")}`);
    }
    if (!parts.length) parts.push("字幕与视觉时间窗共同支持该策略候选");
    return parts.join("；");
  }

  function renderEvents() {
    const cards = skillState.payload.multimodal_events.map((event) => {
      const card = node("article", "event-card");
      const head = node("div", "event-card-head");
      head.append(
        node("strong", "", eventNames[event.type] || event.type.replaceAll("_", " ")),
        node("code", "", `${formatTime(event.start)}–${formatTime(event.end)}`)
      );
      const modalities = node("div", "event-modalities");
      modalities.append(...event.modalities.map((name) => node("span", "", name.toUpperCase())));
      card.append(head, modalities, node("p", "", eventDescription(event)));
      if (event.speech_excerpt) card.append(node("small", "", `“${event.speech_excerpt}”`));
      card.append(
        node(
          "small",
          "",
          `${event.event_id} · 支持 ${event.supports.join(" / ")} · heuristic ${event.confidence.toFixed(2)}`
        )
      );
      return card;
    });
    select("#skillEventTimeline").replaceChildren(...cards);
  }

  function renderStrategies() {
    const header = node("div", "strategy-row strategy-head");
    header.append(node("strong", "", "教学策略"), node("span", "", "TEXT"), node("span", "", "MME"));
    const rows = skillState.payload.skill.strategies.map((strategy) => {
      const item = node("div", "strategy-row");
      item.append(
        node("strong", "", strategy.name),
        node("span", "", formatCount(strategy.text_evidence_count)),
        node("span", "", formatCount(strategy.multimodal_evidence_count))
      );
      return item;
    });
    select("#skillStrategyList").replaceChildren(header, ...rows);
  }

  function readExecutionParameters() {
    const defaults = skillState.payload.skill.parameters;
    const conceptInput = select("#processConcept");
    const levelInput = select("#learnerLevel");
    const concept = conceptInput.value.trim().replace(/\s+/g, " ").slice(0, 160)
      || defaults.concept.default;
    const requestedLevel = levelInput.value || defaults.learner_level.default;
    const learnerLevel = Object.hasOwn(learnerLevelNames, requestedLevel)
      ? requestedLevel
      : "beginner";
    conceptInput.value = concept;
    levelInput.value = learnerLevel;
    skillState.execution = { concept, learnerLevel };
    return {
      concept,
      learner_level: learnerLevel
    };
  }

  function generateTeachingProcess() {
    if (!skillState.payload) return;
    const parameters = readExecutionParameters();
    const skill = skillState.payload.skill;
    const procedure = skill.procedure.map((step) => {
      const observed = step.origin === "observed_method";
      const card = node("article", `generated-step${observed ? "" : " recommended"}`);
      const header = node("header");
      header.append(
        node("span", "", `STEP ${String(step.step).padStart(2, "0")}`),
        node(
          "small",
          "",
          `${step.teacher_action.toUpperCase()} · ${observed ? "OBSERVED" : "RECOMMENDED"}`
        )
      );
      const details = node("dl");
      const signal = node("div");
      signal.append(
        node("dt", "", "观察信号"),
        node("dd", "", fillTemplate(step.expected_signal, parameters))
      );
      const fallback = node("div");
      fallback.append(
        node("dt", "", "未达标"),
        node("dd", "", fillTemplate(step.fallback, parameters))
      );
      details.append(signal, fallback);
      card.append(
        header,
        node("h4", "", step.teaching_phase_name),
        node("p", "", fillTemplate(step.instruction, parameters)),
        details
      );
      return card;
    });
    const checks = skill.verification.map((check, index) => {
      const item = node("div", "generated-check");
      item.append(
        node("strong", "", `${index + 1}. ${check.type}`),
        node(
          "span",
          "",
          `${fillTemplate(check.prompt, parameters)}｜通过条件：${fillTemplate(check.pass_condition, parameters)}`
        )
      );
      return item;
    });
    select("#generatedSkillId").textContent = `${skill.skill_id} · ${learnerLevelNames[parameters.learner_level]}`;
    select("#generatedProcessTitle").textContent = `教学过程：${parameters.concept}`;
    select("#generatedGoal").textContent = `本轮目标：${fillTemplate(skill.goal, parameters)}`;
    select("#generatedProcessStatus").textContent = `GENERATED · ${procedure.length} STEPS`;
    select("#generatedProcedure").replaceChildren(...procedure);
    select("#generatedVerification").replaceChildren(...checks);
    select("#generateProcess").textContent = "重新生成教学过程";
  }

  function prepareTeachingProcess() {
    const parameters = skillState.payload.skill.parameters;
    select("#processConcept").value = parameters.concept.default;
    select("#learnerLevel").value = Object.hasOwn(
      learnerLevelNames,
      parameters.learner_level.default
    )
      ? parameters.learner_level.default
      : "beginner";
    generateTeachingProcess();
  }

  function renderEvaluation() {
    const evaluation = skillState.payload.evaluation;
    const dimensionRows = Object.entries(evaluationDimensions).map(([key, label]) => {
      const score = Number(evaluation.dimensions[key]);
      const weight = Number(evaluation.weights[key]);
      const row = node("div", "evaluation-dimension");
      const meter = document.createElement("meter");
      meter.min = 0;
      meter.max = 100;
      meter.low = 60;
      meter.high = 80;
      meter.optimum = 100;
      meter.value = score;
      meter.setAttribute("aria-label", `${label} ${formatScore(score)} 分`);
      const output = node("output", "", formatScore(score));
      row.append(
        node("strong", "", label),
        node("span", "", `权重 ${(weight * 100).toFixed(0)}%`),
        meter,
        output
      );
      return row;
    });
    const gateRows = Object.entries(evaluationGates).map(([key, label]) => {
      const passed = evaluation.gates[key] === true;
      const row = node("div", `evaluation-gate${passed ? " pass" : " fail"}`);
      row.append(
        node("strong", "", label),
        node("span", "", passed ? "PASS" : "CHECK")
      );
      return row;
    });
    select("#evaluationOverall").textContent = formatScore(evaluation.internal_overall_score);
    select("#evaluationDecision").textContent = evaluation.internal_passed
      ? `INTERNAL PASS · 阈值 ${formatScore(evaluation.internal_threshold)}`
      : `INTERNAL CHECK · 阈值 ${formatScore(evaluation.internal_threshold)}`;
    select("#evaluationGrade").textContent = `内部等级 ${evaluation.internal_grade}`;
    select("#evaluationDimensions").replaceChildren(...dimensionRows);
    select("#evaluationGates").replaceChildren(...gateRows);
    select("#evidenceConsistency").textContent = formatScore(
      evaluation.internal_evidence_consistency_score
    );
    select("#evaluationScope").textContent = "评估范围为结构质量与内部证据一致性";
  }

  function renderProvenance() {
    const labels = {
      dataset_manifest_sha256: "DATASET MANIFEST",
      ablation_report_sha256: "ABLATION REPORT",
      skill_fingerprint: "SKILL FINGERPRINT",
      skill_file_sha256: "SKILL FILE",
      evaluation_file_sha256: "EVALUATION FILE"
    };
    const cells = Object.entries(labels).map(([key, label]) => {
      const cell = node("div");
      cell.append(node("span", "", label), node("code", "", skillState.payload.provenance[key]));
      return cell;
    });
    select("#skillProvenance").replaceChildren(...cells);
  }

  function renderSkill() {
    renderSkillMeta();
    renderProcedure();
    renderStepDetail();
    renderEvents();
    renderStrategies();
    prepareTeachingProcess();
    renderEvaluation();
    renderProvenance();
  }

  function runtimeTurn() {
    const runtime = skillState.runtime;
    const payload = skillState.payload;
    if (!payload || runtime.phase === "ready") return null;
    if (runtime.phase === "procedure") return payload.skill.procedure[runtime.procedureIndex];
    if (runtime.phase === "verification") return payload.skill.verification[runtime.verificationIndex];
    return null;
  }

  function renderRuntimeFlow() {
    const runtime = skillState.runtime;
    const definitions = [
      ["ready", "READY"],
      ["procedure", "PROCEDURE · 9 STEPS"],
      ["verification", "VERIFY · 2 CHECKS"],
      ["completed", "COMPLETE"]
    ];
    const values = [];
    definitions.forEach(([phase, label], index) => {
      values.push(node("span", `runtime-node${runtime.phase === phase ? " active" : ""}`, label));
      if (index < definitions.length - 1) values.push(node("span", "runtime-arrow", "→"));
    });
    select("#runtimeFlow").replaceChildren(...values);
  }

  function renderRuntime() {
    if (!select("#runtimeFlow")) return;
    const runtime = skillState.runtime;
    const turn = runtimeTurn();
    renderRuntimeFlow();
    select("#runtimePhase").textContent = runtime.phase.toUpperCase();
    select("#runtimeFeedback").hidden = !runtime.fallback;
    select("#runtimeFeedback").textContent = runtime.fallback;
    select("#runtimeStart").textContent = runtime.phase === "ready" ? "开始执行" : "重新开始";
    const active = runtime.phase === "procedure" || runtime.phase === "verification";
    select("#runtimePass").disabled = !active;
    select("#runtimeFail").disabled = !active;
    if (runtime.phase === "ready") {
      select("#runtimePosition").textContent = "尚未开始";
      select("#runtimeMessage").textContent = `点击“开始执行”，载入“${skillState.execution.concept || "当前主题"}”的第一步。`;
    } else if (runtime.phase === "procedure") {
      select("#runtimePosition").textContent = `步骤 ${runtime.procedureIndex + 1} / ${skillState.payload.skill.procedure.length}`;
      select("#runtimeMessage").textContent = fillTemplate(turn.instruction, {
        concept: skillState.execution.concept,
        learner_level: skillState.execution.learnerLevel
      });
    } else if (runtime.phase === "verification") {
      select("#runtimePosition").textContent = `验证 ${runtime.verificationIndex + 1} / ${skillState.payload.skill.verification.length}`;
      select("#runtimeMessage").textContent = `${fillTemplate(turn.prompt, {
        concept: skillState.execution.concept,
        learner_level: skillState.execution.learnerLevel
      })}｜通过条件：${fillTemplate(turn.pass_condition, {
        concept: skillState.execution.concept,
        learner_level: skillState.execution.learnerLevel
      })}`;
    } else {
      select("#runtimePosition").textContent = "路径执行完成";
      select("#runtimeMessage").textContent = "九个步骤与两项验证均已按状态机走完；这只表示流程完成，不表示学习效果已经建立。";
    }
  }

  function startRuntime() {
    if (!skillState.payload) return;
    generateTeachingProcess();
    skillState.runtime = {
      phase: "procedure",
      procedureIndex: 0,
      verificationIndex: 0,
      attempts: 0,
      fallback: ""
    };
    renderRuntime();
  }

  function passRuntime() {
    const runtime = skillState.runtime;
    if (runtime.phase === "procedure") {
      runtime.procedureIndex += 1;
      if (runtime.procedureIndex >= skillState.payload.skill.procedure.length) {
        runtime.phase = "verification";
        runtime.verificationIndex = 0;
      }
    } else if (runtime.phase === "verification") {
      runtime.verificationIndex += 1;
      if (runtime.verificationIndex >= skillState.payload.skill.verification.length) {
        runtime.phase = "completed";
      }
    }
    runtime.attempts = 0;
    runtime.fallback = "";
    renderRuntime();
  }

  function failRuntime() {
    const runtime = skillState.runtime;
    const turn = runtimeTurn();
    if (!turn) return;
    runtime.attempts += 1;
    const base = runtime.phase === "procedure"
      ? fillTemplate(turn.fallback, {
        concept: skillState.execution.concept,
        learner_level: skillState.execution.learnerLevel
      })
      : "先指出回答中与通过条件不一致的一处，再提供分层提示并要求重新作答。";
    runtime.fallback = runtime.attempts >= 2
      ? `${base} 已连续两次未达标：降低任务复杂度，并回查必要前置知识。`
      : base;
    renderRuntime();
  }

  async function initializeSkillDemo() {
    try {
      skillState.catalog = await requestJson("api/catalog");
      const track = skillState.catalog.skill_distillation;
      if (!track || !track.available) {
        const button = select("#resultSkillButton");
        button.dataset.loadState = "error";
        button.setAttribute("aria-busy", "false");
        button.title = "当前会话未加载 Skill 产物";
        updateResultActions();
        return;
      }
      renderAggregate();
      renderSkillOptions();
      const loaded = await loadSkill(track.default_skill);
      if (!loaded) throw new Error("默认 Skill 载入失败");
      const button = select("#resultSkillButton");
      button.disabled = false;
      button.dataset.loadState = "ready";
      button.setAttribute("aria-busy", "false");
      if (skillState.requestedResult === "skill") setResult("skill");
      else updateResultActions();
    } catch (error) {
      const button = select("#resultSkillButton");
      button.disabled = true;
      button.dataset.loadState = "error";
      button.setAttribute("aria-busy", "false");
      button.title = String(error.message || error);
      updateResultActions();
    }
  }

  select("#resultRecognitionButton").addEventListener("click", () => setResult("recognition"));
  select("#resultSkillButton").addEventListener("click", () => setResult("skill"));
  select("#skillSelect").addEventListener("change", (event) => void loadSkill(event.target.value));
  select("#generateProcess").addEventListener("click", () => {
    generateTeachingProcess();
    resetRuntime();
  });
  select("#processConcept").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      generateTeachingProcess();
      resetRuntime();
    }
  });
  select("#runtimeStart").addEventListener("click", startRuntime);
  select("#runtimePass").addEventListener("click", passRuntime);
  select("#runtimeFail").addEventListener("click", failRuntime);

  setResult("recognition", false);
  void initializeSkillDemo();
})();
