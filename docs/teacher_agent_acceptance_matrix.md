# 题目二交付与验收矩阵

边界提醒：学生“确认识别文字”只确认转写，确认不等于答案正确。

## 1. 验收原则

本矩阵只记录当前仓库能由代码、Schema、冻结数据、公开 receipt 或自动测试支持的事实。状态分为四类：

- **已实现并测试**：工程能力已存在，且有自动测试或可复现命令；
- **已有开发证据**：已有量化结果，但数据或协议仍属于开发阶段；
- **部分实现 / 受限**：核心接口存在，但使用场景或覆盖范围有限；
- **外部验证待完成**：必须依赖专家标注、锁箱测试、真实学习者或部署记录，不能靠新增合成数据“补齐”。

统一边界：接入 DeepSeek 不等于诊断准确率已建立；页面显示分数不等于真实学习效果；neural-v1 被运行时引用不等于其证据门禁通过；本机页面不等于所有处理离线。

当前题目二运行契约为 V14（`teaching_agent_assess_route_act_v14_state_first_route_adjudication`）。文档中的 `action_only_repair`、`continuity_recall`、`cancel_turn` 和 `transport_cancellation_supported` 均按该版本语义解释。

## 2. 核心教学闭环

| 编号 | 验收问题 | 状态 | 当前交付与证据 | 边界或剩余工作 |
|---|---|---|---|---|
| A01 | 能否输入一个新的教学目标 | 已实现并测试 | Dashboard `api/start`；`goal.concept/objective/materials/success_thresholds/max_rounds`；`tests/test_teacher_agent_dashboard.py` | 目标由教师输入，系统不自动证明目标本身合理 |
| A02 | 能否像 Goal 模式一样拆分和跟踪目标 | 已实现并测试 | `teacher_agent_context.py::build_goal_plan` 生成前置、概念、过程、迁移 4 个可观察子目标及进度 | Goal 进度是内部控制量，不是学习成绩 |
| A03 | 能否输入学生画像和初始掌握状态 | 已实现并测试 | v2 会话 Schema、网页表单、`teacher_agent.py`；自由文本既往内容按行写入 `background_history`，明确标为未标注背景 | `background_history` 不携带 signal/focus，不能直接改变掌握、理解信号或误解；应使用最小化、非身份信息 |
| A04 | 能否根据回答动态更新学生画像 | 已实现并测试 | 每个有效 DeepSeek 回合向 `student_profile.adaptive_observations` 追加 `candidate_unconfirmed`：回答质量、参与度、误解标签、下一重点、脱敏证据和置信度；`adaptive_summary` 汇总最新候选，最多保留 12 条；Schema 与 `tests/test_teacher_agent_live.py` 覆盖 | 低置信度/模型请求复核必须标记人工复核；候选不覆盖教师字段，fallback 不生成；尚无跨 session 长期画像 |
| A05 | 是否显式维护知识掌握情况 | 已实现并测试 | `knowledge_mastery` 的 prerequisite / conceptual / procedural / transfer 四维状态 | 归一化控制值不是校准后的真实掌握概率 |
| A06 | 是否显式维护误解或错误模式 | 已实现并测试 | `misconceptions` 支持发现、重复观察和 active / resolved 生命周期 | 模型诊断需看置信度和证据，不能写成确定心理事实 |
| A07 | 是否显式维护当前轮理解信号 | 已实现并测试 | 状态含 confidence/evidence，并区分原始 DeepSeek、契约修正、当前问题精确命中、教师 `knowledge_spec` 精确命中和规则 fallback 五类来源 | 契约修正/精确命中不能称为未经修正的模型原判；内置 `knowledge_spec` 是作者演示 fixture；降级不是自由文本诊断 |
| A08 | 是否显式维护下一步教学关注点 | 已实现并测试 | `student_state.next_focus` 和 Goal `active_step` | 这是策略输出，不是专家最优性证明 |
| A09 | 是否管理历史对话 | 已实现并测试 | 单一六层 `teaching_context`；每轮活动知识点 → 同关注维度 → 最近轮次；默认最多 10 个相关回合、14000 字符；较早历史保留确定性统计、focus checkpoints、最多 6 条 evidence-linked `teaching_checkpoints`，以及从完整 rollout replay 的 `teaching_memory`。后者只含显式偏好、未解决问题、教师承诺和指代对象，并绑定原话、history version、compaction generation 与固定上下文指纹；`background_history` 进入 `unlabeled_notes` 且 `unlabeled_notes_may_update_state=false` | `teaching_checkpoints_are_selective_extracts=true`、`omitted_turn_semantics_are_exhaustive=false`；不是模型叙事总结、无限记忆或不同 session 自动合并；未标注背景不是学情证据 |
| A10 | 是否每轮只生成一个下一教学动作 | 已实现并测试 | `one_action_per_turn`、`wait_for_student_before_next_action`；默认 `safe_generative` 允许通过最终 Skill/action type、单动作、提问性、问题契约和防泄露门禁的模型话语显示，任一不一致则由确定性 materializer 重建；`action_provenance.executor_origin` 可审计 | “实时”是请求—响应闭环，不是音视频流式感知；`deepseek_safe_generative` 不是无约束直出 |
| A11 | 是否从 Skill Library 自动选主 Skill | 已实现并测试 | DeepSeek 提议 + 主 Skill 白名单校验；记录候选、选择理由和来源 | 开放场景路由最优性待专家锁箱验证 |
| A12 | 是否能在一轮组合多个 Skill | 已实现并测试 | 1 个主 Skill + 最多 2 个 support Skill；主 Skill 的 `supporting_skill_ids` 是硬组合 allowlist，未声明 support 不得执行；声明项仍须通过 support 角色、局部前置/证据和 `max_repeat` 门禁；`composition_plan.one_action_contract=true` | 组合 Skill 不允许生成多个预写回合；allowlist 只证明组合合法，不证明组合教学上最优 |
| A13 | 是否根据自然语言反馈动态切换 Skill | 已实现并测试 | `teacher_agent_live.py` 每轮重新诊断和路由；测试覆盖纠错、手动覆盖、降级和终止 | 当前单轮开发 benchmark 不是完整多轮真实学生准确率 |
| A14 | 是否显示当前 Skill、切换和理由 | 已实现并测试 | Dashboard 展示主/支持 Skill、previous Skill、switch、reason、decision origin | 理由是短审计摘要，不展示 chain-of-thought |
| A15 | 是否支持自动与手动切换 | 已实现并测试 | DeepSeek 在线模式支持 `/+skill 名称`、`/auto`、`/stop`；手动锁从下一条回答起持续，未知/非主 Skill 拒绝，不适用信号、无依据纠错、重复上限或 fallback 会安全释放 | 人工覆盖单独标记，不计作 Agent 自动命中；确定性展示模式不支持手动 stop |
| A16 | 是否能成功停止或安全转人工 | 已实现并测试 | 成功阈值、无活跃误解、最少轮数、连续无进展、最大轮数、受约束停止建议和人工停止 | 内部 `succeeded` 不等于真实学习效果已建立 |
| A17 | 重复提交或切换画像是否会污染会话 | 已实现并测试 | step/command 均绑定 `session_id + expected_round + expected_question_id + expected_context_version + profile_revision`，并各带独立幂等键；start 使用另一独立幂等键。画像替换还绑定旧会话的 round / question / context / profile 四项版本，并在旧 Session 锁内先核验、后构造、再原子提交；前端应用异步结果前核对请求 epoch、session/profile 与不倒退的 context version。Playwright runner 主动制造一次 stale replacement 400，再验证 guard 同步、新幂等键和单次重试。后端保存最多 16 个隔离 Session；合法 fallback 可提交，无合法候选时保留旧 Session | 只证明受控竞态恢复，不等于任意并发或分布式事务验证。默认重启会丢失内存 Session；显式 session store 的冷恢复另见 A19 |
| A18 | 是否支持答案图片作为当前回合证据 | 已实现并测试 | 网页和 `api/attachment` 接受答案图片；本机可使用 Apple Vision 与原图/灰度增强/二值图的多条 Tesseract 路由。原图只在内存/临时目录处理，远程只接收有界并做常见直接标识符模式替换的 OCR 文字。附件 ID 绑定 session/question/round/profile/context，幂等且单次消费。高置信、无冲突、多引擎或足够多预处理路线一致的公式可建立 `formula_transcription_established=true`，但 `formula_accuracy_established=false`；确定性 canonical claim 必须带非空 `knowledge_components` 且与当前动作知识点相交，多行 OCR 按完整答案匹配，跨知识点完整命中降为 `partial / related_but_not_answer`；只有可靠转写与可判定 `question_contract` 或当前知识点内教师 `knowledge_spec` 规范陈述精确匹配才可确定性判对。低置信、无文字、路线冲突、键入/OCR 冲突或未佐证公式仍强制确认；“答案见图/照片显然正确”不能绕过 | 这是 OCR 辅助文本证据，不是 DeepSeek 原生视觉理解。相关开放题词或 rubric 片段不自动构成完整正确；手写、复杂版面、公式 OCR 和判分部署准确率未建立 |
| A19 | 服务重启后能否 cold resume | 已实现并测试（显式 opt-in） | `--session-store` 启用本机 append-only JSONL；连续 seq + `previous_hash` + SHA-256 链，截断尾可修复，完整篡改 fail-closed；step 使用 `turn_started / turn_committed / turn_aborted`，悬空 started turn 重启后补记 aborted，旧键拒绝伪提交。恢复校验无密钥 `runtime_policy_contract`，并允许内容等价、有序的主 Skill 子集，同时强制保留全部 support | 默认仍为内存会话。store 含目标、画像、对话、状态、幂等缓存和尚存附件 OCR 证据；原图不存。哈希链不是加密/认证/跨设备同步，文件必须按私有学生数据保护 |

补充运行语义：生产 dashboard 通过 CLI 显式启用 `state_first_route_adjudication_enabled=true` 与 `action_only_repair_enabled=true`；库级 `LiveAgentOptions` 默认仍关闭，以保留兼容调用。状态优先裁决器按掌握维度、误解、参与度、无进展计数和 Skill 契约确定优先层，模型候选只在同层 tie-break；手动路由、视觉确认和已有安全 retarget 不重复进入该裁决器。`semantic_summary.continuity_recall` 由服务端从显式连续性 cue 确定性生成，只允许 evidence-linked 接续，找不到证据必须请学生重述。活动回合的“停止生成”是可恢复 `cancel_turn`，会使当前 turn 失效并由 commit fence 丢弃迟到响应；`transport_cancellation_supported=false`，远程 HTTP 可能仍完成/计费。显式 `/stop` 才把整个 Session 置为 terminal。

图片“确认识别文字”或输入修订只越过附件消费前的学生确认门，表示学生确认/修订了 OCR 转写，不代表系统已经确认答案正确，也不会建立 `formula_accuracy_established`；只有后续文字精确命中当前问题契约或教师知识规格，才可能记录相应 exact-match 来源。

## 3. DeepSeek V4 Flash 与控制层

| 编号 | 验收问题 | 状态 | 当前交付与证据 | 边界或剩余工作 |
|---|---|---|---|---|
| B01 | 语义与生成骨干是否使用指定模型 | 已实现并测试 | `deepseek_client.py` 默认 `deepseek-v4-flash`；在线会话记录 provider/model/trace | 确定性控制器仍负责安全边界；这不是缺少 Agent，而是受约束 Agent 设计 |
| B02 | 是否在同一回合完成诊断、路由和当前行动 | 已实现并测试 | 生产 live 回合先运行 bounded Agent Loop（多次规划与本地工具结果回传），达到 `route_ready` 后再调用 `teaching_agent_assess_route_act_v14_state_first_route_adjudication` 最终动作规划器；最终 JSON 同时给出相对当前问题的 diagnosis、primary/support Skill、选择理由和一个动作候选。默认 `safe_generative` 只保留通过 Skill/action type、单动作、提问性、问题契约和防泄露门禁且路由/support 未被改写的安全话语，否则先由确定性 materializer 接管；仅当候选动作不满足最终契约、选项启用且 bounded eligibility 通过时，最多追加 1 次 fixed-route action-only repair。repair 只能修 `teacher_action`，不能改变 diagnosis、primary/support Skill、termination 或 route；失败仍保留 materializer 动作。传输层重试不形成额外教学动作 | “同一回合”指一个学生输入只提交一个下一教学动作，不代表每回合只有一个远程请求：生产拓扑是 Loop 规划请求 + 最终动作规划请求，必要时再加一次 action-only repair；各类请求、重试和工具调用分开计数。benchmark runner 当前为兼容性对照而关闭 `agent_loop_enabled`，所以其单轮请求统计不能替代生产 Loop 的在线质量评测；只展示短审计字段，不公开思维链 |
| B03 | 模型能否新增或越权选择 Skill | 已实现并测试 | 未知 Skill、support 充当主 Skill、信号不适用、无证据纠错、关键阶段/材料前置条件不满足、超过 `max_repeat`、未在主 Skill allowlist 中声明或不满足自身门禁的 support 均拒绝；fallback 使用同一安全门，无安全 Skill 时停止转人工 | Skill Library 变更必须单独审查来源和契约；未列入本地形式化子集的自然语言合同仍不等于自动验证 |
| B04 | 能否避免直接泄露最终答案 | 部分实现 / 受限 | 提示契约和明显最终答案模式拦截 | 正则不能证明所有学科、所有表达都不会泄露，仍需行为评测与人工抽查 |
| B05 | 模型停止建议是否受控制 | 已实现并测试 | 只有 human-review 建议且至少两轮无进展时才采纳；硬终止由控制器决定 | 模型不能自行改写终止状态 |
| B06 | API 异常是否可恢复 | 已实现并测试 | 超时/重试、响应大小限制、JSON 校验；默认可见规则降级，也可 `--no-rule-fallback` | 降级轮必须显示 `deterministic_safety_fallback`，不得冒充在线模型结果 |
| B07 | 是否显式授权远程处理学生文本 | 已实现并测试 | 必须传 `--allow-remote-student-data`；否则 fail-closed 或进入明确降级 | 真实学生数据还需伦理、告知和数据处理授权 |
| B08 | API Key 是否安全读取 | 已实现并测试 | `.private/deepseek_api.txt`（gitignored）；启动脚本支持 `DEEPSEEK_API_KEY_FILE`，直接 `tsm` 支持 `TSM_DEEPSEEK_API_KEY_FILE`；Key 不进入网页、trace 或公开 receipt | 只应链接/指向本机私有文件，禁止提交到 Git |
| B09 | 上下文是否最小化和受限 | 已实现并测试 | 单一 `teaching_context`；默认最多 10 个相关回合、14000 字符硬上限；当前回答单份；较早事实通过 evidence-linked checkpoint 与可 replay `teaching_memory` 保留；预算、截断和证据指针可审计；答案原图不发送，图片只增加有界脱敏 OCR 文字包络 | 发送的是教学文本、必要状态和可选 OCR 文字证据，不应笼统称“完全离线”；模式脱敏仍有剩余身份风险 |
| B10 | 是否完成隐私脱敏 | 部分实现 / 受限 | 常见邮箱、手机号、身份证号、URL 和本机路径模式替换；测试覆盖 | 这是模式级保护，不保证任意自由文本完全去标识化 |
| B11 | 诊断来源是否可审计 | 已实现并测试 | `assessment_source` 区分 `deepseek_v4_flash`、`deepseek_v4_flash_constrained_by_deterministic_contract`、`active_question_contract_exact_match`、`teacher_knowledge_spec_exact_match`；失败轮另以 `deterministic_safety_fallback` 标记。动作另以 `action_provenance.executor_origin` 区分模型安全话语与确定性 materializer | 只有第一类是未发生契约归一化的模型诊断；两种精确命中仍保留 model raw signal，fallback 不生成候选画像；动作来源和诊断来源不能混为一谈 |
| B12 | 冷恢复时模型/提示/上下文政策会不会漂移 | 已实现并测试 | live Session 持久化无密钥 `runtime_policy_contract`，绑定 provider、model、base origin、thinking、temperature、远程数据授权、prompt version、fallback、support 上限、最低置信度、上下文预算和 action executor；advance 和 cold recovery 均精确比对 | API Key 本身不写入 contract，可安全轮换；政策任一差异都拒绝继续，不能静默换模型或 prompt |

| B13 | 是否是真实的多步、可观测 Agent，而不是一次性 Prompt | 已实现并测试（本机真实 API 验收） | 生产 Dashboard 显式开启 `agent_loop_enabled=true`。每个回合先运行 bounded plan→allowlisted-tool→local-result→plan 循环；工具白名单固定为 `inspect_student_state`、`inspect_recent_history`、`search_skills`、`select_skills`、`set_next_focus`、`evaluate_termination`。Loop 只在主 Skill 与 `next_focus` 经本地校验后发出 `route_ready`，再调用最终动作规划器；最终仍只提交一个教师动作。当前上限为 6 步、每步 3 次工具调用、重复调用保护为 2 次、模型边界错误重试 1 次。公开 `last_agent_loop` 只含事件、工具名、计数、短原因、元数据和哈希，不含 prompt、思维链、学生原文或媒体。显式 `--session-store` 后，Loop receipt 随 checkpoint 进入带连续 `seq`、`previous_hash` 和 SHA-256 的 append-only JSONL；恢复会重放 turn 生命周期并对运行政策/版本漂移 fail-closed。真实 `deepseek-v4-flash` 健康检查返回 HTTP 200；两轮本机教学分别到达 `route_ready`（6 步/8 工具调用、5 步/7 工具调用），并观察到 Skill 动态切换；loopback 浏览器验收覆盖画像切换、刷新恢复和 `/stop` 终止恢复 | 这证明工具编排、约束和会话工程可运行，不建立真实学生诊断准确率、教学质量、跨 session 泛化、部署准确率或学习效果；哈希链不是加密或身份认证；库级 `LiveAgentOptions()` 默认关闭 Loop，生产 CLI 才显式开启；项目是 Codex-inspired 原创编排，不声称复制 Codex/Claude Code |

生产入口的 route adjudication 与库默认值有意分层：CLI 构造 dashboard 时开启 state-first 和 action-only repair，直接调用 `LiveAgentOptions()` 的兼容路径仍保持关闭。状态优先裁决器不是第二个模型，它只在模型诊断完成后按显式状态和 Skill 契约排序；模型候选只作为同一优先层的 tie-break。连续性召回、视觉确认和手动锁定等已经有安全 retarget 的路径不会再次走 state-first。

## 4. Teaching Skill Library 与 neural-v1 来源

| 编号 | 验收问题 | 状态 | 当前交付与证据 | 边界或剩余工作 |
|---|---|---|---|---|
| C01 | 默认是否使用 v2 Skill Library | 已实现并测试 | `data/teacher_agent_skill_library_v2.json`；CLI 和 Dashboard 默认路径均为 v2 | 题目一的 v0 展示产物仍保留其原边界，不与 v2 混称 |
| C02 | Skill 数量和角色是否明确 | 已实现并测试 | 16 个 Skill：13 个主 Skill + 3 个 support Skill；Schema 和单测验证 | 不得继续使用旧文档的“8 主 + 2 支持”描述 |
| C03 | Skill 是否可执行 | 已实现并测试 | 每个 v2 Skill 含 applicable/contraindications、pre/postconditions、failure_transition、max_repeat、addition_reason；控制器硬校验 ID/角色、`applicable_signals`、纠错证据、材料存在性、部分高风险阶段前置条件、主/支持 `max_repeat`、主 Skill 的 support allowlist、support 自身局部门禁、单动作和停止条件；自动、手动与 fallback 共用安全子集 | 其余自然语言 pre/contra/post/failure 条件进入提示与审计，但尚非形式化规则；字段完备或模型读到合同都不证明全部合同已被机器理解，也不证明教学法有效 |
| C04 | 是否使用 neural-v1 | 已实现并测试 | v2 `derivation.general_skill_id=evidence_grounded_multimodal_teaching_neural_v1`；12 个操作包装 | 使用的是九环节/十三策略本体，不是已确认视频共识 |
| C05 | 新增 Skill 是否说明原因和场景 | 已实现并测试 | 检索式复习、自我解释、参与恢复、信心支持均有单独研究补充来源和 `addition_reason` | 研究补充不得标成课堂视频蒸馏结论 |
| C06 | neural-v1 证据门禁是否通过 | 外部验证待完成 | manifest 固定记录 `passed=false`、eligible 0、excluded 54、observed phase 0、observed consensus strategy 0 | 页面和报告必须显示 `provisional · evidence gate not passed` |
| C07 | neural-v1 识别准确率、跨课程泛化、教学效果是否建立 | 外部验证待完成 | manifest 中三项均为 `false` | 需要合格证据、独立标注、锁箱和真实学习实验 |

## 5. 评估证据

| 编号 | 验收问题 | 状态 | 当前交付与证据 | 边界或剩余工作 |
|---|---|---|---|---|
| D01 | 是否有自由文本诊断测试样例 | 已有开发证据 | 28 个作者构造中文一轮案例；4 领域，7 类各 4 例；冻结 JSON 与输入哈希 | 未经专家验证，且未在提示词开发后保持独立锁定 |
| D02 | 自由文本信号判断表现 | 已有开发证据 | `teacher_agent_free_text_diagnose_route_v3_shared_taxonomy`：Accuracy 0.892857，Macro-F1 0.875325；与 live question-contract 共享 v3 诊断语义 | 仅 post-hoc 单轮 development regression；benchmark prompt 不是完整 live Session prompt，真实诊断 Accuracy 未建立 |
| D03 | Skill 路由与切换表现 | 已有开发证据 | 允许主 Skill 命中率 0.750000；切换 F1 0.787879；终止 F1 1.000000 | Skill gold 为作者构造，只验证单轮路由，不能写成完整 Session 或部署路由质量 |
| D04 | 是否有弱基线 | 已有开发证据 | 常量 partial：Accuracy 0.142857/Macro-F1 0.035714；固定诊断 Skill 命中 0.285714 | 只用于开发 sanity check |
| D05 | 是否有 oracle 上界参照 | 已有开发证据 | 读取 gold signal 的结构化路由器命中 0.964286 | 它使用金标准信息，不是同条件基线 |
| D06 | 是否报告在线运行稳定性 | 已有开发证据 | 28 例失败率 0；P50 994.887 ms，P95 1250.134 ms | 只代表本次单轮 benchmark 运行环境，不是完整 live Session 延迟、SLA 或部署压测 |
| D06a | 是否有多轮上下文与行为 benchmark | 已实现评估管线 | 20 个作者构造 episode、65 个学生回答回合和 1 次画像替换操作，覆盖长期记忆、图片证据、知识纠错、Skill 切换、画像隔离、提示注入、终止与恢复；Schema、blind payload、评分器和 paired runner 已测试。兼容名 `current` 固定为 `deterministic_legacy`，`safe_generative_executor` 使用生产 integrated `safe_generative`。为保持 paired 评分的请求拓扑可解释，benchmark executor 当前沿用 `LiveAgentOptions` 的兼容默认 `agent_loop_enabled=false`，两臂每轮都有 1 次主 plan；安全候选臂仅在选项启用、候选动作不匹配且 eligibility 通过时至多追加 1 次不能改 diagnosis/primary/support/termination/route 的 fixed-route action-only repair。报告分别记录 `request_topology`、`action_provenance`、repair 使用情况、`validated_model_plan_count_delta` 和端到端 turn latency；传输重试不计为额外教学动作，repair 不计为第二个 validated plan | 未经专家复核、不是提示词开发后的锁箱集；benchmark 的单 plan 拓扑不能替代生产多步 Loop 的在线质量，调用拓扑、动作来源、plan 计数和 latency 必须分开解释，当前公开文档不填报尚未完成或尚未审核的在线数值 |
| D06b | 多轮 gold 是否泄露给模型 | 已实现并测试 | runner 递归排除 `gold`、允许信号/Skill、必答词、延迟回忆词、终止/切换期望等全部评分字段；报告固定 `gold_sent_to_model=false`，不保存学生/教师正文、prompt 或供应商响应体 | fixture 与代码仍由同一项目开发者构造；不泄露 gold 不等于专家独立性或真实学生外部效度 |
| D07 | 是否有完整机制回归 | 已实现并测试 | 4 条结构化合成轨迹；3 成功、1 安全转人工 | 使用 gold structured signals，不检验自由文本诊断 |
| D08 | 是否比较固定单 Skill 基线 | 已实现并测试 | 自适应/固定内部增益 37.333250/20.416750，差 +16.916500；迁移通过率 0.75/0 | 内部模拟状态差，不是真实学生效果 |
| D09 | 状态、决策、行为、终止是否量化 | 已实现并测试 | 状态 1.0；允许决策 0.916667；行为 1.0；多 Skill 案例率 1.0；终止 1.0 | 全部是 4 例机制回归指标 |
| D10 | 是否有学习效果计算接口 | 已实现并测试 | 前/后测、迁移测、可选延迟测；输出绝对/归一化增益、迁移和保持率 | 输入分数由 fixture/教师/授权记录提供，系统不独立核验评分正确性 |
| D11 | 随附学习结果是什么 | 已实现并测试 | 作者演示：0.4→0.8，绝对增益 0.4，归一化增益 0.666667，迁移 0.666667 | 不是实人数据，不建立因果学习效果 |
| D12 | 是否完成专家教学质量盲评 | 外部验证待完成 | 已有量表与建议协议，尚无独立双人标注结果 | 至少需双人独立标注、分歧裁决和一致性指标 |
| D13 | 是否完成锁箱/跨 session 测试 | 外部验证待完成 | 当前 28 例单轮集和 20 episode 多轮集都不是 prompt/执行器开发后的 held-out lockbox | 应按 learner/session/problem/KC 分组，开发后锁定并由独立人员保管 gold |
| D14 | 是否完成真实学习者效果实验 | 外部验证待完成 | 学习结果 Schema 和计算器已准备 | 需预先方案、对照条件、前后测/迁移或延迟测和合规授权 |
| D15 | 是否完成部署质量验证 | 外部验证待完成 | 本机多 Session 注册表、并行隔离/恢复/冲突拒绝的自动生命周期验收入口 | 工程并发测试不等于部署验证；仍无真实负载下的接管率、故障率、长期跨课程报告或部署 Accuracy |

多轮请求统计固定使用 `request_accounting_scope=completed_committed_turns_only`：`validated_plan_request_total`、`action_repair_request_total` 和 `logical_model_request_total` 分开报告；repair 不是第二个 validated plan，但确实是第二个 logical request。`action_repair_adoption_rate` 的分母是实际发起 repair 的回合，而 `action_only_repair_rate` 的分母是 generator-eligible 回合；失败、取消和未提交回合不进入请求统计，且 provenance 与请求数不一致时 fail-closed。

## 6. 前端、运行和公开交付

| 编号 | 验收问题 | 状态 | 当前交付与证据 | 边界或剩余工作 |
|---|---|---|---|---|
| E01 | 是否有可现场操作的网页 | 已实现并测试 | 学生对话优先的三栏工作台：三种合成画像、文字/答案图片输入、连续对话与固定输入框、学情/方法/证据检查器、AI 合成头像与四维等权掌握环；响应式抽屉具备 inert、模态语义、焦点约束与归还；静态 UI 合同、HTTP 黑箱 runner 与 Playwright Chrome runner 分层覆盖资源、API 生命周期、image-only 印刷文字回合、明确正确回答识别、一次预期 400 的 stale replacement 单次恢复、画像 A→B 后继续作答、刷新、评估页和三种宽度；另有 `run_teacher_agent_cancel_browser_acceptance.py` 用阻塞合成模型在真实 DOM 中验收“停止生成”与取消后继续下一轮 | 可复现浏览器证据目前只覆盖 Chrome 合成印刷文字、一次受控竞态和一次阻塞模型取消路径；不把 Codex 内置浏览器是否可用或一次人工点击当作验收。Firefox/Safari、任意并发、屏幕阅读器、手写/公式准确率和真实部署仍需独立验证 |
| E02 | 是否展示题目要求的五步演示 | 已实现并测试 | 学习视图展示新目标、连续多轮、Skill 选择和动态切换；独立“实验 / 评估”视图展示样例、基线和结果 | 在线调用与冻结评估明确分区，不能把开发指标当当前学生成绩 |
| E03 | 是否有一键启动 | 已实现并测试 | `打开题目二教学Agent.command`；读取 `.private` 链接或启动脚本专用 `DEEPSEEK_API_KEY_FILE` | 缺 Key 时应明确报错，不能伪造在线调用 |
| E04 | 本机服务是否受限 | 已实现并测试 | `127.0.0.1`、随机 capability URL、CSP、`no-store`、请求体上限、在线远程处理确认、轮次与幂等校验 | 本地端口不要转发到公网 |
| E05 | 会话如何保存 | 部分实现 / 受限 | 默认服务端有界保存最多 16 个相互隔离的内存 Session；浏览器 `sessionStorage` 只保存 opaque handle。显式 `--session-store` 后，本机 append-only JSONL 可冷恢复同一 Session，并持久化目标、画像、对话、状态、幂等缓存和尚存附件的受限 OCR 证据；原图不存 | handle 仍可被同源 JavaScript 或 DevTools 读取；JSONL 是明文私有数据，哈希链只做完整性检测，不具备登录身份、加密、数据库权限模型或跨设备同步 |
| E06 | 是否可无 Key 自检 | 已实现并测试 | `teacher-agent-dashboard --check`；deterministic backend 可展示规则基线 | 离线模式必须显示不是 DeepSeek 在线 Agent |
| E07 | 是否有完整依赖和调用说明 | 已实现并测试 | README、本文、任务说明、CLI `--help`、Schema 与测试；真实浏览器验收提供轻量 `browser-test` extra 与 Playwright 浏览器安装命令 | 在线运行会产生第三方 API 请求和费用；Playwright 浏览器需操作者显式下载 |
| E08 | 是否避免提交密钥和私有记录 | 已实现并测试 | `.private/` gitignore；答案原图不持久化或发送给 DeepSeek；浏览器 runner 只输出聚合 receipt；公开 receipt 只有聚合指标和哈希；隐私审计 | 学生原始回答、OCR 正文、session-store JSONL、答案原图、附件句柄、API Key 和真实媒体不应进入 Git、wheel 或公开 artifacts |

## 7. 自动验收命令

常规工程验收不需要调用收费 API：

```bash
python3 -m teaching_skill_miner teacher-agent-dashboard --check
python3 -m teaching_skill_miner teacher-agent-evaluate \
  --library data/teacher_agent_skill_library_v2.json \
  --output artifacts/private/teacher_agent_evaluation.json
python3 -m teaching_skill_miner teacher-agent-outcome-evaluate \
  --input data/teacher_agent_learning_outcome_demo.json \
  --output artifacts/private/teacher_agent_learning_report.json
python3 scripts/run_teacher_agent_multiturn_benchmark.py --validate-only
python3 -m pytest -q \
  tests/test_deepseek_client.py \
  tests/test_teacher_agent.py \
  tests/test_teacher_agent_context.py \
  tests/test_teacher_agent_live.py \
  tests/test_teacher_agent_benchmark.py \
  tests/test_teacher_agent_outcomes.py \
  tests/test_teacher_agent_dashboard.py \
  tests/test_teacher_agent_store.py \
  tests/test_teacher_agent_memory.py \
  tests/test_teacher_agent_live_memory_integration.py \
  tests/test_teacher_agent_multiturn_benchmark.py \
  tests/test_teacher_agent_vision.py \
  tests/test_teacher_agent_ui_contract.py \
  tests/test_teacher_agent_system_acceptance.py
python3 -m ruff check teaching_skill_miner tests scripts
python3 -m compileall -q teaching_skill_miner scripts
zsh -n 打开题目二教学Agent.command
python3 scripts/audit_repository_privacy.py .
```

在线 smoke test 必须由有权使用密钥和测试文本的人显式触发；不应放入公共 CI：

```bash
tsm teacher-agent-dashboard \
  --agent-backend deepseek \
  --model deepseek-v4-flash \
  --api-key-file .private/deepseek_api.txt \
  --allow-remote-student-data
```

机器可读验收至少包括：

- v2 library 恰为 16 个 Skill，其中 13 个主 Skill、3 个 support；
- 有效 DeepSeek 回合生成 `candidate_unconfirmed` 候选画像，保留上限为 12，`teacher_provided_fields_overwritten=false`；低置信度候选需人工复核，fallback 后候选数不增加；
- 在线有效轮的 `provider=deepseek`、`model=deepseek-v4-flash`、`decision_origin=deepseek_v4_flash_constrained`；
- `assessment_source` 明确区分原始 DeepSeek、确定性契约修正、本问契约精确命中和教师知识规格精确命中；fallback 另标 `deterministic_safety_fallback`；
- `action_provenance.executor_origin` 区分 `deepseek_safe_generative` 与 `deterministic_materializer`；
- fallback 轮明确标为 `deterministic_safety_fallback`；
- 图片原图始终 `remote_media_sent=false` 且不保留；附件与当前 question/round/profile/context 绑定并只消费一次；远程表示固定为有界脱敏 OCR 文字；
- Apple Vision 与多预处理 Tesseract 路由可交叉核对；可靠公式转写可设 `formula_transcription_established=true`，但 `formula_accuracy_established=false`；只有精确命中可判定 question contract 或教师 knowledge spec 才能确定性判对，低置信/冲突/未佐证公式仍确认；
- session store 的 started→committed/aborted、hash-chain、runtime-policy 和 Skill 子集恢复均 fail-closed，原图不进入 store；
- neural-v1 manifest 继续显示 gate `passed=false`，不得在展示层改成 final；
- 28 例 receipt 的 `held_out_after_prompt_development=false`，且诊断/路由/部署/学习效果声明均保持 `false`；
- 20 episode/65 个学生回答回合（另含 1 次画像替换操作）benchmark 的 gold 不进入模型 payload，且未经专家复核、未锁箱、不建立部署准确率；
- 4 例结构化回归的 `free_text_answer_grading_established=false` 和 `simulated_pre_post_is_real_learning_effect=false`；
- 学习 outcome 报告的 `real_learner_effectiveness_established=false`；
- Key、真实学生回答和私有媒体不出现在 tracked files、网页静态资源、wheel 或公开 receipt。

## 8. 答辩可用表述

建议表述：

> 我们实现的是一个真正逐轮运行的混合 Teaching Agent。DeepSeek V4 Flash 负责理解学生自然语言、判断当前问题对齐、建议 Skill 并生成当前动作候选；确定性控制器维护状态与安全边界。通过最终 Skill、单动作、问题契约和防答案泄露门禁的安全话语可以展示，否则由确定性 materializer 接管。系统默认使用 v2 的 13 个主 Skill 与 3 个支持 Skill，并能显示每轮选择、组合、切换、理由和动作来源。28 例作者构造单轮开发集有已记录数值；另有 20 episode、65 个学生回答回合并含 1 次画像替换操作的作者构造多轮评估管线，但当前不填报尚未完成的在线数值。它们都不替代专家锁箱、真实学生学习效果和部署验证。

禁止表述：

- “neural-v1 已经从 10 个视频识别出可靠通用 Skill”；
- “教师 Agent 的真实准确率是 89.29%”；
- “内部模拟增益 +16.9165 证明学生成绩提升”；
- “系统完全离线、学生文本不会离开本机”；
- “答案图片直接由 DeepSeek 看图”或“手写/公式识别准确率已经建立”；
- “session store 已加密”或“哈希链等于隐私保护”；
- “20 episode benchmark 已证明部署准确率或 Codex 等价能力”；
- “项目已经达到多用户部署标准”。
