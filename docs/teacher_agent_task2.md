# 题目二：基于 Teaching Skill Library 的自适应教学 Agent

## 1. 问题定义与当前结论

题目二要求的不是一次性生成整段教案，而是一个实时的多轮闭环：系统收到一个教学目标和学生初始状态，只生成当前一步；学生回答后，系统重新判断状态、选择或组合 Teaching Skill，再生成下一步，直到达标或安全转人工。

当前仓库已经实现这一工程闭环。`deepseek-v4-flash` 负责语义理解、问题对齐诊断和 Skill 路由建议；确定性控制器负责状态边界、Skill 合法性、重复上限、一次一动作和终止条件；服务端 Skill materializer 再按所选 Skill 的 `action_type`、执行契约、目标材料和上一问判据生成最终教师话语。模型自由文本 action 只用于一致性审计，不直接发给学生。三者共同构成一个受约束 Agent，不能把它简化成“只调用一次大模型”，也不能把规则层说成另一个模型。

| 题目要求 | 当前实现 | 可验证状态 |
|---|---|---|
| 新教学目标与成功条件 | 概念、目标、材料、四维阈值、最大轮数；自动展开为 4 个可观察子目标 | 已实现并有单测 |
| 学生画像 | 三个合成初始画像可整组切换水平、掌握、偏好与未标注背景；每个有效 DeepSeek 回合追加带证据的候选观察和摘要 | 已实现并有会话隔离测试；`background_history` 不作为学情标签，头像不对应真人、不参与能力判断 |
| 学生当前状态 | 四维掌握、误解生命周期、当前理解信号、下一关注点、置信度、证据和交互统计 | 已实现并逐轮更新 |
| 历史对话 | 六层 `teaching_context`：固定锚点、当前计划、近期工作记忆、较早历史的统计与证据关联选择性检查点、知识状态和未确认候选记忆 | 已实现并有预算/来源测试；不是完整历史语义总结、无限或跨 session 记忆 |
| Skill 选择、组合与切换 | 13 个主 Skill 中选 1 个，可组合最多 2 个支持 Skill；主 Skill 的 `supporting_skill_ids` 是硬组合 allowlist，support 还须通过自身门禁；每轮重新决策并公开理由 | 已实现；开放场景最优性待专家验证 |
| 实时交互 | 一次请求只消费一条学生回答，只返回一个下一教学动作 | 已实现；不是预写好多轮对话 |
| 答案图片证据 | 学生可提交文字、答案图片或两者；原图本机 OCR 后，只把有界、模式脱敏的文字证据并入当前回答 | 已实现并有单元、HTTP 和 image-only Chrome 验收；不是 DeepSeek 原生视觉理解，手写/公式部署准确率未建立 |
| 终止 | 成功、连续无进展、最大轮数、受约束模型建议或教师 `/stop` | 已实现并有安全门 |
| 评估 | 28 例自由文本开发 benchmark、4 例结构化机制回归、学习结果记录接口 | 已实现；三类证据的含义不能混用 |

## 2. 系统整体设计

```mermaid
flowchart LR
    A["输入<br/>教学目标 · 学生初始画像<br/>当前文字回答"] --> B["Goal 计划器<br/>前置 · 概念 · 过程 · 迁移<br/>各自有可观察成功标准"]
    I["可选答案图片<br/>PNG · JPEG · WebP"] --> X["本机 OCR<br/>macOS: Apple Vision 优先<br/>其他平台或失败: Tesseract"]
    X -->|"有界并做模式脱敏的 OCR 文字"| H
    I -.-> P["原图只在内存/临时目录处理<br/>不发送给 DeepSeek"]
    H["六层受限上下文<br/>目标 · 当前问题契约 · 工作记忆 · 历史检查点<br/>知识状态 · 未确认候选记忆<br/>最多 6 个相关回合 / 14000 字符"] --> C
    B --> C["DeepSeek V4 Flash<br/>① 对齐当前问题并诊断<br/>② 提议主/支持 Skill<br/>③ 给出结构化动作意图"]
    L["v2 Skill Library<br/>13 主 + 3 支持<br/>触发 · 禁用 · 前后置条件<br/>失败转移 · max_repeat"] --> C
    C --> V["确定性验证与控制<br/>ID/角色 · 适用信号 · 纠错证据<br/>高风险前置/材料 · 重复上限<br/>support allowlist/自身门禁 · 终止安全"]
    V --> M["服务端 Skill materializer<br/>action_type · 执行契约 · 教学材料<br/>生成 message / expected_signal / question_contract"]
    M --> S["状态与候选画像更新<br/>掌握 · 误解 · 理解信号<br/>交互统计 · 候选观察 · 证据"]
    S --> O["输出当前一步<br/>Skill 与选择理由<br/>服务端教师动作 · question_id / 判据"]
    O --> R{"等待学生真实反馈"}
    R -->|"继续"| H
    R -->|"达标"| T1["成功终止"]
    R -->|"无进展 / 上限 / 人工停止"| T2["停止并转人工"]
```

一轮的输入与输出边界如下：

```text
输入：目标 + 初始画像 + 当前显式状态 + 相关历史 + 本轮文字回答/本地 OCR 文字证据 + 可用 Skill
  图片支路：答案原图 → 本机 OCR → 长度约束与常见直接标识符模式替换 → teaching_context
             原图、缩略图、本机路径和临时文件不进入 DeepSeek 请求
  ↓
DeepSeek：受约束的“诊断—路由—动作意图”JSON 建议
  ↓
控制器：校验证据与 Skill、更新状态与候选画像、判断是否终止
  ↓
Skill Runtime：按选中 Skill 契约物化最终话语与下一问题判据
  ↓
输出：一个主 Skill、0–2 个支持 Skill、选择理由、一个教师动作、期待信号
```

因此，“实时”在本项目中指学生每回答一次，后端才计算一次下一步；浏览器不会提前得到后续完整对话。学生可以提交文字、答案图片或两者。答案图片不是直接交给 DeepSeek：服务端先校验格式和大小，在本机临时目录运行 OCR；受支持的 macOS 主机优先调用 Apple Vision，本机 Apple Vision 不可用、失败或未识别出文字时回退到 Tesseract，其他平台在安装 Tesseract 后使用 Tesseract。OCR 文本最多保留有界长度，并与普通学生文字一起经过常见直接标识符模式替换后进入 `teaching_context`；原图随后删除且不会发送给远程模型。这是 OCR 辅助的文本证据链，不是 DeepSeek 原生图像理解，也不表示流式摄像头或实时音视频识别。当前高对比度印刷文字可用于演示链路；手写、复杂版面和公式的部署准确率没有建立，检测到公式样文本时必须让学生核对关键式子。

## 3. DeepSeek 在 Agent 中具体做什么

`deepseek-v4-flash` 每轮只接收一个经过最小化和校验的 `teaching_context`，以及有界的 Skill 执行契约。若本轮含答案图片，模型看到的是标有 OCR 状态、引擎置信度和“是否需学生确认”的 `[LOCAL_VISUAL_EVIDENCE]` 文字包络，不是原图；低置信、无文字或公式/代码样证据必须保守处理并请求学生确认。首轮在发送前只保留允许 `not_observed` 的主 Skill 与支持 Skill，避免在没有学生回答时跳过诊断门禁；普通回合发送会话内完整的允许 Library，由模型提出候选，再由确定性控制器按本轮 Skill ID/角色、`applicable_signals`、纠错证据、高风险阶段/材料前置条件、材料存在性、`max_repeat`、主 Skill 的 support allowlist 及 support 自身门禁做最终校验或安全改路。该上下文把固定目标、教师画像、当前 Goal 步骤、当前问题契约、近期对话、较早历史检查点、显式学生状态和未确认画像假设分层组织，再返回严格 JSON：

- `diagnosis`：`correct / partial / misconception / confused / no_response`，以及相对本问的 `answer_alignment`、匹配/缺失概念、置信度、短证据、误解标签、回答质量、参与度和是否建议人工复核；
- `decision`：一个主 Skill、最多两个支持 Skill、选择或切换理由、下一关注维度；
- `teacher_action`：模型给出结构化动作意图，供 action type 一致性审计；最终 `message / expected_signal / question_contract` 由服务端按已选 Skill 重新物化，再绑定 `question_id`；
- `stop_recommendation`：是否建议停止及理由。

模型没有以下权限：新增不存在的 Skill、选择 support Skill 充当主 Skill、组合主 Skill 未在 `supporting_skill_ids` 中声明的 support、绕过 support 自身门禁、超过 Skill 的 `max_repeat`、重写历史、直接改变终止状态，或一次生成多轮教学。模型提供的 message 不会直接进入学生界面；即使 action type 正确，服务端仍用 Skill materializer 覆盖它，并对最终动作再做防答案泄露校验。

视觉证据确认是独立于模型判断的硬合同。只要附件低于数值置信阈值、未识别出文字、被识别为公式/代码样表达，或本机证据记录要求学生确认，学生附带“答案见图”“照片这样显然是对的”等空泛文字也不能把它变成已确认答案。控制器把该轮固定为 `partial / ambiguous`、置信度 `0.0`，不增加任何掌握维度、不解除误解，且不保留模型声称但未执行的 support / `support_execution`。确认动作只允许使用 `skill_self_explanation` 或 `skill_socratic_understanding_check`：优先执行自我解释，触及其连续 `max_repeat` 后切换到苏格拉底理解检查，后续再按两者各自门禁和重复上限契约式轮换；下一动作始终只要求学生用自己的话确认或修正关键概念、符号或第一步。即使模型 JSON 无效并进入 `deterministic_safety_fallback`，同一轮换合同仍然生效。

产品运行和开发 benchmark 现在复用同一份诊断标签定义。只有直接满足当前问题判据才是 `correct / aligned`；相关但没有回答本问是 `partial / related_but_not_answer`；明确表示“不知道、没懂、分不清”才是 `confused`；只有可定位的错误命题才能写入 `misconception`。因此“保存已经算过的结果叫什么？”这一问中，“状态转移方程”是相关线索但不是“记忆化/缓存”的直接答案，应判部分对齐并继续澄清，而不是强行算正确或据此断言学生整体困惑。

若 API、格式或校验失败，默认进入可见的确定性安全降级：空回答记为 `no_response`；明确自报“不知道、没懂、分不清”等困惑时可记为 `confused`；其余非空回答保守记为待复核的 `partial`，不凭规则臆造误解。动作标记为 `decision_origin=deterministic_safety_fallback`。这个结果不是 DeepSeek 结果；只有页面显示 DeepSeek 来源时，才能称该轮使用了模型诊断。

诊断证据还单独记录 `assessment_source`，页面按以下来源分类展示，避免把控制器修正或规则回退混成“模型原判”：

| 来源 | 含义 | 可否称为原始模型诊断 |
|---|---|---|
| `deepseek_v4_flash` | 模型输出通过问题、证据和标签契约，未发生确定性归一化 | 可以，但仍是未建立部署准确率的模型判断 |
| `deepseek_v4_flash_constrained_by_deterministic_contract` | 模型完成诊断，但控制器因问题对齐、置信度、误解证据或其他契约要求修正了最终标签/属性 | 不可以；应称“模型诊断经契约修正” |
| `active_question_contract_exact_match` | 当前短概念回答精确命中服务端绑定的目标概念或同义词，契约结果覆盖内部不一致的模型标签 | 不可以；应称“本问契约精确命中”，并保留原始模型标签供审计 |
| `deterministic_safety_fallback` | API、构建、格式或校验失败后的规则安全信号 | 不可以；它不是自由文本理解结果，也不会生成候选画像 |

## 4. 学生状态、候选画像与历史怎样维护

### 4.1 显式状态

系统每轮都保存至少题目要求的四类信息：

| 状态组 | 字段与含义 |
|---|---|
| 当前掌握情况 | `knowledge_mastery`：`prerequisite`、`conceptual`、`procedural`、`transfer` 四维内部控制值 |
| 误解或错误模式 | `misconceptions`：误解标签、描述、置信度、发现轮次以及 `active/resolved` 状态 |
| 当前理解信号 | `understanding_signal`、`assessment_confidence`、`assessment_evidence` |
| 下一教学重点 | `next_focus`：关注维度、理由和对应 Skill |
| 交互信号 | 尝试次数、各信号计数、滚动正确率、平均回答长度、参与度和回答质量 |

回答产生的动态信息进入可审计的 `student_state`。掌握值是控制状态，不是考试成绩，也不是经校准的真实掌握概率。

### 4.2 动态学生画像候选

教师提供的 `learner_level`、`preferences` 和 `accessibility_needs` 是基础画像，Agent 不会覆盖这些字段。每个通过 JSON、Skill 和证据校验的 DeepSeek 回合，系统会在 `student_profile.adaptive_observations` 追加一条动态候选，内容包括：

- 本轮回答质量和参与度；
- 候选误解标签与下一关注维度；
- 与本轮已脱敏回答绑定的证据摘录和诊断置信度；
- 是否需要人工复核及原因。

每条记录固定为 `status=candidate_unconfirmed`。`adaptive_summary` 汇总最新回答质量、参与度、下一重点、候选误解标签、累计/保留观察数和复核状态；最多保留最近 12 条观察，总观察数继续累计。低于运行时置信度门槛（默认 0.35）、模型主动请求复核，或没有可绑定到本轮学生原话的证据摘录时，都会标记 `needs_human_review=true`；最后一种情况同时记录 `review_reasons=no_grounded_excerpt`，不能当成已有证据支持的画像事实。

只有经过约束层验证的 DeepSeek 诊断才产生候选；初始动作和 `deterministic_safety_fallback` 都不会生成。候选证据在保存前已做常见直接标识符模式替换；非空摘录必须是本轮脱敏回答的真实子串。模型伪造摘录时不会用整段学生回答替换，而是清空摘录；若模型试图在无绑定证据时给出 `correct` 或 `misconception` 这类高影响标签，系统会降为待复核的 `partial`，清除误解标签并拒绝误解除标。后续请求只把最近候选作为明确标记的 `candidate_unconfirmed` 低权重假设，不能当作已知事实，也不能覆盖教师画像。

这个设计实现了“随回答更新画像”，同时保留人工最终解释权：候选观察不是已确认人格或能力事实，也不会跨 session 持久化为长期画像。当前 Session 只存在于本机服务进程内；进程停止或重启后全部会话消失，旧 opaque handle 不能恢复。

### 4.3 Goal 模式

目标计划器把一个目标确定性地展开为 4 个可观察步骤：前置知识、概念理解、操作/推导和迁移。每一步包含目标、验证方式、成功判据和阈值；状态变化后同步更新 `active_step` 与进度。该进度用于控制教学流程，不是学习效果证据。

### 4.4 上下文管理

远程请求不会无界发送整个会话，也不再把 goal、state、history 和 profile 作为互相重叠的散装字段并列发送。唯一权威 `teaching_context` 分为六层：

1. `fixed_context`：教师输入的目标与基础画像，模型不可改写；
2. `current_plan`：当前 Goal 步骤、成功判据、上一教师动作及其问题契约；
3. `working_memory`：本轮回答以及最多 6 个相关回合；
4. `semantic_summary`：较早历史的确定性计数、focus checkpoints，以及最多 6 条 evidence-linked `teaching_checkpoints`；后者只从被省略回合中的明确师生原话或已记录结构化信号选择性抽取，每条绑定 `evidence_ledger`，不调用模型编写叙事摘要；
5. `knowledge_state`：四维掌握、活跃/已解决误解和未解决问题；
6. `candidate_long_term_memory`：最近的未确认画像假设，明确低权重且不可覆盖教师字段。

网页中的“既往上下文（未标注）”按换行拆分为 `student_profile.background_history`，与带 `signal + focus_dimension` 的结构化 `conversation_history` 明确分开。构建上下文时，每条背景只以 `label_status=unlabeled_background_not_state_evidence` 和来源 `teacher_provided_unlabeled_background` 进入 `unlabeled_notes`；`unlabeled_notes_may_update_state=false`。因此它可以帮助模型理解教师提供的先验背景，但不会在会话开始时伪造 `partial/confused` 等学生信号，也不会直接改变四维初始掌握、当前理解信号或误解生命周期。

检索顺序仍为“同知识点优先、同关注维度其次、最近轮次最后”。未填写知识点时，使用教师输入的 `goal.concept` 作为最小真实锚点；填写多个有序知识点时，每个动作只携带文字命中或与当前掌握阶段对应的活动知识点，不再把整份目标复制到所有回合。`teaching_checkpoints` 可以保留尚未清除的困难、学生明确问题/偏好、已验证前置知识或教师明确下一步，但未知的解决/完成状态不会由模型补写；`teaching_checkpoints_are_selective_extracts=true` 且 `omitted_turn_semantics_are_exhaustive=false`，因此统计上记录了省略轮次不等于完整保留其语义。默认总序列化硬上限为 14,000 字符，可配置为 6,000–30,000；每个快照记录实际保留轮次、字符数、截断状态和证据来源。合法超长会话会先丢弃低优先级候选、检查点和较旧回合，最终仍生成 schema-valid 的最小请求；若构建器本身异常则进入明确的规则安全降级。本轮学生回答只出现一次，证据账本保存的是指针，避免重复占用预算。

发送前会最小化画像字段，并对常见邮箱、手机号、中国身份证号、URL 和本机路径模式做替换，真实媒体不会发送。`contains_direct_identity=true` 或非 JSON 布尔值会在任何远程调用前 fail-closed。但系统无法仅靠模式规则证明自然语言姓名、学校或普通学号已全部识别，因此 trace 明确记录 `raw_identity_fields_sent=not_established`、`residual_identity_risk=true`。这里应称“常见直接标识符模式脱敏与上下文最小化”，不能承诺所有自由文本都已完全匿名；使用远程 API 前仍须取得授权，并避免输入不必要的个人信息。

### 4.5 会话生命周期与 Codex 参考

本项目没有复制 Codex，也没有把 Shell、Git、沙箱或多 Agent 能力混入教学系统；参考的是官方开源 Codex 的 thread/turn identity、预期轮次核验、陈旧异步结果隔离和 running/cold resume 等可靠性模式。Teaching Session 对应 thread，教学回合对应 turn。start、step 与 command 都有各自的幂等键；step/command 还必须同时匹配 `session_id + expected_round + expected_question_id + expected_context_version + profile_revision`。逐 Session 锁串行化同一会话的变更，`context_version` 在成功提交后递增，重放同一请求返回缓存，冲突键、旧问题、旧轮次、旧上下文或旧画像版本都 fail-closed。请求 body 指纹与有界响应缓存、画像 prepare-then-commit、16 槽内存注册表、六层教学上下文和问题契约是本项目实现，不归因于 Codex。

画像切换不是原地改写旧学生，而是 prepare-then-commit：服务先校验请求，完整构造新首轮并通过结构、Skill 与 Session 校验，随后才在注册表临界区提交新 Session、退休被替换会话并写入 start 幂等缓存。DeepSeek 失败不必然等于切换失败；若醒目标记的 `deterministic_safety_fallback` 也通过全部校验，它可以合法建立新画像 Session，否则没有合法候选时才保留旧会话。若页面持有的 replacement guards 已因 UI 外的同 Session 推进而陈旧，第一次 start 会按预期 400 拒绝；前端只允许调用一次 `api/session` 同步权威 guards，废弃旧 start 幂等键并用新键重试一次，第二次仍冲突则停止而非循环。Playwright runner 已主动制造并检查了这一受控路径。刷新页面时，浏览器只从 `sessionStorage` 取回随机、无业务语义的 opaque session handle，再向服务端恢复状态；它不保存教学正文，但仍可被同源 JavaScript 或 DevTools 读取，不能称为“不可读”。新标签页没有句柄时可以独立创建 Session；达到 16 个槽位时只回收能够立即加锁的闲置记录，所有槽位都在执行请求则拒绝新建。参考版本固定为 OpenAI Codex `f2d825533c9423728f319a6dbcbb31c21768aa69`；这里只借鉴前述可靠性模式，不声称迁移 Codex 的 UI 或上下文实现。

## 5. v2 Teaching Skill Library

默认库为 `data/teacher_agent_skill_library_v2.json`，共 16 个原子 Skill：13 个主 Skill和 3 个支持 Skill。每个 Skill 都定义适用场景、禁用场景、前置与后置条件、失败转移和连续使用上限。

| 角色 | Skill | 主要用途 |
|---|---|---|
| 主 | 前置知识诊断 | 首轮或前置知识证据不足时提问诊断 |
| 主 | 问题情境建立 | 用任务情境说明为什么需要当前概念 |
| 主 | 直观例子桥接 | 困惑时先建立可感知的具体表征 |
| 主 | 直觉—概念映射 | 从例子过渡到正式概念与边界 |
| 主 | 逐步支架推导 | 把复杂过程拆成一个可回答的小步 |
| 主 | 苏格拉底理解检查 | 用追问核验理由，而非只核对结论 |
| 主 | 误解对比纠错 | 对比错误规则与正确边界，定位误解 |
| 主 | 练习—反馈—再练习 | 从支架过渡到独立操作 |
| 主 | 检索式复习 | 需要恢复前置知识或间隔回忆时使用 |
| 主 | 自我解释与反思 | 要求学生解释自己的步骤和依据 |
| 主 | 变式迁移检查 | 在准备充分后测试新情境迁移 |
| 主 | 学习者总结 | 由学生自己概括方法、边界和易错点 |
| 主 | 参与恢复与重新聚焦 | 跑题、参与度下降或持续短回答时恢复互动 |
| 支持 | 提问后等待 | 约束当前动作留出真实回答空间 |
| 支持 | 最小提示 | 只给足以推进一步的提示，避免代做 |
| 支持 | 信心支持 | 在不替代知识教学的前提下降低挫败感 |

其中 12 个 Skill 是 neural-v1 暂定执行本体的操作包装；检索式复习、自我解释、参与恢复和信心支持含有单独标记的同行评审研究补充。研究补充没有被伪装成视频观察结果。

### neural-v1 的真实边界

neural-v1 的来源是 10 讲、2 门课程，采用 grouped 5-fold OOF 开发协议；训练流程使用多模态 backbone，但公开运行 manifest 的证据物化门禁没有通过：

| 字段 | 当前值 |
|---|---:|
| 证据门禁 | `passed=false` |
| 可物化预测 | 0 |
| 被排除预测 | 54 |
| 已确认观察环节 | 0 |
| 已确认跨讲共识策略 | 0 |
| 运行用途 | `provisional_normative_execution_ontology_only` |

因此题目二确实使用 neural-v1 的九环节、十三策略本体来组织运行时 Skill，但只能称为 **provisional 执行本体**，不能称为已确认的课堂观察共识、已建立识别准确率或已证明跨课程泛化。题目一现有 v0 通用 Skill 的展示边界不因此被改写；两者是不同状态的产物。

## 6. Skill 如何选择、组合、切换和终止

每轮模型先提出决策，控制器随后执行硬校验：

1. 主 Skill 必须来自允许列表且角色不是 `support`；
2. 自动模式下，诊断信号必须落在主 Skill 的 `applicable_signals` 内；纠错 Skill 还必须绑定误解 tag；
3. 若干容易造成阶段越级的高风险条件由本地代码形式化检查，例如具体例子、练习或迁移题材料必须存在，概念映射前应已执行例子，独立练习前应有支架经历，迁移/总结前须达到对应准备线，活跃高置信误解会阻断不合适的后续阶段；
4. 主 Skill 的 `supporting_skill_ids` 是硬组合 allowlist：未声明 support 一律丢弃；声明项仍须确为 `support` 角色，并通过自身的局部前置条件、适用证据和 `max_repeat`，去重后最多保留 2 个；
5. 主 Skill 与 support 都不能超过各自 `max_repeat`；fallback 也不得绕过适用信号、高风险前置条件、材料、allowlist 或重复上限；没有可执行 Skill 时停止并转人工；
6. 输出必须只有一个可响应动作，并明确等待学生；
7. 每轮记录上一 Skill、是否切换、选择理由、模型 trace 和状态证据；
8. 低置信度不会被抬高，也不会把任意非空回答硬改为 `confused`；明确困惑才保留该标签，其余非空不确定回答降为 `partial` 并标记人工复核；
9. 模型建议停止只有同时满足“建议人工复核 + 至少两轮无进展”才会被控制器采纳。

`preconditions`、`contraindications`、`postconditions` 与 `failure_transition` 会完整进入模型请求和审计记录，但自然语言合同并非都能由规则控制器形式化证明。上面列出的是当前机器实际检查的安全关键子集；其他自然语言条件仍用于模型约束和审计，不能把“字段存在”或“模型读到了合同”表述成“系统理解并验证了全部合同”。

控制器还独立处理成功、连续无进展、最大轮数和教师停止。在线模式支持：

- `/+skill 名称`：登记一个持续的主 Skill 锁；若在新建会话时预选，它从第一条学生回答后的路由开始生效，并持续到 `/auto` 或安全门释放；
- `/auto`：恢复自动路由；
- `/stop`：不再消费学生回答，立即停止并建议人工确认。

人工覆盖不会计为 Agent 自动选择命中，也不能绕过适用信号、纠错证据、重复上限、fallback 和终止约束；控制器拒绝不适用选择时会回到安全 Skill，并在页面显示锁已释放的原因。

## 7. 三层评估证据

三套结果回答的是不同问题，不能合并成一个“总 Accuracy”。

### 7.1 28 例自由文本在线开发 benchmark

最新公开聚合 receipt 来自 2026-08-06 的在线运行，使用 `teacher_agent_free_text_diagnose_route_v3_shared_taxonomy`、`deepseek-v4-flash`、thinking disabled、temperature 0、单次重复。28 个作者构造的一轮中文案例覆盖动态规划、线性代数、Python 和概率；7 类标签各 4 例：`correct`、`partial`、`misconception`、`confused`、`no_response`、`off_topic`、`valid_alternative`。该 benchmark 复用与 live question-contract 相同的 v3 诊断 taxonomy / 语义量表，但它的 prompt 是独立的单轮诊断与路由契约，不是包含分层上下文、状态更新、动作生成和多轮生命周期的完整 live Session prompt；因此这些结果只验证单轮诊断/路由，不等同完整 live Session 评测。

| 指标 | DeepSeek V4 Flash | 对照 / 解释 |
|---|---:|---|
| 7 类信号 Accuracy | 0.892857 | 常量 `partial` 基线 0.142857 |
| 7 类信号 Macro-F1 | 0.875325 | 常量 `partial` 基线 0.035714 |
| 误解标签完全匹配率 | 1.000000 | 只在冻结样例定义下解释 |
| 允许主 Skill 命中率 | 0.750000 | 固定诊断 Skill 为 0.285714 |
| Skill 切换 F1 | 0.787879 | 是否应切换的开发集标签 |
| 终止 F1 | 1.000000 | 是否应终止的开发集标签 |
| 端到端调用失败率 | 0.000000 | 本次 28 次运行 |
| P50 / P95 延迟 | 994.887 / 1250.134 ms | 本次 API 运行环境 |

Gold structured-signal oracle router 的允许 Skill 命中率为 0.964286，但它读取金标准信号，不能与端到端模型作同条件比较。

这 28 例是作者构造、未经专家复核、未在提示词开发后保持独立锁定的 **post-hoc development regression**。所以 0.892857 不能写成“真实学生诊断准确率”，0.750000 不能写成“部署路由准确率”，也不能据此证明完整 live Session 质量或学习效果。公开 receipt 只含聚合值和哈希，不含案例正文、供应商原始响应或密钥。

复现在线 benchmark 需要显式允许远程处理该公开 fixture：

```bash
tsm teacher-agent-benchmark \
  --online \
  --allow-remote-benchmark-data \
  --api-key-file .private/deepseek_api.txt \
  --output artifacts/private/teacher_agent_free_text_benchmark.json
```

### 7.2 4 例结构化机制回归

四条合成轨迹使用预先给定的结构化信号，比较动态 Skill Agent 与始终使用 `skill_stepwise_scaffolding` 的固定单 Skill 基线。它检验控制器是否按预期更新和切换，不检验自由文本理解。

| 指标 | 自适应 v2 Agent | 固定单 Skill |
|---|---:|---:|
| 状态轨迹一致率 | 1.000000 | — |
| 允许决策匹配率 | 0.916667 | — |
| 教学行为约束通过率 | 1.000000 | — |
| 使用多个 Skill 的案例率 | 1.000000 | 0.000000 |
| 终止决策匹配率 | 1.000000 | — |
| 内部模拟平均增益 | 37.333250 | 20.416750 |
| 内部模拟增益差 | +16.916500 | — |
| 模拟迁移通过率 | 0.750000 | 0.000000 |

前三例预期成功，第四例预期在持续无进展后安全转人工。内部模拟增益来自状态机，不是真实前后测，也不是因果学习效果。

```bash
tsm teacher-agent-evaluate \
  --library data/teacher_agent_skill_library_v2.json \
  --output artifacts/private/teacher_agent_evaluation.json
```

### 7.3 学习结果记录接口

`teacher-agent-outcome-evaluate` 接受前测、后测、可选迁移测和可选延迟测，计算绝对增益、归一化增益、迁移比例和延迟保持率。输入 provenance 必须明确是作者演示、教师提供记录还是获授权真实学习者记录。

随附 fixture 是作者构造演示：前测 2/5、后测 4/5、迁移 2/3；绝对增益 0.4、归一化增益 0.666667、迁移比例 0.666667。它只证明接口可计算，不证明系统改善了真实学生学习。

```bash
tsm teacher-agent-outcome-evaluate \
  --input data/teacher_agent_learning_outcome_demo.json \
  --output artifacts/private/teacher_agent_learning_report.json
```

## 8. 本机启动与现场演示

### 8.1 配置密钥

密钥不进入仓库。可将外置盘密钥链接到被 `.gitignore` 忽略的本机目录：

```bash
mkdir -p .private
ln -s "/path/to/deepseek_api.txt" .private/deepseek_api.txt
```

也可以不建立链接：双击启动脚本前设置 `DEEPSEEK_API_KEY_FILE`，或直接运行 `tsm` 时设置 `TSM_DEEPSEEK_API_KEY_FILE`。两者都只应指向自己的本机私有密钥文件；密钥只由本机服务读取，不进入页面和公开运行日志。

### 8.2 启动

macOS 直接双击根目录的 `打开题目二教学Agent.command`。等价命令为：

```bash
tsm teacher-agent-dashboard \
  --agent-backend deepseek \
  --model deepseek-v4-flash \
  --api-key-file .private/deepseek_api.txt \
  --allow-remote-student-data
```

离线自检和无 API 的规则基线分别为：

```bash
tsm teacher-agent-dashboard --check
tsm teacher-agent-dashboard --agent-backend deterministic
```

页面只绑定 `127.0.0.1`，使用随机 capability URL、CSP 和 `Cache-Control: no-store`；服务端只保留有界内存会话，浏览器仅保存随机、无业务语义的 opaque 恢复句柄。句柄不含教学正文，但仍可被同源 JavaScript 或 DevTools 读取。页面本地运行不等于全部处理离线：在线模式会把最小化并做常见直接标识符模式替换后的必要文本发送给配置的 DeepSeek API。答案图片原图仅供本机 OCR 临时处理，远程侧只接收长度受限、经过同一模式替换的 OCR 文字包络；图片哈希、OCR 状态和置信度用于本轮审计，原图和临时路径不进入远程请求。公式样或低于共享数值阈值的 OCR 会由服务端强制要求学生确认；客户端不能靠伪报状态绕过。模型引用多行 OCR 时只允许大小写和空白排版差异，并从原始可信文本回取证据片段。模式替换不能证明任意自然语言身份或 OCR 误识别出的身份信息已完全去除，因此仍需学生/操作者先人工移除姓名、学号、学校、地址等直接标识。网页采用“左侧目标与合成画像—中间学习对话—右侧学情/方法/证据检查器”的工作台；论文评估被放在单独视图，不与学生输入混在一起。同一 attachment/step/command 还会校验会话、预期轮次、问题、上下文版本、画像版本和幂等键，网络重试不会把一次逻辑操作计成两轮，也不能把旧图片绑定到新问题或新画像。

真实浏览器验收额外依赖 Pillow 与 Playwright；仓库提供轻量 extra，不需要安装完整视觉模型栈：

```bash
python3 -m pip install -e '.[browser-test]'
python3 -m playwright install chromium
```

自动验收分为三层。`tests/test_teacher_agent_ui_contract.py` 静态检查 HTML/CSS/JavaScript 的资源、可访问性标记、布局契约和请求字段；`scripts/run_teacher_agent_system_acceptance.py` 调用页面使用的同一组 loopback HTTP API，黑箱覆盖 bootstrap、start/resume/step、失败替换保留旧会话、成功替换、并行隔离和过期上下文拒绝，`tests/test_teacher_agent_dashboard.py` 另以直接状态检查和真实 loopback HTTP 覆盖 `api/attachment → api/step` 的绑定、幂等、消费和过期拒绝；`scripts/run_teacher_agent_browser_acceptance.py` 使用 Playwright 启动本机 Chrome/Chromium，覆盖高对比度印刷文字 image-only 回合，并在画像 B 替换前通过 loopback API 于 UI 外推进旧 Session，使页面第一次携带陈旧 `replace_expected_*` 收到一次预期 HTTP 400；runner 随后验证前端同步新 guards、换用新 start idempotency key、恰好重试一次并成功切换，同时继续核对画像隔离、旧手动 Skill 不继承、刷新恢复、表单回填、评估视图、390/768/1440 三种宽度、控制台和 capability 范围。预期 400 与对应浏览器资源错误单独计数，不混入非预期失败；runner 只输出聚合 receipt，不输出 capability URL、session handle、画像正文、学生文本、OCR 正文或附件句柄。这里的“真实浏览器”特指可复现的 Playwright DOM/网络交互，不把 Codex 内置浏览器是否可用或一次人工点击记录算作通过。视觉确认的两 Skill 轮换由 `tests/test_teacher_agent_live.py` 覆盖；loopback 层验证附件绑定、幂等、消费和过期拒绝。Chrome 用例只证明一条合成印刷文字成功路径与一次受控 stale replacement 恢复，不是任意并发、手写/公式 OCR 准确率、Firefox/Safari、屏幕阅读器、真实学生部署或学习效果证据。

### 8.3 3—5 分钟答辩脚本

1. 输入一个新的教学概念、目标和四维初始状态，点击开始；指出系统只输出第一个动作。
2. 可留空文字框，上传一张不含身份信息的高对比度印刷答案图；展示“本机 OCR—文字证据—当前回合”的链路，并主动说明原图不发 DeepSeek、公式必须核对。
3. 指向 Goal 计划、当前主/支持 Skill、选择理由和 `decision_origin`；确认本轮确实为 DeepSeek。
4. 输入一句包含明确错误规则的回答，提交；展示误解证据、状态更新以及 Skill 切换到纠错或理解检查。
5. 输入纠正后的回答；展示误解从 `active` 变为 `resolved`，下一关注点继续变化。
6. 用 `/+skill 名称` 展示一次人工覆盖，再用 `/auto` 恢复自动路由；说明覆盖有审计标记。
7. 展示 28 例开发 benchmark、固定单 Skill 对照和学习结果接口；主动说明它们分别是自由文本开发证据、机制回归和指标计算演示。
8. 若时间允许，用连续无进展回答或 `/stop` 展示安全转人工，而不是无限生成。

## 9. 交付物映射

| 类别 | 文件或入口 |
|---|---|
| 确定性状态机与结构化回归 | `teaching_skill_miner/teacher_agent.py` |
| DeepSeek 客户端 | `teaching_skill_miner/deepseek_client.py` |
| 自然语言实时 Agent | `teaching_skill_miner/teacher_agent_live.py` |
| Goal 与历史上下文 | `teaching_skill_miner/teacher_agent_context.py` |
| 本机答案图片 OCR 与证据包络 | `teaching_skill_miner/teacher_agent_vision.py`；macOS Apple Vision 优先，Tesseract 回退 |
| 本机服务与网页 | `teaching_skill_miner/teacher_agent_dashboard.py`、`teaching_skill_miner/web/teacher_agent_demo.*` |
| HTTP 黑箱系统验收 | `scripts/run_teacher_agent_system_acceptance.py`、`tests/test_teacher_agent_system_acceptance.py` |
| 真实 Chrome 验收 | `scripts/run_teacher_agent_browser_acceptance.py`、`tests/test_teacher_agent_browser_acceptance.py`；覆盖实际 DOM、一次 stale replacement 的 400→同步 guards→新幂等键→单次重试、刷新与响应式链路，不等于任意并发、跨浏览器或辅助技术认证 |
| v2 Skill Library | `data/teacher_agent_skill_library_v2.json` |
| neural-v1 运行边界 | `data/neural_v1_runtime_manifest.json` |
| 28 例 benchmark 与公开 receipt | `data/teacher_agent_free_text_benchmark.json`、`data/teacher_agent_free_text_benchmark_receipt.json` |
| 4 例结构化回归 | `data/teacher_agent_evaluation_cases.json` |
| 学习结果接口与 fixture | `teaching_skill_miner/teacher_agent_outcomes.py`、`data/teacher_agent_learning_outcome_demo.json` |
| Schema | `schema/teacher_agent_*.schema.json`、`schema/neural_v1_runtime_manifest.schema.json` |
| 自动测试 | `tests/test_deepseek_client.py`、`tests/test_teacher_agent*.py` |
| 研究依据与后续协议 | `docs/teacher_agent_references.md` |

## 10. 当前可以说什么，不能说什么

可以说：

> 本项目实现了以 DeepSeek V4 Flash 为语义诊断与 Skill 路由骨干、以显式状态和确定性约束为控制层、以服务端 Skill materializer 为执行层的实时多轮教学 Agent。它在每个真实请求—响应轮次中诊断一条学生文字回答，或由本机答案图片 OCR 得到的有界脱敏文字证据，更新学生状态与待确认候选画像，从 v2 的 13 个主 Skill 和 3 个支持 Skill 中选择、组合或切换，严格按选中 Skill 生成一个下一教学动作，并在成功、无进展或人工停止时终止。答案原图不发送给 DeepSeek。

必须同时补充：

- neural-v1 的运行本体仍为 provisional，证据物化门禁未通过；
- 28 例结果只是在作者构造且 post-hoc 的开发集上的单轮结果；
- 4 例结果只证明结构化控制机制按 fixture 工作；
- 学习结果 fixture 只证明评估接口存在；
- image-only Chrome 用例只证明合成印刷文字链路可运行；手写、复杂版面和公式 OCR 的部署准确率未建立，公式证据必须让学生核对；
- 自由文本诊断、专家教学质量、真实学习效果、跨 session 泛化和部署质量仍需独立、预注册或锁箱的外部验证。

完整验收口径见 [`teacher_agent_acceptance_matrix.md`](teacher_agent_acceptance_matrix.md)，研究参考与后续实验设计见 [`teacher_agent_references.md`](teacher_agent_references.md)。
