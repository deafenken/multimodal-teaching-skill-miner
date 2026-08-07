# Teaching Skill Mining and Evaluation System

题目二边界提醒：学生“确认识别文字”只确认转写，确认不等于答案正确。

本项目实现题目中的四阶段工程闭环：对教学视频联合分析语音转写、提问等待、关键帧、PPT/板书 OCR 与匿名课堂观察，从“教师如何教”中抽取 Teaching Skill，把 Skill 表示成另一个 Agent 可直接执行的 JSON 与状态机，最后进行自动评分、跨领域新任务测试和双人复核覆盖审计。

默认演示链路只使用 Python 3.10+ 标准库，不需要网络或 API。项目内置按 2 门课程组织、每门 5 讲的人工释义演示数据，一条命令即可复现工程闭环；它本身不是完整公开课字幕。项目另已对 10/10 MIT OCW 完整讲次完成官方 WebVTT 文件、来源页、字幕文件哈希、页面关联媒体与整段时间轴绑定，并在私有目录真实运行全程音频静音分析、均匀/场景抽帧、Tesseract OCR、板书/幻灯片变化、CLIP 视觉语义和事件融合。这里的字幕闭环只建立来源、文件身份和时间轴覆盖，没有逐字对听内容，也不建立字幕内容准确率或 WER。第三方视频、字幕正文、帧、OCR 文本和嵌入均不打进 wheel。当前准确状态是“完整视频多模态工程证据已闭环；真实双人复核、事件级识别准确率、确认性多模态增益、学习效果和外部部署验证待完成”，详见 [`docs/project_status.md`](docs/project_status.md)。

## 一键运行

```bash
python3 -m teaching_skill_miner demo --output artifacts
```

### 浏览器答辩看板

项目附带一个完全自包含、无需前端依赖或本地服务的交互式看板。它展示项目流程、合成多模态时间轴、TeachObs 四臂消融、DIPSER 0.9 证据边界、可执行 Skill 状态机和外部验证缺口：

```bash
python3 -m teaching_skill_miner dashboard
# 或安装后：
tsm dashboard
```

在无图形界面的服务器或 CI 中只物化 HTML、不打开浏览器：

```bash
tsm dashboard --no-browser --output artifacts/tsm_dashboard.html
tsm dashboard --check
```

看板只内嵌已经审核的聚合指标与 8 秒合成演示，不读取或打包真实私有视频、字幕正文、逐场景 OCR、标签、预测或嵌入。页面明确区分工程验收、探索性多模态增益和仍未建立的 WER、双人复核、部署准确率与学习效果。

### 题目二：实时自适应教学 Agent

题目二已实现逐轮运行的混合 Teaching Agent：`deepseek-v4-flash` 负责理解学生自由文本、判断与当前问题的对齐关系，并提议主/支持 Skill 与当前教师动作；确定性控制器负责显式状态、Skill 白名单、重复上限和终止安全。默认 `safe_generative` 执行模式不会一律丢弃模型话语：只有当模型动作与最终 Skill 的 `action_type`、单动作、提问性、问题契约和防答案泄露规则全部一致，且路由或 support 没有被控制器改写时，模型生成的安全话语才会进入学生界面。若只是动作候选不匹配、生产入口已启用修复且固定路由 eligibility 通过，系统最多追加一次只能重写当前动作的 action-only repair；未启用、不适用或修复仍失败时才使用服务端确定性 materializer。页面通过 `action_provenance.executor_origin` 区分 `deepseek_safe_generative`、`deepseek_action_only_repair` 与 `deterministic_materializer`。系统不会一次性预写多轮对话。每收到一条学生回答，它都会更新四维掌握、误解、当前理解信号、下一关注点和交互统计，再从 v2 Library 的 **13 个主 Skill + 3 个支持 Skill** 中重新选择、组合或切换，并公开理由。每个问题都附带服务端绑定的 `question_id + question_contract`（目标概念、可接受同义表达、回答类型和成功判据）；诊断先判断回答是否真正对齐当前问题，再决定状态和 Skill。相关但没有回答本问的内容会保守记为 `partial / related_but_not_answer`，不会仅因未命中预设词就写成“仍然困惑”，也不会凭一次相邻概念回答确认误解。页面把诊断来源分为原始 `deepseek_v4_flash`、经确定性契约修正的 `deepseek_v4_flash_constrained_by_deterministic_contract`、本问契约精确命中的 `active_question_contract_exact_match`、教师 `knowledge_spec` 精确命中的 `teacher_knowledge_spec_exact_match`，以及不属于自由文本识别结果的 `deterministic_safety_fallback`；不能把后四者笼统冒充未经修正的模型判断。每个通过约束层验证的 DeepSeek 回合还会在 `student_profile.adaptive_observations` 追加一条 `candidate_unconfirmed` 候选画像；候选不会覆盖教师提供字段，规则 fallback 不会生成候选；候选若没有绑定到学生原话，会明确记录 `no_grounded_excerpt` 并强制进入人工复核。在线模式支持 `/+skill 名称`、`/auto` 和 `/stop`：手动 Skill 从收到第一条学生回答后持续锁定，直到教师恢复自动模式，或适用信号、纠错证据、`max_repeat` / fallback 等安全门主动释放，不能借人工命令绕过契约。

生产 dashboard 通过 CLI 显式开启 `state_first_route_adjudication_enabled=true` 与 `action_only_repair_enabled=true`；DeepSeek 只提出候选，状态优先裁决器先按当前掌握维度、误解、参与度、无进展计数和 Skill 契约确定优先层，模型候选只在同一层内作 tie-break。`LiveAgentOptions` 的默认值仍保持关闭，供兼容性的库调用和单元测试使用；手动路由、视觉确认、已有安全 retarget 等路径不会再次进入该裁决器。Socratic 只有在学生已有实质主张、回答与当前问题对齐、理由或边界仍缺失且上一轮不是 Socratic 时才可优先，避免 `partial` 回合被它默认吸收。

#### 真实多步 Agent Loop：规划—工具—观察—路由就绪—动作

生产 Dashboard 的 CLI 同时显式开启 `agent_loop_enabled=true`。当前生产提示与路由契约版本为 **V14**，并沿用 state-first 路由裁决；这一路径不是把一段多轮对话预先写好，也不是一次 Prompt 的别名，而是一个有边界的模型—工具闭环：DeepSeek 先返回结构化规划，服务端执行本地工具，把工具结果放回下一次规划请求，直到得到 `route_ready` 或进入安全降级；`route_ready` 必须已经有经过本地校验的主 Skill 和 `next_focus`，随后才调用最终动作规划器生成本轮唯一教师动作。动作再进入既有的 Skill/action-type、问题契约、提问性、防答案泄露和 Session 版本门禁。

Loop 的模型工具权限是固定白名单，模型不能执行 Python、访问文件或自行修改学生记录：

| 工具 | 作用 |
|---|---|
| `inspect_student_state` | 读取有界的掌握、误解、理解信号和下一重点 |
| `inspect_recent_history` | 读取最近有限回合，避免把整段原始历史发给模型 |
| `search_skills` | 按目标、信号和动作类型检索候选 Skill |
| `select_skills` | 提交一个主 Skill 与有界支持 Skill 组合，交由服务端校验 |
| `set_next_focus` | 设置 `prerequisite / conceptual / procedural / transfer` 之一 |
| `evaluate_termination` | 按阈值、最新正确性和活跃误解检查是否具备成功终止条件 |

当前生产上限为每回合最多 6 个 Loop 步骤、每步最多 3 个工具调用、同一调用最多重复 2 次，模型边界错误最多重试 1 次；上限耗尽、重复调用、未知工具、非法参数或模型 JSON 错误都会记录可审计事件并触发确定性安全 fallback。fallback 不冒充 DeepSeek：它仍使用同一 Skill 合同和安全门；没有可执行的安全 Skill 时停止并转人工。公共 `last_agent_loop` receipt 只保留步骤、工具名、状态、计数、短原因、响应元数据和消息哈希，不保存 Prompt、思维链、学生原文、原图或 API Key。

显式传入 `--session-store` 时，包含 Loop receipt 的会话 checkpoint 会写入本机 append-only JSONL：每行有连续 `seq`、`previous_hash` 和自身 SHA-256，`turn_started → turn_committed/turn_aborted` 由恢复器重放；崩溃遗留的 started turn 会补记为 aborted，旧幂等键、画像版本或运行政策不匹配会 fail-closed。哈希链只提供篡改检测，不是加密、登录、访问控制或跨设备同步，store 仍是需要权限保护的私有学生数据。

这套编排借鉴了可靠 Agent 常见的 bounded loop、工具 allowlist、幂等/重试和事件审计模式，称为 **Codex-inspired reliability architecture**；项目没有复制或声称等价于 Codex、Claude Code 的内部实现、Shell、Git、沙箱或多 Agent 能力。直接使用库级 `LiveAgentOptions()` 时 Loop 默认关闭以保持兼容；`tsm teacher-agent-dashboard --agent-backend deepseek` 的生产入口会打开它。

本机真实验收已用配置的 `deepseek-v4-flash` 完成健康检查（DeepSeek 返回 HTTP 200）和两轮教学：首轮 Loop 为 `route_ready`（6 步、8 次工具调用、0 次 fallback），下一轮为 `route_ready`（5 步、7 次工具调用），并观察到 Skill 从诊断提问切换到苏格拉底理解检查。loopback 浏览器验收还覆盖了画像切换后继续作答、刷新恢复当前动作和 Loop 摘要、显式 `/stop` 终止后刷新保持终止，以及页面无项目自身错误日志。它们证明的是本机工程链路和真实 API 可运行，不是开放学生群体的教学质量、学习增益、跨 session 泛化或部署准确率。

忙碌回合中的“停止生成”是可恢复的 `cancel_turn`：它只失效当前活动回合、保留 Session，迟到的 DeepSeek 响应会被 commit fence 丢弃；`transport_cancellation_supported=false`，远程 HTTP 请求本身不支持传输层取消，供应商侧仍可能完成并计费。教师显式发送 `/stop` 才会把整个 Session 置为 terminal，二者在页面和审计记录中分开显示。

对“第二种呢”“回到第 1 轮”“按最开始的方式讲”“按约定继续”等明确连续性提示，服务端会确定性生成 `continuity_recall`。`resolved_evidence_linked` 只能按目标摘录和 `evidence_refs` 接续；`unresolved_no_matching_evidence` 必须承认没有找到匹配记录并请学生重述。动作修复请求只接收有界、脱敏、证据链接的 continuity constraints，不得另写一份历史。

每个主 Skill 的 `supporting_skill_ids` 是该主 Skill 自己声明的 **硬组合 allowlist**，不是给模型参考的推荐列表：未声明的 support 即使存在于 Library 中也不能组合；已声明的 support 仍须是 `support` 角色，并通过自身的适用条件、局部前置条件和 `max_repeat`。主 Skill 与规则 fallback 共用机器门禁，覆盖 Skill ID/角色、`applicable_signals`、纠错证据、若干会导致阶段越级的高风险前置条件、目标材料是否真实存在、主/支持 Skill 重复上限、主 Skill 的 support allowlist 以及 support 自身门禁；没有任何安全 Skill 可执行时会停止自动推进并转人工。其余写在 `preconditions / contraindications / postconditions / failure_transition` 中的自然语言合同仍会进入提示词与审计，但系统不声称已经理解并自动证明了全部自然语言条件。

学生本轮可以提交文字、答案图片或两者。答案原图只在本机内存和临时目录中短暂处理：受支持的 macOS 主机优先使用 Apple Vision，本机还可对原图、灰度增强和高对比度二值图运行多条 Tesseract 路由；其他平台在安装 Tesseract 后使用这些本机路由。本机只把有界 OCR 观察写入教学上下文，并在发送前经过与学生文字相同的常见直接标识符模式替换；DeepSeek 不接收原图。这是“本地 OCR 后把文字证据交给文本模型”，不是 DeepSeek 原生视觉理解，也不是实时视频输入。高置信、无实质冲突且由多引擎或足够多独立预处理路线一致支持的公式，可建立 `formula_transcription_established=true`，即“本机对公式转写有交叉佐证”；`formula_accuracy_established` 仍固定为 `false`，因为 OCR 一致不等于数学正确。只有可靠转写与当前 `question_contract` 的可判定短概念/公式答案，或教师提供 `knowledge_spec` 中**非空 `knowledge_components` 与当前动作知识点相交**的规范陈述精确匹配时，控制器才可确定性判对；没有知识点绑定的 claim 只能作为模型参考，开放题相关词或评分 rubric 片段也不会自动判对。多行 OCR 必须按完整答案匹配，不会把其中一行单独当作整段答案；若完整命中的是另一个知识点的教师规范陈述，服务端会强制降级为 `partial / related_but_not_answer`，不增加掌握度。网页编辑器支持 `知识点::标准陈述` 语法，保留原有 evidence/criterion 绑定。低置信、无文字、OCR 路线冲突、键入文本与 OCR 冲突，或公式缺少上述转写佐证时，即使学生同时输入“答案见图”“照片这样显然是对的”，也不能绕过确认：该轮强制记为 `partial / ambiguous`、置信度 `0.0`，不增加掌握、不解除误解，并清空未执行的 support。确认动作只允许在 `skill_self_explanation` 与 `skill_socratic_understanding_check` 中选择；控制器优先使用自我解释，达到其连续 `max_repeat` 后切换到苏格拉底理解检查，后续继续按各自门禁和重复上限轮换。手写、复杂版面、公式 OCR 和答案判分的部署准确率仍未建立。

每次模型请求只发送一个经过校验的 `teaching_context`：固定教学目标与教师画像、当前 Goal 计划、当前问题契约、近期逐轮工作记忆、较早历史的确定性统计与证据关联检查点、知识/误解状态、未确认的低权重画像假设，以及对应证据指针。较早历史的 `teaching_checkpoints` 最多保留 6 条，只从被省略回合中的明确师生原话或已记录结构化信号抽取；另有可重放的 `teaching_memory` 只保留学生明确偏好、未解决问题、教师承诺和“第二种”等指代对象，每一项都绑定原始回合证据，不调用模型编写叙事摘要。它们是选择性审计事实，不是完整语义总结；未知的完成或解决状态保持未知，系统明确记录 `omitted_turn_semantics_are_exhaustive=false`。网页“既往上下文（未标注）”按行写入 `background_history`，在上下文中标为 `teacher_provided_unlabeled_background`；它不携带 `signal` 或 `focus_dimension`，不会直接改变初始掌握、当前理解信号或误解，只有之后提交的真实学生回答才触发学情更新。当前回答只出现一次；默认保留最多 10 个相关回合并受 14,000 字符硬上限约束。未显式填写知识点时，直接使用教师输入的教学概念作为最小检索锚点；多知识点目标则只给每轮动作标注当前实际知识点。合法超长会话会逐层裁剪成仍保留当前问题和回答的可验证最小上下文，而不是越过预算或让整轮崩溃。

题目二网页采用原创的 Codex-inspired 学习工作台：左侧是任务与三种合成学生画像，中间只保留实时对话和固定输入框，右侧用“学情 / 方法 / 证据”三页检查器解释状态、Skill 与上下文；冻结 benchmark、基线和学习结果接口移入独立的“实验 / 评估”视图。三张 AI 合成头像不对应真实学生，也不参与能力判断；环形图只表示四项掌握估计的等权平均。参考官方开源 Codex 固定版本 `15ea598c6e7e0914a7ae8c881ac05dacea2f7902` 的仅是 thread/turn identity、预期轮次核验、陈旧异步结果隔离、追加式事件和 cold-resume 可靠性模式，不是品牌、UI、Shell、Git、沙箱或多 Agent 能力；当前网页和教学状态机仍是本项目原创实现。画像切换采用 prepare-then-commit：先完整生成并校验新 Session，再原子提交并退休旧 Session；替换请求还必须匹配旧会话的 `round + question_id + context_version + profile_revision`。前端应用异步响应前也核对请求 epoch、`session_id`、`profile_revision` 和不倒退的 `context_version`，防止慢响应把新画像或新回合覆盖成旧快照。start、step 和 command 都使用独立幂等键；step/command 同时绑定 `session_id + expected_round + expected_question_id + expected_context_version + profile_revision`，过期、跨画像或冲突重放均 fail-closed。默认仍只在进程内保存最多 16 个隔离 Session；显式传入 `--session-store <local.jsonl>` 后，服务才启用本机追加式 JSONL cold resume。每条事件具有连续序号、前一事件哈希和自身 SHA-256；学生 step 先写 `turn_started`，完成后写 `turn_committed`，异常写 `turn_aborted`，崩溃遗留的 started turn 在重启时被记为 aborted，旧幂等键不能伪装成已提交。恢复还必须精确匹配无密钥 `runtime_policy_contract`（provider、model、endpoint、prompt、上下文预算、fallback 与 action executor），并验证持久化 Skill 是当前 Library 的内容等价、有序主 Skill 子集且保留全部 support；漂移或篡改均 fail-closed。候选画像和对话不会自动合并到另一个 session。

先把外置盘密钥链接到本机私有目录；`.private/` 已被 Git 忽略：

```bash
mkdir -p .private
ln -s "/path/to/deepseek_api.txt" .private/deepseek_api.txt
```

macOS 可双击根目录的 **`打开题目二教学Agent.command`**，也可运行：

```bash
tsm teacher-agent-dashboard \
  --agent-backend deepseek \
  --model deepseek-v4-flash \
  --api-key-file .private/deepseek_api.txt \
  --allow-remote-student-data

# 可选：显式启用本机 cold resume。该文件含会话正文与状态，必须按私有数据保护。
tsm teacher-agent-dashboard \
  --agent-backend deepseek \
  --model deepseek-v4-flash \
  --api-key-file .private/deepseek_api.txt \
  --allow-remote-student-data \
  --session-store .private/teacher-agent-session.jsonl

# 不调用 API 的模板自检 / 规则基线
tsm teacher-agent-dashboard --check
tsm teacher-agent-dashboard --agent-backend deterministic

# 首次运行真实浏览器验收：安装轻量测试 extra 与 Playwright Chromium
python3 -m pip install -e '.[browser-test]'
python3 -m playwright install chromium

# 先启动上面的本机服务，再把它打印的 127.0.0.1 capability URL 传入。
# 使用下载的 Chromium；若本机已装 Google Chrome，可改成 --browser chrome。
python3 scripts/run_teacher_agent_browser_acceptance.py \
  --base-url '<loopback-capability-url>' \
  --browser chromium \
  --acknowledge-remote-demo-text

# 不调用远程模型的真实 Chrome 交互验收：阻塞一轮合成模型请求，
# 点击“停止生成”，再验证会话仍可继续；只使用本机内存合成数据。
python3 scripts/run_teacher_agent_cancel_browser_acceptance.py \
  --browser chrome
```

双击启动脚本可设置 `DEEPSEEK_API_KEY_FILE`；直接运行 `tsm` 时可设置 `TSM_DEEPSEEK_API_KEY_FILE`，两者都只应指向本机私有密钥文件。网页只绑定 `127.0.0.1`，使用随机 capability URL、CSP 和 `no-store`。不传 `--session-store` 时，教学内容只在服务端内存，进程停止后全部 Session 丢失；显式传入该参数后，目标、画像、对话、状态、幂等缓存以及尚存附件的受限 OCR 证据会写入指定本机 JSONL，供同一 Session 冷恢复。原始图片仍不持久化，也不发送给 DeepSeek。该 JSONL 是含学生数据的私有文件，哈希链只能检测篡改，不能提供加密、登录身份、跨设备同步或访问控制，操作者必须自行限制文件权限、备份和删除范围。浏览器 `sessionStorage` 只保存一个随机、无业务语义但可被同源 JavaScript 或 DevTools 读取的 opaque 会话句柄，`localStorage` 仅保存主题与密度等非敏感显示偏好。在线模式会把经过上下文最小化和常见直接标识符模式替换的必要教学文本发送给 DeepSeek；若学生提交答案图片，远程请求中只增加本机 OCR 生成、长度受限并经过同一模式替换的文字证据，原图、缩略图、本机路径和临时文件均不发送。声明 `contains_direct_identity=true` 或给出非 JSON 布尔值的画像会在任何远程请求前被拒绝。模式替换仍无法证明自然语言姓名、学校、普通学号或 OCR 误识别出的身份信息已全部移除，因此隐私 trace 固定保留 `raw_identity_fields_sent=not_established` 与 `residual_identity_risk=true`；不能把页面表述为“所有处理完全离线”或“完全匿名”。若远程调用或模型校验失败，页面会把规则降级明确标为 `deterministic_safety_fallback`，不得冒充 DeepSeek 结果；fallback 仍须通过与在线计划相同的 Skill ID/角色、适用信号、高风险前置条件、材料、重复上限与 support 组合门禁；无安全 Skill 时停止并转人工。仓库保留三层不同验收：静态 UI 合同、复用浏览器 API 的 HTTP 黑箱 runner，以及 Playwright 驱动本机 Chrome/Chromium 的真实 DOM runner。真实浏览器层包含一条输入框留空的 image-only 回合，并主动制造一次画像替换竞态：页面仍持有旧 `replace_expected_*` 时，runner 通过同一 loopback API 在 UI 外先推进旧 Session；第一次替换按预期返回一次 HTTP 400，前端随后调用 `api/session` 同步权威 round/question/context/profile guards，丢弃旧 start 幂等键、生成新键并恰好重试一次，最终成功切换到画像 B。runner 同时检查独立掌握状态、旧手动 Skill 不继承、opaque handle 刷新恢复、画像表单回填、评估视图和 390/768/1440 三种宽度；这次预期 400 单独计数，不冒充零错误路径。这里记录的是可复现 runner 的结果，不把 Codex 内置浏览器是否可用或一次人工点击当作验收证据。该自动样例只证明这一条印刷文字 Chrome 与一次受控陈旧替换恢复路径；低置信/冲突确认与可靠公式转写后的精确契约匹配另由单元测试覆盖，loopback 测试验证附件绑定、幂等、消费与过期拒绝，不建立手写/公式 OCR 准确率、Firefox/Safari 兼容、屏幕阅读器认证、真实学生部署或学习效果。

另有一个不需要 DeepSeek 的浏览器级取消验收：`run_teacher_agent_cancel_browser_acceptance.py` 注入阻塞的合成模型调用，实际在 Chrome DOM 中检查忙碌时“停止生成”可见、`cancel_turn` 不结束 Session、迟到响应被 commit fence 丢弃、输入草稿保留，以及取消后下一轮仍能提交。该 receipt 是工程交互证据，不是模型质量或在线语义准确率证据；取消请求产生的预期 HTTP 400 会单独计数。

当前有四类互相独立的评估证据：

- 28 个作者构造的一轮自由文本开发案例，最新在线运行使用与 live question-contract 共享的 v3 诊断 taxonomy / 语义量表；信号 Accuracy / Macro-F1 为 **0.892857 / 0.875325**，允许主 Skill 命中率为 **0.750000**。benchmark prompt 并非完整 live Session prompt，因此只验证单轮诊断与路由，不等同完整 live Session 评测；该数据未经专家复核、不是提示词开发后的锁箱集，只能称 post-hoc development regression。
- 4 条结构化合成轨迹与固定 `skill_stepwise_scaffolding` 基线的机制回归：状态一致率 1.000000、允许决策匹配率 0.916667、终止匹配率 1.000000；自适应/固定内部模拟平均增益为 37.333250/20.416750。它不检验自由文本理解，也不是实际学习效果。
- 新增 20 个作者构造多轮 episode、共 65 个学生回答回合和 1 次画像替换操作的对抗开发 benchmark，覆盖长期记忆、图片证据、知识纠错、Skill 切换、画像替换隔离、终止与恢复。runner 会把所有 gold 字段排除在模型请求之外；paired 模式保留兼容名称 `current` 作为 `deterministic_legacy` 对照臂，以 `safe_generative_executor` 表示生产默认的 integrated `safe_generative` 候选臂。为保持 paired 评分的请求拓扑可解释，benchmark executor 当前沿用 `LiveAgentOptions` 的兼容默认 `agent_loop_enabled=false`，所以其每轮先发起 1 次诊断、路由与动作候选 plan 请求；这不是生产 Dashboard 的多步 Agent Loop 在线结果。只有候选动作不满足最终路由/动作契约、已启用 action-only repair 且有界 eligibility 门禁通过时，候选臂才最多追加 1 次 fixed-route action-only repair。repair 只能重写本轮 `teacher_action`，不能改变 diagnosis、primary/support Skill、termination 或 route；修复失败继续使用已经通过控制器构造的确定性动作。传输层重试属于同一次请求，不增加教学动作。报告分别记录调用拓扑、生产 `action_provenance`、`validated_model_plan_count_delta` 和端到端 turn latency，不能把 repair 计成第二个 validated plan，也不能只看 latency 推断模型调用次数。当前公开文档只确认 fixture、Schema、盲化载荷和评分管线可复现，不填报尚未完成或尚未审核的在线数值。该集合未经专家复核、不是提示词开发后的锁箱集，也不建立完整 live Session 质量、跨 session 或部署准确率。
- `teacher-agent-outcome-evaluate` 接受前测、后测、迁移测和可选延迟测；随附 0.4→0.8 的记录是作者构造 fixture，只验证计算接口。

多轮 benchmark 的请求统计采用 `request_accounting_scope=completed_committed_turns_only`：`validated_plan_request_total` 只计通过校验的主 plan，`action_repair_request_total` 计实际发起的 action-only repair，`logical_model_request_total` 是两者之和；repair 不是第二个 validated plan，但确实是第二次 logical request。报告还给出三项每完成回合均值、`action_repair_request_turn_count`、`action_repair_adopted_turn_count` 和 `action_repair_adoption_rate`。失败、取消或未提交的回合不进入请求分母，另由完成/失败计数报告。

neural-v1 的九环节/十三策略本体已用于组织 v2 Skill，但其公开 manifest 仍为 `provisional`：证据物化 gate `passed=false`、可物化预测 0/54，不能称为已确认课堂共识。完整方法、运行命令、结果表和声明边界见 [`docs/teacher_agent_task2.md`](docs/teacher_agent_task2.md)；逐项验收见 [`docs/teacher_agent_acceptance_matrix.md`](docs/teacher_agent_acceptance_matrix.md)；现场逐屏讲稿见 [`docs/teacher_agent_defense_guide.md`](docs/teacher_agent_defense_guide.md)；研究依据见 [`docs/teacher_agent_references.md`](docs/teacher_agent_references.md)。

### 本机双成果真实演示

macOS 下可直接双击仓库根目录的 **`打开双成果Demo.command`**：启动器会自动定位项目与两组私有产物，仅在 `127.0.0.1` 启动服务并打开浏览器。演示结束后回到终端窗口按 `Ctrl+C` 停止服务。如 macOS 首次拦截，右键该文件选择“打开”即可。

需要在答辩现场同时展示“真实课堂行为识别”和“完整视频 Skill 蒸馏”时，使用与公开看板分离的本机入口。顶部可在两项成果之间切换：TeachObs 页把完整视频、时间窗字幕、教师行为标签和冻结四臂预测对齐；MIT 页展示 10 个完整讲次生成的真实九环节 Skill、`evi_*` 字幕证据、脱敏后的 `mme_*` 候选事件和达标/回退 Runtime。

```bash
# 1. 只检查通用页面模板，不读取任何私有数据
tsm dashboard-real --check-template

# 2. 同时检查 TeachObs 识别证据与 MIT Skill 产物；不启动服务
tsm dashboard-real \
  --teachobs-root artifacts/private/external_datasets/teachobs \
  --skill-root artifacts/private/full_multimodal \
  --initial-lesson S24 \
  --initial-skill linear_algebra_l03 \
  --check-data

# 3. 检查通过后启动本机回放；端口 0 表示随机选择空闲端口
tsm dashboard-real \
  --teachobs-root artifacts/private/external_datasets/teachobs \
  --skill-root artifacts/private/full_multimodal \
  --initial-lesson S24 \
  --initial-skill linear_algebra_l03 \
  --port 0
```

`--teachobs-root` 和 `--skill-root` 分别默认为上述两个私有目录；默认打开 `S24` 和 `linear_algebra_l03`。需要手动打开页面时可增加 `--no-browser`，再使用终端打印的 capability URL。服务只绑定 `127.0.0.1`，每次启动生成随机 capability token，并对页面、数据与媒体响应设置 `Cache-Control: no-store`。真实视频、字幕、标签和逐样本预测不会写入静态 HTML；Skill 原始 JSON 中的本机路径、URL、OCR 原文和作业目录也不会发送给浏览器。完整参数见 `tsm dashboard-real --help`，准备步骤和安全检查见 [`docs/private_local_dashboard.md`](docs/private_local_dashboard.md)。

这里的“真实”表示两页都读取已完成的离线产物，而不是合成示例。TeachObs 页在启动时验证哈希、顺序和冻结模型绑定，并在内存重算四臂逐场景预测；学生动作因没有逐场景真值、角色跟踪或姿态模型而明确留空。MIT 页验证 10 份 Full Skill 与语义 manifest、消融报告和内部评估指纹：90 个步骤中 62 个是有 `evi_*` 支持的 `observed_method`，28 个是 `recommended_enrichment`；59 条入选 `mme_*` 目前辅助策略计分与证据一致性，procedure 的直接 `mme_*` 引用数为 0。页面还可把新教学主题注入当前 Skill，现场生成九步教学过程，并展示哈希绑定的七维内部评估与六项硬门槛。因此两项成果是互补的独立验证轨，尚不是“TeachObs 预测直接喂给 MIT Skill”的端到端系统。页面不是实时部署；内部 Skill 量表不是 Accuracy，也没有建立专家 Skill 金标准或学习效果。`tsm dashboard` 仍是唯一面向公开分发的聚合看板；`tsm dashboard-real` 及其读取的私有输入不得上传 GitHub、放入 `artifacts/public/` 或打进 wheel。

先分别检查环境能力与内置核心工程证据；这两条命令不替代后文的测试、exact wheel、CI、tracked-file 隐私或外部研究验收：

```bash
python3 -m teaching_skill_miner doctor
python3 -m teaching_skill_miner verify-delivery \
  --output artifacts/delivery_verification.json \
  --markdown artifacts/DELIVERY_VERIFICATION.md
```

运行后生成：

- `artifacts/skills/`：10 个 Teaching Skill，每个视频至少 1 个；
- `artifacts/evaluations/`：逐 Skill 自动评估；
- `artifacts/summary.json` 与 `summary.md`：课程/视频/策略覆盖和通过率；
- `artifacts/teaching_demo.md`：把 Skill 迁移到“二分查找”后生成的教学过程；
- `artifacts/interactive_demo.json`：学生未达标时触发回退、达标后推进的完整状态轨迹；
- `artifacts/transfer_benchmark.json`：6 个留出新主题及静态无 Skill 基线对比；
- `artifacts/data_audit.json`：数据完整性和正式研究就绪状态；
- `artifacts/multimodal_demo/`：真实运行的合成多模态管线演示与独立 fixture 标注报告；
- `artifacts/human_review.csv`：两名独立复核者的录入模板；新生成的模板每行用 `skill_fingerprint` 绑定完整 canonical Skill 内容，已有旧 CSV 不会被自动覆盖；空模板明确为 `incomplete`，不算人工验证结果；正式实验建议在条件允许时采用盲法；
- `artifacts/human_review_status.json`：对 10 个预期 Skill 的覆盖缺口、重复复核者和缺失项审计；空模板不会被写成 10/10 已完成。

### 跨讲次通用 Teaching Skill

单讲 Skill 只说明“一位教师在这一讲里怎样教”。要把它用于下一道新题，必须再做一次跨讲次聚合，而不能任选一讲冒充通用方法。下面的命令读取两门课程各五讲的 Full Skill，以“每讲最多一票”统计策略和九个教学环节；只有总讲次支持率不低于 80%，且每门课支持率都不低于 60% 的项目，才标为 `cross_lecture_observed_consensus`。未过门槛的环节仍可作为可执行脚手架，但必须标为 `recommended_enrichment`，不会计入观察共识：

```bash
python3 -m teaching_skill_miner distill-general-skill \
  --skill-root artifacts/private/full_multimodal/ablation/skills \
  --output-dir output/general_skill_v0 \
  --example-concept "动态规划"

python3 -m teaching_skill_miner apply-general-skill \
  --skill output/general_skill_v0/general_skill.json \
  --concept "新的教学主题" \
  --learner-level beginner \
  --output output/general_skill_v0/next_teaching_process.md

python3 -m teaching_skill_miner validate general-skill \
  output/general_skill_v0/general_skill.json
```

输出包括 `general_skill.json`、独立结构/支持评估、哈希 receipt 和一份可直接讲授的新主题教学过程。正式十讲当前产生 6 个跨课程共识策略；九个执行环节中 5 个达到跨课程观察门槛，另外 4 个明确保留为规范补充。该产物是可复现、可追溯的 `heuristic_provisional` 通用候选，不是人工专家共识，也没有证明教学效果。

当前可运行的 v0 使用已有字幕线索和多模态候选事件生成单讲 Skill，再做课程平衡聚合。真正可训练的神经 v1 已完整设计为 VideoMAE / WavLM / XLM-R / LayoutLMv3 编码、跨模态注意力、长时序建模、事件/阶段/策略联合预测和受监督证据指针；它需要新增双人 phase/strategy/evidence gold 后训练与锁箱验证，尚不能声称已经训练完成。方法、损失函数、数据划分、消融和否证标准见 [`docs/end_to_end_multimodal_general_skill.md`](docs/end_to_end_multimodal_general_skill.md)。

完整本地工程验收（全量测试、Ruff、编译/脚本语法、依赖一致性、schema、公开目录审计、双次可复现 wheel 构建、仓库外安装和 exact-wheel 视频闭环）：

```bash
python3 -m pip install -e '.[recognition,dev]'
sh scripts/verify_project.sh
```

要复现本次 macOS arm64 / Python 3.13.11 的精确项目依赖闭包，可使用：

```bash
python3 -m pip install -c constraints/validated-py313-macos-arm64.txt \
  -r requirements-dev.txt
```

CI 的 Python 3.10–3.13 矩阵用于检验声明版本范围的前向兼容性，故有意不套用这个单平台快照；正式归档运行应另外保存平台对应的完整约束文件和构建产物 SHA-256。

正式发布时使用干净、可复现构建，并直接验收将要分发的同一个 wheel：

```bash
sh scripts/build_release_acceptance.sh
```

若本机存在 TeachObs 私有证据，`verify_project.sh` 会一并重验人工复核骨架与锁箱草案。它们的默认路径是未带日期的文件名，而本仓库随附的是与论文 profile 配对的带日期版本，因此需要显式指向：

```bash
TSM_TEACHOBS_HUMAN_ROOT="$PWD/artifacts/private/external_datasets/teachobs/human_annotation_paper_track1_20260723" \
TSM_TEACHOBS_HUMAN_RECEIPT="$PWD/artifacts/public/teachobs_human_annotation_paper_track1_20260723_receipt.json" \
TSM_TEACHOBS_LOCKBOX_DRAFT="$PWD/artifacts/public/teachobs_new_site_paper_track1_lockbox_preregistration_20260723_draft.json" \
sh scripts/build_release_acceptance.sh
```

不设这三个变量时，验收会在 TeachObs 阶段以"human-review evidence is partial"失败关闭——这是刻意的：`human_annotation/` 下还留着 5,158 行的完整 profile 旧骨架，与 4,945 行论文 profile 的公开 receipt 并不配对，宁可失败也不允许用不匹配的骨架通过。没有 TeachObs 私有证据的环境（例如公开 CI）会跳过整段检查，直接裸跑即可。

这个入口启动时先把旧 acceptance 降级为 `stale_not_accepted`，随后依次执行全量项目验收、两个独立临时源码副本的字节级一致构建、最终候选 wheel 的隔离安装和视频闭环、wheel/公开目录发布审计；候选通过后才原子替换 `dist/` 中的同名 wheel，再对新 acceptance 本身做发布审计并原子替换 `artifacts/release_acceptance_1.2.0.json`。因此中途失败不会遗留看似仍有效的旧验收。acceptance 的测试数、wheel 哈希/大小/成员、公开目录摘要都由绑定同一 wheel SHA-256 的新鲜 receipt 重算；完整包源码、看板、验证工具、README/治理与研究文档，以及显式纳入验收的公开/私有证据只要在验收后变化，当前 receipt 就会失效。未纳入发布范围的任意本地研究产物不在此概括内。为保证相同证据生成相同字节，acceptance 有意不写墙钟时间。

如需逐步排障，底层入口仍可单独运行：

```bash
sh scripts/build_release_wheel.sh dist
sh scripts/verify_release_wheel.sh \
  dist/teaching_skill_miner-1.2.0-py3-none-any.whl
```

构建脚本固定 `SOURCE_DATE_EPOCH`；最终发布仍须由人检查研究边界和精确候选，自动 acceptance 只建立工程验收事实，不能建立双人标注有效性、多模态增益、部署准确率或学习效果。

在真实 Git checkout 中还应运行 tracked-file 隐私扫描：

```bash
python3 scripts/audit_repository_privacy.py
```

该扫描依赖 `git ls-files`；当前若只是无 `.git` 的目录副本，只能确认 CI 配置存在，不能声称已验证远端实际跟踪文件边界。

也可以安装核心命令行入口（无需识别依赖）：

```bash
python3 -m pip install .
tsm demo --output artifacts
```

## 系统流程

```text
视频 ──► ASR / 字幕 ─────────────┐
  ├──► 音频停顿与提问等待 ───────┤
  ├──► 场景关键帧 / OCR / 板书变化 ┤
  └──► 匿名课堂观察 ──────────────┘
                     │
                     ▼
              多模态时间轴事件
        │
        ▼
方法蒸馏：目标 / Bloom 层级 / 策略 / 动作 / 证据
        │
        ▼
可执行 Skill：触发条件 → 步骤 → 学生信号 → fallback → 验证
        │
        ├────────► 新主题教学过程 / 交互状态机
        │
        ▼
自动评估：结构 + 证据 + 可执行性 + 方法忠实度 + 教学质量 + 留出迁移 + 溯源
        │
        ▼
协议工具（尚未执行）：双人独立复核 / 外部锁箱 / 学习效果 A/B
```

流程图最后一行表示仓库已提供任务模板、指纹与覆盖审计、锁箱登记/签名/一次性消费校验和学习效果分析骨架，不表示这些外部研究已经执行。当前没有真实双人评分、被外部治理方消费的目标站点锁箱或真实学习者 A/B 结果。

### 1. 视频采集与预处理

`preprocess` 支持 `.srt`、`.vtt`、`.txt`，以及 `.mp4/.mkv/.mov/.webm/.mp3/.wav/.m4a`。媒体文件模式会调用本机 `ffmpeg` 与 OpenAI `openai-whisper` 包提供的 `whisper` CLI；字幕和文本模式不依赖外部程序。

```bash
python3 -m teaching_skill_miner preprocess new_lesson.srt \
  --video-id new_001 \
  --course-id demo_course \
  --title "New lesson" \
  --source-url "https://example.org/lesson" \
  --language en \
  --output data/new_001.json
```

本地视频的命令相同，只需把输入换成视频路径。兼容 OpenAI Whisper 参数接口的 CLI 可通过 `TSM_WHISPER` 指定，`ffmpeg` 可通过 `TSM_FFMPEG` 指定。当前实现不兼容 `whisper.cpp` 的 `whisper-cli` / `main` 参数和模型文件接口；`tsm doctor` 会把它识别为不同 CLI，而不会误报 ASR 已就绪。

### 多模态视频分析

当转写已存在时，可以直接把视频与转写对齐，无需重复 ASR。外部提供的带时间戳文本标为 `transcript`；只有 ASR provenance 哈希与当前媒体一致时才标为 `speech`，同时历史兼容字段 `audio_content_verified` 为 `true`。这个字段只表示“转写来自同一媒体哈希绑定的 ASR 运行”，不表示人工逐字内容核对，也不建立内容准确率或 WER：

```bash
python3 -m teaching_skill_miner multimodal \
  --video lesson.mp4 \
  --transcript lesson.transcript.json \
  --artifacts-dir artifacts/lesson_multimodal \
  --frame-interval 20 \
  --max-frames 48 \
  --output artifacts/lesson_multimodal/enriched_transcript.json
```

输出包含：

- `question_and_wait`：语音问题与后续静音联合证据；
- `scene_change` / `slide_change`：镜头或 PPT 变化；
- `board_build_up`：OCR 文字稳定重合并逐步增加；
- `code_formula_walkthrough`：画面代码/公式与讲解时间对齐；
- `student_confusion` / `teacher_adjustment`：经授权的匿名课堂观察输入，不冒充视频自动识别结果。

#### 10 讲完整视频多模态实跑

正式链路不是短视频 fixture。它下载并逐文件验证 MIT 18.06 前 5 讲和 MIT 6.0001 前 5 讲的完整 MP4，再按 5 分钟 chunk 可恢复处理；每个 chunk 做全段音频静音检测、15 秒均匀抽帧、场景变化候选、OCR 和像素变化分析，最后将官方字幕、音频活动、视觉证据和事件统一到同一时间轴。原始媒体与逐帧结果只保存在 `artifacts/private/`。

在已经获取正式字幕的前提下，可按下面的可恢复命令复现完整链路。CLIP 是可选的 `visual` extra；核心 wheel 不自动安装这组重依赖，也不自动下载或重新分发第三方模型权重：

```bash
python3 -m pip install 'teaching-skill-miner[visual]'
```

准备好指定 revision 的本地 CLIP snapshot 后，一条命令可串联正式字幕、10 个完整视频、长视频音频/抽帧/OCR/事件、CLIP、审计、四臂消融、公开 receipts 和公开目录审计：

```bash
scripts/run_full_video_multimodal_study.sh \
  artifacts/private/models/openai_clip_vit_b32_fp16_3d74acf9 \
  --acknowledge-source-terms \
  --reuse-downloads
```

`--reuse-downloads` 要求正式字幕和完整媒体 manifest 已存在，并跳过两个网络获取阶段；去掉它则从正式字幕获取和 10 视频下载/验证开始。长视频阶段复用经输入、配置和输出校验的 chunk checkpoint，因此中断后可恢复。视觉推理默认 CPU、batch size 16；获准在本地或授权计算环境使用 GPU 时可设置 `TSM_VISUAL_DEVICE=cuda:0`，并用 `TSM_VISUAL_BATCH_SIZE` 调整 batch。无论 CPU 还是 GPU，产物状态 `complete_hash_bound_inference` 都只表示哈希绑定推理完成，不表示识别准确率。

总 runner 内部等价于下面的分阶段命令；需要诊断或只重跑某一阶段时可逐条执行：

```bash
python3 -m teaching_skill_miner fetch-full-videos \
  --source-manifest data/formal_caption_sources.json \
  --formal-caption-manifest artifacts/private/formal_captions/dataset_manifest.json \
  --output artifacts/private/full_videos \
  --public-receipt artifacts/public/full_video_validation_receipt.json \
  --acknowledge-source-terms

python3 -m teaching_skill_miner multimodal-longform-dataset \
  --media-manifest artifacts/private/full_videos/media_manifest.json \
  --transcript-manifest artifacts/private/formal_captions/dataset_manifest.json \
  --output artifacts/private/full_multimodal \
  --chunk-seconds 300 --overlap-seconds 2 \
  --frame-interval 15 --max-scenes-per-chunk 12 --ocr-workers 4

python3 -m teaching_skill_miner visual-semantic-dataset \
  --manifest artifacts/private/full_multimodal/dataset_manifest.json \
  --model artifacts/private/models/openai_clip_vit_b32_fp16_3d74acf9 \
  --output-dir artifacts/private/full_multimodal/semantic_results \
  --source-model-id openai/clip-vit-base-patch32 \
  --source-revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --device cpu --batch-size 16

python3 -m teaching_skill_miner visual-semantic-apply \
  --manifest artifacts/private/full_multimodal/dataset_manifest.json \
  --semantic-results-dir artifacts/private/full_multimodal/semantic_results \
  --output-manifest artifacts/private/full_multimodal/dataset_manifest.semantic.json

python3 -m teaching_skill_miner audit \
  --manifest artifacts/private/full_multimodal/dataset_manifest.semantic.json \
  --output artifacts/private/full_multimodal/data_audit.semantic.json \
  --require-formal

python3 -m teaching_skill_miner multimodal-ablation \
  --manifest artifacts/private/full_multimodal/dataset_manifest.semantic.json \
  --output artifacts/private/full_multimodal/ablation

python3 scripts/build_multimodal_public_receipts.py
```

本次最终产物声明 pipeline `teaching_skill_miner.longform_multimodal.v9`、extraction `teaching_skill_miner.longform_extraction.v2`。v2 使用正确的 TSV quoting 解析 Tesseract 输出；此前解析结果和由它派生的聚合值已作废并重新跑完 10 讲。新实跑的可核验证据为：10 个完整视频共 `1,038,813,006` bytes，FFprobe 总时长 `26,656.83` 秒（`7.404675` 小时）；抽取 `2,553` 帧，其中 `2,441` 帧有阈值后 OCR 文本、`1,964` 帧至少有 3 个接受词，共接受 `40,191` 个词；生成 `1,974` 个视觉事件和 `2,270` 个融合事件。10/10 讲通过全时间轴采样覆盖门槛，10/10 讲通过官方字幕—媒体时间轴绑定，审计结果为 `formal_empirical_ready=true`、`multimodal_empirical_ready=true`；前者仍只表示字幕来源、文件和时间轴门槛通过，不是字幕内容审计或 WER。

CLIP 对 `2,553/2,553` 个哈希绑定帧完成 512 维视觉嵌入和八类封闭 ontology 的相对 prompt 分数。本次使用 `openai/clip-vit-base-patch32` revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，本地 FP16 `model.safetensors` SHA-256 为 `676093550c9e05bc3ba55256c278c89f0d15a1a1585f3b81d76454c33b852d5e`，最终推理设备为 CPU。曾用 GPU 服务器处理公开模型权重，但没有把视频、帧、字幕、OCR 或其他私有项目数据上传到该服务器。

四臂配对消融对每一讲复用完全相同的 transcript segments，只改变可见的机器派生模态，并排除课堂观察：

| Arm | 平均内部 Skill 分 | 相对 transcript-only | 保留事件数 |
|---|---:|---:|---:|
| transcript-only | 95.37 | 0.00 | 0 |
| transcript + audio | 95.37 | 0.00 | 237 |
| transcript + visual/OCR | 95.30 | -0.07 | 2,033 |
| full | 95.32 | -0.05 | 2,270 |

这些分数来自同一生成与评分流水线的七维内部量表：结构、证据引用一致性、可执行性、方法忠实度、教学质量、迁移和溯源。只有 `method_fidelity` 会从被引用的 evidence 反推 observed 步骤的线索、时间段和引用；其余六维在这 10 份演示 Skill 上标准差均为 0.000。它们仍不是独立人工 Skill 质量评分，也不是 Accuracy、Precision、Recall 或 F1；本次四臂同样没有显示内部 Skill 分数增益。OCR 计数不表示 OCR 正确率，CLIP 相对 prompt 分数不是校准概率，检测器事件数不表示正确事件数。因为没有独立事件 gold label、独立 Skill 质量评分或学习者结果，`recognition_accuracy_established`、`multimodal_gain_established`、`teaching_effectiveness_established` 和 `deployment_accuracy_established` 均为 `false`。完整设计和证据解释见 [`docs/multimodal_design.md`](docs/multimodal_design.md)。

内容最小化的公开证据分别写入 `artifacts/public/full_multimodal_validation_receipt.json` 和 `artifacts/public/multimodal_ablation_receipt.json`。最终总 runner 实跑后的文件 SHA-256 分别为 `33155d14082050ef06302b33f18b5f01894a569147009681d93a18febc90f32b` 与 `81af973544a2a9b49708712ae2cb95f4381791cd4d26e4add7168bf1f95d3410`。它们只保留聚合计数、设计字段和上游私有产物哈希承诺，不包含媒体、字幕/OCR 文本、帧、嵌入、逐讲记录、讲次标识或本地路径；生成后仍须运行 `tsm release-audit artifacts/public` 并做人工披露风险复核。

#### TeachObs 真实标签四臂基准

MIT OCW 10 讲证明了完整视频、官方字幕、音频、OCR、CLIP 和事件融合确实运行，但它没有独立事件真值，不能计算识别 F1。项目因此另接入 commit-pinned TeachObs v0.1：30 个真实课堂完整讲次、5,158 个 15 秒场景、39 个发布共识标签，固定使用官方 23 讲训练/7 讲测试划分。旧的发布文本兼容基线在 1,312 个官方测试场景上得到 Micro-F1 `0.612968`、Macro-F1 `0.360044` 和 Hamming accuracy `0.826180`；它不是当前“官方字幕/审计 ASR 物化文本 + 音频 + 视觉”的四臂结果，且 Hamming accuracy 受多标签负例占比影响，不能单独写成“Accuracy”。

字幕安全检索已实际审计 23/30 讲、34 条轨道，34/34 轨时间轴覆盖率不低于 0.90，均值 0.985841；平台字幕仍缺 7 讲。随后六讲离线 GPU ASR 已按 v2/v4/v4 契约完成并通过导入，当前 coverage matrix 为 19 讲 creator-provided 平台字幕、4 讲平台自动字幕、6 讲审计 ASR，共覆盖 29/30 讲，唯一 pending 为 S4。这里的“审计 ASR”只建立媒体、模型、运行时和时间线 provenance，不是官方字幕，也不建立 WER 或内容准确率。正式物化得到 4,509 个非空场景和 436 个空场景：14,637 条平台 cue 全部保留，其中 12 条只裁剪越界时间戳（合计 69.781 秒、最大 35.522 秒）；3,337 条 ASR segment 保留 3,336 条，另有 1 条位于同一哈希绑定媒体内但完全落在官方 scene 域外，显式排除 2.46 秒。两类处理都只看时间和媒体绑定，静默丢文本计数为 0。`prepare-teachobs-media` 已对论文 profile 的 29 讲/4,945 场景完成完整源视频、逐场景音频统计、中点抽帧、OCR/画面变化和固定 CLIP 特征。为与 arXiv:2605.30673v2 §4.1 的 Track 1 文本/单帧比较对齐，正式总 runner 默认采用 `paper_track1_23_train_6_test`：保留完整 23 讲、3,846 个训练场景，测试固定为 S2/S5/S19/S24/S28/S30 的 1,099 个场景；S4 因论文的中点帧 attachment 口径不统一而在任何拟合、调参和指标前排除，因此媒体/特征分母是 29 讲、4,945 场景，bootstrap cluster 为 6。CLI 为兼容旧流程仍默认 `full_23_train_7_test`（30 讲、5,158 场景、7 讲/1,312 场景测试），可用 `--evaluation-profile` 明确选择。两个 profile 的训练/测试 `source` 元数据均有 3 个取值重叠，且发布数据没有 canonical site/teacher/classroom ID；runner 会把这一事实及集合哈希写入私有结果和聚合公开 receipt，但不会把 `source` 冒充站点或新增伪 site-held-out F1。

四臂修订模型仅在 23 个训练讲次上做 5 折 lesson-grouped OOF：每折重新拟合 transcript/OCR TF-IDF、IDF 与全部数值 scaler，从固定 `C={0.25,1.0}`、`class_weight=balanced` 候选中按 pooled OOF Micro-F1 选择，再从固定 `0.3–0.7` 网格选择逐标签阈值。候选、fold、seed、阈值哈希和 OOF 指标均写入 provenance，冻结 JSON+NPZ 重载后必须与内存概率及预测一致。这个流程是在首轮公开六讲结果已经观察后加入的迭代性探索修订；虽未用六讲标签拟合或选择，也不能写成首次盲测、确认性增益、跨站点或部署准确率。数据审计、命令、双人独立标注和学习效果边界见 [`docs/teachobs_external_benchmark.md`](docs/teachobs_external_benchmark.md)。

正式 v2 结果如下；OOF Micro-F1 只用于训练讲次内选择，测试列来自固定六讲的一次修订后评估：

| Arm | 训练 OOF Micro-F1 | 测试 Micro-F1 | 测试 Macro-F1 | Hamming accuracy | Visual Macro-F1 |
|---|---:|---:|---:|---:|---:|
| transcript-only | 0.611918 | 0.594496 | 0.244338 | **0.818833** | 0.238622 |
| transcript + audio | 0.564436 | **0.604708** | 0.247501 | 0.809967 | 0.244307 |
| transcript + visual | 0.541563 | 0.548328 | 0.303894 | 0.775297 | 0.332851 |
| full | 0.516856 | 0.543812 | **0.309604** | 0.763258 | **0.336426** |

相对 transcript-only，full 的 Macro-F1 增量为 `+0.065266`，六讲配对 cluster bootstrap 95% CI 为 `[+0.014886, +0.095637]`；Visual Macro-F1 增量为 `+0.097804`，CI 为 `[+0.035152, +0.114008]`。但 full 的 Micro-F1/Hamming 分别下降 `0.050684/0.055575`，因此只能说多模态改善了类别均衡和视觉类覆盖，不能说它普遍提高了“Accuracy”。最佳 Micro-F1 是 `transcript + audio`，最佳 Hamming 是 transcript-only；两者都不是 0.9。固定 0.5 阈值首轮结果及旧 bundle 已原样保存在私有 `benchmark_attempts/fixed_threshold_v1/`，避免事后覆盖失败证据。

未来新站点确认性验证另有 fail-closed 预注册层：安装后的 `prepare-teachobs-lockbox-preregistration` 冻结 system/analysis/四臂模型、同一样本主 contrast、39 标签 Macro-F1、classroom/session cluster、固定 2,000 次配对 bootstrap、coverage、前瞻、一次性消费和外部 Ed25519 登记门。四臂现可导出为无 pickle、可重验且预测一致的确定性 JSON+NPZ；schema 1.2 的完成门会解析同一 bundle 的四个 manifest 并安全重载全部 companion `arrays.npz`，把数组 SHA、每臂 39 个 `model_thresholds` 的摘要、标签顺序和训练/软件 provenance 纳入 path-free v2 artifact-set fingerprint。分析计划绑定同一组逐臂阈值摘要，不再错误假定统一 `0.5`。当前公开 draft 仍缺真实目标数据和外部签名，因此所有 established flags 为 false；完整交接见 [`docs/teachobs_confirmatory_lockbox.md`](docs/teachobs_confirmatory_lockbox.md)。

学习效果缺口现在有独立的可执行工具：`prepare-learner-effect-study` 生成 teacher/classroom cluster RCT 预注册、仅 SHA-256 token 的盲化分配表和 pre/post/retention 采集表；`analyze-learner-effect-study` 按冻结的 post-adjusted-for-pre ITT 模型及 cluster bootstrap 做 fail-closed 分析。模板、合成测试和未签名本地结果都保持 `learner_effectiveness_established=false`，不得把内部 Skill 分当学习效果；操作与外部签名交接见 [`docs/learner_effect_study.md`](docs/learner_effect_study.md)。

运行仓库内的可复现多模态演示：

```bash
sh scripts/run_multimodal_demo.sh
```

该脚本生成一段本地合成视频并真实运行 FFmpeg、Tesseract、事件融合、Skill 抽取和评估。当前稳定产物为 1 个静音区间、4 个关键帧、9 个融合事件和 5 种输入模态；其中课堂观察来自人工合成 JSON，音轨为音调加静音，提供的 transcript 并非音轨 ASR。

脚本还会把输出与独立编写的 fixture 标注比较，生成 `artifacts/multimodal_demo/fixture_benchmark.json`。该合成基准当前对 3 个指定事件得到 Precision/Recall/F1 = 1.0、OCR 必需词召回率 = 1.0；报告同时固定写明 `real_world_accuracy_established: false` 与 `teaching_effectiveness_established: false`。这证明已知合成条件下的链路行为，不证明真实课堂准确率。详细审计见 [`docs/validation_report.md`](docs/validation_report.md)，设计见 [`docs/multimodal_design.md`](docs/multimodal_design.md)。

### 真实课堂自动识别（OUC-CGE pilot）

仓库另有一条与内部 Skill 评分相互独立的真实识别链路：使用 [OUC-CGE](https://www.nature.com/articles/s41597-025-04987-w) 官方 OSF 公开 sample 中的真实课堂视频和专家标签，自动输出 `low / medium / high` 小组投入度概率，并计算文件名伪分组 out-of-fold Accuracy、Macro-F1、逐类指标、混淆矩阵、校准指标和 bootstrap 95% 区间。

安装识别依赖并运行：

```bash
python3 -m pip install -e '.[recognition]'

python3 -m teaching_skill_miner real-data-audit \
  --dataset-root data/real/ouc_cge/extracted \
  --output artifacts/real_classroom/dataset_audit.json

python3 -m teaching_skill_miner real-recognition-benchmark \
  --dataset-root data/real/ouc_cge/extracted \
  --output-dir artifacts/real_classroom \
  --folds 4 --frames 12 --seed 2026
```

对新视频执行自动推理：

```bash
python3 -m teaching_skill_miner real-recognition-infer \
  --video /path/to/classroom.mp4 \
  --checkpoint artifacts/real_classroom/checkpoint.json \
  --output artifacts/real_classroom/inference.json
```

当前可直接复现的是 36 个时长不一的官方 `sample` 视频；SHA-256 去重后为 35 个。实验按主视频流时间轴取中心 10 秒，并用文件名末尾数字构造标签无关的**伪分组**做 4 折 `StratifiedGroupKFold`。这能防止已知精确重复和同尾号跨折，却不等于按源视频、课堂 session 或参与者划分。因此报告固定写明 `group_disjoint_evaluation: false`、`surrogate_filename_group_disjoint_evaluation: true`、`session_disjoint_evaluation: false` 和 `real_world_recognition_accuracy_established: false`。

本地实测的视觉、音频和融合 Macro-F1 都是 1.0000，但**不能解读为 100% 真实识别率**：只看时长、码率、分辨率、编码器和流数量的元数据对照也达到 Macro-F1 0.9710，强烈表明来源/编码混杂风险。更关键的是，low 只有 0/11 个视频含重叠有效音频，而 medium/high 都是 12/12；因此 audio/fusion 的满分是无效的模态缺失捷径，本地 pilot 仅选择 visual checkpoint 用于链路调试，并明确不纳入公开 release。视觉特征后的标签置换 sanity check 平均 Macro-F1 为 0.3060，但它不能排除目标泄漏。当前没有建立多模态增益、跨 session 准确率或部署准确率。

下载地址、校验和、隐私要求见 [`data/real/README.md`](data/real/README.md)；完整协议、许可边界和升级为正式实验的条件见 [`docs/real_classroom_protocol.md`](docs/real_classroom_protocol.md)。

#### DIPSER V5 可审计的姿态 + 手表基准

另一条真实数据链路使用 [DIPSER V5](https://doi.org/10.57760/SCIENCEDB.11541) 线下大学课堂数据，将发布方由 RGB 生成的 head/body pose metadata 与五类智能手表传感器对齐；它不读取 raw RGB 像素，因此不是端到端视频模型。每个归档必须恰好解析出四个 publisher expert labeler；ID 可为 `01`–`05`，V5 实际有 `01/02/03/04` 和 `01/02/03/05` 两种组合（后者 22 个 archive），`self_labeling` 排除。这项解析修正不放宽“四专家齐全 + 3/4 多数”真值门槛。Range 读取在所需 non-image 尾部 ≤128 MiB 时做有界合并，异常布局时只逐成员 Range，不下载整包。默认 52-archive 清单是“部分标签清单审查后冻结的探索性分析”，不是正式预注册。

```bash
python3 -m teaching_skill_miner dipser-credible-benchmark \
  --output-dir artifacts/dipser_credible/full_v5_52 \
  --workers 2 --outer-splits 5 --inner-splits 3 --seed 2026
```

主报告写入 `artifacts/dipser_credible/full_v5_52/credible_report.json`，LOCO、LOAO 和 cohort×activity 双重阻断结果写入同目录的 `blocked_descriptive_report.json`。冻结 roster 计划了单站点 `3 cohort × 9 repeated activity = 27` 个 recording cells，但 complete-case 数据实际只有 25 个 cell 含样本；因此现有 OOF、bootstrap/permutation 和阻断指标均只作同设计内描述。多模态增益、可推广跨 session 准确率和 `deployment_accuracy_established` 均保持 `false`。完整运行、完整数值模态、指纹和目标站点前瞻冻结测试要求见 [`docs/dipser_credible_benchmark.md`](docs/dipser_credible_benchmark.md)。

仓库内已完成一次 52/52 官方归档实跑：296 个完整案例、25 个有样本的 recording。原冻结分析的 session-blocked fusion Accuracy/Macro-F1 为 0.6622/0.5346，而多数类 Accuracy 已有 0.7568。随后新增了**明确标记为事后开发**的优化协议：严格因果 Fusion 模型为 0.8176/0.6357；完整 participant-sequence + classroom-session 的层级离线模型，在固定 SGKF5 seed=2026 上触及 0.9020/0.7435。后者的确定性 leave-one-session-out 为 0.8986/0.7397，50 个 outer seed 的 Accuracy 均值仅 0.8883（范围 0.8784–0.9020，21/50 达到 0.9），所以不能表述为稳定跨 session 0.9。

同一 nested 协议下，Visual、Sensor、Fusion 的 SGKF5 分别为 0.6284/0.2882、0.7973/0.5811、0.9020/0.7435，当前数据上存在明显的多模态互补点差。但层级模型需要完整 held-out session 的无标签未来窗口与其他学生上下文：排除当前行后 SGKF5 降为 0.8547，严格时间前缀降为 0.7770。因此该结果的准确名称是 **offline full-session transductive development estimate**，不是实时、单学生、归纳式或部署准确率；确认性多模态增益仍需新冻结外部数据。

不重新下载数据即可复跑优化与全部身份审计：

```bash
python3 scripts/run_dipser_optimization.py \
  --artifact-dir artifacts/dipser_credible/full_v5_52_complete_v5

python3 scripts/run_dipser_hierarchical_challenge.py \
  --artifact-dir artifacts/dipser_credible/full_v5_52_complete_v5
```

可公开核对的 DIPSER 数值与边界见 [`dipser_0_9_aggregate_summary.json`](artifacts/public/dipser_0_9_aggregate_summary.json) 和 [`docs/model_card_dipser.md`](docs/model_card_dipser.md)。逐折、逐样本、participant 标识、特征与本地路径产物保留在私有研究目录，不作为公开链接。层级固定 gate 的 LOCO、LOAO、双重阻断 Accuracy 分别只有 0.6554、0.8919、0.6351。因此 `accuracy_0_9_established`、确认性 `multimodal_gain_established` 和 `deployment_accuracy_established` 均保持 `false`。

两个 runner 现在都会从 records、特征名和精确 float64 矩阵重新计算 dataset / feature-bundle fingerprint；只修改 JSON 中重复声明的哈希无法绕过校验。

通用 `extract-strict-features` 入口可直接从私有目标站点视频/音频与 CSV/JSON/JSONL 数值传感器生成 `strict_feature_bundle.json`，绑定原始文件 SHA-256、时间窗口、精确样本顺序、提取配置、实现代码、运行环境以及 FFmpeg/FFprobe 二进制和版本。输出目录/文件权限为 `0700/0600`；可用 `--model` 只做 checkpoint v3 schema/provenance 兼容性检查，但不会执行预测，也不会建立任何准确率：

```bash
python3 -m teaching_skill_miner extract-strict-features \
  --manifest PRIVATE_LOCKBOX/dataset_manifest.json \
  --raw-root /secure/target_site_lockbox \
  --extractor-config configs/raw_feature_extractor.example.json \
  --identity-field session_id --identity-field participant_id \
  --identity-field cohort_id --identity-field activity_id --identity-field site_id \
  --output-dir PRIVATE_LOCKBOX/features
```

完整 manifest `raw_inputs`、`content_sha256` 计算和支持的提取器见 [`docs/raw_feature_bridge.md`](docs/raw_feature_bridge.md)。若要在新外部数据上进行可信锁箱测试，应冻结一个独立的扁平严格模型、身份契约和指标门槛（该入口不是 0.902 层级模型部署器）：

```bash
python3 -m teaching_skill_miner freeze-recognition-model \
  --manifest PRIVATE_SOURCE/dataset_manifest.json \
  --features PRIVATE_SOURCE/features.json \
  --claim-contract configs/external_claim_contract.example.json \
  --class-name low --class-name medium --class-name high \
  --identity-field session_id --identity-field participant_id \
  --identity-field cohort_id --identity-field activity_id --identity-field site_id \
  --group-field session_id \
  --claim-cluster-field session_id \
  --output PRIVATE_SOURCE/frozen_model.json
```

从创建登记开始，应在锁箱保管方控制的环境内执行，并把精确外部 dataset、严格 feature bundle、coverage 分母证据和 checkpoint v3 绑定到登记请求：

```bash
python3 -m teaching_skill_miner create-freeze-registration \
  --model PRIVATE_SOURCE/frozen_model.json \
  --manifest PRIVATE_LOCKBOX/dataset_manifest.json \
  --features PRIVATE_LOCKBOX/features.json \
  --registration-id target-site-2026-lockbox-001 \
  --output PRIVATE_LOCKBOX/freeze_registration_request.json

python3 -m teaching_skill_miner sign-freeze-registration \
  --request PRIVATE_LOCKBOX/freeze_registration_request.json \
  --private-key CUSTODIAN_PRIVATE/ed25519_private.pem \
  --issuer "External governance custodian" \
  --key-id target-site-key-2026 \
  --output PRIVATE_LOCKBOX/freeze_registration_attestation.json

python3 -m teaching_skill_miner evaluate-frozen-recognition \
  --model PRIVATE_SOURCE/frozen_model.json \
  --manifest PRIVATE_LOCKBOX/dataset_manifest.json \
  --features PRIVATE_LOCKBOX/features.json \
  --registration-attestation PRIVATE_LOCKBOX/freeze_registration_attestation.json \
  --trusted-public-key TRUST_ANCHOR/ed25519_public.pem \
  --one-time-ledger PRIVATE_LOCKBOX/one_time_ledger \
  --output PRIVATE_LOCKBOX/external_evaluation.json

python3 -m teaching_skill_miner sign-evaluation-report \
  --report PRIVATE_LOCKBOX/external_evaluation.json \
  --private-key CUSTODIAN_PRIVATE/ed25519_private.pem \
  --issuer "External governance custodian" \
  --key-id target-site-key-2026 \
  --output PRIVATE_LOCKBOX/external_evaluation_receipt.json

python3 -m teaching_skill_miner verify-delivery \
  --external-deployment-report PRIVATE_LOCKBOX/external_evaluation.json \
  --external-evaluation-receipt PRIVATE_LOCKBOX/external_evaluation_receipt.json \
  --trusted-attestation-public-key TRUST_ANCHOR/ed25519_public.pem
```

只有 checkpoint 内冻结的 Accuracy、Macro-F1、cluster-CI 下界、逐类 Recall/支持、claim-cluster 数、可重算 coverage、前瞻采集、签名登记和一次性消费全部通过，`deployment_accuracy_established` 才可能为 `true`。Ed25519 签名只证明内容由对应私钥签署且未被篡改，**不会自动证明签署者独立**；可信公钥必须由外部治理方在看结果前通过独立渠道固定，开发者自签名不能把开发集变成锁箱。

部署准确率、确认性多模态增益和真实学习者效果使用三条相互独立的外部证据链。本仓库目前只实现这些证据链的协议、校验和签名接入工具，没有实际执行外部目标站点锁箱或真实学习者 A/B。后两类可以用 `prepare-external-research-evidence`、`sign-external-research-evidence` 和 `verify-external-research-evidence` 对聚合研究 manifest 重新计算 gate 并验证外部签名，再分别通过 `verify-delivery` 的 `--external-multimodal-*` 与 `--external-learner-*` 参数接入。两类 manifest 还必须绑定同一个实际交付 system artifact，delivery 会实算其 SHA-256。仓库当前不附带任何正向外部证据；一个有效部署 receipt 不能替代配对模态消融或学习效果实验，准备 manifest 也不自动证明其绑定的原始研究产物真实。完整协议见 [`docs/external_research_evidence.md`](docs/external_research_evidence.md)。

### 新视频一条命令完成现场演示

`pipeline` 按“预处理 → 多模态对齐 → Skill 抽取 → 教学过程 → 交互执行 → 自动评估”运行并保存所有中间结果。输入为视频时默认启用多模态；可用 `--no-multimodal` 显式关闭：

```bash
python3 -m teaching_skill_miner pipeline new_lesson.srt \
  --video-id new_001 \
  --course-id demo_course \
  --title "New lesson" \
  --source-url "https://example.org/lesson" \
  --concept "牛顿法" \
  --output artifacts/new_video_demo
```

若输入已经是本系统的 transcript JSON，只需提供概念和输出目录：

```bash
python3 -m teaching_skill_miner pipeline data/transcripts/python_l03.json \
  --concept "递归调用栈" \
  --output artifacts/pipeline_demo
```

若有真实视频以及官方字幕、人工核验 transcript 或已审计 ASR，可在没有本机 Whisper 模型时仍用一个命令执行真实 FFmpeg/OCR 多模态闭环：

```bash
python3 -m teaching_skill_miner pipeline /path/to/authorized_lesson.mp4 \
  --transcript /path/to/official_or_audited_transcript.json \
  --observations /path/to/anonymized_observations.json \
  --concept "目标概念" \
  --output artifacts/captioned_video_run
```

该路径会在 `pipeline_summary.json` 中写入 `transcript_source_mode`、`language_evidence_status` 和历史字段 `audio_content_verified`。提供的文字只算 transcript 模态；只有同一媒体哈希绑定的 ASR provenance 才会让该历史字段为 `true`。它表示来源绑定，不表示人工听写核对、字幕内容正确或 WER 已建立。

对获授权的真实课堂视频可用答辩脚本一次完成“本机 ASR（若就绪）或第八参数官方/审计 transcript”之后的多模态分析、Skill、教学过程、脚本化 fallback 和自动评估：

```bash
sh scripts/run_defense_demo.sh \
  /path/to/authorized_lesson.mp4 artifacts/defense_demo \
  lesson_001 course_001 "Lesson title" "https://authorized.example/lesson" "目标概念"
```

若现场没有 OpenAI Whisper CLI，可把官方/审计 transcript 作为第八个参数传入同一脚本。

脚本化学生回应只用于稳定展示状态机，不是学习效果证据。真实媒体、逐样本特征和预测遵循 [`PRIVACY.md`](PRIVACY.md)，公开 wheel/目录应先运行 `tsm release-audit`。

### 2. Teaching Skill 抽取

```bash
python3 -m teaching_skill_miner mine \
  --transcript data/new_001.json \
  --backend heuristic \
  --output artifacts/new_001.skill.json
```

离线抽取器对中英文关键词、时间段和教学事件进行可解释打分，覆盖题目要求中的具体例子、逐步拆解、提问、追问、纠错、对比、先直觉后形式化、整体到局部、代码/公式逐行解释、练习反馈、难度递进、回顾、迁移和动态调整等策略。

每个策略和 procedure 步骤都标记 `origin`：`observed_method` 表示有视频/转写证据，`recommended_enrichment` 表示系统补充的通用教学脚手架。每个证据有稳定 `evidence_id`，观察到的步骤必须列出 `evidence_ids`；推荐步骤不会被计入“视频中观察到的方法”覆盖率。

#### 教学步骤：只对有证据的环节还原教师顺序

题目 4.2 列出九个教学环节。[`teaching_phases.py`](teaching_skill_miner/teaching_phases.py) 逐条实现它们，用线索匹配定位有证据环节在转写中的首次出现。只有 `observed_method` 环节按首次证据时间排序，形成对教师实际顺序的可追溯主张；未观察到的 `recommended_enrichment` 会按规范环节位置插入，使 procedure 可以完整执行，但其位置和动作只是系统脚手架，不声称教师在该时刻使用过该环节：

| 环节 | `teaching_phase` |
|---|---|
| 复习前置知识 | `prior_knowledge_review` |
| 提出问题或情境 | `problem_or_context_setup` |
| 给出直观例子 | `intuitive_example` |
| 建立抽象概念 | `abstract_concept_building` |
| 展示推导或操作 | `derivation_or_operation_walkthrough` |
| 检查学生理解 | `understanding_check` |
| 纠正常见错误 | `error_diagnosis_and_correction` |
| 练习与反馈 | `practice_and_feedback` |
| 总结和迁移 | `summary_and_transfer` |

命中的环节带真实时间段、触发线索和 `evidence_ids`，标 `origin=observed_method`；未命中的环节仍然补进 procedure 以保证可执行，但明确标 `origin=recommended_enrichment` 且 `observed_span` 为 `null`，指令里直接写明"视频中未观察到该环节"。推荐环节没有教师时序主张，两者在 JSON 里从不混淆：

```json
{
  "step": 2,
  "teaching_phase": "intuitive_example",
  "teaching_phase_name": "给出直观例子",
  "origin": "observed_method",
  "observed_span": {"start": 0.0, "end": 100.0},
  "matched_cues": ["example"],
  "evidence_ids": ["evi_35d8659f0cbfbac8"],
  "provenance": {"derivation": "observed_teaching_phase_from_timeline"}
}
```

`mining_metadata.teaching_phase_analysis` 记录本讲由线索启发式检测到哪些环节、其首次证据顺序以及哪些没出现。10 份演示转写各检测到 4–7 个环节，没有一份凑满九个。当前没有独立的 phase-level gold label，因此这些计数只描述检测器输出，不建立教学环节识别准确率。

每个 Skill 都带 `source.evidence`；证据包含开始/结束时间、原转写中的精确引文和所支持的策略。评估器会做逐字匹配，伪造证据无法通过 grounding gate。

多模态 Skill 还带 `source.multimodal_evidence`。评估器会核对事件 ID、时间戳、模态集合、策略映射、证据载荷、置信度，以及证据引用的帧/OCR/静音记录；伪造内容会使 `multimodal_consistent` gate 失败。该分数名为 `internal_evidence_consistency`，不是识别 Precision/Recall/F1，也不是教学效果分。

### 3. Skill 可执行化与新任务迁移

Skill 使用 JSON 表示（JSON 同时是 YAML 1.2 的合法子集），结构约束见 [`schema/teaching_skill.schema.json`](schema/teaching_skill.schema.json)。除题目要求的字段外，每一步都有：

- `teacher_action`：Agent 动作；
- `instruction`：带 `{concept}` 参数的指令；
- `expected_signal`：进入下一步的学生信号；
- `fallback`：信号未出现时的降阶、提示或回退策略。

将已抽取方法迁移到新概念：

```bash
python3 -m teaching_skill_miner teach \
  --skill artifacts/skills/python_l03.skill.json \
  --concept "牛顿法" \
  --learner-level beginner \
  --output artifacts/newton_lesson.md
```

静态教案之外，`interact` 会真正执行 Skill 状态机。每轮显示教师动作和预期信号，由教师、上层 Agent 或独立判分器判断是否达标；未达标停留原步骤并执行 fallback，连续两次失败则降低复杂度并回查前置知识：

```bash
python3 -m teaching_skill_miner interact \
  --skill artifacts/skills/python_l03.skill.json \
  --concept "递归调用栈"
```

### 4. 自动评估与人工验证

```bash
python3 -m teaching_skill_miner evaluate \
  --skill artifacts/skills/python_l03.skill.json \
  --transcript data/transcripts/python_l03.json \
  --output artifacts/python_l03.report.json
```

总分为七个维度的加权和：

| 维度 | 权重 | 核心检查 |
|---|---:|---|
| 结构完整性 | 12% | 必填字段、类型、动作词表 |
| 证据落地性 | 18% | 时间戳、精确引文、证据数量 |
| 可执行性 | 18% | 步骤、观察信号、失败回退、参数 |
| 方法忠实度 | 22% | observed_method 步骤能否被其引用证据反推验证 |
| 教学质量 | 12% | 前提、成功标准、失败模式、验证题 |
| 可迁移性 | 9% | 概念参数、近迁移题、边界/反例 |
| 可追溯性 | 9% | 课程、视频、来源、转写类型 |

其余六个维度检查的都是挖掘器按构造必然写出的字段，因此在本数据集上全部饱和（十个 Skill 的每个维度总体标准差均为 0.000）。**方法忠实度**是唯一一个需要重新推导才能得分的维度：它不看步骤"有没有写"，而是拿每个 `observed_method` 步骤引用的证据记录反推它自己的声明。

| 子项 | 权重 | 检查 |
|---|---:|---|
| `phase_coverage` | 30% | 九个规范环节中实际还原了几个 |
| `cue_verification` | 20% | 步骤声称的触发线索是否真的出现在它引用的引文里 |
| `span_consistency` | 15% | 步骤声称的时间区间是否真的包含它引用的每条证据 |
| `evidence_utilisation` | 15% | 已挖掘的证据被 procedure 引用的比例 |
| `temporal_monotonicity` | 10% | observed 步骤是否沿视频时间轴单调推进 |
| `evidence_density` | 10% | 每个 observed 步骤的引文条数（上限 2 条即满分） |

第 1、4、6 项（`phase_coverage`、`evidence_utilisation`、`evidence_density`）在诚实的 Skill 之间本就有差异，衡量"方法还原了多少"；第 2、3、5 项（`cue_verification`、`span_consistency`、`temporal_monotonicity`）在当前十份诚实产物上恒为 1.0，只有在步骤伪造出处或时序时才会塌陷。它们共同构成内部、可证伪的工程量表，不是独立人工质量评分。当前十个 Skill 的方法忠实度落在 83.3–89.8（标准差 1.64），总分落在 92.3–93.7。

`tests/test_method_fidelity.py` 用五种人工降级（纯模板、伪造线索、打乱时间区间、抽掉证据、塌缩环节）验证它确实在测量：每种降级都必须被对应的子项抓到，且总分严格下降；把这个维度钉成常数会让其中八个测试失败。

通过条件为总分至少 75，并且同时通过全部硬门槛：schema 合法、grounding 至少 60、executability 至少 70、至少两种测试、方法忠实度至少 40（`method_distilled_from_video`）；多模态 Skill 还要额外通过 `multimodal_consistent`。自动高分仅表示"工程与量表合规"，不等价于真实学习效果。人工协议、评分锚点与 A/B 指标见 [`docs/human_review_guide.md`](docs/human_review_guide.md)。

留出新任务测试覆盖分数除法、光合作用、递归、条件概率、牛顿第三定律和议论文结构：

```bash
python3 -m teaching_skill_miner benchmark \
  --skills artifacts/skills \
  --cases data/evaluation_cases.json \
  --output artifacts/transfer_benchmark.json
```

报告同时给出静态“解释—示例—总结”基线，并检查概念参数替换、源主题泄漏、fallback 触发和失败后恢复。该 benchmark 是确定性的能力覆盖测试，不冒充真实学生学习增益。

下面是双人复核的协议工具入口，不是已完成的人工实验。当前仓库只有空评分模板、覆盖/指纹校验和汇总代码；真实两名复核者尚未回填。只有未来完成真实评分表后，才可计算逐 Skill 结果和双人二次加权 Cohen's kappa：

```bash
python3 -m teaching_skill_miner human-evaluate \
  --input artifacts/human_review.csv \
  --skills artifacts/skills \
  --output artifacts/human_review_report.json
```

只有预期 Skill 100% 覆盖、每项至少两名不同复核者、无重复 `(skill_id, reviewer_id)`、每行 `skill_fingerprint` 与当前完整 Skill 内容一致、五维门槛和一致性门槛全部通过，人工验证才成立。同一个 `skill_id` 的内容只要发生变化，旧评分也会失败关闭。项目不会填充虚假评分；旧 CSV 若缺 fingerprint，应在新输出目录生成当前模板并重新确认评分对应的确切版本，不能把旧评分无证据地追溯绑定到新 Skill。

## 数据来源与边界

样例选取：

- [MIT 18.06 Linear Algebra 视频列表](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/video_galleries/video-lectures/)前 5 讲；
- [MIT 6.0001 Python 视频列表](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/video_galleries/lecture-videos/)前 5 讲。

逐视频演示文件对应关系见 [`data/dataset_manifest.json`](data/dataset_manifest.json)；10 个官方字幕 URL、固定 SHA-256、页面关联媒体和参考时长见 [`data/formal_caption_sources.json`](data/formal_caption_sources.json)。该索引不含字幕正文或视频。

仓库中的 `data/transcripts/*.json` 仍是为了离线、快速、可复现演示而人工整理的短篇英文释义节选，时间戳为近似值，不冒充完整逐字 ASR。正式字幕使用独立私有 manifest，避免把第三方全文混入项目 wheel：

```bash
python3 -m teaching_skill_miner fetch-formal-captions \
  --source-manifest data/formal_caption_sources.json \
  --output artifacts/private/formal_captions \
  --public-receipt artifacts/public/formal_caption_retrieval_receipt.json \
  --acknowledge-source-terms

python3 -m teaching_skill_miner audit \
  --manifest artifacts/private/formal_captions/dataset_manifest.json \
  --output artifacts/private/formal_captions/independent_data_audit.json \
  --require-formal

python3 -m teaching_skill_miner verify-delivery \
  --formal-manifest artifacts/private/formal_captions/dataset_manifest.json \
  --output artifacts/delivery_verification.json \
  --markdown artifacts/DELIVERY_VERIFICATION.md
```

`fetch-formal-captions` 从每个 MIT OCW 讲次页面重新确认页面声明的 WebVTT 与媒体配对，要求字幕 URL 使用 `https://ocw.mit.edu`，校验固定字幕文件 SHA-256，并用 FFprobe 读取同页所链接媒体的实际时长；该命令本身不保存视频。当前实跑结果为 10/10 字幕来源、文件与时间轴门槛通过，平均 770.5 个合并后 cue、6172.8 个审计 token，首尾 cue 时间轴覆盖率为 98.84%–99.72%；这不是逐字听写内容审计，也没有计算 WER。公开 receipt 只含 URL、哈希、时长和计数，不含字幕文本。随后独立执行 `fetch-full-videos` 已把页面绑定的 10 个完整媒体下载到私有目录，并再次校验本地 SHA-256、容器、音视频流和参考时长；由于上游索引没有发布者固定的媒体哈希，本地 SHA-256 能检测下载后的变化，但不能独立证明发布者原始字节身份。

因此：

- 自动评估会把这种数据的证据忠实度与溯源分限制在 85；
- 正式字幕集使用官方 caption 时应报告页面 URL、字幕文件哈希、媒体时长和时间轴覆盖；这些来源与文件证据不替代独立的逐字内容核对。若改用 ASR，还必须另外报告模型、版本、模型/解码配置指纹、独立 WER 抽检和人工修订比例；
- 使用公开数据时必须遵守来源页面的许可、署名和非商业条款。

TeachObs 的平台字幕缺口另有一条不下载模型/媒体的离线 GPU 交接链：
`prepare-teachobs-asr-handoff` 从私有媒体 manifest 生成逐讲媒体哈希、时长、固定
Whisper revision、拒绝任意 symlink/非普通节点的模型树哈希、解码和冻结了精确
容器镜像 digest 的 CUDA runtime contract；安装后的
`hash-teachobs-asr-model` 与 `run-teachobs-asr-gpu` 只接受预置的本地模型快照，
源码 checkout 也保留等价的 [`run_teachobs_asr_gpu.py`](scripts/run_teachobs_asr_gpu.py)；
[`build_teachobs_asr_container.sh`](scripts/build_teachobs_asr_container.sh) 则从已验收
的 exact wheel 和已预置、ID 匹配的本地 CUDA base 构建，不使用工作区作为 context，
不拉取 base、模型或媒体，并在返回最终 image ID 前于断网只读容器内复验四个固定
GPU package version、GPU CLI help 和 `ffprobe`；
`import-teachobs-asr-results` 重新验媒体、manifest、segment 和完整 provenance，
再按“creator-provided 平台字幕 → 平台自动字幕 → 审计 ASR fallback”生成 30 讲
coverage matrix。ASR job manifest v2 / lesson result v4 / GPU runner v4 以完整媒体
单次输入为前提，把 VAD 第一/最后语音锚点的首尾空白统一按媒体时长比例验收：
first-to-last span 至少 `0.90`，两端各自至多 `0.10`；端点秒数只作诊断，不再用
任意固定 30 秒门槛。旧 v1/v3 证据失败关闭且必须整批重跑，不能事后改写或只重跑
未通过的讲次。六讲 v4 结果现已全部通过导入；当前
[`teachobs_asr_receipt.json`](artifacts/public/teachobs_asr_receipt.json) 记录
`valid_asr_result_count=6`、`covered_lesson_count=29`，并因唯一缺口 S4 而继续保持
pending。ASR 永远不冒充官方字幕，未经独立人工参考抽检时 WER 和内容准确率仍为
false。完整操作见
[`docs/teachobs_audited_asr_handoff.md`](docs/teachobs_audited_asr_handoff.md)。

运行数据审计：

```bash
python3 -m teaching_skill_miner audit \
  --manifest data/dataset_manifest.json \
  --output artifacts/data_audit.json
```

审计分别报告 `dataset_structure_passed` 与 `formal_empirical_ready`，因此默认释义节选保持 `formal_empirical_ready=false`，正式私有字幕 manifest 为 `true`。这里的 `formal_empirical_ready` 只表示完整转写的来源、身份、时间戳和覆盖门槛通过，不等于教学方法标签正确、多模态增益成立、学生学习有效或部署准确率成立。

## API / 推理配置

默认：

```bash
TSM_BACKEND=heuristic
```

可选的 API 后端会先生成结构合法的离线 baseline，再让兼容 Responses API 的模型基于转写证据细化，最后重新做 schema 校验。复制 `.env.example` 中的变量到当前 shell（项目不会自动读取或提交密钥）：

```bash
export TSM_API_BASE="https://api.openai.com/v1"
export TSM_API_KEY="..."
export TSM_MODEL="gpt-4.1-mini"
export TSM_ALLOW_REMOTE_TRANSCRIPT_UPLOAD=1

python3 -m teaching_skill_miner mine \
  --transcript data/transcripts/python_l03.json \
  --backend api \
  --output artifacts/python_l03.api.skill.json
```

推理失败、返回非 JSON 或字段不合法时命令会明确报错，不会静默输出未经校验的 Skill。密钥只能通过环境变量传入；非本地 API 必须使用 HTTPS，并在确认授权、最小化、保留和跨境要求后显式允许转写片段外发。

## 许可

本仓库原创代码和文档采用 [`Academic Evaluation License 1.0`](LICENSE)：允许学术评估、教学和非商业研究使用与修改，但商业使用、再许可和公开部署需要另行书面许可，并禁止把系统作为学生或教师高影响决策的唯一依据。

该许可不覆盖或重新许可任何第三方课堂视频、字幕、标签、姿态/传感器数据、预训练模型、字体、编解码器或外部工具。使用者必须分别核对上游条款、署名、隐私、知情同意、伦理审批和数据使用协议，详见 [`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md) 与 [`PRIVACY.md`](PRIVACY.md)。

wheel 安装会把 `CHANGELOG.md`、`PRIVACY.md`、`SECURITY.md` 和 `THIRD_PARTY_DATA.md` 一并放入 `share/teaching-skill-miner/governance/`，确保脱离源码仓库安装后仍能读取许可、隐私和安全边界。

## 目录结构

```text
teaching_skill_miner/
  preprocess.py       字幕解析、文本切段、媒体 ASR
  multimodal.py       静音、关键帧、OCR、课堂观察与时间轴融合
  longform_multimodal.py  完整讲次分块、全程覆盖、字幕/音频/视觉对齐与事件融合
  multimodal_ablation.py transcript / +audio / +visual / full 配对内部消融
  full_video_dataset.py  MIT OCW 完整媒体下载、哈希、FFprobe 与私有存储验证
  visual_semantics.py    哈希绑定 CLIP 单任务推理与模型/运行时 provenance
  visual_semantics_dataset.py  逐讲批处理与私有 semantic batch receipt
  visual_semantics_apply.py    语义结果回绑 transcript/analysis 与 dataset manifest
  recognition/        真实课堂数据审计、视听觉特征、分组评估与推理
  miner.py            目标/策略/动作抽取与 Skill 生成
  executor.py         参数替换、状态分支、教学过程生成
  runtime.py          学生信号驱动的可执行状态机
  evaluator.py        单 Skill 与数据集级自动评估
  benchmark.py        跨领域留出任务与静态基线对比
  audit.py            数据完整性与研究就绪审计
  formal_captions.py  官方页面/VTT/媒体时长与哈希绑定的正式字幕导入
  human_eval.py       双人复核汇总与一致性计算
  delivery.py         一键交付验收与外部证据缺口
  project_health.py   资源、依赖、工具和 API 安全自检
  release_audit.py    wheel/公开目录隐私发布审计
  llm_backend.py      可选 Responses API 细化
  models.py           数据契约与严格校验
  cli.py              pipeline/interact/benchmark/audit/demo 等命令
data/
  dataset_manifest.json
  formal_caption_sources.json  10 个官方字幕的无正文 URL/哈希索引
  transcripts/        2 门课程 × 5 讲离线样例
  real/               本地真实课堂数据说明；原始视频不得提交
schema/               Skill、严格 manifest/feature bundle、checkpoint v3、登记签名、receipt 和 claim contract schema
configs/              可复制后冻结的外部验收门槛示例
tests/                预处理、抽取、证据、防伪、迁移、覆盖测试
artifacts/            演示生成物；真实课堂逐样本产物保持私有
  private/full_videos/       完整媒体与私有下载 manifest
  private/full_multimodal/   帧、OCR、语义嵌入、事件、审计和四臂消融
  public/full_video_validation_receipt.json  不含媒体/字幕/帧的聚合验证 receipt
  public/full_multimodal_validation_receipt.json  完整链路聚合证据与私有产物哈希承诺
  public/multimodal_ablation_receipt.json  四臂设计、聚合内部指标与结论边界
```

## 当前限制与下一步实验

1. 当前已覆盖完整视频停顿等待、场景变化、OCR 文字演进、CLIP 封闭 ontology 视觉语义和匿名课堂观察；手势指向、复杂图表关系推理仍需专用检测器或经标注验证的视觉语言方法。CLIP 相对 prompt 分数本身不是分类准确率。
2. 自动评估主要检查过程质量，不能替代真实学生的前后测。
3. 完整长视频已经分块并融合为事件时间轴，但一个视频当前仍输出一个主 Skill；抽取器已区分观察证据和推荐脚手架，尚未完成 episode 级多 Skill 聚类和独立专家 gold-set 评测。
4. 下一步应在冻结当前检测/抽取 pipeline 后，由独立复核者标注事件和 Skill 质量，并在新数据上做配对四臂确认性消融；正式字幕、OCR/CLIP 输出和内部量表完成都不自动构成方法 gold label 或多模态增益。
5. 真实课堂公开 sample pilot 已运行 Visual、Audio、Fusion 消融，但音频覆盖与标签完全相关，Audio/Fusion 结果已判无效；本地链路只选择 visual checkpoint 调试，且不公开发布。要声称可泛化准确率，仍需完整数据、按 session/参与者隔离的测试集和跨学校外部验证。事件级问题检测、困惑识别等任务还需要各自的人工时间段真值，不能沿用投入度分类分数。
6. DIPSER 的严格因果开发期 Accuracy 为 0.8176；使用完整 held-out session 的离线 transductive 层级候选在一个固定 SGKF5 划分触及 0.9020，但 LOSO 为 0.8986、50-seed 均值为 0.8883，LOCO/双重阻断仅 0.6554/0.6351。冻结全部模型、gate 和特征后，还需要未参与开发的新 cohort/session 或外部学校锁箱数据，才能建立确认性 0.9 与多模态增益。

逐条题目映射与答辩提示见 [`docs/requirements_traceability.md`](docs/requirements_traceability.md)。
