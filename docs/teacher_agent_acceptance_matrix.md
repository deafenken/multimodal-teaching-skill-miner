# 题目二交付与验收矩阵

## 1. 验收原则

本矩阵只记录当前仓库能由代码、Schema、冻结数据、公开 receipt 或自动测试支持的事实。状态分为四类：

- **已实现并测试**：工程能力已存在，且有自动测试或可复现命令；
- **已有开发证据**：已有量化结果，但数据或协议仍属于开发阶段；
- **部分实现 / 受限**：核心接口存在，但使用场景或覆盖范围有限；
- **外部验证待完成**：必须依赖专家标注、锁箱测试、真实学习者或部署记录，不能靠新增合成数据“补齐”。

统一边界：接入 DeepSeek 不等于诊断准确率已建立；页面显示分数不等于真实学习效果；neural-v1 被运行时引用不等于其证据门禁通过；本机页面不等于所有处理离线。

## 2. 核心教学闭环

| 编号 | 验收问题 | 状态 | 当前交付与证据 | 边界或剩余工作 |
|---|---|---|---|---|
| A01 | 能否输入一个新的教学目标 | 已实现并测试 | Dashboard `api/start`；`goal.concept/objective/materials/success_thresholds/max_rounds`；`tests/test_teacher_agent_dashboard.py` | 目标由教师输入，系统不自动证明目标本身合理 |
| A02 | 能否像 Goal 模式一样拆分和跟踪目标 | 已实现并测试 | `teacher_agent_context.py::build_goal_plan` 生成前置、概念、过程、迁移 4 个可观察子目标及进度 | Goal 进度是内部控制量，不是学习成绩 |
| A03 | 能否输入学生画像和初始掌握状态 | 已实现并测试 | v2 会话 Schema、网页表单、`teacher_agent.py` | 应使用最小化、非身份信息 |
| A04 | 能否根据回答动态更新学生画像 | 已实现并测试 | 每个有效 DeepSeek 回合向 `student_profile.adaptive_observations` 追加 `candidate_unconfirmed`：回答质量、参与度、误解标签、下一重点、脱敏证据和置信度；`adaptive_summary` 汇总最新候选，最多保留 12 条；Schema 与 `tests/test_teacher_agent_live.py` 覆盖 | 低置信度/模型请求复核必须标记人工复核；候选不覆盖教师字段，fallback 不生成；尚无跨 session 长期画像 |
| A05 | 是否显式维护知识掌握情况 | 已实现并测试 | `knowledge_mastery` 的 prerequisite / conceptual / procedural / transfer 四维状态 | 归一化控制值不是校准后的真实掌握概率 |
| A06 | 是否显式维护误解或错误模式 | 已实现并测试 | `misconceptions` 支持发现、重复观察和 active / resolved 生命周期 | 模型诊断需看置信度和证据，不能写成确定心理事实 |
| A07 | 是否显式维护当前轮理解信号 | 已实现并测试 | 在线 DeepSeek 诊断与规则降级均留下来源；状态含 confidence 和 evidence | 降级信号不是 DeepSeek 诊断；页面必须区分来源 |
| A08 | 是否显式维护下一步教学关注点 | 已实现并测试 | `student_state.next_focus` 和 Goal `active_step` | 这是策略输出，不是专家最优性证明 |
| A09 | 是否管理历史对话 | 已实现并测试 | 单一六层 `teaching_context`；每轮活动知识点 → 同关注维度 → 最近轮次；最多 6 个相关回合、默认 14000 字符；较早历史保留确定性聚合与原文抽取检查点；超长合法会话 fail-safe 裁剪 | 不是无限记忆，也没有跨 session 长期记忆；候选画像只是低权重假设 |
| A10 | 是否每轮只生成一个下一教学动作 | 已实现并测试 | `one_action_per_turn`、`wait_for_student_before_next_action`；模型输出契约和控制器共同约束 | “实时”是请求—响应闭环，不是音视频流式感知 |
| A11 | 是否从 Skill Library 自动选主 Skill | 已实现并测试 | DeepSeek 提议 + 主 Skill 白名单校验；记录候选、选择理由和来源 | 开放场景路由最优性待专家锁箱验证 |
| A12 | 是否能在一轮组合多个 Skill | 已实现并测试 | 1 个主 Skill + 最多 2 个 support Skill；`composition_plan.one_action_contract=true` | 组合 Skill 不允许生成多个预写回合 |
| A13 | 是否根据自然语言反馈动态切换 Skill | 已实现并测试 | `teacher_agent_live.py` 每轮重新诊断和路由；测试覆盖纠错、手动覆盖、降级和终止 | 当前单轮开发 benchmark 不是完整多轮真实学生准确率 |
| A14 | 是否显示当前 Skill、切换和理由 | 已实现并测试 | Dashboard 展示主/支持 Skill、previous Skill、switch、reason、decision origin | 理由是短审计摘要，不展示 chain-of-thought |
| A15 | 是否支持自动与手动切换 | 已实现并测试 | DeepSeek 在线模式支持 `/+skill 名称`、`/auto`、`/stop`；未知或非主 Skill 拒绝 | 人工覆盖单独标记，不计作 Agent 自动命中；确定性展示模式不支持手动 stop |
| A16 | 是否能成功停止或安全转人工 | 已实现并测试 | 成功阈值、无活跃误解、最少轮数、连续无进展、最大轮数、受约束停止建议和人工停止 | 内部 `succeeded` 不等于真实学习效果已建立 |
| A17 | 重复提交是否会错误推进多轮 | 已实现并测试 | step 使用 `session_id + expected_round + idempotency_key`；start 使用独立幂等键且替换须匹配当前 session；同请求重放返回缓存，不重复执行；旧轮次、冲突 key 和第二标签页覆盖均 fail-closed | 当前仍是单进程、单会话本机服务，不是分布式事务系统 |

## 3. DeepSeek V4 Flash 与控制层

| 编号 | 验收问题 | 状态 | 当前交付与证据 | 边界或剩余工作 |
|---|---|---|---|---|
| B01 | 语义与生成骨干是否使用指定模型 | 已实现并测试 | `deepseek_client.py` 默认 `deepseek-v4-flash`；在线会话记录 provider/model/trace | 确定性控制器仍负责安全边界；这不是缺少 Agent，而是受约束 Agent 设计 |
| B02 | 是否一次完成诊断、路由和当前行动 | 已实现并测试 | `teaching_agent_assess_route_act_v2_layered_context` JSON 契约；`teacher_agent_live.py` | 只允许简短可审计理由，不请求或公开思维链 |
| B03 | 模型能否新增或越权选择 Skill | 已实现并测试 | 未知 Skill、support 充当主 Skill、超过 `max_repeat` 均拒绝 | Skill Library 变更必须单独审查来源和契约 |
| B04 | 能否避免直接泄露最终答案 | 部分实现 / 受限 | 提示契约和明显最终答案模式拦截 | 正则不能证明所有学科、所有表达都不会泄露，仍需行为评测与人工抽查 |
| B05 | 模型停止建议是否受控制 | 已实现并测试 | 只有 human-review 建议且至少两轮无进展时才采纳；硬终止由控制器决定 | 模型不能自行改写终止状态 |
| B06 | API 异常是否可恢复 | 已实现并测试 | 超时/重试、响应大小限制、JSON 校验；默认可见规则降级，也可 `--no-rule-fallback` | 降级轮必须显示 `deterministic_safety_fallback`，不得冒充在线模型结果 |
| B07 | 是否显式授权远程处理学生文本 | 已实现并测试 | 必须传 `--allow-remote-student-data`；否则 fail-closed 或进入明确降级 | 真实学生数据还需伦理、告知和数据处理授权 |
| B08 | API Key 是否安全读取 | 已实现并测试 | `.private/deepseek_api.txt`（gitignored）或 `DEEPSEEK_API_KEY_FILE`；Key 不进入网页、trace 或公开 receipt | 只应链接/指向本机私有文件，禁止提交到 Git |
| B09 | 上下文是否最小化和受限 | 已实现并测试 | 单一 `teaching_context`；默认最多 6 个相关回合、14000 字符硬上限；当前回答单份；预算、截断和证据指针可审计；媒体不发送 | 发送的是教学文本与必要状态，不应笼统称“完全离线” |
| B10 | 是否完成隐私脱敏 | 部分实现 / 受限 | 常见邮箱、手机号、身份证号、URL 和本机路径模式替换；测试覆盖 | 这是模式级保护，不保证任意自由文本完全去标识化 |

## 4. Teaching Skill Library 与 neural-v1 来源

| 编号 | 验收问题 | 状态 | 当前交付与证据 | 边界或剩余工作 |
|---|---|---|---|---|
| C01 | 默认是否使用 v2 Skill Library | 已实现并测试 | `data/teacher_agent_skill_library_v2.json`；CLI 和 Dashboard 默认路径均为 v2 | 题目一的 v0 展示产物仍保留其原边界，不与 v2 混称 |
| C02 | Skill 数量和角色是否明确 | 已实现并测试 | 16 个 Skill：13 个主 Skill + 3 个 support Skill；Schema 和单测验证 | 不得继续使用旧文档的“8 主 + 2 支持”描述 |
| C03 | Skill 是否可执行 | 已实现并测试 | 每个 v2 Skill 含 applicable/contraindications、pre/postconditions、failure_transition、max_repeat、addition_reason；控制器硬校验 ID/角色、`applicable_signals`、纠错证据、`max_repeat`、单动作和停止条件 | 其余自然语言 pre/contra/post/failure 条件进入提示与审计，但尚非形式化规则；字段完备也不证明教学法有效 |
| C04 | 是否使用 neural-v1 | 已实现并测试 | v2 `derivation.general_skill_id=evidence_grounded_multimodal_teaching_neural_v1`；12 个操作包装 | 使用的是九环节/十三策略本体，不是已确认视频共识 |
| C05 | 新增 Skill 是否说明原因和场景 | 已实现并测试 | 检索式复习、自我解释、参与恢复、信心支持均有单独研究补充来源和 `addition_reason` | 研究补充不得标成课堂视频蒸馏结论 |
| C06 | neural-v1 证据门禁是否通过 | 外部验证待完成 | manifest 固定记录 `passed=false`、eligible 0、excluded 54、observed phase 0、observed consensus strategy 0 | 页面和报告必须显示 `provisional · evidence gate not passed` |
| C07 | neural-v1 识别准确率、跨课程泛化、教学效果是否建立 | 外部验证待完成 | manifest 中三项均为 `false` | 需要合格证据、独立标注、锁箱和真实学习实验 |

## 5. 评估证据

| 编号 | 验收问题 | 状态 | 当前交付与证据 | 边界或剩余工作 |
|---|---|---|---|---|
| D01 | 是否有自由文本诊断测试样例 | 已有开发证据 | 28 个作者构造中文一轮案例；4 领域，7 类各 4 例；冻结 JSON 与输入哈希 | 未经专家验证，且未在提示词开发后保持独立锁定 |
| D02 | 自由文本信号判断表现 | 已有开发证据 | v3 报告文件 / prompt v2：Accuracy 0.892857，Macro-F1 0.875325 | 仅 post-hoc development regression；真实诊断 Accuracy 未建立 |
| D03 | Skill 路由与切换表现 | 已有开发证据 | 允许主 Skill 命中率 0.785714；切换 F1 0.833333；终止 F1 1.000000 | Skill gold 为作者构造，不能写成部署路由质量 |
| D04 | 是否有弱基线 | 已有开发证据 | 常量 partial：Accuracy 0.142857/Macro-F1 0.035714；固定诊断 Skill 命中 0.285714 | 只用于开发 sanity check |
| D05 | 是否有 oracle 上界参照 | 已有开发证据 | 读取 gold signal 的结构化路由器命中 0.964286 | 它使用金标准信息，不是同条件基线 |
| D06 | 是否报告在线运行稳定性 | 已有开发证据 | 28 例失败率 0；P50 1531.827 ms，P95 2384.102 ms | 只代表一次运行环境，不是 SLA 或部署压测 |
| D07 | 是否有完整机制回归 | 已实现并测试 | 4 条结构化合成轨迹；3 成功、1 安全转人工 | 使用 gold structured signals，不检验自由文本诊断 |
| D08 | 是否比较固定单 Skill 基线 | 已实现并测试 | 自适应/固定内部增益 37.333250/20.416750，差 +16.916500；迁移通过率 0.75/0 | 内部模拟状态差，不是真实学生效果 |
| D09 | 状态、决策、行为、终止是否量化 | 已实现并测试 | 状态 1.0；允许决策 0.916667；行为 1.0；多 Skill 案例率 1.0；终止 1.0 | 全部是 4 例机制回归指标 |
| D10 | 是否有学习效果计算接口 | 已实现并测试 | 前/后测、迁移测、可选延迟测；输出绝对/归一化增益、迁移和保持率 | 输入分数由 fixture/教师/授权记录提供，系统不独立核验评分正确性 |
| D11 | 随附学习结果是什么 | 已实现并测试 | 作者演示：0.4→0.8，绝对增益 0.4，归一化增益 0.666667，迁移 0.666667 | 不是实人数据，不建立因果学习效果 |
| D12 | 是否完成专家教学质量盲评 | 外部验证待完成 | 已有量表与建议协议，尚无独立双人标注结果 | 至少需双人独立标注、分歧裁决和一致性指标 |
| D13 | 是否完成锁箱/跨 session 测试 | 外部验证待完成 | 当前 28 例不是 prompt 开发后的 held-out lockbox | 应按 learner/session/problem/KC 分组，开发后锁定 |
| D14 | 是否完成真实学习者效果实验 | 外部验证待完成 | 学习结果 Schema 和计算器已准备 | 需预先方案、对照条件、前后测/迁移或延迟测和合规授权 |
| D15 | 是否完成部署质量验证 | 外部验证待完成 | 本机单会话演示与一次延迟 receipt | 尚无并发、多 session、接管率、故障率和长期跨课程部署报告 |

## 6. 前端、运行和公开交付

| 编号 | 验收问题 | 状态 | 当前交付与证据 | 边界或剩余工作 |
|---|---|---|---|---|
| E01 | 是否有可现场操作的网页 | 已实现并测试 | 学生对话优先的三栏工作台：目标/阶段、连续对话与固定输入框、可折叠状态/Skill/记忆检查器；真实 start/step API 测试 | 不是静态截图，也不是公网多用户服务 |
| E02 | 是否展示题目要求的五步演示 | 已实现并测试 | 学习视图展示新目标、连续多轮、Skill 选择和动态切换；独立“实验 / 评估”视图展示样例、基线和结果 | 在线调用与冻结评估明确分区，不能把开发指标当当前学生成绩 |
| E03 | 是否有一键启动 | 已实现并测试 | `打开题目二教学Agent.command`；读取 `.private` 链接或 `DEEPSEEK_API_KEY_FILE` | 缺 Key 时应明确报错，不能伪造在线调用 |
| E04 | 本机服务是否受限 | 已实现并测试 | `127.0.0.1`、随机 capability URL、CSP、`no-store`、请求体上限、在线远程处理确认、轮次与幂等校验 | 本地端口不要转发到公网 |
| E05 | 会话如何保存 | 部分实现 / 受限 | 当前单个会话仅在服务内存保存，页面不持久化 | 不具备多用户隔离、数据库恢复或跨设备同步能力 |
| E06 | 是否可无 Key 自检 | 已实现并测试 | `teacher-agent-dashboard --check`；deterministic backend 可展示规则基线 | 离线模式必须显示不是 DeepSeek 在线 Agent |
| E07 | 是否有完整依赖和调用说明 | 已实现并测试 | README、本文、任务说明、CLI `--help`、Schema 与测试 | 在线运行会产生第三方 API 请求和费用 |
| E08 | 是否避免提交密钥和私有记录 | 已实现并测试 | `.private/` gitignore；公开 receipt 只有聚合指标和哈希；隐私审计 | 学生原始回答、API Key、真实媒体不应进入 Git、wheel 或公开 artifacts |

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
python3 -m pytest -q \
  tests/test_deepseek_client.py \
  tests/test_teacher_agent.py \
  tests/test_teacher_agent_context.py \
  tests/test_teacher_agent_live.py \
  tests/test_teacher_agent_benchmark.py \
  tests/test_teacher_agent_outcomes.py \
  tests/test_teacher_agent_dashboard.py
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
- fallback 轮明确标为 `deterministic_safety_fallback`；
- neural-v1 manifest 继续显示 gate `passed=false`，不得在展示层改成 final；
- 28 例 receipt 的 `held_out_after_prompt_development=false`，且诊断/路由/部署/学习效果声明均保持 `false`；
- 4 例结构化回归的 `free_text_answer_grading_established=false` 和 `simulated_pre_post_is_real_learning_effect=false`；
- 学习 outcome 报告的 `real_learner_effectiveness_established=false`；
- Key、真实学生回答和私有媒体不出现在 tracked files、网页静态资源、wheel 或公开 receipt。

## 8. 答辩可用表述

建议表述：

> 我们实现的是一个真正逐轮运行的混合 Teaching Agent。DeepSeek V4 Flash 负责理解学生自然语言、建议 Skill 和生成当前一个教师动作；确定性控制器负责显式学生状态、待确认候选画像、Skill 白名单、重复上限和终止安全。系统默认使用 v2 的 13 个主 Skill 与 3 个支持 Skill，并能显示每轮选择、组合、切换和理由。28 例作者构造开发集上的信号 Accuracy 为 0.892857，4 例结构化轨迹验证了闭环机制；两者都不替代专家锁箱、真实学生学习效果和部署验证。

禁止表述：

- “neural-v1 已经从 10 个视频识别出可靠通用 Skill”；
- “教师 Agent 的真实准确率是 89.29%”；
- “内部模拟增益 +16.9165 证明学生成绩提升”；
- “系统完全离线、学生文本不会离开本机”；
- “项目已经达到多用户部署标准”。
