# Teaching Agent 文献依据与设计映射

本文用于说明题目二的外部研究依据，并将可复用结论映射到本项目的实现与评估。文献中的实验结果仅代表其各自数据、模型和设置，不能直接作为本项目效果。

DeepSeek 的工程接入以官方 [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion)、[JSON Output 指南](https://api-docs.deepseek.com/guides/json_mode/) 和 [模型更新记录](https://api-docs.deepseek.com/updates) 为准。它们只说明调用契约和当前模型标识，不构成教学有效性证据。

## 1. 已核验来源与证据等级

| 工作 | 证据状态 | 论文与官方实现 | 对本项目的直接启发 |
|---|---|---|---|
| ScaffoldLM | ACL 2026，同行评审 | [论文](https://aclanthology.org/2026.acl-long.325/)；[代码](https://github.com/BNU-ERC-ITEA/ScaffoldLM) | 先形成分步教学计划，再循环执行“判断—行动—跟踪—记录”；显式保存当前步骤、进度、学生状态和历史。 |
| LongTutor | ACL 2026，同行评审 | [论文](https://aclanthology.org/2026.acl-long.1371/)；[代码](https://github.com/liano3/LongTutor) | 采用“证据—诊断—教学”链路；按知识点检索相关历史并生成统计摘要，避免把全部历史直接塞入上下文。 |
| Pedagogical Steering（StratL） | Findings of ACL 2025，同行评审 | [论文](https://aclanthology.org/2025.findings-acl.1348/) | 将学生状态识别、受约束的教学意图路由和单轮回复生成分离；允许一个主 Skill 搭配一个支持性策略。 |
| TRAVER / DICT | Findings of ACL 2025，同行评审 | [论文](https://aclanthology.org/2025.findings-acl.642/) | 在生成后增加逐轮验证器；终止由独立控制器决定，并设置最大轮数，避免模型过早结束或无限循环。 |
| MRBench | NAACL 2025，同行评审 | [论文](https://aclanthology.org/2025.naacl-long.57/)；[代码](https://github.com/kaushal0494/UnifyingAITutorEvaluation) | 从错误识别、错误定位、是否泄露答案、引导性、可执行性、连贯性、语气和自然度八个维度评价教学回复。 |
| MathTutorBench | EMNLP 2025，同行评审 | [论文](https://aclanthology.org/2025.emnlp-main.11/)；[代码](https://github.com/eth-lre/mathtutorbench) | 重点检查正确性、支架式引导、自我纠错和认知负荷；适合作为外部静态质量测试。 |
| EducationQ | ACL 2025，同行评审 | [论文](https://aclanthology.org/2025.acl-long.1576/) | 提供前测—多轮教学—后测的模拟评估范式，可计算绝对学习增益并观察负向影响。 |
| MathDial | Findings of EMNLP 2023，同行评审 | [论文](https://aclanthology.org/2023.findings-emnlp.372/) | 可作为公开多轮数学辅导对话数据参考；其教师与模拟学生对话不能替代真实学生实验。 |
| Confirming Correct, Missing the Rest | BEA 2026，同行评审 | [论文](https://aclanthology.org/2026.bea-1.56/) | 学生解法不应只分“对/错”，还应区分最优路径、合理替代路径和错误路径。 |
| LLM Knowledge Tracing | arXiv 2024，预印本 | [论文](https://arxiv.org/abs/2409.16490) | 支持按知识点维护掌握状态，而不是用单一总分描述学生。 |
| Tutor CoPilot | arXiv 2024，预印本 | [论文](https://arxiv.org/abs/2410.03017) | 仅用于参考真实场景研究设计；其现场实验结果不能外推到本项目。 |
| TutorBench / MMTutorBench | arXiv 2025，预印本 | [TutorBench](https://arxiv.org/abs/2510.02663)；[MMTutorBench](https://arxiv.org/abs/2510.23477) | 作为后续手写、图表和多模态输入评估的候选参考，不作为当前核心结论依据。 |

## 2. 推荐的系统链路

```text
教学目标与参考答案
        ↓
分步目标规划器：生成 2–7 个可观察、可验收的教学子目标
        ↓
相关历史检索：按知识点、错误类型和最近轮次提取证据与统计量
        ↓
学生状态估计器：DeepSeek 输出受 JSON Schema 约束的状态与证据指针
        ↓
Skill 路由器：规则约束 + 模型排序，选择一个主 Skill 和至多一个支持 Skill
        ↓
单轮教学动作生成：每次只输出当前一轮可实时交互的教学动作
        ↓
教学验证器：检查事实、泄露答案、目标一致性、难度、重复和安全性
        ↓
更新状态并判断：继续 / 切换 Skill / 完成 / 无法继续
```

该结构保留大模型对自然语言和复杂语境的处理能力，同时把状态、路由、终止和审计做成可验证的显式模块。DeepSeek API 是实现组件，不等同于完整 Agent；所有模型输出都应经过结构校验、超时重试和确定性降级。

## 3. 学生状态 Schema

| 字段组 | 建议字段 | 说明 |
|---|---|---|
| 教学计划 | `objectives`、`active_step`、`progress` | 每个子目标包含成功条件；只推进到有证据支持的步骤。 |
| 知识掌握 | `kc_id`、`mastery`、`recent_accuracy`、`attempts`、`hint_count`、`last_seen` | 逐知识点维护；`mastery` 是内部估计，不应表述为真实能力定论。 |
| 认知诊断 | `recall_failure`、`conceptual_gap`、`procedural_error`、`transfer_deficit` | 可多标签，但必须给出当前回答或历史记录中的证据。 |
| 当前轮状态 | `start`、`correct`、`comprehension`、`incorrect`、`confusion`、`question`、`irrelevant`、`end` | 用于触发本轮路由和终止判断。 |
| 错误与路径 | `wrong_method`、`algebraic_error`、`numerical_error`、`incomplete`、`ambiguous`；`optimal/valid_alternative/incorrect` | 避免把不同性质的问题都归为“回答错误”。 |
| 请求与情绪信号 | 请求解释/定理/计算；`motivation`、`confidence`、`frustration`、`uncertainty` | 只描述当前可观察信号，不生成永久性人格标签。 |
| 决策记录 | `next_focus`、`state_confidence`、`evidence_ids`、`primary_skill`、`support_skill`、`no_progress_count` | 每次 Skill 选择、切换和终止都应可追溯。 |

状态更新应遵循三条约束：没有证据不更新；低置信度时优先提问确认；历史摘要与原始证据 ID 同时保留，以便复核。

## 4. Skill Library 的补充原则

第一题蒸馏得到的 `v1 Skill` 继续作为课堂证据来源。依据研究新增的 Skill 必须单独标记为 `source: research_supplement`，不能描述成从第一题视频中蒸馏得到。

建议补充四类诊断对应的主 Skill：检索练习、概念澄清、分步支架、类比迁移；再加入自我纠错引导、追问下一步、提示而不泄露答案、回答学生反问后拉回目标、复述聚焦、总结终止等路由动作。鼓励信心、维持挑战、降低挫败等只能作为支持 Skill，不能替代知识教学。

每个可执行 Skill 至少包含：

```yaml
skill_id: stable_identifier
source: distilled_v1 | research_supplement
trigger: 适用状态与置信度条件
contraindications: 禁用条件
required_inputs: 所需目标、证据和学生状态
action_template: 本轮动作结构，不是预写死的答案
success_signal: 可观察的达标信号
failure_transition: 无效时切换到哪个 Skill
max_repeat: 连续使用上限
evidence_pointers: 来源或运行时证据 ID
```

自动模式应先用硬约束排除不适用 Skill，再由 DeepSeek 对候选项排序并输出理由。手动 `/skill_name` 只覆盖本轮选择，不应绕过安全、前置条件和终止规则。

## 5. 基线、消融与数据划分

至少比较以下方法，并保持相同底模、提示预算、温度、最大轮数和测试样例：

1. `Direct-LLM`：无显式状态、无 Skill，直接生成下一步教学动作。
2. `Fixed-Skill`：全程使用一个固定 Skill。
3. `Static-Sequence`：按预设顺序执行 Skill，不根据反馈切换。
4. `Rule-Only`：规则与模板驱动，不调用大模型做状态推断。
5. `Full-Agent`：计划、状态、检索、动态路由、验证器和终止器完整启用。

消融实验应分别移除教学计划、状态判断、相关历史统计、Skill 路由、验证器，并增加随机 Skill 路由作为负对照。训练、开发与测试必须按学生、session、题目或知识点分组切分，禁止把同一段对话的不同轮次随机分到训练集和测试集。

## 6. 可复现评估协议

### 6.1 状态与决策

- 两名标注者独立标注，分歧由第三人裁决；固定标签定义和示例。
- 状态判断报告 Macro-F1、各类别 F1、混淆矩阵和 Cohen's kappa；掌握概率可补充 Brier Score 或 ECE。
- Skill 选择报告 Top-1 Accuracy、Macro-F1；切换和终止分别报告 Precision、Recall、F1。
- 抽样核验选择理由、状态和 Skill 是否引用了真实且相关的证据 ID。

### 6.2 教学行为与过程

- 采用 MRBench 八维量表，并加入事实正确性、目标一致性、难度适配、重复度和安全性。
- 检查是否一次只做一个可响应动作、是否促进学生思考、是否过早给出答案。
- LLM-as-a-Judge 只能作辅助；需在人工标注子集上校准，并随机化成对比较的答案顺序。

### 6.3 教学效果

- 对低、中、高初始掌握度分别构建样例，覆盖概念缺口、步骤错误、迁移失败、连续答错、跑题和反问等情况。
- 采用前测—最多 8–12 轮教学—后测—未见迁移题；条件允许时再加入延迟测试。
- 报告后测成绩、绝对学习增益 `post - pre`、迁移正确率、限定轮数内成功率、平均轮数和负向增益比例。
- 模拟学生实验至少运行多个随机种子并报告均值与置信区间；关键比较使用配对 bootstrap 或随机化检验。
- 同时记录 P50/P95 延迟、API 失败率、重试次数和降级触发率，证明系统能够现场运行而非只离线生成脚本。

## 7. 结论边界

- 上述论文只证明其自身模型、数据集和评估设置下的结果，没有论文验证本项目或 `DeepSeek-v4-flash` 的实际表现。
- 静态 benchmark 得分不等于多轮教学成功；模拟学生的学习增益也不等于真实学生的因果学习效果。
- `distilled_v1` 与 `research_supplement` 必须分别记录来源，新增 Skill 不得伪装为视频证据蒸馏结果。
- 只有当前代码、冻结配置、测试集、运行日志、产物哈希和独立复核共同完成后，才能报告本项目的工程指标。
- “改善真实学习效果”必须通过真实学生、预先确定的方案和合适的对照实验验证；完成前只能表述为待验证假设。

## 8. 实施优先级

- **P0：可演示闭环。** 分步目标、显式状态、单轮动作、Skill 动态切换、验证器和终止器。
- **P1：可复现实验。** 固定测试样例、四类基线、模块消融、人工标注协议、日志与哈希。
- **P2：外部有效性。** 公开 benchmark、真实教师复核、真实学生前后测与延迟测试，以及多模态输入扩展。
