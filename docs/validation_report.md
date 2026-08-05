# 多模态真实性与有效性复核报告

复核日期：2026-07-27

## 结论

当前系统是**真实可运行的多模态工程原型**：MIT OCW 10/10 完整讲次已完成官方 WebVTT 来源、媒体与整段时间轴绑定；TeachObs 已完成 29/30 讲的字幕或审计 ASR 来源物化，以及固定六讲上的四臂探索性评估；OUC-CGE 与 DIPSER 还提供了两条用途不同的真实课堂识别 pilot。DIPSER V5 的严格因果同站点开发点估计为 0.8176；需要完整 held-out session 的离线 transductive 层级候选在一个固定 SGKF5 划分触及 0.9020。后者不是稳定跨 session、实时或部署 0.9。

| 要确认的主张 | 结论 | 证据 |
|---|---|---|
| FFmpeg/FFprobe/Tesseract 是否真的执行 | 是 | 8 秒媒体被探测、抽帧、静音检测和 OCR，产物含工具版本与媒体哈希 |
| 合成样例能否稳定复现 | 是 | 独立重跑得到同样的 1 个静音区间、4 个关键帧和 9 个融合事件 |
| 已知合成事件是否通过独立 fixture 对照 | 是 | 指定 3 类事件 F1=1.0，静音召回率=1.0，OCR 必需词召回率=1.0 |
| 合成演示是否验证了真实语音 ASR | 否 | 演示音轨是 440/660 Hz 音调加静音，转写由 fixture 提供 |
| TeachObs 六讲是否完成真实媒体 ASR 执行 | 是，仅建立执行与 provenance | 六讲哈希绑定媒体均完成离线 GPU ASR 并通过导入；没有独立人工参考，因此 WER 与内容准确率仍未建立 |
| 10 节 MIT 官方 WebVTT 的来源与时间轴覆盖是否核验 | 是 | MIT OCW 同域 WebVTT 10/10；页面配对、固定 SHA-256、FFprobe 媒体时长和 98.84%–99.72% 首尾 cue 时间轴覆盖均通过，公开 receipt 不含正文；这不等于字幕内容准确率已经独立审计 |
| 是否自动识别学生困惑/教师调整 | 否 | 两类事件来自 `anonymized_observations.json` 的人工合成标签 |
| TeachObs 四臂教学行为识别是否实际运行 | 是，探索性 | 29 讲/4,945 场景完成输入物化；固定六讲 1,099 场景上运行 transcript-only、+audio、+visual、full 四臂，公开测试已在修订前可访问，因此不是确认性或部署结果 |
| 九环节时序蒸馏是否运行 | 是，演示工程证据 | 10 份演示 transcript 各观察到 4–7 个环节；只有 `observed_method` 子序列按输入时间轴排序，未命中环节明确标为 `recommended_enrichment`；没有独立 phase gold，不建立环节识别准确率 |
| 是否运行真实课堂自动识别 | 是 | OUC-CGE 公开 sample 36 段，SHA-256 去重后 35 段，使用专家投入度标签做文件名伪分组折外预测 |
| 是否证明真实课堂事件准确率 | 否 | OUC-CGE 标注是三类小组投入度，不是提问、困惑、教师调整等事件级真值 |
| DIPSER 是否真实自动识别并严格留出 session | 是 | 296 个完整 pose+watch 样本；25 个 recording，外折 session/participant/sample 零交叉 |
| DIPSER 开发期 Accuracy 是否达到约 0.8 | 是 | SGKF5 逐窗 0.8041；因果 0.8176；逐 session 留一因果 0.8176，Macro-F1 0.6249 |
| 离线完整 session 点估计是否触及 0.9 | 是，但仅单一开发划分 | SGKF5 seed=2026 为 0.9020/0.7435；LOSO 0.8986；50-seed 均值 0.8883 |
| 是否确认跨 session 总体 / 部署准确率 | 否 | 层级候选是在查看同一数据后提出；固定 gate 的 LOCO/LOAO/双重阻断为 0.6554/0.8919/0.6351，且只有一个站点 |
| 是否证明生成 Skill 提高学习效果 | 否 | 没有真实学习者前后测、对照组或 A/B 实验 |

## 已复现的工程事实

- 媒体：8.000 秒、1280×720、10 fps，AAC 16 kHz 单声道。
- 已知静音：1.5–3.5 秒，FFmpeg 检测结果为 2.0 秒。
- 视觉内容：从“Concrete Example”幻灯片切换到含 `def solve(x)` 的代码幻灯片。
- OCR：标题和 `def solve` 等必需词被识别；`return x + 1` 仍有乱码，不能称为完整代码 OCR 正确。
- 幂等性：每次抽帧前清理本模块生成的旧 `scene_*.jpg` / `uniform_*.jpg`，避免残留帧污染审计。

可机读复核结果见 `artifacts/multimodal_demo/fixture_benchmark.json`。

## TeachObs 真实字幕/ASR 与四臂探索性实测

TeachObs 的输入来源优先级为 creator-provided 平台字幕、平台自动字幕、审计 ASR fallback。当前聚合 receipt 记录 19 讲 creator-provided、4 讲 automatic、6 讲 audited ASR，共覆盖 29/30 讲，唯一 pending 为 S4。六讲 ASR 的媒体、模型、运行时、解码和时间线 provenance 均已通过，但 `asr_is_official_caption=false`、`content_accuracy_established=false`、`word_error_rate_established=false`；“审计 ASR”不能写成“官方字幕”或“内容准确”。

论文对齐 profile 共 29 讲、4,945 个 15 秒场景，其中 23 讲、3,846 场景用于训练，固定六讲、1,099 场景用于测试，任务包含 39 个多标签教学行为。四臂测试结果如下：

| Arm | Micro-F1 | Macro-F1 | Hamming accuracy | Visual Macro-F1 |
|---|---:|---:|---:|---:|
| transcript-only | 0.594496 | 0.244338 | **0.818833** | 0.238622 |
| transcript + audio | **0.604708** | 0.247501 | 0.809967 | 0.244307 |
| transcript + visual | 0.548328 | 0.303894 | 0.775297 | 0.332851 |
| full | 0.543812 | **0.309604** | 0.763258 | **0.336426** |

相对 transcript-only，full 的 Macro-F1 增量为 `+0.065266`，六讲配对 cluster bootstrap 95% CI 为 `[+0.014886, +0.095637]`；Visual Macro-F1 增量为 `+0.097804`，CI 为 `[+0.035152, +0.114008]`。但 full 的 Micro-F1 与 Hamming accuracy 分别下降 `0.050684` 和 `0.055575`，所以只能说当前探索性结果改善了类别均衡和视觉类覆盖，不能说它普遍提高了“Accuracy”。公开测试标签在实现前可访问，修订又发生在首轮测试结果已经观察之后；因此 `confirmatory_multimodal_gain_established=false`、`external_lockbox_established=false`、`deployment_accuracy_established=false`、`learning_effectiveness_established=false` 和 `site_held_out_accuracy_established=false`。

公开、无字幕正文或逐样本预测的机读证据见 [`teachobs_asr_receipt.json`](../artifacts/public/teachobs_asr_receipt.json) 与 [`teachobs_multimodal_benchmark_receipt.json`](../artifacts/public/teachobs_multimodal_benchmark_receipt.json)。

## OUC-CGE 真实课堂 pilot 的实测结果

2026-07-20 使用 OUC-CGE 官方 OSF 三个公开 sample 包实际运行。36 个 MP4 全部可解码，`low/view1.mp4` 与 `low/view2.mp4` 完全重复，剔除后为 35 个唯一样本。按主视频流取中心 10 秒、12 帧、4 折 `StratifiedGroupKFold` 的折外结果如下：

| 模型或对照 | Accuracy | Macro-F1 | 解释 |
|---|---:|---:|---|
| 多数类基线 | 0.3429 | 0.1702 | 无信息基线 |
| 仅编码元数据 | 0.9714 | 0.9710 | 只看时长、码率、分辨率、编码器和流数量，不看课堂内容 |
| 视觉 | 1.0000 | 1.0000 | 公开样例内折外结果 |
| 音频 | 1.0000 | 1.0000 | 无效：low 0/11 有重叠音频，medium/high 均 12/12，模型可利用模态缺失 |
| 视觉 + 音频 | 1.0000 | 1.0000 | 无效：同一模态缺失捷径，且未超过视觉 |

视觉特征提取后随机置换标签 20 次，平均 Macro-F1 为 0.3060（范围 0.1093–0.6000）；它只是标签关联 sanity check，不能用来排除目标泄漏。仅编码元数据就达到 0.9710，强烈表明这些样例存在来源/编码混杂风险。进一步审计发现 low 没有与视频重叠的有效音频，而 medium/high 全部有，因此 audio/fusion 的 1.000 是无效捷径。本地 pilot 最终只选择 visual checkpoint 用于推理链路调试，并不将该 checkpoint 纳入公开 release。上述 1.000 均是**真实计算所得但不具备真实泛化解释的数值**，不能宣传为 100% 课堂识别率。完整机读结果见 `artifacts/real_classroom/benchmark_report.json`。

## DIPSER V5 的优化实测

DIPSER 链路使用发布方从真实 RGB 课堂图像派生的 30 维 head/body pose metadata，以及 75 维心率、加速度、陀螺仪、旋转向量和光照统计。它预测四位 publisher expert 至少 3/4 共识的 attention low/medium/high，不读取学生自评，也不把身份、路径、文件名、标签或时间作为模型特征。

| 设计 / 推理模式 | Accuracy | Macro-F1 | 判定 |
|---|---:|---:|---|
| 折内多数类 medium | 0.7568 | 0.2872 | 必须超过的无信息对照 |
| Session SGKF5，Fusion linear-SVC 逐窗 | 0.8041 | 0.6362 | 达到开发期目标 |
| Session SGKF5，训练折内选择因果窗口 | 0.8176 | 0.6357 | 只用当前/历史，gap>60s 重置 |
| Leave-one-session-out，训练折内选择因果窗口 | 0.8176 | 0.6249 | 25/25 session 完整 OOF |
| 离线整段 participant-recording pooling | 0.8480 | 0.6631 | 需要完整序列，不能称为实时 |
| 离线层级完整 session，SGKF5 seed=2026 | 0.9020 | 0.7435 | 267/296；offline transductive，事后候选 |
| 离线层级完整 session，LOSO | 0.8986 | 0.7397 | 266/296；不依赖 outer 随机划分 |
| 离线层级 50-seed 均值 | 0.8883 | 0.7172 | 仅 21/50 个 seed 达到 0.9 |

因果 SGKF5 混淆矩阵为 `[[12,12,2],[6,212,6],[3,25,18]]`，并非恒预测 medium；low/medium/high 都产生了正确识别。逐样本审计确认每个历史只来自同一 `session_id + participant_id`，时间不晚于当前样本，超过 60 秒即重置。逐 session 留一也不依赖随机外折，因此 0.8176 是实际计算出的跨 recording 开发期点估计。

层级算法使用 balanced LR 的 participant-sequence 概率池化、在事后开发协议中固定的 linear-SVC low 确认，以及完整 classroom session 的 median/IQR RBF-SVC 上下文；每个 inner fold 都从头拟合 scaler/模型，outer-test 标签不参与 gate 选择。相同 nested 协议的 SGKF5 Visual/Sensor/Fusion 为 0.6284/0.7973/0.9020，观察到 Fusion 相对最佳单模态 +0.1047 Accuracy、+0.1624 Macro-F1，但这是同一事后开发数据上的现象，不是确认性多模态增益。

未来上下文反事实揭示了 0.902 的边界：排除当前预测行后 SGKF5 为 0.8547/0.6603，只使用时间前缀且 gap>60 秒重置时为 0.7770/0.5822；完整模式有 253 行使用未来 sequence、271 行使用未来 session、274 行使用其他 participant 的上下文。固定层级 gate 的 LOCO、LOAO、cohort×activity 双重阻断为 0.6554、0.8919、0.6351。因而当前只能写成 `post_selection_exploratory offline_full_session_transductive`，所有 accuracy/multimodal/deployment `established` 均为 false。

公开、无逐样本记录的 DIPSER 0.9 聚合证据见 [`dipser_0_9_aggregate_summary.json`](../artifacts/public/dipser_0_9_aggregate_summary.json)。逐折、逐样本、未来上下文反事实和优化明细由本地 runner 生成并保持在未发布的研究产物目录中。

## 九个规范教学环节的时序蒸馏边界

题目 4.2 列出的九个环节已逐条映射为结构化 `teaching_phase`。抽取器为每个环节寻找首次证据，只有 `origin=observed_method` 的子序列按输入 transcript 的证据时间排序，并保存命中线索、`evidence_id` 和 `observed_span`。没有观察到的环节会作为 `recommended_enrichment` 加入可执行 scaffold，其 `observed_span=null`，不声称该环节来自原教师，也不对其作教师时序主张。

当前 10 份离线演示 transcript 各观察到 4–7 个环节；这些演示文本是人工释义节选，时间戳为近似值。正式视频链路使用媒体绑定时间轴，但当前仍没有独立专家 phase gold、episode 级多 Skill 聚类或环节时间边界标注。因此这部分证明的是“九环节时序蒸馏的工程能力与 provenance 区分”，不是教学环节识别 Accuracy、Precision、Recall 或 F1。

## 本轮审计后修正

- 删除“只有场景切换 + 附近出现 code 词就推断代码逐行讲解”的弱规则；现在必须有实际 OCR 代码/公式证据。
- 严格验证事件类型、事件 ID、合法模态、媒体时长、置信度、策略映射和类型特定证据。
- Skill 引用还要精确匹配事件的 `evidence` 与 `confidence`，篡改载荷不再通过。
- 多模态分数限制在 0–100，且名称改为“内部证据一致性”，不再暗示识别准确率。
- 等待时长只计算问题结束之后的静音部分。
- 场景切换、公式可见、学生困惑等弱现象不再直接映射为过强的教学策略。
- 匿名课堂观察增加来源标记、常见身份字段和联系方式检查。

## 外部证据当前状态

| 事项 | 当前状态 | 尚缺的外部证据 |
|---|---|---|
| TeachObs 双人独立标注 | 两份不同盲化顺序的任务表已生成；覆盖 4,945 场景和 39 个 code；完整 operational definition 为 0/39，验证标注者为 0，完成标注行数为 0，`human_completion=false` | 外部 39/39 operational codebook、两名真实独立标注者、完整标签、κ 与第三方裁决 |
| TeachObs 新站点锁箱 | 预注册草案与四臂冻结绑定协议已生成；`status=pending_external_registration`、`results_computed=false`、`external_lockbox_established=false` | 新目标站点数据、外部登记与可信公钥、一次性消费证据、最终结果签名 |
| 学习效果 | teacher/classroom cluster RCT 的模板、分配与 fail-closed 分析工具已实现；`learner_effectiveness_established=false` | 真实伦理批准与知情同意/获批豁免、真实参与者、盲化前后测、cluster-aware 结果与外部签名 |

这些状态说明“协议或工具完成”不等于“外部研究完成”。工程 acceptance 也不能把任一 `established=false` 自动改成正向结论。

## 若要确认“真实有效”还缺什么

1. 选取有授权的真实教学视频，并由至少两名标注者独立标注问题、等待、板书/幻灯片变化、代码讲解和教师调整。
2. 划分互不重叠的开发集与测试集，报告逐事件 Precision、Recall、F1、时间边界误差和标注者一致性。
3. TeachObs 已完成公开测试上的探索性四臂消融；下一步须在冻结 pipeline 后的新目标站点进行一次性、配对、cluster-aware 的确认性四臂评测。
4. 用真实学习者进行前后测或对照实验，报告学习增益、置信区间和失败案例。
5. 将当前 DIPSER 模型、因果窗口、阈值和主指标冻结，在未参与开发的新 cohort/session 或外部学校上只运行一次锁箱测试；按 session 聚类报告 Accuracy、Macro-F1、逐类 recall 与预设置信区间门槛。

在完成以上步骤前，准确表述是：**“完成了完整视频多模态工程链、可复现合成验证、TeachObs 29 讲固定六讲四臂探索性识别，以及 OUC-CGE 与 DIPSER 真实课堂 pilot；DIPSER 严格因果开发 Accuracy 为 0.8176，完整 session 离线 transductive 候选在一个固定划分触及 0.9020，但 TeachObs 仍是 post-test exploratory，DIPSER 的 LOSO、跨 seed 与跨 cohort 证据也不足以建立真实部署 0.9；双人复核、确认性多模态增益和教学效果尚未完成。”**

## 2026-07-27 工程交付机制与验收边界

当前代码已提供以下可重跑验收机制；具体测试数量、wheel 字节数、SHA-256 和成员数必须以**最后一次代码变更后**的 `sh scripts/verify_project.sh` 输出为准，不能沿用旧的固定数字：

正式候选使用 `sh scripts/build_release_acceptance.sh`。它只在全量项目验收、双构建字节一致、最终 wheel 隔离验收和两次 release audit 全部成功后生成 `artifacts/release_acceptance_1.2.0.json`；JSON 直接重算 exact wheel 与公开目录，并要求项目/最终 wheel receipt 同时绑定当前 verification-scope 摘要和同一 wheel SHA-256。旧 acceptance 不能作为输入，因而不能把陈旧测试数或未经本轮证明的研究状态带入新版本。

- 测试、compileall、环境诊断、demo、delivery、wheel 构建、wheel 发布审计，以及临时虚拟环境中的非 editable 安装与仓库外 smoke；
- 当 TeachObs 私有 media/caption 输入存在时，最终项目验收会额外重验 annotation/caption/ASR 公开 receipt 对当前私有输入的哈希绑定；若四臂 benchmark 产物存在，还会强制同时验证 result、public receipt 与无 pickle 冻结 bundle。本次证据文件 SHA 组写入项目 receipt，并在最终 acceptance 生成前重算比较；隐私 audit 通过不能替代这项证据新鲜度与构建期竞态检查；
- `tsm demo` 保持 `human_validation_status=incomplete`、`research_validation_complete=false`，且不会覆盖已有人工评分表；新模板逐行绑定完整 canonical Skill fingerprint，内容变化后的旧评分不能复用；
- DIPSER runner 从 records、feature names、提取器 provenance 和精确 float64 matrices 重算严格 fingerprint；
- `extract-strict-features` 从私有原始媒体/传感器生成严格 bundle，绑定 raw hash/window、sample order、代码/配置/环境/工具 provenance，并明确不执行预测或建立准确率；
- checkpoint v3 绑定 identity contract、claim cluster、训练 strict feature bundle、类 schema、模型参数和 ClaimContract；
- 外部锁箱绑定可重算 coverage denominator/evidence，要求外部 Ed25519 登记签名、预先固定的可信公钥、一次性 ledger 与最终签名 receipt；
- 确认性多模态增益与真实学习者效果可分别接入 externally governed aggregate evidence manifest；delivery 会重算研究设计、coverage、效应区间/显著性等 gate，并要求各自的外部签名与可信公钥，当前仓库不附正向证据；
- 真实媒体与逐样本产物目录使用 `0700/0600`，公开研究结果只允许经 aggregate exporter 和 release audit 输出。

本目录当前是 Git checkout；最终 acceptance 已记录 `git ls-files` tracked-file 隐私扫描执行并通过，因此本地发布边界不再只是“配置存在”。截至本次复核，最近一次已提交的 HEAD `13e12c172ae3e81b12a6153c8d45f5d0cc2451b4` 与 `origin/main` 一致，且对应的 [GitHub Actions run 30209305856](https://github.com/deafenken/multimodal-teaching-skill-miner/actions/runs/30209305856) 已成功；该 run 不覆盖此后尚未提交或尚未重新运行 CI 的变化。远端只读核验也尚未写入 acceptance，机器字段仍为 `remote_github_actions_run_verified=false`，因此不能把本地 receipt 本身描述成对远端 run 的密码学绑定。`tsm release-audit` 是发布边界的失败关闭检查，也不是对任意数据的完整去标识或不可重识别证明。

工程状态应写为 `engineering_ready_external_validation_pending`，前提是本次最终验收命令全部通过。10/10 MIT OCW 官方字幕的来源与完整性已由独立正式 manifest 建立，但这仍不包括真实双人复核、学习效果、确认性多模态增益或部署 0.9；签名有效也只证明内容完整性和密钥控制，不自动证明签署者独立。
