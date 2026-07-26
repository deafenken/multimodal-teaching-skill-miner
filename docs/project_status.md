# 项目完备性状态

## 已形成闭环的工程能力

| 能力 | 状态 | 验收入口 |
|---|:---:|---|
| 源码与 exact wheel 安装 | 完成工程链 | `build_release_acceptance.sh` 串联 `verify_project.sh` 双构建一致性、最终 clean wheel、exact 安装/视频 smoke、发布审计和机器生成 acceptance；当前 acceptance 状态为 `engineering_ready_external_validation_pending`，任何后续源码变更都必须重新生成 |
| 浏览器答辩看板 | 完成 | `tsm dashboard` 直接打开自包含 HTML；包含合成多模态时间轴、TeachObs 四臂消融、DIPSER 评估范围、可执行 Skill 和 claim boundary；只使用公开聚合数据，不包含真实私有媒体或逐样本记录 |
| 2 门课×5 节离线演示 | 完成 | `tsm demo` |
| 2 门课×5 节完整正式字幕 | 完成 | `tsm fetch-formal-captions` 实跑 10/10；官方页面/VTT/媒体时长/哈希绑定；私有 manifest 的 `formal_empirical_ready=true` |
| 2 门课×5 节完整 MIT OCW 媒体 | 完成 | 10/10 私有下载、本地 SHA-256、FFprobe 容器/音视频流/参考时长验证；1,038,813,006 bytes、26,656.83 秒；公开 receipt 不含媒体内容 |
| 视频/字幕或 ASR / Skill / 教学过程 / 状态机 | 完成 | `tsm pipeline --transcript` 提供无 Whisper 的真实视频闭环；同媒体哈希 ASR 仍可直接使用 |
| 完整视频音频、抽帧、OCR 与事件 | 完成工程链路 | v9/v2：10/10 全时间轴覆盖和字幕—媒体对齐；2553 帧、2441 非空 OCR 帧、1964 帧至少 3 词、40191 接受词、1974 视觉事件、2270 融合事件 |
| CLIP 视觉语义 | 完成工程链路 | wheel `visual` extra 和三个 `tsm visual-semantic-*` 入口；2553/2553 哈希绑定帧；512 维嵌入、八类相对 prompt 分数、模型 revision/权重 SHA/CPU 环境绑定；没有人工真值，不建立 Accuracy |
| transcript / +audio / +visual / full 消融 | 完成内部分析 | 10 讲完全相同 transcript segments，排除课堂观察；平均内部 Skill 分 100.00/100.00/99.92/99.94，不构成多模态增益 |
| 多模态公开聚合 receipts | 完成工程边界 | `full_multimodal_validation_receipt.json` 与 `multimodal_ablation_receipt.json` 只保留聚合、设计、结论边界和私有来源哈希承诺；发布仍需 release audit 与人工披露复核 |
| 10 讲端到端总 runner | 完成工程入口 | `run_full_video_multimodal_study.sh` 串联获取/验证、可恢复长视频处理、CLIP、审计、消融、公开 receipts 与 release audit；默认 CPU，可通过环境变量选择授权 GPU |
| TeachObs 外部人工标签 | 完成导入与审计 | commit-pinned v0.1：30 讲、5158 场景、39 个共识标签、官方 23/7 划分；来源报告 7 名独立编码员，但无逐编码员文件，不能重算 κ |
| TeachObs 旧发布文本 transcript-only 基线 | 完成探索性实跑 | repository 随附发布文本、1,312 个官方测试场景：Micro-F1 0.612968、Macro-F1 0.360044；它不是当前平台字幕/审计 ASR 物化文本的四臂结果，Hamming accuracy 0.826180 也不单独表述为 Accuracy |
| TeachObs 字幕与转写时间轴审计 | 论文 profile 已覆盖，完整 profile 部分完成 | 平台字幕审计为 23/30 讲、34 轨（creator-provided 19、automatic 15），平均时间轴覆盖 0.985841；其中 23 讲进入优先级 matrix，另六讲 GPU ASR 已全部通过技术导入，形成 creator-provided 19 + automatic 4 + audited ASR 6 = 29/30 讲，唯一 pending 为 S4。时间轴与 provenance 不是 WER，字幕/ASR 内容准确率仍未建立 |
| TeachObs 审计 ASR GPU 交接 | 完成 fail-closed 工程链，六讲已实跑导入 | `prepare/import-teachobs-asr-*` 与离线 GPU runner 固定媒体/模型/解码/runtime/segment；v2/v4/v4 契约对 hash-bound 完整媒体单次输入采用统一的 VAD 相对端点策略（span≥0.90、首尾各≤媒体时长 0.10、绝对秒数只诊断），旧证据失败关闭且整批重跑、不迁移。当前六讲均为有效结果；`teachobs_asr_receipt.json` 每次由当前 media manifest、caption audit、job/import audit 重算，其 aggregate 和 `handoff_status` 仍是权威状态。ASR 不是官方字幕，WER/内容准确率仍只能由独立人工参考建立 |
| TeachObs 双人独立标注包 | 完成 fail-closed 骨架 | `prepare/analyze-teachobs-double-annotation`；A/B 两份空标签表使用不同盲化顺序，论文 profile 为每人 4,945 行×39 标签，完整 profile 为 5,158 行×39；发布的 39 个定义全空，须外部 39/39 operational codebook 和真实两人回填后才计算 κ，当前 human_completion=false |
| TeachObs 完整视频四臂 F1 | 完成正式探索性实跑 | 29 讲/4,945 场景完成媒体、字幕/ASR、音频、OCR、CLIP 与四臂对齐；固定六讲 1,099 场景上，transcript/+audio/+visual/full 的 Micro-F1 为 0.594496/0.604708/0.548328/0.543812，Macro-F1 为 0.244338/0.247501/0.303894/0.309604，Hamming 为 0.818833/0.809967/0.775297/0.763258。full 相对 text 的 Macro-F1 增量 +0.065266（lesson-cluster 95% CI [0.014886,0.095637]），但 Micro/Hamming 明显下降。修订仅在 23 讲内做 5 折 lesson-grouped OOF 并记录 frozen v2 provenance；首轮公开测试结果已观察，因此是 post-test iterative exploratory，不是首次盲测、0.9、确认性或部署结果 |
| TeachObs 新站点确认性四臂预注册 | 完成 fail-closed 草案链 | 精确绑定 system/analysis/四臂模型文件 SHA、同样本主 contrast、39 标签 Macro-F1、classroom/session cluster、固定 2,000 次 bootstrap、coverage、前瞻/一次性与外部 Ed25519 登记门；本地四臂冻结包是否完整必须以草案中的传递校验字段和 checker 为准。在新目标数据、外部登记签名和一次性执行证据到位前，所有 established=false；公开 TeachObs 23/7 被机器规则排除 |
| 自动结构与证据一致性评估 | 完成 | `tsm evaluate` |
| observed method 与推荐脚手架区分 | 完成 | `method_provenance` |
| 人工复核覆盖与版本审计机制 | 完成工程能力 | 复核完成后强制 100% 覆盖、不同 reviewer、canonical `skill_fingerprint` 绑定；当前真实评分仍待完成 |
| DIPSER 数据/特征防篡改 | 完成 | runner 重算指纹 + tamper tests |
| checkpoint v3 与签名外部锁箱协议 | 完成 | 特征/ClaimContract/claim-cluster 绑定、Ed25519 登记、一次性 ledger、签名 receipt |
| 外部研究证据接入 | 完成协议 | 确认性多模态增益/真实学习效果聚合 manifest、重算 gate、system artifact 实算绑定、独立签名与各自可信公钥；当前无正向证据 |
| 学习效果 cluster RCT 工具 | 完成 fail-closed 工程链 | `prepare/analyze-learner-effect-study` 生成 teacher/classroom 预注册、仅哈希 token 分配与盲化 pre/post/retention 表，并执行固定 ITT ANCOVA 和 cluster bootstrap；模板、合成与未签名本地分析均保持 learner_effectiveness_established=false |
| 原始目标站点媒体/传感器到 strict feature bundle | 完成 | `extract-strict-features` 绑定 raw hashes/windows、代码、配置、运行时与工具 provenance；只提取不预测 |
| 环境诊断 | 完成 | `tsm doctor` |
| 发布隐私与证据新鲜度审计 | 完成工程检查 | `tsm release-audit` 检查公开边界；`verify_project.sh` 在 TeachObs 私有 media/caption 输入存在时，还必须重验 annotation/caption/ASR receipt 对当前私有输入的哈希绑定；若四臂 benchmark 任一产物存在，还要求 result、public receipt 和完整冻结 bundle 同时通过 profile-aware checker。项目 receipt 记录本次用到的全部私有/公开证据文件 SHA，最终 acceptance 在发布前再次重算比较，防止长构建期间的并发变化；这些都不等于完整去标识或不可重识别证明 |
| CI 与隔离 wheel smoke | 配置完成 | `.github/workflows/ci.yml`；当前目录不是 Git checkout，远端 tracked-file 与 Actions 实际执行待在真实 checkout 验证 |

## 项目内不能伪造完成的外部证据

| 事项 | 当前状态 | 完成条件 |
|---|:---:|---|
| 10/10 双人独立教学复核 | 待完成 | 真实复核者评分、意见、κ 与复评记录 |
| 生成 Skill 提高学习效果 | 未建立 | 伦理审批后的前后测或对照/A-B 实验；可在完成后接入签名外部 learner evidence |
| 确认性多模态增益 | 未建立 | 冻结 pipeline 后的新数据配对消融与置信区间；可在完成后接入签名 external multimodal evidence |
| 跨学校/部署准确率 | 未建立 | 独立目标站点、前瞻一次性锁箱、全部 claim gate 通过 |
| 可信部署 0.9 | 未建立 | 不能继续调同一开发集取得，必须由上述外部锁箱建立 |

当前准确定位是：**核心工程、10/10 正式字幕与完整媒体、TeachObs 29 讲媒体/字幕或审计 ASR/音频/视觉/OCR/CLIP、固定六讲四臂探索性识别指标和安全冻结均已真实运行、可复现和可审计；真实双人复核、因果学习效果、确认性多模态增益、跨站点与外部部署结论仍明确待真实证据完成。** `formal_empirical_ready=true` 只代表正式转写来源与覆盖门槛通过，`multimodal_empirical_ready=true` 只代表所需模态、全程覆盖、对齐和事件结构齐全；两者本身都不是 Accuracy。当前 TeachObs 数值也只是公开测试上的 post-test exploratory estimate。签名机制只验证由指定密钥签署的内容完整性，不能自动证明签署者独立；可信公钥必须来自开发团队之外的治理渠道。
