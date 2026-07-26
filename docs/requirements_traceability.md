# 题目要求—实现—证据追踪矩阵

状态含义：`完成` 表示本地工程验收闭环；`部分完成` 表示真实运行但证据范围有限；`外部待完成` 表示不能由代码或伪造数据补齐。

| 题目要求 | 状态 | 实现与证据 | 验收入口 |
|---|:---:|---|---|
| 2 门课程、每门至少 5 节 | 完成（演示） | `data/dataset_manifest.json`、10 份合法 transcript | `tsm audit`、`tsm verify-delivery` |
| 10 节完整正式视频转写 | 完成 | 默认离线 demo 仍是释义节选；独立私有数据集已从 10/10 MIT OCW 官方 WebVTT 生成，页面/VTT/媒体时长/哈希绑定，`formal_empirical_ready=true`，全文不进入 wheel | `tsm fetch-formal-captions`、`tsm audit --require-formal`、`artifacts/public/formal_caption_retrieval_receipt.json` |
| 10 节完整真实媒体 | 完成工程证据 | 10 个 MIT OCW 完整 MP4 已私有下载、本地 SHA-256 与 FFprobe 验证；共 1,038,813,006 bytes、26,656.83 秒，上游未提供固定媒体哈希，原视频不进入 wheel | `tsm fetch-full-videos`、`artifacts/private/full_videos/media_manifest.json`、`artifacts/public/full_video_validation_receipt.json` |
| 视频/字幕预处理 | 完成 | `preprocess.py`，流式哈希、ASR 版本/模型 provenance | `tsm preprocess` |
| TeachObs 字幕与审计 ASR 缺口 | 论文 profile 完成技术覆盖；完整 profile 部分完成 | 平台字幕已审计 23/30 讲、34 轨（creator-provided 19、automatic 15），平均时间轴覆盖 0.985841；六讲离线 GPU ASR 已按 v2/v4/v4 契约全部通过导入。当前优先级 matrix 为 creator-provided 19 + automatic 4 + audited ASR 6 = 29/30 讲，唯一 pending 为 S4，满足论文 profile 但不满足完整 30 讲 profile。job/result 逐讲绑定媒体 SHA/时长、固定 Whisper revision/模型树/解码/runtime，并采用完整媒体单次输入、最小 0.90 first-to-last VAD 语音锚点 span、两端各自最大 0.10 媒体时长的对称相对空白；旧证据失败关闭且不迁移。权威状态仍以当前私有输入重算并绑定哈希的 `teachobs_asr_receipt.json` 为准；技术审计不能把 ASR 写成官方字幕，也不能建立内容准确率或 WER | `fetch/audit-teachobs-captions`、`prepare-teachobs-asr-handoff`、`import-teachobs-asr-results`、`docs/teachobs_audited_asr_handoff.md`、`artifacts/public/teachobs_caption_receipt.json`、`artifacts/public/teachobs_asr_receipt.json` |
| 完整视频多模态处理 | 完成工程链路 | v9/v2 对 10/10 讲完成全时间轴抽帧和字幕—媒体对齐；2553 帧、2441 非空 OCR 帧、1964 帧至少 3 个接受词、40191 接受词、1974 视觉事件、2270 融合事件；输出数量不等于识别正确数 | `tsm multimodal-longform-dataset`、`dataset_manifest.semantic.json`、`data_audit.semantic.json` |
| 视觉语义特征 | 完成工程链路 | wheel `visual` extra 提供单帧/数据集/回绑三个入口；CLIP 对 2553/2553 哈希绑定帧完成 512 维嵌入与固定八类相对 prompt 分数；模型 revision/权重 SHA/CPU 环境可追溯；没有人工类别真值，Accuracy 未建立 | `tsm visual-semantic-extract`、`tsm visual-semantic-dataset`、`tsm visual-semantic-apply`、`semantic_batch_receipt.json` |
| 四臂多模态消融 | 完成内部配对分析 | 10 讲相同 transcript segments；transcript-only / +audio / +visual / full 内部分数 100.00/100.00/99.92/99.94，保留事件 0/237/2033/2270；不是准确率、独立 Skill 质量或因果增益 | `tsm multimodal-ablation`、`ablation/ablation_report.json` |
| TeachObs 论文交集四臂 F1 | 完成正式探索性实跑 | `paper_track1_23_train_6_test` 固定 23 讲/3,846 训练场景、六讲/1,099 测试场景，媒体/特征 gate 为 29 讲/4,945 场景。训练内 5 折 lesson-grouped OOF 每折重拟合 TF-IDF/IDF/scaler并选择固定候选/阈值；正式 transcript/+audio/+visual/full Micro-F1 为 0.594496/0.604708/0.548328/0.543812，Hamming 为 0.818833/0.809967/0.775297/0.763258。full Macro-F1 比 text 高 0.065266（六讲 cluster 95% CI [0.014886,0.095637]），但 Micro/Hamming 下降；冻结 v2 JSON+NPZ 预测逐位一致。首轮公开测试已在修订前观察，故始终是 post-test exploratory，不是 0.9、确认性或部署结果 | `tsm benchmark-teachobs-multimodal --evaluation-profile paper_track1_23_train_6_test`、`scripts/run_teachobs_external_study.sh` |
| 多模态公开证据最小化 | 完成工程边界 | 从私有 manifest/audit/semantic/ablation 重算 aggregate-only receipts，保留来源哈希承诺，移除媒体、文本、帧、嵌入、逐讲记录和路径 | `scripts/build_multimodal_public_receipts.py`、`tsm release-audit artifacts/public` |
| 输入新视频的完整闭环 | 完成 | 可选同媒体哈希 ASR，或视频+官方/审计 transcript；随后真实 FFmpeg/OCR→多模态→Skill→教学→交互→评估，并机读区分 transcript 与 audio-verified speech | `tsm pipeline --transcript`、`run_multimodal_demo.sh`、`run_defense_demo.sh` |
| 10 讲完整研究一键复跑 | 完成工程入口 | 总 runner 串联字幕/视频、可恢复长视频处理、CLIP、审计、四臂消融、公开 receipts 与 release audit；默认 CPU，可显式选择授权 GPU | `scripts/run_full_video_multimodal_study.sh MODEL_DIR --acknowledge-source-terms [--reuse-downloads]` |
| 教学目标与 Bloom 层级 | 完成 | `miner.py::infer_bloom`、Skill schema | `tsm mine` |
| 至少 5 类教学策略 | 完成（演示数据） | observed-only 集合级覆盖，不含推荐模板 | `summary.json` |
| 方法来自视频证据 | 部分完成 | 稳定 `evidence_id`；步骤区分 observed/recommended | `evaluate_skill.method_provenance` |
| 长视频 episode 多 Skill 蒸馏 | 部分完成 | 完整视频已按 chunk 可恢复处理并生成全程事件；当前仍每视频一个主 Skill，episode 聚类/多 Skill 和专家 gold-set 待完成 | `longform_multimodal.py`、`docs/project_status.md` |
| 明确触发条件与前提 | 完成 | `trigger`、`preconditions`、严格 schema | `tsm validate skill` |
| 可执行步骤与教师动作 | 完成 | `procedure`、`teacher_actions`、状态机 | `tsm teach`、`tsm interact` |
| 读取学生信号并动态调整 | 完成工程能力 | `runtime.py` 的 retry/fallback/advance | `interactive_demo.json` |
| 成功标准、失败模式 | 完成 | `success_criteria`、`failure_modes`、`verification` | schema + tests |
| 新任务测试 | 完成能力覆盖 | 6 个留出主题与静态基线 | `tsm benchmark` |
| 自动评估 | 完成内部一致性 | 结构、证据、执行、迁移、溯源；不等于学习效果 | `tsm evaluate` |
| 10/10 双人独立人工验证 | 外部待完成 | 覆盖与 canonical Skill fingerprint 审计完成，空模板保持 incomplete | `tsm human-evaluate --skills artifacts/skills` |
| 真实学习效果 | 外部待完成 | 没有前后测/对照组；已提供外部 manifest gate、system artifact 绑定与签名接入，未附正向证据 | `prepare-external-research-evidence`、伦理审批后的 RCT/cluster-RCT |
| 真实课堂自动识别 | 部分完成 | OUC pilot + DIPSER pose/watch | recognition 报告 |
| DIPSER 开发 Accuracy 约 0.8 | 完成开发估计 | 严格因果 0.8176 | `run_dipser_optimization.py` |
| 离线 0.9 挑战 | 完成探索估计 | SGKF5 0.9020；LOSO 0.8986；50-seed 0.8883 | `run_dipser_hierarchical_challenge.py` |
| 防止 artifact 篡改 | 完成 | records 与精确矩阵指纹重新计算 | `test_dipser_artifact_integrity.py` |
| 冻结外部评估 | 完成协议 | checkpoint v3、strict feature binding、ClaimContract、claim cluster、签名登记/receipt、一次性 ledger | `freeze-recognition-model`、`create-freeze-registration`、`evaluate-frozen-recognition` |
| 原始目标站点数据生成严格特征 | 完成工程能力 | raw hash/window、sample order、提取器代码/配置/环境/工具 provenance 绑定；只提取不预测 | `extract-strict-features`、`docs/raw_feature_bridge.md` |
| 确认性多模态增益 | 外部待完成 | 当前同一事后开发数据不允许确认性结论；已提供外部 manifest gate、system artifact 绑定与签名接入 | 新锁箱配对消融、`verify-external-research-evidence` |
| 跨学校/部署 Accuracy | 外部待完成 | 当前随附报告的 deployment flags 保持 false；delivery 仅接受完整签名证据链 | 前瞻一次性目标站点 lockbox |
| 环境与依赖自检 | 完成 | 资源、Python、工具、recognition、API 安全 | `tsm doctor` |
| wheel 非 editable 安装 | 完成验收入口 | 干净双构建字节一致、bundled data/schema/config/governance、核心隔离 smoke、recognition/签名入口及 exact release wheel 验收 | `verify_project.sh`、`build_release_wheel.sh`、`verify_release_wheel.sh` |
| 发布与隐私 | 完成工程边界 | private paths、0600/0700、aggregate export、release audit；当 TeachObs 私有 media/caption 输入可用时，最终验收还会校验 annotation/caption/ASR public receipts 对当前私有输入的哈希绑定；四臂 benchmark 一旦生成任一产物，result、public receipt 和冻结 JSON+NPZ bundle 必须同时通过 profile-aware checker。本次证据文件 SHA 写入项目 receipt，并在生成 acceptance 时重算，防止仅“隐私扫描通过”但语义已陈旧或构建期间改变的 receipt 进入发布；仍需人工披露风险审查 | `PRIVACY.md`、`tsm release-audit`、`scripts/verify_project.sh` |
| CI | 配置完成、远端待验证 | Python 3.10–3.13、测试、wheel、隐私 workflow 已定义；当前目录无 `.git`，不能核验 Git tracked 边界或远端 run | `.github/workflows/ci.yml` |

## 答辩时必须主动说明

1. 自动评分衡量结构、内部证据一致性、可执行性和能力覆盖，不等价于真实学习增益。
2. 内置 2×5 转写仍是离线释义演示；正式 10 讲链路使用私有官方字幕和完整媒体。字幕就绪、10/10 全程覆盖、2553/2553 CLIP 完成只证明来源、覆盖与工程执行，不解决事件真值、方法标签、学习效果或部署准确率。
3. OCR 接受词数、CLIP 相对 prompt 分数、视觉/融合事件数都不是正确数；没有独立 gold label 时不能报告真实 Accuracy/Precision/Recall/F1。
4. 四臂内部量表没有显示正向 Skill 分数增益，并且内部证据一致性不是独立质量评分；确认性多模态增益仍需冻结后新数据配对评测。
5. `observed_method` 才计入视频方法覆盖；`recommended_enrichment` 是系统脚手架。
6. `interact` 的达标信号由教师、上层 Agent 或独立判分器提供，运行时不冒充学科答案判分器。
7. DIPSER 的 0.902 是 `offline full-session transductive post-selection development estimate`，不是 raw-video、实时、单学生、跨学校或部署 0.9。
8. 只有 checkpoint v3 在新目标站点的一次性前瞻 lockbox 上通过全部 ClaimContract、coverage、逐类、claim-cluster 与签名登记 gate，并由预先固定的外部可信公钥验证最终 receipt，才能建立部署准确率；开发者自签名不构成独立锁箱。
