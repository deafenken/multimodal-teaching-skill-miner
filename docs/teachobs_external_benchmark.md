# TeachObs 真实课堂多模态外部基准

## 目标与证据边界

这条链路用于回答一个比“视频管线能否运行”更严格的问题：在相同的真实课堂场景、相同训练/测试讲次和相同 39 个标签上，音频与画面是否给 transcript-only 带来可量化的增量。

TeachObs v0.1 的发布数据包含 30 个完整课堂讲次、5,158 个 15 秒场景、39 个多标签教学行为，其中 20 个被发布方分为 visual、19 个分为 nonvisual。发布方的基础划分是 23 讲训练、7 讲测试。项目保留 `full_23_train_7_test` 作为完整发布划分，并新增与 arXiv:2605.30673v2 §4.1 Track 1 对齐的 `paper_track1_23_train_6_test`：训练 23 讲不变，测试精确使用 S2、S5、S19、S24、S28、S30；S4 因该论文文本/单帧比较中的 mid-frame attachment 口径不统一而在任何拟合、调参和指标之前排除。注释发布采用 CC BY 4.0；源视频仍受各自平台和权利人的原始条款约束，不随注释许可证重新授权。

这不是一次性外部锁箱：公开测试标签在本项目实现前已经可访问，而且固定阈值首轮测试结果在训练 OOF 修订算法实现前已经被观察。因此，论文对齐四臂结果只能称为 `provisional exploratory four-arm multimodal comparison on the publicly accessible TeachObs source-aligned published six-lesson Track 1 text/frame intersection`；完整 7 讲 profile 也仍只是探索性公开测试比较。训练 OOF 没有读取六讲标签做选择，但修订结果仍是 post-test iterative exploratory analysis，不能写成首次盲测。两者都不能建立部署准确率、确认性多模态增益、因果教学效果或跨学校泛化。

## 已审计的数据事实

固定 commit 为 `96c251ae09e79edd06a3a9bbaaa8b8f7fe99a15c`。本地审计得到：

| 项目 | 数值/状态 |
|---|---:|
| 讲次 | 30 |
| 训练/测试讲次 | 23 / 7 |
| 场景 | 5,158 |
| 完整发布 profile 的训练/测试场景 | 3,846 / 1,312 |
| 论文 Track 1 profile 的训练/测试场景 | 3,846 / 1,099 |
| 论文 Track 1 媒体/特征交集 | 29 讲 / 4,945 场景，唯一排除 S4 |
| 标签 | 39（visual 20 / nonvisual 19） |
| 正标签总数 | 42,756 |
| 来源论文报告的独立编码员 | 7 |
| 发布的逐编码员原始文件 | 无 |
| 可由本项目重新计算的编码员间一致性 | 否 |
| 发布的代码定义文本 | 0/39 非空 |
| `source` 元数据取值（训练 / 测试 / 交集） | 7 / 6 / 3 |
| canonical `site_id` / `teacher_id` / `classroom_id` | 均无 |
| source/site-held-out 数值敏感性分析 | 未计算 |

因此，发布的 consensus label 可作为外部人工共识标签使用，但只能准确命名为 provisional consensus gold。不能把“论文报告有 7 名编码员”改写成“本项目已经独立复算 κ”。

`lessons.csv` 的 `source` 是上游发布者、项目或采集来源字段，不是 canonical site ID。两个固定 profile 的训练、测试 `source` 集合都有 3 个重叠取值；代码会从 commit-pinned 元数据自动重算集合规模、交集和集合哈希，漂移时 fail closed。公开 receipt 只保留这些聚合计数、布尔边界及哈希，不包含逐讲映射。由于数据没有 canonical site、teacher 或 classroom 标识，项目明确保持 `site_disjointness_verified=false`、`teacher_disjointness_verified=false`、`classroom_disjointness_verified=false`、`site_held_out_accuracy_established=false`。本项目没有把异质的 `source` 值强行当作站点，也没有为此新增事后 F1 或置信区间。

## 可复现入口

先取得并审计 commit-pinned 注释：

```bash
python3 -m teaching_skill_miner fetch-teachobs \
  --output artifacts/private/external_datasets/teachobs/imported_annotations \
  --public-receipt artifacts/public/teachobs_annotation_receipt.json \
  --acknowledge-source-terms
```

运行旧发布文本的固定 transcript-only 兼容基线：

```bash
python3 -m teaching_skill_miner benchmark-teachobs-text \
  --repository artifacts/private/external_datasets/teachobs/repository \
  --output artifacts/private/external_datasets/teachobs/text_benchmark_result.json \
  --public-receipt artifacts/public/teachobs_text_benchmark_receipt.json
```

这组历史兼容结果如下。它使用 TeachObs repository 随附的发布文本，不是后续从当前媒体绑定的平台字幕/审计 ASR 物化出的四臂 transcript 输入。多标签类别不平衡明显，因此 Micro-F1 与 Macro-F1 是主指标；Hamming accuracy 不能单独写成“Accuracy”。

| 转写处理 | Micro-F1 | Macro-F1 | Hamming accuracy | Exact subset accuracy |
|---|---:|---:|---:|---:|
| 发布文本原样 | 0.612968 | 0.360044 | 0.826180 | 0.001524 |
| 重复 speaker-turn 敏感性处理 | 0.615083 | 0.361306 | 0.827431 | 0.000762 |

在这组旧发布文本兼容基线内，原样文本是主分析。敏感性检查发现训练集 185 个、测试集 52 个场景受重复 speaker-turn 清理影响；它没有改变该基线的结论。这些数值不得填入当前四臂结果或用来计算官方字幕/审计 ASR 的模态增益。

## 完整视频、字幕、音频与视觉特征

只下载 10 讲不足以运行任一固定监督实验：视觉/音频模型必须覆盖全部 23 个训练讲次。正式总 runner 默认使用论文 Track 1 交集，严格要求 29 讲且唯一缺失 S4；`full_23_train_7_test` 则严格要求全部 30 讲。任意训练讲次或 S2/S5/S19/S24/S28/S30 中任何一讲缺失都会失败。`--lesson-id` 仅用于下载或抽取 smoke test，不能冒充任一固定 profile。

媒体准备会：

1. 从 commit-pinned repository 生成含场景起止时间与源 URL 哈希的私有计划；
2. 断点下载完整视频，计算本地 SHA-256，并用 FFprobe 校验容器、音视频流和参考时长；
3. 在每个 15 秒场景的中点抽取一帧：完整 profile 为 5,158 帧；论文 Track 1 profile 为 4,945 帧；
4. 对每个场景计算音频覆盖、RMS、平均/峰值幅度、DC offset、过零率和静音比例；
5. 私下保存 OCR、图像统计量，以及相邻场景的 scene/slide/board-change 启发式事件；
6. 用固定 revision、固定权重哈希的 CLIP 一次加载并生成全部场景视觉向量；
7. 把每个音频行、帧、嵌入、媒体和官方 scene manifest 做逐级 SHA-256/身份绑定。

视觉证据 `v2` 将 `scene_change`、`slide_change`、`board_build_up` 和
`code_or_formula_visible` 四类全部写入 `event_type_counts`；各类计数之和必须
严格等于 `event_count`，并逐条重算核对 `events`。旧 `v1` 只统计前三类，不能
直接复用：流水线仅在媒体、任务、配置、场景、文件哈希和旧三类计数全部验证
后，才从已绑定的完整 `events` 重算四类计数并以原子写入升级为 `v2`；若帧任务
本身也发生升级则重新提取。任何当前缓存的计数篡改都会失败关闭。

```bash
PYTHONPATH=artifacts/private/tools/yt_dlp \
python3 scripts/run_teachobs_media_preparation.py \
  --stage download --jobs 4 --acknowledge-source-terms

PYTHONPATH=artifacts/private/tools/yt_dlp \
python3 scripts/run_teachobs_media_preparation.py \
  --stage features \
  --clip-model artifacts/private/models/openai_clip_vit_b32_fp16_3d74acf9 \
  --clip-source-revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --clip-device cpu --clip-batch-size 32 \
  --acknowledge-source-terms
```

字幕审计优先检索创作者提供的字幕，其次记录 YouTube 自动字幕；两者严格分开。最新安全模式实跑在 30 讲中完成 23 讲平台字幕时间轴审计，取得 34 条轨道（19 条由平台归类为 creator-provided、15 条为 automatic），34/34 轨首尾时间轴覆盖率不低于 0.90，均值为 0.985841；平台字幕通道的其余 7 讲保持 pending，随后其中 6 讲由下述独立 ASR 链补齐。发布的 5,158 个场景文本全部可读且非空，但启发式检查标记 54 个异常重复文件。15 秒文本与字幕 cue 的平均标准化 token/字符 overlap 为 0.290455/0.364663；这只是时间对齐诊断，不是 WER、字幕准确率或人工审计。因此平台字幕本身的 `formal_caption_timeline_audit_completed=false`，`caption_content_accuracy_established=false`、`word_error_rate_established=false`；ASR provenance 通过也不会把这些内容准确性字段改成 true。

字幕缺失课次走独立的可审计 ASR GPU handoff，而不是在字幕抓取命令中静默
fallback。job manifest 逐讲绑定私有媒体 SHA-256/FFprobe 时长、固定模型 ID、40 位
revision、模型树 SHA、精确 faster-whisper/ctranslate2 版本、解码与 coverage policy；
离线 GPU runner 禁止网络模型下载，导入器重新验全部绑定和 segment。私有 30 讲
matrix 固定优先平台 creator-provided、再平台 automatic、最后技术审计 ASR。当前
六讲 GPU ASR 结果已全部通过 v2/v4/v4 导入；matrix 由 19 讲 creator-provided、
4 讲 automatic 和 6 讲 audited-ASR fallback 组成，覆盖 29/30 讲，唯一 pending
为 S4。公开 receipt 因完整 30 讲分母仍未满足而保持 pending；ASR 不是官方字幕，
未经独立人工参考听写时 WER/内容准确率仍不成立。见
[`teachobs_audited_asr_handoff.md`](teachobs_audited_asr_handoff.md)。

## 固定四臂消融

```bash
python3 -m teaching_skill_miner materialize-teachobs-transcripts \
  --media-plan artifacts/private/external_datasets/teachobs/media/media_plan.json \
  --media-manifest artifacts/private/external_datasets/teachobs/media/media_manifest.json \
  --caption-audit artifacts/private/external_datasets/teachobs/captions/caption_audit.json \
  --asr-import-audit artifacts/private/external_datasets/teachobs/asr/import_audit.json \
  --coverage-matrix artifacts/private/external_datasets/teachobs/asr/transcript_coverage_matrix.json \
  --asr-job-manifest artifacts/private/external_datasets/teachobs/asr/job_manifest.json \
  --asr-results artifacts/private/external_datasets/teachobs/asr/results \
  --output artifacts/private/external_datasets/teachobs/materialized_transcripts \
  --public-receipt artifacts/public/teachobs_transcript_materialization_receipt.json

python3 -m teaching_skill_miner benchmark-teachobs-multimodal \
  --evaluation-profile paper_track1_23_train_6_test \
  --repository artifacts/private/external_datasets/teachobs/repository \
  --feature-manifest artifacts/private/external_datasets/teachobs/media/feature_manifest.json \
  --transcript-materialization-manifest artifacts/private/external_datasets/teachobs/materialized_transcripts/manifest.json \
  --output artifacts/private/external_datasets/teachobs/multimodal_benchmark_result.json \
  --public-receipt artifacts/public/teachobs_multimodal_benchmark_receipt.json \
  --frozen-model-output artifacts/private/external_datasets/teachobs/frozen_four_arm_models
```

物化器固定优先使用 creator-provided 平台字幕、再使用平台自动字幕、最后使用与
同一媒体 SHA-256 绑定且通过技术时间线审计的 ASR。它不读取标签，不以发布仓库
transcript 补空；每个空 scene 仍保持空字符串。benchmark 与冻结模型会绑定
manifest 文件哈希、canonical hash、4,945-scene 文本/样本顺序和统一的
`benchmark_input_fingerprint`。四个 arm 使用完全相同的场景与标签：

物化 manifest/公开 receipt 的 v2 时间轴契约不对某一讲硬编码容差。平台 cue 必须
与哈希绑定的官方 scene 总时间轴有正交集；有交集时只裁剪时间戳、完整保留 cue
文本，并记录裁剪条数、总秒数和最大秒数，零交集 cue 直接失败。ASR 的审计域是
完整媒体，而预测/标签域是官方 scene：若媒体尾段 ASR 完全落在 scene 域外，则按
时间戳显式排除，并记录来源条数、保留条数、排除条数/秒数及其仍位于同一
SHA-256 媒体时长内；跨越末端的 ASR 段只裁剪时间戳。两条路径都不以字幕文本、
发布标签或模型输出决定边界，且 `source_text_items_silently_dropped=0`。这些记录
只建立可复现的时间投影 provenance，不建立字幕内容准确率或 WER。

- `transcript_only`：字符 TF-IDF；
- `transcript_audio`：TF-IDF + 7 维场景音频统计；
- `transcript_visual`：transcript TF-IDF + 训练集拟合的 OCR 字/词 TF-IDF + 40 维图像、OCR 状态及相邻画面变化数值 + CLIP；
- `full`：transcript + 音频 + 上述全部视觉特征。

模型选择固定为 5 折 `GroupKFold`，group 是完整训练讲次。每一折都在该折训练讲次内重新拟合 transcript/OCR vocabulary、IDF 和 audio/visual/CLIP scaler；固定的原始媒体绑定特征数组可以复用，但验证讲次不会进入任何学习型预处理统计。每个 arm 比较两个预先声明的 `class_weight=balanced` logistic-regression 候选：`C=0.25` 与 `C=1.0`。每个候选先获得覆盖全部 3,846 个训练场景且每场景恰好一次的 OOF 概率，再从固定 `0.3/0.4/0.5/0.6/0.7` 网格按官方标签顺序做至多三轮逐标签坐标选择；主目标为 pooled OOF Micro-F1，依次以 Hamming accuracy、Macro-F1、距 0.5 最近和较低阈值打破平局。`C=1, threshold=0.5` 包含在候选/起点中，因此训练 OOF 主目标不会因选择过程低于该固定基线。

私有结果记录候选注册表、5 个 fold 的拟合/验证讲次、每折 preprocessing fit count 与 vocabulary hash、classifier seed、每候选 OOF 概率/预测 hash、阈值列表/hash 和选择指标；聚合公开 receipt 只保留无讲次 ID 的协议、候选、阈值 hash 与 OOF 聚合指标。六讲特征和标签均不参与选择；字段明确写为 `thresholds_frozen_before_current_revised_test_scoring=true`、`public_test_outcomes_previously_observed_before_revision=true` 和 `current_revision_is_post_test_exploratory=true`，不得解释成首次盲测。

论文 Track 1 profile 的四臂在完全相同的 1,099 个测试场景上计算 Micro-F1、Macro-F1、Exact subset accuracy、Hamming accuracy、visual/nonvisual Macro-F1，并以 6 个测试讲次为 cluster 做固定 seed 的 2,000 次配对 bootstrap；兼容 profile `full_23_train_7_test` 对应 1,312 个场景和 7 个 cluster。可选冻结输出为四个独立 v2 JSON manifest + 确定性 NPZ，并绑定 profile、选中 lesson/sample 顺序、dataset/profile fingerprint、选中 C、逐标签阈值及训练 OOF selection hash；加载时禁用 pickle，并重验数组、词表、标签、特征契约及上游哈希，跨 profile、未知 C 或网格外阈值都会失败。导出后会自动重载并验证四臂预测逐位一致，但这仍不是部署证据。

## 正式修订结果

正式物化覆盖 29 讲、4,945 场景，其中 4,509 个场景有文本、436 个为空；来源为 19 讲 creator-provided、4 讲 platform automatic、6 讲 audited ASR。平台 14,637 条 cue 全部保留，12 条仅裁剪越界结束时间（总 69.781 秒、最大 35.522 秒）；ASR 3,337 条保留 3,336 条，1 条处于同一媒体内但完全落在官方 scene 域外，显式排除 2.46 秒。所有边界决策不读取文本或标签，静默丢弃计数为 0。

训练 OOF 选择得到 transcript-only/+audio 使用 `C=1.0`，+visual/full 使用 `C=0.25`。正式结果与独立临时重跑的四臂指标、bootstrap 和训练选择记录完全一致；冻结重载预测逐位一致，概率最大绝对差为 0。

| Arm | 训练 OOF Micro-F1 | 测试 Micro-F1 | 测试 Macro-F1 | Hamming accuracy | Visual Macro-F1 |
|---|---:|---:|---:|---:|---:|
| transcript-only | 0.611918 | 0.594496 | 0.244338 | **0.818833** | 0.238622 |
| transcript + audio | 0.564436 | **0.604708** | 0.247501 | 0.809967 | 0.244307 |
| transcript + visual | 0.541563 | 0.548328 | 0.303894 | 0.775297 | 0.332851 |
| full | 0.516856 | 0.543812 | **0.309604** | 0.763258 | **0.336426** |

主要配对差值：

- `transcript_audio - transcript_only` 的 Micro-F1 为 `+0.010212`，95% CI `[-0.009991, +0.029059]`，区间跨 0。
- `full - transcript_only` 的 Macro-F1 为 `+0.065266`，95% CI `[+0.014886, +0.095637]`；Visual Macro-F1 为 `+0.097804`，95% CI `[+0.035152, +0.114008]`。
- 同一 `full - transcript_only` 的 Micro-F1 为 `-0.050684`，95% CI `[-0.065745, -0.040763]`；Hamming accuracy 为 `-0.055575`，95% CI `[-0.077503, -0.039182]`。

因此，多模态对类别均衡的 Macro-F1 和视觉类覆盖有探索性正增益，但没有对所有指标形成增益；最佳 Micro-F1 来自 +audio，最佳 Hamming 来自 transcript-only。任何单写“Accuracy=0.9”或“full 优于 text”的表述都不受这些结果支持。固定 0.5 阈值的首轮结果和四臂 bundle 原样保存在私有 `benchmark_attempts/fixed_threshold_v1/`；修订没有覆盖这份失败/退化证据。

场景级样本不是 1,099（或完整 profile 的 1,312）个独立课堂；不允许用场景级普通 bootstrap 制造过窄区间。即使探索性区间为正，也仍需一个从未用于开发、预注册且标签隐藏的新站点锁箱，才能建立确认性多模态增益。

## 双人独立标注、锁箱和学习效果

TeachObs 发布共识标签不能替代项目自己的双人事件/Skill 质量复核。实际流程必须满足：两名复核者分别收到不含 gold、模型预测和对方结果的盲化任务；先独立完成全部指定条目，再计算逐标签 κ、正例一致率和分歧；分歧由第三人裁决，原始 A/B 记录不能被覆盖。未收到真实外部标注文件前，项目只能生成任务表和校验工具，不能生成评分或签名。

一次性外部锁箱还必须更换新的 cohort/session 或学校，预先冻结模型、特征、阈值、主指标、claim cluster、最低覆盖与失败规则，并由开发团队之外的治理方控制标签和可信公钥。TeachObs 当前测试集不能在事后重新命名为锁箱。

学习效果是另一项研究问题。它需要伦理审批、知情同意、教师/班级层面的防污染随机化、预注册主要学习结果、盲化评分、缺失数据规则与 cluster-aware 分析。事件识别 F1 或生成 Skill 的内部一致性分数不能替代学生前后测或对照实验。
