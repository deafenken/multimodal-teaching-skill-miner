# 多模态教学行为识别设计

## 目标与证据边界

多模态模块不是给整段视频贴一个不可解释的标签，而是把正式字幕、音频活动、画面、OCR 和可选匿名课堂观察统一到同一时间轴，形成能回指媒体、帧和字幕片段的教学事件：

```text
官方字幕 / 同媒体 ASR ─┐
音频静音与等待 ───────┼─► 哈希绑定的时间轴事件 ─► Teaching Skill
均匀帧 / 场景帧 ──────┤
OCR / 像素变化 / CLIP ┤
匿名课堂观察（可选） ─┘
```

“检测到多少事件”“内部证据是否完全一致”和“识别正确率”是三件不同的事。当前完整视频链路建立了前两者的可运行、可追溯证据，没有独立事件 gold label，因此没有建立 OCR、CLIP 或事件检测 Accuracy/Precision/Recall/F1，也没有建立教学效果、因果多模态增益或部署准确率。

## 10 讲完整视频数据链路

当前正式数据由 MIT OpenCourseWare 两门课程各前 5 讲组成：MIT 18.06 Linear Algebra 与 MIT 6.0001 Introduction to Computer Science and Programming in Python。完整 MP4、官方 WebVTT 和全部逐帧衍生物保存在私有目录，不进入 wheel。

处理顺序如下：

1. `fetch-formal-captions` 固定课程页、WebVTT、字幕 SHA-256、页面所声明媒体和参考时长；
2. `fetch-full-videos` 通过受限的 Archive.org HTTPS URL 下载完整媒体，逐文件计算本地 SHA-256，并用 FFprobe 检查容器、音视频流和参考时长；
3. `multimodal-longform-dataset` 按 300 秒 chunk、2 秒解码重叠可恢复运行，对完整音轨执行 `silencedetect`，每 15 秒均匀抽帧，并在每个 chunk 保留最多 12 个场景变化候选；
4. 对每帧保存内容 SHA-256、时间戳、图像统计和 dHash，运行 Tesseract TSV OCR，并根据文字重合、增量和像素变化生成幻灯片/板书候选事件；
5. 将正式字幕片段、静音、帧、OCR 和视觉事件对齐，生成每讲的 `enriched_transcript.json`、`analysis.json` 和视觉语义任务 manifest；
6. 对全部哈希绑定帧运行 CLIP，将结果按帧路径和 SHA-256 重新绑定到语义版 transcript/analysis；
7. 对 10 讲执行 transcript-only、transcript+audio、transcript+visual/OCR、full 四臂配对内部消融。

最终数据声明 `teaching_skill_miner.longform_multimodal.v9` 与 `teaching_skill_miner.longform_extraction.v2`。extraction v2 按 TSV quoting 规则解析 Tesseract 输出，修复了双引号污染；修复后已从完整视频重新生成 10 讲 OCR/事件、重新回绑视觉语义并重新运行消融，旧 OCR 派生聚合不再作为证据。

完整研究 runner 为：

```bash
scripts/run_full_video_multimodal_study.sh \
  artifacts/private/models/openai_clip_vit_b32_fp16_3d74acf9 \
  --acknowledge-source-terms \
  --reuse-downloads
```

它依次执行正式字幕/完整视频（未指定 `--reuse-downloads` 时）、长视频分析、CLIP 数据集推理、语义回绑、正式审计、四臂消融、aggregate-only receipts 和 `artifacts/public` 发布审计。长视频 chunk 可在验证后恢复；视觉阶段默认 `TSM_VISUAL_DEVICE=cpu`、`TSM_VISUAL_BATCH_SIZE=16`，可显式改为 `cuda:0` 和合适 batch。设备选择不改变 evidence status，也不扩大结论边界。

关键产物：

- `artifacts/private/full_videos/media_manifest.json`：10 个本地完整媒体的私有 manifest；
- `artifacts/public/full_video_validation_receipt.json`：不含媒体、字幕正文、帧或本地路径的公开验证 receipt；
- `artifacts/private/full_multimodal/dataset_manifest.json`：长视频音频/抽帧/OCR/事件基础 manifest；
- `artifacts/private/full_multimodal/dataset_manifest.semantic.json`：绑定 CLIP 结果的语义版 manifest；
- `artifacts/private/full_multimodal/data_audit.semantic.json`：正式转写和多模态就绪审计；
- `artifacts/private/full_multimodal/semantic_results/semantic_batch_receipt.json`：10 讲 CLIP 批处理与权重 provenance；
- `artifacts/private/full_multimodal/ablation/ablation_report.json`：四臂配对内部消融。
- `artifacts/public/full_multimodal_validation_receipt.json`：不含媒体/文本/帧/嵌入/逐讲记录的聚合链路证据；
- `artifacts/public/multimodal_ablation_receipt.json`：不含逐讲数据的四臂设计、聚合内部指标与结论边界。

## 各模态的实现

### Transcript / Speech

- 保留 segment 的 `start/end/text`，检测提问、例子、比较、代码讲解等语言行为；
- 官方字幕标为 `transcript`，并记录来源页、WebVTT 哈希、覆盖证明和媒体绑定；
- 只有转写由当前媒体 ASR 产生且 provenance 绑定同一媒体 SHA-256 时才标为 `speech`；
- 本次 10 讲使用官方字幕，不是同媒体 ASR，所以虽然字幕—媒体时间轴 10/10 通过，`audio_content_verified=false`：这表示没有独立验证音频逐字内容，不能把页面绑定误写成 ASR 正确率。

### Audio

- 对完整音轨运行 FFmpeg `silencedetect`，保存静音开始、结束和持续时间；
- 将提问字幕后的静音融合为 `question_and_wait`，避免仅靠问号推断“教师等待”；
- 音频特征表示活动/静音时间证据，不做说话人身份识别，也不把静音检测写成语音内容验证。

### 全程抽帧、场景与板书变化

- 同时使用固定间隔均匀帧和场景变化帧，覆盖门槛检查视频起点、终点、每个 chunk、最大采样间隙和时间邻域；
- chunk checkpoint 包含输入/配置和输出绑定，可在验证后安全恢复；
- dHash 和图像统计用于识别画面变化；OCR 词重合、文字增量与连续时间关系用于生成 `slide_change` 和 `board_build_up`；
- 视觉事件还包括 `scene_change`、`code_or_formula_visible`、`visual_example` 与 `code_formula_walkthrough`。

### OCR

- Tesseract 对每帧分别尝试 PSM 6 与 PSM 11；
- 只接受置信度阈值不低于 35 的词，同时保存 raw/accepted 计数和平均引擎置信度；
- 引擎置信度没有在本数据上校准，接受词数不是正确词数；当前 `ocr_accuracy_established=false`。

### CLIP 视觉语义

- 使用 `openai/clip-vit-base-patch32` revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`；
- 本次本地 FP16 `model.safetensors` SHA-256 为 `676093550c9e05bc3ba55256c278c89f0d15a1a1585f3b81d76454c33b852d5e`；
- 对每个输入帧先核对路径约束和 SHA-256，再生成归一化 512 维嵌入、嵌入哈希和封闭 ontology 的相对 prompt 分数；
- ontology 为 `handwritten_blackboard`、`presentation_slide`、`programming_code`、`mathematical_formula`、`diagram_or_graph`、`instructor_talking`、`classroom_wide_view`、`other_lecture_visual`，每类使用 3 个固定 prompt；
- `2,553/2,553` 帧完成推理，最终设备为 CPU；GPU 服务器只用于公开模型权重处理，没有接收视频、帧、字幕、OCR 或其他私有项目数据；
- top label 和 softmax 相对分数只描述当前固定 prompt 集下的模型输出，不是经过人工真值校准的类别概率或识别准确率。

视觉语义能力随 wheel 的 `visual` extra 安装：`pip install 'teaching-skill-miner[visual]'`。单任务、整数据集和结果回绑入口分别为 `tsm visual-semantic-extract`、`tsm visual-semantic-dataset` 与 `tsm visual-semantic-apply`；原 `scripts/` 文件仅保留为兼容薄包装。三个入口共享相同的路径约束、帧哈希验证、模型 provenance 和 `complete_hash_bound_inference` 状态语义，CPU/GPU 选择不改变“未建立识别准确率”的结论。

### Classroom observation

- 接收人工或经过授权的上游模型产生的匿名事件；
- 支持 `student_confusion`、`student_answer`、`teacher_adjustment`；
- 明确拒绝 `student_name`、`face_id`、`identity` 等身份字段，当前系统不做人脸识别；
- 产物使用 `evidence_origin: provided_anonymized_annotation` 标记来源，不把人工标签表述为视频自动识别；
- 正式 10 讲四臂消融排除了 classroom observation，避免把外部标注泄漏到自动模态比较。

## 实跑结果

语义版 dataset manifest 的聚合结果为：

| 证据 | 结果 | 可解释为 |
|---|---:|---|
| 完整视频 | 10 个，`1,038,813,006` bytes | 本地媒体均已下载、哈希和 FFprobe 验证 |
| 总时长 | `26,656.83` 秒 / `7.404675` 小时 | 本地媒体时间轴总量 |
| 全程覆盖 | 10/10 | 均匀/场景抽帧通过覆盖门槛 |
| 字幕—媒体对齐 | 10/10 | 官方字幕来源与媒体时间轴绑定通过 |
| 帧 | 2,553 | 真实抽取并哈希绑定的分析帧 |
| 非空 OCR 帧 | 2,441 | 阈值后至少保留一个 OCR 词的帧数 |
| 至少 3 词 OCR 帧 | 1,964 | 阈值后至少保留三个 OCR 词的帧数 |
| 接受 OCR 词 | 40,191 | 引擎阈值后的输出量，不是正确词数 |
| CLIP 语义帧 | 2,553/2,553 | 已生成嵌入和相对 prompt 分数 |
| 视觉事件 | 1,974 | 检测器输出量，不是正确事件数 |
| 融合事件 | 2,270 | 全模态保留事件量，不是正确事件数 |

独立审计对语义 manifest 得到 `formal_empirical_ready=true` 和 `multimodal_empirical_ready=true`。前者表示正式字幕 provenance、精确时间戳、覆盖和身份门槛通过；后者表示可信语言来源、音频/视觉记录、全时间轴覆盖和事件结构齐全。两者都不是准确率或效果结论。

## 四臂配对消融

每一讲的四个 arm 使用完全相同的 transcript segment 记录，只改变允许进入分析的机器派生模态；不拟合新模型，也不加入 classroom observation。结果为：

| Arm | 平均内部 Overall | 配对变化 | 有事件讲次 | 保留事件数 |
|---|---:|---:|---:|---:|
| transcript-only | 95.37 | 0.00 | 0/10 | 0 |
| transcript + audio | 95.37 | 0.00 | 10/10 | 237 |
| transcript + visual/OCR | 95.30 | -0.07 | 10/10 | 2,033 |
| full | 95.32 | -0.05 | 10/10 | 2,270 |

`internal_overall_score` 是项目自身对结构、证据、执行性和可追溯性的量表；`internal_evidence_consistency_score` 检查引用是否精确匹配同一分析记录。两者都不衡量事件是否符合独立真值。当前结果只能支持“音频/视觉改变了保留事件和可引用证据”，不能支持“多模态提高准确率或 Skill 质量”；内部 Overall 甚至没有出现正向增益。

要建立确认性多模态增益，必须先冻结 pipeline 和假设，再在未用于开发的新完整讲次上取得独立双人事件/Skill gold label，以讲次或 session 为配对单位报告 Accuracy/Precision/Recall/F1 或独立质量分的差值、置信区间和多重比较规则。学习效果还需要另一条伦理审批后的学习者对照实验，不能由事件消融替代。

私有实跑完成后，`python3 scripts/build_multimodal_public_receipts.py` 会重新读取语义 manifest、审计、CLIP batch receipt 和四臂报告，生成两个 aggregate-only receipt，并用 SHA-256 对私有来源产物和集合做承诺。公开 receipt 不能替代私有证据，也不能证明不可重识别；发布前还须通过 `tsm release-audit artifacts/public` 和人工披露风险复核。

## 统一事件与一致性验证

事件 schema 位于 `schema/multimodal_event.schema.json`。典型事件包含稳定 `event_id`、类型、起止时间、模态集合、证据载荷、启发式 confidence 和支持策略。多模态 Skill 评估会核对：

1. `event_id` 是否存在于原分析；
2. 事件类型和起止时间是否一致；
3. 引用的模态集合是否一致；
4. Skill 声称的策略是否在事件映射中；
5. `evidence` 与 `confidence` 是否和事件原记录完全一致；
6. 帧路径、帧哈希、OCR 文本、转写引文和静音记录是否能回指分析产物；
7. 是否覆盖分析中实际可用的模态。

伪造事件 ID、证据文本、置信度或模态会使 `multimodal_consistent` gate 失败。评估输出把这项指标命名为 `internal_evidence_consistency`，避免被误解成检测准确率。

## 合成 fixture 与正式实跑的区别

`scripts/run_multimodal_demo.sh` 使用两张合成幻灯片、音调加静音、提供的转写和人工匿名观察，用于快速验证 FFmpeg/Tesseract/融合代码。其 fixture 对照只覆盖已知合成事件，即使得到 Precision/Recall/F1 = 1.0，也只证明该固定 fixture 行为。

10 讲正式实跑解决的是“有没有处理完整真实媒体、有没有全程视觉、能否回指来源和哈希”；它仍没有独立事件真值。因此合成 fixture 指标不能外推到正式视频，正式视频的输出数量也不能反向写成 Accuracy。

## 降级策略

- 没有 Tesseract：仍运行 transcript + audio + scene 分析，但 OCR/板书证据缺失，不能声称 full 视觉链路；
- 没有 CLIP：保留帧、OCR、dHash 与事件，semantic status 明确为 pending/absent；
- 没有课堂观察：不推断学生身份或反应；
- 只有字幕没有视频：回退到 transcript-first 基线；
- 使用音频文件而非视频：只运行 ASR/字幕和音频特征，不伪造视觉证据；
- 长讲次中断：只恢复通过输入、配置与输出校验的 chunk checkpoint。

## 尚需外部完成

- 10/10 Teaching Skill 的真实双人独立复核；
- 事件级独立人工 gold set 与 OCR/视觉语义准确率；
- 冻结 pipeline 后的新数据确认性四臂消融；
- 新学校/新 session 的一次性前瞻部署 lockbox；
- 伦理审批后的真实学习者前后测或对照实验；
- 手势与指向目标、板书区域精细增量分割、说话人分离和复杂图表关系推理。
