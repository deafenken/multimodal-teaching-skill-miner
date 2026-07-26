# Teaching Skill Mining and Evaluation System

本项目实现题目中的四阶段工程闭环：对教学视频联合分析语音转写、提问等待、关键帧、PPT/板书 OCR 与匿名课堂观察，从“教师如何教”中抽取 Teaching Skill，把 Skill 表示成另一个 Agent 可直接执行的 JSON 与状态机，最后进行自动评分、跨领域新任务测试和双人复核覆盖审计。

默认演示链路只使用 Python 3.10+ 标准库，不需要网络或 API。项目内置按 2 门课程组织、每门 5 讲的人工释义演示数据，一条命令即可复现工程闭环；它本身不是完整公开课字幕。项目另已对 10/10 MIT OCW 完整讲次完成官方 WebVTT、来源页、媒体、内容哈希与整段时间轴绑定，并在私有目录真实运行全程音频静音分析、均匀/场景抽帧、Tesseract OCR、板书/幻灯片变化、CLIP 视觉语义和事件融合。第三方视频、字幕正文、帧、OCR 文本和嵌入均不打进 wheel。当前准确状态是“完整视频多模态工程证据已闭环；真实双人复核、事件级识别准确率、确认性多模态增益、学习效果和外部部署验证待完成”，详见 [`docs/project_status.md`](docs/project_status.md)。

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

这个入口启动时先把旧 acceptance 降级为 `stale_not_accepted`，随后依次执行全量项目验收、两个独立临时源码副本的字节级一致构建、最终候选 wheel 的隔离安装和视频闭环、wheel/公开目录发布审计；候选通过后才原子替换 `dist/` 中的同名 wheel，再对新 acceptance 本身做发布审计并原子替换 `artifacts/release_acceptance_1.2.0.json`。因此中途失败不会遗留看似仍有效的旧验收。acceptance 的测试数、wheel 哈希/大小/成员、公开目录摘要都由绑定同一 wheel SHA-256 的新鲜 receipt 重算；验证源码或任一产物在验收后变化都会失败，不会复制旧 acceptance 的字段。为保证相同证据生成相同字节，acceptance 有意不写墙钟时间。

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
自动评估：结构 + 证据 + 可执行性 + 教学质量 + 留出迁移 + 溯源
        │
        ▼
两名独立复核者（正式实验建议盲法）与学习效果 A/B
```

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

当转写已存在时，可以直接把视频与转写对齐，无需重复 ASR。外部提供的带时间戳文本标为 `transcript`；只有 ASR provenance 哈希与当前媒体一致时才标为 `speech`，同时产物中的 `audio_content_verified` 为 `true`：

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

本次最终产物声明 pipeline `teaching_skill_miner.longform_multimodal.v9`、extraction `teaching_skill_miner.longform_extraction.v2`。v2 使用正确的 TSV quoting 解析 Tesseract 输出；此前解析结果和由它派生的聚合值已作废并重新跑完 10 讲。新实跑的可核验证据为：10 个完整视频共 `1,038,813,006` bytes，FFprobe 总时长 `26,656.83` 秒（`7.404675` 小时）；抽取 `2,553` 帧，其中 `2,441` 帧有阈值后 OCR 文本、`1,964` 帧至少有 3 个接受词，共接受 `40,191` 个词；生成 `1,974` 个视觉事件和 `2,270` 个融合事件。10/10 讲通过全时间轴采样覆盖门槛，10/10 讲通过官方字幕—媒体时间轴绑定，审计结果为 `formal_empirical_ready=true`、`multimodal_empirical_ready=true`。

CLIP 对 `2,553/2,553` 个哈希绑定帧完成 512 维视觉嵌入和八类封闭 ontology 的相对 prompt 分数。本次使用 `openai/clip-vit-base-patch32` revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，本地 FP16 `model.safetensors` SHA-256 为 `676093550c9e05bc3ba55256c278c89f0d15a1a1585f3b81d76454c33b852d5e`，最终推理设备为 CPU。曾用 GPU 服务器处理公开模型权重，但没有把视频、帧、字幕、OCR 或其他私有项目数据上传到该服务器。

四臂配对消融对每一讲复用完全相同的 transcript segments，只改变可见的机器派生模态，并排除课堂观察：

| Arm | 平均内部 Skill 分 | 相对 transcript-only | 保留事件数 |
|---|---:|---:|---:|
| transcript-only | 95.37 | 0.00 | 0 |
| transcript + audio | 95.37 | 0.00 | 237 |
| transcript + visual/OCR | 95.30 | -0.07 | 2,033 |
| full | 95.32 | -0.05 | 2,270 |

这些分数只是同一流水线的结构、证据引用一致性和可执行性量表；它们不是 Accuracy、Precision、Recall、F1，也没有显示内部 Skill 分数增益。OCR 计数不表示 OCR 正确率，CLIP 相对 prompt 分数不是校准概率，检测器事件数不表示正确事件数。因为没有独立事件 gold label、独立 Skill 质量评分或学习者结果，`recognition_accuracy_established`、`multimodal_gain_established`、`teaching_effectiveness_established` 和 `deployment_accuracy_established` 均为 `false`。完整设计和证据解释见 [`docs/multimodal_design.md`](docs/multimodal_design.md)。

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

因果优化结果见 [`OPTIMIZATION_RESULTS.md`](artifacts/dipser_credible/full_v5_52_complete_v5/OPTIMIZATION_RESULTS.md)；0.9 挑战、模态消融、50-seed、未来上下文反事实和阻断审计见 [`HIERARCHICAL_0_9_RESULTS.md`](artifacts/dipser_credible/full_v5_52_complete_v5/HIERARCHICAL_0_9_RESULTS.md)，逐折/逐样本机读产物为同目录的 `hierarchical_0_9_report.json`。层级固定 gate 的 LOCO、LOAO、双重阻断 Accuracy 分别只有 0.6554、0.8919、0.6351。因此 `accuracy_0_9_established`、确认性 `multimodal_gain_established` 和 `deployment_accuracy_established` 均保持 `false`。原始覆盖流和指纹见 [`RESULTS.md`](artifacts/dipser_credible/full_v5_52_complete_v5/RESULTS.md)。

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

部署准确率、确认性多模态增益和真实学习者效果使用三条相互独立的外部证据链。后两类可以用 `prepare-external-research-evidence`、`sign-external-research-evidence` 和 `verify-external-research-evidence` 对聚合研究 manifest 重新计算 gate 并验证外部签名，再分别通过 `verify-delivery` 的 `--external-multimodal-*` 与 `--external-learner-*` 参数接入。两类 manifest 还必须绑定同一个实际交付 system artifact，delivery 会实算其 SHA-256。仓库当前不附带任何正向外部证据；一个有效部署 receipt 不能替代配对模态消融或学习效果实验，准备 manifest 也不自动证明其绑定的原始研究产物真实。完整协议见 [`docs/external_research_evidence.md`](docs/external_research_evidence.md)。

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

该路径会在 `pipeline_summary.json` 中写入 `transcript_source_mode`、`language_evidence_status` 和 `audio_content_verified`。提供的文字只算 transcript 模态；只有同一媒体哈希绑定的 ASR provenance 才会标记为已核对语音内容。

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

前一、四、六项在诚实的 Skill 之间本就有差异，衡量"方法还原了多少"；后三项在诚实的 Skill 上恒为 1.0，只有在步骤伪造出处时才会塌陷，是这个维度可证伪的一半。当前十个 Skill 的方法忠实度落在 83.3–89.8（标准差 1.64），总分落在 92.3–93.7。

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

完成人工评分表后，可计算逐 Skill 结果和双人二次加权 Cohen's kappa：

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

`fetch-formal-captions` 从每个 MIT OCW 讲次页面重新确认页面声明的 WebVTT 与媒体配对，要求字幕 URL 使用 `https://ocw.mit.edu`，校验固定字幕 SHA-256，并用 FFprobe 读取同页所链接媒体的实际时长；该命令本身不保存视频。当前实跑结果为 10/10 正式字幕通过，平均 770.5 个合并后 cue、6172.8 个审计 token，首尾 cue 时间轴覆盖率为 98.84%–99.72%；公开 receipt 只含 URL、哈希、时长和计数，不含字幕文本。随后独立执行 `fetch-full-videos` 已把页面绑定的 10 个完整媒体下载到私有目录，并再次校验本地 SHA-256、容器、音视频流和参考时长；由于上游索引没有发布者固定的媒体哈希，本地 SHA-256 能检测下载后的变化，但不能独立证明发布者原始字节身份。

因此：

- 自动评估会把这种数据的证据忠实度与溯源分限制在 85；
- 正式字幕集使用官方 caption 时应报告页面 URL、字幕哈希、媒体时长和时间轴覆盖；若改用 ASR，才必须另外报告模型、版本、模型/解码配置指纹、独立 WER 抽检和人工修订比例；
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
