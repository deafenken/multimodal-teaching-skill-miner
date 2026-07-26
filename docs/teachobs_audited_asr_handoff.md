# TeachObs 可审计 ASR GPU 交接

这条链只解决 30 讲字幕覆盖中的技术缺口：把已经下载、哈希并由 FFprobe
核验的私有媒体交给受控 GPU 环境转写，再把结果严格验回。它不会把 ASR
写成“官方字幕”，也不会仅凭哈希、时间覆盖或模型 provenance 宣称 WER、
内容准确率或人工审计已经成立。

## 证据层级

每讲只选择一个来源，优先级固定为：

1. 平台标记为 creator-provided 且时间轴通过审计的字幕；
2. 平台自动字幕且时间轴通过审计；
3. 同一媒体 SHA-256 绑定、模型与运行时均通过验收的 ASR fallback；
4. pending。

私有 `transcript_coverage_matrix.json` 保留逐讲选择和哈希；公开 receipt 只保留
30 讲的聚合计数和证据文件哈希，不含课次 ID、URL、路径、媒体或正文。

## 0. 当前 pending 收据

当 GPU 尚未运行时，只生成诚实的聚合收据：

```bash
python3 -m teaching_skill_miner prepare-teachobs-asr-handoff \
  --pending-only \
  --media-manifest artifacts/private/external_datasets/teachobs/media/media_manifest.json \
  --caption-audit artifacts/private/external_datasets/teachobs/captions/caption_audit.json \
  --public-receipt artifacts/public/teachobs_asr_receipt.json
```

此命令不加载 Whisper、不下载模型、不读媒体内容，只读取私有 manifest 与字幕
审计的哈希/状态。`handoff_status=pending` 是正确结果。`--pending-only`
不需要容器 digest；一旦生成可执行 job，精确 digest 就是必填且不可后改的 contract。

## 1. GPU 环境与模型固定

在获授权的 GPU 服务器上预先放置模型快照和媒体。工作脚本不会下载模型或
媒体，并强制 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；模型必须是本地
目录。哈希器先以 `lstat` 遍历整棵树，拒绝文件、目录或断链形式的任意符号链接，
也拒绝 FIFO、socket、device 等非普通节点，再计算所有普通文件的确定性哈希：

GPU 环境依赖可用 `pip install 'teaching-skill-miner[asr-gpu]'` 安装；生成 job
manifest 时仍必须记录实际解析出的两个精确包版本，不能只记录版本范围。

```bash
tsm hash-teachobs-asr-model \
  --model-directory /private/models/whisper-large-v3-turbo
```

模型 ID 只是名称；真正锁定字节的是：40 位提交 revision、上述模型树 SHA-256、
精确 `faster-whisper` 版本和精确 `ctranslate2` 版本。GPU 容器还必须以
`sha256:<64-hex>` 镜像 digest 运行，不能使用浮动 tag 作为 provenance。

### 1.1 从已验收 wheel 构建 digest 绑定容器

容器构建不接受工作区作为 Docker context。先准备已经通过 release wheel 验收的
单个 `teaching_skill_miner-*-py3-none-any.whl`，并由管理员把 CUDA base image
预置到服务器本地 image store；不要把私有媒体、模型、字幕、结果或整个仓库放进
构建目录。取得本地 base image 的精确 image ID 后运行：

```bash
tsm_base_image_id=$(docker image inspect \
  --format '{{.Id}}' nvidia/cuda:12.4.1-base-ubuntu22.04)

sh scripts/build_teachobs_asr_container.sh \
  --wheel dist/teaching_skill_miner-1.2.0-py3-none-any.whl \
  --base-image nvidia/cuda:12.4.1-base-ubuntu22.04 \
  --expected-base-image-id "$tsm_base_image_id" \
  > /private/handoff/teachobs_asr_container_receipt.json
```

默认使用 Ubuntu 官方源与 PyPI；只能访问国内镜像的受控服务器可另外显式传入
`--ubuntu-mirror https://mirrors.aliyun.com/ubuntu` 和
`--pip-index-url https://mirrors.aliyun.com/pypi/simple`。脚本只接受这两个固定镜像
或官方源，并把所选来源写入不含路径的构建收据；任意其他 URL 都失败关闭。

[`build_teachobs_asr_container.sh`](../scripts/build_teachobs_asr_container.sh)
先比较本地 base 的实际 ID 与显式 expected ID，再将 Dockerfile 和这一个 wheel
复制到 `0700` 的 `mktemp` context，以 `--pull=false` 使用已验证的本地 image tag，
并把精确 base image ID 写入最终镜像标签与收据。
输入 wheel 若是 symlink、名称不符合精确 universal release wheel，或 base ID
发生变化，构建会失败关闭。Dockerfile 再校验 wheel SHA，并固定：

- `faster-whisper==1.2.1`；
- `ctranslate2==4.8.1`；
- `nvidia-cublas-cu12==12.9.2.10`；
- `nvidia-cudnn-cu12==9.24.0.43`。

它还安装 GPU runner 重新探测媒体时所需的 `ffprobe`。镜像构建后，脚本用断网、
只读 rootfs、drop-all-capabilities、no-new-privileges 的临时容器重新读取上述四个
distribution version，实际调用 `run-teachobs-asr-gpu --help` 并检查三个必填入口，
同时检查 `ffprobe`。这里的 help 检查不要求占用 GPU；真正转写仍必须由 runner 的
`nvidia-smi` gate 看到 CUDA 设备。

成功时 stdout 只有一份不含 tag、主机路径、用户名、模型或媒体信息的 JSON。
其中 `container_image_digest` 与 `image_id` 完全相同，都是 Docker 返回的最终
`sha256:<64-hex>` image ID；生成 job manifest 和执行 worker 时必须原样使用这个
值，不能改回浮动 tag，也不能换成另一份 registry manifest digest。构建期间
`apt`/`pip` 仍可能访问管理员配置的软件包源；`--pull=false` 专门保证不隐式更新
base image，而离线环境变量和精简 context 保证镜像构建及运行均不下载或嵌入
Whisper 模型、TeachObs 媒体或字幕。最终 image ID 是本次容器字节配置的权威绑定；
若软件包源或 OS 包发生变化，重建得到不同 ID 时必须重建 job manifest。

## 2. 生成内容为空的私有作业 manifest

在本次预定 ASR job 所选课次的媒体都完成 SHA-256/FFprobe 验收后运行；
默认只给平台字幕缺失课次建 job。完整 30 讲 profile 可以要求所有
fallback 媒体就绪；论文对齐 profile 则应用重复的 `--lesson-id` 在看结果前固定
需要 ASR 的精确课次集合。若任一被选媒体未到齐，会进入 `media_pending`；
正式交接应加 `--require-all-selected-media` 失败关闭。该 flag 约束的是显式选中
的 job 集合，不能把未选课次的缺口冒充为已完成。

```bash
python3 -m teaching_skill_miner prepare-teachobs-asr-handoff \
  --media-manifest artifacts/private/external_datasets/teachobs/media/media_manifest.json \
  --media-root artifacts/private/external_datasets/teachobs/media \
  --caption-audit artifacts/private/external_datasets/teachobs/captions/caption_audit.json \
  --model-id openai/whisper-large-v3-turbo \
  --model-revision <精确40位小写提交SHA> \
  --model-files-sha256 <上一步64位模型树SHA> \
  --faster-whisper-version <精确版本> \
  --ctranslate2-version <精确版本> \
  --container-image-digest sha256:<64位镜像digest> \
  --min-timeline-span-fraction 0.90 \
  --max-endpoint-gap-fraction 0.10 \
  --require-all-selected-media \
  --output artifacts/private/external_datasets/teachobs/asr/job_manifest.json \
  --public-receipt artifacts/public/teachobs_asr_receipt.json
```

job 逐讲绑定媒体相对路径、字节数、SHA-256、实际时长、语言策略、模型、解码、
CUDA runtime contract（包括上述精确容器 digest）、覆盖阈值和结果 schema。它不含媒体字节或转写正文。
默认语言为 `auto`；需要固定语言时可用 `--language`，或用私有 JSON
`--language-map` 对课次逐一指定。

固定解码为 full-media 单次 transcribe、beam 5、temperature 0、禁翻译、
`condition_on_previous_text=false`、word timestamps、VAD、CUDA float16。任何配置
变化都必须重建 manifest，不能在服务器上悄悄覆盖。

coverage policy 使用 `full_media_single_pass_vad_relative_endpoints_v1`。完整媒体
仍以哈希和 FFprobe 时长绑定并只做一次 full-media 输入；但启用 VAD 后，第一/最后
时间戳表示第一/最后个被保留的语音锚点，而不是视频文件边界。固定门槛为
first-to-last span 至少覆盖媒体时长的 `0.90`，且首端和尾端空白各自不得超过媒体
时长的 `0.10`。端点秒数仍记录为诊断量，但不再作为绝对秒数 gate。构建器强制
`--max-endpoint-gap-fraction <= 1 - --min-timeline-span-fraction`，并把同一个相对
门槛对称应用于两端，不能查看结果后分别放宽开头或结尾。该技术门槛只证明时间轴
与完整输入一致，不证明静音段包含语音、转写内容正确或 WER。

这一策略变更使用 job manifest v2、lesson result v4 和 GPU runner v4。旧的 job
manifest v1、result v3 和 runner v3 会失败关闭。coverage policy、job hash、
manifest hash、result hash 与 runner 源码哈希都是证据链的一部分，因此不得把旧
结果原地改写或迁移成新结果；必须由新容器、新 manifest 对全部选中讲次重新执行，
不能只重跑旧策略未通过的课次。

GPU runner v4 仍把 30.5 秒作为最终输出 segment 的固定 wall-clock 上限，但不再
错误假设 faster-whisper 的原始 segment 必然受此上限约束：VAD 会先压缩语音，
word timestamps 恢复到原媒体时间轴后，原始 segment 可能横跨较长静音。对这种
overlong segment，runner 只按已请求的正时长 word timestamp 做确定性贪心拆分；
每组取满足 30.5 秒上限的最长词前缀，并验证所有词有限、正序、不重叠、单词不超限，
以及所有拆分段拼接后与原 segment 的 `text.strip()` 精确一致。

faster-whisper 1.2.1 在 VAD 时间轴恢复后偶尔会输出 `start == end` 的非空 word。
真实 S15 诊断显示这些 point timestamp 不都落在 0.05 秒“同边界”内，因此 v4 沿用
v3 的确定性拆分规则，不再
虚构这种条件。它保留每个有限、有序且位于原 segment 内的 point anchor，但 point
永远不成为输出段边界。每个零时长词只在原词序列中相邻的前、后正时长词之间，按
纯时间距离选择：前项距离取 point 到前词 end，后项距离取后词 start 到 point；
选择较近者，距离相同时固定选择前项。每项分配不得超过同一个 30.5 秒上限，并且
所有分配到的正锚点索引必须单调，否则 fail closed。

零时长词仍按原顺序附着在所选正时长词的前或后；附着只改变文本归属，不改变任何
正时长 word 的起止时间。全零词、非有限、逆序、point 越出原 segment、没有正时长
锚点、最近锚点超过 30.5 秒、分配索引不单调、缺 word timestamp、文本不等、无法
拆成至少两组或任一时间约束失败时都 fail closed。

结果记录原始段数、拆分段数、输出段数、overlong 原始词数、正时长锚点数、附着的
零时长词数、固定最近邻策略、30.5 秒阈值及最大实际附着距离；导入器重验策略字段、
距离范围和 `原始词数 = 正时长锚点数 + 附着零时长词数`。结果同时明确
`zero_duration_point_anchors_within_source_segment_verified=true`、
`zero_duration_assignment_anchor_indices_monotonic_verified=true`、
`positive_word_boundaries_preserved=true`、`reference_or_label_used=false`、
`boundary_selection_uses_text_content=false`。这是技术时间轴规范化，不使用 gold、
人工标签或下游分数，也不建立或提高内容准确率/WER。

## 3. GPU 执行

```bash
tsm run-teachobs-asr-gpu \
  --job-manifest /private/handoff/job_manifest.json \
  --media-root /private/teachobs/media \
  --model-directory /private/models/whisper-large-v3-turbo \
  --container-image-digest sha256:<与job manifest完全相同的64位镜像digest> \
  --output /private/handoff/results
```

执行前会重新计算模型树与每个媒体文件的 SHA-256，并重新探测媒体时长；传入的
容器 digest 必须逐字节匹配 job manifest 已冻结的值，而不是临时填写任意合法格式。
运行环境需要
可见 NVIDIA CUDA 设备。输出是一讲一个 JSON，包含排序且不重叠的非空 segment、
检测语言、完整模型/解码配置、runner 源码哈希、Python/包版本、GPU/驱动和容器
digest。它明确写入：

- `transcript_kind=automatic_speech_recognition`；
- `source_tier=audited_asr_fallback_candidate`；
- `official_caption=false`；
- `content_accuracy_established=false`；
- `word_error_rate_established=false`。

## 4. 结果回收与严格验收

把逐讲 JSON 安全传回私有工作区后执行：

```bash
python3 -m teaching_skill_miner import-teachobs-asr-results \
  --job-manifest artifacts/private/external_datasets/teachobs/asr/job_manifest.json \
  --media-manifest artifacts/private/external_datasets/teachobs/media/media_manifest.json \
  --media-root artifacts/private/external_datasets/teachobs/media \
  --caption-audit artifacts/private/external_datasets/teachobs/captions/caption_audit.json \
  --results artifacts/private/external_datasets/teachobs/asr/results \
  --output artifacts/private/external_datasets/teachobs/asr/import_audit.json \
  --coverage-matrix artifacts/private/external_datasets/teachobs/asr/transcript_coverage_matrix.json \
  --public-receipt artifacts/public/teachobs_asr_receipt.json
```

导入器会重新计算 job manifest、每个 job、每个结果 JSON 的 canonical hash，
重新哈希本地媒体和 FFprobe 时长，并逐字段比较模型、revision、模型树、解码、
运行时与媒体绑定；每讲结果的容器 digest 必须与 manifest 精确相同，因此整批讲次
也必须一致。segment 必须有限、非负、按序、不重叠、正文非空，且首尾和
时间跨度达到 manifest 预注册阈值。缺文件保持 pending；出现额外 JSON、绑定错误、
浮动/伪造 provenance 或 `official_caption=true` 时直接拒绝。

## 仍然必须外部完成的内容核验

技术验收只能证明“这些字来自这个媒体、这个固定模型和这次固定运行”。若要
报告内容准确率或 WER，仍需在转写完成后冻结抽样方案，由不看模型结果来源的人工
听写参考或经过资质核验的正式 transcript 计算 WER，并记录抽样覆盖、双人复核与
修订比例。在那之前，公开 receipt 中这些结论永远为 `false`。
