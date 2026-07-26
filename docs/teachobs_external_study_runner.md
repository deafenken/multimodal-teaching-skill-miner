# TeachObs 真实外部多模态总 runner

## 它解决什么问题

`scripts/run_teachobs_external_study.sh` 把已经拆开的 TeachObs 工程入口串成一条可恢复、失败关闭的真实数据链。它不是演示数据 runner，也不会把“命令退出 0”误当成研究完成：每个重型阶段之后都会调用 `scripts/check_teachobs_external_study.py` 复核产物内容和哈希绑定。

runner 当前只接受论文对齐的 `paper_track1_23_train_6_test`：训练固定 23 讲/3,846 场景，测试固定 S2/S5/S19/S24/S28/S30 共 1,099 场景；S4 在任何拟合和指标前排除，所以媒体/特征 gate 精确要求 29 讲、4,945 场景且唯一缺失 S4，四臂 bootstrap 固定 6 个 lesson cluster。完整 23/7 profile 只有在 S4 的媒体、字幕来源和 5,158-scene 物化链全部完成后才能恢复；当前传入 `full_23_train_7_test` 会失败关闭。Stage 5 先把 23 讲平台字幕和 6 讲审计 ASR 物化为同一 4,945-scene 顺序，Stage 7 才运行四臂并导出不含 pickle 的私有 `bundle_manifest.json + 4 × (manifest.json + arrays.npz)`；checker 会重新读取全部私有物化 JSONL，复核 transcript/input fingerprint，再安全加载 bundle 和每个独立 arm，复核 profile、lesson/sample 顺序、dataset/profile fingerprint、companion arrays 的 SHA-256、`allow_pickle=False` 完整性以及 prediction/probability parity gate。

这条链最终建立的仍然只是：

```text
provisional exploratory four-arm multimodal comparison
on the publicly accessible TeachObs source-aligned published
six-lesson Track 1 text/frame intersection
```

公开测试标签在系统冻结前已经可访问，且首轮固定阈值结果在当前训练 OOF 修订前已经被观察。runner 的 5 折 lesson-grouped OOF 会在每折内部重拟合全部 TF-IDF/IDF/scaler，只用 23 讲训练标签选择固定候选 C 和逐标签阈值；但这仍是 post-test iterative exploratory revision，不是首次盲测。因此 runner 固定输出 `confirmatory_multimodal_gain_established=false`、`external_lockbox_established=false`、`deployment_accuracy_established=false` 和 `learner_effectiveness_established=false`。它不会生成虚构的人工评分、外部签名或学习效果。

## 前置条件

1. 本地必须已有固定 commit `96c251ae09e79edd06a3a9bbaaa8b8f7fe99a15c` 的完整 TeachObs repository，以及与其目录树绑定的私有 `acquisition_receipt.json`。默认位置分别是：
   - `artifacts/private/external_datasets/teachobs/repository`
   - `artifacts/private/external_datasets/teachobs/acquisition_receipt.json`
2. 必须有本地 CLIP snapshot；runner 不下载模型。默认使用 `artifacts/private/models/openai_clip_vit_b32_fp16_3d74acf9`，固定 revision 为 `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。
3. Python 环境需要 NumPy、Pillow、SciPy、scikit-learn、Torch、Transformers 和 yt-dlp；系统需要 FFmpeg、FFprobe、Tesseract。
4. 必须明确接受注释许可证与源视频各自独立的权利/平台条款。没有授权 flag，runner 在任何网络或媒体操作前退出。

完整 repository 与 acquisition receipt 是已审计输入，不由这个 runner 猜测或临时重写。commit、目录树、场景 manifest 或 receipt 任何不一致都会在 media-plan 阶段失败。

## 基本运行

```bash
sh scripts/run_teachobs_external_study.sh \
  --acknowledge-source-terms
```

这会显式采用 `paper_track1_23_train_6_test` 默认值。即使已取得 S4 媒体，在完整字幕物化器和验收器扩展到 30 讲前，runner 也不会把部分链路冒充完整 23/7 结果。

也可以显式选择设备和并行度：

```bash
sh scripts/run_teachobs_external_study.sh \
  --acknowledge-source-terms \
  --clip-model artifacts/private/models/openai_clip_vit_b32_fp16_3d74acf9 \
  --clip-revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --clip-device cuda \
  --clip-batch-size 64 \
  --evaluation-profile paper_track1_23_train_6_test \
  --jobs 4 \
  --feature-jobs 2 \
  --visual-jobs 2 \
  --ocr-jobs 2
```

`feature-jobs × visual-jobs` 和 `feature-jobs × ocr-jobs` 都不能超过 8。GPU 只改变 CLIP 执行设备，不改变固定模型 revision、权重哈希、官方划分或指标协议。

## 恢复与失败语义

重跑同一命令就是恢复：

- 已通过 hash/commit gate 的注释审计直接复用；`--refresh-annotations` 才重新抓取固定注释文件。
- 已绑定的完整视频和 `.part` 由媒体下载器复用；失败讲次保留在私有 failure manifest。完整 profile 在任何缺课时退出；论文 profile 只会在 checker 同时证明 29 讲恰为 S1–S30 去除 S4 后，接受下载命令的非零状态，其他任何失败都不会被吞掉。
- 已抽取且哈希仍匹配的逐讲音频、帧、OCR、图像证据和 CLIP 任务可复用。完整 profile 要求 30/30、5,158/5,158；论文 profile 要求 29/30、4,945/4,945、`failed_lesson_ids=[S4]`、同样的 combined-CLIP lesson IDs 和明确的 partial claim。满足对应精确 gate 后才运行四臂。
- 物化 manifest、29 个 lesson JSONL 和公开 aggregate receipt 全部通过 live 校验时，Stage 5 才复用；任何文件缺失、被改动或尝试回退发布版 transcript 都会停止，而且不会覆盖旧现场。
- 四臂结果、公开 receipt、当前 feature manifest、当前 transcript materialization 和冻结 bundle 全部匹配时，Stage 7 直接复用。冻结目录存在但残缺、过期或被篡改时不会删除或覆盖；runner 会停止并要求先保留现场，再通过新的 `TSM_TEACHOBS_FROZEN_MODEL_ROOT` 重跑。
- 已有合法字幕审计默认复用。`--retry-captions` 会重试 pending 讲次。合法但不完整的字幕审计可以与媒体实验并行存在，但会明确保持 `formal_caption_timeline_audit_completed=false`；时间轴覆盖或文本 overlap 不是 WER。
- ASR 私有证据按 `none → job manifest → import audit + coverage matrix` 单调推进。只要较深层文件存在，runner 就必须通过对应的哈希门禁，绝不因其陈旧、残缺或损坏而退回较浅 pending receipt。论文 profile 的完整导入门禁精确要求平台字幕 23 讲、有效 ASR 6 讲、覆盖 29 讲且唯一 pending 为 S4；因此 `import-teachobs-asr-results` 在六讲全部有效时仍会因总覆盖不是 30/30 返回 2，这是预期的研究状态，不是 ASR 失败。只有 job manifest 或 pending receipt 时 runner 现在会停止；full checker 通过并完成 4,945-scene 物化后才可进入特征/指标阶段。
- 任一必需阶段失败，后续阶段不执行；已经写入的私有下载、帧、OCR 和特征不删除。下一次运行从这些产物继续。

媒体/特征原入口允许把部分成功写入 manifest，因此总 runner 额外执行 profile-aware gate：完整 profile 要求 `selected_complete=true`、`downloaded_lesson_count_total=30`、`complete=true`、`failed_lesson_count=0` 和 `scene_count=5158`；论文 profile 只接受精确 29 讲/4,945 场景、唯一缺口 S4、与之一致的私有 failure receipt 和 combined-CLIP 声明。其他缺口不会因为选了论文 profile 而被忽略。这正是“部分成果可保存”和“只有事先定义的精确交集才可进入指标计算”之间的边界。

## 字幕重试与本地 JavaScript runtime

```bash
sh scripts/run_teachobs_external_study.sh \
  --acknowledge-source-terms \
  --retry-captions \
  --js-runtime node:/opt/homebrew/bin/node
```

runner 默认不指定 JavaScript runtime。即使显式指定本地 Node，字幕与媒体模块仍固定向 yt-dlp 传入 `--no-remote-components`。runner 不安装、不下载、也不执行 remote EJS component。若当前 yt-dlp 版本要求本地 EJS solver，必须事先独立审查并把与 yt-dlp 精确匹配的 `yt-dlp-ejs` wheel 安装到私有 `PYTHONPATH`；媒体 receipt 在确认下载命令确由同一 Python 的 `-m yt_dlp` 执行时，会记录 yt-dlp/EJS/curl_cffi 版本、JavaScript runtime family 和可取得的 EJS wheel SHA-256，但不会记录安装路径。缺少这些字段不能被解释为远程 solver 已获准，`remote_ejs_allowed` 始终必须为 `false`。

## 代理旁路、HTTP impersonation 与浏览器 Cookie

这三项恢复手段彼此独立，而且全部默认关闭。若当前进程继承的代理出口被视频平台限流，可以显式要求 yt-dlp 直连，并使用本地已安装的 `curl_cffi` 后端模拟 Chrome HTTP 指纹：

```bash
PYTHONPATH=/private/path/to/curl_cffi:/private/path/to/yt_dlp_ejs:artifacts/private/tools/yt_dlp \
sh scripts/run_teachobs_external_study.sh \
  --acknowledge-source-terms \
  --yt-dlp-direct \
  --yt-dlp-impersonate chrome \
  --yt-dlp-youtube-client android_vr
```

`--yt-dlp-direct` 只传递 yt-dlp 的空 `--proxy` 参数，不接受也不记录代理 URL；`--yt-dlp-impersonate` 当前只接受固定枚举 `chrome`。`--yt-dlp-youtube-client` 是字幕 metadata/VTT 获取的显式选项，目前只允许已实测的 `android_vr`；该选项默认关闭。完整媒体下载采用独立且固定的选择策略：匿名路径继续使用 `android_vr`；一旦显式启用浏览器 Cookie，就自动切换到支持 Cookie 的 `default,web_safari`，因为 Android VR 客户端不接受浏览器 Cookie。实际 client 会写入私有 receipt 和媒体 binding，媒体复用时不会退化丢失。不要把字幕的 `--yt-dlp-youtube-client android_vr` 与 Cookie 组合成“凭据字幕恢复”方案。直连和 impersonation 同时传给字幕请求和完整视频下载，避免只有某一阶段继续继承被限流的代理。runner 不安装 `curl_cffi`，也不会把“请求了 impersonation”写成“后端已被独立验证”。

只有直连和 impersonation 仍不足、且浏览器账号所有者明确授权时，才可再加入 `--cookies-from-browser chrome` 或安全的 profile 名，例如 `'chrome:Profile 1'`。该接口拒绝 profile 路径、`..`、控制字符、keyring/container 语法和未列入白名单的浏览器。实现只把 `--cookies-from-browser` 交给 yt-dlp 子进程，不传 `--cookies`、不导出 Cookie jar，私有 audit/manifest 仅保留凭据是否使用、浏览器 family、实际 player-client 策略和非敏感 transport/provenance 字段，不保留 profile、Cookie、账号、代理 URL 或 Cookie 路径。credentialed 非零退出和超时都使用固定错误消息并切断可能携带命令行/profile 的异常链；字幕失败记录同样只保留允许列表中的错误类别，不持久化 yt-dlp 原始诊断。

浏览器 Cookie 会把已登录会话暴露给视频平台，可能导致额外验证、限流乃至账号暂停。提供该 flag 本身就是一次显式 opt-in；不得在共享账号、未经账号所有者授权的环境或公开 CI 中启用。

## S4 或其他显式候选镜像

默认不读取目录里可能存在的 S4 override 文件，也不会自动替换 canonical URL。只有同时给出私有 manifest 和第二个独立确认才会启用：

```bash
sh scripts/run_teachobs_external_study.sh \
  --acknowledge-source-terms \
  --source-override-manifest artifacts/private/external_datasets/teachobs/s4_source_override.json \
  --acknowledge-override-source-terms
```

media manifest 必须绑定这个 override 文件的 SHA-256、记录 override terms acknowledgement，并继续保持 `publisher_byte_identity_established=false`。没有第二个确认时 runner 在下载前退出；没有 override 时 postcondition 又会拒绝任何残留的 candidate mirror 记录。因此候选“内容相同”不能被自动写成“发布者字节同一”。

## 双人标注和锁箱为何仍是 pending

第 8 阶段只生成两个不同随机顺序的空白任务表，并要求当前 profile 选中的私有媒体引用都存在（论文 profile 为 29 讲/4,945 场景，完整 profile 为 30 讲/5,158 场景）。若 A/B CSV 已经偏离生成时哈希，runner 视为可能已有人工作，立即拒绝覆盖。它不会读取 gold/预测填标签，也不会调用 agreement analysis；正式执行仍需要外部 39/39 operational codebook、两个真实独立标注人和第三方裁决。

第 9 阶段把 Stage 7 的 `bundle_manifest.json` 作为 system artifact，把四个 arm 的 `manifest.json` 分别放入对应 model slot，并继续绑定当前 `teachobs_lockbox.py`。lockbox 的 transitive verifier 必须重新加载所有 companion `arrays.npz`、核对 bundle/arm/array 哈希和共同 provenance，并逐臂绑定 shape 为 `[39]` 的 `model_thresholds` 数组摘要，才会写出 `frozen_artifact_set_complete=true`。分析计划同时绑定同一组四臂阈值摘要及阈值感知的 artifact-set fingerprint，不再假定统一 `0.5`。公开 draft 只含摘要承诺，不含私有路径、权重或阈值数值。

`frozen_artifact_set_complete=true` 只表示“系统字节已经冻结完整”，不表示“外部研究完成”。草案仍没有外部登记签名、可信公钥、目标站点数据、前瞻时间顺序、一次性消费记录或目标结果，所以 `preregistration_execution_ready=false`，确认性增益、外部锁箱、部署准确率和学习效果仍全部为 `false`。真正的确认性锁箱必须按 `docs/teachobs_confirmatory_lockbox.md` 在新站点前瞻注册和一次性执行。

## 可配置环境变量

| 变量 | 默认值/作用 |
|---|---|
| `TSM_TEACHOBS_SOURCE_TERMS_ACKNOWLEDGED` | 只能为 `true`/`false`；可代替主授权 flag |
| `TSM_TEACHOBS_OVERRIDE_SOURCE_TERMS_ACKNOWLEDGED` | override 的第二个独立确认 |
| `TSM_TEACHOBS_SOURCE_OVERRIDE_MANIFEST` | 默认空，即禁用 override |
| `TSM_TEACHOBS_PYTHON` | `python3` |
| `TSM_TEACHOBS_YT_DLP_PYTHON` | 默认与主 Python 相同 |
| `TSM_TEACHOBS_YT_DLP_MODULE_DIR` | `artifacts/private/tools/yt_dlp`；不存在时使用环境安装 |
| `TSM_TEACHOBS_CLIP_MODEL` | 本地 CLIP snapshot |
| `TSM_TEACHOBS_CLIP_REVISION` | 固定 source revision |
| `TSM_TEACHOBS_CLIP_DEVICE` | `cpu` |
| `TSM_TEACHOBS_CLIP_BATCH_SIZE` | `32` |
| `TSM_TEACHOBS_EVALUATION_PROFILE` | `paper_track1_23_train_6_test`；当前其他值失败关闭 |
| `TSM_TEACHOBS_JS_RUNTIME` | 空；不启用 runtime |
| `TSM_TEACHOBS_COOKIES_FROM_BROWSER` | 空；默认不读取浏览器 Cookie |
| `TSM_TEACHOBS_YT_DLP_DIRECT` | `false`；设为 `true` 时显式忽略继承代理 |
| `TSM_TEACHOBS_YT_DLP_IMPERSONATE` | 空；当前唯一允许值为 `chrome` |
| `TSM_TEACHOBS_YT_DLP_YOUTUBE_CLIENT` | 空；字幕请求可显式设为 `android_vr` |
| `TSM_TEACHOBS_JOBS` | `4` |
| `TSM_TEACHOBS_FEATURE_JOBS` | `2` |
| `TSM_TEACHOBS_VISUAL_JOBS` / `TSM_TEACHOBS_OCR_JOBS` | `1` / `1` |
| `TSM_TEACHOBS_FFMPEG` / `TSM_TEACHOBS_FFPROBE` | `ffmpeg` / `ffprobe` |
| `TSM_TEACHOBS_PRIVATE_ROOT` | TeachObs 私有根目录 |
| `TSM_TEACHOBS_REPOSITORY` | 完整固定 repository |
| `TSM_TEACHOBS_ANNOTATION_IMPORT` | commit-pinned 注释审计输出 |
| `TSM_TEACHOBS_ACQUISITION_RECEIPT` | 完整 repository 的获取/树哈希 receipt |
| `TSM_TEACHOBS_MEDIA_ROOT` / `TSM_TEACHOBS_CAPTION_ROOT` | 私有媒体/字幕输出 |
| `TSM_TEACHOBS_ASR_ROOT` / `TSM_TEACHOBS_ASR_RESULTS` | 私有 ASR manifest/import/coverage 根与六讲 result 目录 |
| `TSM_TEACHOBS_TRANSCRIPT_ROOT` | 私有 29 讲/4,945-scene 字幕物化目录 |
| `TSM_TEACHOBS_HUMAN_ROOT` | 私有空白双人标注骨架 |
| `TSM_TEACHOBS_HUMAN_RECEIPT` | 与所选 human root 精确配对的公开聚合 receipt；保留旧骨架时应显式覆盖 |
| `TSM_TEACHOBS_BENCHMARK_RESULT` | 私有四臂结果 JSON |
| `TSM_TEACHOBS_FROZEN_MODEL_ROOT` | 私有确定性四臂 bundle；默认 `.../frozen_models` |
| `TSM_TEACHOBS_PUBLIC_ROOT` | aggregate-only 公开 receipt 目录 |
| `TSM_TEACHOBS_LOCKBOX_STUDY_ID` | pending draft 的 study id |
| `TSM_TEACHOBS_LOCKBOX_DRAFT` | 与当前 frozen bundle/analysis code 精确绑定的公开 draft 路径 |

所有公开文件最后统一经过 `tsm release-audit`。该自动审计是发布边界检查，不替代人工披露风险复核，也不授权发布私有视频、字幕/OCR 文本、帧、嵌入、逐场景预测或标注人信息。
