# 真实课堂自动识别实验协议

## 1. 这项实验实际证明什么

本仓库把“自动识别”与原有 Teaching Skill 内部一致性评分分开实现。真实数据实验的当前任务是：输入一段真实课堂小组视频，自动输出 `low / medium / high` 三类小组投入度概率，并在独立人工标签上计算 out-of-fold 指标。

当前结论边界如下：

| 断言 | 当前状态 | 依据 |
|---|:---:|---|
| 使用真实线下课堂视频 | 是 | 数据来自 OUC-CGE 官方 OSF 发布 |
| 有独立人工真值 | 是 | OUC-CGE 由专家标注，20% 双标，Cohen's κ = 0.75（95% CI 0.71–0.79） |
| 实现视频到标签的自动推理 | 是 | FFmpeg 解码、视听觉特征、固定分类器与 JSON checkpoint 均实际执行 |
| 当前数值是完整官方基准成绩 | 否 | 本仓库当前只运行 36 个公开 sample 视频，去除 1 个逐字节重复后为 35 个样本 |
| 当前划分是跨课堂 session / 跨参与者划分 | 否 | 公开示例文件名不含可靠 session 或参与者 ID |
| 已证明部署场景准确率 | 否 | 小样本、场景单一、无法做 session-disjoint 外推 |
| 已证明多模态优于纯视觉 | 否 | low 的视频流没有重叠可用音频，而 medium/high 均有；audio/fusion 会直接利用模态缺失捷径 |
| 已证明教学效果或教师能力 | 否 | 标签是小组投入度，不是学习增益、教师能力、困惑或提问检测真值 |

因此，`benchmark_report.json` 中的 Accuracy、Macro-F1 与区间只能表述为“OUC-CGE 官方公开 sample 上的文件名伪分组交叉验证 pilot 指标”，不能写成“真实课堂部署准确率”。原有自动评估中的 `internal_evidence_consistency` 也不能替代这里的识别 Precision / Recall / F1。

## 2. 数据集选择

### 2.1 当前可直接复现：OUC-CGE

[OUC-CGE 论文](https://www.nature.com/articles/s41597-025-04987-w)、[官方 OSF](https://osf.io/brd2c/) 和[官方基线代码](https://github.com/oucyy361/OUC-CGE)共同描述了这一数据集：

- 7,705 个真实课堂小组视频片段，目标标签为低、中、高投入度；论文将最终片段描述为约 10 秒；
- 17 名 23–24 岁本科生参与 21 个 session，包含两种教室布局和前、侧、后三种拍摄角度；
- 全部片段由外部专家编码，20% 由第二位专家独立复标，报告 κ = 0.75；
- 许可出处存在冲突：论文写明“无需审批的非商业使用”，而当前 [OSF 项目 API](https://api.osf.io/v2/nodes/brd2c/) 标为 CC BY 4.0。作者澄清前，本项目采用更严格的“仅非商业研究”解释。CC BY 也不授予隐私、肖像或声音相关人格权；官方代码的 MIT License 只覆盖代码。

当前 OSF 的三个 `sample` 压缩包各含 12 个视频，共 36 个。实际审计发现它们时长不一，显然不是论文定义的完整 7,705 个约 10 秒最终切片；三个 sample 包本身也不附 `train / val / test` 清单。其中有 1 个文件与另一个文件 SHA-256 完全相同。因此本仓库在建模前去除精确重复，并按主视频流时间轴取中心 10 秒；视频流不足 10 秒时使用完整内容。论文还说明课程材料与活动经过选择以诱发不同投入度，故数据也不是无干预抽样的自然课堂分布。

还需注意，论文报告的“12 小时 50 分钟”与 `7,705 × 10 秒` 的算术结果并不一致。正式论文中应以实际下载文件的 `ffprobe` 汇总为准，并把该出处差异写入数据审计，不能自行选择更好看的数字。

### 2.2 更严格的视听觉外部验证：Pri-MCCD

[Pri-MCCD 论文](https://www.nature.com/articles/s41597-026-07280-6)提供 4,357 个约 5 秒的真实小学课堂片段，包含同步 MP4、WAV 和 IS09 声学特征，标签为 `Negative / Neutral / Positive`。共采集 15 节课，质量清理后保留 13 个完整有效课堂录制，涉及 13 名教师和约 546 名学生；两位专家独立标注，报告 κ = 0.821。论文的视听觉基线报告 Accuracy 0.7538、F1 0.7443。

Pri-MCCD 含可识别人脸与声音，而且参与者包括未成年人，所以[官方 OSF](https://osf.io/jw6kt)是受控访问。只有合格学术研究者在接受 DUA 后才能获取；DUA要求安全存储、非商业研究和禁止重新识别。本项目不会代替用户接受协议、不会绕过访问控制，也不会把其数据纳入仓库。获得正式授权后，它适合作为视听觉模型的外部数据集，但任务“课堂气候”与 OUC-CGE 的“小组投入度”并不相同，不能直接混用标签或合并计算准确率。

## 3. 当前可复现 pipeline

### 3.1 数据审计

`real-data-audit` 对每个 MP4 执行 `ffprobe`，记录主视频流起点/时长、容器时长、与视频重叠的有效音轨、大小和 SHA-256；标签来自目录，但标签名和路径文字不会进入数值特征。相同 SHA-256 的视频只保留第一份。只有全部 36 个相对路径和视频 SHA-256 与代码中固定的官方 sample 清单完全一致时，`provenance_verified` 才为 `true`；任意自定义同名目录不会被冒充为真实 OUC-CGE 人工标签数据。

当前公开示例的下载、校验和目录结构见 [`data/real/README.md`](../data/real/README.md)。

```bash
python3 -m teaching_skill_miner real-data-audit \
  --dataset-root data/real/ouc_cge/extracted \
  --output artifacts/real_classroom/dataset_audit.json
```

### 3.2 固定预处理与特征

为避免观察结果后再改预处理，pilot 使用预先固定的参数：

- 时间窗：按主视频流而非容器时长取中心 10 秒；视频流不足 10 秒时用完整片段；
- 视觉：均匀抽取 12 帧，缩放到 `64 × 36`，计算颜色直方图、空间池化亮度、时间方差、边缘和运动统计；
- 音频：只读取与同一视频时间窗重叠的第一条有效音轨，单声道 8 kHz，计算能量、过零率、频谱质心、平坦度、频带能量和静音比例；
- 分类器：只在训练折拟合 `StandardScaler`，再训练固定 `C=1.0`、类别平衡的多项 Logistic Regression；
- 消融：分别运行 `visual`、`audio` 和 `fusion`，融合增益按 Macro-F1 与纯视觉比较。

这是一条 CPU 可复现、可检查的基线，不是对 OUC-CGE 官方深度模型结果的复现。尤其要防止把压缩率、音量、时长或机位当成投入度：中心窗只能减少部分时长捷径，不能消除场景与标签的相关性。

### 3.3 划分与防泄漏

当前 36 个示例在三个类别中都使用 `view1` 至 `view12` 的命名。代码仅提取末尾数字形成与标签无关的伪分组，使用 4 折 `StratifiedGroupKFold` 生成 out-of-fold 预测；同一文件名尾号不跨训练折和测试折。每折的 scaler 和分类器只见训练数据，报告必须满足伪分组 `group_overlap = []`。

这个伪分组只避免同编号样例被拆开，**不是已验证的源视频、session 或参与者分组**。公开 sample 没有足够元数据证明 `view1` 在不同类别、机位或 session 之间的真实关系，也无法保证同一参与者不跨折。报告因此固定保留：

```json
{
  "group_disjoint_evaluation": false,
  "surrogate_filename_group_disjoint_evaluation": true,
  "verified_source_group_disjoint_evaluation": false,
  "session_disjoint_evaluation": false,
  "full_official_test_set_used": false,
  "real_world_recognition_accuracy_established": false
}
```

### 3.4 指标与产物

```bash
python3 -m teaching_skill_miner real-recognition-benchmark \
  --dataset-root data/real/ouc_cge/extracted \
  --output-dir artifacts/real_classroom \
  --folds 4 \
  --frames 12 \
  --seed 2026
```

主要产物：

- `benchmark_report.json`：Accuracy、Balanced Accuracy、Macro / Weighted F1、逐类指标、混淆矩阵、AUROC、Log Loss、Brier、ECE，以及按文件名尾号伪分组 bootstrap 的 95% 区间；
- `oof_predictions.csv`：每个样本唯一的 out-of-fold 概率与预测；
- `dataset_manifest.json`：哈希、分组、标签及媒体审计；
- `checkpoint.json`：在全部 pilot 样本上拟合、仅供本地链路调试的推理模型；不得作为公开发布或部署模型；
- `feature_diagnostics.json` 与 `feature_cache.npz`：特征抽取诊断和可复核缓存。

Macro-F1 是主指标，因为三类样本量和难度可能不同；同时报告多数类基线。当前 bootstrap 单位只是 `filename_index_surrogate_group`，不是已验证的真实来源。由于折外预测恰好全对，样本内区间退化为 `[1.0, 1.0]`；这个异常窄区间没有覆盖 session、参与者、来源与编码混杂的不确定性，不能据此声称泛化稳定。

训练后的 checkpoint 可以对新视频执行真实推理：

```bash
python3 -m teaching_skill_miner real-recognition-infer \
  --video /path/to/classroom.mp4 \
  --checkpoint artifacts/real_classroom/checkpoint.json \
  --output artifacts/real_classroom/inference.json
```

输出是自动模型预测而非人工观察，但 checkpoint 仅由公开 sample 训练，概率未经过生产环境校准。本仓库的示例推理使用训练集中的 `high/view1.mp4`，输出会明确标记 `input_was_in_checkpoint_training_set: true`，它只验证推理链路而不是独立准确率。不得据此自动处罚、排名或评价教师/学生，也不得将其用于高风险教育决策。

### 3.5 2026-07-20 实际运行结果与捷径审计

本仓库当前保存的报告来自真实运行，而非引用论文成绩：

| 模型或对照 | Accuracy | Macro-F1 |
|---|---:|---:|
| 多数类 | 0.3429 | 0.1702 |
| 仅编码元数据 | 0.9714 | 0.9710 |
| 视觉 | 1.0000 | 1.0000 |
| 音频 | 1.0000 | 1.0000（无效：音频缺失与标签完全相关） |
| 视觉 + 音频 | 1.0000 | 1.0000（无效：音频缺失与标签完全相关） |

“仅编码元数据”对照完全不读取帧或音频内容，只使用主视频流时长、每秒字节数、分辨率、编码器、视频/音频流数量和有效音轨标记。它仍取得 0.9710 Macro-F1，强烈表明三个样例包存在来源/导出条件与标签混杂的风险：low 主要为 1920×1080 H.264、视频流约 5–7 秒且没有重叠有效音频；high 全部为 1092×614 MPEG-4、约 10–15 秒且有音频；medium 又呈现不同的码率/编码组合。low 的容器时长虽显示 59–84 秒，但这是稀疏、不连续流造成的误导，不能当作实际课堂内容时长。

在视觉特征提取完成后再随机置换标签 20 次，平均 Macro-F1 为 0.3060（范围 0.1093–0.6000）。这只是“固定特征与真实标签的关联高于置换空分布”的 sanity check，**不能排除目标泄漏**；排除路径标签进入模型依赖于特征接口代码审计和 provenance 测试。视觉、音频和融合同时达到 1.000，融合相对视觉的数值增益为 0，且音频覆盖率为 low 0/11、medium 12/12、high 12/12。因此 audio/fusion 结果被标为模态缺失捷径，本地模型选择自动回退为 `visual`，但该 checkpoint 不纳入公开 release。

```json
{
  "sample_shortcut_risk_detected": true,
  "multimodal_audio_coverage_valid": false,
  "released_checkpoint_modality": "visual",
  "multimodal_gain_established_on_pilot": false,
  "real_world_recognition_accuracy_established": false
}
```

其中历史字段名 `released_checkpoint_modality` 表示本地 pipeline 选中的推理模态，不代表 checkpoint 已获准公开分发或具备部署资格。

这些检查解释了为什么“数值是真实计算的”和“数值能代表真实部署”是两件不同的事。

## 4. 怎样得到可发表而非 pilot 的准确率

OSF 完整分卷已核对为 low 3,028、medium 2,041、high 2,636，共 7,705 段，压缩体积约 25.5 GB；官方 GitHub 另有 80/10/10 的片段级清单。当前网络实测约 1 MB/s，未把需要约 7 小时的半成品下载混入本次报告。即使后续完成全量官方 split，它仍没有 session/参与者 ID，结果最多称为“官方片段划分指标”。

至少完成以下条件后，才可把结果称为“完整真实数据集上的识别性能”：

1. 从官方来源取得完整 7,705 片段、官方 CSV 和可验证校验和，冻结一个数据版本；
2. 获得 session、参与者、机位与课程元数据，按 session 或参与者分组，而不是随机 clip 切分；若官方不提供这些字段，应联系作者，不能声称跨课堂泛化；
3. 预注册主指标、预处理、超参数和停止规则，只用训练/验证集选择模型，最终测试集只评一次；
4. 报告每类样本量、混淆矩阵、Macro-F1、Balanced Accuracy、校准误差和 group-bootstrap 95% 区间，不只报告 Accuracy；
5. 做 `visual / audio / fusion` 消融，并检查音频、编码、背景、机位、课程和时长捷径；
6. 在另一所学校或 Pri-MCCD 等授权数据上做外部验证，明确处理不同标签定义，不能把域内成绩当跨域成绩；
7. 按性别、布局、课程、机位等可合法使用的分组做误差审计，并由人工复核典型误判；
8. 若要宣称促进学习或改进教学，另行进行有对照的前后测或课堂实验；识别准确率本身不提供因果证据。

完整验证前，推荐答辩措辞是：“系统已经在真实、有人类标签的课堂视频上完成自动推理与可复现 pilot 评估；当前数值验证工程可行性和初步可分性，尚未建立跨 session 或部署准确率。”

## 5. 隐私与安全

- 原始视频包含可识别的人脸、声音和课堂关系，只保存在受控本地目录；不要提交到 Git、公开 artifact、模型托管平台或演示网页。
- 不做人脸识别、身份匹配、声纹识别或跨数据源重新识别；日志和预测只使用随机 `sample_id`。
- 对外展示优先使用聚合指标；必须展示画面时，先确认数据条款和机构伦理要求，并对无关人员做模糊化处理。
- 限制访问者、记录数据版本与处理目的，训练完成后按数据协议删除不再需要的副本。
- 涉及 Pri-MCCD 等未成年人数据时，严格执行 DUA；公开论文不代表公开原始数据。
