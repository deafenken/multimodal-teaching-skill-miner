# 真实课堂数据（本地使用，不纳入仓库）

该目录用于本地保存含人脸和声音的课堂视频。不要把压缩包、解压视频、抽帧、音频或可反推身份的中间产物提交到 Git 或上传到公开 artifact。

## OUC-CGE 官方公开示例

来源：

- 数据论文：[A Video Dataset for Classroom Group Engagement Recognition](https://www.nature.com/articles/s41597-025-04987-w)
- 官方数据页：[OSF `brd2c`](https://osf.io/brd2c/)
- 官方基线代码：[oucyy361/OUC-CGE](https://github.com/oucyy361/OUC-CGE)

许可出处存在冲突：论文说明无需审批的非商业使用，而当前 [OSF 项目 API](https://api.osf.io/v2/nodes/brd2c/) 标为 CC BY 4.0。作者澄清前请采用更严格的“仅非商业研究”解释；CC BY 不等于放弃隐私、肖像和声音相关人格权。所有参与者为成年人并同意研究视频在线共享，但仍需遵守机构伦理与数据治理要求。官方代码的 MIT License 只覆盖代码。

当前复现实验使用 OSF 的 3 个示例压缩包：

| 类别 | 官方下载 | 本次核查 SHA-256 | 大小约 |
|---|---|---|---:|
| low | [low sample](https://osf.io/download/xcwj8/) | `4575e7fd9d433ffc1b6da1ba8a28eacd4feb95b7afdd0264cfd7a9f6599aa778` | 8 MB |
| medium（本地目录名 `mid`） | [mid sample](https://osf.io/download/dnex8/) | `ec5f5db9868bf8e3ccfd72ec04f965e6adad767276112240aba96bf7a0f6ad42` | 97 MB |
| high | [high sample](https://osf.io/download/aw2kp/) | `f6cae1ce079b5e698cdc9db3654d163da6a7b03877d7701c7f0d816c2281ee2d` | 34 MB |

下载后建议保持以下结构：

```text
data/real/ouc_cge/
  raw/
    low_sample.zip
    mid_sample.zip
    high_sample.zip
  extracted/
    low/*.mp4
    mid/*.mp4
    high/*.mp4
```

可手工下载，也可执行仓库提供的下载脚本（若已生成）：

```bash
sh scripts/download_ouc_cge_samples.sh
```

不使用脚本时，请至少核对压缩包：

```bash
shasum -a 256 data/real/ouc_cge/raw/low_sample.zip
shasum -a 256 data/real/ouc_cge/raw/mid_sample.zip
shasum -a 256 data/real/ouc_cge/raw/high_sample.zip
```

## 已知数据边界

- 这些是 36 个官方公开 sample 视频，不是完整 7,705 个最终片段；三个 sample 压缩包本身没有随附 `train / val / test` CSV。
- OSF 完整分卷已确认可访问：low 3,028、medium 2,041、high 2,636，共 7,705 段，压缩包约 25.5 GB；当前报告没有使用这些全量分卷。官方 GitHub 提供 80/10/10 的片段级清单，但没有 session/参与者字段。
- sample 视频时间轴不一致；本项目按主视频流而非容器时长取中心 10 秒，不足 10 秒时使用完整视频流。
- 当前文件中 `low/view1.mp4` 与 `low/view2.mp4` 的 SHA-256 相同；数据审计会在建模前删除第二份，留下 35 个唯一视频。
- 文件名不含可靠 session、参与者或机位元数据。按 `view` 数字做的分组只是一种保守防泄漏措施，不能声称 session-disjoint。
- OUC-CGE 论文将任务定义为纯视觉小组投入度识别。审计确认 low 0/11 个视频有与画面重叠的有效音频，medium/high 则都是 12/12；audio/fusion 的 1.0000 会利用模态缺失，不能作为多模态准确率。
- 三个类别包的视频流时长、分辨率、编码器、码率和有效音轨明显不同；本地仅编码元数据基线就达到 Macro-F1 0.9710，强烈表明来源/编码捷径风险。系统因此只在本地选择 visual checkpoint 调试推理链路，并明确不把它纳入公开 release。

先运行审计，再运行基准：

```bash
python3 -m teaching_skill_miner real-data-audit \
  --dataset-root data/real/ouc_cge/extracted \
  --output artifacts/real_classroom/dataset_audit.json

python3 -m teaching_skill_miner real-recognition-benchmark \
  --dataset-root data/real/ouc_cge/extracted \
  --output-dir artifacts/real_classroom \
  --folds 4 --frames 12 --seed 2026
```

完整实验设计、指标解释和不得声称的结论见 [`docs/real_classroom_protocol.md`](../../docs/real_classroom_protocol.md)。

## Pri-MCCD 不在此目录自动下载

[Pri-MCCD](https://www.nature.com/articles/s41597-026-07280-6)含真实小学课堂视听觉数据，因可识别的未成年人面部与声音而在[官方 OSF](https://osf.io/jw6kt)受控开放。只有合格学术研究者接受 DUA 后才可取得。请勿尝试绕过授权、共享账号或转发原始数据；本项目也不会代替用户接受协议。

## DIPSER V5：RGB 派生姿态 + 智能手表探索性基准

来源：

- DOI：[10.57760/SCIENCEDB.11541](https://doi.org/10.57760/SCIENCEDB.11541)
- 官方数据页：[ScienceDB dataset `7856c716c0cc4589a23ee4a23d8a0893`](https://www.scidb.cn/en/detail?dataSetId=7856c716c0cc4589a23ee4a23d8a0893)
- 论文：[DIPSER](https://arxiv.org/abs/2502.20209)

DIPSER 来自真实线下大学课堂，完整数据包含 3 个 cohort、9 个 experiments、54 名学生的相机数据、智能手表传感器、4 位专家的 attention/emotion change-point 标注和学生自评；发布方报告采集系统同步误差小于 0.5 秒。真实归档中每个 subject 应恰好包含四个 publisher expert `labeler_*.json`，ID 可为 `01`–`05`；V5 实际存在 `01/02/03/04` 和 `01/02/03/05` 两种组合，其中 22 个 archive 为后一种，另有不参与专家真值的 `self_labeling`。当前 `visual` 分支**不读取图像像素**，只使用发布方从 RGB 生成的 head/body pose metadata；结果只能写成“RGB-derived pose metadata + watch”，不能写成原始视频或端到端 RGB 识别率。

截至 2026-07-20，ScienceDB 当前把数据列为公开 **V5**、许可标记为 **CC BY 4.0**；论文却同时写有“仅 academic/research purposes”和“商业用途须明确批准”。作者澄清前，本项目采用更严格的学术、非商业边界。CC BY 不等于授权商业监控，也不免除人脸、生物传感器、隐私、知情同意和机构伦理义务。

当前默认清单包含 **52 个 archives、52 个只出现一次的 participant、27 个 cohort×activity recording cells**，覆盖 3 个 cohort 的 9 种 experiments。该清单是在部分标签清单审查后冻结的探索性分析清单，**不是正式预注册**。代码中的 `DEFAULT_PREREGISTERED_ASSIGNMENT` 只是历史名称。身份定义为：

```text
participant_id = group_XX/subject_YY
session_id     = group_XX/experiment_ZZ
activity_id    = experiment_ZZ
```

不要下载或解压约 717 GB 的完整 V5。实现通过 HTTP Range 读取中央目录和 non-image ZIP suffix：所需尾部不超过 128 MiB 时做有界合并读取；异常布局超过上限时，只对必需的 metadata、恰好四个 publisher expert labeler 和稀疏 watch JSON 逐成员 Range 读取，不下载整个 archive。以下命令可用于审计单个 archive 的中央目录或显式成员；成员名必须以 `--list` 的官方结果为准：

```bash
python3 scripts/download_dipser_subset.py \
  --url "https://china.scidb.cn/download?fileId=OFFICIAL_FILE_ID" \
  --size OFFICIAL_ARCHIVE_BYTES \
  --list

python3 scripts/download_dipser_subset.py \
  --url "https://china.scidb.cn/download?fileId=OFFICIAL_FILE_ID" \
  --size OFFICIAL_ARCHIVE_BYTES \
  --output data/real/dipser_v5/group_XX/experiment_ZZ/subject_YY \
  --member "EXACT_PUBLISHER_EXPERT_LABELER_PATH_FROM_LIST"
```

对四个实际 expert labeler 成员分别重复 `--member`；不得假设第四位专家必然是 `labeler_04`，也不得把 `self_labeling` 当作第四位专家。

正式 pipeline 以 15 秒固定网格稀疏选择 watch JSON。watch 文件名时间与 JSON 内部传感器行中位时间差不得超过 1.0 秒，JSON 内时间相对中位数最大偏差不得超过 1.0 秒，然后再与最近的 metadata 官方文件名时间对齐，误差不得超过 0.6 秒。0.6 秒是代码的配对容差，不能与论文报告的采集系统同步误差小于 0.5 秒混写。

一个合格时点必须同时具有：3 个有限 head-pose 数值；恰好 33 个 body landmarks，且每个的 `x/y/z/visibility/presence` 都有限；心率、线性加速度、陀螺仪、旋转向量和环境光五类 watch 数据的每一行都完整且有限；以及归档内恰好四个 publisher expert labeler 在该时点全部有状态、至少 3/4 在 `1–2=low、3=medium、4–5=high` 中同档的真值。`self_labeling` 始终排除。这次按实际 labeler 集合解析的修正不放宽“四专家齐全 + 3 票多数”门槛。availability、presence、sample-count 和 landmark-count 均不进入模型。

评估包含：按实际 25 个有完整样本的 recording/session 分组的 nested 5-fold 同设计内描述性 OOF；因 session overlap 只作描述的 participant-only 次分析；以及固定 `LogisticRegression(C=1.0)` 的 3 折 LOCO、9 折 LOAO 和最多 27 个计划 cohort×activity 双重阻断 cell。后三套不计算 CI 或 `p`，所有增益结论均为 `false`。nested 报告中的 session bootstrap/permutation 仅是诊断：计划的 27 个 cell 正好是 3 cohort×9 repeated activity，实际只有 25 个 cell 含完整样本，二者都不能当作独立课堂总体单位。

运行默认真实清单：

```bash
python3 -m pip install -e '.[recognition]'

python3 -m teaching_skill_miner dipser-credible-benchmark \
  --output-dir artifacts/dipser_credible/full_v5_52 \
  --workers 2 \
  --outer-splits 5 --inner-splits 3 \
  --bootstrap-replicates 2000 \
  --permutation-replicates 5000 \
  --seed 2026
```

报告位于 `artifacts/dipser_credible/full_v5_52/credible_report.json` 和 `blocked_descriptive_report.json`；清单与特征位于同目录的 `dataset_manifest.json` 和 `features.json`。`archive_cache/` 支持断点续跑，缓存键绑定官方 archive 身份、配置与提取源码实现指纹，内容有 SHA-256。特征 artifact 额外绑定有序样本 ID/标签/特征名/精确矩阵，报告记录源码和 Python/NumPy/SciPy/scikit-learn 等环境指纹。这些产物不保存原始 ZIP/JSON/JPEG，但仍属于受控的逐人研究数据，不得提交 Git 或公开分发。

完整声称边界见 [`docs/dipser_credible_benchmark.md`](../../docs/dipser_credible_benchmark.md)。当前所有 `session_disjoint_accuracy_established`、`participant_disjoint_accuracy_established`、多模态增益与 `deployment_accuracy_established` 都必须为 `false`；部署准确率需要冻结模型在与开发数据独立的目标站点上进行前瞻、一次性的正式测试才可能建立。

已完成的 52/52 官方归档实跑、296 个完整案例、三套阻断结果和精确指纹见 [`artifacts/dipser_credible/full_v5_52_complete_v5/RESULTS.md`](../../artifacts/dipser_credible/full_v5_52_complete_v5/RESULTS.md)。该结果没有建立多模态增益、可接受准确率或部署准确率。
