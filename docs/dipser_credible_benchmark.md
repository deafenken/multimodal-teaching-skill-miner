# DIPSER 可信多模态探索性基准

本文定义当前实现如何在 DIPSER 上生成可审计的“RGB 派生姿态 + 智能手表”描述性结果，以及为什么这些结果还不能建立可推广的多模态增益、跨 session 准确率或部署准确率。

这不是正式预注册。当前 52-archive 清单是在做过部分标签清单审查后才冻结的，因此分析状态必须写为 **“部分标签清单审查后冻结的探索性分析”**。代码中保留的 `DEFAULT_PREREGISTERED_ASSIGNMENT` 是历史标识符，不是独立时间戳或正式预注册的证明。本文不预填任何数值成绩；只有实际运行后的清单、折外预测、覆盖率与完整审计才是结果。

## 1. 数据来源、版本与使用边界

正式来源：

- 数据记录 DOI：[10.57760/SCIENCEDB.11541](https://doi.org/10.57760/SCIENCEDB.11541)
- ScienceDB 数据页：[DIPSEER / DIPSER 数据记录](https://www.scidb.cn/en/detail?dataSetId=7856c716c0cc4589a23ee4a23d8a0893)
- 数据论文：[DIPSER: A Dataset for In-Person Student Engagement Recognition in the Wild](https://arxiv.org/abs/2502.20209)

截至 2026-07-20，本项目核查到 ScienceDB 页面/API 将数据列为公开的 **V5**，平台许可标记为 **CC BY 4.0**。论文一方面写明数据公开于 CC BY，另一方面又要求仅用于 academic/research purposes，商业用途须获得数据保管方明确批准。两种措辞存在实质张力。

在作者或数据保管方书面澄清前，本项目采用较严格边界：只用于经伦理和数据治理审查的学术、非商业研究；不把 CC BY 解释为授权商业课堂监控、身份识别或高风险教育决策。商业使用或真实部署还须取得数据保管方明确授权，并满足适用法律、学校审批和知情同意要求。

V5、公开状态和平台许可都可能变化。正式运行必须记录核查日期、ScienceDB dataset ID、版本、每个 source archive 的 file ID、大小和发布方 MD5，不能只保存可变化的下载 URL。

## 2. 数据事实与本实现的视觉边界

DIPSER 是在西班牙阿利坎特大学教育学院真实线下课堂中采集的数据，不是在线视频会议或演员模拟。论文报告：

- 3 个既有学生 cohort，共 54 名学生；
- 每个 cohort 有 9 个教学 experiment；
- 数据源包含个体/环境相机的 RGB image sequences，以及心率、线性加速度、陀螺仪、旋转向量和环境光等智能手表传感器；
- 设备由中央服务器校时，并以亮度变化复核；发布方报告相机与时钟时间戳误差小于 0.5 秒；
- 每名学生有 4 位专家的 attention/emotion change-point 标注和学生自评。发布归档审计显示，每个 subject archive 内应恰好有四个 publisher expert `labeler_*.json` 文件，labeler ID 可为 `01`–`05`；V5 实际存在 `01/02/03/04` 和 `01/02/03/05` 两种组合，其中 22 个归档为后一种，另有 `self_labeling` 学生自评文件。

当前基准的视觉输入不是原始 RGB 像素。它读取发布方由 RGB 处理得到的 metadata JSON，只保留 head pose（pitch/yaw/roll）和 body pose 的去身份化统计；人脸像素、face mesh、年龄、性别、族裔估计和身份字段均不进入模型。因此，结果只能表述为：

> DIPSER V5 发布方 RGB-derived head/body pose metadata 与 watch sensors 的多模态评估。

不得将其写成“原始课堂视频模型”“端到端 RGB 视频识别率”或“RGB 像素 + watch 增益”。DIPSER 的 RGB 来源支持这些姿态 metadata 的视觉来源，但本实现没有读取或训练 raw RGB。

预测目标是学生在同步时点的专家 attention 三档标签，不是学习效果、教师教学能力、情绪诊断或因果干预效果。数据来自同一大学场地，域内跨 session 结果不能直接外推到其他学校。

## 3. 分析冻结的 52-archive 子集

当前默认子集冻结为 **52 个 subject-session archives、52 个不同 participant、27 个 cohort×activity recording cells**。每个 participant 在该子集中只出现一次，因此按 recording/session 分组的 OOF 折之间没有 participant 交叉。但该清单是在部分标签清单审查后冻结，只能作为探索性分析清单，不能声称为标签盲法预注册抽样。

| cohort | `experiment_01` 至 `experiment_09` 的 subject 分配 |
|---|---|
| `group_01` | `01,02 / 03,04 / 05,06 / 07,08 / 09,10 / 12,13 / 14,15 / 16,17 / 18,19` |
| `group_02` | `01,03 / 07,11 / 02,04 / 05,06 / 08,09 / 10,12 / 13,14 / 15,16 / 17,18` |
| `group_03` | `01,02 / 03,04 / 05,06 / 07,09 / 10,11 / 08,12 / 13 / 14 / 15,16` |

斜线依次分隔 9 个 experiments。`group_03/experiment_07` 和 `group_03/experiment_08` 各有 1 个冻结 participant，其余 session 各有 2 个。archive 未恰好解析出四个 publisher expert labeler 文件、缺少 watch 或视觉 metadata 时，代码记录失败或排除原因；运行时不能因标签覆盖率或模型成绩而换人。

身份只从发布方目录层级解析，不做人脸匹配：

```text
participant_id = group_XX/subject_YY
session_id     = group_XX/experiment_ZZ
archive        = group_XX/experiment_ZZ/subject_YY.zip
site_id        = university_of_alicante_faculty_of_education
```

`subject_01` 只在其 cohort 内唯一，不能省略 `group_XX`。相同 experiment 编号在不同 cohort 中是不同 recording/session，但共享同一种 repeated activity。因此 roster 计划的 27 个 session cells 正好是 **3 个 cohort × 9 种 repeated activity** 的交叉设计，complete-case 数据实际只有 25 个 cell 含样本；它们不是 27 个或 25 个可交换、完全独立抽样的课堂总体单位。52 archives 也不是完整 DIPSER；结果必须明确限定为“DIPSER V5 分析冻结的 52-archive 探索性子集”。

## 4. 15 秒稀疏采样、同步与 Range 读取

### 4.1 实际同步规则

当前实现以 **15 秒**为固定采样间隔，处理步骤如下：

1. 从 ZIP 中的 watch JSON 官方成员文件名解析时间，排序后保留首个时点，并按 15 秒固定网格稀疏选择后续 watch 成员；
2. 只对选中的 watch JSON 读取内容，优先使用发布方字段 `time`，并从实际传感器行计算中位时间戳；
3. watch 文件名时间与内部中位时间差不得超过 **1.0 秒**，且选中 JSON 内任一可用内部时间相对中位数的最大偏差不得超过 **1.0 秒**；
4. 从发布方 visual metadata JSON 的官方成员文件名解析采集时间；metadata 内容本身没有可用时间字段；
5. 为每个通过前述审计的 watch 内部中位时间戳寻找最近的 metadata 文件名时间戳；两者循环日内时间差不超过 **0.6 秒**才接受；
6. 用该 watch JSON 和配对 metadata JSON 构成同步样本，并记录文件名时间、内部中位时间、最大内部偏差、metadata 时间、实际误差和容差。

0.6 秒是当前代码的验收容差，不应改写成论文报告的“同步误差小于 0.5 秒”。前者是本实现的配对规则，后者是发布方对数据采集系统的技术验证说明，两者必须分别报告。

每个被接受时点的 watch 特征来自该次选中 JSON 内传感器行的固定统计，不是过去 10 秒或 15 秒滑动窗聚合。视觉特征来自与其配对的发布方 RGB-derived head/body pose metadata，不读取最近的 JPEG 帧。

当前采用严格 complete-case 规则：head pose 的 pitch/yaw/roll 全部为有限数；body pose 必须恰有 **33 个 landmarks**，且每个 landmark 的 `x/y/z/visibility/presence` 都为有限数；心率、线性加速度、陀螺仪、旋转向量和环境光五类 watch 传感器都必须有非空行，且每行预期数值字段全部完整且有限。缺失时间戳、任一时间一致性检查失败、专家无共识或任一数值模态不完整时，该时点从 visual、sensor 和 fusion 三个分支共同排除，防止缺失模式成为捷径。报告同时披露从“有时间的 watch 成员 → 15 秒网格 → 时间/metadata 对齐 → 3/4 专家共识 → 完整数值多模态”的覆盖流。

### 4.2 Range 稀疏读取

ScienceDB V5 全量约 717 GB，subject archive 通常约 1 GB。当前实现不下载 raw RGB image sequences，而是：

1. 通过 HTTP `Range` 读取 ZIP 尾部和中央目录；
2. 识别归档内恰好四个 publisher expert `labeler_*.json`、`self_labeling`、watch JSON 和 RGB-derived metadata JSON；专家文件按归档内实际集合解析，不假设尾号固定为 `01`–`04`；
3. 所需 non-image ZIP suffix 不超过 128 MiB 时进行有界尾部合并读取；异常布局超过该上限时，回退为只对必需的 label/metadata/watch 成员逐成员 HTTP `Range` 读取，不下载整个 archive；
4. 对读取成员执行 ZIP CRC32 和路径检查；
5. 在 provenance 中记录 archive file ID、发布方 MD5、成员路径、大小和 CRC。

Range 读取只降低数据搬运量，不改变数据的敏感性或许可。逐人 watch 轨迹、派生姿态和专家标签仍只能保存在受控本地目录，不能公开发布可重建身份的逐样本数据。

长任务会在输出目录的 `archive_cache/` 为每个成功 archive 保存去身份特征、标签与审计信息，不保存 ZIP 尾部、JSON 原文或图像。缓存键绑定官方 path、file ID、MD5、大小、15 秒间隔、三类时间容差、特征策略和 **当前提取源码实现指纹**，内容另有 SHA-256；配置变化、实现变化、发布方文件变化或缓存损坏时不会复用。该缓存仍包含逐参与者衍生轨迹，必须作为受控研究数据，不得提交 Git 或公开分发。

## 5. 四专家真值

专家身份来自当前 archive 中恰好四个 publisher expert labeler 文件，而不是固定的文件尾号。解析修正只是兼容 V5 的两种真实组合（`01/02/03/04` 和 `01/02/03/05`），**不放宽四专家或 3 票多数门槛**。`self_labeling` 始终从专家真值中排除。

主任务将原始 attention 映射为三档：

| 原始 attention | 主任务标签 |
|---:|---|
| 1–2 | `low` |
| 3 | `medium` |
| 4–5 | `high` |

规则固定如下：

1. 每位专家的 change points 独立按时间排序；attention 和 emotion 状态分别维护；
2. 每位专家只向未来 forward-fill 最近一次 attention，首次 attention change point 之前保持缺失，不 backward-fill；
3. 四位专家分别映射为三档；
4. 四位专家在该时点都必须有有效状态，且至少 3/4 位处于同一档，才接受该档为真值；
5. 无三人多数、任一专家缺失或时间戳不能对齐时排除，并按 archive/session 披露原因和数量。

不能先对 1–5 分取均值或中位数再切档。学生 `self_labeling` 在当前基准中完全排除，不参与真值、特征选择、调参或结论；若未来做敏感性分析，必须另行声明并与专家主分析分表报告。

## 6. 特征与训练折隔离

三个模型分支使用完全相同的合格样本和外层折：

- `visual`：发布方 RGB-derived head/body pose metadata 的固定白名单统计；不把
  detector availability 或 landmark count 输入模型；
- `sensor`：固定 5 个 watch sensor 通道的数值描述统计；不把 channel presence 或
  JSON sample count 输入模型；
- `fusion`：拼接 visual 与 sensor，不增加标签、路径或身份元数据。

年龄、性别、族裔、face mesh、原始像素、任何模态存在/缺失或采样条数标记、`participant_id`、`session_id`、`cohort_id`、`activity_id` 和文件路径均不得进入特征。视觉与传感器特征提取器的 ID、字段集合和 SHA-256 指纹必须写入 artifact，并声明在评估前冻结、未使用真值标签。`feature_bundle_fingerprint` 还绑定有序 `sample_id`、内容哈希、标签、模态特征名顺序以及精确 float64 特征矩阵；严格评估在训练前再验证特征行与清单顺序完全一致，防止静默错位。

运行产物同时记录 `blocked_evaluation.py`、`dipser.py`、`dipser_experiment.py`、`strict_evaluation.py` 和 `metrics.py` 的 SHA-256，以及 Python 实现/版本、平台、zlib、NumPy、SciPy 和 scikit-learn 版本，并生成 `evaluation_environment_fingerprint`。这些指纹用于复现与比对，不替代统计外推证据。

标准化、特征选择、概率校准、正则化参数和可学习融合权重只能在每个外层训练折内拟合。不能先在全部 52 archives 上拟合转换器，再用同一批样本交叉验证。

## 7. 评估设计与它们能回答的问题

### 7.1 recording/session 分组的 nested 5-fold：同设计内描述性 OOF

第一套分析以 27 个 `session_id = group_XX/experiment_ZZ` 为分组键：

- 外层使用带固定随机种子的 5 折 `StratifiedGroupKFold` 生成 OOF 预测；
- 每个外层训练折内部再以 session 分组做 3 折 CV，选择各分支的正则化参数；
- nested best-unimodal 只根据当前外层训练折的内层结果从 visual/sensor 中选择；
- 一个 recording/session 的全部 participant 和时点只进入同一个外层测试折，并审计 participant 不跨折。

这套切分确实留出了完整 recording，但测试 recording 对应的 cohort 或 activity 可能仍在训练折中出现。由于计划的 27 个 recording cells（实际 25 个含完整样本）只是单一站点中 3 个 cohort 与 9 种 repeated activity 的交叉，不能把它们当成从课堂总体独立抽样的可交换单位。因此它只是 **单站点、同一设计内、完整案例条件下的样本加权描述性 OOF**，不是新 cohort、新 activity、新教师、新学校或部署人群准确率。

`session_blocked_oof_estimate_available: true` 只表示分组 OOF、participant 无交叉和特征顺序等机械审计通过，不表示准确率达到某个可接受阈值。包装层因此始终将 `session_disjoint_accuracy_established` 和 `session_disjoint_multimodal_gain_established` 置为 `false`。这不是 leave-one-session-out；报告会写出 `outer_splits = 5`、`inner_splits = 3` 和每折真实分组清单。

### 7.2 participant-only 分组：共享 session 的较弱描述

第二套分析以 52 个 `participant_id` 为分组键，同样使用外层 5 折和训练折内 3 折调参。它保证同一 participant 的全部时点不跨折，但同一个 session 中的另一名 participant 可能落入另一侧，从而共享课堂活动、时间、教师和环境。

代码逐折报告 `session_overlap`。对当前 DIPSER roster，participant-only 分析因存在 session overlap 一律只作描述，`participant_disjoint_accuracy_established` 和 `participant_disjoint_multimodal_gain_established` 必须为 `false`。它不能与第一套结果择优发布，也不提供独立新课堂的证据。

### 7.3 三套更保守的固定模型阻断分析

`blocked_descriptive_report.json` 额外运行三套不调参的固定分析。三个分支均使用 `StandardScaler` 在当前训练折内拟合，然后拟合 `LogisticRegression(C=1.0, solver="lbfgs")`；visual、sensor 和 fusion 使用完全相同的样本与折，没有内层选参或结果后挑模型。

| 设计 | 预期折数 | 测试与训练规则 | 可回答的有限问题 |
|---|---:|---|---|
| leave-one-cohort-out (LOCO) | 3 | 每次测试 1 个 cohort，训练其余 2 个 | 同站点、同 activity 设计下对留出 cohort 的描述 |
| leave-one-activity-out (LOAO) | 9 | 每次测试 1 种 activity，训练其余 8 种 | 同站点、同 cohort 集合下对留出 activity 的描述 |
| cohort×activity 双重阻断 | 27 | 测试 1 个 cell；训练集同时排除测试 cohort 和测试 activity，共享任一因子的其余样本 embargo | 在单站点中同时留出 cohort 与 activity 组合的最保守描述 |

每套分析报告样本加权 pooled OOF Accuracy/Macro-F1、cohort×activity cell 等权指标、fusion 相对 visual/sensor 的描述差值、OOF 覆盖率和不可评估折及原因。这三套分析不产生 CI 或 `p` 值，`contains_inferential_statistics` 为 `false`；即使 fusion 点差值为正，所有 `multimodal_gain_established`、`cross_site_accuracy_established` 和 `deployment_accuracy_established` 仍为 `false`。

### 7.4 查看旧结果后的优化分析：达到 0.8，但不追溯改写原结论

原冻结报告生成后，项目又在同一批 296 个 complete cases 上进行了模型探索。为了不把研究者自由度藏进“可信报告”，优化代码和产物与原 `credible_report.json` 分离，并统一写入 `post_selection_exploratory: true`。当前**事后开发协议中固定**的候选为：

- 输入是按冻结顺序拼接的 30 维 visual pose 与 75 维 watch 特征；身份、路径、文件名、标签和时间都不进入估计器；
- 每个训练折内单独拟合 `StandardScaler + SVC(kernel="linear", C=0.1)`；
- 因果版本只在同一 `session_id + participant_id` 内平均当前及过去 1/2/3/5 个 decision score，超过 60 秒缺口清空历史，绝不使用未来 score；
- 每个外层训练折用 session-grouped inner 5-fold 按 Accuracy、Macro-F1、较短窗口的顺序选窗口；外层测试标签只用于最终计分；
- 同时报告固定模型的 Visual/Sensor/Fusion 消融，以及 LOCO、LOAO、cohort×activity 双重阻断；
- 另有需要完整 participant-recording 的离线 pooling 候选。它在同一序列内对概率做几何平均，只有 minority/medium 比值至少为 3 才从 medium 改判；该模式不能表述为实时识别。

真实复跑结果如下：

| 设计 | Accuracy | Macro-F1 |
|---|---:|---:|
| 多数类 medium | 0.7568 | 0.2872 |
| Session SGKF5，逐窗 Fusion | 0.8041 | 0.6362 |
| Session SGKF5，训练折内选择因果窗口 | 0.8176 | 0.6357 |
| Leave-one-session-out，逐窗 Fusion | 0.7872 | 0.6085 |
| Leave-one-session-out，训练折内选择因果窗口 | 0.8176 | 0.6249 |
| 离线整段 pooling，Session SGKF5 | 0.8480 | 0.6631 |

两个 session 设计的所有外折都实现 session、participant 和 sample 零交叉；逐样本因果历史也经过“不跨序列、不跨 60 秒 gap、不含未来时点”审计。因此这些是**真实计算的同设计跨 recording 开发期 OOF 点估计**，并非靠随机 window 切分或多数类 Macro-F1 伪造的 0.8。

它们仍不能升级为确认性准确率。原因是 linear-SVC、时序窗口集合和离线阈值都是查看本数据旧 OOF 后提出；而同一事后开发协议中固定的 linear-SVC 在 LOCO、LOAO 和双重阻断下的 Fusion Accuracy 只有 0.6791、0.7230 和 0.6520。因果 SGKF5 按 session 聚类的事后 bootstrap Accuracy 区间为 `[0.7157, 0.8866]`，相对折内多数类的 Accuracy 差区间为 `[-0.0163, 0.1667]`。这些区间只能描述 cluster 敏感性，不能消除模型选择偏差。

因此原 `credible_report.json` 不回填新分数，所有 `*_established` 保持 `false`。完整优化报告见 [`OPTIMIZATION_RESULTS.md`](../artifacts/dipser_credible/full_v5_52_complete_v5/OPTIMIZATION_RESULTS.md)；机器可读逐折与逐样本审计见同目录 `optimization_report.json`。

### 7.5 离线层级 0.9 挑战：单一划分触及，不构成稳定 0.9

在上述产物继续被查看后，又形成了一个更强但推理边界更窄的 post-selected 层级候选：

- outer/inner 训练窗的 `StandardScaler + balanced LogisticRegression(C=0.1)` 生成三类概率，在 held-out `session_id + participant_id` 完整序列内取几何均值；minority/medium 比值至少为 3 才从 medium 改判；
- 独立的 `StandardScaler + linear SVC(C=0.1)` 逐窗预测在同一完整序列内多数投票，只用于确认 low；
- 每个训练 session 跨 participant 对 105 维特征取 median+IQR，以训练 session 的窗口多数标签监督 `StandardScaler + RBF-SVC(C=3, gamma=0.003)`；held-out session 使用同样的**无标签完整 session**摘要识别 high context；
- 每个 outer-training partition 内，用 session-grouped inner SGKF5 从 10 个固定 gate 中按 Accuracy、Macro-F1、注册顺序选择；每个 inner fold 重新拟合三套 scaler/model，outer-test 标签不进入聚合或选择；
- 身份、路径、时间、标签和序列长度不作为估计器特征或 gate。SVC hard context 来自 `predict()`，校准概率只用于可选 ambiguous-high guard。

真实结果：

| 设计 | Accuracy | Macro-F1 |
|---|---:|---:|
| SGKF5 seed=2026 | 0.9020 | 0.7435 |
| Leave-one-session-out | 0.8986 | 0.7397 |
| 50 个 SGKF outer seed 均值 | 0.8883 | 0.7172 |
| Visual-only，同 nested 协议 | 0.6284 | 0.2882 |
| Sensor-only，同 nested 协议 | 0.7973 | 0.5811 |

固定 SGKF5 的 0.9020 是 267/296 个窗口正确，混淆矩阵为 `[[7,16,3],[1,218,5],[0,4,42]]`；low/medium/high Recall 为 0.2692/0.9732/0.9130。50 个 outer seed 只有 21 个达到 0.9020，其余 29 个为 0.8784。确定性 LOSO 也低于 0.9，所以“稳定跨 session 0.9”没有建立。

以 outer 描述指标较高的单模态 Sensor 作保守、非预声明比较，Fusion 的 SGKF5 观察增量为 +0.1047 Accuracy、+0.1624 Macro-F1，LOSO 增量为 +0.0811/+0.1577；这说明当前开发数据上存在可复现的模态互补现象。但算法、gate、模态组合和“最佳单模态”比较都来自同一数据，且固定 gate 的 LOCO/LOAO/双重阻断仅为 0.6554/0.8919/0.6351，因此 `multimodal_gain_established` 仍为 false。

最关键的语义限制是该模型需要完整测试 batch。反事实审计在每个 mode 各自 outer-training 内重新选择 gate：

| 上下文 | SGKF5 Accuracy / Macro-F1 | LOSO Accuracy / Macro-F1 |
|---|---:|---:|
| 完整 sequence + 完整 session | 0.9020 / 0.7435 | 0.8986 / 0.7397 |
| 排除当前预测行 | 0.8547 / 0.6603 | 0.8716 / 0.7055 |
| 只用 timestamp 前缀，gap>60 秒重置 | 0.7770 / 0.5822 | 0.8378 / 0.6394 |

完整 SGKF5 中有 253 行使用未来 sequence 特征、271 行使用未来 session 特征、274 行使用其他 participant 上下文；严格前缀的未来来源计数为 0。因此 0.9020 必须命名为 **offline full-session transductive development estimate**，不得称为实时、单学生、inductive 或 deployment Accuracy。完整报告见 [`HIERARCHICAL_0_9_RESULTS.md`](../artifacts/dipser_credible/full_v5_52_complete_v5/HIERARCHICAL_0_9_RESULTS.md)，机器可读产物为 `hierarchical_0_9_report.json`。

## 8. 指标与统计解释边界

nested 5-fold 分析报告三分类 Macro-F1、Accuracy、Balanced Accuracy、逐类 Precision/Recall/F1、混淆矩阵、Log Loss、Brier/ECE、样本数和每类/每组分布。fusion 的对照是每个外层训练折内选出的 nested best-unimodal，不是观察外层结果后再挑 visual 或 sensor。

底层严格评估模块会计算按 session 的 cluster bootstrap 区间和整组置换值，并保留 `session_cluster_only_multimodal_gain_gate_passed` 作为可追溯诊断。但在当前 3 cohort×9 repeated activity 交叉设计中，计划的 27 个 session cells 实际只有 25 个含完整样本，且无论哪种计数都不是完全独立可交换的课堂抽样；因此这些 bootstrap 区间和 permutation `p` 只是 **session-cluster 诊断量**，不能解释为课堂总体 CI 或多模态增益假设检验。participant-only 分析的按 participant bootstrap/permutation 也因训练/测试共享 session 而不能用于建立结论。不能把每 15 秒时点当作独立统计单位来缩窄不确定性。

当前没有在查看结果前正式登记可接受 Accuracy/Macro-F1 阈值，因此 `accuracy_acceptability_established` 也必须为 `false`。本基准所谓“可信”指真实来源、同步、complete-case、防泄漏、特征顺序、源码/环境指纹和声称边界可审计，不指已经获得正向增益或可部署准确率。

## 9. 为什么仍没有部署准确率

DIPSER 是同一大学场地的回顾性数据。即使主分析同时做到 session 和 participant 无交叉，也没有测试独立目标站点、真实上线流程或前瞻漂移，因此 `deployment_accuracy_established` 必须保持 `false`。

建立目标学校部署准确率至少需要：

1. 在开发数据上完成模型、特征提取器、阈值和校准器选择，生成不可变冻结 artifact；
2. 冻结之后在与开发数据独立的目标学校/站点和明确目标人群中前瞻采集；
3. 目标 site、session、教师、学生和内容哈希与开发数据无交叉；
4. 由独立人工标注者在盲法下建立目标真值；
5. 至少 10 个目标 session，并按 session 聚类计算指标和 95% CI；
6. 只执行一次预注册主测试；若因域漂移而重训，必须使用新的未来测试集。

冻结模型在另一学校的历史数据上评估，最多称为“冻结外部站点评估”；只有上述前瞻流程才能考虑部署准确率。在此之前，不得把结果用于处罚、排名、录取、纪律处分、教师绩效考核或商业课堂监控。

## 10. 必须保存的审计材料

当前探索性配置在每次运行开始时记录，但这不会追溯性地变成正式预注册。必须保存：ScienceDB V5 archive 清单及 MD5、52 人/27 cell roster、15 秒间隔、1.0 秒 watch 文件名—内部中位时间容差、1.0 秒 watch 内部最大偏差、0.6 秒 watch—metadata 容差、complete-case 规则、特征白名单、四专家共识规则、nested 5-fold 配置、三套阻断设计、随机种子与诊断重复次数。

运行后保存：dataset/catalog/configuration/feature-bundle/environment fingerprints、源码 SHA-256、成员 CRC、archive 失败与时点排除、分层 coverage flow、实际有效 session/participant/cohort/activity 数、每折身份交叉审计、完整 OOF 预测、visual/sensor/fusion/nested-best-unimodal 指标、三套阻断结果和明确布尔结论。nested 报告中的 paired bootstrap CI 和 group permutation `p` 也要保留以便审计，但必须标为当前交叉设计下的非总体推断诊断量。

## 11. 运行真实数据基准

安装识别依赖后，以默认 52-archive 冻结清单运行：

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

命令会对 ScienceDB 发起真实 HTTP Range 请求，需要网络，且可通过 `archive_cache/` 断点续跑。核心产物为 `dataset_manifest.json`、`features.json`、`credible_report.json` 和 `blocked_descriptive_report.json`。命令成功只意味可用的描述性 OOF 产物已生成，不意味任何 `*_accuracy_established` 或 `*_multimodal_gain_established` 变为 `true`。

本协议本身不预填成绩。当前已完成并通过审计的 52/52 官方归档运行见 [`RESULTS.md`](../artifacts/dipser_credible/full_v5_52_complete_v5/RESULTS.md)；该实跑生成了描述性数值，但没有建立可信多模态增益、可接受准确率或部署准确率。

已有产物上的事后优化无需网络，可单独复跑：

```bash
python3 scripts/run_dipser_optimization.py \
  --artifact-dir artifacts/dipser_credible/full_v5_52_complete_v5

python3 scripts/run_dipser_hierarchical_challenge.py \
  --artifact-dir artifacts/dipser_credible/full_v5_52_complete_v5
```

成功运行表示当前单站点开发数据的严格因果 0.8 级结果和完整 session 离线固定划分 0.902 点估计可复现，不表示 `accuracy_0_9_established`、`cross_session_accuracy_established`、确认性 `multimodal_gain_established` 或 `deployment_accuracy_established` 为 true。
