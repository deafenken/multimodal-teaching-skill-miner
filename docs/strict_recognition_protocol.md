# 可信多模态、跨 session 与部署准确率协议

现有 OUC-CGE public sample 只能用于真实视频管线试运行。它没有可核验的
`session_id` / `teacher_id` / `site_id`，音频完整性还与标签相关，因此不能产出
本协议下的可信结论。

## 输入证据

严格清单遵循 `schema/strict_recognition_manifest.schema.json`。每条样本必须记录：

- 同步样本内容哈希 `content_sha256`，用于去重和训练/测试交叉检查；
- 数据发布方提供且可追溯、并与任务相符的身份字段。教师课堂任务通常要求
  `session_id`、`teacher_id`、`site_id`；学生任务可要求 `session_id`、
  `participant_id`、`cohort_id`、`activity_id`、`site_id`，不能为凑 schema 虚构教师 ID；
- 独立人工真值，以及每个同步模态的完整性；
- 根据清单内容重新计算的 `dataset_fingerprint`。

`validate_strict_manifest` 不会从文件名猜测身份。任何验证标志、证据来源、必要
模态或真实 ID 缺失都会抛出 `CredibilityError`，不会降级成普通随机切分。
冻结类名必须是唯一、非空字符串，并与每条 `records.label_name` 按整数标签逐行一致；
不能用重复类名、重排名称或只改显示文本绕过逐类 Recall/支持 gate。

预计算特征还必须为每个模态提交提取器 ID 和 SHA-256 指纹，并声明提取器在评估
前已经冻结、未使用真值标签、未在本次评估样本上拟合。当前严格模块不接受在全量
评估集上微调后再预计算的特征；需要微调时，应把微调过程放入每个训练折内部。调用者必须
提交有序的 `sample_ids`，严格验证它与 manifest 顺序逐行一致。`strict_feature_bundle_v1`
把 dataset fingerprint、样本精确顺序、模态与特征名顺序、提取器 provenance 以及精确
float64 矩阵共同纳入 fingerprint；布尔值、非有限值、超大整数或任何重复声明与实际内容不符
都会失败关闭。旧 `dipser_feature_bundle_v1` 只对被严格识别为 DIPSER visual/sensor artifact
兼容，进入 checkpoint 后仍升级为通用严格绑定。

## 多模态增益和跨 session

`nested_grouped_multimodal_evaluation` 使用两层分组评估。下列以跨 session 为例：

1. 外层只按发布方提供的 session 切分，生成最终 OOF 预测；
2. 每个外层训练折内部再次按 session 切分，选择正则化参数；
3. “最佳单模态”也只由该外层训练折的内层结果选择，不能查看外层标签；
4. 融合与这条 nested best-unimodal 基线按真实 group 做配对 bootstrap 和整组交换置换检验；
5. Accuracy 与 Macro-F1 的绝对值也按独立统计组做 cluster bootstrap 95% CI，
   不能把同一课堂中的时间窗当作独立样本缩窄区间。

只有统计组对目标总体确实可视为独立可交换抽样、至少有 10 个有效组、macro-F1 增量的
95% CI 下界大于 0，且单侧置换检验 `p < 0.05` 时，`multimodal_gain_established` 才可为
`true`。“分组键不交叉”本身不足以证明可交换性：还必须审计上层的 participant/teacher/
cohort/activity/site 交叉或嵌套。按 session 隔离不等于按教师、参与者、活动或站点隔离。
用于最终准确率声称的聚类单位还必须显式冻结为 `claim_cluster_field`；不能在看到哪个聚类
区间更有利后再从 session、participant、teacher、cohort 或 site 中选择。

## DIPSER V5 的专用保守规则

DIPSER 当前 52-archive roster 是**部分标签清单审查后冻结的探索性分析**，不是正式预注册。
代码中的历史名称 `DEFAULT_PREREGISTERED_ASSIGNMENT` 不得被当作预注册证据。清单计划的
27 个 `session_id` 正好是单一站点的 3 个 cohort × 9 种 repeated activity recording cells；
complete-case 数据实际只有 25 个 cell 含样本。无论按 27 个计划 cell 还是 25 个有效 cell
计数，它们都不是从课堂总体独立抽样的可交换 session。

标签解析必须要求每个 archive 内恰好四个 publisher expert `labeler_*.json` 文件。labeler ID 可为
`01`–`05`；V5 实际有 `01/02/03/04` 和 `01/02/03/05` 两种组合，其中 22 个 archive 为后一种。
`self_labeling` 必须从专家真值中排除。由“固定文件尾号”改为“归档内实际四专家集合”只是发布结构的
解析修正，不放宽四位专家在该时点全部有效、至少 3/4 同档的真值门槛。

远程 ZIP 读取同样必须失败安全：所需 non-image 尾部不超过 128 MiB 时可做有界合并 Range 读取；
异常布局超过该上限时，只逐个 Range 读取必需的 label/metadata/watch 成员，不得回退为整 archive 下载。

合格同步时点严格要求：

- watch 文件名时间与内部传感器行中位时间差不超过 1.0 秒，内部所有可用时间相对中位数的最大偏差不超过 1.0 秒，watch 中位时间与最近 metadata 文件名时间差不超过 0.6 秒；
- head pose 的 pitch/yaw/roll 均为有限数，body pose 恰有 33 个 landmarks，每个的 `x/y/z/visibility/presence` 全部为有限数；
- 心率、线性加速度、陀螺仪、旋转向量和环境光五类 watch 数据都有非空行，并且每行的预期数值都完整且有限；
- availability、presence、sample count 和 landmark count 只用于 complete-case 审计，不进入模型。

按实际 25 个有完整样本的 recording/session 分组的 nested 5-fold 只生成“单站点、同设计内完整案例的描述性
OOF”。训练折仍可能包含测试 recording 的 cohort 或 activity；按 session 的 bootstrap CI 和
permutation `p` 仅作诊断，不得解释为课堂总体推断。`session_blocked_oof_estimate_available`
仅表示 OOF 产物可用，不表示 Accuracy 可接受或多模态增益成立。

按 participant 分组的次分析会计算并披露 train/test session overlap；对当前 DIPSER roster，
该 overlap 使 `participant_disjoint_accuracy_established` 和
`participant_disjoint_multimodal_gain_established` 一律为 `false`。

为降低 cohort/activity 共享带来的乐观性，同时运行三套固定 `LogisticRegression(C=1.0)`
描述性评估：3 折 leave-one-cohort-out、9 折 leave-one-activity-out，以及 27 个
cohort×activity 双重阻断 cell（训练同时排除测试 cohort 和 activity）。标准化只在当前
训练折内拟合，不调参。这三套报告不生成 CI 或 `p`，所有增益/跨站点/部署
`established` 字段均为 `false`。

查看原 OOF 后新增、并在**事后开发协议中固定**的 linear-SVC、因果平滑和离线序列 pooling 必须保存在独立的
`optimization_report.json`，统一标记 `post_selection_exploratory=true`，不能回填或覆盖原
`credible_report.json`。当前因果 session SGKF5 与逐 session 留一的 Accuracy 均为 0.8176，
但 LOCO、LOAO、双重阻断仅为 0.6791、0.7230、0.6520，且按 session 的事后 Accuracy 增益
bootstrap 区间包含 0。因此“达到 0.8”只能作为同站点开发期点估计陈述，所有
`accuracy_established`、`multimodal_gain_established` 和 `deployment_accuracy_established`
继续为 `false`。

进一步查看同一数据后形成的完整 sequence/session 层级候选必须单独写入
`hierarchical_0_9_report.json`。它在 SGKF5 seed=2026 上为 0.9020，但 LOSO 为 0.8986，
50 个 outer seed 均值为 0.8883，且固定 gate 的 LOCO/双重阻断仅为 0.6554/0.6351。
推理时还会读取 held-out session 中未来窗口和其他 participant 的无标签特征；排除当前行后
SGKF5 降为 0.8547，严格 timestamp-prefix 为 0.7770。因此它只能命名为
`offline_full_session_transductive` 点估计，`accuracy_0_9_established` 必须为 `false`，不得
并入实时、单 participant 或 frozen deployment 协议。

DIPSER 的 archive cache 键绑定发布方文件身份、同步/提取配置与处理源码实现指纹，
缓存内容另有 SHA-256。优化与 0.9 runner 会从 records、特征名和精确 float64 矩阵重新计算
dataset / feature-bundle fingerprint，而不是信任两个 JSON 中重复声明的值。最终 artifact 还记录 feature-bundle fingerprint、评估源文件 SHA-256、
Python/平台/zlib/NumPy/SciPy/scikit-learn 版本及环境指纹。这些保证的是可追溯性，不是外推性。

## 冻结外部部署评估

`fit_frozen_deployment_model` 仅在源数据上选择参数、拟合标准化器和分类器，并
固定特征名与顺序、提取器指纹、严格训练 feature-bundle fingerprint、模型参数、训练数据指纹、
任务相关身份契约、每个身份字段的训练集合、唯一非空类名、`ClaimContract` 以及
`claim_cluster_field`。
学生任务可以冻结 `session_id + participant_id + cohort_id + activity_id + site_id`，不允许为凑格式虚构
`teacher_id`；若要声称跨教师泛化，则必须把真实 `teacher_id` 纳入契约。冻结模型可通过
`frozen_model_to_artifact` 保存为 JSON，并由 `load_frozen_deployment_model` 校验完整性。
checkpoint v3 把训练 strict feature bundle、ClaimContract fingerprint 和 claim cluster 一并写入
模型 fingerprint，运行者不能在看完外部标签后更换特征、聚类单位或降低门槛。
`evaluate_frozen_external_deployment` 只允许在以下独立目标站点外部集上运行：

- 明确声明未参与模型开发，且目标部署人群、场景和准确率可接受阈值在查看结果前有文档；
- 数据集、内容以及 identity contract 内的全部身份字段均与训练集无交叉；
- 特征名和顺序与冻结模型完全一致；
- 置信区间以 checkpoint 冻结的 `claim_cluster_field` 为聚类单位；若同一 participant 或 teacher
  横跨多个 claim cluster，`independent_cluster_structure` gate 失败；
- manifest 披露 complete-case `evaluation_coverage_fraction`、eligible/evaluated/excluded 数量、
  exclusion reason 汇总、eligible population fingerprint、分母来源与 coverage evidence fingerprint；
- 明确是否前瞻采集，以及是否为一次性 lockbox 主测试。

ClaimContract 至少冻结主指标、Accuracy、Macro-F1、二者 cluster-CI 下界、最小逐类 Recall、
最小逐类支持、最小 claim-cluster 数、最小 coverage、是否必须前瞻以及是否必须一次性 lockbox。
所有 gate 同时通过才允许 `external_site_frozen_accuracy_established=true`；部署结论还必须实际满足前瞻、
签名登记和一次性测试。模型即使很差，也不能仅凭组数和布尔标志建立准确率。基于 DIPSER 单站点回顾数据的
所有当前随附报告中，`deployment_accuracy_established` 始终为 `false`。

### 外部登记、一次性消费与签名 receipt

外部保管方应在揭示结果前创建登记请求，把 checkpoint、ClaimContract、训练/外部 dataset、训练/外部
strict feature bundle 以及 coverage evidence 的精确 fingerprint 绑定在一起，再使用 Ed25519 私钥签名。
评估必须同时提供登记 attestation、预先固定的可信公钥和一次性 ledger；登记被原子消费后，重复运行同一
registration 必须失败。完成报告再由同一治理密钥签署 receipt，`verify-delivery` 会核对报告 fingerprint、
登记 ID、登记 attestation fingerprint、公钥 fingerprint 和一次性消费回执。

签名只证明“持有该私钥者签署了这些字节且内容未被修改”，**不证明该持钥者在组织上独立、没有接触开发
标签或遵守了前瞻流程**。可信公钥必须由开发团队之外的治理方在查看结果前通过独立渠道分发并固定；把开发者
自己生成的公钥传给验证器，只能形成自签完整性记录，不能升级为独立锁箱证据。

CLI：

```bash
tsm freeze-recognition-model --help
tsm create-freeze-registration --help
tsm sign-freeze-registration --help
tsm evaluate-frozen-recognition --help
tsm sign-evaluation-report --help
tsm verify-delivery --help
```

这套扁平冻结模型用于外部协议闭环，不是 0.902 离线层级模型的部署 checkpoint。原始目标站点媒体/传感器
可先通过 `extract-strict-features` 生成严格 bundle；该入口绑定实际原始 SHA-256、窗口、提取器代码/配置、
空权重声明、Python/NumPy/平台和 FFmpeg/FFprobe 二进制与版本，并可在不预测的情况下验证 checkpoint v3
schema/provenance 兼容性。完整协议见 [`raw_feature_bridge.md`](raw_feature_bridge.md)。人工编辑特征 JSON 不能
替代这条可审计生成链。
