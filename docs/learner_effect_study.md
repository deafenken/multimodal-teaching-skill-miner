# 学习效果实验工具

这个工具把“真实学习效果”缺口变成可以执行、审计和恢复的研究流程，但不会生成参与者、分数、伦理批准或正向证据。空模板、演示数据和本地分析的 `learner_effectiveness_established` 始终为 `false`；正式结论仍需现有的外部 evidence manifest、独立 Ed25519 签名和可信公钥验证链。

## 固定设计

- 采用 teacher 或 classroom 为随机化 cluster 的 1:1 A/B cluster RCT；同一 cluster 内所有参与者只能属于同一随机臂。
- 主要结局固定为 `post_score`，使用 `post ~ assigned_arm + pre_score` 的预注册 ANCOVA。不能在看过结果后切换为 gain score。
- 效应量为 ANCOVA 调整后的 A-B 系数除以全部随机参与者 `pre_score` 的样本标准差。
- 置信区间固定为按随机臂分层、以 teacher/classroom cluster 为单位有放回抽样的 percentile bootstrap。seed、replicate 数和最低有效 replicate 比例在预注册中冻结。
- ITT 始终使用 allocation CSV 中的原始分配，忽略换组或依从性；缺失 `post_score` 按预注册的保守 baseline-carried-forward 规则令 `post=pre`，同时完整报告 coverage、attrition 和缺失原因。
- retention 是预设次要描述性结局，不替代主要结局。
- `grader_blind` 必须逐行明确填写，所有随机 token 必须在 outcome 表中恰好出现一次。

该最小工具不计算 confirmatory p-value，也不允许把内部 Skill 自动评分、识别 Accuracy 或 rubric 分数放进 outcome 表。固定 CSV header 拒绝额外 `skill_score` 等列；学习结局必须来自独立的学习者测验。

## 生成私有模板

```bash
python3 -m teaching_skill_miner prepare-learner-effect-study \
  --output-dir PRIVATE_STUDY \
  --study-id classroom-rct-2026 \
  --cluster-unit classroom \
  --cluster-count 6 \
  --participants-per-cluster 10
```

输出目录权限为 `0700`，三个文件为 `0600`：

- `preregistration.json`：固定主要结局、缺失规则、cluster bootstrap、coverage/attrition 和最小教育意义门槛。
- `participant_allocation.csv`：只有 SHA-256 participant/cluster token 和 A/B 分配，不含姓名、学号、邮箱、教师姓名或课堂名称。真实身份到 token 的映射只能由获授权的站点保管者在项目外保存。
- `outcome_collection_template.csv`：只有 token、pre/post/retention、固定缺失原因、grader blind、不良事件和 protocol deviation。

正式研究应在分配和查看结局前，由研究治理方提供真实伦理批准编号并声明知情同意或获批豁免、预注册冻结和分配隐藏。例如这些参数只是把声明绑定进模板，并不证明声明真实：

```bash
python3 -m teaching_skill_miner prepare-learner-effect-study \
  --output-dir PRIVATE_REAL_STUDY \
  --study-id classroom-rct-2026 \
  --data-origin real \
  --ethics-approval-id IRB-REAL-ID \
  --informed-consent-or-approved-waiver \
  --preregistration-frozen-before-allocation \
  --allocation-concealment-procedure-declared
```

不要把身份映射、同意书、逐人敏感备注或自由文本不良事件详情放入仓库。

## 填表与分析

站点保管者复制 outcome template，填入真实测验结果。观测到的 post/retention 对应 missing reason 必须是 `not_missing`；缺失值必须从 `withdrawn`、`lost_to_followup`、`illness`、`technical_failure`、`other` 中选择原因。`grader_blind` 只能是 `true` 或 `false`，`adverse_event_reported` 只能是 `yes` 或 `no`。`protocol_deviation` 只能使用 `none`、`nonadherence`、`crossover`、`absence`、`other`，不接受可能泄露身份的自由文本。

```bash
python3 -m teaching_skill_miner analyze-learner-effect-study \
  --preregistration PRIVATE_REAL_STUDY/preregistration.json \
  --allocation PRIVATE_REAL_STUDY/participant_allocation.csv \
  --outcomes PRIVATE_REAL_STUDY/completed_outcomes.csv \
  --output PRIVATE_REAL_STUDY/analysis.json
```

分析会拒绝：preregistration/allocation 哈希不一致、未知或重复 token、cluster 错位、跨臂 cluster、参与者缺行、越界/非有限分数、缺失原因不一致、grader 状态未填写、额外 CSV 列、少于预注册 bootstrap 有效率等情况。

本地 `local_preregistered_positive_result_gate` 只有在以下条件同时满足时才可能为 true：真实数据声明；非占位伦理编号和同意/豁免；结果访问前冻结；分配隐藏；至少预设 cluster 和 30 名参与者；coverage/attrition 达标；所有 grader blind；cluster bootstrap 有效；效应及 CI 下界都超过预设正门槛。它仍不等于学习效果已经建立。

## 外部证据交付

真实研究方还需要独立计算并报告预注册 p-value、全部注册结局和伤害，形成 `real_learner_effectiveness` external evidence draft，然后沿用已有链路：

```bash
python3 -m teaching_skill_miner prepare-external-research-evidence ...
python3 -m teaching_skill_miner sign-external-research-evidence ...
python3 -m teaching_skill_miner verify-external-research-evidence \
  --expected-kind real_learner_effectiveness ...
```

签名只能证明精确内容和密钥控制；伦理文件真实性、研究独立性和可信公钥的外部治理仍需人工审计。
