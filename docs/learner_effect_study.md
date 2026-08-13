# 学习效果实验工具

这个工具把“真实学习效果”缺口变成可以执行、审计和恢复的研究流程，但不会生成参与者、分数、伦理批准或正向证据。v2 的空模板、演示数据、本地分析和 assessor receipt 的 `learner_effectiveness_established` 始终为 `false`，状态始终为 `external_validation_pending`；正式结论仍需真实外部受试者、可信治理、完整预注册报告和现有 external-evidence 签名链。

## 固定设计

- 采用 teacher 或 classroom 为随机化 cluster 的 1:1 A/B cluster RCT；同一 cluster 内所有参与者只能属于同一随机臂。
- 主要结局固定为 `post_score`，使用 `post ~ assigned_arm + pre_score` 的预注册 ANCOVA。不能在看过结果后切换为 gain score。
- 效应量为 ANCOVA 调整后的 A-B 系数除以全部随机参与者 `pre_score` 的样本标准差。
- 置信区间固定为按随机臂分层、以 teacher/classroom cluster 为单位有放回抽样的 percentile bootstrap。seed、replicate 数和最低有效 replicate 比例在预注册中冻结。
- ITT 始终使用 allocation CSV 中的原始分配，忽略换组或依从性；缺失 `post_score` 按预注册的保守 baseline-carried-forward 规则令 `post=pre`，同时完整报告 coverage、attrition 和缺失原因。
- transfer 固定为独立测验中的未见题迁移；retention 固定在教学后 14–28 天采集，两者均为预注册次要分析，不替代主要结局。缺失时使用与主要分析一致的保守 baseline-carried-forward sensitivity。
- subgroup 只能用站点保管的 `sg01` 等无语义代码。定义表只以 SHA-256 绑定，不能进入仓库；每个随机臂不足预注册最小样本的 cell 自动抑制，不输出人数或效果。可估计的 subgroup gap 只是公平性诊断，不允许因果 subgroup 宣称。
- teacher burden 按随机 cluster 收集 preparation、delivery、follow-up 分钟和 1–5 workload，报告只保留臂级汇总；缺行会让外部复核门槛失败。
- 安全结果固定记录不良事件 yes/no、严重度和相关性。`serious` 且 `possibly/probably` 会让自动安全门槛失败，但不会抹去或隐瞒事件。
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
- `outcome_collection_template.csv`：只有 token、pre/post/transfer/retention、retention day、无语义 subgroup code、固定缺失原因、grader blind、不良事件严重度/相关性和 protocol deviation。
- `teacher_burden_collection_template.csv`：只有 cluster token、三段耗时、workload 和固定缺失原因。

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

本地 `local_preregistered_positive_result_gate` 只有在以下条件同时满足时才可能为 true：真实数据声明；非占位伦理编号和同意/豁免；结果访问前冻结；分配隐藏；版本与 build hash 冻结；semantic lockbox report hash 预注册；独立 assessor 预注册；至少预设 cluster 和 30 名参与者；primary/transfer/14–28 天 retention coverage、attrition、subgroup/fairness、teacher burden、安全事件、blind grading 和 cluster bootstrap 全部达标；主要效应及 CI 下界都超过预设正门槛。它仍不等于学习效果已经建立。

## 独立 assessor 与 lockbox 绑定

研究方在 allocation 前冻结 intervention version/build SHA-256、独立 assessor organization/key ID、无语义 subgroup definition registry SHA-256，以及已经通过的 aggregate semantic lockbox report 的文件 SHA-256。行为 lockbox 只能证明教学代理在隐藏行为评测上达到阈值，**不是**学习效果证据。

独立 assessor 对精确 analysis、outcomes CSV、teacher-burden CSV、analysis code 和 semantic lockbox report 哈希做 Ed25519 签名。验证函数同时检查签名、预注册 assessor 身份、analysis exact hash、lockbox 文件 exact hash、lockbox `passed=true` 及 behavioral threshold。验证成功仍只返回：

```json
{
  "signature_verified": true,
  "learner_effectiveness_established": false,
  "external_validation_status": "external_validation_pending"
}
```

签名只能证明可信公钥对应私钥对精确内容签过名，不能仅凭签名证明 assessor 真正独立、伦理声明真实、统计结论正确或产生了因果学习效果。

## 外部证据交付

真实研究方还需要独立计算并报告预注册 p-value、primary/transfer/14–28 天 retention、attrition、全部 subgroup/fairness（含被抑制 cell 数）、全部安全事件和教师负担，形成 `real_learner_effectiveness` external evidence draft，然后沿用已有链路：

```bash
python3 -m teaching_skill_miner prepare-external-research-evidence ...
python3 -m teaching_skill_miner sign-external-research-evidence ...
python3 -m teaching_skill_miner verify-external-research-evidence \
  --expected-kind real_learner_effectiveness ...
```

签名只能证明精确内容和密钥控制；伦理文件真实性、研究独立性和可信公钥的外部治理仍需人工审计。
