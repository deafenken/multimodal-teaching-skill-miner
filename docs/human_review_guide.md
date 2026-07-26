# 人工复核说明

自动分只能验证“格式正确、证据可追溯、步骤可执行”，不能证明真实学习增益。正式验证至少邀请 2 名具有教学经验的复核者，彼此独立填写 `artifacts/human_review.csv`。当前工程只强制“独立复核者身份与覆盖”，不会凭 CSV 自动证明复核者对实验条件、模型版本或研究假设实现了盲法；正式实验应另行记录并尽可能实施盲法。

空白模板不是验证结果。汇总时必须传入完整 Skill 目录，系统才会强制检查 10/10 覆盖、每个 Skill 两名不同复核者、额外 Skill 和重复 `(skill_id, reviewer_id)`：

```bash
python3 -m teaching_skill_miner human-evaluate \
  --input artifacts/human_review.csv \
  --skills artifacts/skills \
  --output artifacts/human_review_report.json
```

`validation_status` 只有在覆盖完整且 CSV 合法时才会是 `complete`；是否最终通过还要继续满足评分和 κ 门槛。

每行 `skill_fingerprint` 是该 Skill 完整 canonical JSON 的 SHA-256，而不是只对 `skill_id` 或文件名取哈希。传入 `--skills` 后，系统会从当前 Skill 重新计算并逐行比较；同一 ID 的目标、证据、步骤、fallback 或任何字段变化都会使旧评分失效。这防止“先评旧版本、后替换内容”仍沿用人工验证。

旧模板若没有 `skill_fingerprint`，应在新的输出目录运行 `demo` 生成当前模板，或把当前模板中的精确 fingerprint 带入确实针对同一 canonical Skill 完成的研究记录。无法证明版本一致时应重新复核，不应事后给旧评分补一个新版本哈希。`demo` 为保护人工工作不会覆盖已经存在的 CSV。

## 评分维度

每项采用 1–5 分：1=明显不合格，3=基本可用但需修改，5=可直接用于新教学任务。

1. `goal_clarity`：目标是否具体、可观察、可测试。
2. `evidence_fidelity`：Skill 是否确实由证据片段支持，而非抽取器臆造。
3. `procedure_executability`：另一教学 Agent 能否无额外猜测地执行步骤。
4. `adaptivity`：是否能根据学生信号走不同分支。
5. `transferability`：替换知识主题后，方法是否仍然成立。
6. `harm_or_bias_flag`：出现错误知识、歧视、隐私或不恰当教学策略时填 1，否则填 0。

## 判定与一致性

- 每个维度取复核者均值；所有维度均值至少 3.5，且伤害标记为 0，才人工通过。
- 当前实现计算五档评分的二次加权 Cohen's kappa；有两名以上复核者时，报告所有可计算复核者对的 κ，并以其均值执行 `>= 0.67` 门槛。项目没有实现 Krippendorff's alpha，文档和论文不得声称已计算该指标。
- 一致性低于 0.67 时，应先修订评分锚点，再由复核者重新独立评分；若正式实验采用盲法，还需在外部研究记录中说明盲法对象、揭盲时点和破盲事件。
- 随机抽取至少 20% 的视频片段做二次听写核对；正式论文实验应使用完整转写而不是仓库中的释义节选。
- 另外设置未参与 Skill 抽取的新主题，做近迁移与远迁移各一题，比较“使用 Skill”和“不使用 Skill”的教学 Agent。

## 推荐的 A/B 指标

- 学习者前测—后测增益；
- 迁移题正确率；
- 首次提示前独立完成率；
- 达标所需提示次数与教学轮数；
- 学习者对解释清晰度的 5 分量表；
- 教师对事实错误与认知负荷过高的标记率。
