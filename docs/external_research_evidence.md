# 外部研究证据接入协议

本接口只解决“如何把真实外部研究结果以可校验、可签名、可交付的方式接入项目”，不会生成研究数据、补写不存在的结果，也不会把自报数字变成真实证据。仓库不附带任何通过版 evidence manifest；正式字段以 [`schema/external_research_evidence.schema.json`](../schema/external_research_evidence.schema.json) 和 [`schema/external_research_evidence_attestation.schema.json`](../schema/external_research_evidence_attestation.schema.json) 为准。

## 支持的两类结论

- `confirmatory_multimodal_gain`：真实课堂、同一样本成对比较的外部锁箱消融。至少 10 个 claim clusters，coverage 至少 0.8，使用 cluster-aware inference；观测增益和置信区间下界都必须严格超过预注册的正增益门槛，且 `p <= alpha`。
- `real_learner_effectiveness`：真实学习者的 individual RCT 或以 classroom/teacher 为随机单位的 cluster RCT。至少 30 名参与者；个人随机设计至少 30 个随机单位，cluster RCT 至少 6 个随机 cluster。还要求分配隐藏、ITT、独立结局评估、伦理/同意、完整主要结局与不良事件报告，以及失访不超过预注册门槛且该门槛不得高于 0.2。调整后标准化效应和置信区间下界都必须严格超过预注册的正教育意义门槛，且 `p <= alpha`。

两类 manifest 都必须声明：结果访问前冻结的预注册与统计分析计划、外部治理和开发独立性、真实数据、开发身份隔离、eligible/evaluated/excluded 数量及排除原因、可重算 coverage，以及下列 SHA-256 绑定：

- 数据集或研究数据；
- 实际被评估的系统 artifact；
- comparator；
- 分析代码；
- 统计输出；
- 聚合结果表。

## 生命周期

1. 外部研究方按 schema 形成完整 draft。`evidence_fingerprint` 可以省略，由准备命令按 canonical JSON 计算；若已存在但过期，命令失败。
2. 准备命令重新计算 coverage、效应差、CI/显著性和设计 gate，并要求 manifest 中的 `*_established` 与重算结果完全一致。
3. 外部治理方使用 Ed25519 私钥签署精确 evidence；私钥不得交给开发团队或写入仓库。
4. 验证方使用事先通过独立渠道固定的公钥，验证签名、精确 manifest、evidence fingerprint 和本地 system artifact 的实算 SHA-256。
5. `verify-delivery` 分别接收多模态与学习效果三件套，并使用同一个 `--external-evaluated-system-artifact` 重新绑定实际交付物。

```bash
python3 -m teaching_skill_miner prepare-external-research-evidence \
  --input PRIVATE_STUDY/evidence_draft.json \
  --output PRIVATE_STUDY/evidence.json

python3 -m teaching_skill_miner sign-external-research-evidence \
  --evidence PRIVATE_STUDY/evidence.json \
  --private-key EXTERNAL_GOVERNANCE/ed25519_private.pem \
  --issuer "External research governance" \
  --key-id external-study-key-2026 \
  --output PRIVATE_STUDY/evidence_attestation.json

python3 -m teaching_skill_miner verify-external-research-evidence \
  --evidence PRIVATE_STUDY/evidence.json \
  --attestation PRIVATE_STUDY/evidence_attestation.json \
  --trusted-public-key TRUST_ANCHOR/external_study_public.pem \
  --expected-system-artifact dist/teaching_skill_miner-1.2.0-py3-none-any.whl \
  --expected-kind confirmatory_multimodal_gain \
  --require-claim-established \
  --output PRIVATE_STUDY/evidence_verification.json
```

学习效果证据把 `--expected-kind` 改为 `real_learner_effectiveness`。最终交付可以同时接入两类证据：

```bash
python3 -m teaching_skill_miner verify-delivery \
  --external-multimodal-evidence PRIVATE_MULTIMODAL/evidence.json \
  --external-multimodal-attestation PRIVATE_MULTIMODAL/attestation.json \
  --trusted-multimodal-public-key TRUST_ANCHOR/multimodal_public.pem \
  --external-learner-evidence PRIVATE_LEARNER/evidence.json \
  --external-learner-attestation PRIVATE_LEARNER/attestation.json \
  --trusted-learner-public-key TRUST_ANCHOR/learner_public.pem \
  --external-evaluated-system-artifact dist/teaching_skill_miner-1.2.0-py3-none-any.whl \
  --require-external-validation
```

`--require-external-validation` 还要求正式完整转写、真实两人独立人工复核和签名部署锁箱证据，不会因只提供上述两类 manifest 就把项目标为研究验证完整。

## 信任边界

- `prepare` 能验证 JSON 内部一致性和冻结 gate，不能证明所声明的数据、伦理审批、预注册页面或统计产物真实存在。
- 系统会实算并核对本地 evaluated system artifact 的 SHA-256；其余研究 artifact 由 manifest 指纹和外部签名绑定，正式审计仍需治理方保管并核验原件。
- 签名证明内容完整性和对应私钥控制，不自动证明签署者独立、没有利益冲突或遵守了研究流程。可信公钥必须在查看结果前从外部治理渠道取得。
- Evidence manifest 是聚合交付物，也仍需隐私、许可和披露风险审查；不得包含行级身份、原始媒体或敏感自由文本。
