# TeachObs 四臂确认性外部锁箱

## 结论先行

公开 TeachObs 官方 23/7 划分只能用于开发和探索性复现，不能重新命名为外部锁箱。原因不只是测试集规模：这些测试标签在冻结本项目四臂系统前已经可访问，训练/测试的 `source` 元数据还有 3 个取值重叠，而且发布表没有 canonical site、teacher 或 classroom ID。`source` 表示异质的上游发布者、项目或采集来源，不能冒充 site 分组。确认性多模态增益必须在身份与内容均不属于开发数据的新学校/新课堂上，先冻结系统和统计方案，再前瞻采集并只评估一次。

项目现在提供一层专门的、可指纹化且 fail-closed 的预注册协议：

- [`teaching_skill_miner/teachobs_lockbox.py`](../teaching_skill_miner/teachobs_lockbox.py)；
- [`schema/teachobs_lockbox_preregistration.schema.json`](../schema/teachobs_lockbox_preregistration.schema.json)；
- [`scripts/prepare_teachobs_lockbox_preregistration.py`](../scripts/prepare_teachobs_lockbox_preregistration.py)；
- 定向篡改、文件重哈希和 Ed25519 测试位于 [`tests/test_teachobs_lockbox.py`](../tests/test_teachobs_lockbox.py)。

预注册本身不含任何目标站点标签、预测或指标。即使四个模型文件都已绑定，即使开发者用本地测试密钥签了名，`confirmatory_multimodal_gain_established`、`deployment_accuracy_established` 和 `learner_effectiveness_established` 仍全部为 `false`。签名能证明内容没有被改动，不能自动证明签署者独立。

## 为什么现有通用链不够

项目原有三部分各自有效，但不能单独证明四臂增益：

| 链路 | 已能证明 | 四臂确认性增益仍缺什么 |
|---|---|---|
| TeachObs 四臂 benchmark | 固定四臂、同一批场景、lesson-cluster 2,000 次配对 bootstrap；可导出无 pickle 的确定性 JSON+NPZ 四臂冻结包并重验预测一致性 | 标签在实现前已公开，且测试 lesson cluster 很少；即使本地冻结包 checker 通过，也仍是开发/探索产物，不是外部锁箱 |
| frozen deployment checkpoint | 单个冻结分类器、目标数据/特征哈希、coverage、前瞻声明、一次性本地 ledger | 不是 39 标签四臂比较，也不绑定同一样本的四组预测与唯一主 contrast |
| external research evidence | 独立治理方签最终 aggregate evidence，可重算增益门 | 原协议只绑定 system/comparator 两个摘要，自报 paired inference，未冻结四臂定义和 bootstrap 的 2,000 次/seed/cluster unit |

新预注册层把这些缺口在查看目标结果前锁定；后续仍沿用原有冻结数据登记、一次性消费和最终 external-evidence 签名链。

## 固定的确认性问题

唯一主要问题是：在完全相同的合格目标站点场景上，`full` 的 39 标签 Macro-F1 是否高于 `transcript_only`，且点增益和 95% 配对 cluster-bootstrap CI 下界都超过预先固定的 `0.02`。

固定四臂为：

1. `transcript_only`：仅训练集拟合的 transcript character TF-IDF；
2. `transcript_audio`：transcript + 同场景音频数值；
3. `transcript_visual`：transcript + OCR TF-IDF + 图像/场景/板书变化数值 + hash-bound CLIP；
4. `full`：transcript + audio + 全部 visual block。

四臂必须使用同一个 eligible/evaluated sample set、相同顺序和相同 cluster 映射。任何一臂缺少某场景，都不能在其余三臂上另算一个看似更好的 complete-case 主结果。

主要指标固定为 39 个逐标签 F1 的不加权均值。每一臂使用该冻结模型在训练讲次 OOF 阶段已经选择并写入 `arrays.npz` 的 39 个逐标签阈值，目标站点不得重新选择阈值。预注册不公开阈值数值，而是逐臂绑定经安全加载验证的 `arrays.model_thresholds` SHA-256；因此分析不会再错误地退回统一 `0.5`。主要 contrast 只有 `full - transcript_only`；另外四个 contrast 是 secondary，不能挑选其中最有利的一个冒充主要结果。

不在 scene 层独立重采样。固定以 `classroom_id` 或 `session_id` 表示一节完整课堂/lesson，整簇共同重采样，固定 seed `20260723`、2,000 次、percentile 95% CI。目标站点至少需要 10 个独立 cluster，coverage 至少 `0.8`。

显著性规则也不能事后选择：复用同一组 2,000 个配对 cluster replicate，针对“增益不超过注册最小值 0.02”的单侧尾部，用 `(1 + count(delta_b <= 0.02)) / (2000 + 1)` 计算带 plus-one 修正的 `p` 值；同时仍要求 95% CI 下界超过 0.02。

## 先生成诚实的 pending draft

尚未完成私有四臂导出时，可以先生成待冻结草案。可以把当前实现和分析代码先绑定；缺失的四臂权重及其传递绑定会明确显示为 `bound=false`，签名函数会拒绝签署：

```bash
python3 -m teaching_skill_miner prepare-teachobs-lockbox-preregistration \
  --study-id teachobs-new-site-confirmatory-2026-draft \
  --system-artifact teaching_skill_miner/teachobs_multimodal_benchmark.py \
  --analysis-code teaching_skill_miner/teachobs_lockbox.py \
  --output artifacts/public/teachobs_new_site_lockbox_preregistration_draft.json
```

输出不包含本地路径，只含 logical name、字节数和 SHA-256。草案的固定状态为：

```text
frozen_artifact_set_complete=false
preregistration_execution_ready=false
external_registration_signature_verified=false
target_execution_evidence_complete=false
confirmatory_multimodal_gain_established=false
deployment_accuracy_established=false
learner_effectiveness_established=false
```

## 四臂模型落盘后重新生成

先用 `benchmark-teachobs-multimodal --frozen-model-output PRIVATE_LOCKBOX/frozen_models`（或 `scripts/export_teachobs_frozen_models.py`）导出确定性私有目录。系统 artifact 必须是该目录的 `bundle_manifest.json`，四个 arm artifact 必须是同一目录下各臂的 `manifest.json`；每个 manifest 的 companion `arrays.npz` 必须仍在原位。不能拿预测结果 JSON、普通文本、复制自另一 bundle 的 manifest 或单独一个 manifest 冒充完整模型：

```bash
python3 -m teaching_skill_miner prepare-teachobs-lockbox-preregistration \
  --study-id teachobs-new-site-confirmatory-2026-001 \
  --system-artifact PRIVATE_LOCKBOX/frozen_models/bundle_manifest.json \
  --analysis-code teaching_skill_miner/teachobs_lockbox.py \
  --arm-model transcript_only=PRIVATE_LOCKBOX/frozen_models/transcript_only/manifest.json \
  --arm-model transcript_audio=PRIVATE_LOCKBOX/frozen_models/transcript_audio/manifest.json \
  --arm-model transcript_visual=PRIVATE_LOCKBOX/frozen_models/transcript_visual/manifest.json \
  --arm-model full=PRIVATE_LOCKBOX/frozen_models/full/manifest.json \
  --target-cluster-field classroom_id \
  --output PRIVATE_LOCKBOX/teachobs_new_site_preregistration.json
```

预注册 schema 1.2 不再把“六个文件各有 SHA”误当成完整冻结系统。生成器会安全加载 bundle，逐臂核对标签顺序、训练/软件 provenance、TF-IDF/scaler/LR 状态和固定阈值，并以 `allow_pickle=False` 解析全部 NPZ；ZIP 路径逃逸、重复或额外 member、对象 dtype、异常 shape、过大解压体积都会失败关闭。写入预注册的 path-free `transitive_model_binding` v2 同时承诺 bundle fingerprint、四个 manifest 文件及内容 fingerprint、四个 `arrays.npz` SHA-256、每臂 shape 为 `[39]` 的 `model_thresholds` 数组 SHA-256、标签顺序以及训练/软件 provenance。新的 v2 `artifact_set_fingerprint` 覆盖这些阈值摘要；`analysis_plan.decision_threshold_policy` 再绑定同一组四臂阈值摘要和同一个 artifact-set fingerprint，防止冻结模型与统计方案各用一套阈值。

`verify_teachobs_lockbox_artifact_files` 会在执行前重新读取 bundle、四个 manifest 和四个 companion 数组，而不只是重哈希命令行传入的 JSON，并重算阈值感知的 v2 artifact-set fingerprint。删除或替换任一 `arrays.npz`、跨 bundle 混用 arm、修改阈值摘要、让分析计划指向另一 artifact set、修改任一字节或运行时版本漂移都会失败。占位的全零摘要同样被拒绝。缺少任一冻结 artifact 的 pending draft 仍可生成，但阈值摘要与 artifact-set fingerprint 必须保持 `null`、`binding_complete=false`，不能签署或执行。

外部站点推理时应给 `predict_teachobs_frozen_bundle` 传入稳定且唯一的 `sample_ids`，以及冻结的 audio schema、visual-evidence schema、视觉提取配置 SHA 和 CLIP revision/权重 SHA。目标数据只要换了特征顺序、视觉配置或权重便会失败，而不是静默套用 scaler。返回的 `prediction_input_fingerprint` 绑定样本顺序、transcript/OCR 顺序、三个数值矩阵、标签及特征顺序和全部上述 provenance；一次性执行器应预先登记并通过 `expected_prediction_input_fingerprint` 复核它。未传 `sample_ids` 时会明确返回 `sample_identity_bound=false`，不得作为锁箱共同样本顺序证据。

随后由外部治理方控制的 Ed25519 私钥调用 `sign_teachobs_lockbox_preregistration`。项目方只拿到通过独立渠道预先固定的公钥，并用 `verify_teachobs_lockbox_preregistration_attestation` 验证。函数强制签署以下声明：新站点与开发数据不重叠、前瞻采集、登记时目标 outcome 尚不可见、目标标签未用于改模型/特征/阈值、公开 TeachObs 不是锁箱、主评估最多一次。

## 数据冻结、一次性执行和最终证据

外部预注册签名只是第一阶段，不能据此写“增益成立”。目标站点采集完成后、揭盲前还要：

1. 固定 eligible denominator、排除规则、目标 dataset/feature bundle、样本顺序和 cluster assignment 的 SHA-256；
2. 使用现有 `create-freeze-registration` / `sign-freeze-registration` 把冻结系统和这批确切目标数据绑定；
3. 在外部保管的一次性 ledger 中先 reserve，再原子运行全部四臂，不能逐臂或逐参数试跑；
4. 保存相同样本顺序的四臂预测表、coverage reconciliation、2,000 次配对 cluster-bootstrap 输出和 aggregate result table 的哈希；
5. 形成 `confirmatory_multimodal_gain` external evidence，其中 `protocol_sha256` 必须等于本预注册 fingerprint，baseline modalities 为 transcript，added modalities 为 audio + visual，primary metric 为 Macro-F1；
6. 由同一预先固定的外部信任链签最终 evidence，并公开所有注册的主要分析，包括负结果。

`validate_teachobs_confirmatory_evidence_handoff` 会把最终通用 external-evidence manifest 重新绑定到本预注册：核对 protocol/SAP digest、四臂 system、transcript-only comparator、analysis code、主模态对比、Macro-F1、cluster field、最低 cluster 和 coverage。未验最终外部签名前，它即使发现所有未签名数值 gate 都通过，也仍返回 `confirmatory_multimodal_gain_established=false`。

最终 gate 必须同时满足：冻结四臂重新验哈希、共同样本/cluster 映射一致、新站点且无开发重叠、前瞻登记时间顺序有效、恰好一次消费、至少 10 个独立 cluster、coverage 达标、2,000 次配对 cluster-bootstrap 完整、最终外部签名有效、主要增益和 CI 下界超过注册门槛，并通过注册的显著性规则。任何一项失败都应保留为真实的阴性/不确定结果，不能回到 TeachObs 公共测试集调参后再次消耗同一个锁箱。

这条协议只建立“识别层面的确认性多模态增益”。部署 Accuracy 还要满足部署目标和逐类/coverage checkpoint；真实学习效果仍必须通过单独的伦理审批和集群随机对照实验，不能由识别 F1 推导。
