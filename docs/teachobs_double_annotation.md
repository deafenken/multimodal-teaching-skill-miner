# TeachObs 双人独立标注与一致性复核

该流程用于补充一轮新的、与模型输出及官方 consensus gold 隔离的人工复核。它只读取固定版本的 `lessons.csv`、39 类 coding scheme 和 5158 个场景的时间元数据；不会读取逐场景 gold、模型预测、字幕正文、帧或视频字节。

TeachObs 0.1 发布的 39 个 `definition` 全部为空。标签名称和视觉/非视觉分组不是可执行的标注规则，因此默认生成的只是任务骨架，并明确记录 `operational_definitions_complete=false`、`annotation_execution_ready=false`。本工具不会代替领域专家编造定义。

正式执行前，应由独立领域专家提供 JSON operational codebook。它必须使用 schema `teaching_skill_miner.teachobs_operational_codebook.v1`，设置非空 `version`，完整覆盖同名 39 个 code；每项必须有非空 `definition`、至少一条 `inclusion_criteria`、`exclusion_criteria`、`positive_examples` 和 `negative_examples`。默认生成的 `operational_codebook_template.json` 只含空槽位，不能直接用于标注。

```json
{
  "schema": "teaching_skill_miner.teachobs_operational_codebook.v1",
  "version": "expert-panel-v1",
  "codes": [
    {
      "name": "官方 code 名称",
      "definition": "由领域专家填写",
      "inclusion_criteria": ["由领域专家填写"],
      "exclusion_criteria": ["由领域专家填写"],
      "positive_examples": ["由领域专家填写"],
      "negative_examples": ["由领域专家填写"]
    }
  ]
}
```

## 1. 生成 A/B 盲化任务

```bash
tsm prepare-teachobs-double-annotation \
  --repository artifacts/private/external_datasets/teachobs/repository \
  --media-root artifacts/private/external_datasets/teachobs/media \
  --operational-codebook /private/expert/teachobs_operational_codebook.json \
  --output artifacts/private/external_datasets/teachobs/human_annotation \
  --require-media
```

完整运行覆盖官方 30 讲、5158 个场景和 39 个二元标签。需要先完成私有视频准备；若只试运行部分课程，可重复传入 `--lesson-id S2`。生成内容包括：

- `assignment_A.csv` 与 `assignment_B.csv`：相同稳定 `item_token`，不同确定性随机顺序；
- `codebook.csv`：39 个官方 code 名称和视觉/非视觉分组；
- `operational_codebook.json`：经 39/39 完整性校验的专家规则；未提供时只生成不可执行的空模板；
- `assignment_manifest.json`：输入哈希、任务绑定和隐私/声明边界。

CSV 只引用 `videos/S*.mp4`，不复制媒体。时间段由 `start_seconds` 和 `end_seconds` 给出。每位标注者独立填写全部 `label::*` 列，值只能是整数 `0` 或 `1`；并在 CSV 中至少填写一次且保持一致：

- `rater_id`：不含姓名的稳定代号；
- `completed_by_real_human=YES`；
- `independent_without_gold_or_predictions=YES`；
- `signed_name`：真实签名，仅留在私有文件；
- `attested_at_utc`：带 UTC 时区的 ISO-8601 时间。

assignment CSV 和 manifest 同时绑定 operational codebook 的版本与规范化 SHA-256，A/B 无法混用不同规则。只有 39/39 规则完整且所有所选私有媒体引用存在时，`annotation_execution_ready=true`。在真实结果返回之前，公开收据始终记录 `human_completion=false`，所有 kappa 字段为 `null`。

## 2. 导入两份真实结果

```bash
tsm analyze-teachobs-double-annotation \
  --manifest artifacts/private/external_datasets/teachobs/human_annotation/assignment_manifest.json \
  --assignment-a /private/returned/assignment_A.csv \
  --assignment-b /private/returned/assignment_B.csv \
  --output artifacts/private/external_datasets/teachobs/human_agreement \
  --public-receipt artifacts/public/teachobs_human_annotation_receipt.json
```

导入会先拒绝空发布定义、缺失/变更的 operational codebook 或未就绪媒体绑定；随后拒绝相同 `rater_id`/签名、缺行、重复 item、被修改的时间/媒体/规则绑定、额外列、空标签、非 `0/1` 标签及缺失声明。通过后生成：

- 私有 `agreement_report.json`：每标签 Cohen's kappa、两位标注者的 prevalence、positive/negative agreement，以及 pooled binary kappa 和 macro label kappa；
- 私有 `disagreements.csv`：逐 item-code 分歧；
- 私有 `adjudication_template.csv`：全部分歧的第三方裁决空表，裁决者必须与 A/B 不同；
- 可公开的聚合收据：不含媒体引用、场景/课程 ID、人工逐项标签、code 名称、标注者代号或姓名。

`human_completion=true` 只表示 39/39 操作定义、两份完整二元矩阵及两份不同标注者的真实人工/独立自我声明通过了格式与绑定校验；工具不会把它表述为身份已由第三方核验。除非另行预注册并通过可靠性阈值，计算出 kappa 也不等于“标注可靠性已建立”。人工 kappa 更不是模型 Accuracy、Precision、Recall、F1、部署准确率或学习效果。
