# 原始课堂数据到严格特征包

`extract-strict-features` 是冻结识别协议的原始数据入口。它直接读取目标站点保管的
视频、音频流和 CSV/JSON/JSONL 数值传感器文件，执行固定的无标签特征提取，并生成
可供 `freeze-recognition-model`、`create-freeze-registration` 和
`evaluate-frozen-recognition` 使用的 `strict_feature_bundle_v1`。

该命令只提取特征，不加载分类器、不输出预测，也不建立识别准确率、部署准确率或教学
效果。成功退出只表示原始字节、选定窗口、样本顺序、特征矩阵与提取器 provenance 的
机械绑定通过。

## 原始输入清单

严格 manifest 仍使用 `schema_version: "2.0"`。每条 `records` 另外提供私有
`raw_inputs`，路径必须相对 `--raw-root`，不能逃出该目录：

```json
{
  "sample_id": "site-b-session-07-window-003",
  "content_sha256": "由 strict_synchronized_raw_content_v1 计算的 64 位值",
  "label": 1,
  "label_name": "medium",
  "session_id": "site-b-session-07",
  "teacher_id": "site-b-teacher-02",
  "site_id": "site-b",
  "modalities": {"visual": true, "audio": true, "sensor": true},
  "raw_inputs": {
    "video": {
      "path": "media/session-07.mp4",
      "sha256": "原始视频文件 SHA-256",
      "start_offset_seconds": 30.0,
      "duration_seconds": 15.0
    },
    "sensor": {
      "path": "sensors/session-07-window-003.csv",
      "sha256": "传感器文件 SHA-256"
    }
  }
}
```

`content_sha256` 不是任意占位值。它是对按输入键排序的原始文件 SHA-256 以及可选
视频窗口做域分离规范化哈希的结果。生成器逐文件重新计算 SHA-256，并要求该聚合值与
manifest 完全一致。相同长视频的不同窗口因窗口参数不同而得到不同内容指纹。

可用 Python 在锁箱准备阶段计算该值：

```python
from teaching_skill_miner.recognition import synchronized_content_sha256

record["content_sha256"] = synchronized_content_sha256(record["raw_inputs"])
```

然后使用更新后的有序 records 重新计算 `audit.dataset_fingerprint`。人工标签、身份字段
及审计声明仍必须由数据治理方提供；桥接器不会猜测或自动证明这些内容。

## 冻结提取配置

无私有路径的完整示例位于 `configs/raw_feature_extractor.example.json`。当前内置三类
真实执行后端：

- `builtin_classroom_video_visual_v1`：FFmpeg 解码固定窗口和固定数量帧，输出颜色、
  空间、边缘及运动描述；
- `builtin_classroom_video_audio_v1`：FFmpeg 解码与视频重叠的音频，输出能量、过零率、
  频谱与静音描述；没有可解码重叠音频时失败关闭；
- `builtin_numeric_sensor_summary_v1`：读取 CSV、JSON 或 JSONL 的数值列，验证有限值和
  可选单调时间戳，输出固定的均值、标准差、分位数、变化量与斜率。它不输出缺失标记或
  采样行数特征。

这些后端不使用学习权重。provenance 明确写入
`learned_weights_used: false`、空 `weight_artifacts` 及其指纹；不能把它描述成存在预训练
视觉/音频模型。每个模态同时绑定：

- 桥接器、媒体特征实现和探测代码的文件 SHA-256；
- 规范化模态配置及 SHA-256；
- Python、NumPy、平台、FFmpeg/FFprobe 版本与二进制 SHA-256；
- 无 shell 的实际执行协议、提取器 ID 和最终 extractor fingerprint。

因此训练站点与外部目标站点要被同一冻结模型接受，必须运行完全匹配的提取配置、代码和
运行时。任何变化都会使 provenance 不一致并失败关闭。

## 命令

仅生成严格特征包：

```bash
python3 -m teaching_skill_miner extract-strict-features \
  --manifest private/site_b_manifest.json \
  --raw-root /secure/site_b_lockbox \
  --extractor-config configs/raw_feature_extractor.example.json \
  --identity-field session_id \
  --identity-field teacher_id \
  --identity-field site_id \
  --output-dir private/site_b_features
```

输出固定为 `private/site_b_features/strict_feature_bundle.json`。POSIX 下输出目录为
`0700`，文件为 `0600`。bundle 不保存原始路径或媒体内容，只保存样本 ID、原始文件哈希、
窗口、特征与 provenance；它仍是含行级数据的私有研究产物，不得进入公开 wheel
或 release。

若已经有冻结 v3 checkpoint，可在提取时额外做兼容性检查：

```bash
python3 -m teaching_skill_miner extract-strict-features \
  --manifest private/site_b_manifest.json \
  --raw-root /secure/site_b_lockbox \
  --extractor-config private/frozen_extractor_config.json \
  --model private/frozen_model.json \
  --output-dir private/site_b_features
```

`--model` 只逐模态比较 feature schema 与完整 extractor provenance，不执行模型预测。
之后仍需独立签名冻结登记、一次性 ledger 和 `evaluate-frozen-recognition` 才能运行锁箱
评估；最终是否建立部署准确率继续由冻结 claim contract 的全部 gate 决定。

## 可审计链条

```text
原始文件实际 SHA-256 + 固定窗口
        -> record.content_sha256
        -> audit.dataset_fingerprint
        -> raw_source_binding.binding_fingerprint
        -> 有序 float64 strict feature fingerprint
        -> 冻结模型/外部登记/一次性评估
```

加载严格 bundle 时会重新验证 raw source binding 与 manifest 的每个
`content_sha256`、精确 sample order 和 binding fingerprint。篡改原始哈希、窗口、行顺序、
矩阵、配置、代码 provenance 或声明范围都会失败关闭。
