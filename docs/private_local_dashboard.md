# 本机双成果真实演示看板

## 目的与边界

`tsm dashboard-real` 用于在获授权的本机上同时核对两项项目成果：TeachObs 真实课堂场景的多模态行为识别，以及 MIT OCW 完整讲次的 Teaching Skill 蒸馏。它与 `tsm dashboard` 严格分离：后者是可公开分发的聚合看板，前者是读取私有研究产物的临时本机会话。

“真实”表示页面读取的是实际离线产物，而非演示占位数据。TeachObs 启动检查验证媒体、字幕、特征、样本顺序和冻结模型绑定，并在内存中重算冻结四臂逐样本预测。MIT 启动检查验证 10 讲语义 manifest、四臂报告、Full Skill、评估和 canonical fingerprint，并只向浏览器投影允许展示的字段。浏览器可以把用户输入的教学主题注入当前 Skill，生成目标、九步 procedure、观察信号、fallback 与验证题；这是与 Python executor 同义的确定性参数化执行，不是新的模型推理。浏览器不会在播放时重新执行 ASR、抽帧、OCR、CLIP、训练或调阈值。该页面不是实时识别服务、在线部署、跨站点验证、专家 Skill 金标准或学习效果证明。

## 显示内容

### 成果一：TeachObs 行为识别

页面按同一 15 秒时间轴关联以下内容：

| 区域 | 内容 | 解释边界 |
|---|---|---|
| 视频 | 操作者明确选择的完整课堂媒体 | 本机按需读取，不嵌入 HTML，不获得新的传播许可 |
| 字幕 | 当前时间窗的官方字幕、平台字幕或审计 ASR 文本 | 来源与时间轴 provenance 不等于逐字正确或 WER 已建立 |
| 教师侧可观察动作 | 从当前模型臂已有 39 类结果中显式投影 `Demonstration`、`Board work`、`Pointing`、`Gesture` 等 10 类 | 保留原模型分数、PRED 与发布共识 REF；不是新增姿态模型，分数不是校准概率 |
| 学生动作 | 明确显示“当前未建模” | TeachObs 没有逐场景学生动作真值，项目也没有角色跟踪或学生姿态模型；不得生成举手、记笔记、讨论等假结果 |
| 互动上下文 | `Checking`、`Cueing`、`Monitoring` 等教师教学代码 | 只说明与学生参与有关的教学上下文，不等于识别到学生动作 |
| 四臂预测 | transcript-only、transcript+audio、transcript+visual、full 的冻结逐样本输出 | 启动检查用冻结模型对既有离线特征在内存重算；播放期间不训练、调阈值或再次推理 |
| 样本关联 | 当前 scene/window、时间段、真值与必要的置信信息 | 仅供本地核对，不得导出为公开逐样本记录 |

四臂结果必须来自同一冻结模型族、相同标签顺序和相同样本顺序。页面并不把 Hamming accuracy 改称普通 Accuracy，也不把探索性测试集表现解释成部署表现。

### 成果二：MIT Skill 蒸馏

Skill 页从 10 个完整讲次中按需读取一讲经验证的 Full Skill：

| 区域 | 内容 | 解释边界 |
|---|---|---|
| 四步闭环 | 完整视频与官方字幕 → Skill 抽取 → 教学过程生成 → 自动评估 | 页面回放冻结抽取与评估产物；过程生成在本机按当前 Skill 参数化执行 |
| 九环节程序 | 按 Skill JSON 的实际 `procedure` 顺序展示 9 步 | 不按 canonical rank 重新排序；不同讲次允许有不同起始环节 |
| `observed_method` | 教师动作、instruction、expected signal、fallback、首末证据范围、`evi_*` 字幕片段 | 首末证据范围不是行为连续持续时长 |
| `recommended_enrichment` | 原视频未观察到时加入的规范化教学补充 | 不计作原教师方法证据 |
| `mme_*` 事件 | 类型、时间、模态、策略支持、等待时长或视觉语义摘要 | 是规则检测的候选事件；启发式 confidence 不是校准概率，低质量 OCR 原文不展示 |
| 策略证据 | 每个 observed strategy 的字幕证据数与入选多模态证据数 | 当前主要影响策略计分和一致性核对 |
| 教学过程生成 | 输入新 `concept` 和学习者水平，生成本轮目标、9 个步骤、观察信号、fallback 与 2 项验证 | 使用已经载入的 Skill 模板；不重新识别原视频，也不调用外部生成服务 |
| 自动评估 | 展示冻结 `evaluation.json` 的内部综合量表、7 个加权维度、6 项硬门槛和内部多模态证据一致性 | 评估 Skill 的结构质量与内部证据一致性；不是 Accuracy、专家质量评分或学习效果 |
| Runtime | `READY → PROCEDURE → VERIFY → COMPLETE`，未达标时重试当前步骤并显示已参数化 fallback | 证明生成过程的 schema 可执行；达标/未达标由演示者输入，不是自动判断学生是否学会 |

当前冻结产物包含 10 个完整讲次、2,270 个候选融合事件、59 条进入 Full Skill 的多模态辅助证据，以及 90 个 procedure 步骤（62 个 `observed_method`、28 个 `recommended_enrichment`）。90 步的直接 `mme_*` 引用数为 0，直接步骤证据仍是 `evi_*` 字幕记录；因此应表述为“字幕证据主导、音频和视觉辅助的 Skill 蒸馏”，不能表述为端到端多模态事件直接生成 Skill。

两项成果共享多模态证据和可追溯设计，但目前是两条独立验证轨：TeachObs 的 39 类预测尚未直接作为 MIT procedure 的输入。

## 安全模型

本机入口采用以下最低约束：

- 只监听 `127.0.0.1`；不支持 LAN、公网、反向代理、隧道或端口转发。
- 每次启动生成新的随机 capability token；没有正确 token 的请求被拒绝。
- HTML、JSON、字幕和视频响应均发送 `Cache-Control: no-store`；页面 JSON 不再包含 OCR 文字。
- 私有内容按请求从本机读取，不写进公共 `index.html`，也不复制到 `artifacts/public/`。
- 返回给页面的数据不应暴露绝对路径、源平台 URL、完整私有 manifest、原始 OCR、作业目录、嵌入向量或无关身份字段；原始 Skill JSON 必须先经过字段白名单投影。
- 服务停止后 token 失效；看板不提供持久用户账户，也不应被当作生产认证系统。

这些控制降低误暴露风险，但不能防御已被控制的主机，也不能替代操作系统账户隔离、磁盘加密、最小权限、屏幕共享管理、知情同意、伦理审批、许可检查与留存/删除制度。

## 运行前准备

1. 确认有权在当前设备上处理并展示所选课堂视频及其衍生数据。
2. 将视频、字幕、OCR、标签、特征和冻结模型保留在 `.gitignore` 覆盖的私有目录，例如 `artifacts/private/`；目录建议为 `0700`，文件建议为 `0600`。逐样本预测只在当前进程的内存中重算，不应另存为公开产物。
3. 不要为了演示效果在测试样本上重新训练模型或调阈值。先检查通用页面模板；这一步不读取私有数据：

   ```bash
   tsm dashboard-real --check-template
   ```

4. 对两个默认私有根目录执行 fail-closed 数据检查。它会验证 TeachObs 媒体/帧/字幕/特征/模型，以及 MIT 语义 manifest/消融报告/Full Skill/评估指纹，但不启动 HTTP 服务：

   ```bash
   tsm dashboard-real \
     --teachobs-root artifacts/private/external_datasets/teachobs \
     --skill-root artifacts/private/full_multimodal \
     --initial-lesson S24 \
     --initial-skill linear_algebra_l03 \
     --check-data
   ```

5. 数据检查通过后启动回放。`--port 0` 让操作系统随机选择一个空闲端口，避免把固定端口暴露为项目约定：

   ```bash
   tsm dashboard-real \
     --teachobs-root artifacts/private/external_datasets/teachobs \
     --skill-root artifacts/private/full_multimodal \
     --initial-lesson S24 \
     --initial-skill linear_algebra_l03 \
     --port 0
   ```

   `--teachobs-root`、`--skill-root` 默认就是上述路径；`--initial-lesson` 默认为 `S24`，`--initial-skill` 默认为 `linear_algebra_l03`。如果不希望自动打开浏览器，可增加 `--no-browser`，再手动使用终端打印的 `127.0.0.1` capability URL。完整参数可用 `tsm dashboard-real --help` 查看。
6. 只使用当前命令打印的本机会话地址；不要把完整地址或 capability token 粘贴到聊天、文档、Issue、日志或答辩截图中。行为识别页核对视频时间、字幕、教师动作 PRED/REF 和四臂输出；Skill 页依次核对讲次、抽取证据、Skill procedure，再输入一个新主题生成教学过程，最后展示 7 维自动评估与 6 项硬门槛。自动评估旁必须保留“不是 Accuracy/学习效果”的口径。发现哈希、顺序、范围或冻结产物不一致时应停止展示，而不是跳过检查。
7. 答辩结束后终止本机进程并关闭相关浏览器标签页。按研究数据治理计划处理浏览器痕迹、私有产物和备份；`no-store` 不能替代设备级清理。

## 答辩表述

推荐表述：

> 这个本机站点展示两项成果。第一项把真实 TeachObs 完整视频、字幕、教师行为标签和冻结四臂预测同步到 15 秒场景；第二项把 10 个 MIT 完整讲次的处理结果组织成“视频输入、Skill 抽取、教学过程生成、自动评估”闭环。我可以选择一讲，核对 observed/recommended 环节及其证据，再输入一个新主题，用当前 Skill 生成九步教学过程；页面随后展示与 Skill 文件哈希绑定的七维内部评估和六项硬门槛。两条数据轨目前独立，Skill procedure 仍由字幕证据主导；评估分数是结构与内部证据量表，不是 Accuracy、专家评分或学习效果。

不要表述为“实时识别”“在线部署”“已达到 0.9 部署准确率”或“真实视频进入了公开网页”。

## 发布检查

公开提交和构建前至少执行：

```bash
python3 scripts/audit_repository_privacy.py
python3 -m teaching_skill_miner release-audit artifacts/public
sh scripts/build_release_acceptance.sh
```

允许进入 GitHub 或 wheel 的只有通用服务器/模板代码、样式与脚本、治理文档和通过审查的聚合公开产物。以下内容始终排除：真实视频/音频、完整字幕或 ASR、OCR、帧、标签、逐样本预测、原始 Skill JSON、嵌入、本地路径、私有 manifest 和 capability token。打包 `private_demo.html`、`private_skill_demo.css` 与 `private_skill_demo.js` 不意味着任何私有输入被打包；最终必须审计 exact wheel，不能只检查源码目录。
