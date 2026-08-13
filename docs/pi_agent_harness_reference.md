# Pi Agent Harness 参考设计

## 基线

- 上游：`https://github.com/earendil-works/pi`（旧地址 `badlogic/pi-mono` 会跳转到该项目）
- 本次最终审阅提交：`40a3d8556ab7fb4a6b4da20ffe1f5dfc08ec121d`
- 审阅日期：2026-08-11
- 许可证：MIT
- 本地只读检出：`/tmp/pi-harness-reference.KRy6LK`

本项目不直接复制 Pi 的 Coding Agent，也不把 TypeScript Agent 运行时嵌入 Python 教学内核。参考目标是它的运行语义、事件协议和会话边界；TeachLab 的教学证据、掌握度更新和 gold 隔离仍由现有 Python 领域内核负责。

## Pi 中值得采用的设计

### 1. 运行状态不是一个 `busy` 布尔值

Pi 把一次运行拆成 `agent_start`、`turn_start`、消息开始/增量/结束、工具开始/更新/结束、`turn_end`、`agent_end` 和最终 settled。UI 订阅事件并投影状态，不需要猜测后端正在做什么。

TeachLab 应采用稳定的运行状态：

`idle -> accepted -> reasoning -> tool -> responding -> settling -> completed | failed | aborted`

其中 `accepted` 表示请求已被 Harness 接收；模型或工具的后续失败通过运行事件报告，而不是让前端误以为“发送失败”。

### 2. 当前运行和后续输入分离

Pi 区分：

- `steer`：当前 turn 的工具结束后，下一次模型调用前插入；
- `followUp`：当前运行自然结束后继续；
- 普通 prompt：空闲时建立新运行。

TeachLab 首阶段只为 Chat 接入 `followUp`。Teach 必须等待教师问题完整生成后才能接受学生证据，因此不开放无条件 steer，也不把提前输入计入学习证据。

### 3. 转录树和运行日志分离

Pi 的新 Harness 将会话内容 `Entry` 与运行记录 `Record` 分开：

- Entry：用户消息、助手消息、模型变更、压缩摘要、自定义内容；
- Record：operation started/finished、abort、step attempt、tool started、queue、usage；
- lane 指向当前分支叶子，追加内容形成可导航树；
- 未完成 operation 可在崩溃后识别并恢复或明确失败。

TeachLab 已有事件存储、幂等键、active-turn fence 和 context version，基础并不弱。下一步应把这些能力统一成显式 `run_id` 和 operation journal，而不是在前端继续增加临时状态。

### 4. UI 只消费增量事件，最终消息是权威值

Pi 的 JSON/RPC 边界不在线路中重复发送完整 partial snapshot：`message_start` 建立消息，delta 累加，`message_end` 给出最终权威消息。这能降低网络和 React 更新成本，也避免 partial 与最终结果不一致。

审阅开始时，TeachLab 的 Next SSE 还是“先等待 Python 返回完整 JSON，再按固定间隔切成字符块”的显示动画。本轮已将它替换为类型化 Harness 事件链：

1. 后端先持久化再发送真实 lifecycle/tool 事件；
2. Chat 使用供应商原生 token stream；
3. Teach 的结构化决策继续原子校验和提交，提交前只展示真实阶段，不伪造 token；
4. `message.end` 只表示消息流结束，`action.completed` 与 terminal 才表示已校验、已提交；
5. Next 只透传 SSE 字节流，React reducer 按 run/turn/sequence 去重、拒绝跳号并支持游标重连。

### 5. 工具是可观察、可门禁的 Harness 能力

Pi 为工具提供参数校验、before/after hook、进度更新、终止提示、并行/串行策略和错误归一化。TeachLab 对应的能力应建模成工具，而不是隐藏在一条状态文案后：

- `web_search`：Chat 可用，显示搜索开始、结果数和来源；
- `resource_retrieval`：Teach 可用，只读取教师资源上下文；
- `learner_evidence_check`：Teach 内部能力，输出安全的判定摘要；
- `skill_route`：输出候选、门禁结果和最终选择，但不暴露 gold；
- `profile_commit`：只在回合提交阶段执行。

## TeachLab 目标边界

| 层 | 职责 | 不负责 |
| --- | --- | --- |
| Console UI | 消息投影、运行状态、队列、停止/重试、来源与工具卡片 | 教学决策、掌握度计算 |
| Run Harness | `run_id`、事件顺序、队列、取消、重试、工具调度、恢复 | 判断学生是否掌握 |
| Chat profile | 自由对话、按需 Web Search、follow-up | 写学生画像 |
| Teach profile | 调用教学规划器、资源检索、证据门禁、原子提交 | 读取 gold、用教师资源替代学生作答 |
| Teaching Core | Skill 选择、误解/掌握度、上下文版本、commit fence | UI 动画和网络传输 |

Chat 和 Teach 应是同一个 Harness 的两个 profile，而不是两套互不相干的前端请求状态机。

## 分阶段落地

### Phase 1：交互连续性

- Chat 运行中允许输入并进入可见、可取消的 follow-up 队列；
- 当前回答结束后按顺序自动发送；
- Teach 保留严格回合门禁；
- 停止按钮只终止当前运行，队列状态明确可见。

### Phase 2：统一事件协议

- 每个请求生成 `run_id`；
- SSE 统一为 `run.accepted`、`turn.started`、`tool.*`、`message.*`、`run.settled`；
- 前端用 reducer 从事件投影界面，移除全局 `busy` 对所有交互的粗粒度封锁；
- 历史记录存最终消息和 operation receipt，不存逐 token delta。

### Phase 3：真实流与工具可见性

- Chat 接 DeepSeek 原生流；
- Web Search、资源检索显示真实 tool card；
- 自动重试显示倒计时并可取消；
- 错误按 preflight rejection、run failure、tool failure 分类。

### Phase 4：持久恢复与分支

- 将 active turn、pending writes、queue 和 usage 纳入 durable operation journal；
- 刷新后恢复未完成运行或给出明确的 suspended/failed 状态；
- Chat 支持从历史消息分支；Teach 只有在不破坏证据链时才允许分支，并创建新的教学 session/profile revision。

## 当前不直接采用的部分

- 不直接依赖 Pi 新 `AgentHarness`：在审阅提交中，其 `prompt`、`steer`、`compact`、`resume`、watch 和 lane 主体仍抛出 `HarnessNotImplemented`；接口可参考，运行时尚不能作为生产依赖。
- 不让通用 Agent 直接写学生画像：所有画像更新继续经过 Teaching Core 的验证和原子提交。
- 不把 Teach 改成无限自主工具循环：每轮仍要满足 `wait_for_student_before_next_action`，防止 Agent 自说自话完成多个教学回合。
- 不把供应商 thinking 原文展示给学生；只展示可审计的阶段和工具结果摘要。
