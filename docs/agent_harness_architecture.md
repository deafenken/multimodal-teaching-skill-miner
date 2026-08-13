# TeachLab Agent Harness 架构决策

状态：核心数据链已实现。Python Harness typed SSE、durable-first journal、cursor reconnect、显式取消、DeepSeek 普通文本/Web Search 原生流和 Next strict reducer 已接入 Console。Teach 的跨进程恢复已覆盖三个可证明边界：`start` 在 provider effect 前可安全恢复、未知 effect 不重放而 handoff、领域提交后可补齐外层 SSE；它不等于通用在途续跑，Chat 非终态 run 仍不自动恢复。生产身份、OS sandbox、跨浏览器/辅助技术认证仍未建立。

## 目标

TeachLab 的 Harness 不是给现有同步函数套一层名字。它必须成为一次 Agent 运行的事实拥有者：接收输入，建立有界上下文，调用模型，验证与执行工具，持久化每个确定边界，支持取消、重试、恢复和事件重放，并让 UI 只根据类型化事件更新。

教学策略仍属于 Teacher Agent。通用 Harness 不得自行判断掌握度，也不得让教学资源、gold、学习者答案跨越既有数据边界。

## 固定参考版本

本轮只读检出了以下上游版本，源码位于被 Git 忽略的 `.private/upstream_harness_refs/`，没有 vendor 到发布包：

| 项目 | 固定提交 | 许可/来源边界 | 本轮判断 |
| --- | --- | --- | --- |
| [OpenCode](https://github.com/anomalyco/opencode) | `0d927ba03f36d7f87e3cdb2b6c1f34c44913a099` | MIT | 生产闭环最完整；重点参考类型化事件、Tool registry/executor、权限门禁和消息 part 流 |
| [Pi](https://github.com/earendil-works/pi) | `40a3d8556ab7fb4a6b4da20ffe1f5dfc08ec121d` | MIT；旧 `badlogic/pi-mono` 已重定向 | 生产 Agent Loop 可参考 steer/follow-up；新的 durable AgentHarness 仍是规范与 scaffold，不能作为可运行依赖 |
| [OpenHarness](https://github.com/AgentBoardTT/openharness) | `85c54682a209ca7c3fc8b1ab2e820b6724dc3028` | `pyproject.toml` 声明 MIT，但该提交没有 LICENSE 文件 | 只参考轻量 Provider/Tool Protocol 和测试注入面；不采用其持久化、sandbox 或 hook 安全声明 |
| [Grok Build](https://github.com/xai-org/grok-build) | `b13fa526f5112c0b20dad5f1f2300d3d3b127895` | Apache-2.0；third-party 子树另有许可 | 参考单写者 actor、prepare/approve/dispatch、per-resource lock、durable terminal ordering、UI reducer 与 checkpoint 纪律 |

本实现采纳架构原则并在 Python/TypeScript 边界重新实现，没有复制上游源文件。若未来复制具体实现，必须重新做逐文件许可审计并保留相应 notice。

## 取舍

### 采用

- 一个 session/operation 只有一个事实写入者；所有模型、工具、取消、steer、timer 结果回到同一状态机结算。
- 类型化 `run/model/message/tool/action` 事件，严格单调 sequence，且每次运行恰好一个 terminal event。
- Tool definition、registry、executor 三层分离；schema、permission、risk、timeout、retry、replay policy 在执行边界统一检查。
- 外部 effect 使用 `intent -> effect -> settlement`：先持久化意图，执行后再持久化 receipt；崩溃恢复不靠猜测。
- Provider 原生流统一成 canonical stream event；消息 delta 与最终 durable commit 是两个不同事实。
- steer 与 follow-up 分开：steer 可在尚未提交的当前动作前改变方向，follow-up 只在当前 turn 完成后进入下一 turn。
- snapshot/cursor + 事件增量重连；UI 用 `(run_id, turn_id, sequence)` high-water mark 去重并拒绝过期事件。
- transport retry、operation retry、overflow compact-and-retry 使用独立预算，所有等待可取消。

### 明确拒绝

- 不移植 OpenCode 的 Effect/Layer 服务图、双事件桥或任意项目 TS 动态插件执行。
- 不依赖 Pi 尚未实现的 AgentHarness scaffold，也不把 transcript 恢复宣传成在途 effect 恢复。
- 不采用 Grok Build 的超大单文件、默认关闭 sandbox 或隔离失败后继续裸跑的策略。
- 不采用 OpenHarness 的 shell 字符串 deny、非原子 JSONL、弱 checkpoint 或未进入执行器的 hook 作为安全边界。
- 不把 prompt 中的“请勿调用”视为权限；不把 Project Trust 视为 OS sandbox。
- 不承诺外部副作用 exactly-once。安全工具可重放，幂等工具必须有 idempotency key，`never` 工具状态不明时必须 handoff。

## 本地模块

`teaching_skill_miner/harness/` 是领域中立核心：

- `contracts.py`：预算、重试、模型响应、ToolCall 与稳定 schema 标识。
- `events.py`：单调 event envelope、唯一 terminal、公开内容脱敏投影。
- `tools.py`：ToolSpec/Registry、输入输出 schema、权限、重试、超时、receipt。
- `checkpoint.py`：绑定 context hash 的防篡改 operation checkpoint。
- `journal.py`：SHA-256 hash-chain JSONL、`fsync` ack、断尾恢复、原子 checkpoint 与 cursor replay；单写者热路径复用已经验证的内存 tail/head 与文件 signature，不再为每个 provider delta 全链重扫，打开/恢复或检测到外部文件变化时仍完整验证。
- `runtime.py`：有界模型/工具循环、effect intent、恢复策略和最终结算。
- `controller.py`：run handle、即时取消、steer/follow-up queue 与事件订阅；queue 已有库级契约与测试，但当前 Console 尚未把运行中 steer 暴露为 API，Chat follow-up 仍是在前一轮完成后提交下一轮。
- `providers.py`、`deepseek.py`：Provider capability 与 DeepSeek 原生流适配；普通 Chat 和 Web Search 都按默认约 40 ms 或累计 512 字符聚合 assistant delta，并在工具/usage/结束边界 flush，拼接内容与 provider 原文严格一致，不做假字符切分。

`teacher_agent_harness.py` 是教学领域适配层。它将当前七个 Teacher Agent 工具注册为有版本、分权限、分 data scope 的 ToolSpec：`inspect_student_state`、`inspect_recent_history`、`search_skills`、`select_skills`、`set_next_focus`、`evaluate_termination` 与 `retrieve_resources`。最后一个工具只检索教师导入资源的私有本机索引，返回有界片段和位置/内容哈希 provenance，不能成为 learner-answer evidence 或 scoring gold。Planner 可见的工具名称、版本、输入 schema、权限、data scope、consent 与风险摘要直接来自本次 `HarnessModelRequest.tools` / `ToolRegistry`；隐藏、未授权、scope 未信任或未获得用户 consent 的工具不会残留在另一个提示词 allowlist 中，执行器仍是最终权限边界。恢复 context hash 还绑定 allowed permissions、受信 data scopes 和本轮可见 Tool definitions，不能借 resume 扩权。`teacher_agent_live.py` 通过该适配层运行，旧 `LOOP_SCHEMA` 输出保持兼容，同时只保存内容安全的 `harness_trace`。

## 运行与持久化不变量

1. `run.started` 是第一条事件；恢复会再发 `run.started(resumed=true)`，但使用后续 sequence。
2. sequence 从 1 连续递增；同一 run/turn 不允许分叉写入。
3. `run.completed|cancelled|failed|handoff` 只能出现一个；terminal 后拒绝任何事件。
4. journal `append()` 只有在完整写入并 `fsync` 后才返回 durable ack；SSE 和 UI 只能看到已经越过该边界的事件，terminal 也不例外。
5. journal 热路径在文件 signature 未变化时复用已验证的 records/head/terminal，并只校验、写入和 `fsync` 当前新事件；cursor replay 直接按连续 sequence 切片。首次打开、恢复、写入结果不确定或检测到外部修改时仍重扫并验证整条链，性能优化不能降低破坏检测。
6. journal 只截断没有换行的最后一条 UTF-8/JSON 撕裂记录；完整坏行、hash 错、内部坏行全部 fail-closed。
7. checkpoint 通过临时文件、file `fsync`、atomic replace、directory `fsync` 发布，并绑定 journal sequence/head hash。
8. checkpoint context hash 与恢复请求不一致时拒绝恢复。
9. 工具 intent 已提交而 settlement 未知时：`safe` 可重试，`idempotent` 只凭相同 key/arguments receipt 复用，`never` 不重放并进入 handoff。
10. cancel 与 domain commit 共享一个锁和单调状态：cancel 先赢则禁止进入 commit，commit 先赢则后到 cancel 返回拒绝，不能把已经提交的成功改写成 cancelled。适配器在取消时主动关闭已经建立的 provider 响应；DNS/TCP 建连完成前仍依赖请求 timeout。
11. Provider 在已经发布 assistant 文本或 tool effect 后不得 transport retry；小 delta 可以聚合，但必须保持顺序、完整拼接并在边界最终 flush，不能伪造字符动画。
12. public trace 不含 learner text、prompt、模型正文、tool arguments/results、教学资源正文或 gold，只保留分类、计数、hash、延迟与结果状态。用于重连的私有 stream journal 必须保存 assistant delta，因此不是脱敏 public trace：文件权限为私有边界，启用持久 session store 后要按学生数据一起保护和执行保留策略。
13. Web Search 只公开 `tool.started/progress/completed/failed`、安全错误码、计数和经清洗的有界来源标题/URL；搜索正文、citation text、provider encrypted content、内部 reasoning 与原始 tool payload 不进入公开 SSE/receipt。

## 事件协议

权威 envelope：

```json
{
  "schema": "teaching_skill_miner.agent_harness_event.v1",
  "event_id": "content-derived-id",
  "run_id": "run_...",
  "turn_id": "turn_...",
  "sequence": 1,
  "type": "message.delta",
  "timestamp": "2026-08-11T12:00:00Z",
  "payload": {}
}
```

生命周期主干：

```text
run.started
  -> model.started
  -> message.start -> message.delta* -> message.end
  -> model.completed
  -> tool.requested -> tool.started -> tool.progress* -> tool.completed
  -> action.completed
  -> run.completed
```

模型、工具和运行失败都有独立事件。`message.end` 仅表示 provider 消息流结束；只有 durable `action.completed`/terminal 表示该动作已被 Harness 接受并结算。

对应发布契约：

- `schema/agent_harness_event.schema.json`
- `schema/agent_harness_checkpoint.schema.json`
- `schema/agent_harness_trace.schema.json`
- `schema/harness_journal_record.schema.json`
- `schema/harness_journal_checkpoint.schema.json`

## 教学专用约束

- `explicit confusion`、学习者澄清问题、重复投诉等教学义务由 Teacher Agent 在 planner 前形成结构化 policy；Harness 负责保证该 policy 所调用的工具、动作验证与事件不会被旁路。
- “不会/不懂”不是掌握证据，不更新 mastery/student model；下一动作必须先由教师讲解或示范，禁止换词复问同一个高负担问题。
- 教学资源是 teacher context，不是 learner answer；资源/OCR、画像、评分与 gold 使用不同 Tool permission/data scope。
- 学习项目以本机私有 store 持久化元数据、Chat threads、notes 以及资源/大纲/Teach 引用；回收站通过 token 恢复。项目上下文不进入掌握度证据，项目引用也不提升为评分 gold。
- 项目 Chat 完整 transcript 由服务端权威持久化。向 provider 发送的长上下文最多接受 400 条交替消息，只有旧前缀已经 durable 才允许压缩；压缩是带原 transcript hash 的抽取式索引，不是会改写事实的模型摘要。
- 教师资源原媒体在本机提取/OCR 后删除；私有索引只保存受限文本、chunk offsets 和 provenance。`retrieve_resources` 的结果仍是教师上下文，不能用于伪造学生回答或直接更新 mastery。
- 工具或模型完成不等于教学动作可见。动作仍须通过现有 Skill、阶段、连续性、答案边界与 mastery 门禁。
- Teach-first 必须真实走完讲解、示范、带练、核验和迁移；迁移证据之后还要完成一次学习者总结才能成功。总结作为第六阶段内部收口，不凭教师话语、先验掌握或阶段标签自动完成。

## 安全与隔离边界

当前中央 permission 是应用层 capability gate，不等于 OS sandbox。现有 Teacher Agent 工具是进程内、allowlist、无任意 shell 的受限工具；未来若加入文件写入、代码执行或任意网络访问，必须同时提供：

- server-side 路径/域名/秘密/PII policy；
- 容器或系统级进程、文件系统、网络隔离；
- effect classification、idempotency/replay policy 和审计 receipt；
- fail-closed 的 sandbox 启动失败策略。

DeepSeek web search 只能作为独立 capability/tool 接入，不能因为 provider 声称支持就默认开放，也不能将网页内容当作评分 gold。
Console 默认关闭 Web Search。只有用户显式开启并确认独立远程检索授权，Harness 才会向同时声明 `requires_user_consent=true` 且 data scope 被信任的 provider-managed tool 发放 `remote_consent`；未授权时工具不会出现在模型可见 definitions 中。

## 当前实施状态与恢复边界

- Python `/api/stream` 对 Chat、教学 start 和 step 发布 canonical typed SSE；每条事件先 journal durable ack，再进入连接。客户端断线后用同一 `run_id + turn_id + after_sequence` 继续，服务端拒绝越过 durable head 的 cursor、不同请求复用 identity 和已损坏 journal。
- DeepSeek 普通文本使用 `/chat/completions` 原生 SSE；Web Search 使用 Anthropic Messages server-side tool 原生 SSE。两者隐藏 reasoning，Web Search 映射为类型化工具生命周期并与非流路径复用同一来源清洗规则。
- Next `harness-stream.ts` 是 fail-closed reducer：校验 event schema/type/id、run/turn identity、严格连续 sequence、重复事件是否等价、唯一 `operation.result`、教学 `state.committed` 与结果引用一致、terminal 顺序，并用最终 message hash/字符数核对 Chat 拼接结果。Teach 终态再回取权威 Session，并强制核对 `session_id + context_version + response_sha256`；该摘要覆盖稳定教学快照，刻意排除会在回取前从 active 变 idle 的临时 worker 状态。连接恢复不是重新执行同一请求。
- Console 取消调用独立 `/api/cancel`。取消可以在 run identity 注册前用 request tombstone 抢先登记，也可以关闭活动 provider socket；commit-wins 语义保证最终状态唯一。兼容旧同步 `cancel_turn` 的 `remote_transport_cancellation_supported=false` 与 Harness 的 `harness_sse_transport_cancellation_supported=true` 是两条不同路径。
- 同一进程内的活动 run 支持 cursor reconnect；只有配置持久 `--session-store`、因而 stream journal 根目录可跨进程发现时，Python 重启后才能发现 Teach operation checkpoint 或重放已有 terminal 的 journal。临时目录部署不声明跨进程能力。
- Teach active restart 不是笼统的“续跑”：`start` 在 `context_prepared`、provider effect 尚未开始时可按同一 exact receipt 安全恢复并只执行一次；若 start/step 已进入 provider 而 effect 结果未知，则绝不猜测或重放，直接 `run.handoff(reason=unknown_domain_effect_after_restart)`；学生 `step` 不在重启后重放。若领域 turn 已经 commit、只是外层 SSE 尚未完成，则从持久化 Session/幂等 receipt 补齐同一教师消息和 terminal，不再次推进领域状态。
- Chat 非终态 run 不自动恢复或重新调用 provider；只有已经 terminal 的持久 journal 可跨进程重放。上述 Teach 特例不改变 Chat 边界，也不等于任意 tool/provider 的分布式 workflow recovery。
- 教学 Session 默认仍在内存中，显式 `--session-store` 才支持既有 cold resume；项目与资源索引使用各自显式私有路径。三者都不等于登录、加密、跨设备同步或多副本数据库。
- 每个外部运行在执行前先持久化 exact registered receipt，结算后再 sealed；详细 journal 最多保留 48 条，请求 identity receipt 最多 4096 条且不自动 LRU 淘汰。达到上限会在新 effect 前 fail closed，必须由运维按受保护数据策略归档或清理，不能用删除幂等证据换容量。
- Chat 文本来自 provider 原生增量。Teach 的结构化 planning token 不公开；最终教师动作通过领域提交与教学门禁后，按自然语句无损拆成最多 7 个有序 delta 渐显，末端 hash 必须与完整提交文本一致。这是“已验证结果渐显”，不是未经验证的 provider token stream。
- `scripts/start_teacher_agent_console.mjs` 使用哈希绑定、逐文件校验的 immutable Next standalone production runtime，默认后台驻留并只打印 Console URL；macOS `打开题目二教学Agent.command` 会把 `TEACHLAB_OPEN_BROWSER` 默认设为 `1`，ready 后自动打开一次，设置为 `0` 可关闭。直接调用 Node 启动器仍需显式 `TEACHLAB_OPEN_BROWSER=1`。关闭启动 Terminal 不停止后台服务，`--stop` 或受本地 session/CSRF 保护的 UI stop API 才会回收进程组；空闲停止默认关闭。浏览器仍不持有 Python capability URL。过去由启动器反复调用 `open` 创建的旧标签页需要用户关闭一次，启动器不越权关闭整个浏览器或其他页面。
- Console 启动器使用原子 v3 主锁记录 launcher、launch nonce、子进程 PID/PGID、OS 启动标记和命令摘要；重复启动在执行任何后端、Next 或浏览器 effect 前拒绝。npm/Next 不再作为易变的身份组长，而是运行在持久 Node supervisor 的独立 PGID 下：supervisor 保留 nonce，npm、Next、next-server 与本项目精确路径的 PostCSS worker必须沿完整 PPID 链回到它并命中窄命令白名单。正常退出只有在两个 PGID 明确消失后才释放锁；死 launcher 只在完整身份和成员快照均匹配时发送 `SIGTERM`，PID 复用、断链/陌生成员、`EPERM` 未知态、spawn 登记窗口崩溃和顽固残留一律 fail closed，不能把它宣传成通用进程监管器。
- 上述实现不改变生产边界：当前 opaque handle 不是身份系统，应用层 Tool permission 不是 OS sandbox，受控 Chrome 验收不是 Firefox/Safari/屏幕阅读器认证，Web Search 结果也不是教学评分 gold。

## 验收标准

- 契约：非法 schema、参数、permission、event sequence、terminal 后事件全部 fail-closed。
- 预算：model/tool/repeat/deadline 上限确定性生效，retry 不越总 deadline。
- 取消：provider/tool 进行中可取消，已建立的 DeepSeek 流会被关闭；cancel-wins 后没有 late commit，commit-wins 后取消不得反转结果。
- 恢复：通用安全 effect 可按 replay policy 恢复，未知 `never` effect 不重放；Teach 只额外允许 start pre-provider 安全恢复、未知 start/step effect handoff、domain commit 后补 SSE，Chat 非终态不自动恢复；journal 断尾恢复但内部损坏拒绝。
- 流式：Chat 的首个 `message.delta` 来自完整 provider 回复结束之前的原生流；Teach 明确采用提交后、按自然语句的无损渐显，不泄露 planner 草稿，也不伪称逐 token 传输。两者都禁止前端按字符伪造模型流。
- 重连：`after_sequence` 不丢不重，旧 run/turn 事件不会写入当前消息。
- 隐私：公开 trace、SSE internal channel、日志和错误中无 prompt、资源正文、learner text、搜索正文、原始 tool payload 或 gold；私有 stream journal 因重连需要包含 assistant delta，必须与 session store 分别按其实际内容保护，不能称为匿名 public receipt。
- 教学：现有 mastery/gold/Skill/lesson-flow 回归全部保持，并新增“不会后先讲解”“澄清先回答”“重复问题换法”的跨学科用例。

只有上述链路在真实 DeepSeek、故障注入、Python 全量测试及 Console build/browser 验收同时通过，才能把 Harness 标为完成。
