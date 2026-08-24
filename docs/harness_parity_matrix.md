# Agent Harness 对表：Claude Code 与 Codex CLI

基线日期：2026-08-24。竞品能力只引用官方文档；“当前状态”以本仓库实现为准。

| 能力 | Claude Code | Codex CLI | Agent Harness 当前状态 |
|---|---|---|---|
| 交互式终端 | 完整交互模式、工具进度、中断和状态显示。[官方文档](https://code.claude.com/docs/en/interactive-mode) | 原生 TUI，提供 `/status`、`/permissions`、`/model`、`/review`。[官方文档](https://developers.openai.com/codex/cli/features) | 已实现 curses TUI、原生流、工具状态、取消、follow-up 队列、model/permission/cache 状态、审批 modal，以及 `/effects`、`/reconcile`、`/instructions`、`/hooks`、`/mcp`、`/agents`、`/context`、`/approvals`。 |
| Session | continue、按 ID/名称恢复、fork。[官方 CLI reference](https://code.claude.com/docs/en/cli-reference) | resume、fork，以及交互/headless 续接。[官方 CLI reference](https://developers.openai.com/codex/cli/reference) | 已实现 list/resume/fork/archive 和恢复时只读降级；fork 只是 transcript/元数据复制，不隔离环境、进程堆或 worktree。未决 run 触发 workspace-wide fence，跨进程自动续跑尚未实现。 |
| 权限与沙箱 | allow/deny/ask 规则和多种 permission mode。[官方权限文档](https://code.claude.com/docs/en/permissions) | approval policy 与 read-only/workspace-write/danger-full-access sandbox。[官方安全文档](https://developers.openai.com/codex/agent-approvals-security) | 已实现三档授权、macOS Seatbelt `workspace.patch`/`process.exec`、单独的 `process.exec_host`、medium/high 逐次审批、headless handoff，以及私有 session/workspace 精确规则。MCP tools 只在 `full-access`，强制 high-risk once-only approval，persistent allow 不适用。Seatbelt 是 allow-default 主机策略而非容器；后端尚未跨平台，允许 fork 或 unsandboxed host command 时 daemon 清理仍是 best effort。 |
| 工具生命周期 | 流式工具调用、hooks、中断和 transcript。[交互模式](https://code.claude.com/docs/en/interactive-mode) | 交互及 JSONL 均暴露执行进度。[非交互模式](https://developers.openai.com/codex/noninteractive) | 已实现 typed tool/hook events、重试、拒绝、durable-first journal 和 effect ledger；未决 effect 只能由操作员检查后确认清除 fence，不是自动 reconcile/rollback/replay。 |
| MCP | stdio/HTTP、OAuth、多 scope 和诊断。[官方 MCP 文档](https://code.claude.com/docs/en/mcp) | MCP client/server 管理和 `/mcp`。[官方 MCP 文档](https://developers.openai.com/codex/mcp) | 已实现 **exact-digest trusted local stdio MCP tools client subset**，固定 `2025-06-18`：显式 trust/disable、显式 catalog refresh、每次调用 live-catalog 校验、read-only macOS Seatbelt、high-risk/once-only/never-replay、只读 `/mcp` 状态。未实现 HTTP/OAuth/resources/prompts/sampling/elicitation/tasks、活动 run 动态 catalog、full JSON Schema、二进制渲染或 server mode；direct executable digest 不是传递依赖完整性证明。 |
| 静态项目指令 | `CLAUDE.md` 分层项目上下文。[Memory](https://code.claude.com/docs/en/memory) | 根到 cwd 的 `AGENTS.md` 合并、override 和大小上限。[AGENTS.md](https://developers.openai.com/codex/guides/agents-md) | 已实现 workspace 范围内 `AGENTS.override.md` > `AGENTS.md`、根到 active 目录合并、32 KiB、no-follow/稳定读取、摘要审计和 planner/answer 同快照；尚未兼容 `CLAUDE.md`、imports 或懒加载 scoped rules。 |
| 可执行 hooks | 生命周期 hooks 和策略 hooks。[Hooks](https://code.claude.com/docs/en/hooks) | hooks 可观察或阻断 Agent 生命周期。[Hooks](https://developers.openai.com/codex/hooks) | 已实现可信 command hooks 的同步 `PreToolUse`/`PostToolUse`/`PostToolUseFailure` 子集：项目定义需 exact digest trust，变更后重新确认；pre 只能 `ask`/`deny` 收紧中央权限，post 只观察；入口快照在专用 read-only/no-network macOS Seatbelt profile 中执行且无不安全平台回退；事件只留 input/output digest。未实现其他生命周期、异步 hook、输入改写或完整配置 parity。 |
| Headless | `claude -p`、stream-json、budget、Agent SDK。[官方 headless 文档](https://code.claude.com/docs/en/headless) | `codex exec`、JSONL、resume 与 SDK。[非交互模式](https://developers.openai.com/codex/noninteractive)、[SDK](https://developers.openai.com/codex/sdk) | 已实现 `harness exec`、`--jsonl`、稳定退出码和 session 续接；SDK 与成本 budget 尚未实现。 |
| Subagents | 独立上下文、模型、工具和权限，可后台或用 worktree；另有 agent view、teams 与 `/batch`。[Subagents](https://code.claude.com/docs/en/sub-agents)、[并行 agents](https://code.claude.com/docs/en/agents) | 当前版本默认支持并行 subagent workflow，CLI `/agent` 可检查/切换 agent thread，并可配置 agent 模型与指令。[官方文档](https://developers.openai.com/codex/subagents) | 已实现保守子集：一次 high-risk/never-replay 审批后，1–4 个独立 child session 在 clean committed HEAD 的独立 Git worktree 中前台并发，父取消传播并 wait-all；结果限于有界摘要、状态、任务顺序/层级、改动/失败元数据和不透明 ID，changed/可疑 artifact 保留，CLI/TUI 可检查。未实现后台、恢复/steer、thread 切换、自定义 agent、模型路由、嵌套、自动 merge/apply/PR 或 teams。 |
| Context compaction | 自动压缩及 `/compact`、`/context`。[工作原理](https://code.claude.com/docs/en/how-claude-code-works) | `/compact` 与自动压缩阈值。[配置参考](https://developers.openai.com/codex/config-reference) | 已实现稳定 message ID、append-only 原始 transcript、父子摘要血缘、摘要+后缀 active view、TUI/CLI `/compact`/`context` 和 80%→60% 自动阈值。摘要是有损用户数据而非授权，单段/总 passes 有硬上限；目前只接 DeepSeek compactor。 |
| 文件/图片 | Desktop 可附加文件和图片，CLI 工作流支持引用文件。[Desktop 官方文档](https://code.claude.com/docs/en/desktop)、[常见工作流](https://code.claude.com/docs/en/common-workflows) | CLI 接受图片输入。[官方图片输入文档](https://learn.chatgpt.com/docs/image-inputs?surface=cli) | 已实现 provider-neutral 不可变快照与 descriptor：严格 UTF-8 文本在普通 DeepSeek 模型上可用；PNG/JPEG 只在显式选择 `deepseek-v4-flash-vision-exp` 时内联发送，不自动换模型或调用 Files API；PDF 可安全导入/表示，但当前 provider 会在建 run 前拒绝。CLI 有重复 `--attach`，TUI 有 `/attach`/`/attachments`/`/detach`；无拖拽、粘贴或剪贴板 parity 声称。 |
| 状态与成本 | 可定制 statusline，含模型、目录/git、上下文和成本。[Status line](https://code.claude.com/docs/en/statusline) | `/status`、`/usage`、statusline。[CLI reference](https://developers.openai.com/codex/cli/reference) | 已显示 provider/model/cwd/permission、token 和真实 cache hit/miss；成本只在 provider 能给出可靠价格时才会加入。 |

## P0 已落地

- 独立 `agent_harness` 包和 generic schema namespace。
- Tool-aware provider adapter；planner envelope 与 reasoning 不进入正文。
- Coding workspace toolset 和三档真实授权。
- macOS workspace command 沙箱、显式 host command、逐次 approval broker 和私有精确规则。
- 安全 `AGENTS.md` 快照、provider 双阶段接线与 instruction/context 诊断。
- 可追溯 context compaction：稳定消息身份、不可变原文、增量摘要、手动入口和自动阈值。
- exact-digest 授信的同步 command-hook 子集、单调策略聚合和 digest-only 审计事件。
- exact-digest trusted local stdio MCP tools client 子集、`2025-06-18` 固定协议、显式冻结
  catalog、live-catalog fail-closed 校验和 once-only external-tool approval。
- workspace-scoped private session、run journal、settled-session resume/fork/archive，
  以及 workspace-wide unfinished-run fence 和手工 `effects`/`reconcile` 入口。
- TUI、headless 流和 JSONL 事件流。
- 有界前台 subagent 子集：单批 1–4 并发、取消传播、进程内预算、独立 child session、
  clean committed HEAD Git worktree、一次性父审批、子权限上限、content-free lifecycle 和
  保守 artifact inspection；不声称 background/agent-thread/team parity。
- provider-neutral 附件 descriptor 和 owner-private 不可变 blob；8 个/24 MiB 每轮导入上限、
  16 个/24 MiB DeepSeek 活动请求上限、模型能力 fail-closed 预检、附件血缘和
  不跨越附件消息的 context compaction。

## P1 顺序

1. 稳定 Python/TypeScript SDK。
2. 在不削弱 exact trust、冻结 catalog、once-only approval 和 sandbox 边界的前提下，分阶段
   评估 MCP HTTP/OAuth、resources/prompts 等更广协议面；sampling/elicitation/tasks 和 server
   mode 仍需单独 threat model，不能从 tools client 自动推导。
3. 扩展 hooks 生命周期与异步模式，但只有在不削弱 exact trust、单调权限和沙箱边界时推进。
4. 单独 threat model 后再评估后台/可恢复 agent threads、自定义 agents、模型路由、显式
   merge/apply/PR 工作流与跨进程全局预算；不能从本次前台 wait-all 子集自动推导安全性。

Harness 的差异化主线是 durable journal、tamper evidence 和保守的 effect fence。
当前 `reconcile` 是操作员确认记录，不是自动恢复或效果正确性证明。
