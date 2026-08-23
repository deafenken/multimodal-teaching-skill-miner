# Agent Harness 对表：Claude Code 与 Codex CLI

基线日期：2026-08-24。竞品能力只引用官方文档；“当前状态”以本仓库实现为准。

| 能力 | Claude Code | Codex CLI | Agent Harness 当前状态 |
|---|---|---|---|
| 交互式终端 | 完整交互模式、工具进度、中断和状态显示。[官方文档](https://code.claude.com/docs/en/interactive-mode) | 原生 TUI，提供 `/status`、`/permissions`、`/model`、`/review`。[官方文档](https://developers.openai.com/codex/cli/features) | 已实现 curses TUI、原生流、工具状态、取消、follow-up 队列、model/permission/cache 状态、审批 modal，以及 `/effects`、`/reconcile`、`/instructions`、`/hooks`、`/context`、`/approvals`。 |
| Session | continue、按 ID/名称恢复、fork。[官方 CLI reference](https://code.claude.com/docs/en/cli-reference) | resume、fork，以及交互/headless 续接。[官方 CLI reference](https://developers.openai.com/codex/cli/reference) | 已实现 list/resume/fork/archive 和恢复时只读降级；fork 只是 transcript/元数据复制，不隔离环境、进程堆或 worktree。未决 run 触发 workspace-wide fence，跨进程自动续跑尚未实现。 |
| 权限与沙箱 | allow/deny/ask 规则和多种 permission mode。[官方权限文档](https://code.claude.com/docs/en/permissions) | approval policy 与 read-only/workspace-write/danger-full-access sandbox。[官方安全文档](https://developers.openai.com/codex/agent-approvals-security) | 已实现三档授权、macOS Seatbelt `workspace.patch`/`process.exec`、单独的 `process.exec_host`、medium/high 逐次审批、headless handoff，以及私有 session/workspace 精确规则。Seatbelt 拒绝外部写、网络、signal 和已知 securityd/Keychain Mach 查找，但它是 allow-default 主机策略而非容器；后端尚未跨平台，unsandboxed host command 的 daemon 清理仍是 best effort。 |
| 工具生命周期 | 流式工具调用、hooks、中断和 transcript。[交互模式](https://code.claude.com/docs/en/interactive-mode) | 交互及 JSONL 均暴露执行进度。[非交互模式](https://developers.openai.com/codex/noninteractive) | 已实现 typed tool/hook events、重试、拒绝、durable-first journal 和 effect ledger；未决 effect 只能由操作员检查后确认清除 fence，不是自动 reconcile/rollback/replay。 |
| MCP | stdio/HTTP、OAuth、多 scope 和诊断。[官方 MCP 文档](https://code.claude.com/docs/en/mcp) | MCP client/server 管理和 `/mcp`。[官方 MCP 文档](https://developers.openai.com/codex/mcp) | 未实现，P1。 |
| 静态项目指令 | `CLAUDE.md` 分层项目上下文。[Memory](https://code.claude.com/docs/en/memory) | 根到 cwd 的 `AGENTS.md` 合并、override 和大小上限。[AGENTS.md](https://developers.openai.com/codex/guides/agents-md) | 已实现 workspace 范围内 `AGENTS.override.md` > `AGENTS.md`、根到 active 目录合并、32 KiB、no-follow/稳定读取、摘要审计和 planner/answer 同快照；尚未兼容 `CLAUDE.md`、imports 或懒加载 scoped rules。 |
| 可执行 hooks | 生命周期 hooks 和策略 hooks。[Hooks](https://code.claude.com/docs/en/hooks) | hooks 可观察或阻断 Agent 生命周期。[Hooks](https://developers.openai.com/codex/hooks) | 已实现可信 command hooks 的同步 `PreToolUse`/`PostToolUse`/`PostToolUseFailure` 子集：项目定义需 exact digest trust，变更后重新确认；pre 只能 `ask`/`deny` 收紧中央权限，post 只观察；入口快照在专用 read-only/no-network macOS Seatbelt profile 中执行且无不安全平台回退；事件只留 input/output digest。未实现其他生命周期、异步 hook、输入改写或完整配置 parity。 |
| Headless | `claude -p`、stream-json、budget、Agent SDK。[官方 headless 文档](https://code.claude.com/docs/en/headless) | `codex exec`、JSONL、resume 与 SDK。[非交互模式](https://developers.openai.com/codex/noninteractive)、[SDK](https://developers.openai.com/codex/sdk) | 已实现 `harness exec`、`--jsonl`、稳定退出码和 session 续接；SDK 与成本 budget 尚未实现。 |
| Subagents | 独立上下文、模型、工具和权限，可后台/worktree。[官方文档](https://code.claude.com/docs/en/sub-agents) | 可配置 subagents 与 `/agent`。[官方文档](https://developers.openai.com/codex/subagents) | 未实现，P1。 |
| Context compaction | 自动压缩及 `/compact`、`/context`。[工作原理](https://code.claude.com/docs/en/how-claude-code-works) | `/compact` 与自动压缩阈值。[配置参考](https://developers.openai.com/codex/config-reference) | 已实现稳定 message ID、append-only 原始 transcript、父子摘要血缘、摘要+后缀 active view、TUI/CLI `/compact`/`context` 和 80%→60% 自动阈值。摘要是有损用户数据而非授权，单段/总 passes 有硬上限；目前只接 DeepSeek compactor。 |
| 文件/图片 | `@` 路径和图片粘贴。[官方教程](https://code.claude.com/docs/en/tutorials) | TUI 图片与 `--image`。[CLI features](https://developers.openai.com/codex/cli/features) | 已实现文本工作区工具；TUI 图片输入和 provider-neutral attachment 尚未实现。 |
| 状态与成本 | 可定制 statusline，含模型、目录/git、上下文和成本。[Status line](https://code.claude.com/docs/en/statusline) | `/status`、`/usage`、statusline。[CLI reference](https://developers.openai.com/codex/cli/reference) | 已显示 provider/model/cwd/permission、token 和真实 cache hit/miss；成本只在 provider 能给出可靠价格时才会加入。 |

## P0 已落地

- 独立 `agent_harness` 包和 generic schema namespace。
- Tool-aware provider adapter；planner envelope 与 reasoning 不进入正文。
- Coding workspace toolset 和三档真实授权。
- macOS workspace command 沙箱、显式 host command、逐次 approval broker 和私有精确规则。
- 安全 `AGENTS.md` 快照、provider 双阶段接线与 instruction/context 诊断。
- 可追溯 context compaction：稳定消息身份、不可变原文、增量摘要、手动入口和自动阈值。
- exact-digest 授信的同步 command-hook 子集、单调策略聚合和 digest-only 审计事件。
- workspace-scoped private session、run journal、settled-session resume/fork/archive，
  以及 workspace-wide unfinished-run fence 和手工 `effects`/`reconcile` 入口。
- TUI、headless 流和 JSONL 事件流。

## P1 顺序

1. MCP client manager，先交付 stdio v1，再分阶段做 HTTP/OAuth。
2. subagent tree、取消传播、并发上限与可选 worktree。
3. provider-neutral text/PDF/image attachment contract。
4. 稳定 Python/TypeScript SDK。
5. 扩展 hooks 生命周期与异步模式，但只有在不削弱 exact trust、单调权限和沙箱边界时推进。

Harness 的差异化主线是 durable journal、tamper evidence 和保守的 effect fence。
当前 `reconcile` 是操作员确认记录，不是自动恢复或效果正确性证明。
