# Agent Harness 对表：Claude Code 与 Codex CLI

基线日期：2026-08-24。竞品能力只引用官方文档；“当前状态”以本仓库实现为准。

| 能力 | Claude Code | Codex CLI | Agent Harness 当前状态 |
|---|---|---|---|
| 交互式终端 | 完整交互模式、工具进度、中断和状态显示。[官方文档](https://code.claude.com/docs/en/interactive-mode) | 原生 TUI，提供 `/status`、`/permissions`、`/model`、`/review`。[官方文档](https://developers.openai.com/codex/cli/features) | 已实现 curses TUI、原生流、工具状态、取消、follow-up 队列、model/permission/cache 状态，以及 `/effects`、`/reconcile` 人工处置入口。 |
| Session | continue、按 ID/名称恢复、fork。[官方 CLI reference](https://code.claude.com/docs/en/cli-reference) | resume、fork，以及交互/headless 续接。[官方 CLI reference](https://developers.openai.com/codex/cli/reference) | 已实现 list/resume/fork/archive 和恢复时只读降级；fork 只是 transcript/元数据复制，不隔离环境、进程堆或 worktree。未决 run 触发 workspace-wide fence，跨进程自动续跑尚未实现。 |
| 权限与沙箱 | allow/deny/ask 规则和多种 permission mode。[官方权限文档](https://code.claude.com/docs/en/permissions) | approval policy 与 read-only/workspace-write/danger-full-access sandbox。[官方安全文档](https://developers.openai.com/codex/agent-approvals-security) | 已实现三档工具授权和内建文件/补丁路径检查；`data_scope` 只是授权元数据。`full-access` 是当前 OS 用户的主机级 shell，可读取/输出秘密且能通过 daemon 化逃逸初始进程组；逐次 approval broker 和持久规则仍是 P1。 |
| 工具生命周期 | 流式工具调用、hooks、中断和 transcript。[交互模式](https://code.claude.com/docs/en/interactive-mode) | 交互及 JSONL 均暴露执行进度。[非交互模式](https://developers.openai.com/codex/noninteractive) | 已实现 typed tool events、重试、拒绝、durable-first journal 和 effect ledger；未决 effect 只能由操作员检查后确认清除 fence，不是自动 reconcile/rollback/replay。 |
| MCP | stdio/HTTP、OAuth、多 scope 和诊断。[官方 MCP 文档](https://code.claude.com/docs/en/mcp) | MCP client/server 管理和 `/mcp`。[官方 MCP 文档](https://developers.openai.com/codex/mcp) | 未实现，P1。 |
| Instructions / hooks | `CLAUDE.md`、skills、hooks。[Memory](https://code.claude.com/docs/en/memory)、[Hooks](https://code.claude.com/docs/en/hooks) | `AGENTS.md`、skills、hooks。[AGENTS.md](https://developers.openai.com/codex/guides/agents-md)、[Hooks](https://developers.openai.com/codex/hooks) | 不内置领域 Skill 子系统；通用 `AGENTS.md`/hooks loader 尚未实现，P1。 |
| Headless | `claude -p`、stream-json、budget、Agent SDK。[官方 headless 文档](https://code.claude.com/docs/en/headless) | `codex exec`、JSONL、resume 与 SDK。[非交互模式](https://developers.openai.com/codex/noninteractive)、[SDK](https://developers.openai.com/codex/sdk) | 已实现 `harness exec`、`--jsonl`、稳定退出码和 session 续接；SDK 与成本 budget 尚未实现。 |
| Subagents | 独立上下文、模型、工具和权限，可后台/worktree。[官方文档](https://code.claude.com/docs/en/sub-agents) | 可配置 subagents 与 `/agent`。[官方文档](https://developers.openai.com/codex/subagents) | 未实现，P1。 |
| Context compaction | 自动压缩及 `/compact`、`/context`。[工作原理](https://code.claude.com/docs/en/how-claude-code-works) | `/compact` 与自动压缩阈值。[配置参考](https://developers.openai.com/codex/config-reference) | 有硬 budget 和完整 transcript；具血缘的 compaction 尚未实现，P1。 |
| 文件/图片 | `@` 路径和图片粘贴。[官方教程](https://code.claude.com/docs/en/tutorials) | TUI 图片与 `--image`。[CLI features](https://developers.openai.com/codex/cli/features) | 已实现文本工作区工具；TUI 图片输入和 provider-neutral attachment 尚未实现。 |
| 状态与成本 | 可定制 statusline，含模型、目录/git、上下文和成本。[Status line](https://code.claude.com/docs/en/statusline) | `/status`、`/usage`、statusline。[CLI reference](https://developers.openai.com/codex/cli/reference) | 已显示 provider/model/cwd/permission、token 和真实 cache hit/miss；成本只在 provider 能给出可靠价格时才会加入。 |

## P0 已落地

- 独立 `agent_harness` 包和 generic schema namespace。
- Tool-aware provider adapter；planner envelope 与 reasoning 不进入正文。
- Coding workspace toolset 和三档真实授权。
- workspace-scoped private session、run journal、settled-session resume/fork/archive，
  以及 workspace-wide unfinished-run fence 和手工 `effects`/`reconcile` 入口。
- TUI、headless 流和 JSONL 事件流。

## P1 顺序

1. 逐次 approval broker + project/session 持久规则。
2. `AGENTS.md` / 通用 hooks 和可信项目配置。
3. MCP client manager（stdio、HTTP、OAuth、scope、trust、诊断）。
4. 有血缘的 context compaction 与 `/context`。
5. subagent tree、取消传播、并发上限与可选 worktree。
6. provider-neutral text/PDF/image attachment contract。
7. 稳定 Python/TypeScript SDK。

Harness 的差异化主线是 durable journal、tamper evidence 和保守的 effect fence。
当前 `reconcile` 是操作员确认记录，不是自动恢复或效果正确性证明。
