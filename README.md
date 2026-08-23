# Agent Harness

一个可审计、provider-neutral 的 Coding Agent Harness，包含模型循环、工具执行、权限
边界、持久化事件协议和多种交互入口。

## 直接启动

macOS 双击：

```text
打开Agent Harness.command
```

终端启动：

```bash
./打开Agent\ Harness.command
```

安装为本地命令后：

```bash
.venv/bin/pip install -e .
harness                         # TUI
harness exec "检查当前改动"     # 流式 headless
harness exec --jsonl "运行测试" # JSONL 事件流
harness sessions
harness resume [SESSION_ID]
harness effects                  # 查看当前 workspace 的未决 run
harness reconcile RUN_ID         # 人工检查后确认并清除 fence
```

默认从 `read-only` 开始。需要修改文件时显式切换：

```bash
harness --permissions workspace-write
harness --permissions full-access
```

TUI 中使用 `/permissions workspace-write` 切换当前会话权限。

## 当前能力

- 有界、可取消的 model/tool loop，包含 retry、deadline、step 和 tool-call budget。
- 原生 provider 流；planner JSON 和隐藏推理不会进入对话正文。
- 中央工具生命周期：requested、started、effect-started、progress、completed、failed、
  rejected、replay；非安全工具只有在持久化 effect boundary 后才获准执行实际效果。
- 工作区工具：文件列表、读取、全文检索、统一补丁和命令执行。
- 三档真实工具授权：`read-only`、`workspace-write`、`full-access`。
- 每次 run 使用 0600 hash-chain journal 和原子 checkpoint。任一 session 留下未完成或
  待核对 run 时，整个 workspace 的所有新 run 都会被 fence 阻断；`effects` 只列出证据，
  `reconcile` 只记录操作员已检查并清除 fence，不会验证、回滚或重放外部效果。
- workspace-scoped 本地 session：list、resume、fork、archive。fork 只复制 transcript 和
  session 元数据，不创建新进程、容器或 worktree，也不隔离环境变量、provider client 或
  当前进程堆中的秘密。
- curses TUI：流式正文、工具状态、Ctrl+C 取消、后续输入队列、token/cache 状态栏。
- Headless JSONL：稳定事件 schema、session/run/turn identity 和退出码。

与 Claude Code、Codex 的逐项对表见
[docs/harness_parity_matrix.md](docs/harness_parity_matrix.md)。

## 权限模型

| 模式 | 文件读/检索 | 应用补丁 | shell |
|---|---:|---:|---:|
| `read-only` | 是 | 否 | 否 |
| `workspace-write` | 是 | 是，仅工作区 | 否 |
| `full-access` | 是 | 是，仅工作区 | 是；工作目录固定，但可按当前 OS 用户权限访问工作区外主机资源 |

内建 list/read/search/patch 工具拒绝绝对路径、`..`、符号链接和 `.git`、`.private`、
`.agent-harness`。这些路径检查不约束 `full-access` shell。shell 可以按当前 OS 用户权限
读取工作区秘密、工作区外文件、访问网络，并把内容写入有界工具输出；该输出可能作为
下一轮模型 observation 发送给 provider。环境变量白名单不保护文件、子进程继承状态或
已在进程堆中的秘密。

超时和取消会尽力回收初始进程组，但不是进程沙箱：命令可以通过 `setsid`、双重 fork 或
其他 daemon 化方式脱离该进程组并在 Harness 返回后继续运行。选择 `full-access` 等于授予
shell 当前 OS 用户本来拥有的主机权限。

`data_scope` 只是工具注册与授权时使用的元数据标签，用于决定工具是否可见、可执行；它
不是文件系统、网络、进程或内存隔离机制。实际边界分别来自内建 handler 的路径检查和
操作系统授予进程的权限。

## DeepSeek 配置

优先使用以下任一方式：

```bash
export DEEPSEEK_API_KEY='...'
export HARNESS_DEEPSEEK_API_KEY_FILE='/private/path/deepseek_api.txt'
```

在仓库内运行时也会发现 `.private/deepseek_api.txt`。普通私有文件以及指向 owner-only
私有目标的链接都会经过目录权限、类型、inode、size 和 `O_NOFOLLOW` 校验。

可选配置：

```bash
export HARNESS_DEEPSEEK_MODEL='deepseek-v4-flash'
export HARNESS_DEEPSEEK_TIMEOUT_SECONDS='60'
export HARNESS_DEEPSEEK_MAX_RETRIES='2'
export HARNESS_DEEPSEEK_THINKING='0'
```

TUI 只展示 provider 返回的 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`，不会
伪造缓存命中。planner 使用稳定的 system-prefix，便于服务端前缀缓存复用。

## TUI 命令

```text
/help
/new
/sessions
/resume [SESSION_ID]
/fork
/archive
/effects
/reconcile RUN_ID
/status
/model
/permissions [read-only|workspace-write|full-access]
/tools
/clear
/quit  （/exit 同义）
```

运行中输入普通文本会进入有界 follow-up 队列；`Ctrl+C` 取消当前 run，但不破坏已落盘
的 journal。若不可重放的外部效果已经开始，取消会转为明确 handoff，而不是谎报为安全
取消。`/effects` 展示整个 workspace 的未决 run；只有在人工检查 journal 和真实工作区
状态后才应执行 `/reconcile RUN_ID`。该命令表示“我已承担判断责任”，不是自动核验。

## 架构

```text
TUI / headless CLI
        │
Generic Session Store
        │
Harness Runtime ── Event reducer / Journal / Checkpoint
   │          │
Provider   Tool Registry ── built-in path checks / explicit host shell grant
```

包结构：

- `agent_harness/core/`：domain-neutral runtime、events、tools、journal、recovery。
- `agent_harness/providers/`：provider client 与 tool-aware adapter。
- `agent_harness/toolsets/`：coding workspace 工具。
- `agent_harness/session.py`：本地多轮 session。
- `agent_harness/tui.py`：终端 UI。
- `agent_harness/cli.py`：TUI/headless 入口。

2.0 以前的私有运行数据不会被迁移脚本删除或改写；它们不属于新包、默认 CLI 或当前
Harness session namespace。

## 验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check agent_harness tests_harness
build_dir=$(mktemp -d)
.venv/bin/python -m build --outdir "$build_dir"
.venv/bin/python -m agent_harness --cwd . status
```

Python 需要 3.10 或更高版本。TUI 使用标准库 curses，无额外运行时依赖。
