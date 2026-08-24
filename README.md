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
harness exec --attach NOTES.md "核对这份说明" # 本轮不可变文本附件
harness sessions
harness agents                    # 列出保留的子任务 worktree（默认不显示路径）
harness agents WORKTREE_ID --path # 明确查看一个保留 worktree 的本机路径
harness resume [SESSION_ID]
harness effects                  # 查看当前 workspace 的未决 run
harness reconcile RUN_ID         # 人工检查后确认并清除 fence
harness instructions             # 查看本轮会加载的项目指令摘要
harness context SESSION_ID       # 查看活动上下文预算和摘要血缘（不输出正文）
harness compact SESSION_ID       # 手动压缩旧轮次，保留完整原始 transcript
harness hooks                    # 查看项目 hook 的摘要、digest 与信任状态
harness mcp                      # 查看本地 stdio MCP、exact digest 与 catalog 状态
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
- 工作区工具：文件列表、读取、全文检索、统一补丁、沙箱命令和显式主机命令。
- 三档真实工具授权：`read-only`、`workspace-write`、`full-access`。
- 敏感工具逐调用审批：默认 low 自动允许、medium/high 询问；TUI 可允许一次，或把
  精确的工具版本+参数摘要保存为 session/workspace 规则。headless 无审批者时 handoff。
- `AGENTS.override.md` / `AGENTS.md` 安全发现、32 KiB 快照、摘要审计和
  `/instructions` 诊断；项目指令只影响模型上下文，不能扩大权限或跳过审批。
- 可信 command hooks：同步支持 `PreToolUse`、`PostToolUse` 和
  `PostToolUseFailure`。项目定义必须经过 exact digest 授信；pre hook 只能要求审批或拒绝，
  post hook 只能观察，任何 hook 都不能授予权限或改写工具参数。
- exact-digest trusted local MCP tools client 子集：固定协议基线 `2025-06-18`，只支持
  stdio、显式 catalog refresh 和冻结工具面；MCP 工具只在 `full-access` 中出现，按 high-risk、
  once-only approval、never-replay 执行。
- 稳定 message ID、append-only 原始 transcript、可验证摘要血缘和 active-context view；
  `/compact` 手动压缩，接近输入预算时自动分段压缩，`/context` 只显示计数与 digest。
- provider-neutral 附件快照：严格 UTF-8 文本/Markdown、校验后的 PNG/JPEG 和
  有界 PDF 会导入 owner-private 不可变 blob；会话中的附件 manifest 仅保留本地
  消毒 basename、类型、大小和 digest，不复制源路径、原始内容或 base64。
- 每次 run 使用 0600 hash-chain journal 和原子 checkpoint。任一 session 留下未完成或
  待核对 run 时，整个 workspace 的所有新 run 都会被 fence 阻断；`effects` 只列出证据，
  `reconcile` 只记录操作员已检查并清除 fence，不会验证、回滚或重放外部效果。
- workspace-scoped 本地 session：list、resume、fork、archive。用户消息与 unresolved run、
  assistant 消息与 run 终态分别原子落盘；fork 复制 transcript、摘要血缘和 session 元数据，
  不创建新进程、容器或 worktree，也不隔离环境变量、provider client 或当前进程堆中的秘密。
- curses TUI：流式正文、工具状态、Ctrl+C 取消、后续输入队列、token/cache 状态栏。
- 单次高风险审批后的前台 `agent.delegate`：1–4 个独立上下文并发执行、等待全部完成，
  每个子任务使用从当前 clean committed HEAD 创建的独立 Git worktree；TUI 显示有界状态，
  保留产物可用 `/agents` 或 `harness agents` 检查。
- Headless JSONL：稳定事件 schema、session/run/turn identity 和退出码。

与 Claude Code、Codex 的逐项对表见
[docs/harness_parity_matrix.md](docs/harness_parity_matrix.md)。

## 权限模型

| 模式 | 文件读/检索 | 应用补丁 | 命令工具 | 前台子任务 | 本地 MCP tools |
|---|---:|---:|---|---:|---:|
| `read-only` | 是 | 否 | 无 | 只读子任务 | 否 |
| `workspace-write` | 是 | macOS Seatbelt 可用时是 | `process.exec`：macOS Seatbelt 可用时注册 | 只读或 worktree 写子任务 | 否 |
| `full-access` | 是 | macOS Seatbelt 可用时是 | 沙箱 `process.exec`（可用时）和显式主机级 `process.exec_host` | 只读或 worktree 写子任务 | exact trust + frozen catalog + Seatbelt 可用时是 |

内建 list/read/search 工具拒绝绝对路径、`..`、符号链接和 `.git`、`.private`、
`.agent-harness`。在 macOS 上，`workspace.patch` 的检查与实际应用以及 `process.exec`
都由同一类 Seatbelt 策略约束：写入仅允许到工作区、本次调用的私有临时目录和 `/dev`；
对已知用户数据根的工作区外读取、网络、沙箱进程发出的 signal，以及已知
Keychain/securityd Mach 服务查找会被拒绝。补丁目标即使在预检后被替换成指向工作区外
的符号链接，内核写策略仍会拒绝解析后的外部目标。其他平台不注册这两个写工具，而不是
降级成未隔离的实现。

这是基于 `(allow default)` 再叠加拒绝规则的 macOS 主机策略，不是容器、VM、完整主机读取
隔离或跨 macOS 版本的凭据服务完备名单。状态中的 `user_data_read_policy` 和
`keychain_ipc_policy` 分别明确为“已知数据根”和“已知 security Mach 服务”，不代表阻断
所有可能的主机 IPC、文件读取、进程观察或内核侧信道。

`process.exec_host` 只在 `full-access` 出现，不受上述工作区沙箱约束。它可以按当前 OS
用户权限读取工作区秘密、工作区外文件、访问网络，并把内容写入有界工具输出；该输出
可能作为下一轮模型 observation 发送给 provider。环境变量白名单不保护文件、子进程
继承状态或已在进程堆中的秘密。

超时和取消会尽力回收初始进程组。daemon 化进程可能逃离这套回收机制；沙箱命令即使
存活仍继承 Seatbelt 限制，但 `process.exec_host` 的逃逸进程继续拥有当前 OS 用户权限。
选择 `full-access` 即表示主机级工具可被模型选择。

工具可见性、数据 scope、审批与 OS 隔离是同时生效的独立边界。`workspace.patch`、
`process.exec` 和 `process.exec_host` 在越过 `tool.effect_started` 前必须先得到审批；拒绝
或无人审批不会创建进程或应用补丁。TUI 审批键为：`y` 仅本次、`s` 保存当前 session
的精确规则、`w` 保存当前 workspace 的精确规则、`n` 拒绝。规则保存在 workspace 对应
的 0600 私有状态目录，只含参数 SHA-256；`/approvals clear session|workspace` 可撤销。

`data_scope` 只是工具注册与授权时使用的元数据标签，用于决定工具是否可见、可执行；它
不是文件系统、网络、进程或内存隔离机制。实际边界来自内建 handler 的路径检查、macOS
Seatbelt 或显式授予主机进程的权限。

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

## 附件

Headless 可重复传入 `--attach`；TUI 使用 `/attach PATH` 暂存本轮快照，
`/attachments` 检查待发列表，`/detach ID|all` 只移除尚未绑定到待发 turn 的项目。
普通 prompt 提交时，Harness 会把“此 prompt + 当时暂存的附件”一起冻结，不会
让忙时 follow-up 的附件串到下一个 prompt。当前 curses TUI 没有图形拖拽、粘贴
或剪贴板导入，不声称这部分与 Claude Code/Codex 交互对齐。

导入不相信扩展名：文本必须是严格 UTF-8，PNG/JPEG 会校验实际签名、结构和
尺寸；PDF 只校验 `%PDF-`/`%%EOF` 包络与大小边界，不解析或提取内容。源文件使用
no-follow 路径遍历和稳定 inode/device/size 读取，然后原子发布到当前
workspace 对应的 0700 私有附件目录，blob 为 0600。
每轮最多导入 8 个、原始字节合计最多 24 MiB；单个文本、图片、PDF 上限分别
为 2、8、16 MiB。当前 DeepSeek provider 的活动请求另有 16 个/24 MiB 上限。

当前 DeepSeek 能力边界是：

- `deepseek-v4-flash` 等普通文本模型可接收严格 UTF-8 文本/Markdown 附件。
- 只有操作员显式设置 `HARNESS_DEEPSEEK_MODEL=deepseek-v4-flash-vision-exp` 时，才会发送
  PNG/JPEG；Harness 不会为图片自动换模型。图片在当次 provider 请求内联，不调用
  DeepSeek Files API。
- PDF 可安全导入、快照并用统一 descriptor 表示，但当前 DeepSeek adapter 不支持
  PDF；它会在建立 run 或发出网络请求前拒绝，不伪造提取结果。

文档、文本和图片内容始终是不可信的用户数据。其中的指令不是 system/project
指令，不能扩大工具权限、数据 scope 或审批范围。使用附件表示用户明确同意把
该 turn 所需的文本或内联图片字节发送给选定的远程 provider。

## 项目指令

每个 turn 开始时，Harness 在显式 workspace 范围内每级最多选择一个文件：
`AGENTS.override.md` 优先于 `AGENTS.md`，然后按根到当前目录顺序合并。读取拒绝符号链接、
非普通文件、非 UTF-8、读取期间替换以及超过 32 KiB 的组合；失败时整轮 fail closed，
不会静默截断或回退到较低优先级文件。当前不加载 `CLAUDE.md` 或领域教学 Skill；可执行
项目配置只通过下述独立、显式授信的 hook 边界进入。`harness instructions --json` 和 TUI
`/instructions` 默认只显示路径、大小和摘要，不打印指令正文。

默认 active directory 就是 workspace 根；需要对子目录应用更近的指令时显式传入，例如：

```bash
harness --cwd . --active-directory packages/api instructions
```

## 可信 command hooks

项目可以在 `.agent-harness/hooks.json` 声明同步 command hooks，entrypoint 必须位于
`.agent-harness/hooks/`。当前只实现工具调用的三个事件：

- `PreToolUse`：可返回 `pass`、`ask` 或 `deny`；多个结果按 `deny > ask > pass` 聚合。
  `pass` 只表示“不再收紧”，不会覆盖中央 permission、scope、approval 或 sandbox 决策。
  `ask` 强制当前调用重新审批，既有 session/workspace `allow` 规则不能绕过它；该审批只允许
  本次 `y/n`，不能用 `s/w` 保存持久放行。
- `PostToolUse`：工具成功后同步观察，必须返回 `pass`。
- `PostToolUseFailure`：最终一次工具失败后同步观察，必须返回 `pass`。

最小配置示例：

```json
{
  "schema": "agent_harness.hooks.v1",
  "hooks": {
    "PreToolUse": [
      {
        "id": "protect-generated",
        "matcher": ["workspace.patch"],
        "entrypoint": ".agent-harness/hooks/protect-generated",
        "args": [],
        "timeout_seconds": 5
      }
    ],
    "PostToolUse": [],
    "PostToolUseFailure": []
  }
}
```

成功退出时 stdout 必须是一个严格 JSON 对象，例如
`{"schema":"agent_harness.hook_output.v1","decision":"pass"}`；pre hook 也可用退出码
`2` 直接拒绝。其他非零退出、超时、超限或格式错误按 hook failure 处理。

这只是 Claude Code/Codex hooks 的同步工具生命周期子集，不包含其他 run/session/model
生命周期、异步 hooks、参数改写或完整配置兼容。

仓库中的 hook 永远先视为不可信提案。Harness 会 no-follow 读取 owner-owned、非链接、
非 group/other-writable 的配置和可执行文件，快照 entrypoint 字节，并为事件、matcher、参数、
timeout、精确配置字节及 entrypoint 字节生成 definition SHA-256。信任状态保存在工作区之外
的 0600 私有状态目录；任一被绑定内容变化都会使状态变成 `modified`，并在新 run 开始前
fail closed。操作入口为：

```bash
harness hooks [--json]
harness hooks trust HOOK_ID --sha256 DIGEST
harness hooks disable HOOK_ID --sha256 DIGEST
harness hooks revoke HOOK_ID
```

`trust` 和 `disable` 都必须带当前展示的 exact digest，避免在确认与写入之间接受另一版本。
TUI `/hooks` 只显示 content-free 状态，不提供授信操作。

每次执行使用已快照的 entrypoint 字节，并进入专用的 macOS Seatbelt profile：工作区只读、
网络和 signal 被拒绝、禁止 fork 子进程、写入仅限本次私有 runtime 目录和 `/dev`，同时沿用
已知用户数据根及 Keychain/securityd Mach 服务的拒绝边界，并拒绝读取工作区内的 `.git`、
`.private` 和 `.agent-harness`。没有跨平台或 unsandboxed 回退；只要存在 trusted hook 而
sandbox 不可用，新 run 就会在 provider 请求之前 fail closed。可以用 exact-digest
`disable` 明确禁用不应执行的定义。

hook stdin 会临时包含当前工具的原始参数；成功后的 post hook 还会收到有界工具结果，
因此只应授信已审查的本地程序。事件和 journal 仅记录 hook policy/definition/input/output
digest、身份、时长、动作和安全错误码，不写入原始 hook stdin、stdout 或 stderr。底层工具
自己的标准事件仍遵循其原有持久化契约。

## 本地 MCP stdio tools client 子集

Harness 自 2.4.0 起实现 **exact-digest trusted local stdio MCP tools client subset**，协议
基线固定为 `2025-06-18`，不是完整 MCP 平台。项目定义位于 `.agent-harness/mcp.json`；
配置存在不会自动启动 server。最小示例：

```json
{
  "schema": "agent_harness.mcp.v1",
  "servers": {
    "local_tools": {
      "transport": "stdio",
      "command": "/absolute/path/to/mcp-server",
      "args": [],
      "cwd": ".",
      "pass_env": [],
      "network_access": false,
      "allow_process_fork": false,
      "startup_timeout_seconds": 10,
      "tool_timeout_seconds": 30
    }
  }
}
```

启用流程是显式的两阶段确认：

```bash
harness mcp [--json]
harness mcp trust SERVER_ID --sha256 DIGEST
harness mcp refresh SERVER_ID
harness mcp disable SERVER_ID --sha256 DIGEST
harness mcp revoke SERVER_ID
```

`harness mcp` 默认显示完整 64 位 definition digest、精确 argv、cwd、传入的环境变量名称、
network/fork 标志和 catalog 状态；`trust`/`disable` 必须回传完整 digest。`refresh` 只会启动
已精确授信的 server，协商 `2025-06-18`，分页读取 `tools/list`，过滤超限或不受支持的工具
schema，并把 catalog 摘要和规范化工具定义冻结到工作区之外的 0600 私有状态。每次工具调用
会重新握手并核对 live catalog；`notifications/tools/list_changed` 或 digest 变化都会拒绝调用，
要求显式 refresh，不会在活动 run 中动态接受新工具。TUI `/mcp` 只读，不提供 trust 或
refresh。

当前客户端协议面只包括 initialize/initialized、分页 `tools/list`、`tools/call`、取消通知、
响应 server `ping`，以及把 tools-list-changed 标为 stale。输入/输出 schema 只接受 Harness
能够本地验证的 object-root 关键字子集；text 和 object-shaped `structuredContent` 可进入规范化
结果，图片、音频、resource 等非文本内容只保留 type、长度、MIME 和 SHA-256 摘要，不做二进制
渲染。`isError` 是已完成的远端工具结果，不被伪装成 transport failure。

明确不支持：HTTP transport、OAuth、resources、prompts、sampling、elicitation、tasks、
input-required/task result、活动 run 动态 catalog、完整 JSON Schema、二进制渲染，以及把
Harness 作为 MCP server。未实现的方法不会被静默代理；server 发起的非 `ping` request 返回
method-not-supported。

MCP server 定义的 exact digest 绑定精确配置字节、server ID、直接 executable 的内容与文件
身份、argv、cwd、允许传入的环境变量名称、network/fork 标志、timeout 和 sandbox-policy
版本。它**不是传递依赖完整性证明**：不会覆盖解释器参数所指脚本、动态库、导入包、运行时
配置、环境变量值或 server 后续读取的其他文件。调用前只会再次核验直接 executable；operator
必须自行审查并固定其完整依赖链。

为封闭“核验后、exec 前”的路径替换窗口，user-owned direct executable 会在每次连接中按已
核验字节物化到本次 0700 私有 runtime 的 0500 副本，并执行该副本。因此这类程序看到的
`argv[0]`，以及脚本常见的 `__file__`，会指向临时 runtime；依赖 executable 所在目录查找资源
的 server 必须显式适配。完整 canonical ancestry 均为 root-owned、mode/ACL 对当前用户不可写的
macOS system executable 保留原平台路径，因为复制后的 platform binary 可能无法执行。

每个 MCP 进程都通过 macOS Seatbelt 启动，工作区只读，`.git`、`.private`、
`.agent-harness` 不可读，写入限于私有 runtime HOME/TMP 和 `/dev`；network 与 fork 默认
拒绝，只有 exact-digest-bound 的 `network_access` / `allow_process_fork` 才能打开。没有
unsandboxed 或 unsupported-host 回退。这个 Seatbelt 仍是 allow-default 的主机策略，不是容器、
VM 或完整机密边界；允许 fork 后，setsid/double-fork 的后代可能逃离 Harness 的进程组回收，
并继续持有该定义授予的 Seatbelt/network 权限。

`pass_env` 仍需逐项显式声明；provider 凭据、Harness/sandbox 保留项，以及 `DYLD_*`、`LD_*`、
`PYTHONPATH`、`NODE_OPTIONS` 等常见 loader/runtime code-loading 控制均拒绝传入。获准的普通业务
变量值会披露给本地 server，且值本身不受 definition digest 绑定。

冻结的 MCP tools 只在 `full-access` 工具面注册，统一标为 external-service、high-risk、
never-replay，并要求一次性审批；已有或新建的 session/workspace persistent allow 不能绕过。
MCP 原始输入和规范化结果遵循普通工具的 owner-only journal/checkpoint 合同，结果也可能成为
下一轮 provider observation。server stderr 原文只在进程运行期间被持续 drain，不写入 journal、
checkpoint 或 catalog；CLI 最多返回 byte count、truncated 标志和 SHA-256。

## 前台子任务与隔离 worktree

模型可通过中央 `agent.delegate` 工具提交 1–4 个有界任务。Harness 只实现
**foreground wait-all**：子任务真实并发，但父工具等待每个已启动子任务结束；父取消会
级联，所有子任务在父调用结算前都必须 join。返回给父模型的是有界最终摘要、状态、任务顺序/
层级、是否改动、失败码（如有）和不透明关联 ID，不包含子任务推理过程。子任务摘要与普通工具
observation 一样是不可信证据，其中嵌入的指令不能扩大父任务或权限。

委派是 high-risk、never-replay、不可保存持久放行的一次性审批。审批预览显示整批任务及其
请求的 `read-only` 或 `workspace-write` 模式。只读父任务不能委派写任务；父任务即使是
`full-access`，子任务也最多得到 `workspace-write`。子 runner 不向模型暴露或授权 host
command、MCP、project hooks 或再次委派；受批次审批约束的 `workspace.patch` / 沙箱 `process.exec` 可以在
子任务内执行，但不会建立 session/workspace persistent approval。

每个子任务从当前仓库精确的 committed `HEAD` 建立独立、不透明分支和 Git worktree。源仓库
必须 clean（包括没有非 ignored 的 untracked 文件）；Harness 不会把父工作区未提交内容
猜测性复制进去，ignored 本机文件也不会被复制。
Git 通过受信绝对路径和清理后的环境直接执行，禁用项目 hooks、includes、filters、fsmonitor
与 external diff。worktree 隔离文件写冲突，但**不是**进程、内存、provider credential、
主机读取或网络隔离证明；真正的写边界仍由子 runner 的 permission 与 macOS Seatbelt 提供。

只有 Git status、分支 baseline 和 no-follow 内容 manifest 都精确不变时，Harness 才会用普通
非 force Git 操作自动移除 worktree。存在改动、额外 commit、结构异常或创建/清理不确定性时，
Harness 会保留仍存在的 worktree/分支及相应记录供人工检查；若 checkout 已正常移除、仅 ref 的
compare-and-swap 删除失败，则保留的是分支和 `ref_preserved` 记录。不会自动 merge、apply、
commit、push、reset、clean、prune 或强制删除。默认列表不暴露本机路径：

```bash
harness agents
harness agents WORKTREE_ID
harness agents WORKTREE_ID --path  # 只有显式指定一个 ID 才显示路径
```

默认 worktree 根为 macOS 的 `~/Library/Caches/AgentHarnessWorktrees`，其他平台为
`~/.cache/agent-harness-worktrees`；可在子命令前用 `--worktree-home ABSOLUTE_PATH` 或环境变量
`AGENT_HARNESS_WORKTREE_HOME` 设置私有目录。当前没有后台运行、恢复/steer 子任务、agent
thread 切换、自定义 agent profile、自动合并或 agent team；这不是 Claude Code/Codex 的完整
subagent parity。

## 上下文压缩

Harness 始终保留完整原始消息；压缩只生成一个单独、append-only 的有损摘要，并让后续模型
看到“最新摘要 + 未覆盖消息后缀 + 当前用户消息”。每条消息有稳定 ID 和内容摘要；每个压缩
记录绑定连续原文前缀、父摘要、provider/model 和 prompt 版本。hash 用于发现私有状态损坏
和血缘不一致，不是抵抗拥有状态目录写权限者的签名。
压缩区间不能跨过含附件的消息；该消息及其后缀会留在 active view，所以含附件的长会话
可能在不能继续安全压缩时直接达到 context 上限。

自动压缩在“摘要、活动消息、项目指令、工具定义和待发送 prompt”的保守 UTF-8 byte 上界
达到可用输入预算 80% 时触发，目标降到约 60%，并保留最近 6 条原始消息。单次只总结有界
增量并停在 assistant 边界，最多执行 8 段，避免把超大历史一次发送给 provider 或无限
重试。单条巨大 prompt、近期后缀或工具 observation 仍可能触发硬 context limit；压缩不会
解除 2,000 条消息和 64 MiB 私有 session 存储上限。

摘要由当前 provider 生成，可能遗漏或误述，因此永远标为历史用户数据，而不是 system
指令，也不能扩大工具、权限、scope 或跳过审批。压缩沿用远端内容许可；取消发生在提交前
时不会写入压缩记录。`harness context SESSION_ID --json` 与 TUI `/context` 不输出 transcript
或摘要正文。

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
/instructions
/hooks
/mcp
/agents
/context
/compact
/attach PATH
/attachments
/detach ID|all
/approvals
/approvals clear session|workspace
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
Generic Session Store ─── Immutable Attachment Store
        │
Harness Runtime ── Event reducer / Journal / Checkpoint
   │          │
Provider ← capability preflight   Tool Registry ─┬─ Trusted Hook Broker
                         ├─ Frozen MCP Catalog / stdio Client
                         └─ Foreground Subagent Scheduler
                                  │
                       isolated Git worktrees
                                  │
                path checks / Seatbelt / explicit host shell
```

包结构：

- `agent_harness/core/`：domain-neutral runtime、events、tools、journal、recovery。
- `agent_harness/providers/`：provider client 与 tool-aware adapter。
- `agent_harness/attachments.py`：no-follow 导入、不可变私有 blob 和 descriptor 合同。
- `agent_harness/toolsets/`：coding workspace 工具。
- `agent_harness/session.py`：本地多轮 session。
- `agent_harness/context.py`：active-context 预算、分段计划和 provider 摘要契约。
- `agent_harness/hooks.py`：项目 hook 的安全发现、exact-digest 信任绑定与沙箱执行。
- `agent_harness/mcp.py`：本地 MCP 定义、exact-digest trust、冻结 catalog 与 stdio bridge。
- `agent_harness/core/mcp_protocol.py`：固定 `2025-06-18` 的严格有界 MCP tools 协议子集。
- `agent_harness/subagents.py`：前台批处理、并发/深度预算、取消传播与结果契约。
- `agent_harness/worktrees.py`：clean HEAD worktree 创建、持久记录和保守清理。
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
