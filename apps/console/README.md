# TeachLab Console

现代化前端工作台，视觉和交互基准是用户提供的三栏参考图：左侧会话导航、中间学习者对话与输入、右侧可折叠检查器。内部工具、路由和 usage 生命周期保留在 reducer/检查器边界，不作为对话卡片展示。现在已通过 Next 服务端同源代理接入现有 `teaching_skill_miner` Python Teaching Agent，并由项目根目录启动脚本作为默认入口。

## 当前技术栈

- Next.js 15.5.23 App Router、React 19、TypeScript
- Tailwind CSS 4 与本地 shadcn/ui 风格组件
- TanStack React Query
- 轻量只读 Runtime/学生画像/资源检查器；未使用的 Monaco/xterm 组件与依赖已经移除
- Python Harness typed SSE 客户端与严格 reducer：`lib/api.ts`、`lib/harness-stream.ts`

## 本地运行

```bash
cd apps/console
cp .env.example .env.local
npm ci
npm run dev
```

`npm run dev` 只用于开发，需要设置服务端变量 `TEACHER_AGENT_CAPABILITY_URL`。交付路径使用
`npm run build:runtime`：它从锁文件在隔离工作区执行 `npm ci`，写入独立 distDir，并产出逐文件哈希
绑定的 standalone production runtime。推荐使用项目根目录的 `打开题目二教学Agent.command`：它会验证
runtime、启动 DeepSeek Python 后端、注入 capability URL，并在后台驻留；关闭 Terminal 不会停止服务，
用 `./打开题目二教学Agent.command --stop` 或 Console 的“停止后台服务”按钮关闭。macOS 双击入口默认在
`/ready` 成功后自动打开一次浏览器页面；设置 `TEACHLAB_OPEN_BROWSER=0` 可只启动并打印 URL。直接调用
底层 Node 启动器时仍需显式设置 `TEACHLAB_OPEN_BROWSER=1` 才打开。浏览器只请求同源
`/api/teacher-agent/*`，不会拿到 capability token。完整合同见
`../../docs/teacher_agent_console_production_runtime.md`。

本地 API 也不把“只监听 loopback”当作浏览器安全边界。Next 只接受精确的 `localhost`、`127.0.0.1` 或 `[::1]` Host，要求 `Sec-Fetch-Site: same-origin`，并对所有 mutation 校验精确 Origin、`application/json`、SameSite=Strict HttpOnly 本地会话和 server-minted CSRF token。客户端先通过 `/api/teacher-agent/security/session` 完成同源握手；过期后只在 Next 明确返回本地 session/CSRF rejection 时自动换新一次。`X-Forwarded-*` 不参与信任判断。推荐启动器会注入每次启动独立的 256-bit 签名密钥；直接运行多个 Next worker 时，应显式把同一 64 位十六进制 `TEACHLAB_LOCAL_SECURITY_SECRET` 交给这些 worker。该本地会话只防网页跨站访问，不是用户身份或多租户授权。

`TEACHLAB_HARNESS_MODE` 必须显式选择 `local_python` 或 `authenticated_apps_api`，生产不会在两者之间静默降级。团队部署使用后者，并配置 server-only 的 `TEACHLAB_APPS_API_URL`、`TEACHLAB_CONSOLE_ORIGIN` 与 `TEACHLAB_APPS_API_COOKIE_MODE=secure`，构建时显式设置 `NEXT_PUBLIC_TEACHLAB_AUTH_MODE=oidc`。浏览器从同源 `/api/teacher-agent/security/login` 进入 Authorization Code + PKCE；Next 只转发经过白名单验证的跳转、交易 Cookie 与最终 session/CSRF Cookie，从不接收或处理 access/id token。随后 Python 路由映射到 `/api/v1/harness/*`。本地安全 Cookie、任意其他 Cookie、`x-tenant`/`x-user`/principal override 都不会进入 Harness 请求。客户端只在模块内存保存 CSRF token，并始终使用 `credentials: include`；刷新后通过同源 GET 从已验证会话恢复，未认证 bootstrap 以 `401 authentication_required` 进入组织登录页。

已退出页和首次未登录页都显示“使用组织账号登录”；本地 `local_python` 模式不显示该入口。侧栏“安全退出当前设备”调用 `clearAuthenticatedHarnessSession()`：UI 会先分离本地流，并给已登记 run 一个 1.5 秒的有界 durable-cancel 窗口，然后同源 DELETE；只有 apps/api 已持久撤销且返回两条精确清除 Cookie 后，模块内存 CSRF、查询缓存和当前工作区才会清空并进入已退出状态。撤销存储失败会保留 Cookie/内存身份并显示错误，不会依赖 `localStorage` 假装退出。apps/api 必须把 Console 精确 Origin 纳入 `CORS_ORIGINS`；BFF 不接受浏览器提交 tenant、owner 或 principal JSON 来选择数据作用域。

证据复核 UI 也以 bootstrap fail-closed：只有 `authenticated_apps_api` 同时明确返回 role authorized、Correct mastery enabled、hash-only identity 和 deployment-service assurance 时才显示“认证教师裁决”。浏览器只提交业务字段及稳定 idempotency key；actor、role、tenant、principal 和 authority receipt 均由服务端生成。断网重试会沿用业务 key，但 apps/api 每次签发新的 one-time nonce，Python durable store 返回第一次封存结果。`local_python` 明确显示未认证边界，Correct 不改 mastery；认证模式缺角色或 bootstrap 字段不一致时领取/裁决按钮关闭。收据只证明部署服务的角色授权，不是个人签名。

检查器的“安全保障”面板只渲染 case/version/status、分类、严重度、时间和 SHA-256，可执行中央投递确认、人工接手与关闭。它要求机构系统签发的 AEAD opaque route locator；凭据仅驻留 React 组件内存，关闭检查器立即清除，不写 localStorage、IndexedDB 或离线快照。Next BFF 只白名单六条精确 POST route 并限制 16 KiB；apps/api 在触碰目标 learner worker 前实时查询目录中的 exact `safeguarding` role、解密 locator 并强制同租户。浏览器 Cookie roles、learner 标识、原文和 authority receipt 都不能选择作用域或进入 staff API。

## 推荐的完整架构

```text
Next.js Console
  └─ HttpOnly session / SSE
NestJS + Fastify BFF
  ├─ Clerk 或 Auth0（GitHub OAuth、企业 SAML/OIDC）
  ├─ GitHub App / Git proxy
  ├─ Temporal workflow API
  └─ PostgreSQL / Redis / S3
Temporal workers
  ├─ Claude tool-use + streaming
  ├─ ripgrep + tree-sitter + 可选 pgvector
  ├─ plan → edit → test → rollback
  └─ sandbox adapter
Sandbox plane
  ├─ MVP: Docker + gVisor
  └─ Production: EKS/Kubernetes + Firecracker
```

## 已接生产能力与剩余边界

| 层面 | 当前已实现 | 仍未实现或不声称 |
| --- | --- | --- |
| 身份与数据 | 严格 OIDC Authorization Code + PKCE、HttpOnly session/CSRF、PostgreSQL FORCE RLS、AAL2 账户数据权利和服务端权威教师/安全保障授权 | IdP 与组织目录由部署方运营；没有 GitHub App/Git proxy、Redis/S3 或跨副本授权缓存 |
| Agent 与流 | tenant/owner 私有 Python worker、durable typed SSE、取消、hash-chain journal、checkpoint/recovery 与 DeepSeek 教学链 | 没有 Claude tool-use、BullMQ/Temporal 或 plan→edit→test→rollback 通用编码 Agent；需要双向 steering 时才考虑 WebSocket |
| 进程与主机隔离 | Linux production worker 使用每 scope Landlock allowlist，并在启动/真实 canary 中强制每进程 `RLIMIT_CORE=0`、`NOFILE=256`、`FSIZE=512 MiB`、`AS=1.5 GiB`；parser 子进程继承限制，API 容器另有 4 CPU/4 GiB 与 256 PID 全局预算、只读根文件系统、请求墙钟/并发/体积门禁 | 所有 worker 仍共享 uid `10001`；per-worker RLIMIT 不是 process-tree/cgroup 总量，也没有独立 `RLIMIT_CPU/NPROC`、gVisor、独立 VM/uid、Kubernetes/Firecracker 或跨主机 egress allowlist |
| 可观测与部署 | 低基数 Prometheus、liveness/readiness、digest-pinned 单副本 Compose/Docker/Caddy 与 CI 部署验收 | 没有 Grafana/Loki/Sentry、Terraform/Helm/Argo CD、分布式 worker lease 或多 API replica；主机网络出口政策仍由部署方实施 |

生产浏览器只持有 HttpOnly 会话，不持有模型、数据库、scope、投递或保留密钥。当前生产拓扑的可信边界是单 API replica + Linux 容器，不是 VM 级敌对多租户沙箱。

## 目录

- `components/workbench/`：三栏 UI、可折叠右栏、Chat/Teach 对话 composer、项目/资源/大纲与只读 Runtime 检查器
- `components/ui/`：本地 shadcn 风格原子组件
- `lib/harness-stream.ts`：SSE frame、事件 envelope、严格 sequence/terminal/commit reducer
- `lib/api.ts`：Python 同源代理请求、cursor reconnect、显式取消与 UI delta 批处理边界
- `lib/offline-runtime.ts`：IndexedDB 草稿/outbox、任务恢复决策、SSE 退避与消息渲染预算
- `lib/local-request-security.ts`：loopback Host/Origin/Fetch Metadata、HttpOnly session 与 CSRF 边界
- `lib/auth-boundary.ts`：浏览器凭据禁区声明

页面首次加载会调用 `api/bootstrap` 并恢复 opaque session；Chat、教学 start 与 step 通过 Python Harness typed SSE 运行，`/auto`、`/+skill`、`/stop` 仍走受版本门禁保护的 command API。每个 Harness 事件先追加到 SHA-256 hash-chain journal 并越过 `fsync` durable ack，随后才发布；断线后客户端携带同一 `run_id + turn_id + after_sequence` 从最后接受的连续事件恢复。Next reducer 会拒绝 identity 变化、sequence 缺口/分叉、重复权威结果、commit/result 不一致、缺失 commit 的教学完成以及 terminal 后事件，不以字符切分伪造模型流。

DeepSeek 普通文本走 `/chat/completions` 原生 SSE；Web Search 默认关闭，用户显式开启并确认单独的远程检索授权后，才走 Anthropic Messages server-side tool 原生 SSE。两条路径都只把通过最终门禁的 assistant 文本 delta 暴露给对话；`channel=internal` 以及 reasoning、usage、tool-call delta 仍进入严格 reducer/运行状态，但不会生成学习者消息或活动卡。因而 teach-first 对话直接显示教学回答，不显示“来源证据卡”“选择教学路径”“设置下一学习重点”等内部政策/工具过程。Web Search 映射为要求 `requires_user_consent` 的 provider-managed tool，最终只保留有界、清洗后的来源标题与 URL，不发布搜索正文。Provider 小片段按约 40 ms 或 512 字符聚合并在边界/结束时完整 flush，以保持原始文本顺序和内容，同时避免每个 token 都触发 journal `fsync` 与 React 更新。

V18 最终动作规划器使用稳定 system/Skill 前缀和动态 turn 后缀来利用 DeepSeek 自动上下文缓存；这是 cache-aware 稳定前缀实现，不是正式集成 DeepSeek-Reasonix。前端只接受经边界校验的非负整数 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` usage；静态布局哈希不能证明命中，provider 没有返回字段时 UI/trace 不推算命中率。

停止生成调用独立 Harness cancel API；取消 token 会关闭已经建立的 provider 响应，并与 domain commit 使用同一个 commit-wins 门禁。正在执行的 run 可在同一 Python 进程内 cursor reconnect。持久 journal 下的 Teach 跨进程恢复只有三条精确路径：`start` 在 provider effect 前可安全恢复一次；start/step 的 provider effect 结果未知时不重放而 handoff；领域 turn 已 commit 时只补齐外层 SSE。学生 `step` 不会因重启重放，Chat 非终态 run 也不自动恢复。教学 Session 仍默认只在内存中；只有显式 `--session-store` 才启用已有的 cold resume，且该明文 JSONL 必须按私有学生数据保护。

浏览器离线时会显示明确的只读缓存状态，绝不声称可在本机运行模型；输入草稿仍写入 IndexedDB。production 会注册版本化 Service Worker，但 Cache API 只允许同源 `/_next/static/` 和版本化离线脚本，且响应必须声明 `immutable`；HTML、导航响应、API、bootstrap 和任何私有响应永不写入 Cache API。在线预热后断网硬刷新时，Service Worker 现场合成 `no-store` 只读壳；它只在 localStorage 当前 account cache scope 与 IndexedDB 快照绑定 scope 常量时间一致时显示内容，否则失败关闭。最近的终态项目快照最多 12 个、每个最多 5 MB、7 天到期，运行中/排队消息不允许进入缓存，旧 v1 未绑定快照不迁移；logout 清 IndexedDB 与 TeachLab 静态 cache，账户删除继续清整个 origin Cache API。它只用于离线重载展示，不能作为服务器 mutation 的权威输入。每个流式 mutation 在第一次发送前以原始 `request_id` 写入有 7 天期限的本机 outbox，拿到后台 `task_id + version` 后立刻升级为已注册状态并擦除 learner payload。恢复联网时，只有未注册项会复用原 `request_id` 提交；已注册项只调用 task status/resume，未知外部副作用进入显式 handoff，绝不盲重放 payload。SSE 最多使用 12 个连接、指数退避并在约 3 分钟的最小恢复窗口后交接。对话初始最多挂载最近 300 条消息，可按 300 条增量加载；流式正文用 40ms 批次纯文本预览，结束后再做完整 Markdown 解析，因此 10,000 条历史不会一次创建 10,000 个消息节点。

当前 composer 采用 Claude Code 风格的紧凑交互约定：`›` 提示符、Enter 发送、Shift+Enter 换行、`⌘/Ctrl+K` 聚焦 slash command、方向键选择、Tab 补全、Esc 停止当前生成；原生流式响应期间仍可继续编辑草稿。这里只对齐操作节奏和视觉密度，不声明实现或复制 Claude Code 的内部能力。

输入框左下角的“+”只展示当前 `api/bootstrap` 中 `teaching_resource_formats` 声明的格式，而不是静态承诺所有主机都能解析同一扩展名。TXT/MD/CSV/TSV/XLSX/DOCX/PPTX 使用内置解析；PDF 只在隔离 parser 与 `pdftotext` 可用时声明；legacy DOC/PPT/RTF、图片 OCR、音视频转写也分别要求本机转换器、OCR 引擎或显式注入的本地 temporal provider。默认 Linux 生产镜像没有 legacy Office、OCR 或 ASR 引擎，因而不会广告这些格式。原文件不保存、不发送给 DeepSeek，提取结果是教师上下文，不参与学生评分或掌握度更新。麦克风入口默认禁用，页面 Permissions Policy 也禁止采集音频；浏览器 `SpeechRecognition` 无法提供本服务可验证的政策收据，因此不调用。

Chat composer 的附件按钮、拖放和粘贴共用同一条本地资源管线，单回合最多 6 个、单个最多 12 MB；可选格式始终以服务端运行时能力投影为准。浏览器只在 `/api/resource` 上传时短暂持有 `File`；Chat stream 只提交项目绑定的 `resource_id + staged_resource_id`，重试上下文也只保留这些有界 ID 和展示元数据。后端在每次调用前重新校验项目归属、stage 配对、权威多模态 decision 与 `teacher_resource_excerpt` consent，仅把与当前问题相关的有界文字摘录作为不可信引用上下文发送；冲突、视觉待核验或语义弃权均在模型调用前阻断。SSE result/journal 只含响应哈希等元数据，不含附件全文、摘录或原始媒体。

默认启动器还配置私有学习项目 store 与教学资源索引。项目可保存 Chat thread、笔记及资源/教学大纲/Teach session 引用，支持置顶、归档、删除后 token 恢复；Chat 完整历史以服务端持久副本为权威。客户端可提交最多 400 条严格交替消息，只有旧前缀已经 durable 时服务端才生成带 transcript hash 的抽取式索引并缩小 provider 上下文，完整历史不会被模型摘要覆盖。教学资源索引保存本机提取文本、chunk offset、页码/讲者备注/视觉复核位置与内容哈希；第七个 Teacher tool `retrieve_resources` 只返回有界片段。项目、资源、大纲和检索结果都不是 learner evidence 或 scoring gold。

## 当前边界

- 旧 Python Dashboard 资源仍保留作为后端自带调试页；默认启动入口已切换到新 Console。会话列表、发送动作、会话版本门禁和右侧学情/运行检查器均读取真实 Teaching Agent 响应。
- Runtime 检查器显示真实 session/context/fallback 的只读快照；当前没有挂载 Monaco 编辑器或 xterm 终端。Python Harness 已提供 durable-first typed SSE、cursor reconnect、显式取消和 DeepSeek 文本/Web Search 原生传输层流式；Teach 的窄恢复边界不等于 WebSocket 双向 steering、任意跨进程在途执行恢复或生产队列已经实现。
- `authenticated_apps_api` 生产模式已使用 OIDC、HttpOnly 会话、PostgreSQL FORCE RLS、CSRF 与 per-account Python worker；`local_python` 仍只是 loopback 单机模式。浏览器中的 opaque 教学 handle 不是身份。当前生产边界仍是单 API replica、同 OS uid 的逻辑 worker scope（Landlock 限制路径但不等于独立容器/uid）；没有分布式 worker scheduler、Redis/S3/Kubernetes/Firecracker 或外部可观测平台。
