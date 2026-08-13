# Teaching Console production runtime 与 macOS 交付合同

`打开题目二教学Agent.command` 不再运行 `next dev`。启动器只接受由
`scripts/build_teacher_agent_console_runtime.mjs` 生成的不可变 production artifact：
artifact manifest 绑定当前 `package-lock.json`、源文件指纹、Node 最低版本以及每个运行文件的
SHA-256。构建写入独立的 `.next-production-<source hash>`，完成后复制到版本化 release 目录；
在线实例只读版本化目录，因此 build 与运行实例不共享 `.next`，构建锁也禁止两个 build 并发。

启动先检查 `/health` 的进程存活，再检查 `/ready` 是否能以 200 JSON 访问真实 Python
bootstrap。302、401、404 和 503 都不算 ready。单实例锁、进程组身份校验、PID/PGID 复用防护和
orphan recovery 继续生效。

## 生命周期

- 双击 `打开题目二教学Agent.command` 默认后台启动，并在 `/ready` 成功后打开一次本机 Console 页面；关闭
  Terminal 窗口不停止 Console。设置 `TEACHLAB_OPEN_BROWSER=0` 可只启动并打印 URL；直接调用底层 Node
  启动器时仍需显式设置 `TEACHLAB_OPEN_BROWSER=1` 才打开。
- 首次锁文件安装与隔离 production build 的等待上限默认 10 分钟；可用
  `TEACHLAB_DAEMON_STARTUP_TIMEOUT_MS` 在 1–30 分钟内调整。超时会回收本次新建进程组，不会让一个
  未报告的启动器继续在后台构建。
- `./打开题目二教学Agent.command --stop` 写入绑定本次 launch token 的 0600 stop request；
  supervisor 回收后端和 Console 进程组后才释放锁。
- 本地 UI 可以在取得 HttpOnly session 与 CSRF token 后 POST
  `/api/teacher-agent/runtime/stop`，返回 202 表示 stop request 已持久写入，不表示进程已全部退出。
  Console 右下角“停止后台服务”按钮使用这一路径，不会用“停止当前回答”冒充关闭应用服务。
- 默认 `TEACHLAB_IDLE_SHUTDOWN_MINUTES=0`，即后台持续驻留。设置正数后，教学 API 最后活动超过
  指定分钟才自动停止；SSE 每个数据块都会续租活动时间，运行中的长回答不会被误当空闲，`/health`
  监控探针也不会延长 idle 时间。
- 日志在 `.private/logs/teacher-agent-console.log`，下一次启动时超过默认 10 MiB 会保留一个 `.1` 轮转文件；
  `TEACHLAB_LOG_MAX_BYTES` 可在 1–100 MiB 内调整。macOS `.command` 入口默认设置
  `TEACHLAB_OPEN_BROWSER=1`；`TEACHLAB_OPEN_BROWSER=0` 可显式关闭，避免重复点击创建新页面。

## 密钥

macOS 首选 Generic Password 类型的 Keychain 项，默认 account 为当前用户名：

| 用途 | 默认 service |
| --- | --- |
| DeepSeek API key | `TeachLab DeepSeek API Key` |
| 学习记录密钥 | `TeachLab Learner Key` |
| 同意收据签名密钥 | `TeachLab Consent Signing Key` |

可用 `TEACHLAB_*_KEYCHAIN_SERVICE` 与 `TEACHLAB_KEYCHAIN_ACCOUNT` 改名。Keychain 值只物化为本次
launch 专用的 0600 临时文件（默认位于 `.private/runtime-secrets/`），退出时删除。没有 Keychain 项时，DeepSeek API key 允许一个单层符号链接，
但链接所在目录和目标父目录必须是 mode-0700 私有目录，目标必须是 1–4096 字节、owner-only 的普通文件；
启动器使用 `O_NOFOLLOW`、inode/device/size 复核后才复制到本次运行的临时文件。学习记录与同意签名密钥仍拒绝
符号链接，只接受本机 0600 普通文件（或首次生成的等价 fallback）；组/其他用户可读文件和超大文件均拒绝。
API key 不会伪造，必须由用户提供。

## .app、DMG、签名与升级

运行 `node scripts/package_teacher_agent_macos.mjs` 生成带 production Console runtime 的
`.app`、压缩 DMG 和 `release-manifest.json`。默认产物明确命名和标记为
`unsigned_developer_artifact`，内含 Console CycloneDX SBOM、项目 license、Python 依赖声明以及 wheel
公开清单中的运行时 data/schema；它仍依赖外部 Node 22+ 与 Python 3.10+，**不是可分发正式安装包**。

`TEACHLAB_CODESIGN_IDENTITY` 可启用 hardened-runtime app codesign；同时设置
`TEACHLAB_NOTARY_PROFILE` 可用 `notarytool` 提交并 staple DMG。release manifest 分别记录
`app_codesigned` 与 `dmg_notarized`，不会用一个含糊的 signed 字段混淆两者。

正式自包含路径要求同时设置 `TEACHLAB_DISTRIBUTABLE=1`、签名身份、公证 profile，以及
`TEACHLAB_EMBEDDED_NODE` / `TEACHLAB_EMBEDDED_PYTHON` 两个绝对运行时目录。每个目录必须包含
`embedded-runtime.json`，其 schema 为 `teachlab.embedded-runtime.v1`，声明 kind、精确三段版本、darwin
架构、相对 executable、`self_contained=true`、许可证文件、HTTPS 来源与源归档 SHA，以及排除 manifest
自身后的 canonical tree SHA。发布作业还必须通过
`TEACHLAB_EMBEDDED_{NODE,PYTHON}_MANIFEST_SHA256` 从作业配置独立钉住两个 manifest；只把 manifest 和目录
一起交给脚本不构成供应链校验。脚本会在复制前后复算整棵树，拒绝逃出根目录的符号链接、特殊文件、版本
不一致、缺许可证或 Python 缺少 `cryptography` 的运行时。只有嵌入运行时通过、app codesign 成功、DMG
notary+staple 成功时，release manifest 才能写
`distributable=true / signed_notarized_self_contained_release`；任一步缺失或失败都不会生成正式声明。该收据证明
打包与 Apple 公证链，不替代发布方对上游 runtime provenance、许可证或恶意依赖的人工审计。

channel manifest 同时保存 `current` 与 `previous` release id。升级只能在新 artifact 哈希验证后切换
current；回滚只能在 previous 的完整 runtime manifest 再验证后执行，不能通过覆盖在线目录完成。

## 跨浏览器生产态 smoke

CI 会先在独立目录构建并封存 Next standalone production runtime，再由
`scripts/run_teacher_agent_console_cross_browser_smoke.py` 启动随机端口的本机 fixture backend 和该
production server。Chromium、Firefox、WebKit 都会验证根页、严格 `/health`/`/ready`、浏览器真实
Fetch Metadata 下的 HttpOnly/SameSite 本地 session、bootstrap 与 Chat/Teach 控件。每个浏览器都会把
真实 `axe-core` 注入当前 production 文档，并在 workspace、命令错误 dialog、390px Inspector/Consent
dialog、离线只读壳四个状态执行 WCAG 2 A/AA 审计；任一 serious/critical violation 都会固定码失败，
不是扫描源码 token。runner 还实际键盘操作 project menu、Radix dialog、移动 Inspector、Consent、
永久删除原生 prompt 的取消与 focus restore，检查错误提示不抢焦点，并在 390px、等效 200%（640px）
和等效 400%（320px）验证无水平 document overflow。

同一个 browser job 会等待 production Service Worker 成为 controller，核验 Cache API 中只有同源
immutable 静态资源，再写入一份 account-scope-bound 终态 fixture，切断浏览器网络并执行 hard reload。
只有离线壳能显示该只读快照且通过 axe/focus 顺序才算通过；HTML、API 或私有响应出现在 cache 会直接
失败。runner 不使用 3030、不启动 `next dev`、不打开可见窗口，并会回收自己创建的进程组。

这个 job 验证的是 Linux CI 上的 Playwright Web runtime；axe 与 reflow 结果也不替代人工可用性测试。
Firefox/WebKit 结果不能替代 macOS
`.app`/DMG、codesign、notarization 或原生窗口生命周期验收；后四项仍以 macOS packaging manifest
及真机验收为准，不能把跨浏览器通过描述成 macOS 可分发认证。
