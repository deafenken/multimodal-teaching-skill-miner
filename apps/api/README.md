# TeachLab API / BFF

这是面向现代 Console 的 NestJS + Fastify 边界层。默认配置是明确的单机本地模式；生产配置则以 OIDC、服务端会话、租户授权和 PostgreSQL RLS 为强制启动条件，不会把 loopback 开发身份伪装成多用户登录。

## 已实现的身份与租户闭环

- `GET /api/v1/auth/oidc/begin|callback`：生产登录使用 Authorization Code + S256 PKCE。短期 `HttpOnly; Secure; SameSite=Lax` AES-GCM 交易 Cookie 绑定随机 state、nonce、verifier 与本地 return path；ID/access token 仅在 API 内存中出现。回调严格验证 RS256、issuer、client audience/azp、nonce、iat、auth_time 和可信 tenant claim，先持久登记撤销 authority，成功后才签发短期会话。
- `POST /api/v1/auth/session` 仅保留给明确的本地 development 身份；OIDC 模式拒绝浏览器 bearer-token 换会话。
- `POST/GET /api/v1/auth/session` 的浏览器 JSON 仅返回认证布尔值、CSRF（仅 POST）、过期时间和模式；issuer、tenant、subject、email 与 roles 只存在于服务端权威及 AES-256-GCM 封存的 HttpOnly 票据中，Cookie 的任何 Base64 可解码片段都不含这些原始字段，且响应强制 `Cache-Control: no-store`。旧的明文签名 v1/v2 票据不会迁移或继续认证。
- 会话 Cookie 使用 HMAC-SHA256、`HttpOnly`、`SameSite=Strict`；生产名称使用 `__Host-` 前缀并强制 `Secure`。支持 previous secret 平滑轮换。
- 每个已签发会话先在服务端登记一条仅含 session UUID SHA-256 的 authority 记录；生产使用 PostgreSQL durable FORCE-RLS 表，本地开发才使用有界内存 adapter。每次请求在验签后还必须通过撤销检查。
- `DELETE /api/v1/auth/session` 先事务化撤销当前设备会话，成功后才清 Cookie；重复请求幂等。存储失败返回 `503` 且不清 Cookie，因此不会把未完成的撤销冒充为退出成功。
- 写接口要求与会话绑定的双提交 CSRF token、必需且精确匹配的可信 Origin，以及非 cross-site Fetch Metadata。
- Principal 由服务端生成并在内部绑定 `tenantId`、`subject`、随机 `sessionId`、provider 和 roles；公开会话响应会移除内部 `sessionId`。受保护路由不直接信任开发身份头。
- OIDC 签名 token 还可携带精确的远程处理状态与政策版本。状态仅接受 `denied|adult_roster_authorized|minor_guardian_verified|minor_school_policy_verified`；缺失、未知或版本不匹配统一签入 `remote_processing_eligible=false`。该政策只存在于内部 principal/session，公开会话 JSON 不返回，浏览器 JSON 中任何 age/guardian/provider/policy 字段均在 worker 调用前拒绝。
- Session、Task、Event 的 repository/stream 接口均携带 `{tenantId, ownerId}`；越权访问统一表现为 `404`，减少对象枚举。
- Session PATCH 强制 `If-Match` 版本；PostgreSQL adapter 在事务内 `FOR UPDATE` 并递增版本，旧版本返回 `409`。
- Task idempotency key 按 tenant/user/session 唯一。任务入队先锁定 owned Session，因此不能与终态更新发生静默竞态。
- `migrations/001_tenant_rls.sql` 至 `004_account_deletion_recovery.sql` 为 Session、Task、Event、Auth Session、删除 operation、恢复租约和永久 tombstone 建立 FORCE-RLS/最小系统查询边界；每个 scoped DB 事务用 `set_config(..., true)` 注入租户和用户作用域，并共享账户 lifecycle advisory lock。
- PostgreSQL 模式启动时检查全部表、Auth Session/账户 tombstone 精确安全列、强制 RLS，以及连接角色不是 superuser/BYPASSRLS；任一条件不满足即拒绝启动。

## 真实 Python Teaching Harness 网关

`apps/api` 不再需要把已认证请求降级为一个共享的 loopback Python 实例。启用 `HARNESS_GATEWAY_ENABLED=true` 后，`HarnessWorkerPoolService` 只使用服务端会话中的 `tenantId + subject` 计算 HMAC scope；客户端提交的 tenant/user header、JSON 字段和 query 参数都不参与 worker 选择。

- 每个 tenant+owner scope 是一个独立 Python 进程、独立 capability、独立 mode-0700 根目录。Session、stream journal、后台任务、课程、项目、资源、学习记录、元认知、复核和 consent store 都位于该 scope 根内。
- worker 池达到 `HARNESS_MAX_WORKERS` 时，只会选择超过 `HARNESS_WORKER_IDLE_TIMEOUT_MS`（默认 15 分钟）的 `activeRequests=0` LRU worker：先从可路由 map 原子摘除，再等待 SIGTERM/必要时 SIGKILL 的确定退出，之后才给新 scope 复用容量。starting、stopping、活动 HTTP/SSE worker 都不可淘汰；退出无法确认时本次新 scope 返回 `503` 且旧进程继续计入物理容量。淘汰绝不删除 opaque scope 根，后续请求会从 durable state 启动新 worker。
- scope 目录名是不可逆 HMAC（`scope_<48 hex>`）；pool 元数据和磁盘路径不写 raw tenant/owner。worker capability 绑定稳定的 scope data key、key version 和随机 worker identity。稳定 data key 仅以 active scope key 派生的 AES-256-GCM envelope 落盘，明文 key 不落盘。
- capability、scope key material、store 路径和 provider key-file 路径只通过一次性 stdin bootstrap 发送；不进入 child argv 或 env。worker stdout 只允许有界 readiness/status code，HTTP/SSE 错误也只返回固定安全码/消息。
- 同一浏览器 `request_id` 会在进入 Python 前按 scope 再做 HMAC 命名空间化。因此两个租户使用相同 request/session/task 猜测不会命中同一幂等记录、SSE run 或取消命令。
- worker 异常退出会从 pool 原子摘除；下一次同 scope 请求从同一私有 durable store 启动新 worker。启动窗口、容量耗尽、错误状态或异常退出都返回 `503`，不会借用别的 scope worker。
- 每次请求都从签名 session 私下投影该学习者的精确 subject policy。worker 首次启动绑定 canonical policy SHA-256；同 scope 的新会话若政策变化，活动旧 worker 先被 fence，空闲旧 worker 确认停止后才重建，因此不会继续使用陈旧的 guardian/roster 决策。DeepSeek worker 还必须在启动时收到完整 provider id/version/region/retention/deletion/documentation policy，缺一即 API 启动失败。
- SSE 断开只关闭订阅连接，不隐式取消后台教学；显式取消必须走同 scope 的 cancel route。Python durable task/journal 决定 cancel 与 commit 谁获胜。
- Fastify 的 18 MiB 全局解析上限只用于让 17 MiB 的 `/api/resource` JSON envelope 到达认证路由；`onRequest` 先做全局 32 个正文并发槽，`preParsing` 再按实际收到的字节（包括 chunked）逐路由截断，15 秒 request timeout 防止慢分块长期占槽。controller 在 worker 调用前仍复核实际重编码字节。普通/stream/task 为 64 KiB，consent 16 KiB，syllabus 384 KiB，project 2 MiB，attachment 6 MiB，resource 17 MiB，不能借用 resource 的大包额度。
- 精确的 `GET /api/v1/harness/api/projects/:id/export` 不走默认 32 MiB JSON buffer；它要求 Python 返回可信的 ZIP content type、`Content-Length`、下载名和 manifest SHA-256，并以 256 MiB 硬顶逐块透传。声明超限在发头前返回安全 `502`，实际字节越界/不足或客户端断开都会销毁流；其余非 SSE 响应继续使用有界内存读取。

浏览器契约保持同源 Cookie 身份边界：

1. 先调用 `POST /api/v1/auth/session`，保留服务端签发的 HttpOnly session Cookie 和响应中的 CSRF token。
2. 所有 Harness GET 使用 `credentials: include`；所有 POST 还必须带 `x-csrf-token` 和可信 Origin。
3. Python 原路由放到 `/api/v1/harness/` 后。例如 bootstrap 是 `GET /api/v1/harness/api/bootstrap`；普通 start/step/session 是 `POST /api/v1/harness/api/start|step|session`。
4. 原生流是 `POST /api/v1/harness/api/stream`，请求带 `Accept: text/event-stream`；网关保留 `X-Harness-Run-ID`、`X-Harness-Turn-ID`、后台 task id/version 和逐字节 SSE。重连继续提交浏览器原始 `request_id`、返回的 run/turn 以及 `after_sequence`。
5. 显式流取消是 `POST /api/v1/harness/api/cancel`；durable task status/cancel/resume 分别为 `/api/v1/harness/api/tasks/status|cancel|resume`。所有对象访问都先按已签名 principal 选定隔离 worker。

scope key 轮换由 `HARNESS_SCOPE_KEY_VERSION` + `HARNESS_SCOPE_SECRET` 和可选 `HARNESS_PREVIOUS_SCOPE_KEYS` keyring 完成。新 scope 使用 active key；发现 previous opaque 目录时，pool 在 scope 启动 gate 与跨进程锁内 copy/fsync/manifest verify，再把原稳定 data key 用 active wrapping key AEAD 重封并原子 rename。旧目录只留下签名、hash-only 迁移 tombstone，不再写入；崩溃可重试。只有全部 scope 已成功迁移并通过“移除 previous key 后仍可读取真实 session、signed consent/learning store”的恢复演练，才能下线 previous key。复制目录而不重封 data key 不构成迁移。

## 账户级数据权利

账户操作仅存在于 `AUTH_MODE=oidc` 的 authenticated Apps API 模式。服务端从已验证的规范化 `issuer + tenant + sub` 构造版本化身份 namespace 与不可逆 scope hash；请求 JSON/header 不接受 tenant、user、actor 或 scope。旧 namespace 默认不继承。密钥轮换应保留 previous account scope key 直至迁移验证完成；若删除 saga 已启动，持久化 operation 和签名状态能力仍可在旧 key 退役后完成恢复，并在最终事务补齐当前 key 的永久 tombstone，防止新 scope 重开。

- `GET /api/v1/account/export` 要求最近五分钟内、精确 AAL2 `acr` allowlist 的 IdP step-up。它在同一 scope fence 下汇总 PostgreSQL Session/Task/Event、hash-only auth-session audit，以及 worker root 中的 project/session/syllabus/resource/review/learning/metacognition/adjudication/consent/journal/task registry。ZIP 使用 STORE、canonical manifest、逐项 SHA-256/CRC32、20,000 项/384 MiB 数据/400 MiB archive 硬顶流式输出；不包含 raw tenant/subject、scope host path、Cookie/CSRF、scope key envelope 或 capability。
- `POST /api/v1/account/deletion/prepare` 需要同样的 fresh AAL2 authority 与 CSRF，签发五分钟 challenge、精确英文确认短语、revision 和单独 HttpOnly status capability。`POST .../confirm` 绑定准备时 session、fresh CSRF、challenge token、expected revision、idempotency 与 issuer authority，不能由普通旧会话直接完成。
- confirm 先取得带 token CAS 的 durable recovery lease，再全局 fence scope、停止新 run、有限等待 drain，为 active/previous scope roots 写 durable tombstone、原子 quarantine、完整清除 worker 数据，最后在同一 PostgreSQL advisory-lock transaction 中按 Event→Task→Session→全部 auth session 删除并完成 hash-only tombstone。每个 phase 都校验 lease token + revision；启动扫描器和周期扫描器只用 `FOR UPDATE SKIP LOCKED` 认领到期 operation，多实例不会偷取未过期租约。进程重启会重新建立内存 worker fence；purge 在不可逆目录删除前先 fsync content-free 计数 receipt，避免“文件已删但计数未落库”的崩溃窗口。已越过 HTTP guard 的 scoped DB write 也必须在同一 transaction predicate 下失败，迟到 worker/outbox 不能复活已删 scope。
- commit 后清当前 session/CSRF；30 天 HttpOnly status capability 可在认证会话已经消失后查询 content-free phase/receipt，并可通过同源 `POST .../deletion/resume` 请求立即认领到期/失败的 saga；它不能恢复账户身份或访问内容。Console 会在刷新后恢复尚未过期的 challenge/idempotency、轮询 deleting/retryable failure、完成后验证 receipt 再清浏览器账户缓存。收据只声明本 TeachLab 部署的在线主存储已删除，并固定写明：组织 IdP 账户未删除，须联系组织身份管理员；运维备份未同步删除，等待保留期到期或部署方加密擦除；用户自行保存的导出和 DeepSeek/搜索服务远端副本也不受本地删除控制。这里的 all-devices 撤销只属于账户删除，不是 standalone logout-all 功能。
- Console 必须先比较 bootstrap 的 issuer-bound opaque `cache_scope`，再读取任何 IndexedDB/localStorage/sessionStorage；scope 变化或未认证时先清 drafts、outbox、workspace、Cache API 和内存查询缓存，避免 A 会话过期且未 logout 后直接登录 B 时重放 A 的本机状态。浏览器绝不存 raw tenant/user。

### 认证教师裁决边界

精确的 `POST .../api/resource/review` 与 `POST .../api/adjudication/claim|decide` 都会进入教师权威门。API 在获取或调用任何 worker 前，以已验证 OIDC principal 的 canonical `issuer + tenant + sub` 调用 `TEACHER_ENTITLEMENT_DIRECTORY_URL`；浏览器 Cookie 中的 roles 完全不参与授权。目录必须返回 exact-schema、身份精确绑定、policy version 匹配、未过期且足够新的单调 revision snapshot。outage、超时、redirect、超限响应、stale/revoked/missing/role-missing 都 fail closed。`TEACHER_AUTHORITY_ROLES` 是在该权威 snapshot 上匹配的大小写敏感精确 allowlist（默认仅 `teacher`），`TEACHER_AUTHORITY_TTL_SECONDS` 只控制随后 scope-signed worker assertion 的 30–600 秒寿命。

当前拓扑没有 entitlement change event consumer，因此 settled directory cache 的最坏撤权滞后等于 `TEACHER_ENTITLEMENT_CACHE_TTL_MS`，生产硬上限 5000 ms；到期后的 provider outage 不会沿用旧授权。Bearer credential、principal-binding key、receipt-signing key 必须使用三个彼此独立且与其他服务 secret 不同的 private files。原始身份与 Bearer 只出现在目录 HTTPS POST；返回给 worker 的只是签名 allow receipt 的 SHA-256 绑定，完整 receipt、原始 issuer/tenant/sub 和 Cookie roles 都不进入 worker JSON。浏览器提交的 tenant/user/subject/principal/role/actor/authority/receipt 字段会在目录/worker 调用前递归拒绝，不能用 JSON 或 header 自我升级。

API 从当前服务端 principal 的 issuer+subject 生成 scope-HMAC 不可逆 actor digest，并只把 roles/policy 摘要、route/method、canonical body hash、idempotency hash、scope/key version、时间和 nonce 写入短期签名 envelope；raw identity 不进入 Python 请求或持久日志。Python 在任何裁决 mutation 前验签并把 nonce 写入 mode-0600、hash-chain、append-only 的永久 tombstone journal。journal 只增长，达到公开的 16 MiB 容量时 fail closed；不能通过 TTL 清理旧 nonce 后重新接受 envelope。运维如需迁移容量，必须在停写下保留全部 tombstone 语义，不能截断后继续使用同一 authority key。

Authenticated `correct` 还要把封存证据、model ledger hash、instruction、correction、目标 KC 与教师 rubric authority 重新绑定成服务端签名 receipt，随后才可 supersede 原证据并精确重放目标 KC；非目标 KC 必须不变。`approve` 不改模型，`abstain` 保守回滚；standalone `local_python` 的 Correct 仍只进入 pending。该 receipt 只证明本部署服务曾基于已认证角色授权，不是教师个人不可否认签名。scope key 轮换同时轮换 authority key；previous scope 只能用其原 key 恢复和验签，不能跨 scope/key/route/payload 重放。

## 本地运行（明确 local-only）

```bash
cd apps/api
cp .env.example .env
npm ci
npm run dev
```

默认监听 `http://127.0.0.1:4000`。先向 `POST /api/v1/auth/session` 发请求取得两个 `Set-Cookie`；响应体中的 `csrfToken` 需在后续 POST/PATCH/DELETE 的 `x-csrf-token` 中回传。浏览器请求必须使用 credentials。

`DEV_AUTH_ALLOW_HEADERS=true` 仅用于隔离测试，让 session exchange 接受 `x-dev-user-id` 与 `x-dev-tenant-id`；这些 header 对其他路由没有身份效力。生产启动会拒绝 development auth、header override、demo seed、内存数据、非安全 Cookie、非 HTTPS CORS origin 和非 `verify-full` 数据库 TLS。

## 生产配置

至少需要：

```dotenv
NODE_ENV=production
AUTH_MODE=oidc
OIDC_ISSUER=https://identity.example.test
OIDC_AUDIENCE=teachlab-api
OIDC_CLIENT_ID=teachlab-console
OIDC_CLIENT_SECRET_FILE=/run/teachlab/oidc-client-secret
OIDC_REDIRECT_URI=https://console.example.test/api/teacher-agent/security/login/callback
OIDC_TRANSACTION_SECRET_FILE=/run/teachlab/oidc-transaction-secret
OIDC_EXPECTED_HOST=api.example.test
OIDC_ACCOUNT_STEP_UP_MAX_AGE_SECONDS=300
OIDC_ACCOUNT_AAL2_ACR_VALUES=urn:teachlab:aal2
TEACHER_ENTITLEMENT_DIRECTORY_URL=https://entitlements.organization.example/v1/teacher/snapshot
TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET_FILE=/run/teachlab/teacher-entitlement-directory-bearer
TEACHER_ENTITLEMENT_BINDING_KEY_FILE=/run/teachlab/teacher-entitlement-binding-key
TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE=/run/teachlab/teacher-entitlement-receipt-key
TEACHER_ENTITLEMENT_POLICY_ID=teacher-mutations
TEACHER_ENTITLEMENT_POLICY_VERSION=roles-v7
TEACHER_ENTITLEMENT_FRESHNESS_TTL_MS=30000
TEACHER_ENTITLEMENT_CACHE_TTL_MS=5000
TEACHER_ENTITLEMENT_PROVIDER_TIMEOUT_MS=2000
TEACHER_ENTITLEMENT_MAX_RESPONSE_BYTES=65536
TEACHER_ENTITLEMENT_MAX_CLOCK_SKEW_MS=5000
TEACHER_ENTITLEMENT_MAX_CACHE_ENTRIES=10000
TEACHER_ENTITLEMENT_MIN_ASSURANCE_LEVEL=2
OIDC_TENANT_CLAIM=org_id
OIDC_ALLOWED_ALGORITHMS=RS256
OIDC_REMOTE_PROCESSING_POLICY_CLAIM=teachlab_remote_processing_policy
OIDC_REMOTE_PROCESSING_POLICY_VERSION_CLAIM=teachlab_remote_processing_policy_version
REMOTE_SUBJECT_POLICY_ID=school-remote-processing
REMOTE_SUBJECT_POLICY_VERSION=2026-08-12
SESSION_SECRET_FILE=/run/teachlab/session-secret
ACCOUNT_SCOPE_KEY_VERSION=k1
ACCOUNT_SCOPE_SECRET_FILE=/run/teachlab/account-scope-secret
ACCOUNT_IDENTITY_NAMESPACE_VERSION=ns1
ACCOUNT_IDENTITY_NAMESPACE_SECRET_FILE=/run/teachlab/account-identity-namespace-secret
ACCOUNT_DELETION_STATUS_SECRET_FILE=/run/teachlab/account-deletion-status-secret
ACCOUNT_CACHE_SCOPE_SECRET_FILE=/run/teachlab/account-cache-scope-secret
ACCOUNT_CACHE_SCOPE_EPOCH=epoch1
SESSION_COOKIE_SECURE=true
CORS_ORIGINS=https://console.example.test
DATA_BACKEND=postgres
DATABASE_URL_FILE=/run/teachlab/database-url
PG_SSL_MODE=verify-full
SEED_DEMO_SESSIONS=false
HARNESS_GATEWAY_ENABLED=true
HARNESS_WORKER_ROOT=/var/lib/teachlab/harness-workers
HARNESS_WORKER_CWD=/opt/teachlab/application
HARNESS_WORKER_PYTHON=/opt/teachlab/runtime/bin/python
HARNESS_WORKER_BACKEND=deepseek
HARNESS_WORKER_FILESYSTEM_ISOLATION_REQUIRED=true
HARNESS_WORKER_RLIMIT_CORE_BYTES=0
HARNESS_WORKER_RLIMIT_NOFILE=256
HARNESS_WORKER_RLIMIT_FSIZE_BYTES=536870912
HARNESS_WORKER_RLIMIT_AS_BYTES=1610612736
HARNESS_PROVIDER_API_KEY_FILE=/run/secrets/deepseek-api-key
HARNESS_PROVIDER_POLICY_ID=deepseek-approved-terms
HARNESS_PROVIDER_POLICY_VERSION=2026-08-12

# Safeguarding resources use this trusted server-side language locale. It is
# never selected from a browser request and is not a jurisdiction guarantee.
HARNESS_SAFEGUARDING_LOCALE=zh-CN
HARNESS_SAFEGUARDING_DISPATCH_URL=https://safeguarding.example.test/v1/cases
HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET_FILE=/run/teachlab/safeguarding-dispatch-secret
HARNESS_SAFEGUARDING_DISPATCH_POLICY_VERSION=institution-safeguarding-v1
HARNESS_SAFEGUARDING_DISPATCH_TIMEOUT_MS=5000
HARNESS_SAFEGUARDING_DISPATCH_MAX_RESPONSE_BYTES=32768
HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION=closed-case-retention-v1
HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS=7776000
HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN=128
HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET_FILE=/run/teachlab/safeguarding-retention-authority
HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256=<stable-64-lowercase-hex>
HARNESS_PROVIDER_PROCESSING_REGION=cn_north
HARNESS_PROVIDER_RETENTION_DAYS=7
HARNESS_PROVIDER_DELETION_STATUS=outside_service_control_subject_to_provider_policy
HARNESS_PROVIDER_DOCUMENTATION_URL=https://provider.example/privacy
HARNESS_SCOPE_KEY_VERSION=k2
HARNESS_SCOPE_SECRET_FILE=/run/teachlab/harness-scope-secret
HARNESS_PREVIOUS_SCOPE_KEYS_FILE=/run/teachlab/harness-previous-scope-keys
METRICS_TOKEN_FILE=/run/teachlab/metrics-token
HARNESS_WORKER_IDLE_TIMEOUT_MS=900000
TEACHER_AUTHORITY_ROLES=teacher
TEACHER_AUTHORITY_TTL_SECONDS=120
```

配置 safeguarding dispatcher 后，API 会启动一个全局、content-free 的投递监督器。
接收服务必须在同一 HTTPS endpoint 支持带同一 Bearer 凭证的 `GET` readiness；请求没有
body、不会创建 case，响应必须是 `application/json` 且 exact 为
`{"schema":"teaching_skill_miner.safeguarding_dispatch_readiness.v1","status":"ready","policy_version":"<HARNESS_SAFEGUARDING_DISPATCH_POLICY_VERSION>"}`。
接收端不可达、凭证/策略不匹配、存在失败或逾期投递时 `/ready` fail closed；`/health`
仍作为 liveness 返回 200，并在 `teachingHarness` 中诚实投影聚合状态，避免接收端故障触发
API 重启风暴。dispatcher Bearer secret 只经 supervisor stdin 注入，不进入 argv、环境变量、
健康响应或指标标签。

production 还必须显式配置 retention policy version、最小 closed age、每轮最多 case 数、稳定的
deployment-context SHA-256 和独立 mode-0600 authority secret；五项缺一即拒绝启动，浏览器无覆盖
入口。supervisor 只会压缩已经 `closed` 且 escalation delivery 已 `acknowledged`、并达到最小年龄的
case；open、仅 case-acknowledged、pending、overdue 或 delivery unavailable 均不符合条件。status v2
只投影聚合 capacity/headroom、压缩计数、blocked stores 和 erasure-fence 误报上界；不含 case、scope
或路径 label。retention disabled、blocked、capacity near-limit、操作失败或估计误报上界超过
`1e-6` 都使 `/ready` 返回 503，`/health` 仍返回 200 并诚实投影。

Linux production worker 的同一私有 bootstrap 还必须携带固定的进程资源策略：
`RLIMIT_CORE=0`、`RLIMIT_NOFILE=256`、`RLIMIT_FSIZE=512 MiB`、
`RLIMIT_AS=1.5 GiB`。worker 在创建 dashboard、parser 或网络客户端前同时降低 soft/hard
limit；若宿主已有更严格 hard limit，会采用更低值，但低于运行安全下限时启动失败。普通 worker
与无 learner 数据的 runtime canary 都返回 exact、聚合的生效状态供 API 校验；浏览器不能提交或
覆盖这些值。本地未要求 Landlock 时 bootstrap 固定为 `process_resource_limits=null`。这些限制由
parser 子进程继承，但不是 process-tree/cgroup 总量，也没有设置长寿命进程不适用的
`RLIMIT_CPU`，或共享 uid 下不安全的 `RLIMIT_NPROC`；生产 Compose 另以 container-wide
`pids_limit=256` 限制全部 API/worker/parser 进程，请求墙钟/并发门禁与 4 CPU/4 GiB 容器预算仍是补充边界。

先按 [migrations/README.md](migrations/README.md) 使用独立 migration owner 应用迁移。API 连接角色必须是非 superuser、非 BYPASSRLS；数据库凭证不能进入浏览器或执行沙箱。

## API 摘要

| Method | Path | 用途 |
| --- | --- | --- |
| GET | `/health`, `/ready` | liveness 与持续依赖 readiness；不泄露凭证 |
| POST/GET/DELETE | `/api/v1/auth/session` | OIDC/local exchange、读取会话、退出 |
| GET/POST | `/api/v1/sessions` | tenant/owner-scoped durable Session API |
| GET/PATCH | `/api/v1/sessions/:sessionId` | owned Session 详情与版本化更新 |
| POST | `/api/v1/sessions/:sessionId/tasks` | owned Session 的异步 Agent turn |
| GET | `/api/v1/tasks/:taskId` | owned Task 状态 |
| POST | `/api/v1/tasks/:taskId/cancel` | owned Task 取消 |
| GET | `/api/v1/sessions/:sessionId/events` | owned Event SSE；支持 Last-Event-ID |
| GET | `/api/v1/providers/model` | authenticated provider status; production requires configured durable backend |
| GET/POST | `/api/v1/harness/<python-route>` | 已认证、tenant+owner 隔离的真实 Python Harness |
| POST | `/api/v1/harness/api/stream` | 原生 durable Harness SSE |
| POST | `/api/v1/harness/api/cancel` | 同 scope 显式取消 run/request |
| GET | `/api/v1/account/export` | fresh AAL2、scope 完整 canonical ZIP 导出 |
| POST | `/api/v1/account/deletion/prepare` | fresh AAL2 + CSRF 签发短期删除挑战 |
| POST | `/api/v1/account/deletion/confirm` | 精确 phrase/revision/idempotency 的永久删除 |
| POST | `/api/v1/account/deletion/resume` | HttpOnly status capability 同源恢复已确认 saga，不恢复账户身份 |
| GET | `/api/v1/account/deletion/status` | HttpOnly capability 查询 content-free phase/receipt |

Console 兼容入口 `/api/bootstrap`、`/api/step`、`/api/events?session_id=...` 使用完全相同的会话、CSRF 和 ownership 边界。

## 验证与诚实边界

```bash
npm audit --audit-level=high
npm test
npm run typecheck
npm run build
```

自动测试覆盖 Cookie/CSRF、OIDC RSA 验签、旧签名 key、逐会话撤销、存储失败 fail-closed、两个用户与两个租户的隔离、Session 乐观冲突、Task idempotency/cancel、Event ownership、事务作用域、账户导出/删除 crash phase/CAS/late-write fence、真实 worker scope-key rewrap，以及 RLS migration 契约。常规本机运行继续使用可记录的 transaction-contract fake；CI 另启动一次性 PostgreSQL 17 服务，以独立 migration owner/application role 实际执行全部迁移，并验证 FORCE RLS 隔离、跨租户写拒绝、并发撤销/CAS 和 Task 幂等。该一次性门禁不替代目标云数据库的迁移演练、TLS、备份恢复和故障切换验证。

旧 `/api/bootstrap|api/step|api/events` compatibility contract 仍在 production 返回 410；现代 `/api/v1/sessions|tasks|providers` 使用 PostgreSQL FORCE-RLS storage and durable task leases when `DATA_BACKEND=postgres`. Local development continues to report explicit memory-only adapters. Account deletion fences both task claims and dispatcher event writes for the deleting scope. The worker pool remains a single API-replica scope partitioner; horizontal expansion still requires deployment-level stable scope sharding or a distributed worker scheduler. Health and readiness report these boundaries.
