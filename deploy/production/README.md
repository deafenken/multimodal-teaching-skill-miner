# TeachLab 单副本生产拓扑

这是当前 Harness worker 架构可安全支持的生产边界：一个 API replica 作为所有
tenant/owner scope 的单写者，每个 scope 使用独立 Python worker 与持久目录。它不是多副本
分布式调度器；在引入分布式 lease 之前，不得把第二个 API replica 指向同一 `scope_data`。

## 构建与发布

1. 先完成仓库 release acceptance，取得 wheel 的 SHA-256。
2. 构建时为 Dockerfile 的每个 base-image argument 传入审核过的 `name@sha256:<64hex>`；
   Dockerfile 会拒绝 mutable tag。
3. API image 用验收 wheel 及其 SHA 构建；Console 使用锁文件 `npm ci` 和 standalone build。
4. 扫描两个 image、生成 SBOM/provenance、推送 registry，再把
   `TEACHLAB_API_IMAGE`、`TEACHLAB_CONSOLE_IMAGE`、`TEACHLAB_CADDY_IMAGE` 设为最终
   registry digest。Compose 文件不接受默认 image。

当前仓库没有 registry 凭据、组织签名密钥或受信任 base-image digest，因此不会伪造已签名
image。上述三个外部供应链步骤必须由部署方完成并留下收据。

## 密钥与数据

`TEACHLAB_SECRETS_DIR` 必须指向一个由 uid `10001` 拥有、权限 `0700` 的绝对 host 目录；
Compose 将它整体只读挂载到 `/run/teachlab`。目录内每个约定名称的文件必须同样由 uid
`10001` 拥有、权限 `0600`。符号链接、相对路径、宽权限、缺文件或同时提供 inline 值都会让
API 拒绝启动。不要改回逐文件 bind mount：那会让容器内父目录成为 root 所有，从而破坏
父目录校验。不要把值写入 Compose、shell history、镜像层或日志。`scope_data` 需要加密、
版本化备份和恢复演练；项目/账户永久删除还必须等待对应 deletion receipt，而不是直接删卷。

账户数据权利还依赖四个相互独立的稳定 secret：account scope/tombstone、issuer identity
namespace、post-delete status capability、browser cache scope。`ACCOUNT_*_VERSION/EPOCH` 是协议版本，
不能在无迁移计划时随部署随机变化。IdP 必须配置并实测
`OIDC_ACCOUNT_AAL2_ACR_VALUES`；普通登录或仅持有旧 session 不能导出/删除全账户。

教师资源复核与裁决 claim/decide 另依赖组织权威 entitlement directory。必须配置 exact HTTPS
`TEACHER_ENTITLEMENT_DIRECTORY_URL`、policy ID/version，以及目录 Bearer、principal-binding、
receipt-signing 三个彼此独立且不能复用其他服务 secret 的 mode-0600 文件。API 对三条 mutation
均先查目录再接触 worker，完全忽略 Cookie roles；网络/协议/schema/identity/policy/freshness/revision
任何异常都 fail closed。当前没有 role-change event consumer，因而最坏撤权滞后为 settled cache TTL；
生产代码强制 `TEACHER_ENTITLEMENT_CACHE_TTL_MS <= 5000`。完整 allow receipt 不离开 API，worker
assertion 只绑定其 SHA-256，原始 issuer/tenant/sub 与目录 Bearer 不进入 worker JSON 或持久日志。

安全保护 staff workflow 复用同一权威目录，但每条 list/dispatch/case acknowledge/case close/
escalation overdue/escalation acknowledge 都以 exact operation 查询，且只接受精确单例角色
`safeguarding`，不会接受 Cookie roles 或普通 `teacher` 角色。安全事件只进入 content-free durable
outbox（opaque scope、category、severity、时间、content SHA-256）；原文不会进入 queue 或 staff API。
`HARNESS_SAFEGUARDING_DISPATCH_URL` 与独立 mode-0600 Bearer 文件指向机构 HTTPS receiver，接收方以
`delivery_id` 幂等并返回 exact acknowledgement。API 启动一个独立全局 supervisor，扫描 opaque scope
中的 durable outbox，因此 learner worker 被 idle-evict、没有新请求或 API 重启后仍会以同一
`delivery_id` 指数退避重试；同一 per-store lease 防止并发 supervisor 重复执行。HTTP accepted 只表示 receiver
暂存成功，不会伪造 durable delivery ack；receiver 仍须以 fresh safeguarding entitlement 调用独立 ack API。
跨用户操作依赖租户绑定 AEAD opaque routing locator；
API 每次先 fresh-authorize staff，再解密并限制到同租户 learner scope。supervisor 还用无 body、不会创建
case 的 authenticated GET 验证 receiver 网络、凭据与精确 policy；失败、逾期或不可用 store 会令
`/ready` fail closed，但 `/health` 继续 200 并诚实投影聚合状态。未配置 dispatcher 的自定义部署会保留
case、将 escalation 明示为 unavailable，并让 dispatch fail closed；本 Compose 则强制配置。

本 Compose 也强制显式 closed-case retention policy：policy version、最小 closed age、每轮最多
case 数、稳定 deployment-context SHA-256，以及独立且不得与任何其他服务密钥复用的
`safeguarding-retention-authority` mode-0600 文件。五项仅经服务端 bootstrap v2 进入 supervisor，浏览器
没有覆盖面；deployment 重启必须保留同一 authority/context。只有 `closed` 且 escalation delivery
已 `acknowledged`、达到最小年龄的 case 可被有界压缩；open、仅 case-acknowledged、pending、overdue
或 delivery unavailable 永不自动压缩。capacity near-limit、retention blocked/failed/disabled 或
erasure-fence 估计误报上界超过 `1e-6` 都使 `/ready` 返回 503；`/health` 与 Prometheus 仅投影无
case/scope/path label 的聚合 headroom、bytes、compaction、blocked 与误报上界。

IdP 还必须签发两个精确的学习者远程处理 claim：一个状态、一个政策版本。状态仅允许
`denied`、`adult_roster_authorized`、`minor_guardian_verified`、
`minor_school_policy_verified`；缺失、未知或版本与 `REMOTE_SUBJECT_POLICY_VERSION` 不符时
服务端强制降为 denied。浏览器不能自报年龄、guardian 或 eligibility。DeepSeek 部署同时必须
显式配置 `HARNESS_PROVIDER_POLICY_ID/VERSION/PROCESSING_REGION/RETENTION_DAYS/DELETION_STATUS/DOCUMENTATION_URL`；
显式配置可信服务端 `HARNESS_SAFEGUARDING_LOCALE`（仅语言本地化；通用当地服务指引并非司法辖区已验证资源）；
这些值是运维方对已审批外部条款的绑定，不是仓库对 provider 删除或保留行为的独立证明。

每个 Linux scope worker 在接触 dashboard、parser 或 provider 前同时安装 Landlock allowlist 与
固定 per-process RLIMIT：core dump `0`、open files `256`、单文件 `512 MiB`、地址空间
`1.5 GiB`。parser 子进程继承这些限制，真实 runtime canary 也走同一 bootstrap 并让 API 校验
生效值；宿主 hard limit 若更严格会采用更低值，低于明确运行下限则 fail closed。它不是
process-tree/cgroup 的 per-worker 聚合 CPU、内存、磁盘或进程数预算；worker 仍共享 uid `10001`，因此不设置
长寿命进程不适用的 `RLIMIT_CPU` 或共享 uid 下会互相消耗的 `RLIMIT_NPROC`。请求墙钟/并发/体积
门禁和 API 容器的 4 CPU/4 GiB、256 PID 总量上限仍是补充边界，也不能把它表述成 gVisor、独立
uid 或 VM 隔离。本地/测试模式不声称存在该 container PID cap。

备份必须在同一恢复点包含 PostgreSQL 全部三次 migration 后的数据、整个 `scope_data`（包括
active envelope、previous root 的 hash-only migration tombstone、deletion quarantine/lock）以及
对应 active/仍在轮换期的 previous Harness/account wrapping secrets。scope data key 明文不得进入备份。
恢复到隔离环境后，应先验数据库 tombstone，再以原 wrapping secret 解封 envelope，执行真实
session + signed consent/learning 读取；随后轮换到新 key、移除 previous key并再次读取。不能只恢复
worker 文件而漏掉 PG deletion tombstone，否则迟到数据可能复活；也不能只恢复 DB 而漏掉正在清除的
quarantine。完成删除的 scope 不应从较旧备份直接回灌；恢复流程必须先应用永久 tombstone 并丢弃其
worker payload。

账户 ZIP 与用户另存备份不受服务端删除控制；操作者需独立执行其保留/销毁流程。DeepSeek/搜索
provider 已接收的副本同样受 provider 政策约束，账户删除收据会明确写
`remote_provider_copies_deleted=false`。组织 IdP 账户由 IdP 管理员另行处理，收据固定写
`identity_provider_account_deleted=false`；部署备份只有在保留期到期或完成密钥销毁/加密擦除后才可
另行证明清除，在线删除收据固定写 `operator_backup_copies_deleted=false`，不得把它改成同步删除声明。

## 启动、探针与回滚

用独立的 secrets env 文件只保存非秘密路径/主机名，然后执行：

```sh
docker compose --env-file /private/teachlab/deployment.env \
  -f deploy/production/compose.yaml config >/private/teachlab/rendered-compose.yaml
python scripts/verify_teacher_agent_deployment.py \
  --compose /private/teachlab/rendered-compose.yaml \
  --caddyfile deploy/production/Caddyfile
docker compose -f /private/teachlab/rendered-compose.yaml up -d
```

`/health` 仅表示进程存活；`/ready` 连续探测 PostgreSQL、真实 runtime canary、content-free
DeepSeek models endpoint、receiver readiness、safeguarding backlog，以及 retention capacity/headroom
和 erasure-fence 误报上界。Caddy 对 API、
受保护 metrics 和 Console upstream 每 10 秒主动探测 `/ready`，连续两次失败即停止转发，
连续两次恢复后才重新纳入（最坏摘流窗口约 22 秒）；`/metrics` 默认隐藏，只有私有 token 文件
配置且 Bearer 精确匹配才可读取。边缘只
公开同源 HTTPS，API 和 Console 本身不发布 host port。升级先做数据库迁移/备份，再替换 digest；
回滚只能回到兼容当前 schema 的已验收 digest。恢复演练和 8 小时 soak 均应绑定最终 image digest。

Compose 无法提供跨主机网络 egress policy。生产主机/集群还必须限制 API 仅访问 PostgreSQL、
OIDC issuer、被批准的模型 provider 和 DNS/时间服务；Console 仅访问同源 API；禁止 metadata
service 和任意内网。要横向扩展必须先替换进程内 worker/task/limit 状态为分布式权威实现。
