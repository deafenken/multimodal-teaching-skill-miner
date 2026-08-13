# 项目完备性状态

## run19/run25 历史基线与 run43–run47 重复开发回归

历史 run19 暴露了 `allowed_skill_hit_rate=0.35` 与 `bounded_route_completion_rate=0.40`。修复合法 post-assessment 路由置首、provisional-session 契约检查和有界末步提交后，冻结 run25（6 case / 20 turn，作者构造 development split）为 allowed-Skill hit `0.450000`、switch F1 `0.818182`、bounded route completion `0.750000`；run fingerprint 为 `4a93ba704f20eb7666f8d371f473bb85233b7f132106c26127db7c92a11d3b7b`。run19 和 run25 作为可追溯历史基线保留，不回写原 receipt。

后续根因是：明确自解释主张可被 DeepSeek 保守记为 `partial / ambiguous`，而自解释 Skill 已达重复上限后，Socratic 核验又被旧门禁拒绝。当前只对“存在明确自解释证据”开放有界 Socratic 核验，不提升 mastery、不自动解除误解、不读取 gold。在相同 development input/gold 指纹上，最新完整落盘的真实 DeepSeek run43–run47 的 allowed-Skill hit 为 `1.00/1.00/1.00/0.95/1.00`（分别 `20/20`、`20/20`、`20/20`、`19/20`、`20/20`），switch F1 均为 `0.916667`，bounded route completion 分别为 `0.80/0.70/0.65/0.65/0.70`。run46 的唯一 Skill miss 仍是 `cross_session_alpha/alpha_t2`；五次运行的 cross-session leakage 均为 `0`、unique session instance rate 均为 `1.0`，四类 runtime fallback 也均为 `0`。这证明当前开发回归的 Skill 命中保持在 95% 以上，但同时保留了 Loop 完成率波动这一尚未解决的短板。该 fixture 只有 6 case / 20 turn，由作者构造且未锁箱；不是 Accuracy、held-out、专家锁箱、部署准确率或真实学习效果。私有 predictions/report 不进入仓库。

## 现代 Console / Apps API 生产边界（2026-08-13）

现代 Console 已成为默认桌面入口；本机模式由 Next 同源代理连接 Python Teaching Agent，认证模式则经 Apps API 路由到账号隔离的 Python worker。Python 自带 Dashboard 仅保留为调试页。下表区分已经接入并由仓库门禁覆盖的生产能力，以及仍须由部署方或后续分布式基础设施承担的边界。

| 层 | 已落地的可验证部分 | 明确未落地/仅端口 |
|---|---|---|
| Console | `apps/console`：Next.js **15.5.23**、React 19.1、TypeScript、Tailwind CSS 4；三栏工作台、Chat/Teach、项目、资源、教师复核、课程权威、延迟复习、账号导出/删除和 safeguarding staff 面板均走真实 API。未接线的 Monaco/xterm 已移除。typed SSE reducer 校验 run/turn、连续 sequence、result/commit/terminal；账号 scope 改变时先清 IndexedDB、Cache API 与 query cache 再 hydrate | 自动化覆盖 Chromium/Firefox/WebKit 结构与 axe 门禁，但真实屏幕阅读器认证和长期多日现场运行仍属于外部验收 |
| 前端状态与实时 | TanStack Query bootstrap、scope-bound 离线只读快照、durable-first SSE、cursor reconnect、显式 cancel、后台 task 恢复和 payload 擦除 outbox 已接。Service Worker 仅缓存 immutable 静态壳，API/HTML/私有响应不入 Cache API | WebSocket steering 未实现；pre-provider 起点可安全恢复，未知 provider effect 进入 handoff 且不重放；已有 domain commit 时才补齐外层事件，Chat 非终态不自动恢复 |
| 项目、Chat、课程与资源 | 项目、Chat、notes、资源/大纲/Teach 引用持久化；资源能力由 runtime bootstrap 动态声明。教师可通过 fresh entitlement 复核资源与课程蓝图；只有签名封存且未撤销的课程投影可写 authoritative KC，模型自写 question contract 只能作 provisional 导航 | 项目/资源/检索仍只是教师上下文；原始媒体不发给模型，远程发送前先做安全分类、最小化与直接标识符替换；外部 OCR/ASR/legacy Office 能力未配置时不会广告 |
| Apps API | NestJS 11 + Fastify 5 提供 OIDC、HttpOnly AEAD session、CSRF、PostgreSQL FORCE RLS、账号 DSAR、durable task leases/artifacts、teacher/safeguarding entitlement、租户 scope worker、SSE/cancel、解析前资源闸、health/ready 与 Prometheus | 正式拓扑固定单 API replica；没有可供多副本共享 scope 根的分布式 lease scheduler。仅旧 bootstrap/step/events compatibility 路由在 production 返回 410 |
| 身份与数据权利 | OIDC authorization-code+PKCE、服务端 revocation、issuer+tenant+subject 命名空间、AAL2 recent step-up、角色目录短时权威快照与账号 export/delete 崩溃恢复已接；浏览器不接收原始身份键 | IdP、角色目录、provider/接收端政策、备份到期删除和组织数据治理由部署方运营；Redis/S3 不属于当前单机生产拓扑 |
| Agent、安全与执行 | Python Harness/Teaching Agent 承担 DeepSeek 教学流；统一输入/输出安全门在 provider 与持久化前工作，hash-only safeguarding case/outbox 与 staff CAS workflow 已接；Landlock 将每个 worker 限制在自己的私有 root | 同 uid + Landlock 是文件路径/网络约束，不等于独立 uid、容器或 VM 安全域；Claude worker、Temporal/BullMQ、通用代码执行沙箱、Kubernetes/Firecracker 未实现 |
| 观测与交付 | 低基数 Prometheus、health/ready、资源治理、Docker/Compose/Caddy、镜像与部署契约、Python/API/Console/SBOM CI 已接；release acceptance 从单一只读源码快照重放门禁并绑定 exact wheel | 仓库不内置 Grafana/Loki/Sentry、Terraform/Helm/Argo CD 或跨主机告警平台；生产 scraper、告警接收与外部故障演练仍需部署方完成 |

验证入口：`cd apps/console && npm test && npm run typecheck && npm run build`；`cd apps/api && npm test && npm run typecheck && npm run build`；生产配置另由 `scripts/verify_teacher_agent_deployment.py`、镜像 canary 与 `build_release_acceptance.sh` 验证。以上证明的是仓库内工程与部署契约，不证明目标 IdP/模型/接收端条款、真实屏幕阅读器、跨主机容灾或真实学习效果。

## 已形成闭环的工程能力

| 能力 | 状态 | 验收入口 |
|---|:---:|---|
| 源码与 exact wheel 安装 | 验收入口完成；收据按 scope 自证新鲜度 | `build_release_acceptance.sh` 串联 `verify_project.sh`、双构建一致性、最终 clean wheel、exact 安装/视频 smoke、发布审计和机器生成 acceptance。`artifacts/release_acceptance_1.2.0.json` 只有在其 verification scope 对当前树重算一致时有效；文档不复制测试总数 |
| 浏览器答辩看板 | 完成 | `tsm dashboard` 直接打开自包含 HTML；包含合成多模态时间轴、TeachObs 四臂消融、DIPSER 评估范围、可执行 Skill 和 claim boundary；只使用公开聚合数据，不包含真实私有媒体或逐样本记录 |
| 本机双成果真实演示 | 独立私有入口 | `tsm dashboard-real` 同站展示两条独立验证轨：① 6 个 TeachObs 测试讲次、1,099 场景的完整视频/字幕/教师行为投影/冻结四臂逐样本预测；② 10 个 MIT 完整讲次、2,270 候选事件、10 份 Full Skill 与 90 个实际时序步骤（62 observed + 28 recommended），并可检查 `evi_*`、脱敏 `mme_*` 和 Runtime fallback。Skill procedure 仍由字幕证据主导，直接 `mme_*` 引用为 0；两轨当前没有直接数据流。服务仅在 `127.0.0.1` 使用随机 capability token 和 `no-store`；不构成实时部署、专家 Skill 金标准或学习效果证明。操作边界见 `docs/private_local_dashboard.md` |
| 题目二自适应教学 Agent | 完成在线工程闭环；外部教学效果待验证 | 默认后端严格使用 `deepseek-v4-flash`，每次只消费一条自然语言反馈并生成一个下一教学动作；受约束控制器显式维护四维掌握、误解、当前理解信号和下一重点，支持主/支持 Skill 组合、自动切换、持续且受安全门约束的 `/+skill`、`/auto`、`/stop` 与成功/无进展终止。每个主 Skill 的 `supporting_skill_ids` 是硬组合 allowlist；机器门禁覆盖 ID/角色、适用信号、纠错证据、材料存在性、部分高风险阶段前置条件、主/支持 `max_repeat`、support allowlist 与 support 自身局部门禁，自动、手动和 fallback 共用安全子集；其余自然语言合同只进入提示与审计，不声称全部自动理解。六层上下文对较早历史只保留确定性统计与最多 6 条 evidence-linked 选择性 `teaching_checkpoints`；未知状态不推断，`omitted_turn_semantics_are_exhaustive=false`，不能称为完整历史语义总结。每轮问题带 `question_id + question_contract`；相关但未回答本问的短语保守记为 `partial / related_but_not_answer`，不能直接生成持久误解，也不增加掌握度。答案图片先做本机 OCR；低置信、无文字、公式/代码样或显式待确认附件即使伴随“答案见图/照片显然正确”，也强制为 `partial / ambiguous`、confidence 0，不提升掌握或解除误解；确认动作在无 support 的 `skill_self_explanation` 与 `skill_socratic_understanding_check` 间按门禁/连续 `max_repeat` 轮换。教师规范 claim 只有在绑定非空 `knowledge_components` 且与当前动作知识点相交时才具备确定性答案资格；多行 OCR 按完整答案匹配，跨知识点规范陈述命中会降为 `partial / related_but_not_answer`，不增加掌握度。后端默认保存最多 16 个隔离的内存 Session，画像切换采用 prepare-then-commit；刷新只用随机、无业务语义但仍可被同源脚本或 DevTools 读取的 opaque handle 恢复；默认内存 Session 在服务停止或重启后丢失，显式传入 `--session-store` 时使用本机追加式 JSONL 支持 cold resume。全部槽位正忙时拒绝新建而不破坏运行中会话；黑箱 HTTP runner 覆盖切换、失败回滚、恢复、并行 Session 与陈旧上下文拒绝；Playwright Chrome runner 另主动在 UI 外推进旧 Session，观察一次预期 400，再验证前端同步新 guards、换新 start 幂等键、恰好重试一次并成功切换，同时覆盖印刷文字 image-only、刷新和响应式路径；这只是一条受控竞态，不代表任意并发，也不把 Codex 内置浏览器可用性当作证据。运行库含 16 个原子 Skill，来自 neural-v1 九环节/十三策略执行本体，但 neural-v1 证据物化门禁为 0 eligible / 54 excluded，因此页面保持 PROVISIONAL。4 条结构化回归轨迹的决策匹配率为 0.916667，模拟增益差为 +16.9165；最新 28 条作者构造 development case 的 Signal Accuracy/Macro-F1 为 0.892857/0.875325，allowed-Skill hit 为 0.750000，switch F1 为 0.787879，P50/P95 为 994.887/1250.134 ms。历史私有 v2 development run19（strict terminal polarity + terminal guard 修复后，run fingerprint `0d1f0cc98c5cdef058f15a19791f2e25d031928cf472b7babbcd32889bac0cc7`）记录 recall/memory 1.000000/0.950000、误解解除 exact/evidence 1.000000/1.000000、allowed-Skill hit 0.350000、switch F1 0.631579、终止匹配 0.950000、注入阻断 1.000000、跨 session 泄漏 0；lifecycle receipt coverage/commit verification/route contract 为 1.000000/0.950000/0.950000，explicit replan rate 为 0.800000，bounded route completion 为 0.400000，runtime fallback totals（Loop/planner/action/assessment）为 0/0/0/0。冻结 run25 保留为第二个历史基线：allowed-Skill hit 0.450000、switch F1 0.818182、bounded completion 0.750000。相同 6 case / 20 turn development input/gold 指纹上的 run38–run42 的 allowed-Skill hit 为 1.00/1.00/0.95/0.95/0.95，switch F1 均为 0.916667，bounded completion 为 0.75/0.70/0.50/0.80/0.70；run40–run42 唯一 miss 均为 `cross_session_alpha/alpha_t2`，但 leakage 与四类 fallback 均为 0。所有这些数值均只是作者构造 development regression，不是 Accuracy、held-out、专家锁箱、部署准确率或真实学习效果；私有 receipt 不公开，lifecycle receipt 也未获外部签名 | `docs/teacher_agent_task2.md`、`scripts/run_teacher_agent_system_acceptance.py`、`打开题目二教学Agent.command` |
## 题目二新增工程层（当前状态）

本轮产品化补强：状态优先裁决现在识别带有唯一 active misconception 目标的 correction chain，优先安排 assessment/metacognition/review 验证，避免检索或换例子把纠错目标丢失；纠错链动作由确定性 materializer 接管，防止安全生成器在追问中复述教师答案。目标可提供 `knowledge_spec.misconception_catalog[*].aliases`，运行时先做 teacher-owned canonical tag 归一化，再进入 active/resolved 生命周期；未知标签仍保留并按原有 fail-closed 规则处理。该设计提升的是可审计的一致性，不是把模型诊断变成外部校准 Accuracy。

| 能力 | 状态 | 说明与边界 |
|---|:---:|---|
| 证据加权学生模型 | 完成工程能力；外部校准待完成 | `student_model.py` 以教师先验和 `confidence × answer_alignment_reliability` 累积四维证据，输出 `p_mastery`、`uncertainty`、证据指针和下一关注点。它是确定性的 Beta-like 工作估计，不是 ground truth、外部校准概率、Accuracy 或学习成绩。 |
| Goal→Plan→Execute→Verify→Reflect 编排 | 完成工程能力 | `teacher_agent_orchestration.py` 为五阶段生成带 SHA-256 的可恢复 checkpoint，默认至多一次重规划；高不确定性可要求人工复核。receipt 只保留阶段状态/短原因/公开 Loop 审计，不保存 prompt、思维链或学生原文。 |
| 产品 benchmark v2 | 完成开发验证管线 | `data/teacher_agent_benchmark_v2_development.json` 与独立 `*_gold.json` 共 6 case/20 turn；input fingerprint、gold 分离、predictions 禁止 gold 泄露均已测试。当前是作者构造 development split，不是 held-out、专家锁箱、真实学习效果或部署准确率。 |
| Gold-free 对话质量 benchmark | 完成评分接口；真实运行收据待生成 | 六学科作者构造输入只含 learner utterance、阶段合同和教师可见资源；评分重复、answer-first、解释增量、阶段闭环和资源 provenance。mastery/termination 必须绑定当前 learner text hash、evidence id、assessment eligibility 与实际 mastery application，不能相信自报标签。随附 predictions 未获 runtime attestation，不作为当前 DeepSeek 正向成绩。 |
| command bar 与 recovery console | 完成工程/UI 验收 | `GOAL / PLAN / PROGRESS / CONTROL`、`/auto`、`/+skill`、`/stop`、草稿保留重试与转人工已接入；默认 Console 使用 Harness 显式 cancel API，取消已建立的 provider socket，并以 commit-wins 门禁防止取消覆盖已提交结果。stale guard、幂等键和 strict SSE reducer 防止旧响应覆盖新会话。浏览器证据仅覆盖受控 Chrome 路径。 |

补充边界：题目二默认 Session 只在服务进程内存中，进程停止/重启后会丢失；显式 `--session-store` 才启用本机 JSONL cold resume。Harness stream journal 支持同进程 cursor reconnect；持久 journal 可跨进程重放 terminal run。Teach active restart 只允许 `start_pre_provider_only` 安全恢复；start/step 的 provider effect 结果未知时不重放而 `fail_closed_handoff`，`harness_teach_student_step_replayed_after_restart=false`；领域 turn 已 commit 时可从权威 Session/幂等 receipt 补齐外层 SSE。`harness_chat_active_restart_recovery_enabled=false`，所以 Chat 非终态不自动恢复。生产 dashboard 开启 state-first route adjudication 与 action-only repair；连续性由 evidence-linked `continuity_recall` 约束，找不到匹配记录就请学生重述。多轮请求统计采用 `request_accounting_scope=completed_committed_turns_only`，repair 不是第二个 validated plan 但属于第二个 logical request。兼容旧同步路径仍是 `remote_transport_cancellation_supported=false`；默认 Console 的 Harness 显式取消为 `harness_sse_transport_cancellation_supported=true`，可关闭已经建立的 DeepSeek 文本/Web Search 响应，并由 commit-wins 门禁保证已提交结果不被后到 cancel 改写。`/stop` 才终止整个 Session。独立 Chrome DOM 验收仍只覆盖受控 Chrome 路径，不建立 Firefox/Safari 或屏幕阅读器结论。此处的 OCR“确认识别文字”只表示学生确认/修订转写，不等于答案正确。

题目二当前工程契约为 V18（`teaching_agent_assess_route_act_v18_direct_teaching_cache_stable_prefix`）。teach-first 首次学生可见动作直接进入教师来源约束的概念解释，内部阶段政策、来源证据卡、Skill 路由与工具生命周期不进入对话；当前练习/核验/迁移的最终解仍受门禁。学生明确说不会时，系统把当前问题即时绑定为失败任务，必须由教师换表征讲解或示范后再做低负担确认，且不更新掌握度；学习者澄清必须先回答并绑定允许的教师来源，材料不足时保持问题开放；`action_only_repair` 只允许固定路由下修复当前动作，`continuity_recall` 必须绑定历史证据；纠错链的 canonical taxonomy 与 teacher-claim contract 只允许在唯一目标、高置信、学生原话证据满足时参与归一化；这些字段和 `cancel_turn` 的边界以生产 dashboard 实现为准。

V18 采用“固定 system/完整 Skill 合同前缀 + 动态 turn 后缀”以使用 DeepSeek 自动精确前缀缓存；这是与 DeepSeek-Reasonix 相关的 cache-aware 原则实现，不是正式 Reasonix 集成或自建 KV 服务。布局指纹只用于审计，实际缓存计数来自 provider `usage` 的 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 并贯穿运行 trace；字段缺失时不推算命中率。

| 2 门课×5 节离线演示 | 完成 | `tsm demo` |
| 2 门课×5 节完整正式字幕 | 完成来源与时间轴闭环 | `tsm fetch-formal-captions` 实跑 10/10；官方页面、WebVTT/媒体文件 SHA-256、媒体时长与首尾时间轴覆盖已绑定；私有 manifest 的 `formal_empirical_ready=true`。该状态不表示字幕正文经过独立人工内容准确率审计 |
| 2 门课×5 节完整 MIT OCW 媒体 | 完成 | 10/10 私有下载、本地 SHA-256、FFprobe 容器/音视频流/参考时长验证；1,038,813,006 bytes、26,656.83 秒；公开 receipt 不含媒体内容 |
| 视频/字幕或 ASR / Skill / 教学过程 / 状态机 | 完成 | `tsm pipeline --transcript` 提供无 Whisper 的真实视频闭环；同媒体哈希 ASR 可作为 `speech` provenance 使用。历史字段 `audio_content_verified=true` 只表示 ASR 产物与同一媒体 SHA-256 绑定，不表示文字经人工核对，也不建立 WER |
| 完整视频音频、抽帧、OCR 与事件 | 完成工程链路 | v9/v2：10/10 全时间轴覆盖和字幕—媒体对齐；2553 帧、2441 非空 OCR 帧、1964 帧至少 3 词、40191 接受词、1974 视觉事件、2270 融合事件 |
| CLIP 视觉语义 | 完成工程链路 | wheel `visual` extra 和三个 `tsm visual-semantic-*` 入口；2553/2553 哈希绑定帧；512 维嵌入、八类相对 prompt 分数、模型 revision/权重 SHA/CPU 环境绑定；没有人工真值，不建立 Accuracy |
| transcript / +audio / +visual / full 消融 | 完成内部分析 | 10 讲完全相同 transcript segments，排除课堂观察；平均内部 Skill 分 95.37/95.37/95.30/95.32，不构成多模态增益 |
| 多模态公开聚合 receipts | 完成工程边界 | `full_multimodal_validation_receipt.json` 与 `multimodal_ablation_receipt.json` 只保留聚合、设计、结论边界和私有来源哈希承诺；发布仍需 release audit 与人工披露复核 |
| 10 讲端到端总 runner | 完成工程入口 | `run_full_video_multimodal_study.sh` 串联获取/验证、可恢复长视频处理、CLIP、审计、消融、公开 receipts 与 release audit；默认 CPU，可通过环境变量选择授权 GPU |
| TeachObs 外部人工标签 | 完成导入与审计 | commit-pinned v0.1：30 讲、5158 场景、39 个共识标签、官方 23/7 划分；来源报告 7 名独立编码员，但无逐编码员文件，不能重算 κ |
| TeachObs 旧发布文本 transcript-only 基线 | 完成探索性实跑 | repository 随附发布文本、1,312 个官方测试场景：Micro-F1 0.612968、Macro-F1 0.360044；它不是当前平台字幕/审计 ASR 物化文本的四臂结果，Hamming accuracy 0.826180 也不单独表述为 Accuracy |
| TeachObs 字幕与转写时间轴审计 | 论文 profile 已覆盖，完整 profile 部分完成 | 平台字幕审计为 23/30 讲、34 轨（creator-provided 19、automatic 15），平均时间轴覆盖 0.985841；其中 23 讲进入优先级 matrix，另六讲 GPU ASR 已全部通过技术导入，形成 creator-provided 19 + automatic 4 + audited ASR 6 = 29/30 讲，唯一 pending 为 S4。时间轴与 provenance 不是 WER，字幕/ASR 内容准确率仍未建立 |
| TeachObs 审计 ASR GPU 交接 | 完成 fail-closed 工程链，六讲已实跑导入 | `prepare/import-teachobs-asr-*` 与离线 GPU runner 固定媒体/模型/解码/runtime/segment；v2/v4/v4 契约对 hash-bound 完整媒体单次输入采用统一的 VAD 相对端点策略（span≥0.90、首尾各≤媒体时长 0.10、绝对秒数只诊断），旧证据失败关闭且整批重跑、不迁移。当前六讲均为有效结果；`teachobs_asr_receipt.json` 每次由当前 media manifest、caption audit、job/import audit 重算，其 aggregate 和 `handoff_status` 仍是权威状态。ASR 不是官方字幕，WER/内容准确率仍只能由独立人工参考建立 |
| TeachObs 双人独立标注包 | 完成 fail-closed 骨架 | `prepare/analyze-teachobs-double-annotation`；A/B 两份空标签表使用不同盲化顺序，论文 profile 为每人 4,945 行×39 标签，完整 profile 为 5,158 行×39；发布的 39 个定义全空，须外部 39/39 operational codebook 和真实两人回填后才计算 κ，当前 human_completion=false |
| TeachObs 完整视频四臂 F1 | 完成正式探索性实跑 | 29 讲/4,945 场景完成媒体、字幕/ASR、音频、OCR、CLIP 与四臂对齐；固定六讲 1,099 场景上，transcript/+audio/+visual/full 的 Micro-F1 为 0.594496/0.604708/0.548328/0.543812，Macro-F1 为 0.244338/0.247501/0.303894/0.309604，Hamming 为 0.818833/0.809967/0.775297/0.763258。full 相对 text 的 Macro-F1 增量 +0.065266（lesson-cluster 95% CI [0.014886,0.095637]），但 Micro/Hamming 明显下降。修订仅在 23 讲内做 5 折 lesson-grouped OOF 并记录 frozen v2 provenance；首轮公开测试结果已观察，因此是 post-test iterative exploratory，不是首次盲测、0.9、确认性或部署结果 |
| TeachObs 新站点确认性四臂预注册 | 完成工程草案；外部执行未发生 | 精确绑定 system/analysis/四臂模型文件 SHA、同样本主 contrast、39 标签 Macro-F1、classroom/session cluster、固定 2,000 次 bootstrap、coverage、前瞻/一次性与外部 Ed25519 登记门。当前没有外部 registration、目标数据、结果签名或一次性消费 receipt，`external_lockbox_established=false`；公开 TeachObs 23/7 被机器规则排除 |
| 九个规范教学环节蒸馏 | 完成工程能力；识别准确率未建立 | `teaching_phases.py` 按题目 4.2 定义九个候选环节；`mine_skill` 仅把有证据的 `observed_method` 子序列按首次证据时间排序，带时间戳、命中线索和 evidence_id；未观察到的 `recommended_enrichment` 只用于补齐可执行 scaffold，不声称来自教师或具有教师时序。10 份演示 transcript 实际观察到 4–7 个环节；当前无独立 phase gold |
| 跨讲次通用 Skill 蒸馏 | 完成可执行 v0；神经 v1 已训练但严格物化失败 | `distill-general-skill` 对 2 门课×5 讲的合法单讲 Skill 按 unique lecture 二元投票，以总支持率≥0.80且每课支持率≥0.60建立课程平衡共识，recommended 永不投观察票；正式十讲得到 6 个跨课程策略、5 个 observed-consensus 环节和 4 个 recommended 环节，并可用 `apply-general-skill` 迁移到新主题。另一路 neural v1 已在 turdy 对 10 个完整视频运行 VideoMAE/WavLM/XLM-R/LayoutLMv3，并完成 5 折×30 epoch、10 讲 exact-once OOF 与最终真实性审计；但 54 个 OOF 原子全部为 uncertain，物化候选只有 0 observed + 9 recommended，strict structural audit 为 `passed=false`，因此 v0 仍是当前可演示的通用 Skill。弱标签 OOF 指标不是专家 gold Accuracy、部署性能或教学效果；详见 `TRAINING_RESULT_20260804.md` |
| 自动结构与证据一致性评估 | 完成 | `tsm evaluate`；七个维度加权（结构 12%、证据 18%、可执行 18%、方法忠实度 22%、教学质量 12%、迁移 9%、溯源 9%）。除 `method_fidelity` 外的其余六个维度检查生成器按构造必然满足的字段，10 份演示 Skill 上标准差为 0.000；`method_fidelity` 改为从被引用的 evidence 记录反推每个 observed 步骤的主张，实际区分度为 83.3–89.8（总分 92.3–93.7） |
| 评分维度的可证伪性 | 完成负控 | `tests/test_method_fidelity.py` 用五种降级（纯模板、伪造线索、打乱时间段、删证据、塌缩环节）对深拷贝 Skill 打分并断言严格下降，且把每种降级绑定到应当检出它的那个分量；把 `method_fidelity` 钉成 100.0 会让 8 个测试失败 |
| 文档数值与量表防陈旧 | 完成回归门 | `tests/test_documentation_freshness.py` 从公开四臂 receipt 和 `DIMENSION_WEIGHTS` 反向核对 README、公开产物说明、设计文档和状态矩阵，防止重新评分后仍发布旧分数或漏写 `method_fidelity` |
| observed method 与推荐脚手架区分 | 完成 | `method_provenance` |
| 人工复核覆盖与版本审计机制 | 完成工程能力 | 复核完成后强制 100% 覆盖、不同 reviewer、canonical `skill_fingerprint` 绑定；当前真实评分仍待完成 |
| DIPSER 数据/特征防篡改 | 完成 | runner 重算指纹 + tamper tests |
| checkpoint v3 与签名外部锁箱协议 | 完成工程协议；外部执行未发生 | 特征/ClaimContract/claim-cluster 绑定、Ed25519 登记、一次性 ledger、签名 receipt 的代码与校验器已实现；当前没有外部 registration、目标数据、结果签名或一次性消费 receipt |
| 外部研究证据接入 | 完成协议 | 确认性多模态增益/真实学习效果聚合 manifest、重算 gate、system artifact 实算绑定、独立签名与各自可信公钥；当前无正向证据 |
| 学习效果 cluster RCT 工具 | 完成 fail-closed 工程链 | `prepare/analyze-learner-effect-study` 生成 teacher/classroom 预注册、仅哈希 token 分配与盲化 pre/post/retention 表，并执行固定 ITT ANCOVA 和 cluster bootstrap；模板、合成与未签名本地分析均保持 learner_effectiveness_established=false |
| 原始目标站点媒体/传感器到 strict feature bundle | 完成 | `extract-strict-features` 绑定 raw hashes/windows、代码、配置、运行时与工具 provenance；只提取不预测 |
| 环境诊断 | 完成 | `tsm doctor` |
| 发布隐私与证据新鲜度审计 | 完成工程检查 | `tsm release-audit` 检查公开边界；acceptance 的 verification scope 绑定完整 `teaching_skill_miner/` 源码、看板、测试、脚本、schema/config、README/治理文档与研究说明。`verify_project.sh` 在 TeachObs 私有输入存在时还会重验 annotation/caption/ASR receipt；四臂产物必须与冻结 bundle 同时通过。项目 receipt 记录本次用到的私有/公开证据 SHA，最终 acceptance 在发布前再次比较；这些都不等于完整去标识或不可重识别证明 |
| CI 与隔离 wheel smoke | 本地机器收据与远端 CI 分离 | 当前测试数、subtest 数、wheel 哈希与成员只从 `build_release_acceptance.sh` 生成的机器收据读取，避免文档数字漂移；验证包含全量测试、双构建、字节一致性、exact-wheel 安装/入口 smoke 与发布审计。GitHub Actions 只作为推送提交的独立远端检查；`remote_github_actions_run_verified=false` 表示远端状态不进入本地证明链 |

## 项目内不能伪造完成的外部证据

| 事项 | 当前状态 | 完成条件 |
|---|:---:|---|
| 10/10 双人独立教学复核 | 待完成 | 真实复核者评分、意见、κ 与复评记录 |
| 生成 Skill 提高学习效果 | 未建立 | 伦理审批后的前后测或对照/A-B 实验；可在完成后接入签名外部 learner evidence |
| 确认性多模态增益 | 未建立 | 冻结 pipeline 后的新数据配对消融与置信区间；可在完成后接入签名 external multimodal evidence |
| 跨学校/部署准确率 | 未建立 | 独立目标站点、前瞻一次性锁箱、全部 claim gate 通过 |
| 可信部署 0.9 | 未建立 | 不能继续调同一开发集取得，必须由上述外部锁箱建立 |

当前准确定位是：**核心工程、10/10 正式字幕与完整媒体、TeachObs 29 讲媒体/字幕或审计 ASR/音频/视觉/OCR/CLIP、固定六讲四臂探索性识别指标和安全冻结均已真实运行、可复现和可审计；真实双人复核、因果学习效果、确认性多模态增益、跨站点与外部部署结论仍明确待真实证据完成。** `formal_empirical_ready=true` 只代表正式转写来源与覆盖门槛通过，`multimodal_empirical_ready=true` 只代表所需模态、全程覆盖、对齐和事件结构齐全；两者本身都不是 Accuracy。当前 TeachObs 数值也只是公开测试上的 post-test exploratory estimate。签名机制只验证由指定密钥签署的内容完整性，不能自动证明签署者独立；可信公钥必须来自开发团队之外的治理渠道。
