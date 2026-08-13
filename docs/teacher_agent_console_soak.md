# Teaching Console/Harness 资源预算与 soak 验收

`scripts/run_teacher_agent_console_soak.py` 是一个独立、可重复、只访问回环地址的资源验收器。它不导入 Console/API 实现，也不启动或停止已有服务。默认目标是验收器自己创建的随机端口 fixture；真实 Console/Harness 由操作者单独启动并显式提供 URL、根进程和磁盘边界。

## 三种运行方式

依赖无关的 CI self-test：

```bash
python scripts/run_teacher_agent_console_soak.py --self-test
```

该模式固定执行 6 个请求和 7 次真实 OS 进程采样，通常不到 2 秒。fixture 是独立进程组，只从 49152–65535 中选择随机高位端口，绝不绑定或请求 3030。它证明采样、预算、hash receipt 和清理闭环可用，不证明真实 Console/Harness 的资源表现；收据固定包含 `long_soak_executed: false` 和 `fixture_is_real_console_or_harness: false`。

一分钟隔离 smoke：

```bash
python scripts/run_teacher_agent_console_soak.py \
  --mode smoke \
  --duration-seconds 60 \
  --receipt .private/receipts/console-soak-smoke.json
```

八小时模式只能使用固定的 28,800 秒测量窗口，不能用 `--duration-seconds` 缩短：

```bash
python scripts/run_teacher_agent_console_soak.py \
  --mode 8h \
  --receipt .private/receipts/console-soak-8h.json
```

模式名称不产生长时结论。只有实际测量时长达到 28,800 秒，`long_soak_executed` 才会为 `true`；通过全部预算且未中断时，`long_soak_completed` 才会为 `true`。

## 验收真实本地目标

真实目标必须满足以下边界：

- URL 是显式的 `http://127.0.0.1:<port>/...` 或 `http://[::1]:<port>/...`；不接受 DNS 名、HTTPS、认证信息、query、fragment、路径逃逸或端口 3030。
- `--target-pid` 是该 Console/Harness 的根进程。验收器采样它及每次采样时的全部后代，但绝不会向该外部进程发送信号。
- `--runtime-id` 是非敏感发布身份，例如 release ID；收据只保存它的 SHA-256，并将其标注为操作者声明，而不是自动验证的身份。
- 至少一个 `--disk-path` 是本次目标允许增长的存储边界。扫描只统计普通文件逻辑字节，不跟随符号链接，也不输出路径或文件名。
- 建议把含 capability 的 URL 放在 mode-0600 私有文件，通过 `--target-url-file` 读取，避免把它放进命令行历史。

示例：

```bash
python scripts/run_teacher_agent_console_soak.py \
  --mode smoke \
  --target-url-file .private/console-soak-target.url \
  --target-pid 12345 \
  --runtime-id console-release-1.2.0 \
  --disk-path .private/console-soak-data \
  --request-path '' \
  --receipt .private/receipts/real-console-smoke.json
```

如果希望同时覆盖 UI，可追加 `--browser chromium --browser-tabs 1`。浏览器是验收器自有的独立进程组；收据分别报告其 RSS、FD、子进程和 tab 数。浏览器 renderer 不能可靠映射到单一 tab，因此收据不会声称 per-tab RSS。未启用浏览器时，收据明确写明只测量 HTTP 目标进程资源。Playwright 和对应浏览器可执行文件是可选依赖，缺失会以固定错误码失败，不会悄悄降级成 HTTP-only 通过。

## 指标和默认预算

每个采样点读取主机 `ps` 进程表。Linux FD 来自 `/proc/<pid>/fd`；其他 POSIX 主机使用 `lsof`，不可用时 FD 预算失败关闭。收据保留每个匿名化进程引用的峰值和增长聚合，并保存完整安全采样序列的 SHA-256，而不暴露 PID 或命令行。

默认通过条件为：

- GET 请求 p95 不超过 1,000 ms，失败率不超过 1%；
- 目标组峰值 RSS 不超过 512 MiB，峰值相对基线增长不超过 64 MiB；
- 目标组峰值 FD 不超过 512，相对基线增长不超过 64，FD 可测覆盖率至少 90%；
- 目标子进程不超过 16；
- 指定磁盘边界峰值增长不超过 64 MiB；
- 启用浏览器时，浏览器组默认峰值 RSS/增长不超过 1,024/256 MiB，峰值 FD/增长不超过 2,048/256，子进程不超过 32，并要求配置的 tab 仍存活、没有外部网络请求。

所有预算都有对应的 `--max-*` 参数。长时验收应在冻结配置后运行，并保留 `configuration.content_sha256`、`budgets.content_sha256`、`runtime.identity_sha256`、`sampling.sample_series_sha256` 和顶层 `integrity.content_sha256`。

## 中断和收据边界

SIGINT/SIGTERM 会停止新请求，随后只向验收器自己创建的 fixture/browser 进程组发送 TERM，必要时发送 KILL，并等待回收。外部目标始终标记为 `external_target_signalled: false`。中断收据为 `status: interrupted`，退出码分别是 130/143。

收据不包含目标 URL、capability、PID、命令、磁盘路径、响应正文或原始异常。`verify_receipt()` 会同时校验顶层内容 hash 和采样配置、预算、运行时身份投影的 hash。该 hash 提供内容绑定与篡改检测，不是签名，也不替代发布签名或独立见证。

## CI 边界

现有 CI 的 `pytest` 自动发现 `tests/test_teacher_agent_console_soak.py`。测试只运行快速确定性 self-test、纯校验测试和一次快速中断清理测试；不会运行一分钟 smoke、浏览器下载或八小时 soak。八小时结果必须来自明确的人工/专用 runner 执行，不能由 self-test 收据推断。
