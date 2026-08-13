import "reflect-metadata";

import assert from "node:assert/strict";
import {chmod, mkdtemp, readFile, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import test from "node:test";

import type {AppConfigService} from "../src/config/app-config.service";
import {
  HarnessGatewayError,
  HarnessWorkerPoolService
} from "../src/harness/harness-worker-pool.service";

const PROCESS_LIMIT_POLICY = {
  schema: "teaching_skill_miner.worker_process_resource_limits.v1",
  address_space_bytes: 1_610_612_736,
  file_size_bytes: 536_870_912,
  open_files: 256,
  core_dump_bytes: 0
} as const;

function fakeWorker(observation: string, mode: "valid" | "weak" | "extra"): string {
  return `#!${process.execPath}
const fs = require("node:fs");
const http = require("node:http");
let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => { raw += chunk; });
process.stdin.on("end", () => {
  const bootstrap = JSON.parse(raw);
  fs.writeFileSync(${JSON.stringify(observation)}, JSON.stringify(bootstrap));
  const server = http.createServer((_request, response) => {
    response.writeHead(200, {"content-type": "application/json"});
    response.end("{}" + "\\n");
  });
  server.listen(0, "127.0.0.1", () => {
    const limits = {
      schema: "teaching_skill_miner.worker_process_resource_limits_status.v1",
      enforcement: "linux_rlimit_v1",
      address_space_bytes: ${JSON.stringify(mode)} === "weak" ? 1073741823 : 1610612736,
      file_size_bytes: 536870912,
      open_files: 256,
      core_dump_bytes: 0,
      cpu_time_limit: "not_set_long_lived_worker_uses_request_wall_clock",
      process_count_limit: "not_set_shared_uid_unsafe",
      scope: "per_process_individual_inherited_not_process_tree_aggregate"
    };
    if (${JSON.stringify(mode)} === "extra") limits.private_path = "/private/scope";
    process.stdout.write(JSON.stringify({
      schema: "teaching_skill_miner.gateway_worker_status.v1",
      status: "ready",
      worker_id: bootstrap.worker_id,
      scope_key_version: bootstrap.scope_key_version,
      port: server.address().port,
      backend: bootstrap.agent_backend,
      filesystem_isolation: "linux_landlock_scope_allowlist_abi_3",
      process_resource_limits: limits
    }) + "\\n");
  });
  process.on("SIGTERM", () => server.close(() => process.exit(0)));
});
`;
}

function config(root: string, cwd: string, python: string): AppConfigService {
  return {
    nodeEnv: "test",
    harnessGatewayEnabled: true,
    harnessWorkerRoot: root,
    harnessWorkerCwd: cwd,
    harnessWorkerPython: python,
    harnessWorkerBackend: "deterministic",
    harnessProviderApiKeyFile: undefined,
    harnessRemoteProviderPolicy: undefined,
    harnessSafeguardingLocale: "zh-CN",
    harnessSafeguardingDispatchConfigured: false,
    harnessWorkerFilesystemIsolationRequired: true,
    harnessWorkerProcessResourceLimits: PROCESS_LIMIT_POLICY,
    harnessScopeKeyVersion: "k1",
    harnessScopeKeys: [{
      version: "k1",
      secret: "resource-limit-test-scope-secret-at-least-32-characters"
    }],
    harnessMaxWorkers: 1,
    harnessWorkerIdleTimeoutMs: 60_000,
    harnessWorkerStartupMs: 2_000,
    harnessWorkerShutdownMs: 500
  } as unknown as AppConfigService;
}

async function fixture(mode: "valid" | "weak" | "extra") {
  const temporary = await mkdtemp(join(tmpdir(), "teachlab-worker-limit-ts-"));
  const observation = join(temporary, "bootstrap.json");
  const executable = join(temporary, "fake-worker");
  await writeFile(executable, fakeWorker(observation, mode), {mode: 0o700});
  await chmod(executable, 0o700);
  return {
    temporary,
    observation,
    pool: new HarnessWorkerPoolService(
      config(join(temporary, "workers"), temporary, executable)
    )
  };
}

test("worker bootstrap privately carries the exact production process-limit policy", async () => {
  const current = await fixture("valid");
  try {
    const response = await current.pool.request(
      {tenantId: "tenant-a", ownerId: "learner-a"},
      {method: "GET", path: "api/bootstrap"}
    );
    response.resume();
    await new Promise<void>((resolve, reject) => {
      response.once("end", resolve);
      response.once("error", reject);
    });
    const bootstrap = JSON.parse(await readFile(current.observation, "utf8"));
    assert.deepEqual(bootstrap.process_resource_limits, PROCESS_LIMIT_POLICY);
    assert.equal(JSON.stringify(current.pool.status()).includes("private"), false);
  } finally {
    await current.pool.onApplicationShutdown();
    await rm(current.temporary, {recursive: true, force: true});
  }
});

test("worker status rejects weak limits and any unrecognized process-limit field", async (context) => {
  for (const mode of ["weak", "extra"] as const) {
    await context.test(mode, async () => {
      const current = await fixture(mode);
      try {
        await assert.rejects(
          current.pool.request(
            {tenantId: "tenant-a", ownerId: "learner-a"},
            {method: "GET", path: "api/bootstrap"}
          ),
          (error: unknown) =>
            error instanceof HarnessGatewayError &&
            error.message === "harness_worker_unavailable"
        );
      } finally {
        await current.pool.onApplicationShutdown();
        await rm(current.temporary, {recursive: true, force: true});
      }
    });
  }
});
