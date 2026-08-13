import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {chmod, mkdir, mkdtemp, readdir, rm, writeFile} from "node:fs/promises";
import {realpathSync} from "node:fs";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";
import {test} from "node:test";

import type {AppConfigService} from "../src/config/app-config.service";
import {
  HarnessGatewayError,
  HarnessWorkerPoolService
} from "../src/harness/harness-worker-pool.service";

process.env.NODE_ENV = "test";

function findModernPython(): string {
  for (const command of ["python3.13", "python3.12", "python3.11", "python3.10", "python"]) {
    try {
      const located = execFileSync("/usr/bin/which", [command], {
        encoding: "utf8"
      }).trim();
      execFileSync(
        located,
        ["-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"],
        {stdio: "ignore"}
      );
      return realpathSync(located);
    } catch {
      // Try the next explicit executable.
    }
  }
  throw new Error("A Python >=3.10 runtime is required for the worker canary test");
}

function config(options: {
  workerRoot: string;
  workerCwd: string;
  python: string;
  apiKeyFile: string;
}): AppConfigService {
  return {
    nodeEnv: "production",
    harnessGatewayEnabled: true,
    harnessWorkerRoot: options.workerRoot,
    harnessWorkerCwd: options.workerCwd,
    harnessWorkerPython: options.python,
    harnessWorkerBackend: "deepseek",
    harnessProviderApiKeyFile: options.apiKeyFile,
    harnessRemoteProviderPolicy: {
      policy_id: "deepseek-production-terms",
      policy_version: "2026-08-12",
      policy_source:
        "deployment_operator_asserted_external_terms_not_repository_verified",
      processing_region: "provider_managed",
      provider_retention_days: 30,
      deletion_status:
        "outside_service_control_subject_to_provider_policy",
      documentation_url: "https://provider.example/privacy"
    },
    harnessSafeguardingLocale: "zh-CN",
    harnessWorkerFilesystemIsolationRequired: false,
    harnessScopeKeyVersion: "k1",
    harnessScopeKeys: [
      {
        version: "k1",
        secret: "runtime-canary-test-scope-secret-at-least-32-characters"
      }
    ],
    harnessMaxWorkers: 16,
    harnessWorkerIdleTimeoutMs: 900_000,
    harnessWorkerStartupMs: 5_000,
    harnessWorkerShutdownMs: 2_000
  } as AppConfigService;
}

test("production readiness coalesces an exact no-network worker canary and removes its random scope", async () => {
  const temporary = await mkdtemp(join(tmpdir(), "teachlab-runtime-canary-"));
  const workerRoot = join(temporary, "workers");
  const apiKeyFile = join(temporary, "provider-key");
  await writeFile(apiKeyFile, "test-provider-key-without-whitespace\n", {
    mode: 0o600
  });
  await chmod(apiKeyFile, 0o600);
  const workers = new HarnessWorkerPoolService(
    config({
      workerRoot,
      workerCwd: resolve(process.cwd(), "../.."),
      python: findModernPython(),
      apiKeyFile
    })
  );
  let providerReadinessCalls = 0;
  workers.setProviderReadinessProbeForTesting(async () => {
    providerReadinessCalls += 1;
  });
  try {
    await Promise.all([
      workers.assertReady(),
      workers.assertReady(),
      workers.assertReady()
    ]);
    const status = workers.status();
    assert.equal(status.runtimeCanaryRequired, true);
    assert.equal(status.runtimeCanaryLastResult, "ready");
    assert.equal(status.runtimeCanaryAttempts, 1);
    assert.equal(status.runtimeCanarySuccesses, 1);
    assert.equal(status.runtimeCanaryFailures, 0);
    assert.equal(status.runtimeCanaryRemoteProviderNetworkValidated, false);
    assert.equal(status.runtimeCanaryPersistentTenantDataCreated, false);
    assert.equal(status.providerReadinessRequired, true);
    assert.equal(status.providerReadinessLastResult, "ready");
    assert.equal(status.providerReadinessAttempts, 1);
    assert.equal(status.providerReadinessSuccesses, 1);
    assert.equal(status.providerReadinessFailures, 0);
    assert.equal(status.providerReadinessLearnerContentSent, false);
    assert.equal(status.providerReadinessGenerationCreated, false);
    assert.equal(providerReadinessCalls, 1);
    assert.equal(status.readyWorkers, 0);
    assert.deepEqual(await readdir(workerRoot), []);

    await workers.assertReady();
    assert.equal(workers.status().runtimeCanaryAttempts, 1);
    assert.equal(workers.status().providerReadinessAttempts, 1);
    assert.equal(providerReadinessCalls, 1);
    assert.deepEqual(await readdir(workerRoot), []);
  } finally {
    await workers.onApplicationShutdown();
    await rm(temporary, {recursive: true, force: true});
  }
});

test("a runtime-path startup error is sanitized, cached briefly, and leaves no canary process or scope", async () => {
  const temporary = await mkdtemp(join(tmpdir(), "teachlab-runtime-canary-fail-"));
  const workerRoot = join(temporary, "workers");
  const emptyCwd = join(temporary, "empty-runtime");
  const apiKeyFile = join(temporary, "provider-key");
  await Promise.all([
    writeFile(apiKeyFile, "test-provider-key-without-whitespace\n", {mode: 0o600}),
    mkdir(emptyCwd, {mode: 0o700})
  ]);
  await chmod(apiKeyFile, 0o600);
  const workers = new HarnessWorkerPoolService(
    config({
      workerRoot,
      workerCwd: emptyCwd,
      python: findModernPython(),
      apiKeyFile
    })
  );
  try {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      await assert.rejects(
        workers.assertReady(),
        (error: unknown) =>
          error instanceof HarnessGatewayError &&
          error.statusCode === 503 &&
          error.message === "harness_worker_unavailable" &&
          !error.message.includes(emptyCwd) &&
          !error.message.includes(apiKeyFile)
      );
    }
    assert.equal(workers.status().runtimeCanaryLastResult, "failed");
    assert.equal(workers.status().runtimeCanaryAttempts, 1);
    assert.equal(workers.status().runtimeCanaryFailures, 1);
    assert.deepEqual(await readdir(workerRoot), []);
  } finally {
    await workers.onApplicationShutdown();
    await rm(temporary, {recursive: true, force: true});
  }
});
