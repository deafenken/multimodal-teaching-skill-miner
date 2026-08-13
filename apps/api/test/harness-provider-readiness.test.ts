import "reflect-metadata";

import assert from "node:assert/strict";
import {chmod, mkdtemp, readFile, readdir, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import test from "node:test";

import type {AppConfigService} from "../src/config/app-config.service";
import {
  HarnessGatewayError,
  HarnessWorkerPoolService
} from "../src/harness/harness-worker-pool.service";


interface PoolInternals {
  runtimeCanaryCache: {ready: boolean; expiresAtMs: number};
  runtimeCanaryLastResult: "ready";
}

function fakePython(
  observationPath: string,
  mode: "ready" | "extra_field" | "learner_content_true"
): string {
  return `#!${process.execPath}
const fs = require("node:fs");
let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { raw += chunk; });
process.stdin.on("end", () => {
  const bootstrap = JSON.parse(raw);
  fs.appendFileSync(${JSON.stringify(observationPath)}, JSON.stringify({
    argv: process.argv.slice(2),
    bootstrap,
    environment_keys: Object.keys(process.env).sort()
  }) + "\\n");
  const status = {
    schema: "teaching_skill_miner.gateway_worker_status.v1",
    status: "provider_ready",
    backend: "deepseek",
    credential_validated: true,
    provider_network_validated: true,
    configured_model_available: true,
    learner_content_sent: ${JSON.stringify(mode)} === "learner_content_true",
    generation_created: false,
    persistent_tenant_data_created: false
  };
  if (${JSON.stringify(mode)} === "extra_field") status.model = "private-model";
  process.stdout.write(JSON.stringify(status) + "\\n");
});
`;
}

function config(
  workerRoot: string,
  workerCwd: string,
  python: string,
  apiKeyFile: string
): AppConfigService {
  return {
    nodeEnv: "production",
    harnessGatewayEnabled: true,
    harnessWorkerRoot: workerRoot,
    harnessWorkerCwd: workerCwd,
    harnessWorkerPython: python,
    harnessWorkerBackend: "deepseek",
    harnessProviderApiKeyFile: apiKeyFile,
    harnessRemoteProviderPolicy: {
      policy_id: "deepseek-production-terms",
      policy_version: "2026-08-12",
      policy_source:
        "deployment_operator_asserted_external_terms_not_repository_verified",
      processing_region: "provider_managed",
      provider_retention_days: 30,
      deletion_status: "outside_service_control_subject_to_provider_policy",
      documentation_url: "https://provider.example/privacy"
    },
    harnessSafeguardingLocale: "zh-CN",
    harnessSafeguardingDispatchConfigured: false,
    harnessWorkerFilesystemIsolationRequired: false,
    harnessScopeKeyVersion: "k1",
    harnessScopeKeys: [{
      version: "k1",
      secret: "provider-readiness-scope-secret-at-least-32-characters"
    }],
    harnessMaxWorkers: 2,
    harnessWorkerIdleTimeoutMs: 60_000,
    harnessWorkerStartupMs: 2_000,
    harnessWorkerShutdownMs: 500
  } as unknown as AppConfigService;
}

async function fixture(mode: "ready" | "extra_field" | "learner_content_true") {
  const temporary = await mkdtemp(join(tmpdir(), "teachlab-provider-ready-"));
  const workerRoot = join(temporary, "workers");
  const observation = join(temporary, "observation.ndjson");
  const python = join(temporary, "fake-python");
  const apiKey = join(temporary, "provider-key");
  await Promise.all([
    writeFile(python, fakePython(observation, mode), {mode: 0o700}),
    writeFile(apiKey, "provider-key-never-output\n", {mode: 0o600})
  ]);
  await Promise.all([chmod(python, 0o700), chmod(apiKey, 0o600)]);
  const pool = new HarnessWorkerPoolService(
    config(workerRoot, temporary, python, apiKey)
  );
  const internals = pool as unknown as PoolInternals;
  internals.runtimeCanaryCache = {
    ready: true,
    expiresAtMs: Date.now() + 60_000
  };
  internals.runtimeCanaryLastResult = "ready";
  return {temporary, workerRoot, observation, pool};
}

test("production DeepSeek readiness is content-free, coalesced, cached, and cleans its scope", async () => {
  const current = await fixture("ready");
  try {
    await Promise.all([
      current.pool.assertReady(),
      current.pool.assertReady(),
      current.pool.assertReady()
    ]);
    const status = current.pool.status();
    assert.equal(status.providerReadinessRequired, true);
    assert.equal(status.providerReadinessLastResult, "ready");
    assert.equal(status.providerReadinessAttempts, 1);
    assert.equal(status.providerReadinessSuccesses, 1);
    assert.equal(status.providerReadinessFailures, 0);
    assert.equal(status.providerReadinessLearnerContentSent, false);
    assert.equal(status.providerReadinessGenerationCreated, false);
    assert.equal(status.providerReadinessPersistentTenantDataCreated, false);
    await current.pool.assertReady();
    assert.equal(current.pool.status().providerReadinessAttempts, 1);

    const rows = (await readFile(current.observation, "utf8"))
      .trim().split("\n").map((line) => JSON.parse(line));
    assert.equal(rows.length, 1);
    assert.deepEqual(rows[0].argv, [
      "-m",
      "teaching_skill_miner.teacher_agent_gateway_worker",
      "--provider-readiness"
    ]);
    assert.equal(rows[0].bootstrap.agent_backend, "deepseek");
    assert.equal(rows[0].bootstrap.remote_subject_policy.remote_processing_eligible, false);
    assert.deepEqual(
      rows[0].environment_keys.filter(
        (key: string) => key !== "__CF_USER_TEXT_ENCODING"
      ),
      ["LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"]
    );
    assert.deepEqual(await readdir(current.workerRoot), []);
  } finally {
    await current.pool.onApplicationShutdown();
    await rm(current.temporary, {recursive: true, force: true});
  }
});

test("provider readiness rejects extra fields or any learner-content flag and caches failure briefly", async (context) => {
  for (const mode of ["extra_field", "learner_content_true"] as const) {
    await context.test(mode, async () => {
      const current = await fixture(mode);
      try {
        await assert.rejects(current.pool.assertReady(), HarnessGatewayError);
        const status = current.pool.status();
        assert.equal(status.providerReadinessLastResult, "failed");
        assert.equal(status.providerReadinessAttempts, 1);
        assert.equal(status.providerReadinessSuccesses, 0);
        assert.equal(status.providerReadinessFailures, 1);
        await assert.rejects(current.pool.assertReady(), HarnessGatewayError);
        assert.equal(current.pool.status().providerReadinessAttempts, 1);
        assert.deepEqual(await readdir(current.workerRoot), []);
      } finally {
        await current.pool.onApplicationShutdown();
        await rm(current.temporary, {recursive: true, force: true});
      }
    });
  }
});
