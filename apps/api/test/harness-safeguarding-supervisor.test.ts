import "reflect-metadata";

import assert from "node:assert/strict";
import {chmod, mkdtemp, readFile, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";
import test from "node:test";

import type {AppConfigService} from "../src/config/app-config.service";
import {HealthController} from "../src/health/health.controller";
import {
  HarnessGatewayError,
  HarnessWorkerPoolService
} from "../src/harness/harness-worker-pool.service";
import type {ModelProviderPort} from "../src/providers/model-provider.port";
import type {ContinuousReadinessService} from "../src/health/continuous-readiness.service";


const DISPATCH_SECRET =
  "supervisor-dispatch-secret-present-only-on-stdin-0001";
const RETENTION_SECRET =
  "supervisor-retention-secret-present-only-on-stdin-0002";
const RETENTION_POLICY_VERSION = "closed-case-retention-v1";
const RETENTION_MINIMUM_CLOSED_AGE_SECONDS = 7_776_000;
const RETENTION_MAXIMUM_CASES_PER_RUN = 128;
const RETENTION_DEPLOYMENT_CONTEXT_SHA256 = "b".repeat(64);

type SupervisorMode =
  | "ready"
  | "overdue"
  | "retention_disabled"
  | "retention_policy_mismatch"
  | "retention_blocked"
  | "capacity_near"
  | "false_positive"
  | "extra_field"
  | "oversized"
  | "hang";

function config(root: string, python: string): AppConfigService {
  return {
    nodeEnv: "test",
    dataBackend: "memory",
    authMode: "development",
    harnessGatewayEnabled: true,
    harnessWorkerRoot: root,
    harnessWorkerCwd: resolve(process.cwd(), "../.."),
    harnessWorkerPython: python,
    harnessWorkerBackend: "deterministic",
    harnessProviderApiKeyFile: undefined,
    harnessRemoteProviderPolicy: undefined,
    harnessSafeguardingLocale: "zh-CN",
    harnessSafeguardingDispatchUrl:
      "https://safeguarding.organization.example/v1/cases",
    harnessSafeguardingDispatchBearerSecret: DISPATCH_SECRET,
    harnessSafeguardingDispatchPolicyVersion: "institution-safeguarding-v1",
    harnessSafeguardingDispatchTimeoutMs: 5_000,
    harnessSafeguardingDispatchMaxResponseBytes: 32 * 1024,
    harnessSafeguardingDispatchConfigured: true,
    harnessSafeguardingRetentionPolicyVersion: RETENTION_POLICY_VERSION,
    harnessSafeguardingRetentionMinimumClosedAgeSeconds:
      RETENTION_MINIMUM_CLOSED_AGE_SECONDS,
    harnessSafeguardingRetentionMaximumCasesPerRun:
      RETENTION_MAXIMUM_CASES_PER_RUN,
    harnessSafeguardingRetentionAuthoritySecret: RETENTION_SECRET,
    harnessSafeguardingRetentionDeploymentContextSha256:
      RETENTION_DEPLOYMENT_CONTEXT_SHA256,
    harnessSafeguardingRetentionConfigured: true,
    harnessWorkerFilesystemIsolationRequired: false,
    harnessScopeKeyVersion: "k1",
    harnessScopeKeys: [{
      version: "k1",
      secret: "supervisor-test-scope-secret-at-least-32-characters"
    }],
    harnessMaxWorkers: 2,
    harnessWorkerIdleTimeoutMs: 60_000,
    harnessWorkerStartupMs: 2_000,
    harnessWorkerShutdownMs: 500
  } as unknown as AppConfigService;
}

function fakePythonSource(
  observationPath: string,
  mode: SupervisorMode
): string {
  return `#!${process.execPath}
const fs = require("node:fs");
let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { raw += chunk; });
process.stdin.on("end", () => {
  const bootstrap = JSON.parse(raw);
  const secrets = [
    ${JSON.stringify(DISPATCH_SECRET)},
    ${JSON.stringify(RETENTION_SECRET)}
  ];
  const exposedOutsideStdin = secrets.some((secret) =>
    process.argv.join("\\0").includes(secret)
      || Object.values(process.env).some((value) => String(value).includes(secret))
  );
  fs.appendFileSync(${JSON.stringify(observationPath)}, JSON.stringify({
    kind: "start",
    argv: process.argv.slice(2),
    environment_keys: Object.keys(process.env).sort(),
    exposed_outside_stdin: exposedOutsideStdin,
    bootstrap
  }) + "\\n");
  if (${JSON.stringify(mode)} === "oversized") {
    process.stdout.write("x".repeat(9000));
  } else if (${JSON.stringify(mode)} === "hang") {
    // Exercise bounded shutdown while the first status is still pending.
  } else {
    const now = new Date(Math.floor(Date.now() / 1000) * 1000)
      .toISOString().replace(".000Z", "Z");
    const overdue = ${JSON.stringify(mode)} === "overdue";
    const retentionDisabled = ${JSON.stringify(mode)} === "retention_disabled";
    const retentionPolicyMismatch = ${JSON.stringify(mode)} === "retention_policy_mismatch";
    const retentionBlocked = ${JSON.stringify(mode)} === "retention_blocked";
    const capacityNear = ${JSON.stringify(mode)} === "capacity_near";
    const falsePositive = ${JSON.stringify(mode)} === "false_positive";
    const scoped = overdue || retentionBlocked || capacityNear;
    const status = {
      schema: "teaching_skill_miner.safeguarding_supervisor_status.v2",
      status: retentionBlocked || capacityNear || falsePositive ? "degraded" : "ready",
      scopes_scanned: scoped ? 1 : 0,
      stores_unavailable: 0,
      pending: overdue ? 1 : 0,
      overdue: overdue ? 1 : 0,
      oldest_pending_age_seconds: overdue ? 901 : 0,
      attempted: 0,
      accepted: 0,
      failed: 0,
      attempted_total: 0,
      accepted_total: 0,
      failed_total: 0,
      last_success_at_utc: null,
      receiver_readiness_required: true,
      receiver_readiness_status: "ready",
      receiver_network_validated: true,
      receiver_credential_validated: true,
      receiver_readiness_attempts_total: 1,
      receiver_readiness_successes_total: 1,
      receiver_readiness_failures_total: 0,
      last_receiver_success_at_utc: now,
      retention_enabled: !retentionDisabled,
      retention_status: retentionDisabled
        ? "disabled"
        : retentionBlocked
          ? "blocked"
          : "idle",
      retention_policy_version_sha256: retentionDisabled
        ? null
        : require("node:crypto").createHash("sha256")
            .update(
              retentionPolicyMismatch
                ? "unexpected-retention-policy-v1"
                : ${JSON.stringify(RETENTION_POLICY_VERSION)},
              "utf8"
            )
            .digest("hex"),
      retention_minimum_closed_age_seconds: retentionDisabled
        ? null
        : ${RETENTION_MINIMUM_CLOSED_AGE_SECONDS},
      retention_maximum_cases_per_run: retentionDisabled
        ? null
        : ${RETENTION_MAXIMUM_CASES_PER_RUN},
      retention_eligible_cases: 0,
      retention_cases_compacted: 0,
      retention_events_compacted: 0,
      retention_failures: 0,
      retention_blocked_stores: retentionBlocked ? 1 : 0,
      capacity_near_limit_stores: capacityNear ? 1 : 0,
      capacity_events: scoped ? 99 : 0,
      capacity_event_limit: scoped ? 100 : 0,
      capacity_event_headroom_min: scoped ? 1 : null,
      capacity_store_bytes: scoped ? 999 : 0,
      capacity_store_byte_limit: scoped ? 1000 : 0,
      capacity_store_byte_headroom_min: scoped ? 1 : null,
      capacity_recent_erasure_tombstones: scoped ? 9 : 0,
      capacity_recent_erasure_tombstone_limit: scoped ? 10 : 0,
      capacity_recent_erasure_tombstone_headroom_min: scoped ? 1 : null,
      retention_cases_compacted_total: 0,
      retention_events_compacted_total: 0,
      erasure_fence_inserted_count: 0,
      erasure_fence_estimated_false_positive_upper_bound:
        falsePositive ? 0.000002 : 0,
      erasure_fence_false_positive_target_upper_bound: 0.000001,
      erasure_fence_false_positive_within_target: !falsePositive,
      erasure_fence_false_negative_possible: false,
      erasure_fence_false_positive_policy: "fail_closed_as_erased",
      raw_learner_text_read_or_sent: false,
      scope_identity_labels_exposed: false,
      updated_at_utc: now
    };
    if (${JSON.stringify(mode)} === "extra_field") status.private_path = "/secret";
    process.stdout.write(JSON.stringify(status) + "\\n");
  }
});
process.on("SIGTERM", () => {
  fs.appendFileSync(${JSON.stringify(observationPath)}, JSON.stringify({kind: "term"}) + "\\n");
  process.exit(0);
});
setInterval(() => {}, 1000);
`;
}

async function fixture(
  mode: SupervisorMode
): Promise<{
  temporary: string;
  observation: string;
  pool: HarnessWorkerPoolService;
  configuration: AppConfigService;
}> {
  const temporary = await mkdtemp(join(tmpdir(), "teachlab-supervisor-ts-"));
  const root = join(temporary, "workers");
  const observation = join(temporary, "observation.ndjson");
  const python = join(temporary, "fake-python");
  await writeFile(python, fakePythonSource(observation, mode), {mode: 0o700});
  await chmod(python, 0o700);
  const configuration = config(root, python);
  return {
    temporary,
    observation,
    configuration,
    pool: new HarnessWorkerPoolService(configuration)
  };
}

async function observations(path: string): Promise<Record<string, unknown>[]> {
  return (await readFile(path, "utf8"))
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

test("one global supervisor receives its exact secret only over stdin and stops boundedly", async () => {
  const current = await fixture("ready");
  try {
    await Promise.all([
      current.pool.assertReady(),
      current.pool.assertReady(),
      current.pool.assertReady()
    ]);
    const status = current.pool.status();
    assert.equal(status.safeguardingSupervisorRequired, true);
    assert.equal(status.safeguardingSupervisorRunning, true);
    assert.equal(status.safeguardingSupervisorLastResult, "ready");
    assert.equal(status.safeguardingSupervisorPending, 0);
    assert.equal(status.safeguardingSupervisorRawLearnerTextReadOrSent, false);
    assert.equal(status.safeguardingSupervisorScopeIdentityLabelsExposed, false);

    const rows = await observations(current.observation);
    const starts = rows.filter((row) => row.kind === "start");
    assert.equal(starts.length, 1);
    assert.equal(starts[0]!.exposed_outside_stdin, false);
    assert.deepEqual(starts[0]!.argv, [
      "-m",
      "teaching_skill_miner.teacher_agent_safeguarding_supervisor"
    ]);
    assert.deepEqual(
      (starts[0]!.environment_keys as string[]).filter(
        (key) => key !== "__CF_USER_TEXT_ENCODING"
      ),
      ["LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"]
    );
    const bootstrap = starts[0]!.bootstrap as Record<string, unknown>;
    assert.deepEqual(Object.keys(bootstrap).sort(), [
      "bearer_secret",
      "endpoint",
      "maximum_response_bytes",
      "policy_version",
      "poll_seconds",
      "retention_authority_secret",
      "retention_deployment_context_sha256",
      "retention_maximum_cases_per_run",
      "retention_minimum_closed_age_seconds",
      "retention_policy_version",
      "root",
      "schema",
      "timeout_ms"
    ]);
    assert.equal(
      bootstrap.schema,
      "teaching_skill_miner.safeguarding_supervisor_bootstrap.v2"
    );
    assert.equal(bootstrap.bearer_secret, DISPATCH_SECRET);
    assert.equal(bootstrap.retention_authority_secret, RETENTION_SECRET);
    assert.equal(bootstrap.retention_policy_version, RETENTION_POLICY_VERSION);
    assert.equal(
      bootstrap.retention_minimum_closed_age_seconds,
      RETENTION_MINIMUM_CLOSED_AGE_SECONDS
    );
    assert.equal(
      bootstrap.retention_maximum_cases_per_run,
      RETENTION_MAXIMUM_CASES_PER_RUN
    );
    assert.equal(
      bootstrap.retention_deployment_context_sha256,
      RETENTION_DEPLOYMENT_CONTEXT_SHA256
    );

    await current.pool.onApplicationShutdown();
    assert.equal(
      (await observations(current.observation)).filter((row) => row.kind === "term").length,
      1
    );
  } finally {
    await current.pool.onApplicationShutdown();
    await rm(current.temporary, {recursive: true, force: true});
  }
});

test("retention and aggregate-capacity degradation fail readiness but preserve health truth", async (context) => {
  for (const mode of [
    "retention_disabled",
    "retention_policy_mismatch",
    "retention_blocked",
    "capacity_near",
    "false_positive"
  ] as const) {
    await context.test(mode, async () => {
      const current = await fixture(mode);
      try {
        await assert.rejects(current.pool.assertReady(), HarnessGatewayError);
        const status = current.pool.status();
        assert.equal(status.safeguardingSupervisorRunning, true);
        assert.equal(status.safeguardingSupervisorLastResult, "degraded");
        if (mode === "retention_disabled") {
          assert.equal(status.safeguardingSupervisorRetentionEnabled, false);
          assert.equal(
            status.safeguardingSupervisorRetentionStatus,
            "disabled"
          );
        }
        if (mode === "retention_policy_mismatch") {
          assert.notEqual(
            status.safeguardingSupervisorRetentionPolicyVersionSha256,
            null
          );
        }
        if (mode === "retention_blocked") {
          assert.equal(status.safeguardingSupervisorRetentionBlockedStores, 1);
        }
        if (mode === "capacity_near") {
          assert.equal(status.safeguardingSupervisorCapacityNearLimitStores, 1);
          assert.equal(status.safeguardingSupervisorCapacityEventHeadroomMin, 1);
        }
        if (mode === "false_positive") {
          assert.equal(
            status.safeguardingSupervisorErasureFenceFalsePositiveWithinTarget,
            false
          );
        }
        const health = new HealthController(
          current.configuration,
          {status: async () => ({configured: false})} as ModelProviderPort,
          current.pool,
          {} as ContinuousReadinessService
        );
        assert.equal((await health.health()).status, "ok");
      } finally {
        await current.pool.onApplicationShutdown();
        await rm(current.temporary, {recursive: true, force: true});
      }
    });
  }
});

test("overdue aggregate fails readiness while liveness remains honestly projected", async () => {
  const current = await fixture("overdue");
  try {
    await assert.rejects(
      current.pool.assertReady(),
      (error: unknown) =>
        error instanceof HarnessGatewayError && error.statusCode === 503
    );
    const status = current.pool.status();
    assert.equal(status.safeguardingSupervisorRunning, true);
    assert.equal(status.safeguardingSupervisorLastResult, "overdue");
    assert.equal(status.safeguardingSupervisorPending, 1);
    assert.equal(status.safeguardingSupervisorOverdue, 1);

    const health = new HealthController(
      current.configuration,
      {status: async () => ({configured: false})} as ModelProviderPort,
      current.pool,
      {} as ContinuousReadinessService
    );
    const projection = await health.health();
    assert.equal(projection.status, "ok");
    assert.equal(
      projection.teachingHarness.safeguardingSupervisorLastResult,
      "overdue"
    );
  } finally {
    await current.pool.onApplicationShutdown();
    await rm(current.temporary, {recursive: true, force: true});
  }
});

test("strict bounded NDJSON and supervisor process loss fail closed", async (context) => {
  for (const mode of ["extra_field", "oversized"] as const) {
    await context.test(mode, async () => {
      const current = await fixture(mode);
      try {
        await assert.rejects(current.pool.assertReady(), HarnessGatewayError);
        assert.equal(current.pool.status().safeguardingSupervisorLastResult, "failed");
        assert.equal(current.pool.status().safeguardingSupervisorRunning, false);
      } finally {
        await current.pool.onApplicationShutdown();
        await rm(current.temporary, {recursive: true, force: true});
      }
    });
  }

  await context.test("process exit", async () => {
    const current = await fixture("ready");
    try {
      await current.pool.assertReady();
      assert.equal(
        await current.pool.terminateSafeguardingSupervisorForTesting(),
        true
      );
      await assert.rejects(current.pool.assertReady(), HarnessGatewayError);
      assert.equal(
        current.pool.status().safeguardingSupervisorLastResult,
        "unavailable"
      );
    } finally {
      await current.pool.onApplicationShutdown();
      await rm(current.temporary, {recursive: true, force: true});
    }
  });
});

test("stale status and shutdown during startup are both bounded and fail closed", async () => {
  const ready = await fixture("ready");
  try {
    let now = Date.now();
    ready.pool.setClockForTesting(() => now);
    await ready.pool.assertReady();
    now += 20_000;
    await assert.rejects(ready.pool.assertReady(), HarnessGatewayError);
    assert.equal(ready.pool.status().safeguardingSupervisorRunning, true);
    assert.equal(ready.pool.status().safeguardingSupervisorLastResult, "stale");
  } finally {
    await ready.pool.onApplicationShutdown();
    await rm(ready.temporary, {recursive: true, force: true});
  }

  const hanging = await fixture("hang");
  const startup = hanging.pool.assertReady().then(
    () => undefined,
    (error: unknown) => error
  );
  try {
    const deadline = Date.now() + 1_000;
    while (Date.now() < deadline) {
      try {
        if ((await observations(hanging.observation)).length) break;
      } catch {
        await new Promise<void>((resolve) => setTimeout(resolve, 10));
      }
    }
    const began = Date.now();
    await hanging.pool.onApplicationShutdown();
    assert.ok(Date.now() - began < 1_000);
    assert.ok((await startup) instanceof HarnessGatewayError);
  } finally {
    await hanging.pool.onApplicationShutdown();
    await rm(hanging.temporary, {recursive: true, force: true});
  }
});
