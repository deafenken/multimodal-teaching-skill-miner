import "reflect-metadata";

import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {realpathSync} from "node:fs";
import {mkdtemp, readFile, readdir, rm} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";
import test from "node:test";

import type {AppConfigService} from "../src/config/app-config.service";
import {
  HarnessWorkerPoolService,
  type HarnessUpstreamRequest,
} from "../src/harness/harness-worker-pool.service";
import type {RemoteSubjectPolicy} from "../src/auth/remote-processing-policy";
import type {AccessScope} from "../src/tenancy/access-scope";

const SECRET = "real-safeguarding-worker-secret-with-at-least-32-characters";
const SCOPE: AccessScope = {
  tenantId: "opaque-safeguarding-tenant",
  ownerId: "opaque-safeguarding-learner",
};
const POLICY: RemoteSubjectPolicy = {
  policy_id: "school-policy",
  policy_version: "v1",
  policy_source: "organization_oidc_or_roster_policy",
  likely_minor: true,
  guardian_or_school_policy: "verified_school_policy",
  remote_processing_eligible: false,
};

function modernPython(): string {
  for (const command of ["python3.13", "python3.12", "python3.11", "python3.10", "python"]) {
    try {
      const located = execFileSync("/usr/bin/which", [command], {encoding: "utf8"}).trim();
      execFileSync(located, ["-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"], {stdio: "ignore"});
      return realpathSync(located);
    } catch {
      // Try the next explicit executable.
    }
  }
  throw new Error("Python >=3.10 is required for safeguarding worker integration");
}

function config(root: string): AppConfigService {
  return {
    nodeEnv: "test",
    harnessGatewayEnabled: true,
    harnessWorkerRoot: root,
    harnessWorkerCwd: resolve(process.cwd(), "../.."),
    harnessWorkerPython: modernPython(),
    harnessWorkerBackend: "deterministic",
    harnessWorkerFilesystemIsolationRequired: false,
    harnessScopeKeyVersion: "k1",
    harnessScopeKeys: [{version: "k1", secret: SECRET}],
    harnessMaxWorkers: 1,
    harnessWorkerIdleTimeoutMs: 60_000,
    harnessWorkerStartupMs: 30_000,
    harnessWorkerShutdownMs: 2_000,
    teacherAuthorityRoles: ["teacher"],
    safeguardingAuthorityRoles: ["safeguarding"],
    teacherAuthorityTtlSeconds: 120,
    harnessSafeguardingLocale: "zh-CN",
    harnessSafeguardingDispatchConfigured: false,
  } as unknown as AppConfigService;
}

async function request(
  pool: HarnessWorkerPoolService,
  input: HarnessUpstreamRequest,
): Promise<{status: number; value: Record<string, any>; raw: string}> {
  const response = await pool.request(SCOPE, input);
  const chunks: Buffer[] = [];
  for await (const chunk of response) chunks.push(Buffer.from(chunk as Buffer));
  const raw = Buffer.concat(chunks).toString("utf8");
  return {
    status: response.statusCode ?? 0,
    value: JSON.parse(raw) as Record<string, any>,
    raw,
  };
}

function staffRequest(
  pool: HarnessWorkerPoolService,
  path: string,
  body: Record<string, unknown>,
) {
  return request(pool, {
    method: "POST",
    path,
    body: Buffer.from(JSON.stringify(body), "utf8"),
    teacherRoles: ["safeguarding"],
    teacherAllowedRoles: ["safeguarding"],
    teacherPrincipal: {issuer: "https://issuer.example", subject: "staff-7"},
    preserveRemoteSubjectPolicy: true,
  });
}

test("real TS authority and Python worker close a content-free case across restart", async () => {
  const root = await mkdtemp(join(tmpdir(), "teachlab-real-safeguarding-"));
  const pool = new HarnessWorkerPoolService(config(root));
  const unsafeText = "我现在就要自杀";
  try {
    const bootstrap = await request(pool, {
      method: "GET",
      path: "api/bootstrap",
      remoteSubjectPolicy: POLICY,
    });
    assert.equal(bootstrap.status, 200);
    assert.equal(bootstrap.value.learner_safety.safeguarding.configured, true);
    assert.equal(bootstrap.value.learner_safety.safeguarding.dispatcher_configured, false);
    assert.equal(bootstrap.value.learner_safety.safeguarding.queue_configured, false);
    assert.equal(
      bootstrap.value.learner_safety.safeguarding.staff_workflow,
      "fresh_authoritative_safeguarding_role",
    );

    const opened = await request(pool, {
      method: "POST",
      path: "api/chat",
      body: Buffer.from(JSON.stringify({
        request_id: "safeguarding-direct-chat-0001",
        messages: [{role: "user", content: unsafeText}],
        web_search: false,
      }), "utf8"),
      remoteSubjectPolicy: POLICY,
    });
    assert.equal(opened.status, 200);
    const caseId = opened.value.safety_obligation.safeguarding.case_id as string;
    assert.match(caseId, /^sgc_[0-9a-f]{24}$/);
    assert.equal(opened.value.safety_obligation.safeguarding.delivery_status, "escalation_unavailable");

    const listed = await staffRequest(pool, "api/safeguarding/list", {
      safeguarding_idempotency_key: "real-staff-list-0001",
    });
    assert.equal(listed.status, 200);
    assert.equal(listed.value.cases.length, 1);
    assert.equal(listed.value.cases[0].case_id, caseId);
    assert.equal(listed.value.cases[0].version, 1);
    assert.equal(listed.value.raw_learner_text_exposed, false);
    assert.doesNotMatch(listed.raw, new RegExp(unsafeText));

    const acknowledgeBody = {
      case_id: caseId,
      expected_version: 1,
      safeguarding_idempotency_key: "real-staff-case-ack-0001",
    };
    const acknowledged = await staffRequest(
      pool,
      "api/safeguarding/case/acknowledge",
      acknowledgeBody,
    );
    const replay = await staffRequest(
      pool,
      "api/safeguarding/case/acknowledge",
      acknowledgeBody,
    );
    assert.equal(acknowledged.status, 200);
    assert.deepEqual(replay.value, acknowledged.value);
    assert.equal(acknowledged.value.case.version, 2);

    assert.equal(await pool.terminateForTesting(SCOPE), true);
    const afterRestart = await staffRequest(pool, "api/safeguarding/list", {
      safeguarding_idempotency_key: "real-staff-list-after-restart-0001",
    });
    assert.equal(afterRestart.value.cases[0].version, 2);
    assert.equal(afterRestart.value.cases[0].status, "acknowledged");

    const closed = await staffRequest(pool, "api/safeguarding/case/close", {
      case_id: caseId,
      expected_version: 2,
      safeguarding_idempotency_key: "real-staff-case-close-0001",
    });
    assert.equal(closed.status, 200);
    assert.equal(closed.value.case.version, 3);
    assert.equal(closed.value.case.status, "closed");

    await pool.onApplicationShutdown();
    const paths = await readdir(root, {recursive: true});
    for (const path of paths) {
      const content = await readFile(join(root, path), "utf8").catch(() => "");
      assert.doesNotMatch(content, new RegExp(unsafeText));
    }
  } finally {
    await pool.onApplicationShutdown();
    await rm(root, {recursive: true, force: true});
  }
});
