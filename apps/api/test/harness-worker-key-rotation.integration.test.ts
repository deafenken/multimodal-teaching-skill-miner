import "reflect-metadata";

import assert from "node:assert/strict";
import {createHmac} from "node:crypto";
import {execFileSync} from "node:child_process";
import {realpathSync} from "node:fs";
import {mkdtemp, readFile, readdir, rm} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";
import test from "node:test";

import type {AppConfigService, HarnessScopeKey} from "../src/config/app-config.service";
import {
  HarnessWorkerPoolService,
  type HarnessUpstreamRequest
} from "../src/harness/harness-worker-pool.service";
import {scopeMigrationMarkerNames} from "../src/harness/harness-scope-key-migration";
import type {AccessScope} from "../src/tenancy/access-scope";
import type {RemoteSubjectPolicy} from "../src/auth/remote-processing-policy";

const K1_SECRET = "real-worker-rotation-k1-secret-with-at-least-32-characters";
const K2_SECRET = "real-worker-rotation-k2-secret-with-at-least-32-characters";
const SCOPE: AccessScope = {tenantId: "opaque-db-tenant-a", ownerId: "opaque-db-owner-a"};

function findModernPython(): string {
  for (const command of ["python3.13", "python3.12", "python3.11", "python3.10", "python"]) {
    try {
      const located = execFileSync("/usr/bin/which", [command], {encoding: "utf8"}).trim();
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
  throw new Error("Python >=3.10 is required for worker key-rotation integration");
}

function config(root: string, keys: readonly HarnessScopeKey[]): AppConfigService {
  return {
    nodeEnv: "test",
    harnessGatewayEnabled: true,
    harnessWorkerRoot: root,
    harnessWorkerCwd: resolve(process.cwd(), "../.."),
    harnessWorkerPython: findModernPython(),
    harnessWorkerBackend: "deterministic",
    harnessProviderApiKeyFile: undefined,
    harnessWorkerFilesystemIsolationRequired: false,
    harnessScopeKeyVersion: keys[0]!.version,
    harnessScopeKeys: keys,
    harnessMaxWorkers: 1,
    harnessWorkerIdleTimeoutMs: 60_000,
    harnessWorkerStartupMs: 30_000,
    harnessWorkerShutdownMs: 2_000,
    teacherAuthorityRoles: ["teacher"],
    teacherAuthorityTtlSeconds: 120
  } as unknown as AppConfigService;
}

async function request(
  pool: HarnessWorkerPoolService,
  input: HarnessUpstreamRequest
): Promise<{status: number; body: Buffer}> {
  const response = await pool.request(SCOPE, input);
  const chunks: Buffer[] = [];
  for await (const chunk of response) chunks.push(Buffer.from(chunk as Buffer));
  return {status: response.statusCode ?? 0, body: Buffer.concat(chunks)};
}

function parseSse(body: Buffer): Array<Record<string, unknown>> {
  return body.toString("utf8")
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data: "))
    .map((line) => JSON.parse(line.slice(6)) as Record<string, unknown>);
}

async function bootstrap(pool: HarnessWorkerPoolService): Promise<Record<string, any>> {
  const response = await request(pool, {method: "GET", path: "api/bootstrap"});
  assert.equal(response.status, 200);
  return JSON.parse(response.body.toString("utf8")) as Record<string, any>;
}

const ADULT_POLICY: RemoteSubjectPolicy = {
  policy_id: "school-policy",
  policy_version: "v1",
  policy_source: "organization_oidc_or_roster_policy",
  likely_minor: false,
  guardian_or_school_policy: "not_required",
  remote_processing_eligible: true
};

test("a signed subject policy change stops and rebuilds the exact scope worker", async () => {
  const root = await mkdtemp(join(tmpdir(), "teachlab-policy-rebind-"));
  const service = new HarnessWorkerPoolService(config(root, [{
    version: "k1",
    secret: K1_SECRET
  }]));
  try {
    const first = await request(service, {
      method: "GET",
      path: "api/bootstrap",
      remoteSubjectPolicy: ADULT_POLICY
    });
    assert.equal(first.status, 200);
    const workers = (service as unknown as {
      workers: Map<string, {child: {pid?: number}; remoteSubjectPolicyHash: string}>;
    }).workers;
    const original = [...workers.values()][0]!;
    const originalPid = original.child.pid;
    assert.equal(typeof originalPid, "number");
    const denied: RemoteSubjectPolicy = {
      ...ADULT_POLICY,
      likely_minor: true,
      guardian_or_school_policy: "not_required",
      remote_processing_eligible: false
    };
    const second = await request(service, {
      method: "GET",
      path: "api/bootstrap",
      remoteSubjectPolicy: denied
    });
    assert.equal(second.status, 200);
    const replacement = [...workers.values()][0]!;
    assert.notEqual(replacement.child.pid, originalPid);
    assert.notEqual(replacement.remoteSubjectPolicyHash, original.remoteSubjectPolicyHash);
  } finally {
    await service.onApplicationShutdown();
    await rm(root, {recursive: true, force: true});
  }
});

async function startSession(
  pool: HarnessWorkerPoolService,
  requestId: string,
  startKey: string
): Promise<string> {
  const base = await bootstrap(pool);
  const response = await request(pool, {
    method: "POST",
    path: "api/stream",
    accept: "text/event-stream",
    body: Buffer.from(JSON.stringify({
      operation: "start",
      request_id: requestId,
      payload: {
        goal: base.default_goal,
        student_profile: base.default_student_profile,
        start_idempotency_key: startKey
      }
    }), "utf8")
  });
  assert.equal(response.status, 200);
  const operation = parseSse(response.body).find((event) => event.type === "operation.result");
  const sessionId = (operation?.payload as Record<string, any> | undefined)
    ?.result?.session_ref?.session_id;
  assert.match(sessionId ?? "", /^teach_[A-Za-z0-9_-]{16,64}$/);
  return sessionId;
}

async function resume(
  pool: HarnessWorkerPoolService,
  sessionId: string
): Promise<Record<string, any>> {
  const response = await request(pool, {
    method: "POST",
    path: "api/session",
    body: Buffer.from(JSON.stringify({session_id: sessionId}), "utf8")
  });
  assert.equal(response.status, 200);
  const session = JSON.parse(response.body.toString("utf8")) as Record<string, any>;
  assert.equal(session.session_id, sessionId);
  return session;
}

async function signedStores(pool: HarnessWorkerPoolService, sessionId: string) {
  const session = await resume(pool, sessionId);
  const consent = await request(pool, {
    method: "POST",
    path: "api/consent/list",
    body: Buffer.from("{}", "utf8")
  });
  assert.equal(consent.status, 200);
  assert.equal(JSON.parse(consent.body.toString("utf8")).receipts.length, 1);
  const due = await request(pool, {
    method: "POST",
    path: "api/learning-reviews/due",
    body: Buffer.from(JSON.stringify({
      session_id: session.session_id,
      expected_round: session.rounds_completed,
      expected_question_id: session.expected_question_id,
      expected_context_version: session.context_version,
      profile_revision: session.profile_summary.profile_revision
    }), "utf8")
  });
  assert.equal(due.status, 200);
  assert.equal(JSON.parse(due.body.toString("utf8")).reviews.length, 1);
}

function seedLegacySignedStores(
  python: string,
  privateRoot: string,
  scopeId: string,
  dataKey: Buffer
): void {
  const script = String.raw`
import base64, hashlib, hmac, json, sys
from pathlib import Path
from teaching_skill_miner.teacher_agent_consent import RemoteConsentStore
from teaching_skill_miner.teacher_agent_learning_records import (
    LearningRecordStore, build_learning_evidence_outbox_event, mint_learner_key
)
value = json.loads(sys.stdin.read())
root = Path(value["root"])
key = base64.urlsafe_b64decode(value["data_key"] + "=")
def derived(label):
    return hmac.new(key, b"teachlab-gateway-worker-v1\x00" + label, hashlib.sha256).digest()
consent_key = derived(b"consent")
subject_id = "subject_" + hmac.new(
    consent_key, b"teachlab-local-consent-subject-v1", hashlib.sha256
).hexdigest()[:32]
RemoteConsentStore(root / "consent", signing_secret=consent_key).grant(
    subject_id=subject_id,
    purpose="remote_teaching",
    provider_id="test-provider",
    processing_region="test-region",
    data_categories=["learner_message"],
    provider_retention_days=0,
    validity_days=30,
)
learner_secret = derived(b"learner-key")
profile_ref = "profile_" + derived(b"learner-profile-ref").hex()
learner_key = mint_learner_key(
    profile_ref, tenant_id=value["scope_id"], secret=learner_secret
)
component = {
    "kc_id": "kc_rotation_continuity",
    "label": "not persisted",
    "source": "syllabus_lesson_component",
    "source_ref": "syl_0123456789abcdef01234567:module_01:lesson_01_01#kc_rotation_continuity",
    "teacher_grading_authority_available": True,
}
evidence = {
    "evidence_id": "evidence.rotation",
    "item_id": "item.rotation",
    "question_id": "question.rotation",
    "rubric_id": "rubric.rotation",
    "knowledge_component_id": "kc_rotation_continuity",
    "signal": "correct",
    "answer_alignment": "aligned",
    "assessment_eligible": True,
    "authoritative": True,
    "observed_at": "2000-01-01T00:00:00Z",
    "time_basis": "session_logical",
    "evidence_fingerprint": hashlib.sha256(b"rotation-authoritative-evidence").hexdigest(),
}
event = build_learning_evidence_outbox_event(
    learner_key=learner_key,
    knowledge_component=component,
    evidence=evidence,
    expected_version=0,
    committed_at_utc="2000-01-01T00:00:01Z",
    commit_receipt_id="turn_committed:rotation",
)
LearningRecordStore(root / "learning_records").apply_committed_evidence_event(event)
`;
  execFileSync(python, ["-c", script], {
    cwd: resolve(process.cwd(), "../.."),
    input: JSON.stringify({
      root: privateRoot,
      scope_id: scopeId,
      data_key: dataKey.toString("base64url")
    }),
    stdio: ["pipe", "pipe", "pipe"]
  });
}

test("real worker session plus signed consent/learning survive rotation and previous-key removal", async () => {
  const root = await mkdtemp(join(tmpdir(), "teachlab-real-worker-rotation-"));
  const k1 = {version: "k1", secret: K1_SECRET};
  const k2 = {version: "k2", secret: K2_SECRET};
  let first: HarnessWorkerPoolService | undefined;
  let rotated: HarnessWorkerPoolService | undefined;
  let activeOnly: HarnessWorkerPoolService | undefined;
  try {
    const python = findModernPython();
    first = new HarnessWorkerPoolService(config(root, [k1]));
    const beforeRotation = await startSession(
      first,
      "real-rotation-before",
      "real-rotation-before-start"
    );
    await first.onApplicationShutdown();
    first = undefined;
    const previousScopes = await readdir(join(root, "k1"));
    assert.equal(previousScopes.length, 1);
    const previousScopeId = previousScopes[0]!;
    const previousDataKey = createHmac("sha256", k1.secret)
      .update(`worker-data-v1\0k1\0${previousScopeId}`, "utf8")
      .digest();
    seedLegacySignedStores(
      python,
      join(root, "k1", previousScopeId),
      previousScopeId,
      previousDataKey
    );

    rotated = new HarnessWorkerPoolService(config(root, [k2, k1]));
    await signedStores(rotated, beforeRotation);
    const afterRotation = await startSession(
      rotated,
      "real-rotation-after",
      "real-rotation-after-start"
    );
    await rotated.onApplicationShutdown();
    rotated = undefined;

    const oldScopes = await readdir(join(root, "k1"));
    assert.equal(oldScopes.length, 1);
    assert.deepEqual(await readdir(join(root, "k1", oldScopes[0]!)), [
      scopeMigrationMarkerNames.tombstone
    ]);
    const activeScopes = await readdir(join(root, "k2"));
    assert.equal(activeScopes.length, 1);
    const envelope = await readFile(
      join(root, "k2", activeScopes[0]!, scopeMigrationMarkerNames.dataKeyEnvelope),
      "utf8"
    );
    assert.doesNotMatch(envelope, /opaque-db-tenant-a|opaque-db-owner-a|real-worker-rotation-k1-secret/);

    // The previous key is deliberately absent from this process. Both the old
    // session and a session written after migration remain readable.
    activeOnly = new HarnessWorkerPoolService(config(root, [k2]));
    await signedStores(activeOnly, beforeRotation);
    await resume(activeOnly, afterRotation);
  } finally {
    await first?.onApplicationShutdown();
    await rotated?.onApplicationShutdown();
    await activeOnly?.onApplicationShutdown();
    await rm(root, {recursive: true, force: true});
  }
});
