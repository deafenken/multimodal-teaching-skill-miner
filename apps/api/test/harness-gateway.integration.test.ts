import "reflect-metadata";

import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {mkdtemp, readFile, readdir, rm, stat} from "node:fs/promises";
import {realpathSync} from "node:fs";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";
import {after, before, test} from "node:test";

import type {NestFastifyApplication} from "@nestjs/platform-fastify";

import {createApplication} from "../src/create-application";
import {HarnessWorkerPoolService} from "../src/harness/harness-worker-pool.service";

interface BrowserSession {
  cookie: string;
  csrfToken: string;
  principal: {subject: string; tenantId: string};
}

interface StreamResult {
  response: Response;
  text: string;
  events: Array<Record<string, unknown>>;
}

let app: NestFastifyApplication;
let origin: string;
let workerRoot: string;
let aliceA: BrowserSession;
let aliceB: BrowserSession;
let thirdTenant: BrowserSession;

function findModernPython(): string {
  for (const command of ["python3.13", "python3.12", "python3.11", "python3.10", "python"]) {
    try {
      const located = execFileSync("/usr/bin/which", [command], {
        encoding: "utf8"
      }).trim();
      const supported = execFileSync(
        located,
        ["-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"],
        {stdio: "ignore"}
      );
      void supported;
      return realpathSync(located);
    } catch {
      // Try the next explicit executable. The test fails with a safe message
      // below when no supported runtime is installed.
    }
  }
  throw new Error("A Python >=3.10 runtime is required for the Harness gateway test");
}

function cookieValues(value: string | string[] | undefined): string[] {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

async function signIn(userId: string, tenantId: string): Promise<BrowserSession> {
  const response = await app.inject({
    method: "POST",
    url: "/api/v1/auth/session",
    headers: {
      origin: "http://localhost:3000",
      "sec-fetch-site": "same-site",
      "x-dev-user-id": userId,
      "x-dev-tenant-id": tenantId
    }
  });
  assert.equal(response.statusCode, 201);
  const payload = response.json();
  assert.equal(payload.authenticated, true);
  assert.equal("principal" in payload, false);
  return {
    cookie: cookieValues(response.headers["set-cookie"])
      .map((entry) => entry.split(";", 1)[0])
      .join("; "),
    csrfToken: payload.csrfToken,
    principal: {subject: userId, tenantId}
  };
}

function headers(session: BrowserSession, write = false): Record<string, string> {
  return {
    cookie: session.cookie,
    origin: "http://localhost:3000",
    "sec-fetch-site": "same-site",
    ...(write ? {"x-csrf-token": session.csrfToken} : {})
  };
}

async function gatewayJson(
  session: BrowserSession,
  path: string,
  method: "GET" | "POST" = "GET",
  body?: unknown
): Promise<{status: number; body: Record<string, any>; headers: Headers}> {
  const response = await fetch(`${origin}/api/v1/harness/${path}`, {
    method,
    headers: {
      ...headers(session, method === "POST"),
      ...(method === "POST" ? {"content-type": "application/json"} : {})
    },
    body: method === "POST" ? JSON.stringify(body ?? {}) : undefined
  });
  return {
    status: response.status,
    body: (await response.json()) as Record<string, any>,
    headers: response.headers
  };
}

function parseSse(text: string): Array<Record<string, unknown>> {
  return text
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data: "))
    .map((line) => JSON.parse(line.slice(6)) as Record<string, unknown>);
}

async function startStream(
  session: BrowserSession,
  goal: Record<string, unknown>,
  profile: Record<string, unknown>,
  requestId: string,
  startKey: string
): Promise<StreamResult> {
  const response = await fetch(`${origin}/api/v1/harness/api/stream`, {
    method: "POST",
    headers: {
      ...headers(session, true),
      accept: "text/event-stream",
      "content-type": "application/json"
    },
    body: JSON.stringify({
      operation: "start",
      request_id: requestId,
      payload: {
        goal,
        student_profile: profile,
        start_idempotency_key: startKey
      }
    })
  });
  const text = await response.text();
  return {response, text, events: parseSse(text)};
}

function operationSessionId(result: StreamResult): string {
  const operation = result.events.find((event) => event.type === "operation.result");
  const payload = operation?.payload as Record<string, any> | undefined;
  const sessionId = payload?.result?.session_ref?.session_id;
  assert.equal(typeof sessionId, "string");
  return sessionId;
}

before(async () => {
  workerRoot = await mkdtemp(join(tmpdir(), "teachlab-gateway-test-"));
  await stat(workerRoot).then((metadata) => assert.equal(metadata.isDirectory(), true));
  process.env.NODE_ENV = "test";
  process.env.AUTH_MODE = "development";
  process.env.DATA_BACKEND = "memory";
  process.env.DEV_AUTH_ALLOW_HEADERS = "true";
  process.env.DEV_AUTH_ROLES = "teacher";
  process.env.SEED_DEMO_SESSIONS = "false";
  process.env.SESSION_SECRET = "gateway-integration-session-secret-at-least-32-characters";
  process.env.SESSION_COOKIE_SECURE = "false";
  process.env.CORS_ORIGINS = "http://localhost:3000";
  process.env.HARNESS_GATEWAY_ENABLED = "true";
  process.env.HARNESS_WORKER_ROOT = workerRoot;
  process.env.HARNESS_WORKER_CWD = resolve(process.cwd(), "../..");
  process.env.HARNESS_WORKER_PYTHON = findModernPython();
  process.env.HARNESS_WORKER_BACKEND = "deterministic";
  process.env.HARNESS_SCOPE_KEY_VERSION = "k2";
  process.env.HARNESS_SCOPE_SECRET =
    "gateway-active-scope-secret-with-at-least-32-characters";
  process.env.HARNESS_PREVIOUS_SCOPE_KEYS = JSON.stringify({
    k1: "gateway-previous-scope-secret-with-at-least-32-characters"
  });
  process.env.HARNESS_MAX_WORKERS = "2";
  process.env.HARNESS_WORKER_STARTUP_MS = "30000";

  app = await createApplication({logger: false});
  await app.listen(0, "127.0.0.1");
  const address = app.getHttpServer().address();
  assert.ok(address && typeof address === "object");
  origin = `http://127.0.0.1:${address.port}`;
  aliceA = await signIn("alice", "tenant-a");
  aliceB = await signIn("alice", "tenant-b");
  thirdTenant = await signIn("alice", "tenant-c");
});

after(async () => {
  await app?.close();
  if (workerRoot) await rm(workerRoot, {recursive: true, force: true});
});

test("real HTTP and SSE use distinct workers for two tenants with the same request id", async () => {
  const bootstrapA = await gatewayJson(aliceA, "api/bootstrap");
  const bootstrapB = await gatewayJson(aliceB, "api/bootstrap");
  assert.equal(bootstrapA.status, 200);
  assert.equal(bootstrapB.status, 200);

  const sharedRequestId = "same-browser-request-id";
  const [streamA, streamB] = await Promise.all([
    startStream(
      aliceA,
      bootstrapA.body.default_goal,
      bootstrapA.body.default_student_profile,
      sharedRequestId,
      "same-start-key"
    ),
    startStream(
      aliceB,
      bootstrapB.body.default_goal,
      bootstrapB.body.default_student_profile,
      sharedRequestId,
      "same-start-key"
    )
  ]);
  assert.equal(streamA.response.status, 200);
  assert.equal(streamB.response.status, 200);
  assert.match(streamA.text, /event: run\.completed/);
  assert.match(streamB.text, /event: run\.completed/);
  assert.notEqual(
    streamA.response.headers.get("x-harness-run-id"),
    streamB.response.headers.get("x-harness-run-id")
  );
  assert.notEqual(operationSessionId(streamA), operationSessionId(streamB));

  const replayA = await startStream(
    aliceA,
    bootstrapA.body.default_goal,
    bootstrapA.body.default_student_profile,
    sharedRequestId,
    "same-start-key"
  );
  assert.equal(
    replayA.response.headers.get("x-harness-run-id"),
    streamA.response.headers.get("x-harness-run-id")
  );
  assert.equal(operationSessionId(replayA), operationSessionId(streamA));

  const scopeDirectories = await readdir(join(workerRoot, "k2"));
  assert.equal(scopeDirectories.length, 2);
  assert.ok(scopeDirectories.every((name) => /^scope_[0-9a-f]{48}$/.test(name)));
  assert.ok(scopeDirectories.every((name) => !name.includes("tenant") && !name.includes("alice")));
});

test("real gateway mints a scope-bound teacher envelope and rejects browser authority before worker side effects", async () => {
  const bootstrap = await gatewayJson(aliceA, "api/bootstrap");
  assert.equal(bootstrap.status, 200);
  assert.deepEqual(bootstrap.body.teacher_authority, {
    mode: "authenticated_apps_api",
    role_authorized: true,
    correct_mastery_updates_enabled: true,
    assurance: "deployment_service_role_authorization_not_personal_signature",
    raw_identity_exposed: false,
    nonce_replay_policy: "append_only_permanent_tombstone_bounded_fail_closed",
    replay_store_max_bytes: 16 * 1024 * 1024,
    expired_nonce_tombstones_retained: true
  });
  assert.equal(JSON.stringify(bootstrap.body).includes("tenant-a"), false);
  assert.equal(JSON.stringify(bootstrap.body).includes("alice"), false);

  const request = {
    session_id: "teach_missing_authority_route",
    expected_round: 0,
    expected_question_id: "question_missing",
    expected_context_version: 1,
    profile_revision: "profile_missing",
    item_id: `adj_${"7".repeat(24)}`,
    expected_version: 1,
    adjudication_idempotency_key: "real-gateway-authority-route"
  };
  const signed = await gatewayJson(
    aliceA,
    "api/adjudication/claim",
    "POST",
    request
  );
  assert.equal(signed.status, 400);

  const replayFiles = (await readdir(join(workerRoot, "k2")))
    .map((scope) => join(workerRoot, "k2", scope, "teacher_authority_replay.jsonl"));
  const existing = [];
  for (const path of replayFiles) {
    try {
      existing.push(await readFile(path, "utf8"));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
  assert.equal(existing.flatMap((value) => value.trim().split("\n")).length, 1);
  assert.doesNotMatch(existing.join(""), /tenant-a|alice|real-gateway-authority-route/);

  const forged = await gatewayJson(
    aliceA,
    "api/adjudication/claim",
    "POST",
    {...request, actor: {identity: "authenticated_teacher_server_authorized"}}
  );
  assert.equal(forged.status, 400);
  const afterForgery = [];
  for (const path of replayFiles) {
    try {
      afterForgery.push(await readFile(path, "utf8"));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
  assert.deepEqual(afterForgery, existing);
});

test("session, task status, and cancellation guesses cannot cross tenant scope", async () => {
  const bootstrap = await gatewayJson(aliceA, "api/bootstrap");
  const stream = await startStream(
    aliceA,
    bootstrap.body.default_goal,
    bootstrap.body.default_student_profile,
    "ownership-stream-request",
    "ownership-start-key"
  );
  const sessionId = operationSessionId(stream);
  const taskId = stream.response.headers.get("x-background-task-id");
  const taskVersion = Number(stream.response.headers.get("x-background-task-version"));
  assert.match(taskId ?? "", /^task_[0-9a-f]{40}$/);
  assert.ok(Number.isInteger(taskVersion));

  const ownerResume = await gatewayJson(aliceA, "api/session", "POST", {
    session_id: sessionId
  });
  assert.equal(ownerResume.status, 200);
  const guessedResume = await gatewayJson(aliceB, "api/session", "POST", {
    session_id: sessionId
  });
  assert.notEqual(guessedResume.status, 200);

  const ownerStatus = await gatewayJson(aliceA, "api/tasks/status", "POST", {
    task_id: taskId
  });
  assert.equal(ownerStatus.status, 200);
  const guessedStatus = await gatewayJson(aliceB, "api/tasks/status", "POST", {
    task_id: taskId
  });
  assert.notEqual(guessedStatus.status, 200);

  const guessedTaskCancel = await gatewayJson(aliceB, "api/tasks/cancel", "POST", {
    task_id: taskId,
    expected_version: ownerStatus.body.task.version,
    task_idempotency_key: "cross-tenant-cancel"
  });
  assert.notEqual(guessedTaskCancel.status, 200);
  const guessedStreamCancel = await gatewayJson(aliceB, "api/cancel", "POST", {
    run_id: stream.response.headers.get("x-harness-run-id"),
    request_id: "ownership-stream-request",
    reason: "user_requested"
  });
  assert.notEqual(guessedStreamCancel.status, 200);

  const ownerCancel = await gatewayJson(aliceA, "api/cancel", "POST", {
    run_id: stream.response.headers.get("x-harness-run-id"),
    request_id: "ownership-stream-request",
    reason: "user_requested"
  });
  assert.equal(ownerCancel.status, 200);
  assert.equal(typeof ownerCancel.body.commit_won, "boolean");
  assert.equal((await gatewayJson(aliceA, "api/session", "POST", {session_id: sessionId})).status, 200);
});

test("worker crash cleans the slot and durable state resumes in a fresh worker", async () => {
  const bootstrap = await gatewayJson(aliceA, "api/bootstrap");
  const stream = await startStream(
    aliceA,
    bootstrap.body.default_goal,
    bootstrap.body.default_student_profile,
    "restart-stream-request",
    "restart-start-key"
  );
  const sessionId = operationSessionId(stream);
  const workers = app.get(HarnessWorkerPoolService);
  assert.equal(await workers.terminateForTesting({tenantId: "tenant-a", ownerId: "alice"}), true);
  assert.equal(workers.status().readyWorkers, 1);

  const resumed = await gatewayJson(aliceA, "api/session", "POST", {
    session_id: sessionId
  });
  assert.equal(resumed.status, 200);
  assert.equal(resumed.body.session_id, sessionId);
  assert.equal(workers.status().readyWorkers, 2);

  const capacity = await gatewayJson(thirdTenant, "api/bootstrap");
  assert.equal(capacity.status, 503);
  assert.equal(capacity.body.code, "harness_capacity_exhausted");
  assert.equal(workers.status().readyWorkers, 2);

  assert.equal(
    await workers.terminateForTesting({tenantId: "tenant-b", ownerId: "alice"}),
    true
  );
  const startupWindow = await Promise.all(
    Array.from({length: 4}, () => gatewayJson(thirdTenant, "api/bootstrap"))
  );
  assert.ok(startupWindow.every((response) => response.status === 200));
  assert.equal(workers.status().readyWorkers, 2);
  assert.equal(workers.status().startingWorkers, 0);
});

test("expired LRU eviction admits a new real scope and an evicted scope restarts from durable data", async () => {
  const workers = app.get(HarnessWorkerPoolService);
  let fakeNow = Date.now();
  workers.setClockForTesting(() => fakeNow);
  const bootstrapA = await gatewayJson(aliceA, "api/bootstrap");
  const stream = await startStream(
    aliceA,
    bootstrapA.body.default_goal,
    bootstrapA.body.default_student_profile,
    "idle-eviction-durable-session",
    "idle-eviction-durable-start"
  );
  const durableSessionId = operationSessionId(stream);
  fakeNow += 1;
  assert.equal((await gatewayJson(thirdTenant, "api/bootstrap")).status, 200);

  const fourthTenant = await signIn("alice", "tenant-d");
  const beforeExpiry = await gatewayJson(fourthTenant, "api/bootstrap");
  assert.equal(beforeExpiry.status, 503);
  assert.equal(beforeExpiry.body.code, "harness_capacity_exhausted");

  fakeNow += workers.status().idleTimeoutMs + 1;
  const admitted = await gatewayJson(fourthTenant, "api/bootstrap");
  assert.equal(admitted.status, 200);
  assert.equal(workers.status().readyWorkers, 2);
  assert.equal(workers.status().stoppingWorkers, 0);

  const resumed = await gatewayJson(aliceA, "api/session", "POST", {
    session_id: durableSessionId
  });
  assert.equal(resumed.status, 200);
  assert.equal(resumed.body.session_id, durableSessionId);
  assert.equal(workers.status().readyWorkers, 2);
  assert.equal(workers.status().stoppingWorkers, 0);
});

test("HTTP and SSE failures expose neither private paths nor request capability material", async () => {
  const requestSentinel = "browser-capability-sentinel-do-not-reflect";
  const response = await fetch(`${origin}/api/v1/harness/api/stream`, {
    method: "POST",
    headers: {
      ...headers(aliceA, true),
      accept: "text/event-stream",
      "content-type": "application/json"
    },
    body: JSON.stringify({
      operation: "start",
      request_id: requestSentinel,
      payload: {invalid_private_path: workerRoot}
    })
  });
  const text = await response.text();
  assert.equal(response.status, 200);
  assert.match(text, /event: run\.failed/);
  assert.match(text, /请求未完成，请重试。/);
  assert.equal(text.includes(workerRoot), false);
  assert.equal(text.includes(requestSentinel), false);

  const rejected = await gatewayJson(aliceA, "api/session", "POST", {
    session_id: workerRoot
  });
  assert.notEqual(rejected.status, 200);
  const encoded = JSON.stringify(rejected.body);
  assert.equal(encoded.includes(workerRoot), false);
  assert.equal(encoded.includes("scope_"), false);
});

test("private worker artifacts contain no raw gateway tenant or owner metadata", async () => {
  const versions = await readdir(workerRoot);
  assert.deepEqual(versions.sort(), [".scope-key-migration-locks-v1", "k2"]);
  const migrationLocks = join(workerRoot, ".scope-key-migration-locks-v1");
  assert.equal((await stat(migrationLocks)).mode & 0o777, 0o700);
  assert.deepEqual(await readdir(migrationLocks), []);
  const scopes = await readdir(join(workerRoot, "k2"));
  for (const scope of scopes) {
    const root = join(workerRoot, "k2", scope);
    const mode = (await stat(root)).mode & 0o777;
    assert.equal(mode, 0o700);
    const candidates = await readdir(root, {recursive: true});
    for (const relative of candidates) {
      const path = join(root, relative);
      const metadata = await stat(path);
      if (!metadata.isFile() || metadata.size > 2 * 1024 * 1024) continue;
      const content = await readFile(path);
      assert.equal(content.includes(Buffer.from("tenant-a")), false);
      assert.equal(content.includes(Buffer.from("tenant-b")), false);
    }
  }
});
