import "reflect-metadata";

import assert from "node:assert/strict";
import {EventEmitter} from "node:events";
import {Readable} from "node:stream";
import test from "node:test";

import type {FastifyReply, FastifyRequest} from "fastify";

import {APPS_API_BODY_LIMIT_BYTES} from "../src/create-application";
import {
  HARNESS_REQUEST_LIMITS,
  HarnessGatewayController,
  MAX_PROJECT_EXPORT_RESPONSE_BYTES,
  boundedProjectExportStream,
  harnessRequestBodyAllowed,
  harnessRequestBodyLimit,
  projectExportResponseLength,
} from "../src/harness/harness-gateway.controller";
import type {HarnessWorkerPoolService} from "../src/harness/harness-worker-pool.service";
import type {AppConfigService} from "../src/config/app-config.service";

test("Fastify can admit the complete resource envelope to its authenticated route gate", () => {
  assert.equal(HARNESS_REQUEST_LIMITS.resource, 17 * 1024 * 1024);
  assert.equal(APPS_API_BODY_LIMIT_BYTES, 18 * 1024 * 1024);
  assert.ok(APPS_API_BODY_LIMIT_BYTES > HARNESS_REQUEST_LIMITS.resource);
  assert.equal(harnessRequestBodyLimit("api/resource"), 17 * 1024 * 1024);
  assert.equal(harnessRequestBodyLimit("api/resource/review"), 64 * 1024);
  assert.equal(harnessRequestBodyAllowed("api/resource", 17 * 1024 * 1024), true);
});

test("resource bytes over 17 MiB are rejected before worker forwarding", () => {
  assert.equal(harnessRequestBodyAllowed("api/resource", 17 * 1024 * 1024 + 1), false);
  assert.equal(harnessRequestBodyAllowed("api/resource", APPS_API_BODY_LIMIT_BYTES), false);
});

test("ordinary JSON cannot borrow the larger resource allowance", () => {
  assert.equal(harnessRequestBodyLimit("api/start"), 64 * 1024);
  assert.equal(harnessRequestBodyAllowed("api/start", 64 * 1024), true);
  assert.equal(harnessRequestBodyAllowed("api/start", 64 * 1024 + 1), false);
  assert.equal(harnessRequestBodyAllowed("api/tasks/resume", 2 * 1024 * 1024), false);
});

test("specialized route limits stay aligned with the Python boundary", () => {
  assert.equal(harnessRequestBodyLimit("api/attachment"), 6 * 1024 * 1024);
  assert.equal(harnessRequestBodyLimit("api/syllabi/syl_example/publish"), 384 * 1024);
  assert.equal(harnessRequestBodyLimit("api/projects/project_example/browse"), 2 * 1024 * 1024);
  assert.equal(harnessRequestBodyLimit("api/adjudication/decide"), 64 * 1024);
  assert.equal(harnessRequestBodyLimit("api/consent/grant"), 16 * 1024);
  assert.equal(harnessRequestBodyLimit("api/safeguarding/case/close"), 16 * 1024);
});

function requestBodyWithSerializedBytes(byteLength: number): Record<string, string> {
  const emptyBytes = Buffer.byteLength(JSON.stringify({blob: ""}), "utf8");
  assert.ok(byteLength >= emptyBytes);
  const body = {blob: "x".repeat(byteLength - emptyBytes)};
  assert.equal(Buffer.byteLength(JSON.stringify(body), "utf8"), byteLength);
  return body;
}

function fakeRequest(body: Record<string, unknown>): FastifyRequest {
  const raw = new EventEmitter();
  return {
    method: "POST",
    body,
    headers: {accept: "application/json"},
    raw,
  } as unknown as FastifyRequest;
}

function fakeReply() {
  const state = {status: 200, payload: undefined as unknown, headers: new Map<string, string>()};
  const raw = new EventEmitter() as EventEmitter & {writableEnded: boolean; destroy: () => void};
  raw.writableEnded = false;
  raw.destroy = () => { raw.writableEnded = true; };
  const reply = {
    raw,
    status(code: number) {
      state.status = code;
      return this;
    },
    header(name: string, value: unknown) {
      state.headers.set(name.toLowerCase(), String(value));
      return this;
    },
    send(payload?: unknown) {
      state.payload = payload;
      return this;
    },
  } as unknown as FastifyReply;
  return {reply, state};
}

test("controller forwards a 17 MiB resource but rejects larger and ordinary oversized JSON with zero worker call", async () => {
  let workerCalls = 0;
  const workers = {
    async request() {
      workerCalls += 1;
      const response = Readable.from([Buffer.from("{}", "utf8")]) as Readable & {
        statusCode: number;
        headers: Record<string, string>;
      };
      response.statusCode = 200;
      response.headers = {"content-type": "application/json"};
      return response;
    },
  } as unknown as HarnessWorkerPoolService;
  const config = {harnessResponseMaxBytes: 64 * 1024} as AppConfigService;
  const controller = new HarnessGatewayController(workers, config);
  const principal = {
    tenantId: "tenant-a",
    subject: "owner-a",
    sessionId: "session-a",
    provider: "oidc" as const,
    roles: [],
    scopeTenantId: "tenant-a",
    scopeOwnerId: "owner-a",
  };

  const acceptedReply = fakeReply();
  await controller.proxy(
    principal,
    "api/resource",
    fakeRequest(requestBodyWithSerializedBytes(HARNESS_REQUEST_LIMITS.resource)),
    acceptedReply.reply,
  );
  assert.equal(acceptedReply.state.status, 200);
  assert.equal(workerCalls, 1);

  const tooLargeReply = fakeReply();
  await controller.proxy(
    principal,
    "api/resource",
    fakeRequest(requestBodyWithSerializedBytes(HARNESS_REQUEST_LIMITS.resource + 1)),
    tooLargeReply.reply,
  );
  assert.equal(tooLargeReply.state.status, 413);
  assert.equal(workerCalls, 1);

  const ordinaryReply = fakeReply();
  await controller.proxy(
    principal,
    "api/start",
    fakeRequest(requestBodyWithSerializedBytes(HARNESS_REQUEST_LIMITS.default + 1)),
    ordinaryReply.reply,
  );
  assert.equal(ordinaryReply.state.status, 413);
  assert.equal(workerCalls, 1);
});

test("synchronous Chat requires a stable occurrence identity before any worker call", async () => {
  let workerCalls = 0;
  const workers = {
    async request() {
      workerCalls += 1;
      const response = Readable.from([Buffer.from("{}", "utf8")]) as Readable & {
        statusCode: number;
        headers: Record<string, string>;
      };
      response.statusCode = 200;
      response.headers = {"content-type": "application/json"};
      return response;
    },
  } as unknown as HarnessWorkerPoolService;
  const controller = new HarnessGatewayController(
    workers,
    {harnessResponseMaxBytes: 64 * 1024} as AppConfigService,
  );
  const principal = {
    tenantId: "tenant-a",
    subject: "owner-a",
    sessionId: "session-a",
    provider: "oidc" as const,
    roles: [],
    scopeTenantId: "tenant-a",
    scopeOwnerId: "owner-a",
  };
  for (const requestId of [undefined, "short", "unsafe identity!"]) {
    const rejected = fakeReply();
    await controller.proxy(
      principal,
      "api/chat",
      fakeRequest({
        ...(requestId === undefined ? {} : {request_id: requestId}),
        messages: [{role: "user", content: "hello"}],
      }),
      rejected.reply,
    );
    assert.equal(rejected.state.status, 400);
  }
  assert.equal(workerCalls, 0);

  const accepted = fakeReply();
  await controller.proxy(
    principal,
    "api/chat",
    fakeRequest({
      request_id: "chat-observation-0001",
      messages: [{role: "user", content: "hello"}],
    }),
    accepted.reply,
  );
  assert.equal(accepted.state.status, 200);
  assert.equal(workerCalls, 1);
});

test("browser policy fields are rejected while signed principal policy stays private", async () => {
  const calls: unknown[] = [];
  const workers = {
    async request(_scope: unknown, input: unknown) {
      calls.push(input);
      const response = Readable.from([Buffer.from("{}", "utf8")]) as Readable & {
        statusCode: number;
        headers: Record<string, string>;
      };
      response.statusCode = 200;
      response.headers = {"content-type": "application/json"};
      return response;
    }
  } as unknown as HarnessWorkerPoolService;
  const controller = new HarnessGatewayController(
    workers,
    {harnessResponseMaxBytes: 64 * 1024} as AppConfigService
  );
  const remoteSubjectPolicy = {
    policy_id: "school-policy",
    policy_version: "v3",
    policy_source: "organization_oidc_or_roster_policy" as const,
    likely_minor: false,
    guardian_or_school_policy: "not_required" as const,
    remote_processing_eligible: true
  };
  const principal = {
    tenantId: "tenant-a",
    subject: "owner-a",
    sessionId: "session-a",
    provider: "oidc" as const,
    roles: [],
    scopeTenantId: "tenant-a",
    scopeOwnerId: "owner-a",
    remoteSubjectPolicy
  };
  const rejected = fakeReply();
  await controller.proxy(
    principal,
    "api/start",
    fakeRequest({goal: "learn", student_profile: {likely_minor: false}}),
    rejected.reply
  );
  assert.equal(rejected.state.status, 400);
  assert.equal(calls.length, 0);

  const accepted = fakeReply();
  await controller.proxy(
    principal,
    "api/start",
    fakeRequest({goal: "learn", student_profile: {profile_ref: "browser-choice"}}),
    accepted.reply
  );
  assert.equal(accepted.state.status, 200);
  assert.equal(calls.length, 1);
  const input = calls[0] as Record<string, unknown>;
  assert.deepEqual(input.remoteSubjectPolicy, remoteSubjectPolicy);
  assert.doesNotMatch((input.body as Buffer).toString("utf8"), /school-policy|eligible/);
});

function incomingResponse(
  source: Iterable<Buffer>,
  contentLength: number,
): Readable & {statusCode: number; headers: Record<string, string>} {
  const response = Readable.from(source) as Readable & {
    statusCode: number;
    headers: Record<string, string>;
  };
  response.statusCode = 200;
  response.headers = {
    "content-type": "application/zip",
    "content-length": String(contentLength),
    "content-disposition": 'attachment; filename="project_private.zip"',
    "x-manifest-sha256": "a".repeat(64),
  };
  return response;
}

test("project export metadata accepts a synthetic stream above the JSON buffer limit", async () => {
  const chunk = Buffer.alloc(1024 * 1024, 0x61);
  const declaredBytes = 40 * 1024 * 1024;
  const upstream = incomingResponse(Array.from({length: 40}, () => chunk), declaredBytes);
  assert.equal(projectExportResponseLength(upstream.headers), declaredBytes);
  const stream = boundedProjectExportStream(upstream as never, declaredBytes);
  let received = 0;
  for await (const part of stream) received += Buffer.byteLength(part);
  assert.equal(received, declaredBytes);
});

test("project export controller streams above 32 MiB, caps 256 MiB, and detaches on client close", async () => {
  const chunk = Buffer.alloc(1024 * 1024, 0x61);
  const declaredBytes = 40 * 1024 * 1024;
  let nextResponse = incomingResponse(Array.from({length: 40}, () => chunk), declaredBytes);
  let workerCalls = 0;
  const workers = {
    async request() {
      workerCalls += 1;
      return nextResponse;
    },
  } as unknown as HarnessWorkerPoolService;
  const controller = new HarnessGatewayController(
    workers,
    {harnessResponseMaxBytes: 32 * 1024 * 1024} as AppConfigService,
  );
  const principal = {
    tenantId: "tenant-a",
    subject: "owner-a",
    sessionId: "session-a",
    provider: "oidc" as const,
    roles: [],
    scopeTenantId: "tenant-a",
    scopeOwnerId: "owner-a",
  };
  const projectPath = `api/projects/project_${"1".repeat(24)}/export`;
  const streamedReply = fakeReply();
  await controller.proxy(principal, projectPath, {
    method: "GET",
    headers: {accept: "application/zip"},
    raw: new EventEmitter(),
  } as unknown as FastifyRequest, streamedReply.reply);
  assert.equal(streamedReply.state.status, 200);
  assert.equal(streamedReply.state.headers.get("content-length"), String(declaredBytes));
  let streamedBytes = 0;
  for await (const part of streamedReply.state.payload as Readable) {
    streamedBytes += Buffer.byteLength(part);
  }
  assert.equal(streamedBytes, declaredBytes);

  nextResponse = incomingResponse(
    [Buffer.from("too-large")],
    MAX_PROJECT_EXPORT_RESPONSE_BYTES + 1,
  );
  const rejectedReply = fakeReply();
  await controller.proxy(principal, projectPath, {
    method: "GET",
    headers: {accept: "application/zip"},
    raw: new EventEmitter(),
  } as unknown as FastifyRequest, rejectedReply.reply);
  assert.equal(rejectedReply.state.status, 502);
  assert.equal(workerCalls, 2);

  nextResponse = new Readable({read() { /* remain attached until the client closes */ }}) as Readable & {
    statusCode: number;
    headers: Record<string, string>;
  };
  nextResponse.statusCode = 200;
  nextResponse.headers = {
    "content-type": "application/zip",
    "content-length": String(declaredBytes),
    "content-disposition": 'attachment; filename="project_private.zip"',
    "x-manifest-sha256": "a".repeat(64),
  };
  const detachedReply = fakeReply();
  await controller.proxy(principal, projectPath, {
    method: "GET",
    headers: {accept: "application/zip"},
    raw: new EventEmitter(),
  } as unknown as FastifyRequest, detachedReply.reply);
  detachedReply.reply.raw.emit("close");
  assert.equal(nextResponse.destroyed, true);
});
