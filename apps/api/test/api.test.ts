import "reflect-metadata";

import assert from "node:assert/strict";
import {after, before, test} from "node:test";

import type {NestFastifyApplication} from "@nestjs/platform-fastify";
import {firstValueFrom} from "rxjs";

import {createApplication} from "../src/create-application";
import type {SessionRevocationRepositoryPort} from "../src/auth/session-revocation.repository.port";
import type {EventStreamPort} from "../src/events/event-stream.port";
import {EventsService} from "../src/events/events.service";
import {EVENT_STREAM, SESSION_REVOCATION_REPOSITORY} from "../src/platform/tokens";
import type {AgentTask} from "../src/tasks/task.types";

process.env.NODE_ENV = "test";
process.env.AUTH_MODE = "development";
process.env.DATA_BACKEND = "memory";
process.env.DEV_AUTH_USER_ID = "test-user";
process.env.DEV_AUTH_TENANT_ID = "test-tenant";
process.env.DEV_AUTH_ALLOW_HEADERS = "true";
process.env.SEED_DEMO_SESSIONS = "false";
process.env.SESSION_SECRET = "test-only-session-secret-at-least-32-characters";
process.env.SESSION_COOKIE_SECURE = "false";
process.env.SSE_HEARTBEAT_MS = "1000";

interface BrowserSession {
  cookie: string;
  csrfToken: string;
  principal: {subject: string; tenantId: string};
}

let app: NestFastifyApplication;

function setCookieValues(value: string | string[] | undefined): string[] {
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
  const body = response.json();
  const cookie = setCookieValues(response.headers["set-cookie"])
    .map((entry) => entry.split(";", 1)[0])
    .join("; ");
  assert.match(cookie, /teachlab_session=/);
  assert.match(cookie, /teachlab_csrf=/);
  assert.equal(body.authenticated, true);
  assert.equal("principal" in body, false);
  return {cookie, csrfToken: body.csrfToken, principal: {subject: userId, tenantId}};
}

function readHeaders(session: BrowserSession): Record<string, string> {
  return {cookie: session.cookie};
}

function writeHeaders(session: BrowserSession): Record<string, string> {
  return {
    cookie: session.cookie,
    "x-csrf-token": session.csrfToken,
    origin: "http://localhost:3000",
    "sec-fetch-site": "same-site"
  };
}

before(async () => {
  app = await createApplication({logger: false});
  await app.init();
  await app.getHttpAdapter().getInstance().ready();
});

after(async () => {
  await app.close();
});

test("public health is honest about local-only adapters", async () => {
  const response = await app.inject({method: "GET", url: "/health"});
  assert.equal(response.statusCode, 200);
  const body = response.json();
  assert.equal(body.status, "ok");
  assert.equal(body.runtime.identityBoundary, "server_minted_http_only_session");
  assert.equal(body.runtime.sessionRepository, "memory");
  assert.equal(
    body.runtime.sessionRevocationDurability,
    "process_local_restart_invalidates_sessions"
  );
  assert.equal(body.runtime.queue, "memory_local_only");
  assert.equal(body.runtime.artifactStorage, "memory_local_only");
  assert.equal(body.model.configured, false);
});

test("server mints HttpOnly sessions and rejects unsigned identity headers", async () => {
  const preflight = await app.inject({
    method: "OPTIONS",
    url: "/api/v1/sessions",
    headers: {
      origin: "http://localhost:3000",
      "access-control-request-method": "POST",
      "access-control-request-headers": "content-type,x-csrf-token"
    }
  });
  assert.equal(preflight.statusCode, 204);
  assert.equal(preflight.headers["access-control-allow-origin"], "http://localhost:3000");

  const unauthenticated = await app.inject({
    method: "GET",
    url: "/api/v1/sessions",
    headers: {"x-dev-user-id": "alice", "x-dev-tenant-id": "tenant-a"}
  });
  assert.equal(unauthenticated.statusCode, 401);

  const login = await app.inject({
    method: "POST",
    url: "/api/v1/auth/session",
    headers: {
      origin: "http://localhost:3000",
      "sec-fetch-site": "same-site",
      "x-dev-user-id": "alice",
      "x-dev-tenant-id": "tenant-a"
    }
  });
  assert.equal(login.statusCode, 201);
  const cookies = setCookieValues(login.headers["set-cookie"]);
  const sessionCookie = cookies.find((value) => value.startsWith("teachlab_session="));
  const csrfCookie = cookies.find((value) => value.startsWith("teachlab_csrf="));
  assert.match(sessionCookie ?? "", /HttpOnly/);
  assert.match(sessionCookie ?? "", /SameSite=Strict/);
  assert.doesNotMatch(csrfCookie ?? "", /HttpOnly/);
  assert.equal(login.json().authenticated, true);
  assert.equal("principal" in login.json(), false);
  assert.equal(login.json().mode, "local_only");

  const missingOrigin = await app.inject({
    method: "POST",
    url: "/api/v1/auth/session",
    headers: {"x-dev-user-id": "alice", "x-dev-tenant-id": "tenant-a"}
  });
  assert.equal(missingOrigin.statusCode, 403);

  const missingFetchMetadata = await app.inject({
    method: "POST",
    url: "/api/v1/auth/session",
    headers: {
      origin: "http://localhost:3000",
      "x-dev-user-id": "alice",
      "x-dev-tenant-id": "tenant-a"
    }
  });
  assert.equal(missingFetchMetadata.statusCode, 403);

  const untrusted = await app.inject({
    method: "POST",
    url: "/api/v1/auth/session",
    headers: {origin: "https://attacker.example", "sec-fetch-site": "cross-site"}
  });
  assert.equal(untrusted.statusCode, 403);
});

test("logout durably revokes first, is idempotent, and keeps other sessions active", async () => {
  const current = await signIn("logout-alice", "tenant-a");
  const otherDevice = await signIn("logout-alice", "tenant-a");
  const otherUser = await signIn("logout-bob", "tenant-a");

  const concurrentLogout = await Promise.all([0, 1].map(() => app.inject({
    method: "DELETE",
    url: "/api/v1/auth/session",
    headers: writeHeaders(current)
  })));
  for (const loggedOut of concurrentLogout) {
    assert.equal(loggedOut.statusCode, 204);
    const expiredCookies = setCookieValues(loggedOut.headers["set-cookie"]);
    assert.equal(expiredCookies.length, 2);
    assert.ok(expiredCookies.every((cookie) => /Max-Age=0/.test(cookie)));
  }

  const revokedUse = await app.inject({
    method: "GET",
    url: "/api/v1/sessions",
    headers: readHeaders(current)
  });
  assert.equal(revokedUse.statusCode, 401);

  for (const stillActive of [otherDevice, otherUser]) {
    const response = await app.inject({
      method: "GET",
      url: "/api/v1/sessions",
      headers: readHeaders(stillActive)
    });
    assert.equal(response.statusCode, 200);
  }

  const missingCsrf = await app.inject({
    method: "DELETE",
    url: "/api/v1/auth/session",
    headers: readHeaders(otherDevice)
  });
  assert.equal(missingCsrf.statusCode, 403);

  const missingFetchMetadata = await app.inject({
    method: "DELETE",
    url: "/api/v1/auth/session",
    headers: {
      cookie: otherDevice.cookie,
      "x-csrf-token": otherDevice.csrfToken,
      origin: "http://localhost:3000"
    }
  });
  assert.equal(missingFetchMetadata.statusCode, 403);
});

test("session authority failures fail closed without clearing or issuing cookies", async () => {
  const repository = app.get<SessionRevocationRepositoryPort>(
    SESSION_REVOCATION_REPOSITORY
  );
  const originalRegister = repository.register.bind(repository);
  repository.register = async () => {
    throw new Error("simulated register outage");
  };
  try {
    const failedLogin = await app.inject({
      method: "POST",
      url: "/api/v1/auth/session",
      headers: {
        origin: "http://localhost:3000",
        "sec-fetch-site": "same-site",
        "x-dev-user-id": "store-failure-user",
        "x-dev-tenant-id": "tenant-a"
      }
    });
    assert.equal(failedLogin.statusCode, 503);
    assert.equal(failedLogin.headers["set-cookie"], undefined);
  } finally {
    repository.register = originalRegister;
  }

  const session = await signIn("store-failure-user", "tenant-a");
  const originalRevoke = repository.revoke.bind(repository);
  repository.revoke = async () => {
    throw new Error("simulated revoke outage");
  };
  try {
    const failedLogout = await app.inject({
      method: "DELETE",
      url: "/api/v1/auth/session",
      headers: writeHeaders(session)
    });
    assert.equal(failedLogout.statusCode, 503);
    assert.equal(failedLogout.headers["set-cookie"], undefined);
  } finally {
    repository.revoke = originalRevoke;
  }

  const stillActive = await app.inject({
    method: "GET",
    url: "/api/v1/sessions",
    headers: readHeaders(session)
  });
  assert.equal(stillActive.statusCode, 200);
});

test("two users and two tenants are isolated through signed principals", async () => {
  const alice = await signIn("alice", "tenant-a");
  const bob = await signIn("bob", "tenant-a");
  const otherTenantAlice = await signIn("alice", "tenant-b");

  const invalid = await app.inject({
    method: "POST",
    url: "/api/v1/sessions",
    headers: writeHeaders(alice),
    payload: {title: "", learner: "Ada", unexpected: true}
  });
  assert.equal(invalid.statusCode, 400);

  const created = await app.inject({
    method: "POST",
    url: "/api/v1/sessions",
    headers: {...writeHeaders(alice), "x-dev-user-id": "bob"},
    payload: {title: "Verify dynamic programming", learner: "Ada"}
  });
  assert.equal(created.statusCode, 201);
  const session = created.json();
  assert.equal(session.ownerId, "alice");
  assert.equal(session.tenantId, "tenant-a");
  assert.equal(session.version, 1);
  assert.equal(created.headers.etag, '"1"');

  for (const intruder of [bob, otherTenantAlice]) {
    const hidden = await app.inject({
      method: "GET",
      url: `/api/v1/sessions/${session.id}`,
      headers: readHeaders(intruder)
    });
    assert.equal(hidden.statusCode, 404);
  }

  const bobList = await app.inject({
    method: "GET",
    url: "/api/v1/sessions",
    headers: readHeaders(bob)
  });
  assert.deepEqual(bobList.json(), []);

  const noCsrf = await app.inject({
    method: "PATCH",
    url: `/api/v1/sessions/${session.id}`,
    headers: {...readHeaders(alice), "if-match": "1"},
    payload: {title: "Unsafe overwrite"}
  });
  assert.equal(noCsrf.statusCode, 403);

  const noPrecondition = await app.inject({
    method: "PATCH",
    url: `/api/v1/sessions/${session.id}`,
    headers: writeHeaders(alice),
    payload: {title: "Missing concurrency token"}
  });
  assert.equal(noPrecondition.statusCode, 428);

  const updated = await app.inject({
    method: "PATCH",
    url: `/api/v1/sessions/${session.id}`,
    headers: {...writeHeaders(alice), "if-match": '"1"'},
    payload: {title: "Versioned update"}
  });
  assert.equal(updated.statusCode, 200);
  assert.equal(updated.json().version, 2);
  assert.equal(updated.headers.etag, '"2"');

  const stale = await app.inject({
    method: "PATCH",
    url: `/api/v1/sessions/${session.id}`,
    headers: {...writeHeaders(alice), "if-match": "1"},
    payload: {title: "Lost update"}
  });
  assert.equal(stale.statusCode, 409);
  assert.equal(stale.json().currentVersion, 2);

  const crossUserWrite = await app.inject({
    method: "PATCH",
    url: `/api/v1/sessions/${session.id}`,
    headers: {...writeHeaders(bob), "if-match": "2"},
    payload: {title: "Cross-user overwrite"}
  });
  assert.equal(crossUserWrite.statusCode, 404);

  const terminal = await app.inject({
    method: "PATCH",
    url: `/api/v1/sessions/${session.id}`,
    headers: {...writeHeaders(alice), "if-match": "2"},
    payload: {status: "succeeded"}
  });
  assert.equal(terminal.statusCode, 200);
  assert.equal(terminal.json().version, 3);

  const reopen = await app.inject({
    method: "PATCH",
    url: `/api/v1/sessions/${session.id}`,
    headers: {...writeHeaders(alice), "if-match": "3"},
    payload: {status: "active"}
  });
  assert.equal(reopen.statusCode, 409);
  assert.equal(reopen.json().code, "session_state_conflict");
});

test("task ownership, cancellation, and idempotency remain tenant scoped", async () => {
  const alice = await signIn("task-alice", "tenant-a");
  const bob = await signIn("task-bob", "tenant-a");
  const created = await app.inject({
    method: "POST",
    url: "/api/v1/sessions",
    headers: writeHeaders(alice),
    payload: {title: "Task isolation", learner: "Ada"}
  });
  const sessionId = created.json().id as string;

  const response = await app.inject({
    method: "POST",
    url: `/api/v1/sessions/${sessionId}/tasks`,
    headers: writeHeaders(alice),
    payload: {
      message: "I can explain the state transition.",
      clientRequestId: "turn-1"
    }
  });
  assert.equal(response.statusCode, 202);
  const queued = response.json() as AgentTask;
  assert.equal(queued.ownerId, "task-alice");
  assert.equal(queued.tenantId, "tenant-a");

  const replay = await app.inject({
    method: "POST",
    url: `/api/v1/sessions/${sessionId}/tasks`,
    headers: writeHeaders(alice),
    payload: {
      message: "I can explain the state transition.",
      clientRequestId: "turn-1"
    }
  });
  assert.equal(replay.statusCode, 202);
  assert.equal(replay.json().id, queued.id);

  const conflictingReplay = await app.inject({
    method: "POST",
    url: `/api/v1/sessions/${sessionId}/tasks`,
    headers: writeHeaders(alice),
    payload: {message: "Different input", clientRequestId: "turn-1"}
  });
  assert.equal(conflictingReplay.statusCode, 409);

  const hidden = await app.inject({
    method: "GET",
    url: `/api/v1/tasks/${queued.id}`,
    headers: readHeaders(bob)
  });
  assert.equal(hidden.statusCode, 404);

  const crossUserCancel = await app.inject({
    method: "POST",
    url: `/api/v1/tasks/${queued.id}/cancel`,
    headers: writeHeaders(bob)
  });
  assert.equal(crossUserCancel.statusCode, 404);

  await new Promise<void>((resolve) => setImmediate(resolve));
  const status = await app.inject({
    method: "GET",
    url: `/api/v1/tasks/${queued.id}`,
    headers: readHeaders(alice)
  });
  assert.equal(status.statusCode, 200);
  assert.equal((status.json() as AgentTask).status, "requires_configuration");
});

test("event replay is scoped and the service refuses a non-owner", async () => {
  const alice = await signIn("event-alice", "tenant-a");
  const bob = await signIn("event-bob", "tenant-a");
  const created = await app.inject({
    method: "POST",
    url: "/api/v1/sessions",
    headers: writeHeaders(alice),
    payload: {title: "Event isolation", learner: "Ada"}
  });
  const sessionId = created.json().id as string;
  const events = app.get<EventStreamPort>(EVENT_STREAM);
  const appended = await events.append({
    tenantId: "tenant-a",
    ownerId: "event-alice",
    sessionId,
    type: "message",
    payload: {role: "teacher", body: "Verification question"}
  });
  const replayed = await firstValueFrom(
    events.stream({tenantId: "tenant-a", ownerId: "event-alice"}, sessionId)
  );
  assert.equal(replayed.sessionId, sessionId);
  assert.match(replayed.id, /^evt_/);
  assert.ok(appended.id.localeCompare(replayed.id) >= 0);
  assert.equal("tenantId" in replayed, false);
  assert.equal("ownerId" in replayed, false);

  const eventService = app.get(EventsService);
  await assert.rejects(
    firstValueFrom(
      eventService.streamOwned(
        {tenantId: bob.principal.tenantId, ownerId: bob.principal.subject},
        sessionId
      )
    ),
    /Session not found/
  );

  const deniedRoute = await app.inject({
    method: "GET",
    url: `/api/v1/sessions/${sessionId}/events`,
    headers: readHeaders(bob)
  });
  assert.equal(deniedRoute.statusCode, 404);
});

test("bootstrap requires a signed session and exposes no secrets", async () => {
  const anonymous = await app.inject({method: "GET", url: "/api/bootstrap"});
  assert.equal(anonymous.statusCode, 401);
  const alice = await signIn("bootstrap-alice", "tenant-a");
  const response = await app.inject({
    method: "GET",
    url: "/api/bootstrap",
    headers: readHeaders(alice)
  });
  assert.equal(response.statusCode, 200);
  const body = response.json();
  assert.equal(body.interaction_contract.transport, "sse");
  assert.equal(body.provider_status.provider, "anthropic");
  assert.equal(JSON.stringify(body).includes("SESSION_SECRET"), false);
  assert.equal(JSON.stringify(body).includes("ANTHROPIC_API_KEY"), false);
});
