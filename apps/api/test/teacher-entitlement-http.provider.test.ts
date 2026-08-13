import assert from "node:assert/strict";
import test from "node:test";

import {TeacherEntitlementAuthorizationService} from "../src/auth/teacher-entitlement-authorization";
import {
  TEACHER_ENTITLEMENT_LOOKUP_SCHEMA,
  TEACHER_ENTITLEMENT_SNAPSHOT_SCHEMA,
  TeacherEntitlementHttpProvider
} from "../src/auth/teacher-entitlement-http.provider";
import type {
  CanonicalEntitlementIdentity,
  TeacherEntitlementClock
} from "../src/auth/teacher-entitlement.port";

const NOW_MS = Date.parse("2026-08-12T04:00:00.000Z");
const SECRET = `directory_${"s".repeat(48)}`;
const ENDPOINT = "https://entitlements.organization.example/v1/teacher/snapshot";
const IDENTITY: CanonicalEntitlementIdentity = {
  issuer: "https://identity.organization.example",
  tenantId: "tenant-private-1",
  subject: "teacher-private-1"
};
const CLOCK: TeacherEntitlementClock = {now: () => new Date(NOW_MS)};

function wireSnapshot(overrides: Record<string, unknown> = {}) {
  return {
    schema: TEACHER_ENTITLEMENT_SNAPSHOT_SCHEMA,
    identity: {
      issuer: IDENTITY.issuer,
      tenant_id: IDENTITY.tenantId,
      subject: IDENTITY.subject
    },
    status: "active",
    roles: ["teacher", "curriculum-editor"],
    policy_version: "roles-v7",
    revision: 42,
    evaluated_at: new Date(NOW_MS).toISOString(),
    expires_at: new Date(NOW_MS + 30_000).toISOString(),
    ...overrides
  };
}

function jsonResponse(
  value: unknown,
  options: {status?: number; headers?: Record<string, string>} = {}
) {
  const body = typeof value === "string" ? value : JSON.stringify(value);
  return new Response(body, {
    status: options.status ?? 200,
    headers: {
      "content-type": "application/json",
      ...options.headers
    }
  });
}

function provider(
  fetchImplementation: typeof fetch,
  overrides: Partial<ConstructorParameters<typeof TeacherEntitlementHttpProvider>[0]> = {}
) {
  return new TeacherEntitlementHttpProvider({
    endpoint: ENDPOINT,
    bearerSecret: SECRET,
    timeoutMs: 100,
    maxResponseBytes: 8_192,
    maxClockSkewMs: 1_000,
    clock: CLOCK,
    fetch: fetchImplementation,
    ...overrides
  });
}

async function safeFailure(promise: Promise<unknown>) {
  await assert.rejects(promise, (error: unknown) => {
    assert.equal(error instanceof Error, true);
    const message = error instanceof Error ? error.message : String(error);
    assert.equal(message, "Teacher entitlement directory verification failed");
    assert.doesNotMatch(
      message,
      /directory_|tenant-private|teacher-private|identity\.organization|entitlements\.organization/
    );
    return true;
  });
}

test("HTTPS POST keeps raw identity and Bearer secret in the private request only", async () => {
  let capturedUrl: string | URL | Request | undefined;
  let capturedInit: RequestInit | undefined;
  const fetchImplementation: typeof fetch = async (url, init) => {
    capturedUrl = url;
    capturedInit = init;
    return jsonResponse(wireSnapshot());
  };
  const value = await provider(fetchImplementation).readAuthoritativeSnapshot(IDENTITY);
  assert.equal(capturedUrl, ENDPOINT);
  assert.equal(capturedInit?.method, "POST");
  assert.equal(capturedInit?.redirect, "error");
  assert.equal(capturedInit?.credentials, "omit");
  assert.equal(capturedInit?.referrerPolicy, "no-referrer");
  assert.equal(typeof capturedInit?.signal, "object");
  assert.deepEqual(capturedInit?.headers, {
    accept: "application/json",
    authorization: `Bearer ${SECRET}`,
    "cache-control": "no-store",
    "content-type": "application/json"
  });
  assert.deepEqual(JSON.parse(String(capturedInit?.body)), {
    schema: TEACHER_ENTITLEMENT_LOOKUP_SCHEMA,
    identity: {
      issuer: IDENTITY.issuer,
      tenant_id: IDENTITY.tenantId,
      subject: IDENTITY.subject
    }
  });
  assert.deepEqual(value, {
    identity: IDENTITY,
    status: "active",
    roles: ["curriculum-editor", "teacher"],
    policyVersion: "roles-v7",
    revision: 42,
    evaluatedAt: new Date(NOW_MS).toISOString(),
    expiresAt: new Date(NOW_MS + 30_000).toISOString()
  });
  assert.equal(Object.isFrozen(value), true);
  assert.equal(Object.isFrozen(value.identity), true);
  assert.equal(Object.isFrozen(value.roles), true);
});

test("provider output produces a signed public receipt with neither raw identity nor Bearer secret", async () => {
  const directory = provider(async () => jsonResponse(wireSnapshot()));
  const authorization = new TeacherEntitlementAuthorizationService(
    directory,
    {
      policyId: "teacher-mutations",
      version: "roles-v7",
      requiredRoles: {
        "api/resource/review": ["teacher"],
        "api/curriculum/review": ["teacher"],
        "api/curriculum/seal": ["teacher"],
        "api/curriculum/revoke": ["teacher"],
        "api/adjudication/claim": ["teacher"],
        "api/adjudication/decide": ["teacher"],
        "api/safeguarding/list": ["safeguarding"],
        "api/safeguarding/dispatch": ["safeguarding"],
        "api/safeguarding/case/acknowledge": ["safeguarding"],
        "api/safeguarding/case/close": ["safeguarding"],
        "api/safeguarding/escalation/overdue": ["safeguarding"],
        "api/safeguarding/escalation/acknowledge": ["safeguarding"]
      },
      freshnessTtlMs: 1_000,
      cacheTtlMs: 0,
      providerTimeoutMs: 200,
      minAssuranceLevel: 2
    },
    Buffer.alloc(32, 0x62),
    Buffer.alloc(32, 0x72),
    CLOCK
  );
  const result = await authorization.authorize({
    provider: "oidc",
    identityNamespace: "oidc-issuer-tenant-sub-v1",
    identityIssuer: IDENTITY.issuer,
    tenantId: IDENTITY.tenantId,
    subject: IDENTITY.subject,
    assuranceLevel: 2
  }, "api/adjudication/decide");
  assert.equal(result.allowed, true);
  assert.doesNotMatch(
    JSON.stringify(result),
    new RegExp(`${SECRET}|tenant-private|teacher-private|identity\\.organization|curriculum-editor`)
  );
});

test("non-HTTPS, credentials, query, fragment and non-canonical endpoints are rejected before fetch", () => {
  const invalid = [
    "http://entitlements.organization.example/v1/snapshot",
    "https://user:pass@entitlements.organization.example/v1/snapshot",
    "https://entitlements.organization.example/v1/snapshot?tenant=x",
    "https://entitlements.organization.example/v1/snapshot#fragment",
    "https://entitlements.organization.example:443/v1/snapshot"
  ];
  for (const endpoint of invalid) {
    assert.throws(
      () => provider(async () => jsonResponse(wireSnapshot()), {endpoint}),
      /^Error: Teacher entitlement HTTP provider configuration is invalid$/
    );
  }
});

test("redirect and every non-2xx status fail closed without consuming their body", async () => {
  for (const status of [301, 400, 401, 403, 404, 429, 500, 503]) {
    await safeFailure(provider(async (_url, init) => {
      assert.equal(init?.redirect, "error");
      return jsonResponse(`${SECRET} ${IDENTITY.subject}`, {status});
    }).readAuthoritativeSnapshot(IDENTITY));
  }
  const followed = jsonResponse(wireSnapshot());
  Object.defineProperty(followed, "redirected", {value: true});
  await safeFailure(provider(async () => followed).readAuthoritativeSnapshot(IDENTITY));
});

test("provider timeout and caller abort both fail closed through the injected AbortSignal", async () => {
  let observedTimeoutAbort = false;
  const waitForAbort: typeof fetch = async (_url, init) => new Promise<Response>((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => {
      observedTimeoutAbort = true;
      reject(new Error(`${SECRET} timeout internals`));
    }, {once: true});
  });
  await safeFailure(provider(waitForAbort, {timeoutMs: 5}).readAuthoritativeSnapshot(IDENTITY));
  assert.equal(observedTimeoutAbort, true);

  let observedCallerAbort = false;
  const caller = new AbortController();
  const abortedFetch: typeof fetch = async (_url, init) => new Promise<Response>((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => {
      observedCallerAbort = true;
      reject(new Error("caller abort detail"));
    }, {once: true});
  });
  const attempt = provider(abortedFetch).readAuthoritativeSnapshot(IDENTITY, caller.signal);
  caller.abort();
  await safeFailure(attempt);
  assert.equal(observedCallerAbort, true);
});

test("declared and streamed responses cannot exceed the configured byte ceiling", async () => {
  await safeFailure(provider(async () => jsonResponse(wireSnapshot(), {
    headers: {"content-length": "9000"}
  })).readAuthoritativeSnapshot(IDENTITY));

  const oversized = `${JSON.stringify(wireSnapshot())}${"x".repeat(9_000)}`;
  await safeFailure(provider(async () => jsonResponse(oversized)).readAuthoritativeSnapshot(IDENTITY));
  await safeFailure(provider(async () => jsonResponse(wireSnapshot(), {
    headers: {"content-length": "8e3"}
  })).readAuthoritativeSnapshot(IDENTITY));
});

test("content type, JSON syntax, exact top-level schema and exact identity schema are enforced", async () => {
  const cases: Array<() => Response> = [
    () => new Response(JSON.stringify(wireSnapshot()), {
      status: 200,
      headers: {"content-type": "text/plain"}
    }),
    () => jsonResponse("{"),
    () => jsonResponse({...wireSnapshot(), unexpected: true}),
    () => jsonResponse(({schema: TEACHER_ENTITLEMENT_SNAPSHOT_SCHEMA})),
    () => jsonResponse(wireSnapshot({
      identity: {...wireSnapshot().identity, unexpected: true}
    })),
    () => jsonResponse(wireSnapshot({schema: "teachlab.teacher_entitlement_snapshot.v0"}))
  ];
  for (const createResponse of cases) {
    await safeFailure(provider(async () => createResponse()).readAuthoritativeSnapshot(IDENTITY));
  }
});

test("response identity must exactly bind issuer, tenant and subject", async () => {
  for (const identity of [
    {...wireSnapshot().identity, issuer: "https://other-issuer.example"},
    {...wireSnapshot().identity, tenant_id: "tenant-private-2"},
    {...wireSnapshot().identity, subject: "teacher-private-2"}
  ]) {
    await safeFailure(provider(async () => jsonResponse(wireSnapshot({identity})))
      .readAuthoritativeSnapshot(IDENTITY));
  }
});

test("status, roles, policy version, revision and timestamps have exact bounded semantics", async () => {
  const malformed: Record<string, unknown>[] = [
    {status: "disabled", roles: []},
    {status: "revoked", roles: ["teacher"]},
    {roles: "teacher"},
    {roles: ["teacher", "bad role"]},
    {roles: ["teacher", "teacher"]},
    {roles: Array.from({length: 33}, (_, index) => `role-${index}`)},
    {policy_version: "bad policy version"},
    {revision: -1},
    {revision: 1.5},
    {evaluated_at: "2026-08-12T04:00:00Z"},
    {evaluated_at: new Date(NOW_MS + 1_001).toISOString()},
    {expires_at: new Date(NOW_MS).toISOString()}
  ];
  for (const fields of malformed) {
    await safeFailure(provider(async () => jsonResponse(wireSnapshot(fields)))
      .readAuthoritativeSnapshot(IDENTITY));
  }
});

test("invalid lookup identity and invalid clock fail before private network I/O", async () => {
  let calls = 0;
  const fetchImplementation: typeof fetch = async () => {
    calls += 1;
    return jsonResponse(wireSnapshot());
  };
  await safeFailure(provider(fetchImplementation).readAuthoritativeSnapshot({
    ...IDENTITY,
    subject: "bad subject"
  }));
  await safeFailure(provider(fetchImplementation, {
    clock: {now: () => new Date(Number.NaN)}
  }).readAuthoritativeSnapshot(IDENTITY));
  assert.equal(calls, 0);
});
