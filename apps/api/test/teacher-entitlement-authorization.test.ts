import assert from "node:assert/strict";
import test from "node:test";

import type {AuthenticatedPrincipal} from "../src/auth/auth-provider.port";
import {
  TeacherEntitlementAuthorizationService,
  teacherEntitlementReceiptSha256
} from "../src/auth/teacher-entitlement-authorization";
import type {
  AuthoritativeTeacherEntitlementSnapshot,
  CanonicalEntitlementIdentity,
  TeacherEntitlementClock,
  TeacherEntitlementPolicy,
  TeacherEntitlementSnapshotProvider
} from "../src/auth/teacher-entitlement.port";
import {PRIVILEGED_TEACHER_OPERATIONS} from "../src/auth/teacher-entitlement.port";

const BASE_MS = Date.parse("2026-08-12T04:00:00.000Z");
const IDENTITY: CanonicalEntitlementIdentity = {
  issuer: "https://identity.entitlement.example.test",
  tenantId: "tenant-authority-a",
  subject: "teacher-authority-a"
};
const BINDING_KEY = Buffer.alloc(32, 0x62);
const RECEIPT_KEY = Buffer.alloc(32, 0x72);

class FakeClock implements TeacherEntitlementClock {
  constructor(public value = BASE_MS) {}

  now(): Date {
    return new Date(this.value);
  }
}

class MutableProvider implements TeacherEntitlementSnapshotProvider {
  calls = 0;
  value: AuthoritativeTeacherEntitlementSnapshot | null;
  failure?: Error;

  constructor(value = snapshot()) {
    this.value = value;
  }

  async readAuthoritativeSnapshot() {
    this.calls += 1;
    if (this.failure) throw this.failure;
    return this.value;
  }
}

function policy(
  overrides: Partial<TeacherEntitlementPolicy> = {}
): TeacherEntitlementPolicy {
  return {
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
    cacheTtlMs: 500,
    providerTimeoutMs: 100,
    minAssuranceLevel: 2,
    ...overrides
  };
}

function snapshot(
  overrides: Partial<AuthoritativeTeacherEntitlementSnapshot> = {}
): AuthoritativeTeacherEntitlementSnapshot {
  return {
    identity: {...IDENTITY},
    status: "active",
    roles: ["teacher", "safeguarding"],
    policyVersion: "roles-v7",
    revision: 41,
    evaluatedAt: new Date(BASE_MS).toISOString(),
    expiresAt: new Date(BASE_MS + 60_000).toISOString(),
    ...overrides
  };
}

function oldCookiePrincipal(
  overrides: Partial<AuthenticatedPrincipal> = {}
): AuthenticatedPrincipal {
  return {
    provider: "oidc",
    identityNamespace: "oidc-issuer-tenant-sub-v1",
    identityIssuer: IDENTITY.issuer,
    tenantId: IDENTITY.tenantId,
    subject: IDENTITY.subject,
    sessionId: "old-cookie-session",
    assuranceLevel: 2,
    authenticatedAt: "2026-08-12T03:55:00.000Z",
    scopeTenantId: "opaque-tenant",
    scopeOwnerId: "opaque-owner",
    // This deliberately stale cookie role is not an authorization input.
    roles: ["teacher", "platform-admin"],
    email: "raw-teacher@entitlement.example.test",
    ...overrides
  };
}

function service(
  provider: TeacherEntitlementSnapshotProvider,
  clock = new FakeClock(),
  policyOverrides: Partial<TeacherEntitlementPolicy> = {}
) {
  return new TeacherEntitlementAuthorizationService(
    provider,
    policy(policyOverrides),
    BINDING_KEY,
    RECEIPT_KEY,
    clock
  );
}

test("authoritative snapshot authorizes every exact privileged mutation and receipt is identity-free", async () => {
  const provider = new MutableProvider();
  const authorization = service(provider);
  for (const operation of PRIVILEGED_TEACHER_OPERATIONS) {
    const result = await authorization.authorize(oldCookiePrincipal(), operation);
    assert.equal(result.allowed, true);
    if (!result.allowed) continue;
    assert.equal(result.receipt.operation, operation);
    assert.equal(result.receipt.policy_id, "teacher-mutations");
    assert.equal(result.receipt.policy_version, "roles-v7");
    assert.equal(result.receipt.entitlement_revision, 41);
    assert.match(result.receipt.principal_binding, /^epb1_[A-Za-z0-9_-]{43}$/);
    assert.match(result.receipt.entitlement_binding, /^esb1_[A-Za-z0-9_-]{43}$/);
    assert.match(result.receipt.signature, /^[A-Za-z0-9_-]{43}$/);
    assert.match(teacherEntitlementReceiptSha256(result.receipt), /^[0-9a-f]{64}$/);
    assert.equal(Object.isFrozen(result.receipt), true);

    const serialized = JSON.stringify(result.receipt);
    assert.doesNotMatch(
      serialized,
      /identity\.entitlement|tenant-authority|teacher-authority|raw-teacher|old-cookie|platform-admin/
    );
    assert.deepEqual(Object.keys(result.receipt).sort(), [
      "authorized_at",
      "decision",
      "entitlement_binding",
      "entitlement_revision",
      "fresh_until",
      "operation",
      "policy_id",
      "policy_version",
      "principal_binding",
      "schema",
      "signature"
    ]);
  }
  assert.equal(provider.calls, 1, "all operations share the same fresh snapshot cache");
});

test("revoking the server entitlement blocks an old role-bearing cookie on review, claim and decide", async () => {
  const provider = new MutableProvider();
  const authorization = service(provider);
  const staleCookie = oldCookiePrincipal();

  assert.equal(
    (await authorization.authorize(staleCookie, "api/resource/review")).allowed,
    true
  );
  provider.value = snapshot({
    status: "revoked",
    roles: [],
    revision: 42
  });
  // Production wiring invokes this from the authoritative role-change event.
  authorization.invalidateIdentity(staleCookie);

  let protectedSideEffects = 0;
  for (const operation of PRIVILEGED_TEACHER_OPERATIONS) {
    const decision = await authorization.authorize(staleCookie, operation);
    if (decision.allowed) protectedSideEffects += 1;
    assert.deepEqual(decision, {allowed: false, reason: "entitlement_revoked"});
  }
  assert.equal(protectedSideEffects, 0);
  assert.equal(provider.calls, 2);
});

test("cookie roles are never an authority source", async () => {
  const provider = new MutableProvider(snapshot({roles: ["learner"]}));
  const result = await service(provider).authorize(
    oldCookiePrincipal({roles: ["teacher", "superuser"]}),
    "api/adjudication/decide"
  );
  assert.deepEqual(result, {allowed: false, reason: "role_missing"});

  const invalidPrincipal = oldCookiePrincipal({
    provider: "development",
    identityNamespace: undefined,
    identityIssuer: undefined,
    assuranceLevel: undefined
  });
  assert.deepEqual(
    await service(provider).authorize(invalidPrincipal, "api/resource/review"),
    {allowed: false, reason: "invalid_principal"}
  );
  assert.equal(provider.calls, 1, "invalid sessions fail before provider I/O");
});

test("issuer, tenant and subject must all exactly bind the authoritative snapshot", async () => {
  for (const [field, value] of [
    ["issuer", "https://other-issuer.example.test"],
    ["tenantId", "tenant-authority-b"],
    ["subject", "teacher-authority-b"]
  ] as const) {
    const provider = new MutableProvider(snapshot({
      identity: {...IDENTITY, [field]: value}
    }));
    assert.deepEqual(
      await service(provider).authorize(oldCookiePrincipal(), "api/adjudication/claim"),
      {allowed: false, reason: "identity_mismatch"},
      field
    );
  }
});

test("policy mismatch, malformed snapshot and monotonic revision rollback fail closed", async () => {
  const mismatch = new MutableProvider(snapshot({policyVersion: "roles-v6"}));
  assert.deepEqual(
    await service(mismatch).authorize(oldCookiePrincipal(), "api/resource/review"),
    {allowed: false, reason: "policy_mismatch"}
  );

  const inconsistentRevocation = new MutableProvider(snapshot({
    status: "revoked",
    roles: ["teacher"]
  }));
  assert.deepEqual(
    await service(inconsistentRevocation).authorize(oldCookiePrincipal(), "api/resource/review"),
    {allowed: false, reason: "snapshot_invalid"}
  );

  const revisions = new MutableProvider(snapshot({revision: 42}));
  const authorization = service(revisions);
  assert.equal(
    (await authorization.authorize(oldCookiePrincipal(), "api/resource/review")).allowed,
    true
  );
  revisions.value = snapshot({revision: 41});
  authorization.invalidateIdentity(oldCookiePrincipal());
  assert.deepEqual(
    await authorization.authorize(oldCookiePrincipal(), "api/resource/review"),
    {allowed: false, reason: "snapshot_rollback"}
  );
});

test("concurrent and repeated authorization uses one bounded idempotent cache", async () => {
  const clock = new FakeClock();
  let calls = 0;
  let release: ((value: AuthoritativeTeacherEntitlementSnapshot) => void) | undefined;
  const pending = new Promise<AuthoritativeTeacherEntitlementSnapshot>((resolve) => {
    release = resolve;
  });
  const provider: TeacherEntitlementSnapshotProvider = {
    async readAuthoritativeSnapshot() {
      calls += 1;
      return pending;
    }
  };
  const authorization = service(provider, clock);
  const first = authorization.authorize(oldCookiePrincipal(), "api/resource/review");
  const second = authorization.authorize(oldCookiePrincipal(), "api/adjudication/claim");
  release?.(snapshot());
  assert.deepEqual((await Promise.all([first, second])).map((item) => item.allowed), [true, true]);
  assert.equal(calls, 1);

  clock.value += 499;
  assert.equal(
    (await authorization.authorize(oldCookiePrincipal(), "api/adjudication/decide")).allowed,
    true
  );
  assert.equal(calls, 1);
});

test("cache never survives its TTL and provider outage then fails closed", async () => {
  const clock = new FakeClock();
  const provider = new MutableProvider();
  const authorization = service(provider, clock);
  const first = await authorization.authorize(oldCookiePrincipal(), "api/resource/review");
  assert.equal(first.allowed, true);

  provider.failure = new Error("private directory unavailable with secret details");
  clock.value += 499;
  assert.equal(
    (await authorization.authorize(oldCookiePrincipal(), "api/resource/review")).allowed,
    true,
    "a still-valid authoritative snapshot remains usable within the declared cache window"
  );
  clock.value += 1;
  assert.deepEqual(
    await authorization.authorize(oldCookiePrincipal(), "api/resource/review"),
    {allowed: false, reason: "provider_unavailable"}
  );
  assert.equal(provider.calls, 2);
});

test("provider absence and outage fail closed without leaking provider errors", async () => {
  const missing = new MutableProvider();
  missing.value = null;
  assert.deepEqual(
    await service(missing).authorize(oldCookiePrincipal(), "api/adjudication/claim"),
    {allowed: false, reason: "snapshot_missing"}
  );

  const outage = new MutableProvider();
  outage.failure = new Error(`${IDENTITY.subject}: database password and stack`);
  for (const operation of PRIVILEGED_TEACHER_OPERATIONS) {
    const result = await service(outage).authorize(oldCookiePrincipal(), operation);
    assert.deepEqual(result, {allowed: false, reason: "provider_unavailable"});
    assert.doesNotMatch(JSON.stringify(result), /teacher-authority|password|stack/);
  }
});

test("a role-change invalidation racing an in-flight lookup discards the old snapshot", async () => {
  let release: ((value: AuthoritativeTeacherEntitlementSnapshot) => void) | undefined;
  const pending = new Promise<AuthoritativeTeacherEntitlementSnapshot>((resolve) => {
    release = resolve;
  });
  const provider: TeacherEntitlementSnapshotProvider = {
    async readAuthoritativeSnapshot() {
      return pending;
    }
  };
  const authorization = service(provider);
  const authorizationAttempt = authorization.authorize(
    oldCookiePrincipal(),
    "api/adjudication/decide"
  );
  authorization.invalidateIdentity(oldCookiePrincipal());
  release?.(snapshot());
  assert.deepEqual(
    await authorizationAttempt,
    {allowed: false, reason: "snapshot_stale"}
  );
});

test("provider expiry and freshness TTL cap receipt lifetime independently of cache TTL", async () => {
  const clock = new FakeClock();
  const provider = new MutableProvider(snapshot({
    expiresAt: new Date(BASE_MS + 700).toISOString()
  }));
  const authorization = service(provider, clock, {cacheTtlMs: 900});
  const result = await authorization.authorize(oldCookiePrincipal(), "api/resource/review");
  assert.equal(result.allowed, true);
  if (result.allowed) {
    assert.equal(result.receipt.fresh_until, new Date(BASE_MS + 700).toISOString());
  }

  clock.value += 700;
  assert.deepEqual(
    await authorization.authorize(oldCookiePrincipal(), "api/resource/review"),
    {allowed: false, reason: "snapshot_stale"}
  );
});
