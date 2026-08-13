import assert from "node:assert/strict";
import {test} from "node:test";

import {InMemorySessionRevocationRepository} from "../src/auth/in-memory-session-revocation.repository";
import {SessionCookieAuthProvider} from "../src/auth/session-cookie-auth.provider";
import {SessionCookieService} from "../src/auth/session-cookie";
import {SessionRevocationService, sessionIdSha256} from "../src/auth/session-revocation.service";
import type {SessionRevocationRepositoryPort} from "../src/auth/session-revocation.repository.port";
import type {AppConfigService} from "../src/config/app-config.service";

const config = {
  sessionTtlSeconds: 8 * 60 * 60,
  sessionSecret: "active-session-secret-with-at-least-32-characters",
  sessionVerificationSecrets: [
    "active-session-secret-with-at-least-32-characters",
    "previous-session-secret-with-at-least-32-characters"
  ],
  sessionCookieName: "teachlab_session",
  csrfCookieName: "teachlab_csrf",
  secureSessionCookies: false
  ,accountIdentityNamespaceSecret:
    "test-account-identity-namespace-secret-at-least-32-characters"
  ,accountIdentityNamespaceVersion: "ns1"
} as unknown as AppConfigService;

function oidcIdentity(now = new Date()) {
  return {
    subject: "alice",
    tenantId: "tenant-a",
    provider: "oidc" as const,
    identityNamespace: "oidc-issuer-tenant-sub-v1" as const,
    identityIssuer: "https://identity.example.test",
    authenticatedAt: now.toISOString(),
    assuranceLevel: 2,
    remoteSubjectPolicy: {
      policy_id: "organization-policy",
      policy_version: "v1",
      policy_source: "organization_oidc_or_roster_policy" as const,
      likely_minor: false,
      guardian_or_school_policy: "not_required" as const,
      remote_processing_eligible: true
    }
  };
}

function request(cookieValue: string) {
  return {headers: {cookie: `teachlab_session=${cookieValue}`}, method: "GET"};
}

test("memory authority isolates users and makes revoke idempotent", async () => {
  const repository = new InMemorySessionRevocationRepository();
  const now = new Date("2026-08-12T00:00:00.000Z");
  const expiresAt = new Date("2026-08-12T08:00:00.000Z");
  const digest = "a".repeat(64);
  await repository.register({
    tenantId: "tenant-a",
    ownerId: "alice",
    sessionIdSha256: digest,
    issuedAt: now,
    expiresAt
  });
  assert.equal(
    (await repository.inspect({tenantId: "tenant-a", ownerId: "alice"}, digest, now)).kind,
    "active"
  );
  assert.equal(
    (await repository.inspect({tenantId: "tenant-a", ownerId: "bob"}, digest, now)).kind,
    "missing"
  );
  const input = {
    tenantId: "tenant-a",
    ownerId: "alice",
    sessionIdSha256: digest,
    revokedAt: new Date("2026-08-12T01:00:00.000Z"),
    reason: "user_logout" as const
  };
  assert.deepEqual(
    (await Promise.all([repository.revoke(input), repository.revoke(input)]))
      .map((result) => result.kind)
      .sort(),
    ["already_revoked", "revoked"]
  );
  assert.equal(
    (await repository.inspect({tenantId: "tenant-a", ownerId: "alice"}, digest, now)).kind,
    "revoked"
  );
});

test("expiry cleanup is bounded and expired records fail closed", async () => {
  const repository = new InMemorySessionRevocationRepository();
  const issuedAt = new Date("2026-08-12T00:00:00.000Z");
  const expiresAt = new Date("2026-08-12T01:00:00.000Z");
  for (let index = 0; index < 3; index += 1) {
    await repository.register({
      tenantId: "tenant-a",
      ownerId: "alice",
      sessionIdSha256: index.toString(16).repeat(64),
      issuedAt,
      expiresAt
    });
  }
  const later = new Date("2026-08-12T02:00:00.000Z");
  assert.equal(await repository.cleanupExpired({tenantId: "tenant-a", ownerId: "alice"}, later, 2), 2);
  assert.equal(
    (await repository.inspect(
      {tenantId: "tenant-a", ownerId: "alice"},
      "2".repeat(64),
      later
    )).kind,
    "missing"
  );
});

test("previous-key cookie remains revocable after signing-key rotation", async () => {
  const previousConfig = {
    ...config,
    sessionSecret: "previous-session-secret-with-at-least-32-characters",
    sessionVerificationSecrets: ["previous-session-secret-with-at-least-32-characters"]
  } as unknown as AppConfigService;
  const minted = new SessionCookieService(previousConfig).mint(
    oidcIdentity(),
    new Date()
  );
  const repository = new InMemorySessionRevocationRepository();
  const revocations = new SessionRevocationService(repository);
  await revocations.register(minted.authenticatedSession);
  const provider = new SessionCookieAuthProvider(new SessionCookieService(config), revocations);
  assert.equal((await provider.authenticate(request(minted.cookieValue)))?.principal.subject, "alice");
  await revocations.revokeCurrent(minted.principal);
  assert.equal(await provider.authenticate(request(minted.cookieValue)), null);
  assert.equal(
    (await provider.authenticate(request(minted.cookieValue), {allowRevokedSession: true}))
      ?.principal.subject,
    "alice"
  );
});

test("restart without durable authority rejects an otherwise valid old cookie", async () => {
  const cookies = new SessionCookieService(config);
  const minted = cookies.mint(
    {subject: "alice", tenantId: "tenant-a", provider: "development"},
    new Date()
  );
  const emptyAfterRestart = new SessionRevocationService(
    new InMemorySessionRevocationRepository()
  );
  const provider = new SessionCookieAuthProvider(cookies, emptyAfterRestart);
  assert.equal(await provider.authenticate(request(minted.cookieValue)), null);
});

test("authority-store failure fails authentication closed", async () => {
  const cookies = new SessionCookieService(config);
  const minted = cookies.mint(
    {subject: "alice", tenantId: "tenant-a", provider: "development"},
    new Date()
  );
  const failed = new Proxy({} as SessionRevocationRepositoryPort, {
    get() {
      return async () => {
        throw new Error("database unavailable");
      };
    }
  });
  const provider = new SessionCookieAuthProvider(cookies, new SessionRevocationService(failed));
  await assert.rejects(
    provider.authenticate(request(minted.cookieValue)),
    /Session revocation authority is unavailable/
  );
  assert.equal(sessionIdSha256(minted.principal.sessionId).length, 64);
});

test("authority records must exactly bind signed scope and lifetime", async () => {
  const minted = new SessionCookieService(config).mint(
    oidcIdentity(),
    new Date()
  );
  const mismatched = {
    tenantId: "tenant-a",
    ownerId: "alice",
    sessionIdSha256: sessionIdSha256(minted.principal.sessionId),
    issuedAt: new Date(minted.issuedAt),
    expiresAt: new Date(Date.parse(minted.expiresAt) + 1_000),
    revokedAt: null,
    revocationReason: null,
    version: 1
  };
  const repository = {
    async register() { return mismatched; },
    async inspect() { return {kind: "active" as const, record: mismatched}; },
    async inspectAuthoritatively() {
      return {kind: "active" as const, record: mismatched, databaseNow: new Date()};
    },
    async revoke() { return {kind: "missing" as const}; },
    async cleanupExpired() { return 0; }
  } satisfies SessionRevocationRepositoryPort;
  const service = new SessionRevocationService(repository);
  await assert.rejects(
    service.register(minted.authenticatedSession),
    /Session revocation authority is unavailable/
  );
  await assert.rejects(
    service.inspect(minted.authenticatedSession),
    /Session revocation authority is unavailable/
  );
});
