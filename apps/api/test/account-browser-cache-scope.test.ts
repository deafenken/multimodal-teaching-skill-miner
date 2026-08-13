import assert from "node:assert/strict";
import {test} from "node:test";

import {AccountBrowserCacheScopeIssuer} from "../src/account/account-browser-cache-scope";
import type {AuthenticatedPrincipal} from "../src/auth/auth-provider.port";

const KEY = "browser-cache-scope-secret-000000000000000000000000000000000";

function principal(subject: string, tenantId = "tenant-a"): AuthenticatedPrincipal {
  return {
    subject,
    tenantId,
    sessionId: "00000000-0000-4000-8000-000000000001",
    provider: "oidc",
    roles: [],
    identityNamespace: "oidc-issuer-tenant-sub-v1",
    identityIssuer: "https://identity.example.test",
    authenticatedAt: new Date().toISOString(),
    assuranceLevel: 2
  };
}

test("browser cache scope is opaque, stable, issuer-bound and account-separated", () => {
  const issuer = new AccountBrowserCacheScopeIssuer(
    KEY,
    "https://identity.example.test",
    "account-cache-epoch-1"
  );
  const alice = issuer.issue(principal("alice"));
  assert.match(alice, /^acs1_[A-Za-z0-9_-]{43}$/);
  assert.equal(alice, issuer.issue({...principal("alice"), sessionId: "different-session"}));
  assert.notEqual(alice, issuer.issue(principal("bob")));
  assert.notEqual(alice, issuer.issue(principal("alice", "tenant-b")));
  const rotatedEpoch = new AccountBrowserCacheScopeIssuer(
    KEY,
    "https://identity.example.test",
    "account-cache-epoch-2"
  );
  assert.notEqual(alice, rotatedEpoch.issue(principal("alice")));
  assert.equal(alice.includes("alice"), false);
  assert.equal(alice.includes("tenant-a"), false);
});

test("local development identities cannot mint an account cache namespace", () => {
  const issuer = new AccountBrowserCacheScopeIssuer(
    KEY,
    "https://identity.example.test",
    "account-cache-epoch-1"
  );
  assert.throws(() => issuer.issue({...principal("alice"), provider: "development"}));
});
