import {
  createSign,
  generateKeyPairSync,
  type JsonWebKey,
  type KeyObject
} from "node:crypto";
import assert from "node:assert/strict";
import {after, before, test} from "node:test";

import {AppConfigService} from "../src/config/app-config.service";
import {OidcIdentityVerifier} from "../src/auth/oidc-identity.verifier";

const originalFetch = globalThis.fetch;
const previousEnvironment = new Map<string, string | undefined>();
const environment = {
  NODE_ENV: "test",
  AUTH_MODE: "oidc",
  DATA_BACKEND: "memory",
  CORS_ORIGINS: "https://console.example.test",
  OIDC_ISSUER: "https://identity.example.test",
  OIDC_AUDIENCE: "teachlab-api",
  OIDC_CLIENT_ID: "teachlab-console",
  OIDC_CLIENT_SECRET: "oidc-test-client-secret",
  OIDC_REDIRECT_URI: "https://console.example.test/api/teacher-agent/security/login/callback",
  OIDC_TRANSACTION_SECRET: "oidc-test-transaction-secret-at-least-32-characters",
  OIDC_EXPECTED_HOST: "api.example.test",
  OIDC_TENANT_CLAIM: "org_id",
  OIDC_ROLES_CLAIM: "roles",
  OIDC_ALLOWED_ALGORITHMS: "RS256",
  SESSION_SECRET: "oidc-test-session-secret-at-least-32-characters",
  SESSION_COOKIE_SECURE: "true",
  SEED_DEMO_SESSIONS: "false"
};

let privateKey: KeyObject;
let publicJwk: JsonWebKey;
let verifier: OidcIdentityVerifier;

function encode(value: unknown): string {
  return Buffer.from(JSON.stringify(value), "utf8").toString("base64url");
}

function token(claims: Record<string, unknown>): string {
  const header = encode({alg: "RS256", kid: "test-key", typ: "JWT"});
  const payload = encode(claims);
  const signer = createSign("RSA-SHA256");
  signer.update(`${header}.${payload}`, "utf8");
  signer.end();
  return `${header}.${payload}.${signer.sign(privateKey).toString("base64url")}`;
}

function validClaims(): Record<string, unknown> {
  const now = Math.floor(Date.now() / 1000);
  return {
    iss: "https://identity.example.test",
    aud: "teachlab-api",
    sub: "oidc-user",
    org_id: "tenant-a",
    email: "learner@example.test",
    roles: ["learner", "teacher"],
    iat: now,
    auth_time: now - 5,
    acr: "urn:teachlab:aal2",
    exp: now + 300
  };
}

before(() => {
  for (const [key, value] of Object.entries(environment)) {
    previousEnvironment.set(key, process.env[key]);
    process.env[key] = value;
  }
  const pair = generateKeyPairSync("rsa", {modulusLength: 2048});
  privateKey = pair.privateKey;
  publicJwk = {...(pair.publicKey.export({format: "jwk"}) as JsonWebKey), kid: "test-key", use: "sig"};
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.endsWith("/.well-known/openid-configuration")) {
      return new Response(
        JSON.stringify({
          issuer: "https://identity.example.test",
          jwks_uri: "https://identity.example.test/.well-known/jwks.json"
        }),
        {status: 200, headers: {"content-type": "application/json"}}
      );
    }
    if (url.endsWith("/.well-known/jwks.json")) {
      return new Response(JSON.stringify({keys: [publicJwk]}), {
        status: 200,
        headers: {"content-type": "application/json"}
      });
    }
    return new Response("not found", {status: 404});
  };
  const config = new AppConfigService();
  config.assertSafeForStartup();
  verifier = new OidcIdentityVerifier(config);
});

after(() => {
  globalThis.fetch = originalFetch;
  for (const key of Object.keys(environment)) {
    const previous = previousEnvironment.get(key);
    if (previous === undefined) delete process.env[key];
    else process.env[key] = previous;
  }
});

test("OIDC verifier validates issuer, audience, signature, and trusted tenant claim", async () => {
  const claims = validClaims();
  const identity = await verifier.verify(token(claims));
  assert.deepEqual(identity, {
    subject: "oidc-user",
    tenantId: "tenant-a",
    email: "learner@example.test",
    roles: ["learner", "teacher"],
    identityNamespace: "oidc-issuer-tenant-sub-v1",
    identityIssuer: "https://identity.example.test",
    authenticatedAt: new Date((claims.auth_time as number) * 1000).toISOString(),
    assuranceLevel: 2,
    remoteSubjectPolicy: {
      policy_id: "teachlab-organization-remote-processing",
      policy_version: "v1",
      policy_source: "organization_oidc_or_roster_policy",
      likely_minor: true,
      guardian_or_school_policy: "not_required",
      remote_processing_eligible: false
    }
  });
});

test("OIDC verifier fails closed for a tampered token or missing tenant", async () => {
  const valid = token(validClaims());
  const parts = valid.split(".");
  const signature = parts[2] ?? "";
  const replacement = signature.startsWith("A") ? "B" : "A";
  const tampered = `${parts[0]}.${parts[1]}.${replacement}${signature.slice(1)}`;
  await assert.rejects(verifier.verify(tampered), /signature is invalid/);
  const claims = validClaims();
  delete claims.org_id;
  await assert.rejects(verifier.verify(token(claims)), /must contain sub and org_id/);
  await assert.rejects(
    verifier.verify(token({...validClaims(), sub: "a".repeat(129)})),
    /must contain sub and org_id/
  );
  await assert.rejects(
    verifier.verify(token({...validClaims(), org_id: "tenant\u0000other"})),
    /must contain sub and org_id/
  );
});

test("signed subject policy claims authorize only the exact configured version", async () => {
  const authorized = await verifier.verify(token({
    ...validClaims(),
    teachlab_remote_processing_policy: "minor_guardian_verified",
    teachlab_remote_processing_policy_version: "v1"
  }));
  assert.deepEqual(authorized.remoteSubjectPolicy, {
    policy_id: "teachlab-organization-remote-processing",
    policy_version: "v1",
    policy_source: "organization_oidc_or_roster_policy",
    likely_minor: true,
    guardian_or_school_policy: "verified_guardian",
    remote_processing_eligible: true
  });
  for (const claims of [
    {...validClaims(), teachlab_remote_processing_policy: "adult_roster_authorized"},
    {
      ...validClaims(),
      teachlab_remote_processing_policy: "adult_roster_authorized",
      teachlab_remote_processing_policy_version: "stale"
    },
    {
      ...validClaims(),
      teachlab_remote_processing_policy: "browser_claimed_adult",
      teachlab_remote_processing_policy_version: "v1"
    }
  ]) {
    assert.equal(
      (await verifier.verify(token(claims))).remoteSubjectPolicy.remote_processing_eligible,
      false
    );
  }
});

test("authorization-code ID tokens strictly bind nonce, client azp, iat, and auth_time", async () => {
  const now = Math.floor(Date.now() / 1000);
  const claims = {
    ...validClaims(),
    aud: "teachlab-console",
    azp: "teachlab-console",
    nonce: "n".repeat(43),
    iat: now,
    auth_time: now - 5
  };
  assert.equal(
    (await verifier.verifyIdToken(token(claims), "n".repeat(43))).subject,
    "oidc-user"
  );
  await assert.rejects(
    verifier.verifyIdToken(token({...claims, nonce: "x".repeat(43)}), "n".repeat(43)),
    /nonce does not match/
  );
  await assert.rejects(
    verifier.verifyIdToken(token({...claims, nonce: ` ${"n".repeat(43)} `}), "n".repeat(43)),
    /nonce does not match/
  );
  await assert.rejects(
    verifier.verifyIdToken(token({...claims, azp: "other-client"}), "n".repeat(43)),
    /authorized party/
  );
  await assert.rejects(
    verifier.verifyIdToken(token({...claims, aud: ["teachlab-console", 7]}), "n".repeat(43)),
    /audience does not match/
  );
  await assert.rejects(
    verifier.verifyIdToken(token({...claims, nbf: "not-a-numeric-date"}), "n".repeat(43)),
    /not active yet/
  );
  await assert.rejects(
    verifier.verifyIdToken(token({...claims, iat: now - 4000}), "n".repeat(43)),
    /issued-at time/
  );
  await assert.rejects(
    verifier.verifyIdToken(token({...claims, auth_time: now - 4000}), "n".repeat(43)),
    /authentication time/
  );
});
