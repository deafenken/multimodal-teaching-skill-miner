import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import test from "node:test";

import {
  OidcAuthorizationFlow,
  OidcAuthorizationFlowError,
  OidcTransactionCodec,
  type OidcAuthorizationFlowConfig
} from "../src/auth/oidc-authorization-flow";

const NOW = new Date("2026-08-12T03:00:00.000Z");
const config: OidcAuthorizationFlowConfig = {
  issuer: "https://identity.example.test",
  clientId: "teachlab-console",
  clientSecret: "client-secret-value",
  redirectUri: "https://console.example.test/api/teacher-agent/security/login/callback",
  transactionSecret: "transaction-secret-with-at-least-32-characters",
  discoveryTimeoutMs: 500,
  transactionTtlSeconds: 300,
  authenticationMaxAgeSeconds: 3600,
  accountStepUpMaxAgeSeconds: 300,
  accountAal2AcrValues: ["urn:teachlab:aal2"]
};

const verifiedIdentity = {
  subject: "alice",
  tenantId: "school-a",
  roles: ["teacher"],
  identityNamespace: "oidc-issuer-tenant-sub-v1" as const,
  identityIssuer: config.issuer,
  authenticatedAt: NOW.toISOString(),
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

const metadata = {
  issuer: config.issuer,
  authorization_endpoint: "https://identity.example.test/authorize",
  token_endpoint: "https://identity.example.test/token",
  code_challenge_methods_supported: ["S256"],
  token_endpoint_auth_methods_supported: ["client_secret_basic"]
};

test("authorization flow binds random state, nonce, S256 PKCE, and server-only token exchange", async () => {
  const calls: Array<{url: string; init?: RequestInit}> = [];
  let verifiedToken = "";
  let verifiedNonce = "";
  let verifiedAt: Date | undefined;
  const flow = new OidcAuthorizationFlow(config, {
    async verifyIdToken(token, nonce, now) {
      verifiedToken = token;
      verifiedNonce = nonce;
      verifiedAt = now;
      return verifiedIdentity;
    }
  }, async (input, init) => {
    calls.push({url: String(input), init});
    if (String(input).endsWith("openid-configuration")) return Response.json(metadata);
    return Response.json({id_token: "header.payload.signature", token_type: "Bearer"});
  }, () => NOW);

  const first = await flow.begin("/projects");
  const second = await flow.begin("/projects");
  assert.notEqual(first.transactionCookie, second.transactionCookie);
  const authorization = new URL(first.authorizationUrl);
  assert.equal(authorization.origin, "https://identity.example.test");
  assert.equal(authorization.searchParams.get("code_challenge_method"), "S256");
  assert.equal(authorization.searchParams.get("max_age"), "3600");
  const transaction = new OidcTransactionCodec(config.transactionSecret)
    .open(first.transactionCookie, NOW);
  assert.equal(authorization.searchParams.get("state"), transaction.state);
  assert.equal(authorization.searchParams.get("nonce"), transaction.nonce);
  assert.equal(
    authorization.searchParams.get("code_challenge"),
    createHash("sha256").update(transaction.codeVerifier, "ascii").digest("base64url")
  );

  const result = await flow.complete({
    code: "authorization-code-123",
    state: transaction.state,
    transactionCookie: first.transactionCookie
  });
  assert.deepEqual(result, {
    identity: verifiedIdentity,
    returnTo: "/projects",
    purpose: "login"
  });
  assert.equal(verifiedToken, "header.payload.signature");
  assert.equal(verifiedNonce, transaction.nonce);
  assert.equal(verifiedAt?.toISOString(), NOW.toISOString());
  const exchange = calls.at(-1);
  assert.equal(exchange?.url, metadata.token_endpoint);
  const headers = new Headers(exchange?.init?.headers);
  assert.match(headers.get("authorization") ?? "", /^Basic [A-Za-z0-9+/]+=*$/);
  const body = new URLSearchParams(String(exchange?.init?.body));
  assert.equal(body.get("code_verifier"), transaction.codeVerifier);
  assert.equal(JSON.stringify(result).includes("header.payload.signature"), false);
});

test("transaction rejects fixation, tampering, expiry, and open redirects", async () => {
  const flow = new OidcAuthorizationFlow(config, {
    async verifyIdToken() { throw new Error("must not run"); }
  }, async () => Response.json(metadata), () => NOW);
  await assert.rejects(flow.begin("//evil.example.test"), (error: unknown) =>
    error instanceof OidcAuthorizationFlowError && error.safeCode === "callback_invalid");
  await assert.rejects(flow.begin("/api/v1/secrets"), (error: unknown) =>
    error instanceof OidcAuthorizationFlowError && error.safeCode === "callback_invalid");
  const started = await flow.begin("/");
  const transaction = new OidcTransactionCodec(config.transactionSecret)
    .open(started.transactionCookie, NOW);
  await assert.rejects(flow.complete({
    code: "authorization-code-123",
    state: "A".repeat(43),
    transactionCookie: started.transactionCookie
  }), (error: unknown) => error instanceof OidcAuthorizationFlowError && error.safeCode === "callback_invalid");
  assert.throws(
    () => new OidcTransactionCodec(config.transactionSecret).open(
      `${started.transactionCookie.slice(0, -1)}${started.transactionCookie.endsWith("A") ? "B" : "A"}`,
      NOW
    ),
    /could not be authenticated/
  );
  assert.throws(
    () => new OidcTransactionCodec(config.transactionSecret).open(
      started.transactionCookie,
      new Date(NOW.getTime() + 301_000)
    ),
    /invalid or expired/
  );
  assert.notEqual(transaction.state, "A".repeat(43));
});

test("discovery is exact and token responses are bounded and timed out", async () => {
  const wrongIssuerFlow = new OidcAuthorizationFlow(config, {
    async verifyIdToken() { throw new Error("unused"); }
  }, async () => Response.json({...metadata, issuer: `${config.issuer}/`}), () => NOW);
  await assert.rejects(wrongIssuerFlow.begin("/"), (error: unknown) =>
    error instanceof OidcAuthorizationFlowError && error.safeCode === "metadata_unavailable");

  const overlimit = new OidcAuthorizationFlow(config, {
    async verifyIdToken() { throw new Error("unused"); }
  }, async (input) => String(input).endsWith("openid-configuration")
    ? Response.json(metadata)
    : new Response("x".repeat(65_537)), () => NOW);
  const started = await overlimit.begin("/");
  const transaction = new OidcTransactionCodec(config.transactionSecret).open(started.transactionCookie, NOW);
  await assert.rejects(overlimit.complete({
    code: "authorization-code-123",
    state: transaction.state,
    transactionCookie: started.transactionCookie
  }), (error: unknown) => error instanceof OidcAuthorizationFlowError && error.safeCode === "token_exchange_failed");

  const timeout = new OidcAuthorizationFlow(config, {
    async verifyIdToken() { throw new Error("unused"); }
  }, async (_input, init) => await new Promise<Response>((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => reject(new Error("aborted")), {once: true});
  }), () => NOW);
  await assert.rejects(timeout.begin("/"), (error: unknown) =>
    error instanceof OidcAuthorizationFlowError && error.safeCode === "metadata_unavailable");

  let requests = 0;
  const tokenTimeout = new OidcAuthorizationFlow(config, {
    async verifyIdToken() { throw new Error("unused"); }
  }, async (_input, init) => {
    requests += 1;
    if (requests === 1) return Response.json(metadata);
    return await new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new Error("aborted")), {once: true});
    });
  }, () => NOW);
  const tokenTimeoutStart = await tokenTimeout.begin("/");
  const tokenTimeoutTransaction = new OidcTransactionCodec(config.transactionSecret)
    .open(tokenTimeoutStart.transactionCookie, NOW);
  await assert.rejects(tokenTimeout.complete({
    code: "authorization-code-789",
    state: tokenTimeoutTransaction.state,
    transactionCookie: tokenTimeoutStart.transactionCookie
  }), (error: unknown) =>
    error instanceof OidcAuthorizationFlowError && error.safeCode === "token_exchange_failed");
});
