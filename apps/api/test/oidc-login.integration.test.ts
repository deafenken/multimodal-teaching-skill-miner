import {createSign, generateKeyPairSync, type JsonWebKey} from "node:crypto";
import type {OutgoingHttpHeaders} from "node:http";
import assert from "node:assert/strict";
import {after, before, test} from "node:test";

import type {NestFastifyApplication} from "@nestjs/platform-fastify";

import {createApplication} from "../src/create-application";
import {SessionRevocationService} from "../src/auth/session-revocation.service";

const environment = {
  NODE_ENV: "test",
  API_HOST: "127.0.0.1",
  API_PORT: "4000",
  AUTH_MODE: "oidc",
  DATA_BACKEND: "memory",
  CORS_ORIGINS: "https://console.example.test",
  OIDC_ISSUER: "https://identity.example.test",
  OIDC_AUDIENCE: "teachlab-api",
  OIDC_CLIENT_ID: "teachlab-console",
  OIDC_CLIENT_SECRET: "integration-client-secret",
  OIDC_REDIRECT_URI: "https://console.example.test/api/teacher-agent/security/login/callback",
  OIDC_TRANSACTION_SECRET: "integration-transaction-secret-at-least-32-characters",
  OIDC_EXPECTED_HOST: "api.example.test",
  OIDC_ALLOWED_ALGORITHMS: "RS256",
  OIDC_AUTHENTICATION_MAX_AGE_SECONDS: "3600",
  SESSION_SECRET: "integration-session-secret-at-least-32-characters",
  SESSION_COOKIE_SECURE: "true",
  SEED_DEMO_SESSIONS: "false",
  HARNESS_GATEWAY_ENABLED: "false"
} as const;

const previous = new Map<string, string | undefined>();
const originalFetch = globalThis.fetch;
let app: NestFastifyApplication;
let privateKey: ReturnType<typeof generateKeyPairSync>["privateKey"];
let publicJwk: JsonWebKey;
let expectedNonce = "";
let tokenExchangeCount = 0;
let idTokenSeenByProvider = "";

function encode(value: unknown): string {
  return Buffer.from(JSON.stringify(value), "utf8").toString("base64url");
}

function signedIdToken(): string {
  const now = Math.floor(Date.now() / 1000);
  const header = encode({alg: "RS256", kid: "integration-key", typ: "JWT"});
  const payload = encode({
    iss: environment.OIDC_ISSUER,
    aud: environment.OIDC_CLIENT_ID,
    azp: environment.OIDC_CLIENT_ID,
    sub: "integration-user",
    org_id: "integration-tenant",
    roles: ["teacher"],
    nonce: expectedNonce,
    iat: now,
    auth_time: now - 1,
    exp: now + 300
  });
  const signer = createSign("RSA-SHA256");
  signer.update(`${header}.${payload}`, "utf8");
  signer.end();
  return `${header}.${payload}.${signer.sign(privateKey).toString("base64url")}`;
}

function allSetCookies(headers: OutgoingHttpHeaders): string[] {
  const value = headers["set-cookie"];
  return Array.isArray(value) ? value.map(String) : value === undefined ? [] : [String(value)];
}

function cookiePair(setCookies: readonly string[]): string {
  return setCookies.map((value) => value.split(";", 1)[0]).filter(Boolean).join("; ");
}

before(async () => {
  for (const [key, value] of Object.entries(environment)) {
    previous.set(key, process.env[key]);
    process.env[key] = value;
  }
  const keys = generateKeyPairSync("rsa", {modulusLength: 2048});
  privateKey = keys.privateKey;
  publicJwk = {...(keys.publicKey.export({format: "jwk"}) as JsonWebKey), kid: "integration-key", use: "sig"};
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    if (url.endsWith("/.well-known/openid-configuration")) {
      return Response.json({
        issuer: environment.OIDC_ISSUER,
        authorization_endpoint: `${environment.OIDC_ISSUER}/authorize`,
        token_endpoint: `${environment.OIDC_ISSUER}/token`,
        jwks_uri: `${environment.OIDC_ISSUER}/jwks`,
        code_challenge_methods_supported: ["S256"],
        token_endpoint_auth_methods_supported: ["client_secret_basic"]
      });
    }
    if (url.endsWith("/token")) {
      tokenExchangeCount += 1;
      const headers = new Headers(init?.headers);
      assert.match(headers.get("authorization") ?? "", /^Basic /);
      const body = new URLSearchParams(String(init?.body));
      assert.match(body.get("code_verifier") ?? "", /^[A-Za-z0-9._~-]{43,128}$/);
      idTokenSeenByProvider = signedIdToken();
      return Response.json({
        access_token: "provider-access-token-must-never-reach-browser",
        id_token: idTokenSeenByProvider,
        token_type: "Bearer"
      });
    }
    if (url.endsWith("/jwks")) return Response.json({keys: [publicJwk]});
    return new Response("not found", {status: 404});
  };
  app = await createApplication({logger: false});
  await app.init();
});

after(async () => {
  await app?.close();
  globalThis.fetch = originalFetch;
  for (const key of Object.keys(environment)) {
    const value = previous.get(key);
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
});

test("real HTTP login callback registers authority before cookies, then session and logout work", async () => {
  const begin = await app.inject({
    method: "GET",
    url: "/api/v1/auth/oidc/begin?return_to=%2Fprojects",
    headers: {
      host: environment.OIDC_EXPECTED_HOST,
      origin: "https://console.example.test",
      "sec-fetch-site": "same-origin",
      "sec-fetch-mode": "navigate"
    }
  });
  assert.equal(begin.statusCode, 303);
  const authorization = new URL(begin.headers.location ?? "");
  expectedNonce = authorization.searchParams.get("nonce") ?? "";
  const state = authorization.searchParams.get("state") ?? "";
  assert.match(expectedNonce, /^[A-Za-z0-9_-]{43}$/);
  assert.equal(authorization.searchParams.get("code_challenge_method"), "S256");
  const transactionCookie = cookiePair(allSetCookies(begin.headers));
  assert.match(transactionCookie, /^__Secure-teachlab_oidc_tx=/);
  const beginSetCookie = allSetCookies(begin.headers)[0] ?? "";
  assert.match(beginSetCookie, /; SameSite=Lax/);
  assert.match(beginSetCookie, /; HttpOnly/);
  assert.match(beginSetCookie, /; Secure/);

  const callback = await app.inject({
    method: "GET",
    url: `/api/v1/auth/oidc/callback?code=authorization-code-123&state=${state}`,
    headers: {
      host: environment.OIDC_EXPECTED_HOST,
      cookie: transactionCookie,
      "sec-fetch-site": "cross-site",
      "sec-fetch-mode": "navigate"
    }
  });
  assert.equal(callback.statusCode, 303);
  assert.equal(callback.headers.location, "/projects");
  assert.equal(tokenExchangeCount, 1);
  const callbackCookies = allSetCookies(callback.headers);
  assert.equal(callbackCookies.length, 3);
  assert.ok(callbackCookies.some((value) => value.startsWith("__Secure-teachlab_oidc_tx=") && value.includes("Max-Age=0")));
  assert.ok(callbackCookies.some((value) => value.startsWith("__Host-teachlab_session=")));
  assert.ok(callbackCookies.some((value) => value.startsWith("__Host-teachlab_csrf=")));
  assert.equal(callback.body.includes(idTokenSeenByProvider), false);
  assert.equal(JSON.stringify(callback.headers).includes(idTokenSeenByProvider), false);
  assert.equal(callback.body.includes("provider-access-token"), false);

  const authenticatedCookies = callbackCookies.filter((value) => value.startsWith("__Host-"));
  const browserCookie = cookiePair(authenticatedCookies);
  const session = await app.inject({method: "GET", url: "/api/v1/auth/session", headers: {cookie: browserCookie}});
  assert.equal(session.statusCode, 200);
  assert.deepEqual(session.json(), {authenticated: true});
  assert.match(String(session.headers["cache-control"]), /no-store/);
  for (const privateValue of ["integration-user", "integration-tenant", "teacher"]) {
    assert.equal(session.body.includes(privateValue), false);
  }
  const csrf = decodeURIComponent(
    authenticatedCookies.find((value) => value.startsWith("__Host-teachlab_csrf="))
      ?.split(";", 1)[0]?.split("=", 2)[1] ?? ""
  );
  const logout = await app.inject({
    method: "DELETE",
    url: "/api/v1/auth/session",
    headers: {
      cookie: browserCookie,
      origin: "https://console.example.test",
      "sec-fetch-site": "same-origin",
      "x-csrf-token": csrf
    }
  });
  assert.equal(logout.statusCode, 204);
  assert.equal(allSetCookies(logout.headers).length, 2);
});

test("provider errors and invalid state clear only the transaction and never reflect provider text", async () => {
  const untrusted = await app.inject({
    method: "GET",
    url: "/api/v1/auth/oidc/begin",
    headers: {
      host: "attacker.example.test",
      origin: "https://attacker.example.test",
      "sec-fetch-site": "cross-site",
      "sec-fetch-mode": "navigate"
    }
  });
  assert.equal(untrusted.statusCode, 400);
  assert.equal(allSetCookies(untrusted.headers).length, 0);
  assert.equal(untrusted.body.includes("attacker.example.test"), false);

  const begin = await app.inject({
    method: "GET",
    url: "/api/v1/auth/oidc/begin",
    headers: {
      host: environment.OIDC_EXPECTED_HOST,
      origin: "https://console.example.test",
      "sec-fetch-site": "same-origin",
      "sec-fetch-mode": "navigate"
    }
  });
  const transactionCookie = cookiePair(allSetCookies(begin.headers));
  const response = await app.inject({
    method: "GET",
    url: "/api/v1/auth/oidc/callback?error=access_denied&error_description=secret-provider-detail",
    headers: {
      host: environment.OIDC_EXPECTED_HOST,
      cookie: transactionCookie,
      "sec-fetch-site": "cross-site",
      "sec-fetch-mode": "navigate"
    }
  });
  assert.equal(response.statusCode, 400);
  assert.equal(response.body.includes("secret-provider-detail"), false);
  const cookies = allSetCookies(response.headers);
  assert.equal(cookies.length, 1);
  assert.match(cookies[0] ?? "", /Max-Age=0/);
});

test("durable authority failure never emits a usable session cookie", async () => {
  const begin = await app.inject({
    method: "GET",
    url: "/api/v1/auth/oidc/begin",
    headers: {
      host: environment.OIDC_EXPECTED_HOST,
      origin: "https://console.example.test",
      "sec-fetch-site": "same-origin",
      "sec-fetch-mode": "navigate"
    }
  });
  const authorization = new URL(begin.headers.location ?? "");
  expectedNonce = authorization.searchParams.get("nonce") ?? "";
  const state = authorization.searchParams.get("state") ?? "";
  const revocations = app.get(SessionRevocationService);
  const originalRegister = revocations.register.bind(revocations);
  revocations.register = async () => { throw new Error("durability unavailable"); };
  try {
    const callback = await app.inject({
      method: "GET",
      url: `/api/v1/auth/oidc/callback?code=authorization-code-456&state=${state}`,
      headers: {
        host: environment.OIDC_EXPECTED_HOST,
        cookie: cookiePair(allSetCookies(begin.headers)),
        "sec-fetch-site": "cross-site",
        "sec-fetch-mode": "navigate"
      }
    });
    assert.equal(callback.statusCode, 503);
    const cookies = allSetCookies(callback.headers);
    assert.equal(cookies.some((value) => value.startsWith("__Host-teachlab_session=")), false);
    assert.equal(cookies.length, 1);
  } finally {
    revocations.register = originalRegister;
  }
});
