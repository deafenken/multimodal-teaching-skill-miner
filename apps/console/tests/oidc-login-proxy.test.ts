import assert from "node:assert/strict";
import test from "node:test";

import {proxyOidcBegin, proxyOidcCallback} from "../lib/oidc-login-proxy.ts";

const API_ORIGIN = "http://127.0.0.1:49200";
const CONSOLE_ORIGIN = "http://console.example.test";
const TRANSACTION = "t".repeat(96);
const SESSION = "signed-session-value";
process.env.TEACHLAB_HARNESS_MODE = "authenticated_apps_api";
process.env.TEACHLAB_APPS_API_URL = API_ORIGIN;
process.env.TEACHLAB_CONSOLE_ORIGIN = CONSOLE_ORIGIN;
process.env.TEACHLAB_APPS_API_COOKIE_MODE = "development";

function browserRequest(path: string, headers: Record<string, string> = {}) {
  return new Request(`${CONSOLE_ORIGIN}${path}`, {
    headers: {
      Host: "console.example.test",
      "Sec-Fetch-Site": "same-origin",
      "Sec-Fetch-Mode": "navigate",
      ...headers,
    },
  });
}

test("begin validates same-origin navigation and rewrites only transaction callback Path", async (t) => {
  const calls: Array<{url: string; headers: Headers}> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    calls.push({url: String(input), headers: new Headers(init?.headers)});
    const headers = new Headers({Location: "https://identity.example.test/authorize?state=opaque"});
    headers.append("Set-Cookie", `teachlab_oidc_tx=${TRANSACTION}; Path=/api/v1/auth/oidc/callback; SameSite=Lax; Max-Age=300; HttpOnly`);
    return new Response(null, {status: 303, headers});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const response = await proxyOidcBegin(browserRequest(
    "/api/teacher-agent/security/login?return_to=%2Fprojects",
    {Origin: CONSOLE_ORIGIN},
  ));
  assert.equal(response.status, 303);
  assert.equal(response.headers.get("location"), "https://identity.example.test/authorize?state=opaque");
  assert.match(
    response.headers.getSetCookie()[0] ?? "",
    /Path=\/api\/teacher-agent\/security\/login\/callback/,
  );
  assert.equal(calls.length, 1);
  assert.equal(calls[0]?.url, `${API_ORIGIN}/api/v1/auth/oidc/begin?return_to=%2Fprojects`);
  assert.equal(calls[0]?.headers.get("host"), "127.0.0.1:49200");

  const rejected = await proxyOidcBegin(browserRequest(
    "/api/teacher-agent/security/login?return_to=%2F%2Fevil.example.test",
    {Origin: CONSOLE_ORIGIN},
  ));
  assert.equal(rejected.status, 400);
  assert.equal(calls.length, 1);

  globalThis.fetch = (async () => {
    const headers = new Headers({Location: "https://identity.example.test/authorize?state=opaque"});
    headers.append(
      "Set-Cookie",
      `teachlab_oidc_tx=${TRANSACTION}; Path=/api/v1/auth/oidc/callback; SameSite=Lax; SameSite=None; Max-Age=300; HttpOnly`,
    );
    return new Response(null, {status: 303, headers});
  }) as typeof fetch;
  const ambiguousCookie = await proxyOidcBegin(browserRequest(
    "/api/teacher-agent/security/login",
    {Origin: CONSOLE_ORIGIN},
  ));
  assert.equal(ambiguousCookie.status, 503);
  assert.equal(ambiguousCookie.headers.getSetCookie().length, 0);
});

test("callback forwards only transaction cookie, relays strict session cookies, and rejects open redirect", async (t) => {
  const calls: Array<{headers: Headers; url: string}> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    calls.push({headers: new Headers(init?.headers), url: String(input)});
    const headers = new Headers({Location: "/"});
    headers.append("Set-Cookie", "teachlab_oidc_tx=; Path=/api/v1/auth/oidc/callback; SameSite=Lax; Max-Age=0; HttpOnly");
    headers.append("Set-Cookie", `teachlab_session=${SESSION}; Path=/; SameSite=Strict; Max-Age=3600; HttpOnly`);
    headers.append("Set-Cookie", `teachlab_csrf=${"c".repeat(43)}; Path=/; SameSite=Strict; Max-Age=3600`);
    return new Response(null, {status: 303, headers});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const response = await proxyOidcCallback(browserRequest(
    "/api/teacher-agent/security/login/callback?code=code-12345&state=state",
    {
      Cookie: `analytics=never; teachlab_oidc_tx=${TRANSACTION}; teachlab_session=never-forward`,
      "Sec-Fetch-Site": "cross-site",
    },
  ));
  assert.equal(response.status, 303);
  assert.equal(response.headers.get("location"), "/");
  assert.equal(response.headers.getSetCookie().length, 4);
  assert.match(response.headers.getSetCookie()[0] ?? "", /Path=\/api\/teacher-agent\/security\/login\/callback/);
  const transition = response.headers.getSetCookie().find((value) =>
    value.startsWith("teachlab_account_transition="),
  );
  assert.match(transition ?? "", /^teachlab_account_transition=pending_v1;/);
  assert.match(transition ?? "", /Path=\/; SameSite=Strict; Max-Age=3600$/);
  assert.doesNotMatch(transition ?? "", /HttpOnly|signed-session-value|student|tenant/i);
  assert.equal(calls[0]?.headers.get("cookie"), `teachlab_oidc_tx=${TRANSACTION}`);
  assert.equal(calls[0]?.headers.has("authorization"), false);

  globalThis.fetch = (async () => {
    const headers = new Headers({Location: "https://evil.example.test/steal"});
    headers.append("Set-Cookie", "teachlab_oidc_tx=; Path=/api/v1/auth/oidc/callback; SameSite=Lax; Max-Age=0; HttpOnly");
    headers.append("Set-Cookie", `teachlab_session=${SESSION}; Path=/; SameSite=Strict; Max-Age=3600; HttpOnly`);
    headers.append("Set-Cookie", `teachlab_csrf=${"c".repeat(43)}; Path=/; SameSite=Strict; Max-Age=3600`);
    return new Response(null, {status: 303, headers});
  }) as typeof fetch;
  const rejected = await proxyOidcCallback(browserRequest(
    "/api/teacher-agent/security/login/callback?code=code-12345&state=state",
    {Cookie: `teachlab_oidc_tx=${TRANSACTION}`, "Sec-Fetch-Site": "cross-site"},
  ));
  assert.equal(rejected.status, 503);
  assert.equal(rejected.headers.getSetCookie().length, 1);
  assert.match(rejected.headers.getSetCookie()[0] ?? "", /Max-Age=0/);

  const duplicate = await proxyOidcCallback(browserRequest(
    "/api/teacher-agent/security/login/callback?code=one&code=two&state=state",
    {Cookie: `teachlab_oidc_tx=${TRANSACTION}`, "Sec-Fetch-Site": "cross-site"},
  ));
  assert.equal(duplicate.status, 400);
  assert.equal(duplicate.headers.getSetCookie().length, 1);
});

test("local mode does not expose an organization login proxy", async () => {
  const previous = process.env.TEACHLAB_HARNESS_MODE;
  process.env.TEACHLAB_HARNESS_MODE = "local_python";
  try {
    const response = await proxyOidcBegin(browserRequest("/api/teacher-agent/security/login"));
    assert.equal(response.status, 404);
  } finally {
    process.env.TEACHLAB_HARNESS_MODE = previous;
  }
});
