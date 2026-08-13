import assert from "node:assert/strict";
import test from "node:test";

import {NextRequest} from "next/server.js";

import {GET as proxyGet, POST as proxyPost} from "../app/api/teacher-agent/[...path]/route.ts";
import {POST as cancelPost} from "../app/api/teacher-agent/cancel/route.ts";
import {
  DELETE as sessionDelete,
  GET as sessionGet,
  POST as sessionPost,
} from "../app/api/teacher-agent/security/session/route.ts";
import {POST as streamPost} from "../app/api/teacher-agent/stream/route.ts";
import {
  harnessBffConfiguration,
  harnessProxyRequestBodyLimit,
  harnessReadinessTarget,
} from "../lib/harness-bff.ts";

const API_ORIGIN = "http://127.0.0.1:49200";
const CONSOLE_ORIGIN = "http://console.example.test";
const CSRF_A = "a".repeat(43);
const CSRF_B = "b".repeat(43);

process.env.TEACHLAB_HARNESS_MODE = "authenticated_apps_api";
process.env.TEACHLAB_APPS_API_URL = API_ORIGIN;
process.env.TEACHLAB_CONSOLE_ORIGIN = CONSOLE_ORIGIN;
process.env.TEACHLAB_APPS_API_COOKIE_MODE = "development";

interface ScopeCookie {
  session: string;
  csrf: string;
}

function browserRequest(
  path: string,
  options: {
    method?: "GET" | "POST" | "DELETE";
    scope?: ScopeCookie;
    body?: string;
    headers?: Record<string, string>;
    signal?: AbortSignal;
  } = {},
) {
  const method = options.method ?? "GET";
  const headers = new Headers({
    Host: "console.example.test",
    "Sec-Fetch-Site": "same-origin",
  });
  if (method === "POST" || method === "DELETE") {
    headers.set("Origin", CONSOLE_ORIGIN);
    headers.set("Content-Type", "application/json");
  }
  if (options.scope) {
    headers.set(
      "Cookie",
      `analytics=never-forward; teachlab_local_session=never-forward; teachlab_session=${options.scope.session}; teachlab_csrf=${options.scope.csrf}`,
    );
    headers.set("x-teachlab-csrf-token", options.scope.csrf);
  }
  for (const [name, value] of Object.entries(options.headers ?? {})) headers.set(name, value);
  return new NextRequest(`${CONSOLE_ORIGIN}${path}`, {
    method,
    headers,
    body: options.body,
    signal: options.signal,
  });
}

function chunkedBrowserRequest(
  path: string,
  scope: ScopeCookie,
  chunks: readonly Uint8Array[],
  declaredBytes?: string,
): {request: NextRequest; cancelled: () => boolean; pulls: () => number} {
  let index = 0;
  let wasCancelled = false;
  let pullCount = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      pullCount += 1;
      const chunk = chunks[index++];
      if (chunk) controller.enqueue(chunk);
      else controller.close();
    },
    cancel() {
      wasCancelled = true;
    },
  });
  const headers = new Headers({
    Host: "console.example.test",
    Origin: CONSOLE_ORIGIN,
    "Sec-Fetch-Site": "same-origin",
    "Content-Type": "application/json",
    Cookie: `teachlab_session=${scope.session}; teachlab_csrf=${scope.csrf}`,
    "x-teachlab-csrf-token": scope.csrf,
  });
  if (declaredBytes !== undefined) headers.set("Content-Length", declaredBytes);
  const request = new NextRequest(`${CONSOLE_ORIGIN}${path}`, {
    method: "POST",
    headers,
    body,
    duplex: "half",
  } as never);
  return {
    request,
    cancelled: () => wasCancelled,
    pulls: () => pullCount,
  };
}

test("transport mode is mandatory and production refuses insecure apps/api fallback", () => {
  const mutableEnvironment = process.env as Record<string, string | undefined>;
  const snapshot = {
    mode: process.env.TEACHLAB_HARNESS_MODE,
    api: process.env.TEACHLAB_APPS_API_URL,
    console: process.env.TEACHLAB_CONSOLE_ORIGIN,
    cookieMode: process.env.TEACHLAB_APPS_API_COOKIE_MODE,
    nodeEnv: process.env.NODE_ENV,
  };
  try {
    delete process.env.TEACHLAB_HARNESS_MODE;
    assert.throws(() => harnessBffConfiguration(), /mode_not_configured/);

    process.env.TEACHLAB_HARNESS_MODE = "authenticated_apps_api";
    mutableEnvironment.NODE_ENV = "production";
    process.env.TEACHLAB_APPS_API_URL = API_ORIGIN;
    process.env.TEACHLAB_CONSOLE_ORIGIN = "https://console.example.test";
    process.env.TEACHLAB_APPS_API_COOKIE_MODE = "development";
    assert.throws(() => harnessBffConfiguration(), /upstream_not_configured/);
  } finally {
    process.env.TEACHLAB_HARNESS_MODE = snapshot.mode;
    process.env.TEACHLAB_APPS_API_URL = snapshot.api;
    process.env.TEACHLAB_CONSOLE_ORIGIN = snapshot.console;
    process.env.TEACHLAB_APPS_API_COOKIE_MODE = snapshot.cookieMode;
    if (snapshot.nodeEnv === undefined) delete mutableEnvironment.NODE_ENV;
    else mutableEnvironment.NODE_ENV = snapshot.nodeEnv;
  }
});

test("production readiness uses the private API service address and cannot recurse through edge", () => {
  const mutableEnvironment = process.env as Record<string, string | undefined>;
  const snapshot = {
    mode: process.env.TEACHLAB_HARNESS_MODE,
    api: process.env.TEACHLAB_APPS_API_URL,
    internal: process.env.TEACHLAB_APPS_API_INTERNAL_URL,
    console: process.env.TEACHLAB_CONSOLE_ORIGIN,
    cookieMode: process.env.TEACHLAB_APPS_API_COOKIE_MODE,
    nodeEnv: process.env.NODE_ENV,
  };
  try {
    process.env.TEACHLAB_HARNESS_MODE = "authenticated_apps_api";
    process.env.TEACHLAB_APPS_API_URL = "https://console.example.test";
    process.env.TEACHLAB_APPS_API_INTERNAL_URL = "http://api:4000";
    process.env.TEACHLAB_CONSOLE_ORIGIN = "https://console.example.test";
    process.env.TEACHLAB_APPS_API_COOKIE_MODE = "secure";
    mutableEnvironment.NODE_ENV = "production";
    assert.equal(harnessReadinessTarget(harnessBffConfiguration()).href, "http://api:4000/ready");
    process.env.TEACHLAB_APPS_API_INTERNAL_URL = "https://console.example.test";
    assert.throws(
      () => harnessReadinessTarget(harnessBffConfiguration()),
      /upstream_not_configured/,
    );
  } finally {
    process.env.TEACHLAB_HARNESS_MODE = snapshot.mode;
    process.env.TEACHLAB_APPS_API_URL = snapshot.api;
    process.env.TEACHLAB_APPS_API_INTERNAL_URL = snapshot.internal;
    process.env.TEACHLAB_CONSOLE_ORIGIN = snapshot.console;
    process.env.TEACHLAB_APPS_API_COOKIE_MODE = snapshot.cookieMode;
    if (snapshot.nodeEnv === undefined) delete mutableEnvironment.NODE_ENV;
    else mutableEnvironment.NODE_ENV = snapshot.nodeEnv;
  }
});

test("two browser scopes route to the authenticated gateway with only their apps/api cookies", async (t) => {
  const calls: Array<{url: string; headers: Headers; body: string}> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    calls.push({
      url: String(input),
      headers: new Headers(init?.headers),
      body: String(init?.body ?? ""),
    });
    return Response.json({ok: true});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const scopes = [
    {session: "signed-session-scope-a", csrf: CSRF_A},
    {session: "signed-session-scope-b", csrf: CSRF_B},
  ];
  for (const scope of scopes) {
    const response = await proxyPost(
      browserRequest("/api/teacher-agent/api/tasks/status", {
        method: "POST",
        scope,
        body: JSON.stringify({task_id: "task_same_browser_guess"}),
      }),
      {params: Promise.resolve({path: ["api", "tasks", "status"]})},
    );
    assert.equal(response.status, 200);
    await response.text();
  }

  assert.equal(calls.length, 2);
  assert.ok(calls.every((call) => call.url === `${API_ORIGIN}/api/v1/harness/api/tasks/status`));
  assert.equal(calls[0].headers.get("cookie"), `teachlab_session=signed-session-scope-a; teachlab_csrf=${CSRF_A}`);
  assert.equal(calls[1].headers.get("cookie"), `teachlab_session=signed-session-scope-b; teachlab_csrf=${CSRF_B}`);
  assert.equal(calls[0].headers.get("x-csrf-token"), CSRF_A);
  assert.equal(calls[1].headers.get("x-csrf-token"), CSRF_B);
  for (const call of calls) {
    assert.equal(call.headers.has("x-teachlab-csrf-token"), false);
    assert.equal(call.headers.has("x-tenant"), false);
    assert.equal(call.headers.has("x-user"), false);
    assert.equal(call.headers.has("authorization"), false);
    assert.equal(call.headers.get("origin"), CONSOLE_ORIGIN);
    assert.equal(call.body, JSON.stringify({task_id: "task_same_browser_guess"}));
  }
});

test("safeguarding staff routes are POST-only, bounded, and preserve only the opaque body", async (t) => {
  const calls: Array<{url: string; body: string}> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    calls.push({url: String(input), body: String(init?.body ?? "")});
    return Response.json({ok: true});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const scope = {session: "signed-session-safeguarding", csrf: CSRF_A};
  const locator = `sgr1_k1_${"A".repeat(80)}`;
  const body = JSON.stringify({
    safeguarding_route_locator: locator,
    safeguarding_idempotency_key: "console-safeguarding-list-test",
  });
  const response = await proxyPost(
    browserRequest("/api/teacher-agent/api/safeguarding/list", {
      method: "POST",
      scope,
      body,
    }),
    {params: Promise.resolve({path: ["api", "safeguarding", "list"]})},
  );
  assert.equal(response.status, 200);
  await response.text();
  assert.deepEqual(calls, [{
    url: `${API_ORIGIN}/api/v1/harness/api/safeguarding/list`,
    body,
  }]);

  const getResponse = await proxyGet(
    browserRequest("/api/teacher-agent/api/safeguarding/list", {scope}),
    {params: Promise.resolve({path: ["api", "safeguarding", "list"]})},
  );
  assert.equal(getResponse.status, 404);
  assert.equal(calls.length, 1);

  const oversized = await proxyPost(
    browserRequest("/api/teacher-agent/api/safeguarding/case/acknowledge", {
      method: "POST",
      scope,
      body: JSON.stringify({padding: "x".repeat(17 * 1024)}),
    }),
    {params: Promise.resolve({path: ["api", "safeguarding", "case", "acknowledge"]})},
  );
  assert.equal(oversized.status, 413);
  assert.equal(calls.length, 1);
});

test("all public Harness BFF mutations cancel chunked bodies at cap plus one before upstream", async (t) => {
  let upstreamCalls = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    upstreamCalls += 1;
    return Response.json({unexpected: true});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const scope = {session: "signed-session-bounded-body", csrf: CSRF_A};
  const routes = [
    {
      name: "catch-all",
      maximum: harnessProxyRequestBodyLimit("api/chat"),
      invoke: (request: NextRequest) => proxyPost(request, {
        params: Promise.resolve({path: ["api", "chat"]}),
      }),
    },
    {
      name: "cancel",
      maximum: 4 * 1024,
      invoke: (request: NextRequest) => cancelPost(request),
    },
    {
      name: "stream",
      maximum: 64 * 1024,
      invoke: (request: NextRequest) => streamPost(request),
    },
  ] as const;
  for (const route of routes) {
    await t.test(route.name, async () => {
      const chunked = chunkedBrowserRequest(
        `/api/teacher-agent/${route.name}`,
        scope,
        [new Uint8Array(route.maximum), new Uint8Array(1), new Uint8Array(1)],
      );
      const response = await route.invoke(chunked.request);
      assert.equal(response.status, 413);
      assert.equal(chunked.cancelled(), true);
      assert.ok(chunked.pulls() <= 3);
      assert.equal(upstreamCalls, 0);
    });
  }
});

test("public Harness BFF rejects declared and observed length mismatch before upstream", async (t) => {
  let upstreamCalls = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    upstreamCalls += 1;
    return Response.json({unexpected: true});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const scope = {session: "signed-session-length-mismatch", csrf: CSRF_A};

  for (const scenario of [
    {
      name: "catch-all",
      declared: "1",
      invoke: (request: NextRequest) => proxyPost(request, {
        params: Promise.resolve({path: ["api", "chat"]}),
      }),
    },
    {
      name: "cancel",
      declared: "3",
      invoke: (request: NextRequest) => cancelPost(request),
    },
    {
      name: "stream",
      declared: "1",
      invoke: (request: NextRequest) => streamPost(request),
    },
  ] as const) {
    await t.test(scenario.name, async () => {
      const mismatch = chunkedBrowserRequest(
        `/api/teacher-agent/${scenario.name}`,
        scope,
        [new TextEncoder().encode("{}")],
        scenario.declared,
      );
      assert.equal((await scenario.invoke(mismatch.request)).status, 413);
      assert.equal(upstreamCalls, 0);
    });
  }
  assert.equal(upstreamCalls, 0);
});

test("bootstrap exposes bounded consent terms but never internal signed subject authority", async (t) => {
  const originalFetch = globalThis.fetch;
  let upstreamBody = "";
  globalThis.fetch = (async (_input, init) => {
    upstreamBody = String(init?.body ?? "");
    return Response.json({
      remote_consent: {
        configured: true,
        policies: [{
          purpose: "remote_teaching",
          provider_id: "deepseek",
          processing_region: "cn_north",
          data_categories: ["learner_message"],
          provider_retention_days: 7,
          provider_policy: {
            policy_id: "approved-provider-terms",
            policy_version: "2026-08-12",
            deletion_status: "outside_service_control_subject_to_provider_policy",
            documentation_url: "https://provider.example/privacy"
          },
          provider_policy_sha256: "a".repeat(64),
          subject_policy: {
            policy_id: "school-policy",
            policy_version: "v3",
            likely_minor: true,
            guardian_or_school_policy: "verified_school_policy",
            remote_processing_eligible: true
          },
          subject_policy_sha256: "b".repeat(64),
          remote_processing_eligible: true
        }],
        receipts: []
      }
    });
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const response = await proxyGet(
    browserRequest("/api/teacher-agent/api/bootstrap", {
      scope: {session: "signed-session-policy", csrf: CSRF_A}
    }),
    {params: Promise.resolve({path: ["api", "bootstrap"]})}
  );
  assert.equal(response.status, 200);
  const payload = await response.json() as Record<string, any>;
  const policy = payload.remote_consent.policies[0];
  assert.equal(policy.remote_processing_eligible, true);
  assert.equal(policy.subject_policy.guardian_or_school_policy, "verified_school_policy");
  assert.equal(policy.provider_policy.deletion_status,
    "outside_service_control_subject_to_provider_policy");
  assert.equal("remoteSubjectPolicy" in payload, false);
  assert.equal("principal" in payload, false);
  assert.equal(upstreamBody, "");
});

test("identity override headers and invalid browser authority stop before every backend side effect", async (t) => {
  let backendCalls = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    backendCalls += 1;
    return Response.json({unexpected: true});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const scope = {session: "signed-session-a", csrf: CSRF_A};

  for (const forbidden of ["x-tenant", "x-user-id", "x-principal", "x-dev-tenant-id"]) {
    const response = await proxyPost(
      browserRequest("/api/teacher-agent/api/projects/bootstrap", {
        method: "POST",
        scope,
        body: "{}",
        headers: {[forbidden]: "attacker-controlled"},
      }),
      {params: Promise.resolve({path: ["api", "projects", "bootstrap"]})},
    );
    assert.equal(response.status, 403, forbidden);
    assert.equal(response.headers.get("x-teachlab-security-rejection"), "identity_override_rejected");
  }

  const forgedHost = new NextRequest(`${CONSOLE_ORIGIN}/api/teacher-agent/api/bootstrap`, {
    headers: {
      Host: "evil.example.test",
      "Sec-Fetch-Site": "same-origin",
      Cookie: `teachlab_session=${scope.session}; teachlab_csrf=${scope.csrf}`,
    },
  });
  const hostResponse = await proxyGet(forgedHost, {
    params: Promise.resolve({path: ["api", "bootstrap"]}),
  });
  assert.equal(hostResponse.status, 403);

  const badCsrf = await cancelPost(browserRequest("/api/teacher-agent/cancel", {
    method: "POST",
    scope: {...scope, csrf: CSRF_B},
    body: "{}",
    headers: {"x-teachlab-csrf-token": CSRF_A},
  }));
  assert.equal(badCsrf.status, 403);
  assert.equal(backendCalls, 0);
});

test("legacy session exchange filters principal JSON and never forwards browser tokens or identity headers", async (t) => {
  const calls: Array<{url: string; headers: Headers; body: string}> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    calls.push({url: String(input), headers: new Headers(init?.headers), body: String(init?.body ?? "")});
    const headers = new Headers();
    headers.append("Set-Cookie", "teachlab_session=signed-by-apps-api; Path=/; SameSite=Strict; Max-Age=3600; HttpOnly");
    headers.append("Set-Cookie", `teachlab_csrf=${CSRF_A}; Path=/; SameSite=Strict; Max-Age=3600`);
    return Response.json({
      principal: {tenantId: "must-not-cross", subject: "must-not-cross"},
      csrfToken: CSRF_A,
      expiresAt: "2026-08-12T12:00:00.000Z",
      mode: "oidc",
    }, {status: 201, headers});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const response = await sessionPost(browserRequest("/api/teacher-agent/security/session", {
    method: "POST",
    body: JSON.stringify({principal: {tenantId: "attacker"}}),
    headers: {Authorization: "Bearer signed-oidc-token"},
  }));
  assert.equal(response.status, 200);
  const payload = await response.json() as Record<string, unknown>;
  assert.deepEqual(Object.keys(payload).sort(), ["authenticated", "connection", "csrf_token", "expires_at"]);
  assert.equal(JSON.stringify(payload).includes("must-not-cross"), false);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `${API_ORIGIN}/api/v1/auth/session`);
  assert.equal(calls[0].headers.has("authorization"), false);
  assert.equal(calls[0].headers.has("cookie"), false);
  assert.equal(calls[0].body, "{}");
  assert.equal(response.headers.getSetCookie().length, 2);
});

test("existing session check discards apps/api principal and returns only in-memory CSRF bootstrap data", async (t) => {
  let forwardedCookie = "";
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (_input, init) => {
    forwardedCookie = new Headers(init?.headers).get("cookie") ?? "";
    return Response.json({principal: {tenantId: "private-tenant", subject: "private-owner"}});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const response = await sessionGet(browserRequest("/api/teacher-agent/security/session", {
    scope: {session: "signed-session-a", csrf: CSRF_A},
  }));
  assert.equal(response.status, 200);
  const payload = await response.json() as Record<string, unknown>;
  assert.deepEqual(Object.keys(payload).sort(), ["authenticated", "connection", "csrf_token"]);
  assert.equal(JSON.stringify(payload).includes("private-tenant"), false);
  assert.equal(forwardedCookie, `teachlab_session=signed-session-a; teachlab_csrf=${CSRF_A}`);
});

test("same-origin logout forwards only bound cookies and CSRF, then relays clearing cookies", async (t) => {
  const calls: Array<{method: string; headers: Headers}> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (_input, init) => {
    calls.push({method: init?.method ?? "GET", headers: new Headers(init?.headers)});
    const headers = new Headers();
    headers.append("Set-Cookie", "teachlab_session=; Path=/; SameSite=Strict; Max-Age=0; HttpOnly");
    headers.append("Set-Cookie", "teachlab_csrf=; Path=/; SameSite=Strict; Max-Age=0");
    return new Response(null, {status: 204, headers});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const scope = {session: "signed-session-a", csrf: CSRF_A};
  const response = await sessionDelete(browserRequest(
    "/api/teacher-agent/security/session",
    {method: "DELETE", scope},
  ));
  assert.equal(response.status, 204);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].method, "DELETE");
  assert.equal(calls[0].headers.get("cookie"), `teachlab_session=signed-session-a; teachlab_csrf=${CSRF_A}`);
  assert.equal(calls[0].headers.get("x-csrf-token"), CSRF_A);
  assert.equal(calls[0].headers.get("origin"), CONSOLE_ORIGIN);
  assert.ok(response.headers.getSetCookie().every((cookie) => /Max-Age=0/.test(cookie)));
});

test("logout authority failure is explicit and never clears browser cookies", async (t) => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => Response.json(
    {error: "revocation store unavailable"},
    {status: 503},
  )) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const response = await sessionDelete(browserRequest(
    "/api/teacher-agent/security/session",
    {method: "DELETE", scope: {session: "signed-session-a", csrf: CSRF_A}},
  ));
  assert.equal(response.status, 503);
  assert.equal(response.headers.getSetCookie().length, 0);
  assert.equal((await response.json() as {code: string}).code, "identity_service_unavailable");
});

test("logout rejects a successful upstream response that does not carry exact clearing cookies", async (t) => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    const headers = new Headers();
    headers.append("Set-Cookie", "teachlab_session=reissued; Path=/; SameSite=Strict; Max-Age=3600; HttpOnly");
    headers.append("Set-Cookie", `teachlab_csrf=${CSRF_A}; Path=/; SameSite=Strict; Max-Age=3600`);
    return new Response(null, {status: 204, headers});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const response = await sessionDelete(browserRequest(
    "/api/teacher-agent/security/session",
    {method: "DELETE", scope: {session: "signed-session-a", csrf: CSRF_A}},
  ));
  assert.equal(response.status, 503);
  assert.equal(response.headers.getSetCookie().length, 0);
});

test("SSE, detach, explicit cancel, and project routes use the real harness gateway paths", async (t) => {
  const calls: string[] = [];
  let streamSignal: AbortSignal | null = null;
  let streamCancelled = false;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    const url = String(input);
    calls.push(url);
    if (url.endsWith("/api/stream")) {
      streamSignal = init?.signal ?? null;
      return new Response(new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("event: run.started\ndata: {}\n\n"));
        },
        cancel() {
          streamCancelled = true;
        },
      }), {
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "X-Harness-Run-ID": "run_scope_a",
          "X-Harness-Turn-ID": "turn_scope_a",
          "X-Background-Task-ID": `task_${"1".repeat(40)}`,
          "X-Background-Task-Version": "0",
        },
      });
    }
    return Response.json({ok: true});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const scope = {session: "signed-session-a", csrf: CSRF_A};

  const stream = await streamPost(browserRequest("/api/teacher-agent/stream", {
    method: "POST",
    scope,
    body: JSON.stringify({operation: "chat", payload: {messages: []}, request_id: "same-request"}),
  }));
  assert.equal(stream.status, 200);
  assert.equal(stream.headers.get("x-harness-run-id"), "run_scope_a");
  assert.equal(stream.headers.get("x-harness-turn-id"), "turn_scope_a");
  assert.equal(stream.headers.get("x-background-task-id"), `task_${"1".repeat(40)}`);
  await stream.body?.cancel("navigation_detach");
  assert.equal((streamSignal as AbortSignal | null)?.aborted, true);
  assert.equal(streamCancelled, true);
  assert.deepEqual(calls, [`${API_ORIGIN}/api/v1/harness/api/stream`]);

  const cancel = await cancelPost(browserRequest("/api/teacher-agent/cancel", {
    method: "POST",
    scope,
    body: JSON.stringify({request_id: "same-request", run_id: "run_scope_a"}),
  }));
  assert.equal(cancel.status, 200);
  await cancel.text();

  const project = await proxyPost(
    browserRequest("/api/teacher-agent/api/projects/bootstrap", {method: "POST", scope, body: "{}"}),
    {params: Promise.resolve({path: ["api", "projects", "bootstrap"]})},
  );
  assert.equal(project.status, 200);
  await project.text();
  assert.deepEqual(calls, [
    `${API_ORIGIN}/api/v1/harness/api/stream`,
    `${API_ORIGIN}/api/v1/harness/api/cancel`,
    `${API_ORIGIN}/api/v1/harness/api/projects/bootstrap`,
  ]);
});
