import assert from "node:assert/strict";
import test from "node:test";

import {NextRequest} from "next/server.js";

import {POST as cancelPost} from "../app/api/teacher-agent/cancel/route.ts";
import {GET as proxyGet, POST as proxyPost} from "../app/api/teacher-agent/[...path]/route.ts";
import {GET as securitySessionGet} from "../app/api/teacher-agent/security/session/route.ts";
import {POST as streamPost} from "../app/api/teacher-agent/stream/route.ts";
import {
  LOCAL_CSRF_HEADER_NAME,
  LOCAL_SECURITY_COOKIE_NAME,
  LOCAL_SECURITY_REJECTION_HEADER,
  localRequestSecurityTestApi,
} from "../lib/local-request-security.ts";
import {CONTENT_SECURITY_POLICY, SECURITY_RESPONSE_HEADERS} from "../lib/security-headers.ts";

const capabilityUrl = "http://127.0.0.1:49199/local-capability/";
process.env.TEACHLAB_HARNESS_MODE = "local_python";
process.env.TEACHER_AGENT_CAPABILITY_URL = capabilityUrl;

interface BrowserSession {
  cookie: string;
  csrf: string;
}

function browserRequest(
  path: string,
  options: {
    authority?: string;
    method?: string;
    origin?: string | null;
    fetchSite?: string | null;
    session?: BrowserSession;
    body?: string;
    forwardedHost?: string;
  } = {},
) {
  const authority = options.authority ?? "127.0.0.1:3030";
  const method = options.method ?? "GET";
  const headers = new Headers({host: authority});
  if (options.fetchSite !== null) headers.set("sec-fetch-site", options.fetchSite ?? "same-origin");
  if (options.origin !== null && (options.origin !== undefined || method !== "GET")) {
    headers.set("origin", options.origin ?? `http://${authority}`);
  }
  if (options.forwardedHost) headers.set("x-forwarded-host", options.forwardedHost);
  if (options.session) {
    headers.set("cookie", options.session.cookie);
    headers.set(LOCAL_CSRF_HEADER_NAME, options.session.csrf);
  }
  if (method !== "GET" && method !== "HEAD") headers.set("content-type", "application/json");
  return new NextRequest(`http://${authority}${path}`, {method, headers, body: options.body});
}

async function createBrowserSession(authority = "127.0.0.1:3030"): Promise<BrowserSession> {
  const response = await securitySessionGet(browserRequest("/api/teacher-agent/security/session", {authority}));
  assert.equal(response.status, 200);
  const payload = await response.json() as {connection?: unknown; csrf_token?: unknown};
  assert.equal(payload.connection, "local_python");
  assert.match(String(payload.csrf_token), /^[A-Za-z0-9_-]{43}$/);
  const setCookie = response.headers.get("set-cookie");
  assert.ok(setCookie);
  assert.match(setCookie, /; HttpOnly;/i);
  assert.match(setCookie, /; SameSite=Strict/i);
  assert.match(setCookie, /; Path=\/api\/teacher-agent/i);
  return {
    cookie: setCookie.split(";", 1)[0],
    csrf: String(payload.csrf_token),
  };
}

test("local authority parser accepts only exact localhost, IPv4, and IPv6 loopback spellings", () => {
  const parse = localRequestSecurityTestApi.localAuthority;
  for (const value of ["localhost", "LOCALHOST:3030", "127.0.0.1:3030", "[::1]:3030"]) {
    assert.ok(parse(value), value);
  }
  for (const value of [
    "localhost.",
    "localhost.evil.example",
    "127.0.0.2:3030",
    "2130706433:3030",
    "[::ffff:127.0.0.1]:3030",
    "127.0.0.1:0",
    "127.0.0.1:080",
    "127.0.0.1:65536",
    " 127.0.0.1:3030",
    "127.0.0.1:3030, evil.example",
  ]) {
    assert.equal(parse(value), null, value);
  }
});

test("session minting fails closed for missing Fetch Metadata and hostile origins", async () => {
  const missingMetadata = await securitySessionGet(browserRequest(
    "/api/teacher-agent/security/session",
    {fetchSite: null},
  ));
  assert.equal(missingMetadata.status, 403);
  assert.equal(missingMetadata.headers.get(LOCAL_SECURITY_REJECTION_HEADER), "invalid_fetch_metadata");
  assert.equal(missingMetadata.headers.get("set-cookie"), null);

  const hostileOrigin = await securitySessionGet(browserRequest(
    "/api/teacher-agent/security/session",
    {origin: "https://evil.example"},
  ));
  assert.equal(hostileOrigin.status, 403);
  assert.equal(hostileOrigin.headers.get(LOCAL_SECURITY_REJECTION_HEADER), "invalid_origin");
  assert.equal(hostileOrigin.headers.get("set-cookie"), null);
});

test("a second tab reuses the valid HttpOnly session instead of rotating the shared cookie", async () => {
  const session = await createBrowserSession();
  const response = await securitySessionGet(browserRequest(
    "/api/teacher-agent/security/session",
    {session},
  ));
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("set-cookie"), null);
  const payload = await response.json() as {csrf_token?: unknown; expires_in_seconds?: unknown};
  assert.equal(payload.csrf_token, session.csrf);
  assert.ok(Number(payload.expires_in_seconds) > 0);
});

test("hostile GET, mutation, cancel, and stream requests are 403 before any backend call", async (t) => {
  const session = await createBrowserSession();
  let backendCalls = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    backendCalls += 1;
    return Response.json({unexpected: true});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const hostileGet = await proxyGet(
    browserRequest("/api/teacher-agent/api/bootstrap", {
      origin: "https://evil.example",
      fetchSite: "cross-site",
      session,
    }),
    {params: Promise.resolve({path: ["api", "bootstrap"]})},
  );
  assert.equal(hostileGet.status, 403);

  const hostilePost = await proxyPost(
    browserRequest("/api/teacher-agent/api/start", {
      method: "POST",
      origin: "https://evil.example",
      fetchSite: "cross-site",
      session,
      body: "{}",
    }),
    {params: Promise.resolve({path: ["api", "start"]})},
  );
  assert.equal(hostilePost.status, 403);

  const hostileCancel = await cancelPost(browserRequest("/api/teacher-agent/cancel", {
    method: "POST",
    origin: "https://evil.example",
    fetchSite: "cross-site",
    session,
    body: "{}",
  }));
  assert.equal(hostileCancel.status, 403);

  const hostileStream = await streamPost(browserRequest("/api/teacher-agent/stream", {
    method: "POST",
    origin: "https://evil.example",
    fetchSite: "cross-site",
    session,
    body: JSON.stringify({operation: "chat", payload: {messages: []}}),
  }));
  assert.equal(hostileStream.status, 403);

  const forwardedHostSpoof = await proxyGet(
    browserRequest("/api/teacher-agent/api/bootstrap", {
      authority: "evil.example:3030",
      forwardedHost: "127.0.0.1:3030",
      fetchSite: "same-origin",
      session,
    }),
    {params: Promise.resolve({path: ["api", "bootstrap"]})},
  );
  assert.equal(forwardedHostSpoof.status, 403);
  assert.equal(forwardedHostSpoof.headers.get(LOCAL_SECURITY_REJECTION_HEADER), "invalid_host");
  assert.equal(backendCalls, 0);
});

test("same-origin GET, mutation, cancel, and stream complete the session/CSRF loop", async (t) => {
  const session = await createBrowserSession();
  const backendRequests: Array<{url: string; method: string; headers: Headers}> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    backendRequests.push({
      url: String(input),
      method: init?.method ?? "GET",
      headers: new Headers(init?.headers),
    });
    if (String(input).endsWith("/api/stream")) {
      return new Response("event: run.completed\ndata: {}\n\n", {
        status: 200,
        headers: {"content-type": "text/event-stream; charset=utf-8"},
      });
    }
    return Response.json({ok: true});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const getResponse = await proxyGet(
    browserRequest("/api/teacher-agent/api/bootstrap", {session}),
    {params: Promise.resolve({path: ["api", "bootstrap"]})},
  );
  assert.equal(getResponse.status, 200);
  await getResponse.text();

  const postResponse = await proxyPost(
    browserRequest("/api/teacher-agent/api/start", {method: "POST", session, body: "{}"}),
    {params: Promise.resolve({path: ["api", "start"]})},
  );
  assert.equal(postResponse.status, 200);
  await postResponse.text();

  const adjudicationResponse = await proxyPost(
    browserRequest("/api/teacher-agent/api/adjudication/list", {method: "POST", session, body: "{}"}),
    {params: Promise.resolve({path: ["api", "adjudication", "list"]})},
  );
  assert.equal(adjudicationResponse.status, 200);
  await adjudicationResponse.text();

  for (const action of ["list", "grant", "revoke"]) {
    const consentResponse = await proxyPost(
      browserRequest(`/api/teacher-agent/api/consent/${action}`, {method: "POST", session, body: "{}"}),
      {params: Promise.resolve({path: ["api", "consent", action]})},
    );
    assert.equal(consentResponse.status, 200);
    await consentResponse.text();
  }

  for (const action of ["revisions", "publish", "rollback"]) {
    const syllabusResponse = await proxyPost(
      browserRequest(`/api/teacher-agent/api/syllabi/syl_0123456789abcdef01234567/${action}`, {method: "POST", session, body: "{}"}),
      {params: Promise.resolve({path: ["api", "syllabi", "syl_0123456789abcdef01234567", action]})},
    );
    assert.equal(syllabusResponse.status, 200);
    await syllabusResponse.text();
  }
  const syllabusVersions = await proxyGet(
    browserRequest("/api/teacher-agent/api/syllabi/syl_0123456789abcdef01234567/versions", {session}),
    {params: Promise.resolve({path: ["api", "syllabi", "syl_0123456789abcdef01234567", "versions"]})},
  );
  assert.equal(syllabusVersions.status, 200);
  await syllabusVersions.text();
  const curriculumBlueprint = await proxyGet(
    browserRequest("/api/teacher-agent/api/syllabi/syl_0123456789abcdef01234567/curriculum-blueprint", {session}),
    {params: Promise.resolve({path: ["api", "syllabi", "syl_0123456789abcdef01234567", "curriculum-blueprint"]})},
  );
  assert.equal(curriculumBlueprint.status, 200);
  await curriculumBlueprint.text();
  const curriculumWrite = await proxyPost(
    browserRequest("/api/teacher-agent/api/syllabi/syl_0123456789abcdef01234567/curriculum-blueprint", {method: "POST", session, body: "{}"}),
    {params: Promise.resolve({path: ["api", "syllabi", "syl_0123456789abcdef01234567", "curriculum-blueprint"]})},
  );
  assert.equal(curriculumWrite.status, 404);
  for (const action of ["review", "seal", "revoke"]) {
    const authorityResponse = await proxyPost(
      browserRequest(`/api/teacher-agent/api/curriculum/${action}`, {
        method: "POST",
        session,
        body: JSON.stringify({curriculum_authority_idempotency_key: `curriculum-${action}-0001`}),
      }),
      {params: Promise.resolve({path: ["api", "curriculum", action]})},
    );
    assert.equal(authorityResponse.status, 200);
    await authorityResponse.text();
    const wrongMethod = await proxyGet(
      browserRequest(`/api/teacher-agent/api/curriculum/${action}`, {session}),
      {params: Promise.resolve({path: ["api", "curriculum", action]})},
    );
    assert.equal(wrongMethod.status, 404);
  }
  const unknownCurriculumAction = await proxyPost(
    browserRequest("/api/teacher-agent/api/curriculum/publish", {
      method: "POST",
      session,
      body: "{}",
    }),
    {params: Promise.resolve({path: ["api", "curriculum", "publish"]})},
  );
  assert.equal(unknownCurriculumAction.status, 404);

  const projectBootstrap = await proxyPost(
    browserRequest("/api/teacher-agent/api/projects/bootstrap", {method: "POST", session, body: "{}"}),
    {params: Promise.resolve({path: ["api", "projects", "bootstrap"]})},
  );
  assert.equal(projectBootstrap.status, 200);
  await projectBootstrap.text();
  for (const action of ["browse", "remove-reference"]) {
    const projectResponse = await proxyPost(
      browserRequest(`/api/teacher-agent/api/projects/project_${"1".repeat(24)}/${action}`, {method: "POST", session, body: "{}"}),
      {params: Promise.resolve({path: ["api", "projects", `project_${"1".repeat(24)}`, action]})},
    );
    assert.equal(projectResponse.status, 200);
    await projectResponse.text();
  }
  for (const action of ["list", "predict", "pair"]) {
    const metacognitionResponse = await proxyPost(
      browserRequest(`/api/teacher-agent/api/metacognition/${action}`, {method: "POST", session, body: "{}"}),
      {params: Promise.resolve({path: ["api", "metacognition", action]})},
    );
    assert.equal(metacognitionResponse.status, 200);
    await metacognitionResponse.text();
  }
  const metacognitionGet = await proxyGet(
    browserRequest("/api/teacher-agent/api/metacognition/predict", {session}),
    {params: Promise.resolve({path: ["api", "metacognition", "predict"]})},
  );
  assert.equal(metacognitionGet.status, 404);
  const unknownMetacognitionAction = await proxyPost(
    browserRequest("/api/teacher-agent/api/metacognition/mastery", {method: "POST", session, body: "{}"}),
    {params: Promise.resolve({path: ["api", "metacognition", "mastery"]})},
  );
  assert.equal(unknownMetacognitionAction.status, 404);

  const cancelResponse = await cancelPost(browserRequest("/api/teacher-agent/cancel", {
    method: "POST",
    session,
    body: JSON.stringify({request_id: "request-1"}),
  }));
  assert.equal(cancelResponse.status, 200);
  await cancelResponse.text();

  const streamResponse = await streamPost(browserRequest("/api/teacher-agent/stream", {
    method: "POST",
    session,
    body: JSON.stringify({operation: "chat", payload: {messages: []}}),
  }));
  assert.equal(streamResponse.status, 200);
  assert.match(streamResponse.headers.get("content-type") ?? "", /^text\/event-stream/);
  await streamResponse.text();

  assert.equal(backendRequests.length, 22);
  for (const request of backendRequests) {
    assert.equal(request.headers.has("cookie"), false);
    assert.equal(request.headers.has("origin"), false);
    assert.equal(request.headers.has("x-forwarded-host"), false);
    assert.equal(request.headers.has(LOCAL_CSRF_HEADER_NAME), false);
  }
});

test("project export proxy preserves only the download and manifest receipt headers", async (t) => {
  const session = await createBrowserSession();
  const originalFetch = globalThis.fetch;
  const digest = "a".repeat(64);
  globalThis.fetch = (async () => new Response(new Uint8Array([80, 75, 3, 4]), {
    status: 200,
    headers: {
      "content-type": "application/zip",
      "content-disposition": 'attachment; filename="project_private.zip"',
      "x-manifest-sha256": digest,
      "x-private-backend-header": "must-not-cross-proxy",
    },
  })) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const response = await proxyGet(
    browserRequest("/api/teacher-agent/api/projects/project_111111111111111111111111/export", {session}),
    {params: Promise.resolve({path: ["api", "projects", "project_111111111111111111111111", "export"]})},
  );
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-type"), "application/zip");
  assert.equal(response.headers.get("content-disposition"), 'attachment; filename="project_private.zip"');
  assert.equal(response.headers.get("x-manifest-sha256"), digest);
  assert.equal(response.headers.get("x-private-backend-header"), null);
  assert.deepEqual([...new Uint8Array(await response.arrayBuffer())], [80, 75, 3, 4]);
});

test("localhost, IPv4, and IPv6 loopback each complete an exact-origin mutation", async (t) => {
  let backendCalls = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    backendCalls += 1;
    return Response.json({ok: true});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  for (const authority of ["localhost:3030", "127.0.0.1:3030", "[::1]:3030"]) {
    const session = await createBrowserSession(authority);
    const response = await cancelPost(browserRequest("/api/teacher-agent/cancel", {
      authority,
      method: "POST",
      session,
      body: JSON.stringify({request_id: "request-1"}),
    }));
    assert.equal(response.status, 200, authority);
    await response.text();
  }
  assert.equal(backendCalls, 3);
});

test("unsafe requests require exact Origin, JSON, session, and CSRF", async () => {
  const session = await createBrowserSession();
  const cases = [
    browserRequest("/api/teacher-agent/api/start", {method: "POST", origin: null, session, body: "{}"}),
    browserRequest("/api/teacher-agent/api/start", {method: "POST", session: {...session, csrf: "a".repeat(43)}, body: "{}"}),
    new NextRequest("http://127.0.0.1:3030/api/teacher-agent/api/start", {
      method: "POST",
      headers: {
        host: "127.0.0.1:3030",
        origin: "http://127.0.0.1:3030",
        "sec-fetch-site": "same-origin",
        cookie: session.cookie,
        [LOCAL_CSRF_HEADER_NAME]: session.csrf,
        "content-type": "text/plain",
      },
      body: "{}",
    }),
  ];
  const expected = ["invalid_origin", "invalid_csrf", "invalid_content_type"];
  for (let index = 0; index < cases.length; index += 1) {
    const response = await proxyPost(cases[index], {params: Promise.resolve({path: ["api", "start"]})});
    assert.equal(response.status, index === 2 ? 415 : 403);
    assert.equal(response.headers.get(LOCAL_SECURITY_REJECTION_HEADER), expected[index]);
  }
});

test("global response policy denies framing and exposes no ambient browser capabilities", () => {
  const headers = Object.fromEntries(SECURITY_RESPONSE_HEADERS.map(({key, value}) => [key, value]));
  assert.match(CONTENT_SECURITY_POLICY, /frame-ancestors 'none'/);
  assert.match(CONTENT_SECURITY_POLICY, /object-src 'none'/);
  assert.equal(headers["Cross-Origin-Opener-Policy"], "same-origin");
  assert.equal(headers["Cross-Origin-Resource-Policy"], "same-origin");
  assert.equal(headers["Referrer-Policy"], "no-referrer");
  assert.equal(headers["X-Content-Type-Options"], "nosniff");
  assert.equal(headers["X-Frame-Options"], "DENY");
  assert.match(headers["Permissions-Policy"], /camera=\(\)/);
  assert.match(headers["Permissions-Policy"], /microphone=\(\)/);
});

test("session cookie uses the expected private cookie name", () => {
  assert.equal(LOCAL_SECURITY_COOKIE_NAME, "teachlab_local_session");
});
