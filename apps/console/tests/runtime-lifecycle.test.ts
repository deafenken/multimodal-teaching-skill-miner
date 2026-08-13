import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {NextRequest} from "next/server.js";

import {GET as healthGet} from "../app/health/route.ts";
import {GET as readyGet} from "../app/ready/route.ts";
import {POST as runtimeStopPost} from "../app/api/teacher-agent/runtime/stop/route.ts";
import {GET as securitySessionGet} from "../app/api/teacher-agent/security/session/route.ts";

process.env.TEACHLAB_HARNESS_MODE = "local_python";

test("the global runtime-stop control cannot cover the conversation composer", () => {
  const page = fs.readFileSync(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(page, /RuntimeStopButton/);
});

test("health is liveness-only and ready fails closed without backend", async () => {
  const previous = process.env.TEACHER_AGENT_CAPABILITY_URL;
  delete process.env.TEACHER_AGENT_CAPABILITY_URL;
  try {
    const health = await healthGet();
    assert.equal(health.status, 200);
    const healthPayload = await health.json();
    assert.equal(healthPayload.status, "healthy");
    assert.equal(typeof healthPayload.release_id, "string");
    const ready = await readyGet();
    assert.equal(ready.status, 503);
    assert.equal((await ready.json()).status, "not_ready");
  } finally {
    if (previous === undefined) delete process.env.TEACHER_AGENT_CAPABILITY_URL;
    else process.env.TEACHER_AGENT_CAPABILITY_URL = previous;
  }
});

test("ready requires a 200 JSON backend response and never promotes 503", async (t) => {
  const previousUrl = process.env.TEACHER_AGENT_CAPABILITY_URL;
  const originalFetch = globalThis.fetch;
  process.env.TEACHER_AGENT_CAPABILITY_URL = "http://127.0.0.1:49199/capability/";
  t.after(() => {
    globalThis.fetch = originalFetch;
    if (previousUrl === undefined) delete process.env.TEACHER_AGENT_CAPABILITY_URL;
    else process.env.TEACHER_AGENT_CAPABILITY_URL = previousUrl;
  });
  globalThis.fetch = (async () => Response.json({error: "warming"}, {status: 503})) as typeof fetch;
  assert.equal((await readyGet()).status, 503);
  globalThis.fetch = (async () => Response.json({
    schema_version: "1.1",
    dashboard_kind: "loopback_interactive_teacher_agent",
    provider_status: {configured: true},
  })) as typeof fetch;
  const ready = await readyGet();
  assert.equal(ready.status, 200);
  assert.equal((await ready.json()).status, "ready");
});

test("runtime stop control requires local session+CSRF and deduplicates a private request", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "teachlab-runtime-stop-"));
  const previous = process.env.TEACHLAB_RUNTIME_STOP_REQUEST;
  try {
    const target = path.join(directory, "stop.request");
    process.env.TEACHLAB_RUNTIME_STOP_REQUEST = target;
    const authority = "127.0.0.1:3030";
    const baseHeaders = {host: authority, "sec-fetch-site": "same-origin"};
    const handshake = await securitySessionGet(new NextRequest(
      `http://${authority}/api/teacher-agent/security/session`,
      {headers: baseHeaders},
    ));
    const cookie = handshake.headers.get("set-cookie")?.split(";", 1)[0];
    const csrf = String((await handshake.json() as {csrf_token: unknown}).csrf_token);
    assert.ok(cookie);
    const hostile = await runtimeStopPost(new NextRequest(
      `http://${authority}/api/teacher-agent/runtime/stop`,
      {method: "POST", headers: {host: authority, "content-type": "application/json"}, body: "{}"},
    ));
    assert.equal(hostile.status, 403);
    assert.equal(fs.existsSync(target), false);
    const request = () => new NextRequest(
      `http://${authority}/api/teacher-agent/runtime/stop`,
      {
        method: "POST",
        headers: {
          ...baseHeaders,
          origin: `http://${authority}`,
          "content-type": "application/json",
          cookie,
          "x-teachlab-csrf-token": csrf,
        },
        body: "{}",
      },
    );
    assert.equal((await runtimeStopPost(request())).status, 202);
    assert.equal(fs.lstatSync(target).mode & 0o077, 0);
    assert.equal((await runtimeStopPost(request())).status, 202);
  } finally {
    if (previous === undefined) delete process.env.TEACHLAB_RUNTIME_STOP_REQUEST;
    else process.env.TEACHLAB_RUNTIME_STOP_REQUEST = previous;
    fs.rmSync(directory, {recursive: true, force: true});
  }
});
