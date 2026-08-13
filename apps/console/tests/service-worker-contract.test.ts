import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const worker = readFileSync(new URL("../public/teachlab-sw-v1.js", import.meta.url), "utf8");
const shell = readFileSync(new URL("../public/teachlab-offline-shell-v1.js", import.meta.url), "utf8");
const registration = readFileSync(new URL("../components/service-worker-lifecycle.tsx", import.meta.url), "utf8");
const config = readFileSync(new URL("../next.config.ts", import.meta.url), "utf8");

test("service worker caches only versioned same-origin immutable static assets", () => {
  assert.match(worker, /teachlab-console-static-/);
  assert.match(worker, /url\.pathname\.startsWith\("\/_next\/static\/"\)/);
  assert.match(worker, /\bimmutable\b/);
  assert.match(worker, /responseIsImmutable\(response\)/);
  assert.doesNotMatch(worker, /caches\.open\([^)]*\)[\s\S]{0,300}put\([^,]*(?:api|bootstrap|projects)/i);
  assert.doesNotMatch(worker, /cache\.addAll\(\[?\s*["']\//);
});

test("navigation and private responses are never cached", () => {
  assert.match(worker, /request\.mode === "navigate"/);
  assert.match(worker, /fetch\(request\)\.catch\(\(\) => offlineDocument\(\)\)/);
  assert.match(worker, /"Cache-Control": "no-store"/);
  assert.doesNotMatch(worker, /cache\.put\(request,\s*offlineDocument/);
  assert.doesNotMatch(worker, /pathname\.startsWith\("\/api/);
});

test("offline shell fails closed unless snapshot and current account scope match", () => {
  assert.match(shell, /teachlab\.account-cache-scope\.v1/);
  assert.match(shell, /teachlab\.offline_workspace_snapshot\.v2/);
  assert.match(shell, /constantTimeEqual/);
  assert.match(shell, /value\.expiresAt <= Date\.now\(\)/);
  assert.match(shell, /textContent =/);
  assert.doesNotMatch(shell, /innerHTML|insertAdjacentHTML|document\.write/);
});

test("A-to-B login transition blocks the offline shell before any prior-account storage read", async () => {
  let storageReads = 0;
  const elements = new Map<string, Record<string, unknown>>([
    ["offline-status", {textContent: ""}],
    ["offline-project", {hidden: false}],
    ["offline-retry", {addEventListener() {}}],
    ["offline-main", {focus() {}}],
  ]);
  vm.runInNewContext(shell, {
    document: {
      cookie: "teachlab_account_transition=pending_v1",
      getElementById: (id: string) => elements.get(id) ?? null,
    },
    localStorage: {
      getItem() {
        storageReads += 1;
        return "A PRIVATE PROJECT";
      },
    },
    indexedDB: {open() { throw new Error("must not read IndexedDB"); }},
    location: {reload() {}},
    Date,
    Promise,
    Set,
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(storageReads, 0);
  assert.equal(elements.get("offline-project")?.hidden, true);
  assert.match(String(elements.get("offline-status")?.textContent), /账号切换尚未完成/);
});

test("production registration bypasses HTTP cache and immutable shell asset is versioned", () => {
  assert.match(registration, /process\.env\.NODE_ENV !== "production"/);
  assert.match(registration, /updateViaCache: "none"/);
  assert.match(registration, /PREWARM_IMMUTABLE/);
  assert.match(config, /teachlab-offline-shell-v1\.js/);
  assert.match(config, /max-age=31536000, immutable/);
  assert.match(config, /teachlab-sw-v1\.js/);
  assert.match(config, /no-cache, no-store, must-revalidate/);
});
