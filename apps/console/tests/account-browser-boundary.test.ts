import assert from "node:assert/strict";
import test from "node:test";

import {
  ACCOUNT_CACHE_SCOPE_STORAGE_KEY,
  ACCOUNT_CACHE_SCOPE_PATTERN,
  LOCAL_CACHE_BOUNDARY_MARKER,
  accountTransitionPending,
  clearAccountTransitionFence,
  purgeAccountBrowserState,
  reconcileAccountCacheScope,
  reconcileLocalCacheBoundary,
  type BrowserStoragePort
} from "../lib/account-browser-boundary.ts";

const SCOPE_A = `acs1_${"a".repeat(43)}`;
const SCOPE_B = `acs1_${"b".repeat(43)}`;

class MemoryStorage implements BrowserStoragePort {
  readonly values = new Map<string, string>();
  get length() { return this.values.size; }
  key(index: number) { return [...this.values.keys()][index] ?? null; }
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
  removeItem(key: string) { this.values.delete(key); }
}

test("account transition fence remains until authenticated cache reconciliation clears it", () => {
  let cookies = "analytics=keep; teachlab_account_transition=pending_v1";
  const writes: string[] = [];
  assert.equal(accountTransitionPending(cookies), true);
  const cleared = clearAccountTransitionFence({
    writeCookie: (value) => {
      writes.push(value);
      const name = value.split("=", 1)[0] ?? "";
      cookies = cookies.split(";").map((part) => part.trim())
        .filter((part) => !part.startsWith(`${name}=`)).join("; ");
    },
    readCookies: () => cookies,
  });
  assert.equal(cleared, true);
  assert.equal(accountTransitionPending(cookies), false);
  assert.equal(writes.length, 2);
  assert.match(writes[1] ?? "", /^__Host-teachlab_account_transition=.*Secure$/);
});

test("A expires without logout then B must purge before any account cache read", async () => {
  const marker = new MemoryStorage();
  marker.setItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY, SCOPE_A);
  const privateState = ["A-draft", "A-outbox", "A-workspace"];
  let accountReads = 0;
  const result = await reconcileAccountCacheScope({
    authenticated: true,
    cacheScope: SCOPE_B,
    markerStorage: marker,
    purge: async () => {
      privateState.splice(0);
      return {complete: true, failures: []};
    }
  });
  if (result.allowed) accountReads += privateState.length;
  assert.deepEqual(result, {allowed: true, changed: true});
  assert.equal(accountReads, 0);
  assert.equal(marker.getItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY), SCOPE_B);
});

test("missing legacy marker is purged and cleanup failure fails closed", async () => {
  const marker = new MemoryStorage();
  let reads = 0;
  const failed = await reconcileAccountCacheScope({
    authenticated: true,
    cacheScope: SCOPE_A,
    markerStorage: marker,
    purge: async () => ({complete: false, failures: ["clear_indexeddb"]})
  });
  if (failed.allowed) reads += 1;
  assert.deepEqual(failed, {allowed: false, reason: "cleanup_failed"});
  assert.equal(reads, 0);
  assert.equal(marker.getItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY), null);
});

test("unauthenticated bootstrap purges and can never authorize hydration", async () => {
  const marker = new MemoryStorage();
  marker.setItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY, SCOPE_A);
  let purges = 0;
  const result = await reconcileAccountCacheScope({
    authenticated: false,
    markerStorage: marker,
    purge: async () => {
      purges += 1;
      return {complete: true, failures: []};
    }
  });
  assert.deepEqual(result, {allowed: false, reason: "authentication_required"});
  assert.equal(purges, 1);
  assert.equal(marker.getItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY), null);
});

test("local mode purges an authenticated account scope before authorizing local cache reads", async () => {
  const marker = new MemoryStorage();
  marker.setItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY, SCOPE_A);
  const privateState = ["account-A-outbox"];
  let purges = 0;
  const first = await reconcileLocalCacheBoundary({
    markerStorage: marker,
    purge: async () => {
      purges += 1;
      privateState.splice(0);
      return {complete: true, failures: []};
    },
  });
  assert.deepEqual(first, {allowed: true, changed: true});
  assert.equal(privateState.length, 0);
  assert.equal(marker.getItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY), LOCAL_CACHE_BOUNDARY_MARKER);

  const repeated = await reconcileLocalCacheBoundary({
    markerStorage: marker,
    purge: async () => {
      purges += 1;
      return {complete: true, failures: []};
    },
  });
  assert.deepEqual(repeated, {allowed: true, changed: false});
  assert.equal(purges, 1);
});

test("local boundary cleanup failure stays blocked and cannot mint an account scope", async () => {
  const marker = new MemoryStorage();
  marker.setItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY, SCOPE_A);
  const result = await reconcileLocalCacheBoundary({
    markerStorage: marker,
    purge: async () => ({complete: false, failures: ["clear_indexeddb"]}),
  });
  assert.deepEqual(result, {allowed: false, reason: "cleanup_failed"});
  assert.equal(marker.getItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY), null);
  assert.equal(ACCOUNT_CACHE_SCOPE_PATTERN.test(LOCAL_CACHE_BOUNDARY_MARKER), false);
});

test("authenticated mode purges a prior local marker before installing its issuer-bound scope", async () => {
  const marker = new MemoryStorage();
  marker.setItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY, LOCAL_CACHE_BOUNDARY_MARKER);
  let purges = 0;
  const result = await reconcileAccountCacheScope({
    authenticated: true,
    cacheScope: SCOPE_B,
    markerStorage: marker,
    purge: async () => {
      purges += 1;
      return {complete: true, failures: []};
    },
  });
  assert.deepEqual(result, {allowed: true, changed: true});
  assert.equal(purges, 1);
  assert.equal(marker.getItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY), SCOPE_B);
});

test("committed account cleanup clears IndexedDB, TeachLab storage and origin caches", async () => {
  const session = new MemoryStorage();
  const local = new MemoryStorage();
  session.setItem("teachlab.chat.threads", "SECRET_SOURCE_ANSWER");
  session.setItem("unrelated", "keep");
  local.setItem("teachlab.console.preferences", "private");
  local.setItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY, SCOPE_A);
  local.setItem("other-app", "keep");
  const cacheNames = new Set(["next-data", "teachlab-assets"]);
  const calls: string[] = [];
  const result = await purgeAccountBrowserState({
    sessionStorage: session,
    localStorage: local,
    detachActiveRuns: () => { calls.push("detach"); },
    clearMemoryCaches: () => { calls.push("memory"); },
    clearIndexedDb: async () => { calls.push("indexeddb"); return true; },
    cacheStorage: {
      keys: async () => [...cacheNames],
      delete: async (name) => cacheNames.delete(name)
    }
  });
  assert.equal(result.complete, true);
  assert.deepEqual(calls, ["detach", "memory", "indexeddb"]);
  assert.deepEqual([...session.values], [["unrelated", "keep"]]);
  assert.deepEqual([...local.values], [["other-app", "keep"]]);
  assert.equal(cacheNames.size, 0);
});
