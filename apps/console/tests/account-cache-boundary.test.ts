import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

import {
  ACTIVE_PROJECT_STORAGE_KEY,
  bootstrapAllowsAccountCacheAccess,
  CHAT_HISTORY_STORAGE_KEY,
  clearAccountBoundBrowserStorage,
  LEGACY_MIGRATION_STORAGE_KEY,
  SESSION_INDEX_STORAGE_KEY,
  SESSION_STORAGE_KEY,
  START_KEY_STORAGE_KEY,
  workbenchIdentityBoundary,
} from "../lib/account-cache-boundary.ts";

class MemoryStorage {
  readonly values = new Map<string, string>();

  removeItem(key: string) {
    this.values.delete(key);
  }
}

test("authentication-required is an unconditional login boundary", () => {
  assert.equal(workbenchIdentityBoundary({
    signedOut: false,
    authenticationRequired: true,
    bootstrapPending: false,
  }), "login_required");
  assert.equal(workbenchIdentityBoundary({
    signedOut: true,
    authenticationRequired: false,
    bootstrapPending: false,
  }), "login_required");
});

test("the first bootstrap stays behind an identity checking screen", () => {
  assert.equal(workbenchIdentityBoundary({
    signedOut: false,
    authenticationRequired: false,
    bootstrapPending: true,
  }), "checking");
  assert.equal(workbenchIdentityBoundary({
    signedOut: false,
    authenticationRequired: false,
    bootstrapPending: true,
  }), "checking");
  assert.equal(workbenchIdentityBoundary({
    signedOut: false,
    authenticationRequired: false,
    bootstrapPending: false,
  }), "workspace");
});

test("account boundary clears project, chat, and session pointers but preserves device preferences", () => {
  const session = new MemoryStorage();
  const local = new MemoryStorage();
  for (const key of [SESSION_STORAGE_KEY, SESSION_INDEX_STORAGE_KEY, START_KEY_STORAGE_KEY, CHAT_HISTORY_STORAGE_KEY]) {
    session.values.set(key, "old-account-secret");
  }
  for (const key of [ACTIVE_PROJECT_STORAGE_KEY, LEGACY_MIGRATION_STORAGE_KEY]) {
    local.values.set(key, "old-account-secret");
  }
  session.values.set("device-preference", "keep");
  local.values.set("teachlab.console.preferences", "keep");

  assert.equal(clearAccountBoundBrowserStorage(session, local), true);
  assert.deepEqual([...session.values], [["device-preference", "keep"]]);
  assert.deepEqual([...local.values], [["teachlab.console.preferences", "keep"]]);
});

test("a storage denial is reported while all remaining keys are still attempted", () => {
  const removed: string[] = [];
  const rejectingSession = {
    removeItem(key: string) {
      removed.push(key);
      if (key === SESSION_STORAGE_KEY) throw new Error("denied");
    },
  };
  const local = {removeItem(key: string) { removed.push(key); }};

  assert.equal(clearAccountBoundBrowserStorage(rejectingSession, local), false);
  for (const key of [
    SESSION_STORAGE_KEY,
    SESSION_INDEX_STORAGE_KEY,
    START_KEY_STORAGE_KEY,
    CHAT_HISTORY_STORAGE_KEY,
    ACTIVE_PROJECT_STORAGE_KEY,
    LEGACY_MIGRATION_STORAGE_KEY,
  ]) assert.ok(removed.includes(key), key);
});

test("an old account outbox is never read or replayed before bootstrap authentication", async () => {
  let accountStateReads = 0;
  const denied = await bootstrapAllowsAccountCacheAccess(async () => {
    throw new Error("authentication_required");
  });
  if (denied) accountStateReads += 1;
  assert.equal(denied, false);
  assert.equal(accountStateReads, 0);

  const allowed = await bootstrapAllowsAccountCacheAccess(async () => ({authenticated: true}));
  if (allowed) accountStateReads += 1;
  assert.equal(allowed, true);
  assert.equal(accountStateReads, 1);
});

test("Workbench gates cache hydration and cleanup on the server authentication result", () => {
  const source = readFileSync(
    new URL("../components/workbench/workbench.tsx", import.meta.url),
    "utf8",
  );
  assert.ok(source.includes("const loginEntryAvailable = authenticationRequired || organizationLoginAvailable();"));
  assert.ok(source.includes('if (identityBoundary === "checking")'));
  assert.ok(source.includes("if (!authenticatedBootstrapReady) return;\n    const loaded = parseChatThreads"));
  assert.ok(source.includes("if (signedOut || authenticationRequired || initialIdentityCheckPending"));
  assert.ok(source.includes("void boundedLogoutCleanup(clearOfflineRuntimeForLogout)"));
  assert.equal(source.includes("organizationLoginAvailable() && bootstrap.isError"), false);
  const providers = readFileSync(new URL("../app/providers.tsx", import.meta.url), "utf8");
  assert.ok(
    providers.indexOf("bootstrapAllowsAccountCacheAccess(fetchBootstrap)")
      < providers.indexOf("listDurableOperations()"),
    "Providers must authenticate before reading the account outbox",
  );
});
