export const ACCOUNT_CACHE_SCOPE_STORAGE_KEY = "teachlab.account-cache-scope.v1";
export const ACCOUNT_CACHE_SCOPE_PATTERN = /^acs1_[A-Za-z0-9_-]{43}$/;
export const LOCAL_CACHE_BOUNDARY_MARKER = "local_only_no_account_authority_v1";
export const ACCOUNT_TRANSITION_COOKIE_NAMES = [
  "teachlab_account_transition",
  "__Host-teachlab_account_transition",
] as const;

export function accountTransitionPending(cookieHeader: string): boolean {
  const names = new Set(
    cookieHeader.split(";").map((part) => part.trim().split("=", 1)[0] ?? ""),
  );
  return ACCOUNT_TRANSITION_COOKIE_NAMES.some((name) => names.has(name));
}

/**
 * Remove the non-secret browser-readable fence only after authenticated
 * bootstrap reconciliation has either retained or purged the correct scope.
 */
export function clearAccountTransitionFence(input: {
  writeCookie(value: string): void;
  readCookies(): string;
}): boolean {
  try {
    input.writeCookie(
      "teachlab_account_transition=; Path=/; SameSite=Strict; Max-Age=0",
    );
    input.writeCookie(
      "__Host-teachlab_account_transition=; Path=/; SameSite=Strict; Max-Age=0; Secure",
    );
    return !accountTransitionPending(input.readCookies());
  } catch {
    return false;
  }
}

export interface BrowserStoragePort {
  readonly length: number;
  key(index: number): string | null;
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface BrowserCacheStoragePort {
  keys(): Promise<string[]>;
  delete(name: string): Promise<boolean>;
}

export interface AccountBrowserPurgePorts {
  sessionStorage: BrowserStoragePort;
  localStorage: BrowserStoragePort;
  clearIndexedDb(): Promise<boolean>;
  detachActiveRuns?(): void | Promise<void>;
  clearMemoryCaches?(): void | Promise<void>;
  cacheStorage?: BrowserCacheStoragePort;
}

export interface AccountBrowserPurgeResult {
  complete: boolean;
  failures: readonly string[];
}

export type AccountCacheReconciliation =
  | {allowed: true; changed: boolean}
  | {allowed: false; reason: "authentication_required" | "invalid_cache_scope" | "cleanup_failed"};

/**
 * Standalone loopback mode has no organization principal from which an
 * account cache namespace can be minted.  It still needs an explicit browser
 * boundary before reading IndexedDB: a browser origin may previously have
 * been used by authenticated_apps_api, and replaying that account's outbox
 * against the local backend would cross the identity boundary.
 *
 * The marker is deliberately not an acs1_ account scope.  Account-bound
 * offline snapshots therefore remain unavailable in local mode, while a
 * later authenticated bootstrap treats this marker as a scope transition and
 * purges local state before installing its issuer-bound namespace.
 */
export async function reconcileLocalCacheBoundary(input: {
  markerStorage: BrowserStoragePort;
  purge(): Promise<AccountBrowserPurgeResult>;
}): Promise<AccountCacheReconciliation> {
  let current: string | null;
  try {
    current = input.markerStorage.getItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY);
  } catch {
    await input.purge();
    return {allowed: false, reason: "cleanup_failed"};
  }
  if (current === LOCAL_CACHE_BOUNDARY_MARKER) {
    return {allowed: true, changed: false};
  }

  const purge = await input.purge();
  if (!purge.complete) {
    try {
      input.markerStorage.removeItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY);
    } catch {
      // The boundary remains fail-closed even when storage is unavailable.
    }
    return {allowed: false, reason: "cleanup_failed"};
  }
  try {
    input.markerStorage.setItem(
      ACCOUNT_CACHE_SCOPE_STORAGE_KEY,
      LOCAL_CACHE_BOUNDARY_MARKER,
    );
  } catch {
    await input.purge();
    return {allowed: false, reason: "cleanup_failed"};
  }
  return {allowed: true, changed: true};
}

function constantTimeEqual(left: string, right: string): boolean {
  const maximum = Math.max(left.length, right.length);
  let mismatch = left.length ^ right.length;
  for (let index = 0; index < maximum; index += 1) {
    mismatch |= (left.charCodeAt(index) || 0) ^ (right.charCodeAt(index) || 0);
  }
  return mismatch === 0;
}

function storageKeys(storage: BrowserStoragePort): string[] {
  const keys: string[] = [];
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index);
    if (key !== null) keys.push(key);
  }
  return keys;
}

function removeTeachLabState(storage: BrowserStoragePort): boolean {
  let complete = true;
  let keys: string[];
  try {
    keys = storageKeys(storage);
  } catch {
    return false;
  }
  for (const key of keys) {
    if (!key.startsWith("teachlab.")) continue;
    try {
      storage.removeItem(key);
    } catch {
      complete = false;
    }
  }
  return complete;
}

/**
 * Account deletion cleanup is deliberately broader than logout cleanup: every
 * TeachLab browser key, every offline object store, and every origin Cache API
 * entry is removed. All steps are attempted even if one browser API fails.
 */
export async function purgeAccountBrowserState(
  ports: AccountBrowserPurgePorts
): Promise<AccountBrowserPurgeResult> {
  const failures: string[] = [];
  try {
    await ports.detachActiveRuns?.();
  } catch {
    failures.push("detach_active_runs");
  }
  try {
    await ports.clearMemoryCaches?.();
  } catch {
    failures.push("clear_memory_caches");
  }
  try {
    if (!await ports.clearIndexedDb()) failures.push("clear_indexeddb");
  } catch {
    failures.push("clear_indexeddb");
  }
  if (!removeTeachLabState(ports.sessionStorage)) failures.push("clear_session_storage");
  if (!removeTeachLabState(ports.localStorage)) failures.push("clear_local_storage");
  if (ports.cacheStorage) {
    try {
      const names = await ports.cacheStorage.keys();
      const deleted = await Promise.allSettled(names.map((name) =>
        ports.cacheStorage!.delete(name)
      ));
      if (deleted.some((result) =>
        result.status === "rejected" || result.value !== true
      )) failures.push("clear_cache_storage");
    } catch {
      failures.push("clear_cache_storage");
    }
  }
  return {complete: failures.length === 0, failures};
}

/**
 * Reconcile an authenticated bootstrap before *any* IndexedDB read/replay.
 * Missing legacy markers are treated as an identity transition and purged.
 */
export async function reconcileAccountCacheScope(input: {
  authenticated: boolean;
  cacheScope?: string;
  markerStorage: BrowserStoragePort;
  purge(): Promise<AccountBrowserPurgeResult>;
}): Promise<AccountCacheReconciliation> {
  const supplied = input.cacheScope ?? "";
  if (!input.authenticated || !ACCOUNT_CACHE_SCOPE_PATTERN.test(supplied)) {
    const purge = await input.purge();
    try {
      input.markerStorage.removeItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY);
    } catch {
      return {allowed: false, reason: "cleanup_failed"};
    }
    return purge.complete
      ? {
          allowed: false,
          reason: input.authenticated ? "invalid_cache_scope" : "authentication_required"
        }
      : {allowed: false, reason: "cleanup_failed"};
  }

  let current: string | null;
  try {
    current = input.markerStorage.getItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY);
  } catch {
    await input.purge();
    return {allowed: false, reason: "cleanup_failed"};
  }
  if (current && ACCOUNT_CACHE_SCOPE_PATTERN.test(current)
    && constantTimeEqual(current, supplied)) {
    return {allowed: true, changed: false};
  }
  const purge = await input.purge();
  if (!purge.complete) return {allowed: false, reason: "cleanup_failed"};
  try {
    input.markerStorage.setItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY, supplied);
  } catch {
    await input.purge();
    return {allowed: false, reason: "cleanup_failed"};
  }
  return {allowed: true, changed: true};
}
