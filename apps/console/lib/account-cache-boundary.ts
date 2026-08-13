export const SESSION_STORAGE_KEY = "teachlab.teacher-agent.session-id";
export const SESSION_INDEX_STORAGE_KEY = "teachlab.teacher-agent.session-index";
export const START_KEY_STORAGE_KEY = "teachlab.teacher-agent.start-idempotency-key";
export const CHAT_HISTORY_STORAGE_KEY = "teachlab.chat.threads";
export const ACTIVE_PROJECT_STORAGE_KEY = "teachlab.learning-project.active-id";
export const LEGACY_MIGRATION_STORAGE_KEY = "teachlab.learning-project.legacy-migration-id";

const ACCOUNT_BOUND_SESSION_KEYS = [
  SESSION_STORAGE_KEY,
  SESSION_INDEX_STORAGE_KEY,
  START_KEY_STORAGE_KEY,
  CHAT_HISTORY_STORAGE_KEY,
] as const;

const ACCOUNT_BOUND_LOCAL_KEYS = [
  ACTIVE_PROJECT_STORAGE_KEY,
  LEGACY_MIGRATION_STORAGE_KEY,
] as const;

type StorageRemoval = Pick<Storage, "removeItem">;

/**
 * Remove only browser pointers and legacy payloads that can identify or expose
 * one account's projects, sessions, and chats. Device-wide presentation
 * preferences intentionally survive an identity boundary.
 */
export function clearAccountBoundBrowserStorage(
  sessionStorage: StorageRemoval,
  localStorage: StorageRemoval,
): boolean {
  let cleared = true;
  for (const key of ACCOUNT_BOUND_SESSION_KEYS) {
    try {
      sessionStorage.removeItem(key);
    } catch {
      cleared = false;
    }
  }
  for (const key of ACCOUNT_BOUND_LOCAL_KEYS) {
    try {
      localStorage.removeItem(key);
    } catch {
      cleared = false;
    }
  }
  return cleared;
}

export type WorkbenchIdentityBoundary = "login_required" | "checking" | "workspace";

/** A first bootstrap response is the authority to expose account-bound UI. */
export function workbenchIdentityBoundary(input: {
  signedOut: boolean;
  authenticationRequired: boolean;
  bootstrapPending: boolean;
}): WorkbenchIdentityBoundary {
  if (input.signedOut || input.authenticationRequired) return "login_required";
  if (input.bootstrapPending) return "checking";
  return "workspace";
}

/**
 * IndexedDB outbox/workspace state is account-bound. A server bootstrap must
 * succeed before any caller is allowed to read or replay it, including during
 * the Providers online-recovery hook that runs before Workbench mounts.
 */
export async function bootstrapAllowsAccountCacheAccess(
  probe: () => Promise<unknown>,
): Promise<boolean> {
  try {
    await probe();
    return true;
  } catch {
    return false;
  }
}
