import type {
  AccountBrowserPurgePorts,
  AccountBrowserPurgeResult
} from "./account-browser-boundary.ts";
import {purgeAccountBrowserState} from "./account-browser-boundary.ts";

export const ACCOUNT_DELETION_CONFIRMATION_PHRASE =
  "PERMANENTLY DELETE MY TEACHLAB ACCOUNT";
export const ACCOUNT_DELETION_RECOVERY_STORAGE_KEY =
  "teachlab.account-deletion-recovery.v1";

export const ACCOUNT_DELETION_PHASES = [
  "prepared", "fencing", "draining", "tombstoned", "quarantining",
  "purging_worker_data", "committing_database", "completed"
] as const;
export type AccountDeletionPhase = (typeof ACCOUNT_DELETION_PHASES)[number];

const SHA256 = /^[0-9a-f]{64}$/;
const RECEIPT_ID = /^adelr_[0-9a-f]{32}$/;

export interface AccountDeletionChallenge {
  schema: "teachlab.account_deletion_challenge.v1";
  challenge_id: string;
  confirmation_token: string;
  confirmation_phrase: typeof ACCOUNT_DELETION_CONFIRMATION_PHRASE;
  expires_at: string;
  revision: number;
}

export interface AccountDeletionReceipt {
  schema: "teachlab.account_deletion_receipt.v1";
  status: "permanently_deleted";
  receipt_id: string;
  scope_sha256: string;
  operation_id_sha256: string;
  deleted_at: string;
  deleted_counts: {
    postgres_events: number;
    postgres_tasks: number;
    postgres_artifacts: number;
    postgres_sessions: number;
    postgres_auth_sessions: number;
    worker_files: number;
    worker_bytes: number;
    worker_roots: number;
  };
  all_devices_session_authority_deleted: true;
  scope_tombstone_retained: true;
  user_managed_export_copies_deleted: false;
  user_managed_export_copies_status: "outside_service_control";
  remote_provider_copies_deleted: false;
  remote_provider_copies_status:
    "outside_service_control_subject_to_provider_retention";
  identity_provider_account_deleted: false;
  identity_provider_account_status:
    "outside_service_control_contact_organization_idp";
  operator_backup_copies_deleted: false;
  operator_backup_copies_status:
    "pending_retention_expiry_or_operator_crypto_erasure";
  receipt_sha256: string;
}

export interface AccountDeletionStatus {
  schema: "teachlab.account_deletion_status.v1";
  status: "awaiting_confirmation" | "deleting" | "retryable_failure" | "permanently_deleted";
  phase: AccountDeletionPhase;
  revision: number;
  retryable_failure_code: string | null;
  receipt: AccountDeletionReceipt | null;
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function exactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  return Object.keys(value).sort().join("\0") === [...expected].sort().join("\0");
}

function timestamp(value: unknown): value is string {
  return typeof value === "string" && value.length <= 64
    && Number.isFinite(Date.parse(value));
}

function count(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}

export function parseAccountDeletionChallenge(value: unknown): AccountDeletionChallenge {
  const row = record(value);
  if (
    !row
    || !exactKeys(row, [
      "schema", "challenge_id", "confirmation_token", "confirmation_phrase",
      "expires_at", "revision"
    ])
    || row.schema !== "teachlab.account_deletion_challenge.v1"
    || typeof row.challenge_id !== "string"
    || !/^adelc_[0-9a-f]{32}$/.test(row.challenge_id)
    || typeof row.confirmation_token !== "string"
    || !/^[A-Za-z0-9_-]{43}$/.test(row.confirmation_token)
    || row.confirmation_phrase !== ACCOUNT_DELETION_CONFIRMATION_PHRASE
    || !timestamp(row.expires_at)
    || !Number.isSafeInteger(row.revision)
    || Number(row.revision) < 1
  ) throw new Error("Invalid account deletion challenge");
  return row as unknown as AccountDeletionChallenge;
}

export function parseAccountDeletionReceipt(value: unknown): AccountDeletionReceipt {
  const row = record(value);
  const counts = record(row?.deleted_counts);
  if (
    !row
    || !exactKeys(row, [
      "schema", "status", "receipt_id", "scope_sha256", "operation_id_sha256",
      "deleted_at", "deleted_counts", "all_devices_session_authority_deleted",
      "scope_tombstone_retained", "user_managed_export_copies_deleted",
      "user_managed_export_copies_status", "remote_provider_copies_deleted",
      "remote_provider_copies_status", "identity_provider_account_deleted",
      "identity_provider_account_status", "operator_backup_copies_deleted",
      "operator_backup_copies_status", "receipt_sha256"
    ])
    || row.schema !== "teachlab.account_deletion_receipt.v1"
    || row.status !== "permanently_deleted"
    || typeof row.receipt_id !== "string"
    || !RECEIPT_ID.test(row.receipt_id)
    || typeof row.scope_sha256 !== "string" || !SHA256.test(row.scope_sha256)
    || typeof row.operation_id_sha256 !== "string" || !SHA256.test(row.operation_id_sha256)
    || !timestamp(row.deleted_at)
    || !counts
    || !exactKeys(counts, [
      "postgres_events", "postgres_tasks", "postgres_sessions",
      "postgres_artifacts",
      "postgres_auth_sessions", "worker_files", "worker_bytes", "worker_roots"
    ])
    || ![
      "postgres_events", "postgres_tasks", "postgres_sessions",
      "postgres_artifacts",
      "postgres_auth_sessions", "worker_files", "worker_bytes", "worker_roots"
    ].every((name) => count(counts[name]))
    || row.all_devices_session_authority_deleted !== true
    || row.scope_tombstone_retained !== true
    || row.user_managed_export_copies_deleted !== false
    || row.user_managed_export_copies_status !== "outside_service_control"
    || row.remote_provider_copies_deleted !== false
    || row.remote_provider_copies_status
      !== "outside_service_control_subject_to_provider_retention"
    || row.identity_provider_account_deleted !== false
    || row.identity_provider_account_status
      !== "outside_service_control_contact_organization_idp"
    || row.operator_backup_copies_deleted !== false
    || row.operator_backup_copies_status
      !== "pending_retention_expiry_or_operator_crypto_erasure"
    || typeof row.receipt_sha256 !== "string" || !SHA256.test(row.receipt_sha256)
  ) throw new Error("Invalid account deletion receipt");
  return row as unknown as AccountDeletionReceipt;
}

export function parseAccountDeletionStatus(value: unknown): AccountDeletionStatus {
  const row = record(value);
  if (
    !row
    || !exactKeys(row, [
      "schema", "status", "phase", "revision", "retryable_failure_code", "receipt"
    ])
    || row.schema !== "teachlab.account_deletion_status.v1"
    || !new Set([
      "awaiting_confirmation", "deleting", "retryable_failure", "permanently_deleted"
    ]).has(String(row.status))
    || !new Set<string>(ACCOUNT_DELETION_PHASES).has(String(row.phase))
    || !Number.isSafeInteger(row.revision) || Number(row.revision) < 1
    || !(row.retryable_failure_code === null
      || typeof row.retryable_failure_code === "string"
        && /^[a-z][a-z0-9_]{2,63}$/.test(row.retryable_failure_code))
  ) throw new Error("Invalid account deletion status");
  const receipt = row.receipt === null ? null : parseAccountDeletionReceipt(row.receipt);
  const validState = (
    row.status === "awaiting_confirmation"
      ? row.phase === "prepared" && row.retryable_failure_code === null && receipt === null
      : row.status === "deleting"
        ? row.phase !== "prepared" && row.phase !== "completed"
          && row.retryable_failure_code === null && receipt === null
        : row.status === "retryable_failure"
          ? row.phase !== "prepared" && row.phase !== "completed"
            && row.retryable_failure_code !== null && receipt === null
          : row.phase === "completed"
            && row.retryable_failure_code === null && receipt !== null
  );
  if (!validState) {
    throw new Error("Invalid account deletion status receipt");
  }
  return {...row, receipt} as unknown as AccountDeletionStatus;
}

export interface AccountDeletionBrowserRecovery {
  schema: "teachlab.account_deletion_browser_recovery.v1";
  challenge: AccountDeletionChallenge;
  idempotency_key: string;
}

export function saveAccountDeletionBrowserRecovery(
  storage: Pick<Storage, "setItem">,
  challenge: AccountDeletionChallenge,
  idempotencyKey: string
): boolean {
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$/.test(idempotencyKey)) {
    throw new Error("Invalid account deletion idempotency key");
  }
  try {
    storage.setItem(ACCOUNT_DELETION_RECOVERY_STORAGE_KEY, JSON.stringify({
      schema: "teachlab.account_deletion_browser_recovery.v1",
      challenge: parseAccountDeletionChallenge(challenge),
      idempotency_key: idempotencyKey
    } satisfies AccountDeletionBrowserRecovery));
    return true;
  } catch {
    return false;
  }
}

export function loadAccountDeletionBrowserRecovery(
  storage: Pick<Storage, "getItem" | "removeItem">,
  now = Date.now()
): AccountDeletionBrowserRecovery | null {
  try {
    const raw = storage.getItem(ACCOUNT_DELETION_RECOVERY_STORAGE_KEY);
    if (!raw) return null;
    const value = record(JSON.parse(raw));
    if (
      !value
      || !exactKeys(value, ["schema", "challenge", "idempotency_key"])
      || value.schema !== "teachlab.account_deletion_browser_recovery.v1"
      || !/^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$/.test(String(value.idempotency_key))
    ) throw new Error("invalid");
    const challenge = parseAccountDeletionChallenge(value.challenge);
    if (Date.parse(challenge.expires_at) <= now) throw new Error("expired");
    return {
      schema: "teachlab.account_deletion_browser_recovery.v1",
      challenge,
      idempotency_key: String(value.idempotency_key)
    };
  } catch {
    try {
      storage.removeItem(ACCOUNT_DELETION_RECOVERY_STORAGE_KEY);
    } catch {
      // A storage denial cannot promote malformed recovery authority.
    }
    return null;
  }
}

export function clearAccountDeletionBrowserRecovery(
  storage: Pick<Storage, "removeItem">
): boolean {
  try {
    storage.removeItem(ACCOUNT_DELETION_RECOVERY_STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}

export class CommittedAccountCleanupError extends Error {
  readonly receipt: AccountDeletionReceipt;
  readonly cleanup: AccountBrowserPurgeResult;

  constructor(
    receipt: AccountDeletionReceipt,
    cleanup: AccountBrowserPurgeResult
  ) {
    super("Account deletion committed, but browser cleanup is incomplete");
    this.name = "CommittedAccountCleanupError";
    this.receipt = receipt;
    this.cleanup = cleanup;
  }
}

/** Never clears browser data before a validated committed server receipt. */
export async function completeCommittedAccountDeletion(
  receiptValue: unknown,
  ports: AccountBrowserPurgePorts
): Promise<{receipt: AccountDeletionReceipt; cleanup: AccountBrowserPurgeResult}> {
  const receipt = parseAccountDeletionReceipt(receiptValue);
  const cleanup = await purgeAccountBrowserState(ports);
  if (!cleanup.complete) throw new CommittedAccountCleanupError(receipt, cleanup);
  return {receipt, cleanup};
}
