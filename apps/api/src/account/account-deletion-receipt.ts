import {createHash} from "node:crypto";

import {canonicalAccountJson} from "./account-export-archive";
import {
  ACCOUNT_DELETION_RECEIPT_SCHEMA,
  type AccountDeletionCounts,
  type AccountDeletionOperationRecord,
  type AccountDeletionReceipt,
  type AccountDeletionStatusProjection,
  type AccountDeletionTombstone
} from "./account-data-rights.types";

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const RECEIPT_ID_PATTERN = /^adelr_[0-9a-f]{32}$/;

export function sha256Text(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

export function normalizedDeletionCounts(
  value: Partial<AccountDeletionCounts> | undefined
): AccountDeletionCounts {
  const source = value ?? {};
  const field = (name: keyof AccountDeletionCounts) => {
    const candidate = source[name] ?? 0;
    if (!Number.isSafeInteger(candidate) || candidate < 0) {
      throw new Error("Invalid account deletion count");
    }
    return candidate;
  };
  return {
    postgresEvents: field("postgresEvents"),
    postgresTasks: field("postgresTasks"),
    postgresArtifacts: field("postgresArtifacts"),
    postgresSessions: field("postgresSessions"),
    postgresAuthSessions: field("postgresAuthSessions"),
    workerFiles: field("workerFiles"),
    workerBytes: field("workerBytes"),
    workerRoots: field("workerRoots")
  };
}

export function mergeDeletionCounts(
  current: AccountDeletionCounts,
  patch: Partial<AccountDeletionCounts> | undefined
): AccountDeletionCounts {
  return normalizedDeletionCounts({...current, ...(patch ?? {})});
}

export function accountDeletionReceipt(input: {
  receiptId: string;
  scopeSha256: string;
  operationId: string;
  deletedAt: Date;
  counts: AccountDeletionCounts;
}): AccountDeletionReceipt {
  if (
    !RECEIPT_ID_PATTERN.test(input.receiptId)
    || !SHA256_PATTERN.test(input.scopeSha256)
    || !Number.isFinite(input.deletedAt.getTime())
  ) {
    throw new Error("Invalid account deletion receipt input");
  }
  const counts = normalizedDeletionCounts(input.counts);
  const unsigned = {
    schema: ACCOUNT_DELETION_RECEIPT_SCHEMA,
    status: "permanently_deleted" as const,
    receipt_id: input.receiptId,
    scope_sha256: input.scopeSha256,
    operation_id_sha256: sha256Text(input.operationId),
    deleted_at: input.deletedAt.toISOString(),
    deleted_counts: {
      postgres_events: counts.postgresEvents,
      postgres_tasks: counts.postgresTasks,
      postgres_artifacts: counts.postgresArtifacts,
      postgres_sessions: counts.postgresSessions,
      postgres_auth_sessions: counts.postgresAuthSessions,
      worker_files: counts.workerFiles,
      worker_bytes: counts.workerBytes,
      worker_roots: counts.workerRoots
    },
    all_devices_session_authority_deleted: true as const,
    scope_tombstone_retained: true as const,
    user_managed_export_copies_deleted: false as const,
    user_managed_export_copies_status: "outside_service_control" as const,
    remote_provider_copies_deleted: false as const,
    remote_provider_copies_status:
      "outside_service_control_subject_to_provider_retention" as const,
    identity_provider_account_deleted: false as const,
    identity_provider_account_status:
      "outside_service_control_contact_organization_idp" as const,
    operator_backup_copies_deleted: false as const,
    operator_backup_copies_status:
      "pending_retention_expiry_or_operator_crypto_erasure" as const
  } as const;
  return {
    ...unsigned,
    receipt_sha256: createHash("sha256")
      .update(canonicalAccountJson(unsigned))
      .digest("hex")
  };
}

export function accountDeletionStatusProjection(
  record: AccountDeletionOperationRecord | AccountDeletionTombstone,
  receipt?: AccountDeletionReceipt
): AccountDeletionStatusProjection {
  const completed = record.phase === "completed";
  return {
    schema: "teachlab.account_deletion_status.v1",
    status: completed
      ? "permanently_deleted"
      : record.retryableFailureCode
        ? "retryable_failure"
        : record.phase === "prepared"
          ? "awaiting_confirmation"
          : "deleting",
    phase: record.phase,
    revision: record.revision,
    retryable_failure_code: record.retryableFailureCode,
    receipt: completed ? receipt ?? null : null
  };
}
