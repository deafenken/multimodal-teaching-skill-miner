import type {Readable} from "node:stream";

import type {AccessScope} from "../tenancy/access-scope";

export const ACCOUNT_EXPORT_SCHEMA = "teachlab.account_private_export.v1";
export const ACCOUNT_EXPORT_MANIFEST_SCHEMA =
  "teachlab.account_private_export_manifest.v1";
export const ACCOUNT_DELETION_OPERATION_SCHEMA =
  "teachlab.account_deletion_operation.v1";
export const ACCOUNT_DELETION_RECEIPT_SCHEMA =
  "teachlab.account_deletion_receipt.v1";
export const ACCOUNT_DELETION_CONFIRMATION_PHRASE =
  "PERMANENTLY DELETE MY TEACHLAB ACCOUNT";

export const ACCOUNT_DELETION_PHASES = [
  "prepared",
  "fencing",
  "draining",
  "tombstoned",
  "quarantining",
  "purging_worker_data",
  "committing_database",
  "completed"
] as const;

export type AccountDeletionPhase = (typeof ACCOUNT_DELETION_PHASES)[number];

export interface AccountScopeHashes {
  /** Hash made with the active key and used for newly-created records. */
  active: string;
  /** Active plus retained previous-key hashes, with duplicates removed. */
  candidates: readonly string[];
}

export interface AccountDeletionCounts {
  postgresEvents: number;
  postgresTasks: number;
  postgresArtifacts: number;
  postgresSessions: number;
  postgresAuthSessions: number;
  workerFiles: number;
  workerBytes: number;
  workerRoots: number;
}

export const EMPTY_ACCOUNT_DELETION_COUNTS: Readonly<AccountDeletionCounts> =
  Object.freeze({
    postgresEvents: 0,
    postgresTasks: 0,
    postgresArtifacts: 0,
    postgresSessions: 0,
    postgresAuthSessions: 0,
    workerFiles: 0,
    workerBytes: 0,
    workerRoots: 0
  });

/** Internal durable record. Raw scope fields must never enter an HTTP response. */
export interface AccountDeletionOperationRecord extends AccessScope {
  schema: typeof ACCOUNT_DELETION_OPERATION_SCHEMA;
  operationId: string;
  scopeSha256: string;
  phase: Exclude<AccountDeletionPhase, "completed">;
  revision: number;
  challengeId: string;
  challengeTokenSha256: string;
  challengeCsrfSha256: string;
  challengeSessionSha256: string;
  challengeAuthorityGrantSha256: string;
  challengeCanonicalIdentitySha256: string;
  challengeIssuerSha256: string;
  challengeAuthorityKeyVersion: string;
  challengeAuthenticatedAt: Date;
  challengeAssuranceLevel: number;
  challengeExpiresAt: Date;
  statusCapabilitySha256: string;
  idempotencyKeySha256: string | null;
  confirmationSha256: string | null;
  retryableFailureCode: string | null;
  recoveryLeaseOwnerSha256: string | null;
  recoveryLeaseTokenSha256: string | null;
  recoveryLeaseExpiresAt: Date | null;
  recoveryAfter: Date | null;
  recoveryAttempts: number;
  counts: AccountDeletionCounts;
  createdAt: Date;
  updatedAt: Date;
}

export interface AccountDeletionTombstone {
  scopeSha256: string;
  /** Original active scope hash carried by the content-free receipt. */
  receiptScopeSha256: string;
  operationIdSha256: string;
  phase: Exclude<AccountDeletionPhase, "prepared">;
  revision: number;
  statusCapabilitySha256: string;
  retryableFailureCode: string | null;
  counts: AccountDeletionCounts;
  deletedAt: Date | null;
  receiptId: string | null;
  receiptSha256: string | null;
  createdAt: Date;
  updatedAt: Date;
}

export interface AccountDeletionReceipt {
  schema: typeof ACCOUNT_DELETION_RECEIPT_SCHEMA;
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

export interface AccountDeletionStatusProjection {
  schema: "teachlab.account_deletion_status.v1";
  status:
    | "awaiting_confirmation"
    | "deleting"
    | "retryable_failure"
    | "permanently_deleted";
  phase: AccountDeletionPhase;
  revision: number;
  retryable_failure_code: string | null;
  receipt: AccountDeletionReceipt | null;
}

/**
 * One immutable export entry. Implementations must return a fresh stream and
 * must not expose a host path through `archivePath` or error messages.
 */
export interface AccountExportEntry {
  archivePath: string;
  byteLength: number;
  sha256: string;
  crc32: number;
  mediaType: string;
  dataClass: string;
  open(): Readable;
}

export interface AccountPostgresExportSnapshot {
  capturedAt: string;
  sessions: readonly Record<string, unknown>[];
  tasks: readonly Record<string, unknown>[];
  events: readonly Record<string, unknown>[];
  authSessionAudit: readonly Record<string, unknown>[];
  artifacts: readonly Record<string, unknown>[];
}

export interface AccountWorkerExportLease {
  capturedAt: string;
  entries: readonly AccountExportEntry[];
  release(): Promise<void>;
}

export interface AccountExportArchive {
  filename: string;
  stream: Readable;
  byteLength: number;
  manifestSha256: string;
  entryCount: number;
}

export interface AccountDeletionPrepared {
  schema: "teachlab.account_deletion_challenge.v1";
  challenge_id: string;
  confirmation_token: string;
  confirmation_phrase: typeof ACCOUNT_DELETION_CONFIRMATION_PHRASE;
  expires_at: string;
  revision: number;
}

export interface AccountDeletionConfirmationInput {
  challengeId: string;
  confirmationToken: string;
  confirmationPhrase: string;
  expectedRevision: number;
  idempotencyKey: string;
}

/**
 * Hash-only projection of a server-held, recently reauthenticated OIDC grant.
 *
 * The browser must never construct this value. A production authority adapter
 * derives each tuple from an issuer + tenant + subject canonical identity and
 * retained server keys. `bindings[0]` is active; retained entries permit key
 * rotation during the short confirmation window without persisting raw claims.
 */
export interface AccountDeletionFreshAuthorityBinding {
  authorityGrantSha256: string;
  canonicalIdentitySha256: string;
  issuerSha256: string;
  keyVersion: string;
}

export interface AccountDeletionFreshAuthority {
  bindings: readonly AccountDeletionFreshAuthorityBinding[];
  sessionSha256: string;
  authenticatedAt: Date;
  expiresAt: Date;
  assuranceLevel: number;
}

export interface AccountDeletionStatusCapability {
  operationId: string;
  scopeSha256: string;
  capability: string;
}

export interface AccountRuntimeDrainResult {
  apiTasksCancelled: number;
  apiTasksHandedOff: number;
  workerTasksCancelled: number;
  workerTasksHandedOff: number;
}

export interface AccountWorkerPurgeResult {
  workerFiles: number;
  workerBytes: number;
  workerRoots: number;
}
