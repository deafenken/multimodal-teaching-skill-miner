import type {AccessScope} from "../tenancy/access-scope";
import type {
  AccountDeletionCounts,
  AccountDeletionOperationRecord,
  AccountDeletionPhase,
  AccountDeletionReceipt,
  AccountDeletionTombstone,
  AccountPostgresExportSnapshot,
  AccountScopeHashes
} from "./account-data-rights.types";

export interface PrepareAccountDeletionInput extends AccessScope {
  hashes: AccountScopeHashes;
  operationId: string;
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
  now: Date;
}

export interface BeginAccountDeletionInput extends AccessScope {
  hashes: AccountScopeHashes;
  operationId: string;
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
  expectedRevision: number;
  idempotencyKeySha256: string;
  confirmationSha256: string;
  now: Date;
}

export type BeginAccountDeletionResult =
  | {kind: "started" | "idempotent"; operation: AccountDeletionOperationRecord}
  | {kind: "missing" | "expired" | "revision_conflict" | "payload_conflict"};

export interface AdvanceAccountDeletionInput extends AccessScope {
  hashes: AccountScopeHashes;
  operationId: string;
  expectedPhase: AccountDeletionPhase;
  expectedRevision: number;
  nextPhase: Exclude<AccountDeletionPhase, "prepared" | "completed">;
  counts?: Partial<AccountDeletionCounts>;
  leaseTokenSha256: string;
  now: Date;
}

export interface SealAccountDeletionTombstoneInput extends AccessScope {
  hashes: AccountScopeHashes;
  operationId: string;
  expectedRevision: number;
  leaseTokenSha256: string;
  now: Date;
}

export interface CommitAccountDeletionInput extends AccessScope {
  hashes: AccountScopeHashes;
  operationId: string;
  expectedRevision: number;
  leaseTokenSha256: string;
  workerCounts: Pick<
    AccountDeletionCounts,
    "workerFiles" | "workerBytes" | "workerRoots"
  >;
  receiptId: string;
  deletedAt: Date;
}

export interface AccountDeletionStatusLookup {
  scopeSha256: string;
  operationId: string;
  statusCapabilitySha256: string;
}

export type AccountScopeLifecycleState =
  | {kind: "active"}
  | {kind: "deleting"; operation: AccountDeletionOperationRecord}
  | {kind: "deleted"; tombstone: AccountDeletionTombstone};

export interface AccountDeletionLeaseClaim {
  leaseOwnerSha256: string;
  leaseTokenSha256: string;
  leaseDurationMs: number;
  now: Date;
  /** A user-authorized resume may ignore backoff, but never an active lease. */
  ignoreRecoveryAfter?: boolean;
}

export type AccountDeletionLeaseClaimResult =
  | {kind: "claimed"; operation: AccountDeletionOperationRecord}
  | {kind: "busy"; operation: AccountDeletionOperationRecord}
  | {kind: "completed"; tombstone: AccountDeletionTombstone}
  | {kind: "missing"};

/**
 * Durable account lifecycle and scoped PostgreSQL data boundary.
 *
 * Production implements every mutating method inside a tenant transaction.
 * Final deletion additionally inserts/updates the hash-only tombstone and
 * deletes events -> tasks -> sessions -> auth sessions in the same commit.
 */
export interface AccountDataRightsRepositoryPort {
  lifecycle(
    scope: AccessScope,
    hashes: AccountScopeHashes
  ): Promise<AccountScopeLifecycleState>;

  exportPostgresSnapshot(
    scope: AccessScope,
    hashes: AccountScopeHashes
  ): Promise<AccountPostgresExportSnapshot>;

  prepareDeletion(
    input: PrepareAccountDeletionInput
  ): Promise<AccountDeletionOperationRecord>;

  beginDeletion(
    input: BeginAccountDeletionInput
  ): Promise<BeginAccountDeletionResult>;

  advanceDeletion(
    input: AdvanceAccountDeletionInput
  ): Promise<AccountDeletionOperationRecord>;

  sealTombstone(
    input: SealAccountDeletionTombstoneInput
  ): Promise<AccountDeletionTombstone>;

  claimDeletionForScope(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    claim: AccountDeletionLeaseClaim
  ): Promise<AccountDeletionLeaseClaimResult>;

  claimDeletionByCapability(
    lookup: AccountDeletionStatusLookup,
    claim: AccountDeletionLeaseClaim
  ): Promise<AccountDeletionLeaseClaimResult>;

  claimNextDeletion(
    claim: AccountDeletionLeaseClaim
  ): Promise<AccountDeletionOperationRecord | undefined>;

  renewDeletionLease(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    leaseTokenSha256: string,
    leaseDurationMs: number,
    now: Date
  ): Promise<AccountDeletionOperationRecord | undefined>;

  operationForLease(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    leaseTokenSha256: string,
    now: Date
  ): Promise<AccountDeletionOperationRecord | undefined>;

  markRetryableFailure(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    leaseTokenSha256: string,
    failureCode: string,
    retryAt: Date,
    now: Date
  ): Promise<void>;

  commitDeletion(
    input: CommitAccountDeletionInput
  ): Promise<AccountDeletionReceipt>;

  statusByCapability(
    lookup: AccountDeletionStatusLookup
  ): Promise<AccountDeletionOperationRecord | AccountDeletionTombstone | undefined>;

  statusForScope(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    lookup: AccountDeletionStatusLookup
  ): Promise<AccountDeletionOperationRecord | AccountDeletionTombstone | undefined>;
}

/** Optional process-local data source used only by the non-production adapter. */
export interface AccountInMemoryDataPort {
  snapshot(scope: AccessScope): Promise<AccountPostgresExportSnapshot>;
  deleteScope(scope: AccessScope): Promise<Pick<
    AccountDeletionCounts,
    "postgresEvents" | "postgresTasks" | "postgresArtifacts" | "postgresSessions" | "postgresAuthSessions"
  >>;
}

/** Scope runtime coordination shared by the legacy queue and Harness worker pool. */
export interface AccountScopeRuntimeCoordinatorPort {
  exportWorkerData(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    signal?: AbortSignal
  ): Promise<import("./account-data-rights.types").AccountWorkerExportLease>;

  fence(scope: AccessScope, hashes: AccountScopeHashes, operationId: string): Promise<void>;

  drain(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    timeoutMs: number
  ): Promise<import("./account-data-rights.types").AccountRuntimeDrainResult>;

  quarantine(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string
  ): Promise<void>;

  purgeQuarantine(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string
  ): Promise<import("./account-data-rights.types").AccountWorkerPurgeResult>;

  /** Fence and drain the durable API task queue for this account scope. */
  fenceApiTasks?(scope: AccessScope): Promise<void>;
  drainApiTasks?(
    scope: AccessScope,
    timeoutMs: number
  ): Promise<Pick<import("./account-data-rights.types").AccountRuntimeDrainResult, "apiTasksCancelled" | "apiTasksHandedOff">>;
}

export interface AccountDeletionStatusCookiePort {
  mint(input: {
    operationId: string;
    scopeSha256: string;
    capability: string;
    expiresAt: Date;
  }): string;
  verify(rawCookieHeader: string | undefined, now?: Date):
    | import("./account-data-rights.types").AccountDeletionStatusCapability
    | undefined;
  serialize(value: string, expiresAt: Date): string;
  clear(): string;
}

/**
 * Resolves a server-side OIDC step-up grant for the authenticated session.
 * Implementations must fail closed when the grant is absent, stale, below the
 * configured AAL, or not bound to the canonical issuer/tenant/subject tuple.
 */
export interface AccountDeletionFreshAuthorityPort {
  current(
    principal: import("../auth/auth-provider.port").AuthenticatedPrincipal,
    now: Date
  ): Promise<import("./account-data-rights.types").AccountDeletionFreshAuthority | null>;
}
