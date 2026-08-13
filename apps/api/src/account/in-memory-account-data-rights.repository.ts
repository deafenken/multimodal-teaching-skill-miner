import {Injectable} from "@nestjs/common";

import type {AccessScope} from "../tenancy/access-scope";
import {
  accountDeletionReceipt,
  mergeDeletionCounts,
  normalizedDeletionCounts,
  sha256Text
} from "./account-deletion-receipt";
import {AccountDataRightsError} from "./account-data-rights.errors";
import type {
  AccountDataRightsRepositoryPort,
  AccountDeletionLeaseClaim,
  AccountDeletionLeaseClaimResult,
  AccountDeletionStatusLookup,
  AccountInMemoryDataPort,
  AccountScopeLifecycleState,
  AdvanceAccountDeletionInput,
  BeginAccountDeletionInput,
  BeginAccountDeletionResult,
  CommitAccountDeletionInput,
  PrepareAccountDeletionInput,
  SealAccountDeletionTombstoneInput
} from "./account-data-rights.repository.port";
import {
  ACCOUNT_DELETION_OPERATION_SCHEMA,
  EMPTY_ACCOUNT_DELETION_COUNTS,
  type AccountDeletionOperationRecord,
  type AccountDeletionReceipt,
  type AccountDeletionTombstone,
  type AccountPostgresExportSnapshot,
  type AccountScopeHashes
} from "./account-data-rights.types";

const MAX_LOCAL_ACCOUNT_OPERATIONS = 4_096;
const SAFE_FAILURE_CODE = /^[a-z][a-z0-9_]{2,63}$/;

const EMPTY_MEMORY_DATA: AccountInMemoryDataPort = {
  snapshot: async () => ({
    capturedAt: new Date().toISOString(),
    sessions: [],
    tasks: [],
    events: [],
    authSessionAudit: [],
    artifacts: []
  }),
  deleteScope: async () => ({
    postgresEvents: 0,
    postgresTasks: 0,
    postgresArtifacts: 0,
    postgresSessions: 0,
    postgresAuthSessions: 0
  })
};

function sameScope(record: AccessScope, scope: AccessScope): boolean {
  return record.tenantId === scope.tenantId && record.ownerId === scope.ownerId;
}

function cloneOperation(
  record: AccountDeletionOperationRecord
): AccountDeletionOperationRecord {
  return {
    ...structuredClone(record),
    challengeAuthenticatedAt: new Date(record.challengeAuthenticatedAt),
    challengeExpiresAt: new Date(record.challengeExpiresAt),
    recoveryLeaseExpiresAt: record.recoveryLeaseExpiresAt
      ? new Date(record.recoveryLeaseExpiresAt)
      : null,
    recoveryAfter: record.recoveryAfter ? new Date(record.recoveryAfter) : null,
    createdAt: new Date(record.createdAt),
    updatedAt: new Date(record.updatedAt)
  };
}

function cloneTombstone(record: AccountDeletionTombstone): AccountDeletionTombstone {
  return {
    ...structuredClone(record),
    deletedAt: record.deletedAt ? new Date(record.deletedAt) : null,
    createdAt: new Date(record.createdAt),
    updatedAt: new Date(record.updatedAt)
  };
}

@Injectable()
export class InMemoryAccountDataRightsRepository
  implements AccountDataRightsRepositoryPort {
  private readonly operations = new Map<string, AccountDeletionOperationRecord>();
  private readonly tombstones = new Map<string, AccountDeletionTombstone>();

  constructor(private readonly data: AccountInMemoryDataPort = EMPTY_MEMORY_DATA) {}

  async lifecycle(
    scope: AccessScope,
    hashes: AccountScopeHashes
  ): Promise<AccountScopeLifecycleState> {
    const operation = this.findOperation(scope, hashes);
    if (operation && operation.phase !== "prepared") {
      return {kind: "deleting", operation: cloneOperation(operation)};
    }
    const tombstone = this.findTombstone(hashes);
    if (tombstone) return {kind: "deleted", tombstone: cloneTombstone(tombstone)};
    return {kind: "active"};
  }

  async exportPostgresSnapshot(
    scope: AccessScope,
    hashes: AccountScopeHashes
  ): Promise<AccountPostgresExportSnapshot> {
    const state = await this.lifecycle(scope, hashes);
    if (state.kind === "deleting") {
      throw new AccountDataRightsError(409, "account_deletion_already_started");
    }
    if (state.kind === "deleted") {
      throw new AccountDataRightsError(410, "account_already_deleted");
    }
    return structuredClone(await this.data.snapshot(scope));
  }

  async prepareDeletion(
    input: PrepareAccountDeletionInput
  ): Promise<AccountDeletionOperationRecord> {
    if (this.findTombstone(input.hashes)) {
      throw new AccountDataRightsError(410, "account_already_deleted");
    }
    const existing = this.findOperation(input, input.hashes);
    if (existing && existing.phase !== "prepared") {
      throw new AccountDataRightsError(409, "account_deletion_already_started");
    }
    if (!existing && this.operations.size >= MAX_LOCAL_ACCOUNT_OPERATIONS) {
      // Never evict a deletion operation or tombstone: forgetting either can
      // reopen a scope. Local development fails closed at capacity instead.
      throw new AccountDataRightsError(503, "account_deletion_retry_required");
    }
    if (existing) this.operations.delete(existing.scopeSha256);
    const record: AccountDeletionOperationRecord = {
      schema: ACCOUNT_DELETION_OPERATION_SCHEMA,
      tenantId: input.tenantId,
      ownerId: input.ownerId,
      operationId: input.operationId,
      scopeSha256: input.hashes.active,
      phase: "prepared",
      revision: (existing?.revision ?? 0) + 1,
      challengeId: input.challengeId,
      challengeTokenSha256: input.challengeTokenSha256,
      challengeCsrfSha256: input.challengeCsrfSha256,
      challengeSessionSha256: input.challengeSessionSha256,
      challengeAuthorityGrantSha256: input.challengeAuthorityGrantSha256,
      challengeCanonicalIdentitySha256: input.challengeCanonicalIdentitySha256,
      challengeIssuerSha256: input.challengeIssuerSha256,
      challengeAuthorityKeyVersion: input.challengeAuthorityKeyVersion,
      challengeAuthenticatedAt: new Date(input.challengeAuthenticatedAt),
      challengeAssuranceLevel: input.challengeAssuranceLevel,
      challengeExpiresAt: new Date(input.challengeExpiresAt),
      statusCapabilitySha256: input.statusCapabilitySha256,
      idempotencyKeySha256: null,
      confirmationSha256: null,
      retryableFailureCode: null,
      recoveryLeaseOwnerSha256: null,
      recoveryLeaseTokenSha256: null,
      recoveryLeaseExpiresAt: null,
      recoveryAfter: null,
      recoveryAttempts: 0,
      counts: normalizedDeletionCounts(EMPTY_ACCOUNT_DELETION_COUNTS),
      createdAt: existing?.createdAt ? new Date(existing.createdAt) : new Date(input.now),
      updatedAt: new Date(input.now)
    };
    this.operations.set(record.scopeSha256, record);
    return cloneOperation(record);
  }

  async beginDeletion(
    input: BeginAccountDeletionInput
  ): Promise<BeginAccountDeletionResult> {
    const current = this.findOperation(input, input.hashes);
    if (!current || current.operationId !== input.operationId) return {kind: "missing"};
    if (current.phase !== "prepared") {
      return current.idempotencyKeySha256 === input.idempotencyKeySha256
        && current.confirmationSha256 === input.confirmationSha256
        ? {kind: "idempotent", operation: cloneOperation(current)}
        : {kind: "payload_conflict"};
    }
    if (current.revision !== input.expectedRevision) return {kind: "revision_conflict"};
    if (current.challengeExpiresAt.getTime() <= input.now.getTime()) return {kind: "expired"};
    if (
      current.challengeId !== input.challengeId
      || current.challengeTokenSha256 !== input.challengeTokenSha256
      || current.challengeCsrfSha256 !== input.challengeCsrfSha256
      || current.challengeSessionSha256 !== input.challengeSessionSha256
      || current.challengeAuthorityGrantSha256
        !== input.challengeAuthorityGrantSha256
      || current.challengeCanonicalIdentitySha256
        !== input.challengeCanonicalIdentitySha256
      || current.challengeIssuerSha256 !== input.challengeIssuerSha256
      || current.challengeAuthorityKeyVersion
        !== input.challengeAuthorityKeyVersion
      || current.challengeAuthenticatedAt.getTime()
        !== input.challengeAuthenticatedAt.getTime()
      || current.challengeAssuranceLevel !== input.challengeAssuranceLevel
    ) return {kind: "payload_conflict"};
    const updated: AccountDeletionOperationRecord = {
      ...current,
      phase: "fencing",
      revision: current.revision + 1,
      idempotencyKeySha256: input.idempotencyKeySha256,
      confirmationSha256: input.confirmationSha256,
      retryableFailureCode: null,
      recoveryLeaseOwnerSha256: null,
      recoveryLeaseTokenSha256: null,
      recoveryLeaseExpiresAt: null,
      recoveryAfter: null,
      updatedAt: new Date(input.now)
    };
    this.operations.set(current.scopeSha256, updated);
    return {kind: "started", operation: cloneOperation(updated)};
  }

  async advanceDeletion(
    input: AdvanceAccountDeletionInput
  ): Promise<AccountDeletionOperationRecord> {
    const current = this.findOperation(input, input.hashes);
    if (
      !current
      || current.operationId !== input.operationId
      || current.phase !== input.expectedPhase
      || current.revision !== input.expectedRevision
      || current.recoveryLeaseTokenSha256 !== input.leaseTokenSha256
      || !current.recoveryLeaseExpiresAt
      || current.recoveryLeaseExpiresAt.getTime() <= input.now.getTime()
    ) {
      throw new AccountDataRightsError(409, "account_deletion_revision_conflict");
    }
    const updated: AccountDeletionOperationRecord = {
      ...current,
      phase: input.nextPhase,
      revision: current.revision + 1,
      retryableFailureCode: null,
      counts: mergeDeletionCounts(current.counts, input.counts),
      updatedAt: new Date(input.now)
    };
    this.operations.set(current.scopeSha256, updated);
    this.updateTombstonesForOperation(updated);
    return cloneOperation(updated);
  }

  async sealTombstone(
    input: SealAccountDeletionTombstoneInput
  ): Promise<AccountDeletionTombstone> {
    const current = this.findOperation(input, input.hashes);
    if (
      !current
      || current.operationId !== input.operationId
      || current.phase !== "draining"
      || current.revision !== input.expectedRevision
      || current.recoveryLeaseTokenSha256 !== input.leaseTokenSha256
      || !current.recoveryLeaseExpiresAt
      || current.recoveryLeaseExpiresAt.getTime() <= input.now.getTime()
    ) {
      throw new AccountDataRightsError(409, "account_deletion_revision_conflict");
    }
    const updated: AccountDeletionOperationRecord = {
      ...current,
      phase: "tombstoned",
      revision: current.revision + 1,
      retryableFailureCode: null,
      updatedAt: new Date(input.now)
    };
    this.operations.set(current.scopeSha256, updated);
    const tombstone: AccountDeletionTombstone = {
      scopeSha256: current.scopeSha256,
      receiptScopeSha256: current.scopeSha256,
      operationIdSha256: sha256Text(current.operationId),
      phase: "tombstoned",
      revision: updated.revision,
      statusCapabilitySha256: current.statusCapabilitySha256,
      retryableFailureCode: null,
      counts: normalizedDeletionCounts(current.counts),
      deletedAt: null,
      receiptId: null,
      receiptSha256: null,
      createdAt: new Date(input.now),
      updatedAt: new Date(input.now)
    };
    for (const hash of input.hashes.candidates) {
      this.tombstones.set(hash, {...cloneTombstone(tombstone), scopeSha256: hash});
    }
    return cloneTombstone(tombstone);
  }

  async claimDeletionForScope(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    claim: AccountDeletionLeaseClaim
  ): Promise<AccountDeletionLeaseClaimResult> {
    const current = this.findOperation(scope, hashes);
    if (!current || current.operationId !== operationId) {
      const completed = this.findTombstone(hashes);
      return completed?.phase === "completed"
        ? {kind: "completed", tombstone: cloneTombstone(completed)}
        : {kind: "missing"};
    }
    return this.claim(current, claim);
  }

  async claimDeletionByCapability(
    lookup: AccountDeletionStatusLookup,
    claim: AccountDeletionLeaseClaim
  ): Promise<AccountDeletionLeaseClaimResult> {
    const current = this.operations.get(lookup.scopeSha256);
    if (
      current
      && current.operationId === lookup.operationId
      && current.statusCapabilitySha256 === lookup.statusCapabilitySha256
    ) return this.claim(current, claim);
    const completed = this.tombstones.get(lookup.scopeSha256);
    if (
      completed
      && completed.operationIdSha256 === sha256Text(lookup.operationId)
      && completed.statusCapabilitySha256 === lookup.statusCapabilitySha256
    ) return {kind: "completed", tombstone: cloneTombstone(completed)};
    return {kind: "missing"};
  }

  async claimNextDeletion(
    claim: AccountDeletionLeaseClaim
  ): Promise<AccountDeletionOperationRecord | undefined> {
    const current = [...this.operations.values()]
      .filter((record) => record.phase !== "prepared")
      .filter((record) => !record.recoveryAfter
        || record.recoveryAfter.getTime() <= claim.now.getTime())
      .filter((record) => !record.recoveryLeaseExpiresAt
        || record.recoveryLeaseExpiresAt.getTime() <= claim.now.getTime())
      .sort((left, right) => left.updatedAt.getTime() - right.updatedAt.getTime())[0];
    if (!current) return undefined;
    const claimed = this.claim(current, claim);
    return claimed.kind === "claimed" ? claimed.operation : undefined;
  }

  async renewDeletionLease(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    leaseTokenSha256: string,
    leaseDurationMs: number,
    now: Date
  ): Promise<AccountDeletionOperationRecord | undefined> {
    const current = this.findOperation(scope, hashes);
    if (
      !current
      || current.operationId !== operationId
      || current.recoveryLeaseTokenSha256 !== leaseTokenSha256
      || !current.recoveryLeaseExpiresAt
      || current.recoveryLeaseExpiresAt.getTime() <= now.getTime()
    ) return undefined;
    const updated = {
      ...current,
      recoveryLeaseExpiresAt: new Date(now.getTime() + leaseDurationMs)
    };
    this.operations.set(current.scopeSha256, updated);
    return cloneOperation(updated);
  }

  async operationForLease(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    leaseTokenSha256: string,
    now: Date
  ): Promise<AccountDeletionOperationRecord | undefined> {
    const current = this.findOperation(scope, hashes);
    return current
      && current.operationId === operationId
      && current.recoveryLeaseTokenSha256 === leaseTokenSha256
      && current.recoveryLeaseExpiresAt
      && current.recoveryLeaseExpiresAt.getTime() > now.getTime()
      ? cloneOperation(current)
      : undefined;
  }

  async markRetryableFailure(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    leaseTokenSha256: string,
    failureCode: string,
    retryAt: Date,
    now: Date
  ): Promise<void> {
    if (!SAFE_FAILURE_CODE.test(failureCode)) {
      throw new Error("Invalid content-free account deletion failure code");
    }
    const current = this.findOperation(scope, hashes);
    if (
      !current
      || current.operationId !== operationId
      || current.phase === "prepared"
      || current.recoveryLeaseTokenSha256 !== leaseTokenSha256
    ) {
      return;
    }
    const updated = {
      ...current,
      retryableFailureCode: failureCode,
      recoveryLeaseOwnerSha256: null,
      recoveryLeaseTokenSha256: null,
      recoveryLeaseExpiresAt: null,
      recoveryAfter: new Date(retryAt),
      revision: current.revision + 1,
      updatedAt: new Date(now)
    };
    this.operations.set(current.scopeSha256, updated);
    this.updateTombstonesForOperation(updated);
  }

  async commitDeletion(
    input: CommitAccountDeletionInput
  ): Promise<AccountDeletionReceipt> {
    const current = this.findOperation(input, input.hashes);
    if (
      !current
      || current.operationId !== input.operationId
      || current.phase !== "committing_database"
      || current.revision !== input.expectedRevision
      || current.recoveryLeaseTokenSha256 !== input.leaseTokenSha256
      || !current.recoveryLeaseExpiresAt
      || current.recoveryLeaseExpiresAt.getTime() <= input.deletedAt.getTime()
    ) {
      const completed = this.findTombstone(input.hashes);
      if (completed?.phase === "completed" && completed.receiptId && completed.deletedAt) {
        return accountDeletionReceipt({
          receiptId: completed.receiptId,
          scopeSha256: completed.receiptScopeSha256,
          operationId: input.operationId,
          deletedAt: completed.deletedAt,
          counts: completed.counts
        });
      }
      throw new AccountDataRightsError(409, "account_deletion_revision_conflict");
    }
    const databaseCounts = await this.data.deleteScope(input);
    const counts = mergeDeletionCounts(current.counts, {
      ...databaseCounts,
      ...input.workerCounts
    });
    const receipt = accountDeletionReceipt({
      receiptId: input.receiptId,
      scopeSha256: current.scopeSha256,
      operationId: current.operationId,
      deletedAt: input.deletedAt,
      counts
    });
    const completed: AccountDeletionTombstone = {
      scopeSha256: current.scopeSha256,
      receiptScopeSha256: current.scopeSha256,
      operationIdSha256: receipt.operation_id_sha256,
      phase: "completed",
      revision: current.revision + 1,
      statusCapabilitySha256: current.statusCapabilitySha256,
      retryableFailureCode: null,
      counts,
      deletedAt: new Date(input.deletedAt),
      receiptId: receipt.receipt_id,
      receiptSha256: receipt.receipt_sha256,
      createdAt: this.findTombstone(input.hashes)?.createdAt ?? new Date(input.deletedAt),
      updatedAt: new Date(input.deletedAt)
    };
    for (const hash of input.hashes.candidates) {
      this.tombstones.set(hash, {...cloneTombstone(completed), scopeSha256: hash});
    }
    this.operations.delete(current.scopeSha256);
    return receipt;
  }

  async statusByCapability(
    lookup: AccountDeletionStatusLookup
  ): Promise<AccountDeletionOperationRecord | AccountDeletionTombstone | undefined> {
    const operation = this.operations.get(lookup.scopeSha256);
    if (
      operation
      && operation.operationId === lookup.operationId
      && operation.statusCapabilitySha256 === lookup.statusCapabilitySha256
    ) return cloneOperation(operation);
    const tombstone = this.tombstones.get(lookup.scopeSha256);
    if (
      tombstone
      && tombstone.operationIdSha256 === sha256Text(lookup.operationId)
      && tombstone.statusCapabilitySha256 === lookup.statusCapabilitySha256
    ) return cloneTombstone(tombstone);
    return undefined;
  }

  async statusForScope(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    lookup: AccountDeletionStatusLookup
  ): Promise<AccountDeletionOperationRecord | AccountDeletionTombstone | undefined> {
    if (!hashes.candidates.includes(lookup.scopeSha256)) return undefined;
    const operation = this.findOperation(scope, hashes);
    if (
      operation
      && operation.operationId === lookup.operationId
      && operation.statusCapabilitySha256 === lookup.statusCapabilitySha256
    ) return cloneOperation(operation);
    return this.statusByCapability(lookup);
  }

  private findOperation(
    scope: AccessScope,
    hashes: AccountScopeHashes
  ): AccountDeletionOperationRecord | undefined {
    for (const hash of hashes.candidates) {
      const record = this.operations.get(hash);
      if (record && sameScope(record, scope)) return record;
    }
    return undefined;
  }

  private claim(
    current: AccountDeletionOperationRecord,
    claim: AccountDeletionLeaseClaim
  ): AccountDeletionLeaseClaimResult {
    if (current.phase === "prepared") return {kind: "busy", operation: cloneOperation(current)};
    const nowMs = claim.now.getTime();
    const leased = current.recoveryLeaseExpiresAt
      && current.recoveryLeaseExpiresAt.getTime() > nowMs
      && current.recoveryLeaseTokenSha256 !== claim.leaseTokenSha256;
    const delayed = !claim.ignoreRecoveryAfter
      && current.recoveryAfter
      && current.recoveryAfter.getTime() > nowMs;
    if (leased || delayed) return {kind: "busy", operation: cloneOperation(current)};
    const updated: AccountDeletionOperationRecord = {
      ...current,
      revision: current.revision + 1,
      retryableFailureCode: null,
      recoveryLeaseOwnerSha256: claim.leaseOwnerSha256,
      recoveryLeaseTokenSha256: claim.leaseTokenSha256,
      recoveryLeaseExpiresAt: new Date(nowMs + claim.leaseDurationMs),
      recoveryAfter: null,
      recoveryAttempts: current.recoveryAttempts + 1,
      updatedAt: new Date(claim.now)
    };
    this.operations.set(current.scopeSha256, updated);
    this.updateTombstonesForOperation(updated);
    return {kind: "claimed", operation: cloneOperation(updated)};
  }

  private findTombstone(
    hashes: AccountScopeHashes
  ): AccountDeletionTombstone | undefined {
    for (const hash of hashes.candidates) {
      const record = this.tombstones.get(hash);
      if (record) return record;
    }
    return undefined;
  }

  private updateTombstonesForOperation(record: AccountDeletionOperationRecord): void {
    for (const [hash, tombstone] of this.tombstones) {
      if (tombstone.operationIdSha256 !== sha256Text(record.operationId)) continue;
      this.tombstones.set(hash, {
        ...tombstone,
        phase: record.phase === "prepared" ? "fencing" : record.phase,
        revision: record.revision,
        retryableFailureCode: record.retryableFailureCode,
        counts: normalizedDeletionCounts(record.counts),
        updatedAt: new Date(record.updatedAt)
      });
    }
  }
}
