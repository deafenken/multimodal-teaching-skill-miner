import {Inject, Injectable} from "@nestjs/common";

import type {TenantDatabasePort, TenantTransaction} from "../database/tenant-database.port";
import {TENANT_DATABASE} from "../platform/tokens";
import type {AccessScope} from "../tenancy/access-scope";
import {
  accountDeletionReceipt,
  normalizedDeletionCounts,
  sha256Text
} from "./account-deletion-receipt";
import {AccountDataRightsError} from "./account-data-rights.errors";
import type {
  AccountDataRightsRepositoryPort,
  AccountDeletionLeaseClaim,
  AccountDeletionLeaseClaimResult,
  AccountDeletionStatusLookup,
  AccountScopeLifecycleState,
  AdvanceAccountDeletionInput,
  BeginAccountDeletionInput,
  BeginAccountDeletionResult,
  CommitAccountDeletionInput,
  PrepareAccountDeletionInput,
  SealAccountDeletionTombstoneInput
} from "./account-data-rights.repository.port";
import {ACCOUNT_SYSTEM_DATABASE} from "./account-data-rights.tokens";
import {
  ACCOUNT_DELETION_OPERATION_SCHEMA,
  type AccountDeletionCounts,
  type AccountDeletionOperationRecord,
  type AccountDeletionPhase,
  type AccountDeletionReceipt,
  type AccountDeletionTombstone,
  type AccountPostgresExportSnapshot,
  type AccountScopeHashes
} from "./account-data-rights.types";
import type {
  AccountSystemDatabasePort,
  AccountSystemTransaction
} from "./account-system-database.port";

interface ClockRow {
  database_now: Date | string;
}

interface OperationRow {
  tenant_id: string;
  owner_id: string;
  scope_sha256: string;
  operation_id: string;
  phase: Exclude<AccountDeletionPhase, "completed">;
  revision: number;
  challenge_id: string;
  challenge_token_sha256: string;
  challenge_csrf_sha256: string;
  challenge_session_sha256: string;
  challenge_authority_grant_sha256: string;
  challenge_canonical_identity_sha256: string;
  challenge_issuer_sha256: string;
  challenge_authority_key_version: string;
  challenge_authenticated_at: Date | string;
  challenge_assurance_level: number;
  challenge_expires_at: Date | string;
  status_capability_sha256: string;
  idempotency_key_sha256: string | null;
  confirmation_sha256: string | null;
  retryable_failure_code: string | null;
  recovery_lease_owner_sha256: string | null;
  recovery_lease_token_sha256: string | null;
  recovery_lease_expires_at: Date | string | null;
  recovery_after: Date | string | null;
  recovery_attempts: number;
  postgres_event_count: number;
  postgres_task_count: number;
  postgres_artifact_count: number;
  postgres_session_count: number;
  postgres_auth_session_count: number;
  worker_file_count: number;
  worker_byte_count: number | string;
  worker_root_count: number;
  created_at: Date | string;
  updated_at: Date | string;
}

interface TombstoneRow {
  scope_sha256: string;
  receipt_scope_sha256: string;
  operation_id_sha256: string;
  phase: Exclude<AccountDeletionPhase, "prepared" | "fencing" | "draining">;
  revision: number;
  status_capability_sha256: string;
  retryable_failure_code: string | null;
  postgres_event_count: number;
  postgres_task_count: number;
  postgres_artifact_count: number;
  postgres_session_count: number;
  postgres_auth_session_count: number;
  worker_file_count: number;
  worker_byte_count: number | string;
  worker_root_count: number;
  deleted_at: Date | string | null;
  receipt_id: string | null;
  receipt_sha256: string | null;
  created_at: Date | string;
  updated_at: Date | string;
}

interface ExportSnapshotRow {
  captured_at: Date | string;
  sessions: unknown;
  tasks: unknown;
  events: unknown;
  auth_session_audit: unknown;
  artifacts?: unknown;
}

const OPERATION_COLUMNS = `
  tenant_id, owner_id, scope_sha256, operation_id, phase, revision,
  challenge_id, challenge_token_sha256, challenge_csrf_sha256,
  challenge_session_sha256, challenge_authority_grant_sha256,
  challenge_canonical_identity_sha256, challenge_issuer_sha256,
  challenge_authority_key_version, challenge_authenticated_at,
  challenge_assurance_level, challenge_expires_at, status_capability_sha256,
  idempotency_key_sha256, confirmation_sha256, retryable_failure_code,
  recovery_lease_owner_sha256, recovery_lease_token_sha256,
  recovery_lease_expires_at, recovery_after, recovery_attempts,
  postgres_event_count, postgres_task_count, postgres_artifact_count,
  postgres_session_count,
  postgres_auth_session_count, worker_file_count, worker_byte_count,
  worker_root_count, created_at, updated_at
`;

const TOMBSTONE_COLUMNS = `
  scope_sha256, receipt_scope_sha256, operation_id_sha256, phase, revision,
  status_capability_sha256, retryable_failure_code,
  postgres_event_count, postgres_task_count, postgres_artifact_count,
  postgres_session_count,
  postgres_auth_session_count, worker_file_count, worker_byte_count,
  worker_root_count, deleted_at, receipt_id, receipt_sha256,
  created_at, updated_at
`;

function timestamp(value: Date | string): Date {
  const parsed = value instanceof Date ? new Date(value) : new Date(value);
  if (!Number.isFinite(parsed.getTime())) throw new Error("Invalid account lifecycle time");
  return parsed;
}

function count(value: number | string): number {
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new Error("Invalid account lifecycle count");
  }
  return parsed;
}

function counts(row: OperationRow | TombstoneRow): AccountDeletionCounts {
  return normalizedDeletionCounts({
    postgresEvents: count(row.postgres_event_count),
    postgresTasks: count(row.postgres_task_count),
    postgresArtifacts: count(row.postgres_artifact_count),
    postgresSessions: count(row.postgres_session_count),
    postgresAuthSessions: count(row.postgres_auth_session_count),
    workerFiles: count(row.worker_file_count),
    workerBytes: count(row.worker_byte_count),
    workerRoots: count(row.worker_root_count)
  });
}

function operation(row: OperationRow): AccountDeletionOperationRecord {
  return {
    schema: ACCOUNT_DELETION_OPERATION_SCHEMA,
    tenantId: row.tenant_id,
    ownerId: row.owner_id,
    operationId: row.operation_id,
    scopeSha256: row.scope_sha256,
    phase: row.phase,
    revision: Number(row.revision),
    challengeId: row.challenge_id,
    challengeTokenSha256: row.challenge_token_sha256,
    challengeCsrfSha256: row.challenge_csrf_sha256,
    challengeSessionSha256: row.challenge_session_sha256,
    challengeAuthorityGrantSha256: row.challenge_authority_grant_sha256,
    challengeCanonicalIdentitySha256: row.challenge_canonical_identity_sha256,
    challengeIssuerSha256: row.challenge_issuer_sha256,
    challengeAuthorityKeyVersion: row.challenge_authority_key_version,
    challengeAuthenticatedAt: timestamp(row.challenge_authenticated_at),
    challengeAssuranceLevel: Number(row.challenge_assurance_level),
    challengeExpiresAt: timestamp(row.challenge_expires_at),
    statusCapabilitySha256: row.status_capability_sha256,
    idempotencyKeySha256: row.idempotency_key_sha256,
    confirmationSha256: row.confirmation_sha256,
    retryableFailureCode: row.retryable_failure_code,
    recoveryLeaseOwnerSha256: row.recovery_lease_owner_sha256,
    recoveryLeaseTokenSha256: row.recovery_lease_token_sha256,
    recoveryLeaseExpiresAt: row.recovery_lease_expires_at
      ? timestamp(row.recovery_lease_expires_at)
      : null,
    recoveryAfter: row.recovery_after ? timestamp(row.recovery_after) : null,
    recoveryAttempts: count(row.recovery_attempts),
    counts: counts(row),
    createdAt: timestamp(row.created_at),
    updatedAt: timestamp(row.updated_at)
  };
}

function tombstone(row: TombstoneRow): AccountDeletionTombstone {
  return {
    scopeSha256: row.scope_sha256,
    receiptScopeSha256: row.receipt_scope_sha256,
    operationIdSha256: row.operation_id_sha256,
    phase: row.phase,
    revision: Number(row.revision),
    statusCapabilitySha256: row.status_capability_sha256,
    retryableFailureCode: row.retryable_failure_code,
    counts: counts(row),
    deletedAt: row.deleted_at ? timestamp(row.deleted_at) : null,
    receiptId: row.receipt_id,
    receiptSha256: row.receipt_sha256,
    createdAt: timestamp(row.created_at),
    updatedAt: timestamp(row.updated_at)
  };
}

function jsonRows(value: unknown, name: string): Record<string, unknown>[] {
  const parsed = typeof value === "string" ? JSON.parse(value) : value;
  if (!Array.isArray(parsed) || parsed.some((item) =>
    item === null || typeof item !== "object" || Array.isArray(item)
  )) throw new Error(`Invalid PostgreSQL account export ${name}`);
  return parsed as Record<string, unknown>[];
}

async function databaseNow(
  transaction: Pick<TenantTransaction, "query"> | AccountSystemTransaction
): Promise<Date> {
  const result = await transaction.query<ClockRow>(
    "SELECT statement_timestamp() AS database_now"
  );
  return timestamp(result.rows[0]?.database_now ?? "invalid");
}

async function selectedOperation(
  transaction: TenantTransaction,
  scope: AccessScope,
  lock = false
): Promise<OperationRow | undefined> {
  const result = await transaction.query<OperationRow>(
    `SELECT ${OPERATION_COLUMNS}
       FROM public.teachlab_account_deletion_operations
      WHERE tenant_id = $1 AND owner_id = $2
      ${lock ? "FOR UPDATE" : ""}`,
    [scope.tenantId, scope.ownerId]
  );
  return result.rows[0];
}

async function selectedTombstone(
  transaction: Pick<TenantTransaction, "query"> | AccountSystemTransaction,
  hashes: readonly string[],
  lock = false
): Promise<TombstoneRow | undefined> {
  const result = await transaction.query<TombstoneRow>(
    `SELECT ${TOMBSTONE_COLUMNS}
       FROM public.teachlab_account_deletion_tombstones
      WHERE scope_sha256 = ANY($1::text[])
      ORDER BY CASE WHEN phase = 'completed' THEN 0 ELSE 1 END, updated_at DESC
      LIMIT 1
      ${lock ? "FOR UPDATE" : ""}`,
    [hashes]
  );
  return result.rows[0];
}

@Injectable()
export class PostgresAccountDataRightsRepository
  implements AccountDataRightsRepositoryPort {
  constructor(
    @Inject(TENANT_DATABASE) private readonly database: TenantDatabasePort,
    @Inject(ACCOUNT_SYSTEM_DATABASE)
    private readonly systemDatabase: AccountSystemDatabasePort
  ) {}

  lifecycle(
    scope: AccessScope,
    hashes: AccountScopeHashes
  ): Promise<AccountScopeLifecycleState> {
    return this.withAccountTenant(scope, async (transaction) => {
      const current = await selectedOperation(transaction, scope);
      if (current && current.phase !== "prepared") {
        return {kind: "deleting", operation: operation(current)};
      }
      const deleted = await selectedTombstone(transaction, hashes.candidates);
      if (deleted) return {kind: "deleted", tombstone: tombstone(deleted)};
      return {kind: "active"};
    });
  }

  exportPostgresSnapshot(
    scope: AccessScope,
    hashes: AccountScopeHashes
  ): Promise<AccountPostgresExportSnapshot> {
    return this.withAccountTenant(scope, async (transaction) => {
      const current = await selectedOperation(transaction, scope);
      if (current && current.phase !== "prepared") {
        throw new AccountDataRightsError(409, "account_deletion_already_started");
      }
      if (await selectedTombstone(transaction, hashes.candidates)) {
        throw new AccountDataRightsError(410, "account_already_deleted");
      }
      // One statement gives every CTE the same MVCC snapshot without buffering
      // the eventual ZIP. Tenant/owner columns are deliberately omitted.
      const result = await transaction.query<ExportSnapshotRow>(`
        SELECT statement_timestamp() AS captured_at,
          COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
              'id', id, 'title', title, 'learner', learner, 'status', status,
              'round', round, 'created_at', created_at, 'updated_at', updated_at,
              'version', version
            ) ORDER BY updated_at, id)
              FROM public.teachlab_teaching_sessions
             WHERE tenant_id = $1 AND owner_id = $2
          ), '[]'::jsonb) AS sessions,
          COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
              'id', id, 'session_id', session_id,
              'learner_message', learner_message,
              'client_request_id', client_request_id, 'status', status,
              'failure_code', failure_code, 'created_at', created_at,
              'updated_at', updated_at, 'version', version
            ) ORDER BY created_at, id)
              FROM public.teachlab_agent_tasks
             WHERE tenant_id = $1 AND owner_id = $2
          ), '[]'::jsonb) AS tasks,
          COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
              'sequence_id', sequence_id, 'id', id, 'session_id', session_id,
              'event_type', event_type, 'payload', payload,
              'occurred_at', occurred_at
            ) ORDER BY sequence_id)
              FROM public.teachlab_task_events
             WHERE tenant_id = $1 AND owner_id = $2
          ), '[]'::jsonb) AS events,
          COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
              'session_id_sha256', session_id_sha256,
              'issued_at', issued_at, 'expires_at', expires_at,
              'revoked_at', revoked_at,
              'revocation_reason', revocation_reason,
              'version', version
            ) ORDER BY issued_at, session_id_sha256)
              FROM public.teachlab_auth_sessions
             WHERE tenant_id = $1 AND owner_id = $2
          ), '[]'::jsonb) AS auth_session_audit,
          COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
              'key', artifact_key, 'content_type', content_type,
              'byte_length', byte_length, 'sha256', sha256,
              'metadata', metadata, 'created_at', created_at,
              'updated_at', updated_at, 'version', version
            ) ORDER BY artifact_key)
              FROM public.teachlab_artifacts
             WHERE tenant_id = $1 AND owner_id = $2
          ), '[]'::jsonb) AS artifacts
      `, [scope.tenantId, scope.ownerId]);
      const row = result.rows[0];
      if (!row) throw new Error("PostgreSQL account export snapshot is unavailable");
      return {
        capturedAt: timestamp(row.captured_at).toISOString(),
        sessions: jsonRows(row.sessions, "sessions"),
        tasks: jsonRows(row.tasks, "tasks"),
        events: jsonRows(row.events, "events"),
        authSessionAudit: jsonRows(row.auth_session_audit, "auth audit"),
        artifacts: jsonRows(row.artifacts ?? [], "artifacts")
      };
    });
  }

  prepareDeletion(
    input: PrepareAccountDeletionInput
  ): Promise<AccountDeletionOperationRecord> {
    return this.withAccountTenant(input, async (transaction) => {
      if (await selectedTombstone(transaction, input.hashes.candidates, true)) {
        throw new AccountDataRightsError(410, "account_already_deleted");
      }
      const now = await databaseNow(transaction);
      const current = await selectedOperation(transaction, input, true);
      if (current && current.phase !== "prepared") {
        throw new AccountDataRightsError(409, "account_deletion_already_started");
      }
      const revision = (current?.revision ?? 0) + 1;
      const result = current
        ? await transaction.query<OperationRow>(
          `UPDATE public.teachlab_account_deletion_operations
              SET scope_sha256 = $3, operation_id = $4, phase = 'prepared',
                  revision = $5, challenge_id = $6,
                  challenge_token_sha256 = $7, challenge_csrf_sha256 = $8,
                  challenge_session_sha256 = $9,
                  challenge_authority_grant_sha256 = $10,
                  challenge_canonical_identity_sha256 = $11,
                  challenge_issuer_sha256 = $12,
                  challenge_authority_key_version = $13,
                  challenge_authenticated_at = $14,
                  challenge_assurance_level = $15,
                  challenge_expires_at = $16,
                  status_capability_sha256 = $17,
                  idempotency_key_sha256 = NULL, confirmation_sha256 = NULL,
                  retryable_failure_code = NULL,
                  recovery_lease_owner_sha256 = NULL,
                  recovery_lease_token_sha256 = NULL,
                  recovery_lease_expires_at = NULL, recovery_after = NULL,
                  recovery_attempts = 0, updated_at = $18
            WHERE tenant_id = $1 AND owner_id = $2
            RETURNING ${OPERATION_COLUMNS}`,
          [
            input.tenantId, input.ownerId, input.hashes.active, input.operationId,
            revision, input.challengeId, input.challengeTokenSha256,
            input.challengeCsrfSha256, input.challengeSessionSha256,
            input.challengeAuthorityGrantSha256,
            input.challengeCanonicalIdentitySha256,
            input.challengeIssuerSha256, input.challengeAuthorityKeyVersion,
            input.challengeAuthenticatedAt, input.challengeAssuranceLevel,
            input.challengeExpiresAt, input.statusCapabilitySha256, now
          ]
        )
        : await transaction.query<OperationRow>(
          `INSERT INTO public.teachlab_account_deletion_operations (
             tenant_id, owner_id, scope_sha256, operation_id, phase, revision,
             challenge_id, challenge_token_sha256, challenge_csrf_sha256,
             challenge_session_sha256, challenge_authority_grant_sha256,
             challenge_canonical_identity_sha256, challenge_issuer_sha256,
             challenge_authority_key_version, challenge_authenticated_at,
             challenge_assurance_level, challenge_expires_at,
             status_capability_sha256, created_at, updated_at
           ) VALUES (
             $1, $2, $3, $4, 'prepared', $5, $6, $7, $8, $9, $10, $11,
             $12, $13, $14, $15, $16, $17, $18, $18
           )
           RETURNING ${OPERATION_COLUMNS}`,
          [
            input.tenantId, input.ownerId, input.hashes.active, input.operationId,
            revision, input.challengeId, input.challengeTokenSha256,
            input.challengeCsrfSha256, input.challengeSessionSha256,
            input.challengeAuthorityGrantSha256,
            input.challengeCanonicalIdentitySha256,
            input.challengeIssuerSha256, input.challengeAuthorityKeyVersion,
            input.challengeAuthenticatedAt, input.challengeAssuranceLevel,
            input.challengeExpiresAt, input.statusCapabilitySha256, now
          ]
        );
      const row = result.rows[0];
      if (!row) throw new Error("PostgreSQL deletion challenge was not stored");
      return operation(row);
    });
  }

  beginDeletion(
    input: BeginAccountDeletionInput
  ): Promise<BeginAccountDeletionResult> {
    return this.withAccountTenant(input, async (transaction) => {
      const now = await databaseNow(transaction);
      const current = await selectedOperation(transaction, input, true);
      if (!current || current.operation_id !== input.operationId) return {kind: "missing"};
      if (current.phase !== "prepared") {
        return current.idempotency_key_sha256 === input.idempotencyKeySha256
          && current.confirmation_sha256 === input.confirmationSha256
          ? {kind: "idempotent", operation: operation(current)}
          : {kind: "payload_conflict"};
      }
      if (current.revision !== input.expectedRevision) return {kind: "revision_conflict"};
      if (timestamp(current.challenge_expires_at).getTime() <= now.getTime()) {
        return {kind: "expired"};
      }
      if (
        current.challenge_id !== input.challengeId
        || current.challenge_token_sha256 !== input.challengeTokenSha256
        || current.challenge_csrf_sha256 !== input.challengeCsrfSha256
        || current.challenge_session_sha256 !== input.challengeSessionSha256
        || current.challenge_authority_grant_sha256
          !== input.challengeAuthorityGrantSha256
        || current.challenge_canonical_identity_sha256
          !== input.challengeCanonicalIdentitySha256
        || current.challenge_issuer_sha256 !== input.challengeIssuerSha256
        || current.challenge_authority_key_version
          !== input.challengeAuthorityKeyVersion
        || timestamp(current.challenge_authenticated_at).getTime()
          !== input.challengeAuthenticatedAt.getTime()
        || Number(current.challenge_assurance_level)
          !== input.challengeAssuranceLevel
      ) return {kind: "payload_conflict"};
      const result = await transaction.query<OperationRow>(
        `UPDATE public.teachlab_account_deletion_operations
            SET phase = 'fencing', revision = revision + 1,
                idempotency_key_sha256 = $5, confirmation_sha256 = $6,
                retryable_failure_code = NULL,
                recovery_lease_owner_sha256 = NULL,
                recovery_lease_token_sha256 = NULL,
                recovery_lease_expires_at = NULL, recovery_after = NULL,
                updated_at = $7
          WHERE tenant_id = $1 AND owner_id = $2 AND operation_id = $3
            AND phase = 'prepared' AND revision = $4
          RETURNING ${OPERATION_COLUMNS}`,
        [
          input.tenantId, input.ownerId, input.operationId, input.expectedRevision,
          input.idempotencyKeySha256, input.confirmationSha256, now
        ]
      );
      return result.rows[0]
        ? {kind: "started", operation: operation(result.rows[0])}
        : {kind: "revision_conflict"};
    });
  }

  claimDeletionForScope(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    claim: AccountDeletionLeaseClaim
  ): Promise<AccountDeletionLeaseClaimResult> {
    return this.withAccountTenant(scope, async (transaction) => {
      const currentRow = await selectedOperation(transaction, scope, true);
      if (!currentRow || currentRow.operation_id !== operationId) {
        const completed = await selectedTombstone(transaction, hashes.candidates, true);
        return completed?.phase === "completed"
          ? {kind: "completed", tombstone: tombstone(completed)}
          : {kind: "missing"};
      }
      const result = await this.claimOperation(transaction, currentRow, claim);
      if (result.kind === "claimed") {
        await this.updateTombstones(
          transaction, hashes, result.operation, result.operation.updatedAt
        );
      }
      return result;
    });
  }

  claimDeletionByCapability(
    lookup: AccountDeletionStatusLookup,
    claim: AccountDeletionLeaseClaim
  ): Promise<AccountDeletionLeaseClaimResult> {
    return this.systemDatabase.withAccountRecovery(async (transaction) => {
      const selected = await transaction.query<OperationRow>(
        `SELECT ${OPERATION_COLUMNS}
           FROM public.teachlab_account_deletion_operations
          WHERE scope_sha256 = $1 AND operation_id = $2
            AND status_capability_sha256 = $3
          FOR UPDATE`,
        [lookup.scopeSha256, lookup.operationId, lookup.statusCapabilitySha256]
      );
      const current = selected.rows[0];
      if (current) return this.claimOperation(transaction, current, claim);
      const deleted = await transaction.query<TombstoneRow>(
        `SELECT ${TOMBSTONE_COLUMNS}
           FROM public.teachlab_account_deletion_tombstones
          WHERE scope_sha256 = $1 AND operation_id_sha256 = $2
            AND status_capability_sha256 = $3`,
        [
          lookup.scopeSha256,
          sha256Text(lookup.operationId),
          lookup.statusCapabilitySha256
        ]
      );
      return deleted.rows[0]
        ? {kind: "completed", tombstone: tombstone(deleted.rows[0])}
        : {kind: "missing"};
    });
  }

  claimNextDeletion(
    claim: AccountDeletionLeaseClaim
  ): Promise<AccountDeletionOperationRecord | undefined> {
    return this.systemDatabase.withAccountRecovery(async (transaction) => {
      const now = await databaseNow(transaction);
      const selected = await transaction.query<OperationRow>(
        `SELECT ${OPERATION_COLUMNS}
           FROM public.teachlab_account_deletion_operations
          WHERE phase <> 'prepared'
            AND (recovery_after IS NULL OR recovery_after <= $1)
            AND (recovery_lease_expires_at IS NULL OR recovery_lease_expires_at <= $1)
          ORDER BY recovery_after ASC NULLS FIRST, updated_at ASC, operation_id ASC
          LIMIT 1
          FOR UPDATE SKIP LOCKED`,
        [now]
      );
      const current = selected.rows[0];
      if (!current) return undefined;
      const result = await this.claimOperation(transaction, current, {...claim, now});
      return result.kind === "claimed" ? result.operation : undefined;
    });
  }

  renewDeletionLease(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    leaseTokenSha256: string,
    leaseDurationMs: number,
    _now: Date
  ): Promise<AccountDeletionOperationRecord | undefined> {
    this.assertLeaseInput(leaseTokenSha256, leaseDurationMs);
    return this.withAccountTenant(scope, async (transaction) => {
      const now = await databaseNow(transaction);
      const expiresAt = new Date(now.getTime() + leaseDurationMs);
      const result = await transaction.query<OperationRow>(
        `UPDATE public.teachlab_account_deletion_operations
            SET recovery_lease_expires_at = $5
          WHERE tenant_id = $1 AND owner_id = $2 AND operation_id = $3
            AND recovery_lease_token_sha256 = $4
            AND recovery_lease_expires_at > $6
          RETURNING ${OPERATION_COLUMNS}`,
        [scope.tenantId, scope.ownerId, operationId, leaseTokenSha256, expiresAt, now]
      );
      const row = result.rows[0];
      if (!row || !hashes.candidates.includes(row.scope_sha256)) return undefined;
      return operation(row);
    });
  }

  operationForLease(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    leaseTokenSha256: string,
    _now: Date
  ): Promise<AccountDeletionOperationRecord | undefined> {
    return this.withAccountTenant(scope, async (transaction) => {
      const now = await databaseNow(transaction);
      const result = await transaction.query<OperationRow>(
        `SELECT ${OPERATION_COLUMNS}
           FROM public.teachlab_account_deletion_operations
          WHERE tenant_id = $1 AND owner_id = $2 AND operation_id = $3
            AND recovery_lease_token_sha256 = $4
            AND recovery_lease_expires_at > $5`,
        [scope.tenantId, scope.ownerId, operationId, leaseTokenSha256, now]
      );
      const row = result.rows[0];
      if (!row || !hashes.candidates.includes(row.scope_sha256)) return undefined;
      return operation(row);
    });
  }

  advanceDeletion(
    input: AdvanceAccountDeletionInput
  ): Promise<AccountDeletionOperationRecord> {
    return this.withAccountTenant(input, async (transaction) => {
      const now = await databaseNow(transaction);
      const patch = normalizedDeletionCounts(input.counts);
      const result = await transaction.query<OperationRow>(
        `UPDATE public.teachlab_account_deletion_operations
            SET phase = $6, revision = revision + 1,
                postgres_event_count = GREATEST(postgres_event_count, $7),
                postgres_task_count = GREATEST(postgres_task_count, $8),
                postgres_artifact_count = GREATEST(postgres_artifact_count, $9),
                postgres_session_count = GREATEST(postgres_session_count, $10),
                postgres_auth_session_count = GREATEST(postgres_auth_session_count, $11),
                worker_file_count = GREATEST(worker_file_count, $12),
                worker_byte_count = GREATEST(worker_byte_count, $13),
                worker_root_count = GREATEST(worker_root_count, $14),
                retryable_failure_code = NULL, updated_at = $15
          WHERE tenant_id = $1 AND owner_id = $2 AND operation_id = $3
            AND phase = $4 AND revision = $5
            AND recovery_lease_token_sha256 = $16
            AND recovery_lease_expires_at > $15
          RETURNING ${OPERATION_COLUMNS}`,
        [
          input.tenantId, input.ownerId, input.operationId, input.expectedPhase,
          input.expectedRevision, input.nextPhase, patch.postgresEvents,
          patch.postgresTasks, patch.postgresArtifacts, patch.postgresSessions,
          patch.postgresAuthSessions, patch.workerFiles, patch.workerBytes,
          patch.workerRoots, now,
          input.leaseTokenSha256
        ]
      );
      const row = result.rows[0];
      if (!row) {
        throw new AccountDataRightsError(409, "account_deletion_revision_conflict");
      }
      const updated = operation(row);
      await this.updateTombstones(transaction, input.hashes, updated, now);
      return updated;
    });
  }

  sealTombstone(
    input: SealAccountDeletionTombstoneInput
  ): Promise<AccountDeletionTombstone> {
    return this.withAccountTenant(input, async (transaction) => {
      const now = await databaseNow(transaction);
      const changed = await transaction.query<OperationRow>(
        `UPDATE public.teachlab_account_deletion_operations
            SET phase = 'tombstoned', revision = revision + 1,
                retryable_failure_code = NULL, updated_at = $5
          WHERE tenant_id = $1 AND owner_id = $2 AND operation_id = $3
            AND phase = 'draining' AND revision = $4
            AND recovery_lease_token_sha256 = $6
            AND recovery_lease_expires_at > $5
          RETURNING ${OPERATION_COLUMNS}`,
        [
          input.tenantId, input.ownerId, input.operationId,
          input.expectedRevision, now, input.leaseTokenSha256
        ]
      );
      const row = changed.rows[0];
      if (!row) {
        throw new AccountDataRightsError(409, "account_deletion_revision_conflict");
      }
      const current = operation(row);
      for (const hash of input.hashes.candidates) {
        await transaction.query(
          `INSERT INTO public.teachlab_account_deletion_tombstones (
             scope_sha256, receipt_scope_sha256, operation_id_sha256, phase, revision,
             status_capability_sha256, created_at, updated_at
           ) VALUES ($1, $2, $3, 'tombstoned', $4, $5, $6, $6)
           ON CONFLICT (scope_sha256) DO UPDATE
             SET receipt_scope_sha256 = EXCLUDED.receipt_scope_sha256,
                 operation_id_sha256 = EXCLUDED.operation_id_sha256,
                 phase = EXCLUDED.phase, revision = EXCLUDED.revision,
                 status_capability_sha256 = EXCLUDED.status_capability_sha256,
                 retryable_failure_code = NULL, deleted_at = NULL,
                 receipt_id = NULL, receipt_sha256 = NULL,
                 updated_at = EXCLUDED.updated_at`,
          [hash, current.scopeSha256, sha256Text(current.operationId), current.revision,
            current.statusCapabilitySha256, now]
        );
      }
      const sealed = await selectedTombstone(transaction, input.hashes.candidates);
      if (!sealed) throw new Error("PostgreSQL deletion tombstone was not stored");
      return tombstone(sealed);
    });
  }

  markRetryableFailure(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    operationId: string,
    leaseTokenSha256: string,
    failureCode: string,
    retryAt: Date,
    _now: Date
  ): Promise<void> {
    if (!/^[a-z][a-z0-9_]{2,63}$/.test(failureCode)) {
      throw new Error("Invalid content-free account deletion failure code");
    }
    return this.withAccountTenant(scope, async (transaction) => {
      const now = await databaseNow(transaction);
      const result = await transaction.query<OperationRow>(
        `UPDATE public.teachlab_account_deletion_operations
            SET retryable_failure_code = $4, revision = revision + 1,
                recovery_lease_owner_sha256 = NULL,
                recovery_lease_token_sha256 = NULL,
                recovery_lease_expires_at = NULL,
                recovery_after = GREATEST(
                  $6::timestamptz, $5::timestamptz
                ), updated_at = $5
          WHERE tenant_id = $1 AND owner_id = $2 AND operation_id = $3
            AND phase <> 'prepared'
            AND recovery_lease_token_sha256 = $7
          RETURNING ${OPERATION_COLUMNS}`,
        [
          scope.tenantId, scope.ownerId, operationId, failureCode, now,
          retryAt, leaseTokenSha256
        ]
      );
      if (result.rows[0]) {
        await this.updateTombstones(
          transaction, hashes, operation(result.rows[0]), now
        );
      }
    });
  }

  commitDeletion(
    input: CommitAccountDeletionInput
  ): Promise<AccountDeletionReceipt> {
    return this.withAccountTenant(input, async (transaction) => {
      const now = await databaseNow(transaction);
      const currentRow = await selectedOperation(transaction, input, true);
      if (
        !currentRow
        || currentRow.operation_id !== input.operationId
        || currentRow.phase !== "committing_database"
        || currentRow.revision !== input.expectedRevision
        || currentRow.recovery_lease_token_sha256 !== input.leaseTokenSha256
        || !currentRow.recovery_lease_expires_at
        || timestamp(currentRow.recovery_lease_expires_at).getTime() <= now.getTime()
      ) {
        const prior = await selectedTombstone(transaction, input.hashes.candidates, true);
        if (
          prior?.phase === "completed"
          && prior.receipt_id
          && prior.deleted_at
          && prior.operation_id_sha256 === sha256Text(input.operationId)
        ) {
          return accountDeletionReceipt({
            receiptId: prior.receipt_id,
            scopeSha256: prior.receipt_scope_sha256,
            operationId: input.operationId,
            deletedAt: timestamp(prior.deleted_at),
            counts: counts(prior)
          });
        }
        throw new AccountDataRightsError(409, "account_deletion_revision_conflict");
      }
      const current = operation(currentRow);
      const eventResult = await transaction.query(
        `DELETE FROM public.teachlab_task_events
          WHERE tenant_id = $1 AND owner_id = $2`,
        [input.tenantId, input.ownerId]
      );
      const taskResult = await transaction.query(
        `DELETE FROM public.teachlab_agent_tasks
          WHERE tenant_id = $1 AND owner_id = $2`,
        [input.tenantId, input.ownerId]
      );
      // Artifact bytes share the same tenant transaction and are removed only
      // after child events/tasks, before parent sessions/auth authority.
      const artifactResult = await transaction.query(
        `DELETE FROM public.teachlab_artifacts
          WHERE tenant_id = $1 AND owner_id = $2`,
        [input.tenantId, input.ownerId]
      );
      const sessionResult = await transaction.query(
        `DELETE FROM public.teachlab_teaching_sessions
          WHERE tenant_id = $1 AND owner_id = $2`,
        [input.tenantId, input.ownerId]
      );
      const authResult = await transaction.query(
        `DELETE FROM public.teachlab_auth_sessions
          WHERE tenant_id = $1 AND owner_id = $2`,
        [input.tenantId, input.ownerId]
      );
      const finalCounts = normalizedDeletionCounts({
        postgresEvents: eventResult.rowCount,
        postgresTasks: taskResult.rowCount,
        postgresArtifacts: artifactResult.rowCount,
        postgresSessions: sessionResult.rowCount,
        postgresAuthSessions: authResult.rowCount,
        ...input.workerCounts
      });
      // The deletion timestamp is transaction-authoritative. A caller clock
      // cannot backdate/forward-date the terminal receipt.
      const deletedAt = now;
      const receipt = accountDeletionReceipt({
        receiptId: input.receiptId,
        scopeSha256: current.scopeSha256,
        operationId: current.operationId,
        deletedAt,
        counts: finalCounts
      });
      // A scope-HMAC rotation may have happened after the first tombstone was
      // sealed. Materialize every currently known digest in this same final
      // transaction so the deleted account cannot reopen once an older key is
      // retired. A digest already owned by another operation fails closed.
      for (const hash of input.hashes.candidates) {
        const reconciled = await transaction.query<{scope_sha256: string}>(
          `INSERT INTO public.teachlab_account_deletion_tombstones (
             scope_sha256, receipt_scope_sha256, operation_id_sha256,
             phase, revision, status_capability_sha256,
             postgres_event_count, postgres_task_count, postgres_artifact_count,
             postgres_session_count, postgres_auth_session_count,
             worker_file_count, worker_byte_count, worker_root_count,
             created_at, updated_at
           ) VALUES (
             $1, $2, $3, 'committing_database', $4, $5,
             $6, $7, $8, $9, $10, $11, $12, $13, $14, $14
           )
           ON CONFLICT (scope_sha256) DO UPDATE
             SET receipt_scope_sha256 = EXCLUDED.receipt_scope_sha256,
                 phase = EXCLUDED.phase, revision = EXCLUDED.revision,
                 status_capability_sha256 = EXCLUDED.status_capability_sha256,
                 retryable_failure_code = NULL,
                 postgres_event_count = EXCLUDED.postgres_event_count,
                 postgres_task_count = EXCLUDED.postgres_task_count,
                 postgres_artifact_count = EXCLUDED.postgres_artifact_count,
                 postgres_session_count = EXCLUDED.postgres_session_count,
                 postgres_auth_session_count = EXCLUDED.postgres_auth_session_count,
                 worker_file_count = EXCLUDED.worker_file_count,
                 worker_byte_count = EXCLUDED.worker_byte_count,
                 worker_root_count = EXCLUDED.worker_root_count,
                 deleted_at = NULL, receipt_id = NULL, receipt_sha256 = NULL,
                 updated_at = EXCLUDED.updated_at
             WHERE teachlab_account_deletion_tombstones.operation_id_sha256
                     = EXCLUDED.operation_id_sha256
               AND teachlab_account_deletion_tombstones.phase <> 'completed'
           RETURNING scope_sha256`,
          [
            hash, current.scopeSha256, receipt.operation_id_sha256,
            current.revision, current.statusCapabilitySha256,
            finalCounts.postgresEvents, finalCounts.postgresTasks,
            finalCounts.postgresArtifacts, finalCounts.postgresSessions,
            finalCounts.postgresAuthSessions,
            finalCounts.workerFiles, finalCounts.workerBytes,
            finalCounts.workerRoots, deletedAt
          ]
        );
        if (reconciled.rowCount !== 1) {
          throw new Error("PostgreSQL deletion tombstone rotation conflict");
        }
      }
      // Select by the random operation digest, not only by the keys currently
      // configured. That finalizes every tombstone sealed before a scope-key
      // rotation, while requiring at least one exact sealed row.
      const updated = await transaction.query<TombstoneRow>(
        `UPDATE public.teachlab_account_deletion_tombstones
              SET phase = 'completed', revision = $1,
                  retryable_failure_code = NULL,
                  postgres_event_count = $2, postgres_task_count = $3,
                  postgres_artifact_count = $4,
                  postgres_session_count = $5,
                  postgres_auth_session_count = $6,
                  worker_file_count = $7, worker_byte_count = $8,
                  worker_root_count = $9, deleted_at = $10,
                  receipt_id = $11, receipt_sha256 = $12, updated_at = $10
            WHERE operation_id_sha256 = $13
              AND phase <> 'completed'
            RETURNING ${TOMBSTONE_COLUMNS}`,
        [
          current.revision + 1, finalCounts.postgresEvents,
          finalCounts.postgresTasks, finalCounts.postgresArtifacts,
          finalCounts.postgresSessions, finalCounts.postgresAuthSessions,
          finalCounts.workerFiles, finalCounts.workerBytes,
          finalCounts.workerRoots, deletedAt, receipt.receipt_id,
          receipt.receipt_sha256,
          receipt.operation_id_sha256
        ]
      );
      if (updated.rowCount < 1) {
        throw new Error("PostgreSQL deletion tombstone finalization was not exact");
      }
      const removed = await transaction.query(
        `DELETE FROM public.teachlab_account_deletion_operations
          WHERE tenant_id = $1 AND owner_id = $2 AND operation_id = $3
            AND phase = 'committing_database' AND revision = $4`,
        [input.tenantId, input.ownerId, input.operationId, input.expectedRevision]
      );
      if (removed.rowCount !== 1) {
        throw new Error("PostgreSQL deletion operation finalization was not exact");
      }
      return receipt;
    });
  }

  statusByCapability(
    lookup: AccountDeletionStatusLookup
  ): Promise<AccountDeletionOperationRecord | AccountDeletionTombstone | undefined> {
    // The signed HttpOnly capability is deliberately sufficient across an API
    // restart, including before the tombstone exists. The recovery transaction
    // can see raw scope only inside this repository; HTTP receives the bounded
    // content-free projection.
    return this.systemDatabase.withAccountRecovery(async (transaction) => {
      const active = await transaction.query<OperationRow>(
        `SELECT ${OPERATION_COLUMNS}
           FROM public.teachlab_account_deletion_operations
          WHERE scope_sha256 = $1 AND operation_id = $2
            AND status_capability_sha256 = $3`,
        [lookup.scopeSha256, lookup.operationId, lookup.statusCapabilitySha256]
      );
      if (active.rows[0]) return operation(active.rows[0]);
      const result = await transaction.query<TombstoneRow>(
        `SELECT ${TOMBSTONE_COLUMNS}
           FROM public.teachlab_account_deletion_tombstones
          WHERE scope_sha256 = $1 AND operation_id_sha256 = $2
            AND status_capability_sha256 = $3`,
        [
          lookup.scopeSha256,
          sha256Text(lookup.operationId),
          lookup.statusCapabilitySha256
        ]
      );
      return result.rows[0] ? tombstone(result.rows[0]) : undefined;
    });
  }

  statusForScope(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    lookup: AccountDeletionStatusLookup
  ): Promise<AccountDeletionOperationRecord | AccountDeletionTombstone | undefined> {
    if (!hashes.candidates.includes(lookup.scopeSha256)) {
      return Promise.resolve(undefined);
    }
    return this.withAccountTenant(scope, async (transaction) => {
      const current = await selectedOperation(transaction, scope);
      if (
        current
        && current.operation_id === lookup.operationId
        && current.status_capability_sha256 === lookup.statusCapabilitySha256
      ) return operation(current);
      const deleted = await transaction.query<TombstoneRow>(
        `SELECT ${TOMBSTONE_COLUMNS}
           FROM public.teachlab_account_deletion_tombstones
          WHERE scope_sha256 = $1 AND operation_id_sha256 = $2
            AND status_capability_sha256 = $3`,
        [
          lookup.scopeSha256,
          sha256Text(lookup.operationId),
          lookup.statusCapabilitySha256
        ]
      );
      return deleted.rows[0] ? tombstone(deleted.rows[0]) : undefined;
    });
  }

  private async updateTombstones(
    transaction: TenantTransaction,
    hashes: AccountScopeHashes,
    current: AccountDeletionOperationRecord,
    now: Date
  ): Promise<void> {
    if (current.phase === "prepared" || current.phase === "fencing" || current.phase === "draining") {
      return;
    }
    for (const hash of hashes.candidates) {
      await transaction.query(
        `UPDATE public.teachlab_account_deletion_tombstones
            SET phase = $2, revision = $3, retryable_failure_code = $4,
                postgres_event_count = $5, postgres_task_count = $6,
                postgres_artifact_count = $7,
                postgres_session_count = $8,
                postgres_auth_session_count = $9,
                worker_file_count = $10, worker_byte_count = $11,
                worker_root_count = $12, updated_at = $13
          WHERE scope_sha256 = $1 AND operation_id_sha256 = $14`,
        [
          hash, current.phase, current.revision, current.retryableFailureCode,
          current.counts.postgresEvents, current.counts.postgresTasks,
          current.counts.postgresArtifacts, current.counts.postgresSessions,
          current.counts.postgresAuthSessions, current.counts.workerFiles,
          current.counts.workerBytes, current.counts.workerRoots, now,
          sha256Text(current.operationId)
        ]
      );
    }
  }

  private assertLeaseInput(tokenSha256: string, durationMs: number): void {
    if (
      !/^[0-9a-f]{64}$/.test(tokenSha256)
      || !Number.isInteger(durationMs)
      || durationMs < 1_000
      || durationMs > 5 * 60 * 1_000
    ) throw new Error("Invalid account deletion recovery lease");
  }

  private async claimOperation(
    transaction: TenantTransaction | AccountSystemTransaction,
    currentRow: OperationRow,
    claim: AccountDeletionLeaseClaim
  ): Promise<AccountDeletionLeaseClaimResult> {
    this.assertLeaseInput(claim.leaseTokenSha256, claim.leaseDurationMs);
    if (!/^[0-9a-f]{64}$/.test(claim.leaseOwnerSha256)) {
      throw new Error("Invalid account deletion recovery lease owner");
    }
    const current = operation(currentRow);
    if (current.phase === "prepared") return {kind: "busy", operation: current};
    const now = await databaseNow(transaction);
    const activeOtherLease = current.recoveryLeaseExpiresAt
      && current.recoveryLeaseExpiresAt.getTime() > now.getTime()
      && current.recoveryLeaseTokenSha256 !== claim.leaseTokenSha256;
    const delayed = !claim.ignoreRecoveryAfter
      && current.recoveryAfter
      && current.recoveryAfter.getTime() > now.getTime();
    if (activeOtherLease || delayed) return {kind: "busy", operation: current};
    const expiresAt = new Date(now.getTime() + claim.leaseDurationMs);
    const result = await transaction.query<OperationRow>(
      `UPDATE public.teachlab_account_deletion_operations
          SET revision = revision + 1, retryable_failure_code = NULL,
              recovery_lease_owner_sha256 = $2,
              recovery_lease_token_sha256 = $3,
              recovery_lease_expires_at = $4, recovery_after = NULL,
              recovery_attempts = recovery_attempts + 1, updated_at = $5
        WHERE operation_id = $1 AND phase <> 'prepared'
          AND (
            recovery_lease_token_sha256 = $3
            OR recovery_lease_expires_at IS NULL
            OR recovery_lease_expires_at <= $5
          )
          AND ($6::boolean OR recovery_after IS NULL OR recovery_after <= $5)
        RETURNING ${OPERATION_COLUMNS}`,
      [
        current.operationId, claim.leaseOwnerSha256, claim.leaseTokenSha256,
        expiresAt, now, claim.ignoreRecoveryAfter === true
      ]
    );
    return result.rows[0]
      ? {kind: "claimed", operation: operation(result.rows[0])}
      : {kind: "busy", operation: current};
  }

  private withAccountTenant<T>(
    scope: AccessScope,
    operation: (transaction: TenantTransaction) => Promise<T>
  ): Promise<T> {
    return this.database.withTenant(scope, operation, {allowAccountDeleting: true});
  }
}
