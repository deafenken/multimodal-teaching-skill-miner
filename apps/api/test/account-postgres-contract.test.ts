import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {readFile} from "node:fs/promises";
import {resolve} from "node:path";
import {test} from "node:test";

import type {
  AccountSystemDatabasePort,
  AccountSystemTransaction
} from "../src/account/account-system-database.port";
import {PostgresAccountDataRightsRepository} from "../src/account/postgres-account-data-rights.repository";
import type {
  DatabaseQueryResult,
  TenantDatabasePort,
  TenantTransaction
} from "../src/database/tenant-database.port";
import type {AccessScope} from "../src/tenancy/access-scope";

const SCOPE: AccessScope = {tenantId: "tenant-a", ownerId: "alice"};
const SCOPE_HASH = "a".repeat(64);
const OPERATION_ID = "adel_00000000000000000000000000000001";
const STATUS_HASH = "b".repeat(64);
const LEASE_HASH = "9".repeat(64);
const ROTATED_SCOPE_HASH = "0".repeat(64);

const operationRow = {
  tenant_id: SCOPE.tenantId,
  owner_id: SCOPE.ownerId,
  scope_sha256: SCOPE_HASH,
  operation_id: OPERATION_ID,
  phase: "committing_database",
  revision: 7,
  challenge_id: "adelc_00000000000000000000000000000001",
  challenge_token_sha256: "c".repeat(64),
  challenge_csrf_sha256: "d".repeat(64),
  challenge_session_sha256: "e".repeat(64),
  challenge_authority_grant_sha256: "2".repeat(64),
  challenge_canonical_identity_sha256: "3".repeat(64),
  challenge_issuer_sha256: "4".repeat(64),
  challenge_authority_key_version: "test-v1",
  challenge_authenticated_at: "2026-08-12T00:00:00.000Z",
  challenge_assurance_level: 2,
  challenge_expires_at: "2026-08-12T00:05:00.000Z",
  status_capability_sha256: STATUS_HASH,
  idempotency_key_sha256: "f".repeat(64),
  confirmation_sha256: "1".repeat(64),
  retryable_failure_code: null,
  recovery_lease_owner_sha256: "8".repeat(64),
  recovery_lease_token_sha256: LEASE_HASH,
  recovery_lease_expires_at: "2026-08-12T00:03:00.000Z",
  recovery_after: null,
  recovery_attempts: 1,
  postgres_event_count: 0,
  postgres_task_count: 0,
  postgres_artifact_count: 0,
  postgres_session_count: 0,
  postgres_auth_session_count: 0,
  worker_file_count: 12,
  worker_byte_count: "4096",
  worker_root_count: 1,
  created_at: "2026-08-12T00:00:00.000Z",
  updated_at: "2026-08-12T00:01:00.000Z"
} as const;

class ScriptedTenantDatabase implements TenantDatabasePort {
  readonly calls: Array<{scope: AccessScope; sql: string; parameters: unknown[]}> = [];
  failOnTasks = false;

  async withTenant<T>(
    scope: AccessScope,
    operation: (transaction: TenantTransaction) => Promise<T>
  ): Promise<T> {
    const transaction: TenantTransaction = {
      query: async <Row>(sql: string, parameters: unknown[] = []) => {
        this.calls.push({scope: {...scope}, sql, parameters});
        let result: {rows: unknown[]; rowCount: number};
        if (/statement_timestamp\(\) AS database_now/.test(sql)) {
          result = {rows: [{database_now: "2026-08-12T00:02:00.000Z"}], rowCount: 1};
        } else if (/FROM public\.teachlab_account_deletion_operations/.test(sql)) {
          result = {rows: [operationRow], rowCount: 1};
        } else if (/DELETE FROM public\.teachlab_task_events/.test(sql)) {
          result = {rows: [], rowCount: 3};
        } else if (/DELETE FROM public\.teachlab_agent_tasks/.test(sql)) {
          if (this.failOnTasks) throw new Error("simulated database failure");
          result = {rows: [], rowCount: 2};
        } else if (/DELETE FROM public\.teachlab_teaching_sessions/.test(sql)) {
          result = {rows: [], rowCount: 1};
        } else if (/DELETE FROM public\.teachlab_auth_sessions/.test(sql)) {
          result = {rows: [], rowCount: 4};
        } else if (/DELETE FROM public\.teachlab_artifacts/.test(sql)) {
          result = {rows: [], rowCount: 5};
        } else if (/INSERT INTO public\.teachlab_account_deletion_tombstones/.test(sql)) {
          result = {rows: [{scope_sha256: parameters[0]}], rowCount: 1};
        } else if (/UPDATE public\.teachlab_account_deletion_tombstones/.test(sql)) {
          result = {rows: [{ok: true}], rowCount: 1};
        } else if (/DELETE FROM public\.teachlab_account_deletion_operations/.test(sql)) {
          result = {rows: [], rowCount: 1};
        } else {
          throw new Error(`Unexpected SQL: ${sql}`);
        }
        return result as DatabaseQueryResult<Row>;
      }
    };
    return operation(transaction);
  }
}

class NoopSystemDatabase implements AccountSystemDatabasePort {
  async withAccountSystem<T>(
    operation: (transaction: AccountSystemTransaction) => Promise<T>
  ): Promise<T> {
    return operation({
      query: async <Row>() => ({rows: [] as Row[], rowCount: 0})
    });
  }

  async withAccountRecovery<T>(
    operation: (transaction: AccountSystemTransaction) => Promise<T>
  ): Promise<T> {
    return this.withAccountSystem(operation);
  }
}

class LeaseSystemDatabase implements AccountSystemDatabasePort {
  now = new Date("2026-08-12T00:02:00.000Z");
  row: Record<string, unknown> = {
    ...operationRow,
    phase: "quarantining",
    recovery_lease_owner_sha256: null,
    recovery_lease_token_sha256: null,
    recovery_lease_expires_at: null,
    recovery_attempts: 0
  };
  readonly calls: string[] = [];

  async withAccountSystem<T>(
    operation: (transaction: AccountSystemTransaction) => Promise<T>
  ): Promise<T> {
    return this.withAccountRecovery(operation);
  }

  async withAccountRecovery<T>(
    operation: (transaction: AccountSystemTransaction) => Promise<T>
  ): Promise<T> {
    return operation({
      query: async <Row>(sql: string, parameters: unknown[] = []) => {
        this.calls.push(sql);
        if (/statement_timestamp\(\) AS database_now/.test(sql)) {
          return {rows: [{database_now: this.now}] as Row[], rowCount: 1};
        }
        if (/^\s*SELECT[\s\S]+FROM public\.teachlab_account_deletion_operations/.test(sql)) {
          return {rows: [structuredClone(this.row)] as Row[], rowCount: 1};
        }
        if (/UPDATE public\.teachlab_account_deletion_operations/.test(sql)) {
          this.row = {
            ...this.row,
            revision: Number(this.row.revision) + 1,
            retryable_failure_code: null,
            recovery_lease_owner_sha256: parameters[1],
            recovery_lease_token_sha256: parameters[2],
            recovery_lease_expires_at: parameters[3],
            recovery_after: null,
            recovery_attempts: Number(this.row.recovery_attempts) + 1,
            updated_at: parameters[4]
          };
          return {rows: [structuredClone(this.row)] as Row[], rowCount: 1};
        }
        throw new Error(`Unexpected recovery SQL: ${sql}`);
      }
    });
  }
}

test("Postgres account finalization deletes child rows then all auth sessions in one scoped call", async () => {
  const database = new ScriptedTenantDatabase();
  const repository = new PostgresAccountDataRightsRepository(
    database,
    new NoopSystemDatabase()
  );
  const receipt = await repository.commitDeletion({
    ...SCOPE,
    hashes: {active: ROTATED_SCOPE_HASH, candidates: [SCOPE_HASH, ROTATED_SCOPE_HASH]},
    operationId: OPERATION_ID,
    expectedRevision: 7,
    leaseTokenSha256: LEASE_HASH,
    workerCounts: {workerFiles: 12, workerBytes: 4_096, workerRoots: 1},
    receiptId: "adelr_00000000000000000000000000000001",
    deletedAt: new Date("2026-08-12T00:02:00.000Z")
  });
  const deletes = database.calls
    .map((call) => call.sql)
    .filter((sql) => /DELETE FROM public\.teachlab_/.test(sql));
  assert.match(deletes[0] ?? "", /teachlab_task_events/);
  assert.match(deletes[1] ?? "", /teachlab_agent_tasks/);
  assert.match(deletes[2] ?? "", /teachlab_artifacts/);
  assert.match(deletes[3] ?? "", /teachlab_teaching_sessions/);
  assert.match(deletes[4] ?? "", /teachlab_auth_sessions/);
  assert.match(deletes[5] ?? "", /teachlab_account_deletion_operations/);
  const tombstoneReconciliations = database.calls.filter((call) =>
    /INSERT INTO public\.teachlab_account_deletion_tombstones/.test(call.sql)
  );
  assert.deepEqual(
    tombstoneReconciliations.map((call) => call.parameters[0]),
    [SCOPE_HASH, ROTATED_SCOPE_HASH]
  );
  assert.ok(tombstoneReconciliations.every((call) =>
    call.parameters[2] === createHash("sha256").update(OPERATION_ID).digest("hex")
  ));
  for (const call of database.calls) {
    assert.equal(call.scope.tenantId, SCOPE.tenantId);
    assert.equal(call.scope.ownerId, SCOPE.ownerId);
    if (/DELETE FROM public\.teachlab_(?:task_events|agent_tasks|teaching_sessions|auth_sessions)/.test(call.sql)) {
      assert.deepEqual(call.parameters, [SCOPE.tenantId, SCOPE.ownerId]);
    }
  }
  assert.equal(receipt.deleted_counts.postgres_events, 3);
  assert.equal(receipt.deleted_counts.postgres_tasks, 2);
  assert.equal(receipt.deleted_counts.postgres_artifacts, 5);
  assert.equal(receipt.deleted_counts.postgres_sessions, 1);
  assert.equal(receipt.deleted_counts.postgres_auth_sessions, 4);
  assert.equal(receipt.all_devices_session_authority_deleted, true);
  assert.equal(receipt.deleted_at, "2026-08-12T00:02:00.000Z");
});

test("Postgres account deletion failure cannot advance into parent/auth deletion", async () => {
  const database = new ScriptedTenantDatabase();
  database.failOnTasks = true;
  const repository = new PostgresAccountDataRightsRepository(
    database,
    new NoopSystemDatabase()
  );
  await assert.rejects(repository.commitDeletion({
    ...SCOPE,
    hashes: {active: SCOPE_HASH, candidates: [SCOPE_HASH]},
    operationId: OPERATION_ID,
    expectedRevision: 7,
    leaseTokenSha256: LEASE_HASH,
    workerCounts: {workerFiles: 12, workerBytes: 4_096, workerRoots: 1},
    receiptId: "adelr_00000000000000000000000000000001",
    deletedAt: new Date("2026-08-12T00:02:00.000Z")
  }), /simulated database failure/);
  const sql = database.calls.map((call) => call.sql).join("\n");
  assert.doesNotMatch(sql, /DELETE FROM public\.teachlab_teaching_sessions/);
  assert.doesNotMatch(sql, /DELETE FROM public\.teachlab_auth_sessions/);
  assert.doesNotMatch(sql, /phase = 'completed'/);
});

test("Postgres recovery lease is exclusive across instances and transferable only after expiry", async () => {
  const system = new LeaseSystemDatabase();
  const first = new PostgresAccountDataRightsRepository(
    {} as TenantDatabasePort,
    system
  );
  const second = new PostgresAccountDataRightsRepository(
    {} as TenantDatabasePort,
    system
  );
  const lookup = {
    scopeSha256: SCOPE_HASH,
    operationId: OPERATION_ID,
    statusCapabilitySha256: STATUS_HASH
  };
  const claim = (owner: string, token: string) => ({
    leaseOwnerSha256: owner.repeat(64),
    leaseTokenSha256: token.repeat(64),
    leaseDurationMs: 30_000,
    now: new Date(system.now),
    ignoreRecoveryAfter: true
  });
  assert.equal(
    (await first.claimDeletionByCapability(lookup, claim("7", "8"))).kind,
    "claimed"
  );
  assert.equal(
    (await second.claimDeletionByCapability(lookup, claim("5", "6"))).kind,
    "busy"
  );
  system.now = new Date("2026-08-12T00:02:31.000Z");
  const takeover = await second.claimDeletionByCapability(lookup, claim("5", "6"));
  assert.equal(takeover.kind, "claimed");
  if (takeover.kind !== "claimed") throw new Error("recovery lease was not claimed");
  assert.equal(takeover.operation.recoveryLeaseTokenSha256, "6".repeat(64));
  assert.match(system.calls.join("\n"), /recovery_lease_expires_at <= \$5/);
});

test("account migration keeps active raw scope RLS-bound and terminal tombstones hash-only", async () => {
  const migration = await readFile(
    resolve(process.cwd(), "migrations/003_account_data_rights.sql"),
    "utf8"
  );
  assert.match(migration, /teachlab_account_deletion_operations/);
  assert.match(
    migration,
    /ALTER TABLE teachlab_account_deletion_operations FORCE ROW LEVEL SECURITY/
  );
  assert.match(migration, /current_setting\('app\.tenant_id', true\)/);
  assert.match(migration, /current_setting\('app\.user_id', true\)/);
  const tombstoneDefinition = migration.slice(
    migration.indexOf("CREATE TABLE IF NOT EXISTS teachlab_account_deletion_tombstones")
  );
  assert.doesNotMatch(tombstoneDefinition, /tenant_id|owner_id|subject|email/i);
  assert.match(tombstoneDefinition, /scope_sha256 text PRIMARY KEY/);
  assert.match(tombstoneDefinition, /receipt_scope_sha256 text NOT NULL/);
  assert.doesNotMatch(migration, /raw_token|access_token|refresh_token|cookie_value/i);
  const recoveryMigration = await readFile(
    resolve(process.cwd(), "migrations/004_account_deletion_recovery.sql"),
    "utf8"
  );
  assert.match(recoveryMigration, /recovery_lease_token_sha256/);
  assert.match(recoveryMigration, /FORCE-RLS recovery policy|transaction-local marker/);
  assert.match(recoveryMigration, /app\.account_deletion_recovery/);
  assert.doesNotMatch(recoveryMigration, /raw_token|access_token|refresh_token|cookie_value/i);
  const repositorySource = await readFile(
    resolve(process.cwd(), "src/account/postgres-account-data-rights.repository.ts"),
    "utf8"
  );
  assert.match(repositorySource, /FOR UPDATE SKIP LOCKED/);
  assert.match(repositorySource, /recovery_lease_token_sha256 = \$16/);
});
