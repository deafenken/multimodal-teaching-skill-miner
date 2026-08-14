import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {resolve} from "node:path";
import {test} from "node:test";

import type {
  DatabaseQueryResult,
  TenantDatabasePort,
  TenantTransaction
} from "../src/database/tenant-database.port";
import {PostgresSessionRevocationRepository} from "../src/auth/postgres-session-revocation.repository";
import {AccountDataRightsError} from "../src/account/account-data-rights.errors";
import {PostgresSessionRepository} from "../src/sessions/postgres-session.repository";
import type {AccessScope} from "../src/tenancy/access-scope";
import {
  runTenantTransaction,
  assertPostgresReadiness,
  POSTGRES_READINESS_SQL,
  type PostgresClientContract
} from "../src/database/postgres-database";
import {PostgresTaskRepository} from "../src/tasks/postgres-task.repository";
import type {
  TaskSystemDatabasePort,
  TaskSystemTransaction
} from "../src/tasks/task-system-database.port";
import type {
  TaskLeaseClaimInput,
  TaskLeaseRetryInput,
  TaskLeaseSettlementInput
} from "../src/tasks/durable-task-queue.port";

interface ScriptedResult {
  rows: unknown[];
  rowCount?: number;
}

class RecordingTenantDatabase implements TenantDatabasePort {
  readonly calls: Array<{scope: AccessScope; sql: string; parameters: unknown[]}> = [];

  constructor(private readonly scripted: ScriptedResult[]) {}

  async withTenant<T>(
    scope: AccessScope,
    operation: (transaction: TenantTransaction) => Promise<T>
  ): Promise<T> {
    const transaction: TenantTransaction = {
      query: async <Row>(sql: string, parameters: unknown[] = []) => {
        this.calls.push({scope: {...scope}, sql, parameters});
        const result = this.scripted.shift();
        assert.ok(result, `Unexpected SQL query: ${sql}`);
        return {
          rows: result.rows as Row[],
          rowCount: result.rowCount ?? result.rows.length
        } satisfies DatabaseQueryResult<Row>;
      }
    };
    return operation(transaction);
  }
}

class RecordingTaskSystemDatabase implements TaskSystemDatabasePort {
  readonly calls: Array<{sql: string; parameters: unknown[]}> = [];

  constructor(private readonly scripted: ScriptedResult[]) {}

  async withTaskDispatcher<T>(
    operation: (transaction: TaskSystemTransaction) => Promise<T>
  ): Promise<T> {
    const transaction: TaskSystemTransaction = {
      query: async <Row>(sql: string, parameters: unknown[] = []) => {
        this.calls.push({sql, parameters});
        const result = this.scripted.shift();
        assert.ok(result, `Unexpected dispatcher SQL query: ${sql}`);
        return {
          rows: result.rows as Row[],
          rowCount: result.rowCount ?? result.rows.length
        } satisfies DatabaseQueryResult<Row>;
      }
    };
    return operation(transaction);
  }
}

const taskRow = {
  id: "00000000-0000-4000-8000-000000000002",
  tenant_id: "tenant-a",
  owner_id: "alice",
  session_id: "00000000-0000-4000-8000-000000000001",
  learner_message: "Explain the transition",
  client_request_id: "turn-1",
  status: "queued" as const,
  failure_code: null,
  created_at: "2026-08-12T00:00:00.000Z",
  updated_at: "2026-08-12T00:00:00.000Z",
  version: 1,
  attempt_count: 0,
  available_at: "2026-08-12T00:00:00.000Z",
  lease_owner_sha256: null,
  lease_token_sha256: null,
  lease_expires_at: null
};

function claimInput(overrides: Partial<TaskLeaseClaimInput> = {}): TaskLeaseClaimInput {
  return {
    leaseOwnerSha256: "1".repeat(64),
    leaseTokenSha256: "2".repeat(64),
    leaseDurationMs: 30_000,
    maximumAttempts: 3,
    ...overrides
  };
}

test("real PostgreSQL integration wires the dispatcher system boundary", async () => {
  const integrationSource = await readFile(
    resolve(process.cwd(), "test/postgres-integration.test.ts"),
    "utf8"
  );
  assert.match(
    integrationSource,
    /new PostgresTaskRepository\(database, database\)/
  );
});

const row = {
  id: "00000000-0000-4000-8000-000000000001",
  tenant_id: "tenant-a",
  owner_id: "alice",
  title: "Original",
  learner: "Ada",
  status: "active",
  round: 0,
  created_at: "2026-08-12T00:00:00.000Z",
  updated_at: "2026-08-12T00:00:00.000Z",
  version: 3
};

test("Postgres session updates lock the scoped row and reject a stale version", async () => {
  const database = new RecordingTenantDatabase([{rows: [row]}]);
  const repository = new PostgresSessionRepository(database);
  const result = await repository.updateOwned(
    {tenantId: "tenant-a", ownerId: "alice"},
    row.id,
    {title: "Stale"},
    2
  );
  assert.deepEqual(result, {kind: "version_conflict", currentVersion: 3});
  assert.equal(database.calls.length, 1);
  assert.match(database.calls[0]?.sql ?? "", /FOR UPDATE/);
  assert.deepEqual(database.calls[0]?.parameters, ["tenant-a", "alice", row.id]);
  assert.deepEqual(database.calls[0]?.scope, {tenantId: "tenant-a", ownerId: "alice"});
});

test("Postgres session updates increment versions inside one tenant transaction", async () => {
  const updatedRow = {...row, title: "Updated", version: 4};
  const database = new RecordingTenantDatabase([{rows: [row]}, {rows: [updatedRow]}]);
  const repository = new PostgresSessionRepository(database);
  const result = await repository.updateOwned(
    {tenantId: "tenant-a", ownerId: "alice"},
    row.id,
    {title: "Updated"},
    3
  );
  assert.equal(result.kind, "updated");
  if (result.kind === "updated") assert.equal(result.session.version, 4);
  assert.equal(database.calls.length, 2);
  assert.match(database.calls[1]?.sql ?? "", /version = version \+ 1/);
  assert.deepEqual(database.calls[1]?.scope, {tenantId: "tenant-a", ownerId: "alice"});
});

test("migrations force RLS on all four tenant tables and hash-only auth sessions", async () => {
  const baseMigration = await readFile(
    resolve(process.cwd(), "migrations/001_tenant_rls.sql"),
    "utf8"
  );
  const authMigration = await readFile(
    resolve(process.cwd(), "migrations/002_auth_session_revocation.sql"),
    "utf8"
  );
  const migration = `${baseMigration}\n${authMigration}`;
  for (const table of [
    "teachlab_teaching_sessions",
    "teachlab_agent_tasks",
    "teachlab_task_events",
    "teachlab_auth_sessions"
  ]) {
    assert.match(migration, new RegExp(`ALTER TABLE ${table} FORCE ROW LEVEL SECURITY`));
  }
  assert.match(migration, /current_setting\('app\.tenant_id', true\)/);
  assert.match(migration, /current_setting\('app\.user_id', true\)/);
  assert.match(migration, /client_request_id IS NOT NULL/);
  assert.match(authMigration, /session_id_sha256 text NOT NULL/);
  assert.doesNotMatch(authMigration, /csrf|cookie|raw_token|access_token/i);
});

test("Postgres revocation locks once, uses CAS, and repeats idempotently", async () => {
  const active = {
    tenant_id: "tenant-a",
    owner_id: "alice",
    session_id_sha256: "a".repeat(64),
    issued_at: "2026-08-12T00:00:00.000Z",
    expires_at: "2026-08-12T08:00:00.000Z",
    revoked_at: null,
    revocation_reason: null,
    version: 1
  };
  const revoked = {
    ...active,
    revoked_at: "2026-08-12T01:00:00.000Z",
    revocation_reason: "user_logout" as const,
    version: 2
  };
  const databaseNow = new Date("2026-08-12T01:00:00.000Z");
  const database = new RecordingTenantDatabase([
    {rows: [{database_now: databaseNow}]},
    {rows: [], rowCount: 0},
    {rows: [active]},
    {rows: [revoked]}
  ]);
  const repository = new PostgresSessionRevocationRepository(database);
  const result = await repository.revoke({
    tenantId: "tenant-a",
    ownerId: "alice",
    sessionIdSha256: "a".repeat(64),
    revokedAt: new Date("2026-08-12T01:00:00.000Z"),
    reason: "user_logout"
  });
  assert.equal(result.kind, "revoked");
  assert.match(database.calls[0]?.sql ?? "", /statement_timestamp\(\)/);
  assert.match(database.calls[2]?.sql ?? "", /FOR UPDATE/);
  assert.match(database.calls[3]?.sql ?? "", /revoked_at IS NULL/);
  assert.match(database.calls[3]?.sql ?? "", /version = \$6/);
  assert.deepEqual(database.calls[3]?.parameters[3], databaseNow);
  assert.equal(database.calls[3]?.scope.tenantId, "tenant-a");
  assert.equal(database.calls[3]?.scope.ownerId, "alice");

  const replayDatabase = new RecordingTenantDatabase([
    {rows: [{database_now: new Date("2026-08-12T01:00:01.000Z")}]},
    {rows: [], rowCount: 0},
    {rows: [revoked]}
  ]);
  const replay = await new PostgresSessionRevocationRepository(replayDatabase).revoke({
    tenantId: "tenant-a",
    ownerId: "alice",
    sessionIdSha256: "a".repeat(64),
    revokedAt: new Date("2026-08-12T01:00:01.000Z"),
    reason: "user_logout"
  });
  assert.equal(replay.kind, "already_revoked");
  assert.equal(replayDatabase.calls.length, 3);
});

test("Postgres auth-session cleanup is tenant scoped, ordered, and bounded", async () => {
  const database = new RecordingTenantDatabase([{rows: [], rowCount: 7}]);
  const repository = new PostgresSessionRevocationRepository(database);
  assert.equal(
    await repository.cleanupExpired(
      {tenantId: "tenant-a", ownerId: "alice"},
      new Date("2026-08-13T00:00:00.000Z"),
      100_000
    ),
    7
  );
  assert.match(database.calls[0]?.sql ?? "", /LIMIT \$4/);
  assert.match(database.calls[0]?.sql ?? "", /FOR UPDATE SKIP LOCKED/);
  assert.deepEqual(database.calls[0]?.parameters.slice(0, 2), ["tenant-a", "alice"]);
  assert.equal(database.calls[0]?.parameters[3], 256);
});

test("Postgres authentication uses the database clock for expiry decisions", async () => {
  const databaseNow = new Date("2026-08-12T02:00:00.000Z");
  const active = {
    tenant_id: "tenant-a",
    owner_id: "alice",
    session_id_sha256: "b".repeat(64),
    issued_at: "2026-08-12T00:00:00.000Z",
    expires_at: "2026-08-12T08:00:00.000Z",
    revoked_at: null,
    revocation_reason: null,
    version: 1
  };
  const database = new RecordingTenantDatabase([
    {rows: [{database_now: databaseNow}]},
    {rows: [], rowCount: 0},
    {rows: [active]}
  ]);
  const state = await new PostgresSessionRevocationRepository(database)
    .inspectAuthoritatively(
      {tenantId: "tenant-a", ownerId: "alice"},
      active.session_id_sha256
    );
  assert.equal(state.kind, "active");
  assert.match(database.calls[0]?.sql ?? "", /statement_timestamp\(\)/);
  assert.deepEqual(database.calls[1]?.parameters[2], databaseNow);
  assert.deepEqual(database.calls[2]?.parameters[3], databaseNow);
});

test("tenant transactions set transaction-local tenant and user context", async () => {
  const calls: Array<{sql: string; parameters: unknown[]}> = [];
  let released = false;
  const client: PostgresClientContract = {
    async query(sql, parameters = []) {
      calls.push({sql, parameters});
      return {rows: [], rowCount: 0};
    },
    release() {
      released = true;
    }
  };
  const result = await runTenantTransaction(
    client,
    {tenantId: "tenant-a", ownerId: "alice"},
    async (transaction) => {
      await transaction.query("SELECT 1");
      return "committed";
    }
  );
  assert.equal(result, "committed");
  assert.deepEqual(
    calls.map((call) => call.sql),
    [
      "BEGIN",
      "SELECT set_config('app.tenant_id', $1, true), set_config('app.user_id', $2, true)",
      "SELECT 1",
      "COMMIT"
    ]
  );
  assert.deepEqual(calls[1]?.parameters, ["tenant-a", "alice"]);
  assert.equal(released, true);
});

test("tenant transactions roll back and release the client on failure", async () => {
  const calls: string[] = [];
  let released = false;
  const client: PostgresClientContract = {
    async query(sql) {
      calls.push(sql);
      return {rows: [], rowCount: 0};
    },
    release() {
      released = true;
    }
  };
  await assert.rejects(
    runTenantTransaction(
      client,
      {tenantId: "tenant-a", ownerId: "alice"},
      async () => {
        throw new Error("failed operation");
      }
    ),
    /failed operation/
  );
  assert.equal(calls.at(-1), "ROLLBACK");
  assert.equal(released, true);
});

test("tenant lifecycle fencing blocks only the deleting scope before application SQL", async () => {
  function clientFor(deleting: boolean): {
    client: PostgresClientContract;
    calls: string[];
    released: () => boolean;
  } {
    const calls: string[] = [];
    let wasReleased = false;
    return {
      calls,
      released: () => wasReleased,
      client: {
        async query(sql) {
          calls.push(sql);
          if (sql.includes("teachlab_account_deletion_operations")) {
            return {rows: [{deleting, deleted: false}], rowCount: 1};
          }
          return {rows: [], rowCount: 0};
        },
        release() {
          wasReleased = true;
        }
      }
    };
  }

  const fenced = clientFor(true);
  let fencedOperationCalled = false;
  await assert.rejects(
    runTenantTransaction(
      fenced.client,
      {tenantId: "tenant-deleting", ownerId: "charlie"},
      async () => {
        fencedOperationCalled = true;
      },
      {scopeHashes: ["a".repeat(64)]}
    ),
    (error: unknown) => error instanceof AccountDataRightsError
      && error.statusCode === 409
      && error.code === "account_deletion_already_started"
  );
  assert.equal(fencedOperationCalled, false);
  assert.equal(fenced.calls.at(-1), "ROLLBACK");
  assert.equal(fenced.released(), true);

  const active = clientFor(false);
  const result = await runTenantTransaction(
    active.client,
    {tenantId: "tenant-active", ownerId: "alice"},
    async (transaction) => {
      await transaction.query("SELECT application_work");
      return "committed";
    },
    {scopeHashes: ["b".repeat(64)]}
  );
  assert.equal(result, "committed");
  assert.equal(active.calls.includes("SELECT application_work"), true);
  assert.equal(active.calls.at(-1), "COMMIT");
  assert.equal(active.released(), true);
});

test("Postgres task creation atomically requires an active owned session", async () => {
  const taskRow = {
    id: "00000000-0000-4000-8000-000000000002",
    tenant_id: "tenant-a",
    owner_id: "alice",
    session_id: row.id,
    learner_message: "Explain the transition",
    client_request_id: "turn-1",
    status: "queued",
    failure_code: null,
    created_at: "2026-08-12T00:00:00.000Z",
    updated_at: "2026-08-12T00:00:00.000Z",
    version: 1
  };
  const database = new RecordingTenantDatabase([
    {rows: []},
    {rows: [{status: "active"}]},
    {rows: [taskRow]}
  ]);
  const repository = new PostgresTaskRepository(database);
  const result = await repository.create({
    tenantId: "tenant-a",
    ownerId: "alice",
    sessionId: row.id,
    learnerMessage: "Explain the transition",
    clientRequestId: "turn-1"
  });
  assert.equal(result.created, true);
  assert.equal(result.task.tenantId, "tenant-a");
  assert.equal(result.task.ownerId, "alice");
  assert.match(database.calls[1]?.sql ?? "", /FOR UPDATE/);
  assert.deepEqual(database.calls[1]?.parameters, ["tenant-a", "alice", row.id]);
  assert.match(database.calls[2]?.sql ?? "", /ON CONFLICT/);
  assert.match(database.calls[2]?.sql ?? "", /client_request_id IS NOT NULL/);
  assert.deepEqual(database.calls[2]?.scope, {tenantId: "tenant-a", ownerId: "alice"});
});

test("durable task claim locks the scope before the task row and rechecks the deletion fence", async () => {
  const claimedRow = {
    ...taskRow,
    status: "running" as const,
    attempt_count: 1,
    lease_owner_sha256: "1".repeat(64),
    lease_token_sha256: "2".repeat(64),
    lease_expires_at: "2026-08-12T00:01:00.000Z",
    version: 2
  };
  const system = new RecordingTaskSystemDatabase([
    {rows: [taskRow]},
    {rows: []},
    {rows: [taskRow]},
    {rows: [claimedRow]}
  ]);
  const repository = new PostgresTaskRepository(
    {} as TenantDatabasePort,
    system
  );
  const result = await repository.claimNext(claimInput());
  assert.equal(result.kind, "claimed");
  if (result.kind === "claimed") {
    assert.equal(result.task.status, "running");
    assert.equal(result.task.attemptCount, 1);
    assert.equal(result.leaseTokenSha256, "2".repeat(64));
  }
  assert.match(system.calls[0]?.sql ?? "", /NOT EXISTS[\s\S]+teachlab_account_deletion_operations/);
  assert.match(system.calls[1]?.sql ?? "", /pg_advisory_xact_lock/);
  assert.match(system.calls[2]?.sql ?? "", /FOR UPDATE/);
  assert.match(system.calls[2]?.sql ?? "", /NOT EXISTS[\s\S]+teachlab_account_deletion_operations/);
  assert.match(system.calls[3]?.sql ?? "", /attempt_count = t\.attempt_count \+ 1/);
});

test("expired task at the attempt ceiling is terminalized without exceeding the bounded counter", async () => {
  const exhausted = {
    ...taskRow,
    status: "running" as const,
    attempt_count: 3,
    lease_token_sha256: "3".repeat(64),
    lease_owner_sha256: "4".repeat(64),
    lease_expires_at: "2026-08-11T00:00:00.000Z",
    version: 8
  };
  const terminal = {
    ...exhausted,
    status: "failed" as const,
    failure_code: "task_attempt_limit_exceeded",
    lease_owner_sha256: null,
    lease_token_sha256: null,
    lease_expires_at: null,
    version: 9
  };
  const system = new RecordingTaskSystemDatabase([
    {rows: [exhausted]},
    {rows: []},
    {rows: [exhausted]},
    {rows: [terminal]},
    {rows: []}
  ]);
  const repository = new PostgresTaskRepository({} as TenantDatabasePort, system);
  const result = await repository.claimNext(claimInput({maximumAttempts: 3}));
  assert.equal(result.kind, "terminalized");
  if (result.kind === "terminalized") {
    assert.equal(result.reason, "attempts_exhausted");
    assert.equal(result.task.status, "failed");
    assert.equal(result.task.attemptCount, 3);
    assert.equal(result.leaseTokenSha256, undefined);
  }
  assert.doesNotMatch(system.calls[3]?.sql ?? "", /attempt_count\s*=\s*attempt_count\s*\+/);
  assert.match(system.calls[3]?.sql ?? "", /failure_code = 'task_attempt_limit_exceeded'/);
  assert.match(system.calls[4]?.sql ?? "", /INSERT INTO public\.teachlab_task_events/);
  assert.match(system.calls[4]?.parameters?.[5] as string, /task_attempt_limit_exceeded/);
});

test("stale task lease tokens cannot settle or append dispatcher events", async () => {
  const system = new RecordingTaskSystemDatabase([
    {rows: []},
    {rows: []},
  ]);
  const repository = new PostgresTaskRepository({} as TenantDatabasePort, system);
  const input: TaskLeaseSettlementInput = {
    tenantId: "tenant-a",
    ownerId: "alice",
    taskId: taskRow.id,
    leaseTokenSha256: "f".repeat(64),
    status: "succeeded",
    events: [{type: "status", payload: {kind: "task.succeeded"}}]
  };
  assert.equal(await repository.settleLease(input), undefined);
  assert.equal(system.calls.length, 2);
  assert.match(system.calls[0]?.sql ?? "", /pg_advisory_xact_lock/);
  assert.match(system.calls[1]?.sql ?? "", /lease_token_sha256 = \$6/);
  assert.equal(system.calls.some((call) => /INSERT INTO public\.teachlab_task_events/.test(call.sql)), false);
});

test("retry lease uses the current attempt count for bounded exponential backoff", async () => {
  const queued = {
    ...taskRow,
    status: "queued" as const,
    attempt_count: 2,
    failure_code: "provider_execution_failed",
    available_at: "2026-08-12T00:00:04.000Z",
    version: 3
  };
  const system = new RecordingTaskSystemDatabase([
    {rows: []},
    {rows: [queued]},
    {rows: []}
  ]);
  const repository = new PostgresTaskRepository({} as TenantDatabasePort, system);
  const input: TaskLeaseRetryInput = {
    tenantId: "tenant-a",
    ownerId: "alice",
    taskId: taskRow.id,
    leaseTokenSha256: "2".repeat(64),
    failureCode: "provider_execution_failed",
    maximumAttempts: 3,
    retryBaseMs: 1_000,
    retryMaximumMs: 10_000,
    events: [{type: "error", payload: {kind: "task.failed"}}]
  };
  const result = await repository.retryLease(input);
  assert.equal(result.kind, "retry_scheduled");
  assert.match(system.calls[1]?.sql ?? "", /attempt_count >= \$8/);
  assert.match(system.calls[1]?.sql ?? "", /power\(/);
  assert.match(system.calls[1]?.sql ?? "", /GREATEST\(attempt_count - 1, 0\)/);
  assert.equal(system.calls.length, 3);
  assert.match(system.calls[2]?.sql ?? "", /INSERT INTO public\.teachlab_task_events/);
});

test("production readiness rejects privileged roles and incomplete RLS migrations", () => {
  assert.match(
    POSTGRES_READINESS_SQL,
    /\('public\.teachlab_account_deletion_operations'\)\s*\), auth_session_columns AS \(/
  );
  assert.doesNotMatch(
    POSTGRES_READINESS_SQL,
    /\('public\.teachlab_auth_sessions'\)\s*\)\s*\), auth_session_columns/
  );
  const safe = {
    role_name: "teachlab_app",
    role_is_superuser: false,
    role_bypasses_rls: false,
    owned_table_count: 0,
    table_count: 5,
    rls_enforced: true,
    auth_session_columns_safe: true,
    account_recovery_columns_safe: true,
    account_recovery_policy_present: true,
    tombstone_table_present: true
  };
  assert.doesNotThrow(() => assertPostgresReadiness(safe));
  const durable = {
    ...safe,
    artifact_table_present: true,
    artifact_rls_enforced: true,
    task_lease_columns_safe: true,
    task_dispatcher_policy_present: true
  };
  assert.doesNotThrow(() => assertPostgresReadiness(durable, {requireDurable: true}));
  for (const field of [
    "artifact_table_present",
    "artifact_rls_enforced",
    "task_lease_columns_safe",
    "task_dispatcher_policy_present"
  ] as const) {
    assert.throws(
      () => assertPostgresReadiness({...durable, [field]: false}, {requireDurable: true}),
      /migration is missing/
    );
  }
  assert.throws(
    () => assertPostgresReadiness({...safe, role_is_superuser: true}),
    /must not be superuser/
  );
  assert.throws(
    () => assertPostgresReadiness({...safe, table_count: 2}),
    /migration is missing/
  );
  assert.throws(
    () => assertPostgresReadiness({...safe, rls_enforced: false}),
    /RLS is not forced/
  );
  assert.throws(
    () => assertPostgresReadiness({...safe, owned_table_count: 1}),
    /must not own tenant tables/
  );
});
