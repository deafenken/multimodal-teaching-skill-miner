import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {resolve} from "node:path";
import test from "node:test";

import {Pool} from "pg";

import type {AppConfigService} from "../src/config/app-config.service";
import {PostgresDatabase} from "../src/database/postgres-database";
import {PostgresSessionRevocationRepository} from "../src/auth/postgres-session-revocation.repository";
import {PostgresSessionRepository} from "../src/sessions/postgres-session.repository";
import {PostgresTaskRepository} from "../src/tasks/postgres-task.repository";
import {PostgresArtifactStorageAdapter} from "../src/platform/postgres-artifact-storage.adapter";

const adminUrl = process.env.POSTGRES_TEST_ADMIN_URL?.trim();
const appRole = "teachlab_app_integration";
const appPassword = "test-pass";
const accountScopeKeys = [{
  version: "k1",
  secret: "postgres-integration-account-scope-key-32-bytes"
}];

function applicationUrl(source: string): string {
  const url = new URL(source);
  url.username = appRole;
  url.password = appPassword;
  return url.toString();
}

test(
  "real PostgreSQL enforces tenant RLS, CAS, and task idempotency",
  {skip: adminUrl ? false : "POSTGRES_TEST_ADMIN_URL is not configured"},
  async (context) => {
    assert.ok(adminUrl);
    const admin = new Pool({connectionString: adminUrl, max: 1});
    let database: PostgresDatabase | undefined;

    const reset = async (): Promise<void> => {
      await admin.query(
        "DROP TABLE IF EXISTS public.teachlab_auth_sessions, public.teachlab_task_events, " +
          "public.teachlab_agent_tasks, public.teachlab_teaching_sessions, " +
          "public.teachlab_artifacts, " +
          "public.teachlab_account_deletion_tombstones, " +
          "public.teachlab_account_deletion_operations CASCADE"
      );
      await admin.query(`DROP ROLE IF EXISTS ${appRole}`);
    };
    await reset();
    context.after(async () => {
      await database?.onModuleDestroy();
      await reset();
      await admin.end();
    });

    const migration = await readFile(
      resolve(process.cwd(), "migrations/001_tenant_rls.sql"),
      "utf8"
    );
    await admin.query(migration);
    await admin.query(
      await readFile(
        resolve(process.cwd(), "migrations/002_auth_session_revocation.sql"),
        "utf8"
      )
    );
    await admin.query(
      await readFile(
        resolve(process.cwd(), "migrations/003_account_data_rights.sql"),
        "utf8"
      )
    );
    await admin.query(
      await readFile(
        resolve(process.cwd(), "migrations/004_account_deletion_recovery.sql"),
        "utf8"
      )
    );
    await admin.query(
      await readFile(
        resolve(process.cwd(), "migrations/005_durable_tasks_artifacts.sql"),
        "utf8"
      )
    );
    await admin.query(
      `CREATE ROLE ${appRole} LOGIN PASSWORD '${appPassword}' ` +
        "NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS"
    );
    await admin.query(
      `GRANT SELECT, INSERT, UPDATE, DELETE ON
         public.teachlab_teaching_sessions,
         public.teachlab_agent_tasks,
         public.teachlab_task_events,
         public.teachlab_auth_sessions,
         public.teachlab_account_deletion_operations,
         public.teachlab_account_deletion_tombstones,
         public.teachlab_artifacts
       TO ${appRole}`
    );
    await admin.query(
      `GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ${appRole}`
    );

    database = new PostgresDatabase({
      dataBackend: "postgres",
      databaseUrl: applicationUrl(adminUrl),
      postgresPoolMax: 6,
      postgresSslMode: "disable",
      accountScopeKeys
    } as unknown as AppConfigService);
    await database.assertReady();

    const sessions = new PostgresSessionRepository(database);
    // Production Nest wiring injects PostgresDatabase through both the tenant
    // repository and TASK_SYSTEM_DATABASE boundaries. Mirror that wiring here
    // so lease claims exercise the dispatcher-only RLS transaction.
    const tasks = new PostgresTaskRepository(database, database);
    const artifacts = new PostgresArtifactStorageAdapter(database);
    let revocations = new PostgresSessionRevocationRepository(database);
    const alice = {tenantId: "school-a", ownerId: "alice"};
    const bob = {tenantId: "school-b", ownerId: "bob"};
    const aliceSession = await sessions.create({
      ...alice,
      title: "Algebra review",
      learner: "Alice"
    });
    const bobSession = await sessions.create({
      ...bob,
      title: "Biology review",
      learner: "Bob"
    });

    await database.withTenant(alice, (transaction) => transaction.query(
      `INSERT INTO public.teachlab_account_deletion_operations (
         tenant_id, owner_id, scope_sha256, operation_id, phase, revision,
         challenge_id, challenge_token_sha256, challenge_csrf_sha256,
         challenge_session_sha256, challenge_authority_grant_sha256,
         challenge_canonical_identity_sha256, challenge_issuer_sha256,
         challenge_authority_key_version, challenge_authenticated_at,
         challenge_assurance_level, challenge_expires_at,
         status_capability_sha256
       ) VALUES (
         $1, $2, $3, $4, 'prepared', 1, $5, $6, $7, $8, $9, $10, $11,
         'test-v1', statement_timestamp(), 2,
         statement_timestamp() + interval '5 minutes', $12
       )`,
      [
        alice.tenantId, alice.ownerId, "1".repeat(64),
        `adel_${"2".repeat(32)}`, `adelc_${"3".repeat(32)}`,
        "4".repeat(64), "5".repeat(64), "6".repeat(64), "7".repeat(64),
        "8".repeat(64), "9".repeat(64), "a".repeat(64)
      ]
    ));
    const hiddenFromOrdinarySystem = await database.withAccountSystem((transaction) =>
      transaction.query<{operation_id: string}>(
        "SELECT operation_id FROM public.teachlab_account_deletion_operations"
      )
    );
    assert.equal(hiddenFromOrdinarySystem.rowCount, 0);
    const visibleOnlyToRecoveryBoundary = await database.withAccountRecovery((transaction) =>
      transaction.query<{operation_id: string}>(
        "SELECT operation_id FROM public.teachlab_account_deletion_operations"
      )
    );
    assert.deepEqual(visibleOnlyToRecoveryBoundary.rows, [{
      operation_id: `adel_${"2".repeat(32)}`
    }]);

    assert.deepEqual(
      (await sessions.listByOwner(alice)).map((row) => row.id),
      [aliceSession.id]
    );
    assert.equal(await sessions.findOwnedById(alice, bobSession.id), undefined);
    const rlsRows = await database.withTenant(alice, (transaction) =>
      transaction.query<{tenant_id: string; owner_id: string}>(
        "SELECT tenant_id, owner_id FROM public.teachlab_teaching_sessions"
      )
    );
    assert.deepEqual(rlsRows.rows, [
      {tenant_id: alice.tenantId, owner_id: alice.ownerId}
    ]);
    await assert.rejects(
      database.withTenant(alice, (transaction) =>
        transaction.query(
          `INSERT INTO public.teachlab_teaching_sessions
             (id, tenant_id, owner_id, title, learner)
           VALUES ('00000000-0000-4000-8000-000000000099', $1, $2, 'forged', 'forged')`,
          [bob.tenantId, bob.ownerId]
        )
      ),
      /row-level security policy/
    );

    const updates = await Promise.all([
      sessions.updateOwned(alice, aliceSession.id, {round: 1}, 1),
      sessions.updateOwned(alice, aliceSession.id, {round: 2}, 1)
    ]);
    assert.deepEqual(
      updates.map((result) => result.kind).sort(),
      ["updated", "version_conflict"]
    );

    const command = {
      ...alice,
      sessionId: aliceSession.id,
      learnerMessage: "Please explain the next step",
      clientRequestId: "request-1"
    };
    const firstTask = await tasks.create(command);
    const replayedTask = await tasks.create(command);
    assert.equal(firstTask.created, true);
    assert.equal(replayedTask.created, false);
    assert.equal(firstTask.task.id, replayedTask.task.id);
    assert.equal(await tasks.findOwnedById(bob, firstTask.task.id), undefined);
    const firstTaskClaim = await tasks.claimNext({
      leaseOwnerSha256: "a".repeat(64),
      leaseTokenSha256: "b".repeat(64),
      leaseDurationMs: 30_000,
      maximumAttempts: 3
    });
    assert.equal(firstTaskClaim.kind, "claimed");
    assert.equal(
      (await tasks.settleLease({
        ...alice,
        taskId: firstTask.task.id,
        leaseTokenSha256: "b".repeat(64),
        status: "succeeded",
        events: [{type: "status", payload: {kind: "task.succeeded", taskId: firstTask.task.id}}]
      }))?.status,
      "succeeded"
    );

    const concurrentTask = await tasks.create({
      ...alice,
      sessionId: aliceSession.id,
      learnerMessage: "Claim me exactly once",
      clientRequestId: "claim-once-1"
    });
    const concurrentClaims = await Promise.all([
      tasks.claimNext({
        leaseOwnerSha256: "1".repeat(64),
        leaseTokenSha256: "2".repeat(64),
        leaseDurationMs: 30_000,
        maximumAttempts: 3
      }),
      tasks.claimNext({
        leaseOwnerSha256: "3".repeat(64),
        leaseTokenSha256: "4".repeat(64),
        leaseDurationMs: 30_000,
        maximumAttempts: 3
      })
    ]);
    assert.deepEqual(
      concurrentClaims.map((result) => result.kind).sort(),
      ["claimed", "none"]
    );
    const claimed = concurrentClaims.find((result) => result.kind === "claimed");
    assert.equal(claimed?.task.id, concurrentTask.task.id);
    assert.equal(claimed?.task.attemptCount, 1);

    const staleToken = await tasks.settleLease({
      ...alice,
      taskId: concurrentTask.task.id,
      leaseTokenSha256: "f".repeat(64),
      status: "succeeded",
      events: [{type: "status", payload: {kind: "task.succeeded", taskId: concurrentTask.task.id}}]
    });
    assert.equal(staleToken, undefined);

    const validToken = claimed?.kind === "claimed" ? claimed.leaseTokenSha256 : undefined;
    assert.ok(validToken);
    const settled = await tasks.settleLease({
      ...alice,
      taskId: concurrentTask.task.id,
      leaseTokenSha256: validToken,
      status: "succeeded",
      events: [{type: "status", payload: {kind: "task.succeeded", taskId: concurrentTask.task.id}}]
    });
    assert.equal(settled?.status, "succeeded");

    const recoveryTask = await tasks.create({
      ...alice,
      sessionId: aliceSession.id,
      learnerMessage: "Recover this expired lease",
      clientRequestId: "lease-recovery-1"
    });
    const firstRecoveryClaim = await tasks.claimNext({
      leaseOwnerSha256: "5".repeat(64),
      leaseTokenSha256: "6".repeat(64),
      leaseDurationMs: 1_000,
      maximumAttempts: 3
    });
    assert.equal(firstRecoveryClaim.kind, "claimed");
    await admin.query(
      `UPDATE public.teachlab_agent_tasks
          SET lease_expires_at = statement_timestamp() - interval '1 second'
        WHERE id = $1`,
      [recoveryTask.task.id]
    );
    const recovered = await tasks.claimNext({
      leaseOwnerSha256: "7".repeat(64),
      leaseTokenSha256: "8".repeat(64),
      leaseDurationMs: 30_000,
      maximumAttempts: 3
    });
    assert.equal(recovered.kind, "claimed");
    assert.equal(recovered.kind === "claimed" ? recovered.task.attemptCount : -1, 2);

    const retryTask = await tasks.create({
      ...alice,
      sessionId: aliceSession.id,
      learnerMessage: "Bound retries",
      clientRequestId: "retry-bound-1"
    });
    const retryClaim1 = await tasks.claimNext({
      leaseOwnerSha256: "9".repeat(64),
      leaseTokenSha256: "a".repeat(64),
      leaseDurationMs: 30_000,
      maximumAttempts: 2
    });
    assert.equal(retryClaim1.kind, "claimed");
    const retry1 = await tasks.retryLease({
      ...alice,
      taskId: retryTask.task.id,
      leaseTokenSha256: "a".repeat(64),
      failureCode: "provider_execution_failed",
      maximumAttempts: 2,
      retryBaseMs: 100,
      retryMaximumMs: 100,
      events: [{type: "error", payload: {kind: "task.failed", taskId: retryTask.task.id}}]
    });
    assert.equal(retry1.kind, "retry_scheduled");
    await admin.query(
      `UPDATE public.teachlab_agent_tasks
          SET available_at = statement_timestamp()
        WHERE id = $1`,
      [retryTask.task.id]
    );
    const retryClaim2 = await tasks.claimNext({
      leaseOwnerSha256: "b".repeat(64),
      leaseTokenSha256: "c".repeat(64),
      leaseDurationMs: 30_000,
      maximumAttempts: 2
    });
    assert.equal(retryClaim2.kind, "claimed");
    const retry2 = await tasks.retryLease({
      ...alice,
      taskId: retryTask.task.id,
      leaseTokenSha256: "c".repeat(64),
      failureCode: "provider_execution_failed",
      maximumAttempts: 2,
      retryBaseMs: 100,
      retryMaximumMs: 100,
      events: [{type: "error", payload: {kind: "task.failed", taskId: retryTask.task.id}}]
    });
    assert.equal(retry2.kind, "failed");
    assert.equal(retry2.kind === "failed" ? retry2.task.attemptCount : -1, 2);
    assert.equal(
      (await tasks.claimNext({
        leaseOwnerSha256: "d".repeat(64),
        leaseTokenSha256: "e".repeat(64),
        leaseDurationMs: 30_000,
        maximumAttempts: 2
      })).kind,
      "none"
    );

    const fencedTask = await tasks.create({
      ...alice,
      sessionId: aliceSession.id,
      learnerMessage: "Deletion fence must win",
      clientRequestId: "deletion-fence-1"
    });
    const fencedClaim = await tasks.claimNext({
      leaseOwnerSha256: "f".repeat(64),
      leaseTokenSha256: "0".repeat(64),
      leaseDurationMs: 30_000,
      maximumAttempts: 3
    });
    assert.equal(fencedClaim.kind, "claimed");
    await database.withTenant(alice, (transaction) => transaction.query(
      `UPDATE public.teachlab_account_deletion_operations
          SET phase = 'fencing', revision = revision + 1,
              idempotency_key_sha256 = $3,
              confirmation_sha256 = $4,
              updated_at = statement_timestamp()
        WHERE tenant_id = $1 AND owner_id = $2 AND phase = 'prepared'`,
      [alice.tenantId, alice.ownerId, "1".repeat(64), "2".repeat(64)]
    ), {allowAccountDeleting: true});
    const fencedClaimAfter = await tasks.claimNext({
      leaseOwnerSha256: "1".repeat(64),
      leaseTokenSha256: "3".repeat(64),
      leaseDurationMs: 30_000,
      maximumAttempts: 3
    });
    assert.equal(fencedClaimAfter.kind, "none");
    const fencedSettle = await tasks.settleLease({
      ...alice,
      taskId: fencedTask.task.id,
      leaseTokenSha256: "0".repeat(64),
      status: "succeeded",
      events: [{type: "status", payload: {kind: "task.succeeded", taskId: fencedTask.task.id}}]
    });
    assert.equal(fencedSettle, undefined);
    const fencedEvents = await database.withTenant(alice, (transaction) => transaction.query(
      `SELECT id FROM public.teachlab_task_events
        WHERE tenant_id = $1 AND owner_id = $2
          AND payload @> $3::jsonb`,
      [alice.tenantId, alice.ownerId, JSON.stringify({taskId: fencedTask.task.id})]
    ));
    assert.equal(fencedEvents.rowCount, 0);

    const storedArtifact = await artifacts.put({
      ...alice,
      key: "exports/lesson.json",
      contentType: "application/json",
      bytes: new TextEncoder().encode("{\"ok\":true}"),
      metadata: {kind: "test"}
    });
    assert.equal(storedArtifact.version, 1);
    assert.equal((await artifacts.get(alice, storedArtifact.key))?.artifact.sha256, storedArtifact.sha256);
    assert.equal(await artifacts.get(bob, storedArtifact.key), undefined);
    assert.equal((await artifacts.deleteScope(alice)).count, 1);

    const authorityInput = {
      ...alice,
      sessionIdSha256: "a".repeat(64),
      issuedAt: new Date("2026-08-12T00:00:00.000Z"),
      expiresAt: new Date("2099-08-12T08:00:00.000Z")
    };
    await revocations.register(authorityInput);
    await database.onModuleDestroy();
    database = new PostgresDatabase({
      dataBackend: "postgres",
      databaseUrl: applicationUrl(adminUrl),
      postgresPoolMax: 6,
      postgresSslMode: "disable",
      accountScopeKeys
    } as unknown as AppConfigService);
    await database.assertReady();
    revocations = new PostgresSessionRevocationRepository(database);
    assert.equal(
      (await revocations.inspect(alice, authorityInput.sessionIdSha256, new Date())).kind,
      "active"
    );
    assert.equal(
      (await revocations.inspect(bob, authorityInput.sessionIdSha256, new Date())).kind,
      "missing"
    );
    const revokeInput = {
      ...alice,
      sessionIdSha256: authorityInput.sessionIdSha256,
      revokedAt: new Date("2026-08-12T01:00:00.000Z"),
      reason: "user_logout" as const
    };
    const concurrent = await Promise.all([
      revocations.revoke(revokeInput),
      revocations.revoke(revokeInput)
    ]);
    assert.deepEqual(
      concurrent.map((result) => result.kind).sort(),
      ["already_revoked", "revoked"]
    );
    const rows = await database.withTenant(alice, (transaction) =>
      transaction.query<Record<string, unknown>>(
        "SELECT * FROM public.teachlab_auth_sessions"
      )
    );
    assert.deepEqual(
      Object.keys(rows.rows[0] ?? {}).sort(),
      [
        "expires_at",
        "issued_at",
        "owner_id",
        "revocation_reason",
        "revoked_at",
        "session_id_sha256",
        "tenant_id",
        "version"
      ]
    );
  }
);
