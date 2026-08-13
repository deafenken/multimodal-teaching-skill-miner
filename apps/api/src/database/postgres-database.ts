import {Inject, Injectable, OnModuleDestroy} from "@nestjs/common";
import {Pool} from "pg";

import {AppConfigService} from "../config/app-config.service";
import {AccountDataRightsError} from "../account/account-data-rights.errors";
import {AccountScopeHasher} from "../account/account-scope-hash";
import type {
  AccountSystemDatabasePort,
  AccountSystemTransaction
} from "../account/account-system-database.port";
import type {AccessScope} from "../tenancy/access-scope";
import type {TenantDatabasePort, TenantTransaction} from "./tenant-database.port";
import type {TaskSystemDatabasePort, TaskSystemTransaction} from "../tasks/task-system-database.port";

export interface PostgresClientContract {
  query(
    sql: string,
    parameters?: unknown[]
  ): Promise<{rows: unknown[]; rowCount: number | null}>;
  release(): void;
}

export interface PostgresReadinessRow {
  role_name: string;
  role_is_superuser: boolean;
  role_bypasses_rls: boolean;
  owned_table_count: number | string;
  table_count: number | string;
  rls_enforced: boolean;
  auth_session_columns_safe: boolean;
  account_recovery_columns_safe: boolean;
  account_recovery_policy_present: boolean;
  tombstone_table_present: boolean;
  artifact_table_present?: boolean;
  task_lease_columns_safe?: boolean;
  task_dispatcher_policy_present?: boolean;
  artifact_rls_enforced?: boolean;
}

export const POSTGRES_READINESS_SQL = `
  WITH expected_tables(name) AS (
    VALUES
      ('public.teachlab_teaching_sessions'),
      ('public.teachlab_agent_tasks'),
      ('public.teachlab_task_events'),
      ('public.teachlab_auth_sessions'),
      ('public.teachlab_account_deletion_operations')
  ), auth_session_columns AS (
    SELECT ARRAY_AGG(column_name::text ORDER BY column_name::text) AS names
      FROM information_schema.columns
     WHERE table_schema = 'public'
       AND table_name = 'teachlab_auth_sessions'
  ), account_recovery_columns AS (
    SELECT ARRAY_AGG(column_name::text ORDER BY column_name::text) AS names
      FROM information_schema.columns
     WHERE table_schema = 'public'
       AND table_name = 'teachlab_account_deletion_operations'
       AND column_name IN (
         'recovery_after', 'recovery_attempts', 'recovery_lease_expires_at',
         'recovery_lease_owner_sha256', 'recovery_lease_token_sha256'
       )
  ), account_recovery_policy AS (
    SELECT EXISTS (
      SELECT 1 FROM pg_policies
       WHERE schemaname = 'public'
         AND tablename = 'teachlab_account_deletion_operations'
         AND policyname = 'teachlab_account_deletion_operations_recovery'
    ) AS present
  ), task_lease_columns AS (
    SELECT ARRAY_AGG(column_name::text ORDER BY column_name::text) AS names
      FROM information_schema.columns
     WHERE table_schema = 'public'
       AND table_name = 'teachlab_agent_tasks'
       AND column_name IN (
         'attempt_count', 'available_at', 'lease_expires_at',
         'lease_owner_sha256', 'lease_token_sha256'
       )
  ), task_dispatcher_policy AS (
    SELECT EXISTS (
      SELECT 1 FROM pg_policies
       WHERE schemaname = 'public'
         AND tablename = 'teachlab_agent_tasks'
         AND policyname = 'teachlab_tasks_dispatcher'
    ) AS present
  ), artifact_state AS (
    SELECT
      to_regclass('public.teachlab_artifacts') IS NOT NULL AS present,
      COALESCE((
        SELECT relrowsecurity AND relforcerowsecurity
          FROM pg_class
         WHERE oid = to_regclass('public.teachlab_artifacts')
      ), false) AS rls_enforced
  )
  SELECT current_user AS role_name,
         roles.rolsuper AS role_is_superuser,
         roles.rolbypassrls AS role_bypasses_rls,
         COUNT(classes.oid)::int AS table_count,
         COUNT(classes.oid) FILTER (
           WHERE classes.relowner = roles.oid
         )::int AS owned_table_count,
         COALESCE(
           BOOL_AND(classes.relrowsecurity AND classes.relforcerowsecurity),
           false
         ) AS rls_enforced,
         auth_session_columns.names = ARRAY[
           'expires_at', 'issued_at', 'owner_id', 'revocation_reason',
           'revoked_at', 'session_id_sha256', 'tenant_id', 'version'
         ]::text[] AS auth_session_columns_safe,
         account_recovery_columns.names = ARRAY[
           'recovery_after', 'recovery_attempts', 'recovery_lease_expires_at',
           'recovery_lease_owner_sha256', 'recovery_lease_token_sha256'
         ]::text[] AS account_recovery_columns_safe,
         account_recovery_policy.present AS account_recovery_policy_present,
         to_regclass('public.teachlab_account_deletion_tombstones') IS NOT NULL
           AS tombstone_table_present,
         to_regclass('public.teachlab_artifacts') IS NOT NULL
           AS artifact_table_present,
         artifact_state.rls_enforced AS artifact_rls_enforced,
         task_lease_columns.names = ARRAY[
           'attempt_count', 'available_at', 'lease_expires_at',
           'lease_owner_sha256', 'lease_token_sha256'
         ]::text[] AS task_lease_columns_safe,
         task_dispatcher_policy.present AS task_dispatcher_policy_present
    FROM pg_roles AS roles
    CROSS JOIN expected_tables
    CROSS JOIN auth_session_columns
    CROSS JOIN account_recovery_columns
    CROSS JOIN account_recovery_policy
    CROSS JOIN task_lease_columns
    CROSS JOIN task_dispatcher_policy
    CROSS JOIN artifact_state
    LEFT JOIN pg_class AS classes
      ON classes.oid = to_regclass(expected_tables.name)
   WHERE roles.rolname = current_user
   GROUP BY roles.rolsuper, roles.rolbypassrls, auth_session_columns.names,
            account_recovery_columns.names, account_recovery_policy.present,
            task_lease_columns.names, task_dispatcher_policy.present,
            artifact_state.present, artifact_state.rls_enforced
`;

export function assertPostgresReadiness(
  row: PostgresReadinessRow | undefined,
  options: {requireDurable?: boolean} = {}
): void {
  if (!row) throw new Error("PostgreSQL readiness query returned no role information");
  if (row.role_is_superuser || row.role_bypasses_rls) {
    throw new Error(
      `PostgreSQL role ${row.role_name} must not be superuser or BYPASSRLS`
    );
  }
  if (Number(row.owned_table_count) !== 0) {
    throw new Error(`PostgreSQL application role ${row.role_name} must not own tenant tables`);
  }
  if (
    Number(row.table_count) !== 5 ||
    !row.rls_enforced ||
    row.auth_session_columns_safe !== true ||
    row.account_recovery_columns_safe !== true ||
    row.account_recovery_policy_present !== true ||
    row.tombstone_table_present !== true ||
    (options.requireDurable && (
      row.artifact_table_present !== true ||
      row.artifact_rls_enforced !== true ||
      row.task_lease_columns_safe !== true ||
      row.task_dispatcher_policy_present !== true
    ))
  ) {
    throw new Error(
      "PostgreSQL tenant migration is missing or RLS is not forced on every tenant table"
    );
  }
}

export async function runTenantTransaction<T>(
  client: PostgresClientContract,
  scope: AccessScope,
  operation: (transaction: TenantTransaction) => Promise<T>,
  lifecycle?: {
    scopeHashes: readonly string[];
    allowAccountDeleting?: boolean;
  }
): Promise<T> {
  try {
    await client.query("BEGIN");
    await client.query(
      "SELECT set_config('app.tenant_id', $1, true), set_config('app.user_id', $2, true)",
      [scope.tenantId, scope.ownerId]
    );
    if (lifecycle) {
      // The same transaction-scoped lock serializes every scoped operation
      // with the deletion phase transition/final commit. A request which
      // passed the HTTP guard cannot insert after fencing has started.
      const lockKey = JSON.stringify([scope.tenantId, scope.ownerId]);
      await client.query(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 7640891576956012809))",
        [lockKey]
      );
      const state = await client.query(
        `SELECT
           EXISTS (
             SELECT 1 FROM public.teachlab_account_deletion_operations
              WHERE tenant_id = $1 AND owner_id = $2 AND phase <> 'prepared'
           ) AS deleting,
           EXISTS (
             SELECT 1 FROM public.teachlab_account_deletion_tombstones
              WHERE scope_sha256 = ANY($3::text[])
           ) AS deleted`,
        [scope.tenantId, scope.ownerId, [...lifecycle.scopeHashes]]
      );
      const row = state.rows[0] as {deleting?: unknown; deleted?: unknown} | undefined;
      if (!row || typeof row.deleting !== "boolean" || typeof row.deleted !== "boolean") {
        throw new Error("Account lifecycle predicate is unavailable");
      }
      if (!lifecycle.allowAccountDeleting) {
        if (row.deleted) throw new AccountDataRightsError(410, "account_already_deleted");
        if (row.deleting) {
          throw new AccountDataRightsError(409, "account_deletion_already_started");
        }
      }
    }
    const transaction: TenantTransaction = {
      query: async <Row>(sql: string, parameters: unknown[] = []) => {
        const result = await client.query(sql, parameters);
        return {rows: result.rows as Row[], rowCount: result.rowCount ?? 0};
      }
    };
    const result = await operation(transaction);
    await client.query("COMMIT");
    return result;
  } catch (error) {
    await client.query("ROLLBACK").catch(() => undefined);
    throw error;
  } finally {
    client.release();
  }
}

@Injectable()
export class PostgresDatabase
  implements TenantDatabasePort, AccountSystemDatabasePort, TaskSystemDatabasePort, OnModuleDestroy {
  private readonly pool?: Pool;
  private readonly scopeHasher: AccountScopeHasher;

  constructor(@Inject(AppConfigService) config: AppConfigService) {
    this.scopeHasher = new AccountScopeHasher(
      config.accountScopeKeys[0]!.secret,
      config.accountScopeKeys.slice(1).map((key) => key.secret)
    );
    if (config.dataBackend === "postgres" && config.databaseUrl) {
      this.pool = new Pool({
        connectionString: config.databaseUrl,
        max: config.postgresPoolMax,
        connectionTimeoutMillis: 2_000,
        ssl:
          config.postgresSslMode === "disable"
            ? false
            : {rejectUnauthorized: config.postgresSslMode === "verify-full"},
        application_name: "teachlab-api"
      });
    }
  }

  async withTenant<T>(
    scope: AccessScope,
    operation: (transaction: TenantTransaction) => Promise<T>,
    options: {allowAccountDeleting?: boolean} = {}
  ): Promise<T> {
    if (!this.pool) throw new Error("PostgreSQL data backend is not configured");
    const client = await this.pool.connect();
    return runTenantTransaction(
      client as unknown as PostgresClientContract,
      scope,
      operation,
      {
        scopeHashes: this.scopeHasher.hashes(scope).candidates,
        allowAccountDeleting: options.allowAccountDeleting
      }
    );
  }

  async withAccountSystem<T>(
    operation: (transaction: AccountSystemTransaction) => Promise<T>
  ): Promise<T> {
    if (!this.pool) throw new Error("PostgreSQL data backend is not configured");
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN READ ONLY");
      const transaction: AccountSystemTransaction = {
        query: async <Row>(sql: string, parameters: unknown[] = []) => {
          const result = await client.query(sql, parameters);
          return {rows: result.rows as Row[], rowCount: result.rowCount ?? 0};
        }
      };
      const result = await operation(transaction);
      await client.query("COMMIT");
      return result;
    } catch (error) {
      await client.query("ROLLBACK").catch(() => undefined);
      throw error;
    } finally {
      client.release();
    }
  }

  async withAccountRecovery<T>(
    operation: (transaction: AccountSystemTransaction) => Promise<T>
  ): Promise<T> {
    if (!this.pool) throw new Error("PostgreSQL data backend is not configured");
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      await client.query(
        "SELECT set_config('app.account_deletion_recovery', 'enabled', true)"
      );
      const transaction: AccountSystemTransaction = {
        query: async <Row>(sql: string, parameters: unknown[] = []) => {
          const result = await client.query(sql, parameters);
          return {rows: result.rows as Row[], rowCount: result.rowCount ?? 0};
        }
      };
      const result = await operation(transaction);
      await client.query("COMMIT");
      return result;
    } catch (error) {
      await client.query("ROLLBACK").catch(() => undefined);
      throw error;
    } finally {
      client.release();
    }
  }

  async withTaskDispatcher<T>(
    operation: (transaction: TaskSystemTransaction) => Promise<T>
  ): Promise<T> {
    if (!this.pool) throw new Error("PostgreSQL data backend is not configured");
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      await client.query(
        "SELECT set_config('app.task_dispatcher', 'enabled', true)"
      );
      const transaction: TaskSystemTransaction = {
        query: async <Row>(sql: string, parameters: unknown[] = []) => {
          const result = await client.query(sql, parameters);
          return {rows: result.rows as Row[], rowCount: result.rowCount ?? 0};
        }
      };
      const result = await operation(transaction);
      await client.query("COMMIT");
      return result;
    } catch (error) {
      await client.query("ROLLBACK").catch(() => undefined);
      throw error;
    } finally {
      client.release();
    }
  }

  async assertReady(): Promise<void> {
    if (!this.pool) return;
    const result = await this.pool.query<PostgresReadinessRow>(POSTGRES_READINESS_SQL);
    assertPostgresReadiness(result.rows[0], {requireDurable: true});
  }

  /** Run a bounded live dependency probe without exposing database details. */
  async probe(timeoutMs = 2_000): Promise<"postgres" | "not_configured"> {
    if (!this.pool) return "not_configured";
    if (!Number.isInteger(timeoutMs) || timeoutMs < 100 || timeoutMs > 30_000) {
      throw new Error("PostgreSQL probe timeout is invalid");
    }
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      // timeoutMs is a validated bounded integer and is not user-controlled.
      await client.query(`SET LOCAL statement_timeout = '${timeoutMs}ms'`);
      const result = await client.query<{ready: number}>("SELECT 1::int AS ready");
      await client.query("COMMIT");
      if (result.rowCount !== 1 || result.rows[0]?.ready !== 1) {
        throw new Error("PostgreSQL live probe failed");
      }
      return "postgres";
    } catch (error) {
      await client.query("ROLLBACK").catch(() => undefined);
      throw error;
    } finally {
      client.release();
    }
  }

  async onModuleDestroy(): Promise<void> {
    await this.pool?.end();
  }
}
