import type {DatabaseQueryResult} from "../database/tenant-database.port";

export interface TaskSystemTransaction {
  query<Row>(sql: string, parameters?: unknown[]): Promise<DatabaseQueryResult<Row>>;
}

/**
 * Internal-only PostgreSQL boundary for claiming work across tenant scopes.
 *
 * Implementations enable a transaction-local FORCE-RLS policy marker. This
 * capability must never be injected into controllers, browser-facing
 * services, providers, or worker sandboxes.
 */
export interface TaskSystemDatabasePort {
  withTaskDispatcher<T>(
    operation: (transaction: TaskSystemTransaction) => Promise<T>
  ): Promise<T>;
}
