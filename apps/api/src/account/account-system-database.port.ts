import type {DatabaseQueryResult} from "../database/tenant-database.port";

export interface AccountSystemTransaction {
  query<Row>(sql: string, parameters?: unknown[]): Promise<DatabaseQueryResult<Row>>;
}

/**
 * Narrow unscoped database boundary for hash-only account tombstones.
 * Implementations must use a non-BYPASSRLS application role and expose this
 * port only to the account lifecycle repository.
 */
export interface AccountSystemDatabasePort {
  withAccountSystem<T>(
    operation: (transaction: AccountSystemTransaction) => Promise<T>
  ): Promise<T>;

  /**
   * Internal-only read/write boundary used to lease unfinished deletion rows.
   * The implementation enables the dedicated FORCE-RLS recovery policy for
   * one transaction; no controller or worker receives this capability.
   */
  withAccountRecovery<T>(
    operation: (transaction: AccountSystemTransaction) => Promise<T>
  ): Promise<T>;
}
