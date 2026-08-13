import type {AccessScope} from "../tenancy/access-scope";

export interface DatabaseQueryResult<Row> {
  rows: Row[];
  rowCount: number;
}

export interface TenantTransaction {
  query<Row>(sql: string, parameters?: unknown[]): Promise<DatabaseQueryResult<Row>>;
}

export interface TenantDatabasePort {
  withTenant<T>(
    scope: AccessScope,
    operation: (transaction: TenantTransaction) => Promise<T>,
    options?: {allowAccountDeleting?: boolean}
  ): Promise<T>;
}
