import {Inject, Injectable} from "@nestjs/common";

import {TENANT_DATABASE} from "../platform/tokens";
import type {AccessScope} from "../tenancy/access-scope";
import type {TenantDatabasePort, TenantTransaction} from "../database/tenant-database.port";
import type {
  RegisterSessionAuthorityInput,
  RevokeSessionAuthorityInput,
  RevokeSessionAuthorityResult,
  SessionAuthorityRecord,
  SessionAuthorityInspection,
  SessionAuthorityState,
  SessionRevocationReason,
  SessionRevocationRepositoryPort
} from "./session-revocation.repository.port";

interface SessionAuthorityRow {
  tenant_id: string;
  owner_id: string;
  session_id_sha256: string;
  issued_at: Date | string;
  expires_at: Date | string;
  revoked_at: Date | string | null;
  revocation_reason: SessionRevocationReason | null;
  version: number;
}

interface DatabaseClockRow {
  database_now: Date | string;
}

const CLEANUP_LIMIT_MAX = 256;

function record(row: SessionAuthorityRow): SessionAuthorityRecord {
  return {
    tenantId: row.tenant_id,
    ownerId: row.owner_id,
    sessionIdSha256: row.session_id_sha256,
    issuedAt: new Date(row.issued_at),
    expiresAt: new Date(row.expires_at),
    revokedAt: row.revoked_at ? new Date(row.revoked_at) : null,
    revocationReason: row.revocation_reason,
    version: Number(row.version)
  };
}

@Injectable()
export class PostgresSessionRevocationRepository
  implements SessionRevocationRepositoryPort {
  constructor(
    @Inject(TENANT_DATABASE) private readonly database: TenantDatabasePort
  ) {}

  async register(input: RegisterSessionAuthorityInput): Promise<SessionAuthorityRecord> {
    return this.database.withTenant(input, async (transaction) => {
      const databaseNow = await this.databaseNow(transaction);
      if (input.expiresAt.getTime() <= databaseNow.getTime()) {
        throw new Error("Session authority is already expired at the database boundary");
      }
      await this.cleanup(transaction, input, databaseNow, CLEANUP_LIMIT_MAX);
      const result = await transaction.query<SessionAuthorityRow>(
        `INSERT INTO public.teachlab_auth_sessions
           (tenant_id, owner_id, session_id_sha256, issued_at, expires_at)
         VALUES ($1, $2, $3, $4, $5)
         ON CONFLICT (tenant_id, owner_id, session_id_sha256) DO UPDATE
           SET session_id_sha256 = EXCLUDED.session_id_sha256
         WHERE teachlab_auth_sessions.issued_at = EXCLUDED.issued_at
           AND teachlab_auth_sessions.expires_at = EXCLUDED.expires_at
         RETURNING tenant_id, owner_id, session_id_sha256, issued_at, expires_at,
                   revoked_at, revocation_reason, version`,
        [
          input.tenantId,
          input.ownerId,
          input.sessionIdSha256,
          input.issuedAt,
          input.expiresAt
        ]
      );
      const row = result.rows[0];
      if (!row) throw new Error("Session authority registration conflict");
      return record(row);
    });
  }

  async inspect(
    scope: AccessScope,
    sessionIdSha256: string,
    now: Date
  ): Promise<SessionAuthorityState> {
    return this.database.withTenant(scope, async (transaction) => {
      await this.cleanup(transaction, scope, now, CLEANUP_LIMIT_MAX);
      const result = await transaction.query<SessionAuthorityRow>(
        `SELECT tenant_id, owner_id, session_id_sha256, issued_at, expires_at,
                revoked_at, revocation_reason, version
           FROM public.teachlab_auth_sessions
          WHERE tenant_id = $1 AND owner_id = $2 AND session_id_sha256 = $3
            AND expires_at > $4`,
        [scope.tenantId, scope.ownerId, sessionIdSha256, now]
      );
      const row = result.rows[0];
      if (!row) return {kind: "missing"};
      const value = record(row);
      return value.revokedAt
        ? {kind: "revoked", record: value}
        : {kind: "active", record: value};
    }, {allowAccountDeleting: true});
  }

  async inspectAuthoritatively(
    scope: AccessScope,
    sessionIdSha256: string
  ): Promise<SessionAuthorityInspection> {
    return this.database.withTenant(scope, async (transaction) => {
      const databaseNow = await this.databaseNow(transaction);
      await this.cleanup(transaction, scope, databaseNow, CLEANUP_LIMIT_MAX);
      const result = await transaction.query<SessionAuthorityRow>(
        `SELECT tenant_id, owner_id, session_id_sha256, issued_at, expires_at,
                revoked_at, revocation_reason, version
           FROM public.teachlab_auth_sessions
          WHERE tenant_id = $1 AND owner_id = $2 AND session_id_sha256 = $3
            AND expires_at > $4`,
        [scope.tenantId, scope.ownerId, sessionIdSha256, databaseNow]
      );
      const row = result.rows[0];
      if (!row) return {kind: "missing", databaseNow};
      const value = record(row);
      return value.revokedAt
        ? {kind: "revoked", record: value, databaseNow}
        : {kind: "active", record: value, databaseNow};
    }, {allowAccountDeleting: true});
  }

  async revoke(input: RevokeSessionAuthorityInput): Promise<RevokeSessionAuthorityResult> {
    return this.database.withTenant(input, async (transaction) => {
      const databaseNow = await this.databaseNow(transaction);
      await this.cleanup(transaction, input, databaseNow, CLEANUP_LIMIT_MAX);
      const selected = await transaction.query<SessionAuthorityRow>(
        `SELECT tenant_id, owner_id, session_id_sha256, issued_at, expires_at,
                revoked_at, revocation_reason, version
           FROM public.teachlab_auth_sessions
          WHERE tenant_id = $1 AND owner_id = $2 AND session_id_sha256 = $3
            AND expires_at > $4
          FOR UPDATE`,
        [input.tenantId, input.ownerId, input.sessionIdSha256, databaseNow]
      );
      const current = selected.rows[0];
      if (!current) return {kind: "missing"};
      if (current.revoked_at) {
        return {kind: "already_revoked", record: record(current)};
      }
      const updated = await transaction.query<SessionAuthorityRow>(
        `UPDATE public.teachlab_auth_sessions
            SET revoked_at = $4, revocation_reason = $5, version = version + 1
          WHERE tenant_id = $1 AND owner_id = $2 AND session_id_sha256 = $3
            AND revoked_at IS NULL
            AND version = $6
          RETURNING tenant_id, owner_id, session_id_sha256, issued_at, expires_at,
                    revoked_at, revocation_reason, version`,
        [
          input.tenantId,
          input.ownerId,
          input.sessionIdSha256,
          databaseNow,
          input.reason,
          current.version
        ]
      );
      const row = updated.rows[0];
      if (!row) throw new Error("Session authority revocation CAS failed");
      return {kind: "revoked", record: record(row)};
    });
  }

  async cleanupExpired(scope: AccessScope, now: Date, limit: number): Promise<number> {
    return this.database.withTenant(scope, (transaction) =>
      this.cleanup(transaction, scope, now, limit)
    );
  }

  private async cleanup(
    transaction: TenantTransaction,
    scope: AccessScope,
    now: Date,
    limit: number
  ): Promise<number> {
    const bounded = Math.max(1, Math.min(CLEANUP_LIMIT_MAX, Math.trunc(limit)));
    const result = await transaction.query(
      `DELETE FROM public.teachlab_auth_sessions
        WHERE ctid IN (
          SELECT ctid
            FROM public.teachlab_auth_sessions
           WHERE tenant_id = $1 AND owner_id = $2 AND expires_at <= $3
           ORDER BY expires_at ASC
           LIMIT $4
           FOR UPDATE SKIP LOCKED
        )`,
      [scope.tenantId, scope.ownerId, now, bounded]
    );
    return result.rowCount;
  }

  private async databaseNow(transaction: TenantTransaction): Promise<Date> {
    const clock = await transaction.query<DatabaseClockRow>(
      "SELECT statement_timestamp() AS database_now"
    );
    const databaseNow = new Date(clock.rows[0]?.database_now ?? Number.NaN);
    if (!Number.isFinite(databaseNow.getTime())) {
      throw new Error("PostgreSQL session authority clock is unavailable");
    }
    return databaseNow;
  }
}
