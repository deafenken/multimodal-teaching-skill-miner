import {randomUUID} from "node:crypto";

import {Inject, Injectable} from "@nestjs/common";

import type {TenantDatabasePort} from "../database/tenant-database.port";
import {TENANT_DATABASE} from "../platform/tokens";
import type {AccessScope} from "../tenancy/access-scope";
import type {SessionRepositoryPort} from "./session-repository.port";
import type {
  CreateSessionRecord,
  SessionStatus,
  TeachingSession,
  UpdateSessionRecord,
  UpdateSessionResult
} from "./session.types";

interface SessionRow {
  id: string;
  tenant_id: string;
  owner_id: string;
  title: string;
  learner: string;
  status: SessionStatus;
  round: number;
  created_at: Date | string;
  updated_at: Date | string;
  version: number;
}

const SESSION_COLUMNS = `
  id, tenant_id, owner_id, title, learner, status, round,
  created_at, updated_at, version
`;
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function timestamp(value: Date | string): string {
  return value instanceof Date ? value.toISOString() : new Date(value).toISOString();
}

function mapSession(row: SessionRow): TeachingSession {
  return {
    id: row.id,
    tenantId: row.tenant_id,
    ownerId: row.owner_id,
    title: row.title,
    learner: row.learner,
    status: row.status,
    round: row.round,
    createdAt: timestamp(row.created_at),
    updatedAt: timestamp(row.updated_at),
    version: row.version
  };
}

@Injectable()
export class PostgresSessionRepository implements SessionRepositoryPort {
  constructor(@Inject(TENANT_DATABASE) private readonly database: TenantDatabasePort) {}

  async create(input: CreateSessionRecord): Promise<TeachingSession> {
    const scope = {tenantId: input.tenantId, ownerId: input.ownerId};
    return this.database.withTenant(scope, async (transaction) => {
      const result = await transaction.query<SessionRow>(
        `INSERT INTO public.teachlab_teaching_sessions (
           id, tenant_id, owner_id, title, learner, status, round
         ) VALUES ($1, $2, $3, $4, $5, $6, $7)
         RETURNING ${SESSION_COLUMNS}`,
        [
          input.id ?? randomUUID(),
          input.tenantId,
          input.ownerId,
          input.title,
          input.learner,
          input.status ?? "active",
          input.round ?? 0
        ]
      );
      const row = result.rows[0];
      if (!row) throw new Error("PostgreSQL did not return the created session");
      return mapSession(row);
    });
  }

  listByOwner(scope: AccessScope): Promise<TeachingSession[]> {
    return this.database.withTenant(scope, async (transaction) => {
      const result = await transaction.query<SessionRow>(
        `SELECT ${SESSION_COLUMNS}
           FROM public.teachlab_teaching_sessions
          WHERE tenant_id = $1 AND owner_id = $2
          ORDER BY updated_at DESC`,
        [scope.tenantId, scope.ownerId]
      );
      return result.rows.map(mapSession);
    });
  }

  findOwnedById(scope: AccessScope, id: string): Promise<TeachingSession | undefined> {
    if (!UUID_PATTERN.test(id)) return Promise.resolve(undefined);
    return this.database.withTenant(scope, async (transaction) => {
      const result = await transaction.query<SessionRow>(
        `SELECT ${SESSION_COLUMNS}
           FROM public.teachlab_teaching_sessions
          WHERE tenant_id = $1 AND owner_id = $2 AND id = $3`,
        [scope.tenantId, scope.ownerId, id]
      );
      return result.rows[0] ? mapSession(result.rows[0]) : undefined;
    });
  }

  updateOwned(
    scope: AccessScope,
    id: string,
    patch: UpdateSessionRecord,
    expectedVersion: number
  ): Promise<UpdateSessionResult> {
    if (!UUID_PATTERN.test(id)) return Promise.resolve({kind: "not_found"});
    return this.database.withTenant(scope, async (transaction) => {
      const locked = await transaction.query<SessionRow>(
        `SELECT ${SESSION_COLUMNS}
           FROM public.teachlab_teaching_sessions
          WHERE tenant_id = $1 AND owner_id = $2 AND id = $3
          FOR UPDATE`,
        [scope.tenantId, scope.ownerId, id]
      );
      const current = locked.rows[0];
      if (!current) return {kind: "not_found"};
      if (current.version !== expectedVersion) {
        return {kind: "version_conflict", currentVersion: current.version};
      }
      if (
        patch.status &&
        current.status !== "active" &&
        patch.status !== current.status
      ) {
        return {kind: "state_conflict", currentStatus: current.status};
      }
      const updated = await transaction.query<SessionRow>(
        `UPDATE public.teachlab_teaching_sessions
            SET title = COALESCE($4, title),
                learner = COALESCE($5, learner),
                status = COALESCE($6, status),
                round = COALESCE($7, round),
                updated_at = NOW(),
                version = version + 1
          WHERE tenant_id = $1 AND owner_id = $2 AND id = $3
          RETURNING ${SESSION_COLUMNS}`,
        [
          scope.tenantId,
          scope.ownerId,
          id,
          patch.title ?? null,
          patch.learner ?? null,
          patch.status ?? null,
          patch.round ?? null
        ]
      );
      const row = updated.rows[0];
      if (!row) throw new Error("Locked session disappeared during update");
      return {kind: "updated", session: mapSession(row)};
    });
  }
}
