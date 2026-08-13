import {createHash} from "node:crypto";

import {Inject, Injectable} from "@nestjs/common";

import type {TenantDatabasePort} from "../database/tenant-database.port";
import {TENANT_DATABASE} from "./tokens";
import type {AccessScope} from "../tenancy/access-scope";
import type {
  ArtifactStoragePort,
  StoreArtifactInput,
  StoredArtifact
} from "./artifact-storage.port";

export const MAX_ARTIFACT_BYTES = 16 * 1024 * 1024;
const KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._\/-]{0,511}$/;
const CONTENT_TYPE_PATTERN = /^[\x21-\x7e]{1,255}$/;
const HASH_PATTERN = /^[0-9a-f]{64}$/;
const METADATA_KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;

interface ArtifactRow {
  artifact_key: string;
  content_type: string;
  byte_length: number | string;
  sha256: string;
  metadata: Record<string, string> | string;
  content: Buffer | Uint8Array;
  created_at: Date | string;
  updated_at: Date | string;
  version: number;
}

const ARTIFACT_COLUMNS = `
  artifact_key, content_type, byte_length, sha256, metadata, content,
  created_at, updated_at, version
`;

function validateKey(key: string): void {
  if (
    typeof key !== "string"
    || !KEY_PATTERN.test(key)
    || key.includes("//")
    || key.split("/").some((part) => part === "." || part === "..")
  ) throw new Error("Invalid artifact key");
}

function validateMetadata(value: unknown): Record<string, string> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Invalid artifact metadata");
  }
  const entries = Object.entries(value);
  if (
    entries.length > 64
    || entries.some(([key, entry]) =>
      !METADATA_KEY_PATTERN.test(key)
      || typeof entry !== "string"
      || Buffer.byteLength(entry, "utf8") > 1_024
      || /[\u0000\r\n]/.test(entry)
    )
  ) throw new Error("Invalid artifact metadata");
  return Object.fromEntries(entries) as Record<string, string>;
}

function validate(input: StoreArtifactInput): void {
  if (
    !input
    || typeof input.key !== "string"
    || typeof input.contentType !== "string"
    || !(input.bytes instanceof Uint8Array)
  ) throw new Error("Invalid artifact content");
  validateKey(input.key);
  if (
    !CONTENT_TYPE_PATTERN.test(input.contentType)
    || input.bytes.byteLength > MAX_ARTIFACT_BYTES
  ) throw new Error("Invalid artifact content");
  validateMetadata(input.metadata ?? {});
}

function timestamp(value: Date | string): string {
  const parsed = value instanceof Date ? new Date(value.getTime()) : new Date(value);
  if (!Number.isFinite(parsed.getTime())) {
    throw new Error("Invalid PostgreSQL artifact timestamp");
  }
  return parsed.toISOString();
}

function metadata(value: Record<string, string> | string): Record<string, string> {
  let parsed: unknown;
  try {
    parsed = typeof value === "string" ? JSON.parse(value) : value;
  } catch {
    throw new Error("Invalid PostgreSQL artifact metadata");
  }
  try {
    return validateMetadata(parsed);
  } catch {
    throw new Error("Invalid PostgreSQL artifact metadata");
  }
}

function byteLength(value: number | string): number {
  const size = Number(value);
  if (!Number.isSafeInteger(size) || size < 0 || size > MAX_ARTIFACT_BYTES) {
    throw new Error("Invalid PostgreSQL artifact length");
  }
  return size;
}

function stored(row: ArtifactRow): StoredArtifact {
  if (typeof row.artifact_key !== "string") {
    throw new Error("Invalid PostgreSQL artifact key");
  }
  validateKey(row.artifact_key);
  if (typeof row.content_type !== "string" || !CONTENT_TYPE_PATTERN.test(row.content_type)) {
    throw new Error("Invalid PostgreSQL artifact content type");
  }
  if (typeof row.sha256 !== "string" || !HASH_PATTERN.test(row.sha256)) {
    throw new Error("Invalid PostgreSQL artifact digest");
  }
  const size = byteLength(row.byte_length);
  const version = Number(row.version);
  if (!Number.isSafeInteger(version) || version < 1) {
    throw new Error("Invalid PostgreSQL artifact version");
  }
  return {
    key: row.artifact_key,
    contentType: row.content_type,
    size,
    sha256: row.sha256,
    metadata: metadata(row.metadata),
    createdAt: timestamp(row.created_at),
    updatedAt: timestamp(row.updated_at),
    version
  };
}

function assertIntegrity(row: ArtifactRow, artifact: StoredArtifact): void {
  if (!(Buffer.isBuffer(row.content) || row.content instanceof Uint8Array)) {
    throw new Error("PostgreSQL artifact content is unavailable");
  }
  const bytes = Buffer.from(row.content);
  if (
    bytes.byteLength !== artifact.size
    || createHash("sha256").update(bytes).digest("hex") !== artifact.sha256
  ) throw new Error("PostgreSQL artifact integrity check failed");
}

@Injectable()
export class PostgresArtifactStorageAdapter implements ArtifactStoragePort {
  constructor(@Inject(TENANT_DATABASE) private readonly database: TenantDatabasePort) {}

  put(input: StoreArtifactInput): Promise<StoredArtifact> {
    validate(input);
    const scope = {tenantId: input.tenantId, ownerId: input.ownerId};
    const bytes = Buffer.from(input.bytes);
    const digest = createHash("sha256").update(bytes).digest("hex");
    return this.database.withTenant(scope, async (transaction) => {
      const result = await transaction.query<ArtifactRow>(
        `INSERT INTO public.teachlab_artifacts (
           tenant_id, owner_id, artifact_key, content_type, byte_length,
           sha256, metadata, content
         ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::bytea)
         ON CONFLICT (tenant_id, owner_id, artifact_key) DO UPDATE
           SET content_type = EXCLUDED.content_type,
               byte_length = EXCLUDED.byte_length,
               sha256 = EXCLUDED.sha256,
               metadata = EXCLUDED.metadata,
               content = EXCLUDED.content,
               updated_at = statement_timestamp(),
               version = public.teachlab_artifacts.version + 1
         RETURNING ${ARTIFACT_COLUMNS}`,
        [
          input.tenantId, input.ownerId, input.key, input.contentType,
          bytes.byteLength, digest, JSON.stringify(input.metadata ?? {}), bytes
        ]
      );
      const row = result.rows[0];
      if (!row) throw new Error("PostgreSQL did not return the stored artifact");
      const artifact = stored(row);
      assertIntegrity(row, artifact);
      return artifact;
    });
  }

  get(
    scope: AccessScope,
    key: string
  ): Promise<{artifact: StoredArtifact; bytes: Uint8Array} | undefined> {
    validateKey(key);
    return this.database.withTenant(scope, async (transaction) => {
      const result = await transaction.query<ArtifactRow>(
        `SELECT ${ARTIFACT_COLUMNS}
           FROM public.teachlab_artifacts
          WHERE tenant_id = $1 AND owner_id = $2 AND artifact_key = $3`,
        [scope.tenantId, scope.ownerId, key]
      );
      const row = result.rows[0];
      if (!row) return undefined;
      const artifact = stored(row);
      assertIntegrity(row, artifact);
      const bytes = Buffer.from(row.content);
      return {artifact, bytes: new Uint8Array(bytes)};
    });
  }

  list(scope: AccessScope): Promise<StoredArtifact[]> {
    return this.database.withTenant(scope, async (transaction) => {
      const result = await transaction.query<ArtifactRow>(
        `SELECT ${ARTIFACT_COLUMNS}
           FROM public.teachlab_artifacts
          WHERE tenant_id = $1 AND owner_id = $2
          ORDER BY artifact_key ASC`,
        [scope.tenantId, scope.ownerId]
      );
      return result.rows.map((row) => {
        const artifact = stored(row);
        assertIntegrity(row, artifact);
        return artifact;
      });
    });
  }

  delete(scope: AccessScope, key: string): Promise<boolean> {
    validateKey(key);
    return this.database.withTenant(scope, async (transaction) => {
      const result = await transaction.query(
        `DELETE FROM public.teachlab_artifacts
          WHERE tenant_id = $1 AND owner_id = $2 AND artifact_key = $3`,
        [scope.tenantId, scope.ownerId, key]
      );
      return result.rowCount === 1;
    });
  }

  deleteScope(scope: AccessScope): Promise<{count: number; bytes: number}> {
    return this.database.withTenant(scope, async (transaction) => {
      const result = await transaction.query<{byte_length: number | string}>(
        `DELETE FROM public.teachlab_artifacts
          WHERE tenant_id = $1 AND owner_id = $2
        RETURNING byte_length`,
        [scope.tenantId, scope.ownerId]
      );
      return {
        count: result.rowCount,
        bytes: result.rows.reduce((total, row) => {
          const size = byteLength(row.byte_length);
          const next = total + size;
          if (!Number.isSafeInteger(next)) {
            throw new Error("Invalid PostgreSQL artifact byte total");
          }
          return next;
        }, 0)
      };
    });
  }

  async probe(): Promise<"postgres_durable_force_rls"> {
    return "postgres_durable_force_rls";
  }
}
