import {Injectable} from "@nestjs/common";

import type {AccessScope} from "../tenancy/access-scope";
import type {
  RegisterSessionAuthorityInput,
  RevokeSessionAuthorityInput,
  RevokeSessionAuthorityResult,
  SessionAuthorityRecord,
  SessionAuthorityInspection,
  SessionAuthorityState,
  SessionRevocationRepositoryPort
} from "./session-revocation.repository.port";

const MAX_LOCAL_SESSION_RECORDS = 4_096;
const CLEANUP_BATCH = 256;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

function clone(record: SessionAuthorityRecord): SessionAuthorityRecord {
  return {
    ...record,
    issuedAt: new Date(record.issuedAt),
    expiresAt: new Date(record.expiresAt),
    revokedAt: record.revokedAt ? new Date(record.revokedAt) : null
  };
}

function key(scope: AccessScope, digest: string): string {
  if (!SHA256_PATTERN.test(digest)) throw new Error("Invalid session ID digest");
  return `${scope.tenantId}\u0000${scope.ownerId}\u0000${digest}`;
}

@Injectable()
export class InMemorySessionRevocationRepository
  implements SessionRevocationRepositoryPort {
  private readonly records = new Map<string, SessionAuthorityRecord>();

  async register(input: RegisterSessionAuthorityInput): Promise<SessionAuthorityRecord> {
    this.validateDates(input.issuedAt, input.expiresAt);
    const recordKey = key(input, input.sessionIdSha256);
    const existing = this.records.get(recordKey);
    if (existing) {
      if (
        existing.issuedAt.getTime() !== input.issuedAt.getTime() ||
        existing.expiresAt.getTime() !== input.expiresAt.getTime()
      ) {
        throw new Error("Session authority registration conflicts with an existing record");
      }
      return clone(existing);
    }
    const record: SessionAuthorityRecord = {
      tenantId: input.tenantId,
      ownerId: input.ownerId,
      sessionIdSha256: input.sessionIdSha256,
      issuedAt: new Date(input.issuedAt),
      expiresAt: new Date(input.expiresAt),
      revokedAt: null,
      revocationReason: null,
      version: 1
    };
    this.records.set(recordKey, record);
    this.enforceBound();
    return clone(record);
  }

  async inspect(
    scope: AccessScope,
    sessionIdSha256: string,
    now: Date
  ): Promise<SessionAuthorityState> {
    await this.cleanupExpired(scope, now, CLEANUP_BATCH);
    const record = this.records.get(key(scope, sessionIdSha256));
    if (!record || record.expiresAt.getTime() <= now.getTime()) return {kind: "missing"};
    return record.revokedAt
      ? {kind: "revoked", record: clone(record)}
      : {kind: "active", record: clone(record)};
  }

  async inspectAuthoritatively(
    scope: AccessScope,
    sessionIdSha256: string
  ): Promise<SessionAuthorityInspection> {
    const databaseNow = new Date();
    const state = await this.inspect(scope, sessionIdSha256, databaseNow);
    return state.kind === "missing"
      ? {kind: "missing", databaseNow}
      : {kind: state.kind, record: state.record, databaseNow};
  }

  async revoke(input: RevokeSessionAuthorityInput): Promise<RevokeSessionAuthorityResult> {
    await this.cleanupExpired(input, input.revokedAt, CLEANUP_BATCH);
    const recordKey = key(input, input.sessionIdSha256);
    const record = this.records.get(recordKey);
    if (!record || record.expiresAt.getTime() <= input.revokedAt.getTime()) {
      return {kind: "missing"};
    }
    if (record.revokedAt) return {kind: "already_revoked", record: clone(record)};
    const revoked: SessionAuthorityRecord = {
      ...record,
      revokedAt: new Date(input.revokedAt),
      revocationReason: input.reason,
      version: record.version + 1
    };
    this.records.set(recordKey, revoked);
    return {kind: "revoked", record: clone(revoked)};
  }

  async cleanupExpired(scope: AccessScope, now: Date, limit: number): Promise<number> {
    const maximum = Math.max(1, Math.min(CLEANUP_BATCH, Math.trunc(limit)));
    let removed = 0;
    for (const [recordKey, record] of this.records) {
      if (removed >= maximum) break;
      if (
        record.tenantId === scope.tenantId &&
        record.ownerId === scope.ownerId &&
        record.expiresAt.getTime() <= now.getTime()
      ) {
        this.records.delete(recordKey);
        removed += 1;
      }
    }
    return removed;
  }

  private enforceBound(): void {
    while (this.records.size > MAX_LOCAL_SESSION_RECORDS) {
      let candidate: [string, SessionAuthorityRecord] | undefined;
      for (const entry of this.records) {
        if (!candidate || entry[1].expiresAt < candidate[1].expiresAt) candidate = entry;
      }
      if (!candidate) break;
      // Capacity eviction invalidates a local session rather than forgetting a
      // revocation. Production cannot select this in-memory adapter.
      this.records.delete(candidate[0]);
    }
  }

  private validateDates(issuedAt: Date, expiresAt: Date): void {
    if (
      !Number.isFinite(issuedAt.getTime()) ||
      !Number.isFinite(expiresAt.getTime()) ||
      expiresAt.getTime() <= issuedAt.getTime()
    ) {
      throw new Error("Invalid session authority timestamps");
    }
  }
}
