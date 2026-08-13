import type {AccessScope} from "../tenancy/access-scope";

export type SessionRevocationReason = "user_logout";

export interface SessionAuthorityRecord {
  tenantId: string;
  ownerId: string;
  sessionIdSha256: string;
  issuedAt: Date;
  expiresAt: Date;
  revokedAt: Date | null;
  revocationReason: SessionRevocationReason | null;
  version: number;
}

export interface RegisterSessionAuthorityInput extends AccessScope {
  sessionIdSha256: string;
  issuedAt: Date;
  expiresAt: Date;
}

export interface RevokeSessionAuthorityInput extends AccessScope {
  sessionIdSha256: string;
  revokedAt: Date;
  reason: SessionRevocationReason;
}

export type SessionAuthorityState =
  | {kind: "active"; record: SessionAuthorityRecord}
  | {kind: "revoked"; record: SessionAuthorityRecord}
  | {kind: "missing"};

export type RevokeSessionAuthorityResult =
  | {kind: "revoked"; record: SessionAuthorityRecord}
  | {kind: "already_revoked"; record: SessionAuthorityRecord}
  | {kind: "missing"};

export type SessionAuthorityInspection =
  | {
      kind: "active" | "revoked";
      record: SessionAuthorityRecord;
      databaseNow: Date;
    }
  | {kind: "missing"; databaseNow: Date};

export interface SessionRevocationRepositoryPort {
  register(input: RegisterSessionAuthorityInput): Promise<SessionAuthorityRecord>;
  inspect(
    scope: AccessScope,
    sessionIdSha256: string,
    now: Date
  ): Promise<SessionAuthorityState>;
  inspectAuthoritatively(
    scope: AccessScope,
    sessionIdSha256: string
  ): Promise<SessionAuthorityInspection>;
  revoke(input: RevokeSessionAuthorityInput): Promise<RevokeSessionAuthorityResult>;
  cleanupExpired(scope: AccessScope, now: Date, limit: number): Promise<number>;
}
