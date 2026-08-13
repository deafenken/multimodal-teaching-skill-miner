import {createHash} from "node:crypto";

import {Inject, Injectable, ServiceUnavailableException} from "@nestjs/common";

import {SESSION_REVOCATION_REPOSITORY} from "../platform/tokens";
import type {AccessScope} from "../tenancy/access-scope";
import {accessScopeFor} from "../tenancy/access-scope";
import type {AuthenticatedPrincipal, AuthenticatedSession} from "./auth-provider.port";
import type {
  RevokeSessionAuthorityResult,
  SessionAuthorityRecord,
  SessionAuthorityState,
  SessionRevocationRepositoryPort
} from "./session-revocation.repository.port";

const SESSION_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function sessionIdSha256(sessionId: string): string {
  if (!SESSION_ID_PATTERN.test(sessionId)) throw new Error("Invalid authenticated session ID");
  return createHash("sha256").update(sessionId, "utf8").digest("hex");
}

@Injectable()
export class SessionRevocationService {
  constructor(
    @Inject(SESSION_REVOCATION_REPOSITORY)
    private readonly repository: SessionRevocationRepositoryPort
  ) {}

  async register(session: AuthenticatedSession): Promise<void> {
    const scope = this.scope(session);
    const digest = sessionIdSha256(session.principal.sessionId);
    const issuedAt = new Date(session.issuedAt);
    const expiresAt = new Date(session.expiresAt);
    try {
      const registered = await this.repository.register({
        ...scope,
        sessionIdSha256: digest,
        issuedAt,
        expiresAt
      });
      if (
        !this.matches(registered, scope, digest, issuedAt, expiresAt)
        || registered.revokedAt !== null
        || registered.revocationReason !== null
      ) throw new Error("Session authority registration was not exact");
    } catch {
      this.unavailable();
    }
  }

  async inspect(session: AuthenticatedSession): Promise<SessionAuthorityState> {
    const scope = this.scope(session);
    const digest = sessionIdSha256(session.principal.sessionId);
    const issuedAt = new Date(session.issuedAt);
    const expiresAt = new Date(session.expiresAt);
    try {
      const state = await this.repository.inspectAuthoritatively(scope, digest);
      if (
        !Number.isFinite(state.databaseNow.getTime())
        || expiresAt.getTime() <= state.databaseNow.getTime()
      ) return {kind: "missing"};
      if (
        state.kind !== "missing"
        && !this.matches(state.record, scope, digest, issuedAt, expiresAt)
      ) throw new Error("Session authority does not match signed claims");
      return state.kind === "missing"
        ? state
        : {kind: state.kind, record: state.record};
    } catch {
      return this.unavailable();
    }
  }

  async revokeCurrent(
    principal: AuthenticatedPrincipal,
    now = new Date()
  ): Promise<RevokeSessionAuthorityResult> {
    const digest = sessionIdSha256(principal.sessionId);
    const scope = accessScopeFor(principal);
    try {
      const result = await this.repository.revoke({
        ...scope,
        sessionIdSha256: digest,
        revokedAt: now,
        reason: "user_logout"
      });
      if (
        result.kind !== "missing"
        && (
          result.record.tenantId !== scope.tenantId
          || result.record.ownerId !== scope.ownerId
          || result.record.sessionIdSha256 !== digest
          || result.record.revokedAt === null
          || result.record.revocationReason !== "user_logout"
        )
      ) throw new Error("Session revocation result was not exact");
      return result;
    } catch {
      return this.unavailable();
    }
  }

  private scope(session: AuthenticatedSession): AccessScope {
    return accessScopeFor(session.principal);
  }

  private matches(
    record: SessionAuthorityRecord,
    scope: AccessScope,
    digest: string,
    issuedAt: Date,
    expiresAt: Date
  ): boolean {
    return record.tenantId === scope.tenantId
      && record.ownerId === scope.ownerId
      && record.sessionIdSha256 === digest
      && record.issuedAt.getTime() === issuedAt.getTime()
      && record.expiresAt.getTime() === expiresAt.getTime();
  }

  private unavailable(): never {
    throw new ServiceUnavailableException("Session revocation authority is unavailable");
  }
}
