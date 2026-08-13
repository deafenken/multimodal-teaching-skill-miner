import {createHash, createHmac} from "node:crypto";

import type {AuthenticatedPrincipal} from "../auth/auth-provider.port";
import type {HarnessScopeKey} from "../config/app-config.service";
import type {AccountDeletionFreshAuthorityPort} from "./account-data-rights.repository.port";
import type {AccountDeletionFreshAuthority} from "./account-data-rights.types";

function hash(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function part(value: string): string {
  const bytes = Buffer.from(value, "utf8");
  if (!value || value !== value.trim() || bytes.byteLength > 4_096 || /[\u0000\r\n]/.test(value)) {
    throw new Error("Invalid canonical account authority identity");
  }
  return `${bytes.byteLength}:${value}\0`;
}

function keyed(secret: string, label: string, values: readonly string[]): string {
  return createHmac("sha256", secret)
    .update(`${label}\0${values.map(part).join("")}`, "utf8")
    .digest("hex");
}

/**
 * Revalidates a recent, provider-verified OIDC grant held only inside the
 * signed HttpOnly session. Returned bindings are hash-only and survive a
 * bounded account-key rotation window without persisting raw identity.
 */
export class SessionAccountFreshAuthority
  implements AccountDeletionFreshAuthorityPort {
  constructor(
    private readonly exactIssuer: string,
    private readonly keys: readonly HarnessScopeKey[],
    private readonly maxAgeMs: number
  ) {
    if (!exactIssuer || !keys.length || maxAgeMs < 60_000 || maxAgeMs > 600_000) {
      throw new Error("Account fresh authority configuration is invalid");
    }
  }

  async current(
    principal: AuthenticatedPrincipal,
    _now: Date
  ): Promise<AccountDeletionFreshAuthority | null> {
    if (
      principal.provider !== "oidc"
      || principal.identityNamespace !== "oidc-issuer-tenant-sub-v1"
      || principal.identityIssuer !== this.exactIssuer
      || !principal.authenticatedAt
      || !Number.isInteger(principal.assuranceLevel)
    ) return null;
    const authenticatedAt = new Date(principal.authenticatedAt);
    if (!Number.isFinite(authenticatedAt.getTime())) return null;
    const identity = [
      principal.identityNamespace,
      principal.identityIssuer,
      principal.tenantId,
      principal.subject
    ];
    return {
      bindings: this.keys.map((key) => ({
        keyVersion: key.version,
        canonicalIdentitySha256: keyed(
          key.secret,
          "teachlab-account-canonical-identity-v1",
          identity
        ),
        issuerSha256: keyed(
          key.secret,
          "teachlab-account-issuer-v1",
          [principal.identityIssuer!]
        ),
        authorityGrantSha256: keyed(
          key.secret,
          "teachlab-account-fresh-authority-v1",
          [
            ...identity,
            principal.sessionId,
            authenticatedAt.toISOString(),
            String(principal.assuranceLevel)
          ]
        )
      })),
      sessionSha256: hash(principal.sessionId),
      authenticatedAt,
      expiresAt: new Date(authenticatedAt.getTime() + this.maxAgeMs),
      assuranceLevel: principal.assuranceLevel!
    };
  }
}
