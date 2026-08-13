import type {RemoteSubjectPolicy} from "./remote-processing-policy";

export interface AuthenticatedPrincipal {
  subject: string;
  tenantId: string;
  sessionId: string;
  provider: "development" | "oidc";
  email?: string;
  roles: string[];
  /**
   * Internal, signed-session-only identity metadata. These fields are never
   * projected into browser JSON. Every OIDC session minted by this release
   * carries all four fields; legacy OIDC session namespaces fail closed.
   */
  identityNamespace?: "oidc-issuer-tenant-sub-v1";
  identityIssuer?: string;
  authenticatedAt?: string;
  assuranceLevel?: number;
  /** Opaque internal persistence namespace; never serialize to the browser. */
  scopeTenantId?: string;
  scopeOwnerId?: string;
  /** Internal signed-session policy. It must never be serialized to the browser. */
  remoteSubjectPolicy?: RemoteSubjectPolicy;
}

export interface AuthenticationRequest {
  headers: Record<string, string | string[] | undefined>;
  method: string;
}

export interface AuthenticatedSession {
  principal: AuthenticatedPrincipal;
  csrfTokenHash: string;
  issuedAt: string;
  expiresAt: string;
}

export interface AuthenticationOptions {
  allowRevokedSession?: boolean;
}

export interface AuthProviderPort {
  authenticate(
    request: AuthenticationRequest,
    options?: AuthenticationOptions
  ): Promise<AuthenticatedSession | null>;
}
