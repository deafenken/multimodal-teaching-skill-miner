import type {RemoteSubjectPolicy} from "./remote-processing-policy";

export interface VerifiedExternalIdentity {
  subject: string;
  tenantId: string;
  identityNamespace: "oidc-issuer-tenant-sub-v1";
  identityIssuer: string;
  authenticatedAt: string;
  assuranceLevel: number;
  email?: string;
  roles: string[];
  /** Signed-IdP-derived and server-normalized; never project into browser JSON. */
  remoteSubjectPolicy: RemoteSubjectPolicy;
}

export interface ExternalIdentityVerifierPort {
  verify(token: string): Promise<VerifiedExternalIdentity>;
}
