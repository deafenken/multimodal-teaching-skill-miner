import type {AuthenticatedPrincipal} from "../auth/auth-provider.port";

export interface AccessScope {
  tenantId: string;
  ownerId: string;
}

export function accessScopeFor(principal: AuthenticatedPrincipal): AccessScope {
  if (principal.provider === "oidc") {
    if (!principal.scopeTenantId || !principal.scopeOwnerId) {
      throw new Error("OIDC session is missing its canonical scope namespace");
    }
    return {tenantId: principal.scopeTenantId, ownerId: principal.scopeOwnerId};
  }
  return {tenantId: principal.tenantId, ownerId: principal.subject};
}
