import type {AuthenticatedPrincipal} from "./auth-provider.port";

export const PRIVILEGED_TEACHER_OPERATIONS = [
  "api/resource/review",
  "api/curriculum/review",
  "api/curriculum/seal",
  "api/curriculum/revoke",
  "api/adjudication/claim",
  "api/adjudication/decide",
  "api/safeguarding/list",
  "api/safeguarding/dispatch",
  "api/safeguarding/case/acknowledge",
  "api/safeguarding/case/close",
  "api/safeguarding/escalation/overdue",
  "api/safeguarding/escalation/acknowledge"
] as const;

export type PrivilegedTeacherOperation =
  typeof PRIVILEGED_TEACHER_OPERATIONS[number];

/**
 * Raw identity is an internal provider lookup key only. It must never be
 * projected into an HTTP response, audit receipt, or worker request body.
 */
export interface CanonicalEntitlementIdentity {
  issuer: string;
  tenantId: string;
  subject: string;
}

/**
 * A point-in-time answer from the deployment's authoritative role source.
 * `revision` must increase whenever this identity's entitlement changes.
 */
export interface AuthoritativeTeacherEntitlementSnapshot {
  identity: CanonicalEntitlementIdentity;
  status: "active" | "revoked";
  roles: readonly string[];
  policyVersion: string;
  revision: number;
  evaluatedAt: string;
  expiresAt: string;
}

export interface TeacherEntitlementSnapshotProvider {
  readAuthoritativeSnapshot(
    identity: Readonly<CanonicalEntitlementIdentity>,
    signal?: AbortSignal
  ): Promise<AuthoritativeTeacherEntitlementSnapshot | null>;
}

export interface TeacherEntitlementClock {
  now(): Date;
}

export interface TeacherEntitlementPolicy {
  policyId: string;
  version: string;
  requiredRoles: Readonly<Record<PrivilegedTeacherOperation, readonly string[]>>;
  /** Upper bound on snapshot age, independent of provider-declared expiry. */
  freshnessTtlMs: number;
  /** Settled cache lifetime; it may not exceed freshnessTtlMs. */
  cacheTtlMs: number;
  providerTimeoutMs: number;
  maxClockSkewMs?: number;
  maxCacheEntries?: number;
  minAssuranceLevel?: 1 | 2 | 3;
}

export interface TeacherEntitlementAuthorizationReceipt {
  schema: "teachlab.teacher_entitlement_authorization_receipt.v1";
  decision: "allow";
  operation: PrivilegedTeacherOperation;
  policy_id: string;
  policy_version: string;
  entitlement_revision: number;
  principal_binding: string;
  entitlement_binding: string;
  authorized_at: string;
  fresh_until: string;
  signature: string;
}

export type TeacherEntitlementDenialReason =
  | "invalid_principal"
  | "unsupported_operation"
  | "provider_unavailable"
  | "snapshot_missing"
  | "snapshot_invalid"
  | "identity_mismatch"
  | "policy_mismatch"
  | "snapshot_stale"
  | "snapshot_rollback"
  | "entitlement_revoked"
  | "role_missing"
  | "clock_invalid";

export type TeacherEntitlementAuthorizationResult =
  | {
      allowed: true;
      receipt: Readonly<TeacherEntitlementAuthorizationReceipt>;
    }
  | {
      allowed: false;
      reason: TeacherEntitlementDenialReason;
    };

/**
 * Narrow adapter input accepted by the authorization core. Cookie roles are
 * present on AuthenticatedPrincipal but are deliberately never consulted.
 */
export type TeacherEntitlementPrincipal = Pick<
  AuthenticatedPrincipal,
  | "provider"
  | "identityNamespace"
  | "identityIssuer"
  | "tenantId"
  | "subject"
  | "assuranceLevel"
>;
