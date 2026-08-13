import {createHash, createHmac, timingSafeEqual} from "node:crypto";

import type {
  AuthoritativeTeacherEntitlementSnapshot,
  CanonicalEntitlementIdentity,
  PrivilegedTeacherOperation,
  TeacherEntitlementAuthorizationReceipt,
  TeacherEntitlementAuthorizationResult,
  TeacherEntitlementClock,
  TeacherEntitlementDenialReason,
  TeacherEntitlementPolicy,
  TeacherEntitlementPrincipal,
  TeacherEntitlementSnapshotProvider
} from "./teacher-entitlement.port";
import {PRIVILEGED_TEACHER_OPERATIONS} from "./teacher-entitlement.port";

const SAFE_IDENTITY = /^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$/;
const SAFE_ROLE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;
const SAFE_POLICY_PART = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;
const RECEIPT_SCHEMA = "teachlab.teacher_entitlement_authorization_receipt.v1";

type Denied = Extract<TeacherEntitlementAuthorizationResult, {allowed: false}>;

interface ValidatedSnapshot {
  identity: CanonicalEntitlementIdentity;
  status: "active" | "revoked";
  roles: readonly string[];
  policyVersion: string;
  revision: number;
  evaluatedAtMs: number;
  expiresAtMs: number;
}

interface CachedSnapshot {
  snapshot: ValidatedSnapshot;
  freshUntilMs: number;
  cachedUntilMs: number;
  cachedAtMs: number;
}

interface InFlightSnapshot {
  generation: number;
  promise: Promise<SnapshotResult>;
}

interface SnapshotSuccess {
  ok: true;
  value: CachedSnapshot;
}

interface SnapshotFailure {
  ok: false;
  reason: TeacherEntitlementDenialReason;
}

type SnapshotResult = SnapshotSuccess | SnapshotFailure;
type SnapshotValidationResult =
  | {ok: true; value: ValidatedSnapshot}
  | {ok: false; reason: TeacherEntitlementDenialReason};

function denied(reason: TeacherEntitlementDenialReason): Denied {
  return {allowed: false, reason};
}

function canonicalJson(value: unknown): string {
  const normalize = (input: unknown): unknown => {
    if (
      input === null || typeof input === "string" || typeof input === "boolean"
    ) return input;
    if (typeof input === "number" && Number.isFinite(input)) return input;
    if (Array.isArray(input)) return input.map(normalize);
    if (input && typeof input === "object") {
      return Object.fromEntries(
        Object.entries(input as Record<string, unknown>)
          .filter(([, item]) => item !== undefined)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, item]) => [key, normalize(item)])
      );
    }
    throw new Error("entitlement receipt values must contain canonical JSON");
  };
  return JSON.stringify(normalize(value));
}

function frame(value: string): string {
  return `${Buffer.byteLength(value, "utf8")}:${value}\0`;
}

function exactEqual(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, "utf8");
  const rightBytes = Buffer.from(right, "utf8");
  return leftBytes.length === rightBytes.length && timingSafeEqual(leftBytes, rightBytes);
}

function isoMilliseconds(value: number): string {
  return new Date(value).toISOString();
}

function parseExactTimestamp(value: unknown): number | undefined {
  if (typeof value !== "string" || !value || value !== value.trim()) return undefined;
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed) || new Date(parsed).toISOString() !== value) return undefined;
  return parsed;
}

function validateIssuer(value: unknown): string | undefined {
  if (typeof value !== "string" || !value || value !== value.trim()) return undefined;
  try {
    const parsed = new URL(value);
    const canonical = parsed.toString().replace(/\/$/, "");
    if (
      parsed.protocol !== "https:" || parsed.username || parsed.password ||
      parsed.search || parsed.hash || canonical !== value
    ) return undefined;
    return value;
  } catch {
    return undefined;
  }
}

function validatedIdentity(value: unknown): CanonicalEntitlementIdentity | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const candidate = value as Partial<CanonicalEntitlementIdentity>;
  const issuer = validateIssuer(candidate.issuer);
  if (
    !issuer || typeof candidate.tenantId !== "string" ||
    !SAFE_IDENTITY.test(candidate.tenantId) || typeof candidate.subject !== "string" ||
    !SAFE_IDENTITY.test(candidate.subject)
  ) return undefined;
  return {issuer, tenantId: candidate.tenantId, subject: candidate.subject};
}

function validatePolicy(policy: TeacherEntitlementPolicy): void {
  if (
    !SAFE_POLICY_PART.test(policy.policyId) || !SAFE_POLICY_PART.test(policy.version) ||
    !Number.isInteger(policy.freshnessTtlMs) || policy.freshnessTtlMs < 1 ||
    policy.freshnessTtlMs > 5 * 60_000 ||
    !Number.isInteger(policy.cacheTtlMs) || policy.cacheTtlMs < 0 ||
    policy.cacheTtlMs > policy.freshnessTtlMs ||
    !Number.isInteger(policy.providerTimeoutMs) || policy.providerTimeoutMs < 1 ||
    policy.providerTimeoutMs > 30_000 ||
    !Number.isInteger(policy.maxClockSkewMs ?? 0) ||
    (policy.maxClockSkewMs ?? 0) < 0 || (policy.maxClockSkewMs ?? 0) > 30_000 ||
    !Number.isInteger(policy.maxCacheEntries ?? 10_000) ||
    (policy.maxCacheEntries ?? 10_000) < 1 || (policy.maxCacheEntries ?? 10_000) > 100_000 ||
    ![1, 2, 3].includes(policy.minAssuranceLevel ?? 1)
  ) throw new Error("Teacher entitlement policy configuration is invalid");

  for (const operation of PRIVILEGED_TEACHER_OPERATIONS) {
    const roles = policy.requiredRoles[operation];
    if (
      !Array.isArray(roles) || roles.length < 1 || roles.length > 32 ||
      roles.some((role) => typeof role !== "string" || !SAFE_ROLE.test(role))
    ) throw new Error(`Teacher entitlement roles are invalid for ${operation}`);
  }
}

function copyPolicy(policy: TeacherEntitlementPolicy): TeacherEntitlementPolicy {
  return Object.freeze({
    ...policy,
    requiredRoles: Object.freeze(Object.fromEntries(
      PRIVILEGED_TEACHER_OPERATIONS.map((operation) => [
        operation,
        Object.freeze([...new Set(policy.requiredRoles[operation])].sort())
      ])
    )) as unknown as TeacherEntitlementPolicy["requiredRoles"]
  });
}

export class TeacherEntitlementAuthorizationError extends Error {
  constructor(
    readonly reason: TeacherEntitlementDenialReason
  ) {
    super("Privileged teacher entitlement authorization was denied");
    this.name = "TeacherEntitlementAuthorizationError";
  }
}

/**
 * Fresh, server-authoritative authorization for the three teacher mutations.
 *
 * The signed session supplies only issuer + tenant + subject and assurance.
 * Its role claim is intentionally ignored. Settled snapshots are cached only
 * to the effective freshness boundary. An entitlement-change consumer should
 * call invalidateIdentity for immediate revocation; without that signal the
 * documented worst-case revocation lag is cacheTtlMs (and never beyond the
 * provider expiry or freshnessTtlMs).
 */
export class TeacherEntitlementAuthorizationService {
  private readonly policy: TeacherEntitlementPolicy;
  private readonly cache = new Map<string, CachedSnapshot>();
  private readonly inFlight = new Map<string, InFlightSnapshot>();
  private readonly highestRevision = new Map<string, number>();
  private readonly invalidationGeneration = new Map<string, number>();
  private lastClockMs = Number.NEGATIVE_INFINITY;

  constructor(
    private readonly provider: TeacherEntitlementSnapshotProvider,
    policy: TeacherEntitlementPolicy,
    private readonly bindingKey: Buffer,
    private readonly receiptKey: Buffer,
    private readonly clock: TeacherEntitlementClock = {now: () => new Date()}
  ) {
    validatePolicy(policy);
    if (bindingKey.byteLength < 32 || receiptKey.byteLength < 32) {
      throw new Error("Teacher entitlement keys must each contain at least 32 bytes");
    }
    this.policy = copyPolicy(policy);
  }

  async authorize(
    principal: TeacherEntitlementPrincipal,
    operation: PrivilegedTeacherOperation | string
  ): Promise<TeacherEntitlementAuthorizationResult> {
    if (!PRIVILEGED_TEACHER_OPERATIONS.includes(operation as PrivilegedTeacherOperation)) {
      return denied("unsupported_operation");
    }
    const identity = this.identityFromPrincipal(principal);
    if (!identity) return denied("invalid_principal");
    if ((principal.assuranceLevel ?? 0) < (this.policy.minAssuranceLevel ?? 1)) {
      return denied("invalid_principal");
    }
    const nowMs = this.readClock();
    if (nowMs === undefined) return denied("clock_invalid");
    const cacheKey = this.principalBinding(identity);
    const loaded = await this.snapshot(identity, cacheKey, nowMs);
    if (!loaded.ok) return denied(loaded.reason);
    const {snapshot, freshUntilMs} = loaded.value;
    if (nowMs >= freshUntilMs) return denied("snapshot_stale");
    if (snapshot.status === "revoked") return denied("entitlement_revoked");
    const requiredRoles = this.policy.requiredRoles[operation as PrivilegedTeacherOperation];
    if (!snapshot.roles.some((role) => requiredRoles.includes(role))) {
      return denied("role_missing");
    }
    const receipt = this.receipt(
      identity,
      snapshot,
      operation as PrivilegedTeacherOperation,
      nowMs,
      freshUntilMs
    );
    return {allowed: true, receipt};
  }

  async requireAuthorized(
    principal: TeacherEntitlementPrincipal,
    operation: PrivilegedTeacherOperation | string
  ): Promise<Readonly<TeacherEntitlementAuthorizationReceipt>> {
    const result = await this.authorize(principal, operation);
    if (!result.allowed) throw new TeacherEntitlementAuthorizationError(result.reason);
    return result.receipt;
  }

  /** Call from the authoritative role-change/revocation event consumer. */
  invalidateIdentity(identityOrPrincipal: CanonicalEntitlementIdentity | TeacherEntitlementPrincipal): void {
    const identity = "issuer" in identityOrPrincipal
      ? validatedIdentity(identityOrPrincipal)
      : this.identityFromPrincipal(identityOrPrincipal);
    if (!identity) return;
    const cacheKey = this.principalBinding(identity);
    this.cache.delete(cacheKey);
    this.invalidationGeneration.set(
      cacheKey,
      (this.invalidationGeneration.get(cacheKey) ?? 0) + 1
    );
  }

  /** Call when deploying a new role policy before accepting more requests. */
  invalidateAll(): void {
    this.cache.clear();
    for (const cacheKey of new Set([
      ...this.highestRevision.keys(),
      ...this.inFlight.keys()
    ])) {
      this.invalidationGeneration.set(
        cacheKey,
        (this.invalidationGeneration.get(cacheKey) ?? 0) + 1
      );
    }
  }

  private identityFromPrincipal(
    principal: TeacherEntitlementPrincipal
  ): CanonicalEntitlementIdentity | undefined {
    if (
      principal.provider !== "oidc" ||
      principal.identityNamespace !== "oidc-issuer-tenant-sub-v1"
    ) return undefined;
    return validatedIdentity({
      issuer: principal.identityIssuer,
      tenantId: principal.tenantId,
      subject: principal.subject
    });
  }

  private readClock(): number | undefined {
    const value = this.clock.now();
    const nowMs = value instanceof Date ? value.getTime() : Number.NaN;
    if (!Number.isFinite(nowMs) || nowMs < this.lastClockMs) {
      this.cache.clear();
      return undefined;
    }
    this.lastClockMs = nowMs;
    return nowMs;
  }

  private principalBinding(identity: CanonicalEntitlementIdentity): string {
    const digest = createHmac("sha256", this.bindingKey)
      .update(
        "teachlab-entitlement-principal-v1\0" +
        frame(identity.issuer) + frame(identity.tenantId) + frame(identity.subject),
        "utf8"
      )
      .digest("base64url");
    return `epb1_${digest}`;
  }

  private async snapshot(
    identity: CanonicalEntitlementIdentity,
    cacheKey: string,
    nowMs: number
  ): Promise<SnapshotResult> {
    const cached = this.cache.get(cacheKey);
    if (cached && nowMs < cached.cachedUntilMs && nowMs < cached.freshUntilMs) {
      return {ok: true, value: cached};
    }
    if (cached) this.cache.delete(cacheKey);

    const generation = this.invalidationGeneration.get(cacheKey) ?? 0;
    const pending = this.inFlight.get(cacheKey);
    if (pending?.generation === generation) return pending.promise;
    const request = this.loadSnapshot(identity, cacheKey, nowMs, generation).finally(() => {
      if (this.inFlight.get(cacheKey)?.promise === request) this.inFlight.delete(cacheKey);
    });
    this.inFlight.set(cacheKey, {generation, promise: request});
    return request;
  }

  private async loadSnapshot(
    identity: CanonicalEntitlementIdentity,
    cacheKey: string,
    requestedAtMs: number,
    generation: number
  ): Promise<SnapshotResult> {
    const abort = new AbortController();
    let timeout: NodeJS.Timeout | undefined;
    try {
      const timeoutFailure = new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(() => {
          abort.abort();
          reject(new Error("teacher entitlement provider timeout"));
        }, this.policy.providerTimeoutMs);
        timeout.unref?.();
      });
      const raw = await Promise.race([
        this.provider.readAuthoritativeSnapshot(Object.freeze({...identity}), abort.signal),
        timeoutFailure
      ]);
      if (!raw) return {ok: false, reason: "snapshot_missing"};
      const snapshot = this.validateSnapshot(raw, identity, requestedAtMs);
      if (!snapshot.ok) return snapshot;
      if ((this.invalidationGeneration.get(cacheKey) ?? 0) !== generation) {
        return {ok: false, reason: "snapshot_stale"};
      }
      const previousRevision = this.highestRevision.get(cacheKey);
      if (previousRevision !== undefined && snapshot.value.revision < previousRevision) {
        return {ok: false, reason: "snapshot_rollback"};
      }
      this.highestRevision.set(cacheKey, snapshot.value.revision);
      const freshUntilMs = Math.min(
        snapshot.value.expiresAtMs,
        snapshot.value.evaluatedAtMs + this.policy.freshnessTtlMs,
        requestedAtMs + this.policy.freshnessTtlMs
      );
      if (freshUntilMs <= requestedAtMs) {
        return {ok: false, reason: "snapshot_stale"};
      }
      const entry: CachedSnapshot = Object.freeze({
        snapshot: snapshot.value,
        freshUntilMs,
        cachedUntilMs: Math.min(freshUntilMs, requestedAtMs + this.policy.cacheTtlMs),
        cachedAtMs: requestedAtMs
      });
      if (this.policy.cacheTtlMs > 0) {
        this.pruneCache(requestedAtMs);
        this.cache.set(cacheKey, entry);
      }
      return {ok: true, value: entry};
    } catch {
      return {ok: false, reason: "provider_unavailable"};
    } finally {
      if (timeout) clearTimeout(timeout);
    }
  }

  private validateSnapshot(
    raw: AuthoritativeTeacherEntitlementSnapshot,
    expectedIdentity: CanonicalEntitlementIdentity,
    nowMs: number
  ): SnapshotValidationResult {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
      return {ok: false, reason: "snapshot_invalid"};
    }
    const identity = validatedIdentity(raw.identity);
    if (!identity) return {ok: false, reason: "snapshot_invalid"};
    if (
      !exactEqual(identity.issuer, expectedIdentity.issuer) ||
      !exactEqual(identity.tenantId, expectedIdentity.tenantId) ||
      !exactEqual(identity.subject, expectedIdentity.subject)
    ) return {ok: false, reason: "identity_mismatch"};
    if (!SAFE_POLICY_PART.test(raw.policyVersion) || raw.policyVersion !== this.policy.version) {
      return {ok: false, reason: "policy_mismatch"};
    }
    if (
      raw.status !== "active" && raw.status !== "revoked" ||
      !Number.isSafeInteger(raw.revision) || raw.revision < 0 ||
      !Array.isArray(raw.roles) || raw.roles.length > 32 ||
      raw.roles.some((role) => typeof role !== "string" || !SAFE_ROLE.test(role))
    ) return {ok: false, reason: "snapshot_invalid"};
    const roles = Object.freeze([...new Set(raw.roles)].sort());
    if (raw.status === "revoked" && roles.length > 0) {
      return {ok: false, reason: "snapshot_invalid"};
    }
    const evaluatedAtMs = parseExactTimestamp(raw.evaluatedAt);
    const expiresAtMs = parseExactTimestamp(raw.expiresAt);
    if (
      evaluatedAtMs === undefined || expiresAtMs === undefined ||
      expiresAtMs <= evaluatedAtMs ||
      evaluatedAtMs > nowMs + (this.policy.maxClockSkewMs ?? 0)
    ) return {ok: false, reason: "snapshot_invalid"};
    const value: ValidatedSnapshot = Object.freeze({
      identity: Object.freeze({...identity}),
      status: raw.status,
      roles,
      policyVersion: raw.policyVersion,
      revision: raw.revision,
      evaluatedAtMs,
      expiresAtMs
    });
    return {ok: true, value};
  }

  private pruneCache(nowMs: number): void {
    for (const [key, entry] of this.cache) {
      if (nowMs >= entry.cachedUntilMs || nowMs >= entry.freshUntilMs) this.cache.delete(key);
    }
    const maximum = this.policy.maxCacheEntries ?? 10_000;
    while (this.cache.size >= maximum) {
      let oldestKey: string | undefined;
      let oldestAt = Number.POSITIVE_INFINITY;
      for (const [key, entry] of this.cache) {
        if (entry.cachedAtMs < oldestAt) {
          oldestKey = key;
          oldestAt = entry.cachedAtMs;
        }
      }
      if (!oldestKey) break;
      this.cache.delete(oldestKey);
    }
  }

  private receipt(
    identity: CanonicalEntitlementIdentity,
    snapshot: ValidatedSnapshot,
    operation: PrivilegedTeacherOperation,
    nowMs: number,
    freshUntilMs: number
  ): Readonly<TeacherEntitlementAuthorizationReceipt> {
    const principalBinding = this.principalBinding(identity);
    const entitlementBinding = `esb1_${createHmac("sha256", this.bindingKey)
      .update(canonicalJson({
        domain: "teachlab-entitlement-snapshot-v1",
        principal_binding: principalBinding,
        status: snapshot.status,
        roles: snapshot.roles,
        policy_version: snapshot.policyVersion,
        revision: snapshot.revision,
        evaluated_at: isoMilliseconds(snapshot.evaluatedAtMs),
        expires_at: isoMilliseconds(snapshot.expiresAtMs)
      }), "utf8")
      .digest("base64url")}`;
    const material = {
      schema: RECEIPT_SCHEMA as typeof RECEIPT_SCHEMA,
      decision: "allow" as const,
      operation,
      policy_id: this.policy.policyId,
      policy_version: this.policy.version,
      entitlement_revision: snapshot.revision,
      principal_binding: principalBinding,
      entitlement_binding: entitlementBinding,
      authorized_at: isoMilliseconds(nowMs),
      fresh_until: isoMilliseconds(freshUntilMs)
    };
    const signature = createHmac("sha256", this.receiptKey)
      .update("teachlab-entitlement-receipt-v1\0", "utf8")
      .update(canonicalJson(material), "utf8")
      .digest("base64url");
    return Object.freeze({...material, signature});
  }
}

/** Stable digest helper for an adapter that needs to persist a receipt hash. */
export function teacherEntitlementReceiptSha256(
  receipt: Readonly<TeacherEntitlementAuthorizationReceipt>
): string {
  return createHash("sha256").update(canonicalJson(receipt), "utf8").digest("hex");
}
