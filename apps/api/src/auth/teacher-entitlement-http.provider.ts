import type {
  AuthoritativeTeacherEntitlementSnapshot,
  CanonicalEntitlementIdentity,
  TeacherEntitlementClock,
  TeacherEntitlementSnapshotProvider
} from "./teacher-entitlement.port";

export const TEACHER_ENTITLEMENT_LOOKUP_SCHEMA =
  "teachlab.teacher_entitlement_lookup.v1";
export const TEACHER_ENTITLEMENT_SNAPSHOT_SCHEMA =
  "teachlab.teacher_entitlement_snapshot.v1";

const SAFE_IDENTITY = /^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$/;
const SAFE_ROLE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;
const SAFE_POLICY_VERSION = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;
const SAFE_BEARER = /^[\x21-\x7e]{32,4096}$/;

const RESPONSE_FIELDS = [
  "schema",
  "identity",
  "status",
  "roles",
  "policy_version",
  "revision",
  "evaluated_at",
  "expires_at"
] as const;
const IDENTITY_FIELDS = ["issuer", "tenant_id", "subject"] as const;

interface HttpTeacherEntitlementIdentity {
  issuer: string;
  tenant_id: string;
  subject: string;
}

interface HttpTeacherEntitlementSnapshot {
  schema: typeof TEACHER_ENTITLEMENT_SNAPSHOT_SCHEMA;
  identity: HttpTeacherEntitlementIdentity;
  status: "active" | "revoked";
  roles: string[];
  policy_version: string;
  revision: number;
  evaluated_at: string;
  expires_at: string;
}

export interface TeacherEntitlementHttpProviderOptions {
  endpoint: string;
  bearerSecret: string;
  timeoutMs: number;
  maxResponseBytes: number;
  maxClockSkewMs?: number;
  fetch?: typeof fetch;
  clock?: TeacherEntitlementClock;
}

function exactFields(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length && actual.every((key, index) => key === wanted[index]);
}

function canonicalIssuer(value: unknown): string | undefined {
  if (typeof value !== "string" || !value || value !== value.trim()) return undefined;
  try {
    const url = new URL(value);
    const canonical = url.toString().replace(/\/$/, "");
    if (
      url.protocol !== "https:" || url.username || url.password || url.search ||
      url.hash || canonical !== value
    ) return undefined;
    return value;
  } catch {
    return undefined;
  }
}

function canonicalEndpoint(value: string): string | undefined {
  if (!value || value !== value.trim()) return undefined;
  try {
    const url = new URL(value);
    if (
      url.protocol !== "https:" || url.username || url.password || url.search ||
      url.hash || url.toString() !== value
    ) return undefined;
    return value;
  } catch {
    return undefined;
  }
}

function validIdentity(value: CanonicalEntitlementIdentity): boolean {
  return Boolean(
    canonicalIssuer(value.issuer) && SAFE_IDENTITY.test(value.tenantId) &&
    SAFE_IDENTITY.test(value.subject)
  );
}

function exactIsoTimestamp(value: unknown): number | undefined {
  if (typeof value !== "string" || value !== value.trim()) return undefined;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) && new Date(parsed).toISOString() === value
    ? parsed
    : undefined;
}

function sanitizedFailure(): Error {
  return new Error("Teacher entitlement directory verification failed");
}

/**
 * Production adapter for a trusted organization entitlement directory.
 *
 * Raw issuer/tenant/subject and the Bearer credential exist only in the
 * private outbound request. Every network, protocol, size, parse, and schema
 * failure is collapsed to one public-safe error.
 */
export class TeacherEntitlementHttpProvider
implements TeacherEntitlementSnapshotProvider {
  private readonly endpoint: string;
  private readonly bearerSecret: string;
  private readonly timeoutMs: number;
  private readonly maxResponseBytes: number;
  private readonly maxClockSkewMs: number;
  private readonly fetchImplementation: typeof fetch;
  private readonly clock: TeacherEntitlementClock;

  constructor(options: TeacherEntitlementHttpProviderOptions) {
    const endpoint = canonicalEndpoint(options.endpoint);
    if (
      !endpoint || !SAFE_BEARER.test(options.bearerSecret) ||
      !Number.isInteger(options.timeoutMs) || options.timeoutMs < 1 ||
      options.timeoutMs > 30_000 ||
      !Number.isInteger(options.maxResponseBytes) || options.maxResponseBytes < 256 ||
      options.maxResponseBytes > 1_048_576 ||
      !Number.isInteger(options.maxClockSkewMs ?? 0) ||
      (options.maxClockSkewMs ?? 0) < 0 || (options.maxClockSkewMs ?? 0) > 30_000
    ) throw new Error("Teacher entitlement HTTP provider configuration is invalid");
    this.endpoint = endpoint;
    this.bearerSecret = options.bearerSecret;
    this.timeoutMs = options.timeoutMs;
    this.maxResponseBytes = options.maxResponseBytes;
    this.maxClockSkewMs = options.maxClockSkewMs ?? 0;
    this.fetchImplementation = options.fetch ?? globalThis.fetch;
    this.clock = options.clock ?? {now: () => new Date()};
    if (typeof this.fetchImplementation !== "function") {
      throw new Error("Teacher entitlement HTTP provider configuration is invalid");
    }
  }

  async readAuthoritativeSnapshot(
    identity: Readonly<CanonicalEntitlementIdentity>,
    signal?: AbortSignal
  ): Promise<AuthoritativeTeacherEntitlementSnapshot> {
    if (!validIdentity(identity)) throw sanitizedFailure();
    const now = this.clock.now();
    if (!(now instanceof Date) || !Number.isFinite(now.getTime())) throw sanitizedFailure();

    const controller = new AbortController();
    const abortFromCaller = () => controller.abort();
    if (signal?.aborted) controller.abort();
    else signal?.addEventListener("abort", abortFromCaller, {once: true});

    let timeout: NodeJS.Timeout | undefined;
    try {
      const timeoutFailure = new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(() => {
          controller.abort();
          reject(sanitizedFailure());
        }, this.timeoutMs);
      });
      const operation = this.fetchAndValidate(identity, now.getTime(), controller.signal);
      return await Promise.race([operation, timeoutFailure]);
    } catch {
      throw sanitizedFailure();
    } finally {
      if (timeout) clearTimeout(timeout);
      signal?.removeEventListener("abort", abortFromCaller);
    }
  }

  private async fetchAndValidate(
    identity: Readonly<CanonicalEntitlementIdentity>,
    nowMs: number,
    signal: AbortSignal
  ): Promise<AuthoritativeTeacherEntitlementSnapshot> {
    if (signal.aborted) throw sanitizedFailure();
    const requestBody = JSON.stringify({
      schema: TEACHER_ENTITLEMENT_LOOKUP_SCHEMA,
      identity: {
        issuer: identity.issuer,
        tenant_id: identity.tenantId,
        subject: identity.subject
      }
    });
    const response = await this.fetchImplementation(this.endpoint, {
      method: "POST",
      redirect: "error",
      signal,
      credentials: "omit",
      referrerPolicy: "no-referrer",
      headers: {
        accept: "application/json",
        authorization: `Bearer ${this.bearerSecret}`,
        "cache-control": "no-store",
        "content-type": "application/json"
      },
      body: requestBody
    });
    if (response.redirected || !response.ok || response.status < 200 || response.status >= 300) {
      throw sanitizedFailure();
    }
    const contentType = response.headers.get("content-type")?.toLowerCase();
    if (contentType !== "application/json" && contentType !== "application/json; charset=utf-8") {
      throw sanitizedFailure();
    }
    const declaredLength = response.headers.get("content-length");
    if (declaredLength !== null) {
      if (!/^(0|[1-9][0-9]*)$/.test(declaredLength)) throw sanitizedFailure();
      const bytes = Number(declaredLength);
      if (!Number.isSafeInteger(bytes) || bytes > this.maxResponseBytes) {
        throw sanitizedFailure();
      }
    }
    if (!response.body) throw sanitizedFailure();
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    while (true) {
      if (signal.aborted) {
        await reader.cancel().catch(() => undefined);
        throw sanitizedFailure();
      }
      const next = await reader.read();
      if (next.done) break;
      total += next.value.byteLength;
      if (total > this.maxResponseBytes) {
        await reader.cancel().catch(() => undefined);
        throw sanitizedFailure();
      }
      chunks.push(next.value);
    }
    if (total === 0) throw sanitizedFailure();
    let decoded: unknown;
    try {
      const bytes = Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)));
      const text = new TextDecoder("utf-8", {fatal: true}).decode(bytes);
      decoded = JSON.parse(text);
    } catch {
      throw sanitizedFailure();
    }
    return this.validateResponse(decoded, identity, nowMs);
  }

  private validateResponse(
    decoded: unknown,
    requestedIdentity: Readonly<CanonicalEntitlementIdentity>,
    nowMs: number
  ): AuthoritativeTeacherEntitlementSnapshot {
    if (!decoded || typeof decoded !== "object" || Array.isArray(decoded)) {
      throw sanitizedFailure();
    }
    const value = decoded as Record<string, unknown>;
    if (!exactFields(value, RESPONSE_FIELDS)) throw sanitizedFailure();
    if (!value.identity || typeof value.identity !== "object" || Array.isArray(value.identity)) {
      throw sanitizedFailure();
    }
    const identity = value.identity as Record<string, unknown>;
    if (!exactFields(identity, IDENTITY_FIELDS)) throw sanitizedFailure();
    if (
      value.schema !== TEACHER_ENTITLEMENT_SNAPSHOT_SCHEMA ||
      identity.issuer !== requestedIdentity.issuer ||
      identity.tenant_id !== requestedIdentity.tenantId ||
      identity.subject !== requestedIdentity.subject ||
      value.status !== "active" && value.status !== "revoked" ||
      !Array.isArray(value.roles) || value.roles.length > 32 ||
      value.roles.some((role) => typeof role !== "string" || !SAFE_ROLE.test(role)) ||
      typeof value.policy_version !== "string" ||
      !SAFE_POLICY_VERSION.test(value.policy_version) ||
      !Number.isSafeInteger(value.revision) || (value.revision as number) < 0
    ) throw sanitizedFailure();
    const roles = [...new Set(value.roles as string[])].sort();
    if (
      roles.length !== value.roles.length ||
      value.status === "revoked" && roles.length !== 0
    ) throw sanitizedFailure();
    const evaluatedAtMs = exactIsoTimestamp(value.evaluated_at);
    const expiresAtMs = exactIsoTimestamp(value.expires_at);
    if (
      evaluatedAtMs === undefined || expiresAtMs === undefined ||
      expiresAtMs <= evaluatedAtMs || evaluatedAtMs > nowMs + this.maxClockSkewMs
    ) throw sanitizedFailure();
    return Object.freeze({
      identity: Object.freeze({...requestedIdentity}),
      status: value.status,
      roles: Object.freeze(roles),
      policyVersion: value.policy_version,
      revision: value.revision as number,
      evaluatedAt: value.evaluated_at as string,
      expiresAt: value.expires_at as string
    });
  }
}
