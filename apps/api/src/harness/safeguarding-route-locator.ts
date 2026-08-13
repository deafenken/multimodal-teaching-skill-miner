import {
  createCipheriv,
  createDecipheriv,
  createHmac,
  randomBytes,
  timingSafeEqual
} from "node:crypto";

import type {HarnessScopeKey} from "../config/app-config.service";
import type {AccessScope} from "../tenancy/access-scope";

const LOCATOR_PATTERN = /^sgr1_(k[1-9][0-9]{0,8})_([A-Za-z0-9_-]{54,1800})$/;
const SAFE_SCOPE_PART = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,511}$/;
const ROUTE_SCHEMA = "teachlab.safeguarding_route_locator.v1";

function canonicalJson(value: Record<string, string>): Buffer {
  return Buffer.from(JSON.stringify(Object.fromEntries(
    Object.entries(value).sort(([left], [right]) => left.localeCompare(right))
  )), "utf8");
}

function validateScope(scope: AccessScope): void {
  if (
    !scope || !SAFE_SCOPE_PART.test(scope.tenantId) ||
    !SAFE_SCOPE_PART.test(scope.ownerId)
  ) throw new Error("Safeguarding route scope is invalid");
}

function routeKey(secret: string): Buffer {
  return createHmac("sha256", secret)
    .update("teachlab-safeguarding-route-locator-key-v1", "utf8")
    .digest();
}

function aad(version: string): Buffer {
  return Buffer.from(`teachlab-safeguarding-route-locator-v1\0${version}`, "utf8");
}

function exactEqual(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, "utf8");
  const rightBytes = Buffer.from(right, "utf8");
  return leftBytes.byteLength === rightBytes.byteLength &&
    timingSafeEqual(leftBytes, rightBytes);
}

/**
 * Encrypt an access scope for the central safeguarding routing system.
 *
 * The locator is opaque to the receiver and is never accepted without a
 * fresh safeguarding entitlement. It remains resolvable during key rotation
 * while the corresponding previous account-scope key is retained.
 */
export function issueSafeguardingRouteLocator(
  scope: AccessScope,
  activeKey: HarnessScopeKey
): string {
  validateScope(scope);
  if (!activeKey || !/^k[1-9][0-9]{0,8}$/.test(activeKey.version) ||
    Buffer.byteLength(activeKey.secret, "utf8") < 32) {
    throw new Error("Safeguarding route key is invalid");
  }
  const nonce = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", routeKey(activeKey.secret), nonce, {
    authTagLength: 16
  });
  cipher.setAAD(aad(activeKey.version));
  const plaintext = canonicalJson({
    schema: ROUTE_SCHEMA,
    tenant_id: scope.tenantId,
    owner_id: scope.ownerId
  });
  const encrypted = Buffer.concat([cipher.update(plaintext), cipher.final()]);
  const payload = Buffer.concat([nonce, encrypted, cipher.getAuthTag()]);
  return `sgr1_${activeKey.version}_${payload.toString("base64url")}`;
}

/** Resolve and tenant-bind a trusted route locator without exposing its owner. */
export function resolveSafeguardingRouteLocator(
  locator: string,
  keys: readonly HarnessScopeKey[],
  expectedTenantId: string
): AccessScope {
  if (
    typeof locator !== "string" || locator.length > 2_048 ||
    !SAFE_SCOPE_PART.test(expectedTenantId)
  ) throw new Error("Safeguarding route locator is invalid");
  const match = LOCATOR_PATTERN.exec(locator);
  const key = match
    ? keys.find((candidate) => candidate.version === match[1])
    : undefined;
  if (!match || !key || Buffer.byteLength(key.secret, "utf8") < 32) {
    throw new Error("Safeguarding route locator is invalid");
  }
  try {
    const payload = Buffer.from(match[2]!, "base64url");
    if (
      payload.toString("base64url") !== match[2]
      || payload.byteLength < 12 + 1 + 16
    ) {
      throw new Error("invalid locator length");
    }
    const nonce = payload.subarray(0, 12);
    const tag = payload.subarray(payload.byteLength - 16);
    const ciphertext = payload.subarray(12, payload.byteLength - 16);
    const decipher = createDecipheriv(
      "aes-256-gcm",
      routeKey(key.secret),
      nonce,
      {authTagLength: 16}
    );
    decipher.setAAD(aad(key.version));
    decipher.setAuthTag(tag);
    const plaintext = Buffer.concat([decipher.update(ciphertext), decipher.final()]);
    if (plaintext.byteLength > 2_048) throw new Error("invalid locator payload");
    const parsed: unknown = JSON.parse(plaintext.toString("utf8"));
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("invalid locator payload");
    }
    const value = parsed as Record<string, unknown>;
    if (
      Object.keys(value).sort().join(",") !== "owner_id,schema,tenant_id" ||
      value.schema !== ROUTE_SCHEMA || typeof value.tenant_id !== "string" ||
      typeof value.owner_id !== "string"
    ) throw new Error("invalid locator payload");
    const scope = {tenantId: value.tenant_id, ownerId: value.owner_id};
    validateScope(scope);
    if (!exactEqual(scope.tenantId, expectedTenantId)) {
      throw new Error("cross-tenant locator");
    }
    return scope;
  } catch {
    throw new Error("Safeguarding route locator is invalid");
  }
}
