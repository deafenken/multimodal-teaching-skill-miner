import {createHmac} from "node:crypto";

import type {AccessScope} from "../tenancy/access-scope";
import type {AccountScopeHashes} from "./account-data-rights.types";

const SHA256_PATTERN = /^[0-9a-f]{64}$/;

function boundedIdentityPart(value: string, name: string): Buffer {
  const encoded = Buffer.from(value, "utf8");
  if (!value || encoded.byteLength > 4_096 || /[\u0000\r\n]/.test(value)) {
    throw new Error(`Invalid account ${name}`);
  }
  return encoded;
}

/** Length-prefix the raw claims so concatenation cannot change scope meaning. */
export function canonicalAccountScope(scope: AccessScope): Buffer {
  const tenant = boundedIdentityPart(scope.tenantId, "tenant claim");
  const owner = boundedIdentityPart(scope.ownerId, "subject claim");
  return Buffer.concat([
    Buffer.from("teachlab-account-scope-v1\0", "utf8"),
    Buffer.from(String(tenant.byteLength), "ascii"),
    Buffer.from(":"),
    tenant,
    Buffer.from("\0"),
    Buffer.from(String(owner.byteLength), "ascii"),
    Buffer.from(":"),
    owner
  ]);
}

export function accountScopeSha256(scope: AccessScope, secret: string | Buffer): string {
  const key = typeof secret === "string" ? Buffer.from(secret, "utf8") : secret;
  if (key.byteLength < 32) throw new Error("Account scope hash key is too short");
  return createHmac("sha256", key).update(canonicalAccountScope(scope)).digest("hex");
}

export class AccountScopeHasher {
  private readonly keys: readonly (string | Buffer)[];

  constructor(activeSecret: string | Buffer, previousSecrets: readonly (string | Buffer)[] = []) {
    this.keys = [activeSecret, ...previousSecrets];
    // Validate every configured key at construction, before any request can be
    // accepted. The all-zero dummy scope is never stored.
    for (const key of this.keys) {
      accountScopeSha256({tenantId: "validation", ownerId: "validation"}, key);
    }
  }

  hashes(scope: AccessScope): AccountScopeHashes {
    const candidates = [...new Set(this.keys.map((key) => accountScopeSha256(scope, key)))];
    const active = candidates[0];
    if (!active || !SHA256_PATTERN.test(active)) {
      throw new Error("Account scope hash configuration is unavailable");
    }
    return Object.freeze({active, candidates: Object.freeze(candidates)});
  }
}

export function assertScopeHash(value: string): string {
  if (!SHA256_PATTERN.test(value)) throw new Error("Invalid opaque account scope hash");
  return value;
}
