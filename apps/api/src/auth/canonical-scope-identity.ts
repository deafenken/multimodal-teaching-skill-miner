import {createHmac} from "node:crypto";

const SAFE_VERSION = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,31}$/;

function part(value: string, name: string): Buffer {
  const bytes = Buffer.from(value, "utf8");
  if (!value || value !== value.trim() || bytes.byteLength > 4_096 || /[\u0000\r\n]/.test(value)) {
    throw new Error(`Invalid canonical OIDC ${name}`);
  }
  return Buffer.concat([
    Buffer.from(String(bytes.byteLength), "ascii"),
    Buffer.from(":"),
    bytes,
    Buffer.from("\0")
  ]);
}

function digest(secret: string, label: string, values: readonly string[]): string {
  if (Buffer.byteLength(secret, "utf8") < 32) {
    throw new Error("Canonical scope identity secret is too short");
  }
  return createHmac("sha256", secret)
    .update(Buffer.concat([
      Buffer.from(`${label}\0`, "utf8"),
      ...values.map((value, index) => part(value, `part-${index}`))
    ]))
    .digest("base64url");
}

/**
 * Opaque, issuer-bound database/worker namespace. This secret is deliberately
 * separate from rotatable tombstone/root-discovery keys: rotating it is an
 * explicit account namespace migration, never a silent inheritance path.
 */
export function canonicalOidcAccessScope(input: {
  issuer: string;
  tenantId: string;
  subject: string;
  namespaceSecret: string;
  namespaceVersion: string;
}): {scopeTenantId: string; scopeOwnerId: string} {
  if (!SAFE_VERSION.test(input.namespaceVersion)) {
    throw new Error("Canonical scope namespace version is invalid");
  }
  const tenant = digest(
    input.namespaceSecret,
    "teachlab-oidc-scope-tenant-v1",
    [input.namespaceVersion, input.issuer, input.tenantId]
  );
  const owner = digest(
    input.namespaceSecret,
    "teachlab-oidc-scope-owner-v1",
    [input.namespaceVersion, input.issuer, input.tenantId, input.subject]
  );
  return {
    scopeTenantId: `ot1_${input.namespaceVersion}_${tenant}`,
    scopeOwnerId: `os1_${input.namespaceVersion}_${owner}`
  };
}
