import {createHmac} from "node:crypto";

import type {AuthenticatedPrincipal} from "../auth/auth-provider.port";

const SAFE_EPOCH = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;

function canonicalPart(value: string, name: string): Buffer {
  const bytes = Buffer.from(value, "utf8");
  if (!value || value !== value.trim() || bytes.byteLength > 4_096 || /[\u0000\r\n]/.test(value)) {
    throw new Error(`Invalid account cache ${name}`);
  }
  return Buffer.concat([
    Buffer.from(String(bytes.byteLength), "ascii"),
    Buffer.from(":"),
    bytes,
    Buffer.from("\0")
  ]);
}

/**
 * Browser-safe account cache namespace. It is stable for one canonical OIDC
 * identity and server-controlled epoch, but cannot reveal tenant or subject.
 */
export class AccountBrowserCacheScopeIssuer {
  private readonly key: Buffer;
  private readonly issuer: string;

  constructor(
    secret: string | Buffer,
    issuer: string,
    private readonly epoch: string
  ) {
    this.key = typeof secret === "string"
      ? Buffer.from(secret, "utf8")
      : Buffer.from(secret);
    if (this.key.byteLength < 32) throw new Error("Account cache scope key is too short");
    let parsed: URL;
    try {
      parsed = new URL(issuer);
    } catch {
      throw new Error("Account cache issuer is invalid");
    }
    if (
      parsed.protocol !== "https:"
      || parsed.username
      || parsed.password
      || parsed.search
      || parsed.hash
      || issuer !== parsed.toString().replace(/\/$/, "")
    ) throw new Error("Account cache issuer is invalid");
    if (!SAFE_EPOCH.test(epoch)) throw new Error("Account cache epoch is invalid");
    this.issuer = issuer;
  }

  issue(principal: AuthenticatedPrincipal): string {
    if (principal.provider !== "oidc") {
      throw new Error("Account cache scope requires canonical OIDC identity");
    }
    if (
      principal.identityNamespace !== "oidc-issuer-tenant-sub-v1"
      || principal.identityIssuer !== this.issuer
    ) {
      throw new Error("Account cache scope requires an issuer-bound OIDC identity");
    }
    const message = Buffer.concat([
      Buffer.from("teachlab-browser-account-cache-v1\0", "utf8"),
      canonicalPart(this.issuer, "issuer"),
      canonicalPart(principal.tenantId, "tenant"),
      canonicalPart(principal.subject, "subject"),
      canonicalPart(this.epoch, "epoch")
    ]);
    return `acs1_${createHmac("sha256", this.key).update(message).digest("base64url")}`;
  }
}
