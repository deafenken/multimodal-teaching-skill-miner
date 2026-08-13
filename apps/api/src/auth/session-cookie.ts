import {
  createCipheriv,
  createDecipheriv,
  createHash,
  hkdfSync,
  randomBytes,
  randomUUID,
  timingSafeEqual
} from "node:crypto";

import {Inject, Injectable} from "@nestjs/common";

import {AppConfigService} from "../config/app-config.service";
import {canonicalOidcAccessScope} from "./canonical-scope-identity";
import type {AuthenticatedPrincipal, AuthenticatedSession} from "./auth-provider.port";
import {
  isRemoteSubjectPolicy,
  type RemoteSubjectPolicy
} from "./remote-processing-policy";

interface SessionClaims {
  audience: "teachlab-api";
  csrfTokenHash: string;
  email?: string;
  expiresAt: number;
  issuedAt: number;
  issuer: "teachlab-api";
  provider: "development" | "oidc";
  roles: string[];
  sessionId: string;
  subject: string;
  tenantId: string;
  version: 2;
  identityNamespace?: "oidc-issuer-tenant-sub-v1";
  identityIssuer?: string;
  authenticatedAt?: number;
  assuranceLevel?: number;
  scopeTenantId?: string;
  scopeOwnerId?: string;
  remoteSubjectPolicy?: RemoteSubjectPolicy;
}

export interface MintSessionInput {
  subject: string;
  tenantId: string;
  provider: "development" | "oidc";
  email?: string;
  roles?: string[];
  identityNamespace?: "oidc-issuer-tenant-sub-v1";
  identityIssuer?: string;
  authenticatedAt?: string;
  assuranceLevel?: number;
  remoteSubjectPolicy?: RemoteSubjectPolicy;
}

export interface MintedSession {
  cookieValue: string;
  csrfToken: string;
  issuedAt: string;
  expiresAt: string;
  principal: AuthenticatedPrincipal;
  authenticatedSession: AuthenticatedSession;
}

const IDENTIFIER_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$/;
const ROLE_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;
const EMAIL_PATTERN = /^[^\s@\u0000-\u001f\u007f]{1,64}@[A-Za-z0-9.-]{1,189}$/;
const UUID_V4_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const COOKIE_PART_PATTERN = /^[A-Za-z0-9_-]+$/;
const SESSION_AEAD_INFO = Buffer.from(
  "teachlab.session-cookie.v3.aes-256-gcm",
  "utf8"
);

function sessionEncryptionKey(secret: string): Buffer {
  return Buffer.from(
    hkdfSync(
      "sha256",
      Buffer.from(secret, "utf8"),
      Buffer.from("teachlab.session-cookie.v3", "utf8"),
      SESSION_AEAD_INFO,
      32
    )
  );
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("base64url");
}

function secureEqual(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, "utf8");
  const rightBytes = Buffer.from(right, "utf8");
  return leftBytes.length === rightBytes.length && timingSafeEqual(leftBytes, rightBytes);
}

function parseCookieHeader(header: string | undefined): Map<string, string> {
  const cookies = new Map<string, string>();
  for (const part of header?.split(";") ?? []) {
    const separator = part.indexOf("=");
    if (separator <= 0) continue;
    const name = part.slice(0, separator).trim();
    const value = part.slice(separator + 1).trim();
    if (!name || cookies.has(name)) continue;
    try {
      cookies.set(name, decodeURIComponent(value));
    } catch {
      // A malformed cookie is ignored and cannot authenticate a request.
    }
  }
  return cookies;
}

function singleHeader(
  headers: Record<string, string | string[] | undefined>,
  name: string
): string | undefined {
  const value = headers[name];
  return Array.isArray(value) ? value[0] : value;
}

function validateIdentifier(name: string, value: unknown): string {
  if (typeof value !== "string" || !IDENTIFIER_PATTERN.test(value)) {
    throw new Error(`Invalid ${name} in authenticated identity`);
  }
  return value;
}

function validateClaims(value: unknown, nowSeconds: number): SessionClaims | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const claims = value as Partial<SessionClaims>;
  if (
    claims.version !== 2 ||
    claims.issuer !== "teachlab-api" ||
    claims.audience !== "teachlab-api" ||
    claims.provider !== "development" && claims.provider !== "oidc" ||
    typeof claims.sessionId !== "string" ||
    !UUID_V4_PATTERN.test(claims.sessionId) ||
    typeof claims.subject !== "string" ||
    typeof claims.tenantId !== "string" ||
    typeof claims.csrfTokenHash !== "string" ||
    typeof claims.issuedAt !== "number" ||
    typeof claims.expiresAt !== "number" ||
    !Array.isArray(claims.roles) ||
    claims.roles.length > 32 ||
    !claims.roles.every((role) => typeof role === "string" && ROLE_PATTERN.test(role)) ||
    (claims.email !== undefined && (typeof claims.email !== "string" || !EMAIL_PATTERN.test(claims.email))) ||
    claims.issuedAt > nowSeconds + 60 ||
    claims.expiresAt <= claims.issuedAt ||
    claims.expiresAt <= nowSeconds
  ) {
    return undefined;
  }
  if (claims.provider === "oidc") {
    let issuer: URL;
    try {
      issuer = new URL(claims.identityIssuer ?? "");
    } catch {
      return undefined;
    }
    if (
      claims.identityNamespace !== "oidc-issuer-tenant-sub-v1"
      || typeof claims.identityIssuer !== "string"
      || issuer.protocol !== "https:"
      || issuer.username
      || issuer.password
      || issuer.search
      || issuer.hash
      || claims.identityIssuer !== issuer.toString().replace(/\/$/, "")
      || typeof claims.authenticatedAt !== "number"
      || !Number.isInteger(claims.authenticatedAt)
      || claims.authenticatedAt > claims.issuedAt + 30
      || claims.authenticatedAt > nowSeconds + 30
      || typeof claims.assuranceLevel !== "number"
      || !Number.isInteger(claims.assuranceLevel)
      || claims.assuranceLevel < 1
      || claims.assuranceLevel > 3
      || typeof claims.scopeTenantId !== "string"
      || !/^ot1_[A-Za-z0-9._:-]{1,32}_[A-Za-z0-9_-]{43}$/.test(claims.scopeTenantId)
      || typeof claims.scopeOwnerId !== "string"
      || !/^os1_[A-Za-z0-9._:-]{1,32}_[A-Za-z0-9_-]{43}$/.test(claims.scopeOwnerId)
      || !isRemoteSubjectPolicy(claims.remoteSubjectPolicy)
    ) return undefined;
  } else if (
    claims.identityNamespace !== undefined
    || claims.identityIssuer !== undefined
    || claims.authenticatedAt !== undefined
    || claims.assuranceLevel !== undefined
    || claims.scopeTenantId !== undefined
    || claims.scopeOwnerId !== undefined
    || claims.remoteSubjectPolicy !== undefined
  ) return undefined;
  try {
    validateIdentifier("subject", claims.subject);
    validateIdentifier("tenantId", claims.tenantId);
  } catch {
    return undefined;
  }
  return claims as SessionClaims;
}

@Injectable()
export class SessionCookieService {
  constructor(@Inject(AppConfigService) private readonly config: AppConfigService) {}

  mint(input: MintSessionInput, now = new Date()): MintedSession {
    const subject = validateIdentifier("subject", input.subject);
    const tenantId = validateIdentifier("tenantId", input.tenantId);
    const roles = [...new Set(input.roles ?? [])]
      .filter((role) => ROLE_PATTERN.test(role))
      .slice(0, 32);
    if (input.email !== undefined && !EMAIL_PATTERN.test(input.email)) {
      throw new Error("Invalid email in authenticated identity");
    }
    let oidcIdentity: Pick<SessionClaims,
      "identityNamespace" | "identityIssuer" | "authenticatedAt" | "assuranceLevel"
      | "scopeTenantId" | "scopeOwnerId" | "remoteSubjectPolicy"
    > = {};
    if (input.provider === "oidc") {
      const authenticatedAt = Date.parse(input.authenticatedAt ?? "");
      let issuer: URL;
      try {
        issuer = new URL(input.identityIssuer ?? "");
      } catch {
        throw new Error("OIDC session requires a canonical identity issuer");
      }
      if (
        input.identityNamespace !== "oidc-issuer-tenant-sub-v1"
        || input.identityIssuer !== issuer.toString().replace(/\/$/, "")
        || issuer.protocol !== "https:"
        || issuer.username
        || issuer.password
        || issuer.search
        || issuer.hash
        || !Number.isFinite(authenticatedAt)
        || authenticatedAt > now.getTime() + 30_000
        || !Number.isInteger(input.assuranceLevel)
        || input.assuranceLevel! < 1
        || input.assuranceLevel! > 3
        || !isRemoteSubjectPolicy(input.remoteSubjectPolicy)
      ) throw new Error("OIDC session identity metadata is invalid");
      const scope = canonicalOidcAccessScope({
        issuer: input.identityIssuer,
        tenantId,
        subject,
        namespaceSecret: this.config.accountIdentityNamespaceSecret,
        namespaceVersion: this.config.accountIdentityNamespaceVersion
      });
      oidcIdentity = {
        identityNamespace: input.identityNamespace,
        identityIssuer: input.identityIssuer,
        authenticatedAt: Math.floor(authenticatedAt / 1000),
        assuranceLevel: input.assuranceLevel,
        remoteSubjectPolicy: structuredClone(input.remoteSubjectPolicy),
        ...scope
      };
    } else if (
      input.identityNamespace !== undefined
      || input.identityIssuer !== undefined
      || input.authenticatedAt !== undefined
      || input.assuranceLevel !== undefined
      || input.remoteSubjectPolicy !== undefined
    ) throw new Error("Development sessions cannot carry OIDC identity metadata");
    const sessionId = randomUUID();
    const csrfToken = randomBytes(32).toString("base64url");
    const issuedAt = Math.floor(now.getTime() / 1000);
    const expiresAt = issuedAt + this.config.sessionTtlSeconds;
    const claims: SessionClaims = {
      version: 2,
      issuer: "teachlab-api",
      audience: "teachlab-api",
      sessionId,
      subject,
      tenantId,
      provider: input.provider,
      roles,
      csrfTokenHash: sha256(csrfToken),
      issuedAt,
      expiresAt,
      ...oidcIdentity,
      ...(input.email ? {email: input.email} : {})
    };
    const cookieValue = this.encryptClaims(claims, this.config.sessionSecret);
    if (Buffer.byteLength(cookieValue, "utf8") > 4096) {
      throw new Error("Authenticated session exceeds the cookie limit");
    }
    const authenticatedSession = this.toAuthenticatedSession(claims);
    return {
      cookieValue,
      csrfToken,
      issuedAt: authenticatedSession.issuedAt,
      expiresAt: new Date(expiresAt * 1000).toISOString(),
      principal: authenticatedSession.principal,
      authenticatedSession
    };
  }

  authenticate(
    headers: Record<string, string | string[] | undefined>,
    now = new Date()
  ): AuthenticatedSession | null {
    const cookieHeader = singleHeader(headers, "cookie");
    const token = parseCookieHeader(cookieHeader).get(this.config.sessionCookieName);
    if (!token || Buffer.byteLength(token, "utf8") > 4096) return null;
    // v1/v2 exposed or lacked the issuer-bound identity namespace. They are
    // deliberately invalidated: only AEAD-sealed v3 tickets are accepted.
    const decoded = this.decryptClaims(token);
    if (decoded === undefined) return null;
    const claims = validateClaims(decoded, Math.floor(now.getTime() / 1000));
    if (!claims) return null;
    return this.toAuthenticatedSession(claims);
  }

  csrfMatches(
    session: AuthenticatedSession,
    headers: Record<string, string | string[] | undefined>
  ): boolean {
    const headerToken = singleHeader(headers, "x-csrf-token");
    const cookieToken = parseCookieHeader(singleHeader(headers, "cookie")).get(
      this.config.csrfCookieName
    );
    if (!headerToken || !cookieToken || !secureEqual(headerToken, cookieToken)) return false;
    return secureEqual(sha256(headerToken), session.csrfTokenHash);
  }

  serializeSessionCookie(value: string, maxAgeSeconds = this.config.sessionTtlSeconds): string {
    return this.serializeCookie(this.config.sessionCookieName, value, {
      httpOnly: true,
      maxAgeSeconds
    });
  }

  serializeCsrfCookie(value: string, maxAgeSeconds = this.config.sessionTtlSeconds): string {
    return this.serializeCookie(this.config.csrfCookieName, value, {
      httpOnly: false,
      maxAgeSeconds
    });
  }

  serializeOidcTransactionCookie(value: string, maxAgeSeconds: number): string {
    const attributes = [
      `${this.config.oidcTransactionCookieName}=${encodeURIComponent(value)}`,
      `Path=/api/v1/auth/oidc/callback`,
      "SameSite=Lax",
      `Max-Age=${maxAgeSeconds}`,
      "HttpOnly"
    ];
    if (this.config.secureSessionCookies) attributes.push("Secure");
    return attributes.join("; ");
  }

  private encryptClaims(claims: SessionClaims, secret: string): string {
    const nonce = randomBytes(12);
    const cipher = createCipheriv("aes-256-gcm", sessionEncryptionKey(secret), nonce);
    cipher.setAAD(SESSION_AEAD_INFO);
    const ciphertext = Buffer.concat([
      cipher.update(JSON.stringify(claims), "utf8"),
      cipher.final()
    ]);
    const tag = cipher.getAuthTag();
    return [
      "v3",
      nonce.toString("base64url"),
      ciphertext.toString("base64url"),
      tag.toString("base64url")
    ].join(".");
  }

  private decryptClaims(token: string): unknown | undefined {
    const parts = token.split(".");
    if (
      parts.length !== 4
      || parts[0] !== "v3"
      || parts.slice(1).some((part) => !COOKIE_PART_PATTERN.test(part))
    ) return undefined;
    let nonce: Buffer;
    let ciphertext: Buffer;
    let tag: Buffer;
    try {
      nonce = Buffer.from(parts[1]!, "base64url");
      ciphertext = Buffer.from(parts[2]!, "base64url");
      tag = Buffer.from(parts[3]!, "base64url");
    } catch {
      return undefined;
    }
    if (nonce.length !== 12 || tag.length !== 16 || ciphertext.length < 1) {
      return undefined;
    }
    for (const secret of this.config.sessionVerificationSecrets) {
      try {
        const decipher = createDecipheriv(
          "aes-256-gcm",
          sessionEncryptionKey(secret),
          nonce
        );
        decipher.setAAD(SESSION_AEAD_INFO);
        decipher.setAuthTag(tag);
        const plaintext = Buffer.concat([
          decipher.update(ciphertext),
          decipher.final()
        ]);
        return JSON.parse(plaintext.toString("utf8"));
      } catch {
        // Try the next configured rotation key. No error details or plaintext
        // are observable to the caller.
      }
    }
    return undefined;
  }

  private toPrincipal(claims: SessionClaims): AuthenticatedPrincipal {
    return {
      subject: claims.subject,
      tenantId: claims.tenantId,
      sessionId: claims.sessionId,
      provider: claims.provider,
      roles: [...claims.roles],
      ...(claims.identityNamespace
        ? {
            identityNamespace: claims.identityNamespace,
            identityIssuer: claims.identityIssuer,
            authenticatedAt: new Date(claims.authenticatedAt! * 1000).toISOString(),
            assuranceLevel: claims.assuranceLevel
            ,scopeTenantId: claims.scopeTenantId
            ,scopeOwnerId: claims.scopeOwnerId
            ,remoteSubjectPolicy: structuredClone(claims.remoteSubjectPolicy!)
          }
        : {}),
      ...(claims.email ? {email: claims.email} : {})
    };
  }

  private toAuthenticatedSession(claims: SessionClaims): AuthenticatedSession {
    return {
      principal: this.toPrincipal(claims),
      csrfTokenHash: claims.csrfTokenHash,
      issuedAt: new Date(claims.issuedAt * 1000).toISOString(),
      expiresAt: new Date(claims.expiresAt * 1000).toISOString()
    };
  }

  private serializeCookie(
    name: string,
    value: string,
    options: {httpOnly: boolean; maxAgeSeconds: number}
  ): string {
    const attributes = [
      `${name}=${encodeURIComponent(value)}`,
      "Path=/",
      "SameSite=Strict",
      `Max-Age=${options.maxAgeSeconds}`
    ];
    if (options.httpOnly) attributes.push("HttpOnly");
    if (this.config.secureSessionCookies) attributes.push("Secure");
    return attributes.join("; ");
  }
}
