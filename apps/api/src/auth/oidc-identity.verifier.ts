import {createPublicKey, createVerify, type JsonWebKey} from "node:crypto";

import {UnauthorizedException} from "@nestjs/common";

import {AppConfigService} from "../config/app-config.service";
import type {
  ExternalIdentityVerifierPort,
  VerifiedExternalIdentity
} from "./external-identity-verifier.port";
import {
  REMOTE_PROCESSING_POLICY_STATES,
  remoteSubjectPolicyFor,
  type RemoteProcessingPolicyState
} from "./remote-processing-policy";

interface OidcDiscoveryDocument {
  issuer?: unknown;
  jwks_uri?: unknown;
}

export interface OidcIdTokenVerificationContext {
  nonce: string;
  authenticationMaxAgeSeconds: number;
  requiredAssuranceLevel?: 2;
  now?: Date;
}

interface JsonWebKeySet {
  keys?: unknown;
}

interface JwtHeader {
  alg?: unknown;
  kid?: unknown;
  typ?: unknown;
}

type JwtClaims = Record<string, unknown>;

const MAX_DOCUMENT_BYTES = 1_048_576;
const MAX_TOKEN_BYTES = 16_384;
const CACHE_TTL_MS = 5 * 60 * 1000;
const SAFE_IDENTITY_CLAIM = /^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$/;
const SAFE_ROLE_CLAIM = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;
const SAFE_EMAIL_CLAIM = /^[^\s@\u0000-\u001f\u007f]{1,64}@[A-Za-z0-9.-]{1,189}$/;

function decodeJsonPart(part: string): unknown {
  return JSON.parse(Buffer.from(part, "base64url").toString("utf8"));
}

function isCanonicalBase64Url(part: string): boolean {
  return /^[A-Za-z0-9_-]+$/.test(part) &&
    Buffer.from(part, "base64url").toString("base64url") === part;
}

function stringClaim(claims: JwtClaims, name: string): string | undefined {
  const value = claims[name];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function exactStringClaim(claims: JwtClaims, name: string): string | undefined {
  const value = claims[name];
  return typeof value === "string" && value.length > 0 && value === value.trim()
    ? value
    : undefined;
}

function stringListClaim(claims: JwtClaims, name: string): string[] {
  const value = claims[name];
  if (Array.isArray(value)) {
    return [...new Set(value.filter((item): item is string => typeof item === "string"))];
  }
  return typeof value === "string"
    ? [...new Set(value.split(/[ ,]+/).map((item) => item.trim()).filter(Boolean))]
    : [];
}

function safeIdentityClaim(claims: JwtClaims, name: string): string | undefined {
  const value = stringClaim(claims, name);
  return value && SAFE_IDENTITY_CLAIM.test(value) ? value : undefined;
}

export class OidcIdentityVerifier implements ExternalIdentityVerifierPort {
  private keys: JsonWebKey[] = [];
  private keysExpireAt = 0;
  private discoveredJwksUri?: string;

  constructor(private readonly config: AppConfigService) {}

  async verify(token: string): Promise<VerifiedExternalIdentity> {
    return this.verifyToken(token);
  }

  async verifyIdToken(
    token: string,
    expectedNonce: string,
    now = new Date(),
    requirements: {authenticationMaxAgeSeconds?: number; requiredAssuranceLevel?: 2} = {}
  ): Promise<VerifiedExternalIdentity> {
    return this.verifyToken(token, {
      nonce: expectedNonce,
      authenticationMaxAgeSeconds:
        requirements.authenticationMaxAgeSeconds
        ?? this.config.oidcAuthenticationMaxAgeSeconds,
      ...(requirements.requiredAssuranceLevel
        ? {requiredAssuranceLevel: requirements.requiredAssuranceLevel}
        : {}),
      now
    });
  }

  private async verifyToken(
    token: string,
    context?: OidcIdTokenVerificationContext
  ): Promise<VerifiedExternalIdentity> {
    if (Buffer.byteLength(token, "utf8") > MAX_TOKEN_BYTES) {
      throw new UnauthorizedException("OIDC token is too large");
    }
    const parts = token.split(".");
    if (
      parts.length !== 3 ||
      !parts[0] ||
      !parts[1] ||
      !parts[2] ||
      !parts.every(isCanonicalBase64Url)
    ) {
      throw new UnauthorizedException("Malformed OIDC token");
    }
    let header: JwtHeader;
    let claims: JwtClaims;
    try {
      const decodedHeader = decodeJsonPart(parts[0]);
      const decodedClaims = decodeJsonPart(parts[1]);
      if (
        !decodedHeader || typeof decodedHeader !== "object" || Array.isArray(decodedHeader)
        || !decodedClaims || typeof decodedClaims !== "object" || Array.isArray(decodedClaims)
      ) throw new Error("JWT parts must be objects");
      header = decodedHeader as JwtHeader;
      claims = decodedClaims as JwtClaims;
    } catch {
      throw new UnauthorizedException("Malformed OIDC token");
    }
    if (
      typeof header.alg !== "string" ||
      !this.config.oidcAllowedAlgorithms.includes(header.alg) ||
      header.alg !== "RS256" ||
      typeof header.kid !== "string" ||
      !header.kid
    ) {
      throw new UnauthorizedException("OIDC token algorithm or key id is not allowed");
    }
    let key = await this.findKey(header.kid, false);
    if (!key) key = await this.findKey(header.kid, true);
    if (!key) throw new UnauthorizedException("OIDC signing key was not found");
    let signatureValid = false;
    try {
      const verifier = createVerify("RSA-SHA256");
      verifier.update(`${parts[0]}.${parts[1]}`, "utf8");
      verifier.end();
      signatureValid = verifier.verify(createPublicKey({key, format: "jwk"}), parts[2], "base64url");
    } catch {
      signatureValid = false;
    }
    if (!signatureValid) throw new UnauthorizedException("OIDC token signature is invalid");
    this.validateRegisteredClaims(claims, context);
    const subject = safeIdentityClaim(claims, "sub");
    const tenantId = safeIdentityClaim(claims, this.config.oidcTenantClaim);
    if (!subject || !tenantId) {
      throw new UnauthorizedException(
        `OIDC token must contain sub and ${this.config.oidcTenantClaim}`
      );
    }
    const emailCandidate = stringClaim(claims, "email");
    const email = emailCandidate && SAFE_EMAIL_CLAIM.test(emailCandidate)
      ? emailCandidate
      : undefined;
    const authenticatedAtSeconds = context
      ? claims.auth_time as number
      : typeof claims.auth_time === "number" && Number.isInteger(claims.auth_time)
        ? claims.auth_time
        : typeof claims.iat === "number" && Number.isInteger(claims.iat)
          ? claims.iat
          : Math.floor(Date.now() / 1000);
    const acr = exactStringClaim(claims, "acr");
    const assuranceLevel = acr && this.config.oidcAccountAal2AcrValues.includes(acr)
      ? 2
      : 1;
    if (context?.requiredAssuranceLevel && assuranceLevel < context.requiredAssuranceLevel) {
      throw new UnauthorizedException("OIDC authentication assurance is insufficient");
    }
    const assertedPolicy = exactStringClaim(
      claims,
      this.config.oidcRemoteProcessingPolicyClaim
    );
    const assertedPolicyVersion = exactStringClaim(
      claims,
      this.config.oidcRemoteProcessingPolicyVersionClaim
    );
    const remotePolicyState: RemoteProcessingPolicyState =
      assertedPolicyVersion === this.config.remoteSubjectPolicyVersion
      && REMOTE_PROCESSING_POLICY_STATES.includes(
        assertedPolicy as RemoteProcessingPolicyState
      )
        ? assertedPolicy as RemoteProcessingPolicyState
        : "denied";
    return {
      subject,
      tenantId,
      identityNamespace: "oidc-issuer-tenant-sub-v1",
      identityIssuer: this.config.oidcIssuer!,
      authenticatedAt: new Date(authenticatedAtSeconds * 1000).toISOString(),
      assuranceLevel,
      remoteSubjectPolicy: remoteSubjectPolicyFor(
        remotePolicyState,
        this.config.remoteSubjectPolicyId,
        this.config.remoteSubjectPolicyVersion
      ),
      roles: stringListClaim(claims, this.config.oidcRolesClaim)
        .filter((role) => SAFE_ROLE_CLAIM.test(role))
        .slice(0, 32),
      ...(email ? {email} : {})
    };
  }

  private validateRegisteredClaims(
    claims: JwtClaims,
    context?: OidcIdTokenVerificationContext
  ): void {
    const issuer = exactStringClaim(claims, "iss");
    const expectedIssuer = this.config.oidcIssuer;
    if (!issuer || issuer !== expectedIssuer) {
      throw new UnauthorizedException("OIDC issuer does not match");
    }
    const expectedAudience = context ? this.config.oidcClientId : this.config.oidcAudience;
    const audience = claims.aud;
    const audiences = typeof audience === "string"
      ? [audience]
      : Array.isArray(audience)
        && audience.length > 0
        && audience.every((value): value is string => typeof value === "string" && value.length > 0)
        ? [...new Set(audience)]
        : [];
    if (!expectedAudience || !audiences.length || !audiences.includes(expectedAudience)) {
      throw new UnauthorizedException("OIDC audience does not match");
    }
    if (
      audiences.length > 1 &&
      exactStringClaim(claims, "azp") !== expectedAudience
    ) {
      throw new UnauthorizedException("OIDC authorized party does not match");
    }
    const now = Math.floor((context?.now ?? new Date()).getTime() / 1000);
    if (
      typeof claims.exp !== "number"
      || !Number.isInteger(claims.exp)
      || claims.exp <= now - 30
    ) {
      throw new UnauthorizedException("OIDC token has expired");
    }
    if (
      claims.nbf !== undefined
      && (typeof claims.nbf !== "number" || !Number.isInteger(claims.nbf) || claims.nbf > now + 30)
    ) {
      throw new UnauthorizedException("OIDC token is not active yet");
    }
    if (context) {
      if (exactStringClaim(claims, "nonce") !== context.nonce) {
        throw new UnauthorizedException("OIDC nonce does not match");
      }
      if (
        typeof claims.iat !== "number"
        || !Number.isInteger(claims.iat)
        || claims.iat > now + 30
        || claims.iat < now - context.authenticationMaxAgeSeconds - 30
      ) {
        throw new UnauthorizedException("OIDC issued-at time is invalid");
      }
      if (
        typeof claims.auth_time !== "number"
        || !Number.isInteger(claims.auth_time)
        || claims.auth_time > claims.iat + 30
        || claims.auth_time > now + 30
        || claims.auth_time < now - context.authenticationMaxAgeSeconds - 30
      ) {
        throw new UnauthorizedException("OIDC authentication time is invalid");
      }
      const authorizedParty = exactStringClaim(claims, "azp");
      if (!authorizedParty || authorizedParty !== this.config.oidcClientId) {
        throw new UnauthorizedException("OIDC authorized party does not match client");
      }
    }
  }

  private async findKey(kid: string, forceRefresh: boolean): Promise<JsonWebKey | undefined> {
    if (forceRefresh || Date.now() >= this.keysExpireAt) await this.refreshKeys();
    return this.keys.find(
      (key) =>
        key.kid === kid &&
        key.kty === "RSA" &&
        (!key.use || key.use === "sig") &&
        (!key.alg || key.alg === "RS256")
    );
  }

  private async refreshKeys(): Promise<void> {
    const issuer = this.config.oidcIssuer;
    if (!issuer) throw new UnauthorizedException("OIDC issuer is not configured");
    if (!this.discoveredJwksUri) {
      const discovery = await this.fetchJson<OidcDiscoveryDocument>(
        `${issuer.replace(/\/$/, "")}/.well-known/openid-configuration`
      );
      if (
        typeof discovery.issuer !== "string" ||
        discovery.issuer !== issuer ||
        typeof discovery.jwks_uri !== "string"
      ) {
        throw new UnauthorizedException("OIDC discovery document is invalid");
      }
      const jwksUrl = new URL(discovery.jwks_uri);
      if (
        jwksUrl.protocol !== "https:"
        || jwksUrl.username
        || jwksUrl.password
        || jwksUrl.search
        || jwksUrl.hash
      ) {
        throw new UnauthorizedException("OIDC JWKS endpoint must use HTTPS");
      }
      this.discoveredJwksUri = jwksUrl.toString();
    }
    const jwks = await this.fetchJson<JsonWebKeySet>(this.discoveredJwksUri);
    if (!Array.isArray(jwks.keys)) {
      throw new UnauthorizedException("OIDC JWKS response is invalid");
    }
    this.keys = jwks.keys.filter(
      (key): key is JsonWebKey => Boolean(key && typeof key === "object" && !Array.isArray(key))
    );
    this.keysExpireAt = Date.now() + CACHE_TTL_MS;
  }

  private async fetchJson<T>(url: string): Promise<T> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.config.oidcDiscoveryTimeoutMs);
    try {
      const response = await fetch(url, {
        signal: controller.signal,
        redirect: "error",
        headers: {accept: "application/json"}
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const contentLength = Number(response.headers.get("content-length") ?? "0");
      if (Number.isFinite(contentLength) && contentLength > MAX_DOCUMENT_BYTES) {
        throw new Error("response too large");
      }
      if (!response.body) throw new Error("response body missing");
      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let total = 0;
      while (true) {
        const next = await reader.read();
        if (next.done) break;
        total += next.value.byteLength;
        if (total > MAX_DOCUMENT_BYTES) {
          await reader.cancel("response too large").catch(() => undefined);
          throw new Error("response too large");
        }
        chunks.push(next.value);
      }
      const body = Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString("utf8");
      return JSON.parse(body) as T;
    } catch {
      throw new UnauthorizedException("OIDC metadata could not be verified");
    } finally {
      clearTimeout(timeout);
    }
  }
}

export class DevelopmentOnlyIdentityVerifier implements ExternalIdentityVerifierPort {
  async verify(): Promise<never> {
    throw new UnauthorizedException("External identity exchange is disabled locally");
  }
}
