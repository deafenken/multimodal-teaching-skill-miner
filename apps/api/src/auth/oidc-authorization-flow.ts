import {
  createCipheriv,
  createDecipheriv,
  createHash,
  randomBytes,
  timingSafeEqual
} from "node:crypto";

import type {VerifiedExternalIdentity} from "./external-identity-verifier.port";

export interface OidcAuthorizationFlowConfig {
  issuer: string;
  clientId: string;
  clientSecret: string;
  redirectUri: string;
  transactionSecret: string;
  discoveryTimeoutMs: number;
  transactionTtlSeconds: number;
  authenticationMaxAgeSeconds: number;
  accountStepUpMaxAgeSeconds: number;
  accountAal2AcrValues: readonly string[];
}

export type OidcAuthorizationPurpose = "login" | "account_data_rights";

export interface OidcIdTokenVerifier {
  verifyIdToken(
    token: string,
    expectedNonce: string,
    now?: Date,
    requirements?: {authenticationMaxAgeSeconds?: number; requiredAssuranceLevel?: 2}
  ): Promise<VerifiedExternalIdentity>;
}

interface OidcMetadata {
  issuer?: unknown;
  authorization_endpoint?: unknown;
  token_endpoint?: unknown;
  code_challenge_methods_supported?: unknown;
  token_endpoint_auth_methods_supported?: unknown;
}

interface LoginTransaction {
  version: 2;
  purpose: OidcAuthorizationPurpose;
  state: string;
  nonce: string;
  codeVerifier: string;
  issuedAt: number;
  expiresAt: number;
  returnTo: string;
}

interface TokenResponse {
  id_token?: unknown;
  token_type?: unknown;
}

const TRANSACTION_AAD = Buffer.from("teachlab-oidc-transaction-v2", "utf8");
const TRANSACTION_KEYS = new Set([
  "version",
  "purpose",
  "state",
  "nonce",
  "codeVerifier",
  "issuedAt",
  "expiresAt",
  "returnTo"
]);
const BASE64URL_43 = /^[A-Za-z0-9_-]{43}$/;
const CODE_VERIFIER = /^[A-Za-z0-9._~-]{43,128}$/;
const AUTHORIZATION_CODE = /^[\x21\x23-\x5B\x5D-\x7E]{8,4096}$/;
const MAX_DOCUMENT_BYTES = 1_048_576;
const MAX_TOKEN_RESPONSE_BYTES = 65_536;
const METADATA_CACHE_MS = 5 * 60 * 1000;

export class OidcAuthorizationFlowError extends Error {
  constructor(
    message: string,
    readonly safeCode:
      | "configuration_invalid"
      | "metadata_unavailable"
      | "transaction_invalid"
      | "callback_invalid"
      | "token_exchange_failed"
      | "identity_invalid"
  ) {
    super(message);
    this.name = "OidcAuthorizationFlowError";
  }
}

function exactHttpsUrl(name: string, value: string): URL {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new OidcAuthorizationFlowError(`${name} is invalid`, "configuration_invalid");
  }
  if (
    parsed.protocol !== "https:" ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash
  ) {
    throw new OidcAuthorizationFlowError(`${name} is unsafe`, "configuration_invalid");
  }
  return parsed;
}

function exactEndpoint(name: string, value: unknown): URL {
  if (typeof value !== "string") {
    throw new OidcAuthorizationFlowError(`${name} is missing`, "metadata_unavailable");
  }
  let endpoint: URL;
  try {
    endpoint = new URL(value);
  } catch {
    throw new OidcAuthorizationFlowError(`${name} is invalid`, "metadata_unavailable");
  }
  if (
    endpoint.protocol !== "https:"
    || endpoint.username
    || endpoint.password
    || endpoint.hash
  ) {
    throw new OidcAuthorizationFlowError(`${name} is unsafe`, "metadata_unavailable");
  }
  return endpoint;
}

function safeReturnTo(value: string | undefined): string {
  const candidate = value?.trim() || "/";
  if (
    !candidate.startsWith("/") ||
    candidate.startsWith("//") ||
    candidate.includes("\\") ||
    candidate.includes("#") ||
    candidate.startsWith("/api/") ||
    /%(?:2f|5c)/i.test(candidate) ||
    /[\u0000-\u001f\u007f]/.test(candidate) ||
    candidate.length > 512
  ) {
    throw new OidcAuthorizationFlowError(
      "OIDC return target is invalid",
      "callback_invalid"
    );
  }
  return candidate;
}

function secureEqual(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, "utf8");
  const rightBytes = Buffer.from(right, "utf8");
  return leftBytes.length === rightBytes.length && timingSafeEqual(leftBytes, rightBytes);
}

function formEncodeCredential(value: string): string {
  const encoded = new URLSearchParams({value}).toString();
  return encoded.slice("value=".length);
}

function encryptionKey(secret: string): Buffer {
  if (secret.length < 32) {
    throw new OidcAuthorizationFlowError(
      "OIDC transaction secret is too short",
      "configuration_invalid"
    );
  }
  return createHash("sha256")
    .update("teachlab-oidc-transaction-key-v1\0", "utf8")
    .update(secret, "utf8")
    .digest();
}

function validateTransaction(value: unknown, nowSeconds: number): LoginTransaction {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new OidcAuthorizationFlowError(
      "OIDC transaction is invalid",
      "transaction_invalid"
    );
  }
  const row = value as Partial<LoginTransaction> & Record<string, unknown>;
  if (
    new Set(Object.keys(row)).size !== TRANSACTION_KEYS.size ||
    Object.keys(row).some((key) => !TRANSACTION_KEYS.has(key)) ||
    row.version !== 2 ||
    !new Set(["login", "account_data_rights"]).has(row.purpose ?? "") ||
    typeof row.state !== "string" ||
    !BASE64URL_43.test(row.state) ||
    typeof row.nonce !== "string" ||
    !BASE64URL_43.test(row.nonce) ||
    typeof row.codeVerifier !== "string" ||
    !CODE_VERIFIER.test(row.codeVerifier) ||
    typeof row.issuedAt !== "number" ||
    !Number.isInteger(row.issuedAt) ||
    typeof row.expiresAt !== "number" ||
    !Number.isInteger(row.expiresAt) ||
    row.issuedAt > nowSeconds + 30 ||
    row.expiresAt <= nowSeconds ||
    row.expiresAt <= row.issuedAt ||
    row.expiresAt - row.issuedAt > 10 * 60 ||
    typeof row.returnTo !== "string" ||
    safeReturnTo(row.returnTo) !== row.returnTo
  ) {
    throw new OidcAuthorizationFlowError(
      "OIDC transaction is invalid or expired",
      "transaction_invalid"
    );
  }
  return row as LoginTransaction;
}

export class OidcTransactionCodec {
  private readonly key: Buffer;

  constructor(secret: string) {
    this.key = encryptionKey(secret);
  }

  seal(transaction: LoginTransaction): string {
    validateTransaction(transaction, transaction.issuedAt);
    const initializationVector = randomBytes(12);
    const cipher = createCipheriv("aes-256-gcm", this.key, initializationVector);
    cipher.setAAD(TRANSACTION_AAD);
    const plaintext = Buffer.from(JSON.stringify(transaction), "utf8");
    const ciphertext = Buffer.concat([cipher.update(plaintext), cipher.final()]);
    const tag = cipher.getAuthTag();
    return Buffer.concat([initializationVector, tag, ciphertext]).toString("base64url");
  }

  open(value: string, now = new Date()): LoginTransaction {
    if (!/^[A-Za-z0-9_-]{80,2048}$/.test(value)) {
      throw new OidcAuthorizationFlowError(
        "OIDC transaction cookie is invalid",
        "transaction_invalid"
      );
    }
    try {
      const sealed = Buffer.from(value, "base64url");
      if (sealed.toString("base64url") !== value || sealed.length < 29) throw new Error();
      const initializationVector = sealed.subarray(0, 12);
      const tag = sealed.subarray(12, 28);
      const ciphertext = sealed.subarray(28);
      const decipher = createDecipheriv("aes-256-gcm", this.key, initializationVector);
      decipher.setAAD(TRANSACTION_AAD);
      decipher.setAuthTag(tag);
      const plaintext = Buffer.concat([
        decipher.update(ciphertext),
        decipher.final()
      ]).toString("utf8");
      return validateTransaction(
        JSON.parse(plaintext) as unknown,
        Math.floor(now.getTime() / 1000)
      );
    } catch (error) {
      if (error instanceof OidcAuthorizationFlowError) throw error;
      throw new OidcAuthorizationFlowError(
        "OIDC transaction cookie could not be authenticated",
        "transaction_invalid"
      );
    }
  }
}

export class OidcAuthorizationFlow {
  private readonly issuer: URL;
  private readonly exactIssuer: string;
  private readonly redirectUri: URL;
  private readonly codec: OidcTransactionCodec;
  private metadata?: {
    authorizationEndpoint: URL;
    tokenEndpoint: URL;
    expiresAt: number;
  };

  constructor(
    private readonly config: OidcAuthorizationFlowConfig,
    private readonly verifier: OidcIdTokenVerifier,
    private readonly fetcher: typeof fetch = fetch,
    private readonly clock: () => Date = () => new Date()
  ) {
    this.issuer = exactHttpsUrl("OIDC issuer", config.issuer);
    this.exactIssuer = this.issuer.toString().replace(/\/$/, "");
    this.redirectUri = exactHttpsUrl("OIDC redirect URI", config.redirectUri);
    if (
      !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(config.clientId) ||
      !config.clientSecret ||
      config.clientSecret.length > 4096 ||
      /[\u0000\r\n]/.test(config.clientSecret) ||
      !Number.isInteger(config.discoveryTimeoutMs) ||
      config.discoveryTimeoutMs < 500 ||
      config.discoveryTimeoutMs > 30_000 ||
      !Number.isInteger(config.transactionTtlSeconds) ||
      config.transactionTtlSeconds < 60 ||
      config.transactionTtlSeconds > 10 * 60 ||
      !Number.isInteger(config.authenticationMaxAgeSeconds) ||
      config.authenticationMaxAgeSeconds < 60 ||
      config.authenticationMaxAgeSeconds > 24 * 60 * 60 ||
      !Number.isInteger(config.accountStepUpMaxAgeSeconds) ||
      config.accountStepUpMaxAgeSeconds < 60 ||
      config.accountStepUpMaxAgeSeconds > 10 * 60 ||
      !config.accountAal2AcrValues.length ||
      config.accountAal2AcrValues.some((value) =>
        typeof value !== "string"
        || value.length > 256
        || value !== value.trim()
        || /[\u0000-\u0020\u007f]/.test(value)
      )
    ) {
      throw new OidcAuthorizationFlowError(
        "OIDC authorization flow configuration is invalid",
        "configuration_invalid"
      );
    }
    this.codec = new OidcTransactionCodec(config.transactionSecret);
  }

  async begin(
    returnTo?: string,
    purpose: OidcAuthorizationPurpose = "login"
  ): Promise<{
    authorizationUrl: string;
    transactionCookie: string;
    expiresAt: string;
  }> {
    const metadata = await this.discover();
    const now = this.clock();
    const issuedAt = Math.floor(now.getTime() / 1000);
    const state = randomBytes(32).toString("base64url");
    const nonce = randomBytes(32).toString("base64url");
    const codeVerifier = randomBytes(48).toString("base64url");
    const transaction: LoginTransaction = {
      version: 2,
      purpose,
      state,
      nonce,
      codeVerifier,
      issuedAt,
      expiresAt: issuedAt + this.config.transactionTtlSeconds,
      returnTo: safeReturnTo(returnTo)
    };
    const challenge = createHash("sha256")
      .update(codeVerifier, "ascii")
      .digest("base64url");
    const authorizationUrl = new URL(metadata.authorizationEndpoint);
    authorizationUrl.searchParams.set("client_id", this.config.clientId);
    authorizationUrl.searchParams.set("redirect_uri", this.redirectUri.toString());
    authorizationUrl.searchParams.set("response_type", "code");
    authorizationUrl.searchParams.set("scope", "openid profile email");
    authorizationUrl.searchParams.set("state", state);
    authorizationUrl.searchParams.set("nonce", nonce);
    authorizationUrl.searchParams.set("code_challenge", challenge);
    authorizationUrl.searchParams.set("code_challenge_method", "S256");
    if (purpose === "account_data_rights") {
      authorizationUrl.searchParams.set("max_age", "0");
      authorizationUrl.searchParams.set("prompt", "login");
      authorizationUrl.searchParams.set(
        "acr_values",
        this.config.accountAal2AcrValues.join(" ")
      );
    } else {
      authorizationUrl.searchParams.set("max_age", String(this.config.authenticationMaxAgeSeconds));
    }
    return {
      authorizationUrl: authorizationUrl.toString(),
      transactionCookie: this.codec.seal(transaction),
      expiresAt: new Date(transaction.expiresAt * 1000).toISOString()
    };
  }

  async complete(input: {
    code: string;
    state: string;
    transactionCookie: string;
  }): Promise<{
    identity: VerifiedExternalIdentity;
    returnTo: string;
    purpose: OidcAuthorizationPurpose;
  }> {
    if (!AUTHORIZATION_CODE.test(input.code) || !BASE64URL_43.test(input.state)) {
      throw new OidcAuthorizationFlowError(
        "OIDC callback parameters are invalid",
        "callback_invalid"
      );
    }
    const transaction = this.codec.open(input.transactionCookie, this.clock());
    if (!secureEqual(transaction.state, input.state)) {
      throw new OidcAuthorizationFlowError(
        "OIDC callback state does not match",
        "callback_invalid"
      );
    }
    const metadata = await this.discover();
    const body = new URLSearchParams({
      grant_type: "authorization_code",
      code: input.code,
      redirect_uri: this.redirectUri.toString(),
      code_verifier: transaction.codeVerifier
    });
    const basic = Buffer.from(
      `${formEncodeCredential(this.config.clientId)}:${formEncodeCredential(this.config.clientSecret)}`,
      "utf8"
    ).toString("base64");
    const response = await this.boundedFetch(
      metadata.tokenEndpoint.toString(),
      {
        method: "POST",
        redirect: "error",
        headers: {
          accept: "application/json",
          authorization: `Basic ${basic}`,
          "content-type": "application/x-www-form-urlencoded"
        },
        body
      },
      MAX_TOKEN_RESPONSE_BYTES,
      "token_exchange_failed"
    );
    let token: TokenResponse;
    try {
      token = JSON.parse(response) as TokenResponse;
    } catch {
      throw new OidcAuthorizationFlowError(
        "OIDC token response is invalid",
        "token_exchange_failed"
      );
    }
    if (
      typeof token.id_token !== "string" ||
      token.id_token.length > 16_384 ||
      (token.token_type !== undefined &&
        (typeof token.token_type !== "string" || token.token_type.toLowerCase() !== "bearer"))
    ) {
      throw new OidcAuthorizationFlowError(
        "OIDC token response is incomplete",
        "token_exchange_failed"
      );
    }
    try {
      const identity = await this.verifier.verifyIdToken(
        token.id_token,
        transaction.nonce,
        this.clock(),
        transaction.purpose === "account_data_rights"
          ? {
              authenticationMaxAgeSeconds: this.config.accountStepUpMaxAgeSeconds,
              requiredAssuranceLevel: 2
            }
          : undefined
      );
      return {identity, returnTo: transaction.returnTo, purpose: transaction.purpose};
    } catch {
      throw new OidcAuthorizationFlowError(
        "OIDC identity token could not be verified",
        "identity_invalid"
      );
    }
  }

  private async discover(): Promise<{
    authorizationEndpoint: URL;
    tokenEndpoint: URL;
  }> {
    const now = this.clock().getTime();
    if (this.metadata && this.metadata.expiresAt > now) return this.metadata;
    const url = `${this.exactIssuer}/.well-known/openid-configuration`;
    const body = await this.boundedFetch(
      url,
      {redirect: "error", headers: {accept: "application/json"}},
      MAX_DOCUMENT_BYTES,
      "metadata_unavailable"
    );
    let metadata: OidcMetadata;
    try {
      metadata = JSON.parse(body) as OidcMetadata;
    } catch {
      throw new OidcAuthorizationFlowError(
        "OIDC discovery document is invalid",
        "metadata_unavailable"
      );
    }
    if (
      typeof metadata.issuer !== "string" ||
      metadata.issuer !== this.exactIssuer ||
      !Array.isArray(metadata.code_challenge_methods_supported) ||
      !metadata.code_challenge_methods_supported.includes("S256") ||
      !Array.isArray(metadata.token_endpoint_auth_methods_supported) ||
      !metadata.token_endpoint_auth_methods_supported.includes("client_secret_basic")
    ) {
      throw new OidcAuthorizationFlowError(
        "OIDC discovery capabilities are insufficient",
        "metadata_unavailable"
      );
    }
    const authorizationEndpoint = exactEndpoint(
      "OIDC authorization endpoint",
      metadata.authorization_endpoint
    );
    const tokenEndpoint = exactEndpoint("OIDC token endpoint", metadata.token_endpoint);
    this.metadata = {
      authorizationEndpoint,
      tokenEndpoint,
      expiresAt: now + METADATA_CACHE_MS
    };
    return this.metadata;
  }

  private async boundedFetch(
    url: string,
    init: RequestInit,
    maximumBytes: number,
    failureCode: "metadata_unavailable" | "token_exchange_failed"
  ): Promise<string> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.config.discoveryTimeoutMs);
    try {
      const response = await this.fetcher(url, {...init, signal: controller.signal});
      if (!response.ok) throw new Error("upstream rejected request");
      const contentLength = Number(response.headers.get("content-length") ?? "0");
      if (Number.isFinite(contentLength) && contentLength > maximumBytes) {
        throw new Error("upstream response is too large");
      }
      const body = await this.readBoundedBody(response, maximumBytes);
      if (!body) {
        throw new Error("upstream response is invalid");
      }
      return body;
    } catch {
      throw new OidcAuthorizationFlowError(
        "OIDC upstream request failed",
        failureCode
      );
    } finally {
      clearTimeout(timeout);
    }
  }

  private async readBoundedBody(response: Response, maximumBytes: number): Promise<string> {
    if (!response.body) throw new Error("upstream response body is missing");
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    try {
      while (true) {
        const next = await reader.read();
        if (next.done) break;
        total += next.value.byteLength;
        if (total > maximumBytes) {
          await reader.cancel("response too large").catch(() => undefined);
          throw new Error("upstream response is too large");
        }
        chunks.push(next.value);
      }
    } finally {
      reader.releaseLock();
    }
    return Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString("utf8");
  }
}
