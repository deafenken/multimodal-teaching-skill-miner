import {createHmac, timingSafeEqual} from "node:crypto";

import type {AccountDeletionStatusCookiePort} from "./account-data-rights.repository.port";
import type {AccountDeletionStatusCapability} from "./account-data-rights.types";

export const ACCOUNT_DELETION_STATUS_COOKIE = "teachlab_deletion_status";
export const SECURE_ACCOUNT_DELETION_STATUS_COOKIE =
  "__Host-teachlab_deletion_status";

const COOKIE_VERSION = "v1";
const OPERATION_PATTERN = /^adel_[0-9a-f]{32}$/;
const SCOPE_PATTERN = /^[0-9a-f]{64}$/;
const CAPABILITY_PATTERN = /^[A-Za-z0-9_-]{43}$/;
const SIGNATURE_PATTERN = /^[A-Za-z0-9_-]{43}$/;
const MAX_COOKIE_HEADER_BYTES = 24_000;

function secureEqual(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, "utf8");
  const rightBytes = Buffer.from(right, "utf8");
  return leftBytes.byteLength === rightBytes.byteLength
    && timingSafeEqual(leftBytes, rightBytes);
}

function singleCookie(header: string | undefined, name: string): string | undefined {
  if (!header || Buffer.byteLength(header, "utf8") > MAX_COOKIE_HEADER_BYTES) {
    return undefined;
  }
  const matches = header.split(";").flatMap((part) => {
    const separator = part.indexOf("=");
    if (separator <= 0 || part.slice(0, separator).trim() !== name) return [];
    const value = part.slice(separator + 1).trim();
    return value && !/[\r\n;]/.test(value) ? [value] : [];
  });
  return matches.length === 1 ? matches[0] : undefined;
}

export class AccountDeletionStatusCookieService
  implements AccountDeletionStatusCookiePort {
  readonly cookieName: string;
  private readonly key: Buffer;

  constructor(input: {secret: string | Buffer; secure: boolean}) {
    this.key = typeof input.secret === "string"
      ? Buffer.from(input.secret, "utf8")
      : Buffer.from(input.secret);
    if (this.key.byteLength < 32) {
      throw new Error("Account deletion status cookie key is too short");
    }
    this.cookieName = input.secure
      ? SECURE_ACCOUNT_DELETION_STATUS_COOKIE
      : ACCOUNT_DELETION_STATUS_COOKIE;
  }

  mint(input: {
    operationId: string;
    scopeSha256: string;
    capability: string;
    expiresAt: Date;
  }): string {
    const expires = Math.floor(input.expiresAt.getTime() / 1_000);
    if (
      !OPERATION_PATTERN.test(input.operationId)
      || !SCOPE_PATTERN.test(input.scopeSha256)
      || !CAPABILITY_PATTERN.test(input.capability)
      || !Number.isSafeInteger(expires)
    ) {
      throw new Error("Invalid account deletion status capability");
    }
    const unsigned = [
      COOKIE_VERSION,
      input.operationId,
      input.scopeSha256,
      input.capability,
      expires.toString(36)
    ].join(".");
    return `${unsigned}.${this.signature(unsigned)}`;
  }

  verify(
    rawCookieHeader: string | undefined,
    now = new Date()
  ): AccountDeletionStatusCapability | undefined {
    const value = singleCookie(rawCookieHeader, this.cookieName);
    if (!value || value.length > 320) return undefined;
    const [version, operationId, scopeSha256, capability, expiresRaw, signature, ...extra] =
      value.split(".");
    if (
      extra.length
      || version !== COOKIE_VERSION
      || !OPERATION_PATTERN.test(operationId ?? "")
      || !SCOPE_PATTERN.test(scopeSha256 ?? "")
      || !CAPABILITY_PATTERN.test(capability ?? "")
      || !/^[0-9a-z]{1,12}$/.test(expiresRaw ?? "")
      || !SIGNATURE_PATTERN.test(signature ?? "")
    ) return undefined;
    const expires = Number.parseInt(expiresRaw!, 36);
    const nowSeconds = Math.floor(now.getTime() / 1_000);
    const unsigned = [version, operationId, scopeSha256, capability, expiresRaw].join(".");
    if (
      !Number.isSafeInteger(expires)
      || !Number.isSafeInteger(nowSeconds)
      || expires <= nowSeconds
      || !secureEqual(this.signature(unsigned), signature!)
    ) return undefined;
    return {operationId: operationId!, scopeSha256: scopeSha256!, capability: capability!};
  }

  serialize(value: string, expiresAt: Date): string {
    if (!value || /[\r\n;]/.test(value) || !Number.isFinite(expiresAt.getTime())) {
      throw new Error("Invalid account deletion status cookie");
    }
    const secure = this.cookieName.startsWith("__Host-") ? "; Secure" : "";
    return `${this.cookieName}=${value}; Path=/; HttpOnly; SameSite=Strict; Expires=${expiresAt.toUTCString()}${secure}`;
  }

  clear(): string {
    const secure = this.cookieName.startsWith("__Host-") ? "; Secure" : "";
    return `${this.cookieName}=; Max-Age=0; Path=/; HttpOnly; SameSite=Strict${secure}`;
  }

  private signature(value: string): string {
    return createHmac("sha256", this.key).update(value, "utf8").digest("base64url");
  }
}
