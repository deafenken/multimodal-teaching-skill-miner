import {createHmac, randomBytes, timingSafeEqual} from "node:crypto";

export const LOCAL_SECURITY_COOKIE_NAME = "teachlab_local_session";
export const LOCAL_CSRF_HEADER_NAME = "x-teachlab-csrf-token";
export const LOCAL_SECURITY_REJECTION_HEADER = "x-teachlab-security-rejection";

const SESSION_VERSION = "v1";
const SESSION_TTL_SECONDS = 8 * 60 * 60;
const MAX_CLOCK_SKEW_SECONDS = 60;
const SESSION_ID_PATTERN = /^[A-Za-z0-9_-]{32}$/;
const SIGNATURE_PATTERN = /^[A-Za-z0-9_-]{43}$/;
const SECRET_SYMBOL = Symbol.for("teachlab.console.localSecuritySecret.v1");

interface SecurityGlobal {
  [SECRET_SYMBOL]?: Buffer;
}

interface LocalAuthority {
  canonical: string;
}

export interface LocalSession {
  raw: string;
  sessionId: string;
  expiresAtSeconds: number;
}

export interface LocalRequestSecurityOptions {
  requireSession?: boolean;
  requireCsrf?: boolean;
}

export interface LocalRequestSecuritySuccess {
  ok: true;
  session: LocalSession | null;
}

export interface LocalRequestSecurityFailure {
  ok: false;
  status: 401 | 403 | 415;
  code:
    | "invalid_host"
    | "invalid_fetch_metadata"
    | "invalid_origin"
    | "invalid_session"
    | "invalid_csrf"
    | "invalid_content_type";
  message: string;
}

export type LocalRequestSecurityResult = LocalRequestSecuritySuccess | LocalRequestSecurityFailure;

function securitySecret() {
  const configured = process.env.TEACHLAB_LOCAL_SECURITY_SECRET?.trim();
  if (configured) {
    if (!/^[0-9a-fA-F]{64}$/.test(configured)) {
      throw new Error("TEACHLAB_LOCAL_SECURITY_SECRET must contain exactly 64 hexadecimal characters");
    }
    return Buffer.from(configured, "hex");
  }
  const sharedGlobal = globalThis as typeof globalThis & SecurityGlobal;
  sharedGlobal[SECRET_SYMBOL] ??= randomBytes(32);
  return sharedGlobal[SECRET_SYMBOL];
}

function signature(value: string) {
  return createHmac("sha256", securitySecret()).update(value, "utf8").digest("base64url");
}

function constantTimeEqual(left: string, right: string) {
  const leftBytes = Buffer.from(left, "utf8");
  const rightBytes = Buffer.from(right, "utf8");
  return leftBytes.byteLength === rightBytes.byteLength && timingSafeEqual(leftBytes, rightBytes);
}

/**
 * Parse only the three authorities the local Console intentionally binds to.
 * URL parsers accept surprising spellings (integer IPv4, trailing dots, and
 * encoded hostnames), so the security boundary deliberately uses an exact
 * grammar instead.
 */
function localAuthority(rawValue: string | null): LocalAuthority | null {
  if (!rawValue || rawValue !== rawValue.trim() || rawValue.includes(",")) return null;
  const match = /^(localhost|127\.0\.0\.1|\[::1\])(?::([0-9]{1,5}))?$/i.exec(rawValue);
  if (!match) return null;
  const port = match[2] ?? "";
  if (port && (Number(port) < 1 || Number(port) > 65535 || String(Number(port)) !== port)) return null;
  return {canonical: `${match[1].toLowerCase()}${port ? `:${port}` : ""}`};
}

function requestProtocol(request: Request) {
  try {
    const value = new URL(request.url);
    if (value.protocol !== "http:" && value.protocol !== "https:") return null;
    return value.protocol;
  } catch {
    return null;
  }
}

function exactSameOrigin(request: Request, authority: LocalAuthority) {
  const rawOrigin = request.headers.get("origin");
  if (!rawOrigin || rawOrigin === "null" || rawOrigin.includes(",")) return false;
  try {
    const origin = new URL(rawOrigin);
    const originAuthority = localAuthority(origin.host);
    return rawOrigin === origin.origin
      && origin.username === ""
      && origin.password === ""
      && origin.pathname === "/"
      && origin.search === ""
      && origin.hash === ""
      && originAuthority?.canonical === authority.canonical
      && origin.protocol === requestProtocol(request);
  } catch {
    return false;
  }
}

function cookieValue(cookieHeader: string | null, name: string) {
  if (!cookieHeader) return null;
  const matches = cookieHeader.split(";").flatMap((part) => {
    const separator = part.indexOf("=");
    if (separator < 0 || part.slice(0, separator).trim() !== name) return [];
    return [part.slice(separator + 1).trim()];
  });
  return matches.length === 1 ? matches[0] : null;
}

function parseSession(raw: string | null, nowSeconds = Math.floor(Date.now() / 1000)): LocalSession | null {
  if (!raw || raw.length > 180) return null;
  const [version, sessionId, issuedAtRaw, suppliedSignature, ...extra] = raw.split(".");
  if (
    extra.length
    || version !== SESSION_VERSION
    || !SESSION_ID_PATTERN.test(sessionId ?? "")
    || !/^[0-9a-z]{1,12}$/.test(issuedAtRaw ?? "")
    || !SIGNATURE_PATTERN.test(suppliedSignature ?? "")
  ) return null;
  const issuedAt = Number.parseInt(issuedAtRaw, 36);
  if (
    !Number.isSafeInteger(issuedAt)
    || issuedAt > nowSeconds + MAX_CLOCK_SKEW_SECONDS
    || nowSeconds - issuedAt > SESSION_TTL_SECONDS
  ) return null;
  const unsigned = `${version}.${sessionId}.${issuedAtRaw}`;
  if (!constantTimeEqual(signature(unsigned), suppliedSignature)) return null;
  return {raw, sessionId, expiresAtSeconds: issuedAt + SESSION_TTL_SECONDS};
}

function failure(
  status: LocalRequestSecurityFailure["status"],
  code: LocalRequestSecurityFailure["code"],
  message: string,
): LocalRequestSecurityFailure {
  return {ok: false, status, code, message};
}

export function validateLocalRequest(
  request: Request,
  options: LocalRequestSecurityOptions = {},
): LocalRequestSecurityResult {
  const authority = localAuthority(request.headers.get("host"));
  if (!authority) return failure(403, "invalid_host", "Local Console host is not allowed");

  // Fetch Metadata is emitted by supported browsers and cannot be set by page
  // JavaScript. Requiring it closes cross-site form/navigation paths even when
  // an Origin header is missing on a safe-method fetch.
  if (request.headers.get("sec-fetch-site") !== "same-origin") {
    return failure(403, "invalid_fetch_metadata", "Cross-site requests are not allowed");
  }

  const unsafeMethod = request.method !== "GET" && request.method !== "HEAD";
  const hasOrigin = request.headers.has("origin");
  if ((unsafeMethod || hasOrigin) && !exactSameOrigin(request, authority)) {
    return failure(403, "invalid_origin", "Request origin does not match the local Console");
  }

  if (unsafeMethod) {
    const contentType = request.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
    if (contentType !== "application/json") {
      return failure(415, "invalid_content_type", "Local Console mutations require application/json");
    }
  }

  const rawSession = cookieValue(request.headers.get("cookie"), LOCAL_SECURITY_COOKIE_NAME);
  const session = parseSession(rawSession);
  if (options.requireSession && !session) {
    return failure(401, "invalid_session", "Local Console session is missing or expired");
  }

  if (options.requireCsrf) {
    const suppliedToken = request.headers.get(LOCAL_CSRF_HEADER_NAME) ?? "";
    const expectedToken = session ? signature(`csrf.${session.raw}`) : "";
    if (!session || !SIGNATURE_PATTERN.test(suppliedToken) || !constantTimeEqual(suppliedToken, expectedToken)) {
      return failure(403, "invalid_csrf", "Local Console request token is missing or invalid");
    }
  }
  return {ok: true, session};
}

export function localSecurityRejection(result: LocalRequestSecurityFailure) {
  return Response.json(
    {error: result.message, code: result.code},
    {
      status: result.status,
      headers: {
        "Cache-Control": "no-store, max-age=0",
        "Cross-Origin-Resource-Policy": "same-origin",
        [LOCAL_SECURITY_REJECTION_HEADER]: result.code,
        "X-Content-Type-Options": "nosniff",
      },
    },
  );
}

export function enforceLocalRequest(
  request: Request,
  options: LocalRequestSecurityOptions = {},
) {
  const result = validateLocalRequest(request, options);
  return result.ok ? null : localSecurityRejection(result);
}

export function issueLocalSecuritySession(request: Request, existingSession: LocalSession | null = null) {
  const nowSeconds = Math.floor(Date.now() / 1000);
  const issuedAt = nowSeconds.toString(36);
  const unsigned = `${SESSION_VERSION}.${randomBytes(24).toString("base64url")}.${issuedAt}`;
  const raw = existingSession?.raw ?? `${unsigned}.${signature(unsigned)}`;
  const csrfToken = signature(`csrf.${raw}`);
  const secure = requestProtocol(request) === "https:" ? "; Secure" : "";
  const expiresInSeconds = existingSession
    ? Math.max(1, existingSession.expiresAtSeconds - nowSeconds)
    : SESSION_TTL_SECONDS;
  const headers = new Headers({
    "Cache-Control": "no-store, max-age=0",
    "Cross-Origin-Resource-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
  });
  // A second tab must reuse the existing HttpOnly session instead of rotating
  // the shared cookie underneath the first tab's in-memory CSRF token.
  if (!existingSession) {
    headers.set(
      "Set-Cookie",
      `${LOCAL_SECURITY_COOKIE_NAME}=${raw}; Max-Age=${SESSION_TTL_SECONDS}; Path=/api/teacher-agent; HttpOnly; SameSite=Strict${secure}`,
    );
  }
  return Response.json(
    {connection: "local_python", csrf_token: csrfToken, expires_in_seconds: expiresInSeconds},
    {
      status: 200,
      headers,
    },
  );
}

export const localRequestSecurityTestApi = {
  localAuthority,
};
