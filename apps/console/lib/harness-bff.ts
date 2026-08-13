import {timingSafeEqual} from "node:crypto";

import {enforceLocalRequest} from "./local-request-security.ts";

export type HarnessBffMode = "local_python" | "authenticated_apps_api";

type HarnessBffConfiguration =
  | {
      mode: "local_python";
      upstreamBase: URL;
    }
  | {
      mode: "authenticated_apps_api";
      upstreamBase: URL;
      consoleOrigin: string;
      sessionCookieName: "teachlab_session" | "__Host-teachlab_session";
      csrfCookieName: "teachlab_csrf" | "__Host-teachlab_csrf";
      deletionStatusCookieName:
        | "teachlab_deletion_status"
        | "__Host-teachlab_deletion_status";
    };

export interface HarnessProxyContext {
  mode: HarnessBffMode;
  target: URL;
  headers: Headers;
}

export const HARNESS_BROWSER_CSRF_HEADER = "x-teachlab-csrf-token";
export const HARNESS_SECURITY_REJECTION_HEADER = "x-teachlab-security-rejection";

const APPS_API_CSRF_HEADER = "x-csrf-token";
const TOKEN_PATTERN = /^[A-Za-z0-9_-]{43}$/;
const MAX_COOKIE_VALUE_LENGTH = 8192;
const FORBIDDEN_IDENTITY_HEADERS = [
  "x-tenant",
  "x-tenant-id",
  "x-user",
  "x-user-id",
  "x-owner",
  "x-owner-id",
  "x-subject",
  "x-principal",
  "x-dev-user-id",
  "x-dev-tenant-id",
  "x-forwarded-user",
] as const;

const HARNESS_REQUEST_LIMITS = {
  default: 64 * 1024,
  attachment: 6 * 1024 * 1024,
  resource: 17 * 1024 * 1024,
  syllabus: 384 * 1024,
  project: 2 * 1024 * 1024,
  adjudication: 64 * 1024,
  consent: 16 * 1024,
  safeguarding: 16 * 1024,
} as const;

export function harnessProxyRequestBodyLimit(path: string): number {
  if (path === "api/attachment") return HARNESS_REQUEST_LIMITS.attachment;
  if (path === "api/resource/review") return HARNESS_REQUEST_LIMITS.default;
  if (path.startsWith("api/curriculum/")) return HARNESS_REQUEST_LIMITS.syllabus;
  if (path === "api/resource") return HARNESS_REQUEST_LIMITS.resource;
  if (path === "api/syllabi" || path.startsWith("api/syllabi/")) return HARNESS_REQUEST_LIMITS.syllabus;
  if (path === "api/projects" || path.startsWith("api/projects/")) return HARNESS_REQUEST_LIMITS.project;
  if (path.startsWith("api/adjudication/")) return HARNESS_REQUEST_LIMITS.adjudication;
  if (path.startsWith("api/consent/")) return HARNESS_REQUEST_LIMITS.consent;
  if (path.startsWith("api/safeguarding/")) return HARNESS_REQUEST_LIMITS.safeguarding;
  return HARNESS_REQUEST_LIMITS.default;
}

export class HarnessBffConfigurationError extends Error {
  readonly code: "mode_not_configured" | "upstream_not_configured";

  constructor(code: "mode_not_configured" | "upstream_not_configured") {
    super(code);
    this.code = code;
    this.name = "HarnessBffConfigurationError";
  }
}

function exactUrl(rawValue: string | undefined): URL {
  const raw = rawValue?.trim();
  if (!raw) throw new HarnessBffConfigurationError("upstream_not_configured");
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new HarnessBffConfigurationError("upstream_not_configured");
  }
  if (
    parsed.username
    || parsed.password
    || parsed.search
    || parsed.hash
    || parsed.pathname !== "/"
  ) {
    throw new HarnessBffConfigurationError("upstream_not_configured");
  }
  if (process.env.NODE_ENV === "production") {
    if (parsed.protocol !== "https:") {
      throw new HarnessBffConfigurationError("upstream_not_configured");
    }
  } else if (
    parsed.protocol !== "https:"
    && !(
      parsed.protocol === "http:"
      && new Set(["127.0.0.1", "::1", "localhost"]).has(parsed.hostname.toLowerCase())
    )
  ) {
    throw new HarnessBffConfigurationError("upstream_not_configured");
  }
  parsed.pathname = "/";
  return parsed;
}

function localCapabilityUrl(rawValue: string | undefined): URL {
  const raw = rawValue?.trim();
  if (!raw) throw new HarnessBffConfigurationError("upstream_not_configured");
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new HarnessBffConfigurationError("upstream_not_configured");
  }
  if (
    parsed.protocol !== "http:"
    || parsed.hostname !== "127.0.0.1"
    || !parsed.port
    || parsed.username
    || parsed.password
    || parsed.search
    || parsed.hash
  ) {
    throw new HarnessBffConfigurationError("upstream_not_configured");
  }
  return parsed;
}

function exactOrigin(rawValue: string | undefined): string {
  const raw = rawValue?.trim();
  if (!raw) throw new HarnessBffConfigurationError("upstream_not_configured");
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new HarnessBffConfigurationError("upstream_not_configured");
  }
  if (
    raw !== parsed.origin
    || parsed.username
    || parsed.password
    || parsed.pathname !== "/"
    || parsed.search
    || parsed.hash
    || (process.env.NODE_ENV === "production" && parsed.protocol !== "https:")
    || (parsed.protocol !== "http:" && parsed.protocol !== "https:")
  ) {
    throw new HarnessBffConfigurationError("upstream_not_configured");
  }
  return parsed.origin;
}

export function harnessBffConfiguration(): HarnessBffConfiguration {
  const rawMode = process.env.TEACHLAB_HARNESS_MODE?.trim();
  if (rawMode !== "local_python" && rawMode !== "authenticated_apps_api") {
    throw new HarnessBffConfigurationError("mode_not_configured");
  }
  if (rawMode === "local_python") {
    return {
      mode: rawMode,
      upstreamBase: localCapabilityUrl(process.env.TEACHER_AGENT_CAPABILITY_URL),
    };
  }

  const cookieMode = process.env.TEACHLAB_APPS_API_COOKIE_MODE?.trim()
    || (process.env.NODE_ENV === "production" ? "secure" : "development");
  if (
    (cookieMode !== "secure" && cookieMode !== "development")
    || (process.env.NODE_ENV === "production" && cookieMode !== "secure")
  ) {
    throw new HarnessBffConfigurationError("upstream_not_configured");
  }
  return {
    mode: rawMode,
    upstreamBase: exactUrl(process.env.TEACHLAB_APPS_API_URL),
    consoleOrigin: exactOrigin(process.env.TEACHLAB_CONSOLE_ORIGIN),
    sessionCookieName: cookieMode === "secure" ? "__Host-teachlab_session" : "teachlab_session",
    csrfCookieName: cookieMode === "secure" ? "__Host-teachlab_csrf" : "teachlab_csrf",
    deletionStatusCookieName: cookieMode === "secure"
      ? "__Host-teachlab_deletion_status"
      : "teachlab_deletion_status",
  };
}

/** Server-only readiness traffic must never recurse through the public edge. */
export function harnessReadinessTarget(configuration: HarnessBffConfiguration): URL {
  if (configuration.mode === "local_python") {
    return new URL(
      "api/bootstrap",
      configuration.upstreamBase.href.endsWith("/")
        ? configuration.upstreamBase
        : `${configuration.upstreamBase.href}/`,
    );
  }
  const configured = process.env.TEACHLAB_APPS_API_INTERNAL_URL?.trim();
  if (!configured) {
    if (process.env.NODE_ENV === "production") {
      throw new HarnessBffConfigurationError("upstream_not_configured");
    }
    return new URL("/ready", configuration.upstreamBase);
  }
  let internal: URL;
  try {
    internal = new URL(configured);
  } catch {
    throw new HarnessBffConfigurationError("upstream_not_configured");
  }
  const productionTarget = internal.protocol === "http:"
    && internal.hostname === "api"
    && internal.port === "4000";
  const developmentTarget = internal.protocol === "http:"
    && new Set(["127.0.0.1", "::1", "localhost", "api"]).has(internal.hostname);
  if (
    internal.username
    || internal.password
    || internal.pathname !== "/"
    || internal.search
    || internal.hash
    || (process.env.NODE_ENV === "production" ? !productionTarget : !developmentTarget)
  ) {
    throw new HarnessBffConfigurationError("upstream_not_configured");
  }
  return new URL("/ready", internal);
}

function rejection(status: 401 | 403 | 404 | 415 | 503, code: string, error: string): Response {
  return Response.json(
    {error, code},
    {
      status,
      headers: {
        "Cache-Control": "no-store, max-age=0",
        [HARNESS_SECURITY_REJECTION_HEADER]: code,
        "X-Content-Type-Options": "nosniff",
      },
    },
  );
}

export function harnessConfigurationRejection(error: unknown): Response {
  const code = error instanceof HarnessBffConfigurationError
    ? error.code
    : "upstream_not_configured";
  return rejection(
    503,
    code,
    code === "mode_not_configured"
      ? "Teaching Harness transport mode is not configured"
      : "Teaching Harness upstream is not configured",
  );
}

function singleCookie(cookieHeader: string | null, name: string): string | null {
  if (!cookieHeader || cookieHeader.length > 20_000) return null;
  const values: string[] = [];
  for (const part of cookieHeader.split(";")) {
    const separator = part.indexOf("=");
    if (separator <= 0 || part.slice(0, separator).trim() !== name) continue;
    const value = part.slice(separator + 1).trim();
    if (!value || value.length > MAX_COOKIE_VALUE_LENGTH || /[\r\n;]/.test(value)) return null;
    values.push(value);
  }
  return values.length === 1 ? values[0] : null;
}

function constantTimeEqual(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, "utf8");
  const rightBytes = Buffer.from(right, "utf8");
  return leftBytes.byteLength === rightBytes.byteLength && timingSafeEqual(leftBytes, rightBytes);
}

function authenticatedBrowserRejection(
  request: Request,
  config: Extract<HarnessBffConfiguration, {mode: "authenticated_apps_api"}>,
  options: {requireSession: boolean; requireCsrf: boolean},
): {response: Response} | {cookieHeader: string; csrfToken: string} {
  if (FORBIDDEN_IDENTITY_HEADERS.some((name) => request.headers.has(name))) {
    return {response: rejection(403, "identity_override_rejected", "Client identity overrides are not accepted")};
  }
  let requestUrl: URL;
  try {
    requestUrl = new URL(request.url);
  } catch {
    return {response: rejection(403, "invalid_host", "Console origin does not match its deployment configuration")};
  }
  const expected = new URL(config.consoleOrigin);
  if (
    requestUrl.origin !== config.consoleOrigin
    || request.headers.get("host") !== expected.host
    || request.headers.get("sec-fetch-site") !== "same-origin"
  ) {
    return {response: rejection(403, "invalid_origin", "Console origin does not match its deployment configuration")};
  }
  const unsafe = request.method !== "GET" && request.method !== "HEAD";
  if (unsafe) {
    if (request.headers.get("origin") !== config.consoleOrigin) {
      return {response: rejection(403, "invalid_origin", "Request origin does not match the Console")};
    }
    const contentType = request.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
    if (contentType !== "application/json") {
      return {response: rejection(415, "invalid_content_type", "Console mutations require application/json")};
    }
  }
  const sessionValue = singleCookie(request.headers.get("cookie"), config.sessionCookieName);
  const csrfValue = singleCookie(request.headers.get("cookie"), config.csrfCookieName);
  if (options.requireSession && (!sessionValue || !csrfValue || !TOKEN_PATTERN.test(csrfValue))) {
    return {response: rejection(401, "authentication_required", "TeachLab authentication is required")};
  }
  if (options.requireCsrf) {
    const supplied = request.headers.get(HARNESS_BROWSER_CSRF_HEADER) ?? "";
    if (!csrfValue || !TOKEN_PATTERN.test(supplied) || !constantTimeEqual(supplied, csrfValue)) {
      return {response: rejection(403, "invalid_csrf", "TeachLab request token is missing or invalid")};
    }
  }
  return {
    cookieHeader: sessionValue && csrfValue
      ? `${config.sessionCookieName}=${sessionValue}; ${config.csrfCookieName}=${csrfValue}`
      : "",
    csrfToken: csrfValue ?? "",
  };
}

export function prepareHarnessProxy(
  request: Request,
  path: string,
): HarnessProxyContext | {response: Response} {
  let config: HarnessBffConfiguration;
  try {
    config = harnessBffConfiguration();
  } catch (error) {
    return {response: harnessConfigurationRejection(error)};
  }
  const unsafe = request.method !== "GET" && request.method !== "HEAD";
  if (config.mode === "local_python") {
    const localRejection = enforceLocalRequest(request, {
      requireSession: true,
      requireCsrf: unsafe,
    });
    if (localRejection) return {response: localRejection};
    const headers = new Headers({"Cache-Control": "no-store"});
    const contentType = request.headers.get("content-type");
    if (contentType) headers.set("content-type", contentType);
    const accept = request.headers.get("accept");
    if (accept) headers.set("accept", accept);
    return {
      mode: config.mode,
      target: new URL(path, config.upstreamBase.href.endsWith("/") ? config.upstreamBase : new URL(`${config.upstreamBase.href}/`)),
      headers,
    };
  }

  const security = authenticatedBrowserRejection(request, config, {
    requireSession: true,
    requireCsrf: unsafe,
  });
  if ("response" in security) return security;
  const headers = new Headers({
    Accept: request.headers.get("accept") ?? "application/json",
    "Cache-Control": "no-store",
    Cookie: security.cookieHeader,
    Origin: config.consoleOrigin,
    "Sec-Fetch-Site": "same-origin",
  });
  if (unsafe) {
    headers.set("Content-Type", "application/json");
    headers.set(APPS_API_CSRF_HEADER, security.csrfToken);
  }
  return {
    mode: config.mode,
    target: new URL(`/api/v1/harness/${path}`, config.upstreamBase),
    headers,
  };
}

export function authenticatedSessionContext(
  request: Request,
  options: {requireSession: boolean; requireCsrf: boolean},
):
  | {config: Extract<HarnessBffConfiguration, {mode: "authenticated_apps_api"}>; cookieHeader: string; csrfToken: string}
  | {response: Response} {
  let config: HarnessBffConfiguration;
  try {
    config = harnessBffConfiguration();
  } catch (error) {
    return {response: harnessConfigurationRejection(error)};
  }
  if (config.mode !== "authenticated_apps_api") {
    return {response: rejection(503, "unsupported_auth_mode", "Authenticated apps/api session exchange is not enabled")};
  }
  const security = authenticatedBrowserRejection(request, config, options);
  return "response" in security ? security : {config, ...security};
}

/**
 * Account endpoints additionally carry only the HttpOnly deletion-status
 * capability. It is never accepted as identity and is never forwarded to the
 * Harness/Python gateway.
 */
export function authenticatedAccountContext(
  request: Request,
  options: {requireSession: boolean; requireCsrf: boolean},
):
  | {
      config: Extract<HarnessBffConfiguration, {mode: "authenticated_apps_api"}>;
      cookieHeader: string;
      csrfToken: string;
    }
  | {response: Response} {
  const context = authenticatedSessionContext(request, options);
  if ("response" in context) return context;
  const deletionStatus = singleCookie(
    request.headers.get("cookie"),
    context.config.deletionStatusCookieName,
  );
  const parts = [context.cookieHeader];
  if (deletionStatus) {
    parts.push(`${context.config.deletionStatusCookieName}=${deletionStatus}`);
  }
  return {...context, cookieHeader: parts.filter(Boolean).join("; ")};
}

export function appsApiSessionUrl(
  config: Extract<HarnessBffConfiguration, {mode: "authenticated_apps_api"}>,
): URL {
  return new URL("/api/v1/auth/session", config.upstreamBase);
}

export function appsApiOidcUrl(
  config: Extract<HarnessBffConfiguration, {mode: "authenticated_apps_api"}>,
  action: "begin" | "callback" | "account-step-up",
  search = "",
): URL {
  const target = new URL(`/api/v1/auth/oidc/${action}`, config.upstreamBase);
  target.search = search;
  return target;
}

export function authenticatedLoginContext(request: Request):
  | {config: Extract<HarnessBffConfiguration, {mode: "authenticated_apps_api"}>}
  | {response: Response} {
  if (process.env.TEACHLAB_HARNESS_MODE?.trim() === "local_python") {
    return {response: rejection(404, "login_unavailable", "Organization login is not enabled locally")};
  }
  let config: HarnessBffConfiguration;
  try {
    config = harnessBffConfiguration();
  } catch (error) {
    return {response: harnessConfigurationRejection(error)};
  }
  if (config.mode !== "authenticated_apps_api") {
    return {response: rejection(404, "login_unavailable", "Organization login is not enabled locally")};
  }
  let url: URL;
  try {
    url = new URL(request.url);
  } catch {
    return {response: rejection(403, "invalid_host", "Console origin does not match deployment configuration")};
  }
  const expected = new URL(config.consoleOrigin);
  if (url.origin !== config.consoleOrigin || request.headers.get("host") !== expected.host) {
    return {response: rejection(403, "invalid_origin", "Console origin does not match deployment configuration")};
  }
  return {config};
}

export function localSecurityMode(): "local_python" | "authenticated_apps_api" | "unconfigured" {
  const mode = process.env.TEACHLAB_HARNESS_MODE?.trim();
  return mode === "local_python" || mode === "authenticated_apps_api"
    ? mode
    : "unconfigured";
}

export const harnessBffTestApi = {
  singleCookie,
};
