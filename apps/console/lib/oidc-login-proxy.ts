import {appsApiOidcUrl, authenticatedLoginContext, authenticatedSessionContext} from "./harness-bff.ts";

const RETURN_TO_PATTERN = /^\/(?!\/|api\/)(?!.*[\\#\u0000-\u001f\u007f])(?!.*%(?:2f|5c))[^\s]{0,511}$/i;
const CALLBACK_QUERY_KEYS = new Set(["code", "state", "error", "error_description", "error_uri", "iss"]);
const MAX_COOKIE_BYTES = 4096;

interface ParsedSetCookie {
  name: string;
  value: string;
  attributes: Map<string, string | null>;
}

function parseSetCookie(rawValue: string): ParsedSetCookie | null {
  const raw = rawValue.trim();
  if (!raw || Buffer.byteLength(raw, "utf8") > MAX_COOKIE_BYTES || /[\r\n]/.test(raw)) {
    return null;
  }
  const parts = raw.split(";");
  const first = parts.shift()?.trim() ?? "";
  const separator = first.indexOf("=");
  if (separator <= 0) return null;
  const name = first.slice(0, separator).trim();
  const value = first.slice(separator + 1).trim();
  if (!/^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/.test(name) || /[\u0000-\u001f\u007f;,]/.test(value)) {
    return null;
  }
  const attributes = new Map<string, string | null>();
  for (const part of parts) {
    const attribute = part.trim();
    if (!attribute) return null;
    const equals = attribute.indexOf("=");
    const key = (equals < 0 ? attribute : attribute.slice(0, equals)).trim().toLowerCase();
    const attributeValue = equals < 0 ? null : attribute.slice(equals + 1).trim();
    if (
      !/^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/.test(key)
      || attributes.has(key)
      || (attributeValue !== null && /[\u0000-\u001f\u007f;]/.test(attributeValue))
    ) return null;
    attributes.set(key, attributeValue);
  }
  return {name, value, attributes};
}

function hasExactCookieAttributes(
  cookie: ParsedSetCookie,
  input: {path: string; sameSite: "Lax" | "Strict"; maxAge: string; httpOnly: boolean; secure: boolean},
): boolean {
  const expected = new Map<string, string | null>([
    ["path", input.path],
    ["samesite", input.sameSite],
    ["max-age", input.maxAge],
    ...(input.httpOnly ? [["httponly", null] as const] : []),
    ...(input.secure ? [["secure", null] as const] : []),
  ]);
  if (cookie.attributes.size !== expected.size) return false;
  for (const [key, value] of expected) {
    const actual = cookie.attributes.get(key);
    if (actual === undefined || (key === "samesite"
      ? actual?.toLowerCase() !== value?.toLowerCase()
      : actual !== value)) return false;
  }
  return true;
}

function safeFailure(status: 400 | 403 | 503, code: string, clearCookie?: string): Response {
  const headers = new Headers({"Cache-Control": "no-store, max-age=0", "X-Content-Type-Options": "nosniff"});
  if (clearCookie) headers.append("Set-Cookie", clearCookie);
  return Response.json({error: "TeachLab organization login could not be completed", code}, {
    status,
    headers,
  });
}

function clearBrowserTransactionCookie(secure: boolean): string {
  return `${secure ? "__Secure-" : ""}teachlab_oidc_tx=; Path=/api/teacher-agent/security/login/callback; SameSite=Lax; Max-Age=0; HttpOnly${secure ? "; Secure" : ""}`;
}

function accountTransitionCookie(secure: boolean, maxAge: string): string {
  return `${secure ? "__Host-" : ""}teachlab_account_transition=pending_v1; Path=/; SameSite=Strict; Max-Age=${maxAge}${secure ? "; Secure" : ""}`;
}

function callbackSessionMaxAge(cookies: readonly string[], secure: boolean): string | null {
  const sessionName = `${secure ? "__Host-" : ""}teachlab_session`;
  for (const raw of cookies) {
    const parsed = parseSetCookie(raw);
    if (parsed?.name === sessionName) return parsed.attributes.get("max-age") ?? null;
  }
  return null;
}

function rewriteTransactionCookiePath(raw: string): string {
  return raw.replace(
    /;\s*Path=\/api\/v1\/auth\/oidc\/callback(?:;|$)/i,
    "; Path=/api/teacher-agent/security/login/callback;",
  ).replace(/;;+/g, ";");
}

function validatedTransactionCookies(response: Response, secure: boolean): string[] | null {
  const extended = response.headers as Headers & {getSetCookie?: () => string[]};
  const values = typeof extended.getSetCookie === "function"
    ? extended.getSetCookie()
    : response.headers.get("set-cookie")?.split(/,(?=\s*(?:__Secure-)?teachlab_oidc_tx=)/) ?? [];
  if (!values.length || values.length > 1) return null;
  const raw = values[0]?.trim() ?? "";
  const cookie = parseSetCookie(raw);
  const maxAge = cookie?.attributes.get("max-age") ?? "";
  if (
    !cookie
    || cookie.name !== `${secure ? "__Secure-" : ""}teachlab_oidc_tx`
    || !/^[A-Za-z0-9_-]{80,2048}$/.test(cookie.value)
    || !/^[1-9]\d{0,2}$/.test(maxAge)
    || Number(maxAge) > 600
    || !hasExactCookieAttributes(cookie, {
      path: "/api/v1/auth/oidc/callback",
      sameSite: "Lax",
      maxAge,
      httpOnly: true,
      secure,
    })
  ) return null;
  return [rewriteTransactionCookiePath(raw)];
}

function validatedCallbackCookies(response: Response, secure: boolean): string[] | null {
  const extended = response.headers as Headers & {getSetCookie?: () => string[]};
  const values = typeof extended.getSetCookie === "function"
    ? extended.getSetCookie()
    : response.headers.get("set-cookie")?.split(/,(?=\s*(?:__Host-|__Secure-)?teachlab_)/) ?? [];
  if (!values.length || values.length > 3) return null;
  const transactionName = `${secure ? "__Secure-" : ""}teachlab_oidc_tx`;
  const sessionName = `${secure ? "__Host-" : ""}teachlab_session`;
  const csrfName = `${secure ? "__Host-" : ""}teachlab_csrf`;
  const named = new Map<string, {raw: string; cookie: ParsedSetCookie}>();
  for (const raw of values) {
    const cookie = parseSetCookie(raw);
    if (
      !cookie
      || !new Set([transactionName, sessionName, csrfName]).has(cookie.name)
      || named.has(cookie.name)
    ) return null;
    named.set(cookie.name, {raw, cookie});
  }
  const transaction = named.get(transactionName)?.cookie;
  const session = named.get(sessionName)?.cookie;
  const csrf = named.get(csrfName)?.cookie;
  if (
    !transaction
    || transaction.value !== ""
    || !hasExactCookieAttributes(transaction, {
      path: "/api/v1/auth/oidc/callback",
      sameSite: "Lax",
      maxAge: "0",
      httpOnly: true,
      secure,
    })
  ) return null;
  if (response.status === 303) {
    if (!session || !csrf) return null;
    const sessionMaxAge = session.attributes.get("max-age") ?? "";
    const csrfMaxAge = csrf.attributes.get("max-age") ?? "";
    if (
      !/^[A-Za-z0-9._~-]{16,4096}$/.test(session.value)
      || !/^[A-Za-z0-9_-]{43}$/.test(csrf.value)
      || !/^[1-9]\d{0,6}$/.test(sessionMaxAge)
      || Number(sessionMaxAge) > 7 * 24 * 60 * 60
      || csrfMaxAge !== sessionMaxAge
      || !hasExactCookieAttributes(session, {
        path: "/",
        sameSite: "Strict",
        maxAge: sessionMaxAge,
        httpOnly: true,
        secure,
      })
      || !hasExactCookieAttributes(csrf, {
        path: "/",
        sameSite: "Strict",
        maxAge: csrfMaxAge,
        httpOnly: false,
        secure,
      })
    ) return null;
  } else if (session || csrf || values.length !== 1) return null;
  return values.map((value) => value.trim());
}

function singleCookiePart(header: string | null, name: string): string | null {
  const matches = (header?.split(";") ?? []).map((part) => part.trim())
    .filter((part) => part.startsWith(`${name}=`));
  if (
    matches.length !== 1
    || !new RegExp(`^${name}=[A-Za-z0-9_-]{80,2048}$`).test(matches[0] ?? "")
    || /[\r\n]/.test(matches[0] ?? "")
    || (matches[0]?.length ?? 0) > MAX_COOKIE_BYTES
  ) return null;
  return matches[0] ?? null;
}

export async function proxyOidcBegin(request: Request): Promise<Response> {
  const context = authenticatedLoginContext(request);
  if ("response" in context) return context.response;
  const origin = request.headers.get("origin");
  if (
    (origin !== null && origin !== context.config.consoleOrigin)
    || request.headers.get("sec-fetch-site") !== "same-origin"
    || !new Set(["navigate", "same-origin"]).has(request.headers.get("sec-fetch-mode") ?? "")
  ) return safeFailure(403, "untrusted_login_begin");
  const url = new URL(request.url);
  const beginKeys = [...url.searchParams.keys()];
  if (
    beginKeys.some((key) => key !== "return_to")
    || url.searchParams.getAll("return_to").length > 1
  ) return safeFailure(400, "invalid_return_path");
  const returnTo = url.searchParams.get("return_to") ?? "/";
  if (!RETURN_TO_PATTERN.test(returnTo)) return safeFailure(400, "invalid_return_path");
  const target = appsApiOidcUrl(context.config, "begin");
  target.searchParams.set("return_to", returnTo);
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      redirect: "manual",
      cache: "no-store",
      headers: {
        Accept: "text/html",
        Host: target.host,
        Origin: context.config.consoleOrigin,
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
      },
      signal: request.signal,
    });
  } catch {
    return safeFailure(503, "identity_service_unavailable");
  }
  const location = upstream.headers.get("location");
  const secure = context.config.sessionCookieName.startsWith("__Host-");
  const cookies = validatedTransactionCookies(upstream, secure);
  if (upstream.status !== 303 || !location || !cookies) return safeFailure(503, "invalid_identity_response");
  let redirect: URL;
  try { redirect = new URL(location); } catch { return safeFailure(503, "invalid_identity_response"); }
  if (redirect.protocol !== "https:" || redirect.username || redirect.password) return safeFailure(503, "invalid_identity_response");
  const headers = new Headers({Location: redirect.toString(), "Cache-Control": "no-store, max-age=0"});
  for (const cookie of cookies) headers.append("Set-Cookie", cookie);
  return new Response(null, {status: 303, headers});
}

export async function proxyOidcAccountStepUp(request: Request): Promise<Response> {
  const context = authenticatedSessionContext(request, {
    requireSession: true,
    requireCsrf: false,
  });
  if ("response" in context) return context.response;
  const origin = request.headers.get("origin");
  if (
    (origin !== null && origin !== context.config.consoleOrigin)
    || request.headers.get("sec-fetch-site") !== "same-origin"
    || !new Set(["navigate", "same-origin"]).has(request.headers.get("sec-fetch-mode") ?? "")
  ) return safeFailure(403, "untrusted_login_begin");
  const url = new URL(request.url);
  if (
    [...url.searchParams.keys()].some((key) => key !== "return_to")
    || url.searchParams.getAll("return_to").length > 1
  ) return safeFailure(400, "invalid_return_path");
  const returnTo = url.searchParams.get("return_to") ?? "/";
  if (!RETURN_TO_PATTERN.test(returnTo)) return safeFailure(400, "invalid_return_path");
  const target = appsApiOidcUrl(context.config, "account-step-up");
  target.searchParams.set("return_to", returnTo);
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      redirect: "manual",
      cache: "no-store",
      headers: {
        Accept: "text/html",
        Cookie: context.cookieHeader,
        Host: target.host,
        Origin: context.config.consoleOrigin,
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
      },
      signal: request.signal,
    });
  } catch {
    return safeFailure(503, "identity_service_unavailable");
  }
  const location = upstream.headers.get("location");
  const secure = context.config.sessionCookieName.startsWith("__Host-");
  const cookies = validatedTransactionCookies(upstream, secure);
  if (upstream.status !== 303 || !location || !cookies) {
    return safeFailure(503, "invalid_identity_response");
  }
  let redirect: URL;
  try { redirect = new URL(location); } catch {
    return safeFailure(503, "invalid_identity_response");
  }
  if (redirect.protocol !== "https:" || redirect.username || redirect.password) {
    return safeFailure(503, "invalid_identity_response");
  }
  const headers = new Headers({Location: redirect.toString(), "Cache-Control": "no-store, max-age=0"});
  for (const cookie of cookies) headers.append("Set-Cookie", cookie);
  return new Response(null, {status: 303, headers});
}

export async function proxyOidcCallback(request: Request): Promise<Response> {
  const context = authenticatedLoginContext(request);
  if ("response" in context) return context.response;
  const secure = context.config.sessionCookieName.startsWith("__Host-");
  const clearTransaction = clearBrowserTransactionCookie(secure);
  if (
    !new Set(["cross-site", "same-site", "same-origin", "none"]).has(request.headers.get("sec-fetch-site") ?? "")
    || request.headers.get("sec-fetch-mode") !== "navigate"
  ) return safeFailure(403, "untrusted_login_callback", clearTransaction);
  const incoming = new URL(request.url);
  const queryKeys = [...incoming.searchParams.keys()];
  if (
    incoming.search.length > 8192
    || queryKeys.some((key) => !CALLBACK_QUERY_KEYS.has(key))
    || [...new Set(queryKeys)].some((key) => (
      incoming.searchParams.getAll(key).length !== 1
      || (incoming.searchParams.get(key)?.length ?? 0) > 4096
    ))
  ) {
    return safeFailure(400, "invalid_callback_parameters", clearTransaction);
  }
  const target = appsApiOidcUrl(context.config, "callback", incoming.search);
  const transactionName = context.config.sessionCookieName.startsWith("__Host-")
    ? "__Secure-teachlab_oidc_tx"
    : "teachlab_oidc_tx";
  const cookie = singleCookiePart(request.headers.get("cookie"), transactionName);
  if (!cookie) {
    return safeFailure(400, "missing_login_transaction", clearTransaction);
  }
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      redirect: "manual",
      cache: "no-store",
      headers: {
        Accept: "text/html",
        Cookie: cookie,
        Host: target.host,
        "Sec-Fetch-Site": request.headers.get("sec-fetch-site") ?? "cross-site",
        "Sec-Fetch-Mode": "navigate",
      },
      signal: request.signal,
    });
  } catch {
    return safeFailure(503, "identity_service_unavailable", clearTransaction);
  }
  const cookies = validatedCallbackCookies(upstream, secure);
  if (!cookies) return safeFailure(503, "invalid_identity_response", clearTransaction);
  const headers = new Headers({"Cache-Control": "no-store, max-age=0"});
  for (const value of cookies) {
    headers.append(
      "Set-Cookie",
      /^(?:__Secure-)?teachlab_oidc_tx=/.test(value)
        ? rewriteTransactionCookiePath(value)
        : value,
    );
  }
  if (upstream.status === 303) {
    const location = upstream.headers.get("location") ?? "";
    if (!RETURN_TO_PATTERN.test(location)) return safeFailure(503, "invalid_identity_response", clearTransaction);
    const sessionMaxAge = callbackSessionMaxAge(cookies, secure);
    if (!sessionMaxAge) return safeFailure(503, "invalid_identity_response", clearTransaction);
    // A successful identity transition must make every pre-bootstrap offline
    // document fail closed. The marker contains no identity and is cleared only
    // after the new authenticated cache scope has been reconciled in Providers.
    headers.append("Set-Cookie", accountTransitionCookie(secure, sessionMaxAge));
    headers.set("Location", location);
    return new Response(null, {status: 303, headers});
  }
  await upstream.body?.cancel().catch(() => undefined);
  return new Response("TeachLab 登录失败，请返回并重新开始。", {
    status: upstream.status === 503 ? 503 : 400,
    headers,
  });
}
