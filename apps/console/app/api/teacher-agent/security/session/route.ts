import {
  issueLocalSecuritySession,
  localSecurityRejection,
  validateLocalRequest,
} from "../../../../../lib/local-request-security.ts";
import {
  appsApiSessionUrl,
  authenticatedSessionContext,
  harnessConfigurationRejection,
  HarnessBffConfigurationError,
  localSecurityMode,
} from "../../../../../lib/harness-bff.ts";
import {markRuntimeActivity} from "../../../../../lib/runtime-activity.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function safeAppsApiFailure(status: number) {
  const safeStatus = status === 401 || status === 403 ? status : 503;
  return Response.json(
    {
      error: safeStatus === 401
        ? "TeachLab authentication is required"
        : safeStatus === 403
          ? "TeachLab authentication request was rejected"
          : "TeachLab identity service is unavailable",
      code: safeStatus === 401
        ? "authentication_required"
        : safeStatus === 403
          ? "authentication_rejected"
          : "identity_service_unavailable",
    },
    {
      status: safeStatus,
      headers: {
        "Cache-Control": "no-store, max-age=0",
        "x-teachlab-auth-state": safeStatus === 401 ? "authentication_required" : "unavailable",
        "x-teachlab-security-rejection": safeStatus === 401 ? "authentication_required" : "authentication_rejected",
      },
    },
  );
}

function responseSetCookies(response: Response): string[] {
  const extended = response.headers as Headers & {getSetCookie?: () => string[]};
  if (typeof extended.getSetCookie === "function") return extended.getSetCookie();
  const combined = response.headers.get("set-cookie");
  return combined ? combined.split(/,(?=\s*(?:__Host-)?teachlab_(?:session|csrf)=)/) : [];
}

function validatedSessionCookies(
  response: Response,
  expectedNames: readonly [string, string],
  clearOnly = false,
): string[] | null {
  const found = new Map<string, string>();
  for (const raw of responseSetCookies(response)) {
    if (/\r|\n/.test(raw)) return null;
    const name = raw.slice(0, raw.indexOf("=")).trim();
    if (!expectedNames.includes(name) || found.has(name)) return null;
    if (
      !/;\s*Path=\//i.test(raw)
      || !/;\s*SameSite=Strict/i.test(raw)
      || /;\s*Domain=/i.test(raw)
      || (name === expectedNames[0] && !/;\s*HttpOnly(?:;|$)/i.test(raw))
      || (name.startsWith("__Host-") && !/;\s*Secure(?:;|$)/i.test(raw))
      || (clearOnly && (
        raw.slice(raw.indexOf("=") + 1, raw.indexOf(";")).length !== 0
        || !/;\s*Max-Age=0(?:;|$)/i.test(raw)
      ))
    ) return null;
    found.set(name, raw.trim());
  }
  return expectedNames.map((name) => found.get(name) ?? "").filter(Boolean).length === expectedNames.length
    ? expectedNames.map((name) => found.get(name) as string)
    : null;
}

async function appsApiFetch(
  request: Request,
  init: {method: "GET" | "POST" | "DELETE"; requireSession: boolean; requireCsrf: boolean},
) {
  const context = authenticatedSessionContext(request, {
    requireSession: init.requireSession,
    requireCsrf: init.requireCsrf,
  });
  if ("response" in context) return context.response;
  const headers = new Headers({
    Accept: "application/json",
    Origin: context.config.consoleOrigin,
    "Sec-Fetch-Site": "same-origin",
  });
  if (context.cookieHeader) headers.set("Cookie", context.cookieHeader);
  if (init.requireCsrf) headers.set("x-csrf-token", context.csrfToken);
  if (init.method === "POST") {
    headers.set("Content-Type", "application/json");
  }

  let upstream: Response;
  try {
    upstream = await fetch(appsApiSessionUrl(context.config), {
      method: init.method,
      headers,
      body: init.method === "POST" ? "{}" : undefined,
      cache: "no-store",
      credentials: "include",
      redirect: "manual",
      signal: request.signal,
    });
  } catch {
    return safeAppsApiFailure(503);
  }
  const expectedStatus = init.method === "POST" ? 201 : init.method === "DELETE" ? 204 : 200;
  if (!upstream.ok || upstream.status !== expectedStatus) {
    await upstream.body?.cancel().catch(() => undefined);
    return safeAppsApiFailure(upstream.status);
  }

  const responseHeaders = new Headers({
    "Cache-Control": "no-store, max-age=0",
    "Content-Type": "application/json; charset=utf-8",
    "x-teachlab-auth-state": init.method === "DELETE" ? "signed_out" : "authenticated",
  });
  let mintedCookies: string[] | null = null;
  if (init.method === "POST" || init.method === "DELETE") {
    mintedCookies = validatedSessionCookies(upstream, [
      context.config.sessionCookieName,
      context.config.csrfCookieName,
    ], init.method === "DELETE");
    if (!mintedCookies) {
      await upstream.body?.cancel().catch(() => undefined);
      return safeAppsApiFailure(503);
    }
    for (const cookie of mintedCookies) responseHeaders.append("Set-Cookie", cookie);
  }

  if (init.method === "DELETE") {
    await upstream.body?.cancel().catch(() => undefined);
    return new Response(null, {status: 204, headers: responseHeaders});
  }
  let csrfToken = context.csrfToken;
  let expiresAt: string | undefined;
  if (init.method === "POST") {
    let payload: unknown;
    try {
      payload = await upstream.json();
    } catch {
      return safeAppsApiFailure(503);
    }
    const row = payload && typeof payload === "object" && !Array.isArray(payload)
      ? payload as Record<string, unknown>
      : {};
    csrfToken = typeof row.csrfToken === "string" ? row.csrfToken : "";
    expiresAt = typeof row.expiresAt === "string" ? row.expiresAt : undefined;
  } else {
    await upstream.body?.cancel().catch(() => undefined);
  }
  if (!/^[A-Za-z0-9_-]{43}$/.test(csrfToken)) return safeAppsApiFailure(503);
  if (mintedCookies) {
    const rawCsrf = mintedCookies.find((cookie) => cookie.startsWith(`${context.config.csrfCookieName}=`));
    const encodedCsrf = rawCsrf?.slice(rawCsrf.indexOf("=") + 1, rawCsrf.indexOf(";"));
    let cookieCsrf = "";
    try {
      cookieCsrf = decodeURIComponent(encodedCsrf ?? "");
    } catch {
      return safeAppsApiFailure(503);
    }
    if (cookieCsrf !== csrfToken) return safeAppsApiFailure(503);
  }
  return Response.json(
    {
      authenticated: true,
      connection: "authenticated_apps_api",
      csrf_token: csrfToken,
      ...(expiresAt ? {expires_at: expiresAt} : {}),
    },
    {status: 200, headers: responseHeaders},
  );
}

export async function GET(request: Request) {
  markRuntimeActivity();
  const mode = localSecurityMode();
  if (mode === "authenticated_apps_api") {
    return appsApiFetch(request, {method: "GET", requireSession: true, requireCsrf: false});
  }
  if (mode === "unconfigured") {
    return harnessConfigurationRejection(new HarnessBffConfigurationError("mode_not_configured"));
  }
  const result = validateLocalRequest(request);
  if (!result.ok) return localSecurityRejection(result);
  return issueLocalSecuritySession(request, result.session);
}

export async function POST(request: Request) {
  markRuntimeActivity();
  return appsApiFetch(request, {method: "POST", requireSession: false, requireCsrf: false});
}

export async function DELETE(request: Request) {
  markRuntimeActivity();
  return appsApiFetch(request, {method: "DELETE", requireSession: true, requireCsrf: true});
}
