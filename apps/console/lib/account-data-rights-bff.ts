import {authenticatedAccountContext} from "./harness-bff.ts";

const ACCOUNT_PATHS = new Map<string, {method: "GET" | "POST"; session: boolean}>([
  ["export", {method: "GET", session: true}],
  ["deletion/prepare", {method: "POST", session: true}],
  ["deletion/confirm", {method: "POST", session: true}],
  ["deletion/resume", {method: "POST", session: false}],
  ["deletion/status", {method: "GET", session: false}],
]);
const MAX_BODY_BYTES = 16 * 1024;
const MAX_EXPORT_BYTES = 400 * 1024 * 1024;
const MAX_JSON_RESPONSE_BYTES = 64 * 1024;

function rejected(status: number, code: string): Response {
  return Response.json({error: code, code}, {
    status,
    headers: {"Cache-Control": "no-store, max-age=0", "X-Content-Type-Options": "nosniff"},
  });
}

function responseCookies(response: Response): string[] {
  const headers = response.headers as Headers & {getSetCookie?: () => string[]};
  if (typeof headers.getSetCookie === "function") return headers.getSetCookie();
  return response.headers.get("set-cookie")?.split(/,(?=\s*(?:__Host-)?teachlab_(?:session|csrf|deletion_status)=)/) ?? [];
}

function validatedCookies(
  response: Response,
  allowed: ReadonlySet<string>,
): string[] | null {
  const output: string[] = [];
  const seen = new Set<string>();
  for (const raw of responseCookies(response)) {
    if (/\r|\n/.test(raw) || Buffer.byteLength(raw, "utf8") > 4096) return null;
    const separator = raw.indexOf("=");
    const name = separator > 0 ? raw.slice(0, separator).trim() : "";
    if (
      !allowed.has(name)
      || seen.has(name)
      || !/;\s*Path=\//i.test(raw)
      || !/;\s*SameSite=Strict/i.test(raw)
      || /;\s*Domain=/i.test(raw)
      || ((name !== "teachlab_csrf" && name !== "__Host-teachlab_csrf")
        && !/;\s*HttpOnly(?:;|$)/i.test(raw))
      || (name.startsWith("__Host-") && !/;\s*Secure(?:;|$)/i.test(raw))
    ) return null;
    seen.add(name);
    output.push(raw.trim());
  }
  return output;
}

export async function boundedRequestBody(
  request: Request,
  maximumBytes: number
): Promise<Buffer | null> {
  const rejectBody = async (): Promise<null> => {
    await request.body?.cancel("request_too_large_or_length_mismatch").catch(
      () => undefined
    );
    return null;
  };
  const declaredRaw = request.headers.get("content-length");
  let declared: number | undefined;
  if (declaredRaw !== null) {
    if (!/^(?:0|[1-9][0-9]*)$/.test(declaredRaw)) return rejectBody();
    declared = Number(declaredRaw);
    if (!Number.isSafeInteger(declared) || declared > maximumBytes) {
      return rejectBody();
    }
  }
  if (!request.body) return declared === undefined || declared === 0
    ? Buffer.alloc(0)
    : null;
  const reader = request.body.getReader();
  const chunks: Buffer[] = [];
  let observed = 0;
  try {
    while (true) {
      const next = await reader.read();
      if (next.done) break;
      const chunk = Buffer.from(next.value);
      observed += chunk.byteLength;
      if (observed > maximumBytes || (declared !== undefined && observed > declared)) {
        await reader.cancel("request_too_large").catch(() => undefined);
        return null;
      }
      chunks.push(chunk);
    }
  } catch {
    await reader.cancel("invalid_request_body").catch(() => undefined);
    return null;
  } finally {
    reader.releaseLock();
  }
  if (declared !== undefined && observed !== declared) return null;
  return Buffer.concat(chunks, observed);
}

function boundedUpstreamStream(
  body: ReadableStream<Uint8Array> | null,
  maximumBytes: number,
  exactBytes?: number
): ReadableStream<Uint8Array> | null {
  if (!body) return null;
  const reader = body.getReader();
  let observed = 0;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const next = await reader.read();
        if (next.done) {
          if (exactBytes !== undefined && observed !== exactBytes) {
            controller.error(new Error("upstream_length_mismatch"));
          } else {
            controller.close();
          }
          return;
        }
        observed += next.value.byteLength;
        if (observed > maximumBytes || (exactBytes !== undefined && observed > exactBytes)) {
          await reader.cancel("upstream_response_too_large").catch(() => undefined);
          controller.error(new Error("upstream_response_too_large"));
          return;
        }
        controller.enqueue(next.value);
      } catch {
        controller.error(new Error("upstream_stream_failed"));
      }
    },
    async cancel(reason) {
      await reader.cancel(reason).catch(() => undefined);
    }
  });
}

export async function proxyAccountDataRights(request: Request, rawPath: string[]): Promise<Response> {
  const path = rawPath.join("/");
  const contract = ACCOUNT_PATHS.get(path);
  if (!contract || request.method !== contract.method) return rejected(404, "resource_not_found");
  const context = authenticatedAccountContext(request, {
    requireSession: contract.session,
    requireCsrf: contract.method === "POST" && path !== "deletion/resume",
  });
  if ("response" in context) return context.response;
  let body: string | undefined;
  if (contract.method === "POST") {
    const bounded = await boundedRequestBody(request, MAX_BODY_BYTES);
    if (bounded === null) return rejected(413, "request_too_large");
    body = bounded.toString("utf8");
  }
  const target = new URL(`/api/v1/account/${path}`, context.config.upstreamBase);
  const headers = new Headers({
    Accept: path === "export" ? "application/zip" : "application/json",
    Origin: context.config.consoleOrigin,
    "Sec-Fetch-Site": "same-origin",
    Cookie: context.cookieHeader,
  });
  if (contract.method === "POST") {
    headers.set("Content-Type", "application/json");
    if (context.csrfToken) headers.set("x-csrf-token", context.csrfToken);
  }
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: contract.method,
      headers,
      body,
      credentials: "include",
      redirect: "manual",
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return rejected(503, "account_service_unavailable");
  }
  const responseHeaders = new Headers({
    "Cache-Control": "no-store, max-age=0",
    "X-Content-Type-Options": "nosniff",
  });
  const cookies = validatedCookies(upstream, new Set([
    context.config.sessionCookieName,
    context.config.csrfCookieName,
    context.config.deletionStatusCookieName,
  ]));
  if (cookies === null) {
    await upstream.body?.cancel().catch(() => undefined);
    return rejected(503, "invalid_account_service_response");
  }
  for (const cookie of cookies) responseHeaders.append("Set-Cookie", cookie);
  if (upstream.status === 401) responseHeaders.set("x-teachlab-auth-state", "authentication_required");
  let responseBody: ReadableStream<Uint8Array> | null;
  if (path === "export" && upstream.ok) {
    const declared = Number(upstream.headers.get("content-length") ?? "");
    const disposition = upstream.headers.get("content-disposition") ?? "";
    const manifest = upstream.headers.get("x-manifest-sha256") ?? "";
    if (
      !Number.isSafeInteger(declared)
      || declared < 1
      || declared > MAX_EXPORT_BYTES
      || !/^attachment; filename="[A-Za-z0-9_.-]{1,160}"$/.test(disposition)
      || !/^[0-9a-f]{64}$/.test(manifest)
      || upstream.headers.get("content-type")?.split(";", 1)[0] !== "application/zip"
    ) {
      await upstream.body?.cancel().catch(() => undefined);
      return rejected(503, "invalid_account_export_response");
    }
    responseHeaders.set("Content-Type", "application/zip");
    responseHeaders.set("Content-Length", String(declared));
    responseHeaders.set("Content-Disposition", disposition);
    responseHeaders.set("x-manifest-sha256", manifest);
    responseBody = boundedUpstreamStream(upstream.body, MAX_EXPORT_BYTES, declared);
  } else {
    const declaredRaw = upstream.headers.get("content-length");
    if (declaredRaw !== null && (
      !/^(?:0|[1-9][0-9]*)$/.test(declaredRaw)
      || Number(declaredRaw) > MAX_JSON_RESPONSE_BYTES
    )) {
      await upstream.body?.cancel().catch(() => undefined);
      return rejected(503, "invalid_account_service_response");
    }
    responseHeaders.set("Content-Type", "application/json; charset=utf-8");
    responseBody = boundedUpstreamStream(
      upstream.body,
      MAX_JSON_RESPONSE_BYTES,
      declaredRaw === null ? undefined : Number(declaredRaw)
    );
  }
  return new Response(responseBody, {status: upstream.status, headers: responseHeaders});
}
