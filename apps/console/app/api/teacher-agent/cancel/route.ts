import {NextRequest} from "next/server.js";
import {prepareHarnessProxy} from "../../../../lib/harness-bff.ts";
import {boundedRequestBody} from "../../../../lib/account-data-rights-bff.ts";
import {markRuntimeActivity} from "../../../../lib/runtime-activity.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  markRuntimeActivity();
  const prepared = prepareHarnessProxy(request, "api/cancel");
  if ("response" in prepared) return prepared.response;
  try {
    const body = await boundedRequestBody(request, 4 * 1024);
    if (body === null) {
      return Response.json({error: "request body is too large"}, {status: 413});
    }
    const response = await fetch(prepared.target, {
      method: "POST",
      headers: prepared.headers,
      body: body.toString("utf8"),
      cache: "no-store",
      redirect: "manual",
      signal: request.signal,
    });
    return new Response(response.body, {
      status: response.status,
      headers: {"Cache-Control": "no-store", "Content-Type": response.headers.get("content-type") ?? "application/json; charset=utf-8"},
    });
  } catch {
    return Response.json({error: "Teaching Agent backend is unavailable"}, {status: 503});
  }
}
