import {NextRequest, NextResponse} from "next/server.js";
import {
  harnessProxyRequestBodyLimit,
  prepareHarnessProxy,
} from "../../../../lib/harness-bff.ts";
import {boundedRequestBody} from "../../../../lib/account-data-rights-bff.ts";
import {markRuntimeActivity} from "../../../../lib/runtime-activity.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

async function proxy(request: NextRequest, path: string[]) {
  markRuntimeActivity();
  const joinedPath = path.join("/");
  const curriculumBlueprintPath = /^api\/syllabi\/[^/]+\/curriculum-blueprint$/.test(joinedPath);
  if (curriculumBlueprintPath && request.method !== "GET" && request.method !== "HEAD") {
    return NextResponse.json({error: "resource not found"}, {status: 404});
  }
  const syllabusPath = /^api\/syllabi(?:\/(?:generate|import|[^/]+|[^/]+\/(?:download|versions|curriculum-blueprint|revisions|publish|rollback)|[^/]+\/lessons\/[^/]+\/start-payload))?$/.test(joinedPath);
  const curriculumAuthorityPath = /^api\/curriculum\/(?:review|seal|revoke)$/.test(joinedPath);
  if (curriculumAuthorityPath && request.method !== "POST") {
    return NextResponse.json({error: "resource not found"}, {status: 404});
  }
  const projectPath = /^api\/projects(?:\/(?:trash|bootstrap)|\/project_[0-9a-f]{24}(?:\/(?:update|chat-thread|browse|reference|remove-reference|note|trash|restore|export|purge))?)?$/.test(joinedPath);
  const learningReviewPath = /^api\/learning-reviews\/(?:due|claim|release)$/.test(joinedPath);
  const metacognitionPath = /^api\/metacognition\/(?:list|predict|pair)$/.test(joinedPath);
  if (metacognitionPath && request.method !== "POST") {
    return NextResponse.json({error: "resource not found"}, {status: 404});
  }
  const backgroundTaskPath = /^api\/tasks\/(?:list|status|cancel|resume)$/.test(joinedPath);
  const adjudicationPath = /^api\/adjudication\/(?:candidates|list|enqueue|claim|decide)$/.test(joinedPath);
  const consentPath = /^api\/consent\/(?:grant|list|revoke)$/.test(joinedPath);
  const safeguardingPath = /^api\/safeguarding\/(?:list|dispatch|case\/(?:acknowledge|close)|escalation\/(?:overdue|acknowledge))$/.test(joinedPath);
  if (safeguardingPath && request.method !== "POST") {
    return NextResponse.json({error: "resource not found"}, {status: 404});
  }
  const resourcePath = joinedPath === "api/resource" || joinedPath === "api/resource/review";
  if (joinedPath === "api/resource/review" && request.method !== "POST") {
    return NextResponse.json({error: "resource not found"}, {status: 404});
  }
  if (!path.length || (!syllabusPath && !curriculumAuthorityPath && !projectPath && !learningReviewPath && !metacognitionPath && !backgroundTaskPath && !adjudicationPath && !consentPath && !safeguardingPath && !resourcePath && !["api/bootstrap", "api/chat", "api/start", "api/session", "api/step", "api/command", "api/cancel", "api/attachment"].includes(joinedPath))) {
    return NextResponse.json({error: "resource not found"}, {status: 404});
  }
  const prepared = prepareHarnessProxy(request, joinedPath);
  if ("response" in prepared) return prepared.response;
  let body: string | undefined;
  if (request.method !== "GET" && request.method !== "HEAD") {
    const maximumBytes = harnessProxyRequestBodyLimit(joinedPath);
    const bounded = await boundedRequestBody(request, maximumBytes);
    if (bounded === null) {
      return NextResponse.json({error: "request body is too large"}, {status: 413});
    }
    body = bounded.toString("utf8");
  }
  try {
    const response = await fetch(prepared.target, {
      method: request.method,
      headers: prepared.headers,
      body,
      cache: "no-store",
      redirect: "manual",
      signal: request.signal,
    });
    const responseHeaders = new Headers();
    const responseContentType = response.headers.get("content-type");
    if (responseContentType) responseHeaders.set("content-type", responseContentType);
    const responseContentDisposition = response.headers.get("content-disposition");
    if (responseContentDisposition) responseHeaders.set("content-disposition", responseContentDisposition);
    const responseManifestSha256 = response.headers.get("x-manifest-sha256");
    if (responseManifestSha256) responseHeaders.set("x-manifest-sha256", responseManifestSha256);
    responseHeaders.set("cache-control", "no-store, max-age=0");
    if (response.status === 401) responseHeaders.set("x-teachlab-auth-state", "authentication_required");
    return new NextResponse(response.body, {status: response.status, headers: responseHeaders});
  } catch {
    return NextResponse.json({error: "Teaching Agent backend is unavailable"}, {status: 503});
  }
}

export async function GET(request: NextRequest, context: {params: Promise<{path: string[]}>}) {
  return proxy(request, (await context.params).path);
}

export async function HEAD(request: NextRequest, context: {params: Promise<{path: string[]}>}) {
  return proxy(request, (await context.params).path);
}

export async function POST(request: NextRequest, context: {params: Promise<{path: string[]}>}) {
  return proxy(request, (await context.params).path);
}
