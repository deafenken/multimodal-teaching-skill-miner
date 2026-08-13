import {NextRequest} from "next/server.js";
import {prepareHarnessProxy} from "../../../../lib/harness-bff.ts";
import {boundedRequestBody} from "../../../../lib/account-data-rights-bff.ts";
import {markRuntimeActivity} from "../../../../lib/runtime-activity.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type Operation = "chat" | "start" | "step";

// Kept identical to the Python stream route and apps/api's default Harness
// request gate; large resource bytes travel through /api/resource instead.
const MAX_STREAM_REQUEST_BYTES = 64 * 1024;

interface StreamEnvelope {
  operation: Operation;
  payload: Record<string, unknown>;
  request_id?: string;
  run_id?: string;
  turn_id?: string;
  after_sequence?: number;
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function normalizeRequest(value: unknown): StreamEnvelope | null {
  const body = recordValue(value);
  if (!body) return null;
  const rawOperation = body.operation;
  if (rawOperation !== undefined && rawOperation !== "chat" && rawOperation !== "start" && rawOperation !== "step") return null;
  const operation: Operation = rawOperation === "chat" || rawOperation === "start" || rawOperation === "step"
    ? rawOperation
    : "step";
  const payload = rawOperation === "chat" || rawOperation === "start" || rawOperation === "step"
    ? recordValue(body.payload)
    : body;
  if (!payload) return null;

  const envelope: StreamEnvelope = {operation, payload};
  if (body.request_id !== undefined) {
    if (typeof body.request_id !== "string" || !body.request_id.trim() || body.request_id.length > 160) return null;
    envelope.request_id = body.request_id.trim();
  }
  if (body.run_id !== undefined || body.turn_id !== undefined) {
    if (
      typeof body.run_id !== "string" || !body.run_id.trim() || body.run_id.length > 160
      || typeof body.turn_id !== "string" || !body.turn_id.trim() || body.turn_id.length > 160
    ) return null;
    envelope.run_id = body.run_id.trim();
    envelope.turn_id = body.turn_id.trim();
  }
  if (body.after_sequence !== undefined) {
    if (!Number.isInteger(body.after_sequence) || Number(body.after_sequence) < 0 || !envelope.run_id) return null;
    envelope.after_sequence = Number(body.after_sequence);
  }
  return envelope;
}

function proxyHeaders(response: Response) {
  const headers = new Headers({
    "Cache-Control": "no-store, no-transform",
    "Content-Type": response.headers.get("content-type") ?? "text/event-stream; charset=utf-8",
    "X-Accel-Buffering": "no",
  });
  const retryAfter = response.headers.get("retry-after");
  if (retryAfter) headers.set("Retry-After", retryAfter);
  const runId = response.headers.get("x-harness-run-id");
  const turnId = response.headers.get("x-harness-turn-id");
  const taskId = response.headers.get("x-background-task-id");
  const taskVersion = response.headers.get("x-background-task-version");
  if (runId) headers.set("X-Harness-Run-ID", runId);
  if (turnId) headers.set("X-Harness-Turn-ID", turnId);
  if (taskId) headers.set("X-Background-Task-ID", taskId);
  if (taskVersion) headers.set("X-Background-Task-Version", taskVersion);
  return headers;
}

export async function POST(request: NextRequest) {
  markRuntimeActivity();
  const prepared = prepareHarnessProxy(request, "api/stream");
  if ("response" in prepared) return prepared.response;
  let requestBody: StreamEnvelope | null;
  let removeAbortListener: () => void = () => {};
  const bounded = await boundedRequestBody(request, MAX_STREAM_REQUEST_BYTES);
  if (bounded === null) {
    return Response.json({error: "request body is too large"}, {status: 413});
  }
  try {
    requestBody = normalizeRequest(JSON.parse(bounded.toString("utf8")));
  } catch {
    requestBody = null;
  }
  if (!requestBody) return Response.json({error: "request body is invalid JSON"}, {status: 400});

  try {
    // Forward each upstream byte chunk unchanged so provider token boundaries
    // and harness lifecycle events stay intact. The thin reader wrapper exists
    // only to make downstream cancellation explicitly close and unlock the
    // Python response when the user presses Stop or navigates away.
    const upstreamAbort = new AbortController();
    const abortUpstream = () => upstreamAbort.abort(request.signal.reason);
    if (request.signal.aborted) abortUpstream();
    else {
      request.signal.addEventListener("abort", abortUpstream, {once: true});
      removeAbortListener = () => request.signal.removeEventListener("abort", abortUpstream);
    }
    const headers = new Headers(prepared.headers);
    headers.set("Accept", "text/event-stream");
    headers.set("Content-Type", "application/json");
    const response = await fetch(prepared.target, {
      method: "POST",
      headers,
      body: JSON.stringify(requestBody),
      cache: "no-store",
      redirect: "manual",
      signal: upstreamAbort.signal,
    });
    if (!response.body) {
      removeAbortListener();
      return new Response(null, {status: response.status, headers: proxyHeaders(response)});
    }
    const reader = response.body.getReader();
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      removeAbortListener();
      reader.releaseLock();
    };
    const proxyStream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          const {done, value} = await reader.read();
          if (done) {
            markRuntimeActivity();
            controller.close();
            release();
            return;
          }
          markRuntimeActivity();
          controller.enqueue(value);
        } catch (error) {
          release();
          controller.error(error);
        }
      },
      async cancel(reason) {
        upstreamAbort.abort(reason);
        try {
          await reader.cancel(reason);
        } finally {
          release();
        }
      },
    });
    return new Response(proxyStream, {
      status: response.status,
      headers: proxyHeaders(response),
    });
  } catch (error) {
    removeAbortListener();
    if (request.signal.aborted || (error instanceof DOMException && error.name === "AbortError")) {
      return new Response(null, {status: 499});
    }
    return Response.json({error: "Teaching Agent backend is unavailable"}, {status: 503});
  }
}
