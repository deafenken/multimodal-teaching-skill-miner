import {harnessBffConfiguration, harnessReadinessTarget} from "../../lib/harness-bff.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  let configuration: ReturnType<typeof harnessBffConfiguration>;
  try {
    configuration = harnessBffConfiguration();
  } catch {
    return Response.json({status: "not_ready", reason: "backend_not_configured"}, {status: 503});
  }
  let target: URL;
  try {
    target = harnessReadinessTarget(configuration);
  } catch {
    return Response.json({status: "not_ready", reason: "backend_not_configured"}, {status: 503});
  }
  try {
    const response = await fetch(target, {cache: "no-store", signal: AbortSignal.timeout(2_000)});
    if (!response.ok) {
      await response.body?.cancel().catch(() => undefined);
      return Response.json(
        {status: "not_ready", reason: "backend_rejected_probe", backend_status: response.status},
        {status: 503},
      );
    }
    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.toLowerCase().startsWith("application/json")) {
      await response.body?.cancel().catch(() => undefined);
      return Response.json({status: "not_ready", reason: "backend_contract_mismatch"}, {status: 503});
    }
    const payload = await response.json() as Record<string, unknown>;
    const contractMatches = configuration.mode === "local_python"
      ? payload.schema_version === "1.1"
        && payload.dashboard_kind === "loopback_interactive_teacher_agent"
        && Boolean(payload.provider_status)
        && typeof payload.provider_status === "object"
      : payload.status === "ready" && payload.service === "teachlab-api";
    if (!contractMatches) {
      return Response.json({status: "not_ready", reason: "backend_contract_mismatch"}, {status: 503});
    }
    return Response.json(
      {status: "ready", service: "teachlab-console", backend: "ready", mode: configuration.mode},
      {status: 200, headers: {"Cache-Control": "no-store, max-age=0"}},
    );
  } catch {
    return Response.json({status: "not_ready", reason: "backend_unavailable"}, {status: 503});
  }
}
