import fs from "node:fs";

import {localSecurityMode} from "../../../../../lib/harness-bff.ts";
import {enforceLocalRequest} from "../../../../../lib/local-request-security.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(request: Request) {
  if (localSecurityMode() !== "local_python") {
    return Response.json(
      {error: "runtime stop control is available only in local_python mode"},
      {status: 404, headers: {"Cache-Control": "no-store"}},
    );
  }
  const rejection = enforceLocalRequest(request, {requireSession: true, requireCsrf: true});
  if (rejection) return rejection;
  const target = process.env.TEACHLAB_RUNTIME_STOP_REQUEST?.trim();
  if (!target) return Response.json({error: "runtime stop control is unavailable"}, {status: 503});
  try {
    const descriptor = fs.openSync(target, "wx", 0o600);
    try {
      fs.writeFileSync(descriptor, `${JSON.stringify({requested_at: new Date().toISOString(), source: "local_console_ui"})}\n`);
      fs.fsyncSync(descriptor);
    } finally {
      fs.closeSync(descriptor);
    }
    return Response.json({status: "stopping"}, {status: 202, headers: {"Cache-Control": "no-store"}});
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "EEXIST") {
      return Response.json({status: "stopping"}, {status: 202, headers: {"Cache-Control": "no-store"}});
    }
    return Response.json({error: "runtime stop request could not be recorded"}, {status: 503});
  }
}
