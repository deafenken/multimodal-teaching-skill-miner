import {proxyOidcCallback} from "../../../../../../lib/oidc-login-proxy.ts";
import {markRuntimeActivity} from "../../../../../../lib/runtime-activity.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(request: Request) {
  markRuntimeActivity();
  return proxyOidcCallback(request);
}
