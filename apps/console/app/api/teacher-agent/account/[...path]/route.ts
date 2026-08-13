import {proxyAccountDataRights} from "../../../../../lib/account-data-rights-bff.ts";
import {markRuntimeActivity} from "../../../../../lib/runtime-activity.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

async function proxy(request: Request, context: {params: Promise<{path: string[]}>}) {
  markRuntimeActivity();
  return proxyAccountDataRights(request, (await context.params).path);
}

export const GET = proxy;
export const POST = proxy;
