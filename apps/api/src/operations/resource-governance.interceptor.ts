import {
  CallHandler,
  ExecutionContext,
  HttpException,
  Inject,
  Injectable,
  NestInterceptor
} from "@nestjs/common";
import type {FastifyReply, FastifyRequest} from "fastify";
import {finalize, type Observable} from "rxjs";

import type {AuthenticatedPrincipal} from "../auth/auth-provider.port";
import {
  ResourceGovernor,
  ResourceGovernorError,
  type ResourceLease,
  type ResourceLimitClass
} from "./resource-governor";


export const RESOURCE_GOVERNOR = Symbol("RESOURCE_GOVERNOR");

type GovernedRequest = FastifyRequest & {principal?: AuthenticatedPrincipal};

const STREAM_ROUTES = new Set([
  "/api/events",
  "/api/v1/sessions/:sessionId/events",
  "/api/v1/harness/*"
]);

function routeTemplate(request: FastifyRequest): string {
  const template = request.routeOptions?.url;
  return typeof template === "string" && template.startsWith("/")
    ? template
    : "unmatched";
}

function isStream(request: FastifyRequest): boolean {
  const template = routeTemplate(request);
  if (template === "/api/v1/harness/*") {
    return request.url.split("?", 1)[0]?.endsWith("/api/stream") === true;
  }
  return STREAM_ROUTES.has(template);
}

function limitClass(request: GovernedRequest): ResourceLimitClass {
  if (isStream(request)) return "stream";
  return request.principal ? "authenticated" : "anonymous";
}

function requestIdentity(request: GovernedRequest, kind: ResourceLimitClass): string {
  if (request.principal) {
    return `${request.principal.tenantId.length}:${request.principal.tenantId}|` +
      `${request.principal.subject.length}:${request.principal.subject}`;
  }
  // Production accepts this value only from the private edge peer, which
  // overwrites X-Forwarded-For. Development uses the direct socket.
  const address = request.ip || "unknown-peer";
  return `${kind}:${address}`;
}

function cost(request: FastifyRequest): number {
  const template = routeTemplate(request);
  if (template === "/api/v1/harness/*") {
    const pathname = request.url.split("?", 1)[0] ?? "";
    if (pathname.endsWith("/api/resource")) return 8;
    if (pathname.endsWith("/api/stream")) return 4;
    return request.method === "POST" ? 2 : 1;
  }
  if (request.method === "POST" || request.method === "PATCH" || request.method === "DELETE") {
    return 2;
  }
  return 1;
}

function releaseWithResponse(lease: ResourceLease, reply: FastifyReply): void {
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    lease.release();
    reply.raw.removeListener("finish", release);
    reply.raw.removeListener("close", release);
    reply.raw.removeListener("error", release);
  };
  reply.raw.once("finish", release);
  reply.raw.once("close", release);
  reply.raw.once("error", release);
}

@Injectable()
export class ResourceGovernanceInterceptor implements NestInterceptor {
  constructor(
    @Inject(RESOURCE_GOVERNOR) private readonly governor: ResourceGovernor
  ) {}

  intercept(context: ExecutionContext, next: CallHandler): Observable<unknown> {
    if (context.getType() !== "http") return next.handle();
    const request = context.switchToHttp().getRequest<GovernedRequest>();
    const reply = context.switchToHttp().getResponse<FastifyReply>();
    const kind = limitClass(request);
    let lease: ResourceLease;
    try {
      lease = this.governor.acquire(
        kind,
        requestIdentity(request, kind),
        cost(request)
      );
    } catch (error) {
      if (!(error instanceof ResourceGovernorError)) throw error;
      reply.header("retry-after", String(error.retryAfterSeconds));
      reply.header("cache-control", "no-store");
      throw new HttpException(
        {
          statusCode: 429,
          code: error.code,
          error: "Request capacity is temporarily unavailable"
        },
        429
      );
    }
    if (kind === "stream") {
      releaseWithResponse(lease, reply);
      return next.handle();
    }
    return next.handle().pipe(finalize(() => lease.release()));
  }
}

export const resourceGovernanceInternals = {
  cost,
  isStream,
  limitClass,
  requestIdentity,
  routeTemplate
};
