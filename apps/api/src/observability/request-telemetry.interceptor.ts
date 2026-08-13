import {
  CallHandler,
  ExecutionContext,
  Inject,
  Injectable,
  NestInterceptor
} from "@nestjs/common";
import type {FastifyReply, FastifyRequest} from "fastify";
import {catchError, finalize, Observable, throwError} from "rxjs";
import {randomUUID} from "node:crypto";

import {TELEMETRY} from "../platform/tokens";
import type {TelemetryPort} from "./telemetry.port";

const REQUEST_ID_PATTERN = /^[a-zA-Z0-9._:-]{1,128}$/;

@Injectable()
export class RequestTelemetryInterceptor implements NestInterceptor {
  constructor(@Inject(TELEMETRY) private readonly telemetry: TelemetryPort) {}

  intercept(context: ExecutionContext, next: CallHandler): Observable<unknown> {
    if (context.getType() !== "http") return next.handle();
    const request = context.switchToHttp().getRequest<FastifyRequest>();
    const reply = context.switchToHttp().getResponse<FastifyReply>();
    const suppliedRequestId = request.headers["x-request-id"];
    const candidate = Array.isArray(suppliedRequestId) ? suppliedRequestId[0] : suppliedRequestId;
    const requestId = candidate && REQUEST_ID_PATTERN.test(candidate) ? candidate : randomUUID();
    reply.header("x-request-id", requestId);
    reply.header("x-content-type-options", "nosniff");
    reply.header("content-security-policy", "default-src 'none'; frame-ancestors 'none'");
    reply.header("referrer-policy", "no-referrer");
    if (request.url.startsWith("/api/")) reply.header("cache-control", "no-store");

    const startedAt = performance.now();
    let failed = false;
    return next.handle().pipe(
      catchError((error: unknown) => {
        failed = true;
        return throwError(() => error);
      }),
      finalize(() => {
        this.telemetry.duration("http.server.request", performance.now() - startedAt, {
          method: request.method,
          route:
            typeof request.routeOptions?.url === "string"
              ? request.routeOptions.url
              : "unmatched",
          statusClass:
            Number.isInteger(reply.statusCode) && reply.statusCode >= 100 && reply.statusCode <= 599
              ? `${Math.floor(reply.statusCode / 100)}xx`
              : "other",
          failed
        });
      })
    );
  }
}
