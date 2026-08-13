import {
  CanActivate,
  ExecutionContext,
  GoneException,
  Inject,
  Injectable,
} from "@nestjs/common";
import type {FastifyRequest} from "fastify";

import {AppConfigService} from "../config/app-config.service";

const LEGACY_EXACT = new Set(["/api/bootstrap", "/api/step"]);
const LEGACY_PREFIXES = [
  "/api/events",
] as const;

export function legacySurfacePath(pathname: string): boolean {
  return LEGACY_EXACT.has(pathname)
    || LEGACY_PREFIXES.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
}

/**
 * Only the pre-v1 compatibility endpoints remain disabled in production.
 * Modern sessions/tasks/providers use the durable PostgreSQL repositories and
 * lease-backed dispatcher; the old bootstrap/step/events contract has no such
 * recovery semantics and is therefore not a production surface.
 */
@Injectable()
export class ProductionSurfaceGuard implements CanActivate {
  constructor(@Inject(AppConfigService) private readonly config: AppConfigService) {}

  canActivate(context: ExecutionContext): boolean {
    if (this.config.nodeEnv !== "production") return true;
    const request = context.switchToHttp().getRequest<FastifyRequest>();
    const pathname = request.url.split("?", 1)[0] ?? "";
    if (legacySurfacePath(pathname)) {
      throw new GoneException({
        statusCode: 410,
        code: "legacy_surface_disabled",
        error: "This non-durable compatibility surface is disabled in production",
      });
    }
    return true;
  }
}
