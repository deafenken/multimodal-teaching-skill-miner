import {
  Controller,
  Delete,
  ForbiddenException,
  Get,
  Headers,
  HttpCode,
  Inject,
  Post,
  Res,
  UnauthorizedException
} from "@nestjs/common";
import type {FastifyReply} from "fastify";

import {AppConfigService} from "../config/app-config.service";
import type {AuthenticatedPrincipal} from "./auth-provider.port";
import {AllowRevokedSession} from "./allow-revoked-session.decorator";
import {CurrentPrincipal} from "./current-principal.decorator";
import {PublicRoute} from "./public.decorator";
import {SessionCookieService} from "./session-cookie";
import {SessionRevocationService} from "./session-revocation.service";

function headerValue(
  headers: Record<string, string | string[] | undefined>,
  name: string
): string | undefined {
  const value = headers[name];
  return Array.isArray(value) ? value[0] : value;
}

@Controller("api/v1/auth/session")
export class AuthSessionController {
  constructor(
    @Inject(AppConfigService) private readonly config: AppConfigService,
    @Inject(SessionCookieService) private readonly cookies: SessionCookieService,
    @Inject(SessionRevocationService)
    private readonly revocations: SessionRevocationService
  ) {}

  @Post()
  @PublicRoute()
  @HttpCode(201)
  async create(
    @Headers() headers: Record<string, string | string[] | undefined>,
    @Res({passthrough: true}) reply: FastifyReply
  ) {
    this.assertTrustedBrowserContext(headers);
    if (this.config.authMode !== "development") {
      throw new UnauthorizedException(
        "OIDC browser sessions must use the authorization code login flow"
      );
    }
    const identity =
      {
        subject: this.developmentHeader(headers, "x-dev-user-id") ??
          this.config.developmentUserId,
        tenantId: this.developmentHeader(headers, "x-dev-tenant-id") ??
          this.config.developmentTenantId,
        roles: this.config.developmentRoles
      };
    let minted;
    try {
      minted = this.cookies.mint({
        ...identity,
        provider: this.config.authMode === "development" ? "development" : "oidc"
      });
    } catch {
      throw new UnauthorizedException("Authenticated identity claims are invalid");
    }
    // Do not mint browser authority until the server-side revocation record is
    // durable. A store failure returns 503 without leaking a usable cookie.
    await this.revocations.register(minted.authenticatedSession);
    reply.header("cache-control", "no-store");
    reply.header("set-cookie", [
      this.cookies.serializeSessionCookie(minted.cookieValue),
      this.cookies.serializeCsrfCookie(minted.csrfToken)
    ]);
    return {
      authenticated: true,
      csrfToken: minted.csrfToken,
      expiresAt: minted.expiresAt,
      mode: this.config.authMode === "development" ? "local_only" : "oidc"
    };
  }

  @Get()
  current(@Res({passthrough: true}) reply: FastifyReply) {
    // AuthenticationGuard has already verified the private signed principal
    // and durable revocation authority.  The public status endpoint needs only
    // that boolean; issuer/tenant/subject/email/roles stay server-side.
    reply.header("cache-control", "no-store, max-age=0");
    return {authenticated: true};
  }

  @Delete()
  @AllowRevokedSession()
  @HttpCode(204)
  async clear(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Res({passthrough: true}) reply: FastifyReply
  ): Promise<void> {
    const result = await this.revocations.revokeCurrent(principal);
    if (result.kind === "missing") {
      // A signed cookie without its server record is never converted into a
      // successful logout claim. Keep the browser cookie so the caller sees
      // the failure and can retry after authority storage recovers.
      throw new UnauthorizedException("Authenticated session authority is missing");
    }
    reply.header("cache-control", "no-store");
    reply.header("set-cookie", [
      this.cookies.serializeSessionCookie("", 0),
      this.cookies.serializeCsrfCookie("", 0)
    ]);
  }

  private developmentHeader(
    headers: Record<string, string | string[] | undefined>,
    name: string
  ): string | undefined {
    if (!this.config.allowDevelopmentAuthHeaders) return undefined;
    return headerValue(headers, name)?.trim() || undefined;
  }

  private assertTrustedBrowserContext(
    headers: Record<string, string | string[] | undefined>
  ): void {
    const origin = headerValue(headers, "origin");
    const fetchSite = headerValue(headers, "sec-fetch-site");
    if (
      !this.config.isTrustedOrigin(origin)
      || !new Set(["same-origin", "same-site"]).has(fetchSite ?? "")
    ) {
      throw new ForbiddenException("Untrusted session exchange origin");
    }
  }
}
