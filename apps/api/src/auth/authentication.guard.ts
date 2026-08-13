import {
  CanActivate,
  ExecutionContext,
  ForbiddenException,
  Inject,
  Injectable,
  UnauthorizedException
} from "@nestjs/common";
import {Reflector} from "@nestjs/core";
import type {FastifyRequest} from "fastify";

import {AUTH_PROVIDER} from "../platform/tokens";
import type {AuthenticatedPrincipal, AuthProviderPort} from "./auth-provider.port";
import {IS_PUBLIC_ROUTE} from "./public.decorator";
import {SessionCookieService} from "./session-cookie";
import {AppConfigService} from "../config/app-config.service";
import {ALLOW_REVOKED_SESSION} from "./allow-revoked-session.decorator";
import {ALLOW_ACCOUNT_DELETION_SCOPE} from "../account/allow-account-deletion.decorator";
import {AccountDataRightsError} from "../account/account-data-rights.errors";
import type {AccountDataRightsRepositoryPort} from "../account/account-data-rights.repository.port";
import {ACCOUNT_DATA_RIGHTS_REPOSITORY, ACCOUNT_SCOPE_HASHER} from "../account/account-data-rights.tokens";
import {AccountScopeHasher} from "../account/account-scope-hash";
import {accessScopeFor} from "../tenancy/access-scope";
import {HttpException} from "@nestjs/common";

export type AuthenticatedFastifyRequest = FastifyRequest & {
  principal?: AuthenticatedPrincipal;
};

@Injectable()
export class AuthenticationGuard implements CanActivate {
  constructor(
    @Inject(Reflector) private readonly reflector: Reflector,
    @Inject(AUTH_PROVIDER) private readonly authProvider: AuthProviderPort,
    @Inject(SessionCookieService) private readonly sessionCookies: SessionCookieService,
    @Inject(AppConfigService) private readonly config: AppConfigService
    ,@Inject(ACCOUNT_DATA_RIGHTS_REPOSITORY)
    private readonly accountRights: AccountDataRightsRepositoryPort
    ,@Inject(ACCOUNT_SCOPE_HASHER) private readonly accountScopeHasher: AccountScopeHasher
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const isPublic = this.reflector.getAllAndOverride<boolean>(IS_PUBLIC_ROUTE, [
      context.getHandler(),
      context.getClass()
    ]);
    if (isPublic) return true;

    const allowRevokedSession = this.reflector.getAllAndOverride<boolean>(
      ALLOW_REVOKED_SESSION,
      [context.getHandler(), context.getClass()]
    ) === true;

    const request = context.switchToHttp().getRequest<AuthenticatedFastifyRequest>();
    const session = await this.authProvider.authenticate({
      headers: request.headers,
      method: request.method
    }, {allowRevokedSession});
    if (!session) {
      throw new UnauthorizedException("A valid TeachLab session is required");
    }
    if (!new Set(["GET", "HEAD", "OPTIONS"]).has(request.method.toUpperCase())) {
      const fetchSite = request.headers["sec-fetch-site"];
      const fetchSiteValue = Array.isArray(fetchSite) ? fetchSite[0] : fetchSite;
      const originHeader = request.headers.origin;
      const origin = Array.isArray(originHeader) ? originHeader[0] : originHeader;
      if (
        !new Set(["same-origin", "same-site"]).has(fetchSiteValue ?? "") ||
        !this.config.isTrustedOrigin(origin) ||
        !this.sessionCookies.csrfMatches(session, request.headers)
      ) {
        throw new ForbiddenException("CSRF validation failed");
      }
    }
    request.principal = session.principal;
    const allowDeleting = this.reflector.getAllAndOverride<boolean>(
      ALLOW_ACCOUNT_DELETION_SCOPE,
      [context.getHandler(), context.getClass()]
    ) === true;
    if (!allowDeleting) {
      try {
        const scope = accessScopeFor(session.principal);
        const lifecycle = await this.accountRights.lifecycle(
          scope,
          this.accountScopeHasher.hashes(scope)
        );
        if (lifecycle.kind === "deleting") {
          throw new AccountDataRightsError(409, "account_deletion_already_started");
        }
        if (lifecycle.kind === "deleted") {
          throw new AccountDataRightsError(410, "account_already_deleted");
        }
      } catch (error) {
        if (error instanceof AccountDataRightsError) {
          throw new HttpException(
            {statusCode: error.statusCode, code: error.code, error: error.code},
            error.statusCode
          );
        }
        throw error;
      }
    }
    return true;
  }
}
