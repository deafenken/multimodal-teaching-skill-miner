import {CanActivate, ExecutionContext, Inject, Injectable} from "@nestjs/common";

import type {AuthenticatedFastifyRequest} from "../auth/authentication.guard";
import {SessionsService} from "../sessions/sessions.service";
import {accessScopeFor} from "../tenancy/access-scope";

@Injectable()
export class EventsOwnershipGuard implements CanActivate {
  constructor(@Inject(SessionsService) private readonly sessions: SessionsService) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest<AuthenticatedFastifyRequest>();
    const principal = request.principal;
    if (!principal) return false;
    const parameters = request.params as {sessionId?: string} | undefined;
    const query = request.query as {session_id?: string} | undefined;
    const sessionId = parameters?.sessionId ?? query?.session_id;
    if (!sessionId) return true;
    await this.sessions.getOwned(accessScopeFor(principal), sessionId);
    return true;
  }
}
