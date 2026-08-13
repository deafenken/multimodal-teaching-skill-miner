import type {
  AuthenticationOptions,
  AuthenticationRequest,
  AuthProviderPort
} from "./auth-provider.port";
import {SessionCookieService} from "./session-cookie";
import {SessionRevocationService} from "./session-revocation.service";

export class SessionCookieAuthProvider implements AuthProviderPort {
  constructor(
    private readonly sessions: SessionCookieService,
    private readonly revocations: SessionRevocationService
  ) {}

  async authenticate(request: AuthenticationRequest, options: AuthenticationOptions = {}) {
    const session = this.sessions.authenticate(request.headers);
    if (!session) return null;
    const state = await this.revocations.inspect(session);
    if (state.kind === "active" || options.allowRevokedSession && state.kind === "revoked") {
      return session;
    }
    return null;
  }
}
