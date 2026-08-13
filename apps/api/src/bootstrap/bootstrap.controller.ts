import {Controller, Get, Inject} from "@nestjs/common";

import {MODEL_PROVIDER} from "../platform/tokens";
import type {ModelProviderPort} from "../providers/model-provider.port";
import {HarnessWorkerPoolService} from "../harness/harness-worker-pool.service";
import type {AuthenticatedPrincipal} from "../auth/auth-provider.port";
import {CurrentPrincipal} from "../auth/current-principal.decorator";
import {AccountBrowserCacheScopeIssuer} from "../account/account-browser-cache-scope";
import {AppConfigService} from "../config/app-config.service";

@Controller()
export class BootstrapController {
  constructor(
    @Inject(MODEL_PROVIDER) private readonly modelProvider: ModelProviderPort,
    @Inject(HarnessWorkerPoolService)
    private readonly harnessWorkers: HarnessWorkerPoolService,
    @Inject(AccountBrowserCacheScopeIssuer)
    private readonly cacheScopes: AccountBrowserCacheScopeIssuer,
    @Inject(AppConfigService) private readonly config: AppConfigService
  ) {}

  @Get("api/bootstrap")
  async bootstrap(@CurrentPrincipal() principal: AuthenticatedPrincipal) {
    const providerStatus = await this.modelProvider.status();
    return {
      provider_status: providerStatus,
      teaching_harness_gateway: this.harnessWorkers.status(),
      ...(principal.provider === "oidc"
        ? {
            cache_scope: this.cacheScopes.issue(principal),
            account_data_rights: {
              mode: "authenticated_account_authority",
              export: "/api/v1/account/export",
              deletion_prepare: "/api/v1/account/deletion/prepare",
              deletion_confirm: "/api/v1/account/deletion/confirm",
              deletion_resume: "/api/v1/account/deletion/resume",
              deletion_status: "/api/v1/account/deletion/status",
              step_up: "/api/v1/auth/oidc/account-step-up",
              recent_auth_required: true,
              recent_auth_satisfied: this.recentAccountAuthority(principal),
              minimum_assurance_level: 2,
              remote_provider_copies_deleted: false
            }
          }
        : {
            account_data_rights: {
              mode: "local_only_no_account_authority",
              recent_auth_required: false,
              remote_provider_copies_deleted: false
            }
          }),
      interaction_contract: {
        sessions: "/api/v1/sessions",
        tasks: "/api/v1/sessions/:sessionId/tasks",
        task_status: "/api/v1/tasks/:taskId",
        events: "/api/v1/sessions/:sessionId/events",
        compatibility_events: "/api/events?session_id=:sessionId",
        teaching_harness: "/api/v1/harness/api/bootstrap",
        teaching_harness_stream: "/api/v1/harness/api/stream",
        teaching_harness_cancel: "/api/v1/harness/api/cancel",
        teaching_harness_route_prefix: "/api/v1/harness/",
        transport: "sse",
        steering_transport: "websocket_not_enabled",
        credentials: "server_minted_http_only_session_plus_csrf"
      },
      skills: [
        {
          skill_id: "socratic_verification",
          name: "苏格拉底理解核验",
          role: "primary"
        },
        {skill_id: "self_explanation", name: "自解释", role: "primary"},
        {skill_id: "stepwise_hint", name: "单步提示", role: "fallback"}
      ]
    };
  }

  private recentAccountAuthority(principal: AuthenticatedPrincipal): boolean {
    const authenticatedAt = Date.parse(principal.authenticatedAt ?? "");
    return principal.provider === "oidc"
      && principal.identityIssuer === this.config.oidcIssuer
      && Number.isFinite(authenticatedAt)
      && authenticatedAt <= Date.now() + 30_000
      && authenticatedAt >= Date.now() - this.config.oidcAccountStepUpMaxAgeSeconds * 1_000
      && (principal.assuranceLevel ?? 0) >= 2;
  }
}
