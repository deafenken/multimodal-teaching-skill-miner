import {Module} from "@nestjs/common";
import {APP_GUARD, APP_INTERCEPTOR} from "@nestjs/core";

import {AuthenticationGuard} from "./auth/authentication.guard";
import {TeacherEntitlementAuthorizationService} from "./auth/teacher-entitlement-authorization";
import {TeacherEntitlementHttpProvider} from "./auth/teacher-entitlement-http.provider";
import {AccountDataRightsController} from "./account/account-data-rights.controller";
import {AccountDataRightsService} from "./account/account-data-rights.service";
import {AccountBrowserCacheScopeIssuer} from "./account/account-browser-cache-scope";
import {AccountDeletionStatusCookieService} from "./account/account-deletion-status-cookie";
import {AccountScopeHasher} from "./account/account-scope-hash";
import {HarnessAccountRuntimeCoordinator} from "./account/harness-account-runtime-coordinator";
import {InMemoryAccountDataRightsRepository} from "./account/in-memory-account-data-rights.repository";
import {PostgresAccountDataRightsRepository} from "./account/postgres-account-data-rights.repository";
import {SessionAccountFreshAuthority} from "./account/session-account-fresh-authority";
import {
  ACCOUNT_DATA_RIGHTS_REPOSITORY,
  ACCOUNT_DELETION_FRESH_AUTHORITY,
  ACCOUNT_DELETION_STATUS_COOKIES,
  ACCOUNT_SCOPE_HASHER,
  ACCOUNT_SCOPE_RUNTIME_COORDINATOR,
  ACCOUNT_SYSTEM_DATABASE
} from "./account/account-data-rights.tokens";
import {AuthSessionController} from "./auth/auth-session.controller";
import {
  DevelopmentOnlyIdentityVerifier,
  OidcIdentityVerifier
} from "./auth/oidc-identity.verifier";
import {OidcAuthorizationController} from "./auth/oidc-authorization.controller";
import {OidcAuthorizationFlow} from "./auth/oidc-authorization-flow";
import {SessionCookieAuthProvider} from "./auth/session-cookie-auth.provider";
import {SessionCookieService} from "./auth/session-cookie";
import {InMemorySessionRevocationRepository} from "./auth/in-memory-session-revocation.repository";
import {PostgresSessionRevocationRepository} from "./auth/postgres-session-revocation.repository";
import {SessionRevocationService} from "./auth/session-revocation.service";
import {BootstrapController} from "./bootstrap/bootstrap.controller";
import {AppConfigService} from "./config/app-config.service";
import {PostgresDatabase} from "./database/postgres-database";
import {EventsController} from "./events/events.controller";
import {EventsService} from "./events/events.service";
import {EventsOwnershipGuard} from "./events/events-ownership.guard";
import {InMemoryEventStreamAdapter} from "./events/in-memory-event-stream.adapter";
import {PostgresEventStreamAdapter} from "./events/postgres-event-stream.adapter";
import {HealthController} from "./health/health.controller";
import {ContinuousReadinessService} from "./health/continuous-readiness.service";
import {HarnessGatewayController} from "./harness/harness-gateway.controller";
import {HarnessWorkerPoolService} from "./harness/harness-worker-pool.service";
import {LoggingTelemetryAdapter} from "./observability/logging-telemetry.adapter";
import {MetricsAccessService} from "./observability/metrics-access.service";
import {MetricsController} from "./observability/metrics.controller";
import {PrometheusTelemetryAdapter} from "./observability/prometheus-telemetry.adapter";
import {RequestTelemetryInterceptor} from "./observability/request-telemetry.interceptor";
import {ResourceGovernanceInterceptor, RESOURCE_GOVERNOR} from "./operations/resource-governance.interceptor";
import {DEFAULT_RESOURCE_GOVERNOR_POLICY, ResourceGovernor} from "./operations/resource-governor";
import {ProductionSurfaceGuard} from "./operations/production-surface.guard";
import {InMemoryArtifactStorageAdapter} from "./platform/in-memory-artifact-storage.adapter";
import {PostgresArtifactStorageAdapter} from "./platform/postgres-artifact-storage.adapter";
import {
  ARTIFACT_STORAGE,
  AUTH_PROVIDER,
  EVENT_STREAM,
  EXTERNAL_IDENTITY_VERIFIER,
  MODEL_PROVIDER,
  SESSION_REPOSITORY,
  SESSION_REVOCATION_REPOSITORY,
  TASK_REPOSITORY,
  TASK_SYSTEM_DATABASE,
  DURABLE_TASK_QUEUE,
  TASK_ORCHESTRATOR,
  TENANT_DATABASE,
  TELEMETRY
} from "./platform/tokens";
import {ProvidersController} from "./providers/providers.controller";
import {UnconfiguredAnthropicProvider} from "./providers/unconfigured-anthropic.provider";
import {InMemorySessionRepository} from "./sessions/in-memory-session.repository";
import {PostgresSessionRepository} from "./sessions/postgres-session.repository";
import {SessionsController} from "./sessions/sessions.controller";
import {SessionsService} from "./sessions/sessions.service";
import {InMemoryTaskOrchestratorAdapter} from "./tasks/in-memory-task-orchestrator.adapter";
import {PostgresTaskOrchestratorAdapter} from "./tasks/postgres-task-orchestrator.adapter";
import {InMemoryDurableTaskQueueAdapter} from "./tasks/in-memory-durable-task-queue.adapter";
import {InMemoryTaskRepository} from "./tasks/in-memory-task.repository";
import {PostgresTaskRepository} from "./tasks/postgres-task.repository";
import {TasksController} from "./tasks/tasks.controller";
import {TasksService} from "./tasks/tasks.service";

@Module({
  controllers: [
    HealthController,
    MetricsController,
    AuthSessionController,
    OidcAuthorizationController,
    BootstrapController,
    ProvidersController,
    SessionsController,
    TasksController,
    EventsController,
    HarnessGatewayController
    ,AccountDataRightsController
  ],
  providers: [
    AppConfigService,
    SessionCookieService,
    SessionRevocationService,
    PostgresDatabase,
    InMemorySessionRevocationRepository,
    PostgresSessionRevocationRepository,
    InMemorySessionRepository,
    PostgresSessionRepository,
    InMemoryTaskRepository,
    PostgresTaskRepository,
    InMemoryDurableTaskQueueAdapter,
    InMemoryTaskOrchestratorAdapter,
    PostgresTaskOrchestratorAdapter,
    InMemoryArtifactStorageAdapter,
    PostgresArtifactStorageAdapter,
    InMemoryEventStreamAdapter,
    PostgresEventStreamAdapter,
    LoggingTelemetryAdapter,
    PrometheusTelemetryAdapter,
    ContinuousReadinessService,
    HarnessWorkerPoolService,
    {
      provide: TeacherEntitlementAuthorizationService,
      inject: [AppConfigService],
      useFactory: (config: AppConfigService) => {
        if (
          !config.teacherEntitlementDirectoryUrl
          || !config.teacherEntitlementDirectoryBearerSecret
          || !config.teacherEntitlementBindingKey
          || !config.teacherEntitlementReceiptKey
        ) {
          // Local-only runtimes may omit the organization directory. The
          // optional controller dependency then leaves privileged mutations
          // unavailable; production startup requires every value below.
          return undefined;
        }
        const provider = new TeacherEntitlementHttpProvider({
          endpoint: config.teacherEntitlementDirectoryUrl,
          bearerSecret: config.teacherEntitlementDirectoryBearerSecret,
          timeoutMs: config.teacherEntitlementProviderTimeoutMs,
          maxResponseBytes: config.teacherEntitlementMaxResponseBytes,
          maxClockSkewMs: config.teacherEntitlementMaxClockSkewMs
        });
        return new TeacherEntitlementAuthorizationService(
          provider,
          {
            policyId: config.teacherEntitlementPolicyId,
            version: config.teacherEntitlementPolicyVersion,
            requiredRoles: {
              "api/resource/review": config.teacherAuthorityRoles,
              "api/curriculum/review": config.teacherAuthorityRoles,
              "api/curriculum/seal": config.teacherAuthorityRoles,
              "api/curriculum/revoke": config.teacherAuthorityRoles,
              "api/adjudication/claim": config.teacherAuthorityRoles,
              "api/adjudication/decide": config.teacherAuthorityRoles,
              "api/safeguarding/list": config.safeguardingAuthorityRoles,
              "api/safeguarding/dispatch": config.safeguardingAuthorityRoles,
              "api/safeguarding/case/acknowledge": config.safeguardingAuthorityRoles,
              "api/safeguarding/case/close": config.safeguardingAuthorityRoles,
              "api/safeguarding/escalation/overdue": config.safeguardingAuthorityRoles,
              "api/safeguarding/escalation/acknowledge": config.safeguardingAuthorityRoles
            },
            freshnessTtlMs: config.teacherEntitlementFreshnessTtlMs,
            cacheTtlMs: config.teacherEntitlementCacheTtlMs,
            providerTimeoutMs: config.teacherEntitlementProviderTimeoutMs,
            maxClockSkewMs: config.teacherEntitlementMaxClockSkewMs,
            maxCacheEntries: config.teacherEntitlementMaxCacheEntries,
            minAssuranceLevel: config.teacherEntitlementMinAssuranceLevel
          },
          Buffer.from(config.teacherEntitlementBindingKey, "utf8"),
          Buffer.from(config.teacherEntitlementReceiptKey, "utf8")
        );
      }
    },
    AccountDataRightsService,
    HarnessAccountRuntimeCoordinator,
    PostgresAccountDataRightsRepository,
    {provide: ACCOUNT_SYSTEM_DATABASE, useExisting: PostgresDatabase},
    {
      provide: ACCOUNT_SCOPE_HASHER,
      inject: [AppConfigService],
      useFactory: (config: AppConfigService) => new AccountScopeHasher(
        config.accountScopeKeys[0]!.secret,
        config.accountScopeKeys.slice(1).map((key) => key.secret)
      )
    },
    {
      provide: ACCOUNT_DELETION_STATUS_COOKIES,
      inject: [AppConfigService],
      useFactory: (config: AppConfigService) => new AccountDeletionStatusCookieService({
        secret: config.accountDeletionStatusSecret,
        secure: config.secureSessionCookies
      })
    },
    {
      provide: ACCOUNT_DELETION_FRESH_AUTHORITY,
      inject: [AppConfigService],
      useFactory: (config: AppConfigService) => new SessionAccountFreshAuthority(
        config.oidcIssuer ?? "https://disabled.invalid",
        config.accountScopeKeys,
        config.oidcAccountStepUpMaxAgeSeconds * 1_000
      )
    },
    {
      provide: AccountBrowserCacheScopeIssuer,
      inject: [AppConfigService],
      useFactory: (config: AppConfigService) => new AccountBrowserCacheScopeIssuer(
        config.accountCacheScopeSecret,
        config.oidcIssuer ?? "https://disabled.invalid",
        config.accountCacheScopeEpoch
      )
    },
    {provide: ACCOUNT_SCOPE_RUNTIME_COORDINATOR, useExisting: HarnessAccountRuntimeCoordinator},
    {
      provide: ACCOUNT_DATA_RIGHTS_REPOSITORY,
      inject: [AppConfigService, PostgresAccountDataRightsRepository],
      useFactory: (
        config: AppConfigService,
        postgres: PostgresAccountDataRightsRepository
      ) => config.dataBackend === "postgres"
        ? postgres
        : new InMemoryAccountDataRightsRepository()
    },
    {
      provide: AUTH_PROVIDER,
      inject: [SessionCookieService, SessionRevocationService],
      useFactory: (
        sessions: SessionCookieService,
        revocations: SessionRevocationService
      ) => new SessionCookieAuthProvider(sessions, revocations)
    },
    {
      provide: EXTERNAL_IDENTITY_VERIFIER,
      inject: [AppConfigService],
      useFactory: (config: AppConfigService) =>
        config.authMode === "oidc"
          ? new OidcIdentityVerifier(config)
          : new DevelopmentOnlyIdentityVerifier()
    },
    {
      provide: OidcAuthorizationFlow,
      inject: [AppConfigService, EXTERNAL_IDENTITY_VERIFIER],
      useFactory: (
        config: AppConfigService,
        verifier: OidcIdentityVerifier | DevelopmentOnlyIdentityVerifier
      ) => new OidcAuthorizationFlow({
        issuer: config.oidcIssuer ?? "https://disabled.invalid",
        clientId: config.oidcClientId ?? "disabled",
        clientSecret: config.oidcClientSecret ?? "disabled",
        redirectUri: config.oidcRedirectUri ?? "https://disabled.invalid/api/teacher-agent/security/login/callback",
        transactionSecret: config.oidcTransactionSecret ?? "development-disabled-transaction-secret",
        discoveryTimeoutMs: config.oidcDiscoveryTimeoutMs,
        transactionTtlSeconds: config.oidcTransactionTtlSeconds,
        authenticationMaxAgeSeconds: config.oidcAuthenticationMaxAgeSeconds,
        accountStepUpMaxAgeSeconds: config.oidcAccountStepUpMaxAgeSeconds,
        accountAal2AcrValues: config.oidcAccountAal2AcrValues
      }, {
        verifyIdToken: (token, nonce, now, requirements) => {
          if (!(verifier instanceof OidcIdentityVerifier)) {
            return Promise.reject(new Error("OIDC login is disabled"));
          }
          return verifier.verifyIdToken(token, nonce, now, requirements);
        }
      })
    },
    {provide: TENANT_DATABASE, useExisting: PostgresDatabase},
    {
      provide: SESSION_REVOCATION_REPOSITORY,
      inject: [
        AppConfigService,
        InMemorySessionRevocationRepository,
        PostgresSessionRevocationRepository
      ],
      useFactory: (
        config: AppConfigService,
        memory: InMemorySessionRevocationRepository,
        postgres: PostgresSessionRevocationRepository
      ) => (config.dataBackend === "postgres" ? postgres : memory)
    },
    {
      provide: SESSION_REPOSITORY,
      inject: [AppConfigService, InMemorySessionRepository, PostgresSessionRepository],
      useFactory: (
        config: AppConfigService,
        memory: InMemorySessionRepository,
        postgres: PostgresSessionRepository
      ) => (config.dataBackend === "postgres" ? postgres : memory)
    },
    {
      provide: ARTIFACT_STORAGE,
      inject: [AppConfigService, InMemoryArtifactStorageAdapter, PostgresArtifactStorageAdapter],
      useFactory: (
        config: AppConfigService,
        memory: InMemoryArtifactStorageAdapter,
        postgres: PostgresArtifactStorageAdapter
      ) => config.dataBackend === "postgres" ? postgres : memory
    },
    {
      provide: EVENT_STREAM,
      inject: [AppConfigService, InMemoryEventStreamAdapter, PostgresEventStreamAdapter],
      useFactory: (
        config: AppConfigService,
        memory: InMemoryEventStreamAdapter,
        postgres: PostgresEventStreamAdapter
      ) => (config.dataBackend === "postgres" ? postgres : memory)
    },
    {
      provide: TASK_REPOSITORY,
      inject: [AppConfigService, InMemoryTaskRepository, PostgresTaskRepository],
      useFactory: (
        config: AppConfigService,
        memory: InMemoryTaskRepository,
        postgres: PostgresTaskRepository
      ) => (config.dataBackend === "postgres" ? postgres : memory)
    },
    {provide: TASK_SYSTEM_DATABASE, useExisting: PostgresDatabase},
    {
      provide: DURABLE_TASK_QUEUE,
      inject: [AppConfigService, PostgresTaskRepository, InMemoryDurableTaskQueueAdapter],
      useFactory: (
        config: AppConfigService,
        postgres: PostgresTaskRepository,
        memory: InMemoryDurableTaskQueueAdapter
      ) => config.dataBackend === "postgres" ? postgres : memory
    },
    {
      provide: MetricsAccessService,
      inject: [AppConfigService],
      useFactory: (config: AppConfigService) => {
        const access = new MetricsAccessService();
        access.configureToken(config.metricsAccessToken);
        return access;
      }
    },
    {provide: TELEMETRY, useExisting: PrometheusTelemetryAdapter},
    {
      provide: RESOURCE_GOVERNOR,
      useFactory: () => new ResourceGovernor(DEFAULT_RESOURCE_GOVERNOR_POLICY)
    },
    {
      provide: MODEL_PROVIDER,
      inject: [AppConfigService],
      useFactory: (config: AppConfigService) => new UnconfiguredAnthropicProvider(config)
    },
    {
      provide: TASK_ORCHESTRATOR,
      inject: [
        AppConfigService,
        InMemoryTaskOrchestratorAdapter,
        PostgresTaskOrchestratorAdapter
      ],
      useFactory: (
        config: AppConfigService,
        memory: InMemoryTaskOrchestratorAdapter,
        postgres: PostgresTaskOrchestratorAdapter
      ) => config.dataBackend === "postgres" ? postgres : memory
    },
    SessionsService,
    TasksService,
    EventsService,
    EventsOwnershipGuard,
    {provide: APP_GUARD, useClass: AuthenticationGuard},
    {provide: APP_GUARD, useClass: ProductionSurfaceGuard},
    {provide: APP_INTERCEPTOR, useClass: ResourceGovernanceInterceptor},
    {provide: APP_INTERCEPTOR, useClass: RequestTelemetryInterceptor}
  ]
})
export class AppModule {}
