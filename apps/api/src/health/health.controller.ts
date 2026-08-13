import {Controller, Get, Inject, Optional, ServiceUnavailableException} from "@nestjs/common";

import {PublicRoute} from "../auth/public.decorator";
import {AppConfigService} from "../config/app-config.service";
import {ARTIFACT_STORAGE, MODEL_PROVIDER, TASK_ORCHESTRATOR} from "../platform/tokens";
import type {ArtifactStoragePort} from "../platform/artifact-storage.port";
import type {TaskOrchestratorPort} from "../tasks/task-orchestrator.port";
import type {ModelProviderPort} from "../providers/model-provider.port";
import {HarnessWorkerPoolService} from "../harness/harness-worker-pool.service";
import {ContinuousReadinessService} from "./continuous-readiness.service";

@Controller()
export class HealthController {
  constructor(
    @Inject(AppConfigService) private readonly config: AppConfigService,
    @Inject(MODEL_PROVIDER) private readonly modelProvider: ModelProviderPort,
    @Inject(HarnessWorkerPoolService)
    private readonly harnessWorkers: HarnessWorkerPoolService,
    @Inject(ContinuousReadinessService)
    private readonly readiness: ContinuousReadinessService,
    @Optional() @Inject(TASK_ORCHESTRATOR) private readonly taskOrchestrator?: TaskOrchestratorPort,
    @Optional() @Inject(ARTIFACT_STORAGE) private readonly artifactStorage?: ArtifactStoragePort
  ) {}

  private taskQueueValue(): "postgres_durable_lease" | "memory_local_only" {
    return this.config.dataBackend === "postgres"
      ? "postgres_durable_lease"
      : "memory_local_only";
  }

  private artifactBackendValue(): "postgres_durable_force_rls" | "memory_local_only" {
    return this.config.dataBackend === "postgres"
      ? "postgres_durable_force_rls"
      : "memory_local_only";
  }

  @Get("health")
  @PublicRoute()
  async health() {
    const provider = await this.modelProvider.status();
    const harness = this.harnessWorkers.status();
    return {
      status: "ok",
      service: "teachlab-api",
      version: process.env.TEACHLAB_RELEASE_VERSION ?? "development",
      release_id: process.env.TEACHLAB_RELEASE_ID ?? "unsealed",
      timestamp: new Date().toISOString(),
      runtime: {
        auth: this.config.authMode,
        identityBoundary: "server_minted_http_only_session",
        sessionRevocationAuthority: this.config.dataBackend,
        sessionRevocationDurability:
          this.config.dataBackend === "postgres"
            ? "durable_force_rls"
            : "process_local_restart_invalidates_sessions",
        tenancy: "tenant_and_owner_scoped",
        sessionRepository: this.config.dataBackend,
        taskRepository: this.config.dataBackend,
        queue: this.taskQueueValue(),
        artifactStorage: this.artifactBackendValue(),
        eventTransport:
          this.config.dataBackend === "postgres" ? "postgres_polling_sse" : "memory_sse",
        deploymentBoundary:
          this.config.nodeEnv === "production"
            ? "postgres_required"
            : "explicit_local_only_defaults"
      },
      model: this.config.nodeEnv === "production"
        ? {
            provider: harness.backend,
            execution: "scoped_python_workers",
            credentialReferenceConfigured: harness.remoteProviderCredentialReferenceConfigured,
            workersWithCredentialLoaded: harness.remoteProviderCredentialsLoadedWorkers,
            readiness:
              harness.providerReadinessLastResult === "ready"
                ? "authenticated_content_free_models_probe_ready"
                : "authenticated_content_free_models_probe_unavailable",
            learnerContentSentByReadinessProbe:
              harness.providerReadinessLearnerContentSent,
            generationCreatedByReadinessProbe:
              harness.providerReadinessGenerationCreated
          }
        : provider,
      teachingHarness: harness
    };
  }

  @Get("ready")
  @PublicRoute()
  async ready() {
    try {
      const projection = await this.readiness.check();
      return {
        ...projection,
        service: "teachlab-api",
        startupChecks: {
          sessionRevocationAuthority:
            this.config.dataBackend === "postgres"
              ? "force_rls_schema_and_live_connection_verified"
              : "local_memory_only",
          taskQueue: projection.dependencies.taskQueue,
          artifactStorage: projection.dependencies.artifactStorage
        }
      };
    } catch {
      throw new ServiceUnavailableException({
        status: "not_ready",
        service: "teachlab-api",
        reason: "required_dependency_unavailable"
      });
    }
  }
}
