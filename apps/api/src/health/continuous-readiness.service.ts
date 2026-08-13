import {Inject, Injectable, Optional} from "@nestjs/common";

import {AppConfigService} from "../config/app-config.service";
import {PostgresDatabase} from "../database/postgres-database";
import {HarnessWorkerPoolService} from "../harness/harness-worker-pool.service";
import {TASK_ORCHESTRATOR, ARTIFACT_STORAGE} from "../platform/tokens";
import type {TaskOrchestratorPort} from "../tasks/task-orchestrator.port";
import type {ArtifactStoragePort} from "../platform/artifact-storage.port";

export interface ContinuousReadinessProjection {
  status: "ready";
  checkedAt: string;
  dependencies: {
    database: "postgres_live" | "local_memory_only";
    harnessGateway: "ready" | "disabled";
    workerCapacityState: "available_or_active" | "disabled";
    workerRuntimeCanary:
      | "bootstrap_policy_filesystem_process_limits_runtime_paths_verified_no_provider_network"
      | "not_required_local_or_test"
      | "disabled";
    safeguardingSupervisor:
      | "content_free_delivery_ready"
      | "not_required"
      | "disabled";
    safeguardingRetention:
      | "closed_case_retention_capacity_and_erasure_fence_ready"
      | "not_required_local_or_test"
      | "disabled";
    modelProvider:
      | "authenticated_content_free_models_probe_ready"
      | "not_required"
      | "disabled";
    taskQueue: "postgres_durable_lease" | "memory_local_only";
    artifactStorage: "postgres_durable_force_rls" | "memory_local_only";
  };
  containsPrivatePaths: false;
  containsCredentials: false;
}

interface CachedReadiness {
  value?: ContinuousReadinessProjection;
  error?: Error;
  expiresAtMs: number;
}

@Injectable()
export class ContinuousReadinessService {
  private cached?: CachedReadiness;
  private pending?: Promise<ContinuousReadinessProjection>;
  private attempts = 0;
  private successes = 0;
  private failures = 0;
  private lastResult: "never" | "ready" | "failed" = "never";

  constructor(
    @Inject(AppConfigService) private readonly config: AppConfigService,
    @Inject(PostgresDatabase) private readonly database: PostgresDatabase,
    @Inject(HarnessWorkerPoolService)
    private readonly harnessWorkers: HarnessWorkerPoolService,
    @Optional() @Inject(TASK_ORCHESTRATOR)
    private readonly taskOrchestrator?: TaskOrchestratorPort,
    @Optional() @Inject(ARTIFACT_STORAGE)
    private readonly artifactStorage?: ArtifactStoragePort
  ) {}

  async check(): Promise<ContinuousReadinessProjection> {
    const now = Date.now();
    if (this.cached && this.cached.expiresAtMs > now) {
      if (this.cached.error) throw new Error("continuous readiness failed");
      if (this.cached.value) return structuredClone(this.cached.value);
    }
    this.pending ??= this.runCheck().finally(() => {
      this.pending = undefined;
    });
    return structuredClone(await this.pending);
  }

  private async runCheck(): Promise<ContinuousReadinessProjection> {
    this.attempts += 1;
    try {
      const database = await this.database.probe(2_000);
      if (
        (this.config.dataBackend === "postgres") !== (database === "postgres")
      ) {
        throw new Error("Database readiness does not match configured authority");
      }
      if (this.config.dataBackend === "postgres" && (!this.taskOrchestrator || !this.artifactStorage)) {
        throw new Error("Durable queue and artifact storage providers are unavailable");
      }
      const taskQueue = this.taskOrchestrator
        ? await this.taskOrchestrator.probe()
        : "memory_local_only" as const;
      const artifactStorage = this.artifactStorage
        ? await this.artifactStorage.probe()
        : "memory_local_only" as const;
      if (this.config.dataBackend === "postgres") {
        if (taskQueue !== "postgres_durable_lease") {
          throw new Error("Durable task queue readiness is unavailable");
        }
        if (artifactStorage !== "postgres_durable_force_rls") {
          throw new Error("Durable artifact storage readiness is unavailable");
        }
      } else if (
        taskQueue !== "memory_local_only"
        || artifactStorage !== "memory_local_only"
      ) {
        throw new Error("Local-only readiness projection is inconsistent");
      }
      await this.harnessWorkers.assertReady();
      const workers = this.harnessWorkers.status();
      if (
        workers.enabled &&
        workers.readyWorkers === 0 &&
        workers.startingWorkers === 0 &&
        workers.stoppingWorkers >= workers.capacity
      ) {
        throw new Error("Harness worker capacity is unavailable");
      }
      if (
        workers.enabled &&
        workers.runtimeCanaryRequired &&
        workers.runtimeCanaryLastResult !== "ready"
      ) {
        throw new Error("Harness worker runtime canary is unavailable");
      }
      if (
        workers.enabled &&
        workers.safeguardingSupervisorRequired &&
        (
          !workers.safeguardingSupervisorRunning ||
          workers.safeguardingSupervisorLastResult !== "ready"
        )
      ) {
        throw new Error("Safeguarding delivery supervisor is unavailable");
      }
      if (
        workers.enabled &&
        workers.safeguardingSupervisorRequired &&
        (
          !workers.safeguardingSupervisorRetentionEnabled ||
          workers.safeguardingSupervisorRetentionStatus === "disabled" ||
          workers.safeguardingSupervisorRetentionFailures > 0 ||
          workers.safeguardingSupervisorRetentionBlockedStores > 0 ||
          workers.safeguardingSupervisorCapacityNearLimitStores > 0 ||
          !workers.safeguardingSupervisorErasureFenceFalsePositiveWithinTarget ||
          workers.safeguardingSupervisorErasureFenceFalseNegativePossible
        )
      ) {
        throw new Error("Safeguarding retention capacity is unavailable");
      }
      if (
        workers.enabled &&
        workers.providerReadinessRequired &&
        workers.providerReadinessLastResult !== "ready"
      ) {
        throw new Error("Model provider readiness is unavailable");
      }
      const value: ContinuousReadinessProjection = {
        status: "ready",
        checkedAt: new Date().toISOString(),
        dependencies: {
          database:
            database === "postgres" ? "postgres_live" : "local_memory_only",
          harnessGateway: workers.enabled ? "ready" : "disabled",
          workerCapacityState: workers.enabled ? "available_or_active" : "disabled",
          workerRuntimeCanary: workers.enabled
            ? workers.runtimeCanaryRequired
              ? "bootstrap_policy_filesystem_process_limits_runtime_paths_verified_no_provider_network"
              : "not_required_local_or_test"
            : "disabled",
          safeguardingSupervisor: workers.enabled
            ? workers.safeguardingSupervisorRequired
              ? "content_free_delivery_ready"
              : "not_required"
            : "disabled",
          safeguardingRetention: workers.enabled
            ? workers.safeguardingSupervisorRequired
              ? "closed_case_retention_capacity_and_erasure_fence_ready"
              : "not_required_local_or_test"
            : "disabled",
          modelProvider: workers.enabled
            ? workers.providerReadinessRequired
              ? "authenticated_content_free_models_probe_ready"
              : "not_required"
            : "disabled",
          taskQueue,
          artifactStorage
        },
        containsPrivatePaths: false,
        containsCredentials: false
      };
      this.cached = {value, expiresAtMs: Date.now() + 1_000};
      this.successes += 1;
      this.lastResult = "ready";
      return value;
    } catch {
      this.failures += 1;
      this.lastResult = "failed";
      const error = new Error("continuous readiness failed");
      this.cached = {error, expiresAtMs: Date.now() + 250};
      throw error;
    }
  }

  operationalSnapshot(): {
    attempts: number;
    successes: number;
    failures: number;
    lastResult: "never" | "ready" | "failed";
  } {
    return {
      attempts: this.attempts,
      successes: this.successes,
      failures: this.failures,
      lastResult: this.lastResult,
    };
  }
}
