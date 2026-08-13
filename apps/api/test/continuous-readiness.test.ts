import assert from "node:assert/strict";
import {test} from "node:test";

import type {AppConfigService} from "../src/config/app-config.service";
import type {PostgresDatabase} from "../src/database/postgres-database";
import {ContinuousReadinessService} from "../src/health/continuous-readiness.service";
import type {
  HarnessWorkerPoolService,
  HarnessWorkerPoolStatus
} from "../src/harness/harness-worker-pool.service";


function workerStatus(overrides: Partial<HarnessWorkerPoolStatus> = {}): HarnessWorkerPoolStatus {
  return {
    enabled: true,
    safeguardingDispatcherConfigured: false,
    safeguardingStaffWorkflow: "cases_durable_dispatch_unavailable",
    safeguardingSupervisorRequired: false,
    safeguardingSupervisorRunning: false,
    safeguardingSupervisorLastResult: "not_required",
    safeguardingSupervisorScopesScanned: 0,
    safeguardingSupervisorStoresUnavailable: 0,
    safeguardingSupervisorPending: 0,
    safeguardingSupervisorOverdue: 0,
    safeguardingSupervisorOldestPendingAgeSeconds: 0,
    safeguardingSupervisorAttempted: 0,
    safeguardingSupervisorAccepted: 0,
    safeguardingSupervisorFailed: 0,
    safeguardingSupervisorAttemptedTotal: 0,
    safeguardingSupervisorAcceptedTotal: 0,
    safeguardingSupervisorFailedTotal: 0,
    safeguardingSupervisorLastSuccessAtUtc: null,
    safeguardingSupervisorReceiverReadinessRequired: false,
    safeguardingSupervisorReceiverReadinessStatus: "not_required",
    safeguardingSupervisorReceiverNetworkValidated: false,
    safeguardingSupervisorReceiverCredentialValidated: false,
    safeguardingSupervisorReceiverReadinessAttemptsTotal: 0,
    safeguardingSupervisorReceiverReadinessSuccessesTotal: 0,
    safeguardingSupervisorReceiverReadinessFailuresTotal: 0,
    safeguardingSupervisorLastReceiverSuccessAtUtc: null,
    safeguardingSupervisorRetentionEnabled: false,
    safeguardingSupervisorRetentionStatus: "disabled",
    safeguardingSupervisorRetentionPolicyVersionSha256: null,
    safeguardingSupervisorRetentionMinimumClosedAgeSeconds: null,
    safeguardingSupervisorRetentionMaximumCasesPerRun: null,
    safeguardingSupervisorRetentionEligibleCases: 0,
    safeguardingSupervisorRetentionCasesCompacted: 0,
    safeguardingSupervisorRetentionEventsCompacted: 0,
    safeguardingSupervisorRetentionFailures: 0,
    safeguardingSupervisorRetentionBlockedStores: 0,
    safeguardingSupervisorCapacityNearLimitStores: 0,
    safeguardingSupervisorCapacityEvents: 0,
    safeguardingSupervisorCapacityEventLimit: 0,
    safeguardingSupervisorCapacityEventHeadroomMin: null,
    safeguardingSupervisorCapacityStoreBytes: 0,
    safeguardingSupervisorCapacityStoreByteLimit: 0,
    safeguardingSupervisorCapacityStoreByteHeadroomMin: null,
    safeguardingSupervisorCapacityRecentErasureTombstones: 0,
    safeguardingSupervisorCapacityRecentErasureTombstoneLimit: 0,
    safeguardingSupervisorCapacityRecentErasureTombstoneHeadroomMin: null,
    safeguardingSupervisorRetentionCasesCompactedTotal: 0,
    safeguardingSupervisorRetentionEventsCompactedTotal: 0,
    safeguardingSupervisorErasureFenceInsertedCount: 0,
    safeguardingSupervisorErasureFenceEstimatedFalsePositiveUpperBound: 0,
    safeguardingSupervisorErasureFenceFalsePositiveTargetUpperBound: 1e-6,
    safeguardingSupervisorErasureFenceFalsePositiveWithinTarget: true,
    safeguardingSupervisorErasureFenceFalseNegativePossible: false,
    safeguardingSupervisorErasureFenceFalsePositivePolicy:
      "fail_closed_as_erased",
    safeguardingSupervisorUpdatedAtUtc: null,
    safeguardingSupervisorRawLearnerTextReadOrSent: false,
    safeguardingSupervisorScopeIdentityLabelsExposed: false,
    isolation: "one_logical_worker_scope_per_tenant_owner_same_os_uid_not_security_isolated",
    processResourceLimitsRequired: false,
    processResourceLimitsPolicy: "not_required_local_or_test",
    processResourceLimitsValidation:
      "exact_worker_and_runtime_canary_status_with_operational_minima",
    processResourceLimitsScope:
      "per_process_inherited_not_process_tree_or_cgroup",
    processCpuBoundary: "request_wall_clock_not_rlimit_cpu",
    processCountBoundary: "not_required_local_or_test_no_process_count_limit",
    scopeKeyVersion: "k1",
    keyRotation: "active_previous_atomic_rewrap_to_active",
    scopeDataKeyPolicy: "aead_wrapped_stable_scope_key",
    previousRootRetirement: "signed_hash_only_tombstone",
    migrationCrashRecovery: "copy_fsync_verify_atomic_rename",
    backend: "deterministic",
    remoteProviderCredentialReferenceConfigured: false,
    remoteProviderCredentialsLoadedWorkers: 0,
    runtimeCanaryRequired: false,
    runtimeCanaryLastResult: "not_required",
    runtimeCanaryAttempts: 0,
    runtimeCanarySuccesses: 0,
    runtimeCanaryFailures: 0,
    runtimeCanaryValidation:
      "bootstrap_policy_filesystem_process_limits_runtime_paths",
    runtimeCanaryRemoteProviderNetworkValidated: false,
    runtimeCanaryPersistentTenantDataCreated: false,
    providerReadinessRequired: false,
    providerReadinessLastResult: "not_required",
    providerReadinessAttempts: 0,
    providerReadinessSuccesses: 0,
    providerReadinessFailures: 0,
    providerReadinessValidation: "authenticated_content_free_models_endpoint",
    providerReadinessLearnerContentSent: false,
    providerReadinessGenerationCreated: false,
    providerReadinessPersistentTenantDataCreated: false,
    capacity: 16,
    readyWorkers: 1,
    startingWorkers: 0,
    stoppingWorkers: 0,
    activeRequests: 0,
    idleTimeoutMs: 900_000,
    idleEvictableWorkers: 0,
    idlePolicy: "capacity_triggered_expired_lru_only",
    activeOrStartingWorkersEvictable: false,
    durableScopeDataDeletedOnEviction: false,
    rawTenantMetadataStored: false,
    capabilityExposed: false,
    ...overrides
  };
}


function service(options: {
  database?: "postgres" | "not_configured" | "failure";
  workers?: Partial<HarnessWorkerPoolStatus>;
  workerFailure?: boolean;
  dataBackend?: "postgres" | "memory";
}) {
  let databaseCalls = 0;
  let workerCalls = 0;
  const database = {
    probe: async () => {
      databaseCalls += 1;
      if (options.database === "failure") throw new Error("private database detail");
      return options.database ?? "postgres";
    }
  } as unknown as PostgresDatabase;
  const workers = {
    assertReady: async () => {
      workerCalls += 1;
      if (options.workerFailure) throw new Error("/private/worker/path");
    },
    status: () => workerStatus(options.workers)
  } as unknown as HarnessWorkerPoolService;
  const taskOrchestrator = {
    probe: async () => options.dataBackend === "memory"
      ? "memory_local_only" as const
      : "postgres_durable_lease" as const
  };
  const artifactStorage = {
    probe: async () => options.dataBackend === "memory"
      ? "memory_local_only" as const
      : "postgres_durable_force_rls" as const
  };
  const value = new ContinuousReadinessService(
    {dataBackend: options.dataBackend ?? "postgres"} as AppConfigService,
    database,
    workers,
    taskOrchestrator as never,
    artifactStorage as never
  );
  return {value, calls: () => ({databaseCalls, workerCalls})};
}


test("continuous readiness coalesces and caches bounded live probes", async () => {
  const fixture = service({});
  const [left, right] = await Promise.all([fixture.value.check(), fixture.value.check()]);
  assert.deepEqual(left.dependencies, {
    database: "postgres_live",
    harnessGateway: "ready",
    workerCapacityState: "available_or_active",
    workerRuntimeCanary: "not_required_local_or_test",
    safeguardingSupervisor: "not_required",
    safeguardingRetention: "not_required_local_or_test",
    modelProvider: "not_required",
    taskQueue: "postgres_durable_lease",
    artifactStorage: "postgres_durable_force_rls"
  });
  assert.deepEqual(right, left);
  assert.deepEqual(fixture.calls(), {databaseCalls: 1, workerCalls: 1});
  left.dependencies.database = "local_memory_only";
  assert.equal((await fixture.value.check()).dependencies.database, "postgres_live");
});


test("dependency failures are sanitized and fail closed", async () => {
  for (const fixture of [
    service({database: "failure"}),
    service({workerFailure: true}),
    service({database: "not_configured", dataBackend: "postgres"}),
    service({workers: {readyWorkers: 0, stoppingWorkers: 16}}),
    service({
      workers: {runtimeCanaryRequired: true, runtimeCanaryLastResult: "failed"}
    }),
    service({
      workers: {
        safeguardingSupervisorRequired: true,
        safeguardingSupervisorRunning: true,
        safeguardingSupervisorLastResult: "overdue",
        safeguardingSupervisorPending: 1,
        safeguardingSupervisorOverdue: 1
      }
    }),
    service({
      workers: {
        safeguardingSupervisorRequired: true,
        safeguardingSupervisorRunning: true,
        safeguardingSupervisorLastResult: "ready",
        safeguardingSupervisorRetentionEnabled: false,
        safeguardingSupervisorRetentionStatus: "disabled"
      }
    }),
    service({
      workers: {
        providerReadinessRequired: true,
        providerReadinessLastResult: "failed"
      }
    })
  ]) {
    await assert.rejects(fixture.value.check(), /^Error: continuous readiness failed$/);
  }
});


test("explicit local memory mode remains honestly projected", async () => {
  const fixture = service({
    database: "not_configured",
    dataBackend: "memory",
    workers: {enabled: false, readyWorkers: 0, capacity: 16}
  });
  const result = await fixture.value.check();
  assert.equal(result.dependencies.database, "local_memory_only");
  assert.equal(result.dependencies.harnessGateway, "disabled");
  assert.equal(result.containsPrivatePaths, false);
  assert.equal(result.containsCredentials, false);
});
