import {createHash, createHmac, randomBytes} from "node:crypto";
import {constants as fsConstants} from "node:fs";
import {access, chmod, lstat, mkdir, mkdtemp, readFile, rm} from "node:fs/promises";
import {request as httpRequest, type IncomingMessage} from "node:http";
import {basename, dirname, join, resolve} from "node:path";
import {
  spawn,
  type ChildProcessWithoutNullStreams
} from "node:child_process";

import {Inject, Injectable, OnApplicationShutdown} from "@nestjs/common";

import {
  AppConfigService,
  type HarnessScopeKey,
  type HarnessWorkerProcessResourceLimits
} from "../config/app-config.service";
import type {AccessScope} from "../tenancy/access-scope";
import {
  remotePolicySha256,
  type RemoteSubjectPolicy
} from "../auth/remote-processing-policy";
import {
  TEACHER_AUTHORITY_PATHS,
  requestContainsAuthorityOverride,
  teacherAuthorityEnvelope
} from "./teacher-authority";
import {
  ensureActiveScopeRoot,
  type HarnessAuthorityScopeBinding
} from "./harness-scope-key-migration";
import {issueSafeguardingRouteLocator} from "./safeguarding-route-locator";

const BOOTSTRAP_SCHEMA = "teaching_skill_miner.gateway_worker_bootstrap.v1";
const STATUS_SCHEMA = "teaching_skill_miner.gateway_worker_status.v1";
const SAFEGUARDING_SUPERVISOR_BOOTSTRAP_SCHEMA =
  "teaching_skill_miner.safeguarding_supervisor_bootstrap.v2";
const SAFEGUARDING_SUPERVISOR_STATUS_SCHEMA =
  "teaching_skill_miner.safeguarding_supervisor_status.v2";
const MAX_STATUS_BYTES = 4 * 1024;
const MAX_SUPERVISOR_STDOUT_BUFFER_BYTES = 8 * 1024;
const MAX_STDERR_BYTES = 64 * 1024;
const SAFEGUARDING_SUPERVISOR_POLL_SECONDS = 1;
const SAFEGUARDING_SUPERVISOR_STALE_MAX_MS = 5_000;
const RUNTIME_CANARY_SUCCESS_TTL_MS = 30_000;
const RUNTIME_CANARY_FAILURE_TTL_MS = 1_000;
const RUNTIME_CANARY_STARTUP_MAX_MS = 5_000;
const RUNTIME_CANARY_SHUTDOWN_MAX_MS = 2_000;
const PROVIDER_READINESS_SUCCESS_TTL_MS = 30_000;
const PROVIDER_READINESS_FAILURE_TTL_MS = 1_000;
const PROVIDER_READINESS_STARTUP_MAX_MS = 10_000;
const PROVIDER_READINESS_SHUTDOWN_MAX_MS = 2_000;

interface ScopeIdentity {
  key: string;
  scopeId: string;
  keyVersion: string;
  privateRoot: string;
  dataKey: Buffer;
  learnerScopeId?: string;
  authorityScopeBindings?: readonly HarnessAuthorityScopeBinding[];
  safeguardingRouteLocator?: string;
}

interface WorkerRecord {
  identityKey: string;
  capability: string;
  port: number;
  child: ChildProcessWithoutNullStreams;
  requestNamespaceKey: Buffer;
  teacherAuthorityKey: Buffer;
  scopeId: string;
  scopeKeyVersion: string;
  activeRequests: number;
  lastActivityMs: number;
  stopping: boolean;
  remoteSubjectPolicyHash: string;
}

interface WorkerBootstrap {
  schema: typeof BOOTSTRAP_SCHEMA;
  scope_id: string;
  scope_key_version: string;
  worker_id: string;
  capability_token: string;
  private_root: string;
  scope_key_material: string;
  learner_scope_id: string;
  authority_scope_bindings: readonly {
    scope_id: string;
    key_version: string;
  }[];
  agent_backend: "deterministic" | "deepseek";
  api_key_file: string | null;
  remote_provider_policy: AppConfigService["harnessRemoteProviderPolicy"] | null;
  remote_subject_policy: RemoteSubjectPolicy;
  safeguarding_locale: string;
  safeguarding_dispatcher: {
    endpoint: string;
    bearer_secret: string;
    route_locator: string;
    policy_version: string;
    timeout_ms: number;
    maximum_response_bytes: number;
  } | null;
  filesystem_isolation_required: boolean;
  process_resource_limits: HarnessWorkerProcessResourceLimits | null;
}

interface SafeguardingSupervisorBootstrap {
  schema: typeof SAFEGUARDING_SUPERVISOR_BOOTSTRAP_SCHEMA;
  root: string;
  endpoint: string;
  bearer_secret: string;
  policy_version: string;
  timeout_ms: number;
  maximum_response_bytes: number;
  poll_seconds: number;
  retention_policy_version: string;
  retention_minimum_closed_age_seconds: number;
  retention_maximum_cases_per_run: number;
  retention_authority_secret: string;
  retention_deployment_context_sha256: string;
}

interface SafeguardingSupervisorAggregate {
  status: "ready" | "degraded";
  scopesScanned: number;
  storesUnavailable: number;
  pending: number;
  overdue: number;
  oldestPendingAgeSeconds: number;
  attempted: number;
  accepted: number;
  failed: number;
  attemptedTotal: number;
  acceptedTotal: number;
  failedTotal: number;
  lastSuccessAtUtc: string | null;
  receiverReadinessRequired: boolean;
  receiverReadinessStatus: "ready" | "failed";
  receiverNetworkValidated: boolean;
  receiverCredentialValidated: boolean;
  receiverReadinessAttemptsTotal: number;
  receiverReadinessSuccessesTotal: number;
  receiverReadinessFailuresTotal: number;
  lastReceiverSuccessAtUtc: string | null;
  retentionEnabled: boolean;
  retentionStatus: "disabled" | "idle" | "compacted" | "blocked" | "failed";
  retentionPolicyVersionSha256: string | null;
  retentionMinimumClosedAgeSeconds: number | null;
  retentionMaximumCasesPerRun: number | null;
  retentionEligibleCases: number;
  retentionCasesCompacted: number;
  retentionEventsCompacted: number;
  retentionFailures: number;
  retentionBlockedStores: number;
  capacityNearLimitStores: number;
  capacityEvents: number;
  capacityEventLimit: number;
  capacityEventHeadroomMin: number | null;
  capacityStoreBytes: number;
  capacityStoreByteLimit: number;
  capacityStoreByteHeadroomMin: number | null;
  capacityRecentErasureTombstones: number;
  capacityRecentErasureTombstoneLimit: number;
  capacityRecentErasureTombstoneHeadroomMin: number | null;
  retentionCasesCompactedTotal: number;
  retentionEventsCompactedTotal: number;
  erasureFenceInsertedCount: number;
  erasureFenceEstimatedFalsePositiveUpperBound: number;
  erasureFenceFalsePositiveTargetUpperBound: number;
  erasureFenceFalsePositiveWithinTarget: boolean;
  erasureFenceFalseNegativePossible: false;
  erasureFenceFalsePositivePolicy: "fail_closed_as_erased";
  updatedAtUtc: string;
}

interface SafeguardingSupervisorRecord {
  child: ChildProcessWithoutNullStreams;
  stdout: Buffer;
  stderrBytes: number;
  stopping: boolean;
  protocolFailed: boolean;
  abortStartup?: () => void;
}

interface EphemeralWorkerBootstrap {
  root: string;
  bootstrap: WorkerBootstrap;
}

interface ProviderReadinessOverride {
  run(): Promise<void>;
}

export type SafeguardingSupervisorResult =
  | "not_required"
  | "starting"
  | "ready"
  | "degraded"
  | "overdue"
  | "stale"
  | "failed"
  | "unavailable";

export interface HarnessWorkerPoolStatus {
  enabled: boolean;
  safeguardingDispatcherConfigured: boolean;
  safeguardingStaffWorkflow:
    | "content_free_dispatch_configured"
    | "cases_durable_dispatch_unavailable";
  safeguardingSupervisorRequired: boolean;
  safeguardingSupervisorRunning: boolean;
  safeguardingSupervisorLastResult: SafeguardingSupervisorResult;
  safeguardingSupervisorScopesScanned: number;
  safeguardingSupervisorStoresUnavailable: number;
  safeguardingSupervisorPending: number;
  safeguardingSupervisorOverdue: number;
  safeguardingSupervisorOldestPendingAgeSeconds: number;
  safeguardingSupervisorAttempted: number;
  safeguardingSupervisorAccepted: number;
  safeguardingSupervisorFailed: number;
  safeguardingSupervisorAttemptedTotal: number;
  safeguardingSupervisorAcceptedTotal: number;
  safeguardingSupervisorFailedTotal: number;
  safeguardingSupervisorLastSuccessAtUtc: string | null;
  safeguardingSupervisorReceiverReadinessRequired: boolean;
  safeguardingSupervisorReceiverReadinessStatus:
    | "not_required"
    | "ready"
    | "failed";
  safeguardingSupervisorReceiverNetworkValidated: boolean;
  safeguardingSupervisorReceiverCredentialValidated: boolean;
  safeguardingSupervisorReceiverReadinessAttemptsTotal: number;
  safeguardingSupervisorReceiverReadinessSuccessesTotal: number;
  safeguardingSupervisorReceiverReadinessFailuresTotal: number;
  safeguardingSupervisorLastReceiverSuccessAtUtc: string | null;
  safeguardingSupervisorRetentionEnabled: boolean;
  safeguardingSupervisorRetentionStatus:
    | "disabled"
    | "idle"
    | "compacted"
    | "blocked"
    | "failed";
  safeguardingSupervisorRetentionPolicyVersionSha256: string | null;
  safeguardingSupervisorRetentionMinimumClosedAgeSeconds: number | null;
  safeguardingSupervisorRetentionMaximumCasesPerRun: number | null;
  safeguardingSupervisorRetentionEligibleCases: number;
  safeguardingSupervisorRetentionCasesCompacted: number;
  safeguardingSupervisorRetentionEventsCompacted: number;
  safeguardingSupervisorRetentionFailures: number;
  safeguardingSupervisorRetentionBlockedStores: number;
  safeguardingSupervisorCapacityNearLimitStores: number;
  safeguardingSupervisorCapacityEvents: number;
  safeguardingSupervisorCapacityEventLimit: number;
  safeguardingSupervisorCapacityEventHeadroomMin: number | null;
  safeguardingSupervisorCapacityStoreBytes: number;
  safeguardingSupervisorCapacityStoreByteLimit: number;
  safeguardingSupervisorCapacityStoreByteHeadroomMin: number | null;
  safeguardingSupervisorCapacityRecentErasureTombstones: number;
  safeguardingSupervisorCapacityRecentErasureTombstoneLimit: number;
  safeguardingSupervisorCapacityRecentErasureTombstoneHeadroomMin: number | null;
  safeguardingSupervisorRetentionCasesCompactedTotal: number;
  safeguardingSupervisorRetentionEventsCompactedTotal: number;
  safeguardingSupervisorErasureFenceInsertedCount: number;
  safeguardingSupervisorErasureFenceEstimatedFalsePositiveUpperBound: number;
  safeguardingSupervisorErasureFenceFalsePositiveTargetUpperBound: number;
  safeguardingSupervisorErasureFenceFalsePositiveWithinTarget: boolean;
  safeguardingSupervisorErasureFenceFalseNegativePossible: false;
  safeguardingSupervisorErasureFenceFalsePositivePolicy:
    "fail_closed_as_erased";
  safeguardingSupervisorUpdatedAtUtc: string | null;
  safeguardingSupervisorRawLearnerTextReadOrSent: false;
  safeguardingSupervisorScopeIdentityLabelsExposed: false;
  isolation:
    | "linux_landlock_scope_allowlist_shared_uid_not_vm_or_cgroup"
    | "one_logical_worker_scope_per_tenant_owner_same_os_uid_not_security_isolated";
  processResourceLimitsRequired: boolean;
  processResourceLimitsPolicy:
    | "core0_nofile256_fsize512m_as1536m"
    | "not_required_local_or_test";
  processResourceLimitsValidation:
    "exact_worker_and_runtime_canary_status_with_operational_minima";
  processResourceLimitsScope:
    "per_process_inherited_not_process_tree_or_cgroup";
  processCpuBoundary: "request_wall_clock_not_rlimit_cpu";
  processCountBoundary:
    | "production_container_pid_cap_256_not_per_worker_rlimit_nproc"
    | "not_required_local_or_test_no_process_count_limit";
  scopeKeyVersion: string;
  keyRotation: "active_previous_atomic_rewrap_to_active";
  scopeDataKeyPolicy: "aead_wrapped_stable_scope_key";
  previousRootRetirement: "signed_hash_only_tombstone";
  migrationCrashRecovery: "copy_fsync_verify_atomic_rename";
  backend: "deterministic" | "deepseek";
  remoteProviderCredentialReferenceConfigured: boolean;
  remoteProviderCredentialsLoadedWorkers: number;
  runtimeCanaryRequired: boolean;
  runtimeCanaryLastResult: "not_required" | "never" | "ready" | "failed";
  runtimeCanaryAttempts: number;
  runtimeCanarySuccesses: number;
  runtimeCanaryFailures: number;
  runtimeCanaryValidation:
    "bootstrap_policy_filesystem_process_limits_runtime_paths";
  runtimeCanaryRemoteProviderNetworkValidated: false;
  runtimeCanaryPersistentTenantDataCreated: false;
  providerReadinessRequired: boolean;
  providerReadinessLastResult: "not_required" | "never" | "ready" | "failed";
  providerReadinessAttempts: number;
  providerReadinessSuccesses: number;
  providerReadinessFailures: number;
  providerReadinessValidation: "authenticated_content_free_models_endpoint";
  providerReadinessLearnerContentSent: false;
  providerReadinessGenerationCreated: false;
  providerReadinessPersistentTenantDataCreated: false;
  capacity: number;
  readyWorkers: number;
  startingWorkers: number;
  stoppingWorkers: number;
  activeRequests: number;
  idleTimeoutMs: number;
  idleEvictableWorkers: number;
  idlePolicy: "capacity_triggered_expired_lru_only";
  activeOrStartingWorkersEvictable: false;
  durableScopeDataDeletedOnEviction: false;
  rawTenantMetadataStored: false;
  capabilityExposed: false;
}

export class HarnessGatewayError extends Error {
  constructor(
    readonly statusCode: number,
    readonly code:
      | "harness_gateway_disabled"
      | "harness_capacity_exhausted"
      | "harness_worker_unavailable"
      | "harness_upstream_unavailable"
      | "harness_request_invalid"
      | "harness_scope_fenced"
  ) {
    super(code);
    this.name = "HarnessGatewayError";
  }
}

export interface HarnessUpstreamRequest {
  method: "GET" | "HEAD" | "POST";
  path: string;
  body?: Buffer;
  accept?: string;
  signal?: AbortSignal;
  teacherRoles?: readonly string[];
  teacherAllowedRoles?: readonly string[];
  teacherPrincipal?: Readonly<{issuer: string; subject: string}>;
  /** Server-owned signed-session policy; never accepted from an HTTP body. */
  remoteSubjectPolicy?: RemoteSubjectPolicy;
  /** Staff routing may reuse a learner worker policy but can never upgrade it. */
  preserveRemoteSubjectPolicy?: boolean;
}

function hmac(secret: string | Buffer, value: string): Buffer {
  return createHmac("sha256", secret).update(value, "utf8").digest();
}

function canonicalScopeBinding(scope: AccessScope): string {
  const tenant = Buffer.from(scope.tenantId, "utf8");
  const owner = Buffer.from(scope.ownerId, "utf8");
  return `scope-v1\0${tenant.byteLength}:${scope.tenantId}\0${owner.byteLength}:${scope.ownerId}`;
}

function isPrivateMode(mode: number): boolean {
  return (mode & 0o077) === 0;
}

function safeWorkerProcessResourceLimitsStatus(
  value: unknown,
  expected: HarnessWorkerProcessResourceLimits | null
): boolean {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const row = value as Record<string, unknown>;
  const expectedKeys = new Set([
    "schema",
    "enforcement",
    "address_space_bytes",
    "file_size_bytes",
    "open_files",
    "core_dump_bytes",
    "cpu_time_limit",
    "process_count_limit",
    "scope"
  ]);
  const keys = Object.keys(row);
  if (
    keys.length !== expectedKeys.size ||
    keys.some((key) => !expectedKeys.has(key)) ||
    row.schema !==
      "teaching_skill_miner.worker_process_resource_limits_status.v1"
  ) {
    return false;
  }
  if (expected === null) {
    return (
      row.enforcement === "not_required_local_or_test" &&
      row.address_space_bytes === null &&
      row.file_size_bytes === null &&
      row.open_files === null &&
      row.core_dump_bytes === null &&
      row.cpu_time_limit === "not_required_local_or_test" &&
      row.process_count_limit === "not_required_local_or_test" &&
      row.scope === "none"
    );
  }
  return (
    row.enforcement === "linux_rlimit_v1" &&
    Number.isSafeInteger(row.address_space_bytes) &&
    Number(row.address_space_bytes) >= 1_073_741_824 &&
    Number(row.address_space_bytes) <= expected.address_space_bytes &&
    Number.isSafeInteger(row.file_size_bytes) &&
    Number(row.file_size_bytes) >= 335_544_320 &&
    Number(row.file_size_bytes) <= expected.file_size_bytes &&
    Number.isSafeInteger(row.open_files) &&
    Number(row.open_files) >= 128 &&
    Number(row.open_files) <= expected.open_files &&
    row.core_dump_bytes === 0 &&
    row.cpu_time_limit ===
      "not_set_long_lived_worker_uses_request_wall_clock" &&
    row.process_count_limit === "not_set_shared_uid_unsafe" &&
    row.scope ===
      "per_process_individual_inherited_not_process_tree_aggregate"
  );
}

function safeWorkerStatus(value: unknown, expected: {
  workerId: string;
  keyVersion: string;
  backend: "deterministic" | "deepseek";
  filesystemIsolationRequired: boolean;
  processResourceLimits: HarnessWorkerProcessResourceLimits | null;
}): {port: number} | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const row = value as Record<string, unknown>;
  const expectedKeys = new Set([
    "schema",
    "status",
    "worker_id",
    "scope_key_version",
    "port",
    "backend",
    "filesystem_isolation",
    "process_resource_limits"
  ]);
  const keys = Object.keys(row);
  if (
    keys.length !== expectedKeys.size ||
    keys.some((key) => !expectedKeys.has(key)) ||
    row.schema !== STATUS_SCHEMA ||
    row.status !== "ready" ||
    row.worker_id !== expected.workerId ||
    row.scope_key_version !== expected.keyVersion ||
    row.backend !== expected.backend ||
    (expected.filesystemIsolationRequired
      ? !/^linux_landlock_scope_allowlist_abi_[1-9][0-9]*$/.test(
          String(row.filesystem_isolation ?? "")
        )
      : row.filesystem_isolation !== "not_required_local_or_test") ||
    !safeWorkerProcessResourceLimitsStatus(
      row.process_resource_limits,
      expected.processResourceLimits
    ) ||
    !Number.isInteger(row.port) ||
    Number(row.port) < 1 ||
    Number(row.port) > 65_535
  ) {
    return undefined;
  }
  return {port: Number(row.port)};
}

function safeRuntimeCanaryStatus(value: unknown, expected: {
  workerId: string;
  keyVersion: string;
  backend: "deterministic" | "deepseek";
  filesystemIsolationRequired: boolean;
  processResourceLimits: HarnessWorkerProcessResourceLimits | null;
}): boolean {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const row = value as Record<string, unknown>;
  const keys = new Set(Object.keys(row));
  const expectedKeys = new Set([
    "schema",
    "status",
    "worker_id",
    "scope_key_version",
    "backend",
    "filesystem_isolation",
    "process_resource_limits",
    "remote_provider_network",
    "provider_credential",
    "persistent_tenant_data_created"
  ]);
  if (
    keys.size !== expectedKeys.size ||
    [...keys].some((key) => !expectedKeys.has(key))
  ) {
    return false;
  }
  return (
    row.schema === STATUS_SCHEMA &&
    row.status === "canary_ready" &&
    row.worker_id === expected.workerId &&
    row.scope_key_version === expected.keyVersion &&
    row.backend === expected.backend &&
    (expected.filesystemIsolationRequired
      ? /^linux_landlock_scope_allowlist_abi_[1-9][0-9]*$/.test(
          String(row.filesystem_isolation ?? "")
        )
      : row.filesystem_isolation === "not_required_local_or_test") &&
    safeWorkerProcessResourceLimitsStatus(
      row.process_resource_limits,
      expected.processResourceLimits
    ) &&
    row.remote_provider_network === "not_contacted" &&
    row.provider_credential ===
      (expected.backend === "deepseek"
        ? "loaded_not_provider_validated"
        : "not_applicable") &&
    row.persistent_tenant_data_created === false
  );
}

function safeProviderReadinessStatus(value: unknown): boolean {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const row = value as Record<string, unknown>;
  const expectedKeys = new Set([
    "schema",
    "status",
    "backend",
    "credential_validated",
    "provider_network_validated",
    "configured_model_available",
    "learner_content_sent",
    "generation_created",
    "persistent_tenant_data_created"
  ]);
  const keys = Object.keys(row);
  return (
    keys.length === expectedKeys.size &&
    keys.every((key) => expectedKeys.has(key)) &&
    row.schema === STATUS_SCHEMA &&
    row.status === "provider_ready" &&
    row.backend === "deepseek" &&
    row.credential_validated === true &&
    row.provider_network_validated === true &&
    row.configured_model_available === true &&
    row.learner_content_sent === false &&
    row.generation_created === false &&
    row.persistent_tenant_data_created === false
  );
}

const UTC_SECONDS =
  /^(?:[0-9]{4})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$/;

function safeUtcSeconds(value: unknown): value is string {
  if (typeof value !== "string" || !UTC_SECONDS.test(value)) return false;
  const milliseconds = Date.parse(value);
  return (
    Number.isFinite(milliseconds) &&
    new Date(milliseconds).toISOString().replace(".000Z", "Z") === value
  );
}

function nonnegativeSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}

function nullableNonnegativeSafeInteger(value: unknown): value is number | null {
  return value === null || nonnegativeSafeInteger(value);
}

function finiteProbability(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function safeSafeguardingSupervisorStatus(
  value: unknown,
  previous?: SafeguardingSupervisorAggregate
): SafeguardingSupervisorAggregate | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const row = value as Record<string, unknown>;
  const expectedKeys = new Set([
    "schema",
    "status",
    "scopes_scanned",
    "stores_unavailable",
    "pending",
    "overdue",
    "oldest_pending_age_seconds",
    "attempted",
    "accepted",
    "failed",
    "attempted_total",
    "accepted_total",
    "failed_total",
    "last_success_at_utc",
    "receiver_readiness_required",
    "receiver_readiness_status",
    "receiver_network_validated",
    "receiver_credential_validated",
    "receiver_readiness_attempts_total",
    "receiver_readiness_successes_total",
    "receiver_readiness_failures_total",
    "last_receiver_success_at_utc",
    "retention_enabled",
    "retention_status",
    "retention_policy_version_sha256",
    "retention_minimum_closed_age_seconds",
    "retention_maximum_cases_per_run",
    "retention_eligible_cases",
    "retention_cases_compacted",
    "retention_events_compacted",
    "retention_failures",
    "retention_blocked_stores",
    "capacity_near_limit_stores",
    "capacity_events",
    "capacity_event_limit",
    "capacity_event_headroom_min",
    "capacity_store_bytes",
    "capacity_store_byte_limit",
    "capacity_store_byte_headroom_min",
    "capacity_recent_erasure_tombstones",
    "capacity_recent_erasure_tombstone_limit",
    "capacity_recent_erasure_tombstone_headroom_min",
    "retention_cases_compacted_total",
    "retention_events_compacted_total",
    "erasure_fence_inserted_count",
    "erasure_fence_estimated_false_positive_upper_bound",
    "erasure_fence_false_positive_target_upper_bound",
    "erasure_fence_false_positive_within_target",
    "erasure_fence_false_negative_possible",
    "erasure_fence_false_positive_policy",
    "raw_learner_text_read_or_sent",
    "scope_identity_labels_exposed",
    "updated_at_utc"
  ]);
  const keys = Object.keys(row);
  if (
    keys.length !== expectedKeys.size ||
    keys.some((key) => !expectedKeys.has(key)) ||
    row.schema !== SAFEGUARDING_SUPERVISOR_STATUS_SCHEMA ||
    (row.status !== "ready" && row.status !== "degraded") ||
    row.raw_learner_text_read_or_sent !== false ||
    row.scope_identity_labels_exposed !== false ||
    row.receiver_readiness_required !== true ||
    (row.receiver_readiness_status !== "ready" &&
      row.receiver_readiness_status !== "failed") ||
    typeof row.receiver_network_validated !== "boolean" ||
    typeof row.receiver_credential_validated !== "boolean" ||
    typeof row.retention_enabled !== "boolean" ||
    !new Set(["disabled", "idle", "compacted", "blocked", "failed"]).has(
      String(row.retention_status)
    ) ||
    row.erasure_fence_false_positive_target_upper_bound !== 1e-6 ||
    typeof row.erasure_fence_false_positive_within_target !== "boolean" ||
    row.erasure_fence_false_negative_possible !== false ||
    row.erasure_fence_false_positive_policy !== "fail_closed_as_erased"
  ) {
    return undefined;
  }
  const integers = [
    row.scopes_scanned,
    row.stores_unavailable,
    row.pending,
    row.overdue,
    row.oldest_pending_age_seconds,
    row.attempted,
    row.accepted,
    row.failed,
    row.attempted_total,
    row.accepted_total,
    row.failed_total,
    row.receiver_readiness_attempts_total,
    row.receiver_readiness_successes_total,
    row.receiver_readiness_failures_total,
    row.retention_eligible_cases,
    row.retention_cases_compacted,
    row.retention_events_compacted,
    row.retention_failures,
    row.retention_blocked_stores,
    row.capacity_near_limit_stores,
    row.capacity_events,
    row.capacity_event_limit,
    row.capacity_store_bytes,
    row.capacity_store_byte_limit,
    row.capacity_recent_erasure_tombstones,
    row.capacity_recent_erasure_tombstone_limit,
    row.retention_cases_compacted_total,
    row.retention_events_compacted_total,
    row.erasure_fence_inserted_count
  ];
  if (
    integers.some((item) => !nonnegativeSafeInteger(item)) ||
    !nullableNonnegativeSafeInteger(row.capacity_event_headroom_min) ||
    !nullableNonnegativeSafeInteger(row.capacity_store_byte_headroom_min) ||
    !nullableNonnegativeSafeInteger(
      row.capacity_recent_erasure_tombstone_headroom_min
    ) ||
    !finiteProbability(
      row.erasure_fence_estimated_false_positive_upper_bound
    ) ||
    !safeUtcSeconds(row.updated_at_utc) ||
    (row.last_success_at_utc !== null && !safeUtcSeconds(row.last_success_at_utc)) ||
    (row.last_receiver_success_at_utc !== null &&
      !safeUtcSeconds(row.last_receiver_success_at_utc))
  ) {
    return undefined;
  }
  const aggregate: SafeguardingSupervisorAggregate = {
    status: row.status,
    scopesScanned: Number(row.scopes_scanned),
    storesUnavailable: Number(row.stores_unavailable),
    pending: Number(row.pending),
    overdue: Number(row.overdue),
    oldestPendingAgeSeconds: Number(row.oldest_pending_age_seconds),
    attempted: Number(row.attempted),
    accepted: Number(row.accepted),
    failed: Number(row.failed),
    attemptedTotal: Number(row.attempted_total),
    acceptedTotal: Number(row.accepted_total),
    failedTotal: Number(row.failed_total),
    lastSuccessAtUtc: row.last_success_at_utc,
    receiverReadinessRequired: row.receiver_readiness_required,
    receiverReadinessStatus: row.receiver_readiness_status,
    receiverNetworkValidated: row.receiver_network_validated,
    receiverCredentialValidated: row.receiver_credential_validated,
    receiverReadinessAttemptsTotal: Number(
      row.receiver_readiness_attempts_total
    ),
    receiverReadinessSuccessesTotal: Number(
      row.receiver_readiness_successes_total
    ),
    receiverReadinessFailuresTotal: Number(
      row.receiver_readiness_failures_total
    ),
    lastReceiverSuccessAtUtc: row.last_receiver_success_at_utc,
    retentionEnabled: row.retention_enabled,
    retentionStatus: row.retention_status as SafeguardingSupervisorAggregate["retentionStatus"],
    retentionPolicyVersionSha256:
      row.retention_policy_version_sha256 as string | null,
    retentionMinimumClosedAgeSeconds:
      row.retention_minimum_closed_age_seconds as number | null,
    retentionMaximumCasesPerRun:
      row.retention_maximum_cases_per_run as number | null,
    retentionEligibleCases: Number(row.retention_eligible_cases),
    retentionCasesCompacted: Number(row.retention_cases_compacted),
    retentionEventsCompacted: Number(row.retention_events_compacted),
    retentionFailures: Number(row.retention_failures),
    retentionBlockedStores: Number(row.retention_blocked_stores),
    capacityNearLimitStores: Number(row.capacity_near_limit_stores),
    capacityEvents: Number(row.capacity_events),
    capacityEventLimit: Number(row.capacity_event_limit),
    capacityEventHeadroomMin: row.capacity_event_headroom_min as number | null,
    capacityStoreBytes: Number(row.capacity_store_bytes),
    capacityStoreByteLimit: Number(row.capacity_store_byte_limit),
    capacityStoreByteHeadroomMin:
      row.capacity_store_byte_headroom_min as number | null,
    capacityRecentErasureTombstones: Number(
      row.capacity_recent_erasure_tombstones
    ),
    capacityRecentErasureTombstoneLimit: Number(
      row.capacity_recent_erasure_tombstone_limit
    ),
    capacityRecentErasureTombstoneHeadroomMin:
      row.capacity_recent_erasure_tombstone_headroom_min as number | null,
    retentionCasesCompactedTotal: Number(row.retention_cases_compacted_total),
    retentionEventsCompactedTotal: Number(row.retention_events_compacted_total),
    erasureFenceInsertedCount: Number(row.erasure_fence_inserted_count),
    erasureFenceEstimatedFalsePositiveUpperBound: Number(
      row.erasure_fence_estimated_false_positive_upper_bound
    ),
    erasureFenceFalsePositiveTargetUpperBound: Number(
      row.erasure_fence_false_positive_target_upper_bound
    ),
    erasureFenceFalsePositiveWithinTarget:
      row.erasure_fence_false_positive_within_target,
    erasureFenceFalseNegativePossible: false,
    erasureFenceFalsePositivePolicy: "fail_closed_as_erased",
    updatedAtUtc: row.updated_at_utc
  };
  const retentionStatusExpected = !aggregate.retentionEnabled
    ? "disabled"
    : aggregate.retentionFailures > 0
      ? "failed"
      : aggregate.retentionBlockedStores > 0
        ? "blocked"
        : aggregate.retentionCasesCompacted > 0
          ? "compacted"
          : "idle";
  const headrooms = [
    aggregate.capacityEventHeadroomMin,
    aggregate.capacityStoreByteHeadroomMin,
    aggregate.capacityRecentErasureTombstoneHeadroomMin
  ];
  if (
    aggregate.storesUnavailable > aggregate.scopesScanned ||
    aggregate.overdue > aggregate.pending ||
    (aggregate.pending === 0 && aggregate.oldestPendingAgeSeconds !== 0) ||
    aggregate.accepted + aggregate.failed > aggregate.attempted ||
    aggregate.attempted > aggregate.attemptedTotal ||
    aggregate.accepted > aggregate.acceptedTotal ||
    aggregate.failed > aggregate.failedTotal ||
    aggregate.acceptedTotal + aggregate.failedTotal > aggregate.attemptedTotal ||
    aggregate.receiverReadinessSuccessesTotal +
      aggregate.receiverReadinessFailuresTotal >
      aggregate.receiverReadinessAttemptsTotal ||
    (aggregate.receiverReadinessStatus === "ready") !==
      (
        aggregate.receiverNetworkValidated &&
        aggregate.receiverCredentialValidated
      ) ||
    (aggregate.receiverReadinessSuccessesTotal === 0) !==
      (aggregate.lastReceiverSuccessAtUtc === null) ||
    aggregate.retentionStatus !== retentionStatusExpected ||
    (
      aggregate.retentionEnabled
        ? (
            !/^[0-9a-f]{64}$/.test(
              aggregate.retentionPolicyVersionSha256 ?? ""
            ) ||
            !Number.isSafeInteger(aggregate.retentionMinimumClosedAgeSeconds) ||
            Number(aggregate.retentionMinimumClosedAgeSeconds) < 1 ||
            Number(aggregate.retentionMinimumClosedAgeSeconds) >
              10 * 365 * 24 * 60 * 60 ||
            !Number.isSafeInteger(aggregate.retentionMaximumCasesPerRun) ||
            Number(aggregate.retentionMaximumCasesPerRun) < 1 ||
            Number(aggregate.retentionMaximumCasesPerRun) > 1024
          )
        : aggregate.retentionPolicyVersionSha256 !== null ||
          aggregate.retentionMinimumClosedAgeSeconds !== null ||
          aggregate.retentionMaximumCasesPerRun !== null
    ) ||
    aggregate.retentionCasesCompacted > aggregate.retentionEligibleCases ||
    aggregate.retentionCasesCompacted > aggregate.retentionCasesCompactedTotal ||
    aggregate.retentionEventsCompacted > aggregate.retentionEventsCompactedTotal ||
    aggregate.retentionBlockedStores > aggregate.scopesScanned ||
    aggregate.capacityNearLimitStores > aggregate.scopesScanned ||
    aggregate.capacityEvents > aggregate.capacityEventLimit ||
    aggregate.capacityStoreBytes > aggregate.capacityStoreByteLimit ||
    aggregate.capacityRecentErasureTombstones >
      aggregate.capacityRecentErasureTombstoneLimit ||
    (headrooms.some((value) => value === null) &&
      !headrooms.every((value) => value === null)) ||
    aggregate.erasureFenceFalsePositiveWithinTarget !==
      (
        aggregate.erasureFenceEstimatedFalsePositiveUpperBound <=
        aggregate.erasureFenceFalsePositiveTargetUpperBound
      ) ||
    (aggregate.status === "ready") !==
      (
        aggregate.storesUnavailable === 0 &&
        aggregate.failed === 0 &&
        aggregate.retentionFailures === 0 &&
        aggregate.retentionBlockedStores === 0 &&
        aggregate.capacityNearLimitStores === 0 &&
        aggregate.erasureFenceFalsePositiveWithinTarget &&
        aggregate.receiverReadinessStatus === "ready"
      ) ||
    (aggregate.acceptedTotal === 0) !== (aggregate.lastSuccessAtUtc === null) ||
    (aggregate.lastSuccessAtUtc !== null &&
      Date.parse(aggregate.lastSuccessAtUtc) > Date.parse(aggregate.updatedAtUtc)) ||
    (aggregate.lastReceiverSuccessAtUtc !== null &&
      Date.parse(aggregate.lastReceiverSuccessAtUtc) >
        Date.parse(aggregate.updatedAtUtc))
  ) {
    return undefined;
  }
  if (
    previous &&
    (
      aggregate.attemptedTotal < previous.attemptedTotal ||
      aggregate.acceptedTotal < previous.acceptedTotal ||
      aggregate.failedTotal < previous.failedTotal ||
      aggregate.receiverReadinessAttemptsTotal <
        previous.receiverReadinessAttemptsTotal ||
      aggregate.receiverReadinessSuccessesTotal <
        previous.receiverReadinessSuccessesTotal ||
      aggregate.receiverReadinessFailuresTotal <
        previous.receiverReadinessFailuresTotal ||
      aggregate.retentionCasesCompactedTotal <
        previous.retentionCasesCompactedTotal ||
      aggregate.retentionEventsCompactedTotal <
        previous.retentionEventsCompactedTotal ||
      aggregate.erasureFenceInsertedCount < previous.erasureFenceInsertedCount ||
      Date.parse(aggregate.updatedAtUtc) < Date.parse(previous.updatedAtUtc) ||
      (previous.lastSuccessAtUtc !== null &&
        (aggregate.lastSuccessAtUtc === null ||
          Date.parse(aggregate.lastSuccessAtUtc) <
            Date.parse(previous.lastSuccessAtUtc))) ||
      (previous.lastReceiverSuccessAtUtc !== null &&
        (aggregate.lastReceiverSuccessAtUtc === null ||
          Date.parse(aggregate.lastReceiverSuccessAtUtc) <
            Date.parse(previous.lastReceiverSuccessAtUtc)))
    )
  ) {
    return undefined;
  }
  return aggregate;
}

@Injectable()
export class HarnessWorkerPoolService implements OnApplicationShutdown {
  private readonly workers = new Map<string, WorkerRecord>();
  private readonly starting = new Map<string, Promise<WorkerRecord>>();
  private readonly startingPolicyHashes = new Map<string, string>();
  private readonly retiring = new Set<WorkerRecord>();
  private startGate: Promise<void> = Promise.resolve();
  private clock: () => number = () => Date.now();
  private baseReady?: Promise<void>;
  private runtimeCanaryCache?: {ready: boolean; expiresAtMs: number};
  private runtimeCanaryPending?: Promise<void>;
  private runtimeCanaryLastResult: "not_required" | "never" | "ready" | "failed";
  private runtimeCanaryAttempts = 0;
  private runtimeCanarySuccesses = 0;
  private runtimeCanaryFailures = 0;
  private readonly runtimeCanaryChildren = new Set<ChildProcessWithoutNullStreams>();
  private providerReadinessCache?: {ready: boolean; expiresAtMs: number};
  private providerReadinessPending?: Promise<void>;
  private providerReadinessLastResult: "not_required" | "never" | "ready" | "failed";
  private providerReadinessAttempts = 0;
  private providerReadinessSuccesses = 0;
  private providerReadinessFailures = 0;
  private providerReadinessOverride?: ProviderReadinessOverride;
  private safeguardingSupervisor?: SafeguardingSupervisorRecord;
  private safeguardingSupervisorStart?: Promise<void>;
  private safeguardingSupervisorAggregate?: SafeguardingSupervisorAggregate;
  private safeguardingSupervisorReceivedAtMs?: number;
  private safeguardingSupervisorTerminalResult?: "failed" | "unavailable";
  private shuttingDown = false;
  private readonly accountBlocks = new Map<
    string,
    {kind: "export" | "deletion"; operationId?: string}
  >();

  constructor(@Inject(AppConfigService) private readonly config: AppConfigService) {
    this.runtimeCanaryLastResult =
      this.config.nodeEnv === "production" && this.config.harnessGatewayEnabled
        ? "never"
        : "not_required";
    this.providerReadinessLastResult = this.providerReadinessRequired()
      ? "never"
      : "not_required";
  }

  async assertReady(): Promise<void> {
    if (!this.config.harnessGatewayEnabled) return;
    await this.ensureBaseReady();
    await this.ensureSafeguardingSupervisorStarted();
    this.assertSafeguardingSupervisorReady();
    if (this.config.nodeEnv === "production") {
      await this.ensureRuntimeCanaryReady();
      await this.ensureProviderReadiness();
    }
  }

  status(): HarnessWorkerPoolStatus {
    const now = this.nowMs();
    const supervisorRequired = this.safeguardingSupervisorRequired();
    const supervisorRunning = this.safeguardingSupervisorRunning();
    const supervisorResult = this.safeguardingSupervisorResult(now);
    const supervisor = this.safeguardingSupervisorAggregate;
    return {
      enabled: this.config.harnessGatewayEnabled,
      safeguardingDispatcherConfigured:
        Boolean(this.config.harnessSafeguardingDispatchConfigured),
      safeguardingStaffWorkflow:
        this.config.harnessSafeguardingDispatchConfigured
          ? "content_free_dispatch_configured"
          : "cases_durable_dispatch_unavailable",
      safeguardingSupervisorRequired: supervisorRequired,
      safeguardingSupervisorRunning: supervisorRunning,
      safeguardingSupervisorLastResult: supervisorResult,
      safeguardingSupervisorScopesScanned: supervisor?.scopesScanned ?? 0,
      safeguardingSupervisorStoresUnavailable:
        supervisor?.storesUnavailable ?? 0,
      safeguardingSupervisorPending: supervisor?.pending ?? 0,
      safeguardingSupervisorOverdue: supervisor?.overdue ?? 0,
      safeguardingSupervisorOldestPendingAgeSeconds:
        supervisor?.oldestPendingAgeSeconds ?? 0,
      safeguardingSupervisorAttempted: supervisor?.attempted ?? 0,
      safeguardingSupervisorAccepted: supervisor?.accepted ?? 0,
      safeguardingSupervisorFailed: supervisor?.failed ?? 0,
      safeguardingSupervisorAttemptedTotal: supervisor?.attemptedTotal ?? 0,
      safeguardingSupervisorAcceptedTotal: supervisor?.acceptedTotal ?? 0,
      safeguardingSupervisorFailedTotal: supervisor?.failedTotal ?? 0,
      safeguardingSupervisorLastSuccessAtUtc:
        supervisor?.lastSuccessAtUtc ?? null,
      safeguardingSupervisorReceiverReadinessRequired:
        supervisor?.receiverReadinessRequired ?? supervisorRequired,
      safeguardingSupervisorReceiverReadinessStatus:
        supervisor?.receiverReadinessStatus ??
          (supervisorRequired ? "failed" : "not_required"),
      safeguardingSupervisorReceiverNetworkValidated:
        supervisor?.receiverNetworkValidated ?? false,
      safeguardingSupervisorReceiverCredentialValidated:
        supervisor?.receiverCredentialValidated ?? false,
      safeguardingSupervisorReceiverReadinessAttemptsTotal:
        supervisor?.receiverReadinessAttemptsTotal ?? 0,
      safeguardingSupervisorReceiverReadinessSuccessesTotal:
        supervisor?.receiverReadinessSuccessesTotal ?? 0,
      safeguardingSupervisorReceiverReadinessFailuresTotal:
        supervisor?.receiverReadinessFailuresTotal ?? 0,
      safeguardingSupervisorLastReceiverSuccessAtUtc:
        supervisor?.lastReceiverSuccessAtUtc ?? null,
      safeguardingSupervisorRetentionEnabled:
        supervisor?.retentionEnabled ??
        Boolean(this.config.harnessSafeguardingRetentionConfigured),
      safeguardingSupervisorRetentionStatus:
        supervisor?.retentionStatus ??
        (this.config.harnessSafeguardingRetentionConfigured ? "idle" : "disabled"),
      safeguardingSupervisorRetentionPolicyVersionSha256:
        supervisor?.retentionPolicyVersionSha256 ??
        (this.config.harnessSafeguardingRetentionConfigured
          ? createHash("sha256")
              .update(
                this.config.harnessSafeguardingRetentionPolicyVersion,
                "utf8"
              )
              .digest("hex")
          : null),
      safeguardingSupervisorRetentionMinimumClosedAgeSeconds:
        supervisor?.retentionMinimumClosedAgeSeconds ??
        (this.config.harnessSafeguardingRetentionConfigured
          ? this.config.harnessSafeguardingRetentionMinimumClosedAgeSeconds
          : null),
      safeguardingSupervisorRetentionMaximumCasesPerRun:
        supervisor?.retentionMaximumCasesPerRun ??
        (this.config.harnessSafeguardingRetentionConfigured
          ? this.config.harnessSafeguardingRetentionMaximumCasesPerRun
          : null),
      safeguardingSupervisorRetentionEligibleCases:
        supervisor?.retentionEligibleCases ?? 0,
      safeguardingSupervisorRetentionCasesCompacted:
        supervisor?.retentionCasesCompacted ?? 0,
      safeguardingSupervisorRetentionEventsCompacted:
        supervisor?.retentionEventsCompacted ?? 0,
      safeguardingSupervisorRetentionFailures:
        supervisor?.retentionFailures ?? 0,
      safeguardingSupervisorRetentionBlockedStores:
        supervisor?.retentionBlockedStores ?? 0,
      safeguardingSupervisorCapacityNearLimitStores:
        supervisor?.capacityNearLimitStores ?? 0,
      safeguardingSupervisorCapacityEvents: supervisor?.capacityEvents ?? 0,
      safeguardingSupervisorCapacityEventLimit:
        supervisor?.capacityEventLimit ?? 0,
      safeguardingSupervisorCapacityEventHeadroomMin:
        supervisor?.capacityEventHeadroomMin ?? null,
      safeguardingSupervisorCapacityStoreBytes:
        supervisor?.capacityStoreBytes ?? 0,
      safeguardingSupervisorCapacityStoreByteLimit:
        supervisor?.capacityStoreByteLimit ?? 0,
      safeguardingSupervisorCapacityStoreByteHeadroomMin:
        supervisor?.capacityStoreByteHeadroomMin ?? null,
      safeguardingSupervisorCapacityRecentErasureTombstones:
        supervisor?.capacityRecentErasureTombstones ?? 0,
      safeguardingSupervisorCapacityRecentErasureTombstoneLimit:
        supervisor?.capacityRecentErasureTombstoneLimit ?? 0,
      safeguardingSupervisorCapacityRecentErasureTombstoneHeadroomMin:
        supervisor?.capacityRecentErasureTombstoneHeadroomMin ?? null,
      safeguardingSupervisorRetentionCasesCompactedTotal:
        supervisor?.retentionCasesCompactedTotal ?? 0,
      safeguardingSupervisorRetentionEventsCompactedTotal:
        supervisor?.retentionEventsCompactedTotal ?? 0,
      safeguardingSupervisorErasureFenceInsertedCount:
        supervisor?.erasureFenceInsertedCount ?? 0,
      safeguardingSupervisorErasureFenceEstimatedFalsePositiveUpperBound:
        supervisor?.erasureFenceEstimatedFalsePositiveUpperBound ?? 0,
      safeguardingSupervisorErasureFenceFalsePositiveTargetUpperBound:
        supervisor?.erasureFenceFalsePositiveTargetUpperBound ?? 1e-6,
      safeguardingSupervisorErasureFenceFalsePositiveWithinTarget:
        supervisor?.erasureFenceFalsePositiveWithinTarget ?? true,
      safeguardingSupervisorErasureFenceFalseNegativePossible: false,
      safeguardingSupervisorErasureFenceFalsePositivePolicy:
        "fail_closed_as_erased",
      safeguardingSupervisorUpdatedAtUtc: supervisor?.updatedAtUtc ?? null,
      safeguardingSupervisorRawLearnerTextReadOrSent: false,
      safeguardingSupervisorScopeIdentityLabelsExposed: false,
      isolation: this.config.harnessWorkerFilesystemIsolationRequired
        ? "linux_landlock_scope_allowlist_shared_uid_not_vm_or_cgroup"
        : "one_logical_worker_scope_per_tenant_owner_same_os_uid_not_security_isolated",
      processResourceLimitsRequired:
        this.config.harnessWorkerFilesystemIsolationRequired,
      processResourceLimitsPolicy:
        this.config.harnessWorkerFilesystemIsolationRequired
          ? "core0_nofile256_fsize512m_as1536m"
          : "not_required_local_or_test",
      processResourceLimitsValidation:
        "exact_worker_and_runtime_canary_status_with_operational_minima",
      processResourceLimitsScope:
        "per_process_inherited_not_process_tree_or_cgroup",
      processCpuBoundary: "request_wall_clock_not_rlimit_cpu",
      processCountBoundary:
        this.config.nodeEnv === "production"
          ? "production_container_pid_cap_256_not_per_worker_rlimit_nproc"
          : "not_required_local_or_test_no_process_count_limit",
      scopeKeyVersion: this.config.harnessScopeKeyVersion,
      keyRotation: "active_previous_atomic_rewrap_to_active",
      scopeDataKeyPolicy: "aead_wrapped_stable_scope_key",
      previousRootRetirement: "signed_hash_only_tombstone",
      migrationCrashRecovery: "copy_fsync_verify_atomic_rename",
      backend: this.config.harnessWorkerBackend,
      remoteProviderCredentialReferenceConfigured:
        this.config.harnessWorkerBackend === "deepseek" &&
        Boolean(this.config.harnessProviderApiKeyFile),
      remoteProviderCredentialsLoadedWorkers:
        this.config.harnessWorkerBackend === "deepseek" ? this.workers.size : 0,
      runtimeCanaryRequired:
        this.config.nodeEnv === "production" && this.config.harnessGatewayEnabled,
      runtimeCanaryLastResult: this.runtimeCanaryLastResult,
      runtimeCanaryAttempts: this.runtimeCanaryAttempts,
      runtimeCanarySuccesses: this.runtimeCanarySuccesses,
      runtimeCanaryFailures: this.runtimeCanaryFailures,
      runtimeCanaryValidation:
        "bootstrap_policy_filesystem_process_limits_runtime_paths",
      runtimeCanaryRemoteProviderNetworkValidated: false,
      runtimeCanaryPersistentTenantDataCreated: false,
      providerReadinessRequired: this.providerReadinessRequired(),
      providerReadinessLastResult: this.providerReadinessLastResult,
      providerReadinessAttempts: this.providerReadinessAttempts,
      providerReadinessSuccesses: this.providerReadinessSuccesses,
      providerReadinessFailures: this.providerReadinessFailures,
      providerReadinessValidation: "authenticated_content_free_models_endpoint",
      providerReadinessLearnerContentSent: false,
      providerReadinessGenerationCreated: false,
      providerReadinessPersistentTenantDataCreated: false,
      capacity: this.config.harnessMaxWorkers,
      readyWorkers: this.workers.size,
      startingWorkers: this.starting.size,
      stoppingWorkers: this.retiring.size,
      activeRequests: [...this.workers.values()].reduce(
        (total, worker) => total + worker.activeRequests,
        0
      ),
      idleTimeoutMs: this.config.harnessWorkerIdleTimeoutMs,
      idleEvictableWorkers: [...this.workers.values()].filter(
        (worker) => this.isIdleEvictionCandidate(worker, now)
      ).length,
      idlePolicy: "capacity_triggered_expired_lru_only",
      activeOrStartingWorkersEvictable: false,
      durableScopeDataDeletedOnEviction: false,
      rawTenantMetadataStored: false,
      capabilityExposed: false
    };
  }

  private safeguardingSupervisorRequired(): boolean {
    return Boolean(
      this.config.harnessGatewayEnabled &&
      this.config.harnessSafeguardingDispatchConfigured
    );
  }

  private providerReadinessRequired(): boolean {
    return (
      this.config.nodeEnv === "production" &&
      this.config.harnessGatewayEnabled &&
      this.config.harnessWorkerBackend === "deepseek"
    );
  }

  private safeguardingSupervisorRunning(): boolean {
    const record = this.safeguardingSupervisor;
    return Boolean(
      record &&
      !record.stopping &&
      !record.protocolFailed &&
      record.child.exitCode === null &&
      record.child.signalCode === null
    );
  }

  private safeguardingSupervisorResult(nowMs: number): SafeguardingSupervisorResult {
    if (!this.safeguardingSupervisorRequired()) return "not_required";
    if (this.safeguardingSupervisorTerminalResult) {
      return this.safeguardingSupervisorTerminalResult;
    }
    if (!this.safeguardingSupervisor) return "starting";
    if (!this.safeguardingSupervisorRunning()) return "unavailable";
    const aggregate = this.safeguardingSupervisorAggregate;
    const receivedAtMs = this.safeguardingSupervisorReceivedAtMs;
    if (!aggregate || receivedAtMs === undefined) return "starting";
    if (
      nowMs < receivedAtMs ||
      nowMs - receivedAtMs > this.safeguardingSupervisorStaleMaxMs()
    ) {
      return "stale";
    }
    if (aggregate.overdue > 0) return "overdue";
    if (
      aggregate.status !== "ready" ||
      aggregate.storesUnavailable > 0 ||
      aggregate.failed > 0 ||
      aggregate.receiverReadinessStatus !== "ready" ||
      !aggregate.receiverNetworkValidated ||
      !aggregate.receiverCredentialValidated ||
      aggregate.retentionFailures > 0 ||
      aggregate.retentionBlockedStores > 0 ||
      aggregate.capacityNearLimitStores > 0 ||
      !aggregate.erasureFenceFalsePositiveWithinTarget ||
      aggregate.erasureFenceFalseNegativePossible ||
      (this.config.harnessSafeguardingRetentionConfigured &&
        (
          !aggregate.retentionEnabled ||
          aggregate.retentionStatus === "disabled" ||
          aggregate.retentionPolicyVersionSha256 !==
            createHash("sha256")
              .update(
                this.config.harnessSafeguardingRetentionPolicyVersion,
                "utf8"
              )
              .digest("hex") ||
          aggregate.retentionMinimumClosedAgeSeconds !==
            this.config.harnessSafeguardingRetentionMinimumClosedAgeSeconds ||
          aggregate.retentionMaximumCasesPerRun !==
            this.config.harnessSafeguardingRetentionMaximumCasesPerRun
        ))
    ) {
      return "degraded";
    }
    return "ready";
  }

  private assertSafeguardingSupervisorReady(): void {
    if (!this.safeguardingSupervisorRequired()) return;
    if (
      !this.safeguardingSupervisorRunning() ||
      this.safeguardingSupervisorResult(this.nowMs()) !== "ready"
    ) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
  }

  private safeguardingSupervisorStaleMaxMs(): number {
    return Math.max(
      SAFEGUARDING_SUPERVISOR_STALE_MAX_MS,
      this.config.harnessSafeguardingDispatchTimeoutMs * 2 +
        SAFEGUARDING_SUPERVISOR_POLL_SECONDS * 1_000
    );
  }

  setClockForTesting(clock: () => number): void {
    if (this.config.nodeEnv !== "test" || typeof clock !== "function") {
      throw new Error("test-only worker clock is unavailable");
    }
    this.clock = clock;
  }

  setProviderReadinessProbeForTesting(
    probe: () => Promise<void>
  ): void {
    if (
      (this.config.nodeEnv !== "test" && process.env.NODE_ENV !== "test") ||
      typeof probe !== "function"
    ) {
      throw new Error("test-only provider readiness probe is unavailable");
    }
    this.providerReadinessOverride = {run: probe};
  }

  async request(
    scope: AccessScope,
    input: HarnessUpstreamRequest
  ): Promise<IncomingMessage> {
    if (!this.config.harnessGatewayEnabled || this.shuttingDown) {
      throw new HarnessGatewayError(503, "harness_gateway_disabled");
    }
    const worker = await this.acquire(
      scope,
      input.remoteSubjectPolicy,
      input.preserveRemoteSubjectPolicy === true
    );
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      worker.activeRequests = Math.max(0, worker.activeRequests - 1);
      worker.lastActivityMs = Math.max(worker.lastActivityMs, this.nowMs());
    };
    let body: Buffer | undefined;
    try {
      body = this.scopedRequestBody(
        worker,
        input.path,
        input.body,
        input.teacherRoles,
        input.teacherAllowedRoles,
        input.teacherPrincipal
      );
    } catch (error) {
      release();
      throw error;
    }

    return new Promise<IncomingMessage>((resolve, reject) => {
      const upstream = httpRequest(
        {
          host: "127.0.0.1",
          port: worker.port,
          path: `/${worker.capability}/${input.path}`,
          method: input.method,
          headers: {
            accept: input.accept ?? "application/json",
            "cache-control": "no-store",
            ...(body
              ? {
                  "content-type": "application/json",
                  "content-length": String(body.byteLength)
                }
              : {})
          },
          agent: false
        },
        (response) => {
          response.once("end", release);
          response.once("close", release);
          response.once("error", release);
          resolve(response);
        }
      );
      const fail = () => {
        release();
        reject(new HarnessGatewayError(503, "harness_upstream_unavailable"));
      };
      upstream.once("error", fail);
      if (input.signal) {
        const abort = () => upstream.destroy();
        if (input.signal.aborted) abort();
        else input.signal.addEventListener("abort", abort, {once: true});
        upstream.once("close", () =>
          input.signal?.removeEventListener("abort", abort)
        );
      }
      if (body) upstream.end(body);
      else upstream.end();
    });
  }

  /** Internal account-rights API; paths are never returned over HTTP. */
  async beginAccountExport(scope: AccessScope): Promise<{
    roots: readonly {keyVersion: string; scopeId: string; privateRoot: string}[];
    release(): Promise<void>;
  }> {
    const coordinationKey = this.accountCoordinationKey(scope);
    await this.withStartGate(() => {
      if (this.accountBlocks.has(coordinationKey)) {
        throw new HarnessGatewayError(503, "harness_scope_fenced");
      }
      this.accountBlocks.set(coordinationKey, {kind: "export"});
    });
    let released = false;
    const release = async () => {
      if (released) return;
      released = true;
      await this.withStartGate(() => {
        const current = this.accountBlocks.get(coordinationKey);
        if (current?.kind === "export") this.accountBlocks.delete(coordinationKey);
      });
    };
    try {
      await this.stopScopeForAccount(scope, this.config.harnessWorkerShutdownMs, false);
      return {roots: await this.accountRoots(scope), release};
    } catch (error) {
      await release();
      throw error;
    }
  }

  async fenceAccountScope(scope: AccessScope, operationId: string): Promise<void> {
    if (!/^adel_[0-9a-f]{32}$/.test(operationId)) {
      throw new HarnessGatewayError(400, "harness_request_invalid");
    }
    const key = this.accountCoordinationKey(scope);
    await this.withStartGate(() => {
      const current = this.accountBlocks.get(key);
      if (current?.kind === "deletion" && current.operationId === operationId) return;
      if (current) throw new HarnessGatewayError(503, "harness_scope_fenced");
      this.accountBlocks.set(key, {kind: "deletion", operationId});
    });
  }

  async drainAccountScope(
    scope: AccessScope,
    operationId: string,
    timeoutMs: number
  ): Promise<{activeRequestsCancelled: number}> {
    const key = this.accountCoordinationKey(scope);
    const block = this.accountBlocks.get(key);
    if (block?.kind !== "deletion" || block.operationId !== operationId) {
      throw new HarnessGatewayError(503, "harness_scope_fenced");
    }
    const cancelled = await this.stopScopeForAccount(scope, timeoutMs, true);
    return {activeRequestsCancelled: cancelled};
  }

  async accountRoots(scope: AccessScope): Promise<readonly {
    keyVersion: string;
    scopeId: string;
    privateRoot: string;
  }[]> {
    await this.ensureBaseReady();
    const root = this.config.harnessWorkerRoot;
    if (!root) throw new HarnessGatewayError(503, "harness_worker_unavailable");
    const binding = canonicalScopeBinding(scope);
    const existing: ScopeIdentity[] = [];
    for (const key of this.config.harnessScopeKeys) {
      const identity = this.identityForKey(root, binding, key);
      try {
        const metadata = await lstat(identity.privateRoot);
        if (metadata.isSymbolicLink() || !metadata.isDirectory() || !isPrivateMode(metadata.mode)) {
          throw new HarnessGatewayError(503, "harness_worker_unavailable");
        }
        existing.push(identity);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
    }
    return existing.map(({keyVersion, scopeId, privateRoot}) => ({
      keyVersion, scopeId, privateRoot
    }));
  }

  async terminateForTesting(scope: AccessScope): Promise<boolean> {
    if (this.config.nodeEnv !== "test") {
      throw new Error("test-only worker termination is unavailable");
    }
    const identity = await this.scopeIdentity(scope, false);
    const record = this.workers.get(identity.key);
    if (!record) return false;
    const exited = new Promise<void>((resolve) => record.child.once("exit", () => resolve()));
    record.child.kill("SIGKILL");
    await exited;
    return true;
  }

  async terminateSafeguardingSupervisorForTesting(): Promise<boolean> {
    if (this.config.nodeEnv !== "test") {
      throw new Error("test-only safeguarding supervisor termination is unavailable");
    }
    const record = this.safeguardingSupervisor;
    if (
      !record ||
      record.child.exitCode !== null ||
      record.child.signalCode !== null
    ) {
      return false;
    }
    const exited = new Promise<void>((resolve) =>
      record.child.once("exit", () => resolve())
    );
    record.child.kill("SIGKILL");
    await exited;
    return true;
  }

  async onApplicationShutdown(): Promise<void> {
    this.shuttingDown = true;
    for (const child of this.runtimeCanaryChildren) child.kill("SIGKILL");
    const records = [...new Set([...this.workers.values(), ...this.retiring])];
    await Promise.all([
      ...records.map((record) => this.stopWorker(record)),
      this.stopSafeguardingSupervisor(),
      this.safeguardingSupervisorStart?.catch(() => undefined) ?? Promise.resolve(),
      this.providerReadinessPending?.catch(() => undefined) ?? Promise.resolve(),
      this.runtimeCanaryPending?.catch(() => undefined) ?? Promise.resolve()
    ]);
  }

  private async ensureBaseReady(): Promise<void> {
    this.baseReady ??= this.prepareBase();
    return this.baseReady;
  }

  private async prepareBase(): Promise<void> {
    const root = this.config.harnessWorkerRoot;
    const cwd = this.config.harnessWorkerCwd;
    const python = this.config.harnessWorkerPython;
    if (!root || !cwd || !python) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    try {
      await mkdir(root, {recursive: true, mode: 0o700});
      const rootMetadata = await lstat(root);
      if (rootMetadata.isSymbolicLink() || !rootMetadata.isDirectory()) {
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
      await chmod(root, 0o700);
      const cwdMetadata = await lstat(cwd);
      if (cwdMetadata.isSymbolicLink() || !cwdMetadata.isDirectory()) {
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
      const pythonMetadata = await lstat(python);
      if (pythonMetadata.isSymbolicLink() || !pythonMetadata.isFile()) {
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
      await access(python, fsConstants.X_OK);
      if (this.config.harnessWorkerBackend === "deepseek") {
        const credential = this.config.harnessProviderApiKeyFile;
        if (!credential) {
          throw new HarnessGatewayError(503, "harness_worker_unavailable");
        }
        const credentialMetadata = await lstat(credential);
        if (
          credentialMetadata.isSymbolicLink() ||
          !credentialMetadata.isFile() ||
          !isPrivateMode(credentialMetadata.mode) ||
          credentialMetadata.size < 1 ||
          credentialMetadata.size > 4096
        ) {
          throw new HarnessGatewayError(503, "harness_worker_unavailable");
        }
        const key = (await readFile(credential, "utf8")).trim();
        if (!key || /\s/.test(key)) {
          throw new HarnessGatewayError(503, "harness_worker_unavailable");
        }
      }
    } catch (error) {
      if (error instanceof HarnessGatewayError) throw error;
      // Filesystem exceptions include the configured private path. Collapse
      // them before they reach startup logs, health output, or HTTP errors.
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
  }

  private async ensureSafeguardingSupervisorStarted(): Promise<void> {
    if (!this.safeguardingSupervisorRequired()) return;
    if (this.safeguardingSupervisorStart) return this.safeguardingSupervisorStart;
    if (this.safeguardingSupervisorRunning()) return;
    if (this.safeguardingSupervisor || this.safeguardingSupervisorTerminalResult) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    const pending = this.startSafeguardingSupervisor().finally(() => {
      if (this.safeguardingSupervisorStart === pending) {
        this.safeguardingSupervisorStart = undefined;
      }
    });
    this.safeguardingSupervisorStart = pending;
    return pending;
  }

  private safeguardingSupervisorBootstrap(): SafeguardingSupervisorBootstrap {
    const root = this.config.harnessWorkerRoot;
    const endpoint = this.config.harnessSafeguardingDispatchUrl;
    const bearerSecret = this.config.harnessSafeguardingDispatchBearerSecret;
    const retentionAuthoritySecret =
      this.config.harnessSafeguardingRetentionAuthoritySecret;
    const retentionDeploymentContextSha256 =
      this.config.harnessSafeguardingRetentionDeploymentContextSha256;
    if (
      !root ||
      !endpoint ||
      !bearerSecret ||
      !this.config.harnessSafeguardingRetentionConfigured ||
      !retentionAuthoritySecret ||
      !retentionDeploymentContextSha256
    ) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    return {
      schema: SAFEGUARDING_SUPERVISOR_BOOTSTRAP_SCHEMA,
      root,
      endpoint,
      bearer_secret: bearerSecret,
      policy_version: this.config.harnessSafeguardingDispatchPolicyVersion,
      timeout_ms: this.config.harnessSafeguardingDispatchTimeoutMs,
      maximum_response_bytes:
        this.config.harnessSafeguardingDispatchMaxResponseBytes,
      poll_seconds: SAFEGUARDING_SUPERVISOR_POLL_SECONDS,
      retention_policy_version:
        this.config.harnessSafeguardingRetentionPolicyVersion,
      retention_minimum_closed_age_seconds:
        this.config.harnessSafeguardingRetentionMinimumClosedAgeSeconds,
      retention_maximum_cases_per_run:
        this.config.harnessSafeguardingRetentionMaximumCasesPerRun,
      retention_authority_secret: retentionAuthoritySecret,
      retention_deployment_context_sha256: retentionDeploymentContextSha256
    };
  }

  private async startSafeguardingSupervisor(): Promise<void> {
    const python = this.config.harnessWorkerPython;
    const cwd = this.config.harnessWorkerCwd;
    if (!python || !cwd || this.shuttingDown) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    const bootstrap = this.safeguardingSupervisorBootstrap();
    const child = spawn(
      python,
      ["-m", "teaching_skill_miner.teacher_agent_safeguarding_supervisor"],
      {
        cwd,
        env: {
          LANG: "C.UTF-8",
          LC_ALL: "C.UTF-8",
          PYTHONDONTWRITEBYTECODE: "1",
          PYTHONUNBUFFERED: "1"
        },
        stdio: ["pipe", "pipe", "pipe"]
      }
    );
    const record: SafeguardingSupervisorRecord = {
      child,
      stdout: Buffer.alloc(0),
      stderrBytes: 0,
      stopping: false,
      protocolFailed: false
    };
    this.safeguardingSupervisor = record;
    this.safeguardingSupervisorAggregate = undefined;
    this.safeguardingSupervisorReceivedAtMs = undefined;
    this.safeguardingSupervisorTerminalResult = undefined;
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      const settleReady = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve();
      };
      const fail = (terminal: "failed" | "unavailable") => {
        this.failSafeguardingSupervisor(record, terminal);
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(new HarnessGatewayError(503, "harness_worker_unavailable"));
      };
      record.abortStartup = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(new HarnessGatewayError(503, "harness_worker_unavailable"));
      };
      const timer = setTimeout(
        () => fail("unavailable"),
        this.config.harnessWorkerStartupMs
      );
      timer.unref();
      child.once("error", () => fail("unavailable"));
      child.stdin.once("error", () => fail("failed"));
      child.once("exit", () => {
        if (!record.stopping) fail(record.protocolFailed ? "failed" : "unavailable");
      });
      child.stderr.on("data", (chunk: Buffer) => {
        record.stderrBytes += chunk.byteLength;
        if (record.stderrBytes > MAX_STDERR_BYTES) fail("failed");
      });
      child.stdout.on("data", (chunk: Buffer) => {
        if (record.stopping || record.protocolFailed) return;
        record.stdout = Buffer.concat([record.stdout, chunk]);
        if (record.stdout.byteLength > MAX_SUPERVISOR_STDOUT_BUFFER_BYTES) {
          fail("failed");
          return;
        }
        while (true) {
          const newline = record.stdout.indexOf(0x0a);
          if (newline < 0) break;
          if (newline === 0 || newline > MAX_STATUS_BYTES) {
            fail("failed");
            return;
          }
          const line = record.stdout.subarray(0, newline);
          record.stdout = record.stdout.subarray(newline + 1);
          let parsed: unknown;
          try {
            parsed = JSON.parse(line.toString("utf8"));
          } catch {
            fail("failed");
            return;
          }
          const aggregate = safeSafeguardingSupervisorStatus(
            parsed,
            this.safeguardingSupervisorAggregate
          );
          if (!aggregate) {
            fail("failed");
            return;
          }
          this.safeguardingSupervisorAggregate = aggregate;
          this.safeguardingSupervisorReceivedAtMs = this.nowMs();
          settleReady();
        }
      });
      child.stdin.end(`${JSON.stringify(bootstrap)}\n`);
    });
  }

  private failSafeguardingSupervisor(
    record: SafeguardingSupervisorRecord,
    result: "failed" | "unavailable"
  ): void {
    if (this.safeguardingSupervisor !== record || record.stopping) return;
    record.protocolFailed ||= result === "failed";
    this.safeguardingSupervisorTerminalResult = result;
    if (record.child.exitCode === null && record.child.signalCode === null) {
      record.child.kill("SIGKILL");
    }
  }

  private async stopSafeguardingSupervisor(): Promise<void> {
    const record = this.safeguardingSupervisor;
    if (!record) return;
    record.stopping = true;
    record.abortStartup?.();
    if (record.child.exitCode !== null || record.child.signalCode !== null) return;
    const termSent = record.child.kill("SIGTERM");
    if (
      termSent &&
      await this.waitForChildExit(
        record.child,
        this.config.harnessWorkerShutdownMs
      )
    ) {
      return;
    }
    if (record.child.exitCode === null && record.child.signalCode === null) {
      record.child.kill("SIGKILL");
      await this.waitForChildExit(
        record.child,
        Math.min(1_000, this.config.harnessWorkerShutdownMs)
      );
    }
  }

  private async ensureRuntimeCanaryReady(): Promise<void> {
    const now = this.nowMs();
    if (this.runtimeCanaryCache && this.runtimeCanaryCache.expiresAtMs > now) {
      if (this.runtimeCanaryCache.ready) return;
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    if (this.runtimeCanaryPending) return this.runtimeCanaryPending;
    this.runtimeCanaryAttempts += 1;
    const run = this.runRuntimeCanary().then(
      () => {
        this.runtimeCanaryLastResult = "ready";
        this.runtimeCanarySuccesses += 1;
        this.runtimeCanaryCache = {
          ready: true,
          expiresAtMs: this.nowMs() + RUNTIME_CANARY_SUCCESS_TTL_MS
        };
      },
      () => {
        this.runtimeCanaryLastResult = "failed";
        this.runtimeCanaryFailures += 1;
        this.runtimeCanaryCache = {
          ready: false,
          expiresAtMs: this.nowMs() + RUNTIME_CANARY_FAILURE_TTL_MS
        };
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
    );
    const pending = run.finally(() => {
      if (this.runtimeCanaryPending === pending) {
        this.runtimeCanaryPending = undefined;
      }
    });
    this.runtimeCanaryPending = pending;
    return pending;
  }

  private async ensureProviderReadiness(): Promise<void> {
    if (!this.providerReadinessRequired()) return;
    const now = this.nowMs();
    if (
      this.providerReadinessCache &&
      this.providerReadinessCache.expiresAtMs > now
    ) {
      if (this.providerReadinessCache.ready) return;
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    if (this.providerReadinessPending) return this.providerReadinessPending;
    this.providerReadinessAttempts += 1;
    const run = this.runProviderReadiness().then(
      () => {
        this.providerReadinessLastResult = "ready";
        this.providerReadinessSuccesses += 1;
        this.providerReadinessCache = {
          ready: true,
          expiresAtMs: this.nowMs() + PROVIDER_READINESS_SUCCESS_TTL_MS
        };
      },
      () => {
        this.providerReadinessLastResult = "failed";
        this.providerReadinessFailures += 1;
        this.providerReadinessCache = {
          ready: false,
          expiresAtMs: this.nowMs() + PROVIDER_READINESS_FAILURE_TTL_MS
        };
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
    );
    const pending = run.finally(() => {
      if (this.providerReadinessPending === pending) {
        this.providerReadinessPending = undefined;
      }
    });
    this.providerReadinessPending = pending;
    return pending;
  }

  private async runProviderReadiness(): Promise<void> {
    const root = this.config.harnessWorkerRoot;
    const python = this.config.harnessWorkerPython;
    const cwd = this.config.harnessWorkerCwd;
    if (!root || !python || !cwd || this.shuttingDown) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    let readinessRoot: string | undefined;
    let child: ChildProcessWithoutNullStreams | undefined;
    let childConfirmedExited = false;
    try {
      const ephemeral = await this.createEphemeralWorkerBootstrap(
        ".provider-readiness-",
        "provider-readiness"
      );
      readinessRoot = ephemeral.root;
      if (this.providerReadinessOverride) {
        await this.providerReadinessOverride.run();
        childConfirmedExited = true;
        return;
      }
      child = spawn(
        python,
        [
          "-m",
          "teaching_skill_miner.teacher_agent_gateway_worker",
          "--provider-readiness"
        ],
        {
          cwd,
          env: {
            LANG: "C.UTF-8",
            LC_ALL: "C.UTF-8",
            PYTHONDONTWRITEBYTECODE: "1",
            PYTHONUNBUFFERED: "1"
          },
          stdio: ["pipe", "pipe", "pipe"]
        }
      );
      this.runtimeCanaryChildren.add(child);
      child.once("exit", () => this.runtimeCanaryChildren.delete(child!));
      await this.waitForProviderReadinessStatus(child, ephemeral.bootstrap);
      childConfirmedExited = true;
      if (
        this.shuttingDown ||
        child.exitCode !== 0 ||
        child.signalCode !== null
      ) {
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
    } catch {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    } finally {
      if (child && !childConfirmedExited) {
        childConfirmedExited = await this.forceStopReadinessChild(child);
      }
      if (readinessRoot) {
        if (childConfirmedExited || !child) {
          await this.removeEphemeralWorkerRoot(
            root,
            readinessRoot,
            ".provider-readiness-"
          );
        } else {
          child.once("exit", () => {
            void this.removeEphemeralWorkerRoot(
              root,
              readinessRoot!,
              ".provider-readiness-"
            ).catch(() => undefined);
          });
        }
      }
    }
  }

  private async waitForProviderReadinessStatus(
    child: ChildProcessWithoutNullStreams,
    bootstrap: WorkerBootstrap
  ): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      let stdout = Buffer.alloc(0);
      let stderrBytes = 0;
      const finish = (error?: HarnessGatewayError) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (error) reject(error);
        else resolve();
      };
      const fail = () => finish(
        new HarnessGatewayError(503, "harness_worker_unavailable")
      );
      const timer = setTimeout(
        fail,
        Math.min(
          PROVIDER_READINESS_STARTUP_MAX_MS,
          this.config.harnessWorkerStartupMs
        )
      );
      timer.unref();
      child.once("error", fail);
      child.stdin.once("error", fail);
      child.stderr.on("data", (chunk: Buffer) => {
        stderrBytes += chunk.byteLength;
        if (stderrBytes > MAX_STDERR_BYTES) fail();
      });
      child.stdout.on("data", (chunk: Buffer) => {
        if (settled) return;
        stdout = Buffer.concat([stdout, chunk]);
        if (stdout.byteLength > MAX_STATUS_BYTES) fail();
      });
      child.once("close", (code, signal) => {
        if (settled) return;
        const newline = stdout.indexOf(0x0a);
        if (
          code !== 0 ||
          signal !== null ||
          newline < 1 ||
          newline !== stdout.byteLength - 1
        ) {
          fail();
          return;
        }
        let parsed: unknown;
        try {
          parsed = JSON.parse(stdout.subarray(0, newline).toString("utf8"));
        } catch {
          fail();
          return;
        }
        if (
          bootstrap.agent_backend !== "deepseek" ||
          !safeProviderReadinessStatus(parsed)
        ) {
          fail();
          return;
        }
        finish();
      });
      child.stdin.end(`${JSON.stringify(bootstrap)}\n`);
    });
  }

  private async forceStopReadinessChild(
    child: ChildProcessWithoutNullStreams
  ): Promise<boolean> {
    if (child.exitCode !== null || child.signalCode !== null) return true;
    child.kill("SIGKILL");
    return this.waitForChildExit(child, PROVIDER_READINESS_SHUTDOWN_MAX_MS);
  }

  private async createEphemeralWorkerBootstrap(
    rootPrefix: ".runtime-canary-" | ".provider-readiness-",
    identityPrefix: "runtime-canary" | "provider-readiness"
  ): Promise<EphemeralWorkerBootstrap> {
    const root = this.config.harnessWorkerRoot;
    if (!root || this.shuttingDown) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    const ephemeralRoot = await mkdtemp(join(root, rootPrefix));
    try {
      await chmod(ephemeralRoot, 0o700);
      const keyVersion = this.config.harnessScopeKeyVersion;
      if (!/^k[1-9][0-9]{0,8}$/.test(keyVersion)) {
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
      const scopeId = `scope_${randomBytes(24).toString("hex")}`;
      const workerId = `worker_${randomBytes(16).toString("hex")}`;
      const privateRoot = join(ephemeralRoot, keyVersion, scopeId);
      await mkdir(privateRoot, {recursive: true, mode: 0o700});
      await chmod(join(ephemeralRoot, keyVersion), 0o700);
      await chmod(privateRoot, 0o700);
      const dataKey = randomBytes(32);
      const identity: ScopeIdentity = {
        key: `${identityPrefix}:${workerId}`,
        scopeId,
        keyVersion,
        privateRoot,
        dataKey,
        learnerScopeId: scopeId,
        authorityScopeBindings: [{scopeId, keyVersion}],
        safeguardingRouteLocator: this.config.harnessSafeguardingDispatchConfigured
          ? issueSafeguardingRouteLocator(
              {tenantId: scopeId, ownerId: workerId},
              this.config.accountScopeKeys?.[0] ?? this.config.harnessScopeKeys[0]!
            )
          : undefined
      };
      const capability = createHmac("sha256", dataKey)
        .update(
          `capability-v1\0${keyVersion}\0${scopeId}\0${workerId}`,
          "utf8"
        )
        .digest("base64url");
      return {
        root: ephemeralRoot,
        bootstrap: this.workerBootstrap(
          identity,
          workerId,
          capability,
          {
            policy_id: `${identityPrefix}-remote-denied`,
            policy_version: "canary-v1",
            policy_source: "organization_oidc_or_roster_policy",
            likely_minor: true,
            guardian_or_school_policy: "not_required",
            remote_processing_eligible: false
          }
        )
      };
    } catch (error) {
      await this.removeEphemeralWorkerRoot(root, ephemeralRoot, rootPrefix)
        .catch(() => undefined);
      throw error;
    }
  }

  private async runRuntimeCanary(): Promise<void> {
    const root = this.config.harnessWorkerRoot;
    const python = this.config.harnessWorkerPython;
    const cwd = this.config.harnessWorkerCwd;
    if (!root || !python || !cwd || this.shuttingDown) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    let canaryRoot: string | undefined;
    let child: ChildProcessWithoutNullStreams | undefined;
    let childConfirmedExited = false;
    try {
      const ephemeral = await this.createEphemeralWorkerBootstrap(
        ".runtime-canary-",
        "runtime-canary"
      );
      canaryRoot = ephemeral.root;
      const bootstrap = ephemeral.bootstrap;
      child = spawn(
        python,
        [
          "-m",
          "teaching_skill_miner.teacher_agent_gateway_worker",
          "--runtime-canary"
        ],
        {
          cwd,
          env: {
            LANG: "C.UTF-8",
            LC_ALL: "C.UTF-8",
            PYTHONDONTWRITEBYTECODE: "1",
            PYTHONUNBUFFERED: "1"
          },
          stdio: ["pipe", "pipe", "pipe"]
        }
      );
      this.runtimeCanaryChildren.add(child);
      child.once("exit", () => this.runtimeCanaryChildren.delete(child!));
      await this.waitForRuntimeCanaryStatus(child, bootstrap);
      if (this.shuttingDown) {
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
      const termSent = child.kill("SIGTERM");
      if (
        !termSent &&
        child.exitCode === null &&
        child.signalCode === null
      ) {
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
      childConfirmedExited = await this.waitForChildExit(
        child,
        Math.min(
          RUNTIME_CANARY_SHUTDOWN_MAX_MS,
          this.config.harnessWorkerShutdownMs
        )
      );
      if (
        !childConfirmedExited ||
        child.exitCode !== 0 ||
        child.signalCode !== null
      ) {
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
    } catch {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    } finally {
      if (child && !childConfirmedExited) {
        childConfirmedExited = await this.forceStopCanary(child);
      }
      if (canaryRoot) {
        if (childConfirmedExited || !child) {
          await this.removeEphemeralWorkerRoot(
            root,
            canaryRoot,
            ".runtime-canary-"
          );
        } else {
          // Do not remove an allowlisted root out from under an uncertain live
          // process. The exit hook performs the same bounded-scope cleanup.
          child.once("exit", () => {
            void this.removeEphemeralWorkerRoot(
              root,
              canaryRoot!,
              ".runtime-canary-"
            ).catch(() => undefined);
          });
        }
      }
    }
  }

  private async waitForRuntimeCanaryStatus(
    child: ChildProcessWithoutNullStreams,
    bootstrap: WorkerBootstrap
  ): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      let stdout = Buffer.alloc(0);
      let stderrBytes = 0;
      const fail = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(new HarnessGatewayError(503, "harness_worker_unavailable"));
      };
      const timer = setTimeout(
        fail,
        Math.min(
          RUNTIME_CANARY_STARTUP_MAX_MS,
          this.config.harnessWorkerStartupMs
        )
      );
      timer.unref();
      child.once("error", fail);
      child.stdin.once("error", fail);
      child.once("exit", fail);
      child.stderr.on("data", (chunk: Buffer) => {
        stderrBytes += chunk.byteLength;
        if (stderrBytes > MAX_STDERR_BYTES) fail();
      });
      child.stdout.on("data", (chunk: Buffer) => {
        if (settled) return;
        stdout = Buffer.concat([stdout, chunk]);
        if (stdout.byteLength > MAX_STATUS_BYTES) {
          fail();
          return;
        }
        const newline = stdout.indexOf(0x0a);
        if (newline < 0) return;
        let parsed: unknown;
        try {
          parsed = JSON.parse(stdout.subarray(0, newline).toString("utf8"));
        } catch {
          fail();
          return;
        }
        if (
          stdout.subarray(newline + 1).length ||
          !safeRuntimeCanaryStatus(parsed, {
            workerId: bootstrap.worker_id,
            keyVersion: bootstrap.scope_key_version,
            backend: bootstrap.agent_backend,
            filesystemIsolationRequired: bootstrap.filesystem_isolation_required,
            processResourceLimits: bootstrap.process_resource_limits
          })
        ) {
          fail();
          return;
        }
        settled = true;
        clearTimeout(timer);
        resolve();
      });
      child.stdin.end(`${JSON.stringify(bootstrap)}\n`);
    });
  }

  private async waitForChildExit(
    child: ChildProcessWithoutNullStreams,
    timeoutMs: number
  ): Promise<boolean> {
    if (child.exitCode !== null || child.signalCode !== null) return true;
    return new Promise<boolean>((resolve) => {
      let settled = false;
      const finish = (value: boolean) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        child.removeListener("exit", exited);
        resolve(value);
      };
      const exited = () => finish(true);
      const timer = setTimeout(() => finish(false), Math.max(1, timeoutMs));
      timer.unref();
      child.once("exit", exited);
    });
  }

  private async forceStopCanary(
    child: ChildProcessWithoutNullStreams
  ): Promise<boolean> {
    if (child.exitCode !== null || child.signalCode !== null) return true;
    child.kill("SIGKILL");
    return this.waitForChildExit(child, RUNTIME_CANARY_SHUTDOWN_MAX_MS);
  }

  private async removeEphemeralWorkerRoot(
    workerRoot: string,
    ephemeralRoot: string,
    prefix: ".runtime-canary-" | ".provider-readiness-"
  ): Promise<void> {
    if (
      dirname(ephemeralRoot) !== resolve(workerRoot) ||
      !basename(ephemeralRoot).startsWith(prefix)
    ) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    await rm(ephemeralRoot, {recursive: true, force: true});
  }

  private async scopeIdentity(
    scope: AccessScope,
    create: boolean
  ): Promise<ScopeIdentity> {
    await this.ensureBaseReady();
    const root = this.config.harnessWorkerRoot;
    if (!root) throw new HarnessGatewayError(503, "harness_worker_unavailable");
    const binding = canonicalScopeBinding(scope);
    const candidates = this.config.harnessScopeKeys.map((scopeKey) =>
      this.identityForKey(root, binding, scopeKey)
    );
    const routingKey = (
      this.config.accountScopeKeys?.[0] ?? this.config.harnessScopeKeys[0]
    );
    const safeguardingRouteLocator =
      this.config.harnessSafeguardingDispatchConfigured && routingKey
        ? issueSafeguardingRouteLocator(scope, routingKey)
        : undefined;
    if (create) {
      try {
        return {
          ...(await ensureActiveScopeRoot(root, candidates)),
          safeguardingRouteLocator
        };
      } catch {
        // Migration errors can contain private paths or OS details. They are
        // intentionally collapsed before reaching health output or HTTP.
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
    }
    for (const candidate of candidates) {
      try {
        const metadata = await lstat(candidate.privateRoot);
        if (
          metadata.isSymbolicLink() ||
          !metadata.isDirectory() ||
          !isPrivateMode(metadata.mode)
        ) {
          throw new HarnessGatewayError(503, "harness_worker_unavailable");
        }
        return {...candidate, safeguardingRouteLocator};
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
    }
    throw new HarnessGatewayError(503, "harness_worker_unavailable");
  }

  private identityForKey(
    root: string,
    binding: string,
    scopeKey: HarnessScopeKey
  ): ScopeIdentity {
    const scopeId = `scope_${hmac(scopeKey.secret, binding).toString("hex").slice(0, 48)}`;
    const dataKey = hmac(
      scopeKey.secret,
      `worker-data-v1\0${scopeKey.version}\0${scopeId}`
    );
    return {
      key: `${scopeKey.version}:${scopeId}`,
      scopeId,
      keyVersion: scopeKey.version,
      privateRoot: join(root, scopeKey.version, scopeId),
      dataKey
    };
  }

  private async acquire(
    scope: AccessScope,
    subjectPolicy: RemoteSubjectPolicy | undefined,
    preserveRemoteSubjectPolicy = false
  ): Promise<WorkerRecord> {
    const policy = subjectPolicy ?? {
      policy_id: "local-development-denied",
      policy_version: "local-v1",
      policy_source: "organization_oidc_or_roster_policy" as const,
      likely_minor: true,
      guardian_or_school_policy: "not_required" as const,
      remote_processing_eligible: false
    };
    const policyHash = remotePolicySha256(policy);
    const coordinationKey = this.accountCoordinationKey(scope);
    const selection = await this.withStartGate(async () => {
      if (this.accountBlocks.has(coordinationKey)) {
        throw new HarnessGatewayError(503, "harness_scope_fenced");
      }
      // Root selection/migration and worker selection share the same local
      // start gate. The migration helper additionally takes a cross-process,
      // opaque scope lock before touching a retained previous-key root.
      const identity = await this.scopeIdentity(scope, true);
      const current = this.workers.get(identity.key);
      if (
        current &&
        current.child.exitCode === null &&
        current.child.signalCode === null &&
        !current.stopping
      ) {
        if (!preserveRemoteSubjectPolicy && current.remoteSubjectPolicyHash !== policyHash) {
          // Never let a newly signed subject policy inherit a stale worker.
          // An in-flight worker is fenced until its caller releases it; an
          // idle worker is fully stopped before a replacement can be spawned.
          if (current.activeRequests > 0) {
            throw new HarnessGatewayError(503, "harness_worker_unavailable");
          }
          this.workers.delete(identity.key);
          this.retiring.add(current);
          const stopped = await this.stopWorker(current);
          if (!stopped) {
            throw new HarnessGatewayError(503, "harness_worker_unavailable");
          }
          this.retiring.delete(current);
        } else {
          this.reserveRequest(current);
          return {identity, worker: current};
        }
      }
      if (current) this.workers.delete(identity.key);
      const pending = this.starting.get(identity.key);
      if (pending) {
        if (
          !preserveRemoteSubjectPolicy
          && this.startingPolicyHashes.get(identity.key) !== policyHash
        ) {
          throw new HarnessGatewayError(503, "harness_worker_unavailable");
        }
        return {identity, startup: pending};
      }
      if (this.physicalWorkerCount() >= this.config.harnessMaxWorkers) {
        await this.evictExpiredIdleWorker();
      }
      if (this.physicalWorkerCount() >= this.config.harnessMaxWorkers) {
        throw new HarnessGatewayError(503, "harness_capacity_exhausted");
      }
      const startup = this.startWorker(identity, policy, policyHash).finally(() => {
        this.starting.delete(identity.key);
        this.startingPolicyHashes.delete(identity.key);
      });
      this.starting.set(identity.key, startup);
      this.startingPolicyHashes.set(identity.key, policyHash);
      return {identity, startup};
    });
    if (selection.worker) return selection.worker;
    const worker = await selection.startup;
    const {identity} = selection;
    return this.withStartGate(() => {
      if (this.accountBlocks.has(coordinationKey)) {
        throw new HarnessGatewayError(503, "harness_scope_fenced");
      }
      if (
        worker.stopping ||
        worker.child.exitCode !== null ||
        worker.child.signalCode !== null ||
        this.workers.get(identity.key)?.child !== worker.child
      ) {
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
      this.reserveRequest(worker);
      return worker;
    });
  }

  private nowMs(): number {
    const value = this.clock();
    if (!Number.isFinite(value) || value < 0) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    return value;
  }

  private accountCoordinationKey(scope: AccessScope): string {
    const active = this.config.harnessScopeKeys[0];
    if (!active) throw new HarnessGatewayError(503, "harness_worker_unavailable");
    return hmac(active.secret, `account-coordination-v1\0${canonicalScopeBinding(scope)}`)
      .toString("hex");
  }

  private async stopScopeForAccount(
    scope: AccessScope,
    timeoutMs: number,
    cancelOnTimeout: boolean
  ): Promise<number> {
    if (!Number.isInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 60_000) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    const identities = await this.accountRoots(scope);
    const deadline = this.nowMs() + timeoutMs;
    let cancelled = 0;
    for (const identity of identities) {
      const identityKey = `${identity.keyVersion}:${identity.scopeId}`;
      const startup = this.starting.get(identityKey);
      if (startup) await startup.catch(() => undefined);
      const record = this.workers.get(identityKey);
      if (!record) continue;
      while (record.activeRequests > 0 && this.nowMs() < deadline) {
        await new Promise<void>((resolve) => {
          const timer = setTimeout(resolve, 25);
          timer.unref();
        });
      }
      if (record.activeRequests > 0 && !cancelOnTimeout) {
        throw new HarnessGatewayError(503, "harness_worker_unavailable");
      }
      cancelled += record.activeRequests;
      await this.withStartGate(() => {
        if (this.workers.get(identityKey) === record) {
          this.workers.delete(identityKey);
          this.retiring.add(record);
        }
      });
      const stopped = await this.stopWorker(record);
      if (!stopped) throw new HarnessGatewayError(503, "harness_worker_unavailable");
      this.retiring.delete(record);
    }
    return cancelled;
  }

  private reserveRequest(worker: WorkerRecord): void {
    worker.activeRequests += 1;
    worker.lastActivityMs = Math.max(worker.lastActivityMs, this.nowMs());
  }

  private physicalWorkerCount(): number {
    return this.workers.size + this.starting.size + this.retiring.size;
  }

  private isIdleEvictionCandidate(worker: WorkerRecord, now: number): boolean {
    return (
      worker.activeRequests === 0 &&
      !worker.stopping &&
      worker.child.exitCode === null &&
      worker.child.signalCode === null &&
      now - worker.lastActivityMs >= this.config.harnessWorkerIdleTimeoutMs
    );
  }

  private async withStartGate<T>(operation: () => Promise<T> | T): Promise<T> {
    let release!: () => void;
    const previous = this.startGate;
    this.startGate = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }

  private async evictExpiredIdleWorker(): Promise<void> {
    const now = this.nowMs();
    const candidate = [...this.workers.entries()]
      .filter(([, worker]) => this.isIdleEvictionCandidate(worker, now))
      .sort(
        ([leftKey, left], [rightKey, right]) =>
          left.lastActivityMs - right.lastActivityMs ||
          leftKey.localeCompare(rightKey)
      )[0];
    if (!candidate) {
      throw new HarnessGatewayError(503, "harness_capacity_exhausted");
    }
    const [identityKey, worker] = candidate;
    if (
      this.workers.get(identityKey) !== worker ||
      !this.isIdleEvictionCandidate(worker, this.nowMs())
    ) {
      throw new HarnessGatewayError(503, "harness_capacity_exhausted");
    }
    // Atomically make the worker unavailable before signalling it. Its opaque
    // private root is deliberately untouched and will be reused on restart.
    this.workers.delete(identityKey);
    this.retiring.add(worker);
    const stopped = await this.stopWorker(worker);
    if (!stopped) {
      // Keep an uncertain process charged against capacity. A later confirmed
      // exit removes it through handleWorkerExit; this request never reuses the
      // unproven slot.
      throw new HarnessGatewayError(503, "harness_capacity_exhausted");
    }
    this.retiring.delete(worker);
  }

  private handleWorkerExit(
    identityKey: string,
    child: ChildProcessWithoutNullStreams
  ): void {
    const installed = this.workers.get(identityKey);
    if (installed?.child === child) this.workers.delete(identityKey);
    for (const worker of this.retiring) {
      if (worker.child === child) this.retiring.delete(worker);
    }
  }

  private workerBootstrap(
    identity: ScopeIdentity,
    workerId: string,
    capability: string,
    subjectPolicy: RemoteSubjectPolicy
  ): WorkerBootstrap {
    return {
      schema: BOOTSTRAP_SCHEMA,
      scope_id: identity.scopeId,
      scope_key_version: identity.keyVersion,
      worker_id: workerId,
      capability_token: capability,
      private_root: identity.privateRoot,
      scope_key_material: identity.dataKey.toString("base64"),
      learner_scope_id: identity.learnerScopeId ?? identity.scopeId,
      authority_scope_bindings: (identity.authorityScopeBindings ?? [{
        scopeId: identity.scopeId,
        keyVersion: identity.keyVersion
      }]).map((binding) => ({
        scope_id: binding.scopeId,
        key_version: binding.keyVersion
      })),
      agent_backend: this.config.harnessWorkerBackend,
      api_key_file:
        this.config.harnessWorkerBackend === "deepseek"
          ? this.config.harnessProviderApiKeyFile ?? null
          : null,
      remote_provider_policy: this.config.harnessRemoteProviderPolicy ?? null,
      remote_subject_policy: subjectPolicy,
      safeguarding_locale: this.config.harnessSafeguardingLocale || "zh-CN",
      safeguarding_dispatcher:
        this.config.harnessSafeguardingDispatchConfigured
        && this.config.harnessSafeguardingDispatchUrl
        && this.config.harnessSafeguardingDispatchBearerSecret
        && identity.safeguardingRouteLocator
          ? {
              endpoint: this.config.harnessSafeguardingDispatchUrl,
              bearer_secret: this.config.harnessSafeguardingDispatchBearerSecret,
              route_locator: identity.safeguardingRouteLocator,
              policy_version:
                this.config.harnessSafeguardingDispatchPolicyVersion,
              timeout_ms: this.config.harnessSafeguardingDispatchTimeoutMs,
              maximum_response_bytes:
                this.config.harnessSafeguardingDispatchMaxResponseBytes
            }
          : null,
      filesystem_isolation_required:
        this.config.harnessWorkerFilesystemIsolationRequired,
      process_resource_limits:
        this.config.harnessWorkerProcessResourceLimits ?? null
    };
  }

  private async startWorker(
    identity: ScopeIdentity,
    subjectPolicy: RemoteSubjectPolicy,
    subjectPolicyHash: string
  ): Promise<WorkerRecord> {
    const python = this.config.harnessWorkerPython;
    const cwd = this.config.harnessWorkerCwd;
    if (!python || !cwd || this.shuttingDown) {
      throw new HarnessGatewayError(503, "harness_worker_unavailable");
    }
    const workerId = `worker_${randomBytes(16).toString("hex")}`;
    const capability = createHmac("sha256", identity.dataKey)
      .update(
        `capability-v1\0${identity.keyVersion}\0${identity.scopeId}\0${workerId}`,
        "utf8"
      )
      .digest("base64url");
    const child = spawn(
      python,
      ["-m", "teaching_skill_miner.teacher_agent_gateway_worker"],
      {
        cwd,
        env: {
          LANG: "C.UTF-8",
          LC_ALL: "C.UTF-8",
          PYTHONDONTWRITEBYTECODE: "1",
          PYTHONUNBUFFERED: "1"
        },
        stdio: ["pipe", "pipe", "pipe"]
      }
    );
    const bootstrap = this.workerBootstrap(
      identity,
      workerId,
      capability,
      subjectPolicy
    );

    return new Promise<WorkerRecord>((resolve, reject) => {
      let settled = false;
      let stdout = Buffer.alloc(0);
      let stderrBytes = 0;
      const rejectSafe = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        child.kill("SIGKILL");
        reject(new HarnessGatewayError(503, "harness_worker_unavailable"));
      };
      const timer = setTimeout(rejectSafe, this.config.harnessWorkerStartupMs);
      timer.unref();
      child.once("error", rejectSafe);
      child.stdin.once("error", rejectSafe);
      child.once("exit", () => {
        this.handleWorkerExit(identity.key, child);
        rejectSafe();
      });
      child.stderr.on("data", (chunk: Buffer) => {
        stderrBytes += chunk.byteLength;
        if (stderrBytes > MAX_STDERR_BYTES) rejectSafe();
      });
      child.stdout.on("data", (chunk: Buffer) => {
        if (settled) return;
        stdout = Buffer.concat([stdout, chunk]);
        if (stdout.byteLength > MAX_STATUS_BYTES) {
          rejectSafe();
          return;
        }
        const newline = stdout.indexOf(0x0a);
        if (newline < 0) return;
        let parsed: unknown;
        try {
          parsed = JSON.parse(stdout.subarray(0, newline).toString("utf8"));
        } catch {
          rejectSafe();
          return;
        }
        const status = safeWorkerStatus(parsed, {
          workerId,
          keyVersion: identity.keyVersion,
          backend: this.config.harnessWorkerBackend,
          filesystemIsolationRequired:
            this.config.harnessWorkerFilesystemIsolationRequired,
          processResourceLimits: bootstrap.process_resource_limits
        });
        if (!status || stdout.subarray(newline + 1).length) {
          rejectSafe();
          return;
        }
        settled = true;
        clearTimeout(timer);
        const record: WorkerRecord = {
          identityKey: identity.key,
          capability,
          port: status.port,
          child,
          requestNamespaceKey: hmac(identity.dataKey, "request-namespace-v1"),
          teacherAuthorityKey: hmac(
            identity.dataKey,
            "teachlab-gateway-worker-v1\0teacher-authority-v1"
          ),
          scopeId: identity.scopeId,
          scopeKeyVersion: identity.keyVersion,
          activeRequests: 0,
          lastActivityMs: this.nowMs(),
          stopping: false
          ,remoteSubjectPolicyHash: subjectPolicyHash
        };
        this.starting.delete(identity.key);
        this.workers.set(identity.key, record);
        resolve(record);
      });
      child.stdin.end(`${JSON.stringify(bootstrap)}\n`);
    });
  }

  private scopedRequestBody(
    worker: WorkerRecord,
    path: string,
    body: Buffer | undefined,
    teacherRoles: readonly string[] | undefined,
    teacherAllowedRoles: readonly string[] | undefined,
    teacherPrincipal: Readonly<{issuer: string; subject: string}> | undefined
  ): Buffer | undefined {
    const teacherAuthorityPath = TEACHER_AUTHORITY_PATHS.has(path);
    if (!body || (!teacherAuthorityPath && path !== "api/stream" && path !== "api/cancel")) {
      return body;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(body.toString("utf8"));
    } catch {
      throw new HarnessGatewayError(400, "harness_request_invalid");
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new HarnessGatewayError(400, "harness_request_invalid");
    }
    const envelope = {...(parsed as Record<string, unknown>)};
    if (teacherAuthorityPath) {
      if (
        !teacherRoles || !teacherPrincipal ||
        requestContainsAuthorityOverride(envelope)
      ) {
        throw new HarnessGatewayError(400, "harness_request_invalid");
      }
      try {
        envelope._teacher_authority = teacherAuthorityEnvelope(path, envelope, {
          scopeId: worker.scopeId,
          scopeKeyVersion: worker.scopeKeyVersion,
          authorityKey: worker.teacherAuthorityKey,
          principalIssuer: teacherPrincipal.issuer,
          principalSubject: teacherPrincipal.subject,
          roles: teacherRoles,
          allowedRoles: teacherAllowedRoles ?? this.config.teacherAuthorityRoles,
          ttlSeconds: this.config.teacherAuthorityTtlSeconds
        });
      } catch {
        throw new HarnessGatewayError(400, "harness_request_invalid");
      }
    }
    if (envelope.request_id !== undefined) {
      if (
        typeof envelope.request_id !== "string" ||
        !envelope.request_id.trim() ||
        envelope.request_id.length > 160
      ) {
        throw new HarnessGatewayError(400, "harness_request_invalid");
      }
      const digest = createHmac("sha256", worker.requestNamespaceKey)
        .update(envelope.request_id.trim(), "utf8")
        .digest("hex")
        .slice(0, 48);
      envelope.request_id = `gateway_${digest}`;
    } else if (path === "api/stream" && envelope.run_id === undefined) {
      throw new HarnessGatewayError(400, "harness_request_invalid");
    }
    return Buffer.from(JSON.stringify(envelope), "utf8");
  }

  private async stopWorker(record: WorkerRecord): Promise<boolean> {
    if (record.child.exitCode !== null || record.child.signalCode !== null) {
      return true;
    }
    record.stopping = true;
    let exited = false;
    const exit = new Promise<void>((resolve) => {
      record.child.once("exit", () => {
        exited = true;
        resolve();
      });
    });
    const wait = (milliseconds: number) =>
      new Promise<boolean>((resolve) => {
        const timer = setTimeout(() => resolve(false), milliseconds);
        timer.unref();
        void exit.then(() => {
          clearTimeout(timer);
          resolve(true);
        });
      });
    const termSent = record.child.kill("SIGTERM");
    if (
      !termSent &&
      record.child.exitCode === null &&
      record.child.signalCode === null
    ) {
      return false;
    }
    if (exited || (await wait(this.config.harnessWorkerShutdownMs))) return true;
    const killSent = record.child.kill("SIGKILL");
    if (
      !killSent &&
      record.child.exitCode === null &&
      record.child.signalCode === null
    ) {
      return false;
    }
    return exited || (await wait(Math.min(1_000, this.config.harnessWorkerShutdownMs)));
  }
}
