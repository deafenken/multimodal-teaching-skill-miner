import {randomBytes} from "node:crypto";
import {isAbsolute} from "node:path";

import {Injectable} from "@nestjs/common";

import {privateSecret} from "./private-secret";
import type {RemoteProviderPolicy} from "../auth/remote-processing-policy";

export type AuthMode = "development" | "oidc";
export type DataBackend = "memory" | "postgres";
export type HarnessAgentBackend = "deterministic" | "deepseek";

export interface HarnessWorkerProcessResourceLimits {
  schema: "teaching_skill_miner.worker_process_resource_limits.v1";
  address_space_bytes: number;
  file_size_bytes: number;
  open_files: number;
  core_dump_bytes: 0;
}

const ROLE_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;
const POLICY_COMPONENT_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{1,119}$/;
const CLAIM_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$/;

export interface HarnessScopeKey {
  version: string;
  secret: string;
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  const normalized = value.trim().toLowerCase();
  if (normalized === "true") return true;
  if (normalized === "false") return false;
  throw new Error(`Expected a boolean, received ${JSON.stringify(value)}`);
}

function parseInteger(
  name: string,
  value: string | undefined,
  fallback: number,
  minimum: number,
  maximum: number
): number {
  if (value === undefined || value.trim() === "") return fallback;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}`);
  }
  return parsed;
}

function parseExactInteger(
  name: string,
  value: string | undefined,
  fallback: number,
  minimum: number,
  maximum: number
): number {
  if (value !== undefined && !/^(?:0|[1-9][0-9]*)$/.test(value)) {
    throw new Error(`${name} must be an exact base-10 integer`);
  }
  return parseInteger(name, value, fallback, minimum, maximum);
}

function splitList(value: string | undefined, fallback: string[]): string[] {
  const items = (value ?? fallback.join(","))
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return [...new Set(items)];
}

function requiredHttpsUrl(name: string, value: string | undefined): string {
  if (!value?.trim()) throw new Error(`${name} is required`);
  let parsed: URL;
  try {
    parsed = new URL(value.trim());
  } catch {
    throw new Error(`${name} must be a valid URL`);
  }
  if (parsed.protocol !== "https:") throw new Error(`${name} must use HTTPS`);
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error(`${name} cannot contain credentials, query parameters, or fragments`);
  }
  return parsed.toString().replace(/\/$/, "");
}

function requiredDefaultPortHttpsUrl(
  name: string,
  value: string | undefined
): string {
  const canonical = requiredHttpsUrl(name, value);
  const parsed = new URL(canonical);
  if (parsed.port && parsed.port !== "443") {
    throw new Error(`${name} must use the default HTTPS port`);
  }
  return canonical;
}

function optionalExact(name: string, value: string | undefined): string | undefined {
  if (value === undefined || value === "") return undefined;
  if (value !== value.trim() || /[\u0000-\u001f\u007f]/.test(value)) {
    throw new Error(`${name} must be an exact non-control string`);
  }
  return value;
}

function providerPolicyFromEnvironment(): RemoteProviderPolicy | null {
  const values = {
    policy_id: optionalExact("HARNESS_PROVIDER_POLICY_ID", process.env.HARNESS_PROVIDER_POLICY_ID),
    policy_version: optionalExact(
      "HARNESS_PROVIDER_POLICY_VERSION",
      process.env.HARNESS_PROVIDER_POLICY_VERSION
    ),
    processing_region: optionalExact(
      "HARNESS_PROVIDER_PROCESSING_REGION",
      process.env.HARNESS_PROVIDER_PROCESSING_REGION
    ),
    retention: optionalExact(
      "HARNESS_PROVIDER_RETENTION_DAYS",
      process.env.HARNESS_PROVIDER_RETENTION_DAYS
    ),
    deletion_status: optionalExact(
      "HARNESS_PROVIDER_DELETION_STATUS",
      process.env.HARNESS_PROVIDER_DELETION_STATUS
    ),
    documentation_url: optionalExact(
      "HARNESS_PROVIDER_DOCUMENTATION_URL",
      process.env.HARNESS_PROVIDER_DOCUMENTATION_URL
    )
  };
  const supplied = Object.values(values).filter((value) => value !== undefined).length;
  if (supplied === 0) return null;
  if (supplied !== 6) {
    throw new Error("HARNESS_PROVIDER_POLICY_* must be supplied as one complete policy");
  }
  if (
    !POLICY_COMPONENT_PATTERN.test(values.policy_id!)
    || !POLICY_COMPONENT_PATTERN.test(values.policy_version!)
    || !/^[A-Za-z0-9_.-]{2,40}$/.test(values.processing_region!)
  ) throw new Error("HARNESS provider policy identifiers are invalid");
  const retention = Number(values.retention);
  if (!/^(?:0|[1-9][0-9]{0,2})$/.test(values.retention!) || retention > 365) {
    throw new Error("HARNESS_PROVIDER_RETENTION_DAYS must be an integer from 0 to 365");
  }
  if (!new Set([
    "outside_service_control_subject_to_provider_policy",
    "provider_documents_zero_retention"
  ]).has(values.deletion_status!)) {
    throw new Error("HARNESS_PROVIDER_DELETION_STATUS is invalid");
  }
  const documentationUrl = requiredHttpsUrl(
    "HARNESS_PROVIDER_DOCUMENTATION_URL",
    values.documentation_url
  );
  return {
    policy_id: values.policy_id!,
    policy_version: values.policy_version!,
    policy_source: "deployment_operator_asserted_external_terms_not_repository_verified",
    processing_region: values.processing_region!,
    provider_retention_days: retention,
    deletion_status: values.deletion_status! as RemoteProviderPolicy["deletion_status"],
    documentation_url: documentationUrl
  };
}

function parseVersionedSecretKeys(
  activeVersion: string,
  activeSecret: string,
  previousJson: string | undefined,
  variableName: string
): HarnessScopeKey[] {
  const keys: HarnessScopeKey[] = [{version: activeVersion, secret: activeSecret}];
  if (!previousJson?.trim()) return keys;
  let parsed: unknown;
  try {
    parsed = JSON.parse(previousJson);
  } catch {
    throw new Error(`${variableName} must be a JSON object`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${variableName} must be a JSON object`);
  }
  for (const [version, secret] of Object.entries(parsed)) {
    if (typeof secret !== "string") {
      throw new Error(`${variableName} values must be strings`);
    }
    keys.push({version, secret});
  }
  return keys;
}

function isLoopbackHost(host: string): boolean {
  return new Set(["127.0.0.1", "::1", "localhost"]).has(host.toLowerCase());
}

@Injectable()
export class AppConfigService {
  readonly nodeEnv = process.env.NODE_ENV ?? "development";
  private readonly production = this.nodeEnv === "production";
  readonly host = process.env.API_HOST?.trim() || "127.0.0.1";
  readonly port = parseInteger("API_PORT", process.env.API_PORT, 4000, 1, 65_535);
  readonly authMode = this.parseAuthMode(process.env.AUTH_MODE);
  readonly dataBackend = this.parseDataBackend(process.env.DATA_BACKEND);
  readonly developmentUserId = process.env.DEV_AUTH_USER_ID?.trim() || "local-developer";
  readonly developmentTenantId =
    process.env.DEV_AUTH_TENANT_ID?.trim() || "local-tenant";
  readonly allowDevelopmentAuthHeaders = parseBoolean(
    process.env.DEV_AUTH_ALLOW_HEADERS,
    false
  );
  readonly seedDemoSessions = parseBoolean(
    process.env.SEED_DEMO_SESSIONS,
    this.nodeEnv !== "production"
  );
  readonly sessionTtlSeconds = parseInteger(
    "SESSION_TTL_SECONDS",
    process.env.SESSION_TTL_SECONDS,
    8 * 60 * 60,
    300,
    7 * 24 * 60 * 60
  );
  readonly secureSessionCookies = parseBoolean(
    process.env.SESSION_COOKIE_SECURE,
    this.nodeEnv === "production"
  );
  readonly sessionCookieName = this.secureSessionCookies
    ? "__Host-teachlab_session"
    : "teachlab_session";
  readonly csrfCookieName = this.secureSessionCookies
    ? "__Host-teachlab_csrf"
    : "teachlab_csrf";
  readonly oidcTransactionCookieName = this.secureSessionCookies
    ? "__Secure-teachlab_oidc_tx"
    : "teachlab_oidc_tx";
  readonly sessionSecret = privateSecret({
    name: "SESSION_SECRET",
    file: process.env.SESSION_SECRET_FILE,
    inline: process.env.SESSION_SECRET,
    production: this.production,
    required: this.production,
  }) ?? randomBytes(32).toString("base64url");
  readonly previousSessionSecret = privateSecret({
    name: "SESSION_PREVIOUS_SECRET",
    file: process.env.SESSION_PREVIOUS_SECRET_FILE,
    inline: process.env.SESSION_PREVIOUS_SECRET,
    production: this.production,
    required: false,
  });
  readonly sessionVerificationSecrets = [
    this.sessionSecret,
    ...(this.previousSessionSecret ? [this.previousSessionSecret] : [])
  ];
  readonly databaseUrl = privateSecret({
    name: "DATABASE_URL",
    file: process.env.DATABASE_URL_FILE,
    inline: process.env.DATABASE_URL,
    production: this.production,
    required: this.production && this.dataBackend === "postgres",
    minimumBytes: 12,
    maximumBytes: 4_096,
    pattern: /^postgres(?:ql)?:\/\/[^\s]+$/,
  });
  readonly postgresSslMode = process.env.PG_SSL_MODE?.trim() || "require";
  readonly postgresPoolMax = parseInteger(
    "PG_POOL_MAX",
    process.env.PG_POOL_MAX,
    10,
    1,
    100
  );
  readonly sseHeartbeatMs = parseInteger(
    "SSE_HEARTBEAT_MS",
    process.env.SSE_HEARTBEAT_MS,
    15_000,
    1_000,
    60_000
  );
  readonly eventPollMs = parseInteger(
    "EVENT_POLL_MS",
    process.env.EVENT_POLL_MS,
    1_000,
    100,
    30_000
  );
  readonly taskDispatchPollMs = parseInteger(
    "TASK_DISPATCH_POLL_MS",
    process.env.TASK_DISPATCH_POLL_MS,
    250,
    50,
    30_000
  );
  readonly taskDispatchBatchSize = parseInteger(
    "TASK_DISPATCH_BATCH_SIZE",
    process.env.TASK_DISPATCH_BATCH_SIZE,
    4,
    1,
    32
  );
  readonly taskLeaseDurationMs = parseInteger(
    "TASK_LEASE_DURATION_MS",
    process.env.TASK_LEASE_DURATION_MS,
    30_000,
    1_000,
    5 * 60_000
  );
  readonly taskMaximumAttempts = parseInteger(
    "TASK_MAXIMUM_ATTEMPTS",
    process.env.TASK_MAXIMUM_ATTEMPTS,
    3,
    1,
    20
  );
  readonly taskRetryBaseMs = parseInteger(
    "TASK_RETRY_BASE_MS",
    process.env.TASK_RETRY_BASE_MS,
    2_000,
    100,
    60_000
  );
  readonly taskRetryMaximumMs = parseInteger(
    "TASK_RETRY_MAXIMUM_MS",
    process.env.TASK_RETRY_MAXIMUM_MS,
    60_000,
    this.taskRetryBaseMs,
    10 * 60_000
  );
  readonly corsOrigins = splitList(process.env.CORS_ORIGINS, ["http://localhost:3000"]);
  readonly oidcIssuer = process.env.OIDC_ISSUER?.trim().replace(/\/$/, "");
  readonly oidcAudience = process.env.OIDC_AUDIENCE?.trim();
  readonly oidcClientId = process.env.OIDC_CLIENT_ID?.trim();
  readonly oidcClientSecret = privateSecret({
    name: "OIDC_CLIENT_SECRET",
    file: process.env.OIDC_CLIENT_SECRET_FILE,
    inline: process.env.OIDC_CLIENT_SECRET,
    production: this.production,
    required: this.production && this.authMode === "oidc",
    minimumBytes: 16,
  });
  readonly oidcRedirectUri = process.env.OIDC_REDIRECT_URI?.trim();
  readonly oidcTransactionSecret = privateSecret({
    name: "OIDC_TRANSACTION_SECRET",
    file: process.env.OIDC_TRANSACTION_SECRET_FILE,
    inline: process.env.OIDC_TRANSACTION_SECRET,
    production: this.production,
    required: this.production && this.authMode === "oidc",
  });
  readonly oidcExpectedHost = process.env.OIDC_EXPECTED_HOST?.trim()
    || `${this.host}:${this.port}`;
  readonly oidcTransactionTtlSeconds = parseInteger(
    "OIDC_TRANSACTION_TTL_SECONDS",
    process.env.OIDC_TRANSACTION_TTL_SECONDS,
    5 * 60,
    60,
    10 * 60
  );
  readonly oidcAuthenticationMaxAgeSeconds = parseInteger(
    "OIDC_AUTHENTICATION_MAX_AGE_SECONDS",
    process.env.OIDC_AUTHENTICATION_MAX_AGE_SECONDS,
    60 * 60,
    60,
    24 * 60 * 60
  );
  readonly oidcAccountStepUpMaxAgeSeconds = parseInteger(
    "OIDC_ACCOUNT_STEP_UP_MAX_AGE_SECONDS",
    process.env.OIDC_ACCOUNT_STEP_UP_MAX_AGE_SECONDS,
    5 * 60,
    60,
    10 * 60
  );
  readonly oidcAccountAal2AcrValues = splitList(
    process.env.OIDC_ACCOUNT_AAL2_ACR_VALUES,
    ["urn:teachlab:aal2"]
  );
  readonly oidcTenantClaim = process.env.OIDC_TENANT_CLAIM?.trim() || "org_id";
  readonly oidcRolesClaim = process.env.OIDC_ROLES_CLAIM?.trim() || "roles";
  readonly oidcRemoteProcessingPolicyClaim =
    process.env.OIDC_REMOTE_PROCESSING_POLICY_CLAIM?.trim()
    || "teachlab_remote_processing_policy";
  readonly oidcRemoteProcessingPolicyVersionClaim =
    process.env.OIDC_REMOTE_PROCESSING_POLICY_VERSION_CLAIM?.trim()
    || "teachlab_remote_processing_policy_version";
  readonly remoteSubjectPolicyId =
    process.env.REMOTE_SUBJECT_POLICY_ID?.trim()
    || "teachlab-organization-remote-processing";
  readonly remoteSubjectPolicyVersion =
    process.env.REMOTE_SUBJECT_POLICY_VERSION?.trim() || "v1";
  readonly oidcAllowedAlgorithms = splitList(process.env.OIDC_ALLOWED_ALGORITHMS, [
    "RS256"
  ]);
  readonly oidcDiscoveryTimeoutMs = parseInteger(
    "OIDC_DISCOVERY_TIMEOUT_MS",
    process.env.OIDC_DISCOVERY_TIMEOUT_MS,
    5_000,
    500,
    30_000
  );
  readonly modelProvider = process.env.MODEL_PROVIDER?.trim() || "anthropic";
  readonly modelName = process.env.ANTHROPIC_MODEL?.trim() || "claude-sonnet-4-5";
  readonly anthropicCredentialPresent = Boolean(process.env.ANTHROPIC_API_KEY?.trim());
  readonly harnessGatewayEnabled = parseBoolean(
    process.env.HARNESS_GATEWAY_ENABLED,
    false
  );
  readonly harnessWorkerRoot = process.env.HARNESS_WORKER_ROOT?.trim();
  readonly harnessWorkerCwd = process.env.HARNESS_WORKER_CWD?.trim();
  readonly harnessWorkerPython = process.env.HARNESS_WORKER_PYTHON?.trim();
  readonly harnessWorkerBackend = this.parseHarnessAgentBackend(
    process.env.HARNESS_WORKER_BACKEND
  );
  readonly harnessProviderApiKeyFile =
    process.env.HARNESS_PROVIDER_API_KEY_FILE?.trim();
  readonly harnessRemoteProviderPolicy = providerPolicyFromEnvironment();
  readonly harnessSafeguardingLocale =
    process.env.HARNESS_SAFEGUARDING_LOCALE?.trim() || "zh-CN";
  readonly harnessSafeguardingDispatchUrl = process.env
    .HARNESS_SAFEGUARDING_DISPATCH_URL?.trim()
    ? requiredDefaultPortHttpsUrl(
        "HARNESS_SAFEGUARDING_DISPATCH_URL",
        process.env.HARNESS_SAFEGUARDING_DISPATCH_URL
      )
    : undefined;
  readonly harnessSafeguardingDispatchBearerSecret = privateSecret({
    name: "HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET",
    file: process.env.HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET_FILE,
    inline: process.env.HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET,
    production: this.production,
    required: false,
    minimumBytes: 32,
    maximumBytes: 4_096,
  });
  readonly harnessSafeguardingDispatchPolicyVersion =
    process.env.HARNESS_SAFEGUARDING_DISPATCH_POLICY_VERSION?.trim()
    || "institution-safeguarding-v1";
  readonly harnessSafeguardingDispatchTimeoutMs = parseInteger(
    "HARNESS_SAFEGUARDING_DISPATCH_TIMEOUT_MS",
    process.env.HARNESS_SAFEGUARDING_DISPATCH_TIMEOUT_MS,
    5_000,
    100,
    30_000
  );
  readonly harnessSafeguardingDispatchMaxResponseBytes = parseInteger(
    "HARNESS_SAFEGUARDING_DISPATCH_MAX_RESPONSE_BYTES",
    process.env.HARNESS_SAFEGUARDING_DISPATCH_MAX_RESPONSE_BYTES,
    32 * 1024,
    256,
    1024 * 1024
  );
  readonly harnessSafeguardingDispatchConfigured = Boolean(
    this.harnessSafeguardingDispatchUrl
    && this.harnessSafeguardingDispatchBearerSecret
  );
  readonly harnessSafeguardingRetentionPolicyVersion =
    optionalExact(
      "HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION",
      process.env.HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION
    )
    || (this.production ? "" : "local-retention-disabled");
  readonly harnessSafeguardingRetentionMinimumClosedAgeSeconds = parseExactInteger(
    "HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS",
    process.env.HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS,
    90 * 24 * 60 * 60,
    1,
    10 * 365 * 24 * 60 * 60
  );
  readonly harnessSafeguardingRetentionMaximumCasesPerRun = parseExactInteger(
    "HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN",
    process.env.HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN,
    128,
    1,
    1024
  );
  readonly harnessSafeguardingRetentionAuthoritySecret = privateSecret({
    name: "HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET",
    file: process.env.HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET_FILE,
    inline: process.env.HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET,
    production: this.production,
    required: this.production && this.harnessGatewayEnabled,
    minimumBytes: 32,
    maximumBytes: 4_096,
    pattern: /^[\x21-\x7e]+$/,
  });
  readonly harnessSafeguardingRetentionDeploymentContextSha256 =
    optionalExact(
      "HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256",
      process.env.HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256
    );
  readonly harnessSafeguardingRetentionConfigured = Boolean(
    this.harnessSafeguardingRetentionAuthoritySecret &&
    this.harnessSafeguardingRetentionPolicyVersion &&
    this.harnessSafeguardingRetentionDeploymentContextSha256
  );
  readonly harnessWorkerFilesystemIsolationRequired = parseBoolean(
    process.env.HARNESS_WORKER_FILESYSTEM_ISOLATION_REQUIRED,
    this.production,
  );
  readonly harnessWorkerRlimitAddressSpaceBytes = parseExactInteger(
    "HARNESS_WORKER_RLIMIT_AS_BYTES",
    process.env.HARNESS_WORKER_RLIMIT_AS_BYTES,
    1_536 * 1024 * 1024,
    1024 * 1024 * 1024,
    2 * 1024 * 1024 * 1024
  );
  readonly harnessWorkerRlimitFileSizeBytes = parseExactInteger(
    "HARNESS_WORKER_RLIMIT_FSIZE_BYTES",
    process.env.HARNESS_WORKER_RLIMIT_FSIZE_BYTES,
    512 * 1024 * 1024,
    320 * 1024 * 1024,
    512 * 1024 * 1024
  );
  readonly harnessWorkerRlimitOpenFiles = parseExactInteger(
    "HARNESS_WORKER_RLIMIT_NOFILE",
    process.env.HARNESS_WORKER_RLIMIT_NOFILE,
    256,
    128,
    512
  );
  readonly harnessWorkerRlimitCoreDumpBytes = parseExactInteger(
    "HARNESS_WORKER_RLIMIT_CORE_BYTES",
    process.env.HARNESS_WORKER_RLIMIT_CORE_BYTES,
    0,
    0,
    0
  ) as 0;
  readonly harnessWorkerProcessResourceLimits: HarnessWorkerProcessResourceLimits | null =
    this.harnessWorkerFilesystemIsolationRequired
      ? {
          schema: "teaching_skill_miner.worker_process_resource_limits.v1",
          address_space_bytes: this.harnessWorkerRlimitAddressSpaceBytes,
          file_size_bytes: this.harnessWorkerRlimitFileSizeBytes,
          open_files: this.harnessWorkerRlimitOpenFiles,
          core_dump_bytes: this.harnessWorkerRlimitCoreDumpBytes
        }
      : null;
  readonly harnessScopeKeyVersion =
    process.env.HARNESS_SCOPE_KEY_VERSION?.trim() || "k1";
  readonly harnessScopeSecret = privateSecret({
    name: "HARNESS_SCOPE_SECRET",
    file: process.env.HARNESS_SCOPE_SECRET_FILE,
    inline: process.env.HARNESS_SCOPE_SECRET,
    production: this.production,
    required: this.harnessGatewayEnabled,
  }) ?? "";
  readonly harnessPreviousScopeKeys = privateSecret({
    name: "HARNESS_PREVIOUS_SCOPE_KEYS",
    file: process.env.HARNESS_PREVIOUS_SCOPE_KEYS_FILE,
    inline: process.env.HARNESS_PREVIOUS_SCOPE_KEYS,
    production: this.production,
    required: false,
    minimumBytes: 2,
    maximumBytes: 16_384,
  });
  readonly harnessScopeKeys = parseVersionedSecretKeys(
    this.harnessScopeKeyVersion,
    this.harnessScopeSecret,
    this.harnessPreviousScopeKeys,
    "HARNESS_PREVIOUS_SCOPE_KEYS"
  );
  readonly accountScopeKeyVersion =
    process.env.ACCOUNT_SCOPE_KEY_VERSION?.trim() || "k1";
  readonly accountScopeSecret = privateSecret({
    name: "ACCOUNT_SCOPE_SECRET",
    file: process.env.ACCOUNT_SCOPE_SECRET_FILE,
    inline: process.env.ACCOUNT_SCOPE_SECRET,
    production: this.production,
    required: this.production,
  }) ?? randomBytes(32).toString("base64url");
  readonly accountPreviousScopeKeysJson = privateSecret({
    name: "ACCOUNT_PREVIOUS_SCOPE_KEYS",
    file: process.env.ACCOUNT_PREVIOUS_SCOPE_KEYS_FILE,
    inline: process.env.ACCOUNT_PREVIOUS_SCOPE_KEYS,
    production: this.production,
    required: false,
    minimumBytes: 2,
    maximumBytes: 16_384,
  });
  readonly accountScopeKeys = parseVersionedSecretKeys(
    this.accountScopeKeyVersion,
    this.accountScopeSecret,
    this.accountPreviousScopeKeysJson,
    "ACCOUNT_PREVIOUS_SCOPE_KEYS"
  );
  readonly accountIdentityNamespaceVersion =
    process.env.ACCOUNT_IDENTITY_NAMESPACE_VERSION?.trim() || "ns1";
  readonly accountIdentityNamespaceSecret = privateSecret({
    name: "ACCOUNT_IDENTITY_NAMESPACE_SECRET",
    file: process.env.ACCOUNT_IDENTITY_NAMESPACE_SECRET_FILE,
    inline: process.env.ACCOUNT_IDENTITY_NAMESPACE_SECRET,
    production: this.production,
    required: this.production,
  }) ?? randomBytes(32).toString("base64url");
  readonly accountDeletionStatusSecret = privateSecret({
    name: "ACCOUNT_DELETION_STATUS_SECRET",
    file: process.env.ACCOUNT_DELETION_STATUS_SECRET_FILE,
    inline: process.env.ACCOUNT_DELETION_STATUS_SECRET,
    production: this.production,
    required: this.production,
  }) ?? randomBytes(32).toString("base64url");
  readonly accountCacheScopeSecret = privateSecret({
    name: "ACCOUNT_CACHE_SCOPE_SECRET",
    file: process.env.ACCOUNT_CACHE_SCOPE_SECRET_FILE,
    inline: process.env.ACCOUNT_CACHE_SCOPE_SECRET,
    production: this.production,
    required: this.production,
  }) ?? randomBytes(32).toString("base64url");
  readonly accountCacheScopeEpoch =
    process.env.ACCOUNT_CACHE_SCOPE_EPOCH?.trim() || "epoch1";
  readonly metricsAccessToken = privateSecret({
    name: "METRICS_TOKEN",
    file: process.env.METRICS_TOKEN_FILE,
    inline: process.env.METRICS_TOKEN,
    production: this.production,
    required: this.production,
    pattern: /^[A-Za-z0-9._~-]+$/,
  });
  readonly harnessMaxWorkers = parseInteger(
    "HARNESS_MAX_WORKERS",
    process.env.HARNESS_MAX_WORKERS,
    16,
    1,
    128
  );
  readonly harnessWorkerStartupMs = parseInteger(
    "HARNESS_WORKER_STARTUP_MS",
    process.env.HARNESS_WORKER_STARTUP_MS,
    20_000,
    1_000,
    60_000
  );
  readonly harnessWorkerShutdownMs = parseInteger(
    "HARNESS_WORKER_SHUTDOWN_MS",
    process.env.HARNESS_WORKER_SHUTDOWN_MS,
    5_000,
    500,
    30_000
  );
  readonly harnessWorkerIdleTimeoutMs = parseInteger(
    "HARNESS_WORKER_IDLE_TIMEOUT_MS",
    process.env.HARNESS_WORKER_IDLE_TIMEOUT_MS,
    15 * 60 * 1000,
    1_000,
    24 * 60 * 60 * 1000
  );
  readonly harnessResponseMaxBytes = parseInteger(
    "HARNESS_RESPONSE_MAX_BYTES",
    process.env.HARNESS_RESPONSE_MAX_BYTES,
    32 * 1024 * 1024,
    64 * 1024,
    128 * 1024 * 1024
  );
  readonly teacherAuthorityRoles = splitList(
    process.env.TEACHER_AUTHORITY_ROLES,
    ["teacher"]
  );
  readonly safeguardingAuthorityRoles = splitList(
    process.env.SAFEGUARDING_AUTHORITY_ROLES,
    ["safeguarding"]
  );
  readonly teacherAuthorityTtlSeconds = parseInteger(
    "TEACHER_AUTHORITY_TTL_SECONDS",
    process.env.TEACHER_AUTHORITY_TTL_SECONDS,
    120,
    30,
    10 * 60
  );
  /**
   * Production teacher mutations are authorized against this server-side
   * directory. Browser session role claims are deliberately not an input.
   */
  readonly teacherEntitlementDirectoryUrl = process.env
    .TEACHER_ENTITLEMENT_DIRECTORY_URL?.trim()
    ? requiredHttpsUrl(
        "TEACHER_ENTITLEMENT_DIRECTORY_URL",
        process.env.TEACHER_ENTITLEMENT_DIRECTORY_URL
      )
    : undefined;
  readonly teacherEntitlementDirectoryBearerSecret = privateSecret({
    name: "TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET",
    file: process.env.TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET_FILE,
    inline: process.env.TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET,
    production: this.production,
    required: this.production,
    minimumBytes: 32,
    maximumBytes: 4_096,
    pattern: /^[\x21-\x7e]+$/,
  });
  readonly teacherEntitlementBindingKey = privateSecret({
    name: "TEACHER_ENTITLEMENT_BINDING_KEY",
    file: process.env.TEACHER_ENTITLEMENT_BINDING_KEY_FILE,
    inline: process.env.TEACHER_ENTITLEMENT_BINDING_KEY,
    production: this.production,
    required: this.production,
  });
  readonly teacherEntitlementReceiptKey = privateSecret({
    name: "TEACHER_ENTITLEMENT_RECEIPT_KEY",
    file: process.env.TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE,
    inline: process.env.TEACHER_ENTITLEMENT_RECEIPT_KEY,
    production: this.production,
    required: this.production,
  });
  readonly teacherEntitlementPolicyId =
    process.env.TEACHER_ENTITLEMENT_POLICY_ID?.trim()
    || (this.production ? "" : "teacher-mutations");
  readonly teacherEntitlementPolicyVersion =
    process.env.TEACHER_ENTITLEMENT_POLICY_VERSION?.trim()
    || (this.production ? "" : "roles-v1");
  readonly teacherEntitlementFreshnessTtlMs = parseInteger(
    "TEACHER_ENTITLEMENT_FRESHNESS_TTL_MS",
    process.env.TEACHER_ENTITLEMENT_FRESHNESS_TTL_MS,
    30_000,
    1_000,
    5 * 60_000
  );
  readonly teacherEntitlementCacheTtlMs = parseInteger(
    "TEACHER_ENTITLEMENT_CACHE_TTL_MS",
    process.env.TEACHER_ENTITLEMENT_CACHE_TTL_MS,
    this.production ? 5_000 : 0,
    0,
    5 * 60_000
  );
  readonly teacherEntitlementProviderTimeoutMs = parseInteger(
    "TEACHER_ENTITLEMENT_PROVIDER_TIMEOUT_MS",
    process.env.TEACHER_ENTITLEMENT_PROVIDER_TIMEOUT_MS,
    2_000,
    100,
    30_000
  );
  readonly teacherEntitlementMaxResponseBytes = parseInteger(
    "TEACHER_ENTITLEMENT_MAX_RESPONSE_BYTES",
    process.env.TEACHER_ENTITLEMENT_MAX_RESPONSE_BYTES,
    64 * 1024,
    256,
    1024 * 1024
  );
  readonly teacherEntitlementMaxClockSkewMs = parseInteger(
    "TEACHER_ENTITLEMENT_MAX_CLOCK_SKEW_MS",
    process.env.TEACHER_ENTITLEMENT_MAX_CLOCK_SKEW_MS,
    5_000,
    0,
    30_000
  );
  readonly teacherEntitlementMaxCacheEntries = parseInteger(
    "TEACHER_ENTITLEMENT_MAX_CACHE_ENTRIES",
    process.env.TEACHER_ENTITLEMENT_MAX_CACHE_ENTRIES,
    10_000,
    1,
    100_000
  );
  readonly teacherEntitlementMinAssuranceLevel = parseInteger(
    "TEACHER_ENTITLEMENT_MIN_ASSURANCE_LEVEL",
    process.env.TEACHER_ENTITLEMENT_MIN_ASSURANCE_LEVEL,
    this.production ? 2 : 1,
    1,
    3
  ) as 1 | 2 | 3;
  readonly developmentRoles = splitList(process.env.DEV_AUTH_ROLES, ["developer"]);

  assertSafeForStartup(): void {
    if (this.corsOrigins.length === 0 || this.corsOrigins.includes("*")) {
      throw new Error("CORS_ORIGINS must be a non-empty exact allowlist and cannot contain *");
    }
    for (const origin of this.corsOrigins) {
      const parsed = new URL(origin);
      if (!parsed.origin || parsed.origin !== origin) {
        throw new Error(`CORS_ORIGINS entry must be an exact origin: ${origin}`);
      }
    }
    if (this.sessionSecret.length < 32) {
      throw new Error("SESSION_SECRET must contain at least 32 characters");
    }
    if (
      !this.teacherAuthorityRoles.length ||
      this.teacherAuthorityRoles.some((role) => !ROLE_NAME_PATTERN.test(role))
    ) {
      throw new Error("TEACHER_AUTHORITY_ROLES must contain safe exact role names");
    }
    if (
      this.safeguardingAuthorityRoles.length !== 1
      || this.safeguardingAuthorityRoles[0] !== "safeguarding"
    ) {
      throw new Error(
        "SAFEGUARDING_AUTHORITY_ROLES must be the exact safeguarding role"
      );
    }
    if (
      this.teacherAuthorityRoles.some((role) =>
        this.safeguardingAuthorityRoles.includes(role)
      )
    ) {
      throw new Error(
        "Teacher and safeguarding authority roles must be disjoint"
      );
    }
    if (
      !POLICY_COMPONENT_PATTERN.test(this.teacherEntitlementPolicyId)
      || !POLICY_COMPONENT_PATTERN.test(this.teacherEntitlementPolicyVersion)
      || this.teacherEntitlementCacheTtlMs > this.teacherEntitlementFreshnessTtlMs
    ) {
      throw new Error("Teacher entitlement policy configuration is invalid");
    }
    if (this.production && this.teacherEntitlementCacheTtlMs > 5_000) {
      throw new Error(
        "Production TEACHER_ENTITLEMENT_CACHE_TTL_MS cannot exceed 5000"
      );
    }
    const harnessProviderApiKey = (
      this.production
      && this.harnessGatewayEnabled
      && this.harnessWorkerBackend === "deepseek"
    ) ? privateSecret({
      name: "HARNESS_PROVIDER_API_KEY",
      file: this.harnessProviderApiKeyFile,
      inline: undefined,
      production: true,
      required: true,
      minimumBytes: 1,
      maximumBytes: 4_096,
      pattern: /^[\x21-\x7e]+$/,
    }) : undefined;
    const entitlementSecrets = [
      this.teacherEntitlementDirectoryBearerSecret,
      this.teacherEntitlementBindingKey,
      this.teacherEntitlementReceiptKey
    ].filter((value): value is string => value !== undefined);
    if (new Set(entitlementSecrets).size !== entitlementSecrets.length) {
      throw new Error("Teacher entitlement secrets must be mutually distinct");
    }
    const otherSecrets = new Set([
      ...this.sessionVerificationSecrets,
      this.oidcClientSecret,
      this.oidcTransactionSecret,
      this.databaseUrl,
      ...this.harnessScopeKeys.map((key) => key.secret),
      ...this.accountScopeKeys.map((key) => key.secret),
      this.accountIdentityNamespaceSecret,
      this.accountDeletionStatusSecret,
      this.accountCacheScopeSecret,
      this.metricsAccessToken,
      harnessProviderApiKey
    ].filter((value): value is string => value !== undefined && value !== ""));
    if (entitlementSecrets.some((secret) => otherSecrets.has(secret))) {
      throw new Error("Teacher entitlement secrets must be isolated from other service keys");
    }
    if (
      this.harnessSafeguardingDispatchBearerSecret
      && (
        entitlementSecrets.includes(this.harnessSafeguardingDispatchBearerSecret)
        || otherSecrets.has(this.harnessSafeguardingDispatchBearerSecret)
      )
    ) {
      throw new Error(
        "Safeguarding dispatcher secret must be isolated from other service keys"
      );
    }
    if (
      this.harnessSafeguardingRetentionAuthoritySecret &&
      (
        entitlementSecrets.includes(
          this.harnessSafeguardingRetentionAuthoritySecret
        ) ||
        [
          ...this.sessionVerificationSecrets,
          this.oidcClientSecret,
          this.oidcTransactionSecret,
          this.databaseUrl,
          ...this.harnessScopeKeys.map((key) => key.secret),
          ...this.accountScopeKeys.map((key) => key.secret),
          this.accountIdentityNamespaceSecret,
          this.accountDeletionStatusSecret,
          this.accountCacheScopeSecret,
          this.metricsAccessToken,
          harnessProviderApiKey,
          this.harnessSafeguardingDispatchBearerSecret
        ].some(
          (secret) => secret === this.harnessSafeguardingRetentionAuthoritySecret
        )
      )
    ) {
      throw new Error(
        "Safeguarding retention authority secret must be isolated from every service key"
      );
    }
    if (this.developmentRoles.some((role) => !ROLE_NAME_PATTERN.test(role))) {
      throw new Error("DEV_AUTH_ROLES must contain safe exact role names");
    }
    if (
      this.previousSessionSecret &&
      this.previousSessionSecret.length < 32
    ) {
      throw new Error("SESSION_PREVIOUS_SECRET must contain at least 32 characters");
    }
    if (this.authMode === "oidc") {
      if (!this.secureSessionCookies) {
        throw new Error("OIDC authorization code login requires SESSION_COOKIE_SECURE=true");
      }
      requiredHttpsUrl("OIDC_ISSUER", this.oidcIssuer);
      if (!this.oidcAudience) throw new Error("OIDC_AUDIENCE is required for OIDC auth");
      if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(this.oidcClientId ?? "")) {
        throw new Error("OIDC_CLIENT_ID is required and must be an exact safe identifier");
      }
      if (
        !this.oidcClientSecret
        || this.oidcClientSecret.length > 4096
        || /[\u0000\r\n]/.test(this.oidcClientSecret)
      ) {
        throw new Error("OIDC_CLIENT_SECRET is required and invalid");
      }
      const redirectUri = requiredHttpsUrl("OIDC_REDIRECT_URI", this.oidcRedirectUri);
      const redirect = new URL(redirectUri);
      if (redirect.pathname !== "/api/teacher-agent/security/login/callback") {
        throw new Error("OIDC_REDIRECT_URI must use the exact Console callback path");
      }
      if (!this.corsOrigins.includes(redirect.origin)) {
        throw new Error("OIDC_REDIRECT_URI origin must be present in CORS_ORIGINS");
      }
      if (!this.oidcTransactionSecret || this.oidcTransactionSecret.length < 32) {
        throw new Error("OIDC_TRANSACTION_SECRET must contain at least 32 characters");
      }
      if (
        !this.oidcExpectedHost
        || this.oidcExpectedHost.length > 255
        || /[\s\/?#@]/.test(this.oidcExpectedHost)
      ) {
        throw new Error("OIDC_EXPECTED_HOST must be an exact host[:port]");
      }
      if (!this.oidcAllowedAlgorithms.length) {
        throw new Error("OIDC_ALLOWED_ALGORITHMS cannot be empty");
      }
      if (
        !CLAIM_NAME_PATTERN.test(this.oidcRemoteProcessingPolicyClaim)
        || !CLAIM_NAME_PATTERN.test(this.oidcRemoteProcessingPolicyVersionClaim)
        || this.oidcRemoteProcessingPolicyClaim
          === this.oidcRemoteProcessingPolicyVersionClaim
      ) {
        throw new Error("OIDC remote-processing claim names are invalid");
      }
      if (
        !POLICY_COMPONENT_PATTERN.test(this.remoteSubjectPolicyId)
        || !POLICY_COMPONENT_PATTERN.test(this.remoteSubjectPolicyVersion)
      ) {
        throw new Error("REMOTE_SUBJECT_POLICY_ID/VERSION are invalid");
      }
      if (this.oidcAllowedAlgorithms.some((algorithm) => algorithm !== "RS256")) {
        throw new Error("This release supports only OIDC_ALLOWED_ALGORITHMS=RS256");
      }
      if (
        !this.oidcAccountAal2AcrValues.length
        || this.oidcAccountAal2AcrValues.some((value) =>
          value.length > 256 || value !== value.trim() || /[\u0000-\u0020\u007f]/.test(value)
        )
      ) {
        throw new Error("OIDC_ACCOUNT_AAL2_ACR_VALUES must be an exact safe allowlist");
      }
    }
    if (!/^k[1-9][0-9]{0,8}$/.test(this.accountScopeKeyVersion)) {
      throw new Error("ACCOUNT_SCOPE_KEY_VERSION must match k<positive integer>");
    }
    const accountDomainSecrets = [
      this.accountScopeSecret,
      this.accountIdentityNamespaceSecret,
      this.accountDeletionStatusSecret,
      this.accountCacheScopeSecret,
    ];
    if (new Set(accountDomainSecrets).size !== accountDomainSecrets.length) {
      throw new Error("Account-domain active secrets must be mutually distinct");
    }
    const everyServiceSecret = [
      ...this.sessionVerificationSecrets,
      this.oidcClientSecret,
      this.oidcTransactionSecret,
      this.databaseUrl,
      ...this.harnessScopeKeys.map((key) => key.secret),
      ...this.accountScopeKeys.map((key) => key.secret),
      this.accountIdentityNamespaceSecret,
      this.accountDeletionStatusSecret,
      this.accountCacheScopeSecret,
      this.metricsAccessToken,
      harnessProviderApiKey,
      ...entitlementSecrets,
      this.harnessSafeguardingDispatchBearerSecret,
      this.harnessSafeguardingRetentionAuthoritySecret
    ].filter((value): value is string => value !== undefined && value !== "");
    if (new Set(everyServiceSecret).size !== everyServiceSecret.length) {
      throw new Error("Every service secret must be pairwise isolated");
    }
    const accountVersions = new Set<string>();
    for (const key of this.accountScopeKeys) {
      if (!/^k[1-9][0-9]{0,8}$/.test(key.version) || key.secret.length < 32) {
        throw new Error("Every account scope key needs a valid version and at least 32 characters");
      }
      if (accountVersions.has(key.version)) {
        throw new Error("Account scope key versions must be unique");
      }
      accountVersions.add(key.version);
    }
    if (this.accountDeletionStatusSecret.length < 32) {
      throw new Error("ACCOUNT_DELETION_STATUS_SECRET must contain at least 32 characters");
    }
    if (
      this.accountIdentityNamespaceSecret.length < 32
      || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,31}$/.test(this.accountIdentityNamespaceVersion)
    ) {
      throw new Error("ACCOUNT_IDENTITY_NAMESPACE_* must define a stable safe namespace");
    }
    if (this.accountCacheScopeSecret.length < 32) {
      throw new Error("ACCOUNT_CACHE_SCOPE_SECRET must contain at least 32 characters");
    }
    if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/.test(this.accountCacheScopeEpoch)) {
      throw new Error("ACCOUNT_CACHE_SCOPE_EPOCH must be a safe version identifier");
    }
    if (this.dataBackend === "postgres" && !this.databaseUrl) {
      throw new Error("DATABASE_URL is required when DATA_BACKEND=postgres");
    }
    if (this.databaseUrl) {
      const databaseUrl = new URL(this.databaseUrl);
      if (!new Set(["postgres:", "postgresql:"]).has(databaseUrl.protocol)) {
        throw new Error("DATABASE_URL must use the postgres or postgresql scheme");
      }
      for (const parameter of ["sslmode", "sslcert", "sslkey", "sslrootcert"]) {
        if (databaseUrl.searchParams.has(parameter)) {
          throw new Error(
            `DATABASE_URL cannot contain ${parameter}; configure TLS with PG_SSL_MODE`
          );
        }
      }
    }
    if (this.harnessGatewayEnabled) {
      if (!/^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8}){0,3}$/.test(
        this.harnessSafeguardingLocale
      )) {
        throw new Error("HARNESS_SAFEGUARDING_LOCALE must be a valid locale tag");
      }
      if (
        Boolean(this.harnessSafeguardingDispatchUrl)
        !== Boolean(this.harnessSafeguardingDispatchBearerSecret)
        || !POLICY_COMPONENT_PATTERN.test(
          this.harnessSafeguardingDispatchPolicyVersion
        )
      ) {
        throw new Error(
          "Safeguarding dispatcher requires one complete HTTPS endpoint, bearer secret, and policy"
        );
      }
      const retentionEnvironment = [
        process.env.HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION,
        process.env.HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS,
        process.env.HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN,
        process.env.HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET_FILE ??
          process.env.HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET,
        process.env.HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256
      ];
      const retentionSupplied = retentionEnvironment.filter((value) =>
        value?.trim()
      ).length;
      if (
        retentionSupplied !== 0 &&
        (
          retentionSupplied !== retentionEnvironment.length ||
          !this.harnessSafeguardingRetentionConfigured
        )
      ) {
        throw new Error(
          "Safeguarding retention requires one complete policy and dedicated authority"
        );
      }
      if (
        this.harnessSafeguardingRetentionConfigured &&
        (
          !POLICY_COMPONENT_PATTERN.test(
            this.harnessSafeguardingRetentionPolicyVersion
          ) ||
          !/^[0-9a-f]{64}$/.test(
            this.harnessSafeguardingRetentionDeploymentContextSha256 ?? ""
          )
        )
      ) {
        throw new Error("Safeguarding retention policy identifiers are invalid");
      }
      if (
        this.harnessSafeguardingRetentionConfigured &&
        !this.harnessSafeguardingDispatchConfigured
      ) {
        throw new Error(
          "Safeguarding retention requires the durable safeguarding supervisor"
        );
      }
      if (
        this.production &&
        (
          retentionSupplied !== retentionEnvironment.length ||
          !this.harnessSafeguardingRetentionConfigured ||
          !this.harnessSafeguardingDispatchConfigured
        )
      ) {
        throw new Error(
          "Production requires explicit safeguarding dispatch and retention policy"
        );
      }
      if (this.production && !this.harnessWorkerFilesystemIsolationRequired) {
        throw new Error("production Harness workers require filesystem isolation");
      }
      const processResourceLimitEnvironment = [
        process.env.HARNESS_WORKER_RLIMIT_AS_BYTES,
        process.env.HARNESS_WORKER_RLIMIT_FSIZE_BYTES,
        process.env.HARNESS_WORKER_RLIMIT_NOFILE,
        process.env.HARNESS_WORKER_RLIMIT_CORE_BYTES
      ];
      if (
        !this.harnessWorkerFilesystemIsolationRequired &&
        processResourceLimitEnvironment.some((value) => value?.trim())
      ) {
        throw new Error(
          "HARNESS_WORKER_RLIMIT_* is forbidden when worker isolation is not required"
        );
      }
      if (
        this.production &&
        (
          processResourceLimitEnvironment.some((value) => !value?.trim()) ||
          this.harnessWorkerRlimitAddressSpaceBytes !== 1_610_612_736 ||
          this.harnessWorkerRlimitFileSizeBytes !== 536_870_912 ||
          this.harnessWorkerRlimitOpenFiles !== 256 ||
          this.harnessWorkerRlimitCoreDumpBytes !== 0
        )
      ) {
        throw new Error(
          "Production requires explicit fixed HARNESS_WORKER_RLIMIT_* process limits"
        );
      }
      if (
        !this.harnessWorkerRoot ||
        !isAbsolute(this.harnessWorkerRoot) ||
        this.harnessWorkerRoot === "/"
      ) {
        throw new Error(
          "HARNESS_WORKER_ROOT must be an explicit absolute non-root directory"
        );
      }
      if (!this.harnessWorkerCwd || !isAbsolute(this.harnessWorkerCwd)) {
        throw new Error("HARNESS_WORKER_CWD must be an explicit absolute directory");
      }
      if (
        !this.harnessWorkerPython ||
        !isAbsolute(this.harnessWorkerPython) ||
        this.harnessWorkerPython.length > 512 ||
        /[\u0000\r\n]/.test(this.harnessWorkerPython)
      ) {
        throw new Error(
          "HARNESS_WORKER_PYTHON must be an absolute Python executable path"
        );
      }
      if (!/^k[1-9][0-9]{0,8}$/.test(this.harnessScopeKeyVersion)) {
        throw new Error("HARNESS_SCOPE_KEY_VERSION must match k<positive integer>");
      }
      const versions = new Set<string>();
      for (const key of this.harnessScopeKeys) {
        if (!/^k[1-9][0-9]{0,8}$/.test(key.version) || key.secret.length < 32) {
          throw new Error(
            "Every active or previous Harness scope key needs a valid version and at least 32 characters"
          );
        }
        if (versions.has(key.version)) {
          throw new Error("Harness scope key versions must be unique");
        }
        versions.add(key.version);
      }
      if (
        this.harnessWorkerBackend === "deepseek" &&
        (!this.harnessProviderApiKeyFile ||
          !isAbsolute(this.harnessProviderApiKeyFile))
      ) {
        throw new Error(
          "DeepSeek Harness workers require an absolute HARNESS_PROVIDER_API_KEY_FILE"
        );
      }
      if (
        this.harnessWorkerBackend === "deepseek"
        && !this.harnessRemoteProviderPolicy
      ) {
        throw new Error(
          "DeepSeek Harness workers require a complete HARNESS_PROVIDER_POLICY_* policy"
        );
      }
      if (
        this.harnessWorkerBackend === "deterministic" &&
        this.harnessProviderApiKeyFile
      ) {
        throw new Error(
          "HARNESS_PROVIDER_API_KEY_FILE is forbidden for deterministic workers"
        );
      }
    }
    if (!new Set(["disable", "require", "verify-full"]).has(this.postgresSslMode)) {
      throw new Error("PG_SSL_MODE must be disable, require, or verify-full");
    }
    if (this.authMode === "development" && !isLoopbackHost(this.host)) {
      throw new Error("Development identity requires a loopback API_HOST");
    }
    if (this.allowDevelopmentAuthHeaders && this.nodeEnv !== "test") {
      throw new Error("DEV_AUTH_ALLOW_HEADERS=true is restricted to NODE_ENV=test");
    }
    if (this.nodeEnv !== "production") return;
    if (
      !this.teacherEntitlementDirectoryUrl
      || !process.env.TEACHER_ENTITLEMENT_DIRECTORY_URL?.trim()
      || !process.env.TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET_FILE?.trim()
      || !process.env.TEACHER_ENTITLEMENT_BINDING_KEY_FILE?.trim()
      || !process.env.TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE?.trim()
      || !process.env.TEACHER_ENTITLEMENT_POLICY_ID?.trim()
      || !process.env.TEACHER_ENTITLEMENT_POLICY_VERSION?.trim()
    ) {
      throw new Error(
        "Production requires an authoritative teacher entitlement directory, policy, and private key files"
      );
    }
    if (this.authMode !== "oidc") {
      throw new Error("Production requires AUTH_MODE=oidc; development identity is local-only");
    }
    if (this.dataBackend !== "postgres") {
      throw new Error("Production requires DATA_BACKEND=postgres; memory is local-only");
    }
    if (!process.env.SESSION_SECRET_FILE?.trim()) {
      throw new Error("Production requires SESSION_SECRET_FILE");
    }
    if (!process.env.OIDC_EXPECTED_HOST?.trim()) {
      throw new Error("Production requires an explicit OIDC_EXPECTED_HOST");
    }
    if (
      !process.env.REMOTE_SUBJECT_POLICY_ID?.trim()
      || !process.env.REMOTE_SUBJECT_POLICY_VERSION?.trim()
    ) {
      throw new Error("Production requires explicit REMOTE_SUBJECT_POLICY_ID/VERSION");
    }
    if (!this.secureSessionCookies) {
      throw new Error("Production requires SESSION_COOKIE_SECURE=true");
    }
    if (this.allowDevelopmentAuthHeaders) {
      throw new Error("DEV_AUTH_ALLOW_HEADERS=true is forbidden in production");
    }
    if (this.seedDemoSessions) {
      throw new Error("SEED_DEMO_SESSIONS=true is forbidden in production");
    }
    if (this.postgresSslMode !== "verify-full") {
      throw new Error("Production requires PG_SSL_MODE=verify-full");
    }
    for (const origin of this.corsOrigins) {
      if (!origin.startsWith("https://")) {
        throw new Error("Production CORS_ORIGINS entries must use HTTPS");
      }
    }
    if (!this.harnessGatewayEnabled) {
      throw new Error("Production requires HARNESS_GATEWAY_ENABLED=true");
    }
  }

  isTrustedOrigin(origin: string | undefined): boolean {
    return origin !== undefined && this.corsOrigins.includes(origin);
  }

  principalHasTeacherAuthority(roles: readonly string[]): boolean {
    const authorized = new Set(this.teacherAuthorityRoles);
    return roles.some((role) => authorized.has(role));
  }

  private parseAuthMode(value: string | undefined): AuthMode {
    const normalized = value?.trim().toLowerCase() || "development";
    if (normalized === "development" || normalized === "oidc") return normalized;
    throw new Error("AUTH_MODE must be development or oidc");
  }

  private parseDataBackend(value: string | undefined): DataBackend {
    const normalized = value?.trim().toLowerCase() || "memory";
    if (normalized === "memory" || normalized === "postgres") return normalized;
    throw new Error("DATA_BACKEND must be memory or postgres");
  }

  private parseHarnessAgentBackend(value: string | undefined): HarnessAgentBackend {
    const normalized = value?.trim().toLowerCase() || "deterministic";
    if (normalized === "deterministic" || normalized === "deepseek") {
      return normalized;
    }
    throw new Error("HARNESS_WORKER_BACKEND must be deterministic or deepseek");
  }
}
