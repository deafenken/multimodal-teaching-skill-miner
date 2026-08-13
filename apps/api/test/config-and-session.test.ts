import assert from "node:assert/strict";
import {chmodSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {test} from "node:test";

import {AppConfigService} from "../src/config/app-config.service";
import {SessionCookieService} from "../src/auth/session-cookie";

const MANAGED_KEYS = [
  "NODE_ENV",
  "API_HOST",
  "AUTH_MODE",
  "DATA_BACKEND",
  "DATABASE_URL",
  "DATABASE_URL_FILE",
  "PG_SSL_MODE",
  "CORS_ORIGINS",
  "OIDC_ISSUER",
  "OIDC_AUDIENCE",
  "OIDC_CLIENT_ID",
  "OIDC_CLIENT_SECRET",
  "OIDC_CLIENT_SECRET_FILE",
  "OIDC_REDIRECT_URI",
  "OIDC_TRANSACTION_SECRET",
  "OIDC_TRANSACTION_SECRET_FILE",
  "OIDC_EXPECTED_HOST",
  "OIDC_ALLOWED_ALGORITHMS",
  "OIDC_REMOTE_PROCESSING_POLICY_CLAIM",
  "OIDC_REMOTE_PROCESSING_POLICY_VERSION_CLAIM",
  "REMOTE_SUBJECT_POLICY_ID",
  "REMOTE_SUBJECT_POLICY_VERSION",
  "SESSION_SECRET",
  "SESSION_SECRET_FILE",
  "SESSION_PREVIOUS_SECRET",
  "SESSION_PREVIOUS_SECRET_FILE",
  "SESSION_COOKIE_SECURE",
  "DEV_AUTH_ALLOW_HEADERS",
  "DEV_AUTH_ROLES",
  "SEED_DEMO_SESSIONS",
  "HARNESS_GATEWAY_ENABLED",
  "HARNESS_WORKER_ROOT",
  "HARNESS_WORKER_CWD",
  "HARNESS_WORKER_PYTHON",
  "HARNESS_WORKER_BACKEND",
  "HARNESS_WORKER_FILESYSTEM_ISOLATION_REQUIRED",
  "HARNESS_WORKER_RLIMIT_AS_BYTES",
  "HARNESS_WORKER_RLIMIT_FSIZE_BYTES",
  "HARNESS_WORKER_RLIMIT_NOFILE",
  "HARNESS_WORKER_RLIMIT_CORE_BYTES",
  "HARNESS_SCOPE_KEY_VERSION",
  "HARNESS_SCOPE_SECRET",
  "HARNESS_SCOPE_SECRET_FILE",
  "HARNESS_PREVIOUS_SCOPE_KEYS",
  "HARNESS_PREVIOUS_SCOPE_KEYS_FILE",
  "HARNESS_PROVIDER_API_KEY_FILE",
  "HARNESS_PROVIDER_POLICY_ID",
  "HARNESS_PROVIDER_POLICY_VERSION",
  "HARNESS_PROVIDER_PROCESSING_REGION",
  "HARNESS_PROVIDER_RETENTION_DAYS",
  "HARNESS_PROVIDER_DELETION_STATUS",
  "HARNESS_PROVIDER_DOCUMENTATION_URL",
  "HARNESS_SAFEGUARDING_LOCALE",
  "HARNESS_SAFEGUARDING_DISPATCH_URL",
  "HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET",
  "HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET_FILE",
  "HARNESS_SAFEGUARDING_DISPATCH_POLICY_VERSION",
  "HARNESS_SAFEGUARDING_DISPATCH_TIMEOUT_MS",
  "HARNESS_SAFEGUARDING_DISPATCH_MAX_RESPONSE_BYTES",
  "HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION",
  "HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS",
  "HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN",
  "HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET",
  "HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET_FILE",
  "HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256",
  "HARNESS_WORKER_IDLE_TIMEOUT_MS",
  "TEACHER_AUTHORITY_ROLES",
  "SAFEGUARDING_AUTHORITY_ROLES",
  "TEACHER_AUTHORITY_TTL_SECONDS",
  "TEACHER_ENTITLEMENT_DIRECTORY_URL",
  "TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET",
  "TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET_FILE",
  "TEACHER_ENTITLEMENT_BINDING_KEY",
  "TEACHER_ENTITLEMENT_BINDING_KEY_FILE",
  "TEACHER_ENTITLEMENT_RECEIPT_KEY",
  "TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE",
  "TEACHER_ENTITLEMENT_POLICY_ID",
  "TEACHER_ENTITLEMENT_POLICY_VERSION",
  "TEACHER_ENTITLEMENT_FRESHNESS_TTL_MS",
  "TEACHER_ENTITLEMENT_CACHE_TTL_MS",
  "TEACHER_ENTITLEMENT_PROVIDER_TIMEOUT_MS",
  "TEACHER_ENTITLEMENT_MAX_RESPONSE_BYTES",
  "TEACHER_ENTITLEMENT_MAX_CLOCK_SKEW_MS",
  "TEACHER_ENTITLEMENT_MAX_CACHE_ENTRIES",
  "TEACHER_ENTITLEMENT_MIN_ASSURANCE_LEVEL"
  ,"ACCOUNT_SCOPE_SECRET",
  "ACCOUNT_SCOPE_SECRET_FILE",
  "ACCOUNT_PREVIOUS_SCOPE_KEYS",
  "ACCOUNT_PREVIOUS_SCOPE_KEYS_FILE",
  "ACCOUNT_IDENTITY_NAMESPACE_SECRET",
  "ACCOUNT_IDENTITY_NAMESPACE_SECRET_FILE",
  "ACCOUNT_DELETION_STATUS_SECRET",
  "ACCOUNT_DELETION_STATUS_SECRET_FILE",
  "ACCOUNT_CACHE_SCOPE_SECRET",
  "ACCOUNT_CACHE_SCOPE_SECRET_FILE"
  ,"METRICS_TOKEN",
  "METRICS_TOKEN_FILE"
] as const;

function withEnvironment<T>(values: Record<string, string | undefined>, operation: () => T): T {
  const previous = new Map<string, string | undefined>();
  for (const key of MANAGED_KEYS) {
    previous.set(key, process.env[key]);
    delete process.env[key];
  }
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  try {
    return operation();
  } finally {
    for (const key of MANAGED_KEYS) {
      const value = previous.get(key);
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

const SAFE_PRODUCTION_ENV_BASE = {
  NODE_ENV: "production",
  AUTH_MODE: "oidc",
  DATA_BACKEND: "postgres",
  PG_SSL_MODE: "verify-full",
  CORS_ORIGINS: "https://console.example.test",
  OIDC_ISSUER: "https://identity.example.test",
  OIDC_AUDIENCE: "teachlab-api",
  OIDC_CLIENT_ID: "teachlab-console",
  OIDC_REDIRECT_URI: "https://console.example.test/api/teacher-agent/security/login/callback",
  OIDC_EXPECTED_HOST: "api.example.test",
  OIDC_ALLOWED_ALGORITHMS: "RS256",
  REMOTE_SUBJECT_POLICY_ID: "school-remote-processing",
  REMOTE_SUBJECT_POLICY_VERSION: "2026-08-12",
  SESSION_COOKIE_SECURE: "true",
  DEV_AUTH_ALLOW_HEADERS: "false",
  SEED_DEMO_SESSIONS: "false",
  HARNESS_GATEWAY_ENABLED: "true",
  HARNESS_WORKER_ROOT: "/srv/teachlab/private-workers",
  HARNESS_WORKER_CWD: "/srv/teachlab/application",
  HARNESS_WORKER_PYTHON: "/srv/teachlab/runtime/bin/python",
  HARNESS_WORKER_BACKEND: "deterministic",
  HARNESS_WORKER_FILESYSTEM_ISOLATION_REQUIRED: "true",
  HARNESS_WORKER_RLIMIT_AS_BYTES: "1610612736",
  HARNESS_WORKER_RLIMIT_FSIZE_BYTES: "536870912",
  HARNESS_WORKER_RLIMIT_NOFILE: "256",
  HARNESS_WORKER_RLIMIT_CORE_BYTES: "0",
  HARNESS_SCOPE_KEY_VERSION: "k2",
  TEACHER_ENTITLEMENT_DIRECTORY_URL:
    "https://entitlements.organization.example/v1/teacher/snapshot",
  TEACHER_ENTITLEMENT_POLICY_ID: "teacher-mutations",
  TEACHER_ENTITLEMENT_POLICY_VERSION: "roles-v7",
  TEACHER_ENTITLEMENT_CACHE_TTL_MS: "5000",
};

function safeProductionEnvironment() {
  const directory = mkdtempSync(join(realpathSync(tmpdir()), "teachlab-config-"));
  chmodSync(directory, 0o700);
  const values = {
    database: "postgresql://teachlab@example.invalid/teachlab",
    session: "production-session-secret-with-at-least-32-characters",
    oidcClient: "production-oidc-client-secret",
    oidcTransaction: "production-oidc-transaction-secret-at-least-32-characters",
    harness: "production-harness-scope-secret-at-least-32-characters",
    previousHarness: '{"k1":"previous-production-harness-scope-secret-at-least-32-characters"}',
    metrics: "production-metrics-token-with-at-least-thirty-two-characters",
    accountScope: "production-account-scope-secret-at-least-32-characters",
    accountIdentity: "production-account-identity-secret-at-least-32-characters",
    accountDeletionStatus: "production-account-deletion-status-secret-at-least-32-characters",
    accountCache: "production-account-cache-secret-at-least-32-characters",
    entitlementBearer: "production-entitlement-directory-bearer-secret-0001",
    entitlementBinding: "production-entitlement-binding-key-000000000000000001",
    entitlementReceipt: "production-entitlement-receipt-key-000000000000000002",
    safeguardingDispatch: "production-safeguarding-dispatch-bearer-secret-0001",
    safeguardingRetention:
      "production-safeguarding-retention-authority-secret-0002",
    provider: "sk-production-provider-fixture",
  } as const;
  const paths = Object.fromEntries(Object.keys(values).map((key) => [key, join(directory, key)])) as Record<keyof typeof values, string>;
  for (const [key, value] of Object.entries(values) as Array<[keyof typeof values, string]>) {
    writeFileSync(paths[key], value, {mode: 0o600});
  }
  return {
    directory,
    env: {
      ...SAFE_PRODUCTION_ENV_BASE,
      DATABASE_URL_FILE: paths.database,
      SESSION_SECRET_FILE: paths.session,
      OIDC_CLIENT_SECRET_FILE: paths.oidcClient,
      OIDC_TRANSACTION_SECRET_FILE: paths.oidcTransaction,
      HARNESS_SCOPE_SECRET_FILE: paths.harness,
      HARNESS_PREVIOUS_SCOPE_KEYS_FILE: paths.previousHarness,
      METRICS_TOKEN_FILE: paths.metrics,
      ACCOUNT_SCOPE_SECRET_FILE: paths.accountScope,
      ACCOUNT_IDENTITY_NAMESPACE_SECRET_FILE: paths.accountIdentity,
      ACCOUNT_DELETION_STATUS_SECRET_FILE: paths.accountDeletionStatus,
      ACCOUNT_CACHE_SCOPE_SECRET_FILE: paths.accountCache,
      TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET_FILE: paths.entitlementBearer,
      TEACHER_ENTITLEMENT_BINDING_KEY_FILE: paths.entitlementBinding,
      TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE: paths.entitlementReceipt,
      HARNESS_SAFEGUARDING_DISPATCH_URL:
        "https://safeguarding.organization.example/v1/cases",
      HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET_FILE:
        paths.safeguardingDispatch,
      HARNESS_SAFEGUARDING_DISPATCH_POLICY_VERSION: "school-routing-v1",
      HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION:
        "closed-case-retention-v1",
      HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS: "7776000",
      HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN: "128",
      HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET_FILE:
        paths.safeguardingRetention,
      HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256: "a".repeat(64),
    },
    paths,
    cleanup: () => rmSync(directory, {recursive: true, force: true}),
  };
}

test("production configuration accepts the explicit OIDC/Postgres boundary", () => {
  const fixture = safeProductionEnvironment();
  try {
    withEnvironment(fixture.env, () => {
      const config = new AppConfigService();
      assert.doesNotThrow(() => config.assertSafeForStartup());
      assert.equal(config.authMode, "oidc");
      assert.equal(config.dataBackend, "postgres");
      assert.equal(config.sessionCookieName, "__Host-teachlab_session");
      assert.equal(config.harnessGatewayEnabled, true);
      assert.equal(config.harnessScopeKeys.length, 2);
      assert.equal(config.metricsAccessToken, "production-metrics-token-with-at-least-thirty-two-characters");
      assert.equal(config.teacherEntitlementPolicyVersion, "roles-v7");
      assert.equal(config.teacherEntitlementCacheTtlMs, 5_000);
      assert.equal(config.harnessSafeguardingDispatchConfigured, true);
      assert.equal(config.harnessSafeguardingRetentionConfigured, true);
      assert.equal(
        config.harnessSafeguardingRetentionMinimumClosedAgeSeconds,
        7_776_000
      );
      assert.equal(config.harnessSafeguardingRetentionMaximumCasesPerRun, 128);
      assert.deepEqual(config.harnessWorkerProcessResourceLimits, {
        schema: "teaching_skill_miner.worker_process_resource_limits.v1",
        address_space_bytes: 1_610_612_736,
        file_size_bytes: 536_870_912,
        open_files: 256,
        core_dump_bytes: 0
      });
    });
  } finally {
    fixture.cleanup();
  }
});

test("production fails closed for local identity, memory state, or insecure cookies", () => {
  const fixture = safeProductionEnvironment();
  const cases: Array<[Record<string, string | undefined>, RegExp]> = [
    [{AUTH_MODE: "development"}, /AUTH_MODE=oidc/],
    [{DATA_BACKEND: "memory"}, /DATA_BACKEND=postgres/],
    [{SESSION_COOKIE_SECURE: "false"}, /SESSION_COOKIE_SECURE=true/],
    [{PG_SSL_MODE: "disable"}, /PG_SSL_MODE=verify-full/],
    [{OIDC_CLIENT_ID: ""}, /OIDC_CLIENT_ID/],
    [{OIDC_CLIENT_SECRET_FILE: undefined}, /private secret configuration/],
    [{OIDC_TRANSACTION_SECRET_FILE: undefined}, /private secret configuration/],
    [{TEACHER_ENTITLEMENT_DIRECTORY_URL: undefined}, /authoritative teacher entitlement/],
    [{TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET_FILE: undefined}, /private secret configuration/],
    [{TEACHER_ENTITLEMENT_BINDING_KEY_FILE: undefined}, /private secret configuration/],
    [{TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE: undefined}, /private secret configuration/],
    [{TEACHER_ENTITLEMENT_CACHE_TTL_MS: "5001"}, /cannot exceed 5000/],
    [{HARNESS_SAFEGUARDING_DISPATCH_URL: undefined}, /complete HTTPS endpoint/],
    [{HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION: undefined}, /complete policy/],
    [{HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS: undefined}, /complete policy/],
    [{HARNESS_SAFEGUARDING_RETENTION_MINIMUM_CLOSED_AGE_SECONDS: "7776000x"}, /exact base-10 integer/],
    [{HARNESS_SAFEGUARDING_RETENTION_MAXIMUM_CASES_PER_RUN: "1025"}, /between 1 and 1024/],
    [{HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET_FILE: undefined}, /private secret configuration/],
    [{HARNESS_SAFEGUARDING_RETENTION_DEPLOYMENT_CONTEXT_SHA256: "A".repeat(64)}, /policy identifiers/],
    [{HARNESS_WORKER_RLIMIT_AS_BYTES: undefined}, /explicit fixed HARNESS_WORKER_RLIMIT/],
    [{HARNESS_WORKER_RLIMIT_AS_BYTES: "1073741824"}, /explicit fixed HARNESS_WORKER_RLIMIT/],
    [{HARNESS_WORKER_RLIMIT_AS_BYTES: "1610612736junk"}, /exact base-10 integer/],
    [{HARNESS_WORKER_RLIMIT_FSIZE_BYTES: "335544320"}, /explicit fixed HARNESS_WORKER_RLIMIT/],
    [{HARNESS_WORKER_RLIMIT_NOFILE: "128"}, /explicit fixed HARNESS_WORKER_RLIMIT/],
    [{HARNESS_WORKER_RLIMIT_CORE_BYTES: "1"}, /integer between 0 and 0/],
    [{HARNESS_WORKER_FILESYSTEM_ISOLATION_REQUIRED: "false"}, /filesystem isolation/],
    [{SAFEGUARDING_AUTHORITY_ROLES: "teacher"}, /exact safeguarding role/],
    [{OIDC_REDIRECT_URI: "https://evil.example.test/callback"}, /exact Console callback path/],
    [{OIDC_EXPECTED_HOST: ""}, /OIDC_EXPECTED_HOST/],
    [{
      CORS_ORIGINS: "http://console.example.test",
      OIDC_REDIRECT_URI: "http://console.example.test/api/teacher-agent/security/login/callback"
    }, /must use HTTPS/]
  ];
  try {
    for (const [override, expected] of cases) {
      withEnvironment({...fixture.env, ...override}, () => {
        assert.throws(() => new AppConfigService().assertSafeForStartup(), expected);
      });
    }
  } finally {
    fixture.cleanup();
  }
});

test("production secrets reject inline ambiguity, broad permissions, and symlinks", () => {
  const fixture = safeProductionEnvironment();
  try {
    withEnvironment({...fixture.env, SESSION_SECRET: "x".repeat(40)}, () => {
      assert.throws(() => new AppConfigService(), /private secret configuration/);
    });
    chmodSync(fixture.paths.session, 0o640);
    withEnvironment(fixture.env, () => {
      assert.throws(() => new AppConfigService(), /private secret configuration/);
    });
    chmodSync(fixture.paths.session, 0o600);
    const linked = join(fixture.directory, "linked-session");
    symlinkSync(fixture.paths.session, linked);
    withEnvironment({...fixture.env, SESSION_SECRET_FILE: linked}, () => {
      assert.throws(() => new AppConfigService(), /private secret configuration/);
    });
  } finally {
    fixture.cleanup();
  }
});

test("teacher entitlement credentials are private, distinct, and isolated from service keys", () => {
  const fixture = safeProductionEnvironment();
  try {
    withEnvironment({
      ...fixture.env,
      TEACHER_ENTITLEMENT_DIRECTORY_BEARER_SECRET:
        "inline-entitlement-secret-is-forbidden-in-production"
    }, () => {
      assert.throws(() => new AppConfigService(), /private secret configuration/);
    });
    withEnvironment({
      ...fixture.env,
      TEACHER_ENTITLEMENT_RECEIPT_KEY_FILE: fixture.paths.entitlementBinding
    }, () => {
      assert.throws(
        () => new AppConfigService().assertSafeForStartup(),
        /mutually distinct/
      );
    });
    withEnvironment({
      ...fixture.env,
      TEACHER_ENTITLEMENT_BINDING_KEY_FILE: fixture.paths.session
    }, () => {
      assert.throws(
        () => new AppConfigService().assertSafeForStartup(),
        /isolated from other service keys/
      );
    });
    withEnvironment({
      ...fixture.env,
      HARNESS_WORKER_BACKEND: "deepseek",
      HARNESS_PROVIDER_API_KEY_FILE: fixture.paths.entitlementBinding,
      HARNESS_PROVIDER_POLICY_ID: "deepseek-approved-terms",
      HARNESS_PROVIDER_POLICY_VERSION: "2026-08-12",
      HARNESS_PROVIDER_PROCESSING_REGION: "cn_north",
      HARNESS_PROVIDER_RETENTION_DAYS: "7",
      HARNESS_PROVIDER_DELETION_STATUS:
        "outside_service_control_subject_to_provider_policy",
      HARNESS_PROVIDER_DOCUMENTATION_URL: "https://provider.example/privacy"
    }, () => {
      assert.throws(
        () => new AppConfigService().assertSafeForStartup(),
        /isolated from other service keys/
      );
    });
    withEnvironment({
      ...fixture.env,
      TEACHER_ENTITLEMENT_DIRECTORY_URL:
        "http://entitlements.organization.example/v1/teacher/snapshot"
    }, () => {
      assert.throws(() => new AppConfigService(), /must use HTTPS/);
    });
  } finally {
    fixture.cleanup();
  }
});

test("teacher and safeguarding roles cannot grant each other's authority", () => {
  const fixture = safeProductionEnvironment();
  try {
    for (const roles of ["safeguarding", "teacher,safeguarding"]) {
      withEnvironment({...fixture.env, TEACHER_AUTHORITY_ROLES: roles}, () => {
        assert.throws(
          () => new AppConfigService().assertSafeForStartup(),
          /must be disjoint/
        );
      });
    }
  } finally {
    fixture.cleanup();
  }
});

test("account scope, identity, deletion and browser-cache secrets are distinct", () => {
  const fixture = safeProductionEnvironment();
  try {
    for (const fileVariable of [
      "ACCOUNT_IDENTITY_NAMESPACE_SECRET_FILE",
      "ACCOUNT_DELETION_STATUS_SECRET_FILE",
      "ACCOUNT_CACHE_SCOPE_SECRET_FILE",
    ]) {
      withEnvironment(
        {...fixture.env, [fileVariable]: fixture.paths.accountScope},
        () => {
          assert.throws(
            () => new AppConfigService().assertSafeForStartup(),
            /Account-domain active secrets must be mutually distinct/
          );
        }
      );
    }
  } finally {
    fixture.cleanup();
  }
});

test("safeguarding dispatcher is exact HTTPS and uses an isolated private secret", () => {
  const fixture = safeProductionEnvironment();
  try {
    withEnvironment({
      ...fixture.env,
      HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET:
        "inline-safeguarding-dispatch-secret-is-forbidden"
    }, () => {
      assert.throws(() => new AppConfigService(), /private secret configuration/);
    });
    withEnvironment({
      ...fixture.env,
      HARNESS_SAFEGUARDING_DISPATCH_BEARER_SECRET_FILE: fixture.paths.session
    }, () => {
      assert.throws(
        () => new AppConfigService().assertSafeForStartup(),
        /dispatcher secret must be isolated/
      );
    });
    withEnvironment({
      ...fixture.env,
      HARNESS_WORKER_BACKEND: "deepseek",
      HARNESS_PROVIDER_API_KEY_FILE: fixture.paths.safeguardingDispatch,
      HARNESS_PROVIDER_POLICY_ID: "deepseek-approved-terms",
      HARNESS_PROVIDER_POLICY_VERSION: "2026-08-12",
      HARNESS_PROVIDER_PROCESSING_REGION: "cn_north",
      HARNESS_PROVIDER_RETENTION_DAYS: "7",
      HARNESS_PROVIDER_DELETION_STATUS:
        "outside_service_control_subject_to_provider_policy",
      HARNESS_PROVIDER_DOCUMENTATION_URL: "https://provider.example/privacy"
    }, () => {
      assert.throws(
        () => new AppConfigService().assertSafeForStartup(),
        /dispatcher secret must be isolated/
      );
    });
    withEnvironment({
      ...fixture.env,
      HARNESS_SAFEGUARDING_DISPATCH_URL:
        "http://safeguarding.organization.example/v1/cases"
    }, () => {
      assert.throws(() => new AppConfigService(), /must use HTTPS/);
    });
    withEnvironment({
      ...fixture.env,
      HARNESS_SAFEGUARDING_DISPATCH_URL:
        "https://safeguarding.organization.example:8443/v1/cases"
    }, () => {
      assert.throws(() => new AppConfigService(), /default HTTPS port/);
    });
  } finally {
    fixture.cleanup();
  }
});

test("safeguarding retention is exact, server-only, and pairwise isolated", () => {
  const fixture = safeProductionEnvironment();
  try {
    withEnvironment({
      ...fixture.env,
      HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET:
        "inline-safeguarding-retention-secret-is-forbidden"
    }, () => {
      assert.throws(() => new AppConfigService(), /private secret configuration/);
    });
    for (const reusedPath of [
      fixture.paths.session,
      fixture.paths.safeguardingDispatch,
      fixture.paths.harness
    ]) {
      withEnvironment({
        ...fixture.env,
        HARNESS_SAFEGUARDING_RETENTION_AUTHORITY_SECRET_FILE: reusedPath
      }, () => {
        assert.throws(
          () => new AppConfigService().assertSafeForStartup(),
          /retention authority secret must be isolated/
        );
      });
    }
    withEnvironment({
      ...fixture.env,
      HARNESS_SAFEGUARDING_RETENTION_POLICY_VERSION:
        " closed-case-retention-v1"
    }, () => {
      assert.throws(() => new AppConfigService(), /exact non-control string/);
    });
    chmodSync(fixture.paths.safeguardingRetention, 0o640);
    withEnvironment(fixture.env, () => {
      assert.throws(() => new AppConfigService(), /private secret configuration/);
    });
  } finally {
    fixture.cleanup();
  }
});

test("development identity is restricted to loopback and test-only overrides", () => {
  const localEnvironment = {
    NODE_ENV: "development",
    AUTH_MODE: "development",
    DATA_BACKEND: "memory",
    CORS_ORIGINS: "http://localhost:3000",
    SESSION_SECRET: "development-session-secret-at-least-32-characters",
    SESSION_COOKIE_SECURE: "false",
    SEED_DEMO_SESSIONS: "false"
  };
  withEnvironment({...localEnvironment, API_HOST: "0.0.0.0"}, () => {
    assert.throws(
      () => new AppConfigService().assertSafeForStartup(),
      /requires a loopback API_HOST/
    );
  });
  withEnvironment({...localEnvironment, DEV_AUTH_ALLOW_HEADERS: "true"}, () => {
    assert.throws(
      () => new AppConfigService().assertSafeForStartup(),
      /restricted to NODE_ENV=test/
    );
  });
});

test("worker idle eviction has a bounded explicit configuration", () => {
  const defaults = withEnvironment({}, () => new AppConfigService());
  assert.equal(defaults.harnessWorkerIdleTimeoutMs, 15 * 60 * 1000);
  const configured = withEnvironment(
    {HARNESS_WORKER_IDLE_TIMEOUT_MS: "2500"},
    () => new AppConfigService()
  );
  assert.equal(configured.harnessWorkerIdleTimeoutMs, 2500);
  assert.throws(
    () => withEnvironment(
      {HARNESS_WORKER_IDLE_TIMEOUT_MS: "999"},
      () => new AppConfigService()
    ),
    /HARNESS_WORKER_IDLE_TIMEOUT_MS/
  );
});

test("worker process limits are absent locally and cannot shadow a disabled boundary", () => {
  const defaults = withEnvironment({}, () => new AppConfigService());
  assert.equal(defaults.harnessWorkerFilesystemIsolationRequired, false);
  assert.equal(defaults.harnessWorkerProcessResourceLimits, null);
  assert.throws(
    () => withEnvironment(
      {
        NODE_ENV: "test",
        HARNESS_GATEWAY_ENABLED: "true",
        HARNESS_WORKER_ROOT: "/tmp/teachlab-limit-workers",
        HARNESS_WORKER_CWD: "/tmp/teachlab-limit-app",
        HARNESS_WORKER_PYTHON: "/usr/bin/python3",
        HARNESS_SCOPE_SECRET: "limit-test-harness-secret-at-least-32-characters",
        HARNESS_WORKER_RLIMIT_NOFILE: "256"
      },
      () => new AppConfigService().assertSafeForStartup()
    ),
    /forbidden when worker isolation is not required/
  );
});

test("DeepSeek worker startup requires one complete exact provider policy", () => {
  const base = {
    NODE_ENV: "test",
    HARNESS_GATEWAY_ENABLED: "true",
    HARNESS_WORKER_ROOT: "/tmp/teachlab-policy-workers",
    HARNESS_WORKER_CWD: "/tmp/teachlab-policy-app",
    HARNESS_WORKER_PYTHON: "/usr/bin/python3",
    HARNESS_WORKER_BACKEND: "deepseek",
    HARNESS_SCOPE_SECRET: "policy-test-harness-secret-at-least-32-characters",
    HARNESS_PROVIDER_API_KEY_FILE: "/tmp/provider-key"
  };
  withEnvironment(base, () => {
    assert.throws(
      () => new AppConfigService().assertSafeForStartup(),
      /HARNESS_PROVIDER_POLICY/
    );
  });
  withEnvironment({
    ...base,
    HARNESS_PROVIDER_POLICY_ID: "deepseek-approved-terms",
    HARNESS_PROVIDER_POLICY_VERSION: "2026-08-12",
    HARNESS_PROVIDER_PROCESSING_REGION: "cn_north",
    HARNESS_PROVIDER_RETENTION_DAYS: "7",
    HARNESS_PROVIDER_DELETION_STATUS:
      "outside_service_control_subject_to_provider_policy",
    HARNESS_PROVIDER_DOCUMENTATION_URL: "https://provider.example/privacy"
  }, () => {
    assert.doesNotThrow(() => new AppConfigService().assertSafeForStartup());
  });
});

test("production DeepSeek credential uses the same owned private-file boundary", () => {
  const fixture = safeProductionEnvironment();
  const deepseek = {
    ...fixture.env,
    HARNESS_WORKER_BACKEND: "deepseek",
    HARNESS_PROVIDER_API_KEY_FILE: fixture.paths.provider,
    HARNESS_PROVIDER_POLICY_ID: "deepseek-approved-terms",
    HARNESS_PROVIDER_POLICY_VERSION: "2026-08-12",
    HARNESS_PROVIDER_PROCESSING_REGION: "cn_north",
    HARNESS_PROVIDER_RETENTION_DAYS: "7",
    HARNESS_PROVIDER_DELETION_STATUS:
      "outside_service_control_subject_to_provider_policy",
    HARNESS_PROVIDER_DOCUMENTATION_URL: "https://provider.example/privacy",
  };
  try {
    withEnvironment(deepseek, () => {
      assert.doesNotThrow(() => new AppConfigService().assertSafeForStartup());
    });
    chmodSync(fixture.paths.provider, 0o640);
    withEnvironment(deepseek, () => {
      assert.throws(
        () => new AppConfigService().assertSafeForStartup(),
        /private secret configuration/
      );
    });
    chmodSync(fixture.paths.provider, 0o600);
    const linked = join(fixture.directory, "linked-provider");
    symlinkSync(fixture.paths.provider, linked);
    withEnvironment({...deepseek, HARNESS_PROVIDER_API_KEY_FILE: linked}, () => {
      assert.throws(
        () => new AppConfigService().assertSafeForStartup(),
        /private secret configuration/
      );
    });
  } finally {
    fixture.cleanup();
  }
});

test("sealed session cookies hide identity, reject tampering, and bind CSRF", () => {
  withEnvironment(
    {
      NODE_ENV: "test",
      AUTH_MODE: "development",
      DATA_BACKEND: "memory",
      SESSION_SECRET: "test-session-secret-with-at-least-32-characters",
      SESSION_COOKIE_SECURE: "false",
      SEED_DEMO_SESSIONS: "false"
    },
    () => {
      const config = new AppConfigService();
      config.assertSafeForStartup();
      const service = new SessionCookieService(config);
      const minted = service.mint({
        subject: "raw-student-42",
        tenantId: "raw-school-7",
        provider: "development",
        email: "student42@example.test",
        roles: ["learner"]
      });
      assert.match(minted.cookieValue, /^v3\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/);
      assert.equal(minted.cookieValue.includes("raw-student-42"), false);
      assert.equal(minted.cookieValue.includes("raw-school-7"), false);
      for (const part of minted.cookieValue.split(".").slice(1)) {
        const decoded = Buffer.from(part, "base64url").toString("utf8");
        assert.equal(decoded.includes("raw-student-42"), false);
        assert.equal(decoded.includes("raw-school-7"), false);
        assert.equal(decoded.includes("student42@example.test"), false);
        assert.equal(decoded.includes("learner"), false);
      }
      const headers = {
        cookie: `teachlab_session=${minted.cookieValue}; teachlab_csrf=${minted.csrfToken}`,
        "x-csrf-token": minted.csrfToken
      };
      const authenticated = service.authenticate(headers);
      assert.equal(authenticated?.principal.subject, "raw-student-42");
      assert.equal(authenticated?.principal.tenantId, "raw-school-7");
      assert.equal(authenticated ? service.csrfMatches(authenticated, headers) : false, true);
      assert.equal(
        service.authenticate({cookie: `teachlab_session=${minted.cookieValue}x`}),
        null
      );
      assert.equal(
        authenticated
          ? service.csrfMatches(authenticated, {...headers, "x-csrf-token": "attacker"})
          : true,
        false
      );
    }
  );
});
