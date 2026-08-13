import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {mkdtemp, mkdir, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {test} from "node:test";

import type {AuthenticatedPrincipal} from "../src/auth/auth-provider.port";
import {AccountDataRightsError} from "../src/account/account-data-rights.errors";
import type {
  AccountInMemoryDataPort,
  AccountDeletionFreshAuthorityPort,
  AccountScopeRuntimeCoordinatorPort
} from "../src/account/account-data-rights.repository.port";
import {AccountDataRightsService} from "../src/account/account-data-rights.service";
import type {
  AccountPostgresExportSnapshot,
  AccountRuntimeDrainResult,
  AccountScopeHashes,
  AccountWorkerExportLease,
  AccountWorkerPurgeResult
} from "../src/account/account-data-rights.types";
import {AccountDeletionStatusCookieService} from "../src/account/account-deletion-status-cookie";
import {
  bufferAccountExportEntry,
  validateAccountExportArchive
} from "../src/account/account-export-archive";
import {InMemoryAccountDataRightsRepository} from "../src/account/in-memory-account-data-rights.repository";
import {AccountScopeHasher} from "../src/account/account-scope-hash";
import {HarnessAccountRuntimeCoordinator} from "../src/account/harness-account-runtime-coordinator";
import type {AccessScope} from "../src/tenancy/access-scope";

const CSRF = "c".repeat(43);
const NOW = new Date("2026-08-12T00:00:00.000Z");
const ACTIVE_SCOPE_KEY = "active-account-scope-secret-00000000000000000000000000000000";
const PREVIOUS_SCOPE_KEY = "previous-account-scope-secret-000000000000000000000000000000";
const STATUS_COOKIE_KEY = "account-deletion-status-cookie-secret-000000000000000000000000000";

function principal(subject: string, tenantId = "tenant-a"): AuthenticatedPrincipal {
  return {
    subject,
    tenantId,
    sessionId: subject === "alice"
      ? "00000000-0000-4000-8000-000000000001"
      : "00000000-0000-4000-8000-000000000002",
    provider: "oidc",
    roles: ["learner"],
    scopeTenantId: tenantId,
    scopeOwnerId: subject
  };
}

function scopeKey(scope: AccessScope): string {
  return `${scope.tenantId}\0${scope.ownerId}`;
}

function digest(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

class FreshOidcAuthority implements AccountDeletionFreshAuthorityPort {
  enabled = true;
  assuranceLevel = 2;
  issuer = "https://identity.example.test";
  authenticatedAt = new Date(NOW);
  expiresAt = new Date(NOW.getTime() + 5 * 60 * 1_000);

  async current(value: AuthenticatedPrincipal) {
    if (!this.enabled) return null;
    return {
      bindings: [{
        authorityGrantSha256: digest(`grant-v1\0${value.sessionId}`),
        canonicalIdentitySha256: digest(
          `identity-v1\0${this.issuer}\0${value.tenantId}\0${value.subject}`
        ),
        issuerSha256: digest(`issuer-v1\0${this.issuer}`),
        keyVersion: "test-v1"
      }],
      sessionSha256: digest(value.sessionId),
      authenticatedAt: new Date(this.authenticatedAt),
      expiresAt: new Date(this.expiresAt),
      assuranceLevel: this.assuranceLevel
    };
  }
}

class ScopedMemoryData implements AccountInMemoryDataPort {
  readonly deleted = new Set<string>();

  async snapshot(scope: AccessScope): Promise<AccountPostgresExportSnapshot> {
    if (this.deleted.has(scopeKey(scope))) {
      throw new Error("scope was deleted");
    }
    return {
      capturedAt: NOW.toISOString(),
      sessions: [{id: `session-${scope.ownerId}`, learner: "private learner"}],
      tasks: [{id: `task-${scope.ownerId}`, learner_message: "private response"}],
      events: [{id: `event-${scope.ownerId}`, payload: {private: true}}],
      authSessionAudit: [{session_id_sha256: "a".repeat(64), version: 1}],
      artifacts: [{key: "artifact.txt", byte_length: 3}]
    };
  }

  async deleteScope(scope: AccessScope) {
    this.deleted.add(scopeKey(scope));
    return {
      postgresEvents: 3,
      postgresTasks: 2,
      postgresArtifacts: 1,
      postgresSessions: 1,
      postgresAuthSessions: 4
    };
  }
}

class RecordingRuntime implements AccountScopeRuntimeCoordinatorPort {
  readonly fenced: string[] = [];
  readonly drained: string[] = [];
  readonly quarantined: string[] = [];
  readonly purged: string[] = [];
  releases = 0;
  failQuarantineOnce = false;
  failFenceOnCall: number | null = null;

  async exportWorkerData(
    _scope: AccessScope,
    _hashes: AccountScopeHashes
  ): Promise<AccountWorkerExportLease> {
    const classes = [
      ["projects/projects.json", "learning_projects_private"],
      ["sessions/sessions.jsonl", "teaching_sessions_private"],
      ["syllabi/index.json", "syllabi_private"],
      ["resource_index/index.json", "resource_index_private"],
      ["resource_reviews/reviews.json", "resource_reviews_private"],
      ["learning_records/events.jsonl", "learning_records_private"],
      ["metacognition/events.jsonl", "metacognition_private"],
      ["adjudication/events.jsonl", "adjudication_private"],
      ["consent/receipts.jsonl", "consent_private"],
      ["sessions.jsonl.harness_streams/task_registry.jsonl", "task_registry_private"],
      ["sessions.jsonl.harness_streams/run.jsonl", "harness_journal_private"]
    ] as const;
    return {
      capturedAt: NOW.toISOString(),
      entries: classes.map(([archivePath, dataClass]) =>
        bufferAccountExportEntry({
          archivePath,
          content: "{}\n",
          dataClass
        })
      ),
      release: async () => { this.releases += 1; }
    };
  }

  async fence(scope: AccessScope): Promise<void> {
    this.fenced.push(scopeKey(scope));
    if (this.fenced.length === this.failFenceOnCall) {
      throw new Error("simulated hard-stop boundary");
    }
  }

  async drain(scope: AccessScope): Promise<AccountRuntimeDrainResult> {
    this.drained.push(scopeKey(scope));
    return {
      apiTasksCancelled: 1,
      apiTasksHandedOff: 0,
      workerTasksCancelled: 1,
      workerTasksHandedOff: 0
    };
  }

  async quarantine(scope: AccessScope): Promise<void> {
    this.quarantined.push(scopeKey(scope));
    if (this.failQuarantineOnce) {
      this.failQuarantineOnce = false;
      throw new Error("simulated crash window");
    }
  }

  async purgeQuarantine(scope: AccessScope): Promise<AccountWorkerPurgeResult> {
    this.purged.push(scopeKey(scope));
    return {workerFiles: 12, workerBytes: 4_096, workerRoots: 1};
  }
}

function setup(hasher = new AccountScopeHasher(ACTIVE_SCOPE_KEY)) {
  const data = new ScopedMemoryData();
  const repository = new InMemoryAccountDataRightsRepository(data);
  const runtime = new RecordingRuntime();
  const cookies = new AccountDeletionStatusCookieService({
    secret: STATUS_COOKIE_KEY,
    secure: false
  });
  const authority = new FreshOidcAuthority();
  const service = new AccountDataRightsService(
    repository, runtime, hasher, cookies, authority
  );
  return {authority, data, repository, runtime, cookies, service};
}

function rawCookie(setCookie: string): string {
  return setCookie.split(";", 1)[0]!;
}

function confirmation(
  prepared: Awaited<ReturnType<AccountDataRightsService["prepareDeletion"]>>,
  overrides: Partial<Parameters<AccountDataRightsService["confirmDeletion"]>[1]> = {}
) {
  return {
    challengeId: prepared.response.challenge_id,
    confirmationToken: prepared.response.confirmation_token,
    confirmationPhrase: prepared.response.confirmation_phrase,
    expectedRevision: prepared.response.revision,
    idempotencyKey: "delete-request-00000001",
    ...overrides
  };
}

test("account export covers PostgreSQL audit and every private worker data class", async () => {
  const {service, runtime} = setup();
  const archive = await service.export(principal("alice"), undefined, NOW);
  const chunks: Buffer[] = [];
  for await (const chunk of archive.stream) chunks.push(Buffer.from(chunk));
  const manifest = validateAccountExportArchive(Buffer.concat(chunks));
  assert.deepEqual(
    manifest.entries.filter((entry) => entry.path.startsWith("postgres/"))
      .map((entry) => entry.path),
    [
      "postgres/artifacts.json",
      "postgres/auth-session-audit.json",
      "postgres/events.json",
      "postgres/sessions.json",
      "postgres/tasks.json"
    ]
  );
  for (const required of [
    "projects", "sessions", "syllabi", "resource_index", "resource_reviews",
    "learning_records", "metacognition", "adjudication", "consent"
  ]) {
    assert.ok(manifest.entries.some((entry) => entry.path.startsWith(`worker/${required}/`)), required);
  }
  assert.ok(manifest.entries.some((entry) => entry.path.includes("task_registry")));
  assert.ok(manifest.entries.some((entry) => entry.path.includes("run.jsonl")));
  assert.equal(runtime.releases, 1);
});

test("account worker export excludes only exact curriculum signing security metadata", async () => {
  const root = await mkdtemp(join(tmpdir(), "teachlab-account-export-"));
  try {
    await mkdir(join(root, "syllabi"), {recursive: true});
    const sentinel = "PRIVATE-CURRICULUM-KEY-CIPHERTEXT-SENTINEL";
    await writeFile(
      join(root, "syllabi", ".curriculum_signing_keyring.json"),
      JSON.stringify({private_key_nonce_base64: sentinel, private_key_ciphertext_base64: sentinel})
    );
    await writeFile(
      join(root, "syllabi", "..curriculum_signing_keyring.json.lock"),
      sentinel
    );
    await writeFile(
      join(root, "syllabi", ".curriculum_authority.json"),
      JSON.stringify({public_receipt: "receipt-visible"})
    );
    await writeFile(
      join(root, "syllabi", "teacher-review.json"),
      JSON.stringify({review: "visible"})
    );
    const workers = {
      beginAccountExport: async () => ({
        roots: [{keyVersion: "k1", scopeId: `scope_${"a".repeat(48)}`, privateRoot: root}],
        release: async () => undefined
      })
    };
    const coordinator = new HarnessAccountRuntimeCoordinator(
      workers as never,
      {} as never
    );
    const lease = await coordinator.exportWorkerData(
      {tenantId: "tenant-a", ownerId: "alice"},
      {} as AccountScopeHashes
    );
    const paths = lease.entries.map((entry) => entry.archivePath);
    assert.equal(paths.some((path) => path.includes("curriculum_signing_keyring")), false);
    assert.equal(paths.some((path) => path.endsWith("syllabi/.curriculum_authority.json")), true);
    assert.equal(paths.some((path) => path.endsWith("syllabi/teacher-review.json")), true);
    const chunks: Buffer[] = [];
    for (const entry of lease.entries) {
      for await (const chunk of entry.open()) chunks.push(Buffer.from(chunk));
    }
    const content = Buffer.concat(chunks);
    assert.equal(content.includes(sentinel), false);
    await lease.release();
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test("account deletion derives scope from principal, deletes all devices, and leaves content-free receipt", async () => {
  const {data, repository, runtime, service} = setup();
  const alice = principal("alice");
  const bob = principal("bob");
  const prepared = await service.prepareDeletion(alice, CSRF, NOW);
  const cookie = rawCookie(prepared.statusCookie);

  await assert.rejects(
    service.confirmDeletion(
      bob,
      confirmation(prepared),
      CSRF,
      cookie,
      new Date(NOW.getTime() + 1_000)
    ),
    (error) => error instanceof AccountDataRightsError
      && error.code === "account_deletion_status_invalid"
  );
  assert.equal(runtime.fenced.length, 0);

  const receipt = await service.confirmDeletion(
    alice,
    confirmation(prepared),
    CSRF,
    cookie,
    new Date(NOW.getTime() + 1_000)
  );
  assert.equal(receipt.status, "permanently_deleted");
  assert.equal(receipt.deleted_counts.postgres_auth_sessions, 4);
  assert.equal(receipt.deleted_counts.worker_files, 12);
  assert.equal(receipt.all_devices_session_authority_deleted, true);
  assert.equal(receipt.remote_provider_copies_deleted, false);
  assert.equal(
    receipt.remote_provider_copies_status,
    "outside_service_control_subject_to_provider_retention"
  );
  assert.equal(receipt.identity_provider_account_deleted, false);
  assert.equal(
    receipt.identity_provider_account_status,
    "outside_service_control_contact_organization_idp"
  );
  assert.equal(receipt.operator_backup_copies_deleted, false);
  assert.equal(
    receipt.operator_backup_copies_status,
    "pending_retention_expiry_or_operator_crypto_erasure"
  );
  assert.equal(JSON.stringify(receipt).includes("tenant-a"), false);
  assert.equal(JSON.stringify(receipt).includes("alice"), false);
  assert.equal(data.deleted.has("tenant-a\0alice"), true);
  assert.equal(data.deleted.has("tenant-a\0bob"), false);
  assert.equal(runtime.fenced.length, 6);
  assert.ok(runtime.fenced.every((value) => value === "tenant-a\0alice"));
  assert.deepEqual(runtime.drained, ["tenant-a\0alice"]);

  const status = await service.status(
    cookie,
    undefined,
    new Date(NOW.getTime() + 2_000)
  );
  assert.equal(status.status, "permanently_deleted");
  assert.deepEqual(status.receipt, receipt);
  const bobLifecycle = await repository.lifecycle(
    {tenantId: bob.tenantId, ownerId: bob.subject},
    new AccountScopeHasher(ACTIVE_SCOPE_KEY).hashes({
      tenantId: bob.tenantId,
      ownerId: bob.subject
    })
  );
  assert.equal(bobLifecycle.kind, "active");
});

test("deletion crash window is retryable with exact idempotency and never repeats final database effects", async () => {
  const {data, runtime, service} = setup();
  runtime.failQuarantineOnce = true;
  const alice = principal("alice");
  const prepared = await service.prepareDeletion(alice, CSRF, NOW);
  const cookie = rawCookie(prepared.statusCookie);
  const input = confirmation(prepared);
  await assert.rejects(
    service.confirmDeletion(
      alice,
      input,
      CSRF,
      cookie,
      new Date(NOW.getTime() + 1_000)
    ),
    (error) => error instanceof AccountDataRightsError
      && error.code === "account_deletion_retry_required"
  );
  assert.equal(data.deleted.size, 0);
  const failedStatus = await service.status(
    cookie,
    alice,
    new Date(NOW.getTime() + 2_000)
  );
  assert.equal(failedStatus.status, "retryable_failure");
  assert.equal(failedStatus.phase, "quarantining");

  const receipt = await service.confirmDeletion(
    alice,
    input,
    CSRF,
    cookie,
    new Date(NOW.getTime() + 3_000)
  );
  assert.equal(receipt.status, "permanently_deleted");
  assert.equal(data.deleted.size, 1);
  assert.equal(runtime.fenced.length, 7);
  assert.equal(runtime.drained.length, 1);
  assert.equal(runtime.quarantined.length, 2);
  assert.equal(runtime.purged.length, 1);
});

test("every durable deletion phase resumes in a fresh service through the HttpOnly capability", async () => {
  const phases = [
    "fencing", "draining", "tombstoned", "quarantining",
    "purging_worker_data", "committing_database"
  ] as const;
  for (const [index, phase] of phases.entries()) {
    const {authority, data, repository, runtime, cookies, service} = setup();
    runtime.failFenceOnCall = index + 1;
    const alice = principal("alice");
    const prepared = await service.prepareDeletion(alice, CSRF, NOW);
    const cookie = rawCookie(prepared.statusCookie);
    await assert.rejects(service.confirmDeletion(
      alice,
      confirmation(prepared),
      CSRF,
      cookie,
      new Date(NOW.getTime() + 1_000)
    ), /account_deletion_retry_required/);
    assert.equal((await service.status(cookie, undefined)).phase, phase);

    const restarted = new AccountDataRightsService(
      repository,
      runtime,
      new AccountScopeHasher(ACTIVE_SCOPE_KEY),
      cookies,
      authority
    );
    const recovered = await restarted.resume(cookie);
    assert.equal(recovered.status, "permanently_deleted", phase);
    assert.equal(recovered.receipt?.status, "permanently_deleted", phase);
    assert.equal(data.deleted.size, 1, phase);
  }
});

test("startup recovery respects an active foreign lease then takes over after expiry", async () => {
  const {authority, data, repository, runtime, cookies, service} = setup();
  runtime.failFenceOnCall = 1;
  const alice = principal("alice");
  const prepared = await service.prepareDeletion(alice, CSRF, NOW);
  const cookie = rawCookie(prepared.statusCookie);
  await assert.rejects(service.confirmDeletion(
    alice,
    confirmation(prepared),
    CSRF,
    cookie,
    new Date(NOW.getTime() + 1_000)
  ), /account_deletion_retry_required/);
  const scope = {tenantId: alice.tenantId, ownerId: alice.subject};
  const hashes = new AccountScopeHasher(ACTIVE_SCOPE_KEY).hashes(scope);
  const lifecycle = await repository.lifecycle(scope, hashes);
  assert.equal(lifecycle.kind, "deleting");
  if (lifecycle.kind !== "deleting") throw new Error("deletion operation missing");
  const deadLease = await repository.claimDeletionForScope(
    scope,
    hashes,
    lifecycle.operation.operationId,
    {
      leaseOwnerSha256: "8".repeat(64),
      leaseTokenSha256: "9".repeat(64),
      leaseDurationMs: 1_000,
      now: new Date(),
      ignoreRecoveryAfter: true
    }
  );
  assert.equal(deadLease.kind, "claimed");

  const restarted = new AccountDataRightsService(
    repository,
    runtime,
    new AccountScopeHasher(ACTIVE_SCOPE_KEY),
    cookies,
    authority
  );
  await restarted.onApplicationBootstrap();
  assert.equal(data.deleted.size, 0, "an unexpired foreign lease must not be stolen");
  await new Promise((resolve) => setTimeout(resolve, 1_050));
  await restarted.recoverPendingDeletions();
  assert.equal(data.deleted.size, 1);
  assert.equal((await restarted.status(cookie)).status, "permanently_deleted");
  await restarted.onModuleDestroy();
});

test("a started deletion survives retirement of its original scope-HMAC key", async () => {
  const oldHasher = new AccountScopeHasher(PREVIOUS_SCOPE_KEY);
  const {authority, data, repository, runtime, cookies, service} = setup(oldHasher);
  runtime.failFenceOnCall = 4;
  const alice = principal("alice");
  const prepared = await service.prepareDeletion(alice, CSRF, NOW);
  const cookie = rawCookie(prepared.statusCookie);
  await assert.rejects(service.confirmDeletion(
    alice,
    confirmation(prepared),
    CSRF,
    cookie,
    new Date(NOW.getTime() + 1_000)
  ), /account_deletion_retry_required/);

  // The previous key is deliberately absent. The signed capability selects
  // the durable operation, while raw scope equality prevents cross-account
  // recovery and the new digest is tombstoned before final commit.
  const restarted = new AccountDataRightsService(
    repository,
    runtime,
    new AccountScopeHasher(ACTIVE_SCOPE_KEY),
    cookies,
    authority
  );
  assert.equal((await restarted.status(cookie, alice)).status, "retryable_failure");
  const completed = await restarted.resume(cookie);
  assert.equal(completed.status, "permanently_deleted");
  assert.equal(data.deleted.size, 1);
  const currentHashes = new AccountScopeHasher(ACTIVE_SCOPE_KEY).hashes({
    tenantId: alice.tenantId,
    ownerId: alice.subject
  });
  assert.equal(
    (await repository.lifecycle(
      {tenantId: alice.tenantId, ownerId: alice.subject},
      currentHashes
    )).kind,
    "deleted"
  );
});

test("expired, mutated, and mistyped confirmation fail before runtime side effects", async () => {
  const {runtime, service} = setup();
  const alice = principal("alice");
  const prepared = await service.prepareDeletion(alice, CSRF, NOW);
  const cookie = rawCookie(prepared.statusCookie);
  await assert.rejects(
    service.confirmDeletion(
      alice,
      confirmation(prepared, {confirmationPhrase: "DELETE"}),
      CSRF,
      cookie,
      new Date(NOW.getTime() + 1_000)
    ),
    /account_deletion_confirmation_invalid/
  );
  await assert.rejects(
    service.confirmDeletion(
      alice,
      confirmation(prepared),
      CSRF,
      cookie,
      new Date(NOW.getTime() + 5 * 60 * 1_000 + 1)
    ),
    /account_deletion_challenge_expired/
  );
  assert.equal(runtime.fenced.length, 0);
});

test("scope-key rotation discovers the prior operation and preserves the original receipt hash", async () => {
  const oldHasher = new AccountScopeHasher(PREVIOUS_SCOPE_KEY);
  const {
    authority, data, repository, runtime, cookies, service: oldService
  } = setup(oldHasher);
  const alice = principal("alice");
  const prepared = await oldService.prepareDeletion(alice, CSRF, NOW);
  const cookie = rawCookie(prepared.statusCookie);
  const rotatedService = new AccountDataRightsService(
    repository,
    runtime,
    new AccountScopeHasher(ACTIVE_SCOPE_KEY, [PREVIOUS_SCOPE_KEY]),
    cookies,
    authority
  );
  const receipt = await rotatedService.confirmDeletion(
    alice,
    confirmation(prepared),
    CSRF,
    cookie,
    new Date(NOW.getTime() + 1_000)
  );
  assert.equal(data.deleted.size, 1);
  const status = await rotatedService.status(
    cookie,
    undefined,
    new Date(NOW.getTime() + 2_000)
  );
  assert.equal(status.receipt?.receipt_sha256, receipt.receipt_sha256);
  assert.equal(status.receipt?.scope_sha256, receipt.scope_sha256);
});

test("tampered deletion status capability cannot expose status or authorize deletion", async () => {
  const {runtime, service} = setup();
  const alice = principal("alice");
  const prepared = await service.prepareDeletion(alice, CSRF, NOW);
  const cookie = rawCookie(prepared.statusCookie);
  const tampered = `${cookie.slice(0, -1)}${cookie.endsWith("a") ? "b" : "a"}`;
  await assert.rejects(
    service.status(tampered, alice, new Date(NOW.getTime() + 1_000)),
    /account_deletion_status_invalid/
  );
  await assert.rejects(
    service.confirmDeletion(
      alice,
      confirmation(prepared),
      CSRF,
      tampered,
      new Date(NOW.getTime() + 1_000)
    ),
    /account_deletion_status_invalid/
  );
  assert.equal(runtime.fenced.length, 0);
});

test("account deletion requires a recent issuer-bound AAL2 server authority", async () => {
  const {authority, runtime, service} = setup();
  const alice = principal("alice");
  authority.enabled = false;
  await assert.rejects(
    service.export(alice, undefined, NOW),
    /account_deletion_reauthentication_required/
  );
  assert.equal(runtime.releases, 0);
  await assert.rejects(
    service.prepareDeletion(alice, CSRF, NOW),
    /account_deletion_reauthentication_required/
  );
  authority.enabled = true;
  authority.assuranceLevel = 1;
  await assert.rejects(
    service.prepareDeletion(alice, CSRF, NOW),
    /account_deletion_assurance_insufficient/
  );
  authority.assuranceLevel = 2;
  const prepared = await service.prepareDeletion(alice, CSRF, NOW);
  const cookie = rawCookie(prepared.statusCookie);
  authority.issuer = "https://different-issuer.example.test";
  await assert.rejects(
    service.confirmDeletion(
      alice,
      confirmation(prepared),
      CSRF,
      cookie,
      new Date(NOW.getTime() + 1_000)
    ),
    /account_deletion_reauthentication_required/
  );
  assert.equal(runtime.fenced.length, 0);
});

test("a started deletion retry remains bound to the preparing session and CSRF", async () => {
  const {runtime, service} = setup();
  runtime.failQuarantineOnce = true;
  const alice = principal("alice");
  const prepared = await service.prepareDeletion(alice, CSRF, NOW);
  const cookie = rawCookie(prepared.statusCookie);
  const input = confirmation(prepared);
  await assert.rejects(
    service.confirmDeletion(
      alice, input, CSRF, cookie, new Date(NOW.getTime() + 1_000)
    ),
    /account_deletion_retry_required/
  );
  await assert.rejects(
    service.confirmDeletion(
      {...alice, sessionId: "00000000-0000-4000-8000-000000000099"},
      input,
      CSRF,
      cookie,
      new Date(NOW.getTime() + 2_000)
    ),
    /account_deletion_confirmation_invalid/
  );
  await assert.rejects(
    service.confirmDeletion(
      alice,
      input,
      "x".repeat(43),
      cookie,
      new Date(NOW.getTime() + 2_000)
    ),
    /account_deletion_confirmation_invalid/
  );
  assert.equal(runtime.quarantined.length, 1);
});
