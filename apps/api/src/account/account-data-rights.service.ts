import {createHash, randomBytes} from "node:crypto";

import {
  Inject,
  Injectable,
  Logger,
  OnApplicationBootstrap,
  OnModuleDestroy
} from "@nestjs/common";

import type {AuthenticatedPrincipal} from "../auth/auth-provider.port";
import type {AccessScope} from "../tenancy/access-scope";
import {accessScopeFor} from "../tenancy/access-scope";
import {accountDeletionReceipt, accountDeletionStatusProjection, sha256Text} from "./account-deletion-receipt";
import {AccountDataRightsError} from "./account-data-rights.errors";
import type {
  AccountDataRightsRepositoryPort,
  AccountDeletionLeaseClaim,
  AccountDeletionFreshAuthorityPort,
  AccountDeletionStatusCookiePort,
  AccountScopeRuntimeCoordinatorPort
} from "./account-data-rights.repository.port";
import {
  ACCOUNT_DATA_RIGHTS_REPOSITORY,
  ACCOUNT_DELETION_FRESH_AUTHORITY,
  ACCOUNT_DELETION_STATUS_COOKIES,
  ACCOUNT_SCOPE_HASHER,
  ACCOUNT_SCOPE_RUNTIME_COORDINATOR
} from "./account-data-rights.tokens";
import {
  ACCOUNT_DELETION_CONFIRMATION_PHRASE,
  type AccountDeletionConfirmationInput,
  type AccountDeletionFreshAuthority,
  type AccountDeletionFreshAuthorityBinding,
  type AccountDeletionOperationRecord,
  type AccountDeletionPrepared,
  type AccountDeletionReceipt,
  type AccountDeletionStatusProjection,
  type AccountDeletionTombstone,
  type AccountExportArchive,
  type AccountScopeHashes
} from "./account-data-rights.types";
import {
  bufferAccountExportEntry,
  buildAccountExportArchive,
  canonicalAccountJson
} from "./account-export-archive";
import {AccountScopeHasher} from "./account-scope-hash";

export const ACCOUNT_DELETION_CHALLENGE_TTL_MS = 5 * 60 * 1_000;
export const ACCOUNT_DELETION_STATUS_TTL_MS = 30 * 24 * 60 * 60 * 1_000;
export const ACCOUNT_DELETION_DRAIN_TIMEOUT_MS = 10_000;
export const ACCOUNT_DELETION_REAUTH_MAX_AGE_MS = 5 * 60 * 1_000;
export const ACCOUNT_DELETION_MIN_AUTHORITY_LIFETIME_MS = 30 * 1_000;
export const ACCOUNT_DELETION_REQUIRED_AAL = 2;
export const ACCOUNT_DELETION_RECOVERY_LEASE_MS = 30_000;
export const ACCOUNT_DELETION_RECOVERY_SCAN_MS = 2_000;
export const ACCOUNT_DELETION_RECOVERY_BATCH = 8;
export const ACCOUNT_DELETION_RECOVERY_RETRY_BASE_MS = 2_000;
export const ACCOUNT_DELETION_RECOVERY_RETRY_MAX_MS = 60_000;

const CSRF_PATTERN = /^[A-Za-z0-9_-]{43}$/;
const CHALLENGE_PATTERN = /^adelc_[0-9a-f]{32}$/;
const IDEMPOTENCY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$/;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const AUTHORITY_KEY_VERSION_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;

interface PreparedDeletionResult {
  response: AccountDeletionPrepared;
  statusCookie: string;
}

interface HeldDeletionLease {
  ownerSha256: string;
  tokenSha256: string;
}

function sha256Bytes(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function opaqueId(prefix: "adel" | "adelc" | "adelr"): string {
  return `${prefix}_${randomBytes(16).toString("hex")}`;
}

function safeCsrf(value: string): string {
  if (!CSRF_PATTERN.test(value)) {
    throw new AccountDataRightsError(400, "account_deletion_confirmation_invalid");
  }
  return value;
}

function workerEntryPath(path: string): string {
  return `worker/${path}`;
}

function validAuthorityBinding(
  value: AccountDeletionFreshAuthorityBinding
): boolean {
  return SHA256_PATTERN.test(value.authorityGrantSha256)
    && SHA256_PATTERN.test(value.canonicalIdentitySha256)
    && SHA256_PATTERN.test(value.issuerSha256)
    && AUTHORITY_KEY_VERSION_PATTERN.test(value.keyVersion);
}

@Injectable()
export class AccountDataRightsService
  implements OnApplicationBootstrap, OnModuleDestroy {
  private readonly logger = new Logger(AccountDataRightsService.name);
  private readonly recoveryOwnerSha256 = sha256Text(
    `account-deletion-recovery\0${randomBytes(32).toString("base64url")}`
  );
  private recoveryTimer?: NodeJS.Timeout;
  private recoveryScan?: Promise<void>;
  constructor(
    @Inject(ACCOUNT_DATA_RIGHTS_REPOSITORY)
    private readonly repository: AccountDataRightsRepositoryPort,
    @Inject(ACCOUNT_SCOPE_RUNTIME_COORDINATOR)
    private readonly runtime: AccountScopeRuntimeCoordinatorPort,
    @Inject(ACCOUNT_SCOPE_HASHER)
    private readonly scopeHasher: AccountScopeHasher,
    @Inject(ACCOUNT_DELETION_STATUS_COOKIES)
    private readonly statusCookies: AccountDeletionStatusCookiePort,
    @Inject(ACCOUNT_DELETION_FRESH_AUTHORITY)
    private readonly freshAuthority: AccountDeletionFreshAuthorityPort
  ) {}

  onApplicationBootstrap(): void {
    // Startup must become ready even when a large account is mid-purge. The
    // first scan starts immediately but remains background, bounded and
    // single-flight; the durable lifecycle fence already blocks late writes.
    void this.recoverPendingDeletions();
    this.recoveryTimer = setInterval(() => {
      void this.recoverPendingDeletions();
    }, ACCOUNT_DELETION_RECOVERY_SCAN_MS);
    this.recoveryTimer.unref();
  }

  async onModuleDestroy(): Promise<void> {
    if (this.recoveryTimer) clearInterval(this.recoveryTimer);
    await this.recoveryScan?.catch(() => undefined);
  }

  /** Bounded and single-flight so readiness and periodic scans cannot fan out. */
  async recoverPendingDeletions(): Promise<void> {
    if (this.recoveryScan) return this.recoveryScan;
    const scan = this.runRecoveryScan();
    this.recoveryScan = scan;
    try {
      await scan;
    } finally {
      if (this.recoveryScan === scan) this.recoveryScan = undefined;
    }
  }

  private async runRecoveryScan(): Promise<void> {
    for (let index = 0; index < ACCOUNT_DELETION_RECOVERY_BATCH; index += 1) {
      const lease = this.newLease();
      let operation: AccountDeletionOperationRecord | undefined;
      try {
        operation = await this.repository.claimNextDeletion(
          this.leaseClaim(lease, false)
        );
        if (!operation) return;
        const scope = {tenantId: operation.tenantId, ownerId: operation.ownerId};
        await this.continueDeletion(
          scope,
          this.recoveryHashes(scope, operation),
          lease,
          operation
        );
      } catch (error) {
        this.logger.warn({
          event: "account_deletion_recovery_retry_scheduled",
          phase: operation?.phase ?? "claim",
          operation_id_sha256: operation ? sha256Text(operation.operationId) : undefined,
          code: error instanceof AccountDataRightsError
            ? error.code
            : "account_deletion_retry_required"
        });
        if (!operation) return;
      }
    }
  }

  async export(
    principal: AuthenticatedPrincipal,
    signal?: AbortSignal,
    now = new Date()
  ): Promise<AccountExportArchive> {
    await this.requireFreshAuthority(principal, now);
    const scope = accessScopeFor(principal);
    const hashes = this.scopeHasher.hashes(scope);
    const state = await this.repository.lifecycle(scope, hashes);
    if (state.kind === "deleting") {
      throw new AccountDataRightsError(409, "account_deletion_already_started");
    }
    if (state.kind === "deleted") {
      throw new AccountDataRightsError(410, "account_already_deleted");
    }
    const worker = await this.runtime.exportWorkerData(scope, hashes, signal);
    let releaseOwned = true;
    try {
      const postgres = await this.repository.exportPostgresSnapshot(scope, hashes);
      const postgresEntries = [
        bufferAccountExportEntry({
          archivePath: "postgres/sessions.json",
          content: canonicalAccountJson(postgres.sessions),
          dataClass: "teaching_sessions_private"
        }),
        bufferAccountExportEntry({
          archivePath: "postgres/tasks.json",
          content: canonicalAccountJson(postgres.tasks),
          dataClass: "agent_tasks_private"
        }),
        bufferAccountExportEntry({
          archivePath: "postgres/events.json",
          content: canonicalAccountJson(postgres.events),
          dataClass: "task_events_private"
        }),
        bufferAccountExportEntry({
          archivePath: "postgres/auth-session-audit.json",
          content: canonicalAccountJson(postgres.authSessionAudit),
          dataClass: "hash_only_auth_session_audit"
        })
      ];
      if (postgres.artifacts?.length) {
        postgresEntries.push(bufferAccountExportEntry({
          archivePath: "postgres/artifacts.json",
          content: canonicalAccountJson(postgres.artifacts),
          dataClass: "durable_artifacts_private"
        }));
      }
      const workerEntries = worker.entries.map((entry) => ({
        ...entry,
        archivePath: workerEntryPath(entry.archivePath)
      }));
      const archive = buildAccountExportArchive({
        entries: [...postgresEntries, ...workerEntries],
        exportedAt: new Date().toISOString(),
        postgresCapturedAt: postgres.capturedAt,
        workerCapturedAt: worker.capturedAt,
        signal,
        onFinally: worker.release
      });
      releaseOwned = false;
      return archive;
    } finally {
      if (releaseOwned) await worker.release();
    }
  }

  async prepareDeletion(
    principal: AuthenticatedPrincipal,
    csrfToken: string,
    now = new Date()
  ): Promise<PreparedDeletionResult> {
    safeCsrf(csrfToken);
    const scope = accessScopeFor(principal);
    const hashes = this.scopeHasher.hashes(scope);
    const lifecycle = await this.repository.lifecycle(scope, hashes);
    if (lifecycle.kind === "deleting") {
      throw new AccountDataRightsError(409, "account_deletion_already_started");
    }
    if (lifecycle.kind === "deleted") {
      throw new AccountDataRightsError(410, "account_already_deleted");
    }
    const authority = await this.requireFreshAuthority(principal, now);
    const activeAuthority = authority.bindings[0]!;
    const operationId = opaqueId("adel");
    const challengeId = opaqueId("adelc");
    const confirmationToken = randomBytes(32).toString("base64url");
    const statusCapability = randomBytes(32).toString("base64url");
    const challengeExpiresAt = new Date(Math.min(
      now.getTime() + ACCOUNT_DELETION_CHALLENGE_TTL_MS,
      authority.expiresAt.getTime()
    ));
    const statusExpiresAt = new Date(now.getTime() + ACCOUNT_DELETION_STATUS_TTL_MS);
    const record = await this.repository.prepareDeletion({
      ...scope,
      hashes,
      operationId,
      challengeId,
      challengeTokenSha256: sha256Text(confirmationToken),
      challengeCsrfSha256: sha256Text(csrfToken),
      challengeSessionSha256: sha256Text(principal.sessionId),
      challengeAuthorityGrantSha256: activeAuthority.authorityGrantSha256,
      challengeCanonicalIdentitySha256: activeAuthority.canonicalIdentitySha256,
      challengeIssuerSha256: activeAuthority.issuerSha256,
      challengeAuthorityKeyVersion: activeAuthority.keyVersion,
      challengeAuthenticatedAt: authority.authenticatedAt,
      challengeAssuranceLevel: authority.assuranceLevel,
      challengeExpiresAt,
      statusCapabilitySha256: sha256Text(statusCapability),
      now
    });
    const value = this.statusCookies.mint({
      operationId,
      scopeSha256: record.scopeSha256,
      capability: statusCapability,
      expiresAt: statusExpiresAt
    });
    return {
      response: {
        schema: "teachlab.account_deletion_challenge.v1",
        challenge_id: challengeId,
        confirmation_token: confirmationToken,
        confirmation_phrase: ACCOUNT_DELETION_CONFIRMATION_PHRASE,
        expires_at: challengeExpiresAt.toISOString(),
        revision: record.revision
      },
      statusCookie: this.statusCookies.serialize(value, statusExpiresAt)
    };
  }

  async confirmDeletion(
    principal: AuthenticatedPrincipal,
    input: AccountDeletionConfirmationInput,
    csrfToken: string,
    rawCookieHeader: string | undefined,
    now = new Date()
  ): Promise<AccountDeletionReceipt> {
    this.assertConfirmationInput(input);
    safeCsrf(csrfToken);
    const scope = accessScopeFor(principal);
    const hashes = this.scopeHasher.hashes(scope);
    const capability = this.statusCookies.verify(rawCookieHeader, now);
    if (!capability) {
      throw new AccountDataRightsError(401, "account_deletion_status_invalid");
    }
    const lookup = {
      ...capability,
      statusCapabilitySha256: sha256Text(capability.capability)
    };
    let current = await this.repository.statusForScope(scope, hashes, lookup);
    if (!current) {
      const recoverable = await this.repository.statusByCapability(lookup);
      if (recoverable) {
        if (
          !("operationId" in recoverable)
          || recoverable.tenantId !== scope.tenantId
          || recoverable.ownerId !== scope.ownerId
        ) {
          throw new AccountDataRightsError(401, "account_deletion_status_invalid");
        }
        current = recoverable;
      }
    }
    if (!current || !("operationId" in current)) {
      throw new AccountDataRightsError(410, "account_deletion_challenge_missing");
    }
    const operationHashes = this.recoveryHashes(scope, current);
    if (
      current.challengeSessionSha256 !== sha256Text(principal.sessionId)
      || current.challengeCsrfSha256 !== sha256Text(csrfToken)
    ) {
      throw new AccountDataRightsError(
        400,
        "account_deletion_confirmation_invalid"
      );
    }
    let authorityFields = {
      challengeAuthorityGrantSha256: current.challengeAuthorityGrantSha256,
      challengeCanonicalIdentitySha256: current.challengeCanonicalIdentitySha256,
      challengeIssuerSha256: current.challengeIssuerSha256,
      challengeAuthorityKeyVersion: current.challengeAuthorityKeyVersion,
      challengeAuthenticatedAt: current.challengeAuthenticatedAt,
      challengeAssuranceLevel: current.challengeAssuranceLevel
    };
    if (current.phase === "prepared") {
      if (current.challengeExpiresAt.getTime() <= now.getTime()) {
        throw new AccountDataRightsError(
          410,
          "account_deletion_challenge_expired"
        );
      }
      const authority = await this.requireFreshAuthority(principal, now);
      const matchingBinding = authority.bindings.find((binding) =>
        binding.authorityGrantSha256 === current.challengeAuthorityGrantSha256
        && binding.canonicalIdentitySha256 === current.challengeCanonicalIdentitySha256
        && binding.issuerSha256 === current.challengeIssuerSha256
        && binding.keyVersion === current.challengeAuthorityKeyVersion
      );
      if (
        !matchingBinding
        || authority.authenticatedAt.getTime() !== current.challengeAuthenticatedAt.getTime()
        || authority.assuranceLevel !== current.challengeAssuranceLevel
        || authority.expiresAt.getTime() < current.challengeExpiresAt.getTime()
      ) {
        throw new AccountDataRightsError(
          401,
          "account_deletion_reauthentication_required"
        );
      }
      authorityFields = {
        challengeAuthorityGrantSha256: matchingBinding.authorityGrantSha256,
        challengeCanonicalIdentitySha256: matchingBinding.canonicalIdentitySha256,
        challengeIssuerSha256: matchingBinding.issuerSha256,
        challengeAuthorityKeyVersion: matchingBinding.keyVersion,
        challengeAuthenticatedAt: authority.authenticatedAt,
        challengeAssuranceLevel: authority.assuranceLevel
      };
    }
    const confirmationSha256 = sha256Bytes(canonicalAccountJson({
      challenge_id: input.challengeId,
      confirmation_token: input.confirmationToken,
      confirmation_phrase: input.confirmationPhrase,
      expected_revision: input.expectedRevision,
      idempotency_key: input.idempotencyKey
    }));
    const begun = await this.repository.beginDeletion({
      ...scope,
      hashes: operationHashes,
      operationId: capability.operationId,
      challengeId: input.challengeId,
      challengeTokenSha256: sha256Text(input.confirmationToken),
      challengeCsrfSha256: sha256Text(csrfToken),
      challengeSessionSha256: sha256Text(principal.sessionId),
      ...authorityFields,
      expectedRevision: input.expectedRevision,
      idempotencyKeySha256: sha256Text(input.idempotencyKey),
      confirmationSha256,
      now
    });
    switch (begun.kind) {
      case "missing":
        throw new AccountDataRightsError(410, "account_deletion_challenge_missing");
      case "expired":
        throw new AccountDataRightsError(410, "account_deletion_challenge_expired");
      case "revision_conflict":
        throw new AccountDataRightsError(409, "account_deletion_revision_conflict");
      case "payload_conflict":
        throw new AccountDataRightsError(409, "account_deletion_idempotency_conflict");
      default:
        break;
    }
    const lease = this.newLease();
    const claimed = await this.repository.claimDeletionForScope(
      scope,
      operationHashes,
      begun.operation.operationId,
      this.leaseClaim(lease, true)
    );
    if (claimed.kind === "completed") {
      return this.receiptFromTombstone(claimed.tombstone, capability.operationId);
    }
    if (claimed.kind !== "claimed") {
      throw new AccountDataRightsError(503, "account_deletion_retry_required");
    }
    return this.continueDeletion(
      scope,
      this.recoveryHashes(scope, claimed.operation),
      lease,
      claimed.operation
    );
  }

  async status(
    rawCookieHeader: string | undefined,
    principal?: AuthenticatedPrincipal,
    now = new Date()
  ): Promise<AccountDeletionStatusProjection> {
    const capability = this.statusCookies.verify(rawCookieHeader, now);
    if (!capability) {
      throw new AccountDataRightsError(401, "account_deletion_status_invalid");
    }
    let record: AccountDeletionOperationRecord | AccountDeletionTombstone | undefined;
    if (principal) {
      const scope = accessScopeFor(principal);
      const hashes = this.scopeHasher.hashes(scope);
      const lookup = {
        ...capability,
        statusCapabilitySha256: sha256Text(capability.capability)
      };
      record = await this.repository.statusForScope(scope, hashes, lookup);
      if (!record) {
        const recoverable = await this.repository.statusByCapability(lookup);
        if (
          recoverable
          && "operationId" in recoverable
          && recoverable.tenantId === scope.tenantId
          && recoverable.ownerId === scope.ownerId
        ) record = recoverable;
      }
    } else {
      record = await this.repository.statusByCapability({
        ...capability,
        statusCapabilitySha256: sha256Text(capability.capability)
      });
    }
    if (!record) {
      throw new AccountDataRightsError(401, "account_deletion_status_invalid");
    }
    const receipt = record.phase === "completed"
      ? this.receiptFromTombstone(record, capability.operationId)
      : undefined;
    return accountDeletionStatusProjection(record, receipt);
  }

  /**
   * Resume uses only the signed HttpOnly status capability. It never restores
   * account authority and returns only the same content-free status projection.
   */
  async resume(
    rawCookieHeader: string | undefined,
    now = new Date()
  ): Promise<AccountDeletionStatusProjection> {
    const capability = this.statusCookies.verify(rawCookieHeader, now);
    if (!capability) {
      throw new AccountDataRightsError(401, "account_deletion_status_invalid");
    }
    const lookup = {
      ...capability,
      statusCapabilitySha256: sha256Text(capability.capability)
    };
    const lease = this.newLease();
    const claimed = await this.repository.claimDeletionByCapability(
      lookup,
      this.leaseClaim(lease, true, now)
    );
    if (claimed.kind === "missing") {
      throw new AccountDataRightsError(401, "account_deletion_status_invalid");
    }
    if (claimed.kind === "completed") {
      return accountDeletionStatusProjection(
        claimed.tombstone,
        this.receiptFromTombstone(claimed.tombstone, capability.operationId)
      );
    }
    if (claimed.kind === "busy") {
      return accountDeletionStatusProjection(claimed.operation);
    }
    const scope = {
      tenantId: claimed.operation.tenantId,
      ownerId: claimed.operation.ownerId
    };
    const hashes = this.recoveryHashes(scope, claimed.operation);
    await this.continueDeletion(scope, hashes, lease, claimed.operation);
    const completed = await this.repository.statusByCapability(lookup);
    if (!completed) {
      throw new AccountDataRightsError(503, "account_deletion_retry_required");
    }
    return accountDeletionStatusProjection(
      completed,
      completed.phase === "completed"
        ? this.receiptFromTombstone(completed, capability.operationId)
        : undefined
    );
  }

  private assertConfirmationInput(input: AccountDeletionConfirmationInput): void {
    if (
      !CHALLENGE_PATTERN.test(input.challengeId)
      || !CSRF_PATTERN.test(input.confirmationToken)
      || input.confirmationPhrase !== ACCOUNT_DELETION_CONFIRMATION_PHRASE
      || !Number.isSafeInteger(input.expectedRevision)
      || input.expectedRevision < 1
      || !IDEMPOTENCY_PATTERN.test(input.idempotencyKey)
    ) {
      throw new AccountDataRightsError(400, "account_deletion_confirmation_invalid");
    }
  }

  private async requireFreshAuthority(
    principal: AuthenticatedPrincipal,
    now: Date
  ): Promise<AccountDeletionFreshAuthority> {
    if (principal.provider !== "oidc") {
      throw new AccountDataRightsError(
        401,
        "account_deletion_reauthentication_required"
      );
    }
    const authority = await this.freshAuthority.current(principal, now);
    if (!authority) {
      throw new AccountDataRightsError(
        401,
        "account_deletion_reauthentication_required"
      );
    }
    const authenticatedAt = new Date(authority.authenticatedAt);
    const expiresAt = new Date(authority.expiresAt);
    const nowMs = now.getTime();
    const uniqueBindings = new Set(authority.bindings.map((binding) =>
      `${binding.authorityGrantSha256}\0${binding.canonicalIdentitySha256}`
      + `\0${binding.issuerSha256}\0${binding.keyVersion}`
    ));
    if (
      !Number.isFinite(nowMs)
      || !Number.isFinite(authenticatedAt.getTime())
      || !Number.isFinite(expiresAt.getTime())
      || authority.bindings.length < 1
      || authority.bindings.length > 8
      || uniqueBindings.size !== authority.bindings.length
      || !authority.bindings.every(validAuthorityBinding)
      || authority.sessionSha256 !== sha256Text(principal.sessionId)
      || authenticatedAt.getTime() > nowMs + 30_000
      || authenticatedAt.getTime() < nowMs - ACCOUNT_DELETION_REAUTH_MAX_AGE_MS
      || expiresAt.getTime() < nowMs + ACCOUNT_DELETION_MIN_AUTHORITY_LIFETIME_MS
      || expiresAt.getTime() > authenticatedAt.getTime() + ACCOUNT_DELETION_REAUTH_MAX_AGE_MS
    ) {
      throw new AccountDataRightsError(
        401,
        "account_deletion_reauthentication_required"
      );
    }
    if (
      !Number.isInteger(authority.assuranceLevel)
      || authority.assuranceLevel < ACCOUNT_DELETION_REQUIRED_AAL
      || authority.assuranceLevel > 3
    ) {
      throw new AccountDataRightsError(
        403,
        "account_deletion_assurance_insufficient"
      );
    }
    return {
      bindings: authority.bindings.map((binding) => ({...binding})),
      sessionSha256: authority.sessionSha256,
      authenticatedAt,
      expiresAt,
      assuranceLevel: authority.assuranceLevel
    };
  }

  private async continueDeletion(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    lease: HeldDeletionLease,
    initial: AccountDeletionOperationRecord
  ): Promise<AccountDeletionReceipt> {
    let current = initial;
    try {
      for (;;) {
        current = await this.requireRenewedLease(scope, hashes, lease, current.operationId);
        // The worker fence is process-local. Reassert it before every durable
        // phase so a cold process can safely resume at draining or later.
        await this.withLeaseHeartbeat(scope, hashes, lease, current.operationId, async () => {
          await this.runtime.fenceApiTasks?.(scope);
          await this.runtime.fence(scope, hashes, current.operationId);
        });
        switch (current.phase) {
          case "fencing":
            current = await this.advanceOrReload(
              scope, hashes, lease, current, "draining"
            );
            break;
          case "draining":
            const apiDrain = await this.withLeaseHeartbeat(
              scope, hashes, lease, current.operationId, () =>
                this.runtime.drainApiTasks?.(scope, ACCOUNT_DELETION_DRAIN_TIMEOUT_MS)
                  ?? Promise.resolve({apiTasksCancelled: 0, apiTasksHandedOff: 0})
            );
            await this.withLeaseHeartbeat(
              scope, hashes, lease, current.operationId, () =>
                this.runtime.drain(
                  scope,
                  hashes,
                  current.operationId,
                  ACCOUNT_DELETION_DRAIN_TIMEOUT_MS
                )
            );
            void apiDrain;
            try {
              await this.repository.sealTombstone({
                ...scope,
                hashes,
                operationId: current.operationId,
                expectedRevision: current.revision,
                leaseTokenSha256: lease.tokenSha256,
                now: new Date()
              });
            } catch (error) {
              if (!(error instanceof AccountDataRightsError)
                || error.code !== "account_deletion_revision_conflict") throw error;
            }
            current = await this.reloadOperation(scope, hashes, lease, current.operationId);
            break;
          case "tombstoned":
            current = await this.advanceOrReload(
              scope, hashes, lease, current, "quarantining"
            );
            break;
          case "quarantining":
            await this.withLeaseHeartbeat(
              scope, hashes, lease, current.operationId, () =>
                this.runtime.quarantine(scope, hashes, current.operationId)
            );
            current = await this.advanceOrReload(
              scope, hashes, lease, current, "purging_worker_data"
            );
            break;
          case "purging_worker_data": {
            const purged = await this.withLeaseHeartbeat(
              scope, hashes, lease, current.operationId, () =>
                this.runtime.purgeQuarantine(scope, hashes, current.operationId)
            );
            current = await this.advanceOrReload(
              scope,
              hashes,
              lease,
              current,
              "committing_database",
              purged
            );
            break;
          }
          case "committing_database":
            return await this.repository.commitDeletion({
              ...scope,
              hashes,
              operationId: current.operationId,
              expectedRevision: current.revision,
              leaseTokenSha256: lease.tokenSha256,
              workerCounts: {
                workerFiles: current.counts.workerFiles,
                workerBytes: current.counts.workerBytes,
                workerRoots: current.counts.workerRoots
              },
              receiptId: opaqueId("adelr"),
              deletedAt: new Date()
            });
          case "prepared":
            throw new AccountDataRightsError(409, "account_deletion_revision_conflict");
        }
      }
    } catch (error) {
      const failureCode = `deletion_${current.phase}_retry`;
      const now = new Date();
      const exponent = Math.min(Math.max(current.recoveryAttempts - 1, 0), 8);
      const retryDelay = Math.min(
        ACCOUNT_DELETION_RECOVERY_RETRY_MAX_MS,
        ACCOUNT_DELETION_RECOVERY_RETRY_BASE_MS * (2 ** exponent)
      );
      await this.repository.markRetryableFailure(
        scope,
        hashes,
        current.operationId,
        lease.tokenSha256,
        failureCode,
        new Date(now.getTime() + retryDelay),
        now
      ).catch(() => undefined);
      throw new AccountDataRightsError(503, "account_deletion_retry_required");
    }
  }

  private async advanceOrReload(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    lease: HeldDeletionLease,
    current: AccountDeletionOperationRecord,
    nextPhase: Exclude<AccountDeletionOperationRecord["phase"], "prepared" | "completed">,
    counts?: Partial<AccountDeletionOperationRecord["counts"]>
  ): Promise<AccountDeletionOperationRecord> {
    try {
      return await this.repository.advanceDeletion({
        ...scope,
        hashes,
        operationId: current.operationId,
        expectedPhase: current.phase,
        expectedRevision: current.revision,
        nextPhase,
        counts,
        leaseTokenSha256: lease.tokenSha256,
        now: new Date()
      });
    } catch (error) {
      if (!(error instanceof AccountDataRightsError)
        || error.code !== "account_deletion_revision_conflict") throw error;
      return this.reloadOperation(scope, hashes, lease, current.operationId);
    }
  }

  private async reloadOperation(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    lease: HeldDeletionLease,
    operationId: string
  ): Promise<AccountDeletionOperationRecord> {
    const record = await this.repository.operationForLease(
      scope, hashes, operationId, lease.tokenSha256, new Date()
    );
    if (!record) {
      throw new AccountDataRightsError(409, "account_deletion_revision_conflict");
    }
    return record;
  }

  private newLease(): HeldDeletionLease {
    return {
      ownerSha256: this.recoveryOwnerSha256,
      tokenSha256: sha256Text(
        `account-deletion-lease\0${randomBytes(32).toString("base64url")}`
      )
    };
  }

  /**
   * A durable operation remains recoverable after its scope-HMAC key retires.
   * The stored digest is trusted only after selecting the operation through
   * raw tenant scope or the exact signed status capability; current hashes are
   * still included so sealing prevents the rotated scope from reopening.
   */
  private recoveryHashes(
    scope: AccessScope,
    operation: AccountDeletionOperationRecord
  ): AccountScopeHashes {
    const current = this.scopeHasher.hashes(scope);
    return current.candidates.includes(operation.scopeSha256)
      ? current
      : {
        active: current.active,
        candidates: [operation.scopeSha256, ...current.candidates]
      };
  }

  private leaseClaim(
    lease: HeldDeletionLease,
    ignoreRecoveryAfter: boolean,
    now = new Date()
  ): AccountDeletionLeaseClaim {
    return {
      leaseOwnerSha256: lease.ownerSha256,
      leaseTokenSha256: lease.tokenSha256,
      leaseDurationMs: ACCOUNT_DELETION_RECOVERY_LEASE_MS,
      now,
      ignoreRecoveryAfter
    };
  }

  private async requireRenewedLease(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    lease: HeldDeletionLease,
    operationId: string
  ): Promise<AccountDeletionOperationRecord> {
    const renewed = await this.repository.renewDeletionLease(
      scope,
      hashes,
      operationId,
      lease.tokenSha256,
      ACCOUNT_DELETION_RECOVERY_LEASE_MS,
      new Date()
    );
    if (!renewed) {
      throw new AccountDataRightsError(409, "account_deletion_revision_conflict");
    }
    return renewed;
  }

  private async withLeaseHeartbeat<T>(
    scope: AccessScope,
    hashes: AccountScopeHashes,
    lease: HeldDeletionLease,
    operationId: string,
    action: () => Promise<T>
  ): Promise<T> {
    let leaseLost = false;
    let heartbeat = Promise.resolve();
    const timer = setInterval(() => {
      heartbeat = heartbeat.then(async () => {
        if (leaseLost) return;
        const renewed = await this.repository.renewDeletionLease(
          scope,
          hashes,
          operationId,
          lease.tokenSha256,
          ACCOUNT_DELETION_RECOVERY_LEASE_MS,
          new Date()
        ).catch(() => undefined);
        if (!renewed) leaseLost = true;
      });
    }, Math.floor(ACCOUNT_DELETION_RECOVERY_LEASE_MS / 3));
    timer.unref();
    try {
      const result = await action();
      await heartbeat;
      if (leaseLost) {
        throw new AccountDataRightsError(409, "account_deletion_revision_conflict");
      }
      return result;
    } finally {
      clearInterval(timer);
      await heartbeat.catch(() => undefined);
    }
  }

  private receiptFromTombstone(
    tombstone: AccountDeletionTombstone,
    operationId: string
  ): AccountDeletionReceipt {
    if (!tombstone.deletedAt || !tombstone.receiptId || !tombstone.receiptSha256) {
      throw new AccountDataRightsError(503, "account_deletion_retry_required");
    }
    const receipt = accountDeletionReceipt({
      receiptId: tombstone.receiptId,
      scopeSha256: tombstone.receiptScopeSha256,
      operationId,
      deletedAt: tombstone.deletedAt,
      counts: tombstone.counts
    });
    if (receipt.receipt_sha256 !== tombstone.receiptSha256) {
      throw new AccountDataRightsError(503, "account_deletion_retry_required");
    }
    return receipt;
  }
}
