import {randomUUID} from "node:crypto";

import {Inject, Injectable, Optional} from "@nestjs/common";

import type {TenantDatabasePort} from "../database/tenant-database.port";
import {TASK_SYSTEM_DATABASE, TENANT_DATABASE} from "../platform/tokens";
import type {TaskSystemDatabasePort, TaskSystemTransaction} from "./task-system-database.port";
import type {
  DurableTaskQueuePort,
  TaskLeaseClaimInput,
  TaskLeaseClaimResult,
  TaskLeaseRetryInput,
  TaskLeaseRetryResult,
  TaskLeaseSettlementInput
} from "./durable-task-queue.port";
import type {AccessScope} from "../tenancy/access-scope";
import type {
  CreateTaskResult,
  TaskRepositoryPort
} from "./task-repository.port";
import {TaskSessionNotActiveError} from "./task-repository.port";
import type {
  AgentTask,
  AgentTaskStatus,
  EnqueueTaskCommand,
  TaskTransitionResult
} from "./task.types";

export interface TaskRow {
  id: string;
  tenant_id: string;
  owner_id: string;
  session_id: string;
  learner_message: string;
  client_request_id: string | null;
  status: AgentTaskStatus;
  failure_code: string | null;
  created_at: Date | string;
  updated_at: Date | string;
  version: number;
  attempt_count?: number | string;
  available_at?: Date | string;
  lease_owner_sha256?: string | null;
  lease_token_sha256?: string | null;
  lease_expires_at?: Date | string | null;
}

const TASK_COLUMNS = `
  id, tenant_id, owner_id, session_id, learner_message, client_request_id,
  status, failure_code, created_at, updated_at, version
  , attempt_count, available_at, lease_owner_sha256, lease_token_sha256,
  lease_expires_at
`;
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const HASH_PATTERN = /^[0-9a-f]{64}$/;
const FAILURE_PATTERN = /^[a-z][a-z0-9_]{2,63}$/;

function timestamp(value: Date | string): string {
  return value instanceof Date ? value.toISOString() : new Date(value).toISOString();
}

function mapTask(row: TaskRow): AgentTask {
  return {
    id: row.id,
    tenantId: row.tenant_id,
    ownerId: row.owner_id,
    sessionId: row.session_id,
    learnerMessage: row.learner_message,
    ...(row.client_request_id ? {clientRequestId: row.client_request_id} : {}),
    status: row.status,
    createdAt: timestamp(row.created_at),
    updatedAt: timestamp(row.updated_at),
    ...(row.failure_code ? {failureCode: row.failure_code} : {}),
    attemptCount: Number(row.attempt_count ?? 0),
    availableAt: timestamp(row.available_at ?? row.created_at),
    version: row.version
  };
}

@Injectable()
export class PostgresTaskRepository implements TaskRepositoryPort {
  constructor(
    @Inject(TENANT_DATABASE) private readonly database: TenantDatabasePort,
    @Optional() @Inject(TASK_SYSTEM_DATABASE) private readonly systemDatabase?: TaskSystemDatabasePort
  ) {}

  create(command: EnqueueTaskCommand): Promise<CreateTaskResult> {
    const scope = {tenantId: command.tenantId, ownerId: command.ownerId};
    return this.database.withTenant(scope, async (transaction) => {
      if (command.clientRequestId) {
        const existing = await transaction.query<TaskRow>(
          `SELECT ${TASK_COLUMNS}
             FROM public.teachlab_agent_tasks
            WHERE tenant_id = $1 AND owner_id = $2 AND session_id = $3
              AND client_request_id = $4`,
          [
            command.tenantId,
            command.ownerId,
            command.sessionId,
            command.clientRequestId
          ]
        );
        if (existing.rows[0]) return {created: false, task: mapTask(existing.rows[0])};
      }
      const session = await transaction.query<{status: string}>(
        `SELECT status
           FROM public.teachlab_teaching_sessions
          WHERE tenant_id = $1 AND owner_id = $2 AND id = $3
          FOR UPDATE`,
        [command.tenantId, command.ownerId, command.sessionId]
      );
      if (session.rows[0]?.status !== "active") throw new TaskSessionNotActiveError();
      const inserted = await transaction.query<TaskRow>(
        `INSERT INTO public.teachlab_agent_tasks (
           id, tenant_id, owner_id, session_id, learner_message, client_request_id
         ) VALUES ($1, $2, $3, $4, $5, $6)
         ON CONFLICT (tenant_id, owner_id, session_id, client_request_id)
           WHERE client_request_id IS NOT NULL
         DO NOTHING
         RETURNING ${TASK_COLUMNS}`,
        [
          randomUUID(),
          command.tenantId,
          command.ownerId,
          command.sessionId,
          command.learnerMessage,
          command.clientRequestId ?? null
        ]
      );
      const created = inserted.rows[0];
      if (created) return {created: true, task: mapTask(created)};
      if (command.clientRequestId) {
        const existing = await transaction.query<TaskRow>(
          `SELECT ${TASK_COLUMNS}
             FROM public.teachlab_agent_tasks
            WHERE tenant_id = $1 AND owner_id = $2 AND session_id = $3
              AND client_request_id = $4`,
          [
            command.tenantId,
            command.ownerId,
            command.sessionId,
            command.clientRequestId
          ]
        );
        if (existing.rows[0]) return {created: false, task: mapTask(existing.rows[0])};
      }
      throw new TaskSessionNotActiveError();
    });
  }

  findOwnedById(scope: AccessScope, taskId: string): Promise<AgentTask | undefined> {
    if (!UUID_PATTERN.test(taskId)) return Promise.resolve(undefined);
    return this.database.withTenant(scope, async (transaction) => {
      const result = await transaction.query<TaskRow>(
        `SELECT ${TASK_COLUMNS}
           FROM public.teachlab_agent_tasks
          WHERE tenant_id = $1 AND owner_id = $2 AND id = $3`,
        [scope.tenantId, scope.ownerId, taskId]
      );
      return result.rows[0] ? mapTask(result.rows[0]) : undefined;
    });
  }

  transitionOwned(
    scope: AccessScope,
    taskId: string,
    expectedStatuses: AgentTaskStatus[],
    patch: Partial<Pick<AgentTask, "status" | "failureCode">>
  ): Promise<TaskTransitionResult> {
    if (!UUID_PATTERN.test(taskId)) return Promise.resolve({kind: "not_found"});
    return this.database.withTenant(scope, async (transaction) => {
      const updated = await transaction.query<TaskRow>(
        `UPDATE public.teachlab_agent_tasks
            SET status = $4,
                failure_code = $5,
                lease_owner_sha256 = NULL,
                lease_token_sha256 = NULL,
                lease_expires_at = NULL,
                available_at = statement_timestamp(),
                updated_at = statement_timestamp(),
                version = version + 1
          WHERE tenant_id = $1 AND owner_id = $2 AND id = $3
            AND status = ANY($6::text[])
          RETURNING ${TASK_COLUMNS}`,
        [
          scope.tenantId,
          scope.ownerId,
          taskId,
          patch.status,
          patch.failureCode ?? null,
          expectedStatuses
        ]
      );
      if (updated.rows[0]) return {kind: "updated", task: mapTask(updated.rows[0])};
      const current = await transaction.query<TaskRow>(
        `SELECT ${TASK_COLUMNS}
           FROM public.teachlab_agent_tasks
          WHERE tenant_id = $1 AND owner_id = $2 AND id = $3`,
        [scope.tenantId, scope.ownerId, taskId]
      );
      return current.rows[0]
        ? {kind: "state_conflict", task: mapTask(current.rows[0])}
        : {kind: "not_found"};
    });
  }

  cancelOwned(scope: AccessScope, taskId: string): Promise<TaskTransitionResult> {
    return this.transitionOwned(scope, taskId, ["queued", "running"], {
      status: "cancelled",
      failureCode: undefined
    });
  }

  cancelScope(
    scope: AccessScope,
    options: {allowAccountDeleting?: boolean} = {}
  ): Promise<AgentTask[]> {
    return this.database.withTenant(scope, async (transaction) => {
      const result = await transaction.query<TaskRow>(
        `UPDATE public.teachlab_agent_tasks
            SET status = 'cancelled', failure_code = NULL,
                lease_owner_sha256 = NULL, lease_token_sha256 = NULL,
                lease_expires_at = NULL, available_at = statement_timestamp(),
                updated_at = statement_timestamp(), version = version + 1
          WHERE tenant_id = $1 AND owner_id = $2
            AND status IN ('queued', 'running')
          RETURNING ${TASK_COLUMNS}`,
        [scope.tenantId, scope.ownerId]
      );
      return result.rows.map(mapTask);
    }, options);
  }

  async claimNext(input: TaskLeaseClaimInput): Promise<TaskLeaseClaimResult> {
    assertLeaseInput(input);
    if (!this.systemDatabase) throw new Error("Durable task dispatcher is not configured");
    return this.systemDatabase.withTaskDispatcher(async (transaction) => {
      const selected = await transaction.query<TaskRow>(
        `SELECT ${TASK_COLUMNS}
           FROM public.teachlab_agent_tasks AS t
          WHERE (
            t.status = 'queued'
            OR (
              t.status = 'running'
              AND (t.lease_expires_at IS NULL OR t.lease_expires_at <= statement_timestamp())
            )
          )
            AND t.available_at <= statement_timestamp()
            AND (t.lease_expires_at IS NULL OR t.lease_expires_at <= statement_timestamp())
            AND NOT EXISTS (
              SELECT 1
                FROM public.teachlab_account_deletion_operations AS d
               WHERE d.tenant_id = t.tenant_id
                 AND d.owner_id = t.owner_id
                 AND d.phase <> 'prepared'
            )
          ORDER BY t.available_at ASC, t.created_at ASC, t.id ASC
          LIMIT 1`
      );
      const candidate = selected.rows[0];
      if (!candidate) return {kind: "none"};
      await lockScope(transaction, candidate.tenant_id, candidate.owner_id);
      const current = await transaction.query<TaskRow>(
        `SELECT ${TASK_COLUMNS}
           FROM public.teachlab_agent_tasks AS t
          WHERE t.tenant_id = $1 AND t.owner_id = $2 AND t.id = $3
            AND (
              t.status = 'queued'
              OR (
                t.status = 'running'
                AND (t.lease_expires_at IS NULL OR t.lease_expires_at <= statement_timestamp())
              )
            )
            AND t.available_at <= statement_timestamp()
            AND (t.lease_expires_at IS NULL OR t.lease_expires_at <= statement_timestamp())
            AND NOT EXISTS (
              SELECT 1
                FROM public.teachlab_account_deletion_operations AS d
               WHERE d.tenant_id = t.tenant_id
                 AND d.owner_id = t.owner_id
                 AND d.phase <> 'prepared'
            )
          FOR UPDATE`,
        [candidate.tenant_id, candidate.owner_id, candidate.id]
      );
      if (!current.rows[0]) return {kind: "none"};
      // The attempt counter is the number of executions already admitted.
      // Once it reaches the configured ceiling, an expired lease must be
      // terminalized without incrementing it again. Incrementing first would
      // make a max-attempt task briefly become max+1 and violate the database
      // CHECK constraint (and, more importantly, the bounded-retry contract).
      const currentAttemptCount = Number(current.rows[0].attempt_count ?? 0);
      if (!Number.isSafeInteger(currentAttemptCount) || currentAttemptCount < 0) {
        throw new Error("Invalid durable task attempt count");
      }
      if (currentAttemptCount >= input.maximumAttempts) {
        const terminal = await transaction.query<TaskRow>(
          `UPDATE public.teachlab_agent_tasks
              SET status = 'failed', failure_code = 'task_attempt_limit_exceeded',
                  lease_owner_sha256 = NULL, lease_token_sha256 = NULL,
                  lease_expires_at = NULL, available_at = statement_timestamp(),
                  updated_at = statement_timestamp(), version = version + 1
            WHERE tenant_id = $1 AND owner_id = $2 AND id = $3
              AND (
                status = 'queued'
                OR (
                  status = 'running'
                  AND (lease_expires_at IS NULL OR lease_expires_at <= statement_timestamp())
                )
              )
              AND available_at <= statement_timestamp()
              AND (lease_expires_at IS NULL OR lease_expires_at <= statement_timestamp())
              AND NOT EXISTS (
                SELECT 1
                  FROM public.teachlab_account_deletion_operations AS d
                 WHERE d.tenant_id = $1
                   AND d.owner_id = $2
                   AND d.phase <> 'prepared'
              )
          RETURNING ${TASK_COLUMNS}`,
          [current.rows[0].tenant_id, current.rows[0].owner_id, current.rows[0].id]
        );
        const failed = terminal.rows[0];
        if (!failed) return {kind: "none"};
        await appendSystemEvents(transaction, {
          tenantId: failed.tenant_id,
          ownerId: failed.owner_id,
          taskId: failed.id,
          events: [{
            type: "error",
            payload: {
              kind: "task.failed",
              taskId: failed.id,
              status: "failed",
              code: "task_attempt_limit_exceeded"
            }
          }]
        }, failed.session_id);
        return {kind: "terminalized", task: mapTask(failed), reason: "attempts_exhausted"};
      }
      const result = await transaction.query<TaskRow>(
        `UPDATE public.teachlab_agent_tasks AS t
            SET status = 'running',
                attempt_count = t.attempt_count + 1,
                lease_owner_sha256 = $4,
                lease_token_sha256 = $5,
                lease_expires_at = statement_timestamp() + ($6::bigint * interval '1 millisecond'),
                updated_at = statement_timestamp(),
                version = t.version + 1
          WHERE t.tenant_id = $1 AND t.owner_id = $2 AND t.id = $3
            AND (
              t.status = 'queued'
              OR (
                t.status = 'running'
                AND (t.lease_expires_at IS NULL OR t.lease_expires_at <= statement_timestamp())
              )
            )
            AND t.available_at <= statement_timestamp()
            AND (t.lease_expires_at IS NULL OR t.lease_expires_at <= statement_timestamp())
            AND NOT EXISTS (
              SELECT 1
                FROM public.teachlab_account_deletion_operations AS d
               WHERE d.tenant_id = t.tenant_id
                 AND d.owner_id = t.owner_id
                 AND d.phase <> 'prepared'
            )
        RETURNING ${TASK_COLUMNS}`,
        [candidate.tenant_id, candidate.owner_id, candidate.id,
          input.leaseOwnerSha256, input.leaseTokenSha256, input.leaseDurationMs]
      );
      const row = result.rows[0];
      if (!row) return {kind: "none"};
      const task = mapTask(row);
      return {kind: "claimed", task, leaseTokenSha256: input.leaseTokenSha256};
    });
  }

  renewLease(
    scope: AccessScope,
    taskId: string,
    leaseTokenSha256: string,
    leaseDurationMs: number
  ): Promise<AgentTask | undefined> {
    if (!UUID_PATTERN.test(taskId) || !HASH_PATTERN.test(leaseTokenSha256)) {
      return Promise.resolve(undefined);
    }
    if (!Number.isInteger(leaseDurationMs) || leaseDurationMs < 1_000 || leaseDurationMs > 5 * 60_000) {
      throw new Error("Invalid task lease duration");
    }
    if (!this.systemDatabase) throw new Error("Durable task dispatcher is not configured");
    return this.systemDatabase.withTaskDispatcher(async (transaction) => {
      await lockScope(transaction, scope.tenantId, scope.ownerId);
      const result = await transaction.query<TaskRow>(
        `UPDATE public.teachlab_agent_tasks
            SET lease_expires_at = statement_timestamp() + ($4::bigint * interval '1 millisecond'),
                updated_at = statement_timestamp(), version = version + 1
          WHERE tenant_id = $1 AND owner_id = $2 AND id = $3
            AND status = 'running' AND lease_token_sha256 = $5
            AND lease_expires_at > statement_timestamp()
            AND NOT EXISTS (
              SELECT 1
                FROM public.teachlab_account_deletion_operations AS d
               WHERE d.tenant_id = $1
                 AND d.owner_id = $2
                 AND d.phase <> 'prepared'
            )
        RETURNING ${TASK_COLUMNS}`,
        [scope.tenantId, scope.ownerId, taskId, leaseDurationMs, leaseTokenSha256]
      );
      return result.rows[0] ? mapTask(result.rows[0]) : undefined;
    });
  }

  settleLease(input: TaskLeaseSettlementInput): Promise<AgentTask | undefined> {
    validateSettlement(input);
    if (!this.systemDatabase) throw new Error("Durable task dispatcher is not configured");
    return this.systemDatabase.withTaskDispatcher(async (transaction) => {
      await lockScope(transaction, input.tenantId, input.ownerId);
      const result = await transaction.query<TaskRow>(
        `UPDATE public.teachlab_agent_tasks
            SET status = $4, failure_code = $5,
                lease_owner_sha256 = NULL, lease_token_sha256 = NULL,
                lease_expires_at = NULL, available_at = statement_timestamp(),
                updated_at = statement_timestamp(), version = version + 1
          WHERE tenant_id = $1 AND owner_id = $2 AND id = $3
            AND status = 'running' AND lease_token_sha256 = $6
            AND lease_expires_at > statement_timestamp()
            AND NOT EXISTS (
              SELECT 1
                FROM public.teachlab_account_deletion_operations AS d
               WHERE d.tenant_id = $1
                 AND d.owner_id = $2
                 AND d.phase <> 'prepared'
            )
        RETURNING ${TASK_COLUMNS}`,
        [
          input.tenantId, input.ownerId, input.taskId, input.status,
          input.failureCode ?? null, input.leaseTokenSha256
        ]
      );
      const row = result.rows[0];
      if (!row) return undefined;
      await appendSystemEvents(transaction, input, row.session_id);
      return mapTask(row);
    });
  }

  retryLease(input: TaskLeaseRetryInput): Promise<TaskLeaseRetryResult> {
    validateRetry(input);
    if (!this.systemDatabase) throw new Error("Durable task dispatcher is not configured");
    return this.systemDatabase.withTaskDispatcher(async (transaction) => {
      await lockScope(transaction, input.tenantId, input.ownerId);
      const result = await transaction.query<TaskRow>(
        `UPDATE public.teachlab_agent_tasks
            SET status = CASE WHEN attempt_count >= $8 THEN 'failed' ELSE 'queued' END,
                failure_code = $4,
                lease_owner_sha256 = NULL, lease_token_sha256 = NULL,
                lease_expires_at = NULL,
                available_at = CASE WHEN attempt_count >= $8
                  THEN statement_timestamp()
                  ELSE statement_timestamp() + (
                    LEAST(
                      $7::double precision,
                      ($6::double precision * power(
                        2::double precision,
                        LEAST(GREATEST(attempt_count - 1, 0), 8)
                      ))::bigint
                    ) * interval '1 millisecond'
                  )
                END,
                updated_at = statement_timestamp(), version = version + 1
          WHERE tenant_id = $1 AND owner_id = $2 AND id = $3
            AND status = 'running' AND lease_token_sha256 = $5
            AND lease_expires_at > statement_timestamp()
            AND NOT EXISTS (
              SELECT 1
                FROM public.teachlab_account_deletion_operations AS d
               WHERE d.tenant_id = $1
                 AND d.owner_id = $2
                 AND d.phase <> 'prepared'
            )
        RETURNING ${TASK_COLUMNS}`,
        [
          input.tenantId, input.ownerId, input.taskId, input.failureCode,
          input.leaseTokenSha256, input.retryBaseMs, input.retryMaximumMs,
          input.maximumAttempts
        ]
      );
      const row = result.rows[0];
      if (!row) return {kind: "lease_lost"};
      await appendSystemEvents(transaction, input, row.session_id);
      const task = mapTask(row);
      return task.status === "failed"
        ? {kind: "failed", task}
        : {kind: "retry_scheduled", task};
    });
  }

  async probe(): Promise<void> {
    if (!this.systemDatabase) throw new Error("Durable task dispatcher is not configured");
    await this.systemDatabase.withTaskDispatcher(async (transaction) => {
      const result = await transaction.query<{ready: number}>("SELECT 1::int AS ready");
      if (result.rowCount !== 1 || result.rows[0]?.ready !== 1) {
        throw new Error("Durable task queue probe failed");
      }
    });
  }
}

function assertLeaseInput(input: TaskLeaseClaimInput): void {
  if (!HASH_PATTERN.test(input.leaseOwnerSha256) || !HASH_PATTERN.test(input.leaseTokenSha256)) {
    throw new Error("Invalid task lease hash");
  }
  if (
    !Number.isInteger(input.leaseDurationMs) || input.leaseDurationMs < 1_000
    || input.leaseDurationMs > 5 * 60_000
    || !Number.isInteger(input.maximumAttempts) || input.maximumAttempts < 1 || input.maximumAttempts > 20
  ) throw new Error("Invalid task lease policy");
}

function validateSettlement(input: TaskLeaseSettlementInput): void {
  if (!UUID_PATTERN.test(input.taskId) || !HASH_PATTERN.test(input.leaseTokenSha256)) {
    throw new Error("Invalid task settlement lease");
  }
  if (input.failureCode && !FAILURE_PATTERN.test(input.failureCode)) {
    throw new Error("Invalid task failure code");
  }
}

function validateRetry(input: TaskLeaseRetryInput): void {
  if (!UUID_PATTERN.test(input.taskId) || !HASH_PATTERN.test(input.leaseTokenSha256)) {
    throw new Error("Invalid task retry lease");
  }
  if (
    !Number.isInteger(input.maximumAttempts) || input.maximumAttempts < 1 || input.maximumAttempts > 20
    || !Number.isInteger(input.retryBaseMs) || input.retryBaseMs < 100 || input.retryBaseMs > 60_000
    || !Number.isInteger(input.retryMaximumMs) || input.retryMaximumMs < input.retryBaseMs || input.retryMaximumMs > 10 * 60_000
    || !FAILURE_PATTERN.test(input.failureCode)
  ) throw new Error("Invalid task retry policy");
}

async function appendSystemEvents(
  transaction: TaskSystemTransaction,
  input: {tenantId: string; ownerId: string; taskId: string; events: readonly {type: string; payload: Record<string, unknown>}[]},
  sessionId: string
): Promise<void> {
  for (const event of input.events) {
    await transaction.query(
      `INSERT INTO public.teachlab_task_events (
         id, tenant_id, owner_id, session_id, event_type, payload
       ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)`,
      [randomUUID(), input.tenantId, input.ownerId, sessionId, event.type, JSON.stringify(event.payload)]
    );
  }
}

async function lockScope(
  transaction: TaskSystemTransaction,
  tenantId: string,
  ownerId: string
): Promise<void> {
  await transaction.query(
    "SELECT pg_advisory_xact_lock(hashtextextended($1, 7640891576956012809))",
    [JSON.stringify([tenantId, ownerId])]
  );
}
