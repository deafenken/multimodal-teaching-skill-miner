import {createHash, randomBytes} from "node:crypto";

import {ConflictException, Inject, Injectable, OnApplicationBootstrap, OnModuleDestroy, Optional} from "@nestjs/common";

import type {EventStreamPort} from "../events/event-stream.port";
import type {TelemetryPort} from "../observability/telemetry.port";
import {EVENT_STREAM, MODEL_PROVIDER, TELEMETRY, TASK_REPOSITORY, DURABLE_TASK_QUEUE} from "../platform/tokens";
import type {ModelProviderPort} from "../providers/model-provider.port";
import type {AccessScope} from "../tenancy/access-scope";
import type {TaskRepositoryPort} from "./task-repository.port";
import type {DurableTaskQueuePort} from "./durable-task-queue.port";
import type {TaskOrchestratorPort} from "./task-orchestrator.port";
import type {AgentTask, AgentTaskStatus, EnqueueTaskCommand} from "./task.types";
import {AppConfigService} from "../config/app-config.service";

function scopeFor(task: Pick<AgentTask, "tenantId" | "ownerId">): AccessScope {
  return {tenantId: task.tenantId, ownerId: task.ownerId};
}

function digest(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

@Injectable()
export class PostgresTaskOrchestratorAdapter
  implements TaskOrchestratorPort, OnApplicationBootstrap, OnModuleDestroy {
  private readonly ownerSha256 = digest(
    `teachlab-task-dispatcher\0${process.pid}\0${randomBytes(32).toString("base64url")}`
  );
  private readonly abortControllers = new Map<string, AbortController>();
  private readonly fenced = new Set<string>();
  private pollTimer?: NodeJS.Timeout;
  private polling?: Promise<void>;
  private stopping = false;

  constructor(
    @Inject(EVENT_STREAM) private readonly events: EventStreamPort,
    @Inject(MODEL_PROVIDER) private readonly modelProvider: ModelProviderPort,
    @Inject(TASK_REPOSITORY) private readonly tasks: TaskRepositoryPort,
    @Inject(DURABLE_TASK_QUEUE) private readonly queue: DurableTaskQueuePort,
    @Inject(TELEMETRY) private readonly telemetry: TelemetryPort,
    @Inject(AppConfigService) private readonly config: AppConfigService
  ) {}

  onApplicationBootstrap(): void {
    if (this.config.dataBackend !== "postgres") return;
    void this.pump();
    this.pollTimer = setInterval(() => void this.pump(), this.config.taskDispatchPollMs);
    this.pollTimer.unref();
  }

  async onModuleDestroy(): Promise<void> {
    this.stopping = true;
    if (this.pollTimer) clearInterval(this.pollTimer);
    for (const controller of this.abortControllers.values()) controller.abort();
    await this.polling?.catch(() => undefined);
  }

  async enqueue(command: EnqueueTaskCommand): Promise<AgentTask> {
    if (this.fenced.has(`${command.tenantId}\0${command.ownerId}`)) {
      throw new ConflictException("Account scope is fenced");
    }
    const created = await this.tasks.create(command);
    if (!created.created) {
      if (created.task.learnerMessage !== command.learnerMessage) {
        throw new ConflictException({
          code: "idempotency_key_reused",
          message: "clientRequestId was already used with a different message"
        });
      }
      return created.task;
    }
    const task = created.task;
    await this.events.append({
      tenantId: task.tenantId,
      ownerId: task.ownerId,
      sessionId: task.sessionId,
      type: "status",
      payload: {kind: "task.queued", taskId: task.id, status: task.status}
    });
    this.telemetry.event("task.queued");
    void this.pump();
    return task;
  }

  get(scope: AccessScope, taskId: string): Promise<AgentTask | undefined> {
    return this.tasks.findOwnedById(scope, taskId);
  }

  async cancel(scope: AccessScope, taskId: string): Promise<AgentTask | undefined> {
    const current = await this.tasks.findOwnedById(scope, taskId);
    if (!current) return undefined;
    const terminal: AgentTaskStatus[] = [
      "succeeded", "failed", "cancelled", "requires_configuration"
    ];
    if (terminal.includes(current.status)) return current;
    this.abortControllers.get(taskId)?.abort();
    const result = await this.tasks.cancelOwned(scope, taskId);
    if (result.kind === "updated") {
      await this.events.append({
        ...scope,
        sessionId: current.sessionId,
        type: "status",
        payload: {kind: "task.cancelled", taskId, status: "cancelled"}
      });
      this.telemetry.event("task.cancelled");
      return result.task;
    }
    return result.kind === "state_conflict" ? result.task : current;
  }

  async fenceScope(scope: AccessScope): Promise<void> {
    const key = `${scope.tenantId}\0${scope.ownerId}`;
    this.fenced.add(key);
    for (const [taskId, controller] of this.abortControllers) {
      const current = await this.tasks.findOwnedById(scope, taskId);
      if (current) controller.abort();
    }
    await this.tasks.cancelScope(scope, {allowAccountDeleting: true});
  }

  async drainScope(scope: AccessScope, timeoutMs: number): Promise<{cancelled: number; handedOff: number}> {
    await this.fenceScope(scope);
    const deadline = Date.now() + Math.max(1, timeoutMs);
    while (Date.now() < deadline) {
      const active = [...this.abortControllers.keys()];
      let found = false;
      for (const taskId of active) {
        const current = await this.tasks.findOwnedById(scope, taskId);
        if (current?.status === "running") found = true;
      }
      if (!found) break;
      await new Promise((resolve) => setTimeout(resolve, Math.min(25, Math.max(1, deadline - Date.now()))));
    }
    const cancelled = (await this.tasks.cancelScope(scope, {allowAccountDeleting: true})).length;
    return {cancelled, handedOff: 0};
  }

  async probe(): Promise<"postgres_durable_lease"> {
    await this.queue.probe();
    return "postgres_durable_lease";
  }

  private async pump(): Promise<void> {
    if (this.stopping || this.polling) return this.polling;
    const run = this.runPump().finally(() => {
      if (this.polling === run) this.polling = undefined;
    });
    this.polling = run;
    return run;
  }

  private async runPump(): Promise<void> {
    for (let count = 0; count < this.config.taskDispatchBatchSize && !this.stopping; count += 1) {
      const token = digest(`${this.ownerSha256}\0${randomBytes(32).toString("base64url")}`);
      const claimed = await this.queue.claimNext({
        leaseOwnerSha256: this.ownerSha256,
        leaseTokenSha256: token,
        leaseDurationMs: this.config.taskLeaseDurationMs,
        maximumAttempts: this.config.taskMaximumAttempts
      });
      if (claimed.kind === "none") return;
      if (claimed.kind === "terminalized") {
        this.telemetry.event("task.failed", {reason: claimed.reason});
        continue;
      }
      void this.execute(claimed.task, token);
    }
  }

  private async execute(task: AgentTask, leaseTokenSha256: string): Promise<void> {
    const scope = scopeFor(task);
    const controller = new AbortController();
    this.abortControllers.set(task.id, controller);
    const renewTimer = setInterval(() => {
      void this.queue.renewLease(
        scope, task.id, leaseTokenSha256, this.config.taskLeaseDurationMs
      ).then((renewed) => {
        if (!renewed) controller.abort();
      }).catch(() => controller.abort());
    }, Math.max(500, Math.floor(this.config.taskLeaseDurationMs / 2)));
    renewTimer.unref();
    try {
      await this.events.append({
        ...scope,
        sessionId: task.sessionId,
        type: "status",
        payload: {kind: "task.running", taskId: task.id, status: "running"}
      });
      this.telemetry.event("task.running");
      const providerStatus = await this.modelProvider.status();
      if (!providerStatus.configured) {
        const settled = await this.queue.settleLease({
          ...scope,
          taskId: task.id,
          leaseTokenSha256,
          status: "requires_configuration",
          failureCode: "model_provider_not_configured",
          events: [{
            type: "state",
            payload: {
              kind: "task.requires_configuration", taskId: task.id,
              status: "requires_configuration", provider: providerStatus.provider,
              model: providerStatus.model, reason: providerStatus.reason
            }
          }]
        });
        if (settled) this.telemetry.event("task.requires_configuration", {provider: providerStatus.provider});
        return;
      }
      let body = "";
      for await (const event of this.modelProvider.streamTurn({
        taskId: task.id,
        sessionId: task.sessionId,
        learnerMessage: task.learnerMessage,
        signal: controller.signal
      })) {
        if (controller.signal.aborted) return;
        if (event.type === "text_delta") body += event.text;
        if (event.type === "tool_use" || event.type === "tool_result") {
          await this.events.append({
            ...scope,
            sessionId: task.sessionId,
            type: "tool",
            payload: {taskId: task.id, ...event}
          });
        }
      }
      if (controller.signal.aborted) return;
      const events = body
        ? [{type: "message" as const, payload: {taskId: task.id, role: "teacher", body}},
           {type: "status" as const, payload: {kind: "task.succeeded", taskId: task.id, status: "succeeded" as const}}]
        : [{type: "status" as const, payload: {kind: "task.succeeded", taskId: task.id, status: "succeeded" as const}}];
      const settled = await this.queue.settleLease({
        ...scope, taskId: task.id, leaseTokenSha256, status: "succeeded", events
      });
      if (settled) this.telemetry.event("task.succeeded", {provider: providerStatus.provider});
    } catch {
      if (controller.signal.aborted) return;
      const failed = await this.queue.retryLease({
        ...scope,
        taskId: task.id,
        leaseTokenSha256,
        failureCode: "provider_execution_failed",
        maximumAttempts: this.config.taskMaximumAttempts,
        retryBaseMs: this.config.taskRetryBaseMs,
        retryMaximumMs: this.config.taskRetryMaximumMs,
        events: [{type: "error", payload: {kind: "task.failed", taskId: task.id, code: "provider_execution_failed"}}]
      });
      if (failed.kind === "failed") this.telemetry.event("task.failed");
    } finally {
      clearInterval(renewTimer);
      this.abortControllers.delete(task.id);
    }
  }
}
