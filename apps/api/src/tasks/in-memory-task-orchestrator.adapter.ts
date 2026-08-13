import {ConflictException, Inject, Injectable} from "@nestjs/common";

import type {EventStreamPort} from "../events/event-stream.port";
import type {TelemetryPort} from "../observability/telemetry.port";
import {
  EVENT_STREAM,
  MODEL_PROVIDER,
  TASK_REPOSITORY,
  TELEMETRY
} from "../platform/tokens";
import type {ModelProviderPort} from "../providers/model-provider.port";
import type {AccessScope} from "../tenancy/access-scope";
import type {TaskOrchestratorPort} from "./task-orchestrator.port";
import type {TaskRepositoryPort} from "./task-repository.port";
import type {AgentTask, AgentTaskStatus, EnqueueTaskCommand} from "./task.types";

function scopeFor(task: Pick<AgentTask, "tenantId" | "ownerId">): AccessScope {
  return {tenantId: task.tenantId, ownerId: task.ownerId};
}

@Injectable()
export class InMemoryTaskOrchestratorAdapter implements TaskOrchestratorPort {
  private readonly abortControllers = new Map<string, AbortController>();

  constructor(
    @Inject(EVENT_STREAM) private readonly events: EventStreamPort,
    @Inject(MODEL_PROVIDER) private readonly modelProvider: ModelProviderPort,
    @Inject(TASK_REPOSITORY) private readonly tasks: TaskRepositoryPort,
    @Inject(TELEMETRY) private readonly telemetry: TelemetryPort
  ) {}

  async enqueue(command: EnqueueTaskCommand): Promise<AgentTask> {
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
    setImmediate(() => void this.dispatch(task));
    return task;
  }

  get(scope: AccessScope, taskId: string): Promise<AgentTask | undefined> {
    return this.tasks.findOwnedById(scope, taskId);
  }

  async cancel(scope: AccessScope, taskId: string): Promise<AgentTask | undefined> {
    const current = await this.tasks.findOwnedById(scope, taskId);
    if (!current) return undefined;
    const terminal: AgentTaskStatus[] = [
      "succeeded",
      "failed",
      "cancelled",
      "requires_configuration"
    ];
    if (terminal.includes(current.status)) return current;
    this.abortControllers.get(taskId)?.abort();
    const result = await this.tasks.cancelOwned(scope, taskId);
    const cancelled = result.kind === "updated" ? result.task : current;
    if (result.kind === "updated") {
      await this.events.append({
        ...scope,
        sessionId: current.sessionId,
        type: "status",
        payload: {kind: "task.cancelled", taskId, status: "cancelled"}
      });
      this.telemetry.event("task.cancelled");
    }
    return cancelled;
  }

  async fenceScope(scope: AccessScope): Promise<void> {
    for (const task of await this.tasks.cancelScope(scope)) {
      await this.events.append({
        ...scope,
        sessionId: task.sessionId,
        type: "status",
        payload: {kind: "task.cancelled", taskId: task.id, status: "cancelled"}
      });
    }
  }

  async drainScope(scope: AccessScope, _timeoutMs: number): Promise<{cancelled: number; handedOff: number}> {
    const cancelled = (await this.tasks.cancelScope(scope)).length;
    return {cancelled, handedOff: 0};
  }

  async probe(): Promise<"memory_local_only"> {
    return "memory_local_only";
  }

  private async dispatch(queued: AgentTask): Promise<void> {
    const scope = scopeFor(queued);
    const started = await this.tasks.transitionOwned(scope, queued.id, ["queued"], {
      status: "running"
    });
    if (started.kind !== "updated") return;
    const abortController = new AbortController();
    this.abortControllers.set(queued.id, abortController);
    await this.events.append({
      ...scope,
      sessionId: queued.sessionId,
      type: "status",
      payload: {kind: "task.running", taskId: queued.id, status: "running"}
    });
    this.telemetry.event("task.running");

    const providerStatus = await this.modelProvider.status();
    if (!providerStatus.configured) {
      const result = await this.tasks.transitionOwned(scope, queued.id, ["running"], {
        status: "requires_configuration",
        failureCode: "model_provider_not_configured"
      });
      if (result.kind === "updated") {
        await this.events.append({
          ...scope,
          sessionId: queued.sessionId,
          type: "state",
          payload: {
            kind: "task.requires_configuration",
            taskId: queued.id,
            status: "requires_configuration",
            provider: providerStatus.provider,
            model: providerStatus.model,
            reason: providerStatus.reason
          }
        });
        this.telemetry.event("task.requires_configuration", {
          provider: providerStatus.provider,
          reason: providerStatus.reason
        });
      }
      this.abortControllers.delete(queued.id);
      return;
    }

    let body = "";
    try {
      for await (const event of this.modelProvider.streamTurn({
        taskId: queued.id,
        sessionId: queued.sessionId,
        learnerMessage: queued.learnerMessage,
        signal: abortController.signal
      })) {
        if (event.type === "text_delta") body += event.text;
        if (event.type === "tool_use" || event.type === "tool_result") {
          await this.events.append({
            ...scope,
            sessionId: queued.sessionId,
            type: "tool",
            payload: {taskId: queued.id, ...event}
          });
        }
      }
      if (abortController.signal.aborted) return;
      const result = await this.tasks.transitionOwned(scope, queued.id, ["running"], {
        status: "succeeded"
      });
      if (result.kind !== "updated") return;
      if (body) {
        await this.events.append({
          ...scope,
          sessionId: queued.sessionId,
          type: "message",
          payload: {taskId: queued.id, role: "teacher", body}
        });
      }
      await this.events.append({
        ...scope,
        sessionId: queued.sessionId,
        type: "status",
        payload: {kind: "task.succeeded", taskId: queued.id, status: "succeeded"}
      });
      this.telemetry.event("task.succeeded", {provider: providerStatus.provider});
    } catch {
      if (abortController.signal.aborted) return;
      const failed = await this.tasks.transitionOwned(scope, queued.id, ["running"], {
        status: "failed",
        failureCode: "provider_execution_failed"
      });
      if (failed.kind === "updated") {
        await this.events.append({
          ...scope,
          sessionId: queued.sessionId,
          type: "error",
          payload: {kind: "task.failed", taskId: queued.id, code: "provider_execution_failed"}
        });
        this.telemetry.event("task.failed", {provider: providerStatus.provider});
      }
    } finally {
      this.abortControllers.delete(queued.id);
    }
  }
}
