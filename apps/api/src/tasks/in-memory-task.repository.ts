import {randomUUID} from "node:crypto";

import {Injectable} from "@nestjs/common";

import type {AccessScope} from "../tenancy/access-scope";
import type {CreateTaskResult, TaskRepositoryPort} from "./task-repository.port";
import type {
  AgentTask,
  AgentTaskStatus,
  EnqueueTaskCommand,
  TaskTransitionResult
} from "./task.types";

@Injectable()
export class InMemoryTaskRepository implements TaskRepositoryPort {
  private readonly tasks = new Map<string, AgentTask>();
  private readonly idempotencyIndex = new Map<string, string>();

  async create(command: EnqueueTaskCommand): Promise<CreateTaskResult> {
    const key = command.clientRequestId
      ? `${command.tenantId}\u0000${command.ownerId}\u0000${command.sessionId}\u0000${command.clientRequestId}`
      : undefined;
    const existingId = key ? this.idempotencyIndex.get(key) : undefined;
    const existing = existingId ? this.tasks.get(existingId) : undefined;
    if (existing) return {created: false, task: structuredClone(existing)};
    const now = new Date().toISOString();
    const task: AgentTask = {
      id: randomUUID(),
      tenantId: command.tenantId,
      ownerId: command.ownerId,
      sessionId: command.sessionId,
      learnerMessage: command.learnerMessage,
      clientRequestId: command.clientRequestId,
      status: "queued",
      createdAt: now,
      updatedAt: now,
      attemptCount: 0,
      availableAt: now,
      version: 1
    };
    this.tasks.set(task.id, task);
    if (key) this.idempotencyIndex.set(key, task.id);
    return {created: true, task: structuredClone(task)};
  }

  async findOwnedById(
    scope: AccessScope,
    taskId: string
  ): Promise<AgentTask | undefined> {
    const task = this.tasks.get(taskId);
    return task && task.tenantId === scope.tenantId && task.ownerId === scope.ownerId
      ? structuredClone(task)
      : undefined;
  }

  async transitionOwned(
    scope: AccessScope,
    taskId: string,
    expectedStatuses: AgentTaskStatus[],
    patch: Partial<Pick<AgentTask, "status" | "failureCode">>
  ): Promise<TaskTransitionResult> {
    const task = this.tasks.get(taskId);
    if (!task || task.tenantId !== scope.tenantId || task.ownerId !== scope.ownerId) {
      return {kind: "not_found"};
    }
    if (!expectedStatuses.includes(task.status)) {
      return {kind: "state_conflict", task: structuredClone(task)};
    }
    const updated: AgentTask = {
      ...task,
      ...patch,
      updatedAt: new Date().toISOString(),
      version: task.version + 1
    };
    if (patch.failureCode === undefined) delete updated.failureCode;
    this.tasks.set(taskId, updated);
    return {kind: "updated", task: structuredClone(updated)};
  }

  async cancelOwned(scope: AccessScope, taskId: string): Promise<TaskTransitionResult> {
    return this.transitionOwned(scope, taskId, ["queued", "running"], {
      status: "cancelled"
    });
  }

  async cancelScope(scope: AccessScope): Promise<AgentTask[]> {
    const cancelled: AgentTask[] = [];
    for (const task of this.tasks.values()) {
      if (task.tenantId !== scope.tenantId || task.ownerId !== scope.ownerId) continue;
      if (task.status !== "queued" && task.status !== "running") continue;
      const result = await this.cancelOwned(scope, task.id);
      if (result.kind === "updated") cancelled.push(result.task);
    }
    return cancelled;
  }
}
