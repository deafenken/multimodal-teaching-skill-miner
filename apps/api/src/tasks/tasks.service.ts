import {ConflictException, Inject, Injectable, NotFoundException} from "@nestjs/common";

import {TASK_ORCHESTRATOR} from "../platform/tokens";
import {SessionsService} from "../sessions/sessions.service";
import type {TaskOrchestratorPort} from "./task-orchestrator.port";
import type {AgentTask} from "./task.types";
import type {AccessScope} from "../tenancy/access-scope";
import {TaskSessionNotActiveError} from "./task-repository.port";

@Injectable()
export class TasksService {
  constructor(
    @Inject(SessionsService) private readonly sessions: SessionsService,
    @Inject(TASK_ORCHESTRATOR) private readonly tasks: TaskOrchestratorPort
  ) {}

  async enqueue(
    scope: AccessScope,
    sessionId: string,
    learnerMessage: string,
    clientRequestId?: string
  ): Promise<AgentTask> {
    const session = await this.sessions.getOwned(scope, sessionId);
    if (session.status !== "active") {
      throw new ConflictException("Cannot enqueue work for a terminal session");
    }
    try {
      return await this.tasks.enqueue({
        tenantId: scope.tenantId,
        ownerId: scope.ownerId,
        sessionId,
        learnerMessage,
        clientRequestId
      });
    } catch (error) {
      if (error instanceof TaskSessionNotActiveError) {
        throw new ConflictException("Cannot enqueue work for a terminal session");
      }
      throw error;
    }
  }

  async getOwned(scope: AccessScope, taskId: string): Promise<AgentTask> {
    const task = await this.tasks.get(scope, taskId);
    if (!task) throw new NotFoundException("Task not found");
    return task;
  }

  async cancelOwned(scope: AccessScope, taskId: string): Promise<AgentTask> {
    await this.getOwned(scope, taskId);
    const task = await this.tasks.cancel(scope, taskId);
    if (!task) throw new NotFoundException("Task not found");
    return task;
  }
}
