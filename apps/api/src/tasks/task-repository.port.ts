import type {AccessScope} from "../tenancy/access-scope";
import type {
  AgentTask,
  AgentTaskStatus,
  EnqueueTaskCommand,
  TaskTransitionResult
} from "./task.types";

export interface CreateTaskResult {
  created: boolean;
  task: AgentTask;
}

export class TaskSessionNotActiveError extends Error {
  constructor() {
    super("The task session is absent, unauthorized, or not active");
    this.name = "TaskSessionNotActiveError";
  }
}

export interface TaskRepositoryPort {
  create(command: EnqueueTaskCommand): Promise<CreateTaskResult>;
  findOwnedById(scope: AccessScope, taskId: string): Promise<AgentTask | undefined>;
  transitionOwned(
    scope: AccessScope,
    taskId: string,
    expectedStatuses: AgentTaskStatus[],
    patch: Partial<Pick<AgentTask, "status" | "failureCode">>
  ): Promise<TaskTransitionResult>;
  cancelOwned(scope: AccessScope, taskId: string): Promise<TaskTransitionResult>;
  cancelScope(
    scope: AccessScope,
    options?: {allowAccountDeleting?: boolean}
  ): Promise<AgentTask[]>;
}
