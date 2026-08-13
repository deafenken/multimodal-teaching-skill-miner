export type AgentTaskStatus =
  | "queued"
  | "running"
  | "requires_configuration"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface AgentTask {
  id: string;
  tenantId: string;
  ownerId: string;
  sessionId: string;
  learnerMessage: string;
  clientRequestId?: string;
  status: AgentTaskStatus;
  createdAt: string;
  updatedAt: string;
  failureCode?: string;
  attemptCount: number;
  availableAt: string;
  version: number;
}

export interface EnqueueTaskCommand {
  tenantId: string;
  ownerId: string;
  sessionId: string;
  learnerMessage: string;
  clientRequestId?: string;
}

export type TaskTransitionResult =
  | {kind: "updated"; task: AgentTask}
  | {kind: "not_found"}
  | {kind: "state_conflict"; task: AgentTask};
