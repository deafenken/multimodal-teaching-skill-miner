import type {NewTaskEvent} from "../events/task-event.types";
import type {AccessScope} from "../tenancy/access-scope";
import type {AgentTask} from "./task.types";

export interface TaskLeaseClaimInput {
  leaseOwnerSha256: string;
  leaseTokenSha256: string;
  leaseDurationMs: number;
  maximumAttempts: number;
}

export type TaskLeaseClaimResult =
  | {kind: "claimed"; task: AgentTask; leaseTokenSha256: string}
  | {
      kind: "terminalized";
      task: AgentTask;
      leaseTokenSha256?: string;
      reason: "attempts_exhausted" | "account_deletion_in_progress";
    }
  | {kind: "none"};

export interface TaskLeaseSettlementInput extends AccessScope {
  taskId: string;
  leaseTokenSha256: string;
  status: "succeeded" | "failed" | "requires_configuration";
  failureCode?: string;
  events: readonly Pick<NewTaskEvent, "type" | "payload">[];
}

export interface TaskLeaseRetryInput extends AccessScope {
  taskId: string;
  leaseTokenSha256: string;
  failureCode: string;
  maximumAttempts: number;
  retryBaseMs: number;
  retryMaximumMs: number;
  events: readonly Pick<NewTaskEvent, "type" | "payload">[];
}

export type TaskLeaseRetryResult =
  | {kind: "retry_scheduled"; task: AgentTask}
  | {kind: "failed"; task: AgentTask}
  | {kind: "lease_lost"};

export interface DurableTaskQueuePort {
  claimNext(input: TaskLeaseClaimInput): Promise<TaskLeaseClaimResult>;
  renewLease(
    scope: AccessScope,
    taskId: string,
    leaseTokenSha256: string,
    leaseDurationMs: number
  ): Promise<AgentTask | undefined>;
  settleLease(input: TaskLeaseSettlementInput): Promise<AgentTask | undefined>;
  retryLease(input: TaskLeaseRetryInput): Promise<TaskLeaseRetryResult>;
  probe(): Promise<void>;
}
