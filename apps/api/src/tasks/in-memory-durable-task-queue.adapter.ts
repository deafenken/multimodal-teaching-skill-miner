import {Injectable} from "@nestjs/common";

import type {AccessScope} from "../tenancy/access-scope";
import type {
  DurableTaskQueuePort,
  TaskLeaseClaimInput,
  TaskLeaseClaimResult,
  TaskLeaseRetryInput,
  TaskLeaseRetryResult,
  TaskLeaseSettlementInput
} from "./durable-task-queue.port";
import type {AgentTask} from "./task.types";

/**
 * Explicit local-only placeholder. The in-memory orchestrator does not use
 * the durable queue, but keeping a typed provider prevents accidental
 * injection of the PostgreSQL dispatcher in local mode.
 */
@Injectable()
export class InMemoryDurableTaskQueueAdapter implements DurableTaskQueuePort {
  async claimNext(_input: TaskLeaseClaimInput): Promise<TaskLeaseClaimResult> {
    throw new Error("Durable task queue is unavailable in memory mode");
  }

  async renewLease(
    _scope: AccessScope,
    _taskId: string,
    _leaseTokenSha256: string,
    _leaseDurationMs: number
  ): Promise<AgentTask | undefined> {
    throw new Error("Durable task queue is unavailable in memory mode");
  }

  async settleLease(_input: TaskLeaseSettlementInput): Promise<AgentTask | undefined> {
    throw new Error("Durable task queue is unavailable in memory mode");
  }

  async retryLease(_input: TaskLeaseRetryInput): Promise<TaskLeaseRetryResult> {
    throw new Error("Durable task queue is unavailable in memory mode");
  }

  async probe(): Promise<void> {
    return;
  }
}
