import type {AgentTask, EnqueueTaskCommand} from "./task.types";
import type {AccessScope} from "../tenancy/access-scope";

export interface TaskOrchestratorPort {
  enqueue(command: EnqueueTaskCommand): Promise<AgentTask>;
  get(scope: AccessScope, taskId: string): Promise<AgentTask | undefined>;
  cancel(scope: AccessScope, taskId: string): Promise<AgentTask | undefined>;
  fenceScope(scope: AccessScope): Promise<void>;
  drainScope(
    scope: AccessScope,
    timeoutMs: number
  ): Promise<{cancelled: number; handedOff: number}>;
  probe(): Promise<"postgres_durable_lease" | "memory_local_only">;
}
