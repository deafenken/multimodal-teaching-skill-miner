export type TaskEventType = "status" | "message" | "tool" | "state" | "error";

export interface TaskEvent {
  id: string;
  type: TaskEventType;
  sessionId: string;
  payload: Record<string, unknown>;
  occurredAt: string;
}

export type NewTaskEvent = Omit<TaskEvent, "id" | "occurredAt"> & {
  tenantId: string;
  ownerId: string;
};
