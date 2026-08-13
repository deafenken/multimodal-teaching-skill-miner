export type SessionStatus = "active" | "succeeded" | "terminated_unable";

export interface TeachingSession {
  id: string;
  tenantId: string;
  ownerId: string;
  title: string;
  learner: string;
  status: SessionStatus;
  round: number;
  createdAt: string;
  updatedAt: string;
  version: number;
}

export interface CreateSessionRecord {
  id?: string;
  tenantId: string;
  ownerId: string;
  title: string;
  learner: string;
  status?: SessionStatus;
  round?: number;
}

export interface UpdateSessionRecord {
  title?: string;
  learner?: string;
  status?: SessionStatus;
  round?: number;
}

export type UpdateSessionResult =
  | {kind: "updated"; session: TeachingSession}
  | {kind: "not_found"}
  | {kind: "version_conflict"; currentVersion: number}
  | {kind: "state_conflict"; currentStatus: SessionStatus};
