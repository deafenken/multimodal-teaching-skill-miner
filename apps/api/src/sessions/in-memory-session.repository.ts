import {randomUUID} from "node:crypto";

import {Inject, Injectable} from "@nestjs/common";

import {AppConfigService} from "../config/app-config.service";
import type {SessionRepositoryPort} from "./session-repository.port";
import type {
  CreateSessionRecord,
  TeachingSession,
  UpdateSessionRecord,
  UpdateSessionResult
} from "./session.types";
import type {AccessScope} from "../tenancy/access-scope";

const DEMO_SESSIONS: Array<Omit<CreateSessionRecord, "ownerId" | "tenantId">> = [
  {id: "dp-dark-mode", title: "理解动态规划状态定义", learner: "小雨", round: 3},
  {id: "weekly-review", title: "每周错题回顾", learner: "子墨"},
  {
    id: "binary-search",
    title: "二分边界为什么会错",
    learner: "知行",
    status: "succeeded",
    round: 7
  },
  {id: "recursion", title: "递归与记忆化的区别", learner: "小雨", round: 4},
  {
    id: "transfer",
    title: "把状态转移迁移到新题",
    learner: "子墨",
    status: "terminated_unable",
    round: 6
  }
];

@Injectable()
export class InMemorySessionRepository implements SessionRepositoryPort {
  private readonly sessions = new Map<string, TeachingSession>();

  constructor(@Inject(AppConfigService) config: AppConfigService) {
    if (config.seedDemoSessions) {
      for (const seed of DEMO_SESSIONS) {
        this.createSync({
          ...seed,
          ownerId: config.developmentUserId,
          tenantId: config.developmentTenantId
        });
      }
    }
  }

  async create(input: CreateSessionRecord): Promise<TeachingSession> {
    return this.createSync(input);
  }

  async listByOwner(scope: AccessScope): Promise<TeachingSession[]> {
    return [...this.sessions.values()]
      .filter(
        (session) =>
          session.tenantId === scope.tenantId && session.ownerId === scope.ownerId
      )
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
      .map((session) => structuredClone(session));
  }

  async findOwnedById(
    scope: AccessScope,
    id: string
  ): Promise<TeachingSession | undefined> {
    const session = this.sessions.get(id);
    return session &&
      session.tenantId === scope.tenantId &&
      session.ownerId === scope.ownerId
      ? structuredClone(session)
      : undefined;
  }

  async updateOwned(
    scope: AccessScope,
    id: string,
    patch: UpdateSessionRecord,
    expectedVersion: number
  ): Promise<UpdateSessionResult> {
    const current = this.sessions.get(id);
    if (
      !current ||
      current.tenantId !== scope.tenantId ||
      current.ownerId !== scope.ownerId
    ) {
      return {kind: "not_found"};
    }
    if (current.version !== expectedVersion) {
      return {kind: "version_conflict", currentVersion: current.version};
    }
    if (
      patch.status &&
      current.status !== "active" &&
      patch.status !== current.status
    ) {
      return {kind: "state_conflict", currentStatus: current.status};
    }
    const updated: TeachingSession = {
      ...current,
      ...patch,
      updatedAt: new Date().toISOString(),
      version: current.version + 1
    };
    this.sessions.set(id, updated);
    return {kind: "updated", session: structuredClone(updated)};
  }

  private createSync(input: CreateSessionRecord): TeachingSession {
    const id = input.id ?? randomUUID();
    if (this.sessions.has(id)) throw new Error(`Session ${id} already exists`);
    const now = new Date().toISOString();
    const session: TeachingSession = {
      id,
      tenantId: input.tenantId,
      ownerId: input.ownerId,
      title: input.title,
      learner: input.learner,
      status: input.status ?? "active",
      round: input.round ?? 0,
      createdAt: now,
      updatedAt: now,
      version: 1
    };
    this.sessions.set(id, session);
    return structuredClone(session);
  }
}
