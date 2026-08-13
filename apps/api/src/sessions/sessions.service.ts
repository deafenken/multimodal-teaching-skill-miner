import {ConflictException, Inject, Injectable, NotFoundException} from "@nestjs/common";

import type {EventStreamPort} from "../events/event-stream.port";
import {EVENT_STREAM, SESSION_REPOSITORY} from "../platform/tokens";
import type {SessionRepositoryPort} from "./session-repository.port";
import type {TeachingSession, UpdateSessionRecord} from "./session.types";
import type {AccessScope} from "../tenancy/access-scope";

@Injectable()
export class SessionsService {
  constructor(
    @Inject(SESSION_REPOSITORY) private readonly sessions: SessionRepositoryPort,
    @Inject(EVENT_STREAM) private readonly events: EventStreamPort
  ) {}

  list(scope: AccessScope): Promise<TeachingSession[]> {
    return this.sessions.listByOwner(scope);
  }

  async create(scope: AccessScope, title: string, learner: string): Promise<TeachingSession> {
    const session = await this.sessions.create({
      tenantId: scope.tenantId,
      ownerId: scope.ownerId,
      title,
      learner
    });
    await this.events.append({
      ...scope,
      sessionId: session.id,
      type: "state",
      payload: {kind: "session.created", status: session.status, round: session.round}
    });
    return session;
  }

  async getOwned(scope: AccessScope, sessionId: string): Promise<TeachingSession> {
    const session = await this.sessions.findOwnedById(scope, sessionId);
    if (!session) {
      throw new NotFoundException("Session not found");
    }
    return session;
  }

  async updateOwned(
    scope: AccessScope,
    sessionId: string,
    patch: UpdateSessionRecord,
    expectedVersion: number
  ): Promise<TeachingSession> {
    const result = await this.sessions.updateOwned(
      scope,
      sessionId,
      patch,
      expectedVersion
    );
    if (result.kind === "not_found") throw new NotFoundException("Session not found");
    if (result.kind === "version_conflict") {
      throw new ConflictException({
        code: "session_version_conflict",
        currentVersion: result.currentVersion
      });
    }
    if (result.kind === "state_conflict") {
      throw new ConflictException({
        code: "session_state_conflict",
        currentStatus: result.currentStatus,
        message: "A terminal session cannot be reopened through the update endpoint"
      });
    }
    const updated = result.session;
    await this.events.append({
      ...scope,
      sessionId,
      type: "state",
      payload: {kind: "session.updated", status: updated.status, round: updated.round}
    });
    return updated;
  }
}
