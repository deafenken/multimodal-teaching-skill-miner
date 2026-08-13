import {Injectable} from "@nestjs/common";
import {Observable, Subject} from "rxjs";

import type {AccessScope} from "../tenancy/access-scope";
import type {EventStreamPort} from "./event-stream.port";
import type {NewTaskEvent, TaskEvent} from "./task-event.types";

const MAX_EVENTS_PER_SESSION = 200;

interface ScopedTaskEvent extends TaskEvent {
  tenantId: string;
  ownerId: string;
}

function publicEvent(event: ScopedTaskEvent): TaskEvent {
  const {tenantId: _tenantId, ownerId: _ownerId, ...safe} = event;
  return structuredClone(safe);
}

function streamKey(scope: AccessScope, sessionId: string): string {
  return `${scope.tenantId}\u0000${scope.ownerId}\u0000${sessionId}`;
}

@Injectable()
export class InMemoryEventStreamAdapter implements EventStreamPort {
  private sequence = 0;
  private readonly events = new Map<string, ScopedTaskEvent[]>();
  private readonly subjects = new Map<string, Subject<ScopedTaskEvent>>();

  async append(input: NewTaskEvent): Promise<TaskEvent> {
    const event: ScopedTaskEvent = {
      ...input,
      id: `evt_${String(++this.sequence).padStart(12, "0")}`,
      occurredAt: new Date().toISOString()
    };
    const key = streamKey(
      {tenantId: input.tenantId, ownerId: input.ownerId},
      input.sessionId
    );
    const backlog = this.events.get(key) ?? [];
    backlog.push(event);
    if (backlog.length > MAX_EVENTS_PER_SESSION) backlog.shift();
    this.events.set(key, backlog);
    this.subjectFor(key).next(event);
    return publicEvent(event);
  }

  stream(
    scope: AccessScope,
    sessionId: string,
    afterEventId?: string
  ): Observable<TaskEvent> {
    const key = streamKey(scope, sessionId);
    return new Observable<TaskEvent>((subscriber) => {
      const backlog = this.events.get(key) ?? [];
      const lastSeenIndex = afterEventId
        ? backlog.findIndex((event) => event.id === afterEventId)
        : -1;
      const replay =
        afterEventId && lastSeenIndex >= 0 ? backlog.slice(lastSeenIndex + 1) : backlog;
      for (const event of replay) subscriber.next(publicEvent(event));

      const liveSubscription = this.subjectFor(key).subscribe((event) => {
        subscriber.next(publicEvent(event));
      });
      return () => liveSubscription.unsubscribe();
    });
  }

  private subjectFor(key: string): Subject<ScopedTaskEvent> {
    let subject = this.subjects.get(key);
    if (!subject) {
      subject = new Subject<ScopedTaskEvent>();
      this.subjects.set(key, subject);
    }
    return subject;
  }
}
