import {Inject, Injectable} from "@nestjs/common";
import {defer, from, Observable, switchMap} from "rxjs";

import {EVENT_STREAM} from "../platform/tokens";
import {SessionsService} from "../sessions/sessions.service";
import type {EventStreamPort} from "./event-stream.port";
import type {TaskEvent} from "./task-event.types";
import type {AccessScope} from "../tenancy/access-scope";

@Injectable()
export class EventsService {
  constructor(
    @Inject(SessionsService) private readonly sessions: SessionsService,
    @Inject(EVENT_STREAM) private readonly events: EventStreamPort
  ) {}

  streamOwned(
    scope: AccessScope,
    sessionId: string,
    afterEventId?: string
  ): Observable<TaskEvent> {
    return defer(() => from(this.sessions.getOwned(scope, sessionId))).pipe(
      switchMap(() => this.events.stream(scope, sessionId, afterEventId))
    );
  }
}
