import {
  BadRequestException,
  Controller,
  Headers,
  Inject,
  MessageEvent,
  Param,
  Query,
  Sse,
  UseGuards
} from "@nestjs/common";
import {interval, map, merge, Observable} from "rxjs";

import type {AuthenticatedPrincipal} from "../auth/auth-provider.port";
import {CurrentPrincipal} from "../auth/current-principal.decorator";
import {AppConfigService} from "../config/app-config.service";
import type {TaskEvent} from "./task-event.types";
import {EventsService} from "./events.service";
import {accessScopeFor} from "../tenancy/access-scope";
import {EventsOwnershipGuard} from "./events-ownership.guard";

@Controller()
@UseGuards(EventsOwnershipGuard)
export class EventsController {
  constructor(
    @Inject(EventsService) private readonly events: EventsService,
    @Inject(AppConfigService) private readonly config: AppConfigService
  ) {}

  @Sse("api/events")
  compatibilityStream(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Query("session_id") sessionId: string | undefined,
    @Headers("last-event-id") lastEventId?: string
  ): Observable<MessageEvent> {
    if (!sessionId) throw new BadRequestException("session_id is required");
    return this.toServerSentEvents(principal, sessionId, lastEventId);
  }

  @Sse("api/v1/sessions/:sessionId/events")
  stream(
    @CurrentPrincipal() principal: AuthenticatedPrincipal,
    @Param("sessionId") sessionId: string,
    @Headers("last-event-id") lastEventId?: string
  ): Observable<MessageEvent> {
    return this.toServerSentEvents(principal, sessionId, lastEventId);
  }

  private toServerSentEvents(
    principal: AuthenticatedPrincipal,
    sessionId: string,
    lastEventId?: string
  ): Observable<MessageEvent> {
    const taskEvents = this.events
      .streamOwned(accessScopeFor(principal), sessionId, lastEventId)
      .pipe(
      map((event): MessageEvent => ({id: event.id, data: event}))
      );
    const heartbeats = interval(this.config.sseHeartbeatMs).pipe(
      map(
        (): MessageEvent => ({
          data: {
            id: "heartbeat",
            type: "state",
            sessionId,
            payload: {kind: "heartbeat"},
            occurredAt: new Date().toISOString()
          } satisfies TaskEvent
        })
      )
    );
    return merge(taskEvents, heartbeats);
  }
}
