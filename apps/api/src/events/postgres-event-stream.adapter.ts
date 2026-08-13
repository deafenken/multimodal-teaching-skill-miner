import {randomUUID} from "node:crypto";

import {Inject, Injectable} from "@nestjs/common";
import {Observable} from "rxjs";

import {AppConfigService} from "../config/app-config.service";
import type {TenantDatabasePort} from "../database/tenant-database.port";
import {TENANT_DATABASE} from "../platform/tokens";
import type {AccessScope} from "../tenancy/access-scope";
import type {EventStreamPort} from "./event-stream.port";
import type {
  NewTaskEvent,
  TaskEvent,
  TaskEventType
} from "./task-event.types";

interface EventRow {
  sequence_id: string | number;
  id: string;
  session_id: string;
  event_type: TaskEventType;
  payload: Record<string, unknown> | string;
  occurred_at: Date | string;
}

const EVENT_COLUMNS =
  "sequence_id, id, session_id, event_type, payload, occurred_at";
const EVENT_ID_PATTERN = /^evt_([0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$/i;

function mapEvent(row: EventRow): TaskEvent {
  const payload = typeof row.payload === "string" ? JSON.parse(row.payload) : row.payload;
  return {
    id: `evt_${row.id}`,
    sessionId: row.session_id,
    type: row.event_type,
    payload,
    occurredAt:
      row.occurred_at instanceof Date
        ? row.occurred_at.toISOString()
        : new Date(row.occurred_at).toISOString()
  };
}

@Injectable()
export class PostgresEventStreamAdapter implements EventStreamPort {
  constructor(
    @Inject(TENANT_DATABASE) private readonly database: TenantDatabasePort,
    @Inject(AppConfigService) private readonly config: AppConfigService
  ) {}

  append(input: NewTaskEvent): Promise<TaskEvent> {
    const scope = {tenantId: input.tenantId, ownerId: input.ownerId};
    return this.database.withTenant(scope, async (transaction) => {
      const result = await transaction.query<EventRow>(
        `INSERT INTO public.teachlab_task_events (
           id, tenant_id, owner_id, session_id, event_type, payload
         ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
         RETURNING ${EVENT_COLUMNS}`,
        [
          randomUUID(),
          input.tenantId,
          input.ownerId,
          input.sessionId,
          input.type,
          JSON.stringify(input.payload)
        ]
      );
      const row = result.rows[0];
      if (!row) throw new Error("PostgreSQL did not return the appended event");
      return mapEvent(row);
    });
  }

  stream(
    scope: AccessScope,
    sessionId: string,
    afterEventId?: string
  ): Observable<TaskEvent> {
    return new Observable<TaskEvent>((subscriber) => {
      let stopped = false;
      let timeout: NodeJS.Timeout | undefined;
      let afterSequence: string | undefined;
      let initialCursor = EVENT_ID_PATTERN.exec(afterEventId ?? "")?.[1];

      const poll = async (): Promise<void> => {
        try {
          const rows = await this.fetchBatch(
            scope,
            sessionId,
            afterSequence,
            initialCursor
          );
          initialCursor = undefined;
          for (const row of rows) {
            if (stopped) return;
            afterSequence = String(row.sequence_id);
            subscriber.next(mapEvent(row));
          }
          if (!stopped) timeout = setTimeout(() => void poll(), this.config.eventPollMs);
        } catch (error) {
          if (!stopped) subscriber.error(error);
        }
      };
      void poll();
      return () => {
        stopped = true;
        if (timeout) clearTimeout(timeout);
      };
    });
  }

  private fetchBatch(
    scope: AccessScope,
    sessionId: string,
    afterSequence?: string,
    initialEventId?: string
  ): Promise<EventRow[]> {
    return this.database.withTenant(scope, async (transaction) => {
      if (afterSequence) {
        const result = await transaction.query<EventRow>(
          `SELECT ${EVENT_COLUMNS}
             FROM public.teachlab_task_events
            WHERE tenant_id = $1 AND owner_id = $2 AND session_id = $3
              AND sequence_id > $4::bigint
            ORDER BY sequence_id ASC
            LIMIT 200`,
          [scope.tenantId, scope.ownerId, sessionId, afterSequence]
        );
        return result.rows;
      }
      const result = await transaction.query<EventRow>(
        `WITH cursor AS (
           SELECT sequence_id
             FROM public.teachlab_task_events
            WHERE tenant_id = $1 AND owner_id = $2 AND session_id = $3
              AND id = $4::uuid
         )
         SELECT ${EVENT_COLUMNS}
           FROM (
             SELECT ${EVENT_COLUMNS}
               FROM public.teachlab_task_events
              WHERE tenant_id = $1 AND owner_id = $2 AND session_id = $3
                AND sequence_id > COALESCE((SELECT sequence_id FROM cursor), 0)
              ORDER BY sequence_id DESC
              LIMIT 200
           ) AS recent
          ORDER BY sequence_id ASC`,
        [scope.tenantId, scope.ownerId, sessionId, initialEventId ?? randomUUID()]
      );
      return result.rows;
    });
  }
}
