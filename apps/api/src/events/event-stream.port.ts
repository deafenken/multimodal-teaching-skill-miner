import type {Observable} from "rxjs";

import type {NewTaskEvent, TaskEvent} from "./task-event.types";
import type {AccessScope} from "../tenancy/access-scope";

export interface EventStreamPort {
  append(event: NewTaskEvent): Promise<TaskEvent>;
  stream(
    scope: AccessScope,
    sessionId: string,
    afterEventId?: string
  ): Observable<TaskEvent>;
}
