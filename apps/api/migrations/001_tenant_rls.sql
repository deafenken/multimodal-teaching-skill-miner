BEGIN;

SET LOCAL search_path = public, pg_catalog;

CREATE TABLE IF NOT EXISTS teachlab_teaching_sessions (
  id uuid NOT NULL,
  tenant_id text NOT NULL,
  owner_id text NOT NULL,
  title text NOT NULL CHECK (char_length(title) BETWEEN 1 AND 120),
  learner text NOT NULL CHECK (char_length(learner) BETWEEN 1 AND 80),
  status text NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'succeeded', 'terminated_unable')),
  round integer NOT NULL DEFAULT 0 CHECK (round >= 0),
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW(),
  version integer NOT NULL DEFAULT 1 CHECK (version > 0),
  PRIMARY KEY (tenant_id, owner_id, id)
);

CREATE UNIQUE INDEX IF NOT EXISTS teachlab_teaching_sessions_id_unique
  ON teachlab_teaching_sessions (id);
CREATE INDEX IF NOT EXISTS teachlab_teaching_sessions_owner_updated
  ON teachlab_teaching_sessions (tenant_id, owner_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS teachlab_agent_tasks (
  id uuid NOT NULL,
  tenant_id text NOT NULL,
  owner_id text NOT NULL,
  session_id uuid NOT NULL,
  learner_message text NOT NULL CHECK (char_length(learner_message) BETWEEN 1 AND 20000),
  client_request_id text,
  status text NOT NULL DEFAULT 'queued'
    CHECK (status IN (
      'queued', 'running', 'requires_configuration', 'succeeded', 'failed', 'cancelled'
    )),
  failure_code text,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW(),
  version integer NOT NULL DEFAULT 1 CHECK (version > 0),
  PRIMARY KEY (tenant_id, owner_id, id),
  FOREIGN KEY (tenant_id, owner_id, session_id)
    REFERENCES teachlab_teaching_sessions (tenant_id, owner_id, id)
    ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS teachlab_agent_tasks_id_unique
  ON teachlab_agent_tasks (id);
CREATE UNIQUE INDEX IF NOT EXISTS teachlab_agent_tasks_idempotency_unique
  ON teachlab_agent_tasks (tenant_id, owner_id, session_id, client_request_id)
  WHERE client_request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS teachlab_agent_tasks_session_created
  ON teachlab_agent_tasks (tenant_id, owner_id, session_id, created_at);

CREATE TABLE IF NOT EXISTS teachlab_task_events (
  sequence_id bigint GENERATED ALWAYS AS IDENTITY,
  id uuid NOT NULL,
  tenant_id text NOT NULL,
  owner_id text NOT NULL,
  session_id uuid NOT NULL,
  event_type text NOT NULL CHECK (event_type IN ('status', 'message', 'tool', 'state', 'error')),
  payload jsonb NOT NULL,
  occurred_at timestamptz NOT NULL DEFAULT NOW(),
  PRIMARY KEY (tenant_id, owner_id, sequence_id),
  UNIQUE (tenant_id, owner_id, id),
  FOREIGN KEY (tenant_id, owner_id, session_id)
    REFERENCES teachlab_teaching_sessions (tenant_id, owner_id, id)
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS teachlab_task_events_session_sequence
  ON teachlab_task_events (tenant_id, owner_id, session_id, sequence_id);

ALTER TABLE teachlab_teaching_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE teachlab_teaching_sessions FORCE ROW LEVEL SECURITY;
ALTER TABLE teachlab_agent_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE teachlab_agent_tasks FORCE ROW LEVEL SECURITY;
ALTER TABLE teachlab_task_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE teachlab_task_events FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS teachlab_sessions_tenant_owner ON teachlab_teaching_sessions;
CREATE POLICY teachlab_sessions_tenant_owner ON teachlab_teaching_sessions
  USING (
    tenant_id = current_setting('app.tenant_id', true)
    AND owner_id = current_setting('app.user_id', true)
  )
  WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
    AND owner_id = current_setting('app.user_id', true)
  );

DROP POLICY IF EXISTS teachlab_tasks_tenant_owner ON teachlab_agent_tasks;
CREATE POLICY teachlab_tasks_tenant_owner ON teachlab_agent_tasks
  USING (
    tenant_id = current_setting('app.tenant_id', true)
    AND owner_id = current_setting('app.user_id', true)
  )
  WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
    AND owner_id = current_setting('app.user_id', true)
  );

DROP POLICY IF EXISTS teachlab_events_tenant_owner ON teachlab_task_events;
CREATE POLICY teachlab_events_tenant_owner ON teachlab_task_events
  USING (
    tenant_id = current_setting('app.tenant_id', true)
    AND owner_id = current_setting('app.user_id', true)
  )
  WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
    AND owner_id = current_setting('app.user_id', true)
  );

COMMIT;
