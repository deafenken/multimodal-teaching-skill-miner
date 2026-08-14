BEGIN;

SET LOCAL search_path = public, pg_catalog;

ALTER TABLE teachlab_account_deletion_operations
  ADD COLUMN IF NOT EXISTS postgres_artifact_count integer NOT NULL DEFAULT 0
    CHECK (postgres_artifact_count >= 0);
ALTER TABLE teachlab_account_deletion_tombstones
  ADD COLUMN IF NOT EXISTS postgres_artifact_count integer NOT NULL DEFAULT 0
    CHECK (postgres_artifact_count >= 0);

ALTER TABLE teachlab_agent_tasks
  ADD COLUMN IF NOT EXISTS attempt_count integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS available_at timestamptz NOT NULL DEFAULT statement_timestamp(),
  ADD COLUMN IF NOT EXISTS lease_owner_sha256 text,
  ADD COLUMN IF NOT EXISTS lease_token_sha256 text,
  ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz;

DO $migration$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'public.teachlab_agent_tasks'::regclass AND conname = 'teachlab_agent_tasks_attempt_count') THEN
    ALTER TABLE teachlab_agent_tasks ADD CONSTRAINT teachlab_agent_tasks_attempt_count CHECK (attempt_count >= 0 AND attempt_count <= 20);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'public.teachlab_agent_tasks'::regclass AND conname = 'teachlab_agent_tasks_lease_owner_sha256') THEN
    ALTER TABLE teachlab_agent_tasks ADD CONSTRAINT teachlab_agent_tasks_lease_owner_sha256 CHECK (lease_owner_sha256 IS NULL OR lease_owner_sha256 ~ '^[0-9a-f]{64}$');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'public.teachlab_agent_tasks'::regclass AND conname = 'teachlab_agent_tasks_lease_token_sha256') THEN
    ALTER TABLE teachlab_agent_tasks ADD CONSTRAINT teachlab_agent_tasks_lease_token_sha256 CHECK (lease_token_sha256 IS NULL OR lease_token_sha256 ~ '^[0-9a-f]{64}$');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'public.teachlab_agent_tasks'::regclass AND conname = 'teachlab_agent_tasks_lease_complete') THEN
    ALTER TABLE teachlab_agent_tasks ADD CONSTRAINT teachlab_agent_tasks_lease_complete CHECK (
      (lease_owner_sha256 IS NULL AND lease_token_sha256 IS NULL AND lease_expires_at IS NULL)
      OR (lease_owner_sha256 IS NOT NULL AND lease_token_sha256 IS NOT NULL AND lease_expires_at IS NOT NULL)
    );
  END IF;
END
$migration$;

CREATE INDEX IF NOT EXISTS teachlab_agent_tasks_dispatch_due
  ON teachlab_agent_tasks (available_at ASC, created_at ASC, id)
  WHERE status IN ('queued', 'running');

CREATE TABLE IF NOT EXISTS teachlab_artifacts (
  tenant_id text NOT NULL,
  owner_id text NOT NULL,
  artifact_key text NOT NULL CHECK (
    char_length(artifact_key) BETWEEN 1 AND 512
    AND artifact_key ~ '^[A-Za-z0-9][A-Za-z0-9._/-]*$'
    AND artifact_key !~ '(^|/)\.\.?(/|$)' AND artifact_key !~ '//'
  ),
  content_type text NOT NULL CHECK (char_length(content_type) BETWEEN 1 AND 255),
  byte_length bigint NOT NULL CHECK (byte_length BETWEEN 0 AND 16777216),
  sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  content bytea NOT NULL,
  created_at timestamptz NOT NULL DEFAULT statement_timestamp(),
  updated_at timestamptz NOT NULL DEFAULT statement_timestamp(),
  version integer NOT NULL DEFAULT 1 CHECK (version > 0),
  PRIMARY KEY (tenant_id, owner_id, artifact_key),
  CHECK (octet_length(content) = byte_length),
  CHECK (jsonb_typeof(metadata) = 'object')
);

CREATE INDEX IF NOT EXISTS teachlab_artifacts_owner_updated
  ON teachlab_artifacts (tenant_id, owner_id, updated_at DESC, artifact_key);

ALTER TABLE teachlab_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE teachlab_artifacts FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS teachlab_artifacts_tenant_owner ON teachlab_artifacts;
CREATE POLICY teachlab_artifacts_tenant_owner ON teachlab_artifacts
  USING (tenant_id = current_setting('app.tenant_id', true) AND owner_id = current_setting('app.user_id', true))
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true) AND owner_id = current_setting('app.user_id', true));

DROP POLICY IF EXISTS teachlab_tasks_dispatcher ON teachlab_agent_tasks;
CREATE POLICY teachlab_tasks_dispatcher ON teachlab_agent_tasks
  USING (
    current_setting('app.task_dispatcher', true) = 'enabled'
    AND NOT EXISTS (
      SELECT 1
        FROM public.teachlab_account_deletion_operations AS deletion
       WHERE deletion.tenant_id = teachlab_agent_tasks.tenant_id
         AND deletion.owner_id = teachlab_agent_tasks.owner_id
         AND deletion.phase <> 'prepared'
    )
  )
  WITH CHECK (
    current_setting('app.task_dispatcher', true) = 'enabled'
    AND NOT EXISTS (
      SELECT 1
        FROM public.teachlab_account_deletion_operations AS deletion
       WHERE deletion.tenant_id = teachlab_agent_tasks.tenant_id
         AND deletion.owner_id = teachlab_agent_tasks.owner_id
         AND deletion.phase <> 'prepared'
    )
  );

DROP POLICY IF EXISTS teachlab_events_dispatcher ON teachlab_task_events;
CREATE POLICY teachlab_events_dispatcher ON teachlab_task_events
  USING (
    current_setting('app.task_dispatcher', true) = 'enabled'
    AND NOT EXISTS (
      SELECT 1
        FROM public.teachlab_account_deletion_operations AS deletion
       WHERE deletion.tenant_id = teachlab_task_events.tenant_id
         AND deletion.owner_id = teachlab_task_events.owner_id
         AND deletion.phase <> 'prepared'
    )
  )
  WITH CHECK (
    current_setting('app.task_dispatcher', true) = 'enabled'
    AND NOT EXISTS (
      SELECT 1
        FROM public.teachlab_account_deletion_operations AS deletion
       WHERE deletion.tenant_id = teachlab_task_events.tenant_id
         AND deletion.owner_id = teachlab_task_events.owner_id
         AND deletion.phase <> 'prepared'
    )
  );

DROP POLICY IF EXISTS teachlab_account_deletion_operations_dispatcher ON teachlab_account_deletion_operations;
CREATE POLICY teachlab_account_deletion_operations_dispatcher
  ON teachlab_account_deletion_operations
  FOR SELECT
  USING (current_setting('app.task_dispatcher', true) = 'enabled');

COMMIT;
