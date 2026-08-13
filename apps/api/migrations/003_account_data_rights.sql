BEGIN;

SET LOCAL search_path = public, pg_catalog;

-- Active deletion operations retain raw tenant/owner only while the account is
-- authenticated and deletion is incomplete. The terminal tombstone below is
-- hash-only and remains after every tenant row and auth session is deleted.
CREATE TABLE IF NOT EXISTS teachlab_account_deletion_operations (
  tenant_id text NOT NULL,
  owner_id text NOT NULL,
  scope_sha256 text NOT NULL CHECK (scope_sha256 ~ '^[0-9a-f]{64}$'),
  operation_id text NOT NULL CHECK (operation_id ~ '^adel_[0-9a-f]{32}$'),
  phase text NOT NULL CHECK (phase IN (
    'prepared', 'fencing', 'draining', 'tombstoned', 'quarantining',
    'purging_worker_data', 'committing_database'
  )),
  revision integer NOT NULL CHECK (revision > 0),
  challenge_id text NOT NULL CHECK (challenge_id ~ '^adelc_[0-9a-f]{32}$'),
  challenge_token_sha256 text NOT NULL
    CHECK (challenge_token_sha256 ~ '^[0-9a-f]{64}$'),
  challenge_csrf_sha256 text NOT NULL
    CHECK (challenge_csrf_sha256 ~ '^[0-9a-f]{64}$'),
  challenge_session_sha256 text NOT NULL
    CHECK (challenge_session_sha256 ~ '^[0-9a-f]{64}$'),
  challenge_authority_grant_sha256 text NOT NULL
    CHECK (challenge_authority_grant_sha256 ~ '^[0-9a-f]{64}$'),
  challenge_canonical_identity_sha256 text NOT NULL
    CHECK (challenge_canonical_identity_sha256 ~ '^[0-9a-f]{64}$'),
  challenge_issuer_sha256 text NOT NULL
    CHECK (challenge_issuer_sha256 ~ '^[0-9a-f]{64}$'),
  challenge_authority_key_version text NOT NULL
    CHECK (challenge_authority_key_version ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$'),
  challenge_authenticated_at timestamptz NOT NULL,
  challenge_assurance_level smallint NOT NULL
    CHECK (challenge_assurance_level BETWEEN 2 AND 3),
  challenge_expires_at timestamptz NOT NULL,
  status_capability_sha256 text NOT NULL
    CHECK (status_capability_sha256 ~ '^[0-9a-f]{64}$'),
  idempotency_key_sha256 text CHECK (
    idempotency_key_sha256 IS NULL
    OR idempotency_key_sha256 ~ '^[0-9a-f]{64}$'
  ),
  confirmation_sha256 text CHECK (
    confirmation_sha256 IS NULL
    OR confirmation_sha256 ~ '^[0-9a-f]{64}$'
  ),
  retryable_failure_code text CHECK (
    retryable_failure_code IS NULL
    OR retryable_failure_code ~ '^[a-z][a-z0-9_]{2,63}$'
  ),
  recovery_lease_owner_sha256 text
    CONSTRAINT teachlab_account_deletion_recovery_owner_sha256 CHECK (
    recovery_lease_owner_sha256 IS NULL
    OR recovery_lease_owner_sha256 ~ '^[0-9a-f]{64}$'
  ),
  recovery_lease_token_sha256 text
    CONSTRAINT teachlab_account_deletion_recovery_token_sha256 CHECK (
    recovery_lease_token_sha256 IS NULL
    OR recovery_lease_token_sha256 ~ '^[0-9a-f]{64}$'
  ),
  recovery_lease_expires_at timestamptz,
  recovery_after timestamptz,
  recovery_attempts integer NOT NULL DEFAULT 0
    CONSTRAINT teachlab_account_deletion_recovery_attempts
    CHECK (recovery_attempts >= 0),
  postgres_event_count integer NOT NULL DEFAULT 0 CHECK (postgres_event_count >= 0),
  postgres_task_count integer NOT NULL DEFAULT 0 CHECK (postgres_task_count >= 0),
  postgres_artifact_count integer NOT NULL DEFAULT 0 CHECK (postgres_artifact_count >= 0),
  postgres_session_count integer NOT NULL DEFAULT 0 CHECK (postgres_session_count >= 0),
  postgres_auth_session_count integer NOT NULL DEFAULT 0
    CHECK (postgres_auth_session_count >= 0),
  worker_file_count integer NOT NULL DEFAULT 0 CHECK (worker_file_count >= 0),
  worker_byte_count bigint NOT NULL DEFAULT 0 CHECK (worker_byte_count >= 0),
  worker_root_count integer NOT NULL DEFAULT 0 CHECK (worker_root_count >= 0),
  created_at timestamptz NOT NULL DEFAULT statement_timestamp(),
  updated_at timestamptz NOT NULL DEFAULT statement_timestamp(),
  PRIMARY KEY (tenant_id, owner_id),
  UNIQUE (scope_sha256),
  UNIQUE (operation_id),
  CHECK (
    challenge_authenticated_at <= challenge_expires_at
    AND challenge_expires_at <= challenge_authenticated_at + interval '5 minutes'
  ),
  CHECK (
    (phase = 'prepared' AND idempotency_key_sha256 IS NULL
      AND confirmation_sha256 IS NULL)
    OR
    (phase <> 'prepared' AND idempotency_key_sha256 IS NOT NULL
      AND confirmation_sha256 IS NOT NULL)
  ),
  CONSTRAINT teachlab_account_deletion_recovery_lease_complete CHECK (
    (recovery_lease_owner_sha256 IS NULL
      AND recovery_lease_token_sha256 IS NULL
      AND recovery_lease_expires_at IS NULL)
    OR
    (recovery_lease_owner_sha256 IS NOT NULL
      AND recovery_lease_token_sha256 IS NOT NULL
      AND recovery_lease_expires_at IS NOT NULL)
  )
);

CREATE INDEX IF NOT EXISTS teachlab_account_deletion_operations_updated
  ON teachlab_account_deletion_operations (updated_at ASC);

ALTER TABLE teachlab_account_deletion_operations ENABLE ROW LEVEL SECURITY;
ALTER TABLE teachlab_account_deletion_operations FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS teachlab_account_deletion_operations_tenant_owner
  ON teachlab_account_deletion_operations;
CREATE POLICY teachlab_account_deletion_operations_tenant_owner
  ON teachlab_account_deletion_operations
  USING (
    tenant_id = current_setting('app.tenant_id', true)
    AND owner_id = current_setting('app.user_id', true)
  )
  WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
    AND owner_id = current_setting('app.user_id', true)
  );

DROP POLICY IF EXISTS teachlab_account_deletion_operations_recovery
  ON teachlab_account_deletion_operations;
CREATE POLICY teachlab_account_deletion_operations_recovery
  ON teachlab_account_deletion_operations
  USING (current_setting('app.account_deletion_recovery', true) = 'enabled')
  WITH CHECK (current_setting('app.account_deletion_recovery', true) = 'enabled');

-- This is intentionally outside tenant RLS: after account rows and raw scope
-- claims are gone, startup/authentication must still be able to fail closed by
-- comparing server-derived candidate hashes. It contains no raw identity.
CREATE TABLE IF NOT EXISTS teachlab_account_deletion_tombstones (
  scope_sha256 text PRIMARY KEY CHECK (scope_sha256 ~ '^[0-9a-f]{64}$'),
  receipt_scope_sha256 text NOT NULL
    CHECK (receipt_scope_sha256 ~ '^[0-9a-f]{64}$'),
  operation_id_sha256 text NOT NULL
    CHECK (operation_id_sha256 ~ '^[0-9a-f]{64}$'),
  phase text NOT NULL CHECK (phase IN (
    'tombstoned', 'quarantining', 'purging_worker_data',
    'committing_database', 'completed'
  )),
  revision integer NOT NULL CHECK (revision > 0),
  status_capability_sha256 text NOT NULL
    CHECK (status_capability_sha256 ~ '^[0-9a-f]{64}$'),
  retryable_failure_code text CHECK (
    retryable_failure_code IS NULL
    OR retryable_failure_code ~ '^[a-z][a-z0-9_]{2,63}$'
  ),
  postgres_event_count integer NOT NULL DEFAULT 0 CHECK (postgres_event_count >= 0),
  postgres_task_count integer NOT NULL DEFAULT 0 CHECK (postgres_task_count >= 0),
  postgres_artifact_count integer NOT NULL DEFAULT 0 CHECK (postgres_artifact_count >= 0),
  postgres_session_count integer NOT NULL DEFAULT 0 CHECK (postgres_session_count >= 0),
  postgres_auth_session_count integer NOT NULL DEFAULT 0
    CHECK (postgres_auth_session_count >= 0),
  worker_file_count integer NOT NULL DEFAULT 0 CHECK (worker_file_count >= 0),
  worker_byte_count bigint NOT NULL DEFAULT 0 CHECK (worker_byte_count >= 0),
  worker_root_count integer NOT NULL DEFAULT 0 CHECK (worker_root_count >= 0),
  deleted_at timestamptz,
  receipt_id text CHECK (
    receipt_id IS NULL OR receipt_id ~ '^adelr_[0-9a-f]{32}$'
  ),
  receipt_sha256 text CHECK (
    receipt_sha256 IS NULL OR receipt_sha256 ~ '^[0-9a-f]{64}$'
  ),
  created_at timestamptz NOT NULL DEFAULT statement_timestamp(),
  updated_at timestamptz NOT NULL DEFAULT statement_timestamp(),
  CHECK (
    (phase = 'completed' AND deleted_at IS NOT NULL
      AND receipt_id IS NOT NULL AND receipt_sha256 IS NOT NULL)
    OR
    (phase <> 'completed' AND deleted_at IS NULL
      AND receipt_id IS NULL AND receipt_sha256 IS NULL)
  )
);

CREATE INDEX IF NOT EXISTS teachlab_account_deletion_tombstones_updated
  ON teachlab_account_deletion_tombstones (updated_at ASC);

-- Only the narrowly-scoped PostgreSQL account lifecycle adapter should receive
-- SELECT/INSERT/UPDATE on this table. No browser/API input can submit a hash.

COMMIT;
