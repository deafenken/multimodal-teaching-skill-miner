BEGIN;

SET LOCAL search_path = public, pg_catalog;

CREATE TABLE IF NOT EXISTS teachlab_auth_sessions (
  tenant_id text NOT NULL,
  owner_id text NOT NULL,
  session_id_sha256 text NOT NULL
    CHECK (session_id_sha256 ~ '^[0-9a-f]{64}$'),
  issued_at timestamptz NOT NULL,
  expires_at timestamptz NOT NULL,
  revoked_at timestamptz,
  revocation_reason text,
  version integer NOT NULL DEFAULT 1 CHECK (version > 0),
  PRIMARY KEY (tenant_id, owner_id, session_id_sha256),
  CHECK (expires_at > issued_at),
  CHECK (
    (revoked_at IS NULL AND revocation_reason IS NULL)
    OR
    (revoked_at IS NOT NULL AND revocation_reason = 'user_logout')
  ),
  CHECK (revoked_at IS NULL OR revoked_at >= issued_at)
);

CREATE INDEX IF NOT EXISTS teachlab_auth_sessions_owner_expiry
  ON teachlab_auth_sessions (tenant_id, owner_id, expires_at ASC);

ALTER TABLE teachlab_auth_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE teachlab_auth_sessions FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS teachlab_auth_sessions_tenant_owner
  ON teachlab_auth_sessions;
CREATE POLICY teachlab_auth_sessions_tenant_owner ON teachlab_auth_sessions
  USING (
    tenant_id = current_setting('app.tenant_id', true)
    AND owner_id = current_setting('app.user_id', true)
  )
  WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
    AND owner_id = current_setting('app.user_id', true)
  );

COMMIT;
