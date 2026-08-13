BEGIN;

SET LOCAL search_path = public, pg_catalog;

-- Upgrade existing v003 deployments with a durable, token-CAS recovery lease.
-- The raw scope remains in the short-lived operation row and is deleted in the
-- terminal transaction; the permanent tombstone remains hash-only.
ALTER TABLE teachlab_account_deletion_operations
  ADD COLUMN IF NOT EXISTS recovery_lease_owner_sha256 text,
  ADD COLUMN IF NOT EXISTS recovery_lease_token_sha256 text,
  ADD COLUMN IF NOT EXISTS recovery_lease_expires_at timestamptz,
  ADD COLUMN IF NOT EXISTS recovery_after timestamptz,
  ADD COLUMN IF NOT EXISTS recovery_attempts integer NOT NULL DEFAULT 0;

DO $migration$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
     WHERE conrelid = 'public.teachlab_account_deletion_operations'::regclass
       AND conname = 'teachlab_account_deletion_recovery_owner_sha256'
  ) THEN
    ALTER TABLE teachlab_account_deletion_operations
      ADD CONSTRAINT teachlab_account_deletion_recovery_owner_sha256 CHECK (
        recovery_lease_owner_sha256 IS NULL
        OR recovery_lease_owner_sha256 ~ '^[0-9a-f]{64}$'
      );
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
     WHERE conrelid = 'public.teachlab_account_deletion_operations'::regclass
       AND conname = 'teachlab_account_deletion_recovery_token_sha256'
  ) THEN
    ALTER TABLE teachlab_account_deletion_operations
      ADD CONSTRAINT teachlab_account_deletion_recovery_token_sha256 CHECK (
        recovery_lease_token_sha256 IS NULL
        OR recovery_lease_token_sha256 ~ '^[0-9a-f]{64}$'
      );
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
     WHERE conrelid = 'public.teachlab_account_deletion_operations'::regclass
       AND conname = 'teachlab_account_deletion_recovery_lease_complete'
  ) THEN
    ALTER TABLE teachlab_account_deletion_operations
      ADD CONSTRAINT teachlab_account_deletion_recovery_lease_complete CHECK (
        (recovery_lease_owner_sha256 IS NULL
          AND recovery_lease_token_sha256 IS NULL
          AND recovery_lease_expires_at IS NULL)
        OR
        (recovery_lease_owner_sha256 IS NOT NULL
          AND recovery_lease_token_sha256 IS NOT NULL
          AND recovery_lease_expires_at IS NOT NULL)
      );
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
     WHERE conrelid = 'public.teachlab_account_deletion_operations'::regclass
       AND conname = 'teachlab_account_deletion_recovery_attempts'
  ) THEN
    ALTER TABLE teachlab_account_deletion_operations
      ADD CONSTRAINT teachlab_account_deletion_recovery_attempts
      CHECK (recovery_attempts >= 0);
  END IF;
END
$migration$;

CREATE INDEX IF NOT EXISTS teachlab_account_deletion_recovery_due
  ON teachlab_account_deletion_operations (
    recovery_after ASC NULLS FIRST,
    recovery_lease_expires_at ASC NULLS FIRST,
    updated_at ASC
  ) WHERE phase <> 'prepared';

-- PostgreSQL combines permissive policies with OR. Ordinary tenant traffic
-- still needs the tenant/owner GUCs; only AccountSystemDatabase's internal
-- recovery transaction sets this transaction-local marker.
DROP POLICY IF EXISTS teachlab_account_deletion_operations_recovery
  ON teachlab_account_deletion_operations;
CREATE POLICY teachlab_account_deletion_operations_recovery
  ON teachlab_account_deletion_operations
  USING (current_setting('app.account_deletion_recovery', true) = 'enabled')
  WITH CHECK (current_setting('app.account_deletion_recovery', true) = 'enabled');

COMMIT;
