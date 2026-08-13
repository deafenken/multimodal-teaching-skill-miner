# PostgreSQL migration boundary

`001_tenant_rls.sql` creates the Session, Task, and Event tables. The ordered
`002_auth_session_revocation.sql` adds the hash-only per-device session authority
table. `003_account_data_rights.sql` adds the resumable deletion operation and
permanent hash-only tombstone tables; `004_account_deletion_recovery.sql`
upgrades existing v003 deployments with the token-CAS lease, retry schedule,
and dedicated transaction-local recovery policy. `005_durable_tasks_artifacts.sql`
adds durable task leases, dispatcher policy, and scoped artifact bytes. All scoped tables force row-level security;
the system status-capability lookup has a separate minimum-privilege transaction
port and cannot read learner tables. Apply them with a migration-owner role, then grant
only the required DML privileges to a separate `teachlab_app` login role. The API
refuses startup when its connection role is a superuser, has `BYPASSRLS`, any table
is absent, or RLS is not both enabled and forced.

Example (adapt role names to the deployment):

```sql
\set ON_ERROR_STOP on
\i migrations/001_tenant_rls.sql
\i migrations/002_auth_session_revocation.sql
\i migrations/003_account_data_rights.sql
\i migrations/004_account_deletion_recovery.sql
\i migrations/005_durable_tasks_artifacts.sql

GRANT SELECT, INSERT, UPDATE, DELETE
  ON teachlab_teaching_sessions, teachlab_agent_tasks, teachlab_task_events,
     teachlab_auth_sessions, teachlab_account_deletion_operations,
     teachlab_account_deletion_tombstones, teachlab_artifacts
  TO teachlab_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO teachlab_app;
```

The application sets `app.tenant_id` and `app.user_id` with transaction-local
`set_config` calls before every repository operation. Never expose the database
credential to a browser or worker sandbox. Do not give the application role table
ownership, superuser, or `BYPASSRLS` privileges. `teachlab_auth_sessions` stores
only tenant/owner scope, SHA-256 of the random session UUID, timestamps, bounded
reason and CAS version; raw Cookie, CSRF and OIDC tokens are forbidden by the
startup column allowlist. A completed account deletion removes its raw-scope
operation row and keeps only one or more scope-key-version hash tombstones plus
content-free counts/receipt hash. Tombstones are permanent write fences and must
not be removed by ordinary retention jobs.

The fast local suite exercises the SQL adapter through a transaction-contract fake
and inspects the migration invariants. CI additionally runs this migration against
a disposable PostgreSQL 17 service with separate owner/application roles and checks
real FORCE-RLS isolation, write rejection, CAS, and idempotency. A target staging
database migration, TLS check, backup restore, and failover drill remain deployment
gates because an ephemeral CI database cannot establish those environment-specific
properties.
