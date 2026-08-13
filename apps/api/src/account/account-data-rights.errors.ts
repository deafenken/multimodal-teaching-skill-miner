export type AccountDataRightsErrorCode =
  | "account_export_unavailable"
  | "account_export_limit_exceeded"
  | "account_export_changed_during_stream"
  | "account_deletion_already_started"
  | "account_already_deleted"
  | "account_deletion_challenge_missing"
  | "account_deletion_challenge_expired"
  | "account_deletion_revision_conflict"
  | "account_deletion_confirmation_invalid"
  | "account_deletion_reauthentication_required"
  | "account_deletion_assurance_insufficient"
  | "account_deletion_idempotency_conflict"
  | "account_deletion_retry_required"
  | "account_deletion_status_invalid";

export class AccountDataRightsError extends Error {
  constructor(
    readonly statusCode: 400 | 401 | 403 | 409 | 410 | 413 | 503,
    readonly code: AccountDataRightsErrorCode
  ) {
    super(code);
    this.name = "AccountDataRightsError";
  }
}
