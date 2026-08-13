import {createHash} from "node:crypto";

export const REMOTE_PROCESSING_POLICY_STATES = [
  "denied",
  "adult_roster_authorized",
  "minor_guardian_verified",
  "minor_school_policy_verified"
] as const;

export type RemoteProcessingPolicyState =
  typeof REMOTE_PROCESSING_POLICY_STATES[number];

export interface RemoteSubjectPolicy {
  policy_id: string;
  policy_version: string;
  policy_source: "organization_oidc_or_roster_policy";
  likely_minor: boolean;
  guardian_or_school_policy:
    | "not_required"
    | "verified_guardian"
    | "verified_school_policy";
  remote_processing_eligible: boolean;
}

export interface RemoteProviderPolicy {
  policy_id: string;
  policy_version: string;
  policy_source:
    "deployment_operator_asserted_external_terms_not_repository_verified";
  processing_region: string;
  provider_retention_days: number;
  deletion_status:
    | "outside_service_control_subject_to_provider_policy"
    | "provider_documents_zero_retention";
  documentation_url: string | null;
}

export function remoteSubjectPolicyFor(
  state: RemoteProcessingPolicyState,
  policyId: string,
  policyVersion: string
): RemoteSubjectPolicy {
  const base = {
    policy_id: policyId,
    policy_version: policyVersion,
    policy_source: "organization_oidc_or_roster_policy" as const
  };
  if (state === "adult_roster_authorized") {
    return {
      ...base,
      likely_minor: false,
      guardian_or_school_policy: "not_required",
      remote_processing_eligible: true
    };
  }
  if (state === "minor_guardian_verified") {
    return {
      ...base,
      likely_minor: true,
      guardian_or_school_policy: "verified_guardian",
      remote_processing_eligible: true
    };
  }
  if (state === "minor_school_policy_verified") {
    return {
      ...base,
      likely_minor: true,
      guardian_or_school_policy: "verified_school_policy",
      remote_processing_eligible: true
    };
  }
  // Missing, unknown, stale, or explicitly denied claims share the same
  // conservative projection. No age inference is made public.
  return {
    ...base,
    likely_minor: true,
    guardian_or_school_policy: "not_required",
    remote_processing_eligible: false
  };
}

function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`)
    .join(",")}}`;
}

export function remotePolicySha256(policy: RemoteSubjectPolicy): string {
  return createHash("sha256").update(canonical(policy), "utf8").digest("hex");
}

export function isRemoteSubjectPolicy(value: unknown): value is RemoteSubjectPolicy {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const row = value as Record<string, unknown>;
  if (
    Object.keys(row).sort().join(",") !== [
      "guardian_or_school_policy",
      "likely_minor",
      "policy_id",
      "policy_source",
      "policy_version",
      "remote_processing_eligible"
    ].join(",")
    || typeof row.policy_id !== "string"
    || typeof row.policy_version !== "string"
    || !/^[A-Za-z0-9][A-Za-z0-9_.:-]{1,119}$/.test(row.policy_id)
    || !/^[A-Za-z0-9][A-Za-z0-9_.:-]{1,119}$/.test(row.policy_version)
    || row.policy_source !== "organization_oidc_or_roster_policy"
    || typeof row.likely_minor !== "boolean"
    || typeof row.remote_processing_eligible !== "boolean"
    || !new Set(["not_required", "verified_guardian", "verified_school_policy"])
      .has(String(row.guardian_or_school_policy))
  ) return false;
  if (row.remote_processing_eligible === false) {
    return row.likely_minor === true && row.guardian_or_school_policy === "not_required";
  }
  return row.likely_minor === false
    ? row.guardian_or_school_policy === "not_required"
    : new Set(["verified_guardian", "verified_school_policy"])
      .has(String(row.guardian_or_school_policy));
}
