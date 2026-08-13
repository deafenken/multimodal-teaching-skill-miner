export type SafeguardingCaseStatus = "open" | "acknowledged" | "closed";
export type SafeguardingDeliveryStatus =
  | "pending"
  | "overdue"
  | "acknowledged"
  | "escalation_unavailable";

export interface SafeguardingStaffCase {
  case_id: string;
  version: number;
  status: SafeguardingCaseStatus;
  scope_sha256: string;
  category: string;
  severity: "elevated" | "high" | "urgent";
  observed_at_utc: string;
  content_sha256: string;
  created_at_utc: string;
  updated_at_utc: string;
  delivery_id: string | null;
  delivery_status: SafeguardingDeliveryStatus;
  sla_due_at_utc: string | null;
  overdue_recorded_at_utc: string | null;
  acknowledged_at_utc: string | null;
  raw_learner_text_exposed: false;
}

export interface SafeguardingStaffListResponse {
  schema: "teaching_skill_miner.dashboard_safeguarding_staff_list.v1";
  cases: SafeguardingStaffCase[];
  raw_learner_text_exposed: false;
}

export interface SafeguardingStaffMutationResponse {
  schema: "teaching_skill_miner.dashboard_safeguarding_mutation.v1";
  operation:
    | "case.acknowledged"
    | "case.closed"
    | "escalation.overdue"
    | "escalation.acknowledged";
  case: SafeguardingStaffCase;
  raw_learner_text_exposed: false;
}

const ROUTE_LOCATOR = /^sgr1_k[1-9][0-9]{0,8}_[A-Za-z0-9_-]{54,1800}$/;
const CASE_ID = /^sgc_[0-9a-f]{24}$/;
const DELIVERY_ID = /^sge_[0-9a-f]{24}$/;
const DIGEST = /^[0-9a-f]{64}$/;
const SAFE_CATEGORY = /^[a-z][a-z0-9_]{1,63}$/;
const UTC_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$/;
const CASE_KEYS = new Set([
  "case_id",
  "version",
  "status",
  "scope_sha256",
  "category",
  "severity",
  "observed_at_utc",
  "content_sha256",
  "created_at_utc",
  "updated_at_utc",
  "delivery_id",
  "delivery_status",
  "sla_due_at_utc",
  "overdue_recorded_at_utc",
  "acknowledged_at_utc",
  "raw_learner_text_exposed",
]);

function exactObject(value: unknown, keys: ReadonlySet<string>): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("安全保障工作流返回了无效响应。");
  }
  const row = value as Record<string, unknown>;
  const actual = Object.keys(row);
  if (actual.length !== keys.size || actual.some((key) => !keys.has(key))) {
    throw new Error("安全保障工作流返回了越界字段。");
  }
  return row;
}

function timestamp(value: unknown, nullable = false): value is string | null {
  return (nullable && value === null)
    || (typeof value === "string" && UTC_TIMESTAMP.test(value) && Number.isFinite(Date.parse(value)));
}

export function validSafeguardingRouteLocator(value: string): boolean {
  return value.length <= 2_048 && ROUTE_LOCATOR.test(value);
}

export function newSafeguardingIdempotencyKey(operation: string): string {
  if (!/^[a-z][a-z-]{2,39}$/.test(operation)) {
    throw new Error("安全保障操作名称无效。");
  }
  return `console-safeguarding-${operation}-${crypto.randomUUID()}`;
}

export function safeguardingStaffRequestBody(
  routeLocator: string,
  fields: Record<string, unknown> = {},
): Record<string, unknown> {
  if (!validSafeguardingRouteLocator(routeLocator)) {
    throw new Error("请输入安全保障系统签发的有效路由凭据。");
  }
  return {
    ...fields,
    safeguarding_route_locator: routeLocator,
  };
}

export function parseSafeguardingStaffCase(value: unknown): SafeguardingStaffCase {
  const row = exactObject(value, CASE_KEYS);
  if (
    typeof row.case_id !== "string" || !CASE_ID.test(row.case_id)
    || !Number.isInteger(row.version) || Number(row.version) < 1
    || !new Set(["open", "acknowledged", "closed"]).has(String(row.status))
    || typeof row.scope_sha256 !== "string" || !DIGEST.test(row.scope_sha256)
    || typeof row.category !== "string" || !SAFE_CATEGORY.test(row.category)
    || !new Set(["elevated", "high", "urgent"]).has(String(row.severity))
    || !timestamp(row.observed_at_utc)
    || typeof row.content_sha256 !== "string" || !DIGEST.test(row.content_sha256)
    || !timestamp(row.created_at_utc)
    || !timestamp(row.updated_at_utc)
    || (row.delivery_id !== null && (typeof row.delivery_id !== "string" || !DELIVERY_ID.test(row.delivery_id)))
    || !new Set(["pending", "overdue", "acknowledged", "escalation_unavailable"]).has(String(row.delivery_status))
    || !timestamp(row.sla_due_at_utc, true)
    || !timestamp(row.overdue_recorded_at_utc, true)
    || !timestamp(row.acknowledged_at_utc, true)
    || row.raw_learner_text_exposed !== false
  ) {
    throw new Error("安全保障工作流返回了无效案例投影。");
  }
  return row as unknown as SafeguardingStaffCase;
}

export function parseSafeguardingStaffList(value: unknown): SafeguardingStaffListResponse {
  const row = exactObject(value, new Set(["schema", "cases", "raw_learner_text_exposed"]));
  if (
    row.schema !== "teaching_skill_miner.dashboard_safeguarding_staff_list.v1"
    || !Array.isArray(row.cases)
    || row.cases.length > 10_000
    || row.raw_learner_text_exposed !== false
  ) {
    throw new Error("安全保障工作流返回了无效列表。");
  }
  return {
    schema: row.schema,
    cases: row.cases.map(parseSafeguardingStaffCase),
    raw_learner_text_exposed: false,
  };
}

export function parseSafeguardingStaffMutation(value: unknown): SafeguardingStaffMutationResponse {
  const row = exactObject(value, new Set(["schema", "operation", "case", "raw_learner_text_exposed"]));
  if (
    row.schema !== "teaching_skill_miner.dashboard_safeguarding_mutation.v1"
    || !new Set([
      "case.acknowledged",
      "case.closed",
      "escalation.overdue",
      "escalation.acknowledged",
    ]).has(String(row.operation))
    || row.raw_learner_text_exposed !== false
  ) {
    throw new Error("安全保障工作流返回了无效变更收据。");
  }
  return {
    schema: row.schema,
    operation: row.operation as SafeguardingStaffMutationResponse["operation"],
    case: parseSafeguardingStaffCase(row.case),
    raw_learner_text_exposed: false,
  };
}
