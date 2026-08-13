import assert from "node:assert/strict";
import test from "node:test";

import {
  parseSafeguardingStaffList,
  parseSafeguardingStaffMutation,
  safeguardingStaffRequestBody,
  validSafeguardingRouteLocator,
} from "../lib/safeguarding-staff.ts";

const locator = `sgr1_k1_${"A".repeat(80)}`;
const item = {
  case_id: `sgc_${"a".repeat(24)}`,
  version: 3,
  status: "acknowledged",
  scope_sha256: "b".repeat(64),
  category: "self_harm",
  severity: "urgent",
  observed_at_utc: "2026-08-12T01:02:03Z",
  content_sha256: "c".repeat(64),
  created_at_utc: "2026-08-12T01:02:03Z",
  updated_at_utc: "2026-08-12T01:03:03Z",
  delivery_id: `sge_${"d".repeat(24)}`,
  delivery_status: "acknowledged",
  sla_due_at_utc: "2026-08-12T01:02:33Z",
  overdue_recorded_at_utc: null,
  acknowledged_at_utc: "2026-08-12T01:03:03Z",
  raw_learner_text_exposed: false,
} as const;

test("staff client accepts only opaque locators and content-free exact projections", () => {
  assert.equal(validSafeguardingRouteLocator(locator), true);
  assert.equal(validSafeguardingRouteLocator("tenant:learner"), false);
  assert.deepEqual(safeguardingStaffRequestBody(locator, {status: "open"}), {
    status: "open",
    safeguarding_route_locator: locator,
  });
  const listed = parseSafeguardingStaffList({
    schema: "teaching_skill_miner.dashboard_safeguarding_staff_list.v1",
    cases: [item],
    raw_learner_text_exposed: false,
  });
  assert.deepEqual(listed.cases, [item]);
  assert.equal(JSON.stringify(listed).includes("learner_text"), true);
  assert.equal(JSON.stringify(listed).includes("我现在"), false);
});

test("staff client rejects raw text, extra fields, bad digests and dishonest flags", () => {
  for (const altered of [
    {...item, raw_learner_text: "must never render"},
    {...item, content_sha256: "not-a-digest"},
    {...item, raw_learner_text_exposed: true},
  ]) {
    assert.throws(() => parseSafeguardingStaffList({
      schema: "teaching_skill_miner.dashboard_safeguarding_staff_list.v1",
      cases: [altered],
      raw_learner_text_exposed: false,
    }));
  }
});

test("staff mutation parser binds the exact operation and CAS projection", () => {
  const parsed = parseSafeguardingStaffMutation({
    schema: "teaching_skill_miner.dashboard_safeguarding_mutation.v1",
    operation: "case.closed",
    case: {...item, version: 4, status: "closed"},
    raw_learner_text_exposed: false,
  });
  assert.equal(parsed.operation, "case.closed");
  assert.equal(parsed.case.version, 4);
  assert.throws(() => parseSafeguardingStaffMutation({
    schema: "teaching_skill_miner.dashboard_safeguarding_mutation.v1",
    operation: "case.deleted",
    case: item,
    raw_learner_text_exposed: false,
  }));
});
