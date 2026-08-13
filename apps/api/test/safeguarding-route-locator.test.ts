import assert from "node:assert/strict";
import test from "node:test";

import {
  issueSafeguardingRouteLocator,
  resolveSafeguardingRouteLocator
} from "../src/harness/safeguarding-route-locator";

const OLD = {version: "k1", secret: "old-account-scope-key-with-more-than-32-bytes"};
const ACTIVE = {version: "k2", secret: "new-account-scope-key-with-more-than-32-bytes"};
const SCOPE = {tenantId: "ot1_ns1_tenant-opaque", ownerId: "os1_ns1_owner-opaque"};

test("safeguarding route locators are opaque, tenant-bound, and rotation-safe", () => {
  const oldLocator = issueSafeguardingRouteLocator(SCOPE, OLD);
  const activeLocator = issueSafeguardingRouteLocator(SCOPE, ACTIVE);
  assert.equal(oldLocator.includes(SCOPE.tenantId), false);
  assert.equal(oldLocator.includes(SCOPE.ownerId), false);
  assert.deepEqual(
    resolveSafeguardingRouteLocator(oldLocator, [ACTIVE, OLD], SCOPE.tenantId),
    SCOPE
  );
  assert.deepEqual(
    resolveSafeguardingRouteLocator(activeLocator, [ACTIVE, OLD], SCOPE.tenantId),
    SCOPE
  );
  assert.throws(
    () => resolveSafeguardingRouteLocator(oldLocator, [ACTIVE], SCOPE.tenantId),
    /invalid/
  );
  assert.throws(
    () => resolveSafeguardingRouteLocator(oldLocator, [ACTIVE, OLD], "ot1_ns1_other"),
    /invalid/
  );
});

test("tampering cannot redirect a safeguarding mutation", () => {
  const locator = issueSafeguardingRouteLocator(SCOPE, ACTIVE);
  const replacement = locator.endsWith("A") ? "B" : "A";
  const tampered = locator.slice(0, -1) + replacement;
  assert.throws(
    () => resolveSafeguardingRouteLocator(tampered, [ACTIVE], SCOPE.tenantId),
    /invalid/
  );
});
