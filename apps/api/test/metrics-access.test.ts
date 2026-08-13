import assert from "node:assert/strict";
import {test} from "node:test";

import {
  MetricsAccessConfigurationError,
  MetricsAccessService
} from "../src/observability/metrics-access.service";


test("metrics access accepts exact bearer token and supports explicit disable", () => {
  const token = "metrics-token-with-at-least-thirty-two-characters";
  const access = new MetricsAccessService();
  access.configureToken(token);
  assert.equal(access.enabled, true);
  assert.equal(access.authorize(`Bearer ${token}`), true);
  assert.equal(access.authorize(`Bearer ${token}x`), false);
  assert.equal(access.authorize([`Bearer ${token}`, "Bearer attacker"]), true);
  access.configureToken(undefined);
  assert.equal(access.enabled, false);
  assert.equal(access.authorize(`Bearer ${token}`), false);
});

test("metrics access rejects malformed values after private file resolution", () => {
  const access = new MetricsAccessService();
  assert.throws(() => access.configureToken("short"), MetricsAccessConfigurationError);
  assert.throws(
    () => access.configureToken("token with whitespace and enough characters"),
    MetricsAccessConfigurationError
  );
  assert.throws(
    () => access.configureToken("token/with/forbidden/characters/and/length"),
    MetricsAccessConfigurationError
  );
});
