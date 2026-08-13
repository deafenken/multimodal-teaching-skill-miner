import assert from "node:assert/strict";
import {once} from "node:events";
import {test} from "node:test";

import {
  PreparseBodyLimitError,
  PreparseCapacityError,
  PreparseRequestGovernor,
  RouteLimitedBodyStream,
} from "../src/operations/preparse-request-governor";
import {trustedProxyPolicy} from "../src/create-application";

test("pre-parse concurrency is bounded globally and per direct peer", () => {
  const governor = new PreparseRequestGovernor(3, 2);
  const first = governor.acquire("127.0.0.1");
  const second = governor.acquire("127.0.0.1");
  assert.throws(() => governor.acquire("127.0.0.1"), PreparseCapacityError);
  const third = governor.acquire("127.0.0.2");
  assert.throws(() => governor.acquire("127.0.0.3"), PreparseCapacityError);
  first.release();
  first.release();
  const replacement = governor.acquire("127.0.0.3");
  for (const lease of [second, third, replacement]) lease.release();
  assert.deepEqual(governor.snapshot(), {
    globalActive: 0,
    peerBuckets: 0,
    accepted: 4,
    rejectedCapacity: 2,
    rejectedBodyLimit: 0,
    completed: 4,
    aborted: 0,
  });
});

test("production-style shared reverse proxy can use the whole global body budget", () => {
  const governor = new PreparseRequestGovernor(32, 32);
  const leases = Array.from({length: 32}, () => governor.acquire("caddy-edge"));
  assert.throws(() => governor.acquire("caddy-edge"), PreparseCapacityError);
  for (const lease of leases) lease.release();
  assert.deepEqual(governor.snapshot(), {
    globalActive: 0,
    peerBuckets: 0,
    accepted: 32,
    rejectedCapacity: 1,
    rejectedBodyLimit: 0,
    completed: 32,
    aborted: 0,
  });
});

test("production trusts only private edge peers and development ignores forwarded identity", () => {
  assert.deepEqual(trustedProxyPolicy("production"), ["loopback", "linklocal", "uniquelocal"]);
  assert.equal(trustedProxyPolicy("development"), false);
  assert.equal(trustedProxyPolicy("test"), false);
});

test("chunked bodies are rejected by actual bytes before an oversized JSON value completes", async () => {
  let rejected = 0;
  const stream = new RouteLimitedBodyStream(8, () => { rejected += 1; });
  const errors: Error[] = [];
  stream.on("error", (error) => errors.push(error));
  stream.write(Buffer.from("1234"));
  stream.write(Buffer.from("56789"));
  await once(stream, "error");
  assert.equal(stream.receivedEncodedLength, 9);
  assert.ok(errors[0] instanceof PreparseBodyLimitError);
  assert.equal((errors[0] as PreparseBodyLimitError).statusCode, 413);
  assert.equal(rejected, 1);
});
