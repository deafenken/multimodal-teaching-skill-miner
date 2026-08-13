import assert from "node:assert/strict";
import {test} from "node:test";

import {
  ResourceGovernor,
  ResourceGovernorError,
  type ResourceGovernorPolicy
} from "../src/operations/resource-governor";


const POLICY: ResourceGovernorPolicy = {
  anonymousCapacity: 2,
  anonymousRefillPerSecond: 1,
  authenticatedCapacity: 3,
  authenticatedRefillPerSecond: 2,
  streamCapacity: 1,
  streamRefillPerSecond: 0.5,
  maxAnonymousConcurrent: 2,
  maxAuthenticatedConcurrent: 2,
  maxStreamConcurrent: 1,
  maxAnonymousConcurrentPerIdentity: 2,
  maxAuthenticatedConcurrentPerIdentity: 2,
  maxStreamConcurrentPerIdentity: 1,
  maximumBuckets: 16,
  idleBucketTtlMs: 1_000
};


test("token buckets are isolated by hashed identity and refill deterministically", () => {
  let now = 10_000;
  const governor = new ResourceGovernor(POLICY, () => now);
  governor.acquire("anonymous", "socket:127.0.0.1").release();
  governor.acquire("anonymous", "socket:127.0.0.1").release();
  assert.throws(
    () => governor.acquire("anonymous", "socket:127.0.0.1"),
    (error: unknown) =>
      error instanceof ResourceGovernorError &&
      error.code === "rate_limit_exceeded" &&
      error.retryAfterSeconds === 1
  );
  // A second identity is independent and raw identity is not projected.
  governor.acquire("anonymous", "socket:127.0.0.2").release();
  let snapshot = governor.snapshot();
  assert.equal(snapshot.buckets, 2);
  assert.equal(JSON.stringify(snapshot).includes("127.0.0"), false);
  now += 1_000;
  governor.acquire("anonymous", "socket:127.0.0.1").release();
  snapshot = governor.snapshot();
  assert.equal(snapshot.rejectedRate, 1);
  assert.equal(snapshot.distributedQuotaAuthority, false);
  assert.equal(snapshot.deploymentBoundary, "single_replica_process_local");
});


test("stream concurrency cannot be bypassed by releasing twice", () => {
  const governor = new ResourceGovernor({...POLICY, streamCapacity: 2}, () => 20_000);
  const lease = governor.acquire("stream", "tenant:opaque");
  assert.throws(
    () => governor.acquire("stream", "tenant:opaque"),
    (error: unknown) =>
      error instanceof ResourceGovernorError &&
      error.code === "identity_concurrency_limit_exceeded"
  );
  lease.release();
  lease.release();
  const next = governor.acquire("stream", "tenant:opaque");
  next.release();
  // Both permitted starts consumed a token, even though releases were idempotent.
  assert.throws(
    () => governor.acquire("stream", "tenant:opaque"),
    (error: unknown) =>
      error instanceof ResourceGovernorError && error.code === "rate_limit_exceeded"
  );
  assert.equal(governor.snapshot().activeStream, 0);
});

test("one identity cannot consume the global stream pool", () => {
  const governor = new ResourceGovernor({
    ...POLICY,
    streamCapacity: 20,
    maxStreamConcurrent: 4,
    maxStreamConcurrentPerIdentity: 2,
  }, () => 25_000);
  const a1 = governor.acquire("stream", "tenant-a");
  const a2 = governor.acquire("stream", "tenant-a");
  assert.throws(
    () => governor.acquire("stream", "tenant-a"),
    (error: unknown) => error instanceof ResourceGovernorError
      && error.code === "identity_concurrency_limit_exceeded",
  );
  const b1 = governor.acquire("stream", "tenant-b");
  const b2 = governor.acquire("stream", "tenant-b");
  assert.throws(
    () => governor.acquire("stream", "tenant-c"),
    (error: unknown) => error instanceof ResourceGovernorError
      && error.code === "global_concurrency_limit_exceeded",
  );
  assert.equal(governor.snapshot().rejectedIdentityConcurrency, 1);
  assert.equal(governor.snapshot().rejectedGlobalConcurrency, 1);
  for (const lease of [a1, a2, b1, b2]) lease.release();
});


test("idle buckets are bounded and capacity exhaustion fails closed", () => {
  let now = 30_000;
  const policy = {...POLICY, maximumBuckets: 16};
  const governor = new ResourceGovernor(policy, () => now);
  for (let index = 0; index < 16; index += 1) {
    governor.acquire("authenticated", `principal:${index}`).release();
  }
  assert.throws(
    () => governor.acquire("authenticated", "principal:overflow"),
    (error: unknown) =>
      error instanceof ResourceGovernorError &&
      error.code === "governor_capacity_exhausted"
  );
  now += 1_000;
  governor.acquire("authenticated", "principal:after-sweep").release();
  assert.equal(governor.snapshot().buckets, 1);
});


test("policy and identity validation reject unsafe inputs", () => {
  assert.throws(
    () => new ResourceGovernor({...POLICY, maximumBuckets: 2}),
    /maximumBuckets/
  );
  const governor = new ResourceGovernor(POLICY);
  assert.throws(() => governor.acquire("anonymous", ""), /identity/);
  assert.throws(() => governor.acquire("anonymous", "a\nforged"), /identity/);
  assert.throws(() => governor.acquire("authenticated", "ok", 0), /cost/);
});
