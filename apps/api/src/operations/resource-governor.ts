import {createHash} from "node:crypto";

export type ResourceLimitClass = "anonymous" | "authenticated" | "stream";

export interface ResourceGovernorPolicy {
  anonymousCapacity: number;
  anonymousRefillPerSecond: number;
  authenticatedCapacity: number;
  authenticatedRefillPerSecond: number;
  streamCapacity: number;
  streamRefillPerSecond: number;
  maxAnonymousConcurrent: number;
  maxAuthenticatedConcurrent: number;
  maxStreamConcurrent: number;
  maxAnonymousConcurrentPerIdentity: number;
  maxAuthenticatedConcurrentPerIdentity: number;
  maxStreamConcurrentPerIdentity: number;
  maximumBuckets: number;
  idleBucketTtlMs: number;
}

export interface ResourceLease {
  readonly keySha256: string;
  readonly limitClass: ResourceLimitClass;
  release(): void;
}

export interface ResourceGovernorSnapshot {
  buckets: number;
  activeAnonymous: number;
  activeAuthenticated: number;
  activeStream: number;
  accepted: number;
  rejectedRate: number;
  rejectedConcurrency: number;
  rejectedIdentityConcurrency: number;
  rejectedGlobalConcurrency: number;
  rejectedCapacity: number;
  policy: ResourceGovernorPolicy;
  distributedQuotaAuthority: false;
  deploymentBoundary: "single_replica_process_local";
}

interface Bucket {
  keySha256: string;
  tokens: number;
  updatedAtMs: number;
  lastSeenAtMs: number;
  active: number;
}

export class ResourceGovernorError extends Error {
  constructor(
    readonly code:
      | "rate_limit_exceeded"
      | "identity_concurrency_limit_exceeded"
      | "global_concurrency_limit_exceeded"
      | "governor_capacity_exhausted",
    readonly retryAfterSeconds: number
  ) {
    super(code);
    this.name = "ResourceGovernorError";
  }
}

const CLASS_ORDER: ResourceLimitClass[] = ["anonymous", "authenticated", "stream"];

function positive(name: string, value: number, minimum: number, maximum: number): number {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${name} is outside its safe range`);
  }
  return value;
}

function canonicalKey(limitClass: ResourceLimitClass, identity: string): string {
  if (
    !CLASS_ORDER.includes(limitClass) ||
    typeof identity !== "string" ||
    !identity ||
    identity.length > 512 ||
    /[\u0000\r\n]/.test(identity)
  ) {
    throw new Error("resource governor identity is invalid");
  }
  return createHash("sha256")
    .update(`teachlab-resource-governor-v1\0${limitClass}\0`, "utf8")
    .update(identity, "utf8")
    .digest("hex");
}

export class ResourceGovernor {
  private readonly buckets = new Map<string, Bucket>();
  private readonly active: Record<ResourceLimitClass, number> = {
    anonymous: 0,
    authenticated: 0,
    stream: 0
  };
  private accepted = 0;
  private rejectedRate = 0;
  private rejectedConcurrency = 0;
  private rejectedIdentityConcurrency = 0;
  private rejectedGlobalConcurrency = 0;
  private rejectedCapacity = 0;

  constructor(
    readonly policy: ResourceGovernorPolicy,
    private readonly clock: () => number = () => Date.now()
  ) {
    for (const [name, value, minimum, maximum] of [
      ["anonymousCapacity", policy.anonymousCapacity, 1, 10_000],
      ["anonymousRefillPerSecond", policy.anonymousRefillPerSecond, 0.01, 10_000],
      ["authenticatedCapacity", policy.authenticatedCapacity, 1, 10_000],
      ["authenticatedRefillPerSecond", policy.authenticatedRefillPerSecond, 0.01, 10_000],
      ["streamCapacity", policy.streamCapacity, 1, 10_000],
      ["streamRefillPerSecond", policy.streamRefillPerSecond, 0.01, 10_000],
      ["maxAnonymousConcurrent", policy.maxAnonymousConcurrent, 1, 10_000],
      ["maxAuthenticatedConcurrent", policy.maxAuthenticatedConcurrent, 1, 10_000],
      ["maxStreamConcurrent", policy.maxStreamConcurrent, 1, 10_000],
      ["maxAnonymousConcurrentPerIdentity", policy.maxAnonymousConcurrentPerIdentity, 1, policy.maxAnonymousConcurrent],
      ["maxAuthenticatedConcurrentPerIdentity", policy.maxAuthenticatedConcurrentPerIdentity, 1, policy.maxAuthenticatedConcurrent],
      ["maxStreamConcurrentPerIdentity", policy.maxStreamConcurrentPerIdentity, 1, policy.maxStreamConcurrent],
      ["maximumBuckets", policy.maximumBuckets, 16, 1_000_000],
      ["idleBucketTtlMs", policy.idleBucketTtlMs, 1_000, 86_400_000]
    ] as Array<[string, number, number, number]>) {
      positive(name, value, minimum, maximum);
    }
  }

  acquire(
    limitClass: ResourceLimitClass,
    identity: string,
    cost = 1
  ): ResourceLease {
    positive("resource governor cost", cost, 0.01, 10_000);
    const now = this.now();
    this.sweep(now);
    const keySha256 = canonicalKey(limitClass, identity);
    const capacity = this.capacity(limitClass);
    const refill = this.refill(limitClass);
    let bucket = this.buckets.get(keySha256);
    if (!bucket) {
      if (this.buckets.size >= this.policy.maximumBuckets) {
        this.rejectedCapacity += 1;
        throw new ResourceGovernorError("governor_capacity_exhausted", 1);
      }
      bucket = {
        keySha256,
        tokens: capacity,
        updatedAtMs: now,
        lastSeenAtMs: now,
        active: 0
      };
      this.buckets.set(keySha256, bucket);
    }
    const elapsedSeconds = Math.max(0, now - bucket.updatedAtMs) / 1_000;
    bucket.tokens = Math.min(capacity, bucket.tokens + elapsedSeconds * refill);
    bucket.updatedAtMs = now;
    bucket.lastSeenAtMs = now;
    if (bucket.tokens + Number.EPSILON < cost) {
      this.rejectedRate += 1;
      const missing = cost - bucket.tokens;
      throw new ResourceGovernorError(
        "rate_limit_exceeded",
        Math.max(1, Math.ceil(missing / refill))
      );
    }
    if (bucket.active >= this.concurrentPerIdentity(limitClass)) {
      this.rejectedConcurrency += 1;
      this.rejectedIdentityConcurrency += 1;
      throw new ResourceGovernorError("identity_concurrency_limit_exceeded", 1);
    }
    if (this.active[limitClass] >= this.concurrentGlobal(limitClass)) {
      this.rejectedConcurrency += 1;
      this.rejectedGlobalConcurrency += 1;
      throw new ResourceGovernorError("global_concurrency_limit_exceeded", 1);
    }
    bucket.tokens -= cost;
    bucket.active += 1;
    this.active[limitClass] += 1;
    this.accepted += 1;
    let released = false;
    return {
      keySha256,
      limitClass,
      release: () => {
        if (released) return;
        released = true;
        bucket!.active = Math.max(0, bucket!.active - 1);
        bucket!.lastSeenAtMs = Math.max(bucket!.lastSeenAtMs, this.now());
        this.active[limitClass] = Math.max(0, this.active[limitClass] - 1);
      }
    };
  }

  snapshot(): ResourceGovernorSnapshot {
    this.sweep(this.now());
    return {
      buckets: this.buckets.size,
      activeAnonymous: this.active.anonymous,
      activeAuthenticated: this.active.authenticated,
      activeStream: this.active.stream,
      accepted: this.accepted,
      rejectedRate: this.rejectedRate,
      rejectedConcurrency: this.rejectedConcurrency,
      rejectedIdentityConcurrency: this.rejectedIdentityConcurrency,
      rejectedGlobalConcurrency: this.rejectedGlobalConcurrency,
      rejectedCapacity: this.rejectedCapacity,
      policy: {...this.policy},
      distributedQuotaAuthority: false,
      deploymentBoundary: "single_replica_process_local"
    };
  }

  private now(): number {
    const value = this.clock();
    if (!Number.isFinite(value) || value < 0) throw new Error("resource governor clock failed");
    return value;
  }

  private sweep(now: number): void {
    for (const [key, bucket] of this.buckets) {
      if (
        bucket.active === 0 &&
        now - bucket.lastSeenAtMs >= this.policy.idleBucketTtlMs
      ) {
        this.buckets.delete(key);
      }
    }
  }

  private capacity(limitClass: ResourceLimitClass): number {
    return limitClass === "anonymous"
      ? this.policy.anonymousCapacity
      : limitClass === "stream"
        ? this.policy.streamCapacity
        : this.policy.authenticatedCapacity;
  }

  private refill(limitClass: ResourceLimitClass): number {
    return limitClass === "anonymous"
      ? this.policy.anonymousRefillPerSecond
      : limitClass === "stream"
        ? this.policy.streamRefillPerSecond
        : this.policy.authenticatedRefillPerSecond;
  }

  private concurrentGlobal(limitClass: ResourceLimitClass): number {
    return limitClass === "anonymous"
      ? this.policy.maxAnonymousConcurrent
      : limitClass === "stream"
        ? this.policy.maxStreamConcurrent
        : this.policy.maxAuthenticatedConcurrent;
  }

  private concurrentPerIdentity(limitClass: ResourceLimitClass): number {
    return limitClass === "anonymous"
      ? this.policy.maxAnonymousConcurrentPerIdentity
      : limitClass === "stream"
        ? this.policy.maxStreamConcurrentPerIdentity
        : this.policy.maxAuthenticatedConcurrentPerIdentity;
  }
}

export const DEFAULT_RESOURCE_GOVERNOR_POLICY: ResourceGovernorPolicy = {
  anonymousCapacity: 300,
  anonymousRefillPerSecond: 5,
  authenticatedCapacity: 600,
  authenticatedRefillPerSecond: 10,
  streamCapacity: 30,
  streamRefillPerSecond: 0.5,
  maxAnonymousConcurrent: 8,
  maxAuthenticatedConcurrent: 64,
  maxStreamConcurrent: 16,
  maxAnonymousConcurrentPerIdentity: 4,
  maxAuthenticatedConcurrentPerIdentity: 8,
  maxStreamConcurrentPerIdentity: 3,
  maximumBuckets: 20_000,
  idleBucketTtlMs: 15 * 60 * 1_000
};
