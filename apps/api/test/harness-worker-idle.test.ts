import "reflect-metadata";

import assert from "node:assert/strict";
import {EventEmitter} from "node:events";
import {mkdtemp, readFile, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import test from "node:test";

import type {ChildProcessWithoutNullStreams} from "node:child_process";

import type {AppConfigService} from "../src/config/app-config.service";
import {
  HarnessGatewayError,
  HarnessWorkerPoolService,
} from "../src/harness/harness-worker-pool.service";

class FakeChild extends EventEmitter {
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;
  readonly signals: NodeJS.Signals[] = [];

  constructor(private readonly exitOnSignal = true) {
    super();
  }

  kill(signal: NodeJS.Signals = "SIGTERM"): boolean {
    this.signals.push(signal);
    if (!this.exitOnSignal) return false;
    this.signalCode = signal;
    queueMicrotask(() => {
      this.exitCode = 0;
      this.emit("exit", 0, signal);
    });
    return true;
  }
}

interface FakeRecord {
  identityKey: string;
  child: ChildProcessWithoutNullStreams;
  activeRequests: number;
  lastActivityMs: number;
  stopping: boolean;
  capability: string;
  port: number;
  requestNamespaceKey: Buffer;
  teacherAuthorityKey: Buffer;
  scopeId: string;
  scopeKeyVersion: string;
}

interface TestPool {
  workers: Map<string, FakeRecord>;
  retiring: Set<FakeRecord>;
  evictExpiredIdleWorker(): Promise<void>;
  handleWorkerExit(key: string, child: ChildProcessWithoutNullStreams): void;
  physicalWorkerCount(): number;
}

function record(
  index: number,
  child = new FakeChild(),
  options: Partial<Pick<FakeRecord, "activeRequests" | "lastActivityMs" | "stopping">> = {},
): FakeRecord {
  const identityKey = `k1:scope_${String(index).padStart(48, "0")}`;
  return {
    identityKey,
    child: child as unknown as ChildProcessWithoutNullStreams,
    activeRequests: options.activeRequests ?? 0,
    lastActivityMs: options.lastActivityMs ?? index,
    stopping: options.stopping ?? false,
    capability: `capability-${index}`,
    port: 10_000 + index,
    requestNamespaceKey: Buffer.alloc(32, index),
    teacherAuthorityKey: Buffer.alloc(32, index + 1),
    scopeId: `scope_${String(index).padStart(48, "0")}`,
    scopeKeyVersion: "k1",
  };
}

function pool(maxWorkers = 16, idleTimeoutMs = 1_000) {
  const config = {
    nodeEnv: "test",
    harnessGatewayEnabled: true,
    harnessScopeKeyVersion: "k1",
    harnessWorkerBackend: "deterministic",
    harnessProviderApiKeyFile: undefined,
    harnessMaxWorkers: maxWorkers,
    harnessWorkerIdleTimeoutMs: idleTimeoutMs,
    harnessWorkerShutdownMs: 10,
  } as unknown as AppConfigService;
  const service = new HarnessWorkerPoolService(config);
  return {service, internals: service as unknown as TestPool};
}

test("a full 16-worker pool evicts the expired LRU and preserves its durable scope root", async () => {
  const {service, internals} = pool();
  service.setClockForTesting(() => 20_000);
  const privateRoot = await mkdtemp(join(tmpdir(), "teachlab-idle-scope-"));
  const marker = join(privateRoot, "durable-session-marker");
  await writeFile(marker, "must survive worker eviction", "utf8");
  try {
    const records = Array.from({length: 16}, (_, index) => record(index));
    for (const worker of records) internals.workers.set(worker.identityKey, worker);
    await internals.evictExpiredIdleWorker();
    assert.equal(internals.workers.size, 15);
    assert.equal(internals.workers.has(records[0]!.identityKey), false);
    assert.deepEqual(
      (records[0]!.child as unknown as FakeChild).signals,
      ["SIGTERM"],
    );
    const replacement = record(16, new FakeChild(), {lastActivityMs: 20_000});
    internals.workers.set(replacement.identityKey, replacement);
    assert.equal(internals.physicalWorkerCount(), 16);
    assert.equal(await readFile(marker, "utf8"), "must survive worker eviction");
    const status = service.status();
    assert.equal(status.idleTimeoutMs, 1_000);
    assert.equal(status.idlePolicy, "capacity_triggered_expired_lru_only");
    assert.equal(status.activeOrStartingWorkersEvictable, false);
    assert.equal(status.durableScopeDataDeletedOnEviction, false);
  } finally {
    await rm(privateRoot, {recursive: true, force: true});
  }
});

test("fake clock expiry is required and an active SSE worker is never selected", async () => {
  const {service, internals} = pool(2, 1_000);
  let now = 5_000;
  service.setClockForTesting(() => now);
  const active = record(1, new FakeChild(), {
    activeRequests: 1,
    lastActivityMs: 0,
  });
  const idle = record(2, new FakeChild(), {lastActivityMs: now});
  internals.workers.set(active.identityKey, active);
  internals.workers.set(idle.identityKey, idle);

  await assert.rejects(
    internals.evictExpiredIdleWorker(),
    (error: unknown) =>
      error instanceof HarnessGatewayError
      && error.code === "harness_capacity_exhausted",
  );
  now += 1_001;
  await internals.evictExpiredIdleWorker();
  assert.equal(internals.workers.has(active.identityKey), true);
  assert.equal((active.child as unknown as FakeChild).signals.length, 0);
  assert.equal(internals.workers.has(idle.identityKey), false);
});

test("an uncertain stop remains charged to capacity and a late old exit cannot delete its replacement", async () => {
  const {service, internals} = pool(1, 1_000);
  service.setClockForTesting(() => 10_000);
  const oldChild = new FakeChild(false);
  const old = record(1, oldChild, {lastActivityMs: 0});
  internals.workers.set(old.identityKey, old);
  await assert.rejects(
    internals.evictExpiredIdleWorker(),
    (error: unknown) =>
      error instanceof HarnessGatewayError
      && error.code === "harness_capacity_exhausted",
  );
  assert.equal(internals.workers.size, 0);
  assert.equal(internals.retiring.has(old), true);
  assert.equal(internals.physicalWorkerCount(), 1);

  const replacement = record(1, new FakeChild(), {lastActivityMs: 10_000});
  internals.workers.set(replacement.identityKey, replacement);
  internals.handleWorkerExit(old.identityKey, old.child);
  assert.equal(internals.workers.get(replacement.identityKey), replacement);
  assert.equal(internals.retiring.has(old), false);
});
