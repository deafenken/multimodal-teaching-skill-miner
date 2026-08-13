import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {mkdtemp, mkdir, open, rm, stat} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import test from "node:test";

import {HarnessAccountRuntimeCoordinator} from "../src/account/harness-account-runtime-coordinator";
import type {AppConfigService} from "../src/config/app-config.service";
import type {HarnessWorkerPoolService} from "../src/harness/harness-worker-pool.service";

test("permanent account purge is not blocked by the smaller export byte limit", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "teachlab-account-purge-"));
  t.after(async () => { await rm(root, {recursive: true, force: true}); });
  const operationId = `adel_${"a".repeat(32)}`;
  const operationDigest = createHash("sha256").update(operationId, "utf8").digest("hex");
  const scopeRoot = join(
    root,
    ".account-deletion-quarantine-v1",
    operationDigest,
    "k1-scope_fixture"
  );
  await mkdir(scopeRoot, {recursive: true, mode: 0o700});
  const sparseFile = await open(join(scopeRoot, "large-private-store.bin"), "w", 0o600);
  const byteLength = 513 * 1024 * 1024;
  try {
    await sparseFile.truncate(byteLength);
  } finally {
    await sparseFile.close();
  }
  const coordinator = new HarnessAccountRuntimeCoordinator(
    {} as HarnessWorkerPoolService,
    {harnessWorkerRoot: root} as AppConfigService
  );

  const purged = await coordinator.purgeQuarantine(
    {tenantId: "tenant", ownerId: "owner"},
    {active: "b".repeat(64), candidates: ["b".repeat(64)]},
    operationId
  );

  assert.deepEqual(purged, {
    workerFiles: 1,
    workerBytes: byteLength,
    workerRoots: 1
  });
  assert.deepEqual(await coordinator.purgeQuarantine(
    {tenantId: "tenant", ownerId: "owner"},
    {active: "b".repeat(64), candidates: ["b".repeat(64)]},
    operationId
  ), purged);
  await assert.rejects(stat(scopeRoot), {code: "ENOENT"});
});
