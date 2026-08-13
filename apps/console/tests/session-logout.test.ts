import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

import {
  boundedLogoutCleanup,
  settleRunCancellationsBeforeLogout,
} from "../lib/session-logout.ts";

const workbench = readFileSync(
  new URL("../components/workbench/workbench.tsx", import.meta.url),
  "utf8",
);
const sidebar = readFileSync(
  new URL("../components/workbench/session-sidebar.tsx", import.meta.url),
  "utf8",
);

test("logout waits for every cancellation settlement before session DELETE", async () => {
  const sequence: string[] = [];
  let releaseFirst: (() => void) | undefined;
  const first = new Promise<void>((resolve) => { releaseFirst = resolve; });
  const cancellation = settleRunCancellationsBeforeLogout(
    ["chat", "teach"],
    async (run) => {
      sequence.push(`cancel:${run}`);
      if (run === "chat") await first;
      else throw new Error("already terminal");
      sequence.push(`settled:${run}`);
    },
    1_000,
  ).then(() => sequence.push("delete"));
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(sequence, ["cancel:chat", "cancel:teach"]);
  releaseFirst?.();
  await cancellation;
  assert.deepEqual(sequence, ["cancel:chat", "cancel:teach", "settled:chat", "delete"]);
});

test("a hung cancellation is bounded and cannot prevent session revocation", async () => {
  const startedAt = Date.now();
  const result = await settleRunCancellationsBeforeLogout(
    ["hung"],
    () => new Promise<void>(() => undefined),
    10,
  );
  assert.equal(result, "timed_out");
  assert.ok(Date.now() - startedAt < 500);
});

test("post-revocation local cleanup is also bounded and reports incomplete cleanup", async () => {
  assert.equal(await boundedLogoutCleanup(async () => true, 50), true);
  assert.equal(await boundedLogoutCleanup(
    () => new Promise<boolean>(() => undefined),
    10,
  ), false);
});

test("Workbench exposes explicit same-origin logout and signed-out UI contracts", () => {
  for (const token of [
    "cancelAllModeRequests();",
    "await settleRunCancellationsBeforeLogout(",
    "await clearAuthenticatedHarnessSession();",
    "clearOfflineRuntimeForLogout,",
    "已安全退出 TeachLab",
  ]) assert.ok(workbench.includes(token), token);
  assert.ok(sidebar.includes("安全退出当前设备"));
  assert.ok(sidebar.includes('role="alert"'));
});
