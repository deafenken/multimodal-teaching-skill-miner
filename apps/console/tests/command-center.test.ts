import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

import {commandCenterActions, filterCommandCenterActions, relativeTaskTime} from "../lib/command-center.ts";

test("command center exposes stable navigation and mode-scoped stop actions", () => {
  const idle = commandCenterActions({chatRunning: false, teachRunning: false});
  assert.deepEqual(idle.map((item) => item.id), ["new_current", "open_chat", "open_teach", "open_syllabi", "show_tasks", "open_settings"]);
  const running = commandCenterActions({chatRunning: true, teachRunning: true});
  assert.equal(running.some((item) => item.id === "stop_chat"), true);
  assert.equal(running.some((item) => item.id === "stop_teach"), true);
});

test("command search normalizes multilingual text without fuzzy surprises", () => {
  const actions = commandCenterActions({chatRunning: true, teachRunning: false});
  assert.deepEqual(filterCommandCenterActions(actions, " 大纲 ").map((item) => item.id), ["open_syllabi"]);
  assert.deepEqual(filterCommandCenterActions(actions, "CANCEL").map((item) => item.id), ["stop_chat"]);
  assert.equal(filterCommandCenterActions(actions, "不存在").length, 0);
});

test("relative task time is deterministic and rejects malformed timestamps", () => {
  const now = Date.parse("2026-08-12T12:00:00Z");
  assert.equal(relativeTaskTime("2026-08-12T11:59:30Z", now), "30 秒前");
  assert.equal(relativeTaskTime("2026-08-12T10:00:00Z", now), "2 小时前");
  assert.equal(relativeTaskTime("not-a-time", now), "时间未知");
});

test("task center is content-free, accessible, and uses durable task commands", () => {
  const source = readFileSync(new URL("../components/workbench/command-center.tsx", import.meta.url), "utf8");
  for (const token of [
    "@radix-ui/react-dialog",
    'aria-describedby="teachlab-command-center-description"',
    "listBackgroundTasks",
    "resumeBackgroundTask",
    "cancelHarnessRun",
    "content",
    "显式“停止”才取消任务",
  ]) assert.ok(source.includes(token), `missing command-center contract: ${token}`);
  assert.equal(source.includes("payload.result"), false);
  assert.equal(source.includes("learner_text"), false);
});
