import assert from "node:assert/strict";
import test from "node:test";

import {boundedMessageWindow, newDurableOperation, offlineWorkspaceEnvelope, reconnectDelay, recoveryDecision, redactedDurableOperation, STREAM_MAX_CONNECTIONS, STREAM_RENDER_INTERVAL_MS, validateOfflineWorkspaceEnvelope, validateOfflineWorkspaceProject, WORKSPACE_CACHE_TTL_MS} from "../lib/offline-runtime.ts";

const ACCOUNT_SCOPE = `acs1_${"a".repeat(43)}`;

test("only an unregistered outbox request is eligible for resubmission", () => {
  const pending = newDurableOperation("request-1", "chat", {messages: []}, 100);
  assert.deepEqual(recoveryDecision(pending), {kind: "resubmit_unregistered", requestId: "request-1"});
  const registered = {...pending, state: "registered" as const, taskId: "task_123", taskVersion: 3};
  assert.deepEqual(recoveryDecision(registered), {kind: "observe_registered", taskId: "task_123"});
});

test("registered suspended work uses task resume and never replays its request payload", () => {
  const operation = {...newDurableOperation("request-2", "step", {private: "answer"}, 100), state: "registered" as const, taskId: "task_456", taskVersion: 4};
  assert.deepEqual(recoveryDecision(operation, {
    task_id: "task_456", version: 7, status: "suspended", resumable: true,
  }), {kind: "resume_registered", taskId: "task_456", taskVersion: 7});
});

test("registered and handoff projections erase learner payloads while unregistered recovery keeps them", () => {
  const sentinel = "PRIVATE_LEARNER_ANSWER_SENTINEL";
  const pending = newDurableOperation("request-sensitive", "step", {learner_response: sentinel}, 100);
  assert.equal(redactedDurableOperation(pending), pending);

  const registered = redactedDurableOperation({
    ...pending,
    state: "registered",
    taskId: "task_sensitive",
    taskVersion: 2,
  });
  assert.deepEqual(registered.payload, {});
  assert.equal(registered.payloadRedacted, true);
  assert.equal(JSON.stringify(registered).includes(sentinel), false);
  assert.deepEqual(recoveryDecision(registered), {
    kind: "observe_registered",
    taskId: "task_sensitive",
  });

  const handoff = redactedDurableOperation({...registered, state: "handoff"});
  assert.equal(JSON.stringify(handoff).includes(sentinel), false);
  assert.equal(redactedDurableOperation(handoff), handoff);
});

test("unsafe suspended and handoff tasks require explicit human handoff", () => {
  const operation = {...newDurableOperation("request-3", "start", {}, 100), state: "registered" as const, taskId: "task_789", taskVersion: 1};
  assert.equal(recoveryDecision(operation, {task_id: "task_789", version: 1, status: "handoff", resumable: false}).kind, "handoff");
  assert.equal(recoveryDecision(operation, {task_id: "task_789", version: 1, status: "suspended", resumable: false}).kind, "handoff");
});

test("reconnect delay is exponential, jitter-bounded, and capped", () => {
  assert.equal(reconnectDelay(0, 0), 500);
  assert.equal(reconnectDelay(1, 0), 1_000);
  assert.equal(reconnectDelay(6, 0), 30_000);
  assert.equal(reconnectDelay(20, 1), 31_000);
  const minimumRecoveryWindow = Array.from({length: STREAM_MAX_CONNECTIONS - 1}, (_, attempt) => reconnectDelay(attempt, 0))
    .reduce((total, delay) => total + delay, 0);
  assert.ok(minimumRecoveryWindow >= 180_000);
});

test("outbox operations have a seven-day bounded lifetime", () => {
  const operation = newDurableOperation("request-4", "chat", {}, 1_000);
  assert.equal(operation.expiresAt - operation.createdAt, 7 * 24 * 60 * 60 * 1_000);
});

test("a ten-thousand-message history has a deterministic 300-node initial render budget", () => {
  const source = Array.from({length: 10_000}, (_, index) => index);
  const window = boundedMessageWindow(source, 300);
  assert.equal(window.items.length, 300);
  assert.equal(window.hidden, 9_700);
  assert.equal(window.items[0], 9_700);
  assert.equal(window.items.at(-1), 9_999);
});

test("stream deltas are coalesced into a bounded render cadence", () => {
  assert.ok(STREAM_RENDER_INTERVAL_MS >= 32);
  assert.ok(STREAM_RENDER_INTERVAL_MS <= 50);
});

const cachedProject = {
  schema: "teaching_skill_miner.learning_project.v1",
  project_id: "project_123456789012345678901234",
  title: "动态规划",
  description: "离线复习",
  status: "active",
  pinned: true,
  created_at: "2026-08-12T00:00:00Z",
  updated_at: "2026-08-12T01:00:00Z",
  syllabus_ids: [],
  teaching_session_ids: [],
  resource_ids: [],
  notes: [],
  claim_boundary: {},
  chat_threads: [{
    thread_id: "chat_123456789012345678901234",
    title: "状态定义",
    created_at: "2026-08-12T00:00:00Z",
    updated_at: "2026-08-12T01:00:00Z",
    messages: [{
      message_id: "server_123456789012345678901234",
      role: "assistant",
      content: "状态是对子问题所需信息的最小表达。",
      status: "completed",
      created_at: "2026-08-12T01:00:00Z",
      web_search_used: false,
      sources: [],
    }],
  }],
};

test("offline project cache is bounded, terminal-only, and expires after seven days", () => {
  assert.equal(validateOfflineWorkspaceProject(cachedProject), true);
  const envelope = offlineWorkspaceEnvelope(cachedProject, 1_000, ACCOUNT_SCOPE);
  assert.ok(envelope);
  assert.equal(envelope.expiresAt - envelope.savedAt, WORKSPACE_CACHE_TTL_MS);
  assert.ok(validateOfflineWorkspaceEnvelope(envelope, 1_001, ACCOUNT_SCOPE));
  assert.equal(validateOfflineWorkspaceEnvelope(envelope, envelope.expiresAt, ACCOUNT_SCOPE), null);

  const partial = structuredClone(cachedProject);
  partial.chat_threads[0].messages[0].status = "running";
  assert.equal(validateOfflineWorkspaceProject(partial), false);
  const projected = offlineWorkspaceEnvelope(partial, 1_000, ACCOUNT_SCOPE);
  assert.ok(projected);
  assert.deepEqual((projected.project.chat_threads as typeof cachedProject.chat_threads)[0].messages, []);
});

test("offline cache envelope rejects identity, revision, and size equivocation", () => {
  const envelope = offlineWorkspaceEnvelope(cachedProject, 1_000, ACCOUNT_SCOPE);
  assert.ok(envelope);
  assert.equal(validateOfflineWorkspaceEnvelope({...envelope, projectId: "project_other"}, 1_001, ACCOUNT_SCOPE), null);
  assert.equal(validateOfflineWorkspaceEnvelope({...envelope, projectUpdatedAt: "2026-08-13T01:00:00Z"}, 1_001, ACCOUNT_SCOPE), null);
  assert.equal(validateOfflineWorkspaceEnvelope({...envelope, serializedChars: envelope.serializedChars + 1}, 1_001, ACCOUNT_SCOPE), null);
  assert.equal(validateOfflineWorkspaceEnvelope(envelope, 1_001, `acs1_${"b".repeat(43)}`), null);
  assert.equal(offlineWorkspaceEnvelope(cachedProject, 1_000, null), null);
});
