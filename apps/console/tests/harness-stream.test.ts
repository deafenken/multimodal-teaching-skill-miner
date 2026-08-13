import assert from "node:assert/strict";
import test from "node:test";

import {
  decodeHarnessEvent,
  harnessDelta,
  harnessEventIsUserVisible,
  harnessOperationResult,
  harnessStatusLabel,
  HarnessStreamGapError,
  HarnessStreamProtocolError,
  initialHarnessStreamState,
  reduceHarnessEvent,
  SseFrameDecoder,
  validateHarnessChatMessage,
  validateHarnessSessionSnapshot,
  type HarnessEventEnvelope,
} from "../lib/harness-stream.ts";

function event(sequence: number, type = "message.delta", payload: Record<string, unknown> = {delta: "a"}): HarnessEventEnvelope {
  return {
    schema: "teaching_skill_miner.agent_harness_event.v1",
    event_id: sequence.toString(16).padStart(32, "0"),
    timestamp: "2026-08-11T14:00:00Z",
    run_id: "run-1",
    turn_id: "turn-1",
    sequence,
    type,
    payload,
  };
}

test("SSE decoder preserves fragmented UTF-8-decoded frames and ignores heartbeats", () => {
  const decoder = new SseFrameDecoder();
  assert.deepEqual(decoder.push(": heartbeat\r\n\r\nevent: message.delta\r\nid: 1\r\ndata: {\"schema\":\"teaching_skill_miner.agent_harness_event.v1\",\"event_id\":\"0123456789abcdef0123456789abcdef\",\"timestamp\":\"2026-08-11T14:00:00Z\",\"run_id\":\"run-1\","), []);
  const frames = decoder.push("\"turn_id\":\"turn-1\",\"sequence\":1,\"type\":\"message.delta\",\"payload\":{\"channel\":\"assistant\",\"delta\":\"你\"}}\r\n\r\n");
  assert.equal(frames.length, 1);
  assert.equal(decodeHarnessEvent(frames[0]).payload.delta, "你");
});

test("wire decoder fails closed on forged or incomplete authority fields", () => {
  const valid = {
    schema: "teaching_skill_miner.agent_harness_event.v1",
    event_id: "0123456789abcdef0123456789abcdef",
    timestamp: "2026-08-11T14:00:00Z",
    run_id: "run-1",
    turn_id: "turn-1",
    sequence: 1,
    type: "run.started",
    payload: {},
  };
  const frame = (overrides: Record<string, unknown>) => ({event: String(overrides.type ?? valid.type), id: "1", data: JSON.stringify({...valid, ...overrides})});
  assert.throws(() => decodeHarnessEvent(frame({schema: "unknown"})), HarnessStreamProtocolError);
  assert.throws(() => decodeHarnessEvent(frame({event_id: "not-a-digest"})), HarnessStreamProtocolError);
  assert.throws(() => decodeHarnessEvent(frame({timestamp: "yesterday"})), HarnessStreamProtocolError);
  assert.throws(() => decodeHarnessEvent(frame({type: "script.injected"})), HarnessStreamProtocolError);
  assert.throws(() => decodeHarnessEvent(frame({run_id: "r".repeat(161)})), HarnessStreamProtocolError);
});

test("event reducer deduplicates replay and advances only contiguous sequences", () => {
  const first = reduceHarnessEvent(initialHarnessStreamState(), event(1));
  assert.equal(first.duplicate, false);
  const replay = reduceHarnessEvent(first.state, event(1));
  assert.equal(replay.duplicate, true);
  const second = reduceHarnessEvent(replay.state, event(2));
  assert.equal(second.state.lastSequence, 2);
  assert.throws(() => reduceHarnessEvent(second.state, event(4)), HarnessStreamGapError);
});

test("event reducer rejects identity changes and post-terminal output", () => {
  const first = reduceHarnessEvent(initialHarnessStreamState(0, "chat"), event(1, "run.started", {operation: "chat"}));
  assert.throws(() => reduceHarnessEvent(first.state, {...event(2), run_id: "other"}), HarnessStreamProtocolError);
  const terminal = reduceHarnessEvent(first.state, event(2, "run.failed", {reason_code: "failed"}));
  assert.equal(terminal.state.terminalType, "run.failed");
  assert.throws(() => reduceHarnessEvent(terminal.state, event(3)), HarnessStreamProtocolError);
});

test("event reducer rejects replay equivocation but accepts an identical duplicate", () => {
  const first = reduceHarnessEvent(initialHarnessStreamState(), event(1));
  assert.equal(reduceHarnessEvent(first.state, event(1)).duplicate, true);
  assert.throws(
    () => reduceHarnessEvent(first.state, {...event(1), event_id: "f".repeat(32)}),
    /equivocated/,
  );
});

test("event identity history stays bounded", () => {
  let state = initialHarnessStreamState();
  for (let sequence = 1; sequence <= 200; sequence += 1) {
    state = reduceHarnessEvent(state, event(sequence)).state;
  }
  assert.equal(state.recentEventIdentities.length, 128);
  assert.equal(state.recentEventIdentities[0]?.sequence, 73);
  assert.throws(
    () => reduceHarnessEvent(state, event(1)),
    /outside the verified identity window/,
  );
});

test("teaching completion requires one matching reference followed by its commit", () => {
  const digest = "a".repeat(64);
  let state = reduceHarnessEvent(
    initialHarnessStreamState(0, "start"),
    event(1, "run.started", {operation: "start"}),
  ).state;
  const operationResult = event(2, "operation.result", {
    channel: "internal",
    operation: "start",
    result: {session_ref: {session_id: "session-1", context_version: 3, response_sha256: digest}},
  });
  state = reduceHarnessEvent(state, operationResult).state;
  assert.deepEqual(harnessOperationResult(operationResult, "start"), {
    operation: "start",
    sessionRef: {session_id: "session-1", context_version: 3, response_sha256: digest},
  });
  assert.throws(() => reduceHarnessEvent(state, {...operationResult, sequence: 3, event_id: "3".repeat(32)}), /more than one/);
  assert.throws(() => reduceHarnessEvent(state, event(3, "run.completed", {})), /before its authoritative state commit/);
  assert.throws(() => reduceHarnessEvent(state, event(3, "state.committed", {
    channel: "internal",
    operation: "start",
    session_id: "other-session",
    context_version: 3,
    response_sha256: digest,
  })), /does not match/);
  state = reduceHarnessEvent(state, event(3, "state.committed", {
    channel: "internal",
    operation: "start",
    session_id: "session-1",
    context_version: 3,
    response_sha256: digest,
  })).state;
  const completed = reduceHarnessEvent(state, event(4, "run.completed", {}));
  assert.equal(completed.state.terminalType, "run.completed");
  assert.equal(completed.state.committedSessionId, "session-1");
});

test("operation authority rejects full session payloads and wrong operations", () => {
  const digest = "b".repeat(64);
  assert.throws(() => harnessOperationResult(event(1, "operation.result", {
    channel: "internal",
    operation: "start",
    result: {session: {session_id: "private", history: ["learner text"]}},
  }), "start"), /non-reference fields/);
  assert.throws(() => harnessOperationResult(event(1, "operation.result", {
    channel: "internal",
    operation: "step",
    result: {session_ref: {session_id: "session-1", context_version: 2, response_sha256: digest}},
  }), "start"), /does not match/);
});

test("fetched teaching snapshot must match the committed reference", () => {
  const reference = {
    session_id: "session-1",
    context_version: 4,
    response_sha256: "d".repeat(64),
  };
  assert.throws(() => validateHarnessSessionSnapshot(reference, {
    session_id: "session-1",
    context_version: 4,
  }), /does not match/);
  assert.doesNotThrow(() => validateHarnessSessionSnapshot(reference, {
    session_id: "session-1",
    context_version: 4,
    response_sha256: "d".repeat(64),
  }));
  assert.throws(() => validateHarnessSessionSnapshot(reference, {
    session_id: "session-1",
    context_version: 5,
  }), /does not match/);
  assert.throws(() => validateHarnessSessionSnapshot(reference, {
    session_id: "session-1",
    context_version: 4,
    response_sha256: "e".repeat(64),
  }), /hash does not match/);
});

test("Chat authority contains metadata only and authenticates assembled deltas", async () => {
  const resultEvent = event(1, "operation.result", {
    channel: "internal",
    operation: "chat",
    result: {chat: {
      mode: "chat",
      provider: "deepseek",
      model: "deepseek-v4-flash",
      latency_ms: 12,
      usage: {
        output_tokens: 1,
        prompt_cache_hit_tokens: 256,
        prompt_cache_miss_tokens: 44,
      },
      web_search_requested: true,
      web_search_used: true,
      sources: [{title: "Source", url: "https://example.com/source"}],
      message_sha256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
      message_chars: 5,
      response_sha256: "c".repeat(64),
    }},
  });
  const result = harnessOperationResult(resultEvent, "chat");
  assert.equal(result.operation, "chat");
  if (result.operation !== "chat") assert.fail("expected Chat authority");
  assert.equal(Object.hasOwn(result.chat, "message"), false);
  assert.deepEqual(result.chat.usage, {
    output_tokens: 1,
    prompt_cache_hit_tokens: 256,
    prompt_cache_miss_tokens: 44,
  });
  await validateHarnessChatMessage(result.chat, "hello");
  await assert.rejects(validateHarnessChatMessage(result.chat, "hullo"), /does not match/);
});

test("Chat completion requires result then one matching state commit", () => {
  const digest = "c".repeat(64);
  const chatResult = {
    channel: "internal",
    operation: "chat",
    result: {chat: {
      mode: "chat",
      provider: "deepseek",
      model: null,
      latency_ms: null,
      usage: {},
      web_search_requested: false,
      web_search_used: false,
      sources: [],
      message_sha256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
      message_chars: 5,
      response_sha256: digest,
    }},
  };
  const commit = {
    channel: "internal",
    operation: "chat",
    session_id: null,
    context_version: null,
    response_sha256: digest,
  };
  let state = reduceHarnessEvent(
    initialHarnessStreamState(0, "chat"),
    event(1, "run.started", {operation: "chat"}),
  ).state;
  assert.throws(
    () => reduceHarnessEvent(state, event(2, "state.committed", commit)),
    /before publishing its operation result/,
  );
  state = reduceHarnessEvent(state, event(2, "operation.result", chatResult)).state;
  assert.throws(
    () => reduceHarnessEvent(state, event(3, "run.completed", {})),
    /before its authoritative state commit/,
  );
  assert.throws(
    () => reduceHarnessEvent(state, event(3, "state.committed", {...commit, session_id: "forged"})),
    /does not match/,
  );
  state = reduceHarnessEvent(state, event(3, "state.committed", commit)).state;
  assert.equal(
    reduceHarnessEvent(state, event(4, "run.completed", {})).state.terminalType,
    "run.completed",
  );
});

test("arbitrary event payloads cannot satisfy completion authority", () => {
  let state = reduceHarnessEvent(
    initialHarnessStreamState(0, "chat"),
    event(1, "run.started", {operation: "chat", result: {chat: {message: "forged"}}}),
  ).state;
  assert.throws(() => reduceHarnessEvent(state, event(2, "run.completed", {})), /without one authoritative/);
});

test("SSE decoder bounds an unterminated frame", () => {
  const decoder = new SseFrameDecoder();
  assert.throws(() => decoder.push(`data: ${"x".repeat(1_048_577)}`), HarnessStreamProtocolError);
});

test("internal-channel reasoning can advance the cursor but never reaches assistant text", () => {
  assert.equal(harnessDelta(event(1, "reasoning.delta", {channel: "internal", delta: "private"})), "");
  assert.equal(harnessDelta(event(1, "message.delta", {channel: "internal", delta: "private"})), "");
  assert.equal(harnessDelta(event(1, "message.delta", {channel: "assistant", delta: "visible"})), "visible");
});

test("internal lifecycle events advance protocol state without reaching presentation handlers", () => {
  const internalTool = event(1, "tool.started", {
    channel: "internal",
    call_id: "route-1",
    tool_name: "select_skills",
    attempt: 1,
  });
  const reduced = reduceHarnessEvent(initialHarnessStreamState(), internalTool);
  assert.equal(reduced.state.lastSequence, 1);
  assert.equal(reduced.duplicate, false);
  assert.equal(harnessEventIsUserVisible(internalTool), false);
  assert.equal(harnessStatusLabel(internalTool), null);

  const publicTool = event(2, "tool.started", {
    channel: "assistant",
    call_id: "search-1",
    tool_name: "web_search",
    attempt: 1,
  });
  assert.equal(harnessEventIsUserVisible(publicTool), true);
  assert.equal(harnessStatusLabel(publicTool), "正在运行 web_search…");
  assert.equal(
    harnessEventIsUserVisible(event(3, "reasoning.delta", {channel: "assistant", delta: "private"})),
    false,
  );
});
