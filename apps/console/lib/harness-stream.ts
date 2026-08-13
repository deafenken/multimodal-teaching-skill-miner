export const HARNESS_TERMINAL_EVENTS = new Set([
  "run.completed",
  "run.failed",
  "run.cancelled",
  "run.handoff",
]);

export const HARNESS_EVENT_SCHEMA = "teaching_skill_miner.agent_harness_event.v1";

const HARNESS_EVENT_TYPES = new Set([
  "run.started", "run.completed", "run.failed", "run.cancelled", "run.handoff",
  "model.started", "model.completed", "model.failed", "model.retrying", "model.retry_suppressed",
  "message.start", "message.delta", "message.end", "reasoning.delta", "usage.update", "tool_call.delta",
  "tool.requested", "tool.started", "tool.progress", "tool.retrying", "tool.completed", "tool.failed",
  "tool.rejected", "tool.replayed", "tool.reconciled", "input.steered", "guard.triggered",
  "action.started", "action.completed", "action.superseded", "progress.updated", "operation.result",
  "state.committed",
]);

const MAX_SSE_BUFFER_CHARS = 1_048_576;
const MAX_TRACKED_EVENT_IDENTITIES = 128;

export type HarnessStreamOperation = "chat" | "start" | "step";

export interface HarnessSessionReference {
  session_id: string;
  context_version: number;
  response_sha256: string;
}

export interface HarnessUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  input_tokens?: number;
  output_tokens?: number;
  prompt_cache_hit_tokens?: number;
  prompt_cache_miss_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
  server_tool_use?: {web_search_requests: number};
}

export interface HarnessChatResult {
  mode: "chat";
  provider: string;
  model: string | null;
  latency_ms: number | null;
  usage: HarnessUsage;
  web_search_requested: boolean;
  web_search_used: boolean;
  sources: Array<{title: string; url: string}>;
  message_sha256: string;
  message_chars: number;
  response_sha256: string;
}

export type HarnessOperationResult =
  | {operation: "start" | "step"; sessionRef: HarnessSessionReference}
  | {operation: "chat"; chat: HarnessChatResult};

interface HarnessEventIdentity {
  sequence: number;
  eventId: string;
}

export interface HarnessEventEnvelope {
  schema: string;
  event_id: string;
  run_id: string;
  turn_id: string;
  sequence: number;
  type: string;
  timestamp: string;
  causation_id?: string;
  payload: Record<string, unknown>;
}

export interface HarnessStreamState {
  runId: string | null;
  turnId: string | null;
  lastSequence: number;
  terminalType: string | null;
  expectedOperation: HarnessStreamOperation | null;
  operationResultSequence: number | null;
  resultSessionId: string | null;
  resultContextVersion: number | null;
  resultResponseSha256: string | null;
  stateCommittedSequence: number | null;
  committedSessionId: string | null;
  committedContextVersion: number | null;
  committedResponseSha256: string | null;
  recentEventIdentities: readonly HarnessEventIdentity[];
}

export interface HarnessStreamReduction {
  state: HarnessStreamState;
  event: HarnessEventEnvelope;
  duplicate: boolean;
}

export interface SseFrame {
  event: string;
  data: string;
  id?: string;
}

export class HarnessStreamProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HarnessStreamProtocolError";
  }
}

export class HarnessStreamGapError extends HarnessStreamProtocolError {
  readonly expectedSequence: number;
  readonly receivedSequence: number;

  constructor(expectedSequence: number, receivedSequence: number) {
    super(`Harness event sequence gap: expected ${expectedSequence}, received ${receivedSequence}`);
    this.name = "HarnessStreamGapError";
    this.expectedSequence = expectedSequence;
    this.receivedSequence = receivedSequence;
  }
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

export function initialHarnessStreamState(
  afterSequence = 0,
  expectedOperation: HarnessStreamOperation | null = null,
): HarnessStreamState {
  if (!Number.isInteger(afterSequence) || afterSequence < 0) {
    throw new HarnessStreamProtocolError("Harness event cursor must be a non-negative integer");
  }
  return {
    runId: null,
    turnId: null,
    lastSequence: afterSequence,
    terminalType: null,
    expectedOperation,
    operationResultSequence: null,
    resultSessionId: null,
    resultContextVersion: null,
    resultResponseSha256: null,
    stateCommittedSequence: null,
    committedSessionId: null,
    committedContextVersion: null,
    committedResponseSha256: null,
    recentEventIdentities: [],
  };
}

export function decodeHarnessEvent(frame: SseFrame): HarnessEventEnvelope {
  let parsed: unknown;
  try {
    parsed = JSON.parse(frame.data);
  } catch {
    throw new HarnessStreamProtocolError("Harness SSE event contains invalid JSON");
  }
  const raw = recordValue(parsed);
  const payload = recordValue(raw?.payload);
  if (!raw || !payload) throw new HarnessStreamProtocolError("Harness SSE event is not an envelope");
  const runId = raw.run_id;
  const turnId = raw.turn_id;
  const sequence = raw.sequence;
  const type = raw.type;
  const schema = raw.schema;
  const eventId = raw.event_id;
  const timestamp = raw.timestamp;
  if (schema !== HARNESS_EVENT_SCHEMA) {
    throw new HarnessStreamProtocolError("Harness SSE event schema is unsupported");
  }
  if (typeof eventId !== "string" || !/^[0-9a-f]{32}$/.test(eventId)) {
    throw new HarnessStreamProtocolError("Harness SSE event id is invalid");
  }
  if (
    typeof timestamp !== "string"
    || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/.test(timestamp)
    || !Number.isFinite(Date.parse(timestamp))
  ) {
    throw new HarnessStreamProtocolError("Harness SSE event timestamp is invalid");
  }
  if (
    typeof runId !== "string" || !runId || runId.length > 160
    || typeof turnId !== "string" || !turnId || turnId.length > 160
  ) {
    throw new HarnessStreamProtocolError("Harness SSE event has no run identity");
  }
  if (!Number.isInteger(sequence) || Number(sequence) < 1) {
    throw new HarnessStreamProtocolError("Harness SSE event sequence is invalid");
  }
  if (
    typeof type !== "string" || !type || type.length > 100 || !HARNESS_EVENT_TYPES.has(type)
    || frame.event !== type
  ) {
    throw new HarnessStreamProtocolError("Harness SSE event type does not match its envelope");
  }
  if (frame.id !== String(sequence)) {
    throw new HarnessStreamProtocolError("Harness SSE event id does not match its sequence");
  }
  if (raw.causation_id !== undefined && (typeof raw.causation_id !== "string" || raw.causation_id.length > 160)) {
    throw new HarnessStreamProtocolError("Harness SSE causation id is invalid");
  }
  if (payload.channel !== undefined && payload.channel !== "assistant" && payload.channel !== "internal") {
    throw new HarnessStreamProtocolError("Harness SSE event channel is invalid");
  }
  if (type === "message.delta" && (typeof payload.delta !== "string" || payload.delta.length > 32_000)) {
    throw new HarnessStreamProtocolError("Harness assistant delta is invalid");
  }
  return {
    schema,
    event_id: eventId,
    run_id: runId,
    turn_id: turnId,
    sequence: Number(sequence),
    type,
    timestamp,
    ...(typeof raw.causation_id === "string" ? {causation_id: raw.causation_id} : {}),
    payload,
  };
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]) {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length && actual.every((key, index) => key === wanted[index]);
}

function validDigest(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function streamOperation(value: unknown): HarnessStreamOperation | null {
  return value === "chat" || value === "start" || value === "step" ? value : null;
}

function parseSessionReference(value: unknown): HarnessSessionReference {
  const reference = recordValue(value);
  if (
    !reference
    || !hasExactKeys(reference, ["session_id", "context_version", "response_sha256"])
    || typeof reference.session_id !== "string"
    || !reference.session_id
    || reference.session_id.length > 200
    || !Number.isInteger(reference.context_version)
    || Number(reference.context_version) < 0
    || !validDigest(reference.response_sha256)
  ) {
    throw new HarnessStreamProtocolError("Harness teaching result contains an invalid session reference");
  }
  return {
    session_id: reference.session_id,
    context_version: Number(reference.context_version),
    response_sha256: reference.response_sha256,
  };
}

function parseChatResult(value: unknown): HarnessChatResult {
  const chat = recordValue(value);
  const expectedKeys = [
    "mode",
    "provider",
    "model",
    "latency_ms",
    "usage",
    "web_search_requested",
    "web_search_used",
    "sources",
    "message_sha256",
    "message_chars",
    "response_sha256",
  ] as const;
  const usageCounterKeys = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
  ] as const;
  const rawUsage = recordValue(chat?.usage);
  if (!rawUsage) {
    throw new HarnessStreamProtocolError("Harness Chat usage metadata is invalid");
  }
  const allowedUsageKeys = new Set<string>([
    ...usageCounterKeys,
    "server_tool_use",
  ]);
  if (Object.keys(rawUsage).some((key) => !allowedUsageKeys.has(key))) {
    throw new HarnessStreamProtocolError("Harness Chat usage metadata is invalid");
  }
  const usage: HarnessUsage = {};
  for (const key of usageCounterKeys) {
    if (!Object.hasOwn(rawUsage, key)) continue;
    const amount = rawUsage[key];
    if (typeof amount !== "number" || !Number.isSafeInteger(amount) || amount < 0) {
      throw new HarnessStreamProtocolError("Harness Chat usage metadata is invalid");
    }
    usage[key] = amount;
  }
  if (Object.hasOwn(rawUsage, "server_tool_use")) {
    const serverTools = recordValue(rawUsage.server_tool_use);
    if (
      !serverTools
      || !hasExactKeys(serverTools, ["web_search_requests"])
      || typeof serverTools.web_search_requests !== "number"
      || !Number.isSafeInteger(serverTools.web_search_requests)
      || serverTools.web_search_requests < 0
    ) {
      throw new HarnessStreamProtocolError("Harness Chat usage metadata is invalid");
    }
    usage.server_tool_use = {
      web_search_requests: serverTools.web_search_requests,
    };
  }
  if (
    !chat
    || !hasExactKeys(chat, expectedKeys)
    || chat.mode !== "chat"
    || typeof chat.provider !== "string"
    || !chat.provider
    || chat.provider.length > 80
    || !(chat.model === null || (typeof chat.model === "string" && chat.model.length <= 160))
    || !(chat.latency_ms === null || (typeof chat.latency_ms === "number" && Number.isFinite(chat.latency_ms) && chat.latency_ms >= 0))
    || typeof chat.web_search_requested !== "boolean"
    || typeof chat.web_search_used !== "boolean"
    || !Array.isArray(chat.sources)
    || chat.sources.length > 20
    || !validDigest(chat.message_sha256)
    || !Number.isInteger(chat.message_chars)
    || Number(chat.message_chars) < 0
    || Number(chat.message_chars) > 2_000_000
    || !validDigest(chat.response_sha256)
  ) {
    throw new HarnessStreamProtocolError("Harness Chat result metadata is invalid");
  }
  const sources = chat.sources.map((item) => {
    const source = recordValue(item);
    if (
      !source
      || !hasExactKeys(source, ["title", "url"])
      || typeof source.title !== "string"
      || !source.title
      || source.title.length > 500
      || typeof source.url !== "string"
      || source.url.length > 2_000
    ) {
      throw new HarnessStreamProtocolError("Harness Chat source metadata is invalid");
    }
    let parsed: URL;
    try {
      parsed = new URL(source.url);
    } catch {
      throw new HarnessStreamProtocolError("Harness Chat source URL is invalid");
    }
    if (
      (parsed.protocol !== "http:" && parsed.protocol !== "https:")
      || Boolean(parsed.username)
      || Boolean(parsed.password)
    ) {
      throw new HarnessStreamProtocolError("Harness Chat source URL is invalid");
    }
    return {title: source.title, url: source.url};
  });
  return {
    mode: "chat",
    provider: chat.provider,
    model: chat.model as string | null,
    latency_ms: chat.latency_ms as number | null,
    usage,
    web_search_requested: chat.web_search_requested,
    web_search_used: chat.web_search_used,
    sources,
    message_sha256: chat.message_sha256,
    message_chars: Number(chat.message_chars),
    response_sha256: chat.response_sha256,
  };
}

export function harnessOperationResult(
  event: HarnessEventEnvelope,
  expectedOperation: HarnessStreamOperation | null = null,
): HarnessOperationResult {
  if (event.type !== "operation.result") {
    throw new HarnessStreamProtocolError("Harness authority can only come from operation.result");
  }
  if (
    !hasExactKeys(event.payload, ["channel", "operation", "result"])
    || event.payload.channel !== "internal"
  ) {
    throw new HarnessStreamProtocolError("Harness operation result envelope contains non-public fields");
  }
  const operation = streamOperation(event.payload.operation);
  const result = recordValue(event.payload.result);
  if (!operation || (expectedOperation !== null && operation !== expectedOperation) || !result) {
    throw new HarnessStreamProtocolError("Harness operation result does not match the requested operation");
  }
  if (operation === "chat") {
    if (!hasExactKeys(result, ["chat"])) {
      throw new HarnessStreamProtocolError("Harness Chat result contains non-public fields");
    }
    return {operation, chat: parseChatResult(result.chat)};
  }
  if (!hasExactKeys(result, ["session_ref"])) {
    throw new HarnessStreamProtocolError("Harness teaching result contains non-reference fields");
  }
  return {operation, sessionRef: parseSessionReference(result.session_ref)};
}

export async function validateHarnessChatMessage(
  result: HarnessChatResult,
  assistantMessage: string,
): Promise<void> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(assistantMessage),
  );
  const messageSha256 = Array.from(
    new Uint8Array(digest),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
  if (
    messageSha256 !== result.message_sha256
    || [...assistantMessage].length !== result.message_chars
  ) {
    throw new HarnessStreamProtocolError("Streamed Chat message does not match its completion receipt");
  }
}

export function validateHarnessSessionSnapshot(
  reference: HarnessSessionReference,
  value: unknown,
): void {
  const session = recordValue(value);
  if (
    !session
    || session.session_id !== reference.session_id
    || !Number.isInteger(session.context_version)
    || Number(session.context_version) !== reference.context_version
    || !validDigest(session.response_sha256)
  ) {
    throw new HarnessStreamProtocolError("Fetched teaching session does not match its committed reference");
  }
  if (session.response_sha256 !== reference.response_sha256) {
    throw new HarnessStreamProtocolError("Fetched teaching session hash does not match its committed reference");
  }
}

function appendEventIdentity(
  identities: readonly HarnessEventIdentity[],
  event: HarnessEventEnvelope,
): readonly HarnessEventIdentity[] {
  return [...identities, {sequence: event.sequence, eventId: event.event_id}]
    .slice(-MAX_TRACKED_EVENT_IDENTITIES);
}

export function reduceHarnessEvent(
  state: HarnessStreamState,
  event: HarnessEventEnvelope,
): HarnessStreamReduction {
  if (state.runId !== null && (state.runId !== event.run_id || state.turnId !== event.turn_id)) {
    throw new HarnessStreamProtocolError("Harness stream changed run identity");
  }
  if (event.sequence <= state.lastSequence) {
    const known = state.recentEventIdentities.find((item) => item.sequence === event.sequence);
    if (!known) {
      throw new HarnessStreamProtocolError(
        "Harness replay referenced an event outside the verified identity window",
      );
    }
    if (known.eventId !== event.event_id) {
      throw new HarnessStreamProtocolError("Harness replay equivocated for an existing event sequence");
    }
    return {state, event, duplicate: true};
  }
  const expected = state.lastSequence + 1;
  if (event.sequence !== expected) throw new HarnessStreamGapError(expected, event.sequence);
  if (state.terminalType !== null) {
    throw new HarnessStreamProtocolError("Harness emitted an event after its terminal event");
  }
  let expectedOperation = state.expectedOperation;
  let operationResultSequence = state.operationResultSequence;
  let resultSessionId = state.resultSessionId;
  let resultContextVersion = state.resultContextVersion;
  let resultResponseSha256 = state.resultResponseSha256;
  let stateCommittedSequence = state.stateCommittedSequence;
  let committedSessionId = state.committedSessionId;
  let committedContextVersion = state.committedContextVersion;
  let committedResponseSha256 = state.committedResponseSha256;

  if (event.type === "run.started") {
    const announcedOperation = streamOperation(event.payload.operation);
    if (!announcedOperation || (expectedOperation !== null && announcedOperation !== expectedOperation)) {
      throw new HarnessStreamProtocolError("Harness run operation does not match the request");
    }
    expectedOperation = announcedOperation;
  } else if (event.type === "operation.result") {
    if (operationResultSequence !== null) {
      throw new HarnessStreamProtocolError("Harness emitted more than one operation result");
    }
    const result = harnessOperationResult(event, expectedOperation);
    expectedOperation = result.operation;
    operationResultSequence = event.sequence;
    resultResponseSha256 = result.operation === "chat"
      ? result.chat.response_sha256
      : result.sessionRef.response_sha256;
    if (result.operation !== "chat") {
      resultSessionId = result.sessionRef.session_id;
      resultContextVersion = result.sessionRef.context_version;
    }
  } else if (event.type === "state.committed") {
    if (
      !hasExactKeys(event.payload, [
        "channel",
        "operation",
        "session_id",
        "context_version",
        "response_sha256",
      ])
      || event.payload.channel !== "internal"
    ) {
      throw new HarnessStreamProtocolError("Harness commit envelope contains non-public fields");
    }
    const operation = streamOperation(event.payload.operation);
    if (!operation || (expectedOperation !== null && operation !== expectedOperation)) {
      throw new HarnessStreamProtocolError("Harness commit does not match the requested operation");
    }
    if (stateCommittedSequence !== null) {
      throw new HarnessStreamProtocolError("Harness emitted more than one state commit");
    }
    if (operationResultSequence === null) {
      throw new HarnessStreamProtocolError("Harness committed state before publishing its operation result");
    }
    const sessionId = event.payload.session_id;
    const contextVersion = event.payload.context_version;
    const responseSha256 = event.payload.response_sha256;
    if (operation === "start" || operation === "step") {
      if (
        typeof sessionId !== "string"
        || !sessionId
        || !Number.isInteger(contextVersion)
        || !validDigest(responseSha256)
        || sessionId !== resultSessionId
        || Number(contextVersion) !== resultContextVersion
        || responseSha256 !== resultResponseSha256
      ) {
        throw new HarnessStreamProtocolError("Harness commit does not match its teaching result reference");
      }
      committedSessionId = sessionId;
      committedContextVersion = Number(contextVersion);
      committedResponseSha256 = responseSha256;
    } else {
      if (
        sessionId !== null
        || contextVersion !== null
        || !validDigest(responseSha256)
        || responseSha256 !== resultResponseSha256
      ) {
        throw new HarnessStreamProtocolError("Harness Chat commit hash does not match its result metadata");
      }
      committedResponseSha256 = responseSha256;
    }
    stateCommittedSequence = event.sequence;
  }
  const terminalType = HARNESS_TERMINAL_EVENTS.has(event.type) ? event.type : null;
  if (event.type === "run.completed") {
    if (operationResultSequence === null) {
      throw new HarnessStreamProtocolError("Harness completed without one authoritative operation result");
    }
    if (stateCommittedSequence === null || stateCommittedSequence <= operationResultSequence) {
      throw new HarnessStreamProtocolError("Harness completed before its authoritative state commit");
    }
  }
  return {
    event,
    duplicate: false,
    state: {
      runId: state.runId ?? event.run_id,
      turnId: state.turnId ?? event.turn_id,
      lastSequence: event.sequence,
      terminalType,
      expectedOperation,
      operationResultSequence,
      resultSessionId,
      resultContextVersion,
      resultResponseSha256,
      stateCommittedSequence,
      committedSessionId,
      committedContextVersion,
      committedResponseSha256,
      recentEventIdentities: appendEventIdentity(state.recentEventIdentities, event),
    },
  };
}

export class SseFrameDecoder {
  private buffer = "";

  push(chunk: string): SseFrame[] {
    this.buffer += chunk;
    if (this.buffer.length > MAX_SSE_BUFFER_CHARS) {
      throw new HarnessStreamProtocolError("Harness SSE frame exceeds the client budget");
    }
    const frames: SseFrame[] = [];
    while (true) {
      const match = /\r?\n\r?\n/.exec(this.buffer);
      if (!match || match.index === undefined) break;
      const block = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      const frame = this.parseBlock(block);
      if (frame) frames.push(frame);
    }
    return frames;
  }

  finish(): SseFrame[] {
    const frames = this.push("");
    if (!this.buffer.trim()) {
      this.buffer = "";
      return frames;
    }
    // A final SSE frame may legally end at EOF without an empty line.
    const frame = this.parseBlock(this.buffer);
    this.buffer = "";
    if (frame) frames.push(frame);
    return frames;
  }

  private parseBlock(block: string): SseFrame | null {
    if (!block || block.split(/\r?\n/).every((line) => !line || line.startsWith(":"))) return null;
    let event = "message";
    let id: string | undefined;
    const data: string[] = [];
    for (const line of block.split(/\r?\n/)) {
      if (!line || line.startsWith(":")) continue;
      const separator = line.indexOf(":");
      const field = separator < 0 ? line : line.slice(0, separator);
      const rawValue = separator < 0 ? "" : line.slice(separator + 1);
      const value = rawValue.startsWith(" ") ? rawValue.slice(1) : rawValue;
      if (field === "event") event = value || "message";
      else if (field === "data") data.push(value);
      else if (field === "id" && !value.includes("\0")) id = value;
    }
    if (!data.length) return null;
    return {event, data: data.join("\n"), ...(id === undefined ? {} : {id})};
  }
}

export function harnessDelta(event: HarnessEventEnvelope): string {
  if (event.type !== "message.delta" || event.payload.channel !== "assistant") return "";
  const delta = event.payload.delta ?? event.payload.text;
  return typeof delta === "string" ? delta : "";
}

const PRIVATE_PRESENTATION_EVENT_TYPES = new Set([
  "reasoning.delta",
  "usage.update",
  "tool_call.delta",
]);

/**
 * Keep durable execution events in the protocol reducer without projecting
 * private model or Agent-loop activity into the learner-facing conversation.
 */
export function harnessEventIsUserVisible(event: HarnessEventEnvelope): boolean {
  return event.payload.channel !== "internal"
    && !PRIVATE_PRESENTATION_EVENT_TYPES.has(event.type);
}

export function harnessStatusLabel(event: HarnessEventEnvelope): string | null {
  if (!harnessEventIsUserVisible(event)) return null;
  const payload = event.payload;
  if (event.type === "run.started") return payload.resumed ? "连接已恢复，正在继续…" : "正在建立运行环境…";
  if (event.type === "model.started") return "正在生成回答…";
  if (event.type === "model.completed") return "正在生成回答…";
  if (event.type === "model.failed") return "模型调用未完成，正在检查恢复策略…";
  if (event.type === "model.retrying") return "模型连接波动，正在自动重试…";
  if (event.type === "message.start") return "DeepSeek 正在输出…";
  if (event.type === "tool.started") {
    const name = typeof payload.tool_name === "string" ? payload.tool_name : "工具";
    return `正在运行 ${name}…`;
  }
  if (event.type === "tool.progress") return "工具正在处理…";
  if (event.type === "tool.completed") return "工具已完成，正在继续…";
  return null;
}
