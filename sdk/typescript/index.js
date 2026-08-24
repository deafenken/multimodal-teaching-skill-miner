import { spawn } from "node:child_process";
import {
  accessSync,
  constants as fsConstants,
  realpathSync,
  statSync,
} from "node:fs";
import { homedir } from "node:os";
import {
  isAbsolute,
  relative,
  resolve,
  sep,
} from "node:path";
import { TextDecoder } from "node:util";

const EXEC_RESULT_SCHEMA = "agent_harness.exec_result.v1";
const EVENT_SCHEMA = "agent_harness.event.v1";
const SESSION_ID = /^[a-z][a-z0-9_-]{7,159}$/;
const EVENT_ID = /^[0-9a-f]{32}$/;
const RUN_ID = /^run_[0-9a-f]{32}$/;
const TURN_ID = /^turn_[0-9a-f]{32}$/;
const PERMISSION_MODES = new Set([
  "read-only",
  "workspace-write",
  "full-access",
]);
const TERMINAL_EVENT_STATUS = new Map([
  ["run.completed", "completed"],
  ["run.cancelled", "cancelled"],
  ["run.failed", "failed"],
  ["run.handoff", "handoff"],
]);
const EXPECTED_EXIT_CODE = new Map([
  ["completed", 0],
  ["cancelled", 2],
  ["handoff", 3],
  ["failed", 4],
]);
const EVENT_TYPES = new Set([
  "run.started",
  "run.completed",
  "run.cancelled",
  "run.failed",
  "run.handoff",
  "model.started",
  "model.completed",
  "model.failed",
  "model.retrying",
  "model.retry_suppressed",
  "message.start",
  "message.delta",
  "message.end",
  "reasoning.delta",
  "usage.update",
  "tool_call.delta",
  "approval.requested",
  "approval.resolved",
  "hook.started",
  "hook.effect_started",
  "hook.completed",
  "hook.failed",
  "tool.requested",
  "tool.started",
  "tool.effect_started",
  "tool.progress",
  "tool.retrying",
  "tool.completed",
  "tool.failed",
  "tool.rejected",
  "tool.replayed",
  "tool.reconciled",
  "input.steered",
  "guard.triggered",
  "action.started",
  "action.completed",
  "action.superseded",
  "progress.updated",
  "operation.result",
  "state.committed",
]);

const DEFAULT_LIMITS = Object.freeze({
  maxLineBytes: 2 * 1024 * 1024,
  maxRecords: 20_000,
  maxOutputBytes: 64 * 1024 * 1024,
  maxResponseBytes: 8 * 1024 * 1024,
  abortGraceMs: 1_500,
  resultCloseGraceMs: 1_500,
});

export class HarnessSdkError extends Error {
  constructor(message, code = "HARNESS_SDK_ERROR") {
    super(message);
    this.name = "HarnessSdkError";
    this.code = code;
  }
}

export class HarnessValidationError extends HarnessSdkError {
  constructor(message) {
    super(message, "HARNESS_VALIDATION_ERROR");
    this.name = "HarnessValidationError";
  }
}

export class HarnessProtocolError extends HarnessSdkError {
  constructor(message) {
    super(message, "HARNESS_PROTOCOL_ERROR");
    this.name = "HarnessProtocolError";
  }
}

export class HarnessProcessError extends HarnessSdkError {
  constructor(message, { exitCode = null, signal = null } = {}) {
    super(message, "HARNESS_PROCESS_ERROR");
    this.name = "HarnessProcessError";
    this.exitCode = exitCode;
    this.signal = signal;
  }
}

export class HarnessAbortError extends HarnessSdkError {
  constructor() {
    super("Agent Harness run was aborted.", "ABORT_ERR");
    this.name = "AbortError";
  }
}

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function absorbThenable(value) {
  if (
    value === null ||
    (typeof value !== "object" && typeof value !== "function")
  ) {
    return;
  }
  let then;
  try {
    then = value.then;
  } catch {
    return;
  }
  if (typeof then !== "function") {
    return;
  }
  try {
    Promise.resolve(value).catch(() => {});
  } catch {
    // Observation callbacks and cleanup hooks never control the run outcome.
  }
}

function requireVoidCallbackResult(value, name) {
  if (value === undefined) {
    return;
  }
  absorbThenable(value);
  throw new HarnessValidationError(`${name} must return undefined.`);
}

function requireOptions(value, name) {
  if (value === undefined) {
    return {};
  }
  if (!isRecord(value)) {
    throw new HarnessValidationError(`${name} must be an object.`);
  }
  return value;
}

function assertKnownOptions(value, known, name) {
  for (const key of Object.keys(value)) {
    if (!known.has(key)) {
      throw new HarnessValidationError(`${name} contains an unsupported option.`);
    }
  }
}

function boundedString(value, name, { min = 1, max = 4_096 } = {}) {
  if (
    typeof value !== "string" ||
    value.length < min ||
    value.length > max ||
    value.includes("\0") ||
    value !== value.trim()
  ) {
    throw new HarnessValidationError(`${name} is invalid.`);
  }
  return value;
}

function boundedInteger(value, name, minimum, maximum, fallback) {
  const selected = value === undefined ? fallback : value;
  if (
    !Number.isSafeInteger(selected) ||
    selected < minimum ||
    selected > maximum
  ) {
    throw new HarnessValidationError(
      `${name} must be an integer in [${minimum}, ${maximum}].`,
    );
  }
  return selected;
}

function boundedNumber(value, name, minimum, maximum, fallback) {
  const selected = value === undefined ? fallback : value;
  if (
    typeof selected !== "number" ||
    !Number.isFinite(selected) ||
    selected < minimum ||
    selected > maximum
  ) {
    throw new HarnessValidationError(
      `${name} must be a finite number in [${minimum}, ${maximum}].`,
    );
  }
  return selected;
}

function permissionMode(value, fallback = "read-only") {
  const selected = value === undefined ? fallback : value;
  if (!PERMISSION_MODES.has(selected)) {
    throw new HarnessValidationError("permissionMode is invalid.");
  }
  return selected;
}

function existingDirectory(value, name, base = process.cwd()) {
  const spelling = boundedString(value, name);
  let selected;
  try {
    selected = realpathSync(resolve(base, spelling));
    if (!statSync(selected).isDirectory()) {
      throw new Error("not a directory");
    }
  } catch {
    throw new HarnessValidationError(`${name} must be an existing directory.`);
  }
  return selected;
}

function pathInside(parent, child) {
  const difference = relative(parent, child);
  return (
    difference === "" ||
    (!difference.startsWith(`..${sep}`) && difference !== ".." && !isAbsolute(difference))
  );
}

function optionalAbsolutePath(value, name, base) {
  if (value === undefined) {
    return null;
  }
  const spelling = boundedString(value, name);
  return resolve(base, expandHome(spelling));
}

function expandHome(value) {
  if (value === "~") {
    return homedir();
  }
  if (value.startsWith(`~${sep}`)) {
    return resolve(homedir(), value.slice(2));
  }
  return value;
}

function executable(value, cwd) {
  const selected = boundedString(value ?? "harness", "harnessPath");
  const hasSeparator = selected.includes("/") || selected.includes("\\");
  if (!hasSeparator && !isAbsolute(selected)) {
    if (selected.startsWith("-")) {
      throw new HarnessValidationError("harnessPath is invalid.");
    }
    return selected;
  }
  const absolute = resolve(cwd, expandHome(selected));
  try {
    if (!statSync(absolute).isFile()) {
      throw new Error("not a file");
    }
    if (process.platform !== "win32") {
      accessSync(absolute, fsConstants.X_OK);
    }
  } catch {
    throw new HarnessValidationError(
      "harnessPath must identify an executable file.",
    );
  }
  return absolute;
}

function normalizedEnvironment(value) {
  if (value === undefined) {
    return Object.freeze({ ...process.env });
  }
  if (!isRecord(value)) {
    throw new HarnessValidationError("env must be an object.");
  }
  const merged = { ...process.env };
  for (const [key, raw] of Object.entries(value)) {
    if (
      !key ||
      key.length > 1_024 ||
      key.includes("\0") ||
      key.includes("=")
    ) {
      throw new HarnessValidationError("env contains an invalid variable name.");
    }
    if (raw === undefined) {
      delete merged[key];
      continue;
    }
    if (
      typeof raw !== "string" ||
      raw.length > 1_048_576 ||
      raw.includes("\0")
    ) {
      throw new HarnessValidationError("env contains an invalid variable value.");
    }
    merged[key] = raw;
  }
  return Object.freeze(merged);
}

function signalOption(value) {
  if (value === undefined) {
    return null;
  }
  try {
    if (
      !isRecord(value) ||
      typeof value.aborted !== "boolean" ||
      typeof value.addEventListener !== "function" ||
      typeof value.removeEventListener !== "function"
    ) {
      throw new Error("invalid signal shape");
    }
  } catch {
    throw new HarnessValidationError("signal must be an AbortSignal.");
  }
  return value;
}

function normalizeAttachments(value, activeDirectory) {
  if (value === undefined) {
    return Object.freeze([]);
  }
  if (!Array.isArray(value) || value.length > 8) {
    throw new HarnessValidationError("attachments must contain at most 8 paths.");
  }
  return Object.freeze(
    value.map((item) => {
      const spelling = boundedString(item, "attachment path");
      return resolve(activeDirectory, expandHome(spelling));
    }),
  );
}

function normalizePrompt(value) {
  if (typeof value !== "string") {
    throw new HarnessValidationError("prompt must be a string.");
  }
  const selected = value.trim();
  if (!selected || selected.length > 200_000 || selected.includes("\0")) {
    throw new HarnessValidationError("prompt is invalid.");
  }
  return selected;
}

function normalizeRunOptions(value, activeDirectory, defaultPermissionMode) {
  const options = requireOptions(value, "run options");
  assertKnownOptions(
    options,
    new Set(["attachments", "permissionMode", "signal", "onEvent"]),
    "run options",
  );
  if (options.onEvent !== undefined && typeof options.onEvent !== "function") {
    throw new HarnessValidationError("onEvent must be a function.");
  }
  return Object.freeze({
    attachments: normalizeAttachments(options.attachments, activeDirectory),
    permissionMode: permissionMode(options.permissionMode, defaultPermissionMode),
    signal: signalOption(options.signal),
    onEvent: options.onEvent ?? null,
  });
}

function validateSessionId(value) {
  if (typeof value !== "string" || SESSION_ID.test(value) === false) {
    throw new HarnessValidationError("sessionId is invalid.");
  }
  return value;
}

function exactKeys(value, allowed, required, label) {
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new HarnessProtocolError(`${label} contains an unknown field.`);
    }
  }
  for (const key of required) {
    if (!Object.prototype.hasOwnProperty.call(value, key)) {
      throw new HarnessProtocolError(`${label} is missing a required field.`);
    }
  }
}

function deepFreezeJson(value) {
  if (!isRecord(value) && !Array.isArray(value)) {
    return value;
  }
  const pending = [value];
  let visited = 0;
  while (pending.length > 0) {
    const item = pending.pop();
    if (Object.isFrozen(item)) {
      continue;
    }
    visited += 1;
    if (visited > 200_000) {
      throw new HarnessProtocolError("JSONL record is structurally too complex.");
    }
    for (const child of Object.values(item)) {
      if (isRecord(child) || Array.isArray(child)) {
        pending.push(child);
      }
    }
    Object.freeze(item);
  }
  return value;
}

function protocolIdentifier(value, name) {
  if (
    typeof value !== "string" ||
    value.length < 1 ||
    value.length > 160 ||
    value.includes("\0")
  ) {
    throw new HarnessProtocolError(`${name} is invalid.`);
  }
  return value;
}

function validateEvent(value) {
  if (!isRecord(value)) {
    throw new HarnessProtocolError("JSONL event must be an object.");
  }
  exactKeys(
    value,
    new Set([
      "schema",
      "event_id",
      "run_id",
      "turn_id",
      "sequence",
      "type",
      "timestamp",
      "causation_id",
      "payload",
    ]),
    [
      "schema",
      "event_id",
      "run_id",
      "turn_id",
      "sequence",
      "type",
      "timestamp",
      "payload",
    ],
    "JSONL event",
  );
  if (value.schema !== EVENT_SCHEMA) {
    throw new HarnessProtocolError("JSONL event schema is unsupported.");
  }
  if (typeof value.event_id !== "string" || !EVENT_ID.test(value.event_id)) {
    throw new HarnessProtocolError("JSONL event id is invalid.");
  }
  protocolIdentifier(value.run_id, "JSONL event run_id");
  protocolIdentifier(value.turn_id, "JSONL event turn_id");
  if (!Number.isSafeInteger(value.sequence) || value.sequence < 1) {
    throw new HarnessProtocolError("JSONL event sequence is invalid.");
  }
  if (typeof value.type !== "string" || !EVENT_TYPES.has(value.type)) {
    throw new HarnessProtocolError("JSONL event type is unsupported.");
  }
  if (
    typeof value.timestamp !== "string" ||
    value.timestamp.length > 64 ||
    !value.timestamp.includes("T") ||
    Number.isNaN(Date.parse(value.timestamp))
  ) {
    throw new HarnessProtocolError("JSONL event timestamp is invalid.");
  }
  if (
    value.causation_id !== undefined &&
    (typeof value.causation_id !== "string" ||
      value.causation_id.length < 1 ||
      value.causation_id.length > 160)
  ) {
    throw new HarnessProtocolError("JSONL event causation_id is invalid.");
  }
  if (!isRecord(value.payload)) {
    throw new HarnessProtocolError("JSONL event payload must be an object.");
  }
  if (value.type === "message.delta") {
    if (typeof value.payload.delta !== "string") {
      throw new HarnessProtocolError("message.delta payload is invalid.");
    }
    if (
      value.payload.channel !== undefined &&
      (typeof value.payload.channel !== "string" ||
        value.payload.channel.length > 40)
    ) {
      throw new HarnessProtocolError("message.delta channel is invalid.");
    }
  }
  const payload = deepFreezeJson(value.payload);
  return Object.freeze({ ...value, payload });
}

function validateUsage(value) {
  if (!isRecord(value)) {
    throw new HarnessProtocolError("exec result usage must be an object.");
  }
  const selected = {};
  for (const [key, count] of Object.entries(value)) {
    if (
      !key ||
      key.length > 120 ||
      !Number.isSafeInteger(count) ||
      count < 0
    ) {
      throw new HarnessProtocolError("exec result usage is invalid.");
    }
    selected[key] = count;
  }
  return Object.freeze(selected);
}

function validateExecResult(value) {
  if (!isRecord(value)) {
    throw new HarnessProtocolError("exec result must be an object.");
  }
  exactKeys(
    value,
    new Set([
      "schema",
      "session_id",
      "run_id",
      "turn_id",
      "status",
      "reason",
      "usage",
    ]),
    [
      "schema",
      "session_id",
      "run_id",
      "turn_id",
      "status",
      "reason",
      "usage",
    ],
    "exec result",
  );
  if (value.schema !== EXEC_RESULT_SCHEMA) {
    throw new HarnessProtocolError("exec result schema is unsupported.");
  }
  if (typeof value.session_id !== "string" || !SESSION_ID.test(value.session_id)) {
    throw new HarnessProtocolError("exec result session_id is invalid.");
  }
  if (typeof value.run_id !== "string" || !RUN_ID.test(value.run_id)) {
    throw new HarnessProtocolError("exec result run_id is invalid.");
  }
  if (typeof value.turn_id !== "string" || !TURN_ID.test(value.turn_id)) {
    throw new HarnessProtocolError("exec result turn_id is invalid.");
  }
  if (!EXPECTED_EXIT_CODE.has(value.status)) {
    throw new HarnessProtocolError("exec result status is unsupported.");
  }
  if (typeof value.reason !== "string" || value.reason.length > 1_000) {
    throw new HarnessProtocolError("exec result reason is invalid.");
  }
  return Object.freeze({
    sessionId: value.session_id,
    runId: value.run_id,
    turnId: value.turn_id,
    status: value.status,
    reason: value.reason,
    usage: validateUsage(value.usage),
  });
}

function decodeJsonLine(line, lineNumber) {
  let selected = line;
  if (selected.length > 0 && selected[selected.length - 1] === 13) {
    selected = selected.subarray(0, selected.length - 1);
  }
  if (selected.length === 0) {
    throw new HarnessProtocolError(`JSONL record ${lineNumber} is empty.`);
  }
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(selected);
  } catch {
    throw new HarnessProtocolError(
      `JSONL record ${lineNumber} is not valid UTF-8.`,
    );
  }
  if (text !== text.trim()) {
    throw new HarnessProtocolError(
      `JSONL record ${lineNumber} has non-canonical whitespace.`,
    );
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new HarnessProtocolError(
      `JSONL record ${lineNumber} is not valid JSON.`,
    );
  }
}

async function* jsonLines(readable, limits) {
  let pending = Buffer.alloc(0);
  let totalBytes = 0;
  let recordCount = 0;

  const decode = (line) => {
    if (line.length > limits.maxLineBytes) {
      throw new HarnessProtocolError("JSONL record exceeds maxLineBytes.");
    }
    recordCount += 1;
    if (recordCount > limits.maxRecords) {
      throw new HarnessProtocolError("JSONL stream exceeds maxRecords.");
    }
    return decodeJsonLine(line, recordCount);
  };

  for await (const rawChunk of readable) {
    const chunk = Buffer.isBuffer(rawChunk) ? rawChunk : Buffer.from(rawChunk);
    totalBytes += chunk.length;
    if (totalBytes > limits.maxOutputBytes) {
      throw new HarnessProtocolError("JSONL stream exceeds maxOutputBytes.");
    }
    let cursor = 0;
    while (cursor < chunk.length) {
      const newline = chunk.indexOf(10, cursor);
      if (newline === -1) {
        const remainder = chunk.subarray(cursor);
        if (pending.length + remainder.length > limits.maxLineBytes) {
          throw new HarnessProtocolError("JSONL record exceeds maxLineBytes.");
        }
        pending = Buffer.concat([pending, remainder]);
        break;
      }
      const fragment = chunk.subarray(cursor, newline);
      const line =
        pending.length === 0 ? fragment : Buffer.concat([pending, fragment]);
      pending = Buffer.alloc(0);
      yield decode(line);
      cursor = newline + 1;
    }
  }
  if (pending.length > 0) {
    yield decode(pending);
  }
}

class EventBuffer {
  #items = [];
  #waiters = [];
  #closed = false;
  #error = null;
  #consumed = false;
  #detached = false;

  push(event) {
    if (this.#closed || this.#detached) {
      return;
    }
    const waiter = this.#waiters.shift();
    if (waiter) {
      waiter.resolve({ done: false, value: event });
      return;
    }
    this.#items.push(event);
  }

  close() {
    if (this.#closed) {
      return;
    }
    this.#closed = true;
    this.#flush();
  }

  fail(error) {
    if (this.#closed) {
      return;
    }
    this.#error = error;
    this.#closed = true;
    this.#flush();
  }

  #flush() {
    while (this.#waiters.length > 0 && this.#items.length > 0) {
      this.#waiters.shift().resolve({ done: false, value: this.#items.shift() });
    }
    if (this.#items.length > 0) {
      return;
    }
    while (this.#waiters.length > 0) {
      const waiter = this.#waiters.shift();
      if (this.#error) {
        waiter.reject(this.#error);
      } else {
        waiter.resolve({ done: true, value: undefined });
      }
    }
  }

  iterator() {
    if (this.#consumed) {
      throw new HarnessSdkError(
        "A HarnessRunStream can only be iterated once.",
        "HARNESS_STREAM_CONSUMED",
      );
    }
    this.#consumed = true;
    return {
      next: () => {
        if (this.#items.length > 0) {
          return Promise.resolve({ done: false, value: this.#items.shift() });
        }
        if (this.#closed) {
          return this.#error
            ? Promise.reject(this.#error)
            : Promise.resolve({ done: true, value: undefined });
        }
        return new Promise((resolvePromise, rejectPromise) => {
          this.#waiters.push({ resolve: resolvePromise, reject: rejectPromise });
        });
      },
      return: () => {
        this.#detached = true;
        this.#items.length = 0;
        this.#flush();
        return Promise.resolve({ done: true, value: undefined });
      },
      [Symbol.asyncIterator]() {
        return this;
      },
    };
  }
}

export class HarnessRunStream {
  #buffer;

  constructor(buffer, result) {
    this.#buffer = buffer;
    this.result = result;
    Object.freeze(this);
  }

  [Symbol.asyncIterator]() {
    return this.#buffer.iterator();
  }
}

function appendFinalResponse(currentBytes, currentText, event, maxBytes) {
  if (
    event.type !== "message.delta" ||
    event.payload.channel === "internal"
  ) {
    return { bytes: currentBytes, text: currentText };
  }
  const delta = event.payload.delta;
  const deltaBytes = Buffer.byteLength(delta, "utf8");
  if (currentBytes + deltaBytes > maxBytes) {
    throw new HarnessProtocolError(
      "Assistant response exceeds maxResponseBytes.",
    );
  }
  return { bytes: currentBytes + deltaBytes, text: currentText + delta };
}

function safeSpawnCode(error) {
  const code = isRecord(error) ? error.code : null;
  return typeof code === "string" && /^[A-Z0-9_]{1,32}$/.test(code)
    ? code
    : "UNKNOWN";
}

async function executeTurn(config, sessionId, prompt, options, eventSink) {
  let signalInitiallyAborted = false;
  try {
    signalInitiallyAborted = options.signal?.aborted === true;
  } catch {
    throw new HarnessValidationError("signal state could not be read.");
  }
  if (signalInitiallyAborted) {
    throw new HarnessAbortError();
  }

  const args = [`--cwd=${config.cwd}`];
  if (config.activeDirectory !== config.cwd) {
    args.push(`--active-directory=${config.activeDirectory}`);
  }
  if (config.stateHome) {
    args.push(`--state-home=${config.stateHome}`);
  }
  if (config.worktreeHome) {
    args.push(`--worktree-home=${config.worktreeHome}`);
  }
  if (config.apiKeyFile) {
    args.push(`--api-key-file=${config.apiKeyFile}`);
  }
  if (config.model) {
    args.push(`--model=${config.model}`);
  }
  args.push(`--permissions=${options.permissionMode}`);
  args.push(`--deadline=${config.deadlineSeconds}`);
  args.push(`--max-steps=${config.maxSteps}`);
  args.push("exec", "-");
  if (sessionId !== null) {
    args.push(`--resume=${sessionId}`);
  }
  for (const attachment of options.attachments) {
    args.push(`--attach=${attachment}`);
  }
  args.push("--jsonl");

  let child = null;
  let spawnFailure = null;
  let aborted = false;
  let abortSignalSent = false;
  let abortTimer = null;
  let protocolKillTimer = null;
  let transportTimer = null;
  let transportKillTimer = null;
  let resultCloseTimer = null;
  let resultCloseKillTimer = null;
  let transportTimedOut = false;
  let resultCloseTimedOut = false;
  let resultCommitted = false;
  let childClosed = false;

  const removeAbortListener = () => {
    if (options.signal === null) {
      return;
    }
    try {
      absorbThenable(options.signal.removeEventListener("abort", abort));
    } catch {
      // Listener cleanup is best effort and never changes the run outcome.
    }
  };
  const abort = () => {
    if (resultCommitted) {
      return;
    }
    aborted = true;
    if (child === null || abortSignalSent) {
      return;
    }
    abortSignalSent = true;
    try {
      child.kill("SIGINT");
    } catch {
      // The close result below remains authoritative.
    }
    abortTimer = setTimeout(() => {
      try {
        child?.kill("SIGKILL");
      } catch {
        // The process may already be gone.
      }
    }, config.limits.abortGraceMs);
    abortTimer.unref?.();
  };

  try {
    const listenerResult = options.signal?.addEventListener("abort", abort, {
      once: true,
    });
    requireVoidCallbackResult(listenerResult, "signal listener registration");
  } catch {
    removeAbortListener();
    throw new HarnessValidationError("signal listener registration failed.");
  }

  try {
    child = spawn(config.harnessPath, args, {
      cwd: config.cwd,
      env: config.env,
      shell: false,
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
  } catch (error) {
    removeAbortListener();
    throw new HarnessProcessError(
      `Agent Harness process could not be started (${safeSpawnCode(error)}).`,
    );
  }

  child.once("error", (error) => {
    spawnFailure = error;
  });
  child.stderr.on("error", () => {});
  child.stderr.resume();
  child.stdin.on("error", () => {});

  const closePromise = new Promise((resolvePromise) => {
    child.once("close", (exitCode, exitSignal) => {
      childClosed = true;
      resolvePromise({ exitCode, exitSignal });
    });
  });

  transportTimer = setTimeout(() => {
    transportTimedOut = true;
    try {
      child?.kill("SIGINT");
    } catch {
      // The process may already be gone.
    }
    transportKillTimer = setTimeout(() => {
      try {
        child?.kill("SIGKILL");
      } catch {
        // The process may already be gone.
      }
    }, config.limits.abortGraceMs);
    transportKillTimer.unref?.();
  }, config.limits.transportTimeoutMs);
  transportTimer.unref?.();

  if (aborted) {
    abort();
  } else {
    try {
      if (options.signal?.aborted === true) {
        abort();
      }
    } catch {
      aborted = true;
      abort();
    }
  }

  child.stdin.end(prompt, "utf8");

  let finalRecord = null;
  let expectedSequence = 1;
  let runId = null;
  let turnId = null;
  let terminalType = null;
  let finalResponse = "";
  let finalResponseBytes = 0;
  let streamFailure = null;

  try {
    for await (const record of jsonLines(child.stdout, config.limits)) {
      if (finalRecord !== null) {
        throw new HarnessProtocolError("JSONL data appeared after the exec result.");
      }
      if (isRecord(record) && record.schema === EXEC_RESULT_SCHEMA) {
        finalRecord = validateExecResult(record);
        if (transportTimer !== null) {
          clearTimeout(transportTimer);
          transportTimer = null;
        }
        resultCloseTimer = setTimeout(() => {
          resultCloseTimedOut = true;
          try {
            child?.kill("SIGINT");
          } catch {
            // The process may already be gone.
          }
          resultCloseKillTimer = setTimeout(() => {
            try {
              child?.kill("SIGKILL");
            } catch {
              // The process may already be gone.
            }
          }, config.limits.abortGraceMs);
          resultCloseKillTimer.unref?.();
        }, config.limits.resultCloseGraceMs);
        resultCloseTimer.unref?.();
        resultCommitted =
          terminalType !== null &&
          runId !== null &&
          turnId !== null &&
          finalRecord.runId === runId &&
          finalRecord.turnId === turnId &&
          finalRecord.status === TERMINAL_EVENT_STATUS.get(terminalType) &&
          (sessionId === null || finalRecord.sessionId === sessionId);
        continue;
      }
      const event = validateEvent(record);
      if (terminalType !== null) {
        throw new HarnessProtocolError("JSONL event appeared after a terminal event.");
      }
      if (event.sequence !== expectedSequence) {
        throw new HarnessProtocolError("JSONL event sequence is not contiguous.");
      }
      if (expectedSequence === 1 && event.type !== "run.started") {
        throw new HarnessProtocolError("JSONL stream must begin with run.started.");
      }
      expectedSequence += 1;
      if (runId === null) {
        runId = event.run_id;
        turnId = event.turn_id;
      } else if (event.run_id !== runId || event.turn_id !== turnId) {
        throw new HarnessProtocolError("JSONL event identifiers changed mid-stream.");
      }
      if (TERMINAL_EVENT_STATUS.has(event.type)) {
        terminalType = event.type;
      }
      const appended = appendFinalResponse(
        finalResponseBytes,
        finalResponse,
        event,
        config.limits.maxResponseBytes,
      );
      finalResponseBytes = appended.bytes;
      finalResponse = appended.text;
      try {
        absorbThenable(options.onEvent?.(event));
      } catch {
        // Event callbacks are downstream observation only. They cannot revoke
        // or interrupt the authoritative child process and durable run.
      }
      eventSink?.(event);
    }
    if (finalRecord === null) {
      // EOF makes a future exec-result impossible. Give an already-exiting
      // child one event-loop turn to publish close, then terminate a process
      // that deliberately kept running after closing stdout.
      await Promise.race([
        closePromise,
        new Promise((resolvePromise) => setImmediate(resolvePromise)),
      ]);
      if (!childClosed) {
        try {
          child.kill("SIGINT");
        } catch {
          // The process may already be gone.
        }
        protocolKillTimer = setTimeout(() => {
          try {
            child.kill("SIGKILL");
          } catch {
            // The process may already be gone.
          }
        }, config.limits.abortGraceMs);
        protocolKillTimer.unref?.();
      }
    }
  } catch (error) {
    streamFailure = error;
    try {
      child.kill("SIGINT");
    } catch {
      // The process may already be gone.
    }
    protocolKillTimer = setTimeout(() => {
      try {
        child.kill("SIGKILL");
      } catch {
        // The process may already be gone.
      }
    }, config.limits.abortGraceMs);
    protocolKillTimer.unref?.();
  }

  const { exitCode, exitSignal } = await closePromise;
  removeAbortListener();
  if (abortTimer !== null) {
    clearTimeout(abortTimer);
  }
  if (protocolKillTimer !== null) {
    clearTimeout(protocolKillTimer);
  }
  if (transportTimer !== null) {
    clearTimeout(transportTimer);
  }
  if (transportKillTimer !== null) {
    clearTimeout(transportKillTimer);
  }
  if (resultCloseTimer !== null) {
    clearTimeout(resultCloseTimer);
  }
  if (resultCloseKillTimer !== null) {
    clearTimeout(resultCloseKillTimer);
  }

  let outcomeError = null;
  if (spawnFailure !== null) {
    outcomeError = new HarnessProcessError(
      `Agent Harness process could not be started (${safeSpawnCode(spawnFailure)}).`,
      { exitCode, signal: exitSignal },
    );
  } else if (transportTimedOut) {
    outcomeError = new HarnessProtocolError(
      "Agent Harness process exceeded the transport timeout before an exec result.",
    );
  } else if (streamFailure !== null) {
    outcomeError = streamFailure instanceof HarnessSdkError
      ? streamFailure
      : new HarnessProtocolError("Agent Harness JSONL stream failed.");
  } else if (resultCloseTimedOut) {
    outcomeError = new HarnessProtocolError(
      "Agent Harness process did not exit after the exec result.",
    );
  } else if (finalRecord === null) {
    outcomeError = new HarnessProcessError(
      "Agent Harness process exited before producing a valid result.",
      { exitCode, signal: exitSignal },
    );
  } else if (terminalType === null || runId === null || turnId === null) {
    outcomeError = new HarnessProtocolError(
      "JSONL stream has no terminal run event.",
    );
  } else if (
    finalRecord.runId !== runId ||
    finalRecord.turnId !== turnId ||
    finalRecord.status !== TERMINAL_EVENT_STATUS.get(terminalType)
  ) {
    outcomeError = new HarnessProtocolError(
      "Exec result does not match the event stream.",
    );
  } else if (sessionId !== null && finalRecord.sessionId !== sessionId) {
    outcomeError = new HarnessProtocolError(
      "Exec result session does not match the thread.",
    );
  } else {
    const expectedExitCode = EXPECTED_EXIT_CODE.get(finalRecord.status);
    if (exitSignal !== null || exitCode !== expectedExitCode) {
      outcomeError = new HarnessProtocolError(
        "Process exit does not match the exec result.",
      );
    }
  }
  if (outcomeError !== null) {
    if (aborted && !resultCloseTimedOut && !transportTimedOut) {
      throw new HarnessAbortError();
    }
    throw outcomeError;
  }
  return Object.freeze({ ...finalRecord, finalResponse });
}

export class HarnessThread {
  #client;
  #sessionId;
  #permissionMode;
  #tail = Promise.resolve();

  constructor(client, sessionId, selectedPermissionMode) {
    this.#client = client;
    this.#sessionId = sessionId;
    this.#permissionMode = selectedPermissionMode;
  }

  get id() {
    return this.#sessionId;
  }

  get sessionId() {
    return this.#sessionId;
  }

  get permissionMode() {
    return this.#permissionMode;
  }

  #schedule(operation) {
    const scheduled = this.#tail.then(operation, operation);
    this.#tail = scheduled.then(
      () => undefined,
      () => undefined,
    );
    return scheduled;
  }

  #run(prompt, options, eventSink) {
    return this.#schedule(async () => {
      const result = await this.#client._execute(
        this.#sessionId,
        prompt,
        options,
        eventSink,
      );
      if (this.#sessionId === null) {
        this.#sessionId = result.sessionId;
      } else if (result.sessionId !== this.#sessionId) {
        throw new HarnessProtocolError("Thread session identity changed.");
      }
      return result;
    });
  }

  run(prompt, options = undefined) {
    const selectedPrompt = normalizePrompt(prompt);
    const selectedOptions = this.#client._runOptions(
      options,
      this.#permissionMode,
    );
    return this.#run(selectedPrompt, selectedOptions, null);
  }

  runStream(prompt, options = undefined) {
    const selectedPrompt = normalizePrompt(prompt);
    const selectedOptions = this.#client._runOptions(
      options,
      this.#permissionMode,
    );
    const buffer = new EventBuffer();
    const result = this.#run(selectedPrompt, selectedOptions, (event) => {
      buffer.push(event);
    }).then(
      (value) => {
        buffer.close();
        return value;
      },
      (error) => {
        buffer.fail(error);
        throw error;
      },
    );
    // A caller may consume only the event iterator. Mark the public result
    // promise handled internally while preserving its rejection for awaiters.
    result.catch(() => {});
    return new HarnessRunStream(buffer, result);
  }
}

export class HarnessClient {
  #config;

  constructor(value = undefined) {
    const options = requireOptions(value, "HarnessClient options");
    assertKnownOptions(
      options,
      new Set([
        "harnessPath",
        "cwd",
        "activeDirectory",
        "stateHome",
        "worktreeHome",
        "apiKeyFile",
        "model",
        "permissionMode",
        "deadlineSeconds",
        "maxSteps",
        "env",
        "maxLineBytes",
        "maxRecords",
        "maxOutputBytes",
        "maxResponseBytes",
        "abortGraceMs",
        "resultCloseGraceMs",
        "transportTimeoutMs",
      ]),
      "HarnessClient options",
    );

    const cwd = existingDirectory(options.cwd ?? process.cwd(), "cwd");
    const activeDirectory =
      options.activeDirectory === undefined
        ? cwd
        : existingDirectory(options.activeDirectory, "activeDirectory", cwd);
    if (!pathInside(cwd, activeDirectory)) {
      throw new HarnessValidationError(
        "activeDirectory must stay inside cwd.",
      );
    }
    const deadlineSeconds = boundedNumber(
      options.deadlineSeconds,
      "deadlineSeconds",
      0.05,
      86_400,
      180,
    );
    const abortGraceMs = boundedInteger(
      options.abortGraceMs,
      "abortGraceMs",
      50,
      30_000,
      DEFAULT_LIMITS.abortGraceMs,
    );
    const maxLineBytes = boundedInteger(
      options.maxLineBytes,
      "maxLineBytes",
      1_024,
      16 * 1024 * 1024,
      DEFAULT_LIMITS.maxLineBytes,
    );
    const maxOutputBytes = boundedInteger(
      options.maxOutputBytes,
      "maxOutputBytes",
      maxLineBytes,
      256 * 1024 * 1024,
      DEFAULT_LIMITS.maxOutputBytes,
    );
    const limits = Object.freeze({
      maxLineBytes,
      maxRecords: boundedInteger(
        options.maxRecords,
        "maxRecords",
        1,
        100_000,
        DEFAULT_LIMITS.maxRecords,
      ),
      maxOutputBytes,
      maxResponseBytes: boundedInteger(
        options.maxResponseBytes,
        "maxResponseBytes",
        1_024,
        64 * 1024 * 1024,
        DEFAULT_LIMITS.maxResponseBytes,
      ),
      abortGraceMs,
      resultCloseGraceMs: boundedInteger(
        options.resultCloseGraceMs,
        "resultCloseGraceMs",
        50,
        30_000,
        DEFAULT_LIMITS.resultCloseGraceMs,
      ),
      transportTimeoutMs: boundedInteger(
        options.transportTimeoutMs,
        "transportTimeoutMs",
        100,
        86_500_000,
        Math.ceil(deadlineSeconds * 1_000) + abortGraceMs + 5_000,
      ),
    });

    this.#config = Object.freeze({
      cwd,
      activeDirectory,
      harnessPath: executable(options.harnessPath, cwd),
      stateHome: optionalAbsolutePath(options.stateHome, "stateHome", cwd),
      worktreeHome: optionalAbsolutePath(
        options.worktreeHome,
        "worktreeHome",
        cwd,
      ),
      apiKeyFile: optionalAbsolutePath(
        options.apiKeyFile,
        "apiKeyFile",
        cwd,
      ),
      model:
        options.model === undefined
          ? null
          : boundedString(options.model, "model", { max: 160 }),
      permissionMode: permissionMode(options.permissionMode),
      deadlineSeconds,
      maxSteps: boundedInteger(options.maxSteps, "maxSteps", 1, 64, 12),
      env: normalizedEnvironment(options.env),
      limits,
    });
  }

  get cwd() {
    return this.#config.cwd;
  }

  get activeDirectory() {
    return this.#config.activeDirectory;
  }

  get permissionMode() {
    return this.#config.permissionMode;
  }

  startThread(value = undefined) {
    const options = requireOptions(value, "thread options");
    assertKnownOptions(options, new Set(["permissionMode"]), "thread options");
    return new HarnessThread(
      this,
      null,
      permissionMode(options.permissionMode, this.#config.permissionMode),
    );
  }

  resumeThread(sessionId, value = undefined) {
    const selectedSessionId = validateSessionId(sessionId);
    const options = requireOptions(value, "thread options");
    assertKnownOptions(options, new Set(["permissionMode"]), "thread options");
    return new HarnessThread(
      this,
      selectedSessionId,
      permissionMode(options.permissionMode, this.#config.permissionMode),
    );
  }

  _runOptions(value, defaultPermissionMode) {
    return normalizeRunOptions(
      value,
      this.#config.activeDirectory,
      defaultPermissionMode,
    );
  }

  _execute(sessionId, prompt, options, eventSink) {
    return executeTurn(
      this.#config,
      sessionId,
      prompt,
      options,
      eventSink,
    );
  }
}
