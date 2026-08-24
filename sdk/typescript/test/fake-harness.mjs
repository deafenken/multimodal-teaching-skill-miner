#!/usr/bin/env node

import { appendFileSync } from "node:fs";

const args = process.argv.slice(2);
const execIndex = args.indexOf("exec");
const logPath = process.env.FAKE_HARNESS_LOG;

function log(value) {
  if (logPath) {
    appendFileSync(logPath, `${JSON.stringify(value)}\n`, "utf8");
  }
}

function option(prefix) {
  const selected = args.find((value) => value.startsWith(`${prefix}=`));
  return selected ? selected.slice(prefix.length + 1) : null;
}

function options(prefix) {
  return args
    .filter((value) => value.startsWith(`${prefix}=`))
    .map((value) => value.slice(prefix.length + 1));
}

if (execIndex < 1 || args[execIndex + 1] !== "-") {
  process.stderr.write("invalid fake invocation\n");
  process.exit(64);
}
if (args.slice(execIndex + 1).some((value) => value.startsWith("--cwd="))) {
  process.stderr.write("global option appeared after exec\n");
  process.exit(64);
}

let prompt = "";
for await (const chunk of process.stdin) {
  prompt += chunk.toString("utf8");
}

const sessionId = option("--resume") ?? "session_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const runId = "run_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const turnId = "turn_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const timestamp = "2026-08-24T00:00:00Z";

log({
  kind: "start",
  prompt,
  args,
  attachments: options("--attach"),
  permissionMode: option("--permissions"),
  sessionId,
});

if (["pre-result-hang", "eof-hang", "commit-hang"].includes(prompt)) {
  process.on("SIGINT", () => {
    log({ kind: "signal", prompt, signal: "SIGINT" });
    process.exit(2);
  });
}

if (prompt === "stderr-secret") {
  process.stderr.write("API_KEY=do-not-copy-this-secret\n");
  log({ kind: "end", prompt });
  process.exit(78);
}

if (prompt === "invalid-json") {
  process.stdout.write("not-json\n");
  log({ kind: "end", prompt });
  process.exit(4);
}

if (prompt === "oversized") {
  process.stdout.write(`${"x".repeat(8_192)}\n`);
  log({ kind: "end", prompt });
  process.exit(4);
}

if (prompt === "pre-result-hang") {
  setInterval(() => {}, 1_000);
  await new Promise(() => {});
}

if (prompt === "eof-hang") {
  process.stdout.end();
  setInterval(() => {}, 1_000);
  await new Promise(() => {});
}

const emit = (sequence, type, payload = {}) => {
  process.stdout.write(
    `${JSON.stringify({
      schema: "agent_harness.event.v1",
      event_id: sequence.toString(16).padStart(32, "0"),
      run_id: runId,
      turn_id: turnId,
      sequence,
      type,
      timestamp,
      payload,
    })}\n`,
  );
};

if (prompt === "abort") {
  emit(1, "run.started", { resumed: option("--resume") !== null });
  process.on("SIGINT", () => {
    log({ kind: "signal", prompt, signal: "SIGINT" });
    process.exit(2);
  });
  setInterval(() => {}, 1_000);
} else {
  emit(1, "run.started", { resumed: option("--resume") !== null });
  emit(2, "message.start", { channel: "assistant" });
  emit(3, "reasoning.delta", { channel: "internal", delta: "hidden" });
  emit(4, "message.delta", { channel: "assistant", delta: "hello " });
  emit(5, "message.delta", { channel: "assistant", delta: "world" });
  emit(6, "message.end", { chars: 11 });

  if (prompt === "slow-one") {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 150));
  }

  if (prompt === "terminal-race") {
    process.on("SIGINT", () => {
      log({ kind: "signal", prompt, signal: "SIGINT", ignored: true });
    });
  }
  const deadline = prompt === "deadline";
  if (deadline) {
    emit(7, "run.failed", {
      error_code: "deadline_exceeded",
      reason_code: "deadline_exceeded",
    });
  } else {
    emit(7, "run.completed", { result: { status: "completed" } });
  }
  if (prompt === "terminal-race") {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 40));
  }
  const resultRunId = prompt === "mismatch" ? "run_wrong" : runId;
  process.stdout.write(
    `${JSON.stringify({
      schema: "agent_harness.exec_result.v1",
      session_id: sessionId,
      run_id: resultRunId,
      turn_id: turnId,
      status: deadline ? "failed" : "completed",
      reason: deadline ? "deadline_exceeded" : "",
      usage: { input_tokens: 3, output_tokens: 2 },
    })}\n`,
  );
  if (prompt === "commit-race") {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 150));
  }
  if (prompt === "commit-hang") {
    setInterval(() => {}, 1_000);
  }
  log({ kind: "end", prompt });
  if (deadline) {
    process.exitCode = 4;
  }
}
