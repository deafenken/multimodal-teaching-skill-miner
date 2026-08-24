import assert from "node:assert/strict";
import {
  chmodSync,
  existsSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  realpathSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  HarnessAbortError,
  HarnessClient,
  HarnessProcessError,
  HarnessProtocolError,
  HarnessValidationError,
} from "../index.js";

const here = dirname(fileURLToPath(import.meta.url));
const fakeHarness = resolve(here, "fake-harness.mjs");
chmodSync(fakeHarness, 0o755);

function fixture(options = {}) {
  const root = mkdtempSync(join(tmpdir(), "agent-harness-sdk-"));
  const activeDirectory = join(root, "src");
  mkdirSync(activeDirectory);
  const logPath = join(root, "fake.jsonl");
  const client = new HarnessClient({
    harnessPath: fakeHarness,
    cwd: root,
    activeDirectory,
    permissionMode: "workspace-write",
    deadlineSeconds: 5,
    maxSteps: 4,
    abortGraceMs: 100,
    env: { FAKE_HARNESS_LOG: logPath },
    ...options,
  });
  return { root, activeDirectory, logPath, client };
}

function logRecords(path) {
  return readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

test("startThread is lazy and run uses stdin, ordered global options, and repeated attachments", async () => {
  const { activeDirectory, logPath, client } = fixture();
  const thread = client.startThread();
  assert.equal(thread.id, null);
  assert.equal(thread.sessionId, null);

  const observed = [];
  const result = await thread.run("private prompt text", {
    attachments: ["one.md", "nested/two.png"],
    onEvent: (event) => observed.push(event.type),
  });

  assert.equal(result.status, "completed");
  assert.equal(result.finalResponse, "hello world");
  assert.deepEqual(result.usage, { input_tokens: 3, output_tokens: 2 });
  assert.equal(thread.id, result.sessionId);
  assert.ok(observed.includes("message.delta"));
  assert.ok(!result.finalResponse.includes("hidden"));

  const invocation = logRecords(logPath).find((item) => item.kind === "start");
  const execIndex = invocation.args.indexOf("exec");
  assert.ok(execIndex > 0);
  assert.equal(invocation.args[execIndex + 1], "-");
  assert.ok(invocation.args.slice(0, execIndex).some((value) => value.startsWith("--cwd=")));
  assert.ok(
    invocation.args
      .slice(0, execIndex)
      .some((value) => value === "--permissions=workspace-write"),
  );
  assert.ok(!invocation.args.some((value) => value.includes("private prompt text")));
  assert.equal(invocation.prompt, "private prompt text");
  assert.deepEqual(invocation.attachments, [
    join(realpathSync(activeDirectory), "one.md"),
    join(realpathSync(activeDirectory), "nested/two.png"),
  ]);
});

test("resumeThread preserves a validated session ID", async () => {
  const { client } = fixture();
  const sessionId = "session_cccccccccccccccccccccccccccccccc";
  const thread = client.resumeThread(sessionId, { permissionMode: "read-only" });
  assert.equal(thread.id, sessionId);
  assert.equal(thread.permissionMode, "read-only");
  const result = await thread.run("resume");
  assert.equal(result.sessionId, sessionId);
  assert.equal(thread.id, sessionId);
});

test("turns on one thread are serialized in submission order", async () => {
  const { client, logPath } = fixture();
  const thread = client.startThread();
  const slow = thread.run("slow-one");
  const fast = thread.run("fast-two");
  await Promise.all([slow, fast]);

  assert.deepEqual(
    logRecords(logPath).map((item) => `${item.kind}:${item.prompt}`),
    ["start:slow-one", "end:slow-one", "start:fast-two", "end:fast-two"],
  );
});

test("runStream exposes a single-use typed event stream and verified result", async () => {
  const { client } = fixture();
  const stream = client.startThread().runStream("stream");
  const eventTypes = [];
  for await (const event of stream) {
    eventTypes.push(event.type);
  }
  const result = await stream.result;
  assert.equal(result.finalResponse, "hello world");
  assert.equal(eventTypes[0], "run.started");
  assert.equal(eventTypes.at(-1), "run.completed");
  assert.throws(() => stream[Symbol.asyncIterator](), /only be iterated once/);
});

test("AbortSignal sends SIGINT and rejects with a predictable AbortError", async () => {
  const { client, logPath } = fixture();
  const controller = new AbortController();
  let markStarted;
  const started = new Promise((resolvePromise) => {
    markStarted = resolvePromise;
  });
  const pending = client.startThread().run("abort", {
    signal: controller.signal,
    onEvent: (event) => {
      if (event.type === "run.started") {
        markStarted();
      }
    },
  });
  await started;
  await new Promise((resolvePromise) => setTimeout(resolvePromise, 20));
  controller.abort();

  await assert.rejects(pending, (error) => {
    assert.ok(error instanceof HarnessAbortError);
    assert.equal(error.name, "AbortError");
    assert.equal(error.code, "ABORT_ERR");
    return true;
  });
  assert.ok(
    logRecords(logPath).some(
      (item) => item.kind === "signal" && item.signal === "SIGINT",
    ),
  );
});

test("a verified terminal result wins over a late AbortSignal race", async () => {
  const { client } = fixture();
  const controller = new AbortController();
  const pending = client.startThread().run("commit-race", {
    signal: controller.signal,
  });
  setTimeout(() => controller.abort(), 75);
  const result = await pending;
  assert.equal(result.status, "completed");
  assert.equal(result.finalResponse, "hello world");
});

test("a child that hangs after its exec result is terminated and fails closed", async () => {
  const { client, logPath } = fixture({
    abortGraceMs: 50,
    resultCloseGraceMs: 100,
  });
  const startedAt = Date.now();
  await assert.rejects(client.startThread().run("commit-hang"), (error) => {
    assert.ok(error instanceof HarnessProtocolError);
    assert.match(error.message, /did not exit after the exec result/);
    return true;
  });
  assert.ok(Date.now() - startedAt < 2_000);
  assert.ok(
    logRecords(logPath).some(
      (item) => item.kind === "signal" && item.signal === "SIGINT",
    ),
  );
});

test("pre-result and stdout-EOF child hangs are both bounded", async () => {
  const beforeResult = fixture({
    abortGraceMs: 50,
    transportTimeoutMs: 100,
  });
  const beforeStartedAt = Date.now();
  await assert.rejects(
    beforeResult.client.startThread().run("pre-result-hang"),
    (error) => {
      assert.ok(error instanceof HarnessProtocolError);
      assert.match(error.message, /transport timeout before an exec result/);
      return true;
    },
  );
  assert.ok(Date.now() - beforeStartedAt < 2_000);
  assert.ok(
    logRecords(beforeResult.logPath).some(
      (item) => item.kind === "signal" && item.signal === "SIGINT",
    ),
  );

  const afterEof = fixture({
    abortGraceMs: 50,
    transportTimeoutMs: 1_000,
  });
  const eofStartedAt = Date.now();
  await assert.rejects(
    afterEof.client.startThread().run("eof-hang"),
    HarnessProcessError,
  );
  assert.ok(Date.now() - eofStartedAt < 500);
  assert.ok(
    logRecords(afterEof.logPath).some(
      (item) => item.kind === "signal" && item.signal === "SIGINT",
    ),
  );
});

test("a matching result and exit win when abort arrives between terminal and result", async () => {
  const { client, logPath } = fixture();
  const controller = new AbortController();
  let markTerminal;
  const terminal = new Promise((resolvePromise) => {
    markTerminal = resolvePromise;
  });
  const pending = client.startThread().run("terminal-race", {
    signal: controller.signal,
    onEvent: (event) => {
      if (event.type === "run.completed") {
        markTerminal();
      }
    },
  });
  await terminal;
  controller.abort();
  const result = await pending;
  assert.equal(result.status, "completed");
  assert.ok(
    logRecords(logPath).some(
      (item) =>
        item.kind === "signal" && item.signal === "SIGINT" && item.ignored === true,
    ),
  );
});

test("onEvent failures cannot interrupt an authoritative run", async () => {
  const { client } = fixture();
  let calls = 0;
  const result = await client.startThread().run("callback", {
    onEvent: () => {
      calls += 1;
      throw new Error("observer failure");
    },
  });
  assert.ok(calls > 0);
  assert.equal(result.status, "completed");
});

test("async onEvent rejection is observation-only and never becomes unhandled", async () => {
  const { client } = fixture();
  const result = await client.startThread().run("async-callback", {
    onEvent: async () => {
      throw new Error("private async observer failure");
    },
  });
  await new Promise((resolvePromise) => setImmediate(resolvePromise));
  assert.equal(result.status, "completed");
});

test("deadline failures follow the public failed terminal contract", async () => {
  const { client } = fixture();
  const result = await client.startThread().run("deadline");
  assert.equal(result.status, "failed");
  assert.equal(result.reason, "deadline_exceeded");
});

test("stderr content is never copied into process errors", async () => {
  const { client } = fixture();
  await assert.rejects(client.startThread().run("stderr-secret"), (error) => {
    assert.ok(error instanceof HarnessProcessError);
    assert.equal(error.exitCode, 78);
    assert.ok(!error.message.includes("do-not-copy-this-secret"));
    assert.ok(!JSON.stringify(error).includes("do-not-copy-this-secret"));
    return true;
  });
});

test("invalid, oversized, and identity-mismatched JSONL fail closed", async () => {
  const invalid = fixture();
  await assert.rejects(
    invalid.client.startThread().run("invalid-json"),
    HarnessProtocolError,
  );

  const oversized = fixture({ maxLineBytes: 1_024 });
  await assert.rejects(
    oversized.client.startThread().run("oversized"),
    HarnessProtocolError,
  );

  const mismatch = fixture();
  await assert.rejects(
    mismatch.client.startThread().run("mismatch"),
    HarnessProtocolError,
  );
});

test("configuration, session, prompt, and attachment options are validated", async () => {
  const { client, root } = fixture();
  assert.throws(
    () => new HarnessClient({ cwd: root, permissionMode: "unsafe" }),
    HarnessValidationError,
  );
  assert.throws(
    () => client.resumeThread("../../private"),
    HarnessValidationError,
  );
  assert.throws(() => client.startThread().run("   "), HarnessValidationError);
  assert.throws(
    () =>
      client.startThread().run("too many", {
        attachments: Array.from({ length: 9 }, (_, index) => `${index}.md`),
      }),
    HarnessValidationError,
  );
  assert.throws(
    () => client.startThread({ unexpected: true }),
    HarnessValidationError,
  );
});

test("a hostile signal cannot spawn or orphan a child process", async () => {
  const { client, logPath } = fixture();
  const signal = {
    aborted: false,
    addEventListener() {
      throw new Error("sensitive listener failure");
    },
    removeEventListener() {},
  };
  await assert.rejects(
    client.startThread().run("must-not-start", { signal }),
    (error) => {
      assert.ok(error instanceof HarnessValidationError);
      assert.ok(!error.message.includes("sensitive"));
      return true;
    },
  );
  assert.equal(existsSync(logPath), false);
});

test("rejected signal-listener thenables are absorbed before and after spawn", async () => {
  const registration = fixture();
  const rejectedRegistrationSignal = {
    aborted: false,
    addEventListener() {
      return Promise.reject(new Error("private registration rejection"));
    },
    removeEventListener() {
      return Promise.reject(new Error("private registration cleanup rejection"));
    },
  };
  await assert.rejects(
    registration.client.startThread().run("must-not-start", {
      signal: rejectedRegistrationSignal,
    }),
    HarnessValidationError,
  );
  await new Promise((resolvePromise) => setImmediate(resolvePromise));
  assert.equal(existsSync(registration.logPath), false);

  const cleanup = fixture();
  const controller = new AbortController();
  const rejectedCleanupSignal = {
    get aborted() {
      return controller.signal.aborted;
    },
    addEventListener: controller.signal.addEventListener.bind(controller.signal),
    removeEventListener() {
      return Promise.reject(new Error("private cleanup rejection"));
    },
  };
  const result = await cleanup.client.startThread().run("cleanup", {
    signal: rejectedCleanupSignal,
  });
  await new Promise((resolvePromise) => setImmediate(resolvePromise));
  assert.equal(result.status, "completed");
});

test("listener cleanup failures cannot revoke a verified result", async () => {
  const controller = new AbortController();
  const signal = {
    get aborted() {
      return controller.signal.aborted;
    },
    addEventListener: controller.signal.addEventListener.bind(controller.signal),
    removeEventListener() {
      throw new Error("cleanup failure");
    },
  };
  const { client } = fixture();
  const result = await client.startThread().run("cleanup", { signal });
  assert.equal(result.status, "completed");
});

test("package metadata declares Node 18 ESM with no runtime dependencies", () => {
  const packageJson = JSON.parse(
    readFileSync(resolve(here, "../package.json"), "utf8"),
  );
  assert.equal(packageJson.type, "module");
  assert.equal(packageJson.engines.node, ">=18");
  assert.equal(packageJson.version, "2.7.0");
  assert.equal(packageJson.dependencies, undefined);
  assert.deepEqual(packageJson.devDependencies, { typescript: "5.9.3" });
});
