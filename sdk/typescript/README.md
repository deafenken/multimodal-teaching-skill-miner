# Agent Harness SDK for TypeScript

Dependency-free Node.js 18+ bindings for the Agent Harness headless JSONL
interface. The runtime is strict ESM JavaScript and ships first-party TypeScript
declarations.

## Install

```bash
npm install ./sdk/typescript
```

This repository validates a publishable package but does not claim that
`@agent-harness/sdk` has been uploaded to a public registry. After an operator
publishes the reviewed tarball to that name, consumers can install the scoped
registry package normally.

The Python `harness` executable must be installed separately. Pin the executable
you intend to trust, or pass an absolute `harnessPath`; the SDK never invokes a
shell.

## Run a thread

```ts
import { HarnessClient } from "@agent-harness/sdk";

const client = new HarnessClient({
  harnessPath: "/opt/agent-harness/bin/harness",
  cwd: "/work/project",
  activeDirectory: "/work/project/packages/app",
  permissionMode: "workspace-write",
});

const thread = client.startThread();
console.log(thread.id); // null: startThread is intentionally lazy

const first = await thread.run("Inspect the failing test", {
  attachments: ["failure.md"],
});

console.log(first.finalResponse);
console.log(thread.id); // durable session ID from the first valid exec result

const resumed = client.resumeThread(first.sessionId);
const next = await resumed.run("Implement the smallest safe fix");
```

`startThread()` does not invent an ID or spawn a process. The CLI creates the
durable session on the first turn, and the SDK adopts only the session ID from a
fully validated `agent_harness.exec_result.v1` record. A subprocess failure
before that record can leave CLI-owned recovery state, but the lazy SDK thread
remains ID-less because stderr is not an identity channel.

Calls on the same `HarnessThread` are serialized in submission order. Separate
thread handles are not serialized by this SDK, but the Harness workspace lock
still governs the child processes and can make them wait rather than execute
turns concurrently in one workspace.

## Stream typed events

```ts
const stream = thread.runStream("Explain the change before editing");

for await (const event of stream) {
  if (event.type === "message.delta") {
    process.stdout.write(event.payload.delta);
  }
}

const result = await stream.result;
console.log(result.status, result.usage);
```

A `HarnessRunStream` can be iterated once. Its `result` promise resolves only
after the terminal event, final record, identifiers, status, and subprocess exit
code agree. Stopping iteration does not stop the run; pass an `AbortSignal` when
you need cancellation.

```ts
const controller = new AbortController();
const pending = thread.run("Long task", { signal: controller.signal });
controller.abort();

try {
  await pending;
} catch (error) {
  if (error.name === "AbortError" && error.code === "ABORT_ERR") {
    // The SDK sent SIGINT and no authoritative terminal result committed.
  }
}
```

Once the subprocess produces a matching terminal event, exec result, and exit
code, that committed outcome wins over a late cancellation race. Otherwise an
observed abort rejects with `AbortError`/`ABORT_ERR` after process shutdown.
After a valid exec result, the subprocess must close within
`resultCloseGraceMs` (1.5 seconds by default); a lingering process is terminated
and reported as a protocol failure. Before that record, `transportTimeoutMs`
bounds the whole child transport (by default, the Harness deadline plus abort
grace and five seconds). Closing stdout without a result also triggers immediate
bounded process shutdown. Cancellation and protocol/watchdog shutdown request
cooperative SIGINT first, then use SIGKILL only after the configured grace.

## Security and protocol boundaries

- Prompts are sent through stdin as `-`, not exposed in the process argument
  list. Global options precede `exec`; attachments use repeated `--attach`.
- `spawn()` always uses `shell: false`. Relative attachment paths are made
  absolute against `activeDirectory` before invocation.
- stdout must be canonical, bounded UTF-8 JSONL. Event envelopes, contiguous
  sequence numbers, run identity, one terminal event, the exact
  `agent_harness.exec_result.v1` record, and the exit code are checked
  fail-closed.
- Configurable line, record, total-output, and reconstructed-response limits
  bound memory use. The defaults accept the harness contract while rejecting
  unlimited output.
- Child stderr is drained and discarded. `HarnessProcessError` exposes only a
  generic safe message, exit code, and signal; provider or CLI stderr text is
  never copied into errors.
- `finalResponse` concatenates only non-internal `message.delta` content.
  `reasoning.delta` is never promoted to the answer.
- `onEvent` is observation-only. Exceptions thrown by a callback are isolated
  and cannot interrupt or revoke the authoritative run; rejected async callback
  results are consumed instead of becoming unhandled Node.js rejections.
- The SDK does not add a headless approval broker. Permission policy remains in
  Agent Harness; operations requiring unavailable interactive approval may end
  in a handoff.
- This is a subprocess/JSONL SDK, not an app-server transport and not full API
  parity with other coding-agent SDKs.

The bundled `LICENSE` is the repository's Academic Evaluation License. It must
remain in redistributed package artifacts.
