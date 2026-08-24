import {
  HarnessAbortError,
  HarnessClient,
  HarnessRunStream,
  HarnessThread,
  type HarnessEvent,
  type HarnessRunResult,
  type PermissionMode,
} from "../index.js";

const permission: PermissionMode = "workspace-write";
const client = new HarnessClient({
  cwd: "/work/project",
  activeDirectory: "src",
  permissionMode: permission,
  transportTimeoutMs: 30_000,
  env: { HARNESS_ALLOW_REMOTE_CONTENT: "0", REMOVE_ME: undefined },
});

const lazy = client.startThread();
const lazyId: string | null = lazy.id;
void lazyId;

const resumed = client.resumeThread(
  "session_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  { permissionMode: "read-only" },
);

const resultPromise: Promise<HarnessRunResult> = resumed.run("inspect", {
  attachments: ["notes.md"],
  onEvent(event: HarnessEvent) {
    if (event.type === "message.delta") {
      const delta: string = event.payload.delta;
      void delta;
    }
  },
});
void resultPromise;

const controller = new AbortController();
const streamed = lazy.runStream("stream", { signal: controller.signal });
const streamResult: Promise<HarnessRunResult> = streamed.result;
void streamResult;

async function consume(): Promise<void> {
  for await (const event of streamed) {
    const sequence: number = event.sequence;
    void sequence;
  }
}
void consume;

const abortError: Error = new HarnessAbortError();
void abortError;

// @ts-expect-error thread handles can only be created by HarnessClient
new HarnessThread();

// @ts-expect-error run streams can only be created by HarnessThread
new HarnessRunStream();

// @ts-expect-error permission modes are a closed union
client.startThread({ permissionMode: "unsafe" });

// @ts-expect-error attachments are paths, not arbitrary objects
lazy.run("invalid", { attachments: [{ path: "notes.md" }] });
