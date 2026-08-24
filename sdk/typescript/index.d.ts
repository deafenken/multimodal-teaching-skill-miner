export type PermissionMode =
  | "read-only"
  | "workspace-write"
  | "full-access";

export type HarnessStatus = "completed" | "cancelled" | "handoff" | "failed";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | readonly JsonValue[];
export interface JsonObject {
  readonly [key: string]: JsonValue;
}

export type HarnessEventType =
  | "run.started"
  | "run.completed"
  | "run.cancelled"
  | "run.failed"
  | "run.handoff"
  | "model.started"
  | "model.completed"
  | "model.failed"
  | "model.retrying"
  | "model.retry_suppressed"
  | "message.start"
  | "message.delta"
  | "message.end"
  | "reasoning.delta"
  | "usage.update"
  | "tool_call.delta"
  | "approval.requested"
  | "approval.resolved"
  | "hook.started"
  | "hook.effect_started"
  | "hook.completed"
  | "hook.failed"
  | "tool.requested"
  | "tool.started"
  | "tool.effect_started"
  | "tool.progress"
  | "tool.retrying"
  | "tool.completed"
  | "tool.failed"
  | "tool.rejected"
  | "tool.replayed"
  | "tool.reconciled"
  | "input.steered"
  | "guard.triggered"
  | "action.started"
  | "action.completed"
  | "action.superseded"
  | "progress.updated"
  | "operation.result"
  | "state.committed";

export interface HarnessEventBase<TType extends HarnessEventType, TPayload> {
  readonly schema: "agent_harness.event.v1";
  readonly event_id: string;
  readonly run_id: string;
  readonly turn_id: string;
  readonly sequence: number;
  readonly type: TType;
  readonly timestamp: string;
  readonly causation_id?: string;
  readonly payload: Readonly<TPayload>;
}

export interface MessageDeltaPayload {
  readonly delta: string;
  readonly channel?: string;
  readonly [key: string]: JsonValue | undefined;
}

export type MessageDeltaEvent = HarnessEventBase<
  "message.delta",
  MessageDeltaPayload
>;

export type GenericHarnessEvent = HarnessEventBase<
  Exclude<HarnessEventType, "message.delta">,
  JsonObject
>;

export type HarnessEvent = MessageDeltaEvent | GenericHarnessEvent;

export interface HarnessRunResult {
  readonly sessionId: string;
  readonly runId: string;
  readonly turnId: string;
  readonly status: HarnessStatus;
  readonly reason: string;
  readonly usage: Readonly<Record<string, number>>;
  /** Concatenated non-internal message.delta text. */
  readonly finalResponse: string;
}

export interface HarnessClientOptions {
  /** Executable name on PATH, or a path to an executable. Defaults to harness. */
  readonly harnessPath?: string;
  /** Existing workspace directory. Defaults to process.cwd(). */
  readonly cwd?: string;
  /** Existing subdirectory inside cwd used for scoped instructions and attachments. */
  readonly activeDirectory?: string;
  readonly stateHome?: string;
  readonly worktreeHome?: string;
  readonly apiKeyFile?: string;
  readonly model?: string;
  readonly permissionMode?: PermissionMode;
  readonly deadlineSeconds?: number;
  readonly maxSteps?: number;
  /** Overrides merged onto the inherited process environment; undefined removes a key. */
  readonly env?: Readonly<Record<string, string | undefined>>;
  readonly maxLineBytes?: number;
  readonly maxRecords?: number;
  readonly maxOutputBytes?: number;
  readonly maxResponseBytes?: number;
  readonly abortGraceMs?: number;
  /** Maximum wait for process close after a valid exec result. */
  readonly resultCloseGraceMs?: number;
  /**
   * Overall child transport limit before an exec result. Defaults to the
   * Harness deadline plus abort grace and a five-second protocol allowance.
   */
  readonly transportTimeoutMs?: number;
}

export interface HarnessThreadOptions {
  readonly permissionMode?: PermissionMode;
}

export interface HarnessRunOptions {
  /** Paths are made absolute relative to activeDirectory before spawning. */
  readonly attachments?: readonly string[];
  readonly permissionMode?: PermissionMode;
  readonly signal?: AbortSignal;
  readonly onEvent?: (event: HarnessEvent) => void;
}

export class HarnessSdkError extends Error {
  readonly code: string;
  constructor(message: string, code?: string);
}

export class HarnessValidationError extends HarnessSdkError {
  constructor(message: string);
}

export class HarnessProtocolError extends HarnessSdkError {
  constructor(message: string);
}

export class HarnessProcessError extends HarnessSdkError {
  readonly exitCode: number | null;
  readonly signal: string | null;
  constructor(
    message: string,
    details?: { readonly exitCode?: number | null; readonly signal?: string | null },
  );
}

export class HarnessAbortError extends HarnessSdkError {
  readonly name: "AbortError";
  readonly code: "ABORT_ERR";
  constructor();
}

export class HarnessRunStream implements AsyncIterable<HarnessEvent> {
  private constructor();
  /** Resolves after the process closes and the terminal result is verified. */
  readonly result: Promise<HarnessRunResult>;
  [Symbol.asyncIterator](): AsyncIterator<HarnessEvent>;
}

export class HarnessThread {
  private constructor();
  /** Null for a new lazy thread until its first valid run result. */
  readonly id: string | null;
  /** Alias for id. */
  readonly sessionId: string | null;
  readonly permissionMode: PermissionMode;

  run(prompt: string, options?: HarnessRunOptions): Promise<HarnessRunResult>;
  runStream(prompt: string, options?: HarnessRunOptions): HarnessRunStream;
}

export class HarnessClient {
  constructor(options?: HarnessClientOptions);

  readonly cwd: string;
  readonly activeDirectory: string;
  readonly permissionMode: PermissionMode;

  /** Creates a lazy thread; no session is persisted until its first run. */
  startThread(options?: HarnessThreadOptions): HarnessThread;
  resumeThread(
    sessionId: string,
    options?: HarnessThreadOptions,
  ): HarnessThread;
}
