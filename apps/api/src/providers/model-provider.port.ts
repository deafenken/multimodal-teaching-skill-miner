export interface ModelProviderStatus {
  provider: string;
  model: string;
  configured: boolean;
  streaming: boolean;
  toolUse: boolean;
  reason?: "credential_missing" | "adapter_not_enabled" | "unsupported_provider";
}

export interface ModelTurnCommand {
  taskId: string;
  sessionId: string;
  learnerMessage: string;
  signal: AbortSignal;
}

export type ModelStreamEvent =
  | {type: "text_delta"; text: string}
  | {type: "tool_use"; name: string; input: Record<string, unknown>}
  | {type: "tool_result"; name: string; output: Record<string, unknown>};

export interface ModelProviderPort {
  status(): Promise<ModelProviderStatus>;
  streamTurn(command: ModelTurnCommand): AsyncIterable<ModelStreamEvent>;
}
