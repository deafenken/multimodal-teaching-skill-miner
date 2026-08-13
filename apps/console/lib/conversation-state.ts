import type {ChatTurn, TeachingMessage, TurnStatus} from "@/lib/types";

const validTurnStatuses = new Set<TurnStatus>(["queued", "running", "stopped", "failed", "completed"]);

export function storedTeachingMessage(value: unknown): TeachingMessage | null {
  if (!value || typeof value !== "object") return null;
  const message = value as Partial<TeachingMessage>;
  if (
    typeof message.id !== "string"
    || !["learner", "teacher", "tool"].includes(String(message.role))
    || typeof message.body !== "string"
  ) return null;
  const storedStatus = validTurnStatuses.has(message.status as TurnStatus)
    ? message.status
    : message.role === "teacher" ? "completed" : undefined;
  const status = storedStatus === "running" ? "stopped" : storedStatus;
  return {
    ...message,
    id: message.id,
    role: message.role as TeachingMessage["role"],
    body: message.body,
    createdAt: typeof message.createdAt === "string" ? message.createdAt : "历史",
    status,
    streaming: false,
    thinking: undefined,
    error: undefined,
    retryable: false,
  };
}

export function chatTurnsFromMessages(messages: TeachingMessage[], maxTurns = 400): ChatTurn[] {
  const turns: ChatTurn[] = [];
  for (const message of messages) {
    if (
      message.role === "tool"
      || message.streaming
      || message.status === "failed"
      || message.status === "stopped"
      || !message.body.trim()
    ) continue;
    const role = message.role === "learner" ? "user" : "assistant";
    const previous = turns.at(-1);
    if (previous?.role === role) previous.content = `${previous.content}\n\n${message.body.trim()}`;
    else turns.push({role, content: message.body.trim()});
  }
  turns.splice(0, Math.max(0, turns.length - maxTurns));
  if (turns[0]?.role === "assistant") turns.shift();
  return turns;
}

export function prepareTurnRetry(messages: TeachingMessage[], messageId: string, thinking: string) {
  const lifecyclePrefix = `${messageId}-harness-`;
  return messages
    .filter((message) => !message.id.startsWith(lifecyclePrefix))
    .map((message) => message.id === messageId ? {
      ...message,
      body: "",
      status: "running" as const,
      streaming: true,
      thinking,
      error: undefined,
      retryable: true,
      createdAt: "刚刚",
    } : message);
}

export function prepareCompletedTurnBranch(
  messages: TeachingMessage[],
  messageId: string,
  newMessageId: string,
  thinking: string,
) {
  const responseIndex = messages.findIndex((message) => message.id === messageId);
  const response = messages[responseIndex];
  if (responseIndex < 0 || response?.role !== "teacher" || response.status !== "completed") return null;
  const lifecyclePrefix = `${messageId}-harness-`;
  return [
    ...messages.slice(0, responseIndex).filter((message) => !message.id.startsWith(lifecyclePrefix)),
    {
      ...response,
      id: newMessageId,
      body: "",
      status: "running" as const,
      streaming: true,
      thinking,
      error: undefined,
      retryable: true,
      createdAt: "刚刚",
      webSearchUsed: false,
      sources: undefined,
    },
  ];
}

export function chatBranchTitle(title: string) {
  const suffix = " · 分支";
  const base = title.trim() || "新对话";
  return `${base.slice(0, 160 - suffix.length)}${suffix}`;
}

export function settleTurn(
  messages: TeachingMessage[],
  messageId: string,
  status: Extract<TurnStatus, "completed" | "failed" | "stopped">,
  options?: {body?: string; error?: string},
) {
  return messages.map((message) => message.id === messageId ? {
    ...message,
    ...(options?.body !== undefined ? {body: options.body} : {}),
    status,
    streaming: false,
    thinking: undefined,
    error: options?.error,
    retryable: true,
    createdAt: "刚刚",
  } : message);
}
