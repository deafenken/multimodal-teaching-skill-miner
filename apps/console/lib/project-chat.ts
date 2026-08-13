import type {LearningProject, LearningProjectChatThread, LearningProjectSummary, TeachingMessage} from "@/lib/types";

export interface ProjectChatThread {
  id: string;
  title: string;
  messages: TeachingMessage[];
  createdAt: number;
  updatedAt: number;
}

export function validProjectThreadId(value: string) {
  return /^chat_[0-9a-f]{24}$/.test(value);
}

function utcTimestamp(value: number) {
  return new Date(value).toISOString().replace(/\.\d{3}Z$/, "Z");
}

function messageTimestamp(message: TeachingMessage, fallback: number) {
  const parsed = Date.parse(message.createdAt);
  return Number.isFinite(parsed) ? utcTimestamp(parsed) : utcTimestamp(fallback);
}

function teachingMessageStatus(message: TeachingMessage) {
  return message.status ?? (message.streaming ? "running" : "completed");
}

function boundedContent(value: string) {
  const maximum = 64_000;
  if (value.length <= maximum) return value;
  const notice = "\n\n[内容超过学习项目单条消息上限，已安全截断]";
  return `${value.slice(0, maximum - notice.length)}${notice}`;
}

function boundedSources(sources: TeachingMessage["sources"]) {
  return (sources ?? []).flatMap((source) => {
    const title = source.title.trim().slice(0, 300);
    const url = source.url.trim().slice(0, 2_000);
    if (!title || !url.startsWith("https://") && !url.startsWith("http://")) return [];
    return [{title, url}];
  }).slice(0, 24);
}

export function projectChatThread(thread: ProjectChatThread): LearningProjectChatThread {
  return {
    thread_id: thread.id,
    title: thread.title.trim().slice(0, 160) || "新对话",
    created_at: utcTimestamp(thread.createdAt),
    updated_at: utcTimestamp(thread.updatedAt),
    messages: thread.messages.filter((message) => message.role !== "tool").map((message) => ({
      message_id: message.id.slice(0, 180),
      role: message.role === "learner" ? "user" : "assistant",
      content: boundedContent(message.body),
      status: teachingMessageStatus(message),
      created_at: messageTimestamp(message, thread.createdAt),
      web_search_used: Boolean(message.webSearchUsed),
      sources: boundedSources(message.sources),
    })),
  };
}

export function chatThreadFromProject(thread: LearningProjectChatThread): ProjectChatThread {
  return {
    id: thread.thread_id,
    title: thread.title,
    createdAt: Date.parse(thread.created_at),
    updatedAt: Date.parse(thread.updated_at),
    messages: thread.messages.map((message) => ({
      id: message.message_id,
      role: message.role === "user" ? "learner" : message.role === "assistant" ? "teacher" : "tool",
      body: message.role === "tool" ? "" : message.content,
      toolLabel: message.role === "tool" ? "Chat" : undefined,
      toolDetail: message.role === "tool" ? message.content : undefined,
      createdAt: message.created_at,
      status: message.status === "running" ? "stopped" : message.status,
      streaming: false,
      retryable: false,
      webSearchUsed: message.web_search_used,
      sources: message.sources,
    })),
  };
}

export function projectSummary(project: LearningProject): LearningProjectSummary {
  return {
    project_id: project.project_id,
    title: project.title,
    description: project.description,
    status: project.status,
    pinned: project.pinned,
    updated_at: project.updated_at,
    syllabus_count: project.syllabus_ids.length,
    teaching_session_count: project.teaching_session_ids.length,
    resource_count: project.resource_ids.length,
    chat_thread_count: project.chat_threads.length,
    note_count: project.notes.length,
  };
}
