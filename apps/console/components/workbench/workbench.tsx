"use client";

import {useQuery, useQueryClient} from "@tanstack/react-query";
import {useCallback, useEffect, useMemo, useRef, useState} from "react";

import {ConversationPane} from "@/components/workbench/conversation-pane";
import {CommandCenter} from "@/components/workbench/command-center";
import {InspectorDrawer} from "@/components/workbench/inspector-drawer";
import {AccountDataRightsPanel} from "@/components/workbench/account-data-rights-panel";
import {SessionSidebar} from "@/components/workbench/session-sidebar";
import {SyllabusWorkspace} from "@/components/workbench/syllabus-workspace";
import {
  addLearningProjectReference,
  accountDataRightsStepUpUrl,
  accountDeletionStatus,
  authenticatedHarnessLogoutAvailable,
  isHarnessAuthenticationRequired,
  organizationLoginAvailable,
  organizationLoginUrl,
  bootstrapLearningProject,
  cancelHarnessRun,
  clearAuthenticatedHarnessSession,
  confirmAccountDeletion,
  fetchBootstrap,
  downloadAccountExport,
  createLearningProject,
  exportLearningProject,
  fetchLearningProject,
  fetchTeachingSyllabus,
  grantRemoteConsent,
  importTeachingSyllabus,
  isTeachingSyllabusDocument,
  listLearningProjects,
  listRemoteConsents,
  listTrashedLearningProjects,
  prepareAccountDeletion,
  purgeLearningProject,
  restoreLearningProject,
  resumeAccountDeletion,
  resumeTeachingSession,
  saveProjectChatThread,
  sendTeachingCommand,
  streamChat,
  streamTeachingStart,
  streamTeachingTurn,
  trashLearningProject,
  updateLearningProject,
  uploadTeachingAttachment,
  uploadTeachingResource,
  type HarnessRunIdentity,
  type HarnessLifecycleUpdate,
} from "@/lib/api";
import {
  completeCommittedAccountDeletion,
  CommittedAccountCleanupError,
  type AccountDeletionReceipt,
  type AccountDeletionStatus,
} from "@/lib/account-data-rights";
import {chatBranchTitle, chatTurnsFromMessages, prepareCompletedTurnBranch, prepareTurnRetry, settleTurn, storedTeachingMessage} from "@/lib/conversation-state";
import {MAX_CHAT_ATTACHMENTS, chatAttachmentMimeType, chatResourceRefs, selectChatAttachments, type ChatResourceRef} from "@/lib/chat-attachments";
import {ModeRequestSlots, nextChatFollowUp, type WorkbenchMode} from "@/lib/mode-runtime";
import {boundedLogoutCleanup, settleRunCancellationsBeforeLogout} from "@/lib/session-logout";
import {clearOfflineRuntimeForLogout, deleteOfflineWorkspaceProject, loadOfflineWorkspaceProject, saveOfflineWorkspaceProject} from "@/lib/offline-runtime";
import {
  ACTIVE_PROJECT_STORAGE_KEY,
  CHAT_HISTORY_STORAGE_KEY,
  clearAccountBoundBrowserStorage,
  LEGACY_MIGRATION_STORAGE_KEY,
  SESSION_INDEX_STORAGE_KEY,
  SESSION_STORAGE_KEY,
  START_KEY_STORAGE_KEY,
  workbenchIdentityBoundary,
} from "@/lib/account-cache-boundary";
import {chatThreadFromProject, projectChatThread, projectSummary, validProjectThreadId, type ProjectChatThread} from "@/lib/project-chat";
import {projectScopedItems} from "@/lib/project-workspace";
import {replaceMatchingTeachingResource, uniqueTeachingResources} from "@/lib/resource-review";
import {invalidateLegacyRemoteConsent} from "@/lib/remote-consent";
import type {CommandCenterActionId} from "@/lib/command-center";
import type {
  AgentAction,
  BootstrapPayload,
  ChatTurn,
  LearningProject,
  LearningProjectSummary,
  LearningProjectTrashItem,
  LessonProgress,
  QueuedChatPrompt,
  RemoteConsentPolicy,
  RemoteConsentPurpose,
  RemoteConsentReceipt,
  ResourceUploadItem,
  SessionListItem,
  SyllabusLesson,
  SyllabusLessonStartPayload,
  SyllabusModule,
  TeacherSessionResponse,
  TeachingResourceSummary,
  TeachingSyllabus,
  TeachingMessage,
} from "@/lib/types";

const consentPurposeMessages: Record<RemoteConsentPurpose, string> = {
  remote_chat: "Chat 会把当前问题及必要的对话上下文发送给远程模型处理；选择附件时，还会发送本机检索出的有限文字摘录，但不会发送原文件、原图或音视频。",
  remote_teaching: "Teach 会把学习目标、回答、有限画像和已提取的资源文字发送给远程模型。",
  remote_syllabus_generation: "大纲生成会把主题、受众、目标和有限资源摘录发送给远程模型。",
  public_web_search: "联网搜索会把当前问题及必要的对话上下文发送给远程模型和搜索服务。",
  remote_visual_analysis: "远程视觉分析会把原始图片发送给当前视觉服务商；本地 OCR 不需要这项授权。",
};

function activeConsentMatches(receipt: RemoteConsentReceipt, policy?: RemoteConsentPolicy) {
  return receipt.status === "active"
    && Date.parse(receipt.expires_at_utc) > Date.now()
    && (!policy || (
      receipt.provider_id === policy.provider_id
      && receipt.processing_region === policy.processing_region
      && receipt.provider_retention_days === policy.provider_retention_days
      && receipt.provider_policy_sha256 === policy.provider_policy_sha256
      && receipt.subject_policy_sha256 === policy.subject_policy_sha256
      && policy.data_categories.every((category) => receipt.data_categories.includes(category))
    ));
}

type ChatThread = ProjectChatThread;

interface ChatRetryContext {
  mode: "chat";
  threadId: string;
  prompt: string;
  webSearch: boolean;
  requestMessages: ChatTurn[];
  resourceRefs: ChatResourceRef[];
  resourceItems: ResourceUploadItem[];
}

interface TeachingRetryContext {
  mode: "teach";
  operation: "start" | "step";
  prompt: string;
  attachment?: File;
  session?: TeacherSessionResponse;
  startPayload?: Record<string, unknown>;
  startSkillId?: string | null;
  attachmentIdempotencyKey?: string;
  turnIdempotencyKey?: string;
}

type TurnRetryContext = ChatRetryContext | TeachingRetryContext;

const MAX_RETRY_CONTEXTS = 16;

function rememberRetryContext(
  contexts: Map<string, TurnRetryContext>,
  messageId: string,
  context: TurnRetryContext,
) {
  contexts.delete(messageId);
  contexts.set(messageId, context);
  while (contexts.size > MAX_RETRY_CONTEXTS) {
    const oldest = contexts.keys().next().value;
    if (typeof oldest !== "string") break;
    contexts.delete(oldest);
  }
}

function parseChatThreads(value: string | null): ChatThread[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.flatMap((item): ChatThread[] => {
      if (!item || typeof item !== "object") return [];
      const candidate = item as Partial<ChatThread>;
      if (typeof candidate.id !== "string" || typeof candidate.title !== "string" || !Array.isArray(candidate.messages)) return [];
      const messages = candidate.messages.flatMap((message) => {
        const stored = storedTeachingMessage(message);
        return stored ? [stored] : [];
      });
      const updatedAt = Number(candidate.updatedAt) || Date.now();
      return [{id: validProjectThreadId(candidate.id) ? candidate.id : newChatThreadId(), title: candidate.title, messages, createdAt: Number(candidate.createdAt) || updatedAt, updatedAt}];
    }).sort((left, right) => right.updatedAt - left.updatedAt);
  } catch {
    return [];
  }
}

function newChatThreadId() {
  return `chat_${crypto.randomUUID().replaceAll("-", "").slice(0, 24)}`;
}

function rememberedSessionIds() {
  try {
    const parsed = JSON.parse(window.sessionStorage.getItem(SESSION_INDEX_STORAGE_KEY) ?? "[]") as unknown;
    const ids = Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string" && Boolean(item)) : [];
    const legacy = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    return Array.from(new Set([...(legacy ? [legacy] : []), ...ids]));
  } catch {
    return [];
  }
}

function rememberSessionId(sessionId: string) {
  const ids = [sessionId, ...rememberedSessionIds().filter((item) => item !== sessionId)];
  window.sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  window.sessionStorage.setItem(SESSION_INDEX_STORAGE_KEY, JSON.stringify(ids));
}

function forgetSessionIds(sessionIds: string[]) {
  const selected = new Set(sessionIds);
  if (!selected.size) return;
  const retained = rememberedSessionIds().filter((item) => !selected.has(item));
  window.sessionStorage.setItem(SESSION_INDEX_STORAGE_KEY, JSON.stringify(retained));
  const legacy = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (legacy && selected.has(legacy)) window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
}

function legacyMigrationId() {
  const current = window.localStorage.getItem(LEGACY_MIGRATION_STORAGE_KEY);
  if (current && /^[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}$/.test(current)) return current;
  const created = `migration:${crypto.randomUUID()}`;
  window.localStorage.setItem(LEGACY_MIGRATION_STORAGE_KEY, created);
  return created;
}

function currentAction(session: TeacherSessionResponse): AgentAction | undefined {
  return session.next_action ?? session.current_action;
}

function actionMessage(action: AgentAction | undefined): string {
  return String(action?.teacher_action?.message ?? "").trim();
}

function actionSkill(action: AgentAction | undefined): string | undefined {
  const skill = action?.primary_skill;
  return skill?.name ? String(skill.name) : skill?.skill_id ? String(skill.skill_id) : undefined;
}

function teacherMessage(id: string, action: AgentAction | undefined, body = actionMessage(action), lessonProgress?: LessonProgress | null): TeachingMessage | null {
  if (!body) return null;
  return {id, role: "teacher", body, createdAt: "刚刚", skill: actionSkill(action), lessonPhase: action?.lesson_phase ?? lessonProgress ?? undefined, status: "completed"};
}

const MAX_VISIBLE_LIFECYCLE_ROWS = 24;

function upsertHarnessLifecycleMessage(
  current: TeachingMessage[],
  responseMessageId: string,
  update: HarnessLifecycleUpdate,
): TeachingMessage[] {
  const prefix = `${responseMessageId}-harness-`;
  const id = `${prefix}${update.id}`;
  const toolLabel = update.state === "completed"
    ? `✓ ${update.label}`
    : update.state === "failed"
      ? `! ${update.label}`
      : update.label;
  const nextMessage: TeachingMessage = {
    id,
    role: "tool",
    body: "",
    createdAt: update.state === "running" ? "运行中" : "刚刚",
    toolLabel,
    toolDetail: update.detail,
  };
  const existingIndex = current.findIndex((message) => message.id === id);
  if (existingIndex >= 0) {
    return current.map((message, index) => index === existingIndex ? nextMessage : message);
  }
  if (current.filter((message) => message.id.startsWith(prefix)).length >= MAX_VISIBLE_LIFECYCLE_ROWS) return current;
  const responseIndex = current.findIndex((message) => message.id === responseMessageId);
  if (responseIndex < 0) return [...current, nextMessage];
  return [...current.slice(0, responseIndex), nextMessage, ...current.slice(responseIndex)];
}

function settleHarnessLifecycleMessages(current: TeachingMessage[], responseMessageId: string, detail: string) {
  const prefix = `${responseMessageId}-harness-`;
  return current.map((message) => message.id.startsWith(prefix) && message.createdAt === "运行中" ? {
    ...message,
    createdAt: "刚刚",
    toolDetail: `${message.toolDetail || "正在运行"} · ${detail}`,
  } : message);
}

function settleRunningHarnessLifecycleMessages(current: TeachingMessage[], detail: string) {
  return current.map((message) => message.id.includes("-harness-") && message.createdAt === "运行中" ? {
    ...message,
    createdAt: "刚刚",
    toolDetail: `${message.toolDetail || "正在运行"} · ${detail}`,
  } : message);
}

function messagesFromSession(session: TeacherSessionResponse): TeachingMessage[] {
  const messages: TeachingMessage[] = [];
  const concept = String(session.goal?.concept ?? "").trim();
  if (concept) messages.push({id: `goal-${session.session_id}`, role: "learner", body: concept, createdAt: "历史"});
  for (const event of session.history ?? []) {
    const action = event.action;
    const teacher = teacherMessage(`r${event.round ?? messages.length}-teacher`, action);
    if (teacher) messages.push(teacher);
    const response = String(event.learner_response ?? event.learner_text ?? "").trim();
    if (response) messages.push({id: `r${event.round ?? messages.length}-learner`, role: "learner", body: response, createdAt: "历史"});
  }
  const next = teacherMessage(`current-${session.context_version}`, currentAction(session), undefined, session.lesson_progress);
  const last = messages.at(-1);
  if (next && !(last?.role === "teacher" && last.body === next.body)) messages.push(next);
  return messages;
}

function inferLessonIntent(prompt: string): "teach_first" | "diagnostic_first" | "task_first" {
  if (/(?:考考我|测试|测验|出题|检查(?:一下)?(?:我|理解)|检验(?:一下)?(?:我|理解)|\bquiz\b|\btest\b)/i.test(prompt)) {
    return "diagnostic_first";
  }
  if (/(?:帮我做|帮我解|解(?:一下)?这道|这道题|这题|求解|作业|怎么解|如何解)/i.test(prompt)) {
    return "task_first";
  }
  return "teach_first";
}

function startPayload(bootstrap: BootstrapPayload, startIdempotencyKey: string, firstPrompt: string, stagedResourceIds: string[], goalOverride?: Record<string, unknown>) {
  const prompt = firstPrompt.trim();
  const learningIntent = inferLessonIntent(prompt);
  const configuredMaxRounds = Number(bootstrap.default_goal?.max_rounds ?? 12);
  const goal = goalOverride ?? (prompt ? {
    concept: prompt.slice(0, 120),
    objective: `帮助学习者理解并解决：${prompt.slice(0, 240)}`,
    learning_intent: learningIntent,
    knowledge_components: ["核心概念", "关键步骤", "迁移应用"],
    success_thresholds: bootstrap.default_goal?.success_thresholds ?? {},
    max_rounds: learningIntent === "teach_first" ? Math.max(configuredMaxRounds, 18) : configuredMaxRounds,
    materials: {
      example: `为“${prompt.slice(0, 120)}”提供一个最小、具体的例子`,
      practice: `请学习者用自己的话解释“${prompt.slice(0, 120)}”并完成一个单步练习`,
      transfer_task: `把“${prompt.slice(0, 120)}”应用到一个新的相似情境`,
    },
  } : bootstrap.default_goal ?? {});
  const payload: Record<string, unknown> = {
    goal,
    student_profile: bootstrap.default_student_profile ?? {},
    profile_display_name: "本地学生",
    start_idempotency_key: startIdempotencyKey,
  };
  if (stagedResourceIds.length) payload.staged_resource_ids = stagedResourceIds;
  return payload;
}

function syllabusLessonGoal(bootstrap: BootstrapPayload, syllabus: TeachingSyllabus, module: SyllabusModule, lesson: SyllabusLesson) {
  const provided = lesson.start_payload?.goal ?? lesson.teaching_goal ?? {};
  const stringField = (value: unknown) => typeof value === "string" && value.trim() ? value.trim() : undefined;
  const providedComponents = Array.isArray(provided.knowledge_components)
    ? provided.knowledge_components.flatMap((item) => typeof item === "string" && item.trim() ? [item.trim()] : [])
    : [];
  const knowledgeComponents = (providedComponents.length ? providedComponents : lesson.knowledge_components?.length ? lesson.knowledge_components : [lesson.title]).slice(0, 12);
  const suppliedMaterials = provided.materials && typeof provided.materials === "object" && !Array.isArray(provided.materials)
    ? Object.fromEntries(Object.entries(provided.materials).flatMap(([key, value]) => typeof value === "string" && value.trim() ? [[key, value.trim()]] : []))
    : {};
  const lessonContext = [
    `教学大纲：${syllabus.title}`,
    `当前模块：${module.title}`,
    `当前课节：${lesson.title}`,
    lesson.objective ? `课节目标：${lesson.objective}` : "",
    lesson.summary ? `课节说明：${lesson.summary}` : "",
    lesson.prerequisites?.length ? `前置要求：${lesson.prerequisites.join("、")}` : "",
  ].filter(Boolean).join("\n");
  const configuredMaxRounds = Number(provided.max_rounds ?? bootstrap.default_goal?.max_rounds ?? 18);
  return {
    concept: stringField(provided.concept) ?? lesson.title,
    objective: stringField(provided.objective) ?? lesson.objective ?? `帮助学习者理解并应用：${lesson.title}`,
    learning_intent: "teach_first",
    knowledge_components: knowledgeComponents,
    success_thresholds: provided.success_thresholds ?? bootstrap.default_goal?.success_thresholds ?? {},
    max_rounds: Number.isInteger(configuredMaxRounds) ? Math.min(50, Math.max(18, configuredMaxRounds)) : 18,
    materials: {
      ...lesson.materials,
      ...suppliedMaterials,
      syllabus_context: lessonContext,
      example: suppliedMaterials.example ?? lesson.materials?.example ?? `围绕“${lesson.title}”提供一个最小、具体的例子`,
      practice: suppliedMaterials.practice ?? lesson.materials?.practice ?? `围绕“${lesson.title}”完成一个带提示的单步练习`,
      transfer_task: suppliedMaterials.transfer_task ?? lesson.materials?.transfer_task ?? `把“${lesson.title}”迁移到一个新的相似情境`,
    },
  };
}

function selectTeachingSkill(session: TeacherSessionResponse, skillId: string) {
  if (!session.expected_question_id || !session.profile_summary?.profile_revision) {
    throw new Error("当前会话尚未准备好 Skill 路由");
  }
  return sendTeachingCommand({
    session_id: session.session_id,
    expected_round: session.rounds_completed ?? 0,
    expected_question_id: session.expected_question_id,
    expected_context_version: session.context_version,
    profile_revision: session.profile_summary.profile_revision,
    command: "select_skill",
    skill_id: skillId,
    command_idempotency_key: `console-skill-${crypto.randomUUID()}`,
  });
}

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}

function fileAsBase64(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("无法读取文件"));
    reader.onload = () => {
      const value = String(reader.result ?? "");
      resolve(value.includes(",") ? value.slice(value.indexOf(",") + 1) : value);
    };
    reader.readAsDataURL(file);
  });
}

function resourceMimeType(file: File) {
  if (file.type) return file.type;
  const extension = file.name.split(".").pop()?.toLowerCase();
  return ({
    txt: "text/plain",
    md: "text/markdown",
    markdown: "text/markdown",
    pdf: "application/pdf",
    doc: "application/msword",
    docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    rtf: "application/rtf",
    ppt: "application/vnd.ms-powerpoint",
    pptx: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    webp: "image/webp",
  } as Record<string, string>)[extension ?? ""] ?? "application/octet-stream";
}

function ConnectionStatusBar({state, detail, onReconnect}: {
  state: "connecting" | "disconnected";
  detail?: string;
  onReconnect: () => void;
}) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="absolute left-1/2 top-14 z-30 flex max-w-[calc(100%-2rem)] -translate-x-1/2 items-center gap-2 rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-overlay)] px-3 py-2 text-xs text-[var(--app-muted)] shadow-[0_12px_32px_var(--app-shadow)]"
    >
      <span className={state === "connecting" ? "size-2 animate-pulse rounded-full bg-[var(--app-warning)]" : "size-2 rounded-full bg-[var(--app-danger)]"} />
      <span className="truncate">{state === "connecting" ? "正在连接 TeachLab 后端…" : detail || "TeachLab 后端连接中断，正在自动重连。"}</span>
      <button type="button" onClick={onReconnect} className="shrink-0 rounded-md px-2 py-1 text-[var(--app-text-soft)] hover:bg-[var(--app-hover)]">立即重连</button>
    </div>
  );
}

export function Workbench() {
  const queryClient = useQueryClient();
  const [activeSession, setActiveSession] = useState<TeacherSessionResponse | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [settingsRequest, setSettingsRequest] = useState(0);
  const [accountRightsOpen, setAccountRightsOpen] = useState(false);
  const [accountDeletionState, setAccountDeletionState] = useState<AccountDeletionStatus | null>(null);
  const [accountDeletedReceipt, setAccountDeletedReceipt] = useState<AccountDeletionReceipt | null>(null);
  const [resourcesRequest, setResourcesRequest] = useState(0);
  const [commandCenterOpen, setCommandCenterOpen] = useState(false);
  const [isCompact, setIsCompact] = useState(false);
  const [sidebarView, setSidebarView] = useState<"chat" | "teach">("chat");
  const [teachSurface, setTeachSurface] = useState<"conversation" | "syllabi">("conversation");
  const [openedSyllabus, setOpenedSyllabus] = useState<TeachingSyllabus | null>(null);
  const [syllabusLaunchError, setSyllabusLaunchError] = useState("");
  const [queuedSkillId, setQueuedSkillId] = useState<string | null>(null);
  const [runningModes, setRunningModes] = useState<Record<WorkbenchMode, boolean>>({chat: false, teach: false});
  const [teachingMessages, setTeachingMessages] = useState<TeachingMessage[]>([]);
  const [chatMessages, setChatMessages] = useState<TeachingMessage[]>([]);
  const [chatThreads, setChatThreads] = useState<ChatThread[]>([]);
  const [chatHistoryLoaded, setChatHistoryLoaded] = useState(false);
  const [activeChatId, setActiveChatId] = useState("new-chat");
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [activeProject, setActiveProject] = useState<LearningProject | null>(null);
  const [projectMutationBusy, setProjectMutationBusy] = useState(false);
  const [projectNotice, setProjectNotice] = useState<{text: string; projectId?: string; recoveryToken?: string} | null>(null);
  const [queuedChatPrompts, setQueuedChatPrompts] = useState<QueuedChatPrompt[]>([]);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [visualAnalysisEnabled, setVisualAnalysisEnabled] = useState(false);
  const [resourceUploads, setResourceUploads] = useState<ResourceUploadItem[]>([]);
  const [chatResourceUploads, setChatResourceUploads] = useState<ResourceUploadItem[]>([]);
  const [selectedProjectResource, setSelectedProjectResource] = useState<TeachingResourceSummary | null>(null);
  const [historySessions, setHistorySessions] = useState<TeacherSessionResponse[]>([]);
  const [signedOut, setSignedOut] = useState(false);
  const [signOutBusy, setSignOutBusy] = useState(false);
  const [signOutError, setSignOutError] = useState("");
  const [signedOutWarning, setSignedOutWarning] = useState("");
  const requestSlotsRef = useRef(new ModeRequestSlots());
  const runIdentitiesRef = useRef<Record<WorkbenchMode, HarnessRunIdentity | null>>({chat: null, teach: null});
  const runningModesRef = useRef<Record<WorkbenchMode, boolean>>({chat: false, teach: false});
  const retryContextsRef = useRef(new Map<string, TurnRetryContext>());
  const chatResourceUploadsRef = useRef<ResourceUploadItem[]>([]);
  const initializationPromiseRef = useRef<Promise<TeacherSessionResponse[]> | null>(null);
  const projectInitializedRef = useRef(false);
  const offlineWorkspaceHydratedRef = useRef(false);
  const activeProjectIdRef = useRef<string | null>(null);
  const activeChatIdRef = useRef("new-chat");
  const activeTeachingSessionIdRef = useRef<string | null>(null);
  const projectMutationBusyRef = useRef(false);
  const projectMutationGenerationRef = useRef(0);
  const consentRequestsRef = useRef(new Map<RemoteConsentPurpose, Promise<string | null>>());
  const signedOutRef = useRef(false);
  const accountCleanupReceiptRef = useRef<string | null>(null);
  const signOutBusyRef = useRef(false);
  const authenticationBoundaryHandledRef = useRef(false);
  // Keep the teaching conversation as the primary surface on first load.
  // The inspector is opt-in via the header button and never steals width.
  const desktopInspectorPreferenceRef = useRef(false);

  const updateChatResourceUploads = useCallback((
    update: (current: ResourceUploadItem[]) => ResourceUploadItem[],
  ) => {
    // Mirror composer attachments synchronously so two paste/drop events in
    // one render tick cannot reserve more upload slots than the visible bound.
    const next = update(chatResourceUploadsRef.current).slice(0, MAX_CHAT_ATTACHMENTS);
    chatResourceUploadsRef.current = next;
    setChatResourceUploads(next);
  }, []);

  const setModeRunning = useCallback((mode: WorkbenchMode, running: boolean) => {
    runningModesRef.current = {...runningModesRef.current, [mode]: running};
    setRunningModes((current) => current[mode] === running ? current : {...current, [mode]: running});
  }, []);

  const cancelModeRequest = useCallback((mode: WorkbenchMode) => {
    requestSlotsRef.current.cancel(mode);
    setModeRunning(mode, false);
  }, [setModeRunning]);

  const cancelAllModeRequests = useCallback(() => {
    requestSlotsRef.current.cancelAll();
    runningModesRef.current = {chat: false, teach: false};
    setRunningModes({chat: false, teach: false});
    try {
      window.sessionStorage.removeItem(START_KEY_STORAGE_KEY);
    } catch {
      // The identity boundary performs a second best-effort bounded cleanup.
    }
  }, []);

  const bootstrap = useQuery({
    queryKey: ["teacher-agent-bootstrap"],
    queryFn: fetchBootstrap,
    retry: (failureCount, error) => !isHarnessAuthenticationRequired(error) && failureCount < 3,
    retryDelay: (attempt) => Math.min(4_000, 350 * 2 ** attempt),
    refetchInterval: (query) => query.state.status === "error"
      ? isHarnessAuthenticationRequired(query.state.error) ? false : 4_000
      : 15_000,
    refetchIntervalInBackground: true,
    refetchOnMount: "always",
    refetchOnWindowFocus: true,
    staleTime: 30_000,
    networkMode: "always",
    enabled: !signedOut,
  });
  const authenticationRequired = bootstrap.isError && isHarnessAuthenticationRequired(bootstrap.error);
  const initialIdentityCheckPending = !bootstrap.isFetchedAfterMount && !authenticationRequired;
  const authenticatedBootstrapReady = bootstrap.isFetchedAfterMount && Boolean(bootstrap.data) && !authenticationRequired && !signedOut;
  const supportedResourceExtensions = useMemo(() => {
    const value = bootstrap.data?.interaction_contract?.teaching_resource_formats;
    if (!Array.isArray(value)) return [];
    return [...new Set(value.flatMap((item) => {
      if (typeof item !== "string") return [];
      const normalized = item.trim().toLocaleLowerCase().replace(/^\./, "");
      return normalized && /^[a-z0-9]+$/.test(normalized) ? [normalized] : [];
    }))].sort();
  }, [bootstrap.data?.interaction_contract]);
  const projects = useQuery({
    queryKey: ["learning-projects"],
    queryFn: listLearningProjects,
    enabled: authenticatedBootstrapReady,
    retry: 2,
    staleTime: 10_000,
  });
  const trashedProjects = useQuery({
    queryKey: ["trashed-learning-projects"],
    queryFn: listTrashedLearningProjects,
    enabled: authenticatedBootstrapReady,
    retry: 2,
    staleTime: 10_000,
  });

  const signOut = useCallback(async () => {
    if (signOutBusyRef.current) return;
    signOutBusyRef.current = true;
    setSignOutBusy(true);
    setSignOutError("");

    // Detach the UI before crossing the identity boundary. Registered runs
    // receive a best-effort durable cancel while the current session is still
    // usable; a failed cancel never weakens server-side logout semantics.
    const activeRuns = Object.values(runIdentitiesRef.current)
      .filter((identity): identity is HarnessRunIdentity => Boolean(identity));
    runIdentitiesRef.current = {chat: null, teach: null};
    cancelAllModeRequests();
    try {
      await settleRunCancellationsBeforeLogout(
        activeRuns,
        (identity) => cancelHarnessRun(identity, "session_logout_detach"),
      );
      await clearAuthenticatedHarnessSession();
      signedOutRef.current = true;
      setSignedOut(true);
      projectMutationGenerationRef.current += 1;
      projectMutationBusyRef.current = false;
      projectInitializedRef.current = true;
      offlineWorkspaceHydratedRef.current = true;
      activeProjectIdRef.current = null;
      activeChatIdRef.current = "new-chat";
      activeTeachingSessionIdRef.current = null;
      retryContextsRef.current.clear();
      consentRequestsRef.current.clear();
      void queryClient.cancelQueries();
      queryClient.clear();
      setActiveSession(null);
      setHistorySessions([]);
      setTeachingMessages([]);
      setChatMessages([]);
      setChatThreads([]);
      setActiveChatId("new-chat");
      setActiveProjectId(null);
      setActiveProject(null);
      setResourceUploads([]);
      updateChatResourceUploads(() => []);
      setQueuedChatPrompts([]);
      setQueuedSkillId(null);
      setWebSearchEnabled(false);
      setVisualAnalysisEnabled(false);
      setOpenedSyllabus(null);
      setProjectMutationBusy(false);
      setProjectNotice(null);
      setDrawerOpen(false);
      setCommandCenterOpen(false);
      let synchronousStorageCleared = true;
      try {
        synchronousStorageCleared = clearAccountBoundBrowserStorage(
          window.sessionStorage,
          window.localStorage,
        );
      } catch {
        synchronousStorageCleared = false;
      }
      const offlineStorageCleared = await boundedLogoutCleanup(
        clearOfflineRuntimeForLogout,
      );
      setSignedOutWarning(synchronousStorageCleared && offlineStorageCleared
        ? ""
        : "服务端会话已经撤销，但浏览器拒绝清理离线缓存；请清除该站点数据后再交给其他操作者。"
      );
    } catch (error) {
      setSignOutError(error instanceof Error ? error.message : "退出失败，请重试");
    } finally {
      signOutBusyRef.current = false;
      setSignOutBusy(false);
    }
  }, [cancelAllModeRequests, queryClient, updateChatResourceUploads]);

  const reconnectAfterSignOut = useCallback(() => {
    signedOutRef.current = false;
    initializationPromiseRef.current = null;
    projectInitializedRef.current = false;
    offlineWorkspaceHydratedRef.current = false;
    setSignOutError("");
    setSignedOutWarning("");
    setSignedOut(false);
  }, []);

  useEffect(() => {
    if (!authenticationRequired || authenticationBoundaryHandledRef.current) return;
    authenticationBoundaryHandledRef.current = true;
    signedOutRef.current = true;
    projectMutationGenerationRef.current += 1;
    projectMutationBusyRef.current = false;
    projectInitializedRef.current = true;
    offlineWorkspaceHydratedRef.current = true;
    initializationPromiseRef.current = null;
    activeProjectIdRef.current = null;
    activeChatIdRef.current = "new-chat";
    activeTeachingSessionIdRef.current = null;
    runIdentitiesRef.current = {chat: null, teach: null};
    retryContextsRef.current.clear();
    consentRequestsRef.current.clear();
    cancelAllModeRequests();

    const nonBootstrapQueries = {
      predicate: (query: {queryKey: readonly unknown[]}) => query.queryKey[0] !== "teacher-agent-bootstrap",
    };
    void queryClient.cancelQueries(nonBootstrapQueries);
    queryClient.removeQueries(nonBootstrapQueries);

    setActiveSession(null);
    setHistorySessions([]);
    setTeachingMessages([]);
    setChatMessages([]);
    setChatThreads([]);
    setChatHistoryLoaded(false);
    setActiveChatId("new-chat");
    setActiveProjectId(null);
    setActiveProject(null);
    setSelectedProjectResource(null);
    setResourceUploads([]);
    updateChatResourceUploads(() => []);
    setQueuedChatPrompts([]);
    setQueuedSkillId(null);
    setWebSearchEnabled(false);
    setVisualAnalysisEnabled(false);
    setOpenedSyllabus(null);
    setProjectMutationBusy(false);
    setProjectNotice(null);
    setDrawerOpen(false);
    setCommandCenterOpen(false);

    let synchronousStorageCleared = true;
    try {
      synchronousStorageCleared = clearAccountBoundBrowserStorage(
        window.sessionStorage,
        window.localStorage,
      );
    } catch {
      synchronousStorageCleared = false;
    }
    let active = true;
    void boundedLogoutCleanup(clearOfflineRuntimeForLogout).then((offlineStorageCleared) => {
      if (!active) return;
      setSignedOutWarning(synchronousStorageCleared && offlineStorageCleared
        ? ""
        : "身份会话已失效，但浏览器拒绝清理旧账号缓存；请清除该站点数据后再登录。"
      );
    });
    return () => { active = false; };
  }, [authenticationRequired, cancelAllModeRequests, queryClient, updateChatResourceUploads]);

  const ensureRemoteConsent = useCallback(async (purpose: RemoteConsentPurpose) => {
    const pending = consentRequestsRef.current.get(purpose);
    if (pending) return pending;
    const request = (async () => {
      if (!bootstrap.data?.remote_consent?.configured) {
        throw new Error("后端没有配置服务器同意凭据存储；远程处理已关闭");
      }
      const policy = bootstrap.data.remote_consent.policies?.find((item) => item.purpose === purpose);
      if (!policy) throw new Error("后端没有发布这项远程处理的服务商策略");
      if (!policy.remote_processing_eligible) throw new Error("组织的年龄/监护策略未授权这项远程处理");
      const listed = await listRemoteConsents();
      const current = listed.receipts.find((receipt) => receipt.purpose === purpose && activeConsentMatches(receipt, policy));
      if (current) return current.consent_id;
      const accepted = window.confirm([
        consentPurposeMessages[purpose],
        `服务商：${policy.provider_id}；区域：${policy.processing_region}；保留：${policy.provider_retention_days} 天。`,
        `数据类别：${policy.data_categories.join("、")}。`,
        `组织策略：${policy.subject_policy.policy_id} / ${policy.subject_policy.policy_version}；未成年状态与监护授权不能由浏览器自行声明。`,
        `服务商条款：${policy.provider_policy.policy_id} / ${policy.provider_policy.policy_version}；远端副本删除受服务商政策约束。`,
        "是否签发一张 30 天、可随时撤销的服务器同意收据？",
      ].join("\n\n"));
      if (!accepted) return null;
      const granted = await grantRemoteConsent({
        purpose,
        validity_days: 30,
      });
      return granted.consent_id;
    })();
    consentRequestsRef.current.set(purpose, request);
    try {
      return await request;
    } finally {
      if (consentRequestsRef.current.get(purpose) === request) consentRequestsRef.current.delete(purpose);
    }
  }, [bootstrap.data]);

  const associateProjectReference = useCallback(async (
    projectId: string | null,
    kind: "syllabus" | "teaching_session" | "resource",
    referenceId: string | null | undefined,
  ) => {
    if (!projectId || !referenceId) return;
    try {
      let current = await fetchLearningProject(projectId);
      const operationId = `reference:${crypto.randomUUID()}`;
      let project: LearningProject;
      try {
        project = await addLearningProjectReference(projectId, kind, referenceId, {
          expectedUpdatedAt: current.updated_at,
          operationId,
        });
      } catch (reason) {
        if (!(reason instanceof Error) || !reason.message.includes("(409)")) throw reason;
        // A concurrent successful local mutation may advance the CAS between
        // read and commit. Rebase this commutative set-add once; the stable
        // operation ID prevents duplicate effects if the first response was lost.
        current = await fetchLearningProject(projectId);
        project = await addLearningProjectReference(projectId, kind, referenceId, {
          expectedUpdatedAt: current.updated_at,
          operationId,
        });
      }
      setActiveProject((current) => current?.project_id === projectId ? project : current);
      void projects.refetch();
      return true;
    } catch (error) {
      setProjectNotice({
        text: error instanceof Error
          ? `内容已创建，但暂未关联学习项目：${error.message}`
          : "内容已创建，但暂未关联学习项目",
      });
      return false;
    }
  }, [projects]);

  const teachingSessionItems = useMemo<SessionListItem[]>(() => {
    const unique = new Map<string, TeacherSessionResponse>();
    for (const session of projectScopedItems(historySessions, activeProject)) unique.set(session.session_id, session);
    if (activeSession && activeProject?.teaching_session_ids.includes(activeSession.session_id)) unique.set(activeSession.session_id, activeSession);
    return Array.from(unique.values()).map((session) => {
      const goal = session.goal ?? {};
      return {
        id: session.session_id,
        title: String(goal.concept ?? goal.objective ?? "Teaching Agent 会话"),
        learner: String(session.profile_summary?.display_name ?? "本地学生"),
        status: session.status ?? "active",
        round: session.rounds_completed ?? 0,
        group: session.session_id === activeSession?.session_id ? "pinned" : "recent",
      } satisfies SessionListItem;
    });
  }, [activeProject, activeSession, historySessions]);

  const chatSessionItems = useMemo<SessionListItem[]>(() => chatThreads.map((thread) => ({
    id: thread.id,
    title: thread.title,
    learner: "本地用户",
    status: "active",
    round: thread.messages.filter((message) => message.role === "learner").length,
    group: thread.id === activeChatId ? "pinned" : "recent",
  })), [activeChatId, chatThreads]);

  useEffect(() => {
    if (!authenticatedBootstrapReady) return;
    const loaded = parseChatThreads(window.sessionStorage.getItem(CHAT_HISTORY_STORAGE_KEY));
    setChatThreads(loaded);
    setChatHistoryLoaded(true);
  }, [authenticatedBootstrapReady]);

  useEffect(() => {
    if (signedOut || authenticationRequired || initialIdentityCheckPending || offlineWorkspaceHydratedRef.current || activeProject) return;
    if (typeof navigator !== "undefined" && navigator.onLine && !bootstrap.isError) return;
    const remembered = window.localStorage.getItem(ACTIVE_PROJECT_STORAGE_KEY);
    if (!remembered) return;
    offlineWorkspaceHydratedRef.current = true;
    let active = true;
    void loadOfflineWorkspaceProject(remembered).then((cached) => {
      if (!active || !cached || activeProjectIdRef.current) return;
      const project = cached as unknown as LearningProject;
      const threads = project.chat_threads.map(chatThreadFromProject).sort((left, right) => right.updatedAt - left.updatedAt);
      activeProjectIdRef.current = project.project_id;
      setActiveProjectId(project.project_id);
      setActiveProject(project);
      setChatThreads(threads);
      activeChatIdRef.current = threads[0]?.id ?? "new-chat";
      setActiveChatId(threads[0]?.id ?? "new-chat");
      setChatMessages(threads[0]?.messages ?? []);
      setProjectNotice({text: `当前离线：已打开“${project.title}”的只读缓存；联网后会自动核对服务器版本。`});
    }).catch(() => undefined);
    return () => { active = false; };
  }, [activeProject, authenticationRequired, bootstrap.isError, initialIdentityCheckPending, signedOut]);

  useEffect(() => {
    if (!authenticatedBootstrapReady || !activeProject) return;
    void saveOfflineWorkspaceProject(activeProject).catch(() => undefined);
  }, [activeProject, authenticatedBootstrapReady]);

  const changeWebSearchEnabled = useCallback((enabled: boolean) => {
    if (!enabled) {
      setWebSearchEnabled(false);
      return;
    }
    void ensureRemoteConsent("public_web_search").then((consentId) => {
      setWebSearchEnabled(Boolean(consentId));
      if (!consentId) setProjectNotice({text: "未授权远程搜索，联网搜索仍保持关闭。"});
    }).catch((error) => {
      setWebSearchEnabled(false);
      setProjectNotice({text: error instanceof Error ? error.message : "远程搜索授权失败"});
    });
  }, [ensureRemoteConsent]);

  useEffect(() => {
    activeProjectIdRef.current = activeProjectId;
    setSelectedProjectResource(null);
  }, [activeProjectId]);

  useEffect(() => {
    // Staged Chat IDs are project-scoped capabilities. Never carry a composer
    // attachment across a project boundary.
    updateChatResourceUploads(() => []);
  }, [activeProjectId, updateChatResourceUploads]);

  useEffect(() => {
    activeChatIdRef.current = activeChatId;
  }, [activeChatId]);

  useEffect(() => {
    activeTeachingSessionIdRef.current = activeSession?.session_id ?? null;
  }, [activeSession]);

  useEffect(() => {
    try {
      if (invalidateLegacyRemoteConsent()) {
        setProjectNotice({text: "旧版本地同意记录已失效；远程操作会要求重新签发服务器收据。"});
      }
    } catch {
      // Storage can be unavailable in hardened browser contexts. The backend
      // still rejects every legacy flag and requires a server receipt.
    }
  }, []);

  useEffect(() => {
    try {
      const stored = JSON.parse(window.localStorage.getItem("teachlab.console.preferences") ?? "{}") as {webSearchEnabled?: boolean};
      if (typeof stored.webSearchEnabled === "boolean") {
        setWebSearchEnabled(stored.webSearchEnabled);
      }
    } catch {
      // Invalid local preferences fall back to the safe visible default.
    }
  }, []);

  useEffect(() => {
    if (!authenticatedBootstrapReady || projectInitializedRef.current || !projects.data || runningModes.chat || runningModes.teach) return;
    projectInitializedRef.current = true;
    const remembered = window.localStorage.getItem(ACTIVE_PROJECT_STORAGE_KEY);
    const initialId = projects.data.some((project) => project.project_id === remembered)
      ? remembered
      : projects.data.find((project) => project.pinned && project.status === "active")?.project_id
        ?? projects.data.find((project) => project.status === "active")?.project_id;
    const previousProjectId = activeProjectIdRef.current;
    const mutationGeneration = ++projectMutationGenerationRef.current;
    projectMutationBusyRef.current = true;
    activeProjectIdRef.current = initialId ?? null;
    cancelAllModeRequests();
    setProjectMutationBusy(true);
    const legacyThreads = parseChatThreads(window.sessionStorage.getItem(CHAT_HISTORY_STORAGE_KEY));
    const legacySessionIds = rememberedSessionIds();
    const migrationRequired = legacyThreads.length > 0 || legacySessionIds.length > 0;
    const load = initialId && !migrationRequired
      ? fetchLearningProject(initialId)
      : bootstrapLearningProject({
        idempotency_key: "teachlab-local-default-v1",
        ...(initialId ? {project_id: initialId} : {}),
        ...(migrationRequired ? {
          migration_id: legacyMigrationId(),
          legacy_chat_threads: legacyThreads.map(projectChatThread),
          teaching_session_ids: legacySessionIds,
        } : {}),
      }).then((result) => {
        if (result.legacy_migration_applied_or_replayed) {
          window.sessionStorage.removeItem(CHAT_HISTORY_STORAGE_KEY);
          window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
          window.sessionStorage.removeItem(SESSION_INDEX_STORAGE_KEY);
          window.localStorage.removeItem(LEGACY_MIGRATION_STORAGE_KEY);
        }
        return result.project;
      });
    void load.then((project) => {
      if (projectMutationGenerationRef.current !== mutationGeneration) return;
      activeProjectIdRef.current = project.project_id;
      setActiveProjectId(project.project_id);
      setActiveProject(project);
      window.localStorage.setItem(ACTIVE_PROJECT_STORAGE_KEY, project.project_id);
      const threads = project.chat_threads.map(chatThreadFromProject).sort((left, right) => right.updatedAt - left.updatedAt);
      setChatThreads(threads);
      activeChatIdRef.current = threads[0]?.id ?? "new-chat";
      setActiveChatId(threads[0]?.id ?? "new-chat");
      setChatMessages(threads[0]?.messages ?? []);
      // The legacy full-transcript cache exists only as a one-time migration
      // source. The durable project store is authoritative after bootstrap.
      window.sessionStorage.removeItem(CHAT_HISTORY_STORAGE_KEY);
      activeTeachingSessionIdRef.current = null;
      setActiveSession(null);
      setTeachingMessages([]);
    }).catch((error) => {
      if (projectMutationGenerationRef.current !== mutationGeneration) return;
      activeProjectIdRef.current = previousProjectId;
      window.localStorage.removeItem(ACTIVE_PROJECT_STORAGE_KEY);
      setProjectNotice({text: error instanceof Error ? `持久学习空间初始化失败：${error.message}` : "持久学习空间初始化失败"});
    }).finally(() => {
      if (projectMutationGenerationRef.current !== mutationGeneration) return;
      projectMutationBusyRef.current = false;
      setProjectMutationBusy(false);
    });
  }, [authenticatedBootstrapReady, cancelAllModeRequests, projects.data, runningModes.chat, runningModes.teach]);

  useEffect(() => {
    if (!authenticatedBootstrapReady || !activeProjectId || !activeProject || !chatHistoryLoaded || activeChatId === "new-chat") return;
    const thread = chatThreads.find((item) => item.id === activeChatId);
    if (!thread || thread.messages.some((message) => message.streaming)) return;
    const authoritative = activeProject.chat_threads.find((item) => item.thread_id === thread.id);
    const candidate = projectChatThread(thread);
    if (authoritative && JSON.stringify(authoritative) === JSON.stringify(candidate)) return;
    const timer = window.setTimeout(() => {
      void saveProjectChatThread(activeProjectId, candidate).then((project) => {
        if (activeProjectIdRef.current === project.project_id) setActiveProject(project);
        void projects.refetch();
      }).catch((error) => {
        if (activeProjectIdRef.current === activeProjectId) {
          setProjectNotice({text: error instanceof Error ? `对话暂未保存到学习项目：${error.message}` : "对话暂未保存到学习项目"});
        }
      });
    }, 450);
    return () => window.clearTimeout(timer);
  }, [activeChatId, activeProject, activeProjectId, authenticatedBootstrapReady, chatHistoryLoaded, chatThreads, projects]);

  useEffect(() => {
    if (!chatHistoryLoaded || activeChatId === "new-chat") return;
    setChatThreads((current) => current.map((thread) => thread.id === activeChatId ? {
      ...thread,
      // Preserve the live lifecycle in memory. The session/project decoders
      // already reopen a persisted running turn as stopped, while this flag
      // prevents partial streamed text from being committed as a final answer.
      messages: chatMessages,
      updatedAt: Date.now(),
    } : thread).sort((left, right) => right.updatedAt - left.updatedAt));
  }, [activeChatId, chatHistoryLoaded, chatMessages]);

  const closeInspector = useCallback(() => {
    if (!isCompact) desktopInspectorPreferenceRef.current = false;
    setDrawerOpen(false);
  }, [isCompact]);

  const toggleInspector = useCallback(() => {
    setDrawerOpen((open) => {
      const next = !open;
      if (!isCompact) desktopInspectorPreferenceRef.current = next;
      return next;
    });
  }, [isCompact]);

  const openSettings = useCallback(() => {
    if (!isCompact) desktopInspectorPreferenceRef.current = true;
    setDrawerOpen(true);
    setSettingsRequest((current) => current + 1);
    setAccountRightsOpen(true);
  }, [isCompact]);

  const reauthenticateAccountRights = useCallback(async () => {
    if (bootstrap.data?.account_data_rights?.recent_auth_satisfied) return;
    window.location.assign(accountDataRightsStepUpUrl(window.location.pathname));
    await new Promise<never>(() => undefined);
  }, [bootstrap.data?.account_data_rights?.recent_auth_satisfied]);

  const finishCommittedAccountDeletion = useCallback(async (receipt: AccountDeletionReceipt) => {
    if (accountCleanupReceiptRef.current === receipt.receipt_id) return;
    accountCleanupReceiptRef.current = receipt.receipt_id;
    setSignedOutWarning("");
    setAccountDeletedReceipt(receipt);
    signedOutRef.current = true;
    setSignedOut(true);
    runIdentitiesRef.current = {chat: null, teach: null};
    cancelAllModeRequests();
    try {
      await completeCommittedAccountDeletion(receipt, {
        sessionStorage: window.sessionStorage,
        localStorage: window.localStorage,
        clearIndexedDb: clearOfflineRuntimeForLogout,
        detachActiveRuns: cancelAllModeRequests,
        clearMemoryCaches: () => queryClient.clear(),
        cacheStorage: typeof window.caches === "undefined" ? undefined : window.caches,
      });
    } catch (error) {
      accountCleanupReceiptRef.current = null;
      if (error instanceof CommittedAccountCleanupError) {
        setSignedOutWarning(
          "账户已由服务器永久删除，但浏览器拒绝清理部分本机数据；请清除此站点数据后再交给其他操作者。",
        );
      }
      throw error;
    }
  }, [cancelAllModeRequests, queryClient]);

  useEffect(() => {
    if (signedOut) return;
    let active = true;
    let timer: number | undefined;
    let observedDeletion = false;
    const poll = async () => {
      try {
        const status = await accountDeletionStatus();
        if (!active) return;
        setAccountDeletionState(status);
        observedDeletion = status.status === "deleting"
          || status.status === "retryable_failure";
        if (status.status === "permanently_deleted" && status.receipt) {
          await finishCommittedAccountDeletion(status.receipt);
          return;
        }
      } catch {
        // A completed deletion has already removed the login session. Keep the
        // independent HttpOnly-capability probe reachable across a transient
        // API/BFF outage even when bootstrap now renders login-required.
      }
      if (active) {
        const delay = observedDeletion ? 1_500 : authenticationRequired ? 4_000 : 15_000;
        timer = window.setTimeout(() => {void poll();}, delay);
      }
    };
    void poll();
    return () => {
      active = false;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [authenticationRequired, finishCommittedAccountDeletion, signedOut]);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 1260px)");
    const syncDrawer = (event: MediaQueryList | MediaQueryListEvent) => {
      setIsCompact(event.matches);
      setDrawerOpen(event.matches ? false : desktopInspectorPreferenceRef.current);
    };
    syncDrawer(media);
    media.addEventListener("change", syncDrawer);
    return () => media.removeEventListener("change", syncDrawer);
  }, []);

  useEffect(() => () => {requestSlotsRef.current.cancelAll();}, []);

  useEffect(() => {
    if (!authenticatedBootstrapReady || !bootstrap.data || initializationPromiseRef.current) return;
    const initialize = async () => {
      const initialIds = rememberedSessionIds();
      const results = await Promise.all(initialIds.map(async (storedSessionId) => {
        try {
          return await resumeTeachingSession(storedSessionId);
        } catch {
          // Stale or pruned opaque handles are simply omitted from history.
          return null;
        }
      }));
      const sessions = results.filter((session): session is TeacherSessionResponse => Boolean(session));
      const newIds = rememberedSessionIds().filter((id) => !initialIds.includes(id));
      window.sessionStorage.setItem(SESSION_INDEX_STORAGE_KEY, JSON.stringify(Array.from(new Set([...newIds, ...sessions.map((session) => session.session_id)]))));
      return sessions;
    };
    initializationPromiseRef.current = initialize();
    initializationPromiseRef.current.then((sessions) => {
      if (signedOutRef.current) return;
      setHistorySessions((current) => {
        const merged = new Map<string, TeacherSessionResponse>();
        for (const session of current) merged.set(session.session_id, session);
        for (const session of sessions) if (!merged.has(session.session_id)) merged.set(session.session_id, session);
        return Array.from(merged.values());
      });
    }).catch((error) => {
      if (signedOutRef.current) return;
      setTeachingMessages([{id: "bootstrap-error", role: "tool", body: "", createdAt: "刚刚", toolLabel: "Teaching Agent 未连接", toolDetail: error instanceof Error ? error.message : "无法启动真实会话"}]);
    });
  }, [authenticatedBootstrapReady, bootstrap.data]);

  useEffect(() => {
    if (!bootstrap.data) return;
    setTeachingMessages((current) => current.filter((message) => message.id !== "bootstrap-error"));
  }, [bootstrap.data]);

  const appendControlNotice = useCallback((text: string) => {
    setTeachingMessages((current) => [...current, {id: `tool-${crypto.randomUUID()}`, role: "tool", body: "", createdAt: "刚刚", toolLabel: "会话控制", toolDetail: text}]);
  }, []);

  const appendChatNotice = useCallback((text: string) => {
    setChatMessages((current) => [...current, {id: `chat-tool-${crypto.randomUUID()}`, role: "tool", body: "", createdAt: "刚刚", toolLabel: "Chat", toolDetail: text}]);
  }, []);

  const rememberSession = useCallback((session: TeacherSessionResponse) => {
    rememberSessionId(session.session_id);
    setHistorySessions((current) => [session, ...current.filter((item) => item.session_id !== session.session_id)]);
  }, []);

  const activateSessionFromInspector = useCallback((session: TeacherSessionResponse) => {
    activeTeachingSessionIdRef.current = session.session_id;
    setSidebarView("teach");
    setTeachSurface("conversation");
    setQueuedSkillId(null);
    setResourceUploads([]);
    setActiveSession(session);
    setTeachingMessages(messagesFromSession(session));
    rememberSession(session);
    void associateProjectReference(
      activeProjectIdRef.current,
      "teaching_session",
      session.session_id,
    );
  }, [associateProjectReference, rememberSession]);

  const importResources = useCallback(async (files: File[]) => {
    if (!files.length || runningModesRef.current.teach || projectMutationBusyRef.current || !bootstrap.data) return;
    const projectIdAtImport = activeProjectIdRef.current;
    const sessionIdAtImport = activeTeachingSessionIdRef.current;
    const teachingGeneration = requestSlotsRef.current.generation("teach");
    const importIsCurrent = () => requestSlotsRef.current.generation("teach") === teachingGeneration
      && activeProjectIdRef.current === projectIdAtImport
      && activeTeachingSessionIdRef.current === sessionIdAtImport;
    const syllabusFiles = files.filter((file) => file.type === "application/json" || file.name.toLocaleLowerCase().endsWith(".json"));
    const resourceCandidates = files.filter((file) => !syllabusFiles.includes(file));
    const supportedFormats = new Set(supportedResourceExtensions);
    const resourceFiles = resourceCandidates.filter((file) => supportedFormats.has(file.name.split(".").pop()?.toLocaleLowerCase() ?? ""));
    const unsupportedResourceFiles = resourceCandidates.filter((file) => !resourceFiles.includes(file));
    if (unsupportedResourceFiles.length) {
      appendControlNotice(`${unsupportedResourceFiles.map((file) => file.name).join("、")}：当前服务端没有可用的本地提取器。`);
    }
    if (syllabusFiles.length) {
      const syllabusItems = syllabusFiles.map((file) => ({
        localId: `syllabus-${crypto.randomUUID()}`,
        name: file.name,
        status: "extracting" as const,
        detail: "正在校验教学大纲…",
        removable: false,
      }));
      setResourceUploads((current) => [...current, ...syllabusItems]);
      for (let index = 0; index < syllabusFiles.length; index += 1) {
        const file = syllabusFiles[index];
        const item = syllabusItems[index];
        try {
          if (file.size > 384 * 1024) throw new Error("教学大纲 JSON 上传不能超过 384 KB；内容上限由后端按 256 KB 校验");
          const parsed = JSON.parse(await file.text()) as unknown;
          if (!isTeachingSyllabusDocument(parsed)) throw new Error("这不是 teaching_syllabus.v1 教学大纲文件");
          const imported = await importTeachingSyllabus(parsed);
          await associateProjectReference(projectIdAtImport, "syllabus", imported.syllabus_id);
          setResourceUploads((current) => current.filter((entry) => entry.localId !== item.localId));
          setOpenedSyllabus(imported);
          setSyllabusLaunchError("");
          setSidebarView("teach");
          setTeachSurface("syllabi");
          appendControlNotice(`已导入教学大纲：${imported.title}`);
        } catch (error) {
          setResourceUploads((current) => current.map((entry) => entry.localId === item.localId ? {...entry, status: "failed", detail: error instanceof Error ? error.message : "教学大纲导入失败", removable: true} : entry));
          appendControlNotice(error instanceof Error ? `${file.name}：${error.message}` : `${file.name}：教学大纲导入失败`);
        }
      }
    }
    if (!resourceFiles.length) return;
    const existingCount = activeSession?.teaching_resources?.length ?? resourceUploads.filter((item) => item.resource?.staged_resource_id).length;
    const accepted = resourceFiles.slice(0, Math.max(0, 6 - existingCount));
    if (!accepted.length) {
      appendControlNotice("每个教学会话最多导入 6 个教学资源。");
      return;
    }
    if (accepted.length < resourceFiles.length) appendControlNotice("已达到资源上限，只处理前几个文件。");
    const newItems = accepted.map((file) => ({
      localId: `resource-${crypto.randomUUID()}`,
      name: file.name,
      status: "extracting" as const,
      detail: "正在本机提取文字…",
      removable: !activeSession,
    }));
    setResourceUploads((current) => [...current, ...newItems]);
    let sessionCursor = activeSession;
    for (let index = 0; index < accepted.length; index += 1) {
      const file = accepted[index];
      const item = newItems[index];
      try {
        const payload: Record<string, unknown> = {
          resource_idempotency_key: `console-resource-${crypto.randomUUID()}`,
          mime_type: resourceMimeType(file),
          display_name: file.name,
          data_base64: await fileAsBase64(file),
        };
        if (sessionCursor) {
          payload.session_id = sessionCursor.session_id;
          payload.expected_round = sessionCursor.rounds_completed ?? 0;
          payload.expected_question_id = sessionCursor.expected_question_id;
          payload.expected_context_version = sessionCursor.context_version;
          payload.profile_revision = sessionCursor.profile_summary?.profile_revision;
        }
        const response = await uploadTeachingResource(payload);
        await associateProjectReference(projectIdAtImport, "resource", response.resource.resource_id);
        if (!importIsCurrent()) return;
        if (sessionCursor) {
          if (typeof response.session_id !== "string" || typeof response.context_version !== "number") {
            throw new Error("教学资源已提取，但后端没有返回更新后的会话");
          }
          sessionCursor = response as TeacherSessionResponse;
          setActiveSession(sessionCursor);
          rememberSession(sessionCursor);
          appendControlNotice(`已导入教学资源：${response.resource.display_name}`);
        }
        setResourceUploads((current) => current.map((entry) => entry.localId === item.localId ? {
          ...entry,
          status: response.resource.truncated ? "truncated" : "ready",
          detail: response.resource.truncated
            ? `已提取 ${response.resource.extracted_char_count ?? 0} 字，超长内容已截断`
            : response.resource.needs_review
              ? "已提取，OCR 结果建议人工核对"
              : `已提取 ${response.resource.extracted_char_count ?? 0} 字`,
          resource: response.resource,
          removable: !sessionCursor,
        } : entry));
      } catch (error) {
        if (!importIsCurrent()) return;
        setResourceUploads((current) => current.map((entry) => entry.localId === item.localId ? {
          ...entry,
          status: "failed",
          detail: error instanceof Error ? error.message : "教学资源导入失败",
          removable: true,
        } : entry));
      }
    }
  }, [activeSession, appendControlNotice, associateProjectReference, bootstrap.data, rememberSession, resourceUploads, supportedResourceExtensions]);

  const importChatResources = useCallback(async (files: File[]) => {
    if (!files.length || runningModesRef.current.chat || projectMutationBusyRef.current || !bootstrap.data) return;
    const projectIdAtImport = activeProjectIdRef.current;
    if (!projectIdAtImport) {
      appendChatNotice("请先打开一个持久学习项目，再添加 Chat 附件。");
      return;
    }
    const selected = selectChatAttachments(files, chatResourceUploadsRef.current.length, supportedResourceExtensions);
    if (selected.rejected.length) {
      appendChatNotice(selected.rejected.map((item) => `${item.file.name}：${item.message}`).join("；"));
    }
    if (!selected.accepted.length) return;
    const chatGeneration = requestSlotsRef.current.generation("chat");
    const importIsCurrent = () => requestSlotsRef.current.generation("chat") === chatGeneration
      && activeProjectIdRef.current === projectIdAtImport;
    const newItems: ResourceUploadItem[] = selected.accepted.map((file) => ({
      localId: `chat-resource-${crypto.randomUUID()}`,
      name: file.name,
      mimeType: chatAttachmentMimeType(file),
      status: "extracting",
      detail: "正在本机提取；原始媒体不会发送给远程模型",
      removable: true,
    }));
    updateChatResourceUploads((current) => [...current, ...newItems]);
    for (let index = 0; index < selected.accepted.length; index += 1) {
      const file = selected.accepted[index];
      const item = newItems[index];
      try {
        const response = await uploadTeachingResource({
          resource_idempotency_key: `console-chat-resource-${crypto.randomUUID()}`,
          mime_type: chatAttachmentMimeType(file),
          display_name: file.name,
          data_base64: await fileAsBase64(file),
        });
        const associated = await associateProjectReference(
          projectIdAtImport,
          "resource",
          response.resource.resource_id,
        );
        if (!importIsCurrent()) return;
        if (!associated) throw new Error("资源未能关联到当前学习项目");
        const blocked = response.session_use?.status === "blocked_pending_confirmation"
          || response.session_use?.status === "blocked_abstained";
        updateChatResourceUploads((current) => current.map((entry) => entry.localId === item.localId ? {
          ...entry,
          status: blocked ? "blocked" : response.resource.truncated ? "truncated" : "ready",
          detail: blocked
            ? response.session_use?.status === "blocked_pending_confirmation"
              ? "检测到跨层冲突，需具备权限的服务端复核版本；本附件不会发送给模型"
              : "本地提取无法核验语义，需具备权限的服务端复核版本；本附件不会发送给模型"
            : response.resource.truncated
              ? `本机已提取 ${response.resource.extracted_char_count ?? 0} 字，超长内容已截断；Chat 只会检索相关小片段`
              : `本机已提取 ${response.resource.extracted_char_count ?? 0} 字；Chat 只会检索相关小片段`,
          resource: response.resource,
        } : entry));
      } catch (error) {
        if (!importIsCurrent()) return;
        updateChatResourceUploads((current) => current.map((entry) => entry.localId === item.localId ? {
          ...entry,
          status: "failed",
          detail: error instanceof Error ? error.message : "Chat 附件本地提取失败",
        } : entry));
      }
    }
  }, [appendChatNotice, associateProjectReference, bootstrap.data, supportedResourceExtensions, updateChatResourceUploads]);

  const removeResourceUpload = useCallback((localId: string) => {
    setResourceUploads((current) => current.filter((item) => item.localId !== localId || item.removable === false));
  }, []);

  const removeChatResourceUpload = useCallback((localId: string) => {
    updateChatResourceUploads((current) => current.filter((item) => item.localId !== localId));
  }, [updateChatResourceUploads]);

  const runCommand = useCallback(async (text: string) => {
    if (!activeSession || runningModesRef.current.teach) return;
    const trimmed = text.trim();
    const commandPayload: Record<string, unknown> = {
      session_id: activeSession.session_id,
      expected_round: activeSession.rounds_completed ?? 0,
      expected_question_id: activeSession.expected_question_id,
      expected_context_version: activeSession.context_version,
      profile_revision: activeSession.profile_summary?.profile_revision,
      command_idempotency_key: `console-command-${crypto.randomUUID()}`,
    };
    if (trimmed === "/auto") commandPayload.command = "auto";
    else if (trimmed === "/stop") commandPayload.command = "stop";
    else if (trimmed.startsWith("/+skill")) {
      const query = trimmed.slice("/+skill".length).trim().toLocaleLowerCase();
      const skill = (bootstrap.data?.skills ?? []).find((item) => item.skill_id.toLocaleLowerCase() === query || item.name.toLocaleLowerCase() === query);
      if (!skill || skill.role === "support") {
        appendControlNotice("未找到可锁定的 primary Skill；请使用 /+skill 加 Skill ID 或完整名称。" );
        return;
      }
      commandPayload.command = "select_skill";
      commandPayload.skill_id = skill.skill_id;
    } else {
      appendControlNotice("未知命令。可用命令：/auto、/+skill <Skill>、/stop。" );
      return;
    }
    const projectIdAtRequest = activeProjectIdRef.current;
    const sessionIdAtRequest = activeSession.session_id;
    const abortController = new AbortController();
    const token = requestSlotsRef.current.begin("teach", {projectId: projectIdAtRequest, surfaceId: sessionIdAtRequest}, abortController);
    const requestIsCurrent = () => requestSlotsRef.current.isCurrent(token, {
      projectId: activeProjectIdRef.current,
      surfaceId: activeTeachingSessionIdRef.current,
    });
    setTeachingMessages((current) => [...current, {id: `command-${crypto.randomUUID()}`, role: "learner", body: trimmed, createdAt: "刚刚"}]);
    setModeRunning("teach", true);
    try {
      const response = await sendTeachingCommand(commandPayload);
      if (!requestIsCurrent()) return;
      setActiveSession(response);
      rememberSession(response);
      if (trimmed === "/stop") {
        const terminal = teacherMessage(`terminal-${response.context_version}`, currentAction(response));
        if (terminal) setTeachingMessages((current) => [...current, terminal]);
      } else {
        appendControlNotice(trimmed === "/auto" ? "已恢复自动路由。" : `已锁定 ${String(commandPayload.skill_id)}，从下一次学习者回答生效。`);
      }
    } catch (error) {
      if (requestIsCurrent()) appendControlNotice(error instanceof Error ? error.message : "会话命令执行失败");
    } finally {
      if (requestSlotsRef.current.finish(token, abortController)) setModeRunning("teach", false);
    }
  }, [activeSession, appendControlNotice, bootstrap.data?.skills, rememberSession, setModeRunning]);

  const sendChat = useCallback(async (text: string, webSearch = false, retryMessageId?: string) => {
    const retryContext = retryMessageId ? retryContextsRef.current.get(retryMessageId) : undefined;
    const chatRetry = retryContext?.mode === "chat" ? retryContext : undefined;
    const trimmed = chatRetry?.prompt ?? text.trim();
    if (!trimmed || runningModesRef.current.chat || projectMutationBusyRef.current || !bootstrap.data) return;
    if (!chatRetry && chatResourceUploads.some((item) => item.status === "failed" || item.status === "blocked")) {
      appendChatNotice("有附件提取失败或等待权威复核；请移除后再发送。没有任何附件内容被远程处理。");
      return;
    }
    const resourceItems = chatRetry?.resourceItems ?? chatResourceUploads.filter((item) => item.status === "ready" || item.status === "truncated");
    const resourceRefs = chatRetry?.resourceRefs ?? chatResourceRefs(resourceItems);
    let chatConsentId: string | null = null;
    try {
      chatConsentId = await ensureRemoteConsent("remote_chat");
    } catch (error) {
      appendChatNotice(error instanceof Error ? error.message : "远程 Chat 授权失败");
      return;
    }
    if (!chatConsentId) {
      appendChatNotice("未签发服务器同意收据，Chat 请求没有发送。");
      return;
    }
    if (chatRetry && chatRetry.threadId !== activeChatId) {
      appendChatNotice("请先打开原对话，再重试这一回合。");
      return;
    }
    const retryMessage = retryMessageId ? chatMessages.find((message) => message.id === retryMessageId) : undefined;
    if (chatRetry && !retryMessage) {
      appendChatNotice("原回答已不在当前对话中，无法安全重试。");
      return;
    }
    const forksCompletedAnswer = Boolean(chatRetry && retryMessage?.role === "teacher" && retryMessage.status === "completed");
    const sourceThread = chatThreads.find((thread) => thread.id === activeChatId);
    const projectIdAtRequest = activeProjectIdRef.current;
    const abortController = new AbortController();
    const threadId = forksCompletedAnswer
      ? newChatThreadId()
      : chatRetry?.threadId ?? (activeChatId === "new-chat" ? newChatThreadId() : activeChatId);
    const responseMessageId = forksCompletedAnswer || !retryMessageId
      ? `chat-assistant-${crypto.randomUUID()}`
      : retryMessageId;
    const retryIndex = chatRetry && retryMessageId ? chatMessages.findIndex((message) => message.id === retryMessageId) : -1;
    const completedBranch = forksCompletedAnswer && retryMessageId
      ? prepareCompletedTurnBranch(chatMessages, retryMessageId, responseMessageId, "DeepSeek 正在生成分支回答…")
      : null;
    const nextMessages: TeachingMessage[] = completedBranch ?? (chatRetry
      ? prepareTurnRetry(retryIndex >= 0 ? chatMessages.slice(0, retryIndex + 1) : chatMessages, responseMessageId, "DeepSeek 正在重新生成…")
      : [
          ...chatMessages,
          {id: `chat-user-${crypto.randomUUID()}`, role: "learner", body: trimmed, createdAt: "刚刚", attachmentLabels: resourceItems.map((item) => item.name).slice(0, 6)},
          {id: responseMessageId, role: "teacher", body: "", thinking: "正在生成回答…", createdAt: "刚刚", streaming: true, status: "running", retryable: true},
        ]);
    const requestMessages = chatRetry?.requestMessages ?? chatTurnsFromMessages([
      ...chatMessages,
      {id: "pending-request", role: "learner", body: trimmed, createdAt: "刚刚"},
    ]);
    const webSearchRequestedByTurn = chatRetry?.webSearch ?? webSearch;
    let webSearchConsentId: string | null = null;
    if (webSearchRequestedByTurn) {
      try {
        webSearchConsentId = await ensureRemoteConsent("public_web_search");
      } catch (error) {
        appendChatNotice(error instanceof Error ? error.message : "联网搜索授权失败");
      }
    }
    const requestedWebSearch = webSearchRequestedByTurn && Boolean(webSearchConsentId);
    if (webSearchRequestedByTurn && !requestedWebSearch) {
      setWebSearchEnabled(false);
      appendChatNotice("已取消联网搜索；本条消息将只使用当前对话上下文。");
    }
    rememberRetryContext(retryContextsRef.current, responseMessageId, {
      mode: "chat",
      threadId,
      prompt: trimmed,
      webSearch: requestedWebSearch,
      requestMessages,
      resourceRefs,
      resourceItems: resourceItems.map((item) => ({...item})),
    });
    activeChatIdRef.current = threadId;
    const token = requestSlotsRef.current.begin("chat", {projectId: projectIdAtRequest, surfaceId: threadId}, abortController);
    runIdentitiesRef.current.chat = null;
    const requestIsCurrent = () => requestSlotsRef.current.isCurrent(token, {
      projectId: activeProjectIdRef.current,
      surfaceId: activeChatIdRef.current,
    });
    setModeRunning("chat", true);
    setSidebarView("chat");
    setActiveChatId(threadId);
    setChatMessages(nextMessages);
    if (!chatRetry && resourceRefs.length) updateChatResourceUploads(() => []);
    setChatThreads((current) => {
      if (current.some((thread) => thread.id === threadId)) return current;
      const now = Date.now();
      const title = forksCompletedAnswer
        ? chatBranchTitle(sourceThread?.title ?? trimmed.slice(0, 42))
        : trimmed.slice(0, 42);
      return [{id: threadId, title, messages: nextMessages, createdAt: now, updatedAt: now}, ...current];
    });
    try {
      const response = await streamChat(requestMessages, {
        remoteConsentId: chatConsentId,
        webSearch: requestedWebSearch,
        webSearchConsentId: requestedWebSearch ? webSearchConsentId ?? undefined : undefined,
        projectId: projectIdAtRequest ?? undefined,
        chatThreadId: projectIdAtRequest ? threadId : undefined,
        resourceRefs,
      }, {
        onRunIdentity: (identity) => {
          if (requestIsCurrent()) runIdentitiesRef.current.chat = identity;
        },
        onStatus: (label) => {
          if (!requestIsCurrent()) return;
          setChatMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, thinking: label} : message));
        },
        onDelta: (delta) => {
          if (!requestIsCurrent()) return;
          setChatMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, body: `${message.body}${delta}`, thinking: undefined} : message));
        },
        onMeta: ({webSearchUsed, sourceCount}) => {
          if (!requestIsCurrent() || !webSearchUsed) return;
          setChatMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, thinking: sourceCount ? `已核对 ${sourceCount} 个来源，正在输出…` : "已完成联网搜索，正在输出…"} : message));
        },
        onLifecycle: (update) => {
          if (!requestIsCurrent()) return;
          setChatMessages((current) => upsertHarnessLifecycleMessage(current, responseMessageId, update));
        },
      }, abortController.signal);
      if (!requestIsCurrent()) return;
      setChatMessages((current) => settleTurn(current, responseMessageId, "completed").map((message) => message.id === responseMessageId ? {
        ...message,
        body: message.body || response.message,
        webSearchUsed: Boolean(response.web_search_used),
        sources: response.sources ?? [],
      } : message));
    } catch (error) {
      if (!requestIsCurrent() && isAbortError(error)) return;
      if (requestIsCurrent()) {
        const detail = error instanceof Error ? error.message : "Chat 回答失败，请重试。";
        if (!chatRetry && resourceItems.length) {
          updateChatResourceUploads((current) => current.length ? current : resourceItems.map((item) => ({...item})));
        }
        setChatMessages((current) => settleHarnessLifecycleMessages(
          settleTurn(current, responseMessageId, "failed", {error: detail}),
          responseMessageId,
          "运行已中断",
        ));
      }
    } finally {
      if (requestSlotsRef.current.finish(token, abortController)) {
        runIdentitiesRef.current.chat = null;
        setModeRunning("chat", false);
      }
    }
  }, [activeChatId, appendChatNotice, bootstrap.data, chatMessages, chatResourceUploads, chatThreads, ensureRemoteConsent, setModeRunning, updateChatResourceUploads]);

  const queueChatFollowUp = useCallback((text: string, options?: {webSearch?: boolean}) => {
    const trimmed = text.trim();
    if (!trimmed || !runningModesRef.current.chat || queuedChatPrompts.length >= 5) return false;
    setQueuedChatPrompts((current) => [...current, {
      id: `chat-follow-up-${crypto.randomUUID()}`,
      threadId: activeChatIdRef.current,
      text: trimmed,
      webSearch: Boolean(options?.webSearch),
      status: "queued",
    }]);
    return true;
  }, [queuedChatPrompts.length]);

  const cancelQueuedChatPrompt = useCallback((id: string) => {
    setQueuedChatPrompts((current) => current.filter((prompt) => prompt.id !== id));
  }, []);

  useEffect(() => {
    if (!authenticatedBootstrapReady) return;
    const next = nextChatFollowUp(queuedChatPrompts, runningModes.chat, activeChatIdRef.current);
    if (!next) return;
    setQueuedChatPrompts((current) => current.filter((prompt) => prompt.id !== next.id));
    void sendChat(next.text, next.webSearch);
  }, [authenticatedBootstrapReady, queuedChatPrompts, runningModes.chat, sendChat]);

  const sendTeaching = useCallback(async (text: string, attachment?: File, retryMessageId?: string) => {
    if (runningModesRef.current.teach || projectMutationBusyRef.current || !bootstrap.data) return;
    const retryContext = retryMessageId ? retryContextsRef.current.get(retryMessageId) : undefined;
    const teachingRetry = retryContext?.mode === "teach" ? retryContext : undefined;
    const effectiveAttachment = teachingRetry?.attachment ?? attachment;
    const sessionForRequest = teachingRetry?.session ?? activeSession;
    if (resourceUploads.some((item) => item.status === "extracting")) {
      appendControlNotice("教学资源仍在本机提取，请等待完成后再发送。");
      return;
    }
    const trimmed = teachingRetry?.prompt ?? text.trim();
    if (!sessionForRequest && trimmed.startsWith("/")) {
      appendControlNotice("请先发送学习目标或问题建立会话，再使用 /auto、/+skill 或 /stop。");
      return;
    }
    if (!sessionForRequest && effectiveAttachment) {
      appendControlNotice("请先用文字建立会话，随后即可发送图片。");
      return;
    }
    if (!sessionForRequest && !trimmed) return;
    if (sessionForRequest && !effectiveAttachment && trimmed.startsWith("/") && !teachingRetry) {
      await runCommand(text);
      return;
    }
    let teachingConsentId: string | null = null;
    const remoteTeachingRequired = bootstrap.data.interaction_contract?.remote_processing_server_consent_required === true
      || bootstrap.data.provider_status?.configured === true;
    if (remoteTeachingRequired) {
      try {
        teachingConsentId = await ensureRemoteConsent("remote_teaching");
      } catch (error) {
        appendControlNotice(error instanceof Error ? error.message : "远程 Teach 授权失败");
        return;
      }
      if (!teachingConsentId) {
        appendControlNotice("未签发服务器同意收据，Teach 请求没有发送。");
        return;
      }
    }
    const projectIdAtRequest = activeProjectIdRef.current;
    const abortController = new AbortController();
    const sessionIdAtRequest = sessionForRequest?.session_id ?? null;
    const token = requestSlotsRef.current.begin("teach", {projectId: projectIdAtRequest, surfaceId: sessionIdAtRequest}, abortController);
    runIdentitiesRef.current.teach = null;
    const requestIsCurrent = () => requestSlotsRef.current.isCurrent(token, {
      projectId: activeProjectIdRef.current,
      surfaceId: activeTeachingSessionIdRef.current,
    });
    setSidebarView("teach");
    setModeRunning("teach", true);
    const learnerBody = trimmed || (effectiveAttachment ? `[图片] ${effectiveAttachment.name}` : "");
    const responseMessageId = retryMessageId ?? `teacher-${crypto.randomUUID()}`;
    setTeachingMessages((current) => teachingRetry
      ? prepareTurnRetry(current, responseMessageId, "正在安全重试这一回合…")
      : [...current,
          {id: `learner-${crypto.randomUUID()}`, role: "learner", body: learnerBody, createdAt: "刚刚"},
          {id: responseMessageId, role: "teacher", body: "", thinking: "正在生成回答…", createdAt: "刚刚", streaming: true, status: "running", retryable: true},
        ]);
    try {
      if (!sessionForRequest) {
        const startIdempotencyKey = typeof teachingRetry?.startPayload?.start_idempotency_key === "string"
          ? teachingRetry.startPayload.start_idempotency_key
          : `console-start-${crypto.randomUUID()}`;
        const requestedInitialSkillId = teachingRetry?.startSkillId ?? queuedSkillId;
        window.sessionStorage.setItem(START_KEY_STORAGE_KEY, startIdempotencyKey);
        const stagedResourceIds = resourceUploads.flatMap((item) => item.resource?.staged_resource_id ? [item.resource.staged_resource_id] : []);
        const baseRequestPayload = teachingRetry?.startPayload ?? startPayload(bootstrap.data, startIdempotencyKey, trimmed, stagedResourceIds);
        const requestPayload = {
          ...baseRequestPayload,
          ...(requestedInitialSkillId ? {manual_skill_id: requestedInitialSkillId} : {}),
          ...(teachingConsentId ? {remote_consent_id: teachingConsentId} : {}),
        };
        rememberRetryContext(retryContextsRef.current, responseMessageId, {
          mode: "teach",
          operation: "start",
          prompt: trimmed,
          startPayload: requestPayload,
          startSkillId: requestedInitialSkillId,
        });
        let response = await streamTeachingStart(requestPayload, {
          onRunIdentity: (identity) => {
            if (requestIsCurrent()) runIdentitiesRef.current.teach = identity;
          },
          onStatus: (label) => {
            if (!requestIsCurrent()) return;
            setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, thinking: label} : message));
          },
          onMeta: ({skill}) => {
            if (!requestIsCurrent()) return;
            setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, skill, thinking: "正在生成回答…"} : message));
          },
          onDelta: (delta) => {
            if (!requestIsCurrent()) return;
            setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, body: `${message.body}${delta}`, thinking: undefined} : message));
          },
          onLifecycle: (update) => {
            if (!requestIsCurrent()) return;
            setTeachingMessages((current) => upsertHarnessLifecycleMessage(current, responseMessageId, update));
          },
        }, abortController.signal);
        await associateProjectReference(projectIdAtRequest, "teaching_session", response.session_id);
        if (!requestIsCurrent()) return;
        window.sessionStorage.removeItem(START_KEY_STORAGE_KEY);
        setQueuedSkillId(null);
        activeTeachingSessionIdRef.current = response.session_id;
        setActiveSession(response);
        rememberSession(response);
        setResourceUploads([]);
        retryContextsRef.current.delete(responseMessageId);
        const action = currentAction(response);
        const lessonPhase = action?.lesson_phase ?? response.lesson_progress ?? undefined;
        setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {
          ...message,
          body: actionMessage(action),
          skill: message.skill ?? actionSkill(action),
          lessonPhase,
          streaming: false,
          thinking: undefined,
          status: "completed",
          retryable: false,
        } : message));
        return;
      }
      const attachmentIdempotencyKey = teachingRetry?.attachmentIdempotencyKey ?? (effectiveAttachment ? `console-attachment-${crypto.randomUUID()}` : undefined);
      const turnIdempotencyKey = teachingRetry?.turnIdempotencyKey ?? `console-turn-${crypto.randomUUID()}`;
      rememberRetryContext(retryContextsRef.current, responseMessageId, {
        mode: "teach",
        operation: "step",
        prompt: trimmed,
        attachment: effectiveAttachment,
        session: sessionForRequest,
        attachmentIdempotencyKey,
        turnIdempotencyKey,
      });
      let turnContextVersion = sessionForRequest.context_version;
      let attachmentIds: string[] = [];
      if (effectiveAttachment) {
        if (!effectiveAttachment.type.startsWith("image/")) throw new Error("Teaching Agent 目前只接受图片附件");
        let visualConsentId: string | null = null;
        if (visualAnalysisEnabled && bootstrap.data.visual_semantics?.sends_raw_media_remotely) {
          visualConsentId = await ensureRemoteConsent("remote_visual_analysis");
          if (!visualConsentId) throw new Error("未授权原始图片远程处理；图片没有上传");
        }
        const uploaded = await uploadTeachingAttachment({
          session_id: sessionForRequest.session_id,
          expected_round: sessionForRequest.rounds_completed ?? 0,
          expected_question_id: sessionForRequest.expected_question_id,
          expected_context_version: sessionForRequest.context_version,
          profile_revision: sessionForRequest.profile_summary?.profile_revision,
          attachment_idempotency_key: attachmentIdempotencyKey,
          mime_type: effectiveAttachment.type,
          display_name: effectiveAttachment.name,
          data_base64: await fileAsBase64(effectiveAttachment),
          visual_analysis_requested: visualAnalysisEnabled,
          ...(visualConsentId ? {visual_consent_id: visualConsentId} : {}),
        });
        if (!requestIsCurrent()) return;
        turnContextVersion = uploaded.context_version;
        attachmentIds = [uploaded.attachment.attachment_id];
      }
      const response = await streamTeachingTurn({
        session_id: sessionForRequest.session_id,
        learner_response: trimmed,
        expected_round: sessionForRequest.rounds_completed ?? 0,
        expected_question_id: sessionForRequest.expected_question_id,
        expected_context_version: turnContextVersion,
        profile_revision: sessionForRequest.profile_summary?.profile_revision,
        idempotency_key: turnIdempotencyKey,
        attachment_ids: attachmentIds,
        ...(teachingConsentId ? {remote_consent_id: teachingConsentId} : {}),
      }, {
        onRunIdentity: (identity) => {
          if (requestIsCurrent()) runIdentitiesRef.current.teach = identity;
        },
        onStatus: (label) => {
          if (!requestIsCurrent()) return;
          setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, thinking: label} : message));
        },
        onMeta: ({skill}) => {
          if (!requestIsCurrent()) return;
          setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, skill, thinking: "正在生成回答…"} : message));
        },
        onDelta: (delta) => {
          if (!requestIsCurrent()) return;
          setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, body: `${message.body}${delta}`, thinking: undefined} : message));
        },
        onLifecycle: (update) => {
          if (!requestIsCurrent()) return;
          setTeachingMessages((current) => upsertHarnessLifecycleMessage(current, responseMessageId, update));
        },
      }, abortController.signal);
      await associateProjectReference(projectIdAtRequest, "teaching_session", response.session_id);
      if (!requestIsCurrent()) return;
      activeTeachingSessionIdRef.current = response.session_id;
      setActiveSession(response);
      rememberSession(response);
      setResourceUploads([]);
      retryContextsRef.current.delete(responseMessageId);
      const action = currentAction(response);
      const lessonPhase = action?.lesson_phase ?? response.lesson_progress ?? undefined;
      setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {
        ...message,
        body: actionMessage(action),
        skill: message.skill ?? actionSkill(action),
        lessonPhase,
        streaming: false,
        thinking: undefined,
        status: "completed",
        retryable: false,
      } : message));
    } catch (error) {
      if (!requestIsCurrent() && isAbortError(error)) return;
      if (requestIsCurrent()) {
        window.sessionStorage.removeItem(START_KEY_STORAGE_KEY);
        const detail = error instanceof Error ? error.message : "Teaching Agent 回答失败，请重试。";
        setTeachingMessages((current) => settleHarnessLifecycleMessages(
          settleTurn(current, responseMessageId, "failed", {error: detail}),
          responseMessageId,
          "运行已中断",
        ));
      }
    } finally {
      if (requestSlotsRef.current.finish(token, abortController)) {
        runIdentitiesRef.current.teach = null;
        setModeRunning("teach", false);
      }
    }
  }, [activeSession, appendControlNotice, associateProjectReference, bootstrap.data, ensureRemoteConsent, queuedSkillId, rememberSession, resourceUploads, runCommand, setModeRunning, visualAnalysisEnabled]);

  const startSyllabusLesson = useCallback(async (syllabus: TeachingSyllabus, module: SyllabusModule, lesson: SyllabusLesson, authoritativeStart: SyllabusLessonStartPayload) => {
    if (runningModesRef.current.teach) throw new Error("请等待当前 Teach 回合结束后再开始新课节");
    if (projectMutationBusyRef.current) throw new Error("请等待学习项目切换完成");
    if (!bootstrap.data) throw new Error("Teaching Agent 后端尚未连接");
    const remoteTeachingRequired = bootstrap.data.interaction_contract?.remote_processing_server_consent_required === true
      || bootstrap.data.provider_status?.configured === true;
    const teachingConsentId = remoteTeachingRequired
      ? await ensureRemoteConsent("remote_teaching")
      : null;
    if (remoteTeachingRequired && !teachingConsentId) throw new Error("未签发服务器同意收据，课节没有开始");
    const projectIdAtRequest = activeProjectIdRef.current;
    const abortController = new AbortController();
    const responseMessageId = `teacher-${crypto.randomUUID()}`;
    const learnerMessageId = `syllabus-lesson-${crypto.randomUUID()}`;
    const learnerMessage = `按教学大纲开始：${module.title} / ${lesson.title}`;
    const startIdempotencyKey = `console-syllabus-${crypto.randomUUID()}`;
    const goal: Record<string, unknown> = Object.keys(authoritativeStart.goal).length ? authoritativeStart.goal : syllabusLessonGoal(bootstrap.data, syllabus, module, lesson);
    const goalSyllabusRef = goal.syllabus_ref && typeof goal.syllabus_ref === "object" && !Array.isArray(goal.syllabus_ref) ? goal.syllabus_ref as Record<string, unknown> : undefined;
    const syllabusRef = authoritativeStart.syllabus_ref ?? goalSyllabusRef;
    if (!syllabusRef || typeof syllabusRef.content_sha256 !== "string") throw new Error("后端没有返回完整的课节大纲引用");
    const stagedResourceIds = Array.from(new Set([
      ...(authoritativeStart.staged_resource_ids?.filter((item): item is string => typeof item === "string" && Boolean(item)) ?? []),
      ...resourceUploads.flatMap((item) => item.resource?.staged_resource_id ? [item.resource.staged_resource_id] : []),
    ])).slice(0, 6);
    activeTeachingSessionIdRef.current = null;
    const token = requestSlotsRef.current.begin("teach", {projectId: projectIdAtRequest, surfaceId: null}, abortController);
    runIdentitiesRef.current.teach = null;
    const requestIsCurrent = () => requestSlotsRef.current.isCurrent(token, {
      projectId: activeProjectIdRef.current,
      surfaceId: activeTeachingSessionIdRef.current,
    });
    window.sessionStorage.setItem(START_KEY_STORAGE_KEY, startIdempotencyKey);
    if (activeSession) rememberSession(activeSession);
    setOpenedSyllabus(syllabus);
    setSyllabusLaunchError("");
    setSidebarView("teach");
    setTeachSurface("conversation");
    setQueuedSkillId(null);
    setActiveSession(null);
    setTeachingMessages([
      {id: learnerMessageId, role: "learner", body: learnerMessage, createdAt: "刚刚"},
      {id: responseMessageId, role: "teacher", body: "", thinking: "正在生成回答…", createdAt: "刚刚", streaming: true, status: "running", retryable: true},
    ]);
    setModeRunning("teach", true);
    try {
      const payload = startPayload(bootstrap.data, startIdempotencyKey, lesson.title, stagedResourceIds, goal);
      payload.syllabus_ref = syllabusRef;
      if (teachingConsentId) payload.remote_consent_id = teachingConsentId;
      rememberRetryContext(retryContextsRef.current, responseMessageId, {
        mode: "teach",
        operation: "start",
        prompt: learnerMessage,
        startPayload: payload,
        startSkillId: null,
      });
      const response = await streamTeachingStart(payload, {
        onRunIdentity: (identity) => {
          if (requestIsCurrent()) runIdentitiesRef.current.teach = identity;
        },
        onStatus: (label) => {
          if (!requestIsCurrent()) return;
          setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, thinking: label} : message));
        },
        onMeta: ({skill}) => {
          if (!requestIsCurrent()) return;
          setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, skill, thinking: "正在生成回答…"} : message));
        },
        onDelta: (delta) => {
          if (!requestIsCurrent()) return;
          setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {...message, body: `${message.body}${delta}`, thinking: undefined} : message));
        },
        onLifecycle: (update) => {
          if (!requestIsCurrent()) return;
          setTeachingMessages((current) => upsertHarnessLifecycleMessage(current, responseMessageId, update));
        },
      }, abortController.signal);
      await associateProjectReference(projectIdAtRequest, "syllabus", syllabus.syllabus_id);
      await associateProjectReference(projectIdAtRequest, "teaching_session", response.session_id);
      if (!requestIsCurrent()) return;
      window.sessionStorage.removeItem(START_KEY_STORAGE_KEY);
      activeTeachingSessionIdRef.current = response.session_id;
      setActiveSession(response);
      rememberSession(response);
      retryContextsRef.current.delete(responseMessageId);
      // Staged resources now belong to the newly-created teaching session.
      // Keep them visible until this point so a failed lesson launch can be
      // retried without asking the teacher to upload the source files again.
      setResourceUploads([]);
      const action = currentAction(response);
      const lessonPhase = action?.lesson_phase ?? response.lesson_progress ?? undefined;
      setTeachingMessages((current) => current.map((message) => message.id === responseMessageId ? {
        ...message,
        body: actionMessage(action),
        skill: message.skill ?? actionSkill(action),
        lessonPhase,
        streaming: false,
        thinking: undefined,
        status: "completed",
        retryable: false,
      } : message));
    } catch (error) {
      if (!requestIsCurrent() && isAbortError(error)) return;
      if (requestIsCurrent()) {
        window.sessionStorage.removeItem(START_KEY_STORAGE_KEY);
        const detail = error instanceof Error ? `课节启动失败：${error.message}` : "课节启动失败";
        setTeachingMessages((current) => settleHarnessLifecycleMessages(
          settleTurn(current, responseMessageId, "failed", {error: detail}),
          responseMessageId,
          "运行已中断",
        ));
        setSyllabusLaunchError(detail);
        setTeachSurface("conversation");
      }
    } finally {
      if (requestSlotsRef.current.finish(token, abortController)) {
        runIdentitiesRef.current.teach = null;
        setModeRunning("teach", false);
      }
    }
  }, [activeSession, associateProjectReference, bootstrap.data, ensureRemoteConsent, rememberSession, resourceUploads, setModeRunning]);

  const send = useCallback((text: string, attachment?: File, options?: {webSearch?: boolean}) => {
    if (sidebarView === "chat") return sendChat(text, Boolean(options?.webSearch));
    return sendTeaching(text, attachment);
  }, [sendChat, sendTeaching, sidebarView]);

  const retryTurn = useCallback((messageId: string) => {
    const context = retryContextsRef.current.get(messageId);
    if (!context || runningModesRef.current[context.mode]) return;
    if (context.mode === "chat") {
      setSidebarView("chat");
      void sendChat(context.prompt, context.webSearch, messageId);
      return;
    }
    setSidebarView("teach");
    setTeachSurface("conversation");
    void sendTeaching(context.prompt, context.attachment, messageId);
  }, [sendChat, sendTeaching]);

  const stopMode = useCallback((mode: WorkbenchMode) => {
    if (!runningModesRef.current[mode]) return;
    const runIdentity = runIdentitiesRef.current[mode];
    runIdentitiesRef.current[mode] = null;
    if (runIdentity) void cancelHarnessRun(runIdentity, "user_requested").catch(() => undefined);
    const generation = requestSlotsRef.current.cancel(mode);
    setModeRunning(mode, false);
    if (mode === "chat") {
      setChatMessages((current) => settleRunningHarnessLifecycleMessages(current.map((message) => message.streaming ? {
        ...message,
        status: "stopped",
        streaming: false,
        thinking: undefined,
        error: undefined,
        retryable: true,
      } : message), "已停止"));
      return;
    }
    window.sessionStorage.removeItem(START_KEY_STORAGE_KEY);
    const session = activeSession;
    const projectIdAtCancel = activeProjectIdRef.current;
    const sessionIdAtCancel = activeTeachingSessionIdRef.current;
    setTeachingMessages((current) => settleRunningHarnessLifecycleMessages(current.map((message) => message.streaming ? {...message, status: "stopped", thinking: undefined, streaming: false, retryable: true} : message), "已停止"));
    if (!session) return;
    void sendTeachingCommand({
      session_id: session.session_id,
      expected_round: session.rounds_completed ?? 0,
      expected_question_id: session.expected_question_id,
      expected_context_version: session.context_version,
      profile_revision: session.profile_summary?.profile_revision,
      command: "cancel_turn",
      command_idempotency_key: `console-cancel-${crypto.randomUUID()}`,
    }).then((response) => {
      if (requestSlotsRef.current.generation("teach") !== generation
        || activeProjectIdRef.current !== projectIdAtCancel
        || activeTeachingSessionIdRef.current !== sessionIdAtCancel) return;
      activeTeachingSessionIdRef.current = response.session_id;
      setActiveSession(response);
      rememberSession(response);
      appendControlNotice("已停止当前生成，后端会话已保留。" );
    }).catch(async () => {
      try {
        const refreshed = await resumeTeachingSession(session.session_id);
        if (requestSlotsRef.current.generation("teach") !== generation
          || activeProjectIdRef.current !== projectIdAtCancel
          || activeTeachingSessionIdRef.current !== sessionIdAtCancel) return;
        activeTeachingSessionIdRef.current = refreshed.session_id;
        setActiveSession(refreshed);
        rememberSession(refreshed);
      } catch {
        if (requestSlotsRef.current.generation("teach") === generation
          && activeProjectIdRef.current === projectIdAtCancel
          && activeTeachingSessionIdRef.current === sessionIdAtCancel) appendControlNotice("已停止前端输出；后端状态将在下次打开历史会话时刷新。" );
      }
    });
  }, [activeSession, appendControlNotice, rememberSession, setModeRunning]);

  const stop = useCallback(() => {
    stopMode(sidebarView);
  }, [sidebarView, stopMode]);

  const fenceAllModesForProjectChange = useCallback(() => {
    if (runningModesRef.current.chat) {
      const interrupted = settleRunningHarnessLifecycleMessages(chatMessages.map((message) => message.streaming ? {
        ...message,
        status: "stopped" as const,
        streaming: false,
        thinking: undefined,
        retryable: true,
      } : message), "项目切换，已停止");
      setChatMessages(interrupted);
      if (activeChatIdRef.current !== "new-chat") {
        setChatThreads((current) => current.map((thread) => thread.id === activeChatIdRef.current ? {
          ...thread,
          messages: interrupted,
          updatedAt: Date.now(),
        } : thread));
      }
    }
    if (runningModesRef.current.teach) {
      setTeachingMessages((current) => settleRunningHarnessLifecycleMessages(current.map((message) => message.streaming ? {
        ...message,
        status: "stopped" as const,
        streaming: false,
        thinking: undefined,
        retryable: true,
      } : message), "项目切换，已停止"));
    }
    cancelAllModeRequests();
  }, [cancelAllModeRequests, chatMessages]);

  const startNewTeaching = useCallback(() => {
    for (const message of teachingMessages) retryContextsRef.current.delete(message.id);
    cancelModeRequest("teach");
    window.sessionStorage.removeItem(START_KEY_STORAGE_KEY);
    if (activeSession) rememberSession(activeSession);
    setQueuedSkillId(null);
    activeTeachingSessionIdRef.current = null;
    setActiveSession(null);
    setTeachSurface("conversation");
    setTeachingMessages([]);
    setResourceUploads([]);
    setSidebarOpen(false);
  }, [activeSession, cancelModeRequest, rememberSession, teachingMessages]);

  const startNewChat = useCallback(() => {
    for (const message of chatMessages) retryContextsRef.current.delete(message.id);
    if (runningModesRef.current.chat && activeChatId !== "new-chat") {
      const interrupted = settleRunningHarnessLifecycleMessages(chatMessages.map((message) => message.streaming ? {
        ...message,
        status: "stopped" as const,
        streaming: false,
        thinking: undefined,
        retryable: true,
      } : message), "已停止");
      setChatThreads((current) => current.map((thread) => thread.id === activeChatId ? {...thread, messages: interrupted, updatedAt: Date.now()} : thread));
    }
    cancelModeRequest("chat");
    activeChatIdRef.current = "new-chat";
    setActiveChatId("new-chat");
    setChatMessages([]);
    updateChatResourceUploads(() => []);
    setQueuedChatPrompts([]);
    setSidebarOpen(false);
  }, [activeChatId, cancelModeRequest, chatMessages, updateChatResourceUploads]);

  const startNew = useCallback(() => {
    if (sidebarView === "chat") startNewChat();
    else startNewTeaching();
  }, [sidebarView, startNewChat, startNewTeaching]);

  const openChat = useCallback(() => setSidebarView("chat"), []);
  const openTeach = useCallback(() => {
    setSidebarView("teach");
    setTeachSurface("conversation");
  }, []);
  const openSyllabi = useCallback(() => {
    setSidebarView("teach");
    setTeachSurface("syllabi");
    setSyllabusLaunchError("");
  }, []);

  const runCommandCenterAction = useCallback((action: CommandCenterActionId) => {
    if (action === "new_current") startNew();
    else if (action === "open_chat") openChat();
    else if (action === "open_teach") openTeach();
    else if (action === "open_syllabi") openSyllabi();
    else if (action === "open_settings") openSettings();
    else if (action === "stop_chat") stopMode("chat");
    else if (action === "stop_teach") stopMode("teach");
  }, [openChat, openSettings, openSyllabi, openTeach, startNew, stopMode]);

  const selectTeachingSession = useCallback((sessionId: string) => {
    if (sessionId === activeSession?.session_id && !runningModesRef.current.teach) return;
    const generation = requestSlotsRef.current.cancel("teach");
    setModeRunning("teach", false);
    window.sessionStorage.removeItem(START_KEY_STORAGE_KEY);
    setQueuedSkillId(null);
    setResourceUploads([]);
    const cached = historySessions.find((session) => session.session_id === sessionId) ?? null;
    const projectIdAtSelection = activeProjectIdRef.current;
    activeTeachingSessionIdRef.current = sessionId;
    setSidebarView("teach");
    setTeachSurface("conversation");
    setActiveSession(cached);
    if (cached) {
      setTeachingMessages(messagesFromSession(cached));
    } else {
      setTeachingMessages([]);
    }
    setSidebarOpen(false);
    if (!bootstrap.data) return;
    void resumeTeachingSession(sessionId).then((response) => {
      if (requestSlotsRef.current.generation("teach") !== generation
        || activeProjectIdRef.current !== projectIdAtSelection
        || activeTeachingSessionIdRef.current !== sessionId) return;
      activeTeachingSessionIdRef.current = response.session_id;
      setActiveSession(response);
      rememberSession(response);
      setTeachingMessages(messagesFromSession(response));
    }).catch((error) => {
      if (requestSlotsRef.current.generation("teach") !== generation
        || activeProjectIdRef.current !== projectIdAtSelection
        || activeTeachingSessionIdRef.current !== sessionId) return;
      setHistorySessions((current) => current.filter((session) => session.session_id !== sessionId));
      activeTeachingSessionIdRef.current = null;
      setActiveSession(null);
      setTeachingMessages([]);
      appendControlNotice(error instanceof Error ? error.message : "无法恢复历史会话");
    });
  }, [activeSession, appendControlNotice, bootstrap.data, historySessions, rememberSession, setModeRunning]);

  const selectChatThread = useCallback((threadId: string) => {
    if (threadId === activeChatId) return;
    const thread = chatThreads.find((item) => item.id === threadId);
    if (!thread) return;
    if (runningModesRef.current.chat) {
      const interrupted = settleRunningHarnessLifecycleMessages(chatMessages.map((message) => message.streaming ? {
        ...message,
        status: "stopped" as const,
        streaming: false,
        thinking: undefined,
        retryable: true,
      } : message), "已停止");
      setChatThreads((current) => current.map((item) => item.id === activeChatId ? {...item, messages: interrupted, updatedAt: Date.now()} : item));
      cancelModeRequest("chat");
    }
    activeChatIdRef.current = thread.id;
    setActiveChatId(thread.id);
    setChatMessages(thread.messages);
    updateChatResourceUploads(() => []);
    setQueuedChatPrompts((current) => current.filter((prompt) => prompt.threadId === thread.id));
    setSidebarView("chat");
    setSidebarOpen(false);
  }, [activeChatId, cancelModeRequest, chatMessages, chatThreads, updateChatResourceUploads]);

  const selectSession = useCallback((id: string) => {
    if (sidebarView === "chat") selectChatThread(id);
    else selectTeachingSession(id);
  }, [selectChatThread, selectTeachingSession, sidebarView]);

  const chooseSkill = useCallback(async (skillId: string | null) => {
    const skill = skillId ? (bootstrap.data?.skills ?? []).find((item) => item.skill_id === skillId && item.role !== "support") : null;
    if (skillId && !skill) return;
    if (runningModesRef.current.teach) {
      appendControlNotice("当前回合仍在运行；停止或等待完成后再切换教学 Skill。");
      return;
    }
    setSidebarView("teach");
    setTeachSurface("conversation");
    if (!activeSession) {
      setQueuedSkillId(skillId);
      return;
    }
    const abortController = new AbortController();
    const token = requestSlotsRef.current.begin("teach", {
      projectId: activeProjectIdRef.current,
      surfaceId: activeSession.session_id,
    }, abortController);
    const requestIsCurrent = () => requestSlotsRef.current.isCurrent(token, {
      projectId: activeProjectIdRef.current,
      surfaceId: activeTeachingSessionIdRef.current,
    });
    setModeRunning("teach", true);
    try {
      const response = skillId ? await selectTeachingSkill(activeSession, skillId) : await sendTeachingCommand({
        session_id: activeSession.session_id,
        expected_round: activeSession.rounds_completed ?? 0,
        expected_question_id: activeSession.expected_question_id,
        expected_context_version: activeSession.context_version,
        profile_revision: activeSession.profile_summary?.profile_revision,
        command: "auto",
        command_idempotency_key: `console-auto-${crypto.randomUUID()}`,
      });
      if (!requestIsCurrent()) return;
      setQueuedSkillId(null);
      activeTeachingSessionIdRef.current = response.session_id;
      setActiveSession(response);
      rememberSession(response);
      appendControlNotice(skill ? `已通过后端锁定 ${skill.name}，从下一次学习者回答生效。` : "已通过后端恢复自动教学路由。");
    } catch (error) {
      if (requestIsCurrent()) appendControlNotice(error instanceof Error ? error.message : "Skill 路由失败");
    } finally {
      if (requestSlotsRef.current.finish(token, abortController)) setModeRunning("teach", false);
    }
  }, [activeSession, appendControlNotice, bootstrap.data?.skills, rememberSession, setModeRunning]);

  const selectProject = useCallback(async (projectId: string) => {
    if (projectId === activeProjectIdRef.current || projectMutationBusyRef.current) return;
    const previousProjectId = activeProjectIdRef.current;
    const mutationGeneration = ++projectMutationGenerationRef.current;
    projectMutationBusyRef.current = true;
    fenceAllModesForProjectChange();
    activeProjectIdRef.current = projectId;
    retryContextsRef.current.clear();
    setProjectMutationBusy(true);
    try {
      const project = await fetchLearningProject(projectId);
      if (projectMutationGenerationRef.current !== mutationGeneration) return;
      const threads = project.chat_threads.map(chatThreadFromProject).sort((left, right) => right.updatedAt - left.updatedAt);
      activeProjectIdRef.current = project.project_id;
      setActiveProjectId(project.project_id);
      setActiveProject(project);
      window.localStorage.setItem(ACTIVE_PROJECT_STORAGE_KEY, project.project_id);
      setChatThreads(threads);
      activeChatIdRef.current = threads[0]?.id ?? "new-chat";
      setActiveChatId(threads[0]?.id ?? "new-chat");
      setChatMessages(threads[0]?.messages ?? []);
      setQueuedChatPrompts([]);
      activeTeachingSessionIdRef.current = null;
      setActiveSession(null);
      setTeachingMessages([]);
      setResourceUploads([]);
      setSidebarView("chat");
      setProjectNotice({text: `已打开学习项目“${project.title}”`});
    } catch (error) {
      if (projectMutationGenerationRef.current !== mutationGeneration) return;
      activeProjectIdRef.current = previousProjectId;
      setProjectNotice({text: error instanceof Error ? error.message : "学习项目打开失败"});
    } finally {
      if (projectMutationGenerationRef.current === mutationGeneration) {
        projectMutationBusyRef.current = false;
        setProjectMutationBusy(false);
      }
    }
  }, [fenceAllModesForProjectChange]);

  const createProject = useCallback(async (title: string) => {
    if (projectMutationBusyRef.current) return;
    const previousProjectId = activeProjectIdRef.current;
    const mutationGeneration = ++projectMutationGenerationRef.current;
    projectMutationBusyRef.current = true;
    fenceAllModesForProjectChange();
    // An unknown project is deliberately distinct from the previous project
    // while the create request is in flight, so stale stream callbacks fail closed.
    activeProjectIdRef.current = null;
    retryContextsRef.current.clear();
    setProjectMutationBusy(true);
    try {
      const project = await createLearningProject({title, operation_id: `create:${crypto.randomUUID()}`});
      if (projectMutationGenerationRef.current !== mutationGeneration) return;
      activeProjectIdRef.current = project.project_id;
      setActiveProjectId(project.project_id);
      setActiveProject(project);
      window.localStorage.setItem(ACTIVE_PROJECT_STORAGE_KEY, project.project_id);
      setChatThreads([]);
      activeChatIdRef.current = "new-chat";
      setActiveChatId("new-chat");
      setChatMessages([]);
      setQueuedChatPrompts([]);
      activeTeachingSessionIdRef.current = null;
      setActiveSession(null);
      setTeachingMessages([]);
      setResourceUploads([]);
      setSidebarView("chat");
      setProjectNotice({text: `已创建学习项目“${project.title}”`});
      await projects.refetch();
    } catch (error) {
      if (projectMutationGenerationRef.current !== mutationGeneration) return;
      activeProjectIdRef.current = previousProjectId;
      setProjectNotice({text: error instanceof Error ? error.message : "学习项目创建失败"});
    } finally {
      if (projectMutationGenerationRef.current === mutationGeneration) {
        projectMutationBusyRef.current = false;
        setProjectMutationBusy(false);
      }
    }
  }, [fenceAllModesForProjectChange, projects]);

  const updateProject = useCallback(async (projectId: string, change: {title?: string; description?: string; status?: "active" | "archived"; pinned?: boolean}, success: string) => {
    if (projectMutationBusyRef.current) return;
    projectMutationBusyRef.current = true;
    setProjectMutationBusy(true);
    try {
      const currentRevision = activeProject?.project_id === projectId
        ? activeProject.updated_at
        : projects.data?.find((project) => project.project_id === projectId)?.updated_at;
      if (!currentRevision) throw new Error("项目版本不可用，请刷新项目列表后重试");
      const project = await updateLearningProject(projectId, {
        ...change,
        expected_updated_at: currentRevision,
        operation_id: `update:${crypto.randomUUID()}`,
      });
      if (activeProjectId === projectId) setActiveProject(project);
      setProjectNotice({text: success});
      await projects.refetch();
    } catch (error) {
      setProjectNotice({text: error instanceof Error ? error.message : "学习项目更新失败"});
    } finally {
      projectMutationBusyRef.current = false;
      setProjectMutationBusy(false);
    }
  }, [activeProject, activeProjectId, projects]);

  const trashProject = useCallback(async (projectId: string) => {
    if (projectMutationBusyRef.current) return;
    const title = projects.data?.find((project) => project.project_id === projectId)?.title ?? "学习项目";
    projectMutationBusyRef.current = true;
    setProjectMutationBusy(true);
    try {
      if (activeProjectId === projectId) fenceAllModesForProjectChange();
      const trashed = await trashLearningProject(projectId);
      forgetSessionIds(trashed.browser_session_handles_to_forget);
      const forgotten = new Set(trashed.browser_session_handles_to_forget);
      setHistorySessions((current) => current.filter((session) => !forgotten.has(session.session_id)));
      if (activeProjectId === projectId) {
        activeProjectIdRef.current = null;
        setActiveProjectId(null);
        setActiveProject(null);
        window.localStorage.removeItem(ACTIVE_PROJECT_STORAGE_KEY);
        setChatThreads([]);
        activeChatIdRef.current = "new-chat";
        setActiveChatId("new-chat");
        setChatMessages([]);
        activeTeachingSessionIdRef.current = null;
        setActiveSession(null);
        setTeachingMessages([]);
        setResourceUploads([]);
      }
      setProjectNotice({text: `“${title}”已移入可恢复废纸篓`, projectId, recoveryToken: trashed.recovery_token});
      await Promise.all([projects.refetch(), trashedProjects.refetch()]);
    } catch (error) {
      setProjectNotice({text: error instanceof Error ? error.message : "移入废纸篓失败"});
    } finally {
      projectMutationBusyRef.current = false;
      setProjectMutationBusy(false);
    }
  }, [activeProjectId, fenceAllModesForProjectChange, projects, trashedProjects]);

  const undoTrashProject = useCallback(async () => {
    if (!projectNotice?.projectId || !projectNotice.recoveryToken || projectMutationBusyRef.current) return;
    projectMutationBusyRef.current = true;
    setProjectMutationBusy(true);
    try {
      const project = await restoreLearningProject(projectNotice.projectId, projectNotice.recoveryToken);
      setProjectNotice({text: `已恢复“${project.title}”`});
      await Promise.all([projects.refetch(), trashedProjects.refetch()]);
    } catch (error) {
      setProjectNotice({text: error instanceof Error ? error.message : "学习项目恢复失败"});
    } finally {
      projectMutationBusyRef.current = false;
      setProjectMutationBusy(false);
    }
  }, [projectNotice, projects, trashedProjects]);

  const restoreTrashedProject = useCallback(async (item: LearningProjectTrashItem) => {
    if (projectMutationBusyRef.current) return;
    projectMutationBusyRef.current = true;
    setProjectMutationBusy(true);
    try {
      const project = await restoreLearningProject(item.project_id, item.recovery_token);
      setProjectNotice({text: `已恢复“${project.title}”`});
      await Promise.all([projects.refetch(), trashedProjects.refetch()]);
    } catch (error) {
      setProjectNotice({text: error instanceof Error ? error.message : "学习项目恢复失败"});
    } finally {
      projectMutationBusyRef.current = false;
      setProjectMutationBusy(false);
    }
  }, [projects, trashedProjects]);

  const exportProject = useCallback(async (item: LearningProjectSummary) => {
    if (projectMutationBusyRef.current) return;
    projectMutationBusyRef.current = true;
    setProjectMutationBusy(true);
    try {
      const exported = await exportLearningProject(item.project_id);
      const url = URL.createObjectURL(exported.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = exported.filename;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
      setProjectNotice({text: `“${item.title}”私有数据已导出 · manifest ${exported.manifestSha256.slice(0, 12)}…`});
    } catch (error) {
      setProjectNotice({text: error instanceof Error ? error.message : "学习项目导出失败"});
    } finally {
      projectMutationBusyRef.current = false;
      setProjectMutationBusy(false);
    }
  }, []);

  const purgeTrashedProject = useCallback(async (item: LearningProjectTrashItem) => {
    if (projectMutationBusyRef.current) return;
    const confirmation = `PERMANENTLY DELETE ${item.project_id}`;
    const entered = window.prompt(
      `此操作会永久删除“${item.title}”及其独占的教学会话、大纲、资源索引和运行日志，且无法撤销。\n\n请输入以下确认文本：\n${confirmation}`,
    );
    if (entered === null) return;
    if (entered !== confirmation) {
      setProjectNotice({text: "确认文本不匹配，未删除任何数据。"});
      return;
    }
    projectMutationBusyRef.current = true;
    setProjectMutationBusy(true);
    try {
      const receipt = await purgeLearningProject(item.project_id, item.recovery_token, confirmation);
      let browserCacheRemoved = true;
      try {
        await deleteOfflineWorkspaceProject(item.project_id);
      } catch {
        browserCacheRemoved = false;
      }
      setProjectNotice({text: browserCacheRemoved
        ? `“${item.title}”已永久删除 · 收据 ${receipt.receipt_id}`
        : `“${item.title}”的服务器数据已永久删除 · 收据 ${receipt.receipt_id}；浏览器缓存清理失败，请在浏览器设置中删除此站点数据。`});
      await Promise.all([projects.refetch(), trashedProjects.refetch()]);
    } catch (error) {
      setProjectNotice({text: error instanceof Error ? error.message : "永久删除失败；未确认完成前请视为数据仍被保留"});
    } finally {
      projectMutationBusyRef.current = false;
      setProjectMutationBusy(false);
    }
  }, [projects, trashedProjects]);

  const openProjectSyllabus = useCallback(async (syllabusId: string) => {
    const projectId = activeProjectIdRef.current;
    try {
      const syllabus = await fetchTeachingSyllabus(syllabusId);
      if (activeProjectIdRef.current !== projectId) return;
      setOpenedSyllabus(syllabus);
      setSyllabusLaunchError("");
      setSidebarView("teach");
      setTeachSurface("syllabi");
    } catch (error) {
      setProjectNotice({text: error instanceof Error ? error.message : "项目大纲打开失败"});
    }
  }, []);

  const openProjectResource = useCallback((resourceId: string, metadata?: TeachingResourceSummary) => {
    const session = [activeSession, ...projectScopedItems(historySessions, activeProject)].find((item) => item?.teaching_resources?.some((resource) => resource.resource_id === resourceId));
    const sessionMetadata = session?.teaching_resources?.find((resource) => resource.resource_id === resourceId);
    setSelectedProjectResource(metadata ?? sessionMetadata ?? null);
    if (session) selectTeachingSession(session.session_id);
    else {
      setSidebarView("teach");
      setTeachSurface("conversation");
      setProjectNotice({text: `资源 ${resourceId} 已关联项目；打开相关教学会话后可查看提取详情。`});
    }
    if (!isCompact) desktopInspectorPreferenceRef.current = true;
    setDrawerOpen(true);
    setResourcesRequest((current) => current + 1);
  }, [activeProject, activeSession, historySessions, isCompact, selectTeachingSession]);

  const applyReviewedResource = useCallback((reviewed: TeachingResourceSummary) => {
    const replaceUpload = (item: ResourceUploadItem): ResourceUploadItem => item.resource?.resource_id === reviewed.resource_id ? {
      ...item,
      status: "ready",
      detail: `认证教师复核 v${reviewed.resource_review?.review_version ?? "—"} 已保存；仅作为教学上下文`,
      resource: reviewed,
    } : item;
    const replaceSession = (session: TeacherSessionResponse): TeacherSessionResponse => {
      const resources = session.teaching_resources;
      if (!resources?.some((resource) => resource.resource_id === reviewed.resource_id)) return session;
      return {...session, teaching_resources: resources.map((resource) => replaceMatchingTeachingResource(resource, reviewed))};
    };
    setResourceUploads((current) => current.map(replaceUpload));
    updateChatResourceUploads((current) => current.map(replaceUpload));
    setActiveSession((current) => current ? replaceSession(current) : current);
    setHistorySessions((current) => current.map(replaceSession));
    setSelectedProjectResource((current) => current?.resource_id === reviewed.resource_id ? reviewed : current);
    setProjectNotice({text: `${reviewed.display_name} 的认证复核已保存；不会作为答案键、评分或掌握度证据。`});
  }, [updateChatResourceUploads]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey)) return;
      if (event.key.toLowerCase() === "k") {
        event.preventDefault();
        setCommandCenterOpen((open) => !open);
      } else if (event.key.toLowerCase() === "n") {
        event.preventDefault();
        startNew();
      } else if (event.key === ",") {
        event.preventDefault();
        openSettings();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [openSettings, startNew]);

  useEffect(() => {
    const onModeShortcut = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.shiftKey || event.altKey) return;
      if (event.key === "1") {
        event.preventDefault();
        setSidebarView("chat");
      } else if (event.key === "2") {
        event.preventDefault();
        setSidebarView("teach");
        setTeachSurface("conversation");
      }
    };
    window.addEventListener("keydown", onModeShortcut);
    return () => window.removeEventListener("keydown", onModeShortcut);
  }, []);

  const activeSessionId = activeSession?.session_id ?? "new-teaching";
  const activeGoal = activeSession?.goal ?? {};
  const activeChatThread = chatThreads.find((thread) => thread.id === activeChatId);
  const headerTitle = sidebarView === "chat"
    ? activeChatThread?.title ?? "新对话"
    : activeSession ? String(activeGoal.concept ?? "Teaching Agent") : "新教学";
  const headerSubtitle = sidebarView === "chat"
    ? "Chat · 直接对话"
    : activeSession ? String(activeGoal.objective ?? actionSkill(currentAction(activeSession)) ?? "真实教学会话") : "Teach · 自适应教学";
  const visibleMessages = sidebarView === "chat" ? chatMessages : teachingMessages;
  const visibleItems = sidebarView === "chat" ? chatSessionItems : teachingSessionItems;
  const visibleActiveId = sidebarView === "chat" ? activeChatId : activeSessionId;
  const syllabusSourceResources = Array.from(new Map(
    resourceUploads
      .flatMap((item) => item.resource?.staged_resource_id ? [{id: item.resource.staged_resource_id, name: item.name}] : [])
      .map((resource) => [resource.id, resource] as const),
  ).values()).slice(0, 6);
  const inspectorTeachingResources = useMemo(() => uniqueTeachingResources([
    ...(activeSession?.teaching_resources ?? []),
    ...resourceUploads.flatMap((item) => item.resource ? [item.resource] : []),
    ...chatResourceUploads.flatMap((item) => item.resource ? [item.resource] : []),
    ...(selectedProjectResource ? [selectedProjectResource] : []),
  ]), [activeSession?.teaching_resources, chatResourceUploads, resourceUploads, selectedProjectResource]);
  const visibleProjects = useMemo(() => {
    const values = projects.data ?? [];
    if (!activeProject) return values;
    const summary = projectSummary(activeProject);
    return [summary, ...values.filter((project) => project.project_id !== summary.project_id)]
      .sort((left, right) => Number(right.pinned) - Number(left.pinned) || right.updated_at.localeCompare(left.updated_at));
  }, [activeProject, projects.data]);
  const identityBoundary = workbenchIdentityBoundary({
    signedOut,
    authenticationRequired,
    bootstrapPending: initialIdentityCheckPending,
  });
  const deletionRecoveryActive = accountDeletionState?.status === "deleting"
    || accountDeletionState?.status === "retryable_failure";
  if (deletionRecoveryActive) {
    return (
      <main className="grid h-dvh min-h-0 place-items-center bg-[var(--app-bg)] px-4 text-[var(--app-text)]">
        <section className="w-full max-w-2xl rounded-2xl border border-[var(--app-border-strong)] bg-[var(--app-surface-raised)] p-6 shadow-2xl">
          <AccountDataRightsPanel
            authenticated
            authenticatedAppsApi
            status={accountDeletionState}
            exportAccount={downloadAccountExport}
            reauthenticate={reauthenticateAccountRights}
            prepareDeletion={prepareAccountDeletion}
            confirmDeletion={confirmAccountDeletion}
            refreshStatus={accountDeletionStatus}
            resumeDeletion={resumeAccountDeletion}
            statusChanged={setAccountDeletionState}
            committedDeletion={finishCommittedAccountDeletion}
          />
        </section>
      </main>
    );
  }
  const loginEntryAvailable = authenticationRequired || organizationLoginAvailable();
  if (identityBoundary === "login_required") {
    return (
      <main className="grid h-dvh min-h-0 place-items-center bg-[var(--app-bg)] px-6 text-[var(--app-text)]">
        <section className="w-full max-w-md rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-raised)] p-7 text-center shadow-[0_20px_70px_var(--app-shadow)]" aria-labelledby="signed-out-title">
          <span className="mx-auto grid size-11 place-items-center rounded-xl bg-[var(--app-accent-soft)]" aria-hidden="true">✓</span>
          <h1 id="signed-out-title" className="mt-4 text-lg font-semibold">{accountDeletedReceipt ? "TeachLab 账户数据已删除" : signedOut ? "已安全退出 TeachLab" : "登录 TeachLab"}</h1>
          <p className="mt-2 text-sm leading-6 text-[var(--app-muted)]">{accountDeletedReceipt ? `本 TeachLab 部署的在线主存储及全部设备会话授权已删除。收据：${accountDeletedReceipt.receipt_id}。组织 IdP 账户、运维备份和远程服务商副本仍由各自管理方及保留政策控制。` : signedOut ? "当前设备的服务端会话已撤销，页面运行也已停止或分离。请通过组织身份入口重新登录。" : "使用组织身份账号建立安全会话后，即可进入工作台。"}</p>
          {signedOutWarning && <p role="alert" className="mt-3 rounded-lg bg-[var(--app-danger-soft)] px-3 py-2 text-left text-xs leading-5 text-[var(--app-danger)]">{signedOutWarning}</p>}
          {accountDeletedReceipt && signedOutWarning && (
            <button
              type="button"
              onClick={() => {void finishCommittedAccountDeletion(accountDeletedReceipt).catch(() => undefined);}}
              className="mt-4 rounded-lg bg-[var(--app-accent)] px-4 py-2 text-sm font-medium text-white hover:opacity-90"
            >
              重试清理本机账户数据
            </button>
          )}
          {!accountDeletedReceipt && loginEntryAvailable ? (
            <a href={organizationLoginUrl("/")} className="mt-5 inline-flex rounded-lg bg-[var(--app-accent)] px-4 py-2 text-sm font-medium text-white hover:opacity-90">使用组织账号登录</a>
          ) : !accountDeletedReceipt ? (
            <button type="button" onClick={reconnectAfterSignOut} className="mt-5 rounded-lg bg-[var(--app-accent)] px-4 py-2 text-sm font-medium text-white hover:opacity-90">重新检查登录状态</button>
          ) : null}
        </section>
      </main>
    );
  }
  if (identityBoundary === "checking") {
    return (
      <main className="grid h-dvh min-h-0 place-items-center bg-[var(--app-bg)] px-6 text-[var(--app-text)]">
        <div role="status" aria-live="polite" className="flex items-center gap-3 text-sm text-[var(--app-muted)]">
          <span className="size-2 animate-pulse rounded-full bg-[var(--app-accent)]" aria-hidden="true" />
          正在确认 TeachLab 身份会话…
        </div>
      </main>
    );
  }
  return (
    <div className="flex h-dvh min-h-0 w-full overflow-hidden bg-[var(--app-bg)] text-[var(--app-text)] transition-colors duration-150">
      <div className="flex min-w-0 flex-1" aria-hidden={isCompact && drawerOpen ? true : undefined} inert={isCompact && drawerOpen ? true : undefined}>
        <SessionSidebar
          items={visibleItems}
          activeId={visibleActiveId}
          activeView={sidebarView}
          syllabiOpen={sidebarView === "teach" && teachSurface === "syllabi"}
          onSelect={selectSession}
          onChat={openChat}
          onTeach={openTeach}
          onSyllabi={openSyllabi}
          onNew={startNew}
          onSettings={openSettings}
          onSignOut={() => {void signOut();}}
          signOutAvailable={authenticatedHarnessLogoutAvailable()}
          signOutBusy={signOutBusy}
          signOutError={signOutError}
          onCommandCenter={() => setCommandCenterOpen(true)}
          onChooseSkill={chooseSkill}
          skills={bootstrap.data?.skills ?? []}
          selectedSkillId={queuedSkillId ?? activeSession?.pending_skill_id}
          projects={visibleProjects}
          trashedProjects={trashedProjects.data ?? []}
          activeProject={activeProject}
          activeProjectId={activeProjectId}
          onProjectChange={(project) => {if (activeProjectIdRef.current === project.project_id) setActiveProject(project); void projects.refetch();}}
          onProjectNotice={(text) => setProjectNotice({text})}
          onSelectProject={(projectId) => {void selectProject(projectId);}}
          onCreateProject={(title) => {void createProject(title);}}
          onRenameProject={(projectId, title) => {void updateProject(projectId, {title}, `已重命名为“${title}”`);}}
          onPinProject={(projectId, pinned) => {void updateProject(projectId, {pinned}, pinned ? "项目已置顶" : "已取消置顶");}}
          onArchiveProject={(projectId, archived) => {void updateProject(projectId, {status: archived ? "archived" : "active"}, archived ? "项目已归档" : "项目已恢复使用");}}
          onTrashProject={(projectId) => {void trashProject(projectId);}}
          onRestoreProject={(project) => {void restoreTrashedProject(project);}}
          onExportProject={(project) => {void exportProject(project);}}
          onPurgeProject={(project) => {void purgeTrashedProject(project);}}
          onOpenProjectSyllabus={(syllabusId) => {void openProjectSyllabus(syllabusId);}}
          onOpenProjectTeachingSession={selectTeachingSession}
          onOpenProjectResource={openProjectResource}
          providerStatus={bootstrap.data?.provider_status}
          runtimePolicy={bootstrap.data?.agent_runtime_policy}
          mobileOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          backendConnected={Boolean(bootstrap.data && !bootstrap.isError)}
          controlsDisabled={projectMutationBusy || !bootstrap.data || bootstrap.isError}
        />
        <div className="relative flex min-w-0 flex-1">
          {(!bootstrap.data || bootstrap.isError) && (
            <ConnectionStatusBar
              state={bootstrap.isError ? "disconnected" : "connecting"}
              detail={bootstrap.error instanceof Error ? bootstrap.error.message : undefined}
              onReconnect={() => {void bootstrap.refetch();}}
            />
          )}
          {sidebarView === "teach" && teachSurface === "syllabi" ? <SyllabusWorkspace
            sidebarOpen={sidebarOpen}
            onToggleSidebar={() => setSidebarOpen((open) => !open)}
            inspectorOpen={drawerOpen}
            onToggleInspector={toggleInspector}
            initialSyllabus={openedSyllabus}
            initialError={syllabusLaunchError}
            sourceResources={syllabusSourceResources}
            teacherAuthority={bootstrap.data?.teacher_authority}
            onSyllabusChange={setOpenedSyllabus}
            onSyllabusSaved={(syllabus) => {void associateProjectReference(activeProjectIdRef.current, "syllabus", syllabus.syllabus_id);}}
            onBeforeRemoteGeneration={() => ensureRemoteConsent("remote_syllabus_generation")}
            onStartLesson={startSyllabusLesson}
          /> : <ConversationPane
            mode={sidebarView}
            messages={visibleMessages}
            onSend={send}
            onRetry={retryTurn}
            onQueueFollowUp={queueChatFollowUp}
            queuedChatPrompts={sidebarView === "chat" ? queuedChatPrompts : []}
            onCancelQueuedPrompt={cancelQueuedChatPrompt}
            onImportResources={sidebarView === "chat" ? importChatResources : importResources}
            resourceUploads={sidebarView === "chat" ? chatResourceUploads : resourceUploads}
            onRemoveResource={sidebarView === "chat" ? removeChatResourceUpload : removeResourceUpload}
            onStop={stop}
            onSelectSkill={(skillId) => {void chooseSkill(skillId);}}
            skills={bootstrap.data?.skills ?? []}
            selectedSkillId={sidebarView === "teach" ? queuedSkillId ?? activeSession?.pending_skill_id : null}
            sidebarOpen={sidebarOpen}
            onToggleSidebar={() => setSidebarOpen((open) => !open)}
            inspectorOpen={drawerOpen}
            onToggleInspector={toggleInspector}
            busy={runningModes[sidebarView]}
            disabled={!bootstrap.data || bootstrap.isError || projectMutationBusy}
            disabledReason={projectMutationBusy
              ? "正在切换学习项目，完成后即可发送。"
              : bootstrap.isError ? "后端连接已中断，正在重连。" : undefined}
            canAttach={sidebarView === "teach" && Boolean(activeSession)}
            supportedResourceExtensions={supportedResourceExtensions}
            webSearchEnabled={webSearchEnabled}
            webSearchAvailable={Boolean(bootstrap.data?.provider_status?.web_search_supported)}
            onWebSearchEnabledChange={changeWebSearchEnabled}
            title={headerTitle}
            subtitle={headerSubtitle}
            queuedSkillName={sidebarView === "teach" ? (bootstrap.data?.skills ?? []).find((skill) => skill.skill_id === queuedSkillId)?.name : undefined}
            sessionMeta={sidebarView === "teach" && activeSession ? {
              rounds: activeSession.rounds_completed ?? 0,
              contextVersion: activeSession.context_version,
              skill: actionSkill(currentAction(activeSession)),
              selectionReason: String(currentAction(activeSession)?.selection_reason ?? ""),
              fallbackCount: Number(activeSession.agent_runtime?.fallback_count ?? 0),
              lessonProgress: activeSession.lesson_progress ?? currentAction(activeSession)?.lesson_phase,
            } : undefined}
          />}
        </div>
      </div>
      <InspectorDrawer
        open={drawerOpen}
        onClose={closeInspector}
        session={sidebarView === "teach" ? activeSession : null}
        teachingResources={inspectorTeachingResources}
        providerStatus={bootstrap.data?.provider_status}
        remoteConsent={bootstrap.data?.remote_consent}
        visualSemantics={bootstrap.data?.visual_semantics}
        teacherAuthority={bootstrap.data?.teacher_authority}
        settingsRequest={settingsRequest}
        resourcesRequest={resourcesRequest}
        webSearchEnabled={webSearchEnabled}
        onWebSearchEnabledChange={changeWebSearchEnabled}
        visualAnalysisEnabled={visualAnalysisEnabled}
        onVisualAnalysisEnabledChange={setVisualAnalysisEnabled}
        onSessionChange={activateSessionFromInspector}
        onResourceReviewed={applyReviewedResource}
      />
      <CommandCenter
        open={commandCenterOpen}
        onOpenChange={setCommandCenterOpen}
        chatRunning={runningModes.chat}
        teachRunning={runningModes.teach}
        onAction={runCommandCenterAction}
      />
      {accountRightsOpen && (
        <div className="fixed inset-0 z-[90] grid place-items-center bg-black/55 p-4" role="presentation" onMouseDown={(event) => {
          if (event.target === event.currentTarget) setAccountRightsOpen(false);
        }}>
          <div role="dialog" aria-modal="true" aria-label="账户数据权利" className="max-h-[min(760px,calc(100dvh-2rem))] w-full max-w-2xl overflow-y-auto rounded-2xl border border-[var(--app-border-strong)] bg-[var(--app-surface-raised)] p-6 shadow-2xl">
            <div className="mb-5 flex items-center justify-between gap-4">
              <h2 className="text-base font-semibold">设置 · 数据权利</h2>
              <button type="button" className="rounded-md px-2 py-1 text-[var(--app-muted)] hover:bg-[var(--app-hover)]" onClick={() => setAccountRightsOpen(false)} aria-label="关闭账户数据权利">×</button>
            </div>
            <AccountDataRightsPanel
              authenticated={Boolean(bootstrap.data) && !bootstrap.isError}
              authenticatedAppsApi={bootstrap.data?.account_data_rights?.mode === "authenticated_account_authority"}
              status={accountDeletionState}
              exportAccount={downloadAccountExport}
              reauthenticate={reauthenticateAccountRights}
              prepareDeletion={prepareAccountDeletion}
              confirmDeletion={confirmAccountDeletion}
              refreshStatus={accountDeletionStatus}
              resumeDeletion={resumeAccountDeletion}
              statusChanged={setAccountDeletionState}
              committedDeletion={finishCommittedAccountDeletion}
            />
          </div>
        </div>
      )}
      {projectNotice && (
        <div role="status" aria-live="polite" className="fixed bottom-6 left-1/2 z-[70] flex max-w-[calc(100%-2rem)] -translate-x-1/2 items-center gap-3 rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-overlay)] px-3 py-2 text-xs text-[var(--app-text-soft)] shadow-2xl">
          <span className="truncate">{projectNotice.text}</span>
          {projectNotice.recoveryToken && <button type="button" disabled={projectMutationBusy} onClick={() => {void undoTrashProject();}} className="shrink-0 rounded-md px-2 py-1 text-[var(--app-accent-text)] hover:bg-[var(--app-hover)] disabled:opacity-40">撤销</button>}
          <button type="button" onClick={() => setProjectNotice(null)} aria-label="关闭通知" className="shrink-0 rounded-md px-1.5 py-1 text-[var(--app-faint)] hover:bg-[var(--app-hover)] hover:text-[var(--app-text)]">×</button>
        </div>
      )}
      <div className="pointer-events-none fixed bottom-2 left-1/2 z-20 -translate-x-1/2 text-[9px] text-[var(--app-faint)]">
        {bootstrap.isLoading ? "正在连接 TeachLab…" : bootstrap.isError ? "TeachLab 后端未连接" : [
          sidebarView === "chat"
            ? `${bootstrap.data?.provider_status?.provider ?? "DeepSeek"} · Chat`
            : `${bootstrap.data?.provider_status?.provider ?? "Teaching Agent"} · ${activeSession ? `R${activeSession.rounds_completed ?? 0}` : "新教学"}`,
          runningModes[sidebarView === "chat" ? "teach" : "chat"] ? `${sidebarView === "chat" ? "Teach" : "Chat"} 正在后台运行` : "",
        ].filter(Boolean).join(" · ")}
      </div>
    </div>
  );
}
