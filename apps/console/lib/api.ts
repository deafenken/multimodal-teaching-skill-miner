import type {
  AdjudicationClaimResponse,
  AdjudicationCandidate,
  AdjudicationDecisionResponse,
  AdjudicationReviewList,
  BootstrapPayload,
  ChatResponse,
  ChatTurn,
  CurriculumBlueprintResponse,
  LearningProject,
  LearningProjectChatMessage,
  LearningProjectChatThread,
  LearningProjectChatThreadSummary,
  LearningProjectNote,
  LearningProjectPage,
  LearningProjectReferenceItem,
  LearningProjectDeletionReceipt,
  LearningProjectSummary,
  LearningProjectTrashItem,
  LearningReviewClaimResponse,
  LearningReviewDueList,
  LearningReviewReleaseResponse,
  MetacognitionPairingReceipt,
  MetacognitionPredictionReceipt,
  MetacognitionSessionProjection,
  MetacognitionStrategyCode,
  RemoteConsentListResponse,
  RemoteConsentPurpose,
  RemoteConsentReceipt,
  SyllabusGeneratePayload,
  SyllabusEditableDraft,
  SyllabusLesson,
  SyllabusLessonStartPayload,
  SyllabusModule,
  TeacherAttachmentResponse,
  TeacherCommandResponse,
  TeachingResourceReviewRequest,
  TeachingResourceReviewResponse,
  TeacherResourceResponse,
  TeacherSessionResponse,
  TeachingSyllabus,
} from "@/lib/types";
import {normalizeSyllabusVersionFamily} from "@/lib/syllabus-version";
import type {ChatResourceRef} from "@/lib/chat-attachments";
import {
  parseAccountDeletionChallenge,
  parseAccountDeletionReceipt,
  parseAccountDeletionStatus,
  type AccountDeletionReceipt,
} from "@/lib/account-data-rights";
import {
  decodeHarnessEvent,
  harnessDelta,
  harnessEventIsUserVisible,
  harnessOperationResult,
  harnessStatusLabel,
  HarnessStreamProtocolError,
  initialHarnessStreamState,
  reduceHarnessEvent,
  SseFrameDecoder,
  validateHarnessChatMessage,
  validateHarnessSessionSnapshot,
  type HarnessEventEnvelope,
  type HarnessOperationResult,
  type HarnessStreamState,
} from "@/lib/harness-stream";
import {
  deleteDurableOperation,
  markOperationHandoff,
  markOperationRegistered,
  newDurableOperation,
  putDurableOperation,
  reconnectDelay,
  STREAM_MAX_CONNECTIONS,
  STREAM_RENDER_INTERVAL_MS,
  type DurableOperation,
} from "@/lib/offline-runtime";
import {
  newSafeguardingIdempotencyKey,
  parseSafeguardingStaffList,
  parseSafeguardingStaffMutation,
  safeguardingStaffRequestBody,
  type SafeguardingCaseStatus,
  type SafeguardingStaffMutationResponse,
} from "@/lib/safeguarding-staff";

const configuredBase = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "";
const pythonProxyBase = "/api/teacher-agent";
const harnessSecuritySessionPath = `${pythonProxyBase}/security/session`;
const harnessCsrfHeaderName = "x-teachlab-csrf-token";
const harnessSecurityRejectionHeader = "x-teachlab-security-rejection";
let harnessCsrfToken: string | null = null;
let harnessSecuritySessionRequest: Promise<string> | null = null;
let harnessConnectionMode: "local_python" | "authenticated_apps_api" | null = null;

export class HarnessAuthenticationRequiredError extends Error {
  constructor() {
    super("TeachLab 身份会话尚未建立或已经过期，请先登录后重新连接。");
    this.name = "HarnessAuthenticationRequiredError";
  }
}

export function isHarnessAuthenticationRequired(error: unknown) {
  return error instanceof HarnessAuthenticationRequiredError;
}

export function authenticatedHarnessLogoutAvailable() {
  return harnessConnectionMode === "authenticated_apps_api";
}

export function organizationLoginAvailable() {
  return process.env.NEXT_PUBLIC_TEACHLAB_AUTH_MODE === "oidc";
}

export function organizationLoginUrl(returnTo = "/") {
  if (!/^\/(?!\/|api\/)(?!.*[\\#\u0000-\u001f\u007f])/.test(returnTo)) return `${pythonProxyBase}/security/login`;
  return `${pythonProxyBase}/security/login?return_to=${encodeURIComponent(returnTo)}`;
}

export function apiUrl(path: string) {
  if (/^https?:\/\//.test(path)) return path;
  if (path === pythonProxyBase || path.startsWith(`${pythonProxyBase}/`)) return path;
  return `${configuredBase}${path.startsWith("/") ? path : `/${path}`}`;
}

async function requestHarnessSecuritySession() {
  const response = await fetch(apiUrl(harnessSecuritySessionPath), {
    method: "GET",
    headers: {Accept: "application/json"},
    credentials: "include",
    cache: "no-store",
  });
  if (!response.ok) {
    harnessConnectionMode = null;
    const authenticationRequired = response.status === 401
      || response.headers.get("x-teachlab-auth-state") === "authentication_required";
    if (authenticationRequired) throw new HarnessAuthenticationRequiredError();
    throw new Error(`TeachLab 安全握手失败 (${response.status})`);
  }
  const payload = await response.json() as {connection?: unknown; csrf_token?: unknown};
  if (
    payload.connection !== "local_python"
    && payload.connection !== "authenticated_apps_api"
  ) {
    throw new Error("TeachLab 安全握手返回了未知连接模式");
  }
  if (typeof payload.csrf_token !== "string" || !/^[A-Za-z0-9_-]{43}$/.test(payload.csrf_token)) {
    throw new Error("TeachLab 安全握手返回了无效令牌");
  }
  harnessConnectionMode = payload.connection;
  harnessCsrfToken = payload.csrf_token;
  return payload.csrf_token;
}

function ensureHarnessSecuritySession() {
  if (harnessCsrfToken) return Promise.resolve(harnessCsrfToken);
  harnessSecuritySessionRequest ??= requestHarnessSecuritySession().finally(() => {
    harnessSecuritySessionRequest = null;
  });
  return harnessSecuritySessionRequest;
}

async function fetchTeacherAgent(path: string, init: RequestInit = {}) {
  const method = (init.method ?? "GET").toUpperCase();
  const unsafeMethod = method !== "GET" && method !== "HEAD";
  let token = await ensureHarnessSecuritySession();
  const perform = () => {
    const headers = new Headers(init.headers);
    if (unsafeMethod) headers.set(harnessCsrfHeaderName, token);
    return fetch(apiUrl(path), {...init, method, headers, credentials: "include", cache: "no-store"});
  };

  let response = await perform();
  const rejection = response.headers.get(harnessSecurityRejectionHeader);
  if (rejection === "authentication_required" && harnessCsrfToken === token) {
    harnessCsrfToken = null;
    harnessConnectionMode = null;
  }
  if ((rejection === "invalid_session" || rejection === "invalid_csrf") && !init.signal?.aborted) {
    // Only invalidate the token used by this request. Another concurrent
    // request may already have completed the shared refresh and installed a
    // newer cookie/token pair.
    if (harnessCsrfToken === token) harnessCsrfToken = null;
    await response.body?.cancel().catch(() => undefined);
    token = await ensureHarnessSecuritySession();
    response = await perform();
  }
  return response;
}

export async function clearAuthenticatedHarnessSession() {
  const token = await ensureHarnessSecuritySession();
  const response = await fetch(apiUrl(harnessSecuritySessionPath), {
    method: "DELETE",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      [harnessCsrfHeaderName]: token,
    },
    body: "{}",
    credentials: "include",
    cache: "no-store",
  });
  if (response.status !== 204) {
    throw new Error(`TeachLab 退出失败 (${response.status})`);
  }
  await response.body?.cancel().catch(() => undefined);
  harnessCsrfToken = null;
  harnessSecuritySessionRequest = null;
  harnessConnectionMode = null;
}

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const requestInit: RequestInit = {
    ...init,
    headers: {"Content-Type": "application/json", ...(init?.headers ?? {})},
    credentials: "include",
    cache: "no-store"
  };
  const response = path === pythonProxyBase || path.startsWith(`${pythonProxyBase}/`)
    ? await fetchTeacherAgent(path, requestInit)
    : await fetch(apiUrl(path), requestInit);
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json() as {error?: string};
      detail = typeof payload.error === "string" ? `: ${payload.error}` : "";
    } catch {
      // Keep the status-only error when the upstream did not return JSON.
    }
    throw new Error(`API request failed (${response.status})${detail}`);
  }
  return response.json() as Promise<T>;
}

export function fetchBootstrap() {
  return fetchJson<BootstrapPayload>(`${pythonProxyBase}/api/bootstrap`, {method: "GET"});
}

const safeguardingStaffPath = `${pythonProxyBase}/api/safeguarding`;

export async function listSafeguardingCases(
  routeLocator: string,
  status?: SafeguardingCaseStatus,
) {
  const payload = await fetchJson<unknown>(`${safeguardingStaffPath}/list`, {
    method: "POST",
    body: JSON.stringify(safeguardingStaffRequestBody(routeLocator, {
      safeguarding_idempotency_key: newSafeguardingIdempotencyKey("list"),
      ...(status ? {status} : {}),
    })),
  });
  return parseSafeguardingStaffList(payload);
}

async function mutateSafeguardingCase(
  routeLocator: string,
  route: "case/acknowledge" | "case/close" | "escalation/acknowledge",
  operation: "case-acknowledge" | "case-close" | "delivery-acknowledge",
  caseId: string,
  expectedVersion: number,
): Promise<SafeguardingStaffMutationResponse> {
  const payload = await fetchJson<unknown>(`${safeguardingStaffPath}/${route}`, {
    method: "POST",
    body: JSON.stringify(safeguardingStaffRequestBody(routeLocator, {
      case_id: caseId,
      expected_version: expectedVersion,
      safeguarding_idempotency_key: newSafeguardingIdempotencyKey(operation),
    })),
  });
  return parseSafeguardingStaffMutation(payload);
}

export function acknowledgeSafeguardingCase(
  routeLocator: string,
  caseId: string,
  expectedVersion: number,
) {
  return mutateSafeguardingCase(
    routeLocator,
    "case/acknowledge",
    "case-acknowledge",
    caseId,
    expectedVersion,
  );
}

export function closeSafeguardingCase(
  routeLocator: string,
  caseId: string,
  expectedVersion: number,
) {
  return mutateSafeguardingCase(
    routeLocator,
    "case/close",
    "case-close",
    caseId,
    expectedVersion,
  );
}

export function acknowledgeSafeguardingDelivery(
  routeLocator: string,
  caseId: string,
  expectedVersion: number,
) {
  return mutateSafeguardingCase(
    routeLocator,
    "escalation/acknowledge",
    "delivery-acknowledge",
    caseId,
    expectedVersion,
  );
}

const accountDataRightsPath = `${pythonProxyBase}/account`;

export function accountDataRightsStepUpUrl(returnTo = "/") {
  return `${pythonProxyBase}/security/account-step-up?return_to=${encodeURIComponent(returnTo)}`;
}

export async function downloadAccountExport(): Promise<void> {
  // Let the browser's native download stack consume the same-origin BFF
  // stream. A fetch()+blob() implementation would retain a permitted 400 MiB
  // archive in renderer memory and defeat the server's streaming boundary.
  const link = document.createElement("a");
  link.href = `${accountDataRightsPath}/export`;
  link.rel = "noopener";
  link.style.display = "none";
  document.body.appendChild(link);
  try {
    link.click();
  } finally {
    link.remove();
  }
}

export async function prepareAccountDeletion() {
  return parseAccountDeletionChallenge(await fetchJson<unknown>(
    `${accountDataRightsPath}/deletion/prepare`,
    {method: "POST", body: "{}"},
  ));
}

export async function confirmAccountDeletion(input: {
  challenge_id: string;
  confirmation_token: string;
  confirmation_phrase: string;
  expected_revision: number;
  idempotency_key: string;
}): Promise<AccountDeletionReceipt> {
  return parseAccountDeletionReceipt(await fetchJson<unknown>(
    `${accountDataRightsPath}/deletion/confirm`,
    {method: "POST", body: JSON.stringify(input)},
  ));
}

export async function accountDeletionStatus() {
  return parseAccountDeletionStatus(await fetchJson<unknown>(
    `${accountDataRightsPath}/deletion/status`,
    {method: "GET"},
  ));
}

export async function resumeAccountDeletion() {
  return parseAccountDeletionStatus(await fetchJson<unknown>(
    `${accountDataRightsPath}/deletion/resume`,
    {method: "POST", body: "{}"},
  ));
}

export function stopConsoleRuntime() {
  return fetchJson<{status: "stopping"}>(`${pythonProxyBase}/runtime/stop`, {
    method: "POST",
    body: "{}",
  });
}

const consentPath = `${pythonProxyBase}/api/consent`;

export function listRemoteConsents() {
  return fetchJson<RemoteConsentListResponse>(`${consentPath}/list`, {
    method: "POST",
    body: "{}",
  });
}

export async function grantRemoteConsent(payload: {
  purpose: RemoteConsentPurpose;
  validity_days?: number;
}) {
  const response = await fetchJson<{receipt: RemoteConsentReceipt; server_minted: true}>(`${consentPath}/grant`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return response.receipt;
}

export async function revokeRemoteConsent(consentId: string, reasonCode = "user_revoked") {
  const response = await fetchJson<{receipt: RemoteConsentReceipt}>(`${consentPath}/revoke`, {
    method: "POST",
    body: JSON.stringify({consent_id: consentId, reason_code: reasonCode}),
  });
  return response.receipt;
}

const projectsPath = `${pythonProxyBase}/api/projects`;

export async function listLearningProjects() {
  const payload = await fetchJson<{projects?: LearningProjectSummary[]}>(projectsPath, {method: "GET"});
  return Array.isArray(payload.projects) ? payload.projects : [];
}

export async function listTrashedLearningProjects() {
  const payload = await fetchJson<{projects?: LearningProjectTrashItem[]}>(`${projectsPath}/trash`, {method: "GET"});
  return Array.isArray(payload.projects) ? payload.projects : [];
}

export async function createLearningProject(payload: {title: string; description?: string; operation_id?: string}) {
  const response = await fetchJson<{project?: LearningProject} | LearningProject>(projectsPath, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return "project" in response && response.project ? response.project : response as LearningProject;
}

export async function fetchLearningProject(projectId: string) {
  const response = await fetchJson<{project?: LearningProject} | LearningProject>(`${projectsPath}/${encodeURIComponent(projectId)}`, {method: "GET"});
  return "project" in response && response.project ? response.project : response as LearningProject;
}

export async function updateLearningProject(projectId: string, payload: {title?: string; description?: string; status?: "active" | "archived"; pinned?: boolean; expected_updated_at?: string; operation_id?: string}) {
  const response = await fetchJson<{project?: LearningProject} | LearningProject>(`${projectsPath}/${encodeURIComponent(projectId)}/update`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return "project" in response && response.project ? response.project : response as LearningProject;
}

export async function saveProjectChatThread(projectId: string, chatThread: LearningProjectChatThread) {
  const response = await fetchJson<{project?: LearningProject} | LearningProject>(`${projectsPath}/${encodeURIComponent(projectId)}/chat-thread`, {
    method: "POST",
    body: JSON.stringify({chat_thread: chatThread}),
  });
  return "project" in response && response.project ? response.project : response as LearningProject;
}

export async function addLearningProjectReference(
  projectId: string,
  kind: "syllabus" | "teaching_session" | "resource",
  referenceId: string,
  options?: {expectedUpdatedAt?: string; operationId?: string},
) {
  const response = await fetchJson<{project?: LearningProject} | LearningProject>(`${projectsPath}/${encodeURIComponent(projectId)}/reference`, {
    method: "POST",
    body: JSON.stringify({
      kind,
      reference_id: referenceId,
      ...(options?.expectedUpdatedAt ? {expected_updated_at: options.expectedUpdatedAt} : {}),
      ...(options?.operationId ? {operation_id: options.operationId} : {}),
    }),
  });
  return "project" in response && response.project ? response.project : response as LearningProject;
}

export async function bootstrapLearningProject(payload: {
  idempotency_key: string;
  title?: string;
  description?: string;
  project_id?: string;
  migration_id?: string;
  legacy_chat_threads?: LearningProjectChatThread[];
  teaching_session_ids?: string[];
}) {
  return fetchJson<{
    project: LearningProject;
    created_or_replayed: true;
    legacy_migration_applied_or_replayed: boolean;
    stale_teaching_session_ids: string[];
  }>(`${projectsPath}/bootstrap`, {method: "POST", body: JSON.stringify(payload)});
}

type ProjectBrowseSection = "chat_threads" | "chat_messages" | "notes" | "syllabi" | "teaching_sessions" | "resources";
type ProjectBrowseItem<S extends ProjectBrowseSection> = S extends "chat_threads"
  ? LearningProjectChatThreadSummary
  : S extends "chat_messages"
    ? LearningProjectChatMessage
    : S extends "notes"
      ? LearningProjectNote
      : LearningProjectReferenceItem;

export async function browseLearningProject<S extends ProjectBrowseSection>(
  projectId: string,
  payload: {section: S; query?: string; cursor?: string; limit?: number; thread_id?: string},
) {
  return fetchJson<LearningProjectPage<ProjectBrowseItem<S>>>(`${projectsPath}/${encodeURIComponent(projectId)}/browse`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function saveLearningProjectNote(projectId: string, payload: {
  operation_id: string;
  expected_updated_at: string;
  note: {note_id?: string; title: string; body: string};
}) {
  const response = await fetchJson<{project?: LearningProject} | LearningProject>(`${projectsPath}/${encodeURIComponent(projectId)}/note`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return "project" in response && response.project ? response.project : response as LearningProject;
}

export async function removeLearningProjectReference(projectId: string, payload: {
  kind: "syllabus" | "teaching_session" | "resource";
  reference_id: string;
  expected_updated_at: string;
  operation_id: string;
}) {
  const response = await fetchJson<{project?: LearningProject} | LearningProject>(`${projectsPath}/${encodeURIComponent(projectId)}/remove-reference`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return "project" in response && response.project ? response.project : response as LearningProject;
}

export async function trashLearningProject(projectId: string) {
  return fetchJson<{project_id: string; recovery_token: string; trashed_at: string; browser_session_handles_to_forget: string[]}>(`${projectsPath}/${encodeURIComponent(projectId)}/trash`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export async function restoreLearningProject(projectId: string, recoveryToken: string) {
  const response = await fetchJson<{project?: LearningProject} | LearningProject>(`${projectsPath}/${encodeURIComponent(projectId)}/restore`, {
    method: "POST",
    body: JSON.stringify({recovery_token: recoveryToken}),
  });
  return "project" in response && response.project ? response.project : response as LearningProject;
}

export async function exportLearningProject(projectId: string) {
  const response = await fetchTeacherAgent(
    `${projectsPath}/${encodeURIComponent(projectId)}/export`,
    {method: "GET", headers: {Accept: "application/zip"}},
  );
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json() as {error?: unknown};
      if (typeof payload.error === "string") detail = `: ${payload.error}`;
    } catch {
      // Retain the status-only error for a non-JSON upstream response.
    }
    throw new Error(`Project export failed (${response.status})${detail}`);
  }
  const manifestSha256 = response.headers.get("x-manifest-sha256") ?? "";
  if (!/^[0-9a-f]{64}$/.test(manifestSha256)) {
    throw new Error("Project export returned an invalid manifest receipt");
  }
  const disposition = response.headers.get("content-disposition") ?? "";
  const match = /filename="([A-Za-z0-9_.-]+)"/.exec(disposition);
  return {
    blob: await response.blob(),
    filename: match?.[1] ?? `${projectId}-private-export.zip`,
    manifestSha256,
  };
}

export async function purgeLearningProject(
  projectId: string,
  recoveryToken: string,
  confirmation: string,
) {
  const response = await fetchJson<{deletion_receipt?: LearningProjectDeletionReceipt}>(
    `${projectsPath}/${encodeURIComponent(projectId)}/purge`,
    {
      method: "POST",
      body: JSON.stringify({recovery_token: recoveryToken, confirmation}),
    },
  );
  if (!response.deletion_receipt) throw new Error("Permanent deletion returned no receipt");
  return response.deletion_receipt;
}

export function startTeachingSession(payload: Record<string, unknown>) {
  return fetchJson<TeacherSessionResponse>(`${pythonProxyBase}/api/start`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function resumeTeachingSession(sessionId: string, signal?: AbortSignal) {
  return fetchJson<TeacherSessionResponse>(`${pythonProxyBase}/api/session`, {
    method: "POST",
    body: JSON.stringify({session_id: sessionId}),
    signal,
  });
}

export function sendTeachingTurn(payload: Record<string, unknown>) {
  return fetchJson<TeacherSessionResponse>(`${pythonProxyBase}/api/step`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function sendTeachingCommand(payload: Record<string, unknown>) {
  return fetchJson<TeacherCommandResponse>(`${pythonProxyBase}/api/command`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

function learningSessionGuards(session: TeacherSessionResponse) {
  const expectedRound = session.rounds_completed ?? session.round ?? 0;
  const profileRevision = session.profile_summary?.profile_revision;
  if (!session.expected_question_id || !profileRevision) {
    throw new Error("当前会话缺少长期学习所需的版本守卫");
  }
  return {
    session_id: session.session_id,
    expected_round: expectedRound,
    expected_question_id: session.expected_question_id,
    expected_context_version: session.context_version,
    profile_revision: profileRevision,
  };
}

const learningReviewPath = `${pythonProxyBase}/api/learning-reviews`;
const metacognitionPath = `${pythonProxyBase}/api/metacognition`;

export function listDueLearningReviews(session: TeacherSessionResponse) {
  return fetchJson<LearningReviewDueList>(`${learningReviewPath}/due`, {
    method: "POST",
    body: JSON.stringify(learningSessionGuards(session)),
  });
}

export function claimDueLearningReview(
  session: TeacherSessionResponse,
  reviewId: string,
  expectedVersion: number,
) {
  return fetchJson<LearningReviewClaimResponse>(`${learningReviewPath}/claim`, {
    method: "POST",
    body: JSON.stringify({
      ...learningSessionGuards(session),
      review_id: reviewId,
      expected_version: expectedVersion,
      // A lost HTTP response must be recoverable by pressing the same action
      // again.  The durable CAS version makes this identity unique to one
      // claim attempt while keeping browser retries stable.
      review_idempotency_key: `console-learning-review-claim-${reviewId}-${expectedVersion}`,
    }),
  });
}

export function releaseLearningReview(
  session: TeacherSessionResponse,
  payload: {reviewId: string; leaseId: string; expectedVersion: number},
) {
  return fetchJson<LearningReviewReleaseResponse>(`${learningReviewPath}/release`, {
    method: "POST",
    body: JSON.stringify({
      ...learningSessionGuards(session),
      review_id: payload.reviewId,
      lease_id: payload.leaseId,
      expected_version: payload.expectedVersion,
      // Bind retries to the server lease rather than minting a new mutation
      // identity after an ambiguous transport failure.
      review_idempotency_key: `console-learning-review-release-${payload.leaseId}-${payload.expectedVersion}`,
    }),
  });
}

export function listMetacognitionPredictions(session: TeacherSessionResponse) {
  return fetchJson<MetacognitionSessionProjection>(`${metacognitionPath}/list`, {
    method: "POST",
    body: JSON.stringify(learningSessionGuards(session)),
  });
}

export function recordMetacognitionPrediction(
  session: TeacherSessionResponse,
  payload: {
    learnerJolPercent: number;
    strategyCodes: MetacognitionStrategyCode[];
  },
) {
  return fetchJson<MetacognitionPredictionReceipt>(`${metacognitionPath}/predict`, {
    method: "POST",
    body: JSON.stringify({
      ...learningSessionGuards(session),
      learner_jol_percent: payload.learnerJolPercent,
      strategy_codes: payload.strategyCodes,
    }),
  });
}

export function pairMetacognitionPrediction(
  session: TeacherSessionResponse,
  predictionEventId: string,
) {
  return fetchJson<MetacognitionPairingReceipt>(`${metacognitionPath}/pair`, {
    method: "POST",
    body: JSON.stringify({
      ...learningSessionGuards(session),
      prediction_event_id: predictionEventId,
    }),
  });
}

function adjudicationSessionGuards(session: TeacherSessionResponse) {
  const expectedRound = session.rounds_completed ?? session.round ?? 0;
  const profileRevision = session.profile_summary?.profile_revision;
  if (!session.expected_question_id || !profileRevision) {
    throw new Error("当前会话缺少复核所需的版本守卫");
  }
  return {
    session_id: session.session_id,
    expected_round: expectedRound,
    expected_question_id: session.expected_question_id,
    expected_context_version: session.context_version,
    profile_revision: profileRevision,
  };
}

const adjudicationPath = `${pythonProxyBase}/api/adjudication`;

export function listAdjudicationReviews(session: TeacherSessionResponse) {
  return fetchJson<AdjudicationReviewList>(`${adjudicationPath}/list`, {
    method: "POST",
    body: JSON.stringify(adjudicationSessionGuards(session)),
  });
}

export function listAdjudicationCandidates(session: TeacherSessionResponse) {
  return fetchJson<{
    session_id: string;
    candidates: AdjudicationCandidate[];
    public_candidates_contain_learner_text: false;
  }>(`${adjudicationPath}/candidates`, {
    method: "POST",
    body: JSON.stringify(adjudicationSessionGuards(session)),
  });
}

export function enqueueAdjudicationReview(
  session: TeacherSessionResponse,
  payload: {historyRound: number; knowledgeComponentId: string; reviewReason: string},
) {
  return fetchJson<{item: import("@/lib/types").AdjudicationReviewItem}>(`${adjudicationPath}/enqueue`, {
    method: "POST",
    body: JSON.stringify({
      ...adjudicationSessionGuards(session),
      history_round: payload.historyRound,
      knowledge_component_id: payload.knowledgeComponentId,
      review_reason: payload.reviewReason,
      adjudication_idempotency_key: `console-adjudication-enqueue-${crypto.randomUUID()}`,
    }),
  });
}

export function claimAdjudicationReview(
  session: TeacherSessionResponse,
  itemId: string,
  expectedVersion: number,
) {
  return fetchJson<AdjudicationClaimResponse>(`${adjudicationPath}/claim`, {
    method: "POST",
    body: JSON.stringify({
      ...adjudicationSessionGuards(session),
      item_id: itemId,
      expected_version: expectedVersion,
      // Stable across an ambiguous network retry. apps/api mints a fresh,
      // one-time authority nonce while the durable Python store returns the
      // already sealed business result for this exact key and payload.
      adjudication_idempotency_key: `console-adjudication-claim-${itemId}-${expectedVersion}`,
    }),
  });
}

export function decideAdjudicationReview(
  session: TeacherSessionResponse,
  payload: {
    itemId: string;
    expectedVersion: number;
    claimToken: string;
    decision: "approve" | "correct" | "abstain";
    reasonCode: string;
    correction?: Record<string, unknown>;
  },
) {
  return fetchJson<AdjudicationDecisionResponse>(`${adjudicationPath}/decide`, {
    method: "POST",
    body: JSON.stringify({
      ...adjudicationSessionGuards(session),
      item_id: payload.itemId,
      expected_version: payload.expectedVersion,
      claim_token: payload.claimToken,
      decision: payload.decision,
      reason_code: payload.reasonCode,
      ...(payload.correction ? {correction: payload.correction} : {}),
      adjudication_idempotency_key: [
        "console-adjudication-decide",
        payload.itemId,
        payload.expectedVersion,
        payload.decision,
        payload.decision === "correct"
          ? String(payload.correction?.signal ?? "missing")
          : payload.reasonCode,
      ].join("-"),
    }),
  });
}

export function uploadTeachingAttachment(payload: Record<string, unknown>) {
  return fetchJson<TeacherAttachmentResponse>(`${pythonProxyBase}/api/attachment`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function uploadTeachingResource(payload: Record<string, unknown>) {
  return fetchJson<TeacherResourceResponse>(`${pythonProxyBase}/api/resource`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function reviewTeachingResource(payload: TeachingResourceReviewRequest) {
  return fetchJson<TeachingResourceReviewResponse>(`${pythonProxyBase}/api/resource/review`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

const syllabiPath = `${pythonProxyBase}/api/syllabi`;

function recordValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function stringValue(value: unknown, fallback = "") {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function stringList(value: unknown) {
  return Array.isArray(value) ? value.flatMap((item) => typeof item === "string" && item.trim() ? [item.trim()] : []) : undefined;
}

function validStringList(value: unknown, options?: {allowEmpty?: boolean}) {
  return Array.isArray(value)
    && (options?.allowEmpty || value.length > 0)
    && value.every((item) => typeof item === "string" && Boolean(item.trim()));
}

async function fetchSyllabusJson(path: string, init?: RequestInit) {
  try {
    return await fetchJson<unknown>(path, init);
  } catch (caught) {
    const message = caught instanceof Error ? caught.message : "";
    if (/\((?:502|503|504)\)|failed to fetch|networkerror|backend (?:is )?(?:unavailable|not configured)|capability_url is not configured/i.test(message)) {
      throw new Error("教学后端未连接，请重新启动 Console 后重试。", {cause: caught});
    }
    throw caught;
  }
}

function normalizeSyllabusLesson(value: unknown, index: number): SyllabusLesson {
  const raw = recordValue(value);
  const teachingGoal = recordValue(raw.teaching_goal ?? raw.goal);
  const rawMaterials = recordValue(raw.materials);
  const materials = Object.fromEntries(Object.entries(rawMaterials).flatMap(([key, item]) => typeof item === "string" ? [[key, item]] : []));
  return {
    ...raw,
    lesson_id: stringValue(raw.lesson_id ?? raw.id, `lesson-${index + 1}`),
    title: stringValue(raw.title ?? raw.name, `课节 ${index + 1}`),
    objective: stringValue(raw.objective ?? raw.learning_objective) || undefined,
    summary: stringValue(raw.summary ?? raw.description) || undefined,
    duration_minutes: Number.isFinite(Number(raw.duration_minutes ?? raw.duration)) ? Number(raw.duration_minutes ?? raw.duration) : undefined,
    knowledge_components: stringList(raw.knowledge_components ?? raw.key_concepts),
    prerequisites: stringList(raw.prerequisites),
    materials: Object.keys(materials).length ? materials : undefined,
    teaching_goal: Object.keys(teachingGoal).length ? teachingGoal : undefined,
    start_payload: Object.keys(recordValue(raw.start_payload)).length ? recordValue(raw.start_payload) as SyllabusLesson["start_payload"] : undefined,
    order: Number.isFinite(Number(raw.order)) ? Number(raw.order) : index + 1,
  };
}

function normalizeSyllabusModule(value: unknown, index: number): SyllabusModule {
  const raw = recordValue(value);
  const lessons = Array.isArray(raw.lessons) ? raw.lessons.map(normalizeSyllabusLesson) : [];
  return {
    ...raw,
    module_id: stringValue(raw.module_id ?? raw.id, `module-${index + 1}`),
    title: stringValue(raw.title ?? raw.name, `模块 ${index + 1}`),
    description: stringValue(raw.description ?? raw.summary) || undefined,
    lessons,
    order: Number.isFinite(Number(raw.order)) ? Number(raw.order) : index + 1,
  };
}

function requireTeachingSyllabusDocument(value: unknown) {
  const envelope = recordValue(value);
  const raw = recordValue(envelope.syllabus ?? envelope.data ?? value);
  if (
    raw.schema !== "teaching_syllabus.v1"
    || !/^syl_[0-9a-f]{24}$/.test(stringValue(raw.syllabus_id))
    || !stringValue(raw.title)
    || !stringValue(raw.description)
    || !stringValue(raw.audience)
    || typeof raw.estimated_duration_minutes !== "number"
    || !Number.isInteger(raw.estimated_duration_minutes)
    || raw.estimated_duration_minutes < 15
    || !validStringList(raw.learning_objectives)
    || !validStringList(raw.prerequisites, {allowEmpty: true})
    || !Array.isArray(raw.modules)
    || raw.modules.length === 0
    || raw.modules.some((moduleValue) => {
      const module = recordValue(moduleValue);
      return !/^module_[0-9]{2}$/.test(stringValue(module.module_id))
        || !stringValue(module.title)
        || !stringValue(module.description)
        || !Array.isArray(module.lessons)
        || module.lessons.length === 0
        || module.lessons.some((lessonValue) => {
          const lesson = recordValue(lessonValue);
          const materials = recordValue(lesson.materials);
          return !/^lesson_[0-9]{2}_[0-9]{2}$/.test(stringValue(lesson.lesson_id))
            || !stringValue(lesson.title)
            || !stringValue(lesson.objective)
            || !stringValue(lesson.summary)
            || typeof lesson.duration_minutes !== "number"
            || !Number.isInteger(lesson.duration_minutes)
            || lesson.duration_minutes < 5
            || !validStringList(lesson.knowledge_components)
            || !stringValue(materials.example)
            || !stringValue(materials.practice)
            || !stringValue(materials.transfer_task)
            || !Object.keys(recordValue(lesson.teaching_goal)).length;
        });
    })
  ) {
    throw new Error("教学后端返回了不完整的大纲，请重试。若问题持续，请重新启动 Console。");
  }
  return raw;
}

export function normalizeTeachingSyllabus(value: unknown, options?: {failClosed?: boolean}): TeachingSyllabus {
  const envelope = recordValue(value);
  const raw = options?.failClosed
    ? requireTeachingSyllabusDocument(value)
    : recordValue(envelope.syllabus ?? envelope.data ?? value);
  const modules = Array.isArray(raw.modules) ? raw.modules.map(normalizeSyllabusModule) : [];
  return {
    ...raw,
    syllabus_id: stringValue(raw.syllabus_id ?? raw.id),
    title: stringValue(raw.title ?? raw.topic, "未命名教学大纲"),
    description: stringValue(raw.description ?? raw.summary) || undefined,
    audience: stringValue(raw.audience) || undefined,
    estimated_duration_minutes: typeof raw.estimated_duration_minutes === "number" && Number.isInteger(raw.estimated_duration_minutes) ? raw.estimated_duration_minutes : undefined,
    learning_objectives: stringList(raw.learning_objectives),
    prerequisites: stringList(raw.prerequisites),
    status: stringValue(raw.status) || undefined,
    source: typeof raw.source === "string" ? stringValue(raw.source) || undefined : Object.keys(recordValue(raw.source)).length ? recordValue(raw.source) : undefined,
    created_at: stringValue(raw.created_at) || undefined,
    updated_at: stringValue(raw.updated_at) || undefined,
    modules,
  };
}

export function isTeachingSyllabusDocument(value: unknown) {
  const envelope = recordValue(value);
  const raw = recordValue(envelope.syllabus ?? envelope.data ?? value);
  return [envelope.schema, envelope.schema_version, envelope.schema_id, raw.schema, raw.schema_version, raw.schema_id].some((item) => item === "teaching_syllabus.v1");
}

export async function listTeachingSyllabi() {
  const payload = await fetchSyllabusJson(syllabiPath, {method: "GET"});
  const envelope = recordValue(payload);
  const items = Array.isArray(payload) ? payload : Array.isArray(envelope.syllabi) ? envelope.syllabi : Array.isArray(envelope.items) ? envelope.items : [];
  return items.map((item) => normalizeTeachingSyllabus(item)).filter((item) => item.syllabus_id);
}

export async function fetchTeachingSyllabusVersions(syllabusId: string) {
  const payload = await fetchSyllabusJson(`${syllabiPath}/${encodeURIComponent(syllabusId)}/versions`, {method: "GET"});
  const envelope = recordValue(payload);
  const documents = Array.isArray(envelope.syllabi) ? envelope.syllabi.map((item) => normalizeTeachingSyllabus(item, {failClosed: true})) : [];
  const versionFamily = normalizeSyllabusVersionFamily(envelope);
  if (documents.length !== versionFamily.revisions.length || documents.some((item, index) => item.syllabus_id !== versionFamily.revisions[index]?.syllabus_id)) {
    throw new Error("大纲版本正文与版本账本不一致，请停止编辑并检查本地存储。");
  }
  return {versionFamily, syllabi: documents};
}

export async function fetchCurriculumBlueprint(
  syllabusId: string
): Promise<CurriculumBlueprintResponse> {
  const payload = recordValue(await fetchJson<unknown>(
    `${syllabiPath}/${encodeURIComponent(syllabusId)}/curriculum-blueprint`,
    {method: "GET"}
  ));
  const blueprint = recordValue(payload.curriculum_blueprint);
  if (!Object.keys(blueprint).length || typeof payload.review_status !== "string") {
    throw new Error("教学后端返回了不完整的课程测量蓝图。");
  }
  return payload as unknown as CurriculumBlueprintResponse;
}

export async function reviewCurriculum(payload: {
  syllabus_id: string;
  teacher_spec: Record<string, unknown>;
  expected_syllabus_version: number;
  expected_authority_version: number;
  curriculum_authority_idempotency_key: string;
}) {
  return fetchJson<Record<string, unknown>>("api/curriculum/review", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function sealCurriculum(payload: {
  syllabus_id: string;
  review_id: string;
  teacher_confirmed_authority: true;
  expected_syllabus_version: number;
  expected_authority_version: number;
  curriculum_authority_idempotency_key: string;
}) {
  return fetchJson<Record<string, unknown>>("api/curriculum/seal", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function revokeCurriculum(payload: {
  syllabus_id: string;
  curriculum_id: string;
  reason_code: string;
  expected_syllabus_version: number;
  expected_authority_version: number;
  curriculum_authority_idempotency_key: string;
}) {
  return fetchJson<Record<string, unknown>>("api/curriculum/revoke", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

async function mutateTeachingSyllabusVersion(syllabusId: string, action: "revisions" | "publish" | "rollback", body: Record<string, unknown>) {
  const payload = await fetchSyllabusJson(`${syllabiPath}/${encodeURIComponent(syllabusId)}/${action}`, {
    method: "POST",
    body: JSON.stringify(body),
  });
  return {
    syllabus: normalizeTeachingSyllabus(payload, {failClosed: true}),
    versionFamily: normalizeSyllabusVersionFamily(payload),
  };
}

export async function reviseTeachingSyllabus(syllabusId: string, payload: {
  editable_draft: SyllabusEditableDraft;
  change_summary: string;
  expected_version: number;
  idempotency_key: string;
}) {
  return mutateTeachingSyllabusVersion(syllabusId, "revisions", payload);
}

export async function publishTeachingSyllabusRevision(syllabusId: string, payload: {
  revision_id: string;
  expected_version: number;
  idempotency_key: string;
}) {
  return mutateTeachingSyllabusVersion(syllabusId, "publish", payload);
}

export async function rollbackTeachingSyllabusRevision(syllabusId: string, payload: {
  revision_id: string;
  expected_version: number;
  idempotency_key: string;
}) {
  return mutateTeachingSyllabusVersion(syllabusId, "rollback", payload);
}

export async function fetchTeachingSyllabus(syllabusId: string) {
  return normalizeTeachingSyllabus(await fetchSyllabusJson(`${syllabiPath}/${encodeURIComponent(syllabusId)}`, {method: "GET"}));
}

export async function createTeachingSyllabus(payload: TeachingSyllabus) {
  return normalizeTeachingSyllabus(await fetchSyllabusJson(syllabiPath, {
    method: "POST",
    body: JSON.stringify({syllabus: payload}),
  }));
}

export async function generateTeachingSyllabus(payload: SyllabusGeneratePayload) {
  return normalizeTeachingSyllabus(await fetchSyllabusJson(`${syllabiPath}/generate`, {
    method: "POST",
    body: JSON.stringify(payload),
  }), {failClosed: true});
}

export async function importTeachingSyllabus(syllabus: unknown) {
  const envelope = recordValue(syllabus);
  const document = recordValue(envelope.syllabus ?? envelope.data ?? syllabus);
  return normalizeTeachingSyllabus(await fetchSyllabusJson(`${syllabiPath}/import`, {
    method: "POST",
    body: JSON.stringify({syllabus: document}),
  }));
}

export async function downloadTeachingSyllabus(syllabusId: string) {
  const response = await fetch(apiUrl(`${syllabiPath}/${encodeURIComponent(syllabusId)}/download`), {
    method: "GET",
    credentials: "include",
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`教学大纲下载失败 (${response.status})`);
  const disposition = response.headers.get("content-disposition") ?? "";
  const encodedName = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  const plainName = disposition.match(/filename="?([^";]+)"?/i)?.[1];
  return {
    blob: await response.blob(),
    fileName: encodedName ? decodeURIComponent(encodedName) : plainName || `${syllabusId}.json`,
  };
}

export async function fetchSyllabusLessonStartPayload(syllabusId: string, lessonId: string): Promise<SyllabusLessonStartPayload> {
  const payload = await fetchJson<unknown>(`${syllabiPath}/${encodeURIComponent(syllabusId)}/lessons/${encodeURIComponent(lessonId)}/start-payload`, {method: "GET"});
  const envelope = recordValue(payload);
  const raw = {...envelope, ...recordValue(envelope.start_payload ?? envelope.data ?? payload)};
  const goal = recordValue(raw.goal ?? raw.teaching_goal);
  if (!Object.keys(goal).length) throw new Error("后端没有返回课节教学目标");
  const syllabusRef = recordValue(raw.syllabus_ref);
  const stagedResourceIds = Array.isArray(raw.staged_resource_ids)
    ? raw.staged_resource_ids.flatMap((item) => typeof item === "string" && item ? [item] : [])
    : undefined;
  return {
    ...raw,
    goal,
    syllabus_ref: Object.keys(syllabusRef).length ? syllabusRef : undefined,
    staged_resource_ids: stagedResourceIds,
  };
}

export interface TeachingStreamHandlers {
  onStatus?: (label: string) => void;
  onMeta?: (meta: {skill?: string}) => void;
  onDelta?: (text: string) => void;
  onLifecycle?: (update: HarnessLifecycleUpdate) => void;
  onRunIdentity?: (identity: HarnessRunIdentity) => void;
}

export interface ChatStreamHandlers {
  onStatus?: (label: string) => void;
  onMeta?: (meta: {model?: string; webSearchUsed?: boolean; sourceCount?: number}) => void;
  onDelta?: (text: string) => void;
  onLifecycle?: (update: HarnessLifecycleUpdate) => void;
  onRunIdentity?: (identity: HarnessRunIdentity) => void;
}

export interface HarnessRunIdentity {
  requestId: string;
  runId?: string;
  turnId?: string;
  taskId?: string;
  taskVersion?: number;
}

export interface BackgroundTask {
  schema: "teaching_skill_miner.background_task.v1";
  task_id: string;
  run_id: string;
  turn_id: string;
  operation: StreamOperation;
  status: "queued" | "running" | "cancel_requested" | "suspended" | "completed" | "cancelled" | "failed" | "handoff";
  version: number;
  created_at_utc: string;
  updated_at_utc: string;
  last_sequence: number;
  terminal_type: string | null;
  error_code: string | null;
  restart_policy: "safe_checkpoint_only" | "unsafe_external_effect_handoff";
  cancelable: boolean;
  resumable: boolean;
  cancel_command_pending: boolean;
  resume_command_pending: boolean;
  content_included: false;
}

export interface HarnessLifecycleUpdate {
  id: string;
  label: string;
  detail: string;
  state: "running" | "completed" | "failed";
}

type StreamOperation = "chat" | "start" | "step";

interface CommonStreamHandlers {
  onStatus?: (label: string) => void;
  onDelta?: (text: string) => void;
  onEvent?: (event: HarnessEventEnvelope) => void;
  onLifecycle?: (update: HarnessLifecycleUpdate) => void;
  onRunIdentity?: (identity: HarnessRunIdentity) => void;
}

class HarnessRemoteError extends Error {
  readonly terminalType: string;

  constructor(terminalType: string, message: string) {
    super(message);
    this.name = "HarnessRemoteError";
    this.terminalType = terminalType;
  }
}

class PrematureHarnessStreamEnd extends Error {
  constructor() {
    super("Teaching Agent stream ended before a terminal event");
    this.name = "PrematureHarnessStreamEnd";
  }
}

export class HarnessStreamHandoffError extends Error {
  readonly identity: HarnessRunIdentity;

  constructor(identity: HarnessRunIdentity) {
    super(identity.taskId
      ? `网络长时间中断；后台任务 ${identity.taskId} 已保留。请恢复网络后查询状态，不要重新发送。`
      : "网络长时间中断；请求尚未在服务器注册，可在恢复网络后安全重试。");
    this.name = "HarnessStreamHandoffError";
    this.identity = identity;
  }
}

function recordCandidate(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function remoteError(event: HarnessEventEnvelope) {
  const payload = event.payload;
  const detail = payload.safe_message ?? payload.message ?? payload.error ?? payload.reason_code ?? payload.error_code;
  if (event.type === "run.cancelled") return new HarnessRemoteError(event.type, "当前生成已停止");
  if (event.type === "run.handoff") return new HarnessRemoteError(event.type, String(detail ?? "当前运行需要人工接管"));
  return new HarnessRemoteError(event.type, String(detail ?? "Teaching Agent 运行失败"));
}

function responseErrorMessage(status: number, raw: string) {
  try {
    const payload = JSON.parse(raw) as {error?: unknown};
    if (typeof payload.error === "string" && payload.error) return payload.error;
  } catch {
    // The upstream may terminate before it can serialize an error body.
  }
  return `Teaching Agent 流式请求失败 (${status})`;
}

function abortError() {
  return new DOMException("The operation was aborted", "AbortError");
}

function waitBeforeReconnect(signal: AbortSignal | undefined, milliseconds: number) {
  if (signal?.aborted) return Promise.reject(abortError());
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", cancel);
      resolve();
    }, milliseconds);
    const cancel = () => {
      clearTimeout(timer);
      reject(abortError());
    };
    signal?.addEventListener("abort", cancel, {once: true});
  });
}

function createDeltaBatcher(handler: ((text: string) => void) | undefined) {
  let pending = "";
  let timer: ReturnType<typeof setTimeout> | null = null;
  const flush = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    if (!pending) return;
    const value = pending;
    pending = "";
    handler?.(value);
  };
  return {
    push(value: string) {
      if (!value) return;
      pending += value;
      if (pending.length >= 65_536) {
        flush();
        return;
      }
      if (timer === null) timer = setTimeout(flush, STREAM_RENDER_INTERVAL_MS);
    },
    flush,
  };
}

const toolLabels: Record<string, string> = {
  inspect_student_state: "读取学生状态",
  inspect_recent_history: "读取最近对话",
  search_skills: "查找教学 Skill",
  select_skills: "选择教学路径",
  set_next_focus: "设置下一学习重点",
  evaluate_termination: "检查学习目标",
  web_search: "联网搜索",
};

function lifecycleUpdate(event: HarnessEventEnvelope): HarnessLifecycleUpdate | null {
  if (!harnessEventIsUserVisible(event) || !event.type.startsWith("tool.")) return null;
  const payload = event.payload;
  const toolName = typeof payload.tool_name === "string" && payload.tool_name ? payload.tool_name : "tool";
  const callId = typeof payload.call_id === "string" && payload.call_id ? payload.call_id : `${toolName}-${event.sequence}`;
  const friendlyName = toolLabels[toolName] ?? toolName;
  const attempt = Number.isInteger(payload.attempt) ? ` · 第 ${Number(payload.attempt)} 次` : "";
  if (event.type === "tool.requested") {
    return {id: callId, label: friendlyName, detail: `已准备${attempt}`, state: "running"};
  }
  if (event.type === "tool.started") {
    return {id: callId, label: friendlyName, detail: `正在运行${attempt}`, state: "running"};
  }
  if (event.type === "tool.progress") {
    const progress = typeof payload.progress_kind === "string" ? payload.progress_kind : "处理中";
    return {id: callId, label: friendlyName, detail: progress, state: "running"};
  }
  if (event.type === "tool.completed" || event.type === "tool.replayed") {
    const duration = Number.isFinite(Number(payload.duration_ms)) ? ` · ${Number(payload.duration_ms)}ms` : "";
    return {id: callId, label: friendlyName, detail: `${event.type === "tool.replayed" ? "已安全复用结果" : "已完成"}${duration}`, state: "completed"};
  }
  if (event.type === "tool.failed" || event.type === "tool.rejected") {
    const code = payload.error_code ?? payload.error_type ?? "未完成";
    return {id: callId, label: `${friendlyName} 未完成`, detail: String(code), state: "failed"};
  }
  return null;
}

async function consumeHarnessStream<T>(options: {
  operation: StreamOperation;
  payload: Record<string, unknown>;
  requestId?: string;
  persistOperation?: boolean;
  handlers: CommonStreamHandlers;
  signal?: AbortSignal;
  resolveResult: (
    value: HarnessOperationResult,
    assistantMessage: string,
    signal: AbortSignal | undefined,
  ) => Promise<T>;
}): Promise<T> {
  const requestId = options.requestId ?? `console-stream-${crypto.randomUUID()}`;
  let streamState: HarnessStreamState = initialHarnessStreamState(0, options.operation);
  let cancellationIdentity: HarnessRunIdentity = {requestId};
  let authorityResult: HarnessOperationResult | null = null;
  let assistantMessage = "";
  const deltaBatcher = createDeltaBatcher(options.handlers.onDelta);
  const maxConnections = STREAM_MAX_CONNECTIONS;
  if (options.persistOperation !== false) {
    await putDurableOperation(newDurableOperation(requestId, options.operation, options.payload)).catch(() => false);
  }
  options.handlers.onRunIdentity?.({requestId});
  const consumeEvent = (event: HarnessEventEnvelope) => {
    const reduction = reduceHarnessEvent(streamState, event);
    if (reduction.duplicate) return;
    streamState = reduction.state;
    cancellationIdentity = {
      ...cancellationIdentity,
      requestId,
      runId: event.run_id,
      turnId: event.turn_id,
    };
    options.handlers.onRunIdentity?.(cancellationIdentity);
    const userVisibleEvent = harnessEventIsUserVisible(event);
    if (userVisibleEvent) options.handlers.onEvent?.(event);
    const label = userVisibleEvent ? harnessStatusLabel(event) : null;
    if (label) options.handlers.onStatus?.(label);
    const lifecycle = userVisibleEvent ? lifecycleUpdate(event) : null;
    if (lifecycle) options.handlers.onLifecycle?.(lifecycle);
    const delta = harnessDelta(event);
    if (delta) {
      assistantMessage += delta;
      if (assistantMessage.length > 2_000_000) {
        throw new HarnessStreamProtocolError("Harness assistant message exceeds the client budget");
      }
      deltaBatcher.push(delta);
    }
    if (event.type === "operation.result") {
      authorityResult = harnessOperationResult(event, options.operation);
    }
    if (event.type !== "run.completed" && event.type.startsWith("run.") && streamState.terminalType) {
      throw remoteError(event);
    }
  };

  try {
    for (let connection = 0; connection < maxConnections; connection += 1) {
      if (options.signal?.aborted) throw abortError();
      let response: Response;
      try {
        response = await fetchTeacherAgent(`${pythonProxyBase}/stream`, {
          method: "POST",
          headers: {Accept: "text/event-stream", "Content-Type": "application/json"},
          body: JSON.stringify({
            operation: options.operation,
            payload: options.payload,
            request_id: requestId,
            ...(streamState.runId ? {
              run_id: streamState.runId,
              turn_id: streamState.turnId,
              after_sequence: streamState.lastSequence,
            } : {}),
          }),
          signal: options.signal,
        });
      } catch (error) {
        if (options.signal?.aborted) throw abortError();
        if (connection === maxConnections - 1) throw new HarnessStreamHandoffError(cancellationIdentity);
        const delay = reconnectDelay(connection);
        options.handlers.onStatus?.(`连接中断，${Math.ceil(delay / 1_000)} 秒后恢复…`);
        await waitBeforeReconnect(options.signal, delay);
        continue;
      }
      if (!response.ok || !response.body) {
        const message = responseErrorMessage(response.status, await response.text());
        if (response.status < 500) throw new Error(message);
        if (connection === maxConnections - 1) throw new HarnessStreamHandoffError(cancellationIdentity);
        await waitBeforeReconnect(options.signal, reconnectDelay(connection));
        continue;
      }
      if (!(response.headers.get("content-type") ?? "").toLowerCase().startsWith("text/event-stream")) {
        throw new HarnessStreamProtocolError("Teaching Agent stream returned an unexpected content type");
      }

      const headerRunId = response.headers.get("x-harness-run-id") ?? "";
      const headerTurnId = response.headers.get("x-harness-turn-id") ?? "";
      const headerTaskId = response.headers.get("x-background-task-id") ?? "";
      const rawTaskVersion = response.headers.get("x-background-task-version") ?? "";
      if (headerRunId || headerTurnId) {
        if (
          !/^[A-Za-z0-9_-]{1,160}$/.test(headerRunId)
          || !/^[A-Za-z0-9_-]{1,160}$/.test(headerTurnId)
          || (streamState.runId !== null && (streamState.runId !== headerRunId || streamState.turnId !== headerTurnId))
        ) {
          throw new HarnessStreamProtocolError("Teaching Agent stream returned an invalid run identity");
        }
        const taskVersion = Number(rawTaskVersion);
        if (
          !/^task_[0-9a-f]{40}$/.test(headerTaskId)
          || !Number.isInteger(taskVersion)
          || taskVersion < 0
        ) {
          throw new HarnessStreamProtocolError("Teaching Agent stream returned an invalid task identity");
        }
        cancellationIdentity = {
          requestId,
          runId: headerRunId,
          turnId: headerTurnId,
          taskId: headerTaskId,
          taskVersion,
        };
        options.handlers.onRunIdentity?.(cancellationIdentity);
        await markOperationRegistered(requestId, headerTaskId, taskVersion).catch(() => undefined);
        if (streamState.runId === null) {
          streamState = {...streamState, runId: headerRunId, turnId: headerTurnId};
        }
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const frames = new SseFrameDecoder();
      try {
        while (true) {
          const {done, value} = await reader.read();
          const parsedFrames = done
            ? [...frames.push(decoder.decode()), ...frames.finish()]
            : frames.push(decoder.decode(value, {stream: true}));
          for (const frame of parsedFrames) consumeEvent(decodeHarnessEvent(frame));
          if (streamState.terminalType) {
            deltaBatcher.flush();
            if (streamState.terminalType !== "run.completed") {
              throw new HarnessRemoteError(streamState.terminalType, "Teaching Agent 运行未正常完成");
            }
            if (authorityResult === null) {
              throw new HarnessStreamProtocolError("Teaching Agent 完成事件缺少权威结果");
            }
            const resolved = await options.resolveResult(authorityResult, assistantMessage, options.signal);
            await deleteDurableOperation(requestId).catch(() => undefined);
            return resolved;
          }
          if (done) throw new PrematureHarnessStreamEnd();
        }
      } catch (error) {
        if (options.signal?.aborted) throw abortError();
        if (error instanceof HarnessRemoteError) throw error;
        if (streamState.terminalType !== null) throw error;
        if (connection === maxConnections - 1) throw new HarnessStreamHandoffError(cancellationIdentity);
        // The reducer only advances after accepting a contiguous event. A new
        // POST resumes the same run strictly after that durable sequence.
        await reader.cancel().catch(() => undefined);
        const delay = reconnectDelay(connection);
        options.handlers.onStatus?.(`连接中断，${Math.ceil(delay / 1_000)} 秒后从上次事件恢复…`);
        await waitBeforeReconnect(options.signal, delay);
      } finally {
        await reader.cancel().catch(() => undefined);
        reader.releaseLock();
      }
    }
    throw new HarnessStreamHandoffError(cancellationIdentity);
  } catch (error) {
    if (error instanceof HarnessStreamHandoffError) {
      if (cancellationIdentity.taskId) await markOperationHandoff(requestId).catch(() => undefined);
    } else if (error instanceof HarnessRemoteError && error.terminalType === "run.handoff") {
      await markOperationHandoff(requestId).catch(() => undefined);
    } else if (!(error instanceof DOMException && error.name === "AbortError")) {
      if (cancellationIdentity.taskId) await markOperationHandoff(requestId).catch(() => undefined);
      else await deleteDurableOperation(requestId).catch(() => undefined);
    }
    throw error;
  } finally {
    deltaBatcher.flush();
  }
}

export function cancelHarnessRun(identity: HarnessRunIdentity, reason = "user_requested") {
  if (identity.taskId && Number.isInteger(identity.taskVersion)) {
    return fetchJson<{task: BackgroundTask}>(`${pythonProxyBase}/api/tasks/cancel`, {
      method: "POST",
      keepalive: true,
      body: JSON.stringify({
        task_id: identity.taskId,
        expected_version: identity.taskVersion,
        task_idempotency_key: `console-task-cancel-${crypto.randomUUID()}`,
        reason_code: reason.slice(0, 160),
      }),
    }).then(() => undefined);
  }
  if (!identity.runId && !identity.requestId) return Promise.resolve();
  return fetchJson<Record<string, unknown>>(`${pythonProxyBase}/cancel`, {
    method: "POST",
    keepalive: true,
    body: JSON.stringify({
      ...(identity.runId ? {run_id: identity.runId} : {}),
      request_id: identity.requestId,
      reason: reason.slice(0, 300),
    }),
  }).then(() => undefined);
}

export function listBackgroundTasks() {
  return fetchJson<{schema: string; tasks: BackgroundTask[]; content_included: false}>(`${pythonProxyBase}/api/tasks/list`, {
    method: "POST",
    body: "{}",
  });
}

export function backgroundTaskStatus(taskId: string) {
  return fetchJson<{task: BackgroundTask}>(`${pythonProxyBase}/api/tasks/status`, {
    method: "POST",
    body: JSON.stringify({task_id: taskId}),
  });
}

export function resumeBackgroundTask(task: Pick<BackgroundTask, "task_id" | "version">) {
  return fetchJson<{task: BackgroundTask; resume_requested: boolean}>(`${pythonProxyBase}/api/tasks/resume`, {
    method: "POST",
    body: JSON.stringify({
      task_id: task.task_id,
      expected_version: task.version,
      task_idempotency_key: `console-task-resume-${crypto.randomUUID()}`,
    }),
  });
}

export async function recoverUnregisteredOperation(operation: DurableOperation) {
  if (operation.state !== "unregistered" || operation.taskId) {
    throw new Error("Only an unregistered durable request may be resubmitted");
  }
  const common = {
    operation: operation.operation,
    payload: operation.payload,
    requestId: operation.requestId,
    persistOperation: false,
    handlers: {},
  };
  if (operation.operation === "chat") {
    return consumeHarnessStream({...common, operation: "chat", resolveResult: resolveChatResult});
  }
  return consumeHarnessStream({
    ...common,
    resolveResult: (result, _message, signal) => resolveTeachingResult(
      result,
      signal,
      operation.operation === "step" && typeof operation.payload.session_id === "string"
        ? operation.payload.session_id
        : undefined,
    ),
  });
}

async function resolveTeachingResult(
  result: HarnessOperationResult,
  signal: AbortSignal | undefined,
  expectedSessionId?: string,
) {
  if (result.operation === "chat") {
    throw new HarnessStreamProtocolError("Teaching stream returned Chat authority");
  }
  const reference = result.sessionRef;
  if (expectedSessionId && reference.session_id !== expectedSessionId) {
    throw new HarnessStreamProtocolError("Teaching stream changed the active session identity");
  }
  const session = await resumeTeachingSession(reference.session_id, signal);
  validateHarnessSessionSnapshot(reference, session);
  return session;
}

async function resolveChatResult(
  result: HarnessOperationResult,
  assistantMessage: string,
): Promise<ChatResponse> {
  if (result.operation !== "chat") {
    throw new HarnessStreamProtocolError("Chat stream returned teaching authority");
  }
  const metadata = result.chat;
  await validateHarnessChatMessage(metadata, assistantMessage);
  return {
    schema_version: "1.0",
    mode: "chat",
    message: assistantMessage,
    provider: metadata.provider,
    model: metadata.model,
    latency_ms: metadata.latency_ms,
    usage: metadata.usage,
    web_search_requested: metadata.web_search_requested,
    web_search_used: metadata.web_search_used,
    sources: metadata.sources,
  };
}

export function streamTeachingTurn(
  payload: Record<string, unknown>,
  handlers: TeachingStreamHandlers,
  signal?: AbortSignal,
) {
  return consumeHarnessStream<TeacherSessionResponse>({
    operation: "step",
    payload,
    handlers: {
      ...handlers,
      onEvent: (event) => {
        const output = recordCandidate(event.payload.output);
        const skill = output?.skill;
        const skillRecord = recordCandidate(skill);
        const skillName = skillRecord?.name ?? output?.selected_skill_id ?? event.payload.skill;
        if (typeof skillName === "string" && skillName) handlers.onMeta?.({skill: skillName});
      },
    },
    signal,
    resolveResult: (result, _assistantMessage, requestSignal) => resolveTeachingResult(
      result,
      requestSignal,
      typeof payload.session_id === "string" ? payload.session_id : undefined,
    ),
  });
}

export function streamTeachingStart(
  payload: Record<string, unknown>,
  handlers: TeachingStreamHandlers,
  signal?: AbortSignal,
) {
  return consumeHarnessStream<TeacherSessionResponse>({
    operation: "start",
    payload,
    handlers: {
      ...handlers,
      onEvent: (event) => {
        const output = recordCandidate(event.payload.output);
        const skill = recordCandidate(output?.skill);
        const skillName = skill?.name ?? output?.selected_skill_id ?? event.payload.skill;
        if (typeof skillName === "string" && skillName) handlers.onMeta?.({skill: skillName});
      },
    },
    signal,
    resolveResult: (result, _assistantMessage, requestSignal) => resolveTeachingResult(result, requestSignal),
  });
}

export function streamChat(
  messages: ChatTurn[],
  options: {remoteConsentId: string; webSearch?: boolean; webSearchConsentId?: string; projectId?: string; chatThreadId?: string; resourceRefs?: ChatResourceRef[]},
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
) {
  return consumeHarnessStream<ChatResponse>({
    operation: "chat",
    payload: {
      messages,
      remote_consent_id: options.remoteConsentId,
      web_search: Boolean(options.webSearch),
      ...(options.webSearch ? {web_search_consent_id: options.webSearchConsentId} : {}),
      ...(options.projectId ? {project_id: options.projectId} : {}),
      ...(options.chatThreadId ? {chat_thread_id: options.chatThreadId} : {}),
      ...(options.resourceRefs?.length ? {resource_refs: options.resourceRefs} : {}),
    },
    handlers: {
      ...handlers,
      onEvent: (event) => {
        const payload = event.payload;
        if (event.type === "message.start" || event.type === "operation.meta") {
          handlers.onMeta?.({model: typeof payload.model === "string" ? payload.model : undefined});
        }
      },
    },
    signal,
    resolveResult: async (result, assistantMessage) => {
      const response = await resolveChatResult(result, assistantMessage);
      handlers.onMeta?.({
        model: response.model ?? undefined,
        webSearchUsed: response.web_search_used === true,
        sourceCount: response.sources?.length ?? 0,
      });
      return response;
    },
  });
}
