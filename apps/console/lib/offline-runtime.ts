export const OFFLINE_DB_NAME = "teachlab-console-runtime";
export const OFFLINE_DB_VERSION = 2;
export const OUTBOX_TTL_MS = 7 * 24 * 60 * 60 * 1_000;
export const MAX_PERSISTED_PAYLOAD_CHARS = 2_000_000;
export const WORKSPACE_CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1_000;
export const MAX_WORKSPACE_CACHE_CHARS = 5_000_000;
export const MAX_WORKSPACE_CACHE_ENTRIES = 12;
export const STREAM_RENDER_INTERVAL_MS = 40;
export const STREAM_MAX_CONNECTIONS = 12;
export const OFFLINE_STATIC_CACHE_PREFIX = "teachlab-console-static-";
export const ACCOUNT_CACHE_SCOPE_STORAGE_KEY = "teachlab.account-cache-scope.v1";
export const ACCOUNT_CACHE_SCOPE_PATTERN = /^acs1_[A-Za-z0-9_-]{43}$/;

export type DurableOperationState = "unregistered" | "registered" | "handoff";

export interface DurableOperation {
  requestId: string;
  operation: "chat" | "start" | "step";
  payload: Record<string, unknown>;
  payloadRedacted: boolean;
  state: DurableOperationState;
  taskId?: string;
  taskVersion?: number;
  createdAt: number;
  updatedAt: number;
  expiresAt: number;
}

export interface DurableTaskSnapshot {
  task_id: string;
  version: number;
  status: "queued" | "running" | "cancel_requested" | "suspended" | "completed" | "cancelled" | "failed" | "handoff";
  resumable: boolean;
}

export type RecoveryDecision =
  | {kind: "resubmit_unregistered"; requestId: string}
  | {kind: "observe_registered"; taskId: string}
  | {kind: "resume_registered"; taskId: string; taskVersion: number}
  | {kind: "remove_terminal"; taskId: string}
  | {kind: "handoff"; taskId: string; reason: string};

export interface OfflineWorkspaceSnapshot {
  schema: "teachlab.offline_workspace_snapshot.v2";
  accountCacheScope: string;
  projectId: string;
  projectUpdatedAt: string;
  project: Record<string, unknown>;
  savedAt: number;
  expiresAt: number;
  serializedChars: number;
}

type StoreName = "drafts" | "outbox" | "workspaces";

function databaseAvailable() {
  return typeof indexedDB !== "undefined";
}

function requestValue<T>(request: IDBRequest<T>) {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

function transactionDone(transaction: IDBTransaction) {
  return new Promise<void>((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB transaction failed"));
    transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB transaction aborted"));
  });
}

async function openDatabase() {
  if (!databaseAvailable()) return null;
  const request = indexedDB.open(OFFLINE_DB_NAME, OFFLINE_DB_VERSION);
  request.onupgradeneeded = () => {
    const database = request.result;
    if (!database.objectStoreNames.contains("drafts")) database.createObjectStore("drafts", {keyPath: "key"});
    if (!database.objectStoreNames.contains("outbox")) database.createObjectStore("outbox", {keyPath: "requestId"});
    if (!database.objectStoreNames.contains("workspaces")) database.createObjectStore("workspaces", {keyPath: "projectId"});
  };
  return requestValue(request);
}

async function withStore<T>(name: StoreName, mode: IDBTransactionMode, callback: (store: IDBObjectStore) => Promise<T>) {
  const database = await openDatabase();
  if (!database) return null;
  try {
    const transaction = database.transaction(name, mode);
    const result = await callback(transaction.objectStore(name));
    await transactionDone(transaction);
    return result;
  } finally {
    database.close();
  }
}

export async function loadDraft(key: string) {
  const value = await withStore("drafts", "readonly", async (store) => requestValue(store.get(key)));
  if (!value || typeof value !== "object") return "";
  const text = (value as {text?: unknown}).text;
  return typeof text === "string" ? text : "";
}

export async function saveDraft(key: string, text: string) {
  if (text.length > 200_000) throw new Error("Draft exceeds the local persistence budget");
  await withStore("drafts", "readwrite", async (store) => {
    await requestValue(store.put({key, text, updatedAt: Date.now()}));
  });
}

export async function putDurableOperation(operation: DurableOperation) {
  const serialized = JSON.stringify(operation.payload);
  if (serialized.length > MAX_PERSISTED_PAYLOAD_CHARS) return false;
  await withStore("outbox", "readwrite", async (store) => {
    await requestValue(store.put(operation));
  });
  return true;
}

export async function markOperationRegistered(requestId: string, taskId: string, taskVersion: number) {
  await withStore("outbox", "readwrite", async (store) => {
    const current = await requestValue(store.get(requestId)) as DurableOperation | undefined;
    if (!current) return;
    await requestValue(store.put(redactedDurableOperation({
      ...current,
      state: "registered",
      taskId,
      taskVersion,
      updatedAt: Date.now(),
    })));
  });
}

export async function markOperationHandoff(requestId: string) {
  await withStore("outbox", "readwrite", async (store) => {
    const current = await requestValue(store.get(requestId)) as DurableOperation | undefined;
    if (!current) return;
    await requestValue(store.put(redactedDurableOperation({
      ...current,
      state: "handoff",
      updatedAt: Date.now(),
    })));
  });
}

export async function deleteDurableOperation(requestId: string) {
  await withStore("outbox", "readwrite", async (store) => {
    await requestValue(store.delete(requestId));
  });
}

export async function listDurableOperations(now = Date.now()): Promise<DurableOperation[]> {
  const values = await withStore("outbox", "readwrite", async (store) => {
    const items = await requestValue(store.getAll()) as DurableOperation[];
    const migrated: DurableOperation[] = [];
    for (const item of items) {
      if (item.expiresAt <= now) {
        await requestValue(store.delete(item.requestId));
        continue;
      }
      const safe = item.state === "unregistered" ? item : redactedDurableOperation(item);
      if (safe !== item) await requestValue(store.put(safe));
      migrated.push(safe);
    }
    return migrated;
  });
  return values ?? [];
}

export function recoveryDecision(operation: DurableOperation, task?: DurableTaskSnapshot): RecoveryDecision {
  if (!operation.taskId || operation.state === "unregistered") {
    return {kind: "resubmit_unregistered", requestId: operation.requestId};
  }
  if (!task) return {kind: "observe_registered", taskId: operation.taskId};
  if (["completed", "cancelled", "failed"].includes(task.status)) {
    return {kind: "remove_terminal", taskId: task.task_id};
  }
  if (task.status === "suspended" && task.resumable) {
    return {kind: "resume_registered", taskId: task.task_id, taskVersion: task.version};
  }
  if (task.status === "handoff" || (task.status === "suspended" && !task.resumable)) {
    return {kind: "handoff", taskId: task.task_id, reason: "后台任务不能安全自动恢复，需要人工处理"};
  }
  return {kind: "observe_registered", taskId: task.task_id};
}

export function reconnectDelay(attempt: number, random = Math.random()) {
  const boundedAttempt = Math.max(0, Math.min(10, Math.floor(attempt)));
  const base = Math.min(30_000, 500 * 2 ** boundedAttempt);
  const jitter = Math.max(0, Math.min(1, random)) * Math.min(1_000, base * 0.2);
  return Math.round(base + jitter);
}

export function boundedMessageWindow<T>(items: readonly T[], limit: number) {
  const safeLimit = Math.max(1, Math.floor(limit));
  const hidden = Math.max(0, items.length - safeLimit);
  return {hidden, items: hidden ? items.slice(hidden) : [...items]};
}

export function newDurableOperation(requestId: string, operation: DurableOperation["operation"], payload: Record<string, unknown>, now = Date.now()): DurableOperation {
  return {requestId, operation, payload, payloadRedacted: false, state: "unregistered", createdAt: now, updatedAt: now, expiresAt: now + OUTBOX_TTL_MS};
}

/**
 * Once the server has durably registered a task, recovery uses only the task
 * identity and version. Keeping the original learner text would add no
 * recovery value and would unnecessarily retain it in IndexedDB for days.
 */
export function redactedDurableOperation(operation: DurableOperation): DurableOperation {
  if (operation.state === "unregistered") return operation;
  if (operation.payloadRedacted === true && Object.keys(operation.payload).length === 0) {
    return operation;
  }
  return {...operation, payload: {}, payloadRedacted: true};
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function boundedString(value: unknown, maximum: number, allowEmpty = false): value is string {
  return typeof value === "string" && value.length <= maximum && (allowEmpty || Boolean(value.trim()));
}

function utcTimestamp(value: unknown) {
  return boundedString(value, 64) && Number.isFinite(Date.parse(value));
}

function safeHttpUrl(value: unknown) {
  if (!boundedString(value, 2_000)) return false;
  try {
    const parsed = new URL(value);
    return ["http:", "https:"].includes(parsed.protocol) && Boolean(parsed.hostname) && !parsed.username && !parsed.password;
  } catch {
    return false;
  }
}

/**
 * Validate the intentionally bounded, read-only project snapshot kept for an
 * offline reload. The server remains authoritative: this projection is never
 * submitted back as a mutation and running/queued rows are not cacheable.
 */
export function validateOfflineWorkspaceProject(value: unknown): value is Record<string, unknown> {
  const project = recordValue(value);
  if (!project || project.schema !== "teaching_skill_miner.learning_project.v1") return false;
  if (!boundedString(project.project_id, 160) || !boundedString(project.title, 240)) return false;
  if (!boundedString(project.description, 4_000, true) || !utcTimestamp(project.created_at) || !utcTimestamp(project.updated_at)) return false;
  if (!["active", "archived"].includes(String(project.status)) || typeof project.pinned !== "boolean" || !recordValue(project.claim_boundary)) return false;
  if (!Array.isArray(project.chat_threads) || project.chat_threads.length > 100) return false;
  if (!Array.isArray(project.notes) || project.notes.length > 500) return false;
  for (const threadValue of project.chat_threads) {
    const thread = recordValue(threadValue);
    if (!thread || !boundedString(thread.thread_id, 160) || !boundedString(thread.title, 240)) return false;
    if (!utcTimestamp(thread.created_at) || !utcTimestamp(thread.updated_at)) return false;
    if (!Array.isArray(thread.messages) || thread.messages.length > 400) return false;
    for (const messageValue of thread.messages) {
      const message = recordValue(messageValue);
      if (!message || !boundedString(message.message_id, 200) || !boundedString(message.content, 120_000, true)) return false;
      if (!["user", "assistant", "tool"].includes(String(message.role))) return false;
      if (!["completed", "stopped", "failed"].includes(String(message.status))) return false;
      if (!utcTimestamp(message.created_at) || typeof message.web_search_used !== "boolean") return false;
      if (!Array.isArray(message.sources) || message.sources.length > 24) return false;
      if (message.sources.some((sourceValue) => {
        const source = recordValue(sourceValue);
        return !source || !boundedString(source.title, 300) || !safeHttpUrl(source.url);
      })) return false;
    }
  }
  for (const noteValue of project.notes) {
    const note = recordValue(noteValue);
    if (!note || !boundedString(note.note_id, 160) || !boundedString(note.title, 160) || !boundedString(note.body, 64_000, true)) return false;
    if (!utcTimestamp(note.created_at) || !utcTimestamp(note.updated_at)) return false;
  }
  return ["syllabus_ids", "teaching_session_ids", "resource_ids"].every((key) => {
    const identifiers = project[key];
    return Array.isArray(identifiers)
      && identifiers.length <= 500
      && identifiers.every((identifier) => boundedString(identifier, 200));
  });
}

function boundedOfflineWorkspaceProject(value: unknown): Record<string, unknown> | null {
  const source = recordValue(value);
  if (!source || !Array.isArray(source.chat_threads) || source.chat_threads.length > 100) return null;
  const project = structuredClone(source);
  project.chat_threads = source.chat_threads.map((threadValue) => {
    const thread = recordValue(threadValue);
    if (!thread || !Array.isArray(thread.messages) || thread.messages.length > 10_000) return threadValue;
    return {
      ...structuredClone(thread),
      messages: thread.messages.filter((messageValue) => {
        const message = recordValue(messageValue);
        return message && ["completed", "stopped", "failed"].includes(String(message.status));
      }).slice(-400),
    };
  });
  return validateOfflineWorkspaceProject(project) ? project : null;
}

export function offlineWorkspaceEnvelope(
  project: unknown,
  now = Date.now(),
  accountCacheScope?: string | null,
): OfflineWorkspaceSnapshot | null {
  if (!accountCacheScope || !ACCOUNT_CACHE_SCOPE_PATTERN.test(accountCacheScope)) return null;
  const projection = boundedOfflineWorkspaceProject(project);
  if (!projection) return null;
  const serialized = JSON.stringify(projection);
  if (serialized.length > MAX_WORKSPACE_CACHE_CHARS) return null;
  return {
    schema: "teachlab.offline_workspace_snapshot.v2",
    accountCacheScope,
    projectId: String(projection.project_id),
    projectUpdatedAt: String(projection.updated_at),
    project: projection,
    savedAt: now,
    expiresAt: now + WORKSPACE_CACHE_TTL_MS,
    serializedChars: serialized.length,
  };
}

export function validateOfflineWorkspaceEnvelope(
  value: unknown,
  now = Date.now(),
  accountCacheScope?: string | null,
): OfflineWorkspaceSnapshot | null {
  const envelope = recordValue(value);
  if (!envelope || envelope.schema !== "teachlab.offline_workspace_snapshot.v2") return null;
  if (!accountCacheScope || !ACCOUNT_CACHE_SCOPE_PATTERN.test(accountCacheScope)) return null;
  if (envelope.accountCacheScope !== accountCacheScope) return null;
  if (!boundedString(envelope.projectId, 160) || !boundedString(envelope.projectUpdatedAt, 64)) return null;
  if (!Number.isSafeInteger(envelope.savedAt) || !Number.isSafeInteger(envelope.expiresAt) || !Number.isSafeInteger(envelope.serializedChars)) return null;
  if (Number(envelope.savedAt) > now + 60_000 || Number(envelope.expiresAt) <= now) return null;
  if (Number(envelope.expiresAt) - Number(envelope.savedAt) !== WORKSPACE_CACHE_TTL_MS) return null;
  if (Number(envelope.serializedChars) > MAX_WORKSPACE_CACHE_CHARS || !validateOfflineWorkspaceProject(envelope.project)) return null;
  const project = envelope.project as Record<string, unknown>;
  if (project.project_id !== envelope.projectId || project.updated_at !== envelope.projectUpdatedAt) return null;
  if (JSON.stringify(project).length !== envelope.serializedChars) return null;
  return envelope as unknown as OfflineWorkspaceSnapshot;
}

export async function saveOfflineWorkspaceProject(project: unknown, now = Date.now()) {
  const accountCacheScope = currentAccountCacheScope();
  const envelope = offlineWorkspaceEnvelope(project, now, accountCacheScope);
  if (!envelope) return false;
  await withStore("workspaces", "readwrite", async (store) => {
    const current = await requestValue(store.getAll()) as unknown[];
    const valid = current
      .map((item) => validateOfflineWorkspaceEnvelope(item, now, accountCacheScope))
      .filter((item): item is OfflineWorkspaceSnapshot => Boolean(item))
      .sort((left, right) => right.savedAt - left.savedAt);
    for (const item of current) {
      const candidate = recordValue(item);
      if (candidate && !validateOfflineWorkspaceEnvelope(candidate, now, accountCacheScope) && typeof candidate.projectId === "string") {
        await requestValue(store.delete(candidate.projectId));
      }
    }
    for (const stale of valid.filter((item) => item.projectId !== envelope.projectId).slice(MAX_WORKSPACE_CACHE_ENTRIES - 1)) {
      await requestValue(store.delete(stale.projectId));
    }
    await requestValue(store.put(envelope));
  });
  return true;
}

export async function loadOfflineWorkspaceProject(projectId: string, now = Date.now()) {
  if (!projectId || projectId.length > 160) return null;
  const accountCacheScope = currentAccountCacheScope();
  if (!accountCacheScope) return null;
  const value = await withStore("workspaces", "readwrite", async (store) => {
    const candidate = await requestValue(store.get(projectId));
    const validated = validateOfflineWorkspaceEnvelope(candidate, now, accountCacheScope);
    if (!validated && candidate !== undefined) await requestValue(store.delete(projectId));
    return validated?.project ?? null;
  });
  return value ?? null;
}

function currentAccountCacheScope(): string | null {
  if (typeof localStorage === "undefined") return null;
  try {
    const value = localStorage.getItem(ACCOUNT_CACHE_SCOPE_STORAGE_KEY);
    return value && ACCOUNT_CACHE_SCOPE_PATTERN.test(value) ? value : null;
  } catch {
    return null;
  }
}

export async function deleteOfflineWorkspaceProject(projectId: string) {
  if (!projectId || projectId.length > 160) return;
  await withStore("workspaces", "readwrite", async (store) => {
    await requestValue(store.delete(projectId));
  });
}

/** Remove identity-adjacent browser state before another operator can sign in. */
export async function clearOfflineRuntimeForLogout(): Promise<boolean> {
  let complete = true;
  const database = await openDatabase();
  if (!database) return clearOfflineStaticCaches();
  try {
    const transaction = database.transaction(
      ["drafts", "outbox", "workspaces"],
      "readwrite",
    );
    for (const name of ["drafts", "outbox", "workspaces"] as const) {
      transaction.objectStore(name).clear();
    }
    await transactionDone(transaction);
  } catch {
    complete = false;
  } finally {
    database.close();
  }
  return (await clearOfflineStaticCaches()) && complete;
}

async function clearOfflineStaticCaches(): Promise<boolean> {
  if (typeof caches === "undefined") return true;
  try {
    const names = await caches.keys();
    const results = await Promise.all(names
      .filter((name) => name.startsWith(OFFLINE_STATIC_CACHE_PREFIX))
      .map((name) => caches.delete(name)));
    return results.every(Boolean);
  } catch {
    return false;
  }
}
