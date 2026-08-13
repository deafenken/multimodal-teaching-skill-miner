import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

import {proxyAccountDataRights} from "../lib/account-data-rights-bff.ts";
import {
  completeCommittedAccountDeletion,
  loadAccountDeletionBrowserRecovery,
  parseAccountDeletionReceipt,
  parseAccountDeletionStatus,
  saveAccountDeletionBrowserRecovery,
  ACCOUNT_DELETION_RECOVERY_STORAGE_KEY
} from "../lib/account-data-rights.ts";

const API_ORIGIN = "http://127.0.0.1:49300";
const CONSOLE_ORIGIN = "http://console.example.test";
const CSRF = "c".repeat(43);

process.env.TEACHLAB_HARNESS_MODE = "authenticated_apps_api";
process.env.TEACHLAB_APPS_API_URL = API_ORIGIN;
process.env.TEACHLAB_CONSOLE_ORIGIN = CONSOLE_ORIGIN;
process.env.TEACHLAB_APPS_API_COOKIE_MODE = "development";

function accountRequest(
  path: string,
  options: {
    method?: "GET" | "POST";
    body?: BodyInit;
    headers?: Record<string, string>;
    signal?: AbortSignal;
  } = {}
): Request {
  const method = options.method ?? "GET";
  const headers = new Headers({
    Host: "console.example.test",
    "Sec-Fetch-Site": "same-origin",
    Cookie: `analytics=never-forward; teachlab_local_session=never-forward; teachlab_session=signed-session; teachlab_csrf=${CSRF}; teachlab_deletion_status=status-capability`,
  });
  if (method === "POST") {
    headers.set("Origin", CONSOLE_ORIGIN);
    headers.set("Content-Type", "application/json");
    headers.set("x-teachlab-csrf-token", CSRF);
  }
  for (const [name, value] of Object.entries(options.headers ?? {})) {
    headers.set(name, value);
  }
  return new Request(`${CONSOLE_ORIGIN}/api/teacher-agent/account/${path}`, {
    method,
    headers,
    body: options.body,
    signal: options.signal,
  });
}

function receipt() {
  return {
    schema: "teachlab.account_deletion_receipt.v1",
    status: "permanently_deleted",
    receipt_id: `adelr_${"a".repeat(32)}`,
    scope_sha256: "b".repeat(64),
    operation_id_sha256: "c".repeat(64),
    deleted_at: "2026-08-12T00:00:00.000Z",
    deleted_counts: {
      postgres_events: 2,
      postgres_tasks: 1,
      postgres_artifacts: 1,
      postgres_sessions: 1,
      postgres_auth_sessions: 3,
      worker_files: 10,
      worker_bytes: 1_024,
      worker_roots: 1
    },
    all_devices_session_authority_deleted: true,
    scope_tombstone_retained: true,
    user_managed_export_copies_deleted: false,
    user_managed_export_copies_status: "outside_service_control",
    remote_provider_copies_deleted: false,
    remote_provider_copies_status:
      "outside_service_control_subject_to_provider_retention",
    identity_provider_account_deleted: false,
    identity_provider_account_status:
      "outside_service_control_contact_organization_idp",
    operator_backup_copies_deleted: false,
    operator_backup_copies_status:
      "pending_retention_expiry_or_operator_crypto_erasure",
    receipt_sha256: "d".repeat(64)
  };
}

class MemoryStorage {
  readonly values = new Map<string, string>();
  get length() { return this.values.size; }
  key(index: number) { return [...this.values.keys()][index] ?? null; }
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
  removeItem(key: string) { this.values.delete(key); }
}

test("deletion receipt refuses any stronger remote-provider deletion claim", () => {
  assert.equal(parseAccountDeletionReceipt(receipt()).status, "permanently_deleted");
  assert.throws(() => parseAccountDeletionReceipt({
    ...receipt(),
    remote_provider_copies_deleted: true
  }), /Invalid account deletion receipt/);
  assert.throws(() => parseAccountDeletionReceipt({
    ...receipt(),
    identity_provider_account_deleted: true
  }), /Invalid account deletion receipt/);
  assert.throws(() => parseAccountDeletionReceipt({
    ...receipt(),
    operator_backup_copies_deleted: true
  }), /Invalid account deletion receipt/);
  assert.throws(() => parseAccountDeletionReceipt({
    ...receipt(),
    tenant_id: "must-never-reach-browser"
  }), /Invalid account deletion receipt/);
});

test("browser cleanup cannot run before a validated committed receipt", async () => {
  let clears = 0;
  const storage = new MemoryStorage();
  await assert.rejects(completeCommittedAccountDeletion(
    {...receipt(), status: "deleting"},
    {
      sessionStorage: storage,
      localStorage: storage,
      clearIndexedDb: async () => { clears += 1; return true; }
    }
  ), /Invalid account deletion receipt/);
  assert.equal(clears, 0);
});

test("data rights panel preserves failure state and documents deletion boundaries", () => {
  const source = readFileSync(
    new URL("../components/workbench/account-data-rights-panel.tsx", import.meta.url),
    "utf8"
  );
  assert.ok(source.includes("await props.reauthenticate()"));
  assert.ok(source.includes("await props.committedDeletion(committed)"));
  assert.ok(source.includes("当前挑战和状态已保留"));
  assert.ok(source.includes("服务器删除已提交，但本机浏览器清理不完整"));
  assert.ok(source.includes("模型/搜索提供商"));
  assert.ok(source.includes("组织身份提供方中的账户未被删除"));
  assert.ok(source.includes("运维备份副本仍须等待保留期到期"));
  assert.ok(source.includes("本机 standalone 模式没有认证账户边界"));
  assert.ok(source.includes("立即恢复删除"));
  assert.ok(source.includes("页面刷新或 API 重启不会取消已确认的删除"));
  assert.ok(source.includes("saveAccountDeletionBrowserRecovery"));
});

test("browser challenge/idempotency recovery is session-bound, validated, and expiring", () => {
  const storage = new MemoryStorage();
  const challenge = {
    schema: "teachlab.account_deletion_challenge.v1" as const,
    challenge_id: `adelc_${"a".repeat(32)}`,
    confirmation_token: "b".repeat(43),
    confirmation_phrase: "PERMANENTLY DELETE MY TEACHLAB ACCOUNT" as const,
    expires_at: "2026-08-12T00:05:00.000Z",
    revision: 2
  };
  saveAccountDeletionBrowserRecovery(storage, challenge, "delete-request-00000001");
  assert.equal(
    loadAccountDeletionBrowserRecovery(storage, Date.parse("2026-08-12T00:01:00Z"))
      ?.idempotency_key,
    "delete-request-00000001"
  );
  storage.setItem(ACCOUNT_DELETION_RECOVERY_STORAGE_KEY, JSON.stringify({
    schema: "teachlab.account_deletion_browser_recovery.v1",
    challenge,
    idempotency_key: "delete-request-00000001",
    tenant_id: "must-never-be-restored"
  }));
  assert.equal(
    loadAccountDeletionBrowserRecovery(storage, Date.parse("2026-08-12T00:01:00Z")),
    null
  );
  saveAccountDeletionBrowserRecovery(storage, challenge, "delete-request-00000001");
  assert.equal(
    loadAccountDeletionBrowserRecovery(storage, Date.parse("2026-08-12T00:06:00Z")),
    null
  );
  assert.equal(storage.getItem(ACCOUNT_DELETION_RECOVERY_STORAGE_KEY), null);
});

test("deletion status parser binds each public status to one exact durable phase shape", () => {
  assert.equal(parseAccountDeletionStatus({
    schema: "teachlab.account_deletion_status.v1",
    status: "retryable_failure",
    phase: "quarantining",
    revision: 7,
    retryable_failure_code: "deletion_quarantining_retry",
    receipt: null
  }).phase, "quarantining");
  assert.throws(() => parseAccountDeletionStatus({
    schema: "teachlab.account_deletion_status.v1",
    status: "deleting",
    phase: "completed",
    revision: 8,
    retryable_failure_code: null,
    receipt: null
  }), /Invalid account deletion status receipt/);
});

test("status capability can resume deletion without restoring an auth session", async (t) => {
  const originalFetch = globalThis.fetch;
  const calls: Headers[] = [];
  globalThis.fetch = (async (_input, init) => {
    calls.push(new Headers(init?.headers));
    return Response.json({
      schema: "teachlab.account_deletion_status.v1",
      status: "deleting",
      phase: "quarantining",
      revision: 7,
      retryable_failure_code: null,
      receipt: null
    });
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const request = new Request(`${CONSOLE_ORIGIN}/api/teacher-agent/account/deletion/resume`, {
    method: "POST",
    headers: {
      Host: "console.example.test",
      Origin: CONSOLE_ORIGIN,
      "Sec-Fetch-Site": "same-origin",
      "Content-Type": "application/json",
      Cookie: "teachlab_deletion_status=status-capability"
    },
    body: "{}"
  });
  const response = await proxyAccountDataRights(request, ["deletion", "resume"]);
  assert.equal(response.status, 200);
  assert.equal(calls.length, 1);
  assert.equal(
    calls[0]?.get("cookie"),
    "teachlab_deletion_status=status-capability"
  );
  assert.equal(calls[0]?.get("cookie")?.includes("teachlab_session="), false);
});

test("account export forwards only server authority cookies and streams verified ZIP metadata", async (t) => {
  const calls: Array<{url: string; headers: Headers; method: string}> = [];
  const bytes = new TextEncoder().encode("canonical-zip-fixture");
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    calls.push({
      url: String(input),
      headers: new Headers(init?.headers),
      method: init?.method ?? "GET",
    });
    return new Response(bytes, {
      headers: {
        "Content-Type": "application/zip",
        "Content-Length": String(bytes.byteLength),
        "Content-Disposition": 'attachment; filename="teachlab-account-export.zip"',
        "x-manifest-sha256": "a".repeat(64),
      }
    });
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const response = await proxyAccountDataRights(accountRequest("export"), ["export"]);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-type"), "application/zip");
  assert.equal(response.headers.get("content-length"), String(bytes.byteLength));
  assert.equal(response.headers.get("x-manifest-sha256"), "a".repeat(64));
  assert.equal(await response.text(), "canonical-zip-fixture");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `${API_ORIGIN}/api/v1/account/export`);
  assert.equal(calls[0].method, "GET");
  assert.equal(
    calls[0].headers.get("cookie"),
    `teachlab_session=signed-session; teachlab_csrf=${CSRF}; teachlab_deletion_status=status-capability`
  );
  assert.equal(calls[0].headers.has("authorization"), false);
  assert.equal(calls[0].headers.has("x-tenant"), false);
  assert.equal(calls[0].headers.has("x-user"), false);
  assert.equal(calls[0].headers.get("origin"), CONSOLE_ORIGIN);
});

test("account BFF rejects forged identity and oversized mutations before backend effects", async (t) => {
  let backendCalls = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    backendCalls += 1;
    return Response.json({unexpected: true});
  }) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const forged = await proxyAccountDataRights(accountRequest("deletion/prepare", {
    method: "POST",
    body: "{}",
    headers: {"x-tenant": "attacker-scope"},
  }), ["deletion", "prepare"]);
  assert.equal(forged.status, 403);

  const oversized = await proxyAccountDataRights(accountRequest("deletion/confirm", {
    method: "POST",
    body: JSON.stringify({padding: "x".repeat(16 * 1024)}),
  }), ["deletion", "confirm"]);
  assert.equal(oversized.status, 413);
  assert.equal(backendCalls, 0);
});

test("account BFF bounds upstream JSON and export streams and propagates detach", async (t) => {
  const originalFetch = globalThis.fetch;
  let cancelled = "";
  globalThis.fetch = (async () => new Response(new ReadableStream<Uint8Array>({
    pull() {},
    cancel(reason) { cancelled = String(reason); },
  }), {
    headers: {
      "Content-Type": "application/zip",
      "Content-Length": "12",
      "Content-Disposition": 'attachment; filename="account.zip"',
      "x-manifest-sha256": "b".repeat(64),
    }
  })) as typeof fetch;
  t.after(() => { globalThis.fetch = originalFetch; });

  const streaming = await proxyAccountDataRights(accountRequest("export"), ["export"]);
  assert.equal(streaming.status, 200);
  await streaming.body?.cancel("browser_navigation_detach");
  assert.equal(cancelled, "browser_navigation_detach");

  let oversizedCancelled = false;
  globalThis.fetch = (async () => new Response(new ReadableStream<Uint8Array>({
    cancel() { oversizedCancelled = true; },
  }), {
    headers: {
      "Content-Type": "application/json",
      "Content-Length": String(64 * 1024 + 1),
    }
  })) as typeof fetch;
  const oversizedJson = await proxyAccountDataRights(accountRequest("deletion/status"), [
    "deletion", "status"
  ]);
  assert.equal(oversizedJson.status, 503);
  assert.equal(oversizedCancelled, true);

  let invalidExportCancelled = false;
  globalThis.fetch = (async () => new Response(new ReadableStream<Uint8Array>({
    cancel() { invalidExportCancelled = true; },
  }), {
    headers: {
      "Content-Type": "application/zip",
      "Content-Length": String(400 * 1024 * 1024 + 1),
      "Content-Disposition": 'attachment; filename="account.zip"',
      "x-manifest-sha256": "c".repeat(64),
    }
  })) as typeof fetch;
  const oversizedExport = await proxyAccountDataRights(accountRequest("export"), ["export"]);
  assert.equal(oversizedExport.status, 503);
  assert.equal(invalidExportCancelled, true);
});
