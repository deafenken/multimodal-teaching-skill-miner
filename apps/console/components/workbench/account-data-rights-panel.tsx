"use client";

import {Download, LoaderCircle, ShieldAlert, Trash2} from "lucide-react";
import {useEffect, useMemo, useState} from "react";

import {Button} from "@/components/ui/button";
import {
  ACCOUNT_DELETION_CONFIRMATION_PHRASE,
  type AccountDeletionChallenge,
  type AccountDeletionReceipt,
  type AccountDeletionStatus,
  clearAccountDeletionBrowserRecovery,
  loadAccountDeletionBrowserRecovery,
  saveAccountDeletionBrowserRecovery
} from "@/lib/account-data-rights";

export interface AccountDataRightsPanelProps {
  authenticated: boolean;
  authenticatedAppsApi: boolean;
  status?: AccountDeletionStatus | null;
  exportAccount(): Promise<void>;
  reauthenticate(): Promise<void>;
  prepareDeletion(): Promise<AccountDeletionChallenge>;
  confirmDeletion(input: {
    challenge_id: string;
    confirmation_token: string;
    confirmation_phrase: string;
    expected_revision: number;
    idempotency_key: string;
  }): Promise<AccountDeletionReceipt>;
  refreshStatus(): Promise<AccountDeletionStatus>;
  resumeDeletion(): Promise<AccountDeletionStatus>;
  statusChanged(status: AccountDeletionStatus): void;
  committedDeletion(receipt: AccountDeletionReceipt): Promise<void>;
}

type BusyAction = "export" | "prepare" | "confirm" | "resume" | null;
type ReconciledDeletion = "in_progress" | "completed" | "cleanup_incomplete";

function freshIdempotencyKey(): string {
  return `account-delete-${crypto.randomUUID()}`;
}

export function AccountDataRightsPanel(props: AccountDataRightsPanelProps) {
  const [challenge, setChallenge] = useState<AccountDeletionChallenge | null>(null);
  const [typedPhrase, setTypedPhrase] = useState("");
  const [idempotencyKey, setIdempotencyKey] = useState("");
  const [busy, setBusy] = useState<BusyAction>(null);
  const [error, setError] = useState("");
  const [receipt, setReceipt] = useState<AccountDeletionReceipt | null>(
    props.status?.receipt ?? null
  );
  const deletionActive = props.status?.status === "deleting"
    || props.status?.status === "retryable_failure";
  const available = props.authenticated && props.authenticatedAppsApi && !deletionActive;
  const phraseMatches = typedPhrase === ACCOUNT_DELETION_CONFIRMATION_PHRASE;
  const challengeExpired = useMemo(
    () => challenge ? Date.parse(challenge.expires_at) <= Date.now() : true,
    [challenge]
  );

  useEffect(() => {
    const recovered = loadAccountDeletionBrowserRecovery(window.sessionStorage);
    if (!recovered) return;
    setChallenge(recovered.challenge);
    setIdempotencyKey(recovered.idempotency_key);
  }, []);

  useEffect(() => {
    const status = props.status;
    if (!status) return;
    if (status.phase !== "prepared") {
      setChallenge(null);
      setTypedPhrase("");
      setIdempotencyKey("");
      clearAccountDeletionBrowserRecovery(window.sessionStorage);
    }
    if (status.receipt) setReceipt(status.receipt);
  }, [props.status]);

  if (!props.authenticatedAppsApi) {
    return (
      <section aria-labelledby="account-data-rights-title" className="grid gap-3">
        <h3 id="account-data-rights-title" className="text-sm font-semibold">账户数据权利</h3>
        <p className="text-xs leading-5 text-[var(--app-muted)]">
          本机 standalone 模式没有认证账户边界，因此不能声称提供账户级导出或永久删除。
          项目级导出与清除仍可在各项目中使用。
        </p>
      </section>
    );
  }

  const reconcileDurableStatus = async (
    status: AccountDeletionStatus
  ): Promise<ReconciledDeletion> => {
    props.statusChanged(status);
    if (!status.receipt) return "in_progress";
    setReceipt(status.receipt);
    setChallenge(null);
    setTypedPhrase("");
    setIdempotencyKey("");
    clearAccountDeletionBrowserRecovery(window.sessionStorage);
    try {
      await props.committedDeletion(status.receipt);
      return "completed";
    } catch {
      setError("服务器删除已提交，但本机浏览器清理不完整。账户内容保持锁定，请重试本机清理；不要重新提交删除。");
      return "cleanup_incomplete";
    }
  };

  const prepare = async () => {
    setBusy("prepare");
    setError("");
    try {
      await props.reauthenticate();
      const next = await props.prepareDeletion();
      setChallenge(next);
      setTypedPhrase("");
      const nextIdempotencyKey = freshIdempotencyKey();
      setIdempotencyKey(nextIdempotencyKey);
      if (!saveAccountDeletionBrowserRecovery(
        window.sessionStorage,
        next,
        nextIdempotencyKey
      )) {
        setError("浏览器拒绝保存删除挑战；请勿刷新此页，或修复站点存储权限后重新准备。");
      }
    } catch {
      setError("必须先通过组织身份提供方完成近期高保证重新认证，才能生成删除挑战。");
    } finally {
      setBusy(null);
    }
  };

  const confirm = async () => {
    if (!challenge || challengeExpired || !phraseMatches || !idempotencyKey) return;
    setBusy("confirm");
    setError("");
    let committed: AccountDeletionReceipt;
    try {
      committed = await props.confirmDeletion({
        challenge_id: challenge.challenge_id,
        confirmation_token: challenge.confirmation_token,
        confirmation_phrase: typedPhrase,
        expected_revision: challenge.revision,
        idempotency_key: idempotencyKey
      });
      setReceipt(committed);
      setChallenge(null);
      setIdempotencyKey("");
      clearAccountDeletionBrowserRecovery(window.sessionStorage);
    } catch {
      const reconciled = await props.refreshStatus()
        .then(reconcileDurableStatus)
        .catch(() => "in_progress" as const);
      if (reconciled === "in_progress") {
        setError("删除未确认完成。当前挑战和状态已保留，可查询状态或安全重试；请勿重复创建请求。");
      }
      setBusy(null);
      return;
    }
    try {
      // This callback detaches streams and clears IndexedDB/browser caches only
      // after the server returned a validated committed receipt.
      await props.committedDeletion(committed);
    } catch {
      setError("服务器删除已提交，但本机浏览器清理不完整。账户内容保持锁定，请重试本机清理；不要重新提交删除。");
    } finally {
      setBusy(null);
    }
  };

  const resume = async () => {
    setBusy("resume");
    setError("");
    try {
      const status = await props.resumeDeletion();
      await reconcileDurableStatus(status);
    } catch {
      const reconciled = await props.refreshStatus()
        .then(reconcileDurableStatus)
        .catch(() => "in_progress" as const);
      if (reconciled === "in_progress") {
        setError("删除恢复暂未完成。服务器会继续自动重试；本页也会持续查询持久化状态。");
      }
    } finally {
      setBusy(null);
    }
  };

  return (
    <section aria-labelledby="account-data-rights-title" className="grid gap-4">
      <div>
        <h3 id="account-data-rights-title" className="text-sm font-semibold">账户数据权利</h3>
        <p className="mt-1 text-xs leading-5 text-[var(--app-muted)]">
          导出包含当前组织账户范围内的项目、教学会话、任务、学习记录与安全审计摘要。
          仅存在于此浏览器、尚未提交的草稿、离线快照和界面偏好不在服务器 ZIP 中；
          退出或删除账户会清除这些本地数据。
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant="drawer"
          disabled={!available || busy !== null}
          onClick={() => {
            setBusy("export");
            setError("");
            void props.reauthenticate().then(props.exportAccount).catch(() => {
              setError("账户导出失败；没有删除或更改任何服务器数据。");
            }).finally(() => setBusy(null));
          }}
        >
          {busy === "export" ? <LoaderCircle className="size-3.5 animate-spin" /> : <Download className="size-3.5" />}
          导出账户 ZIP
        </Button>
        <Button
          type="button"
          variant="subtle"
          disabled={!available || busy !== null}
          onClick={() => void prepare()}
        >
          {busy === "prepare" ? <LoaderCircle className="size-3.5 animate-spin" /> : <ShieldAlert className="size-3.5" />}
          重新认证并准备永久删除
        </Button>
      </div>

      {challenge && !receipt && (
        <div className="grid gap-3 rounded-lg border border-red-500/40 bg-red-500/5 p-3">
          <p className="text-xs leading-5 text-[var(--app-text-soft)]">
            此操作会阻止新任务、协调正在运行的任务、永久删除本服务控制的账户数据，
            并撤销此账户的所有设备会话。请精确输入下方英文短语：
          </p>
          <code className="select-all break-all text-xs text-red-300">
            {ACCOUNT_DELETION_CONFIRMATION_PHRASE}
          </code>
          <input
            aria-label="永久删除确认短语"
            autoComplete="off"
            spellCheck={false}
            value={typedPhrase}
            onChange={(event) => setTypedPhrase(event.target.value)}
            className="h-9 rounded-md border border-red-500/40 bg-[var(--app-surface)] px-3 text-xs text-[var(--app-text)] outline-none"
          />
          <Button
            type="button"
            disabled={!phraseMatches || challengeExpired || busy !== null}
            onClick={() => void confirm()}
            className="justify-self-start bg-red-600 text-white hover:bg-red-500"
          >
            {busy === "confirm" ? <LoaderCircle className="size-3.5 animate-spin" /> : <Trash2 className="size-3.5" />}
            永久删除账户
          </Button>
          {challengeExpired && (
            <p role="status" className="text-xs text-red-300">确认挑战已过期，请重新认证。</p>
          )}
        </div>
      )}

      {props.status && new Set(["deleting", "retryable_failure"]).has(props.status.status) && !receipt && (
        <div role="status" aria-live="polite" className="grid gap-2 rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-xs leading-5">
          <strong>账户永久删除正在由服务器恢复</strong>
          <p className="text-[var(--app-muted)]">
            当前阶段：{props.status.phase} · 持久化修订 {props.status.revision}。
            页面刷新或 API 重启不会取消已确认的删除。
          </p>
          {props.status.status === "retryable_failure" && (
            <p className="text-amber-300">
              最近一次尝试可安全重试：{props.status.retryable_failure_code}。
            </p>
          )}
          <Button
            type="button"
            variant="subtle"
            disabled={busy !== null}
            onClick={() => void resume()}
            className="justify-self-start"
          >
            {busy === "resume" ? <LoaderCircle className="size-3.5 animate-spin" /> : <ShieldAlert className="size-3.5" />}
            立即恢复删除
          </Button>
        </div>
      )}

      {receipt && (
        <div role="status" className="rounded-lg border border-[var(--app-border-strong)] p-3 text-xs leading-5">
          <strong>本 TeachLab 部署的在线主存储已完成永久删除</strong>
          <p className="mt-1 text-[var(--app-muted)]">
            删除收据 {receipt.receipt_id}。所有设备的本服务会话授权已删除，并保留不可逆范围 tombstone 防止迟到写入。
          </p>
          <p className="mt-1 text-[var(--app-muted)]">
            你自行保存的导出副本，以及此前发送给模型/搜索提供商且受其保留政策约束的远端副本，不在本服务本地删除控制范围内。
          </p>
          <p className="mt-1 text-[var(--app-muted)]">
            组织身份提供方中的账户未被删除；如需删除请联系组织 IdP 管理员。运维备份副本仍须等待保留期到期，或由部署方完成加密擦除。
          </p>
        </div>
      )}

      {error && <p role="alert" className="text-xs leading-5 text-red-300">{error}</p>}
    </section>
  );
}
