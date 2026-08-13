"use client";

import {QueryClient, QueryClientProvider} from "@tanstack/react-query";
import {useEffect, useRef, useState, type ReactNode} from "react";

import {backgroundTaskStatus, fetchBootstrap, recoverUnregisteredOperation, resumeBackgroundTask} from "@/lib/api";
import {
  clearAccountTransitionFence,
  purgeAccountBrowserState,
  reconcileAccountCacheScope,
  reconcileLocalCacheBoundary,
} from "@/lib/account-browser-boundary";
import {
  deleteDurableOperation,
  listDurableOperations,
  markOperationHandoff,
  markOperationRegistered,
  recoveryDecision,
  clearOfflineRuntimeForLogout,
} from "@/lib/offline-runtime";

export function Providers({children}: {children: ReactNode}) {
  const recoveringRef = useRef(false);
  const [recoveryNotice, setRecoveryNotice] = useState("");
  const [cacheBoundary, setCacheBoundary] = useState<"checking" | "allowed" | "blocked">("checking");
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 15_000,
        retry: 1,
        refetchOnWindowFocus: false
      }
    }
  }));

  useEffect(() => {
    const recover = async () => {
      if (!navigator.onLine || recoveringRef.current) return;
      recoveringRef.current = true;
      try {
        // Do not even read account-bound IndexedDB state until the same-origin
        // server has authenticated the current browser session. In particular,
        // an old operator's outbox must never be replayed while a new operator
        // is waiting for the Workbench bootstrap 401 boundary.
        let bootstrap;
        try {
          bootstrap = await fetchBootstrap();
        } catch {
          await reconcileAccountCacheScope({
            authenticated: false,
            markerStorage: window.localStorage,
            purge: () => purgeAccountBrowserState({
              sessionStorage: window.sessionStorage,
              localStorage: window.localStorage,
              clearIndexedDb: clearOfflineRuntimeForLogout,
              clearMemoryCaches: () => queryClient.clear(),
              cacheStorage: typeof window.caches === "undefined" ? undefined : window.caches,
            }),
          });
          setCacheBoundary("blocked");
          return;
        }
        if (bootstrap.account_data_rights?.mode === "local_only_no_account_authority") {
          // Local mode has no organization account scope, but this browser
          // origin may still contain state from an earlier authenticated
          // deployment. Establish a distinct local boundary (purging on the
          // transition) before reading or replaying any IndexedDB operation.
          const localBoundary = await reconcileLocalCacheBoundary({
            markerStorage: window.localStorage,
            purge: () => purgeAccountBrowserState({
              sessionStorage: window.sessionStorage,
              localStorage: window.localStorage,
              clearIndexedDb: clearOfflineRuntimeForLogout,
              clearMemoryCaches: () => queryClient.clear(),
              cacheStorage: typeof window.caches === "undefined" ? undefined : window.caches,
            }),
          });
          if (!localBoundary.allowed) {
            setCacheBoundary("blocked");
            return;
          }
          setCacheBoundary("allowed");
        } else {
          const cacheBoundary = await reconcileAccountCacheScope({
            authenticated: true,
            cacheScope: bootstrap.cache_scope,
            markerStorage: window.localStorage,
            purge: () => purgeAccountBrowserState({
              sessionStorage: window.sessionStorage,
              localStorage: window.localStorage,
              clearIndexedDb: clearOfflineRuntimeForLogout,
              clearMemoryCaches: () => queryClient.clear(),
              cacheStorage: typeof window.caches === "undefined" ? undefined : window.caches,
            }),
          });
          if (!cacheBoundary.allowed || !clearAccountTransitionFence({
            writeCookie: (value) => { document.cookie = value; },
            readCookies: () => document.cookie,
          })) {
            setCacheBoundary("blocked");
            return;
          }
          setCacheBoundary("allowed");
        }
        const operations = await listDurableOperations();
        let resubmitted = 0;
        let resumed = 0;
        let observed = 0;
        let handoff = 0;
        for (const operation of operations) {
          // Only work without a server task identity is eligible for request
          // resubmission, and it reuses the original identity. Registered work
          // is recovered exclusively through task status/resume.
          if (!operation.taskId || operation.state === "unregistered") {
            try {
              await recoverUnregisteredOperation(operation);
              resubmitted += 1;
            } catch {
              // The stream recovery code keeps ambiguous work in the outbox
              // and removes definitive client/server rejections.
            }
            continue;
          }
          try {
            const {task} = await backgroundTaskStatus(operation.taskId);
            const decision = recoveryDecision(operation, task);
            if (decision.kind === "remove_terminal") {
              await deleteDurableOperation(operation.requestId);
            } else if (decision.kind === "resume_registered") {
              const response = await resumeBackgroundTask({task_id: decision.taskId, version: decision.taskVersion});
              await markOperationRegistered(operation.requestId, response.task.task_id, response.task.version);
              resumed += 1;
            } else if (decision.kind === "handoff") {
              await markOperationHandoff(operation.requestId);
              handoff += 1;
            } else {
              await markOperationRegistered(operation.requestId, task.task_id, task.version);
              observed += 1;
            }
          } catch {
            // Keep the durable identity. A later online event may recover it;
            // deleting or replaying it here could duplicate an external effect.
          }
        }
        const remainingUnregistered = (await listDurableOperations())
          .filter((operation) => !operation.taskId || operation.state === "unregistered").length;
        const notices = [
          resumed ? `${resumed} 个已注册后台任务已按 task identity 恢复` : "",
          resubmitted ? `${resubmitted} 个未注册请求已复用原 request identity 提交` : "",
          observed ? `${observed} 个已注册后台任务仍在运行（未重放请求）` : "",
          handoff ? `${handoff} 个任务需要人工接管` : "",
          remainingUnregistered ? `${remainingUnregistered} 个未注册请求仍留在本机` : "",
        ].filter(Boolean);
        if (notices.length) setRecoveryNotice(notices.join("；"));
      } finally {
        recoveringRef.current = false;
      }
    };
    void recover();
    window.addEventListener("online", recover);
    return () => window.removeEventListener("online", recover);
  }, [queryClient]);

  return <QueryClientProvider client={queryClient}>
    {recoveryNotice && (
      <div role="status" aria-live="polite" className="fixed right-4 top-4 z-[100] flex max-w-[min(560px,calc(100vw-2rem))] items-start gap-3 rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-overlay)] px-3 py-2 text-xs leading-5 text-[var(--app-text-soft)] shadow-2xl">
        <span>{recoveryNotice}</span>
        <button type="button" className="shrink-0 text-[var(--app-muted)] hover:text-[var(--app-text)]" onClick={() => setRecoveryNotice("")} aria-label="关闭恢复状态">×</button>
      </div>
    )}
    {cacheBoundary === "allowed" ? children : (
      <main className="grid min-h-screen place-items-center bg-[var(--app-bg)] px-6 text-center">
        <div role="status" className="max-w-md text-sm leading-6 text-[var(--app-muted)]">
          {cacheBoundary === "checking"
            ? "正在核对当前账户的本机缓存边界…"
            : "尚未建立可信账户缓存边界。请完成组织登录或清除被浏览器阻止删除的站点数据后重试。"}
        </div>
      </main>
    )}
  </QueryClientProvider>;
}
