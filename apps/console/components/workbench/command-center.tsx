"use client";

import * as Dialog from "@radix-ui/react-dialog";
import {CircleStop, LoaderCircle, Play, RefreshCw, Search, TerminalSquare, X} from "lucide-react";
import {useCallback, useEffect, useMemo, useRef, useState} from "react";

import {Button} from "@/components/ui/button";
import {
  cancelHarnessRun,
  listBackgroundTasks,
  resumeBackgroundTask,
  type BackgroundTask,
} from "@/lib/api";
import {
  backgroundTaskStatusLabel,
  commandCenterActions,
  filterCommandCenterActions,
  relativeTaskTime,
  type CommandCenterActionId,
} from "@/lib/command-center";
import {cn} from "@/lib/cn";

const ACTIVE_TASK_STATES = new Set(["queued", "running", "cancel_requested", "suspended", "handoff"]);

export function CommandCenter({open, onOpenChange, chatRunning, teachRunning, onAction}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  chatRunning: boolean;
  teachRunning: boolean;
  onAction: (action: CommandCenterActionId) => void;
}) {
  const [query, setQuery] = useState("");
  const [tasks, setTasks] = useState<BackgroundTask[]>([]);
  const [tasksVisible, setTasksVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [busyTaskId, setBusyTaskId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const actions = useMemo(
    () => filterCommandCenterActions(commandCenterActions({chatRunning, teachRunning}), query),
    [chatRunning, query, teachRunning],
  );

  useEffect(() => {
    if (open) return;
    const remember = (event?: FocusEvent) => {
      const candidate = event?.target ?? document.activeElement;
      if (candidate instanceof HTMLElement && !candidate.closest('[role="dialog"]')) {
        previousFocusRef.current = candidate;
      }
    };
    remember();
    document.addEventListener("focusin", remember);
    return () => document.removeEventListener("focusin", remember);
  }, [open]);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listBackgroundTasks();
      if (response.content_included !== false || response.tasks.some((task) => task.content_included !== false)) {
        throw new Error("后台任务响应包含了禁止公开的内容");
      }
      setTasks(response.tasks);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "后台任务读取失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) {
      setQuery("");
      setTasksVisible(false);
      setError(null);
      return;
    }
    void refresh();
    const timer = window.setInterval(() => {void refresh();}, 5_000);
    return () => window.clearInterval(timer);
  }, [open, refresh]);

  const runAction = (id: CommandCenterActionId) => {
    if (id === "show_tasks") {
      setTasksVisible(true);
      setQuery("");
      return;
    }
    onOpenChange(false);
    onAction(id);
  };

  const mutateTask = async (task: BackgroundTask, operation: "cancel" | "resume") => {
    setBusyTaskId(task.task_id);
    setError(null);
    try {
      if (operation === "cancel") {
        await cancelHarnessRun({
          requestId: task.run_id,
          runId: task.run_id,
          turnId: task.turn_id,
          taskId: task.task_id,
          taskVersion: task.version,
        }, "task_center_user_requested");
      } else {
        await resumeBackgroundTask(task);
      }
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "后台任务操作失败");
    } finally {
      setBusyTaskId(null);
    }
  };

  const orderedTasks = useMemo(() => [...tasks].sort((left, right) => {
    const activeDifference = Number(ACTIVE_TASK_STATES.has(right.status)) - Number(ACTIVE_TASK_STATES.has(left.status));
    return activeDifference || right.updated_at_utc.localeCompare(left.updated_at_utc);
  }), [tasks]);

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[90] bg-black/45 backdrop-blur-[2px]" />
        <Dialog.Content
          aria-describedby="teachlab-command-center-description"
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            const previous = previousFocusRef.current;
            previousFocusRef.current = null;
            if (previous?.isConnected) previous.focus();
          }}
          className="fixed left-1/2 top-[12vh] z-[91] flex max-h-[76vh] w-[min(680px,calc(100vw-24px))] -translate-x-1/2 flex-col overflow-hidden rounded-2xl border border-[var(--app-border-strong)] bg-[var(--app-overlay)] shadow-2xl outline-none"
        >
          <Dialog.Title className="sr-only">命令与后台任务中心</Dialog.Title>
          <Dialog.Description id="teachlab-command-center-description" className="sr-only">搜索工作区命令，查看并管理不含对话正文的持久后台任务。</Dialog.Description>
          <div className="flex items-center gap-3 border-b border-[var(--app-border)] px-4 py-3">
            {tasksVisible ? <TerminalSquare className="size-4 text-[var(--app-accent)]" /> : <Search className="size-4 text-[var(--app-faint)]" />}
            {tasksVisible ? (
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">后台任务</p>
                <p className="text-[11px] text-[var(--app-muted)]">只显示状态与安全标识，不包含学生或教师正文</p>
              </div>
            ) : (
              <input
                autoFocus
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="搜索命令…"
                aria-label="搜索工作区命令"
                className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-[var(--app-faint)]"
              />
            )}
            {tasksVisible && <Button variant="subtle" size="sm" onClick={() => {void refresh();}} disabled={loading} aria-label="刷新后台任务"><RefreshCw className={cn("size-3.5", loading && "animate-spin")} /></Button>}
            <Dialog.Close asChild><Button variant="subtle" size="icon" aria-label="关闭命令中心"><X className="size-4" /></Button></Dialog.Close>
          </div>

          <div className="min-h-0 overflow-y-auto p-2">
            {error && <p role="alert" className="m-2 rounded-lg border border-[var(--app-danger)]/50 bg-[var(--app-danger-soft)] p-3 text-xs text-[var(--app-danger)]">{error}</p>}
            {!tasksVisible ? (
              <div role="listbox" aria-label="工作区命令" className="grid gap-1">
                {actions.map((action) => (
                  <button
                    key={action.id}
                    type="button"
                    role="option"
                    aria-selected="false"
                    onClick={() => runAction(action.id)}
                    className={cn("rounded-xl px-3 py-2.5 text-left hover:bg-[var(--app-hover)] focus-visible:outline-2 focus-visible:outline-[var(--app-accent)]", action.danger && "text-red-300")}
                  >
                    <span className="block text-sm font-medium">{action.title}</span>
                    <span className="mt-0.5 block text-xs text-[var(--app-muted)]">{action.detail}</span>
                  </button>
                ))}
                {!actions.length && <p className="px-3 py-8 text-center text-sm text-[var(--app-muted)]">没有匹配命令</p>}
              </div>
            ) : (
              <div className="grid gap-2" aria-live="polite">
                <button type="button" onClick={() => setTasksVisible(false)} className="w-fit rounded-lg px-3 py-1.5 text-xs text-[var(--app-accent-text)] hover:bg-[var(--app-hover)]">← 返回命令</button>
                {loading && !orderedTasks.length && <p className="flex items-center justify-center gap-2 py-10 text-sm text-[var(--app-muted)]"><LoaderCircle className="size-4 animate-spin" />正在读取任务…</p>}
                {!loading && !orderedTasks.length && <p className="py-10 text-center text-sm text-[var(--app-muted)]">暂无后台任务</p>}
                {orderedTasks.map((task) => {
                  const busy = busyTaskId === task.task_id;
                  return (
                    <article key={task.task_id} className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] p-3">
                      <div className="flex items-start gap-3">
                        <span className={cn("mt-1 size-2 shrink-0 rounded-full", ACTIVE_TASK_STATES.has(task.status) ? "bg-[var(--app-blue)]" : task.status === "completed" ? "bg-[var(--app-success)]" : "bg-[var(--app-faint)]")} />
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                            <strong className="text-sm">{task.operation === "chat" ? "Chat" : task.operation === "start" ? "Teach · 新会话" : "Teach · 回合"}</strong>
                            <span className="text-xs text-[var(--app-muted)]">{backgroundTaskStatusLabel[task.status] ?? task.status}</span>
                            <span className="text-[10px] text-[var(--app-faint)]">{relativeTaskTime(task.updated_at_utc)}</span>
                          </div>
                          <p className="mt-1 truncate font-mono text-[10px] text-[var(--app-faint)]">{task.task_id}</p>
                          {task.error_code && <p className="mt-1 text-xs text-red-300">{task.error_code}</p>}
                        </div>
                        <div className="flex shrink-0 gap-1">
                          {task.resumable && <Button variant="subtle" size="sm" disabled={busy || task.resume_command_pending} onClick={() => {void mutateTask(task, "resume");}}><Play className="mr-1 size-3" />恢复</Button>}
                          {task.cancelable && <Button variant="subtle" size="sm" disabled={busy || task.cancel_command_pending} onClick={() => {void mutateTask(task, "cancel");}}><CircleStop className="mr-1 size-3" />停止</Button>}
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </div>
          <div className="border-t border-[var(--app-border)] px-4 py-2 text-[10px] text-[var(--app-muted)]">⌘K / Ctrl+K 打开 · Esc 关闭 · 切换页面只会分离连接，显式“停止”才取消任务</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
