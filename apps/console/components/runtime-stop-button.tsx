"use client";

import {useState} from "react";

import {stopConsoleRuntime} from "@/lib/api";

export function RuntimeStopButton() {
  const [state, setState] = useState<"idle" | "stopping" | "failed">("idle");

  async function stop() {
    if (state === "stopping" || !window.confirm("停止 Teaching Agent 后台服务？当前进行中的生成会被安全终止。")) return;
    setState("stopping");
    try {
      await stopConsoleRuntime();
    } catch {
      setState("failed");
    }
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 flex items-center gap-2 rounded-lg border border-[var(--app-border)] bg-[var(--app-panel)] p-2 shadow-lg">
      <span className="sr-only" aria-live="polite">
        {state === "stopping" ? "后台服务正在停止" : state === "failed" ? "后台服务停止请求失败" : ""}
      </span>
      <button
        type="button"
        onClick={stop}
        disabled={state === "stopping"}
        className="rounded-md px-3 py-2 text-xs text-[var(--app-muted)] hover:bg-[var(--app-danger-soft)] hover:text-[var(--app-danger)] disabled:cursor-wait disabled:opacity-60"
        aria-label="停止 Teaching Agent 后台服务"
      >
        {state === "stopping" ? "正在停止…" : state === "failed" ? "重试停止后台服务" : "停止后台服务"}
      </button>
    </div>
  );
}
