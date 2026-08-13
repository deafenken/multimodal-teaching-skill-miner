"use client";

import {cn} from "@/lib/cn";

export function Switch({checked, onCheckedChange, label}: {checked: boolean; onCheckedChange: (value: boolean) => void; label: string}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onCheckedChange(!checked)}
      className={cn("relative h-6 w-11 rounded-full border border-transparent transition-colors", checked ? "bg-[var(--app-success)]" : "bg-[var(--app-border-strong)]")}
    >
      <span className={cn("absolute top-0.5 size-5 rounded-full bg-[var(--app-surface-raised)] shadow-sm transition-transform", checked ? "translate-x-5" : "translate-x-0.5")} />
    </button>
  );
}
