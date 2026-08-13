"use client";

import {Archive, ChevronDown, Download, FolderOpen, Pencil, Pin, PinOff, Plus, RotateCcw, Search, ShieldX, Trash2, X} from "lucide-react";
import {useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent as ReactKeyboardEvent} from "react";

import {isRovingMenuKey, nextRovingMenuIndex} from "@/lib/accessibility";
import {cn} from "@/lib/cn";
import type {LearningProjectSummary, LearningProjectTrashItem} from "@/lib/types";

export function ProjectSwitcher({projects, trashedProjects = [], activeProjectId, busy, onSelect, onCreate, onRename, onPin, onArchive, onTrash, onRestore, onExport, onPurge}: {
  projects: LearningProjectSummary[];
  trashedProjects?: LearningProjectTrashItem[];
  activeProjectId?: string | null;
  busy?: boolean;
  onSelect: (projectId: string) => void;
  onCreate: (title: string) => void;
  onRename: (projectId: string, title: string) => void;
  onPin: (projectId: string, pinned: boolean) => void;
  onArchive: (projectId: string, archived: boolean) => void;
  onTrash: (projectId: string) => void;
  onRestore: (project: LearningProjectTrashItem) => void;
  onExport: (project: LearningProjectSummary) => void;
  onPurge: (project: LearningProjectTrashItem) => void;
}) {
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  const [visibleLimit, setVisibleLimit] = useState(40);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const active = projects.find((project) => project.project_id === activeProjectId);
  const filteredProjects = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return projects.filter((project) => !needle || `${project.title}\n${project.description}`.toLocaleLowerCase().includes(needle));
  }, [projects, query]);
  const visibleProjects = filteredProjects.slice(0, visibleLimit);

  useEffect(() => {
    if (!open) return;
    const frame = window.requestAnimationFrame(() => {
      const items = Array.from(menuRef.current?.querySelectorAll<HTMLButtonElement>('[data-roving-menuitem="true"]:not([disabled])') ?? []);
      const selected = items.find((item) => item.getAttribute("aria-current") === "page") ?? items[0];
      items.forEach((item) => { item.tabIndex = item === selected ? 0 : -1; });
      selected?.focus();
    });
    const close = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", close);
    window.addEventListener("keydown", escape);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", escape);
    };
  }, [open]);

  function onMenuKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (!(event.target instanceof HTMLButtonElement)) return;
    const items = Array.from(menuRef.current?.querySelectorAll<HTMLButtonElement>('[data-roving-menuitem="true"]:not([disabled])') ?? []);
    const index = items.indexOf(event.target);
    if (isRovingMenuKey(event.key)) {
      event.preventDefault();
      const next = nextRovingMenuIndex(index, items.length, event.key);
      items.forEach((item, itemIndex) => { item.tabIndex = itemIndex === next ? 0 : -1; });
      items[next]?.focus();
    } else if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
      triggerRef.current?.focus();
    }
  }

  const submitCreate = (event: FormEvent) => {
    event.preventDefault();
    const title = draft.trim();
    if (!title) return;
    onCreate(title);
    setDraft("");
    setCreating(false);
  };

  return (
    <div ref={rootRef} className="relative mx-3 mt-2">
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls="teachlab-project-menu"
        aria-label="选择学习项目"
        onClick={() => setOpen((value) => !value)}
        className="flex min-h-10 w-full items-center gap-2 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-3 text-left text-sm text-[var(--app-text-soft)] hover:bg-[var(--app-hover)]"
      >
        <FolderOpen className="size-4 shrink-0 text-[var(--app-accent)]" />
        <span className="min-w-0 flex-1 truncate">{active?.title ?? "选择学习项目"}</span>
        <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div ref={menuRef} id="teachlab-project-menu" role="menu" aria-label="学习项目" onKeyDown={onMenuKeyDown} className="absolute left-0 right-0 top-[calc(100%+6px)] z-40 max-h-[360px] overflow-y-auto rounded-xl border border-[var(--app-border)] bg-[var(--app-overlay)] p-1.5 shadow-2xl">
          <div className="flex items-center px-2 py-1.5 text-[10px] font-medium uppercase tracking-wider text-[var(--app-faint)]">
            <span>学习项目</span>
            <button type="button" tabIndex={-1} className="ml-auto grid size-6 place-items-center rounded-md hover:bg-[var(--app-hover)]" aria-label="关闭项目列表" onClick={() => {setOpen(false); triggerRef.current?.focus();}}><X className="size-3.5" /></button>
          </div>
          <label className="mb-1 flex h-8 items-center gap-2 rounded-md border border-[var(--app-border)] bg-[var(--app-surface)] px-2"><Search className="size-3" /><input value={query} onChange={(event) => {setQuery(event.target.value); setVisibleLimit(40);}} maxLength={200} placeholder="搜索全部项目" className="min-w-0 flex-1 bg-transparent text-xs font-normal normal-case tracking-normal text-[var(--app-text)] outline-none" /></label>
          <div className="grid gap-1">
            {visibleProjects.map((project) => (
              <div key={project.project_id} className={cn("group grid grid-cols-[minmax(0,1fr)_auto] items-center rounded-lg", activeProjectId === project.project_id && "bg-[var(--app-selected)]")}>
                {editingId === project.project_id ? (
                  <form className="px-2 py-1.5" onSubmit={(event) => {event.preventDefault(); const title = draft.trim(); if (title) onRename(project.project_id, title); setEditingId(null); setDraft("");}}>
                    <input autoFocus maxLength={160} value={draft} onChange={(event) => setDraft(event.target.value)} onBlur={() => {setEditingId(null); setDraft("");}} className="h-7 w-full rounded-md border border-[var(--app-focus-border)] bg-[var(--app-surface)] px-2 text-xs text-[var(--app-text)] outline-none" aria-label="项目名称" />
                  </form>
                ) : (
                  <button type="button" role="menuitem" data-roving-menuitem="true" aria-current={activeProjectId === project.project_id ? "page" : undefined} tabIndex={activeProjectId === project.project_id ? 0 : -1} onClick={() => {onSelect(project.project_id); setOpen(false); window.requestAnimationFrame(() => triggerRef.current?.focus());}} className="min-w-0 px-2.5 py-2 text-left">
                    <span className="flex items-center gap-1.5 text-xs text-[var(--app-text-soft)]">{project.pinned && <Pin className="size-3 shrink-0 text-[var(--app-accent)]" />}<span className="truncate">{project.title}</span></span>
                    <span className="mt-0.5 block truncate text-[9px] text-[var(--app-faint)]">{project.status === "archived" ? "已归档" : `${project.chat_thread_count} 个对话 · ${project.teaching_session_count} 个教学`}</span>
                  </button>
                )}
                <div className="mr-1 flex opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 [@media(hover:none)]:opacity-100">
                  <button type="button" role="menuitem" data-roving-menuitem="true" tabIndex={-1} disabled={busy} onClick={() => {setEditingId(project.project_id); setDraft(project.title);}} className="grid size-7 place-items-center rounded-md text-[var(--app-faint)] hover:bg-[var(--app-hover)] hover:text-[var(--app-text)] disabled:opacity-40" title="重命名" aria-label={`重命名 ${project.title}`}><Pencil className="size-3" /></button>
                  <button type="button" role="menuitem" data-roving-menuitem="true" tabIndex={-1} disabled={busy} onClick={() => onPin(project.project_id, !project.pinned)} className="grid size-7 place-items-center rounded-md text-[var(--app-faint)] hover:bg-[var(--app-hover)] hover:text-[var(--app-text)] disabled:opacity-40" title={project.pinned ? "取消置顶" : "置顶"} aria-label={`${project.pinned ? "取消置顶" : "置顶"} ${project.title}`}>{project.pinned ? <PinOff className="size-3" /> : <Pin className="size-3" />}</button>
                  <button type="button" role="menuitem" data-roving-menuitem="true" tabIndex={-1} disabled={busy} onClick={() => onArchive(project.project_id, project.status !== "archived")} className="grid size-7 place-items-center rounded-md text-[var(--app-faint)] hover:bg-[var(--app-hover)] hover:text-[var(--app-text)] disabled:opacity-40" title={project.status === "archived" ? "取消归档" : "归档"} aria-label={`${project.status === "archived" ? "取消归档" : "归档"} ${project.title}`}><Archive className="size-3" /></button>
                  <button type="button" role="menuitem" data-roving-menuitem="true" tabIndex={-1} disabled={busy} onClick={() => onExport(project)} className="grid size-7 place-items-center rounded-md text-[var(--app-faint)] hover:bg-[var(--app-hover)] hover:text-[var(--app-text)] disabled:opacity-40" title="导出全部私有数据" aria-label={`导出 ${project.title} 的全部私有数据`}><Download className="size-3" /></button>
                  <button type="button" role="menuitem" data-roving-menuitem="true" tabIndex={-1} disabled={busy} onClick={() => onTrash(project.project_id)} className="grid size-7 place-items-center rounded-md text-[var(--app-faint)] hover:bg-[var(--app-danger-soft)] hover:text-[var(--app-danger)] disabled:opacity-40" title="移入废纸篓" aria-label={`移入废纸篓 ${project.title}`}><Trash2 className="size-3" /></button>
                </div>
              </div>
            ))}
            {!filteredProjects.length && <p className="px-3 py-4 text-center text-xs text-[var(--app-muted)]">{projects.length ? "没有匹配项目" : "还没有学习项目"}</p>}
            {visibleLimit < filteredProjects.length && <button type="button" role="menuitem" data-roving-menuitem="true" tabIndex={-1} onClick={() => setVisibleLimit((value) => value + 40)} className="rounded-md px-3 py-2 text-xs text-[var(--app-accent-text)] hover:bg-[var(--app-hover)]">加载更多（{visibleProjects.length}/{filteredProjects.length}）</button>}
          </div>
          {trashedProjects.length > 0 && (
            <div className="mt-1 border-t border-[var(--app-border)] pt-1">
              <p className="px-2 py-1 text-[9px] font-medium uppercase tracking-wider text-[var(--app-faint)]">可恢复的项目</p>
              {trashedProjects.map((project) => (
                <div key={project.project_id} className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-xs text-[var(--app-muted)] hover:bg-[var(--app-hover)]">
                  <Trash2 className="size-3.5 shrink-0" />
                  <span className="min-w-0 flex-1 truncate">{project.title}</span>
                  <button type="button" role="menuitem" data-roving-menuitem="true" tabIndex={-1} disabled={busy} onClick={() => onRestore(project)} className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[10px] text-[var(--app-accent-text)] hover:bg-[var(--app-accent-soft)] disabled:opacity-40" aria-label={`恢复 ${project.title}`}><RotateCcw className="size-3" />恢复</button>
                  <button type="button" role="menuitem" data-roving-menuitem="true" tabIndex={-1} disabled={busy} onClick={(event) => {const target = event.currentTarget; onPurge(project); window.requestAnimationFrame(() => target.focus());}} className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[10px] text-[var(--app-danger)] hover:bg-[var(--app-danger-soft)] disabled:opacity-40" aria-label={`永久删除 ${project.title}`}><ShieldX className="size-3" />永久删除</button>
                </div>
              ))}
            </div>
          )}
          {creating ? (
            <form onSubmit={submitCreate} className="mt-1 border-t border-[var(--app-border)] p-2">
              <input autoFocus maxLength={160} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="项目名称" aria-label="新项目名称" className="h-8 w-full rounded-md border border-[var(--app-focus-border)] bg-[var(--app-surface)] px-2 text-xs text-[var(--app-text)] outline-none" />
              <div className="mt-2 flex justify-end gap-1"><button type="button" onClick={() => {setCreating(false); setDraft("");}} className="rounded-md px-2 py-1 text-[10px] text-[var(--app-muted)] hover:bg-[var(--app-hover)]">取消</button><button type="submit" disabled={!draft.trim() || busy} className="rounded-md bg-[var(--app-accent)] px-2 py-1 text-[10px] text-[#211714] disabled:opacity-40">创建</button></div>
            </form>
          ) : (
            <button type="button" role="menuitem" data-roving-menuitem="true" tabIndex={projects.length ? -1 : 0} disabled={busy} onClick={() => {setCreating(true); setDraft("");}} className="mt-1 flex h-8 w-full items-center gap-2 rounded-lg border-t border-[var(--app-border)] px-2.5 text-xs text-[var(--app-muted)] hover:bg-[var(--app-hover)] hover:text-[var(--app-text)] disabled:opacity-40"><Plus className="size-3.5" />新建学习项目</button>
          )}
        </div>
      )}
    </div>
  );
}
