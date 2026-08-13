"use client";

import {BookOpenText, FileText, GraduationCap, NotebookPen, Search, Trash2} from "lucide-react";
import {useCallback, useEffect, useRef, useState, type FormEvent} from "react";

import {browseLearningProject, removeLearningProjectReference, saveLearningProjectNote} from "@/lib/api";
import type {LearningProject, LearningProjectNote, LearningProjectReferenceItem, TeachingResourceSummary} from "@/lib/types";

type Section = "syllabi" | "teaching_sessions" | "resources" | "notes";
type Item = LearningProjectReferenceItem | LearningProjectNote;

const labels: Record<Section, string> = {
  syllabi: "大纲",
  teaching_sessions: "Teach",
  resources: "资源",
  notes: "笔记",
};

function isNote(item: Item): item is LearningProjectNote {
  return "note_id" in item;
}

function referenceSubtitle(section: Section, item: LearningProjectReferenceItem) {
  if (section === "resources") {
    const ownership = item.ownership === "configured_resource_store_unavailable"
      ? `本地资源库不可用 · 图中引用 ${item.project_reference_count ?? 0}`
      : item.ownership === "shared_content_addressed_library"
        ? `共享引用 ${item.project_reference_count ?? 0} 个项目`
        : "当前项目独占引用";
    const metadata = item.metadata;
    const size = typeof metadata?.byte_size === "number" ? `${Math.max(1, Math.ceil(metadata.byte_size / 1024))} KB` : null;
    const pages = typeof metadata?.page_count === "number" ? `${metadata.page_count} 页` : null;
    return [ownership, metadata?.resource_type ?? (item.available === false ? "索引不可用" : "本地资源"), size, pages, metadata?.needs_review ? "需复核" : null].filter(Boolean).join(" · ");
  }
  if (section === "teaching_sessions") {
    const progress = item.lesson_progress;
    return [
      item.status ?? "不可用",
      `${item.rounds_completed ?? 0} 轮`,
      progress ? `${progress.phase_label} ${progress.phase_index + 1}/${progress.phase_count}` : null,
      progress?.current_concept,
    ].filter(Boolean).join(" · ");
  }
  return `${item.module_count ?? 0} 个模块`;
}

export function ProjectLibrary({project, onProjectChange, onOpenSyllabus, onOpenTeachingSession, onOpenResource, onNotice}: {
  project: LearningProject;
  onProjectChange: (project: LearningProject) => void;
  onOpenSyllabus: (id: string) => void;
  onOpenTeachingSession: (id: string) => void;
  onOpenResource: (id: string, metadata?: TeachingResourceSummary) => void;
  onNotice: (message: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [section, setSection] = useState<Section>("syllabi");
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<Item[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState<LearningProjectNote | "new" | null>(null);
  const [noteTitle, setNoteTitle] = useState("");
  const [noteBody, setNoteBody] = useState("");
  const projectIdRef = useRef(project.project_id);
  useEffect(() => {
    projectIdRef.current = project.project_id;
    setItems([]);
    setNextCursor(null);
    setTotal(0);
    setEditing(null);
    setError("");
  }, [project.project_id]);

  const load = useCallback(async (cursor?: string) => {
    setBusy(true);
    setError("");
    try {
      const page = await browseLearningProject(project.project_id, {
        section,
        query: query.trim(),
        ...(cursor ? {cursor} : {}),
        limit: 40,
      });
      if (page.project_id !== projectIdRef.current) return;
      setItems((current) => cursor ? [...current, ...page.items as Item[]] : page.items as Item[]);
      setNextCursor(page.next_cursor);
      setTotal(page.total);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "项目内容读取失败");
    } finally {
      setBusy(false);
    }
  }, [project.project_id, query, section]);

  useEffect(() => {
    if (!open) return;
    const timer = window.setTimeout(() => {void load();}, 180);
    return () => window.clearTimeout(timer);
  }, [load, open, project.updated_at]);

  const submitNote = async (event: FormEvent) => {
    event.preventDefault();
    const title = noteTitle.trim();
    if (!title || busy) return;
    setBusy(true);
    try {
      const updated = await saveLearningProjectNote(project.project_id, {
        operation_id: `note:${crypto.randomUUID()}`,
        expected_updated_at: project.updated_at,
        note: {
          ...(editing && editing !== "new" ? {note_id: editing.note_id} : {}),
          title,
          body: noteBody.trim(),
        },
      });
      onProjectChange(updated);
      setEditing(null);
      setNoteTitle("");
      setNoteBody("");
      onNotice("项目笔记已保存到本地权威存储");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "项目笔记保存失败");
    } finally {
      setBusy(false);
    }
  };

  const unlinkResource = async (item: LearningProjectReferenceItem) => {
    if (busy || !window.confirm("仅解除当前项目的资源引用。共享内容地址库会保留；永久删除项目时只清理图中独占内容。是否继续？")) return;
    setBusy(true);
    try {
      const updated = await removeLearningProjectReference(project.project_id, {
        kind: "resource",
        reference_id: item.reference_id,
        expected_updated_at: project.updated_at,
        operation_id: `reference:${crypto.randomUUID()}`,
      });
      onProjectChange(updated);
      onNotice("已解除当前项目的资源引用；共享库内容未被误删");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "资源引用移除失败");
    } finally {
      setBusy(false);
    }
  };

  const openReference = (item: LearningProjectReferenceItem) => {
    if (section === "syllabi") onOpenSyllabus(item.reference_id);
    else if (section === "teaching_sessions") onOpenTeachingSession(item.reference_id);
    else onOpenResource(item.reference_id, item.metadata);
  };

  const counts = project.syllabus_ids.length + project.teaching_session_ids.length + project.resource_ids.length + project.notes.length;
  return (
    <details open={open} onToggle={(event) => setOpen(event.currentTarget.open)} className="group mx-3 mt-2 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-soft)]">
      <summary className="flex min-h-9 cursor-pointer list-none items-center gap-2 px-3 text-xs text-[var(--app-muted)]">
        <NotebookPen className="size-3.5 text-[var(--app-accent)]" />
        <span>项目内容与笔记</span><span className="ml-auto text-[10px] text-[var(--app-faint)]">{counts}</span>
      </summary>
      <div className="border-t border-[var(--app-border)] p-2">
        <div className="grid grid-cols-4 gap-1" role="tablist" aria-label="项目内容类型">
          {(Object.keys(labels) as Section[]).map((value) => <button key={value} type="button" role="tab" aria-selected={section === value} onClick={() => {setSection(value); setItems([]); setQuery("");}} className={`rounded-md px-1 py-1.5 text-[10px] ${section === value ? "bg-[var(--app-selected)] text-[var(--app-text)]" : "text-[var(--app-faint)] hover:bg-[var(--app-hover)]"}`}>{labels[value]}</button>)}
        </div>
        <label className="mt-2 flex h-8 items-center gap-2 rounded-md border border-[var(--app-border)] bg-[var(--app-surface)] px-2">
          <Search className="size-3 text-[var(--app-faint)]" />
          <input value={query} onChange={(event) => setQuery(event.target.value)} maxLength={200} placeholder={`搜索${labels[section]}`} className="min-w-0 flex-1 bg-transparent text-[11px] text-[var(--app-text)] outline-none" />
        </label>
        {section === "notes" && <button type="button" onClick={() => {setEditing("new"); setNoteTitle(""); setNoteBody("");}} className="mt-2 w-full rounded-md border border-dashed border-[var(--app-border-strong)] px-2 py-1.5 text-[10px] text-[var(--app-accent-text)] hover:bg-[var(--app-hover)]">新建项目笔记</button>}
        {editing && <form onSubmit={submitNote} className="mt-2 grid gap-2 rounded-md border border-[var(--app-border)] bg-[var(--app-surface)] p-2">
          <input autoFocus required maxLength={160} value={noteTitle} onChange={(event) => setNoteTitle(event.target.value)} placeholder="笔记标题" className="h-7 rounded border border-[var(--app-border)] bg-transparent px-2 text-[11px] outline-none" />
          <textarea maxLength={64_000} value={noteBody} onChange={(event) => setNoteBody(event.target.value)} placeholder="可编辑笔记内容" rows={4} className="resize-y rounded border border-[var(--app-border)] bg-transparent p-2 text-[11px] leading-4 outline-none" />
          <div className="flex justify-end gap-1"><button type="button" onClick={() => setEditing(null)} className="rounded px-2 py-1 text-[10px] text-[var(--app-faint)]">取消</button><button type="submit" disabled={busy || !noteTitle.trim()} className="rounded bg-[var(--app-accent)] px-2 py-1 text-[10px] text-[#211714] disabled:opacity-40">保存</button></div>
        </form>}
        {error && <p role="alert" className="mt-2 text-[10px] leading-4 text-[var(--app-danger)]">{error}</p>}
        <div className="mt-2 grid max-h-52 gap-1 overflow-y-auto">
          {items.map((item) => isNote(item) ? (
            <button key={item.note_id} type="button" onClick={() => {setEditing(item); setNoteTitle(item.title); setNoteBody(item.body);}} className="rounded-md px-2 py-2 text-left hover:bg-[var(--app-hover)]"><span className="block truncate text-[11px] text-[var(--app-text-soft)]">{item.title}</span><span className="mt-0.5 line-clamp-2 block text-[9px] leading-4 text-[var(--app-faint)]">{item.body || "空笔记"}</span></button>
          ) : (
            <div key={item.reference_id} className="group/item flex items-start gap-1 rounded-md hover:bg-[var(--app-hover)]">
              <button type="button" disabled={item.available === false} onClick={() => openReference(item)} className="min-w-0 flex-1 px-2 py-2 text-left disabled:opacity-50">
                <span className="flex items-center gap-1.5 truncate text-[11px] text-[var(--app-text-soft)]">{section === "syllabi" ? <BookOpenText className="size-3 shrink-0" /> : section === "teaching_sessions" ? <GraduationCap className="size-3 shrink-0" /> : <FileText className="size-3 shrink-0" />}{item.title ?? item.metadata?.display_name ?? item.reference_id}</span>
                <span className="mt-0.5 block truncate text-[9px] text-[var(--app-faint)]">{referenceSubtitle(section, item)}</span>
              </button>
              {section === "resources" && <button type="button" disabled={busy} onClick={() => {void unlinkResource(item);}} aria-label={`解除资源引用 ${item.metadata?.display_name ?? item.reference_id}`} title="仅解除当前项目引用" className="mr-1 mt-1 grid size-7 place-items-center rounded text-[var(--app-faint)] opacity-0 hover:bg-[var(--app-danger-soft)] hover:text-[var(--app-danger)] group-hover/item:opacity-100 focus:opacity-100 disabled:opacity-30"><Trash2 className="size-3" /></button>}
            </div>
          ))}
          {!busy && !items.length && !error && <p className="py-3 text-center text-[10px] text-[var(--app-faint)]">没有匹配内容</p>}
        </div>
        <div className="mt-2 flex items-center justify-between text-[9px] text-[var(--app-faint)]"><span>已显示 {items.length} / {total}</span>{nextCursor && <button type="button" disabled={busy} onClick={() => {void load(nextCursor);}} className="rounded px-2 py-1 text-[var(--app-accent-text)] hover:bg-[var(--app-hover)] disabled:opacity-40">加载更多</button>}</div>
      </div>
    </details>
  );
}
