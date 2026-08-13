"use client";

import {
  BookOpenText,
  CheckCircle2,
  ChevronRight,
  ChevronsDown,
  ChevronsUp,
  CircleAlert,
  Clock3,
  Download,
  FileJson,
  GraduationCap,
  LoaderCircle,
  Menu,
  Pencil,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  RefreshCcw,
  RotateCcw,
  Save,
  Sparkles,
  ShieldCheck,
} from "lucide-react";
import {useCallback, useEffect, useMemo, useRef, useState, type FormEvent} from "react";

import {Button} from "@/components/ui/button";
import {
  downloadTeachingSyllabus,
  fetchCurriculumBlueprint,
  fetchTeachingSyllabusVersions,
  fetchSyllabusLessonStartPayload,
  generateTeachingSyllabus,
  importTeachingSyllabus,
  isTeachingSyllabusDocument,
  listTeachingSyllabi,
  publishTeachingSyllabusRevision,
  reviseTeachingSyllabus,
  rollbackTeachingSyllabusRevision,
  reviewCurriculum,
  revokeCurriculum,
  sealCurriculum,
} from "@/lib/api";
import {cn} from "@/lib/cn";
import {teacherAuthorityUiState} from "@/lib/teacher-authority";
import type {BootstrapPayload, CurriculumBlueprintResponse, SyllabusEditableDraft, SyllabusGeneratePayload, SyllabusLesson, SyllabusLessonStartPayload, SyllabusModule, SyllabusVersionFamily, TeachingSyllabus} from "@/lib/types";

type SyllabusGenerateInput = Omit<SyllabusGeneratePayload, "remote_consent_id">;

function lessonCount(syllabus: TeachingSyllabus) {
  return syllabus.modules.reduce((total, module) => total + module.lessons.length, 0);
}

function mergeSyllabi(current: TeachingSyllabus[], incoming: TeachingSyllabus[]) {
  const merged = new Map(current.map((item) => [item.syllabus_id, item]));
  for (const item of incoming) merged.set(item.syllabus_id, item);
  return Array.from(merged.values()).sort((left, right) => String(right.updated_at ?? right.created_at ?? "").localeCompare(String(left.updated_at ?? left.created_at ?? "")));
}

function objectiveList(value: string) {
  const objectives = value
    .split(/\n|；|;/)
    .map((item) => item.trim().replace(/^(?:[-*•]\s+|\d+[.)、]\s*)/, "").trim())
    .filter(Boolean);
  return Array.from(new Set(objectives)).slice(0, 12);
}

function editableDraft(syllabus: TeachingSyllabus): SyllabusEditableDraft {
  return {
    title: syllabus.title,
    description: syllabus.description ?? "",
    audience: syllabus.audience ?? "",
    estimated_duration_minutes: syllabus.estimated_duration_minutes ?? syllabus.modules.reduce((total, module) => total + module.lessons.reduce((sum, lesson) => sum + (lesson.duration_minutes ?? 0), 0), 0),
    learning_objectives: [...(syllabus.learning_objectives ?? [])],
    prerequisites: [...(syllabus.prerequisites ?? [])],
    modules: syllabus.modules.map((module) => ({
      title: module.title,
      description: module.description ?? "",
      lessons: module.lessons.map((lesson) => ({
        title: lesson.title,
        objective: lesson.objective ?? "",
        summary: lesson.summary ?? "",
        duration_minutes: lesson.duration_minutes ?? 5,
        knowledge_components: [...(lesson.knowledge_components ?? [])],
        materials: {
          example: lesson.materials?.example ?? "",
          practice: lesson.materials?.practice ?? "",
          transfer_task: lesson.materials?.transfer_task ?? "",
        },
      })),
    })),
  };
}

function operationId(prefix: string) {
  const suffix = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}:${suffix}`.replace(/[^A-Za-z0-9_.:-]/g, "-").slice(0, 160);
}

function structuralChangeLabels(previous: TeachingSyllabus | undefined, current: TeachingSyllabus | undefined) {
  if (!previous || !current) return [];
  return ([
    ["标题", previous.title, current.title],
    ["说明", previous.description, current.description],
    ["学习者", previous.audience, current.audience],
    ["总时长", previous.estimated_duration_minutes, current.estimated_duration_minutes],
    ["学习目标", previous.learning_objectives, current.learning_objectives],
    ["先修要求", previous.prerequisites, current.prerequisites],
    ["模块与课节", previous.modules, current.modules],
  ] as const).flatMap(([label, before, after]) => JSON.stringify(before) === JSON.stringify(after) ? [] : [label]);
}

function syllabusErrorMessage(caught: unknown, fallback: string) {
  const message = caught instanceof Error ? caught.message.trim() : "";
  if (/\((?:502|503|504)\)|failed to fetch|networkerror|backend (?:is )?(?:unavailable|not configured)|capability_url is not configured/i.test(message)) {
    return "教学后端未连接，请重新启动 Console 后重试。";
  }
  return message || fallback;
}

function lessonMaterialRows(lesson: SyllabusLesson): Array<[string, string]> {
  return [
    ["示例", lesson.materials?.example],
    ["引导练习", lesson.materials?.practice],
    ["迁移任务", lesson.materials?.transfer_task],
  ].flatMap(([label, content]) => typeof content === "string" && content.trim() ? [[label, content.trim()] as [string, string]] : []);
}

function SyllabusGenerator({busy, sourceResources, onCancel, onGenerate}: {
  busy: boolean;
  sourceResources: Array<{id: string; name: string}>;
  onCancel: () => void;
  onGenerate: (payload: SyllabusGenerateInput) => Promise<void>;
}) {
  const [topic, setTopic] = useState("");
  const [audience, setAudience] = useState("");
  const [objectives, setObjectives] = useState("");
  const [duration, setDuration] = useState("90");
  const submittingRef = useRef(false);
  const minutes = Number(duration);
  const durationValid = duration.trim() !== "" && Number.isInteger(minutes) && minutes >= 15 && minutes <= 20000;
  const uniqueSourceResources = useMemo(
    () => Array.from(new Map(sourceResources.flatMap((resource) => {
      const id = resource.id.trim();
      return id ? [[id, {...resource, id}] as const] : [];
    })).values()),
    [sourceResources],
  );

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (busy || submittingRef.current || !topic.trim() || !durationValid) return;
    submittingRef.current = true;
    try {
      await onGenerate({
        topic: topic.trim(),
        audience: audience.trim() || undefined,
        objectives: objectiveList(objectives),
        duration_minutes: minutes,
        source_resource_ids: uniqueSourceResources.map((resource) => resource.id),
      });
    } finally {
      submittingRef.current = false;
    }
  };

  return (
    <form onSubmit={submit} className="message-enter mx-auto w-full max-w-[720px] rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-raised)] p-5 shadow-[0_18px_50px_var(--app-shadow)]">
      <div className="flex items-start gap-3">
        <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-[var(--app-accent-soft)] text-[var(--app-accent-text)]"><Sparkles className="size-4" /></span>
        <div>
          <h2 className="text-base font-semibold text-[var(--app-text)]">生成教学大纲</h2>
          <p className="mt-1 text-xs leading-5 text-[var(--app-muted)]">大纲 Skill 会把主题拆成可连续教学的模块与课节，并保存为 JSON。</p>
          {uniqueSourceResources.length > 0 && (
            <p className="mt-1 text-[10px] leading-4 text-[var(--app-accent-text)]">
              将参考已导入的 {uniqueSourceResources.length} 个教学资源：{uniqueSourceResources.map((resource) => resource.name).join("、")}
            </p>
          )}
        </div>
      </div>
      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <label className="grid gap-1.5 text-xs text-[var(--app-muted)] sm:col-span-2">
          <span>教学主题 <span aria-hidden="true" className="text-[var(--app-accent)]">*</span></span>
          <input autoFocus required value={topic} onChange={(event) => setTopic(event.target.value)} disabled={busy} placeholder="例如：机器学习入门" className="h-10 rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 text-sm text-[var(--app-text)] outline-none placeholder:text-[var(--app-faint)] focus:border-[var(--app-accent)]" />
        </label>
        <label className="grid gap-1.5 text-xs text-[var(--app-muted)]">
          <span>学习者</span>
          <input value={audience} onChange={(event) => setAudience(event.target.value)} disabled={busy} placeholder="例如：零基础本科生" className="h-10 rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 text-sm text-[var(--app-text)] outline-none placeholder:text-[var(--app-faint)] focus:border-[var(--app-accent)]" />
        </label>
        <label className="grid gap-1.5 text-xs text-[var(--app-muted)]">
          <span>预计总时长（分钟） <span aria-hidden="true" className="text-[var(--app-accent)]">*</span></span>
          <input type="number" required min={15} max={20000} step={1} value={duration} onChange={(event) => setDuration(event.target.value)} disabled={busy} aria-invalid={duration.trim() !== "" && !durationValid} aria-describedby="syllabus-duration-help" className="h-10 rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 text-sm text-[var(--app-text)] outline-none focus:border-[var(--app-accent)]" />
          <span id="syllabus-duration-help" className="text-[10px] leading-4 text-[var(--app-faint)]">请输入 15–20000 之间的整数。</span>
        </label>
        <label className="grid gap-1.5 text-xs text-[var(--app-muted)] sm:col-span-2">
          <span>学习目标（每行一项）</span>
          <textarea rows={4} value={objectives} onChange={(event) => setObjectives(event.target.value)} disabled={busy} placeholder={"理解核心概念\n能完成一个最小实践\n能迁移到新情境"} className="resize-none rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 py-2 text-sm leading-6 text-[var(--app-text)] outline-none placeholder:text-[var(--app-faint)] focus:border-[var(--app-accent)]" />
        </label>
      </div>
      <div className="mt-5 flex justify-end gap-2">
        <Button type="button" variant="subtle" onClick={onCancel} disabled={busy}>取消</Button>
        <Button type="submit" disabled={busy || !topic.trim() || !durationValid} className="bg-[var(--app-accent)] text-[#211714] hover:bg-[var(--app-accent-hover)]">
          {busy ? <LoaderCircle className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
          {busy ? "请稍候…" : "生成并保存"}
        </Button>
      </div>
    </form>
  );
}

function SyllabusEditor({syllabus, busy, onCancel, onSave}: {
  syllabus: TeachingSyllabus;
  busy: boolean;
  onCancel: () => void;
  onSave: (draft: SyllabusEditableDraft, changeSummary: string) => Promise<void>;
}) {
  const [draft, setDraft] = useState(() => editableDraft(syllabus));
  const [changeSummary, setChangeSummary] = useState("");
  const [localError, setLocalError] = useState("");
  const cloneAndUpdate = (update: (next: SyllabusEditableDraft) => void) => {
    setDraft((current) => {
      const next = JSON.parse(JSON.stringify(current)) as SyllabusEditableDraft;
      update(next);
      return next;
    });
  };
  const move = <T,>(items: T[], index: number, offset: -1 | 1) => {
    const target = index + offset;
    if (target < 0 || target >= items.length) return;
    [items[index], items[target]] = [items[target]!, items[index]!];
  };
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const lessonMinutes = draft.modules.reduce((total, module) => total + module.lessons.reduce((sum, lesson) => sum + lesson.duration_minutes, 0), 0);
    if (!changeSummary.trim()) {
      setLocalError("请填写本次修订说明，以便审计和回滚。");
      return;
    }
    if (!draft.title.trim() || !draft.description.trim() || !draft.audience.trim() || !draft.modules.length || draft.modules.some((module) => !module.title.trim() || !module.description.trim() || !module.lessons.length)) {
      setLocalError("标题、说明、学习者以及每个模块/课节都必须完整。");
      return;
    }
    const tolerance = Math.max(10, Math.round(draft.estimated_duration_minutes * 0.20));
    if (Math.abs(lessonMinutes - draft.estimated_duration_minutes) > tolerance) {
      setLocalError(`课节合计 ${lessonMinutes} 分钟，与总时长 ${draft.estimated_duration_minutes} 分钟相差过大。`);
      return;
    }
    setLocalError("");
    await onSave(draft, changeSummary.trim());
  };

  return (
    <form onSubmit={submit} className="mx-auto w-full max-w-[920px] rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-raised)] p-5 shadow-[0_18px_50px_var(--app-shadow)]">
      <div className="flex items-start gap-3">
        <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-[var(--app-accent-soft)] text-[var(--app-accent-text)]"><Pencil className="size-4" /></span>
        <div className="min-w-0 flex-1"><h2 className="text-base font-semibold">编辑大纲草稿</h2><p className="mt-1 text-xs leading-5 text-[var(--app-muted)]">保存会创建不可变草稿；发布前不会影响新课节。编辑不会授予评分答案权威。</p></div>
      </div>
      {localError && <p role="alert" className="mt-4 rounded-lg bg-[var(--app-danger-soft)] px-3 py-2 text-xs text-[var(--app-danger)]">{localError}</p>}
      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <label className="grid gap-1 text-xs text-[var(--app-muted)]"><span>标题</span><input required value={draft.title} onChange={(event) => cloneAndUpdate((next) => {next.title = event.target.value;})} className="h-10 rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 text-sm text-[var(--app-text)]" /></label>
        <label className="grid gap-1 text-xs text-[var(--app-muted)]"><span>学习者</span><input required value={draft.audience} onChange={(event) => cloneAndUpdate((next) => {next.audience = event.target.value;})} className="h-10 rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 text-sm text-[var(--app-text)]" /></label>
        <label className="grid gap-1 text-xs text-[var(--app-muted)] sm:col-span-2"><span>大纲说明</span><textarea required rows={2} value={draft.description} onChange={(event) => cloneAndUpdate((next) => {next.description = event.target.value;})} className="rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 py-2 text-sm text-[var(--app-text)]" /></label>
        <label className="grid gap-1 text-xs text-[var(--app-muted)]"><span>总时长（分钟）</span><input required min={15} max={20000} type="number" value={draft.estimated_duration_minutes} onChange={(event) => cloneAndUpdate((next) => {next.estimated_duration_minutes = Number(event.target.value);})} className="h-10 rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 text-sm text-[var(--app-text)]" /></label>
        <label className="grid gap-1 text-xs text-[var(--app-muted)]"><span>先修要求（每行一项）</span><textarea rows={2} value={draft.prerequisites.join("\n")} onChange={(event) => cloneAndUpdate((next) => {next.prerequisites = objectiveList(event.target.value);})} className="rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 py-2 text-sm text-[var(--app-text)]" /></label>
        <label className="grid gap-1 text-xs text-[var(--app-muted)] sm:col-span-2"><span>学习目标（每行一项）</span><textarea required rows={3} value={draft.learning_objectives.join("\n")} onChange={(event) => cloneAndUpdate((next) => {next.learning_objectives = objectiveList(event.target.value);})} className="rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 py-2 text-sm text-[var(--app-text)]" /></label>
      </div>
      <div className="mt-5 grid gap-4">
        {draft.modules.map((module, moduleIndex) => (
          <section key={`module-${moduleIndex}`} className="rounded-xl border border-[var(--app-border)] p-4">
            <div className="flex items-center gap-2"><strong className="text-sm">模块 {moduleIndex + 1}</strong><span className="flex-1" /><Button type="button" size="icon" variant="subtle" disabled={moduleIndex === 0} onClick={() => cloneAndUpdate((next) => move(next.modules, moduleIndex, -1))} aria-label={`上移模块 ${moduleIndex + 1}`}><ChevronsUp className="size-3.5" /></Button><Button type="button" size="icon" variant="subtle" disabled={moduleIndex === draft.modules.length - 1} onClick={() => cloneAndUpdate((next) => move(next.modules, moduleIndex, 1))} aria-label={`下移模块 ${moduleIndex + 1}`}><ChevronsDown className="size-3.5" /></Button></div>
            <div className="mt-3 grid gap-2 sm:grid-cols-2"><label className="grid gap-1 text-xs text-[var(--app-muted)]"><span>模块标题</span><input required value={module.title} onChange={(event) => cloneAndUpdate((next) => {next.modules[moduleIndex]!.title = event.target.value;})} className="h-9 rounded border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-2 text-sm" /></label><label className="grid gap-1 text-xs text-[var(--app-muted)]"><span>模块说明</span><input required value={module.description} onChange={(event) => cloneAndUpdate((next) => {next.modules[moduleIndex]!.description = event.target.value;})} className="h-9 rounded border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-2 text-sm" /></label></div>
            <div className="mt-3 grid gap-3">
              {module.lessons.map((lesson, lessonIndex) => (
                <fieldset key={`lesson-${moduleIndex}-${lessonIndex}`} className="rounded-lg bg-[var(--app-surface-soft)] p-3">
                  <legend className="sr-only">模块 {moduleIndex + 1} 课节 {lessonIndex + 1}</legend>
                  <div className="flex items-center gap-2"><span className="text-xs font-medium">课节 {lessonIndex + 1}</span><span className="flex-1" /><Button type="button" size="icon" variant="subtle" disabled={lessonIndex === 0} onClick={() => cloneAndUpdate((next) => move(next.modules[moduleIndex]!.lessons, lessonIndex, -1))} aria-label={`上移课节 ${lessonIndex + 1}`}><ChevronsUp className="size-3.5" /></Button><Button type="button" size="icon" variant="subtle" disabled={lessonIndex === module.lessons.length - 1} onClick={() => cloneAndUpdate((next) => move(next.modules[moduleIndex]!.lessons, lessonIndex, 1))} aria-label={`下移课节 ${lessonIndex + 1}`}><ChevronsDown className="size-3.5" /></Button></div>
                  <div className="mt-2 grid gap-2 sm:grid-cols-2">
                    <label className="grid gap-1 text-[11px] text-[var(--app-muted)]"><span>课节标题</span><input required value={lesson.title} onChange={(event) => cloneAndUpdate((next) => {next.modules[moduleIndex]!.lessons[lessonIndex]!.title = event.target.value;})} className="h-8 rounded border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-2 text-xs" /></label>
                    <label className="grid gap-1 text-[11px] text-[var(--app-muted)]"><span>时长</span><input required min={5} max={480} type="number" value={lesson.duration_minutes} onChange={(event) => cloneAndUpdate((next) => {next.modules[moduleIndex]!.lessons[lessonIndex]!.duration_minutes = Number(event.target.value);})} className="h-8 rounded border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-2 text-xs" /></label>
                    {["objective", "summary"].map((field) => <label key={field} className="grid gap-1 text-[11px] text-[var(--app-muted)] sm:col-span-2"><span>{field === "objective" ? "可测目标" : "概要"}</span><textarea required rows={2} value={lesson[field as "objective" | "summary"]} onChange={(event) => cloneAndUpdate((next) => {next.modules[moduleIndex]!.lessons[lessonIndex]![field as "objective" | "summary"] = event.target.value;})} className="rounded border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-2 py-1 text-xs" /></label>)}
                    <label className="grid gap-1 text-[11px] text-[var(--app-muted)] sm:col-span-2"><span>知识组件（每行一项）</span><textarea required rows={2} value={lesson.knowledge_components.join("\n")} onChange={(event) => cloneAndUpdate((next) => {next.modules[moduleIndex]!.lessons[lessonIndex]!.knowledge_components = objectiveList(event.target.value);})} className="rounded border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-2 py-1 text-xs" /></label>
                    {(["example", "practice", "transfer_task"] as const).map((field) => <label key={field} className="grid gap-1 text-[11px] text-[var(--app-muted)] sm:col-span-2"><span>{{example: "示例", practice: "练习", transfer_task: "迁移任务"}[field]}</span><textarea required rows={2} value={lesson.materials[field]} onChange={(event) => cloneAndUpdate((next) => {next.modules[moduleIndex]!.lessons[lessonIndex]!.materials[field] = event.target.value;})} className="rounded border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-2 py-1 text-xs" /></label>)}
                  </div>
                </fieldset>
              ))}
            </div>
          </section>
        ))}
      </div>
      <label className="mt-5 grid gap-1 text-xs text-[var(--app-muted)]"><span>修订说明</span><input required maxLength={500} value={changeSummary} onChange={(event) => setChangeSummary(event.target.value)} placeholder="例如：调整先修顺序并细化迁移任务" className="h-10 rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 text-sm" /></label>
      <div className="mt-5 flex justify-end gap-2"><Button type="button" variant="subtle" onClick={onCancel} disabled={busy}>取消</Button><Button type="submit" disabled={busy}><Save className="size-4" />{busy ? "正在保存…" : "保存为草稿版本"}</Button></div>
    </form>
  );
}

function ModuleTree({module, syllabus, expanded, startingLessonId, onToggle, onStartLesson}: {
  module: SyllabusModule;
  syllabus: TeachingSyllabus;
  expanded: boolean;
  startingLessonId: string | null;
  onToggle: () => void;
  onStartLesson: (syllabus: TeachingSyllabus, module: SyllabusModule, lesson: SyllabusLesson) => Promise<void>;
}) {
  const regionId = `syllabus-module-${syllabus.syllabus_id}-${module.module_id}`.replace(/[^a-zA-Z0-9_-]/g, "-");
  return (
    <section className="overflow-hidden rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-raised)]">
      <button type="button" onClick={onToggle} aria-expanded={expanded} aria-controls={regionId} className="flex min-h-14 w-full items-center gap-3 px-4 text-left transition hover:bg-[var(--app-hover)]">
        <ChevronRight aria-hidden="true" className={cn("size-4 shrink-0 text-[var(--app-faint)] transition-transform", expanded && "rotate-90")} />
        <span className="min-w-0 flex-1">
          <strong className="block truncate text-sm font-medium text-[var(--app-text)]">{module.title}</strong>
          {module.description && <span className="mt-0.5 line-clamp-1 block text-xs text-[var(--app-muted)]">{module.description}</span>}
        </span>
        <span className="shrink-0 text-[10px] tabular-nums text-[var(--app-faint)]">{module.lessons.length} 课节</span>
      </button>
      {expanded && (
        <div id={regionId} className="border-t border-[var(--app-border)]">
          {module.lessons.length ? module.lessons.map((lesson, index) => {
            const starting = startingLessonId === lesson.lesson_id;
            const materials = lessonMaterialRows(lesson);
            return (
              <div key={lesson.lesson_id} className="group grid grid-cols-[28px_minmax(0,1fr)_auto] gap-3 border-b border-[var(--app-border)] px-4 py-4 last:border-b-0 max-[620px]:grid-cols-[24px_minmax(0,1fr)]">
                <span className="grid size-7 place-items-center rounded-full bg-[var(--app-surface)] text-[10px] tabular-nums text-[var(--app-muted)]">{index + 1}</span>
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-sm font-medium text-[var(--app-text)]">{lesson.title}</h3>
                    {lesson.duration_minutes ? <span className="inline-flex items-center gap-1 text-[10px] text-[var(--app-faint)]"><Clock3 className="size-3" />{lesson.duration_minutes} 分钟</span> : null}
                  </div>
                  {lesson.objective && <p className="mt-1 text-xs leading-5 text-[var(--app-muted)]">{lesson.objective}</p>}
                  {lesson.summary && (
                    <div className="mt-2 text-xs leading-5 text-[var(--app-muted)]">
                      <span className="font-medium text-[var(--app-text-soft)]">课节概要</span>
                      <p className="mt-0.5">{lesson.summary}</p>
                    </div>
                  )}
                  {lesson.knowledge_components && lesson.knowledge_components.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {lesson.knowledge_components.slice(0, 6).map((item) => <span key={item} className="rounded bg-[var(--app-surface)] px-2 py-1 text-[10px] text-[var(--app-muted)]">{item}</span>)}
                    </div>
                  )}
                  {materials.length > 0 && (
                    <div className="mt-3 grid gap-2 sm:grid-cols-3">
                      {materials.map(([label, content]) => (
                        <div key={label} className="rounded-lg bg-[var(--app-surface)] px-3 py-2.5">
                          <span className="text-[10px] font-medium text-[var(--app-text-soft)]">{label}</span>
                          <p className="mt-1 text-[11px] leading-5 text-[var(--app-muted)]">{content}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <Button type="button" size="sm" variant="subtle" disabled={Boolean(startingLessonId)} onClick={() => void onStartLesson(syllabus, module, lesson)} className="self-center max-[620px]:col-start-2 max-[620px]:justify-self-start" aria-label={`开始教学：${lesson.title}`}>
                  {starting ? <LoaderCircle className="size-3.5 animate-spin" /> : <GraduationCap className="size-3.5" />}
                  {starting ? "正在开始…" : "开始教学"}
                </Button>
              </div>
            );
          }) : <p className="px-12 py-5 text-xs text-[var(--app-muted)]">这个模块还没有课节。</p>}
        </div>
      )}
    </section>
  );
}

export function SyllabusWorkspace({sidebarOpen, onToggleSidebar, inspectorOpen, onToggleInspector, initialSyllabus, initialError, sourceResources = [], teacherAuthority, onSyllabusChange, onSyllabusSaved, onBeforeRemoteGeneration, onStartLesson}: {
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  inspectorOpen: boolean;
  onToggleInspector: () => void;
  initialSyllabus?: TeachingSyllabus | null;
  initialError?: string;
  sourceResources?: Array<{id: string; name: string}>;
  teacherAuthority?: BootstrapPayload["teacher_authority"];
  onSyllabusChange?: (syllabus: TeachingSyllabus) => void;
  onSyllabusSaved?: (syllabus: TeachingSyllabus) => void;
  onBeforeRemoteGeneration?: () => Promise<string | null>;
  onStartLesson: (syllabus: TeachingSyllabus, module: SyllabusModule, lesson: SyllabusLesson, startPayload: SyllabusLessonStartPayload) => Promise<void>;
}) {
  const [syllabi, setSyllabi] = useState<TeachingSyllabus[]>(initialSyllabus ? [initialSyllabus] : []);
  const [selectedId, setSelectedId] = useState(initialSyllabus?.syllabus_id ?? "");
  const [expandedModules, setExpandedModules] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [generatorOpen, setGeneratorOpen] = useState(false);
  const [editorOpen, setEditorOpen] = useState(false);
  const [versionOpen, setVersionOpen] = useState(false);
  const [versionFamily, setVersionFamily] = useState<SyllabusVersionFamily | null>(null);
  const [versionDocuments, setVersionDocuments] = useState<TeachingSyllabus[]>([]);
  const [curriculum, setCurriculum] = useState<CurriculumBlueprintResponse | null>(null);
  const [curriculumOpen, setCurriculumOpen] = useState(false);
  const [curriculumDraft, setCurriculumDraft] = useState("");
  const [curriculumBusy, setCurriculumBusy] = useState<"review" | "seal" | "revoke" | null>(null);
  const [activity, setActivity] = useState<"generating" | "importing" | "downloading" | "editing" | "publishing" | "rolling-back" | null>(null);
  const [startingLessonId, setStartingLessonId] = useState<string | null>(null);
  const [error, setError] = useState(initialError ?? "");
  const importInputRef = useRef<HTMLInputElement>(null);
  const activityGuardRef = useRef(false);
  const curriculumTeacherAuthorized = teacherAuthorityUiState(
    teacherAuthority,
  ).authenticatedTeacher;

  const selected = useMemo(() => syllabi.find((item) => item.syllabus_id === selectedId) ?? syllabi[0] ?? null, [selectedId, syllabi]);

  const refreshVersions = useCallback(async (syllabusId: string) => {
    const result = await fetchTeachingSyllabusVersions(syllabusId);
    setVersionFamily(result.versionFamily);
    setVersionDocuments(result.syllabi);
    return result;
  }, []);

  const refreshCurriculum = useCallback(async (syllabusId: string) => {
    const result = await fetchCurriculumBlueprint(syllabusId);
    setCurriculum(result);
    const authoritySpec = result.curriculum_authority?.review?.teacher_spec;
    const blueprint = result.curriculum_blueprint;
    const teacherSpec = authoritySpec ?? Object.fromEntries(
      Object.entries(blueprint).filter(([key]) => ![
        "schema", "curriculum_id", "origin", "authority", "integrity",
      ].includes(key))
    );
    setCurriculumDraft(JSON.stringify(teacherSpec, null, 2));
    return result;
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const result = await listTeachingSyllabi();
      setSyllabi((current) => mergeSyllabi(current, result));
      setSelectedId((current) => current || result[0]?.syllabus_id || "");
    } catch (caught) {
      setError(syllabusErrorMessage(caught, "无法读取教学大纲"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (!initialSyllabus) return;
    setSyllabi((current) => current.some((item) => item.syllabus_id === initialSyllabus.syllabus_id)
      ? current
      : mergeSyllabi(current, [initialSyllabus]));
    setSelectedId((current) => current === initialSyllabus.syllabus_id ? current : initialSyllabus.syllabus_id);
  }, [initialSyllabus]);
  useEffect(() => {
    if (initialError) setError(initialError);
  }, [initialError]);
  useEffect(() => {
    if (!selected?.modules[0]) return;
    setExpandedModules(new Set([selected.modules[0].module_id]));
  }, [selected?.syllabus_id]);
  useEffect(() => {
    if (!selected?.syllabus_id) {
      setVersionFamily(null);
      setVersionDocuments([]);
      setCurriculum(null);
      return;
    }
    let active = true;
    void Promise.all([
      refreshVersions(selected.syllabus_id),
      refreshCurriculum(selected.syllabus_id),
    ]).catch((caught) => {
      if (active) setError(syllabusErrorMessage(caught, "无法读取大纲版本或课程蓝图"));
    });
    return () => {active = false;};
  }, [refreshCurriculum, refreshVersions, selected?.syllabus_id]);
  const beginActivity = (next: "generating" | "importing" | "downloading" | "editing" | "publishing" | "rolling-back") => {
    if (activityGuardRef.current) return false;
    activityGuardRef.current = true;
    setActivity(next);
    return true;
  };

  const finishActivity = () => {
    activityGuardRef.current = false;
    setActivity(null);
  };

  const upsert = (syllabus: TeachingSyllabus) => {
    setSyllabi((current) => mergeSyllabi(current, [syllabus]));
    setSelectedId(syllabus.syllabus_id);
    setExpandedModules(new Set(syllabus.modules[0] ? [syllabus.modules[0].module_id] : []));
    onSyllabusChange?.(syllabus);
    onSyllabusSaved?.(syllabus);
  };

  const installPublishedRevision = (syllabus: TeachingSyllabus, family: SyllabusVersionFamily) => {
    const familyIds = new Set(family.revisions.map((revision) => revision.syllabus_id));
    setSyllabi((current) => mergeSyllabi(current.filter((item) => !familyIds.has(item.syllabus_id)), [syllabus]));
    setSelectedId(syllabus.syllabus_id);
    setExpandedModules(new Set(syllabus.modules[0] ? [syllabus.modules[0].module_id] : []));
    setVersionFamily(family);
    onSyllabusChange?.(syllabus);
    onSyllabusSaved?.(syllabus);
  };

  const selectSyllabus = (syllabus: TeachingSyllabus) => {
    setSelectedId((current) => current === syllabus.syllabus_id ? current : syllabus.syllabus_id);
    setGeneratorOpen(false);
    setEditorOpen(false);
    onSyllabusChange?.(syllabus);
  };

  const saveRevision = async (draft: SyllabusEditableDraft, changeSummary: string) => {
    if (!selected || !versionFamily || !beginActivity("editing")) return;
    setError("");
    try {
      const result = await reviseTeachingSyllabus(selected.syllabus_id, {
        editable_draft: draft,
        change_summary: changeSummary,
        expected_version: versionFamily.version,
        idempotency_key: operationId("syllabus-revision"),
      });
      setVersionFamily(result.versionFamily);
      await refreshVersions(selected.syllabus_id);
      setEditorOpen(false);
      setVersionOpen(true);
    } catch (caught) {
      setError(syllabusErrorMessage(caught, "无法保存大纲草稿版本"));
    } finally {
      finishActivity();
    }
  };

  const pointVersion = async (revisionId: string, rollback: boolean) => {
    if (!selected || !versionFamily || !beginActivity(rollback ? "rolling-back" : "publishing")) return;
    setError("");
    try {
      const payload = {
        revision_id: revisionId,
        expected_version: versionFamily.version,
        idempotency_key: operationId(rollback ? "syllabus-rollback" : "syllabus-publish"),
      };
      const result = rollback
        ? await rollbackTeachingSyllabusRevision(selected.syllabus_id, payload)
        : await publishTeachingSyllabusRevision(selected.syllabus_id, payload);
      installPublishedRevision(result.syllabus, result.versionFamily);
      await refreshVersions(result.syllabus.syllabus_id);
    } catch (caught) {
      setError(syllabusErrorMessage(caught, rollback ? "大纲回滚失败" : "大纲发布失败"));
    } finally {
      finishActivity();
    }
  };

  const generate = async (payload: SyllabusGenerateInput) => {
    if (!beginActivity("generating")) return;
    setError("");
    try {
      const consentId = await onBeforeRemoteGeneration?.();
      if (!consentId) throw new Error("未授权远程模型处理，大纲生成请求没有发送");
      upsert(await generateTeachingSyllabus({...payload, remote_consent_id: consentId}));
      setGeneratorOpen(false);
    } catch (caught) {
      setError(syllabusErrorMessage(caught, "教学大纲生成失败"));
    } finally {
      finishActivity();
    }
  };

  const importJson = async (file: File) => {
    if (!beginActivity("importing")) {
      if (importInputRef.current) importInputRef.current.value = "";
      return;
    }
    setError("");
    try {
      if (file.size > 384 * 1024) throw new Error("教学大纲 JSON 上传不能超过 384 KB；内容上限由后端按 256 KB 校验");
      const parsed = JSON.parse(await file.text()) as unknown;
      if (!isTeachingSyllabusDocument(parsed)) throw new Error("这不是 teaching_syllabus.v1 教学大纲文件");
      upsert(await importTeachingSyllabus(parsed));
      setGeneratorOpen(false);
    } catch (caught) {
      setError(syllabusErrorMessage(caught, "教学大纲导入失败"));
    } finally {
      finishActivity();
      if (importInputRef.current) importInputRef.current.value = "";
    }
  };

  const download = async () => {
    if (!selected) return;
    if (!beginActivity("downloading")) return;
    setError("");
    try {
      const {blob, fileName} = await downloadTeachingSyllabus(selected.syllabus_id);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = fileName;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (caught) {
      setError(syllabusErrorMessage(caught, "教学大纲下载失败"));
    } finally {
      finishActivity();
    }
  };

  const startLesson = async (syllabus: TeachingSyllabus, module: SyllabusModule, lesson: SyllabusLesson) => {
    setStartingLessonId(lesson.lesson_id);
    setError("");
    try {
      const startPayload = await fetchSyllabusLessonStartPayload(syllabus.syllabus_id, lesson.lesson_id);
      await onStartLesson(syllabus, module, lesson, startPayload);
    } catch (caught) {
      setError(syllabusErrorMessage(caught, "无法开始这个课节"));
      setStartingLessonId(null);
    }
  };

  const curriculumAuthorityVersion = curriculum?.curriculum_authority?.version ?? 0;
  const curriculumReview = async () => {
    if (!selected || !versionFamily || curriculumBusy) return;
    if (!curriculumTeacherAuthorized) {
      setError("只有当前服务端确认的认证教师权限才能复核课程测量蓝图。");
      return;
    }
    setCurriculumBusy("review");
    setError("");
    try {
      const parsed = JSON.parse(curriculumDraft) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("课程蓝图必须是一个 JSON 对象");
      }
      await reviewCurriculum({
        syllabus_id: selected.syllabus_id,
        teacher_spec: parsed as Record<string, unknown>,
        expected_syllabus_version: versionFamily.version,
        expected_authority_version: curriculumAuthorityVersion,
        curriculum_authority_idempotency_key: operationId("curriculum-review"),
      });
      await refreshCurriculum(selected.syllabus_id);
    } catch (caught) {
      setError(syllabusErrorMessage(caught, "课程蓝图复核失败"));
    } finally {
      setCurriculumBusy(null);
    }
  };

  const curriculumSeal = async () => {
    const reviewId = curriculum?.curriculum_authority?.review?.review_id;
    if (!selected || !versionFamily || !reviewId || curriculumBusy) return;
    if (!curriculumTeacherAuthorized) {
      setError("只有当前服务端确认的认证教师权限才能密封评分权威。");
      return;
    }
    if (!window.confirm("确认由当前教师身份密封该课程图、目标、KC、测量题型与量规，并允许运行时按其评分？")) return;
    setCurriculumBusy("seal");
    setError("");
    try {
      await sealCurriculum({
        syllabus_id: selected.syllabus_id,
        review_id: reviewId,
        teacher_confirmed_authority: true,
        expected_syllabus_version: versionFamily.version,
        expected_authority_version: curriculumAuthorityVersion,
        curriculum_authority_idempotency_key: operationId("curriculum-seal"),
      });
      await refreshCurriculum(selected.syllabus_id);
    } catch (caught) {
      setError(syllabusErrorMessage(caught, "课程蓝图密封失败"));
    } finally {
      setCurriculumBusy(null);
    }
  };

  const curriculumRevoke = async () => {
    const curriculumId = curriculum?.curriculum_authority?.seal?.curriculum_id;
    if (!selected || !versionFamily || !curriculumId || curriculumBusy) return;
    if (!curriculumTeacherAuthorized) {
      setError("只有当前服务端确认的认证教师权限才能撤销评分权威。");
      return;
    }
    if (!window.confirm("撤销后，正在进行和新开始的课节都不能再使用这份蓝图评分。确定撤销？")) return;
    setCurriculumBusy("revoke");
    setError("");
    try {
      await revokeCurriculum({
        syllabus_id: selected.syllabus_id,
        curriculum_id: curriculumId,
        reason_code: "teacher_revoked",
        expected_syllabus_version: versionFamily.version,
        expected_authority_version: curriculumAuthorityVersion,
        curriculum_authority_idempotency_key: operationId("curriculum-revoke"),
      });
      await refreshCurriculum(selected.syllabus_id);
    } catch (caught) {
      setError(syllabusErrorMessage(caught, "课程蓝图撤销失败"));
    } finally {
      setCurriculumBusy(null);
    }
  };

  return (
    <main className="relative flex min-w-0 flex-1 flex-col bg-[var(--app-bg)] text-[var(--app-text)]" data-screen-label="教学大纲">
      <header className="teachlab-main-header flex h-12 shrink-0 items-center gap-3 border-b border-[var(--app-border)] px-8 max-[680px]:px-4">
        <Button variant="subtle" size="icon" className="min-[901px]:hidden" onClick={onToggleSidebar} aria-label={sidebarOpen ? "关闭会话栏" : "打开会话栏"} aria-pressed={sidebarOpen}><Menu className="size-4" /></Button>
        <strong className="shrink-0 text-base">教学大纲</strong>
        <span className="text-[var(--app-faint)]">/</span>
        <span className="min-w-0 truncate text-sm text-[var(--app-muted)]">生成、导入并按课节开始教学</span>
        <Button variant="subtle" size="icon" className="ml-auto shrink-0" onClick={onToggleInspector} aria-label={inspectorOpen ? "折叠右侧检查器" : "展开右侧检查器"} aria-pressed={inspectorOpen}>{inspectorOpen ? <PanelRightClose className="size-4" /> : <PanelRightOpen className="size-4" />}</Button>
      </header>

      <div className="flex min-h-0 flex-1 max-[780px]:flex-col">
        <aside aria-label="大纲列表" className="w-56 shrink-0 overflow-y-auto border-r border-[var(--app-border)] bg-[var(--app-surface-soft)] p-3 max-[780px]:h-auto max-[780px]:w-full max-[780px]:overflow-x-auto max-[780px]:border-b max-[780px]:border-r-0">
          <div className="flex items-center gap-1">
            <Button size="sm" className="flex-1 bg-[var(--app-accent)] text-[#211714] hover:bg-[var(--app-accent-hover)]" onClick={() => setGeneratorOpen(true)} disabled={Boolean(activity)}><Plus className="size-3.5" />生成大纲</Button>
            <input ref={importInputRef} type="file" accept=".json,application/json" className="hidden" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importJson(file); }} />
            <Button variant="subtle" size="icon" onClick={() => importInputRef.current?.click()} disabled={Boolean(activity)} aria-label="导入教学大纲 JSON" title="导入 teaching_syllabus.v1 JSON">{activity === "importing" ? <LoaderCircle className="size-4 animate-spin" /> : <FileJson className="size-4" />}</Button>
          </div>
          <div className="mt-3 grid gap-1 max-[780px]:flex max-[780px]:min-w-max">
            {syllabi.map((item) => (
              <button key={item.syllabus_id} type="button" disabled={Boolean(activity)} aria-pressed={selected?.syllabus_id === item.syllabus_id && !generatorOpen} onClick={() => selectSyllabus(item)} className={cn("rounded-lg px-3 py-2.5 text-left transition hover:bg-[var(--app-hover)] disabled:cursor-not-allowed disabled:opacity-60 max-[780px]:w-48", selected?.syllabus_id === item.syllabus_id && !generatorOpen && "bg-[var(--app-selected)]")}>
                <strong className="block truncate text-xs font-medium text-[var(--app-text-soft)]">{item.title}</strong>
                <span className="mt-1 block text-[10px] text-[var(--app-faint)]">{item.modules.length} 模块 · {lessonCount(item)} 课节{item.estimated_duration_minutes ? ` · ${item.estimated_duration_minutes} 分钟` : ""}</span>
              </button>
            ))}
          </div>
        </aside>

        <div className="min-h-0 flex-1 overflow-y-auto px-8 py-7 max-[680px]:px-4">
          {error && <div role="alert" className="mx-auto mb-4 flex max-w-[900px] items-start gap-2 rounded-lg border border-[var(--app-danger)] bg-[var(--app-danger-soft)] px-3 py-2 text-xs leading-5 text-[var(--app-danger)]"><CircleAlert className="mt-0.5 size-3.5 shrink-0" /><span className="min-w-0 flex-1">{error}</span><button type="button" onClick={() => setError("")} aria-label="关闭错误提示">×</button></div>}
          {generatorOpen ? <SyllabusGenerator busy={Boolean(activity)} sourceResources={sourceResources} onCancel={() => setGeneratorOpen(false)} onGenerate={generate} /> : editorOpen && selected ? (
            <SyllabusEditor syllabus={selected} busy={activity === "editing"} onCancel={() => setEditorOpen(false)} onSave={saveRevision} />
          ) : loading && !selected ? (
            <div role="status" className="grid min-h-[55vh] place-items-center text-sm text-[var(--app-muted)]"><span className="inline-flex items-center gap-2"><LoaderCircle className="size-4 animate-spin text-[var(--app-accent)]" />正在读取教学大纲…</span></div>
          ) : selected ? (
            <div className="mx-auto w-full max-w-[900px]">
              <div className="flex items-start gap-4">
                <span className="grid size-10 shrink-0 place-items-center rounded-lg bg-[var(--app-accent-soft)] text-[var(--app-accent-text)]"><BookOpenText className="size-5" /></span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h1 className="text-xl font-semibold tracking-tight text-[var(--app-text)]">{selected.title}</h1>
                    {selected.status === "ready" && <span className="inline-flex items-center gap-1 rounded bg-[var(--app-success-soft)] px-2 py-1 text-[10px] text-[var(--app-success)]"><CheckCircle2 className="size-3" />已就绪</span>}
                  </div>
                  {selected.description && <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--app-muted)]">{selected.description}</p>}
                  <p className="mt-2 text-xs text-[var(--app-faint)]">{selected.modules.length} 个模块 · {lessonCount(selected)} 个课节{selected.estimated_duration_minutes ? ` · ${selected.estimated_duration_minutes} 分钟` : ""}{selected.audience ? ` · ${selected.audience}` : ""}</p>
                </div>
                <div className="flex shrink-0 flex-wrap justify-end gap-1.5">
                  <Button variant="subtle" size="sm" disabled={Boolean(activity)} aria-expanded={curriculumOpen} onClick={() => setCurriculumOpen((current) => !current)}><ShieldCheck className="size-3.5" />测量蓝图</Button>
                  <Button variant="subtle" size="sm" disabled={Boolean(activity) || !versionFamily} onClick={() => setEditorOpen(true)}><Pencil className="size-3.5" />编辑</Button>
                  <Button variant="subtle" size="sm" disabled={Boolean(activity) || !versionFamily} aria-expanded={versionOpen} onClick={() => setVersionOpen((current) => !current)}><RotateCcw className="size-3.5" />版本</Button>
                  <Button variant="subtle" size="sm" disabled={Boolean(activity)} onClick={() => void download()}>{activity === "downloading" ? <LoaderCircle className="size-3.5 animate-spin" /> : <Download className="size-3.5" />}下载 JSON</Button>
                </div>
              </div>
              {curriculumOpen && curriculum && (
                <section aria-label="教师课程测量蓝图" className="mt-5 rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-raised)] p-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="text-sm font-medium">课程图、目标、KC 与测量量规</h2>
                    <span className={cn("rounded px-2 py-1 text-[10px]", curriculum.authoritative_for_runtime_grading ? "bg-[var(--app-success-soft)] text-[var(--app-success)]" : "bg-[var(--app-surface)] text-[var(--app-muted)]")}>{curriculum.authoritative_for_runtime_grading ? "已密封，可用于评分" : curriculum.review_status === "reviewed" ? "已复核，待明确密封" : curriculum.review_status === "revoked" ? "已撤销" : curriculum.review_status === "stale" ? "大纲版本已变更，需重新复核" : "未复核，不可用于评分"}</span>
                    <span className="text-[10px] text-[var(--app-faint)]">authority v{curriculumAuthorityVersion}</span>
                  </div>
                  <p className="mt-2 text-[11px] leading-5 text-[var(--app-muted)]">JSON 包含先修图、可测目标、知识组件、事实声明、题目蓝图、补救分支、延迟复习和量规。只有经当前教师权限复核并显式密封的精确版本才进入运行时评分。</p>
                  {!curriculumTeacherAuthorized && <p role="status" className="mt-2 text-[11px] leading-5 text-[var(--app-danger)]">当前会话没有服务端确认的认证教师权限；蓝图仅供查看，不能复核、密封或撤销。</p>}
                  <textarea aria-label="课程测量蓝图 JSON" spellCheck={false} value={curriculumDraft} onChange={(event) => setCurriculumDraft(event.target.value)} disabled={Boolean(curriculumBusy) || curriculum.review_status === "sealed"} rows={14} className="mt-3 w-full resize-y rounded-lg border border-[var(--app-border-strong)] bg-[var(--app-surface)] px-3 py-2 font-mono text-[11px] leading-5 text-[var(--app-text-soft)] outline-none focus:border-[var(--app-accent)] disabled:opacity-70" />
                  <div className="mt-3 flex flex-wrap justify-end gap-2">
                    <Button variant="subtle" size="sm" onClick={() => void refreshCurriculum(selected.syllabus_id)} disabled={Boolean(curriculumBusy)}><RefreshCcw className="size-3.5" />重新读取</Button>
                    {curriculum.review_status !== "sealed" && <Button size="sm" onClick={() => void curriculumReview()} disabled={Boolean(curriculumBusy) || !versionFamily || !curriculumTeacherAuthorized}>{curriculumBusy === "review" ? <LoaderCircle className="size-3.5 animate-spin" /> : <Save className="size-3.5" />}复核此精确蓝图</Button>}
                    {curriculum.review_status === "reviewed" && <Button size="sm" onClick={() => void curriculumSeal()} disabled={Boolean(curriculumBusy) || !versionFamily || !curriculumTeacherAuthorized}><ShieldCheck className="size-3.5" />确认并密封评分权威</Button>}
                    {curriculum.review_status === "sealed" && <Button variant="subtle" size="sm" onClick={() => void curriculumRevoke()} disabled={Boolean(curriculumBusy) || !versionFamily || !curriculumTeacherAuthorized}><CircleAlert className="size-3.5" />撤销评分权威</Button>}
                  </div>
                </section>
              )}
              {versionOpen && versionFamily && (
                <section aria-label="大纲版本历史" className="mt-5 rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-raised)] p-4">
                  <div className="flex flex-wrap items-center gap-2"><h2 className="text-sm font-medium">版本历史</h2><span className="text-[10px] text-[var(--app-faint)]">v{versionFamily.version} · 本地操作者尚未认证 · 发布不授予评分权威</span></div>
                  <ol className="mt-3 grid gap-2">
                    {[...versionFamily.revisions].reverse().map((revision) => {
                      const document = versionDocuments.find((item) => item.syllabus_id === revision.syllabus_id);
                      const parentRevision = versionFamily.revisions.find((item) => item.revision_id === revision.parent_revision_id);
                      const parent = versionDocuments.find((item) => item.syllabus_id === parentRevision?.syllabus_id);
                      const changes = structuralChangeLabels(parent, document);
                      return (
                        <li key={revision.revision_id} className="flex flex-wrap items-start gap-3 rounded-lg bg-[var(--app-surface-soft)] px-3 py-2.5">
                          <span className="grid size-7 shrink-0 place-items-center rounded-full bg-[var(--app-surface)] text-[10px] tabular-nums">{revision.revision_number}</span>
                          <div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><strong className="text-xs">{document?.title ?? revision.syllabus_id}</strong><span className={cn("rounded px-1.5 py-0.5 text-[9px]", revision.status === "published" ? "bg-[var(--app-success-soft)] text-[var(--app-success)]" : "bg-[var(--app-surface)] text-[var(--app-muted)]")}>{revision.status === "published" ? "已发布" : revision.status === "draft" ? "草稿" : "历史"}</span></div><p className="mt-1 text-[11px] leading-5 text-[var(--app-muted)]">{revision.change_summary}</p>{changes.length > 0 && <p className="mt-0.5 text-[10px] text-[var(--app-faint)]">结构差异：{changes.join("、")}</p>}<time className="mt-0.5 block text-[9px] text-[var(--app-faint)]">{revision.created_at_utc}</time></div>
                          {revision.status === "draft" && <Button type="button" size="sm" disabled={Boolean(activity)} onClick={() => void pointVersion(revision.revision_id, false)}>{activity === "publishing" ? <LoaderCircle className="size-3.5 animate-spin" /> : <CheckCircle2 className="size-3.5" />}发布</Button>}
                          {revision.status === "historical" && <Button type="button" size="sm" variant="subtle" disabled={Boolean(activity)} onClick={() => void pointVersion(revision.revision_id, true)}>{activity === "rolling-back" ? <LoaderCircle className="size-3.5 animate-spin" /> : <RotateCcw className="size-3.5" />}回滚到此版</Button>}
                        </li>
                      );
                    })}
                  </ol>
                </section>
              )}
              <section aria-label="大纲目标与先修要求" className="mt-6 grid gap-5 border-y border-[var(--app-border)] py-4 sm:grid-cols-2">
                <div>
                  <h2 className="text-xs font-medium text-[var(--app-text-soft)]">学习目标</h2>
                  {selected.learning_objectives?.length ? (
                    <ul className="mt-2 grid gap-1.5">
                      {selected.learning_objectives.map((objective) => (
                        <li key={objective} className="flex gap-2 text-xs leading-5 text-[var(--app-muted)]"><span aria-hidden="true" className="mt-2 size-1 shrink-0 rounded-full bg-[var(--app-accent)]" />{objective}</li>
                      ))}
                    </ul>
                  ) : <p className="mt-2 text-xs text-[var(--app-faint)]">未提供整体学习目标。</p>}
                </div>
                <div>
                  <h2 className="text-xs font-medium text-[var(--app-text-soft)]">先修要求</h2>
                  {selected.prerequisites?.length ? (
                    <ul className="mt-2 grid gap-1.5">
                      {selected.prerequisites.map((prerequisite) => (
                        <li key={prerequisite} className="flex gap-2 text-xs leading-5 text-[var(--app-muted)]"><span aria-hidden="true" className="mt-2 size-1 shrink-0 rounded-full bg-[var(--app-faint)]" />{prerequisite}</li>
                      ))}
                    </ul>
                  ) : <p className="mt-2 text-xs text-[var(--app-faint)]">无额外先修要求。</p>}
                </div>
              </section>
              <div className="mt-6 grid gap-3">
                {selected.modules.map((module) => <ModuleTree key={module.module_id} module={module} syllabus={selected} expanded={expandedModules.has(module.module_id)} startingLessonId={startingLessonId} onToggle={() => setExpandedModules((current) => {const next = new Set(current); if (next.has(module.module_id)) next.delete(module.module_id); else next.add(module.module_id); return next;})} onStartLesson={startLesson} />)}
              </div>
            </div>
          ) : (
            <div className="grid min-h-[55vh] place-items-center text-center">
              <div className="max-w-sm">
                <span className="mx-auto grid size-12 place-items-center rounded-xl bg-[var(--app-surface-raised)] text-[var(--app-muted)]"><BookOpenText className="size-5" /></span>
                <h1 className="mt-4 text-base font-medium text-[var(--app-text)]">还没有教学大纲</h1>
                <p className="mt-2 text-xs leading-5 text-[var(--app-muted)]">从主题生成一份结构化大纲，或导入已有的 teaching_syllabus.v1 JSON。</p>
                <div className="mt-4 flex justify-center gap-2"><Button disabled={Boolean(activity)} onClick={() => setGeneratorOpen(true)} className="bg-[var(--app-accent)] text-[#211714] hover:bg-[var(--app-accent-hover)]"><Sparkles className="size-4" />生成大纲</Button><Button variant="subtle" disabled={Boolean(activity)} onClick={() => importInputRef.current?.click()}><FileJson className="size-4" />导入 JSON</Button></div>
                {!loading && <Button variant="subtle" size="sm" className="mt-3" onClick={() => void refresh()}><RefreshCcw className="size-3.5" />重新读取</Button>}
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
