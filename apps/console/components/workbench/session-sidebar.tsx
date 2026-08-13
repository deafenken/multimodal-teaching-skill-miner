"use client";

import {
  BookOpenText,
  BrainCircuit,
  ChevronDown,
  CircleCheck,
  Cpu,
  GraduationCap,
  LogOut,
  MessageSquareText,
  Plus,
  Search,
  Settings,
  Sparkles,
  TerminalSquare,
  UserRound,
  X
} from "lucide-react";
import {useEffect, useState} from "react";

import {Button} from "@/components/ui/button";
import {ProjectSwitcher} from "@/components/workbench/project-switcher";
import {ProjectLibrary} from "@/components/workbench/project-library";
import {cn} from "@/lib/cn";
import type {BootstrapPayload, LearningProject, LearningProjectSummary, LearningProjectTrashItem, SessionListItem} from "@/lib/types";

type SidebarSkill = NonNullable<BootstrapPayload["skills"]>[number];

function SessionGroup({label, items, activeId, onSelect, disabled = false, countPrefix = "R"}: {
  label: string;
  items: SessionListItem[];
  activeId: string;
  onSelect: (id: string) => void;
  disabled?: boolean;
  countPrefix?: string;
}) {
  if (!items.length) return null;
  return (
    <section className="mt-6">
      <h2 className="px-4 text-xs font-medium text-[var(--app-muted)]">{label}</h2>
      <div className="mt-2 grid gap-1 px-2">
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(item.id)}
            className={cn(
              "group flex min-h-10 w-full items-center gap-2 rounded-lg px-3 text-left text-sm text-[var(--app-text-soft)] transition-colors hover:bg-[var(--app-hover)] hover:text-[var(--app-text)] disabled:cursor-not-allowed disabled:opacity-50",
              activeId === item.id && "bg-[var(--app-selected)] text-[var(--app-text)]"
            )}
          >
            <span className={cn("size-2 shrink-0 rounded-full border border-[var(--app-faint)]", item.status === "active" && "border-[var(--app-blue)] bg-[var(--app-blue)]", item.status === "succeeded" && "border-[var(--app-success)] bg-[var(--app-success)]")} />
            <span className="min-w-0 flex-1 truncate">{item.title}</span>
            <span className="hidden text-[10px] text-[var(--app-faint)] group-hover:inline">{countPrefix}{item.round}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

export function SessionSidebar({items, activeId, activeView, syllabiOpen, onSelect, onChat, onTeach, onSyllabi, onNew, onSettings, onSignOut, signOutAvailable, signOutBusy, signOutError, onCommandCenter, onChooseSkill, skills, selectedSkillId, projects, trashedProjects, activeProject, activeProjectId, onProjectChange, onProjectNotice, onSelectProject, onCreateProject, onRenameProject, onPinProject, onArchiveProject, onTrashProject, onRestoreProject, onExportProject, onPurgeProject, onOpenProjectSyllabus, onOpenProjectTeachingSession, onOpenProjectResource, providerStatus, runtimePolicy, backendConnected, controlsDisabled, mobileOpen = false, onClose = () => undefined}: {
  items: SessionListItem[];
  activeId: string;
  activeView: "chat" | "teach";
  syllabiOpen: boolean;
  onSelect: (id: string) => void;
  onChat: () => void;
  onTeach: () => void;
  onSyllabi: () => void;
  onNew: () => void;
  onSettings: () => void;
  onSignOut: () => void;
  signOutAvailable: boolean;
  signOutBusy: boolean;
  signOutError?: string;
  onCommandCenter: () => void;
  onChooseSkill: (skillId: string) => void;
  skills: SidebarSkill[];
  selectedSkillId?: string | null;
  projects: LearningProjectSummary[];
  trashedProjects: LearningProjectTrashItem[];
  activeProject?: LearningProject | null;
  activeProjectId?: string | null;
  onProjectChange: (project: LearningProject) => void;
  onProjectNotice: (message: string) => void;
  onSelectProject: (projectId: string) => void;
  onCreateProject: (title: string) => void;
  onRenameProject: (projectId: string, title: string) => void;
  onPinProject: (projectId: string, pinned: boolean) => void;
  onArchiveProject: (projectId: string, archived: boolean) => void;
  onTrashProject: (projectId: string) => void;
  onRestoreProject: (project: LearningProjectTrashItem) => void;
  onExportProject: (project: LearningProjectSummary) => void;
  onPurgeProject: (project: LearningProjectTrashItem) => void;
  onOpenProjectSyllabus: (syllabusId: string) => void;
  onOpenProjectTeachingSession: (sessionId: string) => void;
  onOpenProjectResource: (resourceId: string) => void;
  providerStatus?: {provider?: string; model?: string; configured?: boolean};
  runtimePolicy?: BootstrapPayload["agent_runtime_policy"];
  backendConnected: boolean;
  controlsDisabled: boolean;
  mobileOpen?: boolean;
  onClose?: () => void;
}) {
  const [isMobile, setIsMobile] = useState(false);
  const [panel, setPanel] = useState<"routines" | "skills" | "more" | null>(null);
  const [historyQuery, setHistoryQuery] = useState("");
  const [historyLimit, setHistoryLimit] = useState(50);
  useEffect(() => {
    const media = window.matchMedia("(max-width: 900px)");
    const sync = (event: MediaQueryList | MediaQueryListEvent) => setIsMobile(event.matches);
    sync(media);
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);
  useEffect(() => {
    if (!isMobile || !mobileOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isMobile, mobileOpen, onClose]);
  useEffect(() => {
    if (activeView === "chat") setPanel(null);
    else setPanel((current) => current ?? "routines");
  }, [activeView]);
  useEffect(() => {
    if (syllabiOpen) setPanel(null);
  }, [syllabiOpen]);
  const selectSession = (id: string) => {
    onSelect(id);
    if (isMobile) onClose();
  };
  const primarySkills = skills.filter((skill) => skill.role !== "support");
  const panelSkills = panel === "routines" ? primarySkills.slice(0, 6) : primarySkills;
  const filteredItems = items.filter((item) => !historyQuery.trim() || `${item.title}\n${item.learner}`.toLocaleLowerCase().includes(historyQuery.trim().toLocaleLowerCase()));
  const visibleItems = filteredItems.slice(0, historyLimit);
  return (
    <>
      <button
        type="button"
        aria-label="关闭会话栏"
        onClick={onClose}
        className={cn("fixed inset-0 z-40 bg-black/30 transition-opacity min-[901px]:hidden", mobileOpen && isMobile ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0")}
      />
      <aside
        aria-label="会话列表"
        aria-hidden={isMobile && !mobileOpen}
        inert={isMobile && !mobileOpen ? true : undefined}
        className={cn(
          "flex h-full min-h-0 w-64 shrink-0 flex-col border-r border-[var(--app-border)] bg-[var(--app-sidebar)] text-[var(--app-text-soft)] transition-[transform,background-color,border-color] duration-200 max-[900px]:fixed max-[900px]:inset-y-0 max-[900px]:left-0 max-[900px]:z-50 max-[900px]:shadow-[18px_0_48px_var(--app-shadow)]",
          isMobile && !mobileOpen && "max-[900px]:pointer-events-none max-[900px]:-translate-x-full"
        )}
      >
      <div className="flex h-12 items-center gap-2 px-4">
        <span className="grid size-8 place-items-center overflow-hidden rounded-lg bg-[var(--app-accent-soft)]" aria-hidden="true"><img src="/teachlab-mascot.svg" alt="" className="size-9 translate-y-0.5" /></span>
        <strong className="text-sm font-semibold tracking-tight text-[var(--app-text)]">TeachLab</strong>
        <Button variant="subtle" size="icon" className="ml-auto min-[901px]:hidden" onClick={onClose} aria-label="关闭会话栏"><X className="size-4" /></Button>
      </div>

      <ProjectSwitcher projects={projects} trashedProjects={trashedProjects} activeProjectId={activeProjectId} busy={controlsDisabled} onSelect={onSelectProject} onCreate={onCreateProject} onRename={onRenameProject} onPin={onPinProject} onArchive={onArchiveProject} onTrash={onTrashProject} onRestore={onRestoreProject} onExport={onExportProject} onPurge={onPurgeProject} />
      {activeProject && <ProjectLibrary project={activeProject} onProjectChange={onProjectChange} onNotice={onProjectNotice} onOpenSyllabus={onOpenProjectSyllabus} onOpenTeachingSession={onOpenProjectTeachingSession} onOpenResource={onOpenProjectResource} />}

      <div className="mx-3 mt-2 grid grid-cols-2 gap-1 rounded-xl bg-[var(--app-surface)] p-1" role="group" aria-label="工作模式">
        <Button
          variant={activeView === "chat" ? "selected" : "subtle"}
          className="h-10"
          aria-pressed={activeView === "chat"}
          title="Chat（⌘/Ctrl+1）"
          onClick={() => {onChat(); if (isMobile) onClose();}}
        >
          <MessageSquareText className="size-4" />Chat
        </Button>
        <Button
          variant={activeView === "teach" ? "selected" : "subtle"}
          className="h-10"
          aria-pressed={activeView === "teach"}
          title="Teach（⌘/Ctrl+2）"
          onClick={() => {onTeach(); if (isMobile) onClose();}}
        >
          <GraduationCap className="size-4" />Teach
        </Button>
      </div>

      <nav className="mt-4 grid gap-1 px-3">
        <Button variant="subtle" className="h-10 justify-start px-2 text-base" onClick={() => {onNew(); if (isMobile) onClose();}}><Plus className="size-5" />{activeView === "chat" ? "新建对话" : "新建教学"}</Button>
        {activeView === "teach" && (
          <>
            <Button variant={syllabiOpen ? "selected" : "subtle"} className="h-10 justify-start px-2 text-base" aria-pressed={syllabiOpen} onClick={() => {onSyllabi(); if (isMobile) onClose();}}><BookOpenText className="size-5" />教学大纲</Button>
            <Button variant={panel === "routines" && !syllabiOpen ? "selected" : "subtle"} className="h-10 justify-start px-2 text-base" onClick={() => {onTeach(); setPanel((current) => current === "routines" && !syllabiOpen ? null : "routines");}}><Sparkles className="size-5" />教学例程</Button>
            <Button variant={panel === "skills" && !syllabiOpen ? "selected" : "subtle"} className="h-10 justify-start px-2 text-base" onClick={() => {onTeach(); setPanel((current) => current === "skills" && !syllabiOpen ? null : "skills");}}><BrainCircuit className="size-5" />手动 Skill</Button>
            <Button variant={panel === "more" && !syllabiOpen ? "selected" : "subtle"} className="h-10 justify-start px-2 text-base" onClick={() => {onTeach(); setPanel((current) => current === "more" && !syllabiOpen ? null : "more");}}><ChevronDown className="size-5" />运行状态</Button>
          </>
        )}
      </nav>

      {activeView === "teach" && panel && !syllabiOpen && (
        <section className="message-enter mx-3 mt-3 max-h-[310px] overflow-y-auto rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-raised)] p-2 shadow-[0_14px_36px_var(--app-shadow)]">
          {panel === "more" ? (
            <div className="grid gap-2 p-1 text-xs text-[var(--app-muted)]">
              <div className="flex items-center justify-between rounded-lg bg-[var(--app-surface)] px-3 py-2"><span className="flex items-center gap-2"><Cpu className="size-3.5" />模型</span><strong className="text-[var(--app-text-soft)]">{providerStatus?.model ?? providerStatus?.provider ?? "未连接"}</strong></div>
              <div className="flex items-center justify-between rounded-lg bg-[var(--app-surface)] px-3 py-2"><span>Agent Loop</span><strong className="text-[var(--app-text-soft)]">{runtimePolicy?.agent_loop_enabled ? "已启用" : "未启用"}</strong></div>
              <div className="flex items-center justify-between rounded-lg bg-[var(--app-surface)] px-3 py-2"><span>可恢复会话</span><strong className="text-[var(--app-text-soft)]">{runtimePolicy?.recoverable_context ? "支持" : "当前进程"}</strong></div>
              <div className="flex items-center justify-between rounded-lg bg-[var(--app-surface)] px-3 py-2"><span>历史会话</span><strong className="text-[var(--app-text-soft)]">{items.length}</strong></div>
            </div>
          ) : (
            <>
              <div className="px-2 pb-2 pt-1">
                <p className="text-xs font-medium text-[var(--app-text-soft)]">{panel === "routines" ? "后端教学例程" : "手动 Skill 路由"}</p>
                <p className="mt-1 text-[10px] leading-4 text-[var(--app-faint)]">{panel === "routines" ? `已从真实 Skill Library 加载 ${primarySkills.length} 个主 Skill。` : "选择后通过后端命令锁定；新会话会在建立后生效。"}</p>
              </div>
              <div className="grid gap-1">
                {panelSkills.map((skill) => (
                  <button
                    key={skill.skill_id}
                    type="button"
                    disabled={controlsDisabled}
                    onClick={() => onChooseSkill(skill.skill_id)}
                    className={cn("group rounded-lg px-2.5 py-2 text-left transition hover:bg-[var(--app-hover)] disabled:cursor-not-allowed disabled:opacity-50", selectedSkillId === skill.skill_id && "bg-[var(--app-accent-soft)]")}
                    title={skill.selection_rationale}
                  >
                    <span className="flex items-center gap-2 text-xs text-[var(--app-text-soft)]">
                      {selectedSkillId === skill.skill_id ? <CircleCheck className="size-3.5 text-[var(--app-accent)]" /> : <BrainCircuit className="size-3.5 text-[var(--app-faint)]" />}
                      <span className="min-w-0 flex-1 truncate">{skill.name}</span>
                      <span className="text-[9px] uppercase text-[var(--app-faint)]">{skill.focus_dimension ?? skill.role}</span>
                    </span>
                    {panel === "routines" && skill.selection_rationale && <span className="mt-1 line-clamp-2 block pl-[22px] text-[10px] leading-4 text-[var(--app-faint)]">{skill.selection_rationale}</span>}
                  </button>
                ))}
              </div>
            </>
          )}
        </section>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto pb-5">
        <label className="mx-3 mt-4 flex h-8 items-center gap-2 rounded-md border border-[var(--app-border)] bg-[var(--app-surface)] px-2"><Search className="size-3 text-[var(--app-faint)]" /><input value={historyQuery} onChange={(event) => {setHistoryQuery(event.target.value); setHistoryLimit(50);}} maxLength={200} placeholder={activeView === "chat" ? "搜索全部对话" : "搜索全部教学历史"} className="min-w-0 flex-1 bg-transparent text-[11px] text-[var(--app-text)] outline-none" /></label>
        <SessionGroup label={activeView === "chat" ? "当前对话" : "当前教学"} items={visibleItems.filter((item) => item.group === "pinned")} activeId={activeId} onSelect={selectSession} countPrefix={activeView === "chat" ? "" : "R"} />
        <SessionGroup label={activeView === "chat" ? "最近" : "教学历史"} items={visibleItems.filter((item) => item.group === "recent")} activeId={activeId} onSelect={selectSession} countPrefix={activeView === "chat" ? "" : "R"} />
        {historyLimit < filteredItems.length && <button type="button" onClick={() => setHistoryLimit((value) => value + 50)} className="mx-4 mt-3 rounded-md px-2 py-1.5 text-xs text-[var(--app-accent-text)] hover:bg-[var(--app-hover)]">加载更多（{visibleItems.length}/{filteredItems.length}）</button>}
      </div>

      <div className="border-t border-[var(--app-border)]">
        {signOutError && <p role="alert" className="px-4 pt-2 text-xs leading-4 text-[var(--app-danger)]">{signOutError}</p>}
        <div className="flex h-14 items-center gap-2 px-4 text-sm text-[var(--app-muted)]">
          <UserRound className="size-4" />
          <span className="min-w-0 flex-1 truncate">当前操作者</span>
          <span role="img" className={cn("size-2 rounded-full", backendConnected ? "bg-[var(--app-success)]" : "bg-[var(--app-faint)]")} aria-label={backendConnected ? "后端已连接" : "后端未连接"} />
          <Button variant="subtle" size="icon" onClick={() => {onCommandCenter(); if (isMobile) onClose();}} aria-label="打开命令与后台任务中心" title="命令与后台任务（⌘/Ctrl+K）"><TerminalSquare className="size-4" /></Button>
          <Button variant="subtle" size="icon" onClick={() => {onSettings(); if (isMobile) onClose();}} aria-label="打开设置" title="设置（⌘/Ctrl+,）"><Settings className="size-4" /></Button>
          {signOutAvailable && <Button variant="subtle" size="icon" disabled={signOutBusy} onClick={onSignOut} aria-label={signOutBusy ? "正在安全退出" : "安全退出当前设备"} title="安全退出当前设备"><LogOut className={cn("size-4", signOutBusy && "animate-pulse")} /></Button>}
        </div>
      </div>
      </aside>
    </>
  );
}
