"use client";

import {
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  CircleAlert,
  CornerDownLeft,
  Globe2,
  ImagePlus,
  Lightbulb,
  LoaderCircle,
  Menu,
  Mic,
  MoreHorizontal,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  RefreshCcw,
  Send,
  Square
} from "lucide-react";
import {memo, useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type ClipboardEvent, type DragEvent, type FormEvent, type KeyboardEvent} from "react";

import {Badge} from "@/components/ui/badge";
import {Button} from "@/components/ui/button";
import {MarkdownMessage} from "@/components/workbench/markdown-message";
import {isRovingMenuKey, nextRovingMenuIndex} from "@/lib/accessibility";
import {chatAttachmentAccept, chatAttachmentTypeLabel, chatResourceRefs} from "@/lib/chat-attachments";
import {cn} from "@/lib/cn";
import {boundedMessageWindow, loadDraft, saveDraft} from "@/lib/offline-runtime";
import {openSourcePreview, SOURCE_PREVIEW_WINDOW} from "@/lib/markdown-security";
import type {BootstrapPayload, LessonProgress, QueuedChatPrompt, ResourceUploadItem, TeachingMessage} from "@/lib/types";

const commands = [
  {value: "/auto", title: "恢复自动路由", detail: "让 Agent 重新选择下一步 Skill"},
  {value: "/+skill ", title: "指定 Skill", detail: "锁定一个教学方法，直到 /auto"},
  {value: "/stop", title: "结束并转人工", detail: "保留当前学情并停止推进"}
];

const INITIAL_MESSAGE_WINDOW = 300;
const MESSAGE_WINDOW_INCREMENT = 300;

function visibleLessonProgress(progress?: LessonProgress) {
  const label = progress?.phase_label?.trim();
  const index = Number(progress?.phase_index);
  if (!label || progress?.phase_count !== 6 || !Number.isInteger(index) || index < 1 || index > 6) return null;
  return {label, index};
}

type ConversationSkill = NonNullable<BootstrapPayload["skills"]>[number];

const Message = memo(function Message({message, mode, onInsert, onRetry}: {message: TeachingMessage; mode: "chat" | "teach"; onInsert: (text: string) => void; onRetry: (messageId: string) => void}) {
  const [copied, setCopied] = useState(false);
  if (message.role === "tool") {
    return (
      <details className="group rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-soft)]">
        <summary className="flex min-h-10 cursor-pointer list-none items-center gap-2 px-3 text-sm text-[var(--app-muted)]">
          <ChevronRight className="size-4 transition-transform group-open:rotate-90" />
          <span>{message.toolLabel}</span>
          <span className="ml-auto text-xs text-[var(--app-faint)]">{message.createdAt}</span>
        </summary>
        <div className="border-t border-[var(--app-border)] px-9 py-3 text-xs text-[var(--app-muted)]">{message.toolDetail}</div>
      </details>
    );
  }

  const learner = message.role === "learner";
  const lessonPhaseLabel = mode === "teach" && !learner ? message.lessonPhase?.phase_label?.trim() : "";
  const safeSources = (message.sources ?? []).filter((source) => {
    try {
      const parsed = new URL(source.url);
      return (parsed.protocol === "https:" || parsed.protocol === "http:") && Boolean(source.title.trim());
    } catch {
      return false;
    }
  });
  return (
    <article className="teachlab-message message-enter group min-w-0">
      <div className="min-w-0">
        {learner ? (
          <div className="flex max-w-[min(100%,760px)] items-start rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)] px-4 py-3 text-[var(--app-text)]">
            <span aria-hidden="true" className="mr-2 select-none pt-px font-mono text-[var(--app-accent)]">›</span>
            <div className="min-w-0 flex-1">
              <MarkdownMessage className="text-[var(--app-text)]">{message.body}</MarkdownMessage>
              {(message.attachmentLabels?.length ?? 0) > 0 && (
                <div className="mt-2 flex flex-wrap gap-1" aria-label="本回合附件">
                  {message.attachmentLabels?.slice(0, 6).map((label) => <Badge key={label} className="max-w-52 truncate bg-[var(--app-selected)] text-[var(--app-muted)]">附件 · {label}</Badge>)}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className={cn("min-w-0", !message.body && message.thinking && "text-[var(--app-muted)]")}>
            {message.body ? message.streaming
              ? <p className="m-0 whitespace-pre-wrap break-words text-[15px] leading-7 text-[var(--app-text)]">{message.body}</p>
              : <MarkdownMessage>{message.body}</MarkdownMessage> : message.thinking ? (
              <p className="m-0 text-[15px] leading-7 text-[var(--app-muted)]"><LoaderCircle aria-hidden="true" className="mr-2 inline-block size-3 animate-spin text-[var(--app-accent)]" />{message.thinking}</p>
            ) : null}
            {message.streaming && message.body && <span aria-hidden="true" className="ml-1 inline-block h-4 w-[2px] translate-y-[3px] animate-pulse bg-[var(--app-accent)]" />}
            {message.status === "failed" && (
              <div role="alert" className="mt-3 flex items-start gap-2 rounded-lg border border-[var(--app-danger)] bg-[var(--app-danger-soft)] px-3 py-2 text-xs text-[var(--app-danger)]">
                <CircleAlert className="mt-0.5 size-3.5 shrink-0" />
                <span>{message.error || "本次回答未完成，可以原位重试。"}</span>
              </div>
            )}
          </div>
        )}
        <div className="mt-2 flex min-h-6 flex-wrap items-center gap-1.5 text-xs text-[var(--app-faint)]">
          {lessonPhaseLabel
            ? <Badge className="bg-[var(--app-accent-soft)] text-[var(--app-accent-text)]">{lessonPhaseLabel}</Badge>
            : mode !== "teach" && message.skill
              ? <Badge className="bg-[var(--app-accent-soft)] text-[var(--app-accent-text)]">{message.skill}</Badge>
              : null}
          {message.webSearchUsed && <Badge className="bg-[var(--app-success-soft)] text-[var(--app-success)]">已联网</Badge>}
          {message.status === "running" && <Badge className="bg-[var(--app-accent-soft)] text-[var(--app-accent-text)]">生成中</Badge>}
          {message.status === "failed" && <Badge className="bg-[var(--app-danger-soft)] text-[var(--app-danger)]">失败</Badge>}
          {message.status === "stopped" && <Badge className="bg-[var(--app-warning-soft)] text-[var(--app-warning)]">已停止</Badge>}
          <button
            type="button"
            onClick={async () => {
              await navigator.clipboard?.writeText(message.body);
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1200);
            }}
            aria-label={copied ? "已复制" : "复制消息"}
            title={copied ? "已复制" : "复制消息"}
            className="inline-flex size-6 items-center justify-center rounded-md hover:bg-[var(--app-hover)] hover:text-[var(--app-text-soft)]"
          >
            {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
          </button>
          {!learner && (
            <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 [@media(hover:none)]:opacity-100">
              {message.retryable && message.status !== "running" && (
                <button
                  type="button"
                  onClick={() => onRetry(message.id)}
                  aria-label={message.status === "failed" ? "重试回答" : "重新生成回答"}
                  title={message.status === "failed" ? "重试" : "重新生成"}
                  className="inline-flex h-6 items-center justify-center gap-1 rounded-md px-1.5 hover:bg-[var(--app-hover)] hover:text-[var(--app-text-soft)]"
                >
                  <RefreshCcw className="size-3" /><span>{message.status === "failed" ? "重试" : "重新生成"}</span>
                </button>
              )}
              <button
                type="button"
                onClick={() => onInsert(mode === "chat" ? "请用一个具体例子说明。" : "请换一个例子，保持当前知识点不变。")}
                aria-label={mode === "chat" ? "举个例子" : "换个例子"}
                title={mode === "chat" ? "举个例子" : "换个例子"}
                className="inline-flex size-6 items-center justify-center rounded-md hover:bg-[var(--app-hover)] hover:text-[var(--app-text-soft)]"
              >
                <RefreshCcw className="size-3" />
              </button>
              <button
                type="button"
                onClick={() => onInsert(mode === "chat" ? "请把上面的回答说得更简洁一些。" : "请只给我一个单步提示，不要直接给出答案。")}
                aria-label={mode === "chat" ? "更简洁" : "单步提示"}
                title={mode === "chat" ? "更简洁" : "单步提示"}
                className="inline-flex size-6 items-center justify-center rounded-md hover:bg-[var(--app-hover)] hover:text-[var(--app-text-soft)]"
              >
                <Lightbulb className="size-3" />
              </button>
              <button
                type="button"
                onClick={() => onInsert(mode === "chat" ? "请继续展开最关键的部分。" : "我认为这一步理解有误，请指出需要纠正的地方。")}
                aria-label={mode === "chat" ? "继续展开" : "纠正理解"}
                title={mode === "chat" ? "继续展开" : "纠正理解"}
                className="inline-flex size-6 items-center justify-center rounded-md hover:bg-[var(--app-hover)] hover:text-[var(--app-text-soft)]"
              >
                <CircleAlert className="size-3" />
              </button>
              <button type="button" onClick={() => onInsert(mode === "chat" ? "请从另一个角度回答。" : "请继续解释当前证据。")} aria-label="更多消息操作" title="更多" className="inline-flex size-6 items-center justify-center rounded-md hover:bg-[var(--app-hover)] hover:text-[var(--app-text-soft)]"><MoreHorizontal className="size-3" /></button>
            </div>
          )}
          <span className="ml-1">{message.createdAt}</span>
        </div>
        {!learner && safeSources.length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-[var(--app-muted)]">
            <span>来源</span>
            {safeSources.slice(0, 6).map((source, index) => (
              <a
                key={`${source.url}-${index}`}
                href={source.url}
                target={SOURCE_PREVIEW_WINDOW}
                onClick={(event) => {
                  event.preventDefault();
                  openSourcePreview(source.url);
                }}
                className="max-w-56 truncate rounded-full border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-2.5 py-1 text-[var(--app-text-soft)] hover:border-[var(--app-border-strong)] hover:bg-[var(--app-hover)]"
                title={source.title}
              >
                {index + 1}. {source.title}
              </a>
            ))}
          </div>
        )}
      </div>
    </article>
  );
});

export function ConversationPane({mode, messages, onSend, onRetry, onQueueFollowUp, queuedChatPrompts, onCancelQueuedPrompt, onImportResources, resourceUploads, onRemoveResource, onStop, onSelectSkill, skills, selectedSkillId, sidebarOpen, onToggleSidebar, inspectorOpen, onToggleInspector, busy, disabled = false, disabledReason, canAttach, supportedResourceExtensions, webSearchEnabled = true, webSearchAvailable = false, onWebSearchEnabledChange, title, subtitle, queuedSkillName, sessionMeta}: {
  mode: "chat" | "teach";
  messages: TeachingMessage[];
  onSend: (text: string, attachment?: File, options?: {webSearch?: boolean}) => void;
  onRetry: (messageId: string) => void;
  onQueueFollowUp: (text: string, options?: {webSearch?: boolean}) => boolean;
  queuedChatPrompts: QueuedChatPrompt[];
  onCancelQueuedPrompt: (id: string) => void;
  onImportResources: (files: File[]) => void;
  resourceUploads: ResourceUploadItem[];
  onRemoveResource: (localId: string) => void;
  onStop: () => void;
  onSelectSkill: (skillId: string | null) => void;
  skills: ConversationSkill[];
  selectedSkillId?: string | null;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  inspectorOpen: boolean;
  onToggleInspector: () => void;
  busy: boolean;
  disabled?: boolean;
  disabledReason?: string;
  canAttach: boolean;
  supportedResourceExtensions: readonly string[];
  webSearchEnabled?: boolean;
  webSearchAvailable?: boolean;
  onWebSearchEnabledChange: (enabled: boolean) => void;
  title: string;
  subtitle: string;
  queuedSkillName?: string;
  sessionMeta?: {rounds: number; contextVersion: number; skill?: string; selectionReason?: string; fallbackCount?: number; lessonProgress?: LessonProgress | null};
}) {
  const [drafts, setDrafts] = useState({chat: "", teach: ""});
  const [draftsLoaded, setDraftsLoaded] = useState(false);
  const [online, setOnline] = useState(true);
  const [messageWindow, setMessageWindow] = useState(INITIAL_MESSAGE_WINDOW);
  const draft = drafts[mode];
  const setDraft = useCallback((value: string) => setDrafts((current) => ({...current, [mode]: value})), [mode]);
  const [commandOpen, setCommandOpen] = useState(false);
  const [controlMenuOpen, setControlMenuOpen] = useState(false);
  const [attachmentName, setAttachmentName] = useState("");
  const [attachment, setAttachment] = useState<File | null>(null);
  const [chatDropActive, setChatDropActive] = useState(false);
  const [selectedCommand, setSelectedCommand] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const controlMenuRef = useRef<HTMLDivElement>(null);
  const controlMenuTriggerRef = useRef<HTMLButtonElement>(null);
  const controlMenuPopupRef = useRef<HTMLDivElement>(null);
  const resourceInputRef = useRef<HTMLInputElement>(null);
  const chatAttachmentInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const stickToBottomRef = useRef(true);
  const filteredCommands = useMemo(() => {
    const query = draft.trim().split(/\s+/, 1)[0];
    if (mode !== "teach") return [];
    return commands.filter((command) => !query || command.value.trim().startsWith(query));
  }, [draft, mode]);
  const resourcesBusy = resourceUploads.some((item) => item.status === "extracting");
  const readyChatAttachmentCount = mode === "chat" ? chatResourceRefs(resourceUploads).length : 0;
  const headerLessonProgress = mode === "teach" ? visibleLessonProgress(sessionMeta?.lessonProgress ?? undefined) : null;
  const primarySkills = useMemo(() => skills.filter((skill) => skill.role !== "support"), [skills]);
  const selectedSkill = primarySkills.find((skill) => skill.skill_id === selectedSkillId);
  const interactionDisabled = disabled || !online;
  const resourceAccept = useMemo(() => {
    const formats = supportedResourceExtensions.map((value) => `.${value.replace(/^\./, "")}`);
    return [...formats, ".json", "application/json"].join(",");
  }, [supportedResourceExtensions]);
  const chatResourceAccept = useMemo(
    () => chatAttachmentAccept(supportedResourceExtensions),
    [supportedResourceExtensions],
  );
  const localImageExtractionAvailable = supportedResourceExtensions.some((value) => ["png", "jpg", "jpeg", "webp"].includes(value.replace(/^\./, "").toLocaleLowerCase()));
  const {hidden: hiddenMessageCount, items: renderedMessages} = useMemo(
    () => boundedMessageWindow(messages, messageWindow),
    [messageWindow, messages],
  );

  useEffect(() => {
    let active = true;
    Promise.all([loadDraft("composer:chat"), loadDraft("composer:teach")]).then(([chat, teach]) => {
      if (!active) return;
      setDrafts({chat, teach});
      setDraftsLoaded(true);
    }).catch(() => {
      if (active) setDraftsLoaded(true);
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!draftsLoaded) return;
    const timer = window.setTimeout(() => {
      void Promise.all([
        saveDraft("composer:chat", drafts.chat),
        saveDraft("composer:teach", drafts.teach),
      ]).catch(() => undefined);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [drafts, draftsLoaded]);

  useEffect(() => {
    const update = () => setOnline(navigator.onLine);
    update();
    window.addEventListener("online", update);
    window.addEventListener("offline", update);
    return () => {
      window.removeEventListener("online", update);
      window.removeEventListener("offline", update);
    };
  }, []);

  useEffect(() => {
    setMessageWindow(INITIAL_MESSAGE_WINDOW);
  }, [mode]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  }, [draft]);

  useEffect(() => {
    if (canAttach) return;
    setAttachment(null);
    setAttachmentName("");
  }, [canAttach]);

  useEffect(() => {
    if (!sessionMeta && !busy && !disabled) textareaRef.current?.focus();
  }, [busy, disabled, sessionMeta]);

  useEffect(() => {
    if (!stickToBottomRef.current || !scrollRef.current) return;
    const node = scrollRef.current;
    node.scrollTo({top: node.scrollHeight, behavior: busy ? "auto" : "smooth"});
  }, [busy, messages.length, messages[messages.length - 1]?.body]);

  useEffect(() => {
    if (!controlMenuOpen) return;
    const frame = window.requestAnimationFrame(() => {
      const items = Array.from(controlMenuPopupRef.current?.querySelectorAll<HTMLButtonElement>('[role^="menuitem"]') ?? []);
      const selected = items.find((item) => item.getAttribute("aria-checked") === "true") ?? items[0];
      items.forEach((item) => { item.tabIndex = item === selected ? 0 : -1; });
      selected?.focus();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [controlMenuOpen, primarySkills.length, selectedSkillId]);

  useEffect(() => {
    const onPointerDown = (event: PointerEvent) => {
      if (controlMenuRef.current && !controlMenuRef.current.contains(event.target as Node)) {
        setControlMenuOpen(false);
      }
    };
    const onWindowKeyDown = (event: globalThis.KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        if (mode !== "teach") return;
        event.preventDefault();
        if (!draft.trim()) setDraft("/");
        textareaRef.current?.focus();
        setCommandOpen(true);
        setSelectedCommand(0);
        return;
      }
      if (event.key === "Escape") {
        if (busy) {
          event.preventDefault();
          onStop();
          return;
        }
        setCommandOpen(false);
        if (controlMenuOpen) {
          setControlMenuOpen(false);
          controlMenuTriggerRef.current?.focus();
        }
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onWindowKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onWindowKeyDown);
    };
  }, [busy, controlMenuOpen, draft, mode, onStop]);

  useEffect(() => {
    setCommandOpen(false);
    setControlMenuOpen(false);
    if (mode === "chat") {
      setAttachment(null);
      setAttachmentName("");
    }
  }, [setDraft]);

  function submit(event?: FormEvent) {
    event?.preventDefault();
    const value = draft.trim() || (mode === "chat" && readyChatAttachmentCount ? "请分析这些附件。" : "");
    if ((!value && !attachment) || interactionDisabled || resourcesBusy) return;
    if (busy) {
      if (mode !== "chat" || !value || attachment) return;
      const queued = onQueueFollowUp(value, {webSearch: webSearchEnabled});
      if (!queued) return;
      setDraft("");
      setCommandOpen(false);
      setControlMenuOpen(false);
      return;
    }
    onSend(value, attachment ?? undefined, {webSearch: mode === "chat" && webSearchEnabled});
    setDraft("");
    setAttachment(null);
    setAttachmentName("");
    setCommandOpen(false);
    setControlMenuOpen(false);
  }

  const insertDraft = useCallback((text: string) => {
    setDraft(text);
    setCommandOpen(false);
    setControlMenuOpen(false);
    textareaRef.current?.focus();
  }, [mode]);

  function chooseControlMode(skillId: string | null) {
    if (busy || interactionDisabled) return;
    setControlMenuOpen(false);
    onSelectSkill(skillId);
    window.requestAnimationFrame(() => controlMenuTriggerRef.current?.focus());
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.nativeEvent.isComposing) return;
    if (commandOpen && ["ArrowDown", "ArrowUp"].includes(event.key)) {
      event.preventDefault();
      if (!filteredCommands.length) return;
      const delta = event.key === "ArrowDown" ? 1 : -1;
      setSelectedCommand((current) => (current + delta + filteredCommands.length) % filteredCommands.length);
      return;
    }
    if (commandOpen && event.key === "Escape") {
      event.preventDefault();
      setCommandOpen(false);
      return;
    }
    if (commandOpen && event.key === "Tab") {
      const command = filteredCommands[selectedCommand];
      if (command) {
        event.preventDefault();
        setDraft(command.value);
        setCommandOpen(false);
      }
      return;
    }
    if (commandOpen && event.key === "Enter" && !event.shiftKey) {
      const command = filteredCommands[selectedCommand];
      if (command && draft.trim() !== command.value.trim()) {
        event.preventDefault();
        setDraft(command.value);
        setCommandOpen(false);
        return;
      }
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  function onMenuKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    const menu = event.currentTarget.closest('[role="menu"]');
    const items = menu ? Array.from(menu.querySelectorAll<HTMLButtonElement>('[role^="menuitem"]')) : [];
    const index = items.indexOf(event.currentTarget);
    if (isRovingMenuKey(event.key)) {
      event.preventDefault();
      const next = nextRovingMenuIndex(index, items.length, event.key);
      items.forEach((item, itemIndex) => { item.tabIndex = itemIndex === next ? 0 : -1; });
      items[next]?.focus();
    } else if (event.key === "Escape") {
      event.preventDefault();
      setControlMenuOpen(false);
      controlMenuTriggerRef.current?.focus();
    }
  }

  function handleImageChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    event.target.value = "";
    if (!file) return;
    if (!canAttach) {
      onImportResources([file]);
      return;
    }
    setAttachment(file);
    setAttachmentName(file.name);
  }

  function handleResourceChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (files.length) onImportResources(files);
  }

  function handleChatAttachmentChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (files.length) onImportResources(files);
  }

  function handleComposerDragOver(event: DragEvent<HTMLDivElement>) {
    if (mode !== "chat" || interactionDisabled || busy || resourcesBusy || !event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setChatDropActive(true);
  }

  function handleComposerDrop(event: DragEvent<HTMLDivElement>) {
    if (mode !== "chat") return;
    event.preventDefault();
    setChatDropActive(false);
    if (interactionDisabled || busy || resourcesBusy) return;
    const files = Array.from(event.dataTransfer.files ?? []);
    if (files.length) onImportResources(files);
  }

  function handleComposerPaste(event: ClipboardEvent<HTMLTextAreaElement>) {
    if (mode !== "chat" || interactionDisabled || busy || resourcesBusy) return;
    const files = Array.from(event.clipboardData.files ?? []);
    if (!files.length) return;
    event.preventDefault();
    onImportResources(files);
  }

  return (
    <main className="relative flex min-w-0 flex-1 flex-col bg-[var(--app-bg)] text-[var(--app-text)]">
      <header className="teachlab-main-header flex h-12 shrink-0 items-center gap-3 border-b border-[var(--app-border)] px-8 max-[680px]:px-4">
        <Button variant="subtle" size="icon" className="min-[901px]:hidden" onClick={onToggleSidebar} aria-label={sidebarOpen ? "关闭会话栏" : "打开会话栏"} aria-pressed={sidebarOpen}><Menu className="size-4" /></Button>
        <strong className="shrink-0 whitespace-nowrap text-base max-[420px]:max-w-[64px] max-[420px]:truncate max-[420px]:text-sm">{title}</strong>
        <span className="text-[var(--app-faint)]">/</span>
        <span className="min-w-0 truncate text-sm text-[var(--app-muted)]" title={subtitle}>{subtitle}</span>
        <div className="ml-auto flex shrink-0 items-center gap-2">
          {headerLessonProgress && (
            <div
              role="status"
              aria-live="polite"
              aria-atomic="true"
              aria-label={`当前教学阶段：正在${headerLessonProgress.label}，第 ${headerLessonProgress.index} 阶段，共 6 阶段`}
              className="flex items-center gap-1.5 whitespace-nowrap text-xs text-[var(--app-muted)]"
            >
              <span aria-hidden="true" className="size-1.5 rounded-full bg-[var(--app-accent)]" />
              <span><span className="max-[680px]:hidden">正在</span>{headerLessonProgress.label}</span>
              <span aria-hidden="true" className="text-[var(--app-faint)]">·</span>
              <span className="tabular-nums text-[var(--app-text-soft)]">{headerLessonProgress.index}/6</span>
            </div>
          )}
          <Button
            variant="subtle"
            size="icon"
            onClick={onToggleInspector}
            aria-label={inspectorOpen ? "折叠右侧检查器" : "展开右侧检查器"}
            aria-pressed={inspectorOpen}
          >
            {inspectorOpen ? <PanelRightClose className="size-4" /> : <PanelRightOpen className="size-4" />}
          </Button>
        </div>
      </header>

      {!online && (
        <div role="status" aria-live="polite" className="shrink-0 border-b border-[var(--app-accent-border)] bg-[var(--app-warning-soft)] px-4 py-2 text-center text-xs leading-5 text-[var(--app-warning)]">
          离线 · 当前只显示本机缓存，不会运行模型或发送请求。草稿仍会保存在本机，联网后可自行发送。
        </div>
      )}

      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-y-auto"
        onScroll={(event) => {
          const node = event.currentTarget;
          stickToBottomRef.current = node.scrollHeight - node.scrollTop - node.clientHeight < 180;
        }}
      >
        <div
          role="log"
          aria-live="polite"
          aria-relevant="additions text"
          aria-atomic="false"
          aria-label={mode === "teach" ? "教学对话记录" : "对话记录"}
          className="teachlab-message-list mx-auto grid w-full max-w-[860px] gap-7 px-8 pb-48 pt-8 max-[680px]:px-4"
        >
          {hiddenMessageCount > 0 && (
            <button
              type="button"
              className="mx-auto rounded-full border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-3 py-1.5 text-xs text-[var(--app-muted)]"
              onClick={() => setMessageWindow((count) => Math.min(messages.length, count + MESSAGE_WINDOW_INCREMENT))}
            >
              加载更早的 {Math.min(hiddenMessageCount, MESSAGE_WINDOW_INCREMENT)} 条消息（另有 {hiddenMessageCount} 条未渲染）
            </button>
          )}
          {renderedMessages.map((message) => <Message key={message.id} message={message} mode={mode} onInsert={insertDraft} onRetry={onRetry} />)}

          {!messages.length && !busy && (
            <div className="grid min-h-[54vh] place-items-center text-center">
              <div className="max-w-[560px]">
                <div className="relative mx-auto size-28">
                  <div className="teachlab-mascot-glow absolute inset-3 rounded-full blur-2xl" aria-hidden="true" />
                  <img className="teachlab-mascot relative size-28" src="/teachlab-mascot.svg" alt="TeachLab 学习伙伴" />
                </div>
                <p className="mt-5 text-[22px] font-medium tracking-tight text-[var(--app-text)]">{mode === "chat" ? "有什么想聊的？" : "今天想一起弄懂什么？"}</p>
                <p className="mt-2 text-sm leading-6 text-[var(--app-muted)]">{mode === "chat" ? "直接提问，我会先回答你的问题，并在需要时继续展开。" : "输入学习目标，TeachLab 会建立独立教学会话并根据你的回答调整路径。"}</p>
                {mode === "teach" && queuedSkillName && (
                  <p className="mt-3 inline-flex items-center gap-2 rounded-full border border-[var(--app-accent-border)] bg-[var(--app-accent-soft)] px-3 py-1 text-xs text-[var(--app-accent-text)]">首轮将使用：{queuedSkillName}</p>
                )}
                <div className="mt-6 flex flex-wrap justify-center gap-2">
                  {(mode === "chat" ? ["解释一个概念", "帮我梳理思路", "总结一段内容"] : ["拆解一个难点", "检查我的理解", "从一个例子开始"]).map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      disabled={interactionDisabled}
                      onClick={() => {
                        setDraft(suggestion);
                        textareaRef.current?.focus();
                      }}
                      className="rounded-full border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-3 py-1.5 text-xs text-[var(--app-muted)] transition hover:border-[var(--app-border-strong)] hover:bg-[var(--app-hover)] hover:text-[var(--app-text)] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
          {busy && !messages.some((message) => message.streaming) && (
            <div role="status" className="flex items-center gap-2 text-xs text-[var(--app-muted)]">
              <LoaderCircle aria-hidden="true" className="size-3 animate-spin text-[var(--app-accent)]" />
              正在生成回答…
            </div>
          )}
        </div>
      </div>

      <form onSubmit={submit} className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-[var(--app-bg)] via-[var(--app-bg)] to-transparent px-5 pb-4 pt-12">
        <div
          className={cn("teachlab-composer relative mx-auto max-w-[860px] rounded-xl border border-[var(--app-border-strong)] bg-[var(--app-surface-raised)] p-3 shadow-[0_18px_50px_var(--app-shadow)] transition-colors focus-within:border-[var(--app-focus-border)]", chatDropActive && "border-[var(--app-accent)] bg-[var(--app-accent-soft)]")}
          onDragOver={handleComposerDragOver}
          onDragLeave={(event) => {
            if (mode === "chat" && !event.currentTarget.contains(event.relatedTarget as Node | null)) setChatDropActive(false);
          }}
          onDrop={handleComposerDrop}
        >
          {disabled && disabledReason && <div role="status" className="mb-2 rounded-lg bg-[var(--app-warning-soft)] px-3 py-2 text-xs leading-5 text-[var(--app-warning)]">{disabledReason}</div>}
          {mode === "teach" && commandOpen && filteredCommands.length > 0 && (
            <div id="teachlab-command-options" role="listbox" aria-label="教学命令" className="absolute bottom-[calc(100%+8px)] left-0 w-[360px] max-w-full rounded-xl border border-[var(--app-border)] bg-[var(--app-overlay)] p-1.5 shadow-2xl">
              {filteredCommands.map((command, index) => (
                <button
                  key={command.value}
                  id={`teachlab-command-${index}`}
                  type="button"
                  role="option"
                  aria-selected={index === selectedCommand}
                  tabIndex={-1}
                  onClick={() => {setDraft(command.value); setCommandOpen(false); setControlMenuOpen(false); textareaRef.current?.focus();}}
                  className={cn("grid w-full grid-cols-[70px_minmax(0,1fr)] gap-2 rounded-lg px-3 py-2 text-left", index === selectedCommand && "bg-[var(--app-selected)]")}
                >
                  <code className="text-xs text-[var(--app-purple)]">{command.value.trim()}</code>
                  <span><strong className="block text-xs text-[var(--app-text)]">{command.title}</strong><small className="mt-0.5 block text-[10px] text-[var(--app-muted)]">{command.detail}</small></span>
                </button>
              ))}
            </div>
          )}
          <div className="flex items-start gap-2">
            <span aria-hidden="true" className="select-none pt-1 font-mono text-sm text-[var(--app-accent)]">›</span>
            <label htmlFor="teachlab-composer-input" className="sr-only">{mode === "chat" ? "输入消息" : "输入学习目标或教学命令"}</label>
            <textarea
              id="teachlab-composer-input"
              ref={textareaRef}
              aria-autocomplete={commandOpen ? "list" : "none"}
              aria-controls={commandOpen ? "teachlab-command-options" : undefined}
              aria-expanded={mode === "teach" ? commandOpen : undefined}
              aria-activedescendant={commandOpen && filteredCommands[selectedCommand] ? `teachlab-command-${selectedCommand}` : undefined}
              value={draft}
              rows={1}
              placeholder={mode === "chat" ? "输入消息" : "描述学习目标，输入 / 查看教学命令"}
              disabled={disabled && online}
              onChange={(event) => {
                const value = event.target.value;
                setDraft(value);
                setCommandOpen(mode === "teach" && value.trimStart().startsWith("/") && !value.includes(" "));
                setSelectedCommand(0);
              }}
              onKeyDown={onKeyDown}
              onPaste={handleComposerPaste}
              className="max-h-40 min-h-8 w-full resize-none border-0 bg-transparent px-0 py-1 text-sm leading-6 text-[var(--app-text)] outline-none placeholder:text-[var(--app-faint)] focus-visible:outline-none"
            />
          </div>
          {mode === "teach" && (resourceUploads.length > 0 || attachmentName) && (
            <div className="mt-2 flex max-h-24 flex-wrap gap-1.5 overflow-y-auto">
              {resourceUploads.map((item) => (
                <div key={item.localId} className={cn("inline-flex max-w-full items-center gap-2 rounded-lg border px-2 py-1 text-[11px]", item.status === "failed" ? "border-[var(--app-danger)] bg-[var(--app-danger-soft)] text-[var(--app-danger)]" : "border-[var(--app-border)] bg-[var(--app-surface)] text-[var(--app-muted)]")} title={item.detail}>
                  {item.status === "extracting" ? <LoaderCircle className="size-3 shrink-0 animate-spin text-[var(--app-accent)]" /> : item.status === "failed" ? <CircleAlert className="size-3 shrink-0" /> : <Check className="size-3 shrink-0 text-[var(--app-success)]" />}
                  <span className="max-w-52 truncate">{item.name}</span>
                  <span className="text-[9px] text-[var(--app-faint)]">{item.status === "extracting" ? "提取中" : item.status === "truncated" ? "已截断" : item.status === "failed" ? "失败" : "已就绪"}</span>
                  {item.removable !== false && <button type="button" onClick={() => onRemoveResource(item.localId)} aria-label={`移除 ${item.name}`} className="text-[var(--app-faint)] hover:text-[var(--app-text)]">×</button>}
                </div>
              ))}
              {attachmentName && <div className="inline-flex max-w-full items-center gap-2 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)] px-2 py-1 text-[11px] text-[var(--app-muted)]"><ImagePlus className="size-3 text-[var(--app-accent)]" /><span className="truncate">学生作答图片：{attachmentName}</span><button type="button" onClick={() => {setAttachment(null); setAttachmentName("");}} aria-label="移除图片" className="text-[var(--app-faint)] hover:text-[var(--app-text)]">×</button></div>}
            </div>
          )}
          {mode === "chat" && resourceUploads.length > 0 && (
            <div className="mt-2 flex max-h-28 flex-wrap gap-1.5 overflow-y-auto" aria-label="Chat 附件">
              {resourceUploads.map((item) => (
                <div key={item.localId} className={cn("inline-flex max-w-full items-center gap-2 rounded-lg border px-2 py-1 text-[11px]", item.status === "failed" || item.status === "blocked" ? "border-[var(--app-danger)] bg-[var(--app-danger-soft)] text-[var(--app-danger)]" : "border-[var(--app-border)] bg-[var(--app-surface)] text-[var(--app-muted)]")} title={item.detail}>
                  {item.status === "extracting" ? <LoaderCircle className="size-3 shrink-0 animate-spin text-[var(--app-accent)]" /> : item.status === "failed" || item.status === "blocked" ? <CircleAlert className="size-3 shrink-0" /> : <Check className="size-3 shrink-0 text-[var(--app-success)]" />}
                  <span className="max-w-52 truncate">{item.name}</span>
                  <span className="text-[9px] text-[var(--app-faint)]">{item.status === "extracting" ? `${chatAttachmentTypeLabel(item)} · 提取中` : item.status === "truncated" ? `${chatAttachmentTypeLabel(item)} · 已截断` : item.status === "blocked" ? "等待权威复核" : item.status === "failed" ? "失败" : `${chatAttachmentTypeLabel(item)} · 已就绪`}</span>
                  <button type="button" onClick={() => onRemoveResource(item.localId)} aria-label={`移除 ${item.name}`} className="text-[var(--app-faint)] hover:text-[var(--app-text)]">×</button>
                </div>
              ))}
            </div>
          )}
          {mode === "chat" && queuedChatPrompts.length > 0 && (
            <div className="mt-2 flex flex-wrap items-center gap-1.5" aria-label="等待发送的后续消息">
              <span className="text-[10px] text-[var(--app-faint)]">接下来</span>
              {queuedChatPrompts.map((prompt, index) => (
                <div key={prompt.id} className="inline-flex max-w-full items-center gap-1.5 rounded-full border border-[var(--app-border)] bg-[var(--app-surface)] px-2.5 py-1 text-[11px] text-[var(--app-muted)]">
                  <span className="text-[9px] text-[var(--app-accent-text)]">{index + 1} · 等待中</span>
                  <span className="max-w-64 truncate" title={prompt.text}>{prompt.text}</span>
                  <button type="button" onClick={() => onCancelQueuedPrompt(prompt.id)} aria-label={`取消后续消息：${prompt.text}`} className="text-[var(--app-faint)] hover:text-[var(--app-text)]">×</button>
                </div>
              ))}
            </div>
          )}
          <div ref={controlMenuRef} className="mt-2 flex flex-wrap items-center gap-1">
            {mode === "teach" && <>
              <input ref={resourceInputRef} type="file" multiple accept={resourceAccept} className="hidden" onChange={handleResourceChange} />
              <input ref={imageInputRef} type="file" accept="image/png,image/jpeg,image/webp" className="hidden" onChange={handleImageChange} />
              <Button type="button" variant="subtle" size="icon" disabled={interactionDisabled || busy || resourcesBusy} aria-label="导入教学资源" title="仅显示当前服务端确认可用的本地提取格式，也可导入教学大纲 JSON" onClick={() => resourceInputRef.current?.click()}><Plus className="size-4" /></Button>
              <Button type="button" variant="subtle" size="icon" disabled={interactionDisabled || busy || resourcesBusy || !localImageExtractionAvailable} aria-label={localImageExtractionAvailable ? canAttach ? "添加学生作答图片" : "导入教学图片" : "图片导入不可用：服务端未配置本地 OCR"} title={localImageExtractionAvailable ? canAttach ? "添加学生作答图片" : "导入教学图片（本机 OCR）" : "当前服务端没有可用的本地 OCR，因此不会接受图片"} onClick={() => imageInputRef.current?.click()}><ImagePlus className="size-4" /></Button>
            </>}
            {mode === "chat" && (
              <>
              <input ref={chatAttachmentInputRef} type="file" multiple accept={chatResourceAccept} className="hidden" onChange={handleChatAttachmentChange} />
              <Button type="button" variant="subtle" size="icon" disabled={interactionDisabled || busy || resourcesBusy || !chatResourceAccept} aria-label="添加 Chat 附件" title={chatResourceAccept ? "仅接受当前服务端确认可用的本地提取格式；原始媒体不发送给远程模型" : "当前服务端没有可用的 Chat 附件提取器"} onClick={() => chatAttachmentInputRef.current?.click()}><Plus className="size-4" /></Button>
              <Button
                type="button"
                variant="subtle"
                size="icon"
                disabled={interactionDisabled || busy || !webSearchAvailable}
                aria-label={webSearchEnabled ? "关闭联网搜索" : "开启联网搜索"}
                aria-pressed={webSearchEnabled}
                title={webSearchAvailable ? webSearchEnabled ? "联网搜索已开启：DeepSeek 会在需要最新信息时按需搜索" : "开启联网搜索" : "当前模型不支持联网搜索"}
                onClick={() => onWebSearchEnabledChange(!webSearchEnabled)}
                className={cn(webSearchEnabled && "bg-[var(--app-success-soft)] text-[var(--app-success)]")}
              >
                <Globe2 className="size-4" />
              </Button>
              </>
            )}
            <Button type="button" variant="subtle" size="icon" disabled aria-label="语音输入不可用：未配置受信任的本地 ASR" title="语音输入已关闭：浏览器语音服务可能远程处理学生音频；配置受信任的本地 ASR 后才可启用"><Mic className="size-4" /></Button>
            {mode === "teach" && <div className="relative ml-1">
              <button
                ref={controlMenuTriggerRef}
                type="button"
                aria-haspopup="menu"
                aria-expanded={controlMenuOpen}
                aria-controls="teachlab-control-menu"
                aria-label="选择教学控制模式"
                onClick={() => {setControlMenuOpen((value) => !value); setCommandOpen(false);}}
                disabled={busy || interactionDisabled}
                className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-[var(--app-muted)] hover:bg-[var(--app-hover)]"
              >
                {selectedSkill ? `手动 · ${selectedSkill.name}` : "自动教学"}<ChevronDown className="size-3" />
              </button>
              {controlMenuOpen && (
                <div ref={controlMenuPopupRef} id="teachlab-control-menu" role="menu" aria-label="教学控制模式" className="absolute bottom-[calc(100%+8px)] left-0 z-30 min-w-44 rounded-xl border border-[var(--app-border)] bg-[var(--app-overlay)] p-1.5 shadow-2xl">
                  <button
                    type="button"
                    role="menuitemradio"
                    aria-checked={!selectedSkillId}
                    tabIndex={!selectedSkillId ? 0 : -1}
                    onClick={() => chooseControlMode(null)}
                    onKeyDown={onMenuKeyDown}
                    className={cn("grid w-full gap-0.5 rounded-lg px-3 py-2 text-left text-xs", !selectedSkillId && "bg-[var(--app-selected)]")}
                  >
                    <span className="text-[var(--app-text)]">自动教学</span>
                    <span className="text-[10px] text-[var(--app-muted)]">根据回答自动选择下一步教学方法</span>
                  </button>
                  {primarySkills.map((skill) => (
                    <button
                      key={skill.skill_id}
                      type="button"
                      role="menuitemradio"
                      aria-checked={selectedSkillId === skill.skill_id}
                      tabIndex={selectedSkillId === skill.skill_id ? 0 : -1}
                      onClick={() => chooseControlMode(skill.skill_id)}
                      onKeyDown={onMenuKeyDown}
                      className={cn("grid w-full gap-0.5 rounded-lg px-3 py-2 text-left text-xs", selectedSkillId === skill.skill_id && "bg-[var(--app-selected)]")}
                    >
                      <span className="text-[var(--app-text)]">{skill.name}</span>
                      <span className="line-clamp-2 text-[10px] text-[var(--app-muted)]">{skill.selection_rationale || `手动锁定 ${skill.focus_dimension || "教学方法"}`}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>}
            <span className="ml-auto hidden items-center gap-1 text-xs text-[var(--app-muted)] sm:inline-flex">
              {busy ? mode === "chat" ? "Enter 加入后续 · Esc 停止" : "Esc 停止 · 可继续编辑草稿" : <><CornerDownLeft className="size-3" />发送 · Shift+Enter 换行</>}
            </span>
            <span className="rounded px-2 py-1 text-xs text-[var(--app-text)]" title="当前后端模型">DeepSeek</span>
            <Button
              type="button"
              size="icon"
              className={cn("bg-[var(--app-accent)] text-[#211714] hover:bg-[var(--app-accent-hover)]", busy && "bg-[var(--app-accent)]")}
              disabled={interactionDisabled || resourcesBusy || (busy ? false : !draft.trim() && !attachment && readyChatAttachmentCount === 0)}
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                if (busy) onStop();
                else submit();
              }}
              aria-label={busy ? "停止生成" : "发送"}
            >
              {busy ? <Square className="size-3 fill-current" /> : <Send className="size-4" />}
            </Button>
          </div>
        </div>
      </form>
    </main>
  );
}
