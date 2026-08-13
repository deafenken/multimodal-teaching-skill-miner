export type CommandCenterActionId =
  | "new_current"
  | "open_chat"
  | "open_teach"
  | "open_syllabi"
  | "open_settings"
  | "show_tasks"
  | "stop_chat"
  | "stop_teach";

export interface CommandCenterAction {
  id: CommandCenterActionId;
  title: string;
  detail: string;
  keywords: string[];
  danger?: boolean;
}

export function commandCenterActions({chatRunning, teachRunning}: {
  chatRunning: boolean;
  teachRunning: boolean;
}): CommandCenterAction[] {
  const actions: CommandCenterAction[] = [
    {id: "new_current", title: "新建当前对话", detail: "在当前 Chat 或 Teach 模式创建新会话", keywords: ["new", "新建", "会话"]},
    {id: "open_chat", title: "打开 Chat", detail: "切换到直接对话工作区", keywords: ["chat", "聊天", "对话"]},
    {id: "open_teach", title: "打开 Teach", detail: "切换到自适应教学工作区", keywords: ["teach", "教学", "学习"]},
    {id: "open_syllabi", title: "打开课程大纲", detail: "浏览、修订和启动课节", keywords: ["syllabus", "课程", "大纲"]},
    {id: "show_tasks", title: "后台任务", detail: "查看、恢复或停止持久 Harness 任务", keywords: ["task", "run", "任务", "运行", "恢复"]},
    {id: "open_settings", title: "设置与同意中心", detail: "主题、可访问性和远程处理授权", keywords: ["settings", "consent", "设置", "同意", "授权"]},
  ];
  if (chatRunning) actions.push({id: "stop_chat", title: "停止 Chat", detail: "显式取消当前 Chat；后台 Teach 不受影响", keywords: ["stop", "cancel", "停止", "取消"], danger: true});
  if (teachRunning) actions.push({id: "stop_teach", title: "停止 Teach", detail: "显式取消当前 Teach；后台 Chat 不受影响", keywords: ["stop", "cancel", "停止", "取消"], danger: true});
  return actions;
}

function normalized(value: string) {
  return value.normalize("NFKC").trim().toLocaleLowerCase();
}

export function filterCommandCenterActions(actions: CommandCenterAction[], query: string) {
  const needle = normalized(query);
  if (!needle) return actions;
  return actions.filter((action) => normalized([
    action.title,
    action.detail,
    ...action.keywords,
  ].join(" ")).includes(needle));
}

export const backgroundTaskStatusLabel: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  cancel_requested: "正在停止",
  suspended: "已暂停",
  completed: "已完成",
  cancelled: "已停止",
  failed: "失败",
  handoff: "需人工接管",
};

export function relativeTaskTime(value: string, now = Date.now()) {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return "时间未知";
  const seconds = Math.max(0, Math.floor((now - timestamp) / 1_000));
  if (seconds < 60) return `${seconds} 秒前`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  return `${Math.floor(hours / 24)} 天前`;
}
