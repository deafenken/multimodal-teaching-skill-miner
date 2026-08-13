import type {SessionListItem, TeachingMessage} from "@/lib/types";

export const sessions: SessionListItem[] = [
  {id: "dp-dark-mode", title: "理解动态规划状态定义", learner: "小雨", status: "active", round: 3, group: "pinned"},
  {id: "weekly-review", title: "每周错题回顾", learner: "子墨", status: "active", round: 0, group: "scheduled"},
  {id: "binary-search", title: "二分边界为什么会错", learner: "知行", status: "succeeded", round: 7, group: "recent"},
  {id: "recursion", title: "递归与记忆化的区别", learner: "小雨", status: "active", round: 4, group: "recent"},
  {id: "transfer", title: "把状态转移迁移到新题", learner: "子墨", status: "terminated_unable", round: 6, group: "recent"}
];

export const initialMessages: TeachingMessage[] = [
  {
    id: "m1",
    role: "learner",
    body: "我能说出状态定义和状态转移，但不确定自己是不是真的理解了。",
    createdAt: "刚刚"
  },
  {
    id: "m2",
    role: "teacher",
    body: "很好，我们不直接提高掌握度。请用你自己的话解释：在爬楼梯问题里，dp[i] 表示什么？为什么它只依赖前两个状态？",
    createdAt: "刚刚",
    skill: "苏格拉底理解核验"
  },
  {
    id: "m3",
    role: "tool",
    body: "",
    createdAt: "刚刚",
    toolLabel: "读取 4 条会话证据，检查重复上限",
    toolDetail: "自解释证据明确 · 不读取 gold · 不提升掌握度"
  }
];

export const editorValue = `type VerificationGate = {
  explicitSelfExplanationEvidence: boolean
  priorSkillRepeatLimitReached: boolean
  diagnosis: "partial" | "ambiguous" | "correct"
}

export function allowSocraticVerification(gate: VerificationGate) {
  return gate.explicitSelfExplanationEvidence
    && gate.priorSkillRepeatLimitReached
    && ["partial", "ambiguous"].includes(gate.diagnosis)
}`;

export const editorOriginalValue = `type VerificationGate = {
  priorSkillRepeatLimitReached: boolean
  diagnosis: "partial" | "ambiguous" | "correct"
}

export function allowSocraticVerification(gate: VerificationGate) {
  return !gate.priorSkillRepeatLimitReached
    && gate.diagnosis === "correct"
}`;
