import type {LearningProject} from "@/lib/types";

export function projectScopedItems<T extends {session_id: string}>(items: T[], project: LearningProject | null) {
  if (!project) return [];
  const allowed = new Set(project.teaching_session_ids);
  return items.filter((item) => allowed.has(item.session_id));
}

export function nextVisibleCount(current: number, total: number, pageSize: number) {
  if (!Number.isInteger(current) || !Number.isInteger(total) || !Number.isInteger(pageSize) || current < 0 || total < 0 || pageSize < 1) {
    throw new Error("visible pagination bounds are invalid");
  }
  return Math.min(total, current + pageSize);
}
