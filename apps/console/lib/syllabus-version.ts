import type {SyllabusVersionFamily} from "./types.ts";

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function text(value: unknown) {
  return typeof value === "string" ? value : "";
}

export function normalizeSyllabusVersionFamily(value: unknown): SyllabusVersionFamily {
  const envelope = record(value);
  const raw = record(envelope.version_family ?? envelope.family ?? value);
  const revisions = Array.isArray(raw.revisions) ? raw.revisions.map(record) : [];
  const actor = record(raw.actor_boundary);
  if (
    raw.schema !== "teaching_skill_miner.syllabus_version_projection.v1"
    || !/^syf_[0-9a-f]{24}$/.test(text(raw.family_id))
    || typeof raw.version !== "number"
    || !Number.isInteger(raw.version)
    || raw.version < 1
    || !/^syr_[0-9a-f]{24}$/.test(text(raw.published_revision_id))
    || !revisions.length
    || revisions.some((revision, index) => (
      !/^syr_[0-9a-f]{24}$/.test(text(revision.revision_id))
      || revision.revision_number !== index + 1
      || !/^syl_[0-9a-f]{24}$/.test(text(revision.syllabus_id))
      || !/^[0-9a-f]{64}$/.test(text(revision.content_sha256))
      || !text(revision.created_at_utc)
      || !text(revision.change_summary)
      || !["published", "draft", "historical"].includes(text(revision.status))
    ))
    || revisions.filter((revision) => revision.status === "published").length !== 1
    || revisions.find((revision) => revision.status === "published")?.revision_id !== raw.published_revision_id
    || actor.actor_kind !== "local_operator_not_authenticated"
    || actor.authenticated !== false
    || actor.teacher_identity_claimed !== false
  ) {
    throw new Error("教学后端返回了无效的大纲版本账本，请停止编辑并检查本地存储。");
  }
  return raw as unknown as SyllabusVersionFamily;
}
