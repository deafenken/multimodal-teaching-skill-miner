import type {
  BootstrapPayload,
  TeachingResourceReviewRequest,
  TeachingResourceReviewResponse,
  TeachingResourceSummary,
} from "@/lib/types";

export const RESOURCE_REVIEW_MAX_TEXT_CHARS = 12_000;
export const RESOURCE_REVIEW_MAX_NOTE_CHARS = 1_000;

const RESOURCE_ID = /^res_[0-9a-f]{20}$/;
const STAGED_RESOURCE_ID = /^stage_[0-9a-f]{24}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const IDEMPOTENCY_KEY = /^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$/;
const UNRESOLVED_MARKERS = [
  "[内容冲突待确认：",
  "[视觉复核：",
  "[公式转写候选：",
  "[扫描页检测：",
] as const;

export type ResourceReviewAttestationKey =
  | "compared_with_original_source"
  | "uncertainties_removed_or_explicit"
  | "not_an_answer_key"
  | "context_only";

export interface ResourceReviewDraft {
  reviewedText: string;
  excludedLayerIds: string[];
  attestations: Record<ResourceReviewAttestationKey, boolean>;
  reviewNote: string;
}

export interface ResourceReviewAvailability {
  enabled: boolean;
  reason: string;
}

export function resourceReviewRequirements(resource: TeachingResourceSummary) {
  const requirements = resource.review_requirements;
  const conflictIds = requirements?.conflicts?.map((item) => item?.conflict_id);
  const layerIds = requirements?.layers?.map((item) => item?.layer_id);
  if (
    requirements?.schema === "teaching_skill_miner.resource_review_requirements.v1"
    && SHA256.test(requirements.original_resource_sha256)
    && requirements.raw_media_included === false
    && requirements.answer_key_or_grading_authority_included === false
    && Array.isArray(requirements.layers)
    && Array.isArray(requirements.conflicts)
    && Array.isArray(requirements.required_resolved_conflict_ids)
    && Array.isArray(requirements.reviewable_layer_ids)
    && conflictIds?.every((id) => typeof id === "string" && id.length > 0)
    && layerIds?.every((id) => typeof id === "string" && id.length > 0)
    && new Set(conflictIds).size === conflictIds?.length
    && new Set(layerIds).size === layerIds?.length
    && requirements.required_resolved_conflict_ids.length === conflictIds?.length
    && requirements.required_resolved_conflict_ids.every((id) => conflictIds?.includes(id))
    && requirements.reviewable_layer_ids.length === layerIds?.length
    && requirements.reviewable_layer_ids.every((id) => layerIds?.includes(id))
  ) return requirements;
  return null;
}

export function resourceReviewAvailability(
  authority: BootstrapPayload["teacher_authority"],
  resource: TeachingResourceSummary,
): ResourceReviewAvailability {
  if (authority?.mode === "local_python") {
    return {
      enabled: false,
      reason: "本机模式没有经过认证的教师身份；请连接已配置教师角色的 Apps API 后复核。",
    };
  }
  if (authority?.mode !== "authenticated_apps_api") {
    return {enabled: false, reason: "教师认证状态不可用，资源复核已安全关闭。"};
  }
  if (authority.role_authorized !== true) {
    return {enabled: false, reason: "当前组织账号没有教师角色，不能提交资源复核。"};
  }
  if (
    !RESOURCE_ID.test(resource.resource_id)
    || !STAGED_RESOURCE_ID.test(resource.staged_resource_id ?? "")
    || !SHA256.test(resource.content_sha256 ?? "")
    || !SHA256.test(resource.original_resource_sha256 ?? "")
  ) {
    return {enabled: false, reason: "后端没有返回完整的不可变资源绑定，不能安全复核。"};
  }
  const requirements = resourceReviewRequirements(resource);
  if (!requirements || requirements.original_resource_sha256 !== resource.original_resource_sha256) {
    return {enabled: false, reason: "后端没有返回不可变资源的复核要求，不能安全复核。"};
  }
  return {enabled: true, reason: "认证教师可以提交仅用于教学上下文的复核文本。"};
}

export function newResourceReviewIdempotencyKey() {
  return `console-resource-review-${crypto.randomUUID()}`;
}

export function buildResourceReviewRequest(
  resource: TeachingResourceSummary,
  draft: ResourceReviewDraft,
  idempotencyKey: string,
): TeachingResourceReviewRequest {
  const resourceId = resource.resource_id;
  const stagedResourceId = resource.staged_resource_id ?? "";
  const contentSha256 = resource.content_sha256 ?? "";
  const originalResourceSha256 = resource.original_resource_sha256 ?? "";
  if (
    !RESOURCE_ID.test(resourceId)
    || !STAGED_RESOURCE_ID.test(stagedResourceId)
    || !SHA256.test(contentSha256)
    || !SHA256.test(originalResourceSha256)
  ) throw new Error("资源的不可变绑定不完整，请刷新后重试。");
  if (!IDEMPOTENCY_KEY.test(idempotencyKey)) throw new Error("资源复核请求标识无效。");

  const reviewedText = draft.reviewedText.trim();
  if (!reviewedText) throw new Error("请输入与原始资料逐项核对后的文本。");
  if (reviewedText.length > RESOURCE_REVIEW_MAX_TEXT_CHARS) {
    throw new Error(`复核文本不能超过 ${RESOURCE_REVIEW_MAX_TEXT_CHARS.toLocaleString()} 字。`);
  }
  if (reviewedText.includes("\0") || UNRESOLVED_MARKERS.some((marker) => reviewedText.includes(marker))) {
    throw new Error("复核文本仍包含待确认标记，请先核对并明确处理不确定内容。");
  }

  const reviewNote = draft.reviewNote.trim();
  if (reviewNote.length > RESOURCE_REVIEW_MAX_NOTE_CHARS) {
    throw new Error(`复核备注不能超过 ${RESOURCE_REVIEW_MAX_NOTE_CHARS.toLocaleString()} 字。`);
  }
  if (Object.values(draft.attestations).some((value) => value !== true)) {
    throw new Error("请完成四项教师声明后再提交。");
  }

  const requirements = resourceReviewRequirements(resource);
  if (!requirements || requirements.original_resource_sha256 !== originalResourceSha256) {
    throw new Error("不可变资源的复核要求缺失，请刷新后重试。");
  }
  const knownLayerIds = new Set(requirements.reviewable_layer_ids);
  const excludedLayerIds = Array.from(new Set(draft.excludedLayerIds));
  if (excludedLayerIds.some((layerId) => !knownLayerIds.has(layerId))) {
    throw new Error("排除的证据层已经变化，请刷新后重新选择。");
  }
  const resolvedConflictIds = [...requirements.required_resolved_conflict_ids];
  if (new Set(resolvedConflictIds).size !== resolvedConflictIds.length) {
    throw new Error("资源冲突标识重复，请刷新后重试。");
  }
  const expectedReviewVersion = resource.resource_review?.review_version ?? 0;
  if (!Number.isInteger(expectedReviewVersion) || expectedReviewVersion < 0 || expectedReviewVersion >= 16) {
    throw new Error("资源复核版本无效，请刷新后重试。");
  }

  return {
    resource_id: resourceId,
    staged_resource_id: stagedResourceId,
    content_sha256: contentSha256,
    expected_review_version: expectedReviewVersion,
    resource_review_idempotency_key: idempotencyKey,
    original_resource_sha256: originalResourceSha256,
    reviewed_text: reviewedText,
    resolved_conflict_ids: resolvedConflictIds,
    excluded_layer_ids: excludedLayerIds,
    attestations: {
      compared_with_original_source: true,
      uncertainties_removed_or_explicit: true,
      not_an_answer_key: true,
      context_only: true,
    },
    review_note: reviewNote,
  };
}

export function replaceMatchingTeachingResource(
  current: TeachingResourceSummary,
  reviewed: TeachingResourceSummary,
) {
  return current.resource_id === reviewed.resource_id ? reviewed : current;
}

export function safeReviewedResource(
  response: TeachingResourceReviewResponse,
  expected: TeachingResourceSummary,
) {
  const reviewed = response.resource;
  if (
    response.schema !== "teaching_skill_miner.dashboard_resource_review.v1"
    || response.original_resource_immutable !== true
    || response.raw_media_sent !== false
    || response.review_scope !== "untrusted_teaching_context_only"
    || response.semantic_understanding_established !== false
    || response.grading_evidence_allowed !== false
    || response.mastery_evidence_allowed !== false
    || response.session_use?.status !== "eligible_untrusted_context"
    || response.session_use.grading_evidence_allowed !== false
    || response.session_use.mastery_evidence_allowed !== false
    || reviewed.resource_id !== expected.resource_id
    || reviewed.staged_resource_id !== expected.staged_resource_id
    || reviewed.content_sha256 !== expected.content_sha256
    || reviewed.original_resource_sha256 !== expected.original_resource_sha256
    || reviewed.resource_review?.reviewed !== true
    || reviewed.resource_review.review_scope !== "untrusted_teaching_context_only"
    || reviewed.resource_review.semantic_understanding_established !== false
    || reviewed.resource_review.grading_evidence_allowed !== false
    || reviewed.resource_review.mastery_evidence_allowed !== false
    || reviewed.grading_evidence_allowed !== false
    || reviewed.mastery_evidence_allowed !== false
  ) throw new Error("资源复核响应越过了仅供教学上下文的安全边界，已拒绝应用。");
  return reviewed;
}

export function uniqueTeachingResources(resources: TeachingResourceSummary[]) {
  const byId = new Map<string, TeachingResourceSummary>();
  for (const resource of resources) byId.set(resource.resource_id, resource);
  return Array.from(byId.values());
}
