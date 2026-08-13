import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

import {
  buildResourceReviewRequest,
  replaceMatchingTeachingResource,
  resourceReviewAvailability,
  safeReviewedResource,
  uniqueTeachingResources,
  type ResourceReviewDraft,
} from "../lib/resource-review.ts";
import type {
  BootstrapPayload,
  TeachingResourceReviewResponse,
  TeachingResourceSummary,
} from "../lib/types.ts";

const HASH = "a".repeat(64);
const ORIGINAL_HASH = "b".repeat(64);

function resource(overrides: Partial<TeachingResourceSummary> = {}): TeachingResourceSummary {
  return {
    resource_id: `res_${"1".repeat(20)}`,
    staged_resource_id: `stage_${"2".repeat(24)}`,
    display_name: "lesson.pptx",
    content_sha256: HASH,
    original_resource_sha256: ORIGINAL_HASH,
    needs_review: true,
    requires_confirmation: true,
    grading_evidence_allowed: false,
    mastery_evidence_allowed: false,
    evidence_contract: {
      layers: [
        {layer_id: "layer_001", kind: "text_transcription", status: "candidate"},
        {layer_id: "layer_002", kind: "speaker_notes", status: "candidate"},
      ],
      conflicts: [{
        conflict_id: "resource_conflict_001",
        kind: "slide_notes_disagreement",
        description: "正文与讲者备注不一致",
      }],
      decision: "requires_confirmation",
      grading_evidence_allowed: false,
      mastery_evidence_allowed: false,
    },
    review_requirements: {
      schema: "teaching_skill_miner.resource_review_requirements.v1",
      original_resource_sha256: ORIGINAL_HASH,
      conflicts: [{
        conflict_id: "resource_conflict_001",
        kind: "slide_notes_disagreement",
        description: "正文与讲者备注不一致",
      }],
      layers: [
        {layer_id: "layer_001", kind: "text_transcription", status: "candidate"},
        {layer_id: "layer_002", kind: "speaker_notes", status: "candidate"},
      ],
      required_resolved_conflict_ids: ["resource_conflict_001"],
      reviewable_layer_ids: ["layer_001", "layer_002"],
      raw_media_included: false,
      answer_key_or_grading_authority_included: false,
    },
    resource_review: {
      reviewed: false,
      review_version: 0,
      review_id: null,
      review_scope: "untrusted_teaching_context_only",
      semantic_understanding_established: false,
      grading_evidence_allowed: false,
      mastery_evidence_allowed: false,
    },
    ...overrides,
  };
}

const authenticatedAuthority: NonNullable<BootstrapPayload["teacher_authority"]> = {
  mode: "authenticated_apps_api",
  role_authorized: true,
  correct_mastery_updates_enabled: false,
  assurance: "deployment_service_role_authorization_not_personal_signature",
  raw_identity_exposed: false,
};

function completeDraft(overrides: Partial<ResourceReviewDraft> = {}): ResourceReviewDraft {
  return {
    reviewedText: "教师已对照原始讲义确认：正文描述三个步骤。",
    excludedLayerIds: ["layer_002"],
    attestations: {
      compared_with_original_source: true,
      uncertainties_removed_or_explicit: true,
      not_an_answer_key: true,
      context_only: true,
    },
    reviewNote: "已对照原始演示文稿。",
    ...overrides,
  };
}

test("resource review UI fails closed outside an authenticated authorized teacher role", () => {
  assert.equal(resourceReviewAvailability(authenticatedAuthority, resource()).enabled, true);
  const local = resourceReviewAvailability({
    mode: "local_python",
    role_authorized: false,
    correct_mastery_updates_enabled: false,
    assurance: "no_authenticated_teacher_identity",
    raw_identity_exposed: false,
  }, resource());
  assert.equal(local.enabled, false);
  assert.match(local.reason, /本机模式.*Apps API/);
  assert.equal(resourceReviewAvailability({...authenticatedAuthority, role_authorized: false}, resource()).enabled, false);
  assert.equal(resourceReviewAvailability(authenticatedAuthority, resource({review_requirements: undefined})).enabled, false);
});

test("review payload has the exact bounded contract and resolves every immutable conflict", () => {
  const reviewed = resource({
    resource_review: {
      reviewed: true,
      review_version: 2,
      review_id: `rrp_${"3".repeat(24)}`,
      review_scope: "untrusted_teaching_context_only",
      semantic_understanding_established: false,
      grading_evidence_allowed: false,
      mastery_evidence_allowed: false,
    },
    // The current projection has no conflicts. Review requirements still bind
    // the next revision to every conflict in the immutable original.
    evidence_contract: {layers: [], conflicts: [], decision: "usable_as_untrusted_teaching_context"},
  });
  const payload = buildResourceReviewRequest(
    reviewed,
    completeDraft({reviewedText: "  教师已对照原始讲义确认：正文描述三个步骤。  ", reviewNote: "  已核对。  "}),
    "console-resource-review-key-0001",
  );
  assert.deepEqual(Object.keys(payload).sort(), [
    "attestations",
    "content_sha256",
    "excluded_layer_ids",
    "expected_review_version",
    "original_resource_sha256",
    "resolved_conflict_ids",
    "resource_id",
    "resource_review_idempotency_key",
    "review_note",
    "reviewed_text",
    "staged_resource_id",
  ]);
  assert.equal(payload.expected_review_version, 2);
  assert.deepEqual(payload.resolved_conflict_ids, ["resource_conflict_001"]);
  assert.deepEqual(payload.excluded_layer_ids, ["layer_002"]);
  assert.equal(payload.reviewed_text, "教师已对照原始讲义确认：正文描述三个步骤。");
  assert.equal(payload.review_note, "已核对。");
  assert.deepEqual(payload.attestations, {
    compared_with_original_source: true,
    uncertainties_removed_or_explicit: true,
    not_an_answer_key: true,
    context_only: true,
  });
  for (const forbidden of ["data_base64", "raw_media", "answer_key", "actor", "authority_receipt", "grading_evidence_allowed", "mastery_evidence_allowed"]) {
    assert.equal(forbidden in payload, false, forbidden);
  }
});

test("client validation blocks unresolved markers, incomplete attestations, and invented layers", () => {
  assert.throws(() => buildResourceReviewRequest(
    resource(),
    completeDraft({reviewedText: "[视觉复核：以后再看]"}),
    "console-resource-review-key-0002",
  ), /待确认标记/);
  assert.throws(() => buildResourceReviewRequest(
    resource(),
    completeDraft({attestations: {...completeDraft().attestations, not_an_answer_key: false}}),
    "console-resource-review-key-0003",
  ), /四项教师声明/);
  assert.throws(() => buildResourceReviewRequest(
    resource(),
    completeDraft({excludedLayerIds: ["layer_attacker"]}),
    "console-resource-review-key-0004",
  ), /证据层已经变化/);
});

test("review response is applied only when every context-only boundary remains false", () => {
  const original = resource();
  const reviewed = resource({
    needs_review: false,
    requires_confirmation: false,
    resource_review: {
      reviewed: true,
      review_version: 1,
      review_id: `rrp_${"3".repeat(24)}`,
      review_scope: "untrusted_teaching_context_only",
      semantic_understanding_established: false,
      grading_evidence_allowed: false,
      mastery_evidence_allowed: false,
    },
  });
  const response: TeachingResourceReviewResponse = {
    schema: "teaching_skill_miner.dashboard_resource_review.v1",
    resource: reviewed,
    session_use: {
      status: "eligible_untrusted_context",
      decision: "usable_as_untrusted_teaching_context",
      reason: null,
      grading_evidence_allowed: false,
      mastery_evidence_allowed: false,
    },
    original_resource_immutable: true,
    raw_media_sent: false,
    review_scope: "untrusted_teaching_context_only",
    semantic_understanding_established: false,
    grading_evidence_allowed: false,
    mastery_evidence_allowed: false,
  };
  assert.equal(safeReviewedResource(response, original), reviewed);
  assert.throws(() => safeReviewedResource({
    ...response,
    resource: {...reviewed, grading_evidence_allowed: true} as unknown as TeachingResourceSummary,
  }, original), /安全边界/);
});

test("reviewed metadata replaces every matching resource without duplicating identities", () => {
  const original = resource();
  const reviewed = resource({needs_review: false});
  const unrelated = resource({resource_id: `res_${"4".repeat(20)}`, display_name: "other.pdf"});
  assert.equal(replaceMatchingTeachingResource(original, reviewed), reviewed);
  assert.equal(replaceMatchingTeachingResource(unrelated, reviewed), unrelated);
  assert.deepEqual(uniqueTeachingResources([original, unrelated, reviewed]), [reviewed, unrelated]);
});

test("Inspector exposes conflicts, layers, four attestations, local disable reason, and exact review API", () => {
  const inspector = readFileSync(new URL("../components/workbench/inspector-drawer.tsx", import.meta.url), "utf8");
  const api = readFileSync(new URL("../lib/api.ts", import.meta.url), "utf8");
  const route = readFileSync(new URL("../app/api/teacher-agent/[...path]/route.ts", import.meta.url), "utf8");
  const harnessBff = readFileSync(new URL("../lib/harness-bff.ts", import.meta.url), "utf8");
  const workbench = readFileSync(new URL("../components/workbench/workbench.tsx", import.meta.url), "utf8");
  const projectLibrary = readFileSync(new URL("../components/workbench/project-library.tsx", import.meta.url), "utf8");
  for (const token of [
    "待处理冲突",
    "证据层",
    "四项教师声明（全部必选）",
    "原文件、图片、音视频不会随复核请求发送",
    "不是答案键、评分或掌握度证据",
    "RESOURCE_REVIEW_MAX_TEXT_CHARS",
    "onResourceReviewed",
  ]) assert.ok(inspector.includes(token), `missing resource review UI contract: ${token}`);
  assert.ok(api.includes("`${pythonProxyBase}/api/resource/review`"));
  assert.match(route, /joinedPath === "api\/resource\/review"/);
  assert.match(route, /joinedPath === "api\/resource\/review" && request\.method !== "POST"/);
  assert.match(harnessBff, /path === "api\/resource\/review"/);
  for (const token of ["setResourceUploads", "updateChatResourceUploads", "setActiveSession", "setHistorySessions", "applyReviewedResource"]) {
    assert.ok(workbench.includes(token), `missing reviewed resource propagation: ${token}`);
  }
  assert.ok(projectLibrary.includes("onOpenResource(item.reference_id, item.metadata)"));
  assert.ok(workbench.includes("selectedProjectResource"));
});
