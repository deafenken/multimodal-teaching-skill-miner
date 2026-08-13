import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

const workspace = readFileSync(
  new URL("../components/workbench/syllabus-workspace.tsx", import.meta.url),
  "utf8",
);
const api = readFileSync(new URL("../lib/api.ts", import.meta.url), "utf8");
const route = readFileSync(
  new URL("../app/api/teacher-agent/[...path]/route.ts", import.meta.url),
  "utf8",
);

test("curriculum authority UI exposes graph review, explicit seal, revoke, and CAS", () => {
  for (const marker of [
    "课程测量蓝图 JSON",
    "复核此精确蓝图",
    "确认并密封评分权威",
    "撤销评分权威",
    "expected_syllabus_version",
    "expected_authority_version",
    "curriculum_authority_idempotency_key",
    "teacher_confirmed_authority: true",
    "window.confirm",
    "curriculumTeacherAuthorized",
    "服务端确认的认证教师权限",
  ]) {
    assert.match(workspace, new RegExp(marker.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }
  assert.match(workspace, /review_status === "sealed"/);
  assert.match(workspace, /authoritative_for_runtime_grading/);
  assert.match(workspace, /teacherAuthorityUiState/);
});

test("curriculum client and BFF expose only exact GET blueprint and POST authority routes", () => {
  assert.match(api, /fetchCurriculumBlueprint/);
  for (const action of ["review", "seal", "revoke"]) {
    assert.match(api, new RegExp(`api/curriculum/${action}`));
  }
  assert.match(route, /\^api\\\/curriculum\\\/\(\?:review\|seal\|revoke\)\$/);
  assert.match(route, /curriculumAuthorityPath && request\.method !== "POST"/);
  assert.match(route, /curriculumBlueprintPath && request\.method !== "GET" && request\.method !== "HEAD"/);
});
