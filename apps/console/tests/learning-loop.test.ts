import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

const api = readFileSync(new URL("../lib/api.ts", import.meta.url), "utf8");
const inspector = readFileSync(new URL("../components/workbench/inspector-drawer.tsx", import.meta.url), "utf8");
const workbench = readFileSync(new URL("../components/workbench/workbench.tsx", import.meta.url), "utf8");

function functionSource(name: string, nextName: string) {
  const start = api.indexOf(`export function ${name}`);
  const end = api.indexOf(`export function ${nextName}`, start + 1);
  assert.ok(start >= 0 && end > start, `missing API function boundary for ${name}`);
  return api.slice(start, end);
}

test("learner review and JOL mutations never accept client outcome or mastery", () => {
  const claimSource = functionSource("claimDueLearningReview", "releaseLearningReview");
  const releaseSource = functionSource("releaseLearningReview", "listMetacognitionPredictions");
  const clientMutationSources = [
    claimSource,
    releaseSource,
    functionSource("recordMetacognitionPrediction", "pairMetacognitionPrediction"),
    functionSource("pairMetacognitionPrediction", "listAdjudicationReviews"),
  ].join("\n");
  for (const forbidden of ["outcome:", "actual_score_percent:", "mastery:", "p_mastery:"]) {
    assert.equal(clientMutationSources.includes(forbidden), false, `client authority field leaked: ${forbidden}`);
  }
  for (const required of ["review_id:", "lease_id:", "expected_version:", "prediction_event_id:"]) {
    assert.ok(clientMutationSources.includes(required), `missing server binding: ${required}`);
  }
  assert.ok(claimSource.includes("${reviewId}-${expectedVersion}"), "claim retry identity must be stable across a lost response");
  assert.ok(releaseSource.includes("${payload.leaseId}-${payload.expectedVersion}"), "release retry identity must be stable across a lost response");
  assert.equal(claimSource.includes("crypto.randomUUID()"), false);
  assert.equal(releaseSource.includes("crypto.randomUUID()"), false);
});

test("inspector opens the server review session and keeps its lease recoverable", () => {
  for (const token of [
    "答题前自评（JOL）",
    "准备使用的策略（1–3 个）",
    "刷新并自动配对",
    "服务端已提交的教学回合收据",
    "没有客户端结果、掌握度或评分输入",
    "onSessionChange(claimed.review_session)",
    "开始复习",
    "releaseOneLearningReview(session, review)",
  ]) assert.ok(inspector.includes(token), `missing learning-loop contract: ${token}`);
  for (const unsafeAutoRelease of [
    "releaseAllLearningReviews(previous)",
    "if (!open) void releaseAllLearningReviews",
  ]) assert.equal(inspector.includes(unsafeAutoRelease), false, `review lease must survive UI lifecycle: ${unsafeAutoRelease}`);
  for (const token of [
    "activateSessionFromInspector",
    "setActiveSession(session)",
    "setTeachingMessages(messagesFromSession(session))",
    "onSessionChange={activateSessionFromInspector}",
  ]) assert.ok(workbench.includes(token), `missing review-session activation: ${token}`);
  assert.equal(inspector.includes("setMastery"), false);
  assert.equal(inspector.includes("setOutcome"), false);
});
