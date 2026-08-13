import assert from "node:assert/strict";
import test from "node:test";

import {chatBranchTitle, chatTurnsFromMessages, prepareCompletedTurnBranch, prepareTurnRetry, settleTurn, storedTeachingMessage} from "../lib/conversation-state.ts";
import {openSourcePreview, safeMarkdownUrl, SOURCE_PREVIEW_WINDOW} from "../lib/markdown-security.ts";
import {invalidateLegacyRemoteConsent, legacyRemoteConsentStorageKey} from "../lib/remote-consent.ts";
import type {TeachingMessage} from "../lib/types.ts";

const messages: TeachingMessage[] = [
  {id: "u1", role: "learner", body: "第一问", createdAt: "刚刚"},
  {id: "a1", role: "teacher", body: "第一答", createdAt: "刚刚", status: "completed"},
  {id: "u2", role: "learner", body: "第二问", createdAt: "刚刚"},
  {id: "a2", role: "teacher", body: "半截", createdAt: "刚刚", status: "failed", error: "网络中断"},
];

test("failed and stopped responses never enter the next Chat model context", () => {
  assert.deepEqual(chatTurnsFromMessages(messages), [
    {role: "user", content: "第一问"},
    {role: "assistant", content: "第一答"},
    {role: "user", content: "第二问"},
  ]);
});

test("Chat forwards up to the backend compaction window instead of silently dropping at 22 turns", () => {
  const history: TeachingMessage[] = Array.from({length: 100}, (_, index) => ({
    id: `m${index}`,
    role: index % 2 === 0 ? "learner" : "teacher",
    body: `message-${index}`,
    createdAt: "历史",
    status: "completed",
  }));
  const turns = chatTurnsFromMessages(history);
  assert.equal(turns.length, 100);
  assert.deepEqual(turns[0], {role: "user", content: "message-0"});
  assert.deepEqual(turns.at(-1), {role: "assistant", content: "message-99"});
});

test("a retry replaces the failed response in place and clears old lifecycle rows", () => {
  const withLifecycle: TeachingMessage[] = [
    ...messages,
    {id: "a2-harness-tool", role: "tool", body: "", createdAt: "刚刚", toolLabel: "搜索"},
  ];
  const retry = prepareTurnRetry(withLifecycle, "a2", "正在重试");
  assert.equal(retry.length, messages.length);
  assert.deepEqual(retry.at(-1), {
    id: "a2",
    role: "teacher",
    body: "",
    createdAt: "刚刚",
    status: "running",
    error: undefined,
    retryable: true,
    streaming: true,
    thinking: "正在重试",
  });
  assert.equal(settleTurn(retry, "a2", "completed", {body: "完整回答"}).at(-1)?.status, "completed");
});

test("regenerating a completed Chat answer creates an immutable branch prefix", () => {
  const original: TeachingMessage[] = [
    {id: "u1", role: "learner", body: "第一问", createdAt: "历史"},
    {id: "a1-harness-search", role: "tool", body: "", toolLabel: "搜索", createdAt: "历史"},
    {id: "a1", role: "teacher", body: "原回答", createdAt: "历史", status: "completed", webSearchUsed: true, sources: [{title: "来源", url: "https://example.com"}]},
    {id: "u2", role: "learner", body: "后续问题", createdAt: "历史"},
  ];
  const branch = prepareCompletedTurnBranch(original, "a1", "a1-branch", "正在生成分支");
  assert.deepEqual(branch?.map((message) => message.id), ["u1", "a1-branch"]);
  assert.deepEqual(branch?.at(-1), {
    id: "a1-branch",
    role: "teacher",
    body: "",
    createdAt: "刚刚",
    status: "running",
    webSearchUsed: false,
    sources: undefined,
    streaming: true,
    thinking: "正在生成分支",
    error: undefined,
    retryable: true,
  });
  assert.equal(original[2]?.body, "原回答");
  assert.equal(chatBranchTitle("动态规划"), "动态规划 · 分支");
  assert.ok(chatBranchTitle("x".repeat(200)).length <= 160);
});

test("a browser reload converts an unfinished running turn into a stopped turn", () => {
  const stored = storedTeachingMessage({id: "a", role: "teacher", body: "部分", createdAt: "刚刚", status: "running", streaming: true});
  assert.equal(stored?.status, "stopped");
  assert.equal(stored?.streaming, false);
  assert.equal(stored?.retryable, false);
});

test("Markdown links allow navigation but reject executable and data URLs", () => {
  assert.equal(safeMarkdownUrl("https://example.com/a"), "https://example.com/a");
  assert.equal(safeMarkdownUrl("mailto:teacher@example.com"), "mailto:teacher@example.com");
  assert.equal(safeMarkdownUrl("#section"), "#section");
  assert.equal(safeMarkdownUrl("/lesson/1"), "/lesson/1");
  assert.equal(safeMarkdownUrl("javascript:alert(1)"), "");
  assert.equal(safeMarkdownUrl("data:text/html,unsafe"), "");
  assert.equal(SOURCE_PREVIEW_WINDOW, "teachlab-source-preview");
  assert.equal(openSourcePreview("https://example.com/a"), false);
});

test("legacy browser consent is invalidated and never upgraded", () => {
  const values = new Map([[legacyRemoteConsentStorageKey, JSON.stringify({version: 1, grants: {chat: "yesterday"}})]]);
  const storage = {
    getItem: (key: string) => values.get(key) ?? null,
    removeItem: (key: string) => { values.delete(key); },
  };
  assert.equal(invalidateLegacyRemoteConsent(storage), true);
  assert.equal(values.has(legacyRemoteConsentStorageKey), false);
  assert.equal(invalidateLegacyRemoteConsent(storage), false);
});
