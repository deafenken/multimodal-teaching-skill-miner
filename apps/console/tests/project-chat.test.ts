import assert from "node:assert/strict";
import test from "node:test";

import {chatThreadFromProject, projectChatThread} from "../lib/project-chat.ts";

test("project Chat serialization preserves roles, turn states, web search and sources", () => {
  const encoded = projectChatThread({
    id: "chat_0123456789abcdef01234567",
    title: "  动态规划  ",
    createdAt: Date.parse("2026-08-12T01:00:00Z"),
    updatedAt: Date.parse("2026-08-12T01:01:00Z"),
    messages: [
      {id: "user-1", role: "learner", body: "什么是状态转移？", createdAt: "2026-08-12T01:00:00Z"},
      {id: "assistant-1", role: "teacher", body: "先看一个例子。", createdAt: "2026-08-12T01:01:00Z", status: "completed", webSearchUsed: true, sources: [{title: "课程资料", url: "https://example.com/course"}]},
    ],
  });
  assert.equal(encoded.title, "动态规划");
  assert.deepEqual(encoded.messages.map(({role, status, web_search_used}) => ({role, status, web_search_used})), [
    {role: "user", status: "completed", web_search_used: false},
    {role: "assistant", status: "completed", web_search_used: true},
  ]);
  assert.deepEqual(encoded.messages[1]?.sources, [{title: "课程资料", url: "https://example.com/course"}]);
});

test("project Chat persistence excludes transient harness lifecycle rows", () => {
  const serialized = projectChatThread({
    id: "chat_00112233445566778899aabb",
    title: "Lifecycle boundary",
    createdAt: Date.parse("2026-08-12T00:00:00Z"),
    updatedAt: Date.parse("2026-08-12T00:01:00Z"),
    messages: [
      {id: "u1", role: "learner", body: "问题", createdAt: "2026-08-12T00:00:00Z"},
      {id: "a1-harness-search", role: "tool", body: "", toolLabel: "联网搜索", toolDetail: "正在运行", createdAt: "刚刚"},
      {id: "a1", role: "teacher", body: "回答", createdAt: "2026-08-12T00:01:00Z", status: "completed"},
    ],
  });
  assert.deepEqual(serialized.messages.map((message) => message.role), ["user", "assistant"]);
  assert.equal(serialized.messages.some((message) => message.content.includes("正在运行")), false);
  assert.deepEqual(chatThreadFromProject(serialized).messages.map((message) => message.role), ["learner", "teacher"]);
});

test("server running messages reopen fail-closed as stopped, not live", () => {
  const decoded = chatThreadFromProject({
    thread_id: "chat_0123456789abcdef01234567",
    title: "未完成回合",
    created_at: "2026-08-12T01:00:00Z",
    updated_at: "2026-08-12T01:00:01Z",
    messages: [{
      message_id: "assistant-1",
      role: "assistant",
      content: "部分内容",
      status: "running",
      created_at: "2026-08-12T01:00:01Z",
      web_search_used: false,
      sources: [],
    }],
  });
  assert.equal(decoded.messages[0]?.status, "stopped");
  assert.equal(decoded.messages[0]?.streaming, false);
  assert.equal(decoded.messages[0]?.retryable, false);
});

test("project serialization enforces the backend message and source bounds", () => {
  const encoded = projectChatThread({
    id: "chat_0123456789abcdef01234567",
    title: "边界",
    createdAt: Date.parse("2026-08-12T01:00:00Z"),
    updatedAt: Date.parse("2026-08-12T01:00:01Z"),
    messages: [{
      id: "assistant-1",
      role: "teacher",
      body: "x".repeat(70_000),
      createdAt: "2026-08-12T01:00:01Z",
      sources: [
        {title: "  有效来源  ", url: "https://example.com/source"},
        {title: "不安全来源", url: "javascript:alert(1)"},
      ],
    }],
  });
  assert.equal(encoded.messages[0]?.content.length, 64_000);
  assert.match(encoded.messages[0]?.content ?? "", /已安全截断\]$/);
  assert.deepEqual(encoded.messages[0]?.sources, [{title: "有效来源", url: "https://example.com/source"}]);
});

test("project serialization retains more than the legacy 400-message UI window", () => {
  const encoded = projectChatThread({
    id: "chat_abcdefabcdefabcdefabcdef",
    title: "完整历史",
    createdAt: Date.parse("2026-08-12T01:00:00Z"),
    updatedAt: Date.parse("2026-08-12T02:00:00Z"),
    messages: Array.from({length: 421}, (_, index) => ({
      id: `message-${index}`,
      role: index % 2 === 0 ? "learner" as const : "teacher" as const,
      body: `消息 ${index}`,
      createdAt: "2026-08-12T01:00:00Z",
      status: "completed" as const,
    })),
  });
  assert.equal(encoded.messages.length, 421);
  assert.equal(encoded.messages[0]?.content, "消息 0");
  assert.equal(encoded.messages.at(-1)?.content, "消息 420");
});
