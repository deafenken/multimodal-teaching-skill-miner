import assert from "node:assert/strict";
import test from "node:test";

import {ModeRequestSlots, nextChatFollowUp} from "../lib/mode-runtime.ts";

test("Chat and Teach occupy independent request slots", () => {
  const slots = new ModeRequestSlots();
  const chatController = new AbortController();
  const teachController = new AbortController();
  const chat = slots.begin("chat", {projectId: "project-a", surfaceId: "chat-a"}, chatController);
  const teach = slots.begin("teach", {projectId: "project-a", surfaceId: "session-a"}, teachController);

  assert.equal(slots.isCurrent(chat, {projectId: "project-a", surfaceId: "chat-a"}), true);
  assert.equal(slots.isCurrent(teach, {projectId: "project-a", surfaceId: "session-a"}), true);
  assert.equal(chatController.signal.aborted, false);
  assert.equal(teachController.signal.aborted, false);
});

test("cancelling Chat leaves the concurrent Teach request current", () => {
  const slots = new ModeRequestSlots();
  const chatController = new AbortController();
  const teachController = new AbortController();
  const chat = slots.begin("chat", {projectId: "project-a", surfaceId: "chat-a"}, chatController);
  const teach = slots.begin("teach", {projectId: "project-a", surfaceId: "session-a"}, teachController);

  slots.cancel("chat");

  assert.equal(chatController.signal.aborted, true);
  assert.equal(teachController.signal.aborted, false);
  assert.equal(slots.isCurrent(chat, {projectId: "project-a", surfaceId: "chat-a"}), false);
  assert.equal(slots.isCurrent(teach, {projectId: "project-a", surfaceId: "session-a"}), true);
});

test("a project switch fences and detaches both transport subscriptions", () => {
  const slots = new ModeRequestSlots();
  const chatController = new AbortController();
  const teachController = new AbortController();
  const chat = slots.begin("chat", {projectId: "project-a", surfaceId: "chat-a"}, chatController);
  const teach = slots.begin("teach", {projectId: "project-a", surfaceId: "session-a"}, teachController);

  slots.cancelAll();

  assert.equal(chatController.signal.aborted, true);
  assert.equal(teachController.signal.aborted, true);
  assert.equal(slots.isCurrent(chat, {projectId: "project-b", surfaceId: "chat-a"}), false);
  assert.equal(slots.isCurrent(teach, {projectId: "project-b", surfaceId: "session-a"}), false);
});

test("project and surface identities prevent late callbacks from crossing conversations", () => {
  const slots = new ModeRequestSlots();
  const chat = slots.begin("chat", {projectId: "project-a", surfaceId: "chat-a"});
  const teach = slots.begin("teach", {projectId: "project-a", surfaceId: "session-a"});

  assert.equal(slots.isCurrent(chat, {projectId: "project-a", surfaceId: "chat-b"}), false);
  assert.equal(slots.isCurrent(chat, {projectId: "project-b", surfaceId: "chat-a"}), false);
  assert.equal(slots.isCurrent(teach, {projectId: "project-a", surfaceId: "session-b"}), false);
  assert.equal(slots.isCurrent(teach, {projectId: "project-b", surfaceId: "session-a"}), false);
});

test("Chat follow-ups wait only for Chat and never cross into another thread", () => {
  const prompts = [
    {id: "old", threadId: "chat-old"},
    {id: "active", threadId: "chat-active"},
  ];

  assert.equal(nextChatFollowUp(prompts, true, "chat-active"), undefined);
  assert.equal(nextChatFollowUp(prompts, false, "chat-active")?.id, "active");
  assert.equal(nextChatFollowUp(prompts, false, "chat-missing"), undefined);
});
