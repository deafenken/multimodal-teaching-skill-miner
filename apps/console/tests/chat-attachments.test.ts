import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

import {
  CHAT_ATTACHMENT_ACCEPT,
  MAX_CHAT_ATTACHMENTS,
  chatAttachmentMimeType,
  chatAttachmentAccept,
  chatAttachmentTypeLabel,
  chatResourceRefs,
  selectChatAttachments,
} from "../lib/chat-attachments.ts";

const descriptor = (name: string, size = 64, type = "") => ({name, size, type});

test("Chat attachment selection shares the bounded local multimodal allowlist", () => {
  const files = [
    descriptor("notes.pdf", 100, "application/pdf"),
    descriptor("table.xlsx"),
    descriptor("voice.m4a", 200, "audio/x-m4a"),
    descriptor("clip.webm", 200, "video/webm"),
    descriptor("scan.png", 300, "image/png"),
    descriptor("payload.exe"),
    descriptor("empty.txt", 0),
    descriptor("huge.txt", 12 * 1024 * 1024 + 1),
  ];
  const selected = selectChatAttachments(files, 0);
  assert.deepEqual(selected.accepted.map((item) => item.name), ["notes.pdf", "table.xlsx", "voice.m4a", "clip.webm", "scan.png"]);
  assert.deepEqual(selected.rejected.map((item) => item.reason), ["unsupported_type", "empty", "too_large"]);
  assert.equal(chatAttachmentMimeType(files[2]), "audio/x-m4a");
  assert.equal(chatAttachmentMimeType(files[3]), "video/webm");
  assert.match(CHAT_ATTACHMENT_ACCEPT, /\.xlsx/);
});

test("selection never exceeds the per-message bound", () => {
  const selected = selectChatAttachments(
    Array.from({length: 8}, (_, index) => descriptor(`note-${index}.txt`)),
    1,
  );
  assert.equal(selected.accepted.length, MAX_CHAT_ATTACHMENTS - 1);
  assert.equal(selected.rejected.filter((item) => item.reason === "attachment_limit").length, 3);
});

test("runtime capability projection hides unavailable host extractors", () => {
  const supported = ["txt", "docx", "pptx", "csv"];
  const selected = selectChatAttachments(
    [descriptor("notes.txt"), descriptor("legacy.doc"), descriptor("scan.png"), descriptor("deck.pptx")],
    0,
    supported,
  );
  assert.deepEqual(selected.accepted.map((item) => item.name), ["notes.txt", "deck.pptx"]);
  assert.deepEqual(selected.rejected.map((item) => item.reason), ["unsupported_type", "unsupported_type"]);
  assert.equal(chatAttachmentAccept(supported), ".pptx,.docx,.txt,.csv");
  assert.equal(chatAttachmentAccept([]), "");
});

test("only extracted, hash-shaped resource/stage pairs enter the Chat request", () => {
  const refs = chatResourceRefs([
    {localId: "1", name: "a.txt", status: "ready", resource: {resource_id: `res_${"a".repeat(20)}`, staged_resource_id: `stage_${"b".repeat(24)}`, display_name: "a.txt"}},
    {localId: "2", name: "b.txt", status: "extracting"},
    {localId: "3", name: "c.txt", status: "failed", resource: {resource_id: `res_${"c".repeat(20)}`, staged_resource_id: `stage_${"d".repeat(24)}`, display_name: "c.txt"}},
    {localId: "4", name: "bad.txt", status: "ready", resource: {resource_id: "res_other", staged_resource_id: "stage_other", display_name: "bad.txt"}},
  ]);
  assert.deepEqual(refs, [{resource_id: `res_${"a".repeat(20)}`, staged_resource_id: `stage_${"b".repeat(24)}`}]);
  assert.equal(JSON.stringify(refs).includes("data_base64"), false);
});

test("attachment type labels expose local processing status", () => {
  assert.equal(chatAttachmentTypeLabel({name: "scan.png", mimeType: "image/png"}), "图片 · 等待本地 OCR");
  assert.equal(chatAttachmentTypeLabel({name: "scan.png", resource: {resource_id: `res_${"a".repeat(20)}`, display_name: "scan.png", resource_type: "image_ocr"}}), "图片 · 本地 OCR");
});

test("Chat composer exposes button, drop, paste, state, and removal controls", () => {
  const source = readFileSync(new URL("../components/workbench/conversation-pane.tsx", import.meta.url), "utf8");
  for (const token of [
    'aria-label="添加 Chat 附件"',
    "onDragOver={handleComposerDragOver}",
    "onDrop={handleComposerDrop}",
    "onPaste={handleComposerPaste}",
    "Chat 附件",
    "chatAttachmentTypeLabel",
    "onRemoveResource(item.localId)",
  ]) assert.ok(source.includes(token), `missing Chat attachment UI contract: ${token}`);
});

test("Chat sends identifier pairs only and retry state never retains browser File objects", () => {
  const workbench = readFileSync(new URL("../components/workbench/workbench.tsx", import.meta.url), "utf8");
  const api = readFileSync(new URL("../lib/api.ts", import.meta.url), "utf8");
  for (const token of [
    "resourceRefs: ChatResourceRef[]",
    "resourceItems: ResourceUploadItem[]",
    "resourceRefs,",
    "resourceItems: resourceItems.map((item) => ({...item}))",
  ]) assert.ok(workbench.includes(token), `missing attachment retry contract: ${token}`);
  const retryStart = workbench.indexOf("type ChatRetryContext");
  const retryShape = workbench.slice(retryStart, retryStart + 500);
  assert.equal(retryShape.includes("File"), false);
  const streamStart = api.indexOf("export function streamChat");
  const streamShape = api.slice(streamStart, streamStart + 1_200);
  assert.ok(streamShape.includes("resource_refs: options.resourceRefs"));
  assert.equal(streamShape.includes("data_base64"), false);
});
