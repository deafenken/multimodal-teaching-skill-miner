import type {ResourceUploadItem} from "@/lib/types";

export const MAX_CHAT_ATTACHMENTS = 6;
export const MAX_CHAT_ATTACHMENT_BYTES = 12 * 1024 * 1024;
const ALL_CHAT_ATTACHMENT_EXTENSIONS = [
  ".pdf", ".ppt", ".pptx", ".doc", ".docx", ".rtf", ".txt", ".md", ".markdown",
  ".csv", ".tsv", ".xlsx", ".png", ".jpg", ".jpeg", ".webp",
  ".wav", ".mp3", ".m4a", ".webm", ".mp4", ".mov",
] as const;

export const CHAT_ATTACHMENT_ACCEPT = ALL_CHAT_ATTACHMENT_EXTENSIONS.join(",");

function normalizedSupportedExtensions(supportedExtensions?: readonly string[]) {
  if (!supportedExtensions) return new Set(ALL_CHAT_ATTACHMENT_EXTENSIONS.map((value) => value.slice(1)));
  return new Set(supportedExtensions.flatMap((value) => {
    const normalized = value.trim().toLocaleLowerCase().replace(/^\./, "");
    return normalized && /^[a-z0-9]+$/.test(normalized) ? [normalized] : [];
  }));
}

export function chatAttachmentAccept(supportedExtensions?: readonly string[]) {
  const supported = normalizedSupportedExtensions(supportedExtensions);
  return ALL_CHAT_ATTACHMENT_EXTENSIONS.filter((extension) => supported.has(extension.slice(1))).join(",");
}

const MIME_BY_EXTENSION: Record<string, string> = {
  pdf: "application/pdf",
  ppt: "application/vnd.ms-powerpoint",
  pptx: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  doc: "application/msword",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  rtf: "application/rtf",
  txt: "text/plain",
  md: "text/markdown",
  markdown: "text/markdown",
  csv: "text/csv",
  tsv: "text/tab-separated-values",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  wav: "audio/wav",
  mp3: "audio/mpeg",
  m4a: "audio/mp4",
  webm: "audio/webm",
  mp4: "video/mp4",
  mov: "video/quicktime",
};

export interface ChatAttachmentDescriptor {
  name: string;
  size: number;
  type: string;
}

export interface ChatResourceRef {
  resource_id: string;
  staged_resource_id: string;
}

export interface RejectedChatAttachment<T> {
  file: T;
  reason: "unsupported_type" | "empty" | "too_large" | "attachment_limit";
  message: string;
}

function extensionOf(name: string) {
  return name.split(".").pop()?.toLocaleLowerCase() ?? "";
}

export function chatAttachmentMimeType(file: ChatAttachmentDescriptor) {
  const extension = extensionOf(file.name);
  const declared = file.type.split(";", 1)[0]?.trim().toLocaleLowerCase() ?? "";
  if (extension === "webm" && (declared === "audio/webm" || declared === "video/webm")) return declared;
  if (extension === "mp4" && declared === "audio/mp4") return declared;
  if (extension === "m4a" && declared === "audio/x-m4a") return declared;
  return MIME_BY_EXTENSION[extension] ?? declared;
}

export function chatAttachmentTypeLabel(item: Pick<ResourceUploadItem, "name" | "mimeType" | "resource">) {
  const resourceType = item.resource?.resource_type;
  if (resourceType === "image_ocr") return "图片 · 本地 OCR";
  if (resourceType === "audio") return "音频 · 本地转写";
  if (resourceType === "video") return "视频 · 本地转写";
  if (resourceType === "spreadsheet") return "表格";
  if (resourceType === "presentation") return "演示文稿";
  if (resourceType === "pdf") return "PDF";
  if (resourceType === "document") return "文稿";
  if (resourceType === "text") return "文本";
  const mime = item.mimeType ?? MIME_BY_EXTENSION[extensionOf(item.name)] ?? "";
  if (mime.startsWith("image/")) return "图片 · 等待本地 OCR";
  if (mime.startsWith("audio/")) return "音频 · 等待本地转写";
  if (mime.startsWith("video/")) return "视频 · 等待本地转写";
  return extensionOf(item.name).toLocaleUpperCase() || "文件";
}

export function selectChatAttachments<T extends ChatAttachmentDescriptor>(
  files: readonly T[],
  existingCount: number,
  supportedExtensions?: readonly string[],
): {accepted: T[]; rejected: Array<RejectedChatAttachment<T>>} {
  const available = Math.max(0, MAX_CHAT_ATTACHMENTS - Math.max(0, Math.trunc(existingCount)));
  const supported = normalizedSupportedExtensions(supportedExtensions);
  const accepted: T[] = [];
  const rejected: Array<RejectedChatAttachment<T>> = [];
  for (const file of files) {
    const extension = extensionOf(file.name);
    if (!Object.hasOwn(MIME_BY_EXTENSION, extension) || !supported.has(extension)) {
      rejected.push({file, reason: "unsupported_type", message: "当前本地提取器不支持这种文件类型"});
      continue;
    }
    if (!Number.isFinite(file.size) || file.size < 1) {
      rejected.push({file, reason: "empty", message: "空文件不能作为 Chat 附件"});
      continue;
    }
    if (file.size > MAX_CHAT_ATTACHMENT_BYTES) {
      rejected.push({file, reason: "too_large", message: "单个附件不能超过 12 MB"});
      continue;
    }
    if (accepted.length >= available) {
      rejected.push({file, reason: "attachment_limit", message: `每条 Chat 消息最多 ${MAX_CHAT_ATTACHMENTS} 个附件`});
      continue;
    }
    accepted.push(file);
  }
  return {accepted, rejected};
}

export function chatResourceRefs(items: readonly ResourceUploadItem[]): ChatResourceRef[] {
  const references = new Map<string, ChatResourceRef>();
  for (const item of items) {
    if (item.status !== "ready" && item.status !== "truncated") continue;
    const resourceId = item.resource?.resource_id;
    const stagedResourceId = item.resource?.staged_resource_id;
    if (!resourceId || !/^res_[0-9a-f]{20}$/.test(resourceId)) continue;
    if (!stagedResourceId || !/^stage_[0-9a-f]{24}$/.test(stagedResourceId)) continue;
    references.set(`${resourceId}\0${stagedResourceId}`, {
      resource_id: resourceId,
      staged_resource_id: stagedResourceId,
    });
  }
  return [...references.values()].slice(0, MAX_CHAT_ATTACHMENTS);
}
