export function safeMarkdownUrl(url: string) {
  const value = url.trim();
  if (!value) return "";
  if (value.startsWith("#") || value.startsWith("/") || value.startsWith("./") || value.startsWith("../")) return value;
  try {
    const parsed = new URL(value);
    return ["http:", "https:", "mailto:"].includes(parsed.protocol) ? value : "";
  } catch {
    return "";
  }
}

export const SOURCE_PREVIEW_WINDOW = "teachlab-source-preview";

/** Open external evidence in one reusable, opener-free browsing context. */
export function openSourcePreview(url: string): boolean {
  const safe = safeMarkdownUrl(url);
  if (!safe || typeof window === "undefined") return false;
  try {
    const preview = window.open("about:blank", SOURCE_PREVIEW_WINDOW);
    if (!preview) return false;
    // Sever the source page's handle before navigating the reusable window.
    // This preserves tab reuse without granting the destination window.opener.
    preview.opener = null;
    preview.location.replace(safe);
    preview.focus();
    return true;
  } catch {
    return false;
  }
}
