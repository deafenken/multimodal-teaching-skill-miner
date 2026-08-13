import fs from "node:fs";

export function markRuntimeActivity() {
  const target = process.env.TEACHLAB_RUNTIME_ACTIVITY_FILE?.trim();
  if (!target) return;
  try {
    const now = new Date();
    fs.utimesSync(target, now, now);
  } catch {
    // Readiness and request handling must not fail merely because optional
    // idle-shutdown bookkeeping is unavailable.
  }
}
