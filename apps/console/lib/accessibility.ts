export type RovingMenuKey = "ArrowDown" | "ArrowUp" | "Home" | "End";

/** Return the next focusable menu index without depending on browser layout. */
export function nextRovingMenuIndex(current: number, count: number, key: RovingMenuKey): number {
  if (!Number.isInteger(count) || count <= 0) return -1;
  if (key === "Home") return 0;
  if (key === "End") return count - 1;
  const safeCurrent = current >= 0 && current < count ? current : 0;
  return (safeCurrent + (key === "ArrowDown" ? 1 : -1) + count) % count;
}

export function isRovingMenuKey(key: string): key is RovingMenuKey {
  return key === "ArrowDown" || key === "ArrowUp" || key === "Home" || key === "End";
}
