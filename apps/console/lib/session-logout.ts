export const LOGOUT_CANCEL_SETTLE_TIMEOUT_MS = 1_500;

export type LogoutCancelSettlement = "settled" | "timed_out";

function boundedTimeout(value: number): number {
  return Math.max(1, Math.min(10_000, Math.trunc(value)));
}

/**
 * Give already-registered runs a bounded opportunity to persist cancellation
 * before the current session is revoked. Rejection is a valid settlement:
 * logout must continue and the UI has already detached from every run.
 */
export async function settleRunCancellationsBeforeLogout<T>(
  runs: readonly T[],
  cancel: (run: T) => Promise<void>,
  timeoutMs = LOGOUT_CANCEL_SETTLE_TIMEOUT_MS,
): Promise<LogoutCancelSettlement> {
  if (!runs.length) return "settled";
  const delay = boundedTimeout(timeoutMs);
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const settled = Promise.allSettled(runs.map((run) => cancel(run)))
    .then(() => "settled" as const);
  const timedOut = new Promise<"timed_out">((resolve) => {
    timeout = setTimeout(() => resolve("timed_out"), delay);
  });
  try {
    return await Promise.race([settled, timedOut]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

export async function boundedLogoutCleanup(
  cleanup: () => Promise<boolean>,
  timeoutMs = LOGOUT_CANCEL_SETTLE_TIMEOUT_MS,
): Promise<boolean> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const completed = cleanup().catch(() => false);
  const timedOut = new Promise<false>((resolve) => {
    timeout = setTimeout(() => resolve(false), boundedTimeout(timeoutMs));
  });
  try {
    return await Promise.race([completed, timedOut]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}
