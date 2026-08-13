"use client";

import {useEffect} from "react";

const SERVICE_WORKER_URL = "/teachlab-sw-v1.js";

function immutableResourcesSeenByThisPage(): string[] {
  const origin = window.location.origin;
  return Array.from(new Set(performance.getEntriesByType("resource").flatMap((entry) => {
    try {
      const url = new URL(entry.name);
      return url.origin === origin && url.pathname.startsWith("/_next/static/")
        ? [url.href]
        : [];
    } catch {
      return [];
    }
  }))).slice(0, 64);
}

/** Register the production-only, static-assets-only offline shell. */
export function ServiceWorkerLifecycle() {
  useEffect(() => {
    if (process.env.NODE_ENV !== "production" || !("serviceWorker" in navigator)) return;
    let active = true;
    let lastUpdate = 0;

    const prewarm = (worker?: ServiceWorker | null) => {
      worker?.postMessage({
        type: "PREWARM_IMMUTABLE",
        urls: immutableResourcesSeenByThisPage(),
      });
    };
    const register = async () => {
      try {
        const registration = await navigator.serviceWorker.register(SERVICE_WORKER_URL, {
          scope: "/",
          updateViaCache: "none",
        });
        if (!active) return;
        const ready = await navigator.serviceWorker.ready;
        if (!active) return;
        prewarm(ready.active ?? registration.active);
        lastUpdate = Date.now();
      } catch {
        // Offline support is an enhancement. The online app remains usable and
        // must never weaken its authentication boundary when registration fails.
      }
    };
    const update = () => {
      if (document.visibilityState !== "visible" || Date.now() - lastUpdate < 60 * 60 * 1_000) return;
      lastUpdate = Date.now();
      void navigator.serviceWorker.getRegistration("/").then(async (registration) => {
        await registration?.update();
        prewarm(registration?.active);
      }).catch(() => undefined);
    };

    void register();
    document.addEventListener("visibilitychange", update);
    window.addEventListener("online", update);
    return () => {
      active = false;
      document.removeEventListener("visibilitychange", update);
      window.removeEventListener("online", update);
    };
  }, []);
  return null;
}
