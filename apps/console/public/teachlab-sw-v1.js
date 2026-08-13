"use strict";

const STATIC_CACHE_PREFIX = "teachlab-console-static-";
const STATIC_CACHE = `${STATIC_CACHE_PREFIX}v1`;
const OFFLINE_SHELL_SCRIPT = "/teachlab-offline-shell-v1.js";
const MAX_PREWARM_URLS = 64;

function sameOriginImmutableStatic(url) {
  return url.origin === self.location.origin
    && (url.pathname.startsWith("/_next/static/") || url.pathname === OFFLINE_SHELL_SCRIPT);
}

function responseIsImmutable(response) {
  const directives = (response.headers.get("Cache-Control") || "")
    .split(",")
    .map((directive) => directive.trim().toLowerCase());
  return response.ok
    && response.type !== "opaque"
    && directives.includes("immutable")
    && directives.some((directive) => /^max-age=\d+$/.test(directive));
}

async function fetchAndCacheImmutable(request) {
  const url = new URL(request.url);
  if (!sameOriginImmutableStatic(url)) return fetch(request);
  const response = await fetch(request);
  if (responseIsImmutable(response)) {
    const cache = await caches.open(STATIC_CACHE);
    await cache.put(request, response.clone());
  }
  return response;
}

function offlineDocument() {
  return new Response(`<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TeachLab 离线只读快照</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#171714;color:#f2eee6;font:16px/1.6 system-ui,sans-serif}main{max-width:52rem;margin:auto;padding:clamp(1rem,5vw,4rem)}.status{border:1px solid #5d4136;background:#31231d;padding:1rem;border-radius:.75rem}.card{margin-top:1rem;border:1px solid #4b4942;background:#24231f;padding:1rem;border-radius:.75rem}article{padding:.75rem 0;border-top:1px solid #34332e}article:first-child{border-top:0}p{overflow-wrap:anywhere}button{font:inherit;padding:.65rem 1rem;border:1px solid #77736b;border-radius:.5rem;background:#292824;color:inherit}button:focus-visible,main:focus-visible{outline:3px solid #d97857;outline-offset:3px}@media(max-width:390px){main{padding:1rem}h1{font-size:1.5rem}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}</style>
<script defer src="${OFFLINE_SHELL_SCRIPT}"></script></head><body>
<main id="offline-main" tabindex="-1"><h1>TeachLab 离线只读模式</h1><p id="offline-status" class="status" role="status" aria-live="polite">正在验证账号绑定快照…</p>
<section id="offline-project" class="card" hidden aria-labelledby="offline-title"><h2 id="offline-title"></h2><p id="offline-description"></p><p id="offline-saved"></p><div id="offline-messages" role="log" aria-label="离线终态消息"></div></section>
<p><button id="offline-retry" type="button">重新连接</button></p></main></body></html>`, {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
      "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; img-src 'none'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
      "Referrer-Policy": "no-referrer",
      "X-Content-Type-Options": "nosniff",
    },
  });
}

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const request = new Request(new URL(OFFLINE_SHELL_SCRIPT, self.location.origin).href, {cache: "reload", credentials: "same-origin"});
    const response = await fetch(request);
    if (!responseIsImmutable(response)) throw new Error("offline shell asset is not immutable");
    const cache = await caches.open(STATIC_CACHE);
    await cache.put(request, response);
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter((name) => name.startsWith(STATIC_CACHE_PREFIX) && name !== STATIC_CACHE)
      .map((name) => caches.delete(name)));
    await self.clients.claim();
  })());
});

self.addEventListener("message", (event) => {
  if (event.data?.type !== "PREWARM_IMMUTABLE" || !Array.isArray(event.data.urls)) return;
  event.waitUntil(Promise.all(event.data.urls.slice(0, MAX_PREWARM_URLS).map(async (candidate) => {
    try {
      const url = new URL(candidate);
      if (!sameOriginImmutableStatic(url)) return;
      await fetchAndCacheImmutable(new Request(url.href, {credentials: "same-origin"}));
    } catch {
      // A missing optional chunk cannot invalidate the already-installed shell.
    }
  })));
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (request.mode === "navigate") {
    event.respondWith(fetch(request).catch(() => offlineDocument()));
    return;
  }
  if (!sameOriginImmutableStatic(url)) return;
  event.respondWith((async () => {
    const cached = await caches.match(request, {cacheName: STATIC_CACHE});
    return cached || fetchAndCacheImmutable(request);
  })());
});
