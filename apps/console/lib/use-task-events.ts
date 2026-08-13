"use client";

import {useEffect, useRef} from "react";

import {apiUrl} from "@/lib/api";
import type {TaskEvent} from "@/lib/types";

export function useTaskEvents(sessionId: string | null, onEvent: (event: TaskEvent) => void) {
  const handlerRef = useRef(onEvent);
  handlerRef.current = onEvent;

  useEffect(() => {
    if (!sessionId || typeof EventSource === "undefined") return;
    const streamPath = process.env.NEXT_PUBLIC_EVENT_STREAM_PATH ?? "/api/events";
    const url = new URL(apiUrl(streamPath), window.location.origin);
    url.searchParams.set("session_id", sessionId);
    const source = new EventSource(url, {withCredentials: true});
    source.onmessage = (message) => {
      try {
        handlerRef.current(JSON.parse(message.data) as TaskEvent);
      } catch {
        // Malformed stream events are ignored here and must be recorded by BFF telemetry.
      }
    };
    return () => source.close();
  }, [sessionId]);
}
