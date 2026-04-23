"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ConnectionStatus } from "../lib/types";

/**
 * Polls the backend /health endpoint every 15 s and surfaces the
 * current OpenAI-service status to the UI so the chat can:
 *   - show a red "System offline" badge + banner
 *   - disable the composer input when upstream AI is down
 *   - recover automatically when the backend flips back to available
 *
 * The legacy `reportSuccess` / `reportFailure` callbacks are kept so
 * existing call sites (useChatSessions.ts) still work — a failed
 * /ask call will eagerly mark the system offline between polls.
 *
 * Status values:
 *   - "online"     — /health succeeded AND openai.available === true
 *   - "offline"    — /health reported openai.available === false, OR
 *                    two consecutive poll failures
 *   - "connecting" — transient: first failed poll (avoids flicker)
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const POLL_INTERVAL_MS = 15_000;

export function useHealthCheck() {
  const [status, setStatus] = useState<ConnectionStatus>("online");
  const [reason, setReason] = useState<string>("");
  const failuresRef = useRef(0);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = (await res.json()) as {
        openai?: { available?: boolean; reason?: string };
      };
      const available = data.openai?.available !== false;
      failuresRef.current = 0;
      if (available) {
        setStatus("online");
        setReason("");
      } else {
        setStatus("offline");
        setReason(data.openai?.reason || "Service unavailable");
      }
    } catch {
      failuresRef.current += 1;
      // First failure → show "connecting" to hide brief network blips.
      // Second consecutive failure → commit to offline.
      if (failuresRef.current === 1) {
        setStatus((prev) => (prev === "online" ? "connecting" : prev));
      } else if (failuresRef.current >= 2) {
        setStatus("offline");
        setReason("Unable to reach server");
      }
    }
  }, []);

  useEffect(() => {
    // Immediate probe so we don't wait 15 s for the first status read.
    poll();
    pollTimer.current = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      if (pollTimer.current) clearInterval(pollTimer.current);
    };
  }, [poll]);

  const reportSuccess = useCallback(() => {
    failuresRef.current = 0;
    setStatus("online");
    setReason("");
  }, []);

  const reportFailure = useCallback(() => {
    // Explicit failures from live /ask calls are authoritative —
    // we don't wait for the polling debounce.
    setStatus("offline");
  }, []);

  return { status, reason, reportSuccess, reportFailure } as const;
}
