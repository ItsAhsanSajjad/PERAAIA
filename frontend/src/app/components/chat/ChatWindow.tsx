"use client";

import { useRef, useEffect, useState, memo } from "react";
import type { Message, Reference } from "../../lib/types";
import { WelcomeScreen } from "./WelcomeScreen";
import { MessageBubble } from "./MessageBubble";
import { ThinkingIndicator } from "./ThinkingIndicator";

interface Props {
  messages: Message[];
  loading: boolean;
  failedPrompt: string | null;
  lastBotIsNew: boolean;
  clearNewFlag: () => void;
  onSendSuggestion: (text: string) => void;
  onOpenPdf: (ref: Reference) => void;
  onRetry: () => void;
  onEditMessage?: (index: number, newText: string) => void;
}

/**
 * ChatWindow manages the typewriter effect in its own state,
 * so the parent (and sidebar/header) does NOT rerender on every character.
 */
export const ChatWindow = memo(function ChatWindow({
  messages,
  loading,
  failedPrompt,
  lastBotIsNew,
  clearNewFlag,
  onSendSuggestion,
  onOpenPdf,
  onRetry,
  onEditMessage,
}: Props) {
  let lastUserIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      lastUserIdx = i;
      break;
    }
  }
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Typewriter state — scoped to this component only
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [shouldAnimate, setShouldAnimate] = useState(false);

  // Only animate when the hook explicitly tells us a fresh bot message arrived
  useEffect(() => {
    if (lastBotIsNew) {
      setShouldAnimate(true);
      clearNewFlag();
    }
  }, [lastBotIsNew, clearNewFlag]);

  // Typewriter effect using requestAnimationFrame (not setInterval)
  useEffect(() => {
    if (!shouldAnimate || messages.length === 0) return;
    const last = messages[messages.length - 1];
    if (last.role !== "assistant") return;
    const full = last.content;

    setIsTyping(true);
    setDisplayedText("");
    let i = 0;
    let lastTime = 0;
    const speed = Math.max(6, Math.min(18, 1500 / full.length));
    let rafId: number;

    const tick = (time: number) => {
      if (!lastTime) lastTime = time;
      const elapsed = time - lastTime;
      const charsToAdvance = Math.max(1, Math.floor(elapsed / speed));

      if (elapsed >= speed) {
        i = Math.min(i + charsToAdvance, full.length);
        setDisplayedText(full.slice(0, i));
        lastTime = time;
      }

      if (i >= full.length) {
        setIsTyping(false);
        setShouldAnimate(false);
        return;
      }
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [shouldAnimate, messages]);

  // Track whether user is pinned to the bottom. We freeze this latch at the
  // start of streaming so per-character scroll decisions don't flap when the
  // user scrolls up to read.
  const pinnedRef = useRef(true);
  const lastScrollAtRef = useRef(0);

  const isNearBottom = () => {
    const c = scrollContainerRef.current;
    if (!c) return true;
    return c.scrollHeight - c.scrollTop - c.clientHeight < 120;
  };

  // Continuously update pinned state from user scroll position
  useEffect(() => {
    const c = scrollContainerRef.current;
    if (!c) return;
    const onScroll = () => {
      pinnedRef.current = isNearBottom();
    };
    c.addEventListener("scroll", onScroll, { passive: true });
    return () => c.removeEventListener("scroll", onScroll);
  }, []);

  // Smooth scroll when loading indicator appears (one-shot, not per char)
  useEffect(() => {
    if (loading && pinnedRef.current) {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [loading]);

  // Smooth scroll to bottom when a new user message is appended
  useEffect(() => {
    const last = messages[messages.length - 1];
    if (last?.role === "user") {
      pinnedRef.current = true;
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // During typing: throttled INSTANT scroll (no smooth) — only if user was
  // pinned when streaming started AND they haven't scrolled up since. Avoids
  // the jittery rubber-band caused by spawning new smooth-scrolls every frame.
  useEffect(() => {
    if (!isTyping) return;
    if (!pinnedRef.current) return;
    const now = performance.now();
    if (now - lastScrollAtRef.current < 120) return;
    lastScrollAtRef.current = now;
    const c = scrollContainerRef.current;
    if (c) c.scrollTop = c.scrollHeight;
  }, [displayedText, isTyping]);

  const showWelcome = messages.length === 0 && !loading;

  return (
    <div
      ref={scrollContainerRef}
      className="flex-1 overflow-y-auto px-4 md:px-0"
      style={{ background: "var(--bg-chat)" }}
    >
      <div className="max-w-3xl mx-auto py-6 space-y-5">
        {showWelcome && <WelcomeScreen onSendMessage={onSendSuggestion} />}

        {messages.map((msg, i) => {
          const isLast = i === messages.length - 1;
          const showTypewriter = isLast && isTyping && msg.role === "assistant";
          return (
            <MessageBubble
              key={`${msg.timestamp}-${i}`}
              message={msg}
              typingText={showTypewriter ? displayedText : undefined}
              isTyping={showTypewriter}
              onOpenPdf={onOpenPdf}
              onRetry={msg.failed ? onRetry : undefined}
              onSendQuery={onSendSuggestion}
              onEdit={
                onEditMessage && i === lastUserIdx && !loading
                  ? (newText) => onEditMessage(i, newText)
                  : undefined
              }
            />
          );
        })}

        {loading && <ThinkingIndicator />}

        <div ref={chatEndRef} />
      </div>
    </div>
  );
});
