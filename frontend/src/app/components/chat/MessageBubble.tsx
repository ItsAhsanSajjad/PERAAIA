"use client";

import Image from "next/image";
import { memo, useState, useCallback } from "react";
import type { Message, Reference } from "../../lib/types";
import { renderMarkdown } from "../../lib/markdown";
import { showToast } from "../common/Toast";

const timeAgo = (ts: number) => {
  const d = Math.floor((Date.now() - ts) / 1000);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
};

interface Props {
  message: Message;
  /** Text to show during typewriter effect (only for the latest message) */
  typingText?: string;
  isTyping?: boolean;
  onOpenPdf?: (ref: Reference) => void;
  onRetry?: () => void;
}

export const MessageBubble = memo(function MessageBubble({
  message,
  typingText,
  isTyping,
  onOpenPdf,
  onRetry,
}: Props) {
  const [copied, setCopied] = useState(false);

  const copyText = useCallback(() => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    showToast("Copied to clipboard", "success");
    setTimeout(() => setCopied(false), 2000);
  }, [message.content]);

  if (message.role === "user") {
    return (
      <div className="flex justify-end gap-2.5">
        <div className="max-w-[82%] md:max-w-[72%] user-bubble">
          <div className="px-4 py-3 relative z-10">
            <p className="text-sm leading-relaxed text-white">{message.content}</p>
          </div>
          <div className="px-4 pb-2 text-[10px] text-right text-white/50">
            {timeAgo(message.timestamp)}
          </div>
        </div>
      </div>
    );
  }

  // Assistant message
  const displayText = isTyping && typingText !== undefined ? typingText : message.content;
  const showRefs = message.references && message.references.length > 0 && !isTyping;

  return (
    <div className="flex justify-start gap-2.5">
      <div className="flex-shrink-0 mt-1">
        <div
          className="w-8 h-8 rounded-xl overflow-hidden flex items-center justify-center"
          style={{ background: "var(--accent-soft)" }}
        >
          <Image src="/pera_logo.png" alt="" width={20} height={20} className="rounded-md" />
        </div>
      </div>
      <div className="max-w-[82%] md:max-w-[72%] bot-bubble">
        <div className="px-4 py-3">
          <div className="msg-bot-text">
            {renderMarkdown(displayText)}
            {isTyping && <span className="typewriter-cursor" />}
          </div>

          {/* References */}
          {showRefs && (
            <div className="mt-3 pt-3 flex flex-wrap gap-1.5" style={{ borderTop: "1px solid var(--border)" }}>
              <span className="text-[10px] font-medium self-center mr-1" style={{ color: "var(--text-faint)" }}>
                Sources:
              </span>
              {message.references!.slice(0, 8).map((ref, ri) => {
                const docName = ref.document?.replace(/\.pdf$/i, "") || "Document";
                const truncated = docName.length > 22 ? docName.slice(0, 22) + "…" : docName;
                const page = ref.page_start || ref.page;
                return (
                  <button
                    key={ri}
                    onClick={() => onOpenPdf?.(ref)}
                    className="ref-chip"
                    title={`${ref.document}${page ? ` — Page ${page}` : ""}`}
                  >
                    [{ref.id ?? ri + 1}] {truncated}
                    {page ? ` · p.${page}` : ""}
                  </button>
                );
              })}
            </div>
          )}

          {/* Retry button for failed messages */}
          {message.failed && onRetry && (
            <button
              onClick={onRetry}
              className="mt-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all"
              style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
            >
              ↻ Retry
            </button>
          )}

          {/* Copy */}
          {!message.failed && (
            <button
              onClick={copyText}
              className="copy-btn absolute top-2 right-2 text-[10px] px-2 py-1 rounded-lg font-medium"
              style={{ background: "var(--bg-hover)", color: "var(--text-faint)" }}
              aria-label="Copy message"
            >
              {copied ? "Copied ✓" : "Copy"}
            </button>
          )}
        </div>
        <div className="px-4 pb-2 text-[10px]" style={{ color: "var(--text-faint)" }}>
          {timeAgo(message.timestamp)}
        </div>
      </div>
    </div>
  );
});
