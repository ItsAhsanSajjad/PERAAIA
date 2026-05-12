"use client";

import Image from "next/image";
import { memo, useState, useCallback, useMemo, useEffect, useRef } from "react";
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


const TOPIC_KEYWORDS: { keyword: string; topic: string }[] = [
  { keyword: "enforcement", topic: "Enforcement Procedures" },
  { keyword: "epo", topic: "Enforcement Procedure Orders" },
  { keyword: "compliance", topic: "Compliance Standards" },
  { keyword: "governance", topic: "Governance Structures" },
  { keyword: "board", topic: "PERA Board Composition" },
  { keyword: "kpi", topic: "KPI Frameworks" },
  { keyword: "performance", topic: "Institutional Performance" },
  { keyword: "salary", topic: "Pay Scales & Benefits" },
  { keyword: "pay scale", topic: "Pay Scales & Benefits" },
  { keyword: "training", topic: "Learning & Development" },
  { keyword: "discipline", topic: "Work Discipline & Ethics" },
  { keyword: "inspection", topic: "Regulatory Inspections" },
  { keyword: "service delivery", topic: "Service Delivery Standards" },
];

function extractRelatedTopics(text: string, maxTopics = 3): string[] {
  const lower = text.toLowerCase();
  const found: string[] = [];
  const seen = new Set<string>();
  for (const { keyword, topic } of TOPIC_KEYWORDS) {
    if (lower.includes(keyword) && !seen.has(topic)) {
      found.push(topic);
      seen.add(topic);
    }
    if (found.length >= maxTopics) break;
  }
  return found;
}

interface Props {
  message: Message;
  typingText?: string;
  isTyping?: boolean;
  onOpenPdf?: (ref: Reference) => void;
  onRetry?: () => void;
  onSendQuery?: (text: string) => void;
  onEdit?: (newText: string) => void;
}

export const MessageBubble = memo(function MessageBubble({
  message,
  typingText,
  isTyping,
  onOpenPdf,
  onRetry,
  onSendQuery,
  onEdit,
}: Props) {
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(message.content);
  const editTextareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (isEditing && editTextareaRef.current) {
      const ta = editTextareaRef.current;
      ta.focus();
      ta.setSelectionRange(ta.value.length, ta.value.length);
      ta.style.height = "auto";
      ta.style.height = `${ta.scrollHeight}px`;
    }
  }, [isEditing]);

  const cancelEdit = useCallback(() => {
    setIsEditing(false);
    setEditText(message.content);
  }, [message.content]);

  const saveEdit = useCallback(() => {
    const trimmed = editText.trim();
    if (!trimmed || trimmed === message.content) {
      cancelEdit();
      return;
    }
    setIsEditing(false);
    onEdit?.(trimmed);
  }, [editText, message.content, onEdit, cancelEdit]);

  const copyText = useCallback(() => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    showToast("Copied to clipboard", "success");
    setTimeout(() => setCopied(false), 2000);
  }, [message.content]);

  // Compute related topics once
  const relatedTopics = useMemo(
    () => (!isTyping && message.role === "assistant" ? extractRelatedTopics(message.content) : []),
    [message.content, message.role, isTyping],
  );

  if (message.role === "user") {
    if (isEditing) {
      return (
        <div className="flex justify-end gap-2.5">
          <div className="max-w-[82%] md:max-w-[72%] user-bubble user-bubble-editing">
            <div className="px-4 py-3 relative z-10">
              <textarea
                ref={editTextareaRef}
                value={editText}
                onChange={(e) => {
                  setEditText(e.target.value);
                  const ta = e.currentTarget;
                  ta.style.height = "auto";
                  ta.style.height = `${ta.scrollHeight}px`;
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    saveEdit();
                  } else if (e.key === "Escape") {
                    e.preventDefault();
                    cancelEdit();
                  }
                }}
                className="msg-edit-textarea"
                rows={1}
                aria-label="Edit message"
              />
              <div className="msg-edit-actions">
                <button onClick={cancelEdit} className="msg-edit-cancel">
                  Cancel
                </button>
                <button
                  onClick={saveEdit}
                  disabled={!editText.trim() || editText.trim() === message.content}
                  className="msg-edit-save"
                >
                  Save &amp; Send
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }
    return (
      <div className="flex flex-col items-end gap-1 user-msg-wrap">
        <div className="max-w-[82%] md:max-w-[72%] user-bubble relative">
          <div className="px-4 py-3 relative z-10">
            <p className="text-sm leading-relaxed text-white">{message.content}</p>
          </div>
          <div className="px-4 pb-2 text-[10px] text-right text-white/50">
            {timeAgo(message.timestamp)}
          </div>
        </div>
        {onEdit && (
          <button
            onClick={() => {
              setEditText(message.content);
              setIsEditing(true);
            }}
            className="msg-edit-btn"
            aria-label="Edit message"
            title="Edit message"
          >
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M12 20h9" />
              <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z" />
            </svg>
            <span>Edit</span>
          </button>
        )}
      </div>
    );
  }

  // Assistant message
  const displayText = isTyping && typingText !== undefined ? typingText : message.content;
  const showRefs = message.references && message.references.length > 0 && !isTyping;
  const showStructure = !isTyping && !message.failed;

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
          {/* Answer Section */}
          {showStructure && (
            <div className="ans-section-label">Answer</div>
          )}
          <div className="msg-bot-text">
            {renderMarkdown(displayText)}
            {isTyping && <span className="typewriter-cursor" />}
          </div>

          {/* Authority / Source Block */}
          {showRefs && (
            <div className="ans-authority-block">
              <div className="ans-section-label">
                <span className="ans-verified-dot" />
                Source — Derived from Official PERA Documents
              </div>
              <div className="flex flex-wrap gap-1.5 mt-1.5">
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
            </div>
          )}


          {/* Related Topics */}
          {showStructure && relatedTopics.length > 0 && (
            <div className="ans-related-block">
              <div className="ans-section-label">Related Topics</div>
              <div className="flex flex-wrap gap-1.5 mt-1">
                {relatedTopics.map((topic) => (
                  <button
                    key={topic}
                    className="ans-related-chip"
                    onClick={() => onSendQuery?.(`Tell me about ${topic} in PERA`)}
                  >
                    {topic}
                  </button>
                ))}
              </div>
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
