"use client";

import Image from "next/image";
import { memo, useState, useCallback, useMemo, useEffect, useRef } from "react";
import type { Message, Reference } from "../../lib/types";
import { renderMarkdown, sanitizeStreamingMarkdown } from "../../lib/markdown";
import { scrollBehavior } from "../../lib/motion";
import { showToast } from "../common/Toast";

const timeAgo = (ts: number) => {
  const d = Math.floor((Date.now() - ts) / 1000);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
};


// Citizen-facing related topics. Each entry maps an answer keyword to a
// public label plus a natural follow-up question sent on click.
const TOPIC_DEFS: { keyword: string; topic: string; prompt: string }[] = [
  { keyword: "complaint", topic: "Filing a Complaint", prompt: "Tell me how to file a complaint with PERA." },
  { keyword: "track", topic: "Tracking a Complaint", prompt: "How can I track the status of my complaint with PERA?" },
  { keyword: "inspection", topic: "Inspection Rights", prompt: "Explain my rights during a PERA inspection." },
  { keyword: "rights", topic: "Inspection Rights", prompt: "Explain my rights during a PERA inspection." },
  { keyword: "appeal", topic: "Appeals and Responses", prompt: "How can I appeal or respond to a PERA action?" },
  { keyword: "penalty", topic: "Notices and Penalties", prompt: "Tell me about PERA notices and penalties." },
  { keyword: "notice", topic: "Notices and Penalties", prompt: "Tell me about PERA notices and penalties." },
  { keyword: "fine", topic: "Notices and Penalties", prompt: "Tell me about PERA notices and penalties." },
  { keyword: "enforcement", topic: "Enforcement Actions", prompt: "What enforcement actions can PERA take?" },
  { keyword: "violation", topic: "Enforcement Actions", prompt: "What happens if a business violates PERA rules?" },
  { keyword: "permit", topic: "Permits and Licensing", prompt: "What permits or licences does PERA require?" },
  { keyword: "licen", topic: "Permits and Licensing", prompt: "What permits or licences does PERA require?" },
  { keyword: "notification", topic: "Official Notifications", prompt: "Where can I find official PERA notifications?" },
  { keyword: "document", topic: "Required Documents", prompt: "What documents do I need for PERA processes?" },
  { keyword: "regulation", topic: "PERA Rules", prompt: "Explain the main PERA rules and regulations that apply to me." },
  { keyword: "rule", topic: "PERA Rules", prompt: "Explain the main PERA rules and regulations that apply to me." },
  { keyword: "contact", topic: "Contacting PERA", prompt: "How can I contact the official PERA office?" },
  { keyword: "office", topic: "Contacting PERA", prompt: "How can I contact the official PERA office?" },
];

function extractRelatedTopics(
  text: string,
  maxTopics = 3,
): { topic: string; prompt: string }[] {
  const lower = text.toLowerCase();
  const found: { topic: string; prompt: string }[] = [];
  const seen = new Set<string>();
  for (const { keyword, topic, prompt } of TOPIC_DEFS) {
    if (lower.includes(keyword) && !seen.has(topic)) {
      found.push({ topic, prompt });
      seen.add(topic);
    }
    if (found.length >= maxTopics) break;
  }
  return found;
}

// Convert a raw PDF filename into a readable, citizen-friendly title.
const DOC_KEEP_UPPER = new Set(["PERA", "HR", "KPI", "EPO", "SOP", "SR", "SRS", "MOM"]);
const DOC_SMALL_WORDS = new Set(["and", "or", "of", "the", "for", "in", "to", "a", "an", "on", "with", "by"]);

function friendlyDocTitle(raw?: string): string {
  if (!raw) return "PERA Document";
  let s = raw.replace(/\.pdf$/i, "");
  s = s.replace(/[_\-]+/g, " ");
  // Drop noise tokens (keep years like 2024/2025 intact — no bare \d match).
  s = s.replace(/\b(final|draft|copy|filecopy|signed|scanned?|updated?|latest|recommended)\b/gi, " ");
  s = s.replace(/\bv(?:er(?:sion)?)?\.?\s?\d+(?:\.\d+)*\b/gi, " ");
  s = s.replace(/\s+/g, " ").trim();
  if (!s) return "PERA Document";
  const words = s.split(" ").map((w, i) => {
    if (/^\d{4}$/.test(w)) return w;
    const up = w.toUpperCase();
    if (DOC_KEEP_UPPER.has(up)) return up;
    const lowered = w.toLowerCase();
    if (i > 0 && DOC_SMALL_WORDS.has(lowered)) return lowered;
    return w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
  });
  return words.join(" ");
}

// Conservative no-answer detection: only used when no sources are present.
const NO_ANSWER_RE = /\b(could not find|cannot find|couldn'?t find|unable to find|not found|do(?:es)? not contain|don'?t have enough|do not have enough|not available in (?:the )?(?:provided|available)|available documents do not|no (?:specific|relevant|clear) (?:answer|information|mention)|insufficient information|not (?:explicitly|specifically|directly) (?:mentioned|stated|defined|covered|addressed))/i;

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
  const chipRefs = useRef<Map<number, HTMLButtonElement>>(new Map());

  // Citation numbers that have a matching source chip — only these are clickable.
  const citeSet = useMemo(() => {
    const s = new Set<number>();
    (message.references || []).forEach((ref, ri) => {
      const n = typeof ref.id === "number" ? ref.id : ri + 1;
      s.add(n);
    });
    return s;
  }, [message.references]);

  // Jump to the source chip matching an inline citation, then flash it.
  const jumpToSource = useCallback((n: number) => {
    const el = chipRefs.current.get(n);
    if (!el) return;
    el.scrollIntoView({ behavior: scrollBehavior(), block: "nearest", inline: "nearest" });
    el.classList.remove("ref-chip-flash");
    // Force reflow so the animation can re-trigger on repeat clicks.
    void el.offsetWidth;
    el.classList.add("ref-chip-flash");
    window.setTimeout(() => el.classList.remove("ref-chip-flash"), 1400);
  }, []);

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

  // No-answer state: only when the assistant returned no sources AND the text
  // reads like an honest "not found" response. Conservative on purpose.
  const isNoAnswer = useMemo(() => {
    if (message.role !== "assistant" || isTyping || message.failed) return false;
    const hasRefs = !!message.references && message.references.length > 0;
    if (hasRefs) return false;
    return NO_ANSWER_RE.test(message.content);
  }, [message.role, message.content, message.references, isTyping, message.failed]);

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
          {message.failed ? (
            /* Error state — clearer than a raw error string */
            <div className="ans-error-block">
              <div className="ans-error-title">
                <span className="ans-error-icon" aria-hidden>!</span>
                I could not generate an answer right now
              </div>
              <p className="ans-error-text">
                Something went wrong while generating the answer. Please try again
                in a moment.
              </p>
            </div>
          ) : (
            <>
              {/* Answer Section */}
              {showStructure && (
                <div className="ans-section-label">Answer</div>
              )}
              <div className="msg-bot-text">
                {isTyping ? (
                  /* While typing, render markdown on a SANITIZED copy of the
                     partial text so the formatting looks clean live (no raw
                     ### / **). The sanitizer hides only incomplete tokens;
                     completed pairs, lists, and citations render normally.
                     Citations are non-clickable until refs arrive on complete. */
                  renderMarkdown(sanitizeStreamingMarkdown(displayText), {
                    onCite: jumpToSource,
                    citeSet,
                  })
                ) : (
                  renderMarkdown(displayText, { onCite: jumpToSource, citeSet })
                )}
                {isTyping && <span className="typewriter-cursor" />}
              </div>
            </>
          )}

          {/* No-answer guidance — keeps the model's text, adds a recovery path */}
          {isNoAnswer && (
            <div className="ans-noanswer-block">
              <div className="ans-noanswer-title">
                Not found in the available PERA documents
              </div>
              <p className="ans-noanswer-text">
                Try rephrasing your question or ask about complaints, inspections,
                notices, rules, or appeals. For official confirmation, please
                contact PERA.
              </p>
              <div className="flex flex-wrap gap-1.5 mt-2">
                {[
                  { label: "Filing a Complaint", prompt: "Tell me how to file a complaint with PERA." },
                  { label: "Inspection Rights", prompt: "Explain my rights during a PERA inspection." },
                  { label: "Notices and Penalties", prompt: "Tell me about PERA notices and penalties." },
                  { label: "Contacting PERA", prompt: "How can I contact the official PERA office?" },
                ].map((t) => (
                  <button
                    key={t.label}
                    className="ans-related-chip"
                    onClick={() => onSendQuery?.(t.prompt)}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Authority / Source Block */}
          {showRefs && (
            <div className="ans-authority-block">
              <div className="ans-section-label">
                <span className="ans-verified-dot" />
                Source — Derived from Official PERA Documents
              </div>
              <div className="flex flex-wrap gap-1.5 mt-1.5">
                {message.references!.slice(0, 8).map((ref, ri) => {
                  const n = typeof ref.id === "number" ? ref.id : ri + 1;
                  const title = friendlyDocTitle(ref.document);
                  const truncated = title.length > 30 ? title.slice(0, 30) + "…" : title;
                  const page = ref.page_start || ref.page;
                  return (
                    <button
                      key={ri}
                      ref={(el) => {
                        if (el) chipRefs.current.set(n, el);
                        else chipRefs.current.delete(n);
                      }}
                      onClick={() => onOpenPdf?.(ref)}
                      className="ref-chip"
                      title={`${friendlyDocTitle(ref.document)}${page ? ` — Page ${page}` : ""}`}
                    >
                      <span className="ref-chip-idx">[{n}]</span>
                      <span className="ref-chip-name">{truncated}</span>
                      {page ? <span className="ref-chip-page">· p.{page}</span> : null}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Answer-level trust note — subtle, shown on completed answers */}
          {showStructure && (showRefs || isNoAnswer) && (
            <p className="ans-disclaimer">
              Based on available PERA documents. Verify case-specific matters with
              official PERA channels.
            </p>
          )}

          {/* Related Topics */}
          {showStructure && !isNoAnswer && relatedTopics.length > 0 && (
            <div className="ans-related-block">
              <div className="ans-section-label">Related Topics</div>
              <div className="flex flex-wrap gap-1.5 mt-1">
                {relatedTopics.map(({ topic, prompt }) => (
                  <button
                    key={topic}
                    className="ans-related-chip"
                    onClick={() => onSendQuery?.(prompt)}
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
