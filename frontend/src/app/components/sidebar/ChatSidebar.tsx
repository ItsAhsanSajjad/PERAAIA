"use client";

import Image from "next/image";
import { memo, useMemo } from "react";
import type { ChatSession } from "../../lib/types";

interface Props {
  isOpen: boolean;
  onToggle: () => void;
  chatHistory: ChatSession[];
  currentChatId: string;
  onNewChat: () => void;
  onLoadChat: (session: ChatSession) => void;
  onDeleteChat: (id: string) => void;
}

function formatRelative(ts?: number): string {
  if (!ts) return "";
  try {
    const d = typeof ts === "number" ? ts : Date.parse(String(ts));
    if (!d || isNaN(d)) return "";
    const diff = Date.now() - d;
    const sec = Math.floor(diff / 1000);
    if (sec < 60) return "just now";
    const min = Math.floor(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.floor(min / 60);
    if (hr < 24) return `${hr}h ago`;
    const day = Math.floor(hr / 24);
    if (day < 7) return `${day}d ago`;
    return new Date(d).toLocaleDateString();
  } catch {
    return "";
  }
}

type TSFields = { updatedAt?: number; timestamp?: number; createdAt?: number };
function getTS(s: ChatSession): number {
  const raw = s as unknown as TSFields;
  const v = raw.updatedAt ?? raw.timestamp ?? raw.createdAt ?? 0;
  if (typeof v === "number") return v;
  const p = Date.parse(String(v));
  return isNaN(p) ? 0 : p;
}

export const ChatSidebar = memo(function ChatSidebar({
  isOpen,
  onToggle,
  chatHistory,
  currentChatId,
  onNewChat,
  onLoadChat,
  onDeleteChat,
}: Props) {
  const sessionCount = chatHistory.length;

  const groupedHistory = useMemo(() => {
    const groups: { label: string; items: ChatSession[] }[] = [
      { label: "Today", items: [] },
      { label: "This week", items: [] },
      { label: "Earlier", items: [] },
    ];
    const now = Date.now();
    const ONE_DAY = 24 * 3600 * 1000;
    for (const s of chatHistory) {
      const raw = getTS(s);
      if (!raw) {
        groups[2].items.push(s);
        continue;
      }
      const diff = now - raw;
      if (diff < ONE_DAY) groups[0].items.push(s);
      else if (diff < 7 * ONE_DAY) groups[1].items.push(s);
      else groups[2].items.push(s);
    }
    return groups.filter((g) => g.items.length > 0);
  }, [chatHistory]);

  return (
    <>
      <aside
        className={`sb ${isOpen ? "sb-open" : "sb-closed"}`}
        aria-label="Chat history sidebar"
      >
        {isOpen && (
          <div className="sb-inner">
            {/* Header — workspace card */}
            <header className="sb-header">
              <div className="sb-workspace">
                <div className="sb-logo">
                  <Image src="/pera_logo.png" alt="" width={28} height={28} className="sb-logo-img" />
                </div>
                <div className="sb-header-text">
                  <div className="sb-brand">Ask PERA</div>
                  <div className="sb-org">
                    Punjab Enforcement &amp; Regulatory Authority
                  </div>
                </div>
                <span className="sb-workspace-badge" title="Official Government System">
                  <svg viewBox="0 0 24 24" width="11" height="11" fill="currentColor" aria-hidden>
                    <path d="M9 12l2 2 4-4 1.4 1.4L11 16.8 7.6 13.4 9 12zM12 2l8 4v6c0 5.5-3.6 10-8 11-4.4-1-8-5.5-8-11V6l8-4z" />
                  </svg>
                </span>
              </div>
            </header>

            {/* Primary action */}
            <button onClick={onNewChat} className="sb-new" aria-label="Start a new conversation">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M12 5v14M5 12h14" />
              </svg>
              <span>New Conversation</span>
              <span className="sb-new-kbd">⌘K</span>
            </button>

            {/* Section header */}
            <div className="sb-list-head">
              <span className="sb-list-title">
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden className="sb-list-title-ic">
                  <path d="M3 12h18M3 6h18M3 18h18" />
                </svg>
                Conversations
              </span>
              {sessionCount > 0 && <span className="sb-list-count">{sessionCount}</span>}
            </div>

            <div className="sb-list" role="list">
              {chatHistory.length === 0 ? (
                <div className="sb-empty">
                  <div className="sb-empty-ic" aria-hidden>
                    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                    </svg>
                  </div>
                  <div className="sb-empty-title">No conversations</div>
                  <div className="sb-empty-sub">Start a new conversation to begin.</div>
                </div>
              ) : (
                groupedHistory.map((group) => (
                  <div key={group.label} className="sb-group">
                    <div className="sb-group-label">{group.label}</div>
                    {group.items.map((s) => {
                      const isActive = s.id === currentChatId;
                      const rel = formatRelative(getTS(s));
                      return (
                        <div
                          key={s.id}
                          role="button"
                          tabIndex={0}
                          onClick={() => onLoadChat(s)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              onLoadChat(s);
                            }
                          }}
                          className={`sb-item ${isActive ? "sb-item-active" : ""}`}
                        >
                          <span className="sb-item-body">
                            <span className="sb-item-title">{s.title}</span>
                            {rel && <span className="sb-item-time">{rel}</span>}
                          </span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteChat(s.id);
                            }}
                            className="sb-item-del"
                            aria-label={`Delete chat: ${s.title}`}
                          >
                            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden>
                              <path d="M6 6l12 12M18 6L6 18" />
                            </svg>
                          </button>
                        </div>
                      );
                    })}
                  </div>
                ))
              )}
            </div>

            {/* Footer */}
            <footer className="sb-footer">
              <span className="sb-footer-status">
                <span className="sb-footer-dot" />
                Secure session
              </span>
              <span className="sb-footer-brand">PERA&nbsp;AI&nbsp;Team</span>
            </footer>
          </div>
        )}
      </aside>

      {isOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/40 md:hidden"
          onClick={onToggle}
          aria-hidden="true"
        />
      )}

      <style dangerouslySetInnerHTML={{ __html: `
        /* ============================================================
           SIDEBAR — clean institutional design
           - Dark: deep neutral #0f1117 with pale type, gold only for
             the primary CTA and brand accent
           - Light: pure white surface with charcoal type
           - No decorative halos, no rotating rings, no ambient orbs
           ============================================================ */
        .sb {
          position: relative;
          z-index: 30;
          overflow: hidden;
          transition: width 280ms cubic-bezier(0.22, 1, 0.36, 1),
            transform 280ms cubic-bezier(0.22, 1, 0.36, 1);
          background: #0b0d14;
          border-right: 1px solid rgba(255, 255, 255, 0.08);
          box-shadow: 2px 0 24px -12px rgba(0, 0, 0, 0.6);
        }
        html:not([data-theme="dark"]) .sb {
          background: #ffffff;
          border-right-color: rgba(17, 24, 39, 0.08);
          box-shadow: 2px 0 24px -18px rgba(17, 24, 39, 0.12);
        }
        .sb.sb-open { width: 276px; transform: translateX(0); }
        .sb.sb-closed { width: 0; transform: translateX(-100%); }

        /* Mobile: overlay instead of consuming flex width (matches the
           md:hidden backdrop). Prevents the main column being squished
           when the sidebar opens on small screens. */
        @media (max-width: 767px) {
          .sb {
            position: fixed;
            top: 0;
            left: 0;
            height: 100%;
          }
        }

        .sb-inner {
          display: flex;
          flex-direction: column;
          height: 100%;
          width: 276px;
          padding: 16px 12px 10px;
          animation: sb-in 300ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }
        @keyframes sb-in {
          from { opacity: 0; transform: translateX(-4px); }
          to { opacity: 1; transform: translateX(0); }
        }

        /* -- Header: workspace card -- */
        .sb-header {
          padding-bottom: 14px;
          margin-bottom: 14px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        html:not([data-theme="dark"]) .sb-header {
          border-bottom-color: rgba(17, 24, 39, 0.08);
        }
        .sb-workspace {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 10px 10px 10px;
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.06);
        }
        html:not([data-theme="dark"]) .sb-workspace {
          background: #f9fafb;
          border-color: rgba(17, 24, 39, 0.08);
        }
        .sb-logo {
          width: 34px;
          height: 34px;
          border-radius: 9px;
          background: rgba(212, 160, 23, 0.1);
          border: 1px solid rgba(212, 160, 23, 0.2);
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
          overflow: hidden;
        }
        html:not([data-theme="dark"]) .sb-logo {
          background: #ffffff;
          border-color: rgba(184, 134, 11, 0.25);
        }
        .sb-logo-img { border-radius: 6px; }
        .sb-header-text { min-width: 0; flex: 1; }
        .sb-brand {
          font-size: 0.88rem;
          font-weight: 700;
          letter-spacing: -0.005em;
          color: #f3f4f6;
          line-height: 1.2;
        }
        html:not([data-theme="dark"]) .sb-brand {
          color: #111827;
        }
        .sb-org {
          margin-top: 2px;
          font-size: 0.66rem;
          font-weight: 400;
          color: rgba(243, 244, 246, 0.5);
          line-height: 1.3;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        html:not([data-theme="dark"]) .sb-org {
          color: rgba(17, 24, 39, 0.55);
        }
        .sb-workspace-badge {
          flex-shrink: 0;
          width: 22px;
          height: 22px;
          border-radius: 6px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background: rgba(16, 185, 129, 0.14);
          color: #10b981;
          border: 1px solid rgba(16, 185, 129, 0.28);
        }
        html:not([data-theme="dark"]) .sb-workspace-badge {
          background: rgba(16, 185, 129, 0.1);
          color: #059669;
          border-color: rgba(16, 185, 129, 0.3);
        }

        /* -- New Conversation -- */
        .sb-new {
          display: flex;
          align-items: center;
          gap: 8px;
          width: 100%;
          height: 38px;
          padding: 0 12px 0 14px;
          margin-bottom: 16px;
          border-radius: 8px;
          border: 0;
          background: #d4a017;
          color: #1a1307;
          font-size: 0.8rem;
          font-weight: 600;
          letter-spacing: 0.005em;
          cursor: pointer;
          transition: background 160ms ease, transform 160ms ease,
            box-shadow 200ms ease;
        }
        .sb-new > span:first-of-type {
          flex: 1;
          text-align: left;
        }
        .sb-new:hover {
          background: #e9c464;
          transform: translateY(-1px);
          box-shadow: 0 8px 16px -8px rgba(212, 160, 23, 0.55);
        }
        .sb-new:active {
          transform: translateY(0);
        }
        .sb-new:focus-visible {
          outline: 2px solid rgba(212, 160, 23, 0.55);
          outline-offset: 2px;
        }
        html:not([data-theme="dark"]) .sb-new {
          background: #b8860b;
          color: #ffffff;
        }
        html:not([data-theme="dark"]) .sb-new:hover {
          background: #d4a017;
          box-shadow: 0 8px 16px -8px rgba(184, 134, 11, 0.5);
        }
        .sb-new-kbd {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          height: 20px;
          padding: 0 6px;
          border-radius: 5px;
          font-size: 0.62rem;
          font-weight: 700;
          letter-spacing: 0.04em;
          background: rgba(26, 19, 7, 0.15);
          color: rgba(26, 19, 7, 0.75);
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }
        html:not([data-theme="dark"]) .sb-new-kbd {
          background: rgba(255, 255, 255, 0.2);
          color: rgba(255, 255, 255, 0.85);
        }

        /* -- List head -- */
        .sb-list-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 8px 8px;
        }
        .sb-list-title {
          display: inline-flex;
          align-items: center;
          gap: 7px;
          font-size: 0.72rem;
          font-weight: 600;
          color: rgba(243, 244, 246, 0.88);
          letter-spacing: 0.005em;
        }
        html:not([data-theme="dark"]) .sb-list-title {
          color: rgba(17, 24, 39, 0.85);
        }
        .sb-list-title-ic {
          color: rgba(243, 244, 246, 0.5);
        }
        html:not([data-theme="dark"]) .sb-list-title-ic {
          color: rgba(17, 24, 39, 0.5);
        }
        .sb-list-count {
          font-size: 0.64rem;
          font-weight: 600;
          font-variant-numeric: tabular-nums;
          color: rgba(243, 244, 246, 0.6);
          background: rgba(255, 255, 255, 0.06);
          border: 1px solid rgba(255, 255, 255, 0.06);
          padding: 2px 8px;
          border-radius: 999px;
        }
        html:not([data-theme="dark"]) .sb-list-count {
          color: rgba(17, 24, 39, 0.65);
          background: rgba(17, 24, 39, 0.04);
          border-color: rgba(17, 24, 39, 0.08);
        }

        /* -- List -- */
        .sb-list {
          flex: 1;
          overflow-y: auto;
          overflow-x: hidden;
          padding-right: 2px;
          margin-right: -2px;
        }
        .sb-list::-webkit-scrollbar { width: 4px; }
        .sb-list::-webkit-scrollbar-track { background: transparent; }
        .sb-list::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 999px;
        }
        html:not([data-theme="dark"]) .sb-list::-webkit-scrollbar-thumb {
          background: rgba(17, 24, 39, 0.12);
        }

        .sb-group { margin-bottom: 8px; }
        .sb-group-label {
          font-size: 0.62rem;
          letter-spacing: 0.08em;
          font-weight: 600;
          color: rgba(243, 244, 246, 0.4);
          text-transform: uppercase;
          padding: 8px 8px 6px;
        }
        html:not([data-theme="dark"]) .sb-group-label {
          color: rgba(17, 24, 39, 0.45);
        }

        .sb-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 9px 10px;
          border-radius: 8px;
          cursor: pointer;
          color: rgba(243, 244, 246, 0.78);
          transition: background 140ms ease, color 140ms ease;
        }
        html:not([data-theme="dark"]) .sb-item {
          color: rgba(17, 24, 39, 0.82);
        }
        .sb-item:hover {
          background: rgba(255, 255, 255, 0.04);
          color: #f3f4f6;
        }
        html:not([data-theme="dark"]) .sb-item:hover {
          background: rgba(17, 24, 39, 0.04);
          color: #111827;
        }
        .sb-item:focus-visible {
          outline: 2px solid rgba(212, 160, 23, 0.5);
          outline-offset: -2px;
        }
        .sb-item-active {
          background: rgba(255, 255, 255, 0.06);
          color: #ffffff;
          box-shadow: inset 2px 0 0 #d4a017;
        }
        html:not([data-theme="dark"]) .sb-item-active {
          background: rgba(17, 24, 39, 0.05);
          color: #111827;
          box-shadow: inset 2px 0 0 #b8860b;
        }

        .sb-item-body {
          flex: 1;
          min-width: 0;
          display: flex;
          flex-direction: column;
          gap: 1px;
        }
        .sb-item-title {
          font-size: 0.78rem;
          font-weight: 500;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          line-height: 1.3;
        }
        .sb-item-time {
          font-size: 0.64rem;
          color: rgba(243, 244, 246, 0.4);
          letter-spacing: 0.01em;
        }
        html:not([data-theme="dark"]) .sb-item-time {
          color: rgba(17, 24, 39, 0.5);
        }
        .sb-item-del {
          opacity: 0;
          width: 22px;
          height: 22px;
          border-radius: 6px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border: 0;
          background: transparent;
          color: rgba(243, 244, 246, 0.55);
          cursor: pointer;
          flex-shrink: 0;
          transition: opacity 160ms ease, background 160ms ease, color 160ms ease;
        }
        html:not([data-theme="dark"]) .sb-item-del {
          color: rgba(17, 24, 39, 0.55);
        }
        .sb-item:hover .sb-item-del,
        .sb-item:focus-visible .sb-item-del { opacity: 1; }
        .sb-item-del:hover {
          background: rgba(239, 68, 68, 0.15);
          color: #fca5a5;
        }
        html:not([data-theme="dark"]) .sb-item-del:hover {
          background: rgba(239, 68, 68, 0.1);
          color: #dc2626;
        }

        /* -- Empty -- */
        .sb-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: 32px 12px;
        }
        .sb-empty-ic {
          width: 44px;
          height: 44px;
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(255, 255, 255, 0.08);
          color: rgba(243, 244, 246, 0.5);
          margin-bottom: 12px;
        }
        html:not([data-theme="dark"]) .sb-empty-ic {
          background: #f9fafb;
          border-color: rgba(17, 24, 39, 0.08);
          color: rgba(17, 24, 39, 0.45);
        }
        .sb-empty-title {
          font-size: 0.8rem;
          font-weight: 600;
          color: rgba(243, 244, 246, 0.85);
          margin-bottom: 3px;
        }
        html:not([data-theme="dark"]) .sb-empty-title {
          color: rgba(17, 24, 39, 0.82);
        }
        .sb-empty-sub {
          font-size: 0.72rem;
          color: rgba(243, 244, 246, 0.5);
          line-height: 1.45;
        }
        html:not([data-theme="dark"]) .sb-empty-sub {
          color: rgba(17, 24, 39, 0.55);
        }

        /* -- Footer -- */
        .sb-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          padding: 10px 6px 2px;
          margin-top: 10px;
          border-top: 1px solid rgba(255, 255, 255, 0.06);
        }
        html:not([data-theme="dark"]) .sb-footer {
          border-top-color: rgba(17, 24, 39, 0.08);
        }
        .sb-footer-status {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-size: 0.64rem;
          font-weight: 500;
          color: rgba(243, 244, 246, 0.55);
          letter-spacing: 0.005em;
        }
        html:not([data-theme="dark"]) .sb-footer-status {
          color: rgba(17, 24, 39, 0.6);
        }
        .sb-footer-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #10b981;
          box-shadow: 0 0 5px rgba(16, 185, 129, 0.7);
        }
        .sb-footer-brand {
          font-size: 0.62rem;
          font-weight: 600;
          color: rgba(243, 244, 246, 0.5);
          letter-spacing: 0.01em;
        }
        html:not([data-theme="dark"]) .sb-footer-brand {
          color: rgba(17, 24, 39, 0.55);
        }
      ` }} />
    </>
  );
});
