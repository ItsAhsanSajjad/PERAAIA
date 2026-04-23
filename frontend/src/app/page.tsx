"use client";

import { useState, useCallback, useEffect } from "react";
import Image from "next/image";
import type { Reference } from "./lib/types";
import { buildPdfUrl } from "./lib/api";
import { useChatSessions } from "./hooks/useChatSessions";
import { useThemePreference } from "./hooks/useThemePreference";
import { useHealthCheck } from "./hooks/useHealthCheck";
import { ChatSidebar } from "./components/sidebar/ChatSidebar";
import { ChatWindow } from "./components/chat/ChatWindow";
import { Composer } from "./components/chat/Composer";
import { PdfModal } from "./components/pdf/PdfModal";
import { ToastContainer, showToast } from "./components/common/Toast";
import type { ConnectionStatus } from "./lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function StatusBadge({
  status,
  reason,
}: {
  status: ConnectionStatus;
  reason?: string;
}) {
  const colors: Record<ConnectionStatus, string> = {
    online: "var(--green)",
    connecting: "var(--gold)",
    offline: "var(--red)",
  };
  const labels: Record<ConnectionStatus, string> = {
    online: "System Status: Active",
    connecting: "Connecting",
    offline: "System Status: Offline",
  };
  const shortLabels: Record<ConnectionStatus, string> = {
    online: "Active",
    connecting: "Connecting",
    offline: "Offline",
  };
  const tooltip =
    status === "offline" && reason
      ? `${labels[status]} — ${reason}`
      : labels[status];
  return (
    <div
      className="flex items-center gap-1.5 px-2 sm:px-3 py-1.5 rounded-lg flex-shrink-0"
      style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
      title={tooltip}
    >
      <div
        className={`w-2 h-2 rounded-full flex-shrink-0 ${status === "online" ? "status-ping" : ""}`}
        style={{ background: colors[status] }}
      />
      <span
        className="text-[11px] font-medium tracking-wide hidden md:inline"
        style={{ color: "var(--text-secondary)" }}
      >
        {labels[status]}
      </span>
      <span
        className="text-[11px] font-medium tracking-wide md:hidden"
        style={{ color: "var(--text-secondary)" }}
      >
        {shortLabels[status]}
      </span>
    </div>
  );
}

function OfflineBanner({ reason }: { reason?: string }) {
  return (
    <div className="offline-banner-wrap" role="alert">
      <div className="offline-banner">
        <span className="offline-banner-icon" aria-hidden>
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="9" />
            <line x1="12" y1="8" x2="12" y2="13" />
            <circle cx="12" cy="16.5" r="0.6" fill="currentColor" />
          </svg>
        </span>
        <div className="offline-banner-body">
          <span className="offline-banner-title">System temporarily unavailable.</span>
          <span className="offline-banner-msg">
            {reason
              ? ` ${reason}. `
              : " "}
            Reconnecting automatically.
          </span>
        </div>
        <span className="offline-banner-retry" aria-label="Retrying connection">
          <span className="offline-banner-retry-dot" />
          <span className="offline-banner-retry-label">Retrying</span>
        </span>
      </div>
      <style dangerouslySetInnerHTML={{ __html: OFFLINE_BANNER_CSS }} />
    </div>
  );
}

const OFFLINE_BANNER_CSS = `
  .offline-banner-wrap {
    width: 100%;
    display: flex;
    justify-content: center;
    padding: 10px 1rem 0;
    pointer-events: none;
  }
  .offline-banner {
    pointer-events: auto;
    width: 100%;
    max-width: 48rem;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 10px;
    background: rgba(24, 18, 18, 0.55);
    border: 1px solid rgba(244, 63, 94, 0.22);
    box-shadow:
      0 4px 14px -8px rgba(244, 63, 94, 0.25),
      inset 0 1px 0 rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(10px) saturate(130%);
    -webkit-backdrop-filter: blur(10px) saturate(130%);
    animation: offline-banner-in 240ms cubic-bezier(0.22, 1, 0.36, 1);
    font-size: 0.78rem;
    line-height: 1.35;
  }
  html:not([data-theme="dark"]) .offline-banner {
    background: rgba(255, 251, 251, 0.92);
    border-color: rgba(190, 18, 60, 0.22);
    box-shadow:
      0 4px 14px -8px rgba(190, 18, 60, 0.2),
      0 0 0 1px rgba(0, 0, 0, 0.02);
  }
  .offline-banner-icon {
    flex-shrink: 0;
    width: 22px;
    height: 22px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(244, 63, 94, 0.14);
    color: #fca5a5;
    border: 1px solid rgba(244, 63, 94, 0.28);
  }
  html:not([data-theme="dark"]) .offline-banner-icon {
    background: rgba(225, 29, 72, 0.08);
    color: #be123c;
    border-color: rgba(190, 18, 60, 0.25);
  }
  .offline-banner-body {
    flex: 1;
    min-width: 0;
    color: rgba(254, 226, 226, 0.92);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  html:not([data-theme="dark"]) .offline-banner-body {
    color: #7f1d1d;
  }
  .offline-banner-title {
    font-weight: 600;
    letter-spacing: 0.01em;
    color: #fecaca;
  }
  html:not([data-theme="dark"]) .offline-banner-title {
    color: #9f1239;
  }
  .offline-banner-msg {
    font-weight: 400;
    color: rgba(254, 202, 202, 0.78);
  }
  html:not([data-theme="dark"]) .offline-banner-msg {
    color: rgba(127, 29, 29, 0.82);
  }
  .offline-banner-retry {
    flex-shrink: 0;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(251, 191, 36, 0.92);
  }
  html:not([data-theme="dark"]) .offline-banner-retry {
    color: #a16207;
  }
  .offline-banner-retry-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #fbbf24;
    box-shadow: 0 0 5px rgba(251, 191, 36, 0.7);
    animation: offline-banner-dot 1.4s ease-in-out infinite;
  }
  @keyframes offline-banner-in {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes offline-banner-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.45; transform: scale(0.8); }
  }
  @media (max-width: 640px) {
    .offline-banner-wrap {
      padding: 8px 0.75rem 0;
    }
    .offline-banner {
      gap: 8px;
      padding: 8px 10px;
      font-size: 0.74rem;
    }
    .offline-banner-body {
      white-space: normal;
    }
    .offline-banner-retry-label {
      display: none;
    }
  }
`;

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [pdfModal, setPdfModal] = useState<{ url: string; title: string } | null>(null);

  const { theme, toggleTheme } = useThemePreference();
  const { status, reason, reportSuccess, reportFailure } = useHealthCheck();
  // System-offline mode has been permanently disabled — the chatbot
  // must always respond regardless of OpenAI quota state. Leave the
  // hook mounted (it still pings /health for latency-style monitoring)
  // but never render the banner / disable the composer.
  const isOffline = false;

  /* — Startup: confirm API origin — */
  useEffect(() => {
    console.log("[PERA] API_URL →", API_URL);
  }, []);

  /* — Recovery toast disabled along with system-offline mode — */

  const {
    messages,
    chatHistory,
    currentChatId,
    loading,
    failedPrompt,
    lastBotIsNew,
    clearNewFlag,
    sendMessage,
    retryLastFailed,
    startNewChat,
    loadChat,
    deleteChat,
  } = useChatSessions({ reportSuccess, reportFailure });

  const handleSend = useCallback(
    (text: string) => {
      sendMessage(text);
    },
    [sendMessage],
  );

  const handleOpenPdf = useCallback((ref: Reference) => {
    const pg = ref.page_start || ref.page || 1;
    const url = buildPdfUrl(ref);
    setPdfModal({ url, title: `${ref.document} — Page ${pg}` });
  }, []);

  const handleSidebarToggle = useCallback(() => {
    setSidebarOpen((p) => !p);
  }, []);

  const handleNewChat = useCallback(() => {
    startNewChat();
    setSidebarOpen(false);
  }, [startNewChat]);

  const handleLoadChat = useCallback(
    (session: Parameters<typeof loadChat>[0]) => {
      loadChat(session);
      setSidebarOpen(false);
    },
    [loadChat],
  );

  return (
    <div
      className="flex overflow-hidden relative"
      style={{ background: "var(--bg-page)", height: "100dvh" }}
    >
      {/* Subtle ambient — reduced for institutional feel */}
      <div className="ambient-bg" />

      {/* Sidebar */}
      <ChatSidebar
        isOpen={sidebarOpen}
        onToggle={handleSidebarToggle}
        chatHistory={chatHistory}
        currentChatId={currentChatId}
        onNewChat={handleNewChat}
        onLoadChat={handleLoadChat}
        onDeleteChat={deleteChat}
      />

      {/* Main Area */}
      <main className="flex-1 flex flex-col relative z-10 min-w-0">
        {/* Institutional Header */}
        <header className="inst-header flex items-center justify-between gap-2 px-3 sm:px-4 md:px-6 py-2.5 z-20 relative">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
            <button
              onClick={handleSidebarToggle}
              className="p-2 rounded-lg transition-colors flex-shrink-0"
              style={{ color: "var(--text-secondary)" }}
              aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
              onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="15" y2="12" /><line x1="3" y1="18" x2="18" y2="18" />
              </svg>
            </button>
            <Image
              src="/pera_logo.png"
              alt="PERA Emblem"
              width={32}
              height={32}
              className="rounded-lg flex-shrink-0"
              priority
            />
            <div className="min-w-0 flex-1">
              <h1 className="font-semibold text-sm tracking-wide truncate" style={{ color: "var(--text-primary)" }}>
                Ask PERA Assistant
              </h1>
              <p
                className="text-[11px] font-medium tracking-wide hidden sm:block truncate"
                style={{ color: "var(--text-secondary)" }}
              >
                Punjab Enforcement &amp; Regulatory Authority
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
            <button
              onClick={toggleTheme}
              className="theme-toggle"
              role="switch"
              aria-checked={theme === "dark"}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            >
              <div className="theme-toggle-knob">{theme === "dark" ? "🌙" : "☀️"}</div>
            </button>
            <StatusBadge status={status} reason={reason} />
          </div>
        </header>

        {/* Offline Banner */}
        {isOffline && <OfflineBanner reason={reason} />}

        {/* Chat Messages */}
        <ChatWindow
          messages={messages}
          loading={loading}
          failedPrompt={failedPrompt}
          lastBotIsNew={lastBotIsNew}
          clearNewFlag={clearNewFlag}
          onSendSuggestion={handleSend}
          onOpenPdf={handleOpenPdf}
          onRetry={retryLastFailed}
        />

        {/* Query Interface */}
        <Composer onSend={handleSend} disabled={loading} systemOffline={isOffline} />
      </main>

      {/* PDF Viewer Modal */}
      {pdfModal && (
        <PdfModal
          url={pdfModal.url}
          title={pdfModal.title}
          onClose={() => setPdfModal(null)}
        />
      )}

      {/* Toasts */}
      <ToastContainer />
    </div>
  );
}
