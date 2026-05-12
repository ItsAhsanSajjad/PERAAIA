"use client";

import { useEffect, useRef, useState, memo } from "react";

interface Props {
  url: string;
  title: string;
  onClose: () => void;
}

export const PdfModal = memo(function PdfModal({ url, title, onClose }: Props) {
  const modalRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const [iframeLoaded, setIframeLoaded] = useState(false);
  const [closing, setClosing] = useState(false);

  // Parse "<doc> — Page N" title into pieces for richer header
  const titleParts = (() => {
    const m = title.match(/^(.*?)\s*[—–-]\s*Page\s*(\d+)\s*$/i);
    if (m) return { doc: m[1].trim(), page: m[2] };
    return { doc: title, page: null as string | null };
  })();

  // Animated close — let the exit transition finish before unmount
  const requestClose = () => {
    if (closing) return;
    setClosing(true);
    window.setTimeout(onClose, 220);
  };

  // Reset loaded state when URL changes
  useEffect(() => {
    setIframeLoaded(false);
  }, [url]);

  // Store the element that had focus when the modal opened
  useEffect(() => {
    previousFocus.current = document.activeElement as HTMLElement | null;
    closeBtnRef.current?.focus();

    return () => {
      previousFocus.current?.focus();
    };
  }, []);

  // Escape key + focus trap
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        requestClose();
        return;
      }
      if (e.key === "Tab" && modalRef.current) {
        const focusable = modalRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isMobile = typeof window !== "undefined" && window.innerWidth < 768;

  return (
    <div
      className={`pdfm-root ${closing ? "pdfm-closing" : "pdfm-opening"}`}
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div className="pdfm-backdrop" onClick={requestClose} aria-hidden="true" />

      <div ref={modalRef} className="pdfm-shell">
        <div className="pdfm-glow" aria-hidden="true" />

        {/* Header */}
        <header className="pdfm-header">
          <div className="pdfm-title-wrap">
            <span className="pdfm-doc-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="8" y1="13" x2="16" y2="13" />
                <line x1="8" y1="17" x2="14" y2="17" />
              </svg>
            </span>
            <div className="pdfm-title-text">
              <div className="pdfm-doc-name" title={titleParts.doc}>
                {titleParts.doc}
              </div>
              {titleParts.page && (
                <span className="pdfm-page-badge">
                  <svg viewBox="0 0 24 24" width="10" height="10" fill="currentColor" aria-hidden>
                    <path d="M4 4h12l4 4v12a0 0 0 0 1 0 0H4z" opacity="0.25" />
                    <path d="M4 4h12l4 4v12H4z" fill="none" stroke="currentColor" strokeWidth="1.5" />
                  </svg>
                  Page {titleParts.page}
                </span>
              )}
            </div>
          </div>

          <div className="pdfm-actions">
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="pdfm-btn pdfm-btn-secondary"
            >
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                <polyline points="15 3 21 3 21 9" />
                <line x1="10" y1="14" x2="21" y2="3" />
              </svg>
              Open in tab
            </a>
            <button
              ref={closeBtnRef}
              onClick={requestClose}
              className="pdfm-btn pdfm-btn-close"
              aria-label="Close PDF viewer"
              title="Close (Esc)"
            >
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden>
                <path d="M6 6l12 12M18 6L6 18" />
              </svg>
            </button>
          </div>
        </header>

        {/* Progress shimmer while loading */}
        <div className={`pdfm-progress ${iframeLoaded ? "pdfm-progress-done" : ""}`} aria-hidden="true">
          <span />
        </div>

        {/* Content */}
        {isMobile ? (
          <div className="pdfm-mobile">
            <div className="pdfm-mobile-ic" aria-hidden>
              <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
            </div>
            <p className="pdfm-mobile-msg">PDF preview is not supported on this device.</p>
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="pdfm-mobile-cta"
            >
              Open PDF in Browser ↗
            </a>
          </div>
        ) : (
          <div className="pdfm-body">
            {!iframeLoaded && (
              <div className="pdfm-loading">
                <div className="pdfm-skeleton">
                  <div className="pdfm-skeleton-page">
                    <span className="pdfm-skeleton-line" style={{ width: "82%" }} />
                    <span className="pdfm-skeleton-line" style={{ width: "94%" }} />
                    <span className="pdfm-skeleton-line" style={{ width: "76%" }} />
                    <span className="pdfm-skeleton-line" style={{ width: "88%" }} />
                    <span className="pdfm-skeleton-line" style={{ width: "70%" }} />
                    <span className="pdfm-skeleton-line" style={{ width: "92%" }} />
                  </div>
                  <div className="pdfm-skeleton-page pdfm-skeleton-page-2">
                    <span className="pdfm-skeleton-line" style={{ width: "60%" }} />
                    <span className="pdfm-skeleton-line" style={{ width: "85%" }} />
                    <span className="pdfm-skeleton-line" style={{ width: "78%" }} />
                  </div>
                </div>
                <div className="pdfm-loading-pill">
                  <span className="pdfm-loading-dot" />
                  <span className="pdfm-loading-dot" />
                  <span className="pdfm-loading-dot" />
                  <span className="pdfm-loading-label">Preparing document</span>
                </div>
              </div>
            )}
            <iframe
              src={url}
              className={`pdfm-iframe ${iframeLoaded ? "pdfm-iframe-ready" : ""}`}
              title="PDF Viewer"
              onLoad={() => setIframeLoaded(true)}
            />
          </div>
        )}

        {/* Footer hint */}
        <footer className="pdfm-footer">
          <span className="pdfm-kbd">Esc</span>
          <span className="pdfm-footer-text">to close</span>
        </footer>
      </div>

      <style
        dangerouslySetInnerHTML={{
          __html: `
        .pdfm-root {
          position: fixed;
          inset: 0;
          z-index: 60;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 16px;
        }

        .pdfm-backdrop {
          position: absolute;
          inset: 0;
          background: rgba(6, 8, 14, 0.78);
          backdrop-filter: blur(14px) saturate(140%);
          -webkit-backdrop-filter: blur(14px) saturate(140%);
        }

        .pdfm-shell {
          position: relative;
          width: 100%;
          max-width: 1100px;
          height: 88vh;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          border-radius: 18px;
          background: linear-gradient(180deg, rgba(20, 22, 32, 0.96), rgba(14, 16, 24, 0.96));
          border: 1px solid rgba(255, 255, 255, 0.08);
          box-shadow:
            0 30px 80px -20px rgba(0, 0, 0, 0.7),
            0 0 0 1px rgba(212, 160, 23, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        html:not([data-theme="dark"]) .pdfm-shell {
          background: linear-gradient(180deg, #ffffff, #fafbfc);
          border: 1px solid rgba(17, 24, 39, 0.08);
          box-shadow:
            0 30px 80px -20px rgba(17, 24, 39, 0.28),
            0 0 0 1px rgba(212, 160, 23, 0.06);
        }

        /* Open animation */
        .pdfm-opening .pdfm-backdrop {
          animation: pdfm-fade-in 220ms ease-out;
        }
        .pdfm-opening .pdfm-shell {
          animation: pdfm-pop-in 360ms cubic-bezier(0.22, 1, 0.36, 1);
        }

        /* Close animation */
        .pdfm-closing .pdfm-backdrop {
          animation: pdfm-fade-out 220ms ease-in forwards;
        }
        .pdfm-closing .pdfm-shell {
          animation: pdfm-pop-out 220ms cubic-bezier(0.4, 0, 0.6, 1) forwards;
        }

        @keyframes pdfm-fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes pdfm-fade-out {
          from { opacity: 1; }
          to { opacity: 0; }
        }
        @keyframes pdfm-pop-in {
          0%   { transform: translateY(18px) scale(0.96); opacity: 0; }
          60%  { opacity: 1; }
          100% { transform: translateY(0) scale(1); opacity: 1; }
        }
        @keyframes pdfm-pop-out {
          from { transform: translateY(0) scale(1); opacity: 1; }
          to   { transform: translateY(10px) scale(0.98); opacity: 0; }
        }

        /* Ambient gold glow ring */
        .pdfm-glow {
          position: absolute;
          inset: -1px;
          pointer-events: none;
          border-radius: 18px;
          background: conic-gradient(
            from 0deg,
            rgba(212, 160, 23, 0) 0%,
            rgba(212, 160, 23, 0.35) 18%,
            rgba(212, 160, 23, 0) 32%,
            rgba(212, 160, 23, 0) 70%,
            rgba(212, 160, 23, 0.25) 86%,
            rgba(212, 160, 23, 0) 100%
          );
          opacity: 0.55;
          filter: blur(14px);
          animation: pdfm-glow-spin 22s linear infinite;
          z-index: -1;
        }
        @keyframes pdfm-glow-spin {
          to { transform: rotate(360deg); }
        }

        /* Header */
        .pdfm-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          padding: 14px 18px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.06);
          background: rgba(255, 255, 255, 0.015);
        }
        html:not([data-theme="dark"]) .pdfm-header {
          border-bottom-color: rgba(17, 24, 39, 0.06);
          background: rgba(0, 0, 0, 0.01);
        }

        .pdfm-title-wrap {
          display: flex;
          align-items: center;
          gap: 12px;
          min-width: 0;
          flex: 1;
        }
        .pdfm-doc-icon {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 34px;
          height: 34px;
          border-radius: 10px;
          background: linear-gradient(135deg, rgba(212, 160, 23, 0.18), rgba(212, 160, 23, 0.06));
          color: #d4a017;
          border: 1px solid rgba(212, 160, 23, 0.25);
          flex-shrink: 0;
        }
        .pdfm-title-text {
          display: flex;
          flex-direction: column;
          gap: 3px;
          min-width: 0;
        }
        .pdfm-doc-name {
          font-size: 13px;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.94);
          letter-spacing: 0.01em;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 580px;
        }
        html:not([data-theme="dark"]) .pdfm-doc-name {
          color: rgba(17, 24, 39, 0.92);
        }
        .pdfm-page-badge {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          font-size: 10px;
          font-weight: 600;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #d4a017;
          padding: 2px 8px;
          border-radius: 6px;
          background: rgba(212, 160, 23, 0.1);
          border: 1px solid rgba(212, 160, 23, 0.22);
          width: fit-content;
        }

        .pdfm-actions {
          display: flex;
          gap: 8px;
          align-items: center;
          flex-shrink: 0;
        }
        .pdfm-btn {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 7px 12px;
          font-size: 12px;
          font-weight: 600;
          border-radius: 9px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(255, 255, 255, 0.04);
          color: rgba(255, 255, 255, 0.85);
          transition: transform 0.16s ease, background 0.16s ease, color 0.16s ease, border-color 0.16s ease;
          cursor: pointer;
          text-decoration: none;
        }
        html:not([data-theme="dark"]) .pdfm-btn {
          border-color: rgba(17, 24, 39, 0.1);
          background: rgba(17, 24, 39, 0.03);
          color: rgba(17, 24, 39, 0.78);
        }
        .pdfm-btn:hover {
          background: rgba(212, 160, 23, 0.14);
          border-color: rgba(212, 160, 23, 0.35);
          color: #f3c244;
          transform: translateY(-1px);
        }
        .pdfm-btn-close {
          width: 32px;
          height: 32px;
          padding: 0;
          justify-content: center;
        }

        /* Progress shimmer */
        .pdfm-progress {
          position: relative;
          height: 2px;
          background: rgba(255, 255, 255, 0.04);
          overflow: hidden;
        }
        .pdfm-progress span {
          position: absolute;
          inset: 0;
          background: linear-gradient(
            90deg,
            rgba(212, 160, 23, 0) 0%,
            rgba(212, 160, 23, 0.85) 50%,
            rgba(212, 160, 23, 0) 100%
          );
          width: 30%;
          animation: pdfm-shimmer 1.2s linear infinite;
        }
        .pdfm-progress-done span {
          animation: pdfm-shimmer-complete 0.3s ease-out forwards;
        }
        @keyframes pdfm-shimmer {
          0%   { transform: translateX(-100%); }
          100% { transform: translateX(420%); }
        }
        @keyframes pdfm-shimmer-complete {
          0%   { width: 30%; opacity: 1; }
          100% { width: 100%; opacity: 0; transform: translateX(0); }
        }

        /* Body + iframe */
        .pdfm-body {
          position: relative;
          flex: 1;
          background: #fafafa;
        }
        .pdfm-iframe {
          width: 100%;
          height: 100%;
          border: 0;
          opacity: 0;
          transition: opacity 0.45s ease;
          background: #ffffff;
        }
        .pdfm-iframe-ready {
          opacity: 1;
        }

        /* Loading state */
        .pdfm-loading {
          position: absolute;
          inset: 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 22px;
          background: linear-gradient(180deg, rgba(18, 20, 28, 0.94), rgba(14, 16, 24, 0.94));
          z-index: 5;
        }
        html:not([data-theme="dark"]) .pdfm-loading {
          background: linear-gradient(180deg, #f4f6f9, #eef0f4);
        }

        .pdfm-skeleton {
          position: relative;
          width: 320px;
          max-width: 70%;
          display: flex;
          flex-direction: column;
          gap: 14px;
        }
        .pdfm-skeleton-page {
          padding: 18px 18px 22px;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(255, 255, 255, 0.06);
          display: flex;
          flex-direction: column;
          gap: 10px;
          animation: pdfm-skel-float 4s ease-in-out infinite;
        }
        .pdfm-skeleton-page-2 {
          animation-delay: 0.6s;
          opacity: 0.7;
        }
        html:not([data-theme="dark"]) .pdfm-skeleton-page {
          background: #ffffff;
          border-color: rgba(17, 24, 39, 0.06);
          box-shadow: 0 4px 14px -4px rgba(17, 24, 39, 0.12);
        }
        .pdfm-skeleton-line {
          display: block;
          height: 9px;
          border-radius: 3px;
          background: linear-gradient(
            90deg,
            rgba(255, 255, 255, 0.04) 0%,
            rgba(255, 255, 255, 0.12) 50%,
            rgba(255, 255, 255, 0.04) 100%
          );
          background-size: 200% 100%;
          animation: pdfm-skel-shimmer 1.8s ease-in-out infinite;
        }
        html:not([data-theme="dark"]) .pdfm-skeleton-line {
          background: linear-gradient(
            90deg,
            rgba(17, 24, 39, 0.06) 0%,
            rgba(17, 24, 39, 0.14) 50%,
            rgba(17, 24, 39, 0.06) 100%
          );
          background-size: 200% 100%;
        }
        @keyframes pdfm-skel-shimmer {
          0%   { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
        @keyframes pdfm-skel-float {
          0%, 100% { transform: translateY(0); }
          50%      { transform: translateY(-3px); }
        }

        .pdfm-loading-pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 8px 16px;
          border-radius: 999px;
          background: rgba(212, 160, 23, 0.1);
          border: 1px solid rgba(212, 160, 23, 0.25);
          color: #d4a017;
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 0.04em;
        }
        .pdfm-loading-dot {
          width: 5px;
          height: 5px;
          border-radius: 50%;
          background: #d4a017;
          animation: pdfm-dot-pulse 1.2s ease-in-out infinite;
        }
        .pdfm-loading-dot:nth-child(2) { animation-delay: 0.15s; }
        .pdfm-loading-dot:nth-child(3) { animation-delay: 0.3s; }
        @keyframes pdfm-dot-pulse {
          0%, 100% { transform: scale(0.7); opacity: 0.5; }
          50%      { transform: scale(1.15); opacity: 1; }
        }
        .pdfm-loading-label {
          margin-left: 4px;
        }

        /* Footer hint */
        .pdfm-footer {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 8px;
          padding: 8px 16px;
          border-top: 1px solid rgba(255, 255, 255, 0.05);
          font-size: 10px;
          color: rgba(255, 255, 255, 0.4);
          letter-spacing: 0.06em;
        }
        html:not([data-theme="dark"]) .pdfm-footer {
          border-top-color: rgba(17, 24, 39, 0.06);
          color: rgba(17, 24, 39, 0.45);
        }
        .pdfm-kbd {
          padding: 2px 7px;
          border-radius: 5px;
          background: rgba(255, 255, 255, 0.06);
          border: 1px solid rgba(255, 255, 255, 0.1);
          font-family: ui-monospace, "JetBrains Mono", Menlo, monospace;
          font-size: 9.5px;
          color: rgba(255, 255, 255, 0.7);
        }
        html:not([data-theme="dark"]) .pdfm-kbd {
          background: rgba(17, 24, 39, 0.04);
          border-color: rgba(17, 24, 39, 0.1);
          color: rgba(17, 24, 39, 0.6);
        }

        /* Mobile */
        .pdfm-mobile {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 18px;
          padding: 40px 24px;
        }
        .pdfm-mobile-ic {
          width: 64px;
          height: 64px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 14px;
          background: rgba(212, 160, 23, 0.12);
          color: #d4a017;
          border: 1px solid rgba(212, 160, 23, 0.22);
        }
        .pdfm-mobile-msg {
          font-size: 13px;
          text-align: center;
          color: rgba(255, 255, 255, 0.7);
          max-width: 280px;
        }
        html:not([data-theme="dark"]) .pdfm-mobile-msg {
          color: rgba(17, 24, 39, 0.65);
        }
        .pdfm-mobile-cta {
          padding: 11px 22px;
          border-radius: 12px;
          font-size: 13px;
          font-weight: 600;
          color: #1a1a1a;
          background: linear-gradient(135deg, #f3c244, #d4a017);
          text-decoration: none;
          box-shadow: 0 8px 24px -6px rgba(212, 160, 23, 0.55);
          transition: transform 0.16s ease, box-shadow 0.16s ease;
        }
        .pdfm-mobile-cta:hover {
          transform: translateY(-1px);
          box-shadow: 0 14px 32px -8px rgba(212, 160, 23, 0.7);
        }

        /* Compact header on narrow screens */
        @media (max-width: 640px) {
          .pdfm-shell { height: 92vh; border-radius: 14px; }
          .pdfm-doc-name { max-width: 180px; font-size: 12px; }
          .pdfm-btn-secondary { display: none; }
        }
      `,
        }}
      />
    </div>
  );
});
