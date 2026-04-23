"use client";

import { memo, useState, useCallback, type KeyboardEvent, useEffect } from "react";
import { useAutoResizeTextarea } from "../../hooks/useAutoResizeTextarea";
import { useVoiceRecorder } from "../../hooks/useVoiceRecorder";
import type { VoiceState } from "../../lib/types";

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
  systemOffline?: boolean;
}

function formatSeconds(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export const Composer = memo(function Composer({ onSend, disabled, systemOffline }: Props) {
  const [input, setInput] = useState("");
  const { ref: textareaRef, resize } = useAutoResizeTextarea(150);

  const handleTranscribed = useCallback((text: string) => {
    setInput(text);
    textareaRef.current?.focus();
  }, [textareaRef]);

  const { voiceState, elapsed, errorMessage, toggleRecording, clearError } =
    useVoiceRecorder(handleTranscribed);

  const handleSend = useCallback(() => {
    if (!input.trim() || disabled || systemOffline) return;
    onSend(input);
    setInput("");
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [input, disabled, systemOffline, onSend, textareaRef]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const handleInput = useCallback(() => {
    resize();
  }, [resize]);

  const isRecordingOrTranscribing = voiceState === "recording" || voiceState === "transcribing";

  return (
    <div className="px-4 md:px-0 pt-3 pb-6 relative z-10" style={{ background: "var(--bg-page)", borderTop: "1px solid var(--border)" }}>
      <div className="max-w-3xl mx-auto">
        {/* Voice error banner */}
        {voiceState === "error" && errorMessage && (
          <div
            className="mb-2 px-3 py-2 rounded-xl text-xs flex items-center justify-between"
            style={{ background: "rgba(239,68,68,0.1)", color: "var(--red)" }}
          >
            <span>🎤 {errorMessage}</span>
            <button
              onClick={clearError}
              className="ml-2 font-bold text-sm"
              aria-label="Dismiss error"
            >
              ✕
            </button>
          </div>
        )}

        {/* Recording / Transcribing bar — inline pro audio app style */}
        {voiceState === "recording" && (
          <div className="rec-pro" role="status" aria-live="polite">
            <div className="rec-pro-inner">
              <span className="rec-pro-live" aria-hidden>
                <span className="rec-pro-live-dot" />
                REC
              </span>
              <div className="rec-pro-divider" />
              <span className="rec-pro-timer" aria-label={`Elapsed time ${formatSeconds(elapsed)}`}>
                {formatSeconds(elapsed)}
              </span>
              <div className="rec-pro-divider" />
              <div className="rec-pro-visualizer" aria-hidden>
                {Array.from({ length: 32 }).map((_, i) => (
                  <span
                    key={i}
                    className="rec-pro-bar"
                    style={{ "--i": i } as React.CSSProperties}
                  />
                ))}
              </div>
              <div className="rec-pro-divider" />
              <span className="rec-pro-status">
                <span className="rec-pro-status-dot" />
                Listening
              </span>
            </div>
          </div>
        )}
        {voiceState === "transcribing" && (
          <div className="trans-pro" role="status" aria-live="polite">
            <div className="trans-pro-inner">
              <span className="trans-pro-label">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M20 12a8 8 0 1 1-3-6.24" />
                </svg>
                PROCESSING
              </span>
              <div className="rec-pro-divider" />
              <span className="trans-pro-text">
                Transcribing your audio
              </span>
              <div className="trans-pro-track" aria-hidden>
                <span className="trans-pro-fill" />
              </div>
            </div>
          </div>
        )}
        <style dangerouslySetInnerHTML={{ __html: REC_CSS }} />

        <div className="flex items-end gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              onInput={handleInput}
              placeholder={
                systemOffline
                  ? "System offline — please try again later when the system is active"
                  : "Ask about PERA regulations, governance, enforcement, or KPIs"
              }
              rows={1}
              className="chat-input w-full resize-none px-4 py-3 pr-28 text-sm"
              style={{ maxHeight: 150, minHeight: 48 }}
              disabled={disabled || isRecordingOrTranscribing || systemOffline}
              aria-label="Message input"
            />

            {/* Voice Button */}
            <button
              onClick={toggleRecording}
              disabled={disabled || voiceState === "transcribing" || systemOffline}
              className={`voice-btn ${voiceState === "recording" ? "voice-btn-rec" : ""}`}
              aria-label={voiceState === "recording" ? "Stop recording" : "Start voice input"}
            >
              {voiceState === "recording" ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" className="voice-btn-ic">
                  <rect x="6" y="6" width="12" height="12" rx="3" />
                </svg>
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="voice-btn-ic">
                  <rect x="9" y="3" width="6" height="12" rx="3" />
                  <path d="M5 11a7 7 0 0 0 14 0" />
                  <line x1="12" y1="18" x2="12" y2="22" />
                </svg>
              )}
            </button>
            <style dangerouslySetInnerHTML={{ __html: VOICE_BTN_CSS }} />

            {/* Send Button */}
            <button
              onClick={handleSend}
              disabled={!input.trim() || disabled || isRecordingOrTranscribing || systemOffline}
              className={`send-btn-pro ${input.trim() && !disabled && !isRecordingOrTranscribing && !systemOffline ? "is-active" : ""}`}
              aria-label="Send message"
            >
              <span className="send-btn-glow" aria-hidden />
              <span className="send-btn-sheen" aria-hidden />
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="send-btn-ic">
                <path d="M4 12h16M14 6l6 6-6 6" />
              </svg>
            </button>
            <style dangerouslySetInnerHTML={{ __html: SEND_BTN_CSS }} />
          </div>
        </div>


      </div>
    </div>
  );
});

const REC_CSS = `
          @keyframes rec-pro-in {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
          }
          @keyframes rec-pro-blink {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(239,68,68,0.7), 0 0 10px rgba(239,68,68,0.8); }
            50% { opacity: 0.4; box-shadow: 0 0 0 6px rgba(239,68,68,0), 0 0 14px rgba(239,68,68,0.5); }
          }
          @keyframes rec-pro-bar {
            0% { height: 3px; }
            100% { height: var(--peak, 14px); }
          }
          @keyframes rec-pro-status-pulse {
            0%, 100% { opacity: 0.55; }
            50% { opacity: 1; }
          }
          @keyframes rec-pro-track-shine {
            0% { left: -40%; }
            100% { left: 140%; }
          }
          @keyframes rec-pro-fill-march {
            0% { background-position: 0% 50%; }
            100% { background-position: 200% 50%; }
          }

          /* =================================================
             PROFESSIONAL AUDIO-APP STYLED RECORDING BAR
             Inspired by Google Recorder / Voice Memos /
             professional DAW transport controls.
             ================================================= */
          .rec-pro {
            margin-bottom: 10px;
            animation: rec-pro-in 340ms cubic-bezier(0.22, 1, 0.36, 1) both;
            width: 100%;
            max-width: 100%;
          }
          .rec-pro-inner {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 10px 16px 10px 14px;
            height: 44px;
            border-radius: 12px;
            background: linear-gradient(
              180deg,
              rgba(20, 22, 30, 0.96) 0%,
              rgba(14, 15, 20, 0.96) 100%
            );
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow:
              0 8px 24px -10px rgba(0, 0, 0, 0.55),
              inset 0 1px 0 rgba(255, 255, 255, 0.04);
            color: #e5e7eb;
            font-size: 11px;
            letter-spacing: 0.14em;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
          }

          .rec-pro-live {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            color: #ef4444;
            font-weight: 700;
            letter-spacing: 0.22em;
            flex-shrink: 0;
          }
          .rec-pro-live-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #ef4444;
            animation: rec-pro-blink 1.2s ease-in-out infinite;
          }

          .rec-pro-divider {
            width: 1px;
            height: 18px;
            background: rgba(255, 255, 255, 0.08);
            flex-shrink: 0;
          }

          .rec-pro-timer {
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: #f9fafb;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            flex-shrink: 0;
          }

          /* Audio visualizer — 32 bars with pseudo-random heights
             that animate with their own phase so they feel live */
          .rec-pro-visualizer {
            flex: 1;
            height: 22px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 2px;
            min-width: 160px;
            overflow: hidden;
          }
          .rec-pro-bar {
            width: 2px;
            height: 3px;
            border-radius: 1px;
            background: linear-gradient(
              180deg,
              #fca5a5 0%,
              #ef4444 50%,
              #b91c1c 100%
            );
            box-shadow: 0 0 4px rgba(239, 68, 68, 0.6);
            animation: rec-pro-bar 0.6s ease-in-out infinite alternate;
            animation-delay: calc(var(--i) * -73ms);
          }
          /* Varied peak heights for realistic waveform silhouette */
          .rec-pro-bar:nth-child(3n) { --peak: 8px; }
          .rec-pro-bar:nth-child(3n+1) { --peak: 14px; }
          .rec-pro-bar:nth-child(3n+2) { --peak: 18px; }
          .rec-pro-bar:nth-child(4n) { --peak: 6px; }
          .rec-pro-bar:nth-child(5n) { --peak: 20px; }
          .rec-pro-bar:nth-child(7n) { --peak: 10px; }
          .rec-pro-bar:nth-child(11n) { --peak: 22px; }

          .rec-pro-status {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            color: rgba(229, 231, 235, 0.65);
            font-weight: 500;
            letter-spacing: 0.08em;
            flex-shrink: 0;
          }
          .rec-pro-status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #34d399;
            box-shadow: 0 0 6px #34d399;
            animation: rec-pro-status-pulse 1.8s ease-in-out infinite;
          }

          /* =================================================
             TRANSCRIBING BAR
             ================================================= */
          .trans-pro {
            margin-bottom: 10px;
            animation: rec-pro-in 340ms cubic-bezier(0.22, 1, 0.36, 1) both;
            width: 100%;
          }
          .trans-pro-inner {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 10px 16px;
            height: 44px;
            border-radius: 12px;
            background: linear-gradient(
              180deg,
              rgba(20, 22, 30, 0.96) 0%,
              rgba(14, 15, 20, 0.96) 100%
            );
            border: 1px solid rgba(212, 160, 23, 0.15);
            box-shadow:
              0 8px 24px -10px rgba(0, 0, 0, 0.55),
              0 0 30px -12px rgba(212, 160, 23, 0.25),
              inset 0 1px 0 rgba(255, 255, 255, 0.04);
            color: #e5e7eb;
            font-size: 11px;
            letter-spacing: 0.14em;
            font-weight: 600;
          }
          .trans-pro-label {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            color: #f4d37a;
            font-weight: 700;
            letter-spacing: 0.22em;
            flex-shrink: 0;
          }
          .trans-pro-label svg {
            animation: pw-spin 1s linear infinite;
          }
          @keyframes pw-spin {
            to { transform: rotate(360deg); }
          }
          .trans-pro-text {
            color: rgba(229, 231, 235, 0.85);
            font-weight: 500;
            letter-spacing: 0.04em;
            text-transform: none;
            font-size: 12px;
            flex-shrink: 0;
          }
          .trans-pro-track {
            flex: 1;
            height: 3px;
            border-radius: 4px;
            background: rgba(212, 160, 23, 0.1);
            overflow: hidden;
            position: relative;
          }
          .trans-pro-fill {
            position: absolute;
            inset: 0;
            border-radius: 4px;
            background: linear-gradient(
              90deg,
              rgba(244, 211, 122, 0) 0%,
              #f4d37a 30%,
              #d4a017 50%,
              #f4d37a 70%,
              rgba(244, 211, 122, 0) 100%
            );
            background-size: 200% 100%;
            animation: rec-pro-fill-march 1.6s linear infinite;
            box-shadow: 0 0 10px rgba(212, 160, 23, 0.6);
          }

          /* =================================================
             MOBILE (≤640px) — compact layout
             On narrow screens the "Listening" status label and
             its divider get truncated to "List..." (see bug
             report). Hide them and shrink gaps so the core
             REC/timer/visualizer trio fits edge-to-edge.
             ================================================= */
          @media (max-width: 640px) {
            .rec-pro-inner {
              gap: 10px;
              padding: 8px 12px;
              height: 42px;
              letter-spacing: 0.1em;
            }
            .rec-pro-live {
              letter-spacing: 0.16em;
              font-size: 10px;
            }
            .rec-pro-timer {
              font-size: 13px;
            }
            .rec-pro-visualizer {
              min-width: 0;
              flex: 1 1 auto;
            }
            /* Hide the trailing "Listening" status + its divider —
               it overflows and displays as "List..." on phones. */
            .rec-pro-status,
            .rec-pro-inner > .rec-pro-divider:last-of-type {
              display: none;
            }

            .trans-pro-inner {
              gap: 10px;
              padding: 8px 12px;
              height: 42px;
            }
            .trans-pro-label {
              letter-spacing: 0.16em;
              font-size: 10px;
            }
            .trans-pro-text {
              font-size: 11px;
            }
          }

          /* Extra-narrow phones (≤380px) — reclaim every pixel */
          @media (max-width: 380px) {
            .rec-pro-inner {
              gap: 8px;
              padding: 8px 10px;
            }
            .rec-pro-visualizer {
              height: 18px;
            }
          }
`;

const VOICE_BTN_CSS = `

              /* ============================================================
                 COMPOSER ACTION BUTTONS — unified premium system
                 The textarea has padding-right: 6rem to reserve space for
                 both the mic (idle) and send (primary) buttons, which sit
                 on the right rail at the same 36px size, 14px apart,
                 vertically centered via bottom: 6px.
                 ============================================================ */
              .voice-btn {
                position: absolute;
                right: 52px;
                top: 50%;
                transform: translateY(-50%);
                width: 36px;
                height: 36px;
                border-radius: 10px;
                flex-shrink: 0;
                border: 1px solid rgba(212, 160, 23, 0.3);
                background: rgba(212, 160, 23, 0.08);
                color: rgba(254, 243, 199, 0.7);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                padding: 0;
                line-height: 0;
                transition: background 180ms ease, color 180ms ease,
                  border-color 180ms ease, transform 200ms ease,
                  box-shadow 260ms ease;
                z-index: 10;
              }
              /* Light-mode voice button — higher contrast on pale bg */
              html:not([data-theme="dark"]) .voice-btn {
                background: #ffffff;
                border-color: rgba(184, 134, 11, 0.55);
                color: #8b6508;
                box-shadow: 0 2px 6px -2px rgba(26, 22, 18, 0.1);
              }
              .voice-btn:hover:not(:disabled):not(.voice-btn-rec) {
                background: rgba(212, 160, 23, 0.16);
                border-color: rgba(212, 160, 23, 0.55);
                color: #f4d37a;
                transform: translateY(-50%) translateY(-1px);
                box-shadow: 0 6px 14px -8px rgba(212, 160, 23, 0.55);
              }
              html:not([data-theme="dark"]) .voice-btn:hover:not(:disabled):not(.voice-btn-rec) {
                background: rgba(212, 160, 23, 0.14);
                border-color: #8b6508;
                color: #6d4e06;
                box-shadow: 0 6px 14px -6px rgba(184, 134, 11, 0.4);
              }
              .voice-btn:active:not(:disabled):not(.voice-btn-rec) {
                transform: translateY(-50%);
              }
              .voice-btn:focus-visible {
                outline: none;
                border-color: rgba(212, 160, 23, 0.6);
                box-shadow: 0 0 0 3px rgba(212, 160, 23, 0.18);
              }
              .voice-btn:disabled {
                opacity: 0.4;
                cursor: not-allowed;
              }
              .voice-btn-ic {
                position: relative;
                z-index: 2;
              }
              /* Recording state: crimson gradient, pulsing halo, ripple ring */
              .voice-btn-rec {
                color: #ffffff;
                background: linear-gradient(145deg, #f87171 0%, #ef4444 55%, #b91c1c 100%);
                border-color: rgba(239, 68, 68, 0.4);
                box-shadow:
                  0 0 0 0 rgba(239, 68, 68, 0.5),
                  0 8px 20px -8px rgba(239, 68, 68, 0.7),
                  inset 0 1px 0 rgba(255, 255, 255, 0.28);
                animation: voice-btn-pulse 1.6s ease-in-out infinite;
              }
              .voice-btn-rec::after {
                content: "";
                position: absolute;
                inset: 0;
                border-radius: inherit;
                border: 1.5px solid rgba(239, 68, 68, 0.85);
                pointer-events: none;
                animation: voice-btn-ripple 1.6s ease-out infinite;
              }
              @keyframes voice-btn-ripple {
                0% { transform: scale(1); opacity: 0.8; }
                100% { transform: scale(1.9); opacity: 0; }
              }
              @keyframes voice-btn-pulse {
                0%, 100% {
                  box-shadow:
                    0 0 0 0 rgba(239, 68, 68, 0.55),
                    0 8px 20px -8px rgba(239, 68, 68, 0.7),
                    inset 0 1px 0 rgba(255, 255, 255, 0.28);
                }
                50% {
                  box-shadow:
                    0 0 0 7px rgba(239, 68, 68, 0),
                    0 14px 28px -8px rgba(239, 68, 68, 0.85),
                    inset 0 1px 0 rgba(255, 255, 255, 0.35);
                }
              }
            
`;

const SEND_BTN_CSS = `

              .send-btn-pro {
                position: absolute;
                right: 8px;
                top: 50%;
                transform: translateY(-50%);
                width: 36px;
                height: 36px;
                border-radius: 10px;
                border: 1px solid rgba(212, 160, 23, 0.3);
                background: linear-gradient(
                  145deg,
                  rgba(244, 211, 122, 0.12) 0%,
                  rgba(212, 160, 23, 0.06) 100%
                );
                color: rgba(244, 211, 122, 0.55);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                overflow: hidden;
                padding: 0;
                line-height: 0;
                transition: all 220ms cubic-bezier(0.22, 1, 0.36, 1);
                z-index: 10;
              }
              /* Light-mode idle send button — readable on pale bg */
              html:not([data-theme="dark"]) .send-btn-pro {
                background: #ffffff;
                border-color: rgba(184, 134, 11, 0.55);
                color: #8b6508;
                box-shadow: 0 2px 6px -2px rgba(26, 22, 18, 0.1);
              }
              .send-btn-pro:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                pointer-events: none;
              }
              html:not([data-theme="dark"]) .send-btn-pro:disabled {
                opacity: 0.55;
              }
              .send-btn-pro.is-active {
                background: linear-gradient(
                  145deg,
                  #f4d37a 0%,
                  #d4a017 55%,
                  #b8860b 100%
                );
                border-color: rgba(255, 255, 255, 0.22);
                color: #1a1307;
                box-shadow:
                  0 8px 18px -8px rgba(212, 160, 23, 0.85),
                  inset 0 1px 0 rgba(255, 255, 255, 0.4);
              }
              html:not([data-theme="dark"]) .send-btn-pro.is-active {
                border-color: rgba(184, 134, 11, 0.5);
                box-shadow:
                  0 8px 18px -6px rgba(184, 134, 11, 0.5),
                  inset 0 1px 0 rgba(255, 255, 255, 0.55);
              }
              .send-btn-pro.is-active:hover {
                transform: translateY(-50%) translateY(-1px) scale(1.02);
                filter: brightness(1.08);
                box-shadow:
                  0 14px 24px -10px rgba(212, 160, 23, 0.9),
                  inset 0 1px 0 rgba(255, 255, 255, 0.5);
              }
              .send-btn-pro.is-active:active {
                transform: translateY(-50%) scale(0.98);
                filter: brightness(0.95);
              }
              .send-btn-pro:focus-visible {
                outline: none;
                box-shadow:
                  0 0 0 3px rgba(212, 160, 23, 0.28),
                  0 8px 18px -8px rgba(212, 160, 23, 0.85);
              }
              .send-btn-ic {
                position: relative;
                z-index: 2;
                transition: transform 220ms cubic-bezier(0.22, 1, 0.36, 1);
              }
              .send-btn-pro.is-active:hover .send-btn-ic {
                transform: translateX(2px);
              }
              .send-btn-glow {
                position: absolute;
                inset: -2px;
                pointer-events: none;
                border-radius: 12px;
                background: conic-gradient(
                  from 0deg at 50% 50%,
                  rgba(244, 211, 122, 0.55),
                  rgba(212, 160, 23, 0) 35%,
                  rgba(244, 211, 122, 0.5) 65%,
                  rgba(212, 160, 23, 0) 100%
                );
                filter: blur(4px);
                opacity: 0;
                transition: opacity 300ms ease;
                animation: send-btn-spin 3.6s linear infinite;
              }
              .send-btn-pro.is-active:hover .send-btn-glow {
                opacity: 0.8;
              }
              .send-btn-sheen {
                position: absolute;
                top: 0;
                left: -120%;
                width: 70%;
                height: 100%;
                background: linear-gradient(
                  100deg,
                  transparent,
                  rgba(255, 255, 255, 0.5),
                  transparent
                );
                transform: skewX(-18deg);
                pointer-events: none;
                transition: left 700ms cubic-bezier(0.22, 1, 0.36, 1);
              }
              .send-btn-pro.is-active:hover .send-btn-sheen {
                left: 130%;
              }
              @keyframes send-btn-spin {
                to { transform: rotate(360deg); }
              }
            
`;

