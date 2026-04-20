"use client";

import Image from "next/image";
import { memo, useEffect, useMemo, useState } from "react";

const PHASES = [
  { text: "Scanning knowledge base" },
  { text: "Cross-referencing documents" },
  { text: "Synthesizing insights" },
  { text: "Composing answer" },
];

export const ThinkingIndicator = memo(function ThinkingIndicator() {
  const [phase, setPhase] = useState(0);
  const [textKey, setTextKey] = useState(0);

  useEffect(() => {
    const t = setInterval(() => {
      setPhase((p) => (p + 1) % PHASES.length);
      setTextKey((k) => k + 1);
    }, 2600);
    return () => clearInterval(t);
  }, []);

  // Stable random positions for ambient particles inside the bubble.
  const particles = useMemo(
    () =>
      Array.from({ length: 10 }).map((_, i) => ({
        id: i,
        left: 6 + Math.random() * 88,
        size: 1.5 + Math.random() * 2,
        delay: Math.random() * 3,
        duration: 3 + Math.random() * 3,
      })),
    [],
  );

  const currentText = PHASES[phase].text;

  return (
    <div className="flex gap-2.5 thinking-container">
      {/* Avatar with pulsing halo + rotating ring */}
      <div className="flex-shrink-0 mt-1">
        <div className="think-avatar">
          <span className="think-avatar-ring" />
          <span className="think-avatar-halo" />
          <div className="think-avatar-inner">
            <Image
              src="/pera_logo.png"
              alt=""
              width={20}
              height={20}
              className="rounded-md think-avatar-logo"
            />
          </div>
        </div>
      </div>

      {/* Bubble with aurora + animated visualizer */}
      <div className="think-bubble">
        <span className="think-border" aria-hidden="true" />
        <span className="think-aurora" aria-hidden="true" />
        <span className="think-sheen" aria-hidden="true" />
        <div className="think-particles" aria-hidden="true">
          {particles.map((p) => (
            <span
              key={p.id}
              className="think-particle"
              style={{
                left: `${p.left}%`,
                width: p.size,
                height: p.size,
                animationDelay: `${p.delay}s`,
                animationDuration: `${p.duration}s`,
              }}
            />
          ))}
        </div>

        <div className="think-inner">
          {/* Equalizer — 7 bars with independent rhythms */}
          <div className="think-wave" aria-hidden="true">
            {Array.from({ length: 7 }).map((_, i) => (
              <span key={i} className="think-bar" style={{ "--i": i } as React.CSSProperties} />
            ))}
          </div>

          {/* Morphing phase label */}
          <div className="think-text-wrap">
            <span key={textKey} className="think-text">
              {currentText}
              <span className="think-dots">
                <span className="think-d">.</span>
                <span className="think-d">.</span>
                <span className="think-d">.</span>
              </span>
            </span>
            <span className="think-progress" aria-hidden="true">
              <span
                className="think-progress-fill"
                style={{
                  width: `${((phase + 1) / PHASES.length) * 100}%`,
                }}
              />
            </span>
          </div>
        </div>
      </div>

      <style jsx>{`
        /* ---------- Avatar ---------- */
        .think-avatar {
          position: relative;
          width: 36px;
          height: 36px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .think-avatar-ring {
          position: absolute;
          inset: -3px;
          border-radius: 14px;
          border: 1px dashed rgba(212, 160, 23, 0.55);
          animation: avatar-spin 4s linear infinite;
        }
        .think-avatar-halo {
          position: absolute;
          inset: -6px;
          border-radius: 18px;
          background: radial-gradient(
            circle,
            rgba(212, 160, 23, 0.25),
            transparent 65%
          );
          filter: blur(6px);
          animation: avatar-halo 2.4s ease-in-out infinite;
        }
        .think-avatar-inner {
          position: relative;
          width: 32px;
          height: 32px;
          border-radius: 12px;
          background: linear-gradient(
            145deg,
            rgba(212, 160, 23, 0.35),
            rgba(184, 134, 11, 0.18)
          );
          border: 1px solid rgba(212, 160, 23, 0.35);
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
          box-shadow: 0 0 18px rgba(212, 160, 23, 0.25) inset;
        }
        :global(.think-avatar-logo) {
          animation: avatar-float 3.4s ease-in-out infinite;
        }

        /* ---------- Bubble ---------- */
        .think-bubble {
          position: relative;
          display: inline-flex;
          border-radius: 18px;
          padding: 14px 18px;
          background: linear-gradient(
            140deg,
            rgba(19, 26, 46, 0.9) 0%,
            rgba(14, 20, 38, 0.9) 100%
          );
          max-width: 360px;
          overflow: hidden;
          backdrop-filter: blur(10px);
          box-shadow:
            0 20px 40px -22px rgba(0, 0, 0, 0.6),
            0 4px 14px -6px rgba(212, 160, 23, 0.22);
        }
        .think-border {
          position: absolute;
          inset: 0;
          border-radius: 18px;
          padding: 1px;
          pointer-events: none;
          background: conic-gradient(
            from 0deg at 50% 50%,
            rgba(212, 160, 23, 0.8),
            rgba(212, 160, 23, 0.1) 25%,
            rgba(212, 160, 23, 0.8) 50%,
            rgba(212, 160, 23, 0.1) 75%,
            rgba(212, 160, 23, 0.8)
          );
          -webkit-mask: linear-gradient(#000 0 0) content-box,
            linear-gradient(#000 0 0);
          -webkit-mask-composite: xor;
          mask-composite: exclude;
          animation: think-border-spin 6s linear infinite;
        }
        .think-aurora {
          position: absolute;
          inset: -40%;
          pointer-events: none;
          background:
            radial-gradient(
              circle at 20% 30%,
              rgba(212, 160, 23, 0.28),
              transparent 45%
            ),
            radial-gradient(
              circle at 80% 60%,
              rgba(88, 70, 180, 0.25),
              transparent 50%
            ),
            radial-gradient(
              circle at 50% 90%,
              rgba(184, 134, 11, 0.22),
              transparent 45%
            );
          filter: blur(22px);
          opacity: 0.7;
          animation: aurora-drift 7s ease-in-out infinite alternate;
        }
        .think-sheen {
          position: absolute;
          top: 0;
          left: -120%;
          width: 80%;
          height: 100%;
          background: linear-gradient(
            100deg,
            transparent,
            rgba(255, 255, 255, 0.08),
            transparent
          );
          transform: skewX(-16deg);
          animation: think-sheen 3.4s ease-in-out infinite;
          pointer-events: none;
        }
        .think-particles {
          position: absolute;
          inset: 0;
          pointer-events: none;
        }
        .think-particle {
          position: absolute;
          bottom: -6px;
          border-radius: 50%;
          background: rgba(244, 211, 122, 0.75);
          box-shadow: 0 0 8px rgba(244, 211, 122, 0.8);
          animation: particle-up linear infinite;
          opacity: 0;
        }
        .think-inner {
          position: relative;
          z-index: 1;
          display: flex;
          align-items: center;
          gap: 12px;
        }

        /* ---------- Visualizer ---------- */
        .think-wave {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 3px;
          height: 30px;
          padding: 0 2px;
        }
        .think-bar {
          display: inline-block;
          width: 3px;
          border-radius: 3px;
          background: linear-gradient(
            180deg,
            #f4d37a,
            #d4a017 55%,
            #b8860b
          );
          box-shadow: 0 0 8px rgba(212, 160, 23, 0.55);
          transform-origin: center;
          animation: think-wave 1.2s ease-in-out infinite;
          animation-delay: calc(var(--i) * 90ms);
        }

        /* ---------- Text ---------- */
        .think-text-wrap {
          display: inline-flex;
          flex-direction: column;
          gap: 6px;
          min-width: 180px;
        }
        .think-text {
          display: inline-flex;
          align-items: baseline;
          gap: 1px;
          font-size: 0.82rem;
          font-weight: 500;
          line-height: 1.2;
          color: #fefce8;
          letter-spacing: 0.01em;
          animation: text-in 420ms cubic-bezier(0.22, 1, 0.36, 1);
        }
        .think-dots {
          display: inline-flex;
          gap: 2px;
          margin-left: 2px;
        }
        .think-d {
          display: inline-block;
          animation: dot-bob 1.4s ease-in-out infinite;
          color: #f4d37a;
        }
        .think-d:nth-child(2) {
          animation-delay: 0.2s;
        }
        .think-d:nth-child(3) {
          animation-delay: 0.4s;
        }

        .think-progress {
          position: relative;
          display: block;
          width: 100%;
          height: 2px;
          border-radius: 2px;
          background: rgba(212, 160, 23, 0.1);
          overflow: hidden;
        }
        .think-progress-fill {
          display: block;
          height: 100%;
          border-radius: 2px;
          background: linear-gradient(90deg, #f4d37a, #d4a017, #f4d37a);
          background-size: 200% 100%;
          animation: fill-shimmer 1.8s linear infinite;
          transition: width 700ms cubic-bezier(0.22, 1, 0.36, 1);
          box-shadow: 0 0 8px rgba(212, 160, 23, 0.6);
        }

        /* ---------- Keyframes ---------- */
        @keyframes avatar-spin {
          to {
            transform: rotate(360deg);
          }
        }
        @keyframes avatar-halo {
          0%, 100% {
            opacity: 0.55;
            transform: scale(1);
          }
          50% {
            opacity: 0.95;
            transform: scale(1.12);
          }
        }
        @keyframes avatar-float {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-2px);
          }
        }
        @keyframes think-border-spin {
          to {
            transform: rotate(360deg);
          }
        }
        @keyframes aurora-drift {
          0% {
            transform: translate(-4%, -2%) rotate(0deg);
          }
          100% {
            transform: translate(4%, 3%) rotate(12deg);
          }
        }
        @keyframes think-sheen {
          0% {
            left: -120%;
          }
          55%, 100% {
            left: 140%;
          }
        }
        @keyframes particle-up {
          0% {
            opacity: 0;
            transform: translateY(0) scale(0.7);
          }
          15%, 80% {
            opacity: 1;
          }
          100% {
            opacity: 0;
            transform: translateY(-42px) scale(1.1);
          }
        }
        @keyframes think-wave {
          0%, 100% {
            transform: scaleY(0.35);
            opacity: 0.55;
          }
          50% {
            transform: scaleY(1);
            opacity: 1;
          }
        }
        .think-bar:nth-child(1) {
          height: 10px;
        }
        .think-bar:nth-child(2) {
          height: 18px;
        }
        .think-bar:nth-child(3) {
          height: 26px;
        }
        .think-bar:nth-child(4) {
          height: 20px;
        }
        .think-bar:nth-child(5) {
          height: 26px;
        }
        .think-bar:nth-child(6) {
          height: 18px;
        }
        .think-bar:nth-child(7) {
          height: 10px;
        }

        @keyframes text-in {
          from {
            opacity: 0;
            transform: translateY(6px);
            filter: blur(3px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
            filter: blur(0);
          }
        }
        @keyframes dot-bob {
          0%, 70%, 100% {
            transform: translateY(0);
            opacity: 0.55;
          }
          35% {
            transform: translateY(-3px);
            opacity: 1;
          }
        }
        @keyframes fill-shimmer {
          0% {
            background-position: 200% 50%;
          }
          100% {
            background-position: 0% 50%;
          }
        }
      `}</style>
    </div>
  );
});
