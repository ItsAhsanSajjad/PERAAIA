"use client";

import Image from "next/image";
import { memo, useEffect, useMemo, useRef, useState } from "react";

/* ═══════════════════════════════════════════════════════════
   PERA AI — Premium Welcome Experience
   Animated hero + interactive prompt grid + glass capability cards
   ═══════════════════════════════════════════════════════════ */

const EXAMPLE_QUERIES = [
  {
    q: "What are PERA enforcement powers?",
    hint: "Regulation & Act",
    glyph: "⚖",
  },
  {
    q: "What KPIs measure PERA performance?",
    hint: "Performance",
    glyph: "📊",
  },
  {
    q: "How is an EPO issued?",
    hint: "Enforcement",
    glyph: "⚡",
  },
  {
    q: "What is the structure of the PERA Board?",
    hint: "Governance",
    glyph: "🏛",
  },
];

const EXPLORE_CARDS = [
  {
    theme: "Governance",
    glyph: "⚖",
    prompts: [
      "What powers does PERA have under the Act?",
      "What is the structure of the PERA Board?",
    ],
  },
  {
    theme: "Roles",
    glyph: "👤",
    prompts: [
      "Responsibilities of the Chief Technology Officer?",
      "Duties of enforcement officers?",
    ],
  },
  {
    theme: "Enforcement",
    glyph: "⚡",
    prompts: [
      "How is an EPO issued?",
      "What enforcement powers does PERA have?",
    ],
  },
  {
    theme: "Performance",
    glyph: "📊",
    prompts: [
      "What KPIs measure operational success?",
      "How is efficiency measured?",
    ],
  },
  {
    theme: "Compliance",
    glyph: "🔒",
    prompts: [
      "Professional conduct standards for PERA employees?",
      "Confidentiality obligations for staff?",
    ],
  },
  {
    theme: "Learning",
    glyph: "📚",
    prompts: [
      "Training expectations for PERA staff?",
      "How is capacity building measured?",
    ],
  },
];

const CAPABILITIES = [
  { label: "Regulations & Powers", glyph: "§" },
  { label: "Governance & Roles", glyph: "⚖" },
  { label: "Enforcement Procedures", glyph: "⚡" },
  { label: "Performance Frameworks", glyph: "◆" },
  { label: "Policies & Standards", glyph: "★" },
];

interface Props {
  onSendMessage: (text: string) => void;
}

export const WelcomeScreen = memo(function WelcomeScreen({
  onSendMessage,
}: Props) {
  const rootRef = useRef<HTMLDivElement | null>(null);

  // Global cursor tracking — drives cursor-reactive layers across the
  // entire viewport, not just inside the welcome column.
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    function onMove(e: MouseEvent) {
      if (!el) return;
      const xp = (e.clientX / window.innerWidth) * 100;
      const yp = (e.clientY / window.innerHeight) * 100;
      el.style.setProperty("--mx", `${xp}%`);
      el.style.setProperty("--my", `${yp}%`);
      el.style.setProperty("--cx", `${e.clientX}px`);
      el.style.setProperty("--cy", `${e.clientY}px`);
    }
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  // Ambient particles — generated only after hydration to avoid any
  // SSR vs client random-value mismatch.
  const [particles, setParticles] = useState<
    Array<{
      id: number;
      left: number;
      size: number;
      delay: number;
      duration: number;
      hue: "mauve" | "gold";
    }>
  >([]);
  useEffect(() => {
    setParticles(
      Array.from({ length: 42 }).map((_, i) => ({
        id: i,
        left: Math.random() * 100,
        size: 1.4 + Math.random() * 3.5,
        delay: Math.random() * 14,
        duration: 14 + Math.random() * 16,
        hue: Math.random() > 0.75 ? "mauve" : "gold",
      })),
    );
  }, []);

  const orbitDots = useMemo(
    () =>
      [0, 1, 2, 3, 4, 5].map((i) => ({
        id: i,
        delay: -i * 2.4,
      })),
    [],
  );

  return (
    <div
      ref={rootRef}
      className="pw-root"
      style={
        {
          "--mx": "50%",
          "--my": "30%",
          "--cx": "50vw",
          "--cy": "40vh",
        } as React.CSSProperties
      }
    >
      {/* ═══ Full-viewport ambient field ═══ */}
      <div className="pw-stage" aria-hidden="true">
        <div className="pw-stage-gradient" />
        <div className="pw-stage-grid" />
        <div className="pw-stage-orb pw-stage-orb-1" />
        <div className="pw-stage-orb pw-stage-orb-2" />
        <div className="pw-stage-orb pw-stage-orb-3" />
        <div className="pw-stage-aurora" />
        <div className="pw-stage-beams">
          <span className="pw-beam" />
          <span className="pw-beam" />
          <span className="pw-beam" />
          <span className="pw-beam" />
          <span className="pw-beam" />
        </div>
        <div className="pw-stage-particles">
          {particles.map((p) => (
            <span
              key={p.id}
              className={`pw-particle pw-particle-${p.hue}`}
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
        <div className="pw-cursor-follow" />
        <div className="pw-stage-vignette" />
        <div className="pw-stage-noise" />
      </div>

      {/* ═══ Content column ═══ */}
      <div className="pw-shell">
        {/* ══ Hero ══ */}
        <section className="pw-hero">
          <div className="pw-hero-crown">
            <span className="pw-crown-line" />
            <span className="pw-crown-mark">GOVERNMENT OF PUNJAB</span>
            <span className="pw-crown-line" />
          </div>

          {/* Emblem with orbiting dots */}
          <div className="pw-emblem">
            <span className="pw-emblem-halo" />
            <span className="pw-emblem-ring pw-emblem-ring-1" />
            <span className="pw-emblem-ring pw-emblem-ring-2" />
            <span className="pw-emblem-ring pw-emblem-ring-3" />
            <div className="pw-emblem-orbit">
              {orbitDots.map((d) => (
                <span
                  key={d.id}
                  className="pw-emblem-orbit-dot"
                  style={{ animationDelay: `${d.delay}s` }}
                />
              ))}
            </div>
            <div className="pw-emblem-inner">
              <Image
                src="/pera_logo.png"
                alt="PERA"
                width={88}
                height={88}
                priority
                className="pw-emblem-img"
              />
            </div>
          </div>

          <p className="pw-hero-kicker">Punjab Enforcement &amp; Regulatory Authority</p>

          <h1 className="pw-hero-title" aria-label="Ask PERA Assistant">
            <span className="pw-title-word">
              {"Ask".split("").map((c, i) => (
                <span
                  key={i}
                  className="pw-char"
                  style={{ animationDelay: `${500 + i * 70}ms` }}
                >
                  {c}
                </span>
              ))}
            </span>
            <span className="pw-title-space">&nbsp;</span>
            <span className="pw-title-word pw-title-brand">
              {"PERA".split("").map((c, i) => (
                <span
                  key={i}
                  className="pw-char"
                  style={{ animationDelay: `${780 + i * 70}ms` }}
                >
                  {c}
                </span>
              ))}
            </span>
            <span className="pw-title-space">&nbsp;</span>
            <span className="pw-title-word">
              {"Assistant".split("").map((c, i) => (
                <span
                  key={i}
                  className="pw-char"
                  style={{ animationDelay: `${1060 + i * 50}ms` }}
                >
                  {c}
                </span>
              ))}
            </span>
          </h1>

          <p className="pw-hero-sub pw-fade" style={{ "--d": "1700ms" } as React.CSSProperties}>
            Official intelligence layer for <span className="pw-hero-em">regulations</span>,
            <span className="pw-hero-em"> governance</span>, <span className="pw-hero-em">enforcement</span>, and
            <span className="pw-hero-em"> Authority performance</span>.
          </p>

          <div className="pw-hero-meta pw-fade" style={{ "--d": "1860ms" } as React.CSSProperties}>
            <span className="pw-meta-chip">
              <span className="pw-live-dot" />
              Official Government System
            </span>
            <span className="pw-meta-chip">
              <span className="pw-meta-bullet">v</span>
              Version 2.0
            </span>
            <span className="pw-meta-chip">
              <span className="pw-meta-bullet">◆</span>
              Knowledge Base · Current
            </span>
          </div>
        </section>

        {/* ══ Notice ribbon ══ */}
        <section
          className="pw-notice pw-fade"
          style={{ "--d": "2040ms" } as React.CSSProperties}
        >
          <span className="pw-notice-bar" />
          <span className="pw-notice-glow" />
          <p>
            Informational guidance derived from official PERA regulations,
            policies, governance documents, and performance frameworks —
            responses are grounded in verified institutional sources.
          </p>
        </section>

        {/* ══ Quick prompts ══ */}
        <section
          className="pw-quick pw-fade"
          style={{ "--d": "2180ms" } as React.CSSProperties}
        >
          <div className="pw-eyebrow">
            <span className="pw-eyebrow-line" />
            <span className="pw-eyebrow-label">Popular questions</span>
            <span className="pw-eyebrow-line" />
          </div>
          <div className="pw-quick-grid">
            {EXAMPLE_QUERIES.map((q, i) => (
              <button
                key={q.q}
                onClick={() => onSendMessage(q.q)}
                className="pw-quick-card"
                style={{ animationDelay: `${2280 + i * 80}ms` }}
              >
                <span className="pw-quick-bg" />
                <span className="pw-quick-glyph">{q.glyph}</span>
                <span className="pw-quick-body">
                  <span className="pw-quick-hint">{q.hint}</span>
                  <span className="pw-quick-q">{q.q}</span>
                </span>
                <span className="pw-quick-arrow">
                  <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
                    <path
                      d="M5 12h14M13 5l7 7-7 7"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </span>
              </button>
            ))}
          </div>
        </section>

        {/* ══ Explore grid ══ */}
        <section
          className="pw-explore pw-fade"
          style={{ "--d": "2600ms" } as React.CSSProperties}
        >
          <div className="pw-eyebrow">
            <span className="pw-eyebrow-line" />
            <span className="pw-eyebrow-label">Explore topics</span>
            <span className="pw-eyebrow-line" />
          </div>
          <div className="pw-explore-grid">
            {EXPLORE_CARDS.map((c, i) => (
              <div
                key={c.theme}
                className="pw-explore-card"
                style={{ animationDelay: `${2700 + i * 90}ms` }}
              >
                <span className="pw-card-sheen" />
                <span className="pw-card-border" />
                <div className="pw-explore-head">
                  <span className="pw-explore-glyph">{c.glyph}</span>
                  <span className="pw-explore-theme">{c.theme}</span>
                </div>
                <div className="pw-explore-list">
                  {c.prompts.map((p) => (
                    <button
                      key={p}
                      onClick={() => onSendMessage(p)}
                      className="pw-explore-prompt"
                    >
                      <span className="pw-prompt-arrow">›</span>
                      <span className="pw-prompt-text">{p}</span>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ══ Capabilities strip ══ */}
        <section
          className="pw-caps pw-fade"
          style={{ "--d": "3400ms" } as React.CSSProperties}
        >
          <div className="pw-eyebrow">
            <span className="pw-eyebrow-line" />
            <span className="pw-eyebrow-label">Capabilities</span>
            <span className="pw-eyebrow-line" />
          </div>
          <div className="pw-caps-row">
            {CAPABILITIES.map((c, i) => (
              <div
                key={c.label}
                className="pw-cap-pill"
                style={{ animationDelay: `${3480 + i * 70}ms` }}
              >
                <span className="pw-cap-glyph">{c.glyph}</span>
                <span className="pw-cap-label">{c.label}</span>
              </div>
            ))}
          </div>
        </section>

        <div className="pw-hint pw-fade" style={{ "--d": "3900ms" } as React.CSSProperties}>
          Ask anything below to begin.
          <span className="pw-hint-arrow">↓</span>
        </div>
      </div>

      <style>{`
        /* ================================================================
           ROOT
        ================================================================ */
        .pw-root {
          position: relative;
          min-height: 100%;
          /* The chat window wraps us in a max-w-3xl container; break out
             so the shell has room for its own max-width and the ambient
             layers fill the full viewport. */
          margin-left: calc(50% - 50vw);
          margin-right: calc(50% - 50vw);
          width: 100vw;
        }

        /* ================================================================
           STAGE — full-viewport ambient field
        ================================================================ */
        .pw-stage {
          position: fixed;
          inset: 0;
          z-index: 0;
          pointer-events: none;
          overflow: hidden;
        }
        .pw-stage-gradient {
          position: absolute;
          inset: 0;
          background: radial-gradient(
              1400px 800px at 50% 0%,
              rgba(212, 160, 23, 0.18),
              transparent 55%
            ),
            radial-gradient(
              1100px 700px at 100% 100%,
              rgba(88, 70, 180, 0.14),
              transparent 55%
            ),
            radial-gradient(
              900px 500px at 0% 80%,
              rgba(184, 134, 11, 0.12),
              transparent 55%
            );
        }
        .pw-stage-grid {
          position: absolute;
          inset: 0;
          background-image: linear-gradient(
              rgba(212, 160, 23, 0.055) 1px,
              transparent 1px
            ),
            linear-gradient(
              90deg,
              rgba(212, 160, 23, 0.055) 1px,
              transparent 1px
            );
          background-size: 72px 72px;
          mask-image: radial-gradient(
            ellipse at 50% 40%,
            black 25%,
            transparent 85%
          );
          opacity: 0.55;
          animation: stage-grid 34s linear infinite;
        }
        .pw-stage-orb {
          position: absolute;
          border-radius: 50%;
          filter: blur(140px);
        }
        .pw-stage-orb-1 {
          width: 720px;
          height: 720px;
          top: -260px;
          left: -220px;
          background: radial-gradient(
            circle,
            rgba(212, 160, 23, 0.65),
            transparent 70%
          );
          opacity: 0.7;
          animation: orb-a 24s ease-in-out infinite;
        }
        .pw-stage-orb-2 {
          width: 680px;
          height: 680px;
          bottom: -260px;
          right: -180px;
          background: radial-gradient(
            circle,
            rgba(184, 134, 11, 0.6),
            transparent 70%
          );
          opacity: 0.55;
          animation: orb-b 28s ease-in-out infinite;
        }
        .pw-stage-orb-3 {
          width: 520px;
          height: 520px;
          top: 40%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: radial-gradient(
            circle,
            rgba(88, 70, 180, 0.55),
            transparent 70%
          );
          opacity: 0.32;
          animation: orb-c 34s ease-in-out infinite;
        }
        .pw-stage-aurora {
          position: absolute;
          inset: 0;
          background: radial-gradient(
            900px circle at var(--mx, 50%) var(--my, 30%),
            rgba(244, 211, 122, 0.16),
            transparent 55%
          );
          transition: background 260ms ease;
        }
        .pw-stage-beams {
          position: absolute;
          inset: 0;
          opacity: 0.55;
        }
        .pw-beam {
          position: absolute;
          top: -12%;
          width: 2px;
          height: 124%;
          background: linear-gradient(
            180deg,
            transparent,
            rgba(244, 211, 122, 0.42),
            rgba(244, 211, 122, 0.08),
            transparent
          );
          transform-origin: top center;
          filter: blur(1.2px);
          animation: beam-sweep 16s linear infinite;
        }
        .pw-beam:nth-child(1) {
          left: 12%;
          transform: rotate(-9deg);
          animation-delay: -1s;
        }
        .pw-beam:nth-child(2) {
          left: 32%;
          transform: rotate(6deg);
          animation-delay: -5s;
          animation-duration: 19s;
        }
        .pw-beam:nth-child(3) {
          left: 54%;
          transform: rotate(-3deg);
          animation-delay: -9s;
          animation-duration: 14s;
        }
        .pw-beam:nth-child(4) {
          left: 74%;
          transform: rotate(8deg);
          animation-delay: -13s;
          animation-duration: 21s;
        }
        .pw-beam:nth-child(5) {
          left: 90%;
          transform: rotate(-5deg);
          animation-delay: -17s;
          animation-duration: 17s;
        }
        .pw-stage-particles {
          position: absolute;
          inset: 0;
        }
        .pw-particle {
          position: absolute;
          bottom: -30px;
          border-radius: 50%;
          opacity: 0;
          animation: particle-rise linear infinite;
        }
        .pw-particle-gold {
          background: rgba(244, 211, 122, 0.85);
          box-shadow: 0 0 14px rgba(244, 211, 122, 0.9);
        }
        .pw-particle-mauve {
          background: rgba(167, 139, 250, 0.75);
          box-shadow: 0 0 14px rgba(167, 139, 250, 0.8);
        }
        .pw-cursor-follow {
          position: absolute;
          top: 0;
          left: 0;
          width: 520px;
          height: 520px;
          pointer-events: none;
          background: radial-gradient(
            circle,
            rgba(212, 160, 23, 0.16),
            transparent 55%
          );
          transform: translate(
            calc(var(--cx, 50vw) - 260px),
            calc(var(--cy, 40vh) - 260px)
          );
          transition: transform 120ms ease-out;
          filter: blur(12px);
        }
        .pw-stage-vignette {
          position: absolute;
          inset: 0;
          background: radial-gradient(
            ellipse at center,
            transparent 42%,
            rgba(0, 0, 0, 0.55) 100%
          );
        }
        .pw-stage-noise {
          position: absolute;
          inset: 0;
          opacity: 0.05;
          mix-blend-mode: overlay;
          background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)' opacity='0.6'/></svg>");
        }

        /* ================================================================
           SHELL
        ================================================================ */
        .pw-shell {
          position: relative;
          z-index: 1;
          max-width: 1020px;
          margin: 0 auto;
          padding: 3rem 1.5rem 5rem;
          display: flex;
          flex-direction: column;
          gap: 2.4rem;
        }

        /* ================================================================
           HERO
        ================================================================ */
        .pw-hero {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          gap: 1.6rem;
          padding: 0.8rem 0 0.6rem;
        }
        .pw-hero-crown {
          display: inline-flex;
          align-items: center;
          gap: 14px;
          animation: fade-up 620ms cubic-bezier(0.22, 1, 0.36, 1) both;
          animation-delay: 140ms;
        }
        .pw-crown-line {
          display: inline-block;
          width: 50px;
          height: 1px;
          background: linear-gradient(
            90deg,
            transparent,
            rgba(212, 160, 23, 0.6),
            transparent
          );
        }
        .pw-crown-mark {
          font-size: 10px;
          letter-spacing: 0.38em;
          font-weight: 700;
          color: rgba(254, 243, 199, 0.62);
        }

        /* Emblem */
        .pw-emblem {
          position: relative;
          width: 140px;
          height: 140px;
          margin: 0.4rem 0 0.8rem;
          display: flex;
          align-items: center;
          justify-content: center;
          animation: emblem-in 900ms cubic-bezier(0.22, 1, 0.36, 1) both;
          animation-delay: 260ms;
        }
        .pw-emblem-halo {
          position: absolute;
          inset: -22px;
          border-radius: 50%;
          background: radial-gradient(
            circle,
            rgba(212, 160, 23, 0.35),
            transparent 62%
          );
          filter: blur(20px);
          animation: halo-breath 3.6s ease-in-out infinite;
        }
        .pw-emblem-ring {
          position: absolute;
          border-radius: 50%;
          pointer-events: none;
        }
        .pw-emblem-ring-1 {
          inset: -6px;
          border: 1px dashed rgba(212, 160, 23, 0.55);
          animation: ring-spin 14s linear infinite;
        }
        .pw-emblem-ring-2 {
          inset: -22px;
          border: 1px solid rgba(212, 160, 23, 0.18);
          animation: ring-spin 30s linear infinite reverse;
        }
        .pw-emblem-ring-3 {
          inset: -42px;
          border: 1px dotted rgba(212, 160, 23, 0.12);
          animation: ring-spin 54s linear infinite;
        }
        .pw-emblem-orbit {
          position: absolute;
          inset: -22px;
          animation: ring-spin 10s linear infinite;
        }
        .pw-emblem-orbit-dot {
          position: absolute;
          top: 50%;
          left: 0;
          width: 4px;
          height: 4px;
          border-radius: 50%;
          background: #f4d37a;
          box-shadow: 0 0 8px #f4d37a;
          transform: translateY(-50%);
          animation: orbit-flicker 2.4s ease-in-out infinite;
        }
        .pw-emblem-orbit-dot:nth-child(2) {
          top: 0;
          left: 50%;
        }
        .pw-emblem-orbit-dot:nth-child(3) {
          top: 50%;
          left: 100%;
        }
        .pw-emblem-orbit-dot:nth-child(4) {
          top: 100%;
          left: 50%;
        }
        .pw-emblem-orbit-dot:nth-child(5) {
          top: 15%;
          left: 85%;
        }
        .pw-emblem-orbit-dot:nth-child(6) {
          top: 85%;
          left: 15%;
        }
        .pw-emblem-inner {
          position: relative;
          width: 108px;
          height: 108px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          background: radial-gradient(
            circle at 30% 25%,
            rgba(244, 211, 122, 0.22),
            rgba(11, 16, 32, 0.92) 70%
          );
          border: 1px solid rgba(212, 160, 23, 0.38);
          box-shadow:
            0 0 40px rgba(212, 160, 23, 0.35),
            inset 0 0 18px rgba(212, 160, 23, 0.18);
          overflow: hidden;
        }
        .pw-emblem-img {
          animation: logo-float 5s ease-in-out infinite;
          filter: drop-shadow(0 0 12px rgba(212, 160, 23, 0.4));
        }

        .pw-hero-kicker {
          font-size: 0.95rem;
          font-weight: 500;
          color: rgba(254, 243, 199, 0.78);
          letter-spacing: 0.01em;
          margin: 0;
          animation: fade-up 620ms cubic-bezier(0.22, 1, 0.36, 1) both;
          animation-delay: 360ms;
        }

        .pw-hero-title {
          position: relative;
          font-size: clamp(2.2rem, 4vw, 3.4rem);
          line-height: 1.1;
          font-weight: 600;
          letter-spacing: -0.015em;
          margin: 1rem 0 1.6rem;
          display: inline-flex;
          flex-wrap: wrap;
          justify-content: center;
          align-items: baseline;
          color: #f5ecd0;
          -webkit-text-fill-color: #f5ecd0;
        }
        /* Professional wordmark treatment: single weight, single color,
           the brand word "PERA" is the only accent — rendered in a
           slightly heavier weight with tracked uppercase so it reads
           like an institutional wordmark rather than a marketing
           splash. */
        .pw-title-brand .pw-char {
          font-weight: 700;
          letter-spacing: 0.04em;
          color: #e9c464;
          -webkit-text-fill-color: #e9c464;
        }
        .pw-hero-title::after {
          content: "";
          position: absolute;
          left: 50%;
          bottom: -12px;
          width: 52px;
          height: 1px;
          transform: translateX(-50%);
          background: linear-gradient(
            90deg,
            transparent,
            rgba(212, 160, 23, 0.55),
            transparent
          );
        }
        .pw-title-word {
          display: inline-flex;
        }
        .pw-title-space {
          display: inline-block;
        }
        .pw-char {
          display: inline-block;
          /* Characters are visible by default so the title never
             ends up permanently invisible if the entrance animation
             is missed (e.g., page restored from bfcache, reduced-
             motion, or any race with hydration). The animation
             plays from the keyframe's "from" state which overrides
             these initial values. */
          opacity: 1;
          transform: none;
          filter: none;
          animation: char-in 680ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }
        @keyframes char-in-fallback {
          from {
            opacity: 0;
            transform: translateY(28px) scale(0.9);
            filter: blur(8px);
          }
          to {
            opacity: 1;
            transform: none;
            filter: none;
          }
        }

        .pw-hero-sub {
          max-width: 680px;
          font-size: 1.02rem;
          line-height: 1.55;
          color: rgba(254, 243, 199, 0.78);
          margin: 0.6rem 0 0;
        }
        .pw-hero-em {
          color: #f4d37a;
          font-weight: 600;
        }

        .pw-hero-meta {
          display: inline-flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 0.8rem;
          justify-content: center;
        }
        .pw-meta-chip {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 6px 12px;
          border-radius: 999px;
          font-size: 0.72rem;
          font-weight: 600;
          letter-spacing: 0.05em;
          color: rgba(254, 243, 199, 0.78);
          background: rgba(19, 26, 46, 0.68);
          border: 1px solid rgba(212, 160, 23, 0.22);
          backdrop-filter: blur(6px);
          transition: border-color 220ms ease, transform 220ms ease,
            background 220ms ease;
        }
        .pw-meta-chip:hover {
          transform: translateY(-2px);
          border-color: rgba(212, 160, 23, 0.55);
          background: rgba(19, 26, 46, 0.82);
        }
        .pw-meta-bullet {
          font-size: 0.7rem;
          color: #f4d37a;
          font-weight: 700;
        }
        .pw-live-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #34d399;
          box-shadow: 0 0 8px #34d399;
          animation: live-pulse 1.8s ease-in-out infinite;
        }

        /* ================================================================
           NOTICE
        ================================================================ */
        .pw-notice {
          position: relative;
          overflow: hidden;
          padding: 1.1rem 1.3rem 1.1rem 1.6rem;
          border-radius: 16px;
          background: linear-gradient(
            165deg,
            rgba(19, 26, 46, 0.72),
            rgba(14, 20, 38, 0.72)
          );
          border: 1px solid rgba(212, 160, 23, 0.16);
          backdrop-filter: blur(10px);
          font-size: 0.88rem;
          color: rgba(254, 243, 199, 0.78);
          line-height: 1.6;
        }
        .pw-notice-bar {
          position: absolute;
          top: 0;
          bottom: 0;
          left: 0;
          width: 3px;
          background: linear-gradient(180deg, #f4d37a, #d4a017, #b8860b);
          box-shadow: 0 0 14px rgba(212, 160, 23, 0.6);
          animation: bar-pulse 2.6s ease-in-out infinite;
        }
        .pw-notice-glow {
          position: absolute;
          inset: 0;
          pointer-events: none;
          background: radial-gradient(
            320px circle at var(--mx, 50%) var(--my, 50%),
            rgba(212, 160, 23, 0.14),
            transparent 55%
          );
          transition: background 300ms ease;
        }

        /* ================================================================
           EYEBROW
        ================================================================ */
        .pw-eyebrow {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 14px;
          margin-bottom: 1.1rem;
        }
        .pw-eyebrow-line {
          display: inline-block;
          flex: 1;
          max-width: 120px;
          height: 1px;
          background: linear-gradient(
            90deg,
            transparent,
            rgba(212, 160, 23, 0.5),
            transparent
          );
        }
        .pw-eyebrow-label {
          font-size: 10px;
          letter-spacing: 0.38em;
          font-weight: 700;
          color: rgba(254, 243, 199, 0.6);
          text-transform: uppercase;
        }

        /* ================================================================
           QUICK PROMPTS
        ================================================================ */
        .pw-quick-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
          gap: 0.9rem;
        }
        .pw-quick-card {
          position: relative;
          display: grid;
          grid-template-columns: auto 1fr auto;
          align-items: center;
          gap: 14px;
          padding: 14px 16px;
          border-radius: 14px;
          background: linear-gradient(
            165deg,
            rgba(19, 26, 46, 0.82),
            rgba(14, 20, 38, 0.82)
          );
          border: 1px solid rgba(212, 160, 23, 0.18);
          color: rgba(254, 243, 199, 0.95);
          text-align: left;
          overflow: hidden;
          cursor: pointer;
          backdrop-filter: blur(8px);
          animation: card-in 560ms cubic-bezier(0.22, 1, 0.36, 1) both;
          transition: transform 240ms cubic-bezier(0.22, 1, 0.36, 1),
            border-color 240ms ease, box-shadow 280ms ease;
        }
        .pw-quick-card:hover {
          transform: translateY(-4px);
          border-color: rgba(212, 160, 23, 0.45);
          box-shadow: 0 22px 42px -22px rgba(212, 160, 23, 0.55);
        }
        .pw-quick-bg {
          position: absolute;
          inset: 0;
          background: radial-gradient(
            260px circle at var(--mx, 50%) var(--my, 50%),
            rgba(212, 160, 23, 0.15),
            transparent 55%
          );
          opacity: 0;
          transition: opacity 240ms ease;
          pointer-events: none;
        }
        .pw-quick-card:hover .pw-quick-bg {
          opacity: 1;
        }
        .pw-quick-glyph {
          font-size: 20px;
          width: 40px;
          height: 40px;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: linear-gradient(
            145deg,
            rgba(212, 160, 23, 0.2),
            rgba(212, 160, 23, 0.06)
          );
          border: 1px solid rgba(212, 160, 23, 0.3);
          color: #f4d37a;
          transition: transform 260ms cubic-bezier(0.22, 1, 0.36, 1);
        }
        .pw-quick-card:hover .pw-quick-glyph {
          transform: rotate(-8deg) scale(1.08);
        }
        .pw-quick-body {
          display: flex;
          flex-direction: column;
          gap: 2px;
          min-width: 0;
        }
        .pw-quick-hint {
          font-size: 0.66rem;
          letter-spacing: 0.16em;
          font-weight: 600;
          text-transform: uppercase;
          color: rgba(254, 243, 199, 0.55);
        }
        .pw-quick-q {
          font-size: 0.9rem;
          font-weight: 500;
          line-height: 1.35;
          color: rgba(254, 243, 199, 0.95);
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .pw-quick-arrow {
          color: rgba(254, 243, 199, 0.4);
          transition: color 220ms ease, transform 220ms ease;
          display: inline-flex;
        }
        .pw-quick-card:hover .pw-quick-arrow {
          color: #f4d37a;
          transform: translateX(4px);
        }

        /* ================================================================
           EXPLORE
        ================================================================ */
        .pw-explore-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
          gap: 0.9rem;
        }
        .pw-explore-card {
          position: relative;
          overflow: hidden;
          padding: 1.1rem 1.2rem;
          border-radius: 16px;
          background: linear-gradient(
            165deg,
            rgba(19, 26, 46, 0.78),
            rgba(14, 20, 38, 0.78)
          );
          backdrop-filter: blur(10px);
          animation: card-in 600ms cubic-bezier(0.22, 1, 0.36, 1) both;
          transition: transform 260ms ease, box-shadow 300ms ease;
        }
        .pw-explore-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 28px 52px -28px rgba(212, 160, 23, 0.5);
        }
        .pw-card-sheen {
          position: absolute;
          inset: 0;
          pointer-events: none;
          background: radial-gradient(
            380px circle at var(--mx, 50%) var(--my, 50%),
            rgba(255, 255, 255, 0.07),
            transparent 45%
          );
          opacity: 0;
          transition: opacity 260ms ease;
        }
        .pw-explore-card:hover .pw-card-sheen {
          opacity: 1;
        }
        .pw-card-border {
          position: absolute;
          inset: 0;
          border-radius: 16px;
          padding: 1px;
          background: linear-gradient(
            140deg,
            rgba(212, 160, 23, 0.45) 0%,
            rgba(212, 160, 23, 0.08) 45%,
            rgba(212, 160, 23, 0.4) 100%
          );
          -webkit-mask: linear-gradient(#000 0 0) content-box,
            linear-gradient(#000 0 0);
          -webkit-mask-composite: xor;
          mask-composite: exclude;
          pointer-events: none;
        }
        .pw-explore-head {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 0.85rem;
        }
        .pw-explore-glyph {
          width: 34px;
          height: 34px;
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: linear-gradient(
            145deg,
            rgba(212, 160, 23, 0.22),
            rgba(212, 160, 23, 0.05)
          );
          border: 1px solid rgba(212, 160, 23, 0.3);
          color: #f4d37a;
          font-size: 16px;
          transition: transform 260ms ease;
        }
        .pw-explore-card:hover .pw-explore-glyph {
          transform: rotate(-6deg) scale(1.08);
        }
        .pw-explore-theme {
          font-size: 1rem;
          font-weight: 700;
          color: #fefce8;
        }
        .pw-explore-list {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        .pw-explore-prompt {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 9px 10px;
          background: transparent;
          border: 0;
          color: rgba(254, 243, 199, 0.78);
          font-size: 0.83rem;
          line-height: 1.4;
          text-align: left;
          cursor: pointer;
          border-radius: 8px;
          transition: background 200ms ease, color 200ms ease,
            transform 200ms ease;
        }
        .pw-explore-prompt:hover {
          background: rgba(212, 160, 23, 0.08);
          color: #fefce8;
          transform: translateX(3px);
        }
        .pw-prompt-arrow {
          color: rgba(212, 160, 23, 0.7);
          font-size: 18px;
          line-height: 1;
          transition: color 200ms ease, transform 200ms ease;
        }
        .pw-explore-prompt:hover .pw-prompt-arrow {
          color: #f4d37a;
          transform: translateX(2px);
        }

        /* ================================================================
           CAPABILITIES
        ================================================================ */
        .pw-caps-row {
          display: flex;
          flex-wrap: wrap;
          justify-content: center;
          gap: 10px;
        }
        .pw-cap-pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 8px 14px;
          border-radius: 999px;
          background: rgba(19, 26, 46, 0.7);
          border: 1px solid rgba(212, 160, 23, 0.22);
          font-size: 0.82rem;
          color: rgba(254, 243, 199, 0.85);
          backdrop-filter: blur(6px);
          animation: fade-up 540ms cubic-bezier(0.22, 1, 0.36, 1) both;
          transition: border-color 220ms ease, transform 220ms ease,
            color 220ms ease;
        }
        .pw-cap-pill:hover {
          border-color: rgba(212, 160, 23, 0.55);
          transform: translateY(-2px);
          color: #fefce8;
        }
        .pw-cap-glyph {
          display: inline-flex;
          width: 20px;
          height: 20px;
          align-items: center;
          justify-content: center;
          border-radius: 50%;
          background: rgba(212, 160, 23, 0.15);
          color: #f4d37a;
          font-size: 11px;
          font-weight: 700;
        }

        /* ================================================================
           HINT
        ================================================================ */
        .pw-hint {
          text-align: center;
          font-size: 0.78rem;
          letter-spacing: 0.08em;
          color: rgba(254, 243, 199, 0.6);
          margin-top: 1rem;
        }
        .pw-hint-arrow {
          display: inline-block;
          margin-left: 6px;
          animation: arrow-bob 1.6s ease-in-out infinite;
          color: #f4d37a;
        }

        /* ================================================================
           FADE HELPERS
        ================================================================ */
        /* Fade helpers. Elements are visible by default so they never
           remain hidden if the entrance animation is missed (e.g. when
           returning from bfcache, during hydration, or under
           prefers-reduced-motion). The "both" fill-mode + explicit
           from/to keyframe still produces the smooth animated entrance
           for fresh page loads. */
        .pw-fade {
          animation: fade-up 720ms cubic-bezier(0.22, 1, 0.36, 1) both;
          animation-delay: var(--d, 0ms);
        }

        /* ================================================================
           LIGHT MODE OVERRIDES
        ================================================================ */
        html:not([data-theme="dark"]) .pw-stage-gradient {
          background: radial-gradient(
              1400px 800px at 50% 0%,
              rgba(212, 160, 23, 0.16),
              transparent 55%
            ),
            radial-gradient(
              1100px 700px at 100% 100%,
              rgba(88, 70, 180, 0.08),
              transparent 55%
            ),
            radial-gradient(
              900px 500px at 0% 80%,
              rgba(184, 134, 11, 0.1),
              transparent 55%
            );
        }
        html:not([data-theme="dark"]) .pw-stage-grid {
          background-image: linear-gradient(
              rgba(184, 134, 11, 0.1) 1px,
              transparent 1px
            ),
            linear-gradient(
              90deg,
              rgba(184, 134, 11, 0.1) 1px,
              transparent 1px
            );
          opacity: 0.6;
        }
        html:not([data-theme="dark"]) .pw-stage-orb-1,
        html:not([data-theme="dark"]) .pw-stage-orb-2 {
          opacity: 0.35;
        }
        html:not([data-theme="dark"]) .pw-stage-orb-3 {
          opacity: 0.15;
        }
        html:not([data-theme="dark"]) .pw-stage-vignette {
          background: radial-gradient(
            ellipse at center,
            transparent 45%,
            rgba(0, 0, 0, 0.12) 100%
          );
        }
        html:not([data-theme="dark"]) .pw-particle-gold {
          background: rgba(184, 134, 11, 0.7);
          box-shadow: 0 0 10px rgba(184, 134, 11, 0.55);
        }
        html:not([data-theme="dark"]) .pw-particle-mauve {
          background: rgba(88, 70, 180, 0.55);
          box-shadow: 0 0 10px rgba(88, 70, 180, 0.55);
        }

        html:not([data-theme="dark"]) .pw-crown-mark {
          color: rgba(26, 22, 18, 0.55);
        }
        html:not([data-theme="dark"]) .pw-crown-line,
        html:not([data-theme="dark"]) .pw-eyebrow-line {
          background: linear-gradient(
            90deg,
            transparent,
            rgba(184, 134, 11, 0.5),
            transparent
          );
        }
        html:not([data-theme="dark"]) .pw-eyebrow-label {
          color: rgba(26, 22, 18, 0.55);
        }
        html:not([data-theme="dark"]) .pw-hero-kicker {
          color: rgba(26, 22, 18, 0.75);
          font-weight: 600;
        }
        html:not([data-theme="dark"]) .pw-hero-sub {
          color: rgba(26, 22, 18, 0.75);
        }
        html:not([data-theme="dark"]) .pw-hero-em {
          color: #8b6508;
          font-weight: 700;
        }

        html:not([data-theme="dark"]) .pw-emblem-inner {
          background: radial-gradient(
            circle at 30% 25%,
            rgba(244, 211, 122, 0.4),
            rgba(255, 251, 245, 0.95) 70%
          );
          border-color: rgba(184, 134, 11, 0.45);
          box-shadow:
            0 10px 40px rgba(184, 134, 11, 0.25),
            inset 0 0 18px rgba(184, 134, 11, 0.15);
        }
        html:not([data-theme="dark"]) .pw-emblem-halo {
          background: radial-gradient(
            circle,
            rgba(212, 160, 23, 0.4),
            transparent 62%
          );
        }
        html:not([data-theme="dark"]) .pw-emblem-ring-1 {
          border-color: rgba(184, 134, 11, 0.55);
        }
        html:not([data-theme="dark"]) .pw-emblem-ring-2 {
          border-color: rgba(184, 134, 11, 0.22);
        }
        html:not([data-theme="dark"]) .pw-emblem-ring-3 {
          border-color: rgba(184, 134, 11, 0.18);
        }
        html:not([data-theme="dark"]) .pw-emblem-orbit-dot {
          background: #b8860b;
          box-shadow: 0 0 6px #b8860b;
        }

        html:not([data-theme="dark"]) .pw-hero-title {
          color: #2a1f10;
          -webkit-text-fill-color: #2a1f10;
        }
        html:not([data-theme="dark"]) .pw-title-brand .pw-char {
          color: #8b6508;
          -webkit-text-fill-color: #8b6508;
        }

        html:not([data-theme="dark"]) .pw-meta-chip {
          background: rgba(255, 255, 255, 0.85);
          border-color: rgba(184, 134, 11, 0.3);
          color: rgba(26, 22, 18, 0.78);
        }
        html:not([data-theme="dark"]) .pw-meta-chip:hover {
          background: rgba(255, 255, 255, 1);
          border-color: rgba(184, 134, 11, 0.6);
        }
        html:not([data-theme="dark"]) .pw-meta-bullet {
          color: #b8860b;
        }

        html:not([data-theme="dark"]) .pw-notice {
          background: linear-gradient(
            165deg,
            rgba(255, 255, 255, 0.88),
            rgba(255, 251, 245, 0.88)
          );
          border-color: rgba(184, 134, 11, 0.2);
          color: rgba(26, 22, 18, 0.82);
          box-shadow: 0 10px 24px -14px rgba(26, 22, 18, 0.12);
        }

        html:not([data-theme="dark"]) .pw-quick-card,
        html:not([data-theme="dark"]) .pw-explore-card,
        html:not([data-theme="dark"]) .pw-cap-pill {
          background: linear-gradient(
            165deg,
            rgba(255, 255, 255, 0.92),
            rgba(255, 251, 245, 0.92)
          );
          border-color: rgba(184, 134, 11, 0.2);
          box-shadow: 0 6px 20px -10px rgba(26, 22, 18, 0.08);
        }
        html:not([data-theme="dark"]) .pw-quick-card:hover,
        html:not([data-theme="dark"]) .pw-explore-card:hover,
        html:not([data-theme="dark"]) .pw-cap-pill:hover {
          background: #ffffff;
          border-color: rgba(184, 134, 11, 0.55);
          box-shadow: 0 22px 40px -22px rgba(184, 134, 11, 0.35);
        }
        html:not([data-theme="dark"]) .pw-quick-glyph,
        html:not([data-theme="dark"]) .pw-explore-glyph,
        html:not([data-theme="dark"]) .pw-cap-glyph {
          background: linear-gradient(
            145deg,
            rgba(212, 160, 23, 0.28),
            rgba(212, 160, 23, 0.1)
          );
          border-color: rgba(184, 134, 11, 0.4);
          color: #8b6508;
        }
        html:not([data-theme="dark"]) .pw-quick-hint {
          color: rgba(26, 22, 18, 0.55);
        }
        html:not([data-theme="dark"]) .pw-quick-q,
        html:not([data-theme="dark"]) .pw-explore-theme,
        html:not([data-theme="dark"]) .pw-cap-label {
          color: #1a1612;
        }
        html:not([data-theme="dark"]) .pw-quick-arrow {
          color: rgba(26, 22, 18, 0.4);
        }
        html:not([data-theme="dark"]) .pw-quick-card:hover .pw-quick-arrow {
          color: #8b6508;
        }
        html:not([data-theme="dark"]) .pw-explore-prompt {
          color: rgba(26, 22, 18, 0.75);
        }
        html:not([data-theme="dark"]) .pw-explore-prompt:hover {
          background: rgba(184, 134, 11, 0.1);
          color: #1a1612;
        }
        html:not([data-theme="dark"]) .pw-prompt-arrow {
          color: rgba(184, 134, 11, 0.75);
        }
        html:not([data-theme="dark"]) .pw-explore-prompt:hover .pw-prompt-arrow {
          color: #8b6508;
        }

        html:not([data-theme="dark"]) .pw-card-border {
          background: linear-gradient(
            140deg,
            rgba(184, 134, 11, 0.35) 0%,
            rgba(184, 134, 11, 0.06) 45%,
            rgba(184, 134, 11, 0.32) 100%
          );
        }

        html:not([data-theme="dark"]) .pw-hint {
          color: rgba(26, 22, 18, 0.6);
        }
        html:not([data-theme="dark"]) .pw-hint-arrow {
          color: #b8860b;
        }

        /* ================================================================
           KEYFRAMES
        ================================================================ */
        @keyframes fade-up {
          from {
            opacity: 0;
            transform: translateY(12px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes card-in {
          from {
            opacity: 0;
            transform: translateY(14px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes char-in {
          from {
            opacity: 0;
            transform: translateY(28px) scale(0.9);
            filter: blur(8px);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
            filter: blur(0);
          }
        }
        @keyframes title-shimmer {
          0%, 100% {
            background-position: 0 0%;
          }
          50% {
            background-position: 0 80%;
          }
        }
        @keyframes emblem-in {
          from {
            opacity: 0;
            transform: scale(0.82);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
        @keyframes halo-breath {
          0%, 100% {
            opacity: 0.55;
            transform: scale(1);
          }
          50% {
            opacity: 1;
            transform: scale(1.08);
          }
        }
        @keyframes ring-spin {
          to {
            transform: rotate(360deg);
          }
        }
        @keyframes orbit-flicker {
          0%, 100% {
            opacity: 0.4;
            transform: translateY(-50%) scale(0.8);
          }
          50% {
            opacity: 1;
            transform: translateY(-50%) scale(1.2);
          }
        }
        @keyframes logo-float {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-4px);
          }
        }
        @keyframes live-pulse {
          0%, 100% {
            opacity: 1;
            transform: scale(1);
          }
          50% {
            opacity: 0.55;
            transform: scale(0.82);
          }
        }
        @keyframes bar-pulse {
          0%, 100% {
            opacity: 0.85;
          }
          50% {
            opacity: 1;
            box-shadow: 0 0 22px rgba(212, 160, 23, 0.85);
          }
        }
        @keyframes stage-grid {
          to {
            background-position: 72px 72px;
          }
        }
        @keyframes orb-a {
          0%, 100% {
            transform: translate(0, 0);
          }
          50% {
            transform: translate(40px, -30px);
          }
        }
        @keyframes orb-b {
          0%, 100% {
            transform: translate(0, 0);
          }
          50% {
            transform: translate(-30px, -40px);
          }
        }
        @keyframes orb-c {
          0%, 100% {
            transform: translate(-50%, -50%);
          }
          50% {
            transform: translate(-46%, -54%) scale(1.08);
          }
        }
        @keyframes beam-sweep {
          0%, 100% {
            opacity: 0;
            filter: blur(3px);
          }
          20%, 80% {
            opacity: 0.8;
            filter: blur(1px);
          }
          50% {
            opacity: 1;
            filter: blur(0.5px);
          }
        }
        @keyframes particle-rise {
          0% {
            opacity: 0;
            transform: translateY(0) scale(0.6);
          }
          12%, 88% {
            opacity: 0.9;
          }
          100% {
            opacity: 0;
            transform: translateY(-420px) scale(1);
          }
        }
        @keyframes arrow-bob {
          0%, 100% {
            transform: translateY(0);
            opacity: 1;
          }
          50% {
            transform: translateY(4px);
            opacity: 0.6;
          }
        }
      `}</style>
    </div>
  );
});
