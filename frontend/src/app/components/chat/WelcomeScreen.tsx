"use client";

import Image from "next/image";
import { memo } from "react";

const SUGGESTIONS = [
  { emoji: "⚖️", title: "PERA Powers", desc: "Authority ke powers kya hain?", cat: "Governance" },
  { emoji: "👤", title: "CTO Role", desc: "CTO ki responsibilities kya hain?", cat: "Roles" },
  { emoji: "💰", title: "Pay & Benefits", desc: "Salary scales & allowances", cat: "Finance" },
  { emoji: "📋", title: "EPO Rules", desc: "EPO kaise issue hota hai?", cat: "Enforcement" },
  { emoji: "🏛️", title: "Board Composition", desc: "Board of Authority ka composition kya hai?", cat: "Governance" },
  { emoji: "🔒", title: "Confidentiality", desc: "Confidentiality rules for employees", cat: "Compliance" },
];

const STATS = [
  { value: "24/7", label: "Available" },
  { value: "Multi-doc", label: "Search" },
  { value: "Bilingual", label: "EN / UR" },
  { value: "Cited", label: "Responses" },
];

const CAPABILITIES = [
  "Voice Input",
  "PDF Viewer",
  "Citation Links",
  "Dark Mode",
  "Chat History",
  "Bilingual",
];

interface Props {
  onSendMessage: (text: string) => void;
}

export const WelcomeScreen = memo(function WelcomeScreen({ onSendMessage }: Props) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] animate-fade-in px-2">
      {/* Logo with Glow Ring */}
      <div className="relative mb-5">
        <div
          className="absolute inset-[-8px] rounded-3xl opacity-40"
          style={{
            background: "conic-gradient(from 0deg, #b8860b, #daa520, #e6b422, #b8860b)",
            filter: "blur(12px)",
            animation: "orbSpin 6s linear infinite",
          }}
        />
        <div
          className="relative w-20 h-20 rounded-2xl overflow-hidden"
          style={{ boxShadow: "0 8px 40px var(--accent-glow)", border: "2px solid var(--border)" }}
        >
          <Image src="/pera_logo.png" alt="PERA AI" width={80} height={80} priority />
        </div>
        <div
          className="absolute -bottom-1 -right-1 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
          style={{ background: "var(--green)", border: "3px solid var(--bg-page)", color: "white" }}
        >
          ✓
        </div>
      </div>

      {/* Hero Text */}
      <h2 className="gradient-text text-3xl md:text-5xl font-extrabold mb-3 text-center tracking-tight">
        PERA AI Assistant
      </h2>
      <p className="text-sm font-semibold tracking-wide mb-5 text-center max-w-md leading-relaxed" style={{ color: "var(--text-primary)" }}>
        An AI-powered assistant for navigating PERA documents, regulations, and governance
      </p>

      {/* Stat Strip */}
      <div
        className="flex items-center justify-center gap-0 rounded-2xl overflow-hidden mb-6 animate-fade-in"
        style={{
          background: "var(--bg-card)",
          border: "1px solid var(--border)",
          boxShadow: "var(--shadow-md)",
        }}
      >
        {STATS.map((s, i) => (
          <div
            key={i}
            className="flex flex-col items-center px-5 sm:px-6 py-3.5"
            style={{ borderRight: i < STATS.length - 1 ? "1px solid var(--border)" : "none" }}
          >
            <span className="font-extrabold text-base sm:text-lg tracking-tight" style={{ color: "var(--gold)" }}>
              {s.value}
            </span>
            <span className="text-[9px] font-semibold tracking-widest uppercase mt-0.5" style={{ color: "var(--text-faint)" }}>
              {s.label}
            </span>
          </div>
        ))}
      </div>

      {/* Suggestion Cards — visible text IS the question sent */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 w-full max-w-2xl mb-6">
        {SUGGESTIONS.map((s, i) => (
          <button
            key={i}
            onClick={() => onSendMessage(s.desc)}
            className="suggestion-card text-left px-4 py-4 animate-fade-in"
            style={{ animationDelay: `${i * 80}ms` }}
          >
            <div className="relative z-10">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-2xl">{s.emoji}</span>
                <span
                  className="text-[9px] font-bold tracking-wider px-2 py-0.5 rounded-full uppercase"
                  style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
                >
                  {s.cat}
                </span>
              </div>
              <h3 className="font-semibold text-sm mb-0.5" style={{ color: "var(--text-primary)" }}>
                {s.title}
              </h3>
              <p className="text-xs" style={{ color: "var(--text-faint)" }}>
                {s.desc}
              </p>
            </div>
          </button>
        ))}
      </div>

      {/* Capability Pills */}
      <div className="flex flex-wrap justify-center gap-2">
        {CAPABILITIES.map((c, i) => (
          <div
            key={i}
            className="px-3 py-1 rounded-full text-[10px] font-medium tracking-wide animate-fade-in"
            style={{
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              color: "var(--text-faint)",
              animationDelay: `${300 + i * 60}ms`,
            }}
          >
            {c}
          </div>
        ))}
      </div>
    </div>
  );
});
