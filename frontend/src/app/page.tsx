"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Image from "next/image";

/* ─── Types ─── */
interface Message {
  role: "user" | "assistant";
  content: string;
  references?: Reference[];
  timestamp?: number;
}
interface Reference {
  id?: number;
  document: string;
  page?: number | string;
  page_start?: number | string;
  open_url?: string;
  snippet?: string;
  score?: number;
  chunk_index?: number;
}
interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
}

/* ─── Constants ─── */
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const genId = () => Math.random().toString(36).slice(2) + Date.now().toString(36);

const THINKING_PHASES = [
  { icon: "🔎", text: "Scanning knowledge base..." },
  { icon: "📑", text: "Cross-referencing documents..." },
  { icon: "⚙️", text: "Synthesizing insights..." },
  { icon: "✍️", text: "Composing answer..." },
];

const SUGGESTIONS = [
  { emoji: "⚖️", title: "PERA Powers", desc: "Authority ke powers kya hain?", q: "What are the powers of PERA?", cat: "Governance" },
  { emoji: "👤", title: "CTO Role", desc: "CTO ki responsibilities", q: "CTO ki powers kia hain?", cat: "Roles" },
  { emoji: "💰", title: "Pay & Benefits", desc: "Salary scales & allowances", q: "CTO pay in numbers", cat: "Finance" },
  { emoji: "📋", title: "EPO Rules", desc: "Emergency orders process", q: "EPO kaise issue hota hai?", cat: "Enforcement" },
  { emoji: "🏛️", title: "Board Composition", desc: "Authority members & structure", q: "Board of Authority ka composition kya hai?", cat: "Governance" },
  { emoji: "🔒", title: "Confidentiality", desc: "Information disclosure rules", q: "Confidentiality rules for employees", cat: "Compliance" },
];

const STATS = [
  { value: "24/7", label: "Available" },
  { value: "99%", label: "Accuracy" },
  { value: "1000+", label: "Sections" },
  { value: "Instant", label: "Response" },
];

const CAPABILITIES = [
  "Voice Input",
  "PDF Viewer",
  "Citation Links",
  "Dark Mode",
  "Chat History",
  "Instant Search",
];

/* ─── Time Helper ─── */
const timeAgo = (ts?: number) => {
  if (!ts) return "";
  const d = Math.floor((Date.now() - ts) / 1000);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
};

/* ═════════════════════════════ COMPONENT ═════════════════════════════ */
export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([]);
  const [currentChatId, setCurrentChatId] = useState(genId());
  const [thinkingPhase, setThinkingPhase] = useState(0);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [latestIsNew, setLatestIsNew] = useState(false);
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [pdfModal, setPdfModal] = useState<{ url: string; title: string } | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("dark");

  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  /* ─── Theme ─── */
  useEffect(() => {
    const saved = localStorage.getItem("pera-theme") as "light" | "dark" | null;
    if (saved) setTheme(saved);
  }, []);
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("pera-theme", theme);
  }, [theme]);
  const toggleTheme = () => setTheme(p => (p === "dark" ? "light" : "dark"));

  /* ─── Thinking Phase Rotation ─── */
  useEffect(() => {
    if (!loading) return;
    setThinkingPhase(0);
    const t = setInterval(() => setThinkingPhase(p => (p + 1) % THINKING_PHASES.length), 2200);
    return () => clearInterval(t);
  }, [loading]);

  /* ─── Auto-scroll ─── */
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, displayedText]);

  /* ─── Typewriter Effect ─── */
  useEffect(() => {
    if (!latestIsNew || messages.length === 0) return;
    const last = messages[messages.length - 1];
    if (last.role !== "assistant") return;
    const full = last.content;
    setIsTyping(true);
    setDisplayedText("");
    let i = 0;
    const speed = Math.max(6, Math.min(18, 1500 / full.length));
    const t = setInterval(() => {
      i++;
      setDisplayedText(full.slice(0, i));
      if (i >= full.length) {
        clearInterval(t);
        setIsTyping(false);
        setLatestIsNew(false);
      }
    }, speed);
    return () => clearInterval(t);
  }, [latestIsNew, messages]);

  /* ─── Save Chat History ─── */
  const saveChat = useCallback(() => {
    if (messages.length === 0) return;
    const title = messages[0].content.slice(0, 38);
    setChatHistory(prev => {
      const idx = prev.findIndex(c => c.id === currentChatId);
      const session: ChatSession = {
        id: currentChatId,
        title: title.length >= 38 ? title + "..." : title,
        messages,
        createdAt: new Date(),
      };
      if (idx >= 0) {
        const u = [...prev];
        u[idx] = session;
        return u;
      }
      return [session, ...prev];
    });
  }, [messages, currentChatId]);

  const startNewChat = () => {
    saveChat();
    setMessages([]);
    setCurrentChatId(genId());
    setSidebarOpen(false);
  };
  const loadChat = (s: ChatSession) => {
    saveChat();
    setMessages(s.messages);
    setCurrentChatId(s.id);
    setSidebarOpen(false);
  };
  const deleteChat = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setChatHistory(prev => prev.filter(c => c.id !== id));
    if (id === currentChatId) startNewChat();
  };

  /* ─── Copy ─── */
  const copyText = (text: string, i: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(i);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  /* ─── Send Message ─── */
  const sendMessage = async (text: string) => {
    if (!text.trim()) return;
    const userMsg: Message = { role: "user", content: text, timestamp: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setLatestIsNew(false);

    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      console.log("Sending request to:", `${API_URL}/api/ask`);

      const res = await fetch(`${API_URL}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, conversation_history: history }),
      });

      if (!res.ok) {
        const errText = await res.text();
        console.error(`API Error: ${res.status} ${res.statusText}`, errText);
        throw new Error(`API Error: ${res.status} ${res.statusText} - ${errText}`);
      }

      const data = await res.json();
      const botMsg: Message = {
        role: "assistant",
        content: data.answer || "Sorry, I could not process that.",
        references: data.references || [],
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, botMsg]);
      setLatestIsNew(true);
    } catch (err: any) {
      console.error("Fetch failed:", err);
      // Show alert for critical errors like 405
      if (err.message && err.message.includes("405")) {
        alert("System Error: 405 Method Not Allowed. Please check backend logs.");
      }
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: `⚠️ Connection error: ${err.message || "Unknown error"}. Please try again.`, timestamp: Date.now() },
      ]);
    } finally {
      setLoading(false);
    }
  };

  /* ─── Voice ─── */
  const toggleRecording = async () => {
    // 1. Check for Browser Support / Secure Context
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert(
        "Microphone Access Blocked!\n\n" +
        "Reason: Browsers block the microphone on insecure (HTTP) connections when using an IP address.\n\n" +
        "Solutions:\n" +
        "1. Access via http://localhost:3000 (if on the same machine).\n" +
        "2. Enable 'Insecure origins treated as secure' in chrome://flags.\n" +
        "3. Set up HTTPS."
      );
      return;
    }

    if (isRecording) {
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      chunksRef.current = [];
      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const fd = new FormData();
        fd.append("audio", blob, "voice.webm");
        try {
          const r = await fetch(`${API_URL}/transcribe`, { method: "POST", body: fd });
          const d = await r.json();
          if (d.text) { setInput(d.text); inputRef.current?.focus(); }
        } catch { /* no-op */ }
      };
      mr.start();
      setIsRecording(true);
    } catch (err: any) {
      console.error(err);
      alert("Microphone Error: " + (err.message || "Could not access microphone. Ensure you are on HTTPS or localhost."));
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(input); }
  };

  /* ─── Open PDF ─── */
  const openPdf = (ref: Reference) => {
    const pg = ref.page_start || ref.page || 1;
    const url = ref.open_url
      ? ref.open_url.replace(/^https?:\/\/[^/]+/, API_URL)
      : `${API_URL}/pdf/${encodeURIComponent(ref.document)}#page=${pg}`;
    setPdfModal({ url, title: `📄 ${ref.document} — Page ${pg}` });
  };

  /* ─── Render Bot Content ─── */
  const renderBotContent = (msg: Message, idx: number) => {
    const isLast = idx === messages.length - 1;
    const text = isLast && isTyping ? displayedText : msg.content;
    const rawLines = text.split("\n");

    // Group consecutive bullets/numbers into proper lists
    const elements: JSX.Element[] = [];
    let listBuffer: { type: "ul" | "ol"; items: string[] } | null = null;

    const flushList = () => {
      if (!listBuffer) return;
      const Tag = listBuffer.type;
      elements.push(
        <Tag key={`list-${elements.length}`} className={`msg-list msg-list-${listBuffer.type}`}>
          {listBuffer.items.map((item, j) => (
            <li key={j} dangerouslySetInnerHTML={{ __html: formatInline(item) }} />
          ))}
        </Tag>
      );
      listBuffer = null;
    };

    const formatInline = (s: string) =>
      s
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/\[(\d+)\]/g, '<span class="cite-badge">[$1]</span>');

    for (let i = 0; i < rawLines.length; i++) {
      const line = rawLines[i];
      const trimmed = line.trim();

      if (!trimmed) {
        flushList();
        if (i > 0 && rawLines[i - 1]?.trim()) elements.push(<div key={`sp-${i}`} className="h-2" />);
        continue;
      }

      // Heading
      if (trimmed.startsWith("## ")) {
        flushList();
        elements.push(<h4 key={`h-${i}`} className="msg-heading" dangerouslySetInnerHTML={{ __html: formatInline(trimmed.slice(3)) }} />);
        continue;
      }

      // Bullet
      if (/^[-•]\s/.test(trimmed)) {
        const content = trimmed.replace(/^[-•]\s*/, "");
        if (listBuffer?.type === "ul") {
          listBuffer.items.push(content);
        } else {
          flushList();
          listBuffer = { type: "ul", items: [content] };
        }
        continue;
      }

      // Numbered
      if (/^\d+[.)]\s/.test(trimmed)) {
        const content = trimmed.replace(/^\d+[.)]\s*/, "");
        if (listBuffer?.type === "ol") {
          listBuffer.items.push(content);
        } else {
          flushList();
          listBuffer = { type: "ol", items: [content] };
        }
        continue;
      }

      // Normal paragraph
      flushList();
      elements.push(<p key={`p-${i}`} className="msg-para" dangerouslySetInnerHTML={{ __html: formatInline(trimmed) }} />);
    }
    flushList();

    return (
      <div className="msg-bot-text">
        {elements}
        {isLast && isTyping && <span className="typewriter-cursor" />}
      </div>
    );
  };

  const showWelcome = messages.length === 0 && !loading;

  /* ═══════════════════════════════ RENDER ═══════════════════════════════ */
  return (
    <div className="flex h-screen overflow-hidden relative" style={{ background: "var(--bg-page)" }}>
      {/* Ambient Glow */}
      <div className="ambient-bg" />

      {/* ─── Sidebar ─── */}
      <aside
        className={`sidebar fixed md:relative h-full flex flex-col transition-all duration-300
          ${sidebarOpen ? "w-72 translate-x-0" : "w-0 -translate-x-full md:w-0 md:-translate-x-full"}`}
        style={{ overflow: "hidden" }}
      >
        {sidebarOpen && (
          <div className="flex flex-col h-full w-72 p-4" style={{ overflow: "hidden" }}>
            {/* Sidebar Header */}
            <div className="flex items-center gap-3 mb-5">
              <div className="w-9 h-9 rounded-xl overflow-hidden" style={{ boxShadow: "var(--shadow-glow)" }}>
                <Image src="/pera_logo.png" alt="PERA" width={36} height={36} className="animate-float" />
              </div>
              <div>
                <h2 className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>PERA AI</h2>
                <p className="text-[10px]" style={{ color: "var(--text-faint)" }}>Chat History</p>
              </div>
            </div>

            {/* New Chat */}
            <button onClick={startNewChat} className="new-chat-btn flex items-center justify-center gap-2 py-2.5 mb-4 text-sm w-full">
              <span>✦</span> New Chat
            </button>

            {/* Chat List */}
            <div className="flex-1 overflow-y-auto space-y-1">
              {chatHistory.length === 0 ? (
                <p className="text-center text-xs py-8" style={{ color: "var(--text-faint)" }}>No conversations yet</p>
              ) : (
                chatHistory.map(s => (
                  <div
                    key={s.id}
                    onClick={() => loadChat(s)}
                    className={`sidebar-item group flex items-center gap-2 ${s.id === currentChatId ? "active" : ""}`}
                  >
                    <span className="text-sm">💬</span>
                    <span className="flex-1 text-xs truncate" style={{ color: "var(--text-primary)" }}>{s.title}</span>
                    <button
                      onClick={e => deleteChat(s.id, e)}
                      className="opacity-0 group-hover:opacity-100 text-[10px] px-1.5 py-0.5 rounded-md transition-opacity"
                      style={{ color: "var(--red)", background: "var(--bg-hover)" }}
                    >✕</button>
                  </div>
                ))
              )}
            </div>

            {/* Sidebar Footer */}
            <div className="pt-3 mt-2" style={{ borderTop: "1px solid var(--border)" }}>
              <p className="text-[10px] text-center" style={{ color: "var(--text-faint)" }}>
                Built by PERA AI TEAM
              </p>
            </div>
          </div>
        )}
      </aside>

      {/* Mobile Backdrop */}
      {sidebarOpen && <div className="fixed inset-0 z-20 bg-black/40 md:hidden" onClick={() => setSidebarOpen(false)} />}

      {/* ─── Main Area ─── */}
      <main className="flex-1 flex flex-col relative z-10 min-w-0">
        {/* Header */}
        <header className="glass-header flex items-center justify-between px-4 md:px-6 py-3 z-20 relative">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-xl transition-colors"
              style={{ color: "var(--text-secondary)" }}
              onMouseEnter={e => (e.currentTarget.style.background = "var(--bg-hover)")}
              onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="15" y2="12" /><line x1="3" y1="18" x2="18" y2="18" />
              </svg>
            </button>
            <Image src="/pera_logo.png" alt="PERA" width={32} height={32} className="rounded-xl animate-float" />
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>PERA AI</h1>
                <span className="badge" style={{ background: "var(--accent-soft)", color: "var(--accent)" }}>v2.0</span>
              </div>
              <p className="text-[10px] hidden sm:block" style={{ color: "var(--text-faint)" }}>Punjab Enforcement & Regulatory Authority</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Theme Toggle */}
            <div className="theme-toggle" onClick={toggleTheme} title={theme === "dark" ? "Switch to Light" : "Switch to Dark"}>
              <div className="theme-toggle-knob">{theme === "dark" ? "🌙" : "☀️"}</div>
            </div>
            {/* Online Badge */}
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full" style={{ background: "var(--bg-hover)" }}>
              <div className="w-2 h-2 rounded-full status-ping" style={{ background: "var(--green)" }} />
              <span className="text-[11px] font-medium" style={{ color: "var(--text-secondary)" }}>Online</span>
            </div>
          </div>
        </header>

        {/* ─── Chat Messages ─── */}
        <div className="flex-1 overflow-y-auto px-4 md:px-0" style={{ background: "var(--bg-chat)" }}>
          <div className="max-w-3xl mx-auto py-6 space-y-5">
            {/* Welcome */}
            {showWelcome && (
              <div className="flex flex-col items-center justify-center min-h-[70vh] animate-fade-in px-2">
                {/* Logo with Glow Ring */}
                <div className="relative mb-5">
                  <div className="absolute inset-[-8px] rounded-3xl opacity-40" style={{ background: "conic-gradient(from 0deg, #b8860b, #daa520, #e6b422, #b8860b)", filter: "blur(12px)", animation: "orbSpin 6s linear infinite" }} />
                  <div className="relative w-20 h-20 rounded-2xl overflow-hidden" style={{ boxShadow: "0 8px 40px var(--accent-glow)", border: "2px solid var(--border)" }}>
                    <Image src="/pera_logo.png" alt="PERA" width={80} height={80} className="animate-float" />
                  </div>
                  <div className="absolute -bottom-1 -right-1 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold" style={{ background: "var(--green)", border: "3px solid var(--bg-page)", color: "white" }}>✓</div>
                </div>

                {/* Hero Text */}
                <h2 className="gradient-text text-3xl md:text-5xl font-extrabold mb-2 text-center tracking-tight">PERA AI Assistant</h2>
                <p className="text-sm mb-1 text-center max-w-md" style={{ color: "var(--text-secondary)" }}>
                  Punjab Enforcement & Regulatory Authority
                </p>
                <p className="text-xs mb-5 text-center max-w-md" style={{ color: "var(--text-faint)" }}>
                  Your AI-powered guide to PERA documents, rules, regulations & governance
                </p>

                {/* Stat Strip */}
                <div className="flex items-center justify-center gap-0 rounded-2xl overflow-hidden mb-6 animate-fade-in" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", boxShadow: "var(--shadow-md)" }}>
                  {STATS.map((s, i) => (
                    <div key={i} className="flex flex-col items-center px-6 py-3.5" style={{ borderRight: i < STATS.length - 1 ? "1px solid var(--border)" : "none" }}>
                      <span className="font-extrabold text-lg tracking-tight" style={{ color: "var(--gold)" }}>{s.value}</span>
                      <span className="text-[9px] font-semibold tracking-widest uppercase mt-0.5" style={{ color: "var(--text-faint)" }}>{s.label}</span>
                    </div>
                  ))}
                </div>

                {/* Suggestion Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 w-full max-w-2xl mb-6">
                  {SUGGESTIONS.map((s, i) => (
                    <button key={i} onClick={() => sendMessage(s.q)} className="suggestion-card text-left px-4 py-4 animate-fade-in" style={{ animationDelay: `${i * 80}ms` }}>
                      <div className="relative z-10">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-2xl">{s.emoji}</span>
                          <span className="text-[9px] font-bold tracking-wider px-2 py-0.5 rounded-full uppercase" style={{ background: "var(--accent-soft)", color: "var(--accent)" }}>{s.cat}</span>
                        </div>
                        <h3 className="font-semibold text-sm mb-0.5" style={{ color: "var(--text-primary)" }}>{s.title}</h3>
                        <p className="text-xs" style={{ color: "var(--text-faint)" }}>{s.desc}</p>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Capability Pills */}
                <div className="flex flex-wrap justify-center gap-2">
                  {CAPABILITIES.map((c, i) => (
                    <div key={i} className="px-3 py-1 rounded-full text-[10px] font-medium tracking-wide animate-fade-in" style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-faint)", animationDelay: `${300 + i * 60}ms` }}>
                      {c}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} gap-2.5`}>
                {/* Bot Avatar */}
                {msg.role === "assistant" && (
                  <div className="flex-shrink-0 mt-1">
                    <div className="w-8 h-8 rounded-xl overflow-hidden flex items-center justify-center" style={{ background: "var(--accent-soft)" }}>
                      <Image src="/pera_logo.png" alt="" width={20} height={20} className="rounded-md" />
                    </div>
                  </div>
                )}

                {/* Message */}
                <div className={`max-w-[82%] md:max-w-[72%] ${msg.role === "user" ? "user-bubble" : "bot-bubble"}`}>
                  <div className={`px-4 py-3 ${msg.role === "user" ? "relative z-10" : ""}`}>
                    {msg.role === "user" ? (
                      <p className="text-sm leading-relaxed text-white">{msg.content}</p>
                    ) : (
                      <>
                        {renderBotContent(msg, i)}

                        {/* References */}
                        {msg.references && msg.references.length > 0 && !(i === messages.length - 1 && isTyping) && (
                          <div className="mt-3 pt-3 flex flex-wrap gap-1.5" style={{ borderTop: "1px solid var(--border)" }}>
                            <span className="text-[10px] font-medium self-center mr-1" style={{ color: "var(--text-faint)" }}>Sources:</span>
                            {msg.references.slice(0, 8).map((ref, ri) => (
                              <button key={ri} onClick={() => openPdf(ref)} className="ref-chip">
                                {ref.id ? `[${ref.id}]` : `[${ri + 1}]`} {ref.document?.replace(/\.pdf$/i, "").slice(0, 20)}{(ref.page_start || ref.page) ? ` · p.${ref.page_start || ref.page}` : ""}
                              </button>
                            ))}
                          </div>
                        )}

                        {/* Copy */}
                        <button
                          onClick={() => copyText(msg.content, i)}
                          className="copy-btn absolute top-2 right-2 text-[10px] px-2 py-1 rounded-lg font-medium"
                          style={{ background: "var(--bg-hover)", color: "var(--text-faint)" }}
                        >
                          {copiedIndex === i ? "Copied ✓" : "Copy"}
                        </button>
                      </>
                    )}
                  </div>
                  {/* Timestamp */}
                  <div className={`px-4 pb-2 text-[10px] ${msg.role === "user" ? "text-right text-white/50" : ""}`} style={msg.role === "assistant" ? { color: "var(--text-faint)" } : {}}>
                    {timeAgo(msg.timestamp)}
                  </div>
                </div>
              </div>
            ))}

            {/* Thinking */}
            {loading && (
              <div className="flex gap-2.5 thinking-container">
                <div className="flex-shrink-0 mt-1">
                  <div className="w-8 h-8 rounded-xl overflow-hidden flex items-center justify-center" style={{ background: "var(--accent-soft)" }}>
                    <Image src="/pera_logo.png" alt="" width={20} height={20} className="rounded-md" />
                  </div>
                </div>
                <div className="bot-bubble px-5 py-4 max-w-xs">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="thinking-wave-wrapper">
                      <div className="thinking-bar" />
                      <div className="thinking-bar" />
                      <div className="thinking-bar" />
                      <div className="thinking-bar" />
                      <div className="thinking-bar" />
                    </div>
                    <p className="text-sm font-medium thinking-text" style={{ color: "var(--text-primary)" }}>
                      {THINKING_PHASES[thinkingPhase].text}
                    </p>
                  </div>
                  <div className="thinking-shimmer w-full" />
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>
        </div>

        {/* ─── Input Bar ─── */}
        <div className="px-4 md:px-0 pt-3 pb-6 relative z-10" style={{ background: "var(--bg-page)", borderTop: "1px solid var(--border)" }}>
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end gap-2">
              {/* Input Area with Integrated Buttons */}
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask about PERA rules, powers, pay scales..."
                  rows={1}
                  className="chat-input w-full resize-none px-4 py-3 pr-24 text-sm"
                  style={{ maxHeight: 120, minHeight: 44 }}
                  disabled={loading}
                />

                {/* Voice Input */}
                <button
                  onClick={toggleRecording}
                  className={`absolute right-12 bottom-2 w-9 h-9 rounded-xl flex items-center justify-center transition-all z-10 ${isRecording ? "recording-pulse" : ""}`}
                  style={{
                    background: isRecording ? "var(--red)" : "transparent",
                    color: isRecording ? "white" : "var(--text-secondary)",
                  }}
                  title={isRecording ? "Stop" : "Voice Input"}
                  onMouseEnter={isRecording ? undefined : e => (e.currentTarget.style.background = "var(--bg-hover)")}
                  onMouseLeave={isRecording ? undefined : e => (e.currentTarget.style.background = "transparent")}
                >
                  {isRecording ? (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
                  ) : (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" y1="19" x2="12" y2="23" /><line x1="8" y1="23" x2="16" y2="23" />
                    </svg>
                  )}
                </button>

                {/* Send Button */}
                <button
                  onClick={() => sendMessage(input)}
                  disabled={!input.trim() || loading}
                  className="send-btn absolute right-2 bottom-2 w-9 h-9 flex items-center justify-center"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                  </svg>
                </button>
              </div>
            </div>
            <p className="text-center text-[10px] mt-2" style={{ color: "var(--text-faint)" }}>
              For verified information, always refer to official PERA documents and notifications
            </p>
          </div>
        </div>
      </main>

      {/* ─── PDF Viewer Modal ─── */}
      {pdfModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="modal-overlay absolute inset-0" onClick={() => setPdfModal(null)} />
          <div className="modal-content relative w-full max-w-5xl h-[85vh] flex flex-col overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <div className="flex items-center gap-2">
                <span className="text-lg">📄</span>
                <span className="font-semibold text-sm" style={{ color: "var(--text-primary)" }}>{pdfModal.title}</span>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={pdfModal.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs px-3 py-1.5 rounded-lg font-medium transition-colors"
                  style={{ background: "var(--bg-hover)", color: "var(--text-secondary)" }}
                >
                  Open in Tab ↗
                </a>
                <button
                  onClick={() => setPdfModal(null)}
                  className="w-8 h-8 rounded-lg flex items-center justify-center transition-colors text-sm"
                  style={{ background: "var(--bg-hover)", color: "var(--text-secondary)" }}
                >✕</button>
              </div>
            </div>
            <iframe src={pdfModal.url} className="flex-1 w-full border-0" title="PDF Viewer" />
          </div>
        </div>
      )}
    </div>
  );
}
