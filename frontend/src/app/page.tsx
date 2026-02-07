"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";

// Types
interface Message {
  role: "user" | "assistant";
  content: string;
  references?: Reference[];
}

interface Reference {
  document: string;
  page_start: number;
  snippet: string;
  open_url?: string;
}

interface PDFSelection {
  filename: string;
  page: number;
  title: string;
  snippet: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
}

const API_URL = "http://localhost:8000";

// Generate unique ID
const generateId = () => Math.random().toString(36).substring(2, 9);

// Format markdown-like text
const formatAnswer = (text: string) => {
  // Highlight creator name first (before other formatting)
  let formatted = text.replace(
    /Muhammad Ahsan Sajjad/gi,
    '<span class="text-amber-400 font-bold bg-amber-500/20 px-1 rounded">Muhammad Ahsan Sajjad</span>'
  );
  formatted = formatted.replace(
    /Ahsan Sajjad/gi,
    '<span class="text-amber-400 font-bold bg-amber-500/20 px-1 rounded">Ahsan Sajjad</span>'
  );

  // Bold
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong class="text-amber-400">$1</strong>');

  // Process line by line
  const lines = formatted.split('\n');
  const processedLines = lines.map(line => {
    const trimmed = line.trim();
    // Bullet points
    if (trimmed.startsWith('- ')) {
      return '<li class="ml-4 my-1">' + trimmed.substring(2) + '</li>';
    }
    // Numbered lists (1. 2. 3. etc)
    const numMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (numMatch) {
      return '<li class="ml-4 my-1"><span class="text-amber-400 font-bold mr-2">' + numMatch[1] + '.</span>' + numMatch[2] + '</li>';
    }
    return line;
  });

  formatted = processedLines.join('\n');

  // Wrap consecutive list items
  formatted = formatted.replace(/(<li[^>]*>[\s\S]*?<\/li>\n?)+/g, '<ul class="my-2">$&</ul>');

  // Newlines to breaks (but not inside lists)
  formatted = formatted.replace(/\n/g, '<br/>');

  // Clean up extra breaks inside lists
  formatted = formatted.replace(/<ul([^>]*)><br\/>/g, '<ul$1>');
  formatted = formatted.replace(/<\/li><br\/>/g, '</li>');

  return formatted;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [selectedPDF, setSelectedPDF] = useState<PDFSelection | null>(null);
  const [showPDF, setShowPDF] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string>(generateId());
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Load chat history from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("pera_chat_history");
    if (saved) {
      const parsed = JSON.parse(saved);
      setChatHistory(parsed.map((c: ChatSession) => ({
        ...c,
        createdAt: new Date(c.createdAt)
      })));
    }
  }, []);

  // Save chat history to localStorage
  useEffect(() => {
    if (chatHistory.length > 0) {
      localStorage.setItem("pera_chat_history", JSON.stringify(chatHistory));
    }
  }, [chatHistory]);

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Save current chat to history
  const saveCurrentChat = () => {
    if (messages.length === 0) return;

    const firstUserMsg = messages.find(m => m.role === "user");
    const title = firstUserMsg?.content.slice(0, 40) || "New Chat";

    setChatHistory(prev => {
      const existing = prev.findIndex(c => c.id === currentChatId);
      const session: ChatSession = {
        id: currentChatId,
        title: title + (title.length >= 40 ? "..." : ""),
        messages: messages,
        createdAt: new Date(),
      };

      if (existing >= 0) {
        const updated = [...prev];
        updated[existing] = session;
        return updated;
      }
      return [session, ...prev];
    });
  };

  // Start new chat
  const startNewChat = () => {
    saveCurrentChat();
    setMessages([]);
    setCurrentChatId(generateId());
    setSelectedPDF(null);
    setShowPDF(false);
  };

  // Load a chat from history
  const loadChat = (session: ChatSession) => {
    saveCurrentChat();
    setMessages(session.messages);
    setCurrentChatId(session.id);
  };

  // Clear all history
  const clearHistory = () => {
    if (confirm("Clear all chat history?")) {
      localStorage.removeItem("pera_chat_history");
      setChatHistory([]);
      startNewChat();
    }
  };

  // Delete single chat
  const deleteChat = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setChatHistory(prev => prev.filter(c => c.id !== id));
    if (id === currentChatId) {
      startNewChat();
    }
  };

  // Send message
  const sendMessage = async (text: string) => {
    if (!text.trim()) return;

    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const history = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const res = await fetch(`${API_URL}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          conversation_history: history,
        }),
      });

      const data = await res.json();

      const botMsg: Message = {
        role: "assistant",
        content: data.answer || "Jawab nahi mila.",
        references: data.references || [],
      };

      setMessages((prev) => [...prev, botMsg]);

      // Auto-select first reference but DO NOT show PDF automatically
      if (data.references?.length > 0) {
        const ref = data.references[0];
        setSelectedPDF({
          filename: ref.document || "unknown",
          page: ref.page_start || 1,
          title: `${ref.document || "Document"} - Page ${ref.page_start || 1}`,
          snippet: ref.snippet || "",
        });
        // setShowPDF(true); // User requested manual open only
      }

      // Auto-save to history
      setTimeout(saveCurrentChat, 500);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error connecting to server." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Voice recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        stream.getTracks().forEach((track) => track.stop());

        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");

        try {
          const res = await fetch(`${API_URL}/transcribe`, {
            method: "POST",
            body: formData,
          });
          const data = await res.json();
          if (data.text) {
            setInput(data.text);
          }
        } catch {
          console.error("Transcription failed");
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch {
      alert("Microphone access denied");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  return (
    <div className="h-screen overflow-hidden flex text-white bg-slate-900">
      {/* Sidebar - Chat History */}
      <div className={`${sidebarOpen ? 'w-72' : 'w-0'} transition-all duration-300 overflow-hidden`}>
        <div className="w-72 h-full glass-dark flex flex-col">
          {/* Sidebar Header */}
          <div className="p-4 border-b border-white/10">
            <button
              onClick={startNewChat}
              className="w-full py-3 px-4 bg-gradient-to-r from-amber-500 to-orange-500 rounded-xl text-white font-semibold flex items-center justify-center gap-2 hover:from-amber-400 hover:to-orange-400 transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Chat
            </button>
          </div>

          {/* Chat List */}
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {chatHistory.length === 0 ? (
              <p className="text-gray-500 text-sm text-center py-4">No chat history</p>
            ) : (
              chatHistory.map((session) => (
                <div
                  key={session.id}
                  onClick={() => loadChat(session)}
                  className={`group p-3 rounded-xl cursor-pointer transition-all hover:bg-white/10 ${session.id === currentChatId ? 'bg-white/10 border border-amber-500/30' : ''
                    }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <p className="text-white text-sm font-medium truncate">{session.title}</p>
                      <p className="text-gray-500 text-xs mt-1">
                        {session.messages.length} messages
                      </p>
                    </div>
                    <button
                      onClick={(e) => deleteChat(session.id, e)}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition-all"
                    >
                      <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Clear All */}
          {chatHistory.length > 0 && (
            <div className="p-3 border-t border-white/10">
              <button
                onClick={clearHistory}
                className="w-full py-2 text-red-400 text-sm hover:bg-red-500/10 rounded-lg transition-all"
              >
                🗑️ Clear All History
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="p-4 flex items-center gap-4 border-b border-white/10">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-white/10 rounded-lg transition-all"
          >
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          <Image
            src="/pera_logo.png"
            alt="PERA Logo"
            width={50}
            height={50}
            className="rounded-xl animate-float"
          />
          <div>
            <h1 className="text-xl font-bold text-white">PERA AI Assistant</h1>
            <p className="text-gray-400 text-xs">Punjab Enforcement & Regulatory Authority</p>
          </div>

          {/* PDF Toggle */}
          {selectedPDF && (
            <button
              onClick={() => setShowPDF(!showPDF)}
              className={`ml-auto px-4 py-2 rounded-lg transition-all ${showPDF
                ? 'bg-amber-500 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
                }`}
            >
              📄 {showPDF ? 'Hide' : 'Show'} Document
            </button>
          )}
        </div>

        {/* Chat + PDF Area */}
        <div className="flex-1 flex overflow-hidden">
          {/* Chat Panel */}
          <div className={`${showPDF ? 'w-full lg:w-1/2' : 'w-full'} flex flex-col transition-all duration-300`}>
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-center animate-fade-in">
                  <div className="w-24 h-24 mb-6 rounded-full bg-gradient-to-br from-amber-500/20 to-orange-500/20 flex items-center justify-center">
                    <svg className="w-12 h-12 text-amber-400" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z" />
                    </svg>
                  </div>
                  <h2 className="text-2xl font-bold text-white mb-2">Assalam-o-Alaikum! 👋</h2>
                  <p className="text-gray-400 max-w-md">
                    Main PERA AI hun - aapka official digital assistant. Aap mujhse PERA se related koi bhi sawal pooch sakte hain.
                  </p>
                  <div className="mt-8 flex flex-wrap gap-2 justify-center">
                    {["CTO ki powers?", "PERA kya hai?", "Complaint kaise karein?"].map((q) => (
                      <button
                        key={q}
                        onClick={() => sendMessage(q)}
                        className="px-4 py-2 bg-white/5 border border-white/10 rounded-full text-gray-300 text-sm hover:bg-white/10 hover:border-amber-500/30 transition-all"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className="max-w-[80%]">
                    <div
                      className={`p-4 ${msg.role === "user"
                        ? "user-bubble text-white"
                        : "bot-bubble text-gray-200"
                        }`}
                    >
                      {msg.role === "assistant" ? (
                        <div
                          className="prose prose-invert prose-sm"
                          dangerouslySetInnerHTML={{ __html: formatAnswer(msg.content) }}
                        />
                      ) : (
                        <p className="whitespace-pre-wrap">{msg.content}</p>
                      )}
                    </div>

                    {/* References */}
                    {msg.role === "assistant" && msg.references && msg.references.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-2">
                        {msg.references.map((ref, j) => (
                          <button
                            key={j}
                            onClick={() => {
                              setSelectedPDF({
                                filename: ref.document || "unknown",
                                page: ref.page_start || 1,
                                title: `${ref.document || "Document"} - Page ${ref.page_start || 1}`,
                                snippet: ref.snippet || "",
                              });
                              setShowPDF(true);
                            }}
                            className="px-3 py-1.5 bg-amber-500/20 border border-amber-500/30 rounded-lg text-amber-400 text-xs hover:bg-amber-500/30 transition-all flex items-center gap-1"
                          >
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                              <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6z" />
                            </svg>
                            {ref.document} (Page {ref.page_start || 1})
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Typing indicator */}
              {loading && (
                <div className="flex justify-start">
                  <div className="bot-bubble p-4 flex gap-1.5">
                    <span className="typing-dot w-2 h-2 bg-amber-400 rounded-full"></span>
                    <span className="typing-dot w-2 h-2 bg-amber-400 rounded-full"></span>
                    <span className="typing-dot w-2 h-2 bg-amber-400 rounded-full"></span>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-white/10">
              <div className="flex gap-3">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && sendMessage(input)}
                  placeholder="Sawal likhein..."
                  className="flex-1 chat-input rounded-xl px-5 py-4 text-white placeholder-gray-400"
                />

                {/* Voice Button */}
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  className={`w-14 h-14 rounded-xl flex items-center justify-center transition-all ${isRecording
                    ? "bg-red-500 recording"
                    : "bg-amber-500 hover:bg-amber-400 voice-glow"
                    }`}
                >
                  {isRecording ? (
                    <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                  ) : (
                    <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                      <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
                    </svg>
                  )}
                </button>

                {/* Send Button */}
                <button
                  onClick={() => sendMessage(input)}
                  disabled={loading || !input.trim()}
                  className="w-14 h-14 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center hover:from-purple-400 hover:to-indigo-500 transition-all disabled:opacity-50 hover:scale-105 active:scale-95"
                >
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                </button>
              </div>
            </div>
          </div>

          {/* PDF Panel - Slide in/out */}
          <div className={`${showPDF ? 'w-full lg:w-1/2 opacity-100' : 'w-0 opacity-0'} transition-all duration-300 overflow-hidden absolute lg:relative right-0 h-full bg-gray-900 lg:bg-transparent z-20`}>
            <div className="w-full h-full p-4">
              <div className="glass rounded-2xl h-full flex flex-col overflow-hidden">
                {selectedPDF && (
                  <>
                    {/* PDF Header */}
                    <div className="p-4 border-b border-white/10 flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <h2 className="text-white font-semibold truncate">{selectedPDF.title}</h2>
                        {selectedPDF.snippet && (
                          <p className="text-amber-400 text-xs mt-1 line-clamp-2 italic">
                            &quot;{selectedPDF.snippet.slice(0, 100)}...&quot;
                          </p>
                        )}
                      </div>
                      <button
                        onClick={() => setShowPDF(false)}
                        className="ml-2 p-2 hover:bg-white/10 rounded-lg transition-all"
                      >
                        <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>

                    {/* PDF Embed */}
                    <div className="flex-1 bg-gray-900">
                      <iframe
                        src={`${API_URL}/pdf/${encodeURIComponent(selectedPDF.filename)}#page=${selectedPDF.page}`}
                        className="w-full h-full"
                        title="PDF Viewer"
                      />
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
