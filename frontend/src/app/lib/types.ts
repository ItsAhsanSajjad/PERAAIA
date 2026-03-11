/* ═══════════════════════════════════════════════════════════
   PERA AI — Domain & API Types
   ═══════════════════════════════════════════════════════════ */

// ─── Domain Models ───

export interface Reference {
  id?: number;
  document: string;
  page?: number | string;
  page_start?: number | string;
  page_end?: number | string;
  open_url?: string;
  snippet?: string;
  score?: number;
  chunk_index?: number;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  references?: Reference[];
  timestamp: number;
  /** When true the message represents a failed send attempt */
  failed?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;   // epoch ms
  updatedAt: number;   // epoch ms
}

// ─── API Request / Response ───

export interface AskRequest {
  question: string;
  conversation_history?: { role: string; content: string }[];
}

export interface AskResponse {
  answer: string;
  decision?: string;
  references: Reference[];
}

export interface TranscribeResponse {
  text: string;
  success: boolean;
}

// ─── API Error ───

export interface ApiError {
  type: "network" | "http" | "parse" | "unknown";
  status?: number;
  message: string;
}

// ─── Health Status ───

export type ConnectionStatus = "connecting" | "online" | "offline";

// ─── Voice Recorder State ───

export type VoiceState = "idle" | "recording" | "transcribing" | "error";

// ─── Toast ───

export interface ToastMessage {
  id: string;
  text: string;
  type: "info" | "success" | "error";
}

// ─── Storage ───

export const STORAGE_VERSION = 1;

export interface StorageEnvelope {
  version: number;
  sessions: ChatSession[];
  activeChatId: string | null;
}
