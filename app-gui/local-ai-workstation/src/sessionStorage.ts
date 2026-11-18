import { invoke } from "@tauri-apps/api/core";

// ============================================================================
// Chat Session Management & Persistence Service
// Handles conversation persistence, archiving, pinning, date grouping,
// search, and interrupted generation states.
// Backed by filesystem via Tauri with localStorage fallback.
// ============================================================================

export interface MessageStats {
  tokPerSec?: number;
  latencyMs?: number;
  tokens?: number;
  promptTokens?: number;
}

export interface ChatMessageItem {
  id: string;
  role: "user" | "assistant";
  text: string;
  thought?: string;
  stats?: MessageStats;
  interrupted?: boolean;
  timestamp: number;
}

export interface ConversationSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessageItem[];
  pinned?: boolean;
  archived?: boolean;
  associatedModel?: string | null;
  presetId?: string;
}

export type DateCategory = "pinned" | "today" | "yesterday" | "past7" | "past30" | "older";

export interface GroupedConversations {
  pinned: ConversationSession[];
  today: ConversationSession[];
  yesterday: ConversationSession[];
  past7: ConversationSession[];
  past30: ConversationSession[];
  older: ConversationSession[];
}

const STORAGE_KEY_SESSIONS = "local_ai_workstation_conversations_v3";
const STORAGE_KEY_ACTIVE_ID = "local_ai_workstation_active_session_v3";

const DEFAULT_INITIAL_SESSION: ConversationSession = {
  id: "conv-welcome-default",
  title: "Zero-copy mmap & inference architecture",
  createdAt: Date.now() - 1000 * 60 * 60,
  updatedAt: Date.now() - 1000 * 60 * 30,
  pinned: false,
  archived: false,
  associatedModel: "qwen2.5-3b-instruct-q2_k.gguf",
  messages: [
    {
      id: "msg-welcome-1",
      role: "user",
      text: "Can you explain how zero-copy mmap architecture helps in native local inference workstations?",
      timestamp: Date.now() - 1000 * 60 * 30,
    },
    {
      id: "msg-welcome-2",
      role: "assistant",
      text: "Zero-copy memory mapping (`mmap`) maps the GGUF model file on disk directly into the virtual address space of the process.\n\n### Key Benefits:\n1. **Near-Instant Load Times**: The OS does not copy gigabytes of tensor weights into heap RAM. The file pointers are mapped in milliseconds.\n2. **Demand Paging**: Model layers and KV matrices are paged into memory by the OS kernel only when touched during the forward pass.\n3. **Page Cache Sharing**: Multiple processes (or restarts) share the clean file-backed pages in the OS cache, saving system memory.",
      thought:
        "Examining memory subsystems: memmap2 crate + LlamaModel::load_from_file zero-copy characteristics.",
      stats: { tokPerSec: 72.4, latencyMs: 195, tokens: 142 },
      timestamp: Date.now() - 1000 * 60 * 29,
    },
  ],
};

/**
 * Load all saved conversations from persistent storage.
 */
export function loadSavedSessions(): ConversationSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_SESSIONS);
    if (!raw) return [DEFAULT_INITIAL_SESSION];
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.length > 0) {
      return parsed.map((item) => ({
        id: item.id || `conv-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
        title: item.title || "Untitled Session",
        createdAt: typeof item.createdAt === "number" ? item.createdAt : Date.now(),
        updatedAt: typeof item.updatedAt === "number" ? item.updatedAt : Date.now(),
        pinned: Boolean(item.pinned),
        archived: Boolean(item.archived),
        associatedModel: item.associatedModel || null,
        presetId: item.presetId || undefined,
        messages: Array.isArray(item.messages)
          ? item.messages.map((m: any, idx: number) => ({
              id: m.id || `msg-${idx}-${Date.now()}`,
              role: m.role === "assistant" ? "assistant" : "user",
              text: typeof m.text === "string" ? m.text : "",
              thought: typeof m.thought === "string" ? m.thought : undefined,
              stats: m.stats || undefined,
              interrupted: Boolean(m.interrupted),
              timestamp: typeof m.timestamp === "number" ? m.timestamp : Date.now(),
            }))
          : [],
      }));
    }
  } catch (err) {
    console.error("Failed to parse saved sessions from localStorage:", err);
  }
  return [DEFAULT_INITIAL_SESSION];
}

/**
 * Load last active session ID from persistent storage.
 */
export function loadActiveSessionId(availableSessions: ConversationSession[]): string {
  try {
    const savedId = localStorage.getItem(STORAGE_KEY_ACTIVE_ID);
    if (savedId && availableSessions.some((s) => s.id === savedId && !s.archived)) {
      return savedId;
    }
  } catch (err) {
    console.error("Failed to load active session ID:", err);
  }
  const firstActive = availableSessions.find((s) => !s.archived);
  return firstActive ? firstActive.id : availableSessions[0]?.id || "conv-welcome-default";
}

/**
 * Persist sessions to storage safely (both filesystem and localStorage).
 */
export function persistSessions(sessions: ConversationSession[], activeId?: string) {
  try {
    const json = JSON.stringify(sessions);
    localStorage.setItem(STORAGE_KEY_SESSIONS, json);
    if (activeId) {
      localStorage.setItem(STORAGE_KEY_ACTIVE_ID, activeId);
    }
    // Async background filesystem backup via Rust
    invoke("save_app_state", { key: "conversations_v3", data: json }).catch(() => {});
    if (activeId) {
      invoke("save_app_state", { key: "active_session_v3", data: activeId }).catch(() => {});
    }
  } catch (err) {
    console.error("Failed to persist sessions:", err);
  }
}

/**
 * Group active (non-archived) conversations into date buckets.
 */
export function groupSessionsByDate(sessions: ConversationSession[]): GroupedConversations {
  const now = Date.now();
  const ONE_DAY = 24 * 60 * 60 * 1000;
  const SEVEN_DAYS = 7 * ONE_DAY;
  const THIRTY_DAYS = 30 * ONE_DAY;

  const grouped: GroupedConversations = {
    pinned: [],
    today: [],
    yesterday: [],
    past7: [],
    past30: [],
    older: [],
  };

  sessions.forEach((s) => {
    if (s.archived) return;

    if (s.pinned) {
      grouped.pinned.push(s);
      return;
    }

    const elapsed = now - s.updatedAt;
    if (elapsed < ONE_DAY) {
      grouped.today.push(s);
    } else if (elapsed < 2 * ONE_DAY) {
      grouped.yesterday.push(s);
    } else if (elapsed < SEVEN_DAYS) {
      grouped.past7.push(s);
    } else if (elapsed < THIRTY_DAYS) {
      grouped.past30.push(s);
    } else {
      grouped.older.push(s);
    }
  });

  return grouped;
}

/**
 * Filter sessions by search query across session title and message text.
 */
export function filterSessionsBySearch(
  sessions: ConversationSession[],
  query: string,
  includeArchived: boolean = false,
): ConversationSession[] {
  const cleanQ = query.trim().toLowerCase();
  if (!cleanQ) {
    return sessions.filter((s) => (includeArchived ? Boolean(s.archived) : !s.archived));
  }

  return sessions.filter((s) => {
    if (includeArchived && !s.archived) return false;
    if (!includeArchived && s.archived) return false;

    if (s.title.toLowerCase().includes(cleanQ)) return true;
    return s.messages.some((m) => m.text.toLowerCase().includes(cleanQ));
  });
}

/**
 * Generate a clean conversation title from the first prompt text.
 */
export function generateSessionTitle(prompt: string): string {
  const clean = prompt.trim().replace(/\s+/g, " ");
  if (clean.length <= 48) return clean || "New Conversation";
  return clean.substring(0, 48) + "...";
}

/**
 * Approximate context token count for a message list (~3.8 characters per token average).
 */
export function estimateContextTokens(messages: ChatMessageItem[]): number {
  let charCount = 0;
  for (const m of messages) {
    charCount += m.text.length + (m.thought ? m.thought.length : 0) + 16;
  }
  return Math.ceil(charCount / 3.8);
}
