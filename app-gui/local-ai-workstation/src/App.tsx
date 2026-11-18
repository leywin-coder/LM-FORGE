import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Brain,
  BrainCircuit,
  Check,
  ChevronDown,
  ChevronRight,
  Circle,
  CloudDownload,
  Copy,
  Cpu,
  Database,
  Download,
  FileCode,
  FolderOpen,
  HardDrive,
  Info,
  Library,
  Loader2,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  Search,
  Send,
  Server,
  Settings,
  SlidersHorizontal,
  Square,
  Terminal,
  Trash2,
  X,
  Zap,
  Upload,
  Pin,
  Archive,
  RotateCcw,
  Activity,
  Anvil,
  Bookmark,
  Paperclip,
  FileText,
  Eye,
  Sliders,
} from "lucide-react";
import { exportConfigToJson, triggerJsonDownload } from "./configParser";
import {
  ConversationSession,
  ChatMessageItem,
  loadSavedSessions,
  loadActiveSessionId,
  persistSessions,
  groupSessionsByDate,
  filterSessionsBySearch,
  generateSessionTitle,
  estimateContextTokens,
} from "./sessionStorage";
import {
  SystemMetrics,
  MetricHistoryPoint,
  InferenceThroughputMetrics,
  INITIAL_INFERENCE_METRICS,
  fetchSystemMetrics,
  calculateThroughput,
} from "./telemetryService";
import { ConfigImportModal } from "./ConfigImportModal";
import { TelemetryModal } from "./TelemetryModal";
import {
  WorkstationSettings,
  InferenceSettings,
  loadWorkstationSettings,
  saveWorkstationSettings,
} from "./settingsManager";
import { SettingsView } from "./SettingsView";

// ==========================================
// Types
// ==========================================
type View = "chat" | "models" | "server" | "discover" | "presets" | "settings";
type MessageRole = "user" | "assistant";

export interface ChatAttachment {
  id: string;
  name: string;
  content: string;
  size: number;
  tokens: number;
}

export interface HfRepoFileItem {
  path: string;
  size?: number;
  type: string;
}

export interface DownloadProgressInfo {
  model_id: string;
  filename: string;
  downloaded_bytes: number;
  total_bytes: number;
  percent: number;
  speed_bytes_sec: number;
  status: string;
  error?: string;
}

export interface LocalModelDetailedItem {
  id: string;
  filename: string;
  display_name: string;
  path: string;
  size: number;
  quant: string;
  is_split: boolean;
  split_part?: number;
  split_total?: number;
  base_name: string;
}
type DownloadState = "idle" | "downloading" | "done" | "failed";
type ModelLoadingState = "not_loaded" | "loading" | "loaded" | "unloading" | "failed";

export type SettingsMode = "simple" | "advanced";
export type ResponseStyle = "precise" | "balanced" | "creative" | "deterministic";
export type ResponseLength = "concise" | "standard" | "detailed";

interface ModelInfo {
  id: string;
  path: string;
  size: number;
}

interface Message {
  role: MessageRole;
  text: string;
  thought?: string;
  stats?: {
    tokPerSec?: number;
    tokens?: number;
    latencyMs?: number;
  };
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
}

export interface GenerationSettings {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  maxTokens: number;
  stopSequences: string[];
  repetitionPenalty: number;
  frequencyPenalty: number;
  presencePenalty: number;
  seed: number | null;
  systemPrompt: string;
}

export interface Preset {
  id: string;
  name: string;
  category: "Coding" | "General" | "Reasoning" | "Creative" | "JSON";
  description: string;
  style: ResponseStyle;
  length: ResponseLength;
  settings: GenerationSettings;
}

interface HfModelItem {
  id: string;
  downloads?: number;
  likes?: number;
  pipeline_tag?: string;
  tags?: string[];
}

interface RequestLog {
  id: string;
  time: string;
  method: "POST" | "GET";
  path: string;
  status: string;
  latency: string;
  rate: string;
  requestJson?: string;
  responseStream?: string;
}

// ==========================================
// Intelligent Presets Catalog
// ==========================================
export const PRESETS: Preset[] = [
  {
    id: "general-balanced",
    name: "General Conversational",
    category: "General",
    description: "Versatile, natural conversation with balanced coherence and creativity.",
    style: "balanced",
    length: "standard",
    settings: {
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      maxTokens: 1024,
      stopSequences: [],
      repetitionPenalty: 1.1,
      frequencyPenalty: 0.0,
      presencePenalty: 0.0,
      seed: null,
      systemPrompt:
        "You are a helpful, conversational AI assistant. Provide thoughtful, well-reasoned, and articulate answers.",
    },
  },
  {
    id: "coding-precise",
    name: "Code & Engineering",
    category: "Coding",
    description:
      "High fidelity, low hallucination, strict adherence to syntax and language typing.",
    style: "precise",
    length: "detailed",
    settings: {
      temperature: 0.2,
      topP: 0.85,
      topK: 30,
      minP: 0.08,
      maxTokens: 2048,
      stopSequences: [],
      repetitionPenalty: 1.15,
      frequencyPenalty: 0.0,
      presencePenalty: 0.0,
      seed: null,
      systemPrompt:
        "You are a senior software architect. Provide clean, correct, and production-ready code with concise technical explanations.",
    },
  },
  {
    id: "factual-analysis",
    name: "Factual & Research Analysis",
    category: "Reasoning",
    description:
      "Focused and predictable responses for documentation, data, and technical specifications.",
    style: "precise",
    length: "standard",
    settings: {
      temperature: 0.1,
      topP: 0.8,
      topK: 20,
      minP: 0.1,
      maxTokens: 1024,
      stopSequences: [],
      repetitionPenalty: 1.15,
      frequencyPenalty: 0.0,
      presencePenalty: 0.0,
      seed: null,
      systemPrompt:
        "You are a technical analyst. State facts precisely, cite constraints, acknowledge uncertainties, and avoid speculative extrapolation.",
    },
  },
  {
    id: "creative-writing",
    name: "Creative & Brainstorming",
    category: "Creative",
    description: "Expressive vocabulary, varied sentence structures, and open-ended exploration.",
    style: "creative",
    length: "detailed",
    settings: {
      temperature: 1.05,
      topP: 0.95,
      topK: 65,
      minP: 0.03,
      maxTokens: 2048,
      stopSequences: [],
      repetitionPenalty: 1.05,
      frequencyPenalty: 0.1,
      presencePenalty: 0.1,
      seed: null,
      systemPrompt:
        "You are an imaginative creative writer. Produce rich, descriptive, and engaging prose with high stylistic variety.",
    },
  },
  {
    id: "strict-json",
    name: "Strict JSON & Extraction",
    category: "JSON",
    description:
      "Zero-temperature greedy decoding for reliable JSON formatting and schema conformance.",
    style: "deterministic",
    length: "standard",
    settings: {
      temperature: 0.0,
      topP: 0.5,
      topK: 1,
      minP: 0.2,
      maxTokens: 1024,
      stopSequences: [],
      repetitionPenalty: 1.05,
      frequencyPenalty: 0.0,
      presencePenalty: 0.0,
      seed: null,
      systemPrompt:
        "You are a strict data formatting engine. Output valid JSON only, without any introductory or concluding markdown or text.",
    },
  },
  {
    id: "concise-summary",
    name: "Concise Executive Summary",
    category: "General",
    description: "Brief, high-density answers without conversational preamble.",
    style: "precise",
    length: "concise",
    settings: {
      temperature: 0.3,
      topP: 0.85,
      topK: 30,
      minP: 0.08,
      maxTokens: 256,
      stopSequences: [],
      repetitionPenalty: 1.1,
      frequencyPenalty: 0.0,
      presencePenalty: 0.0,
      seed: null,
      systemPrompt:
        "You are an executive assistant. Summarize information into terse, actionable bullet points without preamble.",
    },
  },
];

const defaultSettings = PRESETS[0].settings;

// ==========================================
// Helpers
// ==========================================
function getDateGroup(updatedAt: number): string {
  const now = new Date();
  const updated = new Date(updatedAt);
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const weekAgo = new Date(today);
  weekAgo.setDate(weekAgo.getDate() - 7);
  if (updated >= today) return "Today";
  if (updated >= yesterday) return "Yesterday";
  if (updated >= weekAgo) return "Previous 7 Days";
  return "Older";
}

function groupConversations(convs: Conversation[]) {
  const labels = ["Today", "Yesterday", "Previous 7 Days", "Older"];
  const groups: { label: string; items: Conversation[] }[] = labels.map((l) => ({
    label: l,
    items: [],
  }));

  for (const conv of convs) {
    const label = getDateGroup(conv.updatedAt);
    const group = groups.find((g) => g.label === label);
    if (group) group.items.push(conv);
  }

  for (const g of groups) {
    g.items.sort((a, b) => b.updatedAt - a.updatedAt);
  }
  return groups.filter((g) => g.items.length > 0);
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(1)} MB`;
}

function parseModelDetails(fileName: string) {
  const base = fileName.replace(/\.gguf$/i, "").replace(/\.ggml$/i, "");
  const quantMatch = fileName.match(/(q\d+_[a-z0-9_]+|f16|f32|bf16)/i);
  const quant = quantMatch ? quantMatch[1].toUpperCase() : "GGUF";
  const cleanName = base.replace(new RegExp(`[-_]${quant}`, "i"), "").replace(/[-_]/g, " ");
  return { cleanName, quant };
}

// Initial request logs for server view
const initialServerLogs: RequestLog[] = [
  {
    id: "log-1",
    time: "08:14:22.104",
    method: "POST",
    path: "/v1/chat/completions",
    status: "200 OK",
    latency: "214ms",
    rate: "68.4 tok/s",
    requestJson: JSON.stringify(
      {
        model: "default",
        messages: [
          { role: "user", content: "Explain how zero-copy mmap works for local LLM loading." },
        ],
        temperature: 0.7,
        stream: true,
      },
      null,
      2,
    ),
    responseStream: `data: {"choices":[{"delta":{"content":"Memory-mapped (mmap) files allow"}}]}\ndata: {"choices":[{"delta":{"content":" the operating system to map model weights directly."}}]}\ndata: [DONE]`,
  },
];

// Fallback curated Hugging Face models if offline
const curatedHfModels: HfModelItem[] = [
  {
    id: "Qwen/Qwen2.5-3B-Instruct-GGUF",
    downloads: 240000,
    likes: 1840,
    pipeline_tag: "text-generation",
    tags: ["gguf", "qwen2.5", "3b", "chat"],
  },
  {
    id: "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
    downloads: 380000,
    likes: 2950,
    pipeline_tag: "text-generation",
    tags: ["gguf", "qwen2.5-coder", "7b", "code"],
  },
  {
    id: "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    downloads: 510000,
    likes: 4200,
    pipeline_tag: "text-generation",
    tags: ["gguf", "deepseek-r1", "reasoning"],
  },
  {
    id: "meta-llama/Llama-3.2-3B-Instruct-GGUF",
    downloads: 620000,
    likes: 5400,
    pipeline_tag: "text-generation",
    tags: ["gguf", "llama-3.2", "compact"],
  },
];

// ==========================================
// Main Application Component
// ==========================================
export default function App() {
  const [view, setView] = useState<View>("chat");

  // 1. Truthful Backend & Model State (Single Source of Truth)
  const [backendConnected, setBackendConnected] = useState<boolean>(false);
  const [modelStatus, setModelStatus] = useState<ModelLoadingState>("not_loaded");
  const [loadedModel, setLoadedModel] = useState<string | null>(null);
  const [loadingError, setLoadingError] = useState<string | null>(null);
  const [activeOperationModel, setActiveOperationModel] = useState<string | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [serverOn, setServerOn] = useState<boolean>(false);

  // 2. Generation Settings & Preset State
  const [settingsMode, setSettingsMode] = useState<SettingsMode>("simple");
  const [activePresetId, setActivePresetId] = useState<string>("general-balanced");
  const [settings, setSettings] = useState<GenerationSettings>(defaultSettings);
  const [simpleStyle, setSimpleStyle] = useState<ResponseStyle>("balanced");
  const [simpleLength, setSimpleLength] = useState<ResponseLength>("standard");
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try {
      const raw = localStorage.getItem("local_ai_workstation_custom_presets_v1");
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  });
  const allPresets = useMemo(() => [...PRESETS, ...customPresets], [customPresets]);

  // 3. Navigation Modals & Active Downloads
  const [appSettings, setAppSettings] = useState<WorkstationSettings>(() => loadWorkstationSettings());
  const [downloadsOpen, setDownloadsOpen] = useState(false);
  const [configImportModalOpen, setConfigImportModalOpen] = useState(false);
  const [telemetryModalOpen, setTelemetryModalOpen] = useState(false);
  const [activeDownload, setActiveDownload] = useState<{
    modelId: string;
    filename: string;
    status: DownloadState;
    downloadedBytes?: number;
    totalBytes?: number;
    percent?: number;
    speedBytesSec?: number;
    error?: string;
  } | null>(null);

  // Listen to real backend streaming download_progress events
  useEffect(() => {
    let unlistenFn: (() => void) | null = null;
    listen<DownloadProgressInfo>("download_progress", (event) => {
      const p = event.payload;
      setActiveDownload({
        modelId: p.model_id,
        filename: p.filename,
        status: p.status === "done" ? "done" : p.status === "cancelled" || p.status === "failed" ? "failed" : "downloading",
        downloadedBytes: Number(p.downloaded_bytes),
        totalBytes: Number(p.total_bytes),
        percent: p.percent,
        speedBytesSec: p.speed_bytes_sec,
        error: p.error,
      });

      if (p.status === "done") {
        refreshBackendState();
      }
    }).then((fn) => {
      unlistenFn = fn;
    }).catch((e) => console.warn("Failed to listen to download_progress:", e));

    return () => {
      if (unlistenFn) unlistenFn();
    };
  }, []);

  // 4. Real System Resource Monitoring (Part 13)
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [metricHistory, setMetricHistory] = useState<MetricHistoryPoint[]>([]);
  const [inferenceMetrics, setInferenceMetrics] =
    useState<InferenceThroughputMetrics>(INITIAL_INFERENCE_METRICS);

  const refreshTelemetry = async () => {
    const m = await fetchSystemMetrics();
    if (m) {
      setMetrics(m);
      const nowStr = new Date(m.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
      const ramPercent = m.total_ram > 0 ? Math.round((m.used_ram / m.total_ram) * 100) : 0;
      const pt: MetricHistoryPoint = {
        time: nowStr,
        timestamp: m.timestamp,
        cpu: m.cpu_usage,
        ramUsagePercent: ramPercent,
        usedRamGb: Math.round((m.used_ram / (1024 * 1024 * 1024)) * 10) / 10,
        gpuUsage: m.gpu_utilization,
      };
      setMetricHistory((prev) => [...prev.slice(-29), pt]);
    }
  };

  useEffect(() => {
    refreshTelemetry();
    const timer = setInterval(() => {
      refreshTelemetry();
    }, 2500);
    return () => clearInterval(timer);
  }, []);

  // 5. Persistent Chat Sessions (Part 12)
  const [conversations, setConversations] = useState<ConversationSession[]>(() =>
    loadSavedSessions(),
  );
  const [activeConversationId, setActiveConversationId] = useState<string>(() =>
    loadActiveSessionId(loadSavedSessions()),
  );

  useEffect(() => {
    persistSessions(conversations, activeConversationId);
  }, [conversations, activeConversationId]);

  // ----------------------------------------------------
  // Sync state strictly with inference backend on mount
  // ----------------------------------------------------
  const refreshBackendState = async () => {
    try {
      const confirmedLoaded = await invoke<string | null>("get_loaded_model");
      setBackendConnected(true);

      if (confirmedLoaded) {
        setLoadedModel(confirmedLoaded);
        setModelStatus("loaded");
        setLoadingError(null);
      } else {
        setLoadedModel(null);
        setModelStatus("not_loaded");
      }
    } catch (err) {
      console.warn("Backend not reachable for loaded model:", err);
      setBackendConnected(false);
      setLoadedModel(null);
      setModelStatus("not_loaded");
    }

    try {
      const list = await invoke<ModelInfo[]>("get_models");
      setModels(list || []);
    } catch (err) {
      console.warn("Backend not reachable for model list:", err);
    }

    try {
      const sOn = await invoke<boolean>("check_server_status");
      setServerOn(sOn);
    } catch {
      setServerOn(false);
    }
  };

  useEffect(() => {
    refreshBackendState();
  }, []);

  // ----------------------------------------------------
  // Preset Selection & Style Synchronization
  // ----------------------------------------------------
  const handleApplyPreset = (presetId: string) => {
    const target = allPresets.find((p) => p.id === presetId);
    if (!target) return;
    setActivePresetId(presetId);
    setSettings({ ...target.settings });
    setSimpleStyle(target.style);
    setSimpleLength(target.length);
  };

  const handleSaveCustomPreset = (
    presetName: string,
    category: string,
    newSettings: GenerationSettings,
  ) => {
    const newPreset: Preset = {
      id: `custom-${Date.now()}`,
      name: presetName,
      category: (["Coding", "General", "Reasoning", "Creative", "JSON"].includes(category)
        ? category
        : "General") as any,
      description: `Custom preset created on ${new Date().toLocaleDateString()}`,
      style: "balanced",
      length: "standard",
      settings: { ...newSettings },
    };
    setCustomPresets((prev) => {
      const updated = [newPreset, ...prev];
      localStorage.setItem("local_ai_workstation_custom_presets_v1", JSON.stringify(updated));
      return updated;
    });
    setActivePresetId(newPreset.id);
  };

  const handleSimpleStyleChange = (style: ResponseStyle) => {
    setSimpleStyle(style);
    let newSettings = { ...settings };
    if (style === "precise") {
      newSettings = {
        ...newSettings,
        temperature: 0.2,
        topP: 0.85,
        topK: 30,
        minP: 0.08,
        repetitionPenalty: 1.15,
      };
    } else if (style === "balanced") {
      newSettings = {
        ...newSettings,
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        minP: 0.05,
        repetitionPenalty: 1.1,
      };
    } else if (style === "creative") {
      newSettings = {
        ...newSettings,
        temperature: 1.05,
        topP: 0.95,
        topK: 65,
        minP: 0.03,
        repetitionPenalty: 1.05,
      };
    } else if (style === "deterministic") {
      newSettings = {
        ...newSettings,
        temperature: 0.0,
        topP: 0.5,
        topK: 1,
        minP: 0.2,
        repetitionPenalty: 1.05,
      };
    }
    setSettings(newSettings);
  };

  const handleSimpleLengthChange = (length: ResponseLength) => {
    setSimpleLength(length);
    let maxTok = 1024;
    if (length === "concise") maxTok = 256;
    else if (length === "standard") maxTok = 1024;
    else if (length === "detailed") maxTok = 2048;
    setSettings((prev) => ({ ...prev, maxTokens: maxTok }));
  };

  // ----------------------------------------------------
  // Explicit Model Load & Unload Handlers
  // ----------------------------------------------------
  const handleLoadModel = async (modelPath: string) => {
    if (modelStatus === "loading" || modelStatus === "unloading") return;

    const fileName = modelPath.split("/").pop() || modelPath;
    setModelStatus("loading");
    setActiveOperationModel(fileName);
    setLoadingError(null);

    try {
      const confirmedName = await invoke<string>("load_model", { path: modelPath });
      setLoadedModel(confirmedName || fileName);
      setModelStatus("loaded");
      setLoadingError(null);
      setActiveOperationModel(null);
      setServerOn(true);
    } catch (err: unknown) {
      const errMsg =
        typeof err === "string"
          ? err
          : (err as Error)?.message || "Failed to load model into memory.";
      setModelStatus("failed");
      setLoadingError(errMsg);
      setLoadedModel(null);
      setActiveOperationModel(null);
    }
  };

  const handleUnloadModel = async () => {
    if (modelStatus === "loading" || modelStatus === "unloading") return;

    setModelStatus("unloading");
    setLoadingError(null);

    try {
      await invoke("unload_model");
      setLoadedModel(null);
      setModelStatus("not_loaded");
      setActiveOperationModel(null);
      setServerOn(false);
    } catch (err: unknown) {
      const errMsg =
        typeof err === "string"
          ? err
          : (err as Error)?.message || "Failed to unload model from memory.";
      setLoadingError(errMsg);
      try {
        const confirmed = await invoke<string | null>("get_loaded_model");
        if (confirmed) {
          setLoadedModel(confirmed);
          setModelStatus("loaded");
        } else {
          setLoadedModel(null);
          setModelStatus("not_loaded");
        }
      } catch {
        setModelStatus("failed");
      }
    }
  };

  // ----------------------------------------------------
  // Hugging Face Real Download Execution
  // ----------------------------------------------------
  const handleDownloadHfModel = async (modelId: string, filename: string) => {
    setActiveDownload({
      modelId,
      filename,
      status: "downloading",
      downloadedBytes: 0,
      totalBytes: 0,
      percent: 0,
      speedBytesSec: 0,
    });
    setDownloadsOpen(true);

    try {
      await invoke<string>("download_model", {
        modelId,
        filename,
        token: appSettings.downloads.hfToken || null,
      });

      setActiveDownload({
        modelId,
        filename,
        status: "done",
        percent: 100,
      });

      // Rescan models directory to include newly downloaded GGUF file
      await refreshBackendState();
    } catch (err: unknown) {
      const errMsg =
        typeof err === "string" ? err : (err as Error)?.message || "Failed to download model file.";
      setActiveDownload({
        modelId,
        filename,
        status: "failed",
        error: errMsg,
      });
    }
  };

  const handleCancelDownload = async (filename: string) => {
    try {
      await invoke("cancel_download", { filename });
      setActiveDownload((prev) => (prev ? { ...prev, status: "failed", error: "Download cancelled by user" } : null));
    } catch (err) {
      console.warn("Cancel download failed:", err);
    }
  };

  // ----------------------------------------------------
  // Conversation Management
  // ----------------------------------------------------
  const activeConversation = conversations.find((c) => c.id === activeConversationId) ||
    conversations[0] || {
      id: "conv-fallback",
      title: "New conversation",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };

  const handleNewConversation = () => {
    const id = "conv-" + Math.random().toString(36).substring(2, 9);
    const newConv: ConversationSession = {
      id,
      title: "New conversation",
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
      pinned: false,
      archived: false,
      associatedModel: loadedModel,
    };
    setConversations((prev) => [newConv, ...prev]);
    setActiveConversationId(id);
  };

  const handleRenameConversation = (id: string, newTitle: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title: newTitle.trim(), updatedAt: Date.now() } : c)),
    );
  };

  const handleDeleteConversation = (id: string) => {
    if (!window.confirm("Permanently delete this conversation?")) return;
    setConversations((prev) => {
      const filtered = prev.filter((c) => c.id !== id);
      if (filtered.length === 0) {
        const fresh: ConversationSession = {
          id: "conv-" + Math.random().toString(36).substring(2, 9),
          title: "New conversation",
          messages: [],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        setActiveConversationId(fresh.id);
        return [fresh];
      }
      if (activeConversationId === id) {
        setActiveConversationId(filtered[0].id);
      }
      return filtered;
    });
  };

  const handlePinConversation = (id: string) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, pinned: !c.pinned } : c)));
  };

  const handleArchiveConversation = (id: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, archived: !c.archived } : c)),
    );
  };

  const handleDuplicateConversation = (id: string) => {
    const target = conversations.find((c) => c.id === id);
    if (!target) return;
    const dup: ConversationSession = {
      ...target,
      id: "conv-" + Math.random().toString(36).substring(2, 9),
      title: `${target.title} (Copy)`,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      pinned: false,
      archived: false,
      messages: target.messages.map((m) => ({
        ...m,
        id: "msg-" + Math.random().toString(36).substring(2, 9),
      })),
    };
    setConversations((prev) => [dup, ...prev]);
    setActiveConversationId(dup.id);
  };

  const handleClearConversation = (id: string) => {
    if (window.confirm("Clear all messages in this conversation?")) {
      setConversations((prev) =>
        prev.map((c) => (c.id === id ? { ...c, messages: [], updatedAt: Date.now() } : c)),
      );
    }
  };

  return (
    <div className="app-shell flex h-screen w-screen overflow-hidden bg-[#121316] text-[#f1f2f4] select-none font-sans text-xs">
      {/* 1. Activity Rail (Fixed 48px navigation) */}
      <ActivityRail
        view={view}
        setView={setView}
        activeDownloads={activeDownload?.status === "downloading" ? 1 : 0}
        onOpenDownloads={() => setDownloadsOpen((v) => !v)}
        onOpenSettings={() => setView("settings")}
      />

      {/* 2. Main Workstation Area */}
      <div className="flex-1 flex flex-col h-full min-w-0 min-h-0 overflow-hidden relative">
        <main className="flex-1 min-h-0 min-w-0 overflow-hidden relative flex flex-col">
          {view === "chat" && (
            <ChatView
              conversation={activeConversation}
              conversations={conversations}
              activeId={activeConversationId}
              onSelectConversation={setActiveConversationId}
              onNewConversation={handleNewConversation}
              onRenameConversation={handleRenameConversation}
              onDeleteConversation={handleDeleteConversation}
              onPinConversation={handlePinConversation}
              onArchiveConversation={handleArchiveConversation}
              onDuplicateConversation={handleDuplicateConversation}
              onClearConversation={handleClearConversation}
              settingsMode={settingsMode}
              setSettingsMode={setSettingsMode}
              activePresetId={activePresetId}
              allPresets={allPresets}
              onApplyPreset={handleApplyPreset}
              onOpenImportModal={() => setConfigImportModalOpen(true)}
              onExportConfig={() => {
                const cur = allPresets.find((p) => p.id === activePresetId);
                const name = cur ? cur.name : "Custom Generation Config";
                const json = exportConfigToJson(name, settings, cur?.description, loadedModel);
                triggerJsonDownload(`${name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.json`, json);
              }}
              simpleStyle={simpleStyle}
              onSimpleStyleChange={handleSimpleStyleChange}
              simpleLength={simpleLength}
              onSimpleLengthChange={handleSimpleLengthChange}
              settings={settings}
              onUpdateSettings={setSettings}
              backendConnected={backendConnected}
              modelStatus={modelStatus}
              loadedModel={loadedModel}
              loadingError={loadingError}
              activeOperationModel={activeOperationModel}
              onDismissError={() => setLoadingError(null)}
              models={models}
              onLoadModel={handleLoadModel}
              onUnloadModel={handleUnloadModel}
              onSwitchToModels={() => setView("models")}
              setConversations={setConversations}
              inferenceMetrics={inferenceMetrics}
              setInferenceMetrics={setInferenceMetrics}
              inferenceSettings={appSettings.inference}
            />
          )}

          {view === "models" && (
            <ModelsView
              models={models}
              backendConnected={backendConnected}
              modelStatus={modelStatus}
              loadedModel={loadedModel}
              loadingError={loadingError}
              activeOperationModel={activeOperationModel}
              onDismissError={() => setLoadingError(null)}
              onLoadModel={handleLoadModel}
              onUnloadModel={handleUnloadModel}
              onRefresh={refreshBackendState}
              onOpenInChat={(modelId) => {
                const target = models.find((m) => m.id === modelId || m.path.endsWith(modelId));
                if (target && loadedModel !== target.id) {
                  handleLoadModel(target.path);
                }
                setView("chat");
              }}
              onNavigateDiscover={() => setView("discover")}
            />
          )}

          {view === "server" && (
            <ServerView
              serverOn={serverOn}
              setServerOn={setServerOn}
              backendConnected={backendConnected}
              modelStatus={modelStatus}
              loadedModel={loadedModel}
              onUnloadModel={handleUnloadModel}
            />
          )}

          {view === "discover" && (
            <DiscoverView
              onDownloadModel={handleDownloadHfModel}
              activeDownload={activeDownload}
              onSwitchToModels={() => setView("models")}
              hfToken={appSettings.downloads.hfToken}
            />
          )}

          {view === "presets" && (
            <div className="flex-1 flex flex-col h-full overflow-y-auto p-6 space-y-4 custom-scrollbar bg-[#121316] max-w-5xl mx-auto w-full">
              <div className="flex items-center justify-between pb-3 border-b border-[#282b34]">
                <div>
                  <h1 className="text-base font-bold text-[#f1f2f4]">Intelligent Preset Library</h1>
                  <p className="text-xs text-[#8c919e]">
                    Configured parameter sets for coding, reasoning, JSON extraction, and creative generation.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setConfigImportModalOpen(true)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-[#20232a] hover:bg-[#282c35] border border-[#282b34] text-xs text-[#a0a5b2] hover:text-white transition-colors"
                  >
                    <Upload size={12} className="text-blue-400" />
                    <span>Import JSON Preset</span>
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                {allPresets.map((p) => {
                  const isActive = activePresetId === p.id;
                  return (
                    <div
                      key={p.id}
                      className={`p-4 rounded border transition-colors flex flex-col justify-between space-y-3 ${
                        isActive
                          ? "bg-[#18231e] border-emerald-600/60"
                          : "bg-[#181a20] border-[#282b34] hover:border-[#383d4a]"
                      }`}
                    >
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between">
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-[#20232a] text-[#8c919e] uppercase">
                            {p.category}
                          </span>
                          {isActive && (
                            <span className="flex items-center gap-1 text-[11px] text-emerald-400 font-semibold">
                              <Check size={12} />
                              <span>Active in Chat</span>
                            </span>
                          )}
                        </div>
                        <h3 className="font-semibold text-[#f1f2f4] text-sm">{p.name}</h3>
                        <p className="text-xs text-[#8c919e] leading-relaxed">{p.description}</p>
                      </div>

                      <div className="pt-2 border-t border-[#242730] flex items-center justify-between">
                        <div className="font-mono text-[11px] text-[#6c707d] flex items-center gap-2">
                          <span>Temp: {p.settings.temperature}</span>
                          <span>·</span>
                          <span>Top-P: {p.settings.topP}</span>
                          <span>·</span>
                          <span>Max: {p.settings.maxTokens}</span>
                        </div>
                        <button
                          onClick={() => {
                            handleApplyPreset(p.id);
                            setView("chat");
                          }}
                          className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                            isActive
                              ? "bg-emerald-600 hover:bg-emerald-500 text-white"
                              : "bg-[#2563eb] hover:bg-[#1d4ed8] text-white"
                          }`}
                        >
                          {isActive ? "Applied" : "Apply & Chat"}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {view === "settings" && (
            <SettingsView
              settings={appSettings}
              onUpdateSettings={(newSettings) => setAppSettings(newSettings)}
              onClose={() => setView("chat")}
            />
          )}
        </main>

        {/* 3. Pinned Status Bar (Real Telemetry) */}
        <TelemetryBar
          backendConnected={backendConnected}
          modelStatus={modelStatus}
          loadedModel={loadedModel}
          activeOperationModel={activeOperationModel}
          serverOn={serverOn}
          metrics={metrics}
          inferenceMetrics={inferenceMetrics}
          onServerClick={() => {
            setServerOn((v) => !v);
            setView("server");
          }}
          onModelClick={() => setView("models")}
          onTelemetryClick={() => setTelemetryModalOpen(true)}
        />
      </div>

      {/* Downloads Popover */}
      {downloadsOpen && (
        <DownloadsPopover
          activeDownload={activeDownload}
          onClose={() => setDownloadsOpen(false)}
          onLoadDownloaded={(filename) => {
            const target = models.find((m) => m.id === filename || m.path.endsWith(filename));
            if (target) {
              handleLoadModel(target.path);
              setView("chat");
              setDownloadsOpen(false);
            }
          }}
          onCancelDownload={handleCancelDownload}
        />
      )}

      {/* JSON Generation Configuration Import Modal (Part 11) */}
      {configImportModalOpen && (
        <ConfigImportModal
          isOpen={configImportModalOpen}
          onClose={() => setConfigImportModalOpen(false)}
          currentSettings={settings}
          onApplyConfig={(newSettings) => {
            setSettings(newSettings);
          }}
          onSaveAsPreset={handleSaveCustomPreset}
        />
      )}

      {/* Real System Resource Monitoring Modal (Part 13) */}
      {telemetryModalOpen && (
        <TelemetryModal
          isOpen={telemetryModalOpen}
          onClose={() => setTelemetryModalOpen(false)}
          metrics={metrics}
          history={metricHistory}
          inferenceMetrics={inferenceMetrics}
          loadedModel={loadedModel}
          onRefresh={refreshTelemetry}
        />
      )}
    </div>
  );
}

// ==========================================
// Component: Activity Rail
// ==========================================
function ActivityRail({
  view,
  setView,
  activeDownloads,
  onOpenDownloads,
  onOpenSettings,
}: {
  view: View;
  setView: (v: View) => void;
  activeDownloads: number;
  onOpenDownloads: () => void;
  onOpenSettings: () => void;
}) {
  const items = [
    { id: "chat" as const, icon: MessageSquare, label: "Chat" },
    { id: "models" as const, icon: Library, label: "My Models" },
    { id: "presets" as const, icon: Sliders, label: "Intelligent Presets" },
    { id: "server" as const, icon: Server, label: "Local Server" },
    { id: "discover" as const, icon: Download, label: "Hugging Face Discovery" },
  ];

  return (
    <nav className="w-[48px] h-full flex flex-col items-center py-2.5 bg-[#16181d] border-r border-[#282b34] shrink-0 select-none z-30">
      <div
        className="w-8 h-8 rounded bg-[#2563eb] flex items-center justify-center text-white mb-4 cursor-pointer hover:bg-[#1d4ed8] transition-colors"
        onClick={() => setView("chat")}
        title="LM Forge"
      >
        <Anvil size={16} strokeWidth={2.4} />
      </div>

      <div className="flex flex-col gap-1 w-full px-1.5">
        {items.map(({ id, icon: Icon, label }) => {
          const isActive = view === id;
          return (
            <button
              key={id}
              onClick={() => setView(id)}
              className={`group relative w-9 h-9 mx-auto rounded flex items-center justify-center transition-colors ${
                isActive
                  ? "bg-[#2563eb] text-white"
                  : "text-[#9da1ad] hover:text-[#f1f2f4] hover:bg-[#20232a]"
              }`}
              title={label}
            >
              <Icon size={17} />
            </button>
          );
        })}
      </div>

      <div className="flex-1" />

      <div className="flex flex-col gap-1 w-full px-1.5">
        <button
          onClick={onOpenDownloads}
          className="relative w-9 h-9 mx-auto rounded flex items-center justify-center text-[#9da1ad] hover:text-[#f1f2f4] hover:bg-[#20232a] transition-colors"
          title="Model Downloads"
        >
          <CloudDownload size={17} />
          {activeDownloads > 0 && (
            <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
          )}
        </button>

        <button
          onClick={onOpenSettings}
          className={`w-9 h-9 mx-auto rounded flex items-center justify-center transition-colors ${
            view === "settings"
              ? "bg-[#2563eb] text-white"
              : "text-[#9da1ad] hover:text-[#f1f2f4] hover:bg-[#20232a]"
          }`}
          title="Settings"
        >
          <Settings size={17} />
        </button>
      </div>
    </nav>
  );
}

// ==========================================
// Component: Telemetry Bar (Bottom Status Bar)
// ==========================================
function TelemetryBar({
  backendConnected,
  modelStatus,
  loadedModel,
  activeOperationModel,
  serverOn,
  metrics,
  inferenceMetrics,
  onServerClick,
  onModelClick,
  onTelemetryClick,
}: {
  backendConnected: boolean;
  modelStatus: ModelLoadingState;
  loadedModel: string | null;
  activeOperationModel: string | null;
  serverOn: boolean;
  metrics: SystemMetrics | null;
  inferenceMetrics: InferenceThroughputMetrics;
  onServerClick: () => void;
  onModelClick: () => void;
  onTelemetryClick: () => void;
}) {
  const ramStr = metrics
    ? `${(metrics.used_ram / (1024 * 1024 * 1024)).toFixed(1)} / ${(metrics.total_ram / (1024 * 1024 * 1024)).toFixed(1)} GB`
    : "Unavailable";
  const cpuStr = metrics && metrics.cpu_usage !== null ? `${metrics.cpu_usage.toFixed(0)}%` : "—";

  return (
    <footer className="h-[24px] border-t border-[#282b34] bg-[#141519] px-3 flex items-center justify-between font-mono text-[11px] text-[#787d8a] shrink-0 select-none z-20">
      <div className="flex items-center gap-3">
        {/* Backend Connectivity */}
        <div className="flex items-center gap-1.5">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              backendConnected ? "bg-emerald-500" : "bg-red-500"
            }`}
          />
          <span className={backendConnected ? "text-[#a0a5b2]" : "text-red-400 font-semibold"}>
            {backendConnected ? "Backend: Connected" : "Backend: Disconnected"}
          </span>
        </div>

        <span className="h-2.5 w-[1px] bg-[#282b34]" />

        {/* Engine type */}
        <div className="flex items-center gap-1 text-[#8c919e]">
          <Cpu size={12} />
          <span>llama.cpp</span>
        </div>

        <span className="h-2.5 w-[1px] bg-[#282b34]" />

        {/* Truthful Model State Pill */}
        <button
          onClick={onModelClick}
          className="flex items-center gap-1.5 px-1.5 py-0.5 rounded hover:bg-[#1e2027] text-[#a0a5b2] transition-colors"
        >
          {modelStatus === "loading" && (
            <>
              <Loader2 size={11} className="animate-spin text-amber-400" />
              <span className="text-amber-400">Loading {activeOperationModel}...</span>
            </>
          )}
          {modelStatus === "unloading" && (
            <>
              <Loader2 size={11} className="animate-spin text-amber-400" />
              <span className="text-amber-400">Unloading...</span>
            </>
          )}
          {modelStatus === "loaded" && (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              <span className="text-emerald-400 font-medium truncate max-w-[220px]">
                {loadedModel}
              </span>
            </>
          )}
          {modelStatus === "failed" && (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
              <span className="text-red-400">Load Failed</span>
            </>
          )}
          {modelStatus === "not_loaded" && (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-zinc-600" />
              <span className="text-[#6c707d]">No model loaded</span>
            </>
          )}
        </button>

        <span className="h-2.5 w-[1px] bg-[#282b34]" />

        {/* Local Server State */}
        <button
          onClick={onServerClick}
          className={`flex items-center gap-1.5 px-1.5 py-0.5 rounded hover:bg-[#1e2027] transition-colors ${
            serverOn ? "text-emerald-400" : "text-[#6c707d]"
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${serverOn ? "bg-emerald-500" : "bg-zinc-600"}`}
          />
          <span>API: {serverOn ? "127.0.0.1:1234" : "Offline"}</span>
        </button>
      </div>

      {/* Real Hardware & Inference Metrics (Clickable) */}
      <button
        onClick={onTelemetryClick}
        className="flex items-center gap-4 text-[#8c919e] hover:text-[#f1f2f4] hover:bg-[#1a1c22] px-2 py-0.5 rounded transition-colors"
        title="Open Real-Time Hardware Telemetry"
      >
        <span>CPU: {cpuStr}</span>
        <span>RAM: {ramStr}</span>
        <span>
          GPU:{" "}
          {metrics?.gpu_name
            ? metrics.gpu_name.includes("Intel")
              ? "Intel UHD"
              : metrics.gpu_name.split(" ")[0]
            : "Unavailable"}
        </span>
        {inferenceMetrics.lastCompletedTokPerSec !== null && (
          <span className="text-emerald-400 font-semibold">
            {inferenceMetrics.lastCompletedTokPerSec} tok/s
          </span>
        )}
      </button>
    </footer>
  );
}

// ==========================================
// View: Chat View
// ==========================================
interface ChatViewProps {
  conversation: ConversationSession;
  conversations: ConversationSession[];
  activeId: string;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onRenameConversation: (id: string, newTitle: string) => void;
  onDeleteConversation: (id: string) => void;
  onPinConversation: (id: string) => void;
  onArchiveConversation: (id: string) => void;
  onDuplicateConversation: (id: string) => void;
  onClearConversation: (id: string) => void;
  settingsMode: SettingsMode;
  setSettingsMode: (mode: SettingsMode) => void;
  activePresetId: string;
  allPresets: Preset[];
  onApplyPreset: (id: string) => void;
  onOpenImportModal: () => void;
  onExportConfig: () => void;
  simpleStyle: ResponseStyle;
  onSimpleStyleChange: (s: ResponseStyle) => void;
  simpleLength: ResponseLength;
  onSimpleLengthChange: (l: ResponseLength) => void;
  settings: GenerationSettings;
  onUpdateSettings: (s: GenerationSettings) => void;
  backendConnected: boolean;
  modelStatus: ModelLoadingState;
  loadedModel: string | null;
  loadingError: string | null;
  activeOperationModel: string | null;
  onDismissError: () => void;
  models: ModelInfo[];
  onLoadModel: (path: string) => void;
  onUnloadModel: () => void;
  onSwitchToModels: () => void;
  setConversations: React.Dispatch<React.SetStateAction<ConversationSession[]>>;
  inferenceMetrics: InferenceThroughputMetrics;
  setInferenceMetrics: React.Dispatch<React.SetStateAction<InferenceThroughputMetrics>>;
  inferenceSettings: InferenceSettings;
}

function ChatView({
  conversation,
  conversations,
  activeId,
  onSelectConversation,
  onNewConversation,
  onRenameConversation,
  onDeleteConversation,
  onPinConversation,
  onArchiveConversation,
  onDuplicateConversation,
  onClearConversation,
  settingsMode,
  setSettingsMode,
  activePresetId,
  allPresets,
  onApplyPreset,
  onOpenImportModal,
  onExportConfig,
  simpleStyle,
  onSimpleStyleChange,
  simpleLength,
  onSimpleLengthChange,
  settings,
  onUpdateSettings,
  backendConnected,
  modelStatus,
  loadedModel,
  loadingError,
  activeOperationModel,
  onDismissError,
  models,
  onLoadModel,
  onUnloadModel,
  onSwitchToModels,
  setConversations,
  inferenceMetrics,
  setInferenceMetrics,
  inferenceSettings,
}: ChatViewProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [enableThinking, setEnableThinking] = useState<boolean>(() => {
    return localStorage.getItem("local_ai_workstation_enable_thinking") !== "false";
  });
  const [searchFilter, setSearchFilter] = useState("");
  const [viewArchived, setViewArchived] = useState(false);
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target?.result as string;
        if (content) {
          const tokens = Math.ceil(content.length / 3.8);
          const att: ChatAttachment = {
            id: `att-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
            name: file.name,
            content,
            size: file.size,
            tokens,
          };
          setAttachments((prev) => [...prev, att]);
        }
      };
      reader.readAsText(file);
    }
    if (e.target) e.target.value = "";
  };

  const handleRemoveAttachment = (id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id));
  };

  const attachmentTokensTotal = useMemo(() => {
    return attachments.reduce((acc, a) => acc + a.tokens, 0);
  }, [attachments]);

  const filteredConversations = useMemo(() => {
    return filterSessionsBySearch(conversations, searchFilter, viewArchived);
  }, [conversations, searchFilter, viewArchived]);

  const grouped = useMemo(
    () => groupSessionsByDate(filteredConversations),
    [filteredConversations],
  );
  const estTokens = useMemo(
    () => estimateContextTokens(conversation?.messages || []) + attachmentTokensTotal,
    [conversation?.messages, attachmentTokensTotal],
  );

  const maxCtx = settings.maxTokens ? Math.max(4096, settings.maxTokens * 4) : 4096;
  const ctxPercent = Math.min(100, Math.round((estTokens / maxCtx) * 100));

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation?.messages, streaming]);

  // Token listener
  useEffect(() => {
    let isMounted = true;
    let unlistenFn: (() => void) | null = null;

    listen<string>("chat_token", (event) => {
      if (!isMounted) return;
      const token = event.payload;
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeId) return c;
          const msgs = [...c.messages];
          const lastIndex = msgs.length - 1;
          if (lastIndex >= 0 && msgs[lastIndex].role === "assistant") {
            const currentRaw = (msgs[lastIndex].text || "") + token;
            let displayThought = msgs[lastIndex].thought;
            let displayText = currentRaw;

            if (currentRaw.includes("<think>")) {
              const startIdx = currentRaw.indexOf("<think>");
              const endIdx = currentRaw.indexOf("</think>");
              if (endIdx !== -1) {
                displayThought = currentRaw.substring(startIdx + 7, endIdx).trim();
                displayText = (currentRaw.substring(0, startIdx) + currentRaw.substring(endIdx + 8)).trimStart();
              } else {
                displayThought = currentRaw.substring(startIdx + 7).trim();
                displayText = currentRaw.substring(0, startIdx);
              }
            }

            msgs[lastIndex] = {
              ...msgs[lastIndex],
              text: displayText,
              thought: displayThought,
            };
          }
          return { ...c, messages: msgs, updatedAt: Date.now() };
        }),
      );
    })
      .then((fn) => {
        if (isMounted) unlistenFn = fn;
        else fn();
      })
      .catch(() => {});

    return () => {
      isMounted = false;
      if (unlistenFn) unlistenFn();
    };
  }, [activeId, setConversations]);

  const handleStopGeneration = async () => {
    setStreaming(false);
    try {
      await invoke("cancel_generation");
    } catch (err) {
      console.warn("Failed to cancel generation on backend:", err);
    }
  };

  // Send message with real inference timing, retry support, and interrupted handling
  const handleSendPrompt = async (textToSend?: string) => {
    const input = (textToSend !== undefined ? textToSend : prompt).trim();
    if (!input || streaming) return;

    // Strict validation: cannot generate if no model is genuinely loaded
    if (modelStatus !== "loaded" || !loadedModel) {
      alert("No model is currently loaded. Please load a GGUF model first.");
      return;
    }

    const now = Date.now();
    const userMsg: ChatMessageItem = {
      id: "msg-" + now + "-u",
      role: "user",
      text: input,
      timestamp: now,
    };
    const assistantMsg: ChatMessageItem = {
      id: "msg-" + now + "-a",
      role: "assistant",
      text: "",
      timestamp: now,
    };

    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== activeId) return c;
        const newTitle = c.messages.length === 0 ? generateSessionTitle(input) : c.title;
        return {
          ...c,
          title: newTitle,
          messages: [...c.messages, userMsg, assistantMsg],
          updatedAt: Date.now(),
        };
      }),
    );

    setPrompt("");
    setStreaming(true);
    setInferenceMetrics((prev) => ({
      ...prev,
      active: true,
      promptTokens: Math.ceil(input.length / 3.8),
      generatedTokens: 0,
      elapsedMs: 0,
    }));

    const startTime = Date.now();
    try {
      const history: Array<{ role: string; content: string }> = (conversation?.messages || []).map((m) => ({
        role: m.role,
        content: m.text,
      }));

      // Inject attached documents if present
      let finalContent = input;
      if (attachments.length > 0) {
        const attachmentText = attachments
          .map((a) => `--- Attached Document: ${a.name} ---\n${a.content}\n--- End Document ---`)
          .join("\n\n");
        finalContent = `${attachmentText}\n\n${input}`;
        setAttachments([]);
      }

      // If thinking is disabled, instruct reasoning models not to output thinking tags
      if (!enableThinking) {
        history.unshift({
          role: "system",
          content: "You must not include any <think> or reasoning tags in your output. Provide only the direct, final response immediately without internal monologue.",
        });
      }

      history.push({ role: "user", content: finalContent });

      const response = await invoke<string>("chat_completions", {
        request: {
          model: loadedModel,
          messages: history,
          temperature: settings.temperature,
          top_p: settings.topP,
          top_k: settings.topK,
          min_p: settings.minP,
          repetition_penalty: settings.repetitionPenalty,
          frequency_penalty: settings.frequencyPenalty,
          presence_penalty: settings.presencePenalty,
          max_tokens: settings.maxTokens,
          seed: settings.seed,
          stop_tokens: settings.stopSequences.length > 0 ? settings.stopSequences : undefined,
          stream: true,
          port: 1234,
          threads: inferenceSettings.threads > 0 ? inferenceSettings.threads : 4,
          n_ctx: inferenceSettings.defaultContextLength || 512,
          flash_attention: inferenceSettings.flashAttention || false,
        },
      });

      const elapsedMs = Math.max(50, Date.now() - startTime);
      const tokenCount = Math.round((response.length || 100) / 3.8);
      const tokPerSec = calculateThroughput(tokenCount, elapsedMs);

      setInferenceMetrics({
        active: false,
        promptTokens: Math.ceil(input.length / 3.8),
        generatedTokens: tokenCount,
        elapsedMs,
        tokPerSec,
        promptSpeedTokPerSec: null,
        lastCompletedTokPerSec: tokPerSec,
      });

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeId) return c;
          const msgs = [...c.messages];
          const last = msgs[msgs.length - 1];
          let finalFinalText = last.text || response;
          let finalThought = last.thought;
          if (finalFinalText.includes("<think>")) {
            const sIdx = finalFinalText.indexOf("<think>");
            const eIdx = finalFinalText.indexOf("</think>");
            if (eIdx !== -1) {
              finalThought = finalFinalText.substring(sIdx + 7, eIdx).trim();
              finalFinalText = (finalFinalText.substring(0, sIdx) + finalFinalText.substring(eIdx + 8)).trimStart();
            }
          }
          if (last && last.role === "assistant") {
            msgs[msgs.length - 1] = {
              ...last,
              text: finalFinalText,
              thought: finalThought,
              stats: {
                tokens: tokenCount,
                latencyMs: elapsedMs,
                tokPerSec: tokPerSec || undefined,
              },
              interrupted: false,
            };
          }
          return { ...c, messages: msgs, updatedAt: Date.now() };
        }),
      );
    } catch (err: unknown) {
      const errMsg =
        typeof err === "string" ? err : (err as Error)?.message || "Inference interrupted";
      setInferenceMetrics((prev) => ({ ...prev, active: false }));

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeId) return c;
          const msgs = [...c.messages];
          const last = msgs[msgs.length - 1];
          if (last && last.role === "assistant") {
            msgs[msgs.length - 1] = {
              ...last,
              text: last.text
                ? `${last.text}\n\n[Generation Interrupted: ${errMsg}]`
                : `[Generation Failed: ${errMsg}]`,
              interrupted: true,
            };
          }
          return { ...c, messages: msgs, updatedAt: Date.now() };
        }),
      );
    } finally {
      setStreaming(false);
    }
  };

  const isModelReady = modelStatus === "loaded" && !!loadedModel;

  return (
    <div className="flex h-full w-full overflow-hidden relative bg-[#121316]">
      {/* Left Conversation Sidebar */}
      <aside
        className={`h-full flex flex-col bg-[#16181d] border-r border-[#282b34] transition-all duration-150 shrink-0 select-none z-10 ${
          sidebarOpen ? "w-[240px]" : "w-0 overflow-hidden border-none"
        }`}
      >
        {sidebarOpen && (
          <div className="flex flex-col h-full w-[240px]">
            <div className="p-2.5 pb-2 flex items-center justify-between gap-1.5 border-b border-[#242730]">
              <button
                onClick={onNewConversation}
                className="flex-1 flex items-center justify-center gap-1.5 px-2.5 py-1.5 rounded bg-[#2563eb] hover:bg-[#1d4ed8] text-white text-xs font-medium transition-colors"
              >
                <Plus size={14} />
                <span>New Chat</span>
              </button>

              <button
                onClick={() => setSidebarOpen(false)}
                className="p-1.5 rounded text-[#8c919e] hover:text-[#f1f2f4] hover:bg-[#20232a] transition-colors"
                title="Collapse sidebar"
              >
                <PanelLeftClose size={15} />
              </button>
            </div>

            {/* Search Filter & View Mode Tabs */}
            <div className="px-2.5 py-2 space-y-1.5">
              <div className="relative">
                <Search size={12} className="absolute left-2.5 top-2 text-[#595d69]" />
                <input
                  type="text"
                  placeholder="Search chats..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  className="w-full bg-[#1c1e24] border border-[#282b34] focus:border-blue-500 text-xs text-[#f1f2f4] pl-7 pr-2.5 py-1 rounded placeholder-[#595d69] outline-none transition-colors"
                />
              </div>

              <div className="flex rounded bg-[#1c1e24] p-0.5 border border-[#242730]">
                <button
                  onClick={() => setViewArchived(false)}
                  className={`flex-1 py-0.5 text-[10.5px] rounded transition-colors ${
                    !viewArchived
                      ? "bg-[#2563eb] text-white font-medium"
                      : "text-[#8c919e] hover:text-white"
                  }`}
                >
                  Chats ({conversations.filter((c) => !c.archived).length})
                </button>
                <button
                  onClick={() => setViewArchived(true)}
                  className={`flex-1 py-0.5 text-[10.5px] rounded transition-colors ${
                    viewArchived
                      ? "bg-[#2563eb] text-white font-medium"
                      : "text-[#8c919e] hover:text-white"
                  }`}
                >
                  Archived ({conversations.filter((c) => c.archived).length})
                </button>
              </div>
            </div>

            {/* Conversation List */}
            <div className="flex-1 overflow-y-auto px-1.5 py-1 space-y-3 custom-scrollbar">
              {viewArchived ? (
                <div className="space-y-1">
                  <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-[#595d69]">
                    Archived Conversations
                  </div>
                  {filteredConversations.length === 0 ? (
                    <div className="px-3 py-4 text-center text-[#595d69] text-xs">
                      No archived conversations.
                    </div>
                  ) : (
                    filteredConversations.map((conv) => (
                      <div
                        key={conv.id}
                        onClick={() => onSelectConversation(conv.id)}
                        className={`group relative flex items-center justify-between px-2 py-1.5 rounded text-xs cursor-pointer transition-colors ${
                          conv.id === activeId
                            ? "bg-[#2563eb] text-white"
                            : "text-[#a0a5b2] hover:bg-[#20232a] hover:text-white"
                        }`}
                      >
                        <span className="truncate pr-2 flex-1">{conv.title}</span>
                        <div className="flex items-center gap-1 shrink-0">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onArchiveConversation(conv.id);
                            }}
                            className="p-1 rounded text-zinc-400 hover:text-white"
                            title="Restore from archive"
                          >
                            <RotateCcw size={11} />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteConversation(conv.id);
                            }}
                            className="p-1 rounded text-zinc-400 hover:text-red-400"
                            title="Delete permanently"
                          >
                            <Trash2 size={11} />
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              ) : filteredConversations.length === 0 ? (
                <div className="px-3 py-4 text-center text-[#595d69] text-xs">
                  {searchFilter ? "No conversations match search." : "No conversations yet."}
                </div>
              ) : (
                [
                  { label: "Pinned", items: grouped.pinned, isPinnedGroup: true },
                  { label: "Today", items: grouped.today },
                  { label: "Yesterday", items: grouped.yesterday },
                  { label: "Previous 7 Days", items: grouped.past7 },
                  { label: "Previous 30 Days", items: grouped.past30 },
                  { label: "Older", items: grouped.older },
                ]
                  .filter((g) => g.items.length > 0)
                  .map((group) => (
                    <div key={group.label} className="space-y-0.5">
                      <div className="px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-[#595d69] flex items-center gap-1">
                        {group.isPinnedGroup && <Pin size={10} className="text-blue-400" />}
                        <span>{group.label}</span>
                      </div>
                      {group.items.map((conv) => {
                        const isActive = conv.id === activeId;
                        return (
                          <div
                            key={conv.id}
                            onClick={() => onSelectConversation(conv.id)}
                            className={`group relative flex items-center justify-between px-2 py-1.5 rounded text-xs cursor-pointer transition-colors ${
                              isActive
                                ? "bg-[#2563eb] text-white font-medium"
                                : "text-[#a0a5b2] hover:text-white hover:bg-[#20232a]"
                            }`}
                          >
                            <div className="truncate pr-1 flex-1 flex items-center gap-1.5">
                              {conv.pinned && <Pin size={10} className="text-blue-300 shrink-0" />}
                              <span className="truncate">{conv.title}</span>
                            </div>

                            <div className="hidden group-hover:flex items-center gap-0.5 shrink-0">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onPinConversation(conv.id);
                                }}
                                className="p-1 text-zinc-300 hover:text-white rounded"
                                title={conv.pinned ? "Unpin" : "Pin"}
                              >
                                <Pin size={11} className={conv.pinned ? "fill-current" : ""} />
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onDuplicateConversation(conv.id);
                                }}
                                className="p-1 text-zinc-300 hover:text-white rounded"
                                title="Duplicate"
                              >
                                <Copy size={11} />
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onArchiveConversation(conv.id);
                                }}
                                className="p-1 text-zinc-300 hover:text-white rounded"
                                title="Archive"
                              >
                                <Archive size={11} />
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const newT = window.prompt("Rename chat:", conv.title);
                                  if (newT && newT.trim()) onRenameConversation(conv.id, newT);
                                }}
                                className="p-1 text-zinc-300 hover:text-white rounded"
                                title="Rename"
                              >
                                <Pencil size={11} />
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onDeleteConversation(conv.id);
                                }}
                                className="p-1 text-zinc-300 hover:text-red-300 rounded"
                                title="Delete"
                              >
                                <Trash2 size={11} />
                              </button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ))
              )}
            </div>

            {/* Clear conversation action button */}
            <div className="p-2 border-t border-[#242730] bg-[#141519] flex justify-between items-center text-[10.5px]">
              <button
                onClick={() => onClearConversation(activeId)}
                className="text-[#6c707d] hover:text-red-400 transition-colors flex items-center gap-1"
                title="Clear messages in current chat"
              >
                <Trash2 size={11} />
                <span>Clear chat</span>
              </button>
              <span className="text-[#595d69] font-mono">{conversations.length} sessions</span>
            </div>
          </div>
        )}
      </aside>

      {/* Main Chat Workspace */}
      <section className="flex-1 flex flex-col h-full min-w-0 min-h-0 overflow-hidden relative bg-[#121316]">
        {/* Header Bar */}
        <header className="h-[40px] border-b border-[#282b34] bg-[#16181d] px-3.5 flex items-center justify-between shrink-0 select-none z-10">
          <div className="flex items-center gap-2.5 min-w-0">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-1 rounded text-[#8c919e] hover:text-white hover:bg-[#20232a] transition-colors"
                title="Open sidebar"
              >
                <PanelLeftOpen size={15} />
              </button>
            )}

            <div className="text-xs font-semibold text-[#f1f2f4] truncate">
              {conversation?.title || "New Chat"}
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Model Selector Dropdown Button */}
            <div className="relative">
              <button
                onClick={() => setModelDropdownOpen((v) => !v)}
                className="flex items-center gap-2 px-2.5 py-1 rounded bg-[#1c1e24] hover:bg-[#23262e] border border-[#282b34] text-xs font-medium text-[#f1f2f4] transition-colors"
              >
                {modelStatus === "loading" && (
                  <Loader2 size={11} className="animate-spin text-amber-400" />
                )}
                {modelStatus === "unloading" && (
                  <Loader2 size={11} className="animate-spin text-amber-400" />
                )}
                {modelStatus === "loaded" && (
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                )}
                {modelStatus === "failed" && (
                  <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
                )}
                {modelStatus === "not_loaded" && (
                  <span className="w-1.5 h-1.5 rounded-full bg-zinc-500" />
                )}

                <span className="truncate max-w-[200px]">
                  {modelStatus === "loaded" && loadedModel
                    ? loadedModel
                    : modelStatus === "loading"
                      ? `Loading ${activeOperationModel}...`
                      : modelStatus === "unloading"
                        ? "Unloading model..."
                        : modelStatus === "failed"
                          ? "Model load failed"
                          : "No model loaded"}
                </span>

                <ChevronDown size={12} className="text-[#8c919e]" />
              </button>

              {/* Model Dropdown Menu */}
              {modelDropdownOpen && (
                <div className="absolute right-0 top-9 w-80 bg-[#1a1c22] border border-[#2f333e] rounded shadow-xl py-1 z-50 text-xs">
                  <div className="px-3 py-1.5 border-b border-[#282b34] flex items-center justify-between bg-[#141519]">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6c707d]">
                      Loaded Model Status
                    </span>
                    {modelStatus === "loaded" && (
                      <button
                        onClick={() => {
                          setModelDropdownOpen(false);
                          onUnloadModel();
                        }}
                        className="text-[11px] text-red-400 hover:text-red-300 font-medium"
                      >
                        Unload Model
                      </button>
                    )}
                  </div>

                  {modelStatus === "loaded" && loadedModel ? (
                    <div className="px-3 py-2 bg-[#17201c] border-b border-[#282b34] flex items-center justify-between">
                      <div className="truncate pr-2">
                        <div className="font-semibold text-emerald-300 truncate">{loadedModel}</div>
                        <div className="text-[10.5px] text-[#8c919e]">
                          Ready for inference (mmap)
                        </div>
                      </div>
                      <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800">
                        Active
                      </span>
                    </div>
                  ) : (
                    <div className="px-3 py-2 text-[#787d8a] italic border-b border-[#282b34]">
                      {modelStatus === "loading"
                        ? `Loading ${activeOperationModel}...`
                        : "No model is currently in memory."}
                    </div>
                  )}

                  <div className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-[#6c707d]">
                    Available On Disk ({models.length})
                  </div>

                  <div className="max-h-56 overflow-y-auto custom-scrollbar divide-y divide-[#242730]">
                    {models.length === 0 ? (
                      <div className="px-3 py-2 text-[#787d8a]">
                        No GGUF models found in models/ directory.
                      </div>
                    ) : (
                      models.map((m) => {
                        const isThisLoaded =
                          loadedModel === m.id || loadedModel === m.path.split("/").pop();
                        const isThisLoading =
                          activeOperationModel === m.id && modelStatus === "loading";
                        const { cleanName, quant } = parseModelDetails(m.id);

                        return (
                          <div
                            key={m.id}
                            className="px-3 py-2 flex items-center justify-between hover:bg-[#20232b] transition-colors"
                          >
                            <div className="truncate pr-2">
                              <div className="font-medium text-[#f1f2f4] truncate">{cleanName}</div>
                              <div className="text-[10.5px] text-[#787d8a] font-mono">
                                {quant} · {formatBytes(m.size)}
                              </div>
                            </div>

                            {isThisLoaded ? (
                              <button
                                onClick={() => {
                                  setModelDropdownOpen(false);
                                  onUnloadModel();
                                }}
                                className="px-2 py-0.5 rounded bg-red-950/60 hover:bg-red-900/80 text-red-300 border border-red-800/60 text-[11px] font-medium"
                              >
                                Unload
                              </button>
                            ) : (
                              <button
                                onClick={() => {
                                  setModelDropdownOpen(false);
                                  onLoadModel(m.path);
                                }}
                                disabled={modelStatus === "loading" || modelStatus === "unloading"}
                                className="px-2 py-0.5 rounded bg-[#2563eb] hover:bg-[#1d4ed8] disabled:opacity-40 disabled:cursor-not-allowed text-white text-[11px] font-medium transition-colors"
                              >
                                {isThisLoading ? "Loading..." : "Load"}
                              </button>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>

                  <div className="p-2 border-t border-[#282b34] bg-[#141519]">
                    <button
                      onClick={() => {
                        setModelDropdownOpen(false);
                        onSwitchToModels();
                      }}
                      className="w-full text-center py-1 rounded bg-[#20232a] hover:bg-[#282c35] text-[#a0a5b2] hover:text-white text-[11px] font-medium transition-colors border border-[#2f333e]"
                    >
                      Open Models Library
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Context Window Capacity Indicator */}
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#16181d] border border-[#282b34] font-mono text-[10.5px]">
              <div className="w-12 bg-[#20232a] h-1.5 rounded-full overflow-hidden border border-[#282b34]">
                <div
                  className={`h-full transition-all duration-300 ${
                    ctxPercent > 85 ? "bg-red-500" : ctxPercent > 65 ? "bg-amber-400" : "bg-emerald-400"
                  }`}
                  style={{ width: `${ctxPercent}%` }}
                />
              </div>
              <span className={ctxPercent > 85 ? "text-red-400 font-semibold" : "text-[#8c919e]"}>
                {estTokens} / {maxCtx} tok ({ctxPercent}%)
              </span>
            </div>

            {/* Generation Parameters Panel Toggle */}
            <button
              onClick={() => setInspectorOpen((v) => !v)}
              className={`p-1.5 rounded transition-colors ${
                inspectorOpen
                  ? "bg-[#2563eb] text-white"
                  : "text-[#8c919e] hover:text-white hover:bg-[#20232a]"
              }`}
              title="Generation Parameters & Presets"
            >
              <SlidersHorizontal size={14} />
            </button>
          </div>
        </header>

        {/* Global Error Notice if Load Failed */}
        {loadingError && (
          <div className="px-4 py-2 bg-[#2a1417] border-b border-[#5c1d24] flex items-center justify-between text-xs text-red-200 shrink-0">
            <div className="flex items-center gap-2">
              <AlertCircle size={15} className="text-red-400 shrink-0" />
              <span>
                <strong>Model Error:</strong> {loadingError}
              </span>
            </div>
            <button
              onClick={onDismissError}
              className="text-red-400 hover:text-red-200 text-xs px-2 py-0.5 rounded hover:bg-[#3d181e]"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Messages Stream Area */}
        <div className="flex-1 overflow-y-auto px-4 py-5 custom-scrollbar">
          <div className="max-w-3xl mx-auto space-y-5">
            {(!conversation?.messages || conversation.messages.length === 0) && (
              <EmptyChatHero
                isModelReady={isModelReady}
                loadedModel={loadedModel}
                modelStatus={modelStatus}
                onSelectPrompt={(p) => handleSendPrompt(p)}
                onSwitchToModels={onSwitchToModels}
              />
            )}

            {conversation?.messages.map((message, i) => (
              <MessageBubble
                key={message.id || i}
                message={message}
                isLast={i === conversation.messages.length - 1}
                streaming={streaming && i === conversation.messages.length - 1}
                onRetry={
                  message.role === "assistant" && message.interrupted
                    ? () => {
                        const prevUser = conversation.messages[i - 1];
                        if (prevUser && prevUser.role === "user") {
                          handleSendPrompt(prevUser.text);
                        }
                      }
                    : undefined
                }
              />
            ))}

            {streaming && (
              <div className="flex items-center gap-2 text-xs text-[#8c919e] font-mono px-3 py-1.5 rounded bg-[#181a20] border border-[#282b34] w-fit">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                <span>Generating tokens...</span>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Anchored Bottom Composer */}
        <div className="p-3 border-t border-[#282b34] bg-[#16181d] shrink-0">
          <div className="max-w-3xl mx-auto space-y-1.5">
            {/* Attached documents pill list */}
            {attachments.length > 0 && (
              <div className="flex flex-wrap gap-1.5 pb-1">
                {attachments.map((att) => (
                  <div
                    key={att.id}
                    className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#20232a] border border-[#2f333e] text-xs text-[#d4d6dc]"
                  >
                    <FileText size={11} className="text-blue-400" />
                    <span className="font-mono text-[11px] truncate max-w-[160px]">{att.name}</span>
                    <span className="text-[10px] text-[#6c707d] font-mono">({att.tokens} tok)</span>
                    <button
                      onClick={() => handleRemoveAttachment(att.id)}
                      className="p-0.5 text-[#6c707d] hover:text-red-400"
                    >
                      <X size={10} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="relative rounded border border-[#282b34] bg-[#1a1c22] focus-within:border-blue-500 transition-colors">
              <textarea
                value={prompt}
                disabled={!isModelReady || streaming}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendPrompt();
                  }
                }}
                placeholder={
                  !backendConnected
                    ? "Inference daemon not connected..."
                    : modelStatus === "loading"
                      ? `Loading model ${activeOperationModel}... Please wait.`
                      : modelStatus === "unloading"
                        ? "Unloading model... Please wait."
                        : !isModelReady
                          ? "No model loaded. Please load a model to begin inference."
                          : `Message ${loadedModel}... (Enter to send, Shift+Enter for new line)`
                }
                className="w-full bg-transparent text-xs text-[#f1f2f4] placeholder-[#595d69] resize-none outline-none px-3 pt-2.5 pb-2 min-h-[50px] max-h-[160px] custom-scrollbar disabled:cursor-not-allowed disabled:opacity-50"
                rows={2}
              />

              <div className="flex items-center justify-between px-2.5 pb-2 text-[11px] text-[#6c707d]">
                <div className="flex items-center gap-2 font-mono">
                  <label className="flex items-center gap-1 cursor-pointer text-[#8c919e] hover:text-[#f1f2f4] px-1.5 py-0.5 rounded hover:bg-[#20232a] transition-colors" title="Attach text or code file">
                    <Paperclip size={12} className="text-blue-400" />
                    <span className="text-[10.5px]">Attach</span>
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept=".txt,.md,.py,.rs,.js,.ts,.tsx,.jsx,.json,.c,.cpp,.h,.go,.java,.html,.css,.yaml,.toml,.sh"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </label>
                  <span>·</span>
                  <span>~{Math.max(0, Math.round(prompt.length / 4)) + attachmentTokensTotal} tokens</span>
                  <span>·</span>
                  <span>Temp: {settings.temperature}</span>
                  <span>·</span>
                  <span className="text-[#8c919e]">
                    {PRESETS.find((p) => p.id === activePresetId)?.name || "Custom"}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  {/* Thinking / Reasoning Toggle */}
                  <button
                    type="button"
                    onClick={() => {
                      setEnableThinking((prev) => {
                        const next = !prev;
                        localStorage.setItem("local_ai_workstation_enable_thinking", String(next));
                        return next;
                      });
                    }}
                    title={enableThinking ? "Thinking enabled (DeepSeek-R1 / Qwen reasoning). Click to disable." : "Thinking disabled. Click to enable."}
                    className={`flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono border transition-all ${
                      enableThinking
                        ? "bg-purple-950/40 text-purple-300 border-purple-800/60 hover:bg-purple-900/40"
                        : "bg-[#1c1e24] text-[#6c707d] border-[#282b34] hover:text-[#9ea3ae] hover:bg-[#20232a]"
                    }`}
                  >
                    <Brain size={12} className={enableThinking ? "text-purple-400" : "text-[#6c707d]"} />
                    <span>Thinking: {enableThinking ? "ON" : "OFF"}</span>
                  </button>

                  <button
                    onClick={streaming ? () => handleStopGeneration() : () => handleSendPrompt()}
                    disabled={!isModelReady || (!prompt.trim() && attachments.length === 0 && !streaming)}
                    className="flex items-center gap-1.5 px-3 py-1 rounded bg-[#2563eb] hover:bg-[#1d4ed8] disabled:bg-[#20232a] disabled:text-[#4d515c] disabled:cursor-not-allowed text-white text-xs font-medium transition-colors"
                  >
                    {streaming ? (
                      <>
                        <Square size={11} fill="currentColor" />
                        <span>Stop</span>
                      </>
                    ) : (
                      <>
                        <Send size={11} />
                        <span>Send</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between text-[10.5px] text-[#595d69] px-1 font-mono">
              <span>Local inference · zero-copy mmap</span>
              <span>Enter to send · Shift+Enter for new line</span>
            </div>
          </div>
        </div>
      </section>

      {/* Right Generation Parameters & Preset Inspector */}
      {inspectorOpen && (
        <aside className="w-[300px] h-full bg-[#16181d] border-l border-[#282b34] flex flex-col shrink-0 select-none z-10">
          <header className="p-3 border-b border-[#282b34] flex items-center justify-between">
            <span className="text-xs font-semibold text-[#f1f2f4]">Generation Settings</span>
            <button
              onClick={() => setInspectorOpen(false)}
              className="p-0.5 rounded text-[#8c919e] hover:text-white"
            >
              <X size={14} />
            </button>
          </header>

          <div className="flex-1 overflow-y-auto p-3 space-y-4 custom-scrollbar text-xs">
            {/* Mode Switcher Tabs */}
            <div className="flex rounded bg-[#1c1e24] p-0.5 border border-[#282b34]">
              <button
                onClick={() => setSettingsMode("simple")}
                className={`flex-1 py-1 rounded text-xs font-medium transition-colors ${
                  settingsMode === "simple"
                    ? "bg-[#2563eb] text-white"
                    : "text-[#8c919e] hover:text-[#f1f2f4]"
                }`}
              >
                Simple Mode
              </button>
              <button
                onClick={() => setSettingsMode("advanced")}
                className={`flex-1 py-1 rounded text-xs font-medium transition-colors ${
                  settingsMode === "advanced"
                    ? "bg-[#2563eb] text-white"
                    : "text-[#8c919e] hover:text-[#f1f2f4]"
                }`}
              >
                Advanced Mode
              </button>
            </div>

            {/* Presets Quick Selector */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-[#595d69]">
                <span>Active Preset</span>
                <span>{allPresets.length} presets</span>
              </div>
              <select
                value={activePresetId}
                onChange={(e) => onApplyPreset(e.target.value)}
                className="w-full bg-[#1c1e24] border border-[#282b34] text-xs text-[#f1f2f4] p-1.5 rounded outline-none focus:border-blue-500 cursor-pointer"
              >
                <optgroup label="Standard Presets">
                  {PRESETS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name} ({p.category})
                    </option>
                  ))}
                </optgroup>
                {allPresets.length > PRESETS.length && (
                  <optgroup label="Custom Presets">
                    {allPresets.slice(PRESETS.length).map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name} ({p.category})
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
              <p className="text-[11px] text-[#6c707d] leading-relaxed">
                {allPresets.find((p) => p.id === activePresetId)?.description ||
                  "Custom configuration"}
              </p>

              {/* JSON Configuration Import / Export Actions */}
              <div className="flex items-center gap-1.5 pt-1">
                <button
                  onClick={onOpenImportModal}
                  className="flex-1 flex items-center justify-center gap-1.5 py-1 px-2 rounded bg-[#20232a] hover:bg-[#282c35] border border-[#282b34] text-[#a0a5b2] hover:text-white transition-colors"
                  title="Import JSON Configuration"
                >
                  <Upload size={12} className="text-blue-400" />
                  <span>Import JSON</span>
                </button>

                <button
                  onClick={onExportConfig}
                  className="flex-1 flex items-center justify-center gap-1.5 py-1 px-2 rounded bg-[#20232a] hover:bg-[#282c35] border border-[#282b34] text-[#a0a5b2] hover:text-white transition-colors"
                  title="Export Current Settings as JSON"
                >
                  <Download size={12} className="text-emerald-400" />
                  <span>Export JSON</span>
                </button>
              </div>
            </div>

            {/* MODE A: SIMPLE MODE */}
            {settingsMode === "simple" && (
              <div className="space-y-4 pt-2 border-t border-[#282b34]">
                {/* 1. Response Style Presets */}
                <div className="space-y-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-[#595d69]">
                    1. Response Style
                  </div>
                  <div className="grid grid-cols-1 gap-1.5">
                    {[
                      {
                        id: "precise" as ResponseStyle,
                        title: "Precise & Factual",
                        desc: "Focused and predictable. Low variation, ideal for technical queries.",
                      },
                      {
                        id: "balanced" as ResponseStyle,
                        title: "Balanced & Natural",
                        desc: "Conversational tone with a natural balance of coherence and versatility.",
                      },
                      {
                        id: "creative" as ResponseStyle,
                        title: "Creative & Expressive",
                        desc: "Higher stylistic variety and open-ended exploration for ideation.",
                      },
                      {
                        id: "deterministic" as ResponseStyle,
                        title: "Deterministic & Strict",
                        desc: "Greedy sampling (0.0 temp) for reproducible code and strict JSON.",
                      },
                    ].map((item) => (
                      <button
                        key={item.id}
                        onClick={() => onSimpleStyleChange(item.id)}
                        className={`p-2 text-left rounded border transition-colors ${
                          simpleStyle === item.id
                            ? "bg-[#2563eb]/20 border-blue-500 text-white"
                            : "bg-[#1c1e24] border-[#282b34] text-[#a0a5b2] hover:bg-[#20232b] hover:text-white"
                        }`}
                      >
                        <div className="font-medium text-xs flex items-center justify-between">
                          <span>{item.title}</span>
                          {simpleStyle === item.id && <Check size={12} className="text-blue-400" />}
                        </div>
                        <p className="text-[10.5px] text-[#6c707d] mt-0.5 leading-relaxed">
                          {item.desc}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>

                {/* 2. Response Length */}
                <div className="space-y-1.5">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-[#595d69]">
                    2. Response Length
                  </div>
                  <div className="grid grid-cols-3 gap-1">
                    {[
                      { id: "concise" as ResponseLength, label: "Concise", hint: "~256 tokens" },
                      {
                        id: "standard" as ResponseLength,
                        label: "Standard",
                        hint: "~1,024 tokens",
                      },
                      {
                        id: "detailed" as ResponseLength,
                        label: "Detailed",
                        hint: "~2,048 tokens",
                      },
                    ].map((item) => (
                      <button
                        key={item.id}
                        onClick={() => onSimpleLengthChange(item.id)}
                        className={`py-1.5 px-1 text-center rounded border transition-colors ${
                          simpleLength === item.id
                            ? "bg-[#2563eb] border-blue-600 text-white font-medium"
                            : "bg-[#1c1e24] border-[#282b34] text-[#8c919e] hover:bg-[#20232b]"
                        }`}
                      >
                        <div className="text-xs">{item.label}</div>
                        <div className="text-[9.5px] opacity-75">{item.hint}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* 3. System Instructions */}
                <div className="space-y-1.5">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-[#595d69]">
                    3. System Instructions
                  </div>
                  <textarea
                    value={settings.systemPrompt}
                    onChange={(e) =>
                      onUpdateSettings({ ...settings, systemPrompt: e.target.value })
                    }
                    rows={4}
                    className="w-full bg-[#1c1e24] border border-[#282b34] focus:border-blue-500 rounded p-2 text-xs text-[#f1f2f4] placeholder-[#595d69] outline-none"
                    placeholder="Instructions for the AI assistant..."
                  />
                </div>
              </div>
            )}

            {/* MODE B: ADVANCED MODE */}
            {settingsMode === "advanced" && (
              <div className="space-y-3 pt-2 border-t border-[#282b34]">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-[#595d69]">
                  Sampling Parameters (llama.cpp)
                </div>

                <RangeControl
                  label="Temperature"
                  value={settings.temperature}
                  min={0}
                  max={2}
                  step={0.01}
                  onChange={(v) => onUpdateSettings({ ...settings, temperature: v })}
                  hint="Controls randomness. 0.0 is deterministic, 0.7 is balanced, 1.0+ is creative."
                />

                <RangeControl
                  label="Top-P (Nucleus)"
                  value={settings.topP}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={(v) => onUpdateSettings({ ...settings, topP: v })}
                  hint="Cumulative probability cutoff."
                />

                <RangeControl
                  label="Top-K"
                  value={settings.topK}
                  min={1}
                  max={256}
                  step={1}
                  onChange={(v) => onUpdateSettings({ ...settings, topK: v })}
                  hint="Limits sampling to K highest-probability tokens."
                />

                <RangeControl
                  label="Min-P"
                  value={settings.minP}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={(v) => onUpdateSettings({ ...settings, minP: v })}
                  hint="Minimum relative probability threshold against the highest token."
                />

                <RangeControl
                  label="Max Tokens"
                  value={settings.maxTokens}
                  min={32}
                  max={4096}
                  step={32}
                  onChange={(v) => onUpdateSettings({ ...settings, maxTokens: v })}
                  hint="Maximum output token generation limit."
                />

                <RangeControl
                  label="Repetition Penalty"
                  value={settings.repetitionPenalty}
                  min={0.8}
                  max={2.0}
                  step={0.01}
                  onChange={(v) => onUpdateSettings({ ...settings, repetitionPenalty: v })}
                  hint="Penalizes repeating recently evaluated tokens."
                />

                <RangeControl
                  label="Frequency Penalty"
                  value={settings.frequencyPenalty}
                  min={-2}
                  max={2}
                  step={0.05}
                  onChange={(v) => onUpdateSettings({ ...settings, frequencyPenalty: v })}
                  hint="Penalizes tokens based on their frequency in the response."
                />

                <RangeControl
                  label="Presence Penalty"
                  value={settings.presencePenalty}
                  min={-2}
                  max={2}
                  step={0.05}
                  onChange={(v) => onUpdateSettings({ ...settings, presencePenalty: v })}
                  hint="Penalizes tokens that have appeared at least once."
                />

                {/* Seed */}
                <div className="space-y-1 pt-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#a0a5b2]">Random Seed</span>
                    <span className="font-mono text-[11px] text-[#6c707d]">
                      {settings.seed === null ? "Random" : settings.seed}
                    </span>
                  </div>
                  <input
                    type="number"
                    value={settings.seed ?? ""}
                    onChange={(e) =>
                      onUpdateSettings({
                        ...settings,
                        seed: e.target.value ? parseInt(e.target.value) : null,
                      })
                    }
                    placeholder="Random (leave empty)"
                    className="w-full bg-[#1c1e24] border border-[#282b34] text-xs text-[#f1f2f4] p-1.5 rounded outline-none focus:border-blue-500 font-mono"
                  />
                </div>

                {/* System Prompt */}
                <div className="space-y-1 pt-1">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-[#595d69]">
                    System Prompt
                  </div>
                  <textarea
                    value={settings.systemPrompt}
                    onChange={(e) =>
                      onUpdateSettings({ ...settings, systemPrompt: e.target.value })
                    }
                    rows={3}
                    className="w-full bg-[#1c1e24] border border-[#282b34] focus:border-blue-500 rounded p-2 text-xs text-[#f1f2f4] placeholder-[#595d69] outline-none"
                    placeholder="System prompt..."
                  />
                </div>

                <button
                  onClick={() => onApplyPreset("general-balanced")}
                  className="w-full py-1.5 rounded bg-[#20232a] hover:bg-[#282c35] text-xs font-medium text-[#a0a5b2] hover:text-white transition-colors border border-[#282b34]"
                >
                  Reset to General Defaults
                </button>
              </div>
            )}
          </div>
        </aside>
      )}
    </div>
  );
}

// ==========================================
// Subcomponent: Range Control Slider
// ==========================================
function RangeControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  hint?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-[#a0a5b2]" title={hint}>
          {label}
        </span>
        <span className="font-mono text-[11px] text-[#f1f2f4] bg-[#20232a] px-1 py-0.5 rounded border border-[#282b34]">
          {step < 1 ? value.toFixed(2) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

// ==========================================
// Subcomponent: Empty Chat Hero
// ==========================================
function EmptyChatHero({
  isModelReady,
  loadedModel,
  modelStatus,
  onSelectPrompt,
  onSwitchToModels,
}: {
  isModelReady: boolean;
  loadedModel: string | null;
  modelStatus: ModelLoadingState;
  onSelectPrompt: (p: string) => void;
  onSwitchToModels: () => void;
}) {
  const suggestions = [
    {
      title: "Resilient Redis Cache Layer",
      prompt: "Propose a resilient Redis cache layer for high-throughput APIs in TypeScript.",
    },
    {
      title: "Speculative Decoding & KV Cache",
      prompt: "Explain how speculative decoding and KV cache quantization work in llama.cpp.",
    },
    {
      title: "Rust Async Worker Pool",
      prompt: "Write a complete, thread-safe asynchronous worker pool in Rust using Tokio.",
    },
    {
      title: "Strict JSON Schema Generator",
      prompt: "Draft a strict JSON schema for validating multi-tier user permissions.",
    },
  ];

  return (
    <div className="py-10 flex flex-col items-center text-center max-w-lg mx-auto space-y-4">
      <div className="w-12 h-12 rounded bg-[#1e2027] border border-[#282b34] flex items-center justify-center text-[#8c919e]">
        <Anvil size={23} />
      </div>

      <div className="space-y-1.5">
        <h2 className="text-sm font-semibold text-[#f1f2f4]">LM Forge</h2>

        {isModelReady && loadedModel ? (
          <p className="text-xs text-[#8c919e]">
            Ready with <strong className="text-emerald-400 font-mono">{loadedModel}</strong>. Zero
            data leaves your machine.
          </p>
        ) : modelStatus === "loading" ? (
          <div className="flex items-center justify-center gap-2 text-xs text-amber-400">
            <Loader2 size={13} className="animate-spin" />
            <span>Loading model into memory...</span>
          </div>
        ) : (
          <div className="p-3 bg-[#181a20] border border-[#282b34] rounded text-left space-y-2 mt-2">
            <div className="flex items-center gap-1.5 text-xs font-medium text-amber-400">
              <AlertTriangle size={13} />
              <span>No model loaded</span>
            </div>
            <p className="text-xs text-[#8c919e] leading-relaxed">
              Inference is disabled until a model is confirmed loaded by the backend. Select a GGUF
              model from your library to begin.
            </p>
            <button
              onClick={onSwitchToModels}
              className="px-3 py-1 bg-[#2563eb] hover:bg-[#1d4ed8] text-white rounded text-xs font-medium transition-colors"
            >
              Go to Models Library
            </button>
          </div>
        )}
      </div>

      {isModelReady && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 w-full pt-2">
          {suggestions.map((item) => (
            <button
              key={item.title}
              onClick={() => onSelectPrompt(item.prompt)}
              className="p-2.5 text-left rounded bg-[#181a20] hover:bg-[#20232b] border border-[#282b34] text-xs transition-colors group"
            >
              <div className="font-medium text-[#f1f2f4] group-hover:text-blue-400 transition-colors flex items-center justify-between">
                <span>{item.title}</span>
                <ArrowRight
                  size={11}
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                />
              </div>
              <p className="text-[11px] text-[#6c707d] mt-0.5 line-clamp-1">{item.prompt}</p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ==========================================
// Subcomponent: Message Bubble
// ==========================================
function MessageBubble({
  message,
  isLast,
  streaming,
  onRetry,
}: {
  message: ChatMessageItem;
  isLast: boolean;
  streaming?: boolean;
  onRetry?: () => void;
}) {
  const isUser = message.role === "user";
  const [thoughtOpen, setThoughtOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopyMessage = () => {
    if (!message.text) return;
    navigator.clipboard.writeText(message.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="w-6 h-6 rounded bg-[#20232a] border border-[#282b34] flex items-center justify-center text-[#8c919e] shrink-0 mt-0.5">
          <BrainCircuit size={13} />
        </div>
      )}

      <div className={`max-w-[85%] space-y-1.5 ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`rounded px-3 py-2 text-xs leading-relaxed ${
            isUser
              ? "bg-[#202636] border border-[#2d374d] text-[#f1f2f4]"
              : message.interrupted
                ? "bg-[#241c1a] border border-[#5c2e26] text-[#e8c8c2]"
                : "bg-[#181a20] border border-[#282b34] text-[#d4d6dc]"
          }`}
        >
          {message.thought && (
            <div className="mb-2">
              <button
                onClick={() => setThoughtOpen((v) => !v)}
                className="flex items-center gap-1 text-[11px] text-[#8c919e] font-mono hover:text-white transition-colors"
              >
                <ChevronRight size={11} className={thoughtOpen ? "rotate-90" : ""} />
                <span>Reasoning Process</span>
              </button>
              {thoughtOpen && (
                <div className="mt-1 p-2 rounded bg-[#121316] border border-[#282b34] text-[11px] font-mono text-[#8c919e] whitespace-pre-wrap">
                  {message.thought}
                </div>
              )}
            </div>
          )}

          <FormattedMessage content={message.text} />

          {message.interrupted && (
            <div className="flex items-center gap-2 pt-2 mt-2 border-t border-[#4a241e]">
              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-amber-950/40 text-amber-300 border border-amber-800/50 flex items-center gap-1">
                <AlertTriangle size={10} />
                <span>Interrupted / Incomplete</span>
              </span>
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="flex items-center gap-1 px-2 py-0.5 rounded bg-[#20232a] hover:bg-[#282c35] text-xs text-[#a0a5b2] hover:text-white border border-[#282b34] transition-colors"
                >
                  <RotateCcw size={11} />
                  <span>Retry Generation</span>
                </button>
              )}
            </div>
          )}
        </div>

        {!isUser && (
          <div className="flex items-center justify-between gap-2 px-1">
            <div className="flex items-center gap-2 text-[10px] font-mono text-[#595d69]">
              {message.stats?.tokPerSec && <span>{message.stats.tokPerSec} tok/s</span>}
              {message.stats?.latencyMs && (
                <>
                  <span>·</span>
                  <span>{message.stats.latencyMs}ms</span>
                </>
              )}
              {message.stats?.tokens && (
                <>
                  <span>·</span>
                  <span>{message.stats.tokens} tokens</span>
                </>
              )}
            </div>

            {message.text && (
              <button
                onClick={handleCopyMessage}
                className="flex items-center gap-1 text-[10.5px] font-mono text-[#6c707d] hover:text-[#f1f2f4] transition-colors py-0.5 px-1.5 rounded hover:bg-[#20232a]"
                title="Copy response"
              >
                {copied ? (
                  <>
                    <Check size={11} className="text-emerald-400" />
                    <span className="text-emerald-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy size={11} />
                    <span>Copy</span>
                  </>
                )}
              </button>
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-6 h-6 rounded bg-[#2563eb] flex items-center justify-center text-white shrink-0 mt-0.5 font-bold text-[10px]">
          U
        </div>
      )}
    </div>
  );
}

function renderInlineFormatting(text: string): React.ReactNode {
  // Split on bold (**text**), inline code (`code`), or italic (*text*)
  const tokens = text.split(/(\*\*[^*]+?\*\*|`[^`]+?`|\*[^*]+?\*)/g);
  return tokens.map((token, i) => {
    if (token.startsWith("**") && token.endsWith("**") && token.length > 4) {
      return (
        <strong key={i} className="font-semibold text-white">
          {token.slice(2, -2)}
        </strong>
      );
    }
    if (token.startsWith("`") && token.endsWith("`") && token.length > 2) {
      return (
        <code
          key={i}
          className="px-1 py-0.5 rounded bg-[#131519] border border-[#2b2e38] font-mono text-[11px] text-amber-300"
        >
          {token.slice(1, -1)}
        </code>
      );
    }
    if (token.startsWith("*") && token.endsWith("*") && token.length > 2) {
      return <em key={i} className="italic text-[#e1e4ea]">{token.slice(1, -1)}</em>;
    }
    return token;
  });
}

function FormattedMessage({ content }: { content: string }) {
  if (!content) return null;

  const parts = content.split(/(```[\s\S]*?```)/g);

  return (
    <div className="space-y-2">
      {parts.map((part, index) => {
        if (part.startsWith("```") && part.endsWith("```")) {
          const lines = part.slice(3, -3).trim().split("\n");
          const language = lines[0]?.trim() || "text";
          const code = lines.length > 1 ? lines.slice(1).join("\n") : lines[0] || "";
          return <CodeCard key={index} code={code} language={language} />;
        }

        const lines = part.split("\n");
        return (
          <div key={index} className="space-y-1 leading-relaxed">
            {lines.map((line, lIdx) => {
              if (!line.trim()) {
                return <div key={lIdx} className="h-1.5" />;
              }
              return (
                <div key={lIdx} className="break-words">
                  {renderInlineFormatting(line)}
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

function CodeCard({ code, language }: { code: string; language: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded overflow-hidden border border-[#282b34] bg-[#121316] my-2 text-xs">
      <div className="flex items-center justify-between px-3 py-1 bg-[#181a20] border-b border-[#282b34] text-[#8c919e]">
        <span className="font-mono text-[10px] uppercase">{language}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-[11px] text-[#8c919e] hover:text-white transition-colors"
        >
          {copied ? <Check size={11} className="text-emerald-400" /> : <Copy size={11} />}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre className="p-3 overflow-x-auto font-mono text-[#d4d6dc] text-[11.5px] leading-5 custom-scrollbar">
        <code>{code}</code>
      </pre>
    </div>
  );
}

// ==========================================
// View: My Models View
// ==========================================
function ModelsView({
  models,
  backendConnected,
  modelStatus,
  loadedModel,
  loadingError,
  activeOperationModel,
  onDismissError,
  onLoadModel,
  onUnloadModel,
  onRefresh,
  onOpenInChat,
  onNavigateDiscover,
}: {
  models: ModelInfo[];
  backendConnected: boolean;
  modelStatus: ModelLoadingState;
  loadedModel: string | null;
  loadingError: string | null;
  activeOperationModel: string | null;
  onDismissError: () => void;
  onLoadModel: (path: string) => void;
  onUnloadModel: () => void;
  onRefresh: () => void;
  onOpenInChat: (modelId: string) => void;
  onNavigateDiscover: () => void;
}) {
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<"name" | "size">("size");
  const [importPath, setImportPath] = useState("");
  const [importOpen, setImportOpen] = useState(false);
  const [detailedModels, setDetailedModels] = useState<LocalModelDetailedItem[]>([]);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameInput, setRenameInput] = useState("");
  const [benchmarkingId, setBenchmarkingId] = useState<string | null>(null);
  const [benchResults, setBenchResults] = useState<Record<string, { tokPerSec: number; elapsed: number }>>({});

  const handleBenchmark = async (model: ModelInfo) => {
    setBenchmarkingId(model.id);
    try {
      const res = await invoke<{ tokens_per_second: number; total_elapsed_seconds: number }>(
        "run_model_benchmark",
        { path: model.path, promptTokens: 256, genTokens: 64, threads: 4 }
      );
      if (res) {
        setBenchResults((prev) => ({
          ...prev,
          [model.id]: { tokPerSec: res.tokens_per_second, elapsed: res.total_elapsed_seconds },
        }));
      }
    } catch (err) {
      console.error("Benchmark failed:", err);
      alert(`Benchmark failed: ${err}`);
    } finally {
      setBenchmarkingId(null);
    }
  };

  const refreshDetailed = async () => {
    try {
      const items = await invoke<LocalModelDetailedItem[]>("get_models_detailed");
      setDetailedModels(items || []);
    } catch {
      // Fallback to basic models
    }
  };

  useEffect(() => {
    refreshDetailed();
  }, [models]);

  const filtered = useMemo(() => {
    return models
      .filter((m) => m.id.toLowerCase().includes(search.toLowerCase()))
      .sort((a, b) => (sortBy === "size" ? b.size - a.size : a.id.localeCompare(b.id)));
  }, [models, search, sortBy]);

  const totalBytes = models.reduce((acc, m) => acc + m.size, 0);

  const handleImport = async () => {
    const src = importPath.trim();
    if (!src) return;
    try {
      await invoke("import_external_model", { sourcePath: src });
      onRefresh();
      setImportPath("");
      setImportOpen(false);
    } catch (err: unknown) {
      alert(typeof err === "string" ? err : (err as Error)?.message || "Import failed");
    }
  };

  const handleDeleteModel = async (filename: string) => {
    if (!window.confirm(`Permanently delete '${filename}' from models/ directory? This cannot be undone.`)) return;
    try {
      await invoke("delete_local_model", { filename });
      onRefresh();
    } catch (err: unknown) {
      alert(typeof err === "string" ? err : (err as Error)?.message || "Delete failed");
    }
  };

  const handleStartRename = (filename: string) => {
    setRenamingId(filename);
    setRenameInput(filename.replace(/\.gguf$/i, ""));
  };

  const handleConfirmRename = async (oldFilename: string) => {
    if (!renameInput.trim()) {
      setRenamingId(null);
      return;
    }
    try {
      await invoke("rename_local_model", { oldFilename, newFilename: renameInput.trim() });
      setRenamingId(null);
      onRefresh();
    } catch (err: unknown) {
      alert(typeof err === "string" ? err : (err as Error)?.message || "Rename failed");
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto p-5 space-y-4 custom-scrollbar bg-[#121316] max-w-5xl mx-auto w-full">
      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b border-[#282b34]">
        <div>
          <h1 className="text-base font-bold text-[#f1f2f4]">Local Models Library</h1>
          <p className="text-xs text-[#8c919e]">
            Manage GGUF models stored in your workstation cache directory.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setImportOpen((v) => !v)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#20232a] hover:bg-[#282c35] border border-[#282b34] text-xs text-[#f1f2f4] transition-colors"
          >
            <FileCode size={12} className="text-blue-400" />
            <span>Import GGUF</span>
          </button>
          <button
            onClick={onNavigateDiscover}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#2563eb] hover:bg-[#1d4ed8] text-xs text-white font-medium transition-colors"
          >
            <Download size={12} />
            <span>Discover GGUFs</span>
          </button>
          <button
            onClick={onRefresh}
            className="flex items-center gap-1 px-2 py-1 rounded bg-[#1c1e24] hover:bg-[#23262e] border border-[#282b34] text-xs text-[#a0a5b2] hover:text-white transition-colors"
            title="Rescan models directory"
          >
            <RefreshCw size={12} />
          </button>
        </div>
      </div>

      {/* Manual Import Drawer */}
      {importOpen && (
        <div className="p-3.5 rounded bg-[#181a20] border border-[#282b34] space-y-2">
          <div className="text-xs font-semibold text-[#f1f2f4]">Import External GGUF Model</div>
          <p className="text-[11px] text-[#8c919e]">
            Enter the absolute path to an external .gguf file. It will be copied into your workstation <code>models/</code> directory:
          </p>
          <div className="flex gap-2">
            <input
              type="text"
              value={importPath}
              onChange={(e) => setImportPath(e.target.value)}
              placeholder="/home/user/Downloads/model-q4_k_m.gguf"
              className="flex-1 bg-[#121316] border border-[#282b34] text-xs text-[#f1f2f4] px-2.5 py-1 rounded outline-none font-mono focus:border-blue-500"
            />
            <button
              onClick={handleImport}
              className="px-3 py-1 rounded bg-[#2563eb] hover:bg-[#1d4ed8] text-white text-xs font-medium"
            >
              Copy & Index
            </button>
          </div>
        </div>
      )}

      {/* Error Notice */}
      {loadingError && (
        <div className="p-3 rounded bg-[#2a1417] border border-[#5c1d24] flex items-center justify-between text-xs text-red-200">
          <div className="flex items-center gap-2">
            <AlertCircle size={15} className="text-red-400 shrink-0" />
            <span>
              <strong>Load Error:</strong> {loadingError}
            </span>
          </div>
          <button
            onClick={onDismissError}
            className="text-red-400 hover:text-red-200 text-xs px-2 py-0.5 rounded hover:bg-[#3d181e]"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Storage Information Card */}
      <div className="p-3.5 rounded bg-[#181a20] border border-[#282b34] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <HardDrive size={18} className="text-[#8c919e]" />
          <div>
            <div className="text-xs font-semibold text-[#f1f2f4]">
              {formatBytes(totalBytes)} across {models.length} model file(s)
            </div>
            <div className="font-mono text-[11px] text-[#595d69]">
              Location: /home/leywin/Projects/lm-studio/models
            </div>
          </div>
        </div>

        <div className="text-right">
          <div className="text-xs text-[#8c919e]">
            Status:{" "}
            {modelStatus === "loaded" && loadedModel ? (
              <span className="text-emerald-400 font-medium">1 Model Loaded</span>
            ) : modelStatus === "loading" ? (
              <span className="text-amber-400 font-medium">Loading...</span>
            ) : (
              <span className="text-[#6c707d]">No Model In Memory</span>
            )}
          </div>
        </div>
      </div>

      {/* Filter and Sort Toolbar */}
      <div className="flex items-center justify-between gap-3">
        <div className="relative flex-1 max-w-xs">
          <Search size={12} className="absolute left-2.5 top-2 text-[#595d69]" />
          <input
            type="text"
            placeholder="Search local models..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-[#181a20] border border-[#282b34] text-xs text-[#f1f2f4] pl-7 pr-2.5 py-1 rounded outline-none focus:border-blue-500 transition-colors"
          />
        </div>

        <div className="flex items-center gap-2 text-xs text-[#8c919e]">
          <span>Sort:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as "name" | "size")}
            className="bg-[#181a20] border border-[#282b34] text-[#f1f2f4] px-2 py-1 rounded outline-none cursor-pointer"
          >
            <option value="size">File Size</option>
            <option value="name">Name</option>
          </select>
        </div>
      </div>

      {/* Models Table */}
      <div className="rounded border border-[#282b34] bg-[#16181d] overflow-hidden">
        <div className="grid grid-cols-12 px-3 py-2 bg-[#141519] border-b border-[#282b34] text-[10.5px] font-semibold uppercase tracking-wider text-[#6c707d]">
          <div className="col-span-5">Model Filename</div>
          <div className="col-span-2">Quantization</div>
          <div className="col-span-2">File Size</div>
          <div className="col-span-3 text-right">Actions</div>
        </div>

        <div className="divide-y divide-[#242730]">
          {filtered.length === 0 ? (
            <div className="p-6 text-center text-[#6c707d] text-xs">
              No GGUF models found in models/ directory. Use Hugging Face Discovery to download.
            </div>
          ) : (
            filtered.map((m) => {
              const isThisLoaded = loadedModel === m.id || loadedModel === m.path.split("/").pop();
              const isThisLoading = activeOperationModel === m.id && modelStatus === "loading";
              const { cleanName, quant } = parseModelDetails(m.id);
              const detailed = detailedModels.find((d) => d.filename === m.id);

              return (
                <div
                  key={m.id}
                  className={`grid grid-cols-12 px-3 py-2.5 items-center text-xs transition-colors ${
                    isThisLoaded ? "bg-[#18231e]" : "hover:bg-[#1a1c22]"
                  }`}
                >
                  <div className="col-span-5 truncate pr-2">
                    {renamingId === m.id ? (
                      <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="text"
                          value={renameInput}
                          onChange={(e) => setRenameInput(e.target.value)}
                          className="bg-[#121316] border border-blue-500 text-xs text-[#f1f2f4] px-1.5 py-0.5 rounded outline-none font-mono"
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleConfirmRename(m.id);
                            if (e.key === "Escape") setRenamingId(null);
                          }}
                        />
                        <button
                          onClick={() => handleConfirmRename(m.id)}
                          className="p-1 rounded text-emerald-400 hover:text-emerald-300"
                          title="Save Rename"
                        >
                          <Check size={12} />
                        </button>
                        <button
                          onClick={() => setRenamingId(null)}
                          className="p-1 rounded text-[#8c919e] hover:text-white"
                          title="Cancel"
                        >
                          <X size={12} />
                        </button>
                      </div>
                    ) : (
                      <>
                        <div className="font-medium text-[#f1f2f4] truncate flex items-center gap-1.5">
                          <span>{detailed?.display_name || cleanName}</span>
                          {detailed?.is_split && (
                            <span className="px-1.5 py-0.2 rounded text-[9.5px] font-mono bg-purple-950/80 text-purple-300 border border-purple-800">
                              Part {detailed.split_part}/{detailed.split_total}
                            </span>
                          )}
                        </div>
                        <div className="font-mono text-[10.5px] text-[#6c707d] truncate">{m.id}</div>
                      </>
                    )}
                  </div>

                  <div className="col-span-2">
                    <span className="font-mono text-[10.5px] bg-[#20232a] text-[#a0a5b2] px-1.5 py-0.5 rounded border border-[#282b34]">
                      {detailed?.quant || quant}
                    </span>
                  </div>

                  <div className="col-span-2 font-mono text-[#8c919e]">
                    <div>{formatBytes(m.size)}</div>
                    {benchResults[m.id] && (
                      <div className="text-[10px] text-emerald-400 font-semibold">
                        {benchResults[m.id].tokPerSec} t/s
                      </div>
                    )}
                  </div>

                  <div className="col-span-3 flex items-center justify-end gap-1.5">
                    {/* Benchmark Button */}
                    <button
                      onClick={() => handleBenchmark(m)}
                      disabled={benchmarkingId !== null || modelStatus === "loading"}
                      className={`px-2 py-0.5 rounded text-[11px] font-mono border transition-colors flex items-center gap-1 ${
                        benchmarkingId === m.id
                          ? "bg-amber-950/40 border-amber-700/60 text-amber-300"
                          : "bg-[#181a20] hover:bg-[#22252e] text-[#8c919e] hover:text-[#f1f2f4] border-[#2b2e38]"
                      }`}
                      title="Run quick 64-token speed test on this model"
                    >
                      {benchmarkingId === m.id ? (
                        <>
                          <Loader2 size={10} className="animate-spin text-amber-400" />
                          <span>Bench...</span>
                        </>
                      ) : (
                        <>
                          <Zap size={10} className="text-amber-400" />
                          <span>Bench</span>
                        </>
                      )}
                    </button>
                    {renamingId !== m.id && (
                      <button
                        onClick={() => handleStartRename(m.id)}
                        className="p-1 rounded text-[#6c707d] hover:text-[#f1f2f4] hover:bg-[#20232a] transition-colors"
                        title="Rename model"
                      >
                        <Pencil size={11} />
                      </button>
                    )}

                    {!isThisLoaded && (
                      <button
                        onClick={() => handleDeleteModel(m.id)}
                        className="p-1 rounded text-[#6c707d] hover:text-red-400 hover:bg-[#20232a] transition-colors"
                        title="Delete from disk"
                      >
                        <Trash2 size={11} />
                      </button>
                    )}

                    {isThisLoaded ? (
                      <>
                        <button
                          onClick={() => onOpenInChat(m.id)}
                          className="px-2.5 py-1 rounded bg-[#2563eb] hover:bg-[#1d4ed8] text-white font-medium text-xs transition-colors"
                        >
                          Chat
                        </button>
                        <button
                          onClick={onUnloadModel}
                          className="px-2.5 py-1 rounded bg-[#2a171a] hover:bg-[#3d1c22] text-red-300 border border-[#5c212a] font-medium text-xs transition-colors"
                        >
                          Unload
                        </button>
                      </>
                    ) : isThisLoading ? (
                      <button
                        disabled
                        className="flex items-center gap-1 px-2.5 py-1 rounded bg-[#2a2414] text-amber-300 border border-[#5c4a1d] text-xs font-medium cursor-not-allowed"
                      >
                        <Loader2 size={11} className="animate-spin" />
                        <span>Loading...</span>
                      </button>
                    ) : (
                      <button
                        onClick={() => onLoadModel(m.path)}
                        disabled={modelStatus === "loading" || modelStatus === "unloading"}
                        className="flex items-center gap-1 px-2.5 py-1 rounded bg-[#20232a] hover:bg-[#282c35] text-[#f1f2f4] border border-[#282b34] text-xs font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        <Play size={10} />
                        <span>Load</span>
                      </button>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

// ==========================================
// View: Local Server View
// ==========================================
function ServerView({
  serverOn,
  setServerOn,
  backendConnected,
  modelStatus,
  loadedModel,
  onUnloadModel,
}: {
  serverOn: boolean;
  setServerOn: (v: boolean) => void;
  backendConnected: boolean;
  modelStatus: ModelLoadingState;
  loadedModel: string | null;
  onUnloadModel: () => void;
}) {
  const [codeTab, setCodeTab] = useState<"cURL" | "Python" | "Node.js">("cURL");
  const [logs, setLogs] = useState<RequestLog[]>(initialServerLogs);
  const [expandedLog, setExpandedLog] = useState<string | null>("log-1");
  const [copiedUrl, setCopiedUrl] = useState(false);

  const endpoint = "http://127.0.0.1:1234/v1";

  const handleCopyEndpoint = () => {
    navigator.clipboard.writeText(endpoint);
    setCopiedUrl(true);
    setTimeout(() => setCopiedUrl(false), 2000);
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto p-5 space-y-4 custom-scrollbar bg-[#121316] max-w-5xl mx-auto w-full">
      <div className="p-4 rounded bg-[#181a20] border border-[#282b34] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Server size={20} className={serverOn ? "text-emerald-400" : "text-[#6c707d]"} />
          <div>
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${serverOn ? "bg-emerald-500" : "bg-zinc-600"}`}
              />
              <h2 className="text-sm font-semibold text-[#f1f2f4]">
                {serverOn ? "Local OpenAI-Compatible Server Active" : "Server Stopped"}
              </h2>
            </div>
            <div className="flex items-center gap-2 mt-1">
              <code className="text-xs font-mono text-[#a0a5b2]">{endpoint}</code>
              <button
                onClick={handleCopyEndpoint}
                className="text-[#6c707d] hover:text-white transition-colors"
                title="Copy base URL"
              >
                {copiedUrl ? <Check size={11} className="text-emerald-400" /> : <Copy size={11} />}
              </button>
            </div>
          </div>
        </div>

        <button
          onClick={() => setServerOn(!serverOn)}
          className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
            serverOn
              ? "bg-[#2a171a] hover:bg-[#3d1c22] text-red-300 border border-[#5c212a]"
              : "bg-[#2563eb] hover:bg-[#1d4ed8] text-white"
          }`}
        >
          {serverOn ? "Stop Server" : "Start Server"}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
        <div className="md:col-span-5 space-y-4">
          <div className="p-3 rounded bg-[#181a20] border border-[#282b34] space-y-2">
            <span className="text-xs font-semibold text-[#f1f2f4]">Engine Status</span>
            <div className="p-2.5 rounded bg-[#121316] border border-[#282b34] flex items-center justify-between text-xs">
              <div className="truncate pr-2">
                <div className="text-[#a0a5b2] text-[11px]">Loaded Model</div>
                <div className="font-semibold text-white truncate">
                  {modelStatus === "loaded" && loadedModel ? loadedModel : "None"}
                </div>
              </div>
              {modelStatus === "loaded" && (
                <button
                  onClick={onUnloadModel}
                  className="px-2 py-0.5 rounded bg-[#20232a] hover:bg-[#282c35] text-red-300 text-xs"
                >
                  Unload
                </button>
              )}
            </div>
          </div>

          <div className="rounded border border-[#282b34] bg-[#181a20] overflow-hidden">
            <div className="px-3 py-1.5 border-b border-[#282b34] flex items-center justify-between bg-[#141519]">
              <span className="text-xs font-semibold text-[#f1f2f4]">Client Snippets</span>
              <div className="flex gap-1">
                {(["cURL", "Python", "Node.js"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setCodeTab(tab)}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                      codeTab === tab
                        ? "bg-[#2563eb] text-white"
                        : "text-[#6c707d] hover:text-white"
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>
            </div>

            <div className="p-3 bg-[#121316]">
              <pre className="font-mono text-[11px] text-[#d4d6dc] leading-relaxed overflow-x-auto custom-scrollbar">
                {codeTab === "cURL" && (
                  <code>{`curl http://127.0.0.1:1234/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"default",
  "messages":[{"role":"user","content":"Hello!"}]}'`}</code>
                )}
                {codeTab === "Python" && (
                  <code>{`from openai import OpenAI

client = OpenAI(
  base_url="http://127.0.0.1:1234/v1",
  api_key="lm-studio"
)

res = client.chat.completions.create(
  model="default",
  messages=[{"role":"user","content":"Hello!"}]
)`}</code>
                )}
                {codeTab === "Node.js" && (
                  <code>{`import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:1234/v1",
  apiKey: "lm-studio"
});

const res = await client.chat.completions.create({
  model: "default",
  messages: [{ role: "user", content: "Hello!" }]
});`}</code>
                )}
              </pre>
            </div>
          </div>
        </div>

        <div className="md:col-span-7 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[#f1f2f4]">HTTP Request Log</span>
            <button
              onClick={() => setLogs([])}
              className="text-xs text-[#6c707d] hover:text-white transition-colors"
            >
              Clear
            </button>
          </div>

          <div className="rounded border border-[#282b34] bg-[#181a20] overflow-hidden divide-y divide-[#242730]">
            {logs.length === 0 ? (
              <div className="p-4 text-center text-[#6c707d] text-xs">
                No requests received yet.
              </div>
            ) : (
              logs.map((log) => {
                const isExpanded = expandedLog === log.id;
                return (
                  <div key={log.id} className="text-xs">
                    <div
                      onClick={() => setExpandedLog(isExpanded ? null : log.id)}
                      className="p-2.5 flex items-center justify-between cursor-pointer hover:bg-[#1e2027] transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[10.5px] text-[#6c707d]">{log.time}</span>
                        <span className="px-1 py-0.2 rounded font-mono text-[9.5px] bg-[#14532d] text-[#4ade80]">
                          {log.method}
                        </span>
                        <code className="text-[#a0a5b2] font-mono">{log.path}</code>
                      </div>

                      <div className="flex items-center gap-2 text-[#6c707d] font-mono text-[10.5px]">
                        <span className="text-emerald-400">{log.status}</span>
                        <span>{log.latency}</span>
                      </div>
                    </div>

                    {isExpanded && log.requestJson && (
                      <div className="p-2.5 bg-[#121316] border-t border-[#282b34]">
                        <pre className="font-mono text-[10.5px] text-[#a0a5b2] overflow-x-auto">
                          {log.requestJson}
                        </pre>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ==========================================
// View: Discover View (Hugging Face Integration)
// ==========================================
function DiscoverView({
  onDownloadModel,
  activeDownload,
  onSwitchToModels,
  hfToken,
}: {
  onDownloadModel: (modelId: string, filename: string) => Promise<void>;
  activeDownload: {
    modelId: string;
    filename: string;
    status: DownloadState;
    downloadedBytes?: number;
    totalBytes?: number;
    percent?: number;
    speedBytesSec?: number;
    error?: string;
  } | null;
  onSwitchToModels: () => void;
  hfToken?: string;
}) {
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<HfModelItem[]>(curatedHfModels);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Hugging Face filter parameters
  const [pipelineTag, setPipelineTag] = useState("all");
  const [sortBy, setSortBy] = useState("downloads");
  const [modelSize, setModelSize] = useState("all");
  const [customTag, setCustomTag] = useState("");
  const [authorFilter, setAuthorFilter] = useState("");
  const [showFilters, setShowFilters] = useState(false);

  // Selected model for file customization
  const [selectedRepo, setSelectedRepo] = useState<HfModelItem | null>(null);
  const [repoFiles, setRepoFiles] = useState<HfRepoFileItem[]>([]);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [fileFetchError, setFileFetchError] = useState<string | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string>("");

  // Machine specs for quant recommendations
  const [machineSpecs, setMachineSpecs] = useState<{
    total_ram_bytes: number;
    available_ram_bytes: number;
    cpu_model: string;
    physical_cores: number;
    logical_cores: number;
  } | null>(null);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const q = query.trim();

    setSearching(true);
    setSearchError(null);

    try {
      // Build search query incorporating size if selected
      const queryParts = [q];
      if (modelSize !== "all" && !q.toLowerCase().includes(modelSize.toLowerCase())) {
        queryParts.push(modelSize);
      }
      const combinedQuery = queryParts.filter(Boolean).join(" ");

      // Build filter tags (gguf + custom tag if present)
      const tags: string[] = ["gguf"];
      if (customTag.trim()) {
        tags.push(customTag.trim());
      }

      const results = await invoke<HfModelItem[]>("hf_search", {
        query: combinedQuery,
        filter: tags.join(","),
        pipelineTag: pipelineTag !== "all" ? pipelineTag : null,
        author: authorFilter.trim() || null,
        sort: sortBy,
        limit: 28,
      });

      // Filter locally by model size regex if specified
      let filtered = results || [];
      if (modelSize !== "all" && filtered.length > 0) {
        const sizeRegex = new RegExp(`\\b${modelSize}\\b|[-_]${modelSize}[-_]`, "i");
        const strictMatches = filtered.filter(
          (m) => sizeRegex.test(m.id) || (m.tags && m.tags.some((t) => sizeRegex.test(t)))
        );
        if (strictMatches.length > 0) {
          filtered = strictMatches;
        }
      }

      if (filtered.length > 0) {
        setSearchResults(filtered);
      } else if (!q && pipelineTag === "all" && !customTag && !authorFilter && modelSize === "all") {
        setSearchResults(curatedHfModels);
      } else {
        setSearchResults([]);
        setSearchError("No models found matching the selected filters.");
      }
    } catch (err: unknown) {
      // Fallback search in curated list
      let matched = curatedHfModels;
      if (q) {
        matched = matched.filter((m) => m.id.toLowerCase().includes(q.toLowerCase()));
      }
      if (modelSize !== "all") {
        matched = matched.filter((m) => m.id.toLowerCase().includes(modelSize.toLowerCase()));
      }
      if (authorFilter.trim()) {
        matched = matched.filter((m) => m.id.toLowerCase().startsWith(authorFilter.trim().toLowerCase()));
      }
      setSearchResults(matched);
      setSearchError("Offline or network limited. Showing verified cached models.");
    } finally {
      setSearching(false);
    }
  };

  // Returns "best" | "ok" | "toobig" | "unknown" for a given file size vs available RAM
  const getFileFit = (fileSizeBytes: number | undefined): "best" | "ok" | "toobig" | "unknown" => {
    if (!fileSizeBytes || !machineSpecs) return "unknown";
    const avail = machineSpecs.available_ram_bytes;
    const needed = fileSizeBytes * 1.3; // model + ~30% context buffer
    if (needed > avail) return "toobig";
    if (needed > avail * 0.75) return "ok";
    return "best";
  };

  const fitBadge = (fit: ReturnType<typeof getFileFit>) => {
    if (fit === "best") return <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-900/40 text-emerald-400 shrink-0">✓ fits</span>;
    if (fit === "ok") return <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-900/40 text-amber-400 shrink-0">~ tight</span>;
    if (fit === "toobig") return <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-red-900/40 text-red-400 shrink-0">✗ too big</span>;
    return null;
  };

  const handleOpenDownloadModal = async (repo: HfModelItem) => {
    setSelectedRepo(repo);
    setRepoFiles([]);
    setFileFetchError(null);
    setSelectedFileName("");
    setLoadingFiles(true);

    // Fetch machine specs once per modal open
    try {
      const specs = await invoke<typeof machineSpecs>("get_machine_specs");
      setMachineSpecs(specs);
    } catch { /* non-critical */ }

    try {
      const files = await invoke<HfRepoFileItem[]>("hf_model_files", {
        modelId: repo.id,
        token: hfToken || null,
      });

      setRepoFiles(files || []);
      if (files && files.length > 0) {
        // Pick recommended Q4_K_M or first file
        const recommended = files.find((f) => f.path.toLowerCase().includes("q4_k_m")) || files[0];
        setSelectedFileName(recommended.path.split("/").pop() || recommended.path);
      } else {
        const repoBase = repo.id.split("/").pop() || "model";
        setSelectedFileName(`${repoBase.toLowerCase()}-q4_k_m.gguf`);
      }
    } catch (err: unknown) {
      const msg = typeof err === "string" ? err : (err as Error)?.message || "Failed to load repo files.";
      setFileFetchError(msg);
      const repoBase = repo.id.split("/").pop() || "model";
      setSelectedFileName(`${repoBase.toLowerCase()}-q4_k_m.gguf`);
    } finally {
      setLoadingFiles(false);
    }
  };

  const handleConfirmDownload = async () => {
    if (!selectedRepo) return;
    const file = selectedFileName.trim();
    if (!file) return;
    await onDownloadModel(selectedRepo.id, file);
    setSelectedRepo(null);
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto p-5 space-y-4 custom-scrollbar bg-[#121316] max-w-5xl mx-auto w-full">
      <div className="flex items-center justify-between pb-2 border-b border-[#282b34]">
        <div>
          <h1 className="text-base font-bold text-[#f1f2f4]">Hugging Face GGUF Discovery</h1>
          <p className="text-xs text-[#8c919e]">
            Search Hugging Face for quantized weights and download directly to your local
            workstation.
          </p>
        </div>

        <button
          onClick={onSwitchToModels}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#20232a] hover:bg-[#282c35] border border-[#282b34] text-xs text-[#a0a5b2] hover:text-white transition-colors"
        >
          <Library size={12} />
          <span>My Models Library</span>
        </button>
      </div>

      {searchError && (
        <div className="p-2.5 rounded bg-[#20222a] border border-[#303442] flex items-center gap-2 text-xs text-amber-300">
          <Info size={14} className="shrink-0" />
          <span>{searchError}</span>
        </div>
      )}

      {/* Search Input Form */}
      <form onSubmit={handleSearch} className="space-y-2">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search size={13} className="absolute left-3 top-2.5 text-[#595d69]" />
            <input
              type="text"
              placeholder="Search Hugging Face models (e.g. Qwen2.5, DeepSeek-R1, Llama-3.2, Mistral)..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full bg-[#181a20] border border-[#282b34] text-xs text-[#f1f2f4] pl-8 pr-3 py-1.5 rounded outline-none focus:border-blue-500 transition-colors"
            />
          </div>
          <button
            type="button"
            onClick={() => setShowFilters(!showFilters)}
            className={`px-2.5 py-1.5 rounded border text-xs font-medium transition-colors flex items-center gap-1.5 ${
              showFilters || pipelineTag !== "all" || sortBy !== "downloads" || modelSize !== "all" || customTag || authorFilter
                ? "bg-[#2563eb]/20 border-blue-500/50 text-blue-400"
                : "bg-[#181a20] border-[#282b34] text-[#8c919e] hover:text-[#f1f2f4]"
            }`}
            title="Toggle HF Filters"
          >
            <SlidersHorizontal size={13} />
            <span>Filters</span>
          </button>
          <button
            type="submit"
            disabled={searching}
            className="px-3 py-1.5 rounded bg-[#2563eb] hover:bg-[#1d4ed8] disabled:opacity-50 text-white text-xs font-medium transition-colors flex items-center gap-1.5"
          >
            {searching ? <Loader2 size={13} className="animate-spin" /> : <Search size={13} />}
            <span>Search</span>
          </button>
        </div>

        {/* Hugging Face Filters Bar */}
        {showFilters && (
          <div className="p-3 rounded bg-[#181a20] border border-[#282b34] grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3 text-xs">
            {/* Model Size Filter */}
            <div className="space-y-1">
              <label className="text-[11px] font-medium text-[#8c919e]">Model Size</label>
              <select
                value={modelSize}
                onChange={(e) => setModelSize(e.target.value)}
                className="w-full bg-[#141519] border border-[#282b34] text-xs text-[#f1f2f4] px-2 py-1.5 rounded outline-none focus:border-blue-500"
              >
                <option value="all">All Sizes</option>
                <option value="1B">~1B (Ultra light)</option>
                <option value="3B">~3B (Mobile / Low RAM)</option>
                <option value="4B">~4B (Fast compact)</option>
                <option value="7B">~7B (Standard)</option>
                <option value="8B">~8B (Standard+)</option>
                <option value="9B">~9B (Gemma / Qwen)</option>
                <option value="12B">~12B (Mistral NeMo)</option>
                <option value="14B">~14B (High accuracy)</option>
                <option value="27B">~27B (Large)</option>
                <option value="32B">~32B (Qwen / DeepSeek)</option>
                <option value="70B">~70B (Max size)</option>
              </select>
            </div>

            {/* Pipeline / Task Filter */}
            <div className="space-y-1">
              <label className="text-[11px] font-medium text-[#8c919e]">Task / Pipeline</label>
              <select
                value={pipelineTag}
                onChange={(e) => setPipelineTag(e.target.value)}
                className="w-full bg-[#141519] border border-[#282b34] text-xs text-[#f1f2f4] px-2 py-1.5 rounded outline-none focus:border-blue-500"
              >
                <option value="all">All Tasks</option>
                <option value="text-generation">Text Generation</option>
                <option value="conversational">Conversational</option>
                <option value="text-classification">Text Classification</option>
                <option value="image-text-to-text">Vision / Multimodal</option>
                <option value="feature-extraction">Embeddings</option>
              </select>
            </div>

            {/* Sort Filter */}
            <div className="space-y-1">
              <label className="text-[11px] font-medium text-[#8c919e]">Sort By</label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="w-full bg-[#141519] border border-[#282b34] text-xs text-[#f1f2f4] px-2 py-1.5 rounded outline-none focus:border-blue-500"
              >
                <option value="downloads">Most Downloads</option>
                <option value="likes">Most Likes</option>
                <option value="trending">Trending</option>
                <option value="lastModified">Recently Updated</option>
                <option value="createdAt">Recently Created</option>
              </select>
            </div>

            {/* Author / Org Filter */}
            <div className="space-y-1">
              <label className="text-[11px] font-medium text-[#8c919e]">Author / Org</label>
              <input
                type="text"
                placeholder="e.g. Qwen, TheBloke, bartowski"
                value={authorFilter}
                onChange={(e) => setAuthorFilter(e.target.value)}
                className="w-full bg-[#141519] border border-[#282b34] text-xs text-[#f1f2f4] px-2 py-1.5 rounded outline-none focus:border-blue-500"
              />
            </div>

            {/* Custom Tag Filter */}
            <div className="space-y-1">
              <label className="text-[11px] font-medium text-[#8c919e]">Tag / Library</label>
              <input
                type="text"
                placeholder="e.g. conversational, llama, reasoning"
                value={customTag}
                onChange={(e) => setCustomTag(e.target.value)}
                className="w-full bg-[#141519] border border-[#282b34] text-xs text-[#f1f2f4] px-2 py-1.5 rounded outline-none focus:border-blue-500"
              />
            </div>
          </div>
        )}
      </form>

      {/* Discovered Models Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {searchResults.map((m) => {
          const isDownloading =
            activeDownload?.modelId === m.id && activeDownload?.status === "downloading";
          const isDone = activeDownload?.modelId === m.id && activeDownload?.status === "done";

          return (
            <div
              key={m.id}
              className="p-3.5 rounded bg-[#181a20] border border-[#282b34] hover:border-[#383d4a] transition-colors flex flex-col justify-between space-y-3"
            >
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px] font-mono text-[#6c707d]">
                  <span className="px-1.5 py-0.5 rounded bg-[#20232a] text-[#8c919e] uppercase">
                    {m.pipeline_tag || "GGUF"}
                  </span>
                  <span>
                    {m.downloads ? `${Math.round(m.downloads / 1000)}k downloads` : "Popular"}
                  </span>
                </div>
                <h3 className="font-semibold text-[#f1f2f4] text-xs break-all">{m.id}</h3>
                <div className="flex flex-wrap gap-1 pt-0.5">
                  {(m.tags || ["gguf", "chat"]).slice(0, 4).map((tag) => (
                    <span
                      key={tag}
                      className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#141519] text-[#717684]"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                <span className="text-[11px] text-[#6c707d]">Multiple Quants Available</span>

                {isDone ? (
                  <span className="flex items-center gap-1 text-emerald-400 font-medium text-xs">
                    <Check size={12} />
                    <span>Downloaded</span>
                  </span>
                ) : isDownloading ? (
                  <span className="flex items-center gap-1.5 text-amber-400 font-medium text-xs font-mono">
                    <Loader2 size={12} className="animate-spin" />
                    <span>Downloading...</span>
                  </span>
                ) : (
                  <button
                    onClick={() => handleOpenDownloadModal(m)}
                    className="flex items-center gap-1 px-2.5 py-1 rounded bg-[#2563eb] hover:bg-[#1d4ed8] text-white text-xs font-medium transition-colors"
                  >
                    <Download size={11} />
                    <span>Get Model</span>
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Model Quantization & File Download Modal */}
      {selectedRepo && (
        <div
          className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedRepo(null)}
        >
          <div
            className="w-full max-w-lg bg-[#1a1c22] border border-[#2f333e] rounded shadow-2xl p-4 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between pb-2 border-b border-[#282b34]">
              <div>
                <h3 className="font-semibold text-[#f1f2f4] text-xs">Download GGUF Weights</h3>
                <p className="text-[11px] text-[#8c919e] font-mono">{selectedRepo.id}</p>
              </div>
              <button
                onClick={() => setSelectedRepo(null)}
                className="text-[#6c707d] hover:text-white"
              >
                <X size={14} />
              </button>
            </div>

            <div className="space-y-3">
              {/* Machine Specs Banner */}
              {machineSpecs && (
                <div className="p-2.5 rounded bg-[#141519] border border-[#282b34] text-[11px] space-y-1">
                  <div className="flex items-center gap-1.5 text-[#a0a5b2] font-medium">
                    <Cpu size={12} className="text-blue-400" />
                    <span>Your Machine</span>
                  </div>
                  <div className="text-[#6c707d] font-mono leading-relaxed">
                    <span className="text-[#8c919e]">{machineSpecs.cpu_model}</span>
                    <span className="mx-2">·</span>
                    <span>{machineSpecs.physical_cores} cores</span>
                    <span className="mx-2">·</span>
                    <span className="text-emerald-400 font-semibold">{formatBytes(machineSpecs.available_ram_bytes)} free</span>
                    <span className="text-[#4a4f5c]"> / {formatBytes(machineSpecs.total_ram_bytes)} RAM</span>
                  </div>
                  <div className="text-[10px] text-[#6c707d]">
                    Fit check includes ~30% headroom for context buffer.{" "}
                    <span className="text-emerald-400">✓ fits</span> = safe,{" "}
                    <span className="text-amber-400">~ tight</span> = may work,{" "}
                    <span className="text-red-400">✗ too big</span> = will OOM.
                  </div>
                </div>
              )}
              {loadingFiles ? (
                <div className="p-6 text-center space-y-2">
                  <Loader2 size={18} className="animate-spin text-blue-400 mx-auto" />
                  <p className="text-xs text-[#8c919e]">Querying repository file tree from Hugging Face...</p>
                </div>
              ) : fileFetchError ? (
                <div className="p-3 rounded bg-[#20222a] border border-[#303442] text-xs text-amber-300 space-y-1">
                  <div className="flex items-center gap-1.5 font-medium">
                    <Info size={13} />
                    <span>Hugging Face Notice</span>
                  </div>
                  <p className="text-[11px] text-[#8c919e]">{fileFetchError}</p>
                  <div className="pt-2">
                    <label className="text-[11px] font-medium text-[#a0a5b2] block mb-1">
                      Enter GGUF Filename Manually:
                    </label>
                    <input
                      type="text"
                      value={selectedFileName}
                      onChange={(e) => setSelectedFileName(e.target.value)}
                      className="w-full bg-[#121316] border border-[#282b34] text-xs text-[#f1f2f4] px-2.5 py-1.5 rounded outline-none font-mono focus:border-blue-500"
                    />
                  </div>
                </div>
              ) : repoFiles.length > 0 ? (
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <label className="text-[11px] font-medium text-[#a0a5b2]">
                      Select GGUF File ({repoFiles.length} available):
                    </label>
                    <span className="text-[10.5px] text-[#6c707d] font-mono">Live Tree</span>
                  </div>

                  <div className="max-h-52 overflow-y-auto custom-scrollbar border border-[#282b34] rounded bg-[#121316] divide-y divide-[#242730]">
                    {repoFiles.map((file) => {
                      const base = file.path.split("/").pop() || file.path;
                      const isSelected = selectedFileName === base;
                      const fit = getFileFit(file.size);
                      return (
                        <div
                          key={file.path}
                          onClick={() => setSelectedFileName(base)}
                          className={`p-2.5 flex items-center justify-between cursor-pointer text-xs transition-colors ${
                            isSelected ? "bg-[#2563eb]/20 text-white" : "hover:bg-[#1a1c22] text-[#8c919e]"
                          }`}
                        >
                          <div className="truncate pr-2">
                            <div className="font-mono text-xs text-[#f1f2f4] truncate">{base}</div>
                            {file.size && (
                              <div className="text-[10px] text-[#6c707d] font-mono">
                                Size: {formatBytes(file.size)}
                              </div>
                            )}
                          </div>
                          <div className="flex items-center gap-1.5 shrink-0">
                            {fitBadge(fit)}
                            {isSelected && <Check size={14} className="text-blue-400" />}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div className="space-y-1">
                  <label className="text-[11px] font-medium text-[#a0a5b2]">
                    Filename in repository:
                  </label>
                  <input
                    type="text"
                    value={selectedFileName}
                    onChange={(e) => setSelectedFileName(e.target.value)}
                    className="w-full bg-[#121316] border border-[#282b34] text-xs text-[#f1f2f4] px-2.5 py-1 rounded outline-none font-mono focus:border-blue-500"
                  />
                </div>
              )}

              <div className="p-2.5 rounded bg-[#141519] border border-[#282b34] text-[11px] text-[#8c919e] space-y-1">
                <div className="flex items-center gap-1.5 text-zinc-300 font-medium">
                  <Zap size={12} className="text-amber-400" />
                  <span>Direct Hugging Face Resolution</span>
                </div>
                <p>
                  Streams bytes to disk with resume (Range header) and cancellation support. File will be saved into <code>models/{selectedFileName}</code>.
                </p>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2 border-t border-[#282b34]">
              <button
                onClick={() => setSelectedRepo(null)}
                className="px-3 py-1 rounded bg-[#20232a] hover:bg-[#282c35] text-xs font-medium text-[#a0a5b2]"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDownload}
                disabled={!selectedFileName.trim()}
                className="px-3.5 py-1 rounded bg-[#2563eb] hover:bg-[#1d4ed8] disabled:opacity-40 text-white text-xs font-medium"
              >
                Download Model
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ==========================================
// Modal: Downloads Popover
// ==========================================
function DownloadsPopover({
  activeDownload,
  onClose,
  onLoadDownloaded,
  onCancelDownload,
}: {
  activeDownload: {
    modelId: string;
    filename: string;
    status: DownloadState;
    downloadedBytes?: number;
    totalBytes?: number;
    percent?: number;
    speedBytesSec?: number;
    error?: string;
  } | null;
  onClose: () => void;
  onLoadDownloaded: (filename: string) => void;
  onCancelDownload?: (filename: string) => void;
}) {
  return (
    <div className="absolute left-[54px] bottom-10 w-88 bg-[#1a1c22] border border-[#2f333e] rounded shadow-2xl p-3.5 space-y-3 z-50 text-xs">
      <div className="flex items-center justify-between pb-1.5 border-b border-[#282b34]">
        <div className="flex items-center gap-1.5 font-medium text-[#f1f2f4]">
          <CloudDownload size={14} className="text-blue-400" />
          <span>Active Downloads</span>
        </div>
        <button onClick={onClose} className="text-[#6c707d] hover:text-white">
          <X size={13} />
        </button>
      </div>

      {!activeDownload || activeDownload.status === "idle" ? (
        <div className="py-4 text-center text-[#6c707d] text-xs">No active downloads</div>
      ) : (
        <div className="space-y-2.5">
          <div className="truncate">
            <div className="font-medium text-[#f1f2f4] truncate font-mono text-xs">{activeDownload.filename}</div>
            <div className="text-[10.5px] font-mono text-[#6c707d] truncate">
              {activeDownload.modelId}
            </div>
          </div>

          {activeDownload.status === "downloading" && (
            <div className="space-y-2">
              <div className="w-full bg-[#121316] h-1.5 rounded-full overflow-hidden border border-[#282b34]">
                <div
                  className="bg-blue-500 h-full transition-all duration-300"
                  style={{ width: `${Math.round(activeDownload.percent || 0)}%` }}
                />
              </div>

              <div className="flex items-center justify-between text-[10.5px] font-mono text-[#8c919e]">
                <span>
                  {activeDownload.totalBytes && activeDownload.totalBytes > 0
                    ? `${formatBytes(activeDownload.downloadedBytes || 0)} / ${formatBytes(activeDownload.totalBytes)} (${Math.round(activeDownload.percent || 0)}%)`
                    : `${formatBytes(activeDownload.downloadedBytes || 0)} streamed`}
                </span>
                <span>
                  {activeDownload.speedBytesSec
                    ? `${(activeDownload.speedBytesSec / (1024 * 1024)).toFixed(1)} MB/s`
                    : "Connecting..."}
                </span>
              </div>

              <div className="flex items-center justify-between pt-1">
                <div className="flex items-center gap-1.5 text-amber-400 font-mono text-[11px]">
                  <Loader2 size={12} className="animate-spin" />
                  <span>Streaming chunks to disk...</span>
                </div>

                {onCancelDownload && (
                  <button
                    onClick={() => onCancelDownload(activeDownload.filename)}
                    className="px-2 py-0.5 rounded bg-[#2a171a] hover:bg-[#3d1c22] text-red-300 border border-[#5c212a] text-[10.5px] font-medium transition-colors"
                  >
                    Cancel
                  </button>
                )}
              </div>
            </div>
          )}

          {activeDownload.status === "done" && (
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-emerald-400 text-xs font-medium">
                <Check size={14} />
                <span>Download verified & saved to models/</span>
              </div>
              <button
                onClick={() => onLoadDownloaded(activeDownload.filename)}
                className="w-full py-1.5 rounded bg-[#2563eb] hover:bg-[#1d4ed8] text-white text-xs font-medium transition-colors"
              >
                Load Model Now
              </button>
            </div>
          )}

          {activeDownload.status === "failed" && (
            <div className="text-red-400 text-xs space-y-1.5 p-2 rounded bg-[#201417] border border-[#5c1d24]">
              <div className="font-semibold flex items-center gap-1">
                <AlertCircle size={13} />
                <span>Download Error</span>
              </div>
              <p className="text-[11px] text-red-300 font-mono leading-relaxed">{activeDownload.error || "Network error"}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

