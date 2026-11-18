import { invoke } from "@tauri-apps/api/core";

// ==========================================
// Local AI Workstation — Settings Architecture
// Comprehensive typed settings with persistent storage
// ==========================================

export type ThemeMode = "dark" | "graphite" | "system";
export type FontFamily = "system" | "inter" | "mono";
export type GpuOffloadMode = "auto" | "max" | "custom" | "cpu_only";
export type ContextShiftStrategy = "truncate_oldest" | "summarize" | "reject";
export type MarkdownRenderStyle = "rich" | "raw" | "minimal";

export interface AppearanceSettings {
  theme: ThemeMode;
  fontSize: number; // 11 - 16, default 12
  fontFamily: FontFamily;
  uiDensity: "compact" | "default" | "spacious";
  reduceMotion: boolean;
  codeSyntaxHighlighting: boolean;
}

export interface InferenceSettings {
  gpuOffloadMode: GpuOffloadMode;
  gpuLayers: number; // 0 - 128, default 99
  threads: number; // default 0 (auto-detect)
  useMmap: boolean; // default true
  useMlock: boolean; // default false
  flashAttention: boolean; // default false
  defaultContextLength: number; // default 4096 (n_ctx)
  batchSize: number; // default 512 (n_batch)
  idleUnloadMinutes: number; // default 10 (0 = never)
}

export interface DownloadSettings {
  modelsDirectory: string;
  hfToken: string; // for gated repositories
  maxConcurrentDownloads: number; // default 1
  autoResumeDownloads: boolean; // default true
  keepPartialFilesOnCancel: boolean; // default true
  networkTimeoutSeconds: number; // default 30
}

export interface ChatExperienceSettings {
  streamResponse: boolean; // default true
  systemPrompt: string; // default helpful assistant
  historyContextLimit: number; // messages to retain, default 20
  contextShiftStrategy: ContextShiftStrategy;
  enterSendsMessage: boolean; // default true
  renderMarkdown: MarkdownRenderStyle;
  showTokenUsage: boolean; // default true
}

export interface StorageSettings {
  autoSaveIntervalSeconds: number; // default 15
  exportWithConversationHistory: boolean; // default true
  archiveRetentionDays: number; // default 30
}

export interface MonitoringSettings {
  enableTelemetryBar: boolean; // default true
  telemetryIntervalMs: number; // default 2500
  showGpuMetrics: boolean; // default true
  showDiskUsage: boolean; // default true
  warnHighMemoryPercent: number; // default 90
}

export interface PrivacySettings {
  telemetryOptOut: boolean; // default true
  stripPiiFromLogs: boolean; // default true
  localOnlyEnforcement: boolean; // default true
  allowOutboundHfCalls: boolean; // default true
}

export interface AdvancedSettings {
  customLlamaArgs: string;
  enableDebugLogging: boolean; // default false
  localApiPort: number; // default 1234
  enableOpenAiApi: boolean; // default true
  enableAnthropicApi: boolean; // default false
}

export interface WorkstationSettings {
  version: number;
  appearance: AppearanceSettings;
  inference: InferenceSettings;
  downloads: DownloadSettings;
  chat: ChatExperienceSettings;
  storage: StorageSettings;
  monitoring: MonitoringSettings;
  privacy: PrivacySettings;
  advanced: AdvancedSettings;
}

export const DEFAULT_WORKSTATION_SETTINGS: WorkstationSettings = {
  version: 2,
  appearance: {
    theme: "graphite",
    fontSize: 12,
    fontFamily: "system",
    uiDensity: "compact",
    reduceMotion: false,
    codeSyntaxHighlighting: true,
  },
  inference: {
    gpuOffloadMode: "cpu_only",
    gpuLayers: 0,
    threads: 4,
    useMmap: true,
    useMlock: false,
    flashAttention: false,
    defaultContextLength: 512,
    batchSize: 2048,
    idleUnloadMinutes: 10,
  },
  downloads: {
    modelsDirectory: "/home/leywin/Projects/lm-studio/models",
    hfToken: "",
    maxConcurrentDownloads: 1,
    autoResumeDownloads: true,
    keepPartialFilesOnCancel: true,
    networkTimeoutSeconds: 30,
  },
  chat: {
    streamResponse: true,
    systemPrompt: "You are a helpful, knowledgeable AI assistant running locally via llama.cpp.",
    historyContextLimit: 20,
    contextShiftStrategy: "truncate_oldest",
    enterSendsMessage: true,
    renderMarkdown: "rich",
    showTokenUsage: true,
  },
  storage: {
    autoSaveIntervalSeconds: 15,
    exportWithConversationHistory: true,
    archiveRetentionDays: 30,
  },
  monitoring: {
    enableTelemetryBar: true,
    telemetryIntervalMs: 2500,
    showGpuMetrics: true,
    showDiskUsage: true,
    warnHighMemoryPercent: 90,
  },
  privacy: {
    telemetryOptOut: true,
    stripPiiFromLogs: true,
    localOnlyEnforcement: true,
    allowOutboundHfCalls: true,
  },
  advanced: {
    customLlamaArgs: "",
    enableDebugLogging: false,
    localApiPort: 1234,
    enableOpenAiApi: true,
    enableAnthropicApi: false,
  },
};

const SETTINGS_STORAGE_KEY = "local_ai_workstation_app_settings_v2";

export function loadWorkstationSettings(): WorkstationSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return DEFAULT_WORKSTATION_SETTINGS;
    const parsed = JSON.parse(raw);
    return {
      ...DEFAULT_WORKSTATION_SETTINGS,
      ...parsed,
      appearance: { ...DEFAULT_WORKSTATION_SETTINGS.appearance, ...(parsed.appearance || {}) },
      inference: { ...DEFAULT_WORKSTATION_SETTINGS.inference, ...(parsed.inference || {}) },
      downloads: { ...DEFAULT_WORKSTATION_SETTINGS.downloads, ...(parsed.downloads || {}) },
      chat: { ...DEFAULT_WORKSTATION_SETTINGS.chat, ...(parsed.chat || {}) },
      storage: { ...DEFAULT_WORKSTATION_SETTINGS.storage, ...(parsed.storage || {}) },
      monitoring: { ...DEFAULT_WORKSTATION_SETTINGS.monitoring, ...(parsed.monitoring || {}) },
      privacy: { ...DEFAULT_WORKSTATION_SETTINGS.privacy, ...(parsed.privacy || {}) },
      advanced: { ...DEFAULT_WORKSTATION_SETTINGS.advanced, ...(parsed.advanced || {}) },
    };
  } catch {
    return DEFAULT_WORKSTATION_SETTINGS;
  }
}

export function saveWorkstationSettings(settings: WorkstationSettings): void {
  try {
    const json = JSON.stringify(settings, null, 2);
    localStorage.setItem(SETTINGS_STORAGE_KEY, json);
    invoke("save_app_state", { key: "workstation_settings_v2", data: json }).catch(() => {});
  } catch (err) {
    console.error("Failed to save workstation settings:", err);
  }
}

export function resetCategorySettings<K extends keyof Omit<WorkstationSettings, "version">>(
  category: K,
  current: WorkstationSettings,
): WorkstationSettings {
  const updated = {
    ...current,
    [category]: { ...DEFAULT_WORKSTATION_SETTINGS[category] },
  };
  saveWorkstationSettings(updated);
  return updated;
}
