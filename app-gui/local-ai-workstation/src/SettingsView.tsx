import React, { useState } from "react";
import {
  X,
  Palette,
  Cpu,
  Download,
  MessageSquare,
  HardDrive,
  Activity,
  Shield,
  Terminal,
  RotateCcw,
  Check,
  Save,
  FileDown,
  FileUp,
} from "lucide-react";
import {
  WorkstationSettings,
  ThemeMode,
  GpuOffloadMode,
  saveWorkstationSettings,
  resetCategorySettings,
  DEFAULT_WORKSTATION_SETTINGS,
} from "./settingsManager";

interface SettingsViewProps {
  settings: WorkstationSettings;
  onUpdateSettings: (newSettings: WorkstationSettings) => void;
  onClose?: () => void;
}

type TabCategory =
  | "appearance"
  | "inference"
  | "downloads"
  | "chat"
  | "storage"
  | "monitoring"
  | "privacy"
  | "advanced";

export const SettingsView: React.FC<SettingsViewProps> = ({
  settings,
  onUpdateSettings,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<TabCategory>("inference");
  const [copiedSuccess, setCopiedSuccess] = useState(false);

  const parseThemeMode = (value: string): ThemeMode =>
    value === "dark" || value === "graphite" || value === "system" ? value : DEFAULT_WORKSTATION_SETTINGS.appearance.theme;
  const parseDensity = (value: string): WorkstationSettings["appearance"]["uiDensity"] =>
    value === "compact" || value === "default" || value === "spacious" ? value : DEFAULT_WORKSTATION_SETTINGS.appearance.uiDensity;
  const parseGpuOffloadMode = (value: string): GpuOffloadMode =>
    value === "auto" || value === "max" || value === "custom" || value === "cpu_only" ? value : DEFAULT_WORKSTATION_SETTINGS.inference.gpuOffloadMode;

  const handleFieldChange = <C extends keyof Omit<WorkstationSettings, "version">, F extends keyof WorkstationSettings[C]>(
    category: C,
    field: F,
    value: WorkstationSettings[C][F]
  ) => {
    const next = {
      ...settings,
      [category]: {
        ...settings[category],
        [field]: value,
      },
    };
    onUpdateSettings(next);
    saveWorkstationSettings(next);
  };

  const handleResetCategory = (cat: TabCategory) => {
    if (window.confirm(`Reset all ${cat} settings to default?`)) {
      const next = resetCategorySettings(cat, settings);
      onUpdateSettings(next);
    }
  };

  const handleExportJson = () => {
    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `workstation-settings-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setCopiedSuccess(true);
    setTimeout(() => setCopiedSuccess(false), 2000);
  };

  const handleImportJson = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        const merged: WorkstationSettings = {
          ...DEFAULT_WORKSTATION_SETTINGS,
          ...parsed,
        };
        onUpdateSettings(merged);
        saveWorkstationSettings(merged);
        alert("Settings successfully imported.");
      } catch {
        alert("Failed to parse settings JSON file.");
      }
    };
    reader.readAsText(file);
  };

  const navItems: { id: TabCategory; label: string; icon: React.ComponentType<{ size: number }> }[] = [
    { id: "appearance", label: "Appearance & UI", icon: Palette },
    { id: "inference", label: "Model & Inference", icon: Cpu },
    { id: "downloads", label: "Hugging Face & Downloads", icon: Download },
    { id: "chat", label: "Chat Experience", icon: MessageSquare },
    { id: "storage", label: "Storage & Backups", icon: HardDrive },
    { id: "monitoring", label: "System Monitoring", icon: Activity },
    { id: "privacy", label: "Privacy & Telemetry", icon: Shield },
    { id: "advanced", label: "Advanced / Developer", icon: Terminal },
  ];

  return (
    <div className="flex-1 flex flex-col h-full bg-[#121316] text-[#f1f2f4] overflow-hidden">
      {/* Top Header */}
      <header className="px-5 py-3.5 border-b border-[#282b34] bg-[#16181d] flex items-center justify-between shrink-0">
        <div>
          <h1 className="text-sm font-bold text-[#f1f2f4]">Application Preferences</h1>
          <p className="text-[11px] text-[#8c919e]">
            Fine-tune workstation behavior, hardware offloading, directories, and context limits.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#20232a] hover:bg-[#282c35] border border-[#282b34] text-xs text-[#a0a5b2] hover:text-white cursor-pointer transition-colors">
            <FileUp size={12} className="text-blue-400" />
            <span>Import</span>
            <input type="file" accept=".json" onChange={handleImportJson} className="hidden" />
          </label>

          <button
            onClick={handleExportJson}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#20232a] hover:bg-[#282c35] border border-[#282b34] text-xs text-[#a0a5b2] hover:text-white transition-colors"
          >
            {copiedSuccess ? <Check size={12} className="text-emerald-400" /> : <FileDown size={12} className="text-emerald-400" />}
            <span>Export</span>
          </button>

          {onClose && (
            <button onClick={onClose} className="p-1 rounded text-[#8c919e] hover:text-white ml-2">
              <X size={15} />
            </button>
          )}
        </div>
      </header>

      {/* Main Container */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Categories Sidebar */}
        <aside className="w-56 bg-[#16181d] border-r border-[#282b34] p-2 space-y-0.5 shrink-0 select-none overflow-y-auto">
          {navItems.map(({ id, label, icon: Icon }) => {
            const active = activeTab === id;
            return (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded text-xs transition-colors text-left ${
                  active
                    ? "bg-[#2563eb] text-white font-medium"
                    : "text-[#8c919e] hover:bg-[#1f2229] hover:text-[#f1f2f4]"
                }`}
              >
                <Icon size={14} />
                <span className="truncate">{label}</span>
              </button>
            );
          })}
        </aside>

        {/* Content Panel */}
        <main className="flex-1 overflow-y-auto p-6 space-y-6 max-w-3xl custom-scrollbar">
          {/* Header for current tab with reset */}
          <div className="flex items-center justify-between pb-3 border-b border-[#242730]">
            <div>
              <h2 className="text-sm font-semibold capitalize text-[#f1f2f4]">
                {activeTab} Settings
              </h2>
              <p className="text-xs text-[#6c707d]">
                Configuration persists across workstation restarts.
              </p>
            </div>

            <button
              onClick={() => handleResetCategory(activeTab)}
              className="flex items-center gap-1.5 px-2 py-1 rounded text-xs text-[#8c919e] hover:text-amber-300 hover:bg-[#20232a] transition-colors"
              title="Reset category to defaults"
            >
              <RotateCcw size={12} />
              <span>Reset to Defaults</span>
            </button>
          </div>

          {/* TAB: Appearance */}
          {activeTab === "appearance" && (
            <div className="space-y-4 text-xs">
              <div className="p-3.5 rounded bg-[#181a20] border border-[#282b34] space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Application Theme</div>
                    <div className="text-[11px] text-[#6c707d]">Neutral graphite dark workspace palette</div>
                  </div>
                  <select
                    value={settings.appearance.theme}
                    onChange={(e) => handleFieldChange("appearance", "theme", parseThemeMode(e.target.value))}
                    className="bg-[#121316] border border-[#282b34] text-[#f1f2f4] px-2.5 py-1 rounded outline-none"
                  >
                    <option value="graphite">Graphite (Professional Dark)</option>
                    <option value="dark">Pure Black</option>
                    <option value="system">Follow System</option>
                  </select>
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">UI Density</div>
                    <div className="text-[11px] text-[#6c707d]">Controls spacing and information density</div>
                  </div>
                  <select
                    value={settings.appearance.uiDensity}
                    onChange={(e) => handleFieldChange("appearance", "uiDensity", parseDensity(e.target.value))}
                    className="bg-[#121316] border border-[#282b34] text-[#f1f2f4] px-2.5 py-1 rounded outline-none"
                  >
                    <option value="compact">Compact (Dense)</option>
                    <option value="default">Standard</option>
                    <option value="spacious">Spacious</option>
                  </select>
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Code Syntax Highlighting</div>
                    <div className="text-[11px] text-[#6c707d]">Render markdown code blocks with syntax color</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.appearance.codeSyntaxHighlighting}
                    onChange={(e) => handleFieldChange("appearance", "codeSyntaxHighlighting", e.target.checked)}
                    className="w-4 h-4 accent-blue-600 rounded cursor-pointer"
                  />
                </div>
              </div>
            </div>
          )}

          {/* TAB: Inference */}
          {activeTab === "inference" && (
            <div className="space-y-4 text-xs">
              <div className="p-3.5 rounded bg-[#181a20] border border-[#282b34] space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">GPU Offload Mode</div>
                    <div className="text-[11px] text-[#6c707d]">Offload computation layers to GPU VRAM via CUDA/Vulkan</div>
                  </div>
                  <select
                    value={settings.inference.gpuOffloadMode}
                    onChange={(e) => handleFieldChange("inference", "gpuOffloadMode", parseGpuOffloadMode(e.target.value))}
                    className="bg-[#121316] border border-[#282b34] text-[#f1f2f4] px-2.5 py-1 rounded outline-none"
                  >
                    <option value="auto">Auto Detect (Recommended)</option>
                    <option value="max">Max Offload (99 Layers)</option>
                    <option value="custom">Custom Layer Count</option>
                    <option value="cpu_only">CPU Only</option>
                  </select>
                </div>

                {settings.inference.gpuOffloadMode === "custom" && (
                  <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                    <div>
                      <div className="font-medium text-[#f1f2f4]">Offloaded Layers</div>
                      <div className="text-[11px] text-[#6c707d]">Number of transformer layers pushed to GPU</div>
                    </div>
                    <input
                      type="number"
                      min={0}
                      max={128}
                      value={settings.inference.gpuLayers}
                      onChange={(e) => handleFieldChange("inference", "gpuLayers", parseInt(e.target.value) || 0)}
                      className="w-20 bg-[#121316] border border-[#282b34] text-[#f1f2f4] px-2 py-1 rounded text-right"
                    />
                  </div>
                )}

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">CPU Inference Threads</div>
                    <div className="text-[11px] text-[#6c707d]">Physical core allocation for token generation (0 = auto-detect)</div>
                  </div>
                  <input
                    type="number"
                    min={0}
                    max={64}
                    value={settings.inference.threads}
                    onChange={(e) => handleFieldChange("inference", "threads", parseInt(e.target.value) || 0)}
                    className="w-20 bg-[#121316] border border-[#282b34] text-[#f1f2f4] px-2 py-1 rounded text-right"
                  />
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">FlashAttention</div>
                    <div className="text-[11px] text-[#6c707d]">Memory-efficient attention computation (boosts tok/s & reduces VRAM)</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.inference.flashAttention}
                    onChange={(e) => handleFieldChange("inference", "flashAttention", e.target.checked)}
                    className="w-4 h-4 accent-blue-600 rounded cursor-pointer"
                  />
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Zero-Copy Memory Mapping (mmap)</div>
                    <div className="text-[11px] text-[#6c707d]">Load GGUF weights into virtual memory instantly without copying to RAM</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.inference.useMmap}
                    onChange={(e) => handleFieldChange("inference", "useMmap", e.target.checked)}
                    className="w-4 h-4 accent-blue-600 rounded cursor-pointer"
                  />
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Default Context Length (n_ctx)</div>
                    <div className="text-[11px] text-[#6c707d]">Initial context window capacity in tokens</div>
                  </div>
                  <select
                    value={settings.inference.defaultContextLength}
                    onChange={(e) => handleFieldChange("inference", "defaultContextLength", parseInt(e.target.value))}
                    className="bg-[#121316] border border-[#282b34] text-[#f1f2f4] px-2.5 py-1 rounded outline-none"
                  >
                    <option value={2048}>2,048 tokens</option>
                    <option value={4096}>4,096 tokens</option>
                    <option value={8192}>8,192 tokens</option>
                    <option value={16384}>16,384 tokens</option>
                    <option value={32768}>32,768 tokens</option>
                  </select>
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Batch Size (n_batch)</div>
                    <div className="text-[11px] text-[#6c707d]">Prompt evaluation chunk size in tokens</div>
                  </div>
                  <input
                    type="number"
                    step={128}
                    min={128}
                    max={2048}
                    value={settings.inference.batchSize}
                    onChange={(e) => handleFieldChange("inference", "batchSize", parseInt(e.target.value) || 512)}
                    className="w-20 bg-[#121316] border border-[#282b34] text-[#f1f2f4] px-2 py-1 rounded text-right"
                  />
                </div>
              </div>
            </div>
          )}

          {/* TAB: Downloads */}
          {activeTab === "downloads" && (
            <div className="space-y-4 text-xs">
              <div className="p-3.5 rounded bg-[#181a20] border border-[#282b34] space-y-3">
                <div className="space-y-1">
                  <div className="font-medium text-[#f1f2f4]">Models Directory</div>
                  <div className="text-[11px] text-[#6c707d]">Local storage path where .gguf files are indexed</div>
                  <input
                    type="text"
                    value={settings.downloads.modelsDirectory}
                    onChange={(e) => handleFieldChange("downloads", "modelsDirectory", e.target.value)}
                    className="w-full bg-[#121316] border border-[#282b34] text-[#f1f2f4] font-mono text-xs px-2.5 py-1.5 rounded outline-none focus:border-blue-500"
                  />
                </div>

                <div className="space-y-1 pt-2 border-t border-[#242730]">
                  <div className="font-medium text-[#f1f2f4]">Hugging Face API Token</div>
                  <div className="text-[11px] text-[#6c707d]">Required to download gated models (e.g. Meta-Llama-3, Gemma-2)</div>
                  <input
                    type="password"
                    placeholder="hf_..."
                    value={settings.downloads.hfToken}
                    onChange={(e) => handleFieldChange("downloads", "hfToken", e.target.value)}
                    className="w-full bg-[#121316] border border-[#282b34] text-[#f1f2f4] font-mono text-xs px-2.5 py-1.5 rounded outline-none focus:border-blue-500"
                  />
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Auto-Resume Broken Downloads</div>
                    <div className="text-[11px] text-[#6c707d]">Use HTTP Range headers to continue downloading partial files</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.downloads.autoResumeDownloads}
                    onChange={(e) => handleFieldChange("downloads", "autoResumeDownloads", e.target.checked)}
                    className="w-4 h-4 accent-blue-600 rounded cursor-pointer"
                  />
                </div>
              </div>
            </div>
          )}

          {/* TAB: Chat */}
          {activeTab === "chat" && (
            <div className="space-y-4 text-xs">
              <div className="p-3.5 rounded bg-[#181a20] border border-[#282b34] space-y-3">
                <div className="space-y-1">
                  <div className="font-medium text-[#f1f2f4]">Global System Prompt</div>
                  <div className="text-[11px] text-[#6c707d]">Initial instructions sent before the conversation history</div>
                  <textarea
                    rows={3}
                    value={settings.chat.systemPrompt}
                    onChange={(e) => handleFieldChange("chat", "systemPrompt", e.target.value)}
                    className="w-full bg-[#121316] border border-[#282b34] text-[#f1f2f4] text-xs p-2 rounded outline-none focus:border-blue-500"
                  />
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Stream Response Tokens</div>
                    <div className="text-[11px] text-[#6c707d]">Emit generated tokens in real time to the chat UI</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.chat.streamResponse}
                    onChange={(e) => handleFieldChange("chat", "streamResponse", e.target.checked)}
                    className="w-4 h-4 accent-blue-600 rounded cursor-pointer"
                  />
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">History Message Retention</div>
                    <div className="text-[11px] text-[#6c707d]">Number of previous turns sent to the context window</div>
                  </div>
                  <input
                    type="number"
                    min={2}
                    max={100}
                    value={settings.chat.historyContextLimit}
                    onChange={(e) => handleFieldChange("chat", "historyContextLimit", parseInt(e.target.value) || 20)}
                    className="w-20 bg-[#121316] border border-[#282b34] text-[#f1f2f4] px-2 py-1 rounded text-right"
                  />
                </div>
              </div>
            </div>
          )}

          {/* TAB: Storage */}
          {activeTab === "storage" && (
            <div className="space-y-4 text-xs">
              <div className="p-3.5 rounded bg-[#181a20] border border-[#282b34] space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Auto-Save Interval</div>
                    <div className="text-[11px] text-[#6c707d]">Periodically persist chat sessions to local storage</div>
                  </div>
                  <select
                    value={settings.storage.autoSaveIntervalSeconds}
                    onChange={(e) => handleFieldChange("storage", "autoSaveIntervalSeconds", parseInt(e.target.value))}
                    className="bg-[#121316] border border-[#282b34] text-[#f1f2f4] px-2.5 py-1 rounded outline-none"
                  >
                    <option value={5}>Every 5 seconds</option>
                    <option value={15}>Every 15 seconds</option>
                    <option value={60}>Every minute</option>
                  </select>
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Archive Retention Period</div>
                    <div className="text-[11px] text-[#6c707d]">Retain archived chat threads before automatic purge</div>
                  </div>
                  <span className="font-mono text-[#a0a5b2]">{settings.storage.archiveRetentionDays} days</span>
                </div>
              </div>
            </div>
          )}

          {/* TAB: Monitoring */}
          {activeTab === "monitoring" && (
            <div className="space-y-4 text-xs">
              <div className="p-3.5 rounded bg-[#181a20] border border-[#282b34] space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Status Bar Telemetry</div>
                    <div className="text-[11px] text-[#6c707d]">Show live CPU, RAM, and GPU utilization in the bottom bar</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.monitoring.enableTelemetryBar}
                    onChange={(e) => handleFieldChange("monitoring", "enableTelemetryBar", e.target.checked)}
                    className="w-4 h-4 accent-blue-600 rounded cursor-pointer"
                  />
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Refresh Interval</div>
                    <div className="text-[11px] text-[#6c707d]">Hardware polling rate</div>
                  </div>
                  <select
                    value={settings.monitoring.telemetryIntervalMs}
                    onChange={(e) => handleFieldChange("monitoring", "telemetryIntervalMs", parseInt(e.target.value))}
                    className="bg-[#121316] border border-[#282b34] text-[#f1f2f4] px-2.5 py-1 rounded outline-none"
                  >
                    <option value={1000}>1.0 second (High precision)</option>
                    <option value={2500}>2.5 seconds (Balanced)</option>
                    <option value={5000}>5.0 seconds (Low overhead)</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* TAB: Privacy */}
          {activeTab === "privacy" && (
            <div className="space-y-4 text-xs">
              <div className="p-3.5 rounded bg-[#181a20] border border-[#282b34] space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Strict Local Isolation</div>
                    <div className="text-[11px] text-[#6c707d]">Never send inference tokens or prompts outside your machine</div>
                  </div>
                  <span className="font-mono text-emerald-400 font-semibold">Enforced (Air-Gapped)</span>
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Opt Out of Telemetry</div>
                    <div className="text-[11px] text-[#6c707d]">Zero user tracking or error telemetry transmission</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.privacy.telemetryOptOut}
                    onChange={(e) => handleFieldChange("privacy", "telemetryOptOut", e.target.checked)}
                    className="w-4 h-4 accent-blue-600 rounded cursor-pointer"
                  />
                </div>
              </div>
            </div>
          )}

          {/* TAB: Advanced */}
          {activeTab === "advanced" && (
            <div className="space-y-4 text-xs">
              <div className="p-3.5 rounded bg-[#181a20] border border-[#282b34] space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Local Server HTTP Port</div>
                    <div className="text-[11px] text-[#6c707d]">OpenAI-compatible /v1/chat/completions port</div>
                  </div>
                  <input
                    type="number"
                    value={settings.advanced.localApiPort}
                    onChange={(e) => handleFieldChange("advanced", "localApiPort", parseInt(e.target.value) || 1234)}
                    className="w-20 bg-[#121316] border border-[#282b34] text-[#f1f2f4] font-mono px-2 py-1 rounded text-right"
                  />
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-[#242730]">
                  <div>
                    <div className="font-medium text-[#f1f2f4]">Debug Logging</div>
                    <div className="text-[11px] text-[#6c707d]">Enable verbose stdout debugging in inference engine</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.advanced.enableDebugLogging}
                    onChange={(e) => handleFieldChange("advanced", "enableDebugLogging", e.target.checked)}
                    className="w-4 h-4 accent-blue-600 rounded cursor-pointer"
                  />
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};
