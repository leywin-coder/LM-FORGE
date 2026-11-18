import { useEffect, useState } from "react";
import {
  X,
  Cpu,
  Database,
  HardDrive,
  Activity,
  RefreshCw,
  Zap,
  Info,
  Server,
  AlertTriangle,
} from "lucide-react";
import {
  SystemMetrics,
  MetricHistoryPoint,
  InferenceThroughputMetrics,
  fetchSystemMetrics,
} from "./telemetryService";

interface TelemetryModalProps {
  isOpen: boolean;
  onClose: () => void;
  metrics: SystemMetrics | null;
  history: MetricHistoryPoint[];
  inferenceMetrics: InferenceThroughputMetrics;
  loadedModel: string | null;
  onRefresh: () => Promise<void>;
}

export function TelemetryModal({
  isOpen,
  onClose,
  metrics,
  history,
  inferenceMetrics,
  loadedModel,
  onRefresh,
}: TelemetryModalProps) {
  const [refreshing, setRefreshing] = useState(false);

  if (!isOpen) return null;

  const handleManualRefresh = async () => {
    setRefreshing(true);
    await onRefresh();
    setTimeout(() => setRefreshing(false), 300);
  };

  const formatGb = (bytes: number) => (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
  const formatMb = (bytes: number) => (bytes / (1024 * 1024)).toFixed(1) + " MB";

  const ramPercent =
    metrics && metrics.total_ram > 0 ? Math.round((metrics.used_ram / metrics.total_ram) * 100) : 0;

  const diskPercent =
    metrics && metrics.disk_total > 0
      ? Math.round((metrics.disk_used / metrics.disk_total) * 100)
      : 0;

  return (
    <div
      className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 backdrop-blur-[2px]"
      onClick={onClose}
    >
      <div
        className="w-full max-w-3xl bg-[#16181d] border border-[#282b34] rounded shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <header className="px-4 py-3 border-b border-[#282b34] flex items-center justify-between bg-[#141519]">
          <div className="flex items-center gap-2">
            <Activity size={16} className="text-emerald-400" />
            <h3 className="font-semibold text-[#f1f2f4] text-xs">
              System Resource Monitoring & Hardware Telemetry
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleManualRefresh}
              disabled={refreshing}
              className="px-2.5 py-1 rounded bg-[#20232a] hover:bg-[#282c35] border border-[#282b34] text-xs text-[#a0a5b2] hover:text-white transition-colors flex items-center gap-1.5"
            >
              <RefreshCw size={12} className={refreshing ? "animate-spin" : ""} />
              <span>Refresh Now</span>
            </button>
            <button onClick={onClose} className="text-[#6c707d] hover:text-white transition-colors">
              <X size={15} />
            </button>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar text-xs">
          {!metrics ? (
            <div className="p-8 text-center text-[#6c707d] space-y-2">
              <AlertTriangle size={24} className="mx-auto text-amber-400 opacity-80" />
              <div className="text-sm font-semibold text-[#f1f2f4]">
                Telemetry Metrics Unavailable
              </div>
              <p className="text-xs text-[#8c919e] max-w-md mx-auto">
                Unable to query operating system hardware counters. Ensure the local workstation
                backend is running with proper OS query permissions.
              </p>
            </div>
          ) : (
            <>
              {/* Top Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
                {/* CPU Card */}
                <div className="p-3 rounded bg-[#121316] border border-[#282b34] space-y-1">
                  <div className="flex items-center justify-between text-[#8c919e]">
                    <span className="font-semibold uppercase tracking-wider text-[10px]">CPU</span>
                    <Cpu size={14} className="text-blue-400" />
                  </div>
                  <div className="text-lg font-bold text-white font-mono">
                    {metrics.cpu_usage !== null
                      ? `${metrics.cpu_usage.toFixed(1)}%`
                      : "Collecting..."}
                  </div>
                  <div className="text-[10px] text-[#6c707d] truncate">
                    {metrics.cpu_cores} Logical Processors
                  </div>
                </div>

                {/* RAM Card */}
                <div className="p-3 rounded bg-[#121316] border border-[#282b34] space-y-1">
                  <div className="flex items-center justify-between text-[#8c919e]">
                    <span className="font-semibold uppercase tracking-wider text-[10px]">RAM</span>
                    <Database size={14} className="text-emerald-400" />
                  </div>
                  <div className="text-lg font-bold text-white font-mono">
                    {formatGb(metrics.used_ram)}
                  </div>
                  <div className="text-[10px] text-[#6c707d]">
                    {ramPercent}% of {formatGb(metrics.total_ram)}
                  </div>
                </div>

                {/* GPU Card */}
                <div className="p-3 rounded bg-[#121316] border border-[#282b34] space-y-1">
                  <div className="flex items-center justify-between text-[#8c919e]">
                    <span className="font-semibold uppercase tracking-wider text-[10px]">GPU</span>
                    <Zap size={14} className="text-amber-400" />
                  </div>
                  <div className="text-lg font-bold text-white font-mono">
                    {metrics.gpu_utilization !== null
                      ? `${metrics.gpu_utilization.toFixed(0)}%`
                      : "Active"}
                  </div>
                  <div className="text-[10px] text-[#6c707d] truncate">
                    {metrics.gpu_name || "Unavailable"}
                  </div>
                </div>

                {/* Storage Card */}
                <div className="p-3 rounded bg-[#121316] border border-[#282b34] space-y-1">
                  <div className="flex items-center justify-between text-[#8c919e]">
                    <span className="font-semibold uppercase tracking-wider text-[10px]">
                      Disk Space
                    </span>
                    <HardDrive size={14} className="text-purple-400" />
                  </div>
                  <div className="text-lg font-bold text-white font-mono">
                    {formatGb(metrics.disk_available)} free
                  </div>
                  <div className="text-[10px] text-[#6c707d]">
                    {diskPercent}% of {formatGb(metrics.disk_total)} used
                  </div>
                </div>
              </div>

              {/* Real Metric Timeline Graphs */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {/* CPU Real-Time Timeline */}
                <div className="p-3 rounded bg-[#121316] border border-[#282b34] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-[#f1f2f4] text-xs">
                      CPU Utilization History
                    </span>
                    <span className="font-mono text-[10.5px] text-blue-400">
                      {metrics.cpu_usage !== null ? `${metrics.cpu_usage.toFixed(1)}%` : "—"}
                    </span>
                  </div>

                  {history.length < 2 ? (
                    <div className="h-24 flex items-center justify-center text-[#595d69] text-[11px] italic">
                      Collecting real measurement timeline points...
                    </div>
                  ) : (
                    <div className="h-24 w-full flex flex-col justify-end">
                      <svg
                        className="w-full h-20 overflow-visible"
                        viewBox="0 0 100 100"
                        preserveAspectRatio="none"
                      >
                        <polyline
                          fill="none"
                          stroke="#3b82f6"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          points={history
                            .map((pt, idx) => {
                              const x = (idx / (history.length - 1)) * 100;
                              const val = pt.cpu !== null ? pt.cpu : 0;
                              const y = 100 - Math.min(100, Math.max(0, val));
                              return `${x},${y}`;
                            })
                            .join(" ")}
                        />
                      </svg>
                      <div className="flex justify-between text-[9px] font-mono text-[#595d69] pt-1">
                        <span>{history[0]?.time}</span>
                        <span>{history[history.length - 1]?.time}</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* RAM Real-Time Timeline */}
                <div className="p-3 rounded bg-[#121316] border border-[#282b34] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-[#f1f2f4] text-xs">
                      Memory (RAM) Usage History
                    </span>
                    <span className="font-mono text-[10.5px] text-emerald-400">{ramPercent}%</span>
                  </div>

                  {history.length < 2 ? (
                    <div className="h-24 flex items-center justify-center text-[#595d69] text-[11px] italic">
                      Collecting real measurement timeline points...
                    </div>
                  ) : (
                    <div className="h-24 w-full flex flex-col justify-end">
                      <svg
                        className="w-full h-20 overflow-visible"
                        viewBox="0 0 100 100"
                        preserveAspectRatio="none"
                      >
                        <polyline
                          fill="none"
                          stroke="#10b981"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          points={history
                            .map((pt, idx) => {
                              const x = (idx / (history.length - 1)) * 100;
                              const y = 100 - Math.min(100, Math.max(0, pt.ramUsagePercent));
                              return `${x},${y}`;
                            })
                            .join(" ")}
                        />
                      </svg>
                      <div className="flex justify-between text-[9px] font-mono text-[#595d69] pt-1">
                        <span>{history[0]?.time}</span>
                        <span>{history[history.length - 1]?.time}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Detailed Breakdown Sections */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {/* System & Hardware Details */}
                <div className="p-3 rounded bg-[#121316] border border-[#282b34] space-y-2">
                  <div className="font-semibold text-[#f1f2f4] text-xs pb-1 border-b border-[#242730]">
                    Hardware Specifications
                  </div>
                  <div className="space-y-1.5 text-[11px]">
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">CPU Model:</span>
                      <span className="font-medium text-[#f1f2f4] text-right truncate max-w-[240px]">
                        {metrics.cpu_model || "Generic x86_64 CPU"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">CPU Cores:</span>
                      <span className="font-mono text-[#f1f2f4]">{metrics.cpu_cores}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">Physical RAM:</span>
                      <span className="font-mono text-[#f1f2f4]">
                        {formatGb(metrics.total_ram)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">Available RAM:</span>
                      <span className="font-mono text-emerald-400">
                        {formatGb(metrics.available_ram)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">Free RAM:</span>
                      <span className="font-mono text-[#a0a5b2]">{formatGb(metrics.free_ram)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">Application Process (RSS):</span>
                      <span className="font-mono text-blue-400">
                        {metrics.app_ram > 0 ? formatMb(metrics.app_ram) : "Unavailable"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Storage & GPU Telemetry */}
                <div className="p-3 rounded bg-[#121316] border border-[#282b34] space-y-2">
                  <div className="font-semibold text-[#f1f2f4] text-xs pb-1 border-b border-[#242730]">
                    Storage & GPU Details
                  </div>
                  <div className="space-y-1.5 text-[11px]">
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">Models Directory Size:</span>
                      <span className="font-mono text-[#f1f2f4]">
                        {formatGb(metrics.models_dir_size)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">Disk Available:</span>
                      <span className="font-mono text-emerald-400">
                        {formatGb(metrics.disk_available)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">Disk Total:</span>
                      <span className="font-mono text-[#f1f2f4]">
                        {formatGb(metrics.disk_total)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">GPU Device:</span>
                      <span className="font-medium text-[#f1f2f4] text-right truncate max-w-[200px]">
                        {metrics.gpu_name || "Unavailable (No GPU detected)"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">Dedicated VRAM:</span>
                      <span className="font-mono text-[#f1f2f4]">
                        {metrics.gpu_memory_total
                          ? formatGb(metrics.gpu_memory_total)
                          : "Unavailable (Integrated/Shared RAM)"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8c919e]">GPU Temperature:</span>
                      <span className="font-mono text-[#f1f2f4]">
                        {metrics.gpu_temp !== null ? `${metrics.gpu_temp}°C` : "Unavailable"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* llama.cpp Real Inference Statistics */}
              <div className="p-3 rounded bg-[#181a20] border border-[#282b34] space-y-2">
                <div className="flex items-center justify-between pb-1 border-b border-[#242730]">
                  <div className="flex items-center gap-2">
                    <Server size={14} className="text-blue-400" />
                    <span className="font-semibold text-[#f1f2f4] text-xs">
                      llama.cpp Native Inference Engine Telemetry
                    </span>
                  </div>
                  <span className="font-mono text-[10.5px] px-1.5 py-0.2 rounded bg-[#121316] text-[#8c919e] border border-[#282b34]">
                    Engine: llama.cpp (OpenMP / CPU)
                  </span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px] pt-1">
                  <div>
                    <div className="text-[#8c919e]">Loaded Model:</div>
                    <div className="font-semibold text-white truncate font-mono">
                      {loadedModel || "None (Unloaded)"}
                    </div>
                  </div>
                  <div>
                    <div className="text-[#8c919e]">Last Generation Speed:</div>
                    <div className="font-mono font-bold text-emerald-400">
                      {inferenceMetrics.lastCompletedTokPerSec !== null
                        ? `${inferenceMetrics.lastCompletedTokPerSec} tok/s`
                        : "No generation measured"}
                    </div>
                  </div>
                  <div>
                    <div className="text-[#8c919e]">Tokens Generated:</div>
                    <div className="font-mono text-white">
                      {inferenceMetrics.generatedTokens > 0
                        ? `${inferenceMetrics.generatedTokens} tokens`
                        : "—"}
                    </div>
                  </div>
                  <div>
                    <div className="text-[#8c919e]">Last Elapsed Time:</div>
                    <div className="font-mono text-white">
                      {inferenceMetrics.elapsedMs > 0
                        ? `${(inferenceMetrics.elapsedMs / 1000).toFixed(2)}s`
                        : "—"}
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <footer className="px-4 py-2.5 border-t border-[#282b34] bg-[#141519] flex items-center justify-between text-[11px] text-[#6c707d]">
          <div className="flex items-center gap-1.5">
            <Info size={12} />
            <span>
              All measurements are collected directly from OS `/proc` subsystems and llama.cpp
              timings.
            </span>
          </div>
          <span>
            {metrics ? `Updated ${new Date(metrics.timestamp).toLocaleTimeString()}` : "Offline"}
          </span>
        </footer>
      </div>
    </div>
  );
}
