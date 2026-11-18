import { Cpu, Loader2 } from "lucide-react";
import type { SystemMetrics } from "../telemetryService";

export type ModelLoadingState = "not_loaded" | "loading" | "loaded" | "failed" | "unloading";

export interface InferenceThroughputMetrics {
  active: boolean;
  promptTokens: number;
  generatedTokens: number;
  elapsedMs: number;
  tokPerSec: number | null;
  promptSpeedTokPerSec: number | null;
  lastCompletedTokPerSec: number | null;
}

export function TelemetryBar({
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
