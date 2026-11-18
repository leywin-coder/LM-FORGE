// ============================================================================
// Real System Resource Monitoring Service
// Gathers real OS measurements: CPU, RAM, GPU, Disk, and Inference throughput.
// Never uses simulated/random data. Reports "Unavailable" if unsupported.
// ============================================================================

import { invoke } from "@tauri-apps/api/core";

export interface SystemMetrics {
  cpu_usage: number | null;
  cpu_model: string | null;
  cpu_cores: number;
  process_cpu_usage: number | null;
  total_ram: number;
  used_ram: number;
  available_ram: number;
  free_ram: number;
  app_ram: number;
  disk_total: number;
  disk_available: number;
  disk_used: number;
  models_dir_size: number;
  gpu_name: string | null;
  gpu_utilization: number | null;
  gpu_memory_total: number | null;
  gpu_memory_used: number | null;
  gpu_temp: number | null;
  gpu_power: number | null;
  timestamp: number;
}

export interface MetricHistoryPoint {
  time: string;
  timestamp: number;
  cpu: number | null;
  ramUsagePercent: number;
  usedRamGb: number;
  gpuUsage: number | null;
}

export interface InferenceThroughputMetrics {
  active: boolean;
  promptTokens: number;
  generatedTokens: number;
  elapsedMs: number;
  tokPerSec: number | null;
  promptSpeedTokPerSec: number | null;
  lastCompletedTokPerSec: number | null;
}

export const INITIAL_INFERENCE_METRICS: InferenceThroughputMetrics = {
  active: false,
  promptTokens: 0,
  generatedTokens: 0,
  elapsedMs: 0,
  tokPerSec: null,
  promptSpeedTokPerSec: null,
  lastCompletedTokPerSec: null,
};

/**
 * Fetch real system metrics from backend.
 * Returns null if backend command is unreachable or fails.
 */
export async function fetchSystemMetrics(): Promise<SystemMetrics | null> {
  try {
    const res = await invoke<SystemMetrics>("get_system_metrics");
    if (res && typeof res.timestamp === "number") {
      return res;
    }
    return null;
  } catch (err) {
    // Graceful error; real data unavailable
    return null;
  }
}

/**
 * Calculate generation throughput strictly from real measurements.
 */
export function calculateThroughput(generatedTokens: number, elapsedMs: number): number | null {
  if (generatedTokens <= 0 || elapsedMs <= 50) return null;
  const sec = elapsedMs / 1000;
  return Math.round((generatedTokens / sec) * 10) / 10;
}
