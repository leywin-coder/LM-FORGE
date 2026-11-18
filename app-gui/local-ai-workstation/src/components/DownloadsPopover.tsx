import { AlertCircle, Check, CloudDownload, Loader2, X } from "lucide-react";

export type DownloadState = "idle" | "downloading" | "done" | "failed";

export interface ActiveDownloadInfo {
  modelId: string;
  filename: string;
  status: DownloadState;
  downloadedBytes?: number;
  totalBytes?: number;
  percent?: number;
  speedBytesSec?: number;
  error?: string;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

export function DownloadsPopover({
  activeDownload,
  onClose,
  onLoadDownloaded,
  onCancelDownload,
}: {
  activeDownload: ActiveDownloadInfo | null;
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
