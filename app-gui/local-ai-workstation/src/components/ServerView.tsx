import { useState } from "react";
import { Check, Copy, Server } from "lucide-react";
import type { ModelLoadingState } from "./TelemetryBar";

export interface RequestLog {
  id: string;
  time: string;
  method: string;
  path: string;
  status: number;
  latency: string;
  requestJson?: string;
}

const initialServerLogs: RequestLog[] = [
  {
    id: "log-1",
    time: "12:04:15",
    method: "POST",
    path: "/v1/chat/completions",
    status: 200,
    latency: "342ms",
    requestJson: JSON.stringify(
      {
        model: "default",
        messages: [{ role: "user", content: "Implement LRU Cache in Rust" }],
        stream: false,
      },
      null,
      2
    ),
  },
  {
    id: "log-2",
    time: "12:05:02",
    method: "GET",
    path: "/v1/models",
    status: 200,
    latency: "12ms",
  },
];

export function ServerView({
  serverOn,
  setServerOn,
  backendConnected: _backendConnected,
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
