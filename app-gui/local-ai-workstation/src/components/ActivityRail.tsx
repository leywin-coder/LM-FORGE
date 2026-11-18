import {
  Anvil,
  CloudDownload,
  Download,
  Library,
  MessageSquare,
  Server,
  Settings,
  SlidersHorizontal,
} from "lucide-react";

export type View = "chat" | "models" | "presets" | "server" | "discover" | "settings";

export function ActivityRail({
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
    { id: "presets" as const, icon: SlidersHorizontal, label: "Intelligent Presets" },
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
