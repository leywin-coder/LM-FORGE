import { useState, useId } from "react";
import {
  X,
  Upload,
  FileCode,
  AlertTriangle,
  AlertCircle,
  Check,
  Bookmark,
  Layers,
  Info,
} from "lucide-react";
import { parseAndValidateConfig, ConfigPreviewResult, GenerationSettings } from "./configParser";

interface ConfigImportModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentSettings: GenerationSettings;
  onApplyConfig: (newSettings: GenerationSettings, configName: string) => void;
  onSaveAsPreset: (presetName: string, category: string, settings: GenerationSettings) => void;
}

export function ConfigImportModal({
  isOpen,
  onClose,
  currentSettings,
  onApplyConfig,
  onSaveAsPreset,
}: ConfigImportModalProps) {
  const fileInputId = useId();
  const [jsonInput, setJsonInput] = useState<string>("");
  const [fileName, setFileName] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [showSavePresetDialog, setShowSavePresetDialog] = useState<boolean>(false);
  const [newPresetName, setNewPresetName] = useState<string>("");
  const [newPresetCategory, setNewPresetCategory] = useState<string>("General");

  if (!isOpen) return null;

  // Real-time parsing and preview validation
  const preview: ConfigPreviewResult | null = jsonInput.trim()
    ? parseAndValidateConfig(jsonInput, currentSettings)
    : null;

  const handleFile = (file: File) => {
    if (!file.name.endsWith(".json")) {
      alert("Please select a valid .json configuration file.");
      return;
    }
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setJsonInput(content || "");
      if (preview?.name && preview.name !== "Custom Configuration") {
        setNewPresetName(preview.name);
      } else {
        setNewPresetName(file.name.replace(/\.json$/i, ""));
      }
    };
    reader.readAsText(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleConfirmApply = () => {
    if (!preview || !preview.valid) return;
    onApplyConfig(preview.mergedSettings, preview.name);
    onClose();
  };

  const handleConfirmSavePreset = () => {
    if (!preview || !preview.valid) return;
    const name = newPresetName.trim() || preview.name || "Custom Preset";
    onSaveAsPreset(name, newPresetCategory, preview.mergedSettings);
    onApplyConfig(preview.mergedSettings, name);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 backdrop-blur-[2px]"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl bg-[#16181d] border border-[#282b34] rounded shadow-2xl overflow-hidden flex flex-col max-h-[88vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <header className="px-4 py-3 border-b border-[#282b34] flex items-center justify-between bg-[#141519]">
          <div className="flex items-center gap-2">
            <FileCode size={16} className="text-blue-400" />
            <h3 className="font-semibold text-[#f1f2f4] text-xs">
              Import Generation Configuration (JSON)
            </h3>
          </div>
          <button onClick={onClose} className="text-[#6c707d] hover:text-white transition-colors">
            <X size={15} />
          </button>
        </header>

        {/* Modal Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar text-xs">
          {/* File Picker & Drag-and-drop zone */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`border border-dashed rounded p-3 text-center transition-colors ${
              dragActive
                ? "border-blue-500 bg-blue-500/10"
                : "border-[#282b34] bg-[#121316] hover:border-[#383d4a]"
            }`}
          >
            <input
              type="file"
              id={fileInputId}
              accept=".json"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFile(e.target.files[0]);
                }
              }}
            />
            <label
              htmlFor={fileInputId}
              className="cursor-pointer flex flex-col items-center justify-center gap-1.5"
            >
              <Upload size={18} className="text-[#6c707d]" />
              <div className="text-xs text-[#a0a5b2]">
                <span className="text-blue-400 font-medium">Choose a .json file</span> or drag and
                drop here
              </div>
              {fileName && (
                <div className="text-[11px] font-mono text-emerald-400">Selected: {fileName}</div>
              )}
            </label>
          </div>

          {/* Direct JSON Text Area */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] font-semibold text-[#6c707d] uppercase tracking-wider">
              <span>Or Paste JSON Configuration:</span>
              {jsonInput && (
                <button
                  onClick={() => {
                    setJsonInput("");
                    setFileName(null);
                  }}
                  className="text-red-400 hover:text-red-300 normal-case"
                >
                  Clear
                </button>
              )}
            </div>
            <textarea
              value={jsonInput}
              onChange={(e) => setJsonInput(e.target.value)}
              placeholder={`{\n  "name": "Creative Writing",\n  "parameters": {\n    "temperature": 1.0,\n    "top_p": 0.95,\n    "min_p": 0.05,\n    "max_tokens": 2048\n  }\n}`}
              rows={6}
              className="w-full bg-[#121316] border border-[#282b34] focus:border-blue-500 rounded p-2.5 font-mono text-[11px] text-[#f1f2f4] placeholder-[#464a56] outline-none custom-scrollbar"
            />
          </div>

          {/* Validation & Live Preview Section */}
          {preview && (
            <div className="space-y-3 pt-2 border-t border-[#242730]">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-[#f1f2f4] text-xs">{preview.name}</span>
                  <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#1c1e24] text-[#8c919e] border border-[#282b34]">
                    Engine: {preview.engine}
                  </span>
                </div>
                {preview.valid ? (
                  <span className="flex items-center gap-1 text-emerald-400 text-xs font-medium">
                    <Check size={13} /> Valid Configuration
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-red-400 text-xs font-medium">
                    <AlertCircle size={13} /> Validation Errors
                  </span>
                )}
              </div>

              {preview.description && (
                <p className="text-[11px] text-[#6c707d] italic">{preview.description}</p>
              )}

              {/* Errors alert */}
              {preview.errors.length > 0 && (
                <div className="p-2.5 rounded bg-red-950/30 border border-red-800/50 text-red-300 space-y-1">
                  <div className="font-semibold text-xs flex items-center gap-1.5">
                    <AlertCircle size={13} />
                    <span>Configuration Errors (Must resolve to import):</span>
                  </div>
                  <ul className="list-disc list-inside text-[11px] space-y-0.5">
                    {preview.errors.map((err, i) => (
                      <li key={i}>{err}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Warnings alert */}
              {preview.warnings.length > 0 && (
                <div className="p-2.5 rounded bg-amber-950/20 border border-amber-800/40 text-amber-300 space-y-1">
                  <div className="font-semibold text-xs flex items-center gap-1.5">
                    <AlertTriangle size={13} />
                    <span>Warnings:</span>
                  </div>
                  <ul className="list-disc list-inside text-[11px] space-y-0.5">
                    {preview.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Parameters Changed Preview */}
              <div className="space-y-1.5">
                <div className="text-[10.5px] font-semibold text-[#8c919e] uppercase tracking-wider">
                  Parameters Being Changed ({preview.changes.length})
                </div>
                {preview.changes.length === 0 ? (
                  <div className="text-[11px] text-[#6c707d] italic p-2 bg-[#121316] rounded border border-[#242730]">
                    No parameter changes detected.
                  </div>
                ) : (
                  <div className="border border-[#282b34] rounded overflow-hidden divide-y divide-[#20232a] bg-[#121316]">
                    {preview.changes.map((c) => (
                      <div
                        key={c.key}
                        className="px-3 py-1.5 flex items-center justify-between text-[11.5px]"
                      >
                        <span className="font-medium text-[#f1f2f4]">{c.label}</span>
                        <div className="flex items-center gap-2 font-mono text-[11px]">
                          <span className="text-[#6c707d] line-through">
                            {formatVal(c.current)}
                          </span>
                          <span className="text-[#4ade80] font-semibold">
                            → {formatVal(c.imported)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Unsupported Parameters */}
              {preview.unsupported.length > 0 && (
                <div className="space-y-1.5">
                  <div className="text-[10.5px] font-semibold text-amber-400 uppercase tracking-wider">
                    Unsupported Parameters ({preview.unsupported.length})
                  </div>
                  <div className="border border-amber-900/40 rounded p-2.5 bg-[#181614] text-[11px] space-y-1.5">
                    {preview.unsupported.map((u) => (
                      <div key={u.field} className="space-y-0.5">
                        <div className="font-mono text-amber-300 font-semibold">
                          {u.field}: {String(u.value)}
                        </div>
                        <div className="text-[#8c919e] text-[10.5px]">{u.reason}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Unrecognized Fields */}
              {preview.unrecognized.length > 0 && (
                <div className="text-[11px] text-[#6c707d] flex items-center gap-1.5">
                  <Info size={12} />
                  <span>Ignored unrecognized fields: {preview.unrecognized.join(", ")}</span>
                </div>
              )}

              {/* Unchanged Parameters Summary */}
              <div className="text-[11px] text-[#595d69] pt-1">
                Preserved unchanged parameters: {preview.unchanged.length} settings will remain
                exactly as configured.
              </div>
            </div>
          )}

          {/* Save As Preset Subdialog */}
          {showSavePresetDialog && preview && (
            <div className="p-3 rounded bg-[#181a20] border border-blue-500/40 space-y-2.5">
              <div className="font-semibold text-xs text-[#f1f2f4] flex items-center gap-1.5">
                <Bookmark size={13} className="text-blue-400" />
                <span>Save Imported Configuration as a Reusable Preset</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[10.5px] text-[#8c919e]">Preset Name</label>
                  <input
                    type="text"
                    value={newPresetName}
                    onChange={(e) => setNewPresetName(e.target.value)}
                    placeholder="e.g. Creative Storytelling"
                    className="w-full bg-[#121316] border border-[#282b34] rounded px-2 py-1 text-xs text-[#f1f2f4] outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="text-[10.5px] text-[#8c919e]">Category</label>
                  <select
                    value={newPresetCategory}
                    onChange={(e) => setNewPresetCategory(e.target.value)}
                    className="w-full bg-[#121316] border border-[#282b34] rounded px-2 py-1 text-xs text-[#f1f2f4] outline-none cursor-pointer"
                  >
                    <option value="General">General</option>
                    <option value="Coding">Coding</option>
                    <option value="Reasoning">Reasoning</option>
                    <option value="Creative">Creative</option>
                    <option value="JSON">JSON</option>
                  </select>
                </div>
              </div>
              <div className="flex justify-end gap-2 pt-1">
                <button
                  onClick={() => setShowSavePresetDialog(false)}
                  className="px-2.5 py-1 rounded bg-[#20232a] text-[#8c919e] hover:text-white text-xs"
                >
                  Back
                </button>
                <button
                  onClick={handleConfirmSavePreset}
                  className="px-3 py-1 rounded bg-[#2563eb] hover:bg-[#1d4ed8] text-white text-xs font-medium"
                >
                  Save Preset & Apply
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <footer className="px-4 py-2.5 border-t border-[#282b34] bg-[#141519] flex items-center justify-between">
          <button
            onClick={onClose}
            className="px-3 py-1 rounded bg-[#20232a] hover:bg-[#282c35] text-[#a0a5b2] hover:text-white text-xs transition-colors"
          >
            Cancel
          </button>

          <div className="flex items-center gap-2">
            {preview && preview.valid && !showSavePresetDialog && (
              <button
                onClick={() => {
                  setNewPresetName(
                    preview.name || fileName?.replace(/\.json$/i, "") || "Custom Preset",
                  );
                  setShowSavePresetDialog(true);
                }}
                className="px-3 py-1 rounded bg-[#1e222a] hover:bg-[#282d38] border border-[#2e3442] text-[#d4d6dc] text-xs font-medium transition-colors flex items-center gap-1.5"
              >
                <Bookmark size={12} className="text-blue-400" />
                <span>Save as Preset...</span>
              </button>
            )}

            <button
              onClick={handleConfirmApply}
              disabled={!preview || !preview.valid || preview.changes.length === 0}
              className="px-3.5 py-1 rounded bg-[#2563eb] hover:bg-[#1d4ed8] text-white text-xs font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
            >
              <Check size={13} />
              <span>Apply Configuration</span>
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

function formatVal(val: unknown): string {
  if (val === null || val === undefined) return "null";
  if (Array.isArray(val)) return `[${val.join(", ")}]`;
  if (typeof val === "string") return `"${val.length > 24 ? val.slice(0, 24) + "..." : val}"`;
  return String(val);
}
