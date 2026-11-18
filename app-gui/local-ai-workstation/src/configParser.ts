// ============================================================================
// JSON Generation Configuration Import & Export Module
// Validates, parses, previews, and exports llama.cpp generation parameters.
// ============================================================================

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

export interface ExportedConfigJson {
  name: string;
  description?: string;
  version: number;
  engine: "llama.cpp";
  modelAssociation?: string | null;
  parameters: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    min_p?: number;
    max_tokens?: number;
    n_predict?: number;
    repeat_penalty?: number;
    repetition_penalty?: number;
    presence_penalty?: number;
    frequency_penalty?: number;
    seed?: number | null;
    stop_tokens?: string[];
    system_prompt?: string;
    [key: string]: unknown;
  };
}

export interface ConfigChangeItem {
  key: keyof GenerationSettings;
  label: string;
  current: string | number | string[] | null;
  imported: string | number | string[] | null;
}

export interface ConfigUnchangedItem {
  key: keyof GenerationSettings;
  label: string;
  value: string | number | string[] | null;
}

export interface UnsupportedParamItem {
  field: string;
  value: unknown;
  reason: string;
}

export interface ConfigPreviewResult {
  valid: boolean;
  name: string;
  description: string;
  engine: string;
  modelAssociation?: string | null;
  changes: ConfigChangeItem[];
  unchanged: ConfigUnchangedItem[];
  unsupported: UnsupportedParamItem[];
  unrecognized: string[];
  errors: string[];
  warnings: string[];
  mergedSettings: GenerationSettings;
}

// Known llama.cpp parameters supported by the inference runtime
const SUPPORTED_LLAMA_CPP_PARAMS = new Set([
  "temperature",
  "top_p",
  "topP",
  "top_k",
  "topK",
  "min_p",
  "minP",
  "max_tokens",
  "maxTokens",
  "n_predict",
  "repeat_penalty",
  "repetition_penalty",
  "repetitionPenalty",
  "presence_penalty",
  "presencePenalty",
  "frequency_penalty",
  "frequencyPenalty",
  "seed",
  "stop_tokens",
  "stopSequences",
  "stop",
  "system_prompt",
  "systemPrompt",
]);

// Known llama.cpp parameters that are recognized but not yet bound to the active engine sampler
const KNOWN_UNSUPPORTED_PARAMS: Record<string, string> = {
  typical_p: "Typical P sampling is not enabled in the current llama.cpp engine sampler chain.",
  repeat_last_n: "Penalty window length is fixed to 64 tokens in the active engine backend.",
  tfs_z: "Tail Free Sampling is not supported by the active engine configuration.",
  mirostat: "Mirostat adaptive sampling is not active in the current engine pipeline.",
  mirostat_tau: "Mirostat target entropy is not active in the current engine pipeline.",
  mirostat_eta: "Mirostat learning rate is not active in the current engine pipeline.",
  dry_multiplier: "DRY repetition penalty is not implemented in the current engine pipeline.",
};

const DANGEROUS_KEYS = new Set(["__proto__", "constructor", "prototype"]);

/**
 * Validate and parse an incoming JSON configuration string.
 * Supports full config objects, partial configs, or bare parameters.
 */
export function parseAndValidateConfig(
  rawJson: string,
  currentSettings: GenerationSettings,
): ConfigPreviewResult {
  const result: ConfigPreviewResult = {
    valid: false,
    name: "Custom Configuration",
    description: "",
    engine: "llama.cpp",
    changes: [],
    unchanged: [],
    unsupported: [],
    unrecognized: [],
    errors: [],
    warnings: [],
    mergedSettings: { ...currentSettings },
  };

  // 1. Size restriction (max 100 KB to avoid denial of service)
  if (!rawJson || rawJson.trim().length === 0) {
    result.errors.push("Empty JSON configuration string.");
    return result;
  }
  if (rawJson.length > 100 * 1024) {
    result.errors.push("Configuration payload exceeds maximum allowed size (100 KB).");
    return result;
  }

  // 2. Parse JSON safely
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawJson, (key, value) => {
      if (DANGEROUS_KEYS.has(key)) {
        throw new Error(`Prototype pollution attempt detected with key '${key}'.`);
      }
      return value;
    });
  } catch (err: unknown) {
    result.errors.push(`Invalid JSON syntax: ${err instanceof Error ? err.message : String(err)}`);
    return result;
  }

  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    result.errors.push("Root JSON value must be an object.");
    return result;
  }

  const root = parsed as Record<string, unknown>;

  // 3. Extract metadata
  if (typeof root.name === "string" && root.name.trim()) {
    result.name = root.name.trim();
  }
  if (typeof root.description === "string") {
    result.description = root.description.trim();
  }
  if (typeof root.engine === "string") {
    result.engine = root.engine.trim();
    if (result.engine.toLowerCase() !== "llama.cpp") {
      result.warnings.push(
        `Engine '${root.engine}' indicated. Parameters will be adapted to llama.cpp runtime.`,
      );
    }
  }
  if (typeof root.modelAssociation === "string") {
    result.modelAssociation = root.modelAssociation.trim();
  }

  // 4. Locate parameter bag (root.parameters || root)
  let paramBag: Record<string, unknown>;
  if (root.parameters && typeof root.parameters === "object" && !Array.isArray(root.parameters)) {
    paramBag = root.parameters as Record<string, unknown>;
  } else {
    // Partial bare parameters object
    paramBag = root;
  }

  const merged = { ...currentSettings };
  const updatedKeys = new Set<keyof GenerationSettings>();

  // 5. Inspect parameter fields
  for (const [key, val] of Object.entries(paramBag)) {
    // Skip root metadata keys if inspecting root directly
    if (
      ["name", "description", "version", "engine", "modelAssociation", "parameters"].includes(key)
    ) {
      continue;
    }

    // Check for known unsupported parameters
    if (KNOWN_UNSUPPORTED_PARAMS[key]) {
      result.unsupported.push({
        field: key,
        value: val,
        reason: KNOWN_UNSUPPORTED_PARAMS[key],
      });
      continue;
    }

    // Check for completely unrecognized fields
    if (!SUPPORTED_LLAMA_CPP_PARAMS.has(key)) {
      result.unrecognized.push(key);
      continue;
    }

    // Validate supported fields
    switch (key) {
      case "temperature": {
        if (typeof val !== "number" || !Number.isFinite(val)) {
          result.errors.push(
            `'temperature' must be a finite number between 0.0 and 2.0 (got ${val}).`,
          );
        } else if (val < 0.0 || val > 2.0) {
          result.errors.push(`'temperature' ${val} is outside supported range [0.0, 2.0].`);
        } else {
          merged.temperature = Math.round(val * 100) / 100;
          updatedKeys.add("temperature");
        }
        break;
      }

      case "top_p":
      case "topP": {
        if (typeof val !== "number" || !Number.isFinite(val)) {
          result.errors.push(`'top_p' must be a finite number between 0.01 and 1.0 (got ${val}).`);
        } else if (val <= 0.0 || val > 1.0) {
          result.errors.push(`'top_p' ${val} is outside supported range (0.0, 1.0].`);
        } else {
          merged.topP = Math.round(val * 100) / 100;
          updatedKeys.add("topP");
        }
        break;
      }

      case "top_k":
      case "topK": {
        if (typeof val !== "number" || !Number.isInteger(val)) {
          result.errors.push(`'top_k' must be an integer between 1 and 500 (got ${val}).`);
        } else if (val < 1 || val > 500) {
          result.errors.push(`'top_k' ${val} is outside supported range [1, 500].`);
        } else {
          merged.topK = val;
          updatedKeys.add("topK");
        }
        break;
      }

      case "min_p":
      case "minP": {
        if (typeof val !== "number" || !Number.isFinite(val)) {
          result.errors.push(`'min_p' must be a finite number between 0.0 and 1.0 (got ${val}).`);
        } else if (val < 0.0 || val > 1.0) {
          result.errors.push(`'min_p' ${val} is outside supported range [0.0, 1.0].`);
        } else {
          merged.minP = Math.round(val * 1000) / 1000;
          updatedKeys.add("minP");
        }
        break;
      }

      case "max_tokens":
      case "maxTokens":
      case "n_predict": {
        if (typeof val !== "number" || !Number.isInteger(val)) {
          result.errors.push(`'max_tokens' / 'n_predict' must be an integer (got ${val}).`);
        } else if (val < 1 || val > 32768) {
          result.errors.push(`'max_tokens' ${val} is outside supported range [1, 32768].`);
        } else {
          merged.maxTokens = val;
          updatedKeys.add("maxTokens");
        }
        break;
      }

      case "repeat_penalty":
      case "repetition_penalty":
      case "repetitionPenalty": {
        if (typeof val !== "number" || !Number.isFinite(val)) {
          result.errors.push(
            `'repetition_penalty' must be a finite number between 1.0 and 2.0 (got ${val}).`,
          );
        } else if (val < 1.0 || val > 2.0) {
          result.errors.push(`'repetition_penalty' ${val} is outside supported range [1.0, 2.0].`);
        } else {
          merged.repetitionPenalty = Math.round(val * 100) / 100;
          updatedKeys.add("repetitionPenalty");
        }
        break;
      }

      case "frequency_penalty":
      case "frequencyPenalty": {
        if (typeof val !== "number" || !Number.isFinite(val)) {
          result.errors.push(
            `'frequency_penalty' must be a finite number between -2.0 and 2.0 (got ${val}).`,
          );
        } else if (val < -2.0 || val > 2.0) {
          result.errors.push(`'frequency_penalty' ${val} is outside supported range [-2.0, 2.0].`);
        } else {
          merged.frequencyPenalty = Math.round(val * 100) / 100;
          updatedKeys.add("frequencyPenalty");
        }
        break;
      }

      case "presence_penalty":
      case "presencePenalty": {
        if (typeof val !== "number" || !Number.isFinite(val)) {
          result.errors.push(
            `'presence_penalty' must be a finite number between -2.0 and 2.0 (got ${val}).`,
          );
        } else if (val < -2.0 || val > 2.0) {
          result.errors.push(`'presence_penalty' ${val} is outside supported range [-2.0, 2.0].`);
        } else {
          merged.presencePenalty = Math.round(val * 100) / 100;
          updatedKeys.add("presencePenalty");
        }
        break;
      }

      case "seed": {
        if (val === null || val === -1) {
          merged.seed = null;
          updatedKeys.add("seed");
        } else if (typeof val === "number" && Number.isInteger(val)) {
          merged.seed = val;
          updatedKeys.add("seed");
        } else {
          result.errors.push(`'seed' must be an integer or -1/null for random (got ${val}).`);
        }
        break;
      }

      case "stop_tokens":
      case "stopSequences":
      case "stop": {
        if (Array.isArray(val)) {
          const strings = val.filter(
            (item): item is string => typeof item === "string" && item.length > 0,
          );
          merged.stopSequences = strings;
          updatedKeys.add("stopSequences");
        } else if (typeof val === "string" && val.length > 0) {
          merged.stopSequences = [val];
          updatedKeys.add("stopSequences");
        } else {
          result.errors.push(
            `'stop_tokens' must be an array of strings or string (got ${typeof val}).`,
          );
        }
        break;
      }

      case "system_prompt":
      case "systemPrompt": {
        if (typeof val === "string") {
          merged.systemPrompt = val;
          updatedKeys.add("systemPrompt");
        } else {
          result.errors.push(`'system_prompt' must be a string (got ${typeof val}).`);
        }
        break;
      }
    }
  }

  // 6. Assemble human-readable changes & unchanged parameters
  const labels: Record<keyof GenerationSettings, string> = {
    temperature: "Temperature",
    topP: "Top-P Sampling",
    topK: "Top-K Sampling",
    minP: "Min-P Sampling",
    maxTokens: "Max Tokens (n_predict)",
    repetitionPenalty: "Repetition Penalty",
    frequencyPenalty: "Frequency Penalty",
    presencePenalty: "Presence Penalty",
    seed: "Random Seed",
    stopSequences: "Stop Sequences",
    systemPrompt: "System Instructions",
  };

  (Object.keys(currentSettings) as (keyof GenerationSettings)[]).forEach((key) => {
    const curVal = currentSettings[key];
    const newVal = merged[key];

    if (updatedKeys.has(key)) {
      result.changes.push({
        key,
        label: labels[key],
        current: curVal,
        imported: newVal,
      });
    } else {
      result.unchanged.push({
        key,
        label: labels[key],
        value: curVal,
      });
    }
  });

  if (
    result.errors.length === 0 &&
    result.changes.length === 0 &&
    result.unsupported.length === 0
  ) {
    result.warnings.push(
      "No recognized generation parameters were found in the imported configuration.",
    );
  }

  result.valid = result.errors.length === 0;
  result.mergedSettings = merged;

  return result;
}

/**
 * Format active generation settings into a documented llama.cpp JSON configuration.
 */
export function exportConfigToJson(
  name: string,
  settings: GenerationSettings,
  description?: string,
  modelAssociation?: string | null,
): string {
  const exportPayload: ExportedConfigJson = {
    name: name || "Custom Configuration",
    description: description || "Exported generation settings for llama.cpp local workstation",
    version: 1,
    engine: "llama.cpp",
    modelAssociation: modelAssociation || null,
    parameters: {
      temperature: settings.temperature,
      top_k: settings.topK,
      top_p: settings.topP,
      min_p: settings.minP,
      max_tokens: settings.maxTokens,
      n_predict: settings.maxTokens,
      repeat_penalty: settings.repetitionPenalty,
      repetition_penalty: settings.repetitionPenalty,
      presence_penalty: settings.presencePenalty,
      frequency_penalty: settings.frequencyPenalty,
      seed: settings.seed === null ? -1 : settings.seed,
      stop_tokens: settings.stopSequences,
      system_prompt: settings.systemPrompt,
    },
  };

  return JSON.stringify(exportPayload, null, 2);
}

/**
 * Trigger a browser/desktop file download for a JSON string.
 */
export function triggerJsonDownload(filename: string, jsonContent: string) {
  const blob = new Blob([jsonContent], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename.endsWith(".json") ? filename : `${filename}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
