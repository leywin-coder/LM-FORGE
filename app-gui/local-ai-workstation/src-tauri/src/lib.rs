use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use reqwest::header::{HeaderMap, HeaderValue, RANGE, AUTHORIZATION};
use reqwest::Client;
use tauri::Emitter;
use tokio::io::AsyncWriteExt;
use futures_util::StreamExt;
use lm_studio::engine::{InferenceEngine, RuntimeParams};
use lm_studio::discovery::{GgufHeaderParser, HardwareTelemetry, MemoryBudgetEstimator};

fn get_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MODELS_DIR") {
        PathBuf::from(dir)
    } else {
        let p = PathBuf::from("/home/leywin/Projects/lm-studio/models");
        if p.exists() {
            p
        } else {
            let legacy = PathBuf::from("/data/leywin/Projects/lm-studio/models");
            if legacy.exists() {
                legacy
            } else {
                dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("lm-studio/models")
            }
        }
    }
}

static ACTIVE_DOWNLOAD_CANCELLATIONS: Mutex<Option<HashSet<String>>> = Mutex::new(None);

fn is_download_cancelled(filename: &str) -> bool {
    if let Ok(guard) = ACTIVE_DOWNLOAD_CANCELLATIONS.lock() {
        if let Some(set) = guard.as_ref() {
            return set.contains(filename);
        }
    }
    false
}

fn set_download_cancelled(filename: &str, cancelled: bool) {
    if let Ok(mut guard) = ACTIVE_DOWNLOAD_CANCELLATIONS.lock() {
        let set = guard.get_or_insert_with(HashSet::new);
        if cancelled {
            set.insert(filename.to_string());
        } else {
            set.remove(filename);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfRepoFile {
    pub path: String,
    pub size: Option<u64>,
    pub r#type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgressPayload {
    pub model_id: String,
    pub filename: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: f32,
    pub speed_bytes_sec: u64,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelMetadata {
    pub id: String,
    pub filename: String,
    pub display_name: String,
    pub path: String,
    pub size: u64,
    pub quant: String,
    pub is_split: bool,
    pub split_part: Option<u32>,
    pub split_total: Option<u32>,
    pub base_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModel {
    pub id: String,
    pub downloads: u64,
    pub rating: Option<f32>,
    pub popularity: Option<f32>,
    pub parent_model: Option<String>,
    pub sha: Option<String>,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub disabled: Option<bool>,
    pub update_month: Option<String>,
    pub cost: Option<[f32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
    pub stop_tokens: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub port: Option<u16>,
    pub threads: Option<i32>,
    pub n_ctx: Option<u32>,
    pub flash_attention: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub path: String,
    pub size: u64,
}

static ENGINE: RwLock<Option<Arc<InferenceEngine>>> = RwLock::new(None);

#[tauri::command]
fn get_models() -> Vec<ModelInfo> {
    let models_dir = get_models_dir();

    std::fs::read_dir(&models_dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "gguf" || ext == "ggml")
                        .unwrap_or(false)
                })
                .map(|e| {
                    let path = e.path();
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    ModelInfo {
                        id: path.file_name().unwrap_or_default().to_string_lossy().to_string(),
                        path: path.display().to_string(),
                        size,
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn get_app_config_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lm-forge")
}

#[tauri::command]
fn save_app_state(key: String, data: String) -> Result<(), String> {
    let dir = get_app_config_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create config dir: {}", e))?;
    let path = dir.join(format!("{}.json", key));
    std::fs::write(&path, data).map_err(|e| format!("Failed to write app state {}: {}", key, e))?;
    Ok(())
}

#[tauri::command]
fn load_app_state(key: String) -> Result<Option<String>, String> {
    let path = get_app_config_dir().join(format!("{}.json", key));
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read app state {}: {}", key, e))?;
    Ok(Some(data))
}

#[tauri::command]
async fn run_model_benchmark(
    path: String,
    prompt_tokens: Option<usize>,
    gen_tokens: Option<usize>,
    threads: Option<i32>,
) -> Result<serde_json::Value, String> {
    let model_path = PathBuf::from(&path);
    if !model_path.exists() {
        return Err(format!("Model not found at: {}", path));
    }

    let p_toks = prompt_tokens.unwrap_or(512);
    let g_toks = gen_tokens.unwrap_or(128);
    let th = threads.unwrap_or(4);

    let start = Instant::now();
    let params = RuntimeParams {
        n_ctx: (p_toks + g_toks + 64) as u32,
        n_threads: th,
        max_tokens: g_toks,
        ..Default::default()
    };

    let engine = InferenceEngine::load(&model_path, params)
        .map_err(|e| format!("Benchmark failed to load model: {}", e))?;

    let test_prompt = "Hello ".repeat(p_toks / 2);
    let prompt_start = Instant::now();
    let mut rx = engine.generate_stream(&test_prompt, &engine.runtime_params, None).await;
    let mut count = 0;
    while let Some(_) = rx.recv().await {
        count += 1;
    }
    let elapsed = prompt_start.elapsed().as_secs_f64();
    let total_time = start.elapsed().as_secs_f64();
    let tok_per_sec = if elapsed > 0.0 { (count as f64) / elapsed } else { 0.0 };

    Ok(serde_json::json!({
        "model": model_path.file_name().unwrap_or_default().to_string_lossy(),
        "prompt_tokens": p_toks,
        "gen_tokens": count,
        "threads": th,
        "tokens_per_second": (tok_per_sec * 100.0).round() / 100.0,
        "total_elapsed_seconds": (total_time * 100.0).round() / 100.0,
    }))
}

#[tauri::command]
async fn load_model(path: String) -> Result<String, String> {
    let model_path = PathBuf::from(&path);
    let engine = InferenceEngine::load(&model_path, RuntimeParams::default())?;
    let name = model_path.file_name().unwrap_or_default().to_string_lossy().to_string();

    let mut guard = ENGINE.write().map_err(|e| format!("Engine lock error: {}", e))?;
    *guard = Some(Arc::new(engine));

    Ok(name)
}

#[tauri::command]
async fn cancel_generation() -> Result<(), String> {
    if let Ok(guard) = ENGINE.read() {
        if let Some(ref engine) = *guard {
            engine.cancel_active();
        }
    }
    Ok(())
}

#[tauri::command]
fn get_model_metadata(path: String) -> Result<serde_json::Value, String> {
    let p = PathBuf::from(&path);
    let file = std::fs::File::open(&p).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut buffer = vec![0u8; 65536];
    use std::io::Read;
    let mut reader = std::io::BufReader::new(file);
    let bytes_read = reader.read(&mut buffer).map_err(|e| format!("Read error: {}", e))?;
    buffer.truncate(bytes_read);

    let meta = GgufHeaderParser::parse(&buffer)?;
    let file_size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
    let telemetry = HardwareTelemetry::probe();
    let estimate = MemoryBudgetEstimator::estimate(&meta, file_size, &telemetry, Some(meta.context_length));

    Ok(serde_json::json!({
        "architecture": meta.architecture,
        "quantization": meta.quantization,
        "layer_count": meta.layer_count,
        "context_length": meta.context_length,
        "head_count": meta.head_count,
        "head_count_kv": meta.head_count_kv,
        "embedding_length": meta.embedding_length,
        "chat_template": meta.chat_template,
        "file_size_bytes": file_size,
        "estimated_ram_bytes": estimate.total_estimated_ram_bytes,
        "fit_classification": format!("{:?}", estimate.classification),
        "suggested_gpu_layers": estimate.suggested_gpu_layers,
        "recommended_ctx": estimate.recommended_ctx,
        "reason": estimate.reason,
    }))
}

#[tauri::command]
async fn chat_completions(window: tauri::Window, request: ChatRequest) -> Result<String, String> {
    let engine = ENGINE.read().unwrap().clone();
    let Some(ref engine) = engine else {
        return Err("No model loaded. Please load a GGUF model first.".to_string());
    };

    let messages_tuples: Vec<(&str, &str)> = request
        .messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_str()))
        .collect();

    let formatted_prompt = engine.apply_chat_template(&messages_tuples)?;

    let mut runtime_params = engine.runtime_params.clone();
    if let Some(temp) = request.temperature {
        runtime_params.temperature = temp;
    }
    if let Some(top_p) = request.top_p {
        runtime_params.top_p = top_p;
    }
    if let Some(top_k) = request.top_k {
        runtime_params.top_k = top_k;
    }
    if let Some(min_p) = request.min_p {
        runtime_params.min_p = Some(min_p);
    }
    if let Some(rep_pen) = request.repetition_penalty {
        runtime_params.repetition_penalty = Some(rep_pen);
    }
    if let Some(freq_pen) = request.frequency_penalty {
        runtime_params.frequency_penalty = Some(freq_pen);
    }
    if let Some(pres_pen) = request.presence_penalty {
        runtime_params.presence_penalty = Some(pres_pen);
    }
    if let Some(max_tok) = request.max_tokens {
        runtime_params.max_tokens = max_tok;
    }
    if let Some(seed) = request.seed {
        runtime_params.seed = Some(seed);
    }
    if let Some(stop_tokens) = request.stop_tokens {
        runtime_params.stop_tokens = stop_tokens;
    }
    if let Some(threads) = request.threads {
        runtime_params.n_threads = threads;
    }
    if let Some(n_ctx) = request.n_ctx {
        runtime_params.n_ctx = n_ctx;
    }
    if let Some(fa) = request.flash_attention {
        runtime_params.flash_attention = fa;
    }

    let mut rx = engine.generate_stream(&formatted_prompt, &runtime_params, None).await;
    let mut full_response = String::new();
    while let Some(piece) = rx.recv().await {
        let _ = window.emit("chat_token", &piece);
        full_response.push_str(&piece);
    }

    Ok(full_response)
}

#[tauri::command]
async fn check_server_status(_port: Option<u16>) -> bool {
    ENGINE.read().map(|g| g.is_some()).unwrap_or(false)
}

#[tauri::command]
fn get_loaded_model() -> Option<String> {
    ENGINE.read().ok().and_then(|g| {
        g.as_ref().map(|e| {
            e.model_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
        })
    })
}

#[tauri::command]
async fn unload_model() -> Result<(), String> {
    let mut guard = ENGINE.write().map_err(|e| format!("Engine lock error: {}", e))?;
    *guard = None;
    Ok(())
}

#[tauri::command]
async fn hf_search(query: String, limit: Option<usize>) -> Result<Vec<HfModel>, String> {
    let limit = limit.unwrap_or(20).min(40);
    let encoded = query.replace(' ', "%20");
    let url = format!(
        "https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit={}",
        encoded, limit
    );
    let client = Client::new();
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("HTTP {}", status));
    }
    let models: Vec<serde_json::Value> = resp
        .json()
        .await
        .map_err(|e| format!("JSON parse failed: {}", e))?;
    let mut results = Vec::new();
    for item in models {
        let hf = HfModel {
            id: item
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            downloads: item
                .get("downloads")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            rating: item
                .get("rating")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            popularity: item
                .get("popularity")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            parent_model: item
                .get("parent_model")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            sha: item.get("sha").and_then(|v| v.as_str()).map(|s| s.to_string()),
            pipeline_tag: item
                .get("pipeline_tag")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            tags: item
                .get("tags")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect(),
            disabled: item.get("disabled").and_then(|v| v.as_bool()),
            update_month: item
                .get("update_month")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            cost: None,
        };
        results.push(hf);
    }
    Ok(results)
}

#[tauri::command]
async fn hf_model_files(
    model_id: String,
    token: Option<String>,
) -> Result<Vec<HfRepoFile>, String> {
    let clean_id = model_id.trim();
    let url = format!("https://huggingface.co/api/models/{}/tree/main?recursive=true", clean_id);

    let mut client_builder = Client::builder()
        .user_agent("LocalAIWorkstation/2.0 (Linux; x86_64)")
        .timeout(Duration::from_secs(20));

    let mut headers = HeaderMap::new();
    if let Some(tok) = token {
        let tok_str = tok.trim();
        if !tok_str.is_empty() {
            if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", tok_str)) {
                headers.insert(AUTHORIZATION, val);
            }
        }
    }
    client_builder = client_builder.default_headers(headers);

    let client = client_builder.build().map_err(|e| format!("HTTP client init failed: {}", e))?;
    let resp = client.get(&url).send().await.map_err(|e| format!("HF tree error: {}", e))?;

    let status = resp.status();
    if !status.is_success() {
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(format!(
                "HTTP {} Gated/Unauthorized repo: This model requires a Hugging Face API Token.",
                status
            ));
        }
        if status.as_u16() == 404 {
            return Err(format!("HTTP 404: Repository '{}' not found on Hugging Face.", clean_id));
        }
        return Err(format!("Hugging Face API returned HTTP {}", status));
    }

    let items: Vec<serde_json::Value> = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse Hugging Face file tree JSON: {}", e))?;

    let mut gguf_files = Vec::new();
    for item in items {
        let path = item.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let r#type = item.get("type").and_then(|v| v.as_str()).unwrap_or("file").to_string();
        let size = item.get("size").and_then(|v| v.as_u64());

        if path.to_lowercase().ends_with(".gguf") {
            gguf_files.push(HfRepoFile {
                path,
                size,
                r#type,
            });
        }
    }

    // Sort files alphabetically
    gguf_files.sort_by(|a, b| a.path.to_lowercase().cmp(&b.path.to_lowercase()));
    Ok(gguf_files)
}

#[tauri::command]
async fn cancel_download(filename: String) -> Result<(), String> {
    set_download_cancelled(&filename, true);
    Ok(())
}

#[tauri::command]
async fn download_model(
    window: tauri::Window,
    model_id: String,
    filename: Option<String>,
    token: Option<String>,
) -> Result<String, String> {
    let file = filename.unwrap_or_else(|| {
        let name = model_id.split('/').last().unwrap_or(&model_id);
        if name.ends_with(".gguf") {
            name.to_string()
        } else {
            format!("{}.gguf", name)
        }
    });

    let url = if model_id.starts_with("http://") || model_id.starts_with("https://") {
        model_id.clone()
    } else {
        format!("https://huggingface.co/{}/resolve/main/{}", model_id.trim(), file.trim())
    };

    let models_dir = get_models_dir();
    std::fs::create_dir_all(&models_dir).map_err(|e| format!("Failed to create models directory: {}", e))?;

    let target_path = models_dir.join(&file);
    let part_path = models_dir.join(format!("{}.part", &file));

    // Reset cancellation flag
    set_download_cancelled(&file, false);

    // Check if partial file exists for resume support
    let mut downloaded_bytes: u64 = 0;
    if part_path.exists() {
        if let Ok(meta) = std::fs::metadata(&part_path) {
            downloaded_bytes = meta.len();
        }
    }

    let client = Client::new();

    let mut req = client.get(&url);
    if let Some(tok) = token {
        let tok_str = tok.trim();
        if !tok_str.is_empty() {
            req = req.header(AUTHORIZATION, format!("Bearer {}", tok_str));
        }
    }

    if downloaded_bytes > 0 {
        req = req.header(RANGE, format!("bytes={}-", downloaded_bytes));
    }

    let resp = req.send().await.map_err(|e| format!("Download request error: {}", e))?;

    let status = resp.status();
    let is_range_ok = status == reqwest::StatusCode::PARTIAL_CONTENT;
    let is_success_fresh = status == reqwest::StatusCode::OK;

    if !is_range_ok && !is_success_fresh {
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(format!(
                "HTTP {} Gated Model: Hugging Face authentication required. Please provide a valid Access Token.",
                status
            ));
        }
        if status.as_u16() == 404 {
            return Err(format!(
                "HTTP 404 Not Found: File '{}' does not exist in repository '{}'. Please select an actual file from the repository file list.",
                file, model_id
            ));
        }
        return Err(format!("Hugging Face returned HTTP error status: {}", status));
    }

    // Determine total file size
    let content_len = resp.content_length().unwrap_or(0);
    let total_bytes = if is_range_ok {
        downloaded_bytes + content_len
    } else {
        downloaded_bytes = 0;
        content_len
    };

    // Open file for appending or writing
    let mut file_out = if is_range_ok && downloaded_bytes > 0 {
        tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part_path)
            .await
            .map_err(|e| format!("Failed to open .part file for resume: {}", e))?
    } else {
        tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&part_path)
            .await
            .map_err(|e| format!("Failed to create .part file: {}", e))?
    };

    let mut stream = resp.bytes_stream();
    let mut last_progress_instant = Instant::now();
    let mut bytes_since_last_tick: u64 = 0;

    let _ = window.emit(
        "download_progress",
        DownloadProgressPayload {
            model_id: model_id.clone(),
            filename: file.clone(),
            downloaded_bytes,
            total_bytes,
            percent: if total_bytes > 0 { (downloaded_bytes as f32 / total_bytes as f32) * 100.0 } else { 0.0 },
            speed_bytes_sec: 0,
            status: "downloading".to_string(),
            error: None,
        },
    );

    while let Some(chunk_result) = stream.next().await {
        // Check for cancellation
        if is_download_cancelled(&file) {
            let _ = window.emit(
                "download_progress",
                DownloadProgressPayload {
                    model_id: model_id.clone(),
                    filename: file.clone(),
                    downloaded_bytes,
                    total_bytes,
                    percent: if total_bytes > 0 { (downloaded_bytes as f32 / total_bytes as f32) * 100.0 } else { 0.0 },
                    speed_bytes_sec: 0,
                    status: "cancelled".to_string(),
                    error: Some("Download cancelled by user".to_string()),
                },
            );
            return Err("Download cancelled by user".to_string());
        }

        let chunk = chunk_result.map_err(|e| format!("Streaming chunk failed: {}", e))?;
        file_out.write_all(&chunk).await.map_err(|e| format!("Failed writing to disk: {}", e))?;

        let chunk_len = chunk.len() as u64;
        downloaded_bytes += chunk_len;
        bytes_since_last_tick += chunk_len;

        let elapsed = last_progress_instant.elapsed();
        if elapsed >= Duration::from_millis(300) {
            let current_speed = (bytes_since_last_tick as f64 / elapsed.as_secs_f64()) as u64;
            bytes_since_last_tick = 0;
            last_progress_instant = Instant::now();

            let percent = if total_bytes > 0 {
                ((downloaded_bytes as f32 / total_bytes as f32) * 100.0).clamp(0.0, 100.0)
            } else {
                0.0
            };

            let _ = window.emit(
                "download_progress",
                DownloadProgressPayload {
                    model_id: model_id.clone(),
                    filename: file.clone(),
                    downloaded_bytes,
                    total_bytes,
                    percent,
                    speed_bytes_sec: current_speed,
                    status: "downloading".to_string(),
                    error: None,
                },
            );
        }
    }

    file_out.flush().await.map_err(|e| format!("Failed flushing download stream: {}", e))?;
    drop(file_out);

    // Validate downloaded file existence and non-empty
    let final_meta = std::fs::metadata(&part_path).map_err(|e| format!("Verification failed: {}", e))?;
    if final_meta.len() == 0 {
        let _ = std::fs::remove_file(&part_path);
        return Err("Downloaded file is empty (0 bytes). Download discarded.".to_string());
    }

    // Atomic rename from .part to final destination
    std::fs::rename(&part_path, &target_path).map_err(|e| {
        format!("Failed finalizing file from .part to target path: {}", e)
    })?;

    // Emit completed event
    let _ = window.emit(
        "download_progress",
        DownloadProgressPayload {
            model_id: model_id.clone(),
            filename: file.clone(),
            downloaded_bytes,
            total_bytes: downloaded_bytes,
            percent: 100.0,
            speed_bytes_sec: 0,
            status: "done".to_string(),
            error: None,
        },
    );

    Ok(target_path.display().to_string())
}

#[tauri::command]
fn get_models_detailed() -> Vec<LocalModelMetadata> {
    let models_dir = get_models_dir();
    let mut list = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if ext_str == "gguf" || ext_str == "ggml" {
                    let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

                    // Detect split GGUF (e.g. model-00001-of-00004.gguf)
                    let mut is_split = false;
                    let mut split_part = None;
                    let mut split_total = None;
                    let mut base_name = filename.clone();

                    if let Some(idx) = filename.find("-of-") {
                        let before = &filename[..idx];
                        if let Some(dash_idx) = before.rfind('-') {
                            let part_str = &before[dash_idx + 1..];
                            let after = &filename[idx + 4..];
                            let total_str = after.split('.').next().unwrap_or("");

                            if let (Ok(p), Ok(tot)) = (part_str.parse::<u32>(), total_str.parse::<u32>()) {
                                is_split = true;
                                split_part = Some(p);
                                split_total = Some(tot);
                                base_name = before[..dash_idx].to_string();
                            }
                        }
                    }

                    // Extract quantization tag
                    let quant = if filename.to_uppercase().contains("Q4_K_M") {
                        "Q4_K_M".to_string()
                    } else if filename.to_uppercase().contains("Q8_0") {
                        "Q8_0".to_string()
                    } else if filename.to_uppercase().contains("Q5_K_M") {
                        "Q5_K_M".to_string()
                    } else if filename.to_uppercase().contains("Q2_K") {
                        "Q2_K".to_string()
                    } else if filename.to_uppercase().contains("Q4_0") {
                        "Q4_0".to_string()
                    } else if filename.to_uppercase().contains("Q6_K") {
                        "Q6_K".to_string()
                    } else if filename.to_uppercase().contains("F16") {
                        "F16".to_string()
                    } else {
                        "GGUF".to_string()
                    };

                    let display_name = filename
                        .trim_end_matches(".gguf")
                        .trim_end_matches(".GGUF")
                        .replace(['-', '_'], " ");

                    list.push(LocalModelMetadata {
                        id: filename.clone(),
                        filename,
                        display_name,
                        path: path.display().to_string(),
                        size,
                        quant,
                        is_split,
                        split_part,
                        split_total,
                        base_name,
                    });
                }
            }
        }
    }

    list.sort_by(|a, b| a.filename.to_lowercase().cmp(&b.filename.to_lowercase()));
    list
}

#[tauri::command]
async fn delete_local_model(filename: String) -> Result<(), String> {
    let models_dir = get_models_dir();
    let target = models_dir.join(&filename);
    if !target.exists() {
        return Err(format!("Model file '{}' not found.", filename));
    }

    // Safety: ensure target is within models_dir
    if !target.starts_with(&models_dir) {
        return Err("Permission denied: Path is outside the models directory.".to_string());
    }

    std::fs::remove_file(&target).map_err(|e| format!("Failed to delete model file: {}", e))?;
    Ok(())
}

#[tauri::command]
async fn rename_local_model(old_filename: String, new_filename: String) -> Result<String, String> {
    let models_dir = get_models_dir();
    let old_path = models_dir.join(&old_filename);
    let mut new_name = new_filename.trim().to_string();
    if !new_name.ends_with(".gguf") && !new_name.ends_with(".ggml") {
        new_name.push_str(".gguf");
    }
    let new_path = models_dir.join(&new_name);

    if !old_path.exists() {
        return Err(format!("Original model file '{}' does not exist.", old_filename));
    }
    if new_path.exists() {
        return Err(format!("Target filename '{}' already exists.", new_name));
    }

    std::fs::rename(&old_path, &new_path).map_err(|e| format!("Rename failed: {}", e))?;
    Ok(new_name)
}

#[tauri::command]
async fn import_external_model(source_path: String) -> Result<String, String> {
    let src = PathBuf::from(&source_path);
    if !src.exists() {
        return Err(format!("Source file '{}' does not exist.", source_path));
    }

    let file_name = src
        .file_name()
        .ok_or_else(|| "Invalid source file name".to_string())?
        .to_string_lossy()
        .to_string();

    let models_dir = get_models_dir();
    std::fs::create_dir_all(&models_dir).map_err(|e| e.to_string())?;
    let dest = models_dir.join(&file_name);

    if dest.exists() {
        return Err(format!("Model '{}' already exists in local models directory.", file_name));
    }

    // Copy file
    std::fs::copy(&src, &dest).map_err(|e| format!("Failed to copy model file: {}", e))?;
    Ok(dest.display().to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: Option<f32>,
    pub cpu_model: Option<String>,
    pub cpu_cores: usize,
    pub process_cpu_usage: Option<f32>,
    pub total_ram: u64,
    pub used_ram: u64,
    pub available_ram: u64,
    pub free_ram: u64,
    pub app_ram: u64,
    pub disk_total: u64,
    pub disk_available: u64,
    pub disk_used: u64,
    pub models_dir_size: u64,
    pub gpu_name: Option<String>,
    pub gpu_utilization: Option<f32>,
    pub gpu_memory_total: Option<u64>,
    pub gpu_memory_used: Option<u64>,
    pub gpu_temp: Option<f32>,
    pub gpu_power: Option<f32>,
    pub timestamp: u64,
}

static PREV_CPU_STAT: RwLock<Option<(u64, u64)>> = RwLock::new(None);

#[derive(serde::Serialize)]
struct MachineSpecs {
    total_ram_bytes: u64,
    available_ram_bytes: u64,
    cpu_model: String,
    physical_cores: u32,
    logical_cores: u32,
}

#[tauri::command]
fn get_machine_specs() -> MachineSpecs {
    let mut total_ram_bytes = 0u64;
    let mut available_ram_bytes = 0u64;
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            let mut parts = line.split_whitespace();
            if let Some(key) = parts.next() {
                if let Some(val) = parts.next().and_then(|v| v.parse::<u64>().ok()) {
                    match key {
                        "MemTotal:" => total_ram_bytes = val * 1024,
                        "MemAvailable:" => available_ram_bytes = val * 1024,
                        _ => {}
                    }
                }
            }
        }
    }

    let mut cpu_model = String::from("Unknown CPU");
    let mut physical_cores = 0u32;
    let mut logical_cores = 0u32;
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        let mut seen_physical: std::collections::HashSet<String> = std::collections::HashSet::new();
        for line in cpuinfo.lines() {
            if cpu_model == "Unknown CPU" && line.starts_with("model name") {
                if let Some((_, v)) = line.split_once(':') {
                    cpu_model = v.trim().to_string();
                }
            }
            if line.starts_with("processor") {
                logical_cores += 1;
            }
            if line.starts_with("core id") {
                if let Some((_, v)) = line.split_once(':') {
                    seen_physical.insert(v.trim().to_string());
                }
            }
        }
        physical_cores = if seen_physical.is_empty() { logical_cores.max(1) } else { seen_physical.len() as u32 };
    }

    MachineSpecs {
        total_ram_bytes,
        available_ram_bytes,
        cpu_model,
        physical_cores,
        logical_cores,
    }
}

#[tauri::command]
fn get_system_metrics() -> SystemMetrics {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    // 1. Real CPU Metrics from /proc/stat
    let mut cpu_usage = None;
    if let Ok(stat_content) = std::fs::read_to_string("/proc/stat") {
        if let Some(first_line) = stat_content.lines().next() {
            let parts: Vec<&str> = first_line.split_whitespace().collect();
            if parts.len() >= 5 && parts[0] == "cpu" {
                let user: u64 = parts[1].parse().unwrap_or(0);
                let nice: u64 = parts[2].parse().unwrap_or(0);
                let system: u64 = parts[3].parse().unwrap_or(0);
                let idle: u64 = parts[4].parse().unwrap_or(0);
                let iowait: u64 = parts.get(5).and_then(|v| v.parse().ok()).unwrap_or(0);
                let irq: u64 = parts.get(6).and_then(|v| v.parse().ok()).unwrap_or(0);
                let softirq: u64 = parts.get(7).and_then(|v| v.parse().ok()).unwrap_or(0);
                let steal: u64 = parts.get(8).and_then(|v| v.parse().ok()).unwrap_or(0);

                let idle_time = idle + iowait;
                let total_time = user + nice + system + idle + iowait + irq + softirq + steal;

                let mut prev = PREV_CPU_STAT.write().unwrap();
                if let Some((prev_idle, prev_total)) = *prev {
                    if total_time > prev_total {
                        let diff_idle = idle_time.saturating_sub(prev_idle);
                        let diff_total = total_time.saturating_sub(prev_total);
                        if diff_total > 0 {
                            let usage = 100.0 * (1.0 - (diff_idle as f32 / diff_total as f32));
                            cpu_usage = Some(usage.clamp(0.0, 100.0));
                        }
                    }
                }
                *prev = Some((idle_time, total_time));
            }
        }
    }

    // Real CPU Model & Core Count from /proc/cpuinfo
    let mut cpu_model = None;
    let mut cpu_cores = 0;
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if cpu_model.is_none() && line.starts_with("model name") {
                if let Some((_, model)) = line.split_once(':') {
                    cpu_model = Some(model.trim().to_string());
                }
            }
            if line.starts_with("processor") {
                cpu_cores += 1;
            }
        }
    }
    if cpu_cores == 0 {
        cpu_cores = 1;
    }

    // 2. Real RAM Metrics from /proc/meminfo
    let mut total_ram = 0u64;
    let mut free_ram = 0u64;
    let mut available_ram = 0u64;
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            let mut parts = line.split_whitespace();
            if let Some(key) = parts.next() {
                if key == "MemTotal:" {
                    if let Some(val) = parts.next().and_then(|v| v.parse::<u64>().ok()) {
                        total_ram = val * 1024;
                    }
                } else if key == "MemFree:" {
                    if let Some(val) = parts.next().and_then(|v| v.parse::<u64>().ok()) {
                        free_ram = val * 1024;
                    }
                } else if key == "MemAvailable:" {
                    if let Some(val) = parts.next().and_then(|v| v.parse::<u64>().ok()) {
                        available_ram = val * 1024;
                    }
                }
            }
        }
    }
    let used_ram = total_ram.saturating_sub(available_ram);

    // 3. Real Process App RAM from /proc/self/status
    let mut app_ram = 0u64;
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(val) = parts[1].parse::<u64>() {
                        app_ram = val * 1024;
                    }
                }
                break;
            }
        }
    }

    // 4. Real Disk Metrics
    let models_dir = get_models_dir();
    let mut models_dir_size = 0u64;
    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                models_dir_size += meta.len();
            }
        }
    }

    let mut disk_total = 0u64;
    let mut disk_available = 0u64;
    let mut disk_used = 0u64;

    if let Ok(output) = std::process::Command::new("df")
        .args(["-B1", models_dir.to_str().unwrap_or(".")])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = stdout.lines().nth(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    disk_total = parts[1].parse::<u64>().unwrap_or(0);
                    disk_used = parts[2].parse::<u64>().unwrap_or(0);
                    disk_available = parts[3].parse::<u64>().unwrap_or(0);
                }
            }
        }
    }

    // 5. Real GPU detection
    let mut gpu_name = None;
    let mut gpu_utilization = None;
    let mut gpu_memory_total = None;
    let mut gpu_memory_used = None;
    let mut gpu_temp = None;
    let mut gpu_power = None;

    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,utilization.gpu,memory.total,memory.used,temperature.gpu,power.draw", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = stdout.lines().next() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 6 {
                    gpu_name = Some(parts[0].to_string());
                    gpu_utilization = parts[1].parse::<f32>().ok();
                    gpu_memory_total = parts[2].parse::<u64>().map(|m| m * 1024 * 1024).ok();
                    gpu_memory_used = parts[3].parse::<u64>().map(|m| m * 1024 * 1024).ok();
                    gpu_temp = parts[4].parse::<f32>().ok();
                    gpu_power = parts[5].parse::<f32>().ok();
                }
            }
        }
    }

    if gpu_name.is_none() {
        if std::path::Path::new("/sys/class/drm/card1").exists() || std::path::Path::new("/sys/class/drm/card0").exists() {
            gpu_name = Some("Intel UHD Graphics (Integrated)".to_string());
        }
    }

    SystemMetrics {
        cpu_usage,
        cpu_model,
        cpu_cores,
        process_cpu_usage: None,
        total_ram,
        used_ram,
        available_ram,
        free_ram,
        app_ram,
        disk_total,
        disk_available,
        disk_used,
        models_dir_size,
        gpu_name,
        gpu_utilization,
        gpu_memory_total,
        gpu_memory_used,
        gpu_temp,
        gpu_power,
        timestamp: now,
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::default().build())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .setup(|_app| {
            #[cfg(debug_assertions)]
            {
                use tauri::Manager;
                if let Some(window) = _app.get_webview_window("main") {
                    let _ = window.open_devtools();
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_models,
            get_models_detailed,
            delete_local_model,
            rename_local_model,
            import_external_model,
            load_model,
            chat_completions,
            check_server_status,
            get_loaded_model,
            unload_model,
            hf_search,
            hf_model_files,
            download_model,
            cancel_download,
            cancel_generation,
            get_model_metadata,
            get_system_metrics,
            get_machine_specs,
            save_app_state,
            load_app_state,
            run_model_benchmark
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}