use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, RwLock};

use crate::discovery::{HardwareTelemetry, ResumableDownloader};
use crate::engine::{InferenceEngine, RuntimeParams};

pub const DEFAULT_SOCKET_PATH: &str = "/tmp/llm-daemon.sock";

// ============================================================================
// 1. JSON-RPC Protocol Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

impl JsonRpcResponse {
    pub fn success(id: u64, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(id: u64, code: i32, message: &str) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.to_string(),
            }),
        }
    }
}

// ============================================================================
// 2. Daemon State & Idle Memory Management
// ============================================================================

pub struct DaemonState {
    pub engine: RwLock<Option<InferenceEngine>>,
    pub last_activity: Mutex<Instant>,
    pub idle_ttl: Duration,
    pub server_running: RwLock<bool>,
    pub server_port: RwLock<u16>,
}

impl DaemonState {
    pub fn new(idle_ttl_secs: u64) -> Arc<Self> {
        Arc::new(Self {
            engine: RwLock::new(None),
            last_activity: Mutex::new(Instant::now()),
            idle_ttl: Duration::from_secs(idle_ttl_secs),
            server_running: RwLock::new(false),
            server_port: RwLock::new(1234),
        })
    }

    pub async fn touch(&self) {
        let mut la = self.last_activity.lock().await;
        *la = Instant::now();
    }
}

// ============================================================================
// 3. Headless Daemon Server
// ============================================================================

pub struct DaemonServer {
    socket_path: PathBuf,
    state: Arc<DaemonState>,
}

impl DaemonServer {
    pub fn new(socket_path: Option<&Path>, idle_ttl_secs: u64) -> Self {
        let path = socket_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from(DEFAULT_SOCKET_PATH));

        Self {
            socket_path: path,
            state: DaemonState::new(idle_ttl_secs),
        }
    }

    pub async fn run(&self) -> Result<(), String> {
        // Remove stale socket if present
        if self.socket_path.exists() {
            let _ = fs::remove_file(&self.socket_path);
        }

        let listener = UnixListener::bind(&self.socket_path)
            .map_err(|e| format!("Failed to bind UDS {}: {}", self.socket_path.display(), e))?;

        // Start idle memory reaper background task
        let state_clone = self.state.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                let idle_duration = {
                    let la = state_clone.last_activity.lock().await;
                    Instant::now().duration_since(*la)
                };

                if idle_duration >= state_clone.idle_ttl {
                    let mut engine_guard = state_clone.engine.write().await;
                    if engine_guard.is_some() {
                        *engine_guard = None; // Drop model, unmap memory
                        println!(
                            "[Daemon] Idle TTL reached ({}s). Model automatically unloaded to free RAM.",
                            idle_duration.as_secs()
                        );
                    }
                }
            }
        });

        println!("[Daemon] Listening on Unix socket: {}", self.socket_path.display());

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let state = self.state.clone();
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(stream, state).await {
                            eprintln!("[Daemon] Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    eprintln!("[Daemon] Accept error: {}", e);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }

    async fn handle_connection(stream: UnixStream, state: Arc<DaemonState>) -> Result<(), String> {
        let (reader, mut writer) = stream.into_split();
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();

        while buf_reader.read_line(&mut line).await.map_err(|e| e.to_string())? > 0 {
            let req: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    let res = JsonRpcResponse::err(0, -32700, &format!("Parse error: {}", e));
                    let mut res_str = serde_json::to_string(&res)
                        .map_err(|err| format!("Failed to serialize parse error response: {err}"))?;
                    res_str.push('\n');
                    writer.write_all(res_str.as_bytes()).await.map_err(|e| e.to_string())?;
                    line.clear();
                    continue;
                }
            };

            state.touch().await;
            let resp = Self::dispatch_method(&req, &state, &mut writer).await;
            if let Some(res) = resp {
                let mut res_str = serde_json::to_string(&res)
                    .map_err(|err| format!("Failed to serialize RPC response: {err}"))?;
                res_str.push('\n');
                writer.write_all(res_str.as_bytes()).await.map_err(|e| e.to_string())?;
            }

            line.clear();
        }

        Ok(())
    }

    async fn dispatch_method(
        req: &JsonRpcRequest,
        state: &Arc<DaemonState>,
        writer: &mut tokio::net::unix::OwnedWriteHalf,
    ) -> Option<JsonRpcResponse> {
        match req.method.as_str() {
            "load" => {
                let model_path_str = req.params.get("model_path").and_then(|v| v.as_str());
                let Some(model_path) = model_path_str else {
                    return Some(JsonRpcResponse::err(req.id, -32602, "Missing 'model_path' parameter"));
                };

                let params: RuntimeParams = req.params
                    .get("params")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();

                match InferenceEngine::load(Path::new(model_path), params) {
                    Ok(engine) => {
                        let size = engine.mmap_model.size;
                        let mut guard = state.engine.write().await;
                        *guard = Some(engine);
                        Some(JsonRpcResponse::success(
                            req.id,
                            serde_json::json!({
                                "status": "loaded",
                                "model_path": model_path,
                                "model_size_bytes": size,
                                "model_size_mb": size / (1024 * 1024),
                            }),
                        ))
                    }
                    Err(e) => Some(JsonRpcResponse::err(req.id, -32000, &e)),
                }
            }
            "unload" => {
                let mut guard = state.engine.write().await;
                let was_loaded = guard.is_some();
                *guard = None;
                Some(JsonRpcResponse::success(
                    req.id,
                    serde_json::json!({
                        "status": "unloaded",
                        "unloaded": was_loaded,
                    }),
                ))
            }
            "ps" => {
                let guard = state.engine.read().await;
                let telemetry = HardwareTelemetry::probe();
                let la = state.last_activity.lock().await;
                let idle_secs = Instant::now().duration_since(*la).as_secs();

                if let Some(ref engine) = *guard {
                    Some(JsonRpcResponse::success(
                        req.id,
                        serde_json::json!({
                            "status": "active",
                            "model_path": engine.model_path.to_string_lossy(),
                            "model_size_mb": engine.mmap_model.size / (1024 * 1024),
                            "runtime_params": engine.runtime_params,
                            "idle_seconds": idle_secs,
                            "idle_ttl_seconds": state.idle_ttl.as_secs(),
                            "available_ram_mb": telemetry.available_ram_bytes / (1024 * 1024),
                            "total_ram_mb": telemetry.total_ram_bytes / (1024 * 1024),
                            "vram_free_mb": telemetry.vram_free_bytes / (1024 * 1024),
                        }),
                    ))
                } else {
                    Some(JsonRpcResponse::success(
                        req.id,
                        serde_json::json!({
                            "status": "idle",
                            "model_path": null,
                            "idle_seconds": idle_secs,
                            "available_ram_mb": telemetry.available_ram_bytes / (1024 * 1024),
                            "total_ram_mb": telemetry.total_ram_bytes / (1024 * 1024),
                        }),
                    ))
                }
            }
            "chat" => {
                let prompt = req.params.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
                let stream_mode = req.params.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
                let json_schema = req.params.get("json_schema").cloned();
                let max_tokens = req.params.get("max_tokens").and_then(|v| v.as_u64()).map(|v| v as usize);
                let guard = state.engine.read().await;

                let Some(ref engine) = *guard else {
                    return Some(JsonRpcResponse::err(req.id, -32001, "No model loaded. Call 'load' first."));
                };

                let mut params = engine.runtime_params.clone();
                if let Some(schema) = json_schema {
                    // Only apply schema if it's an actual object, not null
                    if schema.is_object() {
                        params.json_schema = Some(schema);
                    }
                }
                if let Some(max_tokens) = max_tokens {
                    params.max_tokens = max_tokens.clamp(1, 4096);
                }
                let started = Instant::now();
                let mut rx = engine.generate_stream(prompt, &params, None).await;

                if stream_mode {
                    // Send streaming events directly over connection
                    while let Some(token) = rx.recv().await {
                        let chunk_event = serde_json::json!({
                            "jsonrpc": "2.0",
                            "method": "chat_chunk",
                            "params": {
                                "token": token,
                            }
                        });
                        let mut s = match serde_json::to_string(&chunk_event) {
                            Ok(value) => value,
                            Err(err) => {
                                eprintln!("[Daemon] Failed to serialize chat chunk: {err}");
                                break;
                            }
                        };
                        s.push('\n');
                        let _ = writer.write_all(s.as_bytes()).await;
                    }
                    Some(JsonRpcResponse::success(
                        req.id,
                        serde_json::json!({ "status": "stream_complete" }),
                    ))
                } else {
                    let mut full_text = String::new();
                    let mut generated_tokens = 0u64;
                    while let Some(token) = rx.recv().await {
                        full_text.push_str(&token);
                        generated_tokens += 1;
                    }
                    let elapsed_ms = started.elapsed().as_millis() as u64;
                    let tokens_per_second = generated_tokens as f64 / started.elapsed().as_secs_f64().max(0.001);
                    Some(JsonRpcResponse::success(
                        req.id,
                        serde_json::json!({
                            "response": full_text,
                            "metrics": {
                                "generated_tokens": generated_tokens,
                                "elapsed_ms": elapsed_ms,
                                "tokens_per_second": tokens_per_second,
                            }
                        }),
                    ))
                }
            }
            "server_start" => {
                let port = req.params.get("port").and_then(|v| v.as_u64()).unwrap_or(1234) as u16;
                let mut running = state.server_running.write().await;
                let mut p = state.server_port.write().await;
                *running = true;
                *p = port;
                Some(JsonRpcResponse::success(
                    req.id,
                    serde_json::json!({ "status": "started", "port": port }),
                ))
            }
            "server_stop" => {
                let mut running = state.server_running.write().await;
                *running = false;
                Some(JsonRpcResponse::success(
                    req.id,
                    serde_json::json!({ "status": "stopped" }),
                ))
            }
            "pull" => {
                let url = req.params.get("url").and_then(|v| v.as_str()).unwrap_or("");
                let dest = req.params.get("dest").and_then(|v| v.as_str()).unwrap_or("");
                let sha = req.params.get("sha256").and_then(|v| v.as_str());

                match ResumableDownloader::download_with_resume(
                    url,
                    Path::new(dest),
                    sha,
                    |_curr, _total| {},
                ).await {
                    Ok(_) => Some(JsonRpcResponse::success(
                        req.id,
                        serde_json::json!({ "status": "download_completed", "path": dest }),
                    )),
                    Err(e) => Some(JsonRpcResponse::err(req.id, -32002, &e)),
                }
            }
            "status" => {
                let telemetry = HardwareTelemetry::probe();
                Some(JsonRpcResponse::success(
                    req.id,
                    serde_json::to_value(&telemetry).unwrap_or_default(),
                ))
            }
            _ => Some(JsonRpcResponse::err(req.id, -32601, "Method not found")),
        }
    }
}

// ============================================================================
// 4. Daemon Client for CLI
// ============================================================================

pub struct DaemonClient {
    socket_path: PathBuf,
}

impl DaemonClient {
    pub fn new(socket_path: Option<&Path>) -> Self {
        Self {
            socket_path: socket_path
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from(DEFAULT_SOCKET_PATH)),
        }
    }

    pub async fn send_request(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, String> {
        let mut stream = UnixStream::connect(&self.socket_path)
            .await
            .map_err(|e| format!("Cannot connect to daemon at {}: {}", self.socket_path.display(), e))?;

        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 1,
            method: method.to_string(),
            params,
        };

        let mut req_str = serde_json::to_string(&req).unwrap();
        req_str.push('\n');
        stream.write_all(req_str.as_bytes()).await.map_err(|e| e.to_string())?;

        let (reader, _) = stream.into_split();
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();

        buf_reader.read_line(&mut line).await.map_err(|e| e.to_string())?;
        let resp: JsonRpcResponse = serde_json::from_str(&line).map_err(|e| e.to_string())?;

        if let Some(err) = resp.error {
            Err(format!("Daemon error {}: {}", err.code, err.message))
        } else {
            Ok(resp.result.unwrap_or_default())
        }
    }

    pub async fn stream_request(&self, method: &str, params: serde_json::Value) -> Result<(), String> {
        let mut stream = UnixStream::connect(&self.socket_path)
            .await
            .map_err(|e| format!("Cannot connect to daemon at {}: {}", self.socket_path.display(), e))?;

        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 1,
            method: method.to_string(),
            params,
        };

        let mut req_str = serde_json::to_string(&req).unwrap();
        req_str.push('\n');
        stream.write_all(req_str.as_bytes()).await.map_err(|e| e.to_string())?;

        let (reader, _) = stream.into_split();
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();

        // Read first line (should be stream_complete or first chunk)
        loop {
            line.clear();
            let bytes_read = buf_reader.read_line(&mut line).await.map_err(|e| e.to_string())?;
            if bytes_read == 0 {
                break;
            }

            // Try to parse as JSON-RPC response (could be stream_complete)
            if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(&line) {
                if let Some(err) = resp.error {
                    return Err(format!("Daemon error {}: {}", err.code, err.message));
                }
                if resp.result.is_some() && resp.result.as_ref().map(|v| v.get("status").and_then(|s| s.as_str()) == Some("stream_complete")).unwrap_or(false) {
                    break;
                }
            }

            // Try to parse as chat_chunk notification
            if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(token) = chunk.get("params").and_then(|p| p.get("token")).and_then(|t| t.as_str()) {
                    print!("{}", token);
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                }
            }
        }

        println!();
        Ok(())
    }
}
