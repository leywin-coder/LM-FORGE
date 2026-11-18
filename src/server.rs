use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::engine::InferenceEngine;
use crate::log_info;

// ============================================================================
// 1. OpenAI Compatible Schemas
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String, // "text", "json_object", or "json_schema"
    #[serde(default)]
    pub json_schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub max_tokens: Option<usize>,
    pub response_format: Option<ResponseFormat>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: serde_json::Value, // string or array of strings
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

// ============================================================================
// 2. Anthropic Messages Compatible Schemas
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value, // string or array of content blocks
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub system: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub usage: AnthropicUsage,
}

// ============================================================================
// 3. Server State & App Router
// ============================================================================

#[derive(Clone)]
pub struct ServerState {
    pub engine: Arc<RwLock<Option<InferenceEngine>>>,
    pub default_model: Arc<RwLock<String>>,
}

pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/completions", post(completions_handler))
        .route("/v1/embeddings", post(embeddings_handler))
        .route("/v1/messages", post(anthropic_messages_handler))
        .route("/v1/models", get(models_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}

pub async fn run_server(state: ServerState, port: u16) -> Result<(), String> {
    let app = create_router(state);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("Failed to bind TCP {}: {}", addr, e))?;

    log_info!("server", "API server listening on http://{}", addr);
    axum::serve(listener, app)
        .await
        .map_err(|e| format!("Server error: {}", e))?;
    Ok(())
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================================================
// 4. Handlers
// ============================================================================

async fn health_handler() -> &'static str {
    "OK"
}

async fn models_handler(State(state): State<ServerState>) -> Json<serde_json::Value> {
    let def_model = state.default_model.read().await.clone();
    let engine_guard = state.engine.read().await;
    let loaded_id = if let Some(ref e) = *engine_guard {
        e.model_path.file_name().unwrap_or_default().to_string_lossy().to_string()
    } else {
        def_model
    };

    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": loaded_id,
                "object": "model",
                "created": now_epoch_secs(),
                "owned_by": "local",
            }
        ]
    }))
}

async fn chat_completions_handler(
    State(state): State<ServerState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let engine_guard = state.engine.read().await;
    let Some(ref engine) = *engine_guard else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "message": "No model loaded. Please load a GGUF model first.",
                    "type": "server_error"
                }
            })),
        ).into_response();
    };

    let mut prompt = String::new();
    for msg in &req.messages {
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    prompt.push_str("<|im_start|>assistant\n");

    let mut runtime_params = engine.runtime_params.clone();
    if let Some(temp) = req.temperature {
        runtime_params.temperature = temp;
    }
    if let Some(top_p) = req.top_p {
        runtime_params.top_p = top_p;
    }
    if let Some(max_tok) = req.max_tokens {
        runtime_params.max_tokens = max_tok;
    }
    if let Some(stops) = req.stop {
        runtime_params.stop_tokens = stops;
    }

    // JSON Schema GBNF constraint extraction
    if let Some(rf) = req.response_format {
        if rf.format_type == "json_schema" || rf.format_type == "json_object" {
            runtime_params.json_schema = rf.json_schema.or_else(|| {
                Some(serde_json::json!({ "type": "object" }))
            });
        }
    }

    let model_name = req.model.unwrap_or_else(|| "local-model".to_string());
    let req_id = format!("chatcmpl-{}", now_epoch_secs());
    let stream_mode = req.stream.unwrap_or(false);

    let mut rx = engine.generate_stream(&prompt, &runtime_params, None).await;

    if stream_mode {
        struct StreamState {
            rx: tokio::sync::mpsc::Receiver<String>,
            req_id: String,
            model_name: String,
            stage: u8,
        }

        let initial_state = StreamState {
            rx,
            req_id,
            model_name,
            stage: 0,
        };

        let stream = futures_util::stream::unfold(initial_state, |mut s| async move {
            if s.stage == 0 {
                if let Some(token) = s.rx.recv().await {
                    let chunk = serde_json::json!({
                        "id": s.req_id,
                        "object": "chat.completion.chunk",
                        "created": now_epoch_secs(),
                        "model": s.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": { "content": token },
                                "finish_reason": null
                            }
                        ]
                    });
                    return Some((Ok::<Event, Infallible>(Event::default().data(chunk.to_string())), s));
                } else {
                    s.stage = 1;
                    let done_chunk = serde_json::json!({
                        "id": s.req_id,
                        "object": "chat.completion.chunk",
                        "created": now_epoch_secs(),
                        "model": s.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    });
                    return Some((Ok(Event::default().data(done_chunk.to_string())), s));
                }
            } else if s.stage == 1 {
                s.stage = 2;
                return Some((Ok(Event::default().data("[DONE]")), s));
            }
            None
        });

        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        let mut full_text = String::new();
        let mut token_count = 0;
        while let Some(tok) = rx.recv().await {
            full_text.push_str(&tok);
            token_count += 1;
        }

        let resp = ChatCompletionResponse {
            id: req_id,
            object: "chat.completion".to_string(),
            created: now_epoch_secs(),
            model: model_name,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: full_text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: prompt.split_whitespace().count(),
                completion_tokens: token_count,
                total_tokens: prompt.split_whitespace().count() + token_count,
            },
        };

        Json(resp).into_response()
    }
}

async fn completions_handler(
    State(state): State<ServerState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let engine_guard = state.engine.read().await;
    let Some(ref engine) = *engine_guard else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "No model loaded" })),
        ).into_response();
    };

    let mut runtime_params = engine.runtime_params.clone();
    if let Some(temp) = req.temperature {
        runtime_params.temperature = temp;
    }
    if let Some(max_tok) = req.max_tokens {
        runtime_params.max_tokens = max_tok;
    }

    let model_name = req.model.unwrap_or_else(|| "local-model".to_string());
    let req_id = format!("cmpl-{}", now_epoch_secs());
    let mut rx = engine.generate_stream(&req.prompt, &runtime_params, None).await;

    let mut full_text = String::new();
    let mut token_count = 0;
    while let Some(tok) = rx.recv().await {
        full_text.push_str(&tok);
        token_count += 1;
    }

    let resp = CompletionResponse {
        id: req_id,
        object: "text_completion".to_string(),
        created: now_epoch_secs(),
        model: model_name,
        choices: vec![CompletionChoice {
            text: full_text,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: req.prompt.split_whitespace().count(),
            completion_tokens: token_count,
            total_tokens: req.prompt.split_whitespace().count() + token_count,
        },
    };

    Json(resp).into_response()
}

async fn embeddings_handler(
    State(state): State<ServerState>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    let engine_guard = state.engine.read().await;
    let Some(ref engine) = *engine_guard else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "No model loaded" })),
        ).into_response();
    };

    let texts = match req.input {
        serde_json::Value::String(s) => vec![s],
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => vec![],
    };

    let mut data = Vec::new();
    let mut total_tokens = 0;

    for (index, text) in texts.iter().enumerate() {
        let embedding = engine.embed(text);
        total_tokens += text.split_whitespace().count();
        data.push(EmbeddingData {
            object: "embedding".to_string(),
            embedding,
            index,
        });
    }

    let resp = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: req.model.unwrap_or_else(|| "nomic-embed-text".to_string()),
        usage: Usage {
            prompt_tokens: total_tokens,
            completion_tokens: 0,
            total_tokens,
        },
    };

    Json(resp).into_response()
}

async fn anthropic_messages_handler(
    State(state): State<ServerState>,
    Json(req): Json<AnthropicRequest>,
) -> Response {
    let engine_guard = state.engine.read().await;
    let Some(ref engine) = *engine_guard else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "No model loaded" })),
        ).into_response();
    };

    let mut prompt = String::new();
    if let Some(ref sys) = req.system {
        prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
    }

    for msg in &req.messages {
        let text = match &msg.content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => {
                let mut joined = String::new();
                for item in arr {
                    if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                        joined.push_str(t);
                    }
                }
                joined
            }
            _ => String::new(),
        };
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, text));
    }
    prompt.push_str("<|im_start|>assistant\n");

    let mut runtime_params = engine.runtime_params.clone();
    if let Some(temp) = req.temperature {
        runtime_params.temperature = temp;
    }
    if let Some(top_p) = req.top_p {
        runtime_params.top_p = top_p;
    }
    if let Some(max_tok) = req.max_tokens {
        runtime_params.max_tokens = max_tok;
    }

    let req_id = format!("msg_{}", now_epoch_secs());
    let stream_mode = req.stream.unwrap_or(false);
    let mut rx = engine.generate_stream(&prompt, &runtime_params, None).await;

    if stream_mode {
        struct AnthropicStreamState {
            rx: tokio::sync::mpsc::Receiver<String>,
            req_id: String,
            model: String,
            stage: u8,
        }

        let initial_state = AnthropicStreamState {
            rx,
            req_id,
            model: req.model,
            stage: 0,
        };

        let stream = futures_util::stream::unfold(initial_state, |mut s| async move {
            match s.stage {
                0 => {
                    s.stage = 1;
                    let msg_start = serde_json::json!({
                        "type": "message_start",
                        "message": {
                            "id": s.req_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": s.model,
                            "stop_reason": null,
                            "usage": { "input_tokens": 10, "output_tokens": 1 }
                        }
                    });
                    Some((Ok::<Event, Infallible>(Event::default().event("message_start").data(msg_start.to_string())), s))
                }
                1 => {
                    s.stage = 2;
                    let block_start = serde_json::json!({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": { "type": "text", "text": "" }
                    });
                    Some((Ok(Event::default().event("content_block_start").data(block_start.to_string())), s))
                }
                2 => {
                    if let Some(tok) = s.rx.recv().await {
                        let delta = serde_json::json!({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": { "type": "text_delta", "text": tok }
                        });
                        Some((Ok(Event::default().event("content_block_delta").data(delta.to_string())), s))
                    } else {
                        s.stage = 3;
                        let block_stop = serde_json::json!({
                            "type": "content_block_stop",
                            "index": 0
                        });
                        Some((Ok(Event::default().event("content_block_stop").data(block_stop.to_string())), s))
                    }
                }
                3 => {
                    s.stage = 4;
                    let msg_stop = serde_json::json!({
                        "type": "message_stop"
                    });
                    Some((Ok(Event::default().event("message_stop").data(msg_stop.to_string())), s))
                }
                _ => None,
            }
        });

        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        let mut full_text = String::new();
        let mut token_count = 0;
        while let Some(tok) = rx.recv().await {
            full_text.push_str(&tok);
            token_count += 1;
        }

        let resp = AnthropicResponse {
            id: req_id,
            msg_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContentBlock {
                block_type: "text".to_string(),
                text: full_text,
            }],
            model: req.model,
            stop_reason: "end_turn".to_string(),
            usage: AnthropicUsage {
                input_tokens: prompt.split_whitespace().count(),
                output_tokens: token_count,
            },
        };

        Json(resp).into_response()
    }
}
