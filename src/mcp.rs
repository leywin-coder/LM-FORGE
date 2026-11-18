use std::process::Stdio;
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::engine::{InferenceEngine, RuntimeParams};

// ============================================================================
// 1. Jinja2 Chat Template Renderer
// ============================================================================

pub struct ChatTemplateRenderer {
    env: Environment<'static>,
    template_source: String,
}

impl ChatTemplateRenderer {
    pub const DEFAULT_CHATML_TEMPLATE: &str = r#"
{%- for message in messages %}
<|im_start|>{{ message.role }}
{%- if message.role == 'system' and tools %}
# Tools Available:
{{ tools }}
{%- endif %}
{{ message.content }}<|im_end|>
{%- endfor %}
<|im_start|>assistant
"#;

    pub fn new(custom_template: Option<&str>) -> Self {
        let env = Environment::new();
        let tmpl = custom_template.unwrap_or(Self::DEFAULT_CHATML_TEMPLATE).to_string();
        // Register default filters if needed
        Self {
            env,
            template_source: tmpl,
        }
    }

    pub fn render_with_tools(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[McpTool]>,
    ) -> Result<String, String> {
        let tools_json = tools.map(|t| serde_json::to_string_pretty(t).unwrap_or_default());

        let mut env = self.env.clone();
        env.add_template("chat_template", &self.template_source)
            .map_err(|e| format!("Failed to compile Jinja template: {}", e))?;

        let tmpl = env
            .get_template("chat_template")
            .map_err(|e| format!("Template error: {}", e))?;

        tmpl.render(context! {
            messages => messages,
            tools => tools_json,
        })
        .map_err(|e| format!("Render error: {}", e))
    }
}

// ============================================================================
// 2. MCP Data Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCallRequest {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCallResult {
    pub content: Vec<McpContentBlock>,
    #[serde(default)]
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpContentBlock {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

// ============================================================================
// 3. MCP Client (Stdio & HTTP Transports)
// ============================================================================

pub enum McpTransport {
    Stdio {
        _child: Child,
        stdin: ChildStdin,
        reader: BufReader<ChildStdout>,
    },
    Http {
        endpoint_url: String,
        client: reqwest::Client,
    },
}

pub struct McpClient {
    transport: Mutex<McpTransport>,
    next_id: Mutex<u64>,
}

impl McpClient {
    pub async fn spawn_stdio(command: &str, args: &[&str]) -> Result<Self, String> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to spawn MCP process {}: {}", command, e))?;

        let stdin = child.stdin.take().ok_or("Failed to open child stdin")?;
        let stdout = child.stdout.take().ok_or("Failed to open child stdout")?;
        let reader = BufReader::new(stdout);

        let client = Self {
            transport: Mutex::new(McpTransport::Stdio {
                _child: child,
                stdin,
                reader,
            }),
            next_id: Mutex::new(1),
        };

        client.initialize().await?;
        Ok(client)
    }

    pub fn new_http(endpoint_url: &str) -> Self {
        Self {
            transport: Mutex::new(McpTransport::Http {
                endpoint_url: endpoint_url.to_string(),
                client: reqwest::Client::new(),
            }),
            next_id: Mutex::new(1),
        }
    }

    async fn initialize(&self) -> Result<(), String> {
        let init_params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "lm-studio-rust",
                "version": "0.1.0"
            }
        });

        let _ = self.call_rpc("initialize", init_params).await?;
        Ok(())
    }

    pub async fn list_tools(&self) -> Result<Vec<McpTool>, String> {
        let res = self.call_rpc("tools/list", serde_json::json!({})).await?;
        let tools_val = res.get("tools").ok_or("No 'tools' field in response")?;
        serde_json::from_value(tools_val.clone()).map_err(|e| e.to_string())
    }

    pub async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<McpToolCallResult, String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });

        let res = self.call_rpc("tools/call", params).await?;
        serde_json::from_value(res).map_err(|e| e.to_string())
    }

    async fn call_rpc(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, String> {
        let id = {
            let mut id_lock = self.next_id.lock().await;
            let current = *id_lock;
            *id_lock += 1;
            current
        };

        let req_json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut transport = self.transport.lock().await;
        match &mut *transport {
            McpTransport::Stdio { stdin, reader, .. } => {
                let mut line = serde_json::to_string(&req_json).map_err(|e| e.to_string())?;
                line.push('\n');
                stdin.write_all(line.as_bytes()).await.map_err(|e| e.to_string())?;
                stdin.flush().await.map_err(|e| e.to_string())?;

                let mut resp_line = String::new();
                reader.read_line(&mut resp_line).await.map_err(|e| e.to_string())?;

                let resp: serde_json::Value = serde_json::from_str(&resp_line).map_err(|e| e.to_string())?;
                if let Some(err) = resp.get("error") {
                    Err(format!("MCP Error: {}", err))
                } else {
                    Ok(resp.get("result").cloned().unwrap_or(serde_json::Value::Null))
                }
            }
            McpTransport::Http { endpoint_url, client } => {
                let res = client
                    .post(endpoint_url.as_str())
                    .json(&req_json)
                    .send()
                    .await
                    .map_err(|e| e.to_string())?;

                let resp: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
                if let Some(err) = resp.get("error") {
                    Err(format!("MCP Error: {}", err))
                } else {
                    Ok(resp.get("result").cloned().unwrap_or(serde_json::Value::Null))
                }
            }
        }
    }
}

// ============================================================================
// 4. Native Tool Calling Execution Loop
// ============================================================================

pub struct ToolCallingLoop;

impl ToolCallingLoop {
    pub async fn run_agent_turn(
        engine: &InferenceEngine,
        mcp_client: Option<&McpClient>,
        chat_template: &ChatTemplateRenderer,
        conversation: &mut Vec<serde_json::Value>,
        params: &RuntimeParams,
    ) -> Result<String, String> {
        let tools = if let Some(client) = mcp_client {
            client.list_tools().await.ok()
        } else {
            None
        };

        let mut final_response = String::new();

        for _iteration in 0..5 {
            // Render prompt with Jinja2 template and tool definitions
            let prompt = chat_template.render_with_tools(conversation, tools.as_deref())?;

            let mut rx = engine.generate_stream(&prompt, params, None).await;
            let mut generated = String::new();

            while let Some(tok) = rx.recv().await {
                generated.push_str(&tok);
            }

            // Check if model emitted a tool call sequence
            // Formats: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
            if let Some((tool_name, tool_args)) = Self::extract_tool_call(&generated) {
                // Pause streaming and execute tool call
                conversation.push(serde_json::json!({
                    "role": "assistant",
                    "content": generated,
                }));

                let tool_output_text = if let Some(client) = mcp_client {
                    match client.call_tool(&tool_name, tool_args).await {
                        Ok(res) => {
                            res.content.iter().map(|b| b.text.as_str()).collect::<Vec<_>>().join("\n")
                        }
                        Err(e) => format!("Error executing tool: {}", e),
                    }
                } else {
                    format!("Tool {} called with mock response.", tool_name)
                };

                conversation.push(serde_json::json!({
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_output_text,
                }));

                // Loop continues to generate final answer using tool output
                continue;
            } else {
                final_response = generated;
                conversation.push(serde_json::json!({
                    "role": "assistant",
                    "content": final_response.clone(),
                }));
                break;
            }
        }

        Ok(final_response)
    }

    fn extract_tool_call(text: &str) -> Option<(String, serde_json::Value)> {
        if let Some(start) = text.find("<tool_call>") {
            if let Some(end) = text[start..].find("</tool_call>") {
                let json_slice = &text[start + 11..start + end].trim();
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_slice) {
                    let name = val.get("name").and_then(|v| v.as_str())?.to_string();
                    let args = val.get("arguments").cloned().unwrap_or(serde_json::Value::Null);
                    return Some((name, args));
                }
            }
        }

        // Try raw JSON object detection with name + arguments
        if let Some(start) = text.find('{') {
            if let Some(end) = text.rfind('}') {
                if end > start {
                    let slice = &text[start..=end];
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(slice) {
                        if let (Some(name), Some(args)) = (val.get("name").and_then(|v| v.as_str()), val.get("arguments")) {
                            return Some((name.to_string(), args.clone()));
                        }
                    }
                }
            }
        }

        None
    }
}
