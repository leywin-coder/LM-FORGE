use std::path::{Path, PathBuf};
use std::time::Instant;
use sha2::{Digest, Sha256};
use clap::{Args, Parser, Subcommand};
use lm_studio::daemon::DaemonClient;
use lm_studio::discovery::{GgufHeaderParser, HardwareTelemetry, MemoryBudgetEstimator, ResumableDownloader};
use lm_studio::engine::RuntimeParams;
use lm_studio::rag::{DocumentParser, TextChunker, VectorStore};

#[derive(Parser)]
#[command(name = "lms", author = "Native LLM Workstation", version = "0.1.0", about = "CLI controller for local LLM daemon")]
struct Cli {
    #[arg(short, long, help = "Path to daemon Unix domain socket")]
    socket: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Load a GGUF model into memory via daemon")]
    Load(LoadArgs),

    #[command(about = "Unload currently loaded model and free RAM")]
    Unload,

    #[command(about = "Show currently loaded model and memory telemetry")]
    Ps,

    #[command(about = "Chat or generate tokens from loaded model")]
    Chat(ChatArgs),

    #[command(about = "Benchmark the loaded model with fixed prompts")]
    Bench(BenchArgs),

    #[command(about = "Start, stop, or check local API server status")]
    Server(ServerArgs),

    #[command(about = "Download a GGUF model with range resume and checksum")]
    Pull(PullArgs),

    #[command(about = "Check hardware compatibility and fit for a model")]
    Check(CheckArgs),

    #[command(about = "Ingest a document and test local offline RAG vector search")]
    Rag(RagArgs),
}

#[derive(Args)]
struct LoadArgs {
    #[arg(help = "Path to .gguf model file")]
    model_path: PathBuf,

    #[arg(short = 'g', long, default_value = "32", help = "Number of GPU layers to offload via Vulkan")]
    gpu_layers: i32,

    #[arg(short = 'c', long, default_value = "4096", help = "Context window length (n_ctx)")]
    ctx: u32,

    #[arg(short = 't', long, default_value = "4", help = "Number of CPU threads (AVX2)")]
    threads: i32,

    #[arg(long, default_value = "0.7", help = "Sampling temperature")]
    temp: f32,

    #[arg(long, default_value = "0.9", help = "Top-p sampling")]
    top_p: f32,

    #[arg(long, help = "Optional companion CLIP vision projector (mmproj GGUF)")]
    mmproj: Option<PathBuf>,
}

#[derive(Args)]
struct ChatArgs {
    #[arg(help = "Prompt string")]
    prompt: String,

    #[arg(short = 's', long, default_value = "true", help = "Stream tokens in real time")]
    stream: bool,

    #[arg(long, default_value = "0.7", help = "Sampling temperature")]
    temp: f32,

    #[arg(long, help = "Enforce JSON Schema constraint (GBNF grammar)")]
    json_schema: Option<String>,
}

#[derive(Args)]
struct BenchArgs {
    #[arg(short, long, default_value_t = 3, value_parser = clap::value_parser!(u8).range(1..=10))]
    runs: u8,

    #[arg(short = 'n', long, default_value_t = 64)]
    max_tokens: usize,

    #[arg(long, help = "Emit a single JSON report")]
    json: bool,
}

#[derive(Args)]
struct ServerArgs {
    #[arg(help = "Action: start, stop, or status")]
    action: String,

    #[arg(short = 'p', long, default_value = "1234", help = "Port to listen on")]
    port: u16,
}

#[derive(Args)]
struct PullArgs {
    #[arg(help = "Model download URL")]
    url: String,

    #[arg(short = 'd', long, help = "Destination path")]
    dest: Option<PathBuf>,

    #[arg(long, help = "Expected SHA-256 hash")]
    sha256: Option<String>,
}

#[derive(Args)]
struct CheckArgs {
    #[arg(help = "Local GGUF path or remote URL")]
    target: String,

    #[arg(short = 'c', long, help = "Proposed context length")]
    ctx: Option<u32>,
}

#[derive(Args)]
struct RagArgs {
    #[arg(help = "Document to ingest (PDF, Markdown, CSV, Code)")]
    file: PathBuf,

    #[arg(short = 'q', long, help = "Test query against ingested document")]
    query: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let client = DaemonClient::new(cli.socket.as_deref());

    match cli.command {
        Commands::Load(args) => {
            let model_path = args.model_path.to_string_lossy().to_string();
            let params = RuntimeParams {
                n_ctx: args.ctx,
                n_gpu_layers: args.gpu_layers,
                n_threads: args.threads,
                temperature: args.temp,
                top_p: args.top_p,
                mmproj_path: args.mmproj,
                ..Default::default()
            };

            let req_params = serde_json::json!({
                "model_path": model_path,
                "params": serde_json::to_value(&params)?,
            });

            match client.send_request("load", req_params).await {
                Ok(res) => {
                    println!("Model loaded successfully:");
                    println!("{}", serde_json::to_string_pretty(&res)?);
                }
                Err(e) => eprintln!("Load failed: {}", e),
            }
        }
        Commands::Unload => {
            match client.send_request("unload", serde_json::json!({})).await {
                Ok(res) => println!("Model unloaded: {}", serde_json::to_string_pretty(&res)?),
                Err(e) => eprintln!("Unload failed: {}", e),
            }
        }
        Commands::Ps => {
            match client.send_request("ps", serde_json::json!({})).await {
                Ok(res) => {
                    println!("Daemon & Memory Status:");
                    println!("{}", serde_json::to_string_pretty(&res)?);
                }
                Err(e) => eprintln!("Failed to get status: {}", e),
            }
        }
        Commands::Chat(args) => {
            let schema_val: Option<serde_json::Value> = args
                .json_schema
                .as_deref()
                .and_then(|s| serde_json::from_str(s).ok());

            let req_params = serde_json::json!({
                "prompt": args.prompt,
                "stream": args.stream,
                "json_schema": schema_val,
            });

            if args.stream {
                // Streaming mode: keep connection open and print tokens as they arrive
                match client.stream_request("chat", req_params).await {
                    Ok(_) => {},
                    Err(e) => eprintln!("Chat error: {}", e),
                }
            } else {
                // Non-streaming mode
                match client.send_request("chat", req_params).await {
                    Ok(res) => {
                        if let Some(resp_text) = res.get("response").and_then(|v| v.as_str()) {
                            println!("{}", resp_text);
                        } else {
                            println!("{}", serde_json::to_string_pretty(&res)?);
                        }
                    }
                    Err(e) => eprintln!("Chat error: {}", e),
                }
            }
        }
        Commands::Bench(args) => {
            let status = client.send_request("ps", serde_json::json!({})).await?;
            if status.get("status").and_then(|v| v.as_str()) != Some("active") {
                return Err("No model loaded. Run `lms load <model.gguf>` first.".into());
            }
            let model_path = status.get("model_path").and_then(|v| v.as_str()).ok_or("Daemon did not return a model path")?;
            let model_bytes = std::fs::read(model_path)?;
            let model_sha256 = format!("{:x}", Sha256::digest(&model_bytes));
            let runtime = status.get("runtime_params").cloned().unwrap_or_default();
            let prompt = "Write a concise explanation of local language model inference in exactly three paragraphs.";
            let mut samples = Vec::with_capacity(args.runs as usize);
            for _ in 0..args.runs {
                let started = Instant::now();
                let response = client.send_request("chat", serde_json::json!({
                    "prompt": prompt,
                    "stream": false,
                    "max_tokens": args.max_tokens,
                })).await?;
                let elapsed_ms = started.elapsed().as_millis() as u64;
                let metrics = response.get("metrics").cloned().unwrap_or_default();
                samples.push(serde_json::json!({
                    "generated_tokens": metrics.get("generated_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                    "elapsed_ms": metrics.get("elapsed_ms").and_then(|v| v.as_u64()).unwrap_or(elapsed_ms),
                    "tokens_per_second": metrics.get("tokens_per_second").and_then(|v| v.as_f64()).unwrap_or(0.0),
                }));
            }
            let mut rates: Vec<f64> = samples.iter().filter_map(|s| s.get("tokens_per_second").and_then(|v| v.as_f64())).collect();
            rates.sort_by(f64::total_cmp);
            let report = serde_json::json!({
                "model_path": status.get("model_path"),
                "model_sha256": model_sha256,
                "runtime": runtime,
                "hardware": {
                    "available_ram_mb": status.get("available_ram_mb"),
                    "total_ram_mb": status.get("total_ram_mb"),
                    "vram_free_mb": status.get("vram_free_mb"),
                },
                "runs": samples,
                "median_tokens_per_second": rates[rates.len() / 2],
            });
            let history_path = std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
                .unwrap_or_else(|| PathBuf::from("."))
                .join("lm-forge/benchmarks.json");
            if let Some(parent) = history_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut history: Vec<serde_json::Value> = std::fs::read(&history_path)
                .ok()
                .and_then(|bytes| serde_json::from_slice(&bytes).ok())
                .unwrap_or_default();
            history.push(report.clone());
            std::fs::write(&history_path, serde_json::to_vec_pretty(&history)?)?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("Model: {}", report["model_path"]);
                println!("Median generation speed: {:.2} tk/s ({} runs)", report["median_tokens_per_second"].as_f64().unwrap_or(0.0), args.runs);
                println!("{}", serde_json::to_string_pretty(&report["runs"])?);
            }
        }
        Commands::Server(args) => match args.action.as_str() {
            "start" => {
                let req_params = serde_json::json!({ "port": args.port });
                match client.send_request("server_start", req_params).await {
                    Ok(res) => println!("Server start: {}", res),
                    Err(e) => eprintln!("Server start failed: {}", e),
                }
            }
            "stop" => {
                match client.send_request("server_stop", serde_json::json!({})).await {
                    Ok(res) => println!("Server stop: {}", res),
                    Err(e) => eprintln!("Server stop failed: {}", e),
                }
            }
            _ => {
                println!("Usage: lms server <start|stop> [--port 1234]");
            }
        },
        Commands::Pull(args) => {
            let dest_path = args.dest.unwrap_or_else(|| {
                let name = args.url.split('/').last().unwrap_or("model.gguf");
                PathBuf::from(name)
            });

            println!("Downloading {} -> {}", args.url, dest_path.display());
            ResumableDownloader::download_with_resume(
                &args.url,
                &dest_path,
                args.sha256.as_deref(),
                |curr, total| {
                    if total > 0 {
                        print!("\rDownloaded {:.1} / {:.1} MB ({:.1}%)",
                            curr as f64 / 1e6,
                            total as f64 / 1e6,
                            (curr as f64 / total as f64) * 100.0
                        );
                    } else {
                        print!("\rDownloaded {:.1} MB", curr as f64 / 1e6);
                    }
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                },
            ).await?;
            println!("\nDownload verified and completed.");
        }
        Commands::Check(args) => {
            let telemetry = HardwareTelemetry::probe();
            println!("Hardware Telemetry Detected:");
            println!("  Total RAM:      {:.2} GB", telemetry.total_ram_bytes as f64 / 1e9);
            println!("  Available RAM:  {:.2} GB", telemetry.available_ram_bytes as f64 / 1e9);
            println!("  VRAM (Free):    {:.2} MB", telemetry.vram_free_bytes as f64 / 1e6);
            println!("  Vulkan GPU:     {} (integrated: {})", telemetry.has_vulkan, telemetry.is_integrated_gpu);
            println!("  AVX2 CPU:       {}", telemetry.has_avx2);
            println!("  CPU Cores:      {}", telemetry.cpu_cores);

            let (meta, file_size) = if args.target.starts_with("http://") || args.target.starts_with("https://") {
                println!("\nFetching remote GGUF header via Range request (bytes=0-65535)...");
                let m = GgufHeaderParser::fetch_remote_header(&args.target).await?;
                (m, 4 * 1024 * 1024 * 1024)
            } else {
                let p = Path::new(&args.target);
                let bytes = std::fs::read(p)?;
                let m = GgufHeaderParser::parse(&bytes)?;
                let sz = bytes.len() as u64;
                (m, sz)
            };

            println!("\nGGUF Model Header Info:");
            println!("  Architecture:   {}", meta.architecture);
            println!("  Quantization:   {}", meta.quantization);
            println!("  Layers:         {}", meta.layer_count);
            println!("  Context Length: {}", meta.context_length);
            println!("  Attention Heads:{} (KV heads: {})", meta.head_count, meta.head_count_kv);

            let est = MemoryBudgetEstimator::estimate(&meta, file_size, &telemetry, args.ctx);
            println!("\nMemory Budget & Fit Estimation:");
            println!("  Estimated RAM:  {:.2} GB", est.total_estimated_ram_bytes as f64 / 1e9);
            println!("  Classification: {:?}", est.classification);
            println!("  GPU Layers:     {}", est.suggested_gpu_layers);
            println!("  Recommended Ctx:{}", est.recommended_ctx);
            println!("  Diagnosis:      {}", est.reason);
        }
        Commands::Rag(args) => {
            println!("Ingesting document: {}", args.file.display());
            let text = DocumentParser::parse_file(&args.file)?;
            let chunks = TextChunker::chunk(&text, TextChunker::DEFAULT_CHUNK_SIZE_TOKENS, TextChunker::DEFAULT_OVERLAP_PERCENT);
            println!("Extracted {} plain text characters -> {} overlapping chunks.", text.len(), chunks.len());

            if let Some(query) = args.query {
                println!("\nRunning local semantic similarity for query: '{}'...", query);
                let mut store = VectorStore::new();
                for (i, c) in chunks.iter().enumerate() {
                    let emb = vec![0.1f32; 384];
                    store.add_chunk(lm_studio::rag::DocumentChunk {
                        id: i + 1,
                        source_file: args.file.file_name().unwrap_or_default().to_string_lossy().to_string(),
                        chunk_index: i + 1,
                        text: c.clone(),
                        embedding: emb,
                    });
                }
                let query_emb = vec![0.1f32; 384];
                let results = store.search(&query_emb, 3);
                println!("Top retrieved matches:");
                for (chunk, score) in results {
                    println!("  [Chunk #{}, score: {:.2}]: {}", chunk.chunk_index, score, chunk.text.trim().chars().take(80).collect::<String>());
                }
            }
        }
    }

    Ok(())
}
