use std::path::PathBuf;
use std::sync::Arc;
use clap::Parser;
use tokio::sync::RwLock;

use lm_studio::daemon::DaemonServer;
use lm_studio::discovery::HardwareTelemetry;
use lm_studio::engine::{InferenceEngine, RuntimeParams};
use lm_studio::server::{run_server, ServerState};
use lm_studio::ui::run_gui;

#[derive(Parser, Debug)]
#[command(name = "lm-studio", author = "Native LLM Workstation", version = "0.1.0", about = "Lightweight Native Local LLM Workstation")]
struct Args {
    #[arg(short, long, help = "Launch headless background daemon")]
    daemon: bool,

    #[arg(short, long, help = "Launch OpenAI & Anthropic compatible API server")]
    server: bool,

    #[arg(short, long, default_value = "1234", help = "HTTP API server port")]
    port: u16,

    #[arg(long, help = "Path to daemon Unix domain socket")]
    socket: Option<PathBuf>,

    #[arg(long, default_value = "300", help = "Idle TTL seconds before model auto-unloading")]
    idle_ttl: u64,

    #[arg(short, long, help = "Path to GGUF model to load at startup")]
    model: Option<PathBuf>,

    #[arg(long, help = "Launch native desktop GUI")]
    gui: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let telemetry = HardwareTelemetry::probe();

    println!("============================================================");
    println!("  ⚡ Native Local LLM Workstation (Rust)");
    println!("  Hardware: {:.1} GB RAM | Vulkan: {} | AVX2: {} | {} CPU Cores",
        telemetry.total_ram_bytes as f64 / 1e9,
        telemetry.has_vulkan,
        telemetry.has_avx2,
        telemetry.cpu_cores
    );
    println!("============================================================");

    if args.daemon {
        println!("[Main] Starting headless background daemon...");
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let daemon = DaemonServer::new(args.socket.as_deref(), args.idle_ttl);
            daemon.run().await.map_err(|e| Box::<dyn std::error::Error>::from(e))
        })?;
    } else if args.server {
        println!("[Main] Starting HTTP API server on port {}...", args.port);
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let engine_state = Arc::new(RwLock::new(None));

            if let Some(ref m_path) = args.model {
                println!("[Main] Pre-loading model: {}", m_path.display());
                match InferenceEngine::load(m_path, RuntimeParams::default()) {
                    Ok(eng) => {
                        let mut g = engine_state.write().await;
                        *g = Some(eng);
                    }
                    Err(e) => eprintln!("[Main] Warning: Could not load initial model: {}", e),
                }
            }

            let model_name = args.model
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                .unwrap_or_else(|| "local-model".to_string());

            let server_state = ServerState {
                engine: engine_state,
                default_model: Arc::new(RwLock::new(model_name)),
            };

            run_server(server_state, args.port).await.map_err(|e| Box::<dyn std::error::Error>::from(e))
        })?;
    } else if args.gui || std::env::var("DISPLAY").is_ok() || std::env::var("WAYLAND_DISPLAY").is_ok() {
        println!("[Main] Launching lightweight native GUI frontend...");
        if let Err(e) = run_gui() {
            eprintln!("[Main] GUI encountered error: {}", e);
        }
    } else {
        println!("[Main] Running in CLI mode. Use --help to view options.");
        println!("  To start daemon:     lm-studio --daemon");
        println!("  To start API server: lm-studio --server --port 1234");
        println!("  To control via CLI:  lms --help");
    }

    Ok(())
}
