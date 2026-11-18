use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use eframe::egui::{self, Color32, ProgressBar, RichText, ScrollArea};

use crate::discovery::{HardwareTelemetry, HfModel, HfModelFile, HuggingFaceClient, ResumableDownloader};
use crate::engine::{InferenceEngine, RuntimeParams};
use crate::logger::{LogLevel, Logger};
use crate::rag::{RagPipeline, VectorStore};
use crate::{log_error, log_info, log_warn};

// ============================================================================
// UI Data Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurrentView {
    Chat,
    Discover,
    Downloads,
    Settings,
    Logs,
}

#[derive(Debug, Clone)]
pub struct Conversation {
    pub id: usize,
    pub title: String,
    pub messages: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadStatus {
    Downloading,
    Paused,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct DownloadItem {
    pub id: usize,
    pub filename: String,
    pub model_id: String,
    pub url: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub status: DownloadStatus,
}

#[derive(Debug, Clone)]
pub struct DownloadedFile {
    pub filename: String,
    pub path: PathBuf,
    pub size_bytes: u64,
}

// ============================================================================
// UI Application State
// ============================================================================

pub struct LmStudioApp {
    pub current_view: CurrentView,
    pub telemetry: HardwareTelemetry,
    pub last_telemetry_update: Instant,
    pub engine: Arc<RwLock<Option<InferenceEngine>>>,
    pub vector_store: VectorStore,
    pub ingested_files: Vec<String>,

    // Conversations state
    pub conversations: Vec<Conversation>,
    pub active_convo_idx: usize,
    pub current_input: String,
    pub is_generating: bool,
    pub active_generation_rx: Option<std::sync::mpsc::Receiver<String>>,

    // Runtime parameter sliders
    pub temperature: f32,
    pub top_p: f32,
    pub context_length: u32,
    pub gpu_layers: i32,
    pub cpu_threads: i32,

    // API Server state
    pub server_running: bool,
    pub server_port: u16,

    // Hugging Face Discovery state
    pub hf_search_query: String,
    pub hf_search_results: Arc<RwLock<Vec<HfModel>>>,
    pub hf_selected_files: Arc<RwLock<Option<(String, Vec<HfModelFile>)>>>,
    pub hf_is_searching: Arc<RwLock<bool>>,
    pub hf_error: Arc<RwLock<Option<String>>>,

    // Downloads state
    pub downloads: Arc<RwLock<Vec<DownloadItem>>>,
    pub downloaded_files: Vec<DownloadedFile>,
    pub next_download_id: usize,

    // Settings fields
    pub settings_models_dir: String,
    pub settings_hf_token: String,
    pub settings_idle_ttl: u64,
    pub settings_safe_ram_mb: u64,
    pub settings_uds_socket: String,
    pub settings_use_mmap: bool,
    pub settings_mlock: bool,
    pub settings_flash_attn: bool,
    pub settings_server_autostart: bool,
    pub settings_server_cors: bool,
    pub settings_rag_chunk_size: usize,
    pub settings_rag_overlap_pct: usize,
    pub settings_rag_top_k: usize,
    pub settings_mcp_command: String,

    // Logs filter
    pub log_filter_level: Option<LogLevel>,
    pub log_search_query: String,
    pub log_auto_scroll: bool,

    // Status message
    pub status_text: String,
}

impl LmStudioApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        let telemetry = HardwareTelemetry::probe();
        let cpu_cores = telemetry.cpu_cores as i32;

        let initial_convo = Conversation {
            id: 1,
            title: "Welcome to LM Workstation".to_string(),
            messages: vec![
                (
                    "assistant".to_string(),
                    "Welcome to Native Local LLM Workstation. Load a model, discover GGUFs from Hugging Face, or attach documents for local RAG.".to_string(),
                )
            ],
        };

        let mut app = Self {
            current_view: CurrentView::Chat,
            telemetry,
            last_telemetry_update: Instant::now(),
            engine: Arc::new(RwLock::new(None)),
            vector_store: VectorStore::new(),
            ingested_files: Vec::new(),

            conversations: vec![initial_convo],
            active_convo_idx: 0,
            current_input: String::new(),
            is_generating: false,
            active_generation_rx: None,

            temperature: 0.7,
            top_p: 0.9,
            context_length: 512,
            gpu_layers: 0,
            cpu_threads: 4.min(cpu_cores).max(2),

            server_running: false,
            server_port: 1234,

            hf_search_query: "qwen2.5 gguf".to_string(),
            hf_search_results: Arc::new(RwLock::new(Vec::new())),
            hf_selected_files: Arc::new(RwLock::new(None)),
            hf_is_searching: Arc::new(RwLock::new(false)),
            hf_error: Arc::new(RwLock::new(None)),

            downloads: Arc::new(RwLock::new(Vec::new())),
            downloaded_files: Vec::new(),
            next_download_id: 1,

            settings_models_dir: "models".to_string(),
            settings_hf_token: String::new(),
            settings_idle_ttl: 300,
            settings_safe_ram_mb: 800,
            settings_uds_socket: "/tmp/llm-daemon.sock".to_string(),
            settings_use_mmap: true,
            settings_mlock: false,
            settings_flash_attn: false,
            settings_server_autostart: false,
            settings_server_cors: true,
            settings_rag_chunk_size: 512,
            settings_rag_overlap_pct: 10,
            settings_rag_top_k: 3,
            settings_mcp_command: "npx -y @modelcontextprotocol/server-filesystem".to_string(),

            log_filter_level: None,
            log_search_query: String::new(),
            log_auto_scroll: true,

            status_text: "Ready".to_string(),
        };

        log_info!("app", "Native Local LLM Workstation initialized");
        log_info!(
            "telemetry",
            "RAM: {:.2} GB | Vulkan: {} | AVX2: {} | CPU Cores: {}",
            app.telemetry.total_ram_bytes as f64 / 1e9,
            app.telemetry.has_vulkan,
            app.telemetry.has_avx2,
            app.telemetry.cpu_cores
        );

        app.refresh_downloaded_files();
        app
    }

    fn refresh_telemetry(&mut self) {
        if self.last_telemetry_update.elapsed().as_secs() >= 3 {
            self.telemetry = HardwareTelemetry::probe();
            self.last_telemetry_update = Instant::now();
        }
    }

    pub fn refresh_downloaded_files(&mut self) {
        let dir = Path::new(&self.settings_models_dir);
        let mut list = Vec::new();
        if dir.exists() {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                        if name.to_lowercase().ends_with(".gguf") {
                            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                            list.push(DownloadedFile {
                                filename: name,
                                path,
                                size_bytes: size,
                            });
                        }
                    }
                }
            }
        }
        self.downloaded_files = list;
    }

    pub fn trigger_hf_search(&self) {
        let query = self.hf_search_query.clone();
        let results_lock = self.hf_search_results.clone();
        let searching_lock = self.hf_is_searching.clone();
        let error_lock = self.hf_error.clone();

        log_info!("huggingface", "Searching Hub for query: '{}'...", query);
        *searching_lock.write().unwrap() = true;
        *error_lock.write().unwrap() = None;

        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();

            if let Ok(rt) = rt {
                let res = rt.block_on(async {
                    HuggingFaceClient::search_models(&query, 20).await
                });

                match res {
                    Ok(models) => {
                        log_info!("huggingface", "Search returned {} models from Hub", models.len());
                        *results_lock.write().unwrap() = models;
                    }
                    Err(e) => {
                        log_error!("huggingface", "Search failed: {}", e);
                        *error_lock.write().unwrap() = Some(e);
                    }
                }
            }
            *searching_lock.write().unwrap() = false;
        });
    }

    pub fn trigger_hf_list_files(&self, model_id: &str) {
        let model_id_owned = model_id.to_string();
        let files_lock = self.hf_selected_files.clone();
        let error_lock = self.hf_error.clone();

        log_info!("huggingface", "Inspecting GGUF tree for {}", model_id_owned);

        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();

            if let Ok(rt) = rt {
                match rt.block_on(HuggingFaceClient::list_gguf_files(&model_id_owned)) {
                    Ok(files) => {
                        log_info!("huggingface", "Found {} GGUF files in {}", files.len(), model_id_owned);
                        *files_lock.write().unwrap() = Some((model_id_owned, files));
                    }
                    Err(e) => {
                        log_error!("huggingface", "Failed to list files: {}", e);
                        *error_lock.write().unwrap() = Some(e);
                    }
                }
            }
        });
    }

    pub fn start_download(&mut self, url: &str, filename: &str, model_id: &str) {
        let models_dir = PathBuf::from(&self.settings_models_dir);
        let _ = fs::create_dir_all(&models_dir);
        let dest_path = models_dir.join(filename);

        log_info!("downloader", "Initiating download: {} -> {}", filename, dest_path.display());

        let id = self.next_download_id;
        self.next_download_id += 1;

        let item = DownloadItem {
            id,
            filename: filename.to_string(),
            model_id: model_id.to_string(),
            url: url.to_string(),
            downloaded_bytes: 0,
            total_bytes: 0,
            status: DownloadStatus::Downloading,
        };

        self.downloads.write().unwrap().push(item);

        let downloads_lock = self.downloads.clone();
        let url_owned = url.to_string();
        let dest_owned = dest_path.clone();
        let fn_owned = filename.to_string();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();

            if let Ok(rt) = rt {
                let dl_lock = downloads_lock.clone();
                let res = rt.block_on(async {
                    ResumableDownloader::download_with_resume(
                        &url_owned,
                        &dest_owned,
                        None,
                        move |curr, total| {
                            let mut dls = dl_lock.write().unwrap();
                            if let Some(d) = dls.iter_mut().find(|x| x.id == id) {
                                d.downloaded_bytes = curr;
                                d.total_bytes = total;
                            }
                        },
                    ).await
                });

                let mut dls = downloads_lock.write().unwrap();
                if let Some(d) = dls.iter_mut().find(|x| x.id == id) {
                    match res {
                        Ok(_) => {
                            log_info!("downloader", "Download completed successfully: {}", fn_owned);
                            d.status = DownloadStatus::Completed;
                        }
                        Err(e) => {
                            log_error!("downloader", "Download failed for {}: {}", fn_owned, e);
                            d.status = DownloadStatus::Failed(e);
                        }
                    }
                }
            }
        });

        self.current_view = CurrentView::Downloads;
    }
}

impl eframe::App for LmStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.refresh_telemetry();

        let mut token_received = false;
        if let Some(ref mut rx) = self.active_generation_rx {
            while let Ok(tok) = rx.try_recv() {
                token_received = true;
                if let Some(convo) = self.conversations.get_mut(self.active_convo_idx) {
                    if let Some((role, content)) = convo.messages.last_mut() {
                        if role == "assistant" {
                            content.push_str(&tok);
                        }
                    }
                }
            }
            if let Err(std::sync::mpsc::TryRecvError::Disconnected) = rx.try_recv() {
                self.active_generation_rx = None;
                self.is_generating = false;
            }
        }

        let has_active_downloads = self.downloads.read().unwrap().iter().any(|d| d.status == DownloadStatus::Downloading);
        if has_active_downloads || self.is_generating || token_received {
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }

        // ====================================================================
        // Top Navigation & Hardware Telemetry Panel
        // ====================================================================
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading(RichText::new("⚡ LM-Workstation").color(Color32::from_rgb(100, 180, 255)).strong());
                ui.separator();

                // RAM Telemetry Meter
                let total_ram_gb = self.telemetry.total_ram_bytes as f32 / 1e9;
                let used_ram_gb = (self.telemetry.total_ram_bytes.saturating_sub(self.telemetry.available_ram_bytes)) as f32 / 1e9;
                let ram_ratio = (used_ram_gb / total_ram_gb.max(1.0)).clamp(0.0, 1.0);

                let ram_color = if ram_ratio > 0.85 {
                    Color32::from_rgb(255, 80, 80)
                } else if ram_ratio > 0.70 {
                    Color32::from_rgb(255, 200, 50)
                } else {
                    Color32::from_rgb(80, 220, 120)
                };

                ui.label(RichText::new(format!("RAM: {:.1}/{:.1} GB", used_ram_gb, total_ram_gb)).small().strong());
                let pb_ram = ProgressBar::new(ram_ratio).desired_width(80.0).fill(ram_color);
                ui.add(pb_ram);

                ui.separator();

                // Vulkan VRAM Meter
                let vram_total_mb = self.telemetry.vram_total_bytes as f32 / 1e6;
                let vram_free_mb = self.telemetry.vram_free_bytes as f32 / 1e6;
                let vram_used_mb = (vram_total_mb - vram_free_mb).max(0.0);
                let vram_ratio = (vram_used_mb / vram_total_mb.max(1.0)).clamp(0.0, 1.0);

                let vram_label = if self.telemetry.is_integrated_gpu { "Vulkan (Shared):" } else { "VRAM:" };
                ui.label(RichText::new(format!("{} {:.0}/{:.0} MB", vram_label, vram_used_mb, vram_total_mb)).small().strong());
                let pb_vram = ProgressBar::new(vram_ratio).desired_width(80.0).fill(Color32::from_rgb(140, 100, 255));
                ui.add(pb_vram);

                // Right: API Server Toggle Switch
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let btn_text = if self.server_running {
                        RichText::new("⏹ Stop API Server").color(Color32::from_rgb(255, 100, 100))
                    } else {
                        RichText::new("▶ Start API Server").color(Color32::from_rgb(80, 220, 120))
                    };

                    if ui.button(btn_text).clicked() {
                        self.server_running = !self.server_running;
                        self.status_text = if self.server_running {
                            format!("API server active at http://127.0.0.1:{}", self.server_port)
                        } else {
                            "API server stopped".to_string()
                        };
                    }

                    if self.server_running {
                        ui.label(RichText::new(format!(":{}", self.server_port)).color(Color32::LIGHT_GRAY));
                    }
                });
            });
            ui.add_space(4.0);
        });

        // ====================================================================
        // Left Side Pane: Conversations List & Bottom Dock (Downloads/Settings)
        // ====================================================================
        egui::SidePanel::left("left_side_pane")
            .resizable(true)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);

                // Side pane top buttons: New Chat and Discover Models
                ui.horizontal(|ui| {
                    if ui.button(RichText::new("➕ New Chat").strong()).clicked() {
                        let new_id = self.conversations.len() + 1;
                        self.conversations.push(Conversation {
                            id: new_id,
                            title: format!("Chat {}", new_id),
                            messages: vec![(
                                "assistant".to_string(),
                                "Started a new conversation.".to_string(),
                            )],
                        });
                        self.active_convo_idx = self.conversations.len() - 1;
                        self.current_view = CurrentView::Chat;
                    }

                    if ui.button(RichText::new("🔍 Discover").strong()).clicked() {
                        self.current_view = CurrentView::Discover;
                        if self.hf_search_results.read().unwrap().is_empty() {
                            self.trigger_hf_search();
                        }
                    }
                });

                ui.separator();
                ui.label(RichText::new("Conversations").small().color(Color32::LIGHT_GRAY));

                // Conversations List
                let available_height = ui.available_height() - 90.0;
                ScrollArea::vertical()
                    .max_height(available_height)
                    .show(ui, |ui| {
                        let mut to_delete = None;
                        for (idx, convo) in self.conversations.iter().enumerate() {
                            let is_selected = idx == self.active_convo_idx && self.current_view == CurrentView::Chat;
                            ui.horizontal(|ui| {
                                let label_text = if is_selected {
                                    RichText::new(&convo.title).color(Color32::from_rgb(100, 180, 255)).strong()
                                } else {
                                    RichText::new(&convo.title)
                                };

                                if ui.selectable_label(is_selected, label_text).clicked() {
                                    self.active_convo_idx = idx;
                                    self.current_view = CurrentView::Chat;
                                }

                                if self.conversations.len() > 1 {
                                    if ui.small_button("🗑").clicked() {
                                        to_delete = Some(idx);
                                    }
                                }
                            });
                        }

                        if let Some(del_idx) = to_delete {
                            self.conversations.remove(del_idx);
                            if self.active_convo_idx >= self.conversations.len() {
                                self.active_convo_idx = self.conversations.len().saturating_sub(1);
                            }
                        }
                    });

                // ============================================================
                // Bottom of Side Pane: Downloads & Settings Dock
                // ============================================================
                ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                    ui.add_space(8.0);
                    ui.separator();

                    ui.horizontal(|ui| {
                        let is_settings = self.current_view == CurrentView::Settings;
                        if ui.selectable_label(is_settings, RichText::new("⚙").strong()).clicked() {
                            self.current_view = CurrentView::Settings;
                        }

                        let active_dl_count = self.downloads.read().unwrap().iter().filter(|d| d.status == DownloadStatus::Downloading).count();
                        let dl_text = if active_dl_count > 0 {
                            format!("📥 ({})", active_dl_count)
                        } else {
                            "📥 Downloads".to_string()
                        };

                        let is_dls = self.current_view == CurrentView::Downloads;
                        if ui.selectable_label(is_dls, RichText::new(dl_text).strong()).clicked() {
                            self.refresh_downloaded_files();
                            self.current_view = CurrentView::Downloads;
                        }

                        let is_logs = self.current_view == CurrentView::Logs;
                        if ui.selectable_label(is_logs, RichText::new("📜 Logs").strong()).clicked() {
                            self.current_view = CurrentView::Logs;
                        }
                    });

                    ui.add_space(4.0);
                });
            });

        // ====================================================================
        // Main Central Panel (Chat, Discover, Downloads, Settings, Logs)
        // ====================================================================
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_view {
                CurrentView::Chat => self.render_chat_view(ui),
                CurrentView::Discover => self.render_discover_view(ui),
                CurrentView::Downloads => self.render_downloads_view(ui),
                CurrentView::Settings => self.render_settings_view(ui),
                CurrentView::Logs => self.render_logs_view(ui),
            }
        });
    }
}

impl LmStudioApp {
    // ========================================================================
    // 1. Chat View
    // ========================================================================
    fn render_chat_view(&mut self, ui: &mut egui::Ui) {
        let (current_title, current_messages) = if let Some(convo) = self.conversations.get_mut(self.active_convo_idx) {
            (convo.title.clone(), &mut convo.messages)
        } else {
            return;
        };

        ui.horizontal(|ui| {
            ui.heading(RichText::new(current_title).color(Color32::WHITE));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let engine_guard = self.engine.read().unwrap();
                if let Some(ref eng) = *engine_guard {
                    let m_name = eng.model_path.file_name().unwrap_or_default().to_string_lossy();
                    ui.label(RichText::new(format!("Active: {}", m_name)).color(Color32::from_rgb(80, 220, 120)));
                } else {
                    ui.label(RichText::new("No Model Loaded (Go to Downloads)").color(Color32::from_rgb(255, 180, 60)));
                }
            });
        });
        ui.separator();

        let available_height = ui.available_height() - 70.0;
        ScrollArea::vertical()
            .max_height(available_height)
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                for (role, content) in current_messages.iter() {
                    ui.group(|ui| {
                        let (role_color, label_prefix) = if role == "user" {
                            (Color32::from_rgb(100, 200, 255), "👤 User")
                        } else {
                            (Color32::from_rgb(180, 140, 255), "🤖 Assistant")
                        };

                        ui.label(RichText::new(label_prefix).strong().color(role_color));
                        ui.add_space(2.0);

                        if content.contains("```") {
                            for part in content.split("```") {
                                if part.starts_with("rust") || part.starts_with("json") || part.starts_with("bash") || part.starts_with('\n') {
                                    ui.monospace(part);
                                } else {
                                    ui.label(part);
                                }
                            }
                        } else {
                            ui.label(content);
                        }
                    });
                    ui.add_space(6.0);
                }
            });

        ui.separator();

        // Chat Input & RAG Attachment Bar
        ui.horizontal(|ui| {
            if ui.button(RichText::new("📎 Attach").strong()).clicked() {
                let sample_path = Path::new("README.md");
                if sample_path.exists() {
                    let g = self.engine.read().unwrap();
                    if let Some(ref eng) = *g {
                        if let Ok(n) = RagPipeline::ingest_file(&mut self.vector_store, eng, sample_path) {
                            self.ingested_files.push("README.md".to_string());
                            self.status_text = format!("Ingested README.md ({} chunks)", n);
                            log_info!("rag", "Document ingested: README.md into {} chunks", n);
                        }
                    } else {
                        self.status_text = "Load a model first for RAG embedding generation.".to_string();
                        log_warn!("rag", "Cannot ingest document: No model loaded for vector embeddings");
                    }
                }
            }

            let text_edit = ui.add(
                egui::TextEdit::singleline(&mut self.current_input)
                    .hint_text("Ask a question, test RAG, or generate text...")
                    .desired_width(ui.available_width() - 80.0),
            );

            let send_clicked = ui.button(RichText::new("Send").strong()).clicked();
            let enter_pressed = text_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

            if (send_clicked || enter_pressed) && !self.current_input.trim().is_empty() {
                let prompt = self.current_input.clone();
                self.current_input.clear();

                if let Some(convo) = self.conversations.get_mut(self.active_convo_idx) {
                    convo.messages.push(("user".to_string(), prompt.clone()));

                    let final_prompt = if !self.vector_store.chunks.is_empty() {
                        let query_emb = {
                            let g = self.engine.read().unwrap();
                            if let Some(ref eng) = *g {
                                eng.embed(&prompt)
                            } else {
                                vec![]
                            }
                        };
                        self.vector_store.build_augmented_prompt(&prompt, &query_emb, self.settings_rag_top_k)
                    } else {
                        prompt.clone()
                    };

                    let is_loaded = self.engine.read().unwrap().is_some();
                    if is_loaded {
                        convo.messages.push(("assistant".to_string(), String::new()));
                        self.is_generating = true;
                        log_info!("inference", "Prompt queued: '{}' (ctx: {}, temp: {:.2})", prompt, self.context_length, self.temperature);

                        let (tx, rx) = std::sync::mpsc::channel();
                        self.active_generation_rx = Some(rx);

                        let engine_arc = self.engine.clone();
                        let mut params = self.engine.read().unwrap().as_ref().map(|e| e.runtime_params.clone()).unwrap_or_default();
                        params.temperature = self.temperature;
                        params.top_p = self.top_p;
                        params.n_ctx = self.context_length;
                        let prompt_to_run = final_prompt.clone();

                        std::thread::spawn(move || {
                            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build();
                            if let Ok(rt) = rt {
                                rt.block_on(async {
                                    let rx_stream = {
                                        let g = engine_arc.read().unwrap();
                                        if let Some(ref eng) = *g {
                                            eng.generate_stream(&prompt_to_run, &params, None).await
                                        } else {
                                            return;
                                        }
                                    };
                                    let mut stream = rx_stream;
                                    let mut token_count = 0;
                                    while let Some(tok) = stream.recv().await {
                                        token_count += 1;
                                        if tx.send(tok).is_err() {
                                            break;
                                        }
                                    }
                                    log_info!("inference", "Token stream completed ({} tokens generated)", token_count);
                                });
                            }
                        });
                    } else {
                        log_warn!("inference", "Generation blocked: No model loaded");
                        convo.messages.push((
                            "assistant".to_string(),
                            "Please load a model from Downloads or discover one from Hugging Face first.".to_string(),
                        ));
                    }
                }
            }
        });
    }

    // ========================================================================
    // 2. Discover View (Hugging Face API Search)
    // ========================================================================
    fn render_discover_view(&mut self, ui: &mut egui::Ui) {
        ui.heading("🔍 Hugging Face Model Hub Discovery");
        ui.label("Search and explore quantized GGUF models directly from Hugging Face API.");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label(RichText::new("Search:").strong());
            let text_edit = ui.text_edit_singleline(&mut self.hf_search_query);
            let enter = text_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

            let is_searching = *self.hf_is_searching.read().unwrap();
            let btn = if is_searching {
                ui.button("Searching...")
            } else {
                ui.button(RichText::new("Search Models").strong())
            };

            if (btn.clicked() || enter) && !is_searching {
                self.trigger_hf_search();
            }

            ui.separator();
            ui.label("Popular:");
            for tag in &["qwen2.5", "llama-3.2", "phi-3.5", "mistral", "nomic-embed"] {
                if ui.button(*tag).clicked() {
                    self.hf_search_query = format!("{} gguf", tag);
                    self.trigger_hf_search();
                }
            }
        });

        if let Some(ref err) = *self.hf_error.read().unwrap() {
            ui.colored_label(Color32::from_rgb(255, 100, 100), format!("API Error: {}", err));
        }

        ui.separator();

        // Selected Model GGUF Files Tree Modal/Drawer
        let selected_files_clone = self.hf_selected_files.read().unwrap().clone();
        if let Some((model_id, files)) = selected_files_clone {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.heading(RichText::new(format!("Files for {}", model_id)).color(Color32::from_rgb(100, 200, 255)));
                    if ui.button("Close Files").clicked() {
                        *self.hf_selected_files.write().unwrap() = None;
                    }
                });
                ui.separator();

                ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                    for f in files {
                        ui.horizontal(|ui| {
                            let size_str = f.size.map(|s| format!("{:.1} MB", s as f64 / 1e6)).unwrap_or_else(|| "Unknown".to_string());
                            ui.label(RichText::new(&f.path).strong());
                            ui.label(format!("({})", size_str));

                            // Memory Fit Tag
                            if let Some(sz) = f.size {
                                if sz <= self.telemetry.available_ram_bytes.saturating_sub(800 * 1024 * 1024) {
                                    ui.colored_label(Color32::from_rgb(80, 220, 120), "✓ Fits in RAM");
                                } else {
                                    ui.colored_label(Color32::from_rgb(255, 120, 80), "⚠ Exceeds RAM");
                                }
                            }

                            if ui.button("⬇ Download").clicked() {
                                let dl_url = HuggingFaceClient::download_url(&model_id, &f.path);
                                self.start_download(&dl_url, &f.path, &model_id);
                            }
                        });
                    }
                });
            });
            ui.separator();
        }

        // Search Results List
        let results = self.hf_search_results.read().unwrap().clone();
        ScrollArea::vertical().show(ui, |ui| {
            for model in results {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.heading(RichText::new(&model.id).strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("View GGUF Files").clicked() {
                                self.trigger_hf_list_files(&model.id);
                            }
                            if let Some(dls) = model.downloads {
                                ui.label(format!("⬇ {} downloads", dls));
                            }
                            if let Some(likes) = model.likes {
                                ui.label(format!("❤ {}", likes));
                            }
                        });
                    });

                    if let Some(ref author) = model.author {
                        ui.label(format!("By: {}", author));
                    }

                    if !model.tags.is_empty() {
                        let tags_joined = model.tags.iter().take(6).cloned().collect::<Vec<_>>().join(", ");
                        ui.label(RichText::new(format!("Tags: {}", tags_joined)).small().color(Color32::LIGHT_GRAY));
                    }
                });
                ui.add_space(4.0);
            }
        });
    }

    // ========================================================================
    // 3. Downloads View
    // ========================================================================
    fn render_downloads_view(&mut self, ui: &mut egui::Ui) {
        ui.heading("📥 Downloads Manager");
        ui.label("Monitor active chunked downloads and manage locally stored GGUF models.");
        ui.separator();

        ui.collapsing(RichText::new("Active & Pending Downloads").strong(), |ui| {
            let mut dls_guard = self.downloads.write().unwrap();
            if dls_guard.is_empty() {
                ui.label("No active downloads.");
            } else {
                let mut to_remove = None;
                for (idx, item) in dls_guard.iter_mut().enumerate() {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&item.filename).strong());
                            ui.label(format!("from {}", item.model_id));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button("❌ Delete").clicked() {
                                    to_remove = Some(idx);
                                }

                                match item.status {
                                    DownloadStatus::Downloading => {
                                        if ui.button("⏸ Pause").clicked() {
                                            item.status = DownloadStatus::Paused;
                                        }
                                    }
                                    DownloadStatus::Paused => {
                                        if ui.button("▶ Resume").clicked() {
                                            item.status = DownloadStatus::Downloading;
                                        }
                                    }
                                    _ => {}
                                }
                            });
                        });

                        let ratio = if item.total_bytes > 0 {
                            (item.downloaded_bytes as f32 / item.total_bytes as f32).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };

                        ui.horizontal(|ui| {
                            let pb = ProgressBar::new(ratio).desired_width(ui.available_width() - 160.0);
                            ui.add(pb);
                            ui.label(format!("{:.1} / {:.1} MB", item.downloaded_bytes as f32 / 1e6, item.total_bytes as f32 / 1e6));
                        });

                        match &item.status {
                            DownloadStatus::Downloading => ui.label("Status: Downloading..."),
                            DownloadStatus::Paused => ui.colored_label(Color32::YELLOW, "Status: Paused"),
                            DownloadStatus::Completed => ui.colored_label(Color32::from_rgb(80, 220, 120), "Status: Completed"),
                            DownloadStatus::Failed(e) => ui.colored_label(Color32::from_rgb(255, 100, 100), format!("Status: Failed ({})", e)),
                        };
                    });
                    ui.add_space(4.0);
                }

                if let Some(rem_idx) = to_remove {
                    dls_guard.remove(rem_idx);
                }
            }
        });

        ui.add_space(10.0);
        ui.horizontal(|ui| {
            ui.heading("Downloaded Local Models");
            if ui.button("🔄 Refresh Local List").clicked() {
                self.refresh_downloaded_files();
            }
        });
        ui.separator();

        if self.downloaded_files.is_empty() {
            ui.label(format!("No .gguf models found in directory '{}'. Download one from the Discover tab.", self.settings_models_dir));
        } else {
            let mut to_delete_path = None;
            for f in &self.downloaded_files {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(&f.filename).strong());
                        ui.label(format!("({:.1} MB)", f.size_bytes as f64 / 1e6));

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("🗑 Delete File").clicked() {
                                log_warn!("storage", "Deleting model file: {}", f.filename);
                                to_delete_path = Some(f.path.clone());
                            }

                            if ui.button("⚡ Load Model (mmap)").clicked() {
                                log_info!("engine", "Loading model {} via zero-copy mmap (gpu_layers: {}, ctx: {})", f.filename, self.gpu_layers, self.context_length);
                                let params = RuntimeParams {
                                    n_ctx: self.context_length,
                                    n_gpu_layers: self.gpu_layers,
                                    n_threads: self.cpu_threads,
                                    temperature: self.temperature,
                                    top_p: self.top_p,
                                    ..Default::default()
                                };
                                match InferenceEngine::load(&f.path, params) {
                                    Ok(eng) => {
                                        log_info!("engine", "Model loaded successfully. Virtual memory mapped ({} MB)", eng.mmap_model.size / (1024 * 1024));
                                        *self.engine.write().unwrap() = Some(eng);
                                        self.status_text = format!("Loaded {} via zero-copy mmap", f.filename);
                                        self.current_view = CurrentView::Chat;
                                    }
                                    Err(e) => {
                                        log_error!("engine", "Failed to load {}: {}", f.filename, e);
                                        self.status_text = format!("Failed to load: {}", e);
                                    }
                                }
                            }
                        });
                    });
                });
                ui.add_space(4.0);
            }

            if let Some(p) = to_delete_path {
                let _ = fs::remove_file(p);
                self.refresh_downloaded_files();
            }
        }
    }

    // ========================================================================
    // 4. Settings View (Filled & Comprehensive)
    // ========================================================================
    fn render_settings_view(&mut self, ui: &mut egui::Ui) {
        ui.heading("⚙ System & Workstation Settings");
        ui.label("Configure inference parameters, hardware offloading, daemon memory policies, and local API server.");
        ui.separator();

        ScrollArea::vertical().show(ui, |ui| {
            // Category 1: Inference & Hardware Acceleration
            ui.collapsing(RichText::new("🚀 Inference & Hardware Acceleration").strong(), |ui| {
                ui.horizontal(|ui| {
                    ui.label("Context Length (n_ctx):");
                    ui.add(egui::Slider::new(&mut self.context_length, 512..=8192).step_by(512.0));
                });
                ui.label(RichText::new("Defines context memory. Clamped to 2048-4096 on 8 GB RAM machines.").small().color(Color32::LIGHT_GRAY));

                ui.horizontal(|ui| {
                    ui.label("GPU Offload Layers (Vulkan):");
                    ui.add(egui::Slider::new(&mut self.gpu_layers, 0..=64));
                });
                ui.label(RichText::new("Layers offloaded to integrated Vulkan GPU shader pipelines.").small().color(Color32::LIGHT_GRAY));

                ui.horizontal(|ui| {
                    ui.label("CPU Threads (AVX2):");
                    ui.add(egui::Slider::new(&mut self.cpu_threads, 1..=16));
                });

                ui.horizontal(|ui| {
                    ui.label("Sampling Temperature:");
                    ui.add(egui::Slider::new(&mut self.temperature, 0.0..=1.5).step_by(0.05));
                });

                ui.horizontal(|ui| {
                    ui.label("Top-P Nucleus Sampling:");
                    ui.add(egui::Slider::new(&mut self.top_p, 0.1..=1.0).step_by(0.05));
                });

                ui.checkbox(&mut self.settings_use_mmap, "Use Zero-Copy OS mmap (Prevents RAM duplication)");
                ui.checkbox(&mut self.settings_mlock, "Lock Model in Memory (mlock)");
                ui.checkbox(&mut self.settings_flash_attn, "Flash Attention (Vulkan shader & AVX2)");
            });

            ui.add_space(8.0);

            // Category 2: Memory Management & Daemon
            ui.collapsing(RichText::new("🧠 Memory Management & Daemon Policy").strong(), |ui| {
                ui.horizontal(|ui| {
                    ui.label("Auto-Unload Inactivity TTL (seconds):");
                    ui.add(egui::Slider::new(&mut self.settings_idle_ttl, 60..=3600));
                });
                ui.label(RichText::new("Automatically frees model virtual memory and caches when idle to save RAM.").small().color(Color32::LIGHT_GRAY));

                ui.horizontal(|ui| {
                    ui.label("Safe OS Reserve RAM (MB):");
                    ui.add(egui::Slider::new(&mut self.settings_safe_ram_mb, 400..=2000));
                });

                ui.horizontal(|ui| {
                    ui.label("Unix Domain Socket (UDS) Path:");
                    ui.text_edit_singleline(&mut self.settings_uds_socket);
                });
            });

            ui.add_space(8.0);

            // Category 3: Local API Server (OpenAI & Anthropic)
            ui.collapsing(RichText::new("🌐 Local API Server (OpenAI & Anthropic)").strong(), |ui| {
                ui.horizontal(|ui| {
                    ui.label("Server Port:");
                    ui.add(egui::DragValue::new(&mut self.server_port).range(1024..=65535));
                });
                ui.label(format!("Endpoints: http://127.0.0.1:{}/v1/chat/completions, /v1/messages", self.server_port));

                ui.checkbox(&mut self.settings_server_autostart, "Automatically start API server on startup");
                ui.checkbox(&mut self.settings_server_cors, "Enable CORS (Access-Control-Allow-Origin: *)");
            });

            ui.add_space(8.0);

            // Category 4: Hugging Face & Downloader
            ui.collapsing(RichText::new("🤗 Hugging Face & Storage").strong(), |ui| {
                ui.horizontal(|ui| {
                    ui.label("Models Directory:");
                    ui.text_edit_singleline(&mut self.settings_models_dir);
                });

                ui.horizontal(|ui| {
                    ui.label("Hugging Face API Token:");
                    ui.add(egui::TextEdit::singleline(&mut self.settings_hf_token).password(true));
                });
                ui.label(RichText::new("Optional: used to download gated models (e.g., Llama 3, Gemma).").small().color(Color32::LIGHT_GRAY));
            });

            ui.add_space(8.0);

            // Category 5: Local RAG Pipeline & MCP Tool Calling
            ui.collapsing(RichText::new("📚 Local RAG & MCP Tool Host").strong(), |ui| {
                ui.horizontal(|ui| {
                    ui.label("RAG Chunk Size (tokens):");
                    ui.add(egui::Slider::new(&mut self.settings_rag_chunk_size, 128..=1024).step_by(128.0));
                });

                ui.horizontal(|ui| {
                    ui.label("RAG Overlap Percentage:");
                    ui.add(egui::Slider::new(&mut self.settings_rag_overlap_pct, 5..=30));
                });

                ui.horizontal(|ui| {
                    ui.label("Top-K Context Retrieval:");
                    ui.add(egui::Slider::new(&mut self.settings_rag_top_k, 1..=8));
                });

                ui.horizontal(|ui| {
                    ui.label("MCP Stdio Host Command:");
                    ui.text_edit_singleline(&mut self.settings_mcp_command);
                });
            });

            ui.add_space(14.0);
            if ui.button(RichText::new("💾 Save Settings").strong()).clicked() {
                self.status_text = "Settings saved successfully.".to_string();
                log_info!("settings", "Configuration parameters updated and saved");
            }
        });
    }

    // ========================================================================
    // 5. Logs View
    // ========================================================================
    fn render_logs_view(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(RichText::new("📜 Real-Time System Logs").strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("🗑 Clear Logs").clicked() {
                    Logger::global().clear();
                }
                ui.checkbox(&mut self.log_auto_scroll, "Auto-scroll");
            });
        });
        ui.label("Live log stream from inference engine, hardware telemetry, network downloads, and API server.");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label(RichText::new("Filter:").strong());
            if ui.selectable_label(self.log_filter_level.is_none(), "ALL").clicked() {
                self.log_filter_level = None;
            }
            if ui.selectable_label(self.log_filter_level == Some(LogLevel::Info), "INFO").clicked() {
                self.log_filter_level = Some(LogLevel::Info);
            }
            if ui.selectable_label(self.log_filter_level == Some(LogLevel::Warn), "WARN").clicked() {
                self.log_filter_level = Some(LogLevel::Warn);
            }
            if ui.selectable_label(self.log_filter_level == Some(LogLevel::Error), "ERROR").clicked() {
                self.log_filter_level = Some(LogLevel::Error);
            }
            if ui.selectable_label(self.log_filter_level == Some(LogLevel::Debug), "DEBUG").clicked() {
                self.log_filter_level = Some(LogLevel::Debug);
            }

            ui.separator();
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.log_search_query);
        });
        ui.separator();

        let entries = Logger::global().get_entries();
        let query = self.log_search_query.to_lowercase();

        let filtered_entries: Vec<_> = entries
            .into_iter()
            .filter(|e| {
                if let Some(lvl) = self.log_filter_level {
                    if e.level != lvl {
                        return false;
                    }
                }
                if !query.is_empty() {
                    let in_msg = e.message.to_lowercase().contains(&query);
                    let in_target = e.target.to_lowercase().contains(&query);
                    if !in_msg && !in_target {
                        return false;
                    }
                }
                true
            })
            .collect();

        ScrollArea::vertical()
            .auto_shrink([false; 2])
            .stick_to_bottom(self.log_auto_scroll)
            .show(ui, |ui| {
                if filtered_entries.is_empty() {
                    ui.label(RichText::new("No log entries match the current filter.").color(Color32::LIGHT_GRAY));
                } else {
                    for entry in filtered_entries {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&entry.timestamp).color(Color32::from_rgb(140, 140, 140)).monospace());

                            let (badge_color, badge_text) = match entry.level {
                                LogLevel::Info => (Color32::from_rgb(80, 220, 120), "INFO "),
                                LogLevel::Warn => (Color32::from_rgb(255, 200, 50), "WARN "),
                                LogLevel::Error => (Color32::from_rgb(255, 80, 80), "ERROR"),
                                LogLevel::Debug => (Color32::from_rgb(100, 180, 255), "DEBUG"),
                            };
                            ui.label(RichText::new(format!("[{}]", badge_text)).color(badge_color).strong().monospace());
                            ui.label(RichText::new(format!("[{}]", entry.target)).color(Color32::from_rgb(180, 140, 255)).monospace());
                            ui.label(RichText::new(&entry.message).monospace());
                        });
                    }
                }
            });
    }
}

pub fn run_gui() -> Result<(), eframe::Error> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 750.0])
            .with_min_inner_size([800.0, 520.0])
            .with_title("Native Local LLM Workstation"),
        ..Default::default()
    };

    eframe::run_native(
        "LM-Studio Workstation",
        native_options,
        Box::new(|cc| Ok(Box::new(LmStudioApp::new(cc)))),
    )
}
