use std::fs::{self, File};
use std::io::{Read, Write as IoWrite};
use std::path::Path;
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareTelemetry {
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
    pub is_integrated_gpu: bool,
    pub has_avx2: bool,
    pub has_vulkan: bool,
    pub cpu_cores: usize,
}

impl HardwareTelemetry {
    pub fn probe() -> Self {
        let (total_ram, avail_ram) = Self::read_meminfo();
        let (vram_total, vram_free, is_integrated, has_vulkan) = Self::probe_gpu(total_ram);
        let has_avx2 = Self::check_avx2();
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            total_ram_bytes: total_ram,
            available_ram_bytes: avail_ram,
            vram_total_bytes: vram_total,
            vram_free_bytes: vram_free,
            is_integrated_gpu: is_integrated,
            has_avx2,
            has_vulkan,
            cpu_cores,
        }
    }

    fn read_meminfo() -> (u64, u64) {
        let mut total = 8 * 1024 * 1024 * 1024; // fallback 8GB
        let mut avail = 4 * 1024 * 1024 * 1024; // fallback 4GB

        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    if let Some(kb) = rest.split_whitespace().next().and_then(|s| s.parse::<u64>().ok()) {
                        total = kb * 1024;
                    }
                } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    if let Some(kb) = rest.split_whitespace().next().and_then(|s| s.parse::<u64>().ok()) {
                        avail = kb * 1024;
                    }
                }
            }
        }
        (total, avail)
    }

    fn check_avx2() -> bool {
        if let Ok(content) = fs::read_to_string("/proc/cpuinfo") {
            content.contains("avx2")
        } else {
            is_x86_feature_detected!("avx2")
        }
    }

    fn probe_gpu(total_ram: u64) -> (u64, u64, bool, bool) {
        let mut vram_total = 0u64;
        let mut vram_used = 0u64;
        let mut has_drm = false;

        // Check Linux sysfs DRM device telemetry
        if let Ok(entries) = fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let path = entry.path();
                let vram_path = path.join("device/mem_info_vram_total");
                let used_path = path.join("device/mem_info_vram_used");
                if vram_path.exists() {
                    has_drm = true;
                    if let Ok(val) = fs::read_to_string(&vram_path) {
                        if let Ok(bytes) = val.trim().parse::<u64>() {
                            vram_total = vram_total.max(bytes);
                        }
                    }
                    if let Ok(val) = fs::read_to_string(&used_path) {
                        if let Ok(bytes) = val.trim().parse::<u64>() {
                            vram_used = bytes;
                        }
                    }
                }
            }
        }

        let has_vulkan = Path::new("/usr/share/vulkan/icd.d").exists()
            || Path::new("/etc/vulkan/icd.d").exists()
            || has_drm;

        let is_integrated = vram_total == 0 || vram_total <= 512 * 1024 * 1024;
        if is_integrated {
            // Integrated Vulkan GPUs dynamically share system RAM (typically up to 50%)
            vram_total = total_ram / 2;
            let vram_free = (total_ram / 2).saturating_sub(vram_used);
            (vram_total, vram_free, true, has_vulkan)
        } else {
            let vram_free = vram_total.saturating_sub(vram_used);
            (vram_total, vram_free, false, has_vulkan)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufMetadata {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub architecture: String,
    pub parameter_count: Option<u64>,
    pub layer_count: u32,
    pub context_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub embedding_length: u32,
    pub quantization: String,
    pub chat_template: Option<String>,
}

impl Default for GgufMetadata {
    fn default() -> Self {
        Self {
            magic: 0x46554747, // 'GGUF'
            version: 3,
            tensor_count: 0,
            kv_count: 0,
            architecture: "llama".to_string(),
            parameter_count: None,
            layer_count: 32,
            context_length: 4096,
            head_count: 32,
            head_count_kv: 8,
            embedding_length: 4096,
            quantization: "Q4_K_M".to_string(),
            chat_template: None,
        }
    }
}

pub struct GgufHeaderParser;

impl GgufHeaderParser {
    pub fn parse(bytes: &[u8]) -> Result<GgufMetadata, String> {
        if bytes.len() < 24 {
            return Err("Header buffer too small for GGUF header".to_string());
        }

        let mut offset = 0;
        let magic = Self::read_u32(bytes, &mut offset).ok_or("Header buffer too small for GGUF magic")?;
        if magic != 0x46554747 {
            return Err(format!("Invalid GGUF magic: 0x{:08X}", magic));
        }

        let version = Self::read_u32(bytes, &mut offset).ok_or("Header buffer too small for GGUF version")?;
        let tensor_count = Self::read_u64(bytes, &mut offset).ok_or("Header buffer too small for GGUF tensor count")?;
        let kv_count = Self::read_u64(bytes, &mut offset).ok_or("Header buffer too small for GGUF metadata count")?;
        let mut meta = GgufMetadata {
            magic,
            version,
            tensor_count,
            kv_count,
            ..Default::default()
        };

        for _ in 0..kv_count {
            if offset + 8 > bytes.len() {
                break;
            }
            let key_len = match Self::read_u64(bytes, &mut offset) {
                Some(len) => len as usize,
                None => break,
            };

            if offset + key_len > bytes.len() {
                break;
            }
            let key = String::from_utf8_lossy(&bytes[offset..offset + key_len]).to_string();
            offset += key_len;

            let val_type = match Self::read_u32(bytes, &mut offset) {
                Some(t) => t,
                None => break,
            };

            Self::extract_kv(&key, val_type, bytes, &mut offset, &mut meta);
        }

        Ok(meta)
    }

    fn read_u32(bytes: &[u8], offset: &mut usize) -> Option<u32> {
        if *offset + 4 > bytes.len() {
            return None;
        }
        let val = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
        *offset += 4;
        Some(val)
    }

    fn read_u64(bytes: &[u8], offset: &mut usize) -> Option<u64> {
        if *offset + 8 > bytes.len() {
            return None;
        }
        let val = u64::from_le_bytes(bytes[*offset..*offset + 8].try_into().ok()?);
        *offset += 8;
        Some(val)
    }

    fn read_string(bytes: &[u8], offset: &mut usize) -> Option<String> {
        let len = Self::read_u64(bytes, offset)? as usize;
        if *offset + len > bytes.len() {
            return None;
        }
        let s = String::from_utf8_lossy(&bytes[*offset..*offset + len]).to_string();
        *offset += len;
        Some(s)
    }

    fn skip_value(val_type: u32, bytes: &[u8], offset: &mut usize) {
        match val_type {
            0 | 1 | 7 => *offset += 1, // uint8, int8, bool
            2 | 3 => *offset += 2,     // uint16, int16
            4 | 5 | 6 => *offset += 4, // uint32, int32, float32
            8 => {
                if let Some(len) = Self::read_u64(bytes, offset) {
                    *offset += len as usize;
                }
            }
            10 | 11 | 12 => *offset += 8, // uint64, int64, float64
            9 => {
                // array: item_type (u32), array_len (u64)
                if let (Some(item_type), Some(arr_len)) = (Self::read_u32(bytes, offset), Self::read_u64(bytes, offset)) {
                    for _ in 0..arr_len {
                        Self::skip_value(item_type, bytes, offset);
                    }
                }
            }
            _ => *offset = bytes.len(),
        }
    }

    fn extract_kv(key: &str, val_type: u32, bytes: &[u8], offset: &mut usize, meta: &mut GgufMetadata) {
        match key {
            "general.architecture" => {
                if val_type == 8 {
                    if let Some(s) = Self::read_string(bytes, offset) {
                        meta.architecture = s;
                    }
                } else {
                    Self::skip_value(val_type, bytes, offset);
                }
            }
            "general.parameter_count" => {
                if val_type == 10 || val_type == 11 {
                    meta.parameter_count = Self::read_u64(bytes, offset);
                } else {
                    Self::skip_value(val_type, bytes, offset);
                }
            }
            "tokenizer.chat_template" => {
                if val_type == 8 {
                    meta.chat_template = Self::read_string(bytes, offset);
                } else {
                    Self::skip_value(val_type, bytes, offset);
                }
            }
            k if k.ends_with(".block_count") => {
                if val_type == 4 {
                    if let Some(v) = Self::read_u32(bytes, offset) {
                        meta.layer_count = v;
                    }
                } else if val_type == 10 {
                    if let Some(v) = Self::read_u64(bytes, offset) {
                        meta.layer_count = v as u32;
                    }
                } else {
                    Self::skip_value(val_type, bytes, offset);
                }
            }
            k if k.ends_with(".context_length") => {
                if val_type == 4 {
                    if let Some(v) = Self::read_u32(bytes, offset) {
                        meta.context_length = v;
                    }
                } else if val_type == 10 {
                    if let Some(v) = Self::read_u64(bytes, offset) {
                        meta.context_length = v as u32;
                    }
                } else {
                    Self::skip_value(val_type, bytes, offset);
                }
            }
            k if k.ends_with(".attention.head_count") => {
                if val_type == 4 {
                    if let Some(v) = Self::read_u32(bytes, offset) {
                        meta.head_count = v;
                    }
                } else if val_type == 10 {
                    if let Some(v) = Self::read_u64(bytes, offset) {
                        meta.head_count = v as u32;
                    }
                } else {
                    Self::skip_value(val_type, bytes, offset);
                }
            }
            k if k.ends_with(".attention.head_count_kv") => {
                if val_type == 4 {
                    if let Some(v) = Self::read_u32(bytes, offset) {
                        meta.head_count_kv = v;
                    }
                } else if val_type == 10 {
                    if let Some(v) = Self::read_u64(bytes, offset) {
                        meta.head_count_kv = v as u32;
                    }
                } else {
                    Self::skip_value(val_type, bytes, offset);
                }
            }
            k if k.ends_with(".embedding_length") => {
                if val_type == 4 {
                    if let Some(v) = Self::read_u32(bytes, offset) {
                        meta.embedding_length = v;
                    }
                } else if val_type == 10 {
                    if let Some(v) = Self::read_u64(bytes, offset) {
                        meta.embedding_length = v as u32;
                    }
                } else {
                    Self::skip_value(val_type, bytes, offset);
                }
            }
            "general.file_type" => {
                if val_type == 4 {
                    if let Some(ft) = Self::read_u32(bytes, offset) {
                        meta.quantization = match ft {
                            0 => "F32",
                            1 => "F16",
                            2 => "Q4_0",
                            3 => "Q4_1",
                            7 => "Q8_0",
                            8 => "Q5_0",
                            9 => "Q5_1",
                            10 => "Q2_K",
                            11 => "Q3_K_S",
                            12 => "Q3_K_M",
                            13 => "Q3_K_L",
                            14 => "Q4_K_S",
                            15 => "Q4_K_M",
                            16 => "Q5_K_S",
                            17 => "Q5_K_M",
                            18 => "Q6_K",
                            _ => "Quantized",
                        }.to_string();
                    }
                } else {
                    Self::skip_value(val_type, bytes, offset);
                }
            }
            _ => {
                Self::skip_value(val_type, bytes, offset);
            }
        }
    }

    pub async fn fetch_remote_header(url: &str) -> Result<GgufMetadata, String> {
        let client = reqwest::Client::new();
        let res = client
            .get(url)
            .header("Range", "bytes=0-65535")
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        if !res.status().is_success() {
            return Err(format!("Remote server returned status {}", res.status()));
        }

        let bytes = res.bytes().await.map_err(|e| format!("Failed to read header bytes: {}", e))?;
        Self::parse(&bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitClassification {
    CompatibleFullOffload,
    PartialOffload,
    IncompatibleExceedsSafeRam,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBudgetEstimate {
    pub model_file_size_bytes: u64,
    pub kv_cache_size_bytes: u64,
    pub compute_buffer_overhead_bytes: u64,
    pub total_estimated_ram_bytes: u64,
    pub classification: FitClassification,
    pub suggested_gpu_layers: u32,
    pub recommended_ctx: u32,
    pub reason: String,
}

pub struct MemoryBudgetEstimator;

impl MemoryBudgetEstimator {
    pub const COMPUTE_BUFFER_OVERHEAD: u64 = 350 * 1024 * 1024; // ~350 MB

    pub fn estimate(
        meta: &GgufMetadata,
        model_file_size_bytes: u64,
        telemetry: &HardwareTelemetry,
        requested_ctx: Option<u32>,
    ) -> MemoryBudgetEstimate {
        let n_ctx = requested_ctx.unwrap_or(meta.context_length).min(8192) as u64;
        let layers = meta.layer_count.max(1) as u64;
        let kv_heads = meta.head_count_kv.max(1) as u64;
        let head_dim = if meta.head_count > 0 {
            (meta.embedding_length / meta.head_count).max(64) as u64
        } else {
            128
        };

        // KV Cache: 2 * layers * kv_heads * head_dim * n_ctx * 2 bytes (fp16)
        let kv_cache_size = 2 * layers * kv_heads * head_dim * n_ctx * 2;
        let total_estimated = model_file_size_bytes + kv_cache_size + Self::COMPUTE_BUFFER_OVERHEAD;

        // Safe RAM calculation for 8 GB constrained machines:
        // Reserve at least 800 MB for OS and background apps
        let safe_ram_budget = telemetry.available_ram_bytes.saturating_sub(800 * 1024 * 1024);

        let (classification, suggested_gpu_layers, recommended_ctx, reason) = if total_estimated <= telemetry.vram_free_bytes && !telemetry.is_integrated_gpu {
            (
                FitClassification::CompatibleFullOffload,
                meta.layer_count + 1,
                n_ctx as u32,
                "Model and KV cache fit entirely in dedicated GPU VRAM.".to_string(),
            )
        } else if total_estimated <= safe_ram_budget {
            let gpu_layers = if telemetry.has_vulkan {
                // Offload proportional to available GPU memory
                let layer_weight = model_file_size_bytes / layers.max(1);
                let offloadable = (telemetry.vram_free_bytes / layer_weight.max(1)) as u32;
                offloadable.min(meta.layer_count)
            } else {
                0
            };

            let class = if gpu_layers >= meta.layer_count {
                FitClassification::CompatibleFullOffload
            } else {
                FitClassification::PartialOffload
            };

            (
                class,
                gpu_layers,
                n_ctx as u32,
                format!(
                    "Fits within available RAM ({:.2} GB required, {:.2} GB safe available).",
                    total_estimated as f64 / 1e9,
                    safe_ram_budget as f64 / 1e9
                ),
            )
        } else {
            // Check if reducing context length makes it fit
            let reduced_ctx = 2048u64;
            let reduced_kv = 2 * layers * kv_heads * head_dim * reduced_ctx * 2;
            let reduced_total = model_file_size_bytes + reduced_kv + Self::COMPUTE_BUFFER_OVERHEAD;

            if reduced_total <= safe_ram_budget {
                (
                    FitClassification::PartialOffload,
                    if telemetry.has_vulkan { meta.layer_count / 2 } else { 0 },
                    reduced_ctx as u32,
                    format!(
                        "Exceeds safe RAM at n_ctx={}, but fits if context is clamped to 2048.",
                        n_ctx
                    ),
                )
            } else {
                (
                    FitClassification::IncompatibleExceedsSafeRam,
                    0,
                    2048,
                    format!(
                        "Requires {:.2} GB, exceeding safe system RAM limit of {:.2} GB.",
                        total_estimated as f64 / 1e9,
                        safe_ram_budget as f64 / 1e9
                    ),
                )
            }
        };

        MemoryBudgetEstimate {
            model_file_size_bytes,
            kv_cache_size_bytes: kv_cache_size,
            compute_buffer_overhead_bytes: Self::COMPUTE_BUFFER_OVERHEAD,
            total_estimated_ram_bytes: total_estimated,
            classification,
            suggested_gpu_layers,
            recommended_ctx,
            reason,
        }
    }
}

pub struct ResumableDownloader;

impl ResumableDownloader {
    pub async fn download_with_resume<F>(
        url: &str,
        dest_path: &Path,
        expected_sha256: Option<&str>,
        progress_callback: F,
    ) -> Result<(), String>
    where
        F: Fn(u64, u64) + Send + 'static,
    {
        let client = reqwest::Client::new();
        let mut existing_bytes = 0u64;

        if dest_path.exists() {
            if let Ok(m) = fs::metadata(dest_path) {
                existing_bytes = m.len();
            }
        }

        let mut req = client.get(url);
        if existing_bytes > 0 {
            req = req.header("Range", format!("bytes={}-", existing_bytes));
        }

        let res = req.send().await.map_err(|e| format!("Download request error: {}", e))?;
        let status = res.status();

        let (append_mode, total_bytes) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
            let total = res
                .headers()
                .get("Content-Range")
                .and_then(|h| h.to_str().ok())
                .and_then(|s| s.split('/').last())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(existing_bytes);
            (true, total)
        } else if status.is_success() {
            let total = res.content_length().unwrap_or(0);
            (false, total)
        } else {
            return Err(format!("Download failed with status: {}", status));
        };

        let mut file = if append_mode {
            fs::OpenOptions::new()
                .write(true)
                .append(true)
                .open(dest_path)
                .map_err(|e| format!("Cannot open file for appending: {}", e))?
        } else {
            existing_bytes = 0;
            fs::File::create(dest_path).map_err(|e| format!("Cannot create file: {}", e))?
        };

        use futures_util::StreamExt;
        let mut stream = res.bytes_stream();
        let mut downloaded = existing_bytes;

        while let Some(chunk_res) = stream.next().await {
            let chunk = chunk_res.map_err(|e| format!("Stream read error: {}", e))?;
            file.write_all(&chunk)
                .map_err(|e| format!("Write chunk error: {}", e))?;
            downloaded += chunk.len() as u64;
            progress_callback(downloaded, total_bytes);
        }

        file.flush().map_err(|e| format!("Flush error: {}", e))?;

        if let Some(expected_hash) = expected_sha256 {
            Self::verify_sha256(dest_path, expected_hash)?;
        }

        Ok(())
    }

    pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<(), String> {
        let mut file = File::open(path).map_err(|e| format!("Failed to open for sha256: {}", e))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 65536];

        loop {
            let n = file.read(&mut buffer).map_err(|e| format!("Read error: {}", e))?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }

        let result = hasher.finalize();
        let calculated = format!("{:x}", result);

        if calculated.eq_ignore_ascii_case(expected_hex) {
            Ok(())
        } else {
            Err(format!(
                "SHA256 mismatch! Expected {}, calculated {}",
                expected_hex, calculated
            ))
        }
    }
}

// ============================================================================
// 5. Hugging Face Discovery & Search Client
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModel {
    pub id: String,
    pub author: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub tags: Vec<String>,
    pub last_modified: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelFile {
    pub path: String,
    pub size: Option<u64>,
}

pub struct HuggingFaceClient;

impl HuggingFaceClient {
    pub async fn search_models(query: &str, limit: usize) -> Result<Vec<HfModel>, String> {
        let client = reqwest::Client::builder()
            .user_agent("lm-studio-rust/0.1.0")
            .build()
            .map_err(|e| e.to_string())?;

        let encoded_q = query.trim().replace(' ', "+");
        let url = format!(
            "https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit={}",
            encoded_q, limit
        );

        let res = client.get(&url).send().await.map_err(|e| format!("HF request error: {}", e))?;
        if !res.status().is_success() {
            return Err(format!("HF API returned {}", res.status()));
        }

        let list: Vec<serde_json::Value> = res.json().await.map_err(|e| format!("HF json error: {}", e))?;
        let mut models = Vec::new();

        for item in list {
            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                let downloads = item.get("downloads").and_then(|v| v.as_u64());
                let likes = item.get("likes").and_then(|v| v.as_u64());
                let tags = item.get("tags")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|t| t.as_str().map(|s| s.to_string())).collect())
                    .unwrap_or_default();
                let last_modified = item.get("lastModified").and_then(|v| v.as_str()).map(|s| s.to_string());
                let author = id.split('/').next().map(|s| s.to_string());

                models.push(HfModel {
                    id: id.to_string(),
                    author,
                    downloads,
                    likes,
                    tags,
                    last_modified,
                });
            }
        }

        Ok(models)
    }

    pub async fn list_gguf_files(model_id: &str) -> Result<Vec<HfModelFile>, String> {
        let client = reqwest::Client::builder()
            .user_agent("lm-studio-rust/0.1.0")
            .build()
            .map_err(|e| e.to_string())?;

        let url = format!("https://huggingface.co/api/models/{}/tree/main", model_id);
        let res = client.get(&url).send().await.map_err(|e| format!("HF tree error: {}", e))?;
        if !res.status().is_success() {
            return Err(format!("HF tree API returned {}", res.status()));
        }

        let list: Vec<serde_json::Value> = res.json().await.map_err(|e| format!("HF json error: {}", e))?;
        let mut files = Vec::new();

        for item in list {
            if let Some(path) = item.get("path").and_then(|v| v.as_str()) {
                if path.to_lowercase().ends_with(".gguf") {
                    let size = item.get("size").and_then(|v| v.as_u64());
                    files.push(HfModelFile {
                        path: path.to_string(),
                        size,
                    });
                }
            }
        }

        Ok(files)
    }

    pub fn download_url(model_id: &str, file_path: &str) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", model_id, file_path)
    }
}

