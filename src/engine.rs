use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use llama_cpp_2::model::{LlamaModel, LlamaChatMessage, LlamaChatTemplate};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::llama_batch::LlamaBatch;

pub struct MmapGgufModel {
    pub path: PathBuf,
    pub mmap: Arc<Mmap>,
    pub size: usize,
}

impl MmapGgufModel {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("Failed to open model file: {}", e))?;
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| format!("Failed to mmap model file: {}", e))?
        };
        let size = mmap.len();
        Ok(Self {
            path: path.to_path_buf(),
            mmap: Arc::new(mmap),
            size,
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.mmap[..]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeParams {
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    pub n_threads: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub max_tokens: usize,
    pub seed: Option<u64>,
    pub flash_attention: bool,
    pub system_prompt: Option<String>,
    pub stop_tokens: Vec<String>,
    pub json_schema: Option<serde_json::Value>,
    pub mmproj_path: Option<PathBuf>,
}

impl Default for RuntimeParams {
    fn default() -> Self {
        Self {
            n_ctx: 512,
            n_gpu_layers: 0,
            n_threads: 4,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            min_p: None,
            repetition_penalty: Some(1.1),
            frequency_penalty: None,
            presence_penalty: None,
            max_tokens: 512,
            seed: None,
            flash_attention: false,
            system_prompt: None,
            stop_tokens: vec![
                "<|im_end|>".to_string(),
                "<|endoftext|>".to_string(),
                "</s>".to_string(),
            ],
            json_schema: None,
            mmproj_path: None,
        }
    }
}

pub struct InferenceEngine {
    pub model_path: PathBuf,
    pub backend: Arc<LlamaBackend>,
    pub model: Arc<LlamaModel>,
    pub mmap_model: MmapGgufModel,
    pub runtime_params: RuntimeParams,
    pub is_loaded: bool,
    pub active_cancel: Arc<AtomicBool>,
}

impl InferenceEngine {
    pub fn load(model_path: &Path, params: RuntimeParams) -> Result<Self, String> {
        let backend = Arc::new(LlamaBackend::init().map_err(|e| format!("Failed to init llama backend: {:?}", e))?);
        let model_params = LlamaModelParams::default().with_n_gpu_layers(params.n_gpu_layers as u32);
        let model = Arc::new(LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| format!("Failed to load model from path {:?}: {:?}", model_path, e))?);
        let mmap_model = MmapGgufModel::open(model_path)?;

        Ok(Self {
            model_path: model_path.to_path_buf(),
            backend,
            model,
            mmap_model,
            runtime_params: params,
            is_loaded: true,
            active_cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn cancel_active(&self) {
        self.active_cancel.store(true, Ordering::Relaxed);
    }

    pub fn apply_chat_template(&self, messages: &[(&str, &str)]) -> Result<String, String> {
        let chat_messages: Result<Vec<LlamaChatMessage>, _> = messages
            .iter()
            .map(|(role, content)| LlamaChatMessage::new(role.to_string(), content.to_string()))
            .collect();

        if let Ok(msgs) = chat_messages {
            let tmpl = self.model
                .chat_template(None)
                .or_else(|_| LlamaChatTemplate::new("chatml").map_err(|_| ()));

            if let Ok(t) = tmpl {
                if let Ok(rendered) = self.model.apply_chat_template(&t, &msgs, true) {
                    return Ok(rendered);
                }
            }
        }

        // Fallback format
        let mut p = String::new();
        for (role, content) in messages {
            p.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content.trim()));
        }
        p.push_str("<|im_start|>assistant\n");
        Ok(p)
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        params: &RuntimeParams,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> mpsc::Receiver<String> {
        let (tx, rx) = mpsc::channel(64);
        let prompt_owned = prompt.to_string();
        let max_tokens = params.max_tokens;
        let backend = Arc::clone(&self.backend);
        let model = Arc::clone(&self.model);
        let n_ctx = params.n_ctx;
        let n_threads_requested = params.n_threads;
        let temperature = params.temperature;
        let top_p = params.top_p;
        let top_k = params.top_k;
        let min_p = params.min_p;
        let repetition_penalty = params.repetition_penalty;
        let frequency_penalty = params.frequency_penalty;
        let presence_penalty = params.presence_penalty;
        let seed = params.seed;
        let flash_attention = params.flash_attention;
        let stop_tokens = params.stop_tokens.clone();
        let internal_cancel = Arc::clone(&self.active_cancel);
        internal_cancel.store(false, Ordering::Relaxed);

        tokio::task::spawn_blocking(move || {
            let vocab = model.vocab();
            let n_threads = if n_threads_requested > 0 {
                n_threads_requested
            } else {
                std::thread::available_parallelism()
                    .map(|n| (n.get() / 2).max(1) as i32)
                    .unwrap_or(4)
            };
            let n_threads_batch = std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(8);

            let mut ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx))
                .with_n_threads(n_threads)
                .with_n_threads_batch(n_threads_batch);

            if flash_attention {
                ctx_params = ctx_params.with_flash_attention_policy(true.into());
            }

            let mut ctx = match model.new_context(&backend, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Failed to create context: {:?}", e);
                    return;
                }
            };

            // Format prompt into ChatML format if it doesn't already contain control tokens
            let formatted_prompt = if !prompt_owned.contains("<|im_start|>") && !prompt_owned.contains("<|start_header_id|>") {
                format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt_owned)
            } else {
                prompt_owned.clone()
            };

            let starts_with_special = formatted_prompt.starts_with("<|begin_of_text|>")
                || formatted_prompt.starts_with("<s>")
                || formatted_prompt.starts_with("<|im_start|>");
            let add_special = !starts_with_special;

            let tokens = vocab.tokenize(formatted_prompt.as_bytes(), add_special, true);
            if tokens.is_empty() {
                return;
            }

            let prompt_len = tokens.len();
            let mut batch = LlamaBatch::new(prompt_len.max(512), 1);

            for (i, &token) in tokens.iter().enumerate() {
                let is_last = i == tokens.len() - 1;
                let _ = batch.add(token, i as i32, &[0], is_last);
            }

            let mut samplers: Vec<LlamaSampler> = vec![
                LlamaSampler::top_k(top_k),
                LlamaSampler::top_p(top_p, 1),
                LlamaSampler::temp(temperature),
            ];

            if let Some(mp) = min_p {
                samplers.push(LlamaSampler::min_p(mp, 1));
            }

            if repetition_penalty.is_some() || frequency_penalty.is_some() || presence_penalty.is_some() {
                let n_vocab = model.n_vocab();
                let penalty_last_n = 64;
                let penalty_repeat = repetition_penalty.unwrap_or(1.1);
                let penalty_freq = frequency_penalty.unwrap_or(0.0);
                let penalty_present = presence_penalty.unwrap_or(0.0);
                samplers.push(LlamaSampler::penalties(
                    n_vocab,
                    penalty_last_n,
                    penalty_repeat,
                    penalty_freq,
                    penalty_present,
                ));
            }

            let seed_val = seed.unwrap_or(42) as u32;
            samplers.push(LlamaSampler::dist(seed_val));

            let mut sampler = LlamaSampler::chain_simple(samplers);
            sampler.accept_many(&tokens);

            if let Err(e) = ctx.decode(&mut batch) {
                eprintln!("Failed prompt decode: {:?}", e);
                return;
            }

            let mut n_gen = 0;
            let mut current_batch = batch;

            while n_gen < max_tokens {
                if internal_cancel.load(Ordering::Relaxed) {
                    break;
                }
                if let Some(ref external_cancel) = cancel_flag {
                    if external_cancel.load(Ordering::Relaxed) {
                        break;
                    }
                }

                let new_token_id = sampler.sample(&ctx, current_batch.n_tokens() - 1);
                sampler.accept(new_token_id);

                if vocab.is_eog(new_token_id) {
                    break;
                }

                let piece_bytes = vocab.token_to_piece(new_token_id, false, None);
                let piece = String::from_utf8_lossy(&piece_bytes).to_string();

                if stop_tokens.iter().any(|s| piece.contains(s)) {
                    break;
                }

                if tx.blocking_send(piece).is_err() {
                    break;
                }

                current_batch = LlamaBatch::new(512, 1);
                let _ = current_batch.add(new_token_id, prompt_len as i32 + n_gen as i32, &[0], true);
                n_gen += 1;

                if let Err(e) = ctx.decode(&mut current_batch) {
                    eprintln!("Failed token decode at step {}: {:?}", n_gen, e);
                    break;
                }
            }
        });

        rx
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let dim = 384;
        let mut vec = vec![0.0f32; dim];
        let bytes = text.as_bytes();

        let mut hash = 2166136261u32;
        for (i, &b) in bytes.iter().enumerate() {
            hash = hash ^ (b as u32);
            hash = hash.wrapping_mul(16777619);
            vec[i % dim] += (b as f32 / 255.0) * (if (hash & 1) == 0 { 1.0 } else { -1.0 });
        }

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in vec.iter_mut() {
            *x /= norm;
        }

        vec
    }
}

pub struct GbnfGrammar;

impl GbnfGrammar {
    pub fn schema_to_gbnf(_schema: &serde_json::Value) -> String {
        let mut grammar = String::from("root ::= object\n");
        grammar.push_str("object ::= \"{\" ws \"}\"\n");
        grammar.push_str("string ::= \"\\\"\" [^\"]* \"\\\"\"\n");
        grammar.push_str("number ::= [0-9]+\n");
        grammar.push_str("ws ::= [ \\t\\n]*\n");
        // ponytail: minimal GBNF schema converter; add recursive AST rules when complex nested schemas are required.
        grammar
    }
}
