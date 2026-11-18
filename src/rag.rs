use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};

use crate::engine::InferenceEngine;

// ============================================================================
// 1. Document Chunk & Vector Store
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: usize,
    pub source_file: String,
    pub chunk_index: usize,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorStore {
    pub chunks: Vec<DocumentChunk>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    pub fn add_chunk(&mut self, chunk: DocumentChunk) {
        self.chunks.push(chunk);
    }

    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(DocumentChunk, f32)> {
        let mut scored: Vec<(DocumentChunk, f32)> = self
            .chunks
            .iter()
            .map(|chunk| {
                let score = Self::cosine_similarity(&chunk.embedding, query_embedding);
                (chunk.clone(), score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    pub fn build_augmented_prompt(&self, query: &str, query_embedding: &[f32], top_k: usize) -> String {
        let results = self.search(query_embedding, top_k);
        if results.is_empty() {
            return query.to_string();
        }

        let mut prompt = String::new();
        prompt.push_str("You are an AI assistant. Answer the user's question using the relevant context retrieved from local documents.\n\n");
        prompt.push_str("--- BEGIN CONTEXT ---\n");
        for (i, (chunk, score)) in results.iter().enumerate() {
            prompt.push_str(&format!(
                "[Snippet {} from {} (similarity: {:.2})]:\n{}\n\n",
                i + 1,
                chunk.source_file,
                score,
                chunk.text.trim()
            ));
        }
        prompt.push_str("--- END CONTEXT ---\n\n");
        prompt.push_str(&format!("User Query: {}\n\nAnswer concisely based on the context above:", query));
        prompt
    }
}

// ============================================================================
// 2. Document Parsers
// ============================================================================

pub struct DocumentParser;

impl DocumentParser {
    pub fn parse_file(path: &Path) -> Result<String, String> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        let bytes = fs::read(path).map_err(|e| format!("Failed to read file {}: {}", path.display(), e))?;

        match ext.as_str() {
            "pdf" => Self::parse_pdf(&bytes),
            "csv" => Self::parse_csv(&bytes),
            "md" | "markdown" => Self::parse_markdown(&bytes),
            _ => {
                // Source code or plain text
                Ok(String::from_utf8_lossy(&bytes).to_string())
            }
        }
    }

    pub fn parse_pdf(bytes: &[u8]) -> Result<String, String> {
        // High-efficiency zero-dependency PDF text stream extractor
        let mut extracted = String::new();
        let content = String::from_utf8_lossy(bytes);

        // Scan for text operator blocks BT ... ET
        let mut in_text_block = false;
        let mut text_buf = String::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed == "BT" {
                in_text_block = true;
                text_buf.clear();
            } else if trimmed == "ET" {
                in_text_block = false;
                if !text_buf.is_empty() {
                    extracted.push_str(&text_buf);
                    extracted.push('\n');
                }
            } else if in_text_block {
                // Extract strings inside parentheses (Text) Tj or [(T)(e)(x)(t)] TJ
                let mut chars = trimmed.chars().peekable();
                while let Some(c) = chars.next() {
                    if c == '(' {
                        let mut str_val = String::new();
                        let mut escaped = false;
                        for sc in chars.by_ref() {
                            if escaped {
                                str_val.push(sc);
                                escaped = false;
                            } else if sc == '\\' {
                                escaped = true;
                            } else if sc == ')' {
                                break;
                            } else {
                                str_val.push(sc);
                            }
                        }
                        text_buf.push_str(&str_val);
                        text_buf.push(' ');
                    }
                }
            }
        }

        if extracted.trim().is_empty() {
            // Fallback: extract printable ASCII strings
            for chunk in content.split(|c: char| c.is_control() && c != '\n' && c != '\t') {
                let trimmed = chunk.trim();
                if trimmed.len() > 20 && trimmed.contains(' ') {
                    extracted.push_str(trimmed);
                    extracted.push('\n');
                }
            }
        }

        Ok(extracted)
    }

    pub fn parse_csv(bytes: &[u8]) -> Result<String, String> {
        let content = String::from_utf8_lossy(bytes);
        let mut lines = content.lines();
        let headers: Vec<&str> = match lines.next() {
            Some(h) => h.split(',').map(|s| s.trim()).collect(),
            None => return Ok(String::new()),
        };

        let mut out = String::new();
        for (row_idx, line) in lines.enumerate() {
            let cells: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            let mut row_str = format!("Row {}: ", row_idx + 1);
            for (i, cell) in cells.iter().enumerate() {
                let header = headers.get(i).copied().unwrap_or("Column");
                row_str.push_str(&format!("{}: {}; ", header, cell));
            }
            out.push_str(&row_str);
            out.push('\n');
        }

        Ok(out)
    }

    pub fn parse_markdown(bytes: &[u8]) -> Result<String, String> {
        let content = String::from_utf8_lossy(bytes);
        let mut clean = String::new();

        for line in content.lines() {
            let trimmed = line.trim();
            // Preserve headers, strip code fences if empty
            if trimmed.starts_with("```") {
                continue;
            }
            clean.push_str(line);
            clean.push('\n');
        }

        Ok(clean)
    }
}

// ============================================================================
// 3. Chunking Engine
// ============================================================================

pub struct TextChunker;

impl TextChunker {
    // 512–1,024 tokens with 10% overlap
    // Approximation: ~4 characters per token
    pub const DEFAULT_CHUNK_SIZE_TOKENS: usize = 512;
    pub const DEFAULT_OVERLAP_PERCENT: usize = 10;

    pub fn chunk(
        text: &str,
        chunk_size_tokens: usize,
        overlap_percent: usize,
    ) -> Vec<String> {
        let chunk_size = chunk_size_tokens.max(64) * 4; // in bytes/chars
        let overlap_size = (chunk_size * overlap_percent.min(50)) / 100;
        let step_size = (chunk_size.saturating_sub(overlap_size)).max(1);

        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let total_chars = chars.len();

        if total_chars <= chunk_size {
            chunks.push(text.to_string());
            return chunks;
        }

        let mut start = 0;
        while start < total_chars {
            let end = (start + chunk_size).min(total_chars);
            // Snap to nearest sentence or word boundary if possible
            let mut actual_end = end;
            if actual_end < total_chars {
                for i in (end.saturating_sub(60)..end).rev() {
                    if chars[i] == '.' || chars[i] == '\n' || chars[i] == ' ' {
                        actual_end = i + 1;
                        break;
                    }
                }
            }

            let slice: String = chars[start..actual_end].iter().collect();
            if !slice.trim().is_empty() {
                chunks.push(slice);
            }

            if actual_end >= total_chars {
                break;
            }
            start += step_size;
        }

        chunks
    }
}

// ============================================================================
// 4. Local RAG Pipeline Coordinator
// ============================================================================

pub struct RagPipeline;

impl RagPipeline {
    pub fn ingest_file(
        store: &mut VectorStore,
        engine: &InferenceEngine,
        file_path: &Path,
    ) -> Result<usize, String> {
        let text = DocumentParser::parse_file(file_path)?;
        let chunks = TextChunker::chunk(&text, TextChunker::DEFAULT_CHUNK_SIZE_TOKENS, TextChunker::DEFAULT_OVERLAP_PERCENT);
        let file_name = file_path.file_name().unwrap_or_default().to_string_lossy().to_string();

        let count = chunks.len();
        for (idx, chunk_text) in chunks.into_iter().enumerate() {
            let embedding = engine.embed(&chunk_text);
            let chunk = DocumentChunk {
                id: store.chunks.len() + 1,
                source_file: file_name.clone(),
                chunk_index: idx + 1,
                text: chunk_text,
                embedding,
            };
            store.add_chunk(chunk);
        }

        Ok(count)
    }
}
