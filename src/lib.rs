pub mod discovery;
pub mod engine;
pub mod daemon;
pub mod server;
pub mod rag;
pub mod mcp;
pub mod ui;
pub mod logger;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logger_ring_buffer() {
        let logger = logger::Logger::new(5);
        logger.log(logger::LogLevel::Info, "test", "Hello log");
        let entries = logger.get_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "Hello log");
        assert_eq!(entries[0].target, "test");
    }

    #[test]
    fn test_gguf_header_and_memory_estimator() {
        // Construct valid minimal synthetic GGUF v3 header
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x46554747u32.to_le_bytes()); // 'GGUF'
        bytes.extend_from_slice(&3u32.to_le_bytes());          // Version 3
        bytes.extend_from_slice(&100u64.to_le_bytes());        // Tensor count
        bytes.extend_from_slice(&0u64.to_le_bytes());          // 0 metadata KV pairs

        let meta = discovery::GgufHeaderParser::parse(&bytes).expect("Should parse GGUF header");
        assert_eq!(meta.magic, 0x46554747);
        assert_eq!(meta.version, 3);
        assert_eq!(meta.tensor_count, 100);

        let telemetry = discovery::HardwareTelemetry {
            total_ram_bytes: 8 * 1024 * 1024 * 1024,
            available_ram_bytes: 4 * 1024 * 1024 * 1024,
            vram_total_bytes: 4 * 1024 * 1024 * 1024,
            vram_free_bytes: 3 * 1024 * 1024 * 1024,
            is_integrated_gpu: true,
            has_avx2: true,
            has_vulkan: true,
            cpu_cores: 8,
        };

        // Estimate for 2 GB model file with 3 GB VRAM -> CompatibleFullOffload
        let est_full = discovery::MemoryBudgetEstimator::estimate(&meta, 2 * 1024 * 1024 * 1024, &telemetry, Some(2048));
        assert!(est_full.total_estimated_ram_bytes > 2 * 1024 * 1024 * 1024);
        assert_eq!(est_full.classification, discovery::FitClassification::CompatibleFullOffload);

        // With limited VRAM (512 MB) -> PartialOffload
        let mut telemetry_limited = telemetry.clone();
        telemetry_limited.vram_free_bytes = 512 * 1024 * 1024;
        let est_partial = discovery::MemoryBudgetEstimator::estimate(&meta, 2 * 1024 * 1024 * 1024, &telemetry_limited, Some(2048));
        assert_eq!(est_partial.classification, discovery::FitClassification::PartialOffload);
    }

    #[test]
    fn test_gguf_header_rejects_truncated_input() {
        assert!(discovery::GgufHeaderParser::parse(&[]).is_err());
        assert!(discovery::GgufHeaderParser::parse(&[0; 23]).is_err());
    }

    #[test]
    fn test_gbnf_grammar_from_json_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            }
        });

        let gbnf = engine::GbnfGrammar::schema_to_gbnf(&schema);
        assert!(gbnf.contains("root ::="));
        assert!(gbnf.contains("string"));
        assert!(gbnf.contains("number"));
    }

    #[test]
    fn test_rag_chunking_and_cosine_similarity() {
        let text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four.";
        let chunks = rag::TextChunker::chunk(text, 5, 20);
        assert!(!chunks.is_empty());

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let v3 = vec![0.0, 1.0, 0.0];

        let sim_same = rag::VectorStore::cosine_similarity(&v1, &v2);
        let sim_ortho = rag::VectorStore::cosine_similarity(&v1, &v3);

        assert!((sim_same - 1.0).abs() < 1e-4);
        assert!(sim_ortho.abs() < 1e-4);
    }

    #[test]
    fn test_mcp_jinja_chat_template() {
        let renderer = mcp::ChatTemplateRenderer::new(None);
        let messages = vec![
            serde_json::json!({ "role": "system", "content": "You are a helpful assistant." }),
            serde_json::json!({ "role": "user", "content": "What is the weather?" }),
        ];

        let tools = vec![mcp::McpTool {
            name: "get_weather".to_string(),
            description: "Get weather for location".to_string(),
            input_schema: serde_json::json!({ "type": "object" }),
        }];

        let rendered = renderer.render_with_tools(&messages, Some(&tools)).expect("Should render");
        assert!(rendered.contains("<|im_start|>system"));
        assert!(rendered.contains("get_weather"));
        assert!(rendered.contains("<|im_start|>user"));
    }
}

