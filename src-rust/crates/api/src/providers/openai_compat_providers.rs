// providers/openai_compat_providers.rs — Factory functions for all
// OpenAI-compatible provider instances.
//
// Each function constructs a pre-configured [`OpenAiCompatProvider`] for a
// specific service.  API keys are read from environment variables; if the
// variable is absent or empty the provider is still constructed but
// `health_check()` will return `ProviderStatus::Unavailable`.

use claurst_core::provider_id::ProviderId;

use super::openai_compat::{OpenAiCompatProvider, ProviderQuirks};

// ---------------------------------------------------------------------------
// Local / self-hosted providers (no API key required)
// ---------------------------------------------------------------------------

/// Ollama — local inference server.
/// Reads `OLLAMA_HOST` for the base URL; defaults to `http://localhost:11434`.
pub fn ollama() -> OpenAiCompatProvider {
    let host = std::env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    let base_url = format!("{}/v1", host.trim_end_matches('/'));
    OpenAiCompatProvider::new(ProviderId::OLLAMA, "Ollama", base_url).with_quirks(
        ProviderQuirks {
            overflow_patterns: vec![
                "prompt too long".to_string(),
                "exceeded.*context length".to_string(),
            ],
            ..Default::default()
        },
    )
}

/// LM Studio — local OpenAI-compatible server.
/// Reads `LM_STUDIO_HOST` for the base URL; defaults to `http://localhost:1234`.
pub fn lm_studio() -> OpenAiCompatProvider {
    let host = std::env::var("LM_STUDIO_HOST")
        .unwrap_or_else(|_| "http://localhost:1234".to_string());
    let base_url = format!("{}/v1", host.trim_end_matches('/'));
    OpenAiCompatProvider::new(ProviderId::LM_STUDIO, "LM Studio", base_url).with_quirks(
        ProviderQuirks {
            overflow_patterns: vec![
                "greater than the context length".to_string(),
            ],
            ..Default::default()
        },
    )
}

/// llama.cpp — lightweight C++ inference server.
/// Reads `LLAMA_CPP_HOST` for the base URL; defaults to `http://localhost:8080`.
pub fn llama_cpp() -> OpenAiCompatProvider {
    let host = std::env::var("LLAMA_CPP_HOST")
        .unwrap_or_else(|_| "http://localhost:8080".to_string());
    let base_url = format!("{}/v1", host.trim_end_matches('/'));
    OpenAiCompatProvider::new(ProviderId::LLAMA_CPP, "llama.cpp", base_url).with_quirks(
        ProviderQuirks {
            overflow_patterns: vec![
                "exceeds the available context size".to_string(),
            ],
            ..Default::default()
        },
    )
}

// ---------------------------------------------------------------------------
// Remote / cloud providers (API key required)
// ---------------------------------------------------------------------------

/// DeepSeek — supports reasoning output via `reasoning_content` field.
/// Reads `DEEPSEEK_API_KEY`.
pub fn deepseek() -> OpenAiCompatProvider {
    let key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::DEEPSEEK,
        "DeepSeek",
        "https://api.deepseek.com/v1",
    )
    .with_api_key(key)
    .with_quirks(ProviderQuirks {
        reasoning_field: Some("reasoning_content".to_string()),
        overflow_patterns: vec!["maximum context length is".to_string()],
        include_usage_in_stream: true,
        ..Default::default()
    })
}

/// Groq — fast inference cloud.  Reads `GROQ_API_KEY`.
pub fn groq() -> OpenAiCompatProvider {
    let key = std::env::var("GROQ_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(ProviderId::GROQ, "Groq", "https://api.groq.com/openai/v1")
        .with_api_key(key)
        .with_quirks(ProviderQuirks {
            overflow_patterns: vec![
                "reduce the length of the messages".to_string(),
            ],
            include_usage_in_stream: true,
            ..Default::default()
        })
}

/// xAI (Grok).  Reads `XAI_API_KEY`.
pub fn xai() -> OpenAiCompatProvider {
    let key = std::env::var("XAI_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(ProviderId::XAI, "xAI (Grok)", "https://api.x.ai/v1")
        .with_api_key(key)
        .with_quirks(ProviderQuirks {
            overflow_patterns: vec!["maximum prompt length is".to_string()],
            ..Default::default()
        })
}

/// DeepInfra — hosted open-weight models.  Reads `DEEPINFRA_API_KEY`.
pub fn deepinfra() -> OpenAiCompatProvider {
    let key = std::env::var("DEEPINFRA_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::DEEPINFRA,
        "DeepInfra",
        "https://api.deepinfra.com/v1/openai",
    )
    .with_api_key(key)
}

/// Cerebras — wafer-scale inference.  Reads `CEREBRAS_API_KEY`.
pub fn cerebras() -> OpenAiCompatProvider {
    let key = std::env::var("CEREBRAS_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::CEREBRAS,
        "Cerebras",
        "https://api.cerebras.ai/v1",
    )
    .with_api_key(key)
}

/// Together AI — hosted open-source models.  Reads `TOGETHER_API_KEY`.
pub fn together_ai() -> OpenAiCompatProvider {
    let key = std::env::var("TOGETHER_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::TOGETHER_AI,
        "Together AI",
        "https://api.together.xyz/v1",
    )
    .with_api_key(key)
}

/// Perplexity — search-augmented LLM API.  Reads `PERPLEXITY_API_KEY`.
pub fn perplexity() -> OpenAiCompatProvider {
    let key = std::env::var("PERPLEXITY_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::PERPLEXITY,
        "Perplexity",
        "https://api.perplexity.ai",
    )
    .with_api_key(key)
}

/// Venice AI — privacy-focused inference.  Reads `VENICE_API_KEY`.
pub fn venice() -> OpenAiCompatProvider {
    let key = std::env::var("VENICE_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::VENICE,
        "Venice AI",
        "https://api.venice.ai/api/v1",
    )
    .with_api_key(key)
}

/// Qwen / Alibaba DashScope.  Reads `DASHSCOPE_API_KEY`.
/// Uses a default temperature of 0.55 as recommended by Alibaba's docs.
pub fn qwen() -> OpenAiCompatProvider {
    let key = std::env::var("DASHSCOPE_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        "qwen",
        "Qwen (Alibaba)",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    .with_api_key(key)
    .with_quirks(ProviderQuirks {
        default_temperature: Some(0.55),
        ..Default::default()
    })
}

/// Mistral AI — Reads `MISTRAL_API_KEY`.
/// Uses OpenAI-compatible format with Mistral-specific quirks:
///   - Tool call IDs must be alphanumeric only, zero-padded to exactly 9 chars.
///   - An assistant "Done." turn is inserted between tool→user message transitions.
pub fn mistral() -> OpenAiCompatProvider {
    let key = std::env::var("MISTRAL_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::MISTRAL,
        "Mistral AI",
        "https://api.mistral.ai/v1",
    )
    .with_api_key(key)
    .with_quirks(ProviderQuirks {
        tool_id_max_len: Some(9),
        tool_id_alphanumeric_only: true,
        fix_tool_user_sequence: true,
        overflow_patterns: vec!["too large for model with".to_string()],
        ..Default::default()
    })
}

/// OpenRouter — unified API gateway to many models.  Reads `OPENROUTER_API_KEY`.
pub fn openrouter() -> OpenAiCompatProvider {
    let key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::OPENROUTER,
        "OpenRouter",
        "https://openrouter.ai/api/v1",
    )
    .with_api_key(key)
    .with_header("HTTP-Referer", "https://claurst.ai/")
    .with_header("X-Title", "Claurst")
    .with_quirks(ProviderQuirks {
        include_usage_in_stream: true,
        ..Default::default()
    })
}
