# Generalizing the LLM Layer

## Goal

The current project is tied to Ollama for inference. The goal is to make the analysis pipeline provider-agnostic so the same document extraction, retrieval, agentic workflow, and prompt-based workflow can run with different LLM providers.

Ollama should remain supported, but it should become one provider behind a common interface.

## Current Coupling

The main coupling is in the backend:

- `customnerd-backend/ollama_executions.py` contains the Ollama-specific client and retry logic.
- `customnerd-backend/helper_functions.py` imports `_retryable_ollama_call` directly.
- Agentic analysis, prompt-based analysis, and JSON-response calls all call the Ollama wrapper directly.
- Configuration is currently Ollama-specific through variables such as `OLLAMA_MODEL` and `OLLAMA_BASE_URL`.

## Proposed Architecture

Introduce a provider-agnostic LLM layer:

helper_functions.py
        |
        v
retryable_llm_call(...)
        |
        v
LLMProvider interface
        |
        |-- OllamaProvider
        |-- OpenAICompatibleProvider
        |-- AnthropicProvider
        |-- MockProvider

## Provider Interface

Each provider should expose the same method:

generate(
    messages: list[dict],
    temperature: float = 0.3,
    top_p: float = 1.0,
    response_format: dict | None = None,
) -> str

The analysis pipeline should not know which provider is being used.

## Configuration

Add provider-neutral environment variables:

LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
LLM_BASE_URL=http://localhost:11434
LLM_API_KEY=

EXECUTION_STRATEGY=agentic

Provider-specific variables can still be supported for backward compatibility:

OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

## Implementation Plan

1. Add a new `llm_providers.py` file.
2. Define a base `LLMProvider` interface.
3. Move Ollama-specific call logic behind `OllamaProvider`.
4. Add `OpenAICompatibleProvider` for OpenAI and OpenAI-compatible APIs.
5. Optionally add `AnthropicProvider`.
6. Add `get_llm_provider()` factory function that reads `LLM_PROVIDER`.
7. Replace direct calls to `_retryable_ollama_call(...)` with `retryable_llm_call(...)`.
8. Keep the existing analysis pipeline unchanged.
9. Keep backward compatibility so the current Ollama setup still works.
10. Add a mock provider later for tests.

## Main Benefit

This keeps the core system stable while making the LLM backend swappable. The project can continue running locally with Ollama, but can also support OpenAI-compatible APIs, Anthropic, or other local/remote model endpoints through configuration.