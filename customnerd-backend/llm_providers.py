from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    NotFoundError,
)

MAX_RETRIES = 3
BACKOFF_SECS = 2

logging.basicConfig(level=logging.INFO)


class LLMProvider(ABC):
    """Common interface for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 1.0,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        raise NotImplementedError


class OpenAICompatibleProvider(LLMProvider):
    """
    Provider for OpenAI-compatible chat completion APIs.

    This works for:
    - OpenAI
    - Ollama's OpenAI-compatible endpoint
    - Local servers exposing an OpenAI-compatible /v1 API
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider_name: str = "openai_compatible",
    ) -> None:
        self.model = model
        self.provider_name = provider_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "unused",
        )

    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 1.0,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }

        if response_format:
            kwargs["response_format"] = response_format

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                logging.warning(
                    "[%s attempt %s/%s] %s",
                    self.provider_name,
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )
                time.sleep(BACKOFF_SECS * (2**attempt))
            except NotFoundError:
                logging.error(
                    "Model '%s' not found for provider '%s'.",
                    self.model,
                    self.provider_name,
                )
                break
            except Exception as exc:
                logging.exception("[%s fatal] %s", self.provider_name, exc)
                break

        return ""


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama provider using Ollama's OpenAI-compatible API."""

    def __init__(self) -> None:
        model = (
            os.getenv("LLM_MODEL")
            or os.getenv("OLLAMA_MODEL")
            or "llama3.2"
        ).strip('"').strip()

        base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).strip('"').strip()

        if not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
            base_url = base_url.rstrip("/") + "/v1/"

        super().__init__(
            model=model,
            base_url=base_url,
            api_key="ollama",
            provider_name="ollama",
        )


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI hosted provider."""

    def __init__(self) -> None:
        model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

        super().__init__(
            model=model,
            base_url=os.getenv("LLM_BASE_URL") or None,
            api_key=api_key,
            provider_name="openai",
        )


class MockProvider(LLMProvider):
    """Simple provider for tests and dry runs."""

    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        top_p: float = 1.0,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        if response_format:
            return '{"summary":"mock response","overall_risk":"unclear","findings":[],"gaps":[],"notes":["mock provider used"]}'
        return "Mock LLM response."


def get_llm_provider() -> LLMProvider:
    provider = (os.getenv("LLM_PROVIDER") or os.getenv("LLM") or "ollama").lower()

    if provider == "ollama":
        return OllamaProvider()

    if provider in {"openai", "openai_compatible", "compatible"}:
        return OpenAIProvider()

    if provider == "mock":
        return MockProvider()

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


_provider: Optional[LLMProvider] = None


def get_cached_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = get_llm_provider()
    return _provider


def reinitialize_llm_provider() -> None:
    global _provider
    _provider = get_llm_provider()


def retryable_llm_call(
    *,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    top_p: float = 1.0,
    response_format: Optional[Dict[str, str]] = None,
) -> str:
    return get_cached_provider().generate(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        response_format=response_format,
    )