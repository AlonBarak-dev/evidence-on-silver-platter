"""Additional LLM client(s), alongside FigBrowse's own AzureOpenAIClient.

``figbrowse.llm.AzureOpenAIClient`` is generic Chat-Completions-over-httpx
with structured-output retry logic (``generate_structured``) that never
touches anything Azure-specific — only ``generate()`` builds the request
URL/headers. Cerebras exposes an OpenAI-compatible Chat Completions
endpoint, so ``CerebrasClient`` subclasses it and only overrides the
endpoint/auth/model resolution, inheriting the retry/caching-compatible
``generate``/``generate_structured`` contract unchanged. FigBrowse itself is
never modified — this is purely additive, in stopBrowse's own package.
"""

from __future__ import annotations

import os

from figbrowse.llm import AzureOpenAIClient

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


class CerebrasClient(AzureOpenAIClient):
    """Cerebras Cloud inference — OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        *,
        api_key_env: str = "CEREBRAS_API_KEY",
        model_env: str = "CEREBRAS_MODEL",
        default_model: str = "gpt-oss-120b",
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        max_retries: int = 5,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(
            base_url_env="__unused_cerebras_base_url__",  # overridden below
            api_key_env=api_key_env,
            model_env=model_env,
            default_model=default_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            timeout_s=timeout_s,
        )

    def _base_url(self) -> str:
        return CEREBRAS_BASE_URL

    def generate(self, **kwargs):
        rec = super().generate(**kwargs)
        return rec.model_copy(update={"provider": "cerebras"})


def build_llm_client(
    *, provider: str, model_name: str, temperature: float, max_output_tokens: int = 2048,
) -> AzureOpenAIClient:
    """Factory used by both experiments' run.py — picks the client class by
    provider, keeping the Azure default path completely unchanged."""
    if provider == "cerebras":
        return CerebrasClient(
            default_model=model_name, temperature=temperature, max_output_tokens=max_output_tokens,
        )
    return AzureOpenAIClient(
        model_env="FIGBROWSE_AGENT_MODEL", default_model=model_name,
        temperature=temperature, max_output_tokens=max_output_tokens,
    )
