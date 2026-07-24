"""Client Mistral — implémente le port partagé `domain.ports.ChatLLM`.

Anti-corruption layer autour du SDK mistralai. La clé API est lue depuis la variable
d'environnement nommée dans la config (jamais stockée dans le code / le YAML).
Température 0 par défaut + logging possible en amont = runs auditables (ARCHITECTURE §8).
"""

from __future__ import annotations

import os
import time

from config.schema import LLMConfig
from domain.types import TokenUsage


class MistralChatLLM:
    """Chat completion via l'API Mistral (free tier compatible)."""

    def __init__(self, config: LLMConfig) -> None:
        from mistralai import Mistral

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Variable d'environnement {config.api_key_env} absente : "
                "impossible d'appeler l'API Mistral."
            )
        self._config = config
        self._client = Mistral(api_key=api_key)

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[str, TokenUsage]:
        start = time.perf_counter()
        resp = self._client.chat.complete(
            model=self._config.model,
            temperature=temperature,
            max_tokens=max_tokens or self._config.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency = time.perf_counter() - start
        text = resp.choices[0].message.content or ""
        usage = TokenUsage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0),
            total_tokens=getattr(resp.usage, "total_tokens", 0),
            latency_s=latency,
            llm_calls=1,
        )
        return text, usage
