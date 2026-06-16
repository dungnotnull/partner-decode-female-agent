"""
llm_client.py — Unified LLM client for partner-decode-female-agent.

Provider chain: Claude (claude-opus-4-8) → OpenAI (gpt-4o) → Ollama (llama3)
Features: exponential backoff retry, streaming, cost logging, PRIVACY_MODE.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Generator, Optional, Tuple

logger = logging.getLogger(__name__)

# Cost per 1K tokens (input, output) in USD
COST_PER_1K: dict = {
    "claude-opus-4-8":          (0.015, 0.075),
    "claude-sonnet-4-6":        (0.003, 0.015),
    "claude-haiku-3-5":         (0.00025, 0.00125),
    "gpt-4o":                   (0.005, 0.015),
    "gpt-4o-mini":              (0.00015, 0.0006),
    "llama3":                   (0.0, 0.0),  # local
    "mistral":                  (0.0, 0.0),  # local
}

RETRY_DELAYS = [1.0, 2.0, 4.0]  # seconds between retries


class LLMClient:
    """
    Unified LLM API client with automatic provider fallback.

    Usage:
        client = LLMClient()
        text, provider = client.complete("Analyze this...", system_prompt="You are...")
    """

    def __init__(
        self,
        memory_manager=None,
        primary_provider: str = "claude",
        primary_model: Optional[str] = None,
        fallback_provider: str = "openai",
        fallback_model: Optional[str] = None,
        offline_provider: str = "ollama",
        offline_model: str = "llama3",
    ) -> None:
        self._memory = memory_manager
        self.primary_provider = primary_provider
        self.primary_model = primary_model or "claude-opus-4-8"
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model or "gpt-4o"
        self.offline_provider = offline_provider
        self.offline_model = offline_model

        # PRIVACY_MODE: force all calls to Ollama
        self.privacy_mode = os.environ.get("PRIVACY_MODE", "false").lower() == "true"
        if self.privacy_mode:
            logger.info("PRIVACY_MODE enabled — all LLM calls routed to Ollama")

        self._provider_order = self._build_provider_order()

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        session_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Call LLM with automatic fallback.
        Returns: (response_text, provider_used)
        Raises: RuntimeError if all providers fail.
        """
        last_error = None
        for provider, model in self._provider_order:
            for attempt, delay in enumerate(RETRY_DELAYS + [None]):
                try:
                    response_text = self._call_provider(
                        provider, model, user_prompt, system_prompt, max_tokens, temperature
                    )
                    # Log cost
                    input_tokens = len(user_prompt.split()) + len(system_prompt.split())
                    output_tokens = len(response_text.split())
                    cost = self._compute_cost(model, input_tokens, output_tokens)
                    if self._memory:
                        try:
                            self._memory.log_llm_cost(
                                provider=provider,
                                model=model,
                                tokens_in=input_tokens,
                                tokens_out=output_tokens,
                                cost_usd=cost,
                                session_id=session_id,
                            )
                        except Exception:
                            pass
                    logger.debug("LLM call succeeded: provider=%s, model=%s", provider, model)
                    return response_text, f"{provider}/{model}"

                except Exception as exc:
                    last_error = exc
                    if delay is None:
                        logger.warning(
                            "Provider %s/%s failed after retries: %s", provider, model, exc
                        )
                        break
                    logger.debug(
                        "Provider %s attempt %d failed: %s — retrying in %.0fs",
                        provider, attempt + 1, exc, delay,
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        )

    def stream(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """
        Streaming LLM call. Yields text chunks as they arrive.
        Falls back gracefully if streaming is unavailable.
        """
        for provider, model in self._provider_order:
            try:
                if provider == "claude":
                    yield from self._stream_claude(user_prompt, system_prompt, model, max_tokens, temperature)
                elif provider == "openai":
                    yield from self._stream_openai(user_prompt, system_prompt, model, max_tokens, temperature)
                elif provider == "ollama":
                    yield from self._stream_ollama(user_prompt, system_prompt, model, max_tokens, temperature)
                return
            except Exception as exc:
                logger.warning("Streaming failed for %s: %s — trying next provider", provider, exc)

        # Last resort: non-streaming fallback
        try:
            text, _ = self.complete(user_prompt, system_prompt, max_tokens, temperature)
            yield text
        except Exception as exc:
            yield f"[Error: all providers failed — {exc}]"

    # ------------------------------------------------------------------
    # Provider Implementations
    # ------------------------------------------------------------------

    def _call_provider(
        self,
        provider: str,
        model: str,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if provider == "claude":
            return self._call_claude(user_prompt, system_prompt, model, max_tokens, temperature)
        elif provider == "openai":
            return self._call_openai(user_prompt, system_prompt, model, max_tokens, temperature)
        elif provider == "ollama":
            return self._call_ollama(user_prompt, system_prompt, model, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _call_claude(
        self, user_prompt: str, system_prompt: str, model: str, max_tokens: int, temperature: float
    ) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    def _call_openai(
        self, user_prompt: str, system_prompt: str, model: str, max_tokens: int, temperature: float
    ) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def _call_ollama(
        self, user_prompt: str, system_prompt: str, model: str, max_tokens: int, temperature: float
    ) -> str:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        import requests
        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _stream_claude(
        self, user_prompt: str, system_prompt: str, model: str, max_tokens: int, temperature: float
    ) -> Generator[str, None, None]:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _stream_openai(
        self, user_prompt: str, system_prompt: str, model: str, max_tokens: int, temperature: float
    ) -> Generator[str, None, None]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield delta.content

    def _stream_ollama(
        self, user_prompt: str, system_prompt: str, model: str, max_tokens: int, temperature: float
    ) -> Generator[str, None, None]:
        import json as _json
        import requests
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        with requests.post(f"{base_url}/api/generate", json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = _json.loads(line.decode("utf-8"))
                    if "response" in data:
                        yield data["response"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_provider_order(self) -> list:
        """Return ordered list of (provider, model) tuples based on config + PRIVACY_MODE."""
        if self.privacy_mode:
            return [(self.offline_provider, self.offline_model)]
        return [
            (self.primary_provider, self.primary_model),
            (self.fallback_provider, self.fallback_model),
            (self.offline_provider, self.offline_model),
        ]

    def _compute_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Estimate cost in USD for a single call."""
        rates = COST_PER_1K.get(model, (0.005, 0.015))  # default to gpt-4o rates
        cost = (tokens_in / 1000) * rates[0] + (tokens_out / 1000) * rates[1]
        return round(cost, 8)
