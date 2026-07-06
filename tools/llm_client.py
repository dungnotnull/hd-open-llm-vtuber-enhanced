"""Unified LLM client for open-llm-vtuber-enhanced.

Provider chain: Claude (primary) -> OpenAI (fallback) -> Ollama (offline/privacy).
Used by the persona dialogue engine and the research-paper improvement proposer.

Design goals:
  * Single ``complete()`` / ``stream()`` surface regardless of provider.
  * Automatic fallback down the priority chain on transient errors.
  * Exponential backoff (1s / 2s / 4s) on a per-provider basis.
  * Cost accounting hook (delegated to MemoryManager) so the agent can
    report spend per session.
  * PRIVACY_MODE env var forces the local Ollama provider.

The client is intentionally dependency-light: the heavy SDKs (anthropic,
openai) are imported lazily so the module imports cleanly even when only a
subset of providers is installed.
"""

from __future__ import annotations

import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional, List, Dict, Any, Callable

logger = logging.getLogger("vtuber.llm_client")

# ---------------------------------------------------------------------------
# Pricing table (USD per 1K tokens, input/output blended estimate).
# Updated from public pricing pages; used only for local cost accounting.
# ---------------------------------------------------------------------------
COST_PER_1K: Dict[str, Dict[str, float]] = {
    "claude-opus-4-8":      {"in": 0.015, "out": 0.075},
    "claude-sonnet-4-6":    {"in": 0.003, "out": 0.015},
    "claude-haiku-4-5":     {"in": 0.0008, "out": 0.004},
    "gpt-4o":               {"in": 0.0025, "out": 0.010},
    "gpt-4o-mini":          {"in": 0.00015, "out": 0.0006},
    "llama3":               {"in": 0.0, "out": 0.0},
    "mistral":              {"in": 0.0, "out": 0.0},
}

PROVIDER_PRIORITY = ["claude", "openai", "ollama"]

DEFAULT_MODELS = {
    "claude": "claude-opus-4-8",
    "openai": "gpt-4o",
    "ollama": "llama3",
}


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    fallback_used: bool = False
    error: Optional[str] = None


@dataclass
class LLMConfig:
    claude_model: str = DEFAULT_MODELS["claude"]
    openai_model: str = DEFAULT_MODELS["openai"]
    ollama_model: str = DEFAULT_MODELS["ollama"]
    ollama_base_url: str = "http://localhost:11434"
    max_tokens: int = 1024
    temperature: float = 0.7
    max_retries: int = 3
    privacy_mode: bool = False
    priority: List[str] = field(default_factory=lambda: list(PROVIDER_PRIORITY))


class LLMClient:
    """Provider-portable LLM client with automatic fallback."""

    def __init__(self, config: Optional[LLMConfig] = None,
                 cost_logger: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.config = config or LLMConfig()
        if os.getenv("PRIVACY_MODE", "").lower() in ("1", "true", "yes"):
            self.config.privacy_mode = True
        self._cost_logger = cost_logger
        self._anthropic = None
        self._openai = None

    # -- provider availability ------------------------------------------------
    def _provider_chain(self) -> List[str]:
        if self.config.privacy_mode:
            return ["ollama"]
        chain = []
        if os.getenv("ANTHROPIC_API_KEY"):
            chain.append("claude")
        if os.getenv("OPENAI_API_KEY"):
            chain.append("openai")
        chain.append("ollama")  # always available as a last resort
        # preserve requested priority ordering
        return [p for p in self.config.priority if p in chain] or ["ollama"]

    def _model_for(self, provider: str) -> str:
        return {
            "claude": self.config.claude_model,
            "openai": self.config.openai_model,
            "ollama": self.config.ollama_model,
        }[provider]

    # -- public API -----------------------------------------------------------
    def complete(self, prompt: str, system: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 session_id: Optional[str] = None) -> LLMResponse:
        """Complete a single prompt, walking the fallback chain on failure."""
        chain = self._provider_chain()
        last_err: Optional[str] = None
        for idx, provider in enumerate(chain):
            for attempt in range(self.config.max_retries):
                try:
                    start = time.time()
                    text, in_tok, out_tok = self._dispatch(
                        provider, prompt, system,
                        max_tokens or self.config.max_tokens,
                        temperature if temperature is not None else self.config.temperature,
                    )
                    latency = (time.time() - start) * 1000.0
                    model = self._model_for(provider)
                    cost = self._estimate_cost(model, in_tok, out_tok)
                    resp = LLMResponse(
                        text=text, provider=provider, model=model,
                        input_tokens=in_tok, output_tokens=out_tok,
                        cost_usd=cost, latency_ms=latency,
                        fallback_used=(idx > 0),
                    )
                    self._log_cost(resp, session_id)
                    return resp
                except Exception as exc:  # noqa: BLE001 - provider-agnostic guard
                    last_err = f"{provider}: {exc}"
                    logger.warning("LLM attempt failed (%s, try %d): %s",
                                   provider, attempt + 1, exc)
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s
        return LLMResponse(text="", provider="none", model="none",
                           fallback_used=True, error=last_err or "all providers failed")

    def stream(self, prompt: str, system: Optional[str] = None,
               max_tokens: Optional[int] = None,
               temperature: Optional[float] = None) -> Iterator[str]:
        """Yield text chunks. Falls back to a single ``complete()`` block if a
        provider does not support streaming. First-token latency matters for the
        VTuber turn loop, so streaming is preferred where available."""
        chain = self._provider_chain()
        for provider in chain:
            try:
                yield from self._dispatch_stream(
                    provider, prompt, system,
                    max_tokens or self.config.max_tokens,
                    temperature if temperature is not None else self.config.temperature,
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("stream failed on %s: %s", provider, exc)
                continue
        # final fallback: non-streaming
        yield self.complete(prompt, system, max_tokens, temperature).text

    # -- dispatch -------------------------------------------------------------
    def _dispatch(self, provider, prompt, system, max_tokens, temperature):
        if provider == "claude":
            return self._call_claude(prompt, system, max_tokens, temperature)
        if provider == "openai":
            return self._call_openai(prompt, system, max_tokens, temperature)
        return self._call_ollama(prompt, system, max_tokens, temperature)

    def _dispatch_stream(self, provider, prompt, system, max_tokens, temperature):
        if provider == "claude":
            return self._stream_claude(prompt, system, max_tokens, temperature)
        if provider == "openai":
            return self._stream_openai(prompt, system, max_tokens, temperature)
        return self._stream_ollama(prompt, system, max_tokens, temperature)

    # -- Claude ---------------------------------------------------------------
    def _client_anthropic(self):
        if self._anthropic is None:
            import anthropic  # lazy
            self._anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._anthropic

    def _call_claude(self, prompt, system, max_tokens, temperature):
        client = self._client_anthropic()
        msg = client.messages.create(
            model=self.config.claude_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if hasattr(block, "text"))
        return text, msg.usage.input_tokens, msg.usage.output_tokens

    def _stream_claude(self, prompt, system, max_tokens, temperature):
        client = self._client_anthropic()
        with client.messages.stream(
            model=self.config.claude_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for chunk in stream.text_stream:
                yield chunk

    # -- OpenAI ---------------------------------------------------------------
    def _client_openai(self):
        if self._openai is None:
            from openai import OpenAI  # lazy
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai

    def _call_openai(self, prompt, system, max_tokens, temperature):
        client = self._client_openai()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.config.openai_model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, usage.prompt_tokens, usage.completion_tokens

    def _stream_openai(self, prompt, system, max_tokens, temperature):
        client = self._client_openai()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        stream = client.chat.completions.create(
            model=self.config.openai_model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # -- Ollama ---------------------------------------------------------------
    def _call_ollama(self, prompt, system, max_tokens, temperature):
        import urllib.request
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        req = urllib.request.Request(
            f"{self.config.ollama_base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        text = data.get("response", "")
        # Ollama reports token counts when available
        in_tok = data.get("prompt_eval_count", len(prompt.split()))
        out_tok = data.get("eval_count", len(text.split()))
        return text, in_tok, out_tok

    def _stream_ollama(self, prompt, system, max_tokens, temperature):
        import urllib.request
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "system": system or "",
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        req = urllib.request.Request(
            f"{self.config.ollama_base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                if not line.strip():
                    continue
                obj = json.loads(line.decode())
                if obj.get("response"):
                    yield obj["response"]
                if obj.get("done"):
                    break

    # -- accounting -----------------------------------------------------------
    def _estimate_cost(self, model: str, in_tok: int, out_tok: int) -> float:
        rate = COST_PER_1K.get(model)
        if not rate:
            return 0.0
        return (in_tok / 1000.0) * rate["in"] + (out_tok / 1000.0) * rate["out"]

    def _log_cost(self, resp: LLMResponse, session_id: Optional[str]):
        if self._cost_logger is None:
            return
        try:
            self._cost_logger({
                "session_id": session_id,
                "provider": resp.provider,
                "model": resp.model,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
                "latency_ms": resp.latency_ms,
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("cost logging failed: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = LLMClient()
    r = client.complete("Say hello as an energetic anime VTuber in one sentence.")
    print(f"[{r.provider}/{r.model}] {r.text}")
    print(f"cost=${r.cost_usd:.5f} latency={r.latency_ms:.0f}ms fallback={r.fallback_used}")
