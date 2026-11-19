"""LLM backends for the DTE harness.

Two backends, both designed to work with models YOU provide:

1. llama_cpp  — local GGUF inference via llama-cpp-python.
   You supply a .gguf file (e.g., from HuggingFace). No download here.

2. openai_compat — any OpenAI-compatible /v1/chat/completions endpoint:
   - llama.cpp's llama-server  (llama-server --model your.gguf)
   - vLLM                      (vllm serve your-model)
   - Ollama                    (ollama serve, base_url=http://localhost:11434/v1)
   - LM Studio, text-generation-webui, etc.

Both expose the same interface:
    complete(prompt: str) -> str
    complete_json(prompt: str) -> dict   # parses JSON from response
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from .config import LLMConfig


class LLMBackend(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Generate a completion for the given prompt. Returns raw text."""
        ...

    def complete_json(self, prompt: str) -> dict:
        """Generate a completion and parse it as JSON.

        Tries to extract a JSON object from the response (handles models
        that wrap JSON in markdown code fences or add prose).
        """
        raw = self.complete(prompt)
        return _extract_json(raw)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output, handling common wrappers."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")


# ---------------------------------------------------------------------------
# llama.cpp backend (local GGUF)
# ---------------------------------------------------------------------------

class LlamaCppBackend(LLMBackend):
    """Local GGUF inference via llama-cpp-python.

    Requires: pip install llama-cpp-python
    Model:    you provide a .gguf file path (DTE_LLM_MODEL_PATH)
    """

    def __init__(self, config: LLMConfig):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install with:\n"
                "  pip install llama-cpp-python\n"
                "(or build from source: CMAKE_ARGS='-DGGML_BLAS=ON' pip install llama-cpp-python)\n"
                f"Original error: {e}"
            ) from e

        self._llm = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            verbose=False,
        )
        self._temp = config.temperature
        self._max_tokens = config.max_tokens

    def complete(self, prompt: str) -> str:
        result = self._llm.create_completion(
            prompts=prompt,
            temperature=self._temp,
            max_tokens=self._max_tokens,
            stop=["\n\n", "###"],
        )
        return result["choices"][0]["text"].strip()


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (llama-server, vLLM, Ollama, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatBackend(LLMBackend):
    """Any OpenAI-compatible /v1/chat/completions endpoint.

    Works with: llama.cpp llama-server, vLLM, Ollama, LM Studio, etc.
    Requires: pip install openai  (already installed)
    """

    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package is not installed. Install with:\n"
                "  pip install openai\n"
                f"Original error: {e}"
            ) from e

        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self._model = config.model_name or "local"
        self._temp = config.temperature
        self._max_tokens = config.max_tokens

    def complete(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temp,
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend(config: LLMConfig) -> LLMBackend:
    """Create the appropriate backend from config."""
    if config.backend == "llama_cpp":
        return LlamaCppBackend(config)
    elif config.backend == "openai_compat":
        return OpenAICompatBackend(config)
    else:
        raise ValueError(f"Unknown LLM backend: {config.backend}")
