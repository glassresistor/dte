"""Configuration for the DTE model harness.

All settings can be provided via environment variables (DTE_*) or a
config.yaml file. No models are downloaded — you must point to local
model files or running server endpoints.

Environment variables:
  DTE_LLM_BACKEND       : "llama_cpp" | "openai_compat"  (default: llama_cpp)
  DTE_LLM_MODEL_PATH    : path to .gguf file (for llama_cpp backend)
  DTE_LLM_BASE_URL      : OpenAI-compatible endpoint URL (for openai_compat)
  DTE_LLM_API_KEY       : API key (default: "not-needed" for local servers)
  DTE_LLM_MODEL_NAME    : model name for API backends (default: from path or "local")
  DTE_EMBED_BACKEND     : "sentence_tf" | "none"  (default: sentence_tf)
  DTE_EMBED_MODEL       : sentence-transformers model name or local path
                          (default: "all-MiniLM-L6-v2")
  DTE_EDIT_BUDGET       : max denoising iterations (default: 5)
  DTE_STOCHASTIC_RATE   : stochastic corruption rate for ablation (default: 0.0)
  DTE_SEED              : random seed for reproducibility (default: 0)
  DTE_EMBED_THRESHOLD   : max allowed embedding distance from original (Strategy A)
                          (default: 0.15, in cosine-distance units)

Example (llama.cpp local GGUF):
  export DTE_LLM_BACKEND=llama_cpp
  export DTE_LLM_MODEL_PATH=/path/to/your-model-q4_k_m.gguf
  python -m dte.harness.case1_grammar

Example (llama-server / vLLM / Ollama):
  export DTE_LLM_BACKEND=openai_compat
  export DTE_LLM_BASE_URL=http://localhost:8080/v1
  export DTE_LLM_MODEL_NAME=your-model-name
  python -m dte.harness.case1_grammar
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM backend configuration."""
    backend: str = "llama_cpp"          # "llama_cpp" | "openai_compat"
    model_path: str = ""                # .gguf path (llama_cpp) or model name (api)
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"
    model_name: str = ""                # for API backends; defaults to model_path
    n_ctx: int = 4096                   # context window (llama_cpp)
    temperature: float = 0.1            # low temp for deterministic edits
    max_tokens: int = 512

    def __post_init__(self):
        if not self.model_name:
            self.model_name = os.path.basename(self.model_path) if self.model_path else "local"


@dataclass
class EmbedConfig:
    """Embedding model configuration (Strategy A/B, §3.3)."""
    backend: str = "sentence_tf"        # "sentence_tf" | "none"
    model_name: str = "all-MiniLM-L6-v2"  # HF name or local path
    threshold: float = 0.15             # max cosine distance from original

    @property
    def enabled(self) -> bool:
        return self.backend != "none"


@dataclass
class DTEHarnessConfig:
    """Top-level harness configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    edit_budget: int = 5
    stochastic_rate: float = 0.0
    seed: int = 0

    @classmethod
    def from_env(cls) -> "DTEHarnessConfig":
        """Load configuration from DTE_* environment variables."""
        llm = LLMConfig(
            backend=os.environ.get("DTE_LLM_BACKEND", "llama_cpp"),
            model_path=os.environ.get("DTE_LLM_MODEL_PATH", ""),
            base_url=os.environ.get("DTE_LLM_BASE_URL", "http://localhost:8080/v1"),
            api_key=os.environ.get("DTE_LLM_API_KEY", "not-needed"),
            model_name=os.environ.get("DTE_LLM_MODEL_NAME", ""),
            n_ctx=int(os.environ.get("DTE_LLM_N_CTX", "4096")),
            temperature=float(os.environ.get("DTE_LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.environ.get("DTE_LLM_MAX_TOKENS", "512")),
        )
        embed = EmbedConfig(
            backend=os.environ.get("DTE_EMBED_BACKEND", "sentence_tf"),
            model_name=os.environ.get("DTE_EMBED_MODEL", "all-MiniLM-L6-v2"),
            threshold=float(os.environ.get("DTE_EMBED_THRESHOLD", "0.15")),
        )
        return cls(
            llm=llm,
            embed=embed,
            edit_budget=int(os.environ.get("DTE_EDIT_BUDGET", "5")),
            stochastic_rate=float(os.environ.get("DTE_STOCHASTIC_RATE", "0.0")),
            seed=int(os.environ.get("DTE_SEED", "0")),
        )

    def validate(self) -> list[str]:
        """Return a list of configuration problems (empty if OK)."""
        problems = []
        if self.llm.backend == "llama_cpp":
            if not self.llm.model_path:
                problems.append(
                    "DTE_LLM_MODEL_PATH is required for llama_cpp backend. "
                    "Set it to a .gguf file path, e.g.: "
                    "export DTE_LLM_MODEL_PATH=/path/to/model-q4_k_m.gguf")
            elif not os.path.isfile(self.llm.model_path):
                problems.append(
                    f"DTE_LLM_MODEL_PATH does not exist: {self.llm.model_path}")
        elif self.llm.backend == "openai_compat":
            if not self.llm.base_url:
                problems.append("DTE_LLM_BASE_URL is required for openai_compat backend")
        else:
            problems.append(f"Unknown DTE_LLM_BACKEND: {self.llm.backend}")

        if self.embed.enabled and self.embed.backend == "sentence_tf":
            # Don't check if model exists — it may be downloaded on first use
            # or available locally. The harness will report a clear error at runtime.
            pass
        return problems


def load_config() -> DTEHarnessConfig:
    """Load config from environment. Raises with helpful message if invalid."""
    cfg = DTEHarnessConfig.from_env()
    problems = cfg.validate()
    if problems:
        msg = "DTE harness configuration errors:\n" + "\n".join(f"  - {p}" for p in problems)
        raise RuntimeError(msg)
    return cfg
