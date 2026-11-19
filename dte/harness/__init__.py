"""DTE model-backed harness.

Extends the rule-based prototype (dte/) with real model backends:
  * LLM denoisers via llama-cpp-python or OpenAI-compatible APIs
  * Embedding-based edit-distance constraints (Strategy A, §3.3)
  * Pluggable backends — no models downloaded at build time

Backends:
  - "llama_cpp"       : local GGUF model via llama_cpp.Llama
  - "openai_compat"   : any OpenAI-compatible endpoint (llama-server, vLLM, Ollama)
  - "sentence_tf"     : sentence-transformers for embeddings (Strategy A/B)

Configuration is via dte/harness/config.yaml or environment variables.
See dte/harness/README.md for setup instructions.
"""

__all__ = ["backend", "config", "embedding", "llm_denoiser"]
