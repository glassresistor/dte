"""DTE Case 1 (LLM-backed) — Grammar/spelling with a real model denoiser.

Same mechanism as dte.case1_grammar (rule-based), but the denoiser is
replaced with an LLM that proposes corrections for masked positions only.
Non-masked tokens are still copied verbatim (structural locality).

Usage:
  # With llama.cpp (local GGUF):
  export DTE_LLM_BACKEND=llama_cpp
  export DTE_LLM_MODEL_PATH=/path/to/your-model-q4_k_m.gguf
  python -m dte.harness.case1_grammar

  # With llama-server / vLLM / Ollama:
  export DTE_LLM_BACKEND=openai_compat
  export DTE_LLM_BASE_URL=http://localhost:8080/v1
  export DTE_LLM_MODEL_NAME=your-model
  python -m dte.harness.case1_grammar

  # Disable embedding constraint (Strategy A):
  export DTE_EMBED_BACKEND=none

  # Enable stochastic corruption ablation (§7.4):
  export DTE_STOCHASTIC_RATE=0.02
"""
from __future__ import annotations

import sys
from typing import List

# Reuse the rule-based oracle and tokenizer from the base package
from ..case1_grammar import (
    GrammarOracle, build_fixture, tokenize, reconstruct,
)
from ..core import DTEConfig, run_dte, edit_precision, metrics_table
from ..ed import render_edit_report

from .config import load_config
from .backend import create_backend
from .embedding import create_embedding_model
from .llm_denoiser import create_grammar_denoiser


def main() -> None:
    # Load configuration
    try:
        cfg = load_config()
    except RuntimeError as e:
        print(f"Configuration error:\n{e}", file=sys.stderr)
        sys.exit(1)

    # Create backends
    print("Initializing LLM backend...")
    llm = create_backend(cfg.llm)
    print(f"  Backend: {cfg.llm.backend}")
    if cfg.llm.backend == "llama_cpp":
        print(f"  Model:   {cfg.llm.model_path}")
    else:
        print(f"  URL:     {cfg.llm.base_url}")
        print(f"  Model:   {cfg.llm.model_name}")

    embed = None
    if cfg.embed.enabled:
        print("Initializing embedding model (Strategy A)...")
        try:
            embed = create_embedding_model(cfg.embed)
            print(f"  Model:   {cfg.embed.model_name}")
            print(f"  Threshold: {cfg.embed.threshold} (cosine distance)")
        except Exception as e:
            print(f"  WARNING: embedding model unavailable ({e}). "
                  f"Continuing without Strategy A constraint.", file=sys.stderr)
            embed = None

    denoiser = create_grammar_denoiser(llm, embed)

    # Run DTE pipeline (same as rule-based, but with LLM denoiser)
    text = build_fixture()
    tokens = tokenize(text)
    oracle = GrammarOracle()

    print(f"\nInput ({len(tokens)} tokens):\n  {text!r}\n")

    log: List[str] = []
    result = run_dte(
        doc=tokens,
        oracle=oracle,
        denoiser=denoiser,
        config=DTEConfig(
            budget=cfg.edit_budget,
            stochastic_rate=cfg.stochastic_rate,
            seed=cfg.seed,
        ),
        log=log,
    )

    for line in log:
        print("  " + line)

    edited_text = reconstruct(result.edited)
    print(f"\nOutput ({len(result.edited)} tokens):\n  {edited_text!r}\n")

    prec = edit_precision(oracle, result.original, result.edited)
    if prec is not None:
        object.__setattr__(result, "edit_precision", prec)

    print("Changed tokens:")
    print(render_edit_report(result.original, result.edited))
    print()
    print(metrics_table(result, extra={
        "Oracle feedback (final)": oracle.feedback(result.edited),
        "LLM backend": cfg.llm.backend,
        "Embedding constraint": "on" if embed else "off",
    }))

    # Acceptance criteria
    gt_ed = 4
    result.ground_truth_ed = gt_ed
    print()
    if result.passed:
        print("PASS: oracle passes on output")
    else:
        print("FAIL: oracle does not pass on output")
    changed = [i for i in range(len(result.original))
               if result.original[i] != result.edited[i]]
    flagged = oracle.mask(result.original)
    if set(changed) <= flagged:
        print(f"PASS: only masked tokens changed ({len(changed)} of {len(flagged)} masked)")
    else:
        extra = set(changed) - flagged
        print(f"FAIL: unmasked tokens changed: {sorted(extra)}")
    if result.tokens_changed == gt_ed:
        print(f"PASS: ED == ground truth ({gt_ed})")
    else:
        print(f"NOTE: ED={result.tokens_changed}, ground truth={gt_ed} "
              f"(LLM may over/under-correct)")


if __name__ == "__main__":
    main()
