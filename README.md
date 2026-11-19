# Diffusive Text Editing (DTE)

Implementation of the DTE mechanism from `REPORT_DTE.txt`:
oracle-guided informed corruption → local denoising → verification →
iterative mask expansion, with edit-budget control.

## Two tiers

### 1. Rule-based prototype (`dte/`) — zero dependencies

Pure Python 3 stdlib. No models, no pip installs, no API keys.
Demonstrates the full DTE mechanism with a heuristic denoiser.

```bash
python -m dte.case1_grammar   # Grammar/spelling (4 planted errors)
python -m dte.case2_code      # Test-passing code (2 planted bugs)
```

Both run clean, print diff + metrics, and pass built-in assertions.

### 2. Model-backed harness (`dte/harness/`) — real LLM denoiser

Same pipeline, but the denoiser is a real model (llama.cpp GGUF or any
OpenAI-compatible API). You provide the model — nothing is downloaded here.

```bash
# With llama.cpp (local GGUF):
export DTE_LLM_BACKEND=llama_cpp
export DTE_LLM_MODEL_PATH=/path/to/your-model-q4_k_m.gguf
python -m dte.harness.case1_grammar

# With llama-server / vLLM / Ollama:
export DTE_LLM_BACKEND=openai_compat
export DTE_LLM_BASE_URL=http://localhost:8080/v1
export DTE_LLM_MODEL_NAME=your-model
python -m dte.harness.case2_code
```

Full setup instructions: `dte/harness/README.md`

## File layout

```
REPORT_DTE.txt              # Original research report (handoff document)
reference_baseline/         # Frozen copy of v1 for comparison
  dte_v1_rulebased/         #   the rule-based prototype as first built
dte/
  __init__.py
  core.py                   # Pipeline, oracle/denoiser interfaces, metrics
  ed.py                     # Token-level Levenshtein, diff helpers
  case1_grammar.py          # v1: rule-based grammar DTE
  case2_code.py             # v1: test-passing code DTE (edit-grammar bandit)
  README.md                 # v1 documentation
  harness/
    __init__.py
    config.py               # Env-var configuration loader
    backend.py              # LLM backends (llama_cpp, openai_compat)
    embedding.py            # Strategy A/B: embedding distance constraint
    llm_denoiser.py         # LLM-backed denoisers for Case 1 & 2
    case1_grammar.py        # Runnable: grammar DTE with LLM
    case2_code.py           # Runnable: code repair DTE with LLM
    README.md               # Harness setup + usage instructions
```

## Concept → code mapping (Appendix A)

| Concept section | v1 (rule-based) | Harness (LLM) |
|----------------|-----------------|---------------|
| Oracle O (§3.2) | Rule-based grammar checker / test suite via subprocess | Same oracles (unchanged) |
| Informed corruption / mask M (§3.4.1) | Error-token mask from oracle | Same masks (unchanged) |
| Local denoising (§3.2, §3.4.2) | Correction table + heuristics | LLM prompted with masked positions only |
| Structural locality (§2.3) | Non-masked tokens copied verbatim | Same guarantee (enforced in pipeline code) |
| Verification + mask expansion (§3.4.3) | Re-run oracle; expand on fail | Same (unchanged) |
| Edit budget B / steps T (§2.3) | Max iterations | Same (`DTE_EDIT_BUDGET`) |
| Embedding edit distance (Strategy A, §3.3) | Extension only | `dte/harness/embedding.py` |
| Metrics (§4.1) | Token Levenshtein ED; edit precision | Same + oracle-call count |

## Integrity notes

- **v1**: every number is computed at runtime. No fabricated results.
- **Harness**: the LLM denoiser adds a model dependency, but the pipeline's
  structural locality guarantee is unchanged — non-masked positions are
  copied verbatim by code, not left to the model.
- **No models are downloaded** by this repository. You provide model files
  or server endpoints via environment variables.
