# DTE Model-Backed Harness

Extends the rule-based DTE prototype (`dte/`) with **real model backends** for
the denoising step. The oracle, mask, pipeline, and metrics are unchanged —
only the "denoiser" is swapped from a heuristic to an LLM (or embedding model).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DTE Pipeline                             │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────────────┐  │
│  │  Oracle   │──▶│ Informed     │──▶│  Denoiser (pluggable)   │  │
│  │(tests/    │   │ Corruption   │   │                         │  │
│  │ linter)   │   │ (mask M)     │   │  ┌───────────────────┐  │  │
│  └──────────┘   └──────────────┘   │  │ Rule-based (v1)    │  │  │
│       ▲                            │  │ LLM: llama_cpp     │  │  │
│       │                            │  │ LLM: openai_compat │  │  │
│       │                            │  │ Embedding (Strat A)│  │  │
│       └────────────────────────────│  └───────────────────┘  │  │
│         verify + expand            └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Key invariant (structural locality, §2.3):** non-masked tokens/lines are
copied verbatim by the pipeline code, *not* left to the model's discretion.
The LLM only sees and can only modify masked positions.

## Setup

### 1. Install dependencies

```bash
# Core (already installed in this environment):
pip install torch transformers sentence-transformers openai numpy

# For local GGUF inference:
pip install llama-cpp-python

# Or, if you prefer to run a server instead of in-process inference:
#   - llama.cpp:  build from source, run `llama-server --model your.gguf`
#   - vLLM:       pip install vllm && vllm serve your-model
#   - Ollama:     brew install ollama && ollama pull llama3.1
```

### 2. Get a model (you provide this — nothing is downloaded here)

Any of these work:

| Backend | What you need | Example |
|---------|--------------|---------|
| `llama_cpp` | A `.gguf` file on disk | Qwen2.5-7B-Instruct-Q4_K_M.gguf |
| `openai_compat` | A running OpenAI-compatible server | `llama-server`, vLLM, Ollama |
| Embedding (Strategy A) | A sentence-transformers model | `all-MiniLM-L6-v2` (~80MB) or local path |

**For `llama_cpp`:** download a GGUF file from HuggingFace (e.g., search
"Qwen2.5 7B GGUF" or "Llama-3.1-8B-Instruct GGUF") and note the path.

**For `openai_compat`:** start a server:
```bash
# Option A: llama.cpp server
llama-server --model /path/to/your-model-q4_k_m.gguf --port 8080

# Option B: Ollama
ollama serve &
ollama pull llama3.1:8b
# base_url becomes http://localhost:11434/v1

# Option C: vLLM
vllm serve /path/to/your-model --port 8080
```

### 3. Configure via environment variables

**Option A — llama.cpp in-process (no server needed):**
```bash
export DTE_LLM_BACKEND=llama_cpp
export DTE_LLM_MODEL_PATH=/path/to/your-model-q4_k_m.gguf
export DTE_LLM_N_CTX=4096
export DTE_LLM_TEMPERATURE=0.1

# Optional: enable embedding constraint (Strategy A)
export DTE_EMBED_BACKEND=sentence_tf
export DTE_EMBED_MODEL=all-MiniLM-L6-v2
# Or point to a local model directory to avoid any download:
# export DTE_EMBED_MODEL=/path/to/local/minilm-model/

python -m dte.harness.case1_grammar
python -m dte.harness.case2_code
```

**Option B — llama-server / vLLM:**
```bash
export DTE_LLM_BACKEND=openai_compat
export DTE_LLM_BASE_URL=http://localhost:8080/v1
export DTE_LLM_MODEL_NAME=your-model-name
export DTE_LLM_API_KEY=not-needed

python -m dte.harness.case1_grammar
python -m dte.harness.case2_code
```

**Option C — Ollama:**
```bash
export DTE_LLM_BACKEND=openai_compat
export DTE_LLM_BASE_URL=http://localhost:11434/v1
export DTE_LLM_MODEL_NAME=llama3.1:8b
export DTE_LLM_API_KEY=not-needed

python -m dte.harness.case1_grammar
```

### 4. Run

```bash
# Case 1: Grammar/spelling (4 planted errors)
python -m dte.harness.case1_grammar

# Case 2: Code repair (2 bugs, easy fixture)
python -m dte.harness.case2_code

# Case 2 hard: 3 bugs in different functions
export DTE_CODE_FIXTURE=hard
python -m dte.harness.case2_code
```

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DTE_LLM_BACKEND` | `llama_cpp` | `llama_cpp` or `openai_compat` |
| `DTE_LLM_MODEL_PATH` | *(required for llama_cpp)* | Path to .gguf file |
| `DTE_LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible endpoint |
| `DTE_LLM_API_KEY` | `not-needed` | API key for the endpoint |
| `DTE_LLM_MODEL_NAME` | *(from path)* | Model name for API backends |
| `DTE_LLM_N_CTX` | `4096` | Context window (llama_cpp) |
| `DTE_LLM_TEMPERATURE` | `0.1` | Sampling temperature (low = deterministic) |
| `DTE_LLM_MAX_TOKENS` | `512` | Max tokens per completion |
| `DTE_EMBED_BACKEND` | `sentence_tf` | `sentence_tf` or `none` |
| `DTE_EMBED_MODEL` | `all-MiniLM-L6-v2` | HF name or local path |
| `DTE_EMBED_THRESHOLD` | `0.15` | Max cosine distance (Strategy A) |
| `DTE_EDIT_BUDGET` | `5` | Max denoising iterations |
| `DTE_STOCHASTIC_RATE` | `0.0` | Stochastic corruption rate (§7.4 ablation) |
| `DTE_SEED` | `0` | Random seed |
| `DTE_CODE_FIXTURE` | `easy` | `easy` (2 bugs) or `hard` (3 bugs) |

## What each file does

| File | Role |
|------|------|
| `dte/harness/config.py` | Environment variable configuration loader |
| `dte/harness/backend.py` | LLM backends: `LlamaCppBackend`, `OpenAICompatBackend` |
| `dte/harness/embedding.py` | Strategy A/B: embedding distance constraint |
| `dte/harness/llm_denoiser.py` | LLM-backed denoisers for Case 1 & 2 |
| `dte/harness/case1_grammar.py` | Runnable: grammar DTE with LLM denoiser |
| `dte/harness/case2_code.py` | Runnable: code repair DTE with LLM denoiser |

## How the LLM denoiser works (vs. rule-based v1)

**v1 (rule-based):** For each masked token, look up a correction in a
hardcoded table (`recieve → receive`). Fast, deterministic, but limited to
known errors.

**Harness (LLM-backed):** For each masked position, the LLM is prompted with:
- The full text
- The specific error positions marked with `[token]`
- An instruction to respond as JSON: `{"position": "corrected_token"}`

The pipeline then:
1. Parses the JSON response
2. Applies corrections **only at masked positions** (hard constraint)
3. Ignores any corrections the LLM proposes for unmasked positions
4. Optionally checks embedding distance (Strategy A) as a soft constraint

This means the LLM can handle *any* grammar/spelling error, not just the
ones in the correction table — but it still can't touch non-error tokens.

## Comparison: v1 vs. harness

| Aspect | v1 (rule-based) | Harness (LLM) |
|--------|-----------------|---------------|
| Denoiser | Correction table + heuristics | LLM (any GGUF or API model) |
| Error coverage | Only known misspellings | Any grammar/spelling error |
| Structural locality | Enforced by code | Enforced by code (same guarantee) |
| Dependencies | Pure stdlib | torch, transformers, llama-cpp-python or openai |
| Model download | None | User provides model file/server |
| Speed | Instant | ~1-5s per iteration (depends on model) |
| Reproducibility | Deterministic | Near-deterministic (temp=0.1) |

## Troubleshooting

**"llama-cpp-python is not installed"**
```bash
pip install llama-cpp-python
# If build fails, try:
CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install llama-cpp-python
```

**"Connection refused" for openai_compat**
Make sure your server is running and the port matches `DTE_LLM_BASE_URL`.
Test with: `curl http://localhost:8080/v1/models`

**"Could not parse JSON from LLM response"**
The model may not be following the JSON format instruction. Try:
- A larger/better instruction-following model (7B+ recommended)
- Lower temperature: `export DTE_LLM_TEMPERATURE=0.0`
- Check the raw response in the log output

**Embedding model download is slow / blocked**
Point to a local copy:
```bash
# Download once (or copy from another machine):
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# Then use the cached path:
export DTE_EMBED_MODEL=~/.cache/torch/sentence_transformers/all-MiniLM-L6-v2
# Or disable embeddings entirely:
export DTE_EMBED_BACKEND=none
```

## Research experiments you can run (§7)

1. **Baseline comparison:** Run v1 (`python -m dte.case1_grammar`) and the
   harness side-by-side. Compare ED, precision, and over-editing.

2. **Model scaling:** Try 1B, 3B, 8B, 70B models. Does a larger model
   produce fewer spurious edits? Measure ED vs. model size.

3. **Strategy A ablation:** Run with `DTE_EMBED_BACKEND=sentence_tf` vs.
   `none`. Does the embedding constraint improve precision?

4. **Stochastic corruption (§7.4):** Set `DTE_STOCHASTIC_RATE=0.02` and
   compare ED/precision against the deterministic mask.

5. **Edit budget scaling:** Vary `DTE_EDIT_BUDGET` from 1 to 10. Does DTE
   converge to AR behavior at high budgets (§4.3 hypothesis)?
