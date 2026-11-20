# DTR/DTE Experiment — Handoff Prompt for Next Agent

## Context

You are continuing a research project on **DTE (Directed Token Editing)** — a technique that constrains an LLM to fix ONLY specific flagged tokens/lines, rather than letting it freely edit. The goal: measure when DTE helps vs hurts across different models, quantizations, and oracle quality levels.

**Project location:** `sandbox/` directory in this project.
**Results:** `sandbox/results/` — all JSON files with `_meta` blocks containing model, quantization, timestamp, params.

## Key Files

| File | Purpose |
|------|---------|
| `run_exp2_flint.py` | Main experiment runner (grammar + code). Auto-detects llama-server. |
| `fix_text.py` | "Fix My Document" harness — standalone text correction tool |
| `dte/experiment/oracle.py` | Oracle implementations: wordlist, heuristic, combined, languagetool, llm |
| `dte/experiment/run_experiment.py` | Core LLM calls, dataset loaders, test runner |
| `dte/experiment/gec_eval.py` | GEC F0.5 metric (MaxMatch) |
| `data/conll2014_gec.json` | Grammar benchmark (CoNLL-2014, 50 sentences) |
| `data/code_repair_bugs_multiline.jsonl` | Multi-line bug dataset (15 instances) |
| `data/mbpp_test.jsonl` | MBPP test cases for code verification |

## What's Been Done

### Experiments Run (Nov 20, 2026)

**Models tested:**
- `unsloth/Qwen3.8-27B-GGUF` (Q4_K) — 27B MoE, strong baseline
- `unsloth/Phi-4-reasoning-plus-GGUF` (Q4_K) — reasoning model, overwrites everything
- `unsloth/gemma-4-26B-A4B-it-GGUF` (Q4_K) — 26B MoE, good raw but DTE hurts
- `unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` — small 4B, neutral

**Grammar results (CoNLL-2014, n=50, oracle=combined):**
- DTE HURTS all models with the `combined` oracle (wordlist+heuristic)
- Qwen3.8: raw F0.5=0.18, DTE F0.5=0.09 (Δ=-0.09)
- All others: DTE F0.5=0.00 (model ignores span constraint or oracle is too noisy)
- **Earlier gold-oracle run showed DTE F0.5=0.71 vs raw 0.28** → oracle quality is THE bottleneck
- LLM oracle (Phi-4): only flagged 18/50 sentences → can't detect its own errors
- LanguageTool: server wasn't running, silently fell back to combined

**Code results (MBPP multi-line, n=15):**
- Phi-4: raw 20% → DTE 40% (+20%) — DTE is a LIFELINE for over-editing models
- Qwen3.8: raw 73% → DTE 87% (+13%) — consistent improvement
- Nemotron E4B: ~53% both ways (neutral, already minimal)
- Gemma-4: raw 60% → DTE 60% with adaptive (was -16% without adaptive)
- **Adaptive DTE fix:** only skip when raw passes tests AND diff ≤ threshold

### Scripts Updated

1. **Unique filenames** — model + quant + params + timestamp in every output file
2. **Quant auto-detection** — reads from `-m` file path in `ps aux` (e.g. `...Q4_K_M.gguf`)
3. **LLM oracle** — new `--oracle llm` option uses the model as error detector
4. **Adaptive DTE** — `--adaptive-threshold N` skips DTE when raw is verified correct + minimal
5. **fix_text.py** — standalone "Fix My Document" harness with pluggable oracles

## What The Data Says (Core Insight)

```
DTE's value = (model's tendency to over-edit) × (oracle quality)
```

- **High over-editing + good oracle → big win** (Phi-4 code: +20%)
- **Low over-editing + bad oracle → net loss** (Gemma grammar: -16%)
- **The oracle is the bottleneck for grammar, not the model**
  - Gold spans: DTE F0.5 = 0.71
  - Combined (wordlist+heuristic): DTE F0.5 = 0.09
  - That 8× gap is entirely oracle quality

## What To Do Next (Prioritized)

### P0: Get a REAL oracle working for grammar

The `combined` oracle is too noisy. Options in order of preference:

1. **Start LanguageTool server** and re-run with `--oracle languagetool`
   ```bash
   # Download and start LT (one-time setup):
   # https://languagetool.org/download/
   java -jar languagetool-server.jar --port 8823
   # Then:
   python run_exp2_flint.py --task grammar --oracle languagetool --n-grammar 50
   ```
   LT is a real grammar checker — should be much closer to gold than wordlist+heuristic.

2. **Improve the LLM oracle prompt** — current one asks for token indices as JSON.
   Try: give it the sentence + ask "which words are wrong? list them by name" then
   map names→indices. The current format (numeric indices) is hard for small models.

3. **Two-pass oracle**: use a SMALL model (Nemotron 4B) as detector, feed its spans
   to a LARGER model (Qwen 27B) as the editor. Separates detection from correction.

### P1: Validate adaptive DTE with the fix

The adaptive threshold now requires `r_fails == 0` before skipping. Re-run:
```bash
# Gemma — should now be safe (only skips verified passes)
python run_exp2_flint.py --task code --code-dataset multi --adaptive-threshold 3

# Qwen — should recover the 2 fixes it lost when all-15-were-skipped
python run_exp2_flint.py --task code --code-dataset multi --adaptive-threshold 3
```

Expected: Gemma stays at ~60% (skips are safe), Qwen goes back to ~87% 
(failing cases now get DTE applied).

### P2: Test fix_text.py on real documents

Create a test document with known errors and run:
```bash
# Create test file
cat > test_doc.txt << 'EOF'
The team will recieve the report quickly and definately review it.
This is a correct sentence with no errors at all.
I went to the store and bought some apples today, and i will bring them to work.
EOF

# Run with different oracles
python fix_text.py test_doc.txt --oracle combined --mode both --diff
python fix_text.py test_doc.txt --oracle llm --mode dte --diff
```

Check: does it actually fix the errors without breaking correct sentences?

### P3: Scale up code experiments

n=15 is too small for reliable deltas. Run n=50:
```bash
python run_exp2_flint.py --task code --code-dataset multi --n-code 50
```
This gives 50× more data points per model. Also try `--code-dataset single` 
to compare single-line vs multi-line bug difficulty.

### P4: Temperature sweep

Phi-4 and Gemma show high variance between runs. Test temperature effect:
- Add a `--temperature` flag to the script (currently hardcoded 0.1)
- Run at temp=0, 0.3, 0.7 on the models with highest variance
- Check if lower temp stabilizes Gemma's DTE regression

## How To Run Experiments

```bash
# Auto-detect running llama-server (port, model, quant all from ps aux)
python run_exp2_flint.py --task both --oracle combined --code-dataset multi --workers 4

# Specific overrides
python run_exp2_flint.py --task grammar --oracle llm --n-grammar 50 --model my-model --quant Q5_K_M

# Adaptive code (only skip DTE when raw is verified correct + minimal)
python run_exp2_flint.py --task code --code-dataset multi --adaptive-threshold 3

# Fix a document
python fix_text.py my_doc.txt --oracle languagetool --mode dte --diff --output fixed.txt
```

## Reading Results

Every JSON file in `results/` has a `_meta` block:
```json
{
  "_meta": {
    "model": "unsloth/Qwen3.8-27B-GGUF",
    "quantization": "Q4_K_M",
    "base_url": "http://localhost:38589/v1",
    "timestamp": "2026-11-20T14:49:14.123456",
    "params": {"task": "code", "code_dataset": "multi", "n_code": 15, ...}
  }
}
```

Quick analysis pattern:
```python
import json, glob
for f in sorted(glob.glob("results/exp2_*unsloth*.json")):
    d = json.load(open(f))
    if isinstance(d, list): continue  # skip _outputs files
    meta = d.get("_meta", {})
    print(meta["model"], meta.get("quantization"), d.get("raw_agg") or d.get("raw_pass"))
```

## Constraints & Notes

- All models run locally via `llama-server` (GGUF quantized, Q4_K class)
- The LLM server must be running before the experiment script starts
- `fix_text.py` auto-detects the server too (same mechanism)
- LanguageTool is NOT currently running — need to start it for `--oracle languagetool`
- The `combined` oracle (wordlist+heuristic) is a fallback, not a serious baseline
- Phi-4 reasoning model outputs chain-of-thought → breaks the "return ONLY the sentence" instruction
  → For grammar tasks, non-reasoning models are more reliable
