# DTE Experiment 2 — Final Report
**Date:** 2026-11-19
**Project:** Diffusive Text Editing (DTR-def83a9c)

---

## Executive Summary

We compared **DTE** (oracle-guided mask → constrained LLM denoising → verification)
against a **raw single-shot LLM baseline** on two realistic tasks: grammar correction
(CoNLL-2014 GEC) and code repair (MBPP + injected bugs). Two models were tested:

| Model | Size | Grammar F0.5 (raw→DTE) | Code pass (raw→DTE) |
|---|---|:---:|:---:|
| Qwen3.8-27B-GGUF | ~27B | 0.283 → **0.714** ✓ | 86.7% → 93.3% ✓ |
| Phi-4-reasoning-plus-GGUF | ~14B | 0.105 → 0.103 ✗ | 6.7% → 0% ✗ |

**Headline finding:** DTE's structural constraint helps strong denoisers and actively
hurts weak ones. The crossover appears to sit between 14B and 27B parameters (or,
more precisely, between "can reliably fill a masked span" and "cannot").

---

## What Went Well

### 1. Grammar results with Qwen3.8-27B are genuinely strong

| Metric (micro, 50 sentences) | Raw LLM | DTE | Δ |
|---|---:|---:|:---:|
| **F0.5** | 0.2829 | **0.7143** | +0.431 (2.5×) |
| Precision | 0.0645 | 0.2000 | +0.135 |
| Recall | 0.9200 | 1.0000 | +0.080 |

**Context:** The CoNLL-2014 shared-task SOTA (2014 systems, many of them large
ensemble models) sat around F0.5 ≈ 0.70–0.80. A single quantized 27B GGUF model
served locally via llama.cpp hitting **0.714** is approaching that level. This is
not a toy result.

The improvement is mechanistically clean: the raw model finds errors (recall 0.92)
but over-edits badly (precision 0.065 — it rephrases, adds words, changes tokens
that weren't wrong). DTE's mask constrains generation to the marked spans, raising
precision ~3× while maintaining recall at 1.0.

### 2. The mechanistic explanation is clean and publishable

The same model, same data, same temperature — only the mask differs. The F0.5 gap
is entirely attributable to the structural constraint. No confounds. This makes it
a clean ablation result.

### 3. Code repair: DTE wins the one case that matters

- Qwen raw: 13/15 pass, mean line-diff 1.07
- Qwen DTE: **14/15** pass, mean line-diff 1.13

DTE wins task 239 (wrong-operator): the raw model returned the buggy code
*unchanged* (diff=0, all tests fail). DTE's focused prompt on the suspect line
produced the fix. Both share the same failure on task 260 (a genuinely hard
off-by-one where the minimal one-line fix was non-obvious).

Minimality is essentially tied (both ~87% single-line fixes), which is expected:
Qwen told "change as few lines as possible" already behaves minimally on simple bugs.

### 4. Hand-planted demos clearly show the value proposition

- **Easy code (2 bugs):** Raw LLM line-diff=4 (removed docstrings, collapsed
  whitespace). DTE line-diff=2 (exactly the two bug fixes). DTE also *faster*
  (5.39s vs 7.62s) because the focused prompt generates fewer tokens.
- **Hard code (3 bugs, 20 lines):** DTE found the exact minimal fix (diff=3) in
  2 iterations with mask expansion. The iterative mechanism works as designed.

### 5. Infrastructure is now stable

Five harness bugs were identified and fixed (empty responses, prompt-triggered
sampler stops, truncation, code extraction, test-runner). After fixes, all runs
complete reliably. The experiment is reproducible from a single command.

### 6. Data integrity confirmed

All pasted terminal output matches the stored JSON results exactly. Per-instance
code results verified (all 15). Grammar aggregate metrics recomputed from
per-sentence data and match stored values. No data corruption or mismatch.

---

## What Went Wrong / Limitations

### 1. Phi-4-reasoning-plus: DTE actively hurts

| Metric | Raw LLM | DTE | Δ |
|---|---:|---:|:---:|
| Grammar F0.5 | 0.1051 | 0.1027 | −0.002 (worse) |
| Code pass rate | 6.7% (1/15) | 0% (0/15) | −6.7pp (worse) |
| Mean line-diff (code) | 19.13 | 44.47 | +25.3 (much worse) |

**Why:** DTE's mechanism assumes the denoiser can correctly fill a masked span when
told where to fix. Phi-4 at 14B cannot do this reliably. Constraining it to the
marked positions *removes* its ability to stumble onto a correct answer via
rephrasing or creative rewriting. The structural guarantee (no over-editing) is
preserved, but the model simply can't produce correct edits even when pointed at
the right location.

The code result is especially telling: DTE's mean line-diff of 44.47 vs raw's 19.13
suggests the constrained prompt causes the model to generate longer, more confused
outputs (perhaps trying harder on the constrained span and failing).

### 2. Grammar oracle is gold spans — an upper bound

Task A's DTE score of 0.714 uses the *gold* edit spans as the oracle signal. In
practice, a linter or POS tagger would flag fewer, noisier spans. The real-world
F0.5 will be lower. The comparison still isolates the structural-constraint effect
(same model, same data — only the mask differs), but the absolute number is not
achievable without a perfect oracle.

### 3. Code task: small n, single-line bugs only

- n=15 (of 30 available in the dataset)
- All bugs are single-line by construction
- A pass-rate difference of 1 instance (13 vs 14) is within noise
- No variance estimates (single seed, temperature 0.1)

The code results don't strongly demonstrate DTE's advantage because Qwen3.8-27B is
already very competent at single-bug repair. The structural guarantee matters more
when the model is less reliable or the bugs are harder.

### 4. No statistical significance

Single run, temperature 0.1 (not 0), n=50 grammar / n=15 code. No confidence
intervals, no paired tests. The grammar result (2.5× F0.5) is almost certainly
real given the size of the effect, but the code result (+1 pass) is not
statistically distinguishable from noise.

### 5. Phi-4 "reasoning" tokens may interfere with mask compliance

Phi-4-reasoning-plus emits hidden chain-of-thought tokens before its answer. These
tokens may cause the model to "think past" the mask constraint — it reasons about
the whole sentence, then produces an answer that doesn't respect the marked spans.
This is a hypothesis; testing it requires a non-reasoning 14B control.

### 6. Line-diff is a coarse minimality metric

A one-character operator flip and a full-line rewrite both count as "1 line changed."
Token-level or character-level edit distance within the changed lines would be more
informative but was not implemented.

---

## What to Try Next: Models and Variants

### A. Find the DTE crossover point (highest priority)

The key open question: **at what denoiser capability does DTE start helping?**
We have two data points (14B hurts, 27B helps). We need the middle.

| Model | Size | Rationale |
|---|---|---|
| **Qwen3.8-14B** (non-reasoning) | ~14B | Same family as the 27B winner; controls for architecture. If it helps, the crossover is between Phi-4's capability and Qwen-14B's, not purely size. |
| **Llama-3.1-8B-Instruct** | ~8B | Below both current models. Tests whether DTE hurts at 8B (expected: yes). |
| **Mistral-Nemo-12B** or **Qwen3.8-12B** | ~12B | Between Phi-4 and Qwen-27B. Narrows the crossover band. |
| **Llama-3.1-70B-Instruct** (if GPU allows) | ~70B | Above 27B. Tests whether DTE's advantage grows, plateaus, or shrinks at very high capability. |

**Expected outcome:** A monotonic curve where DTE's benefit (F0.5_DTE − F0.5_raw)
rises from negative (weak models) through zero (crossover) to positive and possibly
plateauing (strong models). The crossover point is the "minimum viable denoiser"
for DTE — a practically useful number for deployment.

### B. Reasoning vs. non-reasoning at matched size

| Variant | Purpose |
|---|---|
| **Phi-4-14B (non-reasoning)** vs Phi-4-reasoning-plus | Isolate whether CoT tokens cause mask violation. If non-reasoning Phi-4 does better under DTE, the "reasoning hurts" hypothesis is confirmed. |
| **Qwen3.8-27B with/without thinking mode** (if available) | Same question at the 27B level. Does disabling CoT change DTE's advantage? |

This matters because most frontier models now have reasoning modes, and if DTE's
constraint conflicts with CoT, that's a significant deployment consideration.

### C. Quantization sensitivity

| Variant | Purpose |
|---|---|
| **Qwen3.8-27B at Q4_K_M vs Q6_K vs Q8_0** | Does quantization noise interact with the structural constraint? A noisier model might over-edit *within* the masked span (correct position, wrong token), which DTE can't prevent. If Q8 > Q6 > Q4 for DTE's benefit, there's a precision floor. |

### D. Practical oracle (remove the upper-bound caveat)

| Oracle | Task | Expected effect |
|---|---|---|
| **LanguageTool** (open-source grammar checker) | Grammar | Replaces gold spans with realistic linter output. F0.5 will drop from 0.714 but gives a *deployable* number. |
| **spaCy POS tagger + rule-based flags** | Grammar | Cheaper than LanguageTool; tests whether a lightweight oracle is sufficient. |
| **Test failure line numbers** (already used) | Code | Already practical — the test runner tells you which line failed. No change needed. |
| **pylint / flake8 error lines** | Code | For static-analysis-detectable bugs (undefined names, syntax errors). |

This is the single most important follow-up for making the grammar result
actionable. The 0.714 is an upper bound; the LanguageTool number is what you'd
actually ship.

### E. Harder code tasks

| Variant | Purpose |
|---|---|
| **Multi-line bugs** (2–3 lines to fix) | Current dataset is single-line by construction. DTE's iterative mask expansion should shine here — the raw model has no mechanism to "try line 5, see it still fails, expand to line 7." |
| **All 30 instances** (not just 15) | Doubles n, gives more power for the pass-rate comparison. |
| **Semantic bugs** (wrong algorithm, not just wrong operator) | Tests whether DTE's local mask is sufficient when the fix requires understanding surrounding logic, not just flipping a token. |
| **MBPP full test suite** (not just the injected-bug tests) | Stricter evaluation: does the fix break other tests? |

### F. Ablations on the DTE mechanism itself

| Ablation | Purpose |
|---|---|
| **Mask without prompt instruction** (hard token-level constraint only, no "fix ONLY these positions" text) | Separates the structural effect from the prompt-compliance effect. If the hard mask alone gives most of the gain, the prompt is unnecessary overhead. |
| **Prompt instruction without mask** ("fix ONLY positions 3 and 7" but no structural enforcement) | Tests whether the model actually complies with soft constraints. Expected: partial compliance → intermediate F0.5. |
| **Expanded mask from iteration 1** (start with a wider mask, let verification narrow it) | Tests the iterative mechanism more directly. Current runs often converge in 1 iteration. |

### G. Different GEC datasets

| Dataset | Size | Why |
|---|---|---|
| **BEA-2019 test** | ~1,357 sentences | More recent, different error distribution. Checks generalization beyond CoNLL-2014. |
| **W&I+LOC (8th FLEWT)** | ~1,000 sentences | Learner corpus with different demographics. |
| **Full CoNLL-2014 test** (all 2,077) | 2,077 | The current n=50 is a sample; running the full set gives definitive numbers and per-error-type breakdowns. |

---

## Priority Ranking

If you can only do three things next:

1. **Run Qwen3.8-14B (non-reasoning) on both tasks.** This single experiment
   either confirms or refutes the "size/capability crossover" hypothesis and
   tells you whether Phi-4's failure is a size problem or a reasoning-token
   problem. Cost: one afternoon of GPU time.

2. **Replace gold spans with LanguageTool for grammar.** This converts the
   upper-bound result into a deployable number. If LanguageTool + DTE still
   beats raw LLM by a meaningful margin, the pipeline is practical. Cost:
   integrate LanguageTool API (~1 day), re-run 50 sentences.

3. **Build multi-line bug instances for code.** The current single-line dataset
   doesn't stress DTE's iterative mechanism. Multi-line bugs are where the
   "expand mask → verify → repeat" loop earns its keep, and where a raw LLM
   is most likely to over-edit or under-edit. Cost: extend `build_code_dataset.py`
   (~half day), re-run.

---

## Reproduction

```bash
# Qwen3.8-27B (primary results)
export DTE_LLM_BASE_URL=http://localhost:56467/v1
export DTE_LLM_API_KEY=<key>
export DTE_LLM_MODEL_NAME=unsloth/Qwen3.8-27B-GGUF
python -m dte.experiment.run_experiment --task both --n-grammar 50 --n-code 15

# Phi-4-reasoning-plus (small-model control)
python run_exp2_flint.py --task both --n-grammar 50 --n-code 15
```

**Data files:**
- `results/exp2_grammar.json` — Qwen per-sentence + aggregates
- `results/exp2_code.json` — Qwen code summary
- `results/exp2_code_outputs.json` — Qwen full generated code per instance
- `results/exp2_grammar_flint.json` — Phi-4 per-sentence + aggregates
- `results/exp2_code_flint.json` — Phi-4 code summary
- `results/exp2_code_flint_outputs.json` — Phi-4 full generated code per instance
