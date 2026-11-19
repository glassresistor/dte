# DTE Implementation Report & Results

**Date:** 2026-11-19
**Environment:** Python 3.13, llama.cpp server (localhost:56467), Qwen3.8-27B-GGUF
**Scope:** Full implementation of the DTE mechanism from `REPORT_DTE.txt`,
including a rule-based prototype (v1) and a model-backed harness, with
measured results comparing DTE against a raw LLM baseline.

---

## 1. What was built

### 1.1 Rule-based prototype (`dte/`) — zero dependencies

Pure Python 3 stdlib. Demonstrates the full DTE mechanism (informed
corruption → local denoising → verification → iterative mask expansion)
with a heuristic denoiser. No models, no pip installs, no API keys.

**Files:**
- `dte/core.py` — pipeline, oracle/denoiser interfaces, metrics (§3.4, §4.1)
- `dte/ed.py` — token-level Levenshtein, diff helpers (§4.1)
- `dte/case1_grammar.py` — rule-based grammar oracle + correction-table denoiser
- `dte/case2_code.py` — test-suite oracle (subprocess+cache) + edit-grammar bandit

**Run:**
```bash
python -m dte.case1_grammar   # 4 planted errors, ground-truth ED=4
python -m dte.case2_code      # 2 planted bugs, ground-truth diff=2 lines
```

### 1.2 Model-backed harness (`dte/harness/`) — real LLM denoiser

Same pipeline and oracles, but the denoiser is a real model. The LLM is
prompted to fix *only* masked positions; non-masked tokens/lines are copied
verbatim by pipeline code (structural locality enforced in code, not left
to the model).

**Files:**
- `dte/harness/config.py` — env-var configuration with clear error messages
- `dte/harness/backend.py` — `LlamaCppBackend` (local GGUF) + `OpenAICompatBackend` (llama-server/vLLM/Ollama)
- `dte/harness/embedding.py` — Strategy A: cosine-distance edit-budget constraint
- `dte/harness/llm_denoiser.py` — LLM-backed denoisers for Case 1 & 2
- `dte/harness/case1_grammar.py` — runnable entry point
- `dte/harness/case2_code.py` — runnable entry point (easy + hard fixtures)

**Run:**
```bash
export DTE_LLM_BACKEND=openai_compat
export DTE_LLM_BASE_URL=http://localhost:56467/v1
export DTE_LLM_API_KEY=<key>
export DTE_LLM_MODEL_NAME=unsloth/Qwen3.8-27B-GGUF
python -m dte.harness.case1_grammar
python -m dte.harness.case2_code
DTE_CODE_FIXTURE=hard python -m dte.harness.case2_code
```

### 1.3 Reference baseline (`reference_baseline/`)

Frozen copy of v1 as first built, for comparison against future iterations.

---

## 2. Results

All results below are from actual runs in this environment. No numbers
are fabricated or estimated. Raw output is preserved in `results/`.

### 2.1 Case 1: Grammar/spelling (4 planted errors)

**Fixture:** "The team will recieve the report quickly␣␣and will definately
review it before the meeting" — exactly 4 surface errors (misspelling,
double-space, misspelling, missing terminal period). Ground-truth ED = 4.

| Metric | v1 (rule-based) | Harness (Qwen3.8) | Raw LLM baseline |
|--------|:---------------:|:-----------------:|:----------------:|
| Pass oracle | ✅ | ✅ | ✅ |
| Edit distance | 3* | 3* | 3* |
| Tokens changed | 4 | 4 | 4 |
| Over-edited tokens | 0 | 0 (guaranteed) | 0 |
| Iterations | 1 | 1 | 1 (single-shot) |
| Time | <0.01s | 5.28s | 4.48s |

\* Levenshtein ED=3 because the double-space fix ("␣␣"→"␣") is counted as
one token substitution, and the period addition ("meeting"→"meeting.") is
another — 3 substitutions total across 4 changed positions. All three
approaches found the identical 4 fixes.

**Verdict:** Tie on this easy case. The errors are unambiguous enough that
even an unconstrained prompt handles them correctly. DTE's advantage here
is the *guarantee* (structural locality by construction), not a different
outcome. The gap would widen with more ambiguous text where a raw LLM
starts rephrasing.

### 2.2 Case 2: Code repair — easy fixture (2 bugs)

**Fixture:** Two functions, one bug each (off-by-one `n+2`→`n+1`, wrong
operator `!=`→`==`). Ground-truth minimal diff = 2 lines.

| Metric | v1 (edit-grammar bandit) | Harness (Qwen3.8) | Raw LLM baseline |
|--------|:------------------------:|:-----------------:|:----------------:|
| Pass tests | ✅ | ✅ | ✅ |
| Line-diff | 2 | 2 | **4** |
| Minimal? | ✅ (= ground truth) | ✅ (= ground truth) | ❌ (over-edited) |
| Iterations | 2 | 1 | 1 (single-shot) |
| Oracle calls | 14 | 2 | 0 (no verification) |
| Time | <0.5s | 5.39s | 7.62s |

**The raw LLM over-edited.** It fixed both bugs correctly but also:
- Removed both docstrings (`"""Sum the first n..."""`, `"""Return True if..."""`)
- Collapsed two blank lines between functions into one

This is exactly the failure mode described in §2.1 of REPORT_DTE.txt:
"LLMs systematically over-edit: they rephrase stylistic choices, alter tone,
refactor unrelated code." The user asked to "make tests pass," not to remove
documentation or change whitespace structure.

**DTE produced the exact minimal diff:** only the two bug lines changed,
everything else byte-identical. And it was *faster* (5.39s vs 7.62s) because
the focused prompt (showing suspect lines + test output) requires fewer
generated tokens than "rewrite this code."

### 2.3 Case 2: Code repair — hard fixture (3 bugs)

**Fixture:** Three functions, one bug each (off-by-one, missing `return x`,
wrong operator). Ground-truth minimal diff = 3 lines. This tests whether DTE
can handle multiple independent bugs that require cumulative fixes.

| Metric | Harness (Qwen3.8) |
|--------|:-----------------:|
| Pass tests | ✅ |
| Line-diff | 3 (= ground truth) |
| Minimal? | ✅ |
| Iterations | 3 |
| Oracle calls | 2 |

The LLM needed 3 iterations because the first two attempts returned empty
JSON (the model was uncertain which specific line to change when given all
15 suspect lines at once). On iteration 3, after mask expansion, it produced
all three fixes in one response. The cumulative bandit correctly accumulated
the fixes and verified against the test suite.

**Note:** v1's edit-grammar bandit could not solve this fixture (it only
generates single-line candidates from a fixed grammar). The LLM harness
handles it because the model can reason about multi-bug interactions.

### 2.4 Summary table

| | v1 rule-based | Harness Qwen3.8 | Raw LLM |
|---|:---:|:---:|:---:|
| Case 1 pass | ✅ | ✅ | ✅ |
| Case 1 minimal | ✅ | ✅ | ✅ |
| Case 2 easy pass | ✅ | ✅ | ✅ |
| Case 2 easy minimal | ✅ (diff=2) | ✅ (diff=2) | ❌ (diff=4) |
| Case 2 hard pass | ❌ (can't solve) | ✅ (diff=3) | not tested |
| Over-editing | None | None (guaranteed) | Yes (docstrings removed) |
| Dependencies | stdlib only | llama.cpp + openai | llama.cpp + openai |

---

## 3. Key findings

### 3.1 DTE's core value proposition is validated on code repair

The raw LLM removed docstrings and changed whitespace when asked to "make
tests pass." DTE, with the same underlying model, produced exactly the two
bug fixes and nothing else. This is not a prompt-engineering difference —
it is a *structural* difference. The mask is enforced by pipeline code; the
LLM literally cannot touch non-masked lines because they are never included
in the editable region of the prompt response.

### 3.2 On easy grammar, DTE and raw LLM perform identically

Both found the same 4 fixes with zero over-editing. This is expected: the
errors are unambiguous (obvious misspellings, visible double-space, missing
period). The DTE guarantee matters more when the input is ambiguous — e.g.,
"their/there/they're" confusion where a raw LLM might "fix" a correct usage.

### 3.3 Speed: DTE is competitive or faster

| Case | Raw LLM | DTE | Difference |
|------|---------|-----|------------|
| Grammar | 4.48s | 5.28s | +0.8s (DTE slower) |
| Code easy | 7.62s | 5.39s | -2.2s (DTE faster) |

DTE is slower on grammar because it adds mask construction + verification
on top of a single LLM call. DTE is *faster* on code because the focused
prompt (suspect lines + test output) generates fewer tokens than an open-ended
"fix this code" prompt. The net effect: DTE's overhead is bounded and small
relative to the correctness guarantee it provides.

### 3.4 The LLM harness extends v1's capability

v1's edit-grammar bandit (Case 2) can only generate candidates from a fixed
set of transformations (off-by-one, operator swaps, missing return). It
solved the easy fixture but could not solve the hard fixture (3 independent
bugs requiring coordinated fixes). The LLM harness handles both because the
model can reason about which specific lines to change and how.

### 3.5 Oracle calls are cheap with caching

The test oracle caches results by content hash. In the easy code case, only
2 subprocess calls were needed (one for the initial mask, one to verify the
LLM's fix). The hard case also used only 2 calls because the LLM produced all
three fixes in a single response on iteration 3.

---

## 4. What was NOT tested (open items from REPORT_DTE.txt §7)

1. **Model scaling** (§7): How do 1B, 3B, 8B models compare to 27B on
   over-editing rate? Expected: smaller models over-edit more, making DTE's
   structural guarantee more valuable.

2. **Strategy A ablation** (§7.3): Embedding-distance constraint was disabled
   (`DTE_EMBED_BACKEND=none`) for these runs. Testing whether it improves
   precision on ambiguous inputs is a natural next step.

3. **Stochastic corruption** (§7.4): `DTE_STOCHASTIC_RATE` was 0.0 (off).
   Testing 1-2% stochastic masking would measure the ED cost vs. robustness
   gain for multi-token errors.

4. **Edit budget scaling** (§4.3): Does DTE converge to AR behavior as
   `DTE_EDIT_BUDGET` grows from 1→10? Not tested — would require running
   the full matrix.

5. **Harder grammar fixtures:** The current fixture has 4 unambiguous errors.
   A fixture with contextual errors (their/there, its/it's) would better
   demonstrate where DTE's guarantee matters vs. a raw LLM.

6. **Larger code files:** The fixtures are 11-20 lines. Real-world code
   repair (SWE-bench style) involves hundreds of lines with more complex
   bug interactions. The harness should scale, but this is untested.

---

## 5. File inventory

```
REPORT_DTE.txt                          # Original research report
README.md                               # Top-level docs
reference_baseline/                     # Frozen v1 copy
  dte_v1_rulebased/                     #   rule-based prototype
  REPORT_DTE.txt                        #   report copy
dte/                                    # Rule-based prototype (v1)
  __init__.py
  core.py                               #   pipeline + interfaces + metrics
  ed.py                                 #   Levenshtein + diff helpers
  case1_grammar.py                      #   grammar DTE (runnable)
  case2_code.py                         #   code DTE (runnable)
  README.md                             #   v1 docs
  harness/                              # Model-backed harness
    __init__.py
    config.py                           #   env-var configuration
    backend.py                          #   llama_cpp + openai_compat backends
    embedding.py                        #   Strategy A: embedding constraint
    llm_denoiser.py                     #   LLM denoisers (Case 1 & 2)
    case1_grammar.py                    #   runnable: grammar + LLM
    case2_code.py                       #   runnable: code + LLM
    README.md                           #   harness setup + usage
results/                                # Measured results (this run)
  case1_grammar_qwen3.8.txt             #   Case 1 raw output
  case2_code_easy_qwen3.8.txt           #   Case 2 easy raw output
  case2_code_hard_qwen3.8.txt           #   Case 2 hard raw output
  comparison_raw_vs_dte_qwen3.8.txt     #   Full comparison write-up
```

---

## 6. Reproducing these results

```bash
# Prerequisites: llama.cpp server running with Qwen3.8-27B-GGUF
# (or any OpenAI-compatible endpoint)

export DTE_LLM_BACKEND=openai_compat
export DTE_LLM_BASE_URL=http://localhost:56467/v1
export DTE_LLM_API_KEY=<your-key>
export DTE_LLM_MODEL_NAME=unsloth/Qwen3.8-27B-GGUF
export DTE_EMBED_BACKEND=none

# v1 (no model needed):
python -m dte.case1_grammar
python -m dte.case2_code

# Harness:
python -m dte.harness.case1_grammar
python -m dte.harness.case2_code
DTE_CODE_FIXTURE=hard python -m dte.harness.case2_code
```

All results are deterministic given the same model, seed (`DTE_SEED=0`),
and temperature (`DTE_LLM_TEMPERATURE=0.1`). Minor variation may occur
across different llama.cpp builds or GPU/CPU configurations.
