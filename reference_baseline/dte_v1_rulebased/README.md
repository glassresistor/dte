# Diffusive Text Editing (DTE) — Prototype

Dependency-free implementation of the DTE mechanism from `REPORT_DTE.txt`.
Pure Python 3.8+ stdlib. No pip installs, no API keys, no GPU.

## Run

```bash
python -m dte.case1_grammar   # Case 1: grammar/spelling
python -m dte.case2_code      # Case 2: test-passing code
```

Both modules run clean, print a diff + metrics table, and pass built-in
assertions (acceptance criteria from §8.2 / §8.3).

## What it demonstrates

The core DTE mechanism (Appendix B pseudocode, made concrete):

1. **Informed corruption** — an oracle identifies *where* to touch (mask M).
   Non-target positions are structurally preserved by construction.
2. **Local denoising** — only masked positions are revised; non-masked
   tokens/lines are copied verbatim (structural locality, §2.3).
3. **Verification + iterative mask expansion** — re-run the oracle; on
   failure, expand the mask and iterate (bounded by edit budget B).

## File layout → concept-model mapping

| File | Concept section | Role |
|------|----------------|------|
| `core.py` | §3.4 (pipeline), §4.1 (metrics) | Pipeline, oracle/denoiser interfaces, metrics |
| `ed.py` | §4.1 (ED metric) | Token-level Levenshtein, diff helpers |
| `case1_grammar.py` | §8.2 | Rule-based grammar oracle + heuristic denoiser |
| `case2_code.py` | §8.3 | Test-suite oracle (subprocess+cache) + edit-grammar bandit |

### Case 1 components (§8.2)
- **Oracle**: deterministic rule-based checker → per-token error mask
  (spelling, double-space, missing punctuation, capitalization).
- **Corruptor**: masks exactly the flagged token indices (informed corruption).
- **Denoiser**: correction-table + local candidate selection; non-masked
  tokens copied verbatim.
- **Verifier**: re-run oracle; expand mask on fail (bounded iterations).
- **Metrics**: ED (token Levenshtein), edit precision, over-editing rate.

### Case 2 components (§8.3)
- **Oracle**: runs a real test suite via `subprocess`; caches results by
  content hash; parses failing lines from output.
- **Mask source**: AST function-line ranges (static analysis) + traceback
  line numbers. Fallback: all function-body lines when traceback points
  only at the test file.
- **Denoiser**: bounded edit-grammar candidates (off-by-one on integer
  literals, comparison/arithmetic operator swaps, missing-return insertion).
  Each candidate differs by exactly one line.
- **Bandit**: generate N candidates, run cached oracle on each, select the
  minimal-diff passing candidate. Cumulative: fixes accumulate across
  iterations until all tests pass or budget exhausted.
- **Metrics**: line-diff size, oracle-call count (cost), pass rate.

## Key design decisions

1. **Tokenization preserves whitespace runs** (Case 1): double-space is
   visible as a multi-char whitespace token. Non-masked tokens are copied
   verbatim → byte-exact reconstruction.

2. **Oracle and mask are perfectly consistent**: `ok(doc)` is True iff
   `mask(doc)` is empty. This guarantees the verification step is sound.

3. **Cumulative bandit** (Case 2): each iteration finds the best single-line
   improvement and *keeps it*. Fixes accumulate across iterations. This is
   necessary because real bugs are independent — no single-line fix can pass
   all tests when multiple functions have bugs.

4. **Graded oracle signal**: the test suite runs ALL checks (no early exit)
   and prints PASS/FAIL per check. The bandit uses failure count as a proxy
   for distance to passing, enabling hill-climbing through intermediate states.

5. **No fabricated results**: every number in the output is computed at
   runtime by an actual run in this interpreter.

## Acceptance criteria (from §8)

### Case 1 (§8.2)
- [x] (i) output passes the oracle
- [x] (ii) only masked tokens changed
- [x] (iii) ED == ground-truth error count (4)
- [x] (iv) all non-error tokens byte-identical

### Case 2 (§8.3)
- [x] (i) all tests pass
- [x] (ii) minimal line-diff among passing candidates (2 lines = ground truth)
- [x] (iii) oracle-call count reported

## Extension path (§8.4)

The prototype uses rule-based oracles and a heuristic denoiser. The documented
extension layer (not implemented here, requires external dependencies):
- Plug in a real embedding model for Strategy A (edit-distance penalty).
- Plug in an LLM as the denoiser with the mask as a hard constraint.
- Add the stochastic-corruption ablation (§7.4).
