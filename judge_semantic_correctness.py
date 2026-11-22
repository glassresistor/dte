#!/usr/bin/env python3
"""Semantic-correctness LLM judge for the "high recall / low MaxMatch precision" grammar cases.

Motivation: Qwen/Gemma DTE+raw runs on CoNLL-2014 with the `combined` oracle show
recall ~85-90% but near-zero F0.5/precision under strict string matching (MaxMatch).
Two competing explanations for that gap:
  (a) The model's edits are genuinely wrong, OR
  (b) MaxMatch is too literal — the fix touches the right span and is semantically fine,
      just not word-for-word identical to what CoNLL-2014 annotators wrote.

This script re-runs a small sample of sentences through BOTH arms exactly as
run_exp2_flint.py does (same prompts, same oracle), captures the actual output text
(which is NOT persisted in normal result files — only scores are saved there), and asks
an LLM to judge each arm's output independently: "is this semantically correct even if it
doesn't string-match gold?"

CAVEAT: uses the SAME local model as generator for judging (no separate big-model server
available here). This is a diagnostic tool, not ground truth — read the printed side-by-side
text and verify a few judgments by eye before trusting aggregate numbers.

Usage:
  python judge_semantic_correctness.py [--n-sentences 10] [--oracle combined] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import sys

sys.path.insert(0, ".")

# Reuse the exact same connection setup + prompt templates as the real experiment so this
# test corresponds to the actual numbers already in results/, not a slightly different run.
from dte.experiment.run_experiment import (
    make_llm, llm_complete, raw_llm_grammar, load_gec,
)
from dte.experiment.gec_eval import evaluate_sentence
from dte.experiment.oracle import get_oracle_spans
from run_exp2_flint import detect_llama_server, dte_grammar_with_oracle

# ---------------------------------------------------------------------------
# Sampling: pick a mix of easy (1 error) and harder (>=2 errors) sentences so we're not
# just looking at trivial single-token cases.
# ---------------------------------------------------------------------------
def sample_sentences(n_total, seed):
    with open("data/conll2014_gec.json") as f:
        rows = json.load(f)
    easy   = [r for r in rows if r["n_errors"] == 1]
    harder = [r for r in rows if r["n_errors"] >= 2]
    rng = random.Random(seed)
    n_easy = max(1, n_total // 2)
    picked = (rng.sample(easy, min(n_easy, len(easy))) +
              rng.sample(harder, min(n_total - n_easy, len(harder))))
    return picked

# ---------------------------------------------------------------------------
# Judge prompt — deliberately asks for a per-arm verdict with reasoning so we can audit it.
# Shows original / gold-reference / each candidate output explicitly and separately to avoid
# the judge just pattern-matching against one another rather than judging independently.
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are an expert English grammar evaluator.

ORIGINAL (contains errors): {original}
REFERENCE CORRECTION:       {gold}

Candidate A was produced by asking a language model to "fix the grammar" freely, with no hints about which words were wrong.
Candidate B was produced under a constraint that only specific flagged tokens may be changed; other words must stay identical to ORIGINAL.

CANDIDATE A (raw):  {cand_a}
CANDIDATE B (DTE):  {cand_b}

For EACH candidate independently, decide: does it fix the errors present in ORIGINAL without introducing new grammatical or factual errors? It is acceptable for a correct answer to use slightly different wording than REFERENCE CORRECTION as long as meaning and grammar are right.

Respond with EXACTLY two lines, no other text:
A=CORRECT  (or A=INCORRECT) <one short reason>
B=CORRECT  (or B=INCORRECT) <one short reason>"""


def _parse_verdict_line(line):
    """Parse a single 'A=CORRECT <reason>' / 'B=INCORRECT <reason>' style line."""
    import re
    m = re.match(r'^\s*([AB])\s*=\s*(CORRECT|INCORRECT)\b(.*)$', line.strip(), re.IGNORECASE)
    if not m:
        return None, False, "unparsed: %r" % line[:60]
    label = m.group(1).upper()
    ok = (m.group(2).lower() == "correct")
    reason = m.group(3).strip(" :-|") 
    return label, ok, reason


def judge(client, model, original, gold, cand_a, cand_b):
    prompt = JUDGE_PROMPT.format(original=original, gold=gold, cand_a=cand_a, cand_b=cand_b)
    out, _t = llm_complete(client, model, prompt, max_tokens=128)
    verdicts = {"A": None, "B": None}
    for line in out.splitlines():
        label, ok, reason = _parse_verdict_line(line)
        if label:
            verdicts[label] = {"ok": ok, "reason": reason}
    return out.strip(), verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sentences", type=int, default=10)
    ap.add_argument("--oracle", default="combined")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base_url, api_key, model_name, quant = detect_llama_server()
    import os
    os.environ["DTE_LLM_BASE_URL"] = base_url
    os.environ["DTE_LLM_API_KEY"] = api_key or ""
    client, model = make_llm()
    print("Model: %s (%s) @ %s\n" % (model, quant or "?", base_url))

    rows = sample_sentences(args.n_sentences, args.seed)
    results = []
    for i, row in enumerate(rows):
        original = row["original"]
        gold     = row["gold"]
        gold_edits = json.loads(row["edits"])

        r_out, _t1 = raw_llm_grammar(client, model, original)
        oracle_spans = get_oracle_spans(original, method=args.oracle)
        d_out, _t2 = dte_grammar_with_oracle(client, model, original, oracle_spans)

        r_res = evaluate_sentence(original, r_out, gold_edits)
        d_res = evaluate_sentence(original, d_out, gold_edits)

        print("=" * 90)
        print("[%d/%d] id=%s n_errors=%d" % (i + 1, len(rows), row["id"], row["n_errors"]))
        print("ORIGINAL : %s" % original)
        print("GOLD     : %s" % gold)
        print("---")
        print("RAW out  : %s   [F0.5=%.2f P=%.2f R=%.2f]" % (r_out, r_res["f05"], r_res["precision"], r_res["recall"]))
        print("DTE out  : %s   [F0.5=%.2f P=%.2f R=%.2f]  spans=%d" % (d_out, d_res["f05"], d_res["precision"], d_res["recall"], len(oracle_spans)))

        judge_raw_text, verdicts = judge(client, model, original, gold, r_out, d_out)
        print("---")
        print("LLM JUDGE said:\n%s" % judge_raw_text)
        results.append({
            "id": row["id"], "original": original, "gold": gold,
            "raw_out": r_out, "dte_out": d_out,
            "r_f05": r_res["f05"], "d_f05": d_res["f05"],
            "judge_raw_text": judge_raw_text, "verdicts": verdicts,
        })

    # --- Aggregate -------------------------------------------------------
    def _count(verdict_key):
        total = sum(1 for r in results if r["verdicts"][verdict_key] is not None)
        correct = sum(1 for r in results 
                      if r["verdicts"][verdict_key] and r["verdicts"][verdict_key]["ok"])
        return total, correct

    print("\n" + "=" * 90)
    print("SUMMARY (LLM-judged semantic correctness; n=%d sentences)" % len(results))
    a_total, a_correct = _count("A")
    b_total, b_correct = _count("B")
    # Also: how many of the MaxMatch-failed cases did the judge still call correct?
    def _failed_but_ok(verdict_key):
        failed_cases = [r for r in results if (verdict_key == "A" and r["r_f05"] < 1.0) 
                                            or (verdict_key == "B" and r["d_f05"] < 1.0)]
        ok_among_failed = sum(1 for r in failed_cases
                              if r["verdicts"][verdict_key] and r["verdicts"][verdict_key]["ok"])
        return len(failed_cases), ok_among_failed
    a_fail_n, a_fail_ok = _failed_but_ok("A")
    b_fail_n, b_fail_ok = _failed_but_ok("B")
    print("  RAW arm : %d/%d judged semantically correct overall" % (a_correct, a_total))
    print("            of those with F0.5<1 under MaxMatch: %d/%d still called CORRECT by judge"
          "   <-- this is the 'MaxMatch too strict?' signal" % (a_fail_ok, a_fail_n))
    print("  DTE arm : %d/%d judged semantically correct overall" % (b_correct, b_total))
    print("            of those with F0.5<1 under MaxMatch: %d/%d still called CORRECT by judge"
          "   <-- same question for the constrained arm" % (b_fail_ok, b_fail_n))

if __name__ == "__main__":
    main()
