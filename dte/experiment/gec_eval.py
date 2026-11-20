"""GEC evaluation for Experiment 2 (CoNLL-2014).

Scores a system's corrected sentence against the gold edit set using
ERRANT-style span matching:
  * A predicted edit is a token span [s:e) that differs from the original.
  * An edit is "correct" if it matches a gold edit (same span + same
    replacement), or overlaps one (partial credit via F0.5).

We implement the standard MaxMatch / ERRANT F0.5 metric:
  - Extract predicted edits by aligning original vs. corrected token lists.
  - Match each gold edit to at most one predicted edit (MaxMatch greedy).
  - F0.5 = 0.25*P + 0.75*R  (precision weighted 2x, per GEC convention).

This is the same metric family used in the CoNLL-2014 / BEA-2019 shared tasks.
"""
from __future__ import annotations

import difflib
from typing import Dict, List


def tokenize(text: str) -> List[str]:
    """Whitespace tokenization (matches how CoNLL M2 stores tokens)."""
    return text.split()


def extract_predicted_edits(orig_tokens: List[str], pred_tokens: List[str]) -> List[Dict]:
    """Find spans where the prediction differs from the original.

    Returns a list of {"start": s, "end": e, "orig": [...], "pred": [...]}
    using SequenceMatcher opcodes (insert/delete/replace).
    """
    sm = difflib.SequenceMatcher(a=orig_tokens, b=pred_tokens, autojunk=False)
    edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        edits.append({
            "start": i1, "end": i2,
            "orig": orig_tokens[i1:i2],
            "pred": pred_tokens[j1:j2],
        })
    return edits


def _edit_key(e: Dict) -> tuple:
    return (e["start"], e["end"], tuple(e["orig"]), tuple(e["pred"]))


def maxmatch_f05(orig_tokens: List[str], pred_tokens: List[str],
                 gold_edits: List[Dict]) -> Dict:
    """Compute F0.5, precision, recall via MaxMatch against gold edits."""
    predicted = extract_predicted_edits(orig_tokens, pred_tokens)

    # A predicted edit is a "full match" if some gold edit has the same span
    # AND the same replacement tokens. Overlap-only matches count as partial.
    matched_pred = set()
    matched_gold = set()
    full_matches = 0

    # Greedy MaxMatch: pair edits that share a token span.
    for pi, pe in enumerate(predicted):
        for gi, ge in enumerate(gold_edits):
            if gi in matched_gold:
                continue
            # span overlap?
            if pe["start"] < ge["end"] and ge["start"] < pe["end"]:
                matched_pred.add(pi)
                matched_gold.add(gi)
                if (pe["start"] == ge["start"] and pe["end"] == ge["end"]
                        and tuple(pe["pred"]) == tuple(ge.get("pred_tokens", []))):
                    full_matches += 1
                break

    n_pred = len(predicted)
    n_gold = len(gold_edits)
    precision = (full_matches / n_pred) if n_pred else 0.0
    recall = (len(matched_gold) / n_gold) if n_gold else 0.0
    f05 = 0.0 if (2 * precision + recall) == 0 else (5 * precision * recall) / (2 * precision + recall)

    return {
        "f05": round(f05, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "n_predicted_edits": n_pred,
        "n_gold_edits": n_gold,
        "full_matches": full_matches,
    }


def evaluate_sentence(original: str, corrected: str, gold_edits: List[Dict]) -> Dict:
    """Evaluate one sentence. gold_edits items need 'start','end','pred_tokens'."""
    orig_toks = tokenize(original)
    pred_toks = tokenize(corrected)
    # attach replacement tokens to gold edits for full-match checking
    g = []
    for e in gold_edits:
        ge = dict(e)
        if "pred_tokens" not in ge:
            ge["pred_tokens"] = e.get("corr", "").split()
        g.append(ge)
    return maxmatch_f05(orig_toks, pred_toks, g)


def aggregate(results: List[Dict]) -> Dict:
    """Micro-average F0.5 across sentences (standard GEC reporting)."""
    tp = sum(r["full_matches"] for r in results)
    fp = sum(r["n_predicted_edits"] - r["full_matches"] for r in results)
    fn = sum(r["n_gold_edits"] - r["recall"] * r["n_gold_edits"] for r in results)
    # recompute properly from per-sentence counts
    total_pred = sum(r["n_predicted_edits"] for r in results)
    total_gold = sum(r["n_gold_edits"] for r in results)
    total_matched_gold = sum(round(r["recall"] * r["n_gold_edits"]) for r in results)
    precision = tp / total_pred if total_pred else 0.0
    recall = total_matched_gold / total_gold if total_gold else 0.0
    f05 = 0.0 if (2 * precision + recall) == 0 else (5 * precision * recall) / (2 * precision + recall)
    return {
        "micro_f05": round(f05, 4),
        "micro_precision": round(precision, 4),
        "micro_recall": round(recall, 4),
        "n_sentences": len(results),
        "total_gold_edits": total_gold,
    }
