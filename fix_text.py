#!/usr/bin/env python
"""Fix My Document — cheap local grammar/text correction harness.

Takes a text file (or stdin), detects errors per-sentence using a pluggable
oracle, applies DTE-constrained fixes only to flagged spans, and outputs the
corrected text with an inline diff report.

Usage:
  python fix_text.py document.txt
  cat notes.md | python fix_text.py -
  python fix_text.py doc.txt --oracle llm --output fixed.txt --diff

Oracles:
  languagetool — local LanguageTool server (most precise, free)
  llm          — use the same model as a detector (no extra infra needed)
  combined     — wordlist + heuristic fallback (offline, no network)

The DTE fix prompt constrains the model to change ONLY the flagged tokens,
preventing over-editing while still correcting real errors.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from typing import List, Dict, Tuple

# Reuse infrastructure from the experiment package
SANDBOX = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SANDBOX)

from dte.experiment.run_experiment import make_llm, llm_complete
from dte.experiment.oracle import get_oracle_spans


def split_sentences(text: str) -> List[Dict]:
    """Split text into sentences with character offsets.
    
    Returns list of {"text": str, "start": int, "end": int}
    where start/end are char offsets in the original text.
    """
    # Match sentence endings followed by whitespace or end-of-string
    pattern = re.compile(r'([^.!?]*[.!?]+["\')\]]*\s*)')
    sentences = []
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            # Gap between sentences (whitespace, newlines)
            pass
        sent_text = m.group(1)
        if sent_text.strip():
            sentences.append({
                "text": sent_text,
                "start": m.start(),
                "end": m.end(),
            })
        pos = m.end()
    
    # Handle trailing text without sentence-ending punctuation
    if pos < len(text) and text[pos:].strip():
        sentences.append({
            "text": text[pos:],
            "start": pos,
            "end": len(text),
        })
    
    return sentences


def dte_fix_sentence(client, model: str, sentence: str, spans: List[Dict]) -> Tuple[str, float]:
    """Apply DTE-constrained fix to a sentence using flagged spans."""
    toks = sentence.split()
    if not spans:
        return sentence, 0.0

    lines = ["You are a precise grammar editor. Fix ONLY the marked spans.",
             "Do NOT change any other words or rephrase.", "",
             f"Sentence: {sentence}", "",
             "Marked error spans (token indices, 0-based):"]
    for s in spans:
        span = " ".join(toks[s["start"]:s["end"]]) if s["end"] <= len(toks) else " ".join(toks[s["start"]:])
        lines.append(f"  tokens [{s['start']}:{s['end']}] = {span!r}")
    lines += ["",
              "Respond with ONLY the fully corrected sentence.",
              "Change only the marked spans; keep everything else identical."]
    prompt = "\n".join(lines)
    out, t = llm_complete(client, model, prompt, max_tokens=256)
    out = out.strip().strip('"').strip()
    return out, t


def raw_fix_sentence(client, model: str, sentence: str) -> Tuple[str, float]:
    """Baseline: unconstrained grammar fix."""
    prompt = ("Fix the grammatical errors in this sentence. "
              "Return ONLY the corrected sentence, nothing else.\n\n" + sentence)
    out, t = llm_complete(client, model, prompt, max_tokens=256)
    out = out.strip().strip('"').strip()
    return out, t


def diff_line(original: str, corrected: str) -> List[str]:
    """Produce a unified-diff-style report for one sentence."""
    if original == corrected:
        return []
    lines = []
    op = difflib.SequenceMatcher(None, original, corrected).get_opcode()
    return [f"  - {original}", f"  + {corrected}"]


def main():
    ap = argparse.ArgumentParser(description="Fix My Document — local grammar correction with DTE")
    ap.add_argument("input", help="Input text file, or '-' for stdin")
    ap.add_argument("--oracle", choices=["languagetool", "llm", "combined", "wordlist", "heuristic"],
                    default="languagetool",
                    help="Error detection method (default: languagetool)")
    ap.add_argument("--lt-url", default="http://localhost:8823",
                    help="LanguageTool server URL")
    ap.add_argument("--mode", choices=["dte", "raw", "both"], default="dte",
                    help="Fix mode: dte (constrained), raw (unconstrained), both (compare)")
    ap.add_argument("--output", default=None,
                    help="Write corrected text to this file (default: stdout)")
    ap.add_argument("--diff", action="store_true",
                    help="Print a diff report of all changes")
    ap.add_argument("--json", action="store_true",
                    help="Output structured JSON instead of plain text")
    ap.add_argument("--base-url", default=None, help="Override LLM server URL")
    ap.add_argument("--model", default=None, help="Override model name")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="Skip sentences where oracle flags fewer than this fraction of tokens (0-1)")
    args = ap.parse_args()

    # --- Load input text ---
    if args.input == "-":
        text = sys.stdin.read()
    else:
        with open(args.input) as f:
            text = f.read()

    if not text.strip():
        print("ERROR: empty input", file=sys.stderr)
        sys.exit(1)

    # --- Connect to LLM ---
    base_url = args.base_url or os.environ.get("DTE_LLM_BASE_URL", "")
    api_key = os.environ.get("DTE_LLM_API_KEY", "")
    model_name = args.model or os.environ.get("DTE_LLM_MODEL_NAME", "")

    if not base_url:
        print("Auto-detecting llama-server...", file=sys.stderr)
        from run_exp2_flint import detect_llama_server
        base_url, api_key, model_name, _quant = detect_llama_server()

    os.environ["DTE_LLM_BASE_URL"] = base_url
    os.environ["DTE_LLM_API_KEY"] = api_key
    os.environ["DTE_LLM_MODEL_NAME"] = model_name

    client, model = make_llm()
    print(f"Model: {model}  @  {base_url}", file=sys.stderr)
    print(f"Oracle: {args.oracle}  Mode: {args.mode}", file=sys.stderr)

    # --- Process sentences ---
    sentences = split_sentences(text)
    print(f"Found {len(sentences)} sentences", file=sys.stderr)

    results = []
    corrected_parts = []
    pos = 0
    n_fixed = 0
    n_flagged = 0

    for sent in sentences:
        original = sent["text"]
        
        # Get oracle spans
        if args.oracle == "llm":
            spans = get_oracle_spans(original, method="llm", client=client, model=model)
        else:
            spans = get_oracle_spans(original, method=args.oracle, lt_url=args.lt_url)

        # Confidence filter: skip if too many tokens flagged (likely false positives)
        toks = original.split()
        if args.min_confidence > 0 and len(toks) > 0:
            flagged_ratio = len(spans) / len(toks)
            if flagged_ratio > args.min_confidence:
                spans = []

        if not spans:
            # No errors detected — keep original
            corrected_parts.append(original)
            results.append({
                "original": original,
                "corrected": original,
                "spans": [],
                "changed": False,
            })
            continue

        n_flagged += 1

        if args.mode == "dte":
            corrected, t = dte_fix_sentence(client, model, original, spans)
            results.append({
                "original": original,
                "corrected": corrected,
                "spans": spans,
                "changed": corrected != original,
                "time_s": round(t, 2),
            })
            if corrected != original:
                n_fixed += 1
            corrected_parts.append(corrected)

        elif args.mode == "raw":
            corrected, t = raw_fix_sentence(client, model, original)
            results.append({
                "original": original,
                "corrected": corrected,
                "spans": spans,
                "changed": corrected != original,
                "time_s": round(t, 2),
            })
            if corrected != original:
                n_fixed += 1
            corrected_parts.append(corrected)

        elif args.mode == "both":
            dte_corrected, t1 = dte_fix_sentence(client, model, original, spans)
            raw_corrected, t2 = raw_fix_sentence(client, model, original)
            results.append({
                "original": original,
                "corrected_dte": dte_corrected,
                "corrected_raw": raw_corrected,
                "spans": spans,
                "dte_changed": dte_corrected != original,
                "raw_changed": raw_corrected != original,
                "time_s": round(t1 + t2, 2),
            })
            # Use DTE as the primary output
            if dte_corrected != original:
                n_fixed += 1
            corrected_parts.append(dte_corrected)

        print(f"  [{len(results)}/{len(sentences)}] "
              f"{'FIXED' if results[-1].get('changed') or results[-1].get('dte_changed') else 'ok'} "
              f"(spans={len(spans)})", file=sys.stderr)

    # --- Output ---
    corrected_text = "".join(corrected_parts)

    if args.json:
        output = {
            "original": text,
            "corrected": corrected_text,
            "n_sentences": len(sentences),
            "n_flagged": n_flagged,
            "n_fixed": n_fixed,
            "oracle": args.oracle,
            "mode": args.mode,
            "sentences": results,
        }
        print(json.dumps(output, indent=2))
    else:
        if args.output:
            with open(args.output, "w") as f:
                f.write(corrected_text)
            print(f"Corrected text written to: {args.output}", file=sys.stderr)
        else:
            print(corrected_text)

        if args.diff:
            print("\n" + "=" * 72, file=sys.stderr)
            print("DIFF REPORT", file=sys.stderr)
            print("=" * 72, file=sys.stderr)
            for r in results:
                changed = r.get("changed") or r.get("dte_changed")
                if not changed:
                    continue
                corrected = r.get("corrected") or r.get("corrected_dte", "")
                print(f"\n  - {r['original']}", file=sys.stderr)
                print(f"  + {corrected}", file=sys.stderr)

    # Summary
    print(f"\nSummary: {len(sentences)} sentences, {n_flagged} flagged, {n_fixed} corrected", file=sys.stderr)


if __name__ == "__main__":
    main()
