#!/usr/bin/env python
"""Experiment 2 - self-contained runner for any OpenAI-compatible model server.

Auto-detects a running llama-server via `ps aux | grep llama-server` and
extracts port, model alias, and API key automatically. Override with env vars
or the CONFIG block below if needed.

Usage:
  python run_exp2_flint.py --task both --oracle combined --code-dataset multi --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================================
# CONFIG - leave empty to auto-detect from running llama-server
# =====================================================================
BASE_URL = ""   # e.g. "http://localhost:38589/v1"  (empty = auto-detect)
API_KEY  = ""   # (empty = auto-detect from --api-key-file)
MODEL    = ""   # e.g. "unsloth/Qwen3.8-27B-GGUF"   (empty = auto-detect from --alias)

# Absolute paths (this script works from any CWD)
SANDBOX = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SANDBOX, "results")
# =====================================================================


def detect_llama_server():
    """Find a running llama-server process and extract port, alias, api-key-file.
    
    Returns (base_url, api_key, model_name) or raises SystemExit with guidance.
    """
    try:
        ps_out = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception as e:
        print("ERROR: could not run 'ps aux': %s" % e)
        sys.exit(1)

    # Find the llama-server line (not grep itself)
    server_line = None
    for line in ps_out.splitlines():
        if "llama-server" in line and "grep" not in line:
            server_line = line
            break

    if server_line is None:
        print("ERROR: No running llama-server found.")
        print("Start one with:  llama-server -m model.gguf --port PORT --alias MODEL_NAME")
        sys.exit(1)

    # Extract port
    m = re.search(r"--port\s+(\d+)", server_line)
    if not m:
        print("ERROR: Could not find --port in llama-server args.")
        print("Line: %s" % server_line[:200])
        sys.exit(1)
    port = m.group(1)

    # Extract alias (model name)
    m = re.search(r"--alias\s+(\S+)", server_line)
    model_name = m.group(1) if m else "default"

    # Extract API key file
    api_key = ""
    m = re.search(r"--api-key-file\s+(\S+)", server_line)
    if m:
        key_file = m.group(1)
        try:
            with open(key_file) as f:
                api_key = f.read().strip()
        except Exception:
            pass  # will use empty key (some servers don't require auth)

    base_url = "http://localhost:%s/v1" % port
    return base_url, api_key, model_name


def _bug_type_str(res):
    """Get bug type display string from a result dict."""
    bt = res.get("bug_type", "?")
    if isinstance(bt, list):
        # Flatten in case of nested lists
        flat = []
        for item in bt:
            if isinstance(item, list):
                flat.extend(str(x) for x in item)
            else:
                flat.append(str(item))
        return ",".join(flat)
    return str(bt)


def dte_grammar_with_oracle(client, model, sentence, oracle_spans):
    """DTE grammar using oracle-provided spans (not gold)."""
    toks = sentence.split()
    mask_spans = [(s["start"], s["end"]) for s in oracle_spans]

    if not mask_spans:
        return sentence, 0.0

    lines = ["You are a precise grammar editor. Fix ONLY the marked spans.",
             "Do NOT change any other words or rephrase.", "",
             "Sentence: " + sentence, "",
             "Marked error spans (token indices, 0-based):"]
    for (s, e) in mask_spans:
        span = " ".join(toks[s:e]) if e <= len(toks) else " ".join(toks[s:])
        lines.append("  tokens [%d:%d] = %r" % (s, e, span))
    lines += ["",
              "Respond with ONLY the fully corrected sentence.",
              "Change only the marked spans; keep everything else identical."]
    prompt = "\n".join(lines)
    out, t = llm_complete(client, model, prompt, max_tokens=256)
    out = out.strip().strip('"').strip()
    return out, t


def dte_code_multiline(client, model, prompt, buggy, tests, bug_lines):
    """DTE code repair for multi-line bugs. Marks all suspect lines."""
    lines = buggy.split("\n")
    ctx = []
    for i, ln in enumerate(lines):
        marker = "  <-- SUSPECT" if (i + 1) in bug_lines else ""
        ctx.append("%4d | %s%s" % (i + 1, ln, marker))

    p = ("Here is buggy Python code. The marked SUSPECT lines contain bugs.\n\n"
         "Task: " + prompt + "\n\nCode:\n```python\n" + "\n".join(ctx) +
         "\n```\n\nFix ONLY the marked lines; do not change any other line. Respond with "
         "ONLY the corrected full code in a ```python block.\n"
         "Change as few lines as possible.")
    out, t = llm_code(client, model, p, max_tokens=4096)
    return extract_code(out, buggy), t


def load_multiline_code(n):
    """Load multi-line bug dataset."""
    with open(os.path.join(SANDBOX, "data", "code_repair_bugs_multiline.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["grammar", "code", "both"], default="both")
    ap.add_argument("--n-grammar", type=int, default=50)
    ap.add_argument("--n-code", type=int, default=15)
    ap.add_argument("--oracle", choices=["gold", "combined", "wordlist", "heuristic", "languagetool"],
                    default="gold", help="Oracle method for grammar DTE mask")
    ap.add_argument("--lt-url", default="http://localhost:8823",
                    help="LanguageTool server URL (for --oracle languagetool)")
    ap.add_argument("--code-dataset", choices=["single", "multi"], default="single",
                    help="Code bug dataset: single-line or multi-line")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel requests to the LLM server (requires llama-server --parallel N)")
    ap.add_argument("--base-url", default=None, help="Override server URL")
    ap.add_argument("--model", default=None, help="Override model name")
    args = ap.parse_args()

    # Resolve connection: CLI > env > CONFIG block > auto-detect
    base_url = args.base_url or os.environ.get("DTE_LLM_BASE_URL") or BASE_URL
    api_key = os.environ.get("DTE_LLM_API_KEY", "") or API_KEY
    model_name = args.model or os.environ.get("DTE_LLM_MODEL_NAME") or MODEL

    if not base_url:
        print("No server configured. Auto-detecting from running llama-server...")
        detected_url, detected_key, detected_model = detect_llama_server()
        base_url = detected_url
        if not api_key:
            api_key = detected_key
        if not model_name:
            model_name = detected_model
        print("  Detected: %s (model=%s)" % (base_url, model_name))

    os.environ["DTE_LLM_BASE_URL"] = base_url
    os.environ["DTE_LLM_API_KEY"] = api_key
    os.environ["DTE_LLM_MODEL_NAME"] = model_name

    sys.path.insert(0, SANDBOX)
    from dte.experiment.run_experiment import (  # noqa: E402
        make_llm, load_gec, load_code,
        raw_llm_grammar, dte_grammar,
        raw_llm_code, dte_code, run_tests, line_diff,
        llm_complete, llm_code, extract_code,
    )
    from dte.experiment.gec_eval import evaluate_sentence, aggregate  # noqa: E402
    from dte.experiment.oracle import get_oracle_spans  # noqa: E402

    os.makedirs(RESULTS_DIR, exist_ok=True)
    client, model = make_llm()
    print("Model: %s  @  %s" % (model, base_url))
    if args.task in ("grammar", "both"):
        print("Oracle: %s" % args.oracle)
    if args.task in ("code", "both"):
        print("Code dataset: %s" % args.code_dataset)
    if args.workers > 1:
        print("Workers: %d (parallel requests)" % args.workers)

    # --- Task A: Grammar (CoNLL-2014) --------------------------------
    if args.task in ("grammar", "both"):
        print("=" * 72)
        print("TASK A: Grammar - CoNLL-2014 (%d sentences, oracle=%s)" % (args.n_grammar, args.oracle))
        print("=" * 72)

        sents = load_gec(args.n_grammar)
        raw_results = [None] * args.n_grammar
        dte_results = [None] * args.n_grammar
        raw_time = dte_time = 0.0
        oracle_stats = {"total_spans": 0, "n_empty": 0}

        def _process_grammar(i, row):
            original = row["original"]
            gold_edits = json.loads(row["edits"])

            r_out, r_t = raw_llm_grammar(client, model, original)
            r_res = evaluate_sentence(original, r_out, gold_edits)

            if args.oracle == "gold":
                d_out, d_t = dte_grammar(client, model, original, gold_edits)
                n_spans = len(gold_edits)
            else:
                oracle_spans = get_oracle_spans(original, method=args.oracle, lt_url=args.lt_url)
                n_spans = len(oracle_spans)
                d_out, d_t = dte_grammar_with_oracle(client, model, original, oracle_spans)

            d_res = evaluate_sentence(original, d_out, gold_edits)
            return i, r_res, d_res, r_t, d_t, n_spans

        done_count = 0
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_process_grammar, i, row) for i, row in enumerate(sents)]
                for fut in as_completed(futures):
                    i, r_res, d_res, r_t, d_t, n_spans = fut.result()
                    raw_results[i] = r_res
                    dte_results[i] = d_res
                    raw_time += r_t
                    dte_time += d_t
                    oracle_stats["total_spans"] += n_spans
                    if n_spans == 0:
                        oracle_stats["n_empty"] += 1
                    done_count += 1
                    if done_count % 10 == 0:
                        print("  ...%d/%d done" % (done_count, args.n_grammar), flush=True)
        else:
            for i, row in enumerate(sents):
                i, r_res, d_res, r_t, d_t, n_spans = _process_grammar(i, row)
                raw_results[i] = r_res
                dte_results[i] = d_res
                raw_time += r_t
                dte_time += d_t
                oracle_stats["total_spans"] += n_spans
                if n_spans == 0:
                    oracle_stats["n_empty"] += 1
                print("raw: %s" % r_res)
                print("dte: %s" % d_res)
                if (i + 1) % 10 == 0:
                    print("  ...%d/%d done" % (i + 1, args.n_grammar), flush=True)

        raw_agg = aggregate(raw_results)
        dte_agg = aggregate(dte_results)

        print("\n%-24s %12s %12s" % ("Metric", "Raw LLM", "DTE"))
        print("-" * 50)
        print("%-24s %12.4f %12.4f" % ("Micro F0.5", raw_agg["micro_f05"], dte_agg["micro_f05"]))
        print("%-24s %12.4f %12.4f" % ("Micro Precision", raw_agg["micro_precision"], dte_agg["micro_precision"]))
        print("%-24s %12.4f %12.4f" % ("Micro Recall", raw_agg["micro_recall"], dte_agg["micro_recall"]))
        print("%-24s %12.1f %12.1f" % ("Total time (s)", raw_time, dte_time))
        if args.oracle != "gold":
            print("\nOracle stats: %d spans flagged, %d/%d sentences had no flags"
                  % (oracle_stats["total_spans"], oracle_stats["n_empty"], args.n_grammar))

        suffix = "_" + args.oracle if args.oracle != "gold" else ""
        out_path = os.path.join(RESULTS_DIR, "exp2_grammar%s.json" % suffix)
        with open(out_path, "w") as f:
            json.dump({"raw": raw_results, "dte": dte_results,
                       "raw_agg": raw_agg, "dte_agg": dte_agg,
                       "raw_time": raw_time, "dte_time": dte_time,
                       "oracle": args.oracle,
                       "oracle_stats": oracle_stats}, f, indent=1)
        print("Saved: %s" % out_path)

    # --- Task B: Code repair ------------------------------------------
    if args.task in ("code", "both"):
        is_multi = (args.code_dataset == "multi")
        label = "multi-line bugs" if is_multi else "single-line bugs"
        print("\n" + "=" * 72)
        print("TASK B: Code repair - MBPP + injected %s (%d instances)" % (label, args.n_code))
        print("=" * 72)

        if is_multi:
            items = load_multiline_code(args.n_code)
        else:
            items = load_code(args.n_code)

        with open(os.path.join(SANDBOX, "data", "mbpp_test.jsonl")) as f:
            mbpp = {json.loads(l)["task_id"]: json.loads(l) for l in f}

        def _process_code(i, it):
            buggy = it["buggy_code"]
            tests = mbpp[it["task_id"]]["test_list"]

            r_code, r_t = raw_llm_code(client, model, it["prompt"], buggy, tests)
            r_fails, _ = run_tests(r_code, tests)

            if is_multi:
                bug_lines = it["bug_lines_1based"]
                d_code, d_t = dte_code_multiline(client, model, it["prompt"], buggy, tests, bug_lines)
            else:
                d_code, d_t = dte_code(client, model, it["prompt"], buggy, tests,
                                       it["bug_line_1based"])
            d_fails, _ = run_tests(d_code, tests)

            # Normalize bug_type to always be a flat list of strings
            raw_bt = it.get("bug_types", None)
            if raw_bt is None:
                raw_bt = [it.get("bug_type", "?")]
            # Flatten any nested lists
            flat_bt = []
            for item in raw_bt:
                if isinstance(item, list):
                    flat_bt.extend(str(x) for x in item)
                else:
                    flat_bt.append(str(item))

            res = {
                "task_id": it["task_id"],
                "bug_type": flat_bt,
                "bug_lines_1based": it.get("bug_lines_1based", [it.get("bug_line_1based", 0)]),
                "raw": {"code": r_code, "n_fails": r_fails,
                        "pass": r_fails == 0, "line_diff": line_diff(buggy, r_code),
                        "time_s": round(r_t, 2)},
                "dte": {"code": d_code, "n_fails": d_fails,
                        "pass": d_fails == 0, "line_diff": line_diff(buggy, d_code),
                        "time_s": round(d_t, 2)},
            }
            return i, res, r_t, d_t

        per_instance = [None] * len(items)
        raw_time = dte_time = 0.0
        done_count = 0

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_process_code, i, it) for i, it in enumerate(items)]
                for fut in as_completed(futures):
                    i, res, r_t, d_t = fut.result()
                    per_instance[i] = res
                    raw_time += r_t
                    dte_time += d_t
                    done_count += 1
                    print("  [%d/%d] task %s (%s): raw=%s dte=%s"
                          % (done_count, args.n_code, res["task_id"], _bug_type_str(res),
                             "PASS" if res["raw"]["pass"] else "FAIL",
                             "PASS" if res["dte"]["pass"] else "FAIL"), flush=True)
        else:
            for i, it in enumerate(items):
                i, res, r_t, d_t = _process_code(i, it)
                per_instance[i] = res
                raw_time += r_t
                dte_time += d_t
                print("  [%d/%d] task %s (%s): raw=%s dte=%s"
                      % (i + 1, args.n_code, res["task_id"], _bug_type_str(res),
                         "PASS" if res["raw"]["pass"] else "FAIL",
                         "PASS" if res["dte"]["pass"] else "FAIL"), flush=True)

        raw_pass = sum(1 for p in per_instance if p["raw"]["pass"])
        dte_pass = sum(1 for p in per_instance if p["dte"]["pass"])
        raw_diffs = [p["raw"]["line_diff"] for p in per_instance]
        dte_diffs = [p["dte"]["line_diff"] for p in per_instance]

        import statistics
        target = 2 if is_multi else 1
        print("\n%-24s %12s %12s" % ("Metric", "Raw LLM", "DTE"))
        print("-" * 50)
        print("%-24s %12.3f %12.3f" % ("Pass rate", raw_pass / args.n_code, dte_pass / args.n_code))
        print("%-24s %12.2f %12.2f" % ("Mean line-diff", statistics.mean(raw_diffs), statistics.mean(dte_diffs)))
        print("%-24s %12.2f %12.2f" % ("Median line-diff", statistics.median(raw_diffs), statistics.median(dte_diffs)))
        print("%-24s %12.3f %12.3f" % ("Minimal (diff<=%d)" % target,
              sum(1 for d in raw_diffs if d <= target) / args.n_code,
              sum(1 for d in dte_diffs if d <= target) / args.n_code))
        print("%-24s %12.1f %12.1f" % ("Total time (s)", raw_time, dte_time))

        suffix = "_multi" if is_multi else ""
        out_path = os.path.join(RESULTS_DIR, "exp2_code%s.json" % suffix)
        with open(out_path, "w") as f:
            json.dump({"raw_pass": raw_pass, "dte_pass": dte_pass, "n": args.n_code,
                       "raw_diffs": raw_diffs, "dte_diffs": dte_diffs,
                       "raw_time": raw_time, "dte_time": dte_time,
                       "dataset": args.code_dataset}, f, indent=1)
        out_path2 = os.path.join(RESULTS_DIR, "exp2_code%s_outputs.json" % suffix)
        with open(out_path2, "w") as f:
            json.dump(per_instance, f, indent=1)
        print("Saved: %s, %s" % (out_path, out_path2))


if __name__ == "__main__":
    main()
