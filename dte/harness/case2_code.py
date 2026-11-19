"""DTE Case 2 (LLM-backed) — Test-passing code with a real model denoiser.

Same mechanism as dte.case2_code (edit-grammar bandit), but the denoiser
is replaced with an LLM that proposes minimal fixes for suspect lines.
Non-suspect lines are still copied verbatim (structural locality).

The test oracle is unchanged: it runs a real test suite via subprocess
and caches results. The LLM only handles the "denoising" step.

Usage:
  # With llama.cpp (local GGUF):
  export DTE_LLM_BACKEND=llama_cpp
  export DTE_LLM_MODEL_PATH=/path/to/your-model-q4_k_m.gguf
  python -m dte.harness.case2_code

  # With llama-server / vLLM / Ollama:
  export DTE_LLM_BACKEND=openai_compat
  export DTE_LLM_BASE_URL=http://localhost:8080/v1
  export DTE_LLM_MODEL_NAME=your-model
  python -m dte.harness.case2_code

  # For a larger, harder fixture (3 bugs instead of 2):
  export DTE_CODE_FIXTURE=hard
"""
from __future__ import annotations

import re
import sys
import textwrap
from typing import List, Optional, Set

# Reuse the test oracle and AST helpers from the base package
from ..case2_code import (
    TestOracle, build_buggy_code, build_test_code, _ast_function_lines,
)
from ..core import DTEConfig
from ..ed import line_diff_lines, unified_diff

from .config import load_config
from .backend import create_backend
from .embedding import create_embedding_model
from .llm_denoiser import LLMCodeDenoiser


def build_hard_fixture() -> str:
    """Three-function fixture with one bug each (off-by-one, missing return,
    wrong operator). Ground-truth minimal diff: 3 lines."""
    return textwrap.dedent("""\
        def sum_first_n(n):
            \"\"\"Sum the first n positive integers.\"\"\"
            total = 0
            for i in range(1, n + 2):
                total += i
            return total


        def clamp(x, lo, hi):
            \"\"\"Clamp x to the range [lo, hi].\"\"\"
            if x < lo:
                return lo
            if x > hi:
                return hi
            # missing return


        def is_even(n):
            \"\"\"Return True if n is even.\"\"\"
            return n % 2 != 0
    """)


def build_hard_test_code() -> str:
    """Test suite for the hard fixture (3 functions)."""
    return textwrap.dedent("""\
        import sys, os
        sys.path.insert(0, os.path.dirname(CANDIDATE_PATH))
        from candidate import sum_first_n, clamp, is_even

        checks = [
            ("sum_first_n(1)",  lambda: sum_first_n(1) == 1),
            ("sum_first_n(3)",  lambda: sum_first_n(3) == 6),
            ("sum_first_n(5)",  lambda: sum_first_n(5) == 15),
            ("clamp(5,0,10)",   lambda: clamp(5, 0, 10) == 5),
            ("clamp(-3,0,10)",  lambda: clamp(-3, 0, 10) == 0),
            ("clamp(15,0,10)",  lambda: clamp(15, 0, 10) == 10),
            ("is_even(4)",      lambda: is_even(4) is True),
            ("is_even(7)",      lambda: is_even(7) is False),
            ("is_even(0)",      lambda: is_even(0) is True),
        ]

        n_fail = 0
        for name, fn in checks:
            try:
                ok = fn()
            except Exception as e:
                ok = False
                print(f"FAIL {name}: raised {e}")
            if ok:
                print(f"PASS {name}")
            else:
                print(f"FAIL {name}")
                n_fail += 1

        if n_fail == 0:
            print("ALL TESTS PASSED")
            sys.exit(0)
        else:
            print(f"{n_fail}/{len(checks)} checks FAILED")
            sys.exit(1)
    """)


def run_dte_code_llm(
    code: str,
    oracle: TestOracle,
    denoiser: LLMCodeDenoiser,
    budget: int = 5,
    seed: int = 0,
    log: Optional[List[str]] = None,
):
    """DTE pipeline for code with an LLM denoiser.

    Cumulative bandit: each iteration asks the LLM to fix the suspect lines.
    If tests still fail, expand the mask and retry. Fixes accumulate.
    """
    import random
    log = log if log is not None else []

    def say(msg):
        log.append(msg)

    current = code
    mask = oracle.mask(current)
    say(f"iter 0: suspect lines = {sorted(l + 1 for l in mask)} "
        f"({oracle.feedback(current)})")

    passed = oracle.ok(current)
    if passed:
        say("input already passes all tests")
        from ..core import DTEResult
        return DTEResult(
            original=code, edited=current, passed=True, iterations=0,
            final_mask_size=0, tokens_changed=0, ed=0, edit_precision=None,
        )

    def _failure_count(src):
        res = oracle.run(src)
        if res.passed:
            return 0
        return len(re.findall(r"^FAIL ", res.output, re.MULTILINE))

    it = 0
    for it in range(1, budget + 1):
        if passed:
            break

        test_output = oracle.run(current).output
        candidate = denoiser.denoise(current, mask, random.Random(seed),
                                     test_output=test_output)

        if candidate is not None and candidate != current:
            cand_ok = oracle.ok(candidate)
            cand_fail = _failure_count(candidate)
            cur_fail = _failure_count(current)
            diff = line_diff_lines(code, candidate)

            if cand_ok:
                say(f"iter {it}: PASSING fix found (diff={diff} lines)")
                current = candidate
                passed = True
                break
            elif cand_fail < cur_fail:
                say(f"iter {it}: improved (failures {cand_fail}<{cur_fail}, "
                    f"diff={diff} lines)")
                current = candidate
                new_mask = oracle.mask(current)
                if not new_mask:
                    new_mask = {ln - 1 for ln in _ast_function_lines(current)}
                mask = new_mask
            else:
                say(f"iter {it}: no improvement (failures {cand_fail}>={cur_fail}); "
                    f"expanding mask")
        else:
            say(f"iter {it}: LLM returned no fix; expanding mask")

        # Expand mask to adjacent lines
        nlines = len(current.splitlines())
        new_mask = set(mask)
        for ln in list(mask):
            if ln - 1 >= 0:
                new_mask.add(ln - 1)
            if ln + 1 < nlines:
                new_mask.add(ln + 1)
        mask = new_mask

    from ..core import DTEResult
    diff_lines = line_diff_lines(code, current)
    return DTEResult(
        original=code, edited=current, passed=passed, iterations=it,
        final_mask_size=len(mask), tokens_changed=diff_lines, ed=diff_lines,
        edit_precision=None,
    )


def main() -> None:
    import os

    try:
        cfg = load_config()
    except RuntimeError as e:
        print(f"Configuration error:\n{e}", file=sys.stderr)
        sys.exit(1)

    # Select fixture
    fixture_name = os.environ.get("DTE_CODE_FIXTURE", "easy")
    if fixture_name == "hard":
        buggy = build_hard_fixture()
        test_code = build_hard_test_code()
        gt_lines = 3
    else:
        buggy = build_buggy_code()
        test_code = build_test_code()
        gt_lines = 2

    oracle = TestOracle(test_code)
    llm = create_backend(cfg.llm)
    print(f"LLM backend: {cfg.llm.backend}")

    embed = None
    if cfg.embed.enabled:
        try:
            embed = create_embedding_model(cfg.embed)
        except Exception as e:
            print(f"WARNING: embedding unavailable ({e})", file=sys.stderr)

    denoiser = LLMCodeDenoiser(llm, embed)

    print(f"\nFixture: {fixture_name} ({len(buggy.splitlines())} lines, "
          f"ground-truth diff = {gt_lines} lines)")
    for i, line in enumerate(buggy.splitlines(), 1):
        print(f"  {i:3d}  {line}")
    print()

    log: List[str] = []
    result = run_dte_code_llm(
        code=buggy, oracle=oracle, denoiser=denoiser,
        budget=cfg.edit_budget, seed=cfg.seed, log=log,
    )

    for line in log:
        print("  " + line)

    if result.passed:
        print(f"\nFixed source ({len(result.edited.splitlines())} lines):")
        for i, line in enumerate(result.edited.splitlines(), 1):
            orig = result.original.splitlines()
            marker = " *" if (i <= len(orig) and orig[i - 1] != line) else ""
            print(f"  {i:3d}  {line}{marker}")
        print("\nUnified diff:")
        print(unified_diff(result.original, result.edited, "buggy.py", "fixed.py"))
    else:
        print("\nFAILED: no passing candidate found within budget.")

    from ..core import metrics_table
    print()
    print(metrics_table(result, extra={
        "Oracle calls (subprocess)": oracle.call_count,
        "Line-diff size": line_diff_lines(result.original, result.edited),
        "Ground-truth minimal diff (lines)": gt_lines,
        "LLM backend": cfg.llm.backend,
    }))

    print()
    if result.passed:
        print(f"PASS: all tests pass")
        diff = line_diff_lines(result.original, result.edited)
        if diff == gt_lines:
            print(f"PASS: minimal line-diff ({diff} = ground truth)")
        else:
            print(f"NOTE: line-diff={diff}, ground truth={gt_lines}")
    else:
        print("FAIL: tests do not pass")


if __name__ == "__main__":
    main()
