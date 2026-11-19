"""Case 2 — Test-passing code DTE (build second, per section 8.3).

Components:
  * Oracle   : runs a real test suite via subprocess; caches results by hash
  * Mask     : suspect-line mask from AST structure + traceback line numbers
  * Denoiser : bounded edit-grammar candidates (1-3 token changes per line)
  * Bandit   : generate N candidates, run cached oracle on each, pick the
               minimal-diff passing candidate; expand mask on failure

Run:  python -m dte.case2_code
"""
from __future__ import annotations

import ast
import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .core import DTEConfig, DTEResult, Oracle, metrics_table, run_dte
from .ed import line_diff_lines, unified_diff


# ---------------------------------------------------------------------------
# Test-suite oracle (subprocess + cache)
# ---------------------------------------------------------------------------

def _ast_function_lines(code: str) -> Set[int]:
    """Return 1-based line numbers of all function bodies in `code`.

    Used as a static-analysis fallback when the traceback doesn't point into
    the candidate file (e.g., the assert fires in the test file). This gives
    the mask every line inside every def, which is coarse but guarantees the
    buggy line is included. Section 8.3: "static analysis + test-trace ->
    suspect-line mask".
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    lines: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # include the def line through the last line of the body
            end = getattr(node, "end_lineno", None) or (
                max((s.end_lineno for s in ast.walk(node)
                     if hasattr(s, "end_lineno")), default=node.lineno))
            lines.update(range(node.lineno, end + 1))
    return lines

@dataclass
class _TestResult:
    passed: bool
    failing_lines: Set[int]      # 1-based line numbers from tracebacks
    output: str                  # full stdout+stderr for debugging


class TestOracle:
    """Runs a candidate Python file against an embedded test suite.

    The test suite is written to a temp file alongside the candidate and run
    via subprocess. Results are cached by content hash to avoid re-runs
    (section 8.3: "Cache results keyed by candidate hash").
    """

    def __init__(self, test_code: str):
        self._test_code = test_code
        self._cache: Dict[str, _TestResult] = {}
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def _hash(self, code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def run(self, code: str) -> _TestResult:
        h = self._hash(code)
        if h in self._cache:
            return self._cache[h]
        self._call_count += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            cand_path = os.path.join(tmpdir, "candidate.py")
            test_path = os.path.join(tmpdir, "test_candidate.py")
            with open(cand_path, "w") as f:
                f.write(code)
            with open(test_path, "w") as f:
                f.write(self._test_code.replace("CANDIDATE_PATH", repr(cand_path)))

            proc = subprocess.run(
                [sys.executable, test_path],
                capture_output=True, text=True, timeout=30,
            )
            out = proc.stdout + proc.stderr
            passed = proc.returncode == 0

            # Extract failing line numbers from the candidate file in tracebacks.
            # The traceback may show the candidate file when the bug is inside a
            # function call chain (e.g., assert calls sum_first_n which raises).
            failing_lines: Set[int] = set()
            for m in re.finditer(
                    r'File "' + re.escape(cand_path) + r'", line (\d+)', out):
                failing_lines.add(int(m.group(1)))

            # Fallback: if the traceback only points at the test file (assert
            # fired there, not inside the candidate), use AST to find all
            # function bodies in the candidate and flag their lines. This is
            # the "static-analysis signal" from section 8.3.
            if not failing_lines and not passed:
                failing_lines = _ast_function_lines(code)

            result = _TestResult(passed=passed, failing_lines=failing_lines, output=out)
        self._cache[h] = result
        return result

    def ok(self, code: str) -> bool:
        return self.run(code).passed

    def mask(self, code: str) -> Set[int]:
        """Suspect-line mask: lines implicated by the traceback (1-based),
        converted to 0-based line indices."""
        res = self.run(code)
        return {ln - 1 for ln in res.failing_lines}

    def feedback(self, code: str) -> str:
        res = self.run(code)
        if res.passed:
            return "all tests pass"
        lines = ", ".join(f"L{n}" for n in sorted(res.failing_lines)) or "unknown"
        return f"tests FAIL (failing lines: {lines})"


# ---------------------------------------------------------------------------
# Edit-grammar denoiser (bounded local edits, section 8.3)
# ---------------------------------------------------------------------------

@dataclass
class LineEdit:
    """A single-line edit: replace line `line_idx` (0-based) with `new_text`."""
    line_idx: int
    new_text: str


def _generate_candidates(code: str, mask: Set[int], rng) -> List[str]:
    """Generate candidate edits at masked lines using a bounded edit grammar.

    Each candidate differs from the original by exactly ONE line (substitution
    only — no insertions/deletions, so line alignment is preserved and the
    diff is minimal). For each masked line we try:

      * off-by-one in numeric literals:  n+2 -> n+1, n-1 -> n+1, etc.
        (try +/- 1 on every integer literal in the line)
      * comparison operator swaps:  == <-> !=,  < <-> <=,  > <-> >=
      * arithmetic operator swaps:  + <-> -,  * <-> /
      * missing return: replace a trailing comment/blank line inside a
        function with "return <var>" for each local variable in scope

    Returns a list of full candidate source strings.
    """
    lines = code.splitlines(keepends=True)
    candidates: List[str] = []
    seen: Set[str] = set()

    def _add(modified_lines: List[str]) -> None:
        cand = "".join(modified_lines)
        if cand != code and cand not in seen:
            seen.add(cand)
            candidates.append(cand)

    # Precompute: for each line, the set of local variable names assigned
    # before it within the same function (for missing-return candidates).
    def _local_vars_before(line_idx: int) -> List[str]:
        """Find variable names in scope at line_idx (0-based).

        A line belongs to a function if it's within the function's AST range,
        OR if it's a trailing comment/blank line immediately after the
        function body (before the next top-level def). This handles the
        common pattern of a comment at the end of a function body.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        # Collect all top-level function defs with their line ranges,
        # extending each range to just before the next top-level def.
        funcs = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", None) or node.lineno
                funcs.append((node, node.lineno, end))

        # Extend each function's range to the line before the next def
        for i, (node, start, end) in enumerate(funcs):
            if i + 1 < len(funcs):
                next_start = funcs[i + 1][1]
                extended_end = max(end, next_start - 1)
            else:
                extended_end = max(end, len(code.splitlines()))
            funcs[i] = (node, start, extended_end)

        for node, start, end in funcs:
            if not (start <= line_idx + 1 <= end):
                continue
            vars_found: List[str] = []
            for a in node.args.args:
                vars_found.append(a.arg)
            for a in getattr(node.args, "posonlyargs", []):
                vars_found.append(a.arg)
            if node.args.vararg:
                vars_found.append(node.args.vararg.arg)
            if node.args.kwarg:
                vars_found.append(node.args.kwarg.arg)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and sub.lineno <= line_idx + 1:
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            vars_found.append(t.id)
                elif (isinstance(sub, ast.AnnAssign) and sub.target is not None
                      and isinstance(sub.target, ast.Name)
                      and sub.lineno <= line_idx + 1):
                    vars_found.append(sub.target.id)
                elif isinstance(sub, ast.For) and sub.lineno <= line_idx + 1:
                    t = sub.target
                    if isinstance(t, ast.Name):
                        vars_found.append(t.id)
            return list(dict.fromkeys(vars_found))
        return []

    for line_idx in sorted(mask):
        if line_idx >= len(lines):
            continue
        orig_line = lines[line_idx]
        stripped = orig_line.strip()
        indent = orig_line[:len(orig_line) - len(orig_line.lstrip())]

        # --- off-by-one: try +/- 1 on every integer literal in the line ---
        int_pattern = re.compile(r"(\d+)")
        for m in int_pattern.finditer(stripped):
            val = int(m.group(1))
            for delta in (-1, 1):
                new_val = val + delta
                if new_val < 0:
                    continue
                new_line_text = orig_line.replace(m.group(1), str(new_val), 1)
                mod = list(lines)
                mod[line_idx] = new_line_text
                _add(mod)

        # --- comparison operator swaps ---
        for old, new in [("==", "!="), ("!=", "=="),
                         ("<=", "<"), ("<", "<="),
                         (">=", ">"), (">", ">=")]:
            if old in stripped:
                mod = list(lines)
                mod[line_idx] = orig_line.replace(old, new, 1)
                _add(mod)

        # --- missing return: replace this line with "return <var>" for each
        #     local variable. Only if the line is a comment or blank AND it's
        #     inside a function body (not at module level). ---
        if stripped.startswith("#") or not stripped:
            vars_in_scope = _local_vars_before(line_idx)
            if vars_in_scope:  # non-empty means we're inside a function
                for var in vars_in_scope:
                    mod = list(lines)
                    mod[line_idx] = f"{indent}return {var}\n"
                    _add(mod)

    return candidates


class CodeDenoiser:
    """Bounded edit-grammar denoiser for the code case.

    Generates candidate edits at masked lines only; non-masked lines are
    untouched (structural locality).
    """

    def __init__(self, oracle: TestOracle):
        self._oracle = oracle

    def denoise(self, code: str, mask: Set[int], rng) -> Optional[str]:
        """Return the minimal-diff passing candidate, or None if none pass.

        This is the bandit step (section 3.2b): generate N candidates, run
        the cached oracle on each, select the minimal-diff passing one.
        """
        cands = _generate_candidates(code, mask, rng)
        best: Optional[Tuple[int, str]] = None  # (diff_size, code)
        for cand in cands:
            if self._oracle.ok(cand):
                d = line_diff_lines(code, cand)
                if best is None or d < best[0]:
                    best = (d, cand)
        return best[1] if best else None


# ---------------------------------------------------------------------------
# DTE pipeline for code (adapted from core.run_dte for string docs)
# ---------------------------------------------------------------------------

def run_dte_code(
    code: str,
    oracle: TestOracle,
    config: Optional[DTEConfig] = None,
    log: Optional[List[str]] = None,
) -> DTEResult[str]:
    """DTE pipeline for the code case (section 8.3).

    Unlike Case 1 (token-level, single denoising pass), the code case uses a
    *cumulative* bandit: each iteration tries every single-line fix at the
    masked lines and keeps the one that reduces the test-failure count the
    most. Fixes accumulate across iterations until all tests pass or the
    budget is exhausted. This mirrors the "iterative mask expansion" of
    section 3.4 step 3, but in the code domain each accepted edit *stays* —
    the next iteration searches around the improved candidate.

    The search space is local by construction: only masked lines are ever
    modified (structural locality, section 2.3).
    """
    config = config or DTEConfig()
    log = log if log is not None else []

    def say(msg: str) -> None:
        log.append(msg)

    current = code
    mask = oracle.mask(current)
    say(f"iter 0: suspect-line mask = {sorted(mask)} ({oracle.feedback(current)})")

    passed = oracle.ok(current)
    if passed:
        say("input already passes all tests; no edits needed")
        return DTEResult(
            original=code, edited=current, passed=True, iterations=0,
            final_mask_size=0, tokens_changed=0, ed=0, edit_precision=None,
        )

    def _failure_count(src: str) -> int:
        """Count how many test checks fail (graded signal for the bandit)."""
        res = oracle.run(src)
        if res.passed:
            return 0
        # count "FAIL " lines in the output
        return len(re.findall(r"^FAIL ", res.output, re.MULTILINE))

    it = 0
    for it in range(1, config.budget + 1):
        if passed:
            break

        cands = _generate_candidates(current, mask, __import__("random").Random(config.seed))
        best_cand: Optional[str] = None
        best_score = _failure_count(current)  # current failure count as baseline
        best_diff = 0

        for cand in cands:
            if oracle.ok(cand):
                d = line_diff_lines(code, cand)
                if best_cand is None or d < best_diff:
                    best_cand, best_diff = cand, d
                # a fully-passing candidate is always optimal; stop searching
                break
            score = _failure_count(cand)
            if score < best_score:
                best_score = score
                best_cand = cand
                best_diff = line_diff_lines(code, cand)

        if best_cand is not None and (oracle.ok(best_cand) or best_score == 0):
            say(f"iter {it}: PASSING candidate found "
                f"(cumulative diff={line_diff_lines(code, best_cand)} lines)")
            current = best_cand
            passed = True
            break
        elif best_cand is not None:
            say(f"iter {it}: improved candidate (failures {best_score} < prev); "
                f"cumulative diff={line_diff_lines(code, best_cand)} lines")
            current = best_cand
            # recompute mask on the improved candidate (some bugs may be fixed)
            new_mask = oracle.mask(current)
            if not new_mask:
                # no more failing lines detected; keep searching all function lines
                new_mask = {ln - 1 for ln in _ast_function_lines(current)}
            say(f"         updated mask -> {sorted(new_mask)}")
            mask = new_mask
        else:
            # No improvement found at current mask; expand to adjacent lines
            new_mask = set(mask)
            nlines = len(current.splitlines())
            for ln in list(mask):
                if ln - 1 >= 0:
                    new_mask.add(ln - 1)
                if ln + 1 < nlines:
                    new_mask.add(ln + 1)
            say(f"iter {it}: no improvement; expanding mask "
                f"{len(mask)} -> {len(new_mask)} lines")
            mask = new_mask

    diff_lines = line_diff_lines(code, current)
    return DTEResult(
        original=code,
        edited=current,
        passed=passed,
        iterations=it,
        final_mask_size=len(mask),
        tokens_changed=diff_lines,
        ed=diff_lines,
        edit_precision=None,
    )


# ---------------------------------------------------------------------------
# Fixtures (section 10: tiny functions with planted single-line bugs)
# ---------------------------------------------------------------------------

def build_buggy_code() -> str:
    """Two functions with one planted single-line bug each:
      1. off-by-one in a range bound (n + 2 should be n + 1)
      2. wrong comparison operator (!= should be ==)
    Ground-truth minimal diff: 2 lines.
    """
    return textwrap.dedent("""\
        def sum_first_n(n):
            \"\"\"Sum the first n positive integers.\"\"\"
            total = 0
            for i in range(1, n + 2):
                total += i
            return total


        def is_even(n):
            \"\"\"Return True if n is even.\"\"\"
            return n % 2 != 0
    """)


def build_test_code() -> str:
    """Test suite. Runs ALL checks (no early exit) and prints one PASS/FAIL
    line per check so the oracle gives the bandit a graded signal.
    Exits 0 only if everything passes."""
    return textwrap.dedent("""\
        import sys, os
        sys.path.insert(0, os.path.dirname(CANDIDATE_PATH))
        from candidate import sum_first_n, is_even

        checks = [
            ("sum_first_n(1)",  lambda: sum_first_n(1) == 1),
            ("sum_first_n(3)",  lambda: sum_first_n(3) == 6),
            ("sum_first_n(5)",  lambda: sum_first_n(5) == 15),
            ("sum_first_n(10)", lambda: sum_first_n(10) == 55),
            ("is_even(4)",      lambda: is_even(4) is True),
            ("is_even(7)",      lambda: is_even(7) is False),
            ("is_even(0)",      lambda: is_even(0) is True),
            ("is_even(2)",      lambda: is_even(2) is True),
            ("is_even(9)",      lambda: is_even(9) is False),
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


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    buggy = build_buggy_code()
    oracle = TestOracle(build_test_code())

    print("=" * 72)
    print("DTE Case 2 — test-passing code (bandit + cached test oracle)")
    print("=" * 72)
    print(f"Buggy source ({len(buggy.splitlines())} lines):")
    for i, line in enumerate(buggy.splitlines(), 1):
        print(f"  {i:3d}  {line}")
    print()

    log: List[str] = []
    result = run_dte_code(
        code=buggy,
        oracle=oracle,
        config=DTEConfig(budget=5, seed=0),
        log=log,
    )
    for line in log:
        print("  " + line)

    if result.passed:
        print(f"\nFixed source ({len(result.edited.splitlines())} lines):")
        for i, line in enumerate(result.edited.splitlines(), 1):
            orig = result.original.splitlines()
            marker = " *" if (i <= len(orig) and orig[i-1] != line) else ""
            print(f"  {i:3d}  {line}{marker}")

        print("\nUnified diff:")
        print(unified_diff(result.original, result.edited, "buggy.py", "fixed.py"))
    else:
        print("\nFAILED: no passing candidate found within budget.")

    print()
    gt_lines = 2  # ground truth: one line per bug
    print(metrics_table(result, extra={
        "Oracle calls (subprocess)": oracle.call_count,
        "Line-diff size": line_diff_lines(result.original, result.edited),
        "Ground-truth minimal diff (lines)": gt_lines,
    }))

    # ------------------------------------------------------------------
    # Acceptance criteria (section 8.3) — built-in assertions
    # ------------------------------------------------------------------
    assert result.passed, "acceptance (i): all tests must pass"
    diff = line_diff_lines(result.original, result.edited)
    assert diff == gt_lines, (
        f"acceptance (ii): minimal line-diff expected; "
        f"got {diff}, ground truth {gt_lines}")
    print(f"\nAll acceptance criteria PASSED:")
    print(f"  (i)   all tests pass")
    print(f"  (ii)  line-diff = {diff} (ground truth: {gt_lines})")
    print(f"  (iii) oracle calls = {oracle.call_count}")


if __name__ == "__main__":
    main()
