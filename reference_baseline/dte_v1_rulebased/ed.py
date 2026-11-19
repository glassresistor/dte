"""Token-level edit distance (Levenshtein) and diff helpers.

Pure stdlib DP implementation — no external deps. Used by metrics (§4.1):
  * Edit Distance (ED): token-level Levenshtein between x and y
  * Line-diff size for Case 2
"""
from __future__ import annotations

import difflib
from typing import List, Sequence, Tuple


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Classic DP Levenshtein distance. O(len(a)*len(b)) time, O(min) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # ensure `a` is the shorter one to keep the rolling row small
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, bj in enumerate(b, start=1):
        cur = [j]
        append = cur.append
        for i, ai in enumerate(a, start=1):
            cost = 0 if ai == bj else 1
            append(min(prev[i] + 1,          # deletion
                       cur[i - 1] + 1,       # insertion
                       prev[i - 1] + cost))  # substitution
        prev = cur
    return prev[-1]


def token_ed(x: str, y: str) -> int:
    """Token-level Levenshtein over whitespace tokens."""
    return levenshtein(x.split(), y.split())


def changed_positions(x: Sequence, y: Sequence) -> List[int]:
    """Indices i where x[i] != y[i] (position-aligned; assumes equal length).

    For DTE's structural locality check we compare aligned token streams:
    the prototype denoiser never inserts/deletes, so lengths match.
    """
    if len(x) != len(y):
        # fall back to a sequence matcher for insert/delete-heavy diffs
        sm = difflib.SequenceMatcher(a=list(x), b=list(y))
        changed = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                lo = max(i1, j1)
                hi = max(i2, j2)
                changed.extend(range(lo, hi))
        return sorted(set(changed))
    return [i for i in range(len(x)) if x[i] != y[i]]


def line_diff_lines(x: str, y: str) -> int:
    """Number of differing lines (insertions + deletions), via SequenceMatcher."""
    sm = difflib.SequenceMatcher(a=x.splitlines(), b=y.splitlines())
    n = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            n += max(i2 - i1, j2 - j1)
    return n


def unified_diff(x: str, y: str, label_x: str = "x", label_y: str = "y") -> str:
    """Human-readable unified diff string."""
    return "".join(difflib.unified_diff(
        x.splitlines(keepends=True), y.splitlines(keepends=True),
        fromfile=label_x, tofile=label_y))


def render_edit_report(x_tokens: List[str], y_tokens: List[str]) -> str:
    """Show changed tokens side-by-side (for the demo output)."""
    lines = []
    for i in changed_positions(x_tokens, y_tokens):
        xt = x_tokens[i] if i < len(x_tokens) else "<none>"
        yt = y_tokens[i] if i < len(y_tokens) else "<none>"
        lines.append(f"  [{i:>3}] {xt!r} -> {yt!r}")
    return "\n".join(lines) if lines else "  (no token changes)"
