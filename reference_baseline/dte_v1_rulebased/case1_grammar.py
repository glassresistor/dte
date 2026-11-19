"""Case 1 — Grammar/spelling DTE (build first, per section 8.2).

Components:
  * Oracle   : deterministic rule-based checker producing a per-token mask
  * Denoiser : dictionary / correction-table candidate selection, local only
  * Demo     : paragraph with exactly 4 planted surface errors (section 10);
               ground-truth ED = 4. Built-in assertions enforce the
               acceptance criteria of section 8.2:
                 (i)   output passes the oracle
                 (ii)  only masked tokens changed
                 (iii) reported ED == number of true errors
                 (iv)  all non-error tokens byte-identical

Run:  python -m dte.case1_grammar
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Set

from .core import DTEConfig, DTEResult, Oracle, edit_precision, metrics_table, run_dte
from .ed import render_edit_report

# ---------------------------------------------------------------------------
# Vocabulary & candidate generation (heuristic denoiser support)
# ---------------------------------------------------------------------------

VOCAB: frozenset = frozenset(w.lower() for w in """
the a an and or but if then else when while for to of in on at by with from
is are was were be been being am have has had do does did done will would can
could should may might must shall this that these those it its i you he she we
they me him her us them what which who whom whose as into over under again
there here all any both each few more most other some such no nor not only own
same so than too very s t don didn hasn haven isn wasn weren won wouldn
about above after against before below between during out off up down once
one two three first second last next new now just also well back even still
already because through without within along across behind beyond toward
made make makes making like look looks looking see sees seeing get gets getting
know knows knowing think thinks thinking want wants wanting need needs needing
take takes taking give gives giving find finds finding tell tells telling ask
asks asking say says saying work works working run runs running write writes
writing read reads reading use uses using call calls calling put puts putting
set sets setting come comes coming go goes going turn turns turning start
starts starting stop stops stopping open opens opening close closes closing
house home room door window table chair book page word letter name number
time day year week month morning evening night today tomorrow yesterday now
good bad great small big large little long short high low fast slow easy hard
simple clear true false right wrong correct incorrect complete incomplete
full empty half whole part piece point line space text sentence paragraph
grammar spelling error errors fix fixes fixing check checks checking proofread
person people man woman child children friend friends family team group
city town country world place location area region zone side end top bottom
front back left middle center core basic complex difficult important
necessary essential required optional available possible likely certain sure
definitely certainly probably perhaps maybe possibly actually really truly
quickly carefully slowly easily freely openly clearly simply directly
receive received receives receiving accept accepted accepts accepting
believe believed believes believing understand understood understands
understanding explain explained explains explaining describe described describes
describing define defined defines defining identify identified identifies
identifying correct corrected corrects correcting improve improved improves
improving ensure ensures ensuring verify verified verifies verifying test
tested tests testing pass passed passes passing fail failed fails failing
review reviews reviewed reviewing meeting meetings report reports before after
""".split())

# Curated correction table for common misspellings (stands in for a real
# spell-checker's edit-distance-1 candidate list). Deterministic.
CORRECTIONS: Dict[str, str] = {
    "recieve": "receive",
    "definately": "definitely",
    "seperate": "separate",
    "occured": "occurred",
    "beleive": "believe",
    "untill": "until",
    "thier": "their",
    "wich": "which",
}


def _strip_punct(tok: str) -> str:
    return re.sub(r"^[^A-Za-z\']+|[^A-Za-z\']+$", "", tok).lower()


# ---------------------------------------------------------------------------
# Tokenization (keeps whitespace runs as tokens so double-space is visible)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Tokenize keeping whitespace runs as explicit tokens.

    Tokens alternate word/whitespace. Non-masked tokens are copied verbatim,
    so reconstruction is byte-exact when nothing changes there.
    """
    tokens: List[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            j = i
            while j < n and not text[j].isspace():
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def reconstruct(tokens: List[str]) -> str:
    return "".join(tokens)


# ---------------------------------------------------------------------------
# Error detection (deterministic rules -> per-token mask)
# ---------------------------------------------------------------------------

def _find_errors(doc: List[str]) -> Dict[int, str]:
    """Return {token_index: reason}. Deterministic."""
    errors: Dict[int, str] = {}
    # R1 spelling: word token whose core is not in the vocabulary.
    # Words in CORRECTIONS are known misspellings — they ARE flagged.
    for i, tok in enumerate(doc):
        if tok.isspace():
            continue
        core = _strip_punct(tok)
        if core and re.fullmatch(r"[A-Za-z\']+", core):
            if core not in VOCAB:
                errors[i] = "spelling"
    # R2 double space: a whitespace token longer than one character
    # (our tokenizer keeps each whitespace run as a single token, so a
    # double-space appears as a 2-char whitespace token like "  ").
    for i, tok in enumerate(doc):
        if tok.isspace() and len(tok) > 1:
            errors[i] = "double-space"
    # R3 missing terminal punctuation: last non-space token lacks .!?
    last_idx = max((i for i, t in enumerate(doc) if not t.isspace()), default=None)
    if last_idx is not None and doc[last_idx] and doc[last_idx][-1] not in ".!?":
        errors[last_idx] = "missing-terminal-punctuation"
    # R4 capitalization after period: alphabetic word immediately after "."
    for i, tok in enumerate(doc):
        if tok.endswith("."):
            j = i + 1
            while j < len(doc) and doc[j].isspace():
                j += 1
            if (j < len(doc) and not doc[j].isspace()
                    and doc[j][0].isalpha() and not doc[j][0].isupper()):
                errors[j] = "capitalization-after-period"
    return errors


class GrammarOracle(Oracle[List[str]]):
    """Rule-based grammar oracle over a token list (section 8.2).

    ok(doc) is True iff the mask is empty, so verification and masking are
    perfectly consistent — exactly what the pipeline's verify step relies on.
    """

    def mask(self, doc: List[str]) -> Set[int]:
        return set(_find_errors(doc))

    def ok(self, doc: List[str]) -> bool:
        return not _find_errors(doc)

    def feedback(self, doc: List[str]) -> str:
        errs = _find_errors(doc)
        if not errs:
            return "no errors"
        return "; ".join(f"[{i}] {doc[i]!r}: {why}"
                         for i, why in sorted(errs.items()))


# ---------------------------------------------------------------------------
# Denoiser (section 8.2): local candidate selection, masked positions only
# ---------------------------------------------------------------------------

class GrammarDenoiser:
    """For each masked index, apply the minimal fix that resolves the
    oracle's reason at that position. Non-masked tokens are copied verbatim
    (structural locality, section 2.3)."""

    def denoise(self, doc: List[str], mask: Set[int], rng: random.Random) -> List[str]:
        out = list(doc)
        for i in sorted(mask):
            tok = doc[i]
            if tok.isspace():
                # double-space fix: collapse the whitespace run to one space
                out[i] = " "
                continue
            core = _strip_punct(tok)
            if core in CORRECTIONS:
                fixed = CORRECTIONS[core]
                if tok and tok[0].isupper():
                    fixed = fixed.capitalize()
                m = re.search(r"[^A-Za-z\']+$", tok)
                out[i] = fixed + (m.group(0) if m else "")
                continue
            # missing terminal punctuation on the final word token
            if i == len(doc) - 1 and core and re.fullmatch(r"[A-Za-z\']+", core):
                out[i] = tok + "."
        return out


# ---------------------------------------------------------------------------
# Demo (section 10 fixture: exactly 4 planted errors, ground-truth ED = 4)
# ---------------------------------------------------------------------------

def build_fixture() -> str:
    """Paragraph with exactly 4 planted surface errors:
      1. "recieve"     (misspelling of receive)
      2. double space  after "quickly"
      3. "definately"  (misspelling of definitely)
      4. missing terminal period at end of sentence
    All other words are in VOCAB so the oracle flags exactly these four.
    """
    return ("The team will recieve the report quickly  and will definately "
            "review it before the meeting")


def main() -> None:
    text = build_fixture()
    tokens = tokenize(text)
    oracle = GrammarOracle()

    print("=" * 72)
    print("DTE Case 1 — grammar/spelling (oracle-guided informed corruption)")
    print("=" * 72)
    print(f"Input ({len(tokens)} tokens):\n  {text!r}\n")

    log: List[str] = []
    result = run_dte(
        doc=tokens,
        oracle=oracle,
        denoiser=GrammarDenoiser(),
        config=DTEConfig(budget=5, seed=0),
        log=log,
    )
    for line in log:
        print("  " + line)

    edited_text = reconstruct(result.edited)
    print(f"\nOutput ({len(result.edited)} tokens):\n  {edited_text!r}\n")

    # edit precision (section 4.1)
    prec = edit_precision(oracle, result.original, result.edited)
    if prec is not None:
        object.__setattr__(result, "edit_precision", prec)

    print("Changed tokens:")
    print(render_edit_report(result.original, result.edited))
    print()
    print(metrics_table(result, extra={
        "Oracle feedback (final)": oracle.feedback(result.edited),
    }))

    # ------------------------------------------------------------------
    # Acceptance criteria (section 8.2) — built-in assertions
    # ------------------------------------------------------------------
    gt_ed = 4  # ground truth: exactly 4 planted errors
    result.ground_truth_ed = gt_ed

    assert result.passed, "acceptance (i): output must pass the oracle"
    changed = [i for i in range(len(result.original))
               if result.original[i] != result.edited[i]]
    flagged = oracle.mask(result.original)
    assert set(changed) <= flagged, (
        f"acceptance (ii): only masked tokens may change; "
        f"changed={changed}, mask={sorted(flagged)}")
    assert result.tokens_changed == gt_ed, (
        f"acceptance (iii): ED must equal ground-truth error count; "
        f"got {result.tokens_changed}, expected {gt_ed}")
    for i in range(len(result.original)):
        if i not in flagged:
            assert result.original[i] == result.edited[i], (
                f"acceptance (iv): token {i} outside mask was modified")

    print("\nAll acceptance criteria PASSED:")
    print("  (i)   oracle passes on output")
    print("  (ii)  only masked tokens changed")
    print(f"  (iii) ED == ground truth ({gt_ed})")
    print("  (iv)  all non-error tokens byte-identical")


if __name__ == "__main__":
    main()
