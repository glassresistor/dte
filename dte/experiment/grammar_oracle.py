"""Grammar oracle using a real English wordlist.

Extends the v1 rule-based oracle with:
  * A proper wordlist (48k words) for spelling detection — not a tiny vocab
  * More error rules: double space, missing terminal punctuation,
    capitalization after period, repeated words ("the the")
  * A curated correction table for common misspellings (for the denoiser)

The oracle returns a per-token mask. ok(doc) is True iff the mask is empty,
so verification and masking stay perfectly consistent.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Wordlist loading (real English dictionary, not a toy vocab)
# ---------------------------------------------------------------------------

_WORDLIST_PATH = os.path.join(os.path.dirname(__file__), "wordlist_en.txt")


def _load_wordlist() -> frozenset:
    if not os.path.isfile(_WORDLIST_PATH):
        raise FileNotFoundError(
            f"Wordlist not found at {_WORDLIST_PATH}. "
            "It should be bundled with the experiment package.")
    with open(_WORDLIST_PATH, encoding="utf-8") as f:
        return frozenset(line.strip().lower() for line in f if line.strip())


VOCAB: frozenset = _load_wordlist()

# Curated correction table for common misspellings. The wordlist tells us a
# token is *wrong*; this table tells the denoiser what it should be. For
# tokens not in this table, the denoiser falls back to edit-distance-1
# neighbors within the wordlist (see grammar_denoiser.py).
CORRECTIONS: Dict[str, str] = {
    "recieve": "receive",
    "definately": "definitely",
    "concensus": "consensus",
    "seperate": "separate",
    "occured": "occurred",
    "beleive": "believe",
    "untill": "until",
    "thier": "their",
    "wich": "which",
    "comming": "coming",
    "accomodate": "accommodate",
    "calender": "calendar",
    "existance": "existence",
    "persistant": "persistent",
    "neccessary": "necessary",
    "occurence": "occurrence",
    "refered": "referred",
    "supposedly": "supposedly",  # identity (some dicts misspell)
}


def _strip_punct(tok: str) -> str:
    return re.sub(r"^[^A-Za-z\']+|[^A-Za-z\']+$", "", tok).lower()


# ---------------------------------------------------------------------------
# Tokenization (same as v1: whitespace runs kept as tokens)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
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
# Error detection
# ---------------------------------------------------------------------------

def _find_errors(doc: List[str]) -> Dict[int, str]:
    """Return {token_index: reason}. Deterministic."""
    errors: Dict[int, str] = {}

    # R1 spelling: word token whose core is not in the wordlist.
    for i, tok in enumerate(doc):
        if tok.isspace():
            continue
        core = _strip_punct(tok)
        if core and re.fullmatch(r"[A-Za-z\']+", core):
            if core not in VOCAB:
                errors[i] = "spelling"

    # R2 double space: a whitespace token longer than one char.
    for i, tok in enumerate(doc):
        if tok.isspace() and len(tok) > 1:
            # single newline is fine; flag runs of spaces/tabs or multiple newlines
            if re.search(r"[ \t]", tok) or tok.count("\n") > 1:
                errors[i] = "double-space"

    # R3 repeated word: same word token twice in a row (case-insensitive).
    for i in range(1, len(doc)):
        prev, cur = doc[i - 1], doc[i]
        if not prev.isspace() and not cur.isspace():
            # only flag if there's exactly one space between them
            if (i - 2 >= 0 and doc[i - 2].isspace() and len(doc[i - 2]) == 1):
                pc, cc = _strip_punct(prev), _strip_punct(cur)
                if pc and pc == cc:
                    errors[i] = "repeated-word"

    # R4 missing terminal punctuation: last non-space token lacks .!?
    last_idx = max((i for i, t in enumerate(doc) if not t.isspace()), default=None)
    if last_idx is not None and doc[last_idx] and doc[last_idx][-1] not in ".!?":
        errors[last_idx] = "missing-terminal-punctuation"

    # R5 capitalization after period: alphabetic word immediately after "."
    for i, tok in enumerate(doc):
        if tok.endswith("."):
            j = i + 1
            while j < len(doc) and doc[j].isspace():
                j += 1
            if (j < len(doc) and not doc[j].isspace()
                    and doc[j][0].isalpha() and not doc[j][0].isupper()):
                errors[j] = "capitalization-after-period"

    return errors


class GrammarOracle:
    """Rule-based grammar oracle over a token list (real wordlist)."""

    def mask(self, doc: List[str]) -> Set[int]:
        return set(_find_errors(doc))

    def ok(self, doc: List[str]) -> bool:
        return not _find_errors(doc)

    def feedback(self, doc: List[str]) -> str:
        errs = _find_errors(doc)
        if not errs:
            return "no errors"
        # group by reason for a compact summary
        by_reason: Dict[str, List[int]] = {}
        for i, why in sorted(errs.items()):
            by_reason.setdefault(why, []).append(i)
        parts = [f"{r}({len(idx)}): {doc[idx[0]]!r}" for r, idx in by_reason.items()]
        return "; ".join(parts)


def error_summary(doc: List[str]) -> Dict[str, int]:
    """Count errors by type (for reporting)."""
    errs = _find_errors(doc)
    counts: Dict[str, int] = {}
    for why in errs.values():
        counts[why] = counts.get(why, 0) + 1
    return counts
