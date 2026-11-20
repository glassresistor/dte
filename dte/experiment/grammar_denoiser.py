"""Heuristic grammar denoiser using the real wordlist.

For each masked token, proposes a correction:
  1. If in the curated CORRECTIONS table -> use that.
  2. Else find edit-distance-1 neighbors within the wordlist (single-char
     substitution/insertion/deletion) and pick the best by local context.
  3. Whitespace runs -> collapse to single space (or single newline).
  4. Repeated word -> delete the duplicate (replace with empty, handled by
     the caller; here we mark it for removal).
  5. Missing terminal punctuation -> append ".".
  6. Capitalization after period -> capitalize first letter.

Non-masked tokens are copied verbatim (structural locality).
"""
from __future__ import annotations

import random
import re
from typing import List, Optional, Set

from .grammar_oracle import CORRECTIONS, VOCAB, _strip_punct


def _edit_distance_1_neighbors(word: str) -> List[str]:
    """Return wordlist words at edit distance exactly 1 from `word`.

    Only substitution and single-char insertion/deletion. This is the
    "dictionary lookup / edit-distance-1 neighbors" strategy from §8.2.
    Bounded by |alphabet|*len(word) candidates, filtered against VOCAB.
    """
    word = word.lower()
    if not re.fullmatch(r"[a-z]+", word):
        return []
    neighbors: Set[str] = set()
    letters = "abcdefghijklmnopqrstuvwxyz"

    # substitutions
    for pos in range(len(word)):
        for c in letters:
            if c != word[pos]:
                cand = word[:pos] + c + word[pos + 1:]
                if cand in VOCAB:
                    neighbors.add(cand)
    # deletion
    for pos in range(len(word)):
        cand = word[:pos] + word[pos + 1:]
        if cand and cand in VOCAB:
            neighbors.add(cand)
    # insertion
    for pos in range(len(word) + 1):
        for c in letters:
            cand = word[:pos] + c + word[pos + 1:]
            if cand in VOCAB:
                neighbors.add(cand)
    return list(neighbors)


def _pick_best_neighbor(word: str, context_words: List[str]) -> Optional[str]:
    """Pick the most plausible edit-distance-1 neighbor.

    Simple heuristic: prefer a neighbor that shares more prefix/suffix with
    the original (smaller perceptual change), then alphabetical for ties.
    Context is available for future use (e.g., POS-based scoring) but the
    barebones version uses shape similarity.
    """
    neighbors = _edit_distance_1_neighbors(word)
    if not neighbors:
        return None
    # score by longest common prefix+suffix with original
    def score(n: str) -> int:
        w = word.lower()
        pre = 0
        while pre < min(len(w), len(n)) and w[pre] == n[pre]:
            pre += 1
        suf = 0
        while suf < min(len(w), len(n)) - pre and w[-1 - suf] == n[-1 - suf]:
            suf += 1
        return -(pre + suf)  # more shared = better (lower score)
    neighbors.sort(key=lambda n: (score(n), n))
    return neighbors[0]


class GrammarDenoiser:
    """Local denoiser for the grammar case (real wordlist)."""

    def denoise(self, doc: List[str], mask: Set[int], rng: random.Random) -> List[str]:
        out = list(doc)
        # First pass: handle everything except repeated-word deletion
        # (deletion changes token count, handled specially below).
        for i in sorted(mask):
            tok = doc[i]
            if tok.isspace():
                # collapse whitespace: keep a single space, or single newline
                if "\n" in tok:
                    out[i] = "\n"
                else:
                    out[i] = " "
                continue

            core = _strip_punct(tok)
            # capitalization-after-period fix
            if self._is_cap_error(doc, i):
                out[i] = tok[0].upper() + tok[1:] if tok else tok
                continue

            # spelling fixes
            if core in CORRECTIONS:
                fixed = CORRECTIONS[core]
                if tok and tok[0].isupper():
                    fixed = fixed.capitalize()
                m = re.search(r"[^A-Za-z\']+$", tok)
                out[i] = fixed + (m.group(0) if m else "")
                continue

            # fallback: edit-distance-1 neighbor in the wordlist
            context = [doc[j] for j in range(max(0, i - 2), min(len(doc), i + 3))]
            neighbor = _pick_best_neighbor(core, context)
            if neighbor is not None:
                fixed = neighbor
                if tok and tok[0].isupper():
                    fixed = fixed.capitalize()
                m = re.search(r"[^A-Za-z\']+$", tok)
                out[i] = fixed + (m.group(0) if m else "")
            # else: leave unchanged (no confident candidate); verifier will
            # catch it and mask expansion / next iteration can retry.

        return out

    @staticmethod
    def _is_cap_error(doc: List[str], i: int) -> bool:
        """True if doc[i] is a lowercase word immediately after a period."""
        tok = doc[i]
        if not tok or tok.isspace() or not tok[0].isalpha():
            return False
        if tok[0].isupper():
            return False
        # walk back to the previous non-space token
        j = i - 1
        while j >= 0 and doc[j].isspace():
            j -= 1
        return j >= 0 and doc[j].endswith(".")


def apply_repeated_word_deletion(doc: List[str], mask: Set[int]) -> List[str]:
    """Handle repeated-word errors by deleting the duplicate token.

    This is the one edit type that changes token count. We remove the second
    occurrence (and its preceding single space) so "the the" -> "the".
    Called separately from GrammarDenoiser.denoise because it alters length.
    """
    from .grammar_oracle import _find_errors, _strip_punct
    errors = _find_errors(doc)
    out = list(doc)
    to_delete: List[int] = []
    for i in sorted(errors):
        if errors[i] == "repeated-word":
            # delete token i and the single space before it (i-1) if present
            to_delete.append(i)
            if i - 1 >= 0 and out[i - 1].isspace() and len(out[i - 1]) == 1:
                to_delete.append(i - 1)
    for idx in sorted(set(to_delete), reverse=True):
        del out[idx]
    return out
