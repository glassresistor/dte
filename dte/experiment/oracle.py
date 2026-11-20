"""Practical oracles for DTE grammar task.

Replaces gold edit spans with realistic error-detection signals:
  * wordlist — flags tokens not in a large English dictionary (spelling errors)
  * heuristic — rule-based checks (double spaces, missing terminal punctuation,
                 common confusables, repeated words)
  * combined   — union of wordlist + heuristic (the default practical oracle)
  * languagetool — REST API call to a running LanguageTool server

Each function returns a list of spans: [{"start": int, "end": int}, ...]
where start/end are 0-based token indices (same format as gold edits).

Usage:
    from dte.experiment.oracle import get_oracle_spans
    spans = get_oracle_spans(sentence, method="combined")
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Wordlist loading
# ---------------------------------------------------------------------------

_wordlist_cache: Optional[set] = None


def _load_wordlist() -> set:
    """Load a comprehensive English wordlist.
    
    Priority: NLTK words corpus (236K entries) > local wordlist_en.txt fallback.
    """
    global _wordlist_cache
    if _wordlist_cache is not None:
        return _wordlist_cache
    try:
        from nltk.corpus import words
        _wordlist_cache = {w.lower() for w in words.words()}
    except Exception:
        # Fallback to local file
        local_path = os.path.join(os.path.dirname(__file__), "wordlist_en.txt")
        with open(local_path) as f:
            _wordlist_cache = {line.strip().lower() for line in f if line.strip()}
    return _wordlist_cache


# ---------------------------------------------------------------------------
# Common confusables / grammar heuristics
# ---------------------------------------------------------------------------

_CONFUSABLES = {
    "recieve": "receive", "definately": "definitely", "seperate": "separate",
    "occured": "occurred", "untill": "until", "wich": "which",
    "thier": "their", "beleive": "believe", "accomodate": "accommodate",
    "comming": "coming", "begining": "beginning", "calender": "calendar",
    "enviroment": "environment", "existance": "existence", "neccessary": "necessary",
    "persistant": "persistent", "priviledge": "privilege", "recomend": "recommend",
    "refered": "referred", "succesful": "successful", "tommorow": "tomorrow",
    "truely": "truly", "wierd": "weird", "achive": "achieve",
}

_PUNCT_END = set(['.', '!', '?', ';', ':'])


def _tokenize(text: str) -> List[str]:
    return text.split()


def _is_wordlike(tok: str) -> bool:
    """True if the token looks like it should be a word (contains letters)."""
    return any(c.isalpha() for c in tok)


def _strip_punct(tok: str) -> str:
    return re.sub(r"^[^a-zA-Z']+|[^a-zA-Z']+$", "", tok)


# ---------------------------------------------------------------------------
# Oracle methods
# ---------------------------------------------------------------------------

def wordlist_oracle(sentence: str) -> List[Dict]:
    """Flag tokens not in the English wordlist (potential spelling errors)."""
    wl = _load_wordlist()
    toks = _tokenize(sentence)
    spans = []
    for i, tok in enumerate(toks):
        clean = _strip_punct(tok).lower()
        if not clean or len(clean) <= 1:
            continue
        if not _is_wordlike(tok):
            continue
        if clean == tok and tok.isupper():
            continue
        # Check base form (strip common plural -s) as well
        base = clean[:-1] if clean.endswith("s") and len(clean) > 3 else clean
        if clean not in wl and base not in wl and clean not in _CONFUSABLES:
            spans.append({"start": i, "end": i + 1})
    return spans


def heuristic_oracle(sentence: str) -> List[Dict]:
    """Rule-based grammar heuristics. Returns token-index spans."""
    toks = _tokenize(sentence)
    spans = []

    # 1. Double spaces
    if "  " in sentence:
        pos = sentence.index("  ")
        before = sentence[:pos].split()
        idx = len(before)
        if idx < len(toks):
            spans.append({"start": idx, "end": idx + 1})

    # 2. Missing terminal punctuation
    stripped = sentence.rstrip()
    if toks and stripped and stripped[-1] not in _PUNCT_END:
        spans.append({"start": len(toks) - 1, "end": len(toks)})

    # 3. Repeated consecutive words
    for i in range(1, len(toks)):
        a = toks[i].lower().strip(".,!?")
        b = toks[i-1].lower().strip(".,!?")
        if a == b and _is_wordlike(toks[i]):
            spans.append({"start": i, "end": i + 1})

    # 4. Common confusables
    for i, tok in enumerate(toks):
        clean = _strip_punct(tok).lower()
        if clean in _CONFUSABLES:
            spans.append({"start": i, "end": i + 1})

    return spans


def combined_oracle(sentence: str) -> List[Dict]:
    """Union of wordlist + heuristic oracles. Deduplicates overlapping spans."""
    all_spans = wordlist_oracle(sentence) + heuristic_oracle(sentence)
    if not all_spans:
        return []
    all_spans.sort(key=lambda s: (s["start"], s["end"]))
    merged = [dict(all_spans[0])]
    for span in all_spans[1:]:
        last = merged[-1]
        if span["start"] <= last["end"]:
            last["end"] = max(last["end"], span["end"])
        else:
            merged.append(dict(span))
    return merged


def languagetool_oracle(sentence: str, url: str = "http://localhost:8823") -> List[Dict]:
    """Call a running LanguageTool REST API server. Falls back to combined."""
    try:
        import urllib.parse
        data = urllib.parse.urlencode({
            "text": sentence,
            "language": "en-US",
            "level": "Default",
        }).encode()
        req = urllib.request.Request(
            url + "/check", data=data, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
    except Exception:
        return combined_oracle(sentence)

    toks = _tokenize(sentence)
    tok_char_spans = []
    pos = 0
    for tok in toks:
        idx = sentence.index(tok, pos)
        tok_char_spans.append((idx, idx + len(tok)))
        pos = idx + 1

    def char_to_tok_idx(char_pos):
        for i, (cs, ce) in enumerate(tok_char_spans):
            if cs <= char_pos < ce:
                return i
        return len(toks) - 1

    spans = []
    for match in result.get("matches", []):
        offset = match.get("offset", 0)
        length = match.get("length", 1)
        tok_start = char_to_tok_idx(offset)
        tok_end = char_to_tok_idx(offset + length - 1) + 1
        spans.append({"start": tok_start, "end": tok_end})

    if not spans:
        return []
    spans.sort(key=lambda s: (s["start"], s["end"]))
    merged = [dict(spans[0])]
    for span in spans[1:]:
        last = merged[-1]
        if span["start"] <= last["end"]:
            last["end"] = max(last["end"], span["end"])
        else:
            merged.append(dict(span))
    return merged


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

ORACLE_METHODS = {"wordlist", "heuristic", "combined", "languagetool"}


def get_oracle_spans(sentence: str, method: str = "combined",
                     lt_url: str = "http://localhost:8823") -> List[Dict]:
    """Get oracle-flagged spans for a sentence."""
    if method == "wordlist":
        return wordlist_oracle(sentence)
    elif method == "heuristic":
        return heuristic_oracle(sentence)
    elif method == "combined":
        return combined_oracle(sentence)
    elif method == "languagetool":
        return languagetool_oracle(sentence, lt_url)
    else:
        raise ValueError(f"Unknown oracle method: {method}. Choose from {ORACLE_METHODS}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_sentences = [
        ("The team will recieve the report quickly  and will definately review it.",
         "Should flag: recieve, double-space area, definately"),
        ("This is a correct sentence with no errors at all.",
         "Should flag: nothing (or minimal)"),
        ("I went to the store and bought some apples.",
         "Should flag: nothing"),
    ]
    for sent, desc in test_sentences:
        print(f"\n{desc}")
        print(f"  Input: {sent!r}")
        for method in ["wordlist", "heuristic", "combined"]:
            spans = get_oracle_spans(sent, method)
            toks = sent.split()
            flagged = [f"[{s['start']}:{s['end']}]='{' '.join(toks[s['start']:s['end']])}'" for s in spans]
            print(f"  {method:12s}: {flagged if flagged else '[]'}")
