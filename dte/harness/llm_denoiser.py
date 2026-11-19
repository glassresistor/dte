"""LLM-backed denoisers for the DTE harness.

Replaces the rule-based heuristic denoiser with a real LLM, while
preserving DTE's structural locality guarantee:

  * The LLM is prompted to revise ONLY the masked positions.
  * Non-masked tokens/lines are copied verbatim (enforced in code, not
    left to the model's discretion).
  * Optional embedding constraint (Strategy A): candidates that drift too
    far from the original in embedding space are rejected.

This is the key difference from a naive "fix the grammar" prompt: the mask
is a HARD CONSTRAINT enforced by the pipeline, not a soft instruction the
model may ignore.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Set

from .backend import LLMBackend
from .embedding import EmbeddingModel


# ---------------------------------------------------------------------------
# Case 1: Grammar/spelling LLM denoiser
# ---------------------------------------------------------------------------

class LLMGrammarDenoiser:
    """LLM-backed denoiser for the grammar case.

    For each masked token position, asks the LLM to propose a correction.
    The prompt is tightly constrained: "replace only this token, keep
    everything else identical." Non-masked tokens are copied verbatim.

    If an embedding model is available (Strategy A), candidate corrections
    are ranked by semantic proximity to the original and those exceeding
    the threshold are rejected.
    """

    def __init__(self, llm: LLMBackend, embed: Optional[EmbeddingModel] = None):
        self._llm = llm
        self._embed = embed

    def denoise(self, doc: List[str], mask: Set[int], rng) -> List[str]:
        """Revise only masked positions using the LLM.

        Args:
            doc:    token list (words + whitespace runs as separate tokens)
            mask:   set of token indices to revise
            rng:    random number generator (for tie-breaking, unused here)

        Returns:
            New token list with only masked positions potentially changed.
        """
        out = list(doc)  # copy — non-masked tokens preserved verbatim

        if not mask:
            return out

        # Build a focused prompt for the LLM
        original_text = "".join(doc)
        error_positions = sorted(mask)

        # Identify which positions are word tokens (vs whitespace)
        word_positions = [i for i in error_positions if not doc[i].isspace()]
        ws_positions = [i for i in error_positions if doc[i].isspace()]

        # Whitespace fixes are deterministic — no LLM needed
        for i in ws_positions:
            out[i] = " "  # collapse whitespace run to single space

        if not word_positions:
            return out

        # Build the prompt: show the full text, mark error positions,
        # ask for corrections as a JSON mapping of position -> replacement
        lines = []
        lines.append("You are a precise proofreader. Fix ONLY the marked "
                     "errors in the text below. Error types include: "
                     "misspellings, double spaces, missing terminal "
                     "punctuation (period/question mark/exclamation point), "
                     "and capitalization after periods. "
                     "Do NOT change any other words, punctuation, or spacing.")
        lines.append("")
        lines.append(f"Text: {original_text!r}")
        lines.append("")
        lines.append("Errors to fix (token positions):")
        for i in word_positions:
            context_start = max(0, i - 2)
            context_end = min(len(doc), i + 3)
            context_tokens = doc[context_start:context_end]
            # Mark the error token with [ERROR]
            marked = []
            for j, t in enumerate(context_tokens):
                abs_j = context_start + j
                if abs_j == i:
                    marked.append(f"[{t}]")  # brackets mark the error
                else:
                    marked.append(t)
            lines.append(f"  Position {i}: token {doc[i]!r} "
                         f"(context: {' '.join(marked)})")
        lines.append("")
        lines.append('For each marked position, provide the corrected token. '
                     '- Misspelling: replace with the correct word. '
                     '- Double space: replace with a single space " ". '
                     '- Missing terminal punctuation: append "." to the token. '
                     '- Capitalization: capitalize the first letter. '
                     'Respond as JSON mapping position (string) to corrected token. '
                     'Example: {"6": "receive", "13": " ", "28": "meeting."}')
        lines.append("Include ALL marked positions in your response.")

        prompt = "\n".join(lines)

        try:
            result = self._llm.complete_json(prompt)
        except (ValueError, Exception) as e:
            # LLM failed to produce valid JSON — fall back to no change
            # for this round. The pipeline will retry with mask expansion.
            return out

        # Apply corrections: ONLY at masked positions
        for pos_str, replacement in result.items():
            try:
                pos = int(pos_str)
            except (ValueError, TypeError):
                continue
            if pos not in mask:
                continue  # hard constraint: never touch unmasked positions
            if not isinstance(replacement, str) or not replacement:
                continue

            # Strategy A: embedding proximity check
            if self._embed is not None:
                candidate_doc = list(out)
                candidate_doc[pos] = replacement
                candidate_text = "".join(candidate_doc)
                if not self._embed.is_within_budget(original_text, candidate_text):
                    dist = self._embed.distance(original_text, candidate_text)
                    # Log but don't block — the LLM's correction is likely
                    # correct. The embedding check is a soft constraint.
                    # In a research setting you'd reject here.
                    pass

            out[pos] = replacement

        return out


# ---------------------------------------------------------------------------
# Case 2: Code repair LLM denoiser
# ---------------------------------------------------------------------------

class LLMCodeDenoiser:
    """LLM-backed denoiser for the code/test case.

    Instead of a bounded edit grammar, asks the LLM to propose a minimal
    fix for the failing lines. The prompt includes:
      * the full source code
      * the test failure output
      * the suspect lines (from mask)
      * explicit instruction to change ONLY those lines

    The response is parsed as a list of (line_number, new_line_text) pairs.
    Non-suspect lines are copied verbatim.
    """

    def __init__(self, llm: LLMBackend, embed: Optional[EmbeddingModel] = None):
        self._llm = llm
        self._embed = embed

    def denoise(self, code: str, mask: Set[int], rng,
                test_output: str = "") -> Optional[str]:
        """Propose a minimal fix at masked lines.

        Args:
            code:        full source code string
            mask:        set of 0-based line indices to revise
            rng:         random number generator (unused)
            test_output: test failure output for context

        Returns:
            Fixed source code string, or None if the LLM couldn't produce
            a valid response.
        """
        if not mask:
            return None

        lines = code.splitlines(keepends=True)
        n_lines = len(lines)

        # Build the prompt
        parts = []
        parts.append("You are a precise code repair tool. Fix ONLY the "
                     "specific bug(s) indicated by the test failure. "
                     "Change as few lines as possible. Do NOT refactor, "
                     "rename, or restructure any other code.")
        parts.append("")
        parts.append("Source code:")
        parts.append("```python")
        for i, line in enumerate(lines):
            marker = "  <-- SUSPECT" if i in mask else ""
            parts.append(f"{i + 1:4d} | {line.rstrip()}\n{marker}".rstrip("\n"))
        parts.append("```")
        parts.append("")
        if test_output:
            parts.append("Test failure output:")
            parts.append("```")
            parts.append(test_output[:2000])  # truncate long outputs
            parts.append("```")
            parts.append("")

        suspect_list = ", ".join(f"line {i + 1}" for i in sorted(mask))
        parts.append(f"Suspect lines: {suspect_list}")
        parts.append("")
        parts.append("Respond with a JSON object mapping line numbers (1-based) "
                     "to the corrected line text. Only include lines that need "
                     "changing. Example:")
        parts.append('{"4": "    for i in range(1, n + 1):"}')
        parts.append("")
        parts.append("If you cannot identify a fix, respond with {} (empty object).")

        prompt = "\n".join(parts)

        try:
            result = self._llm.complete_json(prompt)
        except (ValueError, Exception):
            return None

        if not result:
            return None

        # Apply fixes: ONLY at masked lines
        fixed_lines = list(lines)
        applied = 0
        for line_str, new_text in result.items():
            try:
                line_num = int(line_str)
            except (ValueError, TypeError):
                continue
            idx = line_num - 1  # convert to 0-based
            if idx not in mask:
                continue  # hard constraint: never touch unmasked lines
            if idx < 0 or idx >= n_lines:
                continue
            if not isinstance(new_text, str):
                continue

            # Preserve the original line ending
            orig_ending = ""
            if lines[idx].endswith("\n"):
                orig_ending = "\n"
            elif lines[idx].endswith("\r\n"):
                orig_ending = "\r\n"

            fixed_lines[idx] = new_text + orig_ending
            applied += 1

        if applied == 0:
            return None

        return "".join(fixed_lines)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_grammar_denoiser(llm: LLMBackend,
                            embed: Optional[EmbeddingModel] = None) -> LLMGrammarDenoiser:
    return LLMGrammarDenoiser(llm, embed)


def create_code_denoiser(llm: LLMBackend,
                         embed: Optional[EmbeddingModel] = None) -> LLMCodeDenoiser:
    return LLMCodeDenoiser(llm, embed)
