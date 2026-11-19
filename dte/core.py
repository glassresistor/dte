"""DTE core: pipeline, oracle/corruptor/denoiser interfaces, metrics.

Maps to the concept model (REPORT_DTE.txt):
  * Oracle            -> section 3.2 (correctness signal + error mask)
  * Corruptor         -> section 3.4 step 1 (informed corruption / mask M)
  * Denoiser          -> section 3.4 step 2 (local denoising, T steps <= B)
  * Verifier          -> section 3.4 step 3 (re-run oracle; expand on fail)
  * Metrics           -> section 4.1 (ED, pass rate, edit precision, over-editing)

The pipeline is task-agnostic: concrete oracles/denoisers live in the
case modules. Everything here is pure Python stdlib.
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, List, Optional, Protocol, Sequence, Set, Tuple, TypeVar

from .ed import changed_positions, token_ed


T = TypeVar("T")  # document representation (list of tokens for Case 1)


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class Oracle(Protocol[T]):
    """Correctness oracle. DTE needs two views of it:

    * ok(doc) -> bool          : is the document correct? (O(y) in {0,1})
    * mask(doc) -> Set[int]    : where does the oracle say to touch? (M)

    The mask is what makes corruption *informed* rather than random.
    """
    def ok(self, doc: T) -> bool: ...
    def mask(self, doc: T) -> Set[int]: ...
    def feedback(self, doc: T) -> str:
        """Optional human-readable explanation (used for logs)."""
        return ""


class Denoiser(Protocol[T]):
    """Local denoiser. Given a document and a set of positions to revise,
    returns a *new* document in which ONLY those positions may have changed.

    Structural locality guarantee: non-masked positions are copied verbatim.
    """
    def denoise(self, doc: T, mask: Set[int], rng: random.Random) -> T: ...


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------

@dataclass
class Corruptor:
    """Informed corruption: mask exactly where the oracle flags errors.

    Optional stochasticity (ablation, section 7.4 / 5.4): with probability
    `stochastic_rate` per non-flagged position, also include it in the mask.
    This lets the model discover multi-token errors that look fine in
    isolation, at a small ED cost.
    """
    stochastic_rate: float = 0.0

    def make_mask(self, oracle: Oracle[T], doc: T, rng: random.Random) -> Set[int]:
        mask = set(oracle.mask(doc))
        if self.stochastic_rate > 0.0:
            n = len(doc)
            for i in range(n):
                if i not in mask and rng.random() < self.stochastic_rate:
                    mask.add(i)
        return mask


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class DTEResult(Generic[T]):
    """Outcome of a DTE run. All numbers here are computed at runtime."""
    original: T
    edited: T
    passed: bool
    iterations: int
    final_mask_size: int
    tokens_changed: int
    ed: int                      # token-level Levenshtein(original, edited)
    edit_precision: Optional[float]  # None if oracle can't certify necessity
    ground_truth_ed: Optional[int] = None

    @property
    def over_editing_rate(self) -> Optional[float]:
        return None if self.edit_precision is None else 1.0 - self.edit_precision


@dataclass
class DTEConfig:
    budget: int = 5                 # max denoising iterations (T <= B, section 2.3)
    stochastic_rate: float = 0.0    # ablation knob (section 7.4)
    seed: Optional[int] = None      # reproducibility
    expand_mask_fn: Optional[Callable[[Oracle[T], T, Set[int]], Set[int]]] = None


def run_dte(
    doc: T,
    oracle: Oracle[T],
    denoiser: Denoiser[T],
    config: Optional[DTEConfig] = None,
    log: Optional[List[str]] = None,
) -> DTEResult[T]:
    """The whole mechanism (Appendix B pseudocode, made concrete).

    1. informed corruption: mask = oracle.mask(x)
    2. denoising loop (<= budget iterations): revise ONLY masked positions,
       verify, and on failure expand the mask using oracle feedback.
    3. return edited doc + measured metrics (never fabricated).
    """
    config = config or DTEConfig()
    rng = random.Random(config.seed)
    corruptor = Corruptor(stochastic_rate=config.stochastic_rate)
    log = log if log is not None else []

    def say(msg: str) -> None:
        log.append(msg)

    mask = corruptor.make_mask(oracle, doc, rng)
    say(f"iter 0: informed corruption flagged {len(mask)} positions")
    current = doc
    passed = oracle.ok(current)
    if passed:
        say("input already passes the oracle; no edits needed")

    for it in range(1, config.budget + 1):
        if passed:
            break
        candidate = denoiser.denoise(current, mask, rng)
        # structural locality check (section 2.3): non-masked positions unchanged
        violated = [i for i in changed_positions(list(current), list(candidate))
                    if i not in mask]
        if violated:
            raise RuntimeError(
                f"denoiser touched unmasked positions {violated}; "
                "structural locality violated")
        candidate_ok = oracle.ok(candidate)
        say(f"iter {it}: revised {len(mask)} positions -> "
            f"{'PASS' if candidate_ok else 'FAIL'}")
        if candidate_ok:
            current, mask, passed = candidate, mask, True
            break
        # verification failed -> iterative mask expansion (section 3.4 step 3)
        if config.expand_mask_fn is not None:
            new_mask = config.expand_mask_fn(oracle, candidate, mask)
        else:
            new_mask = set(oracle.mask(candidate)) | mask
        say(f"iter {it}: expanding mask {len(mask)} -> {len(new_mask)}")
        current, mask = candidate, new_mask

    x_tok, y_tok = list(doc), list(current)
    n_iter = 0 if oracle.ok(doc) else it
    result = DTEResult(
        original=doc,
        edited=current,
        passed=passed,
        iterations=n_iter,
        final_mask_size=len(mask),
        tokens_changed=len(changed_positions(x_tok, y_tok)),
        ed=token_ed(" ".join(map(str, x_tok)), " ".join(map(str, y_tok))),
        edit_precision=None,
    )
    return result


def edit_precision(oracle: Oracle[T], original: T, edited: T) -> Optional[float]:
    """Fraction of changed tokens that are *necessary*: reverting a single
    change breaks the oracle (section 4.1).

    For position-aligned edits this is well-defined; returns None if no
    positions changed.
    """
    changed = changed_positions(list(original), list(edited))
    if not changed:
        return None
    necessary = 0
    for i in changed:
        probe = list(edited)
        probe[i] = original[i]  # revert just this one edit
        if not oracle.ok(probe):
            necessary += 1
    return necessary / len(changed)


def metrics_table(result: DTEResult[T], extra: Optional[dict] = None) -> str:
    """Render the section-4.1 metrics as a small table (all runtime-computed)."""
    rows = [
        ("Pass rate (O(y)=1)", "yes" if result.passed else "NO"),
        ("Edit distance (token Levenshtein)", result.ed),
        ("Tokens changed", result.tokens_changed),
        ("Denoising iterations", result.iterations),
        ("Final mask size", result.final_mask_size),
        ("Edit precision (necessary changes / all changes)",
         f"{result.edit_precision:.3f}" if result.edit_precision is not None else "n/a"),
        ("Over-editing rate (1 - precision)",
         f"{result.over_editing_rate:.3f}" if result.over_editing_rate is not None else "n/a"),
    ]
    if result.ground_truth_ed is not None:
        rows.append(("Ground-truth ED (planted errors)", result.ground_truth_ed))
    if extra:
        rows.extend(extra.items())
    width = max(len(k) for k, _ in rows)
    lines = ["| " + "-" * width + " | " + "-" * 28 + " |"]
    for k, v in rows:
        lines.append(f"| {k:<{width}} | {v} |")
    return "\n".join(lines)
