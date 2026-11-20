"""DTE realistic-scale experiment (Experiment 2).

Larger, more varied fixtures than the v1 smoke tests:
  * Grammar: multi-paragraph business document with ~12 distinct error types
  * Code:     a real multi-function module (~60 lines) with 5 different bug types

The grammar oracle uses a real English wordlist (wordlist_en.txt, extracted
from the system Hunspell dictionary) instead of v1's tiny hardcoded vocab.

Run:
  python -m dte.experiment.case1_grammar   # rule-based + LLM harness
  python -m dte.experiment.case2_code      # edit-grammar bandit + LLM harness
"""
