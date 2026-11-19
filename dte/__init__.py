"""Diffusive Text Editing (DTE) — dependency-free prototype.

Implements the mechanism from REPORT_DTE.txt:
  oracle-guided informed corruption -> local denoising -> verification
  -> iterative mask expansion, with edit-budget control.

Components map to the concept model (see README.md and Appendix A):
  core.py          pipeline, corruptor, verifier interface, metrics
  ed.py            token-level Levenshtein + diff helpers
  case1_grammar.py rule-based grammar oracle + heuristic denoiser
  case2_code.py    test-suite oracle (subprocess+cache) + edit-grammar bandit
"""

__version__ = "0.1.0"
