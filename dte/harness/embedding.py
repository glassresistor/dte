"""Embedding-based edit-distance constraints (Strategy A & B, §3.3).

Uses sentence-transformers to compute dense vectors for text, enabling:

  Strategy A — Embedding-space proximity as edit distance:
    d(x, y) = ||E(x) - E(y)||_2  (or cosine distance)
    During denoising, reject/adjust tokens that push the embedding too far
    from E(x). This is a *soft edit-budget constraint* in continuous space.

  Strategy B — Embedding-space interpolation:
    Compute E(x_corrected) with a strong model; decode the text closest to
    alpha * E(x) + (1 - alpha) * E(x_corrected), biasing toward the original.

The embedding model is loaded once and reused. You provide the model name
or local path — it is NOT downloaded by this module. If the model isn't
available locally, sentence-transformers will attempt to fetch it on first
use (standard HF behavior). To avoid any download, point DTE_EMBED_MODEL
at a local directory containing the model files.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from .config import EmbedConfig


class EmbeddingModel:
    """Wrapper around sentence-transformers with caching and distance helpers."""

    def __init__(self, config: EmbedConfig):
        if not config.enabled:
            return  # no-op mode

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. Install with:\n"
                "  pip install sentence-transformers\n"
                f"Original error: {e}"
            ) from e

        self._model = SentenceTransformer(config.model_name)
        self._threshold = config.threshold
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Encode text to a normalized embedding vector."""
        if use_cache and text in self._cache:
            return self._cache[text]
        vec = self._model.encode(text, normalize_embeddings=True)
        if use_cache:
            self._cache[text] = vec
        return vec

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of texts to normalized embedding vectors."""
        return self._model.encode(list(texts), normalize_embeddings=True)

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance = 1 - cosine_similarity. Range [0, 2]."""
        dot = float(np.dot(a, b))
        return 1.0 - dot

    @staticmethod
    def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two embedding vectors."""
        return float(np.linalg.norm(a - b))

    def is_within_budget(self, original_text: str, candidate_text: str,
                         metric: str = "cosine") -> bool:
        """Strategy A check: is the candidate within the edit budget?

        Returns True if the embedding distance between original and candidate
        is below the configured threshold. This acts as a soft constraint —
        candidates that drift too far from the original in embedding space
        are rejected, even if they pass the oracle.
        """
        e_orig = self.encode(original_text)
        e_cand = self.encode(candidate_text)
        if metric == "cosine":
            dist = self.cosine_distance(e_orig, e_cand)
        else:
            dist = self.l2_distance(e_orig, e_cand)
        return dist < self._threshold

    def distance(self, original_text: str, candidate_text: str,
                 metric: str = "cosine") -> float:
        """Compute the embedding distance between two texts."""
        e_orig = self.encode(original_text)
        e_cand = self.encode(candidate_text)
        if metric == "cosine":
            return self.cosine_distance(e_orig, e_cand)
        return self.l2_distance(e_orig, e_cand)

    def rank_candidates(self, original_text: str, candidates: List[str],
                        metric: str = "cosine") -> List[tuple[float, int]]:
        """Rank candidates by embedding distance to the original (ascending).

        Returns list of (distance, index) sorted closest-first. Used for
        Strategy A: prefer candidates that are semantically close to the
        original text.
        """
        e_orig = self.encode(original_text)
        scored = []
        for i, cand in enumerate(candidates):
            e_cand = self.encode(cand)
            if metric == "cosine":
                d = self.cosine_distance(e_orig, e_cand)
            else:
                d = self.l2_distance(e_orig, e_cand)
            scored.append((d, i))
        scored.sort(key=lambda x: x[0])
        return scored


def create_embedding_model(config: EmbedConfig) -> Optional[EmbeddingModel]:
    """Create the embedding model, or None if disabled."""
    if not config.enabled:
        return None
    return EmbeddingModel(config)
