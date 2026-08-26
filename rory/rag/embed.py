"""Embedder Protocol + a local model via fastembed.

Local rather than an API: embedding is on the hot path for every retrieval
call (a query has to be embedded before it can be compared to anything), and
a local model means that path has no network dependency, no per-call cost,
and no latency variance from a remote provider. See docs/DECISIONS.md.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

# BAAI/bge-small-en-v1.5: 384 dimensions, ~130MB, fastembed's default —
# a reasonable quality/size tradeoff for a knowledge base of a few hundred
# chunks. The first call downloads it to the fastembed cache; expect a delay.
MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array, one row per input text."""
        ...


class FastEmbedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts)), dtype=np.float32)
