"""Load the index, embed a query, cosine search, apply the score threshold.

Retrieval is exposed as an ordinary tool (search_notes, below) rather than a
separate always-on pipeline. There is one path from "the model wants
personal context" to "it gets some text back": through the registry, like
every other tool. See docs/DECISIONS.md for why.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rory.rag.embed import Embedder, FastEmbedder
from rory.rag.ingest import CHUNKS_PATH, INDEX_PATH, normalize
from rory.tools.registry import tool

# Below this cosine similarity, a result is noise, not a match. bge-small (like
# most sentence embedders) has a high anisotropic baseline: cosine similarity
# does not sit near zero for unrelated text the way it intuitively "should."
# Calibrated by hand against this knowledge base — off-topic queries ("what's
# the capital of France", "recommend a pizza recipe") top out around 0.55,
# genuinely relevant queries start around 0.65, even loosely phrased. 0.58
# sits in that gap. Returning zero results below this line is correct
# behaviour — it's what lets Rory say "that's not in my notes" instead of
# answering from the nearest-but-irrelevant chunk.
MIN_SCORE = 0.58
TOP_K = 3

# search_notes's share of the registry's ~2KB total result budget. A chunk is
# ~300 words (~1800 chars) before this cap — returning TOP_K of those whole
# would blow the budget on its own, so each snippet is capped short enough
# that citing the heading and a relevant excerpt fits three of them.
SNIPPET_CHARS = 400


@dataclass
class Match:
    source: str
    heading: str
    text: str
    score: float


class NoIndex(Exception):
    pass


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise NoIndex(f"{path} does not exist — run `python -m rory.rag.ingest` first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_vectors(path: Path = INDEX_PATH) -> np.ndarray:
    if not path.exists():
        raise NoIndex(f"{path} does not exist — run `python -m rory.rag.ingest` first")
    return np.load(path)["vectors"]


def search(
    query: str,
    embedder: Embedder,
    vectors: np.ndarray,
    chunks: list[dict],
    k: int = TOP_K,
    min_score: float = MIN_SCORE,
) -> list[Match]:
    query_vector = normalize(embedder.embed([query]))[0]
    scores = vectors @ query_vector
    ranked = np.argsort(-scores)[:k]
    return [
        Match(source=chunks[i]["source"], heading=chunks[i]["heading"], text=chunks[i]["text"], score=float(scores[i]))
        for i in ranked
        if scores[i] >= min_score
    ]


_embedder: Embedder | None = None


def _default_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = FastEmbedder()
    return _embedder


@tool
def search_notes(query: str) -> dict:
    """Search the personal knowledge base (goals, projects, ideas, about-me
    notes). Use this for any question about the user's life, work, projects,
    plans, or opinions. If it returns no results, that fact is not in the
    notes — say so; do not guess."""
    try:
        chunks = load_chunks()
        vectors = load_vectors()
    except NoIndex as exc:
        return {"ok": False, "error": str(exc)}

    matches = search(query, _default_embedder(), vectors, chunks)
    return {
        "ok": True,
        "results": [
            {
                "source": m.source,
                "heading": m.heading,
                "text": m.text[:SNIPPET_CHARS],
                "score": round(m.score, 3),
            }
            for m in matches
        ],
    }
