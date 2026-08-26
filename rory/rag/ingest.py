"""Load markdown from knowledge/ -> header-aware chunk -> embed -> write the index.

Runnable as `python -m rory.rag.ingest`. Everything this writes to data/ is
derived and disposable: delete it and re-run this to rebuild it from scratch.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rory.rag.embed import Embedder, FastEmbedder

KNOWLEDGE_DIR = Path("knowledge")
INDEX_PATH = Path("data/index.npz")
CHUNKS_PATH = Path("data/chunks.json")

# There's no tokenizer in this project's dependency list, so chunk size is
# measured in words as a stand-in for tokens (roughly 0.7-0.8 tokens/word for
# English BPE tokenizers, so ~300 words undershoots 400 tokens slightly —
# erring smaller keeps chunks focused, which matters more than hitting 400
# exactly). 15% overlap so a fact near a chunk boundary still appears whole in
# at least one chunk.
CHUNK_WORDS = 300
OVERLAP_WORDS = 45

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Section:
    heading_path: list[str]
    lines: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    source: str
    heading: str
    text: str


def split_into_sections(markdown: str) -> list[Section]:
    """One section per heading, holding that heading's body text up to the
    next heading of any level. heading_path is the full breadcrumb down to
    the current heading, so a chunk under a level-3 heading still carries its
    level-1 and level-2 ancestors."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, text)
    current = Section(heading_path=[])

    for line in markdown.splitlines():
        match = _HEADING_RE.match(line)
        if not match:
            current.lines.append(line)
            continue

        if current.lines and any(l.strip() for l in current.lines):
            sections.append(current)

        level, text = len(match.group(1)), match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        current = Section(heading_path=[t for _, t in stack])

    if current.lines and any(l.strip() for l in current.lines):
        sections.append(current)
    return sections


def chunk_section(section: Section, source: str) -> list[Chunk]:
    words = " ".join(section.lines).split()
    heading = " > ".join(section.heading_path) or source

    if len(words) <= CHUNK_WORDS:
        return [Chunk(source=source, heading=heading, text=" ".join(words))]

    chunks = []
    start = 0
    step = CHUNK_WORDS - OVERLAP_WORDS
    while start < len(words):
        piece = words[start : start + CHUNK_WORDS]
        chunks.append(Chunk(source=source, heading=heading, text=" ".join(piece)))
        if start + CHUNK_WORDS >= len(words):
            break
        start += step
    return chunks


def chunk_file(path: Path) -> list[Chunk]:
    sections = split_into_sections(path.read_text(encoding="utf-8"))
    return [chunk for section in sections for chunk in chunk_section(section, path.name)]


def build_chunks(knowledge_dir: Path) -> list[Chunk]:
    # README.md documents the format; it isn't personal knowledge to retrieve.
    files = sorted(p for p in knowledge_dir.glob("*.md") if p.name != "README.md")
    return [chunk for path in files for chunk in chunk_file(path)]


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def run(embedder: Embedder, knowledge_dir: Path = KNOWLEDGE_DIR) -> int:
    chunks = build_chunks(knowledge_dir)
    if not chunks:
        print(f"No markdown files found in {knowledge_dir}/", file=sys.stderr)
        return 0

    vectors = normalize(embedder.embed([c.text for c in chunks]))

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(INDEX_PATH, vectors=vectors)
    CHUNKS_PATH.write_text(
        json.dumps([{"source": c.source, "heading": c.heading, "text": c.text} for c in chunks], indent=2),
        encoding="utf-8",
    )
    return len(chunks)


def main() -> None:
    print("Loading embedding model... (first run downloads it, be patient)")
    count = run(FastEmbedder())
    print(f"Indexed {count} chunks from {KNOWLEDGE_DIR}/ -> {INDEX_PATH}, {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
