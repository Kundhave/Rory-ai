import json

import numpy as np
import pytest

from rory.rag import ingest, retrieve
from tests.fakes import FakeEmbedder


def test_chunker_uses_real_headings_not_a_fixed_splitter():
    markdown = (
        "# Doc\n\n"
        "## Section A\n"
        + ("alpha " * 50)
        + "\n\n## Section B\n"
        + ("beta " * 50)
    )

    chunks = _chunk_text(markdown)

    assert {c.heading for c in chunks} == {"Doc > Section A", "Doc > Section B"}
    assert all("alpha" in c.text or "beta" in c.text for c in chunks)
    # Content from different sections must never share a chunk.
    assert not any("alpha" in c.text and "beta" in c.text for c in chunks)


def _chunk_text(markdown: str):
    sections = ingest.split_into_sections(markdown)
    return [chunk for section in sections for chunk in ingest.chunk_section(section, "doc.md")]


def test_long_section_is_split_with_overlap():
    markdown = "# Doc\n\n## Long\n" + " ".join(f"word{i}" for i in range(700))

    chunks = _chunk_text(markdown)

    assert len(chunks) > 1
    # The overlap region: the tail of chunk 1 should reappear at the head of chunk 2.
    tail_of_first = chunks[0].text.split()[-10:]
    assert any(w in chunks[1].text.split() for w in tail_of_first)


def test_short_section_is_a_single_chunk():
    markdown = "# Doc\n\n## Short\nJust a few words here."

    chunks = _chunk_text(markdown)

    assert len(chunks) == 1
    assert chunks[0].heading == "Doc > Short"


def test_ingest_writes_a_regenerable_index(tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "notes.md").write_text("# Notes\n\n## Redis\nRelay uses Redis for queueing.\n")

    monkeypatch.setattr(ingest, "INDEX_PATH", tmp_path / "data" / "index.npz")
    monkeypatch.setattr(ingest, "CHUNKS_PATH", tmp_path / "data" / "chunks.json")

    count = ingest.run(FakeEmbedder(), knowledge_dir=knowledge)

    assert count == 1
    vectors = np.load(ingest.INDEX_PATH)["vectors"]
    chunks = json.loads(ingest.CHUNKS_PATH.read_text())
    assert vectors.shape[0] == len(chunks) == 1
    # Normalised: every row should have unit length.
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_search_returns_the_relevant_chunk_above_threshold():
    chunks = [
        {"source": "a.md", "heading": "Relay", "text": "Relay uses Redis and Celery for queueing."},
        {"source": "b.md", "heading": "Weather", "text": "It might rain in Tokyo tomorrow."},
    ]
    embedder = FakeEmbedder()
    vectors = ingest.normalize(embedder.embed([c["text"] for c in chunks]))

    results = retrieve.search("what does Relay use for queueing", embedder, vectors, chunks, min_score=0.0)

    assert results[0].source == "a.md"


def test_search_returns_nothing_below_the_threshold_this_is_correct():
    chunks = [{"source": "a.md", "heading": "Relay", "text": "Relay uses Redis and Celery."}]
    embedder = FakeEmbedder()
    vectors = ingest.normalize(embedder.embed([c["text"] for c in chunks]))

    results = retrieve.search("completely unrelated query about baking bread", embedder, vectors, chunks, min_score=0.99)

    # Zero results is the mechanism that lets Rory say "not in my notes"
    # instead of answering from an irrelevant nearest neighbour.
    assert results == []


def test_search_notes_tool_reports_missing_index_as_a_result_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieve, "CHUNKS_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(retrieve, "INDEX_PATH", tmp_path / "missing.npz")

    from rory.tools.registry import dispatch

    result = dispatch("search_notes", {"query": "anything"})

    assert result["ok"] is False
    assert "ingest" in result["error"]


def test_search_notes_tool_returns_results_shape(monkeypatch):
    chunks = [{"source": "a.md", "heading": "Relay", "text": "Relay uses Redis for queueing"}]
    embedder = FakeEmbedder()
    vectors = ingest.normalize(embedder.embed([c["text"] for c in chunks]))

    monkeypatch.setattr(retrieve, "load_chunks", lambda: chunks)
    monkeypatch.setattr(retrieve, "load_vectors", lambda: vectors)
    monkeypatch.setattr(retrieve, "_default_embedder", lambda: embedder)

    from rory.tools.registry import dispatch

    result = dispatch("search_notes", {"query": "what does Relay use for queueing"})

    assert result["ok"] is True
    assert result["results"][0]["source"] == "a.md"
    assert "score" in result["results"][0]


def test_search_notes_registered_in_the_schema():
    from rory.tools.registry import schemas

    names = {s["name"] for s in schemas()}
    assert "search_notes" in names
