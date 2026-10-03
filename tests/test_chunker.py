import pytest
from app.ingestion.chunking.splitter import chunk_text


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_paragraphs():
    doc = "Paragraph 1 about Kubernetes.\n\nParagraph 2 about Intel CPUs.\n\nParagraph 3 about Networking."
    chunks = chunk_text(doc, chunk_size=1500, chunk_overlap=0)
    assert len(chunks) == 1
    assert "Paragraph 1" in chunks[0]
    assert "Paragraph 3" in chunks[0]


def test_chunk_text_exceeds_size():
    p1 = "A" * 800
    p2 = "B" * 800
    doc = f"{p1}\n\n{p2}"
    chunks = chunk_text(doc, chunk_size=1000, chunk_overlap=0)
    assert len(chunks) == 2
    assert "A" in chunks[0]
    assert "B" in chunks[1]


def test_chunk_text_overlap():
    p1 = "Kubernetes pods manage containerized workloads efficiently across clusters."
    p2 = "Deployments ensure declarative updates for Pods and ReplicaSets."
    doc = f"{p1}\n\n{p2}"
    chunks = chunk_text(doc, chunk_size=80, chunk_overlap=20)
    assert len(chunks) >= 2


def test_chunk_sentence_boundaries():
    long_p = "First sentence is clear. Second sentence has technical facts. Third sentence provides guidance."
    chunks = chunk_text(long_p, chunk_size=50, chunk_overlap=0)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 60  # close to max size
