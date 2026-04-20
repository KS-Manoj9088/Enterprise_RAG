"""
Unit tests for src/chunker.py — text chunking (Step C).
"""
import pytest
from langchain_core.documents import Document


def _make_docs(texts):
    return [Document(page_content=t, metadata={"source": f"doc{i}.txt"})
            for i, t in enumerate(texts)]


class TestChunkDocuments:
    """Tests for chunk_documents()."""

    def test_returns_list(self):
        from src.chunker import chunk_documents
        docs = _make_docs(["Short text."])
        result = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert isinstance(result, list)

    def test_small_document_produces_one_chunk(self):
        from src.chunker import chunk_documents
        docs = _make_docs(["This is a very short document."])
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=100)
        assert len(chunks) == 1
        assert "very short document" in chunks[0].page_content

    def test_large_document_is_split(self):
        from src.chunker import chunk_documents
        long_text = "Word " * 1000  # ~5000 chars
        docs = _make_docs([long_text])
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        from src.chunker import chunk_documents
        long_text = "A" * 3000
        docs = _make_docs([long_text])
        chunk_size = 500
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 50  # small tolerance

    def test_overlap_creates_shared_content(self):
        from src.chunker import chunk_documents
        # Create text where overlap would cause shared words between adjacent chunks
        text = " ".join([f"sentence{i}" for i in range(200)])
        docs = _make_docs([text])
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
        assert len(chunks) >= 2
        # Adjacent chunks should share some content due to overlap
        if len(chunks) >= 2:
            c1_end = chunks[0].page_content[-60:]
            c2_start = chunks[1].page_content[:60:]
            # At least some characters should appear in both (overlap)
            shared = set(c1_end.split()) & set(c2_start.split())
            assert len(shared) > 0, "Expected overlap between adjacent chunks"

    def test_multiple_documents_chunked(self):
        from src.chunker import chunk_documents
        docs = _make_docs(["First document. " * 100, "Second document. " * 100])
        chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=30)
        assert len(chunks) >= 2

    def test_source_metadata_preserved(self):
        from src.chunker import chunk_documents
        docs = _make_docs(["Content " * 200])
        docs[0].metadata["source"] = "myfile.txt"
        chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=0)
        for chunk in chunks:
            assert chunk.metadata.get("source") == "myfile.txt"

    def test_empty_documents_list(self):
        from src.chunker import chunk_documents
        chunks = chunk_documents([], chunk_size=500, chunk_overlap=50)
        assert chunks == []

    def test_zero_overlap_works(self):
        from src.chunker import chunk_documents
        docs = _make_docs(["Hello world. " * 100])
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=0)
        assert len(chunks) >= 1

    def test_chunk_count_gte_doc_count(self):
        from src.chunker import chunk_documents
        docs = _make_docs(["Text " * 500, "More " * 500])
        chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=30)
        assert len(chunks) >= len(docs)
