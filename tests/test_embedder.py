"""
Unit tests for src/embedder.py — embedding + ChromaDB storage (Step D).
Uses a temp directory so no real files are written to the project.
Windows note: ChromaDB holds SQLite file locks, so we explicitly close
the client before the TemporaryDirectory tries to delete itself.
"""
import gc
import os
import tempfile
import pytest
from langchain_core.documents import Document


def _close_chroma(*dbs):
    """Release ChromaDB file handles so Windows can delete the temp dir."""
    for db in dbs:
        try:
            db._client.close()
        except Exception:
            pass
    gc.collect()


class TestGetEmbeddingModel:
    """Tests for get_embedding_model()."""

    def test_returns_embedding_model(self):
        from src.embedder import get_embedding_model
        model = get_embedding_model()
        assert model is not None

    def test_model_can_embed_a_string(self):
        from src.embedder import get_embedding_model
        model = get_embedding_model()
        vector = model.embed_query("test sentence")
        assert isinstance(vector, list)
        assert len(vector) == 384  # MiniLM produces 384-dim vectors
        assert all(isinstance(v, float) for v in vector)

    def test_embedding_is_normalized(self):
        """Normalized embeddings have L2 norm ~1.0."""
        import math
        from src.embedder import get_embedding_model
        model = get_embedding_model()
        vec = model.embed_query("hello world")
        norm = math.sqrt(sum(v ** 2 for v in vec))
        assert abs(norm - 1.0) < 0.01, f"Expected normalized vector, got norm={norm}"

    def test_different_texts_produce_different_vectors(self):
        from src.embedder import get_embedding_model
        model = get_embedding_model()
        v1 = model.embed_query("machine learning")
        v2 = model.embed_query("ancient history")
        assert v1 != v2


class TestStoreAndLoadVectorDB:
    """Tests for store_in_vectordb() and load_vectordb()."""

    def test_store_creates_directory(self):
        from src.embedder import get_embedding_model, store_in_vectordb
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            persist_dir = os.path.join(tmpdir, "vectordb")
            embeddings = get_embedding_model()
            docs = [Document(page_content="Test content about Python.", metadata={"source": "test.txt"})]
            db = store_in_vectordb(docs, embeddings, persist_dir)
            assert os.path.exists(persist_dir)
            _close_chroma(db)

    def test_store_returns_chroma_instance(self):
        from src.embedder import get_embedding_model, store_in_vectordb
        from langchain_chroma import Chroma
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            persist_dir = os.path.join(tmpdir, "vectordb")
            embeddings = get_embedding_model()
            docs = [Document(page_content="Sample text.", metadata={"source": "sample.txt"})]
            db = store_in_vectordb(docs, embeddings, persist_dir)
            assert isinstance(db, Chroma)
            _close_chroma(db)

    def test_stored_count_matches_documents(self):
        from src.embedder import get_embedding_model, store_in_vectordb
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            persist_dir = os.path.join(tmpdir, "vectordb")
            embeddings = get_embedding_model()
            docs = [
                Document(page_content="First doc content.", metadata={"source": "a.txt"}),
                Document(page_content="Second doc content.", metadata={"source": "b.txt"}),
                Document(page_content="Third doc content.", metadata={"source": "c.txt"}),
            ]
            db = store_in_vectordb(docs, embeddings, persist_dir)
            assert db._collection.count() == 3
            _close_chroma(db)

    def test_load_vectordb_reads_stored_data(self):
        from src.embedder import get_embedding_model, store_in_vectordb, load_vectordb
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            persist_dir = os.path.join(tmpdir, "vectordb")
            embeddings = get_embedding_model()
            docs = [Document(page_content="Arjun Rao is a detective.", metadata={"source": "story.txt"})]
            db = store_in_vectordb(docs, embeddings, persist_dir)
            _close_chroma(db)

            # Load from disk
            loaded_db = load_vectordb(embeddings, persist_dir)
            assert loaded_db._collection.count() == 1
            _close_chroma(loaded_db)

    def test_similarity_search_returns_relevant_doc(self):
        from src.embedder import get_embedding_model, store_in_vectordb
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            persist_dir = os.path.join(tmpdir, "vectordb")
            embeddings = get_embedding_model()
            docs = [
                Document(page_content="Arjun Rao is a police inspector in Pune.", metadata={"source": "crime.txt"}),
                Document(page_content="The harvest season brings farmers together.", metadata={"source": "land.txt"}),
            ]
            db = store_in_vectordb(docs, embeddings, persist_dir)
            results = db.similarity_search("Who is the police inspector?", k=1)
            assert len(results) == 1
            assert "Arjun" in results[0].page_content
            _close_chroma(db)
