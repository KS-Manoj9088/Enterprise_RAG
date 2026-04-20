"""
Integration tests for src/pipeline.py — full RAG pipeline (end-to-end with mocks).
Windows note: ChromaDB holds SQLite file locks. We explicitly close the
vectordb client before TemporaryDirectory teardown to avoid PermissionError.
"""
import gc
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def _close_pipeline_chroma(rag):
    """Release ChromaDB file handles from a RAGPipeline instance."""
    if rag.vectordb is not None:
        try:
            rag.vectordb._client.close()
        except Exception:
            pass
    gc.collect()


class TestRagPipelineInit:
    """Tests for RAGPipeline.__init__()."""

    def test_default_paths(self):
        from src.pipeline import RAGPipeline
        rag = RAGPipeline()
        assert rag.data_path == "./data"
        assert rag.vectordb_path == "./storage/shared_vectors"

    def test_custom_paths(self):
        from src.pipeline import RAGPipeline
        rag = RAGPipeline(data_path="/tmp/data", vectordb_path="/tmp/vec")
        assert rag.data_path == "/tmp/data"
        assert rag.vectordb_path == "/tmp/vec"

    def test_default_model(self):
        from src.pipeline import RAGPipeline
        rag = RAGPipeline()
        assert rag.llm_model == "z-ai/glm-4.5-air:free"

    def test_components_are_none_before_ingest(self):
        from src.pipeline import RAGPipeline
        rag = RAGPipeline()
        assert rag.embeddings is None
        assert rag.vectordb is None
        assert rag.retriever is None
        assert rag.llm is None
        assert rag.rag_chain is None


class TestRagPipelineIngest:
    """Tests for RAGPipeline.ingest()."""

    def test_ingest_returns_false_when_no_documents(self):
        from src.pipeline import RAGPipeline
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            rag = RAGPipeline(data_path=tmpdir)
            result = rag.ingest()
            assert result is False

    def test_ingest_returns_true_with_documents(self, sample_txt_dir):
        from src.pipeline import RAGPipeline
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as vec_dir:
            rag = RAGPipeline(data_path=sample_txt_dir, vectordb_path=vec_dir)
            result = rag.ingest(chunk_size=300, chunk_overlap=30)
            assert result is True
            _close_pipeline_chroma(rag)

    def test_ingest_populates_vectordb(self, sample_txt_dir):
        from src.pipeline import RAGPipeline
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as vec_dir:
            rag = RAGPipeline(data_path=sample_txt_dir, vectordb_path=vec_dir)
            rag.ingest(chunk_size=300, chunk_overlap=30)
            assert rag.vectordb is not None
            assert rag.embeddings is not None
            _close_pipeline_chroma(rag)

    def test_ingest_populates_embeddings(self, sample_txt_dir):
        from src.pipeline import RAGPipeline
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as vec_dir:
            rag = RAGPipeline(data_path=sample_txt_dir, vectordb_path=vec_dir)
            rag.ingest()
            assert rag.embeddings is not None
            _close_pipeline_chroma(rag)


class TestRagPipelineSetupQa:
    """Tests for RAGPipeline.setup_qa()."""

    def test_setup_qa_returns_false_without_vectordb(self):
        from src.pipeline import RAGPipeline
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            rag = RAGPipeline(vectordb_path=os.path.join(tmpdir, "nonexistent"))
            with patch("src.pipeline.get_llm", return_value=MagicMock()):
                result = rag.setup_qa()
            assert result is False

    def test_setup_qa_creates_retriever(self, sample_txt_dir):
        from src.pipeline import RAGPipeline
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as vec_dir:
            with patch("src.pipeline.get_llm", return_value=MagicMock()), \
                 patch("src.pipeline.create_rag_chain", return_value=MagicMock()):
                rag = RAGPipeline(data_path=sample_txt_dir, vectordb_path=vec_dir)
                rag.ingest()
                rag.setup_qa(top_k=3)
                assert rag.retriever is not None
                _close_pipeline_chroma(rag)

    def test_setup_qa_creates_rag_chain(self, sample_txt_dir):
        from src.pipeline import RAGPipeline
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as vec_dir:
            with patch("src.pipeline.get_llm", return_value=MagicMock()), \
                 patch("src.pipeline.create_rag_chain", return_value=MagicMock()):
                rag = RAGPipeline(data_path=sample_txt_dir, vectordb_path=vec_dir)
                rag.ingest()
                rag.setup_qa()
                assert rag.rag_chain is not None
                _close_pipeline_chroma(rag)


class TestRagPipelineQuery:
    """Tests for RAGPipeline.query()."""

    def test_query_without_setup_triggers_setup(self, sample_txt_dir):
        from src.pipeline import RAGPipeline
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as vec_dir:
            rag = RAGPipeline(data_path=sample_txt_dir, vectordb_path=vec_dir)
            rag.ingest()

            mock_chain = MagicMock()
            mock_llm = MagicMock()
            sources = [Document(page_content="Relevant doc", metadata={"source": "a.txt"})]

            with patch("src.pipeline.get_llm", return_value=mock_llm), \
                 patch("src.pipeline.create_rag_chain", return_value=mock_chain), \
                 patch("src.pipeline.generate_response", return_value=("Mocked answer", sources)):
                answer, srcs = rag.query("What is this about?")

            assert isinstance(answer, str)
            assert isinstance(srcs, list)
            _close_pipeline_chroma(rag)

    def test_query_returns_not_ready_without_ingest(self):
        from src.pipeline import RAGPipeline
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            rag = RAGPipeline(vectordb_path=os.path.join(tmpdir, "noexist"))
            with patch("src.pipeline.get_llm", return_value=MagicMock()):
                answer, sources = rag.query("Any question?")
            assert "not ready" in answer.lower() or "ingest" in answer.lower()
            assert sources == []


# ── Legacy smoke tests ──

def test_loader_loads_txt():
    from src.loader import load_documents
    docs = load_documents("./data")
    assert len(docs) > 0
    assert docs[0].page_content


def test_chunker_splits():
    from src.loader import load_documents
    from src.chunker import chunk_documents
    docs = load_documents("./data")
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= len(docs)
    for chunk in chunks:
        assert len(chunk.page_content) <= 600


def test_embedding_model_loads():
    from src.embedder import get_embedding_model
    model = get_embedding_model()
    assert model is not None


def test_pipeline_init():
    from src.pipeline import RAGPipeline
    rag = RAGPipeline()
    assert rag.data_path == "./data"
    assert rag.vectordb_path == "./storage/shared_vectors"
    assert rag.llm_model == "z-ai/glm-4.5-air:free"


def test_env_has_api_key():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    assert key.startswith("sk-or-"), "OPENROUTER_API_KEY should be set in .env"
