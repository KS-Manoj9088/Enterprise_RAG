"""
Integration tests for /api/rag/* endpoints.
Embedding model and LLM are mocked to avoid slow model loading or API calls.
"""
import io
import os
import pytest
from unittest.mock import MagicMock, patch


# ── Helper to mock the full RAG stack ──
def _patch_rag_stack(answer="Mocked answer", sources=None):
    """Context manager that patches embeddings, vectordb, and LLM."""
    if sources is None:
        from langchain_core.documents import Document
        sources = [Document(page_content="Relevant chunk.", metadata={"source": "doc.txt"})]

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 384

    mock_vectordb = MagicMock()
    mock_vectordb._collection.count.return_value = len(sources)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sources
    mock_vectordb.as_retriever.return_value = mock_retriever

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=answer)

    return (
        patch("backend.routes.rag.get_embedding_model", return_value=mock_embeddings),
        patch("backend.routes.rag.load_vectordb", return_value=mock_vectordb),
        patch("backend.routes.rag.get_llm", return_value=mock_llm),
        patch("backend.routes.rag.store_in_vectordb", return_value=mock_vectordb),
        patch("backend.routes.rag.load_documents", return_value=[
            MagicMock(page_content="Sample doc content", metadata={"source": "doc.txt"})
        ]),
        patch("backend.routes.rag.chunk_documents", return_value=[
            MagicMock(page_content="Sample chunk", metadata={"source": "doc.txt"})
        ]),
        patch("backend.routes.rag.generate_response", return_value=(answer, sources)),
    )


class TestRagStats:
    """Tests for GET /api/rag/stats."""

    def test_returns_stats_object(self, api_client):
        resp = api_client.get("/api/rag/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "vectors" in body
        assert "exists" in body

    def test_stats_vectors_is_integer(self, api_client):
        resp = api_client.get("/api/rag/stats")
        assert isinstance(resp.json()["vectors"], int)


class TestRagProviders:
    """Tests for GET /api/rag/providers."""

    def test_returns_providers_list(self, api_client):
        resp = api_client.get("/api/rag/providers")
        assert resp.status_code == 200
        body = resp.json()
        assert "providers" in body
        assert isinstance(body["providers"], list)

    def test_providers_contain_openrouter(self, api_client):
        resp = api_client.get("/api/rag/providers")
        ids = [p["id"] for p in resp.json()["providers"]]
        assert "openrouter" in ids

    def test_providers_contain_ollama(self, api_client):
        resp = api_client.get("/api/rag/providers")
        ids = [p["id"] for p in resp.json()["providers"]]
        assert "ollama" in ids

    def test_each_provider_has_required_fields(self, api_client):
        resp = api_client.get("/api/rag/providers")
        for provider in resp.json()["providers"]:
            assert "id" in provider
            assert "name" in provider
            assert "available" in provider
            assert "models" in provider


class TestRagIngest:
    """Tests for POST /api/rag/ingest."""

    def test_ingest_fails_when_no_documents_uploaded(self, api_client, tmp_path):
        # Point to an EMPTY temp dir so the "no documents" check triggers
        empty_dir = str(tmp_path / "empty_uploads")
        os.makedirs(empty_dir)
        with patch("backend.routes.rag.get_user_data_path", return_value=empty_dir):
            resp = api_client.post("/api/rag/ingest", json={})
        assert resp.status_code == 400

    def test_ingest_succeeds_with_mocked_stack(self, api_client, sample_txt_dir):
        # Make the user's upload dir point to our sample dir so files exist
        with patch("backend.routes.rag.get_user_data_path", return_value=sample_txt_dir), \
             patch("backend.routes.rag.get_user_vectordb_path", return_value=sample_txt_dir + "_vec"), \
             patch("backend.routes.rag.get_embedding_model", return_value=MagicMock()), \
             patch("backend.routes.rag.store_in_vectordb") as mock_store, \
             patch("backend.routes.rag.load_documents") as mock_load, \
             patch("backend.routes.rag.chunk_documents") as mock_chunk, \
             patch("shutil.rmtree"), \
             patch("os.makedirs"):

            from langchain_core.documents import Document
            mock_load.return_value = [Document(page_content="Text", metadata={"source": "a.txt"})]
            mock_chunk.return_value = [Document(page_content="Chunk", metadata={"source": "a.txt"})]
            mock_db = MagicMock()
            mock_db._collection.count.return_value = 1
            mock_store.return_value = mock_db

            resp = api_client.post("/api/rag/ingest", json={
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "model": "z-ai/glm-4.5-air:free",
                "provider": "openrouter",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["documents_loaded"] == 1
        assert body["chunks_created"] == 1


class TestRagQuery:
    """Tests for POST /api/rag/query."""

    def test_query_fails_when_no_vectordb(self, api_client, tmp_path):
        # Point to a path that does NOT exist so the "no ingested data" check triggers
        nonexistent = str(tmp_path / "no_vectors_here")
        with patch("backend.routes.rag.get_user_vectordb_path", return_value=nonexistent):
            resp = api_client.post("/api/rag/query", json={
                "question": "Who is Arjun?",
                "model": "z-ai/glm-4.5-air:free",
                "provider": "openrouter",
            })
        assert resp.status_code == 400

    def test_query_returns_answer_and_sources(self, api_client, tmp_path):
        vec_path = str(tmp_path / "vectors")
        os.makedirs(vec_path)

        from langchain_core.documents import Document
        sources = [Document(page_content="Arjun Rao is a detective.", metadata={"source": "crime.txt"})]

        with patch("backend.routes.rag.get_user_vectordb_path", return_value=vec_path), \
             patch("backend.routes.rag.get_embedding_model", return_value=MagicMock()), \
             patch("backend.routes.rag.load_vectordb") as mock_load_db, \
             patch("backend.routes.rag.get_llm", return_value=MagicMock()), \
             patch("backend.routes.rag.generate_response", return_value=("Arjun Rao is a detective.", sources)):

            mock_db = MagicMock()
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = sources
            mock_db.as_retriever.return_value = mock_retriever
            mock_load_db.return_value = mock_db

            resp = api_client.post("/api/rag/query", json={
                "question": "Who is Arjun Rao?",
                "model": "z-ai/glm-4.5-air:free",
                "provider": "openrouter",
                "prompt_style": "strict",
            })

        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert "time_seconds" in body
        assert isinstance(body["sources"], list)

    def test_empty_model_returns_400(self, api_client, tmp_path):
        vec_path = str(tmp_path / "vectors")
        os.makedirs(vec_path)

        with patch("backend.routes.rag.get_user_vectordb_path", return_value=vec_path):
            resp = api_client.post("/api/rag/query", json={
                "question": "Test",
                "model": "",
                "provider": "openrouter",
            })
        assert resp.status_code == 400
        assert "No model selected" in resp.json()["detail"]

    def test_document_id_filter_applied(self, api_client, db_engine, sample_user, tmp_path):
        """When document_id is passed, query should scope to that document."""
        from sqlalchemy.orm import sessionmaker
        from backend.models import Document as DocModel
        vec_path = str(tmp_path / "vectors")
        os.makedirs(vec_path)

        # Register a document in DB
        Session = sessionmaker(bind=db_engine)
        session = Session()
        doc = DocModel(
            id="doc-filter-001",
            user_id=sample_user.id,
            filename="abc123_crime_story.pdf",
            original_name="crime_story.pdf",
            size_bytes=1000,
        )
        session.merge(doc)
        session.commit()
        session.close()

        from langchain_core.documents import Document as LCDoc
        sources = [LCDoc(page_content="Crime content", metadata={"source": "abc123_crime_story.pdf"})]

        import backend.routes.rag as rag_module

        # Clear pipeline cache so a fresh retriever is always built
        rag_module._user_pipelines.clear()

        with patch("backend.routes.rag.get_user_vectordb_path", return_value=vec_path), \
             patch("backend.routes.rag.get_embedding_model", return_value=MagicMock()), \
             patch("backend.routes.rag.load_vectordb") as mock_load_db, \
             patch("backend.routes.rag.get_llm", return_value=MagicMock()), \
             patch("backend.routes.rag.generate_response", return_value=("Answer", sources)):

            mock_db = MagicMock()
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = sources
            mock_db.as_retriever.return_value = mock_retriever
            mock_load_db.return_value = mock_db

            resp = api_client.post("/api/rag/query", json={
                "question": "What happened in the crime story?",
                "model": "z-ai/glm-4.5-air:free",
                "provider": "openrouter",
                "document_id": "doc-filter-001",
            })

        assert resp.status_code == 200
        # Verify as_retriever was called with a filter containing the document filename
        assert mock_db.as_retriever.called, "Expected as_retriever to be called on the vectordb"
        call_kwargs = mock_db.as_retriever.call_args.kwargs
        search_kw = call_kwargs.get("search_kwargs", {})
        assert "filter" in search_kw, "Expected a metadata filter when document_id is set"
        assert "abc123_crime_story.pdf" in str(search_kw["filter"])
