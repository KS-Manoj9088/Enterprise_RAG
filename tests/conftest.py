"""
Shared pytest fixtures used across all test modules.
"""
import os
import gc
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import User, Document


# ── In-memory SQLite DB for all tests (one engine per session) ──
@pytest.fixture(scope="session")
def db_engine():
    # StaticPool ensures ALL connections share the exact same in-memory database
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Import models before create_all so all tables are registered
    from backend.models import User, Document  # noqa: F401
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def _session_factory(db_engine):
    return sessionmaker(bind=db_engine)


@pytest.fixture
def db_session(_session_factory):
    """Function-scoped session that rolls back after each test."""
    session = _session_factory()
    yield session
    session.rollback()
    session.close()


# ── Sample user — session-scoped so it's inserted only once ──
@pytest.fixture(scope="session")
def sample_user(db_engine):
    """Create the test user once for the entire test session."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    user = User(
        id="test-user-001",
        email="test@example.com",
        name="Test User",
        picture="",
    )
    session.merge(user)  # upsert — safe on re-runs
    session.commit()
    session.close()
    # Return a detached instance
    Session2 = sessionmaker(bind=db_engine)
    s2 = Session2()
    u = s2.get(User, "test-user-001")
    s2.close()
    return u


# ── Temp directory with sample text files ──
@pytest.fixture
def sample_txt_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # File 1
        with open(os.path.join(tmpdir, "doc1.txt"), "w", encoding="utf-8") as f:
            f.write("Arjun Rao is a detective in Pune. He solved many cases.\n" * 30)
        # File 2
        with open(os.path.join(tmpdir, "doc2.txt"), "w", encoding="utf-8") as f:
            f.write("The land dispute in the village was settled after a long legal battle.\n" * 30)
        yield tmpdir


# ── Mock embedding model ──
@pytest.fixture
def mock_embeddings():
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 384
    mock.embed_documents.return_value = [[0.1] * 384, [0.2] * 384]
    return mock


# ── Mock LLM ──
@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Mocked LLM answer")
    return mock


# ── Mock ChromaDB vectorstore ──
@pytest.fixture
def mock_vectordb():
    mock = MagicMock()
    mock._collection.count.return_value = 10
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock.as_retriever.return_value = retriever
    return mock


def _close_chroma(vectordb):
    """Release all ChromaDB file handles so Windows can delete the temp dir."""
    try:
        vectordb._client.close()
    except Exception:
        pass
    try:
        del vectordb
    except Exception:
        pass
    gc.collect()


# ── FastAPI test client with overridden DB ──
@pytest.fixture
def api_client(db_engine, sample_user):
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.database import get_db
    from backend.auth import get_current_user
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db_engine)

    def override_db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    def override_user():
        # Re-fetch user in a fresh session each time
        session = Session()
        user = session.get(User, "test-user-001")
        session.close()
        return user

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
