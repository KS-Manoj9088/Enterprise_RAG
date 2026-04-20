"""
Unit tests for src/retriever.py — retriever creation and document retrieval (Steps 1-3).
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document


def _make_retriever_mock(return_docs):
    mock = MagicMock()
    mock.invoke.return_value = return_docs
    return mock


class TestGetRetriever:
    """Tests for get_retriever()."""

    def test_returns_retriever(self, mock_vectordb):
        from src.retriever import get_retriever
        retriever = get_retriever(mock_vectordb, top_k=3)
        assert retriever is not None
        mock_vectordb.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

    def test_top_k_respected(self, mock_vectordb):
        from src.retriever import get_retriever
        get_retriever(mock_vectordb, top_k=7)
        _, kwargs = mock_vectordb.as_retriever.call_args
        assert kwargs["search_kwargs"]["k"] == 7

    def test_default_top_k_is_3(self, mock_vectordb):
        from src.retriever import get_retriever
        get_retriever(mock_vectordb)
        _, kwargs = mock_vectordb.as_retriever.call_args
        assert kwargs["search_kwargs"]["k"] == 3


class TestRetrieveRelevantDocs:
    """Tests for retrieve_relevant_docs()."""

    def test_returns_list_of_documents(self):
        from src.retriever import retrieve_relevant_docs
        docs = [
            Document(page_content="Arjun Rao is a detective.", metadata={"source": "crime.txt"}),
            Document(page_content="Land dispute in the village.", metadata={"source": "land.txt"}),
        ]
        retriever = _make_retriever_mock(docs)
        result = retrieve_relevant_docs(retriever, "Who is Arjun Rao?")
        assert result == docs
        retriever.invoke.assert_called_once_with("Who is Arjun Rao?")

    def test_empty_result_when_no_match(self):
        from src.retriever import retrieve_relevant_docs
        retriever = _make_retriever_mock([])
        result = retrieve_relevant_docs(retriever, "Random irrelevant query")
        assert result == []

    def test_calls_retriever_with_exact_query(self):
        from src.retriever import retrieve_relevant_docs
        retriever = _make_retriever_mock([])
        query = "What happened at the crime scene?"
        retrieve_relevant_docs(retriever, query)
        retriever.invoke.assert_called_once_with(query)
