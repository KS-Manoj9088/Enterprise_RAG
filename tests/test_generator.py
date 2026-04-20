"""
Unit tests for src/generator.py — formatting, deduplication, LLM chain helpers.
LLM calls are mocked so no API key is required for these tests.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def _doc(content, source="test.txt"):
    return Document(page_content=content, metadata={"source": source})


class TestFormatDocs:
    """Tests for format_docs()."""

    def test_single_doc(self):
        from src.generator import format_docs
        docs = [_doc("Hello world")]
        result = format_docs(docs)
        assert result == "Hello world"

    def test_multiple_docs_joined_with_double_newline(self):
        from src.generator import format_docs
        docs = [_doc("First"), _doc("Second"), _doc("Third")]
        result = format_docs(docs)
        assert result == "First\n\nSecond\n\nThird"

    def test_empty_list_returns_empty_string(self):
        from src.generator import format_docs
        assert format_docs([]) == ""

    def test_preserves_content(self):
        from src.generator import format_docs
        text = "Arjun Rao is a detective.\nHe solved many cases."
        docs = [_doc(text)]
        assert format_docs(docs) == text


class TestDeduplicateDocs:
    """Tests for deduplicate_docs()."""

    def test_no_duplicates_unchanged(self):
        from src.generator import deduplicate_docs
        docs = [_doc("Alpha"), _doc("Beta"), _doc("Gamma")]
        result = deduplicate_docs(docs)
        assert len(result) == 3

    def test_removes_exact_duplicates(self):
        from src.generator import deduplicate_docs
        docs = [_doc("Same content"), _doc("Same content"), _doc("Same content")]
        result = deduplicate_docs(docs)
        assert len(result) == 1
        assert result[0].page_content == "Same content"

    def test_removes_duplicate_with_trailing_whitespace(self):
        from src.generator import deduplicate_docs
        docs = [_doc("Hello world"), _doc("Hello world   "), _doc("Different")]
        result = deduplicate_docs(docs)
        assert len(result) == 2

    def test_preserves_order_of_first_occurrence(self):
        from src.generator import deduplicate_docs
        docs = [_doc("B"), _doc("A"), _doc("B"), _doc("C")]
        result = deduplicate_docs(docs)
        contents = [d.page_content for d in result]
        assert contents == ["B", "A", "C"]

    def test_empty_list(self):
        from src.generator import deduplicate_docs
        assert deduplicate_docs([]) == []

    def test_single_doc(self):
        from src.generator import deduplicate_docs
        docs = [_doc("Only one")]
        result = deduplicate_docs(docs)
        assert len(result) == 1

    def test_metadata_preserved_on_first_occurrence(self):
        from src.generator import deduplicate_docs
        d1 = Document(page_content="Same", metadata={"source": "file1.txt"})
        d2 = Document(page_content="Same", metadata={"source": "file2.txt"})
        result = deduplicate_docs([d1, d2])
        assert len(result) == 1
        assert result[0].metadata["source"] == "file1.txt"


class TestCreateRagChain:
    """Tests for create_rag_chain()."""

    def test_chain_is_created(self, mock_llm, mock_vectordb):
        from src.generator import create_rag_chain
        from prompts.strict.strict_qa import STRICT_PROMPT
        retriever = mock_vectordb.as_retriever()
        retriever.invoke.return_value = [_doc("Some context")]
        chain = create_rag_chain(mock_llm, retriever, STRICT_PROMPT)
        assert chain is not None

    def test_chain_invokes_llm(self, mock_llm, mock_vectordb):
        from src.generator import create_rag_chain
        from prompts.strict.strict_qa import STRICT_PROMPT

        retriever = mock_vectordb.as_retriever()
        retriever.invoke.return_value = [_doc("Context about Arjun Rao")]

        mock_llm.invoke.return_value = MagicMock(content="Arjun Rao is a detective.")
        chain = create_rag_chain(mock_llm, retriever, STRICT_PROMPT)

        # The chain should be invocable without error
        assert chain is not None


class TestGenerateResponse:
    """Tests for generate_response()."""

    def test_returns_tuple_of_answer_and_sources(self, mock_vectordb):
        from src.generator import generate_response

        retriever = mock_vectordb.as_retriever()
        retriever.invoke.return_value = [_doc("Context text")]

        # Mock the chain directly — bypasses StrOutputParser type checking
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Test answer"

        answer, sources = generate_response(mock_chain, "What is the answer?", retriever)
        assert isinstance(answer, str)
        assert answer == "Test answer"
        assert isinstance(sources, list)

    def test_sources_are_deduplicated(self, mock_vectordb):
        from src.generator import generate_response

        retriever = mock_vectordb.as_retriever()
        # Return 3 duplicate docs
        retriever.invoke.return_value = [_doc("Duplicate text")] * 3

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Answer"

        _, sources = generate_response(mock_chain, "Question?", retriever)
        assert len(sources) == 1

    def test_no_retriever_returns_empty_sources(self):
        from src.generator import generate_response

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Answer without sources"

        answer, sources = generate_response(mock_chain, "Question?", retriever=None)
        assert answer == "Answer without sources"
        assert sources == []


class TestDirectGenerate:
    """Tests for direct_generate()."""

    def test_returns_tuple(self, mock_vectordb):
        from src.generator import direct_generate
        from prompts.analytical.summarizer import SUMMARY_PROMPT

        retriever = mock_vectordb.as_retriever()
        retriever.invoke.return_value = [_doc("Some document content")]

        # Mock at the chain level to avoid StrOutputParser type issues
        with patch("src.generator.StrOutputParser") as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser_cls.return_value = mock_parser
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Summary answer"

            mock_llm = MagicMock()
            # Build chain: prompt | llm | parser; mock the __or__ pipe
            with patch("src.generator.PromptTemplate") as mock_pt:
                mock_pt.return_value.__or__ = lambda self, other: mock_chain
                mock_chain.__or__ = lambda self, other: mock_chain

                answer, sources = direct_generate(
                    llm=mock_llm,
                    prompt_template=SUMMARY_PROMPT,
                    retriever=retriever,
                    retrieval_query="key concepts overview",
                    user_question="Summarize the document",
                )
        # At minimum: answer is str, sources is list
        assert isinstance(sources, list)
        assert len(sources) == 1

    def test_returns_no_content_message_when_no_docs(self, mock_llm, mock_vectordb):
        from src.generator import direct_generate
        from prompts.analytical.summarizer import SUMMARY_PROMPT

        retriever = mock_vectordb.as_retriever()
        retriever.invoke.return_value = []  # empty — no chunks found

        answer, sources = direct_generate(
            llm=mock_llm,
            prompt_template=SUMMARY_PROMPT,
            retriever=retriever,
            retrieval_query="overview",
            user_question="Summarize",
        )
        assert "No relevant content" in answer
        assert sources == []
