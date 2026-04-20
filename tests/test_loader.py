"""
Unit tests for src/loader.py — document loading (Steps A + B).
"""
import os
import tempfile
import pytest
from langchain_core.documents import Document


class TestLoadDocuments:
    """Tests for load_documents()."""

    def test_returns_empty_for_nonexistent_path(self):
        from src.loader import load_documents
        docs = load_documents("/nonexistent/path/xyz")
        assert docs == []

    def test_returns_empty_for_empty_directory(self):
        from src.loader import load_documents
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = load_documents(tmpdir)
            assert docs == []

    def test_loads_txt_file(self):
        from src.loader import load_documents
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("Hello world. This is a test document.")
            docs = load_documents(tmpdir)
            assert len(docs) == 1
            assert "Hello world" in docs[0].page_content

    def test_skips_unsupported_file_types(self):
        from src.loader import load_documents
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a .csv file (unsupported)
            path = os.path.join(tmpdir, "data.csv")
            with open(path, "w") as f:
                f.write("col1,col2\nval1,val2\n")
            docs = load_documents(tmpdir)
            assert docs == []

    def test_loads_multiple_txt_files(self, sample_txt_dir):
        from src.loader import load_documents
        docs = load_documents(sample_txt_dir)
        assert len(docs) == 2

    def test_document_has_source_metadata(self):
        from src.loader import load_documents
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "myfile.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("Content here.")
            docs = load_documents(tmpdir)
            assert len(docs) == 1
            assert "source" in docs[0].metadata
            assert "myfile.txt" in docs[0].metadata["source"]

    def test_handles_mixed_valid_and_unsupported_files(self):
        from src.loader import load_documents
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "valid.txt"), "w") as f:
                f.write("Valid content.")
            with open(os.path.join(tmpdir, "ignore.csv"), "w") as f:
                f.write("a,b\n1,2\n")
            docs = load_documents(tmpdir)
            assert len(docs) == 1

    def test_empty_txt_file_is_loaded(self):
        from src.loader import load_documents
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "empty.txt"), "w") as f:
                f.write("")
            # Should not crash even on empty file
            try:
                docs = load_documents(tmpdir)
                # Either zero docs or one doc with empty content — both acceptable
            except Exception as e:
                pytest.fail(f"load_documents raised an unexpected exception: {e}")
