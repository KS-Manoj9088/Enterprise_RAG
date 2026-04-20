"""
Integration tests for /api/documents/* endpoints.
Uses FastAPI TestClient with an in-memory SQLite DB and auth override.
"""
import io
import pytest


class TestUploadDocument:
    """Tests for POST /api/documents/upload."""

    def test_upload_txt_file_succeeds(self, api_client):
        data = {"file": ("test.txt", io.BytesIO(b"Hello world content"), "text/plain")}
        resp = api_client.post("/api/documents/upload", files=data)
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "test.txt"
        assert "id" in body
        assert body["message"] == "Document uploaded successfully"

    def test_upload_pdf_file_succeeds(self, api_client):
        # Minimal valid PDF bytes
        pdf_bytes = b"%PDF-1.4 1 0 obj<</Type/Catalog>>endobj\nxref\n0 1\n0000000000 65535 f\ntrailer<</Size 1>>\nstartxref\n9\n%%EOF"
        data = {"file": ("sample.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        resp = api_client.post("/api/documents/upload", files=data)
        # Upload should succeed regardless of PDF content validity
        assert resp.status_code == 200

    def test_upload_unsupported_extension_returns_400(self, api_client):
        data = {"file": ("data.csv", io.BytesIO(b"col1,col2\n1,2"), "text/csv")}
        resp = api_client.post("/api/documents/upload", files=data)
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_upload_returns_file_size(self, api_client):
        content = b"Some content here"
        data = {"file": ("sizes.txt", io.BytesIO(content), "text/plain")}
        resp = api_client.post("/api/documents/upload", files=data)
        assert resp.status_code == 200
        assert resp.json()["size_bytes"] == len(content)


class TestListDocuments:
    """Tests for GET /api/documents/."""

    def test_list_returns_empty_initially(self, api_client):
        resp = api_client.get("/api/documents/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_includes_uploaded_document(self, api_client):
        # Upload first
        data = {"file": ("list_test.txt", io.BytesIO(b"Content"), "text/plain")}
        api_client.post("/api/documents/upload", files=data)

        resp = api_client.get("/api/documents/")
        assert resp.status_code == 200
        filenames = [d["filename"] for d in resp.json()]
        assert "list_test.txt" in filenames

    def test_list_document_has_expected_fields(self, api_client):
        data = {"file": ("fields.txt", io.BytesIO(b"Data"), "text/plain")}
        api_client.post("/api/documents/upload", files=data)

        resp = api_client.get("/api/documents/")
        assert resp.status_code == 200
        docs = resp.json()
        if docs:
            doc = docs[0]
            assert "id" in doc
            assert "filename" in doc
            assert "size_bytes" in doc
            assert "uploaded_at" in doc


class TestDeleteDocument:
    """Tests for DELETE /api/documents/{doc_id}."""

    def test_delete_existing_document(self, api_client):
        # Upload
        data = {"file": ("to_delete.txt", io.BytesIO(b"Delete me"), "text/plain")}
        upload_resp = api_client.post("/api/documents/upload", files=data)
        doc_id = upload_resp.json()["id"]

        # Delete
        resp = api_client.delete(f"/api/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Document deleted"

    def test_delete_nonexistent_document_returns_404(self, api_client):
        resp = api_client.delete("/api/documents/nonexistent-id-xyz")
        assert resp.status_code == 404

    def test_deleted_document_not_in_list(self, api_client):
        # Upload
        data = {"file": ("gone.txt", io.BytesIO(b"Bye"), "text/plain")}
        upload_resp = api_client.post("/api/documents/upload", files=data)
        doc_id = upload_resp.json()["id"]

        # Delete
        api_client.delete(f"/api/documents/{doc_id}")

        # Verify gone
        list_resp = api_client.get("/api/documents/")
        ids = [d["id"] for d in list_resp.json()]
        assert doc_id not in ids
