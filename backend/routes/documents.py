"""
Document Routes — Upload, list, delete documents (per-user).
"""
import os
import uuid
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Document
from backend.auth import get_current_user
from backend.ocr import extract_text_with_ocr, is_tesseract_available

router = APIRouter(prefix="/api/documents", tags=["documents"])

# ── Constants ──
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
UPLOAD_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "uploads")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


def get_user_data_path(user_id: str) -> str:
    """Get the data directory for a specific user."""
    path = os.path.join(UPLOAD_ROOT, user_id)
    os.makedirs(path, exist_ok=True)
    return path


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    use_ocr: bool = Query(False, description="Enable OCR for scanned documents"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload a document (PDF, TXT, or image).
    Max 50 MB. Optionally apply OCR for scanned content.
    """
    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Read entire file to check size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            413,
            f"File too large ({len(content) / (1024*1024):.1f} MB). Maximum allowed size is 50 MB.",
        )

    # Save to user folder
    user_dir = get_user_data_path(user.id)
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(user_dir, safe_name)

    with open(file_path, "wb") as f:
        f.write(content)

    # OCR processing (if requested)
    ocr_used = False
    if use_ocr:
        if not is_tesseract_available():
            # Still save the file, but warn about OCR
            pass  # Will use regular extraction at ingest time
        else:
            try:
                extracted = extract_text_with_ocr(file_path)
                if extracted:
                    # Save extracted text alongside original
                    txt_path = file_path + ".ocr.txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(extracted)
                    ocr_used = True
            except Exception as e:
                print(f"[OCR] Warning: OCR failed for {file.filename}: {e}")

    # Save metadata
    doc = Document(
        user_id=user.id,
        filename=safe_name,
        original_name=file.filename or "unknown",
        size_bytes=len(content),
        content_type=file.content_type,
        ocr_used=1 if ocr_used else 0,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {
        "id": doc.id,
        "filename": doc.original_name,
        "size_bytes": doc.size_bytes,
        "ocr_used": ocr_used,
        "message": "Document uploaded successfully",
    }


@router.get("/")
def list_documents(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all documents for the current user."""
    docs = db.query(Document).filter(Document.user_id == user.id).order_by(Document.uploaded_at.desc()).all()
    return [
        {
            "id": d.id,
            "filename": d.original_name,
            "size_bytes": d.size_bytes,
            "ocr_used": bool(d.ocr_used),
            "chunk_count": d.chunk_count,
            "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
        }
        for d in docs
    ]


@router.delete("/{doc_id}")
def delete_document(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a document, its stored file, and all its chunks from the vector DB."""
    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == user.id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Delete file from disk
    user_dir = get_user_data_path(user.id)
    file_path = os.path.join(user_dir, doc.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    # Also remove OCR text if it exists
    ocr_path = file_path + ".ocr.txt"
    if os.path.exists(ocr_path):
        os.remove(ocr_path)

    # ── Remove chunks from ChromaDB ──
    # Chunks are stored with a 'source' metadata field containing the full file path.
    # We delete every chunk whose source contains this document's filename.
    try:
        from src.embedder import get_embedding_model, load_vectordb
        from backend.routes.rag import get_user_vectordb_path, _user_pipelines
        import gc

        vectordb_path = get_user_vectordb_path(user.id)
        if os.path.exists(vectordb_path):
            # Use the cached vectordb if available, otherwise load it
            pipeline = _user_pipelines.get(user.id)
            if pipeline and "vectordb" in pipeline:
                vectordb = pipeline["vectordb"]
            else:
                embeddings = get_embedding_model()
                vectordb = load_vectordb(embeddings, vectordb_path)

            collection = vectordb._collection

            # Find all chunk IDs whose 'source' metadata contains this filename
            results = collection.get(
                where={"source": {"$contains": doc.filename}},
                include=["metadatas"],
            )
            chunk_ids = results.get("ids", [])
            if chunk_ids:
                collection.delete(ids=chunk_ids)
                print(f"[Delete] Removed {len(chunk_ids)} chunks for '{doc.original_name}' from vector DB")
            else:
                print(f"[Delete] No chunks found in vector DB for '{doc.original_name}'")

            # Evict the cached pipeline so it reloads fresh on next query
            evicted = _user_pipelines.pop(user.id, None)
            if evicted:
                try:
                    evicted["vectordb"]._client.close()
                except Exception:
                    pass
                del evicted
            gc.collect()
    except Exception as e:
        # Don't fail the whole delete if chunk cleanup fails — just log it
        print(f"[Delete] Warning: could not remove chunks from vector DB: {e}")

    db.delete(doc)
    db.commit()
    return {"message": "Document deleted"}