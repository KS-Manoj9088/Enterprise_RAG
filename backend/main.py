"""
Enterprise RAG System — FastAPI Backend
========================================
Provides REST API for:
  - Google OAuth authentication
  - Per-user document upload (50MB limit)
  - OCR for scanned documents
  - Per-user RAG ingestion & querying
"""
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from backend.database import init_db
from backend.routes.auth import router as auth_router
from backend.routes.documents import router as docs_router
from backend.routes.rag import router as rag_router

# ── Create App ──
app = FastAPI(
    title="Enterprise RAG API",
    description="Private RAG system with Google OAuth, per-user document management, and OCR",
    version="2.0.0",
)

# ── CORS — allow React dev server ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://localhost:3000",   # CRA dev
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global error handler (ensures CORS headers on 500s) ──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# ── Register Routers ──
app.include_router(auth_router)
app.include_router(docs_router)
app.include_router(rag_router)


@app.on_event("startup")
def on_startup():
    """Initialize database tables on startup."""
    init_db()
    print("[OK] Database initialized")
    print("[OK] Enterprise RAG API ready")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/ocr/status")
def ocr_status():
    """Check if OCR (Tesseract) is available on this system."""
    from backend.ocr import is_tesseract_available
    available = is_tesseract_available()
    return {
        "available": available,
        "message": "Tesseract OCR is installed" if available else "Tesseract OCR is not installed. Install from https://github.com/tesseract-ocr/tesseract",
    }
