"""
RAG Routes — Ingest documents & query the RAG pipeline (per-user).
"""
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database import get_db
from backend.models import User, Document
from backend.auth import get_current_user
from backend.routes.documents import get_user_data_path

# ── RAG imports (reusing existing src/) ──
from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import get_embedding_model, store_in_vectordb, load_vectordb
from src.retriever import get_retriever
from src.generator import get_llm, create_rag_chain, generate_response, is_ollama_available, list_ollama_models

router = APIRouter(prefix="/api/rag", tags=["rag"])

# ── In-memory cache for per-user pipelines ──
_user_pipelines: dict = {}


def get_user_vectordb_path(user_id: str) -> str:
    """Each user gets their own vector DB."""
    path = os.path.join("storage", "vectors", user_id)
    os.makedirs(path, exist_ok=True)
    return path


class IngestRequest(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 100
    model: str = "z-ai/glm-4.5-air:free"
    provider: str = "openrouter"  # 'openrouter' or 'ollama'


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    model: str = "z-ai/glm-4.5-air:free"
    provider: str = "openrouter"  # 'openrouter' or 'ollama'


class IngestResponse(BaseModel):
    message: str
    documents_loaded: int
    chunks_created: int
    vectors_stored: int


class QueryResponse(BaseModel):
    answer: str
    sources: list
    time_seconds: float


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(
    body: IngestRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Ingest all of the user's uploaded documents into their private vector DB.
    A → B → C → D pipeline.
    """
    user_data_path = get_user_data_path(user.id)
    vectordb_path = get_user_vectordb_path(user.id)

    # Check for documents
    if not os.path.exists(user_data_path) or not os.listdir(user_data_path):
        raise HTTPException(400, "No documents uploaded. Upload files first.")

    # Step A+B: Load documents
    documents = load_documents(user_data_path)
    if not documents:
        raise HTTPException(400, "Could not extract text from any document.")

    # Step C: Chunk
    chunks = chunk_documents(documents, body.chunk_size, body.chunk_overlap)

    # Step D: Embed + store
    embeddings = get_embedding_model()
    vectordb = store_in_vectordb(chunks, embeddings, vectordb_path)

    # Update chunk counts in DB
    user_docs = db.query(Document).filter(Document.user_id == user.id).all()
    chunks_per_doc = len(chunks) // max(len(user_docs), 1)
    for doc in user_docs:
        doc.chunk_count = chunks_per_doc
    db.commit()

    # Clear cached pipeline so it reloads
    _user_pipelines.pop(user.id, None)

    vector_count = vectordb._collection.count()
    return {
        "message": "Ingestion complete",
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "vectors_stored": vector_count,
    }


@router.post("/query", response_model=QueryResponse)
def query_rag(
    body: QueryRequest,
    user: User = Depends(get_current_user),
):
    """
    Query the user's private RAG pipeline.
    1 → 2 → 3 → 4 → 5 pipeline.
    """
    import time
    import traceback
    from dotenv import load_dotenv

    # Always re-read .env so key changes take effect without server restart
    load_dotenv(override=True)

    # Validate model is not empty
    if not body.model or not body.model.strip():
        raise HTTPException(status_code=400, detail=f"No model selected for provider '{body.provider}'. Please select a model in the sidebar.")

    vectordb_path = get_user_vectordb_path(user.id)
    if not os.path.exists(vectordb_path):
        raise HTTPException(400, "No ingested data. Please ingest documents first.")

    start = time.time()

    cache_key = user.id
    pipeline = _user_pipelines.get(cache_key)

    # Rebuild the pipeline if model, provider, or top_k changed
    needs_rebuild = (
        pipeline is None
        or pipeline.get("model") != body.model
        or pipeline.get("provider") != body.provider
        or pipeline.get("top_k") != body.top_k
    )

    if needs_rebuild:
        try:
            embeddings = get_embedding_model()
            vectordb = load_vectordb(embeddings, vectordb_path)
            retriever = get_retriever(vectordb, top_k=body.top_k)
            llm = get_llm(body.model, provider=body.provider)
            rag_chain = create_rag_chain(llm, retriever)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM init failed: {exc}")

        pipeline = {
            "model": body.model,
            "provider": body.provider,
            "top_k": body.top_k,
            "retriever": retriever,
            "rag_chain": rag_chain,
            "vectordb": vectordb,
        }
        _user_pipelines[cache_key] = pipeline

    try:
        answer, sources = generate_response(
            pipeline["rag_chain"], body.question, pipeline["retriever"]
        )
    except Exception as exc:
        # Log the full traceback to the terminal for debugging
        print(f"\n[RAG ERROR] provider={body.provider} model={body.model}")
        traceback.print_exc()

        # Evict the broken pipeline so the next request forces a full rebuild
        _user_pipelines.pop(cache_key, None)

        err_str = str(exc).lower()

        # --- Quota / rate-limit errors (429) ---
        is_quota = "429" in str(exc) or "resource_exhausted" in err_str or "quota" in err_str or "rate" in err_str
        if is_quota:
            prov = body.provider
            if prov == "openrouter":
                detail = (
                    f"OpenRouter rate limit hit for model '{body.model}': {exc}. "
                    "Try a different free model, or wait a moment and retry."
                )
            else:
                detail = f"Rate limit exceeded: {exc}"
            raise HTTPException(status_code=429, detail=detail)

        # --- Authentication errors ---
        is_auth = (
            "401" in str(exc)
            or "authentication" in err_str
            or "user not found" in err_str
            or "api key" in err_str
            or "unauthorized" in err_str
        )
        if is_auth:
            prov = body.provider
            if prov == "openrouter":
                detail = (
                    f"OpenRouter authentication failed: {exc}. "
                    "Make sure OPENROUTER_API_KEY in your .env is valid. "
                    "Get a free key at https://openrouter.ai/keys"
                )
            else:
                detail = f"Authentication error: {exc}"
            # Use 400 (not 401) — 401 triggers frontend logout redirect
            raise HTTPException(status_code=400, detail=detail)

        # --- Not found (model removed/invalid) ---
        is_not_found = "404" in str(exc) or "not_found" in err_str or "not found" in err_str
        if is_not_found:
            detail = (
                f"Model '{body.model}' not found for provider '{body.provider}'. "
                "It may have been removed or renamed. Please select a different model."
            )
            raise HTTPException(status_code=400, detail=detail)

        raise HTTPException(status_code=500, detail=f"LLM query failed: {exc}")

    elapsed = time.time() - start

    source_data = []
    for src in sources:
        meta = src.metadata if hasattr(src, "metadata") else {}
        source_data.append({
            "filename": os.path.basename(meta.get("source", "Unknown")),
            "preview": (src.page_content[:250].replace("\n", " ") + "...") if hasattr(src, "page_content") else "",
        })

    return {
        "answer": answer,
        "sources": source_data,
        "time_seconds": round(elapsed, 2),
    }


@router.get("/stats")
def get_stats(user: User = Depends(get_current_user)):
    """Get vector DB stats for the current user."""
    vectordb_path = get_user_vectordb_path(user.id)
    if not os.path.exists(vectordb_path):
        return {"vectors": 0, "exists": False}

    try:
        embeddings = get_embedding_model()
        vectordb = load_vectordb(embeddings, vectordb_path)
        count = vectordb._collection.count()
        return {"vectors": count, "exists": True}
    except Exception:
        return {"vectors": 0, "exists": True}


@router.get("/providers")
def get_providers():
    """
    Return available LLM providers and their models.
    No auth required — used by the UI to populate dropdowns.
    """
    ollama_online = is_ollama_available()
    ollama_models = list_ollama_models() if ollama_online else []

    return {
        "providers": [
            {
                "id": "openrouter",
                "name": "OpenRouter (Cloud)",
                "available": True,
                "requires_key": True,
                "key_env": "OPENROUTER_API_KEY",
                "key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
                "models": {
                    "z-ai/glm-4.5-air:free": "GLM-4.5-Air (Best Balance)",
                    "stepfun/step-3.5-flash:free": "Step-3.5-Flash (Fast)",
                    "openai/gpt-oss-120b:free": "GPT-OSS-120B (Strong)",
                    "arcee-ai/trinity-mini:free": "Trinity-Mini (Efficient)",
                },
            },
            {
                "id": "ollama",
                "name": "Ollama (Local)",
                "available": ollama_online,
                "requires_key": False,
                "key_env": None,
                "key_set": True,
                "models": {m: m for m in ollama_models},
            },
        ]
    }