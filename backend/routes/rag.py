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
from src.generator import get_llm, create_rag_chain, generate_response, direct_generate, is_ollama_available, list_ollama_models
from prompts.analytical.summarizer import SUMMARY_PROMPT
from prompts.analytical.comparison import COMPARISON_PROMPT
from prompts.analytical.step_by_step import STEP_BY_STEP_PROMPT
from prompts.conversational.friendly import CONVERSATIONAL_PROMPT
from prompts.conversational.eli5 import ELI5_PROMPT
from prompts.domain.technical import TECHNICAL_PROMPT
from prompts.domain.academic import ACADEMIC_PROMPT
from prompts.domain.legal import LEGAL_PROMPT
from prompts.strict.strict_qa import STRICT_PROMPT
from prompts.strict.strict_cited import STRICT_CITED_PROMPT

# ── Prompt style registry ──
PROMPT_STYLES: dict[str, str] = {
    "auto":         STRICT_PROMPT,          # overridden at query time
    "strict":       STRICT_PROMPT,
    "strict_cited": STRICT_CITED_PROMPT,
    "general":      CONVERSATIONAL_PROMPT,
    "summary":      SUMMARY_PROMPT,
    "conversational": CONVERSATIONAL_PROMPT,
    "eli5":         ELI5_PROMPT,
    "technical":    TECHNICAL_PROMPT,
    "academic":     ACADEMIC_PROMPT,
    "legal":        LEGAL_PROMPT,
    "comparison":   COMPARISON_PROMPT,
    "step_by_step": STEP_BY_STEP_PROMPT,
}

router = APIRouter(prefix="/api/rag", tags=["rag"])

# ── In-memory cache for per-user pipelines ──
_user_pipelines: dict = {}

# ── Summary query detection ──
_SUMMARY_KEYWORDS = (
    "summarize", "summarise", "summary", "give a summary",
    "summarization", "brief summary",
    "give me a summary", "give an overview",
    "give me an overview", "give a brief",
    "what are the main points", "what are the key points",
    "what is the document about", "what is this document about",
    "overview of", "summarize the", "summarise the",
)
_SUMMARY_TOP_K = 15  # retrieve more chunks for summarization

# Broad retrieval query used when user asks for a summary.
# This spans many topics and retrieves diverse chunks across the whole document
# instead of trying to match the literal "give a summary about file.pdf" query.
_SUMMARY_RETRIEVAL_QUERY = (
    "introduction overview key concepts main topics important ideas "
    "definitions background theory methods results conclusion"
)


def _is_summary_query(question: str) -> bool:
    """Return True if the question looks like a summarization request."""
    q = question.lower()
    return any(kw in q for kw in _SUMMARY_KEYWORDS)


def _extract_filename(question: str) -> str | None:
    """
    Try to extract a filename mentioned in the question.
    E.g. "give a summary about ai_complete_reference.pdf" → "ai_complete_reference.pdf"
    """
    import re
    match = re.search(r'[\w\-_]+\.(?:pdf|txt|docx)', question, re.IGNORECASE)
    return match.group(0).lower() if match else None


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
    top_k: int = 5
    model: str = "z-ai/glm-4.5-air:free"
    provider: str = "openrouter"  # 'openrouter' or 'ollama'
    prompt_style: str = "auto"    # 'auto' | 'strict' | 'strict_cited' | 'summary' | 'conversational' | 'eli5' | 'technical' | 'academic' | 'legal' | 'comparison' | 'step_by_step'
    document_id: str | None = None  # Optional: scope query to a specific document


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

    # Step D: Wipe the existing vector DB then re-embed from scratch.
    # This ensures stale chunks from previously ingested (or deleted) documents
    # are never returned by the retriever.
    #
    # On Windows, ChromaDB keeps binary index files open as long as any
    # Python object holds a reference to the client. We must release all
    # references BEFORE calling shutil.rmtree, otherwise we get WinError 32.
    import shutil
    import gc

    # 1. Release the cached pipeline (holds the live ChromaDB client)
    evicted = _user_pipelines.pop(user.id, None)
    if evicted:
        try:
            # Try to delete the collection via API first (releases file locks)
            col_name = evicted["vectordb"]._collection.name
            evicted["vectordb"]._client.delete_collection(col_name)
        except Exception:
            pass
        try:
            evicted["vectordb"]._client.close()
        except Exception:
            pass
        del evicted

    # 2. Force garbage collection so the file handles are actually closed
    gc.collect()

    # 3. Now safe to delete the directory on Windows
    if os.path.exists(vectordb_path):
        try:
            shutil.rmtree(vectordb_path)
            print(f"[Ingest] Cleared old vector DB at {vectordb_path}")
        except PermissionError as e:
            # Last resort: delete only the SQLite DB and segment files,
            # leaving the directory structure; ChromaDB will overwrite them.
            print(f"[Ingest] Warning: could not fully clear vector DB ({e}). Attempting partial clear.")
            for root, dirs, files in os.walk(vectordb_path):
                for fname in files:
                    try:
                        os.remove(os.path.join(root, fname))
                    except Exception:
                        pass
    os.makedirs(vectordb_path, exist_ok=True)

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
    db: Session = Depends(get_db),
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

    # ── Resolve effective prompt style ──
    style = body.prompt_style if body.prompt_style in PROMPT_STYLES else "strict"
    if style == "auto":
        style = "summary" if _is_summary_query(body.question) else "conversational"
    elif style == "general" and _is_summary_query(body.question):
        # General mode silently promotes to summary when user asks for one
        style = "summary"
    elif style == "strict" and _is_summary_query(body.question):
        # Strict mode blocks full-document summary requests
        return {
            "answer": (
                "⚠️ **Strict Q&A mode does not produce full-document summaries.**\n\n"
                "Switch **Response Style → Summary** in the sidebar to get a summary."
            ),
            "sources": [],
            "time_seconds": 0.0,
        }
    chosen_prompt = PROMPT_STYLES[style]

    # Summary-mode needs more chunks to cover the whole document
    is_summary_mode = style == "summary"
    effective_top_k = _SUMMARY_TOP_K if is_summary_mode else body.top_k

    # ── Resolve document filter ──
    doc_filter = None
    if body.document_id:
        target_doc = db.query(Document).filter(
            Document.id == body.document_id,
            Document.user_id == user.id,
        ).first()
        if target_doc:
            # The 'source' metadata stored in ChromaDB contains the file path
            # which includes the safe filename. Use $contains to match it.
            doc_filter = {"source": {"$contains": target_doc.filename}}
            print(f"[FILTER] Scoping retrieval to document: {target_doc.original_name} ({target_doc.filename})")
        else:
            print(f"[FILTER] document_id={body.document_id} not found, querying all documents")

    cache_key = user.id
    pipeline = _user_pipelines.get(cache_key)

    # Rebuild the cached pipeline if model, provider, or top_k changed
    needs_rebuild = (
        pipeline is None
        or pipeline.get("model") != body.model
        or pipeline.get("provider") != body.provider
        or pipeline.get("top_k") != effective_top_k
    )

    if needs_rebuild:
        try:
            embeddings = get_embedding_model()
            vectordb = load_vectordb(embeddings, vectordb_path)
            llm = get_llm(body.model, provider=body.provider)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM init failed: {exc}")

        pipeline = {
            "model": body.model,
            "provider": body.provider,
            "top_k": effective_top_k,
            "vectordb": vectordb,
        }
        _user_pipelines[cache_key] = pipeline

    # ── Always build a fresh retriever with the current filter ──
    if doc_filter:
        # Single-document mode: plain similarity is fine, filter scopes to that doc
        search_kwargs = {"k": effective_top_k, "filter": doc_filter}
        search_type = "similarity"
        print(f"[RETRIEVER] Single-doc filtered retriever: {doc_filter}")
    else:
        # All-documents mode: use MMR so results are diverse across documents
        # fetch_k >> k ensures candidates from multiple docs before re-ranking
        search_kwargs = {
            "k": effective_top_k,
            "fetch_k": max(effective_top_k * 6, 20),
            "lambda_mult": 0.5,  # 0=max diversity, 1=max relevance
        }
        search_type = "mmr"
        print(f"[RETRIEVER] All-docs MMR retriever (k={effective_top_k}, fetch_k={search_kwargs['fetch_k']})")
    scoped_retriever = pipeline["vectordb"].as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    # ── For summary mode: use direct_generate with a broad retrieval query
    #    so the vector search finds real content instead of matching the
    #    "give a summary about file.pdf" sentence. ──
    if is_summary_mode:
        try:
            llm = get_llm(body.model, provider=body.provider)
            # Use MMR for diverse chunk coverage of the document
            mmr_search_kwargs = {"k": effective_top_k, "fetch_k": effective_top_k * 4}
            if doc_filter:
                mmr_search_kwargs["filter"] = doc_filter
            mmr_retriever = pipeline["vectordb"].as_retriever(
                search_type="mmr",
                search_kwargs=mmr_search_kwargs,
            )
            answer, sources = direct_generate(
                llm=llm,
                prompt_template=chosen_prompt,
                retriever=mmr_retriever,
                retrieval_query=_SUMMARY_RETRIEVAL_QUERY,
                user_question=body.question,
            )
        except Exception as exc:
            import traceback
            print(f"\n[SUMMARY ERROR] provider={body.provider} model={body.model}")
            traceback.print_exc()
            _user_pipelines.pop(cache_key, None)
            raise HTTPException(status_code=500, detail=f"Summary generation failed: {exc}")

    else:
        # Normal (non-summary) mode: rebuild chain with chosen prompt and invoke
        try:
            llm = get_llm(body.model, provider=body.provider)
            active_chain = create_rag_chain(llm, scoped_retriever, prompt_template=chosen_prompt)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM init failed: {exc}")

        try:
            answer, sources = generate_response(
                active_chain, body.question, scoped_retriever
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
