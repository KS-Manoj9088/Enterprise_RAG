"""
========================================
Step D: Embedding + Vector Database
========================================
Embeds chunks using HuggingFace (FREE, local) and stores in ChromaDB.
This maps to step D and the Vector Database in the RAG architecture diagram.
"""
import os

# ── Offline detection ──────────────────────────────────────────────
# huggingface_hub reads HF_HUB_OFFLINE into a module-level constant
# at *import* time, so the env var MUST be set BEFORE importing any
# HuggingFace library.  We use stdlib `socket` (zero extra deps).
def _detect_offline(timeout=2) -> bool:
    """Return True if HuggingFace Hub is unreachable."""
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout)
        return False
    except OSError:
        return True

if _detect_offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("[D] Offline mode detected - will use cached embedding model")
# ───────────────────────────────────────────────────────────────────

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# FREE local embedding model - no API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./storage/shared_vectors"


def get_embedding_model(model_name=EMBEDDING_MODEL):
    """
    Load the embedding model (runs locally, FREE).

    Architecture Mapping:
        D -> Embedding (convert text chunks to vectors)

    If no internet is available the model is loaded from the local
    HuggingFace cache (set once on first online run).

    Returns:
        HuggingFaceEmbeddings model
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"[D] Embedding model loaded: {model_name}")
    return embeddings


def store_in_vectordb(chunks, embeddings, persist_directory=PERSIST_DIR):
    """
    Embed chunks and store them in ChromaDB (Vector Database).

    Architecture Mapping:
        D → Embedding → Vector Database (store vectors)

    Args:
        chunks: List of chunked documents
        embeddings: Embedding model
        persist_directory: Where to save the vector database

    Returns:
        Chroma vector database instance
    """
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"[D] Stored {len(chunks)} chunks in Vector Database (ChromaDB)")
    print(f"[D] Persisted at: {persist_directory}")
    return vectordb


def load_vectordb(embeddings, persist_directory=PERSIST_DIR):
    """
    Load an existing Vector Database from disk.

    Returns:
        Chroma vector database instance
    """
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    count = vectordb._collection.count()
    print(f"[D] Loaded Vector Database with {count} vectors from '{persist_directory}'")
    return vectordb
