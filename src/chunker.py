"""
========================================
Step C: Chunking
========================================
Splits extracted documents into smaller chunks for embedding.
This maps to step C in the RAG architecture diagram.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into smaller chunks.

    Architecture Mapping:
        C → Chunking (split large text into manageable pieces)

    Args:   
        documents: List of Document objects from loader
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"[C] Chunking: {len(documents)} documents -> {len(chunks)} chunks")
    print(f"[C] Settings: chunk_size={chunk_size}, overlap={chunk_overlap}")

    return chunks

"""
========================================
Step C: Chunking
========================================
Splits extracted documents into smaller chunks for embedding.
This maps to step C in the RAG architecture diagram.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into smaller chunks.

    Architecture Mapping:
        C → Chunking (split large text into manageable pieces)

    Args:   
        documents: List of Document objects from loader
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"[C] Chunking: {len(documents)} documents -> {len(chunks)} chunks")
    print(f"[C] Settings: chunk_size={chunk_size}, overlap={chunk_overlap}")

    return chunks
