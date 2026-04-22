"""
========================================
Full RAG Pipeline
========================================
Connects ALL steps from the architecture diagram:

DATA PREPARATION (left side of diagram):
  A (Raw Data Sources) → B (Information Extraction) → C (Chunking) → D (Embedding) → Vector Database

RETRIEVAL AUGMENTED GENERATION (right side of diagram):
  1 (Query) → 2 (Embed Query) → 3 (Retrieve from Vector DB) → 4 (LLM) → 5 (Response)
"""
import os
from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import get_embedding_model, store_in_vectordb, load_vectordb
from src.retriever import get_retriever
from src.generator import get_llm, create_rag_chain, generate_response


class RAGPipeline:
    """
    Complete RAG Pipeline matching the architecture diagram exactly.

    Usage:
        rag = RAGPipeline(data_path="./data", llm_model="z-ai/glm-4.5-air:free")
        rag.ingest()          # A → B → C → D (Data Preparation)
        rag.setup_qa()        # Set up retriever + LLM
        answer = rag.query("What is...?")  # 1 → 2 → 3 → 4 → 5

    Providers:
        provider='openrouter'  → Cloud LLM via OpenRouter (default)
        provider='ollama'      → Local LLM via Ollama
    """

    def __init__(self, data_path="./data", vectordb_path="./storage/shared_vectors", llm_model="z-ai/glm-4.5-air:free", provider="openrouter"):
        self.data_path = data_path
        self.vectordb_path = vectordb_path
        self.llm_model = llm_model
        self.provider = provider

        # Components
        self.embeddings = None
        self.vectordb = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None

    # ========================================
    # DATA PREPARATION: A → B → C → D
    # ========================================
    def ingest(self, chunk_size=500, chunk_overlap=50):
        """
        Run the Data Preparation pipeline (left side of diagram).

        A → Raw Data Sources (load files from ./data/)
        B → Information Extraction (extract text from PDF/TXT)
        C → Chunking (split into smaller pieces)
        D → Embedding + Store in Vector Database
        """
        print("\n" + "=" * 60)
        print("  DATA PREPARATION: A -> B -> C -> D")
        print("=" * 60 + "\n")

        # Step A + B: Load raw data and extract text
        print("--- Step A + B: Raw Data Sources + Information Extraction ---")
        documents = load_documents(self.data_path)

        if not documents:
            print("\n[!] No documents found!")
            print(f"    Put PDF or TXT files in: {os.path.abspath(self.data_path)}")
            return False

        # Step C: Chunking
        print("\n--- Step C: Chunking ---")
        chunks = chunk_documents(documents, chunk_size, chunk_overlap)

        # Step D: Embedding + Vector Database
        print("\n--- Step D: Embedding -> Vector Database ---")
        self.embeddings = get_embedding_model()
        self.vectordb = store_in_vectordb(chunks, self.embeddings, self.vectordb_path)

        print("\n" + "=" * 60)
        print("  DATA PREPARATION COMPLETE!")
        print(f"  {len(documents)} documents -> {len(chunks)} chunks -> Vector DB")
        print("=" * 60 + "\n")
        return True

    # ========================================
    # RAG: 1 → 2 → 3 → 4 → 5
    # ========================================
    def setup_qa(self, top_k=3):
        """
        Set up the Retrieval Augmented Generation chain.
        Connects: Retriever (1,2,3) + LLM (4,5)
        """
        print("\n" + "=" * 60)
        print("  SETTING UP RAG: Retriever + LLM")
        print("=" * 60 + "\n")

        # Load embedding model if not already loaded
        if self.embeddings is None:
            self.embeddings = get_embedding_model()

        # Load vector database if not already loaded
        if self.vectordb is None:
            if not os.path.exists(self.vectordb_path):
                print("[!] No vector database found. Run ingest() first!")
                return False
            self.vectordb = load_vectordb(self.embeddings, self.vectordb_path)

        # Create retriever (handles steps 1, 2, 3)
        self.retriever = get_retriever(self.vectordb, top_k=top_k)

        # Create LLM (handles steps 4, 5)
        self.llm = get_llm(self.llm_model, provider=self.provider)

        # Create full RAG chain
        self.rag_chain = create_rag_chain(self.llm, self.retriever)

        print("\n[OK] RAG system ready for queries!\n")
        return True

    def query(self, question: str):
        """
        Run the full RAG query pipeline (right side of diagram).

        1 → Query comes in
        2 → Query is embedded
        3 → Relevant data retrieved from Vector Database
        4 → Relevant data + Query sent to LLM
        5 → Response generated

        Args:
            question: User's question

        Returns:
            Tuple of (answer, source_documents)
        """
        if self.rag_chain is None:
            success = self.setup_qa()
            if not success:
                return "System not ready. Run ingest() first.", []

        answer, sources = generate_response(self.rag_chain, question, self.retriever)
        return answer, sources